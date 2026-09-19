// Orbit authoring viewport -- the author rig.
//
// Ported from ../Img2Splat/web/orbit.js. The GEOMETRY is carried over verbatim,
// because it mirrors steps.py and every mirror that drifted has already cost a
// day. What changed here: the 360-degree dome backdrop and the depth-pass
// preview are gone (this build has no endpoints for either), and the palette is
// the neutral+maroon one the rest of the tool uses.
//
// Preview only. The control video that actually goes to fal is rendered
// SERVER-SIDE from the identical camera model that gets written to COLMAP, so
// what the engine conditions on and what Brush trains against can never drift
// apart. This viewport exists so the orbit can be judged by eye first.
//
// World is Z-up (matching steps.py); three is Y-up, so a root node maps
// (x, y, z) -> (x, z, -y).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

// The palette, in the tool's own tokens. No hue except maroon.
const C_BG      = 0x0a0a0a;   // --bg-viewport-ish; the free pane
const C_BG_SHOT = 0x000000;   // the camera pane: pure black, it is the frame
const C_GRID    = 0x2b2b2b;
const C_GRID2   = 0x1e1e1e;
const C_PATH    = 0x808080;
const C_FRUST   = 0x3d3d3d;
const C_FRUST0  = 0xc6c6c6;   // frame 0, by weight not by hue
const C_AIM     = 0x761b29;   // maroon: the one accent in the scene
const C_CROP    = 0xc4737e;
// The camera pane stands in for the control video, so it wears the render's
// own backdrop (steps.BG_GREY) and, with clay on, its flat grey (CLAY_GREY).
const SHOT_BACKDROP = 60 / 255;
const CLAY_GREY = 150 / 255;

let scene, cam, renderer, controls, root, card, pathLine, camGroup, raf = null;
let params = null, texture = null;
let cloudPts = null, cloudMat = null, gizmo = null;
let subjectGroup = null;
let cropMesh = null;      // the isolate sphere, world-placed
let canvasEl = null, showRig = true, cardOn = true;
let shotCam = null, shotFrame = 0, splitOn = true, overlayEl = null;
let gridHelper = null;
let clayOn = false, shotBg = null;

function toThree(x, y, z) { return new THREE.Vector3(x, z, -y); }

export function init(canvas) {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  scene = new THREE.Scene();
  scene.background = new THREE.Color(C_BG);

  cam = new THREE.PerspectiveCamera(42, 1, 0.05, 500);
  // far enough back to frame a radius-9 ring without clipping it at the edges
  cam.position.set(17, 13, 21);

  // Controls listen on the canvas itself; both OrbitControls and
  // TransformControls turn pointer positions into NDC using their element's
  // rect. The left pane is half the canvas, so drags in the free pane are
  // correct and the right pane is not interactive by design.
  overlayEl = canvas;
  controls = new OrbitControls(cam, overlayEl);
  controls.enableDamping = true;
  controls.target.set(0, 2.3, 0);

  // the shot camera: the orbit pose the control video is rendered from
  shotCam = new THREE.PerspectiveCamera(22.62, 0.75, 0.05, 800);

  root = new THREE.Group();
  scene.add(root);
  // the lifted cloud lives under its own group so translate/rotate gizmos
  // move ONE object, and the same numbers go to the server verbatim
  subjectGroup = new THREE.Group();
  subjectGroup.rotation.order = "YZX";   // see setSubject: matches the renderer
  root.add(subjectGroup);

  gridHelper = new THREE.GridHelper(24, 24, C_GRID, C_GRID2);
  scene.add(gridHelper);
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));

  camGroup = new THREE.Group();
  root.add(camGroup);

  const resize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    cam.aspect = (splitOn ? w / 2 : w) / h;
    cam.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  canvasEl = canvas;
  const tick = () => {
    raf = requestAnimationFrame(tick);
    controls.update();
    drawPanes();
    syncCursor();
  };
  if (raf === null) tick();
}

// --------------------------------------------------------------- two panes --
// One canvas, two scissored viewports. The right one uses the orbit's own
// camera model, so it is a live stand-in for the control frame rather than a
// second free view.
function placeShotCam() {
  if (!shotCam || !params) return;
  const { frames, sweep_deg, radius, aim_z, vfov_deg, width, height,
          ccw } = params;
  const n = Math.max(1, frames | 0);
  const i = ((shotFrame % n) + n) % n;
  const dir = ccw === false ? -1 : 1;
  // frame 0 sits at -90 deg, matching steps.Orbit.angles()
  // phase_deg rotates the whole rig off the photo viewpoint; the rays stay
  // anchored there, so this must match steps.Orbit.angles() exactly.
  const a = -Math.PI / 2
          + THREE.MathUtils.degToRad(+params.phase_deg || 0)
          + dir * (i / n) * THREE.MathUtils.degToRad(sweep_deg);
  // Height must come from THIS frame, not the nominal cam_z -- otherwise the
  // shot pane shows a flat ring however the path is set, which is the one
  // thing it exists to reveal.
  const cz = camZAt(i);
  shotCam.fov = vfov_deg;
  shotCam.aspect = width / height;
  shotCam.position.copy(
    toThree(radius * Math.cos(a), radius * Math.sin(a), cz));
  shotCam.up.set(0, 1, 0);
  shotCam.lookAt(toThree(0, 0, aim_z));
  shotCam.updateProjectionMatrix();
}

export function setShotFrame(i) { shotFrame = i | 0; placeShotCam(); }
export function getShotFrame() { return shotFrame; }
export function setSplit(on) {
  splitOn = !!on;
  if (canvasEl) {
    const w = canvasEl.clientWidth, h = canvasEl.clientHeight;
    cam.aspect = (splitOn ? w / 2 : w) / h;
    cam.updateProjectionMatrix();
  }
}

function drawPanes() {
  if (!canvasEl) return;
  const w = canvasEl.clientWidth, h = canvasEl.clientHeight;
  if (!w || !h) return;
  renderer.setScissorTest(true);

  const paneW = splitOn ? Math.floor(w / 2) : w;
  renderer.setViewport(0, 0, paneW, h);
  renderer.setScissor(0, 0, paneW, h);
  if (cloudMat) cloudMat.uniforms.uClay.value = clayOn ? 1 : 0;
  renderer.render(scene, cam);
  if (cloudMat) cloudMat.uniforms.uClay.value = 0;

  if (!splitOn) { renderer.setScissorTest(false); return; }

  // right pane: rig helpers hidden -- they are authoring aids, and the point
  // of this pane is to look like the frame that gets rendered
  placeShotCam();
  const pv = pathLine ? pathLine.visible : null;
  const cv = camGroup ? camGroup.visible : null;
  const gv = gridHelper ? gridHelper.visible : null;
  const zv = gizmo ? gizmo.visible : null;
  const kv = cropMesh ? cropMesh.visible : null;
  if (pathLine) pathLine.visible = false;
  if (camGroup) camGroup.visible = false;
  if (gridHelper) gridHelper.visible = false;
  if (gizmo) gizmo.visible = false;
  if (cropMesh) cropMesh.visible = false;

  // letterbox to the render aspect so framing here matches the output
  const ar = shotCam.aspect;
  let vw = w - paneW, vh = h, ox = paneW, oy = 0;
  if (vw / vh > ar) { const nw = Math.floor(vh * ar); ox += (vw - nw) / 2; vw = nw; }
  else { const nh = Math.floor(vw / ar); oy += (vh - nh) / 2; vh = nh; }
  renderer.setViewport(paneW, 0, w - paneW, h);
  renderer.setScissor(paneW, 0, w - paneW, h);
  renderer.setClearColor(C_BG_SHOT, 1);
  renderer.clear(true, true, false);
  // The camera pane is letterboxed to the render aspect, so vh maps exactly
  // onto the rendered canvas height: this is the true pixels-per-render-pixel.
  if (cloudMat && params && params.height) {
    cloudMat.uniforms.uPx.value =
      (vh * renderer.getPixelRatio()) / params.height;
  }
  renderer.setViewport(ox, oy, vw, vh);
  renderer.setScissor(ox, oy, vw, vh);
  if (!shotBg) shotBg = new THREE.Color().setRGB(SHOT_BACKDROP, SHOT_BACKDROP, SHOT_BACKDROP, THREE.SRGBColorSpace);
  const bgWas = scene.background;
  scene.background = shotBg;
  if (cloudMat) cloudMat.uniforms.uClay.value = clayOn ? 1 : 0;
  renderer.render(scene, shotCam);
  if (cloudMat) cloudMat.uniforms.uClay.value = 0;
  scene.background = bgWas;
  renderer.setClearColor(C_BG, 1);

  if (pathLine) pathLine.visible = pv;
  if (camGroup) camGroup.visible = cv;
  if (gridHelper) gridHelper.visible = gv;
  if (gizmo) gizmo.visible = zv;   // NOT true: a detached gizmo stays hidden
  if (cropMesh) cropMesh.visible = kv;
  renderer.setScissorTest(false);
}

/** Force one render. requestAnimationFrame is throttled to zero in a hidden
 *  tab, so this is the only way to exercise the renderer there. */
export function renderOnce(w, h) {
  if (!renderer) return null;
  if (w && h) renderer.setSize(w, h, false);
  const gl = renderer.getContext();
  while (gl.getError() !== gl.NO_ERROR) { /* drain */ }
  renderer.render(scene, cam);
  return {
    glError: gl.getError(),
    programs: renderer.info.programs ? renderer.info.programs.length : -1,
    drawCalls: renderer.info.render.calls,
    size: renderer.getSize(new THREE.Vector2()).toArray(),
  };
}

/** Camera height at frame i -- the ONE mirror of steps.Orbit.cam_zs().
 *
 * This used to be written out three times (shot camera, span readout, rig
 * path) and the copies had already drifted: two of them clamped elev_cycles
 * to a minimum of 1, which silently forbids the 0.5 that makes a helix climb
 * once without turning. One function now, so a fourth caller cannot disagree.
 */
export function camZAt(i, p = params) {
  if (!p) return 0;
  const n = Math.max(1, p.frames | 0);
  if (p.path === "helix") {
    const lo = Math.min(+p.helix_min_z || 0, +p.helix_max_z || 0);
    const hi = Math.max(+p.helix_min_z || 0, +p.helix_max_z || 0);
    const revs = Math.abs(+p.sweep_deg || 0) / 360;
    const span = Math.max(0, +p.elev_cycles || 0) * revs;
    let ph = ((+p.helix_start || 0) + span * (i / n)) % 1;
    if (ph < 0) ph += 1;
    return lo + (hi - lo) * (1 - 2 * Math.abs(ph - 0.5));
  }
  const a = +p.elev_sweep_deg || 0;
  if (!a) return p.cam_z;
  const cyc = Math.max(0, +p.elev_cycles || 0);
  const e = (+p.elev_deg || 0) + a * Math.sin(2 * Math.PI * cyc * i / n);
  return p.aim_z + p.radius * Math.tan(THREE.MathUtils.degToRad(e));
}

/** Lowest and highest camera height the current path produces. */
export function camZSpan() {
  if (!params) return null;
  const n = Math.max(2, params.frames | 0);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const z = camZAt(i);
    lo = Math.min(lo, z); hi = Math.max(hi, z);
  }
  return { min: lo, max: hi };
}

export function memInfo() {
  return renderer
    ? { ...renderer.info.memory, programs: renderer.info.programs?.length ?? 0 }
    : null;
}

/** Hide the card / orbit path / frusta. */
export function setRigVisible(v) {
  showRig = v;
  if (card) card.visible = v && cardOn;
  if (pathLine) pathLine.visible = v;
  if (camGroup) camGroup.visible = v;
}

export function focusSubject() {
  if (!params) return;
  const t = new THREE.Vector3(0, params.aim_z, 0);
  controls.target.copy(t);
  cam.position.set(t.x + 4.5, t.y + 1.6, t.z + 5.5);
  controls.update();
}

/** The free camera's aim point, in WORLD units -- what "centre on the
 *  subject" writes into the crop sphere. */
export function freeTarget() {
  if (!controls) return null;
  const t = controls.target;
  return { x: t.x, y: -t.z, z: t.y };
}

let texGen = 0;   // bumped on every set/clear; a stale async load sees a
                  // mismatch and drops its texture instead of resurrecting it
export function setImage(url) {
  const g = ++texGen;
  new THREE.TextureLoader().load(url, t => {
    if (g !== texGen) { t.dispose(); return; }   // superseded while loading
    t.colorSpace = THREE.SRGBColorSpace;
    if (texture) texture.dispose();
    texture = t;
    rebuild();
  });
}

export function clearImage() {
  texGen++;
  if (texture) texture.dispose();
  texture = null;
  rebuild();
}

/** The flat card. It is a persistent state, not a per-rebuild default:
 *  rebuild() runs on every keystroke, so setting visibility only on the mesh
 *  meant the card came back the moment any field moved. */
export function setCardVisible(v) {
  cardOn = !!v;
  if (card) card.visible = cardOn && showRig;
}
export function cardVisible() { return cardOn; }

// ------------------------------------------------------------- point cloud --
// Layout from /api/cloud: u32 N | f32 N*3 card | f32 N*3 ray | f32 N dt |
// u8 N*3 BGR. Strength/cutoff live in uniforms so the sliders are free.
export function setCloud(buffer) {
  clearCloud();
  const dv = new DataView(buffer);
  const N = dv.getUint32(0, true);
  let o = 4;
  const cardW = new Float32Array(buffer, o, N * 3); o += N * 12;
  const rayW = new Float32Array(buffer, o, N * 3); o += N * 12;
  const dt = new Float32Array(buffer, o, N); o += N * 4;
  const bgr = new Uint8Array(buffer, o, N * 3); o += N * 3;
  // A trailing 20-byte frame (camera 0 and the old bend range) may follow; the
  // bend was removed, so it is ignored.

  // world (x, y, z) -> three (x, z, -y), colours BGR -> RGB
  const cardA = new Float32Array(N * 3), ray = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    cardA[i * 3] = cardW[i * 3];
    cardA[i * 3 + 1] = cardW[i * 3 + 2];
    cardA[i * 3 + 2] = -cardW[i * 3 + 1];
    ray[i * 3] = rayW[i * 3];
    ray[i * 3 + 1] = rayW[i * 3 + 2];
    ray[i * 3 + 2] = -rayW[i * 3 + 1];
    col[i * 3] = bgr[i * 3 + 2] / 255;
    col[i * 3 + 1] = bgr[i * 3 + 1] / 255;
    col[i * 3 + 2] = bgr[i * 3] / 255;
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(cardA, 3));
  g.setAttribute("aRay", new THREE.BufferAttribute(ray, 3));
  g.setAttribute("aDt", new THREE.BufferAttribute(dt.slice(), 1));
  g.setAttribute("aColor", new THREE.BufferAttribute(col, 3));

  cloudMat = new THREE.ShaderMaterial({
    uniforms: {
      uStrength: { value: 1.0 },
      uCutoff: { value: 0.0 },
      // Screen pixels per RENDER pixel, set per frame from the camera pane.
      // The renderer splats 2x2 output pixels per point at any depth, so
      // matching it means a constant screen size, not a perspective one.
      uPx: { value: 1.0 },
      // 1 while the camera pane draws with flat clay on: every point one grey,
      // exactly what the control video will hold.
      uClay: { value: 0.0 },
    },
    vertexShader: `
      attribute vec3 aRay; attribute float aDt; attribute vec3 aColor;
      uniform float uStrength, uCutoff, uPx;
      varying vec3 vC; varying float vKill;
      void main() {
        vec3 p = position + aRay * (aDt * uStrength);
        vKill = (uCutoff > 0.0 && aDt > uCutoff) ? 1.0 : 0.0;
        vC = aColor;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        // NOT distance-scaled: the render's splat is 2x2 output pixels
        // whatever the depth, so this is what the video will actually hold.
        gl_PointSize = max(1.0, 2.0 * uPx);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      precision mediump float;
      uniform float uClay;
      varying vec3 vC; varying float vKill;
      void main() {
        if (vKill > 0.5) discard;
        gl_FragColor = vec4(uClay > 0.5 ? vec3(${CLAY_GREY.toFixed(6)}) : vC, 1.0);
      }`,
  });
  cloudPts = new THREE.Points(g, cloudMat);
  cloudPts.frustumCulled = false;
  subjectGroup.add(cloudPts);
  return N;
}

export function clearCloud() {
  if (cloudPts) {
    subjectGroup.remove(cloudPts);
    cloudPts.geometry.dispose();
    cloudPts.material.dispose();
    cloudPts = null;
    cloudMat = null;
  }
}

export function setCloudParams(p) {
  if (!cloudMat) return;
  if (p.strength !== undefined) cloudMat.uniforms.uStrength.value = p.strength;
  if (p.cutoff !== undefined) cloudMat.uniforms.uCutoff.value = p.cutoff;
  if (p.px !== undefined) cloudMat.uniforms.uPx.value = p.px;
}

/** Flat clay in BOTH panes: the free view is where the cloud is composed, so
 *  it should show what the control video will render, not a different look.
 *  The viewport bar's "Camera: colour" toggle is the way to see the photo
 *  colours while clay is on. */
export function setClay(on) { clayOn = !!on; }
export function hasCloud() { return !!cloudPts; }

// world dz -> three +y; world rotation about +Z -> three rotation about +Y
export function setSubject(s) {
  if (!subjectGroup) return;
  subjectGroup.position.set(+s.x || 0, +s.z || 0, -(+s.y || 0));
  // The renderer rotates about world X, then world Y, then world Z. World
  // (x, y, z) is three (x, z, -y), so those become three's X, then Z, then Y
  // -- which is three's "YZX" Euler order. With "XYZ" the preview and the
  // render agreed on one angle and diverged on two.
  subjectGroup.rotation.order = "YZX";
  subjectGroup.rotation.set(THREE.MathUtils.degToRad(+s.rotX || 0),
                            THREE.MathUtils.degToRad(+s.rot || 0),
                            THREE.MathUtils.degToRad(-(+s.rotY || 0)));
}

export function cloudDebug() {
  return {
    points: cloudPts ? cloudPts.geometry.getAttribute("position").count : 0,
    strength: cloudMat ? cloudMat.uniforms.uStrength.value : null,
    cutoff: cloudMat ? cloudMat.uniforms.uCutoff.value : null,
    gizmo: gizmoMode(),
  };
}

// -------------------------------------------------------------- transforms --
// One gizmo, attached to whichever object is being composed. Dragging it
// writes back through onGizmoChange so the number fields (the values the
// SERVER receives) stay the source of truth.
let onGizmoCb = null;
export function onGizmoChange(cb) { onGizmoCb = cb; }

/** Show/place the isolate sphere. c = {on, x, y, z, r} in WORLD units.
 *
 * Drawn as a translucent wireframe so the points inside stay readable -- the
 * whole job is judging what the sphere contains, which a solid surface hides.
 */
export function setCrop(c) {
  if (!scene) return;
  if (!c || !c.on) {
    if (cropMesh) cropMesh.visible = false;
    if (gizmo && gizmo.object === cropMesh) { gizmo.detach(); gizmo.visible = false; }
    return;
  }
  if (!cropMesh) {
    cropMesh = new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 24),
      new THREE.MeshBasicMaterial({ color: C_CROP, wireframe: true,
                                    transparent: true, opacity: 0.35,
                                    depthWrite: false }));
    cropMesh.renderOrder = 3;
    scene.add(cropMesh);
  }
  cropMesh.visible = true;
  // world (x, y, z) -> three (x, z, -y), the same mapping as everything else
  cropMesh.position.set(+c.x || 0, +c.z || 0, -(+c.y || 0));
  cropMesh.scale.setScalar(Math.max(0.01, +c.r || 1));
}

/** Where the gizmo's centre is DRAWN, in client pixels, and what it is
 *  hovering -- for checking that picking lines up with the drawing. */
export function gizmoDebug() {
  if (!gizmo || !gizmo.object || !canvasEl) return null;
  const p = new THREE.Vector3();
  gizmo.object.getWorldPosition(p);
  p.project(cam);
  const rect = canvasEl.getBoundingClientRect();
  const paneW = splitOn ? Math.floor(rect.width / 2) : rect.width;
  return { axis: gizmo.axis, dragging: !!gizmo.dragging,
           x: rect.left + (p.x + 1) / 2 * paneW,
           y: rect.top + (1 - p.y) / 2 * rect.height };
}

/** The cursor says what a press will do: a hand over a handle you can grab,
 *  a closed hand while dragging, the default everywhere else. TransformControls
 *  already lights the hovered axis; the cursor makes it unmistakable. */
let cursorNow = "";
function syncCursor() {
  if (!overlayEl) return;
  const want = gizmo && gizmo.object && gizmo.visible !== false
    ? (gizmo.dragging ? "grabbing" : (gizmo.axis ? "grab" : ""))
    : "";
  if (want !== cursorNow) {
    cursorNow = want;
    overlayEl.style.cursor = want;
  }
}

export function setGizmo(mode) {
  // TransformControls needs a live camera and DOM element; before init() there
  // is nothing to attach to, and constructing it there throws.
  if (!cam || !overlayEl) return;
  if (!gizmo) {
    gizmo = new TransformControls(cam, overlayEl);
    // Pick in the FREE PANE's coordinates, not the whole canvas'. The free
    // view is drawn into the left half when split is on, but TransformControls
    // turns the pointer into NDC across the full canvas -- so the handle you
    // could grab sat at twice the distance from the pane centre that it was
    // drawn at, and the offset grew toward the edges. Its event handlers call
    // this._getPointer at run time, so replacing it on the instance is enough.
    gizmo._getPointer = event => {
      const rect = overlayEl.getBoundingClientRect();
      const paneW = splitOn ? Math.floor(rect.width / 2) : rect.width;
      return {
        x: (event.clientX - rect.left) / Math.max(paneW, 1) * 2 - 1,
        y: -(event.clientY - rect.top) / Math.max(rect.height, 1) * 2 + 1,
        button: event.button,
      };
    };
    gizmo.addEventListener("dragging-changed",
                           e => { controls.enabled = !e.value; });
    gizmo.addEventListener("objectChange", () => {
      if (!onGizmoCb || !gizmo.object) return;
      if (gizmo.object === cropMesh) {
        onGizmoCb({ kind: "crop",
                    x: cropMesh.position.x, y: -cropMesh.position.z,
                    z: cropMesh.position.y, r: cropMesh.scale.x });
      } else {
        // three (x, y, z) -> world (x, -z, y), and the same swap for the
        // rotations so a drag on a handle moves the axis it is labelled with.
        const e = subjectGroup.rotation;
        onGizmoCb({ kind: "subject",
                    x: subjectGroup.position.x, y: -subjectGroup.position.z,
                    z: subjectGroup.position.y,
                    rotX: THREE.MathUtils.radToDeg(e.x),
                    rotY: THREE.MathUtils.radToDeg(-e.z),
                    rot: THREE.MathUtils.radToDeg(e.y) });
      }
    });
    // three r150+ ships TransformControls as a Controls object whose visual
    // helper is what belongs in the scene; older builds ARE an Object3D.
    scene.add(gizmo.getHelper ? gizmo.getHelper() : gizmo);
  }
  // Every axis is shown in both modes: the rotate gizmo used to hide X and Z,
  // so a subject could be spun but never tilted or rolled.
  gizmo.detach();
  gizmo.showX = gizmo.showY = gizmo.showZ = true;
  if (mode === "move") {
    gizmo.setMode("translate");
    gizmo.attach(subjectGroup);
  } else if (mode === "rotate") {
    gizmo.setMode("rotate");
    gizmo.attach(subjectGroup);
  } else if (mode === "crop" && cropMesh && cropMesh.visible) {
    // Translate only. The sphere's radius is a number field rather than a
    // scale handle: a non-uniform drag would make it an ellipsoid, which the
    // server's distance test cannot represent.
    gizmo.setMode("translate");
    gizmo.attach(cropMesh);
  }
  gizmo.visible = !!gizmo.object;
  const helper = gizmo.getHelper ? gizmo.getHelper() : gizmo;
  helper.visible = !!gizmo.object;
}

/** Which gizmo mode is live, for a viewport button to cycle. */
export function gizmoMode() {
  if (!gizmo || !gizmo.object) return "off";
  if (gizmo.object === cropMesh) return "crop";
  return gizmo.mode === "rotate" ? "rotate" : "move";
}

/** Drop everything that belongs to one project.
 *
 * The viewport holds several pieces of per-project state and none of it used
 * to be scoped: switching projects left the previous subject's point cloud,
 * card texture and subject transform in the scene.
 */
export function resetForProject() {
  texGen++;
  clearCloud();
  if (cropMesh) {
    if (gizmo && gizmo.object === cropMesh) gizmo.detach();
    scene.remove(cropMesh);
    cropMesh.geometry.dispose(); cropMesh.material.dispose();
    cropMesh = null;
  }
  if (texture) { texture.dispose(); texture = null; }
  if (gizmo) gizmo.detach();
  if (subjectGroup) {
    subjectGroup.position.set(0, 0, 0);
    subjectGroup.rotation.set(0, 0, 0);
  }
  shotFrame = 0;
  rebuild();
}

export function setParams(p) { params = p; rebuild(); placeShotCam(); }

function rebuild() {
  if (!scene || !params) return;

  if (card) { root.remove(card); card.geometry.dispose(); card.material.dispose(); card = null; }
  // Detaching is not freeing. rebuild() runs on every keystroke across eleven
  // numeric fields, allocating ~24 frusta plus a sphere each time, so anything
  // not disposed here accumulates on the GPU until the context dies.
  if (pathLine) {
    root.remove(pathLine);
    pathLine.geometry.dispose();
    pathLine.material.dispose();
    pathLine = null;
  }
  while (camGroup.children.length) {
    const c = camGroup.children[0];
    camGroup.remove(c);
    if (c.geometry) c.geometry.dispose();
    if (c.material) c.material.dispose();
  }

  const { frames, sweep_deg, radius, aim_z, vfov_deg, width, height,
          ccw, card_height } = params;

  // the card: image plane standing at the origin, facing -Y (toward frame 0)
  if (texture) {
    const ar = (texture.image?.width || 1) / (texture.image?.height || 1);
    const w = card_height * ar;
    const geo = new THREE.PlaneGeometry(w, card_height);
    const mat = new THREE.MeshBasicMaterial({
      map: texture, transparent: true, side: THREE.DoubleSide, alphaTest: 0.02,
    });
    card = new THREE.Mesh(geo, mat);
    card.position.copy(toThree(0, 0, aim_z));
    root.add(card);
  }

  // orbit path + camera frusta
  const pts = [];
  const n = Math.max(2, frames | 0);
  const span = THREE.MathUtils.degToRad(sweep_deg) * (ccw === false ? -1 : 1);
  const aim = toThree(0, 0, aim_z);
  const every = Math.max(1, Math.round(n / 24));

  for (let i = 0; i < n; i++) {
    const a = -Math.PI / 2
            + THREE.MathUtils.degToRad(+params.phase_deg || 0) + span * i / n;
    const P = toThree(radius * Math.cos(a), radius * Math.sin(a), camZAt(i));
    pts.push(P);
    // frame 0 is separated by WEIGHT (a brighter neutral), not by hue
    if (i % every === 0) {
      camGroup.add(frustum(P, aim, vfov_deg, width / height,
                           i === 0 ? C_FRUST0 : C_FRUST));
    }
  }
  pts.push(pts[0].clone());
  pathLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: C_PATH }));
  root.add(pathLine);

  // aim marker -- the one maroon object in the scene
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 12),
                             new THREE.MeshBasicMaterial({ color: C_AIM }));
  dot.position.copy(aim);
  camGroup.add(dot);

  setRigVisible(showRig);   // rebuilds must respect the current toggle
}

function frustum(pos, target, vfovDeg, aspect, color) {
  // Base at +d, because Object3D.lookAt points +Z at the target for ordinary
  // objects (only cameras look down -Z). With the base at -d every helper
  // faced AWAY from the subject -- the cone opened outward from the ring.
  const d = 1.1;
  const h = Math.tan(THREE.MathUtils.degToRad(vfovDeg) / 2) * d;
  const w = h * aspect;
  const g = new THREE.BufferGeometry();
  const local = [
    [0, 0, 0], [w, h, d], [0, 0, 0], [-w, h, d],
    [0, 0, 0], [w, -h, d], [0, 0, 0], [-w, -h, d],
    [w, h, d], [-w, h, d], [-w, h, d], [-w, -h, d],
    [-w, -h, d], [w, -h, d], [w, -h, d], [w, h, d],
  ].flat();
  g.setAttribute("position", new THREE.Float32BufferAttribute(local, 3));
  const line = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color }));
  line.position.copy(pos);
  // world-up before lookAt keeps the frustum upright: yaw + pitch only, no
  // roll, matching the real rig (its Track-To uses world up the same way)
  line.up.set(0, 1, 0);
  line.lookAt(target);
  return line;
}
