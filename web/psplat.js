// Ported from ../Img2Splat/web/psplat.js -- clear colour restyled, nothing else.
//
// Gaussian-splat viewer backed by the PlayCanvas engine — the same renderer
// SuperSplat (playcanvas/supersplat) is built on. Replaces the hand-rolled
// three.js splat shader, which produced visibly wrong output; the engine's
// gsplat pipeline does correct covariance projection and worker-based
// back-to-front sorting, and loads Brush's INRIA-convention .ply directly, so
// the server-side repacking step is unnecessary too.
//
// This renders on its own canvas. The three.js viewport keeps the orbit rig;
// the UI swaps which canvas is visible.
//
// Brush PLYs are Z-up (the pipeline's world). PlayCanvas is Y-up, so the splat
// entity is rotated -90° about X.

import * as pc from "./vendor/playcanvas.mjs";

let app = null, camera = null, entity = null, asset = null, canvasEl = null;
let resizeObs = null;
let started = false;

// simple orbit state (yaw/pitch in degrees)
const orbit = { yaw: 0, pitch: -8, dist: 6.0, target: new pc.Vec3(0, 2.3, 0) };

export async function init(canvas) {
  if (app) return;
  canvasEl = canvas;

  // pc.Application, not a hand-assembled AppOptions. Registering only
  // GSplatComponentSystem + GSplatHandler produced a loaded asset with
  // hasInstance:false and zero draw calls: engine 2.21 renders splats through a
  // "unified" path that needs more systems than that. Application registers the
  // full set, so the supported bootstrap is the one to use.
  // antialias off: MSAA does little for alpha-blended splats, and a
  // multisampled framebuffer makes readPixels illegal, which blocks
  // verification.
  app = new pc.Application(canvas, {
    graphicsDeviceOptions: { antialias: false, alpha: false, preserveDrawingBuffer: true },
  });
  app.setCanvasFillMode(pc.FILLMODE_NONE);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);

  camera = new pc.Entity("cam");
  camera.addComponent("camera", {
    clearColor: new pc.Color(0, 0, 0),               // --bg-viewport: pure black
    fov: 42,
    nearClip: 0.05,
    farClip: 500,
  });
  app.root.addChild(camera);
  applyOrbit();

  bindControls(canvas);

  // Guarded and disconnectable: destroy() hands the engine to the other
  // workspace, and an observer still firing afterwards reaches into a
  // torn-down graphicsDevice.
  const resize = () => {
    if (!app || !app.graphicsDevice) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    app.resizeCanvas(w, h);
    // resizeCanvas writes the size as INLINE styles, which beat the
    // stylesheet's inset:0. If the container later shrinks, the canvas keeps
    // its old CSS size and spills out over whatever sits below -- in the
    // refine tab that was the entire toolbar, unclickable underneath 131px of
    // canvas. Same pattern, same latent bug here, so the same fix.
    canvas.style.width = "";
    canvas.style.height = "";
  };
  resizeObs = new ResizeObserver(resize);
  resizeObs.observe(canvas);
  resize();

  app.start();
  started = true;
}

function applyOrbit() {
  const yr = orbit.yaw * Math.PI / 180, pr = orbit.pitch * Math.PI / 180;
  const cp = Math.cos(pr);
  camera.setPosition(
    orbit.target.x + orbit.dist * cp * Math.sin(yr),
    orbit.target.y - orbit.dist * Math.sin(pr),   // negative pitch = camera below
    orbit.target.z + orbit.dist * cp * Math.cos(yr));
  camera.lookAt(orbit.target);
}

function bindControls(canvas) {
  let dragging = false, panning = false, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", e => {
    dragging = true;
    panning = e.button === 2 || e.shiftKey;
    lx = e.clientX; ly = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", e => {
    if (!dragging) return;
    const dx = e.clientX - lx, dy = e.clientY - ly;
    lx = e.clientX; ly = e.clientY;
    if (panning) {
      // pan in the camera's screen plane
      const s = orbit.dist * 0.0016;
      const right = new pc.Vec3(); const up = new pc.Vec3();
      const rot = camera.getRotation();
      rot.transformVector(pc.Vec3.RIGHT, right);
      rot.transformVector(pc.Vec3.UP, up);
      orbit.target.sub(right.mulScalar(dx * s)).add(up.mulScalar(dy * s));
    } else {
      orbit.yaw -= dx * 0.35;
      orbit.pitch = Math.max(-89, Math.min(89, orbit.pitch - dy * 0.3));
    }
    applyOrbit();
  });
  const end = e => { dragging = false; };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    orbit.dist = Math.max(0.3, Math.min(80, orbit.dist * Math.exp(e.deltaY * 0.0011)));
    applyOrbit();
  }, { passive: false });
  canvas.addEventListener("contextmenu", e => e.preventDefault());
}

/** Load a splat PLY by URL, replacing the current one. Resolves to splat count. */
export function load(url) {
  return new Promise((resolve, reject) => {
    if (!app) return reject(new Error("psplat not initialised"));
    unload();
    app.assets.loadFromUrl(url, "gsplat", (err, a) => {
      if (err) return reject(new Error(String(err)));
      asset = a;
      entity = new pc.Entity("splat");
      // unified:false is required here. It defaults to TRUE in engine 2.21,
      // and in that mode the component creates no instance and no mesh
      // instance -- rendering goes through a global unified gsplat system that
      // is not active in this standalone setup, so nothing draws at all
      // (asset loaded, 436k splats, zero draw calls). The classic per-entity
      // path renders correctly.
      entity.addComponent("gsplat", { asset: a, unified: false });
      entity.setEulerAngles(-90, 0, 0);      // Z-up world -> Y-up engine
      app.root.addChild(entity);
      const n = a.resource?.gsplatData?.numSplats ?? a.resource?.numSplats ?? 0;
      resolve(n);
    });
  });
}

export function unload() {
  if (entity) { entity.destroy(); entity = null; }
  if (asset) {
    app.assets.remove(asset);
    asset.unload();
    asset = null;
  }
}

export function hasSplat() { return !!entity; }

/** Tear the whole app down.
 *
 * PlayCanvas holds ONE global current application, so a second one anywhere
 * on the page stops this one rendering -- the compare view in the refine tab
 * is a second one. Rather than have the two fight, whichever tab is on screen
 * owns the engine and the other releases it. init() is idempotent, so coming
 * back rebuilds cleanly.
 */
export function destroy() {
  if (!app) return false;
  try { if (resizeObs) resizeObs.disconnect(); } catch (e) { /* already gone */ }
  resizeObs = null;
  try { unload(); } catch (e) { /* already gone */ }
  try { app.destroy(); } catch (e) { /* engine already torn down */ }
  app = camera = entity = asset = canvasEl = null;
  started = false;
  return true;
}

/** Frame the subject. aimZ is the pipeline's Z-up aim height. */
export function focus(aimZ = 2.3, dist = 5.0) {
  orbit.target.set(0, aimZ, 0);   // after the -90° X rotation, world +Z is +Y
  orbit.dist = dist;
  orbit.yaw = 0;
  orbit.pitch = -8;
  applyOrbit();
}

/** Escape hatches for diagnosing engine-side rendering. */
export function _internals() { return { app, camera, entity, asset, pc }; }

/** Introspection for debugging — safe to call any time. */
export function debug() {
  if (!app) return { app: false };
  const comp = entity?.gsplat;
  const inst = comp?.instance;
  return {
    layers: app.scene?.layers?.layerList?.map(l => l.name + ":" + l.id),
    camLayers: camera?.camera?.layers,
    entityEnabled: entity?.enabled,
    compEnabled: comp?.enabled,
    compLayers: comp?.layers,
    hasInstance: !!inst,
    instKeys: inst ? Object.keys(inst).slice(0, 20) : null,
    meshInstance: !!(inst && (inst.meshInstance || inst._meshInstance)),
    sorterReady: inst?.sorter ? "has sorter" : "no sorter field",
    assetLoaded: !!asset?.loaded,
    resKeys: asset?.resource ? Object.keys(asset.resource).slice(0, 20) : null,
  };
}

/** Diagnostics hook: force one frame even when rAF is throttled (hidden pane). */
export function step() {
  if (!app) return null;
  try {
    app.update(1 / 60);
    app.render();
    const gl = app.graphicsDevice.gl;
    return {
      glError: gl ? gl.getError() : -1,
      canvas: canvasEl.width + "x" + canvasEl.height,
      splats: asset?.resource?.gsplatData?.numSplats ?? null,
      drawCalls: app.stats?.drawCalls ?? null,
    };
  } catch (e) {
    return { error: String(e) };
  }
}
