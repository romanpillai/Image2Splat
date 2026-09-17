// The passes workspace viewport: one splat, three views, and the rig drawn on
// top of it.
//
// Ported from ../Img2Splat/web/psplat3.js. Only the palette changed: the rig
// now uses the tool's neutrals with maroon on the aim point, and the two
// rings are separated by solid-vs-dashed rather than by cyan-vs-orange.
//
// It exists because the author viewport and this one need opposite things. The
// author rig is three.js and draws a POINT CLOUD; a trained splat needs
// PlayCanvas. Rather than run both engines (two of anything here has always
// ended in one of them going black), this follows the pattern psplat2.js
// already proved for the refine tab: ONE pc.Application, several cameras with
// rect viewports, and layers deciding which camera sees what.
//
// Three views:
//   free   the big left pane -- orbit it by hand, and it is the only one that
//          draws the rig, because frusta seen through their own frustum are
//          just noise
//   above  what the top ring's camera sees at the current frame
//   below  what the bottom ring's camera sees
//
// The rig is immediate-mode lines rather than meshes. It is rebuilt every
// frame from the current numbers, so there is no second copy of the ring
// geometry to keep in step with the controls -- the failure that made the
// author rings need a hard refresh.

import * as pc from "./vendor/playcanvas.mjs";

export function createPasses(canvas) {
  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: { antialias: false, alpha: false,
                             preserveDrawingBuffer: true },
  });
  app.setCanvasFillMode(pc.FILLMODE_NONE);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);

  const BG = new pc.Color(0.0, 0.0, 0.0);         // --bg-viewport: pure black
  // The control-video backdrop. ONE definition, used by both the preview panes
  // and the capture, because the preview's whole job is to be the render.
  const GREY = new pc.Color(0.5, 0.5, 0.5);

  // A splat layer PER VIEW, and its own gsplat instance in each.
  //
  // This is not tidiness -- gaussian splats must be sorted back-to-front FOR
  // THE CAMERA THAT DRAWS THEM, and an instance carries one sort order. Share
  // a single instance between cameras looking from different directions and
  // at most one of them is sorted correctly; the others draw far splats over
  // near ones, which reads as seeing the inside of the subject with the far
  // side looking larger than the near side. psplat2.js already does it this
  // way, one entity per pane.
  const layers = app.scene.layers;
  const splatLayers = [new pc.Layer({ name: "paSplatFree" }),
                       new pc.Layer({ name: "paSplatTop" }),
                       new pc.Layer({ name: "paSplatBot" })];
  const rigLayer = new pc.Layer({ name: "paRig" });
  // Deliberately empty: this layer exists only so the backdrop camera has one.
  const bgLayer = new pc.Layer({ name: "paBackdrop" });
  for (const l of splatLayers) layers.push(l);
  layers.push(rigLayer);
  layers.push(bgLayer);

  function makeCam(idx, seeRig, rect, priority) {
    const e = new pc.Entity("cam");
    e.addComponent("camera", {
      clearColor: BG, fov: 42, nearClip: 0.05, farClip: 2000,
      layers: seeRig ? [splatLayers[idx].id, rigLayer.id]
                     : [splatLayers[idx].id],
      rect, priority,
    });
    app.root.addChild(e);
    return e;
  }
  // free view left, the two pose views stacked on the right
  const camFree = makeCam(0, true, new pc.Vec4(0, 0, 0.62, 1), 0);
  const camTop = makeCam(1, false, new pc.Vec4(0.62, 0.5, 0.38, 0.5), 1);
  const camBot = makeCam(2, false, new pc.Vec4(0.62, 0, 0.38, 0.5), 2);
  camTop.camera.clearColor = GREY;
  camBot.camera.clearColor = GREY;

  // A camera that draws nothing and clears everything, at the lowest priority.
  //
  // The pose panes no longer fill their half of the canvas -- they are
  // letterboxed to the render's aspect -- and a camera only clears its OWN
  // rect. Without something behind them the margin keeps whatever was last
  // in the buffer.
  const camBack = new pc.Entity("camBack");
  camBack.addComponent("camera", {
    clearColor: BG, layers: [bgLayer.id], priority: -10,
    rect: new pc.Vec4(0, 0, 1, 1), nearClip: 0.1, farClip: 10,
  });
  app.root.addChild(camBack);

  const orbitState = { yaw: 0, pitch: -18, dist: 18, target: new pc.Vec3(0, 2.3, 0) };
  let ents = [], asset = null;
  let split = true;
  let frame = 0;
  let rings = { aimZ: 2.3, topZ: 5, topR: 9, botZ: -1, botR: 9,
                frames: 75, sweep: 360, vfov: 42, aspect: 16 / 9 };

  // world (x, y, z) -> engine (x, z, -y): the same mapping the whole tool uses
  const toE = (x, y, z) => new pc.Vec3(x, z, -y);

  function placeFree() {
    const yr = orbitState.yaw * Math.PI / 180, pr = orbitState.pitch * Math.PI / 180;
    const cp = Math.cos(pr);
    camFree.setPosition(
      orbitState.target.x + orbitState.dist * cp * Math.sin(yr),
      orbitState.target.y - orbitState.dist * Math.sin(pr),
      orbitState.target.z + orbitState.dist * cp * Math.cos(yr));
    camFree.lookAt(orbitState.target);
  }

  /** Where the pass camera sits for frame i on one ring, in WORLD units. */
  function ringCam(i, z, r) {
    const n = Math.max(1, rings.frames | 0);
    const span = (rings.sweep || 360) * Math.PI / 180;
    // frame 0 at -90 degrees, matching Orbit.angles() so the preview and any
    // later render cannot disagree about where the pass starts
    const a = -Math.PI / 2 + span * ((i % n) / n);
    return { x: r * Math.cos(a), y: r * Math.sin(a), z };
  }

  /** Put the two pose panes at the RENDER's aspect, letterboxed in their half.
   *
   * These panes are a preview of an mp4 frame, so they have to be that frame's
   * SHAPE. Before this they were whatever shape the pane happened to be (0.38
   * x 0.5 of the canvas) and the camera framed to that, so the preview cropped
   * differently from the render and could not answer the only question being
   * asked of it: does the subject fit?
   *
   * Fitted, never stretched: the image keeps the render's aspect and the
   * leftover margin is backdrop, so the grey rectangle you see IS the frame
   * edge.
   */
  function layoutPassCams() {
    const cw = Math.max(1, canvas.width), ch = Math.max(1, canvas.height);
    const A = Math.max(0.05, rings.aspect || 1);
    for (const [cam, x0, y0] of [[camTop, 0.62, 0.5], [camBot, 0.62, 0]]) {
      const pw = 0.38 * cw, ph = 0.5 * ch;
      let rw = pw, rh = pw / A;
      if (rh > ph) { rh = ph; rw = ph * A; }
      // a hair of inset so the grey frame reads as a frame, not as the pane
      rw *= 0.93; rh *= 0.93;
      cam.camera.rect = new pc.Vec4(
        x0 + (pw - rw) / 2 / cw, y0 + (ph - rh) / 2 / ch, rw / cw, rh / ch);
    }
  }

  function placePassCams() {
    for (const [cam, z, r] of [[camTop, rings.topZ, rings.topR],
                               [camBot, rings.botZ, rings.botR]]) {
      const c = ringCam(frame, z, r);
      cam.setPosition(toE(c.x, c.y, c.z));
      cam.camera.fov = rings.vfov;
      // Entity.up is a COMPUTED getter -- assigning to it does nothing, which
      // is what the previous line here did. lookAt takes the up hint instead.
      cam.lookAt(toE(0, 0, rings.aimZ), pc.Vec3.UP);
    }
  }

  // ---- the rig, redrawn every frame from the live numbers -------------------
  // No colour vocabulary. The two rings are spatially separated already, so
  // hue was doing no work: above is a BRIGHT neutral, below a dimmer one, and
  // the labels carry the arrow glyphs. Maroon marks the aim point only --
  // the same one accent the rest of the tool uses.
  const C_TOP = new pc.Color(0.878, 0.878, 0.878);   // --n-200
  const C_BOT = new pc.Color(0.502, 0.502, 0.502);   // --n-400
  const C_AIM = new pc.Color(0.769, 0.451, 0.494);   // --m-300
  const C_GRID = new pc.Color(0.169, 0.169, 0.169);  // --n-800

  // `dashed` is how the two rings are told apart without a second hue: the
  // above ring is a solid bright line, the below ring a dashed dimmer one.
  function drawRing(z, r, col, showFrusta, dashed) {
    const seg = 96;
    let prev = null;
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      const v = toE(r * Math.cos(a), r * Math.sin(a), z);
      if (prev && !(dashed && (i % 2))) app.drawLine(prev, v, col, true, rigLayer);
      prev = v;
    }
    if (!showFrusta) return;
    // One frustum every few frames: enough to read the pass, not so many that
    // the ring disappears behind its own wireframe.
    const n = Math.max(1, rings.frames | 0);
    const every = Math.max(1, Math.round(n / 16));
    for (let i = 0; i < n; i += every) drawFrustum(ringCam(i, z, r), col);
  }

  /** A small camera wireframe at c, aimed at the aim point. */
  function drawFrustum(c, col) {
    const P = toE(c.x, c.y, c.z);
    const aim = toE(0, 0, rings.aimZ);
    const fwd = new pc.Vec3().sub2(aim, P).normalize();
    const up0 = new pc.Vec3(0, 1, 0);
    const right = new pc.Vec3().cross(fwd, up0).normalize();
    const up = new pc.Vec3().cross(right, fwd).normalize();
    const d = Math.max(0.35, rings.topR * 0.05);
    const h = d * Math.tan(rings.vfov * Math.PI / 360);
    const w = h * rings.aspect;
    const ctr = new pc.Vec3().add2(P, new pc.Vec3().copy(fwd).mulScalar(d));
    const corner = (sx, sy) => new pc.Vec3()
      .add2(ctr, new pc.Vec3()
        .add2(new pc.Vec3().copy(right).mulScalar(w * sx),
              new pc.Vec3().copy(up).mulScalar(h * sy)));
    const c00 = corner(-1, -1), c10 = corner(1, -1),
          c11 = corner(1, 1), c01 = corner(-1, 1);
    for (const q of [c00, c10, c11, c01]) app.drawLine(P, q, col, true, rigLayer);
    app.drawLine(c00, c10, col, true, rigLayer);
    app.drawLine(c10, c11, col, true, rigLayer);
    app.drawLine(c11, c01, col, true, rigLayer);
    app.drawLine(c01, c00, col, true, rigLayer);
  }

  function drawGrid() {
    const R = Math.max(rings.topR, rings.botR) * 1.35, step = R / 8;
    for (let i = -8; i <= 8; i++) {
      const t = i * step;
      app.drawLine(toE(-R, t, 0), toE(R, t, 0), C_GRID, true, rigLayer);
      app.drawLine(toE(t, -R, 0), toE(t, R, 0), C_GRID, true, rigLayer);
    }
  }

  let showRig = true;
  app.on("update", () => {
    // Keep the backdrop layer non-empty. A layer with no mesh instances can be
    // skipped by the composition, and a skipped layer never runs its camera's
    // CLEAR -- which is the only thing camBack is there to do. One zero-length
    // line is enough to keep it in the pass and draws no pixels.
    const z = new pc.Vec3(0, 0, 0);
    app.drawLine(z, z, BG, true, bgLayer);
    if (!showRig) return;
    drawGrid();
    drawRing(rings.topZ, rings.topR, C_TOP, true);
    drawRing(rings.botZ, rings.botR, C_BOT, true, true);
    // aim point: a small cross, so "where the cameras converge" is visible
    const a = toE(0, 0, rings.aimZ), s = 0.35;
    app.drawLine(new pc.Vec3(a.x - s, a.y, a.z), new pc.Vec3(a.x + s, a.y, a.z), C_AIM, true, rigLayer);
    app.drawLine(new pc.Vec3(a.x, a.y - s, a.z), new pc.Vec3(a.x, a.y + s, a.z), C_AIM, true, rigLayer);
    app.drawLine(new pc.Vec3(a.x, a.y, a.z - s), new pc.Vec3(a.x, a.y, a.z + s), C_AIM, true, rigLayer);
  });

  // ---- input: the free pane only -------------------------------------------
  let dragging = false, panning = false, lx = 0, ly = 0;
  const inFree = (e) => {
    const r = canvas.getBoundingClientRect();
    return !split || (e.clientX - r.left) / r.width < 0.62;
  };
  canvas.addEventListener("pointerdown", e => {
    if (!inFree(e)) return;
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
      const s = orbitState.dist * 0.0016;
      const right = new pc.Vec3(), up = new pc.Vec3();
      const rot = camFree.getRotation();
      rot.transformVector(pc.Vec3.RIGHT, right);
      rot.transformVector(pc.Vec3.UP, up);
      orbitState.target.sub(right.mulScalar(dx * s)).add(up.mulScalar(dy * s));
    } else {
      orbitState.yaw -= dx * 0.35;
      orbitState.pitch = Math.max(-89, Math.min(89, orbitState.pitch - dy * 0.3));
    }
    placeFree();
  });
  const end = () => { dragging = false; };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("wheel", e => {
    if (!inFree(e)) return;
    e.preventDefault();
    orbitState.dist = Math.max(0.3,
      Math.min(4000, orbitState.dist * Math.exp(e.deltaY * 0.0011)));
    placeFree();
  }, { passive: false });
  canvas.addEventListener("contextmenu", e => e.preventDefault());

  // The observer OUTLIVES the app unless it is disconnected. The workspaces
  // hand the engine back and forth (only one PlayCanvas app may live at a
  // time), so after destroy() the canvas keeps resizing -- and every one of
  // those ticks reached into a torn-down graphicsDevice. Measured: eight
  // "Cannot read properties of null (reading 'canvas')" per tab switch.
  let dead = false;
  const resize = () => {
    if (dead || !app || !app.graphicsDevice) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    app.resizeCanvas(w, h);
    // resizeCanvas writes width/height as INLINE styles that beat the sheet;
    // clearing them lets the stylesheet's inset:0 decide the displayed size.
    canvas.style.width = "";
    canvas.style.height = "";
    // rings.aspect used to be recomputed HERE from the pane's shape, which is
    // why the preview and the frusta disagreed with the mp4. It is the
    // render's aspect now, set by the caller from the parent clip, and the
    // panes are laid out to match it instead of the other way round.
    layoutPassCams();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();
  placeFree();
  layoutPassCams();
  placePassCams();
  app.start();

  function clear() {
    for (const e of ents) e.destroy();
    ents = [];
    if (asset) { app.assets.remove(asset); asset.unload(); asset = null; }
  }

  function load(url) {
    return new Promise((resolve, reject) => {
      clear();
      app.assets.loadFromUrl(url, "gsplat", (err, a) => {
        if (err) return reject(new Error(String(err)));
        // ONE asset, three instances -- the splat data is shared, only the
        // per-camera sort order differs, which is the whole point.
        for (let i = 0; i < splatLayers.length; i++) {
          const e = new pc.Entity("splat" + i);
          // unified:false -- the default true path renders through a global
          // system that is not active here and draws nothing at all.
          e.addComponent("gsplat", { asset: a, unified: false,
                                     layers: [splatLayers[i].id] });
          e.setEulerAngles(-90, 0, 0);        // Z-up world -> Y-up engine
          app.root.addChild(e);
          ents.push(e);
        }
        asset = a;
        resolve(a.resource?.gsplatData?.numSplats ?? a.resource?.numSplats ?? 0);
      });
    });
  }

  function setSplit(on) {
    split = !!on;
    camFree.camera.rect = split ? new pc.Vec4(0, 0, 0.62, 1)
                                : new pc.Vec4(0, 0, 1, 1);
    camTop.camera.enabled = split;
    camBot.camera.enabled = split;
  }
  setSplit(true);

  function setRings(r) {
    Object.assign(rings, r || {});
    layoutPassCams();      // aspect may have changed with the parent clip
    placePassCams();
  }
  function setFrame(i) { frame = i | 0; placePassCams(); }
  function setRig(on) { showRig = !!on; }

  function focus(aimZ = 2.3, dist = 0) {
    orbitState.target.set(0, aimZ, 0);
    orbitState.dist = dist || Math.max(rings.topR, rings.botR) * 2.0 || 18;
    orbitState.yaw = 0; orbitState.pitch = -18;
    placeFree();
  }

  // ---- rendering a pass -----------------------------------------------------
  /** Render one frame of a pass and hand back its PNG bytes.
   *
   * The pass camera takes the WHOLE canvas, the rig is hidden and the clear
   * colour becomes flat neutral grey: this is a control video, so anything
   * that is not the subject is noise the video model would try to interpret.
   * Grey rather than black because a black backdrop and a dark subject are
   * indistinguishable to a matte, and the training alpha comes from this.
   *
   * Restores every setting afterwards -- a half-reconfigured viewer that looks
   * broken is worse than a slow render.
   */
  async function renderPassFrame(which, i, w, h) {
    const cam = which === "top" ? camTop : camBot;
    const saved = {
      rect: cam.camera.rect.clone(),
      clear: cam.camera.clearColor.clone(),
      prio: cam.camera.priority,
      free: camFree.camera.enabled,
      back: camBack.camera.enabled,
      other: (which === "top" ? camBot : camTop).camera.enabled,
      rig: showRig, frame,
      cw: canvas.width, ch: canvas.height,
    };
    try {
      // RESOLUTION_AUTO puts the canvas back to its DISPLAYED size on the
      // next render, so setting width/height here was silently ignored -- a
      // request for 832x480 captured at 1332x629, whose odd height then made
      // x264 fail outright. Pin the resolution for the capture instead.
      app.setCanvasResolution(pc.RESOLUTION_FIXED, w, h);
      canvas.width = w; canvas.height = h;
      showRig = false;
      camBack.camera.enabled = false;    // the pass camera covers the frame
      camFree.camera.enabled = false;
      (which === "top" ? camBot : camTop).camera.enabled = false;
      cam.camera.enabled = true;
      cam.camera.rect = new pc.Vec4(0, 0, 1, 1);
      cam.camera.clearColor = GREY;          // the same grey the pane shows
      cam.camera.priority = -1;
      cam.camera.aspectRatioMode = pc.ASPECT_AUTO;
      frame = i;
      placePassCams();
      // LET THE SORTER CATCH UP before capturing.
      //
      // Splat depth sorting runs on a worker and lands a frame or more after
      // the camera moves, so rendering once immediately after placing the
      // camera captures the PREVIOUS pose's sort order. Back-to-front becomes
      // front-to-back, near splats get overdrawn by far ones, and the result
      // reads as seeing the inside of the subject -- at some angles and not
      // others, which is exactly how it presented.
      //
      // Three render passes with a real frame boundary between them, so the
      // worker has delivered before the pixels are read.
      for (let k = 0; k < 3; k++) {
        app.render();
        await new Promise(res => requestAnimationFrame(res));
      }
      app.render();
      // toBlob is async and does not tear the buffer the way a same-frame
      // toDataURL can; preserveDrawingBuffer is on so the pixels survive.
      return await new Promise(res => canvas.toBlob(res, "image/png"));
    } finally {
      cam.camera.rect = saved.rect;
      cam.camera.clearColor = saved.clear;
      cam.camera.priority = saved.prio;
      camFree.camera.enabled = saved.free;
      camBack.camera.enabled = saved.back;
      (which === "top" ? camBot : camTop).camera.enabled = saved.other;
      showRig = saved.rig;
      frame = saved.frame;
      app.setCanvasResolution(pc.RESOLUTION_AUTO);
      canvas.width = saved.cw; canvas.height = saved.ch;
      layoutPassCams();
      placePassCams();
      setSplit(split);
      app.render();
    }
  }

  return {
    load, setSplit, setRings, setFrame, setRig, focus, renderPassFrame,
    hasSplat: () => ents.length > 0,
    frame: () => frame,
    destroy: () => {
      dead = true;
      try { ro.disconnect(); } catch (e) { /* already gone */ }
      clear();
      app.destroy();
    },
    step: () => {
      try { app.update(1 / 60); app.render(); return { frame: app.frame }; }
      catch (e) { return { error: String(e).slice(0, 200) }; }
    },
    debug: () => ({
      split, frame, splatInstances: ents.length, showRig,
      canvas: canvas.width + "x" + canvas.height,
      rings: { ...rings },
      cams: { free: camFree.camera.enabled, top: camTop.camera.enabled,
              bot: camBot.camera.enabled },
    }),
  };
}
