// The author workspace: source -> cloud -> shot -> generate -> review -> train.
//
// Every derived fact in here lands in a STATUS BLOCK row, not in a paragraph.
// If a condition should stop the user, it is a status row in the problem state
// that DISABLES the step's primary button. A red sentence inside prose is not
// a blocker, it is decoration.
//
// The flow does the chores itself wherever a chore has only one right answer:
// a new photo measures its lens and lifts its points; a changed lift setting
// re-lifts; Render re-lifts first when the points on screen are stale and
// softens in the same job; Generate re-renders a stale control video first;
// Approve opens Train. Every one of those says what it did -- on the button,
// over the viewport, and in the console.

import {
  $, num, esc, api, apiRaw, run, runDetached, logUi, S, renderStatus,
  renderStatusFolded, gate, bust, mediaHtml, dialog, fireStateChange, fx,
  openStep, pairSlider, spanSlider, jobStart, jobEnd, jobActive, flashOutcome,
  shortErr,
} from "./core.js";
import * as orbit from "./orbit.js";
import * as psplat from "./psplat.js";
import { apvShow, settingsHtml } from "./approve.js";

const FF_DIAG_MM = 43.267;                 // full-frame sensor diagonal
const MAX_SECONDS = { "720p": 6.0, "480p": 15.0 };   // LTX only -- see report
const LTX_RES = ["720p", "480p"];
const VACE_RES = ["auto", "240p", "360p", "480p", "580p", "720p"];
const ASPECT_RATIOS = { "9:16": 9 / 16, "3:4": 3 / 4, "1:1": 1,
                        "4:3": 4 / 3, "16:9": 16 / 9, "21:9": 21 / 9 };
const CUTOFF_MAX = 25.0;
const CANVAS_LONG = 1920;                  // the author canvas: long edge, locked

// The cutoff slider is non-linear: (t/1000) squared, scaled to CUTOFF_MAX.
// Linear travel put the entire useful range -- the metre or two just behind
// the subject -- into a few pixels near zero.
export const cutoffFromSlider = t => {
  const f = Math.min(1, Math.max(0, (+t || 0) / 1000));
  return Math.round(f * f * CUTOFF_MAX * 1000) / 1000;
};
let cloudPoints = 0;
let liftedWith = null;        // the lift key the points on screen were made with
let reliftTimer = null;
let liftInFlight = null, liftQueued = false;
let gizmoState = "off";
let ckList = [], ckLoaded = null, splatView = false;
let onDirty = () => {};       // set by app.js: persist + repaint
let shownSource = null;       // which image URL the card is currently showing
let playTimer = null;
let filesKey = "", filesMtimes = null;
let autoFlow = "";            // what the new-photo chain is doing right now
let lastLiftError = "";       // shown in Cloud until the next lift succeeds
let clayPreview = true;       // camera pane: flat clay like the render, or colour

export function setDirtyHook(fn) { onDirty = fn; }
const authorVisible = () => $("wsAuthorPane").classList.contains("on");

// ============================================================== geometry ===
/** The canvas: the photo's aspect on a 32px grid, long edge LOCKED at 1920.
 *
 * Source-fit is the only canvas in the author workspace -- the camera view is
 * the photo, front-on. 32px because the video models want multiples of 8 at
 * minimum and 32 keeps the downsampled softening sizes whole as well. A photo
 * smaller than 1920 is rendered at 1920 too (upscaled): if that ever looks
 * soft, the fix is to cap the long edge at the photo's own size again.
 */
export function srcDims() {
  const sw = +S.state.source_w || 0, sh = +S.state.source_h || 0;
  if (!sw || !sh) return null;
  const ar = sw / sh;
  const snap = v => Math.max(256, Math.round(v / 32) * 32);
  return ar >= 1 ? { width: CANVAS_LONG, height: snap(CANVAS_LONG / ar) }
                 : { width: snap(CANVAS_LONG * ar), height: CANVAS_LONG };
}

/** Mirror of steps.card_height_full(). */
function cardHeightFull(p) {
  const camZ = p.aim_z + p.radius * Math.tan(p.elev_deg * Math.PI / 180);
  const d = Math.hypot(p.radius, p.aim_z - camZ);
  let h = 2 * d * Math.tan((p.vfov_deg * Math.PI / 180) / 2);
  const sw = +S.state.source_w || 0, sh = +S.state.source_h || 0;
  if (sw && sh) {
    const arSrc = sw / sh, arCanvas = p.width / p.height;
    if (arSrc > arCanvas) h *= arCanvas / arSrc;   // contain, never crop
  }
  return h;
}

const fpsNow = () => Math.max(1, num("oFps", 24));
export const framesNow = () => Math.max(1, Math.round(num("oDur", 5) * fpsNow()));

function pathMode() {
  const v = $("oPath").value;
  return v === "helix" ? "helix" : "ring";
}

export function orbitParams() {
  const radius = num("oRad", 9), aim_z = num("oAimZ", 2.3);
  // FRONT-ON, always, in this workspace: elevation 0 and no offset from the
  // photo. The passes workspace is where the camera leaves this plane.
  const elev_deg = 0;
  const path = pathMode();
  const helix_min_z = num("oHelixMin", 0), helix_max_z = num("oHelixMax", 3);
  const helix_start = num("oHelixStart", 0) / 100;
  const lo = Math.min(helix_min_z, helix_max_z);
  const hi = Math.max(helix_min_z, helix_max_z);
  const p0 = ((helix_start % 1) + 1) % 1;
  const cam_z = path === "helix"
    ? lo + (hi - lo) * (1 - 2 * Math.abs(p0 - 0.5))
    : aim_z;
  const width = num("oW", 1920), height = num("oH", 1088);
  const vfov_deg = num("oFov", 42);
  // THE CARD IS A FIXED OBJECT IN THE WORLD: seeded so frame 0 reproduces the
  // photograph, re-seeded on demand, a constant in between.
  const card = num("oCard", 0);
  return {
    frames: framesNow(), sweep_deg: num("oSweep", 360),
    radius, elev_deg, aim_z, cam_z,
    path, helix_min_z, helix_max_z, helix_start,
    elev_sweep_deg: $("oPath").value === "ring-rock" ? num("oElevSweep", 0) : 0,
    elev_cycles: Math.max(0, num("oElevCycles", 1)),
    vfov_deg, width, height, ccw: true,
    card_height: (isFinite(card) && card > 0) ? +card.toFixed(4) : 3.6,
    phase_deg: 0,
  };
}

function subjectParams() {
  return { x: num("sX", 0), y: num("sY", 0), z: num("sZ", 0),
           rot: num("sRot", 0), rotX: num("sRotX", 0), rotY: num("sRotY", 0) };
}

export function cropSphere() {
  if (!$("cCropOn").checked) return null;
  const r = num("cCropR", 0);
  if (!(r > 0)) return null;
  return [num("cCropX", 0), num("cCropY", 0), num("cCropZ", 0), r];
}

/** Everything the lift depends on, as one comparable string. When it differs
 *  from what the points on screen were lifted with, the next render re-lifts
 *  first -- there is no warning to act on any more. */
function liftKey() {
  const p = orbitParams();
  return JSON.stringify({
    pts: $("oPoints").value, model: depthModelNow(),
    prune: num("cPrune", 0), crop: cropSphere(),
    radius: p.radius, aim: p.aim_z, card: p.card_height,
  });
}

// ============================================================== push/pull ===
/** Any control that moves a camera PUSHES TO THE VIEWPORT, not just a hint. */
export function pushOrbit() {
  const p = orbitParams();
  const hint = $("oCardHint");
  if (hint) {
    const full = cardHeightFull(p);
    const off = full > 0 ? (p.card_height / full) : 1;
    hint.textContent = Math.abs(off - 1) < 0.005
      ? "Fitted — frame 0 reproduces the photograph at this radius."
      : `The subject fills ${(off * 100).toFixed(0)}% of the frame at this `
        + `radius. Re-fit to make frame 0 match the photograph again.`;
  }
  $("canvasD").textContent = `${p.width} × ${p.height}`;
  orbit.setParams(p);
  orbit.setCloudParams({
    strength: reliefNow(),
    cutoff: cutoffFromSlider(num("cCut", 0)),
  });
  const clayOn = $("oClay").value !== "off";
  orbit.setClay(clayOn && clayPreview);
  const vc = $("vClay");
  if (vc) {
    vc.classList.toggle("hidden", !clayOn);
    vc.textContent = clayPreview ? "Camera: clay" : "Camera: colour";
    vc.title = clayPreview
      ? "The camera view shows flat clay, exactly as the control video renders. Click to preview the photo colours."
      : "The camera view shows photo colours; the control video still renders as clay. Click to go back to clay.";
  }
  orbit.setSubject(subjectParams());
  const c = cropSphere();
  orbit.setCrop(c ? { on: true, x: c[0], y: c[1], z: c[2], r: c[3] } : { on: false });
  const sl = $("fSlider");
  sl.max = String(Math.max(0, p.frames - 1));
  if (+sl.value > +sl.max) sl.value = sl.max;
  setFrame(+sl.value);
  repaint();
}

/** Re-lift after a pause in the input. Dragging a slider fires ONE lift when
 *  you let go, not one per tick. The cleanup and the sphere need neighbour
 *  queries over the whole cloud, so unlike strength and cutoff they cannot
 *  ride a shader uniform -- they need the server. */
function reliftSoon(delay = 450) {
  clearTimeout(reliftTimer);
  reliftTimer = null;
  if (!S.state.source) return;
  reliftTimer = setTimeout(() => {
    reliftTimer = null;
    if (liftedWith !== liftKey() || !cloudPoints) requestLift("settings changed");
    else repaint();
  }, delay);
  repaint();
}

/** One lift at a time. A request while one is running queues ONE follow-up,
 *  which only runs if the settings still differ when the first comes back. */
function requestLift(why, opts = {}) {
  if (!S.state.source) return Promise.resolve(false);
  if (liftInFlight) { liftQueued = true; return liftInFlight; }
  repaint();
  liftInFlight = liftCloud({ why, ...opts })
    .then(() => true, () => false)
    .finally(() => {
      liftInFlight = null;
      repaint();
      if (liftQueued) {
        liftQueued = false;
        if (liftedWith !== liftKey()) requestLift("settings changed while lifting");
      }
    });
  return liftInFlight;
}

// =============================================================== statuses ===
function aspectOption(w, h) {
  const want = w / Math.max(h, 1);
  let best = null, err = 1e9;
  for (const [n, r] of Object.entries(ASPECT_RATIOS)) {
    const e = Math.abs(r - want);
    if (e < err) { best = n; err = e; }
  }
  return err <= 0.02 * want ? best : "adaptive";
}

function lensLook(mm) {
  if (mm < 16) return "ultra-wide";
  if (mm < 28) return "wide";
  if (mm < 45) return "standard-wide";
  if (mm < 65) return "standard";
  if (mm < 105) return "short telephoto";
  if (mm < 200) return "telephoto";
  return "long telephoto";
}

function lensMm() {
  const v = num("oFov", 0), W = num("oW", 0), H = num("oH", 0);
  if (!(v > 0 && W > 0 && H > 0)) return null;
  const tv = Math.tan((v * Math.PI / 180) / 2);
  const th = tv * (W / H);
  const td = Math.hypot(th, tv);
  return { mm: (FF_DIAG_MM / 2) / td,
           h: 2 * Math.atan(th) * 180 / Math.PI,
           d: 2 * Math.atan(td) * 180 / Math.PI };
}

const engineNow = () => (S.engines || []).find(e => e.id === $("gEngine").value) || null;

/** Relief in world units, from the slider (0-400 -> 0.00-4.00). */
function reliefNow() { return num("oDepthStr", 150) / 100; }

/** The lift's depth model. MoGe-2 and DA3 Metric are metric; Depth Anything
 *  V2 and DA3 Mono are relative, so their relief is world units across the
 *  photo. MoGe-2 measures the lens either way. */
const DEPTH_NAMES = { moge2: "MoGe-2", da2: "Depth Anything V2 Small", "da2-large": "Depth Anything V2 Large",
                      "da3-metric": "Depth Anything 3 Metric Large", "da3-mono": "Depth Anything 3 Mono Large" };
const DEPTH_METRIC = new Set(["moge2", "da3-metric"]);
const DEPTH_DOWNLOAD = { "da2-large": "non-commercial licence · 1.3 GB download on first use",
                         "da3-metric": "1.4 GB download on first use", "da3-mono": "1.4 GB download on first use" };
function depthModelNow() { const v = $("oDepthModel").value; return DEPTH_NAMES[v] ? v : "moge2"; }
const depthNameNow = () => DEPTH_NAMES[depthModelNow()];
const depthMetricNow = () => DEPTH_METRIC.has(depthModelNow());

/** Give the card a size, once, and then leave it alone. The server's number
 *  (what control.mp4 was rendered with) wins unless you moved it yourself. */
function seedCard() {
  const box = $("oCard");
  if (!box) return;
  const fromServer = +S.state.card_height || 0;
  if (fromServer > 0) {
    if (!box.dataset.touched || +box.value <= 0) {
      if (Math.abs(+box.value - fromServer) > 0.0005) {
        box.value = fromServer.toFixed(4);
      }
    }
    return;
  }
  if (+box.value > 0) return;
  if (!S.state.source) return;
  fitCard("seeded");
}

/** Fit the card so frame 0 reproduces the photograph at the current radius. */
function fitCard(why = "re-fitted") {
  const p = orbitParams();
  const h = cardHeightFull({ ...p, elev_deg: 0 });
  if (!(h > 0)) return 0;
  $("oCard").value = h.toFixed(4);
  logUi(`card height ${why} at ${h.toFixed(4)} — frame 0 reproduces the source `
      + `photo at radius ${p.radius}`);
  return h;
}

/** The canvas follows the photo: source-fit is the only canvas here. */
function syncCanvas() {
  const d = srcDims();
  if (!d) return;
  if (+$("oW").value === d.width && +$("oH").value === d.height) return;
  const was = `${$("oW").value}×${$("oH").value}`;
  $("oW").value = d.width;
  $("oH").value = d.height;
  logUi(`canvas ${was} → ${d.width}×${d.height} — long edge ${CANVAS_LONG}, `
      + `matching the ${S.state.source_w}×${S.state.source_h} photo`);
  pushOrbit();
}

function statusSource() {
  const st = S.state, fov = st.fov || {};
  const dir = (S.projects_dir || "") + "\\" + S.project;
  const rows = [
    { k: "photo", v: st.source ? `${st.source} · ${st.source_w} × ${st.source_h}` : "none yet — drop one above",
      problem: !st.source },
    { k: "lens", v: fov.vfov_deg != null
        ? `${(+fov.vfov_deg).toFixed(2)}° vertical (MoGe-2)`
        : (st.source ? "not measured yet" : "") },
    { k: "path", v: st.source ? `${dir}\\${st.source}` : "" },
  ];
  // The measurement SEEDS the shot; a lens set by hand is left alone and the
  // difference is offered as one click, never applied behind your back.
  if (fov.vfov_deg != null && Math.abs(num("oFov", 0) - +fov.vfov_deg) > 0.05) {
    rows.push({ k: "shot lens", v: `yours ${num("oFov", 0).toFixed(2)}° · measured ${(+fov.vfov_deg).toFixed(2)}°`,
      fix: { label: "Use measured", onClick: e => {
        const b = e.currentTarget;
        applyMeasuredFov(true);
        fx.toast(`Vertical FOV set to ${(+fov.vfov_deg).toFixed(2)}° (measured)`, "ok");
        if (b && b.isConnected) fx.btnDone(b, true, "Applied");
      } } });
  }
  renderStatusFolded("stSource", "Source", rows, "file and lens");
}

/** Points per pixel of the clip that is actually SENT. Softening halves both
 *  sides, so a 1920 canvas softened 2x needs a quarter of the points. */
function densityNow() {
  const p = orbitParams();
  const f = softOn() ? Math.max(1, num("gSoftDown", 1)) : 1;
  const px = (p.width / f) * (p.height / f);
  const pw = +$("oPoints").value;
  const est = pw * Math.round(pw * (+S.state.source_h || 1) / (+S.state.source_w || 1));
  return { ppp: (cloudPoints || est) / Math.max(px, 1), w: Math.round(p.width / f), h: Math.round(p.height / f) };
}

function statusCloud() {
  const c = cropSphere();
  const dn = densityNow();
  const busy = !!liftInFlight || !!autoFlow, queued = !!reliftTimer;
  const rows = [];
  // ONE live line that is always present -- it changes words, never height, so
  // the Regenerate button does not jump while a lift runs.
  if (S.state.source) {
    const stale = cloudPoints && liftedWith !== liftKey();
    rows.push({ k: "", live: true, note: true,
      v: autoFlow || (liftInFlight ? "Lifting the points…"
        : queued ? "Settings changed — re-lifting in a moment…"
        : lastLiftError && cloudPoints ? `${cloudPoints.toLocaleString()} points on screen — from the last lift that worked`
        : cloudPoints ? `${cloudPoints.toLocaleString()} points on screen${stale ? " — re-lifted on the next render" : " — current"}`
        : "No points yet — press Regenerate") });
  }
  if (lastLiftError && !busy) {
    rows.push({ k: "last lift", problem: true, v: lastLiftError,
      fix: { label: "Try again", onClick: () => $("btnCloud").click() } });
  }
  rows.push({ k: "points", v: cloudPoints ? cloudPoints.toLocaleString() : (busy ? "lifting…" : "none yet"),
    problem: !cloudPoints && !!S.state.source && !busy && !queued && !lastLiftError });
  rows.push({ k: "density", v: `${dn.ppp.toFixed(2)} per pixel of the ${dn.w}×${dn.h} clip`,
    from: dn.ppp < 0.6 ? "sparse" : "dense enough" });
  if (dn.ppp < 0.6 && S.state.source && cloudPoints && !busy && !queued && !lastLiftError) {
    rows.push({ k: "", live: true, note: true,
      v: c && cloudPoints < 60000
        ? `Only ${cloudPoints.toLocaleString()} points inside the isolate sphere — widen it or move it onto the subject.`
        : `Sparse for a ${dn.w}×${dn.h} clip — raise point density or soften more.` });
  }
  const da3Missing = depthModelNow().startsWith("da3") && S.health && S.health.da3 === false;
  rows.push({ k: "depth model", v: `${depthNameNow()} — ${depthMetricNow() ? "metric" : "relative"}`,
    problem: da3Missing,
    from: da3Missing ? "Depth Anything 3 is not installed — see requirements-optional.txt"
                     : (DEPTH_DOWNLOAD[depthModelNow()] || "") });
  rows.push({ k: "isolate", v: c ? `r ${c[3].toFixed(2)} about (${c[0].toFixed(2)}, ${c[1].toFixed(2)}, ${c[2].toFixed(2)})` : "off" });
  rows.push({ k: "depth cutoff", v: cutoffFromSlider(num("cCut", 0)) > 0
      ? `${cutoffFromSlider(num("cCut", 0)).toFixed(2)} behind the card` : "off" });
  rows.push({ k: "saved as", v: cloudPoints ? `${S.state._dir || ""}\\cloud_view.bin` : "—",
    from: cloudPoints ? "restored on refresh" : "" });
  const n = renderStatusFolded("stCloud", "Cloud", rows, "lift detail");
  gate("btnCloud", S.state.source ? 0 : 1, "Regenerate");
  return n;
}

/** How the camera on disk differs from the camera in the rail. ONE list, read
 *  by both the Shot and the Generate status. */
const CAM_FIELDS = [
  ["radius", "radius", 0.001],
  ["aim_z", "aim height", 0.001],
  ["vfov_deg", "vertical FOV", 0.01],
  ["sweep_deg", "azimuth sweep", 0.01],
  ["elev_sweep_deg", "elevation amplitude", 0.01],
  ["elev_cycles", "elevation cycles", 0.001],
  ["helix_min_z", "band low z", 0.001],
  ["helix_max_z", "band high z", 0.001],
  ["helix_start", "helix start", 0.001],
];

function camDrift(p) {
  const rec = S.state.orbit || {};
  if (!S.state.control_video) return [];
  const out = [];
  for (const [k, label, tol] of CAM_FIELDS) {
    if (rec[k] == null || p[k] == null) continue;
    if (Math.abs(+rec[k] - +p[k]) > tol) out.push(`${label} ${+rec[k]} → ${+p[k]}`);
  }
  if (rec.path && p.path && rec.path !== p.path) out.push(`path ${rec.path} → ${p.path}`);
  return out;
}

/** Why control.mp4 no longer matches the shot, as short phrases ([] = fresh). */
function controlStale() {
  const st = S.state, p = orbitParams();
  if (!st.control_video) return [];
  const rec = st.orbit || {};
  const bad = [];
  if (rec.frames && rec.frames !== p.frames) bad.push(`${rec.frames} → ${p.frames} frames`);
  const recFps = +st.fps || 0;
  if (recFps && recFps !== fpsNow()) bad.push(`${recFps} → ${fpsNow()} fps`);
  if (rec.width && rec.height && (rec.width !== p.width || rec.height !== p.height)) {
    bad.push(`canvas ${rec.width}×${rec.height} → ${p.width}×${p.height}`);
  }
  const recCard = +st.card_height || 0;
  if (recCard && Math.abs(recCard - p.card_height) > 0.001) {
    bad.push(`card ${recCard.toFixed(3)} → ${p.card_height.toFixed(3)}`);
  }
  for (const b of camDrift(p)) bad.push(b);
  if (rec.elev_deg && Math.abs(+rec.elev_deg) > 0.01) bad.push(`elevation ${rec.elev_deg}° → 0° (front-on)`);
  if ((st.orbit_info || {}).clay && ((st.orbit_info.clay === "colour") !== ($("oClay").value === "off"))) {
    bad.push(`look → ${$("oClay").value === "off" ? "source colour" : "flat clay"}`);
  }
  return bad;
}

function statusShot() {
  const p = orbitParams();
  const fps = fpsNow();
  const span = orbit.camZSpan();
  const lens = lensMm();
  const rows = [
    { k: "frames × fps", v: `${p.frames} × ${fps} = ${(p.frames / fps).toFixed(2)}s` },
    { k: "degrees / frame", v: (p.sweep_deg / Math.max(p.frames, 1)).toFixed(3) },
    { k: "camera height", v: span ? `${span.min.toFixed(2)} … ${span.max.toFixed(2)}` : "—",
      from: "front-on at the aim height" },
    { k: "canvas", v: `${p.width} × ${p.height}`, from: "from the photo" },
    { k: "card height", v: p.card_height.toFixed(4), from: "world units — fixed until you re-fit" },
    { k: "lens", v: lens ? `${lens.mm.toFixed(0)} mm equivalent (${lensLook(lens.mm)})` : "—" },
  ];
  // The floor collision is a reason NOT TO RENDER.
  if (span && span.min < 0.02) {
    rows.push({ k: "floor", problem: true,
      v: `The camera sits at z ${span.min.toFixed(2)}, at or below the ground `
       + `plane — raise the aim height.` });
  }
  const renderBlockers = rows.filter(r => r && r.problem).length + (S.state.source ? 0 : 1);
  const stale = controlStale();
  if (stale.length) {
    rows.push({ k: "", live: true, note: true,
      v: `The control video is out of date (${stale.slice(0, 3).join(", ")}${stale.length > 3 ? "…" : ""}). `
       + `Render it again — Generate does it for you if you forget.` });
  }
  renderStatusFolded("stShot", "Shot", rows, "shot detail");
  gate("btnOrbit", renderBlockers, S.state.control_video ? "Render control video again" : "Render control video");
  return renderBlockers;
}

function statusEngine() {
  const e = engineNow();
  if (!e) { renderStatus("stEngine", "Engine", [{ k: "", v: "loading…" }]); return 0; }
  const rows = [];
  if (!e.pose_exact) {
    rows.push({ k: "", note: true, problem: true, v: "Not pose-exact — retime before training." });
  }
  rows.push({ k: "pose-exact", v: e.pose_exact ? "yes" : "no — retime before training" });
  rows.push({ k: "endpoint", v: e.endpoint });
  rows.push({ k: "note", v: e.note });
  rows.push({ k: "output fps", v: e.output_fps ? `${e.output_fps} (shot fps set to match)` : "same as the shot" });
  renderStatusFolded("stEngine", "Engine", rows, "engine details");
  // Rendering at the rate the engine returns is what makes the frame count
  // equal the pose count with no retime. It just has to be visible.
  if (e.output_fps && +$("oFps").value !== e.output_fps) {
    $("oFps").value = e.output_fps;
    logUi(`Shot fps set to ${e.output_fps} to match ${e.label}.`);
    fx.toast(`Shot fps set to ${e.output_fps} to match ${e.label}.`, "info");
    pushOrbit();
    onDirty();
  }
  return 0;
}

/** Is the softened clip on disk built from the softening in the rail? */
function softStale() {
  const st = S.state;
  if (!softOn() || !st.control_video) return false;
  if (!st.control_soft) return true;
  return +st.control_soft_factor !== num("gSoftDown", 1)
      || +st.control_soft_blur !== num("gSoftBlur", 0);
}

function statusGenerate() {
  const e = engineNow();
  const p = orbitParams();
  const st = S.state;
  const res = $("uRes").value;
  const rows = [];
  let probs = 0;

  const soft = softOn();
  const f = Math.max(1, num("gSoftDown", 1));
  const sw = Math.round(p.width / f / 2) * 2, sh = Math.round(p.height / f / 2) * 2;

  if (!st.control_video) {
    rows.push({ k: "control", problem: true, v: "No control video yet — render it in step 3.",
      fix: { label: "Open Shot", onClick: () => openStep("stepShot") } });
    probs++;
  }
  const stale = controlStale();
  if (stale.length) {
    rows.push({ k: "", live: true, note: true,
      v: `Control video is out of date — Generate re-renders it first.` });
  } else if (soft && softStale()) {
    rows.push({ k: "", live: true, note: true,
      v: `Softening changed — Generate rebuilds the softened clip first.` });
  }
  if (soft && Math.min(sw, sh) < 256) {
    rows.push({ k: "size", problem: true,
      v: `${sw}×${sh} is under fal's 256px minimum — lower the downsample.` }); probs++;
  }
  if ($("oClay").value !== "off" && $("gFirst").value === "control") {
    rows.push({ k: "reference", problem: true,
      v: "Clay is on, so control frame 0 is grey — use the photo as reference.",
      fix: { label: "Use the photo", onClick: e2 => {
        $("gFirst").value = "source"; logUi("reference → the source photo");
        fx.toast("Reference set to the source photo", "ok");
        repaint(); onDirty();
      } } }); probs++;
  }
  if (e && e.id === "wan22vace") {
    const [lo, hi] = [81, 241];
    if (p.frames < lo || p.frames > hi) {
      rows.push({ k: "frame range", problem: true,
        v: `VACE takes ${lo}–${hi} frames; this shot is ${p.frames} — `
         + `${(lo / fpsNow()).toFixed(1)}–${(hi / fpsNow()).toFixed(1)}s at ${fpsNow()} fps.` }); probs++;
    }
  }
  if (e && e.id === "ltx") {
    const cap = MAX_SECONDS[res] || 6;
    const secs = p.frames / fpsNow();
    if (secs > cap) {
      rows.push({ k: "length", problem: true,
        v: `${secs.toFixed(2)}s is over LTX's ${cap}s at ${res} — shorten the shot.` });
      probs++;
    }
  }
  if (!(S.health.fal_key)) {
    rows.push({ k: "FAL_KEY", problem: true,
      v: "Not set on the server — set it and restart the server." });
    probs++;
  }
  if (!($("gPrompt").value || "").trim()) {
    rows.push({ k: "", note: true, problem: true,
      v: "No subject description — write one, or let Auto-describe do it.",
      fix: { label: "Auto-describe", onClick: () => $("btnDescribe").click() } });
    probs++;
  }

  rows.push({ k: "engine", v: e ? e.label : "—" });
  rows.push({ k: "frames", v: `${p.frames}`, from: "= poses" });
  rows.push({ k: "resolution", v: res || "—" });
  rows.push({ k: "control video", v: soft ? `control_soft.mp4 — ${sw}×${sh}, blur ${num("gSoftBlur", 0)}px`
                                          : `control.mp4 — ${p.width}×${p.height}, sharp` });
  rows.push({ k: "reference", v: $("gFirst").value === "control"
      ? "control_frame0.png" : (st.source || "the source photo") });
  if (e && !e.pose_exact) rows.push({ k: "after it lands", v: "retime to the pose count in step 5" });
  rows.push({ k: "prompt", v: `${($("gPrompt").value || "").trim().length} characters`,
    from: "the full string is shown in the approval window" });

  renderStatusFolded("stGen", "What will be sent", rows, "send details");
  gate("btnGen", probs, "Generate — review before it is sent");
  $("gFirstHint").textContent = $("gFirst").value === "source"
    ? "If the photo's framing differs from frame 0, the clip's first frames can jump." : "";
  return probs;
}

function statusReview() {
  const st = S.state;
  const poses = (st.orbit || {}).frames || 0;
  const n = st.ai_frames || 0;
  const rows = [];
  let probs = 0;

  if (!st.ai_video) {
    rows.push({ k: "", note: true, problem: true, v: "No clip yet — generate one in step 4, or upload your own.",
      fix: { label: "Upload your own video", onClick: () => $("aiFile").click() } });
    probs++;
  }
  if (st.ai_video && n && poses && n !== poses) {
    rows.push({ k: "", note: true, problem: true, v: `${n} frames, ${poses} poses — retime first`,
      fix: { label: "Retime", onClick: () => retime() } });
    probs++;
  }
  const mattes = st.mattes || {};
  const pick = $("uAlpha").value;
  const m = mattes[pick];
  if (!m && st.ai_video && pick && pick !== "authored" && pick !== "distance") {
    rows.push({ k: "", note: true, problem: true, v: "Matte not cut yet — needed before training",
      fix: { label: "Run the matte", onClick: () => $("btnAiRmbg").click() } });
    probs++;
  }
  if (pick === "authored") {
    rows.push({ k: "", note: true, problem: true,
      v: "Authored coverage is for looking at, not for training — pick a matte model." });
    probs++;
  }

  rows.push({ k: "clip", v: st.ai_video || "—" });
  rows.push({ k: "engine", v: st.ai_engine === "upload" ? "your own video (uploaded)"
                                                     : (st.fal_engine || st.ai_engine || "—") });
  rows.push({ k: "frames / poses", v: st.ai_video ? `${n || "?"} / ${poses}` : "—" });
  rows.push({ k: "retimed", v: st.ai_retimed ? "yes" : "no" });
  rows.push({ k: "retime", v: "last resort — see Advanced" });
  rows.push({ k: "alpha", v: $("uAlpha").selectedOptions[0]?.textContent || pick });
  if (m) {
    rows.push({ k: "coverage", v: `${(100 * m.coverage).toFixed(1)}%` });
    rows.push({ k: "ambiguous edge", v: `${(100 * m.ambiguous).toFixed(2)}%` });
  }
  rows.push({ k: "approved", v: st.approved ? "yes" : "no" });

  renderStatusFolded("stReview", "Review", rows, "review details");
  gate("btnApprove", probs, st.approved ? "Approve again" : "Approve this clip");

  // The clip plays inline in this step. It should not be behind a button.
  const box = $("aiClipBox");
  const want = st.ai_video ? bust(`/projects/${S.project}/${st.ai_video}`, st.ai_frames) : "";
  const nobg = st.ai_nobg ? bust(`/projects/${S.project}/${st.ai_nobg}`, st.ai_nobg) : "";
  if (box.dataset.src !== want + "|" + nobg) {
    box.dataset.src = want + "|" + nobg;
    box.innerHTML = want
      ? `<video src="${esc(want)}" controls loop muted playsinline></video>`
        + `<div class="p">${esc(st.ai_video)}</div>`
        + (nobg ? `<video src="${esc(nobg)}" controls loop muted playsinline></video>`
                  + `<div class="p">${esc(st.ai_nobg)} — matte preview</div>` : "")
      : `<div class="p">Nothing has come back from fal for this project yet.</div>`;
  }
  $("mattePromptBox").classList.toggle("hidden", $("uAlpha").value !== "sam2_text");
  return probs;
}

function statusTrain() {
  const st = S.state;
  const rows = [];
  let probs = 0;
  if (!st.approved) {
    rows.push({ k: "", note: true, problem: true, v: "Approve the clip in step 5 first.",
      fix: { label: "Open Review", onClick: () => openStep("stepReview") } });
    probs++;
  }
  const pick = $("uAlpha").value;
  if (st.approved && pick && pick !== "authored" && pick !== "distance" && !(st.mattes || {})[pick]) {
    rows.push({ k: "", note: true, problem: true, v: "Matte not cut yet — run it in Review first.",
      fix: { label: "Open Review", onClick: () => openStep("stepReview") } });
    probs++;
  }
  if (pick === "authored") {
    rows.push({ k: "", note: true, problem: true, v: "Authored coverage cannot be trained on — pick a matte in Review." });
    probs++;
  }
  if (!S.health.brush) {
    rows.push({ k: "", note: true, problem: true,
      v: `Brush not found — set "brush" in config.json.` });
    probs++;
  }
  if (num("sSteps", 50000) > 15000) {
    rows.push({ k: "", live: true, note: true,
      v: `${num("sSteps", 50000).toLocaleString()} steps is past the ~15,000 quality peak.`,
      fix: { label: "Use 15,000", onClick: () => {
        $("sSteps").value = "15000";
        $("sSteps").dispatchEvent(new Event("input", { bubbles: true }));
        fx.flash($("sSteps").closest(".f"));
        logUi("training steps → 15,000"); repaint(); onDirty();
      } } });
  }
  rows.push({ k: "dataset", v: "dataset/ — frames, mattes, COLMAP poses" });
  rows.push({ k: "alpha", v: $("uAlpha").selectedOptions[0]?.textContent || "", from: "from step 5" });
  rows.push({ k: "init cloud", v: "visual hull carved from the mattes" });
  rows.push({ k: "output", v: `${st._dir || ""}\\splat_out` });
  rows.push({ k: "checkpoints", v: ckList.length ? `${ckList.length} on disk` : "none yet" });
  rows.push({ k: "steps", v: `${num("sSteps", 50000).toLocaleString()}` });
  renderStatusFolded("stTrain", "Train", rows, "training setup");
  gate("btnSplat", probs, "Build splat");
  return probs;
}

async function refreshReport() {
  let j;
  try { j = await api(`/api/splat_report?project=${encodeURIComponent(S.project)}`,
                      undefined, { quiet: true }); }
  catch (e) { return; }
  const runs = j.runs || [];
  renderStatus("stReport", "Training runs", runs.length
    ? runs.slice(-6).map(r => ({
        k: r.when ? new Date(r.when * 1000).toLocaleString() : (r.name || "run"),
        v: [r.steps ? `${r.steps} steps` : "", r.splats ? `${(+r.splats).toLocaleString()} splats` : "",
            r.anisotropy ? `aniso ${(+r.anisotropy).toFixed(1)}` : "", r.detail || ""]
             .filter(Boolean).join(" · ") || JSON.stringify(r).slice(0, 120) }))
    : [{ k: "", v: "no runs recorded for this project" }]);
}

// ================================================================ repaint ===
let repaintTimer = null;
export function repaint() {
  clearTimeout(repaintTimer);
  repaintTimer = setTimeout(repaintNow, 30);
}

/** Put the source photo on the card and in the drop zone. Cache-busted by
 *  source_time: a new photo can land on the SAME filename. */
function syncSourceImage() {
  const st = S.state;
  const url = st.source
    ? bust(`/projects/${S.project}/${st.source}`, st.source_time || st.orbit_time || 1)
    : null;
  if (url === shownSource) return;
  shownSource = url;
  if (url) orbit.setImage(url); else orbit.clearImage();
  const drop = $("drop");
  const img = drop.querySelector(".dropimg");
  img.innerHTML = url ? `<img src="${esc(url)}" alt="the source photo">` : "";
  drop.classList.toggle("has", !!url);
  const msg = drop.querySelector(".dropmsg");
  if (!drop.classList.contains("busy") && !drop.classList.contains("bad")) {
    msg.textContent = url ? "Click or drop to replace the photo"
                          : "Drop a photo here, or click to choose";
  }
}

// One place that answers "is softening on".
function softOn() { const e = $("gSoftOn"); return !!(e && e.checked); }

function softSync() {
  const on = softOn();
  $("softBox").classList.toggle("hidden", !on);
  const f = Math.max(1, num("gSoftDown", 1));
  const blur = num("gSoftBlur", 0);
  $("gSoftBlurV").textContent = String(blur);
  $("gSoftDownV").textContent = f > 1 ? `${f}×` : "1× off";
  const st = S.state || {};
  const p = orbitParams();
  const w = Math.round(p.width / f / 2) * 2;
  const hh = Math.round(p.height / f / 2) * 2;
  const rows = [];
  if (!on) {
    renderStatus("stSoft", "Softened clip", [{ k: "state", v: "off — the sharp render is sent" }]);
    return;
  }
  if (Math.min(w, hh) < 256) {
    rows.push({ k: "too small", problem: true,
      v: `${w}×${hh} — fal refuses under 256px. Lower the downsample and use blur instead.` });
  }
  if (f <= 1 && blur <= 0) {
    rows.push({ k: "", note: true, live: true,
      v: "Downsample 1× and blur 0 soften nothing — raise one, or untick the box." });
  }
  rows.push({ k: "size", v: `${p.width}×${p.height} → ${w}×${hh}${blur ? `, blur ${blur}px` : ""}` });
  rows.push({ k: "clip", v: st.control_soft
      ? (softStale() ? `${st.control_soft} — rebuilt on the next render or Generate`
                     : `${st.control_soft} — this is what gets sent`)
      : "made when you render the control video" });
  renderStatus("stSoft", "Softened clip", rows);
}

export function repaintNow() {
  try {
    syncSourceImage();
    syncCanvas();
    seedCard();
    statusSource(); statusCloud(); statusShot(); softSync();
    statusEngine(); statusGenerate(); statusReview(); statusTrain();
    const l = lensMm();
    $("fovMmHint").textContent = l
      ? `≈ ${l.mm.toFixed(0)} mm full-frame (${lensLook(l.mm)}) · `
        + `${l.h.toFixed(1)}° horizontal, ${l.d.toFixed(1)}° diagonal` : "";
    $("oDurVal").textContent = `${num("oDur", 5)}s`;
    $("cCutVal").textContent = cutoffFromSlider(num("cCut", 0)) > 0
      ? cutoffFromSlider(num("cCut", 0)).toFixed(2) : "off";
    $("oDepthStrVal").textContent = reliefNow().toFixed(2);
    $("oDepthStrHint").textContent = depthMetricNow()
      ? "world units of depth across the subject" : "world units of relief across the photo (relative depth)";
    for (const o of $("oDepthModel").options) {
      if (o.value.startsWith("da3")) o.disabled = !!(S.health && S.health.da3 === false);
    }
    $("cPruneVal").textContent = num("cPrune", 0) > 0 ? num("cPrune", 0) + "%" : "off";
    $("vStepsVal").textContent = num("vSteps", 30);
    $("vGuideVal").textContent = (num("vGuide", 50) / 10).toFixed(1);
    $("cropCtl").classList.toggle("hidden", !$("cCropOn").checked);
    $("rerunOpts").classList.toggle("hidden", !$("sRerun").checked);
    fx.syncAllRanges($("railAuthor"));
  } catch (e) { console.error("repaint", e); }
}

// ================================================================= engines ==
/** Sort resolutions by actual size: the lists mix 480p, 768P, 2K and 4K. */
const resRank = r => {
  const s = String(r || "").toLowerCase();
  if (s === "auto") return 1e9;
  const k = s.match(/^(\d+(?:\.\d+)?)k$/);
  if (k) return { 1: 1080, 2: 1440, 4: 2160, 8: 4320 }[+k[1]] || +k[1] * 1000;
  return parseInt(s, 10) || 1e8;
};
export const lowestRes = opts => [...opts].sort((a, b) => resRank(a) - resRank(b))[0] || "";

/** reset = the engine was switched: go to that engine's LOWEST resolution. */
function engineOpts(reset = false) {
  const e = engineNow();
  const id = e ? e.id : "ltx";
  $("ltxOpts").classList.toggle("hidden", id !== "ltx");
  $("vaceOpts").classList.toggle("hidden", id !== "wan22vace");
  $("wanOpts").classList.toggle("hidden", id !== "wan");
  $("mmOpts").classList.toggle("hidden", id !== "minimax");
  const opts = id === "wan" ? (S.engineExtras.wan_resolutions || [])
             : id === "minimax" ? (S.engineExtras.mm_resolutions || [])
             : id === "wan22vace" ? VACE_RES : LTX_RES;
  const keep = $("uRes").value;
  $("uRes").innerHTML = opts.map(o => `<option value="${esc(o)}">${esc(o)}</option>`).join("");
  $("uRes").value = (!reset && opts.includes(keep)) ? keep : lowestRes(opts);
  const cap = (S.refs.caps || {})[id] ?? 0;
  $("refsBox").classList.toggle("hidden", cap <= 0);
  $("refsCap").textContent = cap > 0
    ? `${id} accepts ${cap} reference images including the main one, so up to `
      + `${cap - 1} extras are sent.` : "";
}

/** A resolution saved with a project comes back when the project opens. */
export function restoreRes(saved) {
  if (!saved || saved.engine !== $("gEngine").value) return;
  if ([...$("uRes").options].some(o => o.value === saved.value)) $("uRes").value = saved.value;
}

/** The GenReq body. */
export function genBody(extra = {}) {
  const p = orbitParams();
  const id = $("gEngine").value;
  const seed = ($("uSeed").value || "").trim();
  const b = {
    project: S.project,
    prompt: ($("gPrompt").value || "").trim(),
    engine: id,
    ref_downsample: softOn() ? num("gSoftDown", 1) : 1,
    ref_blur: softOn() ? num("gSoftBlur", 0) : 0,
    reference_mode: $("gFirst").value,
    intensity: $("gInt").value,
    detail_refine: $("gRef").checked,
    resolution: id === "ltx" ? $("uRes").value : "720p",
    vace_resolution: id === "wan22vace" ? $("uRes").value : "720p",
    vace_steps: num("vSteps", 30),
    vace_guidance: num("vGuide", 50) / 10,
    vace_sampler: $("vSampler").value,
    vace_negative: $("uNeg").value || "",
    vace_seed: seed ? +seed : null,
    vace_control: "control",
    vace_reference: $("gFirst").value === "control" ? "frame0" : "source",
    vace_first_frame: $("gFirst").value === "control" ? "frame0" : "source",
    wan_resolution: id === "wan" ? $("uRes").value : "720p",
    wan_aspect: aspectOption(p.width, p.height),
    wan_duration: 0,
    wan_smart_duration: false,
    wan_audio: false,
    wan_prompt_expansion: $("wExpand").checked,
    wan_thinking: $("wThink").checked,
    wan_seed: seed ? +seed : null,
    mm_resolution: id === "minimax" ? $("uRes").value : "768P",
    mm_aspect: aspectOption(p.width, p.height),
    mm_duration: 0,
    mm_smart_duration: false,
    mm_expansion: $("mmExpand").value,
    mm_seed: seed ? +seed : null,
  };
  return Object.assign(b, extra);
}

// ================================================================ actions ===
const imageSize = url => new Promise(res => {
  const im = new Image();
  im.onload = () => res({ w: im.naturalWidth, h: im.naturalHeight });
  im.onerror = () => res(null);
  im.src = url;
});

/** A photo arrives: tick on the drop zone -> Cloud opens -> lens -> lift. */
async function upload(fileObj) {
  const drop = $("drop");
  if (!/^image\//.test(fileObj.type || "") && !/\.(png|jpe?g|webp)$/i.test(fileObj.name || "")) {
    fx.dropFail(drop, "Not an image — use a JPG, PNG or WebP");
    fx.toast(`${fileObj.name} is not an image — use a JPG, PNG or WebP`, "err");
    return;
  }
  const preview = URL.createObjectURL(fileObj);
  const dims = await imageSize(preview);
  fx.dropBusy(drop, preview, fileObj.name);
  const ext = (fileObj.name.split(".").pop() || "png").toLowerCase();
  logUi(`uploading ${fileObj.name} (${(fileObj.size / 1e6).toFixed(2)} MB) as the source photo`);
  let j;
  try {
    j = await apiRaw(
      `/api/upload?project=${encodeURIComponent(S.project)}&ext=${encodeURIComponent(ext)}`,
      fileObj);
  } catch (e) {
    fx.dropFail(drop, `Upload failed — ${e.message}`);
    shownSource = null;
    syncSourceImage();
    URL.revokeObjectURL(preview);
    return;
  }
  logUi(`source saved as ${S.projects_dir}\\${j.project}\\${j.source}`
      + (j.unchanged ? " — identical to the existing image, all derived work kept" : ""));
  shownSource = null;                     // force the served image in
  await fireStateChange("upload");
  URL.revokeObjectURL(preview);
  if (!j.unchanged) {
    orbit.clearCloud(); cloudPoints = 0; liftedWith = null;
    autoFlow = "New photo — measuring the lens, then lifting the points…";
  }
  await fx.dropOk(drop, j.unchanged ? "Same photo — your work is kept" : "Image loaded",
                  dims ? `${dims.w} × ${dims.h} · ${fileObj.name}` : fileObj.name);
  // Hold the tick long enough to be read, then Cloud opens by itself and the
  // lens and the lift run without a press.
  await fx.wait(900);
  openStep("stepCloud");
  await fx.wait(380);
  try {
    if (!j.unchanged) {
      await measureLens({ chain: true, fresh: true });
      autoFlow = "";
      await requestLift("new photo", { final: "Photo ready" });
    } else if (!cloudPoints) {
      await requestLift("no points on screen yet");
    }
  } finally {
    autoFlow = "";
    repaint();
  }
}

/** MoGe-2 measures the lens; the result seeds the shot's vertical FOV. */
async function measureLens({ chain = false, fresh = false, btn = null } = {}) {
  const r = await run("measure the lens with MoGe-2", () => api("/api/fov", { name: S.project }), {
    btn, stage: "fov", busy: "Measuring…", quiet: chain, okBtn: "Measured",
    after: async () => {
      await fireStateChange("fov");
      return applyMeasuredFov(fresh);
    },
  }).catch(() => null);
  return !!(r && !r.jobError);
}

/** Put the measured lens into the shot. A lens you set by hand is kept unless
 *  force is true (a new photo, or "Use measured"). Returns a short summary. */
function applyMeasuredFov(force) {
  const v = (S.state.fov || {}).vfov_deg;
  if (v == null) return "";
  const box = $("oFov");
  if (!force && box.dataset.touched === "1") {
    logUi(`MoGe-2 measured ${(+v).toFixed(2)}° — your hand-set ${box.value}° was kept`);
    return `measured ${(+v).toFixed(2)}°, your ${box.value}° kept`;
  }
  box.value = (+v).toFixed(2);
  delete box.dataset.touched;
  box.dispatchEvent(new Event("input", { bubbles: true }));
  fx.flash(box.closest(".f"));
  if (force && $("oFit").checked && !S.state.orbit) {
    fitCard("fitted to the new photo");
    delete $("oCard").dataset.touched;
  }
  logUi(`vertical FOV set to ${(+v).toFixed(2)}° — measured by MoGe-2 from ${S.state.source}`);
  pushOrbit(); onDirty();
  return `${(+v).toFixed(2)}° vertical field of view`;
}

async function liftCloud({ why = "", chain = false, final = "" } = {}) {
  const p = orbitParams();
  const key = liftKey();
  const pts = +$("oPoints").value;
  // A server job already owns the viewport card (a render, a training run):
  // report the lift in the Cloud status and a toast instead of taking it over.
  const own = !(jobActive() && jobActive().kind === "server");
  const said = { "settings changed": "your change", "before rendering": "before the render",
                 "new photo": "the new photo", "Regenerate": "" }[why] ?? why;
  const tok = own ? jobStart({ title: cloudPoints ? "Re-lifting the points" : "Lifting the points",
                               detail: `${depthNameNow()} · ${pts} points across${said ? ` · for ${said}` : ""}`,
                               kind: "client", stage: "lift", compact: !final }) : 0;
  logUi(`lift: ${pts}px wide via ${depthNameNow()}${why ? ` — ${why}` : ""}`);
  const t0 = performance.now();
  let buf;
  try {
    const r = await fetch("/api/cloud", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({
        project: S.project, orbit: p, card_height: p.card_height,
        depth_strength: reliefNow(), depth_model: depthModelNow(), clay: "off", frame: 0,
        depth_cutoff: 0, isolate_prune: num("cPrune", 0) / 100,
        crop_sphere: cropSphere(),
        subject_x: num("sX", 0), subject_y: num("sY", 0), subject_z: num("sZ", 0),
        subject_rot_deg: num("sRot", 0), subject_rot_x: num("sRotX", 0),
        subject_rot_y: num("sRotY", 0),
        points_w: pts, key,
      }),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.error || `${r.status} /api/cloud`);
    }
    buf = await r.arrayBuffer();
  } catch (e) {
    logUi(`lift failed: ${e.message}`, "error");
    lastLiftError = shortErr(e.message);
    repaint();
    if (own) jobEnd({ kind: "fail", title: "Lift failed — the points on screen are unchanged", detail: lastLiftError }, tok);
    else fx.toast(`Lift failed — ${e.message}`, "err");
    throw e;
  }
  lastLiftError = "";
  cloudPoints = orbit.setCloud(buf) || 0;
  liftedWith = key;
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  logUi(`lift: ${cloudPoints.toLocaleString()} points in the viewport in ${secs}s `
      + `(${(buf.byteLength / 1e6).toFixed(2)} MB, kept at ${S.state._dir}\\cloud_view.bin)`);
  if (orbit.cardVisible()) {
    orbit.setCardVisible(false);
    $("vCard").classList.remove("on");
    logUi("the flat card was hidden because a point cloud is loaded — Card in the viewport bar brings it back");
  }
  pushOrbit();
  const lens = (S.state.fov || {}).vfov_deg;
  const detail = `${cloudPoints.toLocaleString()} points · ${secs}s`
    + (final && lens != null ? ` · lens ${(+lens).toFixed(1)}°` : "");
  // A lift that "worked" but kept almost nothing is not a success to celebrate.
  const thin = cloudPoints < 60000;
  const crop = cropSphere();
  const outcome = thin
    ? { kind: "warn", title: `Only ${cloudPoints.toLocaleString()} points kept`,
        detail: crop ? "The isolate sphere holds very little — widen it or move it onto the subject."
                     : "Raise point density or lower the cleanup.", compact: false }
    : { kind: "ok", title: final || (why === "settings changed" ? "Points re-lifted" : "Points lifted"),
        detail, compact: !final, quiet: chain };
  if (own) jobEnd(outcome, tok);
  else if (!chain) fx.toast(`${outcome.title} — ${outcome.detail}`, thin ? "warn" : "ok");
}

/** Restore the last lift after a refresh or a server restart. */
async function loadSavedCloud() {
  const project = S.project;
  if (!S.state.source) return false;
  let r;
  try { r = await fetch(`/api/cloud/saved?project=${encodeURIComponent(project)}`); }
  catch (e) { return false; }
  if (project !== S.project || !r.ok || r.status === 204) return false;
  const key = decodeURIComponent(r.headers.get("X-Lift-Key") || "");
  const buf = await r.arrayBuffer();
  if (project !== S.project) return false;
  cloudPoints = orbit.setCloud(buf) || 0;
  liftedWith = key || null;
  orbit.setCardVisible(false);
  $("vCard").classList.remove("on");
  pushOrbit();
  logUi(`point cloud restored: ${cloudPoints.toLocaleString()} points from `
      + `${S.state._dir}\\cloud_view.bin`);
  fx.toast(`Point cloud restored — ${cloudPoints.toLocaleString()} points`, "ok");
  if (liftedWith !== liftKey()) reliftSoon(900);
  return true;
}

/** Everything that belongs to one project, reset and reloaded. Called by
 *  app.js once the project's saved controls have been applied. */
export async function projectOpened() {
  togglePlay(false);
  orbit.resetForProject();
  orbit.setCardVisible(true);
  $("vCard").classList.add("on");
  shownSource = null;
  cloudPoints = 0; liftedWith = null;
  clearTimeout(reliftTimer); reliftTimer = null;
  filesKey = ""; filesMtimes = null;
  delete $("oFov").dataset.touched;
  spanShotSliders();
  syncSourceImage();
  pushOrbit();
  const had = await loadSavedCloud();
  const busyHere = S.status && S.status.running && S.status.project === S.project;
  if (!had && S.state.source && !busyHere) requestLift("no saved cloud for this project yet");
}

/** Radius and aim height have no fixed range -- scene scale varies 8x between
 *  projects -- so their sliders are spanned from the project's own values. */
function spanShotSliders() {
  const r = Math.max(0.1, num("oRad", 9));
  spanSlider("oRad", 0.1, +Math.max(4 * r, 2).toFixed(1));
  const a = num("oAimZ", 2.3), R = Math.max(1, r * 0.75);
  spanSlider("oAimZ", +(a - R).toFixed(2), +(a + R).toFixed(2));
  spanSlider("oSweep", 10, 360);
  spanSlider("oFov", 10, 120);
}

/** Render the control video. Re-lifts first when the points on screen were
 *  lifted with other settings; softens in the SAME job when softening is on. */
async function renderControl({ btn = null, chain = false } = {}) {
  if (S.state.source && (!cloudPoints || liftedWith !== liftKey())) {
    logUi("settings changed since the last lift — re-lifting before the render");
    if (btn) fx.btnBusy(btn, "Re-lifting…");
    await requestLift("before rendering", { chain: true });
  } else {
    logUi("lift is current — rendering");
  }
  const p = orbitParams();
  const soft = softOn();
  const r = await run("render the control video", () => api("/api/orbit", {
    project: S.project, orbit: p, card_height: p.card_height,
    auto_fit: $("oFit").checked, clay: $("oClay").value, backdrop: "grey",
    depth: true, depth_strength: reliefNow(),
    backface_cull: $("oCull").checked, depth_model: depthModelNow(),
    fps: fpsNow(), points_w: +$("oPoints").value,
    depth_cutoff: cutoffFromSlider(num("cCut", 0)),
    isolate_prune: num("cPrune", 0) / 100,
    crop_sphere: cropSphere(),
    subject_x: num("sX", 0), subject_y: num("sY", 0), subject_z: num("sZ", 0),
    subject_rot_deg: num("sRot", 0), subject_rot_x: num("sRotX", 0),
    subject_rot_y: num("sRotY", 0),
    soften_downsample: soft ? num("gSoftDown", 1) : 1,
    soften_blur: soft ? num("gSoftBlur", 0) : 0,
  }), {
    btn: btn || null, stage: "orbit", busy: "Rendering…", okBtn: "Rendered", quiet: chain,
    ok: soft ? "Control video rendered and softened" : "Control video rendered",
    after: async () => {
      await fireStateChange("orbit");
      const st = S.state;
      return soft && st.control_soft
        ? `${st.control_soft} · ${p.frames} frames`
        : `control.mp4 · ${p.width}×${p.height} · ${p.frames} frames`;
    },
  }).catch(() => null);
  if (!r || r.jobError) { fireStateChange("orbit"); return false; }
  return true;
}

function retime() {
  const st = S.state;
  const poses = (st.orbit || {}).frames || 0;
  const n = st.ai_frames || 0;
  const already = (n && poses && n === poses)
    ? `<p class="problem-block"><span>NOTE: this clip ALREADY has ${n} frames `
      + `for ${poses} poses. Retiming would re-encode it for no gain and cost `
      + `a generation of quality.</span></p>` : "";
  // The widget changed; the SENTENCE is load-bearing and survives verbatim.
  dialog({
    title: `Resample ${st.ai_video} to ${poses} frames?`,
    okLabel: "Retime",
    bodyHtml:
      `<p>This fixes the frame COUNT so every pose addresses an image. It `
      + `assumes the clip sweeps the authored orbit linearly in time — if it `
      + `does not, the splat will be wrong in a way nothing downstream can `
      + `detect.</p>`
      + `<p>The original is kept alongside as *_orig.mp4. Extracted frames and `
      + `every matte are cleared, because they were cut against the old `
      + `timing.</p>` + already,
  }).then(async ok => {
    if (!ok) { logUi("retime cancelled"); return; }
    await run("retime the AI clip", () => api("/api/ai_retime", { project: S.project }), {
      btn: $("btnRetimeAdv"), stage: "retime", busy: "Retiming…",
      after: async () => { await fireStateChange("retime"); return `${S.state.ai_video} · ${poses} frames`; },
    }).catch(() => {});
  });
}

async function generate(btn) {
  const body = genBody();
  logUi(`asking the server what would be sent (/api/generate/plan, ${body.engine})`);
  if (btn) fx.btnBusy(btn, "Preparing the plan…");
  let plan;
  try { plan = await api("/api/generate/plan", body); }
  catch (e) { if (btn) fx.btnDone(btn, false, "Refused"); return; }
  if (btn) fx.btnRestore(btn);
  apvShow(plan, {
    title: `Send to ${plan.settings.label}`,
    subtitle: `${plan.settings.endpoint} · ${plan.settings.frames} frames · `
            + `the plan resolved by the same functions the send uses`,
    goLabel: "Approve & send",
    // runDetached: the window closes when fal ACCEPTS the request. Progress is
    // the viewport's job, not a modal's.
    onApprove: () => runDetached("generate", () => api("/api/generate", body), {
      btn, stage: "generate", busy: "Sending…", busyRunning: "Generating…",
      okBtn: "Clip ready",
      okDetail: () => (S.state.ai_video ? `${S.state.ai_video} · opening Review` : ""),
    }),
  });
}

/** Generate does the chores first: a stale control video is re-rendered, a
 *  stale softened clip is rebuilt, and only then does the approval window
 *  open with what will really be sent. */
async function generateFlow(btn) {
  const stale = controlStale();
  if (stale.length) {
    fx.toast("The control video is out of date — rendering it first", "info");
    logUi(`generate: control.mp4 is out of date (${stale.join(", ")}) — re-rendering first`);
    const ok = await renderControl({ btn, chain: true });
    if (!ok) return;
  } else if (softStale()) {
    logUi("generate: softening changed — rebuilding the softened clip first");
    const r = await run("rebuild the softened clip", () => api("/api/soften", genBody()), {
      btn, stage: "soften", busy: "Softening…", quiet: true,
      after: () => fireStateChange("soften"),
    }).catch(() => null);
    if (!r || r.jobError) return;
    if (r.started === false) await fireStateChange("soften");
  }
  await generate(btn);
}

async function trainSplat(datasetOnly, btn) {
  const body = {
    project: S.project, steps: num("sSteps", 50000), detail: $("sDetail").value,
    sh_degree: num("sSH", 3), matte_source: $("uAlpha").value,
    dataset_only: !!datasetOnly, clean: $("sClean").checked,
    rerun: $("sRerun").checked, rerun_stats_every: num("sRrStats", 50),
    rerun_splats_every: num("sRrSplats", 0),
  };
  if (btn) fx.btnBusy(btn, "Preparing…");
  let plan;
  try { plan = await api("/api/splat/plan", body); }
  catch (e) { if (btn) fx.btnDone(btn, false, "Refused"); return; }
  if (btn) fx.btnRestore(btn);
  const total = num("sSteps", 50000);
  apvShow(plan, {
    title: datasetOnly ? "Build the COLMAP dataset only" : "Build the splat",
    subtitle: `Brush: ${esc(plan.brush)}${plan.brush_found ? "" : " — NOT FOUND"}`,
    goLabel: datasetOnly ? "Build the dataset" : "Approve & build",
    settingsHtml: settingsHtml(plan.settings),
    onApprove: () => runDetached(
      datasetOnly ? "build the dataset" : "build the splat",
      () => api("/api/splat", body), {
        btn, stage: "splat", busy: "Starting…",
        busyRunning: datasetOnly ? "Building…" : "Training…",
        working: datasetOnly ? "Building the dataset" : "Building the splat",
        ok: datasetOnly ? "Dataset built" : "Splat built",
        okBtn: "Built",
        detailFn: datasetOnly ? null : (st => {
          const last = (S.splats.exports || []).slice(-1)[0];
          if (!last || !last.step) return { detail: String(st.msg || "preparing the dataset…").slice(0, 110) };
          return { detail: `Checkpoint at step ${(+last.step).toLocaleString()} of ${total.toLocaleString()}`,
                   pct: Math.min(99, 100 * last.step / Math.max(total, 1)) };
        }),
      }),
  });
}


// ============================================================ checkpoints ===
export async function refreshCheckpoints() {
  let j;
  try { j = await api(`/api/splats?project=${encodeURIComponent(S.project)}`,
                      undefined, { quiet: true }); }
  catch (e) { return; }
  S.splats = j;
  const prev = ckList.map(x => x.name).join(",");
  ckList = j.exports || [];
  const have = ckList.length > 0;
  for (const id of ["ckSel", "ckFollowL", "ckRig", "ckFocus", "ckReload"]) {
    $(id).classList.toggle("hidden", !have);
  }
  // Only a running TRAINING on this project can be stopped from here.
  $("btnAbort").classList.toggle("hidden", !j.training);
  if (!j.training) fx.btnRestore($("btnAbort"));
  if (prev !== ckList.map(x => x.name).join(",")) {
    $("ckSel").innerHTML = ckList.map(x =>
      `<option value="${esc(x.name)}">${esc(x.name)} — step ${x.step}${x.uploaded ? " (uploaded)" : ""}</option>`).join("");
    if ($("ckFollow").checked && ckList.length) $("ckSel").value = ckList[ckList.length - 1].name;
  }
  if (have && $("ckFollow").checked && ckList.length) {
    const latest = ckList[ckList.length - 1].name;
    if ($("ckSel").value !== latest) $("ckSel").value = latest;
  }
  if (splatView && have && $("ckSel").value !== ckLoaded) await loadCheckpoint();
  // The Train status counts these, and it was drawn before they arrived.
  if (prev !== ckList.map(x => x.name).join(",")) repaint();
  await refreshReport();
}

async function loadCheckpoint() {
  const name = $("ckSel").value;
  if (!name) return;
  const url = bust(`/projects/${S.project}/splat_out/${name}`,
                   (ckList.find(x => x.name === name) || {}).mtime);
  logUi(`loading checkpoint ${S.splats.dir}\\${name}`);
  $("ckInfo").textContent = "loading…";
  try {
    await psplat.init($("cvSplat"));
    const n = await psplat.load(url);
    ckLoaded = name;
    $("ckInfo").textContent = `${(+n).toLocaleString()} splats`;
    logUi(`checkpoint loaded: ${(+n).toLocaleString()} splats from ${S.splats.dir}\\${name}`);
  } catch (e) {
    $("ckInfo").textContent = "load failed";
    logUi(`checkpoint failed to load: ${e.message}`, "error");
    fx.toast(`Checkpoint failed to load — ${e.message}`, "err");
  }
}

/** ONE PlayCanvas app at a time. */
export function releaseSplat() {
  try { psplat.destroy(); } catch (e) { /* already gone */ }
  ckLoaded = null;
}

export async function restoreView() {
  if (splatView) await setSplatView(true);
}

async function setSplatView(on) {
  splatView = !!on;
  $("cvSplat").classList.toggle("hidden", !splatView);
  $("cvAuthor").classList.toggle("hidden", splatView);
  $("ckRig").textContent = splatView ? "Rig view" : "Splat view";
  $("ckRig").classList.toggle("on", splatView);
  if (splatView) await loadCheckpoint();
  else releaseSplat();
}

// =========================================================== files panel ===
/** ONE files panel design, with a full path on every card, in both
 *  workspaces. Cards whose file changed since the last look get a green edge. */
export function renderFiles(listId, cards) {
  $(listId).innerHTML = cards.length
    ? cards.map(c => `<div class="fcard${c.fresh ? " fresh" : ""}">
        <div class="role">${esc(c.role)}<span>${esc(c.amount || "")}</span></div>
        <div class="media">${mediaHtml(c.media, c.role)}</div>
        <div class="path">${esc(c.path)}</div>
        ${c.note ? `<div class="note">${esc(c.note)}</div>` : ""}
      </div>`).join("")
    : `<div class="fcard"><div class="none">Nothing produced yet.</div></div>`;
  $(listId).querySelectorAll(".fcard.fresh").forEach(el => fx.flash(el));
}

/** Rebuilt only when a file actually changed: rebuilding on every poll would
 *  restart every preview video about once a second. Each preview is busted by
 *  its OWN modification time, never a project-wide stamp. */
export async function authorFiles(force = false) {
  if (!$("filesAuthor").classList.contains("on")) return;
  let j;
  try { j = await api(`/api/files?project=${encodeURIComponent(S.project)}`, undefined, { quiet: true }); }
  catch (e) { return; }
  const F = j.files || {};
  const st = S.state, dir = j.dir || st._dir || "", P = `/projects/${S.project}/`;
  const cards = [];
  const add = (role, name, note) => {
    if (!name || !F[name]) return;
    cards.push({ role, name, path: `${dir}\\${name.replace(/\//g, "\\")}`,
                 media: bust(P + name, Math.round(F[name].mtime * 1000)),
                 amount: `${(F[name].bytes / 1e6).toFixed(2)} MB`,
                 mtime: F[name].mtime, note });
  };
  add("source photo", st.source, "the original upload");
  add("control frame 0", st.control_video ? "control_frame0.png" : null, "frame 0 of the control render");
  // Only the clip that is SENT is shown. control.mp4 stays on disk underneath
  // a softened clip; it is its source, not a second deliverable.
  if (softOn() && st.control_soft) add("control video", st.control_soft, "softened — this exact file is sent");
  else add("control video", st.control_video, "sharp — sent as rendered");
  add("AI video", st.ai_video, st.ai_retimed ? "retimed to the pose count" : "as fal returned it");
  add("matte preview", st.ai_nobg, "subject on black");
  add("authored alpha", st.authored_alpha, "what the control render drew — diagnostic only");
  if (st.splat && F[`splat_out/${st.splat}`]) {
    const k = `splat_out/${st.splat}`;
    cards.push({ role: "splat", name: k, path: `${dir}\\splat_out\\${st.splat}`, media: "",
                 amount: `${(F[k].bytes / 1e6).toFixed(1)} MB`, mtime: F[k].mtime,
                 note: "the latest checkpoint" });
  }
  const key = cards.map(c => `${c.name}@${c.mtime}`).join("|");
  if (!force && key === filesKey) return;
  const seen = filesMtimes;
  filesMtimes = Object.fromEntries(cards.map(c => [c.name, c.mtime]));
  for (const c of cards) c.fresh = !!(seen && seen[c.name] !== c.mtime);
  filesKey = key;
  renderFiles("filesAuthorList", cards);
}

// ============================================================== the scrub ===
function setFrame(i) {
  const sl = $("fSlider");
  const max = +sl.max;
  const v = Math.max(0, Math.min(max, Math.round(i)));
  sl.value = v;
  fx.syncRange(sl);
  orbit.setShotFrame(v);
  const p = orbitParams();
  const deg = (p.sweep_deg * v / Math.max(p.frames, 1)).toFixed(1);
  $("fNum").textContent = `frame ${v + 1} / ${max + 1} · ${deg}° · ${(v / fpsNow()).toFixed(2)}s`;
  const clay = $("oClay").value !== "off";
  $("shotLabel2").textContent = `camera view · ${p.width}×${p.height}`
    + (clay ? (clayPreview ? " · flat clay, as rendered" : " · colour preview (renders as clay)") : "");
}

function togglePlay(on) {
  const playing = on === undefined ? !playTimer : !!on;
  if (playTimer) { cancelAnimationFrame(playTimer); playTimer = null; }
  const b = $("fPlay");
  b.classList.toggle("on", playing);
  b.innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
  b.setAttribute("aria-label", playing ? "Pause" : "Play");
  b.title = playing ? "pause (space)" : "play the orbit (space)";
  if (!playing) return;
  let last = performance.now(), acc = 0;
  const step = now => {
    acc += (now - last) / 1000 * fpsNow();
    last = now;
    if (acc >= 1) {
      const n = Math.floor(acc); acc -= n;
      const max = +$("fSlider").max;
      let v = +$("fSlider").value + n;
      if (v > max) v = 0;
      setFrame(v);
    }
    playTimer = requestAnimationFrame(step);
  };
  playTimer = requestAnimationFrame(step);
}

/** ← → step a frame, Home/End jump, Space plays. Called by app.js only when
 *  focus is not in a field. */
export function scrubKey(e) {
  const v = +$("fSlider").value;
  if (e.key === "ArrowLeft") { togglePlay(false); setFrame(v - (e.shiftKey ? 10 : 1)); return true; }
  if (e.key === "ArrowRight") { togglePlay(false); setFrame(v + (e.shiftKey ? 10 : 1)); return true; }
  if (e.key === "Home") { setFrame(0); return true; }
  if (e.key === "End") { setFrame(+$("fSlider").max); return true; }
  if (e.key === " ") { togglePlay(); return true; }
  return false;
}

// =================================================================== wire ===
/** Only a REAL edit re-lifts. applyUI replays every control with synthetic
 *  events when a project opens, and those must not start a lift each. */
function onUserChange(ids, fn) {
  for (const id of ids) {
    const el = $(id);
    if (el) el.addEventListener("change", e => { if (e.isTrusted) fn(e); });
  }
}

export function init() {
  // ---- viewport
  orbit.init($("cvAuthor"));
  orbit.onGizmoChange(g => {
    if (g.kind === "crop") {
      $("cCropX").value = g.x.toFixed(3); $("cCropY").value = g.y.toFixed(3);
      $("cCropZ").value = g.z.toFixed(3); $("cCropR").value = g.r.toFixed(3);
      reliftSoon(650);
    } else {
      $("sX").value = g.x.toFixed(3); $("sY").value = g.y.toFixed(3);
      $("sZ").value = g.z.toFixed(3);
      $("sRotX").value = g.rotX.toFixed(1); $("sRotY").value = g.rotY.toFixed(1);
      $("sRot").value = g.rot.toFixed(1);
    }
    repaint(); onDirty();
  });

  // ---- source
  const drop = $("drop");
  drop.onclick = () => { if (!drop.classList.contains("busy")) $("file").click(); };
  drop.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); drop.click(); }
  });
  let dragDepth = 0;
  drop.addEventListener("dragenter", e => {
    e.preventDefault(); dragDepth++;
    drop.classList.add("over");
    drop.querySelector(".dropmsg").textContent = "Release to load this photo";
  });
  drop.addEventListener("dragover", e => { e.preventDefault(); });
  drop.addEventListener("dragleave", () => {
    if (--dragDepth > 0) return;
    drop.classList.remove("over");
    drop.querySelector(".dropmsg").textContent = drop.classList.contains("has")
      ? "Click or drop to replace the photo" : "Drop a photo here, or click to choose";
  });
  drop.addEventListener("drop", async e => {
    e.preventDefault(); dragDepth = 0; drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f) await upload(f);
  });
  $("file").onchange = async e => {
    const f = e.target.files[0];
    e.target.value = "";                      // the same file can be chosen twice
    if (f) await upload(f);
  };
  $("btnFov").onclick = () => measureLens({ btn: $("btnFov"), fresh: false });

  // ---- cloud
  $("btnCloud").onclick = async () => {
    const b = $("btnCloud");
    fx.btnBusy(b, "Lifting…");
    const ok = await requestLift("Regenerate");
    fx.btnDone(b, ok, ok ? "Lifted" : "Lift failed");
  };
  $("btnCropFit").onclick = () => {
    const t = orbit.freeTarget();
    if (!t) return;
    $("cCropX").value = t.x.toFixed(3);
    $("cCropY").value = t.y.toFixed(3);
    $("cCropZ").value = t.z.toFixed(3);
    logUi(`isolate sphere centred on the free camera's aim point `
        + `(${t.x.toFixed(2)}, ${t.y.toFixed(2)}, ${t.z.toFixed(2)})`);
    fx.btnDone($("btnCropFit"), true, "Centred", 900);
    pushOrbit(); reliftSoon(); onDirty();
  };
  $("btnCropGizmo").onclick = () => {
    $("gizTarget").value = "crop";
    setGizmo("crop");
    fx.toast("Drag the sphere in the viewport — the points re-lift when you let go", "info");
  };
  // Lift settings: a real edit re-lifts after a pause, so the viewport shows
  // the cleaned cloud without a button press.
  onUserChange(["cCropOn", "cCropX", "cCropY", "cCropZ", "cCropR", "cPrune", "oPoints"],
               () => reliftSoon());
  // A different depth model is a different cloud: re-lift straight away.
  onUserChange(["oDepthModel"], () => {
    logUi(`depth model → ${depthNameNow()}`
        + (depthMetricNow() ? " (metric)" : " (relative — relief is world units across the photo)"));
    reliftSoon(150);
  });
  // The lift is made from camera 0's rays, so moving that camera changes it.
  onUserChange(["oRad", "oRad_r", "oAimZ", "oAimZ_r", "oCard"], () => reliftSoon(700));

  // ---- shot
  for (const id of ["oSweep", "oRad", "oAimZ", "oFov"]) pairSlider(id);
  for (const id of ["oFov", "oFov_r"]) {
    $(id).addEventListener("input", e => { if (e.isTrusted) $("oFov").dataset.touched = "1"; });
  }
  $("oPath").addEventListener("change", () => {
    const v = $("oPath").value;
    $("ringOpts").classList.toggle("hidden", v !== "ring-rock");
    $("helixOpts").classList.toggle("hidden", v !== "helix");
    $("cyclesLabel").textContent = v === "helix" ? "Up-down cycles" : "Sweep cycles";
    logUi(`vertical spread → ${$("oPath").selectedOptions[0].textContent}`);
  });
  $("oClay").addEventListener("change", () => {
    pushOrbit();
    logUi(`look → ${$("oClay").selectedOptions[0].textContent} (the camera view shows it now)`);
  });
  $("btnOrbit").onclick = () => renderControl({ btn: $("btnOrbit") });

  // ---- generate
  $("gEngine").addEventListener("change", e => {
    logUi(`engine → ${$("gEngine").selectedOptions[0]?.textContent || ""}`);
    engineOpts(true);
    if (e.isTrusted) fx.toast(`Resolution set to ${$("uRes").value}, the lowest ${engineNow()?.label || "this engine"} offers`, "info");
    repaint();
  });
  $("gFirst").addEventListener("change", () => repaint());
  $("btnDescribe").onclick = () => run("auto-describe", () => api("/api/describe", {
      project: S.project, model: $("gVision").value,
      extra: $("gPrompt").dataset.auto === "1" ? "" : ($("gPrompt").value || "").trim(),
    }), { stage: "describe", busy: "Describing…",
          after: async () => { await fireStateChange("describe"); return `${(S.state.prompt_auto || "").length} characters`; } })
    .catch(() => {});
  $("gPrompt").addEventListener("input", e => {
    if (e.isTrusted) $("gPrompt").dataset.auto = "0";
  });
  $("gSoftOn").addEventListener("change", e => {
    softSync(); repaint();
    if (e.isTrusted) {
      logUi(`softening ${softOn() ? "on" : "off"}`);
      if ($("softBox") && softOn()) fx.expand($("softBox"));
    }
  });
  ["gSoftDown", "gSoftBlur"].forEach(id => $(id).addEventListener("input", () => { softSync(); repaint(); }));

  $("refsAdd").onclick = () => $("refsFile").click();
  $("refsFile").onchange = async e => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    const b = $("refsAdd");
    fx.btnBusy(b, "Adding…");
    try {
      const ext = (f.name.split(".").pop() || "png").toLowerCase();
      const j = await apiRaw(`/api/refs/add?project=${encodeURIComponent(S.project)}`
                           + `&ext=${encodeURIComponent(ext)}`, f);
      S.refs.refs = j.refs;
      logUi(`added the extra reference ${S.state._dir}\\refs\\${j.added}`);
      fx.btnDone(b, true, "Added");
      renderRefs();
    } catch (err) { fx.btnDone(b, false, "Not added"); }
  };
  $("btnGen").onclick = () => generateFlow($("btnGen")).catch(e => logUi(e.message, "error"));

  // ---- review
  $("uAlpha").addEventListener("change", () => {
    logUi(`alpha → ${$("uAlpha").selectedOptions[0]?.textContent || ""}`);
    repaint();
  });
  $("btnAiRmbg").onclick = () => run(`run the ${$("uAlpha").value} matte`, () => api("/api/ai_rmbg", {
      project: S.project, model: $("uAlpha").value,
      prompt: ($("aiMattePrompt").value || "").trim(),
    }), { btn: $("btnAiRmbg"), stage: "matte", busy: "Cutting the matte…", okBtn: "Matte cut",
          after: async () => { await fireStateChange("matte");
            const m = (S.state.mattes || {})[$("uAlpha").value];
            return m ? `coverage ${(100 * m.coverage).toFixed(1)}%` : ""; } })
    .catch(() => {});
  $("btnAuthAlpha").onclick = () => run("render the authored alpha", () =>
      api("/api/authored_alpha", { project: S.project, model: "birefnet" }),
    { stage: "matte", busy: "Rendering…", working: "Rendering the authored alpha",
      ok: "Authored alpha rendered", after: () => fireStateChange("authored_alpha") })
    .catch(() => {});
  $("btnAiUp").onclick = () => $("aiFile").click();
  $("aiFile").onchange = async e => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    const b = $("btnAiUp");
    const ext = (f.name.includes(".") ? f.name.split(".").pop() : "mp4").toLowerCase();
    fx.btnBusy(b, "Uploading…");
    logUi(`uploading ${f.name} (${(f.size / 1e6).toFixed(2)} MB) as the AI video`
        + (ext !== "mp4" ? ` — the server converts .${ext} to mp4` : ""));
    try {
      const j = await apiRaw(`/api/ai_upload?project=${encodeURIComponent(S.project)}&ext=${encodeURIComponent(ext)}`, f);
      logUi(`AI video is now ${S.state._dir}\\${j.ai_video} — ${j.frames} frames at ${j.fps} fps `
          + `against ${j.poses} poses (${j.matches ? "matches" : "DOES NOT MATCH"})`,
          j.matches ? "info" : "warn");
      fx.btnDone(b, true, "Uploaded");
      fx.toast(j.matches ? `${j.ai_video} — ${j.frames} frames, matches the poses`
                         : `${j.ai_video} — ${j.frames} frames against ${j.poses} poses: retime it`,
               j.matches ? "ok" : "warn");
      await fireStateChange("ai_upload");
    } catch (err) { fx.btnDone(b, false, "Upload failed"); }
  };
  $("btnRetimeAdv").onclick = retime;
  // Approve -> wait until the state really says approved -> tick -> Train.
  $("btnApprove").onclick = async () => {
    const b = $("btnApprove");
    fx.btnBusy(b, "Approving…");
    logUi("approve the clip — sent");
    try {
      await api("/api/approve", { name: S.project });
      await fireStateChange("approve");
      if (!S.state.approved) throw new Error("the approval did not land in the project state");
      logUi(`approved ${S.state.ai_video} — the splat step may run`);
      fx.btnDone(b, true, "Approved");
      flashOutcome({ kind: "ok", title: "Clip approved", detail: `${S.state.ai_video} · opening Train` });
      await fx.wait(550);
      if (authorVisible()) openStep("stepTrain");
    } catch (err) {
      logUi(`approve failed: ${err.message}`, "error");
      fx.btnDone(b, false, "Not approved");
      flashOutcome({ kind: "fail", title: "Approval failed", detail: shortErr(err.message) });
    }
  };
  $("btnRetry").onclick = async () => {
    await run("reset the review stage", () => api("/api/reset_stage", { name: S.project }),
              { wait: false, busy: "Resetting…", okBtn: "Reset" }).catch(() => null);
    await fireStateChange("reset");
    fx.toast("Review reset — generate a new clip in step 4", "ok");
    openStep("stepGen");
  };

  // ---- train
  $("btnSplat").onclick = () => trainSplat(false, $("btnSplat")).catch(e => logUi(e.message, "error"));
  $("btnSplatUp").onclick = () => $("splatFile").click();
  $("splatFile").onchange = async e => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    const b = $("btnSplatUp");
    fx.btnBusy(b, "Uploading…");
    try {
      const j = await apiRaw(`/api/splat_upload?project=${encodeURIComponent(S.project)}`, f);
      logUi(`uploaded ${S.state._dir}\\splat_out\\${j.name} — ${(+j.splats).toLocaleString()} splats, ${j.mb} MB`);
      fx.btnDone(b, true, "Uploaded");
      fx.toast(`${j.name} — ${(+j.splats).toLocaleString()} splats`, "ok");
      await fireStateChange("splat_upload");
    } catch (err) { fx.btnDone(b, false, "Upload failed"); }
  };
  $("btnRerunOpen").onclick = async () => {
    try {
      const j = await api("/api/rerun/open", {});
      $("rerunHint").textContent = j.message;
      logUi(`rerun: ${j.message}`);
      fx.toast(j.message, "info");
    } catch (err) { /* reported by api() */ }
  };
  $("btnAbort").onclick = async () => {
    const b = $("btnAbort");
    fx.btnBusy(b, "Stopping…");
    try {
      await api("/api/splat_abort", {});
      logUi("stop requested — Brush is being terminated, checkpoints are kept", "warn");
      fx.toast("Stopping Brush — checkpoints so far are kept", "warn");
    } catch (err) { fx.btnDone(b, false, "Could not stop"); }
  };

  // ---- viewport bar
  $("fSplit").onclick = () => {
    const on = !$("fSplit").classList.contains("on");
    $("fSplit").classList.toggle("on", on);
    orbit.setSplit(on);
    $("shotLabel2").style.visibility = on ? "" : "hidden";
    logUi(`split view ${on ? "on" : "off"}`);
  };
  $("fSlider").addEventListener("input", () => { togglePlay(false); setFrame(+$("fSlider").value); });
  $("fPrev").onclick = () => { togglePlay(false); setFrame(+$("fSlider").value - 1); };
  $("fNext").onclick = () => { togglePlay(false); setFrame(+$("fSlider").value + 1); };
  $("fPlay").onclick = () => togglePlay();

  $("gizBtn").onclick = () => {
    const order = { off: "move", move: "rotate", rotate: "off", crop: "off" };
    setGizmo($("gizTarget").value === "crop" && gizmoState === "off"
             ? "crop" : order[gizmoState] || "off");
  };
  $("gizTarget").addEventListener("change", () => {
    setGizmo($("gizTarget").value === "crop" ? "crop" : "move");
  });

  $("ckSel").addEventListener("change", () => { if (splatView) loadCheckpoint(); });
  $("ckReload").onclick = () => loadCheckpoint();
  $("ckFocus").onclick = () => psplat.focus(num("oAimZ", 2.3), num("oRad", 9) * 0.6);
  $("ckRig").textContent = "Splat view";
  $("ckRig").onclick = () => setSplatView(!splatView);
  $("vClay").onclick = () => {
    clayPreview = !clayPreview;
    pushOrbit();
    logUi(`camera view preview → ${clayPreview ? "flat clay, as the control video renders" : "photo colour (the render stays clay)"}`);
  };
  $("vCard").onclick = () => {
    const on = !$("vCard").classList.contains("on");
    $("vCard").classList.toggle("on", on);
    orbit.setCardVisible(on);
    logUi(`flat card ${on ? "shown" : "hidden"}`);
  };
  $("fPrevPanel").onclick = () => {
    const on = !$("filesAuthor").classList.contains("on");
    $("filesAuthor").classList.toggle("on", on);
    $("fPrevPanel").classList.toggle("on", on);
    document.body.classList.toggle("files-open", on);
    if (on) { filesMtimes = null; authorFiles(true); }
    logUi(`files panel ${on ? "opened" : "closed"}`);
  };
  $("filesAuthorClose").onclick = () => {
    $("filesAuthor").classList.remove("on");
    $("fPrevPanel").classList.remove("on");
    document.body.classList.remove("files-open");
  };

  $("oCard").addEventListener("input", e => {
    if (e.isTrusted) $("oCard").dataset.touched = "1";
  });
  $("btnFitCard").onclick = () => {
    const h = fitCard();
    $("oCard").dataset.touched = "1";
    fx.flash($("oCard").closest(".f"));
    fx.btnDone($("btnFitCard"), h > 0, h > 0 ? `Fitted at ${h.toFixed(3)}` : "Nothing to fit", 1300);
    pushOrbit(); reliftSoon(); onDirty();
  };

  // Any control in the rail that moves a camera pushes to the VIEWPORT.
  for (const id of ["oDur", "oFps", "oSweep", "oRad", "oAimZ", "oFov",
                    "oPath", "oElevSweep", "oElevCycles", "oHelixMin",
                    "oHelixMax", "oHelixStart", "oFit", "oW", "oH",
                    "cCut", "oDepthStr", "oCard",
                    "sX", "sY", "sZ", "sRot", "sRotX", "sRotY",
                    "cCropOn", "cCropX", "cCropY", "cCropZ", "cCropR"]) {
    const el = $(id);
    if (el) el.addEventListener("input", pushOrbit);
  }
}

function setGizmo(mode) {
  gizmoState = mode;
  orbit.setGizmo(mode);
  const label = { off: "off", move: "move", rotate: "rotate", crop: "crop sphere" };
  $("gizBtn").textContent = `Gizmo: ${label[mode] || "off"}`;
  $("gizBtn").classList.toggle("on", mode !== "off");
  logUi(`gizmo → ${label[mode] || "off"}`);
}

export function renderRefs() {
  const el = $("refsStrip");
  const refs = S.refs.refs || [];
  el.innerHTML = refs.map(r =>
    `<div style="border:1px solid var(--line);padding:2px">
       <img src="${esc(r.url)}" style="height:48px;display:block" alt="">
       <button data-ref="${esc(r.name)}" style="height:20px;width:100%;font-size:10px;padding:0">remove</button>
     </div>`).join("")
    || `<span class="dim" style="font:400 11px/16px var(--font-ui)">none</span>`;
  el.querySelectorAll("button[data-ref]").forEach(b => {
    b.onclick = async () => {
      try {
        const j = await api("/api/refs/delete", { project: S.project, name: b.dataset.ref });
        S.refs.refs = j.refs;
        logUi(`removed the extra reference ${S.state._dir}\\refs\\${j.deleted}`);
        fx.toast(`Removed ${j.deleted}`, "ok");
        renderRefs();
      } catch (e) { /* reported by api() */ }
    };
  });
}

/** Populate the selects that come from the server. Called once at boot. */
export function fillServerSelects() {
  $("gEngine").innerHTML = S.engines.map(e =>
    `<option value="${esc(e.id)}">${esc(e.label)}${e.pose_exact ? "" : "  ·  not pose-exact"}</option>`).join("");
  $("gVision").innerHTML = S.visionModels.map(m =>
    `<option value="${esc(m)}">${esc(m)}</option>`).join("");
  const alpha = S.matteModels.map(m =>
    `<option value="${esc(m.id)}">${esc(m.label)}</option>`).join("")
    + `<option value="authored">authored coverage — LOOK at it, do not train on it</option>`
    + `<option value="distance">colour distance to the backdrop only</option>`;
  $("uAlpha").innerHTML = alpha;
  $("uAlpha").value = "birefnet";
  $("dsMatte").innerHTML = alpha;
  $("dsMatte").value = "birefnet";
  $("paEngine").innerHTML = $("gEngine").innerHTML;
  engineOpts(true);
}

export { engineOpts, aspectOption, setSplatView };
