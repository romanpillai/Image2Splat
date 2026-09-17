// The passes workspace: two extra orbits rendered FROM THE TRAINED SPLAT.
//
// Why it is separate, and why there is no central ring: three rings in one
// authored orbit asks a single generation to hold identity across three
// heights at once, and the video model does not do that. Here the splat
// already exists, so each extra pass is generated against something fixed.
//
// The load-bearing rule of this whole workspace is INHERITANCE. Lens, render
// size, frame count, fps and aspect all come from the author AI video and are
// never editable. COLMAP describes a lens in PIXELS -- the same 29.8 degree
// lens is focal 1172.6 at 624x624 and 1804.0 at 960x960 -- so keeping every
// clip identical is what lets ONE camera describe all three rings. Making any
// of these typeable is how the merged dataset stops building.

import {
  $, num, esc, api, apiRaw, run, runDetached, logUi, S, renderStatus,
  renderStatusFolded, gate, bust, fireStateChange, dialog, fx,
  jobStart, jobUpdate, jobEnd, jobActive, shortErr,
} from "./core.js";
import { createPasses } from "./psplat3.js";
import { apvShow } from "./approve.js";
import { renderFiles, genBody, aspectOption, releaseSplat, lowestRes } from "./author.js";

const VACE_RES = ["auto", "240p", "360p", "480p", "580p", "720p"];
const LTX_RES = ["720p", "480p"];

const pa = { v: null, list: [], loaded: null, proj: null };
let which = "top";
let busy = false, cancelWanted = false;
let playTimer = null;
let filesKey = "", filesMtimes = null;
let onDirty = () => {};
export function setDirtyHook(fn) { onDirty = fn; }

const parent = () => S.pass.parent || null;

/** The author camera's vertical FOV, which the passes must share.
 *  The focal length IS the geometry, not a look. */
function paVfov() {
  const v = S.state.orbit && +S.state.orbit.vfov_deg;
  return (v && isFinite(v) && v > 0) ? v : 42;
}

/** The aspect the pass will actually be ENCODED at. */
function paAspect() {
  const p = parent();
  if (p && p.width > 0 && p.height > 0) return p.width / p.height;
  const o = S.state.orbit || {};
  if (+o.width > 0 && +o.height > 0) return o.width / o.height;
  return 1;
}

/** Frame count and fps for a pass -- INHERITED, like the lens.
 *  Measured off the parent clip first, then the authored orbit: fal is free to
 *  return something other than what was asked for. */
function paClip() {
  const p = parent() || {};
  const o = S.state.orbit || {};
  const frames = (+p.frames > 0 && p.frames) || (+o.frames > 0 && o.frames) || 75;
  const fps = Math.round(+p.fps) || Math.round(+S.state.fps) || 24;
  return { frames: Math.round(frames), fps, secs: Math.round(frames) / fps,
           source: (+p.frames > 0) ? (p.name || "author AI video")
                 : (+o.frames > 0) ? "author orbit"
                 : "default — no author clip yet" };
}

function paParams() {
  return {
    frames: paClip().frames, sweep: num("paSweep", 360), aim_z: num("paAimZ", 2.3),
    topZ: num("paTopZ", 5), topR: Math.max(0.1, num("paTopR", 9)),
    botZ: num("paBotZ", -1), botR: Math.max(0.1, num("paBotR", 9)),
  };
}

export function paSync() {
  if (!pa.v) return;
  const q = paParams();
  pa.v.setRings({ aimZ: q.aim_z, topZ: q.topZ, topR: q.topR, botZ: q.botZ,
                  botR: q.botR, frames: q.frames, sweep: q.sweep,
                  vfov: paVfov(), aspect: paAspect() });
  const fr = $("paFrame");
  fr.max = String(Math.max(0, q.frames - 1));
  if (+fr.value > +fr.max) fr.value = fr.max;
  pa.v.setFrame(+fr.value);
  paFrameLabel();
}

function paFrameLabel() {
  const fr = $("paFrame");
  fx.syncRange(fr);
  const c = paClip();
  $("paFrameLbl").textContent = `frame ${+fr.value + 1} / ${+fr.max + 1} · ${(+fr.value / Math.max(c.fps, 1)).toFixed(2)}s`;
}

function setPaFrame(i) {
  const fr = $("paFrame");
  const v = Math.max(0, Math.min(+fr.max, Math.round(i)));
  fr.value = v;
  if (pa.v) pa.v.setFrame(v);
  paFrameLabel();
}

function togglePlay(on) {
  const playing = on === undefined ? !playTimer : !!on;
  if (playTimer) { cancelAnimationFrame(playTimer); playTimer = null; }
  const b = $("paPlay");
  b.classList.toggle("on", playing);
  b.innerHTML = playing ? "&#10074;&#10074;" : "&#9654;";
  b.setAttribute("aria-label", playing ? "Pause" : "Play");
  if (!playing) return;
  let last = performance.now(), acc = 0;
  const step = now => {
    acc += (now - last) / 1000 * paClip().fps;
    last = now;
    if (acc >= 1) {
      const n = Math.floor(acc); acc -= n;
      let v = +$("paFrame").value + n;
      if (v > +$("paFrame").max) v = 0;
      setPaFrame(v);
    }
    playTimer = requestAnimationFrame(step);
  };
  playTimer = requestAnimationFrame(step);
}

export function scrubKey(e) {
  const v = +$("paFrame").value;
  if (e.key === "ArrowLeft") { togglePlay(false); setPaFrame(v - (e.shiftKey ? 10 : 1)); return true; }
  if (e.key === "ArrowRight") { togglePlay(false); setPaFrame(v + (e.shiftKey ? 10 : 1)); return true; }
  if (e.key === "Home") { setPaFrame(0); return true; }
  if (e.key === "End") { setPaFrame(+$("paFrame").max); return true; }
  if (e.key === " ") { togglePlay(); return true; }
  return false;
}

// =============================================================== statuses ===
function statusInherited() {
  const p = parent();
  const c = paClip();
  const rows = [];
  if (!p) {
    rows.push({ k: "parent clip", problem: true,
      v: "No AI video in the author workspace yet — the passes have nothing to "
       + "match. Every button here is blocked until one exists." });
    renderStatus("stInherited",
      "Inherited from the author workspace — never editable", rows);
    return 1;
  }
  rows.push({ k: "parent clip", v: p.name || "(the authored orbit canvas)",
              from: p.source });
  rows.push({ k: "render size", v: `${p.width} × ${p.height}`,
              from: "from the parent, never a setting" });
  rows.push({ k: "frames / fps", v: `${c.frames} / ${c.fps} = ${c.secs.toFixed(2)}s`,
              from: c.source });
  rows.push({ k: "lens (vFOV)", v: `${paVfov().toFixed(2)}°`,
              from: "the author orbit" });
  rows.push({ k: "aspect", v: S.pass.author_aspect || "not a standard option",
              from: "one COLMAP camera for all rings" });
  rows.push({ k: "project folder", v: S.pass.project_dir || "" });
  $("paAspectD").textContent = S.pass.author_aspect || "adaptive";
  renderStatus("stInherited",
    "Inherited from the author workspace — never editable", rows);
  return 0;
}

function statusSplat() {
  const rows = [];
  let probs = 0;
  if (!pa.list.length) {
    rows.push({ k: "splat", problem: true,
      v: "no .ply in this project — train one in the author workspace, or "
       + "upload one here" });
    probs++;
  } else {
    const sel = pa.list.find(x => x.name === $("paPick").value) || pa.list[pa.list.length - 1];
    rows.push({ k: "checkpoint", v: sel ? sel.name : "" });
    rows.push({ k: "path", v: sel ? `${S.splats.dir}\\${sel.name}` : "" });
    rows.push({ k: "size", v: sel ? `${(sel.bytes / 1e6).toFixed(1)} MB` : "" });
    rows.push({ k: "modified", v: sel ? new Date(sel.mtime * 1000).toLocaleString() : "" });
    rows.push({ k: "loaded", v: pa.loaded === (sel && sel.name) ? "yes" : "not yet" });
  }
  renderStatus("stPaSplat", "Splat", rows);
  return probs;
}

function statusRings() {
  const q = paParams();
  const sep = q.topZ - q.botZ;
  const rows = [
    { k: "↑ above", v: `height ${q.topZ.toFixed(2)}, radius ${q.topR.toFixed(2)}` },
    { k: "↓ below", v: `height ${q.botZ.toFixed(2)}, radius ${q.botR.toFixed(2)}` },
    { k: "separation", v: `${sep.toFixed(2)} world units`,
      from: sep < 1 ? "the two passes barely differ" : "" },
    { k: "poses per pass", v: `${q.frames}`, from: "inherited" },
    { k: "total poses", v: `${q.frames * 3} across three rings` },
    { k: "aim height", v: q.aim_z.toFixed(2) },
  ];
  let probs = parent() ? 0 : 1;
  if (!pa.list.length) probs++;
  if (Math.abs(sep) < 0.5) {
    rows.push({ k: "spread", problem: true,
      v: "the two rings are at nearly the same height, so they add no vertical "
       + "parallax over the author ring — which is the only reason to render "
       + "them" });
    probs++;
  }
  const p = parent();
  for (const w of ["top", "bottom"]) {
    const card = S.pass[w];
    rows.push({ k: `${w === "top" ? "↑" : "↓"} render`,
      v: card ? `${card.name} — ${(card.bytes / 1e6).toFixed(2)} MB` : "not rendered yet" });
    if (card) rows.push({ k: "path", v: card.path });
  }
  if (busy) {
    rows.push({ k: "rendering", v: $("paRenderOut").textContent || "in progress" });
  }
  renderStatus("stRings", "Rings", rows);
  gate("btnPaRender", probs, `Render the ${which === "top" ? "above" : "below"} pass`);
  return probs;
}

function statusPaEngine() {
  const e = (S.engines || []).find(x => x.id === $("paEngine").value);
  if (!e) { renderStatus("stPaEngine", "Engine", [{ k: "", v: "loading…" }]); return; }
  const rows = [];
  if (!e.pose_exact) rows.push({ k: "", note: true, problem: true, v: "Not pose-exact — retime before the dataset builds." });
  rows.push({ k: "pose-exact", v: e.pose_exact ? "yes" : "no — retime before the dataset builds" });
  rows.push({ k: "endpoint", v: e.endpoint });
  rows.push({ k: "note", v: e.note });
  renderStatusFolded("stPaEngine", "Engine", rows, "engine details");
}

function statusSend() {
  const rows = [];
  let probs = parent() ? 0 : 1;
  const p = parent();
  const ctl = S.pass[which];
  const soft = S.pass[`soft_${which}`];
  const ai = S.pass[`ai_${which}`];
  const wantSoft = num("paSoftDown", 1) > 1 || num("paSoftBlur", 0) > 0;

  rows.push({ k: "pass", v: which === "top" ? "↑ above" : "↓ below" });
  if (!ctl) {
    rows.push({ k: "control video", problem: true,
      v: `no ${which} pass rendered yet — render it in step 2 first` });
    probs++;
  } else if (wantSoft) {
    if (!soft || soft.stale) {
      rows.push({ k: "control video", problem: true,
        v: `softening is on but ${which}_soft.mp4 is missing or was built from `
         + `different settings. Build it — that exact file is what gets sent.`,
        fix: { label: "Build", onClick: () => $("btnPaSoften").click() } });
      probs++;
    } else {
      rows.push({ k: "control video", v: `${soft.name} — ${soft.factor}× down, `
        + `${soft.blur}px blur`, from: "softened" });
      rows.push({ k: "path", v: soft.path });
    }
  } else {
    rows.push({ k: "control video", v: `${ctl.name} — sharp`, from: "as rendered" });
    rows.push({ k: "path", v: ctl.path });
  }
  rows.push({ k: "reference", v: "the source photo — a pass never references "
    + "control_frame0.png, which is the author's clay render from the author's camera" });
  rows.push({ k: "resolution", v: $("paURes").value || "—" });
  rows.push({ k: "aspect", v: S.pass.author_aspect || "adaptive",
              from: "forced from the author clip" });

  if (ai) {
    rows.push({ k: "came back", v: `${ai.name} — ${ai.width}×${ai.height}, `
      + `${ai.frames} frames @ ${ai.fps}` });
    rows.push({ k: "path", v: ai.path });
    if (p && (ai.width !== p.width || ai.height !== p.height)) {
      rows.push({ k: "size", problem: true,
        v: `${ai.width}×${ai.height} against the author's ${p.width}×${p.height}. `
         + `The merged dataset would need a separate COLMAP camera for it.`,
        fix: { label: "Conform", onClick: () => conform() } });
      probs++;
    }
    const c = paClip();
    if (ai.frames !== c.frames || Math.round(ai.fps) !== c.fps) {
      rows.push({ k: "timing", problem: true,
        v: `${ai.frames} frames @ ${ai.fps} against the author's ${c.frames} @ `
         + `${c.fps}. Every pose would address the wrong image.`,
        fix: { label: "Retime", onClick: () => retimePass() } });
      probs++;
    }
    if (ai.matches) rows.push({ k: "match", v: "identical to the author clip — "
      + "one COLMAP camera covers all three rings" });
  } else {
    rows.push({ k: "came back", v: "nothing yet for this pass" });
  }
  if (!S.health.fal_key) {
    rows.push({ k: "FAL_KEY", problem: true,
      v: "not set in this server's environment — set it and restart the server" });
    probs++;
  }
  renderStatus("stPaSend", "What will be sent", rows);
  gate("paSend", probs, "Send this pass — review before it goes");
  return probs;
}

let dsPlanCache = null;
let dsPlanTimer = null;
/** Debounced: applyUI replays every control at once, and an un-debounced plan
 *  turned a snapshot restore into four identical POSTs in the console this
 *  build exists to keep readable. */
function dsPlan() {
  clearTimeout(dsPlanTimer);
  dsPlanTimer = setTimeout(dsPlanNow, 250);
}
async function dsPlanNow() {
  try {
    dsPlanCache = await api("/api/dataset/complete/plan", dsBody(), { quiet: true });
  } catch (e) { dsPlanCache = null; }
  statusDataset();
}

function dsBody() {
  return { project: S.project, top: $("dsTop").checked,
           bottom: $("dsBottom").checked, matte_source: $("dsMatte").value,
           init_from_splat: $("dsInitSplat").checked, max_init_points: 250000 };
}

function statusDataset() {
  const j = dsPlanCache;
  const rows = [];
  let probs = 0;
  if (!j) {
    rows.push({ k: "", v: "the plan has not been read yet" });
  } else {
    rows.push({ k: "rings", v: (j.rings || []).map(r => r.label).join(", ") || "—" });
    for (const r of (j.rings || [])) {
      rows.push({ k: r.label, v: `${r.name} — cam z ${r.cam_z}, radius ${r.radius}, `
        + `${r.frames} frames` });
      rows.push({ k: "path", v: r.path });
    }
    rows.push({ k: "poses", v: `${j.poses}` });
    rows.push({ k: "init cloud", v: j.init });
    rows.push({ k: "output", v: j.out_dir });
    for (const p of (j.problems || [])) { rows.push({ k: "problem", v: p, problem: true }); probs++; }
    if (!probs) rows.push({ k: "state", v: "ready to build" });
  }
  renderStatus("stDataset", "Dataset", rows);
  gate("btnDsBuild", probs, "Build dataset_complete/");
  return probs;
}

// ================================================================ actions ===
async function conform() {
  await run(`conform the ${which} pass to the author clip`,
            () => api("/api/pass/conform", { project: S.project, which }),
            { working: `Conforming the ${which} pass`, ok: "Pass conformed", busy: "Conforming…" })
    .catch(() => {});
  await refreshPassStatus();
}

async function retimePass() {
  const c = paClip();
  const ai = S.pass[`ai_${which}`] || {};
  const ok = await dialog({
    title: `Resample the ${which} pass to ${c.frames} frames?`,
    okLabel: "Retime",
    bodyHtml:
      `<p>This fixes the frame COUNT so every pose addresses an image. It `
      + `assumes the clip sweeps the authored ring linearly in time — if it `
      + `does not, the splat will be wrong in a way nothing downstream can `
      + `detect.</p><p>The original is kept alongside as *_orig.mp4. Any matte `
      + `cut against the old timing is cleared.</p>`
      + `<p class="dim">${esc(ai.name || "")}: ${ai.frames || "?"} frames at `
      + `${ai.fps || "?"} fps → ${c.frames} at ${c.fps}.</p>`,
  });
  if (!ok) { logUi("pass retime cancelled"); return; }
  await run(`retime the ${which} pass`,
            () => api("/api/pass/retime", { project: S.project, which }),
            { working: `Retiming the ${which} pass`, ok: "Pass retimed", busy: "Retiming…" })
    .catch(() => {});
  await refreshPassStatus();
}

/** Render one pass IN THE BROWSER, posting one frame at a time.
 *
 * One at a time rather than batched: a 90-frame pass at 832x480 is tens of
 * megabytes and a single request carrying all of it is one failure away from
 * losing the whole render. It goes through the approval window because it is
 * expensive browser-side work -- and it has a Cancel, which training has and
 * this did not.
 */
async function renderPass() {
  if (busy) { logUi("a pass is already rendering", "warn"); return; }
  if (!pa.v || !pa.v.hasSplat()) { logUi("load a splat first", "warn"); return; }
  const p = parent();
  if (!p) { logUi("no parent to match — generate an AI video in the author workspace first", "warn"); return; }
  const q = paParams();
  // EVEN dimensions: x264 with yuv420p refuses odd ones outright.
  const w = p.width - (p.width % 2), h = p.height - (p.height % 2);
  const est = q.frames * w * h * 1.2 / 1e6;   // PNG, roughly

  apvShow({
    files: [], problems: [], project_dir: S.pass.project_dir,
    settings: {
      pass: which, frames: q.frames, size: `${w} × ${h}`,
      fps: paClip().fps, lens: `${paVfov().toFixed(2)}° vFOV (inherited)`,
      ring: `height ${which === "top" ? q.topZ : q.botZ}, radius ${which === "top" ? q.topR : q.botR}`,
      writes: `${S.pass.project_dir}\\pass_frames_${which}\\frame_%04d.png`,
      then: `${S.pass.project_dir}\\pass_${which}.mp4`,
      estimated_upload: `${est.toFixed(0)} MB in ${q.frames} requests`,
    },
  }, {
    title: `Render the ${which === "top" ? "above" : "below"} pass`,
    subtitle: "Rendered in this browser from the loaded splat, one frame at a "
            + "time. It can be cancelled from the viewport bar.",
    goLabel: "Render",
    needFiles: false,
    onApprove: () => doRenderPass(w, h, q),
  });
}

async function doRenderPass(w, h, q) {
  busy = true; cancelWanted = false;
  togglePlay(false);
  $("btnPaCancel").classList.remove("hidden");
  const btn = $("btnPaRender");
  const label = which === "top" ? "above" : "below";
  fx.btnBusy(btn, `Rendering the ${label} pass…`);
  const tok = jobStart({ title: `Rendering the ${label} pass`, detail: `${w}×${h} · starting`, kind: "client" });
  let outcome = { kind: "ok", title: `${label[0].toUpperCase() + label.slice(1)} pass rendered` };
  try {
    logUi(`pass ${which}: begin — clearing ${S.pass.project_dir}\\pass_frames_${which}`);
    await api("/api/pass/begin", { project: S.project, name: which });
    for (let i = 0; i < q.frames; i++) {
      if (cancelWanted) {
        logUi(`pass ${which}: cancelled at frame ${i}`, "warn");
        outcome = { kind: "stop", title: "Render cancelled", detail: `stopped at frame ${i} of ${q.frames}` };
        break;
      }
      const blob = await pa.v.renderPassFrame(which, i, w, h);
      if (!blob) throw new Error(`frame ${i} came back empty`);
      const r = await fetch(
        `/api/pass/frame?project=${encodeURIComponent(S.project)}`
        + `&which=${which}&i=${i}`, { method: "POST", body: blob });
      if (!r.ok) throw new Error(`frame ${i} upload failed`);
      $("paRenderOut").textContent = `${which}: frame ${i + 1} of ${q.frames} at ${w}×${h}`;
      jobUpdate({ detail: `Frame ${i + 1} of ${q.frames} · ${w}×${h}`, pct: 100 * (i + 1) / q.frames }, tok);
      if (i % 4 === 0) await new Promise(res => setTimeout(res, 0));   // let the tab breathe
      if (i % 20 === 0) statusRings();
    }
    if (!cancelWanted) {
      const c = paClip();
      $("paRenderOut").textContent = `encoding ${which}…`;
      jobUpdate({ detail: `Encoding ${q.frames} frames at ${c.fps} fps…`, pct: 0 }, tok);
      const j = await run(`encode the ${which} pass`, () =>
        api("/api/pass/encode", { project: S.project, which, fps: c.fps }),
        { btn: null, track: false });
      logUi(`pass ${which}: ${q.frames} frames at ${c.fps} fps `
          + `(${c.secs.toFixed(2)}s, matching the ${c.source}) → ${S.pass.project_dir}\${j.output}`);
      outcome.detail = `${j.output} · ${q.frames} frames · ${w}×${h}`;
      $("paRenderOut").textContent = "";
    }
  } catch (e) {
    $("paRenderOut").textContent = `render failed: ${e.message}`;
    logUi(`pass ${which} failed: ${e.message}`, "error");
    outcome = { kind: "fail", title: "Pass render failed", detail: shortErr(e.message) };
  } finally {
    busy = false; cancelWanted = false;
    $("btnPaCancel").classList.add("hidden");
    fx.btnDone(btn, outcome.kind !== "fail", outcome.kind === "ok" ? "Rendered" : outcome.kind === "stop" ? "Cancelled" : "Failed");
    jobEnd(outcome, tok);
    await refreshPassStatus();
  }
}

function paBody() {
  const id = $("paEngine").value;
  const seed = ($("paUSeed").value || "").trim();
  const p = parent() || {};
  const asp = S.pass.author_aspect
           || aspectOption(p.width || 1, p.height || 1);
  return Object.assign(genBody(), {
    pass: which,
    engine: id,
    prompt: ($("paPrompt").value || "").trim(),
    ref_downsample: num("paSoftDown", 1),
    ref_blur: num("paSoftBlur", 0),
    intensity: $("paInt").value,
    detail_refine: $("paRefine").checked,
    resolution: id === "ltx" ? $("paURes").value : "720p",
    vace_resolution: id === "wan22vace" ? $("paURes").value : "720p",
    vace_steps: num("paVaceSteps", 30),
    vace_guidance: num("paVaceGuide", 50) / 10,
    vace_sampler: $("paVaceSampler").value,
    vace_negative: $("paUNeg").value || "",
    vace_seed: seed ? +seed : null,
    wan_resolution: id === "wan" ? $("paURes").value : "720p",
    wan_aspect: asp,
    wan_prompt_expansion: $("paWanExpand").checked,
    wan_thinking: $("paWanThink").checked,
    wan_seed: seed ? +seed : null,
    mm_resolution: id === "minimax" ? $("paURes").value : "768P",
    mm_aspect: asp,
    mm_expansion: $("paMmExpand").value,
    mm_seed: seed ? +seed : null,
  });
}

async function sendPass() {
  const body = paBody();
  const btn = $("paSend");
  fx.btnBusy(btn, "Preparing the plan…");
  let plan;
  try { plan = await api("/api/generate/plan", body); }
  catch (e) { fx.btnDone(btn, false, "Refused"); return; }
  fx.btnRestore(btn);
  apvShow(plan, {
    title: `Send the ${which === "top" ? "above" : "below"} pass to ${plan.settings.label}`,
    subtitle: `${plan.settings.endpoint} · ${plan.settings.frames} frames · the `
            + `pass prompt carries a camera clause at position 2`,
    goLabel: "Approve & send",
    // Closes on ACCEPTANCE, not on completion -- a pass generation is minutes.
    onApprove: () => runDetached(`send the ${which} pass`,
                                 () => api("/api/generate", body), {
      btn, stage: "generate", busy: "Sending…", busyRunning: "Generating…",
      working: `Generating the ${which === "top" ? "above" : "below"} pass`,
      ok: "Pass video ready", okBtn: "Pass ready",
    }),
  });
}

async function buildDataset() {
  const body = dsBody();
  const btn = $("btnDsBuild");
  fx.btnBusy(btn, "Preparing…");
  let plan;
  try { plan = await api("/api/dataset/complete/plan", body); }
  catch (e) { fx.btnDone(btn, false, "Refused"); return; }
  fx.btnRestore(btn);
  apvShow({ files: [], problems: plan.problems || [], project_dir: S.pass.project_dir,
            settings: { rings: (plan.rings || []).map(r => r.label).join(", "),
                        poses: plan.poses, init: plan.init,
                        output: plan.out_dir,
                        alpha: $("dsMatte").value } },
    { title: "Build dataset_complete/",
      subtitle: "dataset_complete/ never overwrites dataset/, so the set the "
              + "current splat was trained from stays reproducible.",
      goLabel: "Build", needFiles: false,
      notes: (plan.rings || []).map(r => `${r.label}: ${r.path}`),
      onApprove: () => runDetached("build dataset_complete",
                                   () => api("/api/dataset/complete", body),
                                   { btn, stage: "dataset-complete", busyRunning: "Building…", okBtn: "Built" }) });
}

// ================================================================ refresh ===
export async function refreshPassStatus() {
  try {
    S.pass = await api(`/api/pass/status?project=${encodeURIComponent(S.project)}`,
                       undefined, { quiet: true });
  } catch (e) { return; }
  paSync();
  repaint();
  passFiles();
}

export function repaint() {
  try {
    statusInherited(); statusSplat(); statusRings(); statusPaEngine(); statusSend();
    statusDataset();
    $("paSoftBlurV").textContent = num("paSoftBlur", 0);
    $("paVaceStepsVal").textContent = num("paVaceSteps", 30);
    $("paVaceGuideVal").textContent = (num("paVaceGuide", 50) / 10).toFixed(1);
    const id = $("paEngine").value;
    $("paLtxOpts").classList.toggle("hidden", id !== "ltx");
    $("paVaceOpts").classList.toggle("hidden", id !== "wan22vace");
    $("paWanOpts").classList.toggle("hidden", id !== "wan");
    $("paMmOpts").classList.toggle("hidden", id !== "minimax");
  } catch (e) { console.error("passes repaint", e); }
}

/** reset = the engine was switched: that engine's LOWEST resolution. */
function paResOpts(reset = false) {
  const id = $("paEngine").value;
  const opts = id === "wan" ? (S.engineExtras.wan_resolutions || [])
             : id === "minimax" ? (S.engineExtras.mm_resolutions || [])
             : id === "wan22vace" ? VACE_RES : LTX_RES;
  const keep = $("paURes").value;
  $("paURes").innerHTML = opts.map(o => `<option value="${esc(o)}">${esc(o)}</option>`).join("");
  $("paURes").value = (!reset && opts.includes(keep)) ? keep : lowestRes(opts);
}

export function restoreRes(saved) {
  paResOpts(true);
  if (!saved || saved.engine !== $("paEngine").value) return;
  if ([...$("paURes").options].some(o => o.value === saved.value)) $("paURes").value = saved.value;
}

/** Rebuilt only when a file changed; each preview busted by its own mtime. */
function passFiles(force = false) {
  if (!force && !$("filesPasses").classList.contains("on")) return;
  const cards = [];
  const add = (role, c, note) => {
    if (!c) return;
    cards.push({ role, name: c.path, path: c.path, media: bust(c.url, c.mtime),
                 mtime: c.mtime, amount: `${(c.bytes / 1e6).toFixed(2)} MB`, note });
  };
  add("author AI clip", S.pass.author_clip, "everything here is measured against this");
  add("author raw return", S.pass.author_orig, "what fal sent back, before retiming");
  add("↑ above control", S.pass.top, "rendered in the browser from the splat");
  add("↑ above softened", S.pass.soft_top, S.pass.soft_top
      ? `${S.pass.soft_top.factor}× down, ${S.pass.soft_top.blur}px blur`
        + (S.pass.soft_top.stale ? " — STALE, rebuild it" : "") : "");
  add("↑ above returned", S.pass.ai_top, S.pass.ai_top
      ? `${S.pass.ai_top.width}×${S.pass.ai_top.height}, ${S.pass.ai_top.frames} frames` : "");
  add("↓ below control", S.pass.bottom, "rendered in the browser from the splat");
  add("↓ below softened", S.pass.soft_bottom, S.pass.soft_bottom
      ? `${S.pass.soft_bottom.factor}× down, ${S.pass.soft_bottom.blur}px blur`
        + (S.pass.soft_bottom.stale ? " — STALE, rebuild it" : "") : "");
  add("↓ below returned", S.pass.ai_bottom, S.pass.ai_bottom
      ? `${S.pass.ai_bottom.width}×${S.pass.ai_bottom.height}, ${S.pass.ai_bottom.frames} frames` : "");
  const key = cards.map(c => `${c.name}@${c.mtime}`).join("|");
  if (!force && key === filesKey) return;
  const seen = filesMtimes;
  filesMtimes = Object.fromEntries(cards.map(c => [c.name, c.mtime]));
  for (const c of cards) c.fresh = !!(seen && seen[c.name] !== c.mtime);
  filesKey = key;
  renderFiles("filesPassesList", cards);
}

/** Called on the poll while the passes workspace is on screen. */
export async function passFilesTick() {
  if (!$("filesPasses").classList.contains("on")) return;
  try {
    S.pass = await api(`/api/pass/status?project=${encodeURIComponent(S.project)}`,
                       undefined, { quiet: true });
  } catch (e) { return; }
  passFiles();
}

export async function refreshSplatList() {
  try {
    S.splats = await api(`/api/splats?project=${encodeURIComponent(S.project)}`,
                         undefined, { quiet: true });
  } catch (e) { return; }
  pa.list = S.splats.exports || [];
  const keep = $("paPick").value;
  $("paPick").innerHTML = pa.list.map(x =>
    `<option value="${esc(x.name)}">${esc(x.name)} — step ${x.step}${x.uploaded ? " (uploaded)" : ""}</option>`).join("");
  if (pa.list.some(x => x.name === keep)) $("paPick").value = keep;
  else if (pa.list.length) $("paPick").value = pa.list[pa.list.length - 1].name;
  $("paEmpty").classList.toggle("on", pa.list.length === 0);
}

async function loadSplat(name) {
  if (!pa.v || !name) return;
  const item = pa.list.find(x => x.name === name);
  const url = bust(`/projects/${S.project}/splat_out/${name}`, item && item.mtime);
  logUi(`passes: loading ${S.splats.dir}\\${name}`);
  const tok = jobActive() ? 0 : jobStart({ title: "Loading the splat", detail: name, kind: "client" });
  try {
    const n = await pa.v.load(url);
    pa.loaded = name;
    logUi(`passes: ${(+n).toLocaleString()} splats loaded from ${S.splats.dir}\\${name}`);
    pa.v.focus(num("paAimZ", 2.3));
    if (tok) jobEnd({ kind: "ok", title: "Splat loaded", detail: `${(+n).toLocaleString()} splats · ${name}` }, tok);
  } catch (e) {
    logUi(`passes: load failed — ${e.message}`, "error");
    if (tok) jobEnd({ kind: "fail", title: "Splat failed to load", detail: shortErr(e.message) }, tok);
    else fx.toast(`Splat failed to load — ${e.message}`, "err");
  }
  repaint();
}

/** Entering the workspace. ONE PlayCanvas app may live at a time, so the
 *  author's checkpoint viewer is destroyed first -- two apps and one of them
 *  renders black. */
export async function open() {
  releaseSplat();
  if (!pa.v) pa.v = createPasses($("cvPasses"));
  if (pa.proj !== S.project) { pa.loaded = null; pa.proj = S.project; }
  if (!$("paURes").options.length) paResOpts(true);
  await refreshSplatList();
  await refreshPassStatus();
  // Seed the aim height from the author's, once -- it used to default to a
  // hardcoded 2.3 whatever the project was.
  if (!$("paAimZ").dataset.seeded) {
    $("paAimZ").dataset.seeded = "1";
    const a = (S.state.orbit || {}).aim_z;
    if (a != null && !S.state.ui?.paAimZ) $("paAimZ").value = (+a).toFixed(2);
  }
  paSync();
  await dsPlan();
  if (pa.list.length && pa.loaded !== $("paPick").value) await loadSplat($("paPick").value);
}

export function close() {
  togglePlay(false);
  if (pa.v) { try { pa.v.destroy(); } catch (e) { /* already gone */ } pa.v = null; }
  pa.loaded = null;
}

// =================================================================== wire ===
export function init() {
  for (const b of document.querySelectorAll("#paWhich button")) {
    b.onclick = () => {
      which = b.dataset.w;
      for (const o of document.querySelectorAll("#paWhich button")) o.classList.remove("on");
      b.classList.add("on");
      logUi(`pass → ${which === "top" ? "above" : "below"}`);
      repaint();
    };
  }
  $("paReload").onclick = async () => {
    const b = $("paReload");
    fx.btnBusy(b, "Reloading…");
    await refreshSplatList();
    await loadSplat($("paPick").value);
    fx.btnDone(b, pa.loaded === $("paPick").value, pa.loaded ? "Reloaded" : "Nothing loaded");
  };
  $("paPick").addEventListener("change", () => loadSplat($("paPick").value));
  $("paUpload").onclick = () => $("paFile").click();
  $("paFile").onchange = async e => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    const b = $("paUpload");
    fx.btnBusy(b, "Uploading…");
    let j;
    try { j = await apiRaw(`/api/splat_upload?project=${encodeURIComponent(S.project)}`, f); }
    catch (err) { fx.btnDone(b, false, "Upload failed"); return; }
    fx.btnDone(b, true, "Uploaded");
    logUi(`uploaded ${S.splats.dir}\\${j.name} — ${(+j.splats).toLocaleString()} splats, ${j.mb} MB`);
    // /api/splats hides PLYs younger than 2 seconds -- a half-written file
    // parses as garbage -- so poll for the name to appear rather than reading
    // once and concluding it is not there.
    for (let i = 0; i < 12; i++) {
      await new Promise(r => setTimeout(r, 700));
      await refreshSplatList();
      if (pa.list.some(x => x.name === j.name)) break;
    }
    $("paPick").value = j.name;
    await loadSplat(j.name);
  };

  $("btnPaRender").onclick = () => renderPass().catch(e => logUi(e.message, "error"));
  $("btnPaCancel").onclick = () => {
    cancelWanted = true;
    fx.btnBusy($("btnPaCancel"), "Cancelling…");
    logUi("cancel requested — the pass will stop after the current frame", "warn");
  };
  $("btnPaSoften").onclick = async () => {
    await run(`build the softened ${which} pass`, () => api("/api/pass/soften", {
      project: S.project, which, ref_downsample: num("paSoftDown", 1),
      ref_blur: num("paSoftBlur", 0),
    }), { btn: $("btnPaSoften"), working: `Softening the ${which} pass`, ok: "Softened pass built",
          busy: "Softening…", okBtn: "Built" }).catch(() => {});
    await refreshPassStatus();
  };
  $("paEngine").addEventListener("change", e => {
    logUi(`pass engine → ${$("paEngine").selectedOptions[0]?.textContent || ""}`);
    paResOpts(true); repaint();
    if (e.isTrusted) fx.toast(`Resolution set to ${$("paURes").value}, this engine's lowest`, "info");
  });
  $("paSend").onclick = () => sendPass().catch(e => logUi(e.message, "error"));
  $("btnDsBuild").onclick = () => buildDataset().catch(e => logUi(e.message, "error"));
  for (const id of ["dsTop", "dsBottom", "dsInitSplat", "dsMatte"]) {
    $(id).addEventListener("change", dsPlan);
  }

  for (const id of ["paAimZ", "paSweep", "paTopZ", "paTopR", "paBotZ", "paBotR"]) {
    $(id).addEventListener("input", () => { paSync(); repaint(); });
  }
  $("paFrame").addEventListener("input", () => { togglePlay(false); setPaFrame(+$("paFrame").value); });
  $("paFramePrev").onclick = () => { togglePlay(false); setPaFrame(+$("paFrame").value - 1); };
  $("paFrameNext").onclick = () => { togglePlay(false); setPaFrame(+$("paFrame").value + 1); };
  $("paPlay").onclick = () => togglePlay();
  $("paSplit").onclick = () => {
    const on = !$("paSplit").classList.contains("on");
    $("paSplit").classList.toggle("on", on);
    if (pa.v) pa.v.setSplit(on);
    logUi(`passes split view ${on ? "on" : "off"}`);
  };
  $("paShowRings").onclick = () => {
    const on = !$("paShowRings").classList.contains("on");
    $("paShowRings").classList.toggle("on", on);
    if (pa.v) pa.v.setRig(on);
    logUi(`ring rig ${on ? "shown" : "hidden"}`);
  };
  $("paReset").onclick = () => {
    if (pa.v) { pa.v.focus(num("paAimZ", 2.3)); fx.btnDone($("paReset"), true, "View reset", 800); }
  };
  $("paFilesOn").onclick = () => {
    const on = !$("filesPasses").classList.contains("on");
    $("filesPasses").classList.toggle("on", on);
    $("paFilesOn").classList.toggle("on", on);
    document.body.classList.toggle("files-open", on);
    if (on) { filesMtimes = null; passFiles(true); }
    logUi(`files panel ${on ? "opened" : "closed"}`);
  };
  $("filesPassesClose").onclick = () => {
    $("filesPasses").classList.remove("on");
    $("paFilesOn").classList.remove("on");
    document.body.classList.remove("files-open");
  };
  $("paPrompt").addEventListener("input", e => {
    if (e.isTrusted) $("paPrompt").dataset.auto = "0";
  });
}
