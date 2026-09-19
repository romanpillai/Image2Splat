// The shell: project chrome, the two workspaces, the step rail, the poll loop,
// and the UI snapshot.
//
// Rules from the predecessor that a rebuild must not lose:
//
//  * The UI snapshot is a RESTORE, and restoring happens when a PROJECT IS
//    OPENED -- once, not on every poll tick. Re-syncing controls from server
//    state on a timer is the "my camera resets when I press render" bug.
//  * Step unlock follows the ARTEFACTS ON DISK, not the stage pointer, so
//    reopening a finished project does not leave half the rail dead.
//  * ONE step open per rail, and which one is remembered per project.

import {
  $, api, esc, S, logUi, onStateChange, wireDisclosures, fx,
  initDialog, dialog, renderStatus, run, openStep, closeStep, stepKey,
  jobFromStatus, jobEndFromStatus, jobActive, jobLast,
} from "./core.js";
import * as con from "./consolepanel.js";
import { initApproval } from "./approve.js";
import * as author from "./author.js";
import * as passes from "./passes.js";

let workspace = "author";
let uiApplying = false;
let uiAppliedFor = null;
let uiSaveTimer = null;
let UI_DEFAULTS = null;
let lastErrorAt = (() => {
  try { return +localStorage.getItem("img2splat.errAt") || 0; } catch (e) { return 0; }
})();

// Derived values are never persisted: a stored copy of a derived value fights
// whatever derives it. Mirror sliders (class "mirror") are skipped by class.
const UI_SKIP = new Set(["file", "aiFile", "splatFile", "paFile", "importFile",
                         "refsFile", "projList", "conFind", "conFollow",
                         "uRes", "paURes", "oElev", "oPhase", "oW", "oH"]);

// ============================================================ UI snapshot ===
function collectUI() {
  const out = {};
  document.querySelectorAll(
    "#railAuthor input, #railAuthor select, #railAuthor textarea,"
    + " #railPasses input, #railPasses select, #railPasses textarea")
    .forEach(el => {
      if (!el.id || UI_SKIP.has(el.id) || el.type === "file"
          || el.classList.contains("mirror")) return;
      out[el.id] = el.type === "checkbox" ? el.checked : el.value;
    });
  document.querySelectorAll(".adv[data-adv]").forEach(a => {
    out["_adv_" + a.dataset.adv] = a.classList.contains("open");
  });
  // The resolution is saved WITH its engine: the options depend on it, and a
  // project reopens on the resolution it was left at.
  out._res = { engine: $("gEngine").value, value: $("uRes").value };
  out._paRes = { engine: $("paEngine").value, value: $("paURes").value };
  return out;
}

function applyUI(ui) {
  if (!ui) return;
  uiApplying = true;
  try {
    for (const [id, v] of Object.entries(ui)) {
      if (id.startsWith("_adv_")) {
        const a = document.querySelector(`.adv[data-adv="${CSS.escape(id.slice(5))}"]`);
        if (a) {
          a.classList.toggle("open", !!v);
          a.querySelector(".pm").textContent = v ? "−" : "+";
        }
        continue;
      }
      if (id.startsWith("_")) continue;
      const el = document.getElementById(id);
      if (!el || UI_SKIP.has(id) || el.classList.contains("mirror")) continue;
      // oDepthStr was a number box in WORLD UNITS (1.5) and is now a slider in
      // hundredths (150). The two ranges do not overlap, so the reading is
      // unambiguous.
      if (id === "oDepthStr" && +v > 0 && +v <= 4) {
        el.value = String(Math.round(+v * 100));
        continue;
      }
      // MoGe-2 or Depth Anything V2. A model this build no longer offers
      // (SHARP) opens as MoGe-2 -- said once, so a cloud that looks different
      // on the next lift is not a mystery.
      if (id === "oDepthModel") {
        const offered = [...el.options].some(o => o.value === String(v));
        if (v && !offered) {
          logUi(`this project used ${String(v).toUpperCase()} `
              + `— not available in this build, so it lifts with MoGe-2 and the next lift will look different`, "warn");
        }
        el.value = offered ? String(v) : "moge2";
        continue;
      }
      if (el.type === "checkbox") { el.checked = !!v; continue; }
      if (el.tagName === "SELECT" && el.options.length
          && ![...el.options].some(o => o.value === String(v))) continue;
      el.value = v;
    }
    for (const id of Object.keys(ui)) {
      if (id.startsWith("_")) continue;
      const el = document.getElementById(id);
      if (!el || UI_SKIP.has(id) || el.classList.contains("mirror")) continue;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }
    author.restoreRes(ui._res);
    passes.restoreRes(ui._paRes);
  } finally {
    uiApplying = false;
  }
}

function saveUI() {
  if (uiApplying || !S.project) return;
  clearTimeout(uiSaveTimer);
  const p = S.project, ui = collectUI();
  uiSaveTimer = setTimeout(() => {
    if (p !== S.project) return;
    api("/api/ui", { project: p, ui }, { quiet: true }).catch(() => {});
  }, 400);
}

// ============================================================== step lock ===
const NEEDS = [
  null,
  { id: "stepSource", need: () => "" },
  { id: "stepCloud", need: () => S.state.source ? "" : "needs a photo" },
  { id: "stepShot", need: () => S.state.source ? "" : "needs a photo" },
  { id: "stepGen", need: () => S.state.control_video ? "" : "needs a control video" },
  { id: "stepReview", need: () => S.state.ai_video ? "" : "needs a generated clip" },
];

let openedFor = null, paOpenedFor = null;

/** The step to open when a project opens: the one you left open, or -- the
 *  first time -- the step the project has actually reached. */
function restoreOpenStep(railId, fallbackId) {
  const rail = $(railId);
  let want = null;
  try { want = localStorage.getItem(stepKey(rail)); } catch (e) {}
  const el = want === null ? $(fallbackId) : (want ? $(want) : null);
  if (el && el.closest(".rail") === rail) {
    openStep(el, { animate: false, scroll: true, persist: false });
  } else {
    for (const s of rail.querySelectorAll(":scope > .step.open")) {
      closeStep(s, { animate: false, persist: false });
    }
  }
}

function setSteps() {
  let reached = 1;
  if (S.state.source) reached = 3;
  if (S.state.control_video) reached = 4;
  // Review is the last step: the dataset it builds is the deliverable, and
  // Brush is where it goes next.
  if (S.state.ai_video) reached = 5;
  for (let i = 1; i < NEEDS.length; i++) {
    const spec = NEEDS[i];
    const el = $(spec.id);
    if (!el) continue;
    const why = spec.need();
    el.classList.toggle("locked", !!why);
    const nd = el.querySelector(".needs");
    if (nd.textContent !== why) nd.textContent = why;
    el.classList.toggle("done", i < reached && !why);
    el.classList.toggle("cur", i === reached);
  }
  if (openedFor !== S.project) {
    openedFor = S.project;
    restoreOpenStep("railAuthor", S.state.source ? NEEDS[reached].id : "stepSource");
  }
  const paSteps = [
    // Splat is the entry point: its Upload button must stay pressable, so it
    // only SAYS what it needs and never locks.
    ["paStepSplat", () => ""],
    ["paStepRings", () => S.pass.parent ? "" : "needs an author AI video to match"],
    ["paStepSend", () => S.pass.parent ? "" : "needs an author AI video to match"],
    ["paStepDataset", () => S.state.splat ? "" : "needs a trained splat"],
  ];
  for (const [id, fn] of paSteps) {
    const el = $(id);
    if (!el) continue;
    const why = fn();
    el.classList.toggle("locked", !!why);
    const nd = el.querySelector(".needs");
    if (nd.textContent !== why) nd.textContent = why;
  }
  if (workspace === "passes" && paOpenedFor !== S.project) {
    paOpenedFor = S.project;
    restoreOpenStep("railPasses", "paStepSplat");
  }
}

// ================================================================== state ===
let statePrev = { project: null, ai: null };

async function refreshState() {
  try {
    S.state = await api(`/api/state?project=${encodeURIComponent(S.project)}`,
                        undefined, { quiet: true });
  } catch (e) { return; }
  S.state._dir = (S.projects_dir || "") + "\\" + S.project;
  if (S.state._state_error) {
    logUi(`state.json for ${S.project} is unreadable: ${S.state._state_error}. `
        + `Use Recover from files on disk.`, "error");
  }

  let opened = false;
  if (uiAppliedFor !== S.project) {
    uiAppliedFor = S.project;
    applyUI(S.state.ui || UI_DEFAULTS);
    opened = true;
  }

  if (!S.state.prompt_auto) {
    $("descHint").textContent = "";
    $("paDescHint").textContent = "";
  } else {
    for (const [id, hint] of [["gPrompt", "descHint"], ["paPrompt", "paDescHint"]]) {
      const box = $(id);
      const auto = (S.state.prompt_auto || "").trim();
      if (!box.value.trim() || box.dataset.auto === "1" || box.value.trim() === auto) {
        if (box.value !== S.state.prompt_auto) {
          box.value = S.state.prompt_auto;
          if (!opened) fx.flash(box);
        }
        box.dataset.auto = "1";
      }
      const c = S.state.prompt_auto_cost;
      $(hint).textContent = `${S.state.prompt_auto_model || ""}`
        + (typeof c === "number" ? ` · $${c.toFixed(5)}` : "")
        + (id === "gPrompt"
           ? " · edit freely, motion and backdrop are added on top"
           : " · the camera clause is added on top");
    }
  }

  try {
    S.refs = await api(`/api/refs?project=${encodeURIComponent(S.project)}`,
                       undefined, { quiet: true });
    author.renderRefs();
  } catch (e) { /* refs are optional */ }

  con.renderSent(S.state);
  setSteps();
  author.repaintNow();
  if (opened) author.projectOpened();
  if (workspace === "passes") passes.repaint();
  author.authorFiles();

  // A generation that just brought back a NEW clip opens Review -- only in the
  // author workspace, only on success, and only for a real change of ai_video.
  if (statePrev.project === S.project && S.state.ai_video
      && S.state.ai_video !== statePrev.ai && workspace === "author") {
    const a = jobActive(), l = jobLast();
    const fromGen = (a && a.stage === "generate")
      || (l && l.stage === "generate" && l.kind === "ok" && Date.now() - l.t < 30000);
    if (fromGen) {
      logUi(`${S.state._dir}\\${S.state.ai_video} arrived — opened Review`);
      setTimeout(() => openStep("stepReview"), 250);
    }
  }
  statePrev = { project: S.project, ai: S.state.ai_video || null };
}

/** The job bar is read at a glance: server log lines lose their "file:"
 *  prefix and the absolute path after "@" (both stay in the console). */
function plainMsg(m) {
  return String(m || "").replace(/^file:\s*/, "").replace(/\s+@\s+[A-Za-z]:\\.*$/, "")
    .replace(/\bprojects\/[^/\s]+\//g, "").slice(0, 160);
}

let pollTick = 0;
async function poll() {
  let st;
  try {
    st = await api(`/api/status?project=${encodeURIComponent(S.project)}`,
                   undefined, { quiet: true });
  } catch (e) { return; }
  S.status = st;
  // The job bar speaks for THIS project only. Another project's job is named
  // in #elsewhere, never shown as if it were yours; and in-browser work (a
  // lift, a pass render) owns the bar while it runs.
  const own = !st.project || st.project === S.project;
  const last = jobLast();
  // A client result holds the bar for 6 s; a client FAILURE holds it until the
  // next job starts, so the badge never says "idle" beside "Lift failed".
  const client = (jobActive() && jobActive().kind === "client")
    || (last && last.client && !st.running
        && (last.kind === "fail" || Date.now() - last.t < 6000));
  if (!client) {
    const badge = $("bStage");
    if (st.running && own) {
      badge.className = "jobstage running";
      badge.textContent = "◆ " + (st.stage || "running");
    } else if (st.error && own) {
      badge.className = "jobstage failed";
      badge.textContent = "▲ failed";
    } else if (st.stage && own) {
      badge.className = "jobstage done";
      badge.textContent = st.stage;
    } else {
      badge.className = "jobstage";
      badge.textContent = "idle";
    }
    $("fill").style.width = own ? `${Math.max(0, Math.min(100, +st.pct || 0))}%` : "0%";
    // Only a RUNNING job (or the tick it ends on) speaks here. Idle server
    // chatter -- a dataset plan's warnings -- must not replace the last result.
    if (own && st.msg && (st.running || poll._wasRunning)) $("msg").textContent = plainMsg(st.msg);
  }
  $("elsewhere").textContent = (st.remote_elsewhere || []).length
    ? `▲ also running: ${st.remote_elsewhere.join(", ")}`
    : (st.lane_busy && st.lane_busy.project !== S.project
       ? `▲ ${st.lane_busy.project} is running ${st.lane_busy.stage}` : "");

  // A failure is alerted ONCE, by its stamp.
  if (st.error && (+st.error_at || 0) > lastErrorAt) {
    lastErrorAt = +st.error_at || 0;
    try { localStorage.setItem("img2splat.errAt", String(lastErrorAt)); } catch (e) {}
    logUi(`job failed: ${st.error}`, "error");
  }

  // The viewport card follows THIS project's job.
  const mine = !!(st.running && st.project === S.project);
  if (mine) jobFromStatus(st);
  if (poll._wasMine && !mine) jobEndFromStatus(st);
  poll._wasMine = mine;

  if (poll._wasRunning && !st.running) {
    await refreshState();
    await author.refreshCheckpoints();
    if (workspace === "passes") await passes.refreshPassStatus();
  }
  poll._wasRunning = !!st.running;
  if (st.running) await author.refreshCheckpoints();
  // Files panels refresh themselves while open, only when a file changed.
  if (++pollTick % 2 === 0) {
    author.authorFiles();
    if (workspace === "passes") passes.passFilesTick();
  }
  setSteps();
}

// ================================================================ projects ==
async function refreshProjects() {
  const j = await api("/api/projects");
  S.projects_dir = j.dir;
  const keep = S.project;
  $("projList").innerHTML = (j.projects || []).map(p =>
    `<option value="${esc(p.name)}">${esc(p.name)} — ${esc(p.stage || "image")}</option>`).join("");
  if ((j.projects || []).some(p => p.name === keep)) $("projList").value = keep;
  else if (j.projects && j.projects.length) {
    S.project = j.projects[0].name;
    $("projList").value = S.project;
  }
}

async function openProject(name, { announce = false } = {}) {
  S.project = name;
  try { localStorage.setItem("img2splat.project", name); } catch (e) {}
  uiAppliedFor = null;
  logUi(`opened project ${name} — ${S.projects_dir}\\${name}`);
  author.releaseSplat();
  await refreshState();
  await author.refreshCheckpoints();
  if (workspace === "passes") { paOpenedFor = null; await passes.open(); }
  if (announce) fx.toast(`Opened ${name}`, "ok");
}

// ================================================================== shell ===
function setWorkspace(ws) {
  workspace = ws;
  for (const b of document.querySelectorAll("#wsTabs button")) {
    b.classList.toggle("on", b.dataset.ws === ws);
  }
  $("wsAuthorPane").classList.toggle("on", ws === "author");
  $("wsPassesPane").classList.toggle("on", ws === "passes");
  logUi(`workspace → ${ws}`);
  if (ws === "passes") {
    author.releaseSplat();
    passes.open();
    if (paOpenedFor !== S.project) { paOpenedFor = S.project; restoreOpenStep("railPasses", "paStepSplat"); }
  } else { passes.close(); author.restoreView(); }
}

function initZoom() {
  let z = 100;
  try { z = +localStorage.getItem("img2splat.zoom") || 100; } catch (e) {}
  const apply = () => {
    document.documentElement.style.fontSize = (13 * z / 100).toFixed(2) + "px";
    $("zVal").textContent = z + "%";
    try { localStorage.setItem("img2splat.zoom", String(z)); } catch (e) {}
  };
  $("zIn").onclick = () => { z = Math.min(160, z + 5); apply(); };
  $("zOut").onclick = () => { z = Math.max(70, z - 5); apply(); };
  $("zRst").onclick = () => { z = 100; apply(); };
  apply();
}

async function refreshHealth() {
  try { S.health = await api("/api/health", undefined, { quiet: true }); }
  catch (e) { return; }
  const bad = [];
  if (!S.health.fal_key) bad.push("no FAL_KEY");
  if (!S.health.brush) bad.push("no Brush");
  const chip = $("envChip");
  chip.classList.toggle("bad", bad.length > 0);
  chip.textContent = bad.length ? `▲ ${bad.join(" · ")}` : "Environment";
}

function envDialog() {
  const c = S.health.config || {};
  const rows = [
    ["fal key", S.health.fal_key ? `present (from ${S.health.key_source})` : "NOT SET"],
    ["endpoint", S.health.endpoint],
    ["Brush", `${(c.brush || {}).path || "?"} — ${S.health.brush ? "found" : "NOT FOUND"}`],
    ["depth + lens", "MoGe-2"],
    ["depth (options)", `Depth Anything V2 Small / Large · Depth Anything 3 Metric / Mono — ${S.health.da3 ? "installed" : "DA3 not installed"}`],
    ["projects", (c.projects || {}).path || ""],
    ["python (config)", (c.python || {}).path || ""],
    ["python (running)", (c.python || {}).running || ""],
    ["web", c.web || ""],
    ["config file", `${c.config_file || ""}${c.config_loaded ? "" : "  — NOT LOADED"}`],
    ["log ring", `${c.log_ring || ""} entries, served at /api/log`],
  ];
  dialog({
    title: "Environment",
    okLabel: "Close",
    bodyHtml: `<div class="status">` + rows.map(([k, v]) =>
      `<div class="srow"><div class="k">${esc(k)}</div><div class="v">${esc(String(v))}</div></div>`)
      .join("") + `</div>`,
  });
}

/** ← → Home End Space drive the playhead of the visible workspace -- never
 *  while typing in a field, pressing a button, or inside a modal. */
function onKey(e) {
  if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
  if ($("dlg").classList.contains("on") || $("apvWrap").classList.contains("on")) return;
  if (!["ArrowLeft", "ArrowRight", "Home", "End", " "].includes(e.key)) return;
  const t = e.target && e.target.closest ? e.target : document.body;
  // Typing is never hijacked.
  if (t.closest("textarea, select, [contenteditable=true]")) return;
  if (t.tagName === "INPUT") {
    const scrubRange = t.type === "range" && t.closest(".scrub");
    // A focused slider moves itself with the arrows; only the playhead's own
    // slider also takes Space for play.
    if (t.type !== "checkbox" && !(scrubRange && e.key === " ")) return;
  }
  // Space on an ordinary button or step header presses it, as it should.
  // Arrows, Home and End still scrub from there -- a click must not leave the
  // keyboard dead.
  if (e.key === " " && t.closest("button, [role=button], input") && !t.closest(".scrub")) return;
  const handled = workspace === "passes" ? passes.scrubKey(e) : author.scrubKey(e);
  if (handled) e.preventDefault();
}

// =================================================================== boot ===
async function boot() {
  initDialog();
  initApproval();
  con.init();
  initZoom();

  try { S.project = localStorage.getItem("img2splat.project") || ""; } catch (e) {}

  logUi("frontend booting — reading /api/health, /api/engines, /api/aspects, "
      + "/api/vision_models, /api/matte_models");
  if (!window.gsap) logUi("GSAP did not load (web/vendor/gsap.min.js) — feedback still shows, without motion", "warn");

  await refreshHealth();
  const [eng, asp, vis, mat] = await Promise.all([
    api("/api/engines"), api("/api/aspects"), api("/api/vision_models"),
    api("/api/matte_models"),
  ]);
  S.engines = eng.engines || [];
  S.engineExtras = eng;
  S.presets = asp.presets || [];
  S.visionModels = vis.models || [];
  S.matteModels = mat.models || [];

  author.init();
  passes.init();
  author.fillServerSelects();
  wireDisclosures(document, saveUI);

  UI_DEFAULTS = collectUI();

  await refreshProjects();
  if (!S.project) {
    const first = $("projList").value;
    if (first) S.project = first;
    else {
      const j = await api("/api/project", { name: "project" });
      await refreshProjects();
      S.project = j.project;
      $("projList").value = S.project;
    }
  }
  $("buildTag").textContent = S.health.build || "beta";

  // ---- shell wiring
  for (const b of document.querySelectorAll("#wsTabs button")) {
    b.onclick = () => setWorkspace(b.dataset.ws);
  }
  for (const h of document.querySelectorAll(".step > .head")) {
    const toggle = () => {
      const s = h.parentElement;
      const title = h.querySelector(".title")?.textContent || "a step";
      if (s.classList.contains("open")) { closeStep(s); logUi(`closed ${title}`); }
      else { openStep(s); logUi(`opened ${title}`); }
    };
    h.onclick = toggle;
    h.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
    });
  }
  document.addEventListener("keydown", onKey);

  $("projList").addEventListener("change", () => openProject($("projList").value, { announce: true }));
  // Opening the dropdown re-reads the projects directory.
  $("projList").addEventListener("mousedown", () => { refreshProjects(); });
  $("projList").addEventListener("focus", () => { refreshProjects(); });
  $("btnNew").onclick = async () => {
    const name = await dialog({ title: "New project", input: "project",
                                bodyHtml: "<p>Letters, numbers, - and _ only.</p>",
                                okLabel: "Create" });
    if (!name) return;
    try {
      const j = await api("/api/project", { name });
      logUi(`created project ${j.project} at ${j.dir}`);
      await refreshProjects();
      $("projList").value = j.project;
      await openProject(j.project);
      fx.toast(`Created ${j.project} — drop a photo to start`, "ok");
    } catch (e) { /* reported by api() */ }
  };
  $("btnProjMenu").onclick = () => $("projMenu").classList.toggle("on");
  document.addEventListener("click", e => {
    if (!e.target.closest(".projwrap")) $("projMenu").classList.remove("on");
  });
  $("btnRename").onclick = async () => {
    $("projMenu").classList.remove("on");
    const name = await dialog({ title: `Rename ${S.project}`, input: S.project,
      bodyHtml: "<p>The folder moves and the two places that recorded an "
              + "absolute path — fal_sent and splat_report.md — are repointed.</p>",
      okLabel: "Rename" });
    if (!name) return;
    try {
      const j = await api("/api/project/rename", { project: S.project, name });
      logUi(`renamed to ${j.project}${j.fixed?.length ? ` (repointed ${j.fixed.join(", ")})` : ""}`);
      await refreshProjects();
      await openProject(j.project);
      fx.toast(`Renamed to ${j.project}`, "ok");
    } catch (e) { /* reported by api() */ }
  };
  $("btnClone").onclick = async () => {
    $("projMenu").classList.remove("on");
    const name = await dialog({ title: `Save ${S.project} as`, input: S.project + "-copy",
      bodyHtml: "<p>A byte-for-byte copy of the whole folder. The clone opens "
              + "exactly where the original stands and the two share nothing "
              + "from then on.</p>", okLabel: "Copy" });
    if (!name) return;
    // Cloning is a background JOB: refresh the list AFTER it lands.
    const j = await run(`clone ${S.project} → ${name}`,
                        () => api("/api/project/clone", { project: S.project, name }),
                        { btn: $("btnProjMenu"), stage: "clone", busy: "…" }).catch(() => null);
    if (!j || j.jobError) return;
    await refreshProjects();
    const made = (j && j.project) || name;
    if ([...$("projList").options].some(o => o.value === made)) {
      $("projList").value = made;
      await openProject(made, { announce: true });
    } else {
      logUi(`cloned, but ${made} is not in the list yet — press the project `
          + `dropdown to re-read the directory`, "warn");
    }
  };
  $("btnRescan").onclick = async () => {
    $("projMenu").classList.remove("on");
    try {
      const j = await api("/api/project/rescan", { name: S.project });
      logUi(`recovered ${j.project} from the files on disk — stage ${j.stage}`);
      await refreshState();
      fx.toast(`Recovered ${j.project} from disk — stage ${j.stage}`, "ok");
    } catch (e) { /* reported by api() */ }
  };
  $("btnExport").onclick = () => {
    $("projMenu").classList.remove("on");
    logUi(`exporting ${S.project} as a .zip`);
    fx.toast(`Building ${S.project}.zip — the download starts when it is ready`, "info", 5000);
    location.href = `/api/project/export?project=${encodeURIComponent(S.project)}`;
  };
  $("btnImport").onclick = () => { $("projMenu").classList.remove("on"); $("importFile").click(); };
  $("importFile").onchange = async e => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    fx.toast(`Importing ${f.name}…`, "info");
    const r = await fetch("/api/project/import", { method: "POST", body: f });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) {
      logUi(`import failed: ${j.error}`, "error");
      fx.toast(`Import failed — ${j.error || r.status}`, "err");
      return;
    }
    logUi(`imported ${j.project} — ${j.files} files`);
    await refreshProjects();
    await openProject(j.project);
    fx.toast(`Imported ${j.project} — ${j.files} files`, "ok");
  };
  $("btnPrune").onclick = async () => {
    $("projMenu").classList.remove("on");
    const ok = await dialog({ title: "Delete empty projects",
      bodyHtml: "<p>Only folders with no source image and nothing derived "
              + "recorded are removed. Anything holding work is left alone.</p>",
      okLabel: "Delete" });
    if (!ok) return;
    try {
      const j = await api("/api/projects/prune", {});
      logUi(`pruned ${j.count} empty projects${j.count ? ": " + j.removed.join(", ") : ""}`);
      await refreshProjects();
      fx.toast(j.count ? `Deleted ${j.count} empty project${j.count === 1 ? "" : "s"}` : "No empty projects to delete", "ok");
    } catch (e) { /* reported by api() */ }
  };
  $("btnFreeVram").onclick = async () => {
    $("projMenu").classList.remove("on");
    try {
      const j = await api("/api/free_vram", {});
      logUi(`free VRAM: ${JSON.stringify(j)}`);
      fx.toast(`Released ${(+j.released_gb || 0).toFixed(2)} GB — ${(+j.free_gb || 0).toFixed(1)} GB free`, "ok");
    } catch (e) { /* reported by api() */ }
  };
  $("envChip").onclick = envDialog;

  // persistence: every rail control, both workspaces
  for (const id of ["railAuthor", "railPasses"]) {
    $(id).addEventListener("input", saveUI, true);
    $(id).addEventListener("change", saveUI, true);
  }
  author.setDirtyHook(saveUI);
  passes.setDirtyHook(saveUI);
  onStateChange(async () => {
    await refreshState();
    await author.refreshCheckpoints();
    if (workspace === "passes") await passes.refreshPassStatus();
  });

  await openProject(S.project);
  setWorkspace("author");

  setInterval(poll, 1200);
  setInterval(refreshHealth, 30000);
  poll();

  logUi(`ready — project ${S.project} at ${S.projects_dir}\\${S.project}`);
}

window.addEventListener("unhandledrejection", ev => {
  const msg = (ev.reason && ev.reason.message) || String(ev.reason || "");
  if (msg) logUi("unhandled: " + msg, "error");
});

boot().catch(e => {
  console.error(e);
  logUi("BOOT FAILED: " + (e.stack || e.message), "error");
  const b = $("bStage");
  if (b) { b.className = "jobstage failed"; b.textContent = "▲ boot failed"; }
  renderStatus("stSource", "Source", [{ k: "boot", v: e.message, problem: true }]);
});

// expose for console debugging -- named, not a grab bag
window.__img2splat = { S, author, passes, con };
