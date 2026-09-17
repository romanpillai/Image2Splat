// Shared plumbing: the fetch wrapper, the error-attribution pattern, the
// status-block renderer, and the small bits every workspace needs.
//
// Nothing in here draws a workspace. It exists so that author.js and passes.js
// cannot each grow their own copy of "how do I call the server and report what
// happened", which is the shape of most of the bugs the predecessor fixed.

import * as fx from "./fx.js";

export const $ = id => document.getElementById(id);
export { fx };

/** `+el.value || d` turns a legitimate 0 into d. Where 0 is meaningful
 *  (relief, sweep, aim height) read it null-safely instead, from ONE place. */
export const num = (id, d) => {
  const el = $(id);
  if (!el) return d;
  const v = String(el.value).trim();
  return v === "" ? d : (Number.isFinite(+v) ? +v : d);
};

export const esc = s => String(s == null ? "" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");

export const mb = b => `${((+b || 0) / 1e6).toFixed(2)} MB`;
export const clock = (t) => {
  const d = t ? new Date(t * 1000) : new Date();
  return [d.getHours(), d.getMinutes(), d.getSeconds()]
    .map(n => String(n).padStart(2, "0")).join(":");
};

// ---------------------------------------------------------- the UI log ----
// One log, two sources: what YOU changed, and what the server did, interleaved
// by time. It is how a bad render is read back to the setting that caused it.
// Client-side lines go into the same ring the server lines land in, tagged
// "ui", so the console shows one stream.
const uiSinks = [];
export function onUiLog(fn) { uiSinks.push(fn); }
/** Log a UI action. Say the WORDS ON SCREEN, never element ids. */
export function logUi(msg, level = "ui") {
  for (const fn of uiSinks) {
    try { fn({ level, source: "ui", msg: String(msg), t: Date.now() / 1000 }); }
    catch (e) { /* a broken sink must not break the action */ }
  }
}

// ------------------------------------------------------------ the state ---
// One module-level record of what the server says this project is. Written by
// the poller in app.js, read everywhere.
export const S = {
  project: "",
  state: {},          // /api/state
  status: {},         // /api/status
  engines: [],        // /api/engines
  engineExtras: {},   // wan_resolutions etc.
  matteModels: [],
  visionModels: [],
  presets: [],        // /api/aspects
  health: {},
  pass: {},           // /api/pass/status
  splats: { exports: [], dir: "" },
  refs: { refs: [], caps: {} },
};

const changeSinks = [];
export function onStateChange(fn) { changeSinks.push(fn); }
/** Returns a promise, so a flow that needs the refreshed state ("did the
 *  approval land?") can wait for it. Callers that do not care ignore it. */
export function fireStateChange(why) {
  return Promise.all(changeSinks.map(fn =>
    Promise.resolve().then(() => fn(why)).catch(e => console.error(e))));
}

// ----------------------------------------------------------------- api ----
let apiErrorShown = false;

function showApiError(msg) {
  const m = $("msg");
  if (m) m.textContent = msg;
  const b = $("bStage");
  if (b) { b.textContent = "refused"; b.className = "jobstage failed"; }
  logUi("SERVER REFUSED: " + msg, "error");
  // A refusal is shown where the eye is, not only in the job bar.
  fx.toast(msg, "err");
  apiErrorShown = true;
}

function shownError(msg) {
  const e = new Error(msg);
  e.shown = true;
  return e;
}

/** Every call the tool makes goes through here.
 *
 * api() throws on a non-OK response and most handlers are bare
 * `await api(...)`, so a rejected promise used to vanish: the server said
 * "busy" or "no camera rig recorded" and the button simply did nothing.
 * Report at the one place every call passes through.
 */
export async function api(path, body, opts = {}) {
  const isPost = body !== undefined;
  let r;
  try {
    r = await fetch(path, !isPost ? {} : {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    const msg = `cannot reach the server (${e.message}) - is it still running?`;
    if (!opts.quiet) showApiError(msg);
    throw opts.quiet ? new Error(msg) : shownError(msg);
  }
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    let msg = j.error || `${r.status} ${path}`;
    if (Array.isArray(j.detail)) {
      // FastAPI validation: name the field and what it wanted, or the user
      // just sees "422" and has no idea which control is at fault.
      msg = j.detail.map(d => {
        const f = (d.loc || []).filter(x => x !== "body").join(".");
        return `${f || "input"}: ${d.msg}`
             + (d.input !== undefined ? ` (got ${JSON.stringify(d.input)})` : "");
      }).join("; ");
    } else if (typeof j.detail === "string") {
      msg = j.detail;
    }
    if (r.status === 409) {
      msg = "another job is already running - wait for it to finish, then try "
          + "again. (" + msg + ")";
    }
    if (!opts.quiet) showApiError(msg);
    throw opts.quiet ? new Error(msg) : shownError(msg);
  }
  if (apiErrorShown && !opts.quiet) {
    apiErrorShown = false;
    const m = $("msg"); if (m) m.textContent = "";
  }
  return j;
}

/** Raw body POST (uploads). Same reporting. */
export async function apiRaw(path, blob) {
  let r;
  try {
    r = await fetch(path, { method: "POST", body: blob });
  } catch (e) {
    showApiError(`cannot reach the server (${e.message})`);
    throw shownError(e.message);
  }
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { showApiError(j.error || `${r.status} ${path}`); throw shownError(j.error || String(r.status)); }
  return j;
}

// -------------------------------------------- per-button error attribution -
// /api/status keeps the last error until the NEXT job starts, so any code that
// reads st.error after an action reports whatever failed most recently -- not
// necessarily what this action did. That is how a fal failure from a send kept
// reappearing under Preview softening: a different button, a different job,
// the same dead error, with no way to clear it.
//
// The server stamps each failure with error_at. Take the stamp BEFORE
// starting; anything at or below it belongs to somebody else.

export async function errorMark() {
  try {
    const st = await api("/api/status?project=" + encodeURIComponent(S.project),
                         undefined, { quiet: true });
    return +st.error_at || 0;
  } catch (e) { return 0; }
}

/** Resolve when the job finishes: with its error text, or "" if it succeeded
 *  (or if the only error on record predates `mark`). */
export let lastJobStatus = null;
export function awaitJob(mark) {
  const project = S.project;
  return new Promise(resolve => {
    const t = setInterval(async () => {
      let st;
      try {
        st = await api("/api/status?project=" + encodeURIComponent(project),
                       undefined, { quiet: true });
      } catch (e) { return; }        // transient; the next tick retries
      if (st.running && st.project === project) { jobFromStatus(st); return; }
      if (st.running) return;
      clearInterval(t);
      lastJobStatus = st;
      resolve((+st.error_at || 0) > mark ? (st.error || "") : "");
    }, 900);
  });
}

// ------------------------------------------------------------ job tracker -
// ONE working card at a time, over the 3D view. A job is started by run() or
// runDetached() the moment its button is pressed, by the poll when it finds a
// job this page did not start (a refresh mid-render), or by client work that
// has no server job at all (the lift, a pass render). Whoever owns it ends it,
// exactly once; the poll only ends jobs nobody else is waiting on.
export const STAGE = {
  orbit: { work: "Rendering the control video", ok: "Control video rendered", fail: "Render failed" },
  fov: { work: "Measuring the lens", ok: "Lens measured", fail: "Lens measurement failed" },
  soften: { work: "Softening the control video", ok: "Softened clip built", fail: "Softening failed" },
  generate: { work: "Generating with fal", ok: "AI video ready", fail: "Generation failed", long: true },
  describe: { work: "Describing the subject", ok: "Description written", fail: "Auto-describe failed" },
  matte: { work: "Cutting the matte", ok: "Matte cut", fail: "Matte failed" },
  retime: { work: "Retiming the clip", ok: "Clip retimed", fail: "Retime failed" },
  splat: { work: "Training the splat", ok: "Splat trained", fail: "Training failed", stop: "Training stopped", long: true },
  "dataset-complete": { work: "Building dataset_complete", ok: "dataset_complete built", fail: "Dataset build failed", long: true },
  clone: { work: "Copying the project", ok: "Project copied", fail: "Copy failed", long: true },
};

const JT = { cur: null, seq: 0, last: null };
const VPS = () => [...document.querySelectorAll(".vp")];
export const cap1 = t => (t = String(t || "")) && t[0].toUpperCase() + t.slice(1);

/** Client work (a lift, a pass render) has no server job, so the job bar is
 *  told directly -- otherwise it sits on the last server stage and lies. */
function jobBar(text, cls, msg) {
  const b = $("bStage"), m = $("msg");
  if (b) { b.className = "jobstage" + (cls ? " " + cls : ""); b.textContent = text; }
  if (m && msg != null) m.textContent = msg;
}

export function jobStart({ title, detail = "", long = false, stage = "",
                           kind = "server", mark = 0, owner = "", detailFn = null,
                           compact = false, delay = 380 }) {
  const tok = ++JT.seq;
  if (compact && delay < 650) delay = 650;
  JT.cur = { tok, title, detail, long, stage, kind, mark, owner, detailFn, compact,
             delay, pct: 0, t0: Date.now() };
  fx.vpWork(VPS(), JT.cur);
  if (kind === "client") jobBar("◆ " + (stage || "working"), "running", `${title}${detail ? " — " + detail : ""}`);
  return tok;
}

export function jobUpdate(patch, tok) {
  if (!JT.cur || (tok && JT.cur.tok !== tok)) return;
  Object.assign(JT.cur, patch);
  fx.vpWork(VPS(), JT.cur);
}

/** o = {kind: ok|fail|stop, title, detail, quiet}. quiet + ok clears the card
 *  without a celebration -- for a step that another step follows at once. */
export function jobEnd(o, tok) {
  const c = JT.cur;
  if (!c || (tok && c.tok !== tok)) return false;
  JT.last = { stage: c.stage, kind: o.kind, t: Date.now(), client: c.kind === "client" };
  JT.cur = null;
  if (c.kind === "client") {
    jobBar(o.kind === "fail" ? "▲ failed" : (c.stage || "done"),
           o.kind === "fail" ? "failed" : "done", `${o.title}${o.detail ? " — " + o.detail : ""}`);
  }
  if (o.quiet && o.kind === "ok") { fx.vpClear(VPS()); return true; }
  // Work that was over before its card even appeared reports as a pill, not
  // a celebration over the model. Failures are always the full card.
  const compact = o.compact ?? (c.compact || (o.kind === "ok" && !fx.vpWorkShown(VPS())));
  fx.vpOutcome(VPS(), { ...o, compact });
  return true;
}
export const jobActive = () => JT.cur;

/** An outcome with no working phase before it (an approval is instant). If a
 *  job owns the viewport card it is not interrupted: a toast says it instead. */
export function flashOutcome(o) {
  if (JT.cur) { fx.toast(o.title + (o.detail ? ` — ${o.detail}` : ""), o.kind === "fail" ? "err" : "ok"); return; }
  fx.vpOutcome(VPS(), { compact: false, ...o });
}
export const jobLast = () => JT.last;

function statusDetail(st, title = "") {
  const ph = cap1(st.phase || "");
  if (+st.steps > 0) {
    const noun = /frame/i.test(ph) ? "Frame" : (ph || "Step");
    return `${noun} ${st.step} of ${st.steps}`;
  }
  if (ph && !String(title).toLowerCase().includes(ph.toLowerCase())) return `${ph}…`;
  return String(st.msg || "").replace(/^file: /, "").slice(0, 120);
}

/** Called on every poll tick with /api/status. Starts a card for a job this
 *  page did not start, and keeps any server card's detail current. */
export function jobFromStatus(st) {
  if (!st || !st.running || st.project !== S.project) return;
  const L = STAGE[st.stage] || { work: cap1(st.stage || "working") };
  if (!JT.cur) {
    jobStart({ title: L.work, stage: st.stage, long: !!L.long, kind: "server",
               mark: +st.error_at || 0 });
  } else if (JT.cur.kind === "server" && !JT.cur.stage) {
    JT.cur.stage = st.stage;
  }
  if (JT.cur && JT.cur.kind === "server") {
    const extra = JT.cur.detailFn ? JT.cur.detailFn(st) : null;
    if (extra && typeof extra === "object") {
      jobUpdate({ detail: extra.detail || statusDetail(st, JT.cur.title), pct: +extra.pct || 0 });
    } else {
      jobUpdate({ detail: extra || statusDetail(st, JT.cur.title), pct: +st.pct || 0 });
    }
  }
}

/** A job the poll saw end. Only ends a card NOBODY is awaiting -- run() owns
 *  its own ending, with its own words. */
export function jobEndFromStatus(st) {
  const c = JT.cur;
  if (!c || c.kind !== "server" || c.owner) return;
  const L = STAGE[c.stage] || STAGE[st.stage] || {};
  if (st.error && (+st.error_at || 0) > (c.mark || 0)) {
    jobEnd({ kind: "fail", title: L.fail || "Failed", detail: shortErr(st.error) });
  } else if (st.aborted) {
    jobEnd({ kind: "stop", title: L.stop || "Stopped" });
  } else {
    jobEnd({ kind: "ok", title: L.ok || "Done" });
  }
}

export const shortErr = e => {
  const t = String(e || "").replace(/^[A-Za-z]+Error: /, "");
  return t.length > 150 ? t.slice(0, 147) + "…" : t;
};

// ------------------------------------------------------- the pressed button -
// run() gives feedback ON THE BUTTON THAT STARTED IT without every handler
// having to pass itself in: the last button pressed, if it was pressed just
// now and is not inside a modal (those close; the step's own button is passed
// explicitly from the approval flows).
let lastPress = null;
document.addEventListener("click", e => {
  const b = e.target && e.target.closest ? e.target.closest("button") : null;
  if (b) lastPress = { b, t: performance.now() };
}, true);
export function pressedButton(maxAge = 1500) {
  if (!lastPress || performance.now() - lastPress.t > maxAge) return null;
  const b = lastPress.b;
  if (!b.isConnected || b.closest(".dlg, .apvwrap")) return null;
  return b;
}

/** THE way a button reports a result.
 *
 * Stamps first, runs, then waits for the job it started and reports only that
 * job's failure. Making this the only path means a handler added later cannot
 * forget the pattern -- which is exactly what §4.7 of the UX spec asked for.
 */
/** Submit now, watch the job in the BACKGROUND.
 *
 * For anything opened from the approval window. That window has to close as
 * soon as the request is ACCEPTED, not when the work finishes -- a fal
 * generation is minutes, and awaiting the whole job left the window sitting on
 * "starting..." for all of it with no way out.
 *
 * A failure to SUBMIT still throws, so the window can show it and stay open.
 * A failure DURING the job lands in the console and the job bar, which are on
 * screen the whole time anyway.
 */
export async function runDetached(label, fn, opts = {}) {
  const btn = opts.btn || null;
  logUi(`${label} — sent`);
  if (btn) fx.btnBusy(btn, opts.busy || "Sending…");
  const mark = await errorMark();
  let out;
  try {
    out = await fn();
  } catch (e) {
    logUi(`${label} — refused: ${e.message}`, "error");
    if (btn) fx.btnDone(btn, false, "Refused");
    throw e;                       // the approval window renders this
  }
  if (out && out.started === false) {
    logUi(`${label} — done` + (out.cached ? " (already built)" : ""));
    if (btn) fx.btnDone(btn, true, "Done");
    return out;
  }
  const L = STAGE[opts.stage] || {};
  const tok = jobStart({ title: opts.working || L.work || cap1(label),
                         stage: opts.stage || "", long: opts.long ?? !!L.long,
                         kind: "server", mark, owner: "detached",
                         detailFn: opts.detailFn || null });
  if (btn) fx.btnBusy(btn, opts.busyRunning || "Running…");
  fx.toast(`${cap1(label)} — started. Progress is on the viewport.`, "info");
  // Deliberately NOT awaited: the caller returns, the window closes, and this
  // keeps reporting until the job ends.
  (async () => {
    const err = await awaitJob(mark);
    const st = lastJobStatus || {};
    if (err) {
      logUi(`${label} — FAILED: ${err}`, "error");
      if (btn) fx.btnDone(btn, false, "Failed");
      jobEnd({ kind: "fail", title: opts.fail || L.fail || `${cap1(label)} failed`,
               detail: shortErr(err) }, tok);
      fireStateChange(label);
    } else if (st.aborted) {
      logUi(`${label} — stopped`, "warn");
      if (btn) fx.btnDone(btn, true, "Stopped");
      jobEnd({ kind: "stop", title: opts.stop || L.stop || "Stopped" }, tok);
      fireStateChange(label);
    } else {
      logUi(`${label} — finished`);
      if (btn) fx.btnDone(btn, true, opts.okBtn || "Done");
      await fireStateChange(label);
      jobEnd({ kind: "ok", title: opts.ok || L.ok || `${cap1(label)} — done`,
               detail: opts.okDetail ? opts.okDetail() : "" }, tok);
      if (opts.onOk) { try { await opts.onOk(); } catch (e) { console.error(e); } }
    }
  })();
  return out;
}

/** THE way a button reports a result.
 *
 * Stamps first, runs, then waits for the job it started and reports only that
 * job's failure. Feedback lands in three places at once: the button (busy ->
 * done / failed), the viewport (working card -> outcome) and the console.
 *
 * opts: wait, btn (null = none), busy, working, ok, fail, stage, long, quiet,
 *       track (false = no viewport card), after (async, runs before the
 *       outcome is shown; its string return becomes the outcome detail).
 * The returned object carries jobError ("" on success).
 */
export async function run(label, fn, opts = {}) {
  const { wait = true, track = true } = opts;
  const btn = opts.btn === null ? null : (opts.btn || pressedButton());
  logUi(`${label} — sent`);
  if (btn) fx.btnBusy(btn, opts.busy || "Working…");
  const mark = await errorMark();
  let out;
  try {
    out = await fn();
  } catch (e) {
    logUi(`${label} — refused: ${e.message}`, "error");
    if (btn) fx.btnDone(btn, false, "Refused");
    if (!e.shown) fx.toast(`${cap1(label)}: ${e.message}`, "err");
    throw e;
  }
  if (!wait || (out && out.started === false)) {
    logUi(`${label} — done` + (out && out.cached ? " (already built)" : ""));
    if (btn) fx.btnDone(btn, true, out && out.cached ? "Up to date" : (opts.okBtn || "Done"));
    return Object.assign(out || {}, { jobError: "" });
  }
  const L = STAGE[opts.stage] || {};
  const tok = track ? jobStart({ title: opts.working || L.work || cap1(label),
                                 stage: opts.stage || "", long: opts.long ?? !!L.long,
                                 kind: "server", mark, owner: "run" }) : 0;
  const err = await awaitJob(mark);
  const st = lastJobStatus || {};
  if (err) {
    logUi(`${label} — FAILED: ${err}`, "error");
    const m = $("msg"); if (m) m.textContent = err;
    if (btn) fx.btnDone(btn, false, "Failed");
    if (track) jobEnd({ kind: "fail", title: opts.fail || L.fail || `${cap1(label)} failed`,
                        detail: shortErr(err) }, tok);
  } else if (st.aborted) {
    if (btn) fx.btnDone(btn, true, "Stopped");
    if (track) jobEnd({ kind: "stop", title: opts.stop || L.stop || "Stopped" }, tok);
  } else {
    logUi(`${label} — finished`);
    let detail = "";
    if (opts.after) { try { detail = (await opts.after()) || ""; } catch (e) { console.error(e); } }
    if (btn) fx.btnDone(btn, true, opts.okBtn || "Done");
    if (track) jobEnd({ kind: "ok", title: opts.ok || L.ok || `${cap1(label)} — done`,
                        detail, quiet: !!opts.quiet }, tok);
  }
  return Object.assign(out || {}, { jobError: err || "" });
}

// ------------------------------------------------------- status blocks ----
// A fixed grid of label/value rows. Always present, always the same rows in
// the same order. Values change; the shape does not. This replaces the 30
// update*Hint() functions that computed a fact and rendered it as prose.
//
// A row whose value is a PROBLEM gets the maroon left rule, and the step's
// primary button goes to its blocked state carrying the problem count. That is
// the whole point: a red hint is not a hint.

/**
 * rows: [{k, v, from, problem, fix:{label, onClick}}]
 * Anything with `problem: true` counts toward the block's problem total.
 * Returns the number of problems, so the caller can gate its primary button.
 */
/** A status block whose INFORMATION folds away and whose PROBLEMS never do.
 *
 * Same contract as renderStatus -- it returns the problem count, which is what
 * gate() consumes -- so nothing about blocking changes. The only difference is
 * where a row is drawn: anything with `problem`, or carrying a `fix` button,
 * stays on screen; the rest goes behind a "?" that remembers whether it was
 * open.
 *
 * The rule this enforces: hide the RENDERING, never the COMPUTATION. Every row
 * is still built and still counted. A problem folded out of sight would grey a
 * button out with no visible reason, which is worse than the clutter it saves.
 */
export function renderStatusFolded(elId, caption, rows, detailLabel) {
  const el = $(elId);
  if (!el) return 0;
  const live = [], detail = [];
  for (const r of rows) {
    if (!r) continue;
    (r.problem || r.fix || r.live ? live : detail).push(r);
  }
  const key = "img2splat.detail." + elId;
  let open = false;
  try { open = localStorage.getItem(key) === "1"; } catch (e) { /* private */ }

  const n = renderStatus(elId, caption, live);
  if (detail.length) {
    const wrap = document.createElement("div");
    wrap.className = "detail" + (open ? " on" : "");
    const btn = document.createElement("button");
    btn.className = "detailhead";
    btn.type = "button";
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    btn.innerHTML = `<span class="q">?</span><span>`
      + esc(detailLabel || "detail") + `</span>`
      + `<span class="n">${detail.length}</span><span class="chev" aria-hidden="true"></span>`;
    const body = document.createElement("div");
    body.className = "detailbody";
    // Reuse renderStatus so a detail row looks exactly like a live one.
    const tmp = document.createElement("div");
    tmp.id = elId + "__detail";
    body.appendChild(tmp);
    wrap.appendChild(btn);
    wrap.appendChild(body);
    el.appendChild(wrap);
    renderStatus(tmp.id, "", detail);
    const cap = tmp.querySelector(".cap");
    if (cap) cap.remove();                 // the "?" button is the caption
    btn.onclick = () => {
      const nowOpen = !wrap.classList.contains("on");
      wrap.classList.toggle("on", nowOpen);
      btn.setAttribute("aria-expanded", nowOpen ? "true" : "false");
      if (nowOpen) fx.expand(body);
      try { localStorage.setItem(key, nowOpen ? "1" : "0"); } catch (e) {}
    };
  }
  return n;
}

export function renderStatus(elId, caption, rows) {
  const el = $(elId);
  if (!el) return 0;
  const fixes = [];
  let n = 0;
  const html = caption ? [`<div class="cap">${esc(caption)}</div>`] : [];
  for (const r of rows) {
    if (!r) continue;
    if (r.problem) n++;
    const id = r.fix ? `fix_${elId}_${fixes.length}` : "";
    if (r.fix) fixes.push({ id, onClick: r.fix.onClick });
    html.push(
      `<div class="srow${r.problem ? " problem" : ""}${r.note ? " note" : ""}">`
      + `<div class="k">${esc(r.k)}</div>`
      + `<div class="v">${r.html ? r.v : esc(r.v == null || r.v === "" ? "—" : r.v)}`
      + (r.from ? ` <em>${esc(r.from)}</em>` : "")
      + (r.fix ? ` <button class="fix" id="${id}"${r.fix.disabled ? " disabled" : ""}>${esc(r.fix.label)}</button>` : "")
      + `</div></div>`);
  }
  el.innerHTML = html.join("");
  for (const f of fixes) {
    const b = $(f.id);
    if (b) b.onclick = f.onClick;
  }
  return n;
}

/** Put a primary button into its normal or its blocked state.
 *
 * Blocked is NOT "disabled and grey": it keeps full contrast, carries the
 * problem count and cannot be pressed. Disabled-by-opacity makes the label
 * unreadable exactly when the user needs to know why it is off.
 */
export function gate(btnId, problems, label) {
  const b = $(btnId);
  if (!b) return;
  b.dataset.label = label || b.dataset.label || b.textContent;
  b.dataset.gateN = String(problems || 0);
  if (!b._gateWired) {
    b._gateWired = true;
    b.addEventListener("fx:restored", () => applyGate(b));
  }
  // Busy / done / failed belong to the button until they time out; the gate
  // is re-applied the moment they do.
  if (fx.btnTransient(b)) return;
  applyGate(b);
}

function applyGate(b) {
  const problems = +b.dataset.gateN || 0;
  const label = b.dataset.label || "";
  const text = problems > 0
    ? `▲ ${problems} problem${problems === 1 ? "" : "s"} — ${label}` : label;
  b.classList.toggle("blocked", problems > 0);
  b.disabled = problems > 0;
  if (b.textContent !== text) b.textContent = text;
  b.title = problems > 0 ? "The rows marked ▲ above say what is in the way" : "";
}

// -------------------------------------------------------------- accordion -
// ONE open step per rail. The open step turns into a white card; which one is
// open is remembered per project, so a refresh lands where you were.
export const stepKey = rail => `img2splat.open.${rail.id}.${S.project}`;

function setCaret(step) {
  const h = step.querySelector(":scope > .head");
  if (h) h.setAttribute("aria-expanded", step.classList.contains("open") ? "true" : "false");
}

function scrollStep(step) {
  const rail = step.closest(".rail");
  if (!rail) return;
  const top = step.offsetTop;
  const h = Math.min(step.offsetHeight, rail.clientHeight);
  if (top < rail.scrollTop || top + h > rail.scrollTop + rail.clientHeight) {
    rail.scrollTo({ top: Math.max(0, top - 2), behavior: fx.reduced() ? "auto" : "smooth" });
  }
}

export function openStep(step, { animate = true, scroll = true, persist = true } = {}) {
  if (typeof step === "string") step = $(step);
  if (!step) return;
  const rail = step.closest(".rail");
  if (rail) {
    for (const o of rail.querySelectorAll(":scope > .step.open")) {
      if (o !== step) closeStep(o, { animate, persist: false });
    }
  }
  step.classList.remove("closing");
  if (!step.classList.contains("open")) {
    step.classList.add("open");
    setCaret(step);
    fx.syncAllRanges(step);
    const body = step.querySelector(":scope > .body");
    if (animate) fx.expand(body, () => { if (scroll) scrollStep(step); });
    else if (scroll) scrollStep(step);
  } else if (scroll) {
    scrollStep(step);
  }
  if (persist && rail) { try { localStorage.setItem(stepKey(rail), step.id); } catch (e) {} }
}

export function closeStep(step, { animate = true, persist = true } = {}) {
  if (typeof step === "string") step = $(step);
  if (!step || !step.classList.contains("open")) return;
  const body = step.querySelector(":scope > .body");
  step.classList.remove("open");
  setCaret(step);
  if (animate) {
    step.classList.add("closing");
    fx.collapse(body, () => step.classList.remove("closing"));
  }
  const rail = step.closest(".rail");
  if (persist && rail) { try { localStorage.setItem(stepKey(rail), ""); } catch (e) {} }
}

// ---------------------------------------------------------- slider pairs --
/** A slider whose readout is a real, typeable number box.
 *
 * The NUMBER BOX keeps the id, so saved projects and every num() read find it
 * unchanged; the range (id + "_r", class "mirror") only mirrors it and is never
 * persisted. Typing past either end WIDENS the range instead of clamping the
 * value -- a slider capped below a scene's real scale is how the isolate sphere
 * once became unusable.
 */
export function pairSlider(numId) {
  const n = $(numId), r = $(numId + "_r");
  if (!n || !r) return;
  const dec = () => Math.max(0, (String(r.step).split(".")[1] || "").length);
  r.addEventListener("input", () => {
    n.value = (+r.value).toFixed(dec());
    n.dispatchEvent(new Event("input", { bubbles: true }));
  });
  r.addEventListener("change", () => n.dispatchEvent(new Event("change", { bubbles: true })));
  const fromNum = () => {
    if (String(n.value).trim() === "") return;
    const v = +n.value;
    if (!isFinite(v)) return;
    if (v < +r.min) r.min = String(v);
    if (v > +r.max) r.max = String(v);
    if (+r.value !== v) r.value = String(v);
    fx.syncRange(r);
  };
  n.addEventListener("input", fromNum);
  n.addEventListener("change", fromNum);
  n._syncRange = fromNum;
  fromNum();
}

/** Re-span a paired slider around the project's scale, keeping its value. */
export function spanSlider(numId, lo, hi) {
  const n = $(numId), r = $(numId + "_r");
  if (!n || !r || !(hi > lo)) return;
  const v = +n.value;
  r.min = String(Math.min(lo, isFinite(v) ? v : lo));
  r.max = String(Math.max(hi, isFinite(v) ? v : hi));
  if (n._syncRange) n._syncRange();
}

// ------------------------------------------------------------- disclosure -
export function wireDisclosures(scopeEl, onToggle) {
  scopeEl.querySelectorAll(".adv > .advhead").forEach(h => {
    h.onclick = () => {
      const box = h.parentElement;
      box.classList.toggle("open");
      h.querySelector(".pm").textContent = box.classList.contains("open") ? "−" : "+";
      h.setAttribute("aria-expanded", box.classList.contains("open") ? "true" : "false");
      if (box.classList.contains("open")) fx.expand(box.querySelector(".advbody"));
      logUi(`${box.classList.contains("open") ? "opened" : "closed"} the `
            + `Advanced section of ${box.closest(".step")?.querySelector(".title")?.textContent || "a step"}`);
      if (onToggle) onToggle();
    };
    // Count what is inside so "Advanced (n)" is honest.
    const n = h.parentElement.querySelectorAll(
      ".advbody :is(input:not([type=hidden]):not(.mirror):not(.hidden),select,textarea,button)").length;
    h.querySelector(".n").textContent = `(${n})`;
  });
}

// ----------------------------------------------------------------- dialog -
// Replaces every native confirm()/prompt(). The confirm TEXT on retime is
// load-bearing and survives verbatim; only the widget changed.
let dlgResolve = null;
export function dialog({ title, bodyHtml, input, okLabel = "OK" }) {
  return new Promise(resolve => {
    dlgResolve = resolve;
    $("dlgTitle").textContent = title;
    $("dlgBody").innerHTML = bodyHtml || "";
    const inp = $("dlgInput");
    if (input === undefined) inp.classList.add("hidden");
    else { inp.classList.remove("hidden"); inp.value = input; }
    $("dlgOk").textContent = okLabel;
    $("dlg").classList.add("on");
    if (input !== undefined) setTimeout(() => inp.focus(), 0);
  });
}
function dlgClose(v) {
  $("dlg").classList.remove("on");
  const r = dlgResolve; dlgResolve = null;
  if (r) r(v);
}
export function initDialog() {
  $("dlgCancel").onclick = () => dlgClose(null);
  $("dlgOk").onclick = () =>
    dlgClose($("dlgInput").classList.contains("hidden") ? true : $("dlgInput").value);
  $("dlg").addEventListener("click", e => { if (e.target === $("dlg")) dlgClose(null); });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && $("dlg").classList.contains("on")) dlgClose(null);
    if (e.key === "Enter" && $("dlg").classList.contains("on")) $("dlgOk").click();
  });
}

// ------------------------------------------------------------------ misc --
/** Cache-buster. Files are replaced BY NAME on a re-render; without this the
 *  browser shows the previous run's video and it looks like a failed render. */
export const bust = (url, v) => url + (url.includes("?") ? "&" : "?") + "v=" + (v || Date.now());

export const isImg = u => /\.(png|jpe?g|webp)(\?|$)/i.test(u || "");

/** One media card body -- shared by the approval window and the files panel. */
export function mediaHtml(url, alt = "") {
  if (!url) return '<div class="none">no preview for this input</div>';
  if (isImg(url)) return `<img src="${esc(url)}" alt="${esc(alt)}">`;
  return `<video src="${esc(url)}" muted loop autoplay playsinline></video>`;
}
