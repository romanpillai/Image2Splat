// The console.
//
// This is a headline feature of the beta, not a debug aid, so it is a real
// panel rather than a scrolling text box:
//
//   * It polls /api/log by SEQUENCE NUMBER. `since` is the last seq rendered,
//     so the client gets exactly what it has not seen. Indexes would shift as
//     the ring rolls and two entries can share a millisecond.
//   * The reply carries skipped / oldest_seq / newest_seq / held / capacity.
//     All of it is shown: a client that fell behind must SAY it fell behind
//     rather than silently miss lines, and the ring's fill level is the
//     difference between "quiet" and "overflowing".
//   * Levels are load-bearing. info / warn / error / io come from the server;
//     "ui" is added here for what was clicked and what was sent, interleaved
//     by time so a bad render reads back to the setting that caused it.
//   * io lines carry FULL PATHS and are the answer to "which file did that
//     actually write", so they are never hidden by default and they are the
//     one level that carries the accent.
//
// Filtering is client-side on purpose. /api/log's `level` selects exactly ONE
// level; the useful views are combinations, and a single unfiltered poll with
// DOM filtering also means toggling a level never re-fetches or loses history.

import { $, esc, clock, onUiLog } from "./core.js";

const LEVELS = ["info", "warn", "error", "io", "ui"];
const CAP = 4000;                     // what the client holds, not the server's

const rows = [];                      // {seq, t, level, source, project, msg}
let seq = 0;                          // last SERVER seq rendered
let uiSeq = 0;                        // synthetic seq for client-side lines
const on = new Set(LEVELS);           // visible levels; all on = "everything"
let find = "";
let follow = true;
let meta = { skipped: 0, oldest_seq: 0, newest_seq: 0, held: 0, capacity: 0 };
let totalSkipped = 0;
let dirty = true;
let paneEl = null;

function lineHtml(r) {
  const gap = r.gap ? `<span class="gap">▲ ${r.gap} line(s) rolled off the ring before this</span>` : "";
  return `<div class="ln ${r.level}">`
       + `<span class="t">${clock(r.t)}</span>`
       + `<span class="l">${r.level}</span>`
       + `<span class="s">${esc(r.source || "")}</span>`
       + `<span class="m">${gap}${esc(r.msg)}</span></div>`;
}

function visible(r) {
  if (!on.has(r.level)) return false;
  if (!find) return true;
  const f = find.toLowerCase();
  return (r.msg || "").toLowerCase().includes(f)
      || (r.source || "").toLowerCase().includes(f);
}

function paint() {
  if (!dirty || !paneEl) return;
  dirty = false;
  const body = $("conBody");
  const atBot = body.scrollTop + body.clientHeight >= body.scrollHeight - 30;
  body.innerHTML = rows.filter(visible).map(lineHtml).join("");
  if (follow || atBot) body.scrollTop = body.scrollHeight;

  const shown = body.childElementCount;
  const pct = meta.capacity ? Math.round(100 * meta.held / meta.capacity) : 0;
  $("conMeta").innerHTML =
    `seq ${meta.oldest_seq}–${meta.newest_seq} · ring ${meta.held}/${meta.capacity} (${pct}%)`
    + ` · ${shown} shown of ${rows.length}`
    + (totalSkipped ? ` · <b>${totalSkipped} dropped</b>` : "");
}

function push(r) {
  rows.push(r);
  if (rows.length > CAP) rows.splice(0, rows.length - CAP);
  dirty = true;
}

async function tick() {
  try {
    const r = await fetch(`/api/log?since=${seq}&limit=800`);
    if (!r.ok) return;
    const j = await r.json();
    meta = {
      skipped: j.skipped || 0, oldest_seq: j.oldest_seq || 0,
      newest_seq: j.newest_seq || 0, held: j.held || 0,
      capacity: j.capacity || 0,
    };
    // A gap is real information: the ring rolled past what this client had
    // not read yet. Mark it inline rather than letting the log look continuous.
    if (seq && j.oldest_seq > seq + 1) {
      const lost = j.oldest_seq - seq - 1;
      totalSkipped += lost;
      push({ seq: seq + 0.5, t: Date.now() / 1000, level: "warn",
             source: "console", msg: "", gap: lost });
    }
    if (j.skipped) totalSkipped += j.skipped;
    for (const e of j.entries || []) {
      seq = Math.max(seq, e.seq);
      push(e);
    }
    dirty = true;
  } catch (e) {
    // The server is restarting. Say so once rather than going quiet.
    if (!tick._down) {
      tick._down = true;
      push({ seq: 0, t: Date.now() / 1000, level: "error", source: "console",
             msg: "lost contact with the server — retrying every 1.2s" });
      dirty = true;
    }
    return;
  }
  if (tick._down) {
    tick._down = false;
    push({ seq: 0, t: Date.now() / 1000, level: "info", source: "console",
           msg: "server is answering again" });
  }
}

export function init() {
  paneEl = $("conBody");

  // Client-side UI actions land in the SAME stream, tagged "ui".
  onUiLog(e => {
    push({ seq: `u${++uiSeq}`, t: e.t, level: e.level === "ui" ? "ui" : e.level,
           source: "you", msg: e.msg });
    paint();
  });

  for (const b of document.querySelectorAll("#conTabs .lv")) {
    b.onclick = () => {
      const lv = b.dataset.lv;
      if (on.has(lv)) on.delete(lv); else on.add(lv);
      b.classList.toggle("on", on.has(lv));
      dirty = true; paint();
    };
  }
  $("conAll").onclick = () => {
    for (const lv of LEVELS) on.add(lv);
    for (const b of document.querySelectorAll("#conTabs .lv")) b.classList.add("on");
    find = ""; $("conFind").value = "";
    dirty = true; paint();
  };
  $("conFind").addEventListener("input", () => {
    find = $("conFind").value.trim(); dirty = true; paint();
  });
  $("conFollow").addEventListener("change", () => { follow = $("conFollow").checked; });
  $("conClear").onclick = () => {
    rows.length = 0; totalSkipped = 0; dirty = true; paint();
    push({ seq: 0, t: Date.now() / 1000, level: "info", source: "console",
           msg: "cleared the view — the server's ring buffer is untouched, "
              + "reload to read it back from seq " + meta.oldest_seq });
    dirty = true; paint();
  };

  // Console / Sent-to-fal panes
  for (const b of document.querySelectorAll("#conPanes button")) {
    b.onclick = () => {
      for (const o of document.querySelectorAll("#conPanes button")) o.classList.remove("on");
      b.classList.add("on");
      const log = b.dataset.pane === "log";
      $("conBody").classList.toggle("hidden", !log);
      $("conTabs").classList.toggle("hidden", !log);
      $("sentBody").classList.toggle("hidden", log);
    };
  }
  $("conToggle").onclick = () => {
    const c = $("console");
    c.classList.toggle("min");
    $("conToggle").textContent = c.classList.contains("min") ? "▵" : "▽";
  };

  tick();
  setInterval(tick, 1200);
  setInterval(paint, 300);
}

/** "Sent to fal" -- what actually LEFT, read from state.fal_sent, which
 *  falclient writes at the moment of upload. Deliberately NOT a re-render of
 *  the UI's intent: the distinction between "what will be sent" (the plan) and
 *  "what was sent" (this) is the whole value of the panel. */
export function renderSent(state) {
  const el = $("sentBody");
  const items = (state && state.fal_sent) || [];
  if (!items.length) {
    el.innerHTML = `<div class="dim">Nothing has been uploaded from this `
      + `project yet. When it is, this panel shows what actually left — the `
      + `role, the size, and the full path on disk, recorded at the moment of `
      + `upload rather than re-derived from the controls.</div>`;
    return;
  }
  el.innerHTML = items.map(f => `<div class="item">
      <b>${esc(f.role || f.kind || "file")}</b> · ${esc(f.name || "")}
      ${f.bytes ? ` · ${((+f.bytes) / 1e6).toFixed(2)} MB` : ""}
      ${f.when ? ` · ${clock(f.when)}` : ""}
      <div class="p">${esc(f.path || "")}</div>
      ${f.url ? `<div class="p">${esc(f.url)}</div>` : ""}
    </div>`).join("");
}
