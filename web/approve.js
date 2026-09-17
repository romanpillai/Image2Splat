// The approval window -- one window for every "are you sure" in the tool, and
// it opens ITSELF.
//
// Load-bearing, and deliberately not simplified:
//
//   * It shows the ACTUAL MEDIA, not filenames. A clip from the wrong project,
//     or a matte cut from a different video, is obvious at a glance and
//     unreadable as a name.
//   * Every card carries the FULL PATH on disk. "Which file is it really
//     using" is the question this window exists to answer.
//   * The plan comes from the SERVER, resolved by the same functions the
//     submit path uses (/api/generate/plan, /api/splat/plan,
//     /api/dataset/complete/plan). It is not a client-side guess.
//   * The fal prompt is shown CLAUSE BY CLAUSE. The box on screen holds your
//     words; what actually goes is a fixed preamble with those glued on the
//     end, and the difference is invisible until it is shown.

import { $, esc, mediaHtml, logUi } from "./core.js";

const apv = { onApprove: null, plan: null };

function stopMedia() {
  for (const v of $("apvCards").querySelectorAll("video")) {
    try { v.pause(); } catch (e) { /* already gone */ }
  }
}

export function apvClose() {
  $("apvWrap").classList.remove("on");
  stopMedia();
  $("apvCards").innerHTML = "";
  apv.onApprove = null;
  apv.plan = null;
}

function card(f) {
  const amount = f.kind === "dir"
    ? `${(f.count || 0).toLocaleString()} file${f.count === 1 ? "" : "s"}`
    : `${((f.bytes || 0) / 1e6).toFixed(2)} MB`;
  return `<div class="apvcard">
      <div class="role">${esc(f.role)}<span>${esc(amount)}</span></div>
      <div class="media">${mediaHtml(f.media, f.role)}</div>
      <div class="path">${esc(f.path || f.name)}</div>
      ${f.note ? `<div class="note">${esc(f.note)}</div>` : ""}
    </div>`;
}

/** key/value rows, plus the fal prompt broken into its numbered clauses. */
export function settingsHtml(settings) {
  if (!settings) return "";
  const out = [];
  const skip = new Set(["fal_prompt", "prompt"]);
  for (const [k, v] of Object.entries(settings)) {
    if (skip.has(k) || v === null || v === undefined || v === "") continue;
    out.push(`<div class="kv"><div class="k">${esc(k.replace(/_/g, " "))}</div>`
           + `<div class="v">${esc(String(v))}</div></div>`);
  }
  if (settings.prompt) {
    out.push(`<div class="kv"><div class="k">your words</div>`
           + `<div class="v">${esc(settings.prompt)}</div></div>`);
  }
  if (settings.fal_prompt) {
    const clauses = String(settings.fal_prompt).split(",")
      .map(c => c.trim()).filter(Boolean);
    out.push(`<div class="cap" style="margin-top:var(--s-5)">`
      + `The COMPLETE string fal receives — boilerplate and all, clause by `
      + `clause (${clauses.length})</div>`);
    out.push(`<ol class="clauses">`
      + clauses.map(c => `<li>${esc(c)}</li>`).join("") + `</ol>`);
  }
  return out.join("");
}

/** Open the window.
 *  opts = {title, subtitle, settingsHtml, notes[], onApprove, goLabel} */
export function apvShow(plan, opts) {
  apv.plan = plan;
  apv.onApprove = opts.onApprove;
  $("apvTitle").textContent = opts.title || "Approve";
  $("apvSub").innerHTML = opts.subtitle || "";
  $("apvDir").textContent = plan.project_dir
    ? "project folder:  " + plan.project_dir : "";
  $("apvCards").innerHTML = (plan.files || []).map(card).join("");

  const set = opts.settingsHtml !== undefined
    ? opts.settingsHtml : settingsHtml(plan.settings);
  $("apvSettings").innerHTML = set || "";
  $("apvSettingsSec").classList.toggle("hidden", !set);

  const notes = (opts.notes || plan.notes || []);
  $("apvNotes").innerHTML = notes.length
    ? notes.map(n => "· " + esc(n)).join("<br>") : "";
  $("apvNotesSec").classList.toggle("hidden", !notes.length);

  const probs = plan.problems || [];
  $("apvProblems").innerHTML = probs.length
    ? `<div class="problem-block">`
      + probs.map(p => `<div>${esc(p)}</div>`).join("") + `</div>`
    : "";

  $("apvGo").textContent = opts.goLabel || "Approve";
  $("apvGo").disabled = probs.length > 0
    || (opts.needFiles !== false && !(plan.files || []).length);
  const megs = (plan.total_bytes || 0) / 1e6;
  $("apvTotal").textContent = megs > 0.01 ? `${megs.toFixed(2)} MB total` : "";
  $("apvWrap").classList.add("on");

  logUi(`approval window: ${opts.title} — ${(plan.files || []).length} file(s), `
      + `${probs.length} problem(s)`
      + ((plan.files || []).length
         ? ` · ${(plan.files || []).map(f => f.path || f.name).join(" · ")}` : ""));
}

export function initApproval() {
  $("apvCancel").onclick = () => {
    logUi("cancelled at the approval window — nothing was sent");
    apvClose();
  };
  $("apvWrap").addEventListener("click", e => {
    if (e.target === $("apvWrap")) {           // click the backdrop to dismiss
      logUi("dismissed the approval window — nothing was sent");
      apvClose();
    }
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && $("apvWrap").classList.contains("on")) apvClose();
  });
  $("apvGo").onclick = async () => {
    if (!apv.onApprove) return;
    const fn = apv.onApprove;
    const label = $("apvGo").textContent;
    $("apvGo").disabled = true;
    $("apvGo").textContent = "starting…";
    try {
      await fn();
      apvClose();
    } catch (e) {
      $("apvGo").disabled = false;
      $("apvGo").textContent = label;
      $("apvProblems").innerHTML =
        `<div class="problem-block"><div>${esc(e.message)}</div></div>`;
    }
  };
}
