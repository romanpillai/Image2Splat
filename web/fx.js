// Motion and feedback -- the one module that knows GSAP exists.
//
// Every function here DEGRADES. No GSAP (the vendor file failed to load) or
// prefers-reduced-motion means the end state is applied at once and the
// message still appears: feedback is never gated on an animation.
//
// Three surfaces, one vocabulary:
//   buttons   busy (spinner + verb) -> done (check) / failed (cross + shake)
//   viewport  working card over the 3D view -> success / failure / stopped
//   toasts    short confirmations and refusals that belong to no job

const mq = window.matchMedia ? matchMedia("(prefers-reduced-motion: reduce)") : null;
export const reduced = () => !!(mq && mq.matches);
const G = () => (!reduced() && window.gsap) ? window.gsap : null;
export const wait = ms => new Promise(r => setTimeout(r, ms));

// ------------------------------------------------------------------ toasts --
const TICON = { ok: "✓", err: "✕", info: "i", warn: "!" };
let toastBox = null;

/** A short message in the top-right corner. Errors stay longer and can be
 *  clicked away; nothing here blocks the page. */
export function toast(msg, kind = "info", ms) {
  if (!toastBox) {
    toastBox = document.createElement("div");
    toastBox.className = "toasts";
    toastBox.setAttribute("role", "status");
    toastBox.setAttribute("aria-live", "polite");
    document.body.appendChild(toastBox);
  }
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<span class="ti" aria-hidden="true">${TICON[kind] || "i"}</span><span class="tm"></span>`;
  el.querySelector(".tm").textContent = String(msg || "");
  toastBox.appendChild(el);
  while (toastBox.children.length > 4) toastBox.firstElementChild.remove();
  const g = G();
  if (g) g.fromTo(el, { autoAlpha: 0, x: 16 }, { autoAlpha: 1, x: 0, duration: 0.24, ease: "power2.out" });
  const kill = () => {
    if (!el.isConnected) return;
    const gg = G();
    if (gg) gg.to(el, { autoAlpha: 0, x: 12, duration: 0.18, onComplete: () => el.remove() });
    else el.remove();
  };
  el.onclick = kill;
  setTimeout(kill, ms ?? (kind === "err" ? 7000 : kind === "warn" ? 4500 : 2600));
  return el;
}

// ----------------------------------------------------------------- buttons --
export const btnTransient = b => !!b
  && (b.classList.contains("is-busy") || b.classList.contains("is-ok")
      || b.classList.contains("is-err"));

function remember(btn) {
  if (btn.dataset.fxHtml == null) {
    btn.dataset.fxHtml = btn.innerHTML;
    btn.dataset.fxDisabled = btn.disabled ? "1" : "0";
  }
}

/** The button that started something says so, in its own words, until the
 *  work ends. It is disabled while busy so a double click cannot send twice. */
export function btnBusy(btn, label = "Working…") {
  if (!btn || !btn.isConnected) return;
  clearTimeout(btn._fxT);
  remember(btn);
  btn.classList.remove("is-ok", "is-err");
  btn.classList.add("is-busy");
  btn.setAttribute("aria-busy", "true");
  btn.disabled = true;
  btn.innerHTML = `<span class="bspin" aria-hidden="true"></span><span class="bl"></span>`;
  btn.querySelector(".bl").textContent = label;
}

export function btnBusyLabel(btn, label) {
  const l = btn && btn.classList.contains("is-busy") && btn.querySelector(".bl");
  if (l) l.textContent = label;
}

/** Confirm or reject on the button itself, then put it back. */
export function btnDone(btn, ok, label, hold) {
  if (!btn || !btn.isConnected) return;
  clearTimeout(btn._fxT);
  remember(btn);
  btn.classList.remove("is-busy", "is-ok", "is-err");
  btn.removeAttribute("aria-busy");
  btn.classList.add(ok ? "is-ok" : "is-err");
  btn.disabled = true;
  btn.innerHTML = `<span class="bmark" aria-hidden="true">${ok ? "✓" : "✕"}</span><span class="bl"></span>`;
  btn.querySelector(".bl").textContent = label || (ok ? "Done" : "Failed");
  const g = G();
  if (g) {
    if (ok) g.fromTo(btn.querySelector(".bmark"), { scale: 0.3, rotate: -30 },
                     { scale: 1, rotate: 0, duration: 0.4, ease: "back.out(3)" });
    else g.fromTo(btn, { x: -5 }, { x: 5, duration: 0.06, repeat: 5, yoyo: true,
                                    ease: "none", clearProps: "x" });
  }
  btn._fxT = setTimeout(() => btnRestore(btn), hold ?? (ok ? 1100 : 1900));
}

export function btnRestore(btn) {
  if (!btn) return;
  clearTimeout(btn._fxT);
  btn.classList.remove("is-busy", "is-ok", "is-err");
  btn.removeAttribute("aria-busy");
  if (btn.dataset.fxHtml != null) { btn.innerHTML = btn.dataset.fxHtml; delete btn.dataset.fxHtml; }
  if (btn.dataset.fxDisabled != null) { btn.disabled = btn.dataset.fxDisabled === "1"; delete btn.dataset.fxDisabled; }
  // A gated primary button re-applies its own blocked/ready state.
  btn.dispatchEvent(new CustomEvent("fx:restored"));
}

/** A value the tool changed for you blinks, so the change is seen. */
export function flash(el) {
  if (!el) return;
  const g = G();
  if (!g) return;
  g.fromTo(el, { backgroundColor: "rgba(47,125,69,0.28)" },
           { backgroundColor: "rgba(47,125,69,0)", duration: 1.3, ease: "power1.out",
             clearProps: "backgroundColor" });
}

// ------------------------------------------------------------------ ranges --
/** The filled part of a slider track, carried by --pct. */
export function syncRange(el) {
  if (!el || el.type !== "range") return;
  const lo = +el.min || 0, hi = +el.max, v = +el.value;
  const span = (isFinite(hi) ? hi : 100) - lo;
  const pct = span > 0 ? Math.max(0, Math.min(100, ((v - lo) / span) * 100)) : 0;
  el.style.setProperty("--pct", pct.toFixed(2) + "%");
}
export function syncAllRanges(scope = document) {
  scope.querySelectorAll('input[type="range"]').forEach(syncRange);
}
document.addEventListener("input", e => {
  if (e.target && e.target.type === "range") syncRange(e.target);
}, true);

// --------------------------------------------------------------- accordion --
export function expand(body, done) {
  const g = G();
  if (!g || !body) { if (done) done(); return; }
  g.killTweensOf(body);
  g.fromTo(body, { height: 0, opacity: 0, overflow: "hidden" },
           { height: "auto", opacity: 1, duration: 0.32, ease: "power2.out",
             clearProps: "height,opacity,overflow", onComplete: done });
}
export function collapse(body, done) {
  const g = G();
  if (!g || !body) { if (done) done(); return; }
  g.killTweensOf(body);
  g.to(body, { height: 0, opacity: 0, overflow: "hidden", duration: 0.2,
               ease: "power2.in",
               onComplete: () => { g.set(body, { clearProps: "height,opacity,overflow" }); if (done) done(); } });
}

// --------------------------------------------------------------- drop zone --
/** Uploading: show the file at once, dimmed, with a moving stripe. */
export function dropBusy(el, previewUrl, name) {
  if (!el) return;
  el.classList.remove("ok", "bad");
  el.classList.add("busy");
  const img = el.querySelector(".dropimg");
  if (img && previewUrl) img.innerHTML = `<img src="${previewUrl}" alt="">`;
  const msg = el.querySelector(".dropmsg");
  if (msg) msg.textContent = `Uploading ${name || "the photo"}…`;
}

/** Loaded: a ring pulse, a tick badge, the photo settling in. Resolves when
 *  the moment has landed, so the next step can open right after it. */
export async function dropOk(el, title, detail) {
  if (!el) return;
  el.classList.remove("busy", "bad");
  el.classList.add("ok", "has");
  const msg = el.querySelector(".dropmsg");
  if (msg) msg.textContent = "Click or drop to replace the photo";
  const badge = el.querySelector(".dropbadge");
  if (badge) {
    badge.innerHTML = `<span class="bk" aria-hidden="true">✓</span><span><b></b><em></em></span>`;
    badge.querySelector("b").textContent = title;
    badge.querySelector("em").textContent = detail || "";
  }
  const g = G();
  if (g) {
    const tl = g.timeline();
    tl.fromTo(el, { boxShadow: "0 0 0 0 rgba(47,125,69,.55)" },
              { boxShadow: "0 0 0 14px rgba(47,125,69,0)", duration: 0.7, ease: "power2.out", clearProps: "boxShadow" }, 0);
    const img = el.querySelector(".dropimg img");
    if (img) tl.fromTo(img, { scale: 1.04, opacity: 0.6 }, { scale: 1, opacity: 1, duration: 0.5, ease: "power2.out" }, 0);
    if (badge) {
      tl.fromTo(badge, { autoAlpha: 0, y: 10 }, { autoAlpha: 1, y: 0, duration: 0.3, ease: "power2.out" }, 0.08);
      tl.fromTo(badge.querySelector(".bk"), { scale: 0 }, { scale: 1, duration: 0.45, ease: "back.out(3)" }, 0.15);
    }
    await wait(720);
  } else {
    // No motion: the badge simply appears, and stays long enough to be read.
    if (badge) { badge.style.visibility = "visible"; badge.style.opacity = "1"; }
    await wait(450);
  }
  setTimeout(() => {
    const gg = G();
    if (badge && gg) gg.to(badge, { autoAlpha: 0, duration: 0.4, onComplete: () => el.classList.remove("ok") });
    else { if (badge) { badge.style.visibility = ""; badge.style.opacity = ""; } el.classList.remove("ok"); }
  }, 2600);
}

export function dropFail(el, msg) {
  if (!el) return;
  el.classList.remove("busy", "ok");
  el.classList.add("bad");
  const m = el.querySelector(".dropmsg");
  if (m) m.textContent = msg || "That file could not be loaded";
  const g = G();
  if (g) g.fromTo(el, { x: -6 }, { x: 6, duration: 0.06, repeat: 5, yoyo: true, ease: "none", clearProps: "x" });
  setTimeout(() => {
    el.classList.remove("bad");
    if (m) m.textContent = el.classList.contains("has")
      ? "Click or drop to replace the photo" : "Drop a photo here, or click to choose";
  }, 3500);
}

// ---------------------------------------------------------------- viewport --
// One layer per .vp, built lazily. Both workspaces get the same card, so a job
// that ends while you are on the other tab still reports there.
//
// Two sizes. A CARD (centre, dims the view) for work you started and wait on;
// a PILL (bottom edge, no dim) for quick work -- a re-lift after a slider, a
// cached result -- so feedback never covers the model you are adjusting.
// The card appears only after a short delay: a job that is over in a blink
// never flashes a dim over the view at all.
const layers = new Map();
let elapsedTimer = null;
let workState = null;

const PATHS = {
  ok: "M12.5 20.5l5 5 10-11",
  warn: "M20 11v11M20 27.5v1.5",
  fail: "M14 14l12 12M26 14L14 26",
  stop: "M15 15h10v10H15z",
};

function layer(vp) {
  let L = layers.get(vp);
  if (L && L.root.isConnected) return L;
  const root = document.createElement("div");
  root.className = "vpfx";
  root.innerHTML = `
    <div class="vpdim"></div>
    <div class="vpcard vpwork" role="status" aria-live="polite">
      <div class="vpspin" aria-hidden="true"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/></svg></div>
      <div class="vptxt"><div class="vpt"></div><div class="vpd"></div></div>
      <div class="vpel"></div>
      <div class="vpprog"><i></i></div>
    </div>
    <div class="vpcard vpout" role="alert">
      <div class="vpicon" aria-hidden="true"><svg viewBox="0 0 40 40"><circle class="ring" cx="20" cy="20" r="17"/><path class="mk" d=""/></svg></div>
      <div class="vptxt"><div class="vpt"></div><div class="vpd"></div><div class="vph"></div></div>
      <button class="vpx" type="button" aria-label="Dismiss" title="Dismiss">&times;</button>
    </div>`;
  vp.appendChild(root);
  L = { root, dim: root.querySelector(".vpdim"), work: root.querySelector(".vpwork"),
        out: root.querySelector(".vpout"), outT: null, showT: null };
  L.out.querySelector(".vpx").addEventListener("click", e => { e.stopPropagation(); hideOutcome(L); });
  layers.set(vp, L);
  return L;
}

const fmtElapsed = s => {
  s = Math.max(0, Math.round(s));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

function paintWork(L, w) {
  L.work.querySelector(".vpt").textContent = w.title || "Working";
  L.work.querySelector(".vpd").textContent = w.detail || "";
  L.work.querySelector(".vpel").textContent = fmtElapsed((Date.now() - (w.t0 || Date.now())) / 1000);
  const bar = L.work.querySelector(".vpprog");
  const pct = +w.pct || 0;
  bar.classList.toggle("indet", !(pct > 0 && pct < 100));
  bar.querySelector("i").style.width = pct > 0 && pct < 100 ? `${pct.toFixed(1)}%` : "";
  L.root.classList.toggle("long", !!w.long);
  L.root.classList.toggle("compact", !!w.compact);
}

function hideOutcome(L, instant) {
  clearTimeout(L.outT);
  if (!L.out.classList.contains("on")) return;
  const g = instant ? null : G();
  const off = () => {
    L.out.classList.remove("on", "ok", "fail", "stop", "warn", "pill");
    L.root.classList.remove("showout");
  };
  if (g) g.to(L.out, { autoAlpha: 0, y: 4, duration: 0.3, ease: "power1.in", onComplete: () => { off(); g.set(L.out, { clearProps: "all" }); } });
  else { off(); L.out.style.cssText = ""; }
}

function showWork(L, w) {
  L.showT = null;
  if (L.work.classList.contains("on")) return;
  L.work.classList.add("on");
  L.root.classList.add("working");
  const g = G();
  if (g) {
    g.fromTo(L.work, { autoAlpha: 0, y: w.long || w.compact ? 10 : 6, scale: 0.98 },
             { autoAlpha: 1, y: 0, scale: 1, duration: 0.26, ease: "power2.out" });
    g.fromTo(L.dim, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.3 });
  }
}

/** Show (or update) the working card.
 *  w = {title, detail, pct, long, compact, delay, t0}. */
export function vpWork(vps, w) {
  workState = w;
  for (const vp of vps) {
    const L = layer(vp);
    hideOutcome(L, true);
    paintWork(L, w);
    if (L.work.classList.contains("on") || L.showT) continue;
    const delay = +w.delay || 0;
    if (delay > 0) L.showT = setTimeout(() => { if (workState === w || workState) showWork(L, workState || w); }, delay);
    else showWork(L, w);
  }
  if (!elapsedTimer) {
    elapsedTimer = setInterval(() => {
      if (!workState) return;
      for (const L of layers.values()) {
        if (L.work.classList.contains("on")) {
          L.work.querySelector(".vpel").textContent =
            fmtElapsed((Date.now() - (workState.t0 || Date.now())) / 1000);
        }
      }
    }, 1000);
  }
}

/** Returns true if the card had actually been on screen. */
function hideWork(L) {
  if (L.showT) { clearTimeout(L.showT); L.showT = null; }
  if (!L.work.classList.contains("on")) return false;
  L.work.classList.remove("on");
  L.root.classList.remove("working");
  const g = G();
  if (g) { g.set(L.work, { clearProps: "all" }); g.to(L.dim, { autoAlpha: 0, duration: 0.3, clearProps: "all" }); }
  return true;
}

/** End state: {kind: ok|warn|fail|stop, title, detail, compact}.
 *  Success fades by itself; a warning lingers; a failure stays until it is
 *  dismissed or the next job starts. None of them take the pointer: the view
 *  can still be orbited underneath. */
export function vpOutcome(vps, o) {
  workState = null;
  for (const vp of vps) {
    const L = layer(vp);
    hideWork(L);
    clearTimeout(L.outT);
    const card = L.out;
    const pill = !!o.compact && o.kind !== "fail";
    card.classList.remove("ok", "fail", "stop", "warn", "pill");
    card.classList.add("on", o.kind);
    card.classList.toggle("pill", pill);
    L.root.classList.add("showout");
    card.querySelector(".vpt").textContent = o.title || "";
    card.querySelector(".vpd").textContent = o.detail || "";
    card.querySelector(".vph").textContent = o.kind === "fail" ? "The full error is in the console." : "";
    const mk = card.querySelector(".mk");
    mk.setAttribute("d", PATHS[o.kind] || PATHS.ok);
    const g = G();
    if (g) {
      g.killTweensOf([card, mk]);
      const len = mk.getTotalLength ? mk.getTotalLength() : 40;
      const tl = g.timeline();
      tl.fromTo(card, { autoAlpha: 0, scale: pill ? 1 : 0.9, y: pill ? 8 : 0 },
                { autoAlpha: 1, scale: 1, y: 0, duration: pill ? 0.22 : 0.34, ease: pill ? "power2.out" : "back.out(1.8)" }, 0);
      tl.fromTo(mk, { strokeDasharray: len, strokeDashoffset: len },
                { strokeDashoffset: 0, duration: 0.4, ease: "power2.out" }, pill ? 0.06 : 0.14);
      if (o.kind === "fail") {
        tl.fromTo(card, { x: -7 }, { x: 7, duration: 0.07, repeat: 5, yoyo: true, ease: "none", clearProps: "x" }, 0.34);
      }
    } else {
      card.style.opacity = "1";
      card.style.visibility = "visible";
    }
    if (o.kind !== "fail") {
      const life = pill ? 1400 : o.kind === "warn" ? 4200 : o.kind === "stop" ? 2400 : 1700;
      L.outT = setTimeout(() => hideOutcome(L), life);
    }
  }
}

export function vpClear(vps) {
  workState = null;
  for (const vp of vps) {
    const L = layer(vp);
    hideWork(L);
    hideOutcome(L, true);
  }
}

/** Was the working card on screen for this job? (false = it finished inside
 *  the show delay, so a quiet pill is the right way to report it). */
export function vpWorkShown(vps) {
  return vps.some(vp => { const L = layers.get(vp); return !!(L && L.work.classList.contains("on")); });
}
