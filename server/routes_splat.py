"""Matte -> visual hull -> COLMAP with the exact poses -> Brush.

The plan endpoint has the same contract as the fal approval: this is the ONE
place the inputs are resolved, so the panel cannot list one thing while the run
uses another. A splat is tens of minutes, and every fault it catches -- a
frame/pose mismatch, a matte from another clip, an unapproved video -- is one
you would otherwise find out about afterwards.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import steps

from . import config
from .common import _dir_preview, _media_url
from .jobs import (_set, CHILD, JOB, LOCK, _busy_msg, _io, _log, _run, _where,
                   refuse, rx)
from .models import SplatReq
from .state import keeps_background, load_state, pdir, save_state

BRUSH = config.BRUSH
router = APIRouter()


def ensure_rerun_viewer() -> str:
    """Bring up a Rerun viewer if one is not already listening.

    Rerun will happily run several viewers at once, but each is a window and
    a memory buffer, so spawning one per training run would bury the user.
    Reuse whatever is already there.
    """
    if steps.rerun_listening():
        return f"viewer already listening on {steps.RERUN_PORT}"
    exe = steps.rerun_viewer_exe()
    if exe is None:
        return ("viewer NOT installed - run "
                "`uv pip install rerun-sdk` in the server's environment. "
                "Training continues without telemetry.")
    try:
        subprocess.Popen([str(exe), "--port", str(steps.RERUN_PORT)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        return f"viewer failed to start ({e}); training continues without it"
    for _ in range(40):                     # it has a window to open first
        if steps.rerun_listening():
            return f"viewer started on port {steps.RERUN_PORT}"
        time.sleep(0.25)
    return ("viewer did not come up in 10s; training continues and it will "
            "attach if it appears")


@router.get("/api/rerun")
def rerun_status():
    exe = steps.rerun_viewer_exe()
    return {"installed": exe is not None,
            "exe": str(exe) if exe else None,
            "listening": steps.rerun_listening(),
            "port": steps.RERUN_PORT}


@router.post("/api/rerun/open")
def rerun_open():
    """Open the viewer on demand, without starting a training run."""
    rx("/api/rerun/open")
    msg = ensure_rerun_viewer()
    ok = steps.rerun_listening()
    _log(f"rerun: {msg}")
    return JSONResponse({"ok": ok, "message": msg},
                        status_code=200 if ok else 400)


@router.post("/api/splat/plan")
def splat_plan(r: SplatReq):
    """What a splat run would train on, before it starts."""
    d = pdir(r.project)
    s = load_state(d)
    files, problems, notes = [], [], []

    def add(role, path, extra=""):
        p = Path(path)
        if not p.exists():
            problems.append(f"{role}: {p.name} is not on disk")
            return None
        if p.is_dir():
            n = len(list(p.glob("frame_*.png")))
            files.append({"role": role, "name": p.name + "/", "kind": "dir",
                          "count": n, "bytes": 0, "note": extra,
                          "path": str(p.resolve()),
                          "media": _dir_preview(p, d)})
            return n
        files.append({"role": role, "name": p.name, "kind": "file",
                      "count": 1, "bytes": p.stat().st_size, "note": extra,
                      "path": str(p.resolve()),
                      "media": _media_url(p, d)})
        return 1

    if not s.get("approved"):
        problems.append("the AI video has not been approved yet - review it "
                        "in step 5 first")
    orbit = steps.orbit_from_dict(s.get("orbit") or {})
    poses = orbit.frames if s.get("orbit") else 0

    # ---- the imagery -------------------------------------------------------
    ai_name = s.get("ai_video")
    n_ai = 0
    if not ai_name:
        problems.append("no AI video in this project")
    else:
        ai = d / ai_name
        add("AI video", ai, f"{s.get('ai_engine') or 'unknown'} engine")
        if ai.exists():
            n_ai, fps = steps.video_frame_count(ai)
            notes.append(f"{ai.name}: {n_ai} frames at {fps:g} fps")
            if poses and n_ai != poses:
                problems.append(
                    f"{ai.name} has {n_ai} frames but the orbit has {poses} "
                    f"poses. Every pose would address the wrong image - retime "
                    f"the clip first.")

    # ---- the alpha ---------------------------------------------------------
    # A kept background IS the scene. Cutting it out would throw away the thing
    # it was rendered for, so the run trains on full opaque frames and the
    # matte choice is ignored entirely. Say that rather than listing a matte
    # directory the run will not open.
    keep_bg = keeps_background(s)
    src = r.matte_source
    if keep_bg:
        notes.append("This project keeps its BACKGROUND (a legacy scene or "
                     "panorama backdrop), so training uses full opaque "
                     "frames, no alpha, and the matte setting is ignored.")
        if (d / "ai_frames").exists():
            add("frames (no matting)", d / "ai_frames",
                "full frames, alpha left opaque")
        else:
            notes.append("frames will be extracted from the AI video")
    elif src in steps.MATTE_MODELS:
        md = d / "mattes" / src
        n_m = add(f"mattes ({src})", md, "one RGBA per frame")
        if n_m is not None and n_ai and n_m != n_ai:
            problems.append(
                f"mattes/{src}/ holds {n_m} masks but the clip has {n_ai} "
                f"frames - they were cut from a different video. Re-run the "
                f"matte.")
    elif src == "authored":
        add("authored coverage", d / "control_mattes",
            "control-render coverage - NOT a subject silhouette")
        notes.append("authored coverage swings with the orbit; it marks where "
                     "POINTS landed, not where the subject is")
    else:
        notes.append("alpha from colour distance to the backdrop only - no "
                     "mask model involved")

    # ---- the geometry ------------------------------------------------------
    if poses:
        notes.append(f"{poses} authored poses -> COLMAP sparse/0 "
                     f"({orbit.width}x{orbit.height}, "
                     f"{orbit.sweep_deg:g} deg sweep, path {orbit.path})")
    else:
        problems.append("no orbit recorded - render the control video first")
    if not BRUSH.exists():
        problems.append(f"Brush is not at {BRUSH} (set \"brush\" in "
                        f"config.json)")

    out = d / "splat_out"
    prev = len([q for q in out.glob("*.ply")]) if out.is_dir() else 0
    if prev:
        notes.append(f"{prev} checkpoint(s) from a previous run will be moved "
                     f"to splat_out/previous")

    settings = {"steps": r.steps, "detail": r.detail, "sh_degree": r.sh_degree,
                "matte_source": src, "clean": r.clean, "rerun": r.rerun,
                "alpha_supervision": not keep_bg}
    _log(f"splat plan: '{d.name}' - {len(files)} input(s), "
         f"{len(problems)} problem(s), Brush at {BRUSH}", source="splat")
    for f in files:
        _io("would train from", f["path"], source="splat",
            note=f"{f['role']}, {f.get('count', 1)} item(s)")
    for p in problems:
        _log(f"splat plan: PROBLEM {p}", level="warn", source="splat")
    return {"files": files, "problems": problems, "notes": notes,
            "settings": settings, "project_dir": str(d.resolve()),
            "brush": str(BRUSH), "brush_found": BRUSH.exists(),
            "total_bytes": sum(f["bytes"] for f in files)}


@router.post("/api/splat")
def do_splat(r: SplatReq):
    rx("/api/splat", r.project, steps_=r.steps, detail=r.detail,
       sh_degree=r.sh_degree, matte_source=r.matte_source,
       dataset_only=r.dataset_only, clean=r.clean, rerun=r.rerun)
    d = pdir(r.project)
    s = load_state(d)
    if not s.get("approved"):
        return JSONResponse({"error": "approve the video first"}, status_code=400)
    ai = d / s["ai_video"]
    if not ai.exists():
        return JSONResponse({"error": "ai video missing"}, status_code=400)
    orbit = steps.orbit_from_dict(s.get("orbit", {}))

    def work():
        import cv2
        # An ai_rmbg/ left by an older run must not be picked up again.
        stale_rmbg = d / "ai_rmbg"
        if stale_rmbg.exists():
            shutil.rmtree(stale_rmbg, ignore_errors=True)
            _log("splat: removed a stale ai_rmbg/ from an earlier run")
        _log("splat: extracting frames")
        n = steps.mp4_to_frames(ai, d / "ai_frames")
        steps.stamp_write(d / "ai_frames", video=steps._fingerprint(ai))
        _io("extracted", ai, d / "ai_frames")
        _log(f"splat: {n} frames (orbit has {orbit.frames})")
        if n != orbit.frames:
            # Not a truncation: fal resamples the whole sweep, so image i and
            # pose i describe different camera angles for every frame but the
            # first. write_colmap's zip() would silently shorten and train on
            # mismatched pairs for tens of minutes.
            raise RuntimeError(
                f"frame count {n} does not match the {orbit.frames}-pose "
                f"orbit. Every pose would address the wrong image. Use "
                f"'Retime to pose count' in Review to resample this clip "
                f"onto {orbit.frames} frames, re-generate with "
                f"num_frames={orbit.frames}, or re-render the orbit at {n} "
                f"frames.")

        backdrop = s.get("backdrop") or "grey"
        keep_bg = backdrop in ("scene", "panorama")   # legacy projects only
        have_cm = False        # set by the matting branch; the report reads it
        mi = None
        if keep_bg:
            # Whole frame is the subject. No matte, so no silhouette to carve a
            # visual hull from and nothing for --match-alpha-weight to
            # supervise; both are handled below.
            _log("splat: keeping background - no matting, full frames used")
            steps.copy_frames_opaque(d / "ai_frames", d / "dataset" / "images")
            _io("copied", d / "ai_frames", d / "dataset" / "images",
                note="opaque, background kept")
        else:
            # RMBG is NEVER run over AI frames. It was a second, independent
            # opinion about where the subject ends, applied per frame with no
            # temporal coupling, so its silhouette flickered between frames
            # and disagreed with the authored geometry. The backdrop here is
            # the flat grey this pipeline rendered itself, so distance to it
            # is a direct measurement rather than a guess.
            # The control render wrote the authored silhouette per frame.
            # Older projects have none -- fall back to distance alone and say
            # so, since that is measurably fuzzier at the edges.
            cm = d / "control_mattes"
            have_cm = cm.exists() and any(cm.glob("frame_*.png"))
            mask_dir = None
            if r.matte_source in steps.MATTE_MODELS:
                rd = d / "mattes" / r.matte_source
                _mp = (load_state(d).get("matte_prompt") or "").strip()
                _fp = steps._fingerprint(ai)
                # Non-empty is not the same as current. A matte cut against
                # the previous AI video, or against a different text prompt,
                # looks identical on disk and would be paired frame-for-frame
                # with a clip it never saw.
                if not steps.stamp_ok(rd, video=_fp, model=r.matte_source,
                                      prompt=_mp):
                    _log(f"splat: {r.matte_source} over the AI frames "
                         "(no matte for this clip yet, or it was cut against "
                         "an older one)")
                    steps.matte_dir(d / "ai_frames", rd,
                                    model=r.matte_source, log=_log,
                                    prompt=_mp)
                    steps.stamp_write(rd, video=_fp, model=r.matte_source,
                                      prompt=_mp)
                    _io("wrote", rd, note=f"mattes cut from {ai.name}")
                else:
                    _io("reusing", rd, note=f"cut from {ai.name}")
                mask_dir = rd
                _log(f"splat: matting from {r.matte_source} on the AI FRAMES, "
                     "and nothing else")
            elif r.matte_source == "distance":
                _log("splat: matting by distance to the backdrop only")
            elif have_cm:
                mask_dir = cm
                # Say plainly how far it swings, because it looks authoritative
                # -- it is exact, it is just not a subject silhouette.
                import cv2 as _c2
                _cov = []
                for _q in sorted(cm.glob("frame_*.png"))[::20]:
                    _g = _c2.imread(str(_q), _c2.IMREAD_UNCHANGED)
                    if _g is None:
                        continue
                    if _g.ndim == 3:
                        _g = _g[:, :, 3] if _g.shape[2] == 4 else _g[:, :, 0]
                    _cov.append(float((_g > 127).mean()))
                _log("splat: matting from the AUTHORED coverage, and "
                     "nothing else")
                if _cov and (max(_cov) - min(_cov)) > 0.35:
                    _log(f"WARNING authored coverage swings from "
                         f"{100*min(_cov):.0f}% to {100*max(_cov):.0f}% across "
                         f"the orbit. It marks where POINTS landed, not where "
                         f"the subject is, and a single-view depth lift is a "
                         f"shell - orbit behind it and there is nothing to "
                         f"draw. Alpha supervision will read that as 'delete "
                         f"the subject'. Use a matte model instead.",
                         level="warn")
            else:
                _log("splat: no authored silhouette for this project - "
                     "matting by distance alone, which is fuzzier at the "
                     "edges. Re-render the control video to produce one.",
                     level="warn")
            # Whatever source is chosen IS the matte. The distance term only
            # applies when it is the choice, because on a fal clip it reads
            # the invented studio floor as subject.
            _io("matting", d / "ai_frames", d / "dataset" / "images",
                note=f"masks from "
                     f"{_where(mask_dir) if mask_dir else 'distance to backdrop'}")
            mi = steps.matte(d / "ai_frames", mask_dir,
                             d / "dataset" / "images",
                             use_distance=(mask_dir is None))
            _log(f"splat: coverage {100*mi['coverage']:.1f}% "
                 f"partial {100*mi['partial_alpha']:.2f}% "
                 f"bars {mi['bar_px_max']}px")

        # copy_frames_opaque clears its output; matte() does not. A shorter
        # re-run therefore left the previous run's tail frames behind, and
        # write_colmap's zip() silently paired poses with the survivors.
        _imgs = d / "dataset" / "images"
        _imgs.mkdir(parents=True, exist_ok=True)

        first = sorted(_imgs.glob("frame_*.png"))
        im = cv2.imread(str(first[0]), cv2.IMREAD_UNCHANGED)
        H, W = im.shape[:2]

        if keep_bg:
            # Seed from the scene we authored - subject volume, floor and marker
            # ring - rather than a random blob. Same idea as sampling the mesh
            # in V3: start the gaussians where geometry actually is.
            oi = s.get("orbit_info") or {}
            gz = oi.get("ground_z")
            if gz is None:
                gz = steps.ground_z_for(orbit, s.get("card_height") or 3.6)
            pts = steps.scene_init_points(orbit, gz)
            _log(f"splat: {len(pts)} init points seeded from scene "
                 f"geometry (ground z {gz:.2f})")
        else:
            _log("splat: carving visual hull for the init cloud")
            pts = steps.visual_hull(d / "dataset" / "images", orbit)
            _log(f"splat: {len(pts)} init points")

        names = [f.name for f in first]
        ds_info = None      # dataset info, kept before `ci` is reused below
        ci = steps.write_colmap(orbit, d / "dataset" / "sparse" / "0",
                                names, W, H, pts)
        _io("wrote", d / "dataset" / "sparse" / "0",
            note=f"{len(names)} authored poses, {len(pts)} init points")
        _log(f"splat: COLMAP {W}x{H} fx {ci['fx']:.1f} fy {ci['fy']:.1f}")
        # fal returns its own frame size, scaling to cover and cropping the
        # rest. Say what was lost: the authored framing is the thing the whole
        # first-frame alignment work exists to protect.
        _ar_auth = orbit.width / max(orbit.height, 1)
        _ar_img = W / max(H, 1)
        if abs(_ar_auth - _ar_img) > 0.002:
            _s = max(W / orbit.width, H / orbit.height)
            _lost_x = int(round(orbit.width * _s - W))
            _lost_y = int(round(orbit.height * _s - H))
            _log(f"splat: the delivered video is {_ar_img:.4f} aspect but the "
                 f"orbit authored {_ar_auth:.4f} - fal scaled to cover and "
                 f"cropped {_lost_x}px horizontally, {_lost_y}px vertically. "
                 f"Intrinsics account for it, but the cropped strip is gone "
                 f"from every frame. Author the canvas at {_ar_img:.4f} to "
                 f"keep the whole frame.", level="warn")
        ds_info = dict(ci)      # snapshot: `ci` is reused by the cleanup pass

        v = steps.verify(orbit, pts, W, H)
        _log(f"splat: verify depth+{v['depth_positive_min']:.3f} "
             f"in-frame {v['in_frame_min']:.3f} -> "
             f"{'PASS' if v['pass'] else 'FAIL'}")
        # The scene cloud deliberately spans the floor and the marker ring, so
        # much of it falls outside any single frame. Only depth-positivity is
        # meaningful there; in-frame coverage is not.
        if not v["depth_positive_min"] > 0.999:
            raise RuntimeError(
                "pose verification failed: the SUBJECT projects behind the "
                "camera, which means the rig is inverted or flipped.")
        if not keep_bg and not v["pass"]:
            raise RuntimeError("pose verification failed; not training")

        if r.dataset_only:
            # Everything above ran exactly as a training run would, so this
            # folder IS the one Brush would have been handed. Name the path in
            # full -- the whole reason this button exists is to go and open it.
            ds = d / "dataset"
            save_state(d, dataset_time=time.time())
            _io("dataset ready", ds, note="not training - open this in Brush")
            _log(f"dataset: {len(names)} frames, {len(pts):,} init points, "
                 f"COLMAP in sparse/0 - {W}x{H}, fx {ci['fx']:.1f}")
            _log(f"dataset: open this folder in Brush -> {ds}")
            _log("dataset: Brush reads images/ and sparse/0 itself; the matte "
                 "is already in the PNG alpha, so there is nothing to convert")
            return
        out = d / "splat_out"
        out.mkdir(exist_ok=True)
        # Exports are named by STEP, so a stale export_50000 from an earlier
        # run sorts after a fresh export_30000 and "follow latest" silently
        # shows the OLD splat -- which is exactly how a 120x improvement in
        # anisotropy went unnoticed. Noted here, moved further down: the
        # previous run's checkpoints are retired immediately before Brush
        # launches, not at this point. Doing it here meant a run that failed to
        # start left the last good splat hidden in splat_out/previous behind an
        # empty checkpoint list.
        prev = sorted(out.glob("*.ply"))
        if r.rerun:
            started = ensure_rerun_viewer()
            _log(f"splat: Rerun {started}")
        _io("training from", d / "dataset", out, note=f"Brush at {BRUSH}")
        cmd = steps.brush_cmd(BRUSH, d / "dataset", out, r.steps,
                              max_res=max(H, 512),
                              alpha_supervision=not keep_bg,
                              detail=r.detail, sh_degree=r.sh_degree,
                              rerun=r.rerun,
                              rerun_stats_every=r.rerun_stats_every,
                              rerun_splats_every=r.rerun_splats_every,
                              rerun_max_img=r.rerun_max_img)
        _dd = steps.SPLAT_DETAIL.get(r.detail, steps.SPLAT_DETAIL["balanced"])
        _log(f"splat: SH degree {r.sh_degree}"
             + (" (no view-dependent colour - removes the SH null space a "
                "single-elevation ring creates)" if r.sh_degree == 0 else ""))
        _log(f"splat: detail={r.detail} - growth threshold {_dd['grad']}, "
             f"select {_dd['frac']}, cap {int(_dd['max']):,} splats, "
             f"growth runs to step {int(r.steps * _dd.get('stop', 0.6)):,}. "
             f"Lower growth = fewer, broader, more opaque gaussians.")
        # Show the head of the command plus anything notable appended after
        # it -- the rerun flags land at the end and were invisible before.
        _tail = [c for c in cmd if str(c).startswith("--rerun")]
        _log("splat: launching Brush (" + " ".join(cmd[1:8]) + " ...)"
             + (" telemetry: " + " ".join(_tail) if _tail
                else " telemetry: off"))
        _log("splat: full command - " + " ".join(str(c) for c in cmd))
        save_state(d, stage="training", dataset_info=ci, matte_info=mi,
                   verify=v)
        # Retire the previous run's exports now, immediately before the new
        # ones appear.
        if prev:
            attic = out / "previous"
            attic.mkdir(exist_ok=True)
            for f in attic.glob("*.ply"):
                f.unlink()
            for f in prev:
                f.replace(attic / f.name)
            _io("moved", out, attic,
                note=f"{len(prev)} checkpoints from the previous run")
            _log(f"splat: moved {len(prev)} checkpoints from the previous run "
                 f"into splat_out/previous")

        # encoding + errors are load-bearing: text=True alone decodes with the
        # locale codec (cp1252 on this box), and a single UTF-8 byte undefined
        # there raises UnicodeDecodeError -- failing a run that had succeeded.
        # Popen, not run(): the handle is what makes the run abortable. It is
        # published under the lock so /api/splat_abort can reach it.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace")
        with LOCK:
            CHILD["proc"] = proc
            CHILD["aborted"] = False
            JOB["abortable"] = True
        try:
            _out, _err = proc.communicate()
        finally:
            with LOCK:
                CHILD["proc"] = None
                JOB["abortable"] = False
        with LOCK:
            was_aborted = CHILD["aborted"]
        if was_aborted:
            # The client shows a stop as neutral, not as a failure.
            _set(aborted=True)
        plys = sorted(out.glob("*.ply"))
        if was_aborted:
            # Not a failure. Brush writes a PLY per checkpoint as it goes, so
            # everything reached before the abort is on disk and usable.
            if not plys:
                # Aborting before the first checkpoint would otherwise leave
                # the project with nothing: this run retired the PREVIOUS
                # exports into splat_out/previous on the way in. Stopping a
                # run is not the same as discarding what was already trained,
                # so put them back.
                attic = out / "previous"
                restored = 0
                if attic.exists():
                    for f in sorted(attic.glob("*.ply")):
                        if not (out / f.name).exists():
                            f.replace(out / f.name)
                            restored += 1
                _log("splat: stopped before the first checkpoint"
                     + (f" - restored the previous run's {restored} "
                        "checkpoint(s)" if restored else
                        ", and there was no previous run to fall back on"),
                     level="warn")
                save_state(d, stage="review")
                return
            _log(f"splat: ABORTED - {len(plys)} checkpoint(s) written before "
                 "stopping are kept and can be viewed or exported",
                 level="warn")
        elif proc.returncode != 0:
            raise RuntimeError(f"brush exited {proc.returncode}: "
                               f"{(_err or '')[-400:]}")
        else:
            _log(f"splat: done, {len(plys)} exports")
        for q in plys:
            _io("wrote", q, note="Brush checkpoint")
        # Was the splat budget the binding limit? If the run flatlines at the
        # cap, extra steps cannot add detail -- they only reshape what is
        # there, which is how gaussians end up thin. Say so plainly.
        cap = int(_dd["max"])
        # Measure every checkpoint rather than just counting it: the numbers
        # that separate needles from see-through from GAPS are all in the PLY,
        # and without them "this run looks worse" has no evidence behind it.
        ck_stats, notes = [], []
        for q in plys:
            try:
                stt = steps.splat_stats(q)
            except Exception as e:
                stt = {"error": f"{type(e).__name__}: {e}"}
            digits = "".join(c for c in q.stem if c.isdigit())
            stt["step"] = int(digits or 0)
            stt["name"] = q.name
            ck_stats.append(stt)
            if stt.get("splats"):
                _log(f"splat:   {q.name}  {stt['splats']:,} splats  "
                     f"aniso {stt.get('aniso_median')}  "
                     f"opacity {stt.get('opacity_mean')}  "
                     f"gap {stt.get('gap_ratio')}")
        counts = [(q.name, steps.ply_splat_count(q)) for q in plys]
        final = counts[-1][1] if counts else 0
        if final and final >= cap * 0.98:
            _log(f"WARNING splat count reached the {cap:,} cap - "
                 f"densification stopped early, so later steps could only "
                 f"reshape existing gaussians (which thins them). Use a "
                 f"detail preset with a larger budget for more real detail.",
                 level="warn")
            notes.append(f"Hit the {cap:,} splat cap - densification stopped "
                         "early. Gaps and thin coverage usually trace to this: "
                         "the surface needed more gaussians than the budget "
                         "allowed. Try the 'quality' preset (1.2M).")
        elif final:
            _log(f"splat: finished at {final:,} of {cap:,} - the budget was "
                 f"not the limiting factor.")
            notes.append(f"Finished at {final:,} of {cap:,}; the budget was "
                         "not the limiting factor.")
        _last = ck_stats[-1] if ck_stats else {}
        if _last.get("gap_ratio") is not None and _last["gap_ratio"] > 0.35:
            notes.append(f"Gap ratio {_last['gap_ratio']} - a large share of "
                         "the surface shell has no splats beside it. Holes, "
                         "rather than needles or transparency.")
        if (_last.get("below_0_1_pct") or 0) > 30:
            notes.append(f"{_last.get('below_0_1_pct')}% of splats are below "
                         "0.1 opacity - they cost sorting time and add haze "
                         "while contributing almost no colour. The cleanup "
                         "pass drops these.")
        if (_last.get("opacity_mean") or 1) < 0.15:
            notes.append(f"Mean opacity {_last.get('opacity_mean')} - the "
                         "surface is built from near-invisible splats and "
                         "will read as see-through.")
        if (_last.get("aniso_median") or 0) > 12:
            notes.append(f"Median anisotropy {_last.get('aniso_median')} - "
                         "gaussians are needles, not discs.")

        # Brush has no anisotropy constraint, so the trained result is cleaned
        # afterwards: drop near-invisible splats, fatten needles back toward
        # discs, cut floaters. The raw export is kept alongside.
        if r.clean and plys:
            raw_ply = plys[-1]
            cleaned = raw_ply.with_name(raw_ply.stem + "_clean.ply")
            ci = steps.clean_splat_ply(
                raw_ply, cleaned, min_opacity=0.06, max_aniso=12.0,
                max_extent=max(8.0, orbit.radius * 1.6),
                centre=(0.0, 0.0, float(orbit.aim_z)))
            _io("cleaned", raw_ply, cleaned,
                note=f"{ci['in']:,} -> {ci['out']:,} splats")
            _log(f"splat: cleaned {ci['in']:,} -> {ci['out']:,} splats "
                 f"({ci['dropped_low_opacity']:,} below 0.06 opacity, "
                 f"{ci['dropped_far']:,} floaters); {ci['fattened']:,} "
                 f"needles capped at 12:1 aspect. "
                 f"{ci['bytes'] / 1048576:.0f} MB")
            save_state(d, splat_clean=cleaned.name)
        try:
            rep = steps.write_splat_report(d, {
                "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                "outcome": "aborted" if was_aborted else "completed",
                "settings": {"steps": r.steps, "detail": r.detail,
                             "sh_degree": r.sh_degree, "clean": r.clean,
                             "splat_cap": cap, "rerun": r.rerun,
                             "alpha_supervision": not keep_bg,
                             "backdrop": backdrop},
                "dataset": {**(ds_info or {}), "frames": len(names)},
                "matte": ({"source": r.matte_source,
                           "coverage": round(mi.get("coverage", 0), 4),
                           "partial_alpha": round(mi.get("partial_alpha", 0), 4)}
                          if mi else {"source": "none (background kept)"}),
                "verify": v,
                "checkpoints": ck_stats,
                "notes": notes,
                "brush_log_tail": ((_err or "") + (_out or ""))[-2500:],
            })
            _io("wrote", rep, note="training report")
            _log(f"splat: report written to {d.name}/splat_report.md")
        except Exception as e:
            _log(f"splat: could not write the report ({e})", level="warn")

        save_state(d, stage="done", splat=plys[-1].name if plys else None)

    if not _run("splat", work, d.name):
        refuse("splat", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True}


@router.post("/api/splat_abort")
def splat_abort():
    """Stop a running splat training, keeping the checkpoints it reached.

    Only the splat step is abortable. The fal steps are remote work that is
    billed the moment it is submitted, so killing the local wait would throw
    the result away without saving anything.
    """
    rx("/api/splat_abort")
    with LOCK:
        proc = CHILD["proc"]
        stage = JOB.get("stage")
        if proc is None:
            return JSONResponse(
                {"error": ("nothing to abort" if not JOB["running"]
                           else f"the {stage} step cannot be aborted - only "
                                "splat training can")},
                status_code=400)
        CHILD["aborted"] = True
    _log("splat: abort requested - stopping Brush", level="warn")
    try:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _log("splat: did not stop in 10s - killing", level="warn")
            proc.kill()
    except Exception as e:
        return JSONResponse({"error": f"could not stop it: {e}"},
                            status_code=500)
    return {"ok": True}


@router.get("/api/splat_report")
def splat_report(project: str):
    """Every training run recorded for this project, newest last."""
    d = pdir(project)
    f = d / "splat_report.json"
    if not f.exists():
        return {"runs": [], "md": None}
    try:
        runs = json.loads(f.read_text(encoding="utf-8-sig"))
    except Exception as e:
        return JSONResponse({"error": f"report unreadable: {e}"},
                            status_code=500)
    return {"runs": runs,
            "md": f"/projects/{d.name}/splat_report.md"}


@router.get("/api/splats")
def splats(project: str):
    """Checkpoint PLYs for a project, oldest step first.

    Polled during training so the viewer can pick up each new checkpoint as
    Brush writes it, and used to populate the checkpoint switcher afterwards.
    """
    d = pdir(project)
    out = d / "splat_out"
    if not out.exists():
        return {"exports": [], "training": bool(JOB.get("running") and JOB.get("stage") == "splat"
                                     and JOB.get("project") == d.name),
                "dir": str(out)}
    items = steps.list_exports(out)
    # A PLY still being written would parse as truncated garbage; require the
    # size to have settled before offering it.
    now = time.time()
    ready = [i for i in items if now - i["mtime"] > 2.0]
    return {"exports": ready, "training": bool(JOB.get("running") and JOB.get("stage") == "splat"
                                     and JOB.get("project") == d.name),
            "dir": str(out.resolve())}


@router.post("/api/splat_upload")
async def splat_upload(request: Request, project: str):
    """Bring your own splat.

    Lands in splat_out/ beside the trained checkpoints so it can be selected,
    viewed and exported like any other. Validated as a 3DGS PLY first: the
    viewer would otherwise render a mesh or a plain point cloud as noise with
    nothing to explain why.
    """
    d = pdir(project)
    body = await request.body()
    rx("/api/splat_upload", d.name, bytes=len(body))
    if not body:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    out = d / "splat_out"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"upload_{int(time.time())}.ply"
    dst.write_bytes(body)
    try:
        info = steps.inspect_splat_ply(dst)
    except Exception as e:
        dst.unlink(missing_ok=True)
        return JSONResponse({"error": str(e)}, status_code=400)
    _log(f"upload: {dst.name} - {info['splats']:,} splats, "
         f"{info['format']}, SH degree {info['sh_degree']}")
    _io("received", "upload", dst, note="finished splat")
    save_state(d, stage="done", splat=dst.name, splat_uploaded=True)
    return {"ok": True, "name": dst.name, **info,
            "mb": round(len(body) / 1e6, 2)}
