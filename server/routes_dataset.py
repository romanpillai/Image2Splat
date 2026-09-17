"""dataset_complete/ -- the author ring plus whichever passes are on -- and
the matte models that feed it.

A SEPARATE directory from dataset/. The author set is what the current splat
was trained from and is still the thing to fall back to; a build that
overwrote it would make every earlier result unreproducible the first time a
pass turned out to be bad.
"""
from __future__ import annotations

import shutil
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import steps

from .common import _author_clip_spec
from .jobs import _busy_msg, _io, _log, _run, refuse, rx
from .models import AiRmbgReq, DatasetComplete
from .state import keeps_background, load_state, pdir, save_state

router = APIRouter()


def _ring_plan(d, r: DatasetComplete):
    """(rings, problems) for a complete-dataset build.

    rings is [(label, clip, cam_z, radius)] in the order they will be written.
    Every check that can be made from disk is made HERE, so the build either
    runs clean or explains itself before touching anything.
    """
    st = load_state(d)
    ui = st.get("ui") or {}
    o = st.get("orbit") or {}
    if not o:
        return [], ["no orbit in this project - render the control video first"]
    base = steps.orbit_from_dict(o)
    problems = []

    ai = (st.get("ai_video") or "").strip()
    if not ai or not (d / ai).exists():
        return [], ["no author AI video - the author ring is not optional"]
    if not st.get("approved"):
        problems.append("the author AI video has not been approved yet")

    rings = [("author", d / ai, float(o.get("cam_z") or base.aim_z),
              float(o.get("radius") or base.radius))]
    for want, which, zk, rk in ((r.top, "top", "paTopZ", "paTopR"),
                                (r.bottom, "bottom", "paBotZ", "paBotR")):
        if not want:
            continue
        nm = (st.get("pass_ai_" + which) or "").strip()
        if not nm or not (d / nm).exists():
            problems.append(
                f"{which} pass is ticked but has no AI clip yet - send the "
                f"{which} pass to a video model first")
            continue
        if zk not in ui or rk not in ui:
            problems.append(
                f"{which} pass has no ring geometry saved - open the passes "
                f"tab once so its settings are recorded")
            continue
        rings.append((f"{which} pass", d / nm, float(ui[zk]), float(ui[rk])))

    # Every clip must be the same size and the same length as the author's.
    # ONE COLMAP camera describes the whole set only if that holds, and that
    # is the entire reason the pass control canvas is inherited.
    spec = _author_clip_spec(d)
    for label, q, _z, _r in rings:
        try:
            import cv2
            c = cv2.VideoCapture(str(q))
            w_, h_, n_ = int(c.get(3)), int(c.get(4)), int(c.get(7))
            fps_ = int(round(float(c.get(5)) or 0))
            c.release()
        except Exception as e:
            problems.append(f"{label}: cannot read {q.name} ({e})")
            continue
        if n_ != base.frames:
            problems.append(
                f"{label}: {q.name} has {n_} frames but the ring has "
                f"{base.frames} poses. Every pose would address the wrong "
                f"image - retime it first.")
        if spec and (w_, h_) != (spec[0], spec[1]):
            problems.append(
                f"{label}: {q.name} is {w_}x{h_} but the author clip is "
                f"{spec[0]}x{spec[1]}. One COLMAP camera cannot describe both "
                f"- regenerate it at the author's aspect.")
        if spec and fps_ != spec[3]:
            problems.append(
                f"{label}: {q.name} runs at {fps_} fps against the author's "
                f"{spec[3]}")
    return rings, problems


@router.post("/api/dataset/complete/plan")
def dataset_complete_plan(r: DatasetComplete):
    """What the build would write, before it writes it."""
    d = pdir(r.project)
    st = load_state(d)
    rings, problems = _ring_plan(d, r)
    base = steps.orbit_from_dict(st.get("orbit") or {})
    sp = (st.get("splat") or "").strip()
    ply = d / "splat_out" / sp if sp else None
    have_ply = bool(r.init_from_splat and ply is not None and ply.exists())
    if r.init_from_splat and not have_ply:
        problems.append(
            "no trained splat to seed the init cloud from - untick it to "
            "carve a visual hull instead")
    _log(f"dataset plan: {len(rings)} ring(s), {base.frames * len(rings)} "
         f"poses, {len(problems)} problem(s) -> {d / 'dataset_complete'}",
         source="dataset")
    for p in problems:
        _log(f"dataset plan: PROBLEM {p}", level="warn", source="dataset")
    return {
        "rings": [{"label": lb, "name": q.name, "cam_z": round(z, 4),
                   "radius": round(rr, 4), "frames": base.frames,
                   "path": str(q.resolve())}
                  for lb, q, z, rr in rings],
        "poses": base.frames * len(rings),
        "init": (ply.name if have_ply else "visual hull"),
        "out": "dataset_complete",
        "out_dir": str((d / "dataset_complete").resolve()),
        "problems": problems,
    }


@router.post("/api/dataset/complete")
def dataset_complete(r: DatasetComplete):
    """Write dataset_complete/ - the author ring plus whichever passes are on."""
    rx("/api/dataset/complete", r.project, top=r.top, bottom=r.bottom,
       matte_source=r.matte_source, init_from_splat=r.init_from_splat,
       max_init_points=r.max_init_points)
    d = pdir(r.project)
    st = load_state(d)
    rings, problems = _ring_plan(d, r)
    if problems:
        for p in problems:
            _log(f"dataset: REFUSED - {p}", level="warn", source="dataset")
        return JSONResponse({"error": "; ".join(problems)}, status_code=400)
    base = steps.orbit_from_dict(st.get("orbit") or {})
    keep_bg = keeps_background(st)
    sp = (st.get("splat") or "").strip()
    ply = d / "splat_out" / sp if sp else None

    def work():
        import cv2
        import numpy as np
        out = d / "dataset_complete"
        # A full rebuild, always. A set left over from a run with different
        # rings ticked is indistinguishable on disk from a correct one, and it
        # would train for hours before anyone noticed.
        if out.exists():
            _io("clearing", out, note="previous dataset_complete build")
            shutil.rmtree(out, ignore_errors=True)
            _log("dataset: cleared the previous dataset_complete/")
        imgs = out / "images"
        imgs.mkdir(parents=True, exist_ok=True)

        parts, names, k = [], [], 0
        for label, clip, cam_z, radius in rings:
            _log(f"dataset: {label} - {clip.name}")
            work_fr = d / "_ds_frames"
            shutil.rmtree(work_fr, ignore_errors=True)
            n = steps.mp4_to_frames(clip, work_fr)
            _io("extracted", clip, work_fr, note=f"{n} frames")
            if n != base.frames:
                raise RuntimeError(
                    f"{clip.name} gave {n} frames, expected {base.frames}")
            cut = d / "_ds_cut"
            shutil.rmtree(cut, ignore_errors=True)
            if keep_bg:
                steps.copy_frames_opaque(work_fr, cut)
            else:
                mask_dir = None
                if r.matte_source in steps.MATTE_MODELS:
                    _mp = (st.get("matte_prompt") or "").strip()
                    _fp = steps._fingerprint(clip)
                    # The author ring may already have a matte from a normal
                    # splat run. Its stamp says which clip it was cut from, so
                    # reusing it is safe and saves cutting an identical one.
                    _shared = d / "mattes" / r.matte_source
                    rd = (_shared
                          if steps.stamp_ok(_shared, video=_fp,
                                            model=r.matte_source, prompt=_mp)
                          else d / "mattes_complete" / label.split()[0])
                    if not steps.stamp_ok(rd, video=_fp,
                                          model=r.matte_source, prompt=_mp):
                        _log(f"dataset: {r.matte_source} over {label} frames")
                        steps.matte_dir(work_fr, rd, model=r.matte_source,
                                        log=_log, prompt=_mp)
                        steps.stamp_write(rd, video=_fp,
                                          model=r.matte_source, prompt=_mp)
                        _io("wrote", rd, note=f"mattes cut from {clip.name}")
                    else:
                        _io("reusing", rd, note=f"cut from {clip.name}")
                    mask_dir = rd
                mi = steps.matte(work_fr, mask_dir, cut,
                                 use_distance=(mask_dir is None))
                _log(f"dataset: {label} coverage {100*mi['coverage']:.1f}%")
            # Numbered sequentially across the WHOLE set, in ring order,
            # because that is the order write_colmap pairs poses in.
            for q in sorted(cut.glob("frame_*.png")):
                k += 1
                nm = f"frame_{k:05d}.png"
                shutil.move(str(q), str(imgs / nm))
                names.append(nm)
            shutil.rmtree(work_fr, ignore_errors=True)
            shutil.rmtree(cut, ignore_errors=True)
            parts.append(steps.ring_orbit(base, cam_z, radius, base.frames))

        merged = steps.MergedOrbit(parts)
        im = cv2.imread(str(imgs / names[0]), cv2.IMREAD_UNCHANGED)
        H, W = im.shape[:2]

        if r.init_from_splat and ply is not None and ply.exists():
            _io("reading", ply, note="trained splat, for init points")
            xyz, rgb, op, scale, rot = steps.read_ply_splats(ply)
            pts = xyz[op > 0.1]
            if r.max_init_points and len(pts) > r.max_init_points:
                idx = np.linspace(0, len(pts) - 1,
                                  r.max_init_points).astype(int)
                pts = pts[idx]
            _log(f"dataset: {len(pts)} init points from {ply.name} - this "
                 f"build is a REFINEMENT of that splat, not a fresh solve")
        else:
            pts = steps.visual_hull(imgs, merged)
            _log(f"dataset: {len(pts)} init points from a visual hull carved "
                 f"across {len(parts)} ring(s)")

        ci = steps.write_colmap(merged, out / "sparse" / "0", names, W, H, pts)
        _io("wrote", out / "sparse" / "0",
            note=f"{len(names)} poses over {len(parts)} ring(s), "
                 f"{len(pts)} init points")
        _log(f"dataset: COLMAP {W}x{H} fx {ci['fx']:.1f} fy {ci['fy']:.1f} - "
             f"ONE camera for all {len(parts)} ring(s)")
        _log(f"dataset: ready -> {out}")
        save_state(d, dataset_complete={
            "rings": [lb for lb, _q, _z, _r in rings],
            "poses": len(names), "init_points": int(len(pts)),
            "init_from": (ply.name if (r.init_from_splat and ply
                                       and ply.exists()) else "visual hull"),
            "width": W, "height": H, "built": time.time()})

    if not _run("dataset-complete", work, d.name):
        refuse("dataset-complete", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True}


# ------------------------------------------------------------------ mattes --
@router.get("/api/matte_models")
def matte_models():
    """The matte models on offer, with what each costs to fetch."""
    return {"models": [{"id": k, **v} for k, v in steps.MATTE_MODELS.items()]}


@router.post("/api/ai_rmbg")
def ai_rmbg(r: AiRmbgReq):
    """Cut the background out of the AI video, for viewing and for training.

    Writes ai_rmbg/ (RGBA frames, the actual matte) and ai_nobg.mp4 (the same
    frames on black, because alpha does not survive an mp4 and the point is to
    SEE whether the mask is any good before training on it).
    """
    rx("/api/ai_rmbg", r.project, model=r.model, prompt=r.prompt)
    d = pdir(r.project)
    s = load_state(d)
    if not s.get("ai_video"):
        return JSONResponse({"error": "no AI video in this project"},
                            status_code=400)
    if r.model not in steps.MATTE_MODELS:
        return JSONResponse(
            {"error": f"unknown matte model {r.model!r} "
                      f"(have {list(steps.MATTE_MODELS)})"}, status_code=400)
    frames = d / "ai_frames"
    spec = steps.MATTE_MODELS[r.model]

    def work():
        ai = d / s["ai_video"]
        fp = steps._fingerprint(ai)
        if not steps.stamp_ok(frames, video=fp):
            _log("matte: extracting frames from the AI video")
            steps.mp4_to_frames(ai, frames)
            steps.stamp_write(frames, video=fp)
            _io("extracted", ai, frames)
        else:
            _io("reusing", frames, note=f"already cut from {ai.name}")
        # Belt and braces: even with the stamp, refuse to matte a frame count
        # that disagrees with the clip. Every frame would be paired with the
        # wrong moment, and the run is long enough that noticing late is
        # expensive.
        have = len(list(frames.glob("frame_*.png")))
        want, _fps = steps.video_frame_count(ai)
        if have != want:
            raise RuntimeError(
                f"{frames.name}/ holds {have} frames but {ai.name} has {want}. "
                f"Refusing to matte mismatched frames - delete {frames.name}/ "
                f"and run this again.")
        _log(f"matte: {spec['label']} over the AI frames")
        _io("reading", frames, note=f"{have} frames")
        # Each model keeps its own directory and its own preview, so several
        # can be compared on the same footage instead of overwriting.
        mdir = d / "mattes" / r.model
        n = steps.matte_dir(frames, mdir, model=r.model, log=_log,
                            prompt=r.prompt)
        steps.stamp_write(mdir, video=fp, model=r.model,
                          prompt=(r.prompt or "").strip())
        _io("wrote", frames, mdir, note=f"{n} RGBA mattes")
        vid = f"nobg_{r.model}.mp4"
        steps.rmbg_preview_mp4(mdir, d / vid, fps=int(s.get("fps") or 24))
        _io("wrote", mdir, d / vid, note="preview, subject on black")
        cov = amb = 0.0
        import cv2 as _cv2
        got = sorted(mdir.glob("frame_*.png"))[:24]
        for q in got:
            a = _cv2.imread(str(q), _cv2.IMREAD_UNCHANGED)
            if a is None or a.ndim != 3 or a.shape[2] != 4:
                continue
            al = a[:, :, 3].astype("float32") / 255.0
            cov += float((al > 0.5).mean())
            amb += float(((al > 0.02) & (al < 0.98)).mean())
        k = max(len(got), 1)
        _log(f"matte: {n} frames; coverage {100*cov/k:.1f}%, "
             f"ambiguous edge pixels {100*amb/k:.2f}% "
             f"(lower is a more decisive edge)")
        have_m = dict(load_state(d).get("mattes") or {})
        if r.prompt:
            save_state(d, matte_prompt=r.prompt)
        have_m[r.model] = {"video": vid, "label": spec["label"],
                           "prompt": r.prompt,
                           "coverage": round(cov / k, 4),
                           "ambiguous": round(amb / k, 4)}
        save_state(d, mattes=have_m, ai_nobg=vid)

    if not _run("matte", work, d.name):
        refuse("matte", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True}


@router.post("/api/authored_alpha")
def authored_alpha(r: AiRmbgReq):
    """Render the control-render coverage as a video, so it can be compared.

    Not a model output: this is what the renderer actually drew, and it is the
    reference the segmenters are trying to reproduce.
    """
    rx("/api/authored_alpha", r.project)
    d = pdir(r.project)
    cm = d / "control_mattes"
    if not (cm.exists() and any(cm.glob("frame_*.png"))):
        return JSONResponse(
            {"error": "no authored coverage for this project - re-render the "
                      "control video to produce it"}, status_code=400)

    def work():
        s = load_state(d)
        steps.authored_alpha_mp4(cm, d / "authored_alpha.mp4",
                                 fps=int(s.get("fps") or 24))
        _io("wrote", cm, d / "authored_alpha.mp4", note="white = subject")
        _log("matte: authored_alpha.mp4 written (white = subject)")
        save_state(d, authored_alpha="authored_alpha.mp4")

    if not _run("matte", work, d.name):
        refuse("matte", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True}
