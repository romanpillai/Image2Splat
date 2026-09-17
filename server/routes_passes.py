"""The passes workspace: two extra orbits rendered from the TRAINED splat.

Verified, not assumed: a pass lives in the author's COLMAP world. The pass
viewport uses the same camera formula as `steps.Orbit.poses()` -- same -90
degree start azimuth, same aim point, same world-up -- and comparing all 360
camera centres against a three-ring Orbit gave a worst deviation of 9.45e-15
world units. No registration, no ICP, no pose solving: the poses were never in
separate coordinate systems.

Which is why every clip must be the SAME SIZE. COLMAP describes a lens in
PIXELS, so the same 29.8 degree lens is focal 1172.6 at 624x624 and 1804.0 at
960x960. Keep them identical and ONE camera describes all three rings.
"""
from __future__ import annotations

import os
import shutil

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import steps

from .common import (_author_clip_spec, _pass_soft_stamp, _soften_on,
                     author_aspect_option, pass_dir, pass_soft_path,
                     pass_video)
from .jobs import _busy_msg, _io, _log, _run, refuse, rx
from .models import PassConform, PassEncode, PassRetime, PassSoften, RefDel
from .state import load_state, pdir, save_state

router = APIRouter()


@router.post("/api/pass/begin")
def pass_begin(r: RefDel):
    """Clear the frame folder for one pass. r.name carries top|bottom."""
    rx("/api/pass/begin", r.project, which=r.name)
    d = pdir(r.project)
    fr = pass_dir(d, r.name)
    if fr.exists():
        _io("clearing", fr, note="previous pass frames")
    shutil.rmtree(fr, ignore_errors=True)
    fr.mkdir(parents=True, exist_ok=True)
    _log(f"pass: {r.name} - collecting frames into {fr}")
    return {"ok": True, "dir": str(fr)}


@router.post("/api/pass/frame")
async def pass_frame(request: Request, project: str, which: str, i: int):
    """One rendered frame, raw PNG bytes.

    NOT routed through rx(): 120 frames arrive one at a time and a line each
    would bury everything else. /api/pass/begin and /api/pass/encode bracket
    the sequence and both log their full paths.
    """
    d = pdir(project)
    fr = pass_dir(d, which)
    fr.mkdir(parents=True, exist_ok=True)
    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty frame"}, status_code=400)
    # frame_%04d to match steps.frames_to_mp4's input pattern
    (fr / f"frame_{int(i):04d}.png").write_bytes(body)
    return {"ok": True}


@router.post("/api/pass/encode")
def pass_encode(r: PassEncode):
    rx("/api/pass/encode", r.project, which=r.which, fps=r.fps)
    d = pdir(r.project)
    fr = pass_dir(d, r.which)
    files = sorted(fr.glob("frame_*.png"))
    if not files:
        return JSONResponse(
            {"error": f"no frames collected for the {r.which} pass"},
            status_code=400)
    out = pass_video(d, r.which)

    def work():
        # Normalise the names first. frames_to_mp4 reads frame_%04d.png, and
        # an earlier version of this wrote %05d -- renumbering means passes
        # rendered before that change still encode instead of erroring.
        for k, f in enumerate(sorted(fr.glob("frame_*.png"))):
            want = fr / f"frame_{k:04d}.png"
            if f != want:
                f.rename(want)
        # H.264 via the SAME helper every other video here uses. The first
        # version wrote mp4v through cv2.VideoWriter: the file was valid and
        # the frames were correct, but no browser will decode mp4v in a
        # <video> tag, so every preview showed black while the render was fine.
        _io("reading", fr, note=f"{len(files)} rendered pass frames")
        steps.frames_to_mp4(fr, out, fps=max(1, r.fps))
        _io("wrote", out,
            note=f"{len(files)} frames at {r.fps} fps, H.264 "
                 f"({out.stat().st_size / 1e6:.2f} MB)")
        _log(f"pass: {r.which} video ready -> {out}")
        save_state(d, **{f"pass_{r.which}": out.name})

    if not _run(f"pass-{r.which}", work, d.name):
        refuse(f"pass-{r.which}", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True, "output": out.name}


@router.post("/api/pass/soften")
def pass_soften(r: PassSoften):
    """Build the softened copy of one pass -- the file that then gets uploaded.

    The same single-artefact rule as /api/soften: the clip you preview IS the
    clip that is sent. A separate preview would be free to drift from the
    request, and then approving means nothing.
    """
    rx("/api/pass/soften", r.project, which=r.which,
       downsample=r.ref_downsample, blur=r.ref_blur)
    if r.which not in ("top", "bottom"):
        return JSONResponse({"error": f"unknown pass {r.which!r}"},
                            status_code=400)
    d = pdir(r.project)
    src = pass_video(d, r.which)
    if not src.exists():
        return JSONResponse(
            {"error": f"no {r.which} pass rendered yet - render it first"},
            status_code=400)
    dst = pass_soft_path(d, r.which)
    stamp = _pass_soft_stamp(r, d, r.which)

    if not _soften_on(r):
        # Off: delete the artefact, so a stale softened file can never be sent
        # in place of the sharp one it no longer matches.
        for q in (dst, dst.with_suffix(dst.suffix + ".built_from.json")):
            if q.exists():
                _io("deleting", q, note="softening off for this pass")
                q.unlink()
        _log(f"pass: {r.which} softening off - {dst.name} removed, the sharp "
             f"{src.name} is what gets sent")
        save_state(d, **{f"pass_soft_{r.which}": None})
        return {"started": False, "off": True}

    if steps.stamp_ok_file(dst, **stamp):
        _log(f"pass: {dst} is already built from these exact settings")
        return {"started": False, "cached": True, "output": dst.name}

    def work():
        s0 = load_state(d)
        # The pass's OWN fps, not the author's: they are the same number today
        # because the pass inherits it, but reading it from the clip means they
        # cannot silently come apart.
        try:
            import cv2
            c = cv2.VideoCapture(str(src))
            fps = int(round(float(c.get(5)) or 0)) or int(s0.get("fps") or 24)
            c.release()
        except Exception:
            fps = int(s0.get("fps") or 24)
        info = steps.soften_video(src, dst, factor=float(r.ref_downsample),
                                  blur=float(r.ref_blur), fps=fps, log=_log)
        steps.stamp_write_file(dst, **stamp)
        _io("softened", src, dst,
            note=f"{info['src_w']}x{info['src_h']} -> "
                 f"{info['width']}x{info['height']}, blur {info['blur_px']}px, "
                 f"{fps} fps")
        _log(f"pass: this exact file is what gets uploaded for the {r.which} "
             f"pass - the reference image stays sharp")
        save_state(d, **{f"pass_soft_{r.which}": dst.name})

    if not _run(f"pass-soften-{r.which}", work, d.name):
        refuse(f"pass-soften-{r.which}", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True, "output": dst.name}


@router.post("/api/pass/conform")
def pass_conform(r: PassConform):
    """Repair a returned pass clip that does not match the author's AI video.

    Nothing does this automatically. The control video already carries the
    author clip's canvas, so a correctly-configured pass comes back matching
    and this is never needed. It is here for the ones that did not -- an old
    clip, a clip brought in by hand, an engine that ignored the aspect -- and
    it is destructive enough to be a decision: resizing across aspects
    stretches the picture.

    Measured on a real 832x480 return conformed to 624x624: cover + centre
    crop keeps 58% of the width, enough to cut a limb off the subject.
    Rejected. Plain resize keeps everything but stretches the pixels.
    """
    rx("/api/pass/conform", r.project, which=r.which)
    if r.which not in ("top", "bottom"):
        return JSONResponse({"error": f"unknown pass {r.which!r}"},
                            status_code=400)
    d = pdir(r.project)
    st = load_state(d)
    ai = (st.get(f"pass_ai_{r.which}") or "").strip()
    src = d / ai if ai else None
    if src is None or not src.exists():
        return JSONResponse(
            {"error": f"no AI clip for the {r.which} pass yet"},
            status_code=400)
    spec = _author_clip_spec(d)
    if spec is None:
        return JSONResponse(
            {"error": "no author AI video to conform to - generate one in the "
                      "author tab first"}, status_code=400)
    w_, h_, n_, f_ = spec

    def work():
        # Keep the raw return exactly once. Conforming a clip that has already
        # been conformed would otherwise overwrite the only original there is.
        orig = src.with_name(src.stem + "_orig" + src.suffix)
        if not orig.exists():
            src.replace(orig)
            _io("backed up", src, orig, note="raw fal return, kept once")
        ci = steps.conform_video(orig, src, w_, h_, n_, f_, log=_log)
        _io("conformed", orig, src,
            note=f"{ci['from_size'][0]}x{ci['from_size'][1]} "
                 f"{ci['from_frames']}f @{ci['from_fps']:g} -> "
                 f"{w_}x{h_} {n_}f @{f_}, matching the author AI video")
        _log(f"pass: {r.which} AI clip now matches the author clip exactly - "
             f"the merged dataset can use one COLMAP camera")

    if not _run(f"pass-conform-{r.which}", work, d.name):
        refuse(f"pass-conform-{r.which}", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True}


@router.post("/api/pass/retime")
def pass_retime(r: PassRetime):
    """Resample a returned pass clip onto the ring's pose count.

    The same count-only fix the author tab does, and it carries the same
    warning: it assumes the clip sweeps the authored ring linearly in time. If
    it does not, the counts agree while the geometry still disagrees, and
    nothing downstream can detect that.

    Needed because the reference-to-video engines return their own rate --
    measured here, Wan hands back 150 frames at 30 fps for a 120-pose ring at
    24, whatever the control video says.

    The original is kept as *_orig.mp4. Any matte cut against the old timing is
    cleared: frame i of a 150-frame matte is a different moment in the sweep
    from frame i of a 120-frame resample.
    """
    rx("/api/pass/retime", r.project, which=r.which)
    if r.which not in ("top", "bottom"):
        return JSONResponse({"error": f"unknown pass {r.which!r}"},
                            status_code=400)
    d = pdir(r.project)
    st = load_state(d)
    nm = (st.get("pass_ai_" + r.which) or "").strip()
    clip = d / nm if nm else None
    if clip is None or not clip.exists():
        return JSONResponse(
            {"error": f"no AI clip for the {r.which} pass yet"},
            status_code=400)
    spec = _author_clip_spec(d)
    if spec is None:
        return JSONResponse(
            {"error": "no author AI video to match - generate one first"},
            status_code=400)
    n_want, fps_want = spec[2], spec[3]

    def work():
        keep = clip.with_name(clip.stem + "_orig.mp4")
        if not keep.exists():
            shutil.copy2(clip, keep)
            _io("backed up", clip, keep, note="original, never overwritten")
        tmp = clip.with_name(clip.stem + "_retimed.mp4")
        info = steps.retime_to_frames(keep, tmp, n_want, fps=fps_want)
        os.replace(tmp, clip)
        _io("retimed", keep, clip,
            note=f"{info['from_frames']} -> {info['to_frames']} frames at "
                 f"{fps_want} fps, matching the author clip")
        _log(f"pass retime: {r.which} {info['from_frames']} -> "
             f"{info['to_frames']} frames at {fps_want} fps, matching the "
             f"author clip")
        _log("pass retime: frame COUNT now matches the poses. This assumes "
             "the clip follows the authored ring linearly in time - if it "
             "does not, the splat will be wrong in a way nothing can detect.",
             level="warn")
        stale = d / "mattes_complete" / r.which
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
            _io("cleared", stale, note="cut against the old timing")

    if not _run(f"pass-retime-{r.which}", work, d.name):
        refuse(f"pass-retime-{r.which}", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True}


@router.get("/api/pass/status")
def pass_status(project: str):
    """The two passes, plus the PARENT they must match.

    The parent is the author workspace's AI video. Its size is whatever fal
    actually returned -- not the authored canvas, which fal is free to ignore
    (measured here: a 1024x1024 canvas came back as 624x624). A pass rendered
    at a different aspect would be reconciled by the video model with a warp or
    a crop, so the children take their dimensions from the parent, never from
    a setting.
    """
    d = pdir(project)
    st = load_state(d)
    out = {}

    def card(q):
        if q is None or not q.exists():
            return None
        return {"name": q.name, "bytes": q.stat().st_size,
                "url": f"/projects/{d.name}/{q.name}",
                "path": str(q.resolve()),
                "mtime": q.stat().st_mtime}

    for w in ("top", "bottom"):
        out[w] = card(pass_video(d, w))
        # What fal SENT BACK for this pass. Reported here beside the control
        # render it came from, because the pair is the thing worth looking at:
        # a result on its own cannot tell you whether the model followed the
        # camera. Named pass_ai_<w> so it can never be mistaken for ai_video,
        # which belongs to the author pipeline.
        ai = (st.get(f"pass_ai_{w}") or "").strip()
        c_ai = card(d / ai if ai else None)
        if c_ai is not None:
            # Measured, and compared against the author clip. A pass that does
            # not match needs its own COLMAP camera, which is the thing the
            # conform rule exists to avoid -- so say so plainly rather than
            # letting it surface hours later as a dataset that will not build.
            try:
                import cv2
                c = cv2.VideoCapture(str(d / ai))
                c_ai.update(width=int(c.get(3)), height=int(c.get(4)),
                            frames=int(c.get(7)),
                            fps=round(float(c.get(5)) or 0.0, 3))
                c.release()
                spec = _author_clip_spec(d)
                c_ai["matches"] = bool(spec) and (
                    c_ai["width"], c_ai["height"], c_ai["frames"],
                    int(round(c_ai["fps"]))) == tuple(spec)
            except Exception:
                c_ai["matches"] = False
        out[f"ai_{w}"] = c_ai
        # The softened copy, with WHAT IT WAS BUILT FROM. The numbers matter
        # more than the file: "there is a softened top pass" is not useful
        # unless it says 2x and 4px, which is what makes it stale or not.
        sp = pass_soft_path(d, w)
        c_ = card(sp)
        if c_:
            try:
                import json as _json
                b = _json.loads(sp.with_suffix(sp.suffix + ".built_from.json")
                                .read_text(encoding="utf-8"))
                c_["factor"] = b.get("factor")
                c_["blur"] = b.get("blur")
                c_["stale"] = (b.get("src") or {}) != steps._fingerprint(
                    pass_video(d, w))
            except Exception:
                c_["stale"] = True
        out[f"soft_{w}"] = c_

    parent = None
    ai = (st.get("ai_video") or "").strip()
    q = d / ai if ai else None
    if q is not None and q.exists():
        try:
            import cv2
            c = cv2.VideoCapture(str(q))
            w_, h_ = int(c.get(3)), int(c.get(4))
            fps = float(c.get(5)) or 0.0
            n = int(c.get(7))
            c.release()
            if w_ > 0 and h_ > 0:
                parent = {"name": ai, "width": w_, "height": h_,
                          "fps": round(fps, 3), "frames": n,
                          "source": "author AI video"}
        except Exception as e:
            _log(f"pass: could not measure {ai} ({e})", level="warn")
    if parent is None:
        o = st.get("orbit") or {}
        w_, h_ = int(o.get("width") or 0), int(o.get("height") or 0)
        if w_ > 0 and h_ > 0:
            parent = {"name": None, "width": w_, "height": h_,
                      # The orbit's own pose count and the fps the control
                      # video was rendered at. Zero here left the passes with
                      # nothing to inherit but the markup default.
                      "fps": float(st.get("fps") or 24),
                      "frames": int(o.get("frames") or 0),
                      "source": "orbit canvas - no AI video yet"}
    out["parent"] = parent
    out["author_aspect"] = author_aspect_option(d)
    spec = _author_clip_spec(d)
    # The author's own AI clip, playable. It is the thing every pass is
    # measured against, so the passes workspace should be able to SHOW it
    # rather than only quote its numbers.
    ai_name = (st.get("ai_video") or "").strip()
    ac = card(d / ai_name) if ai_name else None
    if ac is not None and spec is not None:
        ac.update(width=spec[0], height=spec[1], frames=spec[2], fps=spec[3])
    out["author_clip"] = ac
    # And the raw return it was derived from, when one was kept. The author
    # pipeline retimes fal's output (150 frames at 30 fps became 120 at 24 on
    # this project) and keeps the original beside it.
    from pathlib import Path as _P
    orig = (d / f"{_P(ai_name).stem}_orig{_P(ai_name).suffix}"
            if ai_name else None)
    out["author_orig"] = card(orig)
    out["project_dir"] = str(d.resolve())
    return out
