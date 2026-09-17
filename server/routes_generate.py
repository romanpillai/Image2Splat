"""fal: the approval plan, the send, auto-describe, retime, approve.

The contract this module exists to keep: **/api/generate/plan and
/api/generate resolve their inputs through the SAME functions.** If the
preview resolved files its own way it would be theatre -- it could show one
set and the request could send another, which is the failure the approval step
exists to catch.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import falclient
import steps

from .common import (_author_clip_spec, _extra_refs, _media_url,
                     _pass_aspect_check, _pass_of, _pass_or_control,
                     _reference_for, _soft_stamp, _soften_on, REF_CAPS,
                     control_first_frame, list_refs, refs_dir, soft_path)
from .jobs import (_busy_msg, _io, _log, _project_busy, _run, refuse, rx)
from .models import DescribeReq, GenReq, NewProject, RefDel, RetimeReq
from .state import keeps_background, load_state, pdir, save_state, source_images

router = APIRouter()


# ---------------------------------------------------------- extra references --
@router.get("/api/refs")
def get_refs(project: str):
    d = pdir(project)
    return {"refs": list_refs(d), "caps": REF_CAPS}


@router.post("/api/refs/add")
async def add_ref(request: Request, project: str, ext: str = "png"):
    """Raw body upload of one extra reference image."""
    d = pdir(project)
    ext = (ext or "png").lower().lstrip(".")
    if ext not in {"png", "jpg", "jpeg", "webp"}:
        ext = "png"
    body = await request.body()
    rx("/api/refs/add", d.name, ext=ext, bytes=len(body))
    if not body:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    rd = refs_dir(d)
    rd.mkdir(parents=True, exist_ok=True)
    # Name by TIME so the ordering is stable and a delete cannot make the next
    # upload collide with a name that is still on disk.
    f = rd / f"ref_{int(time.time() * 1000)}.{ext}"
    f.write_bytes(body)
    _io("saved", f, note=f"extra reference image ({len(body) / 1e6:.2f} MB)")
    return {"refs": list_refs(d), "added": f.name}


@router.post("/api/refs/delete")
def del_ref(r: RefDel):
    rx("/api/refs/delete", r.project, name=r.name)
    d = pdir(r.project)
    f = refs_dir(d) / Path(r.name).name          # no traversal
    if not f.exists():
        return JSONResponse({"error": f"no such reference {r.name!r}"},
                            status_code=400)
    _io("deleting", f, note="extra reference image")
    f.unlink()
    return {"refs": list_refs(d), "deleted": f.name}


# ---------------------------------------------------------------- softening --
@router.post("/api/soften")
def soften(r: GenReq):
    """Build the softened control video, which is also what gets uploaded.

    Deliberately ONE artefact: the clip you preview is the clip that is sent.
    Rendering a separate preview would let the two drift, and then the
    approval panel is showing you something other than the request.
    """
    rx("/api/soften", r.project, downsample=r.ref_downsample, blur=r.ref_blur,
       engine=r.engine)
    d = pdir(r.project)
    src = d / "control.mp4"
    if not src.exists():
        return JSONResponse({"error": f"no {src.name} in this project"},
                            status_code=400)
    dst = soft_path(d)
    stamp = _soft_stamp(r, d)

    if not _soften_on(r):
        # Softening off: remove the artefact so nothing stale can be sent.
        for q in (dst, dst.with_suffix(dst.suffix + ".built_from.json")):
            if q.exists():
                _io("deleting", q, note="softening off - nothing stale may be sent")
                q.unlink()
        save_state(d, control_soft=None, control_soft_factor=None,
                   control_soft_blur=None)
        return {"started": False, "off": True}

    if steps.stamp_ok_file(dst, **stamp):
        _log(f"soften: {dst} is already built from these exact settings")
        return {"started": False, "cached": True, "output": dst.name}

    def work():
        s0 = load_state(d)
        info = steps.soften_video(src, dst, factor=float(r.ref_downsample),
                                  blur=float(r.ref_blur),
                                  fps=int(s0.get("fps") or 24), log=_log)
        steps.stamp_write_file(dst, **stamp)
        _io("softened", src, dst,
            note=f"{info['src_w']}x{info['src_h']} -> "
                 f"{info['width']}x{info['height']}, blur {info['blur_px']}px")
        _log("soften: this exact file is what gets uploaded - the reference "
             "image and first frame stay sharp")
        save_state(d, control_soft=dst.name,
                   control_soft_factor=float(r.ref_downsample),
                   control_soft_blur=float(r.ref_blur))

    if not _run("soften", work, d.name):
        refuse("soften", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True, "output": dst.name}


# ----------------------------------------------------------- the fal inputs --
def _fal_inputs(r, d, s):
    """Exactly which local files this request will upload, per engine.

    The approval panel and the real submit BOTH call this. If the preview
    resolved files its own way it would be theatre: it could show one set and
    the request could send another, which is the failure the approval step
    exists to catch.

    Returns (files, problems) where files is [{role, name, path, bytes}].
    """
    files, problems = [], []

    def add(role, path):
        if path is None:
            return
        p = Path(path)
        if not p.exists():
            problems.append(f"{role}: {p.name} is not on disk")
            return
        files.append({"role": role, "name": p.name, "path": str(p.resolve()),
                      "bytes": p.stat().st_size,
                      "media": _media_url(p, d)})

    def pick(which):
        if which == "frame0":
            return d / "control_frame0.png"
        if which == "source":
            q = source_images(d)
            return q[0] if q else None
        return None

    def softened(orig):
        """The clip actually sent: the softened one when softening is on.

        Refuses rather than silently sending the sharp original -- the whole
        point of the approval panel is that what it lists is what goes.
        """
        if not _soften_on(r):
            return orig
        sp = soft_path(d)
        if not steps.stamp_ok_file(sp, **_soft_stamp(r, d)):
            problems.append(
                "the softened control video is missing or was built from "
                "different settings - press Generate again and it is rebuilt "
                "first")
            return orig
        return sp

    if r.engine == "wan22vace":
        # A PASS overrides the control choice. This branch used to reach
        # straight for control.mp4, so asking for a pass through VACE silently
        # sent the AUTHOR's control video under the pass's name. Softening
        # applies here too, via _pass_or_control.
        ctl = _pass_or_control(r, d, problems)
        if r.vace_control == "depth":
            # The depth pass was removed in this build. Say so rather than
            # sending the point render under the name "depth pass".
            problems.append(
                "the depth pass was removed in this build - the point render "
                "is what will be sent as the control video")
        # _pass_or_control has already applied the pass's own softening; the
        # author softened() keys off control.mp4 and would look for the wrong
        # artefact entirely.
        add("control video", ctl if _pass_of(r) else softened(ctl))
        # reference and first frame stay SHARP: they say what the subject
        # looks like, which is the one thing not to throw away.
        add("reference image",
            (source_images(d) or [None])[0] if _pass_of(r)
            else pick(r.vace_reference))
        for q in _extra_refs(r, d, problems):
            add("extra reference", q)
        # No first frame for a pass. control_frame0.png is the AUTHOR's ring
        # at frame 0 -- a different camera from either pass -- so anchoring
        # frame 1 to it asks the model to start somewhere the pass never goes.
        add("first frame", None if _pass_of(r) else pick(r.vace_first_frame))
        n = int((s.get("orbit") or {}).get("frames") or 0)
        lo, hi = falclient.VACE_FRAME_RANGE
        if n and not (lo <= n <= hi):
            problems.append(f"this orbit is {n} frames; VACE takes {lo}-{hi}")
    else:
        ctl = _pass_or_control(r, d, problems)
        _pass_aspect_check(r, d, problems)
        add("control video", ctl if _pass_of(r) else softened(ctl))
        add("reference image", _reference_for(r, d))   # sharp, deliberately
        for q in _extra_refs(r, d, problems):
            add("extra reference", q)
    return files, problems


@router.post("/api/generate/plan")
def generate_plan(r: GenReq):
    """What WOULD be sent. Uploads nothing, costs nothing."""
    d = pdir(r.project)
    s = load_state(d)
    eng = falclient.ENGINES.get(r.engine)
    files, problems = _fal_inputs(r, d, s)
    n = int((s.get("orbit") or {}).get("frames") or 0)
    settings = {"engine": r.engine,
                "label": eng["label"] if eng else r.engine,
                "endpoint": eng["endpoint"] if eng else "?",
                "frames": n,
                "prompt": (r.prompt or "").strip(),
                # The COMPLETE string fal receives, boilerplate and all. The
                # box on screen shows your words; what actually goes is a fixed
                # preamble with those glued on the end, and the difference is
                # invisible until it is shown. Approval means nothing if it is
                # approving something other than what is sent.
                # the COMPLETE string, including the pass camera clause, so
                # the approval panel cannot show one prompt and send another
                "fal_prompt": falclient.build_prompt(
                    r.prompt, scene=keeps_background(s),
                    pass_view=_pass_of(r))}
    if r.engine == "wan22vace":
        settings.update(resolution=r.vace_resolution, steps=r.vace_steps,
                        guidance=r.vace_guidance, sampler=r.vace_sampler,
                        seed=r.vace_seed,
                        control=(f"{_pass_of(r)} pass render" if _pass_of(r)
                                 else "point render"))
    elif r.engine == "wan":
        settings.update(resolution=r.wan_resolution, aspect=r.wan_aspect)
    elif r.engine == "minimax":
        settings.update(resolution=r.mm_resolution, aspect=r.mm_aspect,
                        expansion=r.mm_expansion)
    else:
        settings.update(resolution=r.resolution, intensity=r.intensity)
    _log(f"plan: {r.engine} for '{d.name}'"
         + (f" ({_pass_of(r)} pass)" if _pass_of(r) else "")
         + f" - {len(files)} file(s), {len(problems)} problem(s)",
         source="plan")
    for f in files:
        _io("would send", f["path"], source="plan",
            note=f"{f['role']}, {f['bytes'] / 1e6:.2f} MB")
    for p in problems:
        _log(f"plan: PROBLEM {p}", level="warn", source="plan")
    return {"files": files, "problems": problems, "settings": settings,
            "project_dir": str(d.resolve()),
            "total_bytes": sum(f["bytes"] for f in files)}


@router.post("/api/generate")
def do_generate(r: GenReq):
    rx("/api/generate", r.project, engine=r.engine, pass_=r.pass_,
       downsample=r.ref_downsample, blur=r.ref_blur,
       resolution=r.resolution, reference_mode=r.reference_mode,
       prompt_chars=len((r.prompt or "").strip()))
    d = pdir(r.project)
    if not (d / "control.mp4").exists():
        return JSONResponse({"error": "render the orbit first"}, status_code=400)
    if not falclient.have_key():
        return JSONResponse(
            {"error": "FAL_KEY is not set in this server's environment. "
                      "Set it and restart the server."}, status_code=400)

    s = load_state(d)
    nframes = int((s.get("orbit") or {}).get("frames") or 0)
    if nframes <= 0:
        # falclient skips BOTH the duration cap and num_frames when this is
        # falsy, so fal would fall back to its own 121-frame default and every
        # pose would address the wrong image. Refuse rather than pay for that.
        return JSONResponse(
            {"error": "no camera rig recorded for this project - re-render "
                      "the orbit before generating"}, status_code=400)
    _fps = int(s.get("fps") or 24)

    # Which image anchors frame 0.
    #
    # These are two different jobs: the control video supplies GEOMETRY and
    # MOTION, image_url supplies the PHOTOREAL LOOK. Sending the composited
    # reference while the 3D set is on put grey marker boxes into the
    # appearance reference, telling the model that crude grey boxes are the
    # target look. The set belongs in the control video only.
    def _pick_reference():
        # Delegates to _reference_for so the approval panel and this path can
        # never disagree; only the explanatory text lives here.
        ref = _reference_for(r, d)
        if r.reference_mode == "control":
            if control_first_frame(d) is not None:
                return ref, "control video frame 0 (same camera and canvas)"
            _log("fal: no control frame 0 on disk (re-render the orbit to "
                 "produce one) - falling back to the source image", level="warn")
        if source_images(d):
            return ref, "source (original, with background)"
        return ref, "composited (legacy reference on disk)"

    eng = falclient.ENGINES.get(r.engine)
    if eng is None:
        return JSONResponse(
            {"error": f"unknown engine {r.engine!r} "
                      f"(expected one of {list(falclient.ENGINES)})"},
            status_code=400)

    def work():
        # Cleared per job: the panel must show what THIS render sent, not a
        # running total across renders.
        falclient.sent_reset()
        _log(f"fal: engine {r.engine} -> {eng['endpoint']}")
        if not eng["pose_exact"]:
            _log("fal: WARNING this engine is NOT pose-exact. " + eng["note"],
                 level="warn")
        ref, why = _pick_reference()
        if ref is None or not ref.exists():
            raise RuntimeError("no reference image - re-render the orbit. "
                               "(cutout.png must NOT be sent: its RGB still "
                               "contains the original background under the "
                               "alpha, which fal flattens away.)")
        _log(f"fal: first-frame reference = {ref.name} | {why}")
        _io("resolved", ref, note=f"reference image | {why}")
        if (r.reference_mode == "control"
                and (s.get("orbit_info") or {}).get("clay") not in (None, "colour")):
            _log("fal: WARNING clay shading was on for this render, so frame 0 "
                 "is grey clay and carries no appearance. Pick 'source image' "
                 "as the first frame, or re-render with clay off.", level="warn")
        scene = keeps_background(s)
        if r.engine == "wan22vace":
            # Resolved by the SAME function the approval panel used, so what
            # was approved is what goes.
            _files, _probs = _fal_inputs(r, d, s)
            # A depth-pass note is advice, not a fault: the point render is a
            # perfectly good VACE control. Everything else still refuses.
            _hard = [p for p in _probs if "depth pass was removed" not in p]
            if _hard:
                raise RuntimeError("; ".join(_hard))
            for p in _probs:
                if p not in _hard:
                    _log("fal: " + p, level="warn")
            _by = {q["role"]: Path(q["path"]) for q in _files}
            ctl = _by.get("control video")
            if ctl is None:
                raise RuntimeError("no control video resolved for this request")
            ref_img = _by.get("reference image")
            ff_img = _by.get("first frame")
            ref_why = r.vace_reference
            ff_why = r.vace_first_frame

            # Say exactly what is going to fal, before it goes.
            _io("sending", ctl, note=f"control video, {nframes} frames")
            _log(f"fal: control        = {ctl.name} (point render)")
            _log(f"fal: reference      = "
                 + (f"{ref_img.name} ({ref_why}) - holds identity"
                    if ref_img else "none"))
            _log(f"fal: first frame    = "
                 + (f"{ff_img.name} ({ff_why}) - anchors frame 1"
                    if ff_img else "none"))
            _log(f"fal: frames         = {nframes}, matched to the control "
                 f"video (no retime needed)")
            _log(f"fal: resolution     = {r.vace_resolution} | "
                 f"steps {r.vace_steps} | guidance {r.vace_guidance} | "
                 f"sampler {r.vace_sampler}"
                 + (f" | seed {r.vace_seed}" if r.vace_seed is not None
                    else " | seed random"))
            if r.vace_negative.strip():
                _log(f"fal: negative       = {r.vace_negative.strip()[:120]}")
            if ref_img is not None:
                _io("sending", ref_img, note="reference image")
            if ff_img is not None:
                _io("sending", ff_img, note="first frame")

            res = falclient.submit_vace(
                ctl, r.prompt, ref_image=ref_img, first_frame=ff_img,
                num_frames=nframes, match_input_frames=True, fps=_fps,
                resolution=r.vace_resolution, steps=r.vace_steps,
                guidance=r.vace_guidance, sampler=r.vace_sampler,
                seed=r.vace_seed, negative_prompt=r.vace_negative,
                scene=scene, extra_images=_extra_refs(r, d),
                pass_view=_pass_of(r), on_log=_log)
        elif r.engine == "wan":
            if r.wan_smart_duration:
                secs = None
                _log("fal: smart duration - Wan picks the length itself")
            elif r.wan_duration:
                secs = int(round(float(r.wan_duration)))
                if abs(float(r.wan_duration) - secs) > 1e-9:
                    _log(f"fal: duration {r.wan_duration}s rounded to {secs}s "
                         "(Wan takes whole seconds)")
                else:
                    _log(f"fal: duration {secs}s (explicit)")
            else:
                # Measure the control video rather than assume: fps may have
                # changed since the orbit was recorded.
                cn, cfps = steps.video_frame_count(d / "control.mp4")
                adv = falclient.wan_exact_duration_options(
                    cn, int(round(cfps)) or _fps)
                secs = adv["request"]
                if adv["exact"]:
                    _log(f"fal: control video is exactly {adv['seconds']}s "
                         f"-> duration {secs}s")
                else:
                    _log(f"fal: control video is {adv['seconds']}s but Wan "
                         f"takes whole seconds -> requesting {secs}s "
                         f"({adv['error_pct']}% off). For an exact match use "
                         f"{adv['frames_fix']} frames at "
                         f"{int(round(cfps))} fps.")
            _log("fal: the returned FRAME count is still Wan's choice, so "
                 "the clip will need retiming before the splat step",
                 level="warn")
            _ctl = _pass_or_control(r, d)
            _io("sending", _ctl, note="control video")
            res = falclient.submit_wan(
                _ctl, ref, r.prompt, duration_s=secs,
                resolution=r.wan_resolution, aspect_ratio=r.wan_aspect,
                audio=r.wan_audio,
                prompt_expansion=r.wan_prompt_expansion,
                enable_thinking=r.wan_thinking, seed=r.wan_seed,
                scene=scene, extra_images=_extra_refs(r, d),
                pass_view=_pass_of(r), on_log=_log)
        elif r.engine == "minimax":
            lo, hi = falclient.MINIMAX_DURATION_RANGE
            if r.mm_smart_duration:
                secs = None
                _log("fal: smart duration - MiniMax picks the length itself")
            elif r.mm_duration:
                secs = int(round(float(r.mm_duration)))
                _log(f"fal: duration {secs}s (explicit)")
            else:
                cn, cfps = steps.video_frame_count(d / "control.mp4")
                exact = cn / max(cfps, 1e-6)
                secs = int(round(exact))
                secs = max(lo, min(hi, secs))
                if abs(exact - secs) < 0.01:
                    _log(f"fal: control video is exactly {exact:.2f}s "
                         f"-> duration {secs}s")
                else:
                    _log(f"fal: control video is {exact:.2f}s but MiniMax "
                         f"takes whole seconds in {lo}-{hi} -> requesting "
                         f"{secs}s. Frame i will not be pose i either way.")
            _log("fal: MiniMax treats the control video as a MOTION reference "
                 "and picks its own frame count, so the clip WILL need "
                 "retiming before the splat step", level="warn")
            _ctl = _pass_or_control(r, d)
            _io("sending", _ctl, note="control video")
            res = falclient.submit_minimax(
                _ctl, ref, r.prompt, duration_s=secs,
                resolution=r.mm_resolution, aspect_ratio=r.mm_aspect,
                prompt_expansion=r.mm_expansion, seed=r.mm_seed,
                scene=scene, extra_images=_extra_refs(r, d),
                pass_view=_pass_of(r), on_log=_log)
        else:
            _log(f"fal: num_frames={nframes} (server default is 121 - a "
                 f"mismatch would misalign every pose), "
                 f"resolution={r.resolution} short side")
            _ctl = _pass_or_control(r, d)
            _io("sending", _ctl, note=f"control video, {nframes} frames")
            res = falclient.submit(_ctl, ref,
                                   r.prompt, r.intensity, r.detail_refine,
                                   num_frames=nframes, fps=_fps,
                                   resolution=r.resolution, scene=scene,
                                   video_quality=r.video_quality,
                                   pass_view=_pass_of(r), on_log=_log)
        url = falclient.result_video_url(res)
        if not url:
            raise RuntimeError(f"no video in fal response: {str(res)[:300]}")
        _io("receiving", url)
        # A pass gets its own name so the two passes and the authored clip
        # can all coexist: pass_top_ai1.mp4, pass_bottom_ai1.mp4, ai_v1.mp4.
        _w = _pass_of(r)
        _stem = f"pass_{_w}_ai" if _w else "ai_v"
        # max, not count: with ai_v1/2/3 present, deleting ai_v2 made this
        # return 3 and silently overwrite ai_v3 -- destroying a paid render.
        _used = [int("".join(c for c in q.stem if c.isdigit()) or 0)
                 for q in d.glob(f"{_stem}*.mp4")]
        n = max(_used, default=0) + 1
        dst = d / f"{_stem}{n}.mp4"
        falclient.download(url, dst)
        _io("saved", url, dst)
        got, got_fps = steps.video_frame_count(dst)
        _log(f"fal: saved {dst.name} - {got} frames at {got_fps:g} fps "
             f"(orbit has {nframes} poses)")
        if got != nframes:
            _log(f"fal: WARNING frame count {got} != {nframes} poses. The "
                 "splat step will refuse this until it is retimed - use "
                 "'Retime to pose count' in Review.", level="warn")
        # Recorded from falclient's own upload calls, so it cannot disagree
        # with what actually went -- which is the entire point of showing it.
        _sent = falclient.sent_list()
        for q in _sent:
            _log(f"fal: SENT {q['role']:16s} {q['name']}  "
                 f"({q['bytes'] / 1e6:.2f} MB)")
        # A PASS must not overwrite the authored clip. Writing ai_video here
        # would replace the video the author pipeline trained from and reset
        # the stage to "review", quietly undoing that work -- the pass is a
        # SEPARATE artefact and is recorded as one.
        if _w:
            # The returned clip is saved EXACTLY as fal sent it.
            #
            # An earlier version conformed it here to the author's AI video.
            # That was the wrong lever: the thing that has to match is the
            # CONTROL video's canvas, which is already inherited from the
            # author AI video (size, frame count and fps). Get the input right
            # and the return matches on its own; resizing the output only
            # papers over an input that did not.
            #
            # The aspect guard on the send is what keeps that true, and
            # /api/pass/conform is still there to repair a clip that came back
            # wrong -- but it is something you press, not something that
            # happens to your footage.
            _spec = _author_clip_spec(d)
            if _spec and (got != _spec[2] or got_fps != _spec[3]):
                _log(f"pass: NOTE {dst.name} is {got} frames at {got_fps:g} "
                     f"fps; the author clip is {_spec[2]} at {_spec[3]}. "
                     f"Left as returned - use 'Conform to the author clip' if "
                     f"you want them to match.", level="warn")
            save_state(d, fal_sent=_sent, fal_engine=r.engine,
                       **{f"pass_ai_{_w}": dst.name},
                       fal_prompt=falclient.build_prompt(
                           r.prompt, scene=scene, pass_view=_w),
                       fal_reference=ref.name)
            _log(f"pass: {_w} AI clip saved as {dst.name} - the authored "
                 f"ai_video is untouched")
        else:
            save_state(d, fal_sent=_sent, fal_engine=r.engine,
                       stage="review", ai_video=dst.name, approved=False,
                       ai_engine=r.engine, ai_frames=got,
                       fal_prompt=falclient.build_prompt(r.prompt, scene=scene),
                       fal_intensity=r.intensity, fal_reference=ref.name)

    # Remote lane: this uploads and then blocks on fal for minutes without
    # touching the GPU, so it must not hold the slot the local work needs.
    if not _run("generate", work, d.name, lane="remote"):
        refuse("generate", r.project, _project_busy(d.name) or "a job")
        return JSONResponse(
            {"error": f"{d.name} is already running "
                      f"{_project_busy(d.name) or 'a job'}"}, status_code=409)
    return {"started": True}


# --------------------------------------------------------------- describe ---
@router.post("/api/describe")
def do_describe(r: DescribeReq):
    # An explicit "" from the client overrides the field default, and fal
    # answers "No models provided" only after the image has been uploaded.
    if not (r.model or "").strip():
        r.model = falclient.VISION_MODELS[0]
    rx("/api/describe", r.project, model=r.model, pass_=r.pass_,
       brief_chars=len((r.extra or "").strip()))
    d = pdir(r.project)
    if not falclient.have_key():
        return JSONResponse({"error": "FAL_KEY not set"}, status_code=400)

    st = load_state(d)
    scene = keeps_background(st)

    # A brief from the operator is welcome; the model's OWN previous output is
    # not. They arrive in the same box, because that box is where a caption
    # lands and where a person then edits it.
    #
    # So subtract: whatever of the stored caption still appears in the box is
    # the model talking to itself and is removed. What survives is what the
    # person actually wrote. Feeding a caption back is what made describe
    # re-describe the previous subject, and no wording in the prompt fixes
    # that -- the text simply must not be sent.
    brief = (r.extra or "").strip()
    prev = (st.get("prompt_auto") or "").strip()
    if brief and prev:
        if brief == prev:
            brief = ""
        elif prev in brief:
            brief = brief.replace(prev, " ").strip()
        else:
            # prompt_auto now ENDS with the previous brief, so the box is
            # "caption + brief". Subtract the caption half on its own, or the
            # brief would be sent, appended, and doubled on every press.
            head = prev.split("\n\n")[0].strip()
            if head and head in brief:
                brief = brief.replace(head, " ").strip()
    # A fragment shorter than this is punctuation left behind by the subtraction
    if len(brief) < 3:
        brief = ""
    r.extra = brief

    # Which image to caption depends on what the prompt has to cover.
    #
    # grey  -> control frame 0: the subject exactly as the video model will
    #          receive it, on the flat grey it will actually see. It cannot
    #          describe scenery that is not in frame, so "standing in a
    #          landscape" cannot leak in and fight the boilerplate's
    #          "empty background".
    # scene -> the ORIGINAL source (legacy projects only).
    f0 = control_first_frame(d)
    # control_frame0.png is DERIVED from the source. Replace the photo without
    # re-rendering the orbit and it still sits there, older than the thing it
    # is supposed to depict -- measured on one project as 15 hours out of
    # date, so the caption described the previous subject. A derived file is
    # only usable while it is newer than what it was derived from.
    # A CLAY render carries no appearance. Captioning it produces "smooth
    # matte grey clay", which is a description of the render rather than of
    # the subject, and materials are the single thing the video model most
    # needs. The source photo is the only image that still holds them.
    clay = (st.get("orbit_info") or {}).get("clay")
    clay_note = ""
    if f0 is not None and clay not in (None, "colour") and source_images(d):
        clay_note = (f"describe: clay shading ({clay}) was on for this render, "
                     f"so frame 0 is grey clay with no materials in it - "
                     f"captioning the SOURCE photo instead.")
        f0 = None

    stale_f0 = ""
    if f0 is not None and f0.exists():
        srcs = [q for q in source_images(d) if q.exists()]
        newest_src = max((q.stat().st_mtime for q in srcs), default=0.0)
        if newest_src and f0.stat().st_mtime < newest_src:
            age = (newest_src - f0.stat().st_mtime) / 3600.0
            stale_f0 = (f"describe: control_frame0.png is {age:.1f}h OLDER than "
                        f"the source photo - it shows the previous subject, so "
                        f"captioning the source instead. Re-render the orbit to "
                        f"refresh it.")
            f0 = None
    if scene:
        cands = [q for q in source_images(d)]
        if f0 is not None:
            cands.append(f0)
    else:
        cands = ([f0] if f0 is not None else []) + list(source_images(d))
    cands += [d / "reference.png", d / "cutout.png"]   # legacy projects
    for img in cands:
        if img is not None and img.exists():
            break
    else:
        return JSONResponse(
            {"error": "nothing to describe yet - upload an image, and render "
                      "the orbit so there is a frame 0 to caption"},
            status_code=400)

    def work():
        _log(f"describe: {falclient.VISION_ENDPOINT} | {r.model} "
             f"| mode={'subject+setting' if scene else 'subject only'}")
        if clay_note:
            _log(clay_note)
        if stale_f0:
            _log(stale_f0, level="warn")
        _io("sending", img, note=f"to {falclient.VISION_ENDPOINT}")
        # The single most useful line when a description comes back describing
        # something else: extra is injected as "context, honour it", so any
        # text here steers the model away from the image.
        _log("describe: brief = "
             + (repr(r.extra)[:180] if r.extra
                else "(none - the image alone)"))
        res = falclient.describe_image(img, model=r.model, extra=r.extra,
                                       scene=scene, on_log=_log)
        cost = res.get("cost")
        _log(f"describe: {res['tokens'] or '?'} tokens"
             + (f", ${cost:.5f}" if isinstance(cost, (int, float)) else ""))
        _log(f"describe: {res['text']}")
        # The caption REPLACES the box, so a brief written there would be
        # consumed and thrown away -- which is what happened to "flash
        # photography style, reflections of the light source". Worse, the
        # subject prompt forbids mentioning lighting at all, so the vision
        # model could not have carried it even if asked.
        #
        # A style note is not a subject description and does not belong inside
        # one. It is kept verbatim and appended, so it survives into the text
        # the VIDEO model actually receives, which is where it does its work.
        text = res["text"]
        if r.extra:
            text = text + "\n\n" + r.extra
            _log(f"describe: your brief is kept verbatim in the prompt - "
                 f"{r.extra[:110]!r}")
        save_state(d, prompt_auto=text, prompt_auto_model=r.model,
                   prompt_auto_brief=r.extra or None,
                   prompt_auto_cost=cost, prompt_auto_scene=scene)
        # Show the WHOLE prompt as fal would receive it, right here at describe
        # time. The boilerplate is only attached at send time, so "the exact
        # prompt" was invisible until the approval window -- a later step, and
        # not the one being watched when the description lands.
        falclient._log_prompt(_log, text, scene, pass_view=_pass_of(r))

    if not _run("describe", work, d.name, lane="remote"):
        refuse("describe", r.project, _project_busy(d.name) or "a job")
        return JSONResponse(
            {"error": f"{d.name} is already running "
                      f"{_project_busy(d.name) or 'a job'}"}, status_code=409)
    return {"started": True}


@router.get("/api/vision_models")
def vision_models():
    return {"models": falclient.VISION_MODELS,
            "endpoint": falclient.VISION_ENDPOINT}


@router.get("/api/engines")
def list_engines():
    """The video engines, and whether each preserves the pose binding."""
    return {"engines": [{"id": k, **v} for k, v in
                        falclient.ENGINES.items()],
            "wan_resolutions": list(falclient.WAN_RESOLUTIONS),
            "wan_aspects": list(falclient.WAN_ASPECTS),
            "mm_resolutions": list(falclient.MINIMAX_RESOLUTIONS),
            "mm_aspects": list(falclient.MINIMAX_ASPECTS),
            "mm_expansion": list(falclient.MINIMAX_EXPANSION),
            "mm_duration_range": list(falclient.MINIMAX_DURATION_RANGE)}


# ----------------------------------------------------------------- review ---
@router.post("/api/ai_upload")
async def ai_upload(request: Request, project: str, ext: str = "mp4"):
    """Bring your own AI video.

    Lands beside the generated ones as the next ai_vN, so an uploaded clip and
    a fal-rendered one are reviewed, retimed and trained identically. The
    frame count is recorded but NOT enforced here -- the splat gate is the one
    place that decides whether a clip may be trained on.
    """
    d = pdir(project)
    rx("/api/ai_upload", d.name, ext=ext)
    # A job in flight is READING this project's files. Swapping the AI video
    # underneath it left a matte grinding through 150 frames of the previous
    # clip while state said the new 120-frame one was current -- it wedged,
    # and the work was wrong anyway.
    busy = _project_busy(d.name)
    if busy:
        refuse("ai_upload", d.name, f"running {busy}")
        return JSONResponse(
            {"error": f"{d.name} is running {busy} right now. Wait for it or "
                      f"abort it before replacing the AI video - swapping it "
                      f"mid-job makes that job process the wrong clip."},
            status_code=409)
    if not load_state(d).get("orbit"):
        return JSONResponse(
            {"error": "render the orbit first - an AI video is only "
                      "meaningful against a recorded camera rig"},
            status_code=400)
    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    ext = "".join(c for c in (ext or "mp4").lower() if c.isalnum())[:4]
    if ext not in ("mp4", "mov", "webm", "mkv", "m4v"):
        return JSONResponse({"error": f"unsupported video type {ext!r}"},
                            status_code=400)
    used = [int("".join(c for c in q.stem if c.isdigit()) or 0)
            for q in d.glob("ai_v*.mp4")]
    dst = d / f"ai_v{max(used, default=0) + 1}.mp4"
    if ext == "mp4":
        dst.write_bytes(body)
    else:
        tmp = d / f"_upload.{ext}"
        tmp.write_bytes(body)
        try:
            subprocess.run([steps._ffmpeg(), "-y", "-i", str(tmp),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "20", "-preset", "veryfast", str(dst)],
                           check=True, capture_output=True)
        finally:
            tmp.unlink(missing_ok=True)
    try:
        got, fps = steps.video_frame_count(dst)
    except Exception as e:
        dst.unlink(missing_ok=True)
        return JSONResponse({"error": f"not a readable video: {e}"},
                            status_code=400)
    orbit = steps.orbit_from_dict(load_state(d).get("orbit", {}))
    _log(f"upload: {dst.name} - {got} frames at {fps:g} fps "
         f"(orbit has {orbit.frames} poses)")
    save_state(d, stage="review", ai_video=dst.name, approved=False,
               ai_engine="upload", ai_frames=got)
    _io("received", f"upload ({len(body) / 1e6:.2f} MB)", dst,
        note=f"{got} frames, treated as a rendered clip")
    return {"ok": True, "ai_video": dst.name, "frames": got,
            "fps": round(fps, 3), "poses": orbit.frames,
            "matches": got == orbit.frames}


@router.post("/api/ai_retime")
def ai_retime(r: RetimeReq):
    """Resample the current AI video onto the orbit's pose count.

    The ONE retime this build keeps. The keyframed remap editor was removed;
    this is the fix that is actually needed, because the reference-to-video
    engines return their own frame rate -- measured, Wan hands back 150 frames
    at 30 fps for a 120-pose ring, whatever the control video said.

    Only a frame-COUNT fix. It assumes the clip sweeps the authored path
    linearly in time; if it does not, the counts will agree while the geometry
    still disagrees. Keeps the original as ai_vN_orig.mp4 so nothing paid for
    is destroyed.
    """
    rx("/api/ai_retime", r.project)
    d = pdir(r.project)
    s = load_state(d)
    if not s.get("ai_video"):
        return JSONResponse({"error": "no AI video in this project"},
                            status_code=400)
    ai = d / s["ai_video"]
    if not ai.exists():
        return JSONResponse({"error": "ai video missing"}, status_code=400)
    orbit = steps.orbit_from_dict(s.get("orbit", {}))
    fps = int(s.get("fps") or 24)

    def work():
        keep = ai.with_name(ai.stem + "_orig.mp4")
        if not keep.exists():
            shutil.copy2(ai, keep)
            _io("backed up", ai, keep, note="original, never overwritten")
        tmp = ai.with_name(ai.stem + "_retimed.mp4")
        info = steps.retime_to_frames(ai, tmp, orbit.frames, fps=fps)
        import os as _os
        _os.replace(tmp, ai)
        _io("retimed", keep, ai,
            note=f"{info['from_frames']} -> {info['to_frames']} frames at "
                 f"{fps} fps, matching the {orbit.frames}-pose orbit")
        _log(f"retime: {info['from_frames']} -> {info['to_frames']} frames "
             f"at {fps} fps")
        _log("retime: frame COUNT now matches the poses. This assumes the "
             "clip follows the authored orbit linearly in time - if it does "
             "not, the splat will be wrong in a way nothing can detect.",
             level="warn")
        # Everything downstream was cut against the OLD timing. ai_frames is
        # re-extracted by the splat step, but the matte directories are only
        # rebuilt when empty -- left alone, frame i of a 150-frame matte would
        # be paired with frame i of a 120-frame resample, which is a different
        # moment in the sweep. Wrong, and nothing downstream can detect it.
        for stale in (d / "ai_frames", d / "mattes", d / "ai_rmbg"):
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
                _io("cleared", stale, note="cut against the old timing")
        for old in list(d.glob("nobg_*.mp4")) + [d / "ai_nobg.mp4"]:
            if old.exists():
                old.unlink()
        save_state(d, ai_frames=orbit.frames, ai_retimed=True,
                   mattes={}, ai_nobg=None)

    if not _run("retime", work, d.name):
        refuse("retime", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True}


@router.post("/api/approve")
def approve(p: NewProject):
    rx("/api/approve", p.name)
    d = pdir(p.name)
    s = load_state(d)
    if not s.get("ai_video"):
        return JSONResponse({"error": "nothing to approve"}, status_code=400)
    save_state(d, approved=True, stage="splat")
    _log(f"approve: {s['ai_video']} is approved - the splat step may now run")
    return {"approved": True, "ai_video": s["ai_video"]}
