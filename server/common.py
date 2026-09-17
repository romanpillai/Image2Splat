"""Resolvers shared by more than one route module.

Every function here answers a question that BOTH an approval preview and the
request it previews must answer the same way -- "which control video", "which
reference image", "what does the author clip look like". Two copies of a
fallback chain is how a panel ends up promising frame 0 while the request
quietly sends the source, so there is exactly one copy and it lives here.
"""
from __future__ import annotations

from pathlib import Path

import steps

from .jobs import _log
from .state import load_state, source_images

# Some engines take an ARRAY of reference images, not just one. Wan 3.0 Prime
# allows 10, MiniMax H3 allows 9, VACE declares no cap. LTX takes a single
# image_url and cannot use these at all -- which is why REF_CAPS drives the UI
# rather than the UI guessing.
REF_CAPS = {"wan": 10, "minimax": 9, "wan22vace": 8, "ltx": 0}

# The aspect options the reference-to-video engines expose, as ratios.
ASPECT_RATIOS = {"9:16": 9 / 16, "3:4": 3 / 4, "1:1": 1.0,
                 "4:3": 4 / 3, "16:9": 16 / 9, "21:9": 21 / 9}


# ------------------------------------------------------------- references ---
def refs_dir(d: Path) -> Path:
    return d / "refs"


def list_refs(d: Path) -> list[dict]:
    """Extra reference images for this project, oldest first.

    The FOLDER is the record, not a list in state.json. A list would drift the
    moment a file was removed by hand, and then the request would upload a name
    that is not there -- the class of bug the approval panel exists to catch.
    """
    rd = refs_dir(d)
    if not rd.is_dir():
        return []
    out = []
    for f in sorted(rd.iterdir()):
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            out.append({"name": f.name, "bytes": f.stat().st_size,
                        "url": f"/projects/{d.name}/refs/{f.name}"})
    return out


# ------------------------------------------------------------------ passes ---
# The pass control video is rendered from the TRAINED SPLAT, in the browser,
# because that is where the only splat renderer lives. The client steps the
# pass camera frame by frame and posts each frame to /api/pass/frame.
#
# Rendering the splat rather than re-lifting the point cloud is deliberate: the
# splat is the thing that has already been trained and looked at, so a pass
# generated from it is consistent with what exists rather than with a fresh
# depth estimate that may disagree.
def pass_dir(d: Path, which: str) -> Path:
    return d / f"pass_frames_{'top' if which == 'top' else 'bottom'}"


def pass_video(d: Path, which: str) -> Path:
    return d / f"pass_{'top' if which == 'top' else 'bottom'}.mp4"


def soft_path(d: Path) -> Path:
    return d / "control_soft.mp4"


def pass_soft_path(d: Path, which: str) -> Path:
    """The softened copy of one pass. Per pass, not shared.

    Two passes look at different amounts of the subject from different
    distances, so the amount of detail worth throwing away is not the same for
    both -- and one shared file would silently mean softening the top pass
    un-softened the bottom one.
    """
    return d / f"pass_{which}_soft.mp4"


def _pass_soft_stamp(r, d, which: str):
    """What a softened pass must have been built from to still be valid."""
    return {"src": steps._fingerprint(pass_video(d, which)),
            "factor": round(float(r.ref_downsample), 3),
            "blur": round(float(r.ref_blur), 3)}


def _soft_stamp(r, d):
    """What the softened clip must have been built from to still be valid.

    Always control.mp4 now: the depth pass, and with it control_depth.mp4, was
    removed in this build.
    """
    return {"src": steps._fingerprint(d / "control.mp4"),
            "factor": round(float(r.ref_downsample), 3),
            "blur": round(float(r.ref_blur), 3)}


def _soften_on(r) -> bool:
    return float(r.ref_downsample) > 1.0 or float(r.ref_blur) > 0.0


# ----------------------------------------------------------------- media ---
def _media_url(p: Path, d: Path) -> str:
    """A browser-playable URL for a file inside a project, or "".

    The approval window shows the ACTUAL media rather than a filename, so a
    wrong clip is obvious at a glance instead of being a name you have to
    recognise.
    """
    try:
        rel = p.resolve().relative_to(d.resolve()).as_posix()
    except Exception:
        return ""
    if p.suffix.lower() not in (".mp4", ".png", ".jpg", ".jpeg", ".webp"):
        return ""
    stamp = int(p.stat().st_mtime) if p.exists() else 0
    return f"/projects/{d.name}/{rel}?v={stamp}"


def _dir_preview(md: Path, d: Path) -> str:
    """The viewable stand-in for a directory of frames.

    A matte directory is thousands of PNGs; its preview video is the thing a
    person can actually judge.
    """
    for cand in (d / f"nobg_{md.name}.mp4", d / "authored_alpha.mp4"):
        if cand.exists():
            return _media_url(cand, d)
    first = sorted(md.glob("frame_*.png"))
    return _media_url(first[0], d) if first else ""


def control_first_frame(d: Path) -> Path | None:
    """Frame 0 of the control video, as an image.

    Prefers the lossless PNG the renderer wrote before encoding; falls back to
    the card path's frame_0001.png, and finally decodes the mp4 itself for
    projects rendered before either existed.
    """
    direct = d / "control_frame0.png"
    if direct.exists():
        return direct
    legacy = d / "control_frames" / "frame_0001.png"
    if legacy.exists():
        return legacy
    mp4 = d / "control.mp4"
    if not mp4.exists():
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(str(mp4))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        cv2.imwrite(str(direct), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return direct
    except Exception:
        return None


# ------------------------------------------------ the author clip, and passes ---
def _author_clip_spec(d: Path):
    """(width, height, frames, fps) of the author's AI video, or None.

    The single definition of what a pass must match. Measured off the file
    rather than read from settings: fal returns its own size and its own frame
    count, and the settings are what was ASKED for.
    """
    st = load_state(d)
    ai = (st.get("ai_video") or "").strip()
    q = d / ai if ai else None
    if q is None or not q.exists():
        return None
    try:
        import cv2
        c = cv2.VideoCapture(str(q))
        w, h = int(c.get(3)), int(c.get(4))
        n, fps = int(c.get(7)), float(c.get(5)) or 0.0
        c.release()
    except Exception:
        return None
    if w <= 0 or h <= 0 or n <= 0:
        return None
    return w, h, n, int(round(fps)) or int(st.get("fps") or 24)


def _pass_of(r) -> str:
    """"top" / "bottom" when this request is an orbit pass, else ""."""
    v = str(getattr(r, "pass_", "") or "").lower()
    return v if v in ("top", "bottom") else ""


def _pass_or_control(r, d, problems=None) -> Path:
    """The control video for this request: a pass render, or the authored one.

    Softening a pass is OFF by default and deliberately opt-in. The reasoning
    that made it off-by-default still holds -- softening exists to stop the
    model copying the point render's texture, and a pass is a render of the
    TRAINED SPLAT, whose appearance is the thing worth keeping. But a splat has
    artefacts of its own (needles, speckle, thin holes) that are just as much
    not-the-subject, and throwing detail away is the same lever for both. It is
    a choice per pass, so make it a choice.
    """
    w = _pass_of(r)
    if not w:
        return d / "control.mp4"
    q = pass_video(d, w)
    if not q.exists():
        if problems is not None:
            problems.append(f"no {w} pass rendered yet - render it first")
        return q
    if not _soften_on(r):
        return q
    sp = pass_soft_path(d, w)
    if not steps.stamp_ok_file(sp, **_pass_soft_stamp(r, d, w)):
        # Refuse to guess. Sending the sharp original while the panel promised
        # a softened one is exactly the drift the approval panel exists to stop.
        if problems is not None:
            problems.append(
                f"the softened {w} pass is missing or was built from different "
                f"settings - press Preview softening in the passes tab")
        return q
    return sp


def _extra_refs(r, d, problems=None) -> list[Path]:
    """The extra reference images this ENGINE will actually accept.

    Capped here rather than at the UI, because the cap is a property of the
    endpoint and the request is what has to honour it. Over the cap the extras
    are dropped and said so -- silently truncating would mean the approval
    panel listed images that never got sent.
    """
    cap = REF_CAPS.get(r.engine, 0)
    have = [refs_dir(d) / x["name"] for x in list_refs(d)]
    if cap <= 0:
        if have and problems is not None:
            problems.append(
                f"{r.engine} takes a single reference image, so the "
                f"{len(have)} extra one(s) will NOT be sent - switch engine "
                f"or remove them")
        return []
    # the main reference occupies one slot of the endpoint's array
    room = max(0, cap - 1)
    if len(have) > room:
        if problems is not None:
            problems.append(
                f"{r.engine} accepts {cap} reference images including the "
                f"main one, so only the first {room} extras will be sent "
                f"({len(have)} present)")
        have = have[:room]
    return have


def author_aspect_option(d: Path) -> str | None:
    """The engine aspect option matching the author's AI video, or None.

    The pass CONTROL video is already the author AI video's canvas -- same
    size, same frame count, same fps. Ask the engine for that same shape and
    the return matches too, and all three rings share one COLMAP camera.
    Ask for a different one and the return is a different shape, which nothing
    downstream can fix without either cropping picture away or stretching it.
    So it is checked here, before anything is spent.
    """
    spec = _author_clip_spec(d)
    if not spec:
        return None
    want = spec[0] / max(spec[1], 1)
    best, err = None, 1e9
    for name, ratio in ASPECT_RATIOS.items():
        e = abs(ratio - want)
        if e < err:
            best, err = name, e
    # 2% is the whole tolerance: 1:1 and 4:3 are 33% apart, so anything that
    # does not land on an option is genuinely not one of them.
    return best if err <= 0.02 * want else None


def _pass_aspect_check(r, d, problems):
    """Warn when a pass would come back at an aspect the author clip is not."""
    if not _pass_of(r) or problems is None:
        return
    want = author_aspect_option(d)
    if want is None:
        return
    got = None
    if r.engine == "wan":
        got = r.wan_aspect
    elif r.engine == "minimax":
        got = r.mm_aspect
    if got is None or got == want:
        return
    spec = _author_clip_spec(d)
    problems.append(
        f"aspect is set to {got!r} but the author clip is {spec[0]}x{spec[1]} "
        f"({want}), and so is the control video this pass sends. A different "
        f"aspect comes back a different shape from every other clip in the "
        f"project, and the merged dataset then needs a separate COLMAP camera "
        f"for it - set aspect to {want} instead")


def _reference_for(r, d):
    """The appearance reference for LTX and Wan 3.0.

    Its own function so the approval preview and the submit path resolve it
    once, together. Two copies of a fallback chain is how a panel ends up
    promising frame 0 while the request quietly sends the source.
    """
    # A PASS always references the SOURCE PHOTO.
    #
    # control_frame0.png is frame 0 of the AUTHOR's point-cloud render: clay
    # shaded, and shot from the author's ring, not from above or below. Sent
    # with a pass it tells the model two false things at once -- that the
    # subject is a white statue, and that this is the camera it was seen from.
    # The source photo is the only image in the project that says what the
    # subject actually looks like, which is the entire job of a reference.
    if _pass_of(r):
        srcs = source_images(d)
        if srcs:
            _log(f"pass: reference image is the source photo {srcs[0].name}, "
                 f"not control_frame0.png (that is the author's clay render "
                 f"from the author's camera)")
            return srcs[0]
        _log("pass: no source photo in this project - falling back",
             level="warn")
    elif r.reference_mode == "control":
        f0 = control_first_frame(d)
        if f0 is not None:
            return f0
    srcs = source_images(d)
    if srcs:
        return srcs[0]
    return d / "reference.png"

