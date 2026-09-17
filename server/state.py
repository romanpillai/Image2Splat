"""Project directories, state.json, and the UI snapshot.

state.json is an INDEX into the files on disk, never the record itself. The
files are the durable thing; when the two disagree, /api/project/rescan makes
the index agree with reality rather than the other way round.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

from fastapi import APIRouter
from pathlib import Path

from . import config
from .jobs import log_push
from .models import UIState

PROJ = config.PROJ
router = APIRouter()


# Settings that describe how the OPERATOR likes to work, and so should follow
# them to a new subject. Everything else in a UI snapshot is measured from, or
# fitted to, one particular photo and must not.
CARRY_OVER_UI = (
    "oFrames", "oSweep", "oFps", "oRad", "oAimZ", "oElevSweep", "oElevCycles",
    "oClay", "oAspect", "oW", "oH", "oFit", "oScene", "oUseCard",
    "oDepth", "oPoints", "oDepthModel", "oDepthStr", "oCull",
    "cSize",
    "gVision", "gInt", "gRes", "gFirst", "gRef",
    "sSteps", "sDetail", "sSH", "sClean",
)


# Windows refuses these as file or directory names, whatever the extension.
# Creating one raises NotADirectoryError deep inside mkdir, which surfaced as a
# bare 500 rather than a sensible message.
_WIN_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def pdir(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in "-_")[:64] or "project"
    if safe.lower() in _WIN_RESERVED:
        safe += "_"          # 'con' -> 'con_', still recognisable
    d = PROJ / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path(d: Path) -> Path:
    return d / "state.json"


def load_state(d: Path) -> dict:
    """Read a project's state, tolerating a hand-edited or BOM-prefixed file.

    utf-8-sig because anything written by a Windows editor or PowerShell carries
    a BOM, which plain json.loads rejects. And a single unreadable state.json
    must not be able to 500 the whole project list.
    """
    p = state_path(d)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as e:
        return {"_state_error": f"{type(e).__name__}: {e}"}


STATE_LOCK = threading.Lock()


def save_state(d: Path, **kw):
    """Merge keys into a project's state, atomically.

    Read-modify-write from several threads at once (a worker finishing while
    /api/ui autosaves the rail) used to lose whole updates -- the later writer
    won wholesale, so a finished job's stage could vanish. And merging onto the
    {"_state_error": ...} stub from an unreadable file would bury the only
    evidence that it was ever corrupt.
    """
    with STATE_LOCK:
        s = load_state(d)
        if "_state_error" in s:
            raise RuntimeError(
                f"refusing to overwrite an unreadable state.json for "
                f"'{d.name}': {s['_state_error']}. Move it aside and use "
                f"Recover to rebuild from the files on disk.")
        s.update(kw)
        # write via a temp file so a crash mid-write cannot truncate the state
        tmp = state_path(d).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(s, indent=2))
        os.replace(tmp, state_path(d))
    # Standing rule: every file written names its full path. state.json is
    # written more often than anything else here, so it is logged at io level
    # with the keys that changed rather than silently.
    log_push(f"file: wrote projects/{d.name}/state.json "
             f"({', '.join(sorted(kw)[:8])}"
             + (" ..." if len(kw) > 8 else "") + f")  @ {state_path(d)}",
             level="io", source="state", project=d.name)
    return s


def has_work(d: Path) -> bool:
    """Does this project hold anything worth preserving?"""
    if any(d.glob("source.*")):
        return True
    st = load_state(d)
    return any(st.get(k) for k in
               ("control_video", "ai_video", "splat", "cutout"))


def next_free_project(stem: str) -> Path:
    """The next unused project name derived from stem: foo, foo-2, foo-3 ..."""
    # Try what was actually asked for FIRST. Stripping "-2" up front meant
    # asking for "shoot-2" could hand back "shoot".
    if not has_work(pdir(stem)):
        return pdir(stem)
    stem = re.sub(r"-\d+$", "", stem) or "project"
    cand = pdir(stem)
    if not has_work(cand):
        return cand
    n = 2
    while True:
        cand = pdir(f"{stem}-{n}")
        if not has_work(cand):
            return cand
        n += 1


def _crop(v) -> tuple | None:
    """Validate an isolate sphere: [x, y, z, r] with r > 0, else None.

    Anything malformed becomes None rather than raising. A bad sphere would
    either crash the lift or -- worse -- silently keep nothing, and "no crop"
    is the safe reading of "I could not understand the crop".
    """
    if not v:
        return None
    try:
        x, y, z, r = (float(t) for t in v)
    except (TypeError, ValueError):
        return None
    return (x, y, z, r) if r > 0 else None


def keeps_background(st: dict) -> bool:
    """Does this project's control video carry a real background?

    "scene" and "panorama" are LEGACY values from the removed 3D set and the
    removed panorama dome. Neither can be produced by this build, and both are
    kept here only so existing projects keep matting correctly. Testing
    `== "scene"` (as three call sites once did) sent the grey boilerplate --
    "no scenery, no landscape, no horizon" -- alongside a control video full of
    scenery.
    """
    return (st.get("backdrop") or "grey") in ("scene", "panorama")


def source_images(d: Path) -> list:
    """Source-image candidates, best first.

    source_upright.png is a valid source ONLY while the current calibration
    actually applied levelling. Preferring it just because it is on disk meant
    a leftover from an earlier run kept winning after the verdict flipped: the
    cutout was rebuilt straight and un-bordered while the reference image went
    on being built from the stale tilted file, black corners and all.
    """
    # GeoCalib levelling is gone from this build, so an old project's
    # source_upright.png is never preferred any more: the photo as uploaded is
    # the source. Its saved calib block is ignored.
    return sorted(q for q in d.glob("source.*")
                  if q.name != "source_upright.png")


# Everything a project accumulates downstream of the source image. Listed once
# so "new project" and "new upload" cannot drift apart -- the bug was that they
# already had, and a fresh image kept showing the previous run's control video.
DERIVED_KEYS = (
    "calib", "calib_time", "cutout", "cutout_info", "suggested_aspect",
    "reference", "reference_info", "prompt_auto", "prompt_auto_scene",
    "prompt_auto_cost", "prompt_auto_model", "prompt_auto_brief",
    "orbit", "orbit_info", "orbit_time", "card_height", "include_card",
    "ai_engine", "ai_frames", "ai_retimed", "splat_uploaded", "ai_nobg",
    "mattes", "authored_alpha", "matte_prompt",
    "control_soft", "control_soft_factor", "control_soft_blur",
    "fal_sent", "fal_engine",
    "backdrop", "control_video", "ai_video", "fal_prompt", "fal_intensity",
    "fal_reference", "matte_info", "dataset_info", "verify", "splat",
    # all derived from the source image, all previously surviving a new upload
    "splat_clean", "fov", "fov_time",
    "control_frame0", "control_mattes",
    "source_w", "source_h",
    # the passes workspace, equally derived from this subject
    "pass_top", "pass_bottom", "pass_ai_top", "pass_ai_bottom",
    "pass_soft_top", "pass_soft_bottom", "dataset_complete",
)


def clear_derived(d: Path, **extra):
    """Reset every derived field to None (0 for the recency stamps)."""
    wipe = {k: (0 if k.endswith("_time") else None) for k in DERIVED_KEYS}
    wipe.update(extra)
    return save_state(d, **wipe)


# ------------------------------------------------------------ UI snapshot ---
@router.post("/api/ui")
def save_ui(r: UIState):
    """Store a snapshot of every rail control for this project.

    Deliberately generic: the client sends {id: value} for the whole rail, so
    new controls persist without anything being added here. This is what makes
    a refresh resume exactly where you were.
    """
    d = pdir(r.project)
    # Not routed through rx(): the rail autosaves on every drag, and one line
    # per slider tick would drown the console this build exists to make
    # readable. The state write itself is still logged at io level.
    log_push(f"POST /api/ui  project={d.name}  keys={len(r.ui)}",
             level="info", source="request", project=d.name)
    # ui_time lets a later calibration override a stored control value. Without
    # it the snapshot always wins and "Calculate camera" silently does nothing.
    save_state(d, ui=r.ui, ui_time=time.time())
    return {"ok": True, "keys": len(r.ui)}
