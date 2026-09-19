"""Health, status, VRAM -- and /api/log, the console the frontend renders.

/api/log is the point of this build's logging work. The frontend polls it with
the last sequence number it saw and renders what has arrived since, filtered by
level. Levels are load-bearing, not decoration:

  info   what happened
  warn   what happened that you should look at
  error  what failed, and it carries the job's error stamp
  io     a file was written, read or sent -- with its FULL PATH
"""
from __future__ import annotations

from fastapi import APIRouter

import falclient
import steps

from . import config
from .jobs import (JOB, LOCK, REMOTE, _job_for, _log, log_entries, rx)

router = APIRouter()


@router.get("/api/health")
def health():
    """What the server is, and what it is pointing at.

    The config block is here on purpose: the single most common failure of a
    path-in-a-file design is that the file names something that is not there,
    and the answer should be one request away rather than buried in a boot
    message nobody scrolled back to.
    """
    return {"fal_key": falclient.have_key(),
            "key_source": falclient.key_source(),
            "brush": config.BRUSH.exists(),
            "sharp": steps.sharp_available(),
            "da3": steps.da3_available(),
            "endpoint": falclient.ENDPOINT,
            "build": "beta",
            "config": config.summary()}


@router.get("/api/log")
def get_log(since: int = 0, limit: int = 500, level: str = ""):
    """The recent log, structured.

    `since` is a SEQUENCE number, not a timestamp or an index -- the client
    passes back the last seq it rendered and gets exactly what it has not seen.
    Indexes would shift as the ring rolls, and two entries can share a
    millisecond.

    `level` filters to one of info / warn / error / io. `limit` caps the reply
    and `skipped` says how many were dropped off the front, so a client that
    fell behind knows it did rather than silently missing lines.
    """
    return log_entries(since=since, limit=max(1, min(int(limit or 500), 5000)),
                       level=level)


@router.get("/api/status")
def status(project: str = ""):
    """Status for one project, plus what is holding the local lane.

    Without `project` this is the old global view, so anything that has not
    been taught to ask still works.
    """
    out = _job_for(project)
    with LOCK:
        busy = ({"stage": JOB["stage"], "project": JOB.get("project")}
                if JOB["running"] else None)
        others = sorted(k for k, r in REMOTE.items()
                        if r["running"] and k != project)
    out["lane_busy"] = busy          # the local slot, whoever owns it
    out["remote_elsewhere"] = others  # renders running on other projects
    return out


@router.post("/api/free_vram")
def free_vram_now():
    """Hand the GPU back without restarting the server.

    torch keeps freed blocks reserved from the OS, so this process can show
    its peak long after the work finished -- which is what Task Manager
    reports, and what stops ComfyUI or SHARP getting the card.
    """
    rx("/api/free_vram")
    return steps.free_vram(log=_log)
