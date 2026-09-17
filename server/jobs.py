"""The job runner, and the log everything writes to.

Two lanes, one lock, one ring buffer.

LOGGING IS A FEATURE HERE, not a debugging aid. This tool is in beta and the
operator is expected to watch it work, so three rules hold everywhere:

  1. Every state-changing request says it arrived, with its key parameters.
     `rx()` does that.
  2. Every file written, read or sent names its FULL PATH. `_io()` does that,
     and it is a standing rule for this project -- if a step opens a file and
     does not say which, that is a bug.
  3. Every failure carries a STAMP (`error_at`) so the client can attribute it
     to the job that produced it. There is a whole class of "stale error" bug
     this exists to prevent: without the stamp, a failed job's error was
     re-alerted on every later poll, forever.

Everything logged also lands in a structured ring buffer with a timestamp, a
LEVEL (info / warn / error / io) and a SOURCE, which /api/log serves. The
levels are not decoration -- the frontend filters on them.
"""
from __future__ import annotations

import itertools
import threading
import time
import traceback
from pathlib import Path

import steps

from . import config

PROJ = config.PROJ

# --------------------------------------------------------------- ring buffer --
# The predecessor kept 400 lines PER JOB and nothing global. 400 is less than a
# single matte pass emits, so the beginning of a run was gone before it
# finished. This is a whole session's worth, held once, and configurable.
LOG_LOCK = threading.Lock()
LOG_RING: list[dict] = []
_LOG_SEQ = itertools.count(1)

LEVELS = ("info", "warn", "error", "io")


def _level_of(msg: str) -> str:
    """Guess a level from a message that did not declare one.

    Every call site COULD pass a level, and the important ones do. But the
    ported code carries hundreds of `_log("... WARNING ...")` lines written
    before levels existed, and re-typing them all would be exactly the
    rewrite-from-scratch this port is avoiding. So the old convention is
    honoured: those messages shout in capitals, and that is what is read.
    """
    m = msg.lstrip()
    if m.startswith("file: "):
        return "io"
    if m.startswith("ERROR") or "Traceback" in m:
        return "error"
    if "WARNING" in m or m.startswith("WARN"):
        return "warn"
    return "info"


def log_push(msg: str, level: str = "", source: str = "", project: str = ""):
    """One structured entry. Called by _log; call it directly only from code
    that has no job context at all."""
    lvl = level if level in LEVELS else _level_of(msg)
    e = {"seq": next(_LOG_SEQ), "t": time.time(), "level": lvl,
         "source": source or "server", "project": project or "",
         "msg": str(msg)[:2000]}
    with LOG_LOCK:
        LOG_RING.append(e)
        if len(LOG_RING) > config.LOG_RING:
            del LOG_RING[:len(LOG_RING) - config.LOG_RING]
    return e


def log_entries(since: int = 0, limit: int = 0, level: str = "") -> dict:
    """A slice of the ring, oldest first.

    `since` is a SEQUENCE number, not a timestamp or an index. The client polls
    with the last seq it saw and gets exactly what it has not seen -- indexes
    would shift as the ring rolls, and two entries can share a timestamp.
    """
    with LOG_LOCK:
        rows = list(LOG_RING)
    if since:
        rows = [r for r in rows if r["seq"] > int(since)]
    if level in LEVELS:
        rows = [r for r in rows if r["level"] == level]
    dropped = 0
    if limit and len(rows) > limit:
        dropped = len(rows) - limit
        rows = rows[-limit:]
    with LOG_LOCK:
        oldest = LOG_RING[0]["seq"] if LOG_RING else 0
        newest = LOG_RING[-1]["seq"] if LOG_RING else 0
        held = len(LOG_RING)
    return {"entries": rows, "count": len(rows), "skipped": dropped,
            "oldest_seq": oldest, "newest_seq": newest,
            "held": held, "capacity": config.LOG_RING,
            "levels": list(LEVELS)}


# --------------------------------------------------------------------- lanes --
LOCK = threading.Lock()


def _blank_job():
    # phase / step / steps are what the viewport's "working" overlay reads:
    # a human phase name and a count ("frame 42 of 120"). aborted separates a
    # deliberate stop from a failure, so a stopped run is not shown in red.
    return {"running": False, "stage": "", "msg": "", "pct": 0.0,
            "error": None, "error_at": 0.0, "log": [], "project": None,
            "abortable": False, "phase": "", "step": 0, "steps": 0,
            "aborted": False}


# The LOCAL lane: one job at a time, because every step in it wants the GPU.
# 12.8 GB does not hold two depth models, or Brush plus anything.
JOB = _blank_job()

# The REMOTE lane: fal renders and captions. These do not touch the GPU at
# all -- they upload, then block on a network subscribe for minutes while this
# machine sits idle. Holding the local lane for that stalled every other
# project for no reason, so they run outside it, one per project.
REMOTE: dict = {}

# Which job record the current thread writes to. Job functions call _log and
# _set without knowing which lane they are in, so the lane is carried on the
# thread rather than threaded through every call site.
_CUR = threading.local()


def _cur():
    return getattr(_CUR, "job", None) or JOB


def _job_for(project: str | None):
    """The job record a client asking about `project` should be shown.

    Its own job wins -- a remote render on THIS project is what its operator
    cares about -- and otherwise it sees the local lane, so a Brush run
    elsewhere still explains why the buttons are refusing.
    """
    with LOCK:
        r = REMOTE.get(project or "")
        # A RUNNING job of its own always wins.
        if r and r["running"]:
            return dict(r)
        if JOB["running"] and JOB.get("project") == project:
            return dict(JOB)
        # Nothing running here. Its own FINISHED remote job still beats the
        # global record -- otherwise a describe or a render vanished from the
        # log the moment it succeeded, and its result could not be read.
        if r and r.get("log"):
            return dict(r)
        return dict(JOB)


def _busy_msg() -> str:
    """Why a local job was refused, naming the project that holds the slot.

    "busy" on its own is useless once several projects are in flight: the
    answer to "busy with what?" is the whole point.
    """
    with LOCK:
        if JOB["running"]:
            who = JOB.get("project") or "another project"
            return f"busy - {who} is running {JOB['stage']}"
    return "busy"


def _project_busy(project: str) -> str:
    """Empty if this project is free, else the stage already running on it.

    One job per PROJECT regardless of lane: a render and a splat on the same
    project would both write ai_frames/, dataset/ and state.json.
    """
    with LOCK:
        r = REMOTE.get(project)
        if r and r["running"]:
            return r["stage"]
        if JOB["running"] and JOB.get("project") == project:
            return JOB["stage"]
    return ""


# The long-running child (Brush) so it can be signalled from another request.
# Guarded by LOCK: the worker thread sets it, an HTTP thread reads it.
CHILD = {"proc": None, "aborted": False}


def _set(**kw):
    with LOCK:
        _cur().update(kw)


def progress(step: int = 0, steps: int = 0, phase: str | None = None):
    """Report where the current job is, for the viewport overlay.

    `phase` is a short human name ("rendering", "softening"); step/steps is a
    count when there is one. pct follows the count so the job bar agrees.
    """
    kw = {"step": int(step), "steps": int(steps)}
    if steps:
        kw["pct"] = round(100.0 * max(0, min(step, steps)) / steps, 1)
    if phase is not None:
        kw["phase"] = phase
    _set(**kw)


def _log(msg, level: str = "", source: str = ""):
    """Append to the current job's log AND to the global ring buffer.

    The per-job list is what the status poll returns and is kept as it was.
    The ring is what /api/log serves, and it survives the job ending -- which
    the per-job list does not, so a failure's explanation used to disappear the
    moment the next job started.
    """
    with LOCK:
        j = _cur()
        j["log"].append(str(msg)[:500])
        # Kept generous: the old 200 truncated a matte run's own output.
        j["log"] = j["log"][-2000:]
        j["msg"] = str(msg)[:200]
        stage = j.get("stage") or ""
        proj = j.get("project") or ""
    log_push(msg, level=level, source=source or stage, project=proj)


def _where(p) -> str:
    """A path as the operator would name it: projects/<project>/<rest>."""
    q = Path(p)
    try:
        return "projects/" + q.resolve().relative_to(PROJ.resolve()).as_posix()
    except Exception:
        return str(q)


def _full(p) -> str:
    """The absolute path, always. `_where` is the short readable form; this is
    the one that can be pasted into Explorer, and the standing rule for this
    project is that both are said."""
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)


def _count(p) -> str:
    q = Path(p)
    if not q.is_dir():
        return f"{q.stat().st_size / 1e6:.2f} MB" if q.exists() else "missing"
    n = len(list(q.glob("frame_*.png"))) or len(list(q.iterdir()))
    return f"{n} files"


def _io(verb: str, src, dst=None, note: str = "", source: str = "") -> None:
    """Log a file movement with both ends named, in FULL.

    Every step here shuffles files between the project directory, fal's CDN,
    the matte directories and the dataset, and the first question when
    something looks wrong is always which file a step actually used. So say
    it, on both sides, every time.

    Beta change: the absolute path is appended as well as the short
    projects/<name>/... form. The short form is what you read; the absolute one
    is what you paste into Explorer when the short one is not enough.
    """
    is_url = isinstance(src, (str, Path)) and "://" in str(src)
    s = str(src) if is_url else _where(src)
    if is_url:
        s = str(src)[:88] + ("..." if len(str(src)) > 88 else "")
    line = f"file: {verb} {s}"
    if dst is not None:
        t = str(dst) if "://" in str(dst) else _where(dst)
        line += f" -> {t}"
        try:
            line += f" [{_count(dst)}]"
        except Exception:
            pass
    if note:
        line += f" ({note})"
    # The absolute path of whichever end is a real local file. Never a URL.
    abs_end = None
    if dst is not None and "://" not in str(dst):
        abs_end = _full(dst)
    elif not is_url:
        abs_end = _full(src)
    if abs_end:
        line += f"  @ {abs_end}"
    _log(line, level="io", source=source)


def rx(endpoint: str, project: str = "", **params) -> None:
    """"This request arrived, and here is what it asked for."

    Called at the top of every state-changing route. The rule is deliberately
    blunt: if a POST can change a file or a state.json and it does not appear
    in the log, the operator cannot tell whether their click did anything --
    which is the single most common question asked of this tool.
    """
    bits = []
    for k, v in params.items():
        if v is None or v == "":
            continue
        if isinstance(v, float):
            v = f"{v:g}"
        elif isinstance(v, (list, tuple)):
            v = f"[{len(v)} items]"
        elif isinstance(v, dict):
            v = f"{{{len(v)} keys}}"
        else:
            v = str(v)
        bits.append(f"{k}={v[:80]}")
    line = f"POST {endpoint}"
    if project:
        line += f"  project={project}"
    if bits:
        line += "  " + " ".join(bits)
    log_push(line, level="info", source="request", project=project)


def _run(stage, fn, proj="", lane="local"):
    """Run fn on a worker thread, recording stage/errors uniformly.

    lane="remote" is for work that only waits on the network. It skips the
    single local slot so a fal render on one project no longer stalls every
    other project for the minutes it spends blocked on a subscribe.
    """
    record = {}

    def work():
        _CUR.job = record            # every _log/_set on this thread lands here
        try:
            _set(running=True, stage=stage, error=None, error_at=0.0,
                 pct=0.0, project=proj)
            _log(f"{stage}: started ({lane} lane)")
            fn()
            _set(running=False, pct=100.0)
            # Give the card back at the END of every job rather than holding
            # the peak until the next one. Cheap, and it is the difference
            # between ComfyUI having 11 GB or 4 GB to work with.
            try:
                if lane == "local":
                    steps.free_vram()
            except Exception:
                pass
            _log(f"{stage}: done")
        except Exception as e:
            # error_at makes the failure IDENTIFIABLE. The error stays in JOB
            # until the next job starts, and the client alerts whenever it
            # polls a finished job that has one -- so every later action that
            # started a poll but no job re-raised the same dead error, forever.
            # With a stamp the client can alert each failure exactly once.
            _set(running=False, error=f"{type(e).__name__}: {e}",
                 error_at=time.time())
            _log(f"{stage}: FAILED {type(e).__name__}: {e}", level="error")
            _log("ERROR " + traceback.format_exc()[-1200:], level="error")
        finally:
            _CUR.job = None

    with LOCK:
        # One job per project in EITHER lane: a render and a splat on the same
        # project would both write ai_frames/, dataset/ and state.json.
        r = REMOTE.get(proj)
        if (r and r["running"]) or (JOB["running"] and JOB.get("project") == proj):
            return False
        if lane == "remote":
            record = REMOTE.setdefault(proj, _blank_job())
        else:
            if JOB["running"]:
                return False
            record = JOB
        # Claim it HERE, under the same lock as the check. Setting it inside
        # the worker left a window between check and thread start in which a
        # second request also saw running=False -- two jobs writing the same
        # frames, dataset and state.
        record.update(running=True, stage=stage, error=None, error_at=0.0,
                      pct=0.0, phase="", step=0, steps=0, aborted=False,
                      msg="", log=[], project=proj)
    threading.Thread(target=work, daemon=True).start()
    return True


def refuse(stage: str, project: str, why: str) -> None:
    """Log a refusal. A 409 that leaves no trace looks like nothing happened."""
    log_push(f"{stage}: REFUSED - {why}", level="warn", source=stage,
             project=project)
