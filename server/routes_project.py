"""Projects: create, list, clone, rename, export/import, rescan, prune, upload.

The project FOLDER is the identity. Everything inside state.json refers to
files by bare name, so a rename is a folder move plus a repoint of the two
places that recorded an absolute path.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from . import config
from .jobs import (_busy_msg, _io, _log, _project_busy, _run, refuse, rx)
from .models import CloneReq, NewProject, RenameReq
from .state import (CARRY_OVER_UI, _WIN_RESERVED, clear_derived, has_work,
                    load_state, next_free_project, pdir, save_state,
                    state_path)

PROJ = config.PROJ
router = APIRouter()


@router.get("/api/projects")
def projects():
    out = []
    for d in sorted(PROJ.iterdir()):
        if d.is_dir():
            s = load_state(d)
            out.append({"name": d.name, **{k: s.get(k) for k in
                        ("stage", "approved", "cutout", "control_video",
                         "ai_video", "splat")}})
    return {"projects": out, "dir": str(PROJ)}


@router.get("/api/state")
def get_state(project: str):
    return load_state(pdir(project))


@router.get("/api/files")
def project_files(project: str):
    """Size and modification time of every file the files panel can show.

    The panel busts each preview with its OWN mtime and rebuilds only when this
    set changes. Busting everything with orbit_time served a rebuilt
    control_soft.mp4 from the browser cache, because a soften never moves
    orbit_time.
    """
    d = pdir(project)
    out = {}
    for q in d.iterdir():
        if q.is_file() and q.suffix.lower() in (".mp4", ".png", ".jpg",
                                                  ".jpeg", ".webp", ".ply"):
            st = q.stat()
            out[q.name] = {"mtime": round(st.st_mtime, 3), "bytes": st.st_size}
    so = d / "splat_out"
    if so.is_dir():
        for q in so.glob("*.ply"):
            st = q.stat()
            out[f"splat_out/{q.name}"] = {"mtime": round(st.st_mtime, 3),
                                          "bytes": st.st_size}
    return {"files": out, "dir": str(d.resolve())}


@router.post("/api/project")
def new_project(p: NewProject):
    """Create a project with a genuinely clean state.

    save_state MERGES, so reusing an existing name used to leave the previous
    run's control_video / ai_video / splat in place while resetting the stage --
    the tool then opened showing artifacts from an image that was no longer
    loaded. A collision now picks the next free name rather than half-wiping
    someone's existing work.
    """
    rx("/api/project", p.name)
    # Only step aside for a project that holds actual WORK -- an empty one from
    # a previous "New" is safe to reuse, and uniquifying past those is what
    # once produced 34 junk projects.
    d = next_free_project(p.name)
    clear_derived(d, stage="image", approved=False, source=None,
                  ui=None, ui_time=0, fps=None)
    _log(f"project: created '{d.name}' at {d}")
    return {"project": d.name, "dir": str(d)}


@router.post("/api/reset_stage")
def reset_stage(p: NewProject):
    """Back out of review so a retry can be generated."""
    rx("/api/reset_stage", p.name)
    d = pdir(p.name)
    save_state(d, stage="generate", approved=False)
    return {"ok": True}


@router.post("/api/project/clone")
def clone_project(r: CloneReq):
    """Copy the whole project folder, byte for byte, under a new name.

    "Save as" for experiments: keep the finished pipeline -- source, orbit,
    control video, AI clips, mattes, dataset, splats, report, state -- and
    tweak the copy without risking the original. Nothing is filtered or
    regenerated; the clone opens exactly where the original stands.
    """
    rx("/api/project/clone", r.project, name=r.name)
    src = pdir(r.project)
    if not (state_path(src).exists() or any(src.glob("source.*"))):
        refuse("clone", r.project, "nothing to clone")
        return JSONResponse({"error": f"{src.name} has nothing to clone"},
                            status_code=400)
    stem = "".join(c for c in (r.name or "").strip()
                   if c.isalnum() or c in "-_")[:64] or f"{src.name}-copy"
    # Uniquify against non-empty directories, not just "projects with work":
    # cloning into a half-empty folder would silently merge two projects.
    dst = PROJ / stem
    n = 2
    while dst.exists() and any(dst.iterdir()):
        dst = PROJ / f"{stem}-{n}"
        n += 1

    def work():
        files = [q for q in src.rglob("*") if q.is_file()]
        mb = sum(q.stat().st_size for q in files) / 1e6
        _log(f"clone: {src.name} -> {dst.name} "
             f"({len(files)} files, {mb:,.0f} MB)")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        _io("cloned", src, dst, note=f"{len(files)} files, {mb:,.0f} MB, "
                                     "byte-for-byte")
        _log(f"clone: done - '{dst.name}' is an independent copy; the two "
             "projects share nothing from here on")

    if not _run("clone", work, src.name):
        refuse("clone", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True, "project": dst.name}


@router.get("/api/project/export")
def export_project(project: str):
    """The whole project as one .zip -- sources, control and AI videos,
    dataset, splats and state. Portable to another machine."""
    d = pdir(project)
    if not state_path(d).exists():
        return JSONResponse({"error": "no such project"}, status_code=404)
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(d.rglob("*")):
            if f.is_file():
                z.write(f, str(Path(project) / f.relative_to(d)))
    buf.seek(0)
    _io("exported", d, note=f"{len(buf.getvalue()) / 1e6:.2f} MB zip -> "
                            f"{d.name}.zip (download)")
    return Response(
        content=buf.getvalue(), media_type="application/zip",
        # d.name is the sanitised form; the raw query string could close the
        # quoted header value or inject another directive.
        headers={"Content-Disposition":
                 f'attachment; filename="{d.name}.zip"',
                 "Cache-Control": "no-store"})


@router.post("/api/project/import")
async def import_project(request: Request):
    """Restore a project from a .zip produced by /api/project/export."""
    rx("/api/project/import")
    import io
    import zipfile
    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    try:
        z = zipfile.ZipFile(io.BytesIO(body))
    except Exception as e:
        return JSONResponse({"error": f"not a zip: {e}"}, status_code=400)
    names = [n for n in z.namelist() if not n.endswith("/")]
    if not names:
        return JSONResponse({"error": "empty archive"}, status_code=400)
    root = names[0].split("/")[0]
    # NEVER import onto a project that already holds work. pdir() would hand
    # back the existing directory and the loop would overwrite it file by
    # file, destroying whatever was there -- an unrecoverable loss of paid
    # renders and trained splats. Land on a free sibling name instead.
    d = next_free_project(root)
    written = 0
    root_dir = d.resolve()
    for n in names:
        rel = n.split("/", 1)[1] if "/" in n else n
        if not rel:
            continue
        # `d / rel` DISCARDS d when rel is absolute or drive-anchored, so a
        # ".." check alone lets an entry named "C:/Windows/..." write anywhere
        # the server user can reach. Reject those, then confirm the resolved
        # destination is genuinely inside the project.
        rp = Path(rel)
        if ".." in rp.parts or rp.is_absolute() or rp.drive or rp.root:
            _log(f"import: refused unsafe entry {n!r}", level="warn")
            continue
        dst = (d / rp).resolve()
        if not str(dst).startswith(str(root_dir)):
            _log(f"import: refused escaping entry {n!r}", level="warn")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(n))
        written += 1
    _io("imported", f"upload ({len(body) / 1e6:.2f} MB zip)", d,
        note=f"{written} files")
    _log(f"imported project '{d.name}' - {written} files")
    return {"ok": True, "project": d.name, "files": written}


@router.post("/api/project/rescan")
def rescan_project(p: NewProject):
    """Rebuild state.json from whatever is on disk.

    The files are the durable record; state.json is only an index into them.
    When the two disagree -- a wiped state, a hand-copied folder, a project
    moved between machines -- this makes the index agree with reality again
    instead of leaving a complete project looking empty.
    """
    rx("/api/project/rescan", p.name)
    d = pdir(p.name)
    st = load_state(d)
    found = {}

    srcs = [q for q in d.glob("source.*") if q.name != "source_upright.png"]
    if srcs:
        found["source"] = srcs[0].name
        import cv2 as _cv2
        _im = _cv2.imread(str(srcs[0]), _cv2.IMREAD_UNCHANGED)
        if _im is not None:
            found["source_w"] = int(_im.shape[1])
            found["source_h"] = int(_im.shape[0])
    if (d / "cutout.png").exists():
        found["cutout"] = "cutout.png"
    if (d / "reference.png").exists():
        found["reference"] = "reference.png"
    if (d / "ai_nobg.mp4").exists():
        found["ai_nobg"] = "ai_nobg.mp4"
    if (d / "control.mp4").exists():
        found["control_video"] = "control.mp4"
    # The passes are equally on-disk facts, and a rescan that ignored them left
    # a fully worked project looking like it had never had a pass rendered.
    for w in ("top", "bottom"):
        if (d / f"pass_{w}.mp4").exists():
            found[f"pass_{w}"] = f"pass_{w}.mp4"
        _pa = sorted(d.glob(f"pass_{w}_ai*.mp4"),
                     key=lambda q: int("".join(c for c in q.stem
                                               if c.isdigit()) or 0))
        _pa = [q for q in _pa if not q.stem.endswith("_orig")]
        if _pa:
            found[f"pass_ai_{w}"] = _pa[-1].name

    ai = sorted(d.glob("ai_v*.mp4"),
                key=lambda q: int("".join(c for c in q.stem if c.isdigit()) or 0))
    ai = [q for q in ai if not q.stem.endswith("_orig")]
    if ai:
        found["ai_video"] = ai[-1].name

    out = d / "splat_out"
    if out.exists():
        plys = [q for q in out.glob("*.ply") if not q.name.endswith("_clean.ply")]
        plys.sort(key=lambda q: int("".join(c for c in q.stem if c.isdigit()) or 0))
        if plys:
            found["splat"] = plys[-1].name
            cl = plys[-1].with_name(plys[-1].stem + "_clean.ply")
            if cl.exists():
                found["splat_clean"] = cl.name

    # furthest stage the artifacts justify
    stage = "image"
    if found.get("source"):
        stage = "orbit"
    if found.get("cutout"):
        stage = "orbit"
    if found.get("control_video"):
        stage = "generate"
    if found.get("ai_video"):
        stage = "review"
    if found.get("splat"):
        stage = "done"
    # Without a rig in state, orbit_from_dict falls back to the dataclass
    # defaults and COLMAP is written for a camera that has nothing to do with
    # the video. Refuse to claim a stage that would let that happen.
    if not st.get("orbit") and stage in ("generate", "review", "done"):
        stage = "orbit"
        _log("rescan: artifacts found but no camera rig in state - "
             "re-render the orbit before generating or splatting", level="warn")
    found["stage"] = stage
    if found.get("ai_video") and st.get("approved") is None:
        found["approved"] = True

    save_state(d, **found)
    _log(f"rescan '{d.name}': stage={stage}, "
         + ", ".join(f"{k}={v}" for k, v in found.items() if k != "stage"))
    return {"ok": True, "project": d.name, **found}


@router.post("/api/projects/prune")
def prune_projects():
    """Delete projects that hold nothing at all.

    Empty means: no source image, and no cutout / control video / AI video /
    splat recorded. Anything with a single one of those is left alone. Returns
    what it removed so the deletion is auditable rather than silent.
    """
    rx("/api/projects/prune")
    removed = []
    for d in sorted(PROJ.iterdir()):
        if not d.is_dir():
            continue
        st = load_state(d)
        if any(d.glob("source.*")):
            continue
        if any(st.get(k) for k in ("control_video", "ai_video", "splat",
                                   "cutout")):
            continue
        extra = [f.name for f in d.iterdir() if f.name != "state.json"]
        if extra:                      # unexpected contents: never touch it
            continue
        _io("deleting", d, note="empty project, nothing derived recorded")
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d.name)
    if removed:
        _log(f"pruned {len(removed)} empty projects: {', '.join(removed[:8])}"
             + (" ..." if len(removed) > 8 else ""))
    else:
        _log("prune: nothing to remove - every project holds work")
    return {"removed": removed, "count": len(removed)}


@router.post("/api/project/rename")
def rename_project(r: RenameReq):
    """Rename the folder, and fix the few places that recorded its old path.

    The folder name IS the project identity -- everything inside state.json
    refers to files by bare name, so nothing breaks structurally. Two things do
    record the old absolute path though, and left alone they quietly describe a
    project that no longer exists:

      fal_sent[].path   the record of what was uploaded, and the whole point of
                        that log is being able to trust it afterwards
      splat_report.md   its title line

    Both are rewritten. Refuses while a job is running, because the worker
    holds the old Path and would write into a directory that has moved out from
    under it.
    """
    rx("/api/project/rename", r.project, to=r.name)
    old = PROJ / "".join(c for c in r.project if c.isalnum() or c in "-_")
    if not old.is_dir():
        return JSONResponse({"error": f"no project called {r.project!r}"},
                            status_code=400)
    safe = "".join(c for c in r.name if c.isalnum() or c in "-_")[:64]
    if not safe:
        return JSONResponse(
            {"error": "a name needs letters, numbers, - or _"},
            status_code=400)
    if safe.lower() in _WIN_RESERVED:
        safe += "_"
    if safe == old.name:
        return {"project": safe, "renamed": False, "note": "already named that"}
    new_d = PROJ / safe
    if new_d.exists():
        return JSONResponse(
            {"error": f"{safe!r} already exists - pick another name"},
            status_code=400)
    busy = _project_busy(old.name)
    if busy:
        refuse("rename", r.project, f"running {busy}")
        return JSONResponse(
            {"error": f"{old.name} is running {busy} - wait for it to finish"},
            status_code=409)

    old_abs = str(old.resolve())
    shutil.move(str(old), str(new_d))
    new_abs = str(new_d.resolve())

    # repoint the recorded paths so the history stays honest
    fixed = []
    st = state_path(new_d)
    if st.exists():
        try:
            # Walk the PARSED json, not the raw text. A Windows path is stored
            # with escaped separators ("C:!!Users!!..." in the file), so a
            # substring search for the real path never matches and the repoint
            # silently did nothing -- measured: fal_sent kept the old folder.
            data = json.loads(st.read_text(encoding="utf-8"))

            def repoint(o):
                if isinstance(o, dict):
                    return {k: repoint(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [repoint(v) for v in o]
                if isinstance(o, str):
                    return (o.replace(old_abs, new_abs)
                             .replace(f"projects/{old.name}",
                                      f"projects/{safe}")
                             .replace(f"projects{os.sep}{old.name}",
                                      f"projects{os.sep}{safe}"))
                return o

            fixed_data = repoint(data)
            if fixed_data != data:
                st.write_text(json.dumps(fixed_data, indent=1),
                              encoding="utf-8")
                fixed.append("state.json")
        except Exception as e:
            _log(f"rename: could not repoint state.json ({e})", level="warn")
    rep = new_d / "splat_report.md"
    if rep.exists():
        try:
            t = rep.read_text(encoding="utf-8")
            if old.name in t:
                rep.write_text(t.replace(old.name, safe), encoding="utf-8")
                fixed.append("splat_report.md")
        except Exception as e:
            _log(f"rename: could not repoint splat_report.md ({e})",
                 level="warn")

    _io("renamed", old, new_d,
        note=("repointed " + ", ".join(fixed)) if fixed
             else "no stored paths needed changing")
    _log(f"rename: {old.name} -> {safe}   {new_abs}")
    return {"project": safe, "renamed": True, "fixed": fixed}


@router.post("/api/upload")
async def upload(request: Request):
    """Raw body upload of the source image; ?project=NAME&ext=png"""
    q = request.query_params
    d = pdir(q.get("project", "project"))
    ext = (q.get("ext") or "png").lower().lstrip(".")
    if ext not in {"png", "jpg", "jpeg", "webp"}:
        ext = "png"
    body = await request.body()
    rx("/api/upload", d.name, ext=ext, bytes=len(body))
    if not body:
        return JSONResponse({"error": "empty body"}, status_code=400)
    src = d / f"source.{ext}"
    # Re-uploading the identical image is a no-op, not a reset. The wipe below
    # exists because a NEW photo invalidates the cutout, orbit and splat -- but
    # the same photo invalidates nothing, and treating it as new is how an
    # entire project's worth of references gets thrown away by accident.
    same = False
    prev = sorted(p for p in d.glob("source.*")
                  if p.name != "source_upright.png")
    if prev and prev[0].exists():
        same = (hashlib.sha256(prev[0].read_bytes()).hexdigest()
                == hashlib.sha256(body).hexdigest())
    if same:
        _log(f"upload: identical to the existing {prev[0].name} - keeping all "
             f"derived work (cutout, orbit, videos, splat)")
        return {"ok": True, "project": d.name, "source": prev[0].name,
                "bytes": len(body), "unchanged": True}

    # A DIFFERENT image is a different subject. Uploading replaces the image in
    # the CURRENT project by default -- use the New button to start a fresh
    # one. (?branch=1 opts into auto-creating a project instead; the machinery
    # is kept but is no longer the default, because deciding when a project
    # begins should be an explicit action.)
    branched_from = None
    if has_work(d) and q.get("branch", "0") in ("1", "true", "yes"):
        prev_ui = {k: v for k, v in (load_state(d).get("ui") or {}).items()
                   if k in CARRY_OVER_UI}
        branched_from = d.name
        d = next_free_project(d.name)
        src = d / f"source.{ext}"
        if prev_ui:
            save_state(d, ui=prev_ui, ui_time=time.time())
        _log(f"upload: new image -> new project '{d.name}' "
             f"('{branched_from}' left untouched"
             + (f", {len(prev_ui)} preferences carried over)" if prev_ui
                else ")"))
    src.write_bytes(body)
    for other in d.glob("source.*"):
        if other != src:
            other.unlink()
    # 'source.*' does NOT match this name (underscore, not dot), so it used to
    # survive uploads -- and every picker prefers it, so a new image could run
    # the whole pipeline with the PREVIOUS image's rotated file.
    (d / "source_upright.png").unlink(missing_ok=True)
    # The saved viewport cloud belongs to the previous photo.
    for stale in ("cloud_view.bin", "cloud_view.json"):
        if (d / stale).exists():
            _io("deleting", d / stale, note="point cloud of the previous photo")
            (d / stale).unlink()
    # A new image invalidates everything derived from the old one. Leaving the
    # previous calib in state meant a fresh upload displayed -- and drove the
    # rig with -- the PREVIOUS photo's camera estimate until cutout re-ran.
    # Record the pixel size: the client mirrors card_height_full() for the
    # viewport, and that rule depends on the source aspect ratio.
    import cv2 as _cv2
    _im = _cv2.imread(str(src), _cv2.IMREAD_UNCHANGED)
    _wh = ({"source_w": int(_im.shape[1]), "source_h": int(_im.shape[0])}
           if _im is not None else {})
    _io("received", f"upload ({len(body) / 1e6:.2f} MB)", src,
        note="source photo")
    # A new photo can land on the SAME filename (source.jpg over source.jpg).
    # Nothing in the state then changes, so the viewport's "is this already
    # the image I am showing?" check said yes and kept the old one on screen.
    # This stamp is what makes a replacement visible.
    clear_derived(d, stage="orbit", source=src.name, approved=False,
                  source_time=time.time(),
                  **_wh)
    return {"ok": True, "project": d.name, "source": src.name,
            "bytes": len(body), "branched_from": branched_from}
