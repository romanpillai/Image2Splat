"""Img2Splat beta -- FastAPI app, static mounting, startup.

Run:  ..\\AnySplat\\.venv\\Scripts\\python.exe run.py
Then: http://localhost:8771

Stages, in order, each gated on the previous:
  1 image    upload a source image
  2 orbit    author the camera path; render the control video (server-side,
             from the same camera model that later goes to COLMAP)
  3 generate fal render-to-real
  4 review   approve or retry -- nothing proceeds until approved
  5 splat    matte -> COLMAP with exact poses -> Brush
  6 passes   two more orbits rendered from the trained splat, then
             dataset_complete/ merging all three rings under one camera

Ported from the predecessor at ../Img2Splat. Removed here: the panorama /
ground-projection dome backdrop, the depth pass, SplatFormer refinement, the
keyframed retime editor and the polish sliders. Everything else -- the camera
geometry, the COLMAP conventions, the focal-length handling, the staleness
stamping -- is carried over as it was, because it encodes measurements rather
than preferences.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import falclient

from . import config
from .jobs import log_push
from . import (routes_dataset, routes_generate, routes_orbit, routes_passes,
               routes_project, routes_splat, routes_system, state)

app = FastAPI(title="Image2Splat beta")


@app.middleware("http")
async def no_cache(request: Request, call_next):
    """Never cache the UI.

    A plain reload was serving stale CSS/JS from the browser cache, so fixes
    only appeared after a hard reload -- which is not always available (remote
    desktop sessions swallow Ctrl+F5). Project media is versioned by query
    string at the call sites, so nothing here relies on caching.
    """
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.startswith("/web") or p.startswith("/api"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    elif p.startswith("/projects"):
        # Project media used to rely on every call site remembering a ?t=
        # cache-buster, and a call site that forgot showed the previous
        # project's file. This does not forbid caching -- StaticFiles sends
        # ETag and Last-Modified, so an unchanged video still answers 304 and
        # costs nothing -- it forbids serving from cache WITHOUT asking. A
        # file rewritten under the same name can no longer go unnoticed.
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


# Order is presentation only -- FastAPI matches on path, not registration --
# but it is the order of the pipeline, so the route table reads as the tool
# works.
app.include_router(routes_system.router)
app.include_router(routes_project.router)
app.include_router(state.router)
app.include_router(routes_orbit.router)
app.include_router(routes_generate.router)
app.include_router(routes_passes.router)
app.include_router(routes_dataset.router)
app.include_router(routes_splat.router)


@app.get("/")
def index():
    return FileResponse(config.WEB / "index.html")


config.WEB.mkdir(parents=True, exist_ok=True)
app.mount("/web", StaticFiles(directory=config.WEB), name="web")
app.mount("/projects", StaticFiles(directory=config.PROJ), name="projects")


def boot_banner() -> list[str]:
    """What this process is pointing at, said once, at the top of the log.

    Every one of these is a path that came from config.json rather than from
    code, so it is the first thing worth knowing and the first thing to be
    wrong.
    """
    c = config.summary()
    lines = [
        f"Image2Splat beta  ->  http://{config.HOST}:{config.PORT}",
        f"  config      : {c['config_file']}"
        + ("" if c["config_loaded"] else "  (NOT LOADED - using defaults"
           + (f": {c['config_error']}" if c["config_error"] else "") + ")"),
        f"  projects    : {c['projects']['path']}",
        f"  brush       : {c['brush']['path']}"
        + ("" if c["brush"]["exists"] else "   <-- NOT FOUND"),
        f"  python (cfg): {c['python']['path']}"
        + ("" if c["python"]["exists"] else "   <-- NOT FOUND"),
        f"  python (run): {c['python']['running']}",
        f"  web         : {c['web']}",
        f"  log ring    : {c['log_ring']} entries, served at /api/log",
        f"  FAL_KEY set : {falclient.have_key()} "
        f"(from {falclient.key_source()})",
    ]
    return lines


@app.on_event("startup")
def _startup():
    for line in boot_banner():
        log_push(line, level="info", source="boot")
    c = config.summary()
    if not c["brush"]["exists"]:
        log_push(f"boot: WARNING Brush is not at {c['brush']['path']} - the "
                 f"splat step will refuse until \"brush\" in config.json "
                 f"points at brush_app.exe", level="warn", source="boot")
    if not c["config_loaded"]:
        log_push("boot: WARNING config.json was not read; every external path "
                 "is falling back to its built-in default", level="warn",
                 source="boot")
