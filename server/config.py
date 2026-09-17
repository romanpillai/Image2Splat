"""Where everything outside this directory lives.

The rule this module exists to enforce: **no path to an external tool is
written in code.** The predecessor hard-coded `ROOT.parent / "brush" /
"brush_app.exe"` and `ROOT.parent / "splatformer_repo"`, so moving either one
meant editing Python. Here every external location is a key in `config.json`
whose default is the current location, and the code only ever asks this module.

Brush is a separate download and is never part of the repository; by default
it is looked for at ./brush/brush_app.exe. config.json is local (gitignored) --
copy config.example.json to make one.

Relative paths in config.json resolve against the file's own directory, so the
whole folder can be moved without editing anything.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# server/config.py -> server/ -> Img2Splat_beta/
ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.json"

# The defaults are what a FRESH CLONE needs: Brush unpacked into ./brush, the
# projects folder beside the code, and the interpreter that is running this
# server. A missing config.json is therefore not an error, and a config.json
# that only overrides `brush` keeps correct defaults for the rest. Copy
# config.example.json to config.json to point anything elsewhere.
DEFAULTS = {
    "brush": "./brush/brush_app.exe",
    "python": "",                  # "" = the interpreter running this server
    "projects": "./projects",
    "host": "127.0.0.1",
    "port": 8771,
    "log_ring": 5000,
}

_RAW: dict = {}
CONFIG_ERROR: str | None = None
try:
    if CONFIG_FILE.exists():
        # utf-8-sig: anything written by a Windows editor or PowerShell carries
        # a BOM, which plain json.loads rejects.
        _RAW = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
        if not isinstance(_RAW, dict):
            raise ValueError("config.json must hold a JSON object")
except Exception as e:                      # never fatal: fall back to defaults
    CONFIG_ERROR = f"{type(e).__name__}: {e}"
    _RAW = {}


def _value(key: str):
    v = _RAW.get(key)
    return DEFAULTS[key] if v in (None, "") else v


def path_of(key: str) -> Path:
    """One configured path, resolved. Relative -> relative to this directory."""
    p = Path(str(_value(key))).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


BRUSH = path_of("brush")
PYTHON = path_of("python") if _value("python") else Path(os.sys.executable)
PROJ = path_of("projects")
PROJ.mkdir(parents=True, exist_ok=True)

HOST = str(_value("host"))
PORT = int(_value("port"))
# The /api/log ring buffer. The predecessor kept 400 lines, which a single
# matte pass overflows; the console is a headline feature here, so it holds a
# whole session's worth.
LOG_RING = max(200, int(_value("log_ring")))

WEB = ROOT / "web"


def summary() -> dict:
    """What the server is actually pointing at, for /api/health and boot.

    `exists` is reported per path rather than assumed, because the single most
    common failure of a config file is that it names something that is not
    there -- and a Brush that is missing should say so on boot, not thirty
    minutes into a job.
    """
    return {
        "config_file": str(CONFIG_FILE),
        "config_loaded": CONFIG_FILE.exists() and CONFIG_ERROR is None,
        "config_error": CONFIG_ERROR,
        "root": str(ROOT),
        "brush": {"path": str(BRUSH), "exists": BRUSH.exists(),
                  "default": BRUSH == (ROOT / DEFAULTS["brush"]).resolve()},
        "python": {"path": str(PYTHON), "exists": PYTHON.exists(),
                   "running": os.sys.executable,
                   "default": not _value("python")},
        "projects": {"path": str(PROJ), "exists": PROJ.exists(),
                     "default": PROJ == (ROOT / DEFAULTS["projects"]).resolve()},
        "web": str(WEB),
        "host": HOST, "port": PORT, "log_ring": LOG_RING,
    }
