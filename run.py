r"""Start the server.

    python run.py

then open http://127.0.0.1:8771 . Host and port come from config.json
(copy config.example.json), defaulting to 127.0.0.1:8771.

There is NO auto-reload. Restart after any .py change.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import config          # noqa: E402
from server.app import app, boot_banner   # noqa: E402

if __name__ == "__main__":
    import uvicorn
    for line in boot_banner():
        print(line)
    print("  (no auto-reload: restart after any .py change)")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")
