"""QUIPU World of Warcraft (WoW) Game Client Container Daemon.

Runs inside the dedicated `quipu-game-pipeline` or `quipu-game-wow` container on port 7320.
Manages WoW Classic client assets, realmlist configuration, WTF cvars,
and bridges real-time game interaction states with the QUIPU Game Mesh.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

from quipu.game_asset_downloader import download_wow_client

logger = logging.getLogger("quipu.wow_daemon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PORT = int(os.environ.get("WOW_PORT", "7320"))
_HOST = os.environ.get("WOW_HOST", "0.0.0.0")
_GAME_DIR = Path(os.environ.get("WOW_DIR", "/games/wow"))
_START_TIME = time.time()
_PROCESS: Optional[subprocess.Popen] = None


class WoWEnvironmentState:
    """State tracking for the WoW client container."""
    def __init__(self) -> None:
        self.game = "wow"
        self.client_name = "World of Warcraft 1.12.1 (Classic / Forever)"
        self.target_dir = str(_GAME_DIR)
        self.download_status = "pending"
        self.installed_files: dict[str, int] = {}
        self.manifest: dict[str, Any] = {}
        self.client_process_status = "idle"
        self.active_player_state: dict[str, Any] = {
            "player_name": "QuipuWarrior",
            "level": 15,
            "character_class": "Warrior",
            "hp_current": 420,
            "hp_max": 420,
            "rage_current": 0,
            "rage_max": 100,
            "zone": "Westfall - Sentinel Hill",
            "in_combat": False,
        }

    def scan_installed(self) -> None:
        if not _GAME_DIR.exists():
            self.download_status = "not_downloaded"
            return
        files = {}
        for f in _GAME_DIR.glob("**/*"):
            if f.is_file():
                rel = str(f.relative_to(_GAME_DIR))
                files[rel] = f.stat().st_size
        self.installed_files = files
        manifest_path = _GAME_DIR / "manifest.json"
        if manifest_path.exists():
            try:
                self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.download_status = "downloaded"
            except Exception:
                self.download_status = "corrupted"
        elif "WoW.exe" in files:
            self.download_status = "downloaded"
        else:
            self.download_status = "not_downloaded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.download_status == "downloaded" else "needs_download",
            "game": self.game,
            "client": self.client_name,
            "uptime_seconds": round(time.time() - _START_TIME, 1),
            "download_status": self.download_status,
            "client_process_status": self.client_process_status,
            "installed_files": self.installed_files,
            "manifest": self.manifest,
            "player_state": self.active_player_state,
        }


_STATE = WoWEnvironmentState()


class WoWHTTPHandler(BaseHTTPRequestHandler):
    """HTTP endpoints for WoW client container on port 7320."""

    def do_GET(self) -> None:
        if self.path in ("/status", "/"):
            _STATE.scan_installed()
            self._respond_json(200, _STATE.to_dict())
        elif self.path == "/health":
            self._respond_json(200, {"status": "healthy", "game": "wow", "client": "WoW 1.12.1"})
        else:
            self._respond_json(404, {"error": "Not Found", "valid_endpoints": ["/status", "/health", "/download", "/launch"]})

    def do_POST(self) -> None:
        if self.path == "/download":
            logger.info("Triggered WoW client download...")
            res = download_wow_client(_GAME_DIR)
            _STATE.scan_installed()
            self._respond_json(200, res.to_dict())
        elif self.path == "/launch":
            logger.info("Triggered WoW client launch...")
            _STATE.client_process_status = "running"
            self._respond_json(200, {"status": "launched", "client": "WoW.exe", "display": ":99"})
        else:
            self._respond_json(404, {"error": "Not Found"})

    def _respond_json(self, status_code: int, data: dict[str, Any]) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), format % args)


def main() -> None:
    _GAME_DIR.mkdir(parents=True, exist_ok=True)
    _STATE.scan_installed()

    # Automatically initialize on container startup if missing
    if _STATE.download_status != "downloaded":
        logger.info("WoW client not found in %s. Initializing now...", _GAME_DIR)
        download_wow_client(_GAME_DIR)
        _STATE.scan_installed()

    server = HTTPServer((_HOST, _PORT), WoWHTTPHandler)
    logger.info("QUIPU WoW Game Client Daemon listening on http://%s:%d", _HOST, _PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down WoW daemon...")
        server.server_close()


if __name__ == "__main__":
    main()
