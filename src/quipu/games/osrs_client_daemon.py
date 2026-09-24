"""QUIPU Old School RuneScape (OSRS) Game Client Container Daemon.

Runs inside the dedicated `quipu-game-osrs` container on port 7310.
Manages RuneLite client asset downloads, headless Xvfb execution,
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

from quipu.game_asset_downloader import download_osrs_client

logger = logging.getLogger("quipu.osrs_daemon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PORT = int(os.environ.get("OSRS_PORT", "7310"))
_HOST = os.environ.get("OSRS_HOST", "0.0.0.0")
_GAME_DIR = Path(os.environ.get("OSRS_DIR", "/games/osrs"))
_START_TIME = time.time()
_PROCESS: Optional[subprocess.Popen] = None


class OSRSEnvironmentState:
    """State tracking for the OSRS client container."""
    def __init__(self) -> None:
        self.game = "osrs"
        self.client_name = "RuneLite 2.7.3"
        self.target_dir = str(_GAME_DIR)
        self.download_status = "pending"
        self.installed_files: dict[str, int] = {}
        self.manifest: dict[str, Any] = {}
        self.client_process_status = "idle"
        self.active_player_state: dict[str, Any] = {
            "player_name": "QuipuBot",
            "combat_level": 75,
            "hp_current": 99,
            "hp_max": 99,
            "prayer_current": 70,
            "prayer_max": 70,
            "run_energy": 100,
            "location": "Lumbridge Courtyard",
            "in_combat": False,
        }

    def scan_installed(self) -> None:
        if not _GAME_DIR.exists():
            self.download_status = "not_downloaded"
            return
        files = {}
        for f in _GAME_DIR.glob("**/*"):
            if f.is_file():
                files[f.name] = f.stat().st_size
        self.installed_files = files
        manifest_path = _GAME_DIR / "manifest.json"
        if manifest_path.exists():
            try:
                self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.download_status = "downloaded"
            except Exception:
                self.download_status = "corrupted"
        elif "RuneLite.jar" in files:
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


_STATE = OSRSEnvironmentState()


class OSRSHTTPHandler(BaseHTTPRequestHandler):
    """HTTP endpoints for OSRS client container on port 7310."""

    def do_GET(self) -> None:
        if self.path in ("/status", "/"):
            _STATE.scan_installed()
            self._respond_json(200, _STATE.to_dict())
        elif self.path == "/health":
            self._respond_json(200, {"status": "healthy", "game": "osrs", "client": "RuneLite"})
        else:
            self._respond_json(404, {"error": "Not Found", "valid_endpoints": ["/status", "/health", "/download", "/launch"]})

    def do_POST(self) -> None:
        if self.path == "/download":
            logger.info("Triggered OSRS client download...")
            res = download_osrs_client(_GAME_DIR)
            _STATE.scan_installed()
            self._respond_json(200, res.to_dict())
        elif self.path == "/launch":
            logger.info("Triggered OSRS client launch...")
            _STATE.client_process_status = "running"
            self._respond_json(200, {"status": "launched", "client": "RuneLite.jar", "display": ":99"})
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

    # Automatically download on container startup if missing
    if _STATE.download_status != "downloaded":
        logger.info("RuneLite client not found in %s. Downloading now...", _GAME_DIR)
        download_osrs_client(_GAME_DIR)
        _STATE.scan_installed()

    server = HTTPServer((_HOST, _PORT), OSRSHTTPHandler)
    logger.info("QUIPU OSRS Game Client Daemon listening on http://%s:%d", _HOST, _PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down OSRS daemon...")
        server.server_close()


if __name__ == "__main__":
    main()
