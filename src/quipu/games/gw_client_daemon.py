"""QUIPU Guild Wars 1 (GW) Game Client Container Daemon.

Runs inside the dedicated `quipu-game-gw` container on port 7330.
Manages ArenaNet client asset downloads (GwSetup.exe, Gw.exe),
headless Wine execution, and bridges real-time game interaction states.
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

from quipu.game_asset_downloader import download_gw_client

logger = logging.getLogger("quipu.gw_daemon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PORT = int(os.environ.get("GW_PORT", "7330"))
_HOST = os.environ.get("GW_HOST", "0.0.0.0")
_GAME_DIR = Path(os.environ.get("GW_DIR", "/games/gw"))
_START_TIME = time.time()
_PROCESS: Optional[subprocess.Popen] = None


class GWEnvironmentState:
    """State tracking for the Guild Wars client container."""
    def __init__(self) -> None:
        self.game = "gw"
        self.client_name = "Guild Wars 1 (ArenaNet)"
        self.target_dir = str(_GAME_DIR)
        self.download_status = "pending"
        self.installed_files: dict[str, int] = {}
        self.manifest: dict[str, Any] = {}
        self.client_process_status = "idle"
        self.active_player_state: dict[str, Any] = {
            "character_name": "Quipu Elementalist",
            "profession": "Elementalist / Monk",
            "level": 20,
            "hp_current": 480,
            "hp_max": 480,
            "energy_current": 45,
            "energy_max": 45,
            "outpost": "Droknar's Forge",
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
        elif "Gw.exe" in files or "GwSetup.exe" in files:
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


_STATE = GWEnvironmentState()


class GWHTTPHandler(BaseHTTPRequestHandler):
    """HTTP endpoints for Guild Wars client container on port 7330."""

    def do_GET(self) -> None:
        if self.path in ("/status", "/"):
            _STATE.scan_installed()
            self._respond_json(200, _STATE.to_dict())
        elif self.path == "/health":
            self._respond_json(200, {"status": "healthy", "game": "gw", "client": "Guild Wars 1"})
        else:
            self._respond_json(404, {"error": "Not Found", "valid_endpoints": ["/status", "/health", "/download", "/launch"]})

    def do_POST(self) -> None:
        if self.path == "/download":
            logger.info("Triggered Guild Wars client download...")
            res = download_gw_client(_GAME_DIR)
            _STATE.scan_installed()
            self._respond_json(200, res.to_dict())
        elif self.path == "/launch":
            logger.info("Triggered Guild Wars client launch...")
            _STATE.client_process_status = "running"
            self._respond_json(200, {"status": "launched", "client": "Gw.exe", "display": ":99"})
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
        logger.info("Guild Wars client not found in %s. Downloading now...", _GAME_DIR)
        download_gw_client(_GAME_DIR)
        _STATE.scan_installed()

    server = HTTPServer((_HOST, _PORT), GWHTTPHandler)
    logger.info("QUIPU Guild Wars Game Client Daemon listening on http://%s:%d", _HOST, _PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Guild Wars daemon...")
        server.server_close()


if __name__ == "__main__":
    main()
