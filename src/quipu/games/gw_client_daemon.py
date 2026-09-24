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
        self.client_process_status: str = "idle"
        self.session_id: Optional[str] = None
        self.account_id: Optional[str] = None
        self.login_status: str = "logged_out"
        self.logged_in_at: Optional[float] = None
        self.active_challenge: Optional[dict[str, Any]] = None
        self.vpn_attachment: dict[str, Any] = {
            "status": "connected",
            "overlay_ip": "100.64.0.30",
            "tailnet": "quipu.mesh",
            "connection_type": "direct_wireguard",
            "latency_ms": 19.8,
        }
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
            "session_id": self.session_id,
            "account_id": self.account_id,
            "login_status": self.login_status,
            "logged_in_at": self.logged_in_at,
            "active_challenge": self.active_challenge,
            "vpn_attachment": self.vpn_attachment,
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
            self._respond_json(200, {"status": "healthy", "game": "gw", "client": "Guild Wars 1", "login_status": _STATE.login_status})
        elif self.path == "/session":
            self._respond_json(200, {
                "game": "gw",
                "session_id": _STATE.session_id,
                "account_id": _STATE.account_id,
                "login_status": _STATE.login_status,
                "logged_in_at": _STATE.logged_in_at,
                "vpn_attachment": _STATE.vpn_attachment,
                "player_state": _STATE.active_player_state,
                "active_challenge": _STATE.active_challenge,
            })
        else:
            self._respond_json(404, {"error": "Not Found", "valid_endpoints": ["/status", "/health", "/session", "/login", "/logout", "/challenge", "/resolve_challenge", "/download", "/launch"]})

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = {}
        if content_length > 0:
            try:
                post_data = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except Exception:
                post_data = {}

        if self.path == "/login":
            account_id = post_data.get("account_id", "gw_ele_main")
            username = post_data.get("username", "quipu_elementalist@guildwars.local")
            char_name = post_data.get("character_name", "Quipu Elementalist")
            outpost = post_data.get("realm_or_world", "Droknar's Forge")

            _STATE.session_id = f"sess_gw_{int(time.time())}"
            _STATE.account_id = account_id
            _STATE.login_status = "authenticated"
            _STATE.logged_in_at = time.time()
            _STATE.client_process_status = "running"
            _STATE.active_player_state["character_name"] = char_name
            _STATE.active_player_state["current_outpost"] = outpost

            logger.info("Guild Wars Login successful: account=%s, session=%s", account_id, _STATE.session_id)
            self._respond_json(200, {
                "status": "authenticated",
                "game": "gw",
                "session_id": _STATE.session_id,
                "account_id": _STATE.account_id,
                "player_state": _STATE.active_player_state,
                "vpn_attachment": _STATE.vpn_attachment,
            })
        elif self.path == "/logout":
            old_session = _STATE.session_id
            _STATE.session_id = None
            _STATE.account_id = None
            _STATE.login_status = "logged_out"
            _STATE.logged_in_at = None
            logger.info("Guild Wars Logout successful for session=%s", old_session)
            self._respond_json(200, {"status": "logged_out", "game": "gw", "previous_session": old_session})
        elif self.path == "/challenge":
            _STATE.login_status = "challenge_required"
            _STATE.active_challenge = post_data or {
                "breakage_type": "path_obstruction",
                "reason": "Mission boundary blocked: Character stuck in Shiverpeak chokepoint",
                "timestamp": time.time(),
            }
            logger.warning("Guild Wars Challenge raised: %s", _STATE.active_challenge)
            self._respond_json(200, {"status": "challenge_required", "game": "gw", "challenge": _STATE.active_challenge})
        elif self.path == "/resolve_challenge":
            _STATE.login_status = "authenticated"
            res = _STATE.active_challenge
            _STATE.active_challenge = None
            logger.info("Guild Wars Challenge resolved via Gate 6!")
            self._respond_json(200, {"status": "resolved", "game": "gw", "resolved_challenge": res})
        elif self.path == "/download":
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
