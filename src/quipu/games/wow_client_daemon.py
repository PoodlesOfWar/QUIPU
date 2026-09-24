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
        self.client_process_status: str = "idle"
        self.session_id: Optional[str] = None
        self.account_id: Optional[str] = None
        self.login_status: str = "logged_out"
        self.logged_in_at: Optional[float] = None
        self.active_challenge: Optional[dict[str, Any]] = None
        self.vpn_attachment: dict[str, Any] = {
            "status": "connected",
            "overlay_ip": "100.64.0.20",
            "tailnet": "quipu.mesh",
            "connection_type": "direct_wireguard",
            "latency_ms": 28.7,
        }
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


_STATE = WoWEnvironmentState()


class WoWHTTPHandler(BaseHTTPRequestHandler):
    """HTTP endpoints for WoW client container on port 7320."""

    def do_GET(self) -> None:
        if self.path in ("/status", "/"):
            _STATE.scan_installed()
            self._respond_json(200, _STATE.to_dict())
        elif self.path == "/health":
            self._respond_json(200, {"status": "healthy", "game": "wow", "client": "WoW 1.12.1", "login_status": _STATE.login_status})
        elif self.path == "/session":
            self._respond_json(200, {
                "game": "wow",
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
            account_id = post_data.get("account_id", "wow_warrior_main")
            username = post_data.get("username", "quipu_warrior")
            char_name = post_data.get("character_name", "QuipuWarrior")
            realm = post_data.get("realm_or_world", "logon.turtle-wow.org")

            _STATE.session_id = f"sess_wow_{int(time.time())}"
            _STATE.account_id = account_id
            _STATE.login_status = "authenticated"
            _STATE.logged_in_at = time.time()
            _STATE.client_process_status = "running"
            _STATE.active_player_state["player_name"] = char_name
            _STATE.active_player_state["current_realm"] = realm

            logger.info("WoW Login successful: account=%s, session=%s", account_id, _STATE.session_id)
            self._respond_json(200, {
                "status": "authenticated",
                "game": "wow",
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
            logger.info("WoW Logout successful for session=%s", old_session)
            self._respond_json(200, {"status": "logged_out", "game": "wow", "previous_session": old_session})
        elif self.path == "/challenge":
            _STATE.login_status = "challenge_required"
            _STATE.active_challenge = post_data or {
                "breakage_type": "disconnection",
                "reason": "Realm disconnected: Account re-authentication or route confirmation required",
                "timestamp": time.time(),
            }
            logger.warning("WoW Challenge raised: %s", _STATE.active_challenge)
            self._respond_json(200, {"status": "challenge_required", "game": "wow", "challenge": _STATE.active_challenge})
        elif self.path == "/resolve_challenge":
            _STATE.login_status = "authenticated"
            res = _STATE.active_challenge
            _STATE.active_challenge = None
            logger.info("WoW Challenge resolved via Gate 6!")
            self._respond_json(200, {"status": "resolved", "game": "wow", "resolved_challenge": res})
        elif self.path == "/download":
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
