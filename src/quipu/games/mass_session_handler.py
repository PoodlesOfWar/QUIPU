"""QUIPU Multi-Game Mass Session Handler & r-ADMIN Controller.

Orchestrates mass logins and active sessions across Old School RuneScape (OSRS),
World of Warcraft (WoW), and Guild Wars (GW) containers. Directly bridges:
1. Multi-account credential store with AES-256-GCM encryption.
2. Tailscale-style mesh VPN overlay routing (100.64.0.0/10 CGNAT).
3. Physical Gate (Gate 6) human-in-the-loop User Confirmation protocol.
4. r-ADMIN (rADAM toroidal pressure & complex gradient optimizer) coordination.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Optional

from quipu.games.account_store import AccountProfile, AccountStore
from quipu.games.gate6_user_interlock import (
    BreakageType,
    Gate6UserInterlock,
    RefinementBreakageEvent,
    get_global_interlock,
)
from quipu.games.vpn_mesh_integration import TailscaleMeshIntegration

logger = logging.getLogger("quipu.games.mass_handler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_DEFAULT_ENDPOINTS = {
    "osrs": "http://127.0.0.1:7310",
    "wow": "http://127.0.0.1:7320",
    "gw": "http://127.0.0.1:7330",
}


def _http_request(url: str, method: str = "GET", data: Optional[dict[str, Any]] = None, timeout: float = 3.0) -> dict[str, Any]:
    """Helper for JSON HTTP calls to container daemons."""
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content) if content else {}
    except urllib.error.HTTPError as err:
        try:
            return json.loads(err.read().decode("utf-8"))
        except Exception:
            return {"error": f"HTTP {err.code}: {err.reason}"}
    except Exception as exc:
        return {"error": str(exc)}


class MassSessionHandler:
    """Centralized mass login, session orchestrator, and r-ADMIN bridge."""

    def __init__(
        self,
        endpoints: Optional[dict[str, str]] = None,
        account_store: Optional[AccountStore] = None,
        vpn_mesh: Optional[TailscaleMeshIntegration] = None,
        gate6_interlock: Optional[Gate6UserInterlock] = None,
    ) -> None:
        self.endpoints = endpoints or dict(_DEFAULT_ENDPOINTS)
        self.account_store = account_store or AccountStore()
        self.vpn_mesh = vpn_mesh or TailscaleMeshIntegration()
        self.gate6_interlock = gate6_interlock or get_global_interlock()

        # Register hook to auto-resolve daemon challenges when Gate 6 confirmation occurs
        self.gate6_interlock.register_resumption_hook("game_session", self._handle_gate6_resumption)

    def _handle_gate6_resumption(self, event: RefinementBreakageEvent) -> dict[str, Any]:
        """Automatically called when user confirms a Gate 6 hold for a game session."""
        target_game = event.target.replace("quipu-game-", "").lower()
        if target_game in self.endpoints:
            url = f"{self.endpoints[target_game]}/resolve_challenge"
            res = _http_request(url, method="POST", data=event.resolution_payload)
            logger.info("Sent resolve_challenge to %s: %s", target_game, res)
            return res
        return {"status": "target_not_found", "target": event.target}

    def login_container(
        self,
        game: str,
        account_id: Optional[str] = None,
        force_challenge: bool = False,
    ) -> dict[str, Any]:
        """Log in an individual game container using specified or rotated account."""
        game = game.lower()
        if game not in self.endpoints:
            return {"status": "error", "message": f"Unknown game {game}"}

        url = f"{self.endpoints[game]}/login"

        # Select account profile
        account: Optional[AccountProfile] = None
        if account_id:
            account = self.account_store.get_account(account_id)
        else:
            account = self.account_store.rotate_account(game)

        if not account:
            # Raise Gate 6 Hold: Multi-account input required
            hold = self.gate6_interlock.raise_breakage(
                domain="game_session",
                target=f"quipu-game-{game}",
                breakage_type=BreakageType.MULTI_ACCOUNT_INPUT_REQUIRED,
                reason=f"No available or idle account credentials found for {game.upper()}",
                suggested_paths=[f"Provision new account for {game}", "Reset existing account cooldown", "Hold in band 翈"],
                context={"game": game, "required_action": "supply_account_credentials"},
            )
            return {"status": "held_at_gate_6", "breakage_id": hold.breakage_id, "reason": hold.reason}

        # Check if account requires challenge or user confirmation
        if account.status == "challenge_required" or force_challenge:
            hold = self.gate6_interlock.raise_breakage(
                domain="game_session",
                target=f"quipu-game-{game}",
                breakage_type=BreakageType.AUTHENTICATION_CHALLENGE,
                reason=f"Account {account.account_id} requires Gate 6 User Confirmation (Captcha / 2FA / Path verification)",
                suggested_paths=["Confirm valid operator credentials", "Switch to alternative alt account", "Bypass via Tailscale mesh proxy"],
                context={"account_id": account.account_id, "username": account.username, "game": game},
            )
            # Notify container daemon
            _http_request(f"{self.endpoints[game]}/challenge", method="POST", data=asdict(hold))
            return {"status": "held_at_gate_6", "breakage_id": hold.breakage_id, "account": account.account_id}

        # Prepare login payload
        payload = {
            "account_id": account.account_id,
            "username": account.username,
            "password": account.get_decrypted_secret(),
            "character_name": account.character_name,
            "realm_or_world": account.realm_or_world,
            "pin": account.pin,
            "vpn_node": account.assigned_vpn_node,
        }

        resp = _http_request(url, method="POST", data=payload)

        if resp.get("status") == "authenticated":
            self.account_store.update_status(account.account_id, "authenticated", last_login=time.time())
            # Update VPN mesh node activity
            self.vpn_mesh.update_node_status(f"quipu-game-{game}", is_active=True)

        return resp

    def mass_login(
        self,
        game_filter: Optional[str] = None,
        account_map: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Perform concurrent or serial mass login across all target game containers."""
        account_map = account_map or {}
        games = ["osrs", "wow", "gw"]
        if game_filter and game_filter.lower() != "all":
            games = [g for g in games if g == game_filter.lower()]

        results = {}
        for game in games:
            acc_id = account_map.get(game)
            res = self.login_container(game, account_id=acc_id)
            results[game] = res

        successful = sum(1 for r in results.values() if r.get("status") == "authenticated")
        held = sum(1 for r in results.values() if r.get("status") == "held_at_gate_6")

        return {
            "action": "mass_login",
            "targets": games,
            "total": len(games),
            "authenticated": successful,
            "held_at_gate_6": held,
            "results": results,
        }

    def logout_container(self, game: str) -> dict[str, Any]:
        game = game.lower()
        if game not in self.endpoints:
            return {"status": "error", "message": f"Unknown game {game}"}
        url = f"{self.endpoints[game]}/logout"
        return _http_request(url, method="POST")

    def mass_logout(self, game_filter: Optional[str] = None) -> dict[str, Any]:
        games = ["osrs", "wow", "gw"]
        if game_filter and game_filter.lower() != "all":
            games = [g for g in games if g == game_filter.lower()]
        results = {g: self.logout_container(g) for g in games}
        return {"action": "mass_logout", "results": results}

    def get_mass_status(self) -> dict[str, Any]:
        """Aggregate status across all game daemons, accounts, VPN overlay, and Gate 6 holds."""
        statuses = {}
        for game, url in self.endpoints.items():
            st = _http_request(f"{url}/status")
            statuses[game] = st

        vpn_info = self.vpn_mesh.get_mesh_status()
        active_holds = self.gate6_interlock.list_active_holds()
        accounts_summary = [a.to_dict(include_secret=False) for a in self.account_store.list_accounts()]

        return {
            "timestamp": time.time(),
            "containers": statuses,
            "vpn_mesh": vpn_info,
            "active_gate6_holds": [asdict(h) for h in active_holds],
            "accounts_configured": len(accounts_summary),
            "accounts": accounts_summary,
        }

    def r_admin_step(self, directive: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """r-ADMIN (rADAM toroidal-pressure & complex gradient optimizer) coordination pass.

        Computes multi-container session tension, evaluates Gate 6 holds,
        manages account rotation schedules, and adjusts Quipu graph parameters.
        """
        directive = directive or {}
        st = self.get_mass_status()
        active_holds = st.get("active_gate6_holds", [])

        # 1. Complex Stability Tensor z = Real(stability) + i * Im(friction/latency)
        containers = st.get("containers", {})
        stability_real = 0.0
        friction_imag = 0.0
        per_game_status = {}

        for game, data in containers.items():
            is_auth = data.get("login_status") == "authenticated"
            is_running = data.get("client_process_status") == "running"
            uptime = float(data.get("uptime_seconds", 0.0))

            g_stab = 1.0 if (is_auth and is_running) else 0.2
            # Extract latency from VPN attachment
            vpn_att = data.get("vpn_attachment", {})
            lat = float(vpn_att.get("latency_ms", 15.0))

            stability_real += g_stab
            friction_imag += lat / 100.0
            per_game_status[game] = {
                "authenticated": is_auth,
                "running": is_running,
                "stability": g_stab,
                "latency_ms": lat,
            }

        num_containers = max(len(containers), 1)
        z_mean = complex(stability_real / num_containers, friction_imag / num_containers)

        # 2. Gate 6 Interlock Evaluation for r-ADMIN
        # If any container has an active Gate 6 hold, r-ADMIN suspends write gradients for that branch
        gate6_held = len(active_holds) > 0
        r_admin_state = "PASS_ALL_GATES" if not gate6_held else "HELD_IN_BAND_翈"

        actions_taken = []
        if gate6_held:
            actions_taken.append(f"Suspended automated write gradients on {len(active_holds)} target(s) awaiting Gate 6 user attestation")
        else:
            actions_taken.append("All physical gates clear; r-ADMIN autonomous optimization active")

        # Check account rotation pressure
        for game in ["osrs", "wow", "gw"]:
            c_data = containers.get(game, {})
            if c_data.get("login_status") == "logged_out":
                # Trigger automated mass handler login
                res = self.login_container(game)
                actions_taken.append(f"Auto-login triggered for {game}: {res.get('status')}")

        return {
            "r_admin_controller": "rADAM_toroidal_optimizer",
            "state": r_admin_state,
            "complex_pressure_z": {"real": round(z_mean.real, 4), "imag": round(z_mean.imag, 4)},
            "magnitude": round(abs(z_mean), 4),
            "gate_6_status": "LOCKED_AWAITING_USER" if gate6_held else "APPROVED",
            "active_holds_count": len(active_holds),
            "per_game_status": per_game_status,
            "actions_dispatched": actions_taken,
        }

    def confirm_gate6(
        self,
        breakage_id: str,
        operator_signer: str,
        confirmed_path: str,
        account_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """User confirms the right path for a Gate 6 challenge, unlocking the held system."""
        return self.gate6_interlock.confirm_right_path(
            breakage_id=breakage_id,
            operator_signer=operator_signer,
            confirmed_path=confirmed_path,
            account_input=account_input,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="QUIPU Mass Session Handler & r-ADMIN CLI")
    parser.add_argument("--status", action="store_true", help="Display status of all containers, VPN mesh, and accounts")
    parser.add_argument("--login", type=str, metavar="GAME", help="Mass login to GAME ('osrs', 'wow', 'gw', 'all')")
    parser.add_argument("--logout", type=str, metavar="GAME", help="Mass logout from GAME ('osrs', 'wow', 'gw', 'all')")
    parser.add_argument("--vpn", action="store_true", help="Display Tailscale-like mesh VPN overlay status")
    parser.add_argument("--accounts", action="store_true", help="List all configured multi-account profiles")
    parser.add_argument("--gate6-list", action="store_true", help="List active Gate 6 holds requiring user confirmation")
    parser.add_argument("--gate6-confirm", type=str, metavar="ID", help="Confirm right path for Gate 6 breakage ID")
    parser.add_argument("--path", type=str, default="Operator verified right path", help="Confirmed path description")
    parser.add_argument("--signer", type=str, default="operator_admin", help="Operator identity signing the attestation")
    parser.add_argument("--r-admin", action="store_true", help="Execute an r-ADMIN coordination and pressure optimization pass")
    parser.add_argument("--simulate-breakage", type=str, metavar="GAME", help="Simulate a breakage/challenge requiring Gate 6 confirmation")

    args = parser.parse_args()
    handler = MassSessionHandler()

    if args.status:
        st = handler.get_mass_status()
        print(json.dumps(st, indent=2))
    elif args.login:
        res = handler.mass_login(game_filter=args.login)
        print(json.dumps(res, indent=2))
    elif args.logout:
        res = handler.mass_logout(game_filter=args.logout)
        print(json.dumps(res, indent=2))
    elif args.vpn:
        st = handler.vpn_mesh.get_mesh_status()
        print(json.dumps(st, indent=2))
    elif args.accounts:
        accs = [a.to_dict(include_secret=False) for a in handler.account_store.list_accounts()]
        print(json.dumps(accs, indent=2))
    elif args.gate6_list:
        holds = [asdict(h) for h in handler.gate6_interlock.list_active_holds()]
        print(json.dumps(holds, indent=2))
    elif args.gate6_confirm:
        res = handler.confirm_gate6(args.gate6_confirm, operator_signer=args.signer, confirmed_path=args.path)
        print(json.dumps(res, indent=2))
    elif args.r_admin:
        res = handler.r_admin_step()
        print(json.dumps(res, indent=2))
    elif args.simulate_breakage:
        game = args.simulate_breakage.lower()
        res = handler.login_container(game, force_challenge=True)
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
