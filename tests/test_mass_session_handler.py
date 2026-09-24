"""Tests for QUIPU Multi-Game Mass Session Handler, VPN Mesh, and Gate 6 User Integration."""

import json
from pathlib import Path
import pytest

from quipu.games.account_store import AccountProfile, AccountStore, encrypt_secret, decrypt_secret
from quipu.games.gate6_user_interlock import (
    BreakageType,
    Gate6UserInterlock,
    RefinementBreakageEvent,
)
from quipu.games.mass_session_handler import MassSessionHandler
from quipu.games.vpn_mesh_integration import TailscaleMeshIntegration
from quipu.systemic_refinement_agent import (
    raise_refinement_breakage,
    confirm_refinement_breakage,
    run_strategy,
)


def test_account_store_encryption_and_rotation(tmp_path: Path):
    """Verify AES-256-GCM encryption, profile persistence, and round-robin rotation."""
    store_file = tmp_path / "accounts.json"
    store = AccountStore(storage_file=store_file, load_from_kv=False)

    # Verify default seeded accounts exist
    all_accs = store.list_accounts()
    assert len(all_accs) >= 6
    osrs_accs = store.list_accounts("osrs")
    assert len(osrs_accs) >= 2
    wow_accs = store.list_accounts("wow")
    assert len(wow_accs) >= 2
    gw_accs = store.list_accounts("gw")
    assert len(gw_accs) >= 2

    # Verify encryption roundtrip
    raw_secret = "SecretPassword123!"
    enc = encrypt_secret(raw_secret)
    assert enc != raw_secret
    dec = decrypt_secret(enc)
    assert dec == raw_secret

    # Verify rotation chooses idle account
    rotated = store.rotate_account("osrs")
    assert rotated is not None
    assert rotated.game == "osrs"
    assert rotated.status == "idle"

    # Mark as authenticated
    store.update_status(rotated.account_id, "authenticated")
    rotated_next = store.rotate_account("osrs")
    assert rotated_next is not None
    assert rotated_next.account_id != rotated.account_id


def test_tailscale_mesh_overlay_topology():
    """Verify Tailscale-style mesh VPN topology, CGNAT allocations, and peer ping."""
    vpn = TailscaleMeshIntegration(tailnet="quipu.test.mesh")
    st = vpn.get_mesh_status()

    assert st["tailnet"] == "quipu.test.mesh"
    assert "100.64.0" in st["cgnat_subnet"]
    assert st["total_nodes"] >= 5

    # Check node assignments
    osrs_node = vpn.get_node("quipu-game-osrs")
    assert osrs_node is not None
    assert osrs_node.overlay_ip == "100.64.0.10"
    assert osrs_node.tailnet_fqdn == "osrs.quipu.test.mesh"

    wow_node = vpn.get_node("quipu-game-wow")
    assert wow_node is not None
    assert wow_node.overlay_ip == "100.64.0.20"

    gw_node = vpn.get_node("quipu-game-gw")
    assert gw_node is not None
    assert gw_node.overlay_ip == "100.64.0.30"

    # Peer ping test
    ping = vpn.ping_mesh("quipu-game-osrs", "quipu-game-wow")
    assert ping["status"] == "connected"
    assert ping["target_ip"] == "100.64.0.20"
    assert ping["latency_ms"] > 0
    assert ping["encrypted"] is True


def test_gate6_user_interlock_breakage_and_resolution():
    """Verify Physical Gate 6 Hold creation, attestation validation, and resolution."""
    interlock = Gate6UserInterlock(operator_id="operator_tester", persist=False)

    # Raise an autonomous breakage event (e.g. Captcha challenge)
    event = interlock.raise_breakage(
        domain="game_session",
        target="quipu-game-osrs",
        breakage_type=BreakageType.CAPTCHA_VERIFICATION,
        reason="Suspicious bot detection alert; manual operator path confirmation required",
        suggested_paths=["Solve visual captcha", "Switch proxy / Tailscale exit node", "Log out"],
        context={"coordinate": [3222, 3218], "game": "osrs"},
    )

    assert event.status == "HELD_AT_GATE_6"
    assert event.breakage_id.startswith("brk_")

    # Verify event appears in active holds
    active = interlock.list_active_holds("game_session")
    assert any(h.breakage_id == event.breakage_id for h in active)

    # Operator confirms right path with Gate 6 attestation
    resolution = interlock.confirm_right_path(
        breakage_id=event.breakage_id,
        operator_signer="operator_tester",
        confirmed_path="Solve visual captcha and path wander to Lumbridge castle",
        user_inputs={"pin_response": "1984"},
    )

    assert resolution["status"] == "resolved"
    assert resolution["gate_6"] == "PASSED"

    # Verify event updated
    updated = interlock.get_event(event.breakage_id)
    assert updated is not None
    assert updated.status == "RESOLVED_BY_USER"
    assert updated.resolved_by == "operator_tester"
    assert updated.confirmed_path == "Solve visual captcha and path wander to Lumbridge castle"


def test_mass_session_handler_and_radmin_integration(tmp_path: Path):
    """Verify MassSessionHandler orchestrates multi-container logins and r-ADMIN steps."""
    store = AccountStore(storage_file=tmp_path / "accounts_test.json", load_from_kv=False)
    vpn = TailscaleMeshIntegration()
    interlock = Gate6UserInterlock(persist=False)

    handler = MassSessionHandler(
        account_store=store,
        vpn_mesh=vpn,
        gate6_interlock=interlock,
    )

    # Status check
    st = handler.get_mass_status()
    assert "containers" in st
    assert "vpn_mesh" in st
    assert st["accounts_configured"] >= 6

    # Test r-ADMIN step pass
    radmin_out = handler.r_admin_step()
    assert "r_admin_controller" in radmin_out
    assert "complex_pressure_z" in radmin_out
    assert "magnitude" in radmin_out

    # Simulate a challenge on WoW
    challenge_event = interlock.raise_breakage(
        domain="game_session",
        target="quipu-game-wow",
        breakage_type=BreakageType.PATH_OBSTRUCTION,
        reason="Westfall patrol blocked by Skull-level defias mob",
        suggested_paths=["Disengage and wander through farm perimeter", "Call r-ADMIN subsidiary escape"],
    )

    # Now r-ADMIN step should detect Gate 6 Hold and lock write gradients
    radmin_held = handler.r_admin_step()
    assert radmin_held["gate_6_status"] == "LOCKED_AWAITING_USER"
    assert radmin_held["active_holds_count"] >= 1
    assert "HELD_IN_BAND_翈" in radmin_held["state"]

    # User confirms right path
    confirm_res = handler.confirm_gate6(
        breakage_id=challenge_event.breakage_id,
        operator_signer="human_supervisor",
        confirmed_path="Disengage and wander through farm perimeter",
    )
    assert confirm_res["status"] == "resolved"

    # r-ADMIN should clear the hold on next step
    radmin_cleared = handler.r_admin_step()
    assert radmin_cleared["gate_6_status"] == "APPROVED"


def test_systemic_refinement_common_gate6_interlock():
    """Verify systemic refinement agent triggers and respects common Gate 6 interlock."""
    # Raise a refinement breakage
    bid = raise_refinement_breakage(
        reason="ACRE specialist emergence divergent: loss curvature anomaly",
        suggested_paths=["Retrain with human prior", "Rollback to checkpoint"],
    )
    assert bid.startswith("brk_")

    # Run strategy; should halt in HELD_AT_GATE_6
    cycle = run_strategy()
    assert cycle["status"] == "held_at_gate_6"
    assert any("held_at_gate_6" in a for a in cycle["actions"])

    # User confirms path
    res = confirm_refinement_breakage(
        breakage_id=bid,
        operator_signer="refinement_curator",
        confirmed_path="Retrain with human prior",
    )
    assert res["status"] == "resolved"
    assert res["gate_6"] == "PASSED"
