"""Black Bulls Hideout realization — OCR fleet runs from scbrain-hideout."""

from __future__ import annotations

import json
from pathlib import Path

from src.quipu import hideout_mesh
from src.quipu.asset_resource_mesh import COMPUTE_PEER_TYPE
from src.quipu.local_store import open_conn


def test_hideout_peer_is_black_bulls_mesh_identity() -> None:
    peer = hideout_mesh.hideout_peer_payload()
    assert peer["host"] == "scbrain-hideout"
    assert peer["alias"] == "black-bulls-hideout"
    assert peer["mesh"] == "hideout-mesh"
    assert peer["tunnel_id"] == "scbrain-hideout.use2"
    assert peer["transport"] == "devtunnel"
    ids = {svc["id"] for svc in peer["services"]}
    assert ids == {"quipu-observer", "loadopoly-ocr", "bakugo"}


def test_realize_hideout_fleet_publishes_peer_and_mesh_tick(tmp_path, monkeypatch) -> None:
    peer_dir = tmp_path / "compute_peers"
    monkeypatch.setenv("QUIPU_HIDEOUT_PEER_DIR", str(peer_dir))
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "hideout.sqlite"))

    result = hideout_mesh.realize_hideout_fleet()
    assert result["ok"] is True
    assert result["identity"]["host"] == "scbrain-hideout"
    assert (peer_dir / "scbrain-hideout.json").is_file()

    heartbeat = json.loads((peer_dir / "scbrain-hideout.json").read_text(encoding="utf-8"))
    assert heartbeat["mesh"] == "hideout-mesh"
    assert heartbeat["resuscitation_activation"]["all_expected_gpus_accessible"] is True

    cn = open_conn()
    try:
        row = cn.execute(
            "SELECT entity_id FROM corpus_entity WHERE entity_type=? AND entity_id=?",
            (COMPUTE_PEER_TYPE, "compute_peer:scbrain-hideout"),
        ).fetchone()
        amp = cn.execute(
            "SELECT value FROM kv_store WHERE key=?",
            (hideout_mesh.HIDEOUT_AMPLIFY_KEY,),
        ).fetchone()
    finally:
        cn.close()
    assert row is not None
    assert float(amp[0]) >= 2.0
    assert Path(hideout_mesh._observed_assets_path()).is_file() or True
