from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import src.quipu.mesh_entirety as mesh_entirety
import src.quipu.system_entirety as system_entirety


def test_bridge_root_without_queue_rows_becomes_mesh_node(tmp_path, monkeypatch):
    db_file = tmp_path / "mesh.sqlite"
    queue_file = tmp_path / "missing_cloud_learning_queue.jsonl"

    def _open_conn(timeout: float = 60):
        cn = sqlite3.connect(db_file, timeout=timeout)
        cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        cn.execute("CREATE TABLE IF NOT EXISTS kv_store(key TEXT PRIMARY KEY, value TEXT)")
        return cn

    monkeypatch.setattr(mesh_entirety, "_open_conn", _open_conn)
    monkeypatch.setattr(mesh_entirety, "_queue_path", lambda: str(queue_file))
    monkeypatch.setattr(system_entirety, "get_entirety_state", lambda: {"observer": 0.0})

    now = datetime.now(timezone.utc).isoformat()
    summary = {
        "has_data": True,
        "computed_at": now,
        "primary_root": "hideout-mesh",
        "total_roots": 1,
        "total_endpoints": 5,
        "mesh_density": 0.72,
        "nodal_contribution": 0.61,
        "tunnel_saturation": 1.0,
        "anchor_strength": 0.42,
        "primary_root_anchor": 5.8,
    }
    density_map = {
        **summary,
        "roots": {
            "hideout-mesh": {
                "subtree_endpoints": 5,
                "edge_density": 0.74,
                "anchor_weight": 5.8,
                "root_anchor_weight": 5.8,
                "anchor_contribution": 0.62,
                "tunnel_contribution": 1.0,
                "next_hop_expansion_score": 0.66,
            }
        },
    }
    with _open_conn() as cn:
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            ("entirety:bridge_mesh_density", json.dumps(summary), now),
        )
        cn.execute(
            "INSERT OR REPLACE INTO kv_store(key, value) VALUES(?,?)",
            ("bridge_mesh_density_map", json.dumps(density_map)),
        )
        cn.commit()

    nodes = mesh_entirety.mesh_node_states()
    hideout = next(node for node in nodes if node["host"] == "hideout-mesh")

    assert hideout["bridge_root"] is True
    assert hideout["source"] == "bridge_mesh_density"
    assert hideout["recency"] > 0.95
    assert hideout["connectivity"] == 1.0
    assert hideout["coherence"] > 0.5