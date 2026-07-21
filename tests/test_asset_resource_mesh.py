"""Unit tests for src.quipu.asset_resource_mesh.

Self-contained: in-memory sqlite graph, fake compute peers, and fake spatial
material processors.  No compute-grid listeners, network calls, or GPUs needed.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pytest

from src.quipu import asset_resource_mesh as arm
from src.quipu.asset_resource_mesh import (
    ASSET_ENTITY_TYPE,
    ASSET_TUNNEL_REL,
    COMPUTE_PEER_TYPE,
    HAS_ASSET_REL,
    REALIZES_PROCESSOR_REL,
    SYSTEM_ENTIRETY_ID,
    SYSTEM_ENTIRETY_TYPE,
    bind_assets_to_material_processors,
    materialize_asset_resources,
    mesh_asset_tunnels,
    resource_share_contracts,
    tick_asset_resource_mesh,
)
from src.quipu.geospatial_relation import ACTUALIZED_SPACE_FRAME, SPATIAL_MATERIAL_PROCESSOR
from src.quipu import system_entirety


def _db() -> sqlite3.Connection:
    cn = sqlite3.connect(":memory:")
    cn.executescript(
        """
        CREATE TABLE corpus_entity(
            entity_id TEXT, entity_type TEXT, label TEXT, props_json TEXT,
            first_seen TEXT, last_seen TEXT, samples INTEGER DEFAULT 1,
            PRIMARY KEY(entity_id, entity_type)
        );
        CREATE TABLE corpus_edge(
            src_id TEXT, src_type TEXT, dst_id TEXT, dst_type TEXT,
            rel TEXT, weight REAL, last_seen TEXT, samples INTEGER DEFAULT 1,
            PRIMARY KEY(src_id, src_type, dst_id, dst_type, rel)
        );
        CREATE TABLE kv_store(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
        """
    )
    return cn


def _peers() -> list[dict]:
    return [
        {
            "host": "alpha",
            "address": "10.0.0.10",
            "port": 8000,
            "cpu_count": 16,
            "cpu_load_1m": 0.25,
            "free_ram_gb": 64.0,
            "total_ram_gb": 128.0,
            "free_vram_gb": 12.0,
            "free_disk_gb": 768.0,
            "total_disk_gb": 1024.0,
            "drive_count": 2,
            "drives": [
                {"device": "C:", "mountpoint": "C:\\", "free_gb": 256.0, "total_gb": 512.0},
                {"device": "D:", "mountpoint": "D:\\", "free_gb": 512.0, "total_gb": 512.0},
            ],
            "gpus": [{"name": "RTX", "free_mb": 12288, "total_mb": 16384}],
            "transport": "tcp",
            "runtime": {
                "runtime_pid": 26020,
                "agent_pid": 26020,
                "watcher_pid": 17344,
                "agent_state": "running",
                "updated_at": "2026-05-20T19:54:30.950730+00:00",
            },
        },
        {
            "host": "beta",
            "address": "10.0.0.11",
            "port": 8000,
            "cpu_count": 8,
            "cpu_load_1m": 0.10,
            "free_ram_gb": 32.0,
            "total_ram_gb": 64.0,
            "free_vram_gb": 0.0,
            "free_disk_gb": 384.0,
            "total_disk_gb": 512.0,
            "drive_count": 1,
            "drives": [
                {"device": "C:", "mountpoint": "C:\\", "free_gb": 384.0, "total_gb": 512.0},
            ],
            "gpus": [],
            "transport": "devtunnel",
            "tunnel_id": "scbrain-beta.use2",
            "runtime": {
                "runtime_pid": 27111,
                "agent_pid": 27111,
                "watcher_pid": 18181,
                "agent_state": "running",
                "updated_at": "2026-05-20T19:54:31.000000+00:00",
            },
        },
    ]


def _seed_processor(cn: sqlite3.Connection, eid: str = "msp:test") -> None:
    props = {
        "world_pose": {
            "xyz_m": [1.0, 2.0, 0.0],
            "precision": 5.0,
            "prob_cert": 0.8,
            "sigma_m": 0.4,
        },
        "actualized_space": {
            "frame_id": ACTUALIZED_SPACE_FRAME,
            "precision": 5.0,
            "prob_cert": 0.8,
            "material_class": "spatial inventory material",
            "material_structure": {
                "processor": "physical_material_science",
                "lattice_strength": 0.85,
            },
        },
        "material_anchor": True,
        "torus_angles": [0.2] * 7,
    }
    cn.execute(
        "INSERT INTO corpus_entity(entity_id, entity_type, label, props_json, "
        " first_seen, last_seen) VALUES(?,?,?,?,?,?)",
        (eid, SPATIAL_MATERIAL_PROCESSOR, eid, json.dumps(props), "2026", "2026"),
    )


def test_materialize_asset_resources_from_compute_peers() -> None:
    cn = _db()
    assets = materialize_asset_resources(cn, _peers())

    # alpha: cores/ram/vram/storage; beta: cores/ram/storage (no vram)
    assert len(assets) == 7
    assert {a.kind for a in assets} == {"cores", "ram", "vram", "storage"}

    peer_count = cn.execute(
        "SELECT COUNT(*) FROM corpus_entity WHERE entity_type=?",
        (COMPUTE_PEER_TYPE,),
    ).fetchone()[0]
    asset_count = cn.execute(
        "SELECT COUNT(*) FROM corpus_entity WHERE entity_type=?",
        (ASSET_ENTITY_TYPE,),
    ).fetchone()[0]
    has_asset_edges = cn.execute(
        "SELECT COUNT(*) FROM corpus_edge WHERE rel=?",
        (HAS_ASSET_REL,),
    ).fetchone()[0]

    assert peer_count == 2
    assert asset_count == 7
    assert has_asset_edges == 7

    row = cn.execute(
        "SELECT props_json FROM corpus_entity WHERE entity_id='asset:alpha:vram' "
        "AND entity_type=?",
        (ASSET_ENTITY_TYPE,),
    ).fetchone()
    props = json.loads(row[0])
    assert props["actualized_resource"]["resource_kind"] == "vram"
    assert props["frame_id"] == ACTUALIZED_SPACE_FRAME
    assert len(props["torus_angles"]) == 7

    storage_row = cn.execute(
        "SELECT props_json FROM corpus_entity WHERE entity_id='asset:alpha:storage' "
        "AND entity_type=?",
        (ASSET_ENTITY_TYPE,),
    ).fetchone()
    storage_props = json.loads(storage_row[0])
    assert storage_props["actualized_resource"]["resource_kind"] == "storage"
    assert storage_props["share_contract"]["unit_family"] == "storage"


def test_assets_bind_to_material_processors_and_mesh_tunnels() -> None:
    cn = _db()
    _seed_processor(cn)
    assets = materialize_asset_resources(cn, _peers())

    processor_stats = bind_assets_to_material_processors(cn, assets)
    tunnel_stats = mesh_asset_tunnels(cn, assets)

    assert processor_stats["processor_edges"] == len(assets)
    assert tunnel_stats["asset_tunnels"] > 0

    realizes_edges = cn.execute(
        "SELECT COUNT(*) FROM corpus_edge WHERE rel=?",
        (REALIZES_PROCESSOR_REL,),
    ).fetchone()[0]
    tunnel_edges = cn.execute(
        "SELECT COUNT(*) FROM corpus_edge WHERE rel=?",
        (ASSET_TUNNEL_REL,),
    ).fetchone()[0]
    assert realizes_edges == len(assets)
    assert tunnel_edges == tunnel_stats["asset_tunnels"]


def test_resource_share_contracts_include_tunnel_metadata() -> None:
    contracts = resource_share_contracts(_peers())

    beta_ram = next(c for c in contracts if c["host"] == "beta" and c["unit_kind"] == "ram")
    alpha_core = next(c for c in contracts if c["host"] == "alpha" and c["unit_kind"] == "cores")
    alpha_storage = next(c for c in contracts if c["host"] == "alpha" and c["unit_kind"] == "storage")

    assert beta_ram["transport"] == "devtunnel"
    assert beta_ram["tunnel_id"] == "scbrain-beta.use2"
    assert beta_ram["unit_family"] == "memory"
    assert beta_ram["runtime"]["runtime_pid"] == 27111
    assert alpha_core["unit_family"] == "core"
    assert alpha_core["runtime"]["agent_pid"] == 26020
    assert alpha_core["watcher_pid"] == 17344
    assert alpha_storage["unit_family"] == "storage"
    assert alpha_storage["capacity_available"] == 768.0


def test_tick_asset_resource_mesh_writes_system_entirety_realization() -> None:
    cn = _db()
    _seed_processor(cn)
    summary = tick_asset_resource_mesh(cn, _peers())

    assert summary["peers"] == 2
    assert summary["assets"] == 7
    assert summary["asset_tunnels"] > 0
    assert summary["processor_edges"] == 7
    assert summary["free_disk_gb"] == 1152.0
    assert summary["drive_count"] == 3
    assert 0.0 < summary["physical_realization"] <= 1.0

    system_row = cn.execute(
        "SELECT props_json FROM corpus_entity WHERE entity_id=? AND entity_type=?",
        (SYSTEM_ENTIRETY_ID, SYSTEM_ENTIRETY_TYPE),
    ).fetchone()
    assert system_row is not None
    system_props = json.loads(system_row[0])
    assert system_props["asset_count"] == 7
    assert system_props["processor_count"] == 1
    assert system_props["resource_sharing"]["core"]["hosts"] == 2
    assert system_props["resource_sharing"]["storage"]["hosts"] == 2
    assert system_props["resource_sharing"]["storage"]["free_disk_gb"] == 1152.0
    assert system_props["resource_sharing"]["storage"]["drive_count"] == 3
    assert system_props["resource_sharing"]["processor"]["processor_edges"] == 7
    assert system_props["resource_sharing"]["mesh"]["distributed"] is True
    assert system_props["resource_sharing"]["mesh"]["peer_count"] == 2
    assert system_props["resource_sharing"]["mesh"]["transport_count"] == 2
    assert system_props["resource_sharing"]["mesh"]["mesh_density"] > 0.0
    assert system_props["resource_sharing"]["mesh"]["active_runtime_pids"] == [26020, 27111]
    assert system_props["mesh_runtime"][0]["runtime_pid"] == 26020

    kv = cn.execute(
        "SELECT value FROM brain_kv WHERE key='entirety:physical_realization'"
    ).fetchone()
    assert kv is not None
    assert json.loads(kv[0])["physical_realization"] == system_props["physical_realization"]


def test_tick_asset_resource_mesh_marks_observed_peers_active(monkeypatch, tmp_path: Path) -> None:
    observed_file = tmp_path / "observed_mesh_assets.json"
    now = datetime.now(timezone.utc).isoformat()
    observed_file.write_text(
        json.dumps(
            {
                "updated_at": now,
                "observer": "DESKTOP-01",
                "peers": [
                    {
                        "host": "LAPTOP-01",
                        "address": "10.0.0.20",
                        "transport": "observed",
                        "port": 0,
                        "cpu_count": 1,
                        "cpu_load_1m": 0.0,
                        "free_ram_gb": 0.0,
                        "free_vram_gb": 0.0,
                        "gpus": [],
                        "rdp_reachable": True,
                        "observed_at": now,
                        "ts": now,
                    },
                    {
                        "host": "LAPTOP-02",
                        "address": "10.0.0.10",
                        "transport": "observed",
                        "port": 0,
                        "cpu_count": 1,
                        "cpu_load_1m": 0.0,
                        "free_ram_gb": 0.0,
                        "free_vram_gb": 0.0,
                        "gpus": [],
                        "rdp_reachable": True,
                        "observed_at": now,
                        "ts": now,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(arm, "_OBSERVED_MESH_ASSETS_FILE", observed_file)

    cn = _db()
    _seed_processor(cn)
    summary = tick_asset_resource_mesh(cn)

    mesh = summary["resource_sharing"]["mesh"]
    runtime_by_host = {entry["host"]: entry for entry in mesh["peer_runtime"]}

    assert summary["peers"] == 2
    assert mesh["active_peer_count"] == 2
    assert mesh["active_hosts"] == ["LAPTOP-01", "LAPTOP-02"]
    assert mesh["active_runtime_pids"] == []
    assert runtime_by_host["LAPTOP-01"]["active"] is True
    assert runtime_by_host["LAPTOP-01"]["agent_state"] == "observed_active"
    assert runtime_by_host["LAPTOP-02"]["active"] is True


def test_tick_preserves_mesh_learning_overlay_payload() -> None:
    cn = _db()
    _seed_processor(cn)
    learned_payload = {
        "physical_realization": 0.55,
        "mesh_learning": {
            "observer": 0.44,
            "recent_learning": {"entry_count": 42, "kind_counts": {"rag_deepdive": 40}},
        },
        "mesh_learning_projection": {
            "window_id": "mesh_learning_window:rolling_24h",
            "kind_nodes": 1,
            "hardening_edges": 5,
        },
        "mesh_learning_updated_at": "2026-05-20T00:00:00+00:00",
    }
    cn.execute(
        "INSERT INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
        ("entirety:physical_realization", json.dumps(learned_payload), "2026-05-20T00:00:00+00:00"),
    )

    tick_asset_resource_mesh(cn, _peers())

    system_props = json.loads(
        cn.execute(
            "SELECT props_json FROM corpus_entity WHERE entity_id=? AND entity_type=?",
            (SYSTEM_ENTIRETY_ID, SYSTEM_ENTIRETY_TYPE),
        ).fetchone()[0]
    )
    kv_payload = json.loads(
        cn.execute(
            "SELECT value FROM brain_kv WHERE key='entirety:physical_realization'"
        ).fetchone()[0]
    )

    assert system_props["mesh_learning"]["recent_learning"]["entry_count"] == 42
    assert system_props["mesh_learning_projection"]["hardening_edges"] == 5
    assert kv_payload["mesh_learning"]["observer"] == pytest.approx(0.44)
    assert kv_payload["mesh_learning_updated_at"] == "2026-05-20T00:00:00+00:00"


def test_tick_projects_gpu_resuscitation_from_hydrated_vram_nodes() -> None:
    cn = _db()
    _seed_processor(cn)
    cn.execute(
        "CREATE TABLE mesh_slm_quipu_node("
        "node_id INTEGER PRIMARY KEY, i INTEGER NOT NULL, j INTEGER NOT NULL, "
        "node_phase REAL NOT NULL, weyl_phase REAL NOT NULL, photon_phase REAL NOT NULL, "
        "neutrino_phase REAL NOT NULL, interaction_gain REAL NOT NULL, "
        "resuscitation_weight REAL NOT NULL, directed_target INTEGER NOT NULL, "
        "source_key TEXT NOT NULL DEFAULT '', source_label TEXT NOT NULL DEFAULT '', "
        "updated_at TEXT NOT NULL)"
    )
    rows = []
    for node_id in range(64):
        rows.append(
            (
                node_id,
                node_id // 8,
                node_id % 8,
                0.1 * node_id,
                2.253,
                0.05 * node_id,
                0.03 * node_id,
                0.55 + 0.003 * node_id,
                1.20 + 0.01 * node_id,
                (node_id + 17) % 4096,
                "vram_rehydrate:asset:alpha:vram",
                "asset:alpha:vram",
                "2026-05-23T00:00:00+00:00",
            )
        )
    cn.executemany(
        "INSERT INTO mesh_slm_quipu_node("
        "node_id, i, j, node_phase, weyl_phase, photon_phase, neutrino_phase, "
        "interaction_gain, resuscitation_weight, directed_target, source_key, source_label, updated_at"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    cn.execute(
        "INSERT INTO kv_store(key, value) VALUES(?, ?)",
        (
            "vram_rehydrate:asset:alpha:vram",
            json.dumps({"target": "asset:alpha:vram", "rehydrate_weight": 3.13508}),
        ),
    )
    cn.execute(
        "INSERT INTO brain_kv(key, value, updated_at) VALUES(?, ?, ?)",
        (
            "ueqgm:adaptive_runtime",
            json.dumps(
                {
                    "active": True,
                    "coherence_depth": 2,
                    "mesh_alignment": 0.63,
                    "observer_alignment": 0.74,
                    "phase_weight": 1.2058,
                    "symbiotic_gain": 0.58,
                }
            ),
            "2026-05-23T00:00:00+00:00",
        ),
    )

    summary = tick_asset_resource_mesh(cn, _peers())

    system_props = json.loads(
        cn.execute(
            "SELECT props_json FROM corpus_entity WHERE entity_id=? AND entity_type=?",
            (SYSTEM_ENTIRETY_ID, SYSTEM_ENTIRETY_TYPE),
        ).fetchone()[0]
    )
    alpha_vram_props = json.loads(
        cn.execute(
            "SELECT props_json FROM corpus_entity WHERE entity_id='asset:alpha:vram' AND entity_type=?",
            (ASSET_ENTITY_TYPE,),
        ).fetchone()[0]
    )
    resuscitation_kv = json.loads(
        cn.execute(
            "SELECT value FROM kv_store WHERE key='gpu_resuscitation:asset:alpha:vram'"
        ).fetchone()[0]
    )
    generic_amp = float(
        cn.execute(
            "SELECT value FROM kv_store WHERE key='torus_amplify:asset:alpha:vram'"
        ).fetchone()[0]
    )
    typed_amp = float(
        cn.execute(
            "SELECT value FROM kv_store WHERE key='torus_amplify:AssetResource:asset:alpha:vram'"
        ).fetchone()[0]
    )

    assert summary["gpu_resuscitation"]["active"] is True
    assert summary["gpu_resuscitation"]["source_label"] == "asset:alpha:vram"
    assert summary["gpu_resuscitation"]["receiver_material"] == "tantalum"
    assert summary["gpu_resuscitation"]["tantalum_binding"]["intermediary"] == "tantalum"
    assert summary["gpu_resuscitation"]["tantalum_binding"]["coherence_depth"] == 2
    assert system_props["resource_sharing"]["memory"]["gpu_resuscitation"]["node_count"] == 64
    assert system_props["gpu_resuscitation"]["targeted_assets"] == 1
    assert alpha_vram_props["gpu_resuscitation"]["tantalum_binding"]["binding_gain"] > 0.0
    assert alpha_vram_props["gpu_resuscitation"]["hydrated_vram_gb"] > 0.0
    assert alpha_vram_props["actualized_resource"]["compression_ratio"] > 0.0
    assert alpha_vram_props["actualized_resource"]["intermediary_binding"] == "tantalum"
    assert alpha_vram_props["actualized_resource"]["receiver_material"] == "tantalum"
    assert resuscitation_kv["compression_ratio"] == alpha_vram_props["gpu_resuscitation"]["compression_ratio"]
    assert generic_amp == pytest.approx(alpha_vram_props["gpu_resuscitation"]["effective_torus_amplify"])
    assert typed_amp == pytest.approx(generic_amp)


def test_tick_redirects_resuscitation_to_gpu_make_buy_bit_flip() -> None:
    cn = _db()
    _seed_processor(cn)
    cn.execute(
        "CREATE TABLE mesh_slm_quipu_node("
        "node_id INTEGER PRIMARY KEY, i INTEGER NOT NULL, j INTEGER NOT NULL, "
        "node_phase REAL NOT NULL, weyl_phase REAL NOT NULL, photon_phase REAL NOT NULL, "
        "neutrino_phase REAL NOT NULL, interaction_gain REAL NOT NULL, "
        "resuscitation_weight REAL NOT NULL, directed_target INTEGER NOT NULL, "
        "source_key TEXT NOT NULL DEFAULT '', source_label TEXT NOT NULL DEFAULT '', "
        "updated_at TEXT NOT NULL)"
    )
    cn.execute(
        "INSERT INTO mesh_slm_quipu_node("
        "node_id, i, j, node_phase, weyl_phase, photon_phase, neutrino_phase, "
        "interaction_gain, resuscitation_weight, directed_target, source_key, source_label, updated_at"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            12,
            1,
            4,
            1.2,
            2.253,
            0.4,
            0.8,
            0.86,
            2.4,
            144,
            "vram_rehydrate:asset:alpha:vram",
            "asset:alpha:vram",
            "2026-05-23T00:00:00+00:00",
        ),
    )
    cn.execute(
        "INSERT INTO kv_store(key, value) VALUES(?, ?)",
        (
            "vram_rehydrate:asset:alpha:vram",
            json.dumps(
                {
                    "target": "asset:alpha:vram",
                    "rehydrate_weight": 2.95,
                    "weyl": 2.253,
                    "gpu_parts": [
                        {
                            "part_number": "GPU-TA-CAP-001",
                            "make_buy": "MAKE",
                            "material": "tantalum",
                            "target": "asset:alpha:vram",
                            "host": "alpha",
                            "bit_flip_node": 12,
                            "directed_target": 144,
                            "bit_flip_weight": 0.91,
                            "connected": True,
                        }
                    ],
                }
            ),
        ),
    )

    summary = tick_asset_resource_mesh(cn, _peers())

    alpha_vram_props = json.loads(
        cn.execute(
            "SELECT props_json FROM corpus_entity WHERE entity_id='asset:alpha:vram' AND entity_type=?",
            (ASSET_ENTITY_TYPE,),
        ).fetchone()[0]
    )
    profile = alpha_vram_props["gpu_resuscitation"]
    bit_flip = profile["bit_flip_resuscitation"]

    assert bit_flip["selected_part"] == "GPU-TA-CAP-001"
    assert bit_flip["selected_make_buy"] == "MAKE"
    assert bit_flip["selected_material"] == "tantalum"
    assert bit_flip["connection_proven"] is True
    assert bit_flip["vram_touch"] is True
    assert bit_flip["bit_flip_node"] == 12
    assert bit_flip["bit_flip_target"] == 144
    assert bit_flip["bridge_reactivation_gain"] > 0.0
    assert profile["hydrated_vram_gb"] > 0.0
    assert alpha_vram_props["actualized_resource"]["part_make_buy"] == "MAKE"
    assert alpha_vram_props["actualized_resource"]["bit_flip_node"] == 12
    assert summary["gpu_resuscitation"]["bit_flip_resuscitation"]["make_buy_modes"] == ["MAKE",]
    assert summary["gpu_resuscitation"]["bit_flip_resuscitation"]["vram_touch"] is True


def test_tick_uses_vault_backed_historic_gpu_receiver_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cn = _db()
    _seed_processor(cn)
    cn.execute(
        "CREATE TABLE mesh_slm_quipu_node("
        "node_id INTEGER PRIMARY KEY, i INTEGER NOT NULL, j INTEGER NOT NULL, "
        "node_phase REAL NOT NULL, weyl_phase REAL NOT NULL, photon_phase REAL NOT NULL, "
        "neutrino_phase REAL NOT NULL, interaction_gain REAL NOT NULL, "
        "resuscitation_weight REAL NOT NULL, directed_target INTEGER NOT NULL, "
        "source_key TEXT NOT NULL DEFAULT '', source_label TEXT NOT NULL DEFAULT '', "
        "updated_at TEXT NOT NULL)"
    )
    cn.execute(
        "INSERT INTO mesh_slm_quipu_node("
        "node_id, i, j, node_phase, weyl_phase, photon_phase, neutrino_phase, "
        "interaction_gain, resuscitation_weight, directed_target, source_key, source_label, updated_at"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            0,
            0,
            0,
            0.0,
            2.253,
            0.0,
            0.0,
            0.75,
            1.5,
            17,
            "vram_rehydrate:asset:scbrain-hideout:vram",
            "asset:scbrain-hideout:vram",
            "2026-05-23T00:00:00+00:00",
        ),
    )
    cn.execute(
        "INSERT INTO kv_store(key, value) VALUES(?, ?)",
        (
            "vram_rehydrate:asset:alpha:vram",
            json.dumps({"target": "asset:alpha:vram", "rehydrate_weight": 2.75}),
        ),
    )
    cn.execute(
        "INSERT INTO kv_store(key, value) VALUES(?, ?)",
        (
            "vram_rehydrate:asset:scbrain-hideout:vram",
            json.dumps({"target": "asset:scbrain-hideout:vram", "rehydrate_weight": 3.25}),
        ),
    )
    vault = tmp_path / "vpn_cred.bin"
    vault.write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("SCB_VPN_VAULT_FILE", str(vault))

    summary = tick_asset_resource_mesh(cn, _peers())

    assert summary["gpu_resuscitation"]["source_label"] == "asset:alpha:vram"
    assert summary["gpu_resuscitation"]["receiver_history"]["credential_mode"] == "vault"
    assert summary["gpu_resuscitation"]["receiver_history"]["selected_source_key"] == "vram_rehydrate:asset:alpha:vram"
    assert summary["gpu_resuscitation"]["receiver_material"] == "tantalum"

    alpha_vram_props = json.loads(
        cn.execute(
            "SELECT props_json FROM corpus_entity WHERE entity_id='asset:alpha:vram' AND entity_type=?",
            (ASSET_ENTITY_TYPE,),
        ).fetchone()[0]
    )
    assert alpha_vram_props["gpu_resuscitation"]["source_key"] == "vram_rehydrate:asset:alpha:vram"
    assert alpha_vram_props["gpu_resuscitation"]["receiver_history"]["selected_target"] == "asset:alpha:vram"
    assert alpha_vram_props["gpu_resuscitation"]["receiver_role"] == "gpu_receiver_metal"


def test_asset_resources_participate_in_torus_pressure() -> None:
    from src.quipu.torus_touch import tick_torus_pressure

    cn = _db()
    tick_asset_resource_mesh(cn, _peers())
    stats = tick_torus_pressure(cn, step=0.25)

    assert stats["material_processors"] >= 5
    assert stats["active_entities"] >= 5
    assert stats["moved"] > 0

    rows = cn.execute(
        "SELECT props_json FROM corpus_entity WHERE entity_type=?",
        (ASSET_ENTITY_TYPE,),
    ).fetchall()
    assert rows
    for (props_json,) in rows:
        props = json.loads(props_json)
        assert "torus_tick" in props
        assert len(props["torus_angles"]) == 7


def test_tick_is_idempotent_and_reinforces_edges() -> None:
    cn = _db()
    _seed_processor(cn)
    first = tick_asset_resource_mesh(cn, _peers())
    second = tick_asset_resource_mesh(cn, _peers())

    assert first["assets"] == second["assets"] == 7
    samples = cn.execute(
        "SELECT samples FROM corpus_edge WHERE rel=? LIMIT 1",
        (ASSET_TUNNEL_REL,),
    ).fetchone()[0]
    assert samples >= 2


def test_asset_mesh_torus_and_entirety_transaction_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.quipu.torus_touch import tick_torus_pressure

    uri = "file:asset-mesh-proof?mode=memory&cache=shared"
    keeper = sqlite3.connect(uri, uri=True, check_same_thread=False)
    keeper.executescript(
        """
        CREATE TABLE corpus_entity(
            entity_id TEXT, entity_type TEXT, label TEXT, props_json TEXT,
            first_seen TEXT, last_seen TEXT, samples INTEGER DEFAULT 1,
            PRIMARY KEY(entity_id, entity_type)
        );
        CREATE TABLE corpus_edge(
            src_id TEXT, src_type TEXT, dst_id TEXT, dst_type TEXT,
            rel TEXT, weight REAL, last_seen TEXT, samples INTEGER DEFAULT 1,
            PRIMARY KEY(src_id, src_type, dst_id, dst_type, rel)
        );
        CREATE TABLE kv_store(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
        """
    )
    _seed_processor(keeper)

    monkeypatch.setattr(
        system_entirety,
        "_open_conn",
        lambda *args, **kwargs: sqlite3.connect(
            uri,
            uri=True,
            timeout=float(kwargs.get("timeout", 30)),
            check_same_thread=False,
        ),
    )

    try:
        mesh_summary = tick_asset_resource_mesh(keeper, _peers())
        torus_stats = tick_torus_pressure(keeper, step=0.25)
        aligned = {
            "vision": 0.23,
            "touch": 0.19,
            "smell": 0.09,
            "body": 0.14,
            "brain": 0.21,
            "perception": 0.14,
        }
        state = system_entirety.system_entirety_state(aligned)
        flip = system_entirety.session_entirety_evaluation(aligned)

        asset_count = keeper.execute(
            "SELECT COUNT(*) FROM corpus_entity WHERE entity_type=?",
            (ASSET_ENTITY_TYPE,),
        ).fetchone()[0]
        tunnel_count = keeper.execute(
            "SELECT COUNT(*) FROM corpus_edge WHERE rel=?",
            (ASSET_TUNNEL_REL,),
        ).fetchone()[0]
        binding_count = keeper.execute(
            "SELECT COUNT(*) FROM corpus_edge WHERE rel=?",
            (REALIZES_PROCESSOR_REL,),
        ).fetchone()[0]
        system_realizes_count = keeper.execute(
            "SELECT COUNT(*) FROM corpus_edge WHERE src_id=? AND src_type=? AND rel='SYSTEM_ENTIRETY_REALIZES'",
            (SYSTEM_ENTIRETY_ID, SYSTEM_ENTIRETY_TYPE),
        ).fetchone()[0]

        assert mesh_summary["assets"] == asset_count == 7
        assert mesh_summary["free_disk_gb"] > 0.0
        assert mesh_summary["asset_tunnels"] == tunnel_count > 0
        assert mesh_summary["processor_edges"] == binding_count > 0
        assert system_realizes_count == asset_count
        assert torus_stats["active_entities"] >= asset_count
        assert torus_stats["moved"] > 0
        assert state["material_bifurcation"]["eligible"] is True
        assert state["material_bifurcation"]["mesh"]["peer_count"] == 2
        assert state["transaction"]["bit_flip"] == "mesh_bifurcated"
        assert state["transaction"]["topology"] == "mesh"
        assert flip["transaction"]["drive"] > 0.0
        assert flip["bit_state"] in (-1, 1)
    finally:
        keeper.close()