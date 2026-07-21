"""Tests for quipu_minimal.py — local-minima distillation overlay on the Quipu graph."""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import types

import pytest


@pytest.fixture
def isolated_qm(tmp_path, monkeypatch):
    """Isolated SQLite brain + reloaded mesh_slm/quipu_minimal modules."""
    db_file = tmp_path / "qm.sqlite"
    monkeypatch.setenv("SCB_DB_PATH", str(db_file))
    monkeypatch.delenv("SCB_QUIPU_MINIMAL", raising=False)

    import importlib
    import src.quipu.local_store as ls
    importlib.reload(ls)
    import src.quipu.mesh_slm as mesh_slm
    importlib.reload(mesh_slm)
    import src.quipu.quipu_minimal as qm
    importlib.reload(qm)

    # Seed corpus tables so train_round has text (same shape as test_mesh_slm).
    cn = sqlite3.connect(str(db_file))
    cn.executescript("""
        CREATE TABLE IF NOT EXISTS corpus_entity(
            entity_id TEXT, entity_type TEXT, label TEXT, props_json TEXT,
            first_seen TEXT, last_seen TEXT, samples INTEGER DEFAULT 1,
            PRIMARY KEY(entity_id, entity_type));
        CREATE TABLE IF NOT EXISTS corpus_edge(
            src_id TEXT, src_type TEXT, dst_id TEXT, dst_type TEXT, rel TEXT,
            weight REAL, last_seen TEXT, samples INTEGER DEFAULT 1,
            PRIMARY KEY(src_id, src_type, dst_id, dst_type, rel));
    """)
    for eid, et, lab in [
        ("part_001", "Part",   "carbide insert assembly"),
        ("part_002", "Part",   "hydraulic cylinder piston rod"),
        ("vend_010", "Vendor", "contoso3pl supply chain"),
        ("site_aa",  "Site",   "siteb plant warehouse"),
    ]:
        cn.execute(
            "INSERT INTO corpus_entity(entity_id, entity_type, label, "
            "first_seen, last_seen) VALUES(?,?,?,'2026-05-01','2026-05-20')",
            (eid, et, lab),
        )
    for s, st, d, dt, rel in [
        ("part_001", "Part", "vend_010", "Vendor", "SUPPLIED_BY"),
        ("vend_010", "Vendor", "site_aa", "Site", "DELIVERS_TO"),
    ]:
        cn.execute(
            "INSERT INTO corpus_edge(src_id, src_type, dst_id, dst_type, rel, "
            "weight, last_seen) VALUES(?,?,?,?,?, 0.8, '2026-05-20')",
            (s, st, d, dt, rel),
        )
    cn.commit()
    cn.close()
    yield mesh_slm, qm


def _seed_controlled_graph(qm) -> dict[str, int]:
    """Insert a 4-token graph where the greedy first hop is a trap.

    Edges (weight):  alpha→beta 0.95,  beta→delta 0.05,
                     alpha→gamma 0.60, gamma→delta 0.95
    All embeddings identical → cat6_distance = 0 for every pair, and no
    quipu_node rows → constant resuscitation cost, so edge-cost ordering is
    governed purely by (1 − qweight).
    """
    ids: dict[str, int] = {}
    with qm._conn() as cn:
        for token, i, j in [
            ("alpha", 1, 1), ("beta", 2, 2),
            ("gamma", 3, 3), ("delta", 4, 4),
        ]:
            cur = cn.execute(
                "INSERT INTO mesh_slm_vocab(token, i, j, freq, first_seen, "
                "last_seen) VALUES(?,?,?,1,'2026-06-01','2026-06-01')",
                (token, i, j),
            )
            tid = int(cur.lastrowid)
            ids[token] = tid
            cn.execute(
                "INSERT INTO mesh_slm_embed(token_id, e_vision, e_touch, "
                "e_smell, e_body, e_brain, e_perception, e_entirety) "
                "VALUES(?, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)",
                (tid,),
            )
        for src, dst, w in [
            ("alpha", "beta", 0.95), ("beta", "delta", 0.05),
            ("alpha", "gamma", 0.60), ("gamma", "delta", 0.95),
        ]:
            cn.execute(
                "INSERT INTO mesh_slm_quipu(src, dst, weight, samples) "
                "VALUES(?,?,?,1)",
                (ids[src], ids[dst], w),
            )
    return ids


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_schema_idempotent_double_init(isolated_qm):
    _, qm = isolated_qm
    for _ in range(2):  # second pass must not raise on existing tables
        with qm._conn() as cn:
            names = {
                r[0] for r in cn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
    assert "mesh_slm_minimal_pair" in names
    assert "mesh_slm_minimal_route" in names


def test_overlay_never_prunes_quipu(isolated_qm):
    slm, qm = isolated_qm
    slm.train_round(max_seconds=10.0, max_chunks=50)
    with qm._conn() as cn:
        before = cn.execute(
            "SELECT COUNT(*) FROM mesh_slm_quipu"
        ).fetchone()[0]
        vocab_before = cn.execute(
            "SELECT COUNT(*) FROM mesh_slm_vocab"
        ).fetchone()[0]
        qm.score_entangled_pairs(cn, top_k=16)
        qm.select_minimal_context(cn, ["carbide", "insert"], 0.0, 0.0)
    with qm._conn() as cn:
        assert cn.execute(
            "SELECT COUNT(*) FROM mesh_slm_quipu"
        ).fetchone()[0] == before
        assert cn.execute(
            "SELECT COUNT(*) FROM mesh_slm_vocab"
        ).fetchone()[0] == vocab_before


# ---------------------------------------------------------------------------
# Phase A — entangled pair scoring
# ---------------------------------------------------------------------------
def test_pair_scores_bounded_and_finite(isolated_qm):
    slm, qm = isolated_qm
    slm.train_round(max_seconds=10.0, max_chunks=50)
    with qm._conn() as cn:
        pairs = qm.score_entangled_pairs(cn, top_k=32)
    assert pairs, "training should yield at least one entangled pair"
    for p in pairs:
        for key in ("pair_score", "qweight", "sufficiency", "cat_distance"):
            assert math.isfinite(p[key])
            assert 0.0 <= p[key] <= 1.0
        assert 0 <= p["specialist_support"] <= 3


# ---------------------------------------------------------------------------
# Phase C — 6D CAT geometry
# ---------------------------------------------------------------------------
def test_cat6_distance_wraparound_and_symmetry(isolated_qm):
    _, qm = isolated_qm
    a = [0.05] * 6
    b = [0.95] * 6
    d_ab = qm.cat6_distance(a, b)
    d_ba = qm.cat6_distance(b, a)
    assert d_ab == pytest.approx(d_ba)          # symmetric
    # wrap: per-axis distance is min(0.9, 0.1) = 0.1 → total 0.6/3.0 = 0.2
    assert d_ab == pytest.approx(0.2)
    assert qm.cat6_distance(a, a) == 0.0
    assert 0.0 <= qm.cat6_distance([0.0] * 6, [0.5] * 6) <= 1.0


# ---------------------------------------------------------------------------
# Phase C — Dijkstra vs greedy
# ---------------------------------------------------------------------------
def test_dijkstra_beats_greedy_on_trap_graph(isolated_qm):
    _, qm = isolated_qm
    ids = _seed_controlled_graph(qm)
    with qm._conn() as cn:
        dj = qm.dijkstra_route(cn, ids["alpha"], ids["delta"], observer=1.0)
        gr = qm.greedy_route(cn, ids["alpha"], ids["delta"], observer=1.0)
    assert dj["found"] and gr["found"]
    # Greedy is lured onto alpha→beta (cheapest first hop) then pays for
    # beta→delta; Dijkstra takes alpha→gamma→delta.
    assert gr["path"] == [ids["alpha"], ids["beta"], ids["delta"]]
    assert dj["path"] == [ids["alpha"], ids["gamma"], ids["delta"]]
    assert dj["cost"] < gr["cost"]
    assert dj["cost"] >= 0.0


def test_dijkstra_deterministic(isolated_qm):
    _, qm = isolated_qm
    ids = _seed_controlled_graph(qm)
    with qm._conn() as cn:
        r1 = qm.dijkstra_route(cn, ids["alpha"], ids["delta"], observer=0.5)
        r2 = qm.dijkstra_route(cn, ids["alpha"], ids["delta"], observer=0.5)
    assert r1["path"] == r2["path"]
    assert r1["cost"] == r2["cost"]


def test_observer_bias_lowers_route_cost(isolated_qm):
    _, qm = isolated_qm
    ids = _seed_controlled_graph(qm)
    with qm._conn() as cn:
        aligned = qm.dijkstra_route(cn, ids["alpha"], ids["delta"], observer=1.0)
        misaligned = qm.dijkstra_route(cn, ids["alpha"], ids["delta"], observer=0.0)
    assert aligned["found"] and misaligned["found"]
    assert aligned["cost"] < misaligned["cost"]


# ---------------------------------------------------------------------------
# Phase B — GLOBAL-MAX objective
# ---------------------------------------------------------------------------
def test_global_max_resource_penalty_monotonic(isolated_qm):
    _, qm = isolated_qm
    base = {
        "retained_information": 0.7, "specialist_consensus": 0.5,
        "route_length": 4, "n_edges": 10,
    }
    small = qm.global_max_score({**base, "token_count": 6}, observer=0.5)
    large = qm.global_max_score({**base, "token_count": 20}, observer=0.5)
    assert math.isfinite(small) and math.isfinite(large)
    assert 0.0 <= large < small <= 1.0


def test_select_minimal_context_fallback_reasons(isolated_qm):
    slm, qm = isolated_qm
    with qm._conn() as cn:
        missing = qm.select_minimal_context(cn, ["zzz_unknown_token"], 0.1, 0.1)
    assert missing["fallback_reason"] == "no_seed_tokens"
    slm.train_round(max_seconds=10.0, max_chunks=50)
    with qm._conn() as cn:
        strict = qm.select_minimal_context(cn, ["carbide"], 0.999, 0.999)
    assert strict["fallback_reason"] == "insufficient_subgraph"


def test_select_minimal_context_connected_and_bounded(isolated_qm):
    _, qm = isolated_qm
    ids = _seed_controlled_graph(qm)
    with qm._conn() as cn:
        sel = qm.select_minimal_context(
            cn, ["alpha", "gamma"], 0.05, 0.05, max_tokens=8
        )
    assert sel["fallback_reason"] is None
    assert set(ids.values()) >= set(sel["nodes"]) or len(sel["nodes"]) <= 8
    assert 0.0 <= sel["sufficiency"] <= 1.0
    assert 0.0 <= sel["confidence"] <= 1.0
    assert 0.0 <= sel["global_max"] <= 1.0
    assert ids["alpha"] in sel["nodes"] and ids["gamma"] in sel["nodes"]


# ---------------------------------------------------------------------------
# Phase D — specialist distillation (orchestrator faked → offline + bounded)
# ---------------------------------------------------------------------------
def test_distill_specialists_requires_multi_support(isolated_qm, monkeypatch):
    _, qm = isolated_qm

    def _fake_orchestration(query, context=None, use_sota=True):
        return {
            "answer": "eoq answer 775 for demand 12000 stock policy",
            "experts_used": ["supply_chain_optimizer", "research_specialist"],
        }

    fake = types.ModuleType("src.quipu.expert_orchestrator")
    fake.run_expert_orchestration = _fake_orchestration
    monkeypatch.setitem(sys.modules, "src.quipu.expert_orchestrator", fake)

    out = qm.distill_specialists("eoq for demand 12000 ordering 50 holding 2")
    assert out["status"] == "ok"
    assert out["support"] == 2
    assert out["pairs"] > 0
    with qm._conn() as cn:
        hist = qm.overlay_summary(cn)["support_histogram"]
    assert hist.get("2", 0) > 0


def test_distill_specialists_single_expert_writes_no_pairs(isolated_qm, monkeypatch):
    _, qm = isolated_qm

    fake = types.ModuleType("src.quipu.expert_orchestrator")
    fake.run_expert_orchestration = lambda q, c=None, use_sota=True: {
        "answer": "answer 42", "experts_used": ["mesh_historian"],
    }
    monkeypatch.setitem(sys.modules, "src.quipu.expert_orchestrator", fake)

    out = qm.distill_specialists("history of the mesh")
    assert out["status"] == "ok"
    assert out["support"] == 1
    assert out["pairs"] == 0  # below _MIN_SPECIALIST_SUPPORT — nothing upserted


def test_distill_specialists_empty_query_skipped(isolated_qm):
    _, qm = isolated_qm
    assert qm.distill_specialists("")["status"] == "skipped"


# ---------------------------------------------------------------------------
# Phase E — minimal_infer flag gating + fallback + happy path
# ---------------------------------------------------------------------------
_CONTRACT_KEYS = {
    "text", "confidence", "selected_pairs", "route", "sufficiency",
    "compression_ratio", "qweight_summary", "resource_reduction",
    "fallback_reason", "minimal",
}


def test_minimal_infer_flag_off_delegates(isolated_qm):
    slm, qm = isolated_qm
    slm.train_round(max_seconds=10.0, max_chunks=50)
    out = qm.minimal_infer("carbide insert assembly")
    assert _CONTRACT_KEYS <= set(out)
    assert out["minimal"] is False
    assert out["fallback_reason"] == "flag_disabled"
    assert "full_context" in out  # delegated to mesh_slm.generate


def test_minimal_infer_insufficient_falls_back(isolated_qm, monkeypatch):
    slm, qm = isolated_qm
    monkeypatch.setenv("SCB_QUIPU_MINIMAL", "1")
    slm.train_round(max_seconds=10.0, max_chunks=50)
    out = qm.minimal_infer("zzz_unknown qqq_nowhere")
    assert out["minimal"] is False
    assert out["fallback_reason"] == "no_seed_tokens"
    with qm._conn() as cn:
        assert qm.overlay_summary(cn)["fallback_count"] >= 1


def test_minimal_infer_happy_path_contract(isolated_qm, monkeypatch):
    _, qm = isolated_qm
    monkeypatch.setenv("SCB_QUIPU_MINIMAL", "1")
    ids = _seed_controlled_graph(qm)
    out = qm.minimal_infer(
        "alpha gamma", sufficiency_threshold=0.05, confidence_threshold=0.05
    )
    assert _CONTRACT_KEYS <= set(out)
    assert out["minimal"] is True
    assert out["fallback_reason"] is None
    assert out["text"]
    assert out["selected_pairs"]
    assert out["compression_ratio"] >= 1.0
    assert 0.0 <= out["resource_reduction"] <= 1.0
    assert 0.0 <= out["sufficiency"] <= 1.0
    assert out["qweight_summary"]["max"] <= 1.0
    if out["route"] is not None and out["route"]["found"]:
        assert out["route"]["path"][0] in ids.values()
    with qm._conn() as cn:
        summary = qm.overlay_summary(cn)
    assert summary["infer_count"] >= 1
    assert summary["route_count"] >= 1


# ---------------------------------------------------------------------------
# Phase F — diagnostics surface
# ---------------------------------------------------------------------------
def test_state_summary_exposes_quipu_minimal(isolated_qm):
    slm, qm = isolated_qm
    with qm._conn():
        pass  # ensure overlay tables exist
    state = slm.state_summary()
    assert "quipu_minimal" in state
    overlay = state["quipu_minimal"]
    assert overlay is not None
    for key in ("pair_count", "route_count", "infer_count",
                "fallback_count", "enabled", "support_histogram"):
        assert key in overlay


# ---------------------------------------------------------------------------
# Offline guarantee — no network imports anywhere in the module
# ---------------------------------------------------------------------------
def test_module_has_no_network_imports(isolated_qm):
    _, qm = isolated_qm
    with open(qm.__file__, encoding="utf-8") as fh:
        source = fh.read()
    for forbidden in ("import requests", "import urllib", "import socket",
                      "import http", "from requests", "from urllib"):
        assert forbidden not in source
