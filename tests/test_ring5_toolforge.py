"""Tests for Ring 5 (tool_forge / expert_orchestrator / systemic_refinement_agent).

Ring 5 in QUIPU is a clean-room, mesh-native adaptation of the parent Supply
Chain Brain's Refinement -> Toolforge ring: it scans QUIPU's own corpus
tables (mesh_slm_vocab / mesh_slm_quipu) instead of the parent's ERP-linked
corpus_entity / corpus_edge, and only the digital auto-implementation path is
carried over (no physical/ERP HITL — that infra isn't part of QUIPU). These
tests seed an isolated mesh DB directly so forge_round has real degree/freq
signal to score, independent of the crawler.
"""
from __future__ import annotations

import importlib
import sqlite3

import pytest


@pytest.fixture
def isolated_mesh(tmp_path, monkeypatch):
    """Fresh mesh DB with a handful of highly-connected vocab tokens so
    tool_forge's cluster scan has real degree/freq/neighbour signal.
    """
    db_file = tmp_path / "ring5.sqlite"
    monkeypatch.setenv("SCB_DB_PATH", str(db_file))

    import src.quipu.local_store as ls
    importlib.reload(ls)
    import src.quipu.mesh_slm as mesh_slm
    importlib.reload(mesh_slm)
    import src.quipu.brain_kv as brain_kv
    importlib.reload(brain_kv)
    import src.quipu.tool_forge as tool_forge
    importlib.reload(tool_forge)
    import src.quipu.systemic_refinement_agent as refinement
    importlib.reload(refinement)
    import src.quipu.expert_orchestrator as orchestrator
    importlib.reload(orchestrator)

    # Trigger mesh_slm's own DDL (creates mesh_slm_vocab / mesh_slm_quipu).
    mesh_slm.state_summary()

    cn = sqlite3.connect(str(db_file))
    cn.row_factory = sqlite3.Row
    tokens = [
        "holographic", "entropy", "bound", "mesh", "compression",
        "agent", "coalition", "emergent", "swarm", "signal",
    ]
    now = "2026-07-20T12:00:00+00:00"
    for i, tok in enumerate(tokens):
        cn.execute(
            "INSERT INTO mesh_slm_vocab(token, i, j, freq, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tok, i, i, 25, now, now),
        )
    cn.commit()
    ids = [r["token_id"] for r in cn.execute("SELECT token_id FROM mesh_slm_vocab ORDER BY token_id")]
    # Fully connect the first 6 tokens to each other -> high degree cluster.
    for a in range(6):
        for b in range(a + 1, 6):
            cn.execute(
                "INSERT OR REPLACE INTO mesh_slm_quipu(src, dst, weight, samples) VALUES (?, ?, ?, ?)",
                (ids[a], ids[b], 1.0, 5),
            )
    cn.commit()
    cn.close()

    return {
        "mesh_slm": mesh_slm, "tool_forge": tool_forge,
        "refinement": refinement, "orchestrator": orchestrator,
        "brain_kv": brain_kv,
    }


def test_scan_vocab_clusters_finds_connected_tokens(isolated_mesh):
    tf = isolated_mesh["tool_forge"]
    clusters = tf._scan_vocab_clusters(limit=10)
    assert clusters, "expected at least one high-degree cluster"
    top = clusters[0]
    assert top["degree"] >= 4
    assert isinstance(top["neighbors"], list)


def test_validate_code_accepts_generated_templates(isolated_mesh):
    tf = isolated_mesh["tool_forge"]
    code = tf._tpl_entropy_bounds("test_tool")
    ok, msg = tf._validate_code(code)
    assert ok, msg


def test_validate_code_rejects_unsafe_import():
    from src.quipu import tool_forge as tf
    bad = "import os\ndef tool_bad(data: dict) -> dict:\n    return {'result': os.getcwd()}\n"
    ok, msg = tf._validate_code(bad)
    assert not ok
    assert "Unsafe" in msg


def test_novelty_certainty_scoring_bounds(isolated_mesh):
    tf = isolated_mesh["tool_forge"]
    cluster = {
        "label": "holographic",
        "degree": 6,
        "freq": 25,
        "neighbors": [{"label": "entropy"}, {"label": "bound"}, {"label": "mesh"}],
    }
    novelty, certainty = tf._novelty_certainty_scores(cluster, existing_tool_labels=set())
    assert 0.0 <= novelty <= 1.0
    assert 0.0 <= certainty <= 1.0


def test_forge_round_synthesizes_and_persists_manifest(isolated_mesh, tmp_path, monkeypatch):
    tf = isolated_mesh["tool_forge"]
    # Redirect generated-tool output into the tmp dir so the test doesn't
    # write into the real package tree.
    monkeypatch.setattr(tf, "_TOOLS_DIR", tmp_path / "generated")
    monkeypatch.setattr(tf, "_MANIFEST", tmp_path / "generated" / "manifest.json")

    summary = tf.forge_round(max_tools=2)
    assert summary["candidates"] >= 1
    # With a fully-connected 6-token cluster and freq=25, novelty*certainty
    # should clear AUTO_IMPL_THRESHOLD for at least one token.
    if summary["tools_forged"] > 0:
        manifest = tf._load_manifest()
        assert manifest["tools"]
        entry = manifest["tools"][0]
        assert (tmp_path / "generated" / entry["file"]).exists()
        loaded = tf.load_generated_tools()
        assert entry["tool_name"] in loaded
        result = loaded[entry["tool_name"]]({})
        assert "confidence" in result


def test_systemic_refinement_run_strategy_records_cycle(isolated_mesh, tmp_path, monkeypatch):
    tf = isolated_mesh["tool_forge"]
    refinement = isolated_mesh["refinement"]
    monkeypatch.setattr(tf, "_TOOLS_DIR", tmp_path / "generated2")
    monkeypatch.setattr(tf, "_MANIFEST", tmp_path / "generated2" / "manifest.json")

    entry = refinement.run_strategy(max_tools=1)
    assert "actions" in entry
    assert entry["actions"]  # never empty - "no_action" is the floor value

    last = refinement.last_cycle()
    assert last.get("ts") == entry["ts"]
    hist = refinement.history()
    assert isinstance(hist, list) and hist


def test_expert_orchestrator_dispatch_degrades_gracefully(isolated_mesh, tmp_path, monkeypatch):
    orchestrator = isolated_mesh["orchestrator"]
    tf = isolated_mesh["tool_forge"]
    monkeypatch.setattr(tf, "_TOOLS_DIR", tmp_path / "generated3")
    monkeypatch.setattr(tf, "_MANIFEST", tmp_path / "generated3" / "manifest.json")

    # Vocab is tiny (10 tokens) so mesh_slm.generate will likely report
    # low/zero confidence - dispatch must still return a well-formed dict.
    result = orchestrator.dispatch("holographic mesh entropy", trigger_forge=False)
    assert result["status"] in ("ok", "error", "skipped")
    if result["status"] == "ok":
        assert "confidence" in result and "specialist" in result


def test_expert_orchestrator_empty_query_is_skipped(isolated_mesh):
    orchestrator = isolated_mesh["orchestrator"]
    result = orchestrator.dispatch("   ")
    assert result["status"] == "skipped"
