"""End-to-end test of the repaired learning chain: feed_corpus -> train_round
-> quipu edges -> tool_forge clusters.

This is the regression guard for the bug the daily digest surfaced: 15k
documents produced 0 quipu edges and 0 training rounds because train_round's
_corpus_stream only read Ring 4 app tables (corpus_entity / corpus_edge /
llm_dispatch_log) that don't exist in QUIPU, so it always bailed "no_corpus".
The fix wires a QUIPU-native mesh_corpus_feed that feed_corpus() writes and
_corpus_stream() reads.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def fresh_mesh(tmp_path, monkeypatch):
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "chain.sqlite"))
    import src.quipu.local_store as ls
    importlib.reload(ls)
    import src.quipu.mesh_slm as mesh_slm
    importlib.reload(mesh_slm)
    return mesh_slm


# Concept-dense sentences (already stopword-light) with strong repeated
# co-occurrence so bigram edges form deterministically.
_FEED = [
    "holographic entropy bound compression mesh corpus tensor",
    "holographic entropy bound bekenstein hawking radiation remnant",
    "mesh corpus tensor compression weyl scalar boundary encoding",
    "agent coalition emergent swarm signal mesh interaction matrix",
    "agent coalition emergent specialist resonance novelty ceiling",
    "weyl scalar tensor compression mesh corpus holographic boundary",
] * 4


def test_feed_corpus_then_train_builds_edges(fresh_mesh):
    mesh_slm = fresh_mesh

    # Baseline: no corpus -> train_round bails, no edges.
    empty = mesh_slm.train_round(max_seconds=5.0, max_chunks=50)
    assert empty.get("status") == "no_corpus"

    # Feed concept-dense text, then train.
    for line in _FEED:
        mesh_slm.feed_corpus(line, source="test")

    # train_round has a 5s min-gap rate limit; the first real call after the
    # bailed one may be rate-limited, so drive the internal timestamp back.
    mesh_slm._LAST_TRAIN_TS = 0.0  # noqa: SLF001 - test reaches into module state
    result = mesh_slm.train_round(max_seconds=10.0, max_chunks=100)

    assert result.get("status") not in ("no_corpus", "skipped"), result
    state = mesh_slm.state_summary()
    assert state["vocab_size"] > 0
    assert state["quipu_edges"] > 0, "train_round must build quipu edges from the feed"
    assert state["rounds"] >= 1


def test_vocab_excludes_stopwords_when_fed_concept_text(fresh_mesh):
    mesh_slm = fresh_mesh
    for line in _FEED:
        mesh_slm.feed_corpus(line, source="test")
    mesh_slm._LAST_TRAIN_TS = 0.0  # noqa: SLF001
    mesh_slm.train_round(max_seconds=10.0, max_chunks=100)

    from src.quipu.local_store import open_conn
    cn = open_conn()
    try:
        tokens = {r["token"] for r in cn.execute("SELECT token FROM mesh_slm_vocab")}
    finally:
        cn.close()
    # Concepts present, function words absent.
    assert "holographic" in tokens
    assert "mesh" in tokens
    assert "the" not in tokens
    assert "of" not in tokens


def test_reset_mesh_learning_clears_state(fresh_mesh):
    mesh_slm = fresh_mesh
    for line in _FEED:
        mesh_slm.feed_corpus(line, source="test")
    mesh_slm._LAST_TRAIN_TS = 0.0  # noqa: SLF001
    mesh_slm.train_round(max_seconds=10.0, max_chunks=100)
    assert mesh_slm.state_summary()["quipu_edges"] > 0

    info = mesh_slm.reset_mesh_learning()
    assert info["status"] == "reset"
    state = mesh_slm.state_summary()
    assert state["vocab_size"] == 0
    assert state["quipu_edges"] == 0
    assert (state["rounds"] or 0) == 0


def test_tool_forge_finds_clusters_after_training(fresh_mesh, tmp_path, monkeypatch):
    mesh_slm = fresh_mesh
    import src.quipu.brain_kv as brain_kv
    importlib.reload(brain_kv)
    import src.quipu.tool_forge as tool_forge
    importlib.reload(tool_forge)
    monkeypatch.setattr(tool_forge, "_TOOLS_DIR", tmp_path / "gen")
    monkeypatch.setattr(tool_forge, "_MANIFEST", tmp_path / "gen" / "manifest.json")

    # Feed heavily so tokens accrue degree > 1 in the quipu graph.
    for line in _FEED * 3:
        mesh_slm.feed_corpus(line, source="test")
    mesh_slm._LAST_TRAIN_TS = 0.0  # noqa: SLF001
    mesh_slm.train_round(max_seconds=15.0, max_chunks=200)

    clusters = tool_forge._scan_vocab_clusters(limit=10)
    # With repeated co-occurrence the concept tokens should now have neighbours
    # in mesh_slm_quipu, so the scan is no longer empty (the digest's
    # "no clusters eligible" symptom is resolved).
    assert clusters, "tool_forge should find clusters once edges exist"
    assert clusters[0]["degree"] >= 1
