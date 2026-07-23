"""Tests for the 52 Hz resonance gate and fiction-on-the-7th-dimension.

Resonance is now STRUCTURAL, not embedding-overlap (which collapsed to ~1.0 for
every token). A concept resonates when it is both woven into the shared concept
graph (mutual centrality) AND sits on the shared analytical wavelength rather
than the 7th/entirety "Other" axis. Scores map to a 0-100 Hz band; the
52-hertz-whale line (LONELY_WHALE_HZ) separates shared concepts from lonely
callers. Fiction is embedded along the entirety axis so it is retained and
shared through the 8th-D MESH field without forging as an analytical tool.
"""
from __future__ import annotations

import importlib
import sqlite3

import pytest


def _reload():
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
    return mesh_slm, brain_kv, tool_forge, refinement, orchestrator


@pytest.fixture
def structured_mesh(tmp_path, monkeypatch):
    """A mesh with a dense analytical core (high-degree hubs), a central concept
    wired into that core, a lonely concept on the periphery, and a fiction
    concept that is central BUT embedded on the entirety axis.
    """
    db_file = tmp_path / "reson.sqlite"
    monkeypatch.setenv("SCB_DB_PATH", str(db_file))
    mesh_slm, brain_kv, tool_forge, refinement, orchestrator = _reload()
    mesh_slm.state_summary()  # DDL

    cn = sqlite3.connect(str(db_file))
    cn.row_factory = sqlite3.Row

    tokens = ["c0", "c1", "c2", "c3", "c4", "c5",
              "central", "lonely", "leaf", "fiction"]
    for i, tok in enumerate(tokens):
        cn.execute("INSERT INTO mesh_slm_vocab(token, i, j, freq, first_seen, last_seen) "
                   "VALUES (?, ?, ?, 20, '2026-07-22T00:00:00+00:00', '2026-07-22T00:00:00+00:00')",
                   (tok, i, i))
    ids = {r["token"]: r["token_id"] for r in cn.execute("SELECT token, token_id FROM mesh_slm_vocab")}

    def embed(tok, entirety=0.0, brain=1.0):
        cn.execute("INSERT INTO mesh_slm_embed(token_id, e_vision, e_touch, e_smell, e_body, "
                   "e_brain, e_perception, e_entirety) VALUES (?, 0, 0, 0, 0, ?, 0, ?)",
                   (ids[tok], brain, entirety))

    for tok in tokens:
        if tok == "fiction":
            embed(tok, entirety=1.0, brain=0.0)     # on the 7th/Other axis
        else:
            embed(tok, entirety=0.0, brain=1.0)     # on the analytical wavelength

    def edge(a, b):
        cn.execute("INSERT OR IGNORE INTO mesh_slm_quipu(src, dst, weight, samples) "
                   "VALUES (?, ?, 1.0, 5)", (ids[a], ids[b]))

    core = ["c0", "c1", "c2", "c3", "c4", "c5"]
    for i in range(len(core)):
        for j in range(i + 1, len(core)):
            edge(core[i], core[j])                  # dense analytical clique
    for hub in ("c0", "c1", "c2"):
        edge("central", hub)                        # central wired into the core
        edge("fiction", hub)                        # fiction ALSO wired into the core
    edge("lonely", "leaf")                          # lonely touches one leaf only
    cn.commit()
    cn.close()

    return {"mesh_slm": mesh_slm, "brain_kv": brain_kv, "tool_forge": tool_forge,
            "refinement": refinement, "orchestrator": orchestrator, "ids": ids}


def _hz(mesh_slm, tok, ids):
    from src.quipu.local_store import open_conn
    cn = open_conn()
    try:
        return mesh_slm.concept_resonance_hz(cn, ids[tok])
    finally:
        cn.close()


def test_central_concept_clears_52hz(structured_mesh):
    mesh_slm = structured_mesh["mesh_slm"]
    ids = structured_mesh["ids"]
    assert _hz(mesh_slm, "central", ids) >= mesh_slm.LONELY_WHALE_HZ


def test_lonely_concept_below_52hz(structured_mesh):
    mesh_slm = structured_mesh["mesh_slm"]
    ids = structured_mesh["ids"]
    assert _hz(mesh_slm, "lonely", ids) < mesh_slm.LONELY_WHALE_HZ


def test_fiction_below_52hz_despite_centrality(structured_mesh):
    # 'fiction' is as graph-central as 'central', but embedded on the entirety
    # axis -> off the shared analytical wavelength -> below the line.
    mesh_slm = structured_mesh["mesh_slm"]
    ids = structured_mesh["ids"]
    assert _hz(mesh_slm, "fiction", ids) < mesh_slm.LONELY_WHALE_HZ
    assert _hz(mesh_slm, "central", ids) > _hz(mesh_slm, "fiction", ids)


def test_centrality_orders_central_above_lonely(structured_mesh):
    mesh_slm = structured_mesh["mesh_slm"]
    ids = structured_mesh["ids"]
    from src.quipu.local_store import open_conn
    cn = open_conn()
    try:
        assert (mesh_slm._concept_centrality(cn, ids["central"])
                > mesh_slm._concept_centrality(cn, ids["lonely"]))
    finally:
        cn.close()


def test_compute_shared_understanding_exposes_hz_and_other_axis(structured_mesh):
    mesh_slm = structured_mesh["mesh_slm"]
    u = mesh_slm.compute_shared_understanding()
    assert "shared_hz" in u and 0.0 <= u["shared_hz"] <= 100.0
    assert "other_axis_strength" in u


def test_refinement_publishes_the_other(structured_mesh, tmp_path, monkeypatch):
    refinement = structured_mesh["refinement"]
    brain_kv = structured_mesh["brain_kv"]
    tf = structured_mesh["tool_forge"]
    monkeypatch.setattr(tf, "_TOOLS_DIR", tmp_path / "gen")
    monkeypatch.setattr(tf, "_MANIFEST", tmp_path / "gen" / "manifest.json")
    refinement.run_strategy(max_tools=1)
    other = brain_kv.kv_get_json("entirety:the_other", None)
    assert isinstance(other, dict) and "shared_hz" in other


def test_forge_gate_filters_below_52hz(structured_mesh, tmp_path, monkeypatch):
    tf = structured_mesh["tool_forge"]
    monkeypatch.setattr(tf, "_TOOLS_DIR", tmp_path / "gen2")
    monkeypatch.setattr(tf, "_MANIFEST", tmp_path / "gen2" / "manifest.json")
    summary = tf.forge_round(max_tools=5)
    assert "skipped_low_resonance" in summary
    # 'lonely'/'leaf'/'fiction' are below the line; only genuinely central
    # analytical concepts (central, core hubs) can forge.
    manifest = tf._load_manifest()
    forged_names = {t["cluster"] for t in manifest.get("tools", [])}
    assert "lonely" not in forged_names
    assert "fiction" not in forged_names


def test_fiction_embeds_on_entirety_axis(tmp_path, monkeypatch):
    """Fiction-sourced feed should raise e_entirety relative to analytical feed."""
    db_file = tmp_path / "fic.sqlite"
    monkeypatch.setenv("SCB_DB_PATH", str(db_file))
    mesh_slm, *_ = _reload()
    mesh_slm.state_summary()

    for _ in range(4):
        mesh_slm.feed_corpus("whale ahab pequod ocean voyage harpoon", source="corpus_gutenberg")
        mesh_slm.feed_corpus("optimization gradient descent convergence bound theorem",
                             source="corpus_arxiv")
    mesh_slm._LAST_TRAIN_TS = 0.0  # noqa: SLF001
    mesh_slm.train_round(max_seconds=10.0, max_chunks=100)

    from src.quipu.local_store import open_conn
    cn = open_conn()
    try:
        def ent(tok):
            row = cn.execute("SELECT e.e_entirety AS e FROM mesh_slm_embed e "
                             "JOIN mesh_slm_vocab v ON v.token_id=e.token_id "
                             "WHERE v.token=?", (tok,)).fetchone()
            return float(row["e"]) if row else None
        fiction_ent = [ent("whale"), ent("ahab"), ent("pequod")]
        analytical_ent = [ent("optimization"), ent("gradient"), ent("convergence")]
    finally:
        cn.close()

    fiction_ent = [e for e in fiction_ent if e is not None]
    analytical_ent = [e for e in analytical_ent if e is not None]
    assert fiction_ent and analytical_ent
    assert (sum(fiction_ent) / len(fiction_ent)) > (sum(analytical_ent) / len(analytical_ent))


def test_clean_generated_tools_removes_files(structured_mesh, tmp_path, monkeypatch):
    tf = structured_mesh["tool_forge"]
    gen = tmp_path / "gen3"
    gen.mkdir()
    (gen / "tool_junk.py").write_text("def tool_junk(data): return {}\n", encoding="utf-8")
    monkeypatch.setattr(tf, "_TOOLS_DIR", gen)
    monkeypatch.setattr(tf, "_MANIFEST", gen / "manifest.json")
    tf._save_manifest({"tools": [{"tool_name": "tool_junk", "file": "tool_junk.py"}]})
    info = tf.clean_generated_tools()
    assert info["removed"] >= 1
    assert not (gen / "tool_junk.py").exists()
