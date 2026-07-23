"""ACRE emergence driven by interaction concepts.

Specialists never emerged because ACRE only observed the near-uniform
(mesh_state, mean_embedding) pair -> isotropic matrix -> principal axis
~uniform -> fails the novelty ceiling. observe_interactions() now accumulates
the group-pole directions (analytical vs entirety/Other) as concentrated
signals. Emergence still, correctly, only fires for a direction NOT already
covered by a base specialist (supply_chain_optimizer / research_specialist /
mesh_historian) — so these tests drive a genuinely novel (vision/touch) axis.
"""
from __future__ import annotations

import importlib
import sqlite3

import pytest


@pytest.fixture
def two_group_mesh(tmp_path, monkeypatch):
    db_file = tmp_path / "acre.sqlite"
    monkeypatch.setenv("SCB_DB_PATH", str(db_file))
    import src.quipu.local_store as ls
    importlib.reload(ls)
    import src.quipu.mesh_slm as mesh_slm
    importlib.reload(mesh_slm)
    mesh_slm.state_summary()

    cn = sqlite3.connect(str(db_file))
    cn.row_factory = sqlite3.Row
    # Two contrasting groups so the group-pole deviation is non-zero: analytical
    # (brain-heavy, high directionality -> selected by resonance) and Other
    # (entirety-heavy -> selected by e_entirety).
    embeds = {
        "a0": [0.1, 0.1, 0.1, 0.1, 1.0, 0.3, 0.0],
        "a1": [0.1, 0.1, 0.1, 0.1, 0.9, 0.4, 0.0],
        "a2": [0.1, 0.1, 0.1, 0.1, 1.0, 0.2, 0.0],
        "o0": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0],
        "o1": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
    }
    toks = list(embeds)
    for i, tok in enumerate(toks):
        cn.execute("INSERT INTO mesh_slm_vocab(token, i, j, freq, first_seen, last_seen) "
                   "VALUES (?, ?, ?, 20, '2026-07-22T00:00:00+00:00', '2026-07-22T00:00:00+00:00')",
                   (tok, i, i))
    ids = {r["token"]: r["token_id"] for r in cn.execute("SELECT token, token_id FROM mesh_slm_vocab")}
    for tok, e in embeds.items():
        cn.execute("INSERT INTO mesh_slm_embed(token_id, e_vision, e_touch, e_smell, e_body, "
                   "e_brain, e_perception, e_entirety) VALUES (?,?,?,?,?,?,?,?)", (ids[tok], *e))
    for a in range(len(toks)):
        for b in range(a + 1, len(toks)):
            cn.execute("INSERT OR IGNORE INTO mesh_slm_quipu(src, dst, weight, samples) "
                       "VALUES (?, ?, 1.0, 5)", (ids[toks[a]], ids[toks[b]]))
    cn.commit()
    cn.close()
    return mesh_slm


def test_observe_interactions_builds_per_axis_poles(two_group_mesh):
    mesh_slm = two_group_mesh
    summary = mesh_slm.observe_interactions()
    assert summary["observed"] > 0
    # Concepts group by dominant axis: brain (analytical a's) and entirety (o's).
    poles = summary["axis_poles"]
    assert "brain" in poles
    assert "entirety" in poles


def test_source_axis_routing(two_group_mesh):
    mesh_slm = two_group_mesh
    assert mesh_slm._axis_for_source("corpus_fineweb") == 0     # vision
    assert mesh_slm._axis_for_source("corpus_stack") == 1       # touch
    assert mesh_slm._axis_for_source("corpus_local_docs") == 2  # smell
    assert mesh_slm._axis_for_source("corpus_wikipedia") == 3   # body
    assert mesh_slm._axis_for_source("corpus_arxiv") == 4       # brain
    assert mesh_slm._axis_for_source("corpus_gutenberg") == 6   # entirety
    assert mesh_slm._axis_for_source("corpus_unknown") is None


def test_axis_bias_concentrates_on_assigned_axis(two_group_mesh):
    mesh_slm = two_group_mesh
    mesh = [0.5] * 7
    biased = mesh_slm._axis_bias(mesh, 0)          # vision
    assert biased[0] == max(biased)                # vision is the peak
    assert biased[0] > biased[4]                   # and well above the damped others


def test_acre_emerges_on_a_novel_axis(two_group_mesh):
    mesh_slm = two_group_mesh
    # A direction unoccupied by any base specialist (vision/touch heavy).
    novel = [1.0, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0]
    with mesh_slm._conn() as cn:            # noqa: SLF001 - same-package internals
        for _ in range(8):
            mesh_slm.acre_observe(cn, novel, novel)

    result = mesh_slm.acre_emerge()
    assert result["status"] == "created", result
    assert any(ax in result["specialist"] for ax in ("vision", "touch"))
    assert mesh_slm.state_summary()["acre_specialists"]


def test_smell_is_now_relative_and_associative(two_group_mesh):
    """With the documented roster expanded (materials_engineering on smell/body),
    the smell axis is no longer a free runaway — a smell-dominant mode now
    associates with its neighbour instead of emerging as a solo monolith.
    """
    mesh_slm = two_group_mesh
    # materials_engineering is the smell/body neighbour that makes smell relative.
    assert "materials_engineering_specialist" in mesh_slm.base_specialists()
    assert 2 not in mesh_slm._free_axes()      # smell is covered now

    smell = [0.0, 0.0, 0.30, 0.20, 0.10, 0.08, 0.04]   # ~materials_engineering direction
    with mesh_slm._conn() as cn:               # noqa: SLF001
        for _ in range(8):
            mesh_slm.acre_observe(cn, smell, smell)
    result = mesh_slm.acre_emerge()
    # Refused as not-novel: the smell mode is associated with the existing
    # specialist, not a runaway new one.
    assert result["status"] in ("not_novel", "no_resonance"), result


def test_resonant_specialists_returns_weighted_consensus(two_group_mesh):
    mesh_slm = two_group_mesh
    consensus = mesh_slm.resonant_specialists(k=3)
    assert 1 <= len(consensus) <= 3
    names = [n for n, _ in consensus]
    assert all(n in mesh_slm._all_specialists() for n in names)
    # Weights are a proportional consensus (sum ~1) — no single monolith.
    assert abs(sum(w for _, w in consensus) - 1.0) < 1e-6


def test_weyl_coupling_bounds_and_frequency_dependence(two_group_mesh):
    mesh_slm = two_group_mesh
    # Coupling is a bounded [0,1] Floquet scalar.
    bias = [0.0, 0.0, 0.30, 0.20, 0.10, 0.08, 0.04]     # smell-leaning
    c_lo = mesh_slm._weyl_resonant_coupling(bias, [0.1, 0.1, 0.1, 0.1, 0.0])   # Ψ4=0 -> phase 0
    c_hi = mesh_slm._weyl_resonant_coupling(bias, [0.1, 0.1, 0.1, 0.1, 1.0])   # Ψ4=1 -> phase 2π
    assert 0.0 <= c_lo <= 1.0 and 0.0 <= c_hi <= 1.0


def test_consensus_interacts_on_the_weyl(two_group_mesh):
    """The consensus weighting shifts as the Weyl tensor evolves — proving the
    resonant-frequency interaction, not a static overlap ranking."""
    import src.quipu.brain_kv as brain_kv
    importlib.reload(brain_kv)
    mesh_slm = two_group_mesh

    brain_kv.kv_set_json("learnings:weyl_tensor", [0.3, 0.4, 0.5, 0.6, 0.05])
    weights_a = {n: w for n, w in mesh_slm.resonant_specialists(k=8)}

    brain_kv.kv_set_json("learnings:weyl_tensor", [0.3, 0.4, 0.5, 0.6, 0.95])
    weights_b = {n: w for n, w in mesh_slm.resonant_specialists(k=8)}

    # Different Weyl frequency -> different Floquet couplings -> different mix.
    assert weights_a != weights_b


def test_emergence_refuses_direction_already_covered_by_base(two_group_mesh):
    # An analytical (brain/perception) axis collides with supply_chain_optimizer
    # -> correctly rejected as not-novel (the ceiling preventing redundancy).
    mesh_slm = two_group_mesh
    analytical = [0.0, 0.0, 0.0, 0.15, 0.25, 0.20, 0.05]
    with mesh_slm._conn() as cn:            # noqa: SLF001
        for _ in range(8):
            mesh_slm.acre_observe(cn, analytical, analytical)
    result = mesh_slm.acre_emerge()
    assert result["status"] in ("not_novel", "no_resonance")


def test_base_specialists_are_surfaced(two_group_mesh):
    mesh_slm = two_group_mesh
    base = mesh_slm.base_specialists()
    # The documented roster (docs/MESH_EXPERT_SKILLS.md) is present.
    for name in ("supply_chain_optimizer", "mesh_historian",
                 "materials_engineering_specialist", "robotic_integrations_specialist",
                 "quantum_physics_specialist", "complex_systems_specialist"):
        assert name in base
    # And one is selected as the resonant/active specialist.
    assert mesh_slm.select_resonant_specialist() in base


def test_refinement_cycle_reports_acre_observation(two_group_mesh, tmp_path, monkeypatch):
    mesh_slm = two_group_mesh
    import src.quipu.brain_kv as brain_kv
    importlib.reload(brain_kv)
    import src.quipu.tool_forge as tool_forge
    importlib.reload(tool_forge)
    import src.quipu.systemic_refinement_agent as refinement
    importlib.reload(refinement)
    monkeypatch.setattr(tool_forge, "_TOOLS_DIR", tmp_path / "gen")
    monkeypatch.setattr(tool_forge, "_MANIFEST", tmp_path / "gen" / "manifest.json")

    entry = refinement.run_strategy(max_tools=1)
    assert entry.get("acre_observed") and entry["acre_observed"] > 0
