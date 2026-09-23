"""qpsi.coherency_depth — the lattice foliated: fibres per token, coherency
between planes, vertical descent, bottom-up accretion, top-down anchoring,
and the write boundary (only mesh_plane_* and entirety:planes:*)."""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

from src.quipu.qpsi import coherency_depth as cd
from src.quipu.qpsi.governance import GovernanceConfig

SENSES6 = ["vision", "touch", "smell", "body", "brain", "perception"]
TOKENS = {
    1: ("inventory", [0.10, 0.12, 0.05, 0.08, 0.60, 0.10, 0.05]),
    2: ("warehouse", [0.30, 0.40, 0.05, 0.30, 0.20, 0.10, 0.02]),
    3: ("part",      [0.25, 0.55, 0.05, 0.10, 0.10, 0.10, 0.02]),
    4: ("flow",      [0.05, 0.05, 0.05, 0.05, 0.90, 0.05, 0.10]),
    5: ("lonely",    [0.2] * 7),
}
EDGES = [(1, 2, 0.8), (1, 3, 0.5), (1, 4, 0.9), (2, 1, 0.6), (2, 3, 0.7), (3, 1, 0.2), (4, 1, 0.9), (4, 2, 0.3)]
SPECS = {"emergent_smell_brain": [0.0469, 0.0462, 0.25, 0.0466, 0.0556, 0.047, 0.0],
         "emergent_touch_perception": [0.025, 0.25, 0.0, 0.0999, 0.0, 0.2385, 0.0]}


@pytest.fixture
def mesh():
    cn = sqlite3.connect(":memory:")
    cn.executescript("""
    CREATE TABLE mesh_slm_vocab(token_id INTEGER PRIMARY KEY, token TEXT UNIQUE NOT NULL, i INTEGER, j INTEGER, freq INTEGER DEFAULT 1);
    CREATE TABLE mesh_slm_embed(token_id INTEGER PRIMARY KEY, e_vision REAL, e_touch REAL, e_smell REAL, e_body REAL, e_brain REAL, e_perception REAL, e_entirety REAL);
    CREATE TABLE mesh_slm_quipu(src INTEGER, dst INTEGER, weight REAL, samples INTEGER DEFAULT 1, PRIMARY KEY(src, dst));
    CREATE TABLE mesh_slm_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
    """)
    for tid, (tok, vec) in TOKENS.items():
        cn.execute("INSERT INTO mesh_slm_vocab VALUES(?,?,?,?,1)", (tid, tok, tid, tid))
        cn.execute("INSERT INTO mesh_slm_embed VALUES(?,?,?,?,?,?,?,?)", (tid, *vec))
    for s, d, w in EDGES:
        cn.execute("INSERT INTO mesh_slm_quipu VALUES(?,?,?,1)", (s, d, w))
    cn.execute("INSERT INTO mesh_slm_meta VALUES('acre_specialists', ?)", (json.dumps(SPECS),))
    cn.execute("INSERT INTO brain_kv VALUES('entirety:conscious_emergence', ?, 't')",
               (json.dumps({"phases": {s: 0.4 for s in SENSES6}, "coherence": 0.8}),))
    yield cn
    cn.close()


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

def test_fidelity_is_the_born_rule_and_kl_halts_on_equality():
    assert cd.fidelity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cd.fidelity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert cd.fidelity([1, 1, 0], [1, 0, 0]) == pytest.approx(0.5)              # cos² 45°
    assert cd.fidelity([0, 0], [1, 1]) == 0.0
    assert cd.kl_divergence([0.2, 0.8], [0.2, 0.8]) == pytest.approx(0.0, abs=1e-9)
    assert cd.kl_divergence([0.9, 0.1], [0.1, 0.9]) > 1.0
    assert cd.floquet_factor(0.0, 1.0) == pytest.approx(1.0)                   # J₀(0) = 1
    assert abs(cd.floquet_factor(2.4048, 1.0)) < 0.01                           # first Bessel zero: dynamic localisation


def test_relational_plane_is_the_weighted_mean_of_out_neighbours(mesh):
    rel = cd.relational_planes(mesh)
    assert set(rel) == {1, 2, 3, 4}                                             # "lonely" has no out-edges
    t1, w = rel[1]
    assert w == pytest.approx(0.8 + 0.5 + 0.9)
    expected_brain = (0.8 * 0.20 + 0.5 * 0.10 + 0.9 * 0.90) / 2.2
    assert t1[4] == pytest.approx(expected_brain)
    assert cd.relational_planes(mesh, token_ids=[5]) == {}
    assert set(cd.relational_planes(mesh, token_ids=[1, 5])) == {1}


def test_physical_plane_passes_the_three_physical_gates_or_holds():
    gov = GovernanceConfig()
    t0 = [0.10, 0.12, 0.05, 0.08, 0.60, 0.10, 0.05]
    t1 = [0.20, 0.12, 0.05, 0.08, 0.60, 0.10, 0.05]                              # a positive vision displacement
    phases = {s: 0.4 for s in SENSES6}
    t2, info = cd.physical_plane(t0, t1, phases, gov)
    assert t2 is not None and info["held_at"] is None and info["events"] >= 1
    assert len(t2) == 7 and t2[6] == pytest.approx(0.05)                        # entirety carried through
    # no displacement at all → held at gate 1
    none, info = cd.physical_plane(t0, list(t0), phases, gov)
    assert none is None and info["held_at"] == "displacement"
    # no phase source → SiCi holds (φ = 0), exactly as it holds the Entirety
    none, info = cd.physical_plane(t0, t1, {s: 0.0 for s in SENSES6}, gov)
    assert none is None and info["held_at"] == "sici"


def test_crystal_planes_need_the_floquet_lock(mesh):
    specs = cd.specialists(mesh)
    assert set(specs) == set(SPECS)
    t0 = [0.2] * 7
    locked = cd.crystal_planes(t0, specs, floquet=1.0, lock_min=0.3, max_planes=8)
    assert locked and locked[0][2] >= locked[-1][2]                             # best lock first
    assert cd.crystal_planes(t0, specs, floquet=0.0, lock_min=0.3, max_planes=8) == []   # at a Bessel zero nothing locks
    assert cd.crystal_planes(t0, specs, floquet=1.0, lock_min=0.3, max_planes=1) and \
        len(cd.crystal_planes(t0, specs, floquet=1.0, lock_min=0.3, max_planes=1)) == 1
    assert cd.crystal_planes(t0, specs, floquet=1.0, lock_min=1.01, max_planes=8) == []


def test_descent_halts_on_absence_or_convergence():
    p0 = [0.1, 0.1, 0.1, 0.1, 0.6, 0.1, 0.1]
    assert cd.descend({0: p0}, kl_epsilon=1e-3)["halted_by"] == "no_plane"
    assert cd.descend({0: p0, 1: [0.3] * 7}, kl_epsilon=1e-3) == {
        **cd.descend({0: p0, 1: [0.3] * 7}, kl_epsilon=1e-3), "depth": 1, "halted_by": "no_plane"}
    d = cd.descend({0: p0, 1: list(p0), 2: [0.5] * 7}, kl_epsilon=1e-3)
    assert d["depth"] == 1 and d["halted_by"] == "converged"                    # plane 1 adds nothing: stop there
    d = cd.descend({0: p0, 1: [0.3] * 7, 2: [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 3: [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]},
                   kl_epsilon=1e-3)
    assert d["depth"] == 3 and d["halted_by"] == "no_plane" and len(d["trace"]) == 3
    # two planes that are the same distribution (uniform 0.3 and uniform 0.5) converge: nothing deeper is reached
    d = cd.descend({0: p0, 1: [0.3] * 7, 2: [0.5] * 7, 3: [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]}, kl_epsilon=1e-3)
    assert d["depth"] == 2 and d["halted_by"] == "converged"
    assert cd.descend({1: p0}, kl_epsilon=1e-3)["halted_by"] == "no_surface"
    assert cd.descend({0: p0, 2: p0}, kl_epsilon=1e-3)["depth"] == 0           # the corridor needs plane 1


# ---------------------------------------------------------------------------
# Accretion and the fibre
# ---------------------------------------------------------------------------

def test_accretion_builds_fibres_coherency_and_depth(mesh):
    s = cd.accrete(mesh, cfg=cd.CoherencyConfig(lock_min=0.3), gov=GovernanceConfig(), now=100.0)
    assert s["tokens"] == 5 and s["plane1"] == 4 and s["K"] == 5 and s["capacity"] == 25
    assert s["plane2"] + sum(s["plane2_held"].values()) == 4
    assert s["crystal_tokens"] >= 1 and s["phases_present"] and s["floquet"] == pytest.approx(1.0)
    fib = cd.fibre(mesh, 1)
    assert 0 in fib and 1 in fib and len(fib[1]) == 7
    assert cd.plane(mesh, 1, 0) == TOKENS[1][1]                                 # plane 0 is mesh_slm_embed, untouched
    c01 = cd.coherency(mesh, 1, 0, 1)
    assert c01 is not None and 0.0 <= c01 <= 1.0 and cd.coherency(mesh, 1, 1, 0) == c01
    d = cd.depth(mesh, 1)
    assert d["depth"] >= 1 and d["planes"][0] == 0
    assert cd.depth(mesh, 5)["depth"] == 0                                      # no out-edges: no corridor
    assert cd.summary(mesh)["at"] == 100.0
    assert cd.token_id_of(mesh, "inventory") == 1 and cd.token_id_of(mesh, "nope") is None
    # re-accretion replaces, never accumulates
    s2 = cd.accrete(mesh, cfg=cd.CoherencyConfig(lock_min=0.3), gov=GovernanceConfig(), now=200.0)
    assert s2["tokens"] == 5
    n_planes = mesh.execute(f"SELECT COUNT(*) FROM {cd.TABLE_EMBED}").fetchone()[0]
    assert n_planes == s2["plane1"] + s2["plane2"] + s2["crystals"]


def test_accretion_writes_only_its_own_tables_and_keys(mesh):
    before = {t: mesh.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("mesh_slm_vocab", "mesh_slm_embed", "mesh_slm_quipu", "mesh_slm_meta")}
    embed_before = mesh.execute("SELECT * FROM mesh_slm_embed ORDER BY token_id").fetchall()
    tables_before = {r[0] for r in mesh.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    cd.accrete(mesh, cfg=cd.CoherencyConfig(lock_min=0.3))
    after = {t: mesh.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
             for t in ("mesh_slm_vocab", "mesh_slm_embed", "mesh_slm_quipu", "mesh_slm_meta")}
    assert after == before
    assert mesh.execute("SELECT * FROM mesh_slm_embed ORDER BY token_id").fetchall() == embed_before
    new_tables = {r[0] for r in mesh.execute("SELECT name FROM sqlite_master WHERE type='table'")} - tables_before
    assert new_tables == {cd.TABLE_EMBED, cd.TABLE_COHERENCY, cd.TABLE_DEPTH}
    keys = {r[0] for r in mesh.execute("SELECT key FROM brain_kv")}
    assert keys == {"entirety:conscious_emergence", cd.KV_SUMMARY}
    with pytest.raises(PermissionError):
        cd._kv_set(mesh, "entirety:state", {})


def test_accretion_on_a_database_without_a_mesh_is_empty_not_an_error():
    cn = sqlite3.connect(":memory:")
    s = cd.accrete(cn)
    assert s["tokens"] == 0 and s["plane1"] == 0 and s["capacity"] == 0
    assert cd.summary(cn)["tokens"] == 0


# ---------------------------------------------------------------------------
# Top-down anchoring
# ---------------------------------------------------------------------------

def test_anchoring_moves_grounded_tokens_and_leaves_ungrounded_alone(mesh):
    cd.accrete(mesh, cfg=cd.CoherencyConfig(lock_min=0.3), gov=GovernanceConfig())
    mesh_state = [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]
    # a token with neither plane 2 nor a crystal is untouched
    grounded = {tid for tid in TOKENS if any(k >= 2 for k in cd.fibre(mesh, tid))}
    plain = [tid for tid in TOKENS if tid not in grounded]
    cands = [(tid, 0.5, TOKENS[tid][0], (tid, tid)) for tid in TOKENS]
    out = cd.anchor(mesh, cands, mesh_state, lam=0.5)
    scores = {c[0]: c[1] for c in out}
    for tid in plain:
        assert scores[tid] == pytest.approx(0.5)
    assert any(scores[tid] != pytest.approx(0.5) for tid in grounded)
    assert [c[1] for c in out] == sorted((c[1] for c in out), reverse=True)     # re-sorted
    assert cd.anchor(mesh, cands, mesh_state, lam=0.0) == cands                  # λ = 0: identity
    assert cd.anchor(mesh, [], mesh_state, lam=0.5) == []


def test_wrapped_scorer_falls_back_to_the_original_on_error(mesh, monkeypatch):
    calls = []

    def original(cn, last_id, last_pos, mesh_vec, top_k=32, **kw):
        calls.append((last_id, top_k))
        return [(1, 0.5, "inventory", (1, 1)), (5, 0.4, "lonely", (5, 5))]

    wrapped = cd.wrap_score_candidates(original, cfg=cd.CoherencyConfig(anchor_lambda=0.5))
    assert wrapped.__wrapped__ is original
    cd.accrete(mesh, cfg=cd.CoherencyConfig(lock_min=0.3))
    out = wrapped(mesh, 1, (1, 1), [0.1] * 7, top_k=12)
    assert calls == [(1, 12)] and len(out) == 2
    monkeypatch.setattr(cd, "anchor", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert wrapped(mesh, 1, (1, 1), [0.1] * 7) == original(mesh, 1, (1, 1), [0.1] * 7)


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("QUIPU_PLANES_ANCHOR_LAMBDA", "0.25")
    monkeypatch.setenv("QUIPU_PLANES_MAX_CRYSTALS", "3")
    monkeypatch.setenv("QUIPU_PLANES_LOCK_MIN", "nope")
    c = cd.CoherencyConfig.from_env()
    assert c.anchor_lambda == 0.25 and c.max_specialist_planes == 3 and c.lock_min == 0.6
