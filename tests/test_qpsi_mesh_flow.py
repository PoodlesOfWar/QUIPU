"""mesh_flow: axis cells → Mott core → poADAM/rADAM → VNN over the MESH graph → proposed shadow edges."""
from __future__ import annotations

import json
import math
import sqlite3

import numpy as np
import pytest

from src.quipu.qpsi import mesh_flow as MF
from src.quipu.qpsi import mott_spintronic as ms

AXES = {"vision": 0.6, "touch": 0.4, "smell": 0.3, "body": 0.5, "brain": 0.7, "perception": 0.2}
P1 = {a: 0.5 for a in MF.AXES}
P2 = {**P1, "brain": 0.9, "smell": 0.1}


@pytest.fixture
def cn():
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
        CREATE TABLE mesh_slm_vocab(token_id INTEGER PRIMARY KEY, token TEXT UNIQUE, i INT, j INT, freq INT);
        CREATE TABLE mesh_slm_embed(token_id INTEGER PRIMARY KEY, e_vision REAL, e_touch REAL, e_smell REAL,
                                    e_body REAL, e_brain REAL, e_perception REAL, e_entirety REAL);
        CREATE TABLE mesh_slm_quipu(src INT, dst INT, weight REAL, samples INT, PRIMARY KEY(src, dst));
    """)
    rng = np.random.default_rng(7)
    for tid, tok in enumerate(["alpha", "beta", "gamma", "delta", "eps", "zeta"], start=1):
        c.execute("INSERT INTO mesh_slm_vocab VALUES(?,?,0,0,?)", (tid, tok, 100 - tid))
        base = np.array([0.6, 0.4, 0.3, 0.5, 0.7, 0.2, 0.5]) + rng.normal(0, 0.15, 7)
        c.execute("INSERT INTO mesh_slm_embed VALUES(?,?,?,?,?,?,?,?)", (tid, *map(float, base)))
    c.executemany("INSERT INTO mesh_slm_quipu VALUES(?,?,?,1)",
                  [(1, 2, 0.9), (2, 3, 0.8), (3, 4, 0.7), (4, 5, 0.9), (5, 6, 0.6)])
    c.commit()
    yield c
    c.close()


def _mesh(c):
    return {t: c.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall()
            for t in ("mesh_slm_vocab", "mesh_slm_embed", "mesh_slm_quipu")}


def test_drive_maps_pressure_zero_to_rheobase_and_silence():
    cfg = ms.MottSpinConfig()
    d = MF.mott_drive({a: 0.0 for a in MF.AXES}, cfg)
    assert d == [pytest.approx(cfg.v_threshold / cfg.tau_leak)] * 7
    assert all(ms.lif_rate(x, cfg) == 0.0 for x in d)
    d1 = MF.mott_drive({"brain": 1.0, "vision": float("nan"), "touch": 7.0}, cfg)
    assert d1[4] == pytest.approx(d[4] + MF.DRIVE_SPAN) and d1[0] == d[0] and d1[1] == d1[4]


def test_shadow_edges_lift_existing_and_propose_new_pairs():
    Z0 = np.array([[1.0, 0, 0, 0, 0, 0, 0], [0.7, 0.7, 0, 0, 0, 0, 0], [0.6, 0.8, 0, 0, 0, 0, 0]])
    Z1 = np.array([[1.0, 0.3, 0, 0, 0, 0, 0], [0.8, 0.6, 0, 0, 0, 0, 0], [0.75, 0.66, 0, 0, 0, 0, 0]])
    out = MF.shadow_edges(Z0, Z1, [10, 11, 12], [(0, 1, 0.9), (1, 2, 0.9)])
    assert {(e["src"], e["dst"]) for e in out["lift"]} == {(10, 11), (11, 12)}
    assert [(e["src"], e["dst"]) for e in out["new"]] == [(10, 12)]
    assert all(0 < e["lift"] <= MF.EDGE_LIFT_MAX for e in out["lift"])
    half = MF.shadow_edges(Z0, Z1, [10, 11, 12], [(0, 1, 0.9), (1, 2, 0.9)], damping=0.5)
    assert half["new"][0]["lift"] == pytest.approx(out["new"][0]["lift"] / 2)


def test_run_never_writes_the_mesh_and_only_its_keys(cn):
    before = _mesh(cn)
    out = MF.run(cn, pressures=P1, axes=AXES, observer=0.5, now=1.0)
    assert _mesh(cn) == before
    keys = {r[0] for r in cn.execute("SELECT key FROM brain_kv")}
    assert keys == {MF.KV_STATE, MF.KV_HISTORY, MF.KV_PROPOSALS, MF.KV_LATEST, "entirety:vnn:latest"}
    assert out["nodes"] == 6 and out["spikes"] > 0
    assert 1 / MF.PHI <= out["stability_number"] <= MF.PHI + 1e-12
    assert out["vnn_graph_cfl"] <= out["stability_number"] + 1e-12


def test_shadow_edges_are_proposed_only_after_two_distinct_states(cn):
    first = MF.run(cn, pressures=P1, axes=AXES, observer=0.5, now=1.0)
    assert first["proposed"] == 0 and sum(first["shadow_this_step"].values()) > 0
    same = MF.run(cn, pressures=P1, axes=AXES, observer=0.5, now=2.0)
    assert len(json.loads(cn.execute("SELECT value FROM brain_kv WHERE key=?",
                                      (MF.KV_HISTORY,)).fetchone()[0])) <= 2
    MF.run(cn, pressures=P2, axes={**AXES, "brain": 0.2}, observer=0.4, now=3.0)
    props = MF.proposals(cn)["edges"]
    assert props and all(p["status"] == "proposed" and p["realisation"].startswith("none") for p in props)
    assert all(len(p["sessions"]) >= 2 for p in props)
    assert {p["kind"] for p in props} <= {"lift", "new"}
    toks = {"alpha", "beta", "gamma", "delta", "eps", "zeta"}
    assert all(p["src"] in toks and p["dst"] in toks for p in props)
    assert same["session"] != MF.latest(cn)["session"]


def test_identical_state_does_not_count_twice(cn):
    a = MF.run(cn, pressures=P1, axes=AXES, observer=0.5, now=1.0)
    saved = cn.execute("SELECT value FROM brain_kv WHERE key=?", (MF.KV_STATE,)).fetchone()[0]
    cn.execute("DELETE FROM brain_kv WHERE key=?", (MF.KV_STATE,))           # reset the core's memory
    b = MF.run(cn, pressures=P1, axes=AXES, observer=0.5, now=2.0)
    assert a["session"] == b["session"] and b["proposed"] == 0
    assert saved


def test_empty_mesh_runs_the_core_and_proposes_nothing():
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE mesh_slm_vocab(token_id INTEGER PRIMARY KEY, token TEXT, i INT, j INT, freq INT);
        CREATE TABLE mesh_slm_embed(token_id INTEGER PRIMARY KEY, e_vision REAL, e_touch REAL, e_smell REAL,
                                    e_body REAL, e_brain REAL, e_perception REAL, e_entirety REAL);
        CREATE TABLE mesh_slm_quipu(src INT, dst INT, weight REAL, samples INT);
    """)
    out = MF.run(c, pressures=P1, axes=AXES, now=1.0)
    assert out["nodes"] == 0 and out["proposed"] == 0 and out["vnn_graph_cfl"] is None


def test_prefix_guard(cn):
    with pytest.raises(PermissionError):
        MF._kv_set(cn, "entirety:tool_gate", {})


def test_self_organising_wires_mesh_flow_inside_a_savepoint():
    import inspect
    from src.quipu.qpsi import self_organising as so
    src = inspect.getsource(so.after_step)
    assert "mesh_flow.run(" in src and "SAVEPOINT qpsi_mesh_flow" in src
    assert "ROLLBACK TO SAVEPOINT qpsi_mesh_flow" in src
    assert so.MESH_FLOW_ENV == "QUIPU_MESH_FLOW"


def test_brain_without_mesh_tables_runs_the_core_only():
    c = sqlite3.connect(":memory:")
    out = MF.run(c, pressures=P1, axes=AXES, now=1.0)
    assert out["nodes"] == 0 and out["spikes"] > 0 and out["proposed"] == 0
