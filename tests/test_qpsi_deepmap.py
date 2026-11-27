"""deepmap: information theory along deep MESH pathways."""
from __future__ import annotations

import json
import math
import sqlite3

import numpy as np
import pytest

from src.quipu.qpsi import deepmap as DM
from src.quipu.qpsi import magic as MG
from src.quipu.qpsi import mesh_flow as MF

PHI = (1 + math.sqrt(5)) / 2
AX = {"vision": .6, "touch": .4, "smell": .3, "body": .5, "brain": .7, "perception": .2}
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
        e = np.array([.6, .4, .3, .5, .7, .2, .5]) + rng.normal(0, .15, 7)
        c.execute("INSERT INTO mesh_slm_embed VALUES(?,?,?,?,?,?,?,?)", (tid, *map(float, e)))
    c.executemany("INSERT INTO mesh_slm_quipu VALUES(?,?,?,?)",
                  [(1, 2, .9, 40), (2, 3, .8, 3), (3, 4, .7, 20), (4, 5, .9, 9), (5, 6, .6, 1),
                   (2, 5, .3, 5), (1, 3, .2, 2)])
    c.commit()
    yield c
    c.close()


def _mesh(c):
    return {t: c.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall()
            for t in ("mesh_slm_vocab", "mesh_slm_embed", "mesh_slm_quipu")}


def _hop(**kw):
    base = {"surprisal": 1.0, "resolvable": 0.5, "entropy": 1.0, "magic": 0.0, "held": 1.0,
            "cpt_asymmetry": 0.0, "planes": [1 / 7] * 7}
    return {**base, **kw}


def test_resolvable_bits_follow_the_crb_step():
    assert DM.resolvable_bits(1) == 0.0 and DM.resolvable_bits(12) == 0.0     # delta* >= 1
    assert DM.resolvable_bits(48) == pytest.approx(1.0)                        # 2*sqrt(3)/sqrt(48) = 1/2
    assert DM.resolvable_bits(4800) > DM.resolvable_bits(48)


def test_branching_entropy_and_hop_planes():
    assert DM.branching_entropy([1, 1, 1, 1]) == pytest.approx(2.0)
    assert DM.branching_entropy([5]) == 0.0 and DM.branching_entropy([]) == 0.0
    pi = DM.hop_planes([0] * 7, [0, 0, 0, 0, 3, 1, 0])
    assert pi[4] == pytest.approx(0.75) and pi[5] == pytest.approx(0.25)
    assert DM.hop_planes(None, [1] * 7) == [pytest.approx(1 / 7)] * 7


def test_composition_rules():
    m = MG.MAX_TWO_QUBIT_MAGIC
    hops = [_hop(surprisal=1.0, resolvable=0.25, magic=m), _hop(surprisal=2.0, resolvable=3.0, magic=m)]
    acc = DM.compose(hops, stability_number=1.0)
    assert acc["surprisal_bits"] == 3.0 and acc["probability"] == pytest.approx(1 / 8)
    assert acc["resolvable_bits"] == pytest.approx(0.25 + 2.0)
    assert acc["interstitial_bits"] == pytest.approx(0.75)
    assert acc["magic"] == pytest.approx(2 * m)                                # additive
    assert acc["stabilizer_purity"] == pytest.approx((7 / 16) ** 2)            # multiplicative
    assert acc["design"] == pytest.approx(1.0) and acc["gain"] == pytest.approx(PHI)
    assert acc["delivered_bits"] == pytest.approx(3.0 / PHI)
    assert acc["landauer_j"] == pytest.approx(1.380649e-23 * 300 * math.log(2) * 3)
    assert acc["info_mass_kg"] == pytest.approx(acc["landauer_j"] / 299_792_458.0 ** 2)
    stab = DM.compose([_hop(), _hop()])
    assert stab["design"] == 0.0 and stab["gain"] == 1.0 and stab["magic_floor"] == 0.0


def test_held_multiplies_and_cpt_takes_the_largest():
    acc = DM.compose([_hop(held=0.8, cpt_asymmetry=0.01), _hop(held=0.5, cpt_asymmetry=0.03)])
    assert acc["held"] == pytest.approx(0.4) and acc["cpt_asymmetry"] == 0.03


def test_build_from_the_mesh_flow_reads_only_and_writes_its_key(cn):
    MF.run(cn, pressures=P1, axes=AX, now=1.0)
    MF.run(cn, pressures=P2, axes={**AX, "brain": .2}, now=2.0, phases={a: .7 for a in MF.AXES})
    before = _mesh(cn)
    dm = DM.build(cn, now=3.0)
    assert _mesh(cn) == before
    keys = {k for (k,) in cn.execute("SELECT key FROM brain_kv WHERE key LIKE 'entirety:deepmap:%'")}
    assert keys == {DM.KV_LATEST}
    assert dm["seed_from"] == "shadow_tokens" and dm["paths"]
    for p in dm["paths"]:
        assert 1 <= p["hops"] <= DM.MAX_DEPTH and len(p["tokens"]) == p["hops"] + 1
        assert len(set(p["tokens"])) == len(p["tokens"])                       # simple walks
        assert 0.0 <= p["design"] <= 1.0 and 0.0 <= p["gain"] <= PHI + 1e-9
        assert p["magic_floor"] <= p["magic"] / p["hops"] + 1e-9
        assert p["info_mass_kg"] > 0.0 and p["plane"] in DM.AXES
    gains = [p["gain"] for p in dm["paths"]]
    assert gains == sorted(gains, reverse=True)
    stored = json.loads(cn.execute("SELECT value FROM brain_kv WHERE key=?", (DM.KV_LATEST,)).fetchone()[0])
    assert stored["paths"][0]["tokens"] == dm["paths"][0]["tokens"]


def test_shadow_edges_are_walkable_and_flagged(cn):
    cn.execute("INSERT INTO brain_kv VALUES('entirety:mesh_flow:proposals', ?, '')", (json.dumps({
        "edges": [{"src": "zeta", "dst": "alpha", "kind": "new", "lift": 0.05}], "tokens": []}),))
    walks = DM.pathways(cn, [6], {}, shadow_edges=[{"src": "zeta", "dst": "alpha", "kind": "new", "lift": 0.05}])
    assert walks and walks[0][0]["dst"] == "alpha" and walks[0][0]["shadow"] is True
    assert DM.compose(walks[0])["shadow_hops"] >= 1


def test_no_mesh_and_no_flow_still_publish(cn):
    dm = DM.build(cn, now=1.0)                       # no flow yet: top tokens seed, no magic
    assert dm["seed_from"] == "top_tokens" and all(p["magic"] == 0.0 for p in dm["paths"])
    bare = sqlite3.connect(":memory:")
    out = DM.build(bare, now=1.0)
    assert out["seed_from"] == "no_mesh" and out["paths"] == []


def test_prefix_guard(cn):
    with pytest.raises(PermissionError):
        DM._kv_set(cn, "entirety:mesh_flow:latest", {})
