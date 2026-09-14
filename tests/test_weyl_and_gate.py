import math
import sqlite3
import random

from src.quipu.qpsi.cat_residual import SENSES, residual, CATState, realize
from src.quipu.qpsi.weyl_channel import split, signature, max_leakage_ratio, decompose
from src.quipu.qpsi.edge_gate import ensure_columns, gated_upsert_edge

REST = {"vision": 0.0302, "touch": 0.0417, "smell": 0.034,
        "body": 0.0318, "brain": 0.0325, "perception": 0.0395}


# ---------------- leakage is a hard requirement, and constant eta fails ------

def test_leakage_ratio_is_0_2725_between_vision_and_touch():
    ratio, k, j = max_leakage_ratio()
    assert abs(ratio - 0.2725) < 5e-4 and {k, j} == {"vision", "touch"}


def test_constant_eta_cannot_be_correct_per_axis():
    # For any eta, a large enough single-axis move realises >1 axis.
    for eta in (0.01, 0.05, 0.1):
        prev = CATState.from_axes(REST, t=0)
        moved = dict(REST); moved["touch"] += 0.6
        cur = CATState.from_axes(moved, t=3)
        r = residual(cur, prev, eta=eta)
        assert sum(1 for v in r.values() if v != 0) > 1


def test_signature_decomposition_realises_one_event_per_axis_moved():
    rng = random.Random(7)
    for _ in range(200):
        delta = {s: 0.0 for s in SENSES}
        k = rng.choice(SENSES)
        delta[k] = rng.uniform(-0.9, 0.9)
        if abs(delta[k]) < 0.05:
            continue
        d = decompose(delta, eta=1e-9)
        assert [e.axis for e in d.events] == [k]
        assert abs(e_amp := d.events[0].amplitude) - abs(delta[k]) < 1e-9
        assert d.remainder_norm < 1e-9


def test_two_axis_move_gives_two_events():
    delta = {s: 0.0 for s in SENSES}
    delta["touch"], delta["brain"] = 0.4, -0.3
    d = decompose(delta, eta=1e-9)
    assert sorted(e.axis for e in d.events) == ["brain", "touch"]
    assert d.remainder_norm < 1e-9


def test_ricci_part_is_the_shared_component():
    delta = {s: 0.1 for s in SENSES}          # uniform lift ≈ along the prior
    sp = split(delta)
    assert sp.ricci > 0 and math.sqrt(sum(v * v for v in sp.weyl.values())) < 0.1


def test_signature_is_trace_free():
    from src.quipu.qpsi.cat_residual import _w_hat
    w = dict(zip(SENSES, _w_hat()))
    for k in SENSES:
        s = signature(k)
        assert abs(sum(s[j] * w[j] for j in SENSES)) < 1e-12


# ---------------- displacement-gated upsert (item 1) ------------------------

def _db():
    cn = sqlite3.connect(":memory:")
    cn.execute(
        "CREATE TABLE corpus_edge(src_id TEXT, src_type TEXT, dst_id TEXT, dst_type TEXT, "
        "rel TEXT, weight REAL, last_seen TEXT, samples INTEGER DEFAULT 1, "
        "PRIMARY KEY(src_id, src_type, dst_id, dst_type, rel))"
    )
    ensure_columns(cn)
    return cn


def test_replay_76_identical_weights_yields_one_sample():
    cn = _db()
    for i in range(76):
        r = gated_upsert_edge(cn, "se", "SystemEntirety", "w", "Window", "EMITS",
                              0.0634, f"t{i}", flip_count=0)
    assert r["samples"] == 1 and r["displaced"] is False
    row = cn.execute("SELECT samples, last_seen FROM corpus_edge").fetchone()
    assert row == (1, "t75")                   # heartbeat kept moving


def test_displacement_and_flip_increment_samples():
    cn = _db()
    gated_upsert_edge(cn, "a", "T", "b", "T", "R", 0.10, "t0", flip_count=0)
    r1 = gated_upsert_edge(cn, "a", "T", "b", "T", "R", 0.10, "t1", flip_count=0)
    r2 = gated_upsert_edge(cn, "a", "T", "b", "T", "R", 0.30, "t2", flip_count=0)
    r3 = gated_upsert_edge(cn, "a", "T", "b", "T", "R", 0.30, "t3", flip_count=1)
    assert (r1["samples"], r2["samples"], r3["samples"]) == (1, 2, 3)
    assert abs(r2["weight"] - 0.20) < 1e-12    # mean over displaced samples only
    fc = cn.execute("SELECT flip_count_at_displacement FROM corpus_edge").fetchone()[0]
    assert fc == 1


def test_columns_are_additive_and_idempotent():
    cn = _db(); ensure_columns(cn); ensure_columns(cn)
    cols = {r[1] for r in cn.execute("PRAGMA table_info(corpus_edge)")}
    assert {"flip_count_at_displacement", "cost_class"} <= cols
