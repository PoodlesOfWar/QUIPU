"""qpsi.learned_prior — the prior is Oja's, advances only on realisation, and is
byte-identical to the designer's table until then."""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

import src.quipu.system_entirety as system_entirety
from src.quipu.qpsi import learned_prior as lp
from src.quipu.qpsi.cat_residual import SENSES, SENSE_WEIGHTS, CATState
from src.quipu.qpsi.residual_checkpoint import CheckpointStore

REST = {"vision": 0.0302, "touch": 0.0417, "smell": 0.034,
        "body": 0.0318, "brain": 0.0325, "perception": 0.0395}


@pytest.fixture
def cn(monkeypatch):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    monkeypatch.setattr(lp, "_open", lambda: _Ctx(c))
    yield c
    c.close()


class _Ctx:
    def __init__(self, c):
        self.c = c

    def __enter__(self):
        return self.c

    def __exit__(self, *a):
        self.c.commit()
        return False


# ---------------------------------------------------------------------------
# Oja
# ---------------------------------------------------------------------------

def test_oja_moves_the_prior_toward_the_realised_axis_and_stays_a_mass_prior():
    w0 = lp.default_weights()
    w1 = lp.oja_step(w0, {"brain": 0.4}, 0.5)
    assert abs(sum(w1.values()) - 1.0) < 1e-12 and all(v > 0 for v in w1.values())
    assert w1["brain"] > w0["brain"] and w1["vision"] < w0["vision"]
    # repeated steps converge toward the fed direction
    w = w0
    for _ in range(200):
        w = lp.oja_step(w, {"brain": 0.4}, 0.5)
    assert w["brain"] > 0.9


def test_oja_is_identity_on_zero_displacement_or_zero_eta():
    w0 = lp.default_weights()
    assert lp.oja_step(w0, {}, 0.5) == pytest.approx(w0)
    assert lp.oja_step(w0, {"brain": 0.3}, 0.0) == pytest.approx(w0)
    # complex displacements use magnitudes
    a = lp.oja_step(w0, {"brain": 0.3j}, 0.1)
    b = lp.oja_step(w0, {"brain": 0.3}, 0.1)
    assert a == pytest.approx(b)


def test_observer_with_default_weights_equals_observer_tangent():
    assert lp.observer_with_weights(REST, lp.default_weights()) == pytest.approx(system_entirety.observer_tangent(REST))
    zero = {s: 0.0 for s in SENSES}
    assert lp.observer_with_weights(zero, lp.default_weights()) == 0.0
    assert 0.0 <= lp.observer_with_weights({s: 1.0 for s in SENSES}, {"vision": 1.0}) <= 1.0


# ---------------------------------------------------------------------------
# Store and the realisation rule
# ---------------------------------------------------------------------------

def test_sense_weights_fall_back_to_designer_table_and_validate_stored(cn):
    assert lp.stored_prior(cn) is None
    assert lp.sense_weights(cn) == pytest.approx({s: SENSE_WEIGHTS[s] for s in SENSES})
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute("INSERT INTO brain_kv VALUES(?,?,?)", (lp.KV_PRIOR, json.dumps({"weights": {"vision": -1}}), "t"))
    assert lp.stored_prior(cn) is None                                   # invalid → ignored
    good = {s: 1.0 for s in SENSES}
    cn.execute("UPDATE brain_kv SET value=? WHERE key=?", (json.dumps({"weights": good, "realised_seen": 2}), lp.KV_PRIOR))
    rec = lp.stored_prior(cn)
    assert rec["realised_seen"] == 2 and rec["weights"] == pytest.approx({s: 1 / 6 for s in SENSES})


def test_prior_does_not_move_while_the_hold_is_held(cn):
    store = CheckpointStore()
    cp = store.load(cn, "system_entirety")
    cur = CATState.from_axes(REST, None, t=100.0)
    r = store.measure(cp, cur)
    store.commit(cn, cp, cur, r, passed=False, failed_at="beautiful_output", at=100.0)
    store.commit(cn, cp, cur, r, passed=False, failed_at="beautiful_output", at=179.0)
    assert lp.advance_on_realisation(cn, "system_entirety", eta=0.5, now=200.0) is None
    assert lp.stored_prior(cn) is None


def test_prior_advances_once_per_realisation_by_the_realised_displacement(cn):
    store = CheckpointStore()
    cp = store.load(cn, "system_entirety")
    first = CATState.from_axes(REST, None, t=100.0)
    store.commit(cn, cp, first, store.measure(cp, first), passed=True, at=100.0)
    got = lp.realised_displacement(cn, "system_entirety")
    assert got is not None
    disp, seq = got
    assert seq == 1 and disp == pytest.approx(REST)                     # first realisation: the whole reference
    rec = lp.advance_on_realisation(cn, "system_entirety", eta=0.5, now=101.0)
    assert rec is not None and rec["realised_seen"] == 1
    expected = lp.oja_step(lp.default_weights(), REST, 0.5)
    assert rec["weights"] == pytest.approx(expected, abs=1e-9)
    assert rec["history"][-1]["seq"] == 1
    # same realisation count → nothing more happens
    assert lp.advance_on_realisation(cn, "system_entirety", eta=0.5, now=102.0) is None
    # a second realisation: displacement is new − old, one more step
    louder = dict(REST); louder["brain"] = 0.5
    second = CATState.from_axes(louder, None, t=200.0)
    store.commit(cn, cp, second, store.measure(cp, second), passed=True, at=200.0)
    disp2, seq2 = lp.realised_displacement(cn, "system_entirety")
    assert seq2 == 2 and disp2["brain"] == pytest.approx(0.5 - REST["brain"]) and disp2["vision"] == pytest.approx(0.0)
    rec2 = lp.advance_on_realisation(cn, "system_entirety", eta=0.5, now=201.0)
    assert rec2["realised_seen"] == 2 and rec2["weights"]["brain"] > rec["weights"]["brain"]
    assert len(rec2["history"]) == 2


def test_wrapped_observer_is_original_until_a_prior_exists(cn):
    calls = []

    def original(signals=None):
        calls.append(signals)
        return 0.123

    wrapped = lp.wrap_observer_tangent(original)
    assert wrapped(REST) == 0.123 and calls == [REST] and wrapped.__wrapped__ is original
    # store a prior → the wrapper evaluates the same formula on the learned weights
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    w = {s: 1.0 for s in SENSES}
    cn.execute("INSERT INTO brain_kv VALUES(?,?,?)", (lp.KV_PRIOR, json.dumps({"weights": w, "realised_seen": 1}), "t"))
    assert wrapped(REST) == pytest.approx(lp.observer_with_weights(REST, {s: 1 / 6 for s in SENSES}))
    assert len(calls) == 1                                                # original not called this time


def test_kv_set_refuses_other_keys(cn):
    with pytest.raises(PermissionError):
        lp._kv_set(cn, "entirety:state", {})


def test_eta_from_env(monkeypatch):
    monkeypatch.setenv(lp.ETA_ENV, "0.2")
    assert lp.eta_from_env() == 0.2
    monkeypatch.setenv(lp.ETA_ENV, "5")
    assert lp.eta_from_env() == lp.DEFAULT_ETA
    monkeypatch.setenv(lp.ETA_ENV, "x")
    assert lp.eta_from_env() == lp.DEFAULT_ETA
