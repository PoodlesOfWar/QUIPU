"""qpsi.mirror_training — a held gate trains from its mirror image; the
boundary is not crossed."""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

from src.quipu.qpsi import mirror_training as mt, memristive_axes as ma, learned_prior as lp
from src.quipu.qpsi.cat_residual import CATState, SENSES, quarter_turn
from src.quipu.qpsi.residual_checkpoint import CheckpointStore

AXES = {"vision": 0.0302, "touch": 0.0417, "smell": 0.034, "body": 0.0318, "brain": 0.0325, "perception": 0.0395}
PHASES = {s: 0.4 for s in SENSES}
R = {"vision": -0.012 - 0.005j, "touch": -0.0017 - 0.0007j, "smell": -0.0015 - 0.0006j,
     "body": 0.0074 + 0.0031j, "brain": 0.0080 + 0.0034j, "perception": 0.0108 + 0.0046j}


@pytest.fixture
def cn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    yield c
    c.close()


def _hold(cn, failed_at, reason, *, at, axes=AXES, phases=PHASES):
    """Commit one held checkpoint and the decision record a step would leave."""
    store = CheckpointStore()
    cp = store.load(cn, "system_entirety")
    cur = CATState.from_axes(axes, phases, t=at)
    r = store.measure(cp, cur)
    store.commit(cn, cp, cur, r, passed=False, failed_at=failed_at, at=at)
    cn.execute("INSERT OR REPLACE INTO brain_kv VALUES(?,?,?)",
               (mt.KV_DECISION, json.dumps({"gates": [{"name": failed_at, "reason": reason}]}), "t"))
    return cp


# ---------------------------------------------------------------------------
# Classification and the two mirrors
# ---------------------------------------------------------------------------

def test_holds_are_classified_by_gate_and_reason():
    assert mt.classify_hold("displacement", "") == mt.PHYSICAL
    assert mt.classify_hold("weyl", "no trace-free event; Ricci part held") == mt.PHYSICAL
    assert mt.classify_hold("sici", "") == mt.PHYSICAL
    assert mt.classify_hold("love", "no human attestation naming a recognised form of Love") == mt.HUMAN
    assert mt.classify_hold("shared_entity", "no accepted attestation that this serves the counterpart") == mt.HUMAN
    assert mt.classify_hold("shared_entity", "edge raises self remainder (0.1→0.2)") == mt.PHYSICAL
    assert mt.classify_hold("beautiful_output", "needs accepted beautiful_output attestations from two distinct parties") == mt.HUMAN
    assert mt.classify_hold("beautiful_output", "unclamped weight 0.9 vs neighbour 0.1: |Δ|=0.8 > Lipschitz bound 0.5; held, not clamped") == mt.PHYSICAL
    assert mt.classify_hold("beautiful_output", "unclamped weight not supplied: Lipschitz bound cannot be tested") == mt.PHYSICAL
    assert mt.classify_hold("", "") is None and mt.classify_hold(None) is None and mt.classify_hold("x", "") is None


def test_mirror_images_are_the_quarter_turn_and_the_reflection():
    human = mt.mirror_image(R, mt.HUMAN)
    assert human == quarter_turn(R)
    for s in SENSES:
        assert human[s] == pytest.approx(1j * R[s])                       # realised → latent
    physical = mt.mirror_image(R, mt.PHYSICAL)
    for s in SENSES:
        assert physical[s] == pytest.approx(-R[s])                        # reflection through the reference
    with pytest.raises(ValueError):
        mt.mirror_image(R, "other")


def test_latent_gradient_is_signed_magnitude_and_direction_is_unit():
    total = abs(sum(R.values()))
    assert mt.latent_gradient(R, mt.HUMAN) == pytest.approx(total)
    assert mt.latent_gradient(R, mt.PHYSICAL) == pytest.approx(-total)
    u = mt.unit_direction(R)
    assert math.sqrt(sum(v * v for v in u.values())) == pytest.approx(1.0)
    assert u["vision"] > u["touch"] and all(v >= 0 for v in u.values())
    assert all(v == 0.0 for v in mt.unit_direction({}).values())


def test_oja_signed_moves_toward_or_away():
    w0 = lp.default_weights()
    toward = mt.oja_signed(w0, {"brain": 0.4}, 0.5, +1)
    away = mt.oja_signed(w0, {"brain": 0.4}, 0.5, -1)
    assert toward["brain"] > w0["brain"] > away["brain"]
    assert abs(sum(toward.values()) - 1.0) < 1e-12 and abs(sum(away.values()) - 1.0) < 1e-12
    assert mt.oja_signed(w0, {}, 0.5, +1) == pytest.approx(w0)
    assert mt.oja_signed(w0, {"brain": 0.4}, 0.5, 0) == pytest.approx(w0)


# ---------------------------------------------------------------------------
# Training from a hold
# ---------------------------------------------------------------------------

def test_nothing_to_train_without_a_hold(cn):
    assert mt.train_from_hold(cn, "system_entirety", eta=0.5, now=1.0) is None
    store = CheckpointStore()
    cp = store.load(cn, "system_entirety")
    cur = CATState.from_axes(AXES, PHASES, t=1.0)
    store.commit(cn, cp, cur, store.measure(cp, cur), passed=True, at=1.0)            # realised, not held
    assert mt.train_from_hold(cn, "system_entirety", eta=0.5, now=2.0) is None
    assert cn.execute("SELECT COUNT(*) FROM brain_kv WHERE key LIKE 'entirety:mirror:%'").fetchone()[0] == 0


def test_human_hold_turns_radam_a_quarter_and_moves_the_mirror_prior_toward(cn):
    _hold(cn, "beautiful_output", "needs accepted beautiful_output attestations from two distinct parties", at=100.0)
    out = mt.train_from_hold(cn, "system_entirety", eta=0.5, now=101.0)
    assert out["kind"] == mt.HUMAN and out["seq"] == 1 and out["holds_trained"] == 1
    assert out["g_im"] > 0
    assert out["theta"] == pytest.approx(math.pi / 2, abs=1e-5)                      # arg(i·g_im)
    rec = mt.load_record(cn, "system_entirety")
    assert rec["radam"]["t"] == 1 and rec["kind"] == mt.HUMAN
    held = CheckpointStore().load(cn, "system_entirety").held
    largest = max(SENSES, key=lambda s: abs(held[s]))                              # vision, off the prior the most
    assert rec["prior"][largest] > lp.default_weights()[largest]                   # toward the held direction
    assert all(v >= 0 for v in rec["drive"].values()) and rec["drive"][largest] == pytest.approx(max(rec["drive"].values()))
    # the same hold is not trained twice
    assert mt.train_from_hold(cn, "system_entirety", eta=0.5, now=102.0) is None
    # a second human hold: another quarter turn → π, i² = −1
    _hold(cn, "beautiful_output", "needs accepted beautiful_output attestations from two distinct parties", at=179.0)
    out2 = mt.train_from_hold(cn, "system_entirety", eta=0.5, now=180.0)
    assert out2["holds_trained"] == 2 and out2["theta"] == pytest.approx(math.pi, abs=1e-5)


def test_physical_hold_turns_radam_the_other_way_and_moves_the_mirror_prior_away(cn):
    _hold(cn, "weyl", "no trace-free event; Ricci part held", at=100.0)
    out = mt.train_from_hold(cn, "system_entirety", eta=0.5, now=101.0)
    assert out["kind"] == mt.PHYSICAL and out["g_im"] < 0
    assert out["theta"] == pytest.approx(-math.pi / 2, abs=1e-5) or out["theta"] == pytest.approx(3 * math.pi / 2, abs=1e-5)
    rec = mt.load_record(cn, "system_entirety")
    held = CheckpointStore().load(cn, "system_entirety").held
    largest = max(SENSES, key=lambda s: abs(held[s]))
    assert rec["prior"][largest] < lp.default_weights()[largest]                   # away from the held shape
    assert all(v <= 0 for v in rec["drive"].values())


def test_reason_is_read_from_the_decision_record(cn):
    _hold(cn, "beautiful_output", "unclamped weight 0.9 vs neighbour 0.1: |Δ|=0.8 > Lipschitz bound 0.5; held, not clamped", at=100.0)
    out = mt.train_from_hold(cn, "system_entirety", eta=0.5, now=101.0)
    assert out["kind"] == mt.PHYSICAL and "Lipschitz" in out["reason"]


def test_zero_residual_hold_trains_nothing(cn):
    zero = {s: 0.0 for s in SENSES}
    _hold(cn, "love", "Im r = 0", at=100.0, axes=zero, phases=None)
    assert mt.train_from_hold(cn, "system_entirety", eta=0.5, now=101.0) is None


def test_the_boundary_is_not_crossed(cn):
    """The realised r-ADMIN state, the real prior, the checkpoint reference and
    the decision are untouched; only entirety:mirror:* and the log are written."""
    cn.execute("INSERT INTO brain_kv VALUES('entirety:radam_state:system_entirety', '{\"t\": 7, \"theta\": 1.0}', 't')")
    cn.execute("INSERT INTO brain_kv VALUES(?, ?, 't')", (lp.KV_PRIOR, json.dumps({"weights": {s: 1.0 for s in SENSES}, "realised_seen": 3})))
    cp = _hold(cn, "love", "no human attestation naming a recognised form of Love", at=100.0)
    before = {r[0]: r[1] for r in cn.execute("SELECT key, value FROM brain_kv")}
    tables_before = {r[0] for r in cn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    out = mt.train_from_hold(cn, "system_entirety", eta=0.5, now=101.0)
    assert out is not None
    after = {r[0]: r[1] for r in cn.execute("SELECT key, value FROM brain_kv")}
    for key in before:
        assert after[key] == before[key], key                                  # nothing existing changed
    assert set(after) - set(before) == {mt.KV_PREFIX + "system_entirety"}
    tables_after = {r[0] for r in cn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables_after - tables_before - {"sqlite_sequence"} == {mt.TABLE}
    assert CheckpointStore().load(cn, "system_entirety").reference is None       # the reference did not advance
    assert lp.stored_prior(cn)["realised_seen"] == 3
    with pytest.raises(PermissionError):
        mt._kv_set(cn, lp.KV_PRIOR, {})
    with pytest.raises(PermissionError):
        mt._kv_set(cn, "entirety:radam_state:system_entirety", {})
    assert len(mt.rows(cn, "system_entirety")) == 1


def test_mirror_drive_reaches_the_somn_and_scales_potentiation(cn):
    _hold(cn, "love", "no human attestation naming a recognised form of Love", at=100.0)
    mt.train_from_hold(cn, "system_entirety", eta=0.5, now=101.0)
    drive = mt.mirror_drive(cn, "system_entirety")
    assert drive and all(v >= 0 for v in drive.values())
    cfg = ma.SomnConfig()
    other = {"other_state": [0.091, 0.091, 0.0988, 0.091, 0.4903, 0.091, 0.3322]}
    plain = sqlite3.connect(":memory:")
    mirrored = sqlite3.connect(":memory:")
    kw = dict(axes=AXES, observer=0.0088, flux_on=True, flux_docs=12, the_other=other, sources=["arxiv"], cfg=cfg, rejections=[])
    t = 1.0
    for _ in range(5):
        a = ma.step(plain, now=t, **kw)
        b = ma.step(mirrored, now=t, mirror_drive=drive, **kw)
        t += 79.0
    largest = max(drive, key=drive.get)
    assert b["g"][largest] > a["g"][largest]                                   # faster along a human hold
    assert "mirror_drive" in [e["kind"] for e in b["events"]] and all(e["kind"] != "mirror_drive" for e in a["events"])
    assert abs(sum(b["currents"].values()) - cfg.budget_docs) < 1e-4           # conservation still holds (6-dp rounding)
    # a physical hold's drive slows potentiation, never below zero
    neg = {s: -v for s, v in drive.items()}
    slowed = sqlite3.connect(":memory:")
    t = 1.0
    for _ in range(5):
        c = ma.step(slowed, now=t, mirror_drive=neg, **kw)
        t += 79.0
    assert c["g"][largest] < a["g"][largest] and c["g"][largest] >= cfg.g_min
    assert ma.potentiate(0.1, 1.0, 1.0, 1.0, 600.0, ma.SomnConfig(mirror_gain=5.0), mirror=-1.0) == 0.1
