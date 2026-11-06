import cmath
import math
import random

from src.quipu.qpsi.cat_residual import (
    SENSES, CATState, residual, observer, observer_legacy, pstar, pivoted_relu,
    rectify, realize, holon_candidates, kolmogorov_eta,
)

REST = {"vision": 0.0302, "touch": 0.0417, "smell": 0.034,
        "body": 0.0318, "brain": 0.0325, "perception": 0.0395}   # live 08-26 frames


def test_legacy_equivalence():
    rng = random.Random(1)
    for _ in range(200):
        axes = {s: rng.random() for s in SENSES}
        r = residual(CATState.from_axes(axes), None)
        assert abs(observer(r) - observer_legacy(axes)) < 1e-12


def test_live_frames_have_zero_residual():
    # 76 identical frames, 3 s apart: legacy observer 0.0088 each; residual 0.
    prev = None
    realised_total = 0
    for i in range(76):
        cur = CATState.from_axes(REST, t=3.0 * i)
        r = residual(cur, prev, nu=0.0)
        assert observer(r) == (observer_legacy(REST) if prev is None else 0.0)
        if prev is not None:
            assert all(v == 0 for v in r.values())
        if prev is not None:
            realised_total += sum(p.realised for p in realize(r, pivot=0.01))
        prev = cur
    assert realised_total == 0          # today: 15 upserts x 76 ticks
    assert round(observer_legacy(REST), 4) == 0.0088


def test_decay_channel_expects_decline_so_holding_level_is_a_residual():
    a = CATState.from_axes(REST, t=0.0)
    b = CATState.from_axes(REST, t=600.0)
    r0 = residual(b, a, nu=0.0)
    r1 = residual(b, a, nu=1e-3)
    assert observer(r0) == 0.0 and observer(r1) > 0.0


def test_eta_gate_zeroes_small_axes():
    a = CATState.from_axes(REST, t=0.0)
    moved = dict(REST); moved["touch"] += 0.05; moved["smell"] += 0.001
    b = CATState.from_axes(moved, t=3.0)
    # The prior projection spreads the touch move onto every axis
    # (smell picks up -0.0103); eta must sit above that leakage.
    r = residual(b, a, eta=0.02)
    assert r["smell"] == 0 and abs(r["touch"]) > 0.03


def test_rectifier_regimes():
    assert pivoted_relu(0.3, 0.5, 1.0) == 0.3               # alpha=1 identity
    assert pivoted_relu(0.3, 0.5, 0.0) == 0.5               # hard gate at pivot
    assert pivoted_relu(0.7, 0.5, 0.0) == 0.7               # above passes
    assert 0 < pstar(1, 5) < pstar(1, 1) < pstar(5, 1) < 1


def test_lipschitz_clamp_bounds_neighbour_gradient():
    r = {s: 0j for s in SENSES}
    r["touch"] = 0.9 + 0j
    free = rectify(r, pivot=0.0, alpha=1.0)
    clamped = rectify(r, pivot=0.0, alpha=1.0, lipschitz=0.2)
    assert free["touch"] == 0.9 and clamped["touch"] <= 0.2 + 1e-12


def test_forcing_without_clamp_grows_gradient_with_finite_mass():
    # Repeated same-direction forcing on one axis: total mass bounded by
    # normalisation, neighbour gradient grows; the clamp holds it.
    state = {s: 0.05 for s in SENSES}
    grad_free, grad_clamped = [], []
    for k in range(50):
        state["touch"] = min(1.0, state["touch"] + 0.02)
        r = {s: complex(state[s]) for s in SENSES}
        f = rectify(r, pivot=0.0)
        c = rectify(r, pivot=0.0, lipschitz=0.1)
        grad_free.append(f["touch"] - f["vision"])
        grad_clamped.append(c["touch"] - c["vision"])
    assert grad_free[-1] > grad_free[0] and max(grad_clamped) <= 0.1 + 1e-12


def test_holons_need_recurrence_not_magnitude():
    loud = [{"touch": 50.0, "vision": 0.0}]
    assert holon_candidates(loud, min_sessions=3) == []
    steady = [{"touch": 0.3, "vision": 0.02 * i} for i in range(4)]
    h = holon_candidates(steady, min_sessions=3, cos_tau=0.9)
    assert len(h) == 1 and h[0].support == 4


def test_kolmogorov_eta_scales():
    assert kolmogorov_eta(0.0, 1.0) == 0.0
    assert kolmogorov_eta(1e-3, 1e-6) > kolmogorov_eta(1e-3, 1e-3)


def test_summary_character_grounding():
    from src.quipu.qpsi.cat_residual import (SUMMARY_CHARACTER, SUMMARY_GROUND, latent, realised_part,
                              quarter_turn, parity_from_turns)
    assert SUMMARY_CHARACTER == "翈" and SUMMARY_GROUND ** 2 == -1
    r = {s: complex(0.3, 0.1) for s in SENSES}
    q = quarter_turn(r)
    assert all(abs(v - complex(-0.1, 0.3)) < 1e-12 for v in q.values())   # realised -> latent
    qq = quarter_turn(q)
    assert all(abs(qq[s] + r[s]) < 1e-12 for s in SENSES)                 # i^2 = -1
    assert realised_part(r)["touch"] == 0.3 and latent(r)["touch"] == 0.1
    assert [parity_from_turns(n) for n in range(4)] == [1, 0, -1, 0]      # broaden, 翈, deepen, 翈
