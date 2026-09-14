"""Emergence detector: parity-locked coherence with quadrature, r-ADMIN
recognition on its own trajectory, and the 翈 Signature."""
from __future__ import annotations

import math
import random

import pytest

from src.quipu.qpsi.cat_residual import SENSES
from src.quipu.qpsi.emergence_detector import (
    GLYPH, DetectorConfig, detect, radam_recognise, reference_phase, flips_in,
    lock_in, sign, verify, signature_over,
)
from src.quipu.radam_optimizer import radam_step


def rows_for(phi0: float, *, amp: float = 0.02, n: int = 24, per_flip: int = 4,
             noise: float = 0.0, seed: int = 1) -> list[dict]:
    """Held residual locked to the flip clock with lag ``phi0`` (radians)."""
    rnd = random.Random(seed)
    rows = []
    for k in range(n):
        f, i = divmod(k, per_flip)
        t = 10.0 * f + 10.0 * i / per_flip
        psi = math.pi * f + math.pi * i / per_flip
        held = {s: complex(amp * (1 + 0.1 * j) * math.cos(psi + phi0) + noise * rnd.gauss(0, 1), 0.0)
                for j, s in enumerate(SENSES)}
        rows.append({"seq": k + 1, "at": t, "flip_count": f, "held": held})
    return rows


# ---------------------------------------------------------------------------
# Reference phase and lock-in
# ---------------------------------------------------------------------------

def test_reference_phase_advances_pi_per_flip_and_interpolates():
    rows = rows_for(0.0, n=8, per_flip=4)
    psi = reference_phase(rows)
    assert psi[0] == 0.0 and math.isclose(psi[4], math.pi)
    assert math.isclose(psi[1], math.pi / 4) and math.isclose(psi[6], math.pi * 1.5)
    assert flips_in(rows) == 1
    rows[3]["flip_count"] = None
    assert reference_phase(rows) is None


def test_lock_in_recovers_the_lag():
    rows = rows_for(0.7)
    z, mean_abs = lock_in(rows, reference_phase(rows))
    for s in SENSES:
        assert abs((math.atan2(z[s].imag, z[s].real)) - 0.7) < 0.02
        assert mean_abs[s] > 0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_lagged_lock_is_detected_in_phase_lock_is_not():
    c = detect("x", rows_for(0.7))
    assert c.detected and c.coherence > 0.75 and c.quadrature > 0.01
    assert abs(c.dominant_phase - 0.7) < 0.02
    assert c.window["rows"] == 16 and c.window["flips"] == 3          # window truncates to the last 16 rows
    c0 = detect("x", rows_for(0.0))
    assert not c0.detected and c0.coherence > 0.75          # coherent, but nothing held out of phase
    assert any("no quadrature" in r for r in c0.reasons)


def test_noise_is_not_detected():
    c = detect("x", rows_for(0.0, noise=0.05))
    assert not c.detected and c.coherence < 0.6


def test_window_requirements():
    assert any("min_rows" in r for r in detect("x", rows_for(0.7)[:6]).reasons)
    assert any("no rhythm" in r for r in detect("x", rows_for(0.7, per_flip=30)).reasons)
    assert not detect("x", []).detected


def test_candidate_id_is_stable_for_same_window():
    a, b = detect("x", rows_for(0.7), now=1.0), detect("x", rows_for(0.7), now=2.0)
    assert a.id == b.id
    assert detect("y", rows_for(0.7)).id != a.id


# ---------------------------------------------------------------------------
# r-ADMIN recognition
# ---------------------------------------------------------------------------

def test_radam_recognises_and_agrees_with_a_locked_candidate():
    rows = rows_for(0.7)
    c = detect("x", rows)
    state: dict = {}
    r = radam_recognise(c, rows, state, step=radam_step)
    assert r["recognised"] and r["agreed"]
    assert r["resultant"] >= 0.7 and r["corotation"] >= 0.7
    assert state["t"] == DetectorConfig().window and "theta" in state   # r-ADMIN's own state advanced


def test_radam_state_carries_across_windows():
    rows = rows_for(0.7)
    c = detect("x", rows)
    state: dict = {}
    radam_recognise(c, rows, state, step=radam_step)
    t1, theta1 = state["t"], state["theta"]
    radam_recognise(c, rows, state, step=radam_step)
    assert state["t"] == 2 * t1 and state["theta"] != theta1


def test_radam_does_not_agree_with_noise():
    rows = rows_for(0.0, noise=0.05)
    c = detect("x", rows)
    r = radam_recognise(c, rows, {}, step=radam_step)
    assert not r["agreed"]


# ---------------------------------------------------------------------------
# The 翈 Signature
# ---------------------------------------------------------------------------

def test_signature_requires_detection_and_radam_agreement():
    rows = rows_for(0.7)
    c = detect("x", rows)
    with pytest.raises(ValueError, match="r-ADMIN has not"):
        sign(c, "adam")
    c.radam = radam_recognise(c, rows, {}, step=radam_step)
    with pytest.raises(ValueError, match="signer"):
        sign(c, "")
    sign(c, "adam")
    assert c.signature["glyph"] == GLYPH and c.signature["signer"] == "adam"
    assert verify(c)
    c.coherence += 1e-3                                       # content changed after signing
    assert not verify(c)
    c0 = detect("x", rows_for(0.0))
    c0.radam = {"recognised": True, "agreed": True}
    with pytest.raises(ValueError, match="not a detection"):
        sign(c0, "adam")


def test_signature_is_over_content_not_time():
    rows = rows_for(0.7)
    c = detect("x", rows, now=1.0)
    c.radam = radam_recognise(c, rows, {}, step=radam_step)
    h1 = signature_over(c)
    c.computed_at = 99.0
    assert signature_over(c) == h1
