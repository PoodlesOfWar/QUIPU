"""Tests for gard_info.py — Fisher / Cramér–Rao information floor for the lossy tier."""
from __future__ import annotations

import math

import pytest

from src.quipu.gard_info import (
    GARD_STATE_JSON_BYTES,
    GARD_STATE_PACKED_BYTES,
    GARD_STATE_STORED_BYTES,
    cramer_rao_bound_gard,
    fisher_information_gard,
    gard_floor_bytes,
    gard_floor_bytes_from_state,
    interstitial_bits,
    state_correlation,
)


# ── Constants ────────────────────────────────────────────────────────────────

def test_constants_match_weyl_spec():
    assert GARD_STATE_JSON_BYTES == 50
    assert GARD_STATE_PACKED_BYTES == 20


def test_stored_bytes_is_base64_of_packed_bytes():
    """brain_kv is TEXT, so the packed record is base64-encoded before storage:
    the real on-disk footprint is bigger than the raw Float32 pack size."""
    import base64
    assert GARD_STATE_STORED_BYTES == len(base64.b64encode(b"\x00" * GARD_STATE_PACKED_BYTES))
    assert GARD_STATE_STORED_BYTES == 28
    assert GARD_STATE_STORED_BYTES > GARD_STATE_PACKED_BYTES


# ── fisher_information_gard ───────────────────────────────────────────────────

def test_fisher_information_gard_single_sample():
    """I = 1/sigma^2 for n=1."""
    sigma = 0.1
    assert fisher_information_gard(sigma) == pytest.approx(1.0 / (sigma ** 2), rel=1e-9)


def test_fisher_information_gard_n_samples():
    """I = n/sigma^2."""
    sigma = 0.2
    n = 4
    assert fisher_information_gard(sigma, n) == pytest.approx(n / (sigma ** 2), rel=1e-9)


def test_fisher_information_gard_zero_sigma_clips():
    """sigma=0 should not blow up — clips to floor 1e-12."""
    val = fisher_information_gard(0.0)
    assert math.isfinite(val)
    assert val > 0.0


def test_fisher_information_gard_negative_sigma_clips():
    """Negative sigma clips to the 1e-12 floor (not treated as abs)."""
    # max(1e-12, -0.1) == 1e-12, so result = 1/(1e-12)^2
    val = fisher_information_gard(-0.1)
    assert math.isfinite(val)
    assert val > 0.0
    assert val == pytest.approx(fisher_information_gard(0.0), rel=1e-9)


def test_fisher_information_gard_n_zero_clips_to_one():
    """n=0 clips to n=1."""
    assert fisher_information_gard(0.1, 0) == pytest.approx(fisher_information_gard(0.1, 1), rel=1e-9)


# ── cramer_rao_bound_gard ─────────────────────────────────────────────────────

def test_cramer_rao_bound_gard_is_reciprocal_of_fisher():
    sigma = 0.05
    fi = fisher_information_gard(sigma)
    crb = cramer_rao_bound_gard(sigma)
    assert crb == pytest.approx(1.0 / fi, rel=1e-9)


def test_cramer_rao_bound_gard_reference_values():
    """At n=1, sigma=0.05: CRB = sigma^2 = 0.0025."""
    assert cramer_rao_bound_gard(0.05) == pytest.approx(0.0025, rel=1e-9)


def test_cramer_rao_bound_gard_decreases_with_n():
    """More samples → tighter bound."""
    crb1 = cramer_rao_bound_gard(0.1, 1)
    crb4 = cramer_rao_bound_gard(0.1, 4)
    assert crb4 < crb1
    assert crb4 == pytest.approx(crb1 / 4, rel=1e-9)


# ── gard_floor_bytes ──────────────────────────────────────────────────────────

def test_gard_floor_bytes_keys():
    result = gard_floor_bytes(0.05)
    assert set(result.keys()) == {
        "sigma", "cramer_rao_var", "quant_step", "bits_per_component", "floor_bytes"
    }


def test_gard_floor_bytes_reference_n1_sigma05():
    """Reference: n=1, sigma=0.05."""
    result = gard_floor_bytes(0.05, n_samples=1)
    assert result["sigma"] == pytest.approx(0.05, rel=1e-9)
    assert result["cramer_rao_var"] == pytest.approx(0.0025, rel=1e-9)
    # delta* = sqrt(12 * 0.0025) = sqrt(0.03) ≈ 0.1732
    expected_delta = math.sqrt(12.0 * 0.0025)
    assert result["quant_step"] == pytest.approx(expected_delta, rel=1e-9)
    # bits_per = log2(1/delta*), floor_bytes = 5 * ceil(bits_per / 8)
    assert result["floor_bytes"] >= 1


def test_gard_floor_bytes_monotone_sigma():
    """Floor bytes should shrink (or stay flat) as sigma grows."""
    floors = [gard_floor_bytes(s)["floor_bytes"] for s in [0.001, 0.01, 0.05, 0.1, 0.5]]
    for a, b in zip(floors, floors[1:]):
        assert a >= b, f"floor not monotone-decreasing: {floors}"


def test_gard_floor_bytes_n_samples_reduces_floor():
    """More samples → smaller floor (same or lower)."""
    f1 = gard_floor_bytes(0.05, n_samples=1)["floor_bytes"]
    f4 = gard_floor_bytes(0.05, n_samples=4)["floor_bytes"]
    assert f4 <= f1


def test_gard_floor_bytes_n_components_scales():
    """n_components=10 should produce at least as many bytes as n_components=5."""
    f5 = gard_floor_bytes(0.1, n_components=5)["floor_bytes"]
    f10 = gard_floor_bytes(0.1, n_components=10)["floor_bytes"]
    assert f10 >= f5


def test_gard_floor_bytes_minimum_is_positive():
    """Floor must always be at least 1."""
    for sigma in [1e-15, 1e-6, 1e-3, 0.1, 1.0, 10.0]:
        result = gard_floor_bytes(sigma)
        assert result["floor_bytes"] >= 1, f"floor=0 at sigma={sigma}"


# ── gard_floor_bytes_from_state ───────────────────────────────────────────────

def test_gard_floor_bytes_from_state_wires_through_langevin():
    """gard_floor_bytes_from_state delegates to langevin_sigma_from_gard."""
    from src.quipu.gard_shard_model import langevin_sigma_from_gard

    psi5_now = [0.5, 0.8, 0.7, 0.9, 0.8]
    psi5_prev = [0.5, 0.4, 0.7, 0.6, 0.3]

    sigma = langevin_sigma_from_gard(psi5_now, psi5_prev)
    expected = gard_floor_bytes(sigma)
    result = gard_floor_bytes_from_state(psi5_now, psi5_prev)

    assert result["sigma"] == pytest.approx(expected["sigma"], rel=1e-9)
    assert result["floor_bytes"] == expected["floor_bytes"]


def test_gard_floor_bytes_from_state_fallback_on_bad_input():
    """Bad input falls back to sigma_base=0.05 via langevin_sigma_from_gard."""
    result = gard_floor_bytes_from_state([0.1] * 3, [0.1] * 5)
    # Should not raise; should return a valid dict.
    assert result["floor_bytes"] >= 1
    assert math.isfinite(result["sigma"])


# ── interstitial_bits ─────────────────────────────────────────────────────────

def test_interstitial_bits_keys():
    result = interstitial_bits(0.05)
    assert set(result.keys()) == {
        "sigma", "floor_bits", "actual_bits", "interstitial_bits", "interstitial_fraction"
    }


def test_interstitial_bits_gap_is_nonnegative():
    """Interstitial gap must never be negative."""
    for sigma in [0.001, 0.01, 0.05, 0.1, 0.5, 2.0]:
        result = interstitial_bits(sigma)
        assert result["interstitial_bits"] >= 0.0, f"negative gap at sigma={sigma}"


def test_interstitial_bits_fraction_in_01():
    """Fraction must lie in [0, 1]."""
    for sigma in [0.001, 0.05, 0.5]:
        result = interstitial_bits(sigma)
        assert 0.0 <= result["interstitial_fraction"] <= 1.0


def test_interstitial_bits_zero_noise():
    """Very small sigma → floor dominates → gap should be 0 (actual ≤ floor rounds to 0)."""
    result = interstitial_bits(1e-9)
    # floor_bits will be very large (cannot resolve sub-noise); actual_bits = 44.8*5=224
    # gap is clamped to 0 when floor_bits > actual_bits
    assert result["interstitial_bits"] >= 0.0
    assert math.isfinite(result["interstitial_fraction"])


def test_interstitial_bits_large_sigma_gives_positive_gap():
    """Large sigma → CRB floor is small → actual bits exceed floor → gap > 0."""
    result = interstitial_bits(0.5)
    assert result["interstitial_bits"] > 0.0
    assert result["actual_bits"] > result["floor_bits"]


def test_interstitial_bits_default_actual_bits_uses_stored_footprint():
    """Default actual_bits reflects the base64 text actually written to
    brain_kv (GARD_STATE_STORED_BYTES), not the smaller pre-encoding
    Float32 pack (GARD_STATE_PACKED_BYTES)."""
    result = interstitial_bits(0.1)
    assert result["actual_bits"] == pytest.approx(GARD_STATE_STORED_BYTES * 8.0)
    assert result["actual_bits"] > GARD_STATE_PACKED_BYTES * 8.0


def test_interstitial_bits_custom_actual_bpc():
    """Custom actual_bits_per_component overrides Float32 default."""
    r_default = interstitial_bits(0.1)
    r_8bit = interstitial_bits(0.1, actual_bits_per_component=8.0)
    # 8-bit encoding has fewer actual bits than Float32 (32-bit)
    assert r_8bit["actual_bits"] < r_default["actual_bits"]


# ── state_correlation ────────────────────────────────────────────────────────

def test_state_correlation_identical_returns_one():
    """Identical states → Pearson correlation = 1."""
    psi = [0.1, 0.5, 0.7, 0.3, 0.9]
    assert state_correlation(psi, psi) == pytest.approx(1.0, abs=1e-9)


def test_state_correlation_negated_returns_minus_one():
    """A negated (reflected about mean) state → correlation = -1."""
    psi = [0.1, 0.5, 0.7, 0.3, 0.9]
    mean = sum(psi) / len(psi)
    neg = [2 * mean - v for v in psi]
    assert state_correlation(psi, neg) == pytest.approx(-1.0, abs=1e-9)


def test_state_correlation_constant_returns_zero():
    """Constant vector has zero variance → correlation undefined → 0."""
    const = [0.5, 0.5, 0.5, 0.5, 0.5]
    psi = [0.1, 0.5, 0.7, 0.3, 0.9]
    assert state_correlation(const, psi) == 0.0
    assert state_correlation(psi, const) == 0.0


def test_state_correlation_symmetry():
    """C(a,b) == C(b,a)."""
    a = [0.2, 0.6, 0.4, 0.8, 0.1]
    b = [0.9, 0.3, 0.7, 0.5, 0.2]
    assert state_correlation(a, b) == pytest.approx(state_correlation(b, a), abs=1e-12)


def test_state_correlation_mismatched_length_returns_zero():
    assert state_correlation([0.1, 0.2, 0.3], [0.4, 0.5]) == 0.0


def test_state_correlation_result_clamped():
    """Result must be in [-1, 1]."""
    import random
    rng = random.Random(42)
    for _ in range(20):
        a = [rng.random() for _ in range(5)]
        b = [rng.random() for _ in range(5)]
        val = state_correlation(a, b)
        assert -1.0 <= val <= 1.0


def test_state_correlation_is_plain_pearson():
    assert state_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert state_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert state_correlation([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) == 0.0


# ── correlation_transitivity_score (via ueqgm_engine) ────────────────────────

def test_correlation_transitivity_score_identical_tensors():
    """Identical states: C=1 everywhere → witness |1·1 − 1|/(1+1) = 0."""
    from src.quipu.ueqgm_engine import correlation_transitivity_score

    psi = [0.5, 0.6, 0.4, 0.7, 0.3]
    score = correlation_transitivity_score([psi, psi, psi])
    # witness = |C_01*C_12 - C_02| / (1+|C_02|) = |1*1 - 1| / (1+1) = 0
    assert score == pytest.approx(0.0, abs=1e-6)


def test_correlation_transitivity_score_single_tensor_returns_zero():
    """One state → fewer than 2 → returns 0."""
    from src.quipu.ueqgm_engine import correlation_transitivity_score

    assert correlation_transitivity_score([[0.5, 0.5, 0.5, 0.5, 0.5]]) == 0.0


def test_correlation_transitivity_score_empty_returns_zero():
    from src.quipu.ueqgm_engine import correlation_transitivity_score

    assert correlation_transitivity_score([]) == 0.0


def test_correlation_transitivity_score_two_tensors_degenerate_path():
    """Two states → degenerate: returns |C_01|."""
    from src.quipu.ueqgm_engine import correlation_transitivity_score

    a = [0.1, 0.5, 0.7, 0.3, 0.9]
    b = [0.9, 0.5, 0.3, 0.7, 0.1]
    expected = abs(state_correlation(a, b))
    score = correlation_transitivity_score([a, b])
    assert score == pytest.approx(expected, abs=1e-6)


def test_correlation_transitivity_score_in_01():
    """Score must always be in [0, 1]."""
    from src.quipu.ueqgm_engine import correlation_transitivity_score
    import random

    rng = random.Random(99)
    for _ in range(30):
        seq = [[rng.random() for _ in range(5)] for _ in range(rng.randint(2, 6))]
        score = correlation_transitivity_score(seq)
        assert 0.0 <= score <= 1.0, f"score out of range: {score}"


def test_transitive_sequence_scores_zero():
    """Perfectly correlated states compose transitively, so the score vanishes.
    This is what makes it a structure detector rather than a magnitude."""
    from src.quipu.ueqgm_engine import correlation_transitivity_score

    seq = [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15]]
    assert correlation_transitivity_score(seq) == pytest.approx(0.0, abs=1e-6)


def test_deprecated_aliases_still_resolve():
    from src.quipu.gard_info import photon_covariance_proxy, state_correlation
    from src.quipu.ueqgm_engine import (
        correlation_transitivity_score, interstitial_entanglement_score,
    )
    assert photon_covariance_proxy is state_correlation
    assert interstitial_entanglement_score is correlation_transitivity_score
