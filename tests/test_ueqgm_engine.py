"""Tests for ueqgm_engine.py — bounded scoring and weighting helpers.

Covers the Si/Ci special functions and the axial phase term built on them,
squared cosine similarity, cosine modulation, the graph-density ratio, the
corpus coverage score, the inverse-distance scaling term, phase evolution, the
diffusion terrain step, and the corpus-backed coherence score.

Several helpers are still imported here under their deprecated physics-flavoured
names (``wavefunction_overlap``, ``holographic_entropy``, …), which doubles as
coverage that those aliases keep resolving. See the bottom of this file for the
alias and behavioural-equivalence tests.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

import pytest

from src.quipu.ueqgm_engine import (
    _G_CONST,
    _C_CONST,
    _PHI_BASE,
    _PHI_STEP,
    _SICI_SCALE_FACTOR,
    _SICI_SERIES_CUTOFF,
    _UEQGM_RUNTIME_KEY,
    _raw_sici,
    coherence_to_phi,
    entropic_bayesian_step,
    floquet_modulation_factor,
    get_adaptive_runtime,
    holographic_entropy,
    metric_perturbation,
    phase_evolution_total,
    refresh_adaptive_runtime,
    sici_axial_decay,
    sici_phase_weight,
    tantalum_intermediary_binding,
    ueqgm_coherence_score,
    wavefunction_overlap,
    weyl_scalar_tensor,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ueqgm_db() -> sqlite3.Connection:
    """In-memory DB with corpus_entity, learning_log, and brain_kv."""
    cn = sqlite3.connect(":memory:")
    cn.execute(
        "CREATE TABLE corpus_entity("
        "entity_id TEXT, entity_type TEXT, label TEXT, props_json TEXT, "
        "first_seen TEXT, last_seen TEXT, samples INTEGER DEFAULT 1, "
        "UNIQUE(entity_id, entity_type))"
    )
    cn.execute(
        "CREATE TABLE learning_log("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "logged_at TEXT, kind TEXT, title TEXT, detail TEXT, signal_strength REAL)"
    )
    cn.execute(
        "CREATE TABLE brain_kv("
        "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
    )
    return cn


def _insert_entity(cn: sqlite3.Connection, eid: str, label: str, props: str) -> None:
    cn.execute(
        "INSERT OR IGNORE INTO corpus_entity "
        "(entity_id, entity_type, label, props_json, first_seen, last_seen) "
        "VALUES(?,?,?,?,?,?)",
        (eid, "CorpusEntity", label, props, "2026-01-01", "2026-01-01"),
    )
    cn.commit()


# ── coherence_to_phi ──────────────────────────────────────────────────────────

def test_coherence_to_phi_zero_is_pi_over_4():
    """coherence=0 maps to φ=π/4 (first sin/cos intersection)."""
    assert coherence_to_phi(0) == pytest.approx(math.pi / 4, abs=1e-9)


def test_coherence_to_phi_steps_by_pi():
    """Each coherence unit increases φ by exactly π."""
    for k in range(1, 5):
        assert coherence_to_phi(k) == pytest.approx(
            math.pi / 4 + k * math.pi, abs=1e-9
        )


def test_coherence_to_phi_all_intersection_points():
    """tan(φ) at every intersection point equals 1.0."""
    for k in range(6):
        phi = coherence_to_phi(k)
        # tan(π/4 + kπ) = tan(π/4) = 1.0  for all integer k
        assert math.tan(phi) == pytest.approx(1.0, abs=1e-9)


# ── _raw_sici: the genuine Si/Ci special functions ───────────────────────────
#
# These pin the mathematical identities rather than the implementation, so the
# scipy path and the pure-Python series/continued-fraction path must both
# satisfy them.

def test_raw_sici_fallback_matches_scipy_when_available(monkeypatch):
    import src.quipu.ueqgm_engine as u

    if not u._HAS_SCIPY:
        pytest.skip("scipy not installed")

    x = 40.0
    si_scipy, ci_scipy = u._scipy_sici(x)
    monkeypatch.setattr(u, "_HAS_SCIPY", False)
    si_fb, ci_fb = u._raw_sici(x)
    assert si_fb == pytest.approx(float(si_scipy), rel=1e-14, abs=1e-14)
    assert ci_fb == pytest.approx(float(ci_scipy), rel=1e-14, abs=1e-14)

def test_raw_sici_origin():
    """Si(0) = 0 exactly; Ci has a logarithmic pole at the origin."""
    si, ci = _raw_sici(0.0)
    assert si == 0.0
    assert ci == -math.inf


def test_raw_sici_si_is_odd():
    """Si(-x) = -Si(x)."""
    for x in (0.3, 1.5, 3.0, 50.0):
        assert _raw_sici(-x)[0] == pytest.approx(-_raw_sici(x)[0], rel=1e-12)


def test_raw_sici_known_reference_values():
    """Reference values of Si/Ci to 12 significant figures."""
    # Si(π) is the Gibbs constant; Ci(1) and Si(1) are standard tabulated values.
    assert _raw_sici(math.pi)[0] == pytest.approx(1.851937051982466, rel=1e-12)
    assert _raw_sici(1.0)[0] == pytest.approx(0.946083070367183, rel=1e-12)
    assert _raw_sici(1.0)[1] == pytest.approx(0.337403922900968, rel=1e-12)
    # Ci's first positive root.
    assert _raw_sici(0.6165054856207162)[1] == pytest.approx(0.0, abs=1e-15)


def test_raw_sici_asymptotic_limits():
    """Si(x) → π/2 and Ci(x) → 0 as x → ∞, with O(1/x) decaying oscillation."""
    for x in (1e3, 1e4, 1e5):
        si, ci = _raw_sici(x)
        assert abs(si - math.pi / 2.0) < 2.0 / x
        assert abs(ci) < 2.0 / x


def test_raw_sici_continuous_across_series_cutoff():
    """The series and continued-fraction branches must agree at the switchover.

    Probed at ±1e-11 so the function's own slope there (Si'(2) = sin(2)/2 ≈
    0.45, so ~9e-12 across the probe) stays well under the tolerance: any real
    disagreement between the two branches would be orders of magnitude larger.
    """
    eps = 1e-11
    lo = _raw_sici(_SICI_SERIES_CUTOFF - eps)
    hi = _raw_sici(_SICI_SERIES_CUTOFF + eps)
    assert lo[0] == pytest.approx(hi[0], abs=1e-9)
    assert lo[1] == pytest.approx(hi[1], abs=1e-9)


def test_raw_sici_satisfies_defining_derivatives():
    """d/dx Si(x) = sin(x)/x and d/dx Ci(x) = cos(x)/x — the defining ODEs.

    Checked by central difference on both sides of the series/CF cutoff, which
    is the strongest implementation-independent evidence that both branches
    compute the actual integrals rather than merely something well-behaved.
    """
    h = 1e-6
    for x in (0.5, 1.5, 3.0, 40.0, 500.0):
        si_hi, ci_hi = _raw_sici(x + h)
        si_lo, ci_lo = _raw_sici(x - h)
        assert (si_hi - si_lo) / (2 * h) == pytest.approx(math.sin(x) / x, rel=1e-6, abs=1e-9)
        assert (ci_hi - ci_lo) / (2 * h) == pytest.approx(math.cos(x) / x, rel=1e-6, abs=1e-9)


def test_raw_sici_large_phi_stays_finite_and_small():
    """Regression: the old truncated power series diverged past |x| ≈ 2π.

    At the coherence values this module actually produces (φ = π/4 + kπ), Ci
    must keep decaying, not explode to ~1e5 as the truncated series did.
    """
    for coherence in (20, 100, 500, 1000):
        phi = coherence_to_phi(coherence)
        si, ci = _raw_sici(phi)
        assert math.isfinite(si) and math.isfinite(ci)
        # Both approach their limits as O(1/φ), so the bound must scale too.
        assert abs(si - math.pi / 2.0) < 2.0 / phi
        assert abs(ci) < 2.0 / phi


# ── sici_axial_decay ──────────────────────────────────────────────────────────

def test_sici_axial_decay_coherence_0_positive():
    """At coherence=0 the axial decay product Si·Ci·tan is positive."""
    phi = coherence_to_phi(0)
    result = sici_axial_decay(phi)
    assert result > 0.0


def test_sici_axial_decay_gamma_scales_linearly():
    """Doubling Γ₀ doubles the axial decay."""
    phi = coherence_to_phi(0)
    v1 = sici_axial_decay(phi, gamma_0=1.0)
    v2 = sici_axial_decay(phi, gamma_0=2.0)
    assert v2 == pytest.approx(2.0 * v1, rel=1e-6)


def test_sici_axial_decay_large_phi_shrinks():
    """At large φ Ci(φ) → 0, so |Δλ_axial| decreases."""
    small = abs(sici_axial_decay(coherence_to_phi(1)))
    large = abs(sici_axial_decay(coherence_to_phi(20)))
    assert large < small


# ── sici_phase_weight ─────────────────────────────────────────────────────────

def test_sici_phase_weight_returns_near_one():
    """The phase weight must stay within [1 − scale, 1 + scale]."""
    lo = 1.0 - _SICI_SCALE_FACTOR
    hi = 1.0 + _SICI_SCALE_FACTOR
    for c in range(12):
        w = sici_phase_weight(c)
        assert lo <= w <= hi, f"sici_phase_weight({c})={w} out of [{lo}, {hi}]"


def test_sici_phase_weight_large_coherence_approaches_one():
    """At large coherence Ci(φ) → 0, so weight → 1.0."""
    for c in (100, 500, 1000):
        w = sici_phase_weight(c)
        assert abs(w - 1.0) < 0.002, f"sici_phase_weight({c})={w} not near 1.0"


def test_sici_phase_weight_coherence_0_above_one():
    """At coherence=0 the axial product is positive so weight > 1.0."""
    assert sici_phase_weight(0) > 1.0


# ── wavefunction_overlap ──────────────────────────────────────────────────────

def test_wavefunction_overlap_identical_vectors_is_one():
    """⟨ψ|ψ⟩² = 1 for any non-zero vector."""
    assert wavefunction_overlap([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0, abs=1e-6)


def test_wavefunction_overlap_orthogonal_vectors_is_zero():
    """⟨ψ_a|ψ_b⟩² = 0 for orthogonal vectors."""
    assert wavefunction_overlap([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_wavefunction_overlap_parallel_scaled_is_one():
    """Scaling a vector does not change the overlap (normalised inner product)."""
    assert wavefunction_overlap([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0, abs=1e-6)


def test_wavefunction_overlap_zero_vector_returns_zero():
    """Zero norm vector → 0.0 (no meaningful state)."""
    assert wavefunction_overlap([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_wavefunction_overlap_mismatched_length_returns_zero():
    """Vectors of different length → 0.0."""
    assert wavefunction_overlap([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_wavefunction_overlap_empty_returns_zero():
    """Empty vectors → 0.0."""
    assert wavefunction_overlap([], []) == 0.0


def test_wavefunction_overlap_range_zero_to_one():
    """Overlap is always in [0, 1] for real vectors."""
    vecs = [
        ([1.0, 0.0], [0.707, 0.707]),
        ([1.0, 1.0, 1.0], [1.0, -1.0, 0.0]),
        ([3.0, 4.0], [4.0, 3.0]),
    ]
    for a, b in vecs:
        ov = wavefunction_overlap(a, b)
        assert 0.0 <= ov <= 1.0


# ── floquet_modulation_factor ─────────────────────────────────────────────────

def test_floquet_modulation_factor_at_zero_is_one():
    """cos(0) = 1 — full coupling at t=0."""
    assert floquet_modulation_factor(0.0, omega=1.0) == pytest.approx(1.0)


def test_floquet_modulation_factor_at_half_period_is_minus_one():
    """cos(ω · π/ω) = cos(π) = −1."""
    omega = 2.5
    assert floquet_modulation_factor(math.pi / omega, omega) == pytest.approx(-1.0, abs=1e-10)


def test_floquet_modulation_factor_period():
    """cos(ω · 2π/ω) = cos(2π) = 1 — full period restores coupling."""
    omega = 3.7
    assert floquet_modulation_factor(2 * math.pi / omega, omega) == pytest.approx(1.0, abs=1e-10)


def test_tantalum_intermediary_binding_is_bounded():
    binding = tantalum_intermediary_binding(
        weyl_phase=2.253,
        coherence=2,
        mesh_alignment=0.63,
        observer_alignment=0.74,
        pulse_weight=1.2058,
    )

    assert binding["intermediary"] == "tantalum"
    assert binding["receiver_material"] == "tantalum"
    assert binding["receiver_role"] == "gpu_receiver_metal"
    assert 0.0 <= binding["pulse_coupling"] <= 1.0
    assert 0.0 <= binding["binding_gain"] <= 1.0
    assert 0.55 <= binding["binding_multiplier"] <= 1.45


def test_tantalum_intermediary_binding_tracks_weyl_alignment():
    aligned = tantalum_intermediary_binding(
        weyl_phase=0.0,
        coherence=1,
        mesh_alignment=0.6,
        observer_alignment=0.6,
        pulse_weight=1.0,
    )
    offset = tantalum_intermediary_binding(
        weyl_phase=math.pi,
        coherence=1,
        mesh_alignment=0.6,
        observer_alignment=0.6,
        pulse_weight=1.0,
    )

    assert aligned["pulse_coupling"] > offset["pulse_coupling"]
    assert aligned["binding_gain"] > offset["binding_gain"]


# ── holographic_entropy ───────────────────────────────────────────────────────

def test_holographic_entropy_scales_with_edges():
    """More boundary edges → higher entropy."""
    assert holographic_entropy(10, 5) > holographic_entropy(5, 5)


def test_holographic_entropy_zero_nodes():
    """n_nodes=0 → S = n_edges (boundary = volume)."""
    assert holographic_entropy(7, 0) == pytest.approx(7.0)


def test_holographic_entropy_non_negative():
    """Entropy must always be ≥ 0."""
    assert holographic_entropy(0, 100) == 0.0
    assert holographic_entropy(50, 200) >= 0.0


# ── metric_perturbation ───────────────────────────────────────────────────────

def test_metric_perturbation_positive_for_positive_mass():
    """h_μν > 0 for mass > 0, r > 0."""
    h = metric_perturbation(1.0e30, 1.0e10)
    assert h > 0.0


def test_metric_perturbation_zero_for_nonpositive_r():
    """h_μν = 0 at/within the event horizon (r ≤ 0)."""
    assert metric_perturbation(1.0e30, 0.0) == 0.0
    assert metric_perturbation(1.0e30, -1.0) == 0.0


def test_metric_perturbation_formula():
    """Verify  h = 2·G·M/(c²·r)."""
    M, r = 2.0e30, 1.0e11
    expected = 2.0 * _G_CONST * M / (_C_CONST ** 2 * r)
    assert metric_perturbation(M, r) == pytest.approx(expected, rel=1e-9)


# ── phase_evolution_total ─────────────────────────────────────────────────────

def test_phase_evolution_total_zero_contributions_is_axial_only():
    """With all δφ contributions = 0, result = axial term only."""
    phi = coherence_to_phi(0)
    axial = sici_axial_decay(phi)
    axial_phase = axial * (2.0 * math.pi)   # gamma_eff defaults to 1.0
    expected = axial_phase
    result = phase_evolution_total(phi)
    assert result == pytest.approx(expected, rel=1e-6)


def test_phase_evolution_total_adds_contributions():
    """δφ contributions are additive."""
    phi = coherence_to_phi(0)
    base = phase_evolution_total(phi)
    with_mu = phase_evolution_total(phi, delta_mu=0.5)
    assert with_mu == pytest.approx(base + 0.5, rel=1e-6)


# ── entropic_bayesian_step ────────────────────────────────────────────────────

def test_entropic_bayesian_step_increases_with_positive_laplacian():
    """Positive Laplacian → terrain increases."""
    phi = coherence_to_phi(0)
    s0 = 1.0
    s1 = entropic_bayesian_step(s0, laplacian_s=1.0, phi=phi, eta_diff=0.05)
    assert s1 > s0


def test_entropic_bayesian_step_decreases_with_negative_laplacian():
    """Sufficiently negative Laplacian dominates → terrain can decrease."""
    phi = coherence_to_phi(0)
    s0 = 1.0
    # Use a very large negative Laplacian to overpower the axial term.
    s1 = entropic_bayesian_step(s0, laplacian_s=-1000.0, phi=phi, eta_diff=0.05)
    assert s1 < s0


def test_entropic_bayesian_step_deterministic():
    """Same inputs always produce the same output."""
    phi = coherence_to_phi(2)
    s1 = entropic_bayesian_step(0.5, 0.1, phi)
    s2 = entropic_bayesian_step(0.5, 0.1, phi)
    assert s1 == s2


# ── ueqgm_coherence_score ─────────────────────────────────────────────────────

def test_ueqgm_coherence_score_missing_entity_returns_zero():
    """Target entity not in DB → 0.0."""
    cn = _ueqgm_db()
    assert ueqgm_coherence_score(cn, "does-not-exist") == 0.0


def test_ueqgm_coherence_score_no_ueqgm_entities_returns_zero():
    """Target exists but no UEQGM-tagged entities in corpus → 0.0."""
    cn = _ueqgm_db()
    _insert_entity(cn, "target-1", "Supply Chain KPI", '{"topic": "OTD"}')
    score = ueqgm_coherence_score(cn, "target-1")
    assert score == 0.0


def test_ueqgm_coherence_score_with_ueqgm_entity_positive():
    """Target overlaps with UEQGM entity → score > 0."""
    cn = _ueqgm_db()
    # Insert a UEQGM-tagged entity that will be matched.
    _insert_entity(
        cn,
        "ueqgm-paper-1",
        "Wavefunction dynamics quantum ueqgm",
        '{"tags": ["ueqgm", "wavefunction", "quantum"], "topic": "quantum dynamics"}',
    )
    # Target entity shares quantum/ueqgm keywords.
    _insert_entity(
        cn,
        "target-q",
        "quantum entropy holographic ueqgm",
        '{"tags": ["ueqgm", "holographic"]}',
    )
    score = ueqgm_coherence_score(cn, "target-q")
    assert score > 0.0


def test_ueqgm_coherence_score_unrelated_target_zero_overlap():
    """Target with no quantum keywords has zero overlap with UEQGM entities."""
    cn = _ueqgm_db()
    _insert_entity(
        cn,
        "ueqgm-entity",
        "quantum wavefunction holographic ueqgm",
        '{"tags": ["ueqgm", "wavefunction"]}',
    )
    _insert_entity(
        cn,
        "sc-entity",
        "inventory OTD delivery",
        '{"topic": "supply chain"}',
    )
    score = ueqgm_coherence_score(cn, "sc-entity")
    # Feature vec for sc-entity is all zeros against UEQGM keywords → 0.0
    assert score == 0.0


def test_ueqgm_coherence_score_bounded():
    """Score must be in [0.0, 1.2] — overlap ∈[0,1] × weight ∈[0.9,1.1]."""
    cn = _ueqgm_db()
    _insert_entity(
        cn,
        "ueqgm-ref",
        "quantum wavefunction holographic ueqgm floquet entanglement topological entropy",
        '{"tags": ["ueqgm"]}',
    )
    _insert_entity(
        cn,
        "target-rich",
        "quantum wavefunction holographic ueqgm floquet entanglement topological entropy",
        '{"tags": ["ueqgm"]}',
    )
    score = ueqgm_coherence_score(cn, "target-rich")
    assert 0.0 <= score <= 1.2


def test_refresh_adaptive_runtime_persists_runtime_state_from_learnings():
    cn = _ueqgm_db()
    _insert_entity(
        cn,
        "ueqgm-runtime-seed",
        "quantum wavefunction holographic ueqgm",
        '{"tags": ["ueqgm", "wavefunction", "quantum"]}',
    )
    cn.execute(
        "INSERT INTO learning_log(logged_at, kind, title, detail, signal_strength) VALUES(?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(),
            "deep_research",
            "Symbiotic mesh observer expansion for UEQGM frontier",
            json.dumps({"note": "mesh-aligned symbiotic observer frontier"}),
            0.92,
        ),
    )
    entirety_state = {
        "observer": 0.81,
        "axes": {
            "vision": 0.9,
            "touch": 0.5,
            "smell": 0.4,
            "body": 0.7,
            "brain": 0.85,
            "perception": 0.95,
        },
        "transaction": {"drive": 0.78},
        "material_bifurcation": {
            "nodal_bifurcation": 0.74,
            "mesh": {"mesh_density": 0.83},
        },
        "pim_planning": {"ss_signal": 0.5, "mm_signal": 0.4, "lt_signal": 0.6},
    }

    runtime = refresh_adaptive_runtime(cn, entirety_state=entirety_state)
    persisted = get_adaptive_runtime(cn)

    assert runtime["active"] is True
    assert persisted["active"] is True
    assert persisted["certainty"] > 0.0
    assert persisted["symbiotic_gain"] > 0.0
    assert persisted["recent_learning_count"] == 1
    assert persisted["ueqgm_entity_count"] == 1
    assert "symbiotic" in persisted["runtime_keywords"]
    assert "mesh" in persisted["runtime_keywords"]
    assert persisted["newness_signal"] > 0.0
    assert persisted["relational_depth"] > 0.0
    assert "symbiotic_gain" in persisted["applied_parameters"]
    assert persisted["parameter_evidence"]["symbiotic_gain"] > persisted["parameter_density_floor"]["symbiotic_gain"]

    row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (_UEQGM_RUNTIME_KEY,)).fetchone()
    assert row is not None


def test_ueqgm_coherence_score_uses_runtime_keywords_to_expand_alignment():
    cn = _ueqgm_db()
    _insert_entity(
        cn,
        "ueqgm-entity",
        "quantum wavefunction holographic ueqgm",
        '{"tags": ["ueqgm", "wavefunction"]}',
    )
    _insert_entity(
        cn,
        "target-runtime",
        "symbiotic mesh observer capability",
        '{"topic": "supply chain symbiotic mesh"}',
    )

    baseline = ueqgm_coherence_score(cn, "target-runtime")

    cn.execute(
        "INSERT INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
        (
            _UEQGM_RUNTIME_KEY,
            json.dumps({
                "active": True,
                "symbiotic_gain": 0.88,
                "runtime_keywords": ["symbiotic", "mesh", "observer", "capability"],
                "axis_drive": {},
            }),
            "2026-05-18T00:00:00+00:00",
        ),
    )
    cn.commit()

    enriched = ueqgm_coherence_score(cn, "target-runtime")
    assert enriched > baseline


def test_refresh_adaptive_runtime_retains_current_parameters_when_evidence_stays_below_corpus_density():
    cn = _ueqgm_db()
    for idx in range(18):
        _insert_entity(
            cn,
            f"ueqgm-dense-{idx}",
            "quantum wavefunction holographic ueqgm entanglement floquet entropy",
            '{"tags": ["ueqgm", "quantum", "wavefunction", "entanglement"]}',
        )

    cn.execute(
        "INSERT INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
        (
            _UEQGM_RUNTIME_KEY,
            json.dumps({
                "active": True,
                "certainty": 0.74,
                "corpus_signal": 0.82,
                "learning_signal": 0.68,
                "newness_signal": 0.66,
                "relational_depth": 0.71,
                "symbiotic_gain": 0.77,
                "phase_weight": 1.04,
                "coherence_depth": 5,
                "phi": 16.493361,
                "terrain_entropy": 0.4421,
                "expansion_pressure": 0.73,
                "mesh_alignment": 0.72,
                "observer_alignment": 0.69,
                "runtime_keywords": ["legacymesh", "stabilityfrontier"],
                "learning_kinds": {"deep_research": 4},
                "axis_drive": {
                    "vision": 0.6,
                    "touch": 0.5,
                    "smell": 0.4,
                    "body": 0.55,
                    "brain": 0.62,
                    "perception": 0.7,
                },
                "parameter_evidence": {
                    "symbiotic_gain": 0.86,
                    "runtime_keywords": 0.84,
                    "coherence_depth": 0.85,
                },
                "parameter_density_floor": {
                    "symbiotic_gain": 0.82,
                    "runtime_keywords": 0.82,
                    "coherence_depth": 0.82,
                },
            }),
            "2026-05-18T00:00:00+00:00",
        ),
    )
    cn.execute(
        "INSERT INTO learning_log(logged_at, kind, title, detail, signal_strength) VALUES(?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(),
            "deep_research",
            "Tentative novel bridge note",
            json.dumps({"note": "speculative bridge"}),
            0.18,
        ),
    )

    entirety_state = {
        "observer": 0.14,
        "axes": {
            "vision": 0.2,
            "touch": 0.2,
            "smell": 0.1,
            "body": 0.2,
            "brain": 0.25,
            "perception": 0.22,
        },
        "transaction": {"drive": 0.12},
        "material_bifurcation": {
            "nodal_bifurcation": 0.16,
            "mesh": {"mesh_density": 0.12},
        },
    }

    refreshed = refresh_adaptive_runtime(cn, entirety_state=entirety_state)

    assert refreshed["symbiotic_gain"] == pytest.approx(0.77)
    assert refreshed["runtime_keywords"] == ["legacymesh", "stabilityfrontier"]
    assert "symbiotic_gain" in refreshed["retained_parameters"]
    assert "runtime_keywords" in refreshed["retained_parameters"]
    assert refreshed["parameter_evidence"]["symbiotic_gain"] <= refreshed["parameter_density_floor"]["symbiotic_gain"]


# ── weyl_scalar_tensor ────────────────────────────────────────────────────────

@pytest.mark.unit
def test_weyl_scalar_tensor_returns_five_scalars():
    result = weyl_scalar_tensor(0.5, 0.4, 20.0, 0.6, 0.75)
    assert len(result) == 5


@pytest.mark.unit
def test_weyl_scalar_tensor_all_zero_inputs():
    psi = weyl_scalar_tensor(0.0, 0.0, 0.0, 0.0, 0.0)
    assert psi == (0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.unit
def test_weyl_scalar_tensor_clips_overflow():
    """Values > 1 or < 0 must be clipped to [0, 1]."""
    psi = weyl_scalar_tensor(2.0, -0.5, 50.0, 1.8, 0.9)
    assert psi[0] == pytest.approx(1.0)   # signal_flux clipped to 1
    assert psi[1] == pytest.approx(0.0)   # topic_entropy clipped to 0
    assert psi[3] == pytest.approx(1.0)   # mesh_alignment clipped to 1
    assert 0.0 <= psi[2] <= 1.0           # corpus_volume compressed via rho/(rho+1)


@pytest.mark.unit
def test_weyl_scalar_tensor_psi2_saturating():
    """Ψ₂ (corpus volume) saturates toward 1 as corpus grows, never exceeds 1."""
    psi_small = weyl_scalar_tensor(0.0, 0.0, 1.0, 0.0, 0.0)
    psi_large = weyl_scalar_tensor(0.0, 0.0, 1_000_000.0, 0.0, 0.0)
    assert psi_small[2] < psi_large[2]
    assert psi_large[2] < 1.0        # asymptotic — never exactly 1 at finite volume
    assert psi_large[2] > 0.999


@pytest.mark.unit
def test_weyl_scalar_tensor_typical_cycle():
    """Typical research-cycle values produce a physically reasonable tensor."""
    # 30 papers, topic entropy 0.85, 42 total corpus items, 60% written, 0.78 remnant
    psi = weyl_scalar_tensor(
        signal_flux=0.30,
        topic_entropy=0.85,
        corpus_volume=42.0,
        mesh_alignment=0.60,
        remnant_score=0.78,
    )
    assert psi[0] == pytest.approx(0.30)
    assert psi[1] == pytest.approx(0.85)
    # Ψ₂ = 42 / 43 ≈ 0.9767
    assert psi[2] == pytest.approx(42.0 / 43.0, abs=1e-6)
    assert psi[3] == pytest.approx(0.60)
    assert psi[4] == pytest.approx(0.78)


@pytest.mark.unit
def test_weyl_scalar_tensor_all_scalars_in_unit_interval():
    for signal, entropy, vol, align, remnant in [
        (0.1, 0.9, 5.0, 0.5, 0.3),
        (0.99, 0.01, 0.0, 0.0, 1.0),
        (0.5, 0.5, 999.0, 0.5, 0.5),
    ]:
        psi = weyl_scalar_tensor(signal, entropy, vol, align, remnant)
        for i, v in enumerate(psi):
            assert 0.0 <= v <= 1.0, f"Ψ{i}={v} out of [0,1] for inputs ({signal},{entropy},{vol},{align},{remnant})"


@pytest.mark.unit
def test_weyl_scalar_tensor_in_math_map():
    from src.quipu.ueqgm_engine import UEQGM_MATH_MAP
    assert "weyl_scalar_tensor" in UEQGM_MATH_MAP
    entry = UEQGM_MATH_MAP["weyl_scalar_tensor"]
    assert "formula" in entry
    assert "Ψ" in entry["formula"]


@pytest.mark.unit
def test_weyl_scalar_tensor_psi2_exact_formula():
    """Ψ₂ = ρ/(ρ+1) exactly for known corpus_volume values."""
    for vol in (0.0, 1.0, 10.0, 100.0):
        psi = weyl_scalar_tensor(0.0, 0.0, vol, 0.0, 0.0)
        expected = vol / (vol + 1.0)
        assert psi[2] == pytest.approx(expected, abs=1e-9), (
            f"Ψ₂ formula mismatch for corpus_volume={vol}: got {psi[2]}, expected {expected}"
        )


@pytest.mark.unit
def test_weyl_scalar_tensor_in_all():
    from src.quipu import ueqgm_engine
    assert "weyl_scalar_tensor" in ueqgm_engine.__all__


# ── GARD CRB + compaction additive tests ──────────────────────────────────────

def test_information_compaction_scalar_floor_bytes_optional():
    """With floor_bytes=None, result is byte-identical to the legacy call."""
    from src.quipu.ueqgm_engine import information_compaction_scalar

    src = 16_492_674_416_640.0   # 15 TB
    mesh = 1_101_884.0

    legacy = information_compaction_scalar(src, mesh)
    explicit_none = information_compaction_scalar(src, mesh, floor_bytes=None)
    assert legacy == explicit_none


def test_information_compaction_scalar_custom_floor_differs_from_default():
    """Providing a floor_bytes different from default changes the scalar value."""
    from src.quipu.ueqgm_engine import information_compaction_scalar

    src = 16_492_674_416_640.0
    mesh = 1_101_884.0

    default_val = information_compaction_scalar(src, mesh)
    # floor=5 bytes (smaller than default 50) → R_max larger → denominator larger → scalar lower
    small_floor = information_compaction_scalar(src, mesh, floor_bytes=5.0)
    # floor=500 bytes (larger than default 50) → R_max smaller → scalar higher
    large_floor = information_compaction_scalar(src, mesh, floor_bytes=500.0)
    assert small_floor < default_val < large_floor


def test_information_compaction_scalar_gard_returns_in_unit_interval():
    """information_compaction_scalar_gard output is in [0, 1]."""
    from src.quipu.ueqgm_engine import information_compaction_scalar_gard

    val = information_compaction_scalar_gard(
        16_492_674_416_640.0, 1_101_884.0, sigma=0.05
    )
    assert 0.0 <= val <= 1.0


def test_mesh_compaction_summary_canonical_ratio_unchanged():
    """Additive gard_crb key must not change compaction_ratio == 14_967_705."""
    from src.quipu.ueqgm_engine import mesh_compaction_summary

    summary = mesh_compaction_summary()
    assert summary["compaction_ratio"] == 14_967_705


def test_mesh_compaction_summary_gard_crb_present():
    """gard_crb is present and has the expected keys."""
    from src.quipu.ueqgm_engine import mesh_compaction_summary

    summary = mesh_compaction_summary()
    assert "gard_crb" in summary
    gard_crb = summary["gard_crb"]
    assert gard_crb.get("sigma") == 0.05
    assert "cramer_rao_var" in gard_crb
    assert "floor_bytes" in gard_crb
    assert "compaction_scalar_at_crb_floor" in gard_crb
    assert 0.0 <= gard_crb["compaction_scalar_at_crb_floor"] <= 1.0


def test_information_compaction_scalar_gard_in_all():
    from src.quipu import ueqgm_engine
    assert "information_compaction_scalar_gard" in ueqgm_engine.__all__


# ── Physics-claim de-naming: aliases and behavioural equivalence ─────────────

def test_deprecated_physics_aliases_resolve_to_renamed_functions():
    """The old physics-flavoured names stay importable for one release.

    Each asserted a physical quantity the arithmetic does not compute (a
    quantum overlap, a Bekenstein-Hawking entropy, Hawking remnants, Floquet
    engineering, a tantalum receiver), so the functions were renamed to
    describe what they actually do.
    """
    from src.quipu import ueqgm_engine as u

    assert u.wavefunction_overlap is u.cosine_similarity_squared
    assert u.floquet_modulation_factor is u.cosine_modulation
    assert u.holographic_entropy is u.edge_per_node_ratio
    assert u.hawking_information_remnant_score is u.corpus_coverage_score
    assert u.tantalum_intermediary_binding is u.intermediary_binding_profile


def test_renaming_did_not_change_any_numeric_behaviour():
    """Pure rename: every value must be identical to the pre-rename output."""
    from src.quipu import ueqgm_engine as u

    assert u.cosine_similarity_squared([1, 2, 3], [2, 4, 6]) == 1.0
    assert u.cosine_similarity_squared([1, 0], [0, 1]) == 0.0
    assert u.cosine_modulation(0.0, 3.0) == 1.0
    assert u.edge_per_node_ratio(10, 4) == 2.0
    assert u.edge_per_node_ratio(7, 0) == 7.0
    assert u.corpus_coverage_score(16, 4) == pytest.approx(64 / 65)
    assert u.corpus_coverage_score(0, 4) == 0.0


def test_renamed_functions_are_exported():
    from src.quipu import ueqgm_engine

    for name in (
        "cosine_similarity_squared",
        "cosine_modulation",
        "edge_per_node_ratio",
        "corpus_coverage_score",
        "intermediary_binding_profile",
    ):
        assert name in ueqgm_engine.__all__, name


def test_math_map_no_longer_asserts_physical_quantities():
    """UEQGM_MATH_MAP must describe the arithmetic, under the current names."""
    from src.quipu.ueqgm_engine import UEQGM_MATH_MAP

    for renamed in ("cosine_similarity_squared", "edge_per_node_ratio", "corpus_coverage_score"):
        assert renamed in UEQGM_MATH_MAP
    for retired in ("wavefunction_overlap", "holographic_entropy",
                    "hawking_information_remnant_score"):
        assert retired not in UEQGM_MATH_MAP


def test_intermediary_binding_keeps_receiver_lookup_labels():
    """The receiver_* strings are consumed by asset_resource_mesh part matching.

    They are lookup keys into a real parts catalogue, so the de-naming pass
    must not drop them even though the surrounding physics claim was removed.
    """
    from src.quipu.ueqgm_engine import intermediary_binding_profile

    profile = intermediary_binding_profile(
        weyl_phase=0.5, coherence=2, mesh_alignment=0.6, observer_alignment=0.4,
    )
    assert profile["receiver_material"] == "tantalum"
    assert 0.0 <= profile["binding_gain"] <= 1.0
    assert profile["binding_multiplier"] >= 0.55
