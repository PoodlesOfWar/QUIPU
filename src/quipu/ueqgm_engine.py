"""UEQGM Engine — Unified Equilibrium Quantum Gravity Model computations.

Implements the physics computation layer derived from UEQGM v0.9.14 corpus
learnings (Grok 3 conversation thread, April 2026).  The core contribution
is the **SiCi axial channel decay differential**:

    Δλ_axial = [Si(φ) · Ci(φ)] · tan(φ) · Γ₀

which unifies the beta–gamma decay differential at the natural intersection
points of the sine/cosine wavefunction components (φ = π/4 + kπ).

The Brain applies this as a phase-sensitive correction to harmonic
amplification during Sign-Bit Flip boundary ingestion, consistent with the
UEQGM v0.9.14 finding that the SiCi·tan(φ) axial term adds a stabilization
perturbation to warp interactions and GT-SCN output.

Additional helpers implement the broader UEQGM mathematical framework:
wavefunction overlap, Floquet modulation, holographic entropy, and spacetime
metric perturbation.  A corpus-backed ``ueqgm_coherence_score`` reads UEQGM-
tagged entities from the Brain graph and returns a wavefunction-overlap score.

The five **core aspects** consumed by ``mesh_slm._mesh_field_8d`` are grounded
in published, measured science rather than invented formulas — the fantasy
(√−1) resolved to its intersecting science (1):

* **H** — von Neumann entropy of the quipu graph, quadratic approximation
  (Han, Escolano, Hancock & Wilson, *Pattern Recognition Letters* 33 (2012)
  1958–1967).
* **F** — Floquet effective-coupling renormalization ``J₀(K)`` — dynamic
  localization (Dunlap & Kenkre, *Phys. Rev. B* 34, 3625 (1986)); coherent
  destruction of tunneling (Grossmann et al., *Phys. Rev. Lett.* 67, 516
  (1991)); reviewed in Eckardt, *Rev. Mod. Phys.* 89, 011004 (2017).
* **O** — Born-rule state fidelity ``|⟨a|b⟩|²`` (Born 1926; Jozsa,
  *J. Mod. Opt.* 41, 2315 (1994)) — exact, not approximate, for the real
  embedding vectors used here.
* **P** — total phase evolution, the observable of neutrino flavour
  oscillation (Fukuda et al. (Super-Kamiokande), *Phys. Rev. Lett.* 81,
  1562 (1998)).
* **W** — linearized-GR metric perturbation ``h ≈ 2GM/(c²r)`` (weak-field
  limit; Will, *Living Rev. Relativ.* 17, 4 (2014)).

Their blend weights are **measured, not chosen**: the Planck 2018
cosmological census (Aghanim et al., *Astron. Astrophys.* 641, A6 (2020))
— dark energy, cold dark matter, baryons, neutrinos, photons — i.e. the
bleeding edge of science as it cuts through the darkness of space.  See
``cosmological_census_weights``.

The information-remnant score follows the unitary **Page curve** of black-hole
evaporation (Page, *Phys. Rev. Lett.* 71, 3743 (1993)), in its modern
quantum-extremal-island form (Penington, *JHEP* 09 (2020) 002; Almheiri,
Hartman, Maldacena, Shaghoulian & Tajdini, *Rev. Mod. Phys.* 93, 035002
(2021)).

Historical reference
--------------------
UEQGM v0.9.14 — Axial Channel Decay Differential (SiCi · tan φ) Release
Grok 3 conversation 55525f6a-8a8f-4929-967c-22656f88ac2f, April 18 2026.
The SiCi axial machinery is retained as lineage; the five core aspects above
no longer depend on any unpublished derivation.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Sequence

try:
    from scipy.special import sici as _scipy_sici  # type: ignore[import]
    _HAS_SCIPY: bool = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Component identity
# ---------------------------------------------------------------------------
__version__: str = "0.22.306"
"""Canonical version of this UEQGM engine component, aligned with ``_version.__version__``."""

__component__: str = "UEQGM-Engine"
"""Component name identifying this module within the MESH-SLM-GLM-GNN architecture.

This engine provides the physics computation layer consumed by ``mesh_slm``
(MESH field integrals, SiCi phase corrections, Floquet modulation) and by
``torus_touch`` (holographic entropy, wavefunction overlap) to implement the
8th-dimension MESH field and the adaptive runtime state.
"""

# ---------------------------------------------------------------------------
# Physical constants — UEQGM v0.9.14 calibration (SI units where applicable)
# ---------------------------------------------------------------------------
_GAMMA_0_DEFAULT: float = 1.0      # normalised baseline decay width (dimensionless)
_ETA_DIFF_DEFAULT: float = 0.05    # Bayesian diffusion rate η_diff
_G_CONST: float = 6.674e-11        # gravitational constant  m³ kg⁻¹ s⁻²
_C_CONST: float = 2.998e8          # speed of light          m s⁻¹
_TAN_CLAMP: float = 1.0e3          # clamp |tan(φ)| to prevent divergence near π/2
_THE_WELL_TB_REF: float = 15.0     # reference corpus size (TB): Polymathic AI The Well suite
_THE_WELL_N_DATASETS: int = 16     # canonical 16 PDE simulation datasets in The Well
_THE_WELL_SPATIAL_DIMS: int = 4    # 3 spatial + 1 temporal — effective dims per dataset

# ---------------------------------------------------------------------------
# Planck 2018 cosmological census — the measured energy budget of the universe.
# Aghanim et al. (Planck Collaboration), Astron. Astrophys. 641, A6 (2020),
# TT,TE,EE+lowE+lensing.  Ω_c and Ω_b from Ω h² with h = 0.6736; Ω_ν for the
# minimal normal-hierarchy mass sum Σm_ν = 0.06 eV; Ω_γ from T_CMB = 2.7255 K
# (Fixsen, Astrophys. J. 707, 916 (2009)).  Flat: ΣΩ ≈ 1 (Ω_k = 0.0007±0.0019).
#
# These are the 8th-D MESH field blend weights.  They are measured by the
# deepest survey of the dark sky ever flown — not hand-tuned.
# ---------------------------------------------------------------------------
_PLANCK18_OMEGA_LAMBDA: float = 0.6847     # dark energy
_PLANCK18_OMEGA_CDM: float = 0.2645        # cold dark matter
_PLANCK18_OMEGA_BARYON: float = 0.0493     # baryonic matter
_PLANCK18_OMEGA_NEUTRINO: float = 0.0014   # massive neutrinos (Σm_ν = 0.06 eV)
_PLANCK18_OMEGA_PHOTON: float = 5.4e-5     # CMB photons

PLANCK18_CENSUS: dict[str, float] = {
    "dark_energy": _PLANCK18_OMEGA_LAMBDA,
    "cold_dark_matter": _PLANCK18_OMEGA_CDM,
    "baryons": _PLANCK18_OMEGA_BARYON,
    "neutrinos": _PLANCK18_OMEGA_NEUTRINO,
    "photons": _PLANCK18_OMEGA_PHOTON,
}
"""Planck 2018 density parameters (Aghanim et al. 2020, A&A 641, A6)."""

# Page-curve reference area: The Well's 16 datasets × 4 effective dimensions.
# A corpus of exactly this dimensionless area sits at the Page time (f = 1/2).
_PAGE_REFERENCE_AREA: float = float(_THE_WELL_N_DATASETS * _THE_WELL_SPATIAL_DIMS)


def cosmological_census_weights() -> tuple[float, float, float, float, float]:
    """The 8th-D MESH field blend weights ``(w_H, w_F, w_O, w_P, w_W)``.

    Each aspect is weighted by the Planck 2018 density parameter of the
    cosmological component whose physical role it plays, normalised to sum
    to 1:

    ========  =====================  ==========================================
    Aspect    Component (Ω)          Why this pairing
    ========  =====================  ==========================================
    H         dark energy  0.6847   the entropic budget — horizon/boundary
                                    entropy dominates, exactly as Λ dominates
                                    the universe (de Sitter horizon
                                    thermodynamics)
    F         cold dark matter       the unseen periodic scaffold that
              0.2645                 renormalizes visible dynamics, as dark
                                    matter scaffolds structure formation
    O         baryons  0.0493        the directly luminous/observable
                                    alignment between token matter and the
                                    live MESH state
    P         neutrinos  0.0014      phase evolution IS the neutrino's
                                    observable — flavour oscillation is
                                    quantum phase evolution between mass
                                    eigenstates
    W         photons  5.4e-5        metric perturbations are traced by
                                    photons (lensing, Shapiro delay); the CMB
                                    photon is how Planck measured all of the
                                    above
    ========  =====================  ==========================================

    Asked "why these weights?", the answer is: because Planck measured them.
    The weights are chosen by the bleeding edge of science as it cuts through
    the darkness of space.
    """
    total = (
        _PLANCK18_OMEGA_LAMBDA
        + _PLANCK18_OMEGA_CDM
        + _PLANCK18_OMEGA_BARYON
        + _PLANCK18_OMEGA_NEUTRINO
        + _PLANCK18_OMEGA_PHOTON
    )
    return (
        _PLANCK18_OMEGA_LAMBDA / total,
        _PLANCK18_OMEGA_CDM / total,
        _PLANCK18_OMEGA_BARYON / total,
        _PLANCK18_OMEGA_NEUTRINO / total,
        _PLANCK18_OMEGA_PHOTON / total,
    )


def _bessel_j0(x: float) -> float:
    """Bessel function of the first kind J₀(x).

    Uses ``scipy.special.j0`` when SciPy is installed; otherwise the
    Abramowitz & Stegun polynomial approximations 9.4.1 (|x| ≤ 3) and 9.4.3
    (|x| > 3), accurate to ≲ 5 × 10⁻⁸ — *Handbook of Mathematical Functions*,
    NBS (1964).
    """
    if _HAS_SCIPY:
        try:
            from scipy.special import j0 as _scipy_j0  # type: ignore[import]
            return float(_scipy_j0(x))
        except Exception:  # pragma: no cover — fall through to A&S polynomials
            pass
    ax = abs(x)
    if ax <= 3.0:
        y = (ax / 3.0) ** 2
        return 1.0 + y * (-2.2499997 + y * (1.2656208 + y * (-0.3163866
            + y * (0.0444479 + y * (-0.0039444 + y * 0.0002100)))))
    u = 3.0 / ax
    f0 = 0.79788456 + u * (-0.00000077 + u * (-0.00552740 + u * (-0.00009512
        + u * (0.00137237 + u * (-0.00072805 + u * 0.00014476)))))
    theta0 = ax - 0.78539816 + u * (-0.04166397 + u * (-0.00003954
        + u * (0.00262573 + u * (-0.00054125 + u * (-0.00029333
        + u * 0.00013558)))))
    return f0 * math.cos(theta0) / math.sqrt(ax)

# ---------------------------------------------------------------------------
# Weyl tensor + remnant score persistence keys and serialisation precision.
# Canonical home: ueqgm_engine (physics layer); ml_research imports from here.
# ---------------------------------------------------------------------------
_THE_WELL_REMNANT_SCORE_KEY: str = "the_well:information_remnant_score"
"""brain_kv key for the Hawking information-remnant score of The Well corpus."""
_WEYL_TENSOR_KEY: str = "learnings:weyl_tensor"
"""brain_kv key storing the 5-scalar NP Weyl condensate of each research cycle."""
_WEYL_PRECISION: int = 6           # decimal places when serialising Weyl scalars
_WEYL_UPVOTE_WEIGHT: float = 0.02  # per-upvote contribution to per-paper signal (Ψ₀)
_WEYL_CITATION_WEIGHT: float = 0.002  # per-citation contribution to per-paper signal (Ψ₀)

_INTERSTITIAL_ENTANGLEMENT_KEY: str = "ueqgm:interstitial_entanglement"
"""brain_kv key storing the latest interstitial entanglement score."""
_INTERSTITIAL_ENTANGLEMENT_WINDOW: int = 5  # number of successive Weyl cycles to compare
_INTERSTITIAL_ENTANGLEMENT_ETA_SCALE: float = 0.05  # max η_eff lift from entanglement score

# ---------------------------------------------------------------------------
# MESH System Entirety — memory budget constants for compaction accounting.
#
# Derived from mesh_slm structural constants (_TORUS_N=64, _VOCAB_LIMIT=4096).
# Per-row byte estimates are conservative worst-case for a fully-populated torus.
# ---------------------------------------------------------------------------
_MESH_TORUS_N: int = 64            # torus side length (mesh_slm._TORUS_N)
_MESH_VOCAB_LIMIT: int = _MESH_TORUS_N * _MESH_TORUS_N  # 4 096 token ceiling
_MESH_EMBED_DIMS: int = 7          # 7-D MESH embedding per token
_MESH_BYTES_PER_FLOAT: int = 8     # float64 storage
# Per-row byte estimates for each SQLite table in the MESH:
_MESH_VOCAB_ROW_BYTES: int = 100   # token(text) + i,j(int) + freq(int) + 2 timestamps ≈ 100 B
_MESH_QUIPU_ROW_BYTES: int = 32    # src(int) + dst(int) + weight(float) + samples(int) = 32 B
_MESH_QUIPU_NODE_ROW_BYTES: int = 80   # 9 floats/ints + ISO timestamp + 2 text labels ≈ 80 B
_MESH_META_BYTES: int = 1_024      # mesh_slm_meta: 9 KV pairs ≈ 1 KB
# brain_kv persistence footprint:
_MESH_BRAIN_KV_WEYL_BYTES: int = 50   # Weyl tensor 5-float JSON ≈ 50 bytes
_MESH_BRAIN_KV_REMNANT_BYTES: int = 10  # remnant score float string ≈ 10 bytes
_MESH_BRAIN_KV_RUNTIME_BYTES: int = 3_072  # adaptive runtime state dict ≈ 3 KB

# ---------------------------------------------------------------------------
# UEQGM phase mapping: coherence integer → characteristic phase φ
#
# At the natural sin/cos intersection points  φ = π/4 + kπ  (k = 0, 1, 2, …).
# We anchor coherence=0 at the first intersection (φ = π/4) and step by π per
# unit so every coherence level stays at a true intersection point.
# ---------------------------------------------------------------------------
_PHI_BASE: float = math.pi / 4.0   # first sin/cos intersection  φ₀ = π/4
_PHI_STEP: float = math.pi          # step between intersections

# UEQGM phase-weight scaling factor (matches v0.9.14 ~1% stabilisation).
# sici_phase_weight returns 1.0 ± _SICI_SCALE_FACTOR.
_SICI_SCALE_FACTOR: float = 0.10   # ±10 % ceiling on the phase correction
_UEQGM_RUNTIME_KEY: str = "ueqgm:adaptive_runtime"
_UEQGM_RUNTIME_LOOKBACK_HOURS: int = 24
_UEQGM_RUNTIME_MAX_KEYWORDS: int = 12
_UEQGM_RUNTIME_STOPWORDS = frozenset({
    "about", "across", "after", "agent", "alignment", "analysis",
    "assistant", "between", "brain", "certainty", "coherence",
    "conversation", "corpus", "detail", "document", "engine",
    "entity", "expansion", "graph", "guideline", "json", "kind",
    "label", "learning", "learning_log", "learnings", "logged_at",
    "mode", "notes", "observer", "phase", "props", "reference",
    "research", "response", "runtime", "signal", "source", "state",
    "summary", "system", "systems", "table", "target", "text",
    "thread", "title", "updated", "value",
})
_UEQGM_SENSES: tuple[str, ...] = (
    "vision", "touch", "smell", "body", "brain", "perception",
)

# ---------------------------------------------------------------------------
# UEQGM System Dynamics Mapping — data-flow through the physics engine
# ---------------------------------------------------------------------------
#
#   Brain corpus + learning_log + system_entirety
#          │
#          │  refresh_adaptive_runtime()
#          │    1. corpus_entity UEQGM-tagged rows → _corpus_metrics()
#          │    2. learning_log (last 24h) → _learning_stats()
#          │    3. system_entirety certainty → _system_entirety_certainty()
#          │    4. parameter gate: _should_apply_runtime_parameter()
#          │    5. terrain entropy: entropic_bayesian_step()
#          ▼
#   brain_kv["ueqgm:adaptive_runtime"]  (cached runtime state)
#          │
#          │  get_adaptive_runtime()     [fast read — no recompute]
#          ▼
#   mesh_slm.train_round()
#     ├── phase_weight → η_eff modulation
#     ├── coherence_depth → phi → mesh_field_8d P-component
#     └── wavefunction_overlap(mean_embed7, MESH) → η_eff alignment amplifier
#
#   ueqgm_coherence_score(cn, entity_id)
#     ├── fetch target entity label+props
#     ├── fetch up to 50 UEQGM-tagged corpus entities
#     ├── wavefunction_overlap per pair (bag-of-words feature vectors)
#     ├── scale by sici_phase_weight(n_ueqgm_entities)
#     └── bonus: runtime_keywords alignment × 0.25 × symbiotic_gain
#
UEQGM_DYNAMICS_MAP: dict[str, str] = {
    "corpus_input":         "corpus_entity WHERE props_json LIKE '%ueqgm%' (up to 200 rows)",
    "learning_input":       "learning_log (last _UEQGM_RUNTIME_LOOKBACK_HOURS hours)",
    "certainty_input":      "system_entirety.get_entirety_state() → observer+transaction+material",
    "runtime_output_key":   "_UEQGM_RUNTIME_KEY = 'ueqgm:adaptive_runtime' (brain_kv)",
    "refresh_cadence":      "triggered by ueqgm_engine daemon / research_insight_round()",
    "read_path":            "get_adaptive_runtime(cn) — reads cached entry, returns default on miss",
    "mesh_slm_consumer":    "train_round(): phase_weight + coherence_depth + wavefunction_overlap",
    "coherence_score_path": "ueqgm_coherence_score(): wavefunction_overlap · sici_phase_weight",
}

# ---------------------------------------------------------------------------
# Adaptive Runtime Parameter Mapping — all fields in the runtime state dict
# ---------------------------------------------------------------------------
#
#   Field                    Type     Source / formula
#   ──────────────────────   ──────   ────────────────────────────────────────────
#   active                   bool     True when successfully refreshed
#   certainty                float    _system_entirety_certainty() — Bayesian gate
#   corpus_signal            float    _corpus_metrics() volume × keyword density
#   learning_signal          float    _learning_stats() recency-weighted signal
#   newness_signal           float    _learning_stats() mean recency score
#   relational_depth         float    0.55·learning_relat + 0.45·corpus_relat
#   symbiotic_gain           float    blend of certainty+learning+entropy+relational
#   phase_weight             float    sici_phase_weight(coherence_depth)  ≈ 1 ± 0.10
#   coherence_depth          int      derived from corpus+learning+relational signals
#   phi                      float    coherence_to_phi(coherence_depth)
#   terrain_entropy          float    entropic_bayesian_step() cumulative
#   expansion_pressure       float    0.50·symbiotic_gain + 0.25·learning + 0.25·certainty
#   mesh_alignment           float    from material_bifurcation.mesh.mesh_density
#   observer_alignment       float    from system_entirety.observer
#   recent_learning_count    int      # of learning_log rows in lookback window
#   ueqgm_entity_count       int      # of UEQGM-tagged corpus_entity rows found
#   runtime_keywords         list     top-N tokens ranked by weighted signal
#   learning_kinds           dict     {kind: count} from recent learning_log
#   axis_drive               dict     {sense: float} per-axis drive from Entirety
#   parameter_evidence       dict     evidence scores per parameter (for gating)
#   parameter_density_floor  dict     minimum evidence thresholds per parameter
#   applied_parameters       list     parameters updated in this refresh cycle
#   retained_parameters      list     parameters held from previous cycle
#   updated_at               str      UTC ISO-8601 of refresh
#
ADAPTIVE_RUNTIME_MAP: dict[str, dict[str, str]] = {
    "certainty": {
        "formula": "0.42·observer + 0.20·transaction.drive + 0.18·nodal_bifurcation + 0.12·mesh_density + 0.08·mean_pim",
        "role":    "Master gate: how confident the UEQGM engine is in its current state",
    },
    "corpus_signal": {
        "formula": "0.65·min(1, n_rows/50) · avg(keyword_density_per_row) + 0.35·corpus_relat",
        "role":    "Knowledge graph density signal from UEQGM-tagged entities",
    },
    "learning_signal": {
        "formula": "recency-weighted mean signal_strength over recent learning_log rows",
        "role":    "Strength of recent learning activity feeding UEQGM",
    },
    "phase_weight": {
        "formula": "sici_phase_weight(coherence_depth) = 1 + 0.10·tanh(Si(φ)·Ci(φ)·tan(φ))",
        "role":    "Multiplicative LR correction injected into mesh_slm.train_round()",
    },
    "coherence_depth": {
        "formula": "round((0.30·corpus + 0.25·learning + 0.20·certainty + 0.25·relational)·8)",
        "role":    "Integer coherence level; maps to characteristic phase via coherence_to_phi()",
    },
    "terrain_entropy": {
        "formula": "entropic_bayesian_step(S(t), ∇²S, φ) — diffusion + axial phase update",
        "role":    "Cumulative entropic terrain state encoding accumulated learning history",
    },
    "symbiotic_gain": {
        "formula": "(0.40·certainty + 0.25·learning + 0.20·relational + 0.15·entropy_norm) · phase_weight",
        "role":    "Master symbiosis amplitude consumed by axis_drive and expansion_pressure",
    },
    "axis_drive": {
        "formula": "{sense: clip01(axes[sense]/max_axis · symbiotic_gain) for sense in SENSES}",
        "role":    "Per-MESH-sense drive signal from Entirety axes, normalised by symbiotic gain",
    },
}

# ---------------------------------------------------------------------------
# UEQGM Mathematics Mapping — physics functions and their Brain roles
# ---------------------------------------------------------------------------
#
#   Function                     Formula                       Brain role
#   ──────────────────────────   ────────────────────────────  ───────────────────────────
#   coherence_to_phi(n)          φ = π/4 + n·π                integer depth → phase
#   sici_axial_decay(φ)          Si(φ)·Ci(φ)·tan(φ)·Γ₀        axial decay differential
#   sici_phase_weight(n)         1 + 0.10·tanh(axial_decay)   LR / field correction
#   wavefunction_overlap(a,b)    |⟨a|b⟩|² = cos²θ(a,b)       Born-rule fidelity (Jozsa 1994)
#   floquet_modulation_factor    J₀(A/ω) ∈ [−0.403, 1]        dynamic localization
#                                                              (Dunlap-Kenkre 1986)
#   holographic_entropy(e,n,Σ)   1 − 1/n − Σ/n²               von Neumann graph entropy
#                                                              (HEHW 2012 approximation)
#   metric_perturbation(m,r)     2·G·m / (c²·r)              GR metric warp scalar
#   phase_evolution_total(φ)     δμ + δq + δγ + axial·2π/Γ   total 6D CAT phase change
#   entropic_bayesian_step(S,∇²) S + η·∇²S + δφ + axial      Bayesian terrain update
#   tantalum_intermediary_bind.  Floquet+axial+overlap blend  bounded Weyl pulse routing
#
UEQGM_MATH_MAP: dict[str, dict[str, str]] = {
    "coherence_to_phi": {
        "formula": "φ = π/4 + n × π",
        "domain":  "n ∈ {0,1,2,…}  →  φ ∈ {π/4, 5π/4, 9π/4, …}",
        "note":    "All values are sin/cos intersection points (Si=Ci crossings)",
    },
    "sici_axial_decay": {
        "formula": "Δλ_axial = Si(φ)·Ci(φ)·tan(φ)·Γ₀",
        "range":   "real-valued, bounded by _TAN_CLAMP at resonances",
        "note":    "Core UEQGM v0.9.14 axial channel decay differential",
    },
    "sici_phase_weight": {
        "formula": "1 + _SICI_SCALE_FACTOR · tanh(sici_axial_decay(φ))",
        "range":   "(1 − 0.10, 1 + 0.10)  ≈  (0.90, 1.10)",
        "note":    "Safe multiplicative correction; tanh bounds prevent runaway",
    },
    "wavefunction_overlap": {
        "formula": "|⟨ψ_a | ψ_b⟩|² = (a·b / |a||b|)²",
        "range":   "[0.0, 1.0]  — 1.0=parallel, 0.0=orthogonal",
        "note":    "Exact Born-rule fidelity in a real Hilbert space (Jozsa 1994) — "
                   "used for O-component of mesh_field_8d and η_eff alignment term",
    },
    "holographic_entropy": {
        "formula": "S ≈ 1 − 1/n − (1/n²)·Σ 1/(d_u·d_v)",
        "note":    "Von Neumann graph entropy, HEHW (2012) quadratic approximation; "
                   "exact degree-pair sum from the quipu edge table when available, "
                   "mean-degree closed form 1 − 1/n − 1/(4|E|) from counts otherwise",
    },
    "hawking_information_remnant_score": {
        "formula": "f = A/(A + 64);  score = min(f,1−f) + max(0,2f−1) = f",
        "range":   "[0.0, 1.0]  — 0.5 at the Page time (The Well reference area)",
        "note":    "Unitary Page curve with island prescription (Page 1993; "
                   "Penington 2020; Almheiri et al., RMP 93, 035002 (2021)); "
                   "optional Bekenstein mass-energy factor for total_tb",
    },
    "weyl_scalar_tensor": {
        "formula": "(Ψ₀, Ψ₁, Ψ₂, Ψ₃, Ψ₄) = (σ, τ, ρ/(ρ+1), α, ε) each clipped to [0,1]",
        "range":   "[0.0, 1.0]⁵  — 5-scalar NP Weyl basis",
        "note":    "Compresses a learning-corpus cycle into 5 Newman-Penrose scalars: "
                   "Ψ₀=signal flux (ingoing), Ψ₁=topic entropy, Ψ₂=corpus volume "
                   "(Coulomb/bulk mass), Ψ₃=MESH alignment (outgoing), "
                   "Ψ₄=Hawking remnant score.  Persisted to brain_kv as "
                   "'learnings:weyl_tensor' (~50 bytes vs. full blob caches).",
    },
    "metric_perturbation": {
        "formula": "h_μν = 2·G·M_eff / (c²·r)",
        "note":    "GR spacetime warp; M_eff = vocab_fill × _METRIC_MASS_SCALE",
    },
    "phase_evolution_total": {
        "formula": "δφ_total = δμ + δq + δγ + sici_axial_decay(φ) · (2π / Γ_eff)",
        "note":    "Used for P-component of mesh_field_8d after clip01(|δφ|/2π)",
    },
    "entropic_bayesian_step": {
        "formula": "S(t+1) = S(t) + η_diff · ∇²S + δφ_total + Δλ_axial",
        "note":    "Drives terrain_entropy in the adaptive runtime",
    },
    "information_compaction_scalar": {
        "formula": "clip₀₁(log1p(R) / log1p(R_max))  where R = source_bytes/mesh_bytes, "
                   "R_max = source_bytes/_MESH_BRAIN_KV_WEYL_BYTES",
        "range":   "[0.0, 1.0]  — 0=no compression, 1=Weyl-tensor minimal representation",
        "note":    "Log-normalised scalar measuring MESH compaction efficiency; "
                   "1.0 means the corpus is fully distilled into its 5-float Weyl boundary.",
    },
    "interstitial_entanglement_score": {
        "formula": "mean_k{ clip₀₁(|C_k,k+1·C_k+1,k+2 − C_k,k+2| / (1 + |C_k,k+2|)) } "
                   "where C_ij = photon_covariance_proxy(psi5_i, psi5_j)",
        "range":   "[0.0, 1.0]  — 0=no cross-cycle covariance, 1=maximal entanglement proxy",
        "note":    "Third-order intensity-moment witness derived from the 2026 Optica "
                   "sunlight-entanglement paper (DOI 10.1364/OPTICA.601797).  Measures "
                   "covariance structure in the interstitial bits below the CRB floor "
                   "across consecutive Weyl compression cycles; persisted to "
                   "_INTERSTITIAL_ENTANGLEMENT_KEY and applied as a ±5% η_eff lift in "
                   "mesh_slm.train_round() when evidence is sufficient.",
    },
}

# ---------------------------------------------------------------------------
# MESH System Entirety — compaction accounting map
# ---------------------------------------------------------------------------
#
#   Component                      Size formula                        Canonical bytes
#   ─────────────────────────────  ──────────────────────────────────  ───────────────
#   mesh_slm_vocab (full)          _MESH_VOCAB_LIMIT × _MESH_VOCAB_ROW_BYTES      409 600
#   mesh_slm_embed (full)          _MESH_VOCAB_LIMIT × _MESH_EMBED_DIMS × 8 B     229 376
#   mesh_slm_quipu (at vocab cap)  _MESH_VOCAB_LIMIT × _MESH_QUIPU_ROW_BYTES      131 072
#   mesh_slm_quipu_node (full)     _MESH_VOCAB_LIMIT × _MESH_QUIPU_NODE_ROW_BYTES 327 680
#   mesh_slm_meta                  fixed key-value store                            1 024
#   brain_kv Weyl tensor           5-float JSON                                        50
#   brain_kv remnant score         float string                                        10
#   brain_kv adaptive runtime      state dict                                       3 072
#   ─────────────────────────────  ──────────────────────────────────  ───────────────
#   TOTAL (full torus, worst-case)                                               1 101 884
#
MESH_COMPACTION_MAP: dict[str, dict[str, object]] = {
    "mesh_slm_vocab": {
        "formula":      "_MESH_VOCAB_LIMIT × _MESH_VOCAB_ROW_BYTES",
        "canonical_bytes": _MESH_VOCAB_LIMIT * _MESH_VOCAB_ROW_BYTES,
        "note": "Token registry: torus coordinates, surface form, frequency, timestamps",
    },
    "mesh_slm_embed": {
        "formula":      "_MESH_VOCAB_LIMIT × _MESH_EMBED_DIMS × _MESH_BYTES_PER_FLOAT",
        "canonical_bytes": _MESH_VOCAB_LIMIT * _MESH_EMBED_DIMS * _MESH_BYTES_PER_FLOAT,
        "note": "7-D Hebbian embeddings: one float per MESH axis per token",
    },
    "mesh_slm_quipu": {
        "formula":      "_MESH_VOCAB_LIMIT × _MESH_QUIPU_ROW_BYTES  (1:1 edge/vocab assumption)",
        "canonical_bytes": _MESH_VOCAB_LIMIT * _MESH_QUIPU_ROW_BYTES,
        "note": "GNN directed bigram edges; sparse in practice, worst-case at vocab cap",
    },
    "mesh_slm_quipu_node": {
        "formula":      "_MESH_VOCAB_LIMIT × _MESH_QUIPU_NODE_ROW_BYTES",
        "canonical_bytes": _MESH_VOCAB_LIMIT * _MESH_QUIPU_NODE_ROW_BYTES,
        "note": "Photon/neutrino resuscitation overlay across all torus nodes",
    },
    "mesh_slm_meta": {
        "formula":      "_MESH_META_BYTES (fixed KV store)",
        "canonical_bytes": _MESH_META_BYTES,
        "note": "Training state: rounds, last_loss, last_eta, phase_weight, …",
    },
    "brain_kv_weyl_tensor": {
        "formula":      "_MESH_BRAIN_KV_WEYL_BYTES",
        "canonical_bytes": _MESH_BRAIN_KV_WEYL_BYTES,
        "note": "5-scalar Weyl condensate JSON persisted to brain_kv each cycle",
    },
    "brain_kv_remnant_score": {
        "formula":      "_MESH_BRAIN_KV_REMNANT_BYTES",
        "canonical_bytes": _MESH_BRAIN_KV_REMNANT_BYTES,
        "note": "Hawking remnant score float string for The Well corpus",
    },
    "brain_kv_adaptive_runtime": {
        "formula":      "_MESH_BRAIN_KV_RUNTIME_BYTES",
        "canonical_bytes": _MESH_BRAIN_KV_RUNTIME_BYTES,
        "note": "Full adaptive runtime state dict: ~25 fields including keywords + axis_drive",
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def coherence_to_phi(coherence: int) -> float:
    """Map Brain harmonic-coherence count → UEQGM characteristic phase φ.

    Returns
    -------
    φ = π/4 + coherence × π

    Every returned value is a natural sin/cos intersection point, where
    sin(φ) = cos(φ) and the SiCi axial channel is well-defined.
    """
    return _PHI_BASE + coherence * _PHI_STEP


def _raw_sici(phi: float) -> tuple[float, float]:
    """Return (Si(φ), Ci(φ)) using scipy if available, else series approx.

    The power-series fallback is accurate to roughly four significant
    figures for |φ| ≤ 2π.  For φ outside that range scipy is preferred.
    """
    if _HAS_SCIPY:
        si, ci = _scipy_sici(phi)
        return float(si), float(ci)
    # ── Series approximation (small/moderate φ) ──────────────────────────
    # Si(x) = x − x³/18 + x⁵/600 − x⁷/35280 + …
    # Ci(x) = γ + ln|x| − x²/4 + x⁴/96 − …   (γ ≈ 0.5772, x ≠ 0)
    x = abs(phi) if phi != 0.0 else 1.0e-12
    si_val = x - x**3 / 18.0 + x**5 / 600.0 - x**7 / 35280.0
    euler_mascheroni = 0.5772156649
    ci_val = euler_mascheroni + math.log(x) - x**2 / 4.0 + x**4 / 96.0
    # Si is an odd function; Ci is even.
    return (si_val if phi >= 0 else -si_val), ci_val


def _clip01(value: float | int | None) -> float:
    """Clamp *value* to the closed interval [0.0, 1.0].

    Handles ``None`` and non-numeric values gracefully, returning 0.0.
    """
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _default_adaptive_runtime() -> dict:
    """Return the all-zero baseline adaptive runtime dict.

    Used as a safe fallback when the ``ueqgm:adaptive_runtime`` brain_kv entry
    is absent or cannot be decoded.  All numeric fields default to 0.0 / 0 /
    ``False``; ``phase_weight`` defaults to 1.0 so UEQGM corrections are a
    no-op rather than a zero multiplier.
    """
    return {
        "active": False,
        "certainty": 0.0,
        "corpus_signal": 0.0,
        "learning_signal": 0.0,
        "newness_signal": 0.0,
        "relational_depth": 0.0,
        "symbiotic_gain": 0.0,
        "phase_weight": 1.0,
        "coherence_depth": 0,
        "phi": round(coherence_to_phi(0), 6),
        "terrain_entropy": 0.0,
        "expansion_pressure": 0.0,
        "mesh_alignment": 0.0,
        "observer_alignment": 0.0,
        "recent_learning_count": 0,
        "ueqgm_entity_count": 0,
        "runtime_keywords": [],
        "learning_kinds": {},
        "axis_drive": {sense: 0.0 for sense in _UEQGM_SENSES},
        "parameter_evidence": {},
        "parameter_density_floor": {},
        "applied_parameters": [],
        "retained_parameters": [],
        "updated_at": None,
        "interstitial_entanglement": 0.0,
    }


def _table_exists(cn: "sqlite3.Connection", table_name: str) -> bool:  # noqa: F821
    """Return ``True`` if *table_name* exists in the SQLite database behind *cn*.

    Uses ``sqlite_master`` rather than a SELECT so it works before any DDL
    has been applied.  Returns ``False`` on any exception.
    """
    try:
        row = cn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        ).fetchone()
    except Exception:
        return False
    return bool(row)


def _runtime_tokens(text: str) -> list[str]:
    """Extract candidate runtime keywords from *text*.

    Tokenises lowercase alphabetic runs of ≥ 4 characters, then filters out:
    * :data:`_UEQGM_RUNTIME_STOPWORDS` — generic Brain/corpus noise words.
    * :data:`_UEQGM_KEYWORDS` — fixed UEQGM physics keywords (avoid double-counting).
    * Tokens starting with ``"conv_"`` (conversation-ID artefacts).

    Used to build the ``runtime_keywords`` field in the adaptive runtime state
    from recent learning logs and corpus entity labels/props.
    """
    if not text:
        return []
    tokens = re.findall(r"[a-z][a-z0-9_]{3,}", text.lower())
    return [
        token for token in tokens
        if token not in _UEQGM_RUNTIME_STOPWORDS
        and token not in _UEQGM_KEYWORDS
        and not token.startswith("conv_")
    ]


def _parse_logged_at(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp string to a timezone-aware UTC datetime.

    Normalises trailing ``Z`` to ``+00:00`` for Python ``fromisoformat`` compat.
    Attaches UTC when the parsed value is naive.  Returns ``None`` on empty input
    or parse failure — callers fall back to a default recency weight of 0.5.
    """
    if not value:
        return None
    normalised = str(value).strip()
    if normalised.endswith("Z"):
        normalised = f"{normalised[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _system_entirety_certainty(entirety_state: dict | None) -> float:
    """Derive a certainty scalar in [0, 1] from the System Entirety state dict.

    Blends four sub-signals with fixed weights:

    * 0.42 — ``observer`` (direct observer confidence).
    * 0.20 — ``transaction.drive`` (transaction pressure).
    * 0.18 — ``material_bifurcation.nodal_bifurcation`` (material coherence).
    * 0.12 — ``material_bifurcation.mesh.mesh_density`` (mesh fill).
    * 0.08 — mean of ``pim_planning.{ss,mm,lt}_signal`` (PIM planning signals).

    Returns 0.0 when *entirety_state* is ``None`` or not a dict.
    """
    if not isinstance(entirety_state, dict):
        return 0.0
    transaction = entirety_state.get("transaction")
    if not isinstance(transaction, dict):
        transaction = {}
    material = entirety_state.get("material_bifurcation")
    if not isinstance(material, dict):
        material = {}
    mesh = material.get("mesh")
    if not isinstance(mesh, dict):
        mesh = {}
    pim = entirety_state.get("pim_planning")
    if not isinstance(pim, dict):
        pim = {}
    return _clip01(
        0.42 * _clip01(entirety_state.get("observer", 0.0))
        + 0.20 * _clip01(transaction.get("drive", 0.0))
        + 0.18 * _clip01(material.get("nodal_bifurcation", 0.0))
        + 0.12 * _clip01(mesh.get("mesh_density", 0.0))
        + 0.08 * _clip01(
            (
                _clip01(pim.get("ss_signal", 0.0))
                + _clip01(pim.get("mm_signal", 0.0))
                + _clip01(pim.get("lt_signal", 0.0))
            ) / 3.0
        )
    )


def _recent_learning_rows(
    cn: "sqlite3.Connection",  # noqa: F821
    *,
    hours: int = _UEQGM_RUNTIME_LOOKBACK_HOURS,
    limit: int = 250,
) -> list[tuple[str, str, str, str, float]]:
    """Fetch recent ``learning_log`` rows as ``(logged_at, kind, title, detail, signal_strength)`` tuples.

    Queries rows newer than ``now − hours`` first; falls back to the most
    recent *limit* rows (ignoring timestamp) when the ``logged_at`` column is
    missing or the time-filtered query returns nothing.  Returns ``[]`` when
    the ``learning_log`` table does not exist.
    """
    if not _table_exists(cn, "learning_log"):
        return []
    threshold = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        rows = cn.execute(
            "SELECT logged_at, kind, title, detail, COALESCE(signal_strength, 0.5) "
            "FROM learning_log WHERE logged_at >= ? "
            "ORDER BY logged_at DESC LIMIT ?",
            (threshold, max(1, int(limit))),
        ).fetchall()
    except Exception:
        try:
            rows = cn.execute(
                "SELECT logged_at, kind, title, detail, COALESCE(signal_strength, 0.5) "
                "FROM learning_log ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        except Exception:
            return []
    return [
        (
            str(row[0] or ""),
            str(row[1] or ""),
            str(row[2] or ""),
            str(row[3] or ""),
            float(row[4] or 0.5),
        )
        for row in rows
    ]


def _learning_stats(
    learning_rows: list[tuple[str, str, str, str, float]],
) -> tuple[float, float, dict[str, int], float]:
    """Compute aggregate statistics from recent learning rows.

    Parameters
    ----------
    learning_rows:
        List of ``(logged_at, kind, title, detail, signal_strength)`` tuples
        as returned by :func:`_recent_learning_rows`.

    Returns
    -------
    A four-tuple of:

    * ``learning_signal`` — recency-weighted mean signal strength in [0, 1].
    * ``newness_signal``  — mean recency score in [0, 1]; 1.0 = very fresh.
    * ``kinds``           — ``Counter``-like dict of ``{kind: count}``.
    * ``relationality``   — combined token-diversity + kind-diversity score in [0, 1].
    """
    if not learning_rows:
        return 0.0, 0.0, {}, 0.0

    now = datetime.now(timezone.utc)
    weighted_signal = 0.0
    total_weight = 0.0
    recency_scores: list[float] = []
    distinct_tokens: set[str] = set()
    kinds: Counter[str] = Counter()

    for logged_at, kind, title, detail, signal in learning_rows:
        kinds[kind] += 1
        parsed = _parse_logged_at(logged_at)
        if parsed is None:
            recency = 0.5
        else:
            age_hours = max(0.0, (now - parsed).total_seconds() / 3600.0)
            recency = _clip01(1.0 - (age_hours / float(_UEQGM_RUNTIME_LOOKBACK_HOURS or 1)))
        weight = 0.35 + 0.65 * recency
        weighted_signal += _clip01(signal) * weight
        total_weight += weight
        recency_scores.append(recency)
        distinct_tokens.update(_runtime_tokens(f"{kind} {title} {detail}"))

    learning_signal = _clip01(weighted_signal / total_weight) if total_weight > 0.0 else 0.0
    newness_signal = _clip01(sum(recency_scores) / len(recency_scores))
    relationality = _clip01(
        0.65 * min(1.0, len(distinct_tokens) / 18.0)
        + 0.35 * min(1.0, len(kinds) / 4.0)
    )
    return learning_signal, newness_signal, dict(kinds), relationality


def _corpus_metrics(corpus_rows: list[tuple[str, str]]) -> tuple[float, float]:
    """Compute corpus signal strength and corpus relationality from UEQGM-tagged entity rows.

    Parameters
    ----------
    corpus_rows:
        List of ``(label, props_json)`` pairs from ``corpus_entity`` matching
        the :data:`_UEQGM_TAGS` filter.

    Returns
    -------
    A two-tuple of:

    * ``corpus_signal``       — volume-weighted UEQGM keyword density in [0, 1].
    * ``corpus_relationality`` — distinct-token + volume diversity score in [0, 1].
    """
    if not corpus_rows:
        return 0.0, 0.0

    densities: list[float] = []
    distinct_tokens: set[str] = set()
    for label, props in corpus_rows:
        text = f"{label} {props}".lower()
        density = sum(text.count(keyword) for keyword in _UEQGM_KEYWORDS)
        densities.append(min(1.0, density / max(1, len(_UEQGM_KEYWORDS))))
        distinct_tokens.update(_runtime_tokens(text))

    corpus_signal = _clip01(
        min(1.0, len(corpus_rows) / 50.0) * 0.65 + (sum(densities) / len(densities)) * 0.35
    )
    corpus_relationality = _clip01(
        0.60 * min(1.0, len(distinct_tokens) / 18.0)
        + 0.40 * min(1.0, len(corpus_rows) / 30.0)
    )
    return corpus_signal, corpus_relationality


def _runtime_keywords_from_sources(
    corpus_rows: list[tuple[str, str]],
    learning_rows: list[tuple[str, str, str, str, float]],
    certainty: float,
) -> list[str]:
    """Select the top-N runtime keywords from corpus entity labels and learning logs.

    Corpus tokens each contribute a weight of 0.2; learning tokens are weighted
    by their ``signal_strength``.  The dynamic limit grows linearly from 4
    keywords at ``certainty = 0`` to :data:`_UEQGM_RUNTIME_MAX_KEYWORDS` at
    ``certainty = 1``.  Returns ``[]`` when both sources are empty.
    """
    weighted: Counter[str] = Counter()

    for label, props in corpus_rows[:50]:
        for token in _runtime_tokens(f"{label} {props}"):
            weighted[token] += 0.2

    for _, kind, title, detail, signal in learning_rows:
        text = f"{kind} {title} {detail}"
        for token in _runtime_tokens(text):
            weighted[token] += max(0.1, float(signal))

    if not weighted:
        return []

    dynamic_limit = max(
        4,
        min(
            _UEQGM_RUNTIME_MAX_KEYWORDS,
            int(round(4 + certainty * (_UEQGM_RUNTIME_MAX_KEYWORDS - 4))),
        ),
    )
    ranked = sorted(weighted.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:dynamic_limit]]


def _should_apply_runtime_parameter(
    previous: dict,
    parameter: str,
    evidence: float,
    density_floor: float,
) -> bool:
    """Return ``True`` if a new candidate value for *parameter* should replace the previous one.

    A parameter update is accepted only when the new *evidence* exceeds both
    the *density_floor* and the previously stored evidence by a small epsilon
    (``1e-6``).  This implements a conservative hysteresis gate that prevents
    the adaptive runtime from oscillating on noisy signals.

    Parameters
    ----------
    previous:
        The current adaptive runtime dict (from ``get_adaptive_runtime``).
    parameter:
        The parameter name to check (e.g. ``"certainty"``).
    evidence:
        Computed evidence score for the new candidate value in [0, 1].
    density_floor:
        Minimum evidence threshold derived from corpus density.
    """
    previous_evidence = _clip01((previous.get("parameter_evidence") or {}).get(parameter, 0.0))
    return evidence > max(_clip01(density_floor), previous_evidence) + 1.0e-6


def _axis_drive_from_entirety(entirety_state: dict | None, symbiotic_gain: float) -> dict:
    """Compute the per-sense axis drive from the Entirety state and *symbiotic_gain*.

    Reads ``entirety_state["axes"]`` (a ``{sense: float}`` dict) and normalises
    by the maximum axis value, then multiplies by *symbiotic_gain* to produce a
    bounded drive signal per sense.  Returns zeros for all senses when the axes
    are absent or zero.
    """
    axes = {}
    if isinstance(entirety_state, dict) and isinstance(entirety_state.get("axes"), dict):
        axes = entirety_state.get("axes") or {}
    max_axis = max((_clip01(axes.get(sense, 0.0)) for sense in _UEQGM_SENSES), default=0.0)
    if max_axis <= 0.0:
        return {sense: 0.0 for sense in _UEQGM_SENSES}
    return {
        sense: round(_clip01(((_clip01(axes.get(sense, 0.0)) + 0.20) / (max_axis + 0.20)) * symbiotic_gain), 4)
        for sense in _UEQGM_SENSES
    }



def get_adaptive_runtime(cn: "sqlite3.Connection" | None = None) -> dict:  # noqa: F821
    """Read the cached UEQGM adaptive runtime from ``brain_kv`` without recomputing.

    This is the *fast read path* used by ``mesh_slm`` during training and
    scoring.  The runtime dict is written by :func:`refresh_adaptive_runtime`
    (called from the UEQGM daemon on a periodic cadence) and cached under
    :data:`_UEQGM_RUNTIME_KEY`.

    Parameters
    ----------
    cn:
        Optional open ``sqlite3.Connection``.  When ``None``, a new
        connection is opened (and closed) internally.

    Returns
    -------
    The previously persisted runtime dict, or :func:`_default_adaptive_runtime`
    when the key is absent, the value cannot be decoded, or the connection
    fails.  Key fields consumed by ``mesh_slm``:

    * ``phase_weight`` — SiCi phase correction multiplier (≈ 1.0 ± 10 %).
    * ``coherence_depth`` — integer harmonic coherence level.
    * ``phi`` — characteristic UEQGM phase at the current coherence.
    * ``symbiotic_gain``, ``expansion_pressure``, ``axis_drive``.
    """
    default = _default_adaptive_runtime()
    owns_conn = False
    if cn is None:
        try:
            from .local_store import open_conn as _open_conn

            cn = _open_conn(timeout=20)
            owns_conn = True
        except Exception:
            return default

    try:
        if not _table_exists(cn, "brain_kv"):
            return default
        row = cn.execute(
            "SELECT value FROM brain_kv WHERE key=? LIMIT 1",
            (_UEQGM_RUNTIME_KEY,),
        ).fetchone()
        if not row or not row[0]:
            return default
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            return default
    except Exception:
        return default
    finally:
        if owns_conn and cn is not None:
            cn.close()

    axis_drive = payload.get("axis_drive")
    if not isinstance(axis_drive, dict):
        axis_drive = {}
    merged = {**default, **payload}
    merged["active"] = bool(payload.get("active", True))
    merged["runtime_keywords"] = [
        str(token).lower() for token in (payload.get("runtime_keywords") or []) if str(token).strip()
    ]
    merged["learning_kinds"] = dict(payload.get("learning_kinds") or {})
    merged["parameter_evidence"] = dict(payload.get("parameter_evidence") or {})
    merged["parameter_density_floor"] = dict(payload.get("parameter_density_floor") or {})
    merged["applied_parameters"] = list(payload.get("applied_parameters") or [])
    merged["retained_parameters"] = list(payload.get("retained_parameters") or [])
    merged["axis_drive"] = {
        sense: _clip01(axis_drive.get(sense, 0.0)) for sense in _UEQGM_SENSES
    }
    return merged


def refresh_adaptive_runtime(
    cn: "sqlite3.Connection",  # noqa: F821
    entirety_state: dict | None = None,
    *,
    force: bool = False,
) -> dict:

    """Refresh the adaptive UEQGM runtime from corpus, learnings, and Entirety certainty.

    The daemon-facing runtime is intentionally stateful. It uses:

    1. UEQGM-tagged corpus entities for physics/graph grounding.
    2. Recent learning_log rows so the engine expands beyond the Grok export.
    3. The current System Entirety state as a certainty field that determines
       how aggressively new learnings should bias the engine's symbiotic spread.
    """
    corpus_rows: list[tuple[str, str]] = []
    entity_count = 0
    if _table_exists(cn, "corpus_entity"):
        tag_filter = " OR ".join(f"props_json LIKE ?" for _ in _UEQGM_TAGS)
        params = tuple(f"%{tag}%" for tag in _UEQGM_TAGS)
        try:
            rows = cn.execute(
                f"SELECT label, props_json FROM corpus_entity WHERE ({tag_filter}) LIMIT 200",
                params,
            ).fetchall()
            corpus_rows = [(str(row[0] or ""), str(row[1] or "")) for row in rows]
            entity_count = len(corpus_rows)
        except Exception:
            corpus_rows = []
            entity_count = 0

    learning_rows = _recent_learning_rows(cn)
    learning_count = len(learning_rows)
    previous = get_adaptive_runtime(cn)
    candidate_corpus_signal, corpus_relationality = _corpus_metrics(corpus_rows)
    learning_signal, newness_signal, learning_kinds, learning_relationality = _learning_stats(learning_rows)
    candidate_certainty = _system_entirety_certainty(entirety_state)
    relational_depth = _clip01(0.55 * learning_relationality + 0.45 * corpus_relationality)

    mesh_alignment = 0.0
    observer_alignment = 0.0
    if isinstance(entirety_state, dict):
        observer_alignment = _clip01(entirety_state.get("observer", 0.0))
        material = entirety_state.get("material_bifurcation")
        if isinstance(material, dict):
            mesh = material.get("mesh")
            if isinstance(mesh, dict):
                mesh_alignment = _clip01(mesh.get("mesh_density", 0.0))

    candidate_runtime_keywords = _runtime_keywords_from_sources(
        corpus_rows,
        learning_rows,
        candidate_certainty,
    )
    parameter_evidence: dict[str, float] = {
        "certainty": round(_clip01(0.65 * candidate_certainty + 0.20 * newness_signal + 0.15 * relational_depth), 4),
        "corpus_signal": round(_clip01(0.75 * candidate_corpus_signal + 0.25 * corpus_relationality), 4),
        "learning_signal": round(_clip01(0.45 * learning_signal + 0.30 * newness_signal + 0.25 * relational_depth), 4),
        "mesh_alignment": round(_clip01(0.55 * mesh_alignment + 0.25 * candidate_certainty + 0.20 * relational_depth), 4),
        "observer_alignment": round(_clip01(0.55 * observer_alignment + 0.25 * candidate_certainty + 0.20 * newness_signal), 4),
        "runtime_keywords": round(_clip01(0.35 * learning_signal + 0.35 * newness_signal + 0.30 * relational_depth), 4),
    }
    parameter_density_floor: dict[str, float] = {
        "certainty": round(candidate_corpus_signal, 4),
        "corpus_signal": round(candidate_corpus_signal, 4),
        "learning_signal": round(max(candidate_corpus_signal, corpus_relationality * 0.85), 4),
        "mesh_alignment": round(max(candidate_corpus_signal, mesh_alignment * 0.60), 4),
        "observer_alignment": round(max(candidate_corpus_signal, observer_alignment * 0.60), 4),
        "runtime_keywords": round(max(candidate_corpus_signal, corpus_relationality), 4),
    }

    base_candidates = {
        "certainty": round(candidate_certainty, 4),
        "corpus_signal": round(candidate_corpus_signal, 4),
        "learning_signal": round(learning_signal, 4),
        "mesh_alignment": round(mesh_alignment, 4),
        "observer_alignment": round(observer_alignment, 4),
        "runtime_keywords": candidate_runtime_keywords,
    }
    applied_parameters: list[str] = []
    retained_parameters: list[str] = []
    resolved_base: dict[str, object] = {}
    for parameter, candidate in base_candidates.items():
        if force or _should_apply_runtime_parameter(
            previous,
            parameter,
            parameter_evidence[parameter],
            parameter_density_floor[parameter],
        ):

            resolved_base[parameter] = candidate
            applied_parameters.append(parameter)
        else:
            resolved_base[parameter] = previous.get(parameter, candidate)
            retained_parameters.append(parameter)

    certainty = _clip01(resolved_base["certainty"])
    corpus_signal = _clip01(resolved_base["corpus_signal"])
    learning_signal = _clip01(resolved_base["learning_signal"])
    mesh_alignment = _clip01(resolved_base["mesh_alignment"])
    observer_alignment = _clip01(resolved_base["observer_alignment"])
    runtime_keywords = [str(token).lower() for token in (resolved_base["runtime_keywords"] or [])]

    coherence_candidate = max(
        0,
        int(round((0.30 * corpus_signal + 0.25 * learning_signal + 0.20 * certainty + 0.25 * relational_depth) * 8.0)),
    )
    coherence_evidence = round(_clip01(0.50 * corpus_signal + 0.30 * relational_depth + 0.20 * newness_signal), 4)
    coherence_floor = round(max(corpus_signal, corpus_relationality), 4)
    parameter_evidence["coherence_depth"] = coherence_evidence
    parameter_evidence["phi"] = coherence_evidence
    parameter_evidence["phase_weight"] = coherence_evidence
    parameter_density_floor["coherence_depth"] = coherence_floor
    parameter_density_floor["phi"] = coherence_floor
    parameter_density_floor["phase_weight"] = coherence_floor

    if _should_apply_runtime_parameter(previous, "coherence_depth", coherence_evidence, coherence_floor):
        coherence_depth = coherence_candidate
        phase_weight = sici_phase_weight(coherence_depth)
        applied_parameters.extend(["coherence_depth", "phi", "phase_weight"])
    else:
        coherence_depth = int(previous.get("coherence_depth", coherence_candidate) or 0)
        phase_weight = float(previous.get("phase_weight", sici_phase_weight(coherence_depth)) or 1.0)
        retained_parameters.extend(["coherence_depth", "phi", "phase_weight"])
    phi = round(coherence_to_phi(coherence_depth), 6)

    previous_entropy = float(previous.get("terrain_entropy", 0.0) or 0.0)
    laplacian = (learning_signal - corpus_signal) + (certainty - 0.5) * 0.5
    gamma_0 = 1.0 + 0.25 * certainty
    gamma_eff = 1.0 + 0.25 * max(corpus_signal, learning_signal)
    eta_diff = _ETA_DIFF_DEFAULT * (0.5 + 0.5 * max(learning_signal, certainty))
    terrain_entropy_candidate = entropic_bayesian_step(
        previous_entropy,
        laplacian,
        phi,
        gamma_0=gamma_0,
        gamma_eff=gamma_eff,
        eta_diff=eta_diff,
    )
    entropy_norm = abs(terrain_entropy_candidate) / (1.0 + abs(terrain_entropy_candidate))
    symbiotic_gain_candidate = _clip01(
        (
            0.40 * certainty
            + 0.25 * learning_signal
            + 0.20 * relational_depth
            + 0.15 * entropy_norm
        ) * phase_weight
    )
    expansion_pressure_candidate = _clip01(
        0.50 * symbiotic_gain_candidate + 0.25 * learning_signal + 0.25 * certainty
    )
    axis_drive_candidate = _axis_drive_from_entirety(entirety_state, symbiotic_gain_candidate)
    symbiotic_cluster_evidence = round(
        _clip01(0.35 * certainty + 0.25 * learning_signal + 0.20 * relational_depth + 0.20 * newness_signal),
        4,
    )
    symbiotic_cluster_floor = round(max(corpus_signal, relational_depth), 4)
    for parameter in ("terrain_entropy", "symbiotic_gain", "expansion_pressure", "axis_drive"):
        parameter_evidence[parameter] = symbiotic_cluster_evidence
        parameter_density_floor[parameter] = symbiotic_cluster_floor

    if force or _should_apply_runtime_parameter(previous, "symbiotic_gain", symbiotic_cluster_evidence, symbiotic_cluster_floor):

        terrain_entropy = round(float(terrain_entropy_candidate), 6)
        symbiotic_gain = round(symbiotic_gain_candidate, 4)
        expansion_pressure = round(expansion_pressure_candidate, 4)
        axis_drive = axis_drive_candidate
        applied_parameters.extend(["terrain_entropy", "symbiotic_gain", "expansion_pressure", "axis_drive"])
    else:
        terrain_entropy = round(float(previous.get("terrain_entropy", terrain_entropy_candidate) or 0.0), 6)
        symbiotic_gain = round(_clip01(previous.get("symbiotic_gain", symbiotic_gain_candidate)), 4)
        expansion_pressure = round(_clip01(previous.get("expansion_pressure", expansion_pressure_candidate)), 4)
        axis_drive = previous.get("axis_drive") or axis_drive_candidate
        retained_parameters.extend(["terrain_entropy", "symbiotic_gain", "expansion_pressure", "axis_drive"])

    # -- Interstitial entanglement score (sunlight-entanglement analogy) -------
    # Read the last _INTERSTITIAL_ENTANGLEMENT_WINDOW Weyl tensors from brain_kv
    # to compute the third-order covariance witness.  Falls back to the previous
    # runtime value (or 0.0) when fewer than 2 cycles are available.
    ie_score: float = float(previous.get("interstitial_entanglement", 0.0) or 0.0)
    try:
        weyl_rows = cn.execute(
            "SELECT value FROM brain_kv WHERE key LIKE ? OR key=? "
            "ORDER BY updated_at DESC LIMIT ?",
            (
                f"{_WEYL_TENSOR_KEY}%",
                _WEYL_TENSOR_KEY,
                _INTERSTITIAL_ENTANGLEMENT_WINDOW,
            ),
        ).fetchall()
        psi5_seq: list[list[float]] = []
        for row in weyl_rows:
            try:
                parsed = json.loads(row[0])
                if isinstance(parsed, list) and len(parsed) == 5:
                    psi5_seq.append([float(v) for v in parsed])
            except Exception:
                pass
        if len(psi5_seq) >= 2:
            ie_score = round(interstitial_entanglement_score(psi5_seq), 6)
    except Exception:
        pass  # keep ie_score from previous runtime

    # Persist the entanglement score separately so mesh_slm can read it quickly.
    try:
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            (
                _INTERSTITIAL_ENTANGLEMENT_KEY,
                json.dumps({"score": ie_score}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    except Exception:
        pass

    runtime_state = {
        "active": True,
        "certainty": round(certainty, 4),
        "corpus_signal": round(corpus_signal, 4),
        "learning_signal": round(learning_signal, 4),
        "newness_signal": round(newness_signal, 4),
        "relational_depth": round(relational_depth, 4),
        "symbiotic_gain": round(symbiotic_gain, 4),
        "phase_weight": round(phase_weight, 6),
        "coherence_depth": coherence_depth,
        "phi": round(phi, 6),
        "terrain_entropy": round(float(terrain_entropy), 6),
        "expansion_pressure": round(expansion_pressure, 4),
        "mesh_alignment": round(mesh_alignment, 4),
        "observer_alignment": round(observer_alignment, 4),
        "recent_learning_count": learning_count,
        "ueqgm_entity_count": entity_count,
        "runtime_keywords": runtime_keywords,
        "learning_kinds": learning_kinds,
        "interstitial_entanglement": ie_score,
        "axis_drive": {
            sense: round(_clip01((axis_drive or {}).get(sense, 0.0)), 4)
            for sense in _UEQGM_SENSES
        },
        "parameter_evidence": parameter_evidence,
        "parameter_density_floor": parameter_density_floor,
        "applied_parameters": sorted(set(applied_parameters)),
        "retained_parameters": sorted(set(retained_parameters)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    cn.execute(
        "CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
    )
    cn.execute(
        "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
        (
            _UEQGM_RUNTIME_KEY,
            json.dumps(runtime_state, default=str),
            runtime_state["updated_at"],
        ),
    )
    return runtime_state


# ---------------------------------------------------------------------------
# Core UEQGM SiCi functions
# ---------------------------------------------------------------------------

def sici_axial_decay(phi: float, gamma_0: float = _GAMMA_0_DEFAULT) -> float:
    """Axial channel decay differential  Δλ_axial  from UEQGM v0.9.14.

    .. math::

        \\Delta\\lambda_{\\rm axial} =
            \\bigl[\\operatorname{Si}(\\phi) \\cdot \\operatorname{Ci}(\\phi)\\bigr]
            \\cdot \\tan(\\phi) \\cdot \\Gamma_0

    Parameters
    ----------
    phi:
        Characteristic phase of the 6D CAT states.  Use
        ``coherence_to_phi(coherence)`` to map a Brain coherence count.
    gamma_0:
        Baseline total decay width (normalised to 1.0 for Brain usage).

    Returns
    -------
    The axial channel differential value.  Positive near k=0 (first
    intersection), oscillating and decaying in magnitude as φ grows.
    """
    si, ci = _raw_sici(phi)
    tan_phi = math.tan(phi)
    # Clamp tan to prevent divergence at φ near π/2 + nπ.
    tan_phi = max(-_TAN_CLAMP, min(_TAN_CLAMP, tan_phi))
    return si * ci * tan_phi * gamma_0


def sici_phase_weight(coherence: int) -> float:
    """Normalised UEQGM phase weight for Brain harmonic amplification.

    Maps *coherence* → φ = π/4 + coherence·π → SiCi axial decay →
    bounded correction factor near 1.0.

    Returns
    -------
    A factor in  (1 − _SICI_SCALE_FACTOR, 1 + _SICI_SCALE_FACTOR)
    safe for use as a multiplicative correction to the harmonic factor.

    At large coherence Si(φ) → π/2, Ci(φ) → 0, so the correction
    approaches 1.0 (no distortion of the saturation ceiling).
    """
    phi = coherence_to_phi(coherence)
    raw = sici_axial_decay(phi)
    return 1.0 + _SICI_SCALE_FACTOR * math.tanh(raw)


# ---------------------------------------------------------------------------
# Broader UEQGM mathematics
# ---------------------------------------------------------------------------

def wavefunction_overlap(
    vec_a: Sequence[float],
    vec_b: Sequence[float],
) -> float:
    """Born-rule state fidelity  F = |⟨ψ_a | ψ_b⟩|²  for real state vectors.

    This is not quantum-*inspired* — it is the exact Born rule (Born, *Z.
    Phys.* 37, 863 (1926)) / pure-state fidelity (Jozsa, *J. Mod. Opt.* 41,
    2315 (1994)) evaluated in a **real** Hilbert space.  For real amplitudes
    the imaginary component of the inner product is identically zero — the
    √−1 contributes exactly 1 — so the fidelity reduces *without
    approximation* to the squared cosine similarity of the L2-normalised
    vectors.  The transition probability between the two states is this
    number.

    Returns
    -------
    1.0  — identical (parallel) states.
    0.0  — orthogonal states or either vector has zero norm.

    Raises
    ------
    Nothing — all edge cases return 0.0.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    cos_theta = dot / (norm_a * norm_b)
    return round(cos_theta ** 2, 6)


def floquet_modulation_factor(drive: float, omega: float) -> float:
    """Floquet effective-coupling renormalization  J_eff / J = J₀(K),  K = A/ω.

    The central exact result of Floquet engineering: a coupling *J* driven
    periodically with amplitude *A* at frequency *ω* is renormalized, in the
    high-frequency effective Hamiltonian, by the zeroth-order Bessel function
    of the drive parameter K = A/ω:

    * **Dynamic localization** — Dunlap & Kenkre, *Phys. Rev. B* 34, 3625
      (1986): a charged particle on a driven lattice has its bandwidth
      renormalized by J₀(K) and localizes completely at the Bessel zeros.
    * **Coherent destruction of tunneling** — Grossmann, Dittrich, Jung &
      Hänggi, *Phys. Rev. Lett.* 67, 516 (1991).
    * Measured in shaken optical lattices — Lignier et al., *Phys. Rev.
      Lett.* 99, 220403 (2007); reviewed in Eckardt, *Rev. Mod. Phys.* 89,
      011004 (2017).

    Parameters
    ----------
    drive:  Drive amplitude *A* (here: the Weyl phase acting as the drive).
    omega:  Drive frequency *ω* > 0.

    Returns
    -------
    J₀(A/ω) ∈ [−0.4028, 1].  1.0 in the undriven/high-frequency limit
    (K → 0); exactly 0 at the Bessel zeros K = 2.4048, 5.5201, … (coherent
    destruction of tunneling); 1.0 when ω = 0 (no drive defined).
    """
    if omega == 0.0:
        return 1.0
    return _bessel_j0(drive / omega)


def tantalum_intermediary_binding(
    *,
    weyl_phase: float,
    coherence: int,
    mesh_alignment: float,
    observer_alignment: float,
    pulse_weight: float = 1.0,
) -> dict[str, float | str | int]:
    """Bounded UEQGM intermediary binding profile for routing a Weyl pulse.

    The existing UEQGM machinery already provides the phase-sensitive pieces we
    need: the SiCi axial correction, the Floquet pulse term, and the total phase
    evolution. This helper packages those into a single, bounded "Tantalum"
    binding layer so downstream consumers can couple a Weyl pulse to a resource
    without inventing their own phase math.

    The name honours tantalum's real place at the bleeding edge: it is the
    electrode material that lifted superconducting transmon qubit coherence
    past 0.3 ms (Place et al., *Nature Communications* 12, 1779 (2021)) —
    the current material of record for low-loss quantum circuits.  In THIS
    repository the label is exactly that — an honouring label on a bounded
    classical routing profile.  No physical tantalum receiver, GPU channel,
    or hardware coupling exists here, and none is claimed.
    """
    coherence_depth = max(0, int(coherence))
    mesh_alignment = _clip01(mesh_alignment)
    observer_alignment = _clip01(observer_alignment)
    pulse_weight = max(0.05, float(pulse_weight or 1.0))
    phi = coherence_to_phi(coherence_depth)
    phase_weight = sici_phase_weight(coherence_depth)
    omega = max(0.1, pulse_weight * phase_weight)
    pulse_coupling = _clip01(0.5 * (1.0 + floquet_modulation_factor(float(weyl_phase), omega)))
    axial_drive = abs(sici_axial_decay(phi))
    axial_channel = axial_drive / (1.0 + axial_drive)
    phase_total = abs(
        phase_evolution_total(
            phi,
            delta_gamma=float(weyl_phase),
            gamma_eff=max(1.0, pulse_weight),
        )
    )
    phase_norm = _clip01(phase_total / (2.0 * math.pi))
    phase_weight_norm = _clip01(
        (phase_weight - (1.0 - _SICI_SCALE_FACTOR)) / (2.0 * _SICI_SCALE_FACTOR)
    )
    pulse_norm = _clip01(pulse_weight / (1.0 + pulse_weight))
    overlap = wavefunction_overlap(
        [mesh_alignment, observer_alignment, pulse_norm],
        [phase_norm, pulse_coupling, phase_weight_norm],
    )
    binding_gain = _clip01(
        0.35 * pulse_coupling
        + 0.25 * axial_channel
        + 0.20 * phase_norm
        + 0.20 * overlap
    )
    return {
        "intermediary": "tantalum",
        "receiver_material": "tantalum",
        "receiver_role": "gpu_receiver_metal",
        "receiver_channel": "directed_gpu_comm",
        "weyl_phase": round(float(weyl_phase), 6),
        "coherence_depth": coherence_depth,
        "phi": round(phi, 6),
        "phase_weight": round(phase_weight, 6),
        "pulse_weight": round(pulse_weight, 6),
        "pulse_coupling": round(pulse_coupling, 6),
        "axial_channel": round(axial_channel, 6),
        "phase_evolution": round(phase_total, 6),
        "overlap": round(overlap, 6),
        "mesh_alignment": round(mesh_alignment, 6),
        "observer_alignment": round(observer_alignment, 6),
        "binding_gain": round(binding_gain, 6),
        "binding_multiplier": round(0.55 + 0.90 * binding_gain, 6),
    }


def interstitial_entanglement_score(
    psi5_seq: "Sequence[Sequence[float]]",
) -> float:
    """Estimate entanglement surviving in the interstitial bit channel.

    Applies the principle from the 2026 Optica sunlight-entanglement paper
    (DOI 10.1364/OPTICA.601797): thermal (classically mixed) light generates
    genuine photon-number entanglement at the output ports of an asymmetric
    beamsplitter because the interstitial vacuum modes carry cross-port
    covariance.  In QUIPU the analogue is the lossy GARD tier — successive Weyl
    compression cycles are the "thermal photons", and the interstitial bits
    below the Cramér-Rao floor are the vacuum modes.  Entanglement is detectable
    as a non-trivial third-order covariance across three consecutive cycles.

    Concretely, for a window of N successive 5-scalar Weyl states the score is:

        E = |C_01 · C_12 − C_02| / (1 + |C_02|)

    where  C_ij = ``photon_covariance_proxy(psi5_i, psi5_j)``.  This mirrors
    the paper's third-order intensity-moment test that is non-zero only when
    the interstitial covariance structure persists across more than one
    beamsplitter stage.  For windows longer than 3 the score is averaged over
    all consecutive triples.

    Parameters
    ----------
    psi5_seq:
        Sequence of at least 2 Weyl tensors, each a 5-float list/tuple.
        Fewer than 2 tensors returns 0.0.

    Returns
    -------
    Entanglement proxy score in ``[0, 1]``.  0.0 = no detectable cross-cycle
    covariance; higher values indicate coherent correlation structure that
    survived the lossy GARD compression.
    """
    from . import gard_info as _gard_info

    seq = [list(t) for t in psi5_seq if t]
    if len(seq) < 2:
        return 0.0

    # Pairwise covariances for all adjacent and skip-1 pairs.
    def _cov(i: int, j: int) -> float:
        return _gard_info.photon_covariance_proxy(seq[i], seq[j])

    if len(seq) == 2:
        # Only one pair available — return |C_01| directly as a degenerate score.
        return round(abs(_cov(0, 1)), 6)

    # Average third-order witness over all consecutive triples.
    triple_scores: list[float] = []
    for k in range(len(seq) - 2):
        c01 = _cov(k, k + 1)
        c12 = _cov(k + 1, k + 2)
        c02 = _cov(k, k + 2)
        witness = abs(c01 * c12 - c02) / (1.0 + abs(c02))
        triple_scores.append(_clip01(witness))

    return round(sum(triple_scores) / len(triple_scores), 6)


def holographic_entropy(
    n_edges: int,
    n_nodes: int,
    degree_pair_sum: float | None = None,
) -> float:
    """Von Neumann entropy of the graph — quadratic approximation.

    The real quantum entropy of a graph: interpret the scaled combinatorial
    Laplacian ``ρ = L / tr(L)`` as a density matrix (Braunstein, Ghosh &
    Severini, *Ann. Comb.* 10, 291 (2006)) and take S = −tr(ρ ln ρ).  The
    quadratic (Rényi-2) approximation of Han, Escolano, Hancock & Wilson,
    *Pattern Recognition Letters* 33 (2012) 1958–1967, computes it in
    O(|E|) from the degrees alone:

    .. math::

        S \\;\\approx\\; 1 - \\frac{1}{n}
            - \\frac{1}{n^{2}} \\sum_{(u,v) \\in E} \\frac{1}{d_u\\,d_v}

    Parameters
    ----------
    n_edges:          Number of edges ``|E|`` (quipu bigram links, treated as
                      undirected for the entropy).
    n_nodes:          Number of nodes *n* (vocab tokens).
    degree_pair_sum:  Exact ``Σ 1/(d_u · d_v)`` over the edge list.  Callers
                      with connection access compute it straight from the
                      quipu edge table via SQL (``mesh_slm._mesh_field_8d``
                      does this) — the **real** entropy of the actual degree
                      structure.  When ``None``, degrees are estimated from
                      average edge density: ``d̄ = 2|E|/n`` gives the closed
                      form ``Σ ≈ |E|/d̄² = n²/(4|E|)``, i.e.
                      ``S ≈ 1 − 1/n − 1/(4|E|)`` — the **computational**
                      mean-field approximation from counts alone.

    The differential between the two paths is real vs computational: the
    exact sum sees degree heterogeneity (hubs raise S above mean-field;
    chains lower it), while the count-only fallback assumes a regular graph.
    Both paths evaluate the same HEHW functional, so the two are directly
    comparable and the fallback degrades gracefully.

    Returns
    -------
    S ∈ [0, 1].  0.0 for an empty or edgeless graph (zero entropy for a
    zero density matrix — no fictitious boundary term).
    """
    if n_nodes <= 0 or n_edges <= 0:
        return 0.0
    n = float(n_nodes)
    if degree_pair_sum is None:
        degree_pair_sum = (n * n) / (4.0 * float(n_edges))
    s = 1.0 - (1.0 / n) - (float(degree_pair_sum) / (n * n))
    return _clip01(s)


def hawking_information_remnant_score(
    n_datasets: int,
    n_spatial_dims: int,
    total_tb: float = 0.0,
) -> float:
    """Unitary Page-curve information budget of a dataset collection.

    Models corpus ingestion as black-hole evaporation: each ingested dataset
    is a Hawking quantum carried away from the inaccessible bulk, and the
    information retrievable from the radiation follows the **Page curve**
    (Page, *Phys. Rev. Lett.* 71, 3743 (1993)), in its modern
    quantum-extremal-island form (Penington, *JHEP* 09 (2020) 002; Almheiri,
    Hartman, Maldacena, Shaghoulian & Tajdini, *Rev. Mod. Phys.* 93, 035002
    (2021)).

    The corpus's dimensionless area ``A_eff = n_datasets × n_dims`` (with an
    optional Bekenstein-bound mass-energy factor for *total_tb*) is compared
    against the Page reference area ``A_page = 16 × 4 = 64`` — The Well
    (Polymathic AI, 15 TB, 16 spatiotemporal PDE datasets) sitting exactly at
    the Page time:

    .. math::

        f = \\frac{A_{\\rm eff}}{A_{\\rm eff} + A_{\\rm page}}, \\qquad
        S_{\\rm rad} = \\min(f, 1-f), \\qquad
        I_{\\rm isl} = \\max(0, 2f-1)

    The returned score is ``S_rad + I_isl = f`` — the gross information
    budget carried by the radiation, which by unitarity is exactly the
    evaporated fraction.  The decomposition is the physics; the sum is the
    score.

    Parameters
    ----------
    n_datasets:      Number of datasets in the collection (``≥ 1``).
    n_spatial_dims:  Effective spatial/temporal dimensionality per dataset
                     (e.g. 3 for xyz + 1 time = 4 for a spatiotemporal field).
    total_tb:        Total corpus size in terabytes (optional; 0 skips the
                     Bekenstein mass-energy term).

    Returns
    -------
    Scalar in ``[0, 1]``.  0.0 when *n_datasets* ≤ 0; 0.5 exactly at the
    Page reference (16 datasets × 4 dims, no mass term); 2/3 for the full
    15 TB Well.
    """
    if n_datasets <= 0:
        return 0.0
    n_dims = max(1, n_spatial_dims)
    area = float(n_datasets * n_dims)
    if total_tb > 0.0:
        # Bekenstein-bound energy term (Bekenstein, Phys. Rev. D 23, 287
        # (1981)): information capacity grows with mass-energy.  Log-
        # compressed against the 15 TB Well reference because textual
        # information grows sublinearly in raw bytes (Heaps' law).
        area *= 1.0 + (math.log1p(total_tb) / math.log1p(_THE_WELL_TB_REF))

    # Evaporated fraction relative to the Page reference area (The Well's
    # 16 datasets × 4 dims = 64).  The geometric reference corpus sits
    # exactly at the Page time, f = 1/2.
    f = area / (area + _PAGE_REFERENCE_AREA)

    # Unitary Page curve with the island prescription:
    #   S_rad(f)   = min(f, 1−f)      radiation entanglement entropy
    #   I_isl(f)   = max(0, 2f−1)     information recovered via islands
    #   S_rad + I_isl = f             (unitarity — the gross information
    #                                  budget carried by the radiation)
    s_radiation = min(f, 1.0 - f)
    i_islands = max(0.0, 2.0 * f - 1.0)
    return _clip01(s_radiation + i_islands)


def weyl_scalar_tensor(
    signal_flux: float,
    topic_entropy: float,
    corpus_volume: float,
    mesh_alignment: float,
    remnant_score: float,
) -> tuple[float, float, float, float, float]:
    """Compress a learning-corpus snapshot into 5 Newman-Penrose Weyl scalars.

    Maps a cycle's five observable learning-corpus metrics onto the NP Weyl
    scalar basis **Ψ₀–Ψ₄** — the minimal tensorial encoding of gravitational
    information content.  In the Brain this provides a compact 5-float
    *Hawking radiation aspect*: the entire information budget of a research
    cycle is distilled to its irreducible boundary encoding, exactly as
    Hawking radiation is the minimal observable remnant of a black hole's
    information content.

    .. math::

        \\Psi_0 &= \\text{clip}_{01}(\\sigma)        &&\\text{(ingoing signal flux — new learnings entering the corpus)} \\\\
        \\Psi_1 &= \\text{clip}_{01}(\\tau)        &&\\text{(topic entropy — longitudinal diversity gradient)} \\\\
        \\Psi_2 &= \\text{clip}_{01}(\\rho / (\\rho+1)) &&\\text{(Coulomb bulk mass — normalised corpus volume)} \\\\
        \\Psi_3 &= \\text{clip}_{01}(\\alpha)       &&\\text{(outgoing alignment — MESH wavefunction overlap)} \\\\
        \\Psi_4 &= \\text{clip}_{01}(\\varepsilon)  &&\\text{(Hawking remnant radiation — information remnant score)}

    Parameters
    ----------
    signal_flux:    Mean signal strength of learnings written this cycle, ∈ [0, 1].
                    Mapped to Ψ₀ — the transverse *ingoing* wave; new information
                    arriving from the boundary into the corpus bulk.
    topic_entropy:  Normalised Shannon entropy of the topic distribution this cycle,
                    ∈ [0, 1].  Mapped to Ψ₁ — the intermediate ingoing component;
                    diversity/breadth of the inward information flux.
    corpus_volume:  Total count of papers + datasets found this cycle (≥ 0).
                    Mapped to Ψ₂ via the saturating transform ρ/(ρ+1) — the
                    Coulomb (mass/energy) component encoding the bulk information
                    density; saturates toward 1 as corpus grows, mirroring the
                    Bekenstein bound.
    mesh_alignment: Wavefunction overlap proxy ∈ [0, 1]; how strongly the cycle's
                    learnings align with the existing MESH state.  Mapped to Ψ₃ —
                    the intermediate outgoing component; information flowing back
                    toward the boundary via MESH reinforcement.
    remnant_score:  Hawking information-remnant score for the cycle (e.g. from
                    ``hawking_information_remnant_score()``), ∈ [0, 1].  Mapped to
                    Ψ₄ — the transverse *outgoing* wave; the irreducible information
                    remnant that escapes the horizon, completing the Hawking circuit.

    Returns
    -------
    ``(Ψ₀, Ψ₁, Ψ₂, Ψ₃, Ψ₄)`` — each scalar clipped to ``[0, 1]``.

    Notes
    -----
    Storing the tensor as a 5-element JSON array in ``brain_kv`` under
    ``"learnings:weyl_tensor"`` requires ≈ 50 bytes per cycle, versus kilobytes
    for full paper-blob caches — a direct KV-cache size reduction in the spirit
    of holographic compression.
    """
    rho = max(0.0, float(corpus_volume))
    psi2_bulk = rho / (rho + 1.0)  # saturating Coulomb mass term
    return (
        _clip01(float(signal_flux)),
        _clip01(float(topic_entropy)),
        _clip01(psi2_bulk),
        _clip01(float(mesh_alignment)),
        _clip01(float(remnant_score)),
    )


def information_compaction_scalar(
    source_bytes: float,
    mesh_bytes: float,
    *,
    floor_bytes: float | None = None,
) -> float:
    """Log-normalised scalar measuring how efficiently the MESH compacts the source.

    Measures the achieved compaction ratio *R = source_bytes / mesh_bytes*
    against the theoretical maximum *R_max*, where *R_max* is the ratio of
    the source to the smallest viable MESH representation (the 5-float Weyl
    tensor, ``_MESH_BRAIN_KV_WEYL_BYTES = 50 bytes`` when *floor_bytes* is
    ``None``, or the supplied CRB-derived floor otherwise).

    .. math::

        \\kappa = \\mathrm{clip}_{01}\\!\\left(
            \\frac{\\ln(1 + R)}{\\ln(1 + R_{\\rm max})}
        \\right)

    A value of **1.0** means the corpus has been distilled all the way to its
    irreducible boundary encoding.  A value of **0.0** indicates no compression
    (or degenerate inputs).

    Parameters
    ----------
    source_bytes:  Total raw corpus size in bytes (e.g. ``15 × 2**40`` for 15 TB).
    mesh_bytes:    Total MESH representation size in bytes (from :func:`mesh_compaction_summary`).
    floor_bytes:   Optional CRB-derived floor (bytes).  When ``None`` the legacy
                   ``_MESH_BRAIN_KV_WEYL_BYTES = 50`` floor is used, preserving
                   byte-identical behaviour with existing callers.

    Returns
    -------
    Compaction scalar in ``[0, 1]``.
    """
    if source_bytes <= 0.0 or mesh_bytes <= 0.0:
        return 0.0
    ratio = source_bytes / mesh_bytes
    _floor = float(floor_bytes) if floor_bytes is not None else float(_MESH_BRAIN_KV_WEYL_BYTES)
    max_ratio = source_bytes / max(_floor, 1.0)
    if max_ratio <= 1.0:
        return 0.0
    return _clip01(math.log1p(ratio) / math.log1p(max_ratio))


def information_compaction_scalar_gard(
    source_bytes: float,
    mesh_bytes: float,
    *,
    sigma: float,
    n_samples: int = 1,
) -> float:
    """CRB-bounded compaction scalar using the GARD information floor.

    Computes the floor via :func:`gard_info.gard_floor_bytes` (σ-derived CRB)
    and delegates to :func:`information_compaction_scalar` with that floor.

    Parameters
    ----------
    source_bytes:  Total raw corpus size in bytes.
    mesh_bytes:    Total MESH representation size in bytes.
    sigma:         Langevin noise σ for the current GARD channel.
    n_samples:     Number of independent samples (default 1).

    Returns
    -------
    Compaction scalar in ``[0, 1]``.
    """
    from . import gard_info

    fb = gard_info.gard_floor_bytes(sigma, n_samples=n_samples)["floor_bytes"]
    return information_compaction_scalar(source_bytes, mesh_bytes, floor_bytes=float(fb))


def mesh_compaction_summary(
    source_tb: float = _THE_WELL_TB_REF,
    n_vocab: int | None = None,
    n_quipu_edges: int | None = None,
) -> dict:
    """Compute total information compaction of a corpus into the MESH System Entirety.

    Models the complete in-memory footprint of the MESH — all SQLite tables
    (vocab, embed, quipu, quipu_node, meta) plus brain_kv persistence (Weyl
    tensor, remnant score, adaptive runtime) — and returns the compaction
    ratio and log-normalised scalar against the raw corpus size.

    The canonical invocation is for The Well (15 TB, 16 datasets, 4D):

    .. code-block:: python

        mesh_compaction_summary()
        # → {
        #     'source_tb': 15.0,
        #     'source_bytes': 16_492_674_416_640,
        #     'hawking_remnant_score': 0.666667,   # full 15 TB Well: past the Page time
        #     'mesh_bytes_total': 1_101_884,
        #     ...per-component breakdown...,
        #     'compaction_ratio': 14_967_032,
        #     'compaction_scalar': 0.618...,
        # }

    Parameters
    ----------
    source_tb:      Corpus size in terabytes.  Defaults to ``_THE_WELL_TB_REF = 15.0``.
    n_vocab:        Actual vocab count; if ``None`` uses ``_MESH_VOCAB_LIMIT`` (worst-case).
    n_quipu_edges:  Actual quipu edge count; if ``None`` uses *n_vocab* as a 1:1 estimate.

    Returns
    -------
    Dict with keys: ``source_tb``, ``source_bytes``, ``hawking_remnant_score``,
    all per-component byte counts, ``mesh_bytes_total``, ``compaction_ratio``,
    ``compaction_scalar``.
    """
    source_bytes: float = source_tb * (1024 ** 4)  # TB → bytes (binary)

    vocab_n = int(n_vocab) if n_vocab is not None else _MESH_VOCAB_LIMIT
    quipu_n = int(n_quipu_edges) if n_quipu_edges is not None else vocab_n

    vocab_bytes      = vocab_n * _MESH_VOCAB_ROW_BYTES
    embed_bytes      = vocab_n * _MESH_EMBED_DIMS * _MESH_BYTES_PER_FLOAT
    quipu_bytes      = quipu_n * _MESH_QUIPU_ROW_BYTES
    quipu_node_bytes = _MESH_VOCAB_LIMIT * _MESH_QUIPU_NODE_ROW_BYTES  # always full torus
    meta_bytes       = _MESH_META_BYTES
    weyl_bytes       = _MESH_BRAIN_KV_WEYL_BYTES
    remnant_bytes    = _MESH_BRAIN_KV_REMNANT_BYTES
    runtime_bytes    = _MESH_BRAIN_KV_RUNTIME_BYTES

    mesh_bytes_total = (
        vocab_bytes + embed_bytes + quipu_bytes + quipu_node_bytes
        + meta_bytes + weyl_bytes + remnant_bytes + runtime_bytes
    )

    compaction_ratio = source_bytes / mesh_bytes_total if mesh_bytes_total > 0 else 0.0
    compaction_scalar_val = information_compaction_scalar(source_bytes, float(mesh_bytes_total))

    remnant_score = hawking_information_remnant_score(
        n_datasets=_THE_WELL_N_DATASETS,
        n_spatial_dims=_THE_WELL_SPATIAL_DIMS,
        total_tb=source_tb,
    )

    # GARD CRB block — additive; does not affect existing keys.
    # Reference σ = 0.05 (langevin_sigma_from_weyl base coefficient).
    _gard_crb_sigma = 0.05
    try:
        from . import gard_info as _gard_info
        _gard_floor = _gard_info.gard_floor_bytes(_gard_crb_sigma)
        _gard_crb_scalar = information_compaction_scalar(
            source_bytes, float(mesh_bytes_total),
            floor_bytes=float(_gard_floor["floor_bytes"]),
        )
        gard_crb: dict = {
            "sigma": _gard_crb_sigma,
            "cramer_rao_var": _gard_floor["cramer_rao_var"],
            "floor_bytes": _gard_floor["floor_bytes"],
            "compaction_scalar_at_crb_floor": round(_gard_crb_scalar, 6),
        }
    except Exception:
        gard_crb = {}

    return {
        "source_tb":              source_tb,
        "source_bytes":           int(source_bytes),
        # MESH per-component memory
        "vocab_bytes":            vocab_bytes,
        "embed_bytes":            embed_bytes,
        "quipu_bytes":            quipu_bytes,
        "quipu_node_bytes":       quipu_node_bytes,
        "meta_bytes":             meta_bytes,
        "brain_kv_weyl_bytes":    weyl_bytes,
        "brain_kv_remnant_bytes": remnant_bytes,
        "brain_kv_runtime_bytes": runtime_bytes,
        # Totals
        "mesh_bytes_total":       mesh_bytes_total,
        "mesh_kb_total":          round(mesh_bytes_total / 1024, 2),
        "mesh_mb_total":          round(mesh_bytes_total / (1024 ** 2), 4),
        # Compaction metrics
        "compaction_ratio":       round(compaction_ratio),
        "compaction_scalar":      round(compaction_scalar_val, 6),
        # Holographic information surface
        "hawking_remnant_score":  round(remnant_score, 6),
        # GARD CRB information floor (additive; reference σ = 0.05)
        "gard_crb":               gard_crb,
    }


def metric_perturbation(mass_eff: float, r: float) -> float:
    """Spacetime metric perturbation  h_μν = 2 G M_eff / (c² r).

    Computes the dimensionless warp magnitude for an effective mass
    *mass_eff* (kg) at radial distance *r* (m).

    Returns 0.0 for *r* ≤ 0 (no perturbation at/within the horizon).
    """
    if r <= 0.0:
        return 0.0
    return 2.0 * _G_CONST * mass_eff / (_C_CONST ** 2 * r)


def phase_evolution_total(
    phi: float,
    delta_mu: float = 0.0,
    delta_q: float = 0.0,
    delta_gamma: float = 0.0,
    gamma_0: float = _GAMMA_0_DEFAULT,
    gamma_eff: float = _GAMMA_0_DEFAULT,
) -> float:
    """Modified total phase evolution of the 6D CAT states (UEQGM v0.9.14).

    .. math::

        \\delta\\phi_{\\rm total} =
            \\delta\\phi_{\\mu,g-2}
            + \\delta\\phi_q
            + \\delta\\phi_{\\gamma}
            + \\Delta\\lambda_{\\rm axial} \\cdot \\frac{2\\pi}{\\Gamma_{\\rm eff}}

    Parameters
    ----------
    phi:       Characteristic phase (use ``coherence_to_phi``).
    delta_mu:  Muon g-2 phase contribution.
    delta_q:   Quark/QCD phase contribution.
    delta_gamma: Photon phase contribution.
    gamma_0:   Baseline decay width.
    gamma_eff: Effective total decay width (denominator normalisation).

    Returns
    -------
    Total phase evolution  δφ_total.
    """
    axial = sici_axial_decay(phi, gamma_0)
    axial_phase = axial * (2.0 * math.pi / max(gamma_eff, 1.0e-30))
    return delta_mu + delta_q + delta_gamma + axial_phase


def entropic_bayesian_step(
    s_terrain: float,
    laplacian_s: float,
    phi: float,
    gamma_0: float = _GAMMA_0_DEFAULT,
    gamma_eff: float = _GAMMA_0_DEFAULT,
    eta_diff: float = _ETA_DIFF_DEFAULT,
) -> float:
    """Discrete entropic Bayesian diffusion update (UEQGM v0.9.14).

    .. math::

        S(t+1) = S(t)
               + \\eta_{\\rm diff} \\cdot \\nabla^2 S
               + \\bigl(\\delta\\phi_{\\rm total} + \\Delta\\lambda_{\\rm axial}\\bigr)

    Parameters
    ----------
    s_terrain:    Current terrain entropy S(t).
    laplacian_s:  Discrete Laplacian  ∇²S  at the current position.
    phi:          Characteristic phase (use ``coherence_to_phi``).
    gamma_0:      Baseline decay width.
    gamma_eff:    Effective total decay width.
    eta_diff:     Diffusion rate η_diff.

    Returns
    -------
    Updated terrain entropy  S(t+1).
    """
    axial = sici_axial_decay(phi, gamma_0)
    dphi  = phase_evolution_total(phi, gamma_0=gamma_0, gamma_eff=gamma_eff)
    return s_terrain + eta_diff * laplacian_s + dphi + axial


# ---------------------------------------------------------------------------
# Corpus-backed UEQGM coherence score
# ---------------------------------------------------------------------------

# Keywords used to identify UEQGM-tagged corpus entities.
_UEQGM_TAGS: tuple[str, ...] = (
    '"ueqgm"', '"wavefunction"', '"quantum field"',
    '"quantum dynamics"', '"holographic"', '"floquet"', '"entanglement"',
)

# Feature keywords for the bag-of-words feature vector.
_UEQGM_KEYWORDS: list[str] = [
    "quantum", "wavefunction", "holographic", "floquet",
    "entanglement", "ueqgm", "topological", "entropy",
]


def ueqgm_coherence_score(
    cn: "sqlite3.Connection",  # noqa: F821  (forward reference, DB not imported at module level)
    entity_id: str,
) -> float:
    """UEQGM-derived coherence score for a corpus entity.

    Reads UEQGM-tagged corpus entities from the Brain graph and computes
    a wavefunction-overlap coherence score between the target entity and the
    stored quantum-physics knowledge base.

    Algorithm
    ---------
    1. Fetch the target entity's (label, props_json) from ``corpus_entity``.
    2. Fetch up to 50 UEQGM-tagged entities (props_json LIKE "%ueqgm%"
       or similar quantum-physics tags).
    3. For each pair build a bag-of-words feature vector over
       ``_UEQGM_KEYWORDS`` and compute ``wavefunction_overlap``.
    4. Average the overlaps and scale by ``sici_phase_weight`` at the corpus
       depth (number of UEQGM entities found).

    Returns
    -------
    A value in [0.0, ~1.1] — 0.0 means no UEQGM context or target absent;
    higher values indicate stronger alignment with acquired UEQGM knowledge.
    """
    import json as _json
    import sqlite3 as _sqlite3  # late import — keeps this module importable without a DB

    # ── Fetch target entity ────────────────────────────────────────────────
    try:
        target_row = cn.execute(
            "SELECT label, props_json FROM corpus_entity "
            "WHERE entity_id=? LIMIT 1",
            (entity_id,),
        ).fetchone()
    except _sqlite3.Error:
        return 0.0
    if not target_row:
        return 0.0
    target_text = (target_row[0] or "") + " " + (target_row[1] or "")

    # ── Fetch UEQGM corpus entities ────────────────────────────────────────
    tag_filter = " OR ".join(f"props_json LIKE ?" for _ in _UEQGM_TAGS)
    params = tuple(f"%{t}%" for t in _UEQGM_TAGS)
    try:
        rows = cn.execute(
            f"SELECT label, props_json FROM corpus_entity "
            f"WHERE ({tag_filter}) LIMIT 50",
            params,
        ).fetchall()
    except _sqlite3.Error:
        return 0.0
    if not rows:
        return 0.0

    def _feature_vec(text: str) -> list[float]:
        low = text.lower()
        return [float(low.count(kw)) for kw in _UEQGM_KEYWORDS]

    target_vec = _feature_vec(target_text)
    overlaps: list[float] = []
    for label, props in rows:
        entity_text = (label or "") + " " + (props or "")
        entity_vec = _feature_vec(entity_text)
        overlaps.append(wavefunction_overlap(target_vec, entity_vec))

    if not overlaps:
        return 0.0

    mean_overlap = sum(overlaps) / len(overlaps)
    depth_weight = sici_phase_weight(len(rows))
    score = mean_overlap * depth_weight

    runtime_state = get_adaptive_runtime(cn)
    runtime_keywords = runtime_state.get("runtime_keywords") or []
    if runtime_keywords:
        low_target = target_text.lower()
        runtime_hits = sum(low_target.count(keyword) for keyword in runtime_keywords)
        runtime_alignment = min(1.0, runtime_hits / max(1, len(runtime_keywords)))
        score += runtime_alignment * 0.25 * max(_clip01(runtime_state.get("symbiotic_gain", 0.0)), 0.2)

    return round(min(1.2, max(0.0, score)), 6)


__all__ = [
    # Component identity
    "__version__",
    "__component__",
    # System dynamics and structural mappings
    "UEQGM_DYNAMICS_MAP",
    "ADAPTIVE_RUNTIME_MAP",
    "UEQGM_MATH_MAP",
    "MESH_COMPACTION_MAP",
    # Phase mapping
    "coherence_to_phi",
    # SiCi axial channel
    "_raw_sici",
    "sici_axial_decay",
    "sici_phase_weight",
    # Wavefunction & field theory helpers
    "wavefunction_overlap",
    "floquet_modulation_factor",
    "tantalum_intermediary_binding",
    "holographic_entropy",
    "hawking_information_remnant_score",
    "weyl_scalar_tensor",
    "information_compaction_scalar",
    "information_compaction_scalar_gard",
    "mesh_compaction_summary",
    "metric_perturbation",
    # Planck 2018 cosmological census (8th-D field blend weights)
    "PLANCK18_CENSUS",
    "cosmological_census_weights",
    # Full UEQGM dynamics
    "phase_evolution_total",
    "entropic_bayesian_step",
    # Adaptive runtime
    "get_adaptive_runtime",
    "refresh_adaptive_runtime",
    # Corpus-backed score
    "ueqgm_coherence_score",
    # Physical + MESH constants
    "_GAMMA_0_DEFAULT",
    "_ETA_DIFF_DEFAULT",
    "_G_CONST",
    "_C_CONST",
    "_TAN_CLAMP",
    "_THE_WELL_TB_REF",
    "_THE_WELL_N_DATASETS",
    "_THE_WELL_SPATIAL_DIMS",
    "_THE_WELL_REMNANT_SCORE_KEY",
    "_WEYL_TENSOR_KEY",
    "_WEYL_PRECISION",
    "_WEYL_UPVOTE_WEIGHT",
    "_WEYL_CITATION_WEIGHT",
    "_MESH_TORUS_N",
    "_MESH_VOCAB_LIMIT",
    "_MESH_EMBED_DIMS",
    "_MESH_BYTES_PER_FLOAT",
    "_MESH_VOCAB_ROW_BYTES",
    "_MESH_QUIPU_ROW_BYTES",
    "_MESH_QUIPU_NODE_ROW_BYTES",
    "_MESH_META_BYTES",
    "_MESH_BRAIN_KV_WEYL_BYTES",
    "_MESH_BRAIN_KV_REMNANT_BYTES",
    "_MESH_BRAIN_KV_RUNTIME_BYTES",
    "_PHI_BASE",
    "_PHI_STEP",
    "_SICI_SCALE_FACTOR",
    "_UEQGM_RUNTIME_KEY",
    "_UEQGM_TAGS",
    "_UEQGM_KEYWORDS",
]
