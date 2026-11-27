"""magic — information-theoretic readings transcribed into quantum "magic" per axis plane.

"Magic" (non-stabilizerness) measures how far a quantum state is from the
stabilizer states a classical computer simulates efficiently.  It was proposed
for, and reported in, top quark–antiquark pairs at the LHC (White & White,
"Magic states of top quarks", *Phys. Rev. D* 110, 116016 (2024)), where it is
read from the pair's spin correlations as the second stabilizer Rényi entropy

    M̃₂(ρ) = −log₂( Σ_P tr(ρP)⁴ / Σ_P tr(ρP)² ),     P over the 16 two-qubit Paulis.

Transcription used here (a declared mapping, not a measurement):

* Each CAT axis plane k carries a matter–antimatter pair of qubits,
      |ψ_k⟩ = cos(θ_k/2)|00⟩ + e^{iφ_k} sin(θ_k/2)|11⟩.
* **Information theory in:** sin²(θ_k/2) = p_k, the plane's realised pressure
  (the PCM cell the rADAM step programs); φ_k = the plane's rADAM phase
  arg(g_re + i·g_im), the latent share.
* **Antimatter attribution:** the antimatter partner is the charge-conjugate,
  φ → −φ.  M̃₂ is even in φ, so both partners carry the same magic and a
  plane's magic is attributed half to each.
* p = 0 or 1 (product state) and p = ½ at φ = 0 (Bell state) are stabilizer
  states: magic 0.  The maximum for this family is log₂(3/2) ≈ 0.585 at
  p = (1 ± 1/√3)/2 ≈ 0.211 or 0.789 (cos²θ = 1/3), φ = π/4.

**The positron (antimatter) partner is held as a one-positron quantum
cyclotron** (Myers, Soucy, Sukra, Sinha & Gabrielse, "One-Positron Quantum
Cyclotron", *Phys. Rev. Lett.* 137, 173001): a single positron suspended in a
Penning trap's magnetic field, cooled to its cyclotron ground state and moved
one quantum jump at a time, towards a positron-vs-electron magnetic-moment
test of CPT invariance.  Two consequences here:

* **Held fraction.** The cyclotron frequency is ν_c = eB/(2π m_e) and the
  thermal ground-state probability P₀ = 1 − e^{−hν_c/k_BT}.  The antimatter
  half of a plane's magic is attributed only in the held share P₀; the rest
  is reported as ``unheld``.  Defaults B = 5.36 T (ν_c ≈ 150 GHz, the order of
  the Gabrielse traps) and T = 0.3 K (as reported for this trap) give
  P₀ ≈ 1 − 4·10⁻¹¹: effectively all held.
* **CPT test.** The antimatter partner's phase is −φ·(1 + r), with
  r = (g_e⁺ − g_e⁻)/g_e⁻ the fractional magnetic-moment difference.  CPT
  invariance (r = 0, the default) makes matter and antimatter magic equal; any
  r ≠ 0 shows up as an asymmetry in the attribution, which is reported.

B, T and r are overridable (``QUIPU_POSITRON_B_T``, ``QUIPU_POSITRON_T_K``,
``QUIPU_CPT_RATIO``).  The PRL page was not readable from the session that
wrote this; the trap description comes from its abstract as indexed and press
coverage, and no value here is taken from its data.

**Non-linear operational gain from recent proofs:**

* *Monotone.* M_α with integer α ≥ 2 is a magic monotone on pure states
  (Leone & Bittel, *Phys. Rev. A* 110, L040403 (2024)), so M̃₂ here cannot be
  raised by free (stabilizer) operations.
* *Proved maximum.* The largest two-qubit M₂ is log₂(16/7) ≈ 1.193 bits
  (numerically, "Maximal Magic for Two-qubit States", *Quantum Sci. Technol.*
  11, 015035 (2026); exact proofs of the two-qubit maximisers claimed by Shen,
  Yin & Zhang, arXiv:2611.08586).  The pair family above reaches only
  log₂(3/2).  At the maximum the stabilizer purity 2^{−M₂} is 7/16.
* *Operational meaning.* A state's Clifford orbit is an approximate design,
  indistinguishable from Haar-random states up to an error exp(−Θ(M_α)), α ≥ 2
  (Bittel & Leone, arXiv:2507.22883).  Taking the Θ constant as 1 with M in
  nats (a declared choice), the error is 2^{−M₂}, which is exactly the
  stabilizer purity Σ⟨P⟩⁴/Σ⟨P⟩².

  :func:`operational_gain` maps this to [1, φ]:

      g(M₂) = 1 + (φ − 1) · (1 − 2^{−M₂}) / (1 − 7/16)

  It is concave and saturating: steep for little magic, flat near the proved
  maximum, where g = φ.  The pair family tops out at g ≈ 1.366.

Pure functions, standard library only; no I/O beyond reading those three
environment variables.
"""
from __future__ import annotations

import math
import os
from typing import Mapping, Optional, Sequence

AXES: tuple[str, ...] = ("vision", "touch", "smell", "body", "brain", "perception", "entirety")
MAX_PAIR_MAGIC: float = math.log2(1.5)
MAX_TWO_QUBIT_MAGIC: float = math.log2(16.0 / 7.0)   # proved two-qubit maximum of M2, in bits
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
E_CHARGE: float = 1.602176634e-19       # C (exact, SI)
M_ELECTRON: float = 9.1093837015e-31    # kg (CODATA 2018; the positron mass is equal under CPT)
H_PLANCK: float = 6.62607015e-34        # J s (exact)
K_BOLTZMANN: float = 1.380649e-23       # J/K (exact)
POSITRON_B_T: float = 5.36
POSITRON_T_K: float = 0.3
CPT_RATIO: float = 0.0
P_AT_MAX: float = (1.0 - 1.0 / math.sqrt(3.0)) / 2.0


def stabilizer_renyi_2(pauli_expectations: Sequence[float]) -> float:
    """M̃₂ = −log₂(Σ⟨P⟩⁴ / Σ⟨P⟩²) over the Pauli expectations given (identity included)."""
    xs = [float(x) for x in pauli_expectations]
    if not xs or not all(math.isfinite(x) for x in xs):
        raise ValueError("pauli expectations must be finite and non-empty")
    s2 = sum(x * x for x in xs)
    s4 = sum(x ** 4 for x in xs)
    if s2 <= 0.0 or s4 <= 0.0:
        raise ValueError("pauli expectations must not all vanish")
    return max(0.0, -math.log2(s4 / s2))


def pair_paulis(theta: float, phi: float) -> dict[str, float]:
    """Non-zero Pauli expectations of cos(θ/2)|00⟩ + e^{iφ} sin(θ/2)|11⟩."""
    s, c = math.sin(theta), math.cos(theta)
    return {"II": 1.0, "ZZ": 1.0, "ZI": c, "IZ": c,
            "XX": s * math.cos(phi), "YY": -s * math.cos(phi),
            "XY": s * math.sin(phi), "YX": s * math.sin(phi)}


def pair_magic(pressure: float, phase: float) -> float:
    """Magic of one plane's pair: sin²(θ/2) = pressure, φ = phase."""
    p, ph = float(pressure), float(phase)
    if not (math.isfinite(p) and math.isfinite(ph)):
        raise ValueError("pressure and phase must be finite")
    p = min(1.0, max(0.0, p))
    theta = 2.0 * math.asin(math.sqrt(p))
    return stabilizer_renyi_2(list(pair_paulis(theta, ph).values()))


def stabilizer_purity(magic_bits: float) -> float:
    """2^{−M₂} = Σ⟨P⟩⁴/Σ⟨P⟩²: 1 for stabilizer states, 7/16 at the proved two-qubit maximum."""
    m = float(magic_bits)
    if not math.isfinite(m):
        raise ValueError("magic must be finite")
    return 2.0 ** (-max(0.0, m))


def design_indistinguishability(magic_bits: float) -> float:
    """1 − 2^{−M₂} normalised by its proved two-qubit maximum 9/16: 0 for stabilizer states, 1 at the maximum."""
    return min(1.0, (1.0 - stabilizer_purity(magic_bits)) / (1.0 - 7.0 / 16.0))


def operational_gain(magic_bits: float) -> float:
    """g = 1 + (φ − 1)·design_indistinguishability(M₂) ∈ [1, φ]: concave, saturating at the proved maximum."""
    return 1.0 + (PHI - 1.0) * design_indistinguishability(magic_bits)


def quantum_cyclotron(b_tesla: float = POSITRON_B_T, temp_k: float = POSITRON_T_K) -> dict:
    """One-positron quantum cyclotron: ν_c = eB/(2π m), n̄ and ground-state probability P₀."""
    b, t = float(b_tesla), float(temp_k)
    if not (math.isfinite(b) and b > 0.0 and math.isfinite(t) and t > 0.0):
        raise ValueError("b_tesla and temp_k must be finite and > 0")
    nu = E_CHARGE * b / (2.0 * math.pi * M_ELECTRON)
    x = H_PLANCK * nu / (K_BOLTZMANN * t)
    n_bar = 1.0 / math.expm1(x) if x < 700.0 else 0.0
    p0 = -math.expm1(-x)
    return {"b_tesla": b, "temp_k": t, "nu_c_hz": nu, "h_nu_over_kT": x,
            "n_bar": n_bar, "ground_probability": p0}


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def positron_config() -> dict:
    """B, T and the CPT ratio r, from the environment or the defaults."""
    return {"b_tesla": _env_float("QUIPU_POSITRON_B_T", POSITRON_B_T),
            "temp_k": _env_float("QUIPU_POSITRON_T_K", POSITRON_T_K),
            "cpt_ratio": _env_float("QUIPU_CPT_RATIO", CPT_RATIO)}


def reading(pressures: Mapping[str, float], phases: Optional[Mapping[str, float]] = None,
            weights: Optional[Sequence[float]] = None,
            source_axes: Optional[Mapping[str, str]] = None,
            b_tesla: Optional[float] = None, temp_k: Optional[float] = None,
            cpt_ratio: Optional[float] = None) -> dict:
    """Magic per axis plane, its weighted total, and the attribution per source.

    ``weights`` default to the rheology axis weights.  ``source_axes`` maps a
    source name to its axis (the observer's SOURCE_PROFILES); each source is
    attributed the magic of its own plane.
    """
    if weights is None:
        from .rheology import axis_weights
        weights = axis_weights()
    w = [max(0.0, float(x)) for x in weights]
    if len(w) != len(AXES) or sum(w) <= 0.0:
        raise ValueError("weights must be 7 non-negative numbers with a positive sum")
    tot = sum(w)
    phases = phases or {}
    env = positron_config()
    cyc = quantum_cyclotron(env["b_tesla"] if b_tesla is None else b_tesla,
                            env["temp_k"] if temp_k is None else temp_k)
    r = env["cpt_ratio"] if cpt_ratio is None else float(cpt_ratio)
    if not math.isfinite(r):
        raise ValueError("cpt_ratio must be finite")
    held = cyc["ground_probability"]
    planes = {}
    total = 0.0
    asym = 0.0
    for a, wk in zip(AXES, w):
        p = float(pressures.get(a, 0.0) or 0.0)
        ph = float(phases.get(a, 0.0) or 0.0)
        if not math.isfinite(p):
            p = 0.0
        if not math.isfinite(ph):
            ph = 0.0
        m = pair_magic(p, ph)
        m_bar = pair_magic(p, -ph * (1.0 + r))          # the positron partner's own magic
        planes[a] = {"pressure": round(min(1.0, max(0.0, p)), 6), "phase": round(ph, 6),
                     "magic": round(m, 6), "matter": round(m / 2.0, 6),
                     "antimatter": round(held * m_bar / 2.0, 6),
                     "unheld": round((1.0 - held) * m_bar / 2.0, 12),
                     "cpt_asymmetry": round(m_bar - m, 12)}
        total += (wk / tot) * m
        asym = max(asym, abs(m_bar - m))
    for rec in planes.values():
        rec["design"] = round(design_indistinguishability(rec["magic"]), 9)
    out = {"planes": planes, "total": round(total, 6), "max": round(MAX_PAIR_MAGIC, 6),
           "share_of_max": round(total / MAX_PAIR_MAGIC, 6),
           "max_two_qubit": round(MAX_TWO_QUBIT_MAGIC, 6),
           "share_of_proved_max": round(total / MAX_TWO_QUBIT_MAGIC, 6),
           "operational_gain": round(operational_gain(total), 9),
           "positron": {**{k: (v if k in ("n_bar", "ground_probability") else round(v, 6)) for k, v in cyc.items()},
                        "source": "one-positron quantum cyclotron, PRL 137, 173001"},
           "cpt": {"ratio": r, "max_asymmetry": asym, "conserved": asym < 1e-12}}
    if source_axes:
        out["by_source"] = {s: planes[a]["magic"] for s, a in sorted(source_axes.items()) if a in planes}
    return out


__all__ = ["AXES", "MAX_PAIR_MAGIC", "MAX_TWO_QUBIT_MAGIC", "P_AT_MAX", "stabilizer_purity",
           "design_indistinguishability", "operational_gain", "quantum_cyclotron", "positron_config", "stabilizer_renyi_2", "pair_paulis", "pair_magic", "reading"]
