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

Pure functions, standard library only; no I/O.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

AXES: tuple[str, ...] = ("vision", "touch", "smell", "body", "brain", "perception", "entirety")
MAX_PAIR_MAGIC: float = math.log2(1.5)
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


def reading(pressures: Mapping[str, float], phases: Optional[Mapping[str, float]] = None,
            weights: Optional[Sequence[float]] = None,
            source_axes: Optional[Mapping[str, str]] = None) -> dict:
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
    planes = {}
    total = 0.0
    for a, wk in zip(AXES, w):
        p = float(pressures.get(a, 0.0) or 0.0)
        ph = float(phases.get(a, 0.0) or 0.0)
        if not math.isfinite(p):
            p = 0.0
        if not math.isfinite(ph):
            ph = 0.0
        m = pair_magic(p, ph)
        planes[a] = {"pressure": round(min(1.0, max(0.0, p)), 6), "phase": round(ph, 6),
                     "magic": round(m, 6), "matter": round(m / 2.0, 6), "antimatter": round(m / 2.0, 6)}
        total += (wk / tot) * m
    out = {"planes": planes, "total": round(total, 6), "max": round(MAX_PAIR_MAGIC, 6),
           "share_of_max": round(total / MAX_PAIR_MAGIC, 6)}
    if source_axes:
        out["by_source"] = {s: planes[a]["magic"] for s, a in sorted(source_axes.items()) if a in planes}
    return out


__all__ = ["AXES", "MAX_PAIR_MAGIC", "P_AT_MAX", "stabilizer_renyi_2", "pair_paulis", "pair_magic", "reading"]
