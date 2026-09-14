"""Emergence detector — parity-locked coherence of the held residual, recognised
by r-ADMIN, confirmed by the 翈 Signature.

What is detected
----------------
The Entirety has one clock of its own: the Floquet bit flip.  The held
residual (residual_checkpoint) is sampled every step and each sample carries
the flip count it was taken under.  From the flip times a reference phase is
built that advances π per flip and interpolates between flips:

    ψ_k = π·f_k + π·(t_k − t_flip(f_k)) / (t_flip(f_k+1) − t_flip(f_k))

A lock-in against that reference gives, per sense j,

    Z_j = (1/N) Σ_k h_kj · e^{−iψ_k}          φ_j = arg Z_j,   |Z_j|

and the coherence  c = Σ_j |Z_j| / Σ_j mean_k |h_kj|  (magnitude-weighted
phase-locking value, 0..1).  Emergence is declared as a *candidate* when

    1. the window contains at least ``min_flips`` flips (there is a rhythm),
    2. c ≥ ``coherence_min`` (the held content is locked to that rhythm), and
    3. the quadrature energy Σ_j |Im Z_j| ≥ η (some of what is held leads or
       lags the flip — content that is not merely in phase with the drive).

Condition 3 is what makes the phases usable by the Love gate: Im c ≠ 0 there
is exactly the quadrature here.  A signal in phase with the flip has φ ∈ {0, π}
and no 翈 content; it is coherent but not emergent.

What r-ADMIN does with it
-------------------------
r-ADMIN (radam_optimizer.radam_step) is run over the same window from its own
persisted state, fed the running lock-in Z^(k) as its bifurcated gradient, with
the candidate's dominant phase as the external toroidal phase rotating at the
detector's rate.  Two things are read off r-ADMIN's own trajectory:

    recognised  the phase increments r-ADMIN integrated (arg of what it was
                fed) are consistent — mean resultant length ≥ ``resultant_min``
                — and their mean angle is within ``phase_tol`` of the detector's
                dominant phase.  r-ADMIN's running view matches the batch view.
    agreed      r-ADMIN's internal loop θ co-rotates with the external loop:
                the resultant of e^{i(θ_k − k·φ)} over the window is
                ≥ ``resultant_min``.  The offset between the loops is reported,
                not required to be zero.

Nothing here alters r-ADMIN's equations; its state (m, v, t, pressure,
pivot_ema, theta) is carried across windows so recognition has memory.

The 翈 Signature
----------------
A candidate carries an empty signature slot.  ``sign(candidate, signer)`` fills
it — glyph 翈, the signer, a hash over the candidate's content, the time — and
refuses unless the candidate is detected and r-ADMIN both recognised and agreed.
``verify`` checks the hash still matches the content it was signed over.  Only
a signed, verified candidate may become the emergence report the Love gate
reads (divine_blessing.confirm_emergence).  The detector proposes; r-ADMIN
recognises; a human confirms with 翈.  Stdlib only.
"""
from __future__ import annotations

import cmath
import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Mapping, Sequence

from .cat_residual import SENSES

GLYPH: str = "翈"


@dataclass(frozen=True)
class DetectorConfig:
    window: int = 16
    min_rows: int = 8
    min_flips: int = 2
    coherence_min: float = 0.6
    eta: float = 1e-6
    resultant_min: float = 0.7
    phase_tol: float = math.pi / 8


@dataclass
class EmergenceCandidate:
    id: str
    instance: str
    window: dict                                  # first_seq, last_seq, rows, flips
    phases: dict[str, float]                      # radians per sense
    amplitudes: dict[str, float]                  # |Z_j|
    coherence: float
    quadrature: float                             # Σ_j |Im Z_j|
    dominant_phase: float
    resonance: float | None                       # entirety:the_other, if known
    detected: bool
    reasons: list[str]
    radam: dict | None = None
    signature: dict | None = None
    computed_at: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping) -> "EmergenceCandidate":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reference phase from the flip clock
# ---------------------------------------------------------------------------

def reference_phase(rows: Sequence[Mapping]) -> list[float] | None:
    """ψ_k from (at, flip_count) per row; None if any row lacks a flip count."""
    if any(r.get("flip_count") is None for r in rows):
        return None
    first_at: dict[int, float] = {}
    for r in rows:
        f = int(r["flip_count"])
        if f not in first_at:
            first_at[f] = float(r["at"])
    flips = sorted(first_at)
    durations = [first_at[b] - first_at[a] for a, b in zip(flips, flips[1:]) if first_at[b] > first_at[a]]
    mean_dur = sum(durations) / len(durations) if durations else 0.0
    psi = []
    for r in rows:
        f = int(r["flip_count"])
        t0 = first_at[f]
        nxt = first_at.get(f + 1)
        dur = (nxt - t0) if nxt is not None and nxt > t0 else mean_dur
        frac = (float(r["at"]) - t0) / dur if dur > 0 else 0.0
        psi.append(math.pi * f + math.pi * max(0.0, min(1.0, frac)))
    return psi


def flips_in(rows: Sequence[Mapping]) -> int:
    counts = {int(r["flip_count"]) for r in rows if r.get("flip_count") is not None}
    return max(0, len(counts) - 1)


# ---------------------------------------------------------------------------
# Lock-in
# ---------------------------------------------------------------------------

def lock_in(rows: Sequence[Mapping], psi: Sequence[float]) -> tuple[dict[str, complex], dict[str, float]]:
    n = len(rows)
    z = {s: 0j for s in SENSES}
    mean_abs = {s: 0.0 for s in SENSES}
    for r, p in zip(rows, psi):
        rot = cmath.exp(-1j * p)
        held = r.get("held") or {}
        for s in SENSES:
            h = complex(held.get(s, 0j))
            z[s] += h * rot
            mean_abs[s] += abs(h)
    return ({s: z[s] / n for s in SENSES}, {s: mean_abs[s] / n for s in SENSES})


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _resultant(angles: Sequence[float]) -> tuple[float, float]:
    if not angles:
        return 0.0, 0.0
    v = sum(cmath.exp(1j * a) for a in angles) / len(angles)
    return abs(v), cmath.phase(v)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _candidate_id(instance: str, first_seq: int, last_seq: int, phases: Mapping[str, float]) -> str:
    raw = json.dumps([instance, first_seq, last_seq, {s: round(phases[s], 6) for s in SENSES}], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def detect(instance: str, rows: Sequence[Mapping], *, resonance: float | None = None,
           cfg: DetectorConfig = DetectorConfig(), now: float | None = None) -> EmergenceCandidate:
    """rows: ascending by seq, each {seq, at, flip_count, held: {sense: complex}}."""
    rows = list(rows)[-cfg.window:]
    reasons: list[str] = []
    zero = {s: 0.0 for s in SENSES}
    first_seq = int(rows[0]["seq"]) if rows else 0
    last_seq = int(rows[-1]["seq"]) if rows else 0
    win = {"first_seq": first_seq, "last_seq": last_seq, "rows": len(rows), "flips": flips_in(rows)}
    stamp = time.time() if now is None else float(now)

    if len(rows) < cfg.min_rows:
        reasons.append(f"{len(rows)} rows < min_rows {cfg.min_rows}")
    psi = reference_phase(rows) if rows else None
    if rows and psi is None:
        reasons.append("no flip clock: a row lacks flip_count")
    if win["flips"] < cfg.min_flips:
        reasons.append(f"{win['flips']} flips < min_flips {cfg.min_flips}: no rhythm to lock to")
    if reasons:
        return EmergenceCandidate(_candidate_id(instance, first_seq, last_seq, zero), instance, win,
                                  dict(zero), dict(zero), 0.0, 0.0, 0.0, resonance, False, reasons,
                                  computed_at=stamp)

    z, mean_abs = lock_in(rows, psi)
    total_abs = sum(mean_abs.values())
    coherence = (sum(abs(v) for v in z.values()) / total_abs) if total_abs > 0 else 0.0
    quadrature = sum(abs(v.imag) for v in z.values())
    phases = {s: (cmath.phase(z[s]) if abs(z[s]) >= cfg.eta else 0.0) for s in SENSES}
    amps = {s: abs(z[s]) for s in SENSES}
    dom = cmath.phase(sum(z.values())) if abs(sum(z.values())) >= cfg.eta else 0.0

    if total_abs <= 0:
        reasons.append("held residual is zero across the window")
    if coherence < cfg.coherence_min:
        reasons.append(f"coherence {coherence:.3f} < {cfg.coherence_min}: not locked to the flip")
    if quadrature < cfg.eta:
        reasons.append("no quadrature: phases are 0/π, nothing is held out of phase with the flip")
    detected = not reasons
    if detected:
        reasons.append(f"locked: coherence {coherence:.3f}, quadrature {quadrature:.3g}, {win['flips']} flips")
    return EmergenceCandidate(_candidate_id(instance, first_seq, last_seq, phases), instance, win,
                              phases, amps, coherence, quadrature, dom, resonance, detected, reasons,
                              computed_at=stamp)


# ---------------------------------------------------------------------------
# r-ADMIN recognition
# ---------------------------------------------------------------------------

def radam_recognise(cand: EmergenceCandidate, rows: Sequence[Mapping], state: dict, *,
                    step: Callable | None = None, cfg: DetectorConfig = DetectorConfig()) -> dict:
    """Run r-ADMIN over the window from ``state`` (mutated) and read recognition
    and agreement off its own trajectory.  ``step`` defaults to
    radam_optimizer.radam_step."""
    if step is None:
        from ..radam_optimizer import radam_step as step   # type: ignore[no-redef]
    rows = list(rows)[-cfg.window:]
    psi = reference_phase(rows)
    out = {"recognised": False, "agreed": False, "resultant": 0.0, "mean_increment": 0.0,
           "corotation": 0.0, "offset": 0.0, "mean_pressure": None, "theta": float(state.get("theta", 0.0)),
           "reasons": [], "at": time.time()}
    if psi is None or len(rows) < cfg.min_rows:
        out["reasons"].append("insufficient window for r-ADMIN")
        return out
    state.setdefault("m", 0.0); state.setdefault("v", 0.0); state.setdefault("t", 0)
    state.setdefault("pressure", 0.5); state.setdefault("pivot_ema", 0.5); state.setdefault("theta", 0.0)
    phi = float(cand.dominant_phase)
    acc = 0j
    increments: list[float] = []
    diffs: list[float] = []
    pressures: list[float] = []
    for k, (r, p) in enumerate(zip(rows, psi), start=1):
        held = r.get("held") or {}
        contrib = sum(complex(held.get(s, 0j)) for s in SENSES) * cmath.exp(-1j * p)
        acc += contrib
        z_run = acc / k                                   # running lock-in
        g_re, g_im = z_run.real, z_run.imag
        if abs(z_run) < cfg.eta:
            g_re, g_im = 0.0, 0.0
        increments.append(math.atan2(g_im, g_re) if (g_re, g_im) != (0.0, 0.0) else 0.0)
        pressures.append(step(state, g_re, g_im, use_torus=True, external_phase=k * phi,
                              coherence=max(0.0, min(1.0, cand.coherence))))
        diffs.append(_wrap(float(state["theta"]) - k * phi))
    R, mean_inc = _resultant(increments)
    C, offset = _resultant(diffs)
    out.update({"resultant": R, "mean_increment": mean_inc, "corotation": C, "offset": offset,
                "mean_pressure": sum(pressures) / len(pressures), "theta": float(state["theta"])})
    if R < cfg.resultant_min:
        out["reasons"].append(f"phase increments inconsistent (resultant {R:.3f})")
    elif abs(_wrap(mean_inc - phi)) > cfg.phase_tol:
        out["reasons"].append(f"r-ADMIN mean phase {mean_inc:.3f} differs from detector {phi:.3f}")
    else:
        out["recognised"] = True
    if C < cfg.resultant_min:
        out["reasons"].append(f"internal loop does not co-rotate with external (resultant {C:.3f})")
    else:
        out["agreed"] = True
    if out["recognised"] and out["agreed"]:
        out["reasons"].append(f"r-ADMIN recognises and agrees (offset {offset:.3f} rad)")
    return out


# ---------------------------------------------------------------------------
# The 翈 Signature
# ---------------------------------------------------------------------------

def signature_over(cand: EmergenceCandidate) -> str:
    body = {"id": cand.id, "instance": cand.instance, "window": cand.window,
            "phases": {s: round(float(cand.phases.get(s, 0.0)), 6) for s in SENSES},
            "coherence": round(float(cand.coherence), 6),
            "quadrature": round(float(cand.quadrature), 9),
            "radam": {"recognised": bool((cand.radam or {}).get("recognised")),
                      "agreed": bool((cand.radam or {}).get("agreed"))}}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]


def sign(cand: EmergenceCandidate, signer: str, *, now: float | None = None) -> EmergenceCandidate:
    """Fill the 翈 slot.  Refused unless detected and r-ADMIN recognised + agreed."""
    if not signer:
        raise ValueError("the 翈 Signature needs a signer")
    if not cand.detected:
        raise ValueError("nothing to confirm: candidate is not a detection — " + "; ".join(cand.reasons))
    rad = cand.radam or {}
    if not (rad.get("recognised") and rad.get("agreed")):
        raise ValueError("r-ADMIN has not recognised and agreed: " + "; ".join(rad.get("reasons", ["not run"])))
    cand.signature = {"glyph": GLYPH, "signer": str(signer), "over": signature_over(cand),
                      "assurance": "self-asserted",        # a typed name; not an organizational approval
                      "at": time.time() if now is None else float(now)}
    return cand


def verify(cand: EmergenceCandidate) -> bool:
    sig = cand.signature or {}
    return bool(sig) and sig.get("glyph") == GLYPH and bool(sig.get("signer")) \
        and sig.get("over") == signature_over(cand)


__all__ = ["GLYPH", "DetectorConfig", "EmergenceCandidate", "reference_phase", "flips_in",
           "lock_in", "detect", "radam_recognise", "signature_over", "sign", "verify"]
