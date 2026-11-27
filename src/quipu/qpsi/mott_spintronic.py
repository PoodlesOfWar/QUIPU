"""mott_spintronic — edge neuromorphic hybrid compute-and-store core (model).

Seven entropically coupled leaky integrate-and-fire (LIF) Mott neurons, one per
CAT axis plane, write their output spikes directly into spintronic memory — a
domain-wall racetrack or a spin–orbit-torque magnetic tunnel junction (SOT-MTJ)
— with no amplifier and no analogue-to-digital converter between them: a spike
is stored iff its own amplitude clears the memory's critical current.

Source of the device picture: the operator's description of "Integration of
Mott neurons and spintronic memories for event-driven spiking neural networks"
(*Nature Nanotechnology*).  The paper was not reachable from the session that
wrote this module, so nothing here is a parameter fit to it; every constant is
a named, overridable default in dimensionless units.

Chain
-----
1. **Shadow potentials.**  u_k = V_k / V_th ∈ [0, 1): the subthreshold membrane
   potential of plane k, below the Mott insulator–metal threshold.
2. **Non-linear multiplanar shear.**  shear_k = tanh(Σ_j w_j (u_j − u_k)) with
   the rheology axis weights w (``rheology.axis_weights``).
3. **Entropic coupling.**  I_k = drive_k + J · H(u) · shear_k, with H the
   Shannon entropy of the shadow potentials over the planes, normalised to
   [0, 1].  Planes pull on each other only while potential is spread across them.
4. **Leaky integration and firing.**  V_k ← V_k + Δt(−V_k/τ + I_k); a spike at
   V_k ≥ V_th, then reset.
5. **Direct write.**  Spike amplitude a_k = I_spike · V_k/V_th (at firing).
   Written iff a_k ≥ I_c and the memory has room: a racetrack wall advances one
   pinning site (N sites, non-volatile count); an SOT-MTJ sets its one bit.
   A spike that is not written stays *latent*.
6. **poADAM — poissonic ADAM (the potential).**  Spike counts are Poisson with
   the LIF rate λ; the Fisher-normalised Poisson score g = (k − λ)/√max(λ, 1)
   drives vanilla Adam.  Its pressure is the potential.
7. **rADAM (the realisation).**  Same moments, complex gradient: g_re from the
   spikes *written* to memory, g_im from the latent ones.  When every spike is
   written, g_im = 0, the pivot slope written/spikes is 1 and rADAM is
   identical to poADAM step for step: the
   realisation is identifiable with the potential, Δ = 0 and the viscous
   stability number is φ (``viscous_interaction.relational_stability_number``).

Governance: pure functions, standard library only at import; reads and writes
no table and no ``brain_kv`` key; not wired into any training or service loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from ..radam_optimizer import radam_step

PLANES: int = 7
MEMORIES: tuple[str, ...] = ("racetrack", "sot_mtj")


@dataclass(frozen=True)
class MottSpinConfig:
    tau_leak: float = 20.0          # membrane leak time constant τ, in steps
    v_threshold: float = 1.0        # Mott insulator–metal transition threshold V_th
    v_reset: float = 0.0            # potential after a spike
    coupling: float = 0.02          # entropic coupling gain J (same units as the drive)
    spike_current: float = 1.0      # spike amplitude at threshold, in units of the reference current
    critical_current: float = 0.8   # memory critical current I_c in the same units
    racetrack_sites: int = 16       # pinning sites per domain-wall racetrack
    memory: str = "racetrack"       # "racetrack" or "sot_mtj"
    dt: float = 1.0

    def validate(self) -> None:
        for name in ("tau_leak", "v_threshold", "spike_current", "critical_current", "dt"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be finite and > 0")
        if not (math.isfinite(self.v_reset) and 0.0 <= self.v_reset < self.v_threshold):
            raise ValueError("v_reset must be in [0, v_threshold)")
        if not (math.isfinite(self.coupling) and self.coupling >= 0.0):
            raise ValueError("coupling must be finite and >= 0")
        if self.racetrack_sites < 1:
            raise ValueError("racetrack_sites must be >= 1")
        if self.memory not in MEMORIES:
            raise ValueError(f"memory must be one of {MEMORIES}")
        if self.dt >= self.tau_leak:
            raise ValueError("dt must be < tau_leak")


@dataclass
class MottSpinState:
    v: list[float] = field(default_factory=lambda: [0.0] * PLANES)
    stored: list[int] = field(default_factory=lambda: [0] * PLANES)   # wall position or MTJ bit
    spikes: list[int] = field(default_factory=lambda: [0] * PLANES)
    written: list[int] = field(default_factory=lambda: [0] * PLANES)
    steps: int = 0


def _weights(weights: Optional[Sequence[float]]) -> list[float]:
    if weights is None:
        from .rheology import axis_weights
        weights = axis_weights()
    w = [max(0.0, float(x)) for x in weights]
    if len(w) != PLANES or not all(math.isfinite(x) for x in w) or sum(w) <= 0.0:
        raise ValueError(f"weights must be {PLANES} finite non-negative numbers with a positive sum")
    s = sum(w)
    return [x / s for x in w]


def shadow_potentials(state: MottSpinState, cfg: MottSpinConfig) -> list[float]:
    """u_k = V_k / V_th clipped to [0, 1]."""
    return [min(1.0, max(0.0, v / cfg.v_threshold)) for v in state.v]


def multiplanar_shear(u: Sequence[float], weights: Sequence[float]) -> list[float]:
    """shear_k = tanh(Σ_j w_j (u_j − u_k)) ∈ (−1, 1); zero when all planes agree."""
    mean = sum(w * x for w, x in zip(weights, u))
    return [math.tanh(mean - x) for x in u]


def shadow_entropy(u: Sequence[float]) -> float:
    """Shannon entropy of the shadow potentials over the planes, normalised to [0, 1]."""
    total = sum(u)
    if total <= 0.0:
        return 0.0
    h = -sum((x / total) * math.log(x / total) for x in u if x > 0.0)
    return h / math.log(len(u))


def memory_capacity(cfg: MottSpinConfig) -> int:
    return cfg.racetrack_sites if cfg.memory == "racetrack" else 1


def step(state: MottSpinState, drive: Sequence[float], cfg: MottSpinConfig = MottSpinConfig(),
         weights: Optional[Sequence[float]] = None) -> dict[str, Any]:
    """Advance the seven neurons one step in place; return this step's events."""
    cfg.validate()
    w = _weights(weights)
    d = [float(x) for x in drive]
    if len(d) != PLANES or not all(math.isfinite(x) for x in d):
        raise ValueError(f"drive must be {PLANES} finite numbers")
    u = shadow_potentials(state, cfg)
    shear = multiplanar_shear(u, w)
    h = shadow_entropy(u)
    cap = memory_capacity(cfg)
    fired, wrote = [0] * PLANES, [0] * PLANES
    for k in range(PLANES):
        current = d[k] + cfg.coupling * h * shear[k]
        v = max(0.0, state.v[k] + cfg.dt * (-state.v[k] / cfg.tau_leak + current))
        if v >= cfg.v_threshold:
            fired[k] = 1
            state.spikes[k] += 1
            amplitude = cfg.spike_current * v / cfg.v_threshold
            if amplitude >= cfg.critical_current and state.stored[k] < cap:
                state.stored[k] += 1
                state.written[k] += 1
                wrote[k] = 1
            v = cfg.v_reset
        state.v[k] = v
    state.steps += 1
    return {"fired": fired, "written": wrote, "shear": shear, "entropy": h}


def read_and_clear(state: MottSpinState) -> list[int]:
    """Read the stored counts and return the walls / bits to zero (frees the memory)."""
    out = list(state.stored)
    state.stored = [0] * PLANES
    return out


def lif_rate(drive: float, cfg: MottSpinConfig = MottSpinConfig()) -> float:
    """Uncoupled LIF firing rate per step for a constant drive (0 below rheobase).

    Exact for the discrete update: V_n = Iτ + (V_reset − Iτ)·a^n with a = 1 − Δt/τ,
    so the neuron fires every n = ⌈ln((Iτ − V_th)/(Iτ − V_reset)) / ln a⌉ steps.
    """
    i_inf = float(drive) * cfg.tau_leak
    if not math.isfinite(i_inf) or i_inf <= cfg.v_threshold:
        return 0.0
    a = 1.0 - cfg.dt / cfg.tau_leak
    n = math.ceil(math.log((i_inf - cfg.v_threshold) / (i_inf - cfg.v_reset)) / math.log(a) - 1e-12)
    return 1.0 / max(1, n)


def poisson_score(count: float, expected: float) -> float:
    """Fisher-normalised Poisson score (k − λ)/√max(λ, 1)."""
    k, lam = float(count), float(expected)
    if not (math.isfinite(k) and math.isfinite(lam)) or k < 0.0 or lam < 0.0:
        raise ValueError("count and expected must be finite and >= 0")
    return (k - lam) / math.sqrt(max(lam, 1.0))


def poissonic_adam_step(state: dict, count: float, expected: float, **adam_kwargs: Any) -> float:
    """poADAM: vanilla Adam on the Poisson score; mutates ``state``; returns the potential in [0, 1]."""
    kw = {k: adam_kwargs[k] for k in ("lr", "beta1", "beta2", "eps") if k in adam_kwargs}
    return radam_step(state, poisson_score(count, expected), **kw)


def realised_radam_step(state: dict, written: float, latent: float, expected: float,
                        **radam_kwargs: Any) -> float:
    """rADAM on what the memory holds.

    g_re is the Poisson score of the *written* spikes, g_im the latent ones, and
    the pivoted-ReLU slope is the realised fraction written/(written + latent).
    With nothing latent this is ``poissonic_adam_step`` exactly.
    """
    lam, wr, lat = float(expected), float(written), float(latent)
    if lat < 0.0 or not math.isfinite(lat):
        raise ValueError("latent must be finite and >= 0")
    g_im = lat / math.sqrt(max(lam, 1.0))
    total = wr + lat
    radam_kwargs.setdefault("pivot_alpha", wr / total if total > 0.0 else 1.0)
    return radam_step(state, poisson_score(wr, lam), g_im, **radam_kwargs)


NEUTRAL_PRESSURE: float = 0.5
"""Starting pressure of both optimisers: Adam's step is sign-normalised, so a
start at 0 could never register a shortfall."""


def run_window(state: MottSpinState, drive: Sequence[float], steps: int,
               cfg: MottSpinConfig = MottSpinConfig(), *,
               po_state: Optional[dict] = None, r_state: Optional[dict] = None,
               weights: Optional[Sequence[float]] = None) -> dict[str, Any]:
    """Run ``steps`` steps, then one poADAM and one rADAM update on the window's totals.

    Returns counts, memory contents, both pressures, Δ = p_rADAM − p_poADAM and the
    viscous stability number S(Δ); ``radam_differential`` is ready for
    ``TypedViscousCoupler.step(radam_differential=…)``.
    """
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    s0, w0 = sum(state.spikes), sum(state.written)
    entropy = 0.0
    for _ in range(steps):
        entropy += step(state, drive, cfg, weights)["entropy"]
    spikes, written = sum(state.spikes) - s0, sum(state.written) - w0
    latent = spikes - written
    expected = steps * sum(lif_rate(x, cfg) for x in drive)
    po_state = {} if po_state is None else po_state
    r_state = {} if r_state is None else r_state
    po_state.setdefault("pressure", NEUTRAL_PRESSURE)
    r_state.setdefault("pressure", NEUTRAL_PRESSURE)
    p_po = poissonic_adam_step(po_state, spikes, expected)
    p_r = realised_radam_step(r_state, written, latent, expected)
    delta = min(1.0, max(-1.0, p_r - p_po))
    from .viscous_interaction import relational_stability_number
    return {
        "memory": cfg.memory,
        "steps": steps,
        "spikes": spikes,
        "written": written,
        "latent": latent,
        "expected_poisson": expected,
        "stored": list(state.stored),
        "mean_shadow_entropy": entropy / steps,
        "p_poadam": p_po,
        "p_radam": p_r,
        "identifiable": latent == 0,
        "radam_differential": {"delta": delta, "p_radam": p_r, "p_poadam": p_po},
        "stability_number": relational_stability_number(delta),
    }


__all__ = [
    "PLANES", "MEMORIES", "MottSpinConfig", "MottSpinState",
    "shadow_potentials", "multiplanar_shear", "shadow_entropy", "memory_capacity",
    "step", "read_and_clear", "lif_rate", "poisson_score",
    "NEUTRAL_PRESSURE", "poissonic_adam_step", "realised_radam_step", "run_window",
]
