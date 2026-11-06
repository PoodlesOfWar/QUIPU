"""cat_residual — the residual operator and the interstitial rectifier written
in CAT amplitudes.

Vocabulary (creator language, 2026-09-09)
------------------------------------------
    lossy channel          filters which residuals are even eligible
    interstitial rectifier decides which eligible residuals become interactions
    node                   temporary carrier of content; may be consumed or stretched away
    edge / vortex          a created interaction that influences later residuals

CAT state (extended)
--------------------
QUIPU's CAT_6 is six real coherences a_k in [0, 1], one per sense
(vision, touch, smell, body, brain, perception).  Here each sense carries a
complex amplitude

    c_k = a_k * exp(i * theta_k)

with theta_k the sense's phase on the torus (rADAM keeps ``theta`` per signal
kind; temporal_spatiality keeps the Weyl coordinate as a circular mean).
Real part = realised (rADAM ``g_re``), imaginary part = latent (``g_im``).
The 7th axis (observer) is the residual; the 8th (mesh Omega) is the residual
of residuals across nodes.

Residual operator  R
--------------------
Two projections are removed, not one:

    r_t = (I - w w*) ( c_t - D c_{t-1} )

* (I - w w*)   removes the part predicted by the static weight prior w
               (this is the existing ``observer_tangent`` construction).
* D c_{t-1}    removes the part predicted by the lossy channel from the
               previous state: per-axis decay exp(-nu dt) plus optional
               mixing across neighbouring senses (the nu grad^2 term).

The existing ``system_entirety.observer_tangent`` is the special case
c_{t-1} = 0, real c.  That is why it read 0.0088 on 76 byte-identical frames:
it measures level, not change.  ``residual`` on the same frames is exactly 0.

Channel gate: |r_k| < eta  =>  r_k := 0.   eta is the resolution floor
(Kolmogorov length when nu and the injection rate are measured; float
resolution as the fallback the Planck gate uses today).

Rectifier  Gamma
----------------
Per axis, after the channel:

    realised_k = pivoted_relu(|r_k|, pivot = p*, alpha)
    p*         = C_wrong / (C_wrong + C_missed)          (unwind-cost bar)

alpha = 1 is the identity (ranking regime); alpha -> 0 is a hard gate
(gating regime).  ``pivoted_relu`` reproduces radam_optimizer.pivoted_relu.
A Lipschitz clamp bounds |realised_k - realised_j| for neighbouring senses;
this is the invariant the forced blow-up result shows is otherwise
unprotected (finite energy, unbounded gradient).

Realisation
-----------
A realised axis names a candidate edge from the node to the entity type that
axis feeds (``AXIS_TARGET``).  Its weight is the realised magnitude, its phase
is arg(r_k), its cost class is the action's.  Edges below the pivot stay in
band (``:leave_in_band``).

Holons
------
A holon is a realised direction that recurs across independent sessions
(one vote per session, the unit the recurrence audit validated).  Holons are
the forge candidates: ``tool_forge`` gates today on a constant 52 Hz and on
corpus-cluster statistics; a holon lets the CAT state drive the forge
directly, with the bar derived from the cost of ``clean_generated_tools``.

No QUIPU imports; stdlib only.  Additive; mesh_slm.py untouched.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

SENSES: tuple[str, ...] = ("vision", "touch", "smell", "body", "brain", "perception")

# Same numbers as temporal_spatiality._SENSE_WEIGHTS.
SENSE_WEIGHTS: dict[str, float] = {
    "vision": 0.22, "touch": 0.22, "smell": 0.18,
    "body": 0.12, "brain": 0.12, "perception": 0.14,
}

# Where a realised residual on each axis points.  Defaults are the entity
# types the overlay already hardens (system_entirety._project_learning_overlay)
# plus the two place types from geospatial_relation.
AXIS_TARGET: dict[str, tuple[str, str]] = {
    "vision":     ("MeshLearningWindow",      "OBSERVES"),
    "touch":      ("AssetResource",           "HARDENS_ASSET_RESOURCE"),
    "smell":      ("LearningKindSummary",     "SUMMARIZES_LEARNING_KIND"),
    "body":       ("Endpoint",                "HARDENS_ENDPOINT"),
    "brain":      ("SpatialMaterialProcessor","HARDENS_MATERIAL_PROCESSOR"),
    "perception": ("UEQGMRuntimeState",       "APPLIES_UEQGM_RUNTIME"),
}

# Neighbour ring for the diffusion term (senses as they are ordered in CAT_6).
_RING = {s: (SENSES[i - 1], SENSES[(i + 1) % 6]) for i, s in enumerate(SENSES)}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class CATState:
    """Complex CAT amplitudes, one per sense, at time t (seconds)."""
    amp: dict[str, complex]
    t: float = 0.0

    @classmethod
    def from_axes(cls, axes: Mapping[str, float], phases: Mapping[str, float] | None = None,
                  t: float = 0.0) -> "CATState":
        phases = phases or {}
        return cls({s: float(axes.get(s, 0.0)) * cmath.exp(1j * float(phases.get(s, 0.0)))
                    for s in SENSES}, t)

    def vector(self) -> list[complex]:
        return [self.amp.get(s, 0j) for s in SENSES]


# ---------------------------------------------------------------------------
# Prior projection  (I - w w*)
# ---------------------------------------------------------------------------

def _w_hat(weights: Mapping[str, float] = SENSE_WEIGHTS) -> list[float]:
    w = [float(weights.get(s, 1.0 / 6)) for s in SENSES]
    n = math.sqrt(sum(x * x for x in w)) or 1.0
    return [x / n for x in w]


def prior_residual(c: Sequence[complex], weights: Mapping[str, float] = SENSE_WEIGHTS) -> list[complex]:
    """(I - w w*) c — the component orthogonal to the static weight prior."""
    w = _w_hat(weights)
    proj = sum(ci * wi for ci, wi in zip(c, w))
    return [ci - proj * wi for ci, wi in zip(c, w)]


# ---------------------------------------------------------------------------
# Lossy channel  D
# ---------------------------------------------------------------------------

def channel(prev: CATState | None, dt: float, nu: float, kappa: float = 0.0) -> list[complex]:
    """What the channel predicts the state to be after dt, from prev.

    Per-axis decay exp(-nu dt); ``kappa`` in [0, 0.5] mixes each axis with
    its two ring neighbours (the discrete nu grad^2).  prev None -> zeros,
    which reduces the residual to the existing observer_tangent.
    """
    if prev is None:
        return [0j] * 6
    decay = math.exp(-max(0.0, nu) * max(0.0, dt))
    base = {s: prev.amp.get(s, 0j) for s in SENSES}
    out = []
    for s in SENSES:
        if kappa > 0.0:
            l, r = _RING[s]
            mixed = (1.0 - 2.0 * kappa) * base[s] + kappa * (base[l] + base[r])
        else:
            mixed = base[s]
        out.append(decay * mixed)
    return out


# ---------------------------------------------------------------------------
# Residual operator  R
# ---------------------------------------------------------------------------

def residual(cur: CATState, prev: CATState | None = None, *, nu: float = 0.0,
             kappa: float = 0.0, eta: float = 0.0,
             weights: Mapping[str, float] = SENSE_WEIGHTS) -> dict[str, complex]:
    """r_t = (I - w w*)(c_t - D c_{t-1}), gated per axis at eta."""
    dt = (cur.t - prev.t) if prev is not None else 0.0
    predicted = channel(prev, dt, nu, kappa)
    delta = [a - b for a, b in zip(cur.vector(), predicted)]
    r = prior_residual(delta, weights)
    return {s: (rk if abs(rk) >= eta else 0j) for s, rk in zip(SENSES, r)}


def observer(r: Mapping[str, complex]) -> float:
    """|r| / sqrt(6), clamped — same normalisation as observer_tangent."""
    mag = math.sqrt(sum(abs(v) ** 2 for v in r.values()))
    return max(0.0, min(1.0, mag / math.sqrt(6.0)))


def observer_legacy(axes: Mapping[str, float], weights: Mapping[str, float] = SENSE_WEIGHTS) -> float:
    """Byte-for-byte the existing system_entirety.observer_tangent."""
    a = [float(axes.get(s, 0.0)) for s in SENSES]
    w = _w_hat(weights)
    proj = sum(ai * wi for ai, wi in zip(a, w))
    n = [ai - proj * wi for ai, wi in zip(a, w)]
    return max(0.0, min(1.0, math.sqrt(sum(x * x for x in n)) / math.sqrt(6.0)))


def kolmogorov_eta(nu: float, epsilon: float) -> float:
    """eta = (nu^3 / epsilon)^(1/4); epsilon = residual-energy injection rate."""
    if nu <= 0.0 or epsilon <= 0.0:
        return 0.0
    return (nu ** 3 / epsilon) ** 0.25


# ---------------------------------------------------------------------------
# Interstitial rectifier  Gamma
# ---------------------------------------------------------------------------

def pstar(cost_wrong: float, cost_missed: float) -> float:
    """Unwind-cost bar: p* = C_wrong / (C_wrong + C_missed)."""
    cw, cm = max(0.0, cost_wrong), max(0.0, cost_missed)
    return cw / (cw + cm) if (cw + cm) > 0 else 1.0


def pivoted_relu(x: float, pivot: float, alpha: float = 1.0) -> float:
    """Same function as radam_optimizer.pivoted_relu."""
    d = x - pivot
    return x if d >= 0.0 else pivot + alpha * d


@dataclass
class EdgeProposal:
    axis: str
    dst_type: str
    rel: str
    weight: float          # realised magnitude after rectifier and clamp
    phase: float           # arg(r_k)
    raw: float             # |r_k| before rectifier
    realised: bool         # above pivot -> :emit_tangential; else :leave_in_band
    cost_class: str
    unclamped: float = 0.0  # rectified magnitude BEFORE the Lipschitz pass — what gate 6 tests


def rectify(r: Mapping[str, complex], *, pivot: float, alpha: float = 1.0,
            lipschitz: float | None = None) -> dict[str, float]:
    """Per-axis realised magnitude.  Lipschitz clamps neighbour differences."""
    out = {s: max(0.0, pivoted_relu(abs(r.get(s, 0j)), pivot, alpha)) for s in SENSES}
    if lipschitz is not None and lipschitz >= 0.0:
        # Iterate until every ring neighbour pair is within the bound.
        for _ in range(12):
            changed = False
            for s in SENSES:
                for nb in _RING[s]:
                    if out[s] - out[nb] > lipschitz:
                        out[s] = out[nb] + lipschitz
                        changed = True
            if not changed:
                break
    return out


def realize(r: Mapping[str, complex], *, pivot: float, alpha: float = 1.0,
            lipschitz: float | None = None, cost_class: str = "default",
            targets: Mapping[str, tuple[str, str]] = AXIS_TARGET) -> list[EdgeProposal]:
    """Residual -> candidate edges.  Only axes with nonzero residual appear.

    ``weight`` is the clamped magnitude that would be written; ``unclamped`` is
    the rectified magnitude before the Lipschitz pass.  Gate 6 must test
    ``unclamped``: the clamp makes every ``weight`` satisfy the bound by
    construction, so a breach is only visible before it.
    """
    free = rectify(r, pivot=pivot, alpha=alpha)
    mags = rectify(r, pivot=pivot, alpha=alpha, lipschitz=lipschitz)
    props = []
    for s in SENSES:
        rk = r.get(s, 0j)
        if rk == 0:
            continue
        dst, rel = targets[s]
        props.append(EdgeProposal(
            axis=s, dst_type=dst, rel=rel, weight=mags[s], phase=cmath.phase(rk),
            raw=abs(rk), realised=abs(rk) >= pivot, cost_class=cost_class,
            unclamped=free[s],
        ))
    return props


# ---------------------------------------------------------------------------
# Holons — realised directions that recur across sessions
# ---------------------------------------------------------------------------

@dataclass
class Holon:
    direction: dict[str, float]   # unit vector over senses
    support: int                  # sessions in which it recurred
    sessions: list[int] = field(default_factory=list)


def _unit(v: Mapping[str, float]) -> dict[str, float]:
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {s: v.get(s, 0.0) / n for s in SENSES}


def _cos(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return sum(a.get(s, 0.0) * b.get(s, 0.0) for s in SENSES)


def holon_candidates(session_directions: Sequence[Mapping[str, float]], *,
                     min_sessions: int = 3, cos_tau: float = 0.9) -> list[Holon]:
    """One realised direction per session in; directions that recur in at
    least ``min_sessions`` sessions (cosine >= cos_tau) out.  A single loud
    session cannot produce a holon regardless of magnitude."""
    units = [_unit(d) for d in session_directions if any(d.values())]
    holons: list[Holon] = []
    used: set[int] = set()
    for i, u in enumerate(units):
        if i in used:
            continue
        members = [j for j, v in enumerate(units) if j not in used and _cos(u, v) >= cos_tau]
        if len(members) >= min_sessions:
            mean = {s: sum(units[j][s] for j in members) / len(members) for s in SENSES}
            holons.append(Holon(direction=_unit(mean), support=len(members), sessions=members))
            used.update(members)
    return holons


def forge_bar(cost_wrong_tool: float, cost_missed_tool: float) -> float:
    """p* for forging: wrong = a tool written, registered, and later cleaned;
    missed = nothing (the forge re-scores on the next round)."""
    return pstar(cost_wrong_tool, cost_missed_tool)


__all__ = [
    "SENSES", "SENSE_WEIGHTS", "AXIS_TARGET", "CATState",
    "prior_residual", "channel", "residual", "observer", "observer_legacy",
    "kolmogorov_eta", "pstar", "pivoted_relu", "rectify", "realize",
    "EdgeProposal", "Holon", "holon_candidates", "forge_bar",
]


# ---------------------------------------------------------------------------
# Summary character 翈, grounded on sqrt(-1)
# ---------------------------------------------------------------------------
# i = sqrt(-1) is the latent axis of this module: Im c_k is rADAM's g_im
# (unrealised), the part the rectifier holds in band.  Multiplying by i is a
# quarter turn: realised -> latent.  Two quarter turns give i^2 = -1, the
# deepen parity of bit_flip_parity.  So bit_state in {+1, -1} = {i^0, i^2},
# and 翈 = i^1 is the state between them: the residual that is held, not yet
# written.  A section's 翈-line is its latent reading — what stays in band.

SUMMARY_CHARACTER: str = "翈"
SUMMARY_GROUND: complex = 1j


def latent(r: Mapping[str, complex]) -> dict[str, float]:
    """翈(r): the imaginary (held, unrealised) part of a residual."""
    return {s: r.get(s, 0j).imag for s in SENSES}


def realised_part(r: Mapping[str, complex]) -> dict[str, float]:
    """Re r: the part the rectifier can write."""
    return {s: r.get(s, 0j).real for s in SENSES}


def quarter_turn(r: Mapping[str, complex]) -> dict[str, complex]:
    """Multiply by i: realised -> latent, latent -> -realised."""
    return {s: SUMMARY_GROUND * r.get(s, 0j) for s in SENSES}


def parity_from_turns(turns: int) -> int:
    """i^turns projected to the bit: 0,4,... -> +1 (broaden); 2,6,... -> -1 (deepen).
    Odd turns are 翈 states: held, no bit."""
    v = SUMMARY_GROUND ** (turns % 4)
    return 0 if abs(v.real) < 1e-12 else (1 if v.real > 0 else -1)


__all__ += ["SUMMARY_CHARACTER", "SUMMARY_GROUND", "latent", "realised_part",
            "quarter_turn", "parity_from_turns"]
