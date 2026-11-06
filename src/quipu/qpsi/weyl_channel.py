"""weyl_channel — the Riemann = Ricci ⊕ Weyl split applied to a CAT residual.

Riemann curvature decomposes into a trace part (Ricci: fixed by local
content, stays at the point) and a trace-free part (Weyl: propagates freely
between points).  On the CAT_6 vector the same split is available with the
static weight prior ŵ as the trace direction:

    δ = ⟨δ, ŵ⟩ ŵ  +  (I − ŵŵ*) δ
        ricci          weyl

* ricci  — the part of the move that is "more of the expected mix".  Bound to
           the node: it updates the node's level and realises no edge.
* weyl   — the trace-free remainder.  Propagates: it is the only part that
           can become an edge.

The projection leakage (§9.2)
-----------------------------
A single-axis move δ e_k has weyl part  δ (e_k − ŵ_k ŵ):  +δ(1 − ŵ_k²) on k
and −δ ŵ_k ŵ_j on every other axis j.  That is the tidal signature of the
Weyl tensor — stretch on one axis, volume-preserving compression on the
others — and it is NOT noise to gate away.  But it means a constant floor η
cannot make per-axis realisation correct: the off-axis components scale
with |δ| (0.2725·|δ| between vision and touch), so for any η a large enough
single-axis event realises six edges.

The fix is structural.  Realise events, not axes.  A single-axis event has a
known signature s_k = e_k − ŵ_k ŵ; decompose the weyl part onto signatures
(closed form, sparsest exact solution — see ``decompose``).
Each explained signature is one event → one candidate edge; the compressions
it predicts on the other axes are its signature, not separate events.  η is
then a floor on *unexplained remainder*, which is exactly zero for any pure
single-axis event of any size — a constant works, and the requirement
"η above leakage" becomes "η above the remainder after signature removal",
which the tests prove.

Stdlib only.  Additive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .cat_residual import SENSES, SENSE_WEIGHTS, _w_hat


@dataclass
class WeylSplit:
    ricci: float                      # ⟨δ, ŵ⟩ — scalar along the prior
    ricci_vec: dict[str, float]       # ricci · ŵ
    weyl: dict[str, float]            # (I − ŵŵ*) δ, trace-free


@dataclass
class WeylEvent:
    axis: str
    amplitude: float                  # signed δ recovered for that axis
    explained: dict[str, float]       # the signature it accounts for


@dataclass
class WeylDecomposition:
    events: list[WeylEvent]
    remainder: dict[str, float]       # unexplained weyl part
    remainder_norm: float
    ricci: float


def split(delta: Mapping[str, float], weights: Mapping[str, float] = SENSE_WEIGHTS) -> WeylSplit:
    w = _w_hat(weights)
    d = [float(delta.get(s, 0.0)) for s in SENSES]
    proj = sum(a * b for a, b in zip(d, w))
    ricci_vec = {s: proj * wi for s, wi in zip(SENSES, w)}
    weyl = {s: di - proj * wi for s, di, wi in zip(SENSES, d, w)}
    return WeylSplit(ricci=proj, ricci_vec=ricci_vec, weyl=weyl)


def signature(axis: str, weights: Mapping[str, float] = SENSE_WEIGHTS) -> dict[str, float]:
    """Weyl part of a unit move on ``axis``: e_k − ŵ_k ŵ."""
    w = dict(zip(SENSES, _w_hat(weights)))
    return {s: (1.0 if s == axis else 0.0) - w[axis] * w[s] for s in SENSES}


def max_leakage_ratio(weights: Mapping[str, float] = SENSE_WEIGHTS) -> tuple[float, str, str]:
    """max_{k≠j} ŵ_k ŵ_j — off-axis leakage per unit single-axis move."""
    w = dict(zip(SENSES, _w_hat(weights)))
    best = (0.0, "", "")
    for k in SENSES:
        for j in SENSES:
            if k != j and w[k] * w[j] > best[0]:
                best = (w[k] * w[j], k, j)
    return best


def _norm(v: Mapping[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values()))


def decompose(delta: Mapping[str, float], *, eta: float, max_events: int = 6,
              weights: Mapping[str, float] = SENSE_WEIGHTS) -> WeylDecomposition:
    """Sparsest exact signature decomposition of the weyl part.

    Σ_k a_k s_k = (I − ŵŵ*) a, so every coefficient vector a with
    (I − ŵŵ*) a = weyl is of the form a = weyl + c·ŵ (the six signatures span
    the 5-D trace-free subspace; ŵ is the null direction).  The sparsest a is
    found in closed form: try c = 0 and c = −weyl_j / ŵ_j for each j, keep
    the c with the most coefficients below eta (ties → smaller |c|).  Each
    surviving coefficient is one event.  Exact; no greedy pursuit.
    """
    sp = split(delta, weights)
    w = dict(zip(SENSES, _w_hat(weights)))
    r = sp.weyl
    candidates = [0.0] + [-r[j] / w[j] for j in SENSES if w[j] != 0.0]
    best_a, best_key = None, None
    for c in candidates:
        a = {s: r[s] + c * w[s] for s in SENSES}
        zeros = sum(1 for v in a.values() if abs(v) < eta)
        key = (-zeros, abs(c))
        if best_key is None or key < best_key:
            best_a, best_key = a, key
    sigs = {k: signature(k, weights) for k in SENSES}
    events: list[WeylEvent] = []
    rem = dict(r)
    for k in sorted(SENSES, key=lambda s: -abs(best_a[s])):
        if abs(best_a[k]) < eta or len(events) >= max_events:
            continue
        explained = {s: best_a[k] * sigs[k][s] for s in SENSES}
        rem = {s: rem[s] - explained[s] for s in SENSES}
        events.append(WeylEvent(axis=k, amplitude=best_a[k], explained=explained))
    return WeylDecomposition(events=events, remainder=rem, remainder_norm=_norm(rem), ricci=sp.ricci)


__all__ = ["WeylSplit", "WeylEvent", "WeylDecomposition", "split", "signature",
           "max_leakage_ratio", "decompose"]
