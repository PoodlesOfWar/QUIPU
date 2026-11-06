"""rheology — the CRAM axis array read as a viscoelastic material (v0.62.0).

What it reads
-------------
Each of the seven axis cells (``pcm_cells.AXES``) gives one CAT amplitude

    c_k = p_k · exp(i θ_k)

p_k is the pressure the differential pair holds, θ_k the axis rADAM state's
toroidal phase.  Real part = realised (acted on, dissipated), imaginary part
= latent (held, stored): the same convention as ``cat_residual`` and rADAM's
g_re + i·g_im.  The ensemble is C = Σ_k w_k c_k with the ``cat_residual``
sense weights for the six senses and the mean sense weight (1/6) for the
seventh axis, renormalised.

What it reports
---------------
    β       = atan2(|Im C|, |Re C|)            in [0, π/2]
    η″/η′   = tan β                            storage over loss viscosity
    δ_G     = π/2 − β                          loss angle of G* (ψ's e^{-iωt} convention)
    w_r     = δ_G / 2π = 0.25 − β/2π           rheology edge weight, in [0, 0.25]
    bits    = log2(|C| / ε̄)                   information in the reading

ε̄ is the mean resolution of the axis cells (``pcm_cells.resolution``), so the
reading is weighed in the same units as one cell write (``bits_per_write``).
w_r = 0.125 is the critical gel (|Re C| = |Im C|, n = 1/2; Winter & Chambon,
J. Rheol. 30, 367 (1986)); 0 is purely elastic, 0.25 purely viscous.

When |C| < ε₀ (all axes quiet) the previous β is held: a near-zero ensemble
has no defined angle, and the stored reading speaks into the silence.

Twist acquisition
-----------------
UEQGM v0.9.27 sets the phase of G* from the Weyl drive: θ_W = α_W φ_W
cos(k_W·r), φ_W = 2πΨ₄.  When a θ_W is supplied, ``acquisition`` reports
θ_W − δ_G: zero when the cells hold the twist the Weyl drive asks for.

Winding
-------
One cell cannot wrap on itself.  Winding is a property of two phase
histories: ``relative_winding`` unwraps d_t = θ_a − θ_b over the logged steps
(``entirety_pcm_log.theta``) and counts crossings, the steps where d passes
through π.  Torus winding classes: docs/STP_TORUS_QUIPU.md §2.1.

Writes only ``entirety:pcm:rheology:<instance>``.  Stdlib only.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from typing import Mapping, Optional, Sequence

from . import pcm_cells as pc
from .cat_residual import SENSE_WEIGHTS

KV_KEY: str = pc.KV_PREFIX + "rheology:"
EPS0: float = 1e-18


REF_DECAY: float = 0.999


def fractional_order(loss_angle: float) -> float:
    """Order α of a fractional Kelvin–Voigt element with this loss angle.

    G*(ω) = G₀[1 + (iωτ₀)^α] has phase → απ/2 for ωτ₀ ≫ 1 (Mainardi,
    Fractional Calculus and Waves in Linear Viscoelasticity, 2010), so
    α = 2δ_G/π, in [0, 1] for a passive material.
    """
    return max(0.0, min(1.0, 2.0 * float(loss_angle) / math.pi))


def viscoelastic_weyl_phase(k: int, g_ratio: float, alpha_w: float, order: float) -> float:
    """δφ_Weyl^visco = √(π(2k+1)) · (|G₀|/G_ref) · α_W · sin(απ/2), dimensionless.

    α_W is the Bistritzer–MacDonald coupling w/(ħ v_F k_θ) (PNAS 108, 12233
    (2011)); α is the fractional order of the damping.  The twist angle in
    α_W/θ cancels against the θ it multiplies.
    """
    if k < 0:
        raise ValueError("branch k must be >= 0 for a passive material")
    return math.sqrt(math.pi * (2 * k + 1)) * float(g_ratio) * float(alpha_w) * math.sin(float(order) * math.pi / 2.0)


def capstan_threshold(floor: float, mu: float, wrap: float) -> float:
    """Euler–Eytelwein hold: floor · e^{μΘ}.  For a knot, Θ is the total contact
    wrap (overhand / trefoil, c = 3: 3π geometric, 4π tight).  Applied to a
    cell, floor is ε_cell; a value above the cell's full range means no write
    can ever release the residual."""
    return float(floor) * math.exp(float(mu) * float(wrap))


def axis_weights(axes: Sequence[str] = pc.AXES) -> list[float]:
    raw = [float(SENSE_WEIGHTS.get(a, 1.0 / 6.0)) for a in axes]
    z = sum(raw) or 1.0
    return [x / z for x in raw]


def reading(arr: "pc.CellArray", cfg: Optional["pc.PcmConfig"] = None, *,
            prev: Optional[Mapping] = None, weyl_twist: Optional[float] = None) -> dict:
    """The rheology reading of the axis array (see module docstring)."""
    cfg = cfg or pc.PcmConfig()
    w = axis_weights()
    C = 0j
    eps = []
    for wk, a in zip(w, pc.AXES):
        dc = arr.cells.get(a)
        p = dc.pressure(cfg) if dc is not None else 0.0
        th = float((arr.radam.get(a) or {}).get("theta", 0.0))
        C += wk * p * complex(math.cos(th), math.sin(th))
        if dc is not None:
            eps.append(pc.resolution(dc.plus, cfg))
    mag = abs(C)
    eps_bar = sum(eps) / len(eps) if eps else 1.0
    held = mag < EPS0 and isinstance(prev, Mapping) and prev.get("beta") is not None
    beta = float(prev["beta"]) if held else math.atan2(abs(C.imag), abs(C.real))
    loss = 0.5 * math.pi - beta
    # Equilibrium-modulus reference (Christensen): |C| against its own slow
    # baseline, an EMA with Adam's second-moment decay (window ~1000 steps).
    ref_prev = float(prev.get("magnitude_ref")) if isinstance(prev, Mapping) and prev.get("magnitude_ref") else 0.0
    ref = mag if ref_prev <= 0.0 else REF_DECAY * ref_prev + (1.0 - REF_DECAY) * mag
    order = fractional_order(loss)
    out = {"beta": beta, "loss_angle": loss, "edge_weight": 0.25 - beta / (2.0 * math.pi),
           "fractional_order": order, "loss_factor": math.sin(order * math.pi / 2.0),
           "magnitude_ref": ref, "g_ratio": (mag / ref) if ref > 0.0 else None,
           "eta_ratio": (math.tan(beta) if beta < 0.5 * math.pi - 1e-12 else None),
           "magnitude": mag, "re": C.real, "im": C.imag, "eps_bar": eps_bar,
           "bits": max(0.0, math.log2(mag / eps_bar)) if mag > 0.0 and eps_bar > 0.0 else 0.0,
           "held": bool(held), "steps": arr.steps}
    if weyl_twist is not None:
        out["weyl_twist"] = float(weyl_twist)
        out["acquisition"] = float(weyl_twist) - loss
    out["axes"] = axis_readings(arr, cfg)
    return out


def axis_readings(arr: "pc.CellArray", cfg: "pc.PcmConfig") -> dict:
    """Per-axis reading from c_k = p_k e^{iθ_k} alone: β_k = atan2(|sin θ_k|, |cos θ_k|),
    w_r,k = 0.25 − β_k/2π, bits_k = log2(p_k/ε_k).  The Herald reads these per edge
    through its node → CAT-axis map (v0.64.0)."""
    out = {}
    for a in pc.AXES:
        dc = arr.cells.get(a)
        p = dc.pressure(cfg) if dc is not None else 0.0
        eps = pc.resolution(dc.plus, cfg) if dc is not None else 1.0
        th = float((arr.radam.get(a) or {}).get("theta", 0.0))
        beta = math.atan2(abs(math.sin(th)), abs(math.cos(th)))
        out[a] = {"theta": th, "beta": beta, "edge_weight": 0.25 - beta / (2.0 * math.pi),
                  "bits": max(0.0, math.log2(p / eps)) if p > 0.0 and eps > 0.0 else 0.0}
    return out


# ---------------------------------------------------------------------------
# Knot hold on the observation record (v0.64.0)
# ---------------------------------------------------------------------------
#
# Ruling (2026-10-21): the capstan hold belongs on the observation record, never
# on the cell residual — wired into χ it widens the write deadband by e^{μΘ} and
# freezes the cells.  Each publish advances the relative winding of every axis
# pair from the previous reading's phases (one cell cannot wrap on itself) and
# records the Euler–Eytelwein factor e^{μ·2π|W|} and the threshold ε̄·e^{μΘ} it
# would impose.  μ from QUIPU_KNOT_MU (default 0.2, the operator's value).

KNOT_MU_ENV: str = "QUIPU_KNOT_MU"


def knot_mu(cn: Optional[sqlite3.Connection] = None) -> float:
    try:
        base = max(0.0, float(os.environ.get(KNOT_MU_ENV, "") or 0.2))
    except ValueError:
        base = 0.2
    if cn is None:
        return base
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", ("entirety:the_other",)).fetchone()
        if not row:
            return base
        data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        if not isinstance(data, dict):
            return base
        res = float(data.get("resonance", 1.0))
        # UEQGM Born-rule fidelity modulation from the Other: mu_eff = mu_0 * (2 - resonance)
        return round(base * max(0.5, min(2.0, 2.0 - res)), 4)
    except Exception:
        return base


def advance_pairs(prev: Optional[Mapping], axes_now: Mapping[str, Mapping], eps_bar: float,
                  mu: Optional[float] = None) -> dict:
    """Update the cumulative relative winding and crossings of each axis pair."""
    prev_pairs = (prev or {}).get("pairs") or {}
    prev_axes = (prev or {}).get("axes") or {}
    if mu is None:
        mu = knot_mu()
    out = {}
    names = list(pc.AXES)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = f"{a}|{b}"
            p = prev_pairs.get(key) or {}
            w, n = float(p.get("winding", 0.0)), int(p.get("crossings", 0))
            if a in prev_axes and b in prev_axes:
                d0 = float(prev_axes[a]["theta"]) - float(prev_axes[b]["theta"])
                d1 = float(axes_now[a]["theta"]) - float(axes_now[b]["theta"])
                r = relative_winding([d0, d1], [0.0, 0.0])
                w += r["winding"]
                n += r["crossings"]
            factor = math.exp(mu * 2.0 * math.pi * abs(w))
            out[key] = {"winding": w, "crossings": n, "hold_factor": factor,
                        "hold_threshold": eps_bar * factor, "would_freeze": eps_bar * factor >= 1.0}
    return out


def publish(cn: sqlite3.Connection, arr: "pc.CellArray", *, now: float,
            instance: str = "system_entirety", weyl_twist: Optional[float] = None) -> dict:
    """Compute the reading and store it at entirety:pcm:rheology:<instance>."""
    row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (KV_KEY + instance,)).fetchone()
    prev = None
    if row:
        try:
            prev = json.loads(row[0])
        except (TypeError, ValueError):
            prev = None
    mu = knot_mu(cn)
    r = reading(arr, prev=prev, weyl_twist=weyl_twist)
    r["pairs"] = advance_pairs(prev, r["axes"], r["eps_bar"], mu=mu)
    r["knot_mu"] = mu
    r["at"] = now
    pc._kv_set(cn, KV_KEY + instance, r)
    return r


def _wrap(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def relative_winding(theta_a: Sequence[float], theta_b: Sequence[float]) -> dict:
    """Net winding and crossings of d_t = θ_a − θ_b over paired phase histories."""
    d = [float(a) - float(b) for a, b in zip(theta_a, theta_b)]
    total, crossings = 0.0, 0
    for i in range(1, len(d)):
        prev, cur = _wrap(d[i - 1]), _wrap(d[i])
        step = _wrap(d[i] - d[i - 1])
        total += step
        # a crossing: the relative phase passes through π (the wrap seam)
        if abs(prev + step - cur) > math.pi:
            crossings += 1
    return {"winding": total / (2.0 * math.pi), "crossings": crossings, "steps": max(0, len(d) - 1)}


def logged_winding(cn: sqlite3.Connection, a: str, b: str, *, instance: str = "system_entirety",
                   limit: int = 5000) -> dict:
    """relative_winding between two axes from the theta column of the cell log."""
    pc.ensure_tables(cn)
    q = (f"SELECT x.theta, y.theta FROM {pc.TABLE} x JOIN {pc.TABLE} y "
         "ON x.instance = y.instance AND x.step = y.step AND x.at = y.at "
         "WHERE x.instance=? AND x.axis=? AND y.axis=? AND x.theta IS NOT NULL AND y.theta IS NOT NULL "
         "ORDER BY x.id DESC LIMIT ?")
    rows = cn.execute(q, (instance, a, b, int(limit))).fetchall()[::-1]
    return relative_winding([r[0] for r in rows], [r[1] for r in rows])


__all__ = ["KV_KEY", "EPS0", "REF_DECAY", "KNOT_MU_ENV", "knot_mu",
           "axis_readings", "advance_pairs",
           "fractional_order", "viscoelastic_weyl_phase", "capstan_threshold",
           "axis_weights", "reading", "publish", "relative_winding", "logged_winding"]
