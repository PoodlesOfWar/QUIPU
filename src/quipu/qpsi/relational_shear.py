"""relational_shear — the shearing λ measured in each axis's relational time, feeding the shadow.

Definitions (each term is something QUIPU already computes; nothing here is a new physical claim)
---------------------------------------------------------------------------------------------

**Relational time** τ_k.  An axis's own clock advances only when its cell actually changes: one tick per
resolvable quantum written, dτ_k = |realised_k| / eps_k (``pcm_cells.ef_step``: ``realised`` and ``eps``).
A step in which the change stays deferred in χ is not time for that axis.  This is the Planck gate's rule
(persistence from displacement, not schedule) applied to time.  Over a window of W steps the axis's
**relational rate** is ρ_k = Δτ_k / mean_j Δτ_j: above 1 the axis lives faster than the ensemble, below 1
slower, 0 frozen.  The **differentiation in relational time** is the spread of log ρ across axes.

**The shearing λ in relational time.**  ``pcm_cells.shear_coupling`` measures each axis's step velocity u_k
per optimizer step.  On the ensemble's clock an axis moves v_k = u_k · ρ_k: a large step on an axis whose
clock did not tick moved nothing.  With the Poisson weights the deepen step already uses (mean σ_u²):

    λ_rel      = Σ_k w_k (v_k − v̄_w)²            (weighted relational shear variance)
    λ_rel_STP  = λ_rel / (v̄_w² + λ_rel)          (its share of step energy, in [0, 1])

**The highest outlier relational to the weighted clamp.**  The CAT rectifier's Lipschitz clamp removes
e_k = unclamped_k − weight_k from each axis (``cat_residual.realize``); gate 6 holds a breach of the bound
instead of writing it.  The clamp's withheld excess, weighted by the axis's Poisson weight and its relational
rate, is s_k = w_k · ρ_k · e_k.  The outlier is k* = argmax s_k when its robust z-score across the axes
(Iglewicz & Hoaglin's modified z, 0.6745 (s − median)/MAD) is at least Z.  Its energy is E* = s_k*.

**Feeding the shadow.**  ``shadow_tokens`` draws each scenario's plane directions with QUIPU's Langevin
noise σ_L.  Fed, the draws use

    σ_eff = σ_L · (1 + g · (λ_rel_STP + E*)),    capped at σ_L · max(1, 0.9 · SNR_min)

so cross-axis shear and the clamp's withheld excess widen the scenarios (more tokens cross between voiced
and pruned: more shadow tokens) while the cap keeps the least resolvable plane at an SNR of about 1 or more.
The gain g is a door: the operator grants [g_min, g_max]; inside it a continuity controller moves g.

**Continuity.**  The shadow's emission is continuous when every voiced plane stays resolvable (SNR ≥ 1) and
its voice still crosses: the potential's crossing rate is at least one token changing per plane per run,
1 / (n · K̄) for n scenarios of K̄ tokens.  Below that the emission is frozen (the same tokens in every
scenario: nothing for the shadow to price); under SNR 1 it dissolves (the mesh cannot tell the movements
apart).  Both bars come from the run itself, not from a tuned band.  After each fed run: frozen → g up;
dissolving → g down; otherwise g is held.

**Paradigmatic events.**  A shadow token (held back or pulled in) that recurs in at least two sessions while
the same axis is the clamp outlier is a two-regime record driven by one shear source: it switches between
voiced and pruned with that axis's withheld excess.  It is returned as a *proposal*
(``brain_kv entirety:shear:proposal`` when the caller persists it).  Proposals are not edges: they reach the
torus only through the six gates, two signers and the grant, like every other realisation.  Shadow tokens
are existing mesh tokens, so no shadow token takes a torus cell (ruling of 2026-09-28).

What this does not do: it never writes the mesh, the shared embedding or the axis cells; it never feeds the
clamp's excess into a weight (that would be crossing gate 6); and it does not move σ_L itself.

翈 — the vane measures the wind by the time each feather took to turn.
"""
from __future__ import annotations

import dataclasses
import math
from collections import deque
from statistics import median
from typing import Any, Iterable, Mapping, Optional, Sequence

AXES: tuple[str, ...] = ("vision", "touch", "smell", "body", "brain", "perception", "entirety")
KV_PROPOSAL = "entirety:shear:proposal"
KV_STATE = "entirety:shear:state"


@dataclasses.dataclass
class ShearConfig:
    window: int = 16            # steps of relational time per rate (the emergence detector's window)
    z_outlier: float = 3.5      # modified z-score for the clamp outlier (Iglewicz & Hoaglin 1993)
    g_doors: tuple[float, float] = (0.0, 4.0)   # the operator's grant for the feeding gain
    g_start: float = 1.0
    snr_floor: float = 1.0
    snr_safety: float = 0.9
    min_sessions: int = 2


# ---------------------------------------------------------------------------
# Relational time
# ---------------------------------------------------------------------------

class RelationalClock:
    """Each axis's proper time, in resolvable quanta written."""

    def __init__(self, axes: Sequence[str] = AXES, window: int = 16) -> None:
        self.axes = tuple(axes)
        self.tau = {a: 0.0 for a in self.axes}
        self._recent = {a: deque(maxlen=window) for a in self.axes}

    def tick(self, step_out: Mapping[str, Any]) -> dict[str, float]:
        """Advance from one ``pcm_cells.step_array`` result; returns this step's dτ per axis."""
        d = {}
        for a in self.axes:
            o = step_out.get(a) or {}
            dec = o.get("decision") or {}
            eps = float(o.get("eps") or 0.0)
            realised = abs(float(dec.get("realised_dg") or 0.0))
            d[a] = realised / eps if eps > 0.0 else 0.0
            self.tau[a] += d[a]
            self._recent[a].append(d[a])
        return d

    def rates(self) -> dict[str, float]:
        """ρ_k = Δτ_k / mean Δτ over the window; all 1.0 before any axis has moved."""
        sums = {a: sum(self._recent[a]) for a in self.axes}
        mean = sum(sums.values()) / len(self.axes)
        if mean <= 0.0:
            return {a: 1.0 for a in self.axes}
        return {a: sums[a] / mean for a in self.axes}

    def dilation(self) -> float:
        """The differentiation in relational time: std of log ρ over the axes that moved (0 if < 2 moved)."""
        logs = [math.log(r) for r in self.rates().values() if r > 0.0]
        if len(logs) < 2:
            return 0.0
        m = sum(logs) / len(logs)
        return math.sqrt(sum((x - m) ** 2 for x in logs) / len(logs))

    def to_json(self) -> dict:
        return {"tau": dict(self.tau), "recent": {a: list(q) for a, q in self._recent.items()}}

    @classmethod
    def from_json(cls, d: Optional[Mapping], window: int = 16) -> "RelationalClock":
        c = cls(window=window)
        if isinstance(d, Mapping):
            for a, v in (d.get("tau") or {}).items():
                if a in c.tau:
                    c.tau[a] = float(v)
            for a, q in (d.get("recent") or {}).items():
                if a in c._recent:
                    c._recent[a].extend(float(x) for x in q)
        return c


# ---------------------------------------------------------------------------
# The shearing λ in relational time
# ---------------------------------------------------------------------------

def poisson_weights(lam: float, n: int = 7) -> list[float]:
    """Poisson(k; lam) for k = 0..n-1, renormalised (the same law as pcm_cells v0.62.0 uses)."""
    lam = max(0.0, float(lam))
    raw = [lam ** k / math.factorial(k) for k in range(n)]
    z = sum(raw) or 1.0
    return [x / z for x in raw]


def ensemble_of(step_out: Mapping[str, Any], axes: Sequence[str] = AXES) -> dict:
    """u per axis and the Poisson weights from a deepen step, whichever pcm_cells version produced it: v0.62.0
    puts them in ``_ensemble``; earlier versions only per axis (``shear.u``) with the dispersion."""
    ens = dict(step_out.get("_ensemble") or {})
    if not ens:
        return {}
    if "u" not in ens:
        ens["u"] = {a: float(((step_out.get(a) or {}).get("shear") or {}).get("u") or 0.0) for a in axes}
    if "weights" not in ens:
        ens["lambda_poisson"] = float(ens.get("dispersion") or 0.0) ** 2
        ens["weights"] = poisson_weights(ens["lambda_poisson"], len(axes))
    return ens


def relational_shear(ensemble: Mapping[str, Any], rho: Mapping[str, float], axes: Sequence[str] = AXES) -> dict:
    """λ_rel and λ_rel_STP from a deepen step's ``_ensemble`` (u per axis, Poisson weights) and the rates."""
    u = ensemble.get("u") or {}
    w = list(ensemble.get("weights") or [1.0 / len(axes)] * len(axes))
    v = {a: float(u.get(a, 0.0)) * float(rho.get(a, 1.0)) for a in axes}
    vbar = sum(wk * v[a] for wk, a in zip(w, axes))
    lam = sum(wk * (v[a] - vbar) ** 2 for wk, a in zip(w, axes))
    den = vbar * vbar + lam
    return {"v": v, "v_bar": vbar, "lambda_rel": lam, "lambda_rel_stp": lam / den if den > 0.0 else 0.0,
            "lambda_poisson": float(ensemble.get("lambda_poisson") or 0.0),
            "counter_rotating": [a for a in axes if v[a] * vbar < 0.0]}


# ---------------------------------------------------------------------------
# The highest outlier relational to the weighted clamp
# ---------------------------------------------------------------------------

def clamp_outlier(unclamped: Mapping[str, float], clamped: Mapping[str, float], weights: Mapping[str, float],
                  rho: Optional[Mapping[str, float]] = None, z: float = 3.5) -> Optional[dict]:
    """k* = argmax w_k ρ_k (unclamped_k − clamped_k) when it stands out (modified z ≥ z); else None."""
    axes = [a for a in unclamped if a in clamped]
    if len(axes) < 3:
        return None
    rho = rho or {}
    s = {a: float(weights.get(a, 0.0)) * float(rho.get(a, 1.0)) * max(0.0, float(unclamped[a]) - float(clamped[a]))
         for a in axes}
    vals = list(s.values())
    med = median(vals)
    mad = median(abs(x - med) for x in vals)
    k = max(axes, key=lambda a: s[a])
    if s[k] <= 0.0:
        return None
    if mad > 0.0:
        zk = 0.6745 * (s[k] - med) / mad
    else:                                   # every other axis identical: the one that differs is the outlier
        others = [s[a] for a in axes if a != k]
        zk = math.inf if all(abs(x - med) == 0.0 for x in others) and s[k] > med else 0.0
    if zk < z:
        return None
    return {"axis": k, "excess": float(unclamped[k]) - float(clamped[k]), "score": s[k], "z": zk, "energy": s[k],
            "scores": s}


# ---------------------------------------------------------------------------
# Feeding the shadow, and keeping it continuous
# ---------------------------------------------------------------------------

def fed_sigma(sigma_l: float, lambda_stp: float, energy: float, gain: float, snr_min: Optional[float],
              cfg: ShearConfig = ShearConfig()) -> tuple[float, dict]:
    raw = sigma_l * (1.0 + gain * (max(0.0, lambda_stp) + max(0.0, energy)))
    cap = sigma_l * max(1.0, cfg.snr_safety * snr_min) if snr_min is not None else raw
    return min(raw, cap), {"raw": raw, "cap": cap, "capped": raw > cap}


def continuity(gain: float, crossing_rate: float, snr_min: float, scenarios: int, k_mean: float,
               cfg: ShearConfig = ShearConfig()) -> tuple[float, str]:
    """Frozen below one crossing per plane per run, dissolving under SNR 1; the gain moves inside the doors."""
    g0, g1 = cfg.g_doors
    frozen_bar = 1.0 / max(1.0, float(scenarios) * max(1.0, float(k_mean)))
    if snr_min < cfg.snr_floor:
        state, g = "dissolving", gain * 0.8
    elif crossing_rate < frozen_bar:
        state, g = "frozen", max(gain * 1.25, gain + 0.05)
    else:
        state, g = "continuous", gain
    return max(g0, min(g1, g)), state


def proposals(history: Iterable[Mapping[str, Any]], cfg: ShearConfig = ShearConfig()) -> list[dict]:
    """Shadow tokens that recur in >= min_sessions sessions under the same clamp-outlier axis."""
    seen: dict[tuple[str, int, str], dict] = {}
    for h in history:
        axis = (h.get("outlier") or {}).get("axis")
        if not axis:
            continue
        for t in h.get("shadow_tokens") or []:
            key = (str(t["token"]), int(t["plane"]), axis)
            rec = seen.setdefault(key, {"token": key[0], "plane": key[1], "axis": axis, "sessions": set(),
                                        "kinds": set(), "weight": 0.0, "energy": 0.0})
            rec["sessions"].add(str(h.get("session")))
            rec["kinds"].add(t.get("kind"))
            rec["weight"] = max(rec["weight"], float(t.get("weight") or 0.0))
            rec["energy"] = max(rec["energy"], float((h.get("outlier") or {}).get("energy") or 0.0))
    out = []
    for rec in seen.values():
        if len(rec["sessions"]) >= cfg.min_sessions:
            out.append({**rec, "sessions": sorted(rec["sessions"]), "kinds": sorted(k for k in rec["kinds"] if k),
                        "status": "proposed", "realisation": "through the six gates, two signers and the grant"})
    out.sort(key=lambda r: (-len(r["sessions"]), -r["energy"], -r["weight"]))
    return out


def fed_run(*, mesh: Any, step_out: Mapping[str, Any], clock: RelationalClock, unclamped: Mapping[str, float],
            clamped: Mapping[str, float], gain: float, snr_min: Optional[float] = None, session: str = "",
            cfg: ShearConfig = ShearConfig(), **shadow_kwargs: Any) -> dict:
    """One shadow run fed by the relational shear and the clamp outlier.  Writes nothing."""
    from .. import shadow_tokens as ST
    clock.tick(step_out)
    rho = clock.rates()
    ens = ensemble_of(step_out)
    shear = relational_shear(ens, rho) if ens else {"lambda_rel": 0.0, "lambda_rel_stp": 0.0, "v": {}, "counter_rotating": []}
    w = {a: wk for a, wk in zip(AXES, ens.get("weights") or [1.0] * len(AXES))}
    out = clamp_outlier(unclamped, clamped, w, rho, cfg.z_outlier)
    sigma, sig_info = fed_sigma(float(mesh.sigma), shear["lambda_rel_stp"], (out or {}).get("energy", 0.0),
                                gain, snr_min, cfg)
    rep = ST.run(mesh=dataclasses.replace(mesh, sigma=sigma), persist=False, **shadow_kwargs)
    if not rep.get("ok"):
        return {"ok": False, "error": rep.get("error"), "sigma": sigma}
    snrs = [e["snr"] for e in rep["emission"]]
    snr_now = min(snrs) if snrs else 0.0
    tot = rep["potential"]["total"]
    imag, rate = float(tot["imag"]), float(tot["crossing_rate"])
    ks = list((rep["problem"].get("k_per_plane") or {}).values()) or [1]
    g_next, state = continuity(gain, rate, snr_now, int(rep["problem"]["scenarios"]), sum(ks) / len(ks), cfg)
    return {"ok": True, "session": session, "rho": rho, "dilation": clock.dilation(), "shear": shear,
            "outlier": out, "sigma_l": float(mesh.sigma), "sigma_eff": sigma, "sigma_info": sig_info,
            "gain": gain, "gain_next": g_next, "continuity": state, "snr_min": snr_now, "imag": imag,
            "crossing_rate": rate,
            "shadow_tokens": rep["shadow_tokens"], "emission": rep["emission"]}
