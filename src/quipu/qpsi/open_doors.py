"""open_doors — the λ run's own settings tuned by the pruned residual potential, inside the
doors the operator opens.

What may move and what may not
------------------------------
The λ run (``shadow_tokens``) has two kinds of setting.  Equity, the cluster threshold and
the plane contrast are the operator's: λ is the price the operator chose to pay for an
equitable share, so nothing here may move a setting to make that price look smaller.  Three
settings are measurements the run can make better, and only these can be doors:

    k_per_plane   how many tokens each movement voices (one door per movement: k:I … k:VI)
    candidates    how many of the mesh's tokens the problem sees (the pool)
    scenarios     how many Monte Carlo draws a run takes

A door is open when the operator's grant (``QUIPU_LAMBDA_DOORS``, JSON or a path to JSON:
``{"k_per_plane": [lo, hi], "candidates": [lo, hi], "scenarios": [lo, hi]}``) gives the
setting bounds.  The system moves an open setting only inside them: the 2026-09-22
constrained-gate ruling, as with the edge allocation and the pulse.  A closed door holds the
operator's value (``QUIPU_LAMBDA_K`` / ``_CANDIDATES`` / ``_SCENARIOS``); narrowing a door
clamps the value at the next cycle; closing it restores the operator's value.  With no grant
nothing moves, and the charges are still measured and reported with where they point.
Hard caps (k ≤ 64, candidates ≤ 4096, scenarios ≤ 512) only narrow a grant, for CPU.

The signal: the pruned residual potential (``shadow_tokens.potential``)
-----------------------------------------------------------------------
Each (token, movement) is a two-regime record across the scenarios, voiced or pruned.  Real
part = the pruned excess (alignment the coupling pruned, plus what the pool cut before the
problem started); imaginary part = crossing rate × separation (the share of the voice that
changes between two draws, times how far apart its two regimes sit).  This is Perceptopoly's
``residual_potential``, applied to the shadow tokens.

One λ period: four operations (named by the operator, 2026-10-02)
-----------------------------------------------------------------
Rogue (absorb).    Each door holds a charge a ∈ [0, 1].  The share of a run's potential the
                   door's setting could discharge is its stimulus x, and a ← a + x(1 − a).
                   Between runs the charge relaxes, a ← a·exp(−Δt/τ), with τ three times the
                   loop's own median gap.  A charge that is not discharged fades.
                     k:<movement>  x = the share of the movement's crossing rate the best
                                   voice size inside the door would remove (``k_profile``)
                     candidates    x = the pool excess's share of the potential's magnitude
                                   (tokens past the pool that clear a plane's cut); it points
                                   down when the pool's last quarter is never voiced and nothing
                                   past the pool would be
                     scenarios     x = 1 after a mirror test the noise could not decide
Gambit (charge).   The most charged open door at a ≥ ½ fires: its setting moves toward the
                   target by the share a of the distance (at least one unit), inside the door.
                   Firing spends the charge in proportion.  One door per period, so each move
                   is tested on its own.
Gauche (mirror).   The moved setting runs again on the same seed, so the plane draws are the
                   same (common random numbers), and the door's own measure is compared with
                   the live run's.  The standard error is a paired bootstrap over the scenarios.
                     k:<movement>  that movement's crossing rate
                     candidates    the pool excess, provided the mean crossing rate does not
                                   rise beyond noise
                   discharged: the measure fell by more than 2 SE.  inconclusive: within 2 SE
                   (this charges the scenarios door).  charged: it rose.
                   A scenarios move changes precision, not the potential, so it is applied
                   without a mirror.
Grey (transform).  A discharged move becomes the live setting, and the previous value is kept.
                   The next live run confirms it: if the measure comes back worse than the
                   baseline by more than 2 combined SE, Grey returns to the previous value.

Every step is recorded in ``brain_kv["entirety:lambda:doors"]`` (history capped at 200) and in
the run's ``doors`` field on ``GET /lambda``.  Nothing here writes to the mesh, the gates or
the marketplace.

    python -m src.quipu.qpsi.open_doors status
    python -m src.quipu.qpsi.open_doors cycle

翈 — the grant opens the door; the potential decides what moves through it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

KV_DOORS = "entirety:lambda:doors"
GRANT_ENV = "QUIPU_LAMBDA_DOORS"
HARD_CAPS: dict[str, tuple[int, int]] = {"k_per_plane": (1, 64), "candidates": (16, 4096), "scenarios": (8, 512)}
OPERATORS_ALONE = ("equity", "threshold", "contrast", "clusters")
FIRE_AT = 0.5
Z = 2.0
BOOTSTRAP = 200
HISTORY_CAP = 200
TAU_FALLBACK_S = 5400.0


# ---------------------------------------------------------------------------
# The operator's settings and grant
# ---------------------------------------------------------------------------

def _env_num(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "").strip() or default)
        return v if v > 0 else default
    except ValueError:
        return default


def base_settings() -> dict[str, Any]:
    """The operator's values: what a closed door holds."""
    return {"k_per_plane": int(_env_num("QUIPU_LAMBDA_K", 8)),
            "candidates": int(_env_num("QUIPU_LAMBDA_CANDIDATES", 384)),
            "scenarios": int(_env_num("QUIPU_LAMBDA_SCENARIOS", 64)),
            "extension": int(_env_num("QUIPU_LAMBDA_EXTENSION", 256)),
            "equity": min(1.0, _env_num("QUIPU_LAMBDA_EQUITY", 0.5))}


def grant(raw: Optional[str] = None) -> dict[str, Any]:
    """The doors the operator opened: ``{"doors": {name: (lo, hi)}, "error": …}``."""
    raw = (os.environ.get(GRANT_ENV, "") if raw is None else raw).strip()
    if not raw:
        return {"doors": {}, "error": None, "grant_ref": None}
    try:
        if not raw.startswith("{"):
            raw = Path(raw).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except Exception as exc:
        return {"doors": {}, "error": f"grant unreadable ({exc}); every door is closed", "grant_ref": None}
    doors: dict[str, tuple[int, int]] = {}
    errors: list[str] = []
    for name, (cap_lo, cap_hi) in HARD_CAPS.items():
        if name not in data:
            continue
        try:
            lo, hi = int(data[name][0]), int(data[name][1])
        except Exception:
            errors.append(f"{name}: expected [lo, hi]")
            continue
        lo, hi = max(lo, cap_lo), min(hi, cap_hi)
        if lo > hi:
            errors.append(f"{name}: empty after the hard caps")
            continue
        doors[name] = (lo, hi)
    alone = sorted(k for k in data if k in OPERATORS_ALONE)
    if alone:
        errors.append("not doors, the operator's alone: " + ", ".join(alone))
    unknown = sorted(k for k in data if k not in HARD_CAPS and k not in OPERATORS_ALONE
                     and k not in ("_doc", "grant_ref"))
    if unknown:
        errors.append("unknown: " + ", ".join(unknown))
    return {"doors": doors, "error": "; ".join(errors) or None, "grant_ref": data.get("grant_ref")}


def _family(door: str) -> str:
    return "k_per_plane" if door.startswith("k:") else door


def _clamp(v: int, rng: tuple[int, int]) -> int:
    return int(min(max(int(v), rng[0]), rng[1]))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _kv():
    from .. import brain_kv
    return brain_kv


def load_state() -> dict[str, Any]:
    st = _kv().kv_get_json(KV_DOORS, None)
    st = st if isinstance(st, dict) else {}
    st.setdefault("values", {})
    st["values"].setdefault("k", {})
    st.setdefault("charges", {})
    st.setdefault("pending", None)
    st.setdefault("clock", [])
    st.setdefault("history", [])
    st.setdefault("mirrors", [])
    return st


def save_state(st: dict[str, Any]) -> None:
    st["history"] = st["history"][-HISTORY_CAP:]
    st["mirrors"] = st["mirrors"][-10:]
    _kv().kv_set_json(KV_DOORS, st)


def live_values(st: dict[str, Any], base: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    """The settings the next live run uses: open doors hold the tuned value inside the
    door; closed doors hold the operator's value."""
    doors = g["doors"]
    out: dict[str, Any] = {"k": {}, "k_default": base["k_per_plane"]}
    if "k_per_plane" in doors:
        out["k_default"] = _clamp(base["k_per_plane"], doors["k_per_plane"])
        out["k"] = {p: _clamp(v, doors["k_per_plane"]) for p, v in st["values"].get("k", {}).items()}
    for name in ("candidates", "scenarios"):
        v = st["values"].get(name, base[name])
        out[name] = _clamp(v, doors[name]) if name in doors else base[name]
    return out


def _k_arg(values: dict[str, Any]) -> dict[str, int]:
    return {"*": int(values["k_default"]), **{p: int(v) for p, v in values["k"].items()}}


# ---------------------------------------------------------------------------
# Measures (fast forms of shadow_tokens.potential for the mirror and the bootstrap)
# ---------------------------------------------------------------------------

def plane_rate(sim, j: int, idx: Optional[np.ndarray] = None) -> float:
    """The share of plane j's voice that changes between two scenarios."""
    sel = sim.milp[:, :, j] if idx is None else sim.milp[idx, :, j]
    f = sel.mean(axis=0)
    return float((2.0 * f * (1.0 - f)).sum() / (2.0 * float(sim.K[j])))


def mean_rate(sim, idx: Optional[np.ndarray] = None) -> float:
    return float(np.mean([plane_rate(sim, j, idx) for j in range(len(sim.planes))]))


def pool_excess(sim, idx: Optional[np.ndarray] = None) -> float:
    if not sim.ext_S.shape[1]:
        return 0.0
    X = sim.ext_S if idx is None else sim.ext_S[idx]
    cf = sim.cut_free if idx is None else sim.cut_free[idx]
    over = np.clip(X.astype(float) - cf[:, None, :], 0.0, None)
    return float((over.sum(axis=(0, 1)) / max(1, X.shape[0]) / sim.K.astype(float)).mean())


def _boot_idx(n: int, seed: int, B: int = BOOTSTRAP) -> list[np.ndarray]:
    rng = np.random.default_rng(seed ^ 0x5EED)
    return [rng.integers(0, n, n) for _ in range(B)]


def _se(sim, measure: Callable[[Any, Optional[np.ndarray]], float]) -> float:
    vals = [measure(sim, idx) for idx in _boot_idx(sim.n, sim.seed)]
    return float(np.std(vals))


def _paired(live, mirror, measure: Callable[[Any, Optional[np.ndarray]], float]) -> dict[str, float]:
    """mirror − live on the same resampled scenarios (both runs share the seed and the draws)."""
    n = min(live.n, mirror.n)
    diffs = [measure(mirror, idx) - measure(live, idx) for idx in _boot_idx(n, live.seed)]
    return {"live": round(measure(live, None), 6), "mirror": round(measure(mirror, None), 6),
            "diff": round(measure(mirror, None) - measure(live, None), 6), "se": round(float(np.std(diffs)), 6)}


def _verdict(d: dict[str, float]) -> str:
    if d["se"] <= 1e-12:
        return "inconclusive" if abs(d["diff"]) <= 1e-12 else ("discharged" if d["diff"] < 0 else "charged")
    if d["diff"] < -Z * d["se"]:
        return "discharged"
    if d["diff"] > Z * d["se"]:
        return "charged"
    return "inconclusive"


def _measure_for(door: str, sim) -> tuple[str, Callable[[Any, Optional[np.ndarray]], float]]:
    if door.startswith("k:"):
        name = door[2:]
        [p.n for p in sim.planes].index(name)          # ValueError when the movement is not voiced
        return f"crossing_rate:{name}", (lambda s, idx, _n=name: plane_rate(
            s, [p.n for p in s.planes].index(_n), idx))
    return "pool_excess", pool_excess


# ---------------------------------------------------------------------------
# Rogue — absorb
# ---------------------------------------------------------------------------

def _tau(clock: list[float]) -> float:
    gaps = [b - a for a, b in zip(clock, clock[1:]) if b > a]
    return 3.0 * statistics.median(gaps[-3:]) if gaps else TAU_FALLBACK_S


def rogue_absorb(st: dict[str, Any], sim, values: dict[str, Any], g: dict[str, Any], now: float,
                 tau: float) -> dict[str, dict[str, Any]]:
    """Relax every charge for the time since it was last touched, then add this run's
    stimulus to each door it points at."""
    from .. import shadow_tokens as ST
    charges = st["charges"]
    for ch in charges.values():
        dt = max(0.0, now - float(ch.get("at", now)))
        ch["a"] = float(ch.get("a", 0.0)) * math.exp(-dt / max(tau, 1.0))

    def add(door: str, x: float, target: Optional[int], why: str) -> None:
        x = float(min(max(x, 0.0), 1.0))
        ch = charges.setdefault(door, {"a": 0.0})
        ch["a"] = ch["a"] + x * (1.0 - ch["a"])
        ch.update({"x": round(x, 4), "target": target, "why": why, "at": now})

    # k:<movement> — the voice size whose cut the noise crosses least, inside the door
    lo, hi = g["doors"].get("k_per_plane", (1, max(2 * int(sim.K.max()), 2)))
    prof = ST.k_profile(sim, lo, hi)
    for j, p in enumerate(sim.planes):
        cur = int(sim.K[j])
        rates = prof[p.n]
        if cur not in rates:                     # the same free-emission measure at the current K
            rates.update(ST.k_profile(sim, cur, cur)[p.n])
        best = min(rates, key=lambda kk: (rates[kk], abs(kk - cur)))
        gain = (rates[cur] - rates[best]) / max(rates[cur], 1e-9)
        add(f"k:{p.n}", gain if best != cur else 0.0, best,
            f"crossing rate {rates[cur]:.3f} at K={cur}, {rates[best]:.3f} at K={best}")

    # candidates — what the pool cut before the problem started
    pot = ST.potential(sim)["total"]
    cur_n = len(sim.mesh.tokens)
    pool = float(pot["pool_excess"])
    if pool > 0:
        add("candidates", pool / max(float(pot["magnitude"]), 1e-12), cur_n + int(sim.ext_S.shape[1]),
            f"pool excess {pool:.4f} of potential {float(pot['magnitude']):.4f}; "
            f"{pot['pool_would_enter']} tokens past the pool clear a plane's cut")
    else:
        tail = slice(int(cur_n * 0.75), cur_n)
        used = bool(sim.milp[:, tail, :].any() or sim.free[:, tail, :].any())
        if not used and sim.ext_S.shape[1]:
            add("candidates", 0.5, max(int(sim.K.sum()) * 2, int(round(cur_n * 0.75))),
                "the pool's last quarter was never voiced and nothing past the pool would be")
        else:
            add("candidates", 0.0, None, "nothing past the pool clears a plane's cut")

    # scenarios — the last mirror could not tell its two runs apart
    last = st["mirrors"][-1] if st["mirrors"] else None
    cur_s = sim.n
    if last and last.get("verdict") == "inconclusive":
        add("scenarios", 1.0, int(math.ceil(cur_s * 1.5)), "the last mirror test was inside the noise")
    elif len(st["mirrors"]) >= 2 and all(m.get("verdict") in ("discharged", "charged")
                                         and abs(m.get("z") or 0.0) > 6 for m in st["mirrors"][-2:]):
        add("scenarios", 0.5, max(8, int(math.floor(cur_s * 0.75))),
            "the last two mirror tests were decided by more than 6 SE")
    else:
        add("scenarios", 0.0, None, "the mirror tests are decided")
    return charges


# ---------------------------------------------------------------------------
# Gambit — charge and release
# ---------------------------------------------------------------------------

def _current(door: str, values: dict[str, Any]) -> int:
    if door.startswith("k:"):
        return int(values["k"].get(door[2:], values["k_default"]))
    return int(values[door])


def gambit_fire(st: dict[str, Any], values: dict[str, Any], g: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The most charged open door at a ≥ ½ moves a share a of the way to its target."""
    ready = []
    for i, (door, ch) in enumerate(sorted(st["charges"].items())):
        fam = _family(door)
        if fam not in g["doors"] or ch.get("target") is None or ch.get("a", 0.0) < FIRE_AT:
            continue
        cur = _current(door, values)
        tgt = _clamp(int(ch["target"]), g["doors"][fam])
        if tgt != cur:
            ready.append((ch["a"], -i, door, cur, tgt))
    if not ready:
        return None
    a, _, door, cur, tgt = max(ready)
    step = int(round(a * (tgt - cur))) or (1 if tgt > cur else -1)
    new = _clamp(cur + step, g["doors"][_family(door)])
    spent = abs(new - cur) / abs(tgt - cur)
    st["charges"][door]["a"] = a * (1.0 - spent)
    return {"door": door, "old": cur, "new": new, "target": tgt, "charge": round(a, 4)}


# ---------------------------------------------------------------------------
# Gauche — mirror
# ---------------------------------------------------------------------------

def gauche_mirror(live, proposal: dict[str, Any], values: dict[str, Any], base: dict[str, Any],
                  full_mesh, simulate_kwargs: dict[str, Any]) -> dict[str, Any]:
    """The moved setting on the same seed; the door's measure, mirror − live, paired."""
    from .. import shadow_tokens as ST
    moved = {"k": dict(values["k"]), "k_default": values["k_default"],
             "candidates": values["candidates"], "scenarios": live.n}
    door = proposal["door"]
    if door.startswith("k:"):
        moved["k"][door[2:]] = proposal["new"]
    else:
        moved[door] = proposal["new"]
    mesh = full_mesh.pool(moved["candidates"], base["extension"])
    mirror, err = ST.simulate(scenarios=live.n, k=_k_arg(moved), equity=live.equity, mesh=mesh,
                              seed=live.seed, contrast=live.contrast, **simulate_kwargs)
    if err is not None:
        return {"door": door, "verdict": "error", "error": err.get("error")}
    name, measure = _measure_for(door, live)
    d = _paired(live, mirror, measure)
    verdict = _verdict(d)
    out = {"door": door, "measure": name, **d, "verdict": verdict,
           "z": round(d["diff"] / d["se"], 2) if d["se"] > 0 else None}
    if door == "candidates" and verdict == "discharged":
        guard = _paired(live, mirror, mean_rate)
        out["crossing_rate_guard"] = guard
        if guard["diff"] > Z * guard["se"] and guard["diff"] > 0:
            out["verdict"] = "charged"
            out["why"] = "the pool excess fell but the voices became less stable"
    return out


# ---------------------------------------------------------------------------
# Grey — transform, confirm, return
# ---------------------------------------------------------------------------

def _set_value(st: dict[str, Any], door: str, v: int) -> None:
    if door.startswith("k:"):
        st["values"]["k"][door[2:]] = int(v)
    else:
        st["values"][door] = int(v)


def _log(st: dict[str, Any], now: float, kind: str, **kw: Any) -> dict[str, Any]:
    e = {"at": now, "kind": kind, **kw}
    st["history"].append(e)
    return e


def grey_transform(st: dict[str, Any], proposal: dict[str, Any], mirror: Optional[dict[str, Any]],
                   live, now: float) -> dict[str, Any]:
    door = proposal["door"]
    if door == "scenarios":
        _set_value(st, door, proposal["new"])
        return _log(st, now, "precision", door=door, old=proposal["old"], new=proposal["new"],
                    note="scenarios change precision, not the potential: applied without a mirror")
    if mirror is None or mirror.get("verdict") != "discharged":
        return _log(st, now, "held", door=door, old=proposal["old"], new=proposal["new"],
                    verdict=(mirror or {}).get("verdict"), measure=(mirror or {}).get("measure"),
                    diff=(mirror or {}).get("diff"), se=(mirror or {}).get("se"))
    _set_value(st, door, proposal["new"])
    _, measure = _measure_for(door, live)
    st["pending"] = {"door": door, "old": proposal["old"], "new": proposal["new"],
                     "measure": mirror["measure"], "baseline": mirror["live"],
                     "baseline_se": _se(live, measure), "at": now}
    return _log(st, now, "transformed", door=door, old=proposal["old"], new=proposal["new"],
                measure=mirror["measure"], live=mirror["live"], mirror=mirror["mirror"],
                diff=mirror["diff"], se=mirror["se"])


def grey_confirm(st: dict[str, Any], live, now: float) -> Optional[dict[str, Any]]:
    """The first live run after a transformation: keep it, or return to the previous value."""
    pend = st.get("pending")
    if not pend:
        return None
    st["pending"] = None
    door = pend["door"]
    try:
        _, measure = _measure_for(door, live)
    except ValueError:
        return _log(st, now, "dropped", door=door, note="the movement is no longer voiced")
    m, se = measure(live, None), _se(live, measure)
    bar = pend["baseline"] + Z * math.hypot(pend.get("baseline_se", 0.0), se)
    if m > bar:
        _set_value(st, door, pend["old"])
        return _log(st, now, "reverted", door=door, old=pend["new"], new=pend["old"],
                    measure=pend["measure"], now_value=round(m, 6), baseline=pend["baseline"],
                    bar=round(bar, 6))
    return _log(st, now, "confirmed", door=door, value=pend["new"], measure=pend["measure"],
                now_value=round(m, 6), baseline=pend["baseline"], bar=round(bar, 6))


# ---------------------------------------------------------------------------
# One λ period
# ---------------------------------------------------------------------------

def cycle(now: Optional[float] = None, *, full_mesh=None, essay: Optional[str] = None,
          clusters: Optional[list[dict[str, Any]]] = None, seed: Optional[int] = None,
          base: Optional[dict[str, Any]] = None, grant_raw: Optional[str] = None,
          persist: bool = True) -> dict[str, Any]:
    """Live run → Grey confirms the last move → Rogue absorbs → Gambit fires → Gauche mirrors
    → Grey transforms.  Returns the live run's report with its ``doors`` field."""
    from .. import shadow_tokens as ST
    now = time.time() if now is None else float(now)
    base = base or base_settings()
    g = grant(grant_raw)
    st = load_state()
    tau = _tau(st["clock"] + [now])
    st["clock"] = (st["clock"] + [now])[-4:]
    values = live_values(st, base, g)
    n_full = max([values["candidates"], g["doors"].get("candidates", (0, 0))[1]])
    full = full_mesh if full_mesh is not None else ST.load_mesh(n_full, n_extension=base["extension"])
    kwargs = {"essay": essay, "clusters": clusters}
    live, err = ST.simulate(scenarios=values["scenarios"], k=_k_arg(values), equity=base["equity"],
                            mesh=full.pool(values["candidates"], base["extension"]), seed=seed, **kwargs)
    if err is not None:
        return err
    # the planes this run voiced fix the k values (a movement no longer voiced keeps its value)
    values["k"].update({p.n: int(live.K[j]) for j, p in enumerate(live.planes)})
    out = ST.report(live)
    confirmed = grey_confirm(st, live, now)
    if confirmed and confirmed["kind"] == "reverted":
        values = live_values(st, base, g)
    rogue_absorb(st, live, values, g, now, tau)
    proposal = gambit_fire(st, values, g)
    mirror = None
    step = None
    if proposal:
        if proposal["door"] != "scenarios":
            mirror = gauche_mirror(live, proposal, values, base, full, kwargs)
            st["mirrors"].append({k: mirror.get(k) for k in ("door", "measure", "diff", "se", "z", "verdict")})
        step = grey_transform(st, proposal, mirror, live, now)
    pressure = sorted(d for d, ch in st["charges"].items()
                      if ch.get("a", 0) >= FIRE_AT and ch.get("target") is not None
                      and _family(d) not in g["doors"] and ch["target"] != _current(d, values))
    out["doors"] = {
        "grant": {k: list(v) for k, v in g["doors"].items()}, "grant_error": g["error"],
        "grant_ref": g.get("grant_ref"), "base": {k: base[k] for k in ("k_per_plane", "candidates", "scenarios")},
        "values": {"k": dict(sorted(live_values(st, base, g)["k"].items())),
                   "k_default": live_values(st, base, g)["k_default"],
                   "candidates": live_values(st, base, g)["candidates"],
                   "scenarios": live_values(st, base, g)["scenarios"]},
        "charges": {d: {"a": round(ch.get("a", 0.0), 4), "x": ch.get("x"), "target": ch.get("target"),
                        "open": _family(d) in g["doors"], "why": ch.get("why")}
                    for d, ch in sorted(st["charges"].items())},
        "tau_s": round(tau, 1), "confirmed": confirmed, "fired": proposal, "mirror": mirror, "step": step,
        "closed_pressure": pressure,
        "note": "doors are the λ run's k per movement, candidate pool and scenarios, inside the operator's "
                "grant (QUIPU_LAMBDA_DOORS); equity, threshold and contrast are the operator's alone.",
    }
    if persist:
        save_state(st)
        _kv().kv_set_json(ST.KV_LATEST, out)
    return out


def status() -> dict[str, Any]:
    st = load_state()
    g = grant()
    base = base_settings()
    return {"grant": {k: list(v) for k, v in g["doors"].items()}, "grant_error": g["error"],
            "base": base, "values": live_values(st, base, g), "charges": st["charges"],
            "pending": st["pending"], "history": st["history"][-20:]}


def _main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="open_doors", description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["status", "cycle"])
    a = ap.parse_args(argv)
    out = status() if a.cmd == "status" else cycle()
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
