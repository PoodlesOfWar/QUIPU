"""Held ACRE tool candidate: the viscous shear chain (forge format).

One callable, ``tool_viscous_shear_chain(data) -> {result, explanation, confidence}``,
with pure, deterministic stages selected by ``data["stage"]``:

* ``decay``        merged linear/quadratic drag decay (Timmerman & van der Weele,
                   Am. J. Phys. 67, 538 (1999)); limits are the muonic decay of
                   antimony (memoryless) and textile cellulose decay (Ekenstam,
                   self-slowing); drag-deepened torus proximity 1 - (d/N)^(1+kappa*gamma).
* ``field``        von Neumann entropy peaks above the GARD lossy floor
                   delta* = 2*sqrt(3)*sigma/sqrt(n), weighted by Weyl Psi2, decayed
                   along the (2,3) trefoil and its mirror and merged as an
                   opposite-chirality Weyl node pair.
* ``gate``         supernova expansion gate from per-NPC decisions (decaying,
                   needs at least one fresh decision; fresh rADAM state per call).
* ``bond``         LiTaSb Morse rupture force as a multiplanar shear on an
                   exciton-polariton condensate, PC/CR-39 Fresnel, Arrhenius
                   asphalt pyrolysis. Constants are estimates. Informational only.
* ``interactions`` token and edge PROPOSALS (lift = field x gate), never writes.
* ``worklink``     Planck length per universal-time unit, the 13 h : 30 min
                   research ratio, the gap to 1 knot in bits (GARD interstitial
                   channel), 1 knot <-> one overhand (2,3) traversal per relational
                   time unit, capstan hold, and the temporal-spatial work bound.
* ``pompeii``      differential distances of the repo's Pompeii 79 CE case
                   against 1 knot and the research velocity.

No I/O, no environment, no clock, no randomness: the caller supplies everything.
"""
import math

PLANCK_LENGTH_M = 1.616255e-35
PLANCK_TIME_S = 5.391247e-44
C_M_S = 299792458.0
KNOT_M_S = 1852.0 / 3600.0
JULIAN_YEAR_S = 31557600.0
NAUTICAL_MILE_M = 1852.0
EV_PER_ANGSTROM_NN = 1.602176634
K_B_EV = 8.617333262e-5
MUON_LIFETIME_S = 2.1969811e-6
SB_HUFF_FACTOR = 0.85
SB_CAPTURE_RATE_S = 1.0e7
DRAG_KAPPA = 1.0
KNOT_DECAY_A = 0.35
KNOT_DECAY_B = 0.15
KNOT_ARC_SCALE = 8.0
KNOT_AGE_SCALE = 0.05
AXES = ("vision", "touch", "smell", "body", "brain", "perception", "entirety")
PLANE_WEIGHTS = (0.188571, 0.188571, 0.154286, 0.102857, 0.102857, 0.12, 0.142857)
GATE_TAU_S = 1800.0
GATE_WINDOW_S = 3600.0
GATE_MAX_NPCS = 256
RADAM_BETA1 = 0.85
RADAM_BETA2 = 0.999
RADAM_LR = 0.30
RADAM_EPS = 1e-6
TOKEN_LIFT_MAX = 0.10
EDGE_LIFT_MAX = 0.05
CAPSTAN_MU = 0.2
POMPEII = {"aggradation_m_per_yr": 0.0022, "years": 1947.0, "burial_depth_m": 6.614,
           "temporal_depression_m": -7.139, "uncertainty_m": 19.622, "elevation_m": 28.0}
STAGES = ("decay", "field", "gate", "bond", "interactions", "worklink", "pompeii")


def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def _num(x, default):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _r6(x):
    return round(float(x), 6)


# --------------------------------------------------------------------- decay

def _decay(v0, a, b, t):
    v0, a, b, t = max(0.0, v0), max(0.0, a), max(0.0, b), max(0.0, t)
    if v0 == 0.0 or t == 0.0:
        return v0
    if a < 1e-12:
        return v0 / (1.0 + b * v0 * t)
    denom = (a + b * v0) * math.exp(min(a * t, 700.0)) - b * v0
    return 0.0 if denom <= 0.0 else max(0.0, min(v0, a * v0 / denom))


def _half_life(v0, a, b):
    if v0 <= 0.0 or a + b * v0 <= 0.0:
        return None
    lo, hi = 0.0, 1.0
    while _decay(v0, a, b, hi) > 0.5 * v0 and hi < 1e9:
        hi *= 2.0
    k = 0
    while k < 80:
        mid = 0.5 * (lo + hi)
        if _decay(v0, a, b, mid) > 0.5 * v0:
            lo = mid
        else:
            hi = mid
        k += 1
    return 0.5 * (lo + hi)


def _proximity(d, n, gamma):
    x = max(0.0, min(1.0, d / float(max(1, n))))
    return max(0.0, 1.0 - x ** (1.0 + DRAG_KAPPA * _clip01(gamma)))


def _stage_decay(data):
    gamma = _clip01(_num(data.get("gamma"), 0.0))
    coherence = _clip01(_num(data.get("coherence"), 1.0))
    a = KNOT_DECAY_A * (1.0 + gamma)
    b = KNOT_DECAY_B * (1.0 - coherence)
    t1, t2 = _half_life(1.0, a, b), _half_life(0.5, a, b)
    n = int(_num(data.get("n"), 64))
    return {
        "a": _r6(a), "b": _r6(b),
        "textile_fraction": _r6(b / (a + b)) if a + b > 0 else 0.0,
        "half_life_ratio": _r6(t2 / t1) if t1 and t2 else None,
        "crossover_amplitude": _r6(a / b) if b > 0 else None,
        "muonic_rate_s": _r6(SB_HUFF_FACTOR / MUON_LIFETIME_S + SB_CAPTURE_RATE_S),
        "proximity": [{"d": d, "k": _r6(_proximity(d, n, gamma))} for d in (0, n // 8, n // 4, n // 2)],
    }


# --------------------------------------------------------------------- field

def _knot_cells(n, p, q):
    if p == 0 and q == 0:
        return [(0, 0)]
    samples = max(1, n * max(abs(p), abs(q)) * 2)
    cells = []
    for k in range(samples):
        s = k / float(samples)
        cell = (int(round(p * n * s)) % n, int(round(q * n * s)) % n)
        if not cells or cells[-1] != cell:
            cells.append(cell)
    if len(cells) > 1 and cells[-1] == cells[0]:
        cells.pop()
    return cells


def _arc_field(knot, peaks, a, b):
    length = len(knot)
    field = {}
    for pk in peaks:
        anchor = pk["idx"] % length
        age_t = pk["age"] * KNOT_AGE_SCALE
        for m, cell in enumerate(knot):
            arc = min(abs(m - anchor), length - abs(m - anchor)) / float(length)
            val = pk["amplitude"] * _decay(1.0, a, b, arc * KNOT_ARC_SCALE + age_t)
            if val > field.get(cell, 0.0):
                field[cell] = val
    return field


def _peaks(series, psi2, delta_star):
    n = len(series)
    span = max(series) - min(series) if n else 0.0
    if n < 3 or span <= 1e-12:
        return []
    out = []
    for i in range(1, n - 1):
        if not (series[i - 1] < series[i] >= series[i + 1]):
            continue
        left = right = series[i]
        k = i - 1
        while k >= 0 and series[k] <= series[i]:
            left = min(left, series[k])
            k -= 1
        k = i + 1
        while k < n and series[k] <= series[i]:
            right = min(right, series[k])
            k += 1
        prom = series[i] - max(left, right)
        if prom > 0.0 and prom >= delta_star:
            out.append({"idx": i, "value": series[i], "prominence": prom,
                        "amplitude": _clip01(psi2 * prom / span), "age": n - 1 - i})
    return out


def _field(data):
    raw = data.get("entropy_series") or []
    series = [v for v in (_num(x, None) for x in raw) if v is not None]
    psi2 = _clip01(_num(data.get("psi2"), 0.5))
    if len(series) < 3:
        return {"peaks": [], "cells": {}, "delta_star": None, "psi2": psi2,
                "reason": "fewer than 3 entropy points"}
    mean = sum(series) / len(series)
    sigma = _num(data.get("sigma"), math.sqrt(sum((x - mean) ** 2 for x in series) / len(series)))
    delta_star = _num(data.get("delta_star"), 2.0 * math.sqrt(3.0) * max(sigma, 1e-9) / math.sqrt(len(series)))
    gamma = _clip01(_num(data.get("gamma"), 0.0))
    coherence = _clip01(_num(data.get("coherence"), 1.0))
    a, b = KNOT_DECAY_A * (1.0 + gamma), KNOT_DECAY_B * (1.0 - coherence)
    n = int(_num(data.get("n"), 64))
    knot = data.get("knot") or [2, 3]
    p, q = int(knot[0]), int(knot[1])
    peaks = _peaks(series, psi2, delta_star)
    right = _arc_field(_knot_cells(n, p, q), peaks, a, b) if peaks else {}
    left = _arc_field(_knot_cells(n, p, -q), peaks, a, b) if peaks else {}
    cells = {}
    for c in set(right) | set(left):
        r, l = right.get(c, 0.0), left.get(c, 0.0)
        cells[c] = (min(1.0, r + l), max(-1.0, min(1.0, r - l)))
    return {"peaks": peaks, "cells": cells, "delta_star": delta_star, "psi2": psi2,
            "reason": "" if peaks else "no peak clears the lossy floor"}


def _lobe(chi):
    return "R" if chi > 0 else ("L" if chi < 0 else "RL")


def _stage_field(data):
    f = _field(data)
    ranked = sorted(f["cells"].items(), key=lambda kv: (-kv[1][0], kv[0]))[:512]
    chis = [abs(v[1]) for v in f["cells"].values()]
    return {
        "peaks": [{k: _r6(v) if isinstance(v, float) else v for k, v in p.items()} for p in f["peaks"]],
        "delta_star": _r6(f["delta_star"]) if f["delta_star"] is not None else None,
        "psi2": _r6(f["psi2"]),
        "cells": [{"i": c[0], "j": c[1], "amp": _r6(v[0]), "chi": _r6(v[1]), "lobe": _lobe(v[1])} for c, v in ranked],
        "chiral_balance": _r6(sum(chis) / len(chis)) if chis else None,
        "reason": f["reason"],
    }


# ---------------------------------------------------------------------- gate

def _radam(g_re, g_im, external_phase, use_torus):
    if g_im == 0.0:
        g_mag, g_phase = g_re, 0.0
    else:
        g_mag = math.copysign(math.hypot(g_re, g_im), g_re if g_re != 0.0 else 1.0)
        g_phase = math.atan2(g_im, g_re)
    m = (1.0 - RADAM_BETA1) * g_mag
    v = (1.0 - RADAM_BETA2) * g_mag * g_mag
    m_hat, v_hat = m / (1.0 - RADAM_BETA1), v / (1.0 - RADAM_BETA2)
    step = RADAM_LR * m_hat / (math.sqrt(v_hat) + RADAM_EPS)
    theta = g_phase % (2.0 * math.pi)
    if use_torus:
        return 0.5 * (1.0 + math.cos(theta) * math.cos(external_phase)), theta
    return max(0.0, min(1.0, step)), theta


def _gate(data):
    now = _num(data.get("now"), None)
    g_re = 0.6 * _clip01(_num(data.get("broaden"), 0.0)) + 0.4 * _clip01(_num(data.get("liveness"), 0.0))
    potential = _clip01(_num(data.get("actionable_potential"), 0.0))
    sw = swm = cx = cy = 0.0
    fresh = ignored = 0
    for npc in (data.get("npcs") or [])[:GATE_MAX_NPCS]:
        t = _num((npc or {}).get("t"), None) if isinstance(npc, dict) else None
        m = _num((npc or {}).get("magnitude"), None) if isinstance(npc, dict) else None
        if now is None or t is None or m is None or not (-60.0 <= now - t <= GATE_WINDOW_S):
            ignored += 1
            continue
        w = math.exp(-max(0.0, now - t) / GATE_TAU_S)
        m = _clip01(m)
        d = _num(npc.get("direction"), 0.0)
        sw += w
        swm += w * m
        cx += w * m * math.cos(d)
        cy += w * m * math.sin(d)
        fresh += 1
    if fresh == 0:
        return {"value": 0.0, "pressure": 0.0, "g_re": _r6(g_re), "g_im": 0.0, "phase": 0.0,
                "fresh": 0, "ignored": ignored, "reason": "no fresh NPC decision"}
    g_im = swm / max(1.0, sw)
    phase = math.atan2(cy, cx) % (2.0 * math.pi) if (cx or cy) else 0.0
    pressure, theta = _radam(g_re, g_im, phase, g_im > 0.0)
    return {"value": _r6(_clip01(pressure * potential)), "pressure": _r6(pressure), "g_re": _r6(g_re),
            "g_im": _r6(g_im), "phase": _r6(phase), "theta": _r6(theta), "fresh": fresh,
            "ignored": ignored, "reason": ""}


# ---------------------------------------------------------------------- bond

def _morse_nn(d_e, a):
    return max(0.0, 0.5 * a * d_e) * EV_PER_ANGSTROM_NN


def _fresnel(n1, n2):
    r = (n1 - n2) / (n1 + n2)
    return _clip01(1.0 - r * r)


def _stage_bond(data):
    bond = data.get("bond") if isinstance(data.get("bond"), dict) else {}
    li = bond.get("li_sb") or {}
    ta = bond.get("ta_sb") or {}
    f_li = _morse_nn(_num(li.get("d_e_ev"), 1.6), _num(li.get("a_inv_angstrom"), 1.3))
    f_ta = _morse_nn(_num(ta.get("d_e_ev"), 3.0), _num(ta.get("a_inv_angstrom"), 1.6))
    f_rup = min(f_li, f_ta)
    force = _num(bond.get("force_nn"), f_rup)
    ratio = _clip01(force / f_rup) if f_rup > 0 else 0.0
    r = _clip01(_num(bond.get("r"), 0.5))
    weights = bond.get("plane_weights") or PLANE_WEIGHTS
    planes, survival = [], 0.0
    for axis, w in zip(AXES, weights):
        pr = w * ratio
        s = r * _clip01(1.0 - pr * pr)
        survival += w * s
        planes.append({"axis": axis, "weight": _r6(w), "ratio": _r6(pr), "survival": _r6(s)})
    n_cav = _num(bond.get("n_cavity"), 2.7)
    t_pc, t_cr = _fresnel(n_cav, 1.586), _fresnel(n_cav, 1.498)
    throughput = _clip01(_num(bond.get("hopfield"), 0.5)) * 0.5 * (t_pc + t_cr)
    pyro = bond.get("pyrolysis") or {}
    k = _num(pyro.get("prefactor_s"), 1e10) * math.exp(
        -_num(pyro.get("e_a_ev"), 1.55) / (K_B_EV * max(1e-9, _num(pyro.get("temp_k"), 723.0))))
    remaining = _decay(1.0, k, 0.0, _num(bond.get("interaction_time_s"), 1.0))
    return {
        "rupture_nn": {"li_sb": _r6(f_li), "ta_sb": _r6(f_ta), "litasb": _r6(f_rup)},
        "weakest": "li_sb" if f_li <= f_ta else "ta_sb",
        "shear_ratio": _r6(ratio), "planes": planes, "survival": _r6(_clip01(survival)),
        "optics": {"t_pc": _r6(t_pc), "t_cr39": _r6(t_cr), "throughput": _r6(throughput)},
        "pyrolysis": {"k_s": _r6(k), "remaining": _r6(remaining)},
        "yield": _r6(_clip01(survival) * throughput * remaining), "estimate": True,
    }


# -------------------------------------------------------------- interactions

def _stage_interactions(data, field, gate_value):
    cap = int(_num(data.get("cap"), 64))
    occupied = data.get("occupied") or {}
    tokens, on_field = [], {}
    for key in sorted(occupied):
        try:
            i, j = (int(x) for x in str(key).split(","))
        except ValueError:
            continue
        cell = field["cells"].get((i, j))
        if cell is None or gate_value <= 0.0:
            continue
        tok = str(occupied[key])
        on_field[tok] = cell
        tokens.append({"token": tok, "cell": [i, j], "lobe": _lobe(cell[1]), "amplitude": _r6(cell[0]),
                       "lift": _r6(TOKEN_LIFT_MAX * cell[0] * gate_value)})
    tokens.sort(key=lambda t: (-t["lift"], t["token"]))
    edges = []
    for e in data.get("edges") or []:
        if not isinstance(e, (list, tuple)) or len(e) < 2:
            continue
        s, d = str(e[0]), str(e[1])
        if s in on_field and d in on_field:
            amp = min(on_field[s][0], on_field[d][0])
            edges.append({"src": s, "dst": d, "lobe": _lobe(on_field[s][1] + on_field[d][1]),
                          "lift": _r6(EDGE_LIFT_MAX * amp * gate_value)})
    edges.sort(key=lambda e: (-e["lift"], e["src"], e["dst"]))
    return {"tokens": tokens[:cap], "edges": edges[:cap],
            "note": "proposals only; realisation through the six gates, two signers and the grant"}


# ------------------------------------------------------------------ worklink

def _gap_bits(v):
    return _r6(math.log2(KNOT_M_S / v)) if v > 0 else None


def _stage_worklink(data):
    base = data.get("time_base") or "utc_second"
    v0 = PLANCK_LENGTH_M / (PLANCK_TIME_S if base == "planck_time" else 1.0)
    hours = data.get("research_hours") or [13.0, 0.5]
    ratio = _num(hours[0], 13.0) / max(1e-12, _num(hours[1], 0.5))
    v_research = ratio * v0
    sigma, n = max(1e-12, _num(data.get("sigma"), 1.0)), max(1, int(_num(data.get("n_samples"), 1)))
    delta_star = 2.0 * math.sqrt(3.0) * sigma / math.sqrt(n)
    floor_bits = 1.0 if delta_star >= 1.0 else max(1.0, math.log2(1.0 / delta_star))
    gap = math.log2(KNOT_M_S / v_research)
    windings = _num(data.get("windings"), 1.0)
    tau = max(1e-12, _num(data.get("tau"), 1.0))
    theta = 4.0 * math.pi if data.get("theta") == "tight" else 3.0 * math.pi
    kappa = _num(data.get("kappa_uN_per_m"), 1.0) * 1e-6
    mean_x = _num(data.get("mean_x"), 0.0)
    work = 0.5 * kappa * mean_x * mean_x
    return {
        "time_base": base, "v0_m_s": v0, "ratio": _r6(ratio), "ratio_bits": _r6(math.log2(ratio)),
        "v_research_m_s": v_research, "v_knot_m_s": _r6(KNOT_M_S), "equity": KNOT_M_S / v_research,
        "gap_bits": _r6(gap), "floor_bits": _r6(floor_bits), "interstitial_bits": _r6(gap - floor_bits),
        "fits_interstitial": gap > floor_bits,
        "overhand": {"knot": [2, 3], "windings": _r6(windings), "tau": _r6(tau),
                     "knot_speed_m_s": _r6(KNOT_M_S * windings / tau),
                     "theta_rad": _r6(theta), "capstan_hold": _r6(math.exp(CAPSTAN_MU * theta))},
        "work": {"bound_j": work, "power_w": work * windings / tau},
        "note": "declared unit equivalence and information distance, not a physical identity",
    }


# ------------------------------------------------------------------- pompeii

def _stage_pompeii(data, v_research):
    p = dict(POMPEII)
    if isinstance(data.get("pompeii"), dict):
        for k, v in data["pompeii"].items():
            p[k] = _num(v, p.get(k))
    v_aggr = p["aggradation_m_per_yr"] / JULIAN_YEAR_S
    accumulated = p["aggradation_m_per_yr"] * p["years"]
    out = {
        "aggradation_m_s": v_aggr, "aggradation_gap_bits": _gap_bits(v_aggr),
        "aggradation_vs_research_bits": _r6(math.log2(v_aggr / v_research)) if v_research > 0 else None,
        "accumulated_aggradation_m": _r6(accumulated),
        "burial_minus_aggradation_m": _r6(p["burial_depth_m"] - accumulated),
        "burial_minus_depression_m": _r6(p["burial_depth_m"] - abs(p["temporal_depression_m"])),
        "depression_z": _r6(abs(p["temporal_depression_m"]) / p["uncertainty_m"]) if p["uncertainty_m"] > 0 else None,
        "burial_planck_bits": _r6(math.log2(p["burial_depth_m"] / PLANCK_LENGTH_M)),
        "burial_nautical_miles": _r6(p["burial_depth_m"] / NAUTICAL_MILE_M),
        "elevation_knot_hours": _r6(p["elevation_m"] / NAUTICAL_MILE_M),
    }
    if "terminal_velocity_m_s" in p and p["terminal_velocity_m_s"]:
        out["terminal_gap_bits"] = _gap_bits(abs(p["terminal_velocity_m_s"]))
    out["resolved"] = out["depression_z"] is not None and out["depression_z"] >= 1.0
    return out


# ---------------------------------------------------------------------- tool

def tool_viscous_shear_chain(data: dict) -> dict:
    """Run the viscous shear chain; see the module docstring for the stages."""
    data = data if isinstance(data, dict) else {}
    stage = data.get("stage") or "all"
    wanted = STAGES if stage == "all" else ((stage,) if stage in STAGES else ())
    if not wanted:
        return {"result": {"stage": stage}, "explanation": f"unknown stage {stage!r}", "confidence": 0.0}
    real = {
        "decay": "gamma" in data or "coherence" in data,
        "field": len(data.get("entropy_series") or []) >= 3,
        "gate": bool(data.get("npcs")),
        "bond": isinstance(data.get("bond"), dict) and "r" in data["bond"],
        "interactions": bool(data.get("occupied")),
        "worklink": "research_hours" in data,
        "pompeii": isinstance(data.get("pompeii"), dict),
    }
    result = {"stage": stage, "inputs_real": {s: real[s] for s in wanted}}
    gate = _gate(data) if ("gate" in wanted or "interactions" in wanted) else None
    field = _field(data) if "interactions" in wanted else None
    if "decay" in wanted:
        result["decay"] = _stage_decay(data)
    if "field" in wanted:
        result["field"] = _stage_field(data)
    if "gate" in wanted:
        result["gate"] = gate
    if "bond" in wanted:
        result["bond"] = _stage_bond(data)
    if "interactions" in wanted:
        result["interactions"] = _stage_interactions(data, field, gate["value"])
    work = _stage_worklink(data) if ("worklink" in wanted or "pompeii" in wanted) else None
    if "worklink" in wanted:
        result["worklink"] = work
    if "pompeii" in wanted:
        result["pompeii"] = _stage_pompeii(data, work["v_research_m_s"])
    n_real = sum(1 for s in wanted if real[s])
    damp = 0.6 if ("bond" in wanted and not real["bond"]) else 1.0
    confidence = round(0.9 * n_real / len(wanted) * damp, 3)
    parts = [s for s in wanted if real[s]]
    explanation = (f"stages {', '.join(wanted)}; real inputs for {', '.join(parts) or 'none'}; "
                   "proposals only, nothing is written")
    return {"result": result, "explanation": explanation, "confidence": confidence}
