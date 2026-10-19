"""hidden_work — work extracted from hidden degrees of freedom (Muruga et al., Nature Physics 2026), as a
measured reference and as a memory test for any sampled signal.

The experiment
--------------
Muruga, Ginot, Loos, …, Bechinger, "The extraction of work from hidden degrees of freedom",
Nature Physics (2026), doi:10.1038/s41567-026-03442-w (published 2026-10-16, open access).
Data and MATLAB analysis code: doi:10.5281/zenodo.19881972 (CC BY 4.0).

A 2.7 µm silica colloid sits in an optical trap (stiffness κ = 0.50–40 µN/m) inside a viscoelastic
micellar solution (structural relaxation τ_r ≈ 6 s). Its position X is observed; the micellar network Y
is not. The position is measured twice, δt apart, each time only as "above or below a threshold"
(x_th = 70 nm), giving four outcomes |ij⟩. Then the trap is translated over t_f = 3 s and the work
w = −∫(∂V/∂λ) λ̇ dt is collected.

What it shows, in the paper's terms
-----------------------------------
* Generalised second law (their eq. 1): W ≤ −ΔF + k_BT·I(Z:M), Z = (X, Y) the full state.
* One measurement cannot reach the hidden part. If M depends on Y only through X (eq. 11),
  I((X,Y):M) = I(X:M) (eq. 13): the bound is the same as if Y did not exist.
* Two measurements can. The bound gains I(Y₂:(M₁,M₂)|X₂) (eq. 14), and the outcome-resolved
  divergence in (x, y) is at least the one in x alone (eq. 15).
* For rigid translations of a harmonic trap, a Markovian system gives at most
  W*_m = (κ/2)⟨X⟩²_m (eq. 9): the potential energy the observable carries.
* The memory order parameter (eq. 6)
      Φ(δt) = Σ_ij P_ij ∫₀^∞ [𝓘_|ij⟩(t) − Ĩ_|ij⟩(t)] dt,
  with 𝓘_|ij⟩(t) = D_KL[ρ_|ij⟩(x,t) ∥ ρ_eq(x)] and Ĩ the same for a history-blind ensemble whose
  initial positions match (eq. 4), built by subsampling single-measurement trajectories.
  Φ ≈ 0 means memoryless. Measured: Φ peaks near δt ≈ 1 s; |10⟩ has the largest Φ and yields up
  to about 3·W* (work beyond what the observable stored); |01⟩ has Φ < 0 (the hidden part speeds
  relaxation, less work); |00⟩ yields none.

What is here
------------
* ``SOURCE`` and ``PUBLISHED``: the paper's setup and results, with the source. Observations, not
  tuned constants; none is a threshold anywhere in QUIPU.
* Bounds: ``information_bound`` (eq. 1), ``markov_energetic_bound`` (eq. 9),
  ``translation_work`` (the work of any rigid-translation protocol on the mean dynamics; harmonic
  trap, overdamped, k_BT = γ = 1 unless given), ``best_jump`` (the eq. 9 optimum, reached at once by
  a jump to λ = ⟨X⟩).
* ``memory_order_parameter(series, …)``: eq. 6 estimated from any evenly sampled signal, with the
  paper's construction: threshold outcomes two lags apart, outcome-resolved KL relaxation against
  the stationary distribution, and a history-blind control drawn from the single-measurement
  ensemble with matched initial positions. ``memory_scan`` runs it over a range of δt and reports
  where memory is largest. ``second_measurement_information`` is the observable part of eq. 14's
  extra term: the information about the future that the earlier outcome adds to the later one.
* ``coupled_trace``: the paper's minimal model (a hidden bath particle elastically coupled to the
  observed one), for tests and for calibrating what Φ looks like with and without memory.
* ``VALIDATION``: the estimator run on the paper's own data (bench/validate_hidden_work.py): Φ peaks
  at δt = 1 s and is gone at 200 s, with the paper's sign for every state.

How the fleet uses it
---------------------
Any fleet signal sampled at a steady rate (probe latencies, queue depths, kernel heartbeat gaps) is a
partly observed system: the observed value is X, everything that drives it unseen is Y. Φ(δt) says
whether two readings δt apart predict where the signal goes better than the latest reading alone,
and at which spacing. Φ ≈ 0: the latest reading is enough. Φ > 0: something hidden carries memory,
and a watcher should keep the earlier reading too (the Fleet Harness plan ``hidden-memory`` runs this
on Holon Workers' probe history and shares the result through QUIPU).

What this module does not do
----------------------------
No mapping from the colloid's k_BT to QUIPU's held potentials or to any gate is asserted. It writes only
``brain_kv["entirety:hidden_work"]`` (``_kv_set`` refuses any other key) and is an input to no gate.
Stdlib only. No network: the published numbers are recorded here; the raw data are read only when
a caller passes them in.

翈 — the hidden part is measured by what a second look adds; what QUIPU does with that stays the
operator's decision.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

KV_KEY: str = "entirety:hidden_work"

# ---------------------------------------------------------------------------
# The source and its published numbers
# ---------------------------------------------------------------------------

SOURCE: dict = {
    "title": "The extraction of work from hidden degrees of freedom",
    "authors": "Muruga, L.; Ginot, F.; Loos, S. A. M.; …; Bechinger, C.",
    "journal": "Nature Physics",
    "year": 2026,
    "published": "2026-10-16",
    "doi": "10.1038/s41567-026-03442-w",
    "url": "https://www.nature.com/articles/s41567-026-03442-w",
    "data_doi": "10.5281/zenodo.19881972",
    "data_license": "CC BY 4.0",
    "access": "open access",
}


@dataclass(frozen=True)
class Setup:
    bead_diameter_um: float = 2.7
    solution: str = "8 mM equimolar cetylpyridinium chloride / sodium salicylate (wormlike micelles)"
    structural_relaxation_s: float = 6.0          # τ_r ≈ 6 s
    trap_stiffness_uN_per_m: tuple = (0.50, 40.0)
    laser_nm: int = 532
    frame_rate_hz: float = 100.0
    position_resolution_nm: float = 5.0
    temperature_c: float = 25.0
    threshold_nm: float = 70.0                    # x_th: each measurement is x > x_th or not
    extraction_time_s: float = 3.0                # t_f, constant-velocity translation
    protocols_per_state: int = 100                # about 100 extraction protocols per |ij⟩
    bootstrap_work: int = 2000
    bootstrap_relaxation: int = 500


@dataclass(frozen=True)
class Finding:
    state: str          # |ij⟩: i = first outcome, j = second (1 = above threshold)
    phi_sign: str       # sign of the state's contribution to Φ
    work: str           # what the extraction gave
    note: str = ""


PUBLISHED = {
    "setup": Setup(),
    "phi_peak_delta_t_s": 1.0,                    # Φ(δt) and η*_|11⟩(δt) both peak near δt ≈ 1 s
    "memory_vanishes": "δt → 0 and δt = 200 s ≫ τ_r: relaxation then depends only on the latest outcome",
    "max_w_over_wstar": {"|10⟩": 3.0},           # up to about three times the Markovian bound
    "findings": [
        Finding("|10⟩", "+ (largest)", "up to ≈3·W*: work beyond the energy the observable stored",
                "energy flows back from the hidden network"),
        Finding("|11⟩", "+", "largest absolute work, mainly from the observable's potential energy",
                "η*_|11⟩(δt) peaks near 1 s and stays below 1"),
        Finding("|01⟩", "−", "less work: the hidden part speeds relaxation"),
        Finding("|00⟩", "≈ 0", "none: W stays negative"),
    ],
    "equilibrium_energy_kT": 0.5,
    "limitations": [
        "one colloid, one micellar solution, one harmonic single-well trap",
        "two binary measurements; resolution and number of measurements not varied",
        "rigid translations only; more flexible protocols could extract more",
        "the hidden network is inferred (data and a fictitious-particle model), not observed",
        "the energetic bound (eq. 8) is Markovian; no general tight non-Markovian bound is given",
    ],
}


# What this module's estimator gives on the paper's own data: bench/validate_hidden_work.py on Raw_Data from
# Zenodo 19881972 (9 one-hour trajectories, 3.24 million frames at 100 fps; MD5 of Lokesh_Information_Data.zip
# checked), with the paper's conventions (x_th = 75 nm, 3 s window). Φ in nats·s; E in k_BT, equilibrium 0.5.
VALIDATION = {
    "computed": "2026-10-16",
    "data": "Lokesh_Information_Data.zip / Raw_Data (doi:10.5281/zenodo.19881972)",
    "threshold_nm": 75.0,                        # as in the paper's Figure_2.m / Data_Analyzer (the text says 70 nm)
    "phi_by_delta_s": {0.01: 0.0269, 0.03: 0.0325, 0.1: 0.0478, 0.3: 0.0627, 1.0: 0.0735,
                       3.0: 0.0591, 10.0: 0.0443, 200.0: 0.0007},
    "peak_delta_s": 1.0,                         # the paper: Φ peaks near δt ≈ 1 s
    "state_phi_at_1s": {"|00⟩": 0.022, "|01⟩": -0.456, "|10⟩": 0.839, "|11⟩": 1.058},
    "energy_10_at_1s_kT": {"start": 0.340, "max": 0.678},   # rises past 0.5: energy flowing back from the network
    "agrees": "peak at 1 s; memory gone at 200 s; |10⟩ and |11⟩ above, |01⟩ below, |00⟩ near 0",
}


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def information_bound(mutual_information_nats: float, delta_free_energy: float = 0.0, kT: float = 1.0) -> float:
    """Eq. 1: W ≤ −ΔF + k_BT·I(Z:M)."""
    return -delta_free_energy + kT * mutual_information_nats


def markov_energetic_bound(kappa: float, mean_x: float) -> float:
    """Eq. 9: the most work rigid translations of a harmonic trap can take from a Markovian particle whose
    conditional mean is ⟨X⟩_m: W*_m = (κ/2)⟨X⟩²_m."""
    return 0.5 * kappa * mean_x * mean_x


def translation_work(kappa: float, x0: float, protocol: Sequence[tuple[float, float]], gamma: float = 1.0,
                     steps_per_unit: int = 2000, drift=None) -> float:
    """Work extracted, w = −∫(∂V/∂λ) λ̇ dt, by moving a harmonic trap V = κ(x−λ)²/2 along ``protocol``
    [(t, λ), …] (piecewise linear; a repeated t is a jump), starting with the trap at λ = 0 and the
    particle's mean at x0. Overdamped mean dynamics γ dx̄/dt = −κ(x̄ − λ) + drift(t, x̄); for a harmonic
    trap the variance does not enter the work. ``drift`` adds a force from hidden degrees of freedom
    (None: Markovian)."""
    pts = [(0.0, 0.0)] + [(float(t), float(l)) for t, l in protocol]
    x, w = float(x0), 0.0
    for (t0, l0), (t1, l1) in zip(pts, pts[1:]):
        if t1 <= t0:                                   # a jump: x does not move during it
            w += kappa * (x * (l1 - l0) - 0.5 * (l1 * l1 - l0 * l0))
            continue
        n = max(1, int((t1 - t0) * steps_per_unit))
        dt = (t1 - t0) / n
        rate = (l1 - l0) / (t1 - t0)
        for k in range(n):
            t = t0 + k * dt
            lam = l0 + rate * (t - t0)
            w += kappa * (x - lam) * rate * dt         # −∂V/∂λ · λ̇ = κ(x − λ)λ̇
            f = -kappa * (x - lam) + (drift(t, x) if drift else 0.0)
            x += f / gamma * dt
    return w


def best_jump(kappa: float, mean_x: float) -> dict:
    """The eq. 9 optimum for a Markovian particle: jump the trap to λ = ⟨X⟩ at once and leave it there."""
    return {"lambda": mean_x, "work": translation_work(kappa, mean_x, [(0.0, mean_x)]),
            "bound": markov_energetic_bound(kappa, mean_x)}


# ---------------------------------------------------------------------------
# Estimating Φ(δt) from a sampled signal
# ---------------------------------------------------------------------------

def _edges(values: Sequence[float], bins: int) -> list[float]:
    s = sorted(values)
    n = len(s)
    out = [s[min(n - 1, int(n * k / bins))] for k in range(1, bins)]
    return [e for i, e in enumerate(out) if i == 0 or e > out[i - 1]]     # equal-probability bins


def _bin(x: float, edges: Sequence[float]) -> int:
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _kl(counts: Sequence[int], ref: Sequence[float], alpha: float = 0.5) -> float:
    """D_KL(p ∥ q) in nats, p from counts with a half-count prior so empty bins stay finite."""
    n = sum(counts) + alpha * len(counts)
    total = 0.0
    for c, q in zip(counts, ref):
        p = (c + alpha) / n
        total += p * math.log(p / q)
    return total


@dataclass
class MemoryEstimate:
    delta: int                    # δt in samples
    delta_t: float                # δt in the signal's time unit
    phi: float                    # Φ(δt), nats × time unit
    phi_se: float                 # spread over history-blind redraws (the control's own noise)
    null_mean: float = 0.0        # Φ with the first outcome shuffled (no memory by construction)
    null_sd: float = 0.0
    z: float = 0.0                # (Φ − null mean) / null sd
    by_state: dict = field(default_factory=dict)    # |ij⟩ → {"p", "n", "phi"}
    horizon: int = 0
    samples: int = 0


def memory_order_parameter(series: Sequence[float], delta: int, *, threshold: float | None = None,
                           horizon: int = 50, bins: int = 12, dt: float = 1.0, draws: int = 8,
                           nulls: int = 8, stride: int = 1, min_per_state: int = 30,
                           seed: int = 7) -> MemoryEstimate:
    """Eq. 6 from one evenly sampled stationary signal.

    Each start time s (every ``stride`` samples) gives outcomes i = x[s−δ] > threshold and
    j = x[s] > threshold. For each |ij⟩ with at least ``min_per_state`` starts, 𝓘_|ij⟩(t) is the KL
    divergence of x[s+t] over those starts from the stationary distribution, for t = 0 … horizon. The
    history-blind control draws the same number of starts from all starts with outcome j (any i),
    bin by bin in x[s], so its initial distribution matches (eq. 4); Ĩ is its divergence, averaged over
    ``draws`` redraws. Φ = Σ P_ij Σ_t (𝓘 − Ĩ)·dt. Threshold defaults to the median.

    A finite sample gives Φ ≠ 0 even without memory, so the same estimate is repeated ``nulls`` times
    with the first outcome i shuffled among starts that share the second outcome and the initial bin:
    that keeps every state's initial distribution and removes any memory. ``z`` compares Φ with that
    null; z > 3 is memory the sample can resolve."""
    x = list(map(float, series))
    if threshold is None:
        threshold = sorted(x)[len(x) // 2]
    edges = _edges(x, bins)
    nb = len(edges) + 1
    eq = [0] * nb
    b = [_bin(v, edges) for v in x]
    for k in b:
        eq[k] += 1
    q = [(c + 0.5) / (len(x) + 0.5 * nb) for c in eq]
    hi = [v > threshold for v in x]
    starts = [s for s in range(delta, len(x) - horizon, max(1, stride))]
    first = {s: int(hi[s - delta]) for s in starts}
    pool: dict[int, dict[int, list[int]]] = {0: {}, 1: {}}      # outcome j → initial bin → starts
    for s in starts:
        pool[int(hi[s])].setdefault(b[s], []).append(s)
    rng = random.Random(seed)

    def curve(ss: Sequence[int]) -> list[float]:
        out = []
        for t in range(horizon + 1):
            c = [0] * nb
            for s in ss:
                c[b[s + t]] += 1
            out.append(_kl(c, q))
        return out

    def phi_of(label: Mapping[int, int], n_draws: int):
        groups: dict[tuple[int, int], list[int]] = {}
        for s in starts:
            groups.setdefault((label[s], int(hi[s])), []).append(s)
        total = len(starts)
        states, per_draw = {}, [0.0] * n_draws
        for (i, j), ss in sorted(groups.items()):
            if len(ss) < min_per_state:
                continue
            p = len(ss) / total
            real = curve(ss)
            need: dict[int, int] = {}
            for s in ss:
                need[b[s]] = need.get(b[s], 0) + 1
            per = []
            for _ in range(n_draws):
                blind = []
                for k, n in need.items():
                    cand = pool[j][k]
                    blind.extend(rng.choice(cand) for _ in range(n))
                ctl = curve(blind)
                per.append(sum(r - c for r, c in zip(real, ctl)) * dt)
            states[f"|{i}{j}⟩"] = {"p": round(p, 4), "n": len(ss), "phi": sum(per) / n_draws}
            for d in range(n_draws):
                per_draw[d] += p * per[d]
        return per_draw, states

    per_draw, by_state = phi_of(first, draws)
    mean = sum(per_draw) / draws
    se = (sum((v - mean) ** 2 for v in per_draw) / max(1, draws - 1) / draws) ** 0.5
    null = []
    for _ in range(nulls):
        shuffled: dict[int, int] = {}
        for j in (0, 1):
            for ss in pool[j].values():
                labels = [first[s] for s in ss]
                rng.shuffle(labels)
                shuffled.update(zip(ss, labels))
        nd, _ = phi_of(shuffled, max(1, draws // 2))
        null.append(sum(nd) / len(nd))
    nm = sum(null) / len(null) if null else 0.0
    nsd = (sum((v - nm) ** 2 for v in null) / max(1, len(null) - 1)) ** 0.5 if len(null) > 1 else 0.0
    z = (mean - nm) / nsd if nsd > 0 else 0.0
    return MemoryEstimate(delta, delta * dt, mean, se, nm, nsd, z, by_state, horizon, len(x))


def memory_scan(series: Sequence[float], deltas: Iterable[int], **kw) -> dict:
    """Φ(δt) over several δt; ``peak`` is the δt where Φ stands furthest above its no-memory null, and
    ``significant`` whether it stands more than three null standard deviations above it."""
    rows = [memory_order_parameter(series, d, **kw) for d in deltas]
    best = max(rows, key=lambda r: r.phi - r.null_mean) if rows else None
    memory = bool(best and best.z > 3)
    return {"rows": [asdict(r) for r in rows], "peak": asdict(best) if best else None,
            "significant": memory, "markovian": not memory}


def second_measurement_information(series: Sequence[float], delta: int, ahead: int, *,
                                   threshold: float | None = None, bins: int = 12) -> float:
    """I(X_{s+ahead} : M₁ | M₂) in nats: what the earlier outcome adds about the later state once the later
    outcome is known, the observable part of eq. 14's extra term. 0 for a Markovian signal."""
    x = list(map(float, series))
    if threshold is None:
        threshold = sorted(x)[len(x) // 2]
    edges = _edges(x, bins)
    nb = len(edges) + 1
    cnt: dict[tuple[int, int, int], int] = {}
    for s in range(delta, len(x) - ahead):
        key = (int(x[s - delta] > threshold), int(x[s] > threshold), _bin(x[s + ahead], edges))
        cnt[key] = cnt.get(key, 0) + 1
    n = sum(cnt.values())
    info = 0.0
    for j in (0, 1):
        nj = sum(v for (a, bj, _), v in cnt.items() if bj == j)
        if not nj:
            continue
        for i in (0, 1):
            nij = sum(v for (a, bj, _), v in cnt.items() if a == i and bj == j)
            for k in range(nb):
                nijk = cnt.get((i, j, k), 0)
                if not nijk:
                    continue
                njk = cnt.get((0, j, k), 0) + cnt.get((1, j, k), 0)
                info += nijk / n * math.log(nijk * nj / (nij * njk))
    return info


# ---------------------------------------------------------------------------
# The paper's minimal model: a hidden bath particle elastically coupled to the observed one
# ---------------------------------------------------------------------------

def coupled_trace(n: int, *, kappa: float = 1.0, coupling: float = 1.0, bath_friction: float = 10.0,
                  dt: float = 0.05, kT: float = 1.0, seed: int = 11) -> list[float]:
    """Observed x of the overdamped pair  ẋ = −κx − k(x−y) + ξ,  γ_b ẏ = −k(y−x) + ξ_b  (γ = 1).
    The bath relaxes on γ_b/k; coupling = 0 leaves x Markovian (an Ornstein–Uhlenbeck process)."""
    rng = random.Random(seed)
    x = y = 0.0
    sx, sy = math.sqrt(2 * kT * dt), math.sqrt(2 * kT * dt / bath_friction)
    out = []
    for _ in range(n + 2000):
        fx = -kappa * x - coupling * (x - y)
        fy = -coupling * (y - x) / bath_friction
        x += fx * dt + sx * rng.gauss(0, 1)
        y += fy * dt + sy * rng.gauss(0, 1)
        out.append(x)
    return out[2000:]                                   # drop the approach to equilibrium


# ---------------------------------------------------------------------------
# Reference document and persistence (observability only)
# ---------------------------------------------------------------------------

def reference() -> dict:
    jump = best_jump(1.0, 1.0)
    return {
        "source": SOURCE,
        "published": {**PUBLISHED, "setup": asdict(PUBLISHED["setup"]),
                      "findings": [asdict(f) for f in PUBLISHED["findings"]]},
        "validation": VALIDATION,
        "checks": {
            "markov_bound_harmonic_k1_x1": markov_energetic_bound(1.0, 1.0),
            "jump_to_mean_reaches_bound": round(jump["work"], 6) == round(jump["bound"], 6),
        },
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _kv_set(cn: sqlite3.Connection, key: str, value) -> None:
    if key != KV_KEY:
        raise PermissionError(f"hidden_work writes only {KV_KEY!r}, not {key!r}")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str, ensure_ascii=False),
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def record(cn: sqlite3.Connection, doc: Mapping | None = None) -> dict:
    doc = dict(doc) if doc is not None else reference()
    _kv_set(cn, KV_KEY, doc)
    return doc


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m src.quipu.qpsi.hidden_work",
                                description="Work from hidden degrees of freedom: reference and memory test.")
    p.add_argument("--record", action="store_true", help=f"write the reference to brain_kv[{KV_KEY!r}]")
    p.add_argument("--scan", metavar="FILE", help="Φ(δt) of a signal: a text file of numbers, one per line")
    p.add_argument("--dt", type=float, default=1.0, help="sampling interval of --scan's signal")
    p.add_argument("--deltas", default="1,2,4,8,16", help="δt values in samples")
    p.add_argument("--horizon", type=int, default=50)
    args = p.parse_args(argv)
    if args.scan:
        with open(args.scan, encoding="utf-8") as fh:
            series = [float(v) for v in fh.read().split() if v.strip()]
        out = memory_scan(series, [int(d) for d in args.deltas.split(",")], dt=args.dt, horizon=args.horizon)
    else:
        out = reference()
        if args.record:
            from .. import system_entirety as se
            with se._conn() as cn:
                record(cn, out)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["KV_KEY", "SOURCE", "Setup", "Finding", "PUBLISHED", "VALIDATION", "information_bound", "markov_energetic_bound",
           "translation_work", "best_jump", "MemoryEstimate", "memory_order_parameter", "memory_scan",
           "second_measurement_information", "coupled_trace", "reference", "record"]
