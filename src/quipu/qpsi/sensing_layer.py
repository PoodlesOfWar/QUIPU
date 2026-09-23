"""sensing_layer — the terminals of qpsi.annealed_senses as one coupled layer,
held at the edge of chaos, fed by every record QUIPU keeps, and able to tell the
SOMN which sources have gone silent.

Three parts, each measured before it was wired (CHANGELOG 0.38.0):

1. Feeds — more than ingest
---------------------------
    ingest     corpus_ingest:history (documents per source per run)
    mesh       proprioception: mesh_slm_vocab.last_seen × mesh_slm_embed.  Every
               token the mesh wrote carries its seven-axis profile; the mass
               written on each axis per minute is an attempt of terminal
               ``mesh:<axis>``.  The mesh reports where it actually moved, not
               what a source said it sent.  Read only; a token re-touched keeps
               only its latest touch, so this record is lossy by construction.
    observer   observer_service counters (observer:<source>:stats).  They are
               cumulative, so the layer keeps the last counters and a capped
               event list at ``entirety:senses:observer`` and writes it only
               when a counter moved (the planck-gate rule: persistence on
               displacement, not on schedule).  First sight is a baseline, not
               an event.  Routed to the profile's axis.
    Each feed is thresholded against its own population (documents against
    documents, embedding mass against embedding mass).  The host's material
    state is not a feed here: system_entirety already injects it into the axes.

2. Criticality — the coupled layer
----------------------------------
Terminals are units of a linear-saturating network.  Between attempts

    dA/dt = M A,   M = −diag(1/τ_i) + g·W,   W_ij = max(0, cos(θ_i − θ_j)), i ≠ j

θ_i the torus angle of unit i's axis (2π·k/7, the geometry weyl_centroid
already uses); unrouted units are uncoupled.  An attempt adds its stimulus,
A_i ← A_i + x(1 − A_i); A is clipped to [0, 1].  M is symmetric, so its
largest eigenvalue λ_max is the largest Lyapunov exponent of the linear part.

The gain is not set; it is measured.  g_edge is the gain at which λ_max
reaches −1/T (T the record's own span: the ordered side of the edge, the
slowest mode remembering the whole record).  Below it, g is searched on a
geometric grid (2⁻¹² … 1 of g_edge, then golden-section on log g) for the
maximum of the layer's response entropy — the Shannon entropy of its mean
activity over the record.  An ordered layer stays low and a saturated one
stays high; both score low.  The peak is where the layer uses its whole range:
the dynamic-range argument for criticality in sensing (Kinouchi & Copelli,
Nat. Phys. 2, 348, 2006; Hochstetter et al., Nat. Commun. 12, 4008, 2021).
Variance was tried first and rejected: it rewards a layer that only swings
between empty and saturated.  Reported: g/g_edge, λ_max, the one-bin branching
ratio σ = e^{λ_max·Δ} on the ingest clock, and the share of activity that
arrived by coupling.  The critical gain is recomputed when a new attempt
arrives and cached until then.

Measured on the live record (2026-09-23 20:29Z): g* = 0.93·g_edge,
λ_max = −4.9×10⁻⁵ s⁻¹ (≈ 5.7 h memory), σ = 0.996 per 76 s bin, 71 % of the
layer's activity arrived by coupling.  Coupling carries activity to torus
neighbours: brain → body (0.98) and smell (1.0) while their own sources are
silent.  That is reach, not input.  So the layer keeps both readings —
``afferent`` (what came back) and ``coupled`` (what the layer holds) — and
``QUIPU_SENSE_LAYER`` chooses which the Entirety reads.  Default ``afferent``:
in the 24 h closed-loop replay, feeding the coupled reading to the SOMN sent
978 documents to silent sources without routing and returned 7,313 documents
against 8,043 for the afferent reading (CHANGELOG 0.38.0).

3. Silent-source routing
------------------------
A terminal is silent from its first zero attempt after its last yield.  A
silent source is probed again only when

    now − last_attempt  ≥  K_s · (last_attempt − silence_start)

— exponential backoff with no constant: each probe it fails doubles the wait,
scaled by its read sharpness (a hot, rarely-read source is re-probed sooner).
When due it gets one document; otherwise nothing.  What it would have drawn
goes to the live planned sources in proportion to their planned documents —
current through an open junction redistributes over the remaining paths — or is
reported dissipated if none are live.  Budget and sources stay inside the plan,
so this is the constrained gate (Invariance #7 ruling, 2026-09-22).

Stdlib only.  mesh_slm.py and system_entirety.py untouched.

翈 — the layer holds what came back, as long as its slowest rhythm needs.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import annealed_senses as A

LAYER_ENV: str = "QUIPU_SENSE_LAYER"          # "afferent" (default) | "critical"
ROUTING_ENV: str = "QUIPU_SILENT_ROUTING"     # default on under the master flag
OBSERVER_KEY: str = "entirety:senses:observer"
OBSERVER_EVENTS_MAX: int = 200                 # the same retention corpus_ingest keeps for its history
LINEAGE: str = ("Hochstetter et al., 'Avalanches and edge-of-chaos learning in neuromorphic nanowire "
                "networks', Nat. Commun. 12, 4008 (2021); Iftakher et al., Nat. Commun. 17, 5246 (2026)")

_EIG_CACHE: dict[tuple, tuple] = {}


# ===========================================================================
# 1. Feeds
# ===========================================================================

def ingest_events(history: Any) -> list[tuple[float, str, float, str]]:
    return [(t, s, y, "ingest") for t, s, y in A.events_from_history(history)]


def mesh_events(cn: sqlite3.Connection, since: float | None) -> list[tuple[float, str, float, str]]:
    """Per-minute embedding mass written on each axis *above the mesh's own mean*.

    Every token carries some mass on every axis; what says the mesh moved on an
    axis is the touched tokens' mass beyond the whole mesh's mean profile,
    Σ max(0, e_k − μ_k), binned by minute of their latest touch.
    """
    try:
        rows = cn.execute(
            "SELECT v.last_seen, e.e_vision, e.e_touch, e.e_smell, e.e_body, e.e_brain, e.e_perception, "
            "e.e_entirety FROM mesh_slm_vocab v JOIN mesh_slm_embed e ON e.token_id = v.token_id").fetchall()
    except sqlite3.Error:
        return []
    n_ax = len(A.AXES)
    vals = []
    for r in rows:
        try:
            vals.append((r[0], [abs(float(r[1 + i] or 0.0)) for i in range(n_ax)]))
        except (TypeError, ValueError):
            continue
    if not vals:
        return []
    mu = [sum(v[1][i] for v in vals) / len(vals) for i in range(n_ax)]
    bins: dict[float, list[float]] = {}
    for ts, e in vals:
        t = A._epoch(ts)
        if t is None or (since is not None and t < since):
            continue
        b = math.floor(t / 60.0) * 60.0
        acc = bins.setdefault(b, [0.0] * n_ax)
        for i in range(n_ax):
            acc[i] += max(0.0, e[i] - mu[i])
    out = []
    for b, acc in sorted(bins.items()):
        for i, ax in enumerate(A.AXES):
            if acc[i] > 0.0:          # nothing above the mean is no motion, not a failed attempt
                out.append((b, f"mesh:{ax}", acc[i], "mesh"))
    return out


def _read_kv(cn: sqlite3.Connection, key: str, default=None):
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else default
    except (sqlite3.Error, json.JSONDecodeError, TypeError):
        return default


def observer_events(cn: sqlite3.Connection, *, write: bool = True) -> list[tuple[float, str, float, str]]:
    """Deltas of observer_service's cumulative token counters, as events.

    Writes ``entirety:senses:observer`` only when a counter moved.
    """
    try:
        rows = cn.execute("SELECT key, value FROM brain_kv WHERE key LIKE 'observer:%:stats'").fetchall()
    except sqlite3.Error:
        rows = []
    rec = _read_kv(cn, OBSERVER_KEY, None) or {}
    counters = dict(rec.get("counters") or {})
    events = [list(e) for e in (rec.get("events") or []) if isinstance(e, (list, tuple)) and len(e) >= 3]
    moved = False
    for key, raw in rows:
        src = str(key)[len("observer:"):-len(":stats")]
        try:
            st = json.loads(raw)
            tok = float(st.get("tokens") or 0.0)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            continue
        t = A._epoch(st.get("last_seen")) or time.time()
        prev = counters.get(src)
        if prev is None:
            counters[src] = {"tokens": tok, "at": t}             # first sight: a baseline
            moved = True
            continue
        d = tok - float(prev.get("tokens") or 0.0)
        if d > 0.0 and t > float(prev.get("at") or 0.0):
            events.append([t, src, d])
            counters[src] = {"tokens": tok, "at": t}
            moved = True
        elif d < 0.0:                                            # counter reset: new baseline
            counters[src] = {"tokens": tok, "at": t}
            moved = True
    events = events[-OBSERVER_EVENTS_MAX:]
    if moved and write:
        try:
            cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
            cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
                       (OBSERVER_KEY, json.dumps({"counters": counters, "events": events}),
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            cn.commit()
        except sqlite3.Error:
            pass
    return [(float(e[0]), f"observer:{e[1]}", float(e[2]), "observer") for e in events]


# ===========================================================================
# 2. The coupled layer
# ===========================================================================

def jacobi_eig(m: Sequence[Sequence[float]], sweeps: int = 60) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and orthonormal eigenvectors (columns) of a symmetric matrix."""
    n = len(m)
    a = [list(map(float, row)) for row in m]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-22:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


def torus_angle(axis: str | None) -> float | None:
    if axis not in A.AXES:
        return None
    return 2.0 * math.pi * A.AXES.index(axis) / len(A.AXES)


def coupling(axes: Sequence[str | None]) -> list[list[float]]:
    th = [torus_angle(a) for a in axes]
    n = len(axes)
    return [[(max(0.0, math.cos(th[i] - th[j])) if (i != j and th[i] is not None and th[j] is not None) else 0.0)
             for j in range(n)] for i in range(n)]


def _lam_max(decay: Sequence[float], w: Sequence[Sequence[float]], g: float) -> float:
    n = len(decay)
    m = [[(-decay[i] if i == j else 0.0) + g * w[i][j] for j in range(n)] for i in range(n)]
    return max(jacobi_eig(m)[0])


def edge_gain(decay: Sequence[float], w: Sequence[Sequence[float]], target: float) -> float:
    """g ≥ 0 with λ_max(−diag(decay) + g·W) = target (0 if already at or above it)."""
    if not decay:
        return 0.0
    if _lam_max(decay, w, 0.0) >= target or not any(any(row) for row in w):
        return 0.0
    lo, hi = 0.0, max(decay)
    while _lam_max(decay, w, hi) < target:
        hi *= 2.0
        if hi > 1e12:
            return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _lam_max(decay, w, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


GRID_OCTAVES: int = 12      # the gain is searched over 2^-12 … 1 of the edge gain (resolution, not a setting)


def _replay(names, vecs, lam, trace, now, idx):
    n = len(names)

    def propagate(a, dt):
        if dt <= 0.0:
            return a
        proj = [sum(vecs[k][j] * a[k] for k in range(n)) for j in range(n)]
        proj = [proj[j] * math.exp(lam[j] * dt) for j in range(n)]
        return [min(1.0, max(0.0, sum(vecs[i][j] * proj[j] for j in range(n)))) for i in range(n)]

    a = [0.0] * n
    t_prev = None
    samples: list[float] = []
    for t, s, x in trace:
        if t > now or s not in idx:
            continue
        if t_prev is not None and t != t_prev:
            a = propagate(a, t - t_prev)
            samples.append(sum(a) / n)
        i = idx[s]
        a[i] = a[i] + x * (1.0 - a[i])
        t_prev = t
    if t_prev is not None:
        a = propagate(a, max(0.0, now - t_prev))
        samples.append(sum(a) / n)
    return a, samples


def _variance(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def _entropy(xs: Sequence[float]) -> float:
    """Shannon entropy (bits) of the layer's mean activity over [0, 1], √n bins.

    Maximal when the layer uses its whole range; a layer stuck low (ordered)
    or pinned high (saturated) scores low.  Variance does not separate these:
    it rewards a layer that only swings between empty and saturated.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    k = max(2, int(math.isqrt(n)))
    counts = [0] * k
    for x in xs:
        counts[min(k - 1, max(0, int(x * k)))] += 1
    return -sum((c / n) * math.log2(c / n) for c in counts if c)


def critical_layer(terms: Mapping[str, A.Terminal], trace: Sequence[tuple[float, str, float]],
                   now: float) -> dict:
    """Replay the attempts through the coupled network at its critical gain.

    The critical gain is *measured*: the layer's response entropy H(g) — the
    Shannon entropy of its mean activity over the record — is computed on a
    geometric grid below the stability edge g_edge (λ_max = −1/T, T the
    record's span) and refined by golden-section search on log g; g* = argmax H.
    Below g* the layer only follows its input (ordered); toward the edge it
    saturates and stops responding.  Maximal response entropy is maximal
    dynamic range, the sensing signature of the critical point.
    """
    names = sorted(terms)
    empty = {"units": names, "activity": {s: 0.0 for s in names}, "g": 0.0, "g_edge": 0.0, "g_over_edge": 0.0,
             "lambda_max": None, "T": None, "sigma": None, "bin_s": None, "entropy_bits": 0.0,
             "entropy_curve": [], "coupling_share": 0.0}
    if not names:
        return empty
    ts_all = [t for t, _, _ in trace if t <= now]
    T = (now - min(ts_all)) if ts_all else None          # the record's own span
    if not T:
        return empty
    tau = [min(max(terms[s].tau, 1e-9), T) if terms[s].tau > 0 else T for s in names]
    decay = [1.0 / t for t in tau]
    axes = [terms[s].axis for s in names]
    w = coupling(axes)
    n = len(names)
    idx = {s: i for i, s in enumerate(names)}
    t_last = max(ts_all)

    def run(g: float, at: float):
        m = [[(-decay[i] if i == j else 0.0) + g * w[i][j] for j in range(n)] for i in range(n)]
        lam, vecs = jacobi_eig(m)
        a, samples = _replay(names, vecs, lam, trace, at, idx)
        return a, _entropy(samples), max(lam)

    # The critical gain depends on the record, not on the moment it is read:
    # cached until a new attempt arrives or a clock changes.
    key = (tuple(names), tuple(round(t, 3) for t in tau), tuple(axes), len(trace), round(t_last, 3))
    if key in _EIG_CACHE:
        g_edge, g, chi_curve = _EIG_CACHE[key]
    else:
        g_edge = edge_gain(decay, w, -1.0 / T)
        grid = [0.0] + ([g_edge * 2.0 ** (-k) for k in range(GRID_OCTAVES, -1, -1)] if g_edge > 0 else [])
        chi_curve = [(gg, run(gg, t_last)[1]) for gg in grid]
        k_best = max(range(len(chi_curve)), key=lambda k: chi_curve[k][1])
        g = chi_curve[k_best][0]
        if g > 0.0:
            # golden-section refinement on log g between the peak's grid neighbours
            lo = math.log(chi_curve[k_best - 1][0]) if k_best > 1 else math.log(g) - math.log(2.0)
            hi = math.log(chi_curve[k_best + 1][0]) if k_best + 1 < len(chi_curve) else math.log(g)
            phi = (math.sqrt(5.0) - 1.0) / 2.0
            c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
            fc, fd = run(math.exp(c), t_last)[1], run(math.exp(d), t_last)[1]
            for _ in range(12):
                if fc > fd:
                    hi, d, fd = d, c, fc
                    c = hi - phi * (hi - lo)
                    fc = run(math.exp(c), t_last)[1]
                else:
                    lo, c, fc = c, d, fd
                    d = lo + phi * (hi - lo)
                    fd = run(math.exp(d), t_last)[1]
            g_ref = math.exp(0.5 * (lo + hi))
            if run(g_ref, t_last)[1] >= chi_curve[k_best][1]:
                g = g_ref
        if len(_EIG_CACHE) > 16:
            _EIG_CACHE.clear()
        _EIG_CACHE[key] = (g_edge, g, chi_curve)
    a, chi, lam_max = run(g, now)
    act = {s: a[idx[s]] for s in names}
    aff = sum(terms[s].activity for s in names)
    tot = sum(act.values())
    ingest_clock = A._local_clock(sorted({t for t, s, _ in trace if terms.get(s) and terms[s].feed == "ingest"}))
    out = {
        "units": names, "activity": act, "g": g, "g_edge": g_edge, "g_over_edge": (g / g_edge if g_edge else 0.0),
        "lambda_max": lam_max, "T": T, "entropy_bits": chi,
        "entropy_curve": [(round(gg / g_edge, 6) if g_edge else 0.0, round(c, 6)) for gg, c in chi_curve],
        "sigma": (math.exp(lam_max * ingest_clock) if (lam_max is not None and ingest_clock) else None),
        "bin_s": ingest_clock,
        "coupling_share": (max(0.0, tot - aff) / tot) if tot > 0 else 0.0,
    }
    return out


# ===========================================================================
# 3. Silent-source routing
# ===========================================================================

def probe_due(t: A.Terminal, now: float) -> bool:
    """now − last ≥ K·(last − silence_start): backoff doubles with every failed probe."""
    if t.silence_start is None:
        return True
    return (now - t.last_t) >= t.k * max(0.0, t.last_t - t.silence_start)


def route_around_silence(docs: Mapping[str, int], terms: Mapping[str, A.Terminal], now: float) -> dict:
    """Move a plan's documents off silent sources, keeping one probe for those due."""
    from .memristive_axes import _largest_remainder
    plan = {k: int(v) for k, v in docs.items() if int(v) > 0}
    silent = {k for k in plan if k in terms and terms[k].silent}
    probes = sorted(k for k in silent if probe_due(terms[k], now))
    held = sorted(silent - set(probes))
    out = {k: v for k, v in plan.items() if k not in silent}
    pool = sum(plan[k] for k in held) + sum(plan[k] - 1 for k in probes)
    for k in probes:
        out[k] = 1
    live = {k: v for k, v in out.items() if k not in silent and v > 0}
    dissipated = 0
    if pool > 0 and live:
        extra = _largest_remainder({k: float(v) for k, v in live.items()}, pool)
        for k, v in extra.items():
            out[k] += v
    elif pool > 0:
        dissipated = pool
    return {
        "docs_per_source": {k: v for k, v in out.items() if v > 0},
        "silent": sorted(silent), "probed": probes, "held": held,
        "rerouted": pool - dissipated, "dissipated": dissipated,
        "next_probe_at": {k: terms[k].last_t + terms[k].k * max(0.0, terms[k].last_t - terms[k].silence_start)
                          for k in held},
    }


def routing_on() -> bool:
    return os.environ.get(ROUTING_ENV, "1").strip() != "0"


# ===========================================================================
# The read
# ===========================================================================

def layer_mode() -> str:
    m = os.environ.get(LAYER_ENV, "afferent").strip().lower()
    return m if m in ("afferent", "critical") else "afferent"


def _conn() -> sqlite3.Connection:
    # The same connection brain_kv uses (looked up at call time), so the layer
    # reads and writes exactly the store the rest of the Entirety does.
    from .. import brain_kv
    return brain_kv.open_conn()


def read(*, history: list | None = None, now: float | None = None,
         route: Callable[[str], int | None] | None = None,
         extra: Mapping[str, Iterable[float]] | None = None,
         mesh: list | None = None, observer: list | None = None,
         cn: sqlite3.Connection | None = None, write: bool = True) -> dict:
    """Every feed, the afferent terminals, the coupled layer, and both sense readings.

    Explicit ``history`` / ``mesh`` / ``observer`` event lists bypass the database
    (tests, replays); otherwise they are read, and observer counters may be written.
    """
    now = time.time() if now is None else float(now)
    own = None
    try:
        if history is None:
            history = A._history()
        ev = ingest_events(history)
        since = min((e[0] for e in ev), default=None)
        if mesh is None or observer is None:
            own = cn or _conn()
        mev = mesh if mesh is not None else mesh_events(own, since)
        oev = observer if observer is not None else observer_events(own, write=write)
    finally:
        if own is not None and cn is None:
            own.close()
    events = list(ev) + [tuple(e) for e in mev] + [tuple(e) for e in oev]
    trace: list = []
    terms = A.terminals(events, now, route, trace=trace)
    ex = A.live_writers() if extra is None else extra
    afferent = A.ports(terms, ex)
    crit = critical_layer(terms, trace, now)
    coupled_terms = {s: A.Terminal(**{**t.__dict__, "activity": crit["activity"].get(s, 0.0)})
                     for s, t in terms.items()}
    coupled = A.ports(coupled_terms, ex)
    mode = layer_mode()
    return {
        "at": now, "mode": mode,
        "senses": coupled if mode == "critical" else afferent,
        "afferent": afferent, "coupled": coupled,
        "critical": {k: v for k, v in crit.items() if k != "activity"} | {
            "activity": {s: round(v, 6) for s, v in crit["activity"].items()}},
        "terminals": {s: t.to_json() for s, t in sorted(terms.items())},
        "feeds": {f: sorted(s for s, t in terms.items() if t.feed == f) for f in ("ingest", "mesh", "observer")},
        "silent": sorted(s for s, t in terms.items() if t.silent and t.feed == "ingest"),
        "entirety_terminals": sorted(s for s, t in terms.items() if t.axis == "entirety"),
        "unrouted": sorted(s for s, t in terms.items() if t.axis is None),
        "live_writers": {k: list(v) for k, v in ex.items()},
        "events": len(events),
        "lineage": LINEAGE,
        "_terms": terms,
    }


_MEMO: dict[str, Any] = {}


def sense_signals() -> dict:
    """The six senses for the Entirety.  One step reads the senses several times
    within the same second; those reads share one computation."""
    history = A._history()
    key = (int(time.time()), layer_mode(), len(history), json.dumps(history[-1:], sort_keys=True, default=str))
    if _MEMO.get("key") == key:
        return dict(_MEMO["senses"])
    senses = {s: float(v) for s, v in read(history=history)["senses"].items()}
    _MEMO.update(key=key, senses=senses)
    return dict(senses)


def public(r: Mapping) -> dict:
    """A read without its in-memory terminal objects (JSON-safe)."""
    return {k: v for k, v in r.items() if not k.startswith("_")}


def plan_with_routing(docs: Mapping[str, int], *, now: float | None = None) -> dict:
    """The SOMN's planned documents, routed around silent sources."""
    now = time.time() if now is None else float(now)
    r = read(now=now, write=False)
    return route_around_silence(docs, r["_terms"], now)


def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="sensing_layer", description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = read(write=False)
    if args.json:
        print(json.dumps(public(r), indent=2, default=str))
        return 0
    c = r["critical"]
    print(f"mode {r['mode']}   events {r['events']}")
    print(f"{'sense':<11}{'afferent':>10}{'coupled':>10}")
    for s in A.SENSES:
        print(f"{s:<11}{r['afferent'][s]:>10.4f}{r['coupled'][s]:>10.4f}")
    if c.get("lambda_max") is not None:
        print(f"\ncritical layer: g/g_edge {c['g_over_edge']:.3f}  lambda_max {c['lambda_max']:.3e}/s  "
              f"sigma(bin {c['bin_s'] or 0:.0f} s) {c['sigma'] or 0:.4f}  coupling share {c['coupling_share']:.3f}")
    print(f"\n{'terminal':<18}{'feed':<9}{'axis':<11}{'n':>5}{'tau s':>9}{'K':>7}{'aff':>8}{'crit':>8}")
    for s, t in r["terminals"].items():
        print(f"{s:<18}{t['feed']:<9}{str(t['axis']):<11}{t['attempts']:>5}{t['tau']:>9.0f}{t['k']:>7.2f}"
              f"{t['activity']:>8.4f}{c['activity'].get(s, 0.0):>8.4f}")
    if r["silent"]:
        print("silent:", ", ".join(r["silent"]))
    try:
        from .self_organising import plan
        p = plan()
        print("\nplan (routed around silence):", json.dumps(p.get("docs_per_source")),
              json.dumps({k: p.get("silence", {}).get(k) for k in ("probed", "held", "rerouted")}))
    except Exception as exc:
        print("plan unavailable:", exc)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "LAYER_ENV", "ROUTING_ENV", "OBSERVER_KEY", "LINEAGE", "ingest_events", "mesh_events", "observer_events",
    "jacobi_eig", "torus_angle", "coupling", "edge_gain", "critical_layer", "probe_due",
    "route_around_silence", "routing_on", "layer_mode", "read", "sense_signals", "public", "plan_with_routing",
]
