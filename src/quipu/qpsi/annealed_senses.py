"""annealed_senses — the six sense axes read by self-annealing memristive terminals.

Why
---
``temporal_spatiality._sense_signals`` read the senses through static sets:

    vision      total documents of *every* source / 1500 (a fixed scale), so
                arXiv's throughput lit vision while vision's own sources
                (fineweb, c4, openwebtext) returned nothing
    touch       share of fixed source names ("stack", "code", "github")
    brain       ingest *runs* per hour / 6 — the pulse counting itself.  With
                the operator's pulse at ~5 min, brain sat at 1.0, the SOMN read
                a large |δv| there, potentiated it to a filament and sent 59 of
                60 documents to arXiv: a self-excited lock
    smell, body tables nothing in this repo writes, read through defaults
    weights     a designer table (_SENSE_WEIGHTS) — left to qpsi.learned_prior

Direction taken (the physics, not a metaphor)
---------------------------------------------
1.  **Every source is a terminal; the set is whatever the history contains.**
    A source joins the moment it appears in ``corpus_ingest:history`` and
    fades when it stops yielding.  Where its documents land is where
    ``mesh_slm._axis_for_source`` routes them (read, never edited) — the sense
    reports the axis the data actually writes to.
2.  **Volatile (diffusive) memristor per terminal.**  Each yield potentiates a
    state a ∈ [0, 1] by saturating growth, a ← a + x(1 − a); between events it
    relaxes, a ← a·exp(−Δt/τ_s).  τ_s is the terminal's *own* present clock,
    the median of its last three gaps — no window, no fixed scale.  At rest
    every sense fades at its own rhythm.
3.  **p-bit read with a self-set threshold.**  The stimulus of an attempt is
        x = σ(K_s · (ln y − ln ȳ_s)) = y^K / (y^K + ȳ_s^K)
    the sigmoid switching law of a stochastic MTJ, P = 1/(1 + e^(−K(V − V₀.₅))),
    with V = ln y and the equiprobable point V₀.₅ = ln ȳ, the fabric's mean
    yield per attempt over every terminal, itself volatile on the fabric's
    clock (ȳ ← ȳ·r + y(1 − r), r = e^(−Δt/τ_fabric)).  A source returning the
    fabric's typical amount reads 0.5, more reads higher, less lower, nothing
    reads 0.  (y/(y+ȳ) is exactly the K = 1 case: divisive normalisation by
    the population is this sigmoid.)  The threshold is the population's and
    not the terminal's own history, so a sense still encodes *amount*: a
    source that adapted to its own level would read 0.5 forever and the SOMN
    field on its axis could never close.
4.  **Intrinsic annealing — the read sets the temperature.**  In the hybrid
    memristor–MTJ Ising machine (Iftakher et al., Nat. Commun. 17, 5246, 2026)
    the read voltage that interrogates the crossbar also biases the p-bits:
    reading harder lowers the effective temperature, with no external
    schedule.  Here the read is the pulse: K_s = ρ_s / ρ̄, the terminal's
    attempt rate over the geometric mean rate of all terminals.  A source read
    often discriminates sharply (cold); one read rarely reads soft (hot).  The
    pulse therefore shapes *how* a sense reads and is never itself a stimulus
    — efference is kept out of afference.
5.  **Ports.**  A sense axis is the probabilistic OR of the terminals routed to
    it, A_k = 1 − Π_s (1 − a_s): bounded without a scale, one strong terminal
    suffices, independent weak ones accumulate.  Terminals routed to the
    entirety axis and unrouted ones are reported, not folded into a sense.
6.  **Live writers still speak.**  If ``perception.get_perception_coherence``
    exists, or ``sense_of_smell`` / ``body_directives`` hold rows, their legacy
    reading joins the port as one more terminal.  Absent writers contribute
    nothing — no default stands in for silence.

What is not built here: criticality control (holding the terminal layer at the
edge of chaos, Hochstetter et al., Nat. Commun. 12, 4008, 2021) and learned
source→axis routing.  Routing stays the mesh's own, because that is where the
documents are written.

The six axis names remain.  They are the Entirety's axes (mesh_slm,
system_entirety), not a sense roster; everything that feeds them is dynamic.

Wiring
------
``enable()`` replaces the module attribute ``temporal_spatiality._sense_signals``
(every consumer imports it at call time) and ``disable()`` restores it.  A
failure inside the annealed read falls back to the original for that call and
logs it.  ``qpsi.self_organising.enable()`` calls ``enable()`` when
``QUIPU_ANNEALED_SENSES`` is not "0" (default on under the master flag).
This module writes nothing: it is a pure read of ``corpus_ingest:history``.

CLI:  python -m src.quipu.qpsi.annealed_senses [--json]
      static and annealed readings side by side, and every terminal.

Stdlib only.  mesh_slm.py and system_entirety.py untouched.

翈 — a sense is what came back, read at the sharpness the asking set.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

ENV: str = "QUIPU_ANNEALED_SENSES"
HISTORY_KEY: str = "corpus_ingest:history"
SENSES: tuple[str, ...] = ("vision", "touch", "smell", "body", "brain", "perception")
AXES: tuple[str, ...] = SENSES + ("entirety",)
MARK: str = "__qpsi_annealed_senses__"
LINEAGE: str = ("Iftakher et al., 'Intrinsic annealing in a hybrid memristor-magnetic tunnel junction "
                "Ising machine', Nat. Commun. 17, 5246 (2026); Caravelli et al., Nat. Rev. Phys. (2026), "
                "arXiv:2509.00747")

_LOG = logging.getLogger(__name__)
_ORIGINAL: Callable[[], dict] | None = None
_ENABLED = False


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def _epoch(ts: Any) -> float | None:
    if isinstance(ts, (int, float)) and math.isfinite(ts):
        return float(ts)
    try:
        dt = datetime.fromisoformat(str(ts or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def events_from_history(history: Any) -> list[tuple[float, str, float]]:
    """(t, source, yield) for every per-source attempt in the history, time-ordered.

    An attempt is a source named in an entry's ``per_source``; its yield is the
    documents it condensed (0 is an attempt that returned nothing).  Malformed
    entries are skipped.
    """
    out: list[tuple[float, str, float]] = []
    if not isinstance(history, list):
        return out
    for e in history:
        if not isinstance(e, Mapping):
            continue
        t = _epoch(e.get("ts"))
        per = e.get("per_source")
        if t is None or not isinstance(per, Mapping):
            continue
        for src, y in per.items():
            try:
                yv = float(y or 0.0)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(yv) or yv < 0.0:
                continue
            out.append((t, str(src), yv))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


# ---------------------------------------------------------------------------
# Terminal physics
# ---------------------------------------------------------------------------

def stimulus(y: float, level: float | None, k: float) -> float:
    """p-bit read: σ(K(ln y − ln ȳ)).  Nothing → 0; first yield of a silent source → 1."""
    if y <= 0.0:
        return 0.0
    if level is None or level <= 0.0:
        return 1.0
    if k <= 0.0:
        return 0.5
    z = k * (math.log(level) - math.log(y))
    if z > 700.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(z))


def _median_gap(ts: Sequence[float]) -> float | None:
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    return statistics.median(gaps) if gaps else None


def _local_clock(ts: Sequence[float]) -> float | None:
    """The terminal's present rhythm: median of its last three gaps.

    Three is the fewest gaps whose median rejects one outlier (a restart, a
    month's silence).  A whole-window median would keep a source on the clock
    of a regime it has left.
    """
    return _median_gap(list(ts[-4:]))


@dataclass
class Terminal:
    source: str
    axis: str | None          # sense / "entirety" / None (unrouted)
    attempts: int
    yielded: float            # documents over the window
    level: float              # fabric ȳ at the end of the replay (the next attempt's V₀.₅)
    tau: float                # own clock (s): median of its last three gaps
    tau_own: bool             # False → only one attempt; the fabric's clock stands in
    k: float                  # read sharpness (inverse temperature) = ρ_s / ρ̄
    last_x: float
    last_t: float
    activity: float           # a_s at `now`

    def to_json(self) -> dict:
        d = asdict(self)
        for key in ("yielded", "level", "tau", "k", "last_x", "activity"):
            d[key] = round(float(d[key]), 6)
        return d


def _axis_name(source: str, route: Callable[[str], int | None]) -> str | None:
    try:
        idx = route(source)
    except Exception:
        idx = None
    if idx is None or not (0 <= int(idx) < len(AXES)):
        return None
    return AXES[int(idx)]


def _default_route(source: str) -> int | None:
    from .memristive_axes import axis_for_source
    return axis_for_source(source)


def terminals(events: Sequence[tuple[float, str, float]], now: float,
              route: Callable[[str], int | None] | None = None) -> dict[str, Terminal]:
    """Replay every terminal's volatile state from the attempts up to ``now``.

    One time-ordered pass.  Each attempt is read against the *fabric* level
    V₀.₅ = ȳ, the volatile mean yield per attempt over every terminal, at the
    sharpness K_s = ρ_s/ρ̄ that the reading rhythm sets at that moment.  Then
    the fabric level and the terminal's own state advance.
    """
    route = route or _default_route
    ev = [e for e in events if e[0] <= now]
    if not ev:
        return {}

    seen: dict[str, list[float]] = {}
    state: dict[str, dict] = {}
    fabric_ts: list[float] = []
    level: float | None = None
    t_f: float | None = None

    for t, s, y in ev:
        seen.setdefault(s, []).append(t)
        if not fabric_ts or fabric_ts[-1] != t:
            fabric_ts.append(t)
        fabric_tau = _local_clock(fabric_ts)

        clocks = {src: _local_clock(ts) for src, ts in seen.items()}
        rates = {src: 1.0 / c for src, c in clocks.items() if c}
        rho_bar = math.exp(sum(math.log(r) for r in rates.values()) / len(rates)) if rates else None
        k = (rates[s] / rho_bar) if (s in rates and rho_bar) else 1.0
        tau = clocks[s] or fabric_tau

        st = state.setdefault(s, {"a": 0.0, "t": None, "n": 0, "total": 0.0, "x": 0.0, "k": 1.0, "tau": 0.0})
        if st["t"] is not None and tau:
            st["a"] *= math.exp(-(t - st["t"]) / tau)
        x = stimulus(y, level, k)
        st["a"] = st["a"] + x * (1.0 - st["a"])
        st.update(t=t, n=st["n"] + 1, total=st["total"] + y, x=x, k=k, tau=float(tau or 0.0))

        # The fabric threshold is volatile on the fabric's own clock: it forgets
        # what the sources used to return as fast as the fabric is read.
        if level is None:
            level = y
        else:
            r = math.exp(-(t - t_f) / fabric_tau) if (fabric_tau and t_f is not None) else 0.0
            level = level * r + y * (1.0 - r)
        t_f = t

    out: dict[str, Terminal] = {}
    for s, st in state.items():
        a = st["a"]
        if st["tau"] > 0.0:
            a *= math.exp(-max(0.0, now - st["t"]) / st["tau"])
        out[s] = Terminal(source=s, axis=_axis_name(s, route), attempts=st["n"], yielded=st["total"],
                          level=float(level or 0.0), tau=st["tau"], tau_own=bool(_local_clock(seen[s])),
                          k=st["k"], last_x=st["x"], last_t=float(st["t"]),
                          activity=min(1.0, max(0.0, a)))
    return out


def ports(terms: Mapping[str, Terminal], extra: Mapping[str, Iterable[float]] | None = None) -> dict[str, float]:
    """A_k = 1 − Π(1 − a_s) over the terminals routed to sense k (and any live writers)."""
    keep = {s: 1.0 for s in SENSES}
    for t in terms.values():
        if t.axis in keep:
            keep[t.axis] *= (1.0 - t.activity)
    for sense, vals in (extra or {}).items():
        if sense in keep:
            for v in vals:
                keep[sense] *= (1.0 - min(1.0, max(0.0, float(v))))
    return {s: 1.0 - keep[s] for s in SENSES}


# ---------------------------------------------------------------------------
# Live writers (legacy tables / perception module) — only if they exist
# ---------------------------------------------------------------------------

def _table_has_rows(cn: sqlite3.Connection, name: str) -> bool:
    try:
        if not cn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
            return False
        return cn.execute(f"SELECT 1 FROM {name} LIMIT 1").fetchone() is not None
    except sqlite3.Error:
        return False


def live_writers() -> dict[str, list[float]]:
    """Readings from writers that actually exist.  Nothing is defaulted."""
    extra: dict[str, list[float]] = {}
    try:
        from ..perception import get_perception_coherence   # type: ignore[import-not-found]
        extra.setdefault("perception", []).append(float(get_perception_coherence()))
    except Exception:
        pass
    try:
        from .. import temporal_spatiality as ts
        with ts._conn() as cn:
            if _table_has_rows(cn, "sense_of_smell"):
                m = ts._safe_scalar(cn, "SELECT carrier_mass FROM sense_of_smell ORDER BY id DESC LIMIT 1",
                                    default=0.5)
                extra.setdefault("smell", []).append(max(0.0, min(1.0, abs(m - 0.5) * 2.0)))
            if _table_has_rows(cn, "body_directives"):
                extra.setdefault("body", []).append(ts._safe_scalar(
                    cn, "SELECT MIN(1.0, CAST(COUNT(*) AS REAL) / 25.0) FROM body_directives "
                        "WHERE status IN ('open','ack','in_progress')"))
    except Exception:
        pass
    return extra


# ---------------------------------------------------------------------------
# The read
# ---------------------------------------------------------------------------

def _history() -> list:
    from .. import brain_kv
    h = brain_kv.kv_get_json(HISTORY_KEY, [])
    return h if isinstance(h, list) else []


def read(*, history: list | None = None, now: float | None = None,
         route: Callable[[str], int | None] | None = None,
         extra: Mapping[str, Iterable[float]] | None = None) -> dict:
    """Full reading: six senses, every terminal, and what fell outside the senses."""
    now = time.time() if now is None else float(now)
    hist = _history() if history is None else history
    ev = events_from_history(hist)
    terms = terminals(ev, now, route)
    ex = live_writers() if extra is None else extra
    senses = ports(terms, ex)
    return {
        "at": now,
        "senses": senses,
        "terminals": {s: t.to_json() for s, t in sorted(terms.items())},
        "entirety_terminals": sorted(s for s, t in terms.items() if t.axis == "entirety"),
        "unrouted": sorted(s for s, t in terms.items() if t.axis is None),
        "silent": sorted(s for s, t in terms.items() if t.yielded <= 0.0),
        "live_writers": {k: list(v) for k, v in ex.items()},
        "fabric_clock_s": _local_clock(sorted({t for t, _, _ in ev if t <= now})),
        "events": len(ev),
        "lineage": LINEAGE,
    }


def sense_signals() -> dict:
    """Drop-in for ``temporal_spatiality._sense_signals``: the six senses in [0, 1]."""
    return {s: float(v) for s, v in read()["senses"].items()}


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def flag_on() -> bool:
    return os.environ.get(ENV, "1").strip() != "0"


def enable() -> bool:
    """Replace temporal_spatiality._sense_signals.  Idempotent; falls back per call."""
    global _ORIGINAL, _ENABLED
    from .. import temporal_spatiality as ts
    cur = ts._sense_signals
    if getattr(cur, MARK, False):
        _ORIGINAL = getattr(cur, "__wrapped__", _ORIGINAL)
        _ENABLED = True
        return True
    original = cur

    def _sense_signals() -> dict:
        try:
            return sense_signals()
        except Exception as exc:          # a failed read never stops the Entirety
            _LOG.warning("[annealed_senses] read failed, static senses for this call: %s", exc)
            return original()

    _sense_signals.__doc__ = (original.__doc__ or "") + "\n\n[qpsi.annealed_senses] " + LINEAGE
    _sense_signals.__wrapped__ = original          # type: ignore[attr-defined]
    setattr(_sense_signals, MARK, True)
    ts._sense_signals = _sense_signals
    _ORIGINAL = original
    _ENABLED = True
    return True


def disable() -> None:
    global _ENABLED
    from .. import temporal_spatiality as ts
    if getattr(ts._sense_signals, MARK, False):
        ts._sense_signals = ts._sense_signals.__wrapped__
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def static_signals() -> dict:
    """The original static reading, whether or not the annealed one is wired."""
    from .. import temporal_spatiality as ts
    fn = ts._sense_signals
    return dict(getattr(fn, "__wrapped__", fn)())


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="annealed_senses", description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = read()
    try:
        old = static_signals()
    except Exception as exc:
        old = {"error": str(exc)}
    if args.json:
        print(json.dumps({"static": old, "annealed": r}, indent=2, default=str))
        return 0
    print(f"{'sense':<11}{'static':>9}{'annealed':>10}")
    for s in SENSES:
        o = old.get(s) if isinstance(old, dict) else None
        print(f"{s:<11}{(o if isinstance(o, float) else float('nan')):>9.4f}{r['senses'][s]:>10.4f}")
    print(f"\nfabric clock {r['fabric_clock_s']} s, {r['events']} attempts")
    print(f"{'terminal':<13}{'axis':<11}{'n':>4}{'docs':>7}{'ybar':>8}{'tau s':>9}{'K':>7}{'a':>8}")
    for s, t in r["terminals"].items():
        print(f"{s:<13}{str(t['axis']):<11}{t['attempts']:>4}{t['yielded']:>7.0f}{t['level']:>8.1f}"
              f"{t['tau']:>9.0f}{t['k']:>7.2f}{t['activity']:>8.4f}")
    if r["silent"]:
        print("silent (attempted, nothing returned):", ", ".join(r["silent"]))
    if r["unrouted"]:
        print("unrouted:", ", ".join(r["unrouted"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "ENV", "HISTORY_KEY", "SENSES", "AXES", "LINEAGE", "Terminal", "events_from_history", "stimulus",
    "terminals", "ports", "live_writers", "read", "sense_signals", "enable", "disable", "is_enabled",
    "flag_on", "static_signals",
]
