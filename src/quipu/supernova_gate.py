"""Supernova expansion gate — the TheSeuScape/Perceptopoly event interface.

TheSeuScape NPCs (Perceptopoly) make decisions for *local expansion* — they
expand their maxima — and emit **supernova events** when they do.  Those
systems live outside this repository, so QUIPU defines them as a typed
event contract rather than assumed code:

    POST /observe  {"source": ..., "text": ..., "meta": {"supernova": {
        "source": "theseuscape",      # emitting system (default)
        "magnitude": 0.0..1.0,        # expansion strength
        "direction": 0..2π,           # bifurcation direction (radians)
        "at": "<ISO-8601 UTC>"        # defaults to now
    }}}

Events are logged under ``brain_kv["perceptopoly:supernova_events"]`` and
echoed on ``GET /guidance``.

The gate itself combines **three choices** into a bifurcated direction by
intersecting the two optimisers the repo already carries:

* the *realised* gradient ``g_re`` — the fleet's own expansion decision
  (``broaden`` phase from :mod:`system_entirety` / :mod:`mesh_entirety`) plus
  the liveness of the continuous synaptic agents;
* the *latent* gradient ``g_im`` — the external supernova drive;
* ``radam_step(g_re, g_im, ...)`` (:mod:`radam_optimizer`) turns the complex
  gradient into a toroidal pressure with the supernova direction as external
  phase — the **rADAM** side;
* ``actionable_potential`` (:mod:`recursive_strengthening`, the ADAM over
  the recursive potential edge) — the **poADAM** side;
* ``gate = clip01(radam_pressure × actionable_potential)`` is their
  intersection.  It is 0 whenever there is no expansion decision, so the
  viscous-decay knot field in :mod:`mesh_slm` strengthens nothing unless
  someone is actually expanding.

Public API
----------
* :func:`record_supernova_event`   — validate + log one event
* :func:`recent_supernova_drive`   — decayed magnitude + circular-mean direction
* :func:`expansion_gate`           — rate-limited driver, persists to brain_kv
* :func:`get_expansion_gate`       — single scalar accessor ∈ [0, 1]
* :func:`get_expansion_gate_state` — full persisted gate dict
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from .local_store import db_path as _local_db_path
from .radam_optimizer import radam_step
from .ueqgm_engine import _SUPERNOVA_GATE_KEY, _SUPERNOVA_KEY

_LOCK = threading.Lock()
_LAST_TS = 0.0
_MIN_SECONDS = 30.0
_EVENT_CAP = 200
_RADAM_STATE_KEY = "supernova_gate:radam"
_DEFAULT_SOURCE = "theseuscape"
# Weights of the realised gradient: the fleet's own broaden decision carries
# 0.6, the synaptic agents' liveness 0.4.
_W_BROADEN = 0.6
_W_SYNAPTIC = 0.4
# Event magnitude decays with age inside the drive window: e^{−age/τ}.
_DRIVE_TAU_S = 1800.0
# Synaptic worker heartbeat keys and their nominal periods (seconds), mirroring
# self_realization_loop._detect_worker_gaps; a worker is live when its
# heartbeat is fresher than 2× its period.
_SYNAPTIC_PERIODS = {
    "synapse_builder_last": 600,
    "synapse_lookahead_7d_last": 900,
    "synapse_convergence_last": 1800,
    "synapse_ueqgm_last": 420,
}


@contextmanager
def _conn():
    cn = sqlite3.connect(_local_db_path())
    cn.row_factory = sqlite3.Row
    try:
        yield cn
        cn.commit()
    finally:
        cn.close()


def _kv_read(cn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    cn.execute(
        "CREATE TABLE IF NOT EXISTS brain_kv("
        "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
    )
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return default
    if not row or row[0] in (None, ""):
        return default
    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return row[0]


def _kv_write(cn: sqlite3.Connection, key: str, val: Any) -> None:
    cn.execute(
        "CREATE TABLE IF NOT EXISTS brain_kv("
        "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
    )
    cn.execute(
        "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
        (key, json.dumps(val, default=str), datetime.now(timezone.utc).isoformat()),
    )


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _parse_ts(raw: Any) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

def record_supernova_event(event: dict | None) -> dict:
    """Validate and append one supernova event; return the stored record.

    Raises ``ValueError`` for a non-dict or non-numeric magnitude so the HTTP
    layer can answer 400.  Magnitude is clipped to [0, 1], direction wrapped
    to [0, 2π).
    """
    if not isinstance(event, dict):
        raise ValueError("supernova event must be an object")
    try:
        magnitude = _clip01(event.get("magnitude", 0.0))
        direction = float(event.get("direction", 0.0)) % (2.0 * math.pi)
    except (TypeError, ValueError) as exc:
        raise ValueError("supernova magnitude/direction must be numeric") from exc
    at = str(event.get("at") or datetime.now(timezone.utc).isoformat())
    if _parse_ts(at) is None:
        at = datetime.now(timezone.utc).isoformat()
    record = {
        "source": str(event.get("source") or _DEFAULT_SOURCE).strip().lower(),
        "magnitude": round(magnitude, 6),
        "direction": round(direction, 6),
        "at": at,
        "label": str(event.get("label") or "")[:120],
    }
    with _conn() as cn:
        events = _kv_read(cn, _SUPERNOVA_KEY, default=[]) or []
        if not isinstance(events, list):
            events = []
        events.append(record)
        _kv_write(cn, _SUPERNOVA_KEY, events[-_EVENT_CAP:])
    return record


def get_supernova_events(limit: int = 50) -> list[dict]:
    with _conn() as cn:
        events = _kv_read(cn, _SUPERNOVA_KEY, default=[]) or []
    if not isinstance(events, list):
        return []
    return events[-max(1, int(limit)):]


def recent_supernova_drive(
    window_s: float = 3600.0, *, now: float | None = None
) -> tuple[float, float, int]:
    """Age-decayed mean magnitude, circular-mean direction, and event count.

    Only events inside ``window_s`` count; each is weighted ``e^{−age/τ}``.
    Returns ``(0.0, 0.0, 0)`` when nothing is in the window.
    """
    now_ts = time.time() if now is None else float(now)
    mag_sum = w_sum = 0.0
    cx = cy = 0.0
    n = 0
    for ev in get_supernova_events(_EVENT_CAP):
        ts = _parse_ts(ev.get("at"))
        if ts is None:
            continue
        age = now_ts - ts
        # Tolerate small clock skew between the emitter and this host; an event
        # stamped a moment "in the future" counts as fresh, not as absent.
        if age < -60.0 or age > window_s:
            continue
        age = max(0.0, age)
        w = math.exp(-age / _DRIVE_TAU_S)
        m = _clip01(ev.get("magnitude", 0.0))
        d = float(ev.get("direction", 0.0))
        mag_sum += w * m
        w_sum += w
        cx += w * m * math.cos(d)
        cy += w * m * math.sin(d)
        n += 1
    if n == 0 or w_sum <= 0.0:
        return 0.0, 0.0, 0
    phase = math.atan2(cy, cx) % (2.0 * math.pi) if (cx or cy) else 0.0
    return _clip01(mag_sum / w_sum), phase, n


# ---------------------------------------------------------------------------
# The three choices → realised / latent gradient
# ---------------------------------------------------------------------------

def _broaden_decision() -> float:
    """1.0 when either the entirety or the mesh expansion phase is ``broaden``."""
    score = 0.0
    try:
        from .system_entirety import get_expansion_phase as _ep
        if _ep("deepen") == "broaden":
            score = 1.0
    except Exception:
        pass
    try:
        from .mesh_entirety import get_mesh_expansion_phase as _mp
        if _mp("deepen") == "broaden":
            score = max(score, 1.0)
    except Exception:
        pass
    return score


def _synaptic_liveness(cn: sqlite3.Connection, *, now: float | None = None) -> float:
    """Fraction of the synaptic agents whose heartbeat is fresher than 2× its period."""
    now_ts = time.time() if now is None else float(now)
    live = total = 0
    for key, period in _SYNAPTIC_PERIODS.items():
        raw = _kv_read(cn, key, default=None)
        if raw in (None, ""):
            continue
        total += 1
        ts = _parse_ts(str(raw).split("|", 1)[0])
        if ts is not None and (now_ts - ts) <= 2.0 * period:
            live += 1
    return (live / total) if total else 0.0


def _actionable_potential() -> float:
    try:
        from .recursive_strengthening import get_actionable_potential as _ap
        return _clip01(_ap(0.0))
    except Exception:
        return 0.0


def _rhythm_coherence() -> float:
    try:
        from .temporal_spatiality import get_rhythm_factor as _rf
        return _clip01(_rf("coherence", 1.0))
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Driver — rate-limited; persists to brain_kv
# ---------------------------------------------------------------------------

def expansion_gate(*, force: bool = False, window_s: float = 3600.0) -> dict:
    """Intersect rADAM (bifurcated gradient) with the potential ADAM edge.

    ``g_re = 0.6·broaden + 0.4·synaptic_liveness``, ``g_im = supernova
    magnitude``; ``radam_step`` with the supernova direction as external phase
    and the rhythm coherence scaling Langevin noise (σ = 0 keeps it
    deterministic); ``gate = clip01(pressure × actionable_potential)``.
    """
    global _LAST_TS
    with _LOCK:
        if not force and time.monotonic() - _LAST_TS < _MIN_SECONDS:
            return {"skipped": True, "reason": "rate-limited"}
        _LAST_TS = time.monotonic()

    broaden = _broaden_decision()
    magnitude, phase, n_events = recent_supernova_drive(window_s)
    potential = _actionable_potential()
    coherence = _rhythm_coherence()

    with _conn() as cn:
        liveness = _synaptic_liveness(cn)
        g_re = _W_BROADEN * broaden + _W_SYNAPTIC * liveness
        g_im = magnitude
        state = _kv_read(cn, _RADAM_STATE_KEY, default={}) or {}
        if not isinstance(state, dict):
            state = {}
        pressure = radam_step(
            state,
            g_re,
            g_im,
            coherence=coherence,
            external_phase=phase,
            use_torus=(g_im > 0.0),
        )
        gate = _clip01(pressure * potential)
        rec = {
            "gate": round(gate, 6),
            "radam_pressure": round(pressure, 6),
            "actionable_potential": round(potential, 6),
            "g_re": round(g_re, 6),
            "g_im": round(g_im, 6),
            "broaden": broaden,
            "synaptic_liveness": round(liveness, 6),
            "supernova_phase": round(phase, 6),
            "theta": round(float(state.get("theta", 0.0)), 6),
            "events_in_window": n_events,
            "coherence": round(coherence, 6),
            "ran_at": datetime.now(timezone.utc).isoformat(),
        }
        _kv_write(cn, _RADAM_STATE_KEY, state)
        _kv_write(cn, _SUPERNOVA_GATE_KEY, rec)
    return rec


# ---------------------------------------------------------------------------
# Public read accessors
# ---------------------------------------------------------------------------

def get_expansion_gate_state() -> dict:
    with _conn() as cn:
        rec = _kv_read(cn, _SUPERNOVA_GATE_KEY, default={}) or {}
    return rec if isinstance(rec, dict) else {}


def get_expansion_gate(default: float = 0.0) -> float:
    try:
        return _clip01(get_expansion_gate_state().get("gate", default))
    except Exception:
        return default


__all__ = [
    "record_supernova_event",
    "get_supernova_events",
    "recent_supernova_drive",
    "expansion_gate",
    "get_expansion_gate",
    "get_expansion_gate_state",
]
