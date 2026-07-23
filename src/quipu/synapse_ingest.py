"""synapse_ingest — continuous corpus-ingest worker.

The five/six senses in :mod:`temporal_spatiality` read ``corpus_ingest:history``
through a **1-hour rolling window** (``_INGEST_WINDOW_HOURS = 1.0``).  Vision,
Touch and Brain are all fed from that window, so if no ``run_ingest()`` call
lands inside the last hour they collapse to ``0.0`` even while every other
synaptic daemon is alive.

Nothing in the mesh actually *drove* ingest continuously — ``run_ingest`` was
only reachable from the ``corpus_ingest`` CLI.  This worker closes that gap: it
runs a bounded ingest round on a **sub-hour cadence** and writes a
``synapse_ingest_last`` heartbeat in the same structured-JSON shape that
:mod:`self_realization_loop` already parses (``{"ran_at": ..., ...}``), so the
gap detector can track its liveness alongside the other ``synapse_*`` workers.

Public API
----------
* :func:`ingest_worker_round`  — run one bounded ingest + write heartbeat.
* :func:`run_forever`          — continuous loop (used by the daemon / CLI).
* :func:`ensure_scheduled`     — start the loop in a background thread once.
* :func:`heartbeat`            — read the current heartbeat payload.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Sequence

log = logging.getLogger(__name__)

# ── Heartbeat / cadence tunables ────────────────────────────────────────────
HEARTBEAT_KEY: str = "synapse_ingest_last"
# Cadence must stay comfortably inside temporal_spatiality._INGEST_WINDOW_HOURS
# (1.0h) so the vision/touch/brain window is never empty between runs.
DEFAULT_INTERVAL_S: float = 1200.0          # 20 minutes (3× per sense window)
_MIN_INTERVAL_S: float = 300.0              # hard floor between rounds (5 min)

# Bounded ingest defaults — keep each continuous round light; the CLI still
# exists for heavy manual/back-fill crawls.
DEFAULT_DOCS_PER_SOURCE: int = 40
DEFAULT_MAX_SECONDS: float = 240.0          # wall-clock cap per round

# ── Module state ────────────────────────────────────────────────────────────
_LAST_ROUND_MONO: float = 0.0
_THREAD: threading.Thread | None = None
_THREAD_LOCK = threading.Lock()
_STOP = threading.Event()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_heartbeat(payload: dict[str, Any]) -> None:
    """Persist the heartbeat in the JSON ``ran_at`` shape the loop parses."""
    try:
        from .brain_kv import kv_set
        kv_set(HEARTBEAT_KEY, json.dumps(payload, default=str))
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        log.warning("[synapse_ingest] heartbeat write failed: %s", exc)


def heartbeat() -> dict[str, Any]:
    """Return the last-written heartbeat payload (``{}`` if never run)."""
    try:
        from .brain_kv import kv_get_json
        val = kv_get_json(HEARTBEAT_KEY, {})
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def ingest_worker_round(
    sources: Sequence[str] | None = None,
    *,
    docs_per_source: int = DEFAULT_DOCS_PER_SOURCE,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    force: bool = False,
) -> dict[str, Any]:
    """Run one bounded ingest round and refresh the heartbeat.

    Rate-limited by ``_MIN_INTERVAL_S`` unless ``force`` is set.  Always writes
    a heartbeat (even on skip/error) so liveness telemetry stays fresh and the
    self-realization gap detector can distinguish "idle" from "dead".
    """
    global _LAST_ROUND_MONO

    now_mono = time.monotonic()
    if not force and _LAST_ROUND_MONO and (now_mono - _LAST_ROUND_MONO) < _MIN_INTERVAL_S:
        payload = {
            "ran_at": _now_iso(),
            "status": "skipped_rate_limited",
            "next_eligible_s": round(_MIN_INTERVAL_S - (now_mono - _LAST_ROUND_MONO), 1),
        }
        _write_heartbeat(payload)
        return payload

    started = time.monotonic()
    try:
        from . import corpus_ingest
        result = corpus_ingest.run_ingest(
            sources,
            docs_per_source=int(docs_per_source),
            max_seconds=float(max_seconds),
        )
        _LAST_ROUND_MONO = time.monotonic()
        payload = {
            "ran_at": _now_iso(),
            "status": "ok",
            "docs": int(result.get("total_condensed") or 0),
            "weyl_cycles": int(result.get("weyl_cycles") or 0),
            "tools_forged": int(result.get("tools_forged_total") or 0),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        _LAST_ROUND_MONO = time.monotonic()
        log.exception("[synapse_ingest] ingest round failed")
        payload = {
            "ran_at": _now_iso(),
            "status": "error",
            "error": str(exc)[:240],
            "elapsed_s": round(time.monotonic() - started, 3),
        }

    _write_heartbeat(payload)
    return payload


def run_forever(
    interval_s: float = DEFAULT_INTERVAL_S,
    *,
    sources: Sequence[str] | None = None,
    docs_per_source: int = DEFAULT_DOCS_PER_SOURCE,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> None:
    """Continuously run bounded ingest rounds every ``interval_s`` seconds."""
    interval_s = max(_MIN_INTERVAL_S, float(interval_s))
    log.info("[synapse_ingest] worker starting: interval=%ss docs/source=%s",
             interval_s, docs_per_source)
    _STOP.clear()
    while not _STOP.is_set():
        try:
            summary = ingest_worker_round(
                sources,
                docs_per_source=docs_per_source,
                max_seconds=max_seconds,
            )
            log.info("[synapse_ingest] round: %s", json.dumps(summary, default=str)[:200])
        except Exception:  # pragma: no cover - loop must never die
            log.exception("[synapse_ingest] unexpected round failure")
        _STOP.wait(interval_s)


def ensure_scheduled(
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    request_now: bool = False,
) -> threading.Thread:
    """Start the ingest loop in a daemon thread exactly once.

    Mirrors ``llm_self_train.ensure_scheduled`` so ``self_realization_loop`` can
    re-arm the worker when its heartbeat goes stale.  ``request_now`` fires an
    immediate bounded round (forced past the rate limiter) before/without
    waiting for the loop's first tick.
    """
    global _THREAD
    with _THREAD_LOCK:
        alive = _THREAD is not None and _THREAD.is_alive()
        if not alive:
            _THREAD = threading.Thread(
                target=run_forever,
                kwargs={"interval_s": interval_s},
                name="synapse_ingest",
                daemon=True,
            )
            _THREAD.start()
    if request_now:
        try:
            ingest_worker_round(force=True)
        except Exception:  # pragma: no cover
            log.exception("[synapse_ingest] request_now round failed")
    return _THREAD  # type: ignore[return-value]


def stop() -> None:
    """Signal the continuous loop to stop after its current wait."""
    _STOP.set()


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Continuous corpus-ingest synaptic worker.")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                    help="seconds between ingest rounds (min 300)")
    ap.add_argument("--docs", type=int, default=DEFAULT_DOCS_PER_SOURCE,
                    help="max documents per source per round")
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                    help="wall-clock cap per round")
    ap.add_argument("--once", action="store_true",
                    help="run a single bounded round and exit")
    ap.add_argument("--sources", default="",
                    help="comma list of source keys (default: corpus_ingest defaults)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sources = [s.strip() for s in args.sources.split(",") if s.strip()] or None

    if args.once:
        summary = ingest_worker_round(
            sources, docs_per_source=args.docs, max_seconds=args.max_seconds, force=True,
        )
        print(json.dumps(summary, indent=2, default=str))
        return

    try:
        run_forever(
            args.interval,
            sources=sources,
            docs_per_source=args.docs,
            max_seconds=args.max_seconds,
        )
    except KeyboardInterrupt:
        stop()
        print("\n[synapse_ingest] stopped.")


__all__ = [
    "HEARTBEAT_KEY",
    "DEFAULT_INTERVAL_S",
    "DEFAULT_DOCS_PER_SOURCE",
    "ingest_worker_round",
    "run_forever",
    "ensure_scheduled",
    "heartbeat",
    "stop",
]


if __name__ == "__main__":
    _cli()
