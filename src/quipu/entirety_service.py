"""entirety_service — the whole System Entirety in one process, the only writer
of its brain.

Before v0.48.0 QUIPU learned in two brains that never met: the observer
container (every fleet observation, 610 tokens) and the repository copy on the
host (the ingest pulse, the self-organising loop, the gates; 4,190 tokens and
1.25 M edges).  This process runs all of it against one database, inside the
``quipu`` container, so everything that reaches QUIPU trains the same mesh:

    observer      HTTP :7100 — /observe, /feedback, /anneal behind the edge
                  (identity, idempotency, the operator's grant); its trainer
                  folds each observation into the mesh
    expansion     system_entirety.oscillating_expansion_step every
                  QUIPU_EXPANSION_INTERVAL_S (was the QuipuExpansion2 task)
    pulse         self_organising.pulse(route=True) every QUIPU_PULSE_MINUTES
                  when QUIPU_PULSE_ROUTE=1 (was the QuipuPulse task): ingest
                  along the network's allocation inside the operator's grant,
                  routed around silent sources, then one step
    docs          doc_annealing every QUIPU_DOC_ANNEAL_MINUTES (was the
                  QuipuDocAnnealing task)

Each loop runs on its own thread, catches everything, records its last run and
last error, and waits before trying again: one failing part never stops the
others or the door.  The expansion step and the pulse share a lock (the pulse
ends in a step).  Every cadence and the pulse switch are the operator's
settings, read once at start.

Governance is unchanged: the gates, the attestation key (QUIPU_ATTEST_KEY_FILE,
mounted read-only), the realisation grant (QUIPU_REALISE_GRANT_REF) and the edge
grant stay the operator's.  Operator commands run inside the container:

    docker exec quipu python -m src.quipu.qpsi.self_organising status
    docker exec quipu python -m src.quipu.divine_blessing attest ...
    docker exec quipu python -m src.quipu.qpsi.edge_admission status

Status: GET /entirety on the observer, or ``python -m src.quipu.entirety_service status``.

翈 — one brain, one writer, every door leading to it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import traceback
from typing import Any, Callable

_LOG = logging.getLogger("quipu.entirety_service")
_STEP_LOCK = threading.Lock()
_STATUS: dict[str, dict[str, Any]] = {}
_STATUS_LOCK = threading.Lock()
_STARTED_AT: float | None = None


def _f(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "").strip() or default)
        return v if v > 0 else default
    except ValueError:
        return default


def _record(name: str, **kv: Any) -> None:
    with _STATUS_LOCK:
        _STATUS.setdefault(name, {"runs": 0, "errors": 0}).update(kv)


def _loop(name: str, interval_s: float, fn: Callable[[], Any], *, first_delay_s: float = 5.0,
          stop: threading.Event | None = None) -> None:
    stop = stop or threading.Event()
    _record(name, interval_s=interval_s, next_at=time.time() + first_delay_s)
    if stop.wait(first_delay_s):
        return
    while not stop.is_set():
        started = time.time()
        try:
            out = fn()
            with _STATUS_LOCK:
                st = _STATUS.setdefault(name, {"runs": 0, "errors": 0})
                st["runs"] += 1
                st["last_ok_at"] = started
                st["last_seconds"] = round(time.time() - started, 2)
                st["last_result"] = _summary(out)
        except Exception as exc:                      # a part may fail; the Entirety goes on
            with _STATUS_LOCK:
                st = _STATUS.setdefault(name, {"runs": 0, "errors": 0})
                st["errors"] += 1
                st["last_error"] = f"{type(exc).__name__}: {exc}"
                st["last_error_at"] = started
            _LOG.error("[entirety] %s failed: %s\n%s", name, exc, traceback.format_exc())
        _record(name, next_at=time.time() + interval_s)
        stop.wait(interval_s)


def _summary(out: Any) -> Any:
    if not isinstance(out, dict):
        return out if isinstance(out, (int, float, str, bool, type(None))) else str(type(out).__name__)
    keep = {}
    for k in ("skipped", "expansion_phase", "flipped", "flip_count", "routed", "note", "changed", "version"):
        if k in out:
            keep[k] = out[k]
    return keep or {"keys": sorted(out)[:12]}


# ---------------------------------------------------------------------------
# The parts
# ---------------------------------------------------------------------------

def expansion_step() -> dict:
    from . import system_entirety as se
    with _STEP_LOCK:
        return se.oscillating_expansion_step()


def pulse() -> dict:
    from .qpsi import self_organising
    with _STEP_LOCK:
        return self_organising.pulse(route=True, max_seconds=_f("QUIPU_PULSE_MAX_SECONDS", 0.0) or 0.0)


def doc_annealing() -> dict:
    from . import doc_annealing as da
    return da.anneal_docs()


def settings() -> dict:
    return {
        "expansion_interval_s": _f("QUIPU_EXPANSION_INTERVAL_S", 600.0),
        "pulse_route": os.environ.get("QUIPU_PULSE_ROUTE", "0").strip() == "1",
        "pulse_minutes": _f("QUIPU_PULSE_MINUTES", 10.0),
        "doc_anneal_minutes": _f("QUIPU_DOC_ANNEAL_MINUTES", 30.0),
        "self_organising": os.environ.get("QUIPU_SELF_ORGANISING", "0").strip() == "1",
        "brain_owner": os.environ.get("QUIPU_BRAIN_OWNER") == "1",
    }


def status() -> dict:
    from .local_store import db_path
    with _STATUS_LOCK:
        loops = json.loads(json.dumps(_STATUS, default=str))
    try:
        brain = str(db_path())
    except Exception as exc:
        brain = f"unavailable: {exc}"
    return {"started_at": _STARTED_AT, "settings": settings(), "brain": brain, "loops": loops}


def start_background(stop: threading.Event | None = None) -> list[threading.Thread]:
    """Start the Entirety's loops (not the HTTP server).  Returns the threads."""
    global _STARTED_AT
    _STARTED_AT = time.time()
    cfg = settings()
    if cfg["self_organising"]:
        try:
            from .qpsi import self_organising
            self_organising.enable()
        except Exception as exc:
            _LOG.error("[entirety] self_organising not enabled: %s", exc)
    plan: list[tuple[str, float, Callable[[], Any], float]] = [
        ("expansion", cfg["expansion_interval_s"], expansion_step, 20.0),
        ("doc_annealing", cfg["doc_anneal_minutes"] * 60.0, doc_annealing, 60.0),
    ]
    if cfg["pulse_route"]:
        plan.append(("pulse", cfg["pulse_minutes"] * 60.0, pulse, 90.0))
    else:
        _record("pulse", disabled="QUIPU_PULSE_ROUTE is not 1 (the operator's switch)")
    threads = []
    for name, every, fn, delay in plan:
        t = threading.Thread(target=_loop, args=(name, every, fn), kwargs={"first_delay_s": delay, "stop": stop},
                             name=f"entirety-{name}", daemon=True)
        t.start()
        threads.append(t)
    return threads


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="entirety_service", description=__doc__.splitlines()[0])
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "status"])
    args = ap.parse_args(argv)
    if args.cmd == "status":
        import urllib.request
        port = os.environ.get("QUIPU_PORT", "7100")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/entirety", timeout=5) as r:
            print(r.read().decode())
        return 0
    logging.basicConfig(level=os.environ.get("QUIPU_LOG_LEVEL", "INFO"))
    if os.environ.get("QUIPU_BRAIN_OWNER") != "1":
        _LOG.warning("[entirety] QUIPU_BRAIN_OWNER is not 1: this process is not declared the brain's writer")
    start_background()
    from . import observer_service
    observer_service.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
