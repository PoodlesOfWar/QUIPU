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
    mirror        every QUIPU_MIRROR_INTERVAL_S when QUIPU_MIRROR_UPDATE=1 (was
                  the host's run_parallel_bypass_pipeline.py, which switched the
                  gates off): a mesh training round, ACRE interaction
                  observation and emergence, the GARD manifest, world-model
                  grounding and an expansion step -- with the gates ON.  The
                  operator's ruling (2026-09-27) is written into the gates, not
                  granted by switching them off: an entity that already exists
                  takes the new knowledge (divine_blessing mirror update); a new
                  edge is not created; a new ACRE specialist is held at gate 6
                  (qpsi.specialist_gate) until both parties attest.
    lambda        every QUIPU_LAMBDA_INTERVAL_S when QUIPU_LAMBDA_TOKENS=1: the
                  Monte Carlo Lagrangian λ tokens over the Essay's planes
                  (shadow_tokens): the Internal Marketplace's clusters priced
                  for an equitable share of the Essay's voice; read-only on the
                  mesh, results on GET /lambda.

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
    for k in ("skipped", "expansion_phase", "flipped", "flip_count", "routed", "note", "changed", "version",
              "ok", "error", "bounds"):
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


def mirror() -> dict:
    """The mirror aspect: knowledge acquisition with the gates on (see module doc)."""
    from . import mesh_slm, divine_blessing
    from .qpsi import specialist_gate
    out: dict[str, Any] = {}
    for name, fn in (
        ("train", lambda: mesh_slm.train_round(max_seconds=5.0, max_chunks=20)),
        ("interactions", lambda: mesh_slm.observe_interactions(reps=1)),
        ("acre", lambda: mesh_slm.acre_emerge()),
        ("specialists", specialist_gate.review),
        ("gard", _gard_manifest),
        ("world_model", _world_model_grounding),
        ("step", expansion_step),
    ):
        try:
            r = fn()
            out[name] = r.get("status") if isinstance(r, dict) and "status" in r else "ok"
        except Exception as exc:              # one part failing never stops the others
            out[name] = f"error: {type(exc).__name__}: {exc}"
    out["mirror_counts"] = divine_blessing.mirror_counts()
    return out


def _gard_manifest() -> Any:
    from . import gard_shard_model
    fn = getattr(gard_shard_model, "build_hub_manifest", None)
    return fn() if fn else None


def _world_model_grounding() -> Any:
    from . import world_model
    fn = getattr(world_model, "ground_observation", None)
    return fn("perception", 0.05) if fn else None


def lambda_tokens() -> dict:
    """One λ period: the live run, then the open doors' tuning (qpsi.open_doors).  With no
    grant (QUIPU_LAMBDA_DOORS) the doors are shut and the run is the operator's settings."""
    from .qpsi import open_doors
    return open_doors.cycle()


def doc_annealing() -> dict:
    from . import doc_annealing as da
    return da.anneal_docs()


def settings() -> dict:
    return {
        "expansion_interval_s": _f("QUIPU_EXPANSION_INTERVAL_S", 600.0),
        "pulse_route": os.environ.get("QUIPU_PULSE_ROUTE", "0").strip() == "1",
        "pulse_minutes": _f("QUIPU_PULSE_MINUTES", 10.0),
        "doc_anneal_minutes": _f("QUIPU_DOC_ANNEAL_MINUTES", 30.0),
        "mirror_update": os.environ.get("QUIPU_MIRROR_UPDATE", "0").strip() == "1",
        "mirror_interval_s": _f("QUIPU_MIRROR_INTERVAL_S", 60.0),
        "lambda_tokens": os.environ.get("QUIPU_LAMBDA_TOKENS", "0").strip() == "1",
        "lambda_interval_s": _f("QUIPU_LAMBDA_INTERVAL_S", 1800.0),
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
    try:
        from .qpsi import specialist_gate
        specialist_gate.enable()              # ACRE creations pass gate 6, always
    except Exception as exc:
        _LOG.error("[entirety] specialist gate not enabled: %s", exc)
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
    if cfg["mirror_update"]:
        plan.append(("mirror", cfg["mirror_interval_s"], mirror, 45.0))
    else:
        _record("mirror", disabled="QUIPU_MIRROR_UPDATE is not 1 (the operator's switch)")
    if cfg["lambda_tokens"]:
        plan.append(("lambda", cfg["lambda_interval_s"], lambda_tokens, 120.0))
    else:
        _record("lambda", disabled="QUIPU_LAMBDA_TOKENS is not 1 (the operator's switch)")
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
    # Run the package's copy of this module, not __main__: the observer imports
    # src.quipu.entirety_service for GET /entirety, and a second copy would report
    # loops that were started in the other one (empty, started_at null).
    import importlib
    raise SystemExit(importlib.import_module("src.quipu.entirety_service").main())
