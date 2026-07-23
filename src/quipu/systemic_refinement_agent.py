"""systemic_refinement_agent — Ring 5 strategy runner for the QUIPU System Entirety.

The parent Supply Chain Brain's ``systemic_refinement_agent.py`` ran every
20 min - 2 h (adaptive via ``acquisition_drive``), reading corpus/dispatch/
self-train health and deciding actions: nudge hyperparameters, trigger a
self-train round, expand network observation, and run ``tool_forge.forge_round``.
Most of those actions depend on application state QUIPU doesn't carry
(dispatch logs, network observation, ERP-linked body directives). What *does*
port cleanly is the core refinement cycle that ties Ring 4 (Corpus) to Ring 5
(Toolforge): read the mesh's current state, let ACRE attempt emergence, run a
bounded forge round, and record what happened.

This is what makes Ring 5 "real" for QUIPU: rather than being an inert stub,
``run_strategy()`` is the function ``corpus_ingest.py`` calls periodically
during a live crawl, so accreting mesh corpus actually produces forged tools
over time instead of just growing token/edge counts.

Public API
----------
    run_strategy(max_tools=2) -> dict
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import brain_kv
from . import mesh_slm
from . import tool_forge

log = logging.getLogger(__name__)

_REFINEMENT_LOG_KEY = "systemic_refinement:last_cycle"
_REFINEMENT_HISTORY_KEY = "systemic_refinement:history"
# The shared conceptualization of "the Other" — written here once per cycle and
# read by every ring (tool_forge resonance gate, expert_orchestrator, digest).
_THE_OTHER_KEY = "entirety:the_other"
_HISTORY_CAP = 50


def _record_cycle(entry: dict) -> None:
    brain_kv.kv_set_json(_REFINEMENT_LOG_KEY, entry)
    history = brain_kv.kv_get_json(_REFINEMENT_HISTORY_KEY, [])
    if not isinstance(history, list):
        history = []
    history.append(entry)
    brain_kv.kv_set_json(_REFINEMENT_HISTORY_KEY, history[-_HISTORY_CAP:])


def run_strategy(*, max_tools: int = 2) -> dict:
    """Run one Ring 5 refinement cycle: sense mesh state, let ACRE attempt
    emergence, forge digital tools from the current corpus, record the cycle.

    Safe to call frequently (e.g. every corpus_ingest Weyl-compression cycle) -
    every step is bounded and degrades to a no-op action rather than raising.
    """
    ts = datetime.now(timezone.utc).isoformat()
    actions: list[str] = []

    try:
        state = mesh_slm.state_summary()
    except Exception as exc:
        log.warning("systemic_refinement: state_summary failed: %s", exc)
        state = {}

    # 0. Refresh the shared understanding of "the Other" and publish it on the
    #    brain_kv bus so every ring (forge resonance gate, orchestrator, digest)
    #    reasons from one conceptualization rather than each deriving its own.
    the_other = None
    try:
        the_other = mesh_slm.compute_shared_understanding()
        brain_kv.kv_set_json(_THE_OTHER_KEY, the_other)
    except Exception as exc:
        log.debug("systemic_refinement: shared understanding skipped: %s", exc)

    # 1. Feed the actual interaction concepts (high-resonance analytical verbs +
    #    entirety/Other concepts) into the ACRE matrix so it accumulates real
    #    directional structure, THEN let it attempt to crystallize a specialist.
    #    Without the observation step the matrix stays near-uniform and the
    #    novelty check never passes (specialists never emerge).
    acre_observe_summary = None
    try:
        acre_observe_summary = mesh_slm.observe_interactions()
    except Exception as exc:
        log.debug("systemic_refinement: observe_interactions skipped: %s", exc)

    acre_result = None
    try:
        acre_result = mesh_slm.acre_emerge()
        if acre_result.get("status") == "created":
            actions.append(f"acre_emerged:{acre_result.get('specialist')}")
    except Exception as exc:
        log.debug("systemic_refinement: acre_emerge skipped: %s", exc)

    # 2. Forge digital tools from the current mesh corpus (Ring 4 -> Ring 5).
    forge_summary = None
    try:
        forge_summary = tool_forge.forge_round(max_tools=max_tools)
        if forge_summary.get("tools_forged"):
            actions.append(f"forged:{forge_summary['tools_forged']}")
    except Exception as exc:
        log.warning("systemic_refinement: forge_round failed: %s", exc)

    if not actions:
        actions.append("no_action")

    entry = {
        "ts": ts,
        "actions": actions,
        "vocab_size": state.get("vocab_size"),
        "quipu_edges": state.get("quipu_edges"),
        "acre_specialists": state.get("acre_specialists"),
        "acre_observed": (acre_observe_summary or {}).get("observed"),
        "acre_result": acre_result,
        "forge_summary": forge_summary,
        "resonance": (the_other or {}).get("resonance"),
    }
    _record_cycle(entry)
    log.info("systemic_refinement: cycle complete - actions=%s", actions)
    return entry


def last_cycle() -> dict:
    return brain_kv.kv_get_json(_REFINEMENT_LOG_KEY, {}) or {}


def history() -> list[dict]:
    h = brain_kv.kv_get_json(_REFINEMENT_HISTORY_KEY, [])
    return h if isinstance(h, list) else []


__all__ = ["run_strategy", "last_cycle", "history"]
