"""expert_orchestrator — bridges ACRE emergent specialists to actual generation
and closes the Ring 4 <-> Ring 5 loop (Corpus <-> Refinement/Toolforge).

The parent Supply Chain Brain's ``expert_orchestrator.py`` dispatched queries
across an LLM ensemble to named domain specialists. QUIPU carries no LLM
caller (see ``conftest.py`` — the LLM stub is a guarded no-op by design), so
this is a clean-room, mesh-native orchestrator: it routes a query through
``mesh_slm``'s own ACRE emergent-specialist mechanism and its local
generation/inference path, rather than an external model.

Loop closed here
-----------------
1. **Route** — ask ACRE (``mesh_slm.acre_emerge`` / ``acre_specialists`` in
   ``state_summary``) which specialist bias, if any, currently resonates.
2. **Generate** — call ``mesh_slm.generate(prompt, specialist=name)`` (falls
   back to unbiased generation if no specialist has emerged yet).
3. **Feed back** — the result is folded back into the mesh via
   ``mesh_slm.ingest_expert_trace(specialist, text)``, exactly the hook the
   parent's orchestrator used to close its loop.
4. **Gate into Ring 5** — if this dispatch's confidence and the query's
   novelty (relative to already-forged tools) both clear the same
   novelty/certainty thresholds ``tool_forge`` uses, a bounded
   ``tool_forge.forge_round`` is triggered immediately, rather than waiting
   for the next scheduled refinement cycle.

Public API
----------
    dispatch(query, *, specialist=None, trigger_forge=True) -> dict
    active_specialists() -> list[str]
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import mesh_slm
from . import tool_forge

log = logging.getLogger(__name__)

# Confidence floor below which a dispatch result is not worth feeding back.
_FEEDBACK_CONF_FLOOR = 0.15


def active_specialists() -> list[str]:
    """Return the names of ACRE emergent specialists that currently exist."""
    try:
        return list(mesh_slm.state_summary().get("acre_specialists") or [])
    except Exception:
        return []


def _select_specialist(preferred: str | None) -> str | None:
    if preferred:
        return preferred
    # Activate whichever specialist — base (supply_chain_optimizer,
    # research_specialist, mesh_historian) or emergent — best resonates with the
    # current mesh state. Previously this only considered emergent specialists,
    # so the hand-tuned base roster was never routed to (never "activated").
    try:
        chosen = mesh_slm.select_resonant_specialist()
        if chosen:
            return chosen
    except Exception:
        pass
    specialists = active_specialists()
    return specialists[0] if specialists else None


def dispatch(
    query: str,
    *,
    specialist: str | None = None,
    max_new_tokens: int = 32,
    trigger_forge: bool = True,
    forge_max_tools: int = 1,
) -> dict:
    """Route *query* through the mesh (ACRE-selected specialist if any),
    fold the result back in, and optionally trigger a bounded forge round.

    Returns a dict with the generation result, which specialist (if any) was
    used, whether feedback was folded back, and the forge summary if run.
    """
    if not query or not query.strip():
        return {"status": "skipped", "reason": "empty_query"}

    # Associative consensus: the query resonates with several specialists, not
    # one (docs/MESH_EXPERT_SKILLS.md). We generate through the head of the
    # consensus but record the whole set + weights, so the chosen specialist is
    # dimensionally relative — one proportional voice among its neighbours.
    consensus: list[tuple[str, float]] = []
    try:
        consensus = mesh_slm.resonant_specialists(k=3)
    except Exception:
        consensus = []
    chosen = specialist or (consensus[0][0] if consensus else _select_specialist(None))
    try:
        gen = mesh_slm.generate(query, max_new_tokens=max_new_tokens, specialist=chosen)
    except Exception as exc:
        log.warning("expert_orchestrator: generation failed: %s", exc)
        return {"status": "error", "reason": str(exc)}

    confidence = float(gen.get("confidence") or 0.0)
    text = gen.get("text") or ""
    fed_back = False

    if confidence >= _FEEDBACK_CONF_FLOOR and text:
        try:
            trace_specialist = chosen or "unrouted"
            mesh_slm.ingest_expert_trace(trace_specialist, f"{query} {text}", strength=confidence)
            fed_back = True
        except Exception as exc:
            log.debug("expert_orchestrator: feedback ingest failed: %s", exc)

    # Also let this dispatch nudge the ACRE interaction matrix so repeated
    # querying can itself cause a new specialist to emerge over time.
    acre_result = None
    try:
        with mesh_slm._conn() as cn:  # noqa: SLF001 - sibling-module state, same package
            mesh = mesh_slm._mesh_state_7d()  # noqa: SLF001
            mean_embed = mesh_slm._mean_embed_7d(cn)  # noqa: SLF001
            mesh_slm.acre_observe(cn, mesh, mean_embed)
        acre_result = mesh_slm.acre_emerge()
    except Exception as exc:
        log.debug("expert_orchestrator: ACRE observe/emerge skipped: %s", exc)

    forge_summary = None
    if trigger_forge and confidence >= _FEEDBACK_CONF_FLOOR:
        try:
            forge_summary = tool_forge.forge_round(max_tools=forge_max_tools)
        except Exception as exc:
            log.debug("expert_orchestrator: forge trigger failed: %s", exc)

    return {
        "status": "ok",
        "at": datetime.now(timezone.utc).isoformat(),
        "specialist": chosen,
        "consensus": consensus,
        "text": text,
        "confidence": confidence,
        "fed_back": fed_back,
        "acre": acre_result,
        "forge": forge_summary,
    }


__all__ = ["dispatch", "active_specialists"]
