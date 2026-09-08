"""QUIPU Market Ingestion & Floquet Resonance Bridge.

Allows QUIPU services and sovereign agents to query, inspect, and ingest
job market demand baselines transformed by Supply Chain Architect (SCA).
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from . import brain_kv, mesh_slm
from .system_entirety import bit_flip_parity, oscillating_expansion_step

log = logging.getLogger(__name__)

_KV_MARKET_BASE = "entirety:market_base"
_KV_DEMAND_VECTOR = "market:demand_vector"
_KV_RESONANCE = "market:competency_resonance"
_OMEGA_BASE = 2.0 * math.pi / 600.0


def get_market_telemetry() -> dict[str, Any]:
    """Retrieve currently anchored market demand base state and resonance."""
    market_base = brain_kv.kv_get_json(_KV_MARKET_BASE, {})
    demand_vector = brain_kv.kv_get_json(_KV_DEMAND_VECTOR, {})
    resonance = brain_kv.kv_get_json(_KV_RESONANCE, {})
    bit_state = brain_kv.kv_get("entirety:bit_state")
    phase = brain_kv.kv_get("entirety:expansion_phase")
    flip_count = brain_kv.kv_get("entirety:flip_count")

    return {
        "market_base": market_base,
        "demand_vector": demand_vector,
        "competency_resonance": resonance,
        "floquet_state": {
            "bit_state": int(bit_state) if bit_state is not None else 0,
            "phase": phase or "unspecified",
            "flip_count": int(flip_count) if flip_count is not None else 0,
            "omega_base": _OMEGA_BASE
        }
    }


def ingest_market_observation(
    observation_text: str,
    confidence: float = 0.95,
    source: str = "jobhawk"
) -> dict[str, Any]:
    """Digest a market demand observation directly into the MESH SLM corpus."""
    axis_source = f"{source}/code/hideout-mesh"
    fed_rows = mesh_slm.feed_corpus(observation_text, source=axis_source)
    tokens = mesh_slm._tokenize(observation_text)

    # Perform instantaneous Floquet expansion step
    step_result = oscillating_expansion_step(force=True)

    return {
        "ok": True,
        "source": source,
        "tokens_fed": len(tokens),
        "fed_rows": fed_rows,
        "floquet_step": step_result
    }
