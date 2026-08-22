"""Epistemic rupture detection for the Observer's world model."""

from __future__ import annotations

import time
from typing import Any

from . import brain_kv, mesh_slm

def world_model_state() -> dict[str, Any]:
    """Return the current world-model dialectic state.
    
    Returns a dict with:
    - phase: str — one of 'receptive_hunger', 'empirical_precedent', 'targeted_epistemic', 'continuous_synthesis'
    - acquisition_pressure: float — 0..1, how strongly the system seeks new data
    - epistemic_surprise: float — 0..1, rolling measure of how much recent observations break existing patterns
    - rupture_events: list[dict] — recent rupture events (concept splits)
    - precedent_depth: float — 0..1, how much established empirical precedent exists
    - retrieval_directive: dict — what the learning retriever should prioritize
    """
    surprise_ema = float(brain_kv.kv_get_json("world_model:surprise_ema", 0.0))
    rupture_log = brain_kv.kv_get_json("world_model:rupture_log", [])
    phase = brain_kv.kv_get_json("world_model:phase", "receptive_hunger")
    precedent_depth = float(brain_kv.kv_get_json("world_model:precedent_depth", 0.0))
    
    # Calculate acquisition pressure
    # Higher surprise -> higher pressure
    # Lower precedent -> higher pressure
    acquisition_pressure = max(0.0, min(1.0, 1.0 - precedent_depth + surprise_ema))
    
    return {
        "phase": phase,
        "acquisition_pressure": round(acquisition_pressure, 4),
        "epistemic_surprise": round(surprise_ema, 4),
        "rupture_events": rupture_log,
        "precedent_depth": round(precedent_depth, 4),
        "retrieval_directive": retrieval_directive(),
    }

def assess_observation(source: str, tokens: list[str], confidence: float | None,
                       coverage: float, novel_tokens: list[str]) -> dict[str, Any]:
    """Called after each /observe to assess epistemic impact.
    
    Computes epistemic surprise for this observation and checks for rupture conditions.
    A rupture occurs when:
    1. High novelty (coverage < 0.3) + high confidence (> 0.8) = the sensor is confident about something the mesh has never seen
    2. STP geodesic gap is diverging (slope > 0) while loss is plateaued = the learned embedding can't explain new trajectories
    3. Entropy differential ΔS spikes above 2σ of its rolling mean = graph structure is being disrupted
    
    Returns dict with 'surprise', 'rupture_detected', 'rupture_event' (if any), 'phase_transition' (if any)
    """
    conf = confidence if confidence is not None else 0.5
    
    # Base surprise from novelty and confidence
    novelty = max(0.0, min(1.0, 1.0 - coverage))
    surprise = max(0.0, min(1.0, novelty * conf))
    
    # Update surprise EMA
    surprise_ema = float(brain_kv.kv_get_json("world_model:surprise_ema", 0.0))
    surprise_ema = max(0.0, min(1.0, 0.9 * surprise_ema + 0.1 * surprise))
    brain_kv.kv_set_json("world_model:surprise_ema", surprise_ema)
    
    # Update precedent depth
    precedent_depth = float(brain_kv.kv_get_json("world_model:precedent_depth", 0.0))
    # Gradually increase precedent as we see more data
    precedent_depth = max(0.0, min(1.0, precedent_depth + 0.001))
    brain_kv.kv_set_json("world_model:precedent_depth", precedent_depth)
    
    # Check rupture conditions
    rupture_detected = False
    rupture_reason = ""
    
    if coverage < 0.3 and conf > 0.8:
        rupture_detected = True
        rupture_reason = "High novelty + high confidence"
        
    try:
        # Check STP gap
        stp_trend = mesh_slm.stp_diagnostic_trend()
        slope = stp_trend.get("slope", 0.0)
        loss_plateaued = stp_trend.get("loss_plateaued", False)
        if slope > 0 and loss_plateaued:
            rupture_detected = True
            rupture_reason = "STP geodesic gap diverging with plateaued loss"
            
        # Check Entropy Differential
        summary = mesh_slm.state_summary()
        entropy_diff = summary.get("entropy_differential", 0.0)
        rolling_mean = float(brain_kv.kv_get_json("world_model:entropy_mean", 0.0))
        rolling_var = float(brain_kv.kv_get_json("world_model:entropy_var", 0.0))
        rolling_std = max(0.0, rolling_var) ** 0.5
        
        if rolling_std > 0 and entropy_diff > rolling_mean + 2 * rolling_std:
            rupture_detected = True
            rupture_reason = "Entropy differential spike"
            
        # Update rolling stats for entropy
        alpha = 0.1
        new_mean = (1 - alpha) * rolling_mean + alpha * entropy_diff
        new_var = (1 - alpha) * rolling_var + alpha * (entropy_diff - rolling_mean) ** 2
        brain_kv.kv_set_json("world_model:entropy_mean", new_mean)
        brain_kv.kv_set_json("world_model:entropy_var", new_var)
    except Exception:
        pass
        
    rupture_event = None
    if rupture_detected:
        rupture_event = {
            "source": source,
            "reason": rupture_reason,
            "surprise": surprise,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        rupture_log = brain_kv.kv_get_json("world_model:rupture_log", [])
        rupture_log.append(rupture_event)
        brain_kv.kv_set_json("world_model:rupture_log", rupture_log[-50:])
        
    # Phase transition
    current_phase = brain_kv.kv_get_json("world_model:phase", "receptive_hunger")
    new_phase = current_phase
    
    if surprise > 0.8:
        new_phase = "receptive_hunger"
        precedent_depth = max(0.0, precedent_depth - 0.2)
        brain_kv.kv_set_json("world_model:precedent_depth", precedent_depth)
    elif precedent_depth < 0.2:
        new_phase = "receptive_hunger"
    elif 0.2 <= precedent_depth < 0.6:
        new_phase = "empirical_precedent"
    elif 0.6 <= precedent_depth < 0.85:
        new_phase = "targeted_epistemic"
    else:
        new_phase = "continuous_synthesis"
        
    phase_transition = None
    if new_phase != current_phase:
        phase_transition = {"from": current_phase, "to": new_phase}
        brain_kv.kv_set_json("world_model:phase", new_phase)
        
    return {
        "surprise": round(surprise, 4),
        "rupture_detected": rupture_detected,
        "rupture_event": rupture_event,
        "phase_transition": phase_transition
    }

def retrieval_directive() -> dict[str, Any]:
    """Generate a directive for the learning retriever based on current world-model state.
    
    Returns dict with:
    - strategy: 'broad_exploration' | 'precedent_building' | 'targeted_gap_closing' | 'synthesis_verification'
    - focus_domains: list[str] — topic areas to prioritize
    - confidence_floor: float — minimum confidence for new acquisitions
    - rationale: str — why this strategy was chosen
    """
    phase = brain_kv.kv_get_json("world_model:phase", "receptive_hunger")
    
    if phase == "receptive_hunger":
        strategy = "broad_exploration"
        focus_domains = ["unstructured_vision", "novel_structures"]
        confidence_floor = 0.2
        rationale = "System is in early acquisition or recovering from major rupture; need broad coverage."
    elif phase == "empirical_precedent":
        strategy = "precedent_building"
        focus_domains = ["established_patterns", "cross_corpus_correlations"]
        confidence_floor = 0.5
        rationale = "Building reliable baseline patterns from empirical data."
    elif phase == "targeted_epistemic":
        strategy = "targeted_gap_closing"
        focus_domains = ["stp_divergence_areas", "low_confidence_nodes"]
        confidence_floor = 0.7
        rationale = "Targeting specific gaps where the mesh fails to explain observations."
    else:
        strategy = "synthesis_verification"
        focus_domains = ["edge_cases", "high_entropy_boundaries"]
        confidence_floor = 0.85
        rationale = "Mature model; focusing on verifying stability and finding subtle ruptures."
        
    return {
        "strategy": strategy,
        "focus_domains": focus_domains,
        "confidence_floor": confidence_floor,
        "rationale": rationale,
    }


def source_guidance_directive(source: str, device_id: str | None = None) -> dict[str, Any]:
    """Generate source-specific, phase-adaptive feedback and sensory priors.
    
    Molds the downstream client (Loadopoly-OCR or Bakugo) based on the
    Observer's annealed cross-corpus model.
    """
    phase = brain_kv.kv_get_json("world_model:phase", "receptive_hunger")
    directive = retrieval_directive()
    
    if source == "loadopoly-ocr":
        # Vision Axis: provide prompt directives, lexicon priors, and confidence gates
        return {
            "source": source,
            "axis": "vision",
            "phase": phase,
            "confidence_floor": directive["confidence_floor"],
            "strategy": directive["strategy"],
            "prompt_guidance": {
                "broad_exploration": "Extract all novel tokens and spatial entity relationships aggressively.",
                "precedent_building": "Prioritize structured catalog numbers and cross-referenced archival dates.",
                "targeted_epistemic": "Focus on high-uncertainty text regions and verify ambiguous character sequences.",
                "continuous_synthesis": "Execute strict semantic validation and verify graph edge consistency.",
            }.get(directive["strategy"], "Standard OCR extraction"),
            "device_isolated": bool(device_id),
        }
    elif source == "bakugo":
        # Touch Axis: provide metrology calibration, Snell refraction indices, SPRT stopping
        return {
            "source": source,
            "axis": "touch",
            "phase": phase,
            "confidence_floor": directive["confidence_floor"],
            "strategy": directive["strategy"],
            "refraction_priors": {
                "pmma_n": 1.491,
                "pc_n": 1.586,
                "air_n": 1.000,
            },
            "sprt_parameters": {
                "alpha": 0.05,
                "beta": 0.05,
                "boundary_threshold": 55.0,
                "info_value_cutoff": 0.02,
            },
            "device_isolated": bool(device_id),
        }
    
    return {
        "source": source,
        "phase": phase,
        "directive": directive,
        "device_isolated": bool(device_id),
    }


def annealing_cycle() -> dict[str, Any]:
    """Execute a self-annealing iteration across the sensory manifold.
    
    Folds sensory feedback into the 7-D manifold, updates cognitive phases,
    and returns the updated living state.
    """
    t0 = time.time()
    state = world_model_state()
    summary = mesh_slm.state_summary()
    
    # Anneal precedent depth slightly towards synthesis if loss is healthy
    precedent = state["precedent_depth"]
    if summary.get("stp_loss", 1.0) < 0.5:
        precedent = min(1.0, precedent + 0.005)
        brain_kv.kv_set_json("world_model:precedent_depth", precedent)
        
    history = brain_kv.kv_get_json("world_model:anneal_history", []) or []
    event = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": state["phase"],
        "precedent_depth": round(precedent, 4),
        "vocab_size": summary.get("vocab_size", 0),
        "quipu_edges": summary.get("quipu_edges", 0),
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
    }
    history.append(event)
    brain_kv.kv_set_json("world_model:anneal_history", history[-50:])
    
    return {
        "status": "annealed",
        "event": event,
        "world_model": world_model_state(),
        "mesh_summary": summary,
    }

