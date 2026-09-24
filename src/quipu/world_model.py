"""Epistemic rupture detection and forward state transition dynamics for the Observer's world model."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
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
    - last_prediction: dict | None — most recent predicted state S_hat_{t+1}
    """
    surprise_ema = float(brain_kv.kv_get_json("world_model:surprise_ema", 0.0))
    rupture_log = brain_kv.kv_get_json("world_model:rupture_log", [])
    phase = brain_kv.kv_get_json("world_model:phase", "receptive_hunger")
    precedent_depth = float(brain_kv.kv_get_json("world_model:precedent_depth", 0.0))
    last_pred = brain_kv.kv_get_json("world_model:last_prediction", None)
    
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
        "last_prediction": last_pred,
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


# ============================================================================
# Forward State Transition Dynamics: \hat{S}_{t+1} = f(S_t, A_t)
# ============================================================================


@dataclass
class WorldState:
    """State vector S_t for the world model.
    
    Captures physical, operational, and network dimensions of an observed node or system,
    including discrete pipeline delays, backlogs, capacity constraints, costs, and epistemic uncertainty.
    """
    inventory: float = 0.0
    backlog: float = 0.0
    pipeline: list[dict[str, Any]] = field(default_factory=list)  # list of {"units": float, "remaining_ticks": int}
    demand_rate: float = 0.0
    lead_time: float = 1.0
    capacity: float = 1000.0
    flux: float = 0.0
    load_ratio: float = 0.0
    cost_accumulated: float = 0.0
    reliability: float = 1.0
    latency_ms: float = 10.0
    uncertainty: dict[str, float] = field(default_factory=lambda: {
        "inventory": 1.0,
        "backlog": 0.5,
        "flux": 0.5,
        "cost": 1.0,
        "reliability": 0.05,
    })
    custom_metrics: dict[str, float] = field(default_factory=dict)
    tick: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldState:
        valid_keys = {
            "inventory", "backlog", "pipeline", "demand_rate", "lead_time",
            "capacity", "flux", "load_ratio", "cost_accumulated", "reliability",
            "latency_ms", "uncertainty", "custom_metrics", "tick",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        if "pipeline" in filtered and isinstance(filtered["pipeline"], list):
            filtered["pipeline"] = [dict(p) for p in filtered["pipeline"]]
        if "uncertainty" in filtered and isinstance(filtered["uncertainty"], dict):
            filtered["uncertainty"] = dict(filtered["uncertainty"])
        if "custom_metrics" in filtered and isinstance(filtered["custom_metrics"], dict):
            filtered["custom_metrics"] = dict(filtered["custom_metrics"])
        return cls(**filtered)


def _to_dict(obj: Any) -> dict[str, Any]:
    """Safely convert WorldState, TransitionEvent, or dict-like object to a plain dict."""
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return dict(obj)


@dataclass
class TransitionEvent:
    """Action or exogenous shock A_t for state transition S_hat_{t+1} = f(S_t, A_t)."""
    # Decisions (controllable interventions)
    order_quantity: float = 0.0
    target_capacity: float | None = None
    target_demand_rate: float | None = None
    expedite_ticks: int = 0
    reroute_ratio: float = 0.0
    
    # Exogenous shocks (disturbances)
    demand_multiplier: float = 1.0
    demand_delta: float = 0.0
    lead_time_shock: int = 0
    capacity_reduction: float = 0.0
    defect_shock: float = 0.0
    
    # Unit economic rates
    holding_cost_rate: float = 0.1
    stockout_cost_rate: float = 1.0
    order_cost_fixed: float = 2.0
    order_cost_unit: float = 0.5
    
    # Custom shocks
    custom_shocks: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionEvent:
        valid_keys = {
            "order_quantity", "target_capacity", "target_demand_rate", "expedite_ticks", "reroute_ratio",
            "demand_multiplier", "demand_delta", "lead_time_shock", "capacity_reduction",
            "defect_shock", "holding_cost_rate", "stockout_cost_rate", "order_cost_fixed",
            "order_cost_unit", "custom_shocks",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        if "custom_shocks" in filtered and isinstance(filtered["custom_shocks"], dict):
            filtered["custom_shocks"] = dict(filtered["custom_shocks"])
        return cls(**filtered)


def transition_step(
    state: WorldState | dict[str, Any],
    action: TransitionEvent | dict[str, Any] | None = None,
    dt: float = 1.0,
) -> WorldState:
    """Calculate the predicted next state S_hat_{t+1} = f(S_t, A_t).
    
    Governed by physical mass balance, pipeline delay queues, capacity clamping,
    backlog propagation, cost accumulation, and uncertainty tracking.
    """
    if hasattr(state, "to_dict") and callable(state.to_dict):
        s = WorldState.from_dict(state.to_dict())
    elif isinstance(state, dict):
        s = WorldState.from_dict(state)
    elif hasattr(state, "__dataclass_fields__"):
        s = WorldState.from_dict(asdict(state))
    else:
        s = WorldState()

    if action is None:
        a = TransitionEvent()
    elif hasattr(action, "to_dict") and callable(action.to_dict):
        a = TransitionEvent.from_dict(action.to_dict())
    elif isinstance(action, dict):
        a = TransitionEvent.from_dict(action)
    elif hasattr(action, "__dataclass_fields__"):
        a = TransitionEvent.from_dict(asdict(action))
    else:
        a = TransitionEvent()

    # 1. Pipeline progression and disruption shocks
    # Each in-transit order has remaining_ticks.
    # Exogenous lead time shock delays arrivals; expedite reduces remaining ticks.
    receipts = 0.0
    updated_pipeline: list[dict[str, Any]] = []

    for item in s.pipeline:
        units = float(item.get("units", 0.0))
        rem = int(item.get("remaining_ticks", 1))

        # Apply shocks / expedite
        rem += a.lead_time_shock
        if a.expedite_ticks > 0:
            rem = max(1, rem - a.expedite_ticks)

        # Tick forward
        rem -= 1
        if rem <= 0:
            receipts += units
        else:
            updated_pipeline.append({"units": units, "remaining_ticks": rem})

    # New order placement
    if a.order_quantity > 0.0:
        base_lt = max(1, int(round(s.lead_time)))
        effective_lt = max(1, base_lt + max(0, a.lead_time_shock))
        updated_pipeline.append({
            "units": float(a.order_quantity),
            "remaining_ticks": effective_lt,
        })

    # 2. Demand calculation
    base_demand = a.target_demand_rate if a.target_demand_rate is not None else s.demand_rate
    eff_demand_rate = max(0.0, base_demand * max(0.0, a.demand_multiplier) + a.demand_delta)
    period_demand = eff_demand_rate * dt
    total_demand = period_demand + s.backlog

    # 3. Capacity constraints
    base_cap = a.target_capacity if a.target_capacity is not None else s.capacity
    cap_reduction = max(0.0, min(1.0, a.capacity_reduction))
    effective_cap = max(0.0, base_cap * (1.0 - cap_reduction)) * dt

    # 4. Conservation of mass & Fulfillment
    available_stock = s.inventory + receipts
    fulfilled = min(available_stock, total_demand, effective_cap)

    next_inventory = max(0.0, available_stock - fulfilled)
    next_backlog = max(0.0, total_demand - fulfilled)
    next_flux = fulfilled / dt if dt > 0 else fulfilled
    next_load_ratio = next_flux / max(1e-6, base_cap)

    # 5. Cost calculation
    holding_cost = a.holding_cost_rate * next_inventory * dt
    stockout_cost = a.stockout_cost_rate * next_backlog * dt
    order_cost = (a.order_cost_fixed + a.order_cost_unit * a.order_quantity) if a.order_quantity > 0 else 0.0
    step_cost = holding_cost + stockout_cost + order_cost
    next_cost = s.cost_accumulated + step_cost

    # 6. Reliability & Latency
    stress_degrade = max(0.0, next_load_ratio - 0.9) * 0.1
    defect_drop = max(0.0, min(1.0, a.defect_shock))
    recovery = 0.01 * (1.0 - s.reliability) if next_load_ratio <= 0.8 else 0.0
    next_reliability = max(0.0, min(1.0, s.reliability * (1.0 - defect_drop) - stress_degrade + recovery))

    base_lat = s.latency_ms if s.latency_ms > 0 else 10.0
    clamped_load = min(0.99, max(0.0, next_load_ratio))
    congestion = clamped_load / (1.001 - clamped_load)
    next_latency = max(1.0, base_lat * (1.0 + 0.5 * congestion))

    # 7. Uncertainty propagation
    prev_unc = s.uncertainty or {}
    inv_sigma = math.sqrt(prev_unc.get("inventory", 1.0) ** 2 + 1.0 + 0.05 * abs(period_demand - fulfilled) + 0.2 * abs(a.lead_time_shock))
    backlog_sigma = math.sqrt(prev_unc.get("backlog", 0.5) ** 2 + 0.5 + 0.1 * next_backlog)
    flux_sigma = math.sqrt(prev_unc.get("flux", 0.5) ** 2 + 0.25 + 0.05 * next_flux)
    cost_sigma = math.sqrt(prev_unc.get("cost", 1.0) ** 2 + 0.1 * step_cost)
    rel_sigma = min(0.5, math.sqrt(prev_unc.get("reliability", 0.05) ** 2 + 0.001 + 0.02 * defect_drop))

    next_uncertainty = {
        "inventory": round(inv_sigma, 4),
        "backlog": round(backlog_sigma, 4),
        "flux": round(flux_sigma, 4),
        "cost": round(cost_sigma, 4),
        "reliability": round(rel_sigma, 4),
    }

    # 8. Custom metrics
    next_custom: dict[str, float] = {}
    all_custom_keys = set(s.custom_metrics.keys()).union(a.custom_shocks.keys())
    for k in all_custom_keys:
        curr_val = s.custom_metrics.get(k, 0.0)
        shock_val = a.custom_shocks.get(k, 0.0)
        next_custom[k] = round(curr_val + shock_val, 4)

    return WorldState(
        inventory=round(next_inventory, 4),
        backlog=round(next_backlog, 4),
        pipeline=updated_pipeline,
        demand_rate=round(base_demand, 4),
        lead_time=round(s.lead_time, 4),
        capacity=round(base_cap, 4),
        flux=round(next_flux, 4),
        load_ratio=round(next_load_ratio, 4),
        cost_accumulated=round(next_cost, 4),
        reliability=round(next_reliability, 4),
        latency_ms=round(next_latency, 4),
        uncertainty=next_uncertainty,
        custom_metrics=next_custom,
        tick=s.tick + 1,
    )


def simulate_rollout(
    initial_state: WorldState | dict[str, Any],
    actions: list[TransitionEvent | dict[str, Any]] | int,
    dt: float = 1.0,
) -> list[WorldState]:
    """Simulate a multi-step forward trajectory [S_0, S_hat_1, S_hat_2, ..., S_hat_T].
    
    If actions is an integer N, simulates N steps with default/passive actions.
    """
    if hasattr(initial_state, "to_dict") and callable(initial_state.to_dict):
        current = WorldState.from_dict(initial_state.to_dict())
    elif isinstance(initial_state, dict):
        current = WorldState.from_dict(initial_state)
    else:
        current = WorldState()

    trajectory = [current]

    if isinstance(actions, int):
        action_list: list[TransitionEvent | dict[str, Any]] = [TransitionEvent() for _ in range(actions)]
    else:
        action_list = actions

    for act in action_list:
        current = transition_step(current, act, dt=dt)
        trajectory.append(current)

    return trajectory


def evaluate_prediction(
    predicted: WorldState | dict[str, Any],
    actual: WorldState | dict[str, Any],
) -> dict[str, Any]:
    """Compare a forward prediction S_hat_{t+1} against real empirical observation S_{t+1}.
    
    Returns residuals, normalized Z-scores, MSE, MAE, epistemic surprise [0..1],
    and a flag indicating whether an epistemic rupture is warranted.
    """
    p = _to_dict(predicted)
    a = _to_dict(actual)

    metrics = ["inventory", "backlog", "flux", "load_ratio", "cost_accumulated", "reliability"]
    residuals: dict[str, float] = {}
    abs_errors: dict[str, float] = {}
    z_scores: dict[str, float] = {}

    uncertainties = p.get("uncertainty", {})
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    count = 0

    for m in metrics:
        if m in p and m in a:
            p_val = float(p[m])
            a_val = float(a[m])
            diff = a_val - p_val
            residuals[m] = round(diff, 4)
            abs_errors[m] = round(abs(diff), 4)
            sigma = max(1e-4, float(uncertainties.get(m, 1.0)))
            z = diff / sigma
            z_scores[m] = round(z, 4)

            sq_err_sum += diff ** 2
            abs_err_sum += abs(diff)
            count += 1

    mse = round(sq_err_sum / max(1, count), 4)
    mae = round(abs_err_sum / max(1, count), 4)
    max_z = round(max((abs(z) for z in z_scores.values()), default=0.0), 4)

    surprise = round(max(0.0, min(1.0, 1.0 - math.exp(-max_z / 3.0))), 4)
    rupture_warranted = bool(max_z >= 3.0 or surprise >= 0.8)

    return {
        "residuals": residuals,
        "abs_errors": abs_errors,
        "z_scores": z_scores,
        "mse": mse,
        "mae": mae,
        "max_z": max_z,
        "epistemic_surprise": surprise,
        "rupture_warranted": rupture_warranted,
    }


def record_transition_prediction(predicted: WorldState | dict[str, Any]) -> None:
    """Store the most recent predicted state S_hat_{t+1} in brain_kv."""
    p_dict = _to_dict(predicted)
    brain_kv.kv_set_json("world_model:last_prediction", p_dict)


def record_transition_observation(actual: WorldState | dict[str, Any]) -> dict[str, Any]:
    """Ingest actual observation S_{t+1}, evaluate against stored prediction, and update epistemic surprise."""
    last_pred = brain_kv.kv_get_json("world_model:last_prediction", None)
    a_dict = _to_dict(actual)

    if last_pred is None:
        return {"status": "no_prior_prediction", "observation": a_dict}

    eval_res = evaluate_prediction(last_pred, a_dict)

    surprise = eval_res["epistemic_surprise"]
    surprise_ema = float(brain_kv.kv_get_json("world_model:surprise_ema", 0.0))
    surprise_ema = max(0.0, min(1.0, 0.8 * surprise_ema + 0.2 * surprise))
    brain_kv.kv_set_json("world_model:surprise_ema", surprise_ema)

    if eval_res["rupture_warranted"]:
        rupture_event = {
            "source": "transition_residual",
            "reason": f"Transition prediction divergence (max z={eval_res['max_z']:.2f})",
            "surprise": surprise,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        rupture_log = brain_kv.kv_get_json("world_model:rupture_log", []) or []
        rupture_log.append(rupture_event)
        brain_kv.kv_set_json("world_model:rupture_log", rupture_log[-50:])
        eval_res["rupture_event"] = rupture_event

    return eval_res


