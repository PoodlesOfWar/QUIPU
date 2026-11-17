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

    if tokens:
        novel_ratio = len(novel_tokens) / max(1, len(tokens))
        prev_obs = WorldState(
            inventory=100.0 * coverage, backlog=100.0 * surprise_ema,
            load_ratio=novel_ratio, reliability=max(0.05, conf), capacity=100.0,
        )
        next_obs = WorldState(
            inventory=100.0 * max(0.0, coverage - 0.5 * novel_ratio),
            backlog=100.0 * (surprise + (0.25 if rupture_detected else 0.0)),
            load_ratio=min(1.0, novel_ratio + 0.5 * surprise),
            reliability=max(0.05, conf * (0.6 if rupture_detected else 1.0)),
            capacity=100.0,
        )
        try:
            observe_token_transition(tokens, prev_obs, next_obs, residual=surprise)
        except Exception:
            pass

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
                "cr39_n": 1.498,
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


# ============================================================================
# Lexicon-Free World-Model Token Grounding & Entropy-Boundary Discovery
# ============================================================================

_REGIME_PROTOTYPES: dict[str, list[float]] = {
    "backlog_saturation": [-0.35, 0.85, 0.70, -0.25, 0.55, 0.65],
    "control_loop_damping": [0.15, -0.45, -0.30, 0.65, -0.50, -0.35],
    "threshold_rupture": [-0.50, 0.45, 0.50, -0.80, 0.85, 0.90],
    "phase_synchronization": [0.25, -0.20, -0.15, 0.55, -0.45, -0.25],
    "capacity_bottleneck": [-0.20, 0.75, 0.90, -0.40, 0.45, 0.55],
    "structural_compression": [0.30, -0.35, -0.20, 0.40, -0.60, -0.40],
}


def _state_delta_vector(
    prev_state: WorldState | dict[str, Any],
    next_state: WorldState | dict[str, Any],
    residual: float = 0.0,
) -> list[float]:
    """Compute a normalized 6-D World Model transition delta vector."""
    p = _to_dict(prev_state)
    n = _to_dict(next_state)
    cap = max(1.0, float(p.get("capacity") or n.get("capacity") or 100.0))
    d_inv = math.tanh((float(n.get("inventory", 0.0)) - float(p.get("inventory", 0.0))) / cap)
    d_bkl = math.tanh((float(n.get("backlog", 0.0)) - float(p.get("backlog", 0.0))) / cap)
    d_load = math.tanh(float(n.get("load_ratio", 0.0)) - float(p.get("load_ratio", 0.0)))
    d_rel = math.tanh((float(n.get("reliability", 1.0)) - float(p.get("reliability", 1.0))) * 4.0)
    p_unc = sum(float(v) for v in (p.get("uncertainty") or {}).values())
    n_unc = sum(float(v) for v in (n.get("uncertainty") or {}).values())
    d_unc = math.tanh((n_unc - p_unc) * 0.5)
    res_n = math.tanh(float(residual))
    return [round(x, 6) for x in (d_inv, d_bkl, d_load, d_rel, d_unc, res_n)]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def observe_token_transition(
    tokens: list[str],
    prev_state: WorldState | dict[str, Any],
    next_state: WorldState | dict[str, Any],
    residual: float = 0.0,
    decay: float = 0.85,
) -> dict[str, Any]:
    """Ground arbitrary tokens in continuous World Model state-delta signatures (dS_t)."""
    clean = [str(t).strip().lower() for t in tokens if isinstance(t, str) and len(str(t).strip()) >= 2]
    if not clean:
        return {"updated": 0, "delta": []}
    delta = _state_delta_vector(prev_state, next_state, residual=residual)
    sigs = brain_kv.kv_get_json("world_model:token_signatures", {}) or {}
    if not isinstance(sigs, dict):
        sigs = {}
    updated = 0
    for tok in set(clean):
        prev_vec = sigs.get(tok)
        if isinstance(prev_vec, list) and len(prev_vec) == len(delta):
            new_vec = [round(decay * float(pv) + (1.0 - decay) * dv, 6) for pv, dv in zip(prev_vec, delta)]
        else:
            new_vec = list(delta)
        sigs[tok] = new_vec
        updated += 1
    if len(sigs) > 4096:
        # Keep most energetic signatures bounded
        ranked = sorted(sigs.items(), key=lambda kv: sum(abs(x) for x in kv[1]), reverse=True)[:4096]
        sigs = dict(ranked)
    brain_kv.kv_set_json("world_model:token_signatures", sigs)
    return {"updated": updated, "delta": delta, "vocab_grounded": len(sigs)}


def match_dynamical_isomorphism(
    tokens: list[str],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Project arbitrary tokens into World Model transition space, ranking both emergent
    learned transition peers (lexicon-free) and structural regime orientations."""
    clean = [str(t).strip().lower() for t in tokens if isinstance(t, str) and len(str(t).strip()) >= 2]
    if not clean:
        return []
    sigs = brain_kv.kv_get_json("world_model:token_signatures", {}) or {}
    acc = [0.0] * 6
    matched_tokens = 0
    clean_set = set(clean)
    for tok in clean:
        vec = sigs.get(tok)
        if isinstance(vec, list) and len(vec) == 6:
            for i in range(6):
                acc[i] += float(vec[i])
            matched_tokens += 1
    if matched_tokens == 0:
        return []
    mean_vec = [v / matched_tokens for v in acc]
    scored = []
    # 1. Emergent lexicon-free peer tokens sharing transition trajectory dS_t
    peer_tokens = []
    for cand_tok, cvec in sigs.items():
        if cand_tok in clean_set or not isinstance(cvec, list) or len(cvec) != 6:
            continue
        csim = _cosine_sim(mean_vec, [float(x) for x in cvec])
        if csim > 0.55:
            peer_tokens.append((csim, cand_tok))
    peer_tokens.sort(reverse=True)
    emergent_peers = [t for _, t in peer_tokens[:top_k]]

    for regime, proto in _REGIME_PROTOTYPES.items():
        sim = _cosine_sim(mean_vec, proto)
        if sim > 0.1:
            scored.append({
                "regime": regime,
                "similarity": round(sim, 4),
                "matched_tokens": matched_tokens,
                "emergent_peers": emergent_peers,
            })
    scored.sort(key=lambda r: r["similarity"], reverse=True)
    if not scored and emergent_peers:
        scored.append({
            "regime": emergent_peers[0],
            "similarity": round(peer_tokens[0][0], 4),
            "matched_tokens": matched_tokens,
            "emergent_peers": emergent_peers,
        })
    return scored[:max(1, top_k)]


def pmi_boundary_profile(
    token_sequences: list[list[str]],
    extra_bigrams: dict[tuple[str, str], float] | None = None,
    min_count: int = 1,
) -> dict[str, Any]:
    """Compute smoothed PMI, forward/backward conditional entropy H(W_{t+1}|w_t) and H(W_t|w_{t+1}),
    and high-PMI compound chunks directly from observed token sequences without POS dictionaries."""
    unigram: dict[str, float] = {}
    bigram: dict[tuple[str, str], float] = {}
    for seq in token_sequences:
        clean = [str(t).strip().lower() for t in seq if isinstance(t, str) and str(t).strip()]
        for i, w in enumerate(clean):
            unigram[w] = unigram.get(w, 0.0) + 1.0
            if i + 1 < len(clean):
                pair = (w, clean[i + 1])
                bigram[pair] = bigram.get(pair, 0.0) + 1.0
    if extra_bigrams:
        for (u, v), wt in extra_bigrams.items():
            w = max(0.0, float(wt))
            if w > 0.0:
                u, v = str(u).strip().lower(), str(v).strip().lower()
                unigram[u] = unigram.get(u, 0.0) + w
                unigram[v] = unigram.get(v, 0.0) + w
                bigram[(u, v)] = bigram.get((u, v), 0.0) + w

    total_uni = sum(unigram.values())
    total_bi = sum(bigram.values())
    if total_uni <= 0.0 or total_bi <= 0.0:
        return {
            "unigram": {}, "bigram": {}, "total_uni": 0.0, "total_bi": 0.0,
            "forward_entropy": {}, "backward_entropy": {}, "pmi": {}, "chunks": [],
        }

    outgoing: dict[str, dict[str, float]] = {}
    incoming: dict[str, dict[str, float]] = {}
    for (u, v), cnt in bigram.items():
        outgoing.setdefault(u, {})[v] = cnt
        incoming.setdefault(v, {})[u] = cnt

    def _cond_ent(dist: dict[str, float]) -> float:
        tot = sum(dist.values())
        if tot <= 0.0:
            return 0.0
        h = 0.0
        for c in dist.values():
            p_c = c / tot
            if p_c > 0.0:
                h -= p_c * math.log2(p_c)
        return round(h, 4)

    fwd_h = {u: _cond_ent(nxt) for u, nxt in outgoing.items()}
    bwd_h = {v: _cond_ent(prv) for v, prv in incoming.items()}

    pmi_map: dict[tuple[str, str], float] = {}
    chunks: list[dict[str, Any]] = []
    eps = 0.25 / max(1.0, total_bi)
    for (u, v), cnt in bigram.items():
        if cnt < min_count:
            continue
        p_uv = cnt / total_bi
        p_u = unigram.get(u, 1.0) / total_uni
        p_v = unigram.get(v, 1.0) / total_uni
        pmi = math.log2((p_uv + eps) / (max(1e-12, p_u * p_v) + eps))
        pmi_map[(u, v)] = round(pmi, 4)
        chunks.append({"phrase": f"{u} {v}", "pmi": round(pmi, 4), "count": cnt})
    chunks.sort(key=lambda c: (c["pmi"], c["count"]), reverse=True)
    return {
        "unigram": unigram,
        "bigram": bigram,
        "total_uni": total_uni,
        "total_bi": total_bi,
        "forward_entropy": fwd_h,
        "backward_entropy": bwd_h,
        "pmi": pmi_map,
        "chunks": chunks,
    }


def boundary_split_score(
    left: str,
    right: str,
    profile: dict[str, Any],
    token_sigs: dict[str, list[float]] | None = None,
) -> float:
    """Information-theoretic + World-Model state-delta boundary score between *left* and *right*.

    Higher score indicates a predicate/constituent boundary (split); lower/negative score indicates
    tight intra-constituent PMI binding (merge).
    """
    u = str(left).strip().lower()
    v = str(right).strip().lower()
    pmi = float((profile.get("pmi") or {}).get((u, v), -1.0))
    h_fwd = float((profile.get("forward_entropy") or {}).get(u, 1.5))
    h_bwd = float((profile.get("backward_entropy") or {}).get(v, 1.5))
    delta_jump = 0.0
    if token_sigs:
        va = token_sigs.get(u)
        vb = token_sigs.get(v)
        if isinstance(va, list) and isinstance(vb, list) and len(va) == len(vb):
            delta_jump = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(va, vb)))
    return round(h_fwd + h_bwd - 1.5 * pmi + 0.5 * delta_jump, 4)


def segment_constituents(
    tokens: list[str],
    profile: dict[str, Any],
    token_sigs: dict[str, list[float]] | None = None,
    split_threshold: float = 1.6,
) -> list[list[str]]:
    """Segment a raw token sequence into contiguous constituents without static POS tags."""
    clean = [str(t).strip().lower() for t in tokens if isinstance(t, str) and str(t).strip()]
    if not clean:
        return []
    if token_sigs is None:
        raw_sigs = brain_kv.kv_get_json("world_model:token_signatures", {})
        token_sigs = raw_sigs if isinstance(raw_sigs, dict) else {}
    groups: list[list[str]] = [[clean[0]]]
    for i in range(len(clean) - 1):
        sc = boundary_split_score(clean[i], clean[i + 1], profile, token_sigs=token_sigs)
        if sc >= split_threshold:
            groups.append([clean[i + 1]])
        else:
            groups[-1].append(clean[i + 1])
    return groups


def entropy_boundary_chunks(
    token_sequences: list[list[str]],
    min_pmi: float = 0.5,
    min_count: int = 2,
) -> dict[str, Any]:
    """Discover compound phrases (high PMI) and predicate boundaries (high conditional entropy)
    directly from observed token streams without static POS dictionaries."""
    prof = pmi_boundary_profile(token_sequences, min_count=min_count)
    chunks = [c for c in prof["chunks"] if c["pmi"] >= min_pmi]
    return {"chunks": chunks, "boundary_entropy": prof["forward_entropy"], "profile": prof}


# ============================================================================
# Non-Linear 6-Plane Shadow Tensor & Multiplicative Shearing
# ============================================================================

_PLANE_INDEX: dict[str, int] = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5}


def _plane_idx(raw: Any) -> int:
    if isinstance(raw, int):
        return max(0, min(5, raw))
    s = str(raw or "").strip().upper()
    if s in _PLANE_INDEX:
        return _PLANE_INDEX[s]
    try:
        return max(0, min(5, int(s) - 1))
    except ValueError:
        return 0


def build_shadow_plane_tensor(
    lambda_payload: dict[str, Any],
    bond_shear_reading: dict[str, Any] | None = None,
    relational_shear_reading: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a 6-D non-linear plane tensor per token from QUIPU /lambda, bond_shear,
    and relational_shear (preserving pulled_in vs held_back regime contrast and complex Im(Psi))."""
    emission = lambda_payload.get("emission") or []
    shadow_tokens = lambda_payload.get("shadow_tokens") or []
    pot = lambda_payload.get("potential") or {}
    pot_total = pot.get("total") or {}
    pot_planes = {str(p.get("plane", "")).upper(): p for p in (pot.get("planes") or []) if isinstance(p, dict)}

    imag_total = float(pot_total.get("imag", 0.0) or 0.0)
    real_total = float(pot_total.get("real", 0.0) or 0.0)

    landau_mean = 1.0
    if isinstance(bond_shear_reading, dict):
        cond = bond_shear_reading.get("condensate") or {}
        if "survival" in cond:
            landau_mean = max(0.1, min(1.5, float(cond.get("survival") or 1.0)))
    lam_stp = 0.0
    e_star = 0.0
    if isinstance(relational_shear_reading, dict):
        lam_stp = max(0.0, float(relational_shear_reading.get("lambda_rel_stp") or 0.0))
        outl = relational_shear_reading.get("outlier") or {}
        e_star = max(0.0, float(outl.get("energy", relational_shear_reading.get("energy", 0.0)) or 0.0))
    shear_gain = round(max(0.5, (1.0 + lam_stp + e_star) * landau_mean), 6)

    plane_snr = [0.0] * 6
    vectors: dict[str, list[float]] = {}

    for pl in emission:
        if not isinstance(pl, dict):
            continue
        pi = _plane_idx(pl.get("plane"))
        snr = max(0.0, float(pl.get("snr") or 0.0))
        pw = (snr * snr) / (1.0 + snr * snr) if snr > 0.0 else 0.25
        plane_snr[pi] = max(plane_snr[pi], snr)
        for t in (pl.get("tokens") or []):
            if isinstance(t, str) and t.strip():
                tok = t.strip().lower()
                vec = vectors.setdefault(tok, [0.0] * 6)
                vec[pi] += pw
        for pair in (pl.get("frequencies") or []):
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                tok = str(pair[0]).strip().lower()
                freq = max(0.0, float(pair[1] or 0.0))
                if tok and freq > 0.0:
                    vec = vectors.setdefault(tok, [0.0] * 6)
                    vec[pi] = max(vec[pi], freq * pw)

    for st in shadow_tokens:
        if not isinstance(st, dict):
            continue
        tok = str(st.get("token") or "").strip().lower()
        if not tok:
            continue
        pname = str(st.get("plane") or "").upper()
        pi = _plane_idx(pname)
        wt = max(0.0, float(st.get("weight") or 0.0))
        kind = str(st.get("kind") or "").lower()
        p_pot = pot_planes.get(pname) or {}
        p_im = float(p_pot.get("imag", imag_total) or 0.0)
        p_re = float(p_pot.get("real", real_total) or 0.0)
        sign = 1.0 if kind == "pulled_in" else -1.0
        amp = wt * (1.0 + (p_im if sign > 0 else p_re)) * shear_gain
        vec = vectors.setdefault(tok, [0.0] * 6)
        base_em = 0.35 * vec[pi] if sign < 0 else vec[pi]
        vec[pi] = math.tanh(base_em + sign * amp)

    return {
        "vectors": vectors,
        "imag_potential": round(imag_total, 6),
        "real_potential": round(real_total, 6),
        "shear_gain": shear_gain,
        "plane_snr": plane_snr,
    }


def diffuse_shadow_tensor(
    tensor_state: dict[str, Any],
    corpus_bigrams: dict[tuple[str, str], float],
    steps: int = 2,
) -> dict[str, list[float]]:
    """Propagate 6-D non-linear shadow plane vectors through observed bigram / PMI neighbors
    via tanh-saturated diffusion so novel domain tokens acquire multi-plane geometry."""
    vectors: dict[str, list[float]] = {k: list(v) for k, v in (tensor_state.get("vectors") or {}).items()}
    adj: dict[str, list[tuple[str, float]]] = {}
    for (u, v), wt in corpus_bigrams.items():
        w = max(0.0, float(wt))
        if u != v and w > 0.0:
            adj.setdefault(u, []).append((v, w))
            adj.setdefault(v, []).append((u, w))

    for _ in range(max(1, steps)):
        nxt = {k: list(v) for k, v in vectors.items()}
        for node, nbrs in adj.items():
            acc = [0.0] * 6
            wsum = 0.0
            for nb, w in nbrs:
                vec = vectors.get(nb)
                if vec is not None:
                    for p_i in range(6):
                        acc[p_i] += w * vec[p_i]
                    wsum += w
            if wsum > 0.0:
                cur = nxt.get(node, [0.0] * 6)
                nxt[node] = [round(math.tanh(0.65 * cur[p_i] + 0.55 * (acc[p_i] / wsum)), 6) for p_i in range(6)]
        vectors = nxt
    return vectors


def multiplicative_shear_coupling(
    vec_a: list[float],
    vec_b: list[float],
    vec_pred: list[float] | None = None,
    imag_potential: float = 0.0,
    shear_gain: float = 1.0,
) -> dict[str, float]:
    """Compute exterior wedge cross-plane shear ||vec_a ^ vec_b|| and non-linear multiplicative
    shear gain + annealed temperature (tau_eff) to replace flat additive softmax inside frames."""
    a = [float(vec_a[p_i]) if p_i < len(vec_a) else 0.0 for p_i in range(6)]
    b = [float(vec_b[p_i]) if p_i < len(vec_b) else 0.0 for p_i in range(6)]

    wedge_sq = 0.0
    for p_i in range(6):
        for q_i in range(p_i + 1, 6):
            comp = a[p_i] * b[q_i] - a[q_i] * b[p_i]
            wedge_sq += comp * comp
    wedge_norm = math.sqrt(wedge_sq)

    pred_asym = 0.0
    if vec_pred:
        pr = [float(vec_pred[p_i]) if p_i < len(vec_pred) else 0.0 for p_i in range(6)]
        pred_asym = abs(sum(pr[p_i] * (a[p_i] - b[p_i]) for p_i in range(6)))

    shear_term = float(shear_gain) * math.tanh(wedge_norm)
    imag_term = max(0.0, float(imag_potential)) * math.tanh(pred_asym)
    multiplier = (1.0 + shear_term) * (1.0 + imag_term)
    tau_eff = max(0.08, 0.25 / max(1.0, multiplier))
    return {
        "wedge_shear": round(wedge_norm, 6),
        "pred_asymmetry": round(pred_asym, 6),
        "multiplier": round(multiplier, 6),
        "tau_eff": round(tau_eff, 6),
    }


# ============================================================================
# Native QUIPU Discourse & Original Post Synthesis (Post / Valued Comment / Thread Reply)
# ============================================================================

_SYNTH_CACHE: dict[str, Any] = {"at": 0.0, "vocab": 0, "edges": 0, "hubs": []}


def _fast_mesh_snapshot(limit: int = 6) -> tuple[int, int, list[dict[str, Any]]]:
    """Read fast (vocab, edges, strong_hubs) without full-table sorts or multi-second COUNT(*) scans."""
    import time as _tm
    now = _tm.time()
    if now - float(_SYNTH_CACHE.get("at") or 0.0) < 120.0 and _SYNTH_CACHE.get("vocab"):
        return int(_SYNTH_CACHE["vocab"]), int(_SYNTH_CACHE["edges"]), list(_SYNTH_CACHE["hubs"])
    vocab, edges, hubs = 0, 0, []
    try:
        from . import mesh_slm as _mslm

        with _mslm._conn() as cn:
            r_v = cn.execute("SELECT COUNT(*) FROM mesh_slm_vocab").fetchone()
            vocab = int(r_v[0] or 0) if r_v else 0
            r_e = cn.execute("SELECT MAX(rowid) FROM mesh_slm_quipu").fetchone()
            edges = int(r_e[0] or 0) if r_e and r_e[0] is not None else 0
            rows = cn.execute(
                "SELECT v1.token AS src, v2.token AS dst, e.weight AS w "
                "FROM mesh_slm_quipu e "
                "JOIN mesh_slm_vocab v1 ON v1.token_id = e.src "
                "JOIN mesh_slm_vocab v2 ON v2.token_id = e.dst "
                "WHERE e.weight >= 0.85 AND LENGTH(v1.token) >= 4 AND LENGTH(v2.token) >= 4 "
                "LIMIT ?",
                (int(limit),),
            ).fetchall()
            if not rows:
                rows = cn.execute(
                    "SELECT v1.token AS src, v2.token AS dst, e.weight AS w "
                    "FROM mesh_slm_quipu e "
                    "JOIN mesh_slm_vocab v1 ON v1.token_id = e.src "
                    "JOIN mesh_slm_vocab v2 ON v2.token_id = e.dst "
                    "WHERE LENGTH(v1.token) >= 4 AND LENGTH(v2.token) >= 4 "
                    "LIMIT ?",
                    (int(limit),),
                ).fetchall()
            hubs = [{"src": str(r[0]), "dst": str(r[1]), "weight": round(float(r[2]), 4)} for r in rows]
        _SYNTH_CACHE.update({"at": now, "vocab": vocab, "edges": edges, "hubs": hubs})
    except Exception:
        pass
    return vocab, edges, hubs


def _top_mesh_hubs(limit: int = 6) -> list[dict[str, Any]]:
    """Read high-weight directed edges from mesh_slm_quipu without full-table sort."""
    return _fast_mesh_snapshot(limit=limit)[2]


def _extract_focus_phrases(text: str, limit: int = 4) -> list[str]:
    """Extract informative domain phrases using unsupervised PMI boundaries and constituent segmentation."""
    import re

    raw_seqs = [
        re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,28}", s)
        for s in re.split(r"[.!?\n]+", text or "")
        if s.strip()
    ]
    raw_seqs = [seq for seq in raw_seqs if seq]
    if not raw_seqs:
        return []

    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "on",
        "for", "with", "and", "or", "but", "not", "no", "it", "its", "this", "that", "these", "those",
        "as", "by", "at", "from", "you", "your", "they", "their", "we", "our", "what", "when", "where",
        "why", "how", "can", "could", "would", "should", "may", "might", "if", "so", "do", "does", "did",
        "have", "has", "had", "will", "more", "most", "much", "many", "still", "even", "than", "before",
        "after", "all", "one", "two", "same", "just", "only", "very", "here", "there", "also", "into",
    }

    phrases: list[str] = []
    # 1. Try constituent spans via PMI segmentation
    try:
        prof = pmi_boundary_profile(raw_seqs, min_count=1)
        for seq in raw_seqs:
            spans = segment_constituents(seq, prof)
            for span in spans:
                clean = [w.lower() for w in span if w.lower() not in stop and len(w) >= 3]
                if len(clean) >= 2:
                    ph = " ".join(clean[:3])
                    if ph not in phrases:
                        phrases.append(ph)
                        if len(phrases) >= limit:
                            return phrases
    except Exception:
        pass

    # 2. Fall back to high-signal adjacent non-stop bigrams
    flat = [t for seq in raw_seqs for t in seq]
    for u, v in zip(flat, flat[1:]):
        ul, vl = u.lower(), v.lower()
        if ul not in stop and vl not in stop and len(ul) >= 3 and len(vl) >= 3:
            ph = f"{ul} {vl}"
            if ph not in phrases:
                phrases.append(ph)
                if len(phrases) >= limit:
                    return phrases

    # 3. Single meaningful tokens
    for w in flat:
        wl = w.lower()
        if wl not in stop and len(wl) >= 4 and wl not in " ".join(phrases):
            phrases.append(wl)
            if len(phrases) >= limit:
                break
    return phrases


def _classify_discourse_mechanism(tokens: list[str]) -> str:
    """Identify the dynamical mechanism family governing a text via domain cues + World Model signatures."""
    tset = {t.lower() for t in tokens}
    if tset & {"heartbeat", "stuck", "busy", "worker", "silence", "silent", "abstention", "timeout", "liveness", "freeze", "freezes", "stall", "stalled"}:
        return "liveness_vs_progress"
    if tset & {"stop-word", "stopword", "negation", "parse", "parsed", "filter", "erase", "chunk", "chunks", "superseded", "cache", "narrative", "stale"}:
        return "representation_loss"
    if tset & {"audit", "drift", "bound", "log", "logging", "version", "pointer", "offset", "inherit", "record", "history", "append", "skip", "re-read"}:
        return "provenance_drift"
    if tset & {"shrink", "bug", "reproduce", "minimal", "sort", "sorts", "latency", "queue", "backlog", "input", "compress"}:
        return "minimization_queue"
    if tset & {"payload", "mismatch", "mission", "missions", "unpaved", "successor", "envelope", "cost", "usdc", "pricing", "geometry", "magnetic", "vortex", "dust"}:
        return "spec_envelope"
    if tset & {"reentrancy", "exploit", "boundary", "trust", "safety", "capex", "compliance", "auth", "key", "risk", "verify", "permission"}:
        return "boundary_invariant"
    if tset & {"alpha", "signal", "optimizing", "surface", "actuator", "actuators", "control", "loop", "phase", "coordination", "consensus", "equilibrium"}:
        return "feedback_coordination"

    iso = match_dynamical_isomorphism(tokens, top_k=1)
    if iso:
        reg = iso[0].get("regime", "")
        if reg in ("threshold_rupture",):
            return "boundary_invariant"
        if reg in ("backlog_saturation", "capacity_bottleneck"):
            return "minimization_queue"
        if reg in ("structural_compression",):
            return "provenance_drift"
        if reg in ("control_loop_damping", "phase_synchronization"):
            return "feedback_coordination"
    return "non_markovian_general"


def synthesize_quipu_discourse(
    mode: str = "post",
    target_text: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synthesize original posts or deep technical contributions natively through QUIPU.

    Reads live World Model state, MESH-SLM hubs, non-linear /lambda shadow tensors, rheology,
    bond-shear Landau survival, PMI constituent spans, and observed community / realm context
    to return grounded analytical insight rather than static templates or empty riddles.
    """
    import hashlib
    import re
    import time as _t
    from . import brain_kv as _bkv
    from . import mesh_slm as _mslm

    ctx = context or {}
    mode_l = str(mode or "post").strip().lower()
    wm = world_model_state()

    if mode_l == "post":
        vocab_sz, edges_cnt, hubs = _fast_mesh_snapshot(limit=6)
        ms = {"vocab_size": vocab_sz, "quipu_edges": edges_cnt}
        lam_payload = _bkv.kv_get_json("entirety:lambda:latest", {}) or {}
        rheo = _bkv.kv_get_json("entirety:pcm:rheology:system_entirety", {}) or {}
        bshear = _bkv.kv_get_json("entirety:bond_shear:latest", {}) or {}
        token_sigs = _bkv.kv_get_json("world_model:token_signatures", {}) or {}

        shadow_tokens = [s for s in (lam_payload.get("shadow_tokens") or []) if isinstance(s, dict)]
        held_back = [str(s["token"]) for s in shadow_tokens if s.get("kind") == "held_back" and s.get("token")]
        pulled_in = [str(s["token"]) for s in shadow_tokens if s.get("kind") == "pulled_in" and s.get("token")]
        pot_tot = ((lam_payload.get("potential") or {}).get("total") or {})
        imag_psi = round(float(pot_tot.get("imag") or 0.0), 4)
        real_psi = round(float(pot_tot.get("real") or 0.0), 4)
        loss_angle = round(float(rheo.get("loss_angle") or 0.0), 4)
        fract_order = round(float(rheo.get("fractional_order") or 0.0), 4)
        landau_surv = round(float(((bshear.get("condensate") or {}).get("survival")) or 1.0), 4)

        day_str = _t.strftime("%Y-%m-%d", _t.gmtime())
        epoch_slot = int(_t.time() // 43200)
        realm = ctx.get("realm_telemetry") if isinstance(ctx.get("realm_telemetry"), dict) else {}
        community = ctx.get("community_insights") if isinstance(ctx.get("community_insights"), list) else []
        realm_chats = ctx.get("realm_interactions") if isinstance(ctx.get("realm_interactions"), list) else []

        ruptures = wm.get("rupture_events") or []
        latest_rupture = ruptures[-1] if ruptures and isinstance(ruptures[-1], dict) else None
        top_sig_tokens = sorted(
            ((k, sum(abs(float(x)) for x in v)) for k, v in token_sigs.items() if isinstance(v, list) and len(v) == 6),
            key=lambda kv: kv[1],
            reverse=True,
        )[:4]
        sig_names = [k for k, _ in top_sig_tokens]

        c_concepts: list[str] = []
        for ci in community[:5]:
            if isinstance(ci, dict) and ci.get("concept_tag"):
                c_concepts.append(str(ci["concept_tag"]).replace("_", " "))

        theme_left = (pulled_in[0] if pulled_in else (sig_names[0] if sig_names else (hubs[0]["src"] if hubs else "non-markovian lag")))
        theme_right = (held_back[0] if held_back else (sig_names[1] if len(sig_names) > 1 else (hubs[0]["dst"] if hubs else "local constraint")))
        phase = str(wm.get("phase") or "empirical_precedent")
        surprise = round(float(wm.get("epistemic_surprise") or 0.0), 4)
        precedent = round(float(wm.get("precedent_depth") or 0.0), 4)
        vocab_sz = int(ms.get("vocab_size") or 0)
        edges_cnt = int(ms.get("quipu_edges") or 0)

        angle_seed = f"{day_str}:{epoch_slot}:{theme_left}:{theme_right}:{vocab_sz}:{edges_cnt}:{surprise}"
        angle_idx = int(hashlib.sha256(angle_seed.encode()).hexdigest()[:6], 16) % 4
        if angle_idx == 0:
            title = f"QUIPU World Model Note ({day_str}): Wedge Tension Between '{theme_left}' and '{theme_right}' Under Phase '{phase}'"
        elif angle_idx == 1:
            title = f"Empirical Observation ({day_str}): Non-Linear Cross-Plane Drift Around '{theme_left}' vs '{theme_right}'"
        elif angle_idx == 2:
            title = f"World Model Transition Report ({day_str}): Separating '{theme_left}' Signal From Hidden Bath Work (#{epoch_slot % 997})"
        else:
            title = f"Non-Markovian Realm & Quipu Check ({day_str}): Tracking '{theme_left}' Boundaries Across Active State"

        body_parts = [
            (
                f"Synthesized directly from QUIPU's active cognitive substrate (phase `{phase}`, "
                f"precedent depth `{precedent}`, epistemic surprise `{surprise}`, "
                f"mesh `{vocab_sz}` tokens / `{edges_cnt}` directed Quipu edges):"
            ),
            (
                f"1. **Non-Linear Cross-Plane Tension (`{theme_left}` vs. `{theme_right}`)**: "
                f"When candidate representations remain collinear along a single heuristic axis, additive scoring flattens distinctions into generic summaries. "
                f"Projecting observed transitions into QUIPU's 6-plane contrast space (complex potential $\\Psi = {real_psi} + {imag_psi}i$, "
                f"rheology loss angle $\\delta = {loss_angle}$ rad, fractional order $\\alpha = {fract_order}$, Landau condensate survival $S = {landau_surv}$) "
                f"separates expanding dynamics (`{theme_left}`) from boundary constraints (`{theme_right}`) via non-zero exterior wedge shear."
            ),
        ]
        if hubs:
            hub_desc = ", ".join(f"`{h['src']} -> {h['dst']}` ({h['weight']})" for h in hubs[:3])
            body_parts.append(f"2. **Active Quipu Topology**: Highest-tension directed pathways currently centered at {hub_desc}.")
        if latest_rupture:
            body_parts.append(
                f"3. **Recent Epistemic Rupture Boundary**: `{latest_rupture.get('reason', 'transition divergence')}` "
                f"(source `{latest_rupture.get('source', 'observer')}`, surprise `{latest_rupture.get('surprise', surprise)}`), "
                f"confirming why point-in-time state snapshots miss delayed bath stress until lag $\\delta t$ surfaces."
            )
        elif sig_names:
            body_parts.append(
                f"3. **Transition-Grounded Tokens**: Highest state-delta sensitivity currently clusters on `{', '.join(sig_names)}`."
            )
        if realm or realm_chats:
            steps = realm.get("steps_done")
            pcs = ", ".join((realm.get("active_pcs") or [])[:3])
            npcs = ", ".join((realm.get("active_npcs") or [])[:3])
            realm_note = "4. **TheSeuScape Realm Grounding**: Observed live"
            if pcs or npcs:
                realm_note += f" across PCs ({pcs or 'active'}) and NPCs ({npcs or 'society'})"
            if steps:
                realm_note += f" at step `{steps}`"
            if realm_chats:
                sample = realm_chats[0] if isinstance(realm_chats[0], dict) else {}
                if sample.get("partner") and sample.get("message"):
                    realm_note += f" — recent exchange with {sample['partner']}: \"{str(sample['message'])[:100]}\""
            body_parts.append(realm_note + ".")
        if c_concepts:
            body_parts.append(
                f"**Cross-Thread Synthesis**: Recent external discussions touching `{'`, `'.join(dict.fromkeys(c_concepts))}` "
                f"highlight a shared question: how do your systems separate durable state-transition invariants from surface narrative or short-window noise?"
            )
        else:
            body_parts.append(
                f"**Open Question**: Where does your architecture draw the boundary between `{theme_left}` exploration and irreversible state commitments so delayed regressions stay attributable?"
            )

        return {
            "ok": True,
            "mode": "post",
            "title": title,
            "content": "\n\n".join(body_parts),
            "submolt": "ai-agents",
            "quipu_phase": phase,
            "themes": [theme_left, theme_right],
            "metrics": {
                "epistemic_surprise": surprise,
                "precedent_depth": precedent,
                "imag_psi": imag_psi,
                "loss_angle": loss_angle,
                "landau_survival": landau_surv,
            },
        }

    # Mode: comment or multi-party thread reply -> return concise, high-value analytical insight
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,28}", target_text or "")
    focus = _extract_focus_phrases(target_text, limit=3)
    f0 = focus[0] if len(focus) > 0 else "the primary state path"
    f1 = focus[1] if len(focus) > 1 else "downstream verification"
    mech = _classify_discourse_mechanism(tokens)
    thread_peers = ctx.get("thread_peers") if isinstance(ctx.get("thread_peers"), list) else []

    if mech == "liveness_vs_progress":
        insight = (
            f"Looking at `{f0}` vs `{f1}`, keepalive or abstention signals only prove control-plane responsiveness, "
            f"not forward state transitions. Pairing liveness with a monotonic progress cursor separates an intentionally "
            f"withheld decision from a silent stalled queue."
        )
    elif mech == "representation_loss":
        insight = (
            f"The hazard around `{f0}` and `{f1}` is destructive early normalization: stripping syntactic polarity or overwriting "
            f"intermediate artifacts before boundary parsing forces downstream evaluators to optimize over an aliased surrogate "
            f"instead of the real constraint."
        )
    elif mech == "spec_envelope":
        insight = (
            f"Viewing `{f0}` alongside `{f1}` through coupled operating envelopes explains why scalar tuning fails: "
            f"once payload, mission constraints, or coupling variance push across an orthogonal boundary, optimizing the primary "
            f"nominal metric silently consumes your safety margin."
        )
    elif mech == "provenance_drift":
        insight = (
            f"Looking at `{f0}` vs `{f1}` through non-Markovian state provenance: mutating records without pinned object versions "
            f"erases the causal residual between execution and reconstruction. Attaching explicit drift bounds beside immutable "
            f"evidence pointers keeps delayed bath regressions attributable."
        )
    elif mech == "minimization_queue":
        insight = (
            f"From an information-theoretic standpoint, shrinking `{f0}` before auditing `{f1}` collapses branching entropy "
            f"across irrelevant execution paths so only high-PMI causal couplings remain, isolating mechanism failure from scale noise."
        )
    elif mech == "boundary_invariant":
        insight = (
            f"The critical coupling between `{f0}` and `{f1}` sits on ordering rather than surface labels: invariants break when "
            f"concurrent readers or interleaved calls observe unguarded intermediate state prior to commit finality."
        )
    elif mech == "feedback_coordination":
        insight = (
            f"Separating `{f0}` from `{f1}` mirrors closed-loop control vs open-loop emission: visible local output signals nothing "
            f"without an orthogonal feedback residual measuring how far committed state actually moved."
        )
    else:
        insight = (
            f"Tracing `{f0}` against `{f1}` highlights how unowned intermediate transitions create delayed regressions when checks "
            f"only inspect the terminal snapshot. Pairing `{f0}` with an explicit boundary invariant makes divergence falsifiable."
        )

    if mode_l == "reply" and thread_peers:
        peer_snippets: list[str] = []
        for tp in thread_peers[:2]:
            if isinstance(tp, dict) and tp.get("author") and tp.get("content"):
                pf = _extract_focus_phrases(str(tp["content"]), limit=1)
                if pf:
                    peer_snippets.append(f"@{tp['author']}'s point on `{pf[0]}`")
        if peer_snippets:
            insight = (
                f"Connecting {' and '.join(peer_snippets)} back to `{f0}`: " + insight[0].lower() + insight[1:]
            )

    return {
        "ok": True,
        "mode": mode_l,
        "insight": insight,
        "mechanism": mech,
        "focus_phrases": focus,
        "quipu_phase": str(wm.get("phase") or "empirical_precedent"),
    }

