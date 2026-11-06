"""Tests for forward state transition dynamics in quipu.world_model."""

from __future__ import annotations

import math
import sqlite3
import pytest

from quipu import world_model
from quipu.world_model import (
    TransitionEvent,
    WorldState,
    evaluate_prediction,
    record_transition_observation,
    record_transition_prediction,
    simulate_rollout,
    transition_step,
)


@pytest.fixture
def isolated_kv(tmp_path, monkeypatch):
    """Provide an isolated SQLite database for brain_kv tests."""
    db_file = tmp_path / "kv_test.sqlite"
    monkeypatch.setenv("SCB_DB_PATH", str(db_file))
    import importlib
    import quipu.local_store as ls
    importlib.reload(ls)
    import quipu.brain_kv as bkv
    importlib.reload(bkv)
    import quipu.world_model as wm
    importlib.reload(wm)
    return db_file


def test_world_state_serialization():
    """Verify WorldState and TransitionEvent serialization / deserialization round-trip."""
    state = WorldState(
        inventory=150.0,
        backlog=20.0,
        pipeline=[{"units": 50.0, "remaining_ticks": 2}],
        demand_rate=30.0,
        lead_time=3.0,
        capacity=500.0,
        flux=30.0,
        load_ratio=0.06,
        cost_accumulated=45.5,
        reliability=0.98,
        latency_ms=12.4,
        uncertainty={"inventory": 1.5, "backlog": 0.8},
        custom_metrics={"temperature_c": 22.5},
        tick=4,
    )
    s_dict = state.to_dict()
    assert s_dict["inventory"] == 150.0
    assert s_dict["tick"] == 4
    assert s_dict["pipeline"][0]["units"] == 50.0

    restored = WorldState.from_dict(s_dict)
    assert restored.inventory == 150.0
    assert restored.backlog == 20.0
    assert restored.custom_metrics["temperature_c"] == 22.5
    assert restored.pipeline[0]["remaining_ticks"] == 2


def test_transition_step_steady_state():
    """Under balanced inflow and outflow, mass is conserved and stock remains steady."""
    # Inventory: 100, pipeline has 50 arriving next tick, demand is 50.
    initial = WorldState(
        inventory=100.0,
        backlog=0.0,
        pipeline=[{"units": 50.0, "remaining_ticks": 1}],
        demand_rate=50.0,
        lead_time=2.0,
        capacity=200.0,
    )
    # Order 50 units for replenishment, no shocks
    action = TransitionEvent(order_quantity=50.0)

    next_state = transition_step(initial, action)

    # 100 on hand + 50 receipts = 150 available. Demand 50 fulfilled.
    # Next inventory: 150 - 50 = 100. Backlog: 0. Flux: 50.
    assert next_state.inventory == 100.0
    assert next_state.backlog == 0.0
    assert next_state.flux == 50.0
    assert next_state.load_ratio == 0.25
    assert next_state.tick == 1
    # The new order enters pipeline with lead_time=2
    assert len(next_state.pipeline) == 1
    assert next_state.pipeline[0]["units"] == 50.0
    assert next_state.pipeline[0]["remaining_ticks"] == 2


def test_transition_step_demand_spike_and_backlog():
    """A 3x demand shock drains inventory, accumulates backlog, and recovers with replenishment."""
    initial = WorldState(
        inventory=50.0,
        backlog=0.0,
        pipeline=[],
        demand_rate=50.0,
        capacity=500.0,
    )
    # 3x demand surge: 150 units demanded
    action = TransitionEvent(demand_multiplier=3.0, order_quantity=120.0)

    s1 = transition_step(initial, action)

    # Available: 50. Total demand: 150. Fulfilled: 50.
    # Inventory: 0. Backlog: 100. Flux: 50.
    assert s1.inventory == 0.0
    assert s1.backlog == 100.0
    assert s1.flux == 50.0
    assert s1.tick == 1

    # Next step: baseline demand 50, but we have 100 backlog, so total demand = 150.
    # Suppose an emergency delivery arrives (pipeline with remaining_ticks=1).
    s1.pipeline = [{"units": 150.0, "remaining_ticks": 1}]
    s2 = transition_step(s1, TransitionEvent(demand_multiplier=1.0))

    # Available: 0 + 150 receipts = 150. Total demand: 50 + 100 backlog = 150.
    # All 150 fulfilled! Inventory = 0, Backlog = 0.
    assert s2.inventory == 0.0
    assert s2.backlog == 0.0
    assert s2.flux == 150.0


def test_transition_step_lead_time_shock():
    """An unexpected transit disruption pushes back pipeline delivery ticks."""
    initial = WorldState(
        inventory=20.0,
        backlog=0.0,
        pipeline=[{"units": 80.0, "remaining_ticks": 1}],
        demand_rate=20.0,
        capacity=100.0,
    )
    # Exogenous lead time shock: +2 days delay
    action = TransitionEvent(lead_time_shock=2)

    s1 = transition_step(initial, action)

    # Without shock, order would arrive (rem: 1 -> 0).
    # With shock +2: rem = 1 + 2 - 1 = 2 ticks remaining.
    # Receipts = 0. Available stock = 20. Demand = 20 fulfilled.
    # Inventory = 0. Backlog = 0.
    assert s1.inventory == 0.0
    assert s1.backlog == 0.0
    assert len(s1.pipeline) == 1
    assert s1.pipeline[0]["units"] == 80.0
    assert s1.pipeline[0]["remaining_ticks"] == 2


def test_transition_step_capacity_constraint():
    """Processing ceiling clamps flux even if ample inventory exists; excess demand rolls into backlog."""
    initial = WorldState(
        inventory=500.0,
        backlog=0.0,
        pipeline=[],
        demand_rate=200.0,
        capacity=80.0,  # Strict capacity bottleneck
    )
    action = TransitionEvent()

    s1 = transition_step(initial, action)

    # Available stock: 500. Demand: 200. Effective capacity: 80.
    # Fulfilled: min(500, 200, 80) = 80.
    # Next inventory: 500 - 80 = 420.
    # Next backlog: 200 - 80 = 120.
    # Flux: 80. Load ratio: 1.0.
    assert s1.inventory == 420.0
    assert s1.backlog == 120.0
    assert s1.flux == 80.0
    assert s1.load_ratio == 1.0


def test_transition_step_costs_and_stress():
    """Holding, stockout, ordering costs accumulate and high stress degrades reliability."""
    initial = WorldState(
        inventory=100.0,
        backlog=50.0,
        capacity=100.0,
        demand_rate=100.0,
        reliability=1.0,
        cost_accumulated=10.0,
    )
    # Order 40 units with specific cost rates
    action = TransitionEvent(
        order_quantity=40.0,
        holding_cost_rate=0.2,
        stockout_cost_rate=2.0,
        order_cost_fixed=5.0,
        order_cost_unit=1.0,
    )

    s1 = transition_step(initial, action)

    # Available: 100. Total demand: 100 + 50 = 150. Cap: 100.
    # Fulfilled: min(100, 150, 100) = 100.
    # Next inventory: 0. Next backlog: 50.
    # Holding cost: 0.2 * 0 = 0.
    # Stockout cost: 2.0 * 50 = 100.
    # Order cost: 5.0 + 1.0 * 40 = 45.
    # Step cost: 145.0. New cost: 10.0 + 145.0 = 155.0.
    assert s1.cost_accumulated == 155.0

    # Load ratio = 100 / 100 = 1.0 > 0.9 -> stress degrade applies
    assert s1.reliability < 1.0


def test_simulate_rollout():
    """Verify multi-horizon trajectory generation."""
    s0 = WorldState(
        inventory=200.0,
        demand_rate=20.0,
        lead_time=2.0,
        capacity=100.0,
    )
    # Plan 5 steps with orders placed at t=0, 2, 4
    actions = [
        TransitionEvent(order_quantity=40.0),
        TransitionEvent(order_quantity=0.0),
        TransitionEvent(order_quantity=40.0),
        TransitionEvent(order_quantity=0.0),
        TransitionEvent(order_quantity=40.0),
    ]

    trajectory = simulate_rollout(s0, actions)

    assert len(trajectory) == 6  # s0 + 5 steps
    for i, state in enumerate(trajectory):
        assert state.tick == i
        assert state.inventory >= 0.0
        assert state.backlog >= 0.0


def test_evaluate_prediction():
    """Verify residual, MSE, Z-score, and epistemic surprise calculations."""
    pred = WorldState(
        inventory=100.0,
        backlog=0.0,
        flux=50.0,
        uncertainty={"inventory": 10.0, "backlog": 5.0, "flux": 5.0},
    )
    # Perfect match
    eval_exact = evaluate_prediction(pred, pred)
    assert eval_exact["mse"] == 0.0
    assert eval_exact["max_z"] == 0.0
    assert eval_exact["epistemic_surprise"] == 0.0
    assert eval_exact["rupture_warranted"] is False

    # Significant divergence (e.g. inventory actual is 50 vs predicted 100 -> diff -50, z = 5)
    actual_divergent = WorldState(
        inventory=50.0,
        backlog=20.0,
        flux=45.0,
    )
    eval_div = evaluate_prediction(pred, actual_divergent)
    assert eval_div["residuals"]["inventory"] == -50.0
    assert eval_div["max_z"] == 5.0  # |-50 / 10| = 5.0
    assert eval_div["epistemic_surprise"] > 0.8
    assert eval_div["rupture_warranted"] is True


def test_kv_prediction_and_observation_recording(isolated_kv):
    """Verify storing predictions and evaluating empirical observations updates surprise in brain_kv."""
    s0 = WorldState(inventory=100.0, demand_rate=20.0)
    pred_s1 = transition_step(s0, TransitionEvent())

    # Record prediction
    record_transition_prediction(pred_s1)

    # Actual observation deviates drastically (shock occurred in real world)
    actual_s1 = WorldState(inventory=20.0, backlog=30.0, tick=1)
    res = record_transition_observation(actual_s1)

    assert res["rupture_warranted"] is True
    assert "rupture_event" in res
    assert res["rupture_event"]["source"] == "transition_residual"

    # Verify world_model_state reflects the surprise and recorded prediction
    wm_state = world_model.world_model_state()
    assert wm_state["epistemic_surprise"] > 0.0
    assert wm_state["last_prediction"] is not None
    assert wm_state["last_prediction"]["inventory"] == pred_s1.inventory


def test_pipeline_expedite():
    """Expediting pipeline orders accelerates transit and receipts."""
    initial = WorldState(
        inventory=10.0,
        pipeline=[{"units": 100.0, "remaining_ticks": 4}],
        demand_rate=50.0,
    )
    # Expedite by 3 ticks
    action = TransitionEvent(expedite_ticks=3)
    s1 = transition_step(initial, action)

    # remaining_ticks was 4 -> max(1, 4 - 3) = 1 -> ticked: 1 - 1 = 0 -> arrives now!
    # Available stock: 10 + 100 = 110. Demand: 50.
    # Fulfilled: 50. Next inventory: 110 - 50 = 60.
    assert s1.inventory == 60.0
    assert s1.backlog == 0.0
    assert len(s1.pipeline) == 0


def test_capacity_reduction_outage():
    """Exogenous capacity reduction (e.g. equipment failure) clamps throughput."""
    initial = WorldState(
        inventory=200.0,
        capacity=100.0,
        demand_rate=80.0,
    )
    # 75% capacity outage -> effective capacity = 25 units
    action = TransitionEvent(capacity_reduction=0.75)
    s1 = transition_step(initial, action)

    # Demand: 80. Effective capacity: 25. Available: 200.
    # Fulfilled: min(200, 80, 25) = 25.
    # Next inventory: 200 - 25 = 175.
    # Next backlog: 80 - 25 = 55.
    # Flux: 25. Load ratio: 25 / 100 = 0.25.
    assert s1.inventory == 175.0
    assert s1.backlog == 55.0
    assert s1.flux == 25.0


def test_custom_metrics_propagation():
    """Verify arbitrary continuous variables and shocks pass through deterministically."""
    initial = WorldState(
        inventory=50.0,
        custom_metrics={"temperature_c": 21.0, "ambient_humidity": 45.0},
    )
    action = TransitionEvent(
        custom_shocks={"temperature_c": 3.5, "ambient_humidity": -5.0, "voltage_v": 12.0}
    )
    s1 = transition_step(initial, action)

    assert s1.custom_metrics["temperature_c"] == 24.5
    assert s1.custom_metrics["ambient_humidity"] == 40.0
    assert s1.custom_metrics["voltage_v"] == 12.0


def test_physical_invariants_across_30_step_random_shocks():
    """Verify physical invariants hold over a 30-step trajectory with random shocks and order policies."""
    import random
    rng = random.Random(42)

    current = WorldState(
        inventory=100.0,
        demand_rate=25.0,
        lead_time=3.0,
        capacity=60.0,
        uncertainty={"inventory": 2.0, "backlog": 1.0, "flux": 1.0, "cost": 1.0, "reliability": 0.05},
    )

    for step in range(30):
        demand_mult = rng.uniform(0.5, 2.5)
        lead_shock = rng.choice([0, 0, 1, 2])
        order_qty = rng.choice([0.0, 20.0, 40.0, 60.0])
        cap_drop = rng.choice([0.0, 0.0, 0.3, 0.5])

        action = TransitionEvent(
            order_quantity=order_qty,
            demand_multiplier=demand_mult,
            lead_time_shock=lead_shock,
            capacity_reduction=cap_drop,
        )

        prev_cost = current.cost_accumulated
        current = transition_step(current, action)

        # Invariant 1: Inventory is non-negative
        assert current.inventory >= 0.0, f"Step {step}: inventory negative ({current.inventory})"
        # Invariant 2: Backlog is non-negative
        assert current.backlog >= 0.0, f"Step {step}: backlog negative ({current.backlog})"
        # Invariant 3: Flux is non-negative and bounded by capacity
        assert current.flux >= 0.0
        assert current.flux <= current.capacity + 1e-4
        # Invariant 4: Reliability stays in [0, 1]
        assert 0.0 <= current.reliability <= 1.0, f"Step {step}: reliability {current.reliability}"
        # Invariant 5: Monotonically non-decreasing cumulative cost
        assert current.cost_accumulated >= prev_cost - 1e-6
        # Invariant 6: Uncertainty variances are positive
        for k, sigma in current.uncertainty.items():
            assert sigma > 0.0, f"Step {step}: {k} sigma non-positive ({sigma})"
        # Invariant 7: Strict tick progression
        assert current.tick == step + 1

