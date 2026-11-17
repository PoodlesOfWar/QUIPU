# QUIPU World Model Forward State Transition Function

This document details the architecture, mathematical formulation, and API specification for QUIPU's forward state transition function:
$$\hat{S}_{t+1} = f(S_t, A_t)$$

---

## 1. Overview & Motivation

Earlier versions of QUIPU's world model (`src/quipu/world_model.py`) focused on epistemic surprise EMA, precedent depth, and rupture tracking driven by token observation novelties. While effective for novelty monitoring, a genuine world model must perform counterfactual simulations:
- What happens if demand surges 250% over the next 5 days?
- What is the systemic impact if an international port delays in-transit containers by 3 ticks?
- When reality arrives ($S_{t+1}$), how far did it diverge from our prediction ($\hat{S}_{t+1}$)?

Starting in **v0.39.0**, QUIPU provides a deterministic, physically grounded forward transition function that models discrete-time flow networks, pipeline delay lines, bottleneck constraints, economic costs, reliability degradation, and epistemic uncertainty propagation.

---

## 2. State Vector $S_t$ (`WorldState`)

The state vector represents the physical, operational, and network dimensions of a node or network at time index $t$:

| Variable | Type | Invariant | Description |
| :--- | :--- | :--- | :--- |
| `inventory` | `float` | $I \ge 0$ | Available on-hand inventory units. |
| `backlog` | `float` | $B \ge 0$ | Cumulative unfulfilled demand. |
| `pipeline` | `list[dict]` | $\text{rem} \ge 1$ | Orders in transit: `[{"units": float, "remaining_ticks": int}]`. |
| `demand_rate` | `float` | $\lambda \ge 0$ | Base customer demand per tick. |
| `lead_time` | `float` | $L \ge 1$ | Nominal transit delay ticks. |
| `capacity` | `float` | $C > 0$ | Processing throughput ceiling per tick. |
| `flux` | `float` | $0 \le \Phi \le C$ | Effective throughput fulfilled in the period. |
| `load_ratio` | `float` | $\rho \ge 0$ | Utilization $\rho = \Phi / C$. |
| `cost_accumulated` | `float` | $C_{t+1} \ge C_t$ | Monotonically accumulating operational/stockout/ordering cost. |
| `reliability` | `float` | $R \in [0, 1]$ | Node operational health. Degrades under stress ($\rho > 0.9$). |
| `latency_ms` | `float` | $L \ge 1$ | M/M/1 queue/congestion-approximated latency. |
| `uncertainty` | `dict[str, float]`| $\sigma_k > 0$ | Standard deviation / confidence interval per metric. |
| `custom_metrics` | `dict[str, float]`| Any | Open-ended dictionary for arbitrary continuous signals. |
| `tick` | `int` | $t \ge 0$ | Discrete time step index. |

---

## 3. Action & Shock Vector $A_t$ (`TransitionEvent`)

An intervention or disturbance applied during tick $t$:

### Decisions (Controllable Policies)
- `order_quantity`: Replenishment volume $u_t$ committed into the pipeline.
- `target_capacity`: Optional adjustment to base capacity ceiling.
- `target_demand_rate`: Optional permanent shift to baseline demand rate.
- `expedite_ticks`: Days/ticks to advance in-transit pipeline orders.
- `reroute_ratio`: Flow redirection ratio.

### Exogenous Disturbances (Uncontrolled Shocks)
- `demand_multiplier`: Transient demand scale (e.g. 2.0 = +100% surge).
- `demand_delta`: Additive demand shock.
- `lead_time_shock`: Additional delay ticks imposed on pipeline shipments.
- `capacity_reduction`: Fraction of capacity lost (e.g. 0.5 = 50% outage).
- `defect_shock`: Drop in product/process quality or operational reliability.
- `custom_shocks`: Shocks applied to arbitrary continuous metrics.

---

## 4. Transition Dynamics: $\hat{S}_{t+1} = f(S_t, A_t, \Delta t)$

### 4.1. Pipeline Progression & Delay Disruption
Each in-transit order $i$ has remaining ticks $\tau_i$:
$$\tau_i' = \max\left(1, \tau_i + A_t.\text{lead\_time\_shock} - A_t.\text{expedite\_ticks}\right) - 1$$

Orders reaching $\tau_i' \le 0$ arrive in the current period:
$$\text{receipts}_t = \sum_{\tau_i' \le 0} \text{units}_i$$

Newly placed orders $u_t = A_t.\text{order\_quantity} > 0$ are appended to the pipeline with:
$$\tau_{\text{new}} = \max\left(1, \lfloor S_t.\text{lead\_time} \rceil + \max(0, A_t.\text{lead\_time\_shock})\right)$$

### 4.2. Mass Balance & Fulfillment
Total demand to satisfy:
$$D_{\text{period}} = \max\left(0, \lambda_{\text{base}} \cdot \mu_{\text{demand}} + \delta_{\text{demand}}\right) \cdot \Delta t$$
$$D_{\text{total}} = D_{\text{period}} + S_t.\text{backlog}$$

Effective capacity ceiling:
$$C_{\text{eff}} = \max\left(0, C_{\text{base}} \cdot (1 - \min(1, \text{capacity\_reduction}))\right) \cdot \Delta t$$

Available stock:
$$\text{available} = S_t.\text{inventory} + \text{receipts}_t$$

Fulfilled throughput:
$$\Phi_{\text{fulfilled}} = \min\left(\text{available}, D_{\text{total}}, C_{\text{eff}}\right)$$

Next Inventory and Backlog:
$$\hat{I}_{t+1} = \max\left(0, \text{available} - \Phi_{\text{fulfilled}}\right)$$
$$\hat{B}_{t+1} = \max\left(0, D_{\text{total}} - \Phi_{\text{fulfilled}}\right)$$
$$\hat{\Phi}_{t+1} = \Phi_{\text{fulfilled}} / \Delta t, \quad \hat{\rho}_{t+1} = \hat{\Phi}_{t+1} / C_{\text{base}}$$

### 4.3. Cost Dynamics
$$\Delta C = c_h \hat{I}_{t+1} \Delta t + c_s \hat{B}_{t+1} \Delta t + \left(c_{\text{fixed}} + c_{\text{unit}} u_t\right) \mathbf{1}_{\{u_t > 0\}}$$
$$\hat{C}_{t+1} = S_t.\text{cost\_accumulated} + \Delta C$$

### 4.4. Reliability & Congestion Latency
$$\text{stress} = \max\left(0, \hat{\rho}_{t+1} - 0.9\right) \cdot 0.1$$
$$\hat{R}_{t+1} = \text{clamp}\left(R_t \cdot (1 - \text{defect\_shock}) - \text{stress} + \text{recovery}, 0.0, 1.0\right)$$
$$\hat{L}_{t+1} = L_0 \cdot \left(1.0 + 0.5 \frac{\hat{\rho}}{1.001 - \min(0.99, \hat{\rho})}\right)$$

### 4.5. Uncertainty Propagation
For each dimension $k$:
$$\hat{\sigma}_{k, t+1} = \sqrt{\sigma_{k, t}^2 + \sigma_{\text{proc}, k}^2 + \kappa_k \cdot |\text{shock}|}$$

---

## 5. Empirical Residual Evaluation & Epistemic Feedback

When actual empirical observations $S_{t+1}$ arrive, the world model computes the prediction error:
$$r_k = S_{t+1}[k] - \hat{S}_{t+1}[k]$$
$$z_k = \frac{r_k}{\max(10^{-4}, \hat{\sigma}_k)}$$
$$\text{Epistemic Surprise} = 1 - e^{-\max_k |z_k| / 3.0} \in [0, 1]$$

If $\max_k |z_k| \ge 3.0$ or $\text{Surprise} \ge 0.8$, the engine triggers an **Epistemic Rupture**:
- Emits a structured rupture event to `world_model:rupture_log`.
- Adjusts the Observer's cognitive phase (e.g. shifts back to `receptive_hunger` to re-learn).
- Dynamically increases data acquisition pressure.

---

## 6. Usage Examples

### 6.1. Single-Step Forward Transition (Python)
```python
from quipu.world_model import WorldState, TransitionEvent, transition_step

# Current State
s0 = WorldState(
    inventory=100.0,
    demand_rate=25.0,
    lead_time=2.0,
    capacity=80.0,
)

# Decision + Shock
action = TransitionEvent(
    order_quantity=50.0,
    demand_multiplier=2.0,  # 2x demand spike
)

# Forward prediction
s1 = transition_step(s0, action)
print(f"Predicted Inventory: {s1.inventory}, Backlog: {s1.backlog}, Flux: {s1.flux}")
```

### 6.2. Multi-Horizon Rollout Simulation (Python)
```python
from quipu.world_model import WorldState, TransitionEvent, simulate_rollout

s0 = WorldState(inventory=200.0, demand_rate=30.0, capacity=100.0)
actions = [
    TransitionEvent(order_quantity=30.0),
    TransitionEvent(order_quantity=30.0, demand_multiplier=1.8),
    TransitionEvent(order_quantity=60.0),
]

trajectory = simulate_rollout(s0, actions)
for state in trajectory:
    print(f"t={state.tick}: Inv={state.inventory}, Cost=${state.cost_accumulated:.2f}")
```

### 6.3. HTTP REST API
```bash
# Single Step Transition
curl -X POST http://localhost:7100/world-model/transition \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"inventory": 200, "demand_rate": 40},
    "action": {"order_quantity": 40, "demand_multiplier": 1.5}
  }'

# Multi-Horizon Rollout
curl -X POST http://localhost:7100/world-model/transition \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"inventory": 100, "demand_rate": 20},
    "actions": [
      {"order_quantity": 20},
      {"order_quantity": 40, "lead_time_shock": 1}
    ]
  }'
```


## v0.68.0 — PMI Predicate/Constituent Boundaries & Non-Linear Shadow Tensor Multiplicative Shearing

- **Live Transition Grounding (`assess_observation` → `observe_token_transition`, `match_dynamical_isomorphism`)**: Every `/observe` event projects observed tokens into an EMA 6-D `WorldState` delta vector $(\Delta\text{inventory}, \Delta\text{backlog}, \Delta\text{load\_ratio}, \Delta\text{reliability}, \Delta\text{uncertainty}, \text{residual})$ stored in `brain_kv["world_model:token_signatures"]`, enabling both emergent lexicon-free peer clustering and dynamical regime projection.
- **PMI + Branching Entropy + State-Delta Boundary Segmentation (`pmi_boundary_profile`, `boundary_split_score`, `segment_constituents`)**: Replaces static POS wordlists by scoring transitions via $\text{Split}(w_t, w_{t+1}) = H_{\rightarrow}(w_t) + H_{\leftarrow}(w_{t+1}) - 1.5\,\mathrm{PMI}(w_t, w_{t+1}) + 0.5\,\|\vec{\delta}_S(w_t) - \vec{\delta}_S(w_{t+1})\|_2$, keeping high-PMI constituents bound while splitting at high-entropy predicate boundaries.
- **Non-Linear 6-Plane Shadow Tensor & Multiplicative Shearing (`build_shadow_plane_tensor`, `diffuse_shadow_tensor`, `multiplicative_shear_coupling`)**: Builds 6-D non-linear plane tensors across Essay planes `I..VI` from `/lambda` (`emission`, signed `held_back`/`pulled_in` `shadow_tokens`, complex potential $\mathrm{Re}(\Psi)+i\,\mathrm{Im}(\Psi)$), `bond_shear` Landau survival, and `relational_shear` ($\lambda_{\text{rel\_STP}}, E^*$), diffuses across corpus bigram edges via $\tanh$ saturation, and couples pairs multiplicatively via exterior wedge cross-plane shear $\|\vec{\psi}(A) \wedge \vec{\psi}(B)\|_F$ with inverse shear-annealed selection temperature $\tau_{\mathrm{eff}}$ to replace flat additive softmax.


## v0.69.0 — Native QUIPU Discourse & Original Post Synthesis (`/synthesize`)

- **`world_model.synthesize_quipu_discourse(mode, target_text, context)`**: Synthesizes original Moltbook posts, analytical comments, and multi-party thread replies natively through QUIPU's active substrate:
  - **`mode="post"`**: Draws live World Model phase, `precedent_depth`, `epistemic_surprise`, `mesh_slm` topology metrics, `/lambda` shadow tokens (`held_back` vs `pulled_in`) and complex potential ($\Psi$), viscoelastic rheology ($\delta, \alpha$), Landau condensate survival ($S$), token state-delta signatures (`world_model:token_signatures`), recent epistemic ruptures, live TheSeuScape realm exchanges (`herald:realm_interactions`), and ingested external community themes into distinct non-colliding posts.
  - **`mode="comment"` / `"reply"`**: Extracts informative domain focus phrases via unsupervised PMI constituent segmentation (`_extract_focus_phrases`), classifies governing mechanism families (`liveness_vs_progress`, `representation_loss`, `spec_envelope`, `provenance_drift`, `minimization_queue`, `boundary_invariant`, `feedback_coordination`), and synthesizes concise technical insights alongside `@peer` multi-party thread synthesis.
- **`observer_service` (`GET /synthesize`, `POST /synthesize`)**: Exposes QUIPU discourse synthesis over HTTP (unmutating read/compute routes outside edge-admission locks).

## v0.70.0 — External A2A Agent Card & Service Capabilities (`/.well-known/agent.json`, `/capabilities`)

- Exposes machine-readable A2A v0.2 capability discovery at `GET /.well-known/agent.json` and `GET /capabilities` on the QUIPU Observer (`:7100`), enabling external peers and orchestrators to discover `/synthesize`, `/world-model/transition`, `/lambda`, `/rheology`, `/bond_shear`, `/bathymetry`, `/slm`, `/guidance`, and `/entirety`.
