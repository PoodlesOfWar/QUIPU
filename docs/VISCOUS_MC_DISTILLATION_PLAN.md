# QUIPU Multidimensional Viscous Interaction, Monte Carlo Decision Pruning & Native Semantic Distillation

**Status**: Implemented & Tested in `QUIPU` (Python + Julia) and `GeoGrappler` (Rust + Julia + WebGPU + Holon View)  
**Verification Date**: 2026-10-26  
**Governance Invariant**: Strictly read-only on `mesh_slm.py` and core mesh tables (`mesh_slm_vocab`, `mesh_slm_embed`, `mesh_slm_quipu`); all structural promotions remain gated behind Physical Gate 6 (`src/quipu/divine_blessing.py` and `src/quipu/qpsi/governance.py`).

---

## 0. Baseline Inventory & Source-Interface Verification

The table below maps all ten mathematical specification components to their exact source files across `QUIPU` and `GeoGrappler`, resolving the historical verification gap between viscoelastic rheology and continuum Navier–Stokes / Viscous Neural Network (VNN) dynamics.

| Component | Mathematical Specification | Pre-existing QUIPU Source (`v0.62–v0.65`) | Newly Implemented & Wired Source | State |
|---|---|---|---|---|
| **Native QUIPU MESH** | Semantic graph $G=(V,E,W)$, 7-D state vectors $Z \in [0,1]^{|V|\times 7}$, weighted interactions | `src/quipu/mesh_slm.py`, `src/quipu/quipu_minimal.py`, `julia/mesh_inference_model.jl` | `src/quipu/qpsi/viscous_interaction.py` (`load_vnn_state_from_conn`) | **Live / Tested** |
| **Multidimensional Navier–Stokes** | Momentum equations, incompressibility $\nabla\cdot\mathbf{u}=0$, Helmholtz–Hodge pressure projection | *Prior viscoelastic proxy only*: Kelvin–Voigt `src/quipu/qpsi/rheology.py`, PCM velocity shear `src/quipu/qpsi/pcm_cells.py`, Kolmogorov scale `cat_residual.kolmogorov_eta` | `src/quipu/qpsi/viscous_interaction.py` (`NavierStokesConfig`, `NavierStokesState`, `project_incompressible`, `ns_step`), `julia/viscous_mc_distillation_model.jl`, `GeoGrappler/rust/crates/gg-shadow/src/kernels.rs` (`ns_torus_step`) | **Implemented & Tested** |
| **Viscous Neural Network (VNN)** | Nonlinear graph convection–diffusion $\frac{dZ}{dt} = -\nu_s L_G Z - \kappa_s N_G(Z) + F_{\mathrm{sem}}(Z,t)$ | *Unassigned adapter slot* in `GeoGrappler/holon/geograppler/panels/overview.js` (`#gg-adapter-vnn`) | `src/quipu/qpsi/viscous_interaction.py` (`VNNConfig`, `VNNState`, `vnn_step`, `TypedViscousCoupler`), `GeoGrappler` kernel **K5a** (`kernels.rs`, `kernels.jl`, `kernels.wgsl`, `gpu.js`, `overview.js`) | **Implemented & Tested** |
| **Semantic distillation** | 5-term teacher–student loss $\mathcal{L}_{\mathrm{CE}} + \beta T^2\mathcal{L}_{\mathrm{KL}} + \gamma\mathcal{L}_{\mathrm{cons}} + \eta\mathcal{L}_{\mathrm{comp}} + \xi\mathcal{L}_{\mathrm{abst}}$ | *Rule-based specialist consensus only*: `quipu_minimal.distill_specialists` | `src/quipu/qpsi/semantic_distillation.py` (`DistillationConfig`, `compute_distillation_loss`, `OpenSourceDecisionStudent`, `compare_teacher_student`), `julia/viscous_mc_distillation_model.jl` | **Implemented & Tested** |
| **Decision pruning** | Cost–complexity regularization $\mathcal{R}_{\mathrm{cc}}(S) = \sum_{e\in S} J(e) + \eta_{\mathrm{cc}}|S|$ & UCB screening | Static top-$K$ SQL neighbor limit in `quipu_minimal._candidate_edges` | `src/quipu/qpsi/mc_decision_pruning.py` (`screen_candidate_edges`), `GeoGrappler` kernel **K6** (`mc_cvar_dijkstra`) | **Implemented & Tested** |
| **Monte Carlo** | Scenario sampling, confidence bounds, Rockafellar–Uryasev $\mathrm{CVaR}_\beta$ | `src/quipu/shadow_tokens.py` (`ShadowConfig`, `shadow_step`), `src/quipu/qpsi/relational_shear.py` | `src/quipu/qpsi/mc_decision_pruning.py` (`compute_var_cvar`, `evaluate_edge_mc`), `GeoGrappler` kernel **K6** | **Implemented & Tested** |
| **Greedy Dijkstra** | Weighted shortest-path optimization after MC screening + centroidal check $\inf - 1 < 4p$ | Deterministic `quipu_minimal.dijkstra_route` and `greedy_route` | `src/quipu/qpsi/mc_decision_pruning.py` (`mc_dijkstra_route`, `mc_greedy_dijkstra_route`, `centroidal_deviation`), `quipu_minimal.mc_pruned_dijkstra_route`, `GeoGrappler` kernel **K6** | **Implemented & Tested** |
| **SLM-MESH shadows** | Comparative inference, uncertainty $U(a)$, $p^*$ abstention bar, and teacher fallback | `src/quipu/shadow_tokens.py`, `GeoGrappler/rust/crates/gg-shadow` | `src/quipu/qpsi/semantic_distillation.py` (`compare_teacher_student`), `GeoGrappler/rust/crates/gg-gateway/src/shadow_run.rs` (`vnn` flag) | **Implemented & Tested** |
| **QUIPU learning** | Native plasticity, Rectified Adam (rADAM), and 5-float Weyl compression | `src/quipu/radam_optimizer.py`, `src/quipu/ueqgm_engine.py` (`weyl_scalar_tensor`), `src/quipu/qpsi/weyl_channel.py`, `julia/mesh_compression_model.jl` | Preserved intact and wired directly into `OpenSourceDecisionStudent.distill_step` and `vnn_step` | **Live / Tested** |
| **Validation** | Numerical stability (CFL, Fourier, Gershgorin), regression & Physical Gate 6 governance | `src/quipu/divine_blessing.py`, `src/quipu/qpsi/governance.py`, `tests/test_qpsi_rheology.py` | `tests/test_viscous_mc_distillation.py`, `julia/viscous_mc_distillation_model.jl` (`selftest()`), `GeoGrappler` `gg-shadow` & `gg-gateway` test suites | **Implemented & Tested** |

### Why External Verification Previously Halted on Navier–Stokes / VNN
Prior to this implementation, `QUIPU` modeled viscoelastic dissipation through a 0-D/1-D Kelvin–Voigt dashpot (`src/quipu/qpsi/rheology.py`: $G^*(\omega) = G' + iG''$, $\delta_G = \arctan(\omega\eta_{\mathrm{eff}}/E_{\mathrm{eff}})$), phase-change memory velocity shear (`src/quipu/qpsi/pcm_cells.py`: $v_k = |\Delta g_k|/\Delta t$), and relational-time shear (`src/quipu/qpsi/relational_shear.py`: $\lambda_{\mathrm{rel}} = |\dot{\tau}_{\mathrm{cat}} - \dot{\tau}_{\mathrm{ent}}| / (\dot{\tau}_{\mathrm{cat}} + \dot{\tau}_{\mathrm{ent}} + \epsilon)$), while `GeoGrappler` left `#gg-adapter-vnn` as an unassigned placeholder. The continuum $D$-dimensional incompressible Navier–Stokes solver, the nonlinear graph VNN evolution, and the typed coupling interface (`TypedViscousCoupler`) are now implemented in `src/quipu/qpsi/viscous_interaction.py` and wired into `GeoGrappler`.

---

## 1. Multidimensional Viscous Interaction (`src/quipu/qpsi/viscous_interaction.py`)

### 1.1 Physical Continuum Dynamics ($D \in \{2, 3\}$)
On a periodic spatial domain $\Omega = \prod_{d=1}^D [0, L_d)$ discretized on a uniform grid $(N_1, \dots, N_D)$, the velocity field $\mathbf{u}(\mathbf{x}, t) \in \mathbb{R}^D$ and pressure field $p(\mathbf{x}, t) \in \mathbb{R}$ evolve according to the incompressible Navier–Stokes equations:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u} + \mathbf{f}, \qquad \nabla\cdot\mathbf{u} = 0
$$

**Chorin–Temam Fractional-Step Projection (`ns_step`, `project_incompressible`)**:
1. **Tentative Momentum Update**:
   $$
   \mathbf{u}^* = \mathbf{u}^n + \Delta t_{\mathrm{sub}}\left[-(\mathbf{u}^n\cdot\nabla_h)\mathbf{u}^n + \nu\nabla_h^2\mathbf{u}^n + \mathbf{f}^n\right]
   $$
2. **Exact Discrete Central-Difference Helmholtz–Hodge Projection**:
   Using the discrete central-difference Fourier symbols $\tilde{k}_d = \frac{\sin(k_d \Delta x_d)}{\Delta x_d}$, the Poisson equation
   $$
   -\left(\sum_{d=1}^D \tilde{k}_d^2\right)\hat{p}(\mathbf{k}) = -i\frac{\rho}{\Delta t_{\mathrm{sub}}}\sum_{d=1}^D \tilde{k}_d \hat{u}_d^*(\mathbf{k})
   $$
   projects $\mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t_{\mathrm{sub}}}{\rho}\nabla_h p$ onto the divergence-free subspace up to IEEE-754 machine precision ($\|\nabla_h \cdot \mathbf{u}^{n+1}\|_\infty < 10^{-12}$).

### 1.2 Nonlinear Semantic-Graph Evolution (Viscous Neural Network)
Over the QUIPU directed weighted graph $G=(V, E, W)$ with 7-D sense embeddings $Z(t) \in [0, 1]^{|V|\times 7}$ (`vision`, `touch`, `smell`, `body`, `brain`, `perception`, `entirety`), the VNN evolves via:

$$
\frac{dZ}{dt} = -\nu_s L_G Z - \kappa_s N_G(Z) + F_{\mathrm{sem}}(Z, t)
$$

where:
- **Random-Walk / Out-Degree Graph Laplacian**: $(L_G Z)_i = \sum_{j \in \mathcal{N}(i)} w_{ij}(Z_i - Z_j)$.
- **Nonlinear Graph Convection**: $(N_G(Z))_i = \sum_{j \in \mathcal{N}(i)} w_{ij}\,\tanh(\langle Z_i, Z_j - Z_i\rangle)\,(Z_j - Z_i)$.
- **Semantic Forcing**: $F_{\mathrm{sem}}(Z_i, t) = \gamma_f(\mathbf{m}^*(t) - Z_i) + \mathbf{b}_i(t)$, followed by trace/Weyl decomposition (`src/quipu/qpsi/weyl_channel.py`) and projection onto $[0,1]^7$.

### 1.3 Typed Coupling Interface (`TypedViscousCoupler`)
`TypedViscousCoupler` enforces a strict type and shape boundary (`spatial_dims` $D \in \{2,3\}$ vs. `semantic_dims` $= 7$; any mismatch raises `TypeError`):
- **Physical $\to$ Semantic**: Dimensionless scalar invariants of the Navier–Stokes field (normalized enstrophy $\mathcal{E}/(\mathcal{E}+1)$, strain rate, pressure variance) combine with `rheology.reading` ($\tan\delta_G$) and `relational_shear` ($\lambda_{\mathrm{rel\_STP}}$) to modulate $\nu_{s,\mathrm{eff}}$ and $\kappa_{s,\mathrm{eff}}$.
- **Semantic $\to$ Physical**: The 7-D trace-free Weyl norm $\|\mathbf{w}\|_2$ (`weyl_channel.split`) and graph Dirichlet energy $\mathcal{E}_D(Z) = \frac{1}{2}\operatorname{Tr}(Z^\top L_G Z)$ modulate the divergence-free physical forcing amplitude.

---

## 2. Monte Carlo Decision Pruning & Greedy-Dijkstra (`src/quipu/qpsi/mc_decision_pruning.py`)

### 2.1 Risk-Averse Decision Objective
For every candidate edge $a = (u \to v)$ out of token $u$, $M$ Monte Carlo scenarios perturb the Quipu edge weight $w_{uv}^{(m)} = \operatorname{clip}(w_{uv} + \sigma_{\mathrm{eff}} z^{(m)}, 0, 1)$ (where $\sigma_{\mathrm{eff}} = \sigma_0(1 + \lambda_{\mathrm{rel\_STP}})$ incorporates relational shear) and evaluate the stochastic edge cost $C^{(m)}(a)$:

$$
J(a) = \mathbb{E}[C(a)] + \lambda\,\mathrm{CVaR}_\beta(C(a)) + \zeta\,U(a)
$$

where:
- $\mathrm{CVaR}_\beta(C(a)) = \inf_{\tau \in \mathbb{R}}\left\{\tau + \frac{1}{(1-\beta)M}\sum_{m=1}^M \max(C^{(m)}(a) - \tau, 0)\right\}$ is the Rockafellar–Uryasev Conditional Value-at-Risk at tail level $\beta \in (0, 1)$.
- $U(a) = \operatorname{clip}\!\left(0.60\,\hat{\sigma}_C(a) + 0.40\,(n_{uv} + 1)^{-1/2}, 0, 1\right)$ combines epistemic sample-scarcity uncertainty and aleatoric scenario standard deviation.

### 2.2 Uncertainty-Bound & Cost–Complexity Pruning (`screen_candidate_edges`)
Before any edge enters Dijkstra:
1. **Upper Confidence Bound Gate**: Edges with $\mathrm{UCB}_{95}(a) = \mathbb{E}[C(a)] + 1.96\,\hat{\sigma}_C(a)/\sqrt{M} + 0.25\,U(a) > C_{\max}$ are pruned.
2. **Cost–Complexity Gate**: Minimizing $\mathcal{R}_{\mathrm{cc}}(S_u) = \sum_{a \in S_u} J(a) + \eta_{\mathrm{cc}}|S_u|$ prunes edges with $J(a) + \eta_{\mathrm{cc}}\cdot\mathrm{rank}(a) > \min_{a'} J(a') + \Delta_{\mathrm{cc}}$.
3. **Liveness Preservation**: If all outgoing edges from a connected node exceed the threshold, the single minimum-$J(a)$ edge is retained with `prune_reason="retained_best_fallback"`.

### 2.3 Greedy-Dijkstra & Centroidal Deviation (`mc_greedy_dijkstra_route`, `centroidal_deviation`)
Eligible edges enter shortest-path optimization (`mc_dijkstra_route` and `mc_greedy_dijkstra_route`). Across the selected token path $P = (v_0, \dots, v_L)$, the 7-D sense embeddings are evaluated for their **centroidal deviation**:
- Componentwise `mean` $\boldsymbol{\mu}$, `median` $\mathbf{m}$, and 10-bin histogram `mode` $\boldsymbol{\nu}$.
- Supremum centroidal deviation:
  $$
  \Delta_\infty = \max_{k \in \{1,\dots,7\}} \max\left(|\mu_k - m_k|, |\mu_k - \nu_k|\right)
  $$
- **Centroidal Acceptance Invariant**:
  $$
  \Delta_\infty - 1 < 4p, \qquad p = \frac{1}{7} \quad \left(\text{i.e., } \Delta_\infty < 1 + \frac{4}{7} \approx 1.571429\right)
  $$

---

## 3. QUIPU-Native 5-Term Semantic Distillation (`src/quipu/qpsi/semantic_distillation.py`)

### 3.1 Five-Term Student Objective (`compute_distillation_loss`)
Given teacher logits $\mathbf{z}_T \in \mathbb{R}^K$, student logits $\mathbf{z}_S \in \mathbb{R}^K$, and perturbed student logits $\tilde{\mathbf{z}}_S \in \mathbb{R}^K$:

$$
\mathcal{L} = \alpha\,\mathcal{L}_{\mathrm{CE}}(y^*, p_S(1)) + \beta\,T^2\,\mathrm{KL}(p_T(T)\,\|\,p_S(T)) + \gamma\,\mathcal{L}_{\mathrm{consistency}}(p_S(1), \tilde{p}_S(1)) + \eta\,\mathcal{L}_{\mathrm{complexity}}(\theta_S) + \xi\,\mathcal{L}_{\mathrm{abstention}}(c_S, e_{\mathrm{align}})
$$

1. $\mathcal{L}_{\mathrm{CE}} = -\log p_{S, y^*}(1)$: hard/top-1 teacher target cross-entropy.
2. $T^2\mathcal{L}_{\mathrm{KL}} = T^2 \sum_{k=1}^K p_{T,k}(T)\log\frac{p_{T,k}(T)}{p_{S,k}(T)}$: temperature-scaled Kullback–Leibler divergence preserving dark knowledge across non-winning candidates.
3. $\mathcal{L}_{\mathrm{consistency}} = \frac{1}{2}\mathrm{KL}(p_S(1)\,\|\,m) + \frac{1}{2}\mathrm{KL}(\tilde{p}_S(1)\,\|\,m)$ with $m = \frac{1}{2}(p_S(1) + \tilde{p}_S(1))$: bounded Jensen–Shannon consistency under relational-shear perturbation.
4. $\mathcal{L}_{\mathrm{complexity}} = \frac{1}{2}\|\theta_S\|_2^2 + \frac{1}{2}\frac{H(p_S(1))}{\log K}$: parameter L2 and normalized predictive entropy penalty.
5. $\mathcal{L}_{\mathrm{abstention}} = (1 - c_S)\cdot(1 - e_{\mathrm{align}}) + \max(0, p^* - c_S)^2$: penalizes overconfident disagreement while enforcing the minimum confidence bar $p^*$.

### 3.2 Native Plasticity (rADAM) & Weyl Compression (`OpenSourceDecisionStudent`)
- `OpenSourceDecisionStudent.distill_step` computes the analytical logit gradient $\nabla_{\mathbf{z}_S}\mathcal{L}$, projects it onto the 7-D sense axes, and updates `axis_gains` through QUIPU's native **Rectified Adam** optimizer (`src/quipu/radam_optimizer.py`, `radam_step`).
- Student weights are compressed into QUIPU's canonical 5-float **Weyl scalar tensor** $(\Psi_0, \Psi_1, \Psi_2, \Psi_3, \Psi_4) \in [0, 1]^5$ via `src/quipu/ueqgm_engine.py` (`weyl_scalar_tensor`) and decomposed into trace + trace-free Weyl components via `src/quipu/qpsi/weyl_channel.py` (`split`).

### 3.3 Comparative Inference, Abstention & Physical Gate 6 (`compare_teacher_student`)
- The student **abstains and falls back to the teacher** (`selected_source = "teacher_fallback"`) whenever:
  1. $c_S < p^*$ (`min_confidence`, default $0.35$),
  2. $\mathrm{KL}(p_T(1)\,\|\,p_S(1)) > \mathrm{KL}_{\max}$ (default $0.45$), or
  3. Top-1 disagreement occurs with margin $< 0.10$.
- Any structural or topology-modifying proposal (`proposes_new_structure=True`) sets `gate6_hold_required=True` for human approval via `src/quipu/divine_blessing.py`.

---

## 4. Six Implementation Work Packages & Verification Matrix

| Work Package | Scope & Deliverables | Mathematical Stability / Acceptance Gate | Verification Status |
|---|---|---|---|
| **WP1: Baseline & Interface Audit** | Verify `rheology.py`, `pcm_cells.py`, `relational_shear.py`, `weyl_channel.py`, `radam_optimizer.py`, `quipu_minimal.py` | Read-only mesh invariant; zero writes to `mesh_slm_vocab` / `mesh_slm_embed` / `mesh_slm_quipu` | **Verified** (`tests/test_qpsi_rheology.py`, `tests/test_qpsi_relational_shear.py`, `tests/test_quipu_minimal.py`) |
| **WP2: Multidimensional Navier–Stokes Solver** | `NavierStokesConfig`, `NavierStokesState`, `project_incompressible`, `check_ns_stability`, `ns_step` in `viscous_interaction.py` + `ns_torus_step` (K5b) in `gg-shadow` | Advective $\mathrm{CFL}_{\mathrm{adv}} \le 0.45$, Viscous Fourier $\mathrm{Fo}_{\mathrm{visc}} \le 0.45$, discrete divergence $\|\nabla_h\cdot\mathbf{u}\|_\infty < 10^{-12}$ | **Tested** (`test_navier_stokes_2d_and_3d_incompressibility_and_viscous_decay`, `test_navier_stokes_stability_substepping`, `gg-shadow` K5b test) |
| **WP3: Viscous Neural Network & Typed Coupler** | `VNNConfig`, `VNNState`, `vnn_step`, `TypedViscousCoupler` in `viscous_interaction.py` + kernel K5a across `gg-shadow` (Rust, Julia, WebGPU) | Gershgorin bound $\Delta t(\nu_s \lambda_{\max}(L_G) + 2\kappa_s d_{\max} + \gamma_f) \le 1.8 < 2.0$; strict $D \ne 7$ type separation; monotone Dirichlet energy decay under pure diffusion | **Tested** (`test_vnn_graph_convection_diffusion_and_typed_coupling`, `gg-shadow` K5a test) |
| **WP4: Monte Carlo $\mathrm{CVaR}_\beta$ Decision Pruning & Greedy-Dijkstra** | `MCPruningConfig`, `compute_var_cvar`, `screen_candidate_edges`, `mc_dijkstra_route`, `mc_greedy_dijkstra_route`, `centroidal_deviation` + kernel K6 in `gg-shadow` | Non-negative edge costs ($J(a) \ge \epsilon > 0$), UCB pruning before Dijkstra, centroidal deviation $\Delta_\infty - 1 < 4p$ | **Tested** (`test_mc_cvar_screening_and_greedy_dijkstra`, `gg-shadow` K6 test) |
| **WP5: 5-Term Semantic Distillation & Shadow Fallback** | `DistillationConfig`, `compute_distillation_loss`, `OpenSourceDecisionStudent`, `compare_teacher_student` in `semantic_distillation.py` | Monotone $\mathcal{L}_{\mathrm{KL}}$ reduction under rADAM steps; Weyl 5-scalar compression in $[0,1]^5$; automatic teacher fallback when $c_S < p^*$ | **Tested** (`test_semantic_distillation_5term_loss_and_student_convergence`, `test_shadow_comparison_abstention_and_gate6_hold`) |
| **WP6: Governance, Holon View & Release Gates** | Wire `#gg-adapter-vnn` in `overview.js`, `vnn` flag in `shadow_run.rs`, Julia reference model `julia/viscous_mc_distillation_model.jl`, Physical Gate 6 hold enforcement | All `QUIPU` and `GeoGrappler` workspace unit, doc, and container tests pass; `gate6_hold_required` enforced on structural proposals | **Tested** (`pytest` 51/51 passed, `cargo test` passed) |

---

## 5. Explicit Separation of Lifecycle States

1. **Designed**:
   - Multi-node distributed GPU MPI partitioning of 3D Navier–Stokes grids larger than $128^3$ (current coupling runs on single-node periodic 2D/3D grids and the $64 \times 64$ QUIPU torus).
2. **Implemented**:
   - Python modules: `src/quipu/qpsi/viscous_interaction.py`, `src/quipu/qpsi/mc_decision_pruning.py`, `src/quipu/qpsi/semantic_distillation.py`, and `quipu_minimal.mc_pruned_dijkstra_route`.
   - Julia reference model: `QUIPU/julia/viscous_mc_distillation_model.jl` and `GeoGrappler/julia/GGShadow/src/kernels.jl` (`k5a_vnn_step!`).
   - Rust & WebGPU shadow kernels: `GeoGrappler/rust/crates/gg-shadow/src/kernels.rs` (`vnn_step`, `ns_torus_step`, `mc_cvar_dijkstra`, `centroidal_deviation`), `GeoGrappler/rust/crates/gg-gateway/src/shadow_run.rs`, `GeoGrappler/holon/geograppler/shadow/kernels.wgsl` (`k5a_vnn_step`), `gpu.js` (`vnnStep`), and `panels/overview.js` (`#gg-adapter-vnn`).
3. **Tested**:
   - `QUIPU/tests/test_viscous_mc_distillation.py` (6/6 acceptance tests passing alongside 45/45 existing `rheology`, `relational_shear`, and `quipu_minimal` tests).
   - `GeoGrappler/rust` workspace tests (`gg-shadow` 33/33 unit tests + `gg-gateway` `shadow_run` tests).
4. **Live**:
   - Native QUIPU 7-D MESH state, Kelvin–Voigt rheology (`rheology.py`), relational shear (`relational_shear.py`), rADAM plasticity (`radam_optimizer.py`), Weyl 5-scalar compression (`ueqgm_engine.py`, `weyl_channel.py`), and Physical Gate 6 (`divine_blessing.py`).
