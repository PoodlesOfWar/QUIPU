# QUIPU VNN Bathymetry (v0.33) & Temporal-Spatial Physics-Constrained Reconstruction Specification

## 1. Architectural Principle & Verification Boundary

QUIPU's temporal-spatial reconstruction extension separates **relational/semantic graph state** from **physical environmental and morphodynamic state**:

1. **Semantic MESH (`Z ∈ ℝ^{|V| × 7}`)**: Retains QUIPU's 7-D sense manifold (`vision`, `touch`, `smell`, `body`, `brain`, `perception`, `entirety`) in `mesh_slm_embed` and `mesh_slm_quipu`.
2. **Unified Temporal-Spatial Physical State (`S(t)`)**: Models physical position, velocity, environmental forcing, time-dependent terrain/seabed elevation, physical parameters, and covariance (`vnn_bathymetry.py`).
3. **6-Feature VNN Bathymetric Adapter**: Connects modern and back-projected historical bathymetry, uncertainty, sediment flux, and seabed slope to `SpatialMaterialProcessor` corpus nodes (`geospatial_relation.py`) and `TypedViscousCoupler` (`viscous_interaction.py`) without overwriting 7-D semantic embeddings.

### Implementation Status Matrix

| Component | Status | Module / Kernel |
|---|---|---|
| Bathymetric grid bilinear interpolation & slope | **Implemented & Tested** | `quipu.qpsi.vnn_bathymetry.BathymetricGrid` |
| NOAA ESRI ASCII & GEBCO XYZ CSV ingestion + inverse-variance fusion | **Implemented & Tested** | `load_esri_ascii_grid`, `load_xyz_csv_grid`, `fuse_bathymetric_grids` |
| Depth-dependent current attenuation $\mathbf{U}(z)=\mathbf{U}_0 e^{-z/L}$ | **Implemented & Tested** | `attenuate_current` |
| Stochastic surface drift (ocean, leeway, wave, diffusion separation) | **Implemented & Tested** | `step_surface_drift`, `SurfaceDriftConfig` |
| Vertical-motion equation & multi-regime vessel/person physics | **Implemented & Tested** | `DescentObjectParams`, `vertical_acceleration`, `simulate_descent` |
| Point-object & extended 3-D hull seabed contact detection | **Implemented & Tested** | `HullGeometry`, `check_seabed_contact` |
| Exner sediment continuity & 1850/79 CE historical back-projection | **Implemented & Tested** | `exner_step`, `reconstruct_historical_bathymetry` |
| Temporal Depression $D_T(x,y,t_0,t_1)$ & 7-state wreck burial lifecycle | **Implemented & Tested** | `temporal_depression`, `evaluate_wreck_burial` |
| 6-feature VNN bathymetric adapter & Planck-gated geospatial anchoring | **Implemented & Tested** | `extract_bathymetric_features`, `geospatial_relation.attach_bathymetry_features` |
| Geodesic Graph ($c_{ij}=-\log p_{ij}$), Voronoi & Pruned Probability-Mass Ledger | **Implemented & Tested** | `GeodesicReconstructionGraph`, `PrunedMassLedger`, `compute_pruned_mass_ledger` |
| 8-Ensemble Shadow Monte Carlo + Bayesian update + GeoJSON/CSV export | **Implemented & Tested** | `run_shadow_mc_ensembles`, `reconstruct_historical_state` |
| Holon Fleet integration (`GeoGrappler` K7, `HubCore`, `HolonWorkers`, `Perceptopoly`) | **Implemented & Tested** | `gg-shadow`, `gg-gateway`, `GGShadow.jl`, `kernels.wgsl`, `spatial.py`, `annealing.py`, `workers.py`, `herald_cram.py` |

---

## 2. Mathematical Framework

### 2.1 Unified Temporal-Spatial Physical State & Inverse Operator

$$
\mathbf{S}(t) =
\begin{bmatrix}
x(t) \\
y(t) \\
z(t) \\
\mathbf{v}(t) \\
\mathbf{F}(t) \\
B(x,y,t) \\
\boldsymbol{\theta}(t) \\
\mathbf{P}(t)
\end{bmatrix},
\qquad
\mathbf{S}(t_0) = \mathcal{R}\!\left(\mathbf{S}(t_1),\, \mathbf{E},\, \boldsymbol{\theta}\right)
$$

Rather than collapsing historical reconstruction to a single deterministic point, `reconstruct_historical_state` retains a probability-weighted ensemble $\{\mathbf{S}_i(t_0), w_i\}_{i=1}^N$ updated against historical and modern evidence $\mathbf{E}$ via Bayes' rule:

$$
p(\mathbf{S}_i \mid \mathbf{E}) \propto p(\mathbf{E} \mid \mathbf{S}_i)\, p(\mathbf{S}_i)
$$

### 2.2 Ocean, Atmospheric & Vertical Dynamics

1. **Rotating-Frame Incompressible Navier–Stokes (`viscous_interaction.py`)**:
   $$
   \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} + f_{\mathrm{cor}}\,\hat{\mathbf{k}}\times\mathbf{u}
   = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u} + \mathbf{f},
   \qquad \nabla\cdot\mathbf{u} = 0
   $$
2. **Stochastic Surface Drift (`step_surface_drift`)**:
   $$
   d\mathbf{X}_t = \left(\mathbf{U}_o(\mathbf{X}_t, t) + \alpha\mathbf{U}_a(\mathbf{X}_t, t) + \mathbf{U}_w(\mathbf{X}_t, t)\right)dt + \sqrt{2\mathbf{K}}\,d\mathbf{W}_t
   $$
3. **Depth-Dependent Current Attenuation (`attenuate_current`)**:
   $$
   \mathbf{U}(z) = \mathbf{U}_0 e^{-z/L}, \qquad z \ge 0,\; L > 0
   $$
4. **Vertical Motion & Multi-Regime Hull/Person Descent (`simulate_descent`)**:
   $$
   (m + m_a)\frac{dv}{dt} = (m - \rho_w V)g - \frac{1}{2}\rho_w C_D A v|v|, \qquad \frac{dz}{dt} = v
   $$

### 2.3 Seabed Morphodynamics, Temporal Depression & Wreck Burial

1. **Exner Sediment Continuity (`exner_step`)**:
   $$
   (1 - \lambda_p)\frac{\partial B}{\partial t} = -\nabla\cdot\mathbf{q}_s + S
   $$
2. **Historical Bathymetry Reconstruction (`reconstruct_historical_bathymetry`)**:
   $$
   B_{t_0}(x,y) = B_{\mathrm{modern}}(x,y) - \int_{t_0}^{t_{\mathrm{modern}}} \frac{\partial B}{\partial t}\,dt
   $$
3. **Temporal Depression (`temporal_depression`)**:
   $$
   D_T(x,y,t_0,t_1) = B(x,y,t_0) - B(x,y,t_1)
   $$
   *Guardrail*: $D_T$ is a multiscale relational elevation difference, not a quantum-gravitational mechanism.
4. **Burial Depth & 7-State Wreck Lifecycle (`evaluate_wreck_burial`)**:
   $$
   d_{\mathrm{burial}}(t) = \max\!\left(0,\, B(x_w, y_w, t) - Z_{\mathrm{top}}(t)\right)
   $$
   Evaluates `initial_seabed_contact`, `partial_burial`, `scour`, `sediment_accumulation`, `structural_collapse`, `fragmentation`, and `re_exposure` while keeping vertical sediment burial strictly distinct from horizontal displacement.

---

## 3. Synthetic & Historical Benchmark Results

| Test Case | Result | Metric |
|---|---|---|
| v0.33 Check 1: Bathymetric interpolation | **PASS** | $H(500,500) = 100.0\text{ m}$, $s = 0.0$ |
| v0.33 Check 2: Surface-current attenuation | **PASS** | $\|\mathbf{U}(L)\|/\|\mathbf{U}_0\| = e^{-1}$ |
| v0.33 Check 3: Initial downward acceleration | **PASS** | $dv/dt\big|_{t=0} > 0$ |
| v0.33 Check 4: Seabed-contact detection (100 m) | **PASS** | $t_{\mathrm{contact}} = 31.48\text{ s}$ |
| v0.33 Check 5: Zero-flux Exner stability | **PASS** | $\|\Delta B\|_\infty = 0.0$ |
| *Helena Sloman* (1850) 8-ensemble reconstruction | **PASS** | Extended hull vs. person separation + GeoJSON/CSV |
| Pompeii–Vesuvius (79 CE) reclamation | **PASS** | $D_T < 0$ tephra aggradation & burial |
