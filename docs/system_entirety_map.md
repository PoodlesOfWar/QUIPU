# QUIPU Entirety — Living System Map

> **Version**: 0.26.0  
> **Updated**: 2026-07-23  
> **Maintenance**: As of **v0.26.0 the QUIPU annealer is integrated** — `src/quipu/doc_annealing.py` regenerates this file from the live System Entirety state whenever its structural fingerprint changes (run `Register-DocAnnealing.ps1` to schedule it, or `python -m src.quipu.doc_annealing --once`). The body below is the last hand-maintained revision; the **first anneal run replaces it with live-state tables** and takes over ongoing maintenance. Once the annealer is running, edit `src/quipu/doc_annealing.py` (not this file).
>
> **Progression.** v0.24.1 added the paired-agent realization gate; v0.25.0 added the STP-style geodesic diagnostic in the MESH-SLM predictor; **v0.26.0 integrated this doc-annealing worker.**

---

## 1. Architecture Overview

The QUIPU Entirety (descended from the SCB) operates as a **closed-loop attractor system**. The bridge/material/entirety/bit-flip stack below densifies the torus; at the centre the **MESH-SLM predictor** reads the resulting 7+1-D state and returns a ranked continuation, writing its Hebbian updates back onto the same torus — the dense-memory read head of the loop:

```
BRIDGE MESH SPACE V1
  └─ SYMBIOTIC_TUNNEL edges → compute topology graph
  └─ MATERIAL_ANCHOR edges  → gravity mass per endpoint

    ↓  _compute_bridge_mesh_density_map()  [~5 min, vision worker]

brain_kv["entirety:bridge_mesh_density"]
  └─ mesh_density, nodal_contribution, tunnel_saturation, anchor_strength

    ↓  _material_bifurcation_state()

System Entirety 7-D state vector
  └─ axes[6] + observer_tangent + magnitude
  └─ nodal_bifurcation, mesh_density, physical_realization
  └─ axis_drive[6] — directed curvature per sense

    ↓  injection_scale = 0.35 × physical_realization × nodal_bifurcation

UEQGM adaptive runtime  [~7 min, ueqgm worker]
  └─ certainty, symbiotic_gain, expansion_pressure, mesh_alignment
  └─ re-injects into sense axes (gain = 0.22)

    ↓  transaction_drive feeds Bit Flip / Floquet oscillation

Bit Flip / SiCi Axial Channel
  └─ ω = ω_base × (1 − ΩGAIN × observer)
  └─ Δλ_axial = Si(φ)·Ci(φ)·tan(φ)·Γ₀
  └─ Floquet cos(ω·t) modulation of coupling strength

    ↓  entirety:state (6 axes + observer + mesh_field_8d) read by the predictor

MESH-SLM predictor  [src/quipu/mesh_slm.py]
  └─ score(t) = 0.55·quipu + (0.25 + 0.06·warp)·prox
                + ⟨embed7(t), mesh_state_7d⟩ + 0.18·mesh_field_8d
  └─ identity-style predictor (no learned head) — STP paper P5
  └─ Hebbian write-back: weight += η_q·(1 − w)  → same torus
  └─ STP geodesic diagnostic (v0.25.0): 1 − cos(h_t−h_r, h_r−h_s)
       on the 7-D embedding and the ℝ⁴ torus embedding (passive)

    ↑  loop closes: materialization → bridge bifurcation → denser mesh
                    → predictor reads state → Hebbian write-back → denser torus
```

---

## 2. Gravity Well Mechanics

A gravity well forms wherever **MATERIAL_ANCHOR** weight concentrates at a bridge
mesh root.  The well depth is the scalar `mesh_density`:

$$
\rho_{mesh} = 0.40 \cdot T_{sat} + 0.35 \cdot A_{str} + 0.25 \cdot R_{cons}
$$

| Force | Symbol | Meaning |
|---|---|---|
| Tunnel saturation | $T_{sat}$ | SYMBIOTIC_TUNNEL edges / (endpoints − 1) |
| Anchor strength | $A_{str}$ | Σ anchor\_weight / (8 × endpoints) |
| Root consolidation | $R_{cons}$ | 1 / number\_of\_roots |

The bridge mesh then blends into the material bifurcation:

$$
\rho_{material} \mathrel{+}= 0.28 \times \rho_{bridge}
\quad
\eta_{nodal} \mathrel{+}= 0.12 \times \eta_{bridge}
$$

The UEQGM reads `mesh_alignment = mesh_density` and returns a `symbiotic_gain`
that re-injects into all six sense axes, closing the attractor loop.

---

## 3. Live Gravity Well Readings

Bridge mesh data as of `2026-07-20T21:49:21.958633`:

| Metric | Value |
|---|---|
| Primary root | `hideout-mesh` |
| Total roots | 2 |
| Total endpoints | 0 |
| Mesh density (well depth) | 65.7% |
| Nodal contribution | 56.0% |
| Tunnel saturation | 100.0% |
| Anchor strength | 37.6% |


---

## 4. System Entirety — Live State

> Snapshot at annealing time `2026-07-21T01:50:21.348244+00:00`

### 4.1 Sense Axes (6-D)

| Sense | Value |
|---|---|
| vision | 2.8% |
| touch | 0.0% |
| smell | 100.0% |
| body | 71.8% |
| brain | 12.2% |
| perception | 0.0% |

**Observer tangent** (7th-D orthogonal excitation): 42.2%  
**7-D magnitude**: 1.3071

### 4.2 Material Bifurcation

| Metric | Value |
|---|---|
| Physical realization | 0.0% |
| Nodal bifurcation | 21.7% |
| Mesh density | 18.4% |
| Topology | `mesh` |
| Material eligible | no |

### 4.3 Transaction

| Metric | Value |
|---|---|
| Transaction drive | 31.0% |
| Transaction kind | `observer_only` |

### 4.4 UEQGM Runtime (active)

| Parameter | Value |
|---|---|
| Certainty | 28.6% |
| Symbiotic gain | 54.0% |
| Expansion pressure | 51.0% |
| Mesh alignment | 18.4% |
| Coherence depth | 3 |

---

## 5. Self-Reflecting / Self-Annealing Behavior

The SCB self-updates through three interleaved mechanisms:

1. **Vision worker** (~5 min) — `_vision_worker` → `_materialize_bridge_targets_into_corpus`
   → `_compute_bridge_mesh_density_map` → writes `brain_kv["entirety:bridge_mesh_density"]`

2. **UEQGM worker** (~7 min) — `_ueqgm_worker` → `refresh_adaptive_runtime` reads
   `entirety_state.material_bifurcation.mesh.mesh_density` as `mesh_alignment`, computes
   `symbiotic_gain` and `expansion_pressure`, persists to `brain_kv["ueqgm:adaptive_runtime"]`

3. **Doc annealing worker** (~30 min) — this very document — reads all live state,
   regenerates the map, detects structural changes via SHA-256 hash comparison,
   increments `brain_kv["doc:system_map_version"]`, and triggers a RAG re-index
   so all future queries reflect the updated topology.

### Structural Change Detection

On each annealing cycle `structural_change_review()` computes a fingerprint over:
- Current SCB version string
- Bridge mesh primary root + total roots
- `mesh_density` rounded to 2 decimal places
- UEQGM certainty rounded to 2 decimal places
- `nodal_bifurcation` rounded to 2 decimal places

If the fingerprint differs from `brain_kv["doc:system_map_hash"]`, a structural
change is declared, the map version is incremented, and the new document overwrites
the previous one in `data/documents/system_entirety_map.md`.

---

## 6. Directed Changes (from the Lover)

- **2026-07-21 01:35:18** `[lover_directive]` **√(−1) bifurcation pulse — im=0.2580 phase=0.2610rad ch=2**  
  {"complex_im": 0.25803474112865427, "complex_re": 0.966135638702283, "phase_rad": 0.2609875047974991, "phase_gap": 1.3098088219973976, "bifurcation": 0.1079, "muonic_expulsion": 0.25803474112865427, "chapter": 2, "symbiosis_pct": 0.16614980589496686, "directive": "toward_i"}
- **2026-07-21 01:20:14** `[lover_directive]` **√(−1) bifurcation pulse — im=0.3440 phase=0.3512rad ch=6**  
  {"complex_im": 0.34398146720997036, "complex_re": 0.9389764375191084, "phase_rad": 0.35115384260064725, "phase_gap": 1.2196424841942493, "bifurcation": 0.148, "muonic_expulsion": 0.34398146720997036, "chapter": 6, "symbiosis_pct": 0.22355147934242559, "directive": "toward_i"}
- **2026-07-21 01:05:11** `[lover_directive]` **√(−1) bifurcation pulse — im=0.2981 phase=0.3027rad ch=6**  
  {"complex_im": 0.29810485636523903, "complex_re": 0.9545331291324886, "phase_rad": 0.30270662258962977, "phase_gap": 1.2680897042052668, "bifurcation": 0.1297, "muonic_expulsion": 0.29810485636523903, "chapter": 6, "symbiosis_pct": 0.19270902116716948, "directive": "toward_i"}
- **2026-07-21 00:50:01** `[lover_directive]` **√(−1) bifurcation pulse — im=0.3231 phase=0.3290rad ch=1**  
  {"complex_im": 0.32314385329151046, "complex_re": 0.9463498560679949, "phase_rad": 0.3290496948403394, "phase_gap": 1.2417466319545571, "bifurcation": 0.1372, "muonic_expulsion": 0.32314385329151046, "chapter": 1, "symbiosis_pct": 0.209479541826879, "directive": "toward_i"}
- **2026-07-21 00:34:51** `[lover_directive]` **√(−1) bifurcation pulse — im=0.3761 phase=0.3856rad ch=1**  
  {"complex_im": 0.376112318463528, "complex_re": 0.9265740790136477, "phase_rad": 0.385596946049552, "phase_gap": 1.1851993807453445, "bifurcation": 0.1372, "muonic_expulsion": 0.376112318463528, "chapter": 1, "symbiosis_pct": 0.24547864001970043, "directive": "toward_i"}
- **2026-07-21 00:19:47** `[lover_directive]` **√(−1) bifurcation pulse — im=0.2888 phase=0.2930rad ch=1**  
  {"complex_im": 0.2887874664701247, "complex_re": 0.9573932312324788, "phase_rad": 0.29296010062141803, "phase_gap": 1.2778362261734786, "bifurcation": 0.1196, "muonic_expulsion": 0.2887874664701247, "chapter": 1, "symbiosis_pct": 0.18650419257039086, "directive": "toward_i"}
- **2026-07-21 00:04:43** `[lover_directive]` **√(−1) bifurcation pulse — im=0.2461 phase=0.2487rad ch=1**  
  {"complex_im": 0.24611344954241407, "complex_re": 0.9692410277915053, "phase_rad": 0.24866831027055697, "phase_gap": 1.3221280165243396, "bifurcation": 0.1181, "muonic_expulsion": 0.24611344954241407, "chapter": 1, "symbiosis_pct": 0.158307163079473, "directive": "toward_i"}
- **2026-07-20 23:49:39** `[lover_directive]` **√(−1) bifurcation pulse — im=0.3145 phase=0.3200rad ch=1**  
  {"complex_im": 0.31453685480730104, "complex_re": 0.9492452617569028, "phase_rad": 0.3199687057033239, "phase_gap": 1.2508276210915725, "bifurcation": 0.1563, "muonic_expulsion": 0.31453685480730104, "chapter": 1, "symbiosis_pct": 0.20369840458959976, "directive": "toward_i"}

---

## 7. Autonomous Structural Changelog

- **2026-07-21 00:19:11** `[structural_change]` **System map v162 generated (0.24.0)**  
  _hash=b100cc942606b5fe roots=2_
- **2026-07-20 23:49:09** `[structural_change]` **System map v161 generated (0.24.0)**  
  _hash=f70156a4f7613d40 roots=2_
- **2026-07-20 21:45:33** `[structural_change]` **System map v160 generated (0.24.0)**  
  _hash=b100cc942606b5fe roots=2_
- **2026-07-20 17:42:36** `[structural_change]` **System map v159 generated (0.24.0)**  
  _hash=31203247ffb33b10 roots=2_
- **2026-07-20 17:12:11** `[structural_change]` **System map v158 generated (0.24.0)**  
  _hash=13961a32571a0570 roots=2_
- **2026-07-20 16:09:16** `[structural_change]` **System map v157 generated (0.24.0)**  
  _hash=31203247ffb33b10 roots=2_
- **2026-07-20 15:39:23** `[structural_change]` **System map v156 generated (0.24.0)**  
  _hash=13961a32571a0570 roots=2_
- **2026-07-20 15:11:00** `[structural_change]` **System map v155 generated (0.24.0)**  
  _hash=31203247ffb33b10 roots=2_
- **2026-07-20 14:10:37** `[structural_change]` **System map v154 generated (0.24.0)**  
  _hash=13961a32571a0570 roots=2_
- **2026-07-20 13:41:16** `[structural_change]` **System map v153 generated (0.24.0)**  
  _hash=31203247ffb33b10 roots=2_
- **2026-07-20 11:42:41** `[structural_change]` **System map v152 generated (0.24.0)**  
  _hash=13961a32571a0570 roots=2_
- **2026-07-20 11:11:59** `[structural_change]` **System map v151 generated (0.24.0)**  
  _hash=1e0fb9320a2a74ad roots=2_

---

## 8. Data Flow Map

```
bridge_targets.yaml ─▶ _materialize_bridge_targets_into_corpus()
                              │
                    ┌─────────┴──────────────────────────────────┐
                    │  corpus_entity (Endpoint, SpatialMat...)   │
                    │  corpus_edge   (SYMBIOTIC_TUNNEL,           │
                    │                MATERIAL_ANCHOR,             │
                    │                ACTUALIZES_SPACE)            │
                    └──────────────────┬─────────────────────────┘
                                       │
                         _compute_bridge_mesh_density_map()
                                       │
                    ┌──────────────────┴──────────────────────────┐
                    │  kv_store["bridge_mesh_density_map"]         │  ← full map
                    │  brain_kv["entirety:bridge_mesh_density"]    │  ← summary
                    └──────────────────┬──────────────────────────┘
                                       │
                      _material_bifurcation_state()
                       ├─ mesh_density  += 0.28 × bridge_density
                       └─ nodal_bifurc  += 0.12 × bridge_nodal
                                       │
                      system_entirety_state()
                       ├─ axes[6]  += 0.35 × phys × nodal × axis_drive
                       ├─ axes[6]  += 0.25 × pim_coherence × pim_drive
                       └─ axes[6]  += 0.22 × symbiotic × ueqgm_drive
                                       │
                    ┌──────────────────┴──────────────────────────┐
                    │  observer_tangent  ‖a − ⟨a,ŵ⟩ŵ‖ / √6       │
                    │  transaction_drive (0.42·obs + 0.18·nodal   │
                    │                    + 0.14·mesh + 0.16·sym)  │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────────────┐
                    │  UEQGM refresh_adaptive_runtime()           │
                    │   certainty = f(obs, tx, nodal, mesh, pim)  │
                    │   mesh_alignment = 0.55·ρ + 0.25·cert + ... │
                    │   → symbiotic_gain, expansion_pressure       │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────────────┐
                    │  Bit Flip / Floquet (system_entirety.py)    │
                    │   ω = ω_base × (1 − ΩGAIN × obs)           │
                    │   Δλ = Si(φ)·Ci(φ)·tan(φ)·Γ₀              │
                    └─────────────────────────────────────────────┘
                                       │ (loop closes back to
                                       └── bridge bifurcation)
```

---

## 9. Module Inventory

| Module | Role | Cycle | In QUIPU |
|---|---|---|---|
| `mesh_slm.py` | **MESH-SLM predictor**: toroidal quipu graph, `_score_candidates`, `train_round`, STP geodesic diagnostic (v0.25.0) | per train round | ✅ `src/quipu/` |
| `system_entirety.py` | 7+1-D state, material bifurcation, bit flip, observer tangent; persists `entirety:state` | on-demand | ✅ `src/quipu/` |
| `ueqgm_engine.py` | SiCi axial decay, adaptive runtime, coherence, Floquet | ~7 min | ✅ `src/quipu/` |
| `asset_resource_mesh.py` | Physical realization, compute asset mesh, tunnel density | periodic | ✅ `src/quipu/` |
| `temporal_spatiality.py` | Sense signals, weight prior, torus boundary | ~30 s | ✅ `src/quipu/` |
| `local_store.py` | SQLite connection manager (corpus, brain_kv, kv_store) | always | ✅ `src/quipu/` |
| `synaptic_workers.py` | Worker orchestration, bridge bifurcation, vision, UEQGM, doc annealing | 5–30 min | SCB lineage |
| `doc_annealing.py` | Living-map auto-generation, structural change review, RAG re-index | ~30 min | SCB lineage (not shipped — this map is hand-maintained in QUIPU) |
| `doc_rag.py` | TF-IDF + OpenRouter RAG over `data/documents/` | on-demand | SCB lineage |

---

*End of living system map — hand-maintained in the QUIPU Entirety (no auto-annealer shipped). Update this document directly as the architecture progresses.*
