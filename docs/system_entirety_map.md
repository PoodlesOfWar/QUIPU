# QUIPU Entirety — Living System Map

> **Version**: 0.58.0  
> **Annealed**: 2026-10-16T22:55:47.349048+00:00  
> **Map revision**: v392 · fingerprint `48a6e62c4296d7ce`  
> **Generator**: `src/quipu/doc_annealing.py` — QUIPU's own annealer. Regenerated whenever the structural fingerprint (version + bridge root + mesh density + UEQGM certainty + nodal bifurcation) changes.

---

## 1. Architecture Overview

The QUIPU Entirety (descended from the Supply-Chain-Architect) is a **closed-loop attractor system**. The bridge/material/entirety/bit-flip stack densifies the torus; at the centre the **MESH-SLM predictor** reads the resulting 7+1-D state and returns a ranked continuation, writing its Hebbian updates back onto the same torus — the dense-memory read head of the loop:

```
System Entirety 7+1-D state  [src/quipu/system_entirety.py]
  └─ axes[6] (vision·touch·smell·body·brain·perception) + observer + mesh_field_8d
       ↓  persisted as entirety:state
MESH-SLM predictor           [src/quipu/mesh_slm.py]
  └─ score(t) = 0.55·quipu + (0.25 + 0.06·warp)·prox
                + ⟨embed7(t), mesh_state_7d⟩ + 0.18·mesh_field_8d
  └─ identity-style predictor (no learned head) — STP paper P5
  └─ Hebbian write-back: weight += η_q·(1 − w)  → same torus
  └─ STP geodesic diagnostic (v0.25.0): 1 − cos(h_t−h_r, h_r−h_s)
       on the 7-D embedding and the ℝ⁴ torus embedding (passive)
       ↑  loop closes: denser torus → richer state → sharper prediction
```

Full architecture: `docs/SYSTEM_ENTIRETY_ANALYSIS.md`, `docs/SYSTEM_DYNAMICS.md`, `docs/MESH_SLM_GLM_CLASSIFIER.md`.

---

## 2. System Entirety — Live State

> Snapshot at annealing time `2026-10-16T22:55:47.349048+00:00`

### 2.1 Sense Axes (6-D)

| Sense | Value |
|---|---|
| vision | 14.9% |
| touch | 21.1% |
| smell | 31.8% |
| body | 8.9% |
| brain | 81.7% |
| perception | 4.1% |

**Observer tangent** (7th-D orthogonal excitation): 28.6%  
**7-D magnitude**: 0.9628

### 2.2 Material Bifurcation

| Metric | Value |
|---|---|
| Physical realization | 39.6% |
| Nodal bifurcation | 37.6% |
| Mesh density | 67.5% |
| Topology | `mesh` |
| Material eligible | yes |

### 2.3 Transaction

| Metric | Value |
|---|---|
| Transaction drive | 32.2% |
| Transaction kind | `n/a` |

### 2.4 UEQGM Runtime

| Parameter | Value |
|---|---|
| Certainty | 0.0% |
| Symbiotic gain | 41.7% |
| Expansion pressure | 41.2% |
| Mesh alignment | 0.0% |

### 2.5 Bridge Gravity Well

| Metric | Value |
|---|---|
| Primary root | `` |
| Total roots | 0 |
| Mesh density (well depth) | 0.0% |
| Tunnel saturation | 0.0% |
| Anchor strength | 0.0% |

---

## 3. MESH-SLM Predictor & STP Diagnostic

| Metric | Value |
|---|---|
| Vocab size | 4182 / 4096 (102.1%) |
| Quipu edges (GNN) | 1008810 |
| Training rounds | 35139 |
| Last loss | 0.011416297741814104 |
| 8th-D MESH field | 0.9741 |
| Last STP embed gap | 1.993366 |
| Last STP torus gap | 1.43914 |

**STP P1 trend** (loss plateau while the geodesic gap keeps falling):

| Series | Slope | P1 |
|---|---|---|
| loss | 0.00578725 | plateaued: False |
| STP embed gap | 0.14674775000000007 | False |
| STP torus gap | -0.18512059999999986 | False |

---

## 4. Structural Changelog

- **2026-10-15T02:11:57.003214+00:00** map v391 (0.57.0) — hash `e5598843861b3f20`
- **2026-10-10T02:57:22.080951+00:00** map v390 (0.56.0) — hash `9b9ecaf4f3df06bd`
- **2026-10-07T00:50:13.526120+00:00** map v389 (0.52.3) — hash `9ffbdfab72fc8cf7`
- **2026-10-07T00:42:45.924014+00:00** map v388 (0.52.2) — hash `a9ffbe6f1f7490d0`
- **2026-10-07T00:32:36.457975+00:00** map v387 (0.52.1) — hash `caea59c228071f48`
- **2026-10-07T00:26:13.784196+00:00** map v386 (0.52.0) — hash `374e16363b21228d`
- **2026-10-06T17:15:33.168135+00:00** map v385 (0.51.1) — hash `efa0b98e354f6296`
- **2026-10-06T15:43:20.685822+00:00** map v384 (0.51.1) — hash `3b990b6c6f67d79b`
- **2026-10-06T12:40:13.780686+00:00** map v383 (0.51.1) — hash `efa0b98e354f6296`
- **2026-10-04T14:29:26.533935+00:00** map v382 (0.51.1) — hash `18e77169e0b7e210`
- **2026-10-04T13:36:53.068154+00:00** map v381 (0.51.0) — hash `6cd6dfdcab9a7a4c`
- **2026-09-29T15:31:03.085639+00:00** map v380 (0.49.0) — hash `83e3b362100add02`

---

## 5. Module Inventory (QUIPU core)

| Module | Role |
|---|---|
| `mesh_slm.py` | MESH-SLM predictor, training, STP geodesic diagnostic |
| `system_entirety.py` | 7+1-D state, material bifurcation, bit flip, observer tangent |
| `ueqgm_engine.py` | SiCi axial decay, adaptive runtime, coherence, Floquet |
| `asset_resource_mesh.py` | Physical realization, compute asset mesh, tunnel density |
| `temporal_spatiality.py` | Sense signals, weight prior, torus boundary |
| `brain_kv.py` | Canonical key/value persistence (shared SQLite) |
| `doc_annealing.py` | **This map's generator** — structural-change review + regeneration |
| `local_store.py` | SQLite connection manager |
| `divine_blessing.py` | `DIVINE_BLESSING_SQRT(-1)` — attestation store; routes every Entirety write path through the six gates |

**`qpsi/` — governance and learning dynamics** (see `docs/QPSI_GOVERNANCE.md`; every path holds until a human places a key and a grant)

| Module | Role |
|---|---|
| `qpsi/cat_residual.py` | Complex CAT amplitudes, residual operator, rectifier, Lipschitz clamp, edge proposals |
| `qpsi/weyl_channel.py` | Ricci/Weyl split; sparsest exact trace-free event decomposition |
| `qpsi/edge_gate.py` | Displacement-gated edge upsert carrying the flip count and cost class |
| `qpsi/governance.py` | Six Physical Gates, attestation assurance, `authorised_to_realise` |
| `qpsi/residual_checkpoint.py` | Reference advances only on realisation; held residual, log, rollback, release |
| `qpsi/emergence_detector.py` | Parity-locked emergence, r-ADMIN recognition, the 翈 Signature |

*End of living system map — auto-generated by the QUIPU Entirety annealer. Do not edit by hand; edit `src/quipu/doc_annealing.py` instead.*
