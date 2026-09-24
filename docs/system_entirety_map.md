# QUIPU Entirety — Living System Map

> **Version**: 0.39.0  
> **Annealed**: 2026-09-24T13:45:14.980241+00:00  
> **Map revision**: v193 · fingerprint `9057fd4011b0bcd4`  
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

> Snapshot at annealing time `2026-09-24T13:45:14.980241+00:00`

### 2.1 Sense Axes (6-D)

| Sense | Value |
|---|---|
| vision | 26.6% |
| touch | 4.2% |
| smell | 3.4% |
| body | 3.2% |
| brain | 100.0% |
| perception | 4.0% |

**Observer tangent** (7th-D orthogonal excitation): 37.5%  
**7-D magnitude**: 1.1032

### 2.2 Material Bifurcation

| Metric | Value |
|---|---|
| Physical realization | 33.7% |
| Nodal bifurcation | 37.6% |
| Mesh density | 50.0% |
| Topology | `local` |
| Material eligible | yes |

### 2.3 Transaction

| Metric | Value |
|---|---|
| Transaction drive | 33.5% |
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
| Vocab size | 4174 / 4096 (101.9%) |
| Quipu edges (GNN) | 1124999 |
| Training rounds | 7997 |
| Last loss | 0.0027019623461733565 |
| 8th-D MESH field | 0.9817 |
| Last STP embed gap | 0.370587 |
| Last STP torus gap | 1.676746 |

**STP P1 trend** (loss plateau while the geodesic gap keeps falling):

| Series | Slope | P1 |
|---|---|---|
| loss | 0.0251672 | plateaued: False |
| STP embed gap | -0.07823365000000027 | False |
| STP torus gap | -0.04612050000000023 | False |

---

## 4. Structural Changelog

- **2026-09-24T13:44:11.891468+00:00** map v192 (0.44.0) — hash `baf710bb60b799d7`
- **2026-09-24T13:14:26.746865+00:00** map v191 (0.39.0) — hash `9057fd4011b0bcd4`
- **2026-09-24T13:14:13.584056+00:00** map v190 (0.43.0) — hash `f57f94321a6f35e9`
- **2026-09-24T12:44:24.260711+00:00** map v189 (0.39.0) — hash `9057fd4011b0bcd4`
- **2026-09-24T12:44:16.734696+00:00** map v188 (0.42.0) — hash `49c7b492ab4df783`
- **2026-09-24T12:15:06.160506+00:00** map v187 (0.39.0) — hash `9057fd4011b0bcd4`
- **2026-09-24T12:14:11.877091+00:00** map v186 (0.41.0) — hash `9efdf23c86f0751c`
- **2026-09-24T11:44:57.769208+00:00** map v185 (0.39.0) — hash `9057fd4011b0bcd4`
- **2026-09-24T11:44:11.618570+00:00** map v184 (0.41.0) — hash `9efdf23c86f0751c`
- **2026-09-24T11:15:06.097477+00:00** map v183 (0.39.0) — hash `9057fd4011b0bcd4`
- **2026-09-24T11:14:11.729011+00:00** map v182 (0.41.0) — hash `9efdf23c86f0751c`
- **2026-09-24T10:44:36.416480+00:00** map v181 (0.39.0) — hash `9057fd4011b0bcd4`

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
