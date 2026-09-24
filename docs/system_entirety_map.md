# QUIPU Entirety — Living System Map

> **Version**: 0.39.0  
> **Annealed**: 2026-09-24T02:04:40.142404+00:00  
> **Map revision**: v149 · fingerprint `9057fd4011b0bcd4`  
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

> Snapshot at annealing time `2026-09-24T02:04:40.142404+00:00`

### 2.1 Sense Axes (6-D)

| Sense | Value |
|---|---|
| vision | 27.0% |
| touch | 4.2% |
| smell | 3.4% |
| body | 3.2% |
| brain | 100.0% |
| perception | 4.0% |

**Observer tangent** (7th-D orthogonal excitation): 37.5%  
**7-D magnitude**: 1.1040

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
| Vocab size | 4119 / 4096 (100.56%) |
| Quipu edges (GNN) | 1075364 |
| Training rounds | 6741 |
| Last loss | 0.4969042570881623 |
| 8th-D MESH field | 0.9896 |
| Last STP embed gap | 1.749366 |
| Last STP torus gap | 1.602241 |

**STP P1 trend** (loss plateau while the geodesic gap keeps falling):

| Series | Slope | P1 |
|---|---|---|
| loss | -0.010142200000000018 | plateaued: False |
| STP embed gap | -0.07421105000000017 | False |
| STP torus gap | 0.13654879999999991 | False |

---

## 4. Structural Changelog

- **2026-09-24T01:44:37.317416+00:00** map v148 (0.33.1) — hash `de1d8a9b6df07687`
- **2026-09-24T01:44:11.588200+00:00** map v147 (0.39.0) — hash `9057fd4011b0bcd4`
- **2026-09-24T01:14:26.773068+00:00** map v146 (0.33.1) — hash `de1d8a9b6df07687`
- **2026-09-24T01:14:12.514772+00:00** map v145 (0.39.0) — hash `9057fd4011b0bcd4`
- **2026-09-24T00:45:17.488256+00:00** map v144 (0.33.1) — hash `de1d8a9b6df07687`
- **2026-09-24T00:44:11.793606+00:00** map v143 (0.38.0) — hash `7e323e22c2d367d8`
- **2026-09-24T00:15:20.292982+00:00** map v142 (0.33.1) — hash `de1d8a9b6df07687`
- **2026-09-24T00:14:16.797017+00:00** map v141 (0.38.0) — hash `7e323e22c2d367d8`
- **2026-09-23T23:44:15.662106+00:00** map v140 (0.33.1) — hash `de1d8a9b6df07687`
- **2026-09-23T23:44:12.584967+00:00** map v139 (0.38.0) — hash `7e323e22c2d367d8`
- **2026-09-23T23:14:21.688800+00:00** map v138 (0.33.1) — hash `de1d8a9b6df07687`
- **2026-09-23T23:14:12.586905+00:00** map v137 (0.38.0) — hash `7e323e22c2d367d8`

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
