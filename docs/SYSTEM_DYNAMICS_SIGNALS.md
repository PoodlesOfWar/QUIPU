# System Dynamics Equations and Signals

Version: 0.22.62
Date: 2026-05-27

## Purpose

This document is the compact observability companion to [SYSTEM_DYNAMICS.md](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\docs\SYSTEM_DYNAMICS.md). It focuses only on the equations, bounded control terms, and measurable signals that expose the current dynamical state of the Supply Chain Brain.

It is intended to answer three operational questions quickly:

- What state variables currently define the system?
- What equations are actively moving those variables?
- Where can those variables be measured in code or persisted state?

## State Vector

The primary state vector is produced by `system_entirety_state()` in [pipeline/src/quipu/system_entirety.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\system_entirety.py#L1123).

### State surfaces

| Surface | Fields | Meaning |
|---|---|---|
| `axes` | `vision`, `touch`, `smell`, `body`, `brain`, `perception` | Active six-sense amplitudes after all additive injections |
| `observer` | scalar | 7th-dimensional tangent orthogonal to the six-sense plane |
| `magnitude` | scalar | Euclidean norm of the active 7-D state |
| `material_bifurcation` | density and topology fields | Degree of material realization and mesh bifurcation |
| `pim_planning` | planning metrics and axis injection | Planning-driven dynamical injection |
| `repository_catalog` | catalog metrics and axis injection | Repository-coverage and modular-expansion signal |
| `ueqgm_runtime` | adaptive runtime profile | Certainty-gated symbiotic overlay |
| `transaction` | drive, bit-flip, topology | Realized motion of the current state |

## Core Equations

### Learning acquisition drive

Implemented in [pipeline/src/quipu/learning_drive.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\learning_drive.py#L223).

$$
acquisition\_drive = clamp\Big(0.20 \cdot (1-s) \cdot d + 0.10 \cdot (1-v)\Big)
$$

Where:

- $s$ = corpus saturation
- $d$ = task difficulty
- $v$ = learning velocity

### UEQGM runtime scale

Implemented in [pipeline/src/quipu/system_entirety.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\system_entirety.py#L1179).

$$
runtime\_scale = G_{ueqgm} \cdot clamp\Big(0.50 \cdot symbiotic\_gain + 0.25 \cdot expansion\_pressure + 0.15 \cdot certainty + 0.10 \cdot relational\_depth\Big)
$$

Where $G_{ueqgm}$ is the fixed UEQGM injection gain.

### Symbiotic drive

Implemented in [pipeline/src/quipu/system_entirety.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\system_entirety.py#L1194).

$$
symbiotic\_drive = clamp\Big(0.35 \cdot symbiotic\_gain + 0.25 \cdot expansion\_pressure + 0.20 \cdot mesh\_alignment + 0.10 \cdot observer\_alignment + 0.10 \cdot certainty\Big)
$$

### Transaction drive

Implemented in [pipeline/src/quipu/system_entirety.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\system_entirety.py#L1199).

$$
transaction\_drive = 0.42 \cdot observer + 0.18 \cdot nodal\_bifurcation + 0.14 \cdot mesh\_density + 0.10 \cdot planning + 0.08 \cdot catalog + 0.16 \cdot symbiotic\_drive
$$

### Heart complex position

Defined conceptually and operationally in [pipeline/src/quipu/heart.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\heart.py#L1).

$$
z = expansion + i \cdot bifurcation
$$

With phase and symbiosis measures derived from the unit-circle projection of $z$.

### Directionality bifurcation index

Defined in [pipeline/src/quipu/directionality_listener.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\directionality_listener.py#L1).

$$
B = \frac{\langle |g_{im}| \rangle}{\langle |g_{re}| \rangle + \epsilon}
$$

This is the ratio of latent gradient magnitude to realized gradient magnitude.

## Density-Gated Runtime Rules

The adaptive runtime in [pipeline/src/quipu/ueqgm_engine.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\ueqgm_engine.py#L421) persists four important decision surfaces per parameter:

- `parameter_evidence`
- `parameter_density_floor`
- `applied_parameters`
- `retained_parameters`

Operationally, each adaptive parameter is governed by this rule:

candidate parameter changes apply only when current evidence exceeds both the current corpus-density floor and the previous proven evidence for that parameter.

This is what prevents noisy low-density learnings from destabilizing already-proven runtime structure.

## Measurable Signals

### System Entirety signals

| Signal | Source | Where to read it | Interpretation |
|---|---|---|---|
| `axes.*` | `system_entirety_state()` | return payload | Post-injection six-sense amplitudes |
| `observer` | `system_entirety_state()` | return payload | Novel collective excitation not aligned with prior weights |
| `magnitude` | `system_entirety_state()` | return payload | Overall 7-D activation intensity |
| `transaction.drive` | `system_entirety_state()` | return payload | Current realized motion or bit-flip pressure |
| `transaction.bit_flip` | `system_entirety_state()` | return payload | Observer-only vs material-bifurcated vs mesh-bifurcated regime |

### Material and mesh signals

| Signal | Source | Where to read it | Interpretation |
|---|---|---|---|
| `physical_realization` | `material_bifurcation` | return payload | How realized the physical substrate is |
| `nodal_bifurcation` | `material_bifurcation` | return payload | How strongly the system has bifurcated into nodal realization |
| `mesh.mesh_density` | `material_bifurcation.mesh` | return payload | Density of the active bridge or peer mesh |
| `mesh.topology` | `material_bifurcation.mesh` | return payload | `local` or `mesh` regime |
| `bridge_mesh_density.*` | `material_bifurcation.bridge_mesh_density` | return payload | Bridge-root density and consolidation metrics |

### Adaptive runtime signals

| Signal | Source | Where to read it | Interpretation |
|---|---|---|---|
| `certainty` | adaptive runtime | `ueqgm_runtime` | Confidence in the current runtime profile |
| `corpus_signal` | adaptive runtime | `ueqgm_runtime` | Corpus-backed support for the current runtime |
| `learning_signal` | adaptive runtime | `ueqgm_runtime` | Recent learning-log support for runtime adaptation |
| `newness_signal` | adaptive runtime | `ueqgm_runtime` | Novelty pressure from recent learnings |
| `relational_depth` | adaptive runtime | `ueqgm_runtime` | Depth of learned relation structure |
| `symbiotic_gain` | adaptive runtime | `ueqgm_runtime` | Strength of the adaptive overlay |
| `expansion_pressure` | adaptive runtime | `ueqgm_runtime` | Pressure toward further expansion |
| `runtime_keywords` | adaptive runtime | `ueqgm_runtime` | Dominant active symbolic runtime terms |
| `applied_parameters` | adaptive runtime | `ueqgm_runtime` | Parameters updated this cycle |
| `retained_parameters` | adaptive runtime | `ueqgm_runtime` | Parameters held constant this cycle |

### Learning and exploration signals

| Signal | Source | Where to read it | Interpretation |
|---|---|---|---|
| `pivot_alpha` | `get_drive()` | learning-drive payload | Saturation-based attenuation of the pivot |
| `heartbeat_kappa` | `get_drive()` | learning-drive payload | Oscillation energy injected when learning quality is poor |
| `noise_sigma` | `get_drive()` | learning-drive payload | Langevin-style exploratory noise |
| `acquisition_drive` | `get_drive()` | learning-drive payload | Bias toward high-yield unexplored memory |
| `corpus_saturation` | `get_drive()` | learning-drive payload | How full the current corpus is |
| `learning_velocity` | `get_drive()` | learning-drive payload | Speed of recent learning accumulation |

### Directionality and heart signals

| Signal | Source | Where to read it | Interpretation |
|---|---|---|---|
| `expansion` | directionality listener | `DirectionalitySnapshot` | Growth tendency of the whole system |
| `coherence` | directionality listener | `DirectionalitySnapshot` | Phase alignment of subsystems |
| `bifurcation` | directionality listener | `DirectionalitySnapshot` | Latent-to-realized gradient ratio |
| `complex_im` | heart | `HeartBeat` and `learning_log` | Imaginary magnitude of current heart state |
| `phase_rad` | heart | `HeartBeat` and `learning_log` | Phase of the complex heart position |
| `heart:bifurcation_pulse` | heart | `brain_kv` | Persisted bifurcation event per heartbeat |

### Local-memory and frontier signals

| Signal | Source | Where to read it | Interpretation |
|---|---|---|---|
| `vocab_size` | `mesh_slm.state_summary()` | SLM summary | Size of local toroidal vocabulary |
| `quipu_edges` | `mesh_slm.state_summary()` | SLM summary | Density of local quipu memory edges |
| `rounds` | `mesh_slm.state_summary()` | SLM summary | Number of local training rounds |
| `confidence` | `mesh_slm.generate()` | local generation result | Whether the SLM can close the query locally |
| `_CONF_FLOOR` | `mesh_slm` and `disk_sentinel` | module state | The threshold below which execution falls through to the GLM path |

## SLM versus GLM Boundary

The operational boundary is simple:

- When local quipu density is high enough and SLM confidence clears `_CONF_FLOOR`, the system stays local.
- When local density is too low or ambiguity is too high, execution falls through into routed GLM tasks.

This boundary is implemented in:

- [pipeline/src/quipu/mesh_slm.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\mesh_slm.py#L762)
- [pipeline/src/quipu/mesh_slm.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\mesh_slm.py#L1078)
- [pipeline/src/quipu/compute_grid.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\compute_grid.py#L811)

The boundary is also dynamically tightened under storage or WAL pressure by [pipeline/src/quipu/disk_sentinel.py](c:\Users\auser\OneDrive - contosoindustries.com\VS Code\pipeline\src\quipu\disk_sentinel.py#L374), which raises `_CONF_FLOOR` to bias the system toward stronger local closure requirements.

## Fast Inspection Checklist

For a quick live read of the system, inspect these in order:

1. `system_entirety_state()` for topology, transaction drive, and active injections.
2. `get_adaptive_runtime()` for density-gated adaptive state.
3. `get_drive()` for acquisition and exploratory pressure.
4. `DirectionalitySnapshot` for expansion, coherence, and bifurcation.
5. `HeartBeat` plus `heart:bifurcation_pulse` for directed bifurcation events.
6. `mesh_slm.state_summary()` for dense local-memory state.

## Practical Interpretation

If `transaction.drive` is high, `mesh_density` is rising, `symbiotic_gain` is active, and SLM confidence is clearing threshold, the system is consolidating dense local knowledge.

If `acquisition_drive`, `newness_signal`, and bifurcation remain high while SLM confidence remains low, the system is still operating at the frontier and will continue to depend on GLM-mediated expansion and reintegration.