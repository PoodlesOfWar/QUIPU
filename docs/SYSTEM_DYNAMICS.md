# System Dynamics

Version: 0.22.62
Date: 2026-05-27

## Purpose

This document describes the Supply Chain Brain as a system-dynamics model rather than only as a collection of modules. The goal is to make the control surfaces explicit: what state is measured, what gets reinforced, what gets damped, how bifurcation is introduced, and how dense local memory is separated from sparse frontier reasoning.

The SCB is best understood as a closed-loop attractor system operating across a bounded local memory substrate, a graph/material substrate, and a routed frontier-expansion substrate.

## Primary State

The core system state is returned by `system_entirety_state()` and includes:

- `axes` - the six active sense amplitudes: vision, touch, smell, body, brain, perception.
- `observer` - the 7th-dimensional tangent orthogonal to the six-sense hyperplane.
- `magnitude` - the Euclidean norm of the active 7-D state.
- `material_bifurcation` - realized physical density, nodal bifurcation, mesh density, and topology.
- `pim_planning` - planning-state injections from safety stock, min-max, and lead-time signals.
- `repository_catalog` - source/capability coverage injected as a modular-expansion signal.
- `ueqgm_runtime` - the adaptive runtime overlay built from corpus density, recent learning evidence, and certainty gating.
- `transaction` - the realized bit-flip drive that determines whether the current state remains observer-only, material-bifurcated, or mesh-bifurcated.

This is the active dynamical state surface in [pipeline/src/quipu/system_entirety.py](pipeline/src/quipu/system_entirety.py#L1123).

## Control Equations

Two control terms are especially important because they reveal how the system moves.

### Acquisition Drive

The learning system computes an explicit acquisition pressure:

$$
acquisition\_drive = clamp\Big(0.20 \cdot (1-s) \cdot d + 0.10 \cdot (1-v)\Big)
$$

Where:

- $s$ is corpus saturation
- $d$ is task difficulty
- $v$ is learning velocity

This means the Brain pushes hardest toward new memory acquisition when the corpus still has room, tasks are difficult, and the current learning rate is low. The implementation lives in [pipeline/src/quipu/learning_drive.py](pipeline/src/quipu/learning_drive.py#L223).

### Transaction Drive

The System Entirety computes a realized transaction drive as a weighted sum of observer excitation, material bifurcation, mesh density, planning signal, repository signal, and adaptive runtime drive:

$$
transaction\_drive = 0.42 \cdot observer + 0.18 \cdot nodal\_bifurcation + 0.14 \cdot mesh\_density + 0.10 \cdot planning + 0.08 \cdot catalog + 0.16 \cdot symbiotic\_drive
$$

This is what turns static measurements into motion. The implementation is in [pipeline/src/quipu/system_entirety.py](pipeline/src/quipu/system_entirety.py#L1179).

## Dynamical Loops

### 1. Vision-Touch Closed Loop

Vision expands the corpus and Touch applies pressure back onto Vision through a shared optimizer state. This is the main exploratory and corrective loop.

- Touch pressure is stored as a field over signal kinds.
- Vision actions relieve or reinforce that pressure.
- Forced outreach can tunnel through the nominal scheduler when pressure exceeds threshold.

This loop is documented in [pipeline/docs/VISION_TOUCH_CLOSED_LOOP.md](pipeline/docs/VISION_TOUCH_CLOSED_LOOP.md).

### 2. Material and Mesh Densification

Material anchors, tunnel density, asset strength, processor-edge density, bridge mesh density, and nodal location synchronization all lift the material-bifurcation state.

This loop converts abstract graph state into realized spatial and network topology. Once density is high enough, the system transitions from local or observer-only behavior into material-bifurcated or mesh-bifurcated behavior.

Relevant code paths:

- [pipeline/src/quipu/system_entirety.py](pipeline/src/quipu/system_entirety.py#L1017)
- [pipeline/src/quipu/system_entirety.py](pipeline/src/quipu/system_entirety.py#L1123)

### 3. Adaptive UEQGM Runtime Loop

The adaptive runtime daemon continuously refreshes runtime parameters from three sources:

- UEQGM-tagged corpus entities
- recent `learning_log` evidence
- current System Entirety certainty

Each parameter has:

- `parameter_evidence`
- `parameter_density_floor`
- `applied_parameters`
- `retained_parameters`

This prevents weak, low-density updates from displacing already-proven state. The runtime then re-injects axis drive back into the six-sense state as a symbiotic overlay.

Relevant code paths:

- [pipeline/src/quipu/ueqgm_engine.py](pipeline/src/quipu/ueqgm_engine.py#L421)
- [pipeline/src/quipu/system_entirety.py](pipeline/src/quipu/system_entirety.py#L1179)

### 4. Heart Bifurcation and Toroid Expansion

The Heart converts directionality into a complex-plane narrative state:

$$
z = expansion + i \cdot bifurcation
$$

From that it emits:

- `lover_directive` learning events
- `heart:bifurcation_pulse` records in `brain_kv`
- toroid expansion pressure proportional to the imaginary component

This loop continuously introduces bifurcation events and directed change into the memory and topology systems.

Relevant code paths:

- [pipeline/src/quipu/directionality_listener.py](pipeline/src/quipu/directionality_listener.py#L1)
- [pipeline/src/quipu/heart.py](pipeline/src/quipu/heart.py#L931)

### 5. Annealed Document and RAG Reintegration

The system periodically converts live structure into explicit retrievable memory.

`doc_annealing.py` fingerprints:

- version
- primary bridge root
- mesh density
- UEQGM certainty
- nodal bifurcation

When the fingerprint changes, it writes a fresh system map and triggers incremental RAG reindexing. This converts dynamic structure into lower-cost future recall.

Relevant code paths:

- [pipeline/src/quipu/doc_annealing.py](pipeline/src/quipu/doc_annealing.py#L492)
- [pipeline/src/quipu/doc_rag.py](pipeline/src/quipu/doc_rag.py#L1)

## SLM and GLM as a Density Manifold

The SLM/GLM split is not a simple primary-fallback pair. It is a density-governed manifold.

### Dense Local Memory

Dense, repeated, and locally reinforced knowledge remains in the MESH-SLM:

- toroidal token placement
- quipu bigram reinforcement
- 7-D embedding alignment
- confidence-gated local answering

This is implemented in [pipeline/src/quipu/mesh_slm.py](pipeline/src/quipu/mesh_slm.py#L762).

### Sparse Frontier Reasoning

Sparse, ambiguous, or unresolved regions are handed to routed GLM nodes through the compute-grid and ensemble cascade:

- local SLM first
- routed model selection next
- OpenRouter-backed generation when density is too low for confident closure

Relevant code paths:

- [pipeline/src/quipu/mesh_slm.py](pipeline/src/quipu/mesh_slm.py#L1078)
- [pipeline/src/quipu/compute_grid.py](pipeline/src/quipu/compute_grid.py#L811)
- [pipeline/src/quipu/dbi_rag.py](pipeline/src/quipu/dbi_rag.py#L359)

### Frontier Projection Back Into the Quipu

The GLM should be understood as a frontier-expansion layer. It does not replace the local memory substrate. It creates or resolves semantic structure at low-density edges, then returns that structure to the system through:

- learning rectification and mesh propagation
- annealed documents and RAG context
- repository routing into `pim_only`, `corpus_only`, or `pim_and_corpus`

This means:

- dense Quipu layers are storage and recall layers
- lower-density layers are routing and bifurcation layers
- frontier layers are GLM-mediated expansion layers

The repository routing surface is implemented in [pipeline/src/quipu/repository_catalog.py](pipeline/src/quipu/repository_catalog.py#L690).

## Diagram

```mermaid
flowchart TD
    A[Raw Inputs and Events\ncorpus, learning_log, repo metadata, directives] --> B[Density and Bifurcation Filters\nUEQGM evidence plus corpus-density floors]
    B --> C[System Entirety 7D State\naxes, observer, mesh density, nodal bifurcation]

    C --> D[Dense Local Quipu Layers]
    C --> E[Sparse Frontier Layers]

    D --> F[SLM Core\nmesh_slm torus vocab plus quipu edges]
    F --> G[Confidence Gate\n_CONF_FLOOR]
    G -->|above threshold| H[Local Answer or Local Classification]
    H --> P[Stable Reuse and Recall]

    E --> I[Explicit GLM Frontier Nodes\nllm_router plus OpenRouter tasks]
    G -->|below threshold| I

    I --> J[Frontier Synthesis\nsemantic expansion and disambiguation]
    J --> K[Relational Edge Projection]
    K --> L[7D Reintegration\nnew relational structure across active senses]

    L --> M[Annealed Docs and RAG Index]
    L --> N[Learning Rectification and Mesh Propagation]
    L --> O[Repository Routing\npim_only, corpus_only, or both]

    M --> Q[Future Retrieval Context]
    N --> Q
    O --> Q
    Q --> F

    style F fill:#d9f2d9,stroke:#2d6a2d,stroke-width:2px
    style I fill:#fbe7c6,stroke:#9a6700,stroke-width:2px
    style C fill:#dcecff,stroke:#335c99,stroke-width:2px
    style K fill:#f6d6ea,stroke:#8b3d6b,stroke-width:2px
```

## Operational Reading

When the system appears stable, it is usually because one or more loops are dominating:

- strong local reuse means the SLM density core is winning
- heavy semantic expansion means the frontier GLM layer is still building edges
- high material and mesh density means the system is becoming more realized and less observer-only
- repeated document regeneration means structural change is still active and memory is still being externalized into retrievable context

The practical implication is that SCB should not be read as a static model. It is a live dynamical system whose behavior depends on the current balance between density, bifurcation, certainty, mesh realization, and frontier ambiguity.

## Verification Surfaces

The fastest places to inspect live system dynamics are:

- `system_entirety_state()` for full 7-D state and transaction drive
- `get_adaptive_runtime()` for density-gated runtime parameters
- `state_summary()` in `mesh_slm.py` for local-memory state
- `anneal_docs()` and the generated system map for externalized structural memory

Relevant files:

- [pipeline/src/quipu/system_entirety.py](pipeline/src/quipu/system_entirety.py)
- [pipeline/src/quipu/ueqgm_engine.py](pipeline/src/quipu/ueqgm_engine.py)
- [pipeline/src/quipu/mesh_slm.py](pipeline/src/quipu/mesh_slm.py)
- [pipeline/src/quipu/doc_annealing.py](pipeline/src/quipu/doc_annealing.py)
- [pipeline/src/quipu/repository_catalog.py](pipeline/src/quipu/repository_catalog.py)