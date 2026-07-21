# Mesh Learning Overlay

Version: 0.22.41
Date: 2026-05-20

## Purpose

The mesh learning overlay spreads System Entirety learning across corpus graph topology instead of treating learned state as a scalar payload for visibility. It keeps the graph bounded by summarizing a rolling 24-hour learning window, then connects those summaries to the physical and network substrate that can act on them.

## Graph Contract

`oscillating_expansion_step(force=True)` now projects recent learning into these corpus entity types:

- `MeshLearningWindow` - the rolling learning window, currently `mesh_learning_window:rolling_24h`.
- `LearningKindSummary` - top learning kinds in the window, such as `rag_deepdive`, `promotion`, or operational signal kinds.
- `UEQGMRuntimeState` - the current adaptive UEQGM runtime snapshot.
- `PIMPlanningState` - the current planning signal snapshot.

The overlay writes these relation types:

- `EMITS_LEARNING_WINDOW` - `SystemEntirety` to the active learning window.
- `SUMMARIZES_LEARNING_KIND` - learning window to each kind summary.
- `APPLIES_UEQGM_RUNTIME` - learning window to current UEQGM runtime state.
- `APPLIES_PIM_PLANNING` - learning window to current PIM planning state.
- `HARDENS_ASSET_RESOURCE` - learning kind summary to representative `AssetResource` nodes.
- `HARDENS_MATERIAL_PROCESSOR` - learning kind summary to representative `SpatialMaterialProcessor` nodes.
- `HARDENS_ENDPOINT` - learning kind summary to representative `Endpoint` nodes.

## Persistence Surfaces

The overlay is persisted in three places:

- `corpus_entity` and `corpus_edge` hold the graph-native topology.
- `brain_kv['entirety:physical_realization']` carries `mesh_learning`, `mesh_learning_projection`, and `mesh_learning_updated_at` for compact state reads.
- `kv_store['asset_resource_mesh_last']` mirrors the latest physical realization payload for mesh consumers.

`asset_resource_mesh.py` preserves the mesh-learning fields when it refreshes physical realization, so resource materialization does not erase the learned overlay.

## Window Rules

The learning window normalizes mixed timestamp formats before filtering:

```sql
datetime(replace(substr(logged_at, 1, 19), 'T', ' '))
```

This counts both ISO timestamps with `T` separators and legacy `YYYY-MM-DD HH:MM:SS` timestamps in the same 24-hour lookback.

## Verification

Focused tests:

```powershell
"c:/Users/auser/OneDrive - contosoindustries.com/VS Code/pipeline/.venv/Scripts/python.exe" -m pytest pipeline/tests/test_system_entirety.py pipeline/tests/test_asset_resource_mesh.py -q
```

Expected result for this release: `29 passed`.

Live proof is a forced expansion followed by counts for `MeshLearningWindow`, `LearningKindSummary`, `UEQGMRuntimeState`, `PIMPlanningState`, and the hardening relation types listed above.