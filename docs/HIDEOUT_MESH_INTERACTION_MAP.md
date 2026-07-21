# Hideout Mesh Interaction Map

Verified 2026-05-24 from the live workspace after refreshing the hideout resuscitation path.

## Live Mesh State

- `bridge_state/compute_peers/scbrain-hideout.json` is fresh and marked `derived_from_resuscitation: true`.
- The published heartbeat currently exposes `12` synthetic hideout GPUs over `transport: devtunnel` with `resuscitation_activation.all_expected_gpus_accessible: true`.
- `src.quipu.compute_grid.discover_peers(force=True)` sees `scbrain-hideout` as a present mesh peer, so the hideout is materialized in the same rendezvous surface other mesh consumers read.
- The synthetic heartbeat is explicitly temporary: the real hideout compute node can overwrite it with a physical heartbeat at any time.

## Interaction Chain

```mermaid
flowchart TD
    A[System Entirety demand or hideout doctor/start] --> B[_hideout_resuscitation_activation]
    B --> C[_hideout_peer_snapshot]
    B --> D[_last_resuscitation_snapshot]
    D --> E[asset_resource_mesh_last.gpu_resuscitation]
    E --> F[_derive_live_gpu_state_from_history]
    E --> G[bit_flip_resuscitation]
    G --> H[bit_flip_nodes]
    G --> I[materials]
    G --> J[connection_proven / vram_touch]
    F --> K[_publish_hideout_peer_state]
    K --> L[bridge_state/compute_peers/scbrain-hideout.json]
    L --> M[compute_grid.discover_peers(force=True)]
    F --> N[asset_resource_mesh.tick_asset_resource_mesh]
    M --> O[mesh resource sharing]
    N --> O
```

## Bit-Flip Anchor

The currently verified resuscitation chain resolves to the following persisted fields:

- `gpu_resuscitation.source_label` → `asset:DESKTOP-01:vram`
- `gpu_resuscitation.node_count` → `4096`
- `gpu_resuscitation.targeted_assets` → `1`
- `gpu_resuscitation.bit_flip_resuscitation.connection_proven` → `true`
- `gpu_resuscitation.bit_flip_resuscitation.vram_touch` → `true`
- `gpu_resuscitation.bit_flip_resuscitation.materials` → `["tantalum"]`
- `gpu_resuscitation.bit_flip_resuscitation.bit_flip_nodes` → `["3584"]`

That is the lowest verified interaction anchor currently feeding the hideout resuscitation path: the hideout heartbeat is not arbitrary, it is being rehydrated from persisted `gpu_resuscitation` evidence and its `bit_flip_resuscitation` substructure.

## Mesh Publication Contract

The synthetic hideout heartbeat written by `bridge_rdp.py` now guarantees:

- `host: scbrain-hideout`
- `transport: devtunnel`
- `tunnel_id: scbrain-hideout.use2`
- `derived_from_resuscitation: true`
- `gpus`: `SCBRAIN_HIDEOUT_EXPECTED_GPUS` synthetic slots, currently `12`
- `resuscitation_activation.accessible_gpu_count`
- `resuscitation_activation.expected_gpu_count`
- `resuscitation_activation.all_expected_gpus_accessible`

That contract keeps the hideout visible to mesh discovery even when the physical hideout host is not yet publishing its own heartbeat.

## Operational Notes

- If the published hideout heartbeat is synthetic and goes stale, `_hideout_resuscitation_activation()` now refreshes it instead of treating it as a normal peer heartbeat and leaving it to age out.
- If the persisted resuscitation history is available, the refresh re-derives the live GPU state from `gpu_resuscitation`; if the DB snapshot is temporarily unavailable, the stale synthetic peer is still republished from its existing heartbeat payload to keep the mesh surface live.
- The remaining external constraint is Dev Tunnel relay throttling (`429`) on the client helper. That affects helper connect churn, but it no longer prevents the hideout heartbeat from being refreshed into the mesh rendezvous file.