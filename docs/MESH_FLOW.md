# Mesh flow: from the axis cells to proposed shadow edges

**Module:** `src/quipu/qpsi/mesh_flow.py`
**Called from:** `qpsi/self_organising.after_step`, right after the PCM cells and the bond-shear publish. It is controlled by `QUIPU_MESH_FLOW`, which is on by default whenever the cells run, and it runs inside its own savepoint.
**Read-only route:** `GET /mesh_flow` returns `{latest, proposals}`.

## The chain, once per self-organising step

```
PCM axis cells (rADAM-programmed pressures p_k)
  → Mott core: I_k = 1/τ + 0.10·p_k into 7 LIF neurons, 64 steps
      spikes written to the racetrack, or left latent
  → poADAM (poissonic) vs rADAM (realised): Δ = p_rADAM − p_poADAM
  → TypedViscousCoupler
      stability number S(Δ) = φ^(1−2|Δ|)
      readings: rheology (from the cells) and bond shear
      target: the live 7-D MESH state (the six axes, with the observer as entirety)
  → VNN step over a read-only copy of the MESH graph
      the top 128 tokens become nodes; their Quipu edges become the graph
  → shadow edges between the pre- and post-step states
      lift: an existing edge whose two ends converged
            lift = 0.05·(S/φ)·(cos₁−cos₀)/(1−cos₀)
      new:  a pair with no edge that the flow made near-parallel
            cos₁ ≥ 0.9 and rising; weight = 0.05·(S/φ)·cos₁
  → recurrence: an edge is *proposed* once it appears in ≥ 2 distinct input states
```

The damping S/φ ties proposals to the optimisers. When the realisation matches the potential (every spike stored), proposals pass at full strength. When spikes stay latent, rADAM diverges from poADAM, S tightens and the proposals shrink.

A pressure of 0 puts a neuron exactly at its firing threshold (rheobase), so silent cells produce no spikes.

## What it writes

| Key | Contents |
|---|---|
| `entirety:mesh_flow:state` | Mott membrane potentials and the poADAM and rADAM moments, carried to the next step |
| `entirety:mesh_flow:history` | Shadow edges per distinct input state (the session is a hash of the drive, target, node set, Δ and edges); last 64 kept |
| `entirety:mesh_flow:proposals` | Edges that recur in ≥ 2 sessions, with `status: proposed` and `realisation: "none: through the six gates, two signers and the grant"` |
| `entirety:mesh_flow:latest` | The step summary: spikes, written and latent counts, both pressures, Δ, S, node and edge counts, graph CFL, and how many edges were found and proposed |
| `entirety:vnn:latest` | The coupler's own report |

**It never writes `mesh_slm_vocab`, `mesh_slm_embed` or `mesh_slm_quipu`.** A failure rolls back to the savepoint, and the rest of the step still commits.

## What it does not do

- **Proposals are not applied.** There is still no governed writer for `mesh_slm_quipu`. Applying a proposal is a separate, reviewed step: the six gates, both signers and the grant.
- **The physical field is synthetic.** It is a fresh 8×8 Taylor–Green vortex each step: a placeholder for a real physical drive.
- **The VNN advances one coupled step per self-organising step,** from the MESH's current embeddings. Proposals are about direction, not a converged state.
