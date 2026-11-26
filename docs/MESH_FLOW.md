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

## Magic: information theory read as a matter–antimatter pair (v0.78.0)

**Module:** `src/quipu/qpsi/magic.py`. The reading is published at `entirety:mesh_flow:magic`, served on `GET /mesh_flow`, and delivered to every source, Perceptopoly included, as the `magic` field of `GET /guidance?source=…`.

"Magic" (non-stabilizerness) measures how far a quantum state is from the states a classical computer simulates efficiently. It was proposed for top quark–antiquark pairs at the LHC by White & White (*Phys. Rev. D* 110, 116016 (2024)), read from the pair's spin correlations as the second stabilizer Rényi entropy:

```
M̃₂(ρ) = −log₂( Σ_P tr(ρP)⁴ / Σ_P tr(ρP)² )      P over the 16 two-qubit Paulis
```

**The transcription used here is a declared mapping, not a measurement:**

| Step | What it is |
|---|---|
| Pair per plane | \|ψ_k⟩ = cos(θ_k/2)\|00⟩ + e^{iφ_k} sin(θ_k/2)\|11⟩, one per CAT axis plane |
| Information theory in | sin²(θ_k/2) = p_k, the plane's PCM pressure. φ_k is the plane's rADAM phase: its latent share. |
| Antimatter attribution | The antimatter partner is the charge conjugate, φ → −φ. M̃₂ is even in φ, so a plane's magic is split half to matter and half to antimatter. |
| Zero magic | p = 0 or 1 (nothing or everything realised), and p = ½ with φ = 0 (a Bell pair) |
| Maximum | log₂(3/2) ≈ 0.585, at p ≈ 0.211 or 0.789 with φ = π/4 |

**Through the levels of the system diagram.** Every observer source sits on an axis plane (`SOURCE_PROFILES`), so each is attributed the magic of its own plane in `by_source`:

- Stockopoly and Bakugo sit on touch.
- Supply Chain Brain sits on brain.
- Loadopoly-OCR sits on vision.
- Perceptopoly sits on perception.

Sources with no routed plane are left out. `/guidance?source=perceptopoly` returns its own plane's magic with the matter and antimatter halves, the weighted total, and every plane.

**Not consulted.** The Google Flow video and the ScienceDaily article (2026-11-24) named with this request were unreachable from the session, so nothing here is taken from them.
