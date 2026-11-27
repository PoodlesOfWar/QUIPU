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

## The positron partner: a one-positron quantum cyclotron (v0.79.0)

The antimatter side of each plane's pair follows **"One-Positron Quantum Cyclotron"** (T. G. Myers, L. Soucy, B. A. D. Sukra, B. Sinha & G. Gabrielse, *Phys. Rev. Lett.* 137, 173001). In that work a single positron is suspended indefinitely in a Penning trap's magnetic field, cooled into its cyclotron ground state and moved one quantum jump at a time. The aim is to measure the positron's magnetic moment and compare it with the electron's, a test of CPT invariance for leptons.

| In the reading | Meaning |
|---|---|
| `positron.nu_c_hz` | Cyclotron frequency ν_c = eB/(2π m_e): ≈ 150 GHz at the default B = 5.36 T |
| `positron.ground_probability` | P₀ = 1 − e^{−hν_c/k_BT}: ≈ 1 − 4·10⁻¹¹ at the default T = 0.3 K |
| `planes.<axis>.antimatter` | The positron half of the plane's magic, in the share held in the ground state (P₀) |
| `planes.<axis>.unheld` | The remainder, (1 − P₀)·magic/2. Negligible when cold; at 20 K most of the half is unheld |
| `cpt.ratio` | r = (g_e⁺ − g_e⁻)/g_e⁻. The positron partner's phase is −φ(1 + r) |
| `cpt.max_asymmetry`, `cpt.conserved` | Matter vs antimatter magic. Equal whenever r = 0 (CPT holds); any r ≠ 0 shows here |

**Overrides:** `QUIPU_POSITRON_B_T`, `QUIPU_POSITRON_T_K` and `QUIPU_CPT_RATIO`. A non-finite value falls back to the default.

`/guidance` adds `own.unheld`, `positron_held` and `cpt` for each source, Perceptopoly included.

**Limits**
- **Defaults, not data.** B is the order of the Gabrielse electron traps, not a value from this paper. T is the ≈ 0.3 K reported in coverage of the trap. r defaults to 0 because the experiment's aim is to test it.
- **Source not read directly.** The PRL page was not reachable; the description comes from its indexed abstract and press coverage.
- **Declared mapping.** The CPT ratio acting on the partner's phase is this repo's mapping. It is not a model of the spin–cyclotron dynamics.

## Magic on the shadow token pathways (v0.80.0)

The magic reading now acts on the shadow pathways instead of sitting beside them. It is computed before the shadow edges, and each edge and token is weighted by the magic of the planes it actually travelled.

| Step | Formula |
|---|---|
| Pathway of an edge (u, v) | π_k ∝ \|ΔZ_u,k\| + \|ΔZ_v,k\|: the plane mix the VNN moved its two tokens along. If nothing moved, it falls back to where they sit. |
| Pathway magic | Σ π_k · magic_k. The matter half, the held antimatter half, the unheld remainder and the CPT asymmetry are summed the same way. |
| held | (matter + antimatter) / (matter + antimatter + unheld): how much of the pathway's antimatter half the positron cyclotron holds |
| Gain | g = (1 + (φ − 1)·magic/max) · held ∈ [0, φ] |
| Lift | lift × g, re-capped at 0.05 (the same ceilings as before) |

**What the gain does:**
- A pathway through stabilizer planes, which carry no magic, keeps its lift (g = 1).
- A pathway through the most magic planes is raised by up to φ.
- Antimatter the trap no longer holds (a warm trap) lowers the lift.
- At the defaults CPT holds, so the gain is the same for matter and antimatter.

**Shadow tokens.** Every endpoint of a shadow edge becomes a shadow token. Its plane is where its own state moved most, and it carries its own pathway magic and the largest lift among its edges. Tokens recur like edges (≥ 2 distinct input states) and are proposed with the same `realisation: none: …`.

**Where to read them:**
- `entirety:mesh_flow:proposals` now holds `edges` and `tokens`, each with its `pathway`.
- `/mesh_flow` serves them.
- `/guidance?source=…` lists the proposed shadow tokens on the source's own plane under `magic.shadow_tokens`. Perceptopoly sees those on perception.

**Still proposals only.** Nothing writes `mesh_slm_*`.
