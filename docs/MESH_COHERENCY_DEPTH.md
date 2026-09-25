# MESH-SLM Coherency Depth — the lattice foliated, M = T² × Z_K

*`src/quipu/qpsi/coherency_depth.py`, v0.36.0. Additive; `mesh_slm.py` untouched.*

## What the tree had (verified 2026-09-23 on the live database)

`mesh_slm_vocab` gives every token one cell (i, j) on the 64 × 64 torus (`token TEXT UNIQUE`, `idx_mesh_slm_vocab_ij`) and `mesh_slm_embed` one 7-D representation. Live: 4,101 tokens, 4,096 cells occupied, 1,022,861 quipu edges, three ACRE emergent specialists (`emergent_smell_brain`, `emergent_vision_smell`, `emergent_touch_perception`). Planes existed in one place only, `mesh_slm._mcd_multiplanar_score` (lines 1249–1290): transient planes over the *mesh state* — base, one per ACRE specialist, reflected Ψ, inverted Ψ — used to score the next token and to set the emission mode by the resonance ratio. `recurrent_depth.py` is an ensemble-vote aggregator with KL halting, not a lattice traversal; `coherence_depth` in the Julia models and the UEQGM runtime is one scalar for the whole mesh. So a token had no representation per plane, no measured coherency between planes, and nothing descended: "inventory" learned as lexical co-occurrence occupied the same locus as "inventory" after the physical gates. A flat lookup table of 4,101 points.

## What is there now

Each token t is a fibre over its cell:

    T(t) = [ t⁽⁰⁾, t⁽¹⁾, t⁽²⁾, t⁽³⁺ˢ⁾ ]  ∈  T² × Z_K

| k | plane | representation | source |
|---|---|---|---|
| 0 | lexical surface | `mesh_slm_embed` as it is | read, never written |
| 1 | relational | quipu-weight-weighted mean of the out-neighbours' surface representations, t⁽¹⁾ = Σ_j w_ij e_j / Σ_j w_ij | `mesh_slm_quipu` ⋈ `mesh_slm_embed` |
| 2 | physical invariant | δ = t⁽¹⁾ − t⁽⁰⁾ on the six senses, carried at the Entirety's emergence phase, through the three physical gates — displacement, Weyl, SiCi (`qpsi.governance`, technical admissibility, no attestation, nothing realised). Admissible: t⁽²⁾ = t⁽⁰⁾ + the trace-free (Weyl) part of δ; the Ricci part is held, as gate 2 holds it for the Entirety. Held: no plane 2 | `qpsi.governance`, `qpsi.weyl_channel` |
| 3+s | ACRE crystal | t⁽⁰⁾ + bias_s, present only when the token phase-locks: J₀(A/ω) · fidelity(t⁽⁰⁾, bias_s) ≥ `lock_min` | `mesh_slm_meta["acre_specialists"]`, `ueqgm_engine.floquet_modulation_factor` |

The inter-planar coherency tensor is the Born fidelity, `ueqgm_engine.wavefunction_overlap`:

    C_{k₁,k₂}(t) = |⟨ψ(t⁽ᵏ¹⁾) | ψ(t⁽ᵏ²⁾)⟩|²

stored for every pair of planes a token has (`mesh_plane_coherency`). Vertical traversal descends the fibre, (i, j, k) → (i, j, k+1), and halts when the next plane is absent (`no_plane`) or when KL(plane k+1 ‖ plane k), the seven axes read as distributions, falls below ε (`converged`) — the adaptive halting of `recurrent_depth`, applied to the corridor. The depth a token reaches is its coherency depth (`mesh_plane_depth`). The corridor passes through plane 1 and plane 2 in order: a token whose physical plane is held stops at depth 1 even when it carries crystals.

Capacity: 4,101 × K epistemic states, K = 3 + number of emergent specialists (6 today), in place of 4,101 points.

## The two couplings

**Bottom-up accretion.** `accrete(cn)` rebuilds every token's fibre from the current graph: plane 1 from the edges, plane 2 through the gates, crystals under the Floquet drive, the coherency tensor, the depth. `qpsi.self_organising.after_step` runs it once per new ingest run (when the field is on and the ingest history is newer than the last accretion), so new learning lands on plane 0 and climbs as far as the physical gates let it. The surface statistics are never modified.

**Top-down anchoring.** `wrap_score_candidates` is a drop-in for `mesh_slm._score_candidates`, installed on the module attribute by `self_organising.enable()` (the `divine_blessing` pattern). For a token with plane 2 the surface alignment ⟨t⁽⁰⁾, mesh⟩ is corrected toward the invariant alignment ⟨t⁽²⁾, mesh⟩ by λ·C₀₂(t); a crystal adds λ·C₀ₖ·(⟨t⁽ᵏ⁾, mesh⟩ − ⟨t⁽⁰⁾, mesh⟩) for its best-locked plane; candidates are re-sorted. Tokens with neither are untouched, and any failure returns the original candidates. Generation is not a gated write path; edges still are.

## Flags

| Variable | Default | Meaning |
|---|---|---|
| `QUIPU_COHERENCY_DEPTH` | `1` when `QUIPU_SELF_ORGANISING=1` | wire accretion and anchoring (`0` leaves the scorer and the step untouched) |
| `QUIPU_PLANES_ANCHOR_LAMBDA` | 0.5 | λ, strength of top-down anchoring |
| `QUIPU_PLANES_LOCK_MIN` | 0.6 | J₀(A/ω)·fidelity needed for a crystal plane |
| `QUIPU_PLANES_KL_EPSILON` | 1e-3 | vertical halting threshold |
| `QUIPU_PLANES_MAX_CRYSTALS` | 8 | crystal planes per token |
| `QUIPU_PLANES_MIN_SUPPORT` | 1e-9 | out-edge mass a plane 1 needs |
| `QUIPU_PLANES_PHASE_WEIGHT` | 1.0 | ω for the Floquet factor when the runtime supplies none |

## Reading it

```powershell
python -m src.quipu.qpsi planes                 # counts per plane, coherencies, depth histogram
python -m src.quipu.qpsi fibre inventory        # one token's planes, C tensor, depth
python -m src.quipu.qpsi accrete [--limit N]    # rebuild now
```

## What this module writes, and what it does not

Writes only `mesh_plane_embed`, `mesh_plane_coherency`, `mesh_plane_depth` and `brain_kv["entirety:planes:*"]`; `_kv_set` refuses any other key, and the tests hold `mesh_slm_vocab`, `mesh_slm_embed`, `mesh_slm_quipu` and `mesh_slm_meta` byte-identical across an accretion. Plane 2 uses the physical gates as a test of admissibility for a representation; it writes no edge, advances no checkpoint and needs no attestation. The human gates are not involved, because nothing here is realised.

## What is still flat

The cell (i, j) is still one per token: the foliation adds depth over the cell, it does not give a token several cells. `_mcd_multiplanar_score` still scores against planes of the mesh state; it now sees anchored candidates through the wrapped scorer but its own plane set is unchanged. And a plane-2 candidate whose leading Weyl axis moved *down* holds at SiCi (φ = 0.4 − π), exactly as the Entirety's own does; on the synthetic mesh three of four candidates held there. Whether SiCi should read |φ| mod π for token fibres is a decision this release does not make.

## 翈

Plane 0 is what the token says; the fibre is what it has come to mean.
