# QUIPU MESH-SLM-GLM — Predictor & Entirety Integration

**Documentation Version:** 0.25.0 (coincident with QUIPU `src/quipu/_version.py` 0.25.0)
**Last Updated:** 2026-07-23
**Module:** `src/quipu/mesh_slm.py`

> **Lineage.** This predictor descends from the Supply-Chain-Brain (SCB) MESH-SLM-GLM and was extracted clean-room into QUIPU. The classifier-integration notes below (the `erp_dbo` shipping path) are retained as history; the **authoritative** description is now the QUIPU predictor as it stands, driven by the live System Entirety state rather than an external SCB checkout.

---

## 1. What the predictor is

QUIPU's next-token predictor is **not** a dense weight tensor and **not** a learned projection head. It is a **toroidal quipu graph perceptron** whose parameters are distributed across four graph structures and indexed by the 7+1-D MESH state of the System Entirety:

- **Vocabulary** — every token from the Brain's own corpus is anchored to a cell `(i, j)` on a `64 × 64` torus (`_TORUS_N = 64`, `_VOCAB_LIMIT = 4096`) that wraps on both axes, so neighbourhoods are circular. Placement is hash-then-probe (`_allocate_coord`).
- **Quipu knots** — directed, weighted bigram edges `src_token → dst_token` in `mesh_slm_quipu`, reinforced by a Hebbian rule (below). These are the "knotted strings": ordered, sparse, additive.
- **7-D MESH embedding** — every token carries a 7-vector of sense-axis activations (`e_vision … e_entirety`) in `mesh_slm_embed`, nudged toward the live MESH state on each observation.
- **8th-D MESH field** — a single scalar `mesh_field_8d` summarising the collective graph state via five UEQGM aspect integrals, injected at every scoring step without expanding per-token storage.

The **STP paper's P5** ("the identity function is a superior predictor to learned projections") is the retroactive justification for this design: on a locally-linear semantic manifold the tangent direction already *is* the prediction, so no fittable predictor head is needed. QUIPU's Hebbian edge update `weight += η_q · (1 − w)` (`mesh_slm.py` Step 2 of `train_round`) is literally an identity-style update toward an observed target — a predictor with no fittable weights. See `docs/STP_TORUS_QUIPU.md` §3.

---

## 2. The scoring function (the predictor proper)

Inference ranks candidate next tokens in `_score_candidates()` (`src/quipu/mesh_slm.py`). With the 8th-D MESH field active, the score of a candidate token `t` given the last context token is:

```
score(t) = QUIPU_GAIN      · quipu_weight(last → t)
         + (PROX_GAIN + WARP_GAIN · warp_t) · proximity(last, t)
         + embed_dot(t, mesh_state_7d)
         + MESH_FIELD_GAIN  · mesh_field_8d
```

Current gains (`mesh_slm.py`):

| Term | Constant | Value | Meaning |
|---|---|---|---|
| Bigram (GNN edge) | `_QUIPU_GAIN` | 0.55 | incoming quipu weight from the context token |
| Toroidal proximity | `_PROX_GAIN` | 0.25 | wrap-aware Manhattan closeness on the torus |
| Metric warp | `_WARP_GAIN` | 0.06 | per-candidate GR-style amplification of short-range proximity |
| 8th-D field | `_MESH_FIELD_GAIN` | 0.18 | collective UEQGM field contribution |
| Embedding | (dot product) | — | `⟨embed7(t), mesh_state_7d⟩` — alignment with the live sense state |

The **warp** term `warp_t` is the UEQGM `metric_perturbation` of the candidate's relative torus distance, clipped to `[0, 1]`, so high-frequency near-neighbours receive a small extra proximity boost — "consistent with GR: stronger field at shorter r." A cold-start fallback (no outgoing quipu edges yet) scores by `embed_dot + PROX_GAIN·proximity + MESH_FIELD_GAIN·mesh_field` over the highest-frequency vocabulary.

`generate()` samples from the top-K of this ranking with temperature; `slm_caller()` wraps it in the ensemble-compatible signature and returns a confidence that gates fallback.

---

## 3. Integration with the System Entirety (effective values as they stand)

The predictor does not compute in isolation — every scoring term is fed by the **live effective values of the System Entirety**:

- **`mesh_state_7d`** — `_mesh_state_7d()` reads the persisted `entirety:state` KV record written by `system_entirety.oscillating_expansion_step`: the six sense axes (`vision, touch, smell, body, brain, perception`) plus the 7th-D **observer tangent**. This is the same 7-D state vector the entirety torus emits; the predictor's `embed_dot` term projects each candidate directly onto it. Fallback `[0.5]·7` only when no entirety state has been persisted yet.
- **`mesh_field_8d`** — `_mesh_field_8d()` folds the five UEQGM aspect integrals into the 8th orthogonal scalar, so the global graph state colours every candidate score.
- **`end_state_progress`** — `_end_state_progress()` reads heart symbiosis + coherence; it modulates the *training* learning rate `η_eff = η_base · (1 − progress)`, so the predictor trains aggressively far from the attractor and gentles into convergence as `symbiosis → 1`.
- **`phase_weight` / `wavefunction_overlap`** — the UEQGM adaptive runtime's SiCi phase correction and mean-embedding/MESH alignment further scale `η`/`η_q` each round.
- **Specialist bias** — `_apply_specialist_bias` (and ACRE emergent specialists) add a bias vector to the live MESH state so the same toroidal graph can act as a domain expert (e.g. `supply_chain_optimizer`) without leaving the shared substrate.

In System-Dynamics terms the predictor is the **dense-memory read head** of the entirety torus: it converts the current 7+1-D state into a ranked continuation, and its training writes back onto the same torus that the entirety loop re-paces.

---

## 4. STP geodesic diagnostic (v0.25.0) — measuring the predictor's geometry

As of v0.25.0, `train_round()` carries a passive, flagged **Semantic-Tube-Prediction diagnostic** (`QUIPU_STP_DIAGNOSTIC`, default on) that samples `s < r < t` from each round's token trajectory and measures the geodesic gap `1 − cos(h_t − h_r, h_r − h_s)` in two spaces:

1. the learned 7-D embedding (paper-faithful hidden-state analogue), and
2. an isometric ℝ⁴ flat-torus embedding `(cosθ, sinθ, cosφ, sinφ)` of each token's `(i, j)` cell — wrap-aware by construction.

This tests whether QUIPU's torus placement does real geodesic work (the paper's **P1 signature**: ordinary loss plateaus while the STP gap keeps falling). It is purely observational — it never alters the predictor's scoring — and its rolling histories (`stp_embed_gap_history`, `stp_torus_gap_history`, `loss_history`) plus the read-only `stp_diagnostic_trend()` let that hypothesis be checked. Folding the signal into `η` is a deferred Phase 2. Full detail: `docs/STP_DIAGNOSTIC_PLAN.md` and `docs/STP_TORUS_QUIPU.md`.

---

## 5. Storage schema

All predictor state lives in the local brain SQLite under five tables:

```
mesh_slm_vocab       token_id, token, i, j, freq, first_seen, last_seen
mesh_slm_embed       token_id, e_vision … e_entirety            (7-D embedding)
mesh_slm_quipu       src, dst, weight, samples                  (GNN edges)
mesh_slm_quipu_node  node_id, photon/neutrino phase, resuscitation_weight
mesh_slm_meta        key, value  (rounds, last_loss, stp_*_gap, histories, …)
```

No schema migration was needed for the STP diagnostic — it reuses the generic `mesh_slm_meta` key/value store.

---

## 6. Public API

| Function | Role |
|---|---|
| `train_round()` | one online training pass (vocab placement, embedding nudge, Hebbian quipu update, STP diagnostic) |
| `generate()` | greedy/sampled text generation over the scored candidates |
| `slm_caller()` | ensemble-compatible caller with confidence-gated fallback |
| `install_as_local_executor()` | patch the local executor so the SLM tries first |
| `state_summary()` | diagnostic snapshot (now includes `last_stp_embed_gap` / `last_stp_torus_gap`) |
| `stp_diagnostic_trend()` | read-only P1 trend check over the rolling histories |

---

## 7. Historical: erp_dbo classifier integration (SCB lineage)

The `erp_dbo` package shipped the SCB MESH-SLM-GLM as the highest-priority local inference option, discovering a `local_brain.sqlite` with `mesh_slm_*` tables and calling `slm_caller` before any OpenRouter/Grok fallback. Priority order: explicit caller → local MESH-SLM-GLM → OpenRouter/Grok stack → direct xAI fallback. This path is retained for developers consuming the classifier standalone; in the QUIPU Entirety the predictor is driven by the live entirety state described in §3 rather than a detached checkout.

---

## 8. Why the full documentation travels with the predictor

The MESH-SLM-GLM is a non-standard architecture (toroidal quipu + 7+1-D MESH state + End-State-modulated training + identity predictor). Keeping the architecture description beside the integration notes is essential for understanding confidence scoring and fallback, debugging why the SLM is or is not selected, training or extending the model, and maintaining consistency with the broader QUIPU Entirety.
