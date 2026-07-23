# Semantic Tube Prediction × Torus QUIPU

**Source paper**: Hai Huang, Yann LeCun, Randall Balestriero — *"Semantic Tube Prediction: Beating LLM Data Efficiency with JEPA"*, arXiv:2602.22617v1 [cs.LG], 26 Feb 2026. Code: `github.com/galilai-group/llm-jepa#stp`.
**Local copy**: attached to the session as `SemanticTubePrediction_LeCun_Huang_Balestriero.pdf` (annotated, 5 pages transcribed below — see note in Appendix A on why the PDF is transcribed rather than binary-embedded).
**Scope**: this document reviews the reader's margin annotations against QUIPU's existing torus/geodesic machinery and records the mapping for future reference. A companion implementation plan lives in `docs/STP_DIAGNOSTIC_PLAN.md`.

---

## 1. Why this paper matters to QUIPU

STP's closing claim — "principled geometric priors can surpass brute-force scaling" — is, almost word for word, `LEARNINGS.md`'s standing design principle: *"Geometry is a useful substrate for attention... route learning pressure along explicit geometric structure instead of flat scoring."* The paper arrives at that principle from a different direction (JEPA-style regularization of hidden-state trajectories in a pretrained transformer) than QUIPU does (a native toroidal graph substrate with no dense weight matrix at all), which makes it a useful independent check on the bet QUIPU has already made. The sections below work through the reader's six margin notes in order, then note two results the paper gets "for free" that also explain existing QUIPU behavior, and close with where the analogy breaks.

---

## 2. Note-by-note mapping

### 2.1 "Torus Quipu" (p.5, beside "geodesic geometry to LLMs remains underexplored")

This is the strongest and most consequential connection. STP treats geodesics **extrinsically**: hidden states live in flat ℝ^d, and the Semantic Tube is a straight cylindrical neighborhood around a locally-linear path (Definition 3.1, Theorem 3.3). QUIPU treats them **intrinsically**: every token is anchored to a cell `(i, j)` on an `N × N` torus that wraps on both axes (`src/quipu/mesh_slm.py:47-48`, `_TORUS_N = 64` at line 711), and adjacency/proximity is wrap-aware by construction (`_torus_dist`, `mesh_slm.py:2023-2031`; `_proximity`, `mesh_slm.py:2034-2043`).

The torus actually **strengthens** the paper's own headline claim (P3, diversity preservation / no mode collapse). STP's argument is Picard–Lindelöf: distinct initial conditions never intersect, so distinct prompts can't collapse to the same continuation. But their own Figure 2 shows the failure mode — trajectories with similar prefixes pass through the *same Voronoi cell* at different locations and can drift onto the wrong geodesic inside that cell, because flat ℝ^d gives no structural reason two nearby-but-distinct paths can't converge. On a torus, trajectories can carry a **winding number** around each axis. Two paths in different winding classes are *topologically* prevented from ever coinciding — a strictly stronger, structural guarantee than "generic non-intersection almost everywhere." QUIPU's geometry closes exactly the gap Figure 2 illustrates.

A torus also admits **closed geodesics**, a concept the paper has no use for (an autoregressive sequence never returns to its start). QUIPU is built from them: the six-ring System Entirety flux loop (`docs/SYSTEM_ENTIRETY_ANALYSIS.md` §1 — perception → dispatch → self-train → corpus → refinement → network → back to perception) is a closed geodesic on the *system* torus. STP describes the geometry of a single generation pass; QUIPU additionally describes the geometry of recurrence. The natural synthesis (see the companion plan) is to use STP as a **local** regularizer measured along segments of that larger loop, not a replacement for it.

### 2.2 "SiCi Tangential leads to STP loss" (p.4)

Correct, and it sharpens into a precise correspondence. The STP loss is a tangential-direction penalty:

```
L_STP = 1 − cos(h_t − h_r, h_r − h_s)          (noise term ⊥ the chord; signal term ∥ it)
```

QUIPU's SiCi axial channel (`src/quipu/ueqgm_engine.py:1-26`) is likewise a phase-sensitive weighting of a tangential/axial quantity, but harmonic rather than linear-chord-based:

```
Δλ_axial = [Si(φ) · Ci(φ)] · tan(φ) · Γ₀           (resonant at φ = π/4 + kπ)
```

Both are corrections applied *along* a direction of motion rather than to the state itself. The paper's P4 ("we expect λ ≪ 1 to accommodate instances where the geodesic deviates from a straight line") is the same discipline as QUIPU's bounded torus latent gradient (`_torus_latent_grad` clamped to `[0, 0.30]`, per `CHANGELOG.md:733`) — both cap how hard the tangential correction is allowed to pull. The concrete follow-up (detailed in the plan doc): modulate STP's λ per-round by the already-computed `phase_weight`, i.e. `λ_eff = λ · phase_weight`, so tube pressure breathes with harmonic coherence instead of sitting at a fixed constant.

### 2.3 "rADAM" (p.4, beside Theorem 3.3)

Also correct, and it identifies where STP's *loss-level* constraint already has an *optimizer-level* counterpart in QUIPU. rADAM's T² toroidal pressure projection

```
p_t = 0.5 · (1 + cos θ_t · cos φ_t)          (internal + external loop phases, CHANGELOG.md:705)
```

is the same cosine-alignment functional form as `L_STP`, just applied inside the update rule instead of the objective. More striking: rADAM's **complex bifurcated gradient** `g_re + i·g_im` — real component from Touch/Vision/Body firings, imaginary component from the torus gap field (`CHANGELOG.md:701`) — *is* STP's signal/noise decomposition (Figure 1: component parallel to the chord vs. component perpendicular to it), just carried as a complex number instead of a vector projection. The difference worth keeping in mind: STP estimates the tube centerline empirically from the trajectory itself (`h*` is latent, estimated via the Geodesic Hypothesis); QUIPU's rADAM pins the centerline to an internal clock — the heart-lock invariant holds `theta` exactly equal to `heart.phase_rad`, with any rADAM influence recorded as separate `radam_phase_bias` metadata rather than perturbing the toroidal ground (`RELEASE_NOTES.md:347`). STP infers where the tube is; QUIPU declares it and measures deviation from a known reference.

One correction on the crossed-out math beside this note: Theorem 3.3 does **not** give `h_r ⊥ h_t`. The orthogonality in the proof is between the *noise component* of `(h_r − h_s)` and the chord `(h_t − h_s)` (Definition 3.1's `⊥` notation) — the hidden states themselves are never claimed to be orthogonal to each other.

### 2.4 "Synthetic View Conundrum?" (p.4, bottom)

STP's answer, stated directly in the paper: no synthetic views are needed, because the random triplet `s < r < t` makes the sequence supply its own two views — segment `h_r − h_s` versus `h_t − h_r` — which must corroborate for the loss to go to zero. That is structurally identical to the LEARNINGS.md finding *"bilateral evidence beats unilateral confidence"* (paired Vision/Touch interactions only advance the optimizer when both sides corroborate). QUIPU sidesteps the conundrum even more directly than STP does: the seven sense-axes (`vision, touch, smell, body, brain, perception, entirety`) are **native orthogonal views** — different corpus domains accrete on different axes by construction (`mesh_slm._SOURCE_AXIS_MAP`, `README.md:103-106`) — rather than augmentations manufactured from a single signal.

### 2.5 "Dim collapse... Probability Bounding" (p.4)

Both flags land correctly. Dimensional-collapse risk is lower for STP than for vanilla JEPA specifically because `L_NTP` anchors representations: Voronoi cells have to stay separated to emit the *correct* next token, so `L_STP` can shape geometry within a cell without being able to flatten the whole space — the paper's own related-work section makes this comparison explicit. "Probability Bounding" is Corollary 3.4 (Random Tube): `P(‖h_r − h*‖₂ > ε + ε) → 0` as `L_STP → 0`, for *randomly* selected `s < r < t` — the guarantee doesn't depend on picking good indices.

QUIPU has already lived the failure case this section warns about: the resonance-gate work exists precisely because raw overlap scoring collapsed to reading ~1.0 for everything (`README.md:96-97`, the embedding-collapse note under Ring 5). The fix — gating synthesis on structural resonance (mutual centrality *and* shared analytical wavelength, `RESONANCE_FLOOR_HZ = 52`) rather than raw cosine overlap — is a concrete, already-deployed defense against exactly the collapse mode STP argues is merely *unlikely*, not impossible.

### 2.6 "Teacher Forcing... validated by experience" (p.4)

This note is QUIPU's own causality-boundary lesson, restated in the paper's vocabulary. Teacher forcing keeps *training* on the manifold (the ODE of §2.1) by re-injecting ground truth at every step; *inference* has no such correction and drifts (the SDE with a Brownian cone of radius `∝ σ_t√t`, §3.1 and Appendix F). QUIPU's v0.24.1 realization-gate finding is the identical boundary drawn in a different system: *"forward decisions were restricted to strictly-pre-decision (t<0) observer history... any read at or after decision time leaked the future into the gradient and corrupted eligibility"* (`LEARNINGS.md:7`). Same failure mode — future information (ground truth at inference time; post-decision observation at training time) leaking across a boundary that's supposed to be one-directional.

The sharper point: QUIPU already runs the paper's *inference-time* model as its *standing* learning dynamic, not just at generation time. The Langevin update `Δx = drift·dt + σ·dW·√dt` (`mesh_slm.py:1137`, `1400`) is the same SDE form the paper derives for inference drift — except QUIPU's `σ` is not fixed; it's set every cycle by `langevin_sigma_from_weyl` from the live Weyl condensate. STP is an open-loop tube regularizer with a fixed `λ`. QUIPU is closed-loop: the boundary-distilled state feeds back to widen or narrow its own noise term.

---

## 3. Two results the paper gets "for free" that also explain QUIPU behavior

**Identity predictor (P5).** STP finds a learned projection head is *worse* than just using the identity function as the predictor, because on a locally-linear manifold the tangent direction already **is** the prediction — no network needed. This is a retroactive explanation for why QUIPU's Hebbian quipu edges work with no predictor at all: `weight += η_q · (1 − w)` (`mesh_slm.py:2247-2273`) is literally an identity-style update toward an observed target, consistent with the same "predictor with no fittable weights" architecture, and mirrors `tool_forge`'s stated ethos of "pure stdlib... no LLM call."

**Linear Representation Hypothesis / Figure 3.** The paper shows `v_Paris − v_France + v_Italy ≈ v_Rome` emerges *because* the four points lie on (approximately) a straight geodesic — vector arithmetic is a corollary of path linearity, not an independent fact about embeddings. QUIPU's proximity/metric-warp scoring (`score = QUIPU_GAIN·quipu_weight + (PROX_GAIN + WARP_GAIN·warp)·proximity + embed_dot(...) + MESH_FIELD_GAIN·mesh_field_8d`, `mesh_slm.py:78-89`) is the same relationship in flat torus-cell coordinates; the toroidal generalization of "vector arithmetic" is angular displacement on the torus with GR-style metric warp amplifying short-range proximity (`mesh_slm.py:87-89`: "consistent with GR: stronger field at shorter r").

---

## 4. Where the analogy breaks — cautions worth keeping

- **Extrinsic vs. intrinsic geometry.** STP's tube is a neighborhood in flat ambient ℝ^d around a chord. QUIPU's torus is a curved space with its own intrinsic metric (wraparound). "Locally linear" means something different in each: a chord in ℝ^d vs. a geodesic on T². Porting the STP loss onto torus-cell coordinates naively (subtracting raw `(i, j)` pairs) breaks at the wrap boundary — cell `(63, 0)` and `(0, 0)` are adjacent on the torus but 63 apart in raw coordinate subtraction. Any diagnostic built on torus positions has to embed through `(cos θ, sin θ, cos φ, sin φ)` or use `_torus_dist`, never raw coordinate differences. The companion plan handles this explicitly.
- **Fixed λ vs. adaptive coupling.** STP tunes `λ` once per experiment (§4, "Lastly we show how to tune λ in practice"). QUIPU's existing analogues (SiCi phase weight, rADAM pressure) are already *time-varying*, driven by live corpus/coherence state. A direct port that hard-codes `λ` would be a regression relative to how the rest of the system already operates — the plan couples it to `phase_weight` instead.
- **What "hidden state" means.** STP's `h_t` is a transformer's per-position residual-stream activation. QUIPU has no such object; the nearest analogue is the 7-D per-token embedding (`e_vision … e_entirety`, `mesh_slm_embed`), which is coarser (7 scalars, not a few-thousand-dimension vector) and updated by an explicit Hebbian rule rather than backprop. The STP loss transfers as a diagnostic on this 7-D space, but expecting transformer-scale geometric richness from a 7-D substrate would over-claim what the analogy supports.

---

## 5. Follow-up

The cheapest empirically-decisive next step — sample `s < r < t` from a training trajectory, compute the STP-style cosine gap on QUIPU's own torus geometry, log it, and see whether it keeps falling after `L_NTP`-style loss plateaus (the paper's P1 signature) — is planned in detail in **`docs/STP_DIAGNOSTIC_PLAN.md`**, scoped to `train_round()` in `src/quipu/mesh_slm.py`.

---

## Appendix A — Source paper (annotated pages 1–5, transcribed)

**Why transcribed instead of binary-embedded:** Markdown has no native mechanism for embedding a PDF's bytes, and rendering the remaining pages to images required `pdftoppm` (poppler-utils), which was not available in this session's sandbox (the shell tool itself was unavailable — `HYPERVISOR_VIRT_DISABLED`). The first five pages — which is where every one of the reader's annotations appears — were available as pre-rendered page images and are transcribed faithfully below, including the handwritten margin notes (best-effort reading of handwriting). The original PDF is provided alongside this doc so both are accessible together.

### Abstract (p.1)

> Large Language Models (LLMs) obey consistent scaling laws — empirical power-law fits that predict how loss decreases with compute, data, and parameters. While predictive, these laws are descriptive rather than prescriptive: they characterize typical training, not optimal training. Surprisingly few works have successfully challenged the data-efficiency bounds implied by these laws — which is our primary focus. To that end, we introduce the Geodesic Hypothesis, positing that token sequences trace geodesics on a smooth semantic manifold and are therefore locally linear. Building on this principle, we propose a novel Semantic Tube Prediction (STP) task, a JEPA-style regularizer that confines hidden-state trajectories to a tubular neighborhood of the geodesic. STP generalizes JEPA to language without requiring explicit multi-view augmentations. We show this constraint improves signal-to-noise ratio, and consequently preserves diversity by preventing trajectory collisions during inference. Empirically, STP allows LLMs to match baseline accuracy with 16× less training data on the NL-RX-SYNTH dataset, directly violating the data term of Chinchilla-style scaling laws and demonstrating that principled geometric priors can surpass brute-force scaling.

**Figure 1** (p.1): two-panel figure. **(a) Semantic Tube** — a geodesic (dotted line) with a "Semantic Tube" cylinder around it; hidden states scatter around the tube, with "signal" labeling the component along the tube and the perpendicular offset representing noise; token labels along the geodesic read "...an AI researcher... and won the Nobel Prize to..." **(b) Data Efficiency** — accuracy (%) vs. data size (1/32 to 1/2 of full dataset) for `L_NTP + L_STP` (ours) vs. `L_NTP` alone and several flop/lr/λ ablations; the combined-loss curve is roughly flat from 1/16 to 1/2 data size while the `L_NTP`-only curve degrades sharply as data shrinks.

### 1. Introduction (p.1–2, excerpt)

> We argue that empirical scaling laws characterize *typical* rather than *optimal* training, suggesting the rigid power-law barrier is an artifact of current objectives. The core limitation is next-token prediction: a local objective that conflates surface statistical noise with global semantic signal. We propose a fundamental shift: explicitly constraining hidden state dynamics to separate the error-free semantic trajectory from this noise.
>
> First, we formally demonstrate that, although tokens are discrete, token sequences can be modeled by an Ordinary Differential Equation (ODE). The Picard-Lindelöf (Existence and Uniqueness) Theorem (Coddington & Levinson, 1955) guarantees that if the velocity is smooth enough, there is only one possible path forward from any starting point. In other words, trajectories originating from distinct initial states will never intersect. In the context of LLMs, if the ODE model holds, this implies that error-free generations from distinct prompts maintain their semantic separation, theoretically ruling out mode collapse and preserving diversity.
>
> Next, we hypothesize that the Principle of Least Action (Lanczos, 1966) is at work. This principle states that the path taken by a system between two points minimizes the "Action" (the integral of the Lagrangian over time), resulting in a "straight line" or geodesic on the underlying manifold. We further hypothesize that, as the manifold is an artifact of the training process, it admits a smooth structure. Consequently, the geodesics are locally linear almost everywhere. In the context of LLMs, this implies that the trajectories of error-free token sequences — and by extension, the trajectories of error-free hidden states — are confined within a tube centered along a straight line.
>
> We designate this structure the **Semantic Tube** (Figure 1) and leverage it to regularize the LLM training process. The Semantic Tube posits that the noise — which causes deviations from the error-free trajectories — concentrates along the directions perpendicular to the tube. Let `s < r < t` denote the indices of three tokens. We define the *noise* term as `(h_r − h_s)⊥h_t−h_s`, representing the component of `h_r − h_s` perpendicular to `h_t − h_s`, and the *signal* term as `(h_r − h_s)∥h_t−h_s`, representing the component parallel to `h_t − h_s`. Minimizing the noise term is expected to improve the Signal-to-Noise Ratio (SNR) during training. We formulate this as an auxiliary loss term, the Semantic Tube Prediction (STP) loss `L_STP`, which can be seamlessly integrated into the training objective:
>
> `L = L_NTP + λ · L_STP`
>
> where `L_NTP` is the cross-entropy loss for Next Token Prediction (NTP) and `λ` is a hyperparameter controlling the strength of the STP loss.
>
> Semantic Tube draws inspiration from the Joint-Embedding Predictive Architecture (JEPA) (Assran et al., 2023; Baevski et al., 2022), which learns to predict the representation of one view based on another. In our approach, we postulate that any segment of a token sequence aligns with the global trajectory; consequently, the predictor reduces to an identity function.

**Predictions P1–P5** (p.2), stated verbatim:

> - (P1) `L_NTP` alone is insufficient for high-quality generation. Consequently, we expect to observe `L_NTP` plateau even as `L_STP` continues to decrease.
> - (P2) Semantic Tube improves SNR, resulting in superior data efficiency (Figure 1) and accuracy.
> - (P3) Semantic Tube preserves diversity.
> - (P4) We expect to see `λ ≪ 1` to accommodate instances where the geodesic deviates from a straight line.
> - (P5) The identity function serves as a superior predictor compared to learned projections.

> We conducted extensive experiments validating predictions (P1) through (P5). These results provide a strong indication that the Geodesic Hypothesis represents a simplified form of self-consistency for autoregressive sequence models.

### 2. Training and Inference Dynamics (p.2–3, excerpt)

> **2.1. Training ODE.** Let `x≤t` denote a token sequence of length `t`... Each hidden state `h_t` is subsequently unembedded to predict the next token `x_{t+1}`. During training, the predicted token `u(h_t)` may diverge from the ground truth `x_{t+1}`; this discrepancy constitutes the training loss. However, due to teacher forcing, we invariably feed the ground truth sequence `x≤t+1` into `f(·)` to generate `h_{t+1}`. Consequently, assuming a converged network where the loss is minimized, the training dynamics can be modeled as:
>
> `x_{t+1} = û ∘ f̊(x≤t)`     (1)
> `h_t = f̊(x≤t) + ε_t`       (2)
>
> ...
>
> **Proposition 2.1 (Training ODE).** *The LLM training process can be modeled as a solution in the token sequence space* `ℝ^{T×d_model}` *to the ODE:*
>
> `dx≤t = û ∘ f̊(x≤t) dt`
>
> [The theorem that follows] models `x≤t` as following a ballistic trajectory in `ℝ^{T×d_model}`. The Picard-Lindelöf Theorem guarantees that if `û ∘ f̊(·)` and its partial derivatives with respect to `x≤t` are continuous, the ODE admits a unique solution for a given initial condition. Consequently, within this ODE framework, sequences generated from distinct prompts (initial conditions) cannot intersect, theoretically ruling out mode collapse, and preserving diversity.
>
> **2.2. Mode Collapse at Inference Time.** Let `h*_t` denote the optimal trajectory of hidden states, defined as:
>
> `h*_t = h_t − ε_t = f̊(x≤t)`     (3)
>
> If `f̊(·)` is Lipschitz-continuous (Khalil, 2002), then the trajectory `h*` is also ballistic. However, `L_NTP` alone may not suffice to drive `ε_t` to zero. Recall the goal of `L_NTP` is to converge `u(h_t)` to `x_{t+1}`. Since the hidden state `h_t` is continuous while the token `x_{t+1}` is discrete, the training process can be modeled as finding the correct Voronoi cell (Okabe et al., 2000), without stipulating the exact location within the cell. This flexibility is necessary for the Picard-Lindelöf Theorem to apply: as illustrated in Figure 2, it allows error-free geodesics (`h*_t`) to traverse the same Voronoi cell at distinct locations, thereby avoiding intersection. Nevertheless, `h_t` may drift onto an incorrect geodesic within the cell, leading to mode collapse.
>
> This analysis indicates that `L_NTP` alone is insufficient for generation quality, strongly motivating an additional loss term (`L_STP`) to explicitly minimize the error `ε_t`. It also implies that within the correct Voronoi cell, `L_NTP` may plateau while `L_STP` continuously decreases. Therefore, (P1).

**Figure 2** (p.3): two hidden-state trajectories with similar prefixes ("...an AI researcher..." / "...an AI...") pass through the Voronoi cell of the "researcher" token at different locations, leading to different next hidden states and hence different next tokens. Because `L_NTP` cannot guarantee `h_t` converges to `h*_t` (optimal hidden state), `h_t` can be misplaced on another geodesic. This leads to mode collapse (a red dotted line mistakenly continues the generation, misattributing "Hinton's Nobel Prize" to an arbitrary person, "and won the Nobel Prize to <eos>"), or the error deviates in the opposite direction and precludes a winner.

> In Section B, we demonstrate that in the infinite-width limit (Yang & Littwin, 2021), the inference process can be modeled as a Stochastic Differential Equation (SDE) with a Brownian motion term.

### 3. Semantic Tube Prediction (p.3–4, excerpt)

> **3.1. Semantic Tube.** If the Principle of Least Action holds, the trajectories of the token sequence `x≤t+1` in Equation (1) must be geodesics, which are locally linear almost everywhere. Since `h*_t = f̊(x≤t)`, when `f̊(·)` is smooth enough, `h*_t` is also expected to be locally linear almost everywhere. Hence the **Geodesic Hypothesis**:
>
> *The trajectory of `x≤t ∈ ℝ^{T×d_model}` is locally linear almost everywhere. Similarly, the trajectory `h_t − ε_t ∈ ℝ^d` is locally linear almost everywhere.*
>
> **Definition 3.1 (Local Linearity).** A time-indexed trajectory `h*` is defined as locally linear if `∃τ, ∃ε` such that for any time indices `s < r < t` satisfying `|t − s| ≤ τ`, we have:
>
> `‖(h*_r − h*_s)⊥h*_t−h*_s‖₂ ≤ ε`     (4)
>
> where `x⊥v` denotes the component of vector `x` that is perpendicular to vector `v`.
>
> **Lemma 3.2 (Straightening Lemma).** *If* `h_s = h*_s`, `h_t = h*_t`, *and* `L_STP ≤ ε` *for all* `r` *satisfying* `s < r < t`, *then*
>
> `‖(h_r − h_s)⊥h*_t−h*_s‖₂ ≤ √2ε · ‖h_r − h_s‖₂`
>
> **Theorem 3.3 (Semantic Tube).** *If* `h*` *is locally linear and for all* `r` *satisfying* `0 ≤ s < r < t ≤ τ`, `L_STP → 0`, *then*
>
> `‖h_r − h*‖₂ ≲ ε`
>
> *Proof sketch.* Only prove for the case `h_s = h*_s` and `h_t = h*_t`. In this scenario, `‖h_r − h_s‖₂ = ‖h*_r − h*_s‖`. Applying the triangle inequality yields `‖h_r − h*_r‖₂ ≤ ‖h*_s − h*_r‖₂ + ε_r`. Notice `h*_r` and `h*_s` are fixed; by Theorem 3.2 and the triangle inequality, `‖(h_r − h_s)⊥h*_t−h*_s‖₂ → 0`. By Theorem 3.1 and the triangle inequality, it follows that `‖h_r − h*‖₂ ≲ ε`. ∎
>
> In LLMs, it is standard to assume all sequences begin with `<bos>` and end with `<eos>`; thus, it is reasonable to assume the boundary conditions `h_0 = h*_0` and `h_τ = h*_τ`. This is formally proven in Section E, via the proof of Theorem 3.3, which completes the corollary.
>
> In practice, the indices `s < r < t` are selected randomly. Consequently, minimizing `L_STP` effectively drives `E[1 − cos(h_t − h_r, h_r − h_s)] → 0`. By Markov's inequality, for any `ε`, `P(1 − cos(h_t − h_r, h_r − h_s) > ε) → 0`. This leads to the following corollary.
>
> **Corollary 3.4 (Random Tube).** *For randomly selected* `s < r < t`, *if* `L_STP → 0`, *then for any* `ε`,
>
> `P(‖h_r − h*‖₂ > ε + ε) → 0`
>
> Theorem 3.4 implies that if `L_STP → 0` for a given sequence, then with high probability, the trajectory of the sequence's hidden states is confined within a tube centered around the optimal trajectory `h*`.
>
> However, at inference time, the Brownian motion term diverges into a cone whose radius scales as `∝ σ_t√t`, see Section F for details.
>
> **3.2. Practical Considerations.** Since the forward pass naturally computes `h_s`, `h_r`, and `h_t`, the STP loss introduces negligible computational overhead — primarily the cost of computing cosine similarity. This is significantly more efficient than the fractional extra forward passes required by LLM-JEPA (Huang et al., 2025). Furthermore, because indices `s`, `r`, and `t` can be selected randomly, STP eliminates the need for manual scaffolding of a two-view structure. In summary, STP effectively addresses the two primary limitations that have hindered the broader adoption of LLM-JEPA. Additionally, STP avoids the complexity of a predictor network (often a requirement in LLM-JEPA), as local linearity implies an identity predictor. Like LLM-JEPA, the STP loss is applied exclusively during training and is not required at inference time.

**Implementation note** (p.5, excerpt from §4): computed with HuggingFace `transformers`; per-token `hidden_state h` from the last layer, random indices `s < r < t`, loss `1 − cos(h_t − h_r, h_r − h_s)`.

> **3.3. Related Work** (excerpt). Our approach addresses the classic **Exposure Bias** problem (Bengio et al., 2015)... The problem arises because the model is trained with **Teacher Forcing** (Williams & Zipser, 1989) — conditioning on the ground-truth history — but must rely on its own potentially drifting predictions during inference. Although Maximum Likelihood Estimation (`L_NTP` in the case of LLMs) is empirically effective, Huszár (2015) argues that it optimizes an objective different from generation quality, motivating our combined loss `L_NTP + L_STP`.
>
> **JEPAs** (Assran et al., 2023; Baevski et al., 2022) learn predictive representations across views, offering theoretical benefits (Littwin et al.; 2024) despite the risk of dimensional collapse (Jing et al., 2021; Kenneweg et al., 2025)... LLM-JEPA (Huang et al., 2025) is bottlenecked by manual two-view scaffolding and the computational cost of additional forward passes, neither is a problem for `L_STP`.
>
> Our framework extends the philosophy of **Energy-Based Models** (EBMs) (LeCun et al., 2006), which learn to assign low energy to compatible configurations of variables. While EBMs and recent architectures like JEPA (LeCun, 2022) typically minimize energy at specific states, our approach invokes the Principle of Least Action to minimize the action — the integral of the Lagrangian along the generation trajectory. By enforcing geodesic constraints via `L_STP`, we generalize state-wise (or local) energy minimization to trajectory-wise action minimization, ensuring the generation follows the path of least resistance.
>
> **Scaling Laws** govern the power-law relationship between compute, data, and parameters in both pre-training (Kaplan et al., 2020; Hoffmann et al., 2022) and fine-tuning (Zhang et al., 2024)... `L_STP` enhances the training SNR directly, obviating the need for explicit data subset selection.
>
> **SDE/ODE Perspective**: Kong et al. (2020) interpreted ResNets as "Neural SDEs" with a Brownian motion term. While Tong et al. (2025) recently adapted ODEs for LLMs, they model evolution across network depth (layers). Our work takes an orthogonal approach, focusing instead on the temporal dynamics of hidden states across the token sequence.
>
> **The Linear Representation Hypothesis (LRH)** (Park et al., 2024; 2025) posits that simple concepts are encoded as directions in the representation space, whereas the Geodesic Hypothesis suggests that both simple and composed concepts (expressed as token sequences) follow locally linear trajectories. Consequently, the vector arithmetic observed in LRH (`v_Paris − v_France + v_Italy ≈ v_Rome`) emerges naturally from path linearity (`v_Paris, v_to, v_France, v_is, v_Rome, v_to, v_Italy` align on almost a straight line, see Figure 3).
>
> **The Manifold Hypothesis** (Kiani et al., 2024; Robinson et al., 2025; Whiteley et al., 2025) posits that learned representations form a simple and smooth manifold. Under the Geodesic Hypothesis, this structure is a natural consequence of the Principle of Least Action.
>
> **The Curvature Straightening Phenomenon** (Hosseini & Fedorenko, 2023; Hénaff et al., 2021) observes that the training process tends to straighten the curvature between consecutive tokens. We interpret this as a manifestation of the underlying geodesic, which approximates a straight line.
>
> **The Neural Tangent Kernel (NTK)** simplifies infinite-width dynamics (Jacot et al., 2018), a framework generalized to Transformers (Hron et al., 2020; Yang & Littwin, 2021) and compatible feature learning regimes (Yang & Hu, 2021). Seleznova & Kutyniok (2022) note the importance of the depth-to-width ratio; modern LLMs typically operate in the requisite width ≫ depth regime.
>
> The application of geodesic geometry to LLMs remains underexplored, with existing studies primarily restricted to interpolating representations across models (Deng et al., 2025; Yu et al., 2024).

**Figure 3** (p.5): "When the sentence aligns on a geodesic, the concept direction naturally aligns" — shows `Paris → to → France → is → Rome → to → Italy` labeled as "Geodesic" (dashed arrow) with a separate "Concept direction" arrow from Paris to Rome running almost parallel to it.

### 4. Experiments (p.5, opening)

> We conduct extensive experiments to show the performance of Semantic Tube across models, datasets, and model sizes. We also show that accuracy barely budges when the training dataset is halved. Both accuracy and data efficiency are solid evidence that Semantic Tube improves SNR. We ablate on various setups, including LLM-JEPA style explicit two-views and curvature straightening. Lastly we show how to tune `λ` in practice.
>
> Implementing `L_STP` is straightforward with HuggingFace `transformers`. When computing loss, we grab per-token `hidden_state h` from last layer, pick (random) indices `s < r < t`, and compute `1 − cos(h_t − h_r, h_r − h_s)`. Across all experiments, we follow LLM-JEPA (Huang et al., 2025) to pick 5 random seeds: 82, 23, 37, 84, and 4, and report both mean accuracy and standard deviation. This also allows us to report `p`-value of paired, single-tailed `t`-Test. We inherit optimal number of epochs and learning rate from LLM-JEPA. `λ` is separately tuned.
>
> **4.1. Loss Landscape.** We begin by analyzing the loss landscape by fine-tuning Llama-3.2-1B-Instruct (Grattafiori et al., 2024) on the NL-RX-SYNTH (Locascio et al., 2016) dataset. Figure 4(a) demonstrates that in regular fine-tuning, minimizing `L_NTP` does not automatically minimize `L_STP`. With the Semantic Tube, however, `L_STP` continues to decrease even after `L_NTP` plateaus, corroborating (P1). Moreover,...

*[Transcription ends at the bottom of page 5 — pages 6 onward were not retrievable in this session; see the note above.]*

### Reader annotations (verbatim locations, best-effort reading of handwriting)

| Page | Near | Annotation (as read) |
|---|---|---|
| 4 | Lemma 3.2 / proof | Strikethrough math + "Torus Quipu"-style arrow (see p.5 entry below — same note, drawn across the page break) |
| 4 | "SiCi" margin, top-left | "SiCi Tangential leads to STP loss in `h*` resulting predictions of future interaction[s]... local linearity becoming a certain[ty]... reducing through 3 loss[es] flops" |
| 4 | Theorem 3.3 / "Related Work" margin | "rADAM"; "potential Dim collapse, but still less than JEPA"; "Probability Bounding"; "What about Synthetic View Conundrum" |
| 4 | Bottom margin, beside Teacher Forcing paragraph | "Teacher Forcing? Likelihood... if has both real places... but even with... validated by experience of the same idea" |
| 5 | Beside "geodesic geometry to LLMs remains underexplored" | "Torus Quipu" |

---

## Sources

- Huang, LeCun, Balestriero — *Semantic Tube Prediction*, arXiv:2602.22617v1 (attached PDF, pp. 1–5 with reader annotations)
- `QUIPU/LEARNINGS.md`, `README.md`, `CHANGELOG.md`, `RELEASE_NOTES.md`
- `QUIPU/docs/SYSTEM_ENTIRETY_ANALYSIS.md`
- `QUIPU/src/quipu/mesh_slm.py`, `ueqgm_engine.py`
