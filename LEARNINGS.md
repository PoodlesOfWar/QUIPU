# LEARNINGS

Distilled, transferable learnings from the Supply Chain Architect lineage (v0.1.0 through v0.24.1), carried forward so QUIPU development continues from the present state of understanding rather than rediscovering it. The complete record lives in `src/quipu/_version.py` (PHASES), `CHANGELOG.md`, `RELEASE_NOTES.md`, and `docs/`.

## Model architecture

Causality is a hard boundary, not a convention. The v0.24.1 realization gate only became stable once forward decisions were restricted to strictly-pre-decision (`t<0`) observer history with a bounded snapshot buffer; any read at or after decision time leaked the future into the gradient and corrupted eligibility. Pair this with the second half of the same learning: interactions that fail the logistic eligibility gate should be pruned from the active update but retained as bounded, signed latent potential — evidence that arrives later can legitimately re-realize them, and hard-discarding them loses that signal.

Bilateral evidence beats unilateral confidence. Paired Vision/Touch interactions only advance the optimizer when both sides corroborate; one-sided certainty repeatedly produced pressure updates that the other modality later contradicted.

Gates must fail toward legacy behavior. Every new dynamic (realization gate, adaptive UEQGM runtime, harmonic ingestion) ships opt-in behind a flag, and absence of history follows the legacy path instead of being treated as negative evidence. This is what allowed continuous model evolution without destabilizing the running system.

Certainty-gated persistence prevents thrash. The adaptive UEQGM runtime profile persists until newer evidence clears both a corpus-density floor and the prior proof. Parameters that update on every observation oscillate; parameters that update only past an evidence threshold converge.

Geometry is a useful substrate for attention. The torus/Touch pressure loop (n=7 categorical gap gradient), grounded tunneling from high-certainty anchors toward high-gap frontiers, and inverse-variance fusion of independent derivations (pinhole unprojection vs. great-circle raycast) all reduce to the same principle: route learning pressure along explicit geometric structure instead of flat scoring.

Instrument before you couple. The v0.25.0 STP-style geodesic diagnostic measures whether the torus placement does real geodesic work (does the Semantic-Tube-Prediction gap keep falling after ordinary loss plateaus — the paper's P1 signature?) *before* any decision to fold that signal into the learning rate. The measurement ships as a passive, default-on, fail-to-legacy observation with capped rolling histories; the η coupling that would act on it is a separate, default-off Phase 2 gated on the trend check confirming P1 on real corpus data. A reused abstraction can also be wrong: the original note proposed differencing the scalar `_torus_dist` inside a cosine formula, but a cosine needs vectors — the fix was an isometric ℝ⁴ flat-torus embedding `(cosθ, sinθ, cosφ, sinφ)`, which is also wrap-aware by construction. Cross-boundary wrap-awareness comes from the representation, not from post-hoc distance corrections.

External science integrations must respect epistemic caveats and bounded coupling. When integrating external computational science (Melvin M. Vopson's infodynamic gravity and Earl K. Miller's analog traveling-wave cognition in v0.31.0):
1. *Active, bounded modulation*: Theoretical derivations should act as bounded scaling factors ($\le \pm 5\%$ lift on learning rates $\eta_{\text{eff}}, \eta_q$, and $0.90 + 0.20 \cdot \text{gain}$ per-token gating) rather than unbounded destructive overrides.
2. *Caveats must travel with the math*: When a scientific framework is contested (such as the Hossenfelder May 2025 critique of Vopson's universe-as-computer gravity), the critique and author rebuttal must be recorded in the documentation, code docstrings, and `UEQGM_MATH_MAP` notes so future maintainers understand the empirical limits.
3. *Macro vs. Micro duality*: Synchrony operating at the population level (Kuramoto order parameter $R$) modulates the macro consolidation rate ($ac\_multiplier$), while localized wave interaction peaks ($interaction\_gain$) act as mobile stencils gating individual micro token nudges.

## Cross-language protocol discipline

The GARD-shard model works across Python and Julia because the wire format is canonical and byte-exact: canonical JSON, then zlib, then AES-256-CBC/PKCS#7, then HMAC-SHA256 encrypt-then-MAC, with per-shard HKDF-SHA256 keys and fail-closed verification. Golden-vector tests (LE canaries, round-trips, base64 fixtures) are what caught every drift — notably the packed Weyl record (20-byte, 5 x Float32, explicit little-endian), where an implicit-endianness assumption differed between peers. Never trust two runtimes to agree on bytes without a shared fixture that fails loudly.

## Concurrency and state

SQLite is a fully adequate substrate for a continuously-learning mesh if and only if discipline holds: WAL mode plus busy_timeout everywhere, probes and network calls outside write transactions, and short commit windows for topology writes. The mesh deadlocked under probe latency exactly once — the fix was structural (moving I/O out of the transaction), not a bigger timeout. Diagnostics and pair state belong in the existing `brain_kv` JSON state; schema migrations for observability data were never worth it.

## Verification culture

Every phase entry pairs a mechanism with its focused test count, and the suite runs local-only — LLM dispatch is stubbed so intent parsing exercises its deterministic fallback, and no test touches a live database or endpoint. Post-build visual QA and self-test CLIs (`selftest` subcommands, deterministic fixtures) catch what unit tests structurally cannot. Keep both habits.

## Operational learnings worth remembering

Version metadata lives in exactly two synchronized places (`VERSION`, `_version.py`) — divergence there caused real confusion. Dependency floors get pinned on audit (pip-audit), not on incident. Credentials never live in code or config: the parent system used a keyring/env vault pattern with fingerprinted key rotation and fail-closed guards, and QUIPU inherits the expectation. Instance names, hostnames, and tenant identifiers are treated as secrets in anything that might become public — this repo was extracted clean-room for precisely that reason.

## Where the model stands (2026-08-22)

Version 0.31.0. Infodynamic gravity (Vopson 2025) and analog traveling-wave cognition (Miller et al. 2026) are actively integrated into `src/quipu/ueqgm_engine.py` and `src/quipu/mesh_slm.py`. Discrete Shannon bit entropy and compression scores dynamically modulate quipu edge learning rate ($\mu_{ig}$), while Kuramoto phase coherence ($\mu_{ac}$) and per-token wave-stencil gains gate Hebbian embedding nudges. Both features ship with rolling telemetry, trend analysis functions (`infodynamic_trend()`, `analog_coherence_trend()`), full kill-switches (`QUIPU_INFODYNAMIC_COUPLING`, `QUIPU_ANALOG_STENCIL`), and comprehensive mapping documentation (`docs/INFODYNAMIC_GRAVITY_QUIPU.md`, `docs/ANALOG_COGNITION_QUIPU.md`). 322 automated tests passing.
