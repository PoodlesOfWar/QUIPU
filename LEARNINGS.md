# LEARNINGS

Distilled, transferable learnings from the Supply Chain Brain lineage (v0.1.0 through v0.24.1), carried forward so QUIPU development continues from the present state of understanding rather than rediscovering it. The complete record lives in `src/quipu/_version.py` (PHASES), `CHANGELOG.md`, `RELEASE_NOTES.md`, and `docs/`.

## Model architecture

Causality is a hard boundary, not a convention. The v0.24.1 realization gate only became stable once forward decisions were restricted to strictly-pre-decision (`t<0`) observer history with a bounded snapshot buffer; any read at or after decision time leaked the future into the gradient and corrupted eligibility. Pair this with the second half of the same learning: interactions that fail the logistic eligibility gate should be pruned from the active update but retained as bounded, signed latent potential — evidence that arrives later can legitimately re-realize them, and hard-discarding them loses that signal.

Bilateral evidence beats unilateral confidence. Paired Vision/Touch interactions only advance the optimizer when both sides corroborate; one-sided certainty repeatedly produced pressure updates that the other modality later contradicted.

Gates must fail toward legacy behavior. Every new dynamic (realization gate, adaptive UEQGM runtime, harmonic ingestion) ships opt-in behind a flag, and absence of history follows the legacy path instead of being treated as negative evidence. This is what allowed continuous model evolution without destabilizing the running system.

Certainty-gated persistence prevents thrash. The adaptive UEQGM runtime profile persists until newer evidence clears both a corpus-density floor and the prior proof. Parameters that update on every observation oscillate; parameters that update only past an evidence threshold converge.

Geometry is a useful substrate for attention. The torus/Touch pressure loop (n=7 categorical gap gradient), grounded tunneling from high-certainty anchors toward high-gap frontiers, and inverse-variance fusion of independent derivations (pinhole unprojection vs. great-circle raycast) all reduce to the same principle: route learning pressure along explicit geometric structure instead of flat scoring.

## Cross-language protocol discipline

The GARD-shard model works across Python and Julia because the wire format is canonical and byte-exact: canonical JSON, then zlib, then AES-256-CBC/PKCS#7, then HMAC-SHA256 encrypt-then-MAC, with per-shard HKDF-SHA256 keys and fail-closed verification. Golden-vector tests (LE canaries, round-trips, base64 fixtures) are what caught every drift — notably the packed Weyl record (20-byte, 5 x Float32, explicit little-endian), where an implicit-endianness assumption differed between peers. Never trust two runtimes to agree on bytes without a shared fixture that fails loudly.

## Concurrency and state

SQLite is a fully adequate substrate for a continuously-learning mesh if and only if discipline holds: WAL mode plus busy_timeout everywhere, probes and network calls outside write transactions, and short commit windows for topology writes. The mesh deadlocked under probe latency exactly once — the fix was structural (moving I/O out of the transaction), not a bigger timeout. Diagnostics and pair state belong in the existing `brain_kv` JSON state; schema migrations for observability data were never worth it.

## Verification culture

Every phase entry pairs a mechanism with its focused test count, and the suite runs local-only — LLM dispatch is stubbed so intent parsing exercises its deterministic fallback, and no test touches a live database or endpoint. Post-build visual QA and self-test CLIs (`selftest` subcommands, deterministic fixtures) catch what unit tests structurally cannot. Keep both habits.

## Operational learnings worth remembering

Version metadata lives in exactly two synchronized places (`VERSION`, `_version.py`) — divergence there caused real confusion. Dependency floors get pinned on audit (pip-audit), not on incident. Credentials never live in code or config: the parent system used a keyring/env vault pattern with fingerprinted key rotation and fail-closed guards, and QUIPU inherits the expectation. Instance names, hostnames, and tenant identifiers are treated as secrets in anything that might become public — this repo was extracted clean-room for precisely that reason.

## Where the model stands (2026-07-20)

Version 0.24.1. The realization gate is opt-in (`BRAIN_USE_REALIZATION_GATE=1`) and verified (65 tests green in the parent). The unified Julia MESH-SLM-SCM-GLM-GNN model is the current generation, with inference/compression peers wire-compatible at v2.2. Known open threads from the parent lineage: the self-train loop was replaying zero samples because dispatch logging stalled (see the v0.22.18 System Entirety analysis) — QUIPU should wire a corpus/telemetry feed before resuming self-training; and ensemble-weight learning (online SGD over model outcomes) remains in the parent app, not extracted, so QUIPU currently learns from its own store only.
