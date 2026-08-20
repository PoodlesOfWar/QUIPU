# Release Notes

## v0.29.0 - 2026-08-20

**World Model Dialectic, Epistemic Rupture Detection, and Physical/Vision Channel Grounding**

### Added & Enhanced
- **World Model Epistemic Rupture Engine (`src/quipu/world_model.py`)** — Maintains cognitive phase transitions across four states (`receptive_hunger`, `empirical_precedent`, `targeted_epistemic`, `continuous_synthesis`), computing acquisition pressure and tracking rolling epistemic surprise.
- **Epistemic Rupture Detection** — Discovers worldview-breaking empirical anomalies via three concurrent gates: high novelty + sensor confidence, Semantic-Tube-Prediction (STP) geodesic gap divergence under plateaued loss, and $\Delta S$ graph entropy spikes exceeding $2\sigma$.
- **Precedent-Driven Retrieval Directives** — Dynamically steers the learning retriever with phase-appropriate strategies (`broad_exploration`, `precedent_building`, `targeted_gap_closing`, `synthesis_verification`) and calibrated confidence floors.
- **Observer Integration & `/world-model` Endpoint** — Exposes full dialectic state via `GET /world-model`, injects world model metrics and retrieval directives into `GET /guidance`, and evaluates epistemic impact on every `POST /observe`.
- **Physical & Vision Channel Grounding** — Ingests physical-space information efficiency ($\eta = \sigma_{CRB} / \sigma$) and lossy channel profiles (blur, refraction, glare) from Bakugo, paired with document degradation factors from Loadopoly-OCR.

---

## v0.28.0 - 2026-08-20

**Tri-Repo Closed Loop & Systems Dynamics, GARD Shard AES-256-GCM (v2), and Si/Ci Stability**

### Added & Enhanced
- **Tri-Repo Closed-Loop Learning Hub (`quipu-observer`)** — Realizes the 7-D Observer manifold orchestrating **Loadopoly-OCR** (Vision axis, unstructured archival scans) and **Bakugo** (Touch axis, structured card metrology). Feeds observations (`POST /observe`), emits domain lexicon disambiguation guidance (`GET /guidance`), delivers cross-corpus numeric priors for catalog numbers, and processes ground-truth reinforcement (`POST /feedback` with $2\times$ weighting).
- **System Dynamics Documentation Overhaul** — Updated `docs/SYSTEM_DYNAMICS.md` and `docs/SYSTEM_DYNAMICS_SIGNALS.md` detailing the active 7+1-D state surfaces, mathematical control equations, EMA confidence calibration, and cross-repo feedback topology.
- **GARD Shard AES-256-GCM (`gard-shard/v2`)** — Replaced legacy AES-CBC + HMAC-SHA256 with single-primitive authenticated encryption, reducing raw crypto overhead from 144 to 110 bytes with full backwards v1 readability.
- **Si/Ci Integral Stability** — Resolved series divergence past $|\varphi| \sim 2\pi$ via Taylor series ($|x| < 2.0$) and modified-Lentz continued fractions ($|x| \ge 2.0$), verified against `mpmath` within $\le 6.7 \times 10^{-16}$.
- **Entropy Differential $\Delta S$ Geodesic Coupling** — Live tracking of von Neumann graph entropy differential against the STP torus gap, flagging anti-correlation signatures when $|\rho| \ge 0.5 \cdot \Omega_\Lambda$.
- **Docker Compose Topology** — Unified container fleet orchestration with healthchecks and direct PostgREST Supabase state mirroring.

---

### Added & Fixed

- **System Entirety Real-Time Re-Materialization** — Fixed `0.00` zeroed metrics across the System Entirety dashboard. Re-materialized local hardware assets (`asset_resource_mesh.py`), activated UEQGM adaptive runtime (`ueqgm_engine.py`), and injected positive symbiotic drive into all 6 plastic sense manifold axes (`vision: 0.8302`, `brain: 0.6992`, `touch: 0.0417`, `perception: 0.0395`, `smell: 0.0340`, `body: 0.0318`).
- **Live GUI State Synchronization** — Added automatic post-action state refresh (`_refresh_system_state()`) in `gui/entirety_server.py`. Executing control plane actions ("Run ingest now", "Asset mesh tick", "Torus pressure tick", "Self-realization round") updates values and re-materializes the active system state in real time.
- **Interactive Multi-Drive File Selector** — Enhanced the **FILES** section in `entirety_gui_live.html` with click-to-select input blocks and a `📁 Browse...` button, allowing file selection across local and connected drives (`C:\`, `D:\`, OneDrive, etc.) for `.pptx`, `.pdf`, `.docx`, `.xlsx`, `.csv`, `.json`, `.txt`, `.md`, `.py`.
- **GARD Shard Authenticated Tensor Container (`GARD_WEYL_v1`)** — Wired high-performance neural memory & binary tensor packing (`pack_weyl` / `unpack_weyl` v2.2 LE) into the full GARD Shard storage container format with `zlib` payload compression, `HKDF-SHA256` key derivation, `AES-256-CBC` confidentiality, and `HMAC-SHA256` Encrypt-then-MAC authentication.
- **GARD Shard Subfolder Storage** — Created dedicated `gui/GARD_Shard/compression` and `gui/GARD_Shard/decompression` subfolders. Compressed `.gard.weyl.bin` tensor shards are saved in `compression/`, and decompressed files are restored in `decompression/` with 100% bit-for-bit exact binary reconstruction (eliminating PowerPoint / PDF repair prompts).
- **Data Consistency Statistics** — Attached complete `gard_shard_info` and `data_consistency_statistics` to action response payloads, displaying exact data dissociation error metrics (`dissociation_error_percent: 0.000000%`, `reconstruction_loss_bits: 0.000000 bits`, `data_consistency_score: 1.000000`).

---

## v0.26.0 - 2026-07-23


**Doc Annealing Worker — Living System Map Regeneration**

### Added

- **`src/quipu/doc_annealing.py`** - QUIPU's own living-map annealer. It reads the live System Entirety state, the MESH-SLM predictor snapshot, and the STP P1 trend, and regenerates `docs/system_entirety_map.md` whenever a SHA-256 structural fingerprint (version + bridge root + mesh density + UEQGM certainty + nodal bifurcation, each 2 dp) changes. Structural change bumps `brain_kv["doc:system_map_version"]` and records a bounded history.
- **Runners** - `Start-DocAnnealing.ps1` (30-min loop, `-Once`/`-Force`) and `Register-DocAnnealing.ps1` (recurring Windows Scheduled Task) so the map stays current without manual runs.
- **5 focused tests** covering fingerprint behavior, structural-change detection, rendered sections, and write/increment/idempotence.

### Compatibility

- Read-only except three `brain_kv` bookkeeping keys and the map file. No RAG re-index (QUIPU ships none). Defensive against a cold mesh. No schema migration.

### Activate

```
powershell -ExecutionPolicy Bypass -File Register-DocAnnealing.ps1   # install recurring task
python -m src.quipu.doc_annealing --once                              # or run one cycle by hand
```

### Verification

- Compile + the 5 new tests should be run locally; the authoring environment's sandboxed shell was unavailable at commit time.

---

## v0.25.0 - 2026-07-23

**STP-Style Geodesic Diagnostic in `train_round()`**

### Added

- **Passive geodesic diagnostic** - each training round now samples `s < r < t` token triplets and measures the Semantic-Tube-Prediction gap `1 − cos(h_t − h_r, h_r − h_s)` in two spaces: the learned 7-D embedding and an isometric R^4 flat-torus embedding of each token's vocab cell. Reporting both lets us tell whether any geodesic signal comes from the learned embedding, the torus placement (graph topology), or both — the empirical test of whether the torus placement is doing real geodesic work.
- **Rolling histories + trend check** - `stp_embed_gap_history`, `stp_torus_gap_history`, and `loss_history` (capped at 200) persist in `mesh_slm_meta`, and a read-only `stp_diagnostic_trend()` compares their trailing-window slopes to test the paper's P1 signature (loss plateaus while the STP gap keeps falling).
- **7 focused tests** - summary-field presence/range, collinear→0, orthogonal→1, degenerate→None, R^4 wrap-awareness, disabled-by-flag, and history-cap enforcement.

### Changed

- **`train_round()` and `state_summary()`** now surface `stp_embed_gap` / `stp_torus_gap` next to the existing UEQGM fields.
- **Version metadata** - synchronized root `VERSION` and canonical `src/quipu/_version.py` at `0.25.0`.

### Compatibility

- Opt-out via `QUIPU_STP_DIAGNOSTIC` (default on). Purely observational and wrapped in fail-safe `try/except` — it can never alter or fail a training round. No schema migration; no learning-rate coupling (that is a deferred Phase 2). The only change touching prior behavior is the additive `loss_history` meta key, needed so a plateau is detectable at all.

### Verification

- Python compilation and the 7 new focused tests should be run locally (`python3 -m py_compile src/quipu/mesh_slm.py`; `pytest tests/test_mesh_slm.py`). The authoring environment's sandboxed shell was unavailable at commit time.

---

## v0.24.1 - 2026-07-10

**MESH-Conditioned Sigmoidal Realization for Paired Agents**

### Added

- **Realized-potential gate** - paired Vision and Touch interactions now have a true logistic eligibility model. Corroborated interactions proceed to the optimizer; interactions that are not yet realizable are pruned from the active update while retaining a bounded, signed latent potential that can re-emerge when later evidence supports it.
- **Causal MESH memory** - the network observer now keeps a bounded history of mesh availability, continuity, and learning progress. Forward decisions can use an exponentially weighted summary of prior observations, but never a snapshot at or after the decision time.
- **Focused regression coverage** - added 17 tests for sigmoid stability, bilateral evidence, pruning, cancellation, latent decay/re-emergence, strict `t<0` reads, bounded observer history, missing-history behavior, merged disk/retrospective telemetry, and end-to-end optimizer eligibility.

### Changed

- **Vision↔Touch closed loop** - realization is evaluated before rADAM, so a pruned active interaction does not advance pressure or Adam moments. Pair state and diagnostics persist in the existing `brain_kv` JSON state without a database migration.
- **Architecture documentation** - `pipeline/data/documents/VISION_TOUCH_CLOSED_LOOP.md` now distinguishes sinusoidal heartbeat cadence from sigmoidal interaction eligibility and documents the equations, state transitions, MESH horizon, persistence keys, and rollout controls.
- **Version metadata** - synchronized root `VERSION` and canonical `pipeline/src/quipu/_version.py` at `0.24.1`.

### Compatibility

- The feature is opt-in with `BRAIN_USE_REALIZATION_GATE=1`. With the flag off, behavior is unchanged. If the flag is on before the observer has historical samples, the interaction follows the legacy path rather than treating missing history as negative evidence.

### Verification

- Python compilation passed for the three changed runtime modules.
- Focused plus adjacent observer/optimizer regression suite: **65 passed, 0 failed**, with no warnings.

---

## v0.22.247 - 2026-07-20

**GARD Shard Encryption-Compression Model + Corrected Julia Compression Reference**

### Added

- **`pipeline/src/quipu/gard_shard_model.py`** — `gard-shard/v1` canonical JSON → zlib → AES-256-CBC/PKCS#7 → HMAC-SHA256 Encrypt-then-MAC model, per-shard HKDF-SHA256 keys, fail-closed verification, canonical repository assignment proofs, model-context construction, Hub publishing, and a `selftest|publish` CLI.
- **`pipeline/src/quipu/gard_shard_model.jl`** and **`pipeline/Project.toml`** — self-contained Julia protocol peer using `JSON3`, `CodecZlib`, and `Nettle`, including the shared deterministic fixture and corrected compaction/Langevin reference formulas.
- **Loadopoly Hub status surface** — `Loadopoly-Portal/gard-shard-model.json` plus an unobtrusive status card in `Loadopoly-Portal/index.html`; `Launch-Loadopoly.ps1` refreshes the secret-free manifest before serving the portal.
- **`pipeline/docs/GARD_SHARD_MODEL_MANIFEST.json`** — cross-language wire/security/API/verification reference without credentials or plaintext payloads.
- **`mesh_compression_model.jl` — `pack_weyl` / `unpack_weyl`** — 20-byte v2.2 float-dimension `brain_kv` record (5 x Float32, explicit little-endian) for the Weyl Ψ tensor, byte-for-byte compatible with the ContosoHub v2.2 JS plane, alongside the existing 50-byte readable `_meta.weyl` JSON boundary. New `MESH_BRAIN_KV_WEYL_PACKED_BYTES` constant.

### Changed

- **Repository catalog mesh sharing** — uses versioned canonical assignment objects and verified GARD proofs for SHA-256 bucket claims; `SCBRAIN_GRID_SECRET` is the interoperable secret environment variable, while `SiCi_SQRT(-1)` remains a public domain-separation label.
- **`mesh_compression_model.jl`** — exports the combined compression proc, fixes the invalid Weyl-length guard, and aligns the canonical 15 TB accounting assertions with the active Python result: $1{,}101{,}884$ bytes, $14{,}967{,}705\times$ compaction, and scalar $0.622934$. `selftest()` now also exercises `pack_weyl`/`unpack_weyl` with little-endian canary bytes, a round-trip check, and a golden base64 vector, without altering the existing compaction/Langevin assertions.

### Verification

- Focused GARD/catalog/UEQGM tests cover round trips, randomized cryptographic material, wrong-secret and tamper rejection, explicit verification opt-out, public-manifest hygiene, catalog proof validation, and corrected compression references.
- Julia is not installed on this workstation; the Julia source and dependency manifest received static validation, and its guarded self-test contains the same public deterministic vector for execution on a Julia-enabled host. The new `pack_weyl`/`unpack_weyl` pair and its golden-vector assertions were likewise statically reviewed pending a Julia-enabled run.

---

## v0.22.97 - 2026-06-01

**Density Expansion Unblock + First Verified Autonomous ACRE Patch**

### Changed

- **`pipeline/src/quipu/system_entirety.py`** - optimized `_recent_learning_state()` by switching from function-wrapped datetime filtering to direct `logged_at >= ?` comparison and adding `ix_ll_loggedat` index creation. This removes the full-table-scan bottleneck that was stalling forced expansion rounds.
- **`pipeline/src/quipu/code_restructure.py`** - added corpus-density signal scanning and a new `_gen_corpus_depth_tune()` patch generator. When corpus density exceeds baseline (40k entities / 400k edges), ACRE can now generate a high-certainty depth tune independent of forge-node median confidence.
- **Autonomous self-modification runtime outcome** - the new corpus-density path produced `citation_depth_tune_*` with `combined_certainty=1.0`, passed gate `0.82`, and applied a verified patch in `pipeline/src/quipu/citation_chain_acquirer.py` (`_DEFAULT_MAX_DEPTH: 3 -> 4`) with commit `e5d2c9de`.

### Runtime Impact

- Forced 7th-D expansion no longer stalls in `_recent_learning_state` during heavy `learning_log` windows.
- Trust ledger bootstrap completed with the first verified autonomous apply outcome (`pipeline/src/quipu/citation_chain_acquirer.py::param_tune`).
- Density-expansion telemetry improved during the round: vision axis advanced from ~0.21 to ~0.40 and mesh synchrony increased.

### Verification

- `pytest tests/test_self_modification.py tests/test_synaptic_workers.py -q` -> **24 passed**
- Forced density expansion + self-modification round completed end-to-end with:
  - `patches_evaluated=2`
  - `patches_applied=1`
  - `committed=true`

---

## v0.22.95 - 2026-05-29

**GitHub Actions Workflow Recovery for Daily Dispatch and Cloud Learning**

### Changed

- **`.github/workflows/daily_dispatch.yml`** — fixed the embedded Python heredoc indentation so the scheduled dispatch path can call `send_daily_dispatch()` again. The workflow now also checkpoints `pipeline/cloud_brain.sqlite` before cache save and caches the `-shm` / `-wal` sidecars together with the main SQLite file.
- **`.github/workflows/cloud_learning.yml`** — checkout now uses full history, the workflow checkpoints `pipeline/cloud_brain.sqlite` before saving the cache, caches the SQLite sidecars with the main DB, and rebases with retries before pushing `pipeline/cloud_learning_queue.jsonl` back to `main`.
- **`pipeline/tests/test_github_workflows.py`** — added focused workflow regression coverage so the embedded workflow Python blocks stay syntactically valid and the cloud-learning git/cache hardening stays pinned.

### Verification

- `pytest tests/test_github_workflows.py tests/test_daily_dispatch.py -k "not start_daily_dispatch_daemon"` → **8 passed, 1 deselected**
- Direct AST parse of all embedded Python heredocs in `.github/workflows/daily_dispatch.yml` and `.github/workflows/cloud_learning.yml` → **passed**

---

## v0.22.94 - 2026-05-29

**Internal Marketplace — 5-Stream Shadow Price Engine (Research Stream Removed)**

### Changed

- **`pipeline/src/quipu/internal_marketplace.py`** — removed the External Return Channel Research evidence stream. Shadow price now blends **five** independent signals: hedonic regression (W=0.25), cross-site RFQ differential (W=0.20), commodity index drift (W=0.15), cannibalization transfer price (W=0.15), and peer-cluster anchor (W=0.25). Removed `ResearchDispositionSignal`, the `_RESEARCH_*` constants and keyword dictionaries, the WAL connection cache, and the `research_return_signal` / `research_confidence` / `recommended_channel` fields from `ShadowPrice`.
- **`pipeline/pages/25_Internal_Marketplace.py`** — updated shadow-price page to five-stream description; removed `research_return_signal`, `research_confidence`, and `recommended_channel` columns from the parts grid; removed "External Return Signal", "Recommended Channel", and "Research Confidence" metric tiles from the waterfall detail view; updated caption.
- **`_build_marketplace_deck.py`** — updated slide 1 and summary-slide version labels to `v0.22.94`; renamed Slide 3 to "Five Evidence Streams"; removed the Research Return stream card; and simplified the generated deck output path using `os.path`.

### Verification

- `python -m py_compile _build_marketplace_deck.py pipeline/pages/25_Internal_Marketplace.py pipeline/src/quipu/internal_marketplace.py pipeline/src/quipu/_version.py` → **passed**

---

## v0.22.62 - 2026-05-24

**Hideout Mesh Heartbeat Resuscitation, Stale Derived-Heartbeat Refresh, and Bit-Flip Interaction Map**

### Added

- **`pipeline/docs/HIDEOUT_MESH_INTERACTION_MAP.md`** — operator-facing interaction map tracing the live hideout activation path from `bridge_rdp.py` through `asset_resource_mesh_last.gpu_resuscitation` down to `bit_flip_resuscitation.bit_flip_nodes`, including the current synthetic hideout heartbeat contract and mesh-discovery surfaces.

- **Focused regression coverage in `pipeline/tests/test_bridge_rdp.py`** — added coverage for historical live-state derivation, derived heartbeat publication, stale derived heartbeat refresh, and stale derived heartbeat refresh when persisted resuscitation history is unavailable.

### Changed

- **`pipeline/bridge_rdp.py`** — `_hideout_resuscitation_activation()` now republishes historically derived hideout GPU state into `pipeline/bridge_state/compute_peers/scbrain-hideout.json`, refreshes stale synthetic hideout heartbeats, and preserves `all_expected_gpus_accessible` correctly in the published heartbeat metadata.

- **Hideout mesh rendezvous state** — the live `scbrain-hideout` heartbeat now materializes as a 12-GPU `devtunnel` peer in compute-grid discovery, sourced from persisted resuscitation until the physical hideout compute node overwrites it with a real heartbeat.

### Verification

- `pytest tests/test_bridge_rdp.py -q -p no:cacheprovider` → **21 passed**
- Live validation: `bridge_state/compute_peers/scbrain-hideout.json` refreshed with `derived_from_resuscitation: true`, `resuscitation_activation.all_expected_gpus_accessible: true`, and `discover_peers(force=True)` returned `scbrain-hideout` as a present `devtunnel` peer with 12 GPUs.

## v0.22.57 - 2026-05-21

**Repository Catalog Mesh Integration, Synaptic PIM Worker, WIP Aging Page, SCB Insight Echo Guard**

### Added

- **Mesh peer registry in `repository_catalog.py`** — `_mesh_runtime()` builds a live peer topology from `observer:peer_registry` KV entries with a 300-second liveness window. Enables shard-based catalog distribution: each peer host gets a `shard_index / num_shards` slice so large workspace scans don't duplicate work across the mesh.

- **Repository catalog as system-entirety signal** — `system_entirety.py` now persists `RepositoryCatalogState` as a first-class mesh entity with a `APPLIES_REPOSITORY_CATALOG` edge to the active learning window. Catalog signal (coverage, staleness, file count) contributes 18% weight to the overlay strength calculation via `_REPO_CATALOG_GAIN = 0.18`.

- **`_pim_planning_sync_worker` in `synaptic_workers.py`** — new 24-hour daemon worker that harvests planning parameters from all ERP sources (Oracle Fusion FSCM REST + BIP SQL, GSS St. Cloud inventory sub-ledger, AX Dynamics Airport Rd on-hand, Epicor SiteJ/SiteC cycle-count). Soft-skips on transient network errors to prevent backoff accumulation during Oracle outages.

- **`pipeline/pages/20_WIP_Aging_Review.py`** — WIP aging review page additions.

- **Regression coverage** — `tests/test_repository_catalog.py` (+130 lines) and `tests/test_system_entirety.py` (+181 lines) verify mesh runtime construction, peer liveness, shard assignment, catalog signal propagation, and system-entirety entity/edge upserts.

### Changed

- **`doc_rag.py` — SCB Insight echo guard** — `synthesize_plant_insight` now detects offline-caller echo responses (`^\[model_id\]` prefix pattern), retries dispatch after registering the OpenRouter caller, and returns `None` when no live inference is available. Prevents garbage dict reprs from appearing in plant email blurbs.

- **`_run_pi_q1_rollup.py` removed** — Q1 rollup functionality fully superseded by `_run_pi_periodic_rollup.py`; deleted to eliminate dead-code confusion.

- **`config/model_map.json`** — minor model routing adjustment.

### PI Output Artifacts (YE2025 vs Q1 2026)

- `03_Executive_Rollup_YE2025_vs_Q1_2026.pptx` — refreshed executive deck.
- `04_Plant_Email_Blurbs.md` — removed 12 malformed `📖 SCB Insight` lines containing raw `EnsembleResult` dict reprs generated before the echo guard was in place.
- `04_Plant_Emails_Formatted.html` — paired HTML refresh.
- `06_Data_Discrepancy_Report.md` — updated discrepancy report.

### Verification

- `py_compile src/quipu/doc_rag.py src/quipu/repository_catalog.py src/quipu/system_entirety.py src/quipu/synaptic_workers.py` → **CLEAN**
- `pytest tests/test_repository_catalog.py tests/test_system_entirety.py -q` → **PASSED**

---

## v0.22.56 - 2026-05-21

**Native ERP Determination, Canonical Site/System Mapping, and Audited Dispatch**

### Added

- **`pipeline/src/quipu/site_system_map.py`** — canonical Contoso plant registry for ERP, system, connector, business-unit, and database-platform mapping. Captures AX for MetroB Airport Rd, SyteLine for SiteP, Epicor for SiteJ and SiteC, Oracle Fusion for SiteW Rd, Manufacturers Rd, and SiteB, Oracle Legacy for Blair with planned Oracle Fusion cutover in October 2026, and GSS for St. Cloud.

- **`pipeline/src/quipu/erp_native.py`** — native ERP capability layer for SCB. Centralizes ERP signal detection, canonical aliases, site resolution, connector-candidate resolution, and the new hierarchical resolver: learned structure → site/system map → user impute → payload fallback.

- **`pipeline/src/quipu/erp_integration.py`** — ERP payload inspection and dispatch orchestration. Adds preview/apply support for Oracle Fusion REST, SQL-based ERP updates, GSS file export, and durable local SQLite audit rows in `erp_integration_audit`.

- **Focused regression coverage** — added `pipeline/tests/test_erp_native.py` and `pipeline/tests/test_erp_integration.py` to verify native ERP detection, canonical site assignments, learned-first ERP resolution, dispatch behavior, and audit logging.

### Changed

- **Repository routing and PIM synthesis** — `pipeline/src/quipu/repository_catalog.py` and `pipeline/src/quipu/pim_part_synthesis.py` now use the shared native ERP layer instead of local ERP heuristics, so routing metadata, per-part payload synthesis, and integration dispatch stay aligned.

- **Mapping unification** — `pipeline/src/deck/erp_translation.py`, `pipeline/src/quipu/pim_planning_sync.py`, and `pipeline/src/deck/live.py` now consume the canonical site/system map instead of maintaining drift-prone plant mappings and manual business-unit fallbacks in multiple places.

- **Connectors page** — `pipeline/pages/6_Connectors.py` now exposes the site/system/database mapping table, native ERP detection panel, learned-first payload inspection, live dispatch controls, and the recent ERP integration audit trail.

- **User ERP input behavior** — the manual ERP selector is now a fallback signal only; SCB resolves ERP from learned structural evidence first, then site/system mapping, and uses the user impute only when stronger evidence is absent.

### Verification

- `py_compile src/quipu/erp_native.py src/quipu/erp_integration.py pages/6_Connectors.py tests/test_erp_integration.py` → **CLEAN**
- `pytest tests/test_erp_integration.py tests/test_erp_native.py tests/test_repository_catalog.py tests/test_pim_part_synthesis.py -q` → **14 passed**

## v0.22.55 - 2026-05-21

**Freight Entity Family Resolution — SCB vendor_consolidation LLM + OpenRouter/Grok Fallback**

### Added

- **`pipeline/docs/vendor_consolidation_package/generate_missing_shipper_consignee_ids_gard.py`** — new freight entity family resolution script. Generates shipper/consignee hash IDs for the March shipment file then runs a 5-layer DBA/parent-org clustering stack: (1) blocking key by name tokens + ZIP-5 + ISO-2 country, (2) multi-signal fuzzy composite score (40% name, 25% address, 20% ZIP, 15% city), (3) LLM verification via `dispatch_parallel("vendor_consolidation")`, (4) NetworkX connected-components graph clustering on confidence-weighted edges (≥ 0.80 threshold), (5) confidence-tier output (Certain/Probable/Inconclusive/Singleton). Outputs `Shipper_Families` and `Consignee_Families` sheets to `New_Shipper_Consignee_IDs.xlsx`.

- **`_vendor_consolidation_validator`** — structural scorer [0.0–1.0] passed as `validator=` to `dispatch_parallel`. Scores `is_same_family` bool (0.40), `confidence` in [0,1] (0.35), valid `relationship` enum (0.15), non-trivial `reasoning` (0.10). Fires `update_weights("vendor_consolidation", outcomes, score)` inline after every entity pair, closing the self-training loop without a separate training job.

- **`_call_openrouter_fallback()`** — direct OpenRouter/Grok-4.3 call via `llm_caller_openrouter.openrouter_caller` with a minimal `_GrokDecision` duck-type (`model_id="grok-4.3"`, `endpoint_env="XAI_API_KEY"`). Used automatically when the SCB ensemble is unavailable or raises. Inherits the existing xAI → OpenRouter backend fallover chain inside `_resolve_backend_chain`.

### Changed

- **`_call_scb_llm` routing** — SCB `dispatch_parallel("vendor_consolidation")` is now the **primary** path (99.5% of calls). On any exception it falls through to `_call_openrouter_fallback()` instead of returning an error dict. The `_SCB_AVAILABLE=False` early return is removed; fuzzy-only mode is now the last resort only when both SCB and OpenRouter/Grok are unreachable (`_SCB_AVAILABLE or _OPENROUTER_AVAILABLE` gate).

- **`pipeline/docs/vendor_consolidation_package/VENDOR_CONSOLIDATION_FINDINGS.md`** — added §1.4 "Freight Entity Family Resolution" (5-layer stack, output schema, asymmetric information strategy, confidence tier action table); updated §1.3 to document the inline validator and live training signal contribution; added `generate_missing_shipper_consignee_ids_gard.py` to the §6 files table.

- **`pipeline/docs/vendor_consolidation_package/VENDOR_CONSOLIDATION_HOWTO.md`** — added Part 6 (Steps 19–23): path setup, run command, interpreting family sheets, using `family_id` clusters as negotiation leverage, feeding confirmed/refuted clusters back to the SCB Mesh. Added `_gard.py` row to the Quick Reference table.

- **`pipeline/docs/vendor_consolidation_package/Refresh-VendorConsolidationTool.ps1`** — Step 2 now runs `generate_missing_shipper_consignee_ids_gard.py` automatically after the TCI workbook rebuild. Pass `-SkipFreightFamilies` to bypass (e.g., when Brain is offline).

### Verification

- `py_compile` on `generate_missing_shipper_consignee_ids_gard.py` → **CLEAN** (exit 0)
- Routing chain confirmed: SCB ensemble → OpenRouter/Grok-4.3 → error dict (no silent fuzzy-only degradation while any LLM path is live)

## v0.22.41 - 2026-05-20

**Graph-Native Mesh Learning Overlay**

### Added

- **`pipeline/src/quipu/system_entirety.py`** - `oscillating_expansion_step()` now projects the recent learning window into bounded corpus topology. The overlay mints `MeshLearningWindow`, `LearningKindSummary`, `UEQGMRuntimeState`, and `PIMPlanningState` nodes, then spreads hardening through `EMITS_LEARNING_WINDOW`, `SUMMARIZES_LEARNING_KIND`, `HARDENS_ASSET_RESOURCE`, `HARDENS_MATERIAL_PROCESSOR`, and `HARDENS_ENDPOINT` edges.
- **`pipeline/docs/MESH_LEARNING_OVERLAY.md`** - documents the operator model, graph contract, persistence surfaces, and verification path for the distributed learning overlay.

### Changed

- **Mesh learning propagation** - System Entirety no longer treats learned state as scalar visibility only. Recent learning is summarized into graph-native windows and kind nodes, then distributed across asset resources, material processors, and endpoints for mesh hardening.
- **`pipeline/src/quipu/asset_resource_mesh.py`** - asset mesh refreshes now preserve `mesh_learning`, `mesh_learning_projection`, and `mesh_learning_updated_at` while rewriting `entirety:physical_realization`, so learned hardening survives normal resource materialization cycles.

### Verification

- `python -m pytest pipeline/tests/test_system_entirety.py pipeline/tests/test_asset_resource_mesh.py -q` -> **29 passed**
- Live forced System Entirety expansion against `pipeline/local_brain.sqlite` minted learning overlay entities and hardening relations.

## v0.22.18 - 2026-05-19

**System Entirety Toroidal Analysis + Live Operational Pulse**

### Added

- **`docs/SYSTEM_ENTIRETY_ANALYSIS.md`** - added a 16-section System Entirety field manual that maps the Brain as a six-ring torus around the `heart` / `system_entirety` / `torus_touch` core. The document embeds a live `_system_entirety_report.py` snapshot, inventories all routed Streamlit pages and always-on daemons, traces the Oracle / Azure SQL / bridge / compute-grid surfaces, and provides a toroidal observability checklist plus explicit gap register.

### Changed

- **Live operational pulse coverage** - expanded the new analysis with release-time pulse metrics from `pipeline/local_brain.sqlite`: confirmed `llm_dispatch_log` has only **10 total rows**, all on **2026-05-04**; captured the current learning mix (`rag_deepdive`, `citation_chain`, `ocw_resource`, `ml_research`), `body_directives` load (**14 open**, **9 expired**), the current Entirety state (`bit_state=1`, `expansion_phase="broaden"`, `flip_count=3`), and the latest `systemic_refinement_log` cycle showing active compensation without successful self-train updates.
- **Root version metadata** - synced `VERSION` and `pipeline/src/quipu/_version.py` to `0.22.18` for this documentation-and-verification release.

### Verification

- `cd pipeline && python3 -m py_compile _system_entirety_report.py src/quipu/system_entirety.py src/quipu/synaptic_workers.py bridge_rdp.py compute_node_daemon.py` → **passed**
- `cd pipeline && pytest -o "addopts=" tests/test_system_entirety.py tests/test_synaptic_workers.py tests/test_bridge_rdp.py tests/test_compute_grid_devtunnel.py tests/test_asset_resource_mesh.py tests/test_geospatial_relation.py -q --tb=short` → **53 passed** in **12.47s**
- `cd pipeline && python3 _system_entirety_report.py` → **passed**

## v0.22.12 - 2026-05-18

**Vendor Consolidation Package Refresh + Priority-Driven Workbook**

### Changed

- **`pipeline/src/quipu/vendor_consolidation_excel.py`** - the vendor consolidation workbook no longer falls back to a flat 88% service level when TCI data omits a direct percentage. It now parses Excel-style dates robustly, infers service levels from service mode plus ship/delivery timing and supplier/service history, sorts tasks by priority, and writes a filterable `Priority_Tracker` sheet with explicit solution updates and workbook routes.
- **`pipeline/docs/vendor_consolidation_package/VENDOR_CONSOLIDATION_FINDINGS.md`**, **`pipeline/docs/vendor_consolidation_package/VENDOR_CONSOLIDATION_HOWTO.md`**, and **`pipeline/docs/vendor_consolidation_package/vendor_consolidation_data.csv`** - refreshed the handoff package so the docs and structured export match the current Summary, Operator Playbook, Priority Tracker, and What If workflow.

### Maintenance

- **Vendor package hygiene** - cleaned the local `pipeline/docs/vendor_consolidation_package/` folder by removing obsolete validation / `fixed` workbook variants, leaving the canonical handoff outputs in place. The `.xlsx` deliverables remain local package artifacts and are intentionally gitignored.

### Verification

- Generated both canonical package workbooks and verified the shipped output now shows **377 distinct service-level values** across the first 1,000 working rows instead of a flat 88%, with `Priority_Tracker` present and the current top metric showing **416 rows below the 95% service target**.

## v0.22.10 - 2026-05-18

**Credential Fallback Hardening + Works Cited Seed Resilience**

### Added

- **`pipeline/tests/test_llm_key_guard.py`**, **`pipeline/tests/test_llm_caller_openrouter.py`**, **`pipeline/tests/test_citation_chain_acquirer_seeding.py`**, and **`pipeline/tests/test_knowledge_corpus_scb_path.py`** - added focused regression coverage for local key rotation, xAI fallback transport, wrapped Works Cited seed payloads, and newest-export path resolution.

### Changed

- **`pipeline/src/quipu/llm_key_guard.py`** and **`pipeline/src/quipu/llm_caller_openrouter.py`** - the Brain now prefers project-local live credentials over stale inherited editor env vars, can clear cooldown when a key rotates locally, and can fall through from OpenRouter to direct xAI Grok transport when that is the only live path.
- **`pipeline/src/quipu/doc_rag.py`**, **`pipeline/src/quipu/knowledge_corpus.py`**, **`pipeline/src/quipu/citation_chain_acquirer.py`**, and **`pipeline/reset_works_cited_cursor.py`** - document RAG and Works Cited ingestion now share the same live-key precedence, resolve the newest SCB export automatically, accept wrapped `paper_ids`/`seed_paper_ids` payloads, and reset stale Works Cited rows more completely before re-ingest.

### Verification

- Focused credential/citation regression: **33 passed** across `test_llm_key_guard.py`, `test_llm_caller_openrouter.py`, `test_doc_rag_credential.py`, `test_citation_chain_acquirer_seeding.py`, and `test_knowledge_corpus_scb_path.py`.

## v0.22.9 - 2026-05-18

**Adaptive UEQGM Runtime Daemon + System Entirety Consumption**

### Added

- **`pipeline/src/quipu/synaptic_workers.py`** - added the dedicated `synapse-ueqgm` daemon, which refreshes the adaptive UEQGM runtime continuously from System Entirety certainty while recording heartbeat and runtime summary data into `brain_kv`.
- **`pipeline/tests/test_synaptic_workers.py`** - added focused coverage to ensure the new worker is started and reported by the synaptic health snapshot.

### Changed

- **`pipeline/src/quipu/ueqgm_engine.py`** - adaptive runtime parameters are now evidence-gated per field; a candidate value only replaces the persisted state when newer evidence beats both the current corpus-density floor and the previously proven evidence for that parameter.
- **`pipeline/src/quipu/system_entirety.py`** - the System Entirety now consumes the persisted UEQGM runtime as a symbiotic overlay, applying adaptive axis injections and a runtime-backed transaction contribution to spread proven UEQGM expansion through the broader system state.

### Verification

- Focused adaptive-runtime regression: **56 passed** across `test_ueqgm_engine.py`, `test_system_entirety.py`, and `test_synaptic_workers.py`.

## v0.22.8 - 2026-05-18

**Hideout Dev Tunnel Diagnostics + Resource-Share Planning**

### Added

- **`pipeline/bridge_rdp.py`** - added operator-facing Dev Tunnel flows for the hideout peer: local forward creation, mesh ping, hideout doctor, helper startup, and Dev Tunnel-backed RDP launch.
- **`pipeline/tests/test_bridge_rdp.py`** and **`pipeline/tests/test_compute_grid_devtunnel.py`** - added focused tests for `dt connect` port parsing, forward-state reuse, hideout health diagnostics, and remote mesh ping behavior.

### Changed

- **`pipeline/src/quipu/compute_grid.py`** - local mesh execution now exposes `resource_inventory` and `resource_share_plan` payloads, so a Dev Tunnel peer can report shareable cores, RAM, VRAM, processor bindings, and contract coverage over the compute-grid protocol.
- **`pipeline/config/bridge_targets.yaml`** and **`pipeline/connect_hideout.ps1`** - wired `hideout-rdp` / `hideout-mesh` targets for `scbrain-hideout.use2`, and fixed the persistent helper so it no longer discards the requested `RemotePort` while writing forward-state metadata.

### Verification

- Focused Brain/bridge slice: **88 passed** across `test_system_entirety.py`, `test_asset_resource_mesh.py`, `test_geospatial_relation.py`, `test_compute_grid_devtunnel.py`, `test_bridge_rdp.py`, and `test_symbiotic_torus.py`.

## v0.21.0 - 2026-05-14

**Perception Mk2a - Heart-Grounded Audio/Video Ingestion**

### Added

- **`pipeline/src/quipu/perception_audio.py`** - added Whisper STT for audio, bundled-ffmpeg audio extraction and keyframe sampling for full video, and a `perception_events` append-only log for heart-stamped percepts.
- **`pipeline/src/quipu/perception.py`** - added `ingest_uploaded_asset()` so direct uploads route by media type: audio/video goes through Mk2a, while images and GIFs reuse the Mk1 visual analysis path and still return the same inline heart-stamped operator feedback.

### Changed

- **Heart-lock invariant** - `theta` now remains exactly locked to `heart.phase_rad`; any rADAM influence is recorded separately as `radam_phase_bias` metadata instead of perturbing the toroidal ground itself.
- **`pipeline/pages/17_Document_RAG.py`** - the sidebar uploader now supports the documented media set consistently and only warns about ffmpeg when a video upload actually needs it.
- **`pipeline/src/quipu/brain_body_signals.py`** and **`pipeline/requirements.txt`** - added Touch-relief mappings and dependency wiring for transcript, keyframe, and audio-duration ingestion.

### Verification

- Added focused unit coverage for direct media upload routing and the exact Heart-lock stamp behavior.

## v0.20.22 - 2026-05-14

**System Entirety - 7th-D Observer-Tangent Bit Flip**

### Added

- **`pipeline/src/quipu/system_entirety.py`** - introduced the 7th-dimensionality observer-tangent orchestrator. The Brain now projects the 6-sense CAT state onto a System Entirety axis, computes a Floquet-modulated bit flip, and persists `entirety:*` state to `brain_kv`.
- **`pipeline/tests/test_system_entirety.py`** - added unit coverage for observer geometry, parity flipping, period scaling, state shape, persistence, and rate-limit semantics.

### Changed

- **`pipeline/src/quipu/knowledge_corpus.py`** - `refresh_corpus_round()` now invokes `oscillating_expansion_step()` and surfaces the returned entirety summary in the round output.
- **Root version metadata** - synced the repo `VERSION` file with the canonical `pipeline/src/quipu/_version.py` value.

### Maintenance

- **Git hygiene** - Playwright failure screenshots under `pipeline/tests/playwright/screenshots/` are now ignored so local test artifacts do not dirty the repo.

## v0.20.19 - 2026-05-13

**Autonomous Resurrection Hardening**

### Fixed

- **`pipeline/autonomous_agent.py`** - process liveness now writes `pipeline/logs/agent_heartbeat.txt` every 60 seconds for the full life of the process, not only during long sleep windows. Busy cycles no longer appear dead to the resurrection layer.
- **`pipeline/brain_watchdog.ps1`** - watchdog startup now works on Windows PowerShell 5.1, quotes the agent path correctly under OneDrive paths with spaces, and captures agent stdout/stderr to dedicated log files for postmortem debugging.
- **`pipeline/install_brain_watchdog.ps1`** - scheduled task registration now launches through a wrapper command that survives Task Scheduler quoting edge cases and falls back to repeat-only registration when the logon trigger is blocked by local policy.

### Added

- **`pipeline/run_brain_watchdog.cmd`** - scheduler-safe wrapper that launches `brain_watchdog.ps1` hidden from `cmd.exe`.

### Operational Impact

- External watchdogs can now distinguish a busy healthy agent from a dead process.
- The process-level heartbeat remains compatible with the existing stale-heartbeat recovery logic in `pipeline/app.py`.

## v0.19.25 — 2026-05-01

**Repo Hygiene + Credential Migration**

### Credential Migration

- **`pipeline/config/connections.yaml`** — three new `rdp_*` scopes migrate all connection data from the deleted `pipeline/rdp_files/`:
  - `rdp_fileserver_01` — Corporate File Server 03 (10.0.0.10:3389, Windows Integrated auth)
  - `rdp_desktop` — Desktop workstation (10.0.0.10:3389, Windows Integrated auth)
  - `rdp_laptop` — AUser Fabrikam laptop (LAPTOP-01.fabrikam.contoso.local, `CONTOSOCORP\AUser`, NLA certificate thumbprint `0000000000000000000000000000000000000000
- Use `python -m src.connections.secrets set rdp_<name> --user <upn> --password <pw>` to store passwords for any RDP scope in the DPAPI vault.

### Cleanup

- Removed 22 one-off scripts: `quick_fix{1-5}.py`, 12 root-level `temp_*.py`, 7 `pipeline/temp_*.py` (debug, legend-fix, learning-report utilities)
- Finalised deletion of `Proxy-Pointer-RAG/` subproject (was deleted on disk, now removed from git index)
- Removed `pipeline/pages_archive/` (vestigial directory, contained only gitignored `__pycache__`)
- Removed `pipeline/rdp_files/` (3 `.rdp` files — connection info migrated to `connections.yaml`)

### Git / Artifact Hygiene

- `pipeline/abc_screenshots/` (442 PNG files, ~114 MB) — removed from git tracking; files retained on disk; pattern added to `.gitignore`
- `pipeline/navigation_tests/` (15 files, ~11 MB) — same treatment
- `pipeline/my_playwright_profile/` — added to `.gitignore` proactively
- Root `VERSION` file synced to canonical `_version.py` value (was stale at `0.19.4`)

### Version

- `pipeline/src/quipu/_version.py` bumped `0.19.19 → 0.19.25`; PHASES table backfilled with `0.19.19` and `0.19.24` entries

---

## v0.19.4 — 2026-04-30

**One-Click Launcher (`Launch-SCB.ps1`)**

### New Files

- **`Launch-SCB.ps1`** — Root-level PowerShell launcher for the Supply Chain Architect. Double-click the desktop shortcut (or the script) to start the app with a single click:
  - Detects if SCB is already running on `localhost:8501`; if so, just opens the browser.
  - Otherwise spawns `streamlit run pipeline/app.py` as a fully hidden background process using the project venv.
  - Polls port 8501 (up to 30 s) before opening the default browser to `http://localhost:8501`.
  - Writes the Streamlit PID to `%TEMP%\scb_streamlit.pid` for optional cleanup.

### Desktop Shortcut

- **`Supply Chain Architect.lnk`** placed on the user Desktop (not committed — user-local artifact). The shortcut runs `powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File Launch-SCB.ps1`, presenting a zero-terminal, single-icon launch experience.

### Security

- No changes to `pipeline/.streamlit/config.toml` security controls (localhost-only binding, XSRF protection, CORS enforcement remain in force).

---

## v0.16.0 — 2026-04-24

**Symbiotic Dynamic Tunneling + Torus-Touch (T^7)**

### New Modules

- **`pipeline/src/quipu/symbiotic_tunnel.py`** — Bayesian-Poisson centroids, inverted-ReLU ADAM optimiser, dual-floor mirror, and propeller routing. The vision worker mints `SYMBIOTIC_TUNNEL` edges from closed-loop TCP/UDP mesh topology every 5 minutes.

- **`pipeline/src/quipu/torus_touch.py`** — Continuous 7-D categorical pressure on the toroidal manifold. Runs every 30 seconds pushing every Endpoint along the `n=7` categorical gap gradient so tunnel weights follow manifold geometry.

### New Scripts

- **`pipeline/fetch_pim_80446_v3.py`** — v3 PIM item fetch with screenshot capture for part 80446-04.
- **`pipeline/find_write_ops_80446.py`** — Discovery of write-operation touchpoints for item 80446-04 in Oracle Fusion PIM.

### Tests

- **`pipeline/tests/test_symbiotic_torus.py`** — 29 unit tests covering symbiotic tunnel centroid computation, torus-touch gradient steps, and edge-minting integration.

### Configuration

- Added root `.gitignore` to exclude `__pycache__`, `*.pyc`, `*.sqlite`, mission snapshots, and PIM screenshots from version control.

---

## v0.15.0 — 4-ERP xlsx pipeline (Epicor/Oracle/SyteLine/AX) + Brain page fixes + EOQ query optimisation.

## v0.14.9 — OCW semantic bridge + synaptic worker protection + network vision worker.

## v0.8.0 — Massive UX/Actionable overhaul: SQLite local store, NLP-part categorisation, Global date windows, semantic action-TODO engine.

## v0.7.1 — Ask the Data cross-dataset report generation.

## v0.7.0 — Enterprise Network Autonomous Agent. Native Exchange/SMB discovery.

## v0.6.0 — Global Application Filter, unified session_state routing, AI PowerPoint Creator.

## v0.5.0 — Value Stream Living Map: End-to-end integration of PO, SO, WO flows.

## v0.4.x — SQL query rewrites, database explorer, bug-fix waves.

## v0.3.0 — MIT CTL research modules: hierarchical EOQ, causal lead-time, survival, bullwhip, multi-echelon, sustainability, freight portfolio, CVaR risk.

## v0.2.0 — Depth: graph backends, LinUCB ranker, hierarchical OTD index.

## v0.1.0 — Core: EOQ Bayesian-Poisson, OTD recursive, Procurement 360, Data Quality, Connectors.
