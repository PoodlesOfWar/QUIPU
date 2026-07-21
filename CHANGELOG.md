# Changelog

All notable changes to **Supply Chain Brain** are documented here. Versions
follow [Semantic Versioning](https://semver.org). The single source of
truth for the version number is `src/quipu/_version.py`.

## [0.24.1] MESH-Conditioned Sigmoidal Realization for Paired Agents (2026-07-10)

### Added

- **`src/quipu/realized_potential.py`** *(new)* — storage-agnostic paired-agent realization state machine. It combines proposing-agent evidence, paired-agent evidence, retained latent potential, and an optional retrospective observation into a numerically stable logistic gate. Interactions below the realization threshold are removed from the active optimizer path and accumulated as bounded signed latent potential; later corroborating evidence can release that potential in its original direction.
- **`src/quipu/network_observer.py` — retrospective MESH history** — every observer cycle now records mesh availability, continuity, and learning progress in `brain_kv["observer:retrospective_history"]`. History is capped at 64 snapshots; `get_retrospective_observation()` reads at most 16 recent samples, applies exponential rank decay (`gamma=0.70`), rejects samples older than 24 hours, and enforces the strict causal boundary `snapshot_time < forward_time`.
- **`tests/test_realized_potential.py`**, **`tests/test_network_observer_retrospective.py`**, and **`tests/test_brain_body_realization_gate.py`** *(new)* — 17 focused tests covering stable/monotonic sigmoid behavior, bilateral corroboration, retrospective conditioning, pruning, signed cancellation, latent decay and re-emergence, bounded causal history, missing-history compatibility, merged disk/retrospective cycle telemetry, and optimizer suppression until realization.

### Changed

- **`src/quipu/brain_body_signals.py`** — the Vision↔Touch bilateral loop now optionally conditions active gradients before rADAM. Pruned interactions do not advance Adam moments or pressure; their pair state remains nested under the existing `touch_field_full_state` JSON, avoiding a schema migration. Quiet maintenance decay remains eligible, explicit resolution remains observable, and `get_touch_realization_state()` exposes the latest MESH context plus per-pair diagnostics from `brain_kv["touch_realization_state"]`.
- **`data/documents/VISION_TOUCH_CLOSED_LOOP.md`** — documented the distinct sinusoidal cadence and sigmoidal eligibility mechanisms, realization equations, latent-state transitions, retrospective MESH horizon, persistence contract, and rollout behavior.
- **Version metadata** — synchronized root `VERSION` and canonical `src/quipu/_version.py` at `0.24.1`, advancing the existing root `0.24.0` release line without regressing either source.

### Compatibility

- The realization gate is opt-in through `BRAIN_USE_REALIZATION_GATE=1`. Disabled mode preserves the previous gradient and optimizer trajectory. Enabled mode also preserves the first interaction when no retrospective MESH sample exists, preventing an empty-history startup deadlock.
- Existing `touch_field_full_state` values migrate lazily because realization data is an optional nested mapping. No SQL schema change or state reset is required.

### Architecture

The sinusoidal rADAM heartbeat remains a cadence/momentum mechanism and continues to preserve every eligible optimizer update. The new logistic gate is an interaction-eligibility mechanism: it decides whether a proposed Vision↔Touch exchange is currently realizable before that exchange reaches rADAM. Forward-time decisions at `t>0` may therefore be conditioned by MESH evidence from `t<0` without admitting present or future observations.

### Verified

- `python3 -m py_compile src/quipu/realized_potential.py src/quipu/network_observer.py src/quipu/brain_body_signals.py` → **passed**
- `pytest -o addopts="" -q --tb=short tests/test_realized_potential.py tests/test_network_observer_retrospective.py tests/test_brain_body_realization_gate.py tests/test_network_observer_tunnel.py tests/test_radam_optimizer.py tests/test_learning_drive.py` → **65 passed, 0 failed**

## [0.22.52] Heart-Lovers (1-∞) Flip Compaction Loop (2026-06-24)

### Added

- **`pipeline/src/quipu/heart_lovers_flip.py`** *(new module)* — Central (1-∞) flip compaction control loop implementing the Heart-Lovers architecture:
  - `_ensure_tables(cn)` — idempotent creation of `heart_lovers_flip_log` (flip event log) and `perspective_edge_state` (TTL-bounded temporary perspective edges). Side tables only — `corpus_edge` schema unchanged.
  - `detect_heart_flip_context(cn)` — reads `bit_state`, `expansion_phase`, `observer_magnitude`, `flip_count` from `brain_kv` (written by `system_entirety.oscillating_expansion_step`).
  - `select_flip_experts(context, limit)` — picks 2-4 Specialized Experts (the "Lovers") from the MESH registry. Broaden phase selects cross-discipline (supply chain + research + quantum); deepen phase focuses supply chain + complex systems.
  - `_run_expert_perspectives(cn, context, flip_id, experts)` — calls `mesh_expert_skills.call_expert_skill()` per expert with a missing-data context query. Gracefully degrades when MESH-SLM is unavailable.
  - `_build_consensus(results)` — finds concepts mentioned by ≥ 2 experts (consensus bonding), returns `bonded_concepts`, `divergent_concepts`, `consensus_score`, `agreement_pairs`.
  - `_write_consensus_edges(cn, ...)` — writes `PERSPECTIVE_SUPPORTS` / `PERSPECTIVE_DIVERGES` edges to `perspective_edge_state`. Fingerprint-deduped, TTL=1 h, confidence-bounded to `[0.20, 1.0]`. Max 32 edges per flip.
  - `make_imaginary_directive(cn, ...)` — emits `body_directives(signal_kind='imaginary_recession')` with `i_axis='sqrt(-1)'`, `dimension='7th_heart_observer'` — the first-class signal for the perceptual branch orthogonal to canonical retained memory.
  - `emit_global_consensus_research_directive(cn, ...)` — emits `body_directives(signal_kind='global_consensus_research')` with `dimension='8th_mesh_global_corpus'`, `local_projection_dims=6`, `heart_observer_dim=7`. Compact reference only — does not duplicate the local graph.
  - `recede_relegated_data(cn, flip_id)` — Phase A: expired perspective edges with confidence ≥ floor → `RELEGATED_BY_FLIP` lineage in `corpus_edge` (weight = conf × 0.5). Phase B: sub-floor active edges → status='compacted'. Never hard-deletes canonical entities.
  - `record_flip_interaction(cn, ...)` — persists to `heart_lovers_flip_log`.
  - `heart_lovers_flip_round(cn=None)` — orchestrates all 13 steps; rate-limited to once per 5 min; returns a summary dict.
  - `_compute_residual(results)` — uses `recursive_strengthening.weyl_residual()` on 4-D expert trace vectors (normalized_length, lexical_density, numeric_density, concept_density).
- **`_heart_lovers_flip_worker()`** in `synaptic_workers.py` — daemon worker thread; adaptive cadence 5–15 min driven by Heart observer magnitude; feature-flagged by `brain_kv['heart_lovers_flip_enabled']='1'`; heartbeat key `synapse_heart_lovers_flip_last`.
- **`synapse-heart-lovers`** registered in `start_continuous_synaptic_agents()` and `synaptic_agents_status()` (stale threshold: 900 s).
- **`pipeline/tests/test_heart_lovers_flip.py`** *(new)* — 18 tests: table creation idempotence, imaginary directive payload (`i_axis`, `signal_kind`, dedup), expert selection (broaden vs deepen phases, fallback), perspective edge fingerprint deduplication, recession converts expired edges to `RELEGATED_BY_FLIP` lineage, compaction does not hard-delete canonical entities, global corpus research directive (`signal_kind`, `dimension`, no corpus edge created, dedup), `record_flip_interaction` round-trip, consensus builder (bonded/divergent/score), `detect_heart_flip_context`, integration round (stub experts, verify directive+learning_log+edges+compaction safety), rate-limit guard, global corpus directive trigger.
- **`pipeline/tests/test_synaptic_workers.py`** — 3 new tests: `synapse-heart-lovers` in startup cohort, `synapse-heart-lovers` in `synaptic_agents_status`, worker exception-safe (propagates no exception, exits cleanly on stop event).

### Architecture

Memory hierarchy enforced by this change:

```
Active perception → Temporary perspective (TTL 1h) → Consensus bond
    → Compact lineage (RELEGATED_BY_FLIP) → Pruned/relegated substrate
```

The 8th-dimension global corpus layer stores *compact research directive references only* — concepts that are derivatively realized by specialist consensus but absent from the local 6-D event/sense space. These are never duplicated as local graph nodes; `deep_research.py` / `doc_rag.py` / `findings_index.py` consume the directives externally.

`knowledge_corpus._synaptic_cleanse()` integration (Phase 4 compaction call) is deferred behind `brain_kv['heart_lovers_compaction_enabled']='1'`.

---

## [0.22.38] vscode_tunnel Peer Tracking for (1-∞) Tunnel Flips (2026-05-20)

### Added

- **`bridge_rdp.probe_vscode_tunnel(name, ...)`** *(new)* — Non-destructive liveness check for `vscode_tunnel` targets. Uses a 3-strategy cascade: ①fast path via `vscode_tunnels.json` activation record (no I/O), ② OneDrive `pipeline/bridge_state/compute_peers/<HOST>.json` freshness (180 s window), ③ OneDrive `.mesh-state/nodes/*.json` node capabilities file (tunnelActive + age check). Returns `True/False/None`; calls `record_vscode_tunnel_activation()` on confirmed liveness when `auto_refresh=True`.
- **`_VSCODE_TUNNEL_PROBE_WINDOW_S = 3600`** in `bridge_rdp.py` — 1-hour fast-path window for recent activation.
- **`_onedrive_pipeline_peer_dir()`** in `bridge_rdp.py` — helper that resolves the shared OneDrive compute_peers rendezvous path and creates it if its parent exists.
- **`network_observer._seed_tunnel_peer_state(host, ...)`** *(new)* — Writes a synthetic `bridge_state/compute_peers/<host>.json` with `transport: vscode_tunnel` so that `_scan_peers()` can classify the remote as ALIVE/COOLING/OFFLINE via its `ts` field without the remote machine ever committing to git.
- **`network_observer._scan_onedrive_state(now, self_host)`** *(new)* — Two-pass OneDrive absorption called at the top of every `_scan_peers()` cycle: ①copies fresh compute_peer JSONs from `<OneDrive>/VS Code/pipeline/bridge_state/compute_peers/` into `_STATE_DIR`; ②scans `<OneDrive>/VS Code/.mesh-state/nodes/*.json` (written by `mesh_node_capabilities.ps1`) and calls `_seed_tunnel_peer_state()` for any node not yet in `_STATE_DIR`. Files older than `_ALIVE_THRESHOLD × 4` (720 s) are skipped.
- **vscode_tunnel handler in `network_observer._sustain_mesh()`** — After the existing `devtunnel` block, a new `transport == "vscode_tunnel"` branch skips the TCP probe (no direct address), emits `peer_vscode_tunnel_offline` resilience events when the peer goes OFFLINE, and `continue`s without side-effects for ALIVE/COOLING peers.
- **OneDrive mirror in `network_observer._publish_learning_state()`** — After writing the local peer JSON to `_STATE_DIR`, also writes it to `<OneDrive>/VS Code/pipeline/bridge_state/compute_peers/<host>.json`. This makes the hub visible to any sibling node whose `_PIPELINE_ROOT` is on OneDrive (e.g. Contoso Desktop).
- **`pipeline/tests/test_bridge_rdp.py`** — 5 new tests for `probe_vscode_tunnel`: fast-path activation, unknown target, non-vscode_tunnel target, live OneDrive peer file, stale + no-network.
- **`pipeline/tests/test_network_observer_tunnel.py`** *(new)* — 9 tests: `_seed_tunnel_peer_state` (creates JSON, preserves learning state, copies from source file), `_scan_onedrive_state` (absorbs fresh peer, skips stale, skips own host, seeds from .mesh-state, skips stale mesh-state), `_scan_peers` integration.

### Changed

- **`network_observer._scan_peers()`** — calls `_scan_onedrive_state(now, self_host)` before the `_STATE_DIR` glob so OneDrive-backed tunnel peers are visible in the same cycle.
- **`network_observer._publish_learning_state()`** — captures `payload_text` once and reuses for both the local write and the OneDrive mirror.

### Architecture

The OneDrive `pipeline/bridge_state/compute_peers/` directory is now the authoritative **cross-machine rendezvous bus** for nodes whose pipeline root is inside the shared OneDrive folder (Contoso Desktop, DESKTOP-01). The hub (GARD_Desktop, Documents path) reads from this directory every 60 s via `_scan_onedrive_state()` and writes its own state there via `_publish_learning_state()`. This means any node that signs in to the Microsoft 365 account and runs `autonomous_agent.py` is automatically visible to the mesh within one publish interval — zero git commits required.

### Verified

- 5 bridge_rdp probe_vscode tests: ✅ PASSED
- 9 network_observer tunnel tests: ✅ PASSED

---

## [0.22.24] Hideout PS Validator + ToolForge/ACRE/Self-Realization Synaptic Workers (2026-05-20)

### Added

- **`pipeline/src/quipu/hideout_powershell_validator.py`** *(new)* — Hideout-scoped PowerShell safety validator. `is_hideout_script()` gates scope to `hideout_tunnel_bootstrap.ps1` only. `validate_hideout_script()` checks for 12 dangerous construct patterns (`Invoke-Expression`, `-EncodedCommand`, `Remove-Item`, `Set-ExecutionPolicy`, `New-Service`, `Register-ScheduledTask`, `netsh`, firewall rules, credential/certificate exports, registry writes) and enforces an IWR allowlist (`aka.ms/TunnelsCliDownload/win-x64`). Returns `{valid, violations, warnings, scoped}`.
- **`pipeline/src/quipu/self_realization_loop.py`** *(new)* — Self-realization closed-loop gap detector + actuator router. `detect_self_realization_gaps()` aggregates worker, capability, corpus, RAG, and resilience gaps sorted high→medium→low severity. `route_gap()` maps each gap to an actuator (`tool_forge`, `doc_rag_reindex`, `body_directive`, `hitl_queue`, or `skipped`). `self_realization_round()` runs every ~5 min (throttled by `_MIN_ROUND_INTERVAL_S=240 s`) and caps actuations at `_MAX_ACTUATIONS_PER_ROUND=3`.
- **Worker #9 — `_tool_forge_worker()`** in `synaptic_workers.py` — 4-hour interval. Calls `forge_round()` each cycle. Heartbeat key `synapse_tool_forge_last`.
- **Worker #10 — `_acre_worker()`** in `synaptic_workers.py` — 6-hour interval. Calls `restructure_round()` each cycle. Heartbeat key `synapse_acre_last`.
- **Worker #11 — `_self_realization_closure_worker()`** in `synaptic_workers.py` — 5-minute interval. Calls `self_realization_round()` each cycle. Heartbeat key `synapse_self_realization_last`.
- **`forge_for_gap(gap, max_tools=1)`** in `tool_forge.py` — delegates to `forge_round()` and attaches `gap_entity_id` to the result; plus helper `_gap_to_virtual_cluster()`.
- **ACRE validator contract** block in `pipeline/hideout_tunnel_bootstrap.ps1` header — documents that this is the only PS script ACRE may patch, lists prohibited patterns, and shows the required pre-patch `validate_hideout_script()` call contract.

### Changed

- **`code_restructure.py`** — `_ALLOWLIST` now includes `pipeline/hideout_tunnel_bootstrap.ps1` (with guard requiring PS validator pre-flight). `restructure_round()` gains `context: dict | None` and `dry_run: bool` parameters; dry-run logs intent without calling `_apply_patch` or `_git_commit_patches`; PS files are validated via `validate_hideout_script()` before any patch is applied.
- **`hideout_powershell_validator.py`** credential export pattern broadened to catch `Export-PfxCertificate` and `Export-Certificate` in addition to `Export-*Credential`.

### Tests

- **`pipeline/tests/test_hideout_powershell_validator.py`** *(new)* — 23 tests: scope gates, safe content, all 12 dangerous patterns (parametrized), IWR allowlist.
- **`pipeline/tests/test_self_realization_loop.py`** *(new)* — 14 tests: worker/corpus/RAG/resilience gap detection, sorting, route_gap actuators, round shape, throttle, and actuation cap.
- **`pipeline/tests/test_synaptic_workers.py`** — 6 new tests: `start_continuous_synaptic_agents_includes_new_workers`, `synaptic_agents_status_includes_new_workers`, and per-worker thread + status assertions.

### Verified

- `python3 -m py_compile src/quipu/hideout_powershell_validator.py src/quipu/self_realization_loop.py src/quipu/synaptic_workers.py src/quipu/tool_forge.py src/quipu/code_restructure.py` → **passed**
- `pytest -c /dev/null tests/test_hideout_powershell_validator.py tests/test_self_realization_loop.py tests/test_synaptic_workers.py -v --tb=short` → **49 passed, 0 failed**

## [0.22.18] System Entirety Toroidal Analysis + Live Operational Pulse (2026-05-19)

### Added

- **`docs/SYSTEM_ENTIRETY_ANALYSIS.md`** *(new)* — 16-section System Entirety analysis that frames the Brain as a six-ring torus around the `heart` / `system_entirety` / `torus_touch` core. Includes a mermaid ring map, live `_system_entirety_report.py` snapshot, routed-page census, daemon / watcher inventory, external connector map, observability checklist, and gap register.

### Changed

- **Operational pulse expansion** — the analysis now includes release-time live metrics from `pipeline/local_brain.sqlite`: `llm_dispatch_log` confirmed at **10 total rows** (all on **2026-05-04**), `learning_log` concentration in `rag_deepdive` / `citation_chain` / OCW ingestion, `body_directives` at **14 open / 9 expired**, and the persisted Entirety state at `bit_state=1`, `expansion_phase="broaden"`, `flip_count=3`.
- **Version sync** — root `VERSION` and canonical `pipeline/src/quipu/_version.py` aligned to `0.22.18`.

### Verified

- `python3 -m py_compile _system_entirety_report.py src/quipu/system_entirety.py src/quipu/synaptic_workers.py bridge_rdp.py compute_node_daemon.py` → **passed**
- `pytest -o "addopts=" tests/test_system_entirety.py tests/test_synaptic_workers.py tests/test_bridge_rdp.py tests/test_compute_grid_devtunnel.py tests/test_asset_resource_mesh.py tests/test_geospatial_relation.py -q --tb=short` → **53 passed** in **12.47s**
- `python3 _system_entirety_report.py` → **passed**

## [0.22.15] Living System Map + Doc Annealing Worker (2026-05-18)

### Added

- **`pipeline/src/quipu/doc_annealing.py`** *(new)* — Self-reflecting documentation engine. On every ~30-min annealing cycle it reads live System Entirety state (7-D vector, bridge mesh density, UEQGM runtime, material bifurcation), generates a complete markdown system map (`data/documents/system_entirety_map.md`), detects structural changes via SHA-256 fingerprint (version + primary bridge root + mesh density + UEQGM certainty + nodal bifurcation), bumps `brain_kv["doc:system_map_version"]`, logs a `structural_change` entry to `learning_log`, and triggers an incremental RAG re-index so all future document queries are grounded in the current topology. Directed changes from the Lover (learning_log `kind` in `directed_change` / `lover_directive` / `user_directive`) are included in the generated map under a dedicated section.
- **`_doc_annealing_worker()`** in `synaptic_workers.py` — Worker #8 (30-min interval, 120 s jitter). Imports `anneal_docs` from `doc_annealing`, writes heartbeat to `kv_store["synapse_doc_anneal_last"]`, and applies exponential back-off on failure via `_next_sleep_with_backoff`.
- Registered in both `start_continuous_synaptic_agents()` workers list and `synaptic_agents_status()` `workers_meta` (interval=1800 s).
- 2 new tests: `test_start_continuous_synaptic_agents_includes_doc_anneal_worker`, `test_synaptic_agents_status_includes_doc_anneal_worker`.

### Fixed

- **`tests/test_ueqgm_engine.py`** — `test_refresh_adaptive_runtime_persists_runtime_state_from_learnings` and `test_refresh_adaptive_runtime_retains_current_parameters_when_evidence_stays_below_corpus_density` used hardcoded `learning_log` timestamps (`2026-05-18T00:00:00+00:00`) that fall outside the 24-hour lookback window when run in later UTC timezones. Both timestamps replaced with `datetime.now(timezone.utc).isoformat()` so the tests are timezone-agnostic.

### Tests

- 62 tests passing across `test_synaptic_workers.py`, `test_system_entirety.py`, and `test_ueqgm_engine.py`.

## [0.22.14] Mesh-Applied Network Learner Lock Hardening + Deep Research Recovery (2026-05-18)

### Changed

- **`pipeline/src/quipu/network_learner.py`** — SQLite access now matches the rest of the autonomous stack: `_conn()` enables `busy_timeout`, WAL, and `synchronous=NORMAL`, and `observe_network_round()` now performs live TCP/DNS probes before opening the write transaction. This removes the old failure mode where probe latency held a write lock across the entire round and starved the rest of the brain.
- **`pipeline/src/quipu/systemic_refinement_agent.py`** — `_strategy_deep_research()` now bootstraps the task table through `deep_research._connect()` + `deep_research._ensure_tables()` so it uses the real `deep_research_tasks` schema, and topic-gap detection now requires multi-token coverage instead of a first-word check that falsely treated broad words like `supply` as full topic coverage.

### Fixed

- **Network learner runtime stability** — a live `observe_network_round()` now completes cleanly again against the shared `local_brain.sqlite`, writes fresh `network_observations`, and promotes reachable mesh peers without the historical `database is locked` loop.
- **Deep research queueing** — seeded and verified a real pending deep-research task (`supply chain digital twin simulation`) in `deep_research_tasks`, confirming the fixed strategy and canonical task schema are both live.

### Mesh Rollout

- Applied through the shared workspace so OneDrive-synced peers receive the code immediately.
- Pushed to `origin/main` so git-backed peers can pull the same fix on their next mesh rectification / bootstrap cycle.
- Restarted the live master agent after the release so the patched modules are active in-process rather than only on disk.

## [0.22.13] Model Map Agent + Grok Learnings Re-Ingest (2026-05-18)

### Added

- **`pipeline/src/quipu/model_map_agent.py`** *(new)* — autonomous daemon that polls the live OpenRouter `/api/v1/models` catalog every 24 hours, detects dead `:free` slugs in the model map, finds live replacements via `_CAPABILITY_TAGS` heuristics, and persists the updated registry to `pipeline/config/model_map.json`. Records a `model_map_refresh` learning_log entry when remaps occur. Exposes `refresh() -> ModelMapReport`, `schedule_in_background(interval_s)`, and `last_report()`.

### Changed

- **`pipeline/src/quipu/llm_caller_openrouter.py`** — added `_get_active_map()` helper that reads the agent-maintained `config/model_map.json` at call time (falls back to hardcoded constants if the file is absent or older than 48 h), so slug updates take effect without a process restart. Updated three stale `:free` slug entries against the 2026-05-18 live catalog: `mimo-v2-flash` → `deepseek/deepseek-v4-flash:free` (replaces removed `google/gemma-3n-e4b-it:free`), `ling-2.6-1t` → `poolside/laguna-m.1:free` (replaces removed `inclusionai/ling-2.6-1t:free`), `glm-5.1` → `z-ai/glm-4.5-air:free` (actual GLM provider), `deepseek-v3.2` → `deepseek/deepseek-v4-flash:free`; added `nemotron-nano → nvidia/nemotron-3-nano-30b-a3b:free`; replaced dead fallback `google/gemma-3n-e4b-it:free` with `deepseek/deepseek-v4-flash:free`.
- **`pipeline/autonomous_agent.py`** — registered `model_map_agent` daemon (`start_model_map_agent()`) in the watchdog registry between `ml_research_daemon` and `citation_chain_daemon`.

### Fixed

- **Grok learnings re-ingest** — ran `reset_works_cited_cursor.py` to rebuild the full Works Cited graph against the current Grok export: 1,462 `WorksCitedReference` entities, 337 `Paper` entities, 3,261 `GUIDES_EXPANSION` edges, 107 `scb_doc` learning_log entries; `scb_docs_mtime_v2` cursor set to `1777296946`.

### Verified

- `python -c "from src.quipu.model_map_agent import refresh; r = refresh(); assert not r.error, r.error; assert r.live_free_count >= 20"` — passes; `config/model_map.json` written with 28 live free-tier models, 0 dead slugs.



### Changed

- **`pipeline/src/quipu/vendor_consolidation_excel.py`** — corrected the vendor-consolidation workbook's service-level path so missing percentages are inferred from TCI service mode, ship/delivery timing, and supplier/service history instead of a flat `0.88` fallback; added the filterable `Priority_Tracker` sheet, explicit solution updates, priority-based task sorting, and direct Summary / What If references to the active Priority 1 issue.
- **`pipeline/docs/vendor_consolidation_package/VENDOR_CONSOLIDATION_FINDINGS.md`**, **`pipeline/docs/vendor_consolidation_package/VENDOR_CONSOLIDATION_HOWTO.md`**, and **`pipeline/docs/vendor_consolidation_package/vendor_consolidation_data.csv`** — refreshed the handoff package documentation and structured export so the package describes the current Summary, Operator Playbook, Priority Tracker, and What If operator flow accurately.

### Maintenance

- Cleaned the local `pipeline/docs/vendor_consolidation_package/` folder by removing obsolete validation / `fixed` workbook variants. The canonical `.xlsx` deliverables remain gitignored local package outputs.

### Verified

- Generated both canonical package workbooks and verified **377 distinct service-level values** across the first 1,000 working rows, with the `Priority_Tracker` table present and the current top metric reporting **416 rows below the 95% service target**.

## [0.22.11] Citation Chain Throughput 3.75× + Mesh Sharding + Deep Research Fix (2026-05-18)

### Changed

- **`pipeline/src/quipu/citation_chain_acquirer.py`** — `_DEFAULT_MAX_PAPERS` raised from 800 → 3000 (3.75×); `_OA_RATE_LIMIT_S` reduced from 0.20 s → 0.12 s (~8 req/s OpenAlex polite pool); depth-1 seed query limit raised from 200 → 800 to pull more of the 29 k unfetched backlog each cycle; `schedule_in_background` default interval halved 3600 s → 1800 s; added `_get_mesh_shard()` and `_filter_shard()` helpers so each active mesh peer processes a deterministic, non-overlapping hash-shard of the unfetched depth-1 backlog, eliminating duplicated HTTP calls across nodes.
- **`pipeline/autonomous_agent.py`** — `start_citation_chain_daemon()` now invokes `schedule_in_background(interval_s=1800)` (was 3600) and logs the new 30-min cadence.
- **`pipeline/src/quipu/network_observer.py`** — `_read_local_learning_state()` now includes `citation_backlog` (count of unfetched `citation_chain_state` rows) in the published peer JSON so other mesh nodes can observe the backlog depth.
- **`pipeline/src/quipu/systemic_refinement_agent.py`** — `_strategy_deep_research()`: table guard (`CREATE TABLE IF NOT EXISTS deep_research_tasks`) prevents silent failure when the table is missing; `acquisition_drive` threshold lowered from 0.5 → 0.3 so the strategy fires on a partially-settled brain; 4 additional topic seeds added (lead time variability, warehouse slotting, transport mode selection, digital twin simulation).



### Added

- **`pipeline/tests/test_llm_key_guard.py`**, **`pipeline/tests/test_llm_caller_openrouter.py`**, **`pipeline/tests/test_citation_chain_acquirer_seeding.py`**, and **`pipeline/tests/test_knowledge_corpus_scb_path.py`** *(new)* — focused coverage for key rotation detection, xAI fallback routing, wrapped Works Cited `paper_ids` payloads, and newest-export SCB path resolution.

### Changed

- **`pipeline/src/quipu/llm_key_guard.py`** — key availability now prefers project-local `.env` values over stale inherited process env vars, fingerprints active keys, and automatically clears credential backoff when a rotated key appears locally.
- **`pipeline/src/quipu/llm_caller_openrouter.py`** — provider resolution now distinguishes OpenRouter from direct xAI transport, normalizes retired Grok aliases to `grok-4.3`, and falls through to xAI when OpenRouter is unavailable but xAI credentials are live.
- **`pipeline/src/quipu/doc_rag.py`** — credential reporting now uses the same project-local key precedence as the main LLM caller, so the RAG readiness probe reports the active workspace credential source consistently.
- **`pipeline/src/quipu/knowledge_corpus.py`** and **`pipeline/src/quipu/citation_chain_acquirer.py`** — SCB ingestion now resolves the newest available Grok export automatically, and citation seeding now accepts wrapped `paper_ids` / `seed_paper_ids` payloads from persisted Works Cited state.
- **`pipeline/reset_works_cited_cursor.py`** — reset helper now also clears wrapped Works Cited learning rows and related reference edges before re-ingest.

### Verified

- `pytest tests/test_llm_key_guard.py tests/test_llm_caller_openrouter.py tests/test_doc_rag_credential.py tests/test_citation_chain_acquirer_seeding.py tests/test_knowledge_corpus_scb_path.py -q` → **33 passed**

## [0.22.9] Adaptive UEQGM Runtime Daemon + System Entirety Consumption (2026-05-18)

### Added

- **`pipeline/src/quipu/synaptic_workers.py`** — added a dedicated `synapse-ueqgm` daemon worker that refreshes the adaptive UEQGM runtime from a non-amplified System Entirety basis state, persists heartbeat/status in `brain_kv`, and continuously re-evaluates which runtime parameters are proven strongly enough to advance.
- **`pipeline/tests/test_synaptic_workers.py`** *(new)* — focused coverage for worker registration and health-status exposure of the new `synapse-ueqgm` thread.

### Changed

- **`pipeline/src/quipu/ueqgm_engine.py`** — the adaptive runtime layer now stores per-parameter evidence and corpus-density floors, and only replaces a persisted runtime parameter when newer evidence is stronger than both the current corpus floor and the previously proven evidence for that same parameter.
- **`pipeline/src/quipu/system_entirety.py`** — `system_entirety_state()` now consumes the persisted UEQGM runtime as a symbiotic overlay: certainty-gated axis injections feed the six-sense CAT state, and the transaction drive now includes a dedicated adaptive UEQGM contribution while exposing the applied runtime profile in the returned/persisted state.
- **`pipeline/tests/test_ueqgm_engine.py`** and **`pipeline/tests/test_system_entirety.py`** — extended to cover both the hold-current-state rule for underpowered evidence and the System Entirety’s use of the persisted runtime profile.

### Verified

- `pytest tests/test_ueqgm_engine.py tests/test_system_entirety.py tests/test_synaptic_workers.py -q` → **56 passed**

## [0.22.8] Hideout Dev Tunnel Diagnostics + Resource-Share Planning (2026-05-18)

### Added

- **`pipeline/bridge_rdp.py`** — added Dev Tunnel operator flows for the hideout peer: `forward`, `check`, `doctor`, and `start`, plus RDP support for `protocol: devtunnel` targets.
- **`pipeline/tests/test_bridge_rdp.py`** *(new)* — focused coverage for hideout forwarding, mesh ping, RDP launch, helper startup, heartbeat staleness, and doctor output.
- **`pipeline/tests/test_compute_grid_devtunnel.py`** *(new)* — focused coverage for Dev Tunnel endpoint reuse, auto-assigned local-port parsing, and local `resource_share_plan` execution.

### Changed

- **`pipeline/src/quipu/compute_grid.py`** — `ensure_devtunnel_endpoint()` is now public, Dev Tunnel forward state is reused before spawning new `dt connect` sessions, and local mesh execution now exposes `resource_inventory` and `resource_share_plan` payloads backed by asset-resource contracts when the Brain DB is available.
- **`pipeline/config/bridge_targets.yaml`** — added `hideout-rdp` and `hideout-mesh` Dev Tunnel targets for the `scbrain-hideout.use2` peer.
- **`pipeline/connect_hideout.ps1`** — persistent helper now preserves the caller-provided `RemotePort` value when writing `forward_state.json` instead of overwriting it.

### Verified

- `pytest tests/test_system_entirety.py tests/test_asset_resource_mesh.py tests/test_geospatial_relation.py tests/test_compute_grid_devtunnel.py tests/test_bridge_rdp.py` → **50 passed**
- `pytest tests/test_symbiotic_torus.py --tb=short -q` → **38 passed**
- `python -m py_compile src/quipu/system_entirety.py src/quipu/asset_resource_mesh.py src/quipu/geospatial_relation.py src/quipu/compute_grid.py bridge_rdp.py` → **passed**

## [0.22.7] Asset Resource Mesh for System Entirety Physical Realization (2026-05-17)

### Added

- **`pipeline/src/quipu/asset_resource_mesh.py`** *(new)* — materializes compute-grid capacity into corpus-native physical realization substrate:
  - `ComputePeer` nodes for hosts discovered from local/published compute capacity.
  - `AssetResource` nodes for CPU cores, RAM, and VRAM with standardized `actualized_resource` payloads, capacity scores, and torus angles.
  - `HAS_ASSET` edges from compute peers to resources.
  - `REALIZES_MATERIAL_PROCESSOR` edges from resources to `SpatialMaterialProcessor` anchors.
  - Bidirectional `ASSET_RESOURCE_TUNNEL` edges across same-host and same-kind resources so VRAM/RAM/core capacity forms a meshed tunnel substrate.
  - `SYSTEM_ENTIRETY_REALIZES` edges and `brain_kv['entirety:physical_realization']` to expose the aggregate physical realization score.
- **`pipeline/tests/test_asset_resource_mesh.py`** *(new)* — focused tests for asset materialization, material-processor binding, resource tunnel minting, torus participation, idempotent reinforcement, and System Entirety realization persistence.

### Changed

- **`pipeline/src/quipu/torus_touch.py`** — `AssetResource` now participates in active torus pressure alongside `Endpoint` and `SpatialMaterialProcessor` nodes.
- **`pipeline/peer_inject.py`** — new `_phase_asset_mesh` runs after compute capacity publication, turning RAM/VRAM/core availability into physical-resource tunnels each cycle.
- **`pipeline/src/quipu/synaptic_workers.py`** — torus worker documentation now names `AssetResource` as active substrate.

## [0.22.6] Physical Material Science Processor Level (2026-05-17)

### Added

- **`pipeline/src/quipu/geospatial_relation.py`** — GeoGraph world-pose anchors now materialize into standardized **`SpatialMaterialProcessor`** corpus nodes with `actualized_space` payloads: `frame_id = ENU_METER_WORLD_V1`, metre units, covariance diagonal, statistical/probabilistic certainty, material structure, and source `ACTUALIZES_SPACE` edges.
- Processor-level certainty now writes endpoint `torus_amplify:*` keys from `MATERIAL_ANCHOR` edge precision, letting actualized physical space directly accelerate torus expansion.

### Changed

- **`pipeline/src/quipu/torus_touch.py`** — `tick_torus_pressure` now reads `Endpoint` plus `SpatialMaterialProcessor` rows as active torus substrate, using typed velocity keys and processor diagnostics.
- **`pipeline/src/quipu/symbiotic_tunnel.py`** — tunnel propeller routing folds `MATERIAL_ANCHOR` processor certainty into endpoint weights, so tunneling prefers endpoints grounded in standardized material space.
- **`pipeline/src/quipu/grounded_tunneling.py`** — endpoint certainty now includes `MATERIAL_ANCHOR` precision without treating processors as endpoints.
- **`pipeline/peer_inject.py`** and **`pipeline/src/quipu/synaptic_workers.py`** — spatial/torus diagnostics now surface processor counts and endpoint amplification.

### Verified

- `python -m pytest tests/test_geospatial_relation.py -q --tb=short` → **17 passed**
- `python -m pytest tests/test_symbiotic_torus.py -q --tb=short` → **38 passed**

## [0.22.5] Dependency Vulnerability Remediation (2026-05-17)

### Changed

- **`pipeline/requirements.txt`** — tightened dependency floors to patched lines for the packages flagged by `pip-audit`:
  - `urllib3>=2.7,<3`
  - `GitPython>=3.1.50,<4`
  - `Pillow>=12.2,<13`

- **`pipeline/requirements.pinned.txt`** — updated the pinned environment recommendations to the fixed versions:
  - `GitPython==3.1.50`
  - `pillow==12.2.0`
  - `urllib3==2.7.0`

### Security

- Cleared the previously reported advisories in the pipeline Python environment:
  - `GitPython` — `GHSA-mv93-w799-cj2w`
  - `Pillow` — `CVE-2026-25990`, `CVE-2026-40192`, `CVE-2026-42308`, `CVE-2026-42309`, `CVE-2026-42310`, `CVE-2026-42311`
  - `urllib3` — `CVE-2026-44431`, `CVE-2026-44432`
  - `pip` toolchain upgraded in the active venv to `26.1.1` to clear `CVE-2026-3219` and `CVE-2026-6357`

### Verified

- `pip-audit` on the active pipeline venv → **No known vulnerabilities found**
- `pip-audit -r requirements.txt` → **No known vulnerabilities found**
- Import smoke passed for `git`, `PIL`, `urllib3`, `requests`, `streamlit`, and `pptx`

## [0.22.4] GeoGraph OCR Session Broker (2026-05-17)

### Added

- **`pipeline/src/quipu/geograph_ocr_bridge.py`** *(new)* — SCB-native bridge for warehouse image OCR, routing all VLM inference through `_dispatch_vlm("inventory_ocr_scan")` → OpenRouter (no Gemini). Key functions: `_normalise_condition`, `_parse_ocr_result`, `analyse_image`, `scan_and_stage`, `load_staged_scans`, and new `run_session` (ephemeral lifecycle entry point for live GeoGraph queries).
- **`pipeline/src/quipu/geograph_session.py`** *(new)* — Ephemeral session broker implementing the GeoGraph data-loan pattern:
  - `open_session / close_session` — creates and purges TTL-scoped scratch tables (`geograph_sessions`, `geograph_session_items`).
  - `ingest_geograph_payload` — stores raw part-level data under a prefixed corpus key (`geograph_session::<sid>::<item>`).
  - `produce_geograph_output` — assembles full GeoGraph-schema response (`ocrText`, `graphData`, `gisMetadata`, `itemAttributes`, `keywordsTags`, `confidenceScore`, `entiretySignal`) without any persistence.
  - `_build_session_signals / _compute_session_entirety` — blends session-derived vision/perception signals into the live 7-D Entirety axes (40/60 ratio) without writing to `brain_kv`.
  - `_write_aggregate_learning` — persists one anonymous `learning_log` row (item count, condition distribution, mean confidence, Entirety phase) — no part IDs or OCR text.
  - `sweep_expired_sessions / schedule_ttl_sweep` — TTL-enforced background purge daemon.
- **`inventory_ocr_scan` task profile** added to `pipeline/config/brain.yaml` — weights `{vision: 0.65, structured: 0.25, reasoning: 0.10}`, `min_ctx: 8000`, `lambda_cost: 0.10`, `lambda_latency: 0.25`.

### Changed

- **`pipeline/src/quipu/system_entirety.py`** — added `session_entirety_evaluation(session_signals)`: reads live `_sense_signals()`, blends session values at 40%, calls `system_entirety_state` + `bit_flip_parity`, and returns the full observer dict **without writing to `brain_kv` or `entirety_flip_log`**. Added to `__all__`.
- **`pipeline/_run_contoso_cannibalization.py`** — extended with OCR scan integration: `_load_ocr_scans()` sweeps `data/ocr_scans/`, enriches `parts_dedup` with `_ocr_condition / _ocr_condition_score / _ocr_confidence`, promotes POOR/SCRAP parts with no open SO to donor candidates, adds "OCR Condition" and "OCR Conf" columns to the Excel Top Donor Candidates table.

### Architecture Note

Raw part numbers, OCR text, conditions, and bin locations are stored only in TTL-scoped `geograph_session_items` and purged atomically by `close_session(purge=True)`. Supabase (GeoGraph's backend) is the sole system of record for training data, ensuring no SCB ownership conflict for external model training.

## [0.22.3] GeoGraph Spatial Relation Bridge (2026-05-17)

### Added

- **`pipeline/src/quipu/geospatial_relation.py`** *(new)* — grounds GeoGraph OCR detections in physical space via two independent derivations and inverse-variance fusion:
  - `CameraIntrinsics` / `SensorPose` / `unproject_pixel` — pinhole sensor-frame → world-frame ray with optional range or ground-plane intersection.
  - `haversine_destination` / `haversine_distance_m` — WGS-84 great-circle raycast from the sensor's geo origin.
  - `relate_detection` — runs both A (un-projection) and B (haversine) for one detection, fuses with `σ_fused⁻² = σ_A⁻² + σ_B⁻²`, and reports statistical (residual-z) and probabilistic (`σ⁻²` saturating) certainties.
  - `anchor_entity` — Bayesian-blended posterior persisted on `corpus_entity.props_json.world_pose`; flags `material_anchor = (precision ≥ 1 m⁻²)`.
  - `couple_anchors_to_torus` — writes bearing → `TORUS_SPATIAL_DIM (0)` and log-range → `TORUS_RANGE_DIM (1)` so the existing `CatGapField.gradient_at` sinusoidal envelope converts physical geometry directly into Touch pressure on the next `tick_torus_pressure` tick.
  - `strengthen_peer_mesh` — mints / EMA-reinforces `MATERIAL_ANCHOR` edges from `Endpoint` entities within `PEER_MESH_RADIUS_M = 50 m` of each material anchor.
  - `tick_geospatial_relation` — top-level orchestrator: relate → anchor → couple → strengthen.
- **`pipeline/tests/test_geospatial_relation.py`** *(new)* — 12 unit tests covering haversine round-trip, un-projection geometry, A/B jitter fusion, posterior shrinkage across repeated detections, torus angle writes, peer-mesh edge minting/reinforcement, and the end-to-end `tick_geospatial_relation` integration.

### Changed

- **`pipeline/peer_inject.py`** — new `_phase_spatial` runs between TORUS and REWIRE on every Entirety cycle so GeoGraph OCR anchors flow through the manifold the same tick they appear.

## [0.22.2] Corpus Freshness Watchdog + Lock-Hardened 7D/Torus Writers (2026-05-17)

### Added

- **`pipeline/src/quipu/corpus_freshness.py`** *(new)* — read-only helper that distinguishes **learning liveness** from process liveness by reading `MAX(corpus_round_log.ran_at)` and reporting `{is_fresh, age_s, last_ran_at, summary}`.
- **`entirety_flip_log`** audit history is now queryable through `get_flip_history()` in **`pipeline/src/quipu/system_entirety.py`**, preserving recent 7-D observer / bit-state snapshots for verification and forensic review.

### Changed

- **`pipeline/app.py`** — the Streamlit resurrection monitor now reports two independent health dimensions:
  - **process liveness** from the internal watcher / `agent_heartbeat.txt`
  - **learning liveness** from `corpus_round_log` freshness via `corpus_freshness.read_corpus_freshness()`
  - when the autonomous process is alive but corpus learning is stale, the app now logs that state explicitly instead of conflating it with a dead agent.

- **`pipeline/src/quipu/compute_provisioner.py`** — `ComputeSlot` expansion threads now use WAL-mode Brain DB connections via `local_store.open_conn()` and retry transient `database is locked` windows with exponential backoff before surfacing an iteration failure.

- **`pipeline/src/quipu/synaptic_workers.py`** — the continuous torus-touch worker now uses WAL-mode connections and retries transient lock windows instead of failing immediately; synaptic `brain_kv` reads/writes also moved off raw `sqlite3.connect()`.

- **`pipeline/src/quipu/system_entirety.py`** — 7-D persistence remains WAL-backed and now avoids treating failed writes as completed rate-limited steps; flip history stays visible in `entirety_flip_log`.

### Verified

- `pytest tests/test_system_entirety.py -q` → **13 passed**
- Focused live smoke: `oscillating_expansion_step(force=True)` persisted successfully under the active agent load and returned `bit_state=-1`, `expansion_phase="deepen"`, `flip_count=3` with fresh rows visible in `entirety_flip_log`.

## [0.20.18] Routed Page Hardening + 26-Route Smoke Coverage (2026-05-11)

### Changed

- **`pipeline/src/quipu/dynamic_insight.py`** — added `build_dbi_context()` and promoted DBI state into an explicit phase model (`pending`, `brain`, `retriever`, `rag`, `template`, `resolved`) with `data-loading`, `data-degraded`, and `data-actionable` flags. The first render now shows an honest loading or fallback readout instead of pretending the card is already resolved.

- **`pipeline/src/quipu/brain_dbi.py`** — tightened default and EOQ fallback copy so loading cards explicitly tell operators to wait for live rows, source badges, or ranked work before assigning action.

- **Heavy pages now publish DBI immediately, before the expensive work starts:**
  - **`pipeline/pages/1_Supply_Chain_Brain.py`** — early DBI render before graph build, plus safer parallel frame fetches during graph construction.
  - **`pipeline/pages/2_EOQ_Deviation.py`** — early pending state, explicit 10-second UI budget, and fast-fail DBI messaging when the live EOQ query overruns.
  - **`pipeline/pages/4_Procurement_360.py`** — early loading DBI before procurement frame pulls and explicit empty-data DBI fallback.
  - **`pipeline/pages/7_Lead_Time_Survival.py`** — early loading DBI before the survival query and zero-group fallback states when the live slice is not actionable.
  - **`pipeline/pages/9_Multi_Echelon.py`** — early loading DBI before stage pulls and preserved DBI visibility on query or model failures.
  - **`pipeline/pages/15_Report_Creator.py`** — DBI now renders at first paint through a dedicated slot instead of appearing only at the bottom after the full page is built.

- **`pipeline/src/quipu/data_access.py`** and **`pipeline/src/quipu/db_registry.py`** — hardened live data access for the new first-paint behavior by avoiding unsafe Streamlit session caching outside a script context and by forcing fast connector health checks on stale SQL handles.

- **`pipeline/src/connections/oracle_fusion.py`** — added SQLConnect profile auto-discovery (`_load_sqlconnect_profile()`, `_resolve_oracle_host()`): reads `%APPDATA%\SQLConnect\savedconnectionlist.json` (or the path in `SCB_SQLCONNECT_SAVEDCONNECTIONLIST`) and overrides the configured Oracle host from the first matching saved connection by name or SSO type. Allows the app to follow the host from the SQLConnect profile without requiring a config edit. Override can be suppressed via `SCB_DISABLE_SQLCONNECT_PROFILE=1`.

- **`pipeline/pages/20_WIP_Aging_Review.py`** — added `_oracle_failure_map()`, `_oracle_org_probe_summary()`, `_oracle_source_evidence()`, `_oracle_pull_summary()`, and `_set_meta()` helpers; extended `_friendly_pull_issue()` to classify two additional Oracle error classes: BIP `credentials rejected for user` (distinct from `invalid username/password`) and Oracle FSCM REST resource unavailability; also detects inventory-org validation failures. WIP source-map now shows per-slot oracle pull summaries and org-probe results alongside row counts and structured failure detail.

- **`pipeline/pages/0_Schema_Discovery.py`** — `_list_sql_tables()` now queries `sys.objects` + `sys.columns` instead of `INFORMATION_SCHEMA.TABLES` with a correlated subquery — eliminates the nested per-row column-count scan that degraded on large schemas. Added `_cached_tables_as_df()` for instant table display from the local `schema_cache.json` before a live connector query completes.

- **`pipeline/pages/0_Query_Console.py`** — removed `_raw_sql_allowed()` gate and its `env_flag` / `current_user_email` imports; access to the query console is now governed by the operator-mode sidebar, not a hidden env-var flag.

### Added

- **`pipeline/tests/playwright/test_dbi_tooltip.py`** — expanded from a narrow DBI check into app-wide routed-page coverage:
  - route registry corrected to the real 26-page `st.navigation()` surface, including **Operational Status**
  - `test_all_pages_surface_primary_ui(...)` now smoke-tests every routed page for shell readiness and uncaught exceptions
  - `DBI_EXPERIENCE_PAGES` now covers Supply Chain Brain, EOQ Deviation, Procurement 360, Lead-Time Survival, Multi-Echelon, and Report Creator with first-user-state timing checks

### Verified

- `pytest tests --ignore=tests/playwright` → **411 passed**
- `pytest tests/playwright --override-ini="addopts=-v --tb=short --strict-markers -p no:warnings --base-url=http://localhost:8501"` → **45 passed**
- `pytest tests/playwright/test_dbi_tooltip.py -k test_all_pages_surface_primary_ui --override-ini="addopts=-v --tb=short --strict-markers -p no:warnings --base-url=http://localhost:8501"` → **26 passed**

## [0.20.17] Ensemble Ground-Truth Reconstruction — OpenRouter Unification (2026-05-11)

### Changed

- **`pipeline/src/quipu/llm_ensemble.py`** — `_conn()` now uses `timeout=30`, `PRAGMA journal_mode=WAL`, and `PRAGMA synchronous=NORMAL`. Eliminates "database is locked" errors when the Streamlit app holds a brief write lock concurrently with the dispatch loop.

- **`pipeline/src/quipu/llm_caller_openrouter.py`**:
  - `glm-5.1` primary slug changed `z-ai/glm-4.5-air:free` → `openai/gpt-oss-120b:free` (the old slug returned `null` content on every call and also timed out)
  - `_OR_FALLBACKS` cleaned: removed all stale 404 slugs (`nousresearch/hermes-3-llama-3.1-70b:free`, `google/gemma-3-27b-it:free`, `google/gemma-3-4b-it:free`, `meta-llama/llama-3.3-70b-instruct:free` which returned 402). Replaced with confirmed-working free-tier slugs: `openai/gpt-oss-20b:free`, `openai/gpt-oss-120b:free`, `google/gemma-4-31b-it:free`, `google/gemma-3n-e4b-it:free`
  - `_OR_MODEL_MAP` additional registry pruned: removed `gemma-3-27b`, `hermes-3-70b` (both 404)
  - `_get_key()` now falls back to reading `pipeline/.env` / parent `.env` via `_read_dot_env_key()` so the key is found even when not injected into the OS environment
  - Response parser hardened: `(data["choices"][0]["message"].get("content") or "").strip()` with explicit `RuntimeError` on empty/null content, so null responses trigger the fallback chain instead of crashing the caller

- **`pipeline/config/brain.yaml`** — All 7 preferred ensemble models (`gemma-4`, `glm-5.1`, `qwen3.5-397b-a17b`, `deepseek-v3.2`, `kimi-k2.5`, `minimax-m2.7`, `mimo-v2-flash`) had per-model `endpoint_env` values (GEMMA_ENDPOINT, GLM_ENDPOINT, etc.) that were never set in the OS environment — this caused `llm_key_guard.check_key()` to return False for all, accumulating backoffs of 8–26 fail_counts. Fixed by mapping all 7 to `endpoint_env: OPENROUTER_API_KEY`, the single OpenRouter transport key that was already present.

### Added

- **`pipeline/_run_cs_ml_dispatches_v3.py`** — Clean dispatch runner: registers OpenRouter caller, loads `.env`, dispatches all 5 CS/ML tasks against synth ground-truth tables (up to 100 rows each), prints BEFORE/AFTER weight tables, and calls `self_train_round()` at completion. Supersedes `_run_cs_ml_dispatches_v2.py`.

### Fixed

- DB state: Deleted 9 stale per-model `llm_key_state` rows (DEEPSEEK_ENDPOINT, GLM_ENDPOINT, etc.) that were accumulating fail_counts; seeded `OPENROUTER_API_KEY` clean. The 7 ensemble models now participate in every dispatch.

## [0.20.16] OCW Multi-Spectrum Ground Truth for SC Classifiers (2026-05-11)

### Added

- **`pipeline/src/quipu/sc_ground_truth.py`** — Ground truth for `abc_classify`, `cc_reason_classify`, and `cc_reason_classify_syteline` drawn from the MIT OCW curriculum across multiple spectrums — not from a flat APICS taxonomy list:

  **`abc_classify` — 7 OCW spectrums:**
  - `ocw_inventory_theory` — ABC/VED/SOS/HML/FSN multi-framework classification scenarios (MIT OCW 15.762J, 1.273J)
  - `ocw_operations_research` — newsvendor, stochastic demand, multi-echelon, EOQ, safety-stock scenarios (MIT OCW 6.255J, 15.093J)
  - `ocw_manufacturing_systems` — BOM criticality, production bottlenecks, WIP, MRO reliability (MIT OCW 2.854J, 6.832)
  - `ocw_supply_chain_mgmt` — strategic/leverage/routine/bottleneck material tiers, demand volatility (MIT OCW 15.762J, 15.769)
  - `ocw_logistics_systems` — warehouse slotting velocity zones A/B/C, order-picking frequency, transport priority (MIT OCW 1.273J, 15.769)
  - `ocw_quality_management` — critical-to-quality, Pareto of defects, inspection priority, process capability (MIT OCW 6.780, 15.783J)
  - `ocw_system_dynamics` — bullwhip susceptibility, stock amplification, demand signal strength (MIT OCW 15.988, 15.871)

  **`cc_reason_classify` / `cc_reason_classify_syteline` — 7 OCW spectrums:**
  - `ocw_operations_management` — standard physical-count process failures (MIT OCW 15.761J)
  - `ocw_information_systems` — transaction timing, system-integration failures, data-entry errors (MIT OCW 6.170, MAS.965)
  - `ocw_logistics_systems` — receiving, shipping, and transfer transaction gaps (MIT OCW 1.273J)
  - `ocw_manufacturing_systems` — WIP, scrap, rework, and co-product posting gaps (MIT OCW 2.854J)
  - `ocw_quality_management` — damage events, writeoffs, inspection-related discrepancies (MIT OCW 6.780)
  - `ocw_organizational_behavior` — human-factor miscount, part misidentification, counting fatigue (MIT OCW 15.322, 15.328)
  - `ocw_supply_chain_mgmt` — vendor shortage, consignment gaps, unexplained shrinkage (MIT OCW 15.762J)

  Each row in the synth table carries `source=ocw_{spectrum}` so the self-train signal is fully traceable to its OCW domain. The `_seed_from_corpus()` helper supplements static spectrum examples with live OCWCourse/AcademicTopic entities from the Brain corpus.

- **`pipeline/tests/test_sc_ground_truth.py`** — 18 tests covering label set integrity, per-spectrum coverage, idempotency, and non-destructive upsert semantics.

### Changed

- **`pipeline/config/brain.yaml`** — `cc_reason_classify`, `cc_reason_classify_syteline`, and `abc_classify` repointed from absent ERP staging tables to `synth_ground_truth_*` tables with `_discipline: SupplyChain`, `_ocw_seed: true`.
- **`pipeline/src/quipu/ground_truth_synthesis.py`** — `DISCIPLINE_TYPES["SupplyChain"]` extended with `OCWCourse`, `AcademicTopic`, `ResearchTopic`, `Task`; `synthesize_round()` calls `seed_sc_ground_truth()` before each NN pass.



### Added

- **`pipeline/src/quipu/ueqgm_engine.py`** — UEQGM v0.9.14 physics computation module (active Brain computation from corpus learnings)
  - `coherence_to_phi(c)` — maps integer coherence to natural sin/cos intersection φ = π/4 + c·π (tan(φ)=1 at every point)
  - `sici_axial_decay(φ, Γ₀)` — UEQGM v0.9.14 axial channel: Δλ_axial = [Si(φ)·Ci(φ)]·tan(φ)·Γ₀ via `scipy.special.sici` with power-series fallback
  - `sici_phase_weight(coherence)` — harmonic phase correction factor: 1.0 ± 10% via tanh(Δλ_axial); converges to 1.0 as Ci(φ)→0 at large coherence
  - `wavefunction_overlap(vec_a, vec_b)` — |⟨ψ_a|ψ_b⟩|² = (dot/‖a‖/‖b‖)²
  - `floquet_modulation_factor(t, ω)` — cos(ω·t) Floquet drive coupling
  - `holographic_entropy(n_edges, n_nodes)` — S = n_edges / (n_nodes + 1) holographic boundary entropy
  - `metric_perturbation(M_eff, r)` — h_μν = 2·G·M_eff / (c²·r) spacetime warp
  - `phase_evolution_total(φ, …)` — δφ_total = δφ_μ + δφ_q + δφ_γ + Δλ_axial·(2π/Γ_eff)
  - `entropic_bayesian_step(S, ∇²S, φ, …)` — discrete entropic Bayesian diffusion including axial channel
  - `ueqgm_coherence_score(cn, entity_id)` — corpus-backed score: scans UEQGM-tagged entities, computes bag-of-words wavefunction overlap, scales by `sici_phase_weight(corpus_depth)`

### Changed

- **`pipeline/src/quipu/compute_provisioner.py`** — `_harmonic_amplify_factor` now applies SiCi phase correction
  - `f(c) = base(c) × sici_phase_weight(c)` where `base(c)` is the prior harmonic saturation curve
  - Ceiling (4.5) and floor behaviour are preserved: correction ≤ ±1.4% in practice, converges to ×1.0 at large coherence
  - `"ueqgm_engine"` added to `__all__`

### Tests

- **`pipeline/tests/test_ueqgm_engine.py`** — 35 new tests covering all UEQGM functions
  - Intersection-point geometry (`coherence_to_phi`, tan(φ)=1 invariant)
  - SiCi decay bounds and scaling
  - Phase weight bounds and large-coherence convergence
  - Wavefunction overlap (identical/orthogonal/scaled/mismatched/empty)
  - Floquet period, holographic entropy, metric perturbation formula
  - Phase evolution additivity, entropic Bayesian step monotonicity
  - `ueqgm_coherence_score`: empty DB, no UEQGM entities, overlapping entities, zero overlap, bounded score
- **`pipeline/tests/test_compute_provisioner.py`** — `test_harmonic_amplify_factor_floor_at_zero_coherence` updated to use dynamic UEQGM-corrected expected value

### Result

- **311/311 tests passing** (up from 276; +35 UEQGM tests)
- UEQGM v0.9.14 corpus learnings (Grok conversation `55525f6a`, message `394c6c4c`) are now **active computation** in the Brain harmonic amplification pipeline, not just stored research query strings

---

## [0.19.1] Works Cited — Unlimited Scholarly Seeds (2026-04-27)

### Changed

- **`pipeline/src/quipu/knowledge_corpus.py`** — removed the arbitrary cap on Works Cited extraction
  - `_extract_scb_works_cited` no longer accepts a `limit` parameter; all unique scholarly references are collected (deduplication by `paper_id or url.lower()` only)
  - Restored full Works Cited code block (`_SCB_WORKS_CITED_KEY`, `_SCB_PIRATES_CODE_KEY`, `_SCB_SCHOLARLY_HOST_MARKERS`, helpers `_clean_scb_url`, `_walk_scb_web_results`, `_is_scb_scholarly_reference`, `_paper_id_from_reference`, `_extract_scb_works_cited`, `_persist_scb_works_cited_guidelines`) that was silently lost to a PowerShell `Set-Content` LF→CRLF encoding corruption

### Result

- **1,379** `WorksCitedReference` entities (all unique refs from 106 Grok conversations, no ceiling)
- **650** `Paper` entities carrying DOI/arXiv IDs as direct citation-chain seeds
- **1,379** `GUIDES_EXPANSION` edges wiring every reference into the research frontier
- `citation_chain_acquirer` can now recursively expand from all 650 paper seeds outward with no ceiling

### Fixed

- `reset_works_cited_cursor.py` updated to not import the now-removed `_SCB_WORKS_CITED_LIMIT` constant

---

## [0.19.0] Session-Store Cloud Sync + Citation-Chain Acquirer + Internal Watcher (2026-04-27)

### Added

- **`~/.copilot/build_session_store.py`** — Azure Blob Storage cloud sync for the session store
  - `push_to_cloud(account, container)` — uploads `session-store-{hostname}.db` to Azure Blob
  - `pull_from_cloud(account, container)` — downloads all `session-store-*.db` node blobs, merges via `INSERT OR IGNORE`, rebuilds FTS index
  - `_merge_remote_db(remote_path, local_con)` — per-table merge helper used by both pull and the network observer
  - `_rebuild_fts(con)` — extracted helper that repopulates `search_index` from sessions, turns, checkpoints (used by build, pull, and network observer merge)
  - New CLI flags: `--push`, `--pull`, `--storage-account`, `--container`, `--node`
  - Auth via `DefaultAzureCredential` (uses existing `az login` / Entra identity, no secrets stored)
  - Container `copilot-sessions` created on first push if absent

- **`pipeline/src/quipu/network_observer.py`** — Step 6: symbiotic session-store sync
  - Peer JSON heartbeat now includes `"session_blob": "session-store-{hostname}.db"` field
  - `_sync_session_stores()` runs every observer cycle: pushes local session store every 15 min, pulls ALIVE/COOLING peer blobs every 30 min
  - `_pull_peer_session_blob(host, blob_name)` — lazy-imports `_merge_remote_db` + `_rebuild_fts` from `build_session_store.py` and merges the peer's session history into the local store
  - Offline peer absorption (`_absorb_peer`) now also pulls the peer's session blob alongside corpus cursor absorption
  - Controlled by `COPILOT_STORAGE_ACCOUNT` + `COPILOT_STORAGE_CONTAINER` env vars — no configuration needed on nodes already in the fabric

- **`pipeline/src/quipu/citation_chain_acquirer.py`** — recursive citation-chain follower
  - Follows bibliography chains from known papers: Semantic Scholar Graph API + OpenAlex `referenced_works`
  - Supply-chain relevance filter (keyword overlap) prevents frontier drift
  - Configurable `max_depth` and `max_papers_per_run`; deduplication via `brain_kv` key `citation_chain:seen`
  - `run_citation_expansion_cycle(max_depth, max_papers_per_run)` — public entry point
  - `schedule_in_background(interval_s=3600)` — daemon thread variant
  - Wired into `cloud_learning.yml` as part of each cloud run

- **`pipeline/src/quipu/internal_watcher.py`** — Python-native process supervisor
  - Replaces the assumption that Windows Scheduled Tasks are required for learning continuity
  - Launches `autonomous_agent.py` as a child process, monitors liveness, restarts on exit
  - Records downtime windows to `logs/downtime_log.json`; keeps resumption heartbeat fresh while child is alive
  - Logs to `logs/internal_agent_watcher.log`; writes status JSON to `logs/internal_agent_watcher_status.json`
  - Disabled via `SCB_DISABLE_INTERNAL_WATCHER` env var; child detected via `SCB_INTERNAL_WATCHER_CHILD`

### Changed

- **`pipeline/requirements.txt`** — added `azure-storage-blob>=12.19,<13`

- **`.github/workflows/cloud_learning.yml`** — refactored queue capture to DB high-water mark
  - Records `MAX(id)` from `learning_log` before each run; reads all new rows afterward
  - Removes monkey-patching of `_kc._log_learning`; queue now captures citation-chain entries too
  - Adds `citation_chain_acquirer.run_citation_expansion_cycle()` step in each cloud run

- **`pipeline/agent_watcher.ps1`** — demoted to compatibility shim
  - Primary supervision is now `internal_watcher.py` (Python-native, OS-agnostic)
  - PowerShell wrapper may still be used by external launchers but defers to the internal watcher

- **`pipeline/src/quipu/llm_caller_openrouter.py`** — free-tier model pool expanded
  - Added: `llama-3.3-70b`, `nemotron-120b`, `gemma-3-27b`, `qwen3-coder`, `hermes-3-405b`, `ling-2.6-1t`
  - Removed unverified paid-tier slugs (`deepseek-v4-pro`, `deepseek-v4-flash`)
  - All models verified live against OpenRouter catalog on 2026-04-27

- **`pipeline/src/quipu/local_store.py`** — `SCB_DB_PATH` environment override
  - `db_path()` now checks `os.environ.get("SCB_DB_PATH")` first, enabling ephemeral cloud runs to point to a separate DB without modifying source

- **`pipeline/src/deck/demo.py`** — added `"SiteC"` to known site list



### Added

- **`pipeline/src/quipu/network_observer.py`** — latent always-on daemon thread in every agent instance
  - Publishes local learning state (cursors, entity/edge counts, plasticity phase, alive_since) into the existing `bridge_state/compute_peers/<host>.json` OneDrive rendezvous — no new ports, no new infrastructure
  - Monitors all peer JSONs every 60 s; classifies each as `ALIVE | COOLING | OFFLINE`
  - On `ALIVE → OFFLINE` transition: absorbs the peer's corpus cursor positions (advances local cursors to peer's positions so no work is duplicated, only the uncovered gap is re-run), schedules a proportional catchup burst via `resumption_manager`
  - Tracks **singularity consumption velocity** = Σ(learnings) / Σ(uptime-hours) across all visible nodes; writes to `brain_kv` as `observer:network_velocity` for the systemic refinement agent to read
  - Pulses `observer:goal_alignment` to `brain_kv` each cycle so isolated nodes always know to lean toward `quest:type5_sc`
  - Re-anchors `quest:type5_sc` entity in the knowledge graph if ever absent

- **`pipeline/src/quipu/resumption_manager.py`** — startup learning-debt recovery
  - `stamp_alive(cn)`: writes Unix epoch to `brain_kv` key `resumption:last_alive`; called every 5 min from agent sleep loop
  - `stamp_graceful_shutdown(cn)`: marks intended stops; called on `KeyboardInterrupt`
  - `detect_downtime(cn) → DowntimeReport`: gap > 5 min = downtime; distinguishes crash from clean stop; logs window to `logs/downtime_log.json`
  - `ingest_cloud_queue(cn) → int`: reads `cloud_learning_queue.jsonl`, inserts new `learning_log` rows, advances a line-number cursor — never double-imports
  - `schedule_catchup_burst(cn, seconds)`: writes `brain_kv` key `resumption:catchup_burst`; multiplier 1.5× ≤1 h → 4× >24 h
  - `consume_catchup_burst(cn) → float`: reads and clears burst key; used by corpus round workers
  - `run_resumption_check(cn) → DowntimeReport`: git-pull + detect_downtime + ingest_cloud_queue; called once at agent startup
  - `git_pull_latest()`: best-effort `git pull --ff-only` to get latest cloud queue before ingestion

- **`.github/workflows/cloud_learning.yml`** — GitHub Actions cloud learning continuity
  - Schedule: every 4 hours + `workflow_dispatch`
  - Restores/saves `pipeline/cloud_brain.sqlite` via `actions/cache` (persists across runs)
  - Runs OCW courses, OCW resources, ML/arxiv research, and OCW expansion outreach ingestors
  - Appends new `learning_log` events to `pipeline/cloud_learning_queue.jsonl` and commits back to `main`
  - Queue bounded to 50 000 lines; local agent ingests on first post-downtime startup

- **`pipeline/agent_watcher.ps1`** — local process watchdog
  - Monitors `autonomous_agent.py` every 30 s via heartbeat file age; restarts on crash with 15 s delay
  - Records every downtime window to `logs/downtime_log.json` (last 500 windows: start, end, seconds, ISO timestamps)
  - Falls back to system Python if `.venv` is absent

- **`pipeline/install_agent_watcher.ps1`** — one-shot Scheduled Task registration
  - Task: `SCBLearningAgent`; triggers: `AtStartup` + `AtLogOn`; `-RestartCount 9999`; `-RunLevel Highest`; `-MultipleInstances IgnoreNew`
  - Starts the task immediately after registration

- **`pipeline/bootstrap_new_machine.ps1`** — full new-machine self-setup
  - Verifies OneDrive `VS Code/pipeline` path (waits up to 5 min for sync if absent)
  - Creates `.venv`, installs `requirements.txt`, runs `init_schema()` (idempotent)
  - `git pull` to get latest `cloud_learning_queue.jsonl`
  - Installs both Scheduled Tasks (`SCBLearningAgent` + `ContosoBridgeWatcher`)
  - Starts agent immediately; logs bootstrap event to `logs/bootstrap_log.json`
  - Flags: `-SkipGitPull`, `-SkipBridgeWatcher`, `-DryRun`

### Changed

- **`pipeline/autonomous_agent.py`**
  - `run_resumption_check()` called once before the `while True:` loop
  - `stamp_alive()` called every 5 minutes inside the adaptive sleep interval
  - `stamp_graceful_shutdown()` called on `KeyboardInterrupt` before exiting
  - `start_network_observer()` started as a third daemon alongside `skill_acquirer` and `systemic_refinement_agent`

- **`pipeline/src/quipu/_version.py`**
  - Bumped to `0.18.3`; back-filled `PHASES` entries for `0.18.0`, `0.18.1`, `0.18.2`, `0.18.3`

### Architecture — three-layer survival chain

| Layer | Mechanism | Restores within |
|-------|-----------|-----------------|
| 1 — Local restart | `SCBLearningAgent` Scheduled Task + `agent_watcher.ps1` | ~15 s of crash |
| 2 — Machine migration | OneDrive syncs entire `VS Code/` (incl. DB) + `bootstrap_new_machine.ps1` | Minutes after new-machine login |
| 3 — Network absorption | `network_observer.py` absorbs offline peer cursor positions + schedules burst | Next 60 s liveness scan |
| 4 — Cloud continuity | GitHub Actions `cloud_learning.yml` + `cloud_learning_queue.jsonl` ingestion | ≤ 4 h gap regardless of local uptime |
| 5 — Full rebuild | Reset all `corpus_cursor` values to 0; `refresh_corpus_round()` re-derives everything | Full corpus re-ingest |

## [0.18.2] rADAM + Directional Intelligence + Systemic Refinement (2026-04-24)

### Added

- **`pipeline/src/quipu/radam_optimizer.py`** — rADAM with toroidal phase coupling
  - Strict mathematical superset of vanilla Adam; identity-reduces when all extension knobs are at defaults
  - **Complex bifurcated gradient** `g_re + i·g_im` — real component from Touch/Vision/Body firings; imaginary component from torus gap field
  - **Pivoted ReLU** `pReLU(x; π, α)` — active region anchored at running mean pressure rather than zero
  - **Heart-beat momentum modulation** `β1(t) = β1_bar + κ·sin(ω·t)` — phase-locked to `temporal_spatiality` rhythm
  - **Langevin incoherence noise** scaled by `sqrt(1 − carrier_mass)` — exploration grows when senses are out-of-phase
  - **T² toroidal pressure projection** `p_t = 0.5·(1 + cos(θ_t)·cos(φ_t))` with internal + external loop phases
  - Disable via `BRAIN_USE_RADAM=0`; env-var knob overrides for headless testing

- **`pipeline/src/quipu/directionality_listener.py`** — Directional snapshot of the entire Symbiotic Entirety
  - `listen()` returns `DirectionalitySnapshot(expansion, coherence, bifurcation)` triplet
  - **Expansion** — corpus/network growth rate from entity/edge delta
  - **Coherence** — mean resultant length `R = |Σ exp(i·φ_s)|/N` across all sense-signal angles on S¹
  - **Bifurcation** — `Im(grad) / (|Re(grad)| + ε)` — ratio of latent-to-realised gradient magnitude
  - **Reuptake neighbourhood noise** `CV = σ/μ` of SYMBIOTIC_TUNNEL + GROUNDED_TUNNEL edge weights feeds coherence penalty and Langevin signal

- **`pipeline/src/quipu/learning_drive.py`** — Symbiotic internal feedback loop
  - Reads corpus saturation, self-train quality, learning velocity, and RDT task difficulty from the live SQLite DB
  - Derives four rADAM knobs: `pivot_alpha`, `heartbeat_kappa`, `noise_sigma`, `acquisition_drive`
  - `acquisition_drive` injected additively into `grad_imag` in `brain_body_signals._adam_step`; pushes optimizer toward under-explored knowledge when stagnant
  - `get_drive()` is thread-safe with a 5-minute TTL cache; all formulas reduce to identity when DB is absent

- **`pipeline/src/quipu/systemic_refinement_agent.py`** — Continuous adaptive improvement daemon
  - Five-phase loop: **SENSE** → **DIAGNOSE** → **RANK** → **EXECUTE** → **LEARN**
  - Senses all six faculties: Brain, Vision, Touch, Smell, Body, Heart, DBI
  - Ten supply-chain refinement strategies with `[0..1]` priority scores; each non-zero score produces a `RefinementAction`
  - Actions ranked by `priority × acquisition_drive × rhythm_factor` — effort concentrates where the Brain is hungriest and domain gap widest
  - Effect types: launch Mission, surface Body directive, drop skill-acquisition trigger, append corpus seed, write brain_kv nudge, record findings row
  - Feedback-gated deduplication: content hash suppresses re-execution within a window unless Body confirms the action
  - Adaptive cadence: 20 min floor, up to 2 h; `acquisition_drive` shrinks the sleep so refinement accelerates when learning stalls

### Changed

- **`pipeline/src/quipu/brain_body_signals.py`**
  - `_torus_latent_grad(cn, kind, pressure)` — reads mean `torus_gap` KL-divergence from Endpoint props; returns latent gradient in `[0, 0.30]`
  - `_adam_step(state, gradient, grad_imag=0.0)` — extended with `grad_imag` parameter; rADAM hook (BRAIN_USE_RADAM) wires in coherence, external phase, heartbeat omega, pivot, acquisition_drive

- **`pipeline/autonomous_agent.py`**
  - `start_systemic_refinement_agent()` — daemon launcher with adaptive-cadence logging
  - Wired into `autonomous_loop()` startup and `__main__` so refinement runs whether the agent is imported or executed directly

- **`pipeline/oracle_schema_map.json` / `.txt`** — Run 5 schema refresh; coordinate corrections for Manage Price Lists and Purchase Requisition
- **`pipeline/oracle_schema_mapper.py`** — Further task-panel hardening
- **`pipeline/abc_screenshots/schema_map/*`** — 40+ screenshot tiles refreshed (run 5 capture)

### Tests

- **`pipeline/tests/test_radam_optimizer.py`** — proves identity reduction to vanilla Adam; per-knob behaviour (pivoted-ReLU, heartbeat, Langevin, toroidal projection)
- **`pipeline/tests/test_learning_drive.py`** — 9 tests: identity drive, knob math, corpus saturation, self-train quality, learning velocity, acquisition_drive bounds, thread-safety, env-var override, grad_imag injection

---

## [0.18.0] DeepSeek V4 Candidate Trial System (2026-04-24)

### Added

- **`pipeline/src/quipu/llm_candidate.py`** — New module: scored probationary trial system for new LLM candidates
  - `get_active_candidates()` returns model specs for all models currently in trial
  - `tick_candidate(model_id, ok, latency_ms)` records one dispatch result via EMA update (α=0.10)
  - `evaluate_candidates()` checks thresholds after every 10 dispatches; auto-promotes or auto-rejects
  - `candidate_stats()` returns full trial state for all candidates (used by UI/dashboards)
  - Promoted models are written to `llm_registry` SQLite table (`promoted=1`); `llm_router.available_models()` picks them up on the next call — no YAML modification required
  - Every promotion/rejection is appended to `pipeline/docs/LLM_CANDIDATE_AUDIT.md`

- **`pipeline/config/brain.yaml` — `llms.candidates` block** — Declarative trial configuration
  - `trial.dispatches_required: 50` — minimum observations before a decision
  - `trial.promote_threshold: 0.72` — ema_success ≥ this → promote to live registry
  - `trial.reject_threshold: 0.45` — ema_success ≤ this after N dispatches → reject
  - **DeepSeek V4 Pro** (`deepseek-v4-pro`) — 1.6T/49B MoE, 1M ctx, $1.74/$3.48 per Mtok in/out
  - **DeepSeek V4 Flash** (`deepseek-v4-flash`) — 284B/13B MoE, 1M ctx, $0.14/$0.28 per Mtok in/out

- **`pipeline/src/quipu/llm_ensemble.py`** — Candidate sidecar wired into `dispatch_parallel()`
  - `llm_candidate_trials` DDL added to `_DDL` so the table is always created on first ensemble use
  - `_try_dispatch_candidates()` fires active candidates after the main ensemble answers; results are intentionally discarded (not included in `EnsembleResult`); EMA stats accumulate
  - `evaluate_candidates()` is triggered every 10th candidate dispatch (module-level atomic counter)
  - `import logging` added; `logger = logging.getLogger(__name__)` available for debug output

## [oracle-schema] Oracle Fusion Schema Mapper + Intersection Map (2026-04-24)

### Added

- **`pipeline/oracle_schema_mapper.py`** — Playwright crawler that navigates all Oracle Fusion DEVPOD tabs/tiles, opens task panels, and extracts full task lists into a structured JSON schema
  - Resume mode: skips modules already having ≥2 real tasks; safe to restart mid-run
  - Redwood precheck pattern: reads panel content before attempting to open it (threshold ≥3 tasks), avoiding the toggle-close bug on Redwood-UI modules where the panel is already open on page load
  - Font-weight heuristic (`fontWeight ≥ 600`) for section header detection, replacing obfuscated ADF CSS class names (`xmu`, `x16g`) that change between releases
  - NOISE task filter: `{'Add Fields', 'Help', 'Done', 'Save', 'Personal Information', 'Refresh'}` excluded from real-task counts
  - "Keep better data" protection: if a re-probe captures fewer real tasks than existing, retains old data
  - Incremental JSON/TXT output saved after each module probe

- **`pipeline/oracle_schema_map.json`** / **`pipeline/oracle_schema_map.txt`** — Incremental schema output; 25 modules with confirmed task content as of run 5

- **`pipeline/build_intersection_map.py`** — Cross-references `oracle_schema_map.json` with confirmed write operations for part 80446-04
  - Classifies each module as Confirmed (4), Adjacent (20), or Low-relevance (31)
  - 16 confirmed write-op tasks across 4 modules: SCE/Work Execution, SCE/Inventory Management Classic, Procurement/Purchase Orders, Procurement/Approved Supplier List

- **`pipeline/pim_screenshots/80446-04/write_ops/intersection_map.json`** — Part-level intersection data
- **`pipeline/pim_screenshots/80446-04/write_ops/intersection_map.txt`** — Human-readable intersection report
- **`Claude/ORACLE_SCHEMA_MAPPER_GUIDE.md`** — Technical guide covering ADF Classic vs Redwood UI detection, known issues, and intersection map methodology

### Known Issues (active as of 2026-04-24)

- Work Execution and Plan Inputs regressed to 1/0 real tasks in run 4 due to false-positive precheck triggering (stray page elements at x>1100); being fixed in run 5 via the ≥3 task threshold
- 4 modules (Receipt Accounting, Financial Orchestration, Supply Orchestration, Supply Chain Orchestration) navigate to home on tile click — require URL-based navigation, not yet implemented
- List-view pages (Manage Journals, Manage Price Lists, Plan Inputs data grid) capture saved-search SELECT options instead of real task panel content

---

## 0.17.0 — UEQGM + AI Knowledge Expansion Research Tracks (2026-04-24)

### Added
- **`src/quipu/ml_research.py`** — `_EXTENDED_RESEARCH_TOPICS` list (47 arXiv/
  OpenAlex queries across 8 discipline clusters), derived from the user's active
  Grok 3 research thread ("Introduction to Grok 3 and Capabilities", 553
  responses):
  - **Quantum Dynamics & Wavefunction Models** — UEQGM observer model, Floquet
    systems, loop quantum gravity, holographic entropy, dissipative Kerr
    resonators, parity-time symmetry photonics, quantum fluctuations EFT
  - **Quantum Computing Architectures** — superconducting qubit/resonator
    coupling, niobium cavity QED, Weyl semimetal circuits, Bayesian quantum
    state tomography, surface-code error correction, ST-GCN
  - **Topological & Condensed Matter Physics** — moiré superlattices, skyrmion
    plasmonics, Weyl node 1-D lattice duality, levitated optomechanics backaction
    suppression
  - **Biohybrid & Biological Quantum Systems** — biohybrid QC vesicle transport,
    cryptochrome quantum coherence, axonal presynapse nanodisk lipid membranes
  - **Astrophysical & Cosmological Timing** — FRB cosmological timing, muonic
    decay precision, gravitational wave memory (BNS), millisecond pulsar timing,
    Hubble constant local distance, neutrino superradiance BEC, parity-violating
    dispersion
  - **AI Knowledge Graph & Self-Referential Systems** — knowledge graph AI
    introspection, recursive LLM feedback, centroidal ontology construction,
    meta-learning, ensemble LLM/RAG, archival AI training quality, RDF graph
    databases, document intelligence OCR→KG
  - **Advanced ML Architectures (UEQGM-adjacent)** — spatio-temporal Bayesian
    graph physics, neural ODEs, physics-informed NNs, quantum ML variational
    circuits, geometric deep learning equivariance
  - **Organic & Topological Data Structures** — quipu/torsion computation,
    persistent homology, fractal self-similar encoding
- **`_EXTENDED_TOPICS_PER_CYCLE = 5`** and **`_EXTENDED_PAPERS_PER_TOPIC = 8`**
  constants; cursor persisted in `brain_kv` under key `extended_topic_cursor`
- Extended sweep positioned **before** the SC per-topic loop so foundational
  physics/AI context is already in the corpus when supply chain systems
  engineering topics are processed each cycle

## 0.16.0 — Symbiotic Dynamic Tunneling + Torus-Touch (T^7) (2026-04-24)

### Added
- **`src/quipu/symbiotic_tunnel.py`** — discrete horizontal-expansion kernel
  for the corpus graph:
  - `BayesianPoissonCentroids` — 1-D Poisson/Gamma(α,β) conjugate clustering;
    empty clusters are pulled toward `α/β = 1.0` instead of NaN
  - `InvertedReluAdam` — ADAM whose pre-activation gradient is `−ReLU(g) +
    sgd_mix · g`, used to nudge edge weights toward their assigned centroid
  - `DualFloorMirror` — returns `(+x, −x)` clipped to `1 − max(|w|)` so
    freshly minted edges always carry usable signal in both polarities
  - `PropellerRouter` — softmax over weights → axel + blade selection,
    skips existing pairs, joint-probability coupling
  - `touch_couple(a, b) = exp(ln(1+|a|)+ln(1+|b|)) − 1` — exp/ln identity
    coupling (numerically stable at small weights)
  - `vision_horizontal_expand(cn)` — orchestrates the above against
    `corpus_edge` rows whose `rel ∈ {REACHABLE, BRIDGES_TO, SERVES}` and
    inserts new `SYMBIOTIC_TUNNEL` edges
- **`src/quipu/torus_touch.py`** — continuous boundary-pressure agent on
  `T^7 = (S^1)^7`:
  - `CatGapField` — per-dim categorical PMF (default 16 bins/dim) with
    Laplace smoothing; KL-from-uniform measures the informational gap
  - `TouchPressure` — momentum + step + jitter, wrapped mod 2π each tick
  - `tick_torus_pressure(cn)` — reads every `Endpoint`, builds the gap
    field, walks each endpoint up `∇G`, persists `torus_angles`,
    `torus_gap`, and per-endpoint velocity in `kv_store`
  - `touch_couple_torus(θ_a, θ_b)` — wrap-aware angular Touch
  - `endpoint_angles()`, `gap_field_summary()` helpers
- **`src/quipu/synaptic_workers.py`** — registered `_torus_touch_worker` as a
  30-second daemon thread alongside the existing five workers; added
  `synapse_torus_last` heartbeat (`endpoints | moved | gap | spread%`) and
  `_vision_worker` Step 4 calls `vision_horizontal_expand` after each
  bridge/network probe pass
- **`tests/test_symbiotic_torus.py`** — 29 unit tests covering primitives,
  horizontal expansion, manifold geometry, DB-driven ticks, and cross-module
  manifold-aware coupling

### Closed-loop architecture
```
torus_touch (30 s)            vision_horizontal_expand (5 min)
─────────────────             ────────────────────────────────
read Endpoints                read Endpoints + corpus_edge
build CAT pmf                 cluster weights via Bayesian/Poisson centroids
∇G gap field                  propeller route over top-tier
push θ_i along ∇G ──► writes  ─► touch_couple_torus(θ_a, θ_b) ◄── consumes T^7
torus_angles into             write SYMBIOTIC_TUNNEL edges weighted by
corpus_entity.props_json      manifold proximity, not just scalar weight
```

### Test Results (2026-04-24)
```
tests/test_symbiotic_torus.py ......................... 29/29 PASS
  TestPrimitives                  11/11
  TestHorizontalExpansion          4/4
  TestTorusGeometry                9/9
  TestTorusTick                    4/4
  TestTunnelManifoldCoupling       1/1
```

---

## 1.4.1 — DBI Playwright Suite · LLM timeout · Procurement 360 expanders (2026-04-23)

### Added
- **`tests/playwright/test_dbi_tooltip.py`** — 19-page Playwright E2E suite for the Dynamic Brain Insight (DBI) widget:
  - `_wait_for_server_stable()`: waits up to 60 s for Streamlit `stAppViewContainer` before running tests; prevents false failures from slow cold starts
  - `_check_popover` `src` retry loop: 4 × 1.6 s attempts to locate "Insight source" text; re-locates the trigger button on each retry to survive `@st.fragment(run_every=2)` DOM rebuilds (stale-reference fix)
  - `_check_help_tooltips` stExpander ancestor walk: 8-level DOM traversal to correctly classify metrics inside `st.expander` blocks

### Fixed
- **`tests/playwright/test_dbi_tooltip.py`**: `_check_popover` trigger locator replaced lambda pattern with `.filter(has_text=…)` to avoid stale closures
- **`tests/playwright/test_dbi_tooltip.py`**: `passed` property `expanders_ok` guard — pages with zero metrics now pass without requiring expanders
- **`tests/playwright/test_dbi_tooltip.py`**: `wait_for_function` timeout increased to 20 000 ms; `wait_for_selector` for dbi-card to 25 000 ms
- **`src/quipu/llm_caller_openrouter.py`**: LLM per-model `timeout` reduced 40 s → 7 s; worst-case with 2-model fallback = 15 s < 20 s test window
- **`pages/4_Procurement_360.py`**: Restructured all 7 KPI metrics into inline `st.expander` blocks (5 in main KPI strip + 2 in obsolescence tab) so DBI expander check returns `expanders=7/7`

### Test Results (run 2026-04-23, fresh server PID 26756)
```
11/19 PASS
  PASS: Query Console, Schema Discovery, Supply Chain Brain (5/5 expanders),
        Supply Chain Pipeline (2/2), Connectors, Lead-Time Survival (4/4),
        Multi-Echelon (4/4), Sustainability (4/4), What-If, Decision Log (4/4),
        Benchmarks (5/5)
  FAIL (Azure SQL offline — expected): EOQ Deviation, OTD Recursive, Bullwhip Effect
  FAIL (stale DOM, re-locate fix applied): Procurement 360, Report Creator
  FAIL (LLM timeout >20 s): Data Quality, Freight Portfolio, Cycle Count Accuracy
```

### Infrastructure Notes
- Kill orphaned `chrome-headless-shell` processes before each run: `Get-Process -Name "chrome-headless-shell" | Stop-Process -Force`
- Restart Streamlit server between test runs to prevent memory bloat (276 MB → 1 GB after 5+ runs)

---

## 0.15.0 — 4-ERP xlsx Pipeline · Brain Page Fixes · EOQ Optimisation (2026-04-23)

### Added
- **`src/extract/xlsx_extractor.py`** — OneDrive-based live data pipeline for all four ERP systems without requiring SQL credentials:
  - 16 registered aliases across Epicor 9, Oracle Fusion, SyteLine (SiteP), and Microsoft Dynamics AX (MetroB Airport Rd)
  - Canonical column names (`part_number`, `warehouse_code`, `frozen_qty`, `count_qty`, `abc_class`, etc.) normalised across all ERPs
  - `fetch(alias)`, `fetch_all_cc_data()`, `fetch_all_abc_data()`, `available_aliases()` public API
  - Path override via `ONEDRIVE_ROOT` env var
  - Real row counts verified: Epicor CCMerger 14,562 · Oracle on-hand 130 · SyteLine item count 44 · AX CC journal 65
- **`src/connections/ax.py`** — Microsoft Dynamics AX connector for MetroB Airport Rd (AX 2012, `MicrosoftDynamicsAX` database), following the same pattern as `epicor.py` and `syteline.py`
- **`data_access.py`**: `fetch_xlsx_source(alias)` and `fetch_xlsx_all_cc()` wired into the Brain’s session-cached data layer
- **`brain.yaml`**: `xlsx_sources:` section mapping all 16 sheet aliases; AX staging table entries added
- **`test_connector_assumptions.py` Group 8**: 11 live xlsx tests against real OneDrive files — all pass (61 PASS / 0 FAIL / 10 WARN)

### Fixed
- **`1_Supply_Chain_Brain.py`**: `_build_graph()` switched from `@st.cache_data` to `@st.cache_resource` — `GraphContext` (NetworkX graph) is not pickle-serialisable so `cache_data` raised `UnserializableReturnValueError`
- **`1_Supply_Chain_Brain.py`**: Connector status bar removed from the Brain page; it now lives exclusively in the Connectors page
- **`6_Connectors.py`**: Status summary row added above the expanders; shows 🟢 green for connectors with an active handle, 🟡 yellow for unconfigured ones
- **`connections.yaml`**: SyteLine SiteP database corrected from `PFI_App` → `PFI_SLMiscApps_DB`; `schema: cycle_count` added
- **`connections.yaml`**: `ax_airport_rd` block added (`MicrosoftDynamicsAX`, `ActiveDirectoryIntegrated`)
- **`ax.py`**: Removed broken `from . import load_connections_config, DPAPIVault` import; replaced with `yaml.safe_load` + `from . import secrets as _secrets` matching the epicor.py pattern

### Improved
- **`2_EOQ_Deviation.py`**: Column schema resolution cached via `@st.cache_data(ttl=1800)` — eliminates ~5 `INFORMATION_SCHEMA` round-trips per page load
- **EOQ query**: `TOP 5000` → `TOP 2000`; `OPTION (RECOMPILE, MAXDOP 4)` added for better query plan; timeout raised from 120 s → 300 s
- **`db_registry.py`**: AX connector registered; SyteLine description updated to reflect correct database name
- **`mappings.yaml`**: Verified 28 entries (9 Epicor · 5 SyteLine · 14 Azure/Oracle)

### Test Results
```
PASS: 61  WARN: 10 (expected — servers not configured)  FAIL: 0
All .py files outside .venv compile clean
```


## 0.14.9 — Network Vision Worker + OCW Semantic Bridge + Synaptic Worker Protection (2026-04-23)

### Added
- **`_vision_worker` — Network Vision** (`src/quipu/synaptic_workers.py`)  
  Fifth synaptic thread (interval 5 min) that gives the Brain eyes over its own
  compute/network topology:
  - `bridge_rdp.probe_all()` — TCP-probes every declared bridge target (RDP,
    SQL-server, VSCode tunnel) and records live/down status.
  - `network_learner.observe_network_round()` — full endpoint observation round
    across connections.yaml, brain.yaml, SMB mappings, compute peers, and seeds.
  - Materialises observations as `Endpoint` corpus entities with `REACHABLE` /
    `UNREACHABLE` edges to linked `Site` entities, `SERVES` edges to `Peer`
    entities, and `BRIDGES_TO` edges when a piggyback RDP route is alive.
  - All network errors treated as soft skips (no backoff accumulation).

- **`_ingest_bridge_observations`** (`src/quipu/knowledge_corpus.py`)  
  Corpus refresh now promotes every `network_topology` row and every
  `bridge_rdp` target into the corpus graph on each 30-min convergence cycle —
  so network vision is persistent across restarts, not just in-memory.

- **OCW → Task/Quest semantic bridges** (`temp_correct_bridge.py`, run once)  
  All 13 `AcademicTopic` entities now have `INFORMS` edges to the two `Quest`
  entities and curated `Task` entities (`abc_classify`, `otd_classify`,
  `vendor_consolidation`, etc.).  35 SC-relevant `OCWCourse` entities now have
  `INFORMS` edges to `Task` and `Quest` hubs, enabling the RAG deepdive to
  find structural holes that cross the academic/operational divide.

### Changed
- **Synaptic workers protected from autonomous-agent rewrites**  
  All 5 synaptic worker functions moved to `src/quipu/synaptic_workers.py`.
  `autonomous_agent.py` imports them via a try/except fallback stub so
  autonomous LLM rewrites of `autonomous_agent.py` cannot strip the workers.

- **Sweeper treats network errors as soft skips**  
  `_dispersed_sweeper_worker` now uses `_is_network_error(exc)` to detect
  host-DOWN / timeout conditions and sets `ok=True` — preventing exponential
  backoff from accumulating when `desktop-sql` (10.0.0.10) is unreachable.

- **`synaptic_agents_status()`** updated to include `synapse-vision` heartbeat
  with 300 s expected interval.



### Added
- **Global Timeline Windows (`global_filters.py`)** — Start/End date lookbacks now reliably filter dashboards on the SQL side using YYYYMMDD integer `date_key` constructs (`CAST(receipt_date_key AS bigint) BETWEEN {sk} AND {ek}`).
- **Local Persistence (`local_store.py`)** — Added a local SQLite database (`local_brain.sqlite`) for storing state independent of the Azure Replica. Support added for action bookmarks, NLP part categories, and manual OTD workflow comments/owners.
- **NLP Semantic Categorization (`nlp_categorize.py`)** — Parts are now bucketed into taxonomic categories (e.g. Steel, Fasteners, Wiring, Hydraulics) dynamically using a scikit-learn TF-IDF / cosine_similarity model falling back to heuristic keyword-matching.
- **Action Evaluation Engine (`actions.py`)** — Academic outputs are converted into layperson tasks via a deterministic Friction-to-Action semantic mapping that computes Annual Impact ($ / yr), Prioritization, Confidence metrics, and Action Owners.
- **Brain Expert TODO List** — `1_Supply_Chain_Brain.py` now leverages `actions_for_pipeline` to load a unified list of pipeline tasks sorted by monetary value per year.
- **Intercompany Inventory Transfer Scan** — `4_Procurement_360.py` now cross-references obsolete list parts with global network-wide `on_hand` metrics to locate viable transfer sites.
- **Executive ESG ROI Panel** — `10_Sustainability.py` now includes a net-present 5-Year ROI evaluation per abatement lever (mode-shifts, LTL/FTL).
- **Interactive Daily Plant Review** — `3_OTD_Recursive.py` integrates directly into the local SQLite store allowing analysts to review rows "Opened Yesterday", claim assignment, and drop updates manually via Streamlit's `data_editor`.

### Fixed
- **Multi-Echelon Decimal/Float TypeErrors** — Enforced complete `float` casting during safety-stock calculations preventing decimal schema type collisions.
- **Bullwhip Query Timeouts** — All 3 primary CTEs (`demand`, `mfg`, `supplier`) dynamically bound to the global 365-day timeline window by default, resolving arbitrary lockups on `fact_po_receipt`.
- **Goldfish Lane Exclusions** — Repaired Freight Portfolio SQL logic matching correct unit-price schemas. OD (Origin→Destination) pairs now display via normalized `get_supplier_labels` mappings.
- **Cross-Page Findings Mapping** — Refined Report Creator and Overview UI explanations for index labels (part, cluster, supplier, lane, node, vendor).

## 0.7.1 — Ask the Data & Cross-Dataset Reports

### Fixed
- Stabilized Oracle connection pooling across Streamlit app states during extensive cross-dataset AI reporting.

## 0.6.0 — Global Filter & Deck Creator

### Added
- **Global Application Filter**: Implemented a global "Mfg Site" dropdown in pp.py that syncs state across all dashboards via st.session_state["g_site"]. Removed hardcoded/local filters from individual pages to streamline unified navigation across the entire toolkit.
- **PowerPoint & Reports Manager**: Added 15_Report_Creator.py to auto-generate PowerPoint Cross-Dataset performance reviews directly from the UI.
- **Presentation Template Auto-Scrubber**: Allow users to upload .pptx (corporate slide masters, slide decks) which the pipeline visually "scrubs" empty using python-pptx, securely retaining localized styles/fonts without carrying over extraneous information, creating logic hooks to populate new reviews natively.

## 0.5.0 — Value Stream Living Map

### Added
- **Value Stream Pipeline**: Replaced generic graph on \Pages/1_Supply_Chain_Brain.py\ with an interactive Value Stream Map.
- **Formulaic Friction Points**: Added integrated bottleneck algorithms based on MIT SCALE principles, calculating friction dynamically using \due_date_key\ tracking for POs/WOs, and \promised_ship_day_key\ for SOs.
- **Enhanced Topology Filtering**: Added specific MIT Design Lab UI filters for Production Plant (Business Unit) and Value Stream (Part Types), pushing filter complexity upstream and using non-linear marker scaling.
- **Function & Schema Intersection Guide** (`docs/REPO_FUNCTION_AND_SCHEMA_GUIDE.md`) — end-to-end reference mapping every brain module, MIT CTL research module, Streamlit page and `src/deck/` PPTX builder to the underlying replica tables/columns. Documents the four confirmed schema gaps (`failure_reason`, `fact_cycle_count`, point-in-time inventory, ABC part codes on `dim_part`) that surface as empty/Unknown slides in the agent-generated PowerPoint.
- **ABC Inventory Catalog "D" Candidates Fallback**: Updated the `src/deck/live.py` SQL generation to strictly respect the existing `ABC Inventory Catalog` codes (which are locked at the beginning of the year). The live query now intelligently identifies D-Code candidates by outputting "D" only when the existing classification is null *and* there is active `quantity_on_hand` present.

## 0.4.6 — Unified Database Explorer

### Added
- **Unified Database Explorer** (`pages/0_Schema_Discovery.py`) — a dynamic dropdown interface that queries all registered database connectors on the platform (Azure SQL, Oracle Fusion) to let users independently browse any schema, subject area, and table.
- **Automated Schema Reviews** — schema UI dynamically parses contextual notes, table grains, definitions, and usage dependencies from `DATA_DICTIONARY.md` and `DWH_DASHBOARD_TABLES.md` directly into the app view when inspecting a table.

---

## 0.4.5 — YYYYMMDD date fix · session-cache SQL · graph label enrichment

### Fixed

- **YYYYMMDD integer-date conversion** — all MIT CTL research pages (7–11) and
  the EOQ page now convert fact-table integer date keys with
  `TRY_CONVERT(date, CONVERT(varchar(8), CAST([col] AS bigint)), 112)`.
  The previous `TRY_CONVERT(date, [col])` silently returned NULL for integer
  inputs, producing zero-row results on every page.

- **HYT00 query-timeout cascade eliminated** — every `_build_xxx_sql()`
  function was being called at module load time on each Streamlit rerun,
  firing 2–5 `INFORMATION_SCHEMA` discovery queries before the actual data
  query. All SQL builders are now lazily evaluated and cached in
  `st.session_state` (keys: `_eoq_default_sql`, `_lt_sql`, `_bw_sql`,
  `_me_sql`, `_sus_sql`, `_port_sql`). SQL is built at most once per
  browser session.

- **`9_Multi_Echelon.py` — orphaned code after `return`** — two unreachable
  `st.text_area` / `st.file_uploader` lines were left floating after the
  `return` statement inside `_get_me_sql()`; removed. Reference to undefined
  `default_sql` replaced with `_get_me_sql()` call.

- **`10_Sustainability.py` / `11_Freight_Portfolio.py`** — same YYYYMMDD date
  fix applied; absolute fallback SQL added; `_load()` / `_port()` timeout
  raised to 120 s.

- **Graph node labels** (`1_Supply_Chain_Brain.py` + `graph_context.py`) —
  nodes were labelled with raw integer keys (e.g. `221273`) instead of human
  names. Fixed by:
  - `graph_context.add_parts()` now accepts `label_col=` parameter.
  - `graph_context.add_suppliers()` writes `label=` from `name_col`.
  - `graph_context.add_edges()` accepts `src_label_col=` / `dst_label_col=`
    and upgrades implicit node labels from raw key → human name whenever
    a richer label is available.
  - `_build_graph()` in page 1 now calls `enrich_labels()` on all three
    DataFrames and passes resolved `*_label` column names into the graph
    builder.

### Changed

- **Default query timeout raised to 120 s** across `db_registry.read_sql()`,
  `data_access.query_df()`, and `demo_data.auto_load()` (was 30 s).
- **`db_registry._healthy_conn()`** — connection handle is now validated with
  a `SELECT 1` ping before use; stale handles are discarded and reconnected
  automatically without requiring a new MFA prompt.
- **`WITH (NOLOCK)` + `OPTION (MAXDOP 4)`** added to all fact-table reads in
  pages 2, 7, 8, 10, 11 to reduce lock contention and cap parallel workers.

---

## 0.4.4 — Replica-table rewire (vw_* view elimination)

### Fixed
- Pages 7–12 SQL queries rewritten against base replica tables
  (`fact_po_receipt`, `fact_sales_order_line`, `fact_inventory_on_hand`,
  `fact_inventory_open_mfg_orders`). Removed dependency on non-existent
  `vw_*` views that caused immediate connection errors on every page load.

---

## 0.4.3 — Bug-fix wave

### Fixed
- VOI Timestamp / datetime columns converted to `int64` epoch before LightGBM
  fit — eliminates `TypeError: float() argument must be … Timestamp`.
- Graph node-kind propagation restored; discovery panel explains why high-degree
  nodes (e.g. `('part','221273')`) are central.
- EOQ outlier heatmap + quadrant chart added alongside the ranked table.
- Procurement 360 supplier/part fields resolved to human-readable names via
  `label_resolver.enrich_labels()`.
- Benchmarks `rows_per_s` merge collision fixed (suffixed columns deduplicated).
- Sidebar node-type filters now correctly hide/show graph nodes.

---

## 0.4.2 — Full Plotly rewrite

### Changed
- All 14 pages converted from Altair/Vega to Plotly Express for consistent
  drill-down and cross-filter behaviour.
- `page_header` / `drilldown_table` helper retired; each page manages its own
  `st.plotly_chart(use_container_width=True)` layout.
- Data Quality VOI section now renders heatmaps for missing-value impact.
- Connectors page modernised with live ping status and edit-in-place YAML.

---

## 0.4.1 — Self-driving live pages

### Changed
- **Every sidebar page now auto-loads from the live database on first paint.**
  No more "click Run / Compute / Build" gates — pages 1, 2, 3, 5, 7, 8, 9, 10,
  11, 12 all execute their default Azure-SQL-replica queries inside an
  `@st.cache_data(ttl=600)` loader the moment the page is opened, and the user
  refines via collapsed expanders rather than primary buttons.
- All data is pulled **only** from the registered SQL connectors (Azure SQL
  replica + Oracle Fusion). Synthetic data is reserved for `bench_brain` and
  is never used in the UI.

### Added
- `src/quipu/demo_data.py` (now a live-only loader) — `auto_load(sql, connector)`
  + `render_diagnostics()` that, when a live query fails, shows the SQL, the
  error, and an inline **schema browser** (INFORMATION_SCHEMA tables → columns
  → 25-row sample) so the user can see the real shape and fix
  `config/brain.yaml` mappings without leaving the page.
- `first_existing_table(connector, candidates)` helper for pages that want to
  probe several physical mappings before failing.

### Fixed
- Sidebar `_safe_page_link` markdown fallback was emitting `/11_Freight_Portfolio`
  style URLs which don't match Streamlit MPA's actual `/Freight_Portfolio`
  routing — every fallback link redirected to the DWH query console root.
  The leading `\d+_` prefix is now stripped from the slug, so the markdown
  fallback works correctly when `st.page_link` itself isn't available.

## 0.4.0 — Phase 4 (platform)

### Added
- `bench/bench_brain.py` — synthetic-data benchmark suite with 18 timings
  covering EOQ, hierarchical EB shrinkage, OTD cleaning, missingness +
  mass-impute, bullwhip, KM/per-group lead-time, GLEC emissions, lane
  volatility & portfolio mix, CVaR Pareto, multi-echelon safety stock,
  graph centrality (degree + eigenvector), and findings-index round-trip.
- `pages/14_Benchmarks.py` — in-app dashboard for the latest run.
- `bench/results/latest.csv` and timestamped historical runs.
- `requirements.pinned.txt` — version-bounded reference set validated
  together on Python 3.14 / Windows.
- `docs/ARCHITECTURE.md`, `docs/RESEARCH.md`, `docs/CONFIG.md`,
  `docs/RUNBOOK.md` — full operational + reference documentation.
- `src/quipu/_version.py` — single source of truth for `__version__`.
- App sidebar now shows `Brain v{__version__}`.

### Fixed
- `brain.graph_backend.NetworkXBackend` was importing a non-existent
  `SCGraph` symbol from `graph_context`; rewritten to wrap an
  `nx.MultiDiGraph` directly so all 25 brain modules import cleanly.
- `bench_brain.py` deprecation warnings (`datetime.utcnow`, `'d'` unit)
  cleared.

## 0.3.0 — Phase 3 (MIT CTL research suite)

### Added
- `src/quipu/research/`:
  - `hierarchical_eoq.py` — empirical-Bayes shrinkage on Poisson rates.
  - `causal_lead_time.py` — `econml` causal forest with permutation-importance
    fallback.
  - `lead_time_survival.py` — KM + Cox PH via `lifelines`, empirical-quantile
    fallback.
  - `bullwhip.py` — Lee/Padmanabhan/Whang variance ratio + heatmap frame.
  - `multi_echelon.py` — Graves-Willems guaranteed-service safety stock.
  - `sustainability.py` — GLEC / ISO 14083 Scope-3 freight emissions.
  - `freight_portfolio.py` — CV-thresholded contract/spot/mini-bid mix
    + goldfish-memory rejection score.
  - `risk_design.py` — Monte-Carlo CVaR + Pareto frontier on supplier
    scenarios.
- `pages/7_Lead_Time_Survival.py`, `8_Bullwhip.py`, `9_Multi_Echelon.py`,
  `10_Sustainability.py`, `11_Freight_Portfolio.py`.
- `ips_freight.ghost_lane_survival()` — gradient-boosted survival on
  contract-vs-actual volume (logistic fallback if `scikit-survival` not
  installed).
- `procurement_360` extended with **CVaR Pareto frontier** + **causal-forest
  lead-time attribution**.
- `drilldown.CITATIONS` — every research page renders a citation footer
  back to its originating MIT CTL lab.

## 0.2.0 — Phase 2 (depth)

### Added
- `src/quipu/graph_backend.py` — pluggable graph backend behind one API
  (NetworkX default; Neo4j and Cosmos Gremlin opt-in).
- LinUCB contextual-bandit ranker so the EOQ table self-reshapes after
  each user resolution.
- OTD recursive page now indexes every cluster path into the findings
  index so other pages can drill through.

## 0.1.0 — Phase 1 (core)

### Added
- `src/quipu/` package skeleton: `db_registry`, `data_access`,
  `schema_introspect`, `cleaning`, `eoq`, `otd_recursive`,
  `graph_context`, `imputation`, `ips_freight`, `findings_index`,
  `drilldown`.
- Six Streamlit pages: 🧠 Brain · 📦 EOQ Deviation · 🚚 OTD Recursive
  · 🏭 Procurement 360 · 🧩 Data Quality · 🔌 Connectors.
- Drill-down + cross-page findings index baked into `app.py`.
- `config/brain.yaml` — single source of truth for connectors, column
  mappings, and analytics defaults.
