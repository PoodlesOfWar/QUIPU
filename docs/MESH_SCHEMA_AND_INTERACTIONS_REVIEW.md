# MESH Schema and Interactions Review

**Date:** 2026-05 (current session)  
**Scope:** `pipeline/` — learning transport (git jsonl), corpus graph, SLM, System/Mesh Entirety, compute grid, network observer, resumption  
**Reviewer:** Grok (pipeline context)  
**Status:** Structural + operational review against live code  
**Fixes Applied (v0.22.96):**  
- Advisory `mesh_upload_critical` lock (O_EXCL + stale steal) protecting the entire upload sequence and direct resuscitation appends (`mesh_lock.py`, `autonomous_agent.py`, `mesh_entirety.py`).  
- `mesh:upload:health` (last_success, consecutive_failures, last_error, last_attempt) persisted in brain_kv and surfaced inside `mesh:entirety:state`.  
- Cheap `cloud_run_id` dedup + supporting index in `ingest_cloud_queue` (catches resus bypasses and normal cross-node entries).  
- See `src/quipu/mesh_lock.py` and the health writer in `autonomous_agent.py: _write_upload_health`.

These directly address the top two "Critical" race conditions identified in the original review.

---

## Executive Summary

The MESH is a **dual-transport distributed learning and compute fabric** for the Supply Chain Architect:

- **Learning / Knowledge MESH**: append-only event sourcing + graph replication over the git repository itself (`cloud_learning_queue.jsonl` + `cloud_corpus_queue.jsonl`).
- **Physical / Compute MESH**: OneDrive-synced rendezvous (`bridge_state/compute_peers/*.json`, `bridge_triggers/`) + direct TCP jobs (HMAC-protected).

The schema is **mostly SQLite ad-hoc** (created on first use with `CREATE TABLE IF NOT EXISTS` + opportunistic `PRAGMA` migrations) plus two unbounded JSONL files that serve as the cross-node broadcast medium. The "schema" lives in the union of:
- `local_brain.sqlite` tables (learning_log, corpus_*, brain_kv, mesh_slm_*)
- JSONL wire format (implicit, versioned only by code)
- `brain_kv` key namespace under `mesh:*` and `resumption:*`

**Dominant risks** (detailed below):
1. **Multi-writer git race on the shared log** (no coordination, push failures swallowed).
2. **Two different cursor schemes** (rowid upload vs line-number download) + direct append bypass in resuscitation.
3. **Unbounded growth + no compaction/rotation** of the jsonl transport.
4. **Hybrid transport latency mismatch** (git vs OneDrive) with no unified observability.
5. **Ad-hoc schema evolution** with no central migration or compatibility layer.

The design is clever (repo-as-mesh, OneDrive-as-rendezvous, toroidal SLM) but operationally brittle for >2-3 nodes or high churn.

---

## 1. Schema Inventory (Canonical)

### 1.1 Core Local Tables (local_brain.sqlite)

| Table                  | Primary Key / Identity                  | Key Columns                                      | Notes / Writers |
|------------------------|-----------------------------------------|--------------------------------------------------|-----------------|
| `learning_log`        | `id INTEGER PRIMARY KEY AUTOINCREMENT` | `logged_at, kind, title, detail(JSON), signal_strength, source_table, source_row_id` | Every agent (SLM, research, forge, self_realization, mesh_resus, etc.). Cursor: `mesh:upload_cursor` (rowid) |
| `corpus_entity`       | `(entity_id, entity_type)`             | `label, props_json, first_seen, last_seen, samples` | `system_entirety`, `asset_resource_mesh`, `tool_forge`, `knowledge_corpus`, etc. Cursor: `mesh:entity_upload_cursor` (rowid) |
| `corpus_edge`         | `(src_id,src_type,dst_id,dst_type,rel)` | `weight, last_seen, samples`                     | Same as entity. Averaging upsert on weight. Cursor: `mesh:edge_upload_cursor` |
| `brain_kv`            | `key TEXT PRIMARY KEY`                 | `value(JSON or scalar), updated_at`              | **All control plane**. 30+ `mesh:*` keys documented below. |
| `kv_store`            | `key`                                  | `value`                                          | Legacy / some RAG use |
| `mesh_slm_vocab`      | `token_id`                             | `token, i, j, freq` (torus coords)               | `mesh_slm.py` |
| `mesh_slm_embed`      | `token_id`                             | `e_vision … e_entirety` (7 floats)               | 7-D mesh-state embedding |
| `mesh_slm_quipu`      | `(src, dst)`                           | `weight, samples` (bigrams)                      | Hebbian |
| `mesh_slm_meta`       | `key`                                  | `value` (rounds, loss, …)                        | Training metadata |
| `flip_log_*` (various)| auto                                   | `ran_at, observer, …`                            | Bit-flip / Floquet history (system_entirety + mesh_entirety) |
| `network_observations`| composite                              | host, transport, ok_ratio, last_seen, …          | `network_observer.py` |
| `llm_dispatch_log`    | auto                                   | Used for SLM training corpus                     | `llm_ensemble` etc. |

**Migration style**: Every `_ensure_*` does `CREATE IF NOT EXISTS` + `PRAGMA table_info` + `ALTER ADD COLUMN` (no version table, no transactional migration script).

### 1.2 Transport Files (git root, committed)

- `cloud_learning_queue.jsonl` — append-only NDJSON. Fields: `logged_at, kind, title, detail, signal, cloud_run_id` (the latter carries `local:<host>:<local_id>` or `mesh_resus:...`).
- `cloud_corpus_queue.jsonl` — same, with `kind: "corpus_entity"|"corpus_edge"`, full row payload + `src_host`.

These are the **single source of truth** for cross-node learning propagation. They are never rotated or truncated in the reviewed code.

### 1.3 brain_kv Mesh Namespace (partial, live keys observed)

**Upload / Download cursors**
- `mesh:upload_cursor`
- `mesh:entity_upload_cursor`
- `mesh:edge_upload_cursor`
- `resumption:cloud_queue_cursor` (line # in jsonl)
- `resumption:cloud_corpus_cursor` (line #)

**Mesh Entirety (8th axis)**
- `mesh:entirety:observer`, `synchrony`, `bit_state`, `flip_count`, `expansion_phase`, `node_count`, `state`, `ran_at`, `resus:last_report`

**Resuscitation / Resilience**
- `mesh:resus:<host>:last_at`, `mesh:resus:<host>:attempts`, `mesh:resus:alarm:<host>`

**Disk / Runtime sentinel**
- `mesh:disk_pressure_level`, `mesh:wal_size_gb`, `mesh:disk_free_gb`, `mesh:heart_stale_seconds`, `mesh:sentinel_version`, `mesh:runtime_version`, `mesh:runtime_build_date`

**Other important**
- `resumption:*` family (last_alive, catchup_burst, graceful_shutdown, …)
- `entirety:physical_realization`, `entirety:state`
- `torus_amplify:*` etc.

---

## 2. Core Interaction Flows

### 2.1 Learning Upload (local → shared log)
`upload_learning_rectification()` (autonomous_agent.py:1338)
1. Read rowid cursors from `brain_kv`.
2. Drain `learning_log`, `corpus_entity`, `corpus_edge` in batches (5k/10k/25k).
3. Append NDJSON to the two `cloud_*.jsonl` files.
4. Advance cursors, commit.
5. `git add` only the two queue files → `git commit` (tolerates nothing-to-commit) → `git push origin HEAD`.
6. Failures → only `WARNING` log; upload count still returned.

**Called from**: continuous `mesh-sync-daemon` (every N s), SLM heartbeat path, and manually.

### 2.2 Learning Download (shared log → local)
`download_learning_rectification()` (autonomous_agent:1535)
1. `git_pull_latest()` (resolves upstream dynamically).
2. `ingest_cloud_queue()` + `ingest_cloud_corpus_queue()` (resumption_manager).
3. Line-number cursors (`resumption:cloud_*_cursor`) decide "new lines".
4. `INSERT OR IGNORE` into `learning_log`; entity upsert + edge weight-average into corpus tables.
5. After successful ingest in daemon: `train_round()` on the SLM (post-download).

### 2.3 SLM Lifecycle
- `register()` → patches `llm_ensemble.set_caller` + `install_as_local_executor` (compute_grid patch).
- `train_round()` runs after every meaningful download + on 30-min heartbeat + after `refresh_corpus`.
- Training data: recent corpus_entity/edge labels + dispatch responses + (implicitly) learning_log titles.
- 7-D embedding conditioned on live `_mesh_state_7d()` (vision/touch/…/entirety from system_entirety + mesh_entirety).
- Falls back gracefully via `MeshSLMUnavailable`.

### 2.4 Mesh Entirety (observer-of-observers)
`oscillating_mesh_step()` / `mesh_observer_tangent()` (mesh_entirety.py)
- Stratified strided scan (10 × 3 MB segments + 8 MB tail) over the multi-GB `cloud_learning_queue.jsonl` to surface *all* nodes, not just the loudest recent writer.
- Combines per-node (volume, mean signal, recency 36 h half-life, network_observations ok-ratio) into coherence vector.
- Produces Ω (mesh tangent), synchrony, bit-flip parity (reuses local Floquet).
- Persists full `mesh:entirety:state` + individual keys.
- Runs `resuscitate_stale_nodes()` on the same clock (warm/cold/dead escalation, including direct append of `mesh_node_dead_resuscitation` events).

### 2.5 Physical/Compute Side (separate transport)
- `compute_grid.discover_peers()` reads `bridge_state/compute_peers/<host>.json` (OneDrive).
- Triggers via drop files in `bridge_triggers/`.
- Job execution: TCP 8000 with `SCBRAIN_GRID_SECRET` HMAC.
- `network_observer` also absorbs OneDrive `.mesh-state/nodes/*.json` (powershell heartbeat schema) and writes `network_observations`.
- `asset_resource_mesh.tick_asset_resource_mesh()` materializes peers + cores/ram/vram as first-class `ComputePeer` / `AssetResource` nodes in the *same* corpus graph used by learning MESH.

### 2.6 Daemon / Continuous Loops (autonomous_agent)
- `start_mesh_sync_daemon()` — upload + download on timer (default 300 s).
- SLM training daemon hooks.
- `start_slm_training_daemon()` emits heartbeats as `mesh_slm_train` learning_log rows so peers can see training progress.

---

## 3. Issues & Findings

### Critical (Race / Correctness)

**Issue 1 — Multi-writer append log over git has no coordination**  
Severity: bug  
- `upload_learning_rectification` does bare `git commit` + `push` while other autonomous loops, the optimization path (line 1299), and manual ops can also touch the working tree or the same two jsonl files.  
- Concurrent pushes from two physical nodes will 99% collide; code only logs warning and continues (data may be lost or duplicated in history).  
- Direct append in `_resuscitate_dead` (mesh_entirety:666) writes the jsonl **without** going through the upload cursor or the git-commit block. The next upload cycle may see the file dirty or the commit may include "someone else's" resuscitation line under a generic message.  
- File: `autonomous_agent.py:1504-1526`, `mesh_entirety.py:646-670`

**Issue 2 — Cursor model is fragile across restarts and mixed writers**  
Severity: bug  
- Upload cursors are *local rowid* (host-specific).  
- Download cursors are *global line numbers* in the jsonl.  
- If a node force-pushes, rebases, or edits the jsonl, all downstream line cursors become wrong (they will either re-ingest or skip forever).  
- `INSERT OR IGNORE` on learning_log helps a little (no PK on natural key), but `cloud_run_id` is only advisory.  
- No per-host watermark or content-hash dedup.  
- File: `resumption_manager.py:258` (line cursor), `autonomous_agent:1369` (rowid)

**Issue 3 — No transactional boundary around "append + advance cursor + commit"**  
Severity: bug  
- Between advancing the rowid cursor in brain_kv and the subsequent git commit, a crash leaves the local DB thinking the rows were uploaded while they were not (or vice-versa). Next upload skips them permanently.  
- The jsonl append itself is not atomic with the cursor write.

### High (Reliability / Growth)

**Issue 4 — Unbounded jsonl growth with only sampling-based readers**  
Severity: bug (operational)  
- Files mentioned as "multi-GB". Stratified reader caps at 600 k lines for entirety math.  
- No rotation, truncation, or compaction policy. A node that has been up for months will eventually choke on disk or on `read_text().splitlines()` in ingest (resumption_manager:260).  
- Related: `disk_sentinel` monitors WAL and free space but does not act on the jsonl files themselves.

**Issue 5 — Git push failures are non-fatal warnings only**  
Severity: bug  
- The entire learning propagation can silently stall for a node if its PAT/SSH key expires, branch protection rejects the commit message, or rate limits hit. No alert, no fallback transport, no "queue health" metric surfaced to Mesh Entirety.

### Medium (Schema / Maintainability)

**Issue 6 — Schema is distributed and versionless**  
Severity: suggestion  
- 15+ locations create/alter the same tables (`system_entirety.py`, `resumption_manager.py`, `doc_rag.py`, `self_realization_loop.py`, tests, …).  
- No `schema_version` row in brain_kv or a `_migrations` table.  
- Ad-hoc `PRAGMA` + ALTER means a partially-upgraded node can have different column sets than its peers, yet still exchange data via jsonl (which carries whatever the writer had).

**Issue 7 — learning_log has no natural unique key**  
Severity: suggestion  
- Relies on AUTOINCREMENT id + `INSERT OR IGNORE` (which does nothing useful without a conflicting unique constraint). Duplicate cloud entries can accumulate if cursors are reset.

**Issue 8 — JSON detail column is untyped text everywhere**  
Severity: nit  
- No CHECK constraint, no JSON1 validation on insert. Typos in kind or structure only discovered at read time by ad-hoc `.get()` chains.

### Lower / Observations

- SLM training after every download is aggressive but rate-limited + locked; good.
- Direct TCP grid has HMAC (good), but secret distribution is out-of-band (env var).
- OneDrive transport for compute state has different failure modes (throttling 429 on devtunnel mentioned in hideout docs) than git.
- Resuscitation writes resilience events with `INSERT OR IGNORE` but no unique constraint on the dedup key it tries to use elsewhere.
- `git_pull_latest` dynamically resolves `@{u}` — nice, but still assumes every node pushes to the same primary branch.

---

## 4. Recommendations (prioritized)

1. **Short-term (stability)**  
   - Add a coarse file lock (or use `git` lock files + `flock` equivalent) around the entire upload sequence (read cursors → append → advance → git add/commit/push).  
   - Make resuscitation append go through a "bypass upload" helper that also does the git dance (or at least stages + best-effort commit) so the line is attributed and not lost.  
   - Add a `mesh:upload_health` / last_success timestamp in brain_kv and surface it in Mesh Entirety state.

2. **Medium-term (correctness)**  
   - Introduce a content-addressed or `(cloud_run_id, kind, hash)` dedup key on `learning_log` and make the ingest side use it.  
   - Add a small header or first line to the jsonl files with a format version + rotation epoch.  
   - Bound the jsonl (e.g., rotate at 500 MB or 30 days, keep manifest of active segments). Update the stratified reader and the two ingest functions to handle segmented logs.

3. **Longer-term (architecture)**  
   - Consider extracting a tiny "mesh transport" abstraction so learning events can flow over git *or* a future NATS/Redis Streams/ S3 + notification without changing every writer.  
   - Central `ensure_mesh_schema(cn)` in one place (local_store.py?) called at startup by autonomous_agent and all daemons.  
   - Add `mesh_schema_version` and a documented migration path.

4. **Observability quick wins**  
   - Expose `mesh:upload_cursor` deltas, last push success, current jsonl byte size, and ingest lag (line cursor vs file length) into the Mesh Entirety snapshot and/or a `/mesh_status` page.  
   - The existing `_check_mesh_cursors.py` and `_mesh_catchup.py` scripts are good; wire them into the daily review or heart process.

---

## 5. Positive Notes

- The stratified scan in `mesh_entirety.py` is a pragmatic and correct solution to the "loudest writer wins" problem in a shared log.
- The 7-D conditioning of the SLM on the live mesh state (including the 8th-axis observer) is elegant and true to the "toroidal quipu" vision.
- Graceful degradation everywhere (SLM unavailable → grid → OpenRouter → echo) is well engineered.
- Using the repo itself as the learning broadcast medium is minimalist and leverages existing CI + git history for free auditability.
- Resuscitation + bit-flip + entirety observer together form a surprisingly complete self-healing + self-observing substrate for a small fleet.

---

## Files Most Relevant to This Review (for follow-up)

- `autonomous_agent.py` (mesh daemons, upload/download)
- `src/quipu/resumption_manager.py` (ingest, git pull, downtime)
- `src/quipu/mesh_entirety.py` (Ω, resuscitation, direct queue append)
- `src/quipu/mesh_slm.py` (register, train, embed)
- `src/quipu/system_entirety.py` (_ensure_mesh_overlay_tables + fusion math)
- `src/quipu/asset_resource_mesh.py` + `compute_grid.py` + `network_observer.py`
- `src/quipu/local_store.py` (WAL + part of schema)
- `docs/HIDEOUT_MESH_INTERACTION_MAP.md`, `MESH_LEARNING_OVERLAY.md`

---

**End of review.** No changes were made to source. This document is the deliverable.

To action: run the existing mesh cursor checker + entirety report after any proposed fixes, then re-execute a full autonomous cycle on at least two nodes and verify cross-ingestion + SLM training + observer Ω > 0.
