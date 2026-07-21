## 2026-07-20 - v0.22.247 Mesh Compression Model: Packed Weyl KV Record
- **ADDED** `pipeline/src/quipu/mesh_compression_model.jl`: `pack_weyl` / `unpack_weyl` serialize the 5-scalar Weyl Ψ tensor to a 20-byte v2.2 float-dimension `brain_kv` record (5 x Float32, explicit little-endian), byte-for-byte compatible with the ContosoHub v2.2 JS plane, alongside the existing 50-byte readable `_meta.weyl` JSON boundary. New `MESH_BRAIN_KV_WEYL_PACKED_BYTES = 20` constant and `Base64` stdlib import.
- **CHANGED** `selftest()`: extended with little-endian canary bytes, a pack/unpack round-trip check, and a golden base64 vector for the new KV record, while leaving the existing reconciled compaction-ratio (`14_967_705`) and compaction-scalar (`0.622934`) assertions untouched — these are asserted verbatim by `pipeline/tests/test_gard_shard_model.py::test_julia_peer_and_corrected_compression_reference_are_declared`.
- **VERSION** `_version.py` bumped `0.22.246 → 0.22.247`; new `PHASES` entry added.
- **DOCS** `RELEASE_NOTES.md` "Unreleased" section (GARD Shard + corrected Julia compression reference) extended with the new pack/unpack capability.

## 2026-05-13 - v0.20.19 Autonomous Resurrection Hardening
- **FIXED** `pipeline/autonomous_agent.py`: process-level heartbeat now refreshes `pipeline/logs/agent_heartbeat.txt` every 60 seconds for the full lifetime of the process, so long active cycles no longer look dead.
- **FIXED** `pipeline/brain_watchdog.ps1`: `Get-Python()` is now PowerShell 5.1-safe, the agent path is quoted correctly under OneDrive paths with spaces, and stdout/stderr are redirected to watchdog log files.
- **ADDED** `pipeline/run_brain_watchdog.cmd`: Task Scheduler wrapper for hidden watchdog startup through `cmd.exe`.
- **CHANGED** `pipeline/install_brain_watchdog.ps1`: scheduled task registration now runs through the wrapper and falls back to repeat-only registration when logon-trigger registration is denied.


## 2026-05-01 — v0.19.25 Repo Hygiene + Credential Migration
- **CREDENTIAL MIGRATION** `pipeline/config/connections.yaml`: added `rdp_fileserver_01` (10.0.0.10), `rdp_desktop` (10.0.0.10), `rdp_laptop` (LAPTOP-01.fabrikam.contoso.local / CONTOSOCORP\AUser / NLA cert `0000000000000000000000000000000000000000 — migrated from deleted `rdp_files/`
- **REMOVED** 22 one-off scripts: `quick_fix{1-5}.py`, 12 root `temp_*.py`, 7 `pipeline/temp_*.py`
- **REMOVED** `Proxy-Pointer-RAG/` subproject from git index (already deleted on disk)
- **REMOVED** `pipeline/pages_archive/` (vestigial, pycache-only)
- **UNTRACKED** `pipeline/abc_screenshots/` (442 PNGs, ~114 MB) + `pipeline/navigation_tests/` (15 files) — added to `.gitignore`, files kept on disk
- **GITIGNORE** `pipeline/my_playwright_profile/` added
- **VERSION** `_version.py` bumped `0.19.19 → 0.19.25`; root `VERSION` synced from stale `0.19.4`

## 2026-04-24 — v0.16.0 Symbiotic Dynamic Tunneling + Torus-Touch (T^7)
- **NEW MODULE** `src/quipu/symbiotic_tunnel.py` (350 LOC): Bayesian-Poisson centroids, inverted-ReLU ADAM, dual-floor mirror, propeller routing → mints `SYMBIOTIC_TUNNEL` edges
- **NEW MODULE** `src/quipu/torus_touch.py` (300 LOC): continuous boundary pressure on n=7 toroidal manifold; constant outward push along categorical-gap gradient
- **NEW WORKER** `_torus_touch_worker` (30 s daemon, registered in `start_continuous_synaptic_agents`); heartbeat key `synapse_torus_last`
- `_vision_worker` Step 4 added: calls `vision_horizontal_expand(cn)` after each bridge/network probe pass; tunnel weights re-scaled by manifold geometry when `torus_angles` are present
- **NEW TESTS** `tests/test_symbiotic_torus.py`: 29/29 PASS (primitives, expansion, geometry, ticks, cross-module coupling)
- Version bumped 0.15.0 → 0.16.0 in `src/quipu/_version.py`

## 2026-04-22 08:11:22
- Autonomous cycle completed. Benchmarks recorded.
- Synced latest data structure schemas into relational dictionary.

## 2026-04-22 23:10:27
- Autonomous cycle completed. Benchmarks recorded.
- Synced latest data structure schemas into relational dictionary.

## 2026-04-23 09:44:04
- Autonomous cycle completed. Benchmarks recorded.
- Synced latest data structure schemas into relational dictionary.

## 2026-04-23 10:48:02
- Autonomous cycle completed. Benchmarks recorded.
- Synced latest data structure schemas into relational dictionary.

## 2026-04-23 11:51:59
- Autonomous cycle completed. Benchmarks recorded.
- Synced latest data structure schemas into relational dictionary.

## 2026-04-23 12:55:24
- Autonomous cycle completed. Benchmarks recorded.
- Synced latest data structure schemas into relational dictionary.

## 2026-04-23 12:00:00 (Manual Agent Upgrade)
- Implemented integrated_skill_acquirer.py for automated web scraping and pip installations over proxied connections.
- Upgraded piggyback_router.py to v0.7.3 to support transparent HTTP CONNECT and SOCKS5 proxy tunnelling for internet-bound API dependencies.
- Added 1080 (SOCKS5) and 3128 (HTTP) firewall port pass-throughs to bridge_watcher.ps1.
- Linked autonomous_agent.py to seamlessly spawn the background skill acquirer alongside Synaptic Workers.


## 2026-04-30 08:20:17
- Corpus state: **51297** entities, **100506** edges.
- New learnings this cycle:
  - ocw_resource: OCW expansion outreach: crawled 3 courses, wrote 0 log rows
  - ocw_resource: OCW deep-fetch: 10-492-1-integrated-chemical-engineering-topics-i-process-control-by-design-fall-2004 → 0 new rows (7 resources, 0 related, 11 external)
  - ocw_resource: OCW deep-fetch: 1-012-introduction-to-civil-engineering-design-spring-2002 → 0 new rows (6 resources, 0 related, 11 external)
  - ocw_resource: OCW deep-fetch: esd-71-engineering-systems-analysis-for-design-fall-2008 → 0 new rows (6 resources, 0 related, 14 external)
  - vision: SCB Vision scan: 3 JSON + 260 asset files in Introduction to SCB
  - schema_vision: Schema vision: affirmed 104 DW tables in corpus
  - mission: Mission m_1777551040_cf814819: refreshed
  - mission: Mission m_1777551040_cf814819: progress
  - mission: Mission m_1777551040_cf814819: artifact_attached
  - mission: Mission m_1777551040_cf814819: artifact_attached
- ToolForge: synthesised → tool_quest_optimize_supply_chains_bench
- Autonomous cycle completed. Benchmarks recorded.
