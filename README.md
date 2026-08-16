# QUIPU

QUIPU is the extracted, self-contained continuation of the **Supply Chain Brain** model core: the MESH-SLM-SCM-GLM-GNN generational models, the Quipu knotted-memory minimal model, the GARD-shard encryption-compression protocol, and the learning-dynamics machinery (System Entirety, plasticity, rADAM, realization gating) that trains and governs them. It carries the full documentation, version lineage, and learnings of the parent system at **v0.24.1**, with none of the application bloat — no ERP connectors, no Streamlit UI, no network-bridge tooling, no operational data.

All employer-specific identifiers from the parent project were redacted at extraction time (`Contoso`/`Fabrikam`/`SiteX` placeholders, `10.0.0.x` addresses, `DESKTOP-01`-style hostnames). The lineage and mechanics are intact; the private context is not.

## Layout

```
src/quipu/          Python model core (package: src.quipu)
  mesh_slm.py             MESH SLM - online token-level learner over SQLite
  quipu_minimal.py        Quipu minimal knotted-memory model (builds on mesh_slm)
  ueqgm_engine.py         Bounded scoring/weighting helpers - Si/Ci phase term, cosine
                          similarity, graph-density ratio, coverage scoring (see its
                          module docstring: these are heuristics, not physics)
  system_entirety.py      System Entirety self-model (bit-flip parity, certainty axes)
  mesh_entirety.py        Mesh-level entirety aggregation
  asset_resource_mesh.py  Physical realization - compute peers/assets as graph substrate
  gard_shard_model.py     gard-shard/v2 - canonical JSON > zlib > AES-256-GCM (AEAD),
                          per-shard HKDF keys, fail-closed verification, selftest CLI;
                          still reads legacy gard-shard/v1 (AES-256-CBC + HMAC-SHA256)
  neural_plasticity.py    Plasticity dynamics
  self_realization_loop.py  Realization loop (v0.24.1 sigmoidal eligibility lineage)
  radam_optimizer.py      rADAM optimizer
  recurrent_depth.py      Recurrent depth control
  recursive_strengthening.py  Recursive strengthening pass
  temporal_spatiality.py  Temporal-spatial (Weyl tensor) state
  llada_signbit_children.py  LLaDA2 sign-bit child acquisition
  local_store.py          SQLite local store (WAL) shared by all modules
  brain_kv.py             Canonical key/value persistence (Weyl tensor, ACRE state, ...)
  corpus_ingest.py        The Well - stream + holographically compress open corpora
  tool_forge.py           Ring 5 - digital tool synthesis from mesh corpus clusters
  expert_orchestrator.py  Ring 5 - ACRE-routed dispatch, closes the mesh feedback loop
  systemic_refinement_agent.py  Ring 5 strategy runner (ACRE emergence + forge_round)
  daily_digest.py         "What has the System Entirety learned today?" report
  _version.py             Canonical version + full PHASES history

Show-TodaysLearning.ps1    Windows launcher for daily_digest.py

julia/              Julia protocol peers and generational models (Project.toml included)
  mesh_slm_scm_glm_gnn_model.jl   Unified MESH-SLM-SCM-GLM-GNN model (latest generation)
  mesh_slm_glm_gnn_model.jl       Prior generation (kept for lineage)
  mesh_inference_model.jl         Inference model
  mesh_compression_model.jl       Compression model incl. pack_weyl/unpack_weyl (20-byte
                                  v2.2 float-dimension brain_kv record, 5 x Float32 LE)
  scm_predictor.jl                SCM predictor
  gard_shard_model.jl             GARD-shard protocol peer (JSON3/CodecZlib/Nettle)

tests/              Focused pytest suites for every core module
bench/              bench_mesh_slm_vs_peers.py benchmark harness
docs/               Architecture, dynamics, schema, research, and changelog lineage
```

## Quickstart

```
pip install -r requirements.txt
pytest                          # core suites; all local, no network, no live DB
python -m src.quipu.gard_shard_model selftest
julia --project=julia julia/gard_shard_model.jl   # Julia peer selftest
```

## Corpus ingestion — The Well (optional)

`src/quipu/corpus_ingest.py` streams the openly-published corpora that pretrain
current SOTA models (FineWeb, C4, Wikipedia, Dolma, arXiv, Gutenberg, …) and
folds them into the mesh using QUIPU's **boundary distillation**: each cycle is
distilled to a fixed 5-float record (~50 bytes as JSON, or a 20-byte
`5×Float32 LE` pack that is stored base64-encoded as 28 bytes — byte-compatible
with the Julia `mesh_compression_model` peer), scored by
`mesh_compaction_summary` and `corpus_coverage_score`. The Ψ₀–Ψ₄ slot names are
wire-format labels, not Newman-Penrose scalars. The record persists to
`brain_kv["learnings:weyl_tensor"]` and re-radiates dynamics via
`langevin_sigma_from_weyl`. Boundary distillation, not document reconstruction —
see `docs/CORPUS_INGEST.md`.

```
python -m quipu.corpus_ingest --list
python -m quipu.corpus_ingest --sources arxiv,gutenberg --docs 200   # no extra deps
pip install datasets && python -m quipu.corpus_ingest --sources fineweb,wikipedia --docs 1000
```

## Ring 5 — Refinement -> Toolforge

`tool_forge.py`, `expert_orchestrator.py`, and `systemic_refinement_agent.py`
implement Ring 5 of the System Entirety map: the ring that turns corpus growth
into synthesized capability. This is a clean-room, QUIPU-native adaptation —
the parent's `tool_forge.py` scanned ERP-linked `corpus_entity`/`corpus_edge`
tables and had PHYSICAL/ERP synthesis paths (Astec-specific prompts, an
Oracle/Azure-linked HITL review page) that don't belong in QUIPU. Here:

- **`tool_forge.forge_round()`** scans QUIPU's own Ring 4 tables
  (`mesh_slm_vocab`/`mesh_slm_quipu`) for high-degree token clusters, then gates
  synthesis on **structural resonance** measured on the 52-hertz-whale band
  (`RESONANCE_FLOOR_HZ = 52`). A concept resonates only when it is BOTH woven into
  the shared concept graph (mutual centrality — answered by other well-connected
  concepts, not a lonely caller) AND sitting on the shared analytical wavelength
  rather than the 7th/entirety "Other" axis. This replaces the parent's
  typed-entity eligibility (Concept/Mechanism/MLPaper) that QUIPU lacks, and is
  robust to the embedding collapse that made raw overlap read ~1.0 for everything.
  Concepts below 52 Hz — single-source nouns, fiction on the entirety axis — are
  kept in the mesh and shared through the 8th-D field, but not forged. Survivors
  are scored `novelty x certainty` and auto-implemented (pure stdlib,
  AST-validated, no LLM call). Generated tools land in
  `src/quipu/tools/generated/` (gitignored — regenerated locally).
- **Per-domain axis routing (7 senses).** Each corpus domain accretes on a
  distinct sense-axis, so different knowledge modes become distinct *concentrated*
  directions rather than one broad analytical blob: web→vision, code→touch,
  self-docs (`local_docs`)→smell, wikipedia→body, arxiv→brain, fiction→entirety
  (`mesh_slm._SOURCE_AXIS_MAP`; `train_round` applies `_axis_bias`). The **free
  axes** — those no base specialist occupies (touch, smell) — are where a NOVEL
  emergent specialist can form. `observe_interactions` groups the vocabulary by
  each concept's dominant axis, weights the free-axis modes higher, and observes
  them last (EMA recency), giving a populated free-axis domain (e.g. `local_docs`
  → smell → a "self-model" specialist) the best chance to clear ACRE's
  dominant-eigenvector novelty check. Domains on taken axes reinforce the base
  roster instead. This generalizes the original fiction→entirety rule.
- **Shared understanding of "the Other".** Each refinement cycle,
  `mesh_slm.compute_shared_understanding()` computes the self-state (live MESH),
  the Other-state (collective corpus embedding), the shared-wavelength coherence
  in Hz, the entirety/Other axis strength, and the ACRE interaction axis, and
  publishes it to `brain_kv["entirety:the_other"]`. Every ring reads this one
  conceptualization — the forge resonance gate, the orchestrator, and the daily
  digest — rather than each deriving its own.
- **Resonance-gated resuscitation.** The same signal drives recovery:
  `map_resuscitation_quipu` scales each torus node's revival weight by the
  resonance of the concept occupying it (~0.4x for a lonely caller, up to 1.0x on
  the shared wavelength), so when the mesh rehydrates it revives its high-resonance
  shared concepts first and lets lonely/fiction nodes stay quiet. Opt out with
  `QUIPU_RESUSCITATION_RESONANCE=0` (fails toward the legacy purely-geometric
  weighting).
- **`expert_orchestrator.dispatch()`** routes a query through whichever ACRE
  emergent specialist (`mesh_slm.acre_emerge`) currently resonates, generates
  via `mesh_slm.generate`, feeds the result back with `ingest_expert_trace`,
  and can trigger an immediate bounded forge round.
- **The specialist roster + associative consensus** (`docs/MESH_EXPERT_SKILLS.md`).
  The documented roster spans the sense-axes — `supply_chain_optimizer`,
  `research_specialist`, `mesh_historian`, `robotic_integrations_specialist`,
  `advanced_manufacturing_specialist`, `materials_engineering_specialist`,
  `quantum_physics_specialist`, `complex_systems_specialist` — plus ACRE emergent
  ones. Specialists are *associative*, not solo: `mesh_slm.resonant_specialists(k)`
  returns a weighted consensus set, and `expert_orchestrator.dispatch` answers
  through that consensus, so any one specialist (including a newly emerged mode) is
  **dimensionally relative** — one proportional voice, not a monolith.
- **ACRE emergence, relative and associative.** `observe_interactions()`
  accumulates per-axis interaction directions; a NEW specialist crystallizes only
  for a direction not already covered by the roster and distinct from the uniform
  axis (the 0.60 novelty ceiling). Because the roster now spans the axes, an
  emergent mode (e.g. `emergent_smell_brain`, the self-model specialist born from
  `local_docs`) lands *next to* a documented neighbour (`materials_engineering` on
  smell/body) rather than alone on a free axis — grounding it as one interacting
  member of the mesh instead of a runaway.
- **`systemic_refinement_agent.run_strategy()`** is the scheduled tie between
  Ring 4 and Ring 5: read mesh state, let ACRE attempt emergence, run
  `forge_round`, record the cycle. `corpus_ingest.run_ingest(..., refine_every=1)`
  calls this after every Weyl compression cycle by default, so a live crawl
  actually produces forged tools rather than just growing token/edge counts.

```python
from quipu import tool_forge, systemic_refinement_agent as refinement
refinement.run_strategy(max_tools=2)          # one Ring 5 cycle
tool_forge.load_generated_tools()             # {tool_name: callable}
```

## Daily digest — "what has the System Entirety learned today?"

`daily_digest.py` is a read-only report over QUIPU's own mesh database: new
vs. reinforced vocabulary, Weyl compression cycles and compaction ratio, Ring 5
refinement cycles and forged tools, and which ACRE specialists are new since
before today. All timestamps this codebase writes are UTC, so "today" means
the current UTC calendar day — the report says so explicitly to avoid local
timezone confusion when checking it late at night.

```
python -m src.quipu.daily_digest                  # today, formatted
python -m src.quipu.daily_digest --date 2026-07-19  # a past day
python -m src.quipu.daily_digest --json           # raw JSON
```

Or on Windows: `powershell -File Show-TodaysLearning.ps1`.

## Continuation notes

- `VERSION` and `src/quipu/_version.py` are synchronized at 0.24.1; `_version.py` `PHASES` holds the complete generational history. Continue appending there.
- The parent app's `config/brain.yaml` is replaced by a shim: `src.quipu.load_config()` returns `{}` (module defaults) unless `QUIPU_CONFIG_JSON` points to a JSON file.
- Feature flags behave as in the parent: e.g. `BRAIN_USE_REALIZATION_GATE=1` opts into the v0.24.1 realization gate.
- SQLite state is created locally on first use; no seed data ships with this repo by design.
- `LEARNINGS.md` distills the transferable design learnings; `docs/` carries the full architecture record; `CHANGELOG.md`, `RELEASE_NOTES.md`, and the two autonomous changelogs preserve the complete development lineage.
