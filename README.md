# QUIPU

QUIPU is the extracted, self-contained continuation of the **Supply Chain Brain** model core: the MESH-SLM-SCM-GLM-GNN generational models, the Quipu knotted-memory minimal model, the GARD-shard encryption-compression protocol, and the learning-dynamics machinery (System Entirety, plasticity, rADAM, realization gating) that trains and governs them. It carries the full documentation, version lineage, and learnings of the parent system at **v0.24.1**, with none of the application bloat — no ERP connectors, no Streamlit UI, no network-bridge tooling, no operational data.

All employer-specific identifiers from the parent project were redacted at extraction time (`Contoso`/`Fabrikam`/`SiteX` placeholders, `10.0.0.x` addresses, `DESKTOP-01`-style hostnames). The lineage and mechanics are intact; the private context is not.

## Layout

```
src/quipu/          Python model core (package: src.quipu)
  mesh_slm.py             MESH SLM - online token-level learner over SQLite
  quipu_minimal.py        Quipu minimal knotted-memory model (builds on mesh_slm)
  ueqgm_engine.py         UEQGM v0.9.x engine - SiCi axial decay, wavefunction overlap,
                          Floquet modulation, holographic entropy, coherence scoring
  system_entirety.py      System Entirety self-model (bit-flip parity, certainty axes)
  mesh_entirety.py        Mesh-level entirety aggregation
  asset_resource_mesh.py  Physical realization - compute peers/assets as graph substrate
  gard_shard_model.py     gard-shard/v1 - canonical JSON > zlib > AES-256-CBC > HMAC-SHA256,
                          per-shard HKDF keys, fail-closed verification, selftest CLI
  neural_plasticity.py    Plasticity dynamics
  self_realization_loop.py  Realization loop (v0.24.1 sigmoidal eligibility lineage)
  radam_optimizer.py      rADAM optimizer
  recurrent_depth.py      Recurrent depth control
  recursive_strengthening.py  Recursive strengthening pass
  temporal_spatiality.py  Temporal-spatial (Weyl tensor) state
  llada_signbit_children.py  LLaDA2 sign-bit child acquisition
  local_store.py          SQLite local store (WAL) shared by all modules
  _version.py             Canonical version + full PHASES history

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

## Continuation notes

- `VERSION` and `src/quipu/_version.py` are synchronized at 0.24.1; `_version.py` `PHASES` holds the complete generational history. Continue appending there.
- The parent app's `config/brain.yaml` is replaced by a shim: `src.quipu.load_config()` returns `{}` (module defaults) unless `QUIPU_CONFIG_JSON` points to a JSON file.
- Feature flags behave as in the parent: e.g. `BRAIN_USE_REALIZATION_GATE=1` opts into the v0.24.1 realization gate.
- SQLite state is created locally on first use; no seed data ships with this repo by design.
- `LEARNINGS.md` distills the transferable design learnings; `docs/` carries the full architecture record; `CHANGELOG.md`, `RELEASE_NOTES.md`, and the two autonomous changelogs preserve the complete development lineage.
