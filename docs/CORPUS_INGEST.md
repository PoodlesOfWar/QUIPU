# Corpus Ingestion — The Well holographic compression

`src/quipu/corpus_ingest.py` streams the openly published corpora that pretrain
current SOTA language models and folds them into QUIPU using the mesh's own
**holographic compression** — "The Well." An entire ingest cycle is distilled to
a 5-float Newman-Penrose Weyl tensor (~50 bytes as JSON, or a 20-byte packed
record), scored by the Bekenstein-Hawking information surface.

## The mechanism

Each cycle runs four stages:

1. **Stream** documents from public dataset interfaces — never a full download.
2. **Fold** each document into the mesh bulk via `mesh_slm.ingest_expert_trace`
   (tokens → 7-D torus, Hebbian quipu edges), settled by `train_round`.
3. **Compress** the cycle to its Weyl tensor Ψ₀–Ψ₄ with
   `ueqgm_engine.weyl_scalar_tensor` over five observables, persist it to
   `brain_kv["learnings:weyl_tensor"]` and as a 20-byte packed record, and report
   `mesh_compaction_summary` (compaction ratio + scalar) and
   `hawking_information_remnant_score`.
4. **Decompress** — `gard_shard_model.langevin_sigma_from_weyl` lifts the
   current/previous tensors back into the diffusion reference σ that drives mesh
   emission. The tensor is the boundary encoding; σ is what it reconstitutes.

### Weyl scalar mapping

| Scalar | Meaning | Source observable |
|--------|---------|-------------------|
| Ψ₀ | ingoing signal flux | mean document salience this cycle |
| Ψ₁ | topic entropy | normalised Shannon entropy of salient terms |
| Ψ₂ | Coulomb bulk mass | documents this cycle, saturating ρ/(ρ+1) |
| Ψ₃ | outgoing alignment | mesh wavefunction overlap (`state_summary`) |
| Ψ₄ | Hawking remnant | information-remnant score over the source set |

### Packed record (cross-language)

`pack_weyl` / `unpack_weyl` serialise the tensor to **20 bytes (5 × Float32,
little-endian)** — byte-for-byte identical to the Julia
`mesh_compression_model.jl` peer and the v2.2 KV wire format. Golden vector
`(0.0, 0.25, 0.5, 0.75, 1.0)` → base64 `AAAAAAAAgD4AAAA/AABAPwAAgD8=` on every
peer.

## Honest scope

Holographic compression here is **boundary distillation, not document
reconstruction**. You cannot decompress the Weyl tensor back into the original
web pages — as with Hawking radiation, the irreducible information remnant is
retained in ~20–50 bytes per cycle while the mesh bulk accretes structure. A
frontier pretraining set is petabyte-scale and is never ingested whole; this
streams and distills a continuous sample. Access is via sanctioned dataset
interfaces only; HTTP sources are rate-limited.

## Sources

| key         | kind | corpus | needs `datasets`? |
|-------------|------|--------|-------------------|
| `fineweb`   | hf   | HuggingFaceFW/fineweb (filtered Common Crawl) | yes |
| `c4`        | hf   | allenai/c4 (Colossal Clean Crawl) | yes |
| `wikipedia` | hf   | wikimedia/wikipedia | yes |
| `openwebtext` | hf | Skylion007/openwebtext | yes |
| `dolma`     | hf   | allenai/dolma (OLMo corpus) | yes |
| `stack`     | hf   | bigcode/the-stack-smol (code) | yes |
| `arxiv`     | http | arXiv export API (abstracts) | no |
| `gutenberg` | http | Project Gutenberg (public domain) | no |

Default mix: `fineweb, wikipedia, arxiv, gutenberg`.

## Usage

```bash
python -m quipu.corpus_ingest --list

# HTTP-only (no extra install) — safe first smoke test
python -m quipu.corpus_ingest --sources arxiv,gutenberg --docs 200

# Full web-scale mix (after: pip install datasets)
python -m quipu.corpus_ingest --sources fineweb,wikipedia,arxiv --docs 1000 --compress-every 200

# Bounded background run (stop after 30 min)
python -m quipu.corpus_ingest --max-seconds 1800
```

Each `--compress-every` documents emits one Weyl cycle; the CLI prints the
latest tensor, compaction ratio/scalar, remnant score, packed base64, and the
reconstituted σ.

Programmatic:

```python
from quipu import corpus_ingest
res = corpus_ingest.run_ingest(["fineweb", "wikipedia"], docs_per_source=500)
print(res["weyl_latest"])          # Ψ tensor + compaction metrics for the last cycle
```

## Persistence

The tensor lands in the shared mesh SQLite `brain_kv` table (added as
`src/quipu/brain_kv.py`), under:

- `learnings:weyl_tensor` — current Ψ₀–Ψ₄ (JSON)
- `learnings:weyl_tensor_prev` — previous cycle (for the σ delta)
- `learnings:weyl_tensor_packed_b64` — 20-byte packed record, base64

`gard_shard_model` reads `learnings:weyl_tensor` back through `brain_kv`, so the
compressed corpus state feeds the model context and the Langevin emission path.

## Enabling the streaming corpora

```
pip install datasets
```

Without it, the `hf` sources are skipped and the `http` sources still run. Some
HuggingFace datasets are access-gated and require `huggingface-cli login`.
