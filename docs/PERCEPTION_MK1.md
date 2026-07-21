# Perception Mk1 — Visual and Embodied Reasoning

**Introduced**: v0.20.16 (2025-07)
**Module**: `pipeline/src/quipu/perception.py`

---

## Overview

Perception Mk1 is the **sixth sense** of the Symbiotic System. Where the
existing Vision sense discovers knowledge through structured outreach and
file-system scanning, Perception operates on raw sensory input — images,
animated GIF frame sequences, and spatial diagrams — using Vision-Language
Models (VLMs) to extract meaning that feeds back into the corpus knowledge
graph and the bilateral Vision↔Touch pressure field.

### How it differs from "Vision"

| Sense  | Module              | Input                        | Method                          |
|--------|---------------------|------------------------------|---------------------------------|
| Vision | `knowledge_corpus`  | OCW pages, DW schema, JSON   | Structured outreach / scanning  |
| Perception | `perception`    | PNG, JPG, GIF, WebP files    | VLM multipart message dispatch  |

---

## Architecture

```
docs/ and data/documents/
        │
        ▼
┌─────────────────────────────────┐
│  scan_visual_assets()           │  Enumerate .png / .jpg / .gif / .webp
│  _file_hash()                   │  SHA-1 deduplication (brain_kv registry)
└───────────────┬─────────────────┘
                │
      ┌─────────┴──────────┐
      ▼                    ▼
 Static image           Animated GIF
 _encode_image_b64()    _sample_gif_frames()
 (max 512×512 px,       (up to 8 frames,
  ≤500 KB raw)           256×256 px each)
      │                    │
      ▼                    ▼
 _build_image_message() _build_video_message()
 (OpenAI multipart)     (multi-image sequence)
      │                    │
      └─────────┬──────────┘
                ▼
  dispatch_parallel(
    "perception_visual" | "perception_video",
    {"messages": [<multipart>]}
  )
       │
       │  brain.yaml task_profiles route to
       │  VLMs with highest vision score:
       │    qwen3.5-397b-a17b (0.78)
       │    minimax-m2.7      (0.74)
       │    gemma-4           (0.70)
       ▼
  EnsembleResult.answer  (JSON string)
       │
       ├─── _extract_first_json_object()  →  parsed dict
       │
       ├─── _log_learning()
       │      learning_log: kind='perception' | 'perception_video'
       │
       ├─── _add_entities_to_corpus()
       │      corpus_entity: type='VisualObservation'
       │      corpus_edge:   rel='PERCEIVED_IN'
       │
       └─── brain_kv.perception_ops
              { perception_entities: int, perception_frames: int }
                  │
                  ▼
            Touch gradient relief
            (brain_body_signals._VISION_OPS_MAP)
```

---

## VLM Task Profiles

Defined in `config/brain.yaml` under `llms.task_profiles`:

```yaml
perception_visual:
  # Static image → description + entity extraction
  weights: { vision: 0.60, reasoning: 0.30, structured: 0.10 }
  min_ctx: 16000
  lambda_cost: 0.10
  lambda_latency: 0.20

perception_video:
  # Temporal multi-frame → embodied / motion reasoning
  weights: { vision: 0.50, reasoning: 0.35, long_ctx: 0.15 }
  min_ctx: 32000
  lambda_cost: 0.10
  lambda_latency: 0.30
```

The `vision` weight (0.50–0.60) routes to models with the highest visual
capability in the `brain.yaml` model registry:

| Model               | vision score | ctx    |
|---------------------|-------------|--------|
| qwen3.5-397b-a17b   | 0.78        | 256k   |
| minimax-m2.7        | 0.74        | 4M     |
| gemma-4             | 0.70        | 1M     |
| glm-5.1             | 0.66        | —      |
| kimi-k2.5           | 0.62        | —      |

---

## VLM Prompts

### Static image (`_VISUAL_PROMPT`)
```
Analyse this image in the context of supply chain / manufacturing /
operations management. Describe:
1. What you see (objects, text, charts, diagrams, layouts)
2. Any part numbers, supplier names, site names, or product names visible
3. Key data values or metrics shown
4. Potential supply-chain relevance
Respond in JSON: {"description": str, "entities": [str],
"metrics": {key: value}, "relevance": str, "confidence": float}
```

### Temporal sequence (`_VIDEO_PROMPT`)
```
These are sequential frames from a GIF/video in a supply chain context.
Analyse the temporal sequence: what process or motion is shown, changes
visible over time, part numbers or labels that appear, operational significance.
Respond in JSON: {"sequence_description": str, "entities": [str],
"events": [str], "trend": str, "confidence": float}
```

---

## Rate Limiting and Deduplication

- **Rate limit**: 90 seconds minimum between `perception_step()` calls.  
  Prevents flooding during rapid corpus rounds.  
- **Deduplication**: SHA-1 hash of first 64 KB of each file is stored in  
  `brain_kv.perception_analysed_files`. Re-processed only if the file
  changes.  
- **Max per round**: Controlled by `perception.max_images_per_round` dial  
  (default 4, grows to 16 at full corpus richness via neural_plasticity).

---

## Corpus Integration

Visual entities are stored as `type='VisualObservation'` in `corpus_entity`,
distinguished from structured DW entities. Each entity has a `PERCEIVED_IN`
edge back to its `PerceptualSource` node so the knowledge graph retains
provenance.

Learning log kinds used:

| kind               | Trigger                     |
|--------------------|-----------------------------|
| `perception`       | Static image analysis       |
| `perception_video` | GIF/temporal sequence       |
| `perception_map`   | Reserved — Mk2 spatial maps |

---

## Touch Gradient Relief

Perception ops are merged into `vision_ops` at the tail of
`knowledge_corpus.refresh_corpus_round()` and processed by
`brain_body_signals._VISION_OPS_MAP`:

| Op key               | Relieves signal_kind        | Gradient / unit |
|----------------------|-----------------------------|-----------------|
| `perception_entities`| `missing_category`          | −0.015          |
| `perception_frames`  | `corpus_rag_saturated`      | −0.010          |

**Rationale**: Discovering new visual entities (parts, suppliers, sites)
relieves the "we don't know about this category" pressure. Processing more
video frames relieves "corpus RAG is over-saturated" pressure by showing
the system is consuming diverse visual modalities.

---

## Neural Plasticity Dials

The `perception` dial group is managed by `neural_plasticity.py`:

| Dial                    | Default | Growth target (at richness=1) |
|-------------------------|---------|-------------------------------|
| `max_images_per_round`  | 4.0     | 16.0                          |
| `confidence_threshold`  | 0.40    | 0.25                          |
| `scan_depth`            | 1.0     | 2.0                           |

`scan_depth`:
- `0` — `docs/` only  
- `1` — `docs/` + `data/documents/` (default)  
- `2` — all visual assets in `pipeline/`  

---

## Temporal Spatiality Weight

Perception is the 6th sense in `temporal_spatiality._SENSE_WEIGHTS`.
Existing sense weights were each reduced slightly to keep the sum at 1.0:

| Sense      | Pre-Mk1 | Post-Mk1 |
|------------|---------|----------|
| vision     | 0.25    | 0.22     |
| touch      | 0.25    | 0.22     |
| smell      | 0.20    | 0.18     |
| body       | 0.15    | 0.12     |
| brain      | 0.15    | 0.12     |
| perception | —       | **0.14** |
| **Total**  | **1.00**| **1.00** |

Perception coherence is measured as:
```
min(1.0, learning_log_rows_last_1h / 4.0)
```
where rows have `kind IN ('perception', 'perception_video', 'perception_map')`.

---

## Public API

```python
from src.quipu.perception import (
    perception_step,        # main entry point — rate-limited round
    get_perception_ops,     # latest ops dict for Touch gradient
    get_perception_coherence,  # [0,1] signal for temporal_spatiality
    get_perception_dial,    # read a single plasticity dial
    scan_visual_assets,     # enumerate all visual files in scan paths
)
```

### `perception_step() → dict`

```python
{
    "skipped": bool,                # True if rate-limited
    "images_analysed": int,
    "videos_analysed": int,
    "entities_added": int,
    "perception_entities": int,     # for Touch gradient
    "perception_frames": int,       # for Touch gradient
    "total_assets": int,
    "unanalysed_remaining": int,
    "notes": list[str],             # debug notes
}
```

---

## Mk1 → Mk2 Expansion Roadmap

| Milestone | Capability                                               |
|-----------|----------------------------------------------------------|
| **Mk1**   | PNG/JPG/GIF/WebP analysis (this release)                 |
| **Mk2**   | PDF page rendering via pdf2image/poppler                 |
| **Mk2**   | Video file support via ffmpeg frame extraction           |
| **Mk3**   | Spatial embodied reasoning (floor plans, schematics)     |
| **Mk3**   | Dashboard screenshot loop (Playwright → VLM → KPI)       |

---

*See also*: [VISION_TOUCH_CLOSED_LOOP.md](VISION_TOUCH_CLOSED_LOOP.md),
[NEURAL_PLASTICITY.md](NEURAL_PLASTICITY.md), [ARCHITECTURE.md](ARCHITECTURE.md)
