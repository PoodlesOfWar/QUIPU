"""corpus_ingest — stream open LLM-scale corpora and *holographically compress*
them into the QUIPU mesh (The Well mechanism).

The real condensation mechanism
-------------------------------
QUIPU already implements "The Well" as **holographic compression**: an entire
learning cycle is distilled to a 5-float Newman-Penrose Weyl tensor Ψ₀–Ψ₄
(~50 bytes as JSON, or a 20-byte packed ``5 × Float32 LE`` record), and the
achieved compression is scored by ``mesh_compaction_summary`` (compaction ratio
+ scalar) and ``hawking_information_remnant_score`` (the unitary Page-curve
information budget — Page 1993; Almheiri et al., RMP 93, 035002 (2021)).
See ``ueqgm_engine`` and the Julia ``mesh_compression_model``.

This module drives that mechanism from live data:

1. **Stream** documents from the same public corpora that pretrain SOTA models
   (FineWeb, C4, Wikipedia, Dolma, arXiv, Gutenberg, …) — never a full download.
2. **Fold** each document into the mesh bulk via ``mesh_slm.ingest_expert_trace``
   (tokens → 7-D torus, Hebbian quipu edges), settled by ``train_round``.
3. **Compress** each cycle to its Weyl tensor with ``weyl_scalar_tensor`` over
   five observables (signal flux, topic entropy, corpus volume, mesh alignment,
   Hawking remnant), persist it to ``brain_kv["learnings:weyl_tensor"]`` and as a
   20-byte packed record, and report the compaction ratio/scalar.
4. **Decompress** — the Weyl tensor re-radiates dynamics: ``langevin_sigma_from_weyl``
   turns the current/previous tensors back into the diffusion reference σ that
   drives emission. The tensor is the boundary encoding; σ is what it reconstitutes.

Honest scope
------------
Holographic compression here is *boundary distillation*, not literal document
reconstruction. You cannot decompress the Weyl tensor back into the original
web pages — the point (as with Hawking radiation) is that the irreducible
information remnant is retained in ~20–50 bytes per cycle while the mesh bulk
accretes structure. Access is via sanctioned dataset interfaces only; HTTP
sources are rate-limited.

CLI
---
    python -m quipu.corpus_ingest --list
    python -m quipu.corpus_ingest --sources arxiv,gutenberg --docs 200
    python -m quipu.corpus_ingest --sources fineweb,wikipedia --docs 1000
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import struct
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

from . import mesh_slm
from .ueqgm_engine import (
    hawking_information_remnant_score,
    mesh_compaction_summary,
    weyl_scalar_tensor,
)

# gard_shard_model pulls in `cryptography` at import time (for the encrypted
# shard protocol) which corpus_ingest does not otherwise need — only the pure
# Weyl decompression helper. Gate it the same way the rest of the codebase
# gates optional dependencies ("gates fail toward legacy behavior" per
# LEARNINGS.md) so a missing `cryptography` install degrades corpus_ingest's
# sigma reconstitution to a constant rather than blocking the whole module.
try:
    from .gard_shard_model import langevin_sigma_from_weyl
except Exception:                       # pragma: no cover - optional dep path
    def langevin_sigma_from_weyl(psi5_now, psi5_prev, *, kappa: float = 18.0,
                                 sigma_base: float = 0.05) -> float:
        return float(sigma_base)

try:
    from . import brain_kv
except Exception:                       # pragma: no cover - brain_kv is in-repo
    brain_kv = None                     # type: ignore

try:
    from . import systemic_refinement_agent as _refinement
except Exception:                       # pragma: no cover - Ring 5 optional
    _refinement = None                  # type: ignore

log = logging.getLogger("quipu.corpus_ingest")

_WEYL_TENSOR_KEY = "learnings:weyl_tensor"
_WEYL_PACKED_KEY = "learnings:weyl_tensor_packed_b64"
_WEYL_PREV_KEY = "learnings:weyl_tensor_prev"

# Bounded, timestamped run history — the single "learnings:weyl_tensor" key only
# ever holds the LATEST cycle, which can't answer "what happened today" versus
# some earlier run. This gives daily_digest.py (and anything else that wants a
# same-day view) one authoritative, timestamped feed to filter.
_INGEST_HISTORY_KEY = "corpus_ingest:history"
_INGEST_HISTORY_CAP = 200

# The Well reference geometry (matches ueqgm_engine constants): 16 datasets, 4-D.
_THE_WELL_SPATIAL_DIMS = 4

# 20-byte packed record: 5 x Float32, explicit little-endian.
MESH_BRAIN_KV_WEYL_PACKED_BYTES = 20


# ===========================================================================
# Packed Weyl record — byte-for-byte twin of julia/mesh_compression_model.jl
# pack_weyl / unpack_weyl (v2.2 float-dimension brain_kv record).
# ===========================================================================
def pack_weyl(psi5: Sequence[float]) -> bytes:
    """Pack 5 Weyl scalars into 20 bytes (5 × Float32, little-endian).

    Byte-identical to the Julia ``pack_weyl`` and the ContosoHub v2.2 JS plane,
    so a record written by any peer round-trips on the others. Round-trip error
    vs Float64 inputs is ≤ Float32 eps (~6e-8 rel).
    """
    if len(psi5) < 5:
        raise ValueError("need 5 Weyl scalars")
    return struct.pack("<5f", *[float(x) for x in psi5[:5]])


def unpack_weyl(data: bytes) -> tuple[float, float, float, float, float]:
    """Inverse of :func:`pack_weyl`: 20-byte LE record → 5-tuple of floats."""
    if len(data) < MESH_BRAIN_KV_WEYL_PACKED_BYTES:
        raise ValueError("need >= 20 bytes (5 x Float32 LE)")
    return tuple(float(x) for x in struct.unpack("<5f", data[:20]))  # type: ignore[return-value]


# ===========================================================================
# Text salience — used only to weight signal flux, NOT as the compressor.
# ===========================================================================
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")
_STOP = frozenset(
    # articles / conjunctions / core prepositions
    "the a an and or but nor so yet if then else of to in on at for with by from "
    "as into onto upon over under above below between among through during before "
    "after within without across around about against toward towards per via "
    # pronouns / determiners
    "this that these those it its it's we you they he she him her them us me i my "
    "our your his their mine ours yours theirs who whom whose which what whatever "
    "whichever each every either neither both all any some none such same other "
    "another one ones "
    # verbs (auxiliary / linking / common light verbs)
    "is are was were be been being am do does did done doing have has had having "
    "will would can could shall should may might must ought need dare used get "
    "got gets getting make makes made made say said says "
    # adverbs / fillers / connectives
    "not no nor only just also too very more most less least much many few "
    "so than then thus hence therefore however moreover meanwhile still even ever "
    "never always often sometimes usually here there where when while because "
    "since although though whereas whether "
    # discourse / misc high-frequency function words
    "up down out off again further once new like well back".split()
)


def _salience(text: str) -> tuple[str, float, list[str]]:
    """Return (condensed_trace, salience∈[0,1], salient_terms).

    Salience is the mean informational density of the retained sentences — the
    signal-flux proxy Ψ₀ consumes. The condensed trace is what we feed into the
    mesh bulk; the terms feed topic-entropy accounting.
    """
    text = (text or "").strip()
    if not text:
        return "", 0.0, []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 20] or [text]
    freq: Counter[str] = Counter()
    for s in sents:
        for w in _WORD.findall(s.lower()):
            if w not in _STOP and len(w) > 2:
                freq[w] += 1
    if not freq:
        return text[:1200], 0.0, []
    peak = max(freq.values())
    scored: list[tuple[float, int, str]] = []
    for idx, s in enumerate(sents):
        words = [w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 2]
        if not words:
            continue
        dens = sum(freq[w] for w in words) / (len(words) * peak)
        scored.append((dens, idx, s))
    if not scored:
        return text[:1200], 0.0, []
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:6]
    trace = " ".join(s for _, _, s in sorted(top, key=lambda t: t[1]))[:1200]
    salience = sum(d for d, _, _ in top) / len(top)
    terms = [w for w, _ in freq.most_common(8)]
    return trace, max(0.0, min(1.0, salience)), terms


def _concept_text(text: str, *, max_tokens: int = 60) -> str:
    """Stopword-stripped, concept-dense rendering of *text* for the mesh feed.

    train_round() tokenizes whatever it reads and builds quipu bigram edges
    between consecutive tokens. If we feed it raw prose, the highest-frequency
    tokens are function words (the/of/and) which then saturate the fixed 4096
    torus vocab and — because eviction is lowest-frequency-first — can never be
    displaced, starving real concepts. Feeding concept-only text makes the
    bigrams concept->concept and keeps the vocab meaningful.
    """
    toks = [w for w in _WORD.findall((text or "").lower())
            if w not in _STOP and len(w) > 2]
    return " ".join(toks[:max_tokens])


def _norm_entropy(counts: Counter) -> float:
    """Shannon entropy of a distribution, normalised to [0,1] by log(k)."""
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
    return max(0.0, min(1.0, h / math.log(len(counts))))


# ===========================================================================
# Source registry
# ===========================================================================
@dataclass(frozen=True)
class Source:
    key: str
    kind: str                 # "hf" | "http"
    label: str
    loader: Callable[[int], Iterator[str]]
    note: str = ""


def _hf_stream(dataset: str, text_field: str, *, config: str | None = None,
               split: str = "train") -> Callable[[int], Iterator[str]]:
    def _load(limit: int) -> Iterator[str]:
        try:
            from datasets import load_dataset
        except Exception:
            log.warning("[%s] 'datasets' not installed - skipping. pip install datasets", dataset)
            return
        try:
            ds = load_dataset(dataset, config, split=split, streaming=True)
        except Exception as exc:
            log.warning("[%s] stream open failed: %s", dataset, exc)
            return
        n = 0
        for row in ds:
            txt = (row.get(text_field) or "").strip() if isinstance(row, dict) else ""
            if len(txt) > 200:
                yield txt
                n += 1
                if n >= limit:
                    return
    return _load


def _gutenberg_loader() -> Callable[[int], Iterator[str]]:
    def _load(limit: int) -> Iterator[str]:
        try:
            import requests
        except Exception:
            return
        ids = [1342, 84, 2701, 1661, 11, 98, 1400, 74, 345, 2542,
               1080, 4300, 5200, 174, 2600, 158, 1260, 768, 1232, 55]
        got = 0
        for gid in ids:
            if got >= limit:
                return
            try:
                r = requests.get(f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt", timeout=30)
                if r.status_code == 200 and len(r.text) > 500:
                    body = r.text
                    for i in range(0, min(len(body), 200_000), 4000):
                        yield body[i:i + 4000]
                        got += 1
                        if got >= limit:
                            return
            except Exception as exc:
                log.debug("gutenberg %s failed: %s", gid, exc)
            time.sleep(1.0)
    return _load


def _arxiv_loader() -> Callable[[int], Iterator[str]]:
    def _load(limit: int) -> Iterator[str]:
        try:
            import requests
        except Exception:
            return
        cats = ["cs.LG", "cs.CL", "cs.AI", "stat.ML", "math.OC",
                "eess.SY", "cs.DC", "cs.NE", "q-fin.EC", "physics.data-an"]
        got = 0
        per = max(1, limit // len(cats) + 1)
        for cat in cats:
            if got >= limit:
                return
            url = ("http://export.arxiv.org/api/query?search_query=cat:"
                   f"{cat}&start=0&max_results={per}&sortBy=submittedDate&sortOrder=descending")
            try:
                r = requests.get(url, timeout=30)
                for m in re.findall(r"<summary>(.*?)</summary>", r.text, re.S):
                    txt = re.sub(r"\s+", " ", m).strip()
                    if len(txt) > 200:
                        yield txt
                        got += 1
                        if got >= limit:
                            return
            except Exception as exc:
                log.debug("arxiv %s failed: %s", cat, exc)
            time.sleep(3.0)
    return _load


def _local_docs_loader() -> Callable[[int], Iterator[str]]:
    """Ingest the System Entirety's OWN knowledge corpus — QUIPU's docs/ folder
    (UEQGM engine notes, system-entirety maps, MESH schema, research, perception,
    plasticity, synaptic-agent docs). This is the self-referential expansion the
    external web sources can't provide: the mesh learning its own architecture,
    in its own distinctive vocabulary (weyl, torus, quipu, holographic, floquet,
    entirety, resonance) — a knowledge mode unlike arXiv or fiction, and a
    genuine candidate for a novel ACRE specialist.
    """
    def _load(limit: int) -> Iterator[str]:
        docs_dir = Path(__file__).resolve().parents[2] / "docs"
        if not docs_dir.is_dir():
            return
        got = 0
        for md in sorted(docs_dir.glob("*.md")):
            try:
                text = md.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i in range(0, len(text), 2000):
                chunk = text[i:i + 2000]
                if len(chunk) > 100:
                    yield chunk
                    got += 1
                    if got >= limit:
                        return
    return _load


SOURCES: dict[str, Source] = {
    "local_docs":  Source("local_docs", "local", "QUIPU docs/ (System Entirety self-knowledge)",
                          _local_docs_loader(),
                          "UEQGM / MESH / system-entirety / research docs; the system's own corpus."),
    "fineweb":     Source("fineweb", "hf", "HuggingFaceFW/fineweb (CC web text)",
                          _hf_stream("HuggingFaceFW/fineweb", "text", config="sample-10BT"),
                          "Large filtered Common Crawl - core of modern web pretraining."),
    "c4":          Source("c4", "hf", "allenai/c4 (Colossal Clean Crawl)",
                          _hf_stream("allenai/c4", "text", config="en"),
                          "T5 / many-model pretraining corpus."),
    "wikipedia":   Source("wikipedia", "hf", "wikimedia/wikipedia (20231101.en)",
                          _hf_stream("wikimedia/wikipedia", "text", config="20231101.en"),
                          "Encyclopedic backbone present in nearly all LLM mixes."),
    "openwebtext": Source("openwebtext", "hf", "Skylion007/openwebtext",
                          _hf_stream("Skylion007/openwebtext", "text"),
                          "Open reproduction of the WebText corpus."),
    "dolma":       Source("dolma", "hf", "allenai/dolma (sample)",
                          _hf_stream("allenai/dolma", "text"),
                          "OLMo's open 3T-token pretraining corpus."),
    "stack":       Source("stack", "hf", "bigcode/the-stack-smol (code)",
                          _hf_stream("bigcode/the-stack-smol", "content"),
                          "Permissively-licensed source code."),
    "gutenberg":   Source("gutenberg", "http", "Project Gutenberg (public domain)",
                          _gutenberg_loader(),
                          "Long-form public-domain books; no datasets dep needed."),
    "arxiv":       Source("arxiv", "http", "arXiv export API (abstracts)",
                          _arxiv_loader(),
                          "Scientific abstracts across ML/OR/systems; no datasets dep."),
}

DEFAULT_SOURCES = ["local_docs", "arxiv", "wikipedia", "fineweb", "gutenberg"]


# ===========================================================================
# Holographic compression of an ingest cycle
# ===========================================================================
def compress_cycle_to_weyl(
    *,
    saliences: list[float],
    term_counts: Counter,
    source_counts: Counter,
    approx_source_bytes: float,
) -> tuple[tuple[float, float, float, float, float], dict]:
    """Distill one ingest cycle to its 5-float Newman-Penrose Weyl tensor.

    Maps the cycle's five observables onto Ψ₀–Ψ₄ exactly as
    ``ueqgm_engine.weyl_scalar_tensor`` documents:

    * Ψ₀ signal flux    ← mean document salience
    * Ψ₁ topic entropy  ← normalised Shannon entropy of salient terms
    * Ψ₂ corpus volume  ← documents this cycle (saturating ρ/(ρ+1))
    * Ψ₃ mesh alignment ← mesh wavefunction overlap (from state_summary)
    * Ψ₄ Hawking remnant ← information-remnant score over the source set
    """
    n_docs = len(saliences)
    signal_flux = (sum(saliences) / n_docs) if n_docs else 0.0
    topic_entropy = _norm_entropy(term_counts)

    # Mesh alignment: the mesh's own last wavefunction overlap.
    try:
        st = mesh_slm.state_summary()
        mesh_alignment = float(st.get("last_wavefunction_overlap") or 0.0)
        vocab_size = int(st.get("vocab_size") or 0)
        quipu_edges = int(st.get("quipu_edges") or 0)
    except Exception:
        mesh_alignment, vocab_size, quipu_edges = 0.0, 0, 0

    total_tb = approx_source_bytes / (1024 ** 4)
    remnant = hawking_information_remnant_score(
        n_datasets=max(1, len(source_counts)),
        n_spatial_dims=_THE_WELL_SPATIAL_DIMS,
        total_tb=total_tb,
    )

    psi5 = weyl_scalar_tensor(
        signal_flux=signal_flux,
        topic_entropy=topic_entropy,
        corpus_volume=float(n_docs),
        mesh_alignment=mesh_alignment,
        remnant_score=remnant,
    )

    # Compaction achieved against the actual mesh footprint.
    try:
        comp = mesh_compaction_summary(
            source_tb=max(total_tb, 1e-12),
            n_vocab=vocab_size or None,
            n_quipu_edges=quipu_edges or None,
        )
    except Exception:
        comp = {}

    meta = {
        "psi5": [round(x, 6) for x in psi5],
        "signal_flux": round(signal_flux, 6),
        "topic_entropy": round(topic_entropy, 6),
        "corpus_volume": n_docs,
        "mesh_alignment": round(mesh_alignment, 6),
        "hawking_remnant_score": round(remnant, 6),
        "vocab_size": vocab_size,
        "quipu_edges": quipu_edges,
        "compaction_ratio": comp.get("compaction_ratio"),
        "compaction_scalar": comp.get("compaction_scalar"),
        "mesh_bytes_total": comp.get("mesh_bytes_total"),
        "approx_source_bytes": int(approx_source_bytes),
    }
    return psi5, meta


def _persist_weyl(psi5: Sequence[float]) -> str | None:
    """Write the tensor to brain_kv (canonical key + prev + 20-byte packed b64)."""
    if brain_kv is None:
        return None
    import base64
    try:
        prev = brain_kv.kv_get_json(_WEYL_TENSOR_KEY, None)
        if isinstance(prev, list) and len(prev) == 5:
            brain_kv.kv_set_json(_WEYL_PREV_KEY, prev)
        brain_kv.kv_set_json(_WEYL_TENSOR_KEY, [float(x) for x in psi5])
        # GARD information-science mirror key (additive; weyl key kept for wire compat).
        brain_kv.kv_set_json("learnings:gard_state", [float(x) for x in psi5])
        packed_b64 = base64.b64encode(pack_weyl(psi5)).decode("ascii")
        brain_kv.kv_set(_WEYL_PACKED_KEY, packed_b64)
        return packed_b64
    except Exception as exc:
        log.debug("weyl persist failed: %s", exc)
        return None


def decompress_sigma() -> float:
    """Reconstitute the Langevin diffusion reference σ from the stored tensors.

    This is the decompression side of the holographic pair: the current and
    previous Weyl tensors are lifted back into the σ that drives mesh emission
    (``gard_shard_model.langevin_sigma_from_weyl``). Returns the base σ if no
    tensor pair is available yet.
    """
    if brain_kv is None:
        return 0.05
    now = brain_kv.kv_get_json(_WEYL_TENSOR_KEY, None)
    prev = brain_kv.kv_get_json(_WEYL_PREV_KEY, now)
    if not (isinstance(now, list) and isinstance(prev, list)):
        return 0.05
    return langevin_sigma_from_weyl(now, prev)


# ===========================================================================
# Ingestion loop
# ===========================================================================
def run_ingest(
    sources: Sequence[str] | None = None,
    *,
    docs_per_source: int = 250,
    train_every: int = 40,
    compress_every: int = 120,
    refine_every: int = 1,
    max_tools_per_cycle: int = 2,
    max_seconds: float = 0.0,
    strength: float = 1.0,
) -> dict:
    """Stream, fold into the mesh bulk, and holographically compress to Weyl.

    Every ``train_every`` docs the mesh settles (``train_round``); every
    ``compress_every`` docs the accumulated cycle is distilled to its Weyl
    tensor, persisted to ``brain_kv``, and reported with its compaction ratio.

    Every ``refine_every`` Weyl cycles (default: every cycle), this also runs
    one Ring 5 refinement pass (``systemic_refinement_agent.run_strategy``):
    ACRE gets a chance to crystallize a new emergent specialist, and
    ``tool_forge`` scans the freshly-grown mesh corpus for clusters worth
    auto-implementing as tools. This is what makes Ring 5 real rather than a
    stub — the crawl feeds Ring 4 (Corpus), which this loop immediately hands
    to Ring 5 (Refinement -> Toolforge) instead of leaving it inert.
    """
    keys = list(sources) if sources else list(DEFAULT_SOURCES)
    unknown = [k for k in keys if k not in SOURCES]
    if unknown:
        raise ValueError(f"unknown sources {unknown}; known: {sorted(SOURCES)}")

    started = time.monotonic()
    per_source: dict[str, dict] = {}
    weyl_snapshots: list[dict] = []
    refinement_cycles: list[dict] = []
    total = 0
    weyl_cycle_count = 0

    # Rolling accumulators for the current compression cycle.
    cyc_sal: list[float] = []
    cyc_terms: Counter = Counter()
    cyc_sources: Counter = Counter()
    cyc_bytes = 0.0

    def _emit_weyl() -> None:
        nonlocal cyc_sal, cyc_terms, cyc_sources, cyc_bytes, weyl_cycle_count
        if not cyc_sal:
            return
        psi5, meta = compress_cycle_to_weyl(
            saliences=cyc_sal, term_counts=cyc_terms,
            source_counts=cyc_sources, approx_source_bytes=cyc_bytes,
        )
        packed = _persist_weyl(psi5)
        meta["packed_b64"] = packed
        meta["sigma_reconstituted"] = round(decompress_sigma(), 6)
        meta["at"] = datetime.now(timezone.utc).isoformat()
        weyl_snapshots.append(meta)
        weyl_cycle_count += 1
        log.info("[weyl] Ψ=%s ratio=%s scalar=%s remnant=%s σ=%s",
                 meta["psi5"], meta.get("compaction_ratio"),
                 meta.get("compaction_scalar"), meta["hawking_remnant_score"],
                 meta["sigma_reconstituted"])
        cyc_sal, cyc_terms, cyc_sources, cyc_bytes = [], Counter(), Counter(), 0.0

        # Ring 4 -> Ring 5: hand the freshly-compressed corpus to refinement.
        if _refinement is not None and refine_every > 0 and weyl_cycle_count % refine_every == 0:
            try:
                cycle = _refinement.run_strategy(max_tools=max_tools_per_cycle)
                refinement_cycles.append(cycle)
                log.info("[ring5] refinement cycle: actions=%s", cycle.get("actions"))
            except Exception as exc:
                log.debug("[ring5] refinement cycle failed: %s", exc)

    for key in keys:
        src = SOURCES[key]
        got = 0
        log.info("[ingest] %s - %s", key, src.label)
        try:
            for raw in src.loader(docs_per_source):
                if max_seconds and (time.monotonic() - started) > max_seconds:
                    log.info("[ingest] time budget reached.")
                    break
                trace, sal, terms = _salience(raw)
                if len(trace) < 60:
                    continue
                # Feed concept-dense (stopword-stripped) text into the mesh
                # corpus feed. train_round() reads this and builds the vocab +
                # quipu edges. (The old ingest_expert_trace path only created
                # tokens, never edges, and kept stopwords - which saturated the
                # 4096 vocab with function words and starved real concepts.)
                concept = _concept_text(trace)
                if len(concept.split()) < 3:
                    continue
                try:
                    mesh_slm.feed_corpus(concept, source=f"corpus_{key}")
                except Exception as exc:
                    log.debug("[ingest] mesh feed failed: %s", exc)
                    continue

                cyc_sal.append(sal)
                cyc_terms.update(terms)
                cyc_sources[key] += 1
                cyc_bytes += len(raw.encode("utf-8", "ignore"))
                got += 1
                total += 1

                if total % train_every == 0:
                    try:
                        mesh_slm.train_round(max_seconds=8.0, max_chunks=120)
                    except Exception as exc:
                        log.debug("[ingest] train_round hiccup: %s", exc)
                if total % compress_every == 0:
                    _emit_weyl()
        except Exception as exc:
            log.warning("[ingest] source %s aborted: %s", key, exc)
        per_source[key] = {"ingested": got, "label": src.label}
        if max_seconds and (time.monotonic() - started) > max_seconds:
            break

    # Final settle + final holographic compression.
    try:
        mesh_slm.train_round(max_seconds=15.0, max_chunks=240)
    except Exception:
        pass
    _emit_weyl()

    latest = weyl_snapshots[-1] if weyl_snapshots else {}
    tools_forged_total = sum(
        (c.get("forge_summary") or {}).get("tools_forged", 0) for c in refinement_cycles
    )
    result = {
        "status": "ok",
        "at": datetime.now(timezone.utc).isoformat(),
        "total_condensed": total,
        "elapsed_s": round(time.monotonic() - started, 1),
        "per_source": per_source,
        "weyl_cycles": len(weyl_snapshots),
        "weyl_latest": latest,
        "weyl_history": weyl_snapshots,
        "refinement_cycles": refinement_cycles,
        "tools_forged_total": tools_forged_total,
    }
    log.info("[ingest] done: %s docs, %s Weyl cycles, %s Ring-5 cycles (%s tools forged), %ss",
             total, len(weyl_snapshots), len(refinement_cycles), tools_forged_total,
             result["elapsed_s"])
    _record_ingest_run(result)
    return result


def _record_ingest_run(result: dict) -> None:
    """Append a compact summary of this run to the bounded history in brain_kv."""
    if brain_kv is None:
        return
    try:
        entry = {
            "ts": result["at"],
            "total_condensed": result["total_condensed"],
            "elapsed_s": result["elapsed_s"],
            "per_source": {k: v.get("ingested", 0) for k, v in result["per_source"].items()},
            "weyl_cycles": result["weyl_cycles"],
            "weyl_latest_psi5": (result.get("weyl_latest") or {}).get("psi5"),
            "compaction_ratio": (result.get("weyl_latest") or {}).get("compaction_ratio"),
            "compaction_scalar": (result.get("weyl_latest") or {}).get("compaction_scalar"),
            "tools_forged_total": result["tools_forged_total"],
        }
        history = brain_kv.kv_get_json(_INGEST_HISTORY_KEY, [])
        if not isinstance(history, list):
            history = []
        history.append(entry)
        brain_kv.kv_set_json(_INGEST_HISTORY_KEY, history[-_INGEST_HISTORY_CAP:])
    except Exception as exc:
        log.debug("corpus_ingest: history record failed: %s", exc)


def ingest_history() -> list[dict]:
    """Return the bounded history of past run_ingest() calls (newest last)."""
    if brain_kv is None:
        return []
    h = brain_kv.kv_get_json(_INGEST_HISTORY_KEY, [])
    return h if isinstance(h, list) else []


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Holographically compress open LLM corpora into the QUIPU mesh (The Well).")
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="comma list; see --list")
    ap.add_argument("--docs", type=int, default=250, help="max documents per source")
    ap.add_argument("--train-every", type=int, default=40)
    ap.add_argument("--compress-every", type=int, default=120, help="docs per Weyl cycle")
    ap.add_argument("--refine-every", type=int, default=1, help="Weyl cycles per Ring-5 refinement pass; 0 disables")
    ap.add_argument("--max-tools-per-cycle", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0 = no time limit")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--reset", action="store_true",
                    help="clear the (possibly stopword-saturated) mesh vocab/edges/feed before ingesting")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.reset:
        info = mesh_slm.reset_mesh_learning()
        print(f"Mesh learning state reset - cleared {info.get('cleared')}")
        try:
            from . import tool_forge
            tinfo = tool_forge.clean_generated_tools()
            print(f"Forged tools cleared - removed {tinfo.get('removed')} files")
        except Exception as exc:
            print(f"(tool cleanup skipped: {exc})")

    if args.list:
        try:
            import datasets  # noqa: F401
            hf_ok = True
        except Exception:
            hf_ok = False
        print("Available sources:")
        for k, s in SOURCES.items():
            flag = "" if s.kind in ("http", "local") or hf_ok else "  (needs: pip install datasets)"
            print(f"  {k:12s} [{s.kind}] {s.label}{flag}")
            if s.note:
                print(f"               {s.note}")
        if not hf_ok:
            print("\nNote: 'datasets' not installed - only http sources will run.")
        return

    keys = [k.strip() for k in args.sources.split(",") if k.strip()]
    res = run_ingest(keys, docs_per_source=args.docs, train_every=args.train_every,
                     compress_every=args.compress_every, refine_every=args.refine_every,
                     max_tools_per_cycle=args.max_tools_per_cycle, max_seconds=args.max_seconds)
    print(f"\nCondensed {res['total_condensed']} documents into {res['weyl_cycles']} Weyl cycles "
          f"in {res['elapsed_s']}s")
    w = res.get("weyl_latest") or {}
    if w:
        print(f"  latest Weyl tensor Ψ0-4 : {w.get('psi5')}")
        print(f"  compaction ratio        : {w.get('compaction_ratio')}")
        print(f"  compaction scalar       : {w.get('compaction_scalar')}")
        print(f"  hawking remnant score   : {w.get('hawking_remnant_score')}")
        print(f"  packed 20-byte (b64)    : {w.get('packed_b64')}")
        print(f"  reconstituted sigma     : {w.get('sigma_reconstituted')}")
    print(f"\nRing 5: {len(res['refinement_cycles'])} refinement cycles, "
          f"{res['tools_forged_total']} tools forged")
    if res["refinement_cycles"]:
        last = res["refinement_cycles"][-1]
        print(f"  last cycle actions      : {last.get('actions')}")
        print(f"  acre specialists        : {last.get('acre_specialists')}")


if __name__ == "__main__":
    _cli()
