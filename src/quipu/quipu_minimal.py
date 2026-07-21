"""Quipu Local-Minima Distillation — reversible sparse overlay on the Quipu graph.

Distills the toroidal MESH-SLM Quipu graph down to the **smallest connected
context subgraph** ("local-minima context") that still satisfies a GLOBAL-MAX
objective, so inference can run with minimal token, routing, and
mechanical/electrical (CPU/RAM/VRAM/storage) requirements.

Design principles
-----------------
* **Reversible overlay** — all state lives in two *new* tables
  (``mesh_slm_minimal_pair``, ``mesh_slm_minimal_route``).  The full Quipu
  graph (``mesh_slm_vocab`` / ``mesh_slm_quipu`` / ``mesh_slm_quipu_node``)
  is NEVER destructively pruned by this module.
* **"Entanglement" is classical** — an *entangled key-pair* is a strongly
  coupled directed Quipu token pair ranked by qweight, specialist agreement,
  information sufficiency, and MESH alignment.  No physical
  quantum-computing claim is made or implied.
* **GLOBAL-MAX objective** — maximize retained information, specialist
  consensus, and Observer alignment while penalizing token count, route
  length, and normalized resource requirements.
* **True Dijkstra** — shortest routes use non-negative edge costs with
  deterministic tie-breaking; ``directed_target`` resuscitation links are
  *candidate edges only*, never a greedy replacement for Dijkstra.
* **Offline-only** — no network calls; everything reads/writes the local
  SQLite brain.
* **Default-off** — gated behind ``SCB_QUIPU_MINIMAL`` (env, default off).
  When disabled or when the minimal subgraph is insufficient, callers fall
  back to full-context :func:`mesh_slm.generate`.

Public API
----------
* :func:`minimal_enabled`        — feature-flag probe
* :func:`score_entangled_pairs`  — rank + persist entangled key-pairs
* :func:`select_minimal_context` — smallest sufficient connected subgraph
* :func:`global_max_score`       — GLOBAL-MAX objective for a subgraph
* :func:`cat6_coords` / :func:`cat6_distance` — 6D CAT toroidal geometry
* :func:`dijkstra_route`         — true shortest path over candidate edges
* :func:`greedy_route`           — locally-greedy baseline (for comparison)
* :func:`distill_specialists`    — chain-of-specialists → compact key-pairs
* :func:`minimal_infer`          — flag-gated minimal-context inference
* :func:`overlay_summary`        — diagnostics for ``state_summary()``
"""
from __future__ import annotations

import heapq
import json
import logging
import math
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from . import mesh_slm as _slm
from .ueqgm_engine import wavefunction_overlap as _wavefunction_overlap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_FLAG_ENV = "SCB_QUIPU_MINIMAL"

# Entangled pair score blend (sums to 1.0; every term clipped to [0, 1]).
_PAIR_W_QWEIGHT = 0.40
_PAIR_W_SUPPORT = 0.25
_PAIR_W_SUFFICIENCY = 0.20
_PAIR_W_ALIGNMENT = 0.15

# Dijkstra edge-cost blend (non-negative; every term clipped to [0, 1]).
_COST_W_QWEIGHT = 0.45      # w1 · (1 − qweight)
_COST_W_CAT = 0.25          # w2 · cat6_distance
_COST_W_RESUSC = 0.20       # w3 · resuscitation cost
_COST_W_OBSERVER = 0.10     # w4 · (1 − observer bias)
_COST_EPSILON = 1e-6        # strictly positive costs → guaranteed termination

# GLOBAL-MAX objective blend.
_GAIN_W_INFORMATION = 0.40
_GAIN_W_CONSENSUS = 0.30
_GAIN_W_OBSERVER = 0.30
_PENALTY_W_TOKENS = 0.15
_PENALTY_W_ROUTE = 0.10
_PENALTY_W_RESOURCE = 0.15

# Subgraph selection defaults.
_DEFAULT_SUFFICIENCY_THRESHOLD = 0.35
_DEFAULT_CONFIDENCE_THRESHOLD = 0.20
_DEFAULT_MAX_TOKENS = 24
_MAX_ROUTE_HOPS = 16
_MAX_DIJKSTRA_NODES = 512
_NEIGHBOR_LIMIT = 64

# Specialist distillation bounds (router caps at 3 experts).
_MAX_SPECIALISTS = 3
_MIN_SPECIALIST_SUPPORT = 2

# Storage-footprint estimates (bytes/row — mirrors the architecture map).
_BYTES_VOCAB_ROW = 100
_BYTES_EMBED_ROW = 56
_BYTES_QUIPU_ROW = 32

# Resource normalization ceilings (mirrors asset_resource_mesh._asset_capacity).
_NORM_CORES = 16.0
_NORM_RAM_GB = 64.0
_NORM_VRAM_GB = 24.0
_NORM_STORAGE_GB = 1024.0

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

# ---------------------------------------------------------------------------
# Schema — overlay tables only; the full Quipu graph is never altered here.
# ---------------------------------------------------------------------------
_MINIMAL_DDL = """
CREATE TABLE IF NOT EXISTS mesh_slm_minimal_pair (
    src                 INTEGER NOT NULL,
    dst                 INTEGER NOT NULL,
    pair_score          REAL NOT NULL DEFAULT 0.0,
    qweight             REAL NOT NULL DEFAULT 0.0,
    specialist_support  INTEGER NOT NULL DEFAULT 0,
    sufficiency         REAL NOT NULL DEFAULT 0.0,
    cat_distance        REAL NOT NULL DEFAULT 0.0,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (src, dst)
);
CREATE INDEX IF NOT EXISTS idx_mesh_slm_minimal_pair_score
    ON mesh_slm_minimal_pair(pair_score DESC);
CREATE TABLE IF NOT EXISTS mesh_slm_minimal_route (
    route_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    src             INTEGER NOT NULL,
    dst             INTEGER NOT NULL,
    path_json       TEXT NOT NULL DEFAULT '[]',
    cost            REAL NOT NULL DEFAULT 0.0,
    hops            INTEGER NOT NULL DEFAULT 0,
    fallback_reason TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def _conn():
    """Yield the shared mesh_slm connection with the overlay DDL applied.

    Wraps :func:`mesh_slm._conn` so the base Quipu schema is guaranteed,
    then idempotently ensures the two overlay tables.  Commit/close are
    handled by the wrapped context manager.
    """
    with _slm._conn() as cn:
        cn.executescript(_MINIMAL_DDL)
        yield cn


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------
def minimal_enabled() -> bool:
    """Return True when ``SCB_QUIPU_MINIMAL`` is set to a truthy value (default off)."""
    return str(os.environ.get(_FLAG_ENV, "")).strip().lower() in (
        "1", "true", "yes", "on",
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _finite01(value: Any) -> float:
    """Clamp *value* to [0, 1], mapping NaN/inf/errors to 0.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _observer_bias() -> float:
    """Observer-tangent magnitude in [0, 1] (8D Observer alignment).

    Falls back to the neutral midpoint 0.5 when system_entirety is
    unavailable (e.g. isolated test databases).
    """
    try:
        from .system_entirety import observer_tangent
        return _finite01(observer_tangent())
    except Exception:
        return 0.5


def _embed7(cn: sqlite3.Connection, token_id: int) -> list[float] | None:
    """Return the stored 7-D embedding for *token_id*, or None when absent."""
    row = cn.execute(
        "SELECT e_vision, e_touch, e_smell, e_body, e_brain, e_perception, "
        "e_entirety FROM mesh_slm_embed WHERE token_id=?",
        (token_id,),
    ).fetchone()
    if row is None:
        return None
    return [float(row[k]) for k in (
        "e_vision", "e_touch", "e_smell", "e_body",
        "e_brain", "e_perception", "e_entirety",
    )]


def _token_of(cn: sqlite3.Connection, token_id: int) -> str:
    """Return the surface form for *token_id* (empty string when unknown)."""
    row = cn.execute(
        "SELECT token FROM mesh_slm_vocab WHERE token_id=?", (token_id,)
    ).fetchone()
    return str(row["token"]) if row else ""


# ---------------------------------------------------------------------------
# Phase C — 6D CAT geometry (vision, touch, smell, body, brain, perception)
# ---------------------------------------------------------------------------
def cat6_coords(cn: sqlite3.Connection, token_id: int) -> list[float]:
    """Map *token_id* onto 6D CAT coordinates from its embedding axes.

    Uses the first six embedding components (vision…perception), each
    clipped to [0, 1] so the coordinates live on a unit 6-torus.  Unknown
    tokens sit at the neutral midpoint ``[0.5]·6``.
    """
    vec = _embed7(cn, token_id)
    if vec is None:
        return [0.5] * 6
    return [_finite01(v) for v in vec[:6]]


def cat6_distance(a: list[float], b: list[float]) -> float:
    """Toroidal wrap-aware L1 distance between two 6D CAT coordinates.

    Each axis wraps on the unit interval (``d = min(|Δ|, 1 − |Δ|)``,
    maximum 0.5/axis), generalizing :func:`mesh_slm._torus_dist` to the
    6D CAT unit torus.  Normalized to [0, 1] by the 3.0 (= 6 · 0.5)
    maximum, and symmetric by construction.
    """
    total = 0.0
    for ai, bi in zip(a[:6], b[:6]):
        d = abs(_finite01(ai) - _finite01(bi))
        total += min(d, 1.0 - d)
    return _finite01(total / 3.0)


def _resuscitation_cost(cn: sqlite3.Connection, token_id: int) -> float:
    """Resuscitation cost in [0, 1] for routing *into* a token's torus node.

    Reads the ``mesh_slm_quipu_node`` overlay at the token's flat torus
    index; a higher stamped ``resuscitation_weight`` means the node is
    cheaper to keep alive (lower cost).  Neutral 0.5 when unstamped.
    """
    row = cn.execute(
        "SELECT v.i AS i, v.j AS j FROM mesh_slm_vocab v WHERE v.token_id=?",
        (token_id,),
    ).fetchone()
    if row is None:
        return 0.5
    flat = int(row["i"]) * _slm._TORUS_N + int(row["j"])
    node = cn.execute(
        "SELECT resuscitation_weight FROM mesh_slm_quipu_node WHERE node_id=?",
        (flat,),
    ).fetchone()
    if node is None:
        return 0.5
    return 1.0 - _finite01(node["resuscitation_weight"])


def _edge_cost(
    cn: sqlite3.Connection,
    src: int,
    dst: int,
    qweight: float,
    observer: float,
) -> float:
    """Non-negative Dijkstra edge cost for the directed pair ``src → dst``.

    ``cost = w1·(1−qweight) + w2·cat6_distance + w3·resuscitation_cost
             + w4·(1−observer) + ε``

    Every term is clipped to [0, 1] and ε > 0, so costs are strictly
    positive and finite — the preconditions for Dijkstra correctness.
    """
    cat = cat6_distance(cat6_coords(cn, src), cat6_coords(cn, dst))
    resusc = _resuscitation_cost(cn, dst)
    cost = (
        _COST_W_QWEIGHT * (1.0 - _finite01(qweight))
        + _COST_W_CAT * cat
        + _COST_W_RESUSC * resusc
        + _COST_W_OBSERVER * (1.0 - _finite01(observer))
        + _COST_EPSILON
    )
    return cost if math.isfinite(cost) else 1.0


def _candidate_edges(
    cn: sqlite3.Connection, token_id: int
) -> list[tuple[int, float]]:
    """Return candidate ``(dst_token_id, qweight)`` edges out of *token_id*.

    Candidates are the token's Quipu bigram successors plus (when the
    resuscitation overlay stamps its torus node) the ``directed_target``
    link mapped back to whatever vocab token occupies that torus cell.
    The directed link is a *candidate* with qweight 0.0 — Dijkstra decides
    whether it is worth taking; it is never followed greedily.
    """
    edges: list[tuple[int, float]] = []
    seen: set[int] = set()
    for r in cn.execute(
        "SELECT dst, weight FROM mesh_slm_quipu WHERE src=? "
        "ORDER BY weight DESC LIMIT ?",
        (token_id, _NEIGHBOR_LIMIT),
    ):
        dst = int(r["dst"])
        if dst != token_id and dst not in seen:
            edges.append((dst, _finite01(r["weight"])))
            seen.add(dst)

    row = cn.execute(
        "SELECT i, j FROM mesh_slm_vocab WHERE token_id=?", (token_id,)
    ).fetchone()
    if row is not None:
        flat = int(row["i"]) * _slm._TORUS_N + int(row["j"])
        node = cn.execute(
            "SELECT directed_target FROM mesh_slm_quipu_node WHERE node_id=?",
            (flat,),
        ).fetchone()
        if node is not None:
            ti, tj = divmod(int(node["directed_target"]), _slm._TORUS_N)
            target = cn.execute(
                "SELECT token_id FROM mesh_slm_vocab WHERE i=? AND j=?",
                (ti, tj),
            ).fetchone()
            if target is not None:
                dst = int(target["token_id"])
                if dst != token_id and dst not in seen:
                    edges.append((dst, 0.0))
    return edges


def dijkstra_route(
    cn: sqlite3.Connection,
    src: int,
    dst: int,
    *,
    observer: float | None = None,
    max_nodes: int = _MAX_DIJKSTRA_NODES,
    allowed_nodes: set[int] | None = None,
) -> dict:
    """True Dijkstra shortest route from *src* to *dst* over candidate edges.

    Non-negative edge costs from :func:`_edge_cost`; deterministic
    tie-breaking via ``(cost, node_id)`` heap ordering.  *allowed_nodes*
    optionally restricts the search to a selected minimal subgraph.

    Returns ``{"found", "path", "tokens", "cost", "hops", "explored"}``.
    """
    obs = _observer_bias() if observer is None else _finite01(observer)
    dist: dict[int, float] = {src: 0.0}
    prev: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, src)]
    visited: set[int] = set()

    while heap and len(visited) < max_nodes:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == dst:
            break
        for nxt, qw in _candidate_edges(cn, node):
            if allowed_nodes is not None and nxt not in allowed_nodes:
                continue
            nd = d + _edge_cost(cn, node, nxt, qw, obs)
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = node
                heapq.heappush(heap, (nd, nxt))

    if dst not in dist or dst not in visited:
        return {
            "found": False, "path": [], "tokens": [], "cost": math.inf,
            "hops": 0, "explored": len(visited),
        }
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return {
        "found": True,
        "path": path,
        "tokens": [_token_of(cn, t) for t in path],
        "cost": round(dist[dst], 6),
        "hops": len(path) - 1,
        "explored": len(visited),
    }


def greedy_route(
    cn: sqlite3.Connection,
    src: int,
    dst: int,
    *,
    observer: float | None = None,
    max_hops: int = _MAX_ROUTE_HOPS,
) -> dict:
    """Locally-greedy baseline: always take the cheapest immediate edge.

    Kept for comparison/verification only — demonstrates why
    ``directed_target``-style greedy hops are not a substitute for
    :func:`dijkstra_route`.
    """
    obs = _observer_bias() if observer is None else _finite01(observer)
    path = [src]
    total = 0.0
    visited = {src}
    node = src
    for _ in range(max_hops):
        if node == dst:
            break
        best: tuple[float, int] | None = None
        for nxt, qw in _candidate_edges(cn, node):
            if nxt in visited:
                continue
            c = _edge_cost(cn, node, nxt, qw, obs)
            if best is None or (c, nxt) < best:
                best = (c, nxt)
        if best is None:
            break
        total += best[0]
        node = best[1]
        visited.add(node)
        path.append(node)
    found = node == dst
    return {
        "found": found,
        "path": path if found else path,
        "tokens": [_token_of(cn, t) for t in path],
        "cost": round(total, 6) if found else math.inf,
        "hops": len(path) - 1,
    }


# ---------------------------------------------------------------------------
# Phase A — entangled key-pair scoring
# ---------------------------------------------------------------------------
def score_entangled_pairs(
    cn: sqlite3.Connection, top_k: int = 64
) -> list[dict]:
    """Rank and persist the strongest entangled key-pairs from the Quipu graph.

    Blend per directed pair (all terms finite, clipped to [0, 1]):

    * **qweight** — the Hebbian bigram weight ``mesh_slm_quipu.weight``.
    * **specialist support** — distinct-specialist agreement recorded by
      :func:`distill_specialists`, normalized by the 3-expert router cap.
    * **information sufficiency** — log-frequency mass of the rarer endpoint
      relative to the corpus maximum (rare-but-linked pairs carry signal).
    * **MESH alignment** — wavefunction overlap of the destination embedding
      with the live 7-D MESH state.

    Upserts results into ``mesh_slm_minimal_pair`` (overlay only — the base
    Quipu tables are untouched) and returns the ranked pair dicts.
    """
    mesh = _slm._mesh_state_7d()
    max_freq_row = cn.execute(
        "SELECT MAX(freq) AS m FROM mesh_slm_vocab"
    ).fetchone()
    max_freq = float(max_freq_row["m"] or 1.0)
    log_max = math.log1p(max_freq) or 1.0

    rows = cn.execute(
        "SELECT q.src AS src, q.dst AS dst, q.weight AS w, "
        "       vs.freq AS fsrc, vd.freq AS fdst, "
        "       COALESCE(p.specialist_support, 0) AS support "
        "FROM mesh_slm_quipu q "
        "JOIN mesh_slm_vocab vs ON vs.token_id = q.src "
        "JOIN mesh_slm_vocab vd ON vd.token_id = q.dst "
        "LEFT JOIN mesh_slm_minimal_pair p ON p.src = q.src AND p.dst = q.dst "
        "ORDER BY q.weight DESC LIMIT ?",
        (max(1, top_k) * 2,),
    ).fetchall()

    now_iso = datetime.now(timezone.utc).isoformat()
    scored: list[dict] = []
    for r in rows:
        src, dst = int(r["src"]), int(r["dst"])
        qweight = _finite01(r["w"])
        support = max(0, int(r["support"] or 0))
        support_norm = _finite01(support / float(_MAX_SPECIALISTS))
        rare_freq = float(min(r["fsrc"] or 1, r["fdst"] or 1))
        sufficiency = _finite01(math.log1p(rare_freq) / log_max)
        dst_embed = _embed7(cn, dst)
        try:
            alignment = _finite01(
                _wavefunction_overlap(dst_embed, mesh) if dst_embed else 0.0
            )
        except Exception:
            alignment = 0.0
        cat = cat6_distance(cat6_coords(cn, src), cat6_coords(cn, dst))
        pair_score = _finite01(
            _PAIR_W_QWEIGHT * qweight
            + _PAIR_W_SUPPORT * support_norm
            + _PAIR_W_SUFFICIENCY * sufficiency
            + _PAIR_W_ALIGNMENT * alignment
        )
        cn.execute(
            "INSERT INTO mesh_slm_minimal_pair("
            "src, dst, pair_score, qweight, specialist_support, sufficiency, "
            "cat_distance, updated_at) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(src, dst) DO UPDATE SET "
            "pair_score=excluded.pair_score, qweight=excluded.qweight, "
            "sufficiency=excluded.sufficiency, "
            "cat_distance=excluded.cat_distance, updated_at=excluded.updated_at",
            (src, dst, pair_score, qweight, support, sufficiency, cat, now_iso),
        )
        scored.append({
            "src": src, "dst": dst, "pair_score": pair_score,
            "qweight": qweight, "specialist_support": support,
            "sufficiency": sufficiency, "cat_distance": cat,
        })

    scored.sort(key=lambda p: (-p["pair_score"], p["src"], p["dst"]))
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Phase B — GLOBAL-MAX objective + local-minima subgraph selection
# ---------------------------------------------------------------------------
def _resource_norm(n_tokens: int, n_edges: int) -> float:
    """Normalized mechanical/electrical requirement of a (sub)graph in [0, 1].

    Estimates the resuscitation working set from the storage footprint and
    normalizes CPU/RAM/VRAM/storage against the same capacity ceilings used
    by ``asset_resource_mesh._asset_capacity`` (16 cores, 64 GB RAM, 24 GB
    VRAM, 1024 GB storage).  Monotonically increasing in graph size.
    """
    bytes_est = (
        n_tokens * (_BYTES_VOCAB_ROW + _BYTES_EMBED_ROW)
        + n_edges * _BYTES_QUIPU_ROW
    )
    storage_gb = bytes_est / 1e9
    ram_gb = 3.0 * storage_gb          # working set ≈ 3× on-disk
    cores = n_edges / 2048.0           # candidate-scoring CPU pressure
    vram_gb = 0.0                      # no GPU requirement
    return _finite01(
        (
            cores / _NORM_CORES
            + ram_gb / _NORM_RAM_GB
            + vram_gb / _NORM_VRAM_GB
            + storage_gb / _NORM_STORAGE_GB
        )
        / 4.0
    )


def global_max_score(
    stats: dict,
    *,
    observer: float | None = None,
    token_budget: int = _DEFAULT_MAX_TOKENS,
) -> float:
    """GLOBAL-MAX objective for a candidate minimal subgraph.

    ``score = 0.40·retained_information + 0.30·specialist_consensus
            + 0.30·observer_alignment
            − 0.15·token_penalty − 0.10·route_penalty − 0.15·resource_penalty``

    *stats* keys (missing keys default to 0):
    ``retained_information``, ``specialist_consensus``, ``token_count``,
    ``route_length``, ``n_edges``.  Result is finite and bounded to [0, 1].
    """
    obs = _observer_bias() if observer is None else _finite01(observer)
    retained = _finite01(stats.get("retained_information", 0.0))
    consensus = _finite01(stats.get("specialist_consensus", 0.0))
    n_tokens = max(0, int(stats.get("token_count", 0)))
    route_len = max(0, int(stats.get("route_length", 0)))
    n_edges = max(0, int(stats.get("n_edges", 0)))

    gain = (
        _GAIN_W_INFORMATION * retained
        + _GAIN_W_CONSENSUS * consensus
        + _GAIN_W_OBSERVER * obs
    )
    penalty = (
        _PENALTY_W_TOKENS * _finite01(n_tokens / float(max(1, token_budget)))
        + _PENALTY_W_ROUTE * _finite01(route_len / float(_MAX_ROUTE_HOPS))
        + _PENALTY_W_RESOURCE * _resource_norm(n_tokens, n_edges)
    )
    return _finite01(gain - penalty)


def select_minimal_context(
    cn: sqlite3.Connection,
    seed_tokens: list[str],
    sufficiency_threshold: float = _DEFAULT_SUFFICIENCY_THRESHOLD,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> dict:
    """Select the smallest connected subgraph that meets the thresholds.

    Prim-style growth from the resolved seed tokens: repeatedly attach the
    strongest available Quipu edge (by qweight) until the token budget is
    reached or the frontier is exhausted.  The full Quipu graph is only
    *read* — nothing is deleted or modified.

    Returns a dict with ``nodes``, ``edges``, ``sufficiency``,
    ``confidence``, ``global_max``, and ``fallback_reason`` (``None`` on
    success; ``"no_seed_tokens"`` / ``"insufficient_subgraph"`` otherwise,
    signalling the caller to fall back to full-context inference).
    """
    resolved: list[int] = []
    for tok in seed_tokens or []:
        hit = _slm._load_token(cn, tok)
        if hit is not None and hit[0] not in resolved:
            resolved.append(hit[0])

    if not resolved:
        return {
            "nodes": [], "edges": [], "sufficiency": 0.0, "confidence": 0.0,
            "global_max": 0.0, "seed_coverage": 0.0,
            "fallback_reason": "no_seed_tokens",
        }

    nodes: set[int] = set(resolved[:max_tokens])
    edges: list[dict] = []
    # Max-heap on qweight with deterministic (src, dst) tie-break.
    frontier: list[tuple[float, int, int]] = []

    def _push_edges(tid: int) -> None:
        for dst, qw in _candidate_edges(cn, tid):
            heapq.heappush(frontier, (-qw, tid, dst))

    for tid in nodes:
        _push_edges(tid)

    while frontier and len(nodes) < max_tokens:
        neg_qw, src, dst = heapq.heappop(frontier)
        qw = -neg_qw
        if dst in nodes:
            if not any(e["src"] == src and e["dst"] == dst for e in edges):
                edges.append({"src": src, "dst": dst, "qweight": qw})
            continue
        nodes.add(dst)
        edges.append({"src": src, "dst": dst, "qweight": qw})
        _push_edges(dst)

    seed_coverage = _finite01(
        len([t for t in resolved if t in nodes]) / float(max(1, len(seed_tokens)))
    )
    edge_mass = _finite01(
        sum(e["qweight"] for e in edges) / float(len(edges))
    ) if edges else 0.0

    # Connectivity: fraction of resolved seeds reachable in the undirected
    # selected subgraph from the first seed.
    adjacency: dict[int, set[int]] = {n: set() for n in nodes}
    for e in edges:
        adjacency.setdefault(e["src"], set()).add(e["dst"])
        adjacency.setdefault(e["dst"], set()).add(e["src"])
    reached: set[int] = set()
    stack = [resolved[0]]
    while stack:
        n = stack.pop()
        if n in reached:
            continue
        reached.add(n)
        stack.extend(adjacency.get(n, ()))
    connectivity = _finite01(
        len([t for t in resolved if t in reached]) / float(len(resolved))
    )

    sufficiency = _finite01(
        0.45 * seed_coverage + 0.30 * edge_mass + 0.25 * connectivity
    )
    confidence = _finite01(0.5 * edge_mass + 0.5 * sufficiency)

    support_row = None
    try:
        placeholders = ",".join("?" for _ in nodes) or "-1"
        support_row = cn.execute(
            f"SELECT AVG(specialist_support) AS s FROM mesh_slm_minimal_pair "
            f"WHERE src IN ({placeholders})",
            tuple(nodes),
        ).fetchone()
    except sqlite3.OperationalError:
        support_row = None
    consensus = _finite01(
        float((support_row["s"] if support_row else 0.0) or 0.0)
        / float(_MAX_SPECIALISTS)
    )

    stats = {
        "retained_information": sufficiency,
        "specialist_consensus": consensus,
        "token_count": len(nodes),
        "route_length": 0,
        "n_edges": len(edges),
    }
    result = {
        "nodes": sorted(nodes),
        "edges": edges,
        "sufficiency": sufficiency,
        "confidence": confidence,
        "seed_coverage": seed_coverage,
        "global_max": global_max_score(stats, token_budget=max_tokens),
        "fallback_reason": None,
    }
    if sufficiency < sufficiency_threshold or confidence < confidence_threshold:
        result["fallback_reason"] = "insufficient_subgraph"
    return result


# ---------------------------------------------------------------------------
# Phase D — chain-of-specialists distillation
# ---------------------------------------------------------------------------
def distill_specialists(query: str, *, max_specialists: int = _MAX_SPECIALISTS) -> dict:
    """Distill specialist answers into canonical assertions and compact pairs.

    Runs the offline chain of specialists via
    :func:`expert_orchestrator.run_expert_orchestration`, extracts numeric
    canonical assertions (``specialist <name> answer <n>``) from the
    synthesized answer, feeds them back through
    :func:`mesh_slm.ingest_expert_trace`, and upserts token pairs into the
    overlay with their multi-specialist support count.  Pairs are only
    persisted when support ≥ ``_MIN_SPECIALIST_SUPPORT`` (bounded by the
    ≤ 3 expert router cap).  Execution stays fully local/offline.
    """
    if not query:
        return {"status": "skipped", "reason": "empty_query", "pairs": 0}
    try:
        from .expert_orchestrator import run_expert_orchestration
        result = run_expert_orchestration(
            query, {"source": "quipu_minimal_distill"}
        )
    except Exception as exc:
        logger.debug("distill_specialists: orchestration unavailable: %s", exc)
        return {"status": "error", "reason": str(exc)[:200], "pairs": 0}

    experts = list(result.get("experts_used") or [])[:max_specialists]
    support = min(len(experts), _MAX_SPECIALISTS)
    answer = str(result.get("answer") or "")
    numbers = _NUM_RE.findall(answer)[:4]

    assertions: list[str] = []
    for exp in experts:
        if numbers:
            assertions.append(f"specialist {exp} answer {numbers[0]}")
        try:
            _slm.ingest_expert_trace(exp, answer[:600])
        except Exception:
            pass

    pairs_written = 0
    if support >= _MIN_SPECIALIST_SUPPORT and assertions:
        now_iso = datetime.now(timezone.utc).isoformat()
        with _conn() as cn:
            mesh = _slm._mesh_state_7d()
            for assertion in assertions:
                toks = _slm._tokenize(assertion)[:12]
                ids = [
                    _slm._upsert_token(cn, t, now_iso, mesh=mesh) for t in toks
                ]
                for a, b in zip(ids, ids[1:]):
                    cn.execute(
                        "INSERT INTO mesh_slm_minimal_pair("
                        "src, dst, pair_score, qweight, specialist_support, "
                        "sufficiency, cat_distance, updated_at) "
                        "VALUES(?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(src, dst) DO UPDATE SET "
                        "specialist_support=MAX(specialist_support, excluded.specialist_support), "
                        "updated_at=excluded.updated_at",
                        (
                            a, b,
                            _finite01(_PAIR_W_SUPPORT * support / _MAX_SPECIALISTS),
                            0.0, support, 0.0,
                            cat6_distance(cat6_coords(cn, a), cat6_coords(cn, b)),
                            now_iso,
                        ),
                    )
                    pairs_written += 1
            score_entangled_pairs(cn)
    return {
        "status": "ok",
        "experts": experts,
        "support": support,
        "assertions": assertions,
        "pairs": pairs_written,
    }


# ---------------------------------------------------------------------------
# Phase E — minimal inference API (flag-gated, fallback-safe)
# ---------------------------------------------------------------------------
def _bump_counter(cn: sqlite3.Connection, key: str) -> None:
    """Increment a persisted ``minimal:*`` diagnostics counter."""
    _slm._meta_set(cn, key, int(_slm._meta_get(cn, key, 0) or 0) + 1)


def _full_footprint_bytes(cn: sqlite3.Connection) -> int:
    """Estimated on-disk footprint of the full Quipu graph in bytes."""
    n_vocab = int(
        cn.execute("SELECT COUNT(*) AS c FROM mesh_slm_vocab").fetchone()["c"]
    )
    n_edges = int(
        cn.execute("SELECT COUNT(*) AS c FROM mesh_slm_quipu").fetchone()["c"]
    )
    return (
        n_vocab * (_BYTES_VOCAB_ROW + _BYTES_EMBED_ROW)
        + n_edges * _BYTES_QUIPU_ROW
    )


def _fallback_result(prompt: str, reason: str, kwargs: dict) -> dict:
    """Delegate to full-context :func:`mesh_slm.generate` and wrap the result."""
    gen = _slm.generate(prompt, **kwargs)
    return {
        "text": gen.get("text", ""),
        "confidence": gen.get("confidence", 0.0),
        "selected_pairs": [],
        "route": None,
        "sufficiency": 0.0,
        "compression_ratio": 1.0,
        "qweight_summary": {},
        "resource_reduction": 0.0,
        "fallback_reason": reason,
        "minimal": False,
        "full_context": gen,
    }


def minimal_infer(
    prompt: str,
    *,
    max_new_tokens: int = 24,
    specialist: str | None = None,
    sufficiency_threshold: float = _DEFAULT_SUFFICIENCY_THRESHOLD,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Minimal-context inference over the local-minima Quipu subgraph.

    Behaviour:

    * Flag off (``SCB_QUIPU_MINIMAL`` unset/falsy) → full-context
      :func:`mesh_slm.generate` with ``fallback_reason="flag_disabled"``.
    * Insufficient subgraph → full-context fallback with the selection's
      fallback reason (never an error).
    * Otherwise → constrained emission along the Dijkstra route through the
      selected subgraph, with compression/resource diagnostics.

    Returns the plan-specified contract: ``{text, selected_pairs, route,
    sufficiency, compression_ratio, qweight_summary, resource_reduction,
    fallback_reason, confidence, minimal}``.
    """
    gen_kwargs = {"max_new_tokens": max_new_tokens, "specialist": specialist}
    if not minimal_enabled():
        return _fallback_result(prompt, "flag_disabled", gen_kwargs)

    with _conn() as cn:
        seed_tokens = _slm._tokenize(prompt or "")
        selection = select_minimal_context(
            cn, seed_tokens, sufficiency_threshold, confidence_threshold,
            max_tokens=max(4, min(max_new_tokens, _DEFAULT_MAX_TOKENS)),
        )
        if selection["fallback_reason"]:
            _bump_counter(cn, "minimal:fallbacks")
            reason = selection["fallback_reason"]
            # Delegate outside this connection scope? generate() opens its
            # own connection; commit ours first via context-manager exit.
            fallback = True
        else:
            fallback = False

    if fallback:
        return _fallback_result(prompt, reason, gen_kwargs)

    with _conn() as cn:
        nodes = set(selection["nodes"])
        edges = selection["edges"]
        pairs = score_entangled_pairs(cn, top_k=len(edges) or 8)
        selected_pairs = [
            p for p in pairs if p["src"] in nodes and p["dst"] in nodes
        ][: max(4, len(edges))]

        # Route: anchor = first resolved seed present; target = destination
        # of the strongest selected pair (deterministic).
        anchor = next(
            (t for t in (
                _slm._load_token(cn, tok) for tok in seed_tokens
            ) if t is not None and t[0] in nodes),
            None,
        )
        route: dict | None = None
        if anchor is not None and selected_pairs:
            target = selected_pairs[0]["dst"]
            if target == anchor[0] and len(selected_pairs) > 1:
                target = selected_pairs[1]["dst"]
            route = dijkstra_route(
                cn, anchor[0], target, allowed_nodes=nodes
            )
            cn.execute(
                "INSERT INTO mesh_slm_minimal_route("
                "src, dst, path_json, cost, hops, fallback_reason) "
                "VALUES(?,?,?,?,?,NULL)",
                (
                    anchor[0], target,
                    json.dumps(route.get("path", [])),
                    route.get("cost", 0.0) if route.get("found") else -1.0,
                    route.get("hops", 0),
                ),
            )

        if route and route.get("found"):
            emitted = route["tokens"][: max_new_tokens]
        else:
            emitted = [
                _token_of(cn, p["dst"]) for p in selected_pairs
            ][: max_new_tokens]
        text = " ".join(t for t in emitted if t)

        qweights = [e["qweight"] for e in edges] or [0.0]
        qweight_summary = {
            "min": round(min(qweights), 4),
            "max": round(max(qweights), 4),
            "avg": round(sum(qweights) / len(qweights), 4),
        }

        minimal_bytes = (
            len(nodes) * (_BYTES_VOCAB_ROW + _BYTES_EMBED_ROW)
            + len(edges) * _BYTES_QUIPU_ROW
        )
        full_bytes = _full_footprint_bytes(cn)
        compression_ratio = round(
            full_bytes / float(max(1, minimal_bytes)), 4
        )
        full_norm = _resource_norm(
            int(cn.execute(
                "SELECT COUNT(*) AS c FROM mesh_slm_vocab"
            ).fetchone()["c"]),
            int(cn.execute(
                "SELECT COUNT(*) AS c FROM mesh_slm_quipu"
            ).fetchone()["c"]),
        )
        minimal_norm = _resource_norm(len(nodes), len(edges))
        resource_reduction = _finite01(
            1.0 - (minimal_norm / full_norm) if full_norm > 0 else 0.0
        )

        _bump_counter(cn, "minimal:infers")
        _slm._meta_set(
            cn, "minimal:last_route_cost",
            route.get("cost") if route and route.get("found") else None,
        )
        _slm._meta_set(cn, "minimal:last_compression", compression_ratio)

        return {
            "text": text,
            "confidence": selection["confidence"],
            "selected_pairs": selected_pairs,
            "route": route,
            "sufficiency": selection["sufficiency"],
            "global_max": selection["global_max"],
            "compression_ratio": compression_ratio,
            "qweight_summary": qweight_summary,
            "resource_reduction": resource_reduction,
            "fallback_reason": None,
            "minimal": True,
        }


# ---------------------------------------------------------------------------
# Phase F — diagnostics for mesh_slm.state_summary()
# ---------------------------------------------------------------------------
def overlay_summary(cn: sqlite3.Connection) -> dict:
    """Overlay diagnostics: sizes, compression, routes, support, fallbacks.

    Defensive against missing overlay tables (returns zeroed defaults) so
    ``state_summary()`` never fails on databases created before this module
    existed.
    """
    summary = {
        "pair_count": 0,
        "route_count": 0,
        "avg_pair_score": None,
        "support_histogram": {},
        "last_route_cost": _slm._meta_get(cn, "minimal:last_route_cost", None),
        "last_compression": _slm._meta_get(cn, "minimal:last_compression", None),
        "infer_count": int(_slm._meta_get(cn, "minimal:infers", 0) or 0),
        "fallback_count": int(_slm._meta_get(cn, "minimal:fallbacks", 0) or 0),
        "enabled": minimal_enabled(),
    }
    try:
        row = cn.execute(
            "SELECT COUNT(*) AS c, AVG(pair_score) AS s "
            "FROM mesh_slm_minimal_pair"
        ).fetchone()
        summary["pair_count"] = int(row["c"] or 0)
        summary["avg_pair_score"] = (
            round(float(row["s"]), 4) if row["s"] is not None else None
        )
        summary["route_count"] = int(cn.execute(
            "SELECT COUNT(*) AS c FROM mesh_slm_minimal_route"
        ).fetchone()["c"] or 0)
        summary["support_histogram"] = {
            str(r["specialist_support"]): int(r["c"])
            for r in cn.execute(
                "SELECT specialist_support, COUNT(*) AS c "
                "FROM mesh_slm_minimal_pair GROUP BY specialist_support"
            )
        }
    except sqlite3.OperationalError:
        pass
    return summary
