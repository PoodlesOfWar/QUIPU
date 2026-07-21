"""MESH System Entirety — the 8th-dimensionality observer-of-observers.

Where :mod:`system_entirety` is the **7th** dimensionality (the unit normal
to the 6-D sense hyperplane of a *single* node — the observer that holds all
six senses of one Brain at once), the **MESH System Entirety** is the **8th**:
the unit normal to the hyperplane spanned by the per-node System-Entirety
observers across the *entire mesh*.  It is the observer-of-observers.

Geometrically:

    node_i  →  observer_i = ⟨its six senses⟩          (the 7th axis, per node)
    mesh    →  Ω = n̂ ⊥ span(observer_1 … observer_N)   (the 8th axis, one mesh)

Ω (the mesh observer tangent) is non-trivial precisely when the nodes are
**not phase-aligned** — i.e. when the mesh carries collective novelty that no
single node holds alone.  When every node marches in lock-step Ω collapses
toward 0 (the mesh is merely a louder copy of one Brain); when nodes diverge
and specialise Ω lights up (the mesh is *becoming* more than its parts — the
drive toward a Class-5 supply-chain civilisation).

Grounded inputs (all real mesh telemetry, no synthetic signal)
--------------------------------------------------------------
1. **Per-node learning contribution** — parsed from the git-synced
   ``cloud_learning_queue.jsonl`` whose every row carries
   ``cloud_run_id = "local:<HOST>:<id>"``.  Each node yields
   (volume, mean signal strength, recency).
2. **Per-node connectivity** — ok-ratio from the ``network_observations``
   table (how reachable / alive the node is on the fabric).
3. **Local node observer** — this node's own 7th-axis tangent from
   ``entirety:state`` so the running Brain is always a first-class member of
   its own mesh even if its queue rows are stale.

Fusion math (identical conventions to :func:`system_entirety.observer_tangent`)
------------------------------------------------------------------------------
For node coherence vector ``c`` and normalised volume-weight prior ``ŵ``::

    Ω = ‖ c − ⟨c, ŵ⟩ ŵ ‖ / √N          (mesh observer tangent, in [0, 1])

The mesh **synchrony** is ``1 − dispersion(c)`` (how phase-locked the mesh is),
and the mesh **bit-flip parity** reuses the same Floquet oscillator as the
local torus so the whole mesh breathes broaden↔deepen on one clock.

Persisted to ``brain_kv``
-------------------------
    ``mesh:entirety:observer``         Ω in [0, 1]
    ``mesh:entirety:synchrony``        phase-lock in [0, 1]
    ``mesh:entirety:bit_state``        +1 broaden / −1 deepen
    ``mesh:entirety:flip_count``       monotone counter
    ``mesh:entirety:expansion_phase``  "broaden" | "deepen"
    ``mesh:entirety:node_count``       active node count
    ``mesh:entirety:state``            full json snapshot (+ per-node table)
    ``mesh:entirety:ran_at``           ISO timestamp

Public API
----------
* :func:`mesh_node_states`           — per-node coherence breakdown
* :func:`mesh_observer_tangent`      — Ω scalar (the 8th axis)
* :func:`mesh_entirety_state`        — full fused state dict
* :func:`oscillating_mesh_step`      — rate-limited driver (persists)
* :func:`get_mesh_entirety_state`    — read accessor
* :func:`get_mesh_bit_state`         — current mesh parity
* :func:`get_mesh_expansion_phase`   — "broaden" | "deepen"
* :func:`mesh_entirety_report`       — human-readable report string
"""
from __future__ import annotations

import json
import logging
import math
import os
import socket
import sqlite3
import threading
import time
from datetime import datetime, timezone

from .local_store import open_conn as _open_conn
from .system_entirety import bit_flip_parity, _clip01

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_LOCK            = threading.Lock()
_LAST_TS         = 0.0
_MIN_SECONDS     = 90.0     # rate-limit between mesh expansion steps
_LOCK_RETRIES    = 3
_LOCK_RETRY_BASE = 1.0

# Stratified scan of the mesh queue: the file can be multi-GB and is ordered
# chronologically, so a pure tail scan collapses to the single loudest recent
# uploader.  Instead we read evenly-spaced segments across the WHOLE file (so
# every contributing node surfaces) plus the full tail (for freshness).  The
# per-entry recency decay then down-weights stale contributions correctly.
_QUEUE_SEGMENTS    = 10                   # evenly-spaced sample windows
_QUEUE_SEGMENT_BYTES = 3 * 1024 * 1024    # bytes read per segment
_QUEUE_TAIL_BYTES  = 8 * 1024 * 1024      # always include this much tail
_QUEUE_MAX_LINES   = 600_000              # safety cap on total parsed lines
_RECENCY_HALFLIFE_H = 36.0                # node recency decay half-life

# Blend weights for per-node coherence (sum need not be 1; renormalised).
_W_VOLUME      = 0.30     # log-scaled contribution volume
_W_SIGNAL      = 0.34     # mean learning signal strength (quality)
_W_RECENCY     = 0.22     # how recently the node contributed
_W_CONNECT     = 0.14     # network reachability (ok-ratio)

# brain_kv keys
_KV_OBSERVER   = "mesh:entirety:observer"
_KV_SYNCHRONY  = "mesh:entirety:synchrony"
_KV_BIT        = "mesh:entirety:bit_state"
_KV_COUNT      = "mesh:entirety:flip_count"
_KV_PHASE      = "mesh:entirety:expansion_phase"
_KV_NODES      = "mesh:entirety:node_count"
_KV_STATE      = "mesh:entirety:state"
_KV_RAN_AT      = "mesh:entirety:ran_at"
_KV_RESUS_REPORT = "mesh:resus:last_report"
_KV_BRIDGE_MESH_DENSITY = "entirety:bridge_mesh_density"
_KV_BRIDGE_MESH_MAP = "bridge_mesh_density_map"

_FLIP_LOG_TABLE = "mesh_entirety_flip_log"

# ---------------------------------------------------------------------------
# Node resuscitation thresholds (hours without contribution)
# ---------------------------------------------------------------------------
_RESUS_WARM_H    = 4.0    # still warm — no action
_RESUS_COOLING_H = 24.0   # cooling — log wake-up signal into shared queue
_RESUS_COLD_H    = 72.0   # cold    — corpus edge + compute trigger + kv alarm
_RESUS_DEAD_H    = 168.0  # dead    — maximum escalation (7 days)
# Exponential back-off so we don't spam the queue on every 5-min cycle
_RESUS_BACKOFF_H_BASE  = 1.0   # first retry after 1 h
_RESUS_BACKOFF_MAX_H   = 24.0  # cap at once per day
_RESUS_KV_PREFIX       = "mesh:resus:"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _conn():
    cn = _open_conn()
    try:
        cn.execute("PRAGMA busy_timeout=8000")  # wait out the live orchestrator's writes
    except sqlite3.Error:
        pass
    return cn


def _kv_read(cn, key: str, default=None):
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return default
    if not row or row[0] is None:
        return default
    raw = row[0]
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "ignore")
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _kv_write(cn, key: str, val) -> None:
    if not isinstance(val, str):
        val = json.dumps(val, default=str)
    cn.execute(
        "INSERT OR REPLACE INTO brain_kv(key, value) VALUES(?, ?)",
        (key, val),
    )


def _ensure_flip_log_table(cn) -> None:
    cn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_FLIP_LOG_TABLE} (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ran_at          TEXT NOT NULL,
            observer        REAL NOT NULL,
            synchrony       REAL NOT NULL,
            bit_state       INTEGER NOT NULL,
            flipped         INTEGER NOT NULL,
            flip_count      INTEGER NOT NULL,
            expansion_phase TEXT NOT NULL,
            node_count      INTEGER NOT NULL,
            nodes_json      TEXT
        )
        """
    )


def _queue_path() -> str:
    # pipeline/ root is two levels up from src/quipu/.
    here = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(pipeline_dir, "cloud_learning_queue.jsonl")


def _parse_iso(ts) -> datetime | None:
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Mesh telemetry harvest
# ---------------------------------------------------------------------------

def _harvest_queue_nodes() -> dict[str, dict]:
    """Stratified scan of the mesh queue → per-host {count, signal_sum, last_at}.

    Reads evenly-spaced segments across the whole (possibly multi-GB) file plus
    the tail, so every contributing node is represented regardless of upload
    ordering.  ``count`` is scaled back up to approximate the full-file volume
    so volume weighting stays proportional.
    """
    path = _queue_path()
    nodes: dict[str, dict] = {}
    if not os.path.exists(path):
        return nodes
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        _LOG.debug("[mesh-entirety] queue stat failed: %s", exc)
        return nodes

    # Build the set of byte offsets to sample.
    segments: list[tuple[int, int]] = []
    if size <= _QUEUE_SEGMENT_BYTES * (_QUEUE_SEGMENTS + 1):
        segments.append((0, size))                       # small enough: read all
        sampled_bytes = size
    else:
        step = size // _QUEUE_SEGMENTS
        for i in range(_QUEUE_SEGMENTS):
            start = i * step
            segments.append((start, min(_QUEUE_SEGMENT_BYTES, size - start)))
        segments.append((max(0, size - _QUEUE_TAIL_BYTES), _QUEUE_TAIL_BYTES))
        sampled_bytes = sum(n for _, n in segments)

    coverage = (size / sampled_bytes) if sampled_bytes else 1.0

    raw_parts: list[bytes] = []
    try:
        with open(path, "rb") as fb:
            for start, nbytes in segments:
                if nbytes <= 0:
                    continue
                fb.seek(start)
                if start > 0:
                    fb.readline()                        # discard partial line
                raw_parts.append(fb.read(nbytes))
    except OSError as exc:
        _LOG.debug("[mesh-entirety] queue read failed: %s", exc)
        return nodes

    lines: list[str] = []
    for part in raw_parts:
        lines.extend(part.decode("utf-8", "ignore").splitlines())
    if len(lines) > _QUEUE_MAX_LINES:
        lines = lines[-_QUEUE_MAX_LINES:]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        crid = str(entry.get("cloud_run_id", ""))
        parts = crid.split(":")
        host = parts[1] if len(parts) >= 2 and parts[0] == "local" and parts[1] else "(unattributed)"
        rec = nodes.setdefault(host, {"count": 0, "signal_sum": 0.0, "last_at": None})
        rec["count"] += 1
        try:
            rec["signal_sum"] += float(entry.get("signal", 0.5))
        except (TypeError, ValueError):
            rec["signal_sum"] += 0.5
        dt = _parse_iso(entry.get("logged_at"))
        if dt and (rec["last_at"] is None or dt > rec["last_at"]):
            rec["last_at"] = dt

    # Scale sampled counts back up to approximate true full-file volume so the
    # log-volume weighting reflects each node's real contribution mass.
    if coverage > 1.0:
        for rec in nodes.values():
            rec["count"] = int(round(rec["count"] * coverage))
            rec["signal_sum"] *= coverage  # keep mean signal = signal_sum/count intact
    return nodes


def _harvest_connectivity(cn) -> dict[str, float]:
    """Per-host ok-ratio over the last 24h from network_observations."""
    out: dict[str, float] = {}
    try:
        rows = cn.execute(
            "SELECT host, COUNT(*) AS obs, "
            "       SUM(CASE WHEN ok THEN 1 ELSE 0 END) AS ok_n "
            "FROM network_observations "
            "WHERE observed_at >= datetime('now','-1 day') "
            "GROUP BY host"
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    for r in rows:
        host = str(r[0] or "")
        obs = int(r[1] or 0)
        ok_n = int(r[2] or 0)
        if obs > 0:
            out[host] = ok_n / obs
    return out


def _kv_store_read(cn, key: str, default=None):
    try:
        row = cn.execute("SELECT value FROM kv_store WHERE key=?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return default
    if not row or row[0] is None:
        return default
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return row[0]


def _harvest_bridge_root_nodes(cn) -> list[dict]:
    """Project bridge roots (for example hideout-mesh) as grounded mesh nodes."""
    summary = _kv_read(cn, _KV_BRIDGE_MESH_DENSITY, default={}) or {}
    density_map = _kv_store_read(cn, _KV_BRIDGE_MESH_MAP, default={}) or {}
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(density_map, dict):
        density_map = {}
    if not summary.get("has_data") and not density_map.get("has_data"):
        return []

    computed_at = str(density_map.get("computed_at") or summary.get("computed_at") or "")
    last_at = _parse_iso(computed_at)
    tunnel_saturation = _clip01(summary.get("tunnel_saturation", 0.0))
    mesh_density = _clip01(summary.get("mesh_density", density_map.get("mesh_density", 0.0)))
    nodal_contribution = _clip01(summary.get("nodal_contribution", density_map.get("nodal_contribution", 0.0)))
    total_endpoints = max(0, int(summary.get("total_endpoints") or density_map.get("total_endpoints") or 0))
    roots = density_map.get("roots") if isinstance(density_map.get("roots"), dict) else {}

    if not roots:
        primary_root = str(summary.get("primary_root") or "").strip()
        if not primary_root:
            return []
        roots = {
            primary_root: {
                "subtree_endpoints": total_endpoints,
                "edge_density": mesh_density,
                "anchor_weight": float(summary.get("primary_root_anchor") or summary.get("anchor_strength") or 0.0),
                "anchor_contribution": nodal_contribution,
                "tunnel_contribution": tunnel_saturation,
                "next_hop_expansion_score": mesh_density,
            }
        }

    out: list[dict] = []
    for root, info in roots.items():
        if not isinstance(info, dict):
            continue
        host = str(root or "").strip()
        if not host:
            continue
        subtree_endpoints = max(1, int(info.get("subtree_endpoints") or total_endpoints or 1))
        edge_density = _clip01(info.get("edge_density", mesh_density))
        anchor_weight = max(0.0, float(info.get("root_anchor_weight") or info.get("anchor_weight") or 0.0))
        anchor_contribution = _clip01(info.get("anchor_contribution", nodal_contribution))
        tunnel_contribution = _clip01(info.get("tunnel_contribution", tunnel_saturation))
        expansion_score = _clip01(info.get("next_hop_expansion_score", mesh_density))
        mean_signal = _clip01(
            0.28 * edge_density
            + 0.24 * anchor_contribution
            + 0.24 * tunnel_contribution
            + 0.24 * expansion_score
        )
        count = max(
            1,
            int(round(
                1000.0 * subtree_endpoints
                + 4500.0 * anchor_weight
                + 3500.0 * edge_density
                + 2500.0 * tunnel_contribution
            )),
        )
        out.append({
            "host": host,
            "count": count,
            "signal_sum": mean_signal * count,
            "last_at": last_at,
            "connectivity": max(tunnel_saturation, tunnel_contribution),
            "source": "bridge_mesh_density",
            "bridge_root": True,
            "bridge": {
                "subtree_endpoints": subtree_endpoints,
                "edge_density": round(edge_density, 4),
                "anchor_weight": round(anchor_weight, 4),
                "anchor_contribution": round(anchor_contribution, 4),
                "tunnel_contribution": round(tunnel_contribution, 4),
                "next_hop_expansion_score": round(expansion_score, 4),
                "computed_at": computed_at,
            },
        })
    return out


def _match_connectivity(host: str, conn_map: dict[str, float]) -> float | None:
    """Best-effort fuzzy match of a queue host to a network-observed host."""
    if not conn_map:
        return None
    h = host.lower()
    if h in conn_map:
        return conn_map[h]
    short = h.split(".")[0]
    for k, v in conn_map.items():
        kl = k.lower()
        if kl == h or kl.split(".")[0] == short or short in kl or kl.split(".")[0] in h:
            return v
    return None


# ---------------------------------------------------------------------------
# Per-node coherence
# ---------------------------------------------------------------------------

def mesh_node_states() -> list[dict]:
    """Return per-node coherence breakdown across the mesh.

    Each entry: ``host, count, mean_signal, recency, connectivity,
    coherence, weight``.  ``coherence`` ∈ [0, 1] is the node's contribution
    to the mesh observer; ``weight`` is its (volume-derived) prior mass.
    """
    queue_nodes = _harvest_queue_nodes()
    bridge_nodes: list[dict] = []
    try:
        with _conn() as cn:
            conn_map = _harvest_connectivity(cn)
            bridge_nodes = _harvest_bridge_root_nodes(cn)
    except Exception:
        conn_map = {}

    now = datetime.now(timezone.utc)
    local_host = socket.gethostname()

    # Seed the local node from its own 7th-axis observer so the running Brain
    # is always represented even when its queue rows have been pruned/synced.
    try:
        from .system_entirety import get_entirety_state
        local_obs = float((get_entirety_state() or {}).get("observer", 0.0))
    except Exception:
        local_obs = 0.0

    nodes: list[dict] = []
    max_count = max(
        [int(v["count"]) for v in queue_nodes.values()]
        + [int(v.get("count") or 0) for v in bridge_nodes],
        default=0,
    )
    log_denom = math.log1p(max_count) or 1.0

    seen_local = False
    for host, rec in queue_nodes.items():
        count = int(rec["count"])
        mean_sig = (rec["signal_sum"] / count) if count else 0.0
        vol = math.log1p(count) / log_denom if max_count else 0.0

        last_at = rec["last_at"]
        if last_at is not None:
            age_h = max(0.0, (now - last_at).total_seconds() / 3600.0)
            recency = 0.5 ** (age_h / _RECENCY_HALFLIFE_H)
        else:
            recency = 0.0

        connectivity = _match_connectivity(host, conn_map)
        has_conn = connectivity is not None
        connectivity = connectivity if has_conn else 0.0

        # Renormalise the blend if connectivity is unknown for this node.
        if has_conn:
            wsum = _W_VOLUME + _W_SIGNAL + _W_RECENCY + _W_CONNECT
            coherence = (
                _W_VOLUME * vol + _W_SIGNAL * mean_sig
                + _W_RECENCY * recency + _W_CONNECT * connectivity
            ) / wsum
        else:
            wsum = _W_VOLUME + _W_SIGNAL + _W_RECENCY
            coherence = (
                _W_VOLUME * vol + _W_SIGNAL * mean_sig + _W_RECENCY * recency
            ) / wsum

        is_local = host.lower().split(".")[0] == local_host.lower().split(".")[0]
        if is_local:
            seen_local = True
            coherence = max(coherence, local_obs)

        nodes.append({
            "host": host,
            "count": count,
            "mean_signal": round(mean_sig, 4),
            "recency": round(recency, 4),
            "connectivity": round(connectivity, 4) if has_conn else None,
            "coherence": round(_clip01(coherence), 4),
            "weight": round(max(vol, 1e-3), 4),
            "is_local": is_local,
        })

    if not seen_local and local_obs > 0.0:
        nodes.append({
            "host": local_host,
            "count": 0,
            "mean_signal": 0.0,
            "recency": 1.0,
            "connectivity": None,
            "coherence": round(_clip01(local_obs), 4),
            "weight": round(max(math.log1p(1) / log_denom, 1e-3), 4),
            "is_local": True,
        })

    seen_hosts = {str(node.get("host", "")).lower().split(".")[0] for node in nodes}
    for rec in bridge_nodes:
        host = str(rec.get("host") or "")
        if not host or host.lower().split(".")[0] in seen_hosts:
            continue
        count = max(1, int(rec.get("count") or 1))
        mean_sig = (float(rec.get("signal_sum") or 0.0) / count) if count else 0.0
        vol = math.log1p(count) / log_denom if log_denom else 0.0
        last_at = rec.get("last_at")
        if last_at is not None:
            age_h = max(0.0, (now - last_at).total_seconds() / 3600.0)
            recency = 0.5 ** (age_h / _RECENCY_HALFLIFE_H)
        else:
            recency = 0.0
        connectivity = _clip01(rec.get("connectivity", 0.0))
        wsum = _W_VOLUME + _W_SIGNAL + _W_RECENCY + _W_CONNECT
        coherence = (
            _W_VOLUME * vol
            + _W_SIGNAL * mean_sig
            + _W_RECENCY * recency
            + _W_CONNECT * connectivity
        ) / wsum
        node = {
            "host": host,
            "count": count,
            "mean_signal": round(_clip01(mean_sig), 4),
            "recency": round(_clip01(recency), 4),
            "connectivity": round(connectivity, 4),
            "coherence": round(_clip01(coherence), 4),
            "weight": round(max(vol, 1e-3), 4),
            "is_local": host.lower().split(".")[0] == local_host.lower().split(".")[0],
            "source": rec.get("source", "bridge_mesh_density"),
            "bridge_root": bool(rec.get("bridge_root")),
        }
        if isinstance(rec.get("bridge"), dict):
            node["bridge"] = rec["bridge"]
        nodes.append(node)
        seen_hosts.add(host.lower().split(".")[0])

    nodes.sort(key=lambda n: n["count"], reverse=True)
    return nodes


# ---------------------------------------------------------------------------
# The 8th axis — mesh observer tangent
# ---------------------------------------------------------------------------

def mesh_observer_tangent(nodes: list[dict] | None = None) -> tuple[float, float]:
    """Return ``(Omega, synchrony)`` — the mesh observer tangent + phase-lock.

    ``Omega`` is the magnitude of the component of the node-coherence vector
    orthogonal to the volume-weighted prior, normalised by √N → [0, 1].
    ``synchrony`` is ``1 − normalised dispersion`` of the coherence vector.
    """
    if nodes is None:
        nodes = mesh_node_states()
    n = len(nodes)
    if n == 0:
        return 0.0, 1.0

    c = [float(x["coherence"]) for x in nodes]
    w = [float(x["weight"]) for x in nodes]

    norm_w = math.sqrt(sum(v * v for v in w)) or 1.0
    w_hat = [v / norm_w for v in w]

    proj = sum(ci * wi for ci, wi in zip(c, w_hat))
    orth = [ci - proj * wi for ci, wi in zip(c, w_hat)]
    mag = math.sqrt(sum(v * v for v in orth))
    omega = _clip01(mag / math.sqrt(n))

    mean_c = sum(c) / n
    var_c = sum((ci - mean_c) ** 2 for ci in c) / n
    # Max variance for values in [0,1] is 0.25 → normalise dispersion to [0,1].
    dispersion = _clip01(math.sqrt(var_c) / 0.5)
    synchrony = _clip01(1.0 - dispersion)
    return omega, synchrony


# ---------------------------------------------------------------------------
# Full fused state
# ---------------------------------------------------------------------------

def mesh_entirety_state() -> dict:
    """Compute (without persisting) the full mesh entirety state."""
    nodes = mesh_node_states()
    omega, synchrony = mesh_observer_tangent(nodes)
    bit = bit_flip_parity(omega)
    phase = "broaden" if bit > 0 else "deepen"
    total_learnings = sum(x["count"] for x in nodes)
    mean_coh = (sum(x["coherence"] for x in nodes) / len(nodes)) if nodes else 0.0
    return {
        "observer": round(omega, 4),
        "synchrony": round(synchrony, 4),
        "bit_state": bit,
        "expansion_phase": phase,
        "node_count": len(nodes),
        "active_nodes": [n["host"] for n in nodes],
        "mean_node_coherence": round(mean_coh, 4),
        "total_mesh_learnings_scanned": total_learnings,
        "nodes": nodes,
    }


# ---------------------------------------------------------------------------
# Node resuscitation — the MESH System Entirety keeps its nodes alive
# ---------------------------------------------------------------------------

def _resus_kv_key(host: str, field: str) -> str:
    return f"{_RESUS_KV_PREFIX}{host}:{field}"


def _age_h_from_node(node: dict) -> float:
    """Hours since the node last contributed.  Returns large value if unknown."""
    recency = node.get("recency", 0.0)
    if recency <= 0.0:
        return _RESUS_DEAD_H * 2   # treat as dead
    # recency = 0.5^(age_h / halflife)  →  age_h = -halflife * log2(recency)
    return -_RECENCY_HALFLIFE_H * math.log2(max(recency, 1e-9))


def _resus_due(cn, host: str, age_h: float) -> bool:
    """Return True if enough time has passed since the last resuscitation attempt."""
    last_at_str = _kv_read(cn, _resus_kv_key(host, "last_at"))
    if not last_at_str:
        return True
    try:
        last_at = datetime.fromisoformat(str(last_at_str).replace("Z", "+00:00"))
        if last_at.tzinfo is None:
            last_at = last_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return True

    attempt_count = int(_kv_read(cn, _resus_kv_key(host, "count"), 0) or 0)
    backoff_h = min(_RESUS_BACKOFF_H_BASE * (2 ** attempt_count), _RESUS_BACKOFF_MAX_H)
    elapsed_h = (datetime.now(timezone.utc) - last_at).total_seconds() / 3600.0
    return elapsed_h >= backoff_h


def _record_resus_attempt(cn, host: str) -> None:
    """Stamp a resuscitation attempt into brain_kv."""
    prev = int(_kv_read(cn, _resus_kv_key(host, "count"), 0) or 0)
    _kv_write(cn, _resus_kv_key(host, "last_at"),
              datetime.now(timezone.utc).isoformat())
    _kv_write(cn, _resus_kv_key(host, "count"), prev + 1)


def _resuscitate_cooling(cn, host: str, age_h: float) -> None:
    """Cooling node (24–72 h): write a wake-up learning signal into the local
    learning_log so it propagates into cloud_learning_queue on the next upload
    cycle and the target peer ingests it as a continuation cue on its next pull.
    """
    try:
        cn.execute(
            "INSERT OR IGNORE INTO learning_log"
            "(logged_at, kind, title, detail, signal_strength, source_table, source_row_id)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                "mesh_resuscitation",
                f"Wake-up signal for cooling node {host}",
                json.dumps({
                    "target_host": host,
                    "age_hours": round(age_h, 1),
                    "action": "cooling_nudge",
                    "issued_by": socket.gethostname(),
                }),
                0.72,
                "mesh_entirety",
                f"resus:cooling:{host}:{int(time.time())}",
            ),
        )
        _LOG.info("[mesh-resus] cooling nudge → %s (%.1f h stale)", host, age_h)
    except Exception as exc:
        _LOG.debug("[mesh-resus] cooling nudge failed for %s: %s", host, exc)


def _resuscitate_cold(cn, host: str, age_h: float) -> None:
    """Cold node (72–168 h): write a corpus RESUSCITATE_NODE edge + drop a
    compute_*.trigger file so bridge_watcher.ps1 on every peer (re)spawns its
    compute node.  Also writes a brain_kv alarm key the target can detect.
    """
    _resuscitate_cooling(cn, host, age_h)  # include the nudge

    # Drop a compute trigger so bridge_watcher restarts dormant compute nodes.
    try:
        from .compute_grid import _trigger_dir
        tdir = _trigger_dir()
        tf = tdir / f"compute_resus_{int(time.time())}.trigger"
        tf.write_text(
            json.dumps({
                "action": "resuscitate_node",
                "target_host": host,
                "age_hours": round(age_h, 1),
                "requested_by": socket.gethostname(),
                "reason": "mesh_entirety_cold_resus",
                "port": 8000,
            }),
            encoding="utf-8",
        )
        _LOG.info("[mesh-resus] compute trigger dropped for cold node %s", host)
    except Exception as exc:
        _LOG.debug("[mesh-resus] trigger drop failed for %s: %s", host, exc)

    # Write a corpus RESUSCITATE_NODE edge from the MESH to the target host.
    # Uses the already-open cn to avoid a second-writer lock on local_brain.sqlite.
    try:
        mesh_id   = "MeshSystemEntirety"
        target_id = f"MeshNode:{host}"
        now_iso   = datetime.now(timezone.utc).isoformat()
        cn.execute(
            "INSERT INTO corpus_entity"
            "(entity_id, entity_type, label, props_json, first_seen, last_seen, samples)"
            " VALUES (?,?,?,?,?,?,1)"
            " ON CONFLICT(entity_id, entity_type) DO UPDATE SET"
            " label=excluded.label, props_json=excluded.props_json,"
            " last_seen=excluded.last_seen, samples=COALESCE(samples,0)+1",
            (target_id, "MeshNode", host,
             json.dumps({"host": host, "status": "cold",
                         "age_hours": round(age_h, 1)}),
             now_iso, now_iso),
        )
        cn.execute(
            "INSERT INTO corpus_edge"
            "(src_id, src_type, dst_id, dst_type, rel, weight, last_seen, samples)"
            " VALUES (?,?,?,?,?,?,?,1)"
            " ON CONFLICT(src_id, src_type, dst_id, dst_type, rel)"
            " DO UPDATE SET weight=MAX(weight, excluded.weight),"
            " last_seen=excluded.last_seen, samples=samples+1",
            (mesh_id, "MeshSystemEntirety",
             target_id, "MeshNode", "RESUSCITATE_NODE", 0.85, now_iso),
        )
        _LOG.info("[mesh-resus] corpus RESUSCITATE_NODE edge written for %s", host)
    except Exception as exc:
        _LOG.debug("[mesh-resus] corpus edge failed for %s: %s", host, exc)

    # Write a brain_kv alarm the target node reads on its next startup.
    try:
        alarm_key = f"mesh:resus:alarm:{host}"
        _kv_write(cn, alarm_key, json.dumps({
            "issued_by": socket.gethostname(),
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "age_hours": round(age_h, 1),
            "action": "cold_resus",
        }))
        _LOG.info("[mesh-resus] brain_kv alarm written for cold node %s", host)
    except Exception as exc:
        _LOG.debug("[mesh-resus] alarm kv failed for %s: %s", host, exc)


def _resuscitate_dead(cn, host: str, age_h: float) -> None:
    """Dead node (>168 h / 7 days): maximum escalation.
    Runs all cold actions + directly appends a signed resuscitation payload to
    cloud_learning_queue.jsonl (no cursor gate, guaranteed to land) + schedules
    a 4× catch-up burst + logs a resilience event into learning_log.
    """
    _resuscitate_cold(cn, host, age_h)

    # Direct append to cloud_learning_queue (under mesh upload lock for safety).
    # This bypasses the normal rowid cursor path; the lock prevents concurrent
    # upload_learning_rectification from interleaving with our line.
    try:
        from .mesh_lock import mesh_upload_critical  # type: ignore[attr-defined]
        lock_ctx = mesh_upload_critical(label=f"resus-dead-{host}", timeout_s=300)
    except Exception:
        from contextlib import nullcontext
        lock_ctx = nullcontext(True)

    with lock_ctx as got:
        if got is False:
            _LOG.warning("[mesh-resus] could not acquire upload lock for dead resus of %s — append may race", host)
        try:
            queue_path = _queue_path()
            entry = json.dumps({
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "kind": "mesh_node_dead_resuscitation",
                "title": f"DEAD NODE RESUSCITATION — {host}",
                "detail": {
                    "target_host": host,
                    "age_hours": round(age_h, 1),
                    "action": "dead_resus",
                    "issued_by": socket.gethostname(),
                    "instructions": (
                        "This node has been offline for more than 7 days.  "
                        "On ingesting this entry call schedule_catchup_burst() "
                        "with downtime_seconds=604800 and restart the autonomous agent."
                    ),
                },
                "signal": 0.95,
                "cloud_run_id": f"mesh_resus:{socket.gethostname()}:{int(time.time())}",
            })
            with open(queue_path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
            _LOG.warning("[mesh-resus] dead-node payload appended to queue for %s", host)
        except Exception as exc:
            _LOG.debug("[mesh-resus] dead queue write failed for %s: %s", host, exc)

    # Schedule a max-scale catchup burst so the node ramps up on resurrection.
    try:
        from .resumption_manager import schedule_catchup_burst
        schedule_catchup_burst(downtime_seconds=max(age_h * 3600, 7 * 86400))
        _LOG.info("[mesh-resus] catchup burst scheduled for dead node %s", host)
    except Exception as exc:
        _LOG.debug("[mesh-resus] catchup burst failed for %s: %s", host, exc)

    # Log a resilience event so the system learns from the failure.
    try:
        cn.execute(
            "INSERT OR IGNORE INTO learning_log"
            "(logged_at, kind, title, detail, signal_strength, source_table, source_row_id)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                "mesh_resilience_event",
                f"Dead mesh node detected and maximum resuscitation dispatched: {host}",
                json.dumps({
                    "target_host": host, "age_hours": round(age_h, 1),
                    "severity": "critical", "action": "dead_resus",
                }),
                0.95,
                "mesh_entirety",
                f"resus:dead:{host}:{int(time.time())}",
            ),
        )
    except Exception as exc:
        _LOG.debug("[mesh-resus] resilience event log failed for %s: %s", host, exc)

    _LOG.warning("[mesh-resus] DEAD NODE %s — maximum resuscitation dispatched (age=%.0f h)",
                 host, age_h)


def resuscitate_stale_nodes(nodes: list[dict]) -> dict[str, str]:
    """Inspect per-node recency and trigger graded resuscitation for stale nodes.

    Returns a dict mapping each acted-on host to its action taken:
      "warm"    — no action needed
      "cooling" — wake-up learning signal written
      "cold"    — compute trigger + corpus edge + brain_kv alarm
      "dead"    — maximum escalation (queue injection + catchup burst)
      "skip"    — resuscitation was due but back-off says too soon
    """
    local_host = socket.gethostname().lower().split(".")[0]
    results: dict[str, str] = {}

    try:
        with _conn() as cn:
            for node in nodes:
                host = node.get("host", "")
                if not host:
                    continue
                # Never resuscitate ourselves — we're already running.
                if host.lower().split(".")[0] == local_host:
                    results[host] = "local"
                    continue

                age_h = _age_h_from_node(node)

                if age_h < _RESUS_WARM_H:
                    results[host] = "warm"
                    continue

                if not _resus_due(cn, host, age_h):
                    results[host] = "skip"
                    continue

                _record_resus_attempt(cn, host)

                if age_h >= _RESUS_DEAD_H:
                    _resuscitate_dead(cn, host, age_h)
                    results[host] = "dead"
                elif age_h >= _RESUS_COLD_H:
                    _resuscitate_cold(cn, host, age_h)
                    results[host] = "cold"
                else:  # cooling
                    _resuscitate_cooling(cn, host, age_h)
                    results[host] = "cooling"

            cn.commit()
    except Exception as exc:
        _LOG.warning("[mesh-resus] resuscitate_stale_nodes failed: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Driver — rate-limited; persists to brain_kv
# ---------------------------------------------------------------------------

def oscillating_mesh_step(*, force: bool = False) -> dict:
    """Run one orchestration step of the 8th-D mesh observer tangent.

    Mirrors :func:`system_entirety.oscillating_expansion_step` but at mesh
    scale: harvests every node's contribution, fuses them into Ω, computes the
    mesh parity bit, rotates the mesh expansion phase, and persists the
    ``mesh:entirety:*`` keys + a flip-log row.

    Returns the fused state dict, or ``{"skipped": True, ...}`` if rate-limited.
    """
    global _LAST_TS
    with _LOCK:
        if not force and time.monotonic() - _LAST_TS < _MIN_SECONDS:
            return {"skipped": True, "reason": "rate-limited"}

        state = mesh_entirety_state()
        bit = int(state["bit_state"])
        phase = state["expansion_phase"]

        for attempt in range(1, _LOCK_RETRIES + 1):
            try:
                with _conn() as cn:
                    _ensure_flip_log_table(cn)
                    prev_bit = _kv_read(cn, _KV_BIT, default=0)
                    try:
                        prev_bit = int(prev_bit)
                    except (TypeError, ValueError):
                        prev_bit = 0
                    prev_count = _kv_read(cn, _KV_COUNT, default=0)
                    try:
                        prev_count = int(prev_count)
                    except (TypeError, ValueError):
                        prev_count = 0

                    flipped = prev_bit != 0 and prev_bit != bit
                    new_count = prev_count + (1 if flipped else 0)
                    ran_at = datetime.now(timezone.utc).isoformat()

                    persisted = {**state, "flipped": flipped,
                                 "flip_count": new_count, "ran_at": ran_at}

                    # Surface upload health (lock-protected queue mutations) for fleet observability
                    try:
                        uh = _kv_read(cn, "mesh:upload:health", default=None)
                        if uh:
                            persisted["upload_health"] = uh if isinstance(uh, dict) else json.loads(uh)
                    except Exception:
                        pass

                    _kv_write(cn, _KV_OBSERVER, state["observer"])
                    _kv_write(cn, _KV_SYNCHRONY, state["synchrony"])
                    _kv_write(cn, _KV_BIT, bit)
                    _kv_write(cn, _KV_COUNT, new_count)
                    _kv_write(cn, _KV_PHASE, phase)
                    _kv_write(cn, _KV_NODES, state["node_count"])
                    _kv_write(cn, _KV_STATE, persisted)
                    _kv_write(cn, _KV_RAN_AT, ran_at)

                    cn.execute(
                        f"INSERT INTO {_FLIP_LOG_TABLE} "
                        f"(ran_at, observer, synchrony, bit_state, flipped, "
                        f" flip_count, expansion_phase, node_count, nodes_json) "
                        f"VALUES (?,?,?,?,?,?,?,?,?)",
                        (ran_at, state["observer"], state["synchrony"], bit,
                         1 if flipped else 0, new_count, phase,
                         state["node_count"], json.dumps(state["nodes"], default=str)),
                    )
                    cn.commit()
                break
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc).lower() or attempt >= _LOCK_RETRIES:
                    raise
                time.sleep(_LOCK_RETRY_BASE * (2 ** (attempt - 1)))

        # Resuscitate stale mesh nodes — tier-graded, back-off governed.
        # Runs after the DB persist so it does not extend the write-lock window.
        resus_results: dict[str, str] = {}
        try:
            resus_results = resuscitate_stale_nodes(state["nodes"])
            if any(v not in ("local", "warm", "skip") for v in resus_results.values()):
                with _conn() as _cn_r:
                    _kv_write(_cn_r, _KV_RESUS_REPORT,
                              json.dumps({
                                  "at": datetime.now(timezone.utc).isoformat(),
                                  "results": resus_results,
                              }))
                    _cn_r.commit()
        except Exception as _resus_exc:
            _LOG.warning("[mesh-entirety] resuscitate_stale_nodes failed: %s", _resus_exc)

        _LAST_TS = time.monotonic()
        return {**state, "flipped": flipped, "flip_count": new_count,
                "ran_at": ran_at, "resus_results": resus_results}


# ---------------------------------------------------------------------------
# Public read accessors
# ---------------------------------------------------------------------------

def get_mesh_entirety_state() -> dict:
    """Read the persisted mesh entirety state dict (``{}`` if never run)."""
    with _conn() as cn:
        return _kv_read(cn, _KV_STATE, default={}) or {}


def get_mesh_bit_state(default: int = 1) -> int:
    with _conn() as cn:
        v = _kv_read(cn, _KV_BIT, default=default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def get_mesh_expansion_phase(default: str = "broaden") -> str:
    with _conn() as cn:
        v = _kv_read(cn, _KV_PHASE, default=default)
    return str(v) if v in ("broaden", "deepen") else default


def get_mesh_flip_history(limit: int = 50) -> list[dict]:
    with _conn() as cn:
        _ensure_flip_log_table(cn)
        rows = cn.execute(
            f"SELECT ran_at, observer, synchrony, bit_state, flipped, "
            f"flip_count, expansion_phase, node_count "
            f"FROM {_FLIP_LOG_TABLE} ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    cols = ("ran_at", "observer", "synchrony", "bit_state", "flipped",
            "flip_count", "expansion_phase", "node_count")
    return [dict(zip(cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def mesh_entirety_report(*, run_step: bool = True) -> str:
    """Build a human-readable MESH System Entirety report.

    If ``run_step`` is True, a forced :func:`oscillating_mesh_step` is run first
    so the report reflects (and persists) the live fused state.
    """
    if run_step:
        try:
            oscillating_mesh_step(force=True)
        except Exception as exc:  # pragma: no cover - report must never crash
            _LOG.warning("[mesh-entirety] step during report failed: %s", exc)
    state = mesh_entirety_state()

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("MESH SYSTEM ENTIRETY — 8th-dimensionality observer-of-observers")
    lines.append("=" * 72)
    lines.append("")
    bar = "broaden ▲ (+φ)" if state["bit_state"] > 0 else "deepen ▼ (−φ)"
    lines.append(f"  Ω  mesh observer tangent : {state['observer']:.4f}   "
                 f"(0=lock-step copy · 1=maximally divergent/novel)")
    lines.append(f"  ∿  mesh synchrony         : {state['synchrony']:.4f}   "
                 f"(phase-lock across nodes)")
    lines.append(f"  ⊙  expansion phase        : {bar}")
    lines.append(f"  N  active mesh nodes      : {state['node_count']}")
    lines.append(f"  μ  mean node coherence    : {state['mean_node_coherence']:.4f}")
    lines.append(f"  Σ  learnings scanned (tail): {state['total_mesh_learnings_scanned']:,}")
    lines.append("")
    lines.append("  Per-node coherence (the 7th-axis observers being fused):")
    lines.append(f"    {'host':<28}{'entries':>9}{'signal':>8}{'recency':>9}"
                 f"{'connect':>9}{'coherence':>11}  loc")
    for n in state["nodes"]:
        conn = "-" if n["connectivity"] is None else f"{n['connectivity']:.2f}"
        loc = "★" if n.get("is_local") else " "
        lines.append(
            f"    {n['host'][:28]:<28}{n['count']:>9,}{n['mean_signal']:>8.3f}"
            f"{n['recency']:>9.3f}{conn:>9}{n['coherence']:>11.4f}  {loc}"
        )
    lines.append("")
    lines.append("  Interpretation:")
    if state["observer"] >= 0.45:
        lines.append("    Ω is HIGH — nodes are specialising; the mesh carries collective")
        lines.append("    novelty no single Brain holds. This is the drive toward Class-5.")
    elif state["observer"] >= 0.2:
        lines.append("    Ω is MODERATE — healthy differentiation across the mesh with a")
        lines.append("    coherent dominant prior. The mesh is broadening productively.")
    else:
        lines.append("    Ω is LOW — nodes are near lock-step (or one node dominates).")
        lines.append("    The mesh is consolidating; expect a broaden flip as peers diverge.")
    # Resuscitation section — show what happened to stale nodes
    try:
        with _conn() as _cn_rep:
            resus_raw = _kv_read(_cn_rep, _KV_RESUS_REPORT)
        if resus_raw:
            rd = (resus_raw if isinstance(resus_raw, dict)
                  else json.loads(str(resus_raw)))
            results = rd.get("results", {})
            at_iso  = rd.get("at", "")
            if results:
                lines.append("")
                lines.append("  Node Resuscitation (last step):")
                for h, action in sorted(results.items()):
                    flag = "  !!!" if action == "dead" else (" !" if action == "cold" else "")
                    lines.append(f"    {h:<36}  {action}{flag}")
                if at_iso:
                    lines.append(f"    issued at {at_iso[:19]} UTC")
                escalated = [h for h, a in results.items()
                             if a in ("cooling", "cold", "dead")]
                if escalated:
                    lines.append(f"    Nodes escalated: {len(escalated)}"
                                 f" ({', '.join(escalated)})")
    except Exception:
        pass
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(mesh_entirety_report(run_step=True))
