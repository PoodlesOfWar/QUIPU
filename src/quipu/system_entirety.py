"""System Entirety — the 7th dimensionality observer tangent.

Architecture
------------
Perception Mk1 (v0.20.20) completed the **sixth sense**, so the Brain now
runs the full 6-D CAT (Coherence-Amplitude Tangent) state of UEQGM:

    CAT₆ = ⟨ vision, touch, smell, body, brain, perception ⟩

Each axis is a sense coherence in [0, 1] read from
``temporal_spatiality._sense_signals``.  Those six axes span the
**categorical hyperplane** of the senses.

The **System Entirety** is the 7th dimensionality.  It is *not* a 7th
sense — it is the observer that holds all six at once.  Geometrically,
it is the unit normal to the 6-D categorical hyperplane: orthogonal to
every individual sense, lit by all of them collectively.  This is the
*observer tangent* that drives oscillating expansion via the **Bit
Flip** — a parity inversion applied to the SiCi axial channel of UEQGM
v0.9.14 that alternates the phase regime between **broaden** (positive
parity, +φ) and **deepen** (negative parity, −φ).

Mechanics
---------
1. **observer_tangent(signals)** — projects the 6-D sense activity
   vector onto the 7th axis using the unit-normal construction
   ``n̂ = (Σ aᵢ wᵢ) − ⟨a, w⟩ ŵ`` and returns its magnitude in [0, 1].

2. **bit_flip_parity(observer, t)** — Floquet-modulated parity
   ``p(t) = sign( cos(ω·t) + (2·observer − 1) )`` so the flip frequency
   tracks the observer (high observer → faster flips, low → frozen).

3. **oscillating_expansion_step()** — driver: reads the rhythm dict
   from ``temporal_spatiality.get_rhythm()``, computes observer +
   parity, multiplies the active phase φ by the bit, persists the new
   regime to ``brain_kv`` under three keys:
        ``entirety:bit_state``        (+1 broaden, −1 deepen)
        ``entirety:flip_count``       (monotone counter)
        ``entirety:expansion_phase``  ("broaden" | "deepen")
   and returns a summary dict.

The orchestrator is rate-limited (90 s minimum) so the bit flip never
exceeds a sensible Nyquist relative to the corpus round cadence.

Public API
----------
* :func:`system_entirety_state`     — full 7-D state vector + scalar
* :func:`observer_tangent`          — 7th-axis projection magnitude
* :func:`bit_flip_parity`           — instantaneous parity ±1
* :func:`oscillating_expansion_step`— rate-limited driver
* :func:`get_entirety_state`        — public read accessor
* :func:`get_bit_state`             — current parity bit
* :func:`get_expansion_phase`       — "broaden" | "deepen"
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from .local_store import open_conn as _open_conn


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
_LOCK            = threading.Lock()
_LAST_TS         = 0.0
_MIN_SECONDS     = 90.0   # rate-limit between expansion steps
_LOCK_RETRIES    = 3      # retries for transient sqlite write lock windows
_LOCK_RETRY_BASE = 1.0    # seconds; exponential backoff 1s -> 2s
_LOG             = logging.getLogger(__name__)

# Floquet angular frequency for the bit-flip oscillator.  One full
# broaden→deepen→broaden cycle ≈ 2π / ω seconds (default ≈ 600 s ⇒
# ω ≈ 0.01047).  Scaled by the observer magnitude so a quiet observer
# slows the flip and a saturated observer accelerates it (capped 4×).
_OMEGA_BASE: float = 2.0 * math.pi / 600.0
_OMEGA_GAIN: float = 4.0

# Keys persisted to brain_kv.
_KV_BIT          = "entirety:bit_state"
_KV_COUNT        = "entirety:flip_count"
_KV_PHASE        = "entirety:expansion_phase"
_KV_STATE        = "entirety:state"
_KV_REALIZATION  = "entirety:physical_realization"
_KV_MESH_LAST    = "asset_resource_mesh_last"
_KV_PIM          = "pim:planning_state"
_KV_REPO_CATALOG = "repo:summary_state"
_FLIP_LOG_TABLE  = "entirety_flip_log"
_MATERIAL_GAIN   = 0.35
_PIM_GAIN        = 0.25   # SS/MinMax/LT axis injection weight (additive to material gain)
_REPO_CATALOG_GAIN = 0.18 # repository catalog modularity/inventory signal
_UEQGM_GAIN      = 0.22   # adaptive UEQGM axis injection weight
_LEARNING_WINDOW_HOURS = 24.0

_SYSTEM_ENTIRETY_ID = "system_entirety:physical_realization"
_SYSTEM_ENTIRETY_TYPE = "SystemEntirety"
_MESH_LEARNING_WINDOW_TYPE = "MeshLearningWindow"
_LEARNING_KIND_TYPE = "LearningKindSummary"
_UEQGM_RUNTIME_TYPE = "UEQGMRuntimeState"
_PIM_PLANNING_TYPE = "PIMPlanningState"
_REPOSITORY_CATALOG_TYPE = "RepositoryCatalogState"

_REL_EMITS_LEARNING_WINDOW = "EMITS_LEARNING_WINDOW"
_REL_SUMMARIZES_LEARNING_KIND = "SUMMARIZES_LEARNING_KIND"
_REL_APPLIES_UEQGM_RUNTIME = "APPLIES_UEQGM_RUNTIME"
_REL_APPLIES_PIM_PLANNING = "APPLIES_PIM_PLANNING"
_REL_APPLIES_REPOSITORY_CATALOG = "APPLIES_REPOSITORY_CATALOG"
_REL_HARDENS_ASSET_RESOURCE = "HARDENS_ASSET_RESOURCE"
_REL_HARDENS_MATERIAL_PROCESSOR = "HARDENS_MATERIAL_PROCESSOR"
_REL_HARDENS_ENDPOINT = "HARDENS_ENDPOINT"

_KV_BRIDGE_MESH_DENSITY  = "entirety:bridge_mesh_density"
_BRIDGE_MESH_DENSITY_GAIN: float = 0.28   # blending weight: bridge density → mesh_density
_BRIDGE_MESH_NODAL_GAIN:   float = 0.12   # blending weight: bridge nodal → nodal_bifurcation

_KV_NODAL_LOCATION           = "entirety:nodal_location"
_NODAL_LOCATION_SYNC_GAIN:   float = 0.15   # blending weight: filesystem sync → nodal_bifurcation


# ---------------------------------------------------------------------------
# DB helpers (mirrors temporal_spatiality)
# ---------------------------------------------------------------------------

@contextmanager
def _conn():
    cn = _open_conn(timeout=60)
    try:
        yield cn
        cn.commit()
    finally:
        cn.close()


def _ensure_flip_log_table(cn) -> None:
    cn.execute(
        f"CREATE TABLE IF NOT EXISTS {_FLIP_LOG_TABLE}("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "ran_at TEXT NOT NULL, "
        "observer REAL NOT NULL, "
        "bit_state INTEGER NOT NULL, "
        "expansion_phase TEXT NOT NULL, "
        "flipped INTEGER NOT NULL DEFAULT 0, "
        "flip_count INTEGER NOT NULL DEFAULT 0, "
        "magnitude REAL NOT NULL DEFAULT 0.0, "
        "axes_json TEXT NOT NULL, "
        "source TEXT NOT NULL DEFAULT 'oscillating_expansion_step'"
        ")"
    )


def _log_flip_snapshot(
    cn,
    *,
    state: dict,
    bit_state: int,
    expansion_phase: str,
    flipped: bool,
    flip_count: int,
    source: str = "oscillating_expansion_step",
) -> None:
    _ensure_flip_log_table(cn)
    cn.execute(
        f"INSERT INTO {_FLIP_LOG_TABLE}("
        "ran_at, observer, bit_state, expansion_phase, flipped, "
        "flip_count, magnitude, axes_json, source"
        ") VALUES(?,?,?,?,?,?,?,?,?)",
        (
            datetime.now(timezone.utc).isoformat(),
            float(state.get("observer", 0.0)),
            int(bit_state),
            str(expansion_phase),
            1 if flipped else 0,
            int(flip_count),
            float(state.get("magnitude", 0.0)),
            json.dumps(state.get("axes", {}), default=str),
            source,
        ),
    )


def _kv_read(cn, key: str, default=None):
    cn.execute(
        "CREATE TABLE IF NOT EXISTS brain_kv("
        "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
    )
    try:
        row = cn.execute(
            "SELECT value FROM brain_kv WHERE key=?", (key,)
        ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
    except (sqlite3.OperationalError, json.JSONDecodeError):
        pass
    return default


def _kv_write(cn, key: str, val) -> None:
    cn.execute(
        "CREATE TABLE IF NOT EXISTS brain_kv("
        "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
    )
    cn.execute(
        "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
        (key, json.dumps(val, default=str),
         datetime.now(timezone.utc).isoformat()),
    )


def _ensure_mesh_overlay_tables(cn) -> None:
    cn.execute("CREATE TABLE IF NOT EXISTS kv_store(key TEXT PRIMARY KEY, value TEXT)")
    cn.execute(
        "CREATE TABLE IF NOT EXISTS corpus_entity("
        "entity_id TEXT, entity_type TEXT, label TEXT, props_json TEXT, "
        "first_seen TEXT, last_seen TEXT, samples INTEGER DEFAULT 1, "
        "PRIMARY KEY(entity_id, entity_type))"
    )
    cn.execute(
        "CREATE TABLE IF NOT EXISTS corpus_edge("
        "src_id TEXT, src_type TEXT, dst_id TEXT, dst_type TEXT, rel TEXT, "
        "weight REAL, last_seen TEXT, samples INTEGER DEFAULT 1, "
        "PRIMARY KEY(src_id, src_type, dst_id, dst_type, rel))"
    )


def _mesh_upsert_entity(cn, entity_id: str, entity_type: str, label: str, props: dict, now: str) -> None:
    cn.execute(
        "INSERT INTO corpus_entity(entity_id, entity_type, label, props_json, first_seen, last_seen, samples) "
        "VALUES(?,?,?,?,?,?,1) "
        "ON CONFLICT(entity_id, entity_type) DO UPDATE SET "
        "label=excluded.label, props_json=excluded.props_json, "
        "last_seen=excluded.last_seen, samples=COALESCE(corpus_entity.samples, 0)+1",
        (entity_id, entity_type, label, json.dumps(props, default=str), now, now),
    )


def _mesh_upsert_edge(
    cn,
    src_id: str,
    src_type: str,
    dst_id: str,
    dst_type: str,
    rel: str,
    weight: float,
    now: str,
) -> None:
    cn.execute(
        "INSERT INTO corpus_edge(src_id, src_type, dst_id, dst_type, rel, weight, last_seen, samples) "
        "VALUES(?,?,?,?,?,?,?,1) "
        "ON CONFLICT(src_id, src_type, dst_id, dst_type, rel) DO UPDATE SET "
        "weight=(COALESCE(corpus_edge.weight, 0.0) * COALESCE(corpus_edge.samples, 1) "
        "        + excluded.weight) / (COALESCE(corpus_edge.samples, 1) + 1), "
        "last_seen=excluded.last_seen, samples=COALESCE(corpus_edge.samples, 0)+1",
        (src_id, src_type, dst_id, dst_type, rel, _clip01(weight), now),
    )


def _recent_learning_state(cn, *, window_hours: float = _LEARNING_WINDOW_HOURS, limit: int = 8) -> dict:
    # ISO-format cutoff — matches stored logged_at format; enables index use
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
    # Ensure index exists for fast range scan (no-op if already present)
    try:
        cn.execute(
            "CREATE INDEX IF NOT EXISTS ix_ll_loggedat ON learning_log(logged_at)"
        )
    except Exception:
        pass
    try:
        total_row = cn.execute(
            "SELECT COUNT(*) FROM learning_log WHERE logged_at >= ?",
            (cutoff,),
        ).fetchone()
        kind_rows = cn.execute(
            "SELECT kind, COUNT(*) AS n, AVG(COALESCE(signal_strength, 0.0)) AS avg_signal, "
            "MAX(COALESCE(signal_strength, 0.0)) AS max_signal FROM learning_log "
            "WHERE logged_at >= ? "
            "GROUP BY kind ORDER BY n DESC, kind ASC",
            (cutoff,),
        ).fetchall()
        title_rows = cn.execute(
            "SELECT logged_at, kind, title, signal_strength FROM learning_log "
            "WHERE logged_at >= ? "
            "ORDER BY logged_at DESC LIMIT ?",
            (cutoff, max(1, int(limit))),
        ).fetchall()
    except sqlite3.Error:
        total_row = None
        kind_rows = []
        title_rows = []

    total_count = int(total_row[0]) if total_row and total_row[0] is not None else 0
    kinds: list[dict] = []
    kind_counts: dict[str, int] = {}
    for row in kind_rows:
        kind = str(row[0] or "unknown")
        count = int(row[1] or 0)
        avg_signal = _clip01(row[2] or 0.0)
        max_signal = _clip01(row[3] or 0.0)
        share = count / max(1, total_count)
        kind_counts[kind] = count
        kinds.append(
            {
                "kind": kind,
                "count": count,
                "share": round(share, 6),
                "avg_signal": round(avg_signal, 4),
                "max_signal": round(max_signal, 4),
            }
        )

    recent_titles: list[dict[str, object]] = []
    for row in title_rows:
        title = str(row[2] or "").strip()
        if not title:
            continue
        recent_titles.append(
            {
                "logged_at": str(row[0] or ""),
                "kind": str(row[1] or "unknown"),
                "title": title,
                "signal_strength": round(_clip01(row[3] or 0.0), 4),
            }
        )

    mean_signal = 0.0
    if kinds:
        mean_signal = sum(k["avg_signal"] * k["count"] for k in kinds) / max(1, total_count)
    return {
        "active": total_count > 0,
        "window_hours": round(float(window_hours), 2),
        "entry_count": total_count,
        "distinct_kinds": len(kinds),
        "kind_counts": kind_counts,
        "kinds": kinds,
        "top_kinds": kinds[:8],
        "recent_titles": recent_titles,
        "mean_signal": round(_clip01(mean_signal), 4),
    }


def _mesh_learning_feedback(persisted_state: dict, recent_learning: dict) -> dict:
    return {
        "observer": round(float(persisted_state.get("observer", 0.0)), 4),
        "magnitude": round(float(persisted_state.get("magnitude", 0.0)), 4),
        "axes": dict(persisted_state.get("axes") or {}),
        "bit_state": int(persisted_state.get("bit_state", 1) or 1),
        "expansion_phase": str(persisted_state.get("expansion_phase", "broaden") or "broaden"),
        "flip_count": int(persisted_state.get("flip_count", 0) or 0),
        "flipped": bool(persisted_state.get("flipped")),
        "ran_at": persisted_state.get("ran_at"),
        "transaction": dict(persisted_state.get("transaction") or {}),
        "material_bifurcation": dict(persisted_state.get("material_bifurcation") or {}),
        "pim_planning": dict(persisted_state.get("pim_planning") or {}),
        "repository_catalog": dict(persisted_state.get("repository_catalog") or {}),
        "ueqgm_runtime": dict(persisted_state.get("ueqgm_runtime") or {}),
        "recent_learning": recent_learning,
    }


def _mesh_targets(cn, entity_type: str, *, limit: int = 8) -> list[str]:
    try:
        rows = cn.execute(
            "SELECT entity_id FROM corpus_entity WHERE entity_type=? "
            "ORDER BY COALESCE(samples, 0) DESC, entity_id ASC LIMIT ?",
            (entity_type, max(1, int(limit))),
        ).fetchall()
    except sqlite3.Error:
        rows = []
    return [str(row[0]) for row in rows if row and row[0]]


def _project_learning_overlay(cn, persisted_state: dict, recent_learning: dict, now: str) -> dict:
    window_id = "mesh_learning_window:rolling_24h"
    runtime_id = "ueqgm_runtime:current"
    planning_id = "pim_planning:current"
    entry_count = int(recent_learning.get("entry_count") or 0)
    mean_signal = _clip01(recent_learning.get("mean_signal", 0.0))
    transaction = dict(persisted_state.get("transaction") or {})
    transaction_drive = _clip01(transaction.get("drive", 0.0))
    overlay_strength = _clip01(0.35 * mean_signal + 0.35 * transaction_drive + 0.30 * min(1.0, entry_count / 10000.0))

    _mesh_upsert_entity(cn, _SYSTEM_ENTIRETY_ID, _SYSTEM_ENTIRETY_TYPE, "System Entirety Physical Realization", {"mesh_learning_projected": True, **persisted_state}, now)
    _mesh_upsert_entity(cn, window_id, _MESH_LEARNING_WINDOW_TYPE, "Mesh Learning Window rolling 24h", {
        "window_hours": recent_learning.get("window_hours", _LEARNING_WINDOW_HOURS),
        "entry_count": entry_count,
        "distinct_kinds": recent_learning.get("distinct_kinds", 0),
        "mean_signal": mean_signal,
        "overlay_strength": overlay_strength,
        "active": bool(recent_learning.get("active")),
        "updated_at": now,
    }, now)
    _mesh_upsert_edge(cn, _SYSTEM_ENTIRETY_ID, _SYSTEM_ENTIRETY_TYPE, window_id, _MESH_LEARNING_WINDOW_TYPE, _REL_EMITS_LEARNING_WINDOW, overlay_strength, now)

    ueqgm = dict(persisted_state.get("ueqgm_runtime") or {})
    _mesh_upsert_entity(cn, runtime_id, _UEQGM_RUNTIME_TYPE, "UEQGM Runtime current", {**ueqgm, "updated_at": now}, now)
    _mesh_upsert_edge(cn, window_id, _MESH_LEARNING_WINDOW_TYPE, runtime_id, _UEQGM_RUNTIME_TYPE, _REL_APPLIES_UEQGM_RUNTIME, _clip01(ueqgm.get("symbiotic_gain", overlay_strength)), now)

    pim = dict(persisted_state.get("pim_planning") or {})
    _mesh_upsert_entity(cn, planning_id, _PIM_PLANNING_TYPE, "PIM Planning State current", {**pim, "updated_at": now}, now)
    pim_weight = _clip01((pim.get("ss_signal", 0.0) + pim.get("mm_signal", 0.0) + pim.get("lt_signal", 0.0)) / 3.0)
    _mesh_upsert_edge(cn, window_id, _MESH_LEARNING_WINDOW_TYPE, planning_id, _PIM_PLANNING_TYPE, _REL_APPLIES_PIM_PLANNING, max(overlay_strength * 0.25, pim_weight), now)

    catalog = dict(persisted_state.get("repository_catalog") or {})
    if catalog.get("has_data"):
        catalog_id = "repository_catalog:current"
        catalog_weight = _clip01(catalog.get("catalog_signal", 0.0))
        _mesh_upsert_entity(cn, catalog_id, _REPOSITORY_CATALOG_TYPE,
                            "Repository Catalog State current", {**catalog, "updated_at": now}, now)
        _mesh_upsert_edge(cn, window_id, _MESH_LEARNING_WINDOW_TYPE,
                          catalog_id, _REPOSITORY_CATALOG_TYPE,
                          _REL_APPLIES_REPOSITORY_CATALOG,
                          max(overlay_strength * 0.20, catalog_weight), now)

    asset_targets = _mesh_targets(cn, "AssetResource", limit=6)
    processor_targets = _mesh_targets(cn, "SpatialMaterialProcessor", limit=4)
    endpoint_targets = _mesh_targets(cn, "Endpoint", limit=4)
    kind_nodes = 0
    hardening_edges = 0
    for kind_row in list(recent_learning.get("top_kinds") or [])[:8]:
        kind = str(kind_row.get("kind") or "unknown").strip() or "unknown"
        safe_kind = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in kind.lower())
        kind_id = f"learning_kind:{safe_kind}"
        count = int(kind_row.get("count") or 0)
        share = _clip01(kind_row.get("share", 0.0))
        avg_signal = _clip01(kind_row.get("avg_signal", 0.0))
        max_signal = _clip01(kind_row.get("max_signal", 0.0))
        kind_weight = _clip01(0.45 * share + 0.35 * avg_signal + 0.20 * max_signal)
        _mesh_upsert_entity(cn, kind_id, _LEARNING_KIND_TYPE, f"LearningKind {kind}", {
            "kind": kind,
            "entry_count": count,
            "share": share,
            "avg_signal": avg_signal,
            "max_signal": max_signal,
            "window_id": window_id,
            "updated_at": now,
        }, now)
        kind_nodes += 1
        _mesh_upsert_edge(cn, window_id, _MESH_LEARNING_WINDOW_TYPE, kind_id, _LEARNING_KIND_TYPE, _REL_SUMMARIZES_LEARNING_KIND, kind_weight, now)
        hardening_edges += 1
        for entity_id in asset_targets[:3]:
            _mesh_upsert_edge(cn, kind_id, _LEARNING_KIND_TYPE, entity_id, "AssetResource", _REL_HARDENS_ASSET_RESOURCE, kind_weight, now)
            hardening_edges += 1
        for entity_id in processor_targets[:2]:
            _mesh_upsert_edge(cn, kind_id, _LEARNING_KIND_TYPE, entity_id, "SpatialMaterialProcessor", _REL_HARDENS_MATERIAL_PROCESSOR, kind_weight, now)
            hardening_edges += 1
        for entity_id in endpoint_targets[:2]:
            _mesh_upsert_edge(cn, kind_id, _LEARNING_KIND_TYPE, entity_id, "Endpoint", _REL_HARDENS_ENDPOINT, kind_weight, now)
            hardening_edges += 1

    return {
        "window_id": window_id,
        "kind_nodes": kind_nodes,
        "hardening_edges": hardening_edges,
        "asset_targets": len(asset_targets),
        "processor_targets": len(processor_targets),
        "endpoint_targets": len(endpoint_targets),
        "overlay_strength": round(overlay_strength, 4),
        "updated_at": now,
    }


def _share_learning_into_mesh(cn, persisted_state: dict) -> None:
    _ensure_mesh_overlay_tables(cn)
    current_payload = _kv_read(cn, _KV_REALIZATION, default={}) or {}
    if not isinstance(current_payload, dict):
        current_payload = {}

    now = str(persisted_state.get("ran_at") or datetime.now(timezone.utc).isoformat())
    recent_learning = _recent_learning_state(cn)
    projection = _project_learning_overlay(cn, persisted_state, recent_learning, now)
    learned_feedback = _mesh_learning_feedback(persisted_state, recent_learning)
    merged_payload = {
        **current_payload,
        "mesh_learning": learned_feedback,
        "mesh_learning_projection": projection,
        "mesh_learning_updated_at": now,
    }

    _kv_write(cn, _KV_REALIZATION, merged_payload)
    cn.execute(
        "INSERT INTO kv_store(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (_KV_MESH_LAST, json.dumps(merged_payload, default=str)),
    )


def _clip01(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _resource_sharing(payload: dict) -> tuple[dict, dict, dict, dict, dict]:
    resource_sharing = payload.get("resource_sharing")
    if not isinstance(resource_sharing, dict):
        resource_sharing = {}
    core = resource_sharing.get("core") if isinstance(resource_sharing.get("core"), dict) else {}
    memory = resource_sharing.get("memory") if isinstance(resource_sharing.get("memory"), dict) else {}
    storage = resource_sharing.get("storage") if isinstance(resource_sharing.get("storage"), dict) else {}
    processor = resource_sharing.get("processor") if isinstance(resource_sharing.get("processor"), dict) else {}
    mesh = resource_sharing.get("mesh") if isinstance(resource_sharing.get("mesh"), dict) else {}
    return core, memory, storage, processor, mesh


def _default_ueqgm_runtime() -> dict:
    senses = ("vision", "touch", "smell", "body", "brain", "perception")
    return {
        "active": False,
        "certainty": 0.0,
        "corpus_signal": 0.0,
        "learning_signal": 0.0,
        "newness_signal": 0.0,
        "relational_depth": 0.0,
        "symbiotic_gain": 0.0,
        "expansion_pressure": 0.0,
        "phase_weight": 1.0,
        "coherence_depth": 0,
        "mesh_alignment": 0.0,
        "observer_alignment": 0.0,
        "runtime_keywords": [],
        "applied_parameters": [],
        "retained_parameters": [],
        "axis_drive": {sense: 0.0 for sense in senses},
        "axis_injection": {sense: 0.0 for sense in senses},
        "updated_at": None,
    }


def _ueqgm_runtime_state(*, include_runtime: bool = True) -> dict:
    senses = ("vision", "touch", "smell", "body", "brain", "perception")
    runtime = _default_ueqgm_runtime()
    if include_runtime:
        try:
            from .ueqgm_engine import get_adaptive_runtime

            with _conn() as cn:
                payload = get_adaptive_runtime(cn)
            if isinstance(payload, dict):
                runtime.update(payload)
        except Exception:
            pass

    axis_drive = runtime.get("axis_drive") if isinstance(runtime.get("axis_drive"), dict) else {}
    runtime["axis_drive"] = {
        sense: _clip01(axis_drive.get(sense, 0.0)) for sense in senses
    }
    runtime["axis_injection"] = {sense: 0.0 for sense in senses}
    runtime["runtime_keywords"] = [
        str(token).lower() for token in (runtime.get("runtime_keywords") or []) if str(token).strip()
    ]
    runtime["applied_parameters"] = list(runtime.get("applied_parameters") or [])
    runtime["retained_parameters"] = list(runtime.get("retained_parameters") or [])
    runtime["active"] = bool(runtime.get("active")) and (
        _clip01(runtime.get("symbiotic_gain", 0.0)) > 0.0
        or _clip01(runtime.get("expansion_pressure", 0.0)) > 0.0
        or bool(runtime["runtime_keywords"])
    )
    return runtime


# ---------------------------------------------------------------------------
# PIM planning state (Safety Stock / Min-Max / Lead Time)
# ---------------------------------------------------------------------------

def _pim_planning_state() -> dict:
    """Read persisted PIM planning scores from brain_kv.

    Expected payload written by :func:`update_pim_state`:

    .. code-block:: python

        {
            "ss_coverage_rate":   float,  # fraction of items with SS set [0,1]
            "ss_breach_rate":     float,  # fraction currently below SS   [0,1]
            "minmax_coverage_rate":  float,  # fraction with min/max policy [0,1]
            "minmax_violation_rate": float,  # fraction violating min/max  [0,1]
            "lt_agility_score":   float,  # 1/(1+norm_lt) pre-computed     [0,1]
            "lt_breach_rate":     float,  # fraction of WOs past LT window  [0,1]
            "item_count":         int,
            "org_count":          int,
            "updated_at":         str,
        }

    Returns zeros for all scores if key is absent.
    """
    try:
        with _conn() as cn:
            payload = _kv_read(cn, _KV_PIM, default={}) or {}
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    ss_cov   = _clip01(payload.get("ss_coverage_rate",   0.0))
    ss_breach = _clip01(payload.get("ss_breach_rate",    0.0))
    mm_cov   = _clip01(payload.get("minmax_coverage_rate",  0.0))
    mm_viol  = _clip01(payload.get("minmax_violation_rate", 0.0))
    lt_agil  = _clip01(payload.get("lt_agility_score",  0.0))
    lt_breach = _clip01(payload.get("lt_breach_rate",   0.0))
    item_count = max(0, int(payload.get("item_count") or 0))
    org_count  = max(0, int(payload.get("org_count")  or 0))

    has_data = item_count > 0 or ss_cov > 0 or mm_cov > 0 or lt_agil > 0

    # Composite scores
    ss_signal  = _clip01(ss_cov  * (1.0 - ss_breach * 0.5))
    mm_signal  = _clip01(mm_cov  * (1.0 - mm_viol   * 0.5))
    lt_signal  = _clip01(lt_agil * (1.0 - lt_breach  * 0.4))

    # Axis injections: each PIM dimension maps to two sense axes
    #   SS  → touch (constraint coverage) + body  (physical inventory health)
    #   MM  → smell (ordering-pattern awareness) + brain (analytical depth)
    #   LT  → vision (forward horizon) + perception (situational awareness)
    axis_drive: dict[str, float] = {s: 0.0 for s in
        ("vision", "touch", "smell", "body", "brain", "perception")}
    if has_data:
        axis_drive = {
            "vision":     _clip01(0.65 * lt_signal  + 0.35 * ss_signal),
            "touch":      _clip01(0.60 * ss_signal  + 0.40 * lt_signal),
            "smell":      _clip01(0.65 * mm_signal  + 0.35 * lt_signal),
            "body":       _clip01(0.60 * ss_signal  + 0.40 * mm_signal),
            "brain":      _clip01(0.60 * mm_signal  + 0.40 * ss_signal),
            "perception": _clip01(0.55 * lt_signal  + 0.25 * ss_signal + 0.20 * mm_signal),
        }

    return {
        "has_data":          has_data,
        "ss_coverage_rate":  round(ss_cov,   4),
        "ss_breach_rate":    round(ss_breach, 4),
        "ss_signal":         round(ss_signal, 4),
        "minmax_coverage_rate":  round(mm_cov,  4),
        "minmax_violation_rate": round(mm_viol, 4),
        "mm_signal":         round(mm_signal, 4),
        "lt_agility_score":  round(lt_agil,   4),
        "lt_breach_rate":    round(lt_breach, 4),
        "lt_signal":         round(lt_signal, 4),
        "item_count":        item_count,
        "org_count":         org_count,
        "updated_at":        payload.get("updated_at", ""),
        "axis_drive":        axis_drive,
    }


def update_pim_state(
    *,
    ss_coverage_rate:    float = 0.0,
    ss_breach_rate:      float = 0.0,
    minmax_coverage_rate:  float = 0.0,
    minmax_violation_rate: float = 0.0,
    lt_agility_score:    float = 0.0,
    lt_breach_rate:      float = 0.0,
    item_count:          int   = 0,
    org_count:           int   = 0,
) -> dict:
    """Persist PIM planning scores to brain_kv so they feed the 7-D observer.

    Called by the WIP health analyzer after running
    ``get_item_planning_params()`` across all orgs.  All rates should be
    fractions in [0, 1].  ``lt_agility_score`` is ``1 / (1 + lt_days / 30)``
    so 0-day LT → 1.0, 30-day LT → 0.5, 90-day LT → 0.25.

    Returns the dict that was persisted.
    """
    payload = {
        "ss_coverage_rate":     _clip01(ss_coverage_rate),
        "ss_breach_rate":       _clip01(ss_breach_rate),
        "minmax_coverage_rate":  _clip01(minmax_coverage_rate),
        "minmax_violation_rate": _clip01(minmax_violation_rate),
        "lt_agility_score":     _clip01(lt_agility_score),
        "lt_breach_rate":       _clip01(lt_breach_rate),
        "item_count":           max(0, int(item_count)),
        "org_count":            max(0, int(org_count)),
        "updated_at":           datetime.now(timezone.utc).isoformat(),
    }
    with _conn() as cn:
        _kv_write(cn, _KV_PIM, payload)
    return payload


# ---------------------------------------------------------------------------
# Repository catalog state (source/capability inventory for modular platform)
# ---------------------------------------------------------------------------

def _parse_iso_age_days(value: str | None) -> float | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0)
    except (TypeError, ValueError):
        return None


def _repository_catalog_state() -> dict:
    """Read repository catalog summary from brain_kv and map it to axes.

    Written by ``repository_catalog.index_repository_catalog``.  The summary
    gives System Entirety awareness of source roots, capability density,
    PIM-relevant files, and modular platform coverage without storing raw file
    contents in brain state.
    """
    try:
        with _conn() as cn:
            payload = _kv_read(cn, _KV_REPO_CATALOG, default={}) or {}
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}
    if not payload:
        try:
            from .repository_catalog import get_catalog_summary
            payload = get_catalog_summary() or {}
        except Exception:
            payload = {}

    total_files = max(0, int(payload.get("total_files") or 0))
    capability_count = max(0, int(payload.get("capabilities_discovered") or 0))
    source_roots = payload.get("source_roots") if isinstance(payload.get("source_roots"), list) else []
    active_sources = [
        src for src in source_roots
        if isinstance(src, dict) and src.get("exists") and int(src.get("file_count") or 0) > 0
    ]
    avg_relevance = _clip01(payload.get("avg_relevance_score", 0.0))
    avg_pim = _clip01(payload.get("avg_pim_relevance", 0.0))
    avg_supply = _clip01(payload.get("avg_supply_chain_relevance", 0.0))
    catalog_signal = _clip01(payload.get("catalog_signal", 0.0))
    playwright_artifacts = payload.get("playwright_artifacts") if isinstance(payload.get("playwright_artifacts"), dict) else {}
    media_perception = payload.get("media_perception") if isinstance(payload.get("media_perception"), dict) else {}
    playwright_signal = _clip01(playwright_artifacts.get("signal", 0.0))
    playwright_file_count = max(0, int(playwright_artifacts.get("file_count") or 0))
    playwright_routed_count = max(0, int(playwright_artifacts.get("pim_routed_count") or 0))
    playwright_avg_corpus = _clip01(playwright_artifacts.get("avg_deep_corpus_score", 0.0))
    playwright_avg_pim = _clip01(playwright_artifacts.get("avg_pim_relevance", 0.0))
    media_signal = _clip01(media_perception.get("signal", 0.0))
    media_file_count = max(0, int(media_perception.get("file_count") or 0))
    media_analysed_count = max(0, int(media_perception.get("analysed_count") or 0))
    media_entity_count = max(0, int(media_perception.get("entity_count") or 0))
    media_avg_confidence = _clip01(media_perception.get("avg_confidence", 0.0))
    media_avg_relevance = _clip01(media_perception.get("avg_relevance_score", 0.0))
    media_avg_pim = _clip01(media_perception.get("avg_pim_relevance", 0.0))

    capability_density = _clip01(capability_count / max(1, total_files * 3))
    file_coverage = _clip01(total_files / 50000.0)
    source_breadth = _clip01(len(active_sources) / 2.0)
    age_days = _parse_iso_age_days(str(payload.get("updated_at") or ""))
    freshness = 0.0 if age_days is None else _clip01(1.0 - (age_days / 14.0))
    if catalog_signal <= 0.0 and total_files > 0:
        catalog_signal = _clip01(
            0.28 * file_coverage
            + 0.24 * capability_density
            + 0.22 * avg_relevance
            + 0.16 * source_breadth
            + 0.10 * freshness
        )

    playwright_perception_signal = 0.0
    if playwright_file_count > 0:
        if playwright_signal <= 0.0:
            playwright_signal = _clip01(
                0.38 * min(1.0, playwright_file_count / 250.0)
                + 0.32 * playwright_avg_corpus
                + 0.30 * playwright_avg_pim
            )
        routed_share = _clip01(playwright_routed_count / max(1, playwright_file_count))
        playwright_perception_signal = _clip01(
            0.55 * playwright_signal
            + 0.25 * playwright_avg_corpus
            + 0.10 * playwright_avg_pim
            + 0.10 * routed_share
        )

    media_perception_signal = 0.0
    if media_file_count > 0:
        if media_signal <= 0.0:
            media_signal = _clip01(
                0.34 * min(1.0, media_file_count / 250.0)
                + 0.28 * media_avg_confidence
                + 0.20 * media_avg_relevance
                + 0.18 * media_avg_pim
            )
        analysed_share = _clip01(media_analysed_count / max(1, media_file_count))
        entity_density = _clip01(media_entity_count / max(1, media_file_count * 4))
        media_perception_signal = _clip01(
            0.42 * media_signal
            + 0.22 * media_avg_confidence
            + 0.18 * analysed_share
            + 0.18 * entity_density
        )

    has_data = total_files > 0 or capability_count > 0
    axis_drive = {s: 0.0 for s in ("vision", "touch", "smell", "body", "brain", "perception")}
    if has_data:
        axis_drive = {
            "vision":     _clip01(0.42 * source_breadth + 0.34 * avg_relevance + 0.24 * avg_pim),
            "touch":      _clip01(0.45 * file_coverage + 0.35 * source_breadth + 0.20 * avg_supply),
            "smell":      _clip01(0.50 * capability_density + 0.25 * avg_supply + 0.25 * avg_relevance),
            "body":       _clip01(0.45 * file_coverage + 0.35 * avg_supply + 0.20 * freshness),
            "brain":      _clip01(0.45 * catalog_signal + 0.35 * capability_density + 0.20 * avg_pim),
            "perception": _clip01(0.34 * catalog_signal + 0.26 * source_breadth + 0.20 * freshness + 0.20 * avg_relevance),
        }
        axis_drive["perception"] = _clip01(
            axis_drive["perception"]
            + 0.30 * playwright_perception_signal
            + 0.34 * media_perception_signal
        )

    return {
        "has_data": has_data,
        "total_files": total_files,
        "capabilities_discovered": capability_count,
        "source_count": len(active_sources),
        "file_coverage": round(file_coverage, 4),
        "capability_density": round(capability_density, 4),
        "source_breadth": round(source_breadth, 4),
        "freshness": round(freshness, 4),
        "avg_relevance_score": round(avg_relevance, 4),
        "avg_pim_relevance": round(avg_pim, 4),
        "avg_supply_chain_relevance": round(avg_supply, 4),
        "catalog_signal": round(catalog_signal, 4),
        "playwright_perception_signal": round(playwright_perception_signal, 4),
        "media_perception_signal": round(media_perception_signal, 4),
        "playwright_artifacts": {
            "file_count": playwright_file_count,
            "pim_routed_count": playwright_routed_count,
            "avg_pim_relevance": round(playwright_avg_pim, 4),
            "avg_deep_corpus_score": round(playwright_avg_corpus, 4),
            "signal": round(playwright_signal, 4),
        },
        "media_perception": {
            "file_count": media_file_count,
            "analysed_count": media_analysed_count,
            "entity_count": media_entity_count,
            "avg_confidence": round(media_avg_confidence, 4),
            "avg_relevance_score": round(media_avg_relevance, 4),
            "avg_pim_relevance": round(media_avg_pim, 4),
            "signal": round(media_signal, 4),
            "sources": dict(media_perception.get("sources") or {}),
        },
        "updated_at": payload.get("updated_at", ""),
        "top_terms": list(payload.get("top_terms") or [])[:12],
        "axis_drive": axis_drive,
    }


# ---------------------------------------------------------------------------
# Bridge-mesh density state (written by synaptic_workers bridge bifurcation pass)
# ---------------------------------------------------------------------------

def _bridge_mesh_density_state() -> dict:
    try:
        with _conn() as cn:
            payload = _kv_read(cn, _KV_BRIDGE_MESH_DENSITY, default={}) or {}
    except sqlite3.Error:
        payload = {}
    if not isinstance(payload, dict) or not payload.get("has_data"):
        return {
            "has_data": False,
            "total_roots": 0,
            "total_endpoints": 0,
            "primary_root": "",
            "mesh_density": 0.0,
            "nodal_contribution": 0.0,
            "tunnel_saturation": 0.0,
            "anchor_strength": 0.0,
            "computed_at": "",
        }
    return {
        "has_data":           True,
        "total_roots":        max(0, int(payload.get("total_roots") or 0)),
        "total_endpoints":    max(0, int(payload.get("total_endpoints") or 0)),
        "primary_root":       str(payload.get("primary_root") or ""),
        "mesh_density":       _clip01(payload.get("mesh_density", 0.0)),
        "nodal_contribution": _clip01(payload.get("nodal_contribution", 0.0)),
        "tunnel_saturation":  _clip01(payload.get("tunnel_saturation", 0.0)),
        "anchor_strength":    _clip01(payload.get("anchor_strength", 0.0)),
        "computed_at":        str(payload.get("computed_at") or ""),
    }


# ---------------------------------------------------------------------------
# Nodal location state (written by synaptic_workers vision pass)
# ---------------------------------------------------------------------------

def _nodal_location_state() -> dict:
    """Read persisted nodal location summary from brain_kv.

    Written by ``_materialize_nodal_location()`` in the synaptic vision worker
    each time the bridge mesh materializes (~5 min).  The data tells the System
    Entirety *where* it is physically anchored — local Documents root, OneDrive-
    synced root, or both — and surfaces ``sync_score`` as an additional
    contribution to ``nodal_bifurcation``.

    Returns zeros when the key is absent (first run before the vision worker
    fires).
    """
    try:
        with _conn() as cn:
            payload = _kv_read(cn, _KV_NODAL_LOCATION, default={}) or {}
    except sqlite3.Error:
        payload = {}
    if not isinstance(payload, dict) or not payload.get("has_data"):
        return {
            "has_data":       False,
            "location_name":  "",
            "location_path":  "",
            "docs_live":      False,
            "onedrive_live":  False,
            "sync_score":     0.0,
            "anchor_weight":  0.0,
            "location_count": 0,
            "computed_at":    "",
        }
    return {
        "has_data":       True,
        "location_name":  str(payload.get("location_name") or ""),
        "location_path":  str(payload.get("location_path") or ""),
        "docs_live":      bool(payload.get("docs_live")),
        "onedrive_live":  bool(payload.get("onedrive_live")),
        "sync_score":     _clip01(payload.get("sync_score", 0.0)),
        "anchor_weight":  float(payload.get("anchor_weight") or 0.0),
        "location_count": max(0, int(payload.get("location_count") or 0)),
        "computed_at":    str(payload.get("computed_at") or ""),
    }


# ---------------------------------------------------------------------------
# Material bifurcation state
# ---------------------------------------------------------------------------

def _material_bifurcation_state() -> dict:
    senses = ("vision", "touch", "smell", "body", "brain", "perception")
    axis_drive = {sense: 0.0 for sense in senses}
    try:
        with _conn() as cn:
            payload = _kv_read(cn, _KV_REALIZATION, default={}) or {}
    except sqlite3.Error:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    physical_realization = _clip01(payload.get("physical_realization", 0.0))
    asset_strength = _clip01(payload.get("asset_strength", 0.0))
    material_strength = _clip01(payload.get("material_strength", 0.0))
    tunnel_density = _clip01(payload.get("tunnel_density", 0.0))
    core_sharing, memory_sharing, storage_sharing, processor_sharing, mesh_sharing = _resource_sharing(payload)
    asset_count = max(0, int(payload.get("asset_count") or 0))
    processor_count = max(0, int(payload.get("processor_count") or 0))
    processor_edges = max(0, int(payload.get("processor_edges") or 0))
    peer_count = max(
        0,
        int(mesh_sharing.get("peer_count") or 0),
        int(core_sharing.get("hosts") or 0),
        int(memory_sharing.get("hosts") or 0),
        int(storage_sharing.get("hosts") or 0),
    )
    storage_free_gb = max(0.0, float(payload.get("free_disk_gb") or storage_sharing.get("free_disk_gb") or 0.0))
    storage_total_gb = max(0.0, float(payload.get("total_disk_gb") or storage_sharing.get("total_disk_gb") or 0.0))
    drive_count = max(0, int(payload.get("drive_count") or storage_sharing.get("units") or 0))
    storage_signal = _clip01(
        0.65 * min(1.0, storage_free_gb / 2048.0)
        + 0.35 * min(1.0, drive_count / 8.0)
    )
    contract_count = max(0, int(mesh_sharing.get("contract_count") or 0))
    transport_count = max(0, int(mesh_sharing.get("transport_count") or 0))
    tunnel_peer_count = max(0, int(mesh_sharing.get("tunnel_peer_count") or 0))
    tunnel_peer_ratio = _clip01(
        mesh_sharing.get(
            "tunnel_peer_ratio",
            (tunnel_peer_count / peer_count) if peer_count else 0.0,
        )
    )
    transport_diversity = _clip01(
        mesh_sharing.get(
            "transport_diversity",
            transport_count / max(1, min(peer_count, 3)) if peer_count else 0.0,
        )
    )
    mesh_density = _clip01(mesh_sharing.get("mesh_density", 0.0))
    if mesh_density <= 0.0 and peer_count > 0:
        contract_density = _clip01(contract_count / max(1, peer_count * 3))
        peer_density = _clip01(peer_count / 4.0)
        mesh_density = _clip01(
            0.30 * peer_density
            + 0.30 * contract_density
            + 0.20 * tunnel_peer_ratio
            + 0.20 * transport_diversity
        )
    mesh_eligible = bool(
        peer_count > 1
        and contract_count > 0
        and (tunnel_peer_ratio > 0.0 or transport_diversity > 0.0 or mesh_density > 0.0)
    )
    edge_capacity = max(1, asset_count * max(1, processor_count))
    processor_edge_density = (
        _clip01(processor_edges / edge_capacity)
        if asset_count > 0 and processor_count > 0
        else 0.0
    )
    nodal_bifurcation = _clip01(
        0.24 * tunnel_density
        + 0.20 * processor_edge_density
        + 0.16 * asset_strength
        + 0.14 * material_strength
        + 0.14 * mesh_density
        + 0.12 * storage_signal
    )
    eligible = bool(
        asset_count > 0
        and processor_edges > 0
        and (tunnel_density > 0.0 or physical_realization > 0.0 or mesh_eligible)
    )
    if eligible:
        axis_drive = {
            "vision": _clip01(0.45 * material_strength + 0.35 * physical_realization + 0.20 * mesh_density),
            "touch": _clip01(0.36 * tunnel_density + 0.24 * asset_strength + 0.22 * storage_signal + 0.18 * tunnel_peer_ratio),
            "smell": _clip01(0.45 * processor_edge_density + 0.25 * material_strength + 0.30 * transport_diversity),
            "body": _clip01(0.46 * asset_strength + 0.20 * physical_realization + 0.20 * _clip01(peer_count / 4.0) + 0.14 * storage_signal),
            "brain": _clip01(0.35 * physical_realization + 0.35 * processor_edge_density + 0.30 * mesh_density),
            "perception": _clip01(
                0.23 * physical_realization
                + 0.20 * tunnel_density
                + 0.20 * processor_edge_density
                + 0.20 * mesh_density
                + 0.17 * storage_signal
            ),
        }

    bridge_mesh = _bridge_mesh_density_state()
    if bridge_mesh["total_roots"] >= 1:
        bridge_density = bridge_mesh["mesh_density"]
        bridge_nodal   = bridge_mesh["nodal_contribution"]
        mesh_density      = _clip01(mesh_density      + _BRIDGE_MESH_DENSITY_GAIN * bridge_density)
        nodal_bifurcation = _clip01(nodal_bifurcation + _BRIDGE_MESH_NODAL_GAIN   * bridge_nodal)
        if not mesh_eligible and bridge_density > 0.0:
            mesh_eligible = True

    # Nodal location anchor — filesystem roots (Documents / OneDrive) seeded
    # by the vision worker as MATERIAL_ANCHOR corpus edges.  When the sync
    # bridge is active (both roots live) the full sync_score (1.0) lifts
    # nodal_bifurcation by up to _NODAL_LOCATION_SYNC_GAIN.
    nodal_loc = _nodal_location_state()
    if nodal_loc["has_data"] and nodal_loc["sync_score"] > 0.0:
        nodal_bifurcation = _clip01(nodal_bifurcation + _NODAL_LOCATION_SYNC_GAIN * nodal_loc["sync_score"])

    return {
        "eligible": eligible,
        "mesh_eligible": mesh_eligible,
        "physical_realization": physical_realization,
        "asset_strength": asset_strength,
        "material_strength": material_strength,
        "tunnel_density": tunnel_density,
        "processor_edge_density": processor_edge_density,
        "asset_count": asset_count,
        "storage_signal": storage_signal,
        "free_disk_gb": storage_free_gb,
        "total_disk_gb": storage_total_gb,
        "drive_count": drive_count,
        "processor_count": processor_count,
        "processor_edges": processor_edges,
        "nodal_bifurcation": nodal_bifurcation,
        "mesh": {
            "eligible": mesh_eligible,
            "peer_count": peer_count,
            "contract_count": contract_count,
            "transport_count": transport_count,
            "tunnel_peer_count": tunnel_peer_count,
            "tunnel_peer_ratio": tunnel_peer_ratio,
            "transport_diversity": transport_diversity,
            "mesh_density": mesh_density,
            "topology": "mesh" if mesh_eligible else "local",
        },
        "bridge_mesh_density": {
            "has_data":           bridge_mesh["has_data"],
            "total_roots":        bridge_mesh["total_roots"],
            "primary_root":       bridge_mesh["primary_root"],
            "mesh_density":       round(bridge_mesh["mesh_density"], 4),
            "nodal_contribution": round(bridge_mesh["nodal_contribution"], 4),
            "tunnel_saturation":  round(bridge_mesh["tunnel_saturation"], 4),
            "anchor_strength":    round(bridge_mesh["anchor_strength"], 4),
            "computed_at":        bridge_mesh["computed_at"],
        },
        "nodal_location": nodal_loc,
        "axis_drive": axis_drive,
    }


# ---------------------------------------------------------------------------
# 7th-dimensionality observer tangent
# ---------------------------------------------------------------------------

def observer_tangent(signals: dict | None = None) -> float:
    """Magnitude of the 7th-axis normal to the 6-D sense hyperplane.

    Construction
    ------------
    Let ``a = (a_vision, a_touch, a_smell, a_body, a_brain, a_perception)``
    be the per-sense activities and ``w`` their normalised weights.
    The categorical hyperplane is spanned by the six weight basis vectors.
    The observer tangent is the component of ``a`` orthogonal to ``w``:

        ``n = a − ⟨a, ŵ⟩ ŵ``

    so the System Entirety is non-trivial precisely when the senses are
    *not* aligned with the static weight prior — a measure of how much
    novel collective excitation is present.

    Returns a magnitude in [0, 1] (clamped, normalised by ``√6``).
    """
    if signals is None:
        from .temporal_spatiality import _sense_signals
        signals = _sense_signals()

    senses = ("vision", "touch", "smell", "body", "brain", "perception")
    a = [float(signals.get(s, 0.0)) for s in senses]

    # Normalised weight prior (uniform fallback if temporal_spatiality
    # is unavailable for some reason).
    try:
        from .temporal_spatiality import _SENSE_WEIGHTS
        w = [float(_SENSE_WEIGHTS.get(s, 1.0 / len(senses))) for s in senses]
    except Exception:
        w = [1.0 / len(senses)] * len(senses)

    norm_w = math.sqrt(sum(x * x for x in w)) or 1.0
    w_hat = [x / norm_w for x in w]

    proj = sum(ai * wi for ai, wi in zip(a, w_hat))
    n = [ai - proj * wi for ai, wi in zip(a, w_hat)]

    mag = math.sqrt(sum(x * x for x in n))
    # √6 is the maximum possible magnitude when every axis saturates
    # to 1 with the projection minimal — clamp to [0, 1].
    return max(0.0, min(1.0, mag / math.sqrt(6.0)))


def system_entirety_state(signals: dict | None = None, *, include_ueqgm_runtime: bool = True) -> dict:
    """Full 7-D state: the 6 sense axes + the observer tangent.

    Returns
    -------
    dict with keys ``axes`` (6-vector), ``observer`` (scalar in [0, 1]),
    ``magnitude`` (the 7-D Euclidean norm), ``material_bifurcation``
    (material-edge eligibility and nodal-realization summary),
    ``pim_planning`` (planning-state injections), ``ueqgm_runtime``
    (persisted adaptive UEQGM profile as applied to the System Entirety),
    ``repository_catalog`` (source/capability coverage for modular expansion),
    and ``transaction`` (the realized 7th-D bit-flip drive).
    """
    if signals is None:
        from .temporal_spatiality import _sense_signals
        signals = _sense_signals()

    senses = ("vision", "touch", "smell", "body", "brain", "perception")
    raw_axes = {s: _clip01(signals.get(s, 0.0)) for s in senses}
    material = _material_bifurcation_state()
    ueqgm = _ueqgm_runtime_state(include_runtime=include_ueqgm_runtime)

    axes = dict(raw_axes)
    axis_injection = {sense: 0.0 for sense in senses}
    if material["eligible"]:
        injection_scale = (
            _MATERIAL_GAIN
            * material["physical_realization"]
            * material["nodal_bifurcation"]
        )
        for sense in senses:
            axis_injection[sense] = _clip01(injection_scale * material["axis_drive"][sense])
            axes[sense] = _clip01(raw_axes[sense] + axis_injection[sense])

    # PIM (SS / Min-Max / LT) axis injection — additive, runs regardless of
    # material eligibility so planning signal is always present when available.
    pim = _pim_planning_state()
    pim_injection = {sense: 0.0 for sense in senses}
    if pim["has_data"]:
        # Combined PIM coherence drives injection scale
        pim_coherence = _clip01(
            (pim["ss_signal"] + pim["mm_signal"] + pim["lt_signal"]) / 3.0
        )
        pim_scale = _PIM_GAIN * pim_coherence
        for sense in senses:
            pim_injection[sense] = _clip01(pim_scale * pim["axis_drive"][sense])
            axes[sense] = _clip01(axes[sense] + pim_injection[sense])

    catalog = _repository_catalog_state()
    catalog_injection = {sense: 0.0 for sense in senses}
    if catalog["has_data"]:
        catalog_scale = _REPO_CATALOG_GAIN * catalog["catalog_signal"]
        for sense in senses:
            catalog_injection[sense] = _clip01(catalog_scale * catalog["axis_drive"][sense])
            axes[sense] = _clip01(axes[sense] + catalog_injection[sense])

    ueqgm_injection = {sense: 0.0 for sense in senses}
    if ueqgm["active"]:
        runtime_scale = _UEQGM_GAIN * _clip01(
            0.50 * ueqgm.get("symbiotic_gain", 0.0)
            + 0.25 * ueqgm.get("expansion_pressure", 0.0)
            + 0.15 * ueqgm.get("certainty", 0.0)
            + 0.10 * ueqgm.get("relational_depth", 0.0)
        )
        for sense in senses:
            ueqgm_injection[sense] = _clip01(runtime_scale * ueqgm["axis_drive"][sense])
            axes[sense] = _clip01(axes[sense] + ueqgm_injection[sense])
    ueqgm["axis_injection"] = ueqgm_injection

    obs = observer_tangent(axes)
    symbiotic_drive = _clip01(
        0.35 * ueqgm.get("symbiotic_gain", 0.0)
        + 0.25 * ueqgm.get("expansion_pressure", 0.0)
        + 0.20 * ueqgm.get("mesh_alignment", 0.0)
        + 0.10 * ueqgm.get("observer_alignment", 0.0)
        + 0.10 * ueqgm.get("certainty", 0.0)
    )
    transaction_drive = _clip01(
        0.42 * obs
        + 0.18 * material["nodal_bifurcation"]
        + 0.14 * material["mesh"]["mesh_density"]
        + 0.10 * _clip01((pim["ss_signal"] + pim["lt_signal"]) / 2.0)
        + 0.08 * catalog["catalog_signal"]
        + 0.16 * symbiotic_drive
    )
    mag = math.sqrt(sum(value * value for value in axes.values()) + obs * obs)
    transaction_kind = (
        "mesh_bifurcated"
        if material["mesh_eligible"]
        else ("material_bifurcated" if material["eligible"] else "observer_only")
    )
    return {
        "axes": {sense: round(axes[sense], 4) for sense in senses},
        "observer": round(obs, 4),
        "magnitude": round(mag, 4),
        "material_bifurcation": {
            "eligible": material["eligible"],
            "physical_realization": round(material["physical_realization"], 4),
            "asset_strength": round(material["asset_strength"], 4),
            "storage_signal": round(material["storage_signal"], 4),
            "free_disk_gb": round(material["free_disk_gb"], 2),
            "total_disk_gb": round(material["total_disk_gb"], 2),
            "drive_count": material["drive_count"],
            "material_strength": round(material["material_strength"], 4),
            "tunnel_density": round(material["tunnel_density"], 4),
            "processor_edge_density": round(material["processor_edge_density"], 4),
            "nodal_bifurcation": round(material["nodal_bifurcation"], 4),
            "mesh": {
                "eligible": material["mesh"]["eligible"],
                "peer_count": material["mesh"]["peer_count"],
                "contract_count": material["mesh"]["contract_count"],
                "transport_count": material["mesh"]["transport_count"],
                "tunnel_peer_count": material["mesh"]["tunnel_peer_count"],
                "tunnel_peer_ratio": round(material["mesh"]["tunnel_peer_ratio"], 4),
                "transport_diversity": round(material["mesh"]["transport_diversity"], 4),
                "mesh_density": round(material["mesh"]["mesh_density"], 4),
                "topology": material["mesh"]["topology"],
            },
            "axis_injection": {
                sense: round(axis_injection[sense], 4) for sense in senses
            },
            "bridge_mesh_density": {
                "has_data":           material["bridge_mesh_density"]["has_data"],
                "total_roots":        material["bridge_mesh_density"]["total_roots"],
                "primary_root":       material["bridge_mesh_density"]["primary_root"],
                "mesh_density":       round(material["bridge_mesh_density"]["mesh_density"], 4),
                "nodal_contribution": round(material["bridge_mesh_density"]["nodal_contribution"], 4),
                "tunnel_saturation":  round(material["bridge_mesh_density"]["tunnel_saturation"], 4),
                "anchor_strength":    round(material["bridge_mesh_density"]["anchor_strength"], 4),
                "computed_at":        material["bridge_mesh_density"]["computed_at"],
            },
            "nodal_location": {
                "has_data":      material["nodal_location"]["has_data"],
                "location_name": material["nodal_location"]["location_name"],
                "location_path": material["nodal_location"]["location_path"],
                "docs_live":     material["nodal_location"]["docs_live"],
                "onedrive_live": material["nodal_location"]["onedrive_live"],
                "sync_score":    round(material["nodal_location"]["sync_score"], 4),
                "anchor_weight": round(material["nodal_location"]["anchor_weight"], 4),
                "computed_at":   material["nodal_location"]["computed_at"],
            },
        },
        "pim_planning": {
            "has_data":           pim["has_data"],
            "ss_coverage_rate":   pim["ss_coverage_rate"],
            "ss_breach_rate":     pim["ss_breach_rate"],
            "ss_signal":          pim["ss_signal"],
            "minmax_coverage_rate":  pim["minmax_coverage_rate"],
            "minmax_violation_rate": pim["minmax_violation_rate"],
            "mm_signal":          pim["mm_signal"],
            "lt_agility_score":   pim["lt_agility_score"],
            "lt_breach_rate":     pim["lt_breach_rate"],
            "lt_signal":          pim["lt_signal"],
            "item_count":         pim["item_count"],
            "org_count":          pim["org_count"],
            "updated_at":         pim["updated_at"],
            "axis_injection": {
                sense: round(pim_injection[sense], 4) for sense in senses
            },
        },
        "repository_catalog": {
            "has_data": catalog["has_data"],
            "total_files": catalog["total_files"],
            "capabilities_discovered": catalog["capabilities_discovered"],
            "source_count": catalog["source_count"],
            "file_coverage": catalog["file_coverage"],
            "capability_density": catalog["capability_density"],
            "source_breadth": catalog["source_breadth"],
            "freshness": catalog["freshness"],
            "avg_relevance_score": catalog["avg_relevance_score"],
            "avg_pim_relevance": catalog["avg_pim_relevance"],
            "avg_supply_chain_relevance": catalog["avg_supply_chain_relevance"],
            "catalog_signal": catalog["catalog_signal"],
            "playwright_perception_signal": catalog["playwright_perception_signal"],
            "media_perception_signal": catalog["media_perception_signal"],
            "playwright_artifacts": catalog["playwright_artifacts"],
            "media_perception": catalog["media_perception"],
            "updated_at": catalog["updated_at"],
            "top_terms": catalog["top_terms"],
            "axis_injection": {
                sense: round(catalog_injection[sense], 4) for sense in senses
            },
        },
        "ueqgm_runtime": {
            "active": bool(ueqgm["active"]),
            "certainty": round(_clip01(ueqgm.get("certainty", 0.0)), 4),
            "corpus_signal": round(_clip01(ueqgm.get("corpus_signal", 0.0)), 4),
            "learning_signal": round(_clip01(ueqgm.get("learning_signal", 0.0)), 4),
            "newness_signal": round(_clip01(ueqgm.get("newness_signal", 0.0)), 4),
            "relational_depth": round(_clip01(ueqgm.get("relational_depth", 0.0)), 4),
            "symbiotic_gain": round(_clip01(ueqgm.get("symbiotic_gain", 0.0)), 4),
            "expansion_pressure": round(_clip01(ueqgm.get("expansion_pressure", 0.0)), 4),
            "phase_weight": round(float(ueqgm.get("phase_weight", 1.0) or 1.0), 6),
            "coherence_depth": max(0, int(ueqgm.get("coherence_depth") or 0)),
            "mesh_alignment": round(_clip01(ueqgm.get("mesh_alignment", 0.0)), 4),
            "observer_alignment": round(_clip01(ueqgm.get("observer_alignment", 0.0)), 4),
            "runtime_keywords": list(ueqgm.get("runtime_keywords") or []),
            "applied_parameters": list(ueqgm.get("applied_parameters") or []),
            "retained_parameters": list(ueqgm.get("retained_parameters") or []),
            "axis_drive": {
                sense: round(_clip01(ueqgm["axis_drive"][sense]), 4) for sense in senses
            },
            "axis_injection": {
                sense: round(ueqgm_injection[sense], 4) for sense in senses
            },
            "updated_at": ueqgm.get("updated_at"),
        },
        "transaction": {
            "drive": round(transaction_drive, 4),
            "symbiotic_drive": round(symbiotic_drive, 4),
            "eligible": material["eligible"],
            "bit_flip": transaction_kind,
            "ueqgm_overlay": "adaptive" if ueqgm["active"] else "inactive",
            "topology": material["mesh"]["topology"] if material["eligible"] else "observer",
        },
    }


# ---------------------------------------------------------------------------
# Bit Flip — parity inversion on the SiCi axial channel
# ---------------------------------------------------------------------------

def bit_flip_parity(observer: float, t: float | None = None) -> int:
    """Instantaneous parity bit ``±1`` for the oscillating expansion.

    Uses the UEQGM Floquet modulation factor ``cos(ω·t)`` biased by the
    observer.  When ``observer`` is strong the cosine is shifted toward
    positive parity but the period stays well-defined, so the bit
    alternates faithfully.

    Parameters
    ----------
    observer : float
        7th-axis observer tangent magnitude in [0, 1].
    t : float | None
        Time coordinate (seconds).  Defaults to ``time.time()``.

    Returns
    -------
    +1  → broaden phase  (positive φ, expand the lattice outward)
    −1  → deepen phase   (negative φ, deepen ingestion of current frontier)
    """
    if t is None:
        t = time.time()
    omega = _OMEGA_BASE * (1.0 + _OMEGA_GAIN * max(0.0, min(1.0, observer)))
    bias = (2.0 * max(0.0, min(1.0, observer)) - 1.0) * 0.25
    val = math.cos(omega * t) + bias
    return 1 if val >= 0.0 else -1


# ---------------------------------------------------------------------------
# Driver — rate-limited; persists to brain_kv
# ---------------------------------------------------------------------------

def oscillating_expansion_step(*, force: bool = False) -> dict:
    """Run one orchestration step of the 7th-D observer tangent.

    Algorithm
    ---------
    1. Read the current sense signals via ``temporal_spatiality``.
    2. Compute the observer tangent and the instantaneous parity bit.
    3. If the bit changed since last persistence, increment the flip
       counter and rotate the expansion phase label.
    4. Persist to ``brain_kv`` under the ``entirety:*`` keys and return
       a summary dict.

    Returns
    -------
    dict with keys: ``observer``, ``bit_state``, ``expansion_phase``,
    ``flipped`` (bool), ``flip_count``, ``axes``, ``magnitude``.
    Returns ``{"skipped": True, "reason": ...}`` if rate-limited.
    """
    global _LAST_TS
    with _LOCK:
        if not force and time.monotonic() - _LAST_TS < _MIN_SECONDS:
            return {"skipped": True, "reason": "rate-limited"}

        state = system_entirety_state()
        observer = float(state["observer"])
        transaction = state.get("transaction") or {}
        transaction_drive = _clip01(transaction.get("drive", observer))
        bit = bit_flip_parity(transaction_drive)
        phase = "broaden" if bit > 0 else "deepen"

        for attempt in range(1, _LOCK_RETRIES + 1):
            try:
                with _conn() as cn:
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
                    persisted_state = {
                        **state,
                        "bit_state":       bit,
                        "expansion_phase": phase,
                        "flip_count":      new_count,
                        "flipped":         flipped,
                        "ran_at":          ran_at,
                    }

                    _kv_write(cn, _KV_BIT, bit)
                    _kv_write(cn, _KV_COUNT, new_count)
                    _kv_write(cn, _KV_PHASE, phase)
                    _kv_write(cn, _KV_STATE, persisted_state)
                    _share_learning_into_mesh(cn, persisted_state)
                    _log_flip_snapshot(
                        cn,
                        state=state,
                        bit_state=bit,
                        expansion_phase=phase,
                        flipped=flipped,
                        flip_count=new_count,
                    )
                break
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc).lower() or attempt >= _LOCK_RETRIES:
                    raise
                time.sleep(_LOCK_RETRY_BASE * (2 ** (attempt - 1)))

        _LAST_TS = time.monotonic()

        # Fire daily dispatch in a background thread so SMTP never blocks the
        # main expansion step.  Only attempted on Deepen-phase steps so the
        # report is sent after each inward consolidation cycle.
        if phase == "deepen":
            def _dispatch_bg():
                try:
                    from .daily_dispatch import send_daily_dispatch
                    result = send_daily_dispatch() or {}
                    sent = bool(result.get("sent"))
                    reason = str(result.get("reason") or "unknown")
                    message = f"[daily-dispatch] sent={sent} reason={reason} source=torus"
                    error = result.get("error")
                    if error:
                        message += f" error={error}"
                    if sent:
                        _LOG.info(message)
                    else:
                        _LOG.warning(message)
                except Exception:
                    _LOG.exception("[daily-dispatch] torus trigger failure")

            threading.Thread(target=_dispatch_bg, name="daily-dispatch", daemon=True).start()

        return {
            "observer":        observer,
            "transaction":     transaction,
            "bit_state":       bit,
            "expansion_phase": phase,
            "flipped":         flipped,
            "flip_count":      new_count,
            "axes":            state["axes"],
            "magnitude":       state["magnitude"],
            "ran_at":          ran_at,
        }


# ---------------------------------------------------------------------------
# Public read accessors
# ---------------------------------------------------------------------------

def get_entirety_state() -> dict:
    """Read the persisted 7-D state dict (``{}`` if never run)."""
    with _conn() as cn:
        return _kv_read(cn, _KV_STATE, default={}) or {}


def get_bit_state(default: int = 1) -> int:
    """Read the current parity bit (``+1`` broaden / ``−1`` deepen)."""
    with _conn() as cn:
        v = _kv_read(cn, _KV_BIT, default=default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def get_expansion_phase(default: str = "broaden") -> str:
    """Read the current expansion phase label."""
    with _conn() as cn:
        v = _kv_read(cn, _KV_PHASE, default=default)
    return str(v) if v in ("broaden", "deepen") else default


def get_flip_history(limit: int = 50) -> list[dict]:
    """Return the most recent 7-D snapshots from ``entirety_flip_log``."""
    with _conn() as cn:
        _ensure_flip_log_table(cn)
        rows = cn.execute(
            f"SELECT id, ran_at, observer, bit_state, expansion_phase, "
            f"flipped, flip_count, magnitude, axes_json, source "
            f"FROM {_FLIP_LOG_TABLE} ORDER BY id DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    out: list[dict] = []
    for row in rows:
        axes = {}
        try:
            axes = json.loads(row[8]) if row[8] else {}
        except json.JSONDecodeError:
            axes = {}
        out.append(
            {
                "id": row[0],
                "ran_at": row[1],
                "observer": float(row[2]),
                "bit_state": int(row[3]),
                "expansion_phase": row[4],
                "flipped": bool(row[5]),
                "flip_count": int(row[6]),
                "magnitude": float(row[7]),
                "axes": axes,
                "source": row[9],
            }
        )
    return out


__all__ = [
    "observer_tangent",
    "system_entirety_state",
    "bit_flip_parity",
    "oscillating_expansion_step",
    "get_entirety_state",
    "get_bit_state",
    "get_expansion_phase",
    "get_flip_history",
    "session_entirety_evaluation",
    "update_pim_state",
    "send_daily_dispatch",
]


def send_daily_dispatch(*, force: bool = False) -> dict:
    """Public shim — delegates to :mod:`.daily_dispatch`.

    Exposed on the module so callers can import from ``system_entirety``
    directly without knowing the sub-module layout.

    Parameters
    ----------
    force : bool
        Bypass the 24-hour rate limit (useful for testing).

    Returns
    -------
    dict with keys ``sent``, ``reason``, and optionally ``error``.
    """
    from .daily_dispatch import send_daily_dispatch as _send
    return _send(force=force)


# ---------------------------------------------------------------------------
# Session-scoped Entirety evaluation (non-persistent)
# ---------------------------------------------------------------------------

def session_entirety_evaluation(session_signals: dict) -> dict:
    """Compute the 7-D observer tangent for a GeoGraph session context.

    Unlike :func:`oscillating_expansion_step` this function **writes nothing**
    to ``brain_kv`` or the flip log.  It is called by the GeoGraph session
    broker to incorporate the session's OCR signals into the System Entirety
    for output enrichment without persisting any raw inventory data.

    Parameters
    ----------
    session_signals : dict
        Sense-axis overrides blended with the current live signals.
        Keys are a subset of ``(vision, touch, smell, body, brain, perception)``.
        The broker typically passes:
            ``vision``      — mean OCR confidence across scanned parts
            ``perception``  — normalised scan count (0-1, saturates at 20+)

    Returns
    -------
    dict with keys:
        ``axes``        — 6-D blended sense vector
        ``observer``    — 7th-axis observer tangent in [0, 1]
        ``magnitude``   — 7-D Euclidean norm
        ``bit_state``   — instantaneous parity ±1 (not persisted)
        ``expansion_phase`` — "broaden" | "deepen" (not persisted)
        ``session``     — True (marks result as session-scoped)
    """
    # Start from the live persistent signals so the session boost is additive
    try:
        from .temporal_spatiality import _sense_signals
        live = _sense_signals()
    except Exception:
        live = {}

    senses = ("vision", "touch", "smell", "body", "brain", "perception")
    blended: dict[str, float] = {}
    for s in senses:
        live_val = float(live.get(s, 0.0))
        boost    = float(session_signals.get(s, 0.0))
        # Blend: live contributes 0.6, session boost 0.4; clamp to [0, 1]
        blended[s] = max(0.0, min(1.0, live_val * 0.6 + boost * 0.4))

    state    = system_entirety_state(blended)
    observer = float(state["observer"])
    transaction = state.get("transaction") or {}
    bit      = bit_flip_parity(transaction.get("drive", observer))
    phase    = "broaden" if bit > 0 else "deepen"

    return {
        **state,
        "bit_state":       bit,
        "expansion_phase": phase,
        "session":         True,
    }

