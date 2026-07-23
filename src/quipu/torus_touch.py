"""Torus Touch — continuous active-substrate pressure on the 7-D torus manifold.

Every tick, active corpus entities (Endpoint, SpatialMaterialProcessor,
AssetResource) are pushed along the categorical-gap gradient of the toroidal
manifold using a sinusoidal envelope.  The result is that physical geometry
(bearing, range), material certainty (precision, prob_cert), and hardware
capacity (cores, RAM, VRAM, storage) all flow through one unified gradient
field — tunnel weights naturally track manifold geometry.

Key KV keys
-----------
``torus_amplify:<entity_id>``
    Read: per-entity step multiplier written by grounded_tunneling when an
    active resistance path is found.
``torus_vel:<entity_id>``
    Written by ``_write_velocity()`` on every tick; read on the next tick
    for EMA momentum carryover.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any

TORUS_DIMS: int = 7

# Entity types that participate in torus pressure
_ACTIVE_TYPES: tuple[str, ...] = ("Endpoint", "SpatialMaterialProcessor", "AssetResource")

_BASE_STEP: float = 0.10       # radians per tick (scaled by step arg and amplify)
_MOMENTUM: float = 0.85        # EMA momentum for velocity carryover
_TWO_PI: float = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Categorical-gap gradient (CatGapField)
# ---------------------------------------------------------------------------

def _cat_gap_gradient(angles: list[float], dim: int) -> float:
    """Sinusoidal CatGapField gradient on one torus dimension.

    Gradient of the periodic gap potential − cos(2θ)/2 is sin(2θ).
    We negate it so the force *pushes toward* the gap at 0 / 2π.
    """
    theta = angles[dim] if dim < len(angles) else 0.0
    return -math.sin(2.0 * theta) * 0.5


# ---------------------------------------------------------------------------
# KV helpers — velocity and amplify
# ---------------------------------------------------------------------------

def _read_velocity(cn: sqlite3.Connection, entity_id: str) -> list[float]:
    try:
        row = cn.execute(
            "SELECT value FROM kv_store WHERE key=?",
            (f"torus_vel:{entity_id}",),
        ).fetchone()
        if row and row[0]:
            val = json.loads(row[0])
            if isinstance(val, list) and len(val) == TORUS_DIMS:
                return [float(v) for v in val]
    except Exception:
        pass
    return [0.0] * TORUS_DIMS


def _write_velocity(cn: sqlite3.Connection, entity_id: str, velocity: list[float]) -> None:
    cn.execute(
        "INSERT INTO kv_store(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (f"torus_vel:{entity_id}", json.dumps([round(v, 6) for v in velocity])),
    )


def _read_amplify(cn: sqlite3.Connection, entity_id: str, entity_type: str) -> float:
    """Return the per-entity step multiplier (≥ 1.0)."""
    for key in (
        f"torus_amplify:{entity_type}:{entity_id}",
        f"torus_amplify:{entity_id}",
    ):
        try:
            row = cn.execute("SELECT value FROM kv_store WHERE key=?", (key,)).fetchone()
            if row and row[0] not in (None, ""):
                return max(1.0, float(row[0]))
        except Exception:
            pass
    return 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tick_torus_pressure(
    cn: sqlite3.Connection,
    *,
    step: float = 0.1,
    entity_types: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Apply one torus-pressure tick across all active corpus entities.

    For each entity the 7-D gradient is computed, EMA velocity is applied,
    and ``torus_angles`` + ``torus_tick`` are written back to ``corpus_entity``.

    Returns::

        {
            "material_processors": int,  # all entities in the active substrate
            "active_entities":     int,  # same count (alias for clarity)
            "moved":               int,  # entities whose angles changed
            "processor_nodes":     int,  # SpatialMaterialProcessor count only
        }
    """
    if entity_types is None:
        entity_types = _ACTIVE_TYPES

    now = datetime.now(timezone.utc).isoformat()

    placeholders = ",".join("?" * len(entity_types))
    try:
        rows = cn.execute(
            f"SELECT entity_id, entity_type, props_json FROM corpus_entity "
            f"WHERE entity_type IN ({placeholders})",
            entity_types,
        ).fetchall()
    except sqlite3.Error:
        return {"material_processors": 0, "active_entities": 0, "moved": 0, "processor_nodes": 0}

    processor_nodes = 0
    active_count = 0
    moved_count = 0

    for entity_id, entity_type, props_json in rows:
        try:
            props: dict[str, Any] = json.loads(props_json) if props_json else {}
        except Exception:
            props = {}

        # Ensure 7D torus angles exist
        raw_angles = props.get("torus_angles")
        if raw_angles and isinstance(raw_angles, list):
            angles = [float(a) for a in raw_angles[:TORUS_DIMS]]
        else:
            angles = [0.0] * TORUS_DIMS
        while len(angles) < TORUS_DIMS:
            angles.append(0.0)

        amplify = _read_amplify(cn, entity_id, entity_type)
        effective_step = step * _BASE_STEP * amplify

        velocity = _read_velocity(cn, entity_id)
        new_angles: list[float] = []
        new_velocity: list[float] = []
        any_moved = False

        for dim in range(TORUS_DIMS):
            grad = _cat_gap_gradient(angles, dim)
            delta = grad * effective_step
            vel = _MOMENTUM * velocity[dim] + delta
            new_angle = (angles[dim] + vel) % _TWO_PI
            if abs(new_angle - angles[dim]) > 1e-9:
                any_moved = True
            new_angles.append(round(new_angle, 6))
            new_velocity.append(round(vel, 6))

        _write_velocity(cn, entity_id, new_velocity)

        props["torus_angles"] = new_angles
        props["torus_tick"] = {
            "step": round(effective_step, 6),
            "amplify": round(amplify, 6),
            "ticked_at": now,
        }
        props["torus_ticked_at"] = now

        cn.execute(
            "UPDATE corpus_entity SET props_json=?, last_seen=? "
            "WHERE entity_id=? AND entity_type=?",
            (json.dumps(props, default=str), now, entity_id, entity_type),
        )

        if entity_type == "SpatialMaterialProcessor":
            processor_nodes += 1
        active_count += 1
        if any_moved:
            moved_count += 1

    return {
        "material_processors": active_count,
        "active_entities": active_count,
        "moved": moved_count,
        "processor_nodes": processor_nodes,
    }


__all__ = [
    "TORUS_DIMS",
    "tick_torus_pressure",
]
