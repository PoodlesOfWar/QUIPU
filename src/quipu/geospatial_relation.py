"""Geospatial Relation — grounds GeoGraph OCR detections in physical space.

Materializes world-pose anchors as SpatialMaterialProcessor corpus nodes
with actualized_space payloads in the ENU_METER_WORLD_V1 reference frame.

Two independent derivations are fused via inverse-variance weighting:

  Path A — un-projection:  pixel → ENU via pinhole sensor model
  Path B — haversine:      sensor geo-origin → bearing/range raycast

  σ_fused⁻² = σ_A⁻² + σ_B⁻²

Anchor certainty drives the ``material_anchor`` flag (precision ≥ 1 m⁻²)
and writes bearing → TORUS_SPATIAL_DIM (0) and log-range → TORUS_RANGE_DIM (1)
so the existing torus-pressure gradient converts physical geometry directly
into Touch pressure on the next tick.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

ACTUALIZED_SPACE_FRAME: str = "ENU_METER_WORLD_V1"
SPATIAL_MATERIAL_PROCESSOR: str = "SpatialMaterialProcessor"

TORUS_SPATIAL_DIM: int = 0   # bearing → torus dimension 0
TORUS_RANGE_DIM: int = 1     # log-range → torus dimension 1
TORUS_DIMS: int = 7

PEER_MESH_RADIUS_M: float = 50.0

# Relationship labels
ACTUALIZES_SPACE: str = "ACTUALIZES_SPACE"
MATERIAL_ANCHOR: str = "MATERIAL_ANCHOR"

# Precision threshold (m⁻²) for material_anchor flag
_MATERIAL_ANCHOR_MIN_PRECISION: float = 1.0

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Camera / sensor geometry data classes
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    fx: float           # focal length x (pixels)
    fy: float           # focal length y (pixels)
    cx: float           # principal point x (pixels)
    cy: float           # principal point y (pixels)
    width: int = 640
    height: int = 480


@dataclass
class SensorPose:
    """Position and orientation of the imaging sensor."""
    lat: float = 0.0            # WGS-84 latitude (degrees)
    lon: float = 0.0            # WGS-84 longitude (degrees)
    alt: float = 0.0            # altitude above ground (metres)
    bearing_deg: float = 0.0   # azimuth clockwise from north (degrees)
    pitch_deg: float = 0.0     # elevation angle, + = up (degrees)
    roll_deg: float = 0.0      # roll (degrees)
    xyz_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_distance_m(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2.0 * R * math.asin(min(1.0, math.sqrt(a)))


def haversine_destination(
    lat: float,
    lon: float,
    bearing_deg: float,
    distance_m: float,
) -> tuple[float, float]:
    """Return (lat, lon) of the point ``distance_m`` along ``bearing_deg`` from (lat, lon)."""
    R = 6_371_000.0
    d = distance_m / R
    bearing_rad = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(d)
        + math.cos(phi1) * math.sin(d) * math.cos(bearing_rad)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(bearing_rad) * math.sin(d) * math.cos(phi1),
        math.cos(d) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), math.degrees(lambda2)


def unproject_pixel(
    px: float,
    py: float,
    intrinsics: CameraIntrinsics,
    pose: SensorPose,
    *,
    range_m: float | None = None,
    ground_plane_z: float = 0.0,
) -> tuple[list[float], float]:
    """Pinhole un-project: pixel → ENU world-frame point.

    Returns ``(xyz_m, sigma_m)``.  If ``range_m`` is given the ray is scaled to
    that depth; otherwise it is intersected with the ground plane at
    ``ground_plane_z``.
    """
    # Normalised sensor-frame direction
    rx = (px - intrinsics.cx) / max(1e-6, intrinsics.fx)
    ry = (py - intrinsics.cy) / max(1e-6, intrinsics.fy)
    rz = 1.0

    # Rotate ray into ENU using bearing + pitch (roll ignored for ground case)
    bearing_rad = math.radians(pose.bearing_deg)
    pitch_rad = math.radians(pose.pitch_deg)
    cos_b, sin_b = math.cos(bearing_rad), math.sin(bearing_rad)
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)

    wx = rx * cos_b + rz * sin_b * cos_p
    wy = -rx * sin_b + rz * cos_b * cos_p
    wz = -ry + rz * (-sin_p)

    sensor_z = (pose.xyz_m[2] if pose.xyz_m and len(pose.xyz_m) > 2 else pose.alt) or 0.0

    if range_m is not None:
        length = math.sqrt(wx * wx + wy * wy + wz * wz) or 1.0
        scale = range_m / length
    else:
        if abs(wz) < 1e-6:
            scale = 100.0
        else:
            scale = max(0.0, (ground_plane_z - sensor_z) / wz)

    origin = list(pose.xyz_m) if pose.xyz_m and len(pose.xyz_m) >= 3 else [0.0, 0.0, sensor_z]
    xyz = [origin[0] + wx * scale, origin[1] + wy * scale, origin[2] + wz * scale]
    sigma_m = max(0.3, 0.05 * max(1.0, scale))
    return xyz, sigma_m


# ---------------------------------------------------------------------------
# Sensor-space relation (A + B → fused)
# ---------------------------------------------------------------------------

def relate_detection(
    detection: dict[str, Any],
    pose: SensorPose,
    *,
    intrinsics: CameraIntrinsics | None = None,
) -> dict[str, Any]:
    """Ground one detection in physical space via two independent paths and fuse.

    Returns a world-pose dict with fused ``xyz_m``, ``sigma_m``, ``precision``,
    ``stat_certainty`` (residual-z), and ``prob_cert`` (σ⁻² saturation).
    """
    px = float(detection.get("px") or detection.get("x") or 0.0)
    py = float(detection.get("py") or detection.get("y") or 0.0)
    bearing_deg = float(detection.get("bearing_deg") or pose.bearing_deg)
    range_m = detection.get("range_m")
    if range_m is not None:
        range_m = float(range_m)

    # Path A — unproject
    if intrinsics is not None:
        xyz_a, sigma_a = unproject_pixel(px, py, intrinsics, pose, range_m=range_m)
    else:
        sigma_a = 5.0
        xyz_a = list(pose.xyz_m) if pose.xyz_m else [0.0, 0.0, 0.0]

    # Path B — haversine raycast
    effective_range = range_m if range_m is not None else max(
        1.0, math.sqrt(xyz_a[0] ** 2 + xyz_a[1] ** 2)
    )
    dest_lat, dest_lon = haversine_destination(pose.lat, pose.lon, bearing_deg, effective_range)
    dx_m = (haversine_distance_m(pose.lat, pose.lon, pose.lat, dest_lon)
            * math.copysign(1.0, dest_lon - pose.lon))
    dy_m = (haversine_distance_m(pose.lat, pose.lon, dest_lat, pose.lon)
            * math.copysign(1.0, dest_lat - pose.lat))
    xyz_b = [dx_m, dy_m, 0.0]
    sigma_b = max(0.5, 0.08 * effective_range)

    # Inverse-variance fusion: σ_fused⁻² = σ_A⁻² + σ_B⁻²
    w_a = 1.0 / max(1e-9, sigma_a * sigma_a)
    w_b = 1.0 / max(1e-9, sigma_b * sigma_b)
    w_total = w_a + w_b
    xyz_fused = [(xyz_a[i] * w_a + xyz_b[i] * w_b) / w_total for i in range(3)]
    sigma_fused = 1.0 / math.sqrt(max(1e-9, w_total))
    precision = 1.0 / max(1e-9, sigma_fused * sigma_fused)

    # Statistical certainty — residual z-score between A and B
    residual = math.sqrt(sum((xyz_a[i] - xyz_b[i]) ** 2 for i in range(3)))
    combined_sigma = math.sqrt(sigma_a * sigma_a + sigma_b * sigma_b)
    z_score = residual / max(1e-6, combined_sigma)
    stat_certainty = max(0.0, min(1.0, 1.0 - z_score / 10.0))

    # Probabilistic certainty — σ⁻² saturation curve
    prob_cert = min(1.0, w_total / (w_total + 1.0))

    return {
        "xyz_m": [round(v, 4) for v in xyz_fused],
        "sigma_m": round(sigma_fused, 4),
        "precision": round(precision, 4),
        "stat_certainty": round(stat_certainty, 4),
        "prob_cert": round(prob_cert, 4),
        "bearing_deg": bearing_deg,
        "range_m": effective_range,
        "frame_id": ACTUALIZED_SPACE_FRAME,
        "path_a": {"xyz_m": [round(v, 4) for v in xyz_a], "sigma_m": round(sigma_a, 4)},
        "path_b": {"xyz_m": [round(v, 4) for v in xyz_b], "sigma_m": round(sigma_b, 4)},
    }


# ---------------------------------------------------------------------------
# Corpus anchoring
# ---------------------------------------------------------------------------

def anchor_entity(
    cn: sqlite3.Connection,
    entity_id: str,
    entity_type: str,
    world_pose: dict[str, Any],
    *,
    now: str | None = None,
    material_class: str = "spatial inventory material",
    material_structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a Bayesian-blended world-pose anchor as a SpatialMaterialProcessor node.

    Sets ``material_anchor = True`` when ``precision >= 1 m⁻²``.
    Returns the written props dict.
    """
    if now is None:
        now = datetime.now(timezone.utc).isoformat()

    precision = float(world_pose.get("precision", 0.0))
    prob_cert = float(world_pose.get("prob_cert", 0.0))
    sigma_m = float(world_pose.get("sigma_m", 1.0))
    xyz_m = list(world_pose.get("xyz_m", [0.0, 0.0, 0.0]))
    bearing_deg = float(world_pose.get("bearing_deg", 0.0))
    range_m = float(world_pose.get("range_m") or max(1.0, sigma_m * 10.0))

    if material_structure is None:
        material_structure = {
            "processor": "physical_material_science",
            "lattice_strength": round(min(1.0, precision / (precision + 10.0)), 4) if precision > 0 else 0.0,
        }

    material_anchor = precision >= _MATERIAL_ANCHOR_MIN_PRECISION

    # Read existing SpatialMaterialProcessor node for Bayesian blending
    row = cn.execute(
        "SELECT props_json FROM corpus_entity WHERE entity_id=? AND entity_type=?",
        (entity_id, SPATIAL_MATERIAL_PROCESSOR),
    ).fetchone()
    existing: dict[str, Any] = {}
    if row and row[0]:
        try:
            existing = json.loads(row[0])
        except Exception:
            existing = {}

    old_pose = existing.get("world_pose") or {}
    old_precision = float(old_pose.get("precision") or 0.0)

    if old_precision > 0.0 and precision > 0.0:
        # Bayesian posterior: combine information sums
        blended_precision = old_precision + precision
        blended_sigma = 1.0 / math.sqrt(max(1e-9, blended_precision))
        blend = old_precision / blended_precision
        blended_prob_cert = min(1.0, prob_cert + (1.0 - prob_cert) * blend * 0.5)
        old_xyz = old_pose.get("xyz_m") or [0.0, 0.0, 0.0]
        blended_xyz = [
            blend * float(old_xyz[i] if i < len(old_xyz) else 0.0)
            + (1.0 - blend) * float(xyz_m[i] if i < len(xyz_m) else 0.0)
            for i in range(3)
        ]
    else:
        blended_precision = precision
        blended_sigma = sigma_m
        blended_prob_cert = prob_cert
        blended_xyz = xyz_m

    updated_pose: dict[str, Any] = {
        "xyz_m": [round(v, 4) for v in blended_xyz],
        "precision": round(blended_precision, 4),
        "prob_cert": round(blended_prob_cert, 4),
        "sigma_m": round(blended_sigma, 4),
        "bearing_deg": bearing_deg,
        "range_m": range_m,
        "frame_id": ACTUALIZED_SPACE_FRAME,
    }

    actualized_space: dict[str, Any] = {
        "frame_id": ACTUALIZED_SPACE_FRAME,
        "precision": round(blended_precision, 4),
        "prob_cert": round(blended_prob_cert, 4),
        "material_class": material_class,
        "material_structure": material_structure,
        "covariance_diag": [round(blended_sigma ** 2, 6)] * 3,
        "sigma_m": round(blended_sigma, 4),
    }

    torus_angles = _world_pose_to_torus_angles(
        bearing_deg, range_m, blended_precision, blended_prob_cert, blended_xyz,
        existing=list(existing.get("torus_angles") or []),
    )

    props: dict[str, Any] = {
        **existing,
        "entity_id": entity_id,
        "entity_type": SPATIAL_MATERIAL_PROCESSOR,
        "world_pose": updated_pose,
        "actualized_space": actualized_space,
        "material_anchor": material_anchor,
        "torus_angles": torus_angles,
        "frame_id": ACTUALIZED_SPACE_FRAME,
        "updated_at": now,
    }

    _upsert_entity(cn, entity_id, SPATIAL_MATERIAL_PROCESSOR,
                   f"SpatialMaterialProcessor {entity_id}", props, now)

    # ACTUALIZES_SPACE edge from source entity to the processor node
    _upsert_edge(cn, entity_id, entity_type,
                 entity_id, SPATIAL_MATERIAL_PROCESSOR,
                 ACTUALIZES_SPACE, max(0.1, min(1.0, blended_prob_cert)), now)

    return props


def couple_anchors_to_torus(
    cn: sqlite3.Connection,
    *,
    now: str | None = None,
) -> dict[str, int]:
    """Write bearing → TORUS_SPATIAL_DIM and log-range → TORUS_RANGE_DIM for all anchors.

    Returns ``{updated: int}``.
    """
    if now is None:
        now = datetime.now(timezone.utc).isoformat()
    rows = cn.execute(
        "SELECT entity_id, props_json FROM corpus_entity WHERE entity_type=?",
        (SPATIAL_MATERIAL_PROCESSOR,),
    ).fetchall()
    updated = 0
    for entity_id, props_json in rows:
        try:
            props = json.loads(props_json) if props_json else {}
        except Exception:
            props = {}
        world_pose = props.get("world_pose") or {}
        bearing_deg = float(world_pose.get("bearing_deg") or 0.0)
        range_m = float(world_pose.get("range_m") or 1.0)
        precision = float(world_pose.get("precision") or 0.0)
        prob_cert = float(world_pose.get("prob_cert") or 0.0)
        xyz_m = world_pose.get("xyz_m") or [0.0, 0.0, 0.0]
        existing = list(props.get("torus_angles") or [])
        angles = _world_pose_to_torus_angles(
            bearing_deg, range_m, precision, prob_cert, xyz_m, existing=existing
        )
        props["torus_angles"] = angles
        props["updated_at"] = now
        cn.execute(
            "UPDATE corpus_entity SET props_json=?, last_seen=? "
            "WHERE entity_id=? AND entity_type=?",
            (json.dumps(props, default=str), now, entity_id, SPATIAL_MATERIAL_PROCESSOR),
        )
        updated += 1
    return {"updated": updated}


def strengthen_peer_mesh(
    cn: sqlite3.Connection,
    *,
    now: str | None = None,
) -> dict[str, int]:
    """Mint / EMA-reinforce MATERIAL_ANCHOR edges from nearby Endpoint entities.

    Endpoints within ``PEER_MESH_RADIUS_M`` of a material anchor get a
    ``MATERIAL_ANCHOR`` edge weighted by proximity.
    Returns ``{edges_created: int, endpoints_checked: int}``.
    """
    if now is None:
        now = datetime.now(timezone.utc).isoformat()

    processors = cn.execute(
        "SELECT entity_id, props_json FROM corpus_entity WHERE entity_type=?",
        (SPATIAL_MATERIAL_PROCESSOR,),
    ).fetchall()
    try:
        endpoints = cn.execute(
            "SELECT entity_id, props_json FROM corpus_entity WHERE entity_type='Endpoint'",
        ).fetchall()
    except sqlite3.Error:
        endpoints = []

    edges_created = 0
    for proc_id, proc_json in processors:
        try:
            proc_props = json.loads(proc_json) if proc_json else {}
        except Exception:
            proc_props = {}
        if not proc_props.get("material_anchor"):
            continue
        proc_xyz = (proc_props.get("world_pose") or {}).get("xyz_m") or [0.0, 0.0, 0.0]
        proc_prec = float((proc_props.get("world_pose") or {}).get("precision") or 0.0)

        for ep_id, ep_json in endpoints:
            try:
                ep_props = json.loads(ep_json) if ep_json else {}
            except Exception:
                ep_props = {}
            ep_xyz = (ep_props.get("world_pose") or {}).get("xyz_m") or [0.0, 0.0, 0.0]
            dist_m = math.sqrt(sum(
                (float(proc_xyz[i] if i < len(proc_xyz) else 0.0)
                 - float(ep_xyz[i] if i < len(ep_xyz) else 0.0)) ** 2
                for i in range(3)
            ))
            if dist_m > PEER_MESH_RADIUS_M:
                continue
            weight = max(0.05, min(1.0, proc_prec / (proc_prec + 1.0) * (1.0 - dist_m / PEER_MESH_RADIUS_M)))
            _upsert_edge(cn, ep_id, "Endpoint",
                         proc_id, SPATIAL_MATERIAL_PROCESSOR,
                         MATERIAL_ANCHOR, weight, now)
            edges_created += 1

    return {"edges_created": edges_created, "endpoints_checked": len(endpoints)}


def tick_geospatial_relation(
    cn: sqlite3.Connection,
    detections: list[dict[str, Any]],
    pose: SensorPose,
    *,
    intrinsics: CameraIntrinsics | None = None,
    material_class: str = "spatial inventory material",
    source_entity_type: str = "Endpoint",
) -> dict[str, Any]:
    """Top-level orchestrator: relate → anchor → couple → strengthen."""
    now = datetime.now(timezone.utc).isoformat()
    anchored = 0

    for det in detections:
        entity_id = str(det.get("entity_id") or det.get("id") or "")
        if not entity_id:
            continue
        world_pose = relate_detection(det, pose, intrinsics=intrinsics)
        anchor_entity(
            cn, entity_id, source_entity_type, world_pose,
            now=now, material_class=material_class,
        )
        anchored += 1

    couple_stats = couple_anchors_to_torus(cn, now=now)
    mesh_stats = strengthen_peer_mesh(cn, now=now)

    return {
        "anchored": anchored,
        "torus_updates": couple_stats.get("updated", 0),
        "mesh_edges": mesh_stats.get("edges_created", 0),
        "now": now,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _world_pose_to_torus_angles(
    bearing_deg: float,
    range_m: float,
    precision: float,
    prob_cert: float,
    xyz_m: list[float],
    *,
    existing: list[float] | None = None,
) -> list[float]:
    """Map world-pose scalars onto TORUS_DIMS torus angles ∈ [0, 2π]."""
    angles: list[float] = list(existing) if existing else []
    while len(angles) < TORUS_DIMS:
        angles.append(0.0)
    angles = angles[:TORUS_DIMS]

    # Dim 0 (TORUS_SPATIAL_DIM): bearing → [0, 2π]
    angles[TORUS_SPATIAL_DIM] = math.radians(float(bearing_deg) % 360.0)

    # Dim 1 (TORUS_RANGE_DIM): log-range → [0, 2π]
    log_range = math.log1p(max(0.0, float(range_m))) / math.log1p(1000.0)
    angles[TORUS_RANGE_DIM] = _TWO_PI * min(1.0, log_range)

    # Dim 2: precision saturation
    angles[2] = _TWO_PI * (precision / (precision + 10.0) if precision > 0.0 else 0.0)

    # Dim 3: probabilistic certainty
    angles[3] = _TWO_PI * max(0.0, min(1.0, float(prob_cert)))

    # Dims 4–6: ENU components (log-scaled)
    xyz = list(xyz_m) if xyz_m else [0.0, 0.0, 0.0]
    for offset, coord in enumerate(xyz[:3]):
        angles[4 + offset] = _TWO_PI * min(1.0, math.log1p(max(0.0, abs(float(coord)))) / math.log1p(200.0))

    return [round(a % _TWO_PI, 6) for a in angles]


def _upsert_entity(
    cn: sqlite3.Connection,
    entity_id: str,
    entity_type: str,
    label: str,
    props: dict[str, Any],
    now: str,
) -> None:
    cn.execute(
        "INSERT INTO corpus_entity(entity_id, entity_type, label, props_json, "
        " first_seen, last_seen, samples) VALUES(?,?,?,?,?,?,1) "
        "ON CONFLICT(entity_id, entity_type) DO UPDATE SET "
        "  label=excluded.label, props_json=excluded.props_json, "
        "  last_seen=excluded.last_seen, samples=COALESCE(samples, 0)+1",
        (entity_id, entity_type, label, json.dumps(props, default=str), now, now),
    )


def _upsert_edge(
    cn: sqlite3.Connection,
    src_id: str,
    src_type: str,
    dst_id: str,
    dst_type: str,
    rel: str,
    weight: float,
    now: str,
) -> None:
    cn.execute(
        "INSERT INTO corpus_edge(src_id, src_type, dst_id, dst_type, rel, "
        "  weight, last_seen, samples) VALUES(?,?,?,?,?,?,?,1) "
        "ON CONFLICT(src_id, src_type, dst_id, dst_type, rel) DO UPDATE SET "
        "  weight=(COALESCE(corpus_edge.weight,0.0)*COALESCE(corpus_edge.samples,1)"
        "          +excluded.weight)/(COALESCE(corpus_edge.samples,1)+1), "
        "  last_seen=excluded.last_seen, samples=COALESCE(corpus_edge.samples,0)+1",
        (src_id, src_type, dst_id, dst_type, rel, float(weight), now),
    )


__all__ = [
    "ACTUALIZED_SPACE_FRAME",
    "SPATIAL_MATERIAL_PROCESSOR",
    "TORUS_SPATIAL_DIM",
    "TORUS_RANGE_DIM",
    "TORUS_DIMS",
    "PEER_MESH_RADIUS_M",
    "ACTUALIZES_SPACE",
    "MATERIAL_ANCHOR",
    "CameraIntrinsics",
    "SensorPose",
    "haversine_distance_m",
    "haversine_destination",
    "unproject_pixel",
    "relate_detection",
    "anchor_entity",
    "couple_anchors_to_torus",
    "strengthen_peer_mesh",
    "tick_geospatial_relation",
]
