"""Asset Resource Mesh - physical realization tunnels for System Entirety.

Spatial material processors standardize *where* physical space is actualized.
This module standardizes *with what* it becomes executable: CPU cores, RAM,
VRAM, and attached storage published by the compute grid.  Each asset becomes a corpus node with a
stable actualized-resource contract, and the assets are meshed through explicit
``ASSET_RESOURCE_TUNNEL`` edges.

The mesh is deliberately corpus-native:

* ``ComputePeer`` nodes represent hosts from ``compute_grid`` capacity JSON.
* ``AssetResource`` nodes represent ``cores``, ``ram``, ``vram``, and
    ``storage`` units.
* ``HAS_ASSET`` binds a peer to its resources.
* ``REALIZES_MATERIAL_PROCESSOR`` binds capacity to ``SpatialMaterialProcessor``
  anchors created by ``geospatial_relation``.
* ``ASSET_RESOURCE_TUNNEL`` connects resources intra-host and cross-host so the
  Brain can route physical realization through whichever substrate has the
    best available RAM / VRAM / core / attached-drive profile.
* ``SYSTEM_ENTIRETY_REALIZES`` summarizes the aggregate physical-realization
  score back onto an ``SystemEntirety`` corpus node and into ``brain_kv``.

The result is a meshed resource network where torus expansion and tunneling
can reason over material-space anchors and hardware capacity in one graph.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .geospatial_relation import (
        ACTUALIZED_SPACE_FRAME,
        SPATIAL_MATERIAL_PROCESSOR,
    )
except Exception:  # pragma: no cover - defensive import fallback
    ACTUALIZED_SPACE_FRAME = "ENU_METER_WORLD_V1"
    SPATIAL_MATERIAL_PROCESSOR = "SpatialMaterialProcessor"


ASSET_ENTITY_TYPE = "AssetResource"
COMPUTE_PEER_TYPE = "ComputePeer"
SYSTEM_ENTIRETY_TYPE = "SystemEntirety"
SYSTEM_ENTIRETY_ID = "system_entirety:physical_realization"

HAS_ASSET_REL = "HAS_ASSET"
REALIZES_PROCESSOR_REL = "REALIZES_MATERIAL_PROCESSOR"
ASSET_TUNNEL_REL = "ASSET_RESOURCE_TUNNEL"
SYSTEM_REALIZES_REL = "SYSTEM_ENTIRETY_REALIZES"

RESOURCE_KINDS = ("cores", "ram", "vram", "storage")
_PIPELINE_ROOT = Path(__file__).resolve().parents[2]
_OBSERVED_MESH_ASSETS_FILE = _PIPELINE_ROOT / "bridge_state" / "observed_mesh_assets.json"


def _resource_family(kind: str) -> str:
    if kind == "cores":
        return "core"
    if kind == "storage":
        return "storage"
    return "memory"


@dataclass(frozen=True)
class AssetVector:
    """Normalized view of one hardware asset on one compute peer."""
    entity_id: str
    host: str
    kind: str
    available: float
    total: float
    load: float
    score: float
    torus_angles: list[float]


def _ensure_tables(cn: sqlite3.Connection) -> None:
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
    cn.execute("CREATE TABLE IF NOT EXISTS kv_store(key TEXT PRIMARY KEY, value TEXT)")
    cn.execute(
        "CREATE TABLE IF NOT EXISTS brain_kv("
        "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
    )


def _safe_host(host: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(host or "unknown")).strip("-")
    return cleaned or "unknown"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _json_load(raw: Any, default: Any) -> Any:
    if raw in (None, ""):
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "active", "running"}
    return False


def _runtime_active(runtime: dict[str, Any]) -> bool:
    if not isinstance(runtime, dict):
        return False
    if "active" in runtime:
        return _truthy_flag(runtime.get("active"))
    agent_state = str(runtime.get("agent_state") or "").strip().lower()
    if agent_state in {"running", "active", "observed_active"}:
        return True
    return any(runtime.get(key) not in (None, "") for key in ("runtime_pid", "agent_pid", "watcher_pid", "publisher_pid"))


def _unit_hash(*parts: str) -> float:
    digest = hashlib.blake2b(":".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _as_peer_dict(peer: Any) -> dict[str, Any]:
    if isinstance(peer, dict):
        return dict(peer)
    if hasattr(peer, "__dict__"):
        return dict(vars(peer))
    return {}


def _load_observed_mesh_assets() -> list[dict[str, Any]]:
    try:
        if not _OBSERVED_MESH_ASSETS_FILE.exists():
            return []
        payload = json.loads(_OBSERVED_MESH_ASSETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows = payload.get("peers") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []

    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        host = _safe_host(str(row.get("host") or ""))
        if not host:
            continue
        peer = dict(row)
        peer["host"] = host
        peer.setdefault("transport", "observed")
        peer.setdefault("cpu_count", 1)
        peer.setdefault("cpu_load_1m", 0.0)
        peer.setdefault("free_ram_gb", 0.0)
        peer.setdefault("free_vram_gb", 0.0)
        peer.setdefault("total_ram_gb", peer.get("free_ram_gb", 0.0))
        peer.setdefault("drives", [])
        peer.setdefault("drive_count", len(peer.get("drives") or []))
        peer.setdefault("attached_drive_count", peer.get("drive_count", 0))
        peer.setdefault("free_disk_gb", 0.0)
        peer.setdefault("total_disk_gb", 0.0)
        peer.setdefault("gpus", [])
        peer.setdefault("ts", datetime.now(timezone.utc).isoformat())
        peer.setdefault("port", 0)
        peer.setdefault(
            "active",
            bool(
                _truthy_flag(peer.get("rdp_reachable"))
                or _truthy_flag(peer.get("mesh_listener_reachable"))
                or _truthy_flag(peer.get("connection_proven"))
            ),
        )
        out.append(peer)
    return out


def _load_peers(
    peers: Iterable[Any] | None,
    local_capacity: Any | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_hosts: set[str] = set()

    def add(peer_obj: Any) -> None:
        peer = _as_peer_dict(peer_obj)
        host = _safe_host(str(peer.get("host") or "unknown"))
        if not peer or host in seen_hosts:
            return
        peer["host"] = host
        out.append(peer)
        seen_hosts.add(host)

    if peers is not None:
        for peer in peers:
            add(peer)
    else:
        for peer in _load_observed_mesh_assets():
            add(peer)

    if local_capacity is not None:
        add(local_capacity)

    if peers is None and not out:
        try:
            from .compute_grid import discover_peers, publish_local_capacity

            try:
                add(publish_local_capacity(ensure_listener=False))
            except Exception:
                pass
            for peer in discover_peers():
                add(peer)
        except Exception:
            pass

    return out


def _asset_capacity(peer: dict[str, Any], kind: str) -> tuple[float, float, float]:
    cpu_count = max(0.0, float(peer.get("cpu_count") or 0.0))
    cpu_load = min(1.0, max(0.0, float(peer.get("cpu_load_1m") or 0.0)))
    if kind == "cores":
        available = max(0.0, cpu_count * (1.0 - cpu_load))
        return available, cpu_count, cpu_load
    if kind == "ram":
        free_ram = max(0.0, float(peer.get("free_ram_gb") or 0.0))
        total_ram = max(0.0, float(peer.get("total_ram_gb") or 0.0))
        return free_ram, max(total_ram, free_ram, 1.0), 0.0
    if kind == "vram":
        free_vram = max(0.0, float(peer.get("free_vram_gb") or 0.0))
        total_vram = 0.0
        for gpu in peer.get("gpus") or []:
            try:
                total_vram += float(gpu.get("total_mb") or gpu.get("free_mb") or 0.0) / 1024.0
            except Exception:
                continue
        return free_vram, max(total_vram, free_vram, 1.0), 0.0
    if kind == "storage":
        free_disk = max(0.0, float(peer.get("free_disk_gb") or 0.0))
        total_disk = max(0.0, float(peer.get("total_disk_gb") or 0.0))
        if (free_disk <= 0.0 or total_disk <= 0.0) and isinstance(peer.get("drives"), list):
            drive_free = 0.0
            drive_total = 0.0
            for drive in peer.get("drives") or []:
                if not isinstance(drive, dict):
                    continue
                try:
                    drive_free += max(0.0, float(drive.get("free_gb") or 0.0))
                    drive_total += max(0.0, float(drive.get("total_gb") or 0.0))
                except Exception:
                    continue
            free_disk = max(free_disk, drive_free)
            total_disk = max(total_disk, drive_total)
        load = 0.0
        if total_disk > 0.0:
            load = min(1.0, max(0.0, 1.0 - (free_disk / total_disk)))
        return free_disk, max(total_disk, free_disk, 1.0), load
    return 0.0, 0.0, 0.0


def _capacity_score(kind: str, available: float, total: float, load: float) -> float:
    if kind == "cores":
        saturation = min(1.0, available / 16.0)
        return max(0.0, min(1.0, saturation * (1.0 - 0.35 * load)))
    if kind == "ram":
        return max(0.0, min(1.0, available / 64.0))
    if kind == "vram":
        return max(0.0, min(1.0, available / 24.0))
    if kind == "storage":
        return max(0.0, min(1.0, (available / 1024.0) * (1.0 - 0.20 * load)))
    return 0.0


def _kind_angle(kind: str) -> float:
    idx = RESOURCE_KINDS.index(kind) if kind in RESOURCE_KINDS else 0
    return 2.0 * math.pi * idx / len(RESOURCE_KINDS)


def _asset_angles(host: str, kind: str, score: float, available: float, load: float) -> list[float]:
    host_angle = 2.0 * math.pi * _unit_hash(host, kind)
    cap_angle = 2.0 * math.pi * max(0.0, min(1.0, score))
    avail_angle = 2.0 * math.pi * (math.log1p(max(0.0, available)) / math.log1p(128.0))
    return [
        host_angle,
        _kind_angle(kind),
        cap_angle,
        min(2.0 * math.pi, avail_angle),
        2.0 * math.pi * max(0.0, min(1.0, load)),
        2.0 * math.pi * _unit_hash(kind, "material"),
        2.0 * math.pi * _unit_hash(host, "entirety"),
    ]


def _peer_entity_id(host: str) -> str:
    return f"compute_peer:{_safe_host(host)}"


def _asset_entity_id(host: str, kind: str) -> str:
    return f"asset:{_safe_host(host)}:{kind}"


def _peer_runtime(peer: dict[str, Any]) -> dict[str, Any]:
    runtime = peer.get("runtime") or {}
    if not isinstance(runtime, dict):
        runtime = {}
    out: dict[str, Any] = {}
    for key in (
        "runtime_pid",
        "agent_pid",
        "watcher_pid",
        "publisher_pid",
        "agent_state",
        "restarts",
        "updated_at",
        "published_at",
    ):
        value = runtime.get(key)
        if value not in (None, ""):
            out[key] = value
    if "runtime_pid" not in out:
        out["runtime_pid"] = out.get("agent_pid") or out.get("publisher_pid")
    if "updated_at" not in out:
        observed_at = peer.get("observed_at") or peer.get("ts")
        if observed_at not in (None, ""):
            out["updated_at"] = observed_at
    if "published_at" not in out:
        published_at = peer.get("ts") or peer.get("observed_at")
        if published_at not in (None, ""):
            out["published_at"] = published_at

    active = runtime.get("active")
    if active is None and runtime:
        active = _runtime_active(runtime)
    if active is None:
        active = peer.get("active")
    if active is None and str(peer.get("transport") or "").strip().lower() == "observed":
        active = bool(
            _truthy_flag(peer.get("rdp_reachable"))
            or _truthy_flag(peer.get("mesh_listener_reachable"))
            or _truthy_flag(peer.get("connection_proven"))
        )
    if active is not None:
        out["active"] = _truthy_flag(active)
    if "agent_state" not in out and str(peer.get("transport") or "").strip().lower() == "observed":
        out["agent_state"] = "observed_active" if out.get("active") else "observed"
    return {key: value for key, value in out.items() if value not in (None, "")}


def _latest_kv_prefix(cn: sqlite3.Connection, prefix: str) -> tuple[str, Any] | None:
    try:
        row = cn.execute(
            "SELECT key, value FROM kv_store WHERE key LIKE ? ORDER BY rowid DESC LIMIT 1",
            (f"{prefix}%",),
        ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    return str(row[0]), row[1]


def _kv_set(cn: sqlite3.Connection, key: str, value: str) -> None:
    cn.execute(
        "INSERT INTO kv_store(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _kv_get(cn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    try:
        row = cn.execute("SELECT value FROM kv_store WHERE key=?", (key,)).fetchone()
    except sqlite3.Error:
        row = None
    if not row:
        return default
    return row[0]


def _existing_torus_amplify(cn: sqlite3.Connection, *keys: str) -> float:
    strongest = 0.0
    for key in keys:
        try:
            row = cn.execute("SELECT value FROM kv_store WHERE key=?", (key,)).fetchone()
        except sqlite3.Error:
            row = None
        if not row or row[0] in (None, ""):
            continue
        try:
            strongest = max(strongest, float(row[0]))
        except Exception:
            continue
    return strongest


def _vpn_credential_available() -> bool:
    override = os.environ.get("SCB_VPN_VAULT_FILE", "").strip()
    if override:
        return Path(override).expanduser().exists()
    appdata = os.environ.get("APPDATA", "").strip()
    if not appdata:
        return False
    return (Path(appdata) / "SCBrain" / "vpn_cred.bin").exists()


def _normalize_make_buy(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith("MAKE"):
        return "MAKE"
    if text.startswith("BUY"):
        return "BUY"
    return text


def _maybe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _distinct_ordered(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _coerce_gpu_part_record(
    raw: Any,
    *,
    source_key: str,
    default_target: str,
    default_host: str = "",
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"part_number": raw}
    if not isinstance(raw, dict):
        return None

    part_number = str(
        raw.get("part_number")
        or raw.get("part")
        or raw.get("item")
        or raw.get("component")
        or raw.get("label")
        or ""
    ).strip()
    make_buy = _normalize_make_buy(
        raw.get("make_buy")
        or raw.get("make_buy_indicator")
        or raw.get("planning_make_buy_code")
        or raw.get("make_or_buy")
    )
    material = str(
        raw.get("material")
        or raw.get("component_material")
        or raw.get("receiver_material")
        or raw.get("metal")
        or raw.get("material_class")
        or ""
    ).strip().lower()
    target = str(
        raw.get("target")
        or raw.get("asset_id")
        or raw.get("asset")
        or raw.get("receiver_target")
        or default_target
        or ""
    ).strip()
    host = str(
        raw.get("host")
        or raw.get("receiver_host")
        or raw.get("asset_host")
        or default_host
        or ""
    ).strip()
    if not host and target.startswith("asset:"):
        parts = target.split(":")
        if len(parts) >= 4:
            host = parts[1]
    host = _safe_host(host) if host else ""
    bit_flip_node = _maybe_int(
        raw.get("bit_flip_node")
        or raw.get("flip_node")
        or raw.get("node_id")
        or raw.get("peak_node")
    )
    directed_target = _maybe_int(
        raw.get("bit_flip_target")
        or raw.get("bit_flip_target_node")
        or raw.get("directed_target")
    )
    bit_flip_weight = _maybe_float(
        raw.get("bit_flip_weight")
        or raw.get("resuscitation_weight")
        or raw.get("interaction_gain")
        or raw.get("weight")
    )
    connected = bool(
        raw.get("connected")
        or raw.get("active")
        or raw.get("connection_proven")
    )
    if not any(
        (
            part_number,
            make_buy,
            material,
            bit_flip_node is not None,
            directed_target is not None,
            bit_flip_weight is not None,
        )
    ):
        return None
    return {
        "part_number": part_number,
        "make_buy": make_buy,
        "material": material,
        "target": target,
        "host": host,
        "bit_flip_node": bit_flip_node,
        "directed_target": directed_target,
        "bit_flip_weight": bit_flip_weight,
        "connected": connected,
        "source_key": source_key,
    }


def _merge_gpu_part_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        key = (
            str(record.get("part_number") or "").strip().lower(),
            _normalize_make_buy(record.get("make_buy")),
            str(record.get("material") or "").strip().lower(),
            str(record.get("target") or "").strip().lower(),
            str(record.get("host") or "").strip().lower(),
            _maybe_int(record.get("bit_flip_node")),
            _maybe_int(record.get("directed_target")),
            str(record.get("source_key") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
    return merged


def _payload_gpu_part_records(
    payload: Any,
    *,
    source_key: str,
    default_target: str,
    default_host: str = "",
    depth: int = 0,
) -> list[dict[str, Any]]:
    if depth > 2 or not isinstance(payload, dict):
        return []

    records: list[dict[str, Any]] = []
    has_direct_part_signal = any(
        payload.get(key) not in (None, "", [], {})
        for key in (
            "part_number",
            "part",
            "item",
            "component",
            "make_buy",
            "make_buy_indicator",
            "planning_make_buy_code",
            "material",
            "component_material",
            "receiver_material",
            "metal",
            "bit_flip_node",
            "flip_node",
            "node_id",
            "peak_node",
            "bit_flip_target",
            "directed_target",
        )
    )
    if has_direct_part_signal:
        direct = _coerce_gpu_part_record(
            payload,
            source_key=source_key,
            default_target=default_target,
            default_host=default_host,
        )
        if direct is not None:
            records.append(direct)

    for key in (
        "gpu_parts",
        "parts",
        "receiver_parts",
        "part_records",
        "historic_parts",
        "component_parts",
        "bit_flip_parts",
        "interaction_parts",
    ):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                records.extend(
                    _payload_gpu_part_records(
                        row,
                        source_key=source_key,
                        default_target=str(row.get("target") or default_target or ""),
                        default_host=str(row.get("host") or default_host or ""),
                        depth=depth + 1,
                    )
                )
            else:
                coerced = _coerce_gpu_part_record(
                    row,
                    source_key=source_key,
                    default_target=default_target,
                    default_host=default_host,
                )
                if coerced is not None:
                    records.append(coerced)

    for key in ("tunnel_history", "interaction_log", "history", "events", "bit_flips"):
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            records.extend(
                _payload_gpu_part_records(
                    row,
                    source_key=source_key,
                    default_target=str(row.get("target") or default_target or ""),
                    default_host=str(row.get("host") or default_host or ""),
                    depth=depth + 1,
                )
            )

    return _merge_gpu_part_records(records)


def _historic_gpu_receiver_paths(
    cn: sqlite3.Connection,
    vram_assets: list[AssetVector],
) -> dict[str, Any]:
    active_ids = {asset.entity_id for asset in vram_assets}
    active_hosts = {asset.host for asset in vram_assets}
    vault_available = _vpn_credential_available()
    history: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_score = -1.0

    try:
        rows = cn.execute(
            "SELECT key, value FROM kv_store WHERE key LIKE ? ORDER BY rowid DESC LIMIT 16",
            ("vram_rehydrate:%",),
        ).fetchall()
    except sqlite3.Error:
        rows = []

    for key, value in rows:
        payload = _json_load(value, {})
        target = str(payload.get("target") or "").strip()
        if not target:
            continue
        host = ""
        if target.startswith("asset:"):
            parts = target.split(":")
            if len(parts) >= 4:
                host = _safe_host(parts[1])
        part_records = _payload_gpu_part_records(
            payload,
            source_key=str(key),
            default_target=target,
            default_host=host,
        )
        materials = _distinct_ordered(record.get("material") for record in part_records)
        make_buy_modes = _distinct_ordered(record.get("make_buy") for record in part_records)
        active = target in active_ids or (bool(host) and host in active_hosts)
        candidate = {
            "source_key": str(key),
            "target": target,
            "host": host,
            "active": bool(active),
            "part_count": len(part_records),
            "materials": materials,
            "make_buy_modes": make_buy_modes,
            "part_records": part_records[:8],
        }
        history.append(candidate)
        if vault_available and active:
            candidate_score = 1.0
            if any("tantalum" in material.lower() for material in materials):
                candidate_score += 0.20
            if "MAKE" in make_buy_modes:
                candidate_score += 0.15
            candidate_score += min(0.20, len(part_records) * 0.05)
            if candidate_score > selected_score:
                selected = candidate
                selected_score = candidate_score

    return {
        "credential_mode": "vault" if vault_available else "none",
        "selected_source_key": str((selected or {}).get("source_key") or ""),
        "selected_target": str((selected or {}).get("target") or ""),
        "historic_targets": history,
    }


def _receiver_part_records(
    receiver_history: dict[str, Any],
    *,
    source_key: str,
    source_label: str,
    target_hosts: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    matched = False
    for candidate in receiver_history.get("historic_targets") or []:
        if not isinstance(candidate, dict):
            continue
        candidate_source_key = str(candidate.get("source_key") or "")
        candidate_target = str(candidate.get("target") or "")
        candidate_host = _safe_host(candidate.get("host") or "") if candidate.get("host") else ""
        if (
            candidate_source_key == source_key
            or candidate_target == source_label
            or (candidate_host and candidate_host in target_hosts)
        ):
            matched = True
            records.extend(candidate.get("part_records") or [])
    if not matched:
        for candidate in receiver_history.get("historic_targets") or []:
            if not isinstance(candidate, dict) or not candidate.get("active"):
                continue
            records.extend(candidate.get("part_records") or [])
    return _merge_gpu_part_records(records)


def _build_bit_flip_resuscitation(
    part_records: list[dict[str, Any]],
    *,
    asset: AssetVector,
    source_key: str,
    source_label: str,
    peak_node: int,
    peak_target: int,
    receiver_material: str,
    binding_gain: float,
) -> dict[str, Any]:
    receiver_material = str(receiver_material or "").strip().lower()
    fallback_target = str(source_label or asset.entity_id or "")
    fallback = {
        "part_number": "",
        "make_buy": "",
        "material": receiver_material,
        "target": fallback_target,
        "host": asset.host,
        "bit_flip_node": peak_node,
        "directed_target": peak_target,
        "bit_flip_weight": None,
        "connected": bool(source_key or source_label),
        "source_key": source_key,
    }
    candidate_records = _merge_gpu_part_records([*part_records, fallback])

    best_record = fallback
    best_score = -1.0
    for record in candidate_records:
        target = str(record.get("target") or "")
        host = _safe_host(record.get("host") or "") if record.get("host") else ""
        material = str(record.get("material") or "").strip().lower()
        make_buy = _normalize_make_buy(record.get("make_buy"))
        connection_proven = bool(record.get("connected")) or target in {asset.entity_id, source_label} or (host and host == asset.host)
        vram_touch = ":vram" in target or (connection_proven and asset.kind == "vram")
        material_match = bool(receiver_material) and (not material or receiver_material in material)
        score = 0.0
        score += 0.30 if connection_proven else 0.0
        score += 0.25 if material_match else 0.0
        score += 0.20 if vram_touch else 0.0
        score += 0.15 if make_buy == "MAKE" else 0.05 if make_buy == "BUY" else 0.0
        score += 0.10 if record.get("bit_flip_node") is not None or record.get("directed_target") is not None else 0.0
        score += 0.05 if record.get("part_number") else 0.0
        if score > best_score:
            best_record = record
            best_score = score

    selected_make_buy = _normalize_make_buy(best_record.get("make_buy"))
    selected_material = str(best_record.get("material") or receiver_material or "").strip().lower()
    selected_target = str(best_record.get("target") or fallback_target or "")
    selected_host = _safe_host(best_record.get("host") or asset.host)
    connection_proven = bool(best_record.get("connected")) or selected_target in {asset.entity_id, source_label} or selected_host == asset.host
    vram_touch = ":vram" in selected_target or (connection_proven and asset.kind == "vram")
    material_match = bool(receiver_material) and (not selected_material or receiver_material in selected_material)
    bit_flip_node = _maybe_int(best_record.get("bit_flip_node"))
    bit_flip_target = _maybe_int(best_record.get("directed_target"))
    interaction_gain = _clip01(
        0.32 * (1.0 if material_match else 0.35)
        + 0.28 * (1.0 if connection_proven else 0.0)
        + 0.24 * (1.0 if vram_touch else 0.0)
        + 0.16 * (1.0 if selected_make_buy == "MAKE" else 0.5 if selected_make_buy == "BUY" else 0.25)
    )
    bridge_reactivation_gain = _clip01(0.65 * interaction_gain + 0.35 * _clip01(binding_gain))
    resuscitation_multiplier = round(
        max(
            1.0,
            1.0 + 0.18 * bridge_reactivation_gain + (0.07 if selected_make_buy == "MAKE" else 0.0),
        ),
        6,
    )
    return {
        "active": True,
        "selected_part": str(best_record.get("part_number") or ""),
        "selected_make_buy": selected_make_buy,
        "selected_material": selected_material,
        "receiver_material": receiver_material,
        "connection_proven": connection_proven,
        "vram_touch": vram_touch,
        "material_match": material_match,
        "bit_flip_node": bit_flip_node if bit_flip_node is not None else int(peak_node),
        "bit_flip_target": bit_flip_target if bit_flip_target is not None else int(peak_target),
        "bit_flip_weight": round(_clip01(best_record.get("bit_flip_weight") or interaction_gain), 6),
        "interaction_gain": round(interaction_gain, 6),
        "bridge_reactivation_gain": round(bridge_reactivation_gain, 6),
        "resuscitation_multiplier": resuscitation_multiplier,
        "part_count": len(candidate_records),
        "part_numbers": _distinct_ordered(record.get("part_number") for record in candidate_records),
        "make_buy_modes": _distinct_ordered(record.get("make_buy") for record in candidate_records),
        "materials": _distinct_ordered(record.get("material") for record in candidate_records),
        "selected_source_key": str(best_record.get("source_key") or source_key or ""),
    }


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
        " label=excluded.label, props_json=excluded.props_json, "
        " last_seen=excluded.last_seen, samples=COALESCE(samples, 0)+1",
        (entity_id, entity_type, label, json.dumps(props), now, now),
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
) -> bool:
    before = cn.total_changes
    cn.execute(
        "INSERT INTO corpus_edge(src_id, src_type, dst_id, dst_type, rel, "
        " weight, last_seen, samples) VALUES(?,?,?,?,?,?,?,1) "
        "ON CONFLICT(src_id, src_type, dst_id, dst_type, rel) DO UPDATE SET "
        " weight=(COALESCE(corpus_edge.weight, 0.0) * COALESCE(corpus_edge.samples, 1) "
        "         + excluded.weight) / (COALESCE(corpus_edge.samples, 1) + 1), "
        " last_seen=excluded.last_seen, samples=COALESCE(corpus_edge.samples, 0)+1",
        (src_id, src_type, dst_id, dst_type, rel, float(weight), now),
    )
    return cn.total_changes > before


def _material_processor_rows(cn: sqlite3.Connection) -> list[tuple[str, dict[str, Any]]]:
    rows = cn.execute(
        "SELECT entity_id, props_json FROM corpus_entity WHERE entity_type=?",
        (SPATIAL_MATERIAL_PROCESSOR,),
    ).fetchall()
    out: list[tuple[str, dict[str, Any]]] = []
    for entity_id, props_json in rows:
        try:
            props = json.loads(props_json) if props_json else {}
        except Exception:
            props = {}
        out.append((str(entity_id), props))
    return out


def _processor_strength(props: dict[str, Any]) -> float:
    actualized = props.get("actualized_space") or {}
    world_pose = props.get("world_pose") or {}
    precision = float(actualized.get("precision") or world_pose.get("precision") or 0.0)
    prob_cert = float(actualized.get("prob_cert") or world_pose.get("prob_cert") or 0.0)
    material_structure = actualized.get("material_structure") or {}
    lattice = float(material_structure.get("lattice_strength") or (precision / (precision + 1.0) if precision > 0 else 0.0))
    return max(0.0, min(1.0, 0.45 * (precision / (precision + 1.0) if precision > 0 else 0.0) + 0.35 * prob_cert + 0.20 * lattice))


def _kind_affinity(kind: str, processor_props: dict[str, Any]) -> float:
    actualized = processor_props.get("actualized_space") or {}
    material_class = str(actualized.get("material_class") or "").lower()
    if kind == "vram" and any(token in material_class for token in ("vision", "ocr", "graph", "spatial")):
        return 1.0
    if kind == "ram" and any(token in material_class for token in ("inventory", "document", "material")):
        return 0.9
    if kind == "storage" and any(token in material_class for token in ("inventory", "document", "material", "archive", "spatial")):
        return 0.95
    if kind == "cores":
        return 0.8
    return 0.7


def _asset_payload(peer: dict[str, Any], asset: AssetVector, now: str) -> dict[str, Any]:
    runtime = _peer_runtime(peer)
    return {
        "asset_kind": asset.kind,
        "host": asset.host,
        "available": asset.available,
        "total": asset.total,
        "load": asset.load,
        "score": asset.score,
            "unit": "cores" if asset.kind == "cores" else "gb",
        "frame_id": ACTUALIZED_SPACE_FRAME,
        "actualized_resource": {
            "version": "asset-resource-mesh/1.0",
            "resource_kind": asset.kind,
            "host": asset.host,
            "capacity_available": asset.available,
            "capacity_total": asset.total,
            "capacity_score": asset.score,
            "physical_realization_axis": "compute" if asset.kind == "cores" else asset.kind,
            "standardized_at": now,
        },
        "torus_angles": asset.torus_angles,
        "share_contract": {
            "unit_family": _resource_family(asset.kind),
            "unit_kind": asset.kind,
            "host": asset.host,
            "transport": peer.get("transport", "tcp"),
            "address": peer.get("address"),
            "port": peer.get("port"),
            "tunnel_id": peer.get("tunnel_id", ""),
            "capacity_available": asset.available,
            "capacity_total": asset.total,
            "capacity_score": asset.score,
            "runtime": runtime,
            "runtime_pid": runtime.get("runtime_pid"),
            "agent_pid": runtime.get("agent_pid"),
            "watcher_pid": runtime.get("watcher_pid"),
            "share_mode": "compute_grid_job",
            "binding_rel": REALIZES_PROCESSOR_REL,
        },
        "peer": {
            "address": peer.get("address"),
            "port": peer.get("port"),
            "transport": peer.get("transport", "tcp"),
            "tunnel_id": peer.get("tunnel_id", ""),
            "runtime": runtime,
        },
    }


def resource_share_contracts(
    peers: Iterable[Any] | None = None,
    *,
    local_capacity: Any | None = None,
) -> list[dict[str, Any]]:
    """Return Core/Memory share contracts for the current compute peers."""

    contracts: list[dict[str, Any]] = []
    for peer in _load_peers(peers, local_capacity):
        host = _safe_host(str(peer.get("host") or "unknown"))
        runtime = _peer_runtime(peer)
        for kind in RESOURCE_KINDS:
            available, total, load = _asset_capacity(peer, kind)
            if kind == "vram" and available <= 0.0:
                continue
            if kind in {"cores", "ram", "storage"} and total <= 0.0:
                continue
            score = _capacity_score(kind, available, total, load)
            if score <= 0.0 and kind != "cores":
                continue
            contracts.append(
                {
                    "entity_id": _asset_entity_id(host, kind),
                    "unit_family": _resource_family(kind),
                    "unit_kind": kind,
                    "host": host,
                    "transport": peer.get("transport", "tcp"),
                    "address": peer.get("address"),
                    "port": peer.get("port"),
                    "tunnel_id": peer.get("tunnel_id", ""),
                    "runtime": runtime,
                    "runtime_pid": runtime.get("runtime_pid"),
                    "agent_pid": runtime.get("agent_pid"),
                    "watcher_pid": runtime.get("watcher_pid"),
                    "capacity_available": float(available),
                    "capacity_total": float(total),
                    "capacity_score": float(score),
                    "drive_count": int(peer.get("drive_count") or len(peer.get("drives") or [])) if kind == "storage" else 0,
                    "attached_drive_count": int(peer.get("attached_drive_count") or peer.get("drive_count") or len(peer.get("drives") or [])) if kind == "storage" else 0,
                    "load": float(load),
                    "share_mode": "compute_grid_job",
                    "binding_rel": REALIZES_PROCESSOR_REL,
                }
            )
    contracts.sort(
        key=lambda row: (
            str(row.get("unit_family", "")),
            -float(row.get("capacity_score", 0.0)),
            str(row.get("host", "")),
        )
    )
    return contracts


def _mesh_sharing_summary(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    hosts = {
        str(contract.get("host") or "")
        for contract in contracts
        if contract.get("host")
    }
    tunnel_hosts = {
        str(contract.get("host") or "")
        for contract in contracts
        if contract.get("host") and contract.get("tunnel_id")
    }
    transports = {
        str(contract.get("transport") or "")
        for contract in contracts
        if contract.get("transport")
    }
    peer_count = len(hosts)
    contract_count = len(contracts)
    transport_count = len(transports)
    peer_density = min(1.0, peer_count / 4.0)
    contract_density = min(1.0, contract_count / max(1, peer_count * len(RESOURCE_KINDS)))
    tunnel_peer_ratio = min(1.0, len(tunnel_hosts) / max(1, peer_count))
    transport_diversity = min(1.0, transport_count / max(1, min(peer_count, len(RESOURCE_KINDS))))
    mesh_density = min(
        1.0,
        0.30 * peer_density
        + 0.30 * contract_density
        + 0.20 * tunnel_peer_ratio
        + 0.20 * transport_diversity,
    )
    peer_runtime: list[dict[str, Any]] = []
    seen_runtime_hosts: set[str] = set()
    for contract in contracts:
        host = str(contract.get("host") or "")
        runtime = contract.get("runtime") or {}
        if not host or host in seen_runtime_hosts or not isinstance(runtime, dict) or not runtime:
            continue
        active = _runtime_active(runtime)
        peer_runtime.append(
            {
                "host": host,
                "runtime_pid": runtime.get("runtime_pid"),
                "agent_pid": runtime.get("agent_pid"),
                "watcher_pid": runtime.get("watcher_pid"),
                "agent_state": runtime.get("agent_state"),
                "updated_at": runtime.get("updated_at") or runtime.get("published_at"),
                "active": active,
            }
        )
        seen_runtime_hosts.add(host)
    active_hosts = sorted(
        {
            str(entry.get("host") or "")
            for entry in peer_runtime
            if entry.get("host") and _truthy_flag(entry.get("active"))
        }
    )
    active_runtime_pids = sorted(
        {
            int(entry["runtime_pid"])
            for entry in peer_runtime
            if _truthy_flag(entry.get("active")) and entry.get("runtime_pid") not in (None, "")
        }
    )
    return {
        "distributed": peer_count > 1,
        "peer_count": peer_count,
        "contract_count": contract_count,
        "transport_count": transport_count,
        "transport_modes": sorted(transports),
        "tunnel_peer_count": len(tunnel_hosts),
        "tunnel_peer_ratio": tunnel_peer_ratio,
        "transport_diversity": transport_diversity,
        "mesh_density": mesh_density,
        "peer_runtime": peer_runtime,
        "active_peer_count": len(active_hosts),
        "active_hosts": active_hosts,
        "active_runtime_pids": active_runtime_pids,
    }


def materialize_asset_resources(
    cn: sqlite3.Connection,
    peers: Iterable[Any] | None = None,
    *,
    local_capacity: Any | None = None,
) -> list[AssetVector]:
    """Materialize ComputePeer and AssetResource corpus nodes."""
    _ensure_tables(cn)
    now = datetime.now(timezone.utc).isoformat()
    peer_dicts = _load_peers(peers, local_capacity)
    assets: list[AssetVector] = []

    for peer in peer_dicts:
        host = _safe_host(str(peer.get("host") or "unknown"))
        peer_id = _peer_entity_id(host)
        runtime = _peer_runtime(peer)
        _upsert_entity(
            cn,
            peer_id,
            COMPUTE_PEER_TYPE,
            f"ComputePeer {host}",
            {
                "host": host,
                "address": peer.get("address"),
                "port": peer.get("port"),
                "cpu_count": peer.get("cpu_count", 0),
                "cpu_load_1m": peer.get("cpu_load_1m", 0.0),
                "free_ram_gb": peer.get("free_ram_gb", 0.0),
                "total_ram_gb": peer.get("total_ram_gb", 0.0),
                "free_vram_gb": peer.get("free_vram_gb", 0.0),
                "drives": peer.get("drives", []),
                "drive_count": peer.get("drive_count", len(peer.get("drives") or [])),
                "attached_drive_count": peer.get("attached_drive_count", peer.get("drive_count", len(peer.get("drives") or []))),
                "free_disk_gb": peer.get("free_disk_gb", 0.0),
                "total_disk_gb": peer.get("total_disk_gb", 0.0),
                "transport": peer.get("transport", "tcp"),
                "tunnel_id": peer.get("tunnel_id", ""),
                "runtime": runtime,
                "published_at": peer.get("ts") or now,
            },
            now,
        )
        for kind in RESOURCE_KINDS:
            available, total, load = _asset_capacity(peer, kind)
            if kind == "vram" and available <= 0.0:
                continue
            if kind in {"cores", "ram", "storage"} and total <= 0.0:
                continue
            score = _capacity_score(kind, available, total, load)
            if score <= 0.0 and kind != "cores":
                continue
            asset = AssetVector(
                entity_id=_asset_entity_id(host, kind),
                host=host,
                kind=kind,
                available=float(available),
                total=float(total),
                load=float(load),
                score=float(score),
                torus_angles=_asset_angles(host, kind, score, available, load),
            )
            _upsert_entity(
                cn,
                asset.entity_id,
                ASSET_ENTITY_TYPE,
                f"{host} {kind.upper()} asset",
                _asset_payload(peer, asset, now),
                now,
            )
            _upsert_edge(cn, peer_id, COMPUTE_PEER_TYPE, asset.entity_id, ASSET_ENTITY_TYPE,
                         HAS_ASSET_REL, max(0.05, asset.score), now)
            assets.append(asset)

    return assets


def _mesh_weight(a: AssetVector, b: AssetVector, material_factor: float) -> float:
    same_host = a.host == b.host
    same_kind = a.kind == b.kind
    topology = 1.0 if same_host else (0.75 if same_kind else 0.45)
    capacity = math.sqrt(max(0.0, a.score) * max(0.0, b.score))
    return max(0.01, min(1.0, capacity * topology * (0.55 + 0.45 * material_factor)))


def mesh_asset_tunnels(
    cn: sqlite3.Connection,
    assets: list[AssetVector],
    *,
    max_tunnels: int = 96,
) -> dict[str, int]:
    """Mint resource-to-resource tunnels across the asset substrate."""
    now = datetime.now(timezone.utc).isoformat()
    processors = _material_processor_rows(cn)
    material_factor = 0.0
    if processors:
        material_factor = sum(_processor_strength(p) for _pid, p in processors) / len(processors)

    pairs: list[tuple[AssetVector, AssetVector, float]] = []
    for i, a in enumerate(assets):
        for b in assets[i + 1:]:
            if a.host != b.host and a.kind != b.kind:
                continue
            weight = _mesh_weight(a, b, material_factor)
            pairs.append((a, b, weight))
    pairs.sort(key=lambda row: row[2], reverse=True)

    tunnels = 0
    for a, b, weight in pairs[:max_tunnels]:
        _upsert_edge(cn, a.entity_id, ASSET_ENTITY_TYPE, b.entity_id, ASSET_ENTITY_TYPE,
                     ASSET_TUNNEL_REL, weight, now)
        _upsert_edge(cn, b.entity_id, ASSET_ENTITY_TYPE, a.entity_id, ASSET_ENTITY_TYPE,
                     ASSET_TUNNEL_REL, weight, now)
        tunnels += 2
    return {"asset_tunnels": tunnels}


def bind_assets_to_material_processors(
    cn: sqlite3.Connection,
    assets: list[AssetVector],
) -> dict[str, int]:
    """Create realization edges from assets to spatial material processors."""
    now = datetime.now(timezone.utc).isoformat()
    processors = _material_processor_rows(cn)
    edges = 0
    for asset in assets:
        for processor_id, processor_props in processors:
            processor_strength = _processor_strength(processor_props)
            if processor_strength <= 0.0:
                continue
            affinity = _kind_affinity(asset.kind, processor_props)
            weight = max(0.01, min(1.0, asset.score * processor_strength * affinity))
            _upsert_edge(cn, asset.entity_id, ASSET_ENTITY_TYPE,
                         processor_id, SPATIAL_MATERIAL_PROCESSOR,
                         REALIZES_PROCESSOR_REL, weight, now)
            edges += 1
    return {"processor_edges": edges}


def _preserved_mesh_learning(cn: sqlite3.Connection) -> dict[str, Any]:
    for table_name, key in (
        ("brain_kv", "entirety:physical_realization"),
        ("kv_store", "asset_resource_mesh_last"),
    ):
        try:
            row = cn.execute(f"SELECT value FROM {table_name} WHERE key=?", (key,)).fetchone()
            if not row or not row[0]:
                continue
            payload = json.loads(row[0])
            if not isinstance(payload, dict):
                continue
            preserved = {}
            for field in ("mesh_learning", "mesh_learning_projection", "mesh_learning_updated_at"):
                if field in payload:
                    preserved[field] = payload[field]
            if preserved:
                return preserved
        except Exception:
            continue
    return {}


def _project_gpu_resuscitation(
    cn: sqlite3.Connection,
    assets: list[AssetVector],
    *,
    mesh_density: float,
    physical_realization: float,
) -> dict[str, Any]:
    vram_assets = [asset for asset in assets if asset.kind == "vram" and asset.available > 0.0]
    if not vram_assets:
        return {}

    try:
        row = cn.execute(
            "SELECT COUNT(*), COALESCE(AVG(interaction_gain), 0.0), "
            "COALESCE(AVG(resuscitation_weight), 0.0), COALESCE(MAX(resuscitation_weight), 0.0), "
            "COALESCE(AVG(weyl_phase), 0.0) "
            "FROM mesh_slm_quipu_node"
        ).fetchone()
    except sqlite3.Error:
        return {}

    if not row:
        return {}

    node_count = int(row[0] or 0)
    if node_count <= 0:
        return {}

    peak = cn.execute(
        "SELECT node_id, directed_target, source_key, source_label, resuscitation_weight "
        "FROM mesh_slm_quipu_node "
        "ORDER BY resuscitation_weight DESC, interaction_gain DESC LIMIT 1"
    ).fetchone()
    peak_node = int(peak[0] or 0) if peak else 0
    peak_target = int(peak[1] or 0) if peak else 0
    peak_source_key = str(peak[2] or "") if peak else ""
    peak_source_label = str(peak[3] or "") if peak else ""

    latest_rehydrate = _latest_kv_prefix(cn, "vram_rehydrate:")
    latest_key = latest_rehydrate[0] if latest_rehydrate else peak_source_key
    latest_payload = _json_load(latest_rehydrate[1] if latest_rehydrate else None, {})
    source_key = latest_key or peak_source_key
    source_label = str(latest_payload.get("target") or peak_source_label or source_key or "resuscitation")
    receiver_history = _historic_gpu_receiver_paths(cn, vram_assets)

    targeted_assets = [asset for asset in vram_assets if asset.entity_id == source_label]
    if not targeted_assets and source_label.startswith("asset:"):
        parts = source_label.split(":")
        if len(parts) >= 4:
            target_host = _safe_host(parts[1])
            targeted_assets = [asset for asset in vram_assets if asset.host == target_host]
    if not targeted_assets:
        historic_source_key = str(receiver_history.get("selected_source_key") or "")
        historic_target = str(receiver_history.get("selected_target") or "")
        if historic_target:
            historic_payload = _json_load(_kv_get(cn, historic_source_key, "{}"), {})
            source_key = historic_source_key or source_key
            source_label = historic_target
            latest_payload = historic_payload or latest_payload
            targeted_assets = [asset for asset in vram_assets if asset.entity_id == source_label]
            if not targeted_assets and source_label.startswith("asset:"):
                parts = source_label.split(":")
                if len(parts) >= 4:
                    target_host = _safe_host(parts[1])
                    targeted_assets = [asset for asset in vram_assets if asset.host == target_host]
    if not targeted_assets:
        targeted_assets = vram_assets

    avg_interaction = _clip01(float(row[1] or 0.0))
    avg_resuscitation_weight = max(0.0, float(row[2] or 0.0))
    peak_weight = max(1e-6, float(row[3] or 0.0))
    weyl_phase = float(latest_payload.get("weyl") or latest_payload.get("weyl_phase") or row[4] or 0.0)
    nodal_coverage = _clip01(node_count / 4096.0)
    weight_ratio = _clip01(avg_resuscitation_weight / peak_weight)
    rehydrate_seed = float(latest_payload.get("rehydrate_weight") or (peak[4] if peak else 1.0) or 1.0)
    rehydrate_weight = max(0.05, rehydrate_seed)
    coherence_depth = max(0, int(round(node_count / 1024.0)))
    tantalum_binding: dict[str, Any] = {
        "intermediary": "tantalum",
        "weyl_phase": round(weyl_phase, 6),
        "coherence_depth": coherence_depth,
        "phase_weight": 1.0,
        "pulse_weight": round(float(latest_payload.get("weyl_boost") or 1.0), 6),
        "pulse_coupling": 0.5,
        "axial_channel": 0.0,
        "phase_evolution": 0.0,
        "overlap": 0.0,
        "mesh_alignment": round(mesh_density, 6),
        "observer_alignment": round(physical_realization, 6),
        "binding_gain": 0.0,
        "binding_multiplier": 1.0,
    }
    try:
        from .ueqgm_engine import get_adaptive_runtime, tantalum_intermediary_binding

        runtime = get_adaptive_runtime(cn)
        coherence_depth = max(0, int(runtime.get("coherence_depth") or coherence_depth))
        pulse_weight = float(
            runtime.get("phase_weight")
            or latest_payload.get("weyl_boost")
            or 1.0
        )
        tantalum_binding = tantalum_intermediary_binding(
            weyl_phase=weyl_phase,
            coherence=coherence_depth,
            mesh_alignment=float(runtime.get("mesh_alignment") or mesh_density),
            observer_alignment=float(runtime.get("observer_alignment") or physical_realization),
            pulse_weight=pulse_weight,
        )
        tantalum_binding["runtime_active"] = bool(runtime.get("active"))
        tantalum_binding["symbiotic_gain"] = round(_clip01(runtime.get("symbiotic_gain", 0.0)), 6)
    except Exception:
        pass
    now = datetime.now(timezone.utc).isoformat()

    profiles: list[dict[str, Any]] = []
    for asset in targeted_assets:
        binding_gain = _clip01(tantalum_binding.get("binding_gain", 0.0))
        binding_multiplier = max(1.0, float(tantalum_binding.get("binding_multiplier", 1.0) or 1.0))
        receiver_parts = _receiver_part_records(
            receiver_history,
            source_key=source_key,
            source_label=source_label,
            target_hosts={asset.host},
        )
        bit_flip_resuscitation = _build_bit_flip_resuscitation(
            receiver_parts,
            asset=asset,
            source_key=source_key,
            source_label=source_label,
            peak_node=peak_node,
            peak_target=peak_target,
            receiver_material="tantalum",
            binding_gain=binding_gain,
        )
        bit_flip_multiplier = max(
            1.0,
            float(bit_flip_resuscitation.get("resuscitation_multiplier") or 1.0),
        )
        hydration_ratio = _clip01(
            weight_ratio
            * (0.55 + 0.45 * avg_interaction)
            * (0.60 + 0.40 * binding_gain)
            * min(1.35, bit_flip_multiplier)
        )
        hydrated_vram_gb = round(asset.available * hydration_ratio, 6)
        compression_ratio = _clip01(
            (hydrated_vram_gb / max(asset.total, 1e-6))
            * (0.55 + 0.45 * mesh_density)
            * (0.55 + 0.45 * physical_realization)
            * (0.55 + 0.45 * min(1.0, binding_multiplier / 1.45))
            * (0.75 + 0.25 * _clip01(bit_flip_resuscitation.get("bridge_reactivation_gain", 0.0)))
        )
        computed_torus_amplify = round(
            max(
                0.05,
                rehydrate_weight
                * (0.45 + 0.55 * nodal_coverage)
                * (0.50 + 0.50 * avg_interaction)
                * (0.50 + 0.50 * compression_ratio),
                binding_multiplier,
                bit_flip_multiplier,
            ),
            6,
        )
        existing_torus_amplify = _existing_torus_amplify(
            cn,
            f"torus_amplify:{asset.entity_id}",
            f"torus_amplify:{ASSET_ENTITY_TYPE}:{asset.entity_id}",
        )
        effective_torus_amplify = round(max(existing_torus_amplify, computed_torus_amplify), 6)
        profile = {
            "entity_id": asset.entity_id,
            "host": asset.host,
            "receiver_material": "tantalum",
            "receiver_role": "gpu_receiver_metal",
            "node_count": node_count,
            "nodal_coverage": round(nodal_coverage, 6),
            "avg_interaction_gain": round(avg_interaction, 6),
            "avg_resuscitation_weight": round(avg_resuscitation_weight, 6),
            "peak_node": peak_node,
            "directed_target": peak_target,
            "weyl_phase": round(weyl_phase, 6),
            "hydration_ratio": round(hydration_ratio, 6),
            "hydrated_vram_gb": hydrated_vram_gb,
            "compression_ratio": round(compression_ratio, 6),
            "mesh_density": round(mesh_density, 6),
            "physical_realization": round(physical_realization, 6),
            "source_key": source_key,
            "source_label": source_label,
            "receiver_history": receiver_history,
            "receiver_parts": receiver_parts,
            "bit_flip_resuscitation": bit_flip_resuscitation,
            "tantalum_binding": tantalum_binding,
            "computed_torus_amplify": computed_torus_amplify,
            "effective_torus_amplify": effective_torus_amplify,
            "updated_at": now,
        }
        _kv_set(cn, f"gpu_resuscitation:{asset.entity_id}", json.dumps(profile, default=str))
        _kv_set(cn, f"torus_amplify:{asset.entity_id}", f"{effective_torus_amplify:.6f}")
        _kv_set(cn, f"torus_amplify:{ASSET_ENTITY_TYPE}:{asset.entity_id}", f"{effective_torus_amplify:.6f}")

        row_asset = cn.execute(
            "SELECT label, props_json FROM corpus_entity WHERE entity_id=? AND entity_type=?",
            (asset.entity_id, ASSET_ENTITY_TYPE),
        ).fetchone()
        label = str(row_asset[0] or f"{asset.host} {asset.kind.upper()} asset") if row_asset else f"{asset.host} {asset.kind.upper()} asset"
        props = _json_load(row_asset[1] if row_asset else None, {})
        actualized = props.get("actualized_resource") or {}
        if not isinstance(actualized, dict):
            actualized = {}
        actualized.update(
            {
                "hydrated_vram_gb": hydrated_vram_gb,
                "compression_ratio": round(compression_ratio, 6),
                "resuscitation_nodes": node_count,
                "resuscitation_gain": effective_torus_amplify,
                "resuscitation_source": source_label,
                "resuscitation_target": peak_target,
                "intermediary_binding": "tantalum",
                "receiver_material": "tantalum",
                "receiver_role": "gpu_receiver_metal",
                "weyl_binding_gain": round(binding_gain, 6),
                "bit_flip_node": bit_flip_resuscitation["bit_flip_node"],
                "bit_flip_target": bit_flip_resuscitation["bit_flip_target"],
                "bit_flip_interaction_gain": bit_flip_resuscitation["interaction_gain"],
                "bridge_reactivation_gain": bit_flip_resuscitation["bridge_reactivation_gain"],
                "part_make_buy": bit_flip_resuscitation["selected_make_buy"],
                "part_material": bit_flip_resuscitation["selected_material"],
            }
        )
        props["actualized_resource"] = actualized
        props["gpu_resuscitation"] = profile
        _upsert_entity(cn, asset.entity_id, ASSET_ENTITY_TYPE, label, props, now)
        profiles.append(profile)

    if not profiles:
        return {}

    return {
        "active": True,
        "receiver_material": "tantalum",
        "receiver_role": "gpu_receiver_metal",
        "node_count": node_count,
        "nodal_coverage": round(nodal_coverage, 6),
        "source_key": source_key,
        "source_label": source_label,
        "receiver_history": receiver_history,
        "weyl_phase": round(weyl_phase, 6),
        "peak_node": peak_node,
        "peak_directed_target": peak_target,
        "mesh_density": round(mesh_density, 6),
        "physical_realization": round(physical_realization, 6),
        "tantalum_binding": tantalum_binding,
        "bit_flip_resuscitation": {
            "bridge_reactivation_gain": round(
                max(p["bit_flip_resuscitation"]["bridge_reactivation_gain"] for p in profiles),
                6,
            ),
            "interaction_gain": round(
                sum(p["bit_flip_resuscitation"]["interaction_gain"] for p in profiles) / max(1, len(profiles)),
                6,
            ),
            "bit_flip_nodes": _distinct_ordered(p["bit_flip_resuscitation"]["bit_flip_node"] for p in profiles),
            "bit_flip_targets": _distinct_ordered(p["bit_flip_resuscitation"]["bit_flip_target"] for p in profiles),
            "part_numbers": _distinct_ordered(
                part
                for p in profiles
                for part in p["bit_flip_resuscitation"].get("part_numbers", [])
            ),
            "make_buy_modes": _distinct_ordered(
                mode
                for p in profiles
                for mode in p["bit_flip_resuscitation"].get("make_buy_modes", [])
            ),
            "materials": _distinct_ordered(
                material
                for p in profiles
                for material in p["bit_flip_resuscitation"].get("materials", [])
            ),
            "vram_touch": any(p["bit_flip_resuscitation"].get("vram_touch") for p in profiles),
            "connection_proven": any(p["bit_flip_resuscitation"].get("connection_proven") for p in profiles),
        },
        "hydrated_vram_gb": round(sum(p["hydrated_vram_gb"] for p in profiles), 6),
        "compression_ratio": round(
            sum(p["compression_ratio"] for p in profiles) / max(1, len(profiles)),
            6,
        ),
        "torus_amplify": round(max(p["effective_torus_amplify"] for p in profiles), 6),
        "targeted_assets": len(profiles),
        "asset_ids": [p["entity_id"] for p in profiles],
        "updated_at": now,
    }


def _write_system_realization(
    cn: sqlite3.Connection,
    assets: list[AssetVector],
    *,
    asset_tunnels: int,
    processor_edges: int,
    mesh_contracts: list[dict[str, Any]],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    processors = _material_processor_rows(cn)
    preserved_learning = _preserved_mesh_learning(cn)
    cores = sum(a.available for a in assets if a.kind == "cores")
    ram = sum(a.available for a in assets if a.kind == "ram")
    vram = sum(a.available for a in assets if a.kind == "vram")
    storage = sum(a.available for a in assets if a.kind == "storage")
    storage_total = sum(a.total for a in assets if a.kind == "storage")
    storage_contracts = [contract for contract in mesh_contracts if contract.get("unit_kind") == "storage"]
    drive_count = sum(max(1, int(contract.get("attached_drive_count") or contract.get("drive_count") or 0)) for contract in storage_contracts)
    asset_strength = (
        0.28 * min(1.0, cores / 32.0)
        + 0.27 * min(1.0, ram / 128.0)
        + 0.25 * min(1.0, vram / 48.0)
        + 0.20 * min(1.0, storage / 2048.0)
    )
    material_strength = 0.0
    if processors:
        material_strength = sum(_processor_strength(p) for _pid, p in processors) / len(processors)
    max_tunnels = max(1, len(assets) * max(0, len(assets) - 1))
    tunnel_density = min(1.0, asset_tunnels / max_tunnels)
    realization = max(0.0, min(1.0, 0.40 * asset_strength + 0.35 * material_strength + 0.25 * tunnel_density))
    mesh_sharing = _mesh_sharing_summary(mesh_contracts)
    mesh_runtime = mesh_sharing.get("peer_runtime", [])

    resource_sharing = {
        "core": {
            "available": cores,
            "units": sum(1 for asset in assets if asset.kind == "cores"),
            "hosts": len({asset.host for asset in assets if asset.kind == "cores"}),
            "share_mode": "compute_grid_job",
        },
        "memory": {
            "ram_available_gb": ram,
            "vram_available_gb": vram,
            "units": sum(1 for asset in assets if asset.kind in {"ram", "vram"}),
            "hosts": len({asset.host for asset in assets if asset.kind in {"ram", "vram"}}),
            "share_mode": "compute_grid_job",
        },
        "storage": {
            "free_disk_gb": storage,
            "total_disk_gb": storage_total,
            "units": sum(1 for asset in assets if asset.kind == "storage"),
            "drive_count": drive_count,
            "hosts": len({asset.host for asset in assets if asset.kind == "storage"}),
            "share_mode": "compute_grid_job",
        },
        "processor": {
            "processor_count": len(processors),
            "processor_edges": processor_edges,
            "binding_rel": REALIZES_PROCESSOR_REL,
        },
        "mesh": {
            **mesh_sharing,
            "share_mode": "compute_grid_job",
            "binding_rel": REALIZES_PROCESSOR_REL,
        },
    }
    gpu_resuscitation = _project_gpu_resuscitation(
        cn,
        assets,
        mesh_density=float(mesh_sharing.get("mesh_density", 0.0)),
        physical_realization=realization,
    )
    if gpu_resuscitation:
        resource_sharing["memory"]["gpu_resuscitation"] = gpu_resuscitation

    props = {
        "physical_realization": realization,
        "asset_strength": asset_strength,
        "material_strength": material_strength,
        "tunnel_density": tunnel_density,
        "cores_available": cores,
        "ram_available_gb": ram,
        "vram_available_gb": vram,
        "free_disk_gb": storage,
        "total_disk_gb": storage_total,
        "drive_count": drive_count,
        "asset_count": len(assets),
        "processor_count": len(processors),
        "asset_tunnels": asset_tunnels,
        "processor_edges": processor_edges,
        "resource_sharing": resource_sharing,
        "mesh_runtime": mesh_runtime,
        "updated_at": now,
    }
    if gpu_resuscitation:
        props["gpu_resuscitation"] = gpu_resuscitation
    props.update(preserved_learning)
    _upsert_entity(cn, SYSTEM_ENTIRETY_ID, SYSTEM_ENTIRETY_TYPE,
                   "System Entirety Physical Realization", props, now)
    for asset in assets:
        _upsert_edge(cn, SYSTEM_ENTIRETY_ID, SYSTEM_ENTIRETY_TYPE,
                     asset.entity_id, ASSET_ENTITY_TYPE, SYSTEM_REALIZES_REL,
                     max(0.01, asset.score * realization), now)

    payload = json.dumps(props, default=str)
    cn.execute(
        "INSERT INTO kv_store(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        ("asset_resource_mesh_last", payload),
    )
    try:
        cn.execute(
            "INSERT INTO brain_kv(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            ("entirety:physical_realization", payload, now),
        )
    except sqlite3.Error:
        pass
    return props


def tick_asset_resource_mesh(
    cn: sqlite3.Connection,
    peers: Iterable[Any] | None = None,
    *,
    local_capacity: Any | None = None,
    max_tunnels: int = 96,
) -> dict[str, Any]:
    """Materialize compute assets and tunnel them through material space."""
    _ensure_tables(cn)
    mesh_contracts = resource_share_contracts(peers, local_capacity=local_capacity)
    assets = materialize_asset_resources(cn, peers, local_capacity=local_capacity)
    processor_stats = bind_assets_to_material_processors(cn, assets)
    tunnel_stats = mesh_asset_tunnels(cn, assets, max_tunnels=max_tunnels)
    realization = _write_system_realization(
        cn,
        assets,
        asset_tunnels=int(tunnel_stats.get("asset_tunnels", 0)),
        processor_edges=int(processor_stats.get("processor_edges", 0)),
        mesh_contracts=mesh_contracts,
    )
    cn.commit()
    return {
        "peers": len({asset.host for asset in assets}),
        "assets": len(assets),
        "asset_tunnels": tunnel_stats.get("asset_tunnels", 0),
        "processor_edges": processor_stats.get("processor_edges", 0),
        "processor_count": realization.get("processor_count", 0),
        "physical_realization": realization.get("physical_realization", 0.0),
        "cores_available": realization.get("cores_available", 0.0),
        "ram_available_gb": realization.get("ram_available_gb", 0.0),
        "vram_available_gb": realization.get("vram_available_gb", 0.0),
        "free_disk_gb": realization.get("free_disk_gb", 0.0),
        "total_disk_gb": realization.get("total_disk_gb", 0.0),
        "drive_count": realization.get("drive_count", 0),
        "gpu_resuscitation": realization.get("gpu_resuscitation", {}),
        "resource_sharing": realization.get("resource_sharing", {}),
        "at": realization.get("updated_at"),
    }


__all__ = [
    "ASSET_ENTITY_TYPE",
    "COMPUTE_PEER_TYPE",
    "SYSTEM_ENTIRETY_TYPE",
    "SYSTEM_ENTIRETY_ID",
    "HAS_ASSET_REL",
    "REALIZES_PROCESSOR_REL",
    "ASSET_TUNNEL_REL",
    "SYSTEM_REALIZES_REL",
    "AssetVector",
    "materialize_asset_resources",
    "bind_assets_to_material_processors",
    "mesh_asset_tunnels",
    "resource_share_contracts",
    "tick_asset_resource_mesh",
]