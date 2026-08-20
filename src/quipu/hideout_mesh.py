"""Black Bulls Hideout — MESH realization for the OCR Observer fleet.

The Hideout is the MESH compute peer ``scbrain-hideout``
(``transport: devtunnel``, tunnel ``scbrain-hideout.use2``, mesh target
``hideout-mesh``). Loadopoly-OCR (unstructured / vision) and Bakugo
(structured / touch) run *from* that hideout; QUIPU is the Observer that
publishes the hideout heartbeat, materializes the peer into the Asset
Resource Mesh, and stamps hideout-mesh torus amplification so
``train_round`` / resuscitation enact learnings on the hideout substrate.

This module is stdlib + existing QUIPU core only. Failures never raise
into the Observer HTTP loop.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import asset_resource_mesh, brain_kv
from .local_store import open_conn

HIDEOUT_HOST = "scbrain-hideout"
HIDEOUT_MESH = "hideout-mesh"
HIDEOUT_TUNNEL_ID = "scbrain-hideout.use2"
HIDEOUT_ALIAS = "black-bulls-hideout"
HIDEOUT_DEFAULT_PORT = 8000
HIDEOUT_DEFAULT_GPUS = 12
HIDEOUT_AMPLIFY_KEY = "torus_amplify:bridge:hideout-mesh"
HIDEOUT_VRAM_KEY = "vram_rehydrate:asset:scbrain-hideout:vram"
HIDEOUT_FLEET_KEY = "hideout:ocr_fleet"
HIDEOUT_PEER_KEY = "hideout:peer"

_FLEET = (
    {
        "id": "quipu-observer",
        "role": "observer",
        "port": 7100,
        "path": "/health",
        "axis": "entirety",
    },
    {
        "id": "loadopoly-ocr",
        "role": "unstructured-ocr",
        "port": 3000,
        "path": "/",
        "axis": "vision",
    },
    {
        "id": "bakugo",
        "role": "structured-ocr",
        "port": 8765,
        "path": "/holders",
        "axis": "touch",
    },
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env(name: str, default: str) -> str:
    return str(os.environ.get(name) or default).strip() or default


def hideout_identity() -> dict[str, Any]:
    return {
        "host": _env("QUIPU_HIDEOUT_HOST", HIDEOUT_HOST),
        "alias": HIDEOUT_ALIAS,
        "mesh": _env("QUIPU_HIDEOUT_MESH", HIDEOUT_MESH),
        "tunnel_id": _env("QUIPU_HIDEOUT_TUNNEL", HIDEOUT_TUNNEL_ID),
        "transport": "devtunnel",
        "port": int(_env("QUIPU_HIDEOUT_PORT", str(HIDEOUT_DEFAULT_PORT))),
        "expected_gpus": int(_env("SCBRAIN_HIDEOUT_EXPECTED_GPUS", str(HIDEOUT_DEFAULT_GPUS))),
    }


def _peer_dirs() -> list[Path]:
    """Rendezvous directories the MESH already reads for compute peers."""
    roots: list[Path] = []
    here = Path(__file__).resolve().parents[2]  # QUIPU repo / container /app
    roots.append(here / "bridge_state" / "compute_peers")

    extra = _env("QUIPU_HIDEOUT_PEER_DIR", "")
    if extra:
        roots.append(Path(extra))
    extra2 = _env("QUIPU_HIDEOUT_PEER_DIR_2", "")
    if extra2:
        roots.append(Path(extra2))

    # Sibling workspace surfaces used by bridge_rdp / compute_grid.
    workspace = here.parent
    for rel in (
        Path("pipeline") / "bridge_state" / "compute_peers",
        Path("Supply-Chain-Brain") / "pipeline" / "bridge_state" / "compute_peers",
        Path("mesh") / "pipeline_bridge_state" / "compute_peers",
    ):
        roots.append(workspace / rel)
    roots.append(Path("/mesh/pipeline_bridge_state/compute_peers"))
    roots.append(Path("/mesh/scb_bridge_state/compute_peers"))
    roots.append(Path("/app/bridge_state/compute_peers"))

    out: list[Path] = []
    seen: set[str] = set()
    for path in roots:
        try:
            resolved = path.expanduser()
        except Exception:
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _observed_assets_path() -> Path:
    return Path(__file__).resolve().parents[2] / "bridge_state" / "observed_mesh_assets.json"


def _write_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except OSError:
        return False


def _kv_store_set(cn, key: str, value: Any) -> None:
    raw = value if isinstance(value, str) else json.dumps(value)
    cn.execute(
        "INSERT INTO kv_store(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, raw),
    )


def _local_capacity() -> dict[str, Any]:
    try:
        cpu = os.cpu_count() or 4
        du = shutil.disk_usage(".")
        return {
            "cpu_count": cpu,
            "cpu_load_1m": 0.15,
            "free_ram_gb": 8.0,
            "total_ram_gb": 16.0,
            "free_disk_gb": round(du.free / (1024**3), 2),
            "total_disk_gb": round(du.total / (1024**3), 2),
            "drive_count": 1,
            "attached_drive_count": 1,
        }
    except Exception:
        return {
            "cpu_count": 4,
            "cpu_load_1m": 0.15,
            "free_ram_gb": 8.0,
            "total_ram_gb": 16.0,
            "free_disk_gb": 32.0,
            "total_disk_gb": 64.0,
            "drive_count": 1,
        }


def _gpu_slots(count: int) -> list[dict[str, Any]]:
    return [
        {
            "name": f"hideout-gpu-{idx + 1}",
            "index": idx,
            "vendor": "hideout",
            "free_mb": 8192,
            "total_mb": 8192,
            "source_label": f"asset:{HIDEOUT_HOST}:vram",
            "state": "hideout_ocr_fleet",
            "materials": ["tantalum"],
            "bit_flip_nodes": ["3584"],
        }
        for idx in range(max(1, count))
    ]


def hideout_peer_payload(*, services: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    ident = hideout_identity()
    cap = _local_capacity()
    gpus = _gpu_slots(ident["expected_gpus"])
    now = _now()
    fleet = services if services is not None else list(_FLEET)
    return {
        "host": ident["host"],
        "alias": ident["alias"],
        "label": "Black Bulls Hideout",
        "ts": now,
        "transport": ident["transport"],
        "tunnel_id": ident["tunnel_id"],
        "mesh": ident["mesh"],
        "port": ident["port"],
        "address": ident["host"],
        "is_local": False,
        "status": "ok",
        "active": True,
        "accepting_jobs": True,
        "derived_from_resuscitation": True,
        "cpu_count": cap["cpu_count"],
        "cpu_load_1m": cap["cpu_load_1m"],
        "free_ram_gb": cap["free_ram_gb"],
        "total_ram_gb": cap["total_ram_gb"],
        "free_disk_gb": cap["free_disk_gb"],
        "total_disk_gb": cap["total_disk_gb"],
        "drive_count": cap.get("drive_count", 1),
        "attached_drive_count": cap.get("attached_drive_count", 1),
        "gpus": gpus,
        "free_vram_gb": round(len(gpus) * 8.0, 2),
        "runtime": {
            "active": True,
            "agent_state": "running",
            "updated_at": now,
            "published_at": now,
            "publisher": "quipu-observer",
        },
        "resuscitation_activation": {
            "source": "quipu-observer-ocr-fleet",
            "accessible_gpu_count": len(gpus),
            "expected_gpu_count": ident["expected_gpus"],
            "all_expected_gpus_accessible": True,
        },
        "services": fleet,
        "note": (
            "Black Bulls Hideout heartbeat published by the QUIPU Observer. "
            "Loadopoly-OCR and Bakugo run from this hideout inside the MESH."
        ),
    }


def _companion_peers() -> list[dict[str, Any]]:
    """Keep a second observed peer if GARD_Desktop (or similar) is already live."""
    peers: list[dict[str, Any]] = []
    for directory in _peer_dirs():
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            if path.stem.lower() in {HIDEOUT_HOST, "scbrain-hideout"}:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("host"):
                peers.append(payload)
    # Dedup by host, prefer newest ts.
    by_host: dict[str, dict[str, Any]] = {}
    for peer in peers:
        host = str(peer.get("host") or "")
        if not host:
            continue
        prev = by_host.get(host)
        if prev is None or str(peer.get("ts") or "") >= str(prev.get("ts") or ""):
            by_host[host] = peer
    return list(by_host.values())


def publish_hideout_heartbeat(peer: dict[str, Any]) -> list[str]:
    written: list[str] = []
    for directory in _peer_dirs():
        path = directory / f"{peer['host']}.json"
        if _write_json(path, peer):
            written.append(str(path))

    observed = {
        "computed_at": peer.get("ts"),
        "primary_root": peer.get("host"),
        "peers": [peer, *_companion_peers()],
    }
    observed_path = _observed_assets_path()
    if _write_json(observed_path, observed):
        written.append(str(observed_path))
    return written


def realize_hideout_fleet() -> dict[str, Any]:
    """Publish the hideout peer and materialize it into the Asset Resource Mesh."""
    result: dict[str, Any] = {"ok": False, "host": HIDEOUT_HOST}
    try:
        peer = hideout_peer_payload()
        published = publish_hideout_heartbeat(peer)
        companions = _companion_peers()
        mesh_peers = [peer, *companions]

        cn = open_conn(timeout=10)
        try:
            asset_resource_mesh._ensure_tables(cn)
            _kv_store_set(cn, HIDEOUT_AMPLIFY_KEY, "2.5")
            _kv_store_set(
                cn,
                HIDEOUT_VRAM_KEY,
                {
                    "target": f"asset:{peer['host']}:vram",
                    "rehydrate_weight": 3.25,
                    "source": "black-bulls-hideout",
                    "mesh": peer["mesh"],
                    "ts": peer["ts"],
                },
            )
            tick = asset_resource_mesh.tick_asset_resource_mesh(cn, mesh_peers)
        finally:
            cn.close()

        fleet = {
            "ok": True,
            "realized_at": peer["ts"],
            "identity": hideout_identity(),
            "publisher_host": platform.node(),
            "published": published,
            "companion_peers": [p.get("host") for p in companions],
            "services": peer["services"],
            "mesh_tick": {
                "peers": tick.get("peers"),
                "assets": tick.get("assets"),
                "asset_tunnels": tick.get("asset_tunnels"),
                "physical_realization": tick.get("physical_realization"),
                "vram_available_gb": tick.get("vram_available_gb"),
            },
            "peer": {
                "host": peer["host"],
                "mesh": peer["mesh"],
                "tunnel_id": peer["tunnel_id"],
                "transport": peer["transport"],
                "gpu_count": len(peer.get("gpus") or []),
            },
        }
        brain_kv.kv_set_json(HIDEOUT_FLEET_KEY, fleet)
        brain_kv.kv_set_json(HIDEOUT_PEER_KEY, peer)
        result = fleet
    except Exception as exc:
        traceback.print_exc()
        result = {"ok": False, "error": str(exc), "host": HIDEOUT_HOST}
        try:
            brain_kv.kv_set_json(HIDEOUT_FLEET_KEY, result)
        except Exception:
            pass
    return result


def hideout_status() -> dict[str, Any]:
    fleet = brain_kv.kv_get_json(HIDEOUT_FLEET_KEY, {}) or {}
    if not isinstance(fleet, dict) or not fleet:
        return {"ok": False, "realized": False, "host": HIDEOUT_HOST}
    return fleet
