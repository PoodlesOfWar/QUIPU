"""deepmap — the fleet's link map: who is connected to whom, as the devices report it.

A device (a phone running the Termux deepmap node, a hub, a peer) posts one
``deepmap.link/1`` record per observation: the node it is, and the links it can
see from where it stands (its Wi-Fi link, whether a BurgerPhone peer answers on
a port).  QUIPU keeps the latest record per node and assembles every reporter's
view into one graph: nodes, edges, and the networks they share.

Privacy is part of the contract, not an option:

* A record is rebuilt from a whitelist of fields; anything else is dropped.
* A record containing a MAC address anywhere is refused.  Devices send salted
  hashes of access-point and network identifiers (``ap``, ``network_id``), so the
  map can say "the same access point" without saying *which* one, and a stored
  record cannot be used to look a router up in a Wi-Fi location database.
* Latency, signal strength and reachability are the only measurements kept.

Storage is the existing relational-record store (``relational:<source>:<node>``
and ``relational:index``): nothing here reaches the mesh corpus, the stats or the
world model, and nothing is input to a gate.  Stdlib only.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

SCHEMA_LINK = "deepmap.link/1"
MAX_EDGES = 64
MAX_LABEL = 64
DEFAULT_MAX_AGE_S = 3600.0

ROLES = ("phone", "ap", "peer", "hub", "other")
EDGE_KINDS = ("wifi", "tcp", "bluetooth", "usb", "tunnel")

_MAC = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{2}([:-])[0-9a-f]{2}(?:\1[0-9a-f]{2}){4}(?![0-9a-f])")
_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

# measurement -> (minimum, maximum); a value outside is refused, not clamped:
# a device reporting an impossible number is broken, and the map should say so.
_NUM = {
    "rssi_dbm": (-130.0, 0.0),
    "latency_ms": (0.0, 60000.0),
    "link_mbps": (0.0, 100000.0),
    "freq_mhz": (0.0, 100000.0),
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(rec: Mapping[str, Any], key: str) -> Optional[float]:
    if key not in rec or rec[key] is None:
        return None
    v = rec[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{key} must be a number")
    lo, hi = _NUM[key]
    if not (lo <= float(v) <= hi):
        raise ValueError(f"{key} {v} is outside [{lo:g}, {hi:g}]")
    return float(v)


def _ident(value: Any, what: str) -> str:
    s = str(value or "").strip()
    if not _ID.match(s):
        raise ValueError(f"{what} must be 1-64 characters of letters, digits and . _ : -")
    return s


def validate_link(rec: Any) -> tuple[Optional[dict[str, Any]], str]:
    """Rebuild a deepmap.link/1 record from its whitelisted fields.

    Returns ``(clean, "")`` or ``(None, reason)``.  Nothing outside the whitelist
    survives, and a MAC address anywhere in the record refuses it.
    """
    if not isinstance(rec, Mapping):
        return None, "record must be an object"
    if rec.get("schema") != SCHEMA_LINK:
        return None, f"schema must be {SCHEMA_LINK}"
    try:
        flat = json.dumps(rec, default=str)
    except (TypeError, ValueError):
        return None, "record is not JSON-serialisable"
    if _MAC.search(flat):
        return None, ("record contains a MAC address; send a salted hash instead "
                      "(deepmap never stores device or access-point addresses)")
    try:
        node_id = _ident(rec.get("agent_id"), "agent_id")
        node_in = rec.get("node") if isinstance(rec.get("node"), Mapping) else {}
        role = str(node_in.get("role") or "other")
        if role not in ROLES:
            raise ValueError(f"node.role must be one of {', '.join(ROLES)}")
        node: dict[str, Any] = {"role": role, "label": str(node_in.get("label") or "")[:MAX_LABEL]}
        if node_in.get("network_id"):
            node["network_id"] = _ident(node_in["network_id"], "node.network_id")

        wifi: dict[str, Any] = {}
        w_in = rec.get("wifi") if isinstance(rec.get("wifi"), Mapping) else {}
        if w_in.get("ap"):
            wifi["ap"] = _ident(w_in["ap"], "wifi.ap")
        for key in ("rssi_dbm", "freq_mhz", "link_mbps"):
            v = _num(w_in, key)
            if v is not None:
                wifi[key] = v
        if w_in.get("state"):
            wifi["state"] = str(w_in["state"])[:32]

        edges_in = rec.get("edges") or []
        if not isinstance(edges_in, list):
            raise ValueError("edges must be a list")
        if len(edges_in) > MAX_EDGES:
            raise ValueError(f"at most {MAX_EDGES} edges per record")
        edges: list[dict[str, Any]] = []
        for e in edges_in:
            if not isinstance(e, Mapping):
                raise ValueError("each edge must be an object")
            kind = str(e.get("kind") or "")
            if kind not in EDGE_KINDS:
                raise ValueError(f"edge.kind must be one of {', '.join(EDGE_KINDS)}")
            edge: dict[str, Any] = {"to": _ident(e.get("to"), "edge.to"), "kind": kind}
            if "ok" in e:
                edge["ok"] = bool(e["ok"])
            for key in ("rssi_dbm", "latency_ms", "link_mbps"):
                v = _num(e, key)
                if v is not None:
                    edge[key] = v
            if e.get("port") is not None:
                port = e["port"]
                if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                    raise ValueError("edge.port must be 1-65535")
                edge["port"] = port
            edges.append(edge)
    except ValueError as exc:
        return None, str(exc)

    observed = rec.get("observed_at")
    if isinstance(observed, str) and observed:
        try:
            datetime.fromisoformat(observed.replace("Z", "+00:00"))
        except ValueError:
            return None, "observed_at must be an ISO-8601 timestamp"
    else:
        observed = _iso_now()
    clean: dict[str, Any] = {"agent_id": node_id, "schema": SCHEMA_LINK, "node": node,
                             "edges": edges, "observed_at": observed}
    if wifi:
        clean["wifi"] = wifi
    return clean, ""


def _age_s(observed_at: Any, now: float) -> Optional[float]:
    if not isinstance(observed_at, str):
        return None
    try:
        t = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return max(0.0, now - t.timestamp())


def build_graph(records: list[Mapping[str, Any]], *, now: float | None = None,
                max_age_s: float = DEFAULT_MAX_AGE_S) -> dict[str, Any]:
    """Assemble every reporter's view into one graph.

    A reporter is a node; every edge target is a node too (an access point or peer
    that never reported still appears, with the role its edge implies).  An edge is
    ``stale`` when its reporter has not been heard from within ``max_age_s``, so a
    phone that left the network drops out of the picture instead of lingering as
    "connected".
    """
    t = time.time() if now is None else now
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for rec in records:
        nid = rec.get("agent_id")
        if not nid:
            continue
        node_in = rec.get("node") if isinstance(rec.get("node"), Mapping) else {}
        age = _age_s(rec.get("observed_at"), t)
        stale = age is None or age > max_age_s
        n = nodes.setdefault(nid, {"id": nid, "role": "other", "label": "", "reporter": False})
        n.update(role=node_in.get("role", n["role"]), label=node_in.get("label", n["label"]),
                 reporter=True, last_seen=rec.get("observed_at"), stale=stale)
        if node_in.get("network_id"):
            n["network_id"] = node_in["network_id"]
        wifi = rec.get("wifi") if isinstance(rec.get("wifi"), Mapping) else {}
        e_list = list(rec.get("edges") or [])
        if wifi.get("ap") and not any(e.get("kind") == "wifi" and e.get("to") == wifi["ap"] for e in e_list):
            e_list.append({"to": wifi["ap"], "kind": "wifi", "ok": True,
                           **{k: wifi[k] for k in ("rssi_dbm", "link_mbps") if k in wifi}})
        for e in e_list:
            to = e.get("to")
            if not to:
                continue
            nodes.setdefault(to, {"id": to, "role": "ap" if e.get("kind") == "wifi" else "peer",
                                  "label": "", "reporter": False})
            edges.append({"from": nid, "observed_at": rec.get("observed_at"), "stale": stale, **e})
    networks: dict[str, int] = {}
    for n in nodes.values():
        if n.get("network_id") and not n.get("stale"):
            networks[n["network_id"]] = networks.get(n["network_id"], 0) + 1
    return {"ok": True, "generated_at": _iso_now(), "max_age_s": max_age_s,
            "reporters": sum(1 for n in nodes.values() if n["reporter"]),
            "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
            "edges": edges, "networks": networks}
