"""knot_shadow — the viscous shear chain's output as token and edge proposals.

Reads the brain read-only (its own connection with ``PRAGMA query_only``) and
feeds the held ACRE tool ``tool_viscous_shear_chain``
(``qpsi/tool_candidates/viscous_shear_chain.py``):

* the von Neumann entropy differential series ``mesh_slm_meta
  ["entropy_differential_history"]`` (no closed-form fallback: that series is
  monotone and yields no peaks);
* Weyl Psi2 (``learnings:weyl_tensor``), the synaptic-wash rhythm
  (``temporal_spatiality_rhythm``);
* supernova decisions: per-NPC relational records posted by Perceptopoly to
  ``POST /relational`` with ``schema == "perceptopoly.supernova/1"``
  (``{agent_id, schema, magnitude, direction, at}``, latest per NPC);
* the fleet's expansion phase, synaptic heartbeats, the actionable potential;
* the occupied torus cells and the existing edges between on-field tokens.

``run()`` calls the tool only when ``tool_gate.realised_callable`` returns it.
While the tool is held, ``run()`` executes nothing; ``run(dry_run=True)`` calls
the held candidate explicitly, publishes under ``...:dry_run`` and never counts
toward proposals.  A live run appends one history entry per distinct input
state; tokens and edges that recur in at least two distinct states are
published as proposals at ``entirety:knot_shadow:proposal``.

Proposals are never applied.  Nothing here writes the mesh tables, the residual,
or any key outside ``entirety:knot_shadow:``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
from datetime import datetime
from typing import Any, Iterable

KV_PREFIX = "entirety:knot_shadow:"
TOOL = "viscous_shear_chain"
SCHEMA = "perceptopoly.supernova/1"
HISTORY_CAP = 64
MIN_SESSIONS = 2
MAX_NPCS = 256
EDGE_ID_CHUNK = 400
EDGE_LIMIT = 2000
HEARTBEATS = {"synapse_builder_last": 600, "synapse_lookahead_7d_last": 900,
              "synapse_convergence_last": 1800, "synapse_ueqgm_last": 420}
REALISATION = "none: through the six gates, two signers and the grant"


def _kv_set(key: str, value: Any) -> None:
    if not key.startswith(KV_PREFIX):
        raise PermissionError(f"knot_shadow writes only {KV_PREFIX}* keys, not {key!r}")
    from .. import brain_kv
    brain_kv.kv_set_json(key, value)


def _kv_get(key: str, default: Any = None) -> Any:
    from .. import brain_kv
    return brain_kv.kv_get_json(key, default)


def _ro_conn() -> sqlite3.Connection:
    from ..local_store import db_path
    cn = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True, timeout=10)
    cn.execute("PRAGMA query_only=ON")
    return cn


def _json(cn, sql: str, args=(), default=None):
    try:
        row = cn.execute(sql, args).fetchone()
    except sqlite3.Error:
        return default
    if not row or row[0] in (None, ""):
        return default
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return row[0]


def _kv(cn, key, default=None):
    return _json(cn, "SELECT value FROM brain_kv WHERE key=?", (key,), default)


def _epoch(raw) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(raw).split("|", 1)[0].replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _npcs(cn) -> list[dict]:
    index = _kv(cn, "relational:index", {}) or {}
    out = []
    for key, line in sorted(index.items()) if isinstance(index, dict) else []:
        if not (isinstance(line, dict) and line.get("schema") == SCHEMA and key.startswith("perceptopoly:")):
            continue
        rec = _kv(cn, f"relational:{key}", None)
        if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
            continue
        body = rec.get("payload") if isinstance(rec.get("payload"), dict) else rec
        times = [t for t in (_epoch(body.get("at")), _epoch(rec.get("received_at"))) if t is not None]
        try:
            mag, direction = float(body.get("magnitude")), float(body.get("direction", 0.0))
        except (TypeError, ValueError):
            continue
        if not times or not (math.isfinite(mag) and math.isfinite(direction)):
            continue
        out.append({"id": str(rec.get("agent_id")), "magnitude": mag, "direction": direction, "t": min(times)})
        if len(out) >= MAX_NPCS:
            break
    return out


def gather(cn: sqlite3.Connection, now: float) -> dict:
    series = _json(cn, "SELECT value FROM mesh_slm_meta WHERE key='entropy_differential_history'", (), []) or []
    weyl = _kv(cn, "learnings:weyl_tensor", None)
    if isinstance(weyl, dict):
        weyl = weyl.get("psi5") or weyl.get("weyl")
    psi2 = float(weyl[2]) if isinstance(weyl, list) and len(weyl) > 2 else 0.5
    rhythm = _kv(cn, "temporal_spatiality_rhythm", {}) or {}
    phases = (_kv(cn, "entirety:expansion_phase", ""), _kv(cn, "mesh:entirety:expansion_phase", ""))
    beats = [(_epoch(_kv(cn, k)), p) for k, p in HEARTBEATS.items()]
    beats = [(t, p) for t, p in beats if t is not None]
    potential = (_kv(cn, "recursive_strengthening", {}) or {}).get("actionable_potential", 0.0)
    return {
        "entropy_series": [float(x) for x in series if isinstance(x, (int, float))],
        "psi2": psi2,
        "gamma": float(rhythm.get("gradient", 0.0) or 0.0) if isinstance(rhythm, dict) else 0.0,
        "coherence": float(rhythm.get("coherence", 1.0) or 1.0) if isinstance(rhythm, dict) else 1.0,
        "now": now,
        "npcs": _npcs(cn),
        "broaden": 1.0 if "broaden" in phases else 0.0,
        "liveness": (sum(1 for t, p in beats if now - t <= 2 * p) / len(beats)) if beats else 0.0,
        "actionable_potential": float(potential or 0.0),
    }


def _chunks(seq: list, n: int) -> Iterable[list]:
    for k in range(0, len(seq), n):
        yield seq[k:k + n]


def _occupied_and_edges(cn, cells: list[dict]) -> tuple[dict, list]:
    on = {(c["i"], c["j"]) for c in cells}
    if not on:
        return {}, []
    try:
        rows = cn.execute("SELECT token_id, token, i, j FROM mesh_slm_vocab").fetchall()
    except sqlite3.Error:
        return {}, []
    occupied, ids = {}, {}
    for tid, tok, i, j in rows:
        if (i, j) in on:
            occupied[f"{i},{j}"] = tok
            ids[tid] = tok
    edges = []
    keys = sorted(ids)
    for a in _chunks(keys, EDGE_ID_CHUNK):
        for b in _chunks(keys, EDGE_ID_CHUNK):
            q = (f"SELECT src, dst, weight FROM mesh_slm_quipu WHERE src IN ({','.join('?' * len(a))}) "
                 f"AND dst IN ({','.join('?' * len(b))}) LIMIT ?")
            try:
                for s, d, w in cn.execute(q, (*a, *b, EDGE_LIMIT - len(edges))):
                    edges.append([ids[s], ids[d], float(w)])
            except sqlite3.Error:
                return occupied, edges
            if len(edges) >= EDGE_LIMIT:
                return occupied, edges
    return occupied, edges


def _session(data: dict) -> str:
    stable = {k: v for k, v in data.items() if k != "now"}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()[:16]


def recurring(history: list[dict], min_sessions: int = MIN_SESSIONS) -> dict:
    """Tokens and edges that recur in at least ``min_sessions`` distinct sessions."""
    toks: dict[tuple, dict] = {}
    edges: dict[tuple, dict] = {}
    for h in history:
        s = h.get("session")
        for t in h.get("tokens") or []:
            r = toks.setdefault((t["token"], t["lobe"]), {"token": t["token"], "lobe": t["lobe"],
                                                          "sessions": set(), "lift": 0.0})
            r["sessions"].add(s)
            r["lift"] = max(r["lift"], float(t.get("lift") or 0.0))
        for e in h.get("edges") or []:
            r = edges.setdefault((e["src"], e["dst"], e["lobe"]), {"src": e["src"], "dst": e["dst"],
                                                                   "lobe": e["lobe"], "sessions": set(), "lift": 0.0})
            r["sessions"].add(s)
            r["lift"] = max(r["lift"], float(e.get("lift") or 0.0))

    def done(d):
        out = [{**r, "sessions": sorted(r["sessions"]), "status": "proposed", "realisation": REALISATION}
               for r in d.values() if len(r["sessions"]) >= min_sessions]
        return sorted(out, key=lambda r: (-len(r["sessions"]), -r["lift"]))
    return {"tokens": done(toks), "edges": done(edges)}


def run(*, now: float | None = None, dry_run: bool = False) -> dict:
    from . import tool_gate
    now = time.time() if now is None else float(now)
    fn = tool_gate.candidate_callable(TOOL) if dry_run else tool_gate.realised_callable(TOOL)
    if fn is None:
        return {"status": "held" if not dry_run else "no_candidate", "executed": False,
                "note": "the tool is not realised; nothing ran" if not dry_run else "no candidate to dry-run"}
    cn = _ro_conn()
    try:
        data = gather(cn, now)
        field = fn({**data, "stage": "field"})["result"]["field"]
        occupied, edges = _occupied_and_edges(cn, field["cells"])
    finally:
        cn.close()
    data = {**data, "occupied": occupied, "edges": edges, "stage": "all"}
    out = fn(data)
    res = out["result"]
    token = tool_gate.new_shadow_token()
    entry = {"session": _session(data), "at": now, "shadow_token": token,
             "gate": res["gate"]["value"], "tokens": res["interactions"]["tokens"],
             "edges": res["interactions"]["edges"]}
    record = {"schema": "knot_shadow/1", "at": now, "shadow_token": token, "tool": TOOL,
              "mode": "dry_run" if dry_run else "live", "session": entry["session"],
              "gate": res["gate"], "inputs_real": res["inputs_real"], "confidence": out["confidence"],
              "field": {"peaks": len(res["field"]["peaks"]), "cells": len(res["field"]["cells"]),
                        "reason": res["field"]["reason"]},
              "worklink": res["worklink"], "pompeii": res["pompeii"], "bond": res["bond"],
              "realisation": REALISATION}
    if dry_run:
        record["candidate_interactions"] = res["interactions"]
        _kv_set(KV_PREFIX + "dry_run", record)
        return {"status": "dry_run", "executed": True, **record}
    history = _kv_get(KV_PREFIX + "history", []) or []
    if not any(h.get("session") == entry["session"] for h in history):
        history = (history + [entry])[-HISTORY_CAP:]
        _kv_set(KV_PREFIX + "history", history)
    props = recurring(history)
    record.update({"token_proposals": props["tokens"], "edge_proposals": props["edges"],
                   "sessions": len(history)})
    _kv_set(KV_PREFIX + "proposal", record)
    return {"status": "live", "executed": True, **record}


def latest() -> dict:
    return _kv_get(KV_PREFIX + "proposal", {}) or {}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="knot_shadow", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true")
    sub.add_parser("show")
    a = ap.parse_args(argv)
    out = run(dry_run=a.dry_run) if a.cmd == "run" else latest()
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
