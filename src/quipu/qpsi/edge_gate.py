"""edge_gate — displacement-gated replacement for system_entirety._mesh_upsert_edge.

Same table, same primary key, same call shape plus two optional arguments.
Differences from the current upsert:

    samples          increments ONLY on displacement: |Δweight| ≥ eta, or a
                     parity flip since the last displacement, or first insert.
    weight           running mean over DISPLACED samples only (the at-rest
                     value no longer dominates).
    last_seen        rewritten on every call — the heartbeat.  Unchanged.
    flip_count_at_displacement   additive column: flip_count when the edge
                     last displaced (regime stamp, §3 / §8).
    cost_class       additive column: action cost class (§3 / §7.3).

Columns are added with ALTER TABLE if missing; existing rows read NULL,
which the code treats as 0 / 'default'.  Nothing else in the schema moves.

Validation: replaying 76 identical weights yields samples == 1 (today: 76).
"""
from __future__ import annotations

import sqlite3

DEFAULT_ETA = 1e-6          # float-resolution floor; replace with measured η


def _clip01(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def ensure_columns(cn: sqlite3.Connection) -> None:
    cols = {r[1] for r in cn.execute("PRAGMA table_info(corpus_edge)")}
    if "flip_count_at_displacement" not in cols:
        cn.execute("ALTER TABLE corpus_edge ADD COLUMN flip_count_at_displacement INTEGER")
    if "cost_class" not in cols:
        cn.execute("ALTER TABLE corpus_edge ADD COLUMN cost_class TEXT")


def gated_upsert_edge(cn: sqlite3.Connection, src_id: str, src_type: str, dst_id: str,
                      dst_type: str, rel: str, weight: float, now: str, *,
                      eta: float = DEFAULT_ETA, flip_count: int = 0,
                      cost_class: str = "default") -> dict:
    """Returns {'displaced': bool, 'samples': int, 'weight': float}."""
    weight = _clip01(weight)
    row = cn.execute(
        "SELECT weight, samples, flip_count_at_displacement FROM corpus_edge "
        "WHERE src_id=? AND src_type=? AND dst_id=? AND dst_type=? AND rel=?",
        (src_id, src_type, dst_id, dst_type, rel),
    ).fetchone()

    if row is None:
        cn.execute(
            "INSERT INTO corpus_edge(src_id, src_type, dst_id, dst_type, rel, weight, "
            "last_seen, samples, flip_count_at_displacement, cost_class) "
            "VALUES(?,?,?,?,?,?,?,1,?,?)",
            (src_id, src_type, dst_id, dst_type, rel, weight, now, int(flip_count), cost_class),
        )
        return {"displaced": True, "samples": 1, "weight": weight}

    prev_w = float(row[0] or 0.0)
    prev_n = int(row[1] or 1)
    prev_flip = int(row[2] or 0)
    displaced = abs(weight - prev_w) >= eta or int(flip_count) != prev_flip

    if displaced:
        new_n = prev_n + 1
        new_w = (prev_w * prev_n + weight) / new_n
        cn.execute(
            "UPDATE corpus_edge SET weight=?, samples=?, last_seen=?, "
            "flip_count_at_displacement=?, cost_class=? "
            "WHERE src_id=? AND src_type=? AND dst_id=? AND dst_type=? AND rel=?",
            (new_w, new_n, now, int(flip_count), cost_class,
             src_id, src_type, dst_id, dst_type, rel),
        )
        return {"displaced": True, "samples": new_n, "weight": new_w}

    # Heartbeat only.
    cn.execute(
        "UPDATE corpus_edge SET last_seen=? "
        "WHERE src_id=? AND src_type=? AND dst_id=? AND dst_type=? AND rel=?",
        (now, src_id, src_type, dst_id, dst_type, rel),
    )
    return {"displaced": False, "samples": prev_n, "weight": prev_w}


__all__ = ["DEFAULT_ETA", "ensure_columns", "gated_upsert_edge"]
