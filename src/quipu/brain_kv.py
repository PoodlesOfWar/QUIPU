"""brain_kv — canonical key/value persistence for the QUIPU mesh.

The parent Supply Chain Architect shipped a ``brain_kv`` module that several
extracted modules still expect:

* ``mesh_slm`` reads a ``brain_kv`` table directly (``_brain_kv_get``) for
  ``temporal_spatiality_rhythm``, ``entirety:*`` and the adaptive runtime.
* ``gard_shard_model._safe_kv_json`` imports ``.brain_kv.kv_get_json`` to read
  the holographic ``learnings:weyl_tensor`` snapshot.
* The holographic compression path (``ueqgm_engine._WEYL_TENSOR_KEY =
  "learnings:weyl_tensor"``) is *written* here so it survives across cycles.

This clean-room reimplementation stores rows in the same SQLite database as the
rest of the mesh (``local_store.db_path()`` / ``open_conn``), so every reader —
``mesh_slm``, ``gard_shard_model``, ``corpus_ingest`` — sees one shared
``brain_kv`` table with no schema coordination required.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from .local_store import open_conn

_DDL = """
CREATE TABLE IF NOT EXISTS brain_kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _ensure(cn: sqlite3.Connection) -> None:
    cn.executescript(_DDL)


def kv_set(key: str, value: str) -> None:
    """Persist a raw string *value* under *key* (upsert)."""
    cn = open_conn()
    try:
        _ensure(cn)
        cn.execute(
            "INSERT INTO brain_kv(key, value, updated_at) VALUES(?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
            (key, value),
        )
        cn.commit()
    finally:
        cn.close()


def kv_get(key: str, default: Any = None) -> Any:
    """Return the raw string stored under *key*, or *default*."""
    cn = open_conn()
    try:
        _ensure(cn)
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default
    finally:
        cn.close()


def kv_set_json(key: str, value: Any) -> None:
    """JSON-encode *value* and persist it under *key*."""
    kv_set(key, json.dumps(value, separators=(",", ":"), ensure_ascii=False))


def kv_get_json(key: str, default: Any = None) -> Any:
    """Return the JSON-decoded value under *key*, or *default* on miss/parse error."""
    raw = kv_get(key, None)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def kv_delete(key: str) -> None:
    cn = open_conn()
    try:
        _ensure(cn)
        cn.execute("DELETE FROM brain_kv WHERE key=?", (key,))
        cn.commit()
    finally:
        cn.close()


__all__ = ["kv_get", "kv_set", "kv_get_json", "kv_set_json", "kv_delete"]
