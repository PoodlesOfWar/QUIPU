"""brain_migrate — make the two brains one (v0.48.0).

    python -m src.quipu.brain_migrate --base HOST.sqlite --observer OBSERVER.sqlite --out MERGED.sqlite

The host brain (the Entirety: ingest, self-organising loop, gates, emergence)
is the base.  The observer container's brain holds every fleet observation.
Rather than splicing tables, the observer's own training record is *replayed*
through the normal learning path:

1. ``mesh_corpus_feed`` rows from the observer (its full record: every chunk the
   fleet sent) are appended to the merged brain's feed in their original order
   and source, and ``mesh_slm.train_round`` folds exactly those rows into the
   mesh — the same code the observer ran, now on the collective mesh.
2. ``brain_kv``: observer counters (``observer:<src>:stats``) are summed;
   observer frames, feedback, oscillation, ``world_model:*``, ``hideout:*`` and
   ``market:*`` keys the base lacks are copied.  The base's ``entirety:*`` keys
   stand (its gates, checkpoint and attestations are the authoritative ones);
   the observer's own ``entirety:*`` keys are archived under
   ``migration:observer_entirety`` and are inert.
3. ``corpus_entity`` / ``corpus_edge``: union; samples add where both have a row.
4. The observer's own Entirety logs (flip log, checkpoint log) are not merged —
   they are the history of a second, disconnected Entirety — and are counted in
   ``migration:v0.48.0``.

Nothing is read from or written to the inputs: the base is copied with the
SQLite backup API and every change goes to ``--out``.  Idempotent: a second run
over the same output does nothing (``migration:v0.48.0`` records the replay).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

MARK_KEY = "migration:v0.48.0"
COPY_PREFIXES = ("observer:", "world_model:", "hideout:", "market:")


def _kv(cn, key):
    row = cn.execute("SELECT value FROM brain_kv WHERE key = ?", (key,)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return row[0]


def _kv_set(cn, key, value):
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def _tables(cn):
    return {r[0] for r in cn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def migrate(base: Path, observer: Path, out: Path, *, train: bool = True) -> dict:
    out = Path(out)
    if not out.exists():
        src = sqlite3.connect(f"file:{base}?mode=ro", uri=True)
        dst = sqlite3.connect(str(out))
        src.backup(dst)
        dst.close()
        src.close()
    ob = sqlite3.connect(f"file:{observer}?mode=ro", uri=True)
    cn = sqlite3.connect(str(out))
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    done = _kv(cn, MARK_KEY)
    if isinstance(done, dict) and done.get("replayed") is not None:
        return {"already": True, **done}

    report: dict = {"base": str(base), "observer": str(observer), "at": time.time()}

    # 2. brain_kv
    ob_kv = {k: v for k, v in ob.execute("SELECT key, value FROM brain_kv")}
    copied, summed = [], []
    for key, raw in ob_kv.items():
        try:
            val = json.loads(raw)
        except (TypeError, ValueError):
            val = raw
        if key.startswith("entirety:"):
            continue
        mine = _kv(cn, key)
        if key.startswith("observer:") and key.endswith(":stats") and isinstance(mine, dict) and isinstance(val, dict):
            merged = dict(mine)
            for f in ("observations", "tokens"):
                merged[f] = int(mine.get(f, 0) or 0) + int(val.get(f, 0) or 0)
            merged["last_seen"] = max(str(mine.get("last_seen") or ""), str(val.get("last_seen") or ""))
            _kv_set(cn, key, merged)
            summed.append(key)
        elif mine is None and key.startswith(COPY_PREFIXES):
            _kv_set(cn, key, val)
            copied.append(key)
    _kv_set(cn, "migration:observer_entirety",
            {k: json.loads(v) if v and v[:1] in "[{\"0123456789-tfn" else v
             for k, v in ob_kv.items() if k.startswith("entirety:")})
    report["kv_copied"], report["kv_summed"] = copied, summed

    # 3. corpus graph
    t = _tables(cn)
    if "corpus_entity" in _tables(ob) and "corpus_entity" in t:
        n = 0
        for row in ob.execute("SELECT entity_id, entity_type, label, props_json, first_seen, last_seen, samples "
                              "FROM corpus_entity"):
            cur = cn.execute("UPDATE corpus_entity SET samples = samples + ?, last_seen = MAX(last_seen, ?) "
                             "WHERE entity_id = ? AND entity_type = ?", (row[6] or 1, row[5], row[0], row[1]))
            if not cur.rowcount:
                cn.execute("INSERT INTO corpus_entity(entity_id, entity_type, label, props_json, first_seen, "
                           "last_seen, samples) VALUES(?,?,?,?,?,?,?)", row)
            n += 1
        report["corpus_entities"] = n
    if "corpus_edge" in _tables(ob) and "corpus_edge" in t:
        n = 0
        for row in ob.execute("SELECT src_id, src_type, dst_id, dst_type, rel, weight, last_seen, samples "
                              "FROM corpus_edge"):
            cur = cn.execute("UPDATE corpus_edge SET samples = samples + ?, last_seen = MAX(last_seen, ?), "
                             "weight = (weight * samples + ? * ?) / (samples + ?) "
                             "WHERE src_id=? AND src_type=? AND dst_id=? AND dst_type=? AND rel=?",
                             (row[7] or 1, row[6], row[5] or 0.0, row[7] or 1, row[7] or 1, *row[:5]))
            if not cur.rowcount:
                cn.execute("INSERT INTO corpus_edge(src_id, src_type, dst_id, dst_type, rel, weight, last_seen, "
                           "samples) VALUES(?,?,?,?,?,?,?,?)", row)
            n += 1
        report["corpus_edges"] = n

    obt = _tables(ob)
    report["observer_entirety_logs_not_merged"] = {
        name: ob.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        for name in ("entirety_flip_log", "entirety_residual_checkpoint") if name in obt}
    feed = list(ob.execute("SELECT text, source FROM mesh_corpus_feed ORDER BY id")) \
        if "mesh_corpus_feed" in obt else []
    cn.commit()
    cn.close()
    ob.close()

    # 1. replay the observer's training record through the normal learning path
    os.environ["SCB_DB_PATH"] = str(out.resolve())
    from . import mesh_slm
    for text, source in feed:
        mesh_slm.feed_corpus(text, source=source or "")
    trained = None
    if train and feed:
        mesh_slm._LAST_TRAIN_TS = 0.0
        trained = mesh_slm.train_round(max_seconds=600.0, max_chunks=len(feed))
    report["replayed"] = len(feed)
    report["by_source"] = {}
    for _, s in feed:
        report["by_source"][s] = report["by_source"].get(s, 0) + 1
    report["train"] = {k: v for k, v in (trained or {}).items() if isinstance(v, (int, float, str, bool))}
    cn = sqlite3.connect(str(out))
    _kv_set(cn, MARK_KEY, report)
    cn.commit()
    cn.close()
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="brain_migrate", description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--observer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-train", action="store_true")
    a = ap.parse_args(argv)
    for p in (a.base, a.observer):
        if not Path(p).exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2
    print(json.dumps(migrate(Path(a.base), Path(a.observer), Path(a.out), train=not a.no_train), indent=2,
                     default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
