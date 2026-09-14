"""Residual hold checkpoints — the reference advances only on realisation.

The residual r_t is measured against a *reference* CAT state.  Under the
Governance Protocol the reference is the last state that was realised (all
six gates passed), not the last state observed.  A held residual therefore
accumulates: every step re-measures the displacement from the last
realisation, and a potential that is held is held until it passes, is rolled
back, or a human releases it.  Nothing here decides; it only keeps the books.

    Checkpoint         the current reference, the held residual, how long it
                       has been held, and the last decision — one per instance
    CheckpointStore    persistence: the live checkpoint under
                       brain_kv["entirety:residual_checkpoint:<instance>"]
                       and an append-only log table
                       ``entirety_residual_checkpoint``; both written through
                       the caller's sqlite connection so they commit with the
                       step that produced them.

Log kinds: "held", "realised", "rollback", "released".  ``rollback`` restores
an earlier reference (the hold is re-measured from there); ``released`` is a
human dropping a held potential by naming the current state as the reference.
Both are logged with who did it.  Stdlib only.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Mapping

from .cat_residual import SENSES, CATState, residual

KV_PREFIX: str = "entirety:residual_checkpoint:"
TABLE: str = "entirety_residual_checkpoint"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def state_to_json(s: CATState | None) -> dict | None:
    if s is None:
        return None
    return {"amp": {k: [float(v.real), float(v.imag)] for k, v in s.amp.items()}, "t": float(s.t)}


def state_from_json(d: Mapping | None) -> CATState | None:
    if not d or not isinstance(d, Mapping) or "amp" not in d:
        return None
    amp = {}
    for k, v in (d.get("amp") or {}).items():
        try:
            amp[str(k)] = complex(float(v[0]), float(v[1]))
        except (TypeError, ValueError, IndexError):
            amp[str(k)] = 0j
    return CATState(amp, float(d.get("t", 0.0)))


def residual_to_json(r: Mapping[str, complex]) -> dict:
    return {s: [float(r.get(s, 0j).real), float(r.get(s, 0j).imag)] for s in SENSES}


def residual_from_json(d: Mapping | None) -> dict[str, complex]:
    out: dict[str, complex] = {}
    for s in SENSES:
        v = (d or {}).get(s)
        try:
            out[s] = complex(float(v[0]), float(v[1])) if v is not None else 0j
        except (TypeError, ValueError, IndexError):
            out[s] = 0j
    return out


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    instance: str
    seq: int = 0
    reference: CATState | None = None          # last realised state; None = never realised
    held: dict[str, complex] = field(default_factory=dict)   # displacement from reference
    steps_held: int = 0
    held_since: float | None = None
    realised: int = 0
    last: dict = field(default_factory=dict)   # {"kind", "passed", "failed_at", "at"}

    @property
    def held_norm(self) -> float:
        return sum(abs(v) ** 2 for v in self.held.values()) ** 0.5

    def to_json(self) -> dict:
        return {"instance": self.instance, "seq": self.seq,
                "reference": state_to_json(self.reference),
                "held": residual_to_json(self.held), "steps_held": self.steps_held,
                "held_since": self.held_since, "realised": self.realised, "last": dict(self.last)}

    @classmethod
    def from_json(cls, instance: str, d: Mapping | None) -> "Checkpoint":
        if not d:
            return cls(instance)
        return cls(instance=instance, seq=int(d.get("seq", 0)),
                   reference=state_from_json(d.get("reference")),
                   held=residual_from_json(d.get("held")),
                   steps_held=int(d.get("steps_held", 0)),
                   held_since=d.get("held_since"), realised=int(d.get("realised", 0)),
                   last=dict(d.get("last") or {}))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def ensure_tables(cn: sqlite3.Connection) -> None:
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE}("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, instance TEXT NOT NULL, seq INTEGER NOT NULL, "
        "kind TEXT NOT NULL, at REAL NOT NULL, reference TEXT, held TEXT, held_norm REAL, "
        "steps_held INTEGER, failed_at TEXT, flip_count INTEGER, actor TEXT)")
    cn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_inst ON {TABLE}(instance, seq)")


class CheckpointStore:
    """All reads and writes go through the connection the caller is in."""

    def load(self, cn: sqlite3.Connection, instance: str) -> Checkpoint:
        ensure_tables(cn)
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (KV_PREFIX + instance,)).fetchone()
        try:
            return Checkpoint.from_json(instance, json.loads(row[0]) if row and row[0] else None)
        except (json.JSONDecodeError, TypeError):
            return Checkpoint(instance)

    def save(self, cn: sqlite3.Connection, cp: Checkpoint) -> None:
        ensure_tables(cn)
        cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
                   (KV_PREFIX + cp.instance, json.dumps(cp.to_json()),
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))

    def _log(self, cn, cp: Checkpoint, kind: str, at: float, *, failed_at=None,
             flip_count=None, actor=None) -> None:
        cn.execute(
            f"INSERT INTO {TABLE}(instance, seq, kind, at, reference, held, held_norm, steps_held, "
            "failed_at, flip_count, actor) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (cp.instance, cp.seq, kind, at, json.dumps(state_to_json(cp.reference)),
             json.dumps(residual_to_json(cp.held)), cp.held_norm, cp.steps_held,
             failed_at, flip_count, actor))

    # -- the two operations every step uses --------------------------------

    def measure(self, cp: Checkpoint, cur: CATState, *, eta: float = 0.0, nu: float = 0.0,
                kappa: float = 0.0) -> dict[str, complex]:
        """Residual of ``cur`` against the checkpoint's reference (last realised state)."""
        return residual(cur, cp.reference, nu=nu, kappa=kappa, eta=eta)

    def commit(self, cn: sqlite3.Connection, cp: Checkpoint, cur: CATState, r: Mapping[str, complex],
               *, passed: bool, failed_at: str | None = None, flip_count: int | None = None,
               at: float | None = None) -> Checkpoint:
        """Record the decision.  Pass → reference advances, hold cleared.  Hold → reference
        stays, the held residual is the accumulated displacement from it."""
        now = time.time() if at is None else float(at)
        cp.seq += 1
        if passed:
            cp.reference = cur
            cp.held = {}
            cp.steps_held = 0
            cp.held_since = None
            cp.realised += 1
            kind = "realised"
        else:
            cp.held = dict(r)
            cp.steps_held += 1
            cp.held_since = cp.held_since if cp.held_since is not None else now
            kind = "held"
        cp.last = {"kind": kind, "passed": bool(passed), "failed_at": failed_at, "at": now}
        self.save(cn, cp)
        self._log(cn, cp, kind, now, failed_at=failed_at, flip_count=flip_count)
        return cp

    def rows(self, cn: sqlite3.Connection, instance: str, limit: int = 64) -> list[dict]:
        """The last ``limit`` log rows, ascending, with the held residual decoded —
        the evidence window the emergence detector reads."""
        ensure_tables(cn)
        rs = cn.execute(
            f"SELECT seq, kind, at, held, flip_count FROM {TABLE} WHERE instance=? "
            "AND kind IN ('held','realised') ORDER BY seq DESC, id DESC LIMIT ?",
            (instance, int(limit))).fetchall()
        out = []
        for seq, kind, at, held, flip in reversed(rs):
            try:
                hd = residual_from_json(json.loads(held) if held else None)
            except (json.JSONDecodeError, TypeError):
                hd = residual_from_json(None)
            out.append({"seq": int(seq), "kind": kind, "at": float(at), "held": hd,
                        "flip_count": (int(flip) if flip is not None else None)})
        return out

    # -- human operations ------------------------------------------------------

    def history(self, cn: sqlite3.Connection, instance: str, limit: int = 50) -> list[dict]:
        ensure_tables(cn)
        rows = cn.execute(
            f"SELECT id, seq, kind, at, held_norm, steps_held, failed_at, flip_count, actor "
            f"FROM {TABLE} WHERE instance=? ORDER BY seq DESC, id DESC LIMIT ?", (instance, int(limit)))
        cols = ["id", "seq", "kind", "at", "held_norm", "steps_held", "failed_at", "flip_count", "actor"]
        return [dict(zip(cols, row)) for row in rows.fetchall()]

    def rollback(self, cn: sqlite3.Connection, instance: str, seq: int, *, actor: str) -> Checkpoint:
        """Restore the reference recorded at log ``seq``; the hold is re-measured from it."""
        if not actor:
            raise ValueError("rollback needs an actor (human signer)")
        ensure_tables(cn)
        row = cn.execute(f"SELECT reference FROM {TABLE} WHERE instance=? AND seq=? "
                         "ORDER BY id DESC LIMIT 1", (instance, int(seq))).fetchone()
        if row is None:
            raise KeyError(f"no checkpoint seq={seq} for {instance}")
        cp = self.load(cn, instance)
        cp.seq += 1
        cp.reference = state_from_json(json.loads(row[0]) if row[0] else None)
        cp.held = {}
        cp.steps_held = 0
        cp.held_since = None
        now = time.time()
        cp.last = {"kind": "rollback", "passed": None, "failed_at": None, "at": now, "to_seq": int(seq)}
        self.save(cn, cp)
        self._log(cn, cp, "rollback", now, actor=actor)
        return cp

    def release(self, cn: sqlite3.Connection, instance: str, cur: CATState, *, actor: str) -> Checkpoint:
        """A human drops the held potential: the current state becomes the reference
        without being realised.  Logged with the actor; nothing is written to the mesh."""
        if not actor:
            raise ValueError("release needs an actor (human signer)")
        cp = self.load(cn, instance)
        cp.seq += 1
        cp.reference = cur
        cp.held = {}
        cp.steps_held = 0
        cp.held_since = None
        now = time.time()
        cp.last = {"kind": "released", "passed": None, "failed_at": None, "at": now}
        self.save(cn, cp)
        self._log(cn, cp, "released", now, actor=actor)
        return cp


__all__ = ["KV_PREFIX", "TABLE", "Checkpoint", "CheckpointStore", "ensure_tables",
           "state_to_json", "state_from_json", "residual_to_json", "residual_from_json"]
