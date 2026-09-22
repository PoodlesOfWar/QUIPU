"""mirror_training — a held gate trains from its mirror image; the boundary is
not crossed.

Ruling (operator, 2026-09-22): *any gate that has a hold should use the mirror
image to train from without boundary crossing, in order to further
QUIPU_SELF_ORGANISING.*

What a hold taught the system before this module
------------------------------------------------
Nothing.  A Decision held at gate k leaves the residual in the checkpoint
(``residual_checkpoint``: kind ``held``), makes r-ADMIN's realised step a
no-op (``divine_blessing._blessed_radam``) and leaves the prior where it was
(``learned_prior``: advance only on realisation).  The next step re-measures
the same residual against the same reference and holds it again.  On the live
instance that was 70 steps over four days on one unchanged residual: a fixed
point, which is the opposite of self-organisation.

The mirror image
----------------
The CAT state is complex — realised = Re, latent (翈) = Im — and
``cat_residual.quarter_turn`` multiplies by i: realised → latent.  A hold has
one of two causes, and each has its own mirror:

    human hold      love, shared_entity or beautiful_output held for want of
                    an accepted attestation.  The shape is admissible; it is
                    not yet attested.  Mirror = i·r, the quarter turn: the
                    held content becomes latent potential and is learned as
                    such.
    physical hold   displacement, weyl, sici; beautiful_output's Lipschitz
                    breach; shared_entity's "edge raises … remainder".  The
                    shape itself failed.  Mirror = −r, the reflection through
                    the reference: learn away from the held shape.

Where the training goes — three latent stores, none of which a gate reads
---------------------------------------------------------------------------
1.  **r-ADMIN's mirror state** (``entirety:mirror:<instance>`` → ``radam``).
    ``radam_step(state, grad_real=0, grad_imag=±|Σ r|)``: only the latent
    channel of the bifurcated gradient is fed, so the toroidal phase advances
    by exactly arg(±i) = ±π/2 per hold — the quarter turn in r-ADMIN's own
    coordinates (i² = −1: two human holds are the deepen parity).  The
    realised state ``entirety:radam_state:<instance>`` is not touched.
2.  **The mirror prior** (same record → ``prior``): Oja on |r| with +η under a
    human hold and −η under a physical one.  ``entirety:prior`` is not
    touched and ``observer_tangent`` never reads the mirror prior.  The gap
    between the two priors is the held potential measured in prior space.
3.  **The SOMN mirror drive** (same record → ``drive``): ±u, u the unit |r|
    over the senses, consumed by ``memristive_axes.step`` on the next broaden
    step — potentiation faster along a human hold, slower along a physical
    one.  It changes where the next pulse's budget flows, which the
    2026-09-22 ruling on Invariance #7 places inside the operator's grant.

Without boundary crossing
-------------------------
None of the three is an input to any gate.  The checkpoint reference does not
advance, no edge is written, and no attestation, decision, emergence report
or checkpoint row is touched.  This module writes only
``brain_kv["entirety:mirror:<instance>"]`` and the append-only table
``entirety_mirror_log``; ``_kv_set`` refuses any other key.  Stdlib only.

翈 — the mirror is what the hold would have written, kept on the latent side
of the line.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Mapping

from .cat_residual import SENSES, quarter_turn
from .learned_prior import _normalise_sum, default_weights, eta_from_env
from .residual_checkpoint import CheckpointStore

KV_PREFIX: str = "entirety:mirror:"
KV_DECISION: str = "entirety:divine_blessing"       # read only (divine_blessing.KV_DECISION)
TABLE: str = "entirety_mirror_log"

PHYSICAL_GATES: tuple[str, ...] = ("displacement", "weyl", "sici")
HUMAN_GATES: tuple[str, ...] = ("love", "shared_entity", "beautiful_output")
PHYSICAL: str = "physical"
HUMAN: str = "human"
RULING: str = ("2026-09-22 operator ruling: any gate that has a hold should use the mirror "
               "image to train from without boundary crossing, in order to further "
               "QUIPU_SELF_ORGANISING")


# ---------------------------------------------------------------------------
# Classification and the two mirrors
# ---------------------------------------------------------------------------

def classify_hold(failed_at: str | None, reason: str = "") -> str | None:
    """``physical`` when the shape failed, ``human`` when an attestation is
    missing, None when there is nothing to classify."""
    gate = (failed_at or "").strip()
    text = reason or ""
    if not gate:
        return None
    if gate in PHYSICAL_GATES:
        return PHYSICAL
    if gate == "beautiful_output" and ("Lipschitz" in text or "unclamped weight not supplied" in text):
        return PHYSICAL
    if gate == "shared_entity" and ("raises" in text or "not measured" in text or "no counterpart" in text):
        return PHYSICAL
    if gate in HUMAN_GATES:
        return HUMAN
    return None


def mirror_image(r: Mapping[str, complex], kind: str) -> dict[str, complex]:
    """i·r for a human hold (quarter turn into the latent axis); −r for a
    physical hold (reflection through the reference)."""
    if kind == HUMAN:
        return quarter_turn(r)
    if kind == PHYSICAL:
        return {s: -complex(r.get(s, 0j)) for s in SENSES}
    raise ValueError(f"unknown hold kind {kind!r}")


def latent_gradient(r: Mapping[str, complex], kind: str) -> float:
    """What r-ADMIN's latent channel receives: ±|Σ_k r_k| (the realised
    channel receives 0).  |i·z| = |−z| = |z|, so the magnitude is the mirror's."""
    total = sum(complex(r.get(s, 0j)) for s in SENSES)
    mag = abs(total)
    return mag if kind == HUMAN else -mag


def unit_direction(r: Mapping[str, complex]) -> dict[str, float]:
    mags = {s: abs(complex(r.get(s, 0j))) for s in SENSES}
    n = math.sqrt(sum(v * v for v in mags.values()))
    return {s: (v / n if n > 0.0 else 0.0) for s, v in mags.items()}


def oja_signed(weights: Mapping[str, float], r: Mapping[str, complex], eta: float, sign: int) -> dict[str, float]:
    """One Oja step (+1) or anti-Oja step (−1) of the unit prior along |r|."""
    a = [abs(complex(r.get(s, 0j))) for s in SENSES]
    w = [float(weights.get(s, 1.0 / 6)) for s in SENSES]
    nw = math.sqrt(sum(x * x for x in w)) or 1.0
    w_hat = [x / nw for x in w]
    if eta <= 0.0 or not any(a) or sign == 0:
        return _normalise_sum(dict(zip(SENSES, w)))
    y = sum(ai * wi for ai, wi in zip(a, w_hat))
    new_hat = [wi + sign * eta * y * (ai - y * wi) for ai, wi in zip(a, w_hat)]
    return _normalise_sum(dict(zip(SENSES, new_hat)))


# ---------------------------------------------------------------------------
# Persistence — only entirety:mirror:* and the log table
# ---------------------------------------------------------------------------

def ensure_tables(cn: sqlite3.Connection) -> None:
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE}("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, instance TEXT NOT NULL, seq INTEGER NOT NULL, "
        "at REAL NOT NULL, failed_at TEXT NOT NULL, kind TEXT NOT NULL, reason TEXT, "
        "mirror TEXT NOT NULL, g_im REAL NOT NULL, theta REAL NOT NULL, pressure REAL NOT NULL, "
        "prior TEXT NOT NULL, drive TEXT NOT NULL)")


def _kv_get(cn: sqlite3.Connection, key: str, default=None):
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else default
    except Exception:
        return default


def _kv_set(cn: sqlite3.Connection, key: str, value) -> None:
    if not key.startswith(KV_PREFIX):
        raise PermissionError(f"mirror_training writes only {KV_PREFIX}* keys, not {key!r}")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def _new_record() -> dict:
    return {"last_seq": 0, "holds_trained": 0, "radam": {}, "prior": default_weights(),
            "drive": {}, "kind": None, "history": [], "ruling": RULING}


def load_record(cn: sqlite3.Connection, instance: str) -> dict:
    rec = _kv_get(cn, KV_PREFIX + instance, None)
    if not isinstance(rec, Mapping):
        return _new_record()
    out = _new_record()
    out["last_seq"] = int(rec.get("last_seq", 0) or 0)
    out["holds_trained"] = int(rec.get("holds_trained", 0) or 0)
    out["radam"] = dict(rec.get("radam") or {})
    prior = rec.get("prior")
    out["prior"] = dict(prior) if isinstance(prior, Mapping) and all(s in prior for s in SENSES) else default_weights()
    drive = rec.get("drive")
    out["drive"] = {s: float(drive.get(s, 0.0)) for s in SENSES} if isinstance(drive, Mapping) else {}
    out["kind"] = rec.get("kind")
    out["history"] = list(rec.get("history") or [])
    return out


def mirror_drive(cn: sqlite3.Connection, instance: str = "system_entirety") -> dict[str, float]:
    """The drive the last trained hold left for the SOMN ({} when none)."""
    return dict(load_record(cn, instance).get("drive") or {})


def _hold_reason(cn: sqlite3.Connection, failed_at: str) -> str:
    d = _kv_get(cn, KV_DECISION, None)
    if not isinstance(d, Mapping):
        return ""
    for g in d.get("gates") or []:
        if isinstance(g, Mapping) and g.get("name") == failed_at:
            return str(g.get("reason") or "")
    return ""


# ---------------------------------------------------------------------------
# Train from the hold
# ---------------------------------------------------------------------------

def train_from_hold(cn: sqlite3.Connection, instance: str = "system_entirety", *,
                    eta: float | None = None, now: float | None = None) -> dict | None:
    """If the instance's last decision is a hold this module has not trained
    on yet, train the three latent stores from its mirror image and record
    it.  Returns the summary, or None when there was nothing to train from."""
    from ..radam_optimizer import radam_step          # the module attribute: divine_blessing's wrapper,
    #                                                   which lets a state with no failed Decision run
    eta = eta_from_env() if eta is None else float(eta)
    now = time.time() if now is None else float(now)
    cp = CheckpointStore().load(cn, instance)
    if (cp.last or {}).get("kind") != "held" or not cp.held:
        return None
    rec = load_record(cn, instance)
    if cp.seq <= int(rec["last_seq"]):
        return None
    failed_at = str((cp.last or {}).get("failed_at") or "")
    reason = _hold_reason(cn, failed_at)
    kind = classify_hold(failed_at, reason)
    if kind is None:
        return None
    r = {s: complex(cp.held.get(s, 0j)) for s in SENSES}
    if not any(abs(v) > 0.0 for v in r.values()):
        return None

    m = mirror_image(r, kind)
    g_im = latent_gradient(r, kind)
    sign = 1 if kind == HUMAN else -1

    # 1. r-ADMIN's mirror state: latent channel only.
    radam_state = dict(rec["radam"])
    pressure = radam_step(radam_state, 0.0, g_im, use_torus=True)
    theta = float(radam_state.get("theta", 0.0))

    # 2. the mirror prior
    prior = oja_signed(rec["prior"], r, eta, sign)

    # 3. the SOMN mirror drive
    u = unit_direction(r)
    drive = {s: sign * u[s] for s in SENSES}

    rec.update({"last_seq": int(cp.seq), "holds_trained": int(rec["holds_trained"]) + 1,
                "radam": radam_state, "prior": prior, "drive": drive, "kind": kind,
                "updated_at": now})
    entry = {"seq": int(cp.seq), "at": now, "failed_at": failed_at, "kind": kind,
             "g_im": round(g_im, 9), "theta": round(theta, 6), "pressure": round(float(pressure), 6)}
    rec["history"] = (list(rec["history"]) + [entry])[-32:]
    ensure_tables(cn)
    _kv_set(cn, KV_PREFIX + instance, rec)
    cn.execute(
        f"INSERT INTO {TABLE}(instance, seq, at, failed_at, kind, reason, mirror, g_im, theta, pressure, prior, drive) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (instance, int(cp.seq), now, failed_at, kind, reason,
         json.dumps({s: [m[s].real, m[s].imag] for s in SENSES}), g_im, theta, float(pressure),
         json.dumps(prior), json.dumps(drive)))
    return {**entry, "reason": reason, "mirror": {s: [round(m[s].real, 9), round(m[s].imag, 9)] for s in SENSES},
            "prior": {s: round(v, 9) for s, v in prior.items()}, "drive": {s: round(v, 6) for s, v in drive.items()},
            "holds_trained": rec["holds_trained"]}


def rows(cn: sqlite3.Connection, instance: str = "system_entirety", limit: int = 64) -> list[dict]:
    ensure_tables(cn)
    rs = cn.execute(
        f"SELECT seq, at, failed_at, kind, reason, g_im, theta, pressure FROM {TABLE} "
        "WHERE instance=? ORDER BY id DESC LIMIT ?", (instance, int(limit))).fetchall()
    cols = ["seq", "at", "failed_at", "kind", "reason", "g_im", "theta", "pressure"]
    return [dict(zip(cols, row)) for row in reversed(rs)]


__all__ = ["KV_PREFIX", "KV_DECISION", "TABLE", "PHYSICAL_GATES", "HUMAN_GATES", "PHYSICAL", "HUMAN",
           "RULING", "classify_hold", "mirror_image", "latent_gradient", "unit_direction", "oja_signed",
           "ensure_tables", "load_record", "mirror_drive", "train_from_hold", "rows"]
