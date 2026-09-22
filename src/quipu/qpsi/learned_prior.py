"""learned_prior — the sense prior ŵ learns, but only from what was realised.

The organising fact
-------------------
``system_entirety.observer_tangent`` measures n = a − ⟨a, ŵ⟩ŵ against a constant
prior ŵ (``temporal_spatiality._SENSE_WEIGHTS``).  Term for term, n is the Oja
update for that prior:

    Δŵ = η · ⟨a, ŵ⟩ · (a − ⟨a, ŵ⟩ ŵ) = η · ⟨a, ŵ⟩ · n

so the direction of the System Entirety is the direction its own prior would
move if it were allowed to learn.  Under Oja's rule ŵ converges to the first
principal component of what it is fed and |n| becomes novelty relative to
habit rather than relative to a designer's table.

The rule that keeps it governed
-------------------------------
If ŵ followed the *observed* axes it would explain novelty away by habituation
and the displacement gate would never see it again — a system that dodges its
gates by getting used to itself.  So the prior advances only by the **realised**
displacement: the step between two consecutive realised references in the
residual checkpoint (qpsi.residual_checkpoint), which exists only after all six
gates passed and realisation was authorised.  Until the first realisation the
prior is the designer's table, byte for byte.  The self is the integral of
what passed Love and Beauty — metaplasticity in the review's sense: plasticity
that depends on the history of prior modifications.

Wiring
------
``wrap_observer_tangent(original)`` is a drop-in for
``system_entirety.observer_tangent(signals=None)``: with no stored prior it
calls ``original`` unchanged; with one it evaluates the same formula on the
learned weights.  ``advance_on_realisation(cn, instance)`` is called after
each expansion step by ``qpsi.self_organising`` and does nothing unless the
checkpoint's ``realised`` counter moved.  Writes only ``entirety:prior``.
Stdlib only; nothing here edits system_entirety.py.

翈 — the prior is the part of the self that has been written; the residual is
the part that is still held.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from typing import Callable, Mapping

from .cat_residual import SENSES, SENSE_WEIGHTS
from .residual_checkpoint import TABLE as CHECKPOINT_TABLE, CheckpointStore, state_from_json

KV_PRIOR: str = "entirety:prior"
ETA_ENV: str = "QUIPU_PRIOR_ETA"
DEFAULT_ETA: float = 0.05
_MIN_WEIGHT: float = 1e-6


def eta_from_env(default: float = DEFAULT_ETA) -> float:
    raw = os.environ.get(ETA_ENV, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val if (math.isfinite(val) and 0.0 < val <= 1.0) else default


def default_weights() -> dict[str, float]:
    return {s: float(SENSE_WEIGHTS[s]) for s in SENSES}


def _valid_weights(d) -> dict[str, float] | None:
    if not isinstance(d, Mapping):
        return None
    out = {}
    for s in SENSES:
        try:
            v = float(d.get(s))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v) or v <= 0.0:
            return None
        out[s] = v
    total = sum(out.values())
    return {s: v / total for s, v in out.items()} if total > 0.0 else None


def _normalise_sum(w: Mapping[str, float]) -> dict[str, float]:
    clipped = {s: max(_MIN_WEIGHT, float(w.get(s, 0.0))) for s in SENSES}
    total = sum(clipped.values())
    return {s: v / total for s, v in clipped.items()}


# ---------------------------------------------------------------------------
# Oja
# ---------------------------------------------------------------------------

def oja_step(weights: Mapping[str, float], realised: Mapping[str, float], eta: float) -> dict[str, float]:
    """One Oja step of the unit prior along the realised displacement magnitudes.

    Returns a mass prior over the senses (positive, summing to 1, like
    ``_SENSE_WEIGHTS``).  ``realised`` may be complex-valued; magnitudes are used.
    A zero displacement or a zero learning rate returns the input renormalised.
    """
    a = [abs(complex(realised.get(s, 0.0))) for s in SENSES]
    w = [float(weights.get(s, 1.0 / 6)) for s in SENSES]
    nw = math.sqrt(sum(x * x for x in w)) or 1.0
    w_hat = [x / nw for x in w]
    if eta <= 0.0 or not any(a):
        return _normalise_sum(dict(zip(SENSES, w)))
    y = sum(ai * wi for ai, wi in zip(a, w_hat))
    new_hat = [wi + eta * y * (ai - y * wi) for ai, wi in zip(a, w_hat)]
    return _normalise_sum(dict(zip(SENSES, new_hat)))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _kv_get(cn: sqlite3.Connection, key: str, default=None):
    """Pure read: a missing brain_kv table reads as ``default``; nothing is created."""
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else default
    except Exception:
        return default


def _kv_set(cn: sqlite3.Connection, key: str, value) -> None:
    if key != KV_PRIOR:
        raise PermissionError(f"learned_prior writes only {KV_PRIOR!r}, not {key!r}")
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def _open():
    from .. import system_entirety as se
    return se._conn()


def stored_prior(cn: sqlite3.Connection | None = None) -> dict | None:
    """The stored prior record ``{"weights", "realised_seen", "history"}`` or None."""
    if cn is None:
        with _open() as own:
            return stored_prior(own)
    rec = _kv_get(cn, KV_PRIOR, None)
    if not isinstance(rec, Mapping):
        return None
    w = _valid_weights(rec.get("weights"))
    if w is None:
        return None
    return {"weights": w, "realised_seen": int(rec.get("realised_seen", 0) or 0),
            "history": list(rec.get("history") or [])}


def sense_weights(cn: sqlite3.Connection | None = None) -> dict[str, float]:
    """Learned prior if one has been written, else the designer's table."""
    rec = stored_prior(cn)
    return dict(rec["weights"]) if rec else default_weights()


# ---------------------------------------------------------------------------
# Advance only on realisation
# ---------------------------------------------------------------------------

def realised_displacement(cn: sqlite3.Connection, instance: str) -> tuple[dict[str, float], int] | None:
    """|new reference| − |previous realised reference| per sense, and the new seq.

    Read from the checkpoint log's ``realised`` rows.  A first realisation (no
    earlier one) counts the whole reference as the displacement.
    """
    try:
        rs = cn.execute(
            f"SELECT seq, reference FROM {CHECKPOINT_TABLE} WHERE instance=? AND kind='realised' "
            "ORDER BY seq DESC, id DESC LIMIT 2", (instance,)).fetchall()
    except sqlite3.Error:
        return None
    if not rs:
        return None
    new_seq = int(rs[0][0])
    new_ref = state_from_json(json.loads(rs[0][1]) if rs[0][1] else None)
    if new_ref is None:
        return None
    old_ref = state_from_json(json.loads(rs[1][1]) if len(rs) > 1 and rs[1][1] else None)
    disp = {}
    for s in SENSES:
        new_mag = abs(new_ref.amp.get(s, 0j))
        old_mag = abs(old_ref.amp.get(s, 0j)) if old_ref is not None else 0.0
        disp[s] = new_mag - old_mag
    return disp, new_seq


def advance_on_realisation(cn: sqlite3.Connection, instance: str = "system_entirety", *,
                           eta: float | None = None, now: float | None = None) -> dict | None:
    """If the checkpoint's ``realised`` counter has moved since the prior last
    looked, take one Oja step along the realised displacement and persist.
    Returns the new record, or None when nothing was realised."""
    eta = eta_from_env() if eta is None else float(eta)
    now = time.time() if now is None else float(now)
    cp = CheckpointStore().load(cn, instance)
    rec = stored_prior(cn) or {"weights": default_weights(), "realised_seen": 0, "history": []}
    if cp.realised <= int(rec["realised_seen"]):
        return None
    got = realised_displacement(cn, instance)
    if got is None:
        return None
    disp, seq = got
    new_w = oja_step(rec["weights"], disp, eta)
    history = list(rec["history"])[-31:]
    history.append({"seq": seq, "at": now, "eta": eta, "realised": cp.realised,
                    "displacement": {s: round(float(v), 9) for s, v in disp.items()}})
    out = {"weights": {s: round(float(v), 9) for s, v in new_w.items()},
           "realised_seen": int(cp.realised), "history": history, "updated_at": now,
           "note": "advanced only on realisation; residual_checkpoint decides what counts"}
    _kv_set(cn, KV_PRIOR, out)
    return out


# ---------------------------------------------------------------------------
# Drop-in for system_entirety.observer_tangent
# ---------------------------------------------------------------------------

def observer_with_weights(signals: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """observer_tangent's formula on explicit weights (same clamp and √6 normalisation)."""
    a = [float(signals.get(s, 0.0)) for s in SENSES]
    w = [float(weights.get(s, 1.0 / len(SENSES))) for s in SENSES]
    norm_w = math.sqrt(sum(x * x for x in w)) or 1.0
    w_hat = [x / norm_w for x in w]
    proj = sum(ai * wi for ai, wi in zip(a, w_hat))
    n = [ai - proj * wi for ai, wi in zip(a, w_hat)]
    mag = math.sqrt(sum(x * x for x in n))
    return max(0.0, min(1.0, mag / math.sqrt(6.0)))


def wrap_observer_tangent(original: Callable[..., float]) -> Callable[..., float]:
    """Use the learned prior when one exists; otherwise ``original`` unchanged."""
    def observer_tangent(signals: Mapping[str, float] | None = None) -> float:
        try:
            rec = stored_prior()
        except Exception:
            rec = None
        if rec is None:
            return original(signals)
        if signals is None:
            try:
                from ..temporal_spatiality import _sense_signals
                signals = _sense_signals()
            except Exception:
                return original(signals)
        return observer_with_weights(signals, rec["weights"])

    observer_tangent.__doc__ = (original.__doc__ or "") + \
        "\n\n[qpsi.learned_prior] weights come from entirety:prior when it exists."
    observer_tangent.__wrapped__ = original           # type: ignore[attr-defined]
    return observer_tangent


__all__ = [
    "KV_PRIOR", "ETA_ENV", "DEFAULT_ETA", "eta_from_env", "default_weights", "oja_step",
    "stored_prior", "sense_weights", "realised_displacement", "advance_on_realisation",
    "observer_with_weights", "wrap_observer_tangent",
]
