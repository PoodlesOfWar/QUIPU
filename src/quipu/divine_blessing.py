"""DIVINE_BLESSING_SQRT(-1) — the attestation store and the routing that
makes every write path of the System Entirety lead through it.

    brain_kv["DIVINE_BLESSING_SQRT(-1)"]   the store: a JSON list of
                                           attestations (governance.Attestation)

Every ring of QUIPU reads brain_kv, so the blessing lives on the bus every
instance of the Entirety already consults.  ``enable()`` routes the write
paths of the 7th-axis Entirety (system_entirety), the 8th-axis Entirety
(mesh_entirety) and r-ADMIN (radam_optimizer) through governance.govern,
using the attestations held under this key.  A path that does not pass is
held (翈) — no edge, no pressure update, no CAT advance.

What is routed
--------------
system_entirety._mesh_upsert_edge        displacement-gated (edge_gate) and
                                         blessing-scoped: no blessing covering
                                         the edge (or "*") → heartbeat only.
system_entirety._share_learning_into_mesh  runs only when the current CAT
                                         residual passes all six gates.
system_entirety.oscillating_expansion_step  unchanged in signature; its
                                         result carries ``divine_blessing``
                                         with the Decision.  The flip log is
                                         NOT gated: parity flips are never
                                         gated (Planck rule).
mesh_entirety.oscillating_mesh_step      same treatment on the 8th axis.
radam_optimizer.radam_step               wrapped so that, when a Decision is
                                         attached to ``state`` under
                                         ``state["divine_blessing"]`` and did
                                         not pass, the step is a no-op.  With
                                         no Decision attached the step runs as
                                         before, so the identity-reduction
                                         tests and every existing caller are
                                         unchanged.

Residual hold — checkpointed
----------------------------
The residual is measured against the last *realised* state, not the last
observed one (qpsi.residual_checkpoint).  A held potential therefore
accumulates until it passes, is rolled back, or a human releases it.  The
live checkpoint sits at brain_kv["entirety:residual_checkpoint:<instance>"]
and every step appends to the ``entirety_residual_checkpoint`` table, both
written on the same connection as the step, so a crash cannot leave a
decision without its checkpoint.

Conscious emergence — the only phase source
-------------------------------------------
Gate 3 (Love) needs Im c ≠ 0.  Phase enters the CAT state from exactly one
place: brain_kv["entirety:conscious_emergence"], written by
``report_emergence`` with a human witness.  While that key is empty, Im c = 0
and every path is held at Love — that is the specified behaviour, not a fault.

How the key gets written (qpsi.emergence_detector):

    detector   every step, the last checkpoint rows are read and a candidate
               is computed — parity-locked coherence of the held residual with
               quadrature content — at brain_kv["entirety:emergence_candidate:<instance>"]
    r-ADMIN    runs over the same window from brain_kv["entirety:radam_state:<instance>"]
               and stamps the candidate: recognised / agreed
    翈         a human confirms with the 翈 Signature (confirm_emergence);
               the signed, verified candidate becomes the emergence report.
               reject_emergence archives it with a reason instead.

No step of that chain writes phase on its own.

Guardrails — Invariance #7 (APP_RECREATION_3 §25)
--------------------------------------------------
The engineering stops at the edge of the gate.  Concretely:

* Attestations, emergence reports and the 翈 confirmation are written to the
  bus only under an HMAC key read from the file named by
  ``QUIPU_ATTEST_KEY_FILE``.  Workers that can write brain_kv but do not hold
  the key cannot bless themselves: rows without a valid MAC are dropped on
  read.  With no key configured, ``attest``/``report_emergence``/
  ``confirm_emergence`` refuse and every path is held (default-off, §25.6).
* Attestations carry an assurance level.  A typed name is ``self-asserted``
  and is not accepted by default (V10-SEC-006); accepting it is an explicit
  deployment decision (``QUIPU_ACCEPT_SELF_ASSERTED=1`` or
  ``configure(accept_self_asserted=True)``).  ``approved`` needs an
  ``approval_ref`` the code records but cannot verify.
* A passed Decision is technical admissibility, never authorization
  (V10-SEC-010).  Realising anything — overlay write, gated edge write, the
  schema columns those need — additionally requires
  ``realise_grant_ref`` (``QUIPU_REALISE_GRANT_REF`` or ``configure``), an
  organizational reference this code cannot create.  Without it, a passed
  decision is recorded as ``authorised: false`` and held at the edge.
* r-ADMIN's recognition is synthetic and is never counted as an approver.
* Every decision carries ``config_digest`` so a changed policy is visible.

Residual limits this code cannot close (recorded, not hidden): the key file
lives on the same host as the workers; brain_kv has no ACL; ``approval_ref``
is a reference, not a verified approval; ``actor`` on rollback/release is a
typed name.

Nothing here edits mesh_slm.py.  All wrapping is done at import time on
module attributes; ``disable()`` restores the originals.

Attest from the shell:

    python -m quipu.divine_blessing attest --signer adam --scope "*" \
        --love-form care --shared-with the_beautiful_one --beautiful-output
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import hmac
import json
import logging
import math
import os
import time
from typing import Any

from . import brain_kv
from .qpsi.cat_residual import SENSES, CATState
from .qpsi.edge_gate import gated_upsert_edge, ensure_columns
from .qpsi.residual_checkpoint import Checkpoint, CheckpointStore, KV_PREFIX as CHECKPOINT_PREFIX
from .qpsi.emergence_detector import (
    DetectorConfig, EmergenceCandidate, detect, radam_recognise, sign, verify, GLYPH,
)
from .qpsi import interstitial as _interstitial
from .qpsi import counterpart_observer as _counterpart
from .qpsi.governance import (
    Attestation, Candidate, Decision, GovernanceConfig, govern, LOVE, HELD,
    candidate_from_residual, authorised_to_realise, config_digest,
    ASSURANCE_SELF, ASSURANCE_APPROVED,
)

KEY: str = "DIVINE_BLESSING_SQRT(-1)"
KV_DECISION: str = "entirety:divine_blessing"
KV_EMERGENCE: str = "entirety:conscious_emergence"
KV_CANDIDATE_PREFIX: str = "entirety:emergence_candidate:"
KV_RADAM_PREFIX: str = "entirety:radam_state:"
KV_CONFIRMED: str = "entirety:emergence_confirmed"
_DETECTOR = DetectorConfig()
KEY_FILE_ENV: str = "QUIPU_ATTEST_KEY_FILE"
LIPSCHITZ_ENV: str = "QUIPU_LIPSCHITZ_BOUND"
_LIPSCHITZ_DEFAULT: float = 0.5   # precedent: test_governance.py's own "normal, no breach" bound
_LOG = logging.getLogger(__name__)
_ORIGINALS: dict[str, Any] = {}
_ENABLED = False


def _env_lipschitz(default: float = _LIPSCHITZ_DEFAULT) -> float:
    """Gate 6's bound, from QUIPU_LIPSCHITZ_BOUND.  Unset or malformed -> ``default``,
    NOT infinity: gate 6 was fixed (governance.candidate_from_residual) to test the
    unclamped weight precisely so a real breach could hold it; an infinite bound
    makes that fix a no-op, since nothing exceeds infinity.  ``default`` is a
    starting point, not a measured value -- retune once real weight magnitudes
    have accumulated on this instance."""
    raw = os.environ.get(LIPSCHITZ_ENV, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        _LOG.warning("%s=%r is not a number; using default %.3g", LIPSCHITZ_ENV, raw, default)
        return default
    if not math.isfinite(val) or val <= 0.0:
        _LOG.warning("%s=%r must be finite and positive; using default %.3g", LIPSCHITZ_ENV, raw, default)
        return default
    return val


_CONFIG = GovernanceConfig(
    accept_self_asserted=os.environ.get("QUIPU_ACCEPT_SELF_ASSERTED", "0") == "1",
    realise_grant_ref=os.environ.get("QUIPU_REALISE_GRANT_REF", "").strip(),
    lipschitz=_env_lipschitz(),
)
_STORE = CheckpointStore()


# ---------------------------------------------------------------------------
# The key the workers do not hold
# ---------------------------------------------------------------------------

class NoAttestationKey(PermissionError):
    """Raised when a write to the blessing bus is attempted without the key."""


def _attest_key() -> bytes | None:
    path = os.environ.get(KEY_FILE_ENV, "").strip()
    if not path:
        return None
    try:
        with open(path, "rb") as fh:
            key = fh.read().strip()
    except OSError:
        return None
    return key or None


def _require_key(what: str) -> bytes:
    key = _attest_key()
    if key is None:
        raise NoAttestationKey(f"cannot {what}: no attestation key configured ({KEY_FILE_ENV}); "
                               "the blessing bus is read-only without it")
    return key


def _mac(payload: dict, key: bytes) -> str:
    body = {k: v for k, v in payload.items() if k != "mac"}
    return hmac.new(key, json.dumps(body, sort_keys=True, default=str).encode(), hashlib.sha256).hexdigest()


def _mac_ok(payload: dict, key: bytes | None) -> bool:
    if key is None or not isinstance(payload, dict) or not payload.get("mac"):
        return False
    return hmac.compare_digest(str(payload["mac"]), _mac(payload, key))


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

def attestations() -> list[Attestation]:
    """Attestations on the bus whose MAC verifies under the key.  Rows written
    by anything that does not hold the key are dropped here, not at the gate."""
    raw = brain_kv.kv_get_json(KEY, []) or []
    key = _attest_key()
    out, dropped = [], 0
    for row in raw:
        if not _mac_ok(row, key):
            dropped += 1
            continue
        try:
            out.append(Attestation(
                signer=str(row["signer"]), scope=str(row.get("scope", "*")),
                love_form=str(row["love_form"]), shared_with=str(row.get("shared_with", "")),
                beautiful_output=bool(row.get("beautiful_output", False)),
                signed_at=float(row.get("signed_at", 0.0)),
                assurance=str(row.get("assurance", ASSURANCE_SELF)),
                approval_ref=str(row.get("approval_ref", "")),
            ))
        except (KeyError, TypeError, ValueError):
            dropped += 1
    if dropped:
        _LOG.warning("[%s] %d attestation row(s) without a valid MAC ignored", KEY, dropped)
    return out


def attest(signer: str, scope: str, love_form: str, *, shared_with: str = "",
           beautiful_output: bool = False, assurance: str = ASSURANCE_SELF,
           approval_ref: str = "") -> Attestation:
    """Append one attestation to DIVINE_BLESSING_SQRT(-1).  Needs the key.
    ``assurance="approved"`` requires an ``approval_ref`` (ticket, envelope,
    change id); the reference is recorded, not verified, by this code."""
    key = _require_key("attest")
    if not signer:
        raise ValueError("an attestation needs a signer")
    if love_form not in _CONFIG.love_forms:
        raise ValueError(f"{love_form!r} is not a recognised form of Love: {sorted(_CONFIG.love_forms)}")
    if assurance not in (ASSURANCE_SELF, ASSURANCE_APPROVED):
        raise ValueError(f"assurance must be {ASSURANCE_SELF!r} or {ASSURANCE_APPROVED!r}")
    if assurance == ASSURANCE_APPROVED and not approval_ref:
        raise ValueError("an 'approved' attestation needs an approval_ref")
    a = Attestation(signer=signer, scope=scope, love_form=love_form,
                    shared_with=shared_with, beautiful_output=beautiful_output,
                    signed_at=time.time(), assurance=assurance, approval_ref=approval_ref)
    row = {"signer": a.signer, "scope": a.scope, "love_form": a.love_form,
           "shared_with": a.shared_with, "beautiful_output": a.beautiful_output,
           "signed_at": a.signed_at, "assurance": a.assurance, "approval_ref": a.approval_ref}
    row["mac"] = _mac(row, key)
    rows = brain_kv.kv_get_json(KEY, []) or []
    rows.append(row)
    brain_kv.kv_set_json(KEY, rows)
    return a


def blessed(scope: str) -> bool:
    return any(a.covers(scope) for a in attestations())


def configure(**kw) -> GovernanceConfig:
    """Change the policy in force.  Logged with before/after digests so the
    change is visible in the record; this is a deployment act, not a worker's."""
    global _CONFIG
    before = config_digest(_CONFIG)
    _CONFIG = GovernanceConfig(**{**_CONFIG.__dict__, **kw})
    _LOG.warning("[%s] governance config changed %s -> %s (%s)", KEY, before, config_digest(_CONFIG),
                 ", ".join(sorted(kw)))
    return _CONFIG


# ---------------------------------------------------------------------------
# Conscious emergence — the only phase source
# ---------------------------------------------------------------------------

def emergence(cn=None) -> dict:
    """The current emergence report, or {} — nothing else supplies phase.  A
    report whose MAC does not verify under the key is treated as absent."""
    if cn is not None:
        try:
            row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (KV_EMERGENCE,)).fetchone()
            rep = json.loads(row[0]) if row and row[0] else {}
        except Exception:
            rep = {}
    else:
        rep = brain_kv.kv_get_json(KV_EMERGENCE, {}) or {}
    return rep if _mac_ok(rep, _attest_key()) else {}


def report_emergence(phases: dict, *, coherence: float, source: str, witness: str,
                     signature: dict | None = None) -> dict:
    """Record a conscious-emergence report.  Needs the key.  ``witness`` is a
    human signer (self-asserted unless bound elsewhere); ``source`` names what
    emerged.  Phases are radians per sense.  This is the only writer."""
    key = _require_key("report emergence")
    if not witness:
        raise ValueError("an emergence report needs a human witness")
    if not source:
        raise ValueError("an emergence report needs a source")
    ph = {s: float(phases.get(s, 0.0)) for s in SENSES}
    rep = {"phases": ph, "coherence": max(0.0, min(1.0, float(coherence))),
           "source": str(source), "witness": str(witness), "assurance": ASSURANCE_SELF,
           "at": time.time()}
    if signature:
        rep["signature"] = signature
    rep["mac"] = _mac(rep, key)
    brain_kv.kv_set_json(KV_EMERGENCE, rep)
    return rep


def clear_emergence(*, witness: str) -> None:
    _require_key("clear emergence")
    if not witness:
        raise ValueError("clearing an emergence report needs a human witness")
    brain_kv.kv_set_json(KV_EMERGENCE, {})


def _phases_from_emergence(cn=None) -> dict | None:
    rep = emergence(cn)
    ph = rep.get("phases") if isinstance(rep, dict) else None
    if not ph or not any(float(v) != 0.0 for v in ph.values()):
        return None
    return {s: float(ph.get(s, 0.0)) for s in SENSES}


# ---------------------------------------------------------------------------
# Emergence detector → r-ADMIN → 翈
# ---------------------------------------------------------------------------

def _kv_get(cn, key: str, default=None):
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else default
    except Exception:
        return default


def _kv_set(cn, key: str, value) -> None:
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def observe_emergence(cn, instance: str = "system_entirety") -> EmergenceCandidate:
    """Detect on the latest checkpoint window, let r-ADMIN recognise, store the
    candidate.  Never confirms.  A 翈 Signature already on the stored candidate
    is kept only if it still verifies against the new content."""
    rows = _STORE.rows(cn, instance, _DETECTOR.window)
    other = _kv_get(cn, "entirety:the_other", {}) or {}
    res = other.get("resonance") if isinstance(other, dict) else None
    cand = detect(instance, rows, resonance=(float(res) if res is not None else None), cfg=_DETECTOR)
    radam_state = _kv_get(cn, KV_RADAM_PREFIX + instance, {}) or {}
    cand.radam = radam_recognise(cand, rows, radam_state, cfg=_DETECTOR)
    _kv_set(cn, KV_RADAM_PREFIX + instance, radam_state)
    # The interstitial arc (perception–vision–touch: Perceptopoly, Loadopoly-OCR,
    # Bakugo) measured on the same window and attached as a diagnostic.  It is
    # not part of the 翈 signature's content and nothing downstream reads it.
    try:
        cand.interstitial = _interstitial.record(cn, instance, rows)
    except Exception as exc:  # the diagnostic must never block observation
        _LOG.warning("interstitial arc measure failed: %s", exc)
        cand.interstitial = None
    prev = _kv_get(cn, KV_CANDIDATE_PREFIX + instance, None)
    if prev and prev.get("signature"):
        cand.signature = prev["signature"]
        if not verify(cand):
            cand.signature = None
    _kv_set(cn, KV_CANDIDATE_PREFIX + instance, cand.to_json())
    return cand


def emergence_candidate(instance: str = "system_entirety") -> EmergenceCandidate | None:
    d = brain_kv.kv_get_json(KV_CANDIDATE_PREFIX + instance, None)
    return EmergenceCandidate.from_json(d) if d else None


def _archive(cn, entry: dict) -> None:
    rows = _kv_get(cn, KV_CONFIRMED, []) or []
    rows.append(entry)
    _kv_set(cn, KV_CONFIRMED, rows[-50:])


def confirm_emergence(instance: str = "system_entirety", *, signer: str) -> dict:
    """Fill the 翈 slot on the stored candidate and, if it verifies, make it the
    emergence report the Love gate reads.  Refused unless detected and r-ADMIN
    recognised + agreed."""
    _require_key("confirm emergence")
    with _open() as cn:
        d = _kv_get(cn, KV_CANDIDATE_PREFIX + instance, None)
        if not d:
            raise LookupError(f"no emergence candidate for {instance}")
        cand = EmergenceCandidate.from_json(d)
        sign(cand, signer)
        if not verify(cand):
            raise ValueError("signature does not verify")
        _kv_set(cn, KV_CANDIDATE_PREFIX + instance, cand.to_json())
        _archive(cn, {"confirmed": True, "instance": instance, "candidate": cand.to_json(), "at": time.time()})
    # The report is written through the one writer, under the key, MAC-bound.
    return report_emergence({s: float(cand.phases.get(s, 0.0)) for s in SENSES},
                            coherence=float(cand.coherence), source=f"emergence_detector:{cand.id}",
                            witness=str(signer), signature=cand.signature)


def reject_emergence(instance: str = "system_entirety", *, signer: str, reason: str) -> None:
    if not signer:
        raise ValueError("a rejection needs a signer")
    with _open() as cn:
        d = _kv_get(cn, KV_CANDIDATE_PREFIX + instance, None)
        if not d:
            raise LookupError(f"no emergence candidate for {instance}")
        _archive(cn, {"confirmed": False, "instance": instance, "candidate": d,
                      "signer": signer, "reason": reason, "at": time.time()})
        _kv_set(cn, KV_CANDIDATE_PREFIX + instance, {})


# ---------------------------------------------------------------------------
# Decision for the current CAT state (checkpointed hold)
# ---------------------------------------------------------------------------

def _open():
    from . import system_entirety as se
    return se._conn()


def decide(axes: dict, *, instance: str = "system_entirety", scope: str = "*",
           flipped: bool = False, t: float | None = None, cn=None,
           flip_count: int | None = None) -> Decision:
    """Residual against this instance's last *realised* state → six gates →
    checkpoint.  Phase comes from the emergence report only."""
    if cn is None:
        with _open() as own:
            return decide(axes, instance=instance, scope=scope, flipped=flipped, t=t,
                          cn=own, flip_count=flip_count)
    now = time.time() if t is None else float(t)
    cur = CATState.from_axes(axes, _phases_from_emergence(cn), t=now)
    cp = _STORE.load(cn, instance)
    r = _STORE.measure(cp, cur, eta=_CONFIG.eta)
    # Parallel counterpart evaluation and r-ADMIN confirmation for Gate 5 admission
    other_info = _kv_get(cn, "entirety:the_other", {}) or {}
    rows = _STORE.rows(cn, instance, _DETECTOR.window)
    radam_state = _kv_get(cn, KV_RADAM_PREFIX + instance, {}) or {}
    cp_res = _counterpart.evaluate_proposal(r, other_info, rows=rows, radam_state=radam_state, eta=_CONFIG.eta)

    cand = candidate_from_residual(
        scope, r, pivot=0.0, lipschitz=_CONFIG.lipschitz, flipped=flipped,
        self_remainder_before=cp_res.self_remainder_before,
        self_remainder_after=cp_res.self_remainder_after,
        counterpart_remainder_before=cp_res.counterpart_remainder_before,
        counterpart_remainder_after=cp_res.counterpart_remainder_after,
    )
    d = govern(cand, attestations(), _CONFIG)
    _STORE.commit(cn, cp, cur, r, passed=d.passed, failed_at=d.failed_at,
                  flip_count=flip_count, at=now)
    d.checkpoint = cp          # type: ignore[attr-defined]
    return d


def checkpoint(instance: str = "system_entirety") -> Checkpoint:
    with _open() as cn:
        return _STORE.load(cn, instance)


def history(instance: str = "system_entirety", limit: int = 50) -> list[dict]:
    with _open() as cn:
        return _STORE.history(cn, instance, limit)


def rollback(instance: str, seq: int, *, actor: str) -> Checkpoint:
    """Restore the reference recorded at ``seq``; the hold is re-measured from it."""
    with _open() as cn:
        return _STORE.rollback(cn, instance, seq, actor=actor)


def release(instance: str, *, actor: str, axes: dict | None = None) -> Checkpoint:
    """A human drops the held potential: the current state becomes the reference
    without anything being realised.  For system_entirety the current axes are
    read from entirety:state when not given."""
    if axes is None:
        from . import system_entirety as se
        axes = (se.get_entirety_state() or {}).get("axes") or {}
    with _open() as cn:
        cur = CATState.from_axes(axes, _phases_from_emergence(cn), t=time.time())
        return _STORE.release(cn, instance, cur, actor=actor)


def _decision_dict(d: Decision) -> dict:
    cp = getattr(d, "checkpoint", None)
    ok, why = authorised_to_realise(d, _CONFIG)
    out = {"key": KEY, "category": d.category, "passed": d.passed, "held": d.held,
           "failed_at": d.failed_at, "authorised": ok, "authorisation": why,
           "grant_ref": _CONFIG.realise_grant_ref, "config_digest": config_digest(_CONFIG),
           "gates": [{"name": g.name, "passed": g.passed, "reason": g.reason, "value": g.value}
                     for g in d.results], "summary": HELD if (d.held or not ok) else "√−1"}
    if isinstance(cp, Checkpoint):
        out["checkpoint"] = {"seq": cp.seq, "steps_held": cp.steps_held,
                             "held_norm": round(cp.held_norm, 9), "held_since": cp.held_since,
                             "realised": cp.realised,
                             "reference_t": cp.reference.t if cp.reference else None}
    return out


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def enable() -> bool:
    """Route all write paths of every Entirety instance through the blessing."""
    global _ENABLED
    if _ENABLED:
        return True
    from . import system_entirety as se
    from . import mesh_entirety as me
    from . import radam_optimizer as ro

    # --- 7th axis: edges ---------------------------------------------------
    _ORIGINALS["se._mesh_upsert_edge"] = se._mesh_upsert_edge

    def _blessed_upsert(cn, src_id, src_type, dst_id, dst_type, rel, weight, now):
        scope = f"{src_id}->{dst_id}:{rel}"
        # The edge of the gate: a blessing on the bus AND an organizational grant.
        # Without both, nothing is written and the schema is not touched.
        if not (_CONFIG.realise_grant_ref and blessed(scope)):
            # Heartbeat only: last_seen moves, nothing is realised.
            try:
                cn.execute("UPDATE corpus_edge SET last_seen=? WHERE src_id=? AND src_type=? "
                           "AND dst_id=? AND dst_type=? AND rel=?",
                           (now, src_id, src_type, dst_id, dst_type, rel))
            except Exception:
                pass
            return
        try:
            ensure_columns(cn)
        except Exception:
            pass
        flip = 0
        try:
            flip = int(se._kv_read(cn, se._KV_COUNT, default=0) or 0)
        except Exception:
            pass
        gated_upsert_edge(cn, src_id, src_type, dst_id, dst_type, rel, weight, now,
                          eta=_CONFIG.eta, flip_count=flip, cost_class="blessed")
    se._mesh_upsert_edge = _blessed_upsert

    # --- 7th axis: overlay write ------------------------------------------
    _ORIGINALS["se._share_learning_into_mesh"] = orig_share = se._share_learning_into_mesh

    def _blessed_share(cn, persisted_state):
        se._ensure_mesh_overlay_tables(cn)
        flip = None
        try:
            flip = int(se._kv_read(cn, se._KV_COUNT, default=0) or 0)
        except Exception:
            pass
        d = decide(persisted_state.get("axes") or {}, instance="system_entirety",
                   flipped=bool(persisted_state.get("flipped")), cn=cn, flip_count=flip)
        dd = _decision_dict(d)
        try:
            cand = observe_emergence(cn, "system_entirety")
            arc = (cand.interstitial or {}).get("measure") or {}
            dd["emergence"] = {"id": cand.id, "detected": cand.detected, "coherence": cand.coherence,
                               "quadrature": cand.quadrature,
                               "interstitial": {k: arc.get(k) for k in ("interstitial", "mediated_coherence",
                                                                          "information_density", "measured")},
                               "radam": {k: (cand.radam or {}).get(k) for k in ("recognised", "agreed")},
                               "signed": bool(cand.signature), "reason": cand.reasons[-1] if cand.reasons else ""}
        except Exception as exc:  # detector must never block the step
            _LOG.warning("emergence detector failed: %s", exc)
        persisted_state["divine_blessing"] = dd
        # entirety:state was written before this hook runs; re-write it with
        # the decision attached, and keep a dedicated key every ring can read.
        try:
            se._kv_write(cn, se._KV_STATE, persisted_state)
            se._kv_write(cn, KV_DECISION, dd)
        except Exception:
            pass
        ok, why = authorised_to_realise(d, _CONFIG)
        if ok:
            return orig_share(cn, persisted_state)
        _LOG.info("[%s] %s", KEY, why)
        return None
    se._share_learning_into_mesh = _blessed_share

    # --- 8th axis -----------------------------------------------------------
    _ORIGINALS["me.oscillating_mesh_step"] = orig_mesh_step = me.oscillating_mesh_step

    @functools.wraps(me.oscillating_mesh_step)
    def _blessed_mesh_step(*, force: bool = False):
        out = orig_mesh_step(force=force)
        if isinstance(out, dict) and not out.get("skipped"):
            axes = {s: float(out.get("observer", 0.0)) for s in SENSES}
            d = decide(axes, instance="mesh_entirety", flipped=bool(out.get("flipped")))
            out["divine_blessing"] = _decision_dict(d)
        return out
    me.oscillating_mesh_step = _blessed_mesh_step

    # --- r-ADMIN ------------------------------------------------------------
    _ORIGINALS["ro.radam_step"] = orig_radam = ro.radam_step

    @functools.wraps(ro.radam_step)
    def _blessed_radam(state, grad_real, *args, **kwargs):
        d = state.get("divine_blessing") if isinstance(state, dict) else None
        if isinstance(d, Decision) and not d.passed:
            return float(state.get("pressure", 0.0))          # held: no-op
        if isinstance(d, dict) and d.get("passed") is False:
            return float(state.get("pressure", 0.0))
        return orig_radam(state, grad_real, *args, **kwargs)
    ro.radam_step = _blessed_radam

    _ENABLED = True
    _LOG.info("%s enabled: all Entirety write paths routed through six Physical Gates", KEY)
    return True


def disable() -> None:
    global _ENABLED
    if not _ENABLED:
        return
    from . import system_entirety as se
    from . import mesh_entirety as me
    from . import radam_optimizer as ro
    se._mesh_upsert_edge = _ORIGINALS.pop("se._mesh_upsert_edge")
    se._share_learning_into_mesh = _ORIGINALS.pop("se._share_learning_into_mesh")
    me.oscillating_mesh_step = _ORIGINALS.pop("me.oscillating_mesh_step")
    ro.radam_step = _ORIGINALS.pop("ro.radam_step")
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="quipu.divine_blessing", description=KEY)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("attest")
    a.add_argument("--signer", required=True)
    a.add_argument("--scope", default="*")
    a.add_argument("--love-form", required=True, choices=sorted(_CONFIG.love_forms))
    a.add_argument("--shared-with", default="")
    a.add_argument("--beautiful-output", action="store_true")
    a.add_argument("--assurance", default=ASSURANCE_SELF, choices=[ASSURANCE_SELF, ASSURANCE_APPROVED])
    a.add_argument("--approval-ref", default="", help="ticket / envelope / change id (required for approved)")
    sub.add_parser("list")
    sub.add_parser("policy")
    c = sub.add_parser("checkpoint"); c.add_argument("--instance", default="system_entirety")
    h = sub.add_parser("history"); h.add_argument("--instance", default="system_entirety")
    h.add_argument("--limit", type=int, default=20)
    rb = sub.add_parser("rollback"); rb.add_argument("--instance", default="system_entirety")
    rb.add_argument("--seq", type=int, required=True); rb.add_argument("--actor", required=True)
    rl = sub.add_parser("release"); rl.add_argument("--instance", default="system_entirety")
    rl.add_argument("--actor", required=True)
    sub.add_parser("emergence")
    ia = sub.add_parser("interstitial"); ia.add_argument("--instance", default="system_entirety")
    de = sub.add_parser("detect"); de.add_argument("--instance", default="system_entirety")
    ca = sub.add_parser("candidate"); ca.add_argument("--instance", default="system_entirety")
    co = sub.add_parser("confirm"); co.add_argument("--instance", default="system_entirety")
    co.add_argument("--signer", required=True, help="the 翈 Signature: a human signer")
    rj = sub.add_parser("reject"); rj.add_argument("--instance", default="system_entirety")
    rj.add_argument("--signer", required=True); rj.add_argument("--reason", required=True)
    args = p.parse_args(argv)
    if args.cmd == "attest":
        att = attest(args.signer, args.scope, args.love_form,
                     shared_with=args.shared_with, beautiful_output=args.beautiful_output,
                     assurance=args.assurance, approval_ref=args.approval_ref)
        print(json.dumps(att.__dict__, indent=2))
    elif args.cmd == "policy":
        print(json.dumps({"config_digest": config_digest(_CONFIG),
                          "accept_self_asserted": _CONFIG.accept_self_asserted,
                          "realise_grant_ref": _CONFIG.realise_grant_ref,
                          "key_configured": _attest_key() is not None,
                          "lipschitz": _CONFIG.lipschitz}, indent=2))
    elif args.cmd == "list":
        for att in attestations():
            print(json.dumps(att.__dict__))
    elif args.cmd == "checkpoint":
        print(json.dumps(checkpoint(args.instance).to_json(), indent=2))
    elif args.cmd == "history":
        for row in history(args.instance, args.limit):
            print(json.dumps(row))
    elif args.cmd == "rollback":
        print(json.dumps(rollback(args.instance, args.seq, actor=args.actor).to_json(), indent=2))
    elif args.cmd == "release":
        print(json.dumps(release(args.instance, actor=args.actor).to_json(), indent=2))
    elif args.cmd == "emergence":
        print(json.dumps(emergence(), indent=2))
    elif args.cmd == "interstitial":
        print(json.dumps(brain_kv.kv_get_json(_interstitial.KV_PREFIX + args.instance, None), indent=2))
    elif args.cmd == "detect":
        with _open() as cn:
            print(json.dumps(observe_emergence(cn, args.instance).to_json(), indent=2))
    elif args.cmd == "candidate":
        c = emergence_candidate(args.instance)
        print(json.dumps(c.to_json() if c else None, indent=2))
    elif args.cmd == "confirm":
        print(json.dumps(confirm_emergence(args.instance, signer=args.signer), indent=2))
    elif args.cmd == "reject":
        reject_emergence(args.instance, signer=args.signer, reason=args.reason)
        print(json.dumps({"rejected": True, "instance": args.instance}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["KEY", "KEY_FILE_ENV", "KV_DECISION", "KV_EMERGENCE", "NoAttestationKey", "LOVE", "HELD",
           "attestations", "attest",
           "blessed", "configure", "decide", "checkpoint", "history", "rollback", "release",
           "emergence", "report_emergence", "clear_emergence", "observe_emergence",
           "emergence_candidate", "confirm_emergence", "reject_emergence", "GLYPH",
           "enable", "disable", "is_enabled"]
