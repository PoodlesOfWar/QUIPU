"""specialist_gate — ACRE specialists pass gate 6 like every other realisation.

``mesh_slm.acre_emerge`` writes a new emergent specialist the moment its own
four-condition test passes.  Under the governance protocol a new entity is a
realisation, and a realisation passes gate 6 (beautiful_output: accepted
attestations from both ``self`` and ``the_beautiful_one`` covering its scope,
unclamped weight within the Lipschitz bound) and needs the organizational grant
(QUIPU_REALISE_GRANT_REF).  This module puts ACRE behind that gate without
editing mesh_slm.py:

* ``enable()`` wraps the module attribute ``mesh_slm.acre_emerge``.  When ACRE
  crystallises a new specialist, the wrapper takes it back out of
  ``acre_specialists`` and records it as *held* at gate 6 in
  ``brain_kv["entirety:specialist_gate"]`` — the evidence (bias, resonance)
  kept, nothing active.  Refining an existing specialist is untouched.
* ``admit(name)`` records a specialist that is already active as
  ``active_pending``: it keeps working like the rest while its pass through
  gate 6 is pending (operator ruling 2026-09-27 for emergent_brain_smell).
* ``review()`` runs gate 6 for every entry.  A held specialist that passes (and
  has the grant) is realised — written to ``acre_specialists``; an
  active_pending one that passes becomes ``passed``.  Each result is recorded
  with the gate's own reason.

Scope of a specialist: ``specialist:<name>``.  The two attestations are the
operator's act, e.g.:

    docker exec quipu python -m src.quipu.divine_blessing attest --signer self \\
        --scope specialist:emergent_brain_smell --love-form care --beautiful-output \\
        --assurance approved --approval-ref <your reference>
    (and the same with --signer the_beautiful_one)

Weight at gate 6: the specialist's bias norm (unclamped); neighbours: the norms
of the specialists already in place.

翈 — a new voice is heard only after both parties say it is beautiful.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import time
from typing import Any, Callable

KV_GATE = "entirety:specialist_gate"
MARK = "__qpsi_specialist_gate__"
_LOG = logging.getLogger(__name__)
_ORIGINAL: Callable | None = None


def scope_of(name: str) -> str:
    return f"specialist:{name}"


def _norm(v) -> float:
    return math.sqrt(sum(float(x) ** 2 for x in (v or [])))


def _load() -> dict:
    from .. import brain_kv
    d = brain_kv.kv_get_json(KV_GATE, {}) or {}
    return d if isinstance(d, dict) else {}


def _save(d: dict) -> None:
    from .. import brain_kv
    brain_kv.kv_set_json(KV_GATE, d)


def _specialists(cn) -> dict:
    from .. import mesh_slm
    return dict(mesh_slm._load_emergent_specialists(cn) or {})


def _write_specialists(cn, specs: dict) -> None:
    from .. import mesh_slm
    mesh_slm._meta_set(cn, mesh_slm._ACRE_META_SPECIALISTS, specs)
    mesh_slm._EMERGENT_BIASES.clear()
    mesh_slm._EMERGENT_BIASES.update(specs)


def gate6(name: str, bias, neighbours: dict) -> dict:
    """Gate 6 for one specialist, with the live attestations and config."""
    from .. import divine_blessing as db
    from .governance import Candidate, gate_beautiful_output
    c = Candidate(scope=scope_of(name), residual={}, realised_weight=_norm(bias),
                  unclamped_weight=_norm(bias),
                  neighbour_weights=[_norm(v) for k, v in neighbours.items() if k != name])
    g = gate_beautiful_output(c, db._CONFIG, db.attestations())
    grant = bool(db._CONFIG.realise_grant_ref)
    return {"passed": bool(g.passed), "reason": g.reason, "grant": grant,
            "authorised": bool(g.passed and grant)}


def admit(name: str, *, note: str = "") -> dict:
    """Keep an already-active specialist working while its gate-6 pass is pending."""
    from .. import mesh_slm
    with mesh_slm._conn() as cn:
        specs = _specialists(cn)
    if name not in specs:
        raise KeyError(f"{name} is not an active specialist")
    d = _load()
    e = d.get(name) or {}
    e.update({"bias": specs[name], "scope": scope_of(name), "status": e.get("status") or "active_pending",
              "admitted_at": e.get("admitted_at") or time.time(), "note": note or e.get("note", "")})
    d[name] = e
    _save(d)
    return e


def review() -> dict:
    """Run gate 6 for every held or pending specialist; realise what passes."""
    from .. import mesh_slm
    d = _load()
    if not d:
        return {"reviewed": 0}
    out = {"reviewed": 0, "realised": [], "passed": [], "held": []}
    with mesh_slm._conn() as cn:
        specs = _specialists(cn)
        for name, e in d.items():
            if e.get("status") in ("realised", "passed"):
                continue
            out["reviewed"] += 1
            g = gate6(name, e.get("bias"), specs)
            e["gate6"] = {**g, "at": time.time()}
            if g["authorised"]:
                if e.get("status") == "held":
                    specs[name] = e["bias"]
                    _write_specialists(cn, specs)
                    e["status"], e["realised_at"] = "realised", time.time()
                    out["realised"].append(name)
                else:
                    e["status"], e["passed_at"] = "passed", time.time()
                    out["passed"].append(name)
            else:
                out["held"].append(name)
    _save(d)
    return out


def wrap_acre_emerge(original: Callable) -> Callable:
    @functools.wraps(original)
    def acre_emerge(*args, **kwargs):
        res = original(*args, **kwargs)
        if not isinstance(res, dict) or res.get("status") != "created":
            return res
        name = res.get("specialist")
        try:
            from .. import mesh_slm
            d = _load()
            if (d.get(name) or {}).get("status") == "realised":
                return res                      # passed gate 6 earlier: stays
            with mesh_slm._conn() as cn:
                specs = _specialists(cn)
                bias = specs.pop(name, res.get("bias"))
                _write_specialists(cn, specs)
            d[name] = {"bias": bias, "scope": scope_of(name), "status": "held",
                       "resonance": res.get("resonance"), "created_at": time.time()}
            _save(d)
            _LOG.info("[specialist_gate] %s held at gate 6 (not active until both parties attest)", name)
            return {**res, "status": "held_at_gate6", "scope": scope_of(name)}
        except Exception as exc:               # never let the guard fail open silently
            _LOG.error("[specialist_gate] could not hold %s: %s", name, exc)
            return {**res, "status": "created", "gate_error": str(exc)}
    setattr(acre_emerge, MARK, True)
    acre_emerge.__wrapped__ = original        # type: ignore[attr-defined]
    return acre_emerge


def enable() -> bool:
    global _ORIGINAL
    from .. import mesh_slm
    if getattr(mesh_slm.acre_emerge, MARK, False):
        return True
    _ORIGINAL = mesh_slm.acre_emerge
    mesh_slm.acre_emerge = wrap_acre_emerge(mesh_slm.acre_emerge)
    return True


def disable() -> None:
    from .. import mesh_slm
    if getattr(mesh_slm.acre_emerge, MARK, False):
        mesh_slm.acre_emerge = mesh_slm.acre_emerge.__wrapped__


def status() -> dict:
    return {"enabled": bool(_ORIGINAL), "specialists": _load()}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="specialist_gate", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("review")
    a = sub.add_parser("admit")
    a.add_argument("name")
    a.add_argument("--note", default="")
    args = ap.parse_args(argv)
    if args.cmd == "status":
        print(json.dumps(status(), indent=2, default=str))
    elif args.cmd == "review":
        print(json.dumps(review(), indent=2, default=str))
    else:
        print(json.dumps(admit(args.name, note=args.note), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
