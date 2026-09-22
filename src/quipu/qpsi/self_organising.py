"""self_organising — wires the System Entirety into a closed, governed loop.

What "self-organising" means here, and what it does not
--------------------------------------------------------
A self-organising memristive network (Nat. Rev. Phys. 2026, arXiv:2509.00747)
is self-organising, not self-driving: it reconfigures under a field that the
experimenter applies, along paths that conservation and thresholds select, and
it relaxes when the field is removed.  Nothing in it decides to be stimulated.
That is the boundary kept here.

    the field          ingest — applied by the operator's pulse, never by this code
    the phase          read from the field (qpsi.flux_phase), not from a clock
    the organisation   conductance on the seven axes, conserved budget, winner-take-all,
                       volatility, mobility, depression (qpsi.memristive_axes)
    the memory         the prior ŵ, advanced only by realised displacement
                       (qpsi.learned_prior)
    the gates          untouched: DIVINE_BLESSING_SQRT(-1) still decides every
                       edge; realisation still needs the grant (divine_blessing)

``enable()`` wraps three attributes of ``system_entirety`` at import time, the
way ``divine_blessing.enable()`` does, and ``disable()`` restores them:

    bit_flip_parity             → flux_phase.wrap_bit_flip_parity
    observer_tangent            → learned_prior.wrap_observer_tangent
    oscillating_expansion_step  → the original, then ``after_step``: one SOMN
                                  step and the prior's realisation check, on
                                  their own connection; the result gains a
                                  ``self_organising`` summary

Default off.  ``src/quipu/__init__.py`` calls ``enable()`` only when
``QUIPU_SELF_ORGANISING=1``.  With the flag unset the three attributes are the
original function objects and the live system is byte-identical to before.
Sub-flags ``QUIPU_FLUX_PHASE``, ``QUIPU_LEARNED_PRIOR``, ``QUIPU_SOMN`` (each
default on when the master is on) switch the three parts independently.

The operator's pulse
--------------------
``python -m src.quipu.qpsi.self_organising pulse`` prints the plan the network
recorded (documents per source, from the Kirchhoff partition of the budget).
With ``--route`` it runs ``corpus_ingest.run_ingest`` over exactly those
sources with exactly those counts, refinement off unless ``--refine`` is
given, then takes one expansion step so the flux registers.  Routing is an
explicit flag at the call site, never an ambient environment variable: the
correct gateway takes its authority from the caller (Anti-Inverse Contract).
The sources are ``corpus_ingest.SOURCES`` and nothing else; the pulse adds no
source, task kind, access or gate (Invariance #7, APP_RECREATION_3 §25).

The ruling (operator, 2026-09-22): the operator gave the access, so r-ADMIN has
enabled a *constrained gate* — allocating the granted budget among the granted
sources is not a widening.  The constraint is enforced by ``constrained_gate``
on every routed pulse (sources ⊆ grant, Σ documents ≤ budget); a plan outside
the grant is reported and not routed.  ``Register-Pulse.ps1`` may therefore
schedule ``Start-Pulse.ps1 -Route``.

Holds train (operator, 2026-09-22): a decision held at any gate trains the
three latent stores from its mirror image (``qpsi.mirror_training``) — never
the gate's inputs.  The SOMN reads the resulting mirror drive on its next step.

Stdlib only; nothing here edits mesh_slm.py or system_entirety.py.

翈 — the loop closes through the senses; authority does not loop at all.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from . import flux_phase, learned_prior, memristive_axes, mirror_training
from .memristive_axes import SomnConfig

ENV: str = "QUIPU_SELF_ORGANISING"
FLUX_ENV: str = "QUIPU_FLUX_PHASE"
PRIOR_ENV: str = "QUIPU_LEARNED_PRIOR"
SOMN_ENV: str = "QUIPU_SOMN"
MIRROR_ENV: str = "QUIPU_MIRROR_TRAINING"
PULSE_SOURCES_ENV: str = "QUIPU_PULSE_SOURCES"      # optional comma list narrowing the operator's grant

# Invariance #7 ruling (operator, 2026-09-22): allocating an operator-granted
# budget among operator-granted sources is not a widening of autonomy — the
# operator gave the access, so r-ADMIN operates a *constrained gate* inside it.
# The constraint is enforced, not assumed: a pulse routes only when its plan
# lies within the grant (sources ⊆ granted, documents ≤ budget).
RULING_INVARIANCE_7: str = ("2026-09-22 operator ruling: the operator gave the access, so r-ADMIN has "
                            "enabled a constrained gate — allocation within the granted budget and "
                            "sources is not a widening under Invariance #7")

_LOG = logging.getLogger(__name__)
_ORIGINALS: dict[str, Any] = {}
_ENABLED = False
MARK: str = "__qpsi_self_organising__"      # set on every wrapper this module installs


@dataclass(frozen=True)
class Flags:
    flux_phase: bool = True
    learned_prior: bool = True
    somn: bool = True
    mirror_training: bool = True

    @classmethod
    def from_env(cls) -> "Flags":
        def on(name: str) -> bool:
            return os.environ.get(name, "1").strip() != "0"
        return cls(flux_phase=on(FLUX_ENV), learned_prior=on(PRIOR_ENV), somn=on(SOMN_ENV),
                   mirror_training=on(MIRROR_ENV))


_FLAGS = Flags()


def master_switch_on() -> bool:
    return os.environ.get(ENV, "0").strip() == "1"


# ---------------------------------------------------------------------------
# The post-step hook
# ---------------------------------------------------------------------------

def _epoch(value) -> float:
    if value is None:
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (TypeError, ValueError):
        return time.time()


def after_step(summary: dict, *, flags: Flags | None = None, cfg: SomnConfig | None = None,
               now: float | None = None) -> dict:
    """Run the SOMN step and the prior's realisation check for one expansion step.

    Reads the axes and observer from the step's own summary, the flux from the
    ingest feed, and writes only entirety:somn:* / entirety_somn_log /
    entirety:prior.  Never raises into the caller.
    """
    from .. import system_entirety as se
    flags = _FLAGS if flags is None else flags
    now = _epoch(summary.get("ran_at")) if now is None else float(now)
    flux = flux_phase.read_flux(now=now)
    out: dict[str, Any] = {"flux": flux.to_json(), "step_phase": summary.get("expansion_phase")}
    axes = summary.get("axes") or {}
    observer = float(summary.get("observer") or 0.0)
    with se._conn() as cn:
        drive = mirror_training.mirror_drive(cn, "system_entirety") if flags.mirror_training else None
        if flags.somn:
            res = memristive_axes.step(cn, axes=axes, observer=observer, flux_on=flux.on,
                                       flux_docs=flux.docs, now=now, cfg=cfg or SomnConfig.from_env(),
                                       mirror_drive=drive)
            out["somn"] = {
                "phase": res["phase"], "dt": round(res["dt"], 3),
                "top_axis": res["metrics"]["top_axis"], "top_share": res["metrics"]["top_share"],
                "participation_ratio": res["metrics"]["participation_ratio"],
                "filaments": res["metrics"]["filaments"], "field_norm": res["metrics"]["field_norm"],
                "proposals": len(res["proposals"]),
                "docs_per_source": res["allocation"]["docs_per_source"],
                "dissipated": res["allocation"]["dissipated"],
                "events": [e.get("kind") for e in res["events"]],
                "steps": res["steps"],
            }
        if flags.learned_prior:
            adv = learned_prior.advance_on_realisation(cn, "system_entirety", now=now)
            out["prior_advanced"] = bool(adv)
            if adv:
                out["prior"] = adv["weights"]
        if flags.mirror_training:
            # The step's own decision (divine_blessing) and checkpoint were
            # written before this hook; a hold is trained from its mirror now.
            mt = mirror_training.train_from_hold(cn, "system_entirety", now=now)
            out["mirror"] = ({"seq": mt["seq"], "failed_at": mt["failed_at"], "kind": mt["kind"],
                              "g_im": mt["g_im"], "theta": mt["theta"], "holds_trained": mt["holds_trained"]}
                             if mt else None)
    return out


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def enable(flags: Flags | None = None) -> bool:
    """Wrap system_entirety's three attributes.  Idempotent."""
    global _ENABLED, _FLAGS
    if _ENABLED:
        return True
    from .. import system_entirety as se
    if any(getattr(getattr(se, n, None), MARK, False)
           for n in ("bit_flip_parity", "observer_tangent", "oscillating_expansion_step")):
        # Another copy of this module (e.g. run as __main__) already wrapped these
        # attributes.  Adopt its wrapping instead of stacking a second one.
        for n in ("bit_flip_parity", "observer_tangent", "oscillating_expansion_step"):
            fn = getattr(se, n)
            if getattr(fn, MARK, False):
                _ORIGINALS[f"se.{n}"] = fn.__wrapped__
        _ENABLED = True
        return True
    _FLAGS = Flags.from_env() if flags is None else flags

    if _FLAGS.flux_phase:
        _ORIGINALS["se.bit_flip_parity"] = se.bit_flip_parity
        se.bit_flip_parity = flux_phase.wrap_bit_flip_parity(se.bit_flip_parity)
        setattr(se.bit_flip_parity, MARK, True)

    if _FLAGS.learned_prior:
        _ORIGINALS["se.observer_tangent"] = se.observer_tangent
        se.observer_tangent = learned_prior.wrap_observer_tangent(se.observer_tangent)
        setattr(se.observer_tangent, MARK, True)

    if _FLAGS.somn or _FLAGS.learned_prior or _FLAGS.mirror_training:
        _ORIGINALS["se.oscillating_expansion_step"] = orig_step = se.oscillating_expansion_step

        @functools.wraps(orig_step)
        def _self_organising_step(*, force: bool = False):
            out = orig_step(force=force)
            if isinstance(out, dict) and not out.get("skipped"):
                try:
                    out["self_organising"] = after_step(out)
                except Exception as exc:          # the loop must never break the step
                    _LOG.warning("[self_organising] after_step failed: %s", exc)
                    out["self_organising"] = {"error": str(exc)}
            return out
        setattr(_self_organising_step, MARK, True)
        se.oscillating_expansion_step = _self_organising_step

    _ENABLED = True
    _LOG.info("self_organising enabled: flux_phase=%s learned_prior=%s somn=%s",
              _FLAGS.flux_phase, _FLAGS.learned_prior, _FLAGS.somn)
    return True


def disable() -> None:
    global _ENABLED
    if not _ENABLED:
        return
    from .. import system_entirety as se
    if "se.bit_flip_parity" in _ORIGINALS:
        se.bit_flip_parity = _ORIGINALS.pop("se.bit_flip_parity")
    if "se.observer_tangent" in _ORIGINALS:
        se.observer_tangent = _ORIGINALS.pop("se.observer_tangent")
    if "se.oscillating_expansion_step" in _ORIGINALS:
        se.oscillating_expansion_step = _ORIGINALS.pop("se.oscillating_expansion_step")
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def flags() -> Flags:
    return _FLAGS


# ---------------------------------------------------------------------------
# Status and the operator's pulse
# ---------------------------------------------------------------------------

def status() -> dict:
    from .. import brain_kv
    from .. import system_entirety as se
    with se._conn() as cn:
        cfg = SomnConfig.from_env()
        st = memristive_axes.load_state(cn, cfg, create=False)      # status is a pure read
        prior = learned_prior.stored_prior(cn)
        mirror = mirror_training.load_record(cn, "system_entirety")
    return {
        "enabled": _ENABLED, "master_switch": master_switch_on(), "flags": _FLAGS.__dict__,
        "flux": flux_phase.read_flux().to_json(),
        "somn": {"g": st.g, "m": st.m, "steps": st.steps, "last_t": st.last_t,
                 "potentiation_steps": st.potentiation_steps, "relaxation_steps": st.relaxation_steps},
        "allocation": brain_kv.kv_get_json(memristive_axes.KV_ALLOCATION, None),
        "proposal": brain_kv.kv_get_json(memristive_axes.KV_PROPOSAL, None),
        "prior": prior or {"weights": learned_prior.default_weights(), "realised_seen": 0, "source": "designer table"},
        "mirror": {"holds_trained": mirror["holds_trained"], "last_seq": mirror["last_seq"], "kind": mirror["kind"],
                   "prior": mirror["prior"], "drive": mirror["drive"],
                   "radam": {k: mirror["radam"].get(k) for k in ("t", "theta", "pressure")}},
        "grant": grant(),
        "config": cfg.to_json(),
    }


def grant() -> dict:
    """The operator's grant the constrained gate is measured against: the
    sources corpus_ingest already knows (narrowed by QUIPU_PULSE_SOURCES when
    set) and the budget.  Nothing here can add a source."""
    known = memristive_axes.enabled_sources()
    raw = os.environ.get(PULSE_SOURCES_ENV, "").strip()
    if raw:
        wanted = [k.strip() for k in raw.split(",") if k.strip()]
        sources = [k for k in known if k in wanted]
    else:
        sources = list(known)
    return {"sources": sources, "budget_docs": SomnConfig.from_env().budget_docs, "ruling": RULING_INVARIANCE_7}


def constrained_gate(plan_docs: Mapping[str, int], g: Mapping | None = None) -> dict:
    """Is this plan inside the grant?  sources ⊆ granted and Σ docs ≤ budget."""
    g = g or grant()
    granted = set(g["sources"])
    outside = sorted(k for k in plan_docs if k not in granted)
    total = sum(int(v) for v in plan_docs.values())
    within = not outside and total <= float(g["budget_docs"])
    return {"within_grant": within, "outside_sources": outside, "docs": total,
            "budget_docs": g["budget_docs"], "ruling": g["ruling"]}


def plan() -> dict:
    """The allocation the network recorded on its last step."""
    from .. import brain_kv
    alloc = brain_kv.kv_get_json(memristive_axes.KV_ALLOCATION, None) or {}
    return {"docs_per_source": dict(alloc.get("docs_per_source") or {}),
            "dissipated": dict(alloc.get("dissipated") or {}),
            "unrouted_sources": list(alloc.get("unrouted_sources") or []),
            "budget": alloc.get("budget"), "at": alloc.get("at"), "phase": alloc.get("phase"),
            "metrics": alloc.get("metrics")}


def pulse(*, route: bool = False, refine: bool = False, max_seconds: float = 0.0) -> dict:
    """Apply one pulse of the field along the recorded allocation.

    ``route=False`` returns the plan and does nothing else.  ``route=True`` runs
    corpus_ingest over the planned sources with the planned counts (refinement
    off unless ``refine``), then one forced expansion step so the flux is read.
    """
    p = plan()
    docs = {k: int(v) for k, v in p["docs_per_source"].items() if int(v) > 0}
    gate = constrained_gate(docs)
    result: dict[str, Any] = {"plan": p, "constrained_gate": gate, "routed": False, "runs": []}
    if not route:
        result["note"] = "plan only; pass --route to apply the field"
        return result
    if not gate["within_grant"]:
        result["note"] = "plan outside the operator's grant; not routed"
        return result
    from .. import corpus_ingest
    from .. import system_entirety as se
    known = set(corpus_ingest.SOURCES.keys()) & set(grant()["sources"])
    for key, n in docs.items():
        if key not in known:                   # cannot happen after constrained_gate(); refuse anyway
            result["runs"].append({"source": key, "skipped": "outside the grant"})
            continue
        res = corpus_ingest.run_ingest([key], docs_per_source=n, refine_every=(1 if refine else 0),
                                       max_seconds=max_seconds)
        result["runs"].append({"source": key, "requested": n,
                               "ingested": (res.get("per_source") or {}).get(key, {}).get("ingested"),
                               "elapsed_s": res.get("elapsed_s")})
    result["routed"] = True
    result["step"] = se.oscillating_expansion_step(force=True)
    return result


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="quipu.qpsi.self_organising", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="flags, flux, conductances, allocation, proposals, prior")
    sub.add_parser("step", help="one forced expansion step with the loop enabled")
    pl = sub.add_parser("pulse", help="print the plan; --route applies it")
    pl.add_argument("--route", action="store_true", help="run corpus_ingest along the plan (operator's act)")
    pl.add_argument("--refine", action="store_true", help="also run Ring-5 refinement per Weyl cycle")
    pl.add_argument("--max-seconds", type=float, default=0.0)
    args = ap.parse_args(argv)
    enable()
    if args.cmd == "status":
        print(json.dumps(status(), indent=2, default=str))
    elif args.cmd == "step":
        from .. import system_entirety as se
        print(json.dumps(se.oscillating_expansion_step(force=True), indent=2, default=str))
    elif args.cmd == "pulse":
        print(json.dumps(pulse(route=args.route, refine=args.refine, max_seconds=args.max_seconds),
                         indent=2, default=str))
    return 0


if __name__ == "__main__":
    # ``python -m src.quipu.qpsi.self_organising`` executes this file as __main__
    # while the package may already hold the canonical module (src/quipu/__init__
    # imports it under QUIPU_SELF_ORGANISING=1).  Dispatch to that copy so there
    # is exactly one set of wrappers and one _ENABLED flag.
    import importlib
    _canonical = importlib.import_module(f"{__package__}.self_organising") if __package__ else None
    raise SystemExit((_canonical._main if _canonical is not None else _main)())


__all__ = ["ENV", "FLUX_ENV", "PRIOR_ENV", "SOMN_ENV", "Flags", "master_switch_on", "after_step",
           "enable", "disable", "is_enabled", "flags", "status", "plan", "pulse"]
