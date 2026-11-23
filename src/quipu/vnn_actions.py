"""vnn_actions — translate the VNN's output into staged Hub Holon actions.

The Viscous Neural Network publishes a coupled report (``entirety:vnn:latest``:
physical-flow invariants, semantic-graph invariants, stability diagnostics).  This
module reads it against a set of **rules** and, where one fires, turns it into a
``hub.body_directive`` *proposal* on the x_ProcessStudio ledger.

A VNN output never executes anything by itself.  The path is:

    VNN report --rule--> staged proposal (v-ADMIN, held) --human authorises-->
        r-ADMIN realises --> HubCore task --Reviewer accepts--> Hub Floor acts

so the translation is direct in *content* (the directive carries the observed value,
the threshold and a pointer to the report) and governed in *authority*: HubCore's
spec is categorical that AI cannot acquire authority (Parts 0.3.10, 10.2), and
QUIPU's V10-SEC-010 makes admissibility not authorization.  Actions that move
physical things are where that matters most.

Rules
-----
``DEFAULT_RULES`` cover **solver health only** (the stability flags and the
incompressibility residual have objective thresholds in the diagnostics themselves).
They do not know your plant.  Rules that map the flow onto your equipment belong in
an operator-owned JSON file named by ``QUIPU_VNN_ACTION_RULES`` (read-only to QUIPU;
see docs/HOLON_HUB_WORKER.md for the format).  Every rule is validated at load by
rendering it and running ``hub_actions.validate``, so a rule that could never
produce a valid directive is refused rather than failing later; a rule may not
raise a directive above severity ``act``.

Not wired into the Entirety step (like ``relational_shear``): staging is an operator
act (CLI, or the Desk's button over ``POST /vnn/actions/stage``).  Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from . import hub_actions

SOURCE = "quipu-vnn"
KV_VNN_LATEST = "entirety:vnn:latest"        # = viscous_interaction.KV_VNN_LATEST (kept literal: no numpy import)
RULES_ENV = "QUIPU_VNN_ACTION_RULES"
MAX_RULES = 32
RULE_SEVERITIES = ("info", "watch", "act")   # the translator never raises "critical"
_OPS = (">", ">=", "<", "<=", "is_true", "is_false")
_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_PATH = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){0,4}$")
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_.]*)\}")

DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "id": "vnn.graph.unstable", "path": "vnn_diagnostics.stable", "op": "is_false",
        "directive": {
            "title": "VNN semantic-graph step is outside its stability bound",
            "why_it_matters": ("The graph convection-diffusion step exceeded its stability envelope "
                               "(graph CFL {vnn_diagnostics.graph_cfl}), so the semantic field it feeds "
                               "may be unreliable."),
            "do_this": ("Review the VNN settings (dt, nu_s, kappa_s, graph_cfl_max) and confirm the graph "
                        "is stable before relying on the VNN output."),
            "owner_role": "IT", "severity": "act", "priority": 0.8, "target_entity": "vnn::graph",
        },
    },
    {
        "id": "vnn.ns.divergence", "path": "physical.div_linf", "op": ">", "value": 0.001,
        "directive": {
            "title": "VNN velocity field is not divergence-free",
            "why_it_matters": ("Incompressibility is enforced by pressure projection; the largest "
                               "divergence is {physical.div_linf} against a tolerance of 0.001."),
            "do_this": "Check the pressure projection (Poisson solve iterations and tolerance) in the Navier-Stokes step.",
            "owner_role": "IT", "severity": "watch", "priority": 0.5, "target_entity": "vnn::navier_stokes",
        },
    },
    {
        "id": "vnn.ns.substeps", "path": "ns_diagnostics.stable_single_step", "op": "is_false",
        "directive": {
            "title": "VNN needed substeps to stay stable",
            "why_it_matters": ("A single Navier-Stokes step was outside its limits (CFL {ns_diagnostics.cfl_adv}, "
                               "Fourier {ns_diagnostics.fo_visc}); the solver substepped to compensate."),
            "do_this": ("No action is needed unless this persists; if it does, consider a smaller dt "
                        "(recommended substeps: {ns_diagnostics.recommended_substeps})."),
            "owner_role": "IT", "severity": "info", "priority": 0.3, "target_entity": "vnn::navier_stokes",
        },
    },
]


# --------------------------------------------------------------------------- rules
def _lookup(report: Mapping[str, Any], path: str) -> Any:
    node: Any = report
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)[:60]


def render(template: Any, report: Mapping[str, Any]) -> str:
    """Substitute ``{dotted.path}`` placeholders from the report; nothing else is evaluated."""
    return _PLACEHOLDER.sub(lambda m: _fmt(_lookup(report, m.group(1))), str(template or ""))


def validate_rules(rules: Any) -> tuple[Optional[list], str]:
    if not isinstance(rules, list) or not rules:
        return None, "rules must be a non-empty list"
    if len(rules) > MAX_RULES:
        return None, f"at most {MAX_RULES} rules"
    seen: set[str] = set()
    for r in rules:
        if not isinstance(r, Mapping):
            return None, "each rule must be an object"
        rid = str(r.get("id") or "")
        if not _ID.match(rid) or rid in seen:
            return None, f"rule id {rid!r} must be unique, 1-64 characters of letters, digits and . _ -"
        seen.add(rid)
        if not _PATH.match(str(r.get("path") or "")):
            return None, f"rule {rid}: path must be a dotted lower-case path into the report"
        op = r.get("op")
        if op not in _OPS:
            return None, f"rule {rid}: op must be one of {', '.join(_OPS)}"
        if op in (">", ">=", "<", "<="):
            v = r.get("value")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None, f"rule {rid}: value must be a number for {op}"
        d = r.get("directive")
        if not isinstance(d, Mapping):
            return None, f"rule {rid}: directive must be an object"
        if d.get("severity", "info") not in RULE_SEVERITIES:
            return None, f"rule {rid}: severity may not exceed 'act' (the translator never raises 'critical')"
        sample = {k: render(v, {}) if isinstance(v, str) else v for k, v in d.items()}
        _, err = hub_actions.validate("hub.body_directive", sample)
        if err:
            return None, f"rule {rid}: {err}"
    return list(rules), ""


def load_rules(path: str | None = None) -> list[dict[str, Any]]:
    """The operator's rules file when ``QUIPU_VNN_ACTION_RULES`` names one, else the defaults.
    A rules file that fails validation is an error, not a silent fallback to the defaults."""
    p = path if path is not None else os.environ.get(RULES_ENV, "").strip()
    if not p:
        return [dict(r) for r in DEFAULT_RULES]
    try:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {RULES_ENV} file {p}: {exc}") from exc
    rules, err = validate_rules(doc.get("rules") if isinstance(doc, Mapping) else doc)
    if rules is None:
        raise ValueError(f"{RULES_ENV} file {p}: {err}")
    return rules


# --------------------------------------------------------------------------- evaluation
def _fires(rule: Mapping[str, Any], observed: Any) -> bool:
    op = rule["op"]
    if op == "is_true":
        return observed is True
    if op == "is_false":
        return observed is False
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        return False                                   # a missing or non-numeric reading never fires a threshold
    t = float(rule["value"])
    return {">": observed > t, ">=": observed >= t, "<": observed < t, "<=": observed <= t}[op]


def latest_report() -> Optional[dict[str, Any]]:
    from . import brain_kv
    rep = brain_kv.kv_get_json(KV_VNN_LATEST, None)
    return rep if isinstance(rep, dict) else None


def evaluate(report: Mapping[str, Any], rules: list | None = None) -> list[dict[str, Any]]:
    """Every rule that fires on this report, with the directive it would stage (rendered and validated)."""
    out: list[dict[str, Any]] = []
    for rule in (rules if rules is not None else load_rules()):
        observed = _lookup(report, rule["path"])
        if not _fires(rule, observed):
            continue
        d = {k: (render(v, report) if isinstance(v, str) else v) for k, v in rule["directive"].items()}
        d["fingerprint"] = "vnn:" + rule["id"]
        d["evidence"] = {"report_at": report.get("at"), "path": rule["path"], "observed": observed,
                         "rule": rule["id"], **({"threshold": rule["value"], "op": rule["op"]} if "value" in rule
                                                else {"op": rule["op"]})}
        clean, err = hub_actions.validate("hub.body_directive", d)
        if clean is None:                              # a rendered value broke a limit: say so, stage nothing
            out.append({"rule": rule["id"], "error": err})
            continue
        out.append({"rule": rule["id"], "observed": observed, "arguments": clean,
                    "reason": f"VNN rule {rule['id']}: {rule['path']} = {_fmt(observed)}"})
    return out


def dry_run(report: Mapping[str, Any] | None = None, rules: list | None = None) -> dict[str, Any]:
    rep = report if report is not None else latest_report()
    if rep is None:
        return {"ok": True, "report": None, "triggered": [], "note": "no VNN report published yet"}
    return {"ok": True, "report_at": rep.get("at"), "triggered": evaluate(rep, rules)}


def stage(report: Mapping[str, Any] | None = None, rules: list | None = None) -> dict[str, Any]:
    """Stage a proposal for every firing rule, once: a rule whose directive is already waiting
    (PROPOSED or AUTHORISED) is skipped, so a persistent condition is one directive, not a flood."""
    from . import x_process_studio as ps
    rep = report if report is not None else latest_report()
    if rep is None:
        return {"ok": True, "staged": [], "skipped": [], "note": "no VNN report published yet"}
    staged, skipped = [], []
    for item in evaluate(rep, rules):
        if "error" in item:
            skipped.append({"rule": item["rule"], "why": item["error"]})
            continue
        fp = item["arguments"]["fingerprint"]
        open_one = ps.find_open(fp)
        if open_one:
            skipped.append({"rule": item["rule"], "why": "already open", "proposal_id": open_one})
            continue
        p = ps.propose(SOURCE, "hub.body_directive", item["arguments"], reason=item["reason"],
                       meta={"fingerprint": fp, "vnn": {"report_at": rep.get("at"), "rule": item["rule"]}})
        staged.append({"rule": item["rule"], "proposal_id": p.proposal_id})
    return {"ok": True, "staged": staged, "skipped": skipped}


def _main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vnn_actions", description="translate VNN output into staged hub actions")
    ap.add_argument("cmd", choices=("rules", "dry-run", "stage"))
    args = ap.parse_args(argv)
    try:
        out: Any = load_rules() if args.cmd == "rules" else (dry_run() if args.cmd == "dry-run" else stage())
    except ValueError as exc:
        print(f"vnn_actions: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":          # pragma: no cover
    sys.exit(_main())
