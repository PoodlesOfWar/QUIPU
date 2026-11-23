"""hub_actions — the closed catalogue of actions QUIPU may ask Hub Holon (HubCore) to queue.

Everything an agent or a VNN rule can *want* from the hub is named here, in code.
There is no way to add an entry at runtime (HubCore Part 10.2: a synthetic party
cannot add a tool), and ``x_process_studio.propose`` refuses any ``hub.*`` tool that
is not in this table, with arguments that do not pass its validator.

An action does not *do* anything by itself.  It becomes a ``HubCore`` task only
after a human operator authorises the proposal (``x_process_studio.authorise``),
and HubCore then holds that task's result for a Reviewer to accept before any
promotion.  The catalogue names what may be asked, not what may happen.

``hub.body_directive`` mirrors a row of the Brain's ``body_directives`` table:
an operational directive for a human role or an AGV (title, why it matters, what
to do, owner role, severity, priority), with a fingerprint so a repeat is one
directive rather than a flood.  Stdlib only.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

OWNER_ROLES = ("Buyer", "Planner", "Quality", "IT", "Ops", "Anyone", "AGV")
SEVERITIES = ("info", "watch", "act", "critical")

_CONTROL = re.compile(r"[\x00-\x1f\x7f  ]+")
_FINGERPRINT = re.compile(r"^[A-Za-z0-9:._-]{1,120}$")
_MAX_EVIDENCE_BYTES = 4096

#: tool name -> description of the catalogue entry.
CATALOGUE: dict[str, dict[str, Any]] = {
    "hub.body_directive": {
        "title": "Body directive",
        "task_kind": "body.directive",
        "effect": ("Queues an operational directive for a human role or an AGV on Hub Floor as a "
                   "HubCore task. HubCore holds its result for a Reviewer to accept. It does not "
                   "move or switch anything by itself."),
        "fields": {
            "title": "one line, at most 140 characters (required)",
            "why_it_matters": "plain-language context, at most 600 characters",
            "do_this": "the specific action asked of the owner, at most 600 characters (required)",
            "owner_role": "one of " + ", ".join(OWNER_ROLES) + " (required)",
            "severity": "one of " + ", ".join(SEVERITIES) + " (default info)",
            "priority": "number 0..1 (default 0.5)",
            "fingerprint": "dedupe key, letters/digits and : . _ -, at most 120 characters",
            "target_entity": "what it concerns, at most 120 characters",
            "evidence": "object, at most 4 KB, pointers back to what triggered it",
        },
    },
}


def _line(value: Any, limit: int) -> str:
    text = " ".join(_CONTROL.sub(" ", "" if value is None else str(value)).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def is_hub_tool(tool: str) -> bool:
    return str(tool or "").startswith("hub.")


def catalogue_public() -> list[dict[str, Any]]:
    return [{"tool": name, **{k: v for k, v in spec.items()}} for name, spec in sorted(CATALOGUE.items())]


def validate(tool: str, arguments: Any) -> tuple[Optional[dict[str, Any]], str]:
    """Check a ``hub.*`` action.  Returns ``(clean_arguments, "")`` or ``(None, reason)``.

    Unknown tools and unknown argument names are refused, not dropped: nothing rides
    along that the operator did not see in the proposal.
    """
    spec = CATALOGUE.get(str(tool))
    if spec is None:
        return None, ("no such hub action " + repr(_line(tool, 48)) + "; the catalogue is fixed in code: "
                      + ", ".join(sorted(CATALOGUE)))
    if not isinstance(arguments, Mapping):
        return None, "arguments must be an object"
    if tool == "hub.body_directive":
        return _validate_body_directive(arguments)
    return None, "unsupported hub action"          # pragma: no cover - catalogue and validator move together


def _validate_body_directive(args: Mapping[str, Any]) -> tuple[Optional[dict[str, Any]], str]:
    allowed = set(CATALOGUE["hub.body_directive"]["fields"])
    extra = sorted(set(args) - allowed)
    if extra:
        return None, "unknown argument(s): " + ", ".join(_line(k, 40) for k in extra)
    title = _line(args.get("title"), 140)
    do_this = _line(args.get("do_this"), 600)
    if not title:
        return None, "title is required"
    if not do_this:
        return None, "do_this is required"
    role = str(args.get("owner_role") or "")
    if role not in OWNER_ROLES:
        return None, "owner_role must be one of " + ", ".join(OWNER_ROLES)
    severity = str(args.get("severity") or "info")
    if severity not in SEVERITIES:
        return None, "severity must be one of " + ", ".join(SEVERITIES)
    priority = args.get("priority", 0.5)
    if isinstance(priority, bool) or not isinstance(priority, (int, float)) or not 0.0 <= float(priority) <= 1.0:
        return None, "priority must be a number from 0 to 1"
    clean: dict[str, Any] = {
        "title": title,
        "why_it_matters": _line(args.get("why_it_matters"), 600),
        "do_this": do_this,
        "owner_role": role,
        "severity": severity,
        "priority": round(float(priority), 3),
    }
    if args.get("fingerprint"):
        fp = str(args["fingerprint"])
        if not _FINGERPRINT.match(fp):
            return None, "fingerprint must be 1-120 characters of letters, digits and : . _ -"
        clean["fingerprint"] = fp
    if args.get("target_entity"):
        clean["target_entity"] = _line(args["target_entity"], 120)
    if args.get("evidence") is not None:
        ev = args["evidence"]
        if not isinstance(ev, Mapping):
            return None, "evidence must be an object"
        try:
            blob = json.dumps(ev, default=str, sort_keys=True)
        except (TypeError, ValueError):
            return None, "evidence is not JSON-serialisable"
        if len(blob) > _MAX_EVIDENCE_BYTES:
            return None, f"evidence is over {_MAX_EVIDENCE_BYTES} bytes"
        clean["evidence"] = json.loads(blob)
    return clean, ""


def task_for(tool: str, clean: Mapping[str, Any], provenance: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """The HubCore task (kind, payload) an authorised ``hub.*`` proposal becomes.
    Provenance (who proposed, who authorised, under which grant) travels with it."""
    return CATALOGUE[tool]["task_kind"], {"directive": dict(clean), "provenance": dict(provenance)}
