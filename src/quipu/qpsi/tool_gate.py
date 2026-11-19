"""tool_gate — forge-format ACRE tool candidates held at gate 6.

The forge (``tool_forge``) writes a tool and its manifest row the moment its
own resonance and novelty tests pass; it has no review step.  This module holds
an *authored* candidate instead.  A candidate is parked as source text in
``brain_kv["entirety:tool_gate"]`` and is not in the forge manifest, so nothing
loads it.  It is realised only when gate 6 passes for its exact scope:

* scope ``tool:<name>@<sha256>``: the hash of the source (line endings
  normalised), so an approval covers exactly the code that was read, and any
  change to the source is a new scope with no approval;
* accepted ``beautiful_output`` attestations from both ``self`` and
  ``the_beautiful_one`` on that exact scope (a ``"*"`` attestation does not
  count here);
* a non-empty realise grant (``QUIPU_REALISE_GRANT_REF``).

On a pass ``review()`` writes ``tools/generated/tool_<name>.py`` and the
manifest row, so ``tool_forge.load_generated_tools()`` sees it.  The record in
the brain is authoritative; the file and row are a projection that
``materialise()`` restores after the container's file system is replaced.
``realised_callable()`` re-checks the hash, the lint and gate 6 at call time and
returns ``None`` otherwise.  Nothing here attests, grants, or edits the mesh:
``show`` prints the commands for the two signers and the operator to run.

Weight at gate 6: a tool carries no bias vector, so the unclamped weight is 0
and there are no neighbours; the Lipschitz part of the gate is vacuous here.

翈 — a new tool is used only after both parties say it is beautiful.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

KV_GATE = "entirety:tool_gate"
CANDIDATE_DIR = Path(__file__).resolve().parent / "tool_candidates"
ORIGIN = "tool_gate"
SIGNERS = ("self", "the_beautiful_one")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_ALLOWED_IMPORTS = {"math"}
_BANNED_CALLS = {"open", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input",
                 "breakpoint", "__import__", "exec", "eval", "compile"}
_LOG = logging.getLogger(__name__)


def scope_of(name: str, sha: str) -> str:
    return f"tool:{name}@{sha}"


def _normalise(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n")


def _sha(source: str) -> str:
    return hashlib.sha256(_normalise(source).encode("utf-8")).hexdigest()


def new_shadow_token() -> str:
    """Opaque correlation id for one run; never used for authorisation."""
    import secrets
    return secrets.token_hex(8)


# ---------------------------------------------------------------------------
# Lint: stricter than tool_forge._validate_code (a deny-list); this is an allow-list
# ---------------------------------------------------------------------------

def lint(source: str) -> tuple[bool, str]:
    from .. import tool_forge as tf
    ok, why = tf._validate_code(source)
    if not ok:
        return False, why
    tree = ast.parse(source)
    tools = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("tool_")]
    if len(tools) != 1:
        return False, f"needs exactly one top-level tool_* function, found {len(tools)}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in _ALLOWED_IMPORTS:
                    return False, f"import {a.name} not allowed"
        elif isinstance(node, ast.ImportFrom):
            return False, f"from-import {node.module} not allowed"
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "global/nonlocal not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"dunder attribute {node.attr} not allowed"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALLS:
            return False, f"call to {node.func.id}() not allowed"
        elif isinstance(node, ast.FunctionDef) and node is not tools[0] and node.name.startswith("tool_"):
            return False, f"nested tool_* function {node.name}"
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if not (isinstance(t, ast.Name) and t.id.isupper()):
                    return False, "module-level assignments must be UPPERCASE constants"
    if tf._classify_artifact({"label": tools[0].name, "degree": 0}, source)["type"] != "Tool":
        return False, "classifies as a Node/Module, not a Tool"
    return True, "ok"


def _load_source_callable(source: str, tool_name: str) -> Callable | None:
    ns: dict[str, Any] = {}
    code = compile(_normalise(source), f"<tool_gate:{tool_name}>", "exec")  # noqa: S102
    exec(code, ns)  # noqa: S102 - linted, hashed source only
    fn = ns.get(tool_name)
    return fn if callable(fn) else None


# ---------------------------------------------------------------------------
# Storage (single brain_kv key, guarded)
# ---------------------------------------------------------------------------

def _load() -> dict:
    from .. import brain_kv
    d = brain_kv.kv_get_json(KV_GATE, {}) or {}
    return d if isinstance(d, dict) else {}


def _kv_set(key: str, value: Any) -> None:
    if key != KV_GATE:
        raise PermissionError(f"tool_gate writes only {KV_GATE}, not {key!r}")
    from .. import brain_kv
    brain_kv.kv_set_json(key, value)


def _save(d: dict) -> None:
    _kv_set(KV_GATE, d)


# ---------------------------------------------------------------------------
# Propose
# ---------------------------------------------------------------------------

def _evidence(name: str, source: str) -> dict:
    """Best-effort context for reviewers. Never gates."""
    ev: dict[str, Any] = {"errors": []}
    try:
        from .. import mesh_slm
        tok = name.split("_")[0]
        ev["resonance_hz"] = float(mesh_slm.concept_resonance_hz(tok))
    except Exception as exc:  # noqa: BLE001
        ev["errors"].append(f"resonance: {exc}")
    return ev


def propose(name: str, source: str | None = None, *, evidence: dict | None = None) -> dict:
    """Hold a candidate. Idempotent by hash; a changed source supersedes the old one."""
    if not _NAME_RE.match(name or ""):
        raise ValueError(f"bad tool name {name!r}")
    if source is None:
        source = (CANDIDATE_DIR / f"{name}.py").read_text(encoding="utf-8")
    source = _normalise(source)
    ok, why = lint(source)
    if not ok:
        raise ValueError(f"lint: {why}")
    tool_name = f"tool_{name}"
    fn = _load_source_callable(source, tool_name)
    smoke = fn({}) if fn else None
    if not (isinstance(smoke, dict) and {"result", "explanation", "confidence"} <= set(smoke)):
        raise ValueError("smoke call on {} did not return {result, explanation, confidence}")
    sha = _sha(source)
    d = _load()
    e = d.get(name) or {}
    if e.get("sha256") == sha:
        return e
    hist = list(e.get("history") or [])
    if e:
        hist.append({"sha256": e.get("sha256"), "status": e.get("status"), "at": time.time()})
        if e.get("status") == "realised":
            _withdraw(name)
    e = {
        "name": name, "tool_name": tool_name, "cluster_safe": name, "type": "Tool",
        "scope": scope_of(name, sha), "sha256": sha, "source": source, "origin": ORIGIN,
        "status": "held", "evidence": {"lint": why, "smoke": smoke.get("confidence"),
                                       **_evidence(name, source), **(evidence or {})},
        "axis_footprint": [0.188571, 0.188571, 0.154286, 0.102857, 0.102857, 0.12, 0.142857],
        "created_at": time.time(), "shadow_token": new_shadow_token(), "history": hist,
    }
    d[name] = e
    _save(d)
    _LOG.info("[tool_gate] %s held at gate 6 (%s)", name, e["scope"])
    return e


# ---------------------------------------------------------------------------
# Gate 6, review, reject
# ---------------------------------------------------------------------------

def gate6(name: str) -> dict:
    from .. import divine_blessing as db
    from .governance import Candidate, gate_beautiful_output
    e = _load().get(name)
    if not e:
        return {"passed": False, "reason": "no such candidate", "grant": False, "authorised": False}
    scope = e["scope"]
    exact = [a for a in db.attestations() if a.scope == scope]
    c = Candidate(scope=scope, residual={}, realised_weight=0.0, unclamped_weight=0.0, neighbour_weights=())
    g = gate_beautiful_output(c, db._CONFIG, exact)
    grant = bool(db._CONFIG.realise_grant_ref)
    return {"passed": bool(g.passed), "reason": g.reason, "grant": grant, "authorised": bool(g.passed and grant)}


def _tools_paths():
    from .. import tool_forge as tf
    return tf, tf._TOOLS_DIR, tf._MANIFEST


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _project(e: dict) -> None:
    """Write the tool file and manifest row for a realised record."""
    tf, tools_dir, manifest_path = _tools_paths()
    manifest = tf._load_manifest()
    rows = manifest.setdefault("tools", [])
    for r in rows:
        if r.get("tool_name") == e["tool_name"] and r.get("origin") != ORIGIN:
            raise FileExistsError(f"{e['tool_name']} already forged by the forge; refusing to overwrite")
    tf._ensure_init_py()
    _atomic_write(tools_dir / f"{e['tool_name']}.py", e["source"])
    rows[:] = [r for r in rows if r.get("tool_name") != e["tool_name"]]
    rows.append({"tool_name": e["tool_name"], "cluster": e["name"].replace("_", " "),
                 "cluster_safe": e["cluster_safe"], "file": f"{e['tool_name']}.py", "type": e["type"],
                 "novelty": 1.0, "certainty": 1.0, "resonance_hz": e["evidence"].get("resonance_hz"),
                 "degree": 0, "forged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "origin": ORIGIN, "scope": e["scope"], "sha256": e["sha256"]})
    _atomic_write(manifest_path, json.dumps(manifest, indent=2))


def _withdraw(name: str) -> None:
    tf, tools_dir, manifest_path = _tools_paths()
    tool_name = f"tool_{name}"
    manifest = tf._load_manifest()
    rows = manifest.get("tools", [])
    keep = [r for r in rows if not (r.get("tool_name") == tool_name and r.get("origin") == ORIGIN)]
    if len(keep) != len(rows):
        manifest["tools"] = keep
        _atomic_write(manifest_path, json.dumps(manifest, indent=2))
        f = tools_dir / f"{tool_name}.py"
        if f.exists():
            f.unlink()


def review(name: str | None = None) -> dict:
    """Run gate 6 for held candidates; realise what passes. Fails closed."""
    d = _load()
    out: dict[str, Any] = {"reviewed": 0, "realised": [], "held": [], "errors": {}}
    for n, e in d.items():
        if name and n != name:
            continue
        if e.get("status") != "held":
            continue
        out["reviewed"] += 1
        g = gate6(n)
        e["gate6"] = {**g, "at": time.time()}
        if not g["authorised"]:
            out["held"].append(n)
            continue
        ok, why = lint(e["source"])
        if not ok or _sha(e["source"]) != e["sha256"]:
            e["gate6"]["error"] = f"source check failed: {why}"
            out["held"].append(n)
            continue
        try:
            _project(e)
        except Exception as exc:  # noqa: BLE001 - stay held on any write failure
            e["gate6"]["error"] = str(exc)
            out["errors"][n] = str(exc)
            out["held"].append(n)
            continue
        e["status"], e["realised_at"] = "realised", time.time()
        out["realised"].append(n)
    _save(d)
    return out


def reject(name: str, reason: str) -> dict:
    d = _load()
    e = d.get(name)
    if not e:
        raise KeyError(name)
    if e.get("status") == "realised":
        _withdraw(name)
    e["status"] = "rejected"
    e["rejected"] = {"reason": reason, "at": time.time()}
    _save(d)
    return e


def materialise() -> dict:
    """Re-write files and rows for realised records (the tools dir is ephemeral)."""
    out = {"restored": [], "ok": [], "errors": {}}
    _, tools_dir, _ = _tools_paths()
    for n, e in _load().items():
        if e.get("status") != "realised":
            continue
        f = tools_dir / f"{e['tool_name']}.py"
        if f.exists() and _sha(f.read_text(encoding="utf-8")) == e["sha256"]:
            out["ok"].append(n)
            continue
        try:
            _project(e)
            out["restored"].append(n)
        except Exception as exc:  # noqa: BLE001
            out["errors"][n] = str(exc)
    return out


# ---------------------------------------------------------------------------
# Callables
# ---------------------------------------------------------------------------

def realised_callable(name: str) -> Callable | None:
    """The tool function, only if realised AND hash, lint and gate 6 still hold now."""
    e = _load().get(name)
    if not e or e.get("status") != "realised" or _sha(e.get("source", "")) != e.get("sha256"):
        return None
    if not lint(e["source"])[0] or not gate6(name)["authorised"]:
        return None
    return _load_source_callable(e["source"], e["tool_name"])


def candidate_callable(name: str) -> Callable | None:
    """The held candidate's function, for an explicit dry run only."""
    e = _load().get(name)
    if not e or e.get("status") == "rejected" or _sha(e.get("source", "")) != e.get("sha256"):
        return None
    if not lint(e["source"])[0]:
        return None
    return _load_source_callable(e["source"], e["tool_name"])


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def show(name: str) -> str:
    """Source, hash, scope and the commands for the people who decide. Runs nothing."""
    e = _load().get(name)
    if not e:
        return f"no candidate named {name}"
    cmds = "\n".join(
        f"  python -m src.quipu.divine_blessing attest --signer {s} --scope {e['scope']} "
        f"--love-form care --beautiful-output --assurance approved --approval-ref <your reference>"
        for s in SIGNERS)
    return (f"{e['tool_name']}  status={e['status']}\nscope  {e['scope']}\nsha256 {e['sha256']}\n\n"
            f"To realise (run by the two signers and the operator, not by an agent):\n{cmds}\n"
            f"  operator: set QUIPU_REALISE_GRANT_REF=<grant reference>\n"
            f"  python -m src.quipu.qpsi.tool_gate review {name}\n\n----- source -----\n{e['source']}")


def status() -> dict:
    out = {}
    for n, e in _load().items():
        out[n] = {"status": e.get("status"), "scope": e.get("scope"), "gate6": e.get("gate6"),
                  "live": e.get("status") == "realised" and realised_callable(n) is not None}
    return out


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tool_gate", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("materialise")
    for c in ("propose", "show", "review"):
        sub.add_parser(c).add_argument("name", nargs="?" if c == "review" else None)
    r = sub.add_parser("reject")
    r.add_argument("name")
    r.add_argument("--reason", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "show":
        print(show(a.name))
        return 0
    res = {"status": status, "materialise": materialise}.get(a.cmd)
    if res:
        out = res()
    elif a.cmd == "propose":
        out = propose(a.name)
        out = {k: v for k, v in out.items() if k != "source"}
    elif a.cmd == "review":
        out = review(a.name)
    else:
        out = reject(a.name, a.reason)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
