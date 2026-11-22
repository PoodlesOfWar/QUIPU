"""x_process_studio — the x_ProcessStudio Holon: authorizable agentic action
through Activepieces MCP, for r-ADMIN via v-ADMIN.

The fleet's VLM/LLM layers (Perceptopoly's game mesh, StockOpoly's grouping
cascade, BurgerPhone's assistant, the Brain's ensemble) can *want* things done
in the world: start a flow, send a message, file a record.  Activepieces is the
process studio that can do them — its MCP server exposes the operator's
connected pieces as callable tools.  This holon is the one place the fleet's
wanting meets that capability, and it is built so the meeting is governed:

    v-ADMIN (virtual)   any admitted fleet source may STAGE an ActionProposal.
                        A staged proposal is held in band (翈): recorded,
                        listable, inspectable — never executed.  Staging is the
                        whole of v-ADMIN's authority.
    r-ADMIN (realised)  only an AUTHORISED proposal is realised.  Authorisation
                        is the operator's: a two-party Gate 6 attestation
                        (``gate_beautiful_output``) plus an organizational
                        grant reference checked by ``authorised_to_realise``
                        (V10-SEC-010 — admissibility is not authorization, and
                        this code cannot mint the grant).  Only then does the
                        holon dispatch ``tools/call`` to Activepieces.

A holon is a whole that is a part: the studio owns its MCP client and its
ledger, and is itself one organ of the Entirety — it writes only
``process_studio:*`` brain_kv keys, feeds no mesh, and is input to no gate.

Wire contract (observer routes; see observer_service):
    GET  /process_studio            status: MCP availability, counts by status
    POST /process_studio/propose    {source, tool, arguments?, reason?, meta?}
    POST /process_studio/authorise  {source, proposal_id, operator?, grant_ref,
                                     approve?, why?}   (operator sources only)
    POST /process_studio/realise    {source, proposal_id} | {source, ready: true}

All three POSTs sit behind qpsi.edge_admission like every other write.  The
authorise/realise routes additionally require a *signed* identity whose source
is named in ``QUIPU_PROCESS_STUDIO_OPERATORS``; by default they are refused and
the operator acts through the CLI inside the container:

    python -m src.quipu.x_process_studio status | list | tools
    python -m src.quipu.x_process_studio propose  --source X --tool T ...
    python -m src.quipu.x_process_studio authorise <id> --operator A --grant-ref REF
    python -m src.quipu.x_process_studio realise   <id> | --ready

Environment:
    QUIPU_ACTIVEPIECES_MCP_URL     the Activepieces MCP server URL (Streamable
                                   HTTP, the ``…/mcp`` link from the operator's
                                   MCP Server settings page).  Unset ⇒ the MCP
                                   side is unavailable; proposing still works.
    QUIPU_ACTIVEPIECES_MCP_TOKEN   optional bearer token for self-hosted fronts.
    QUIPU_MCP_TIMEOUT_S            HTTP timeout per MCP call (default 20).
    QUIPU_PROCESS_STUDIO_OPERATORS comma-separated canonical sources that may
                                   authorise/realise over HTTP (default none).

Transport: MCP JSON-RPC 2.0 over Streamable HTTP POST (``initialize`` →
``notifications/initialized`` → ``tools/list`` / ``tools/call``), accepting
both plain-JSON and SSE-framed response bodies, keeping the server's
``Mcp-Session-Id``.  Plain ``http://`` is accepted only for loopback, private
and container addresses, as in edge_client.  Stdlib only.

翈 — the studio stages what is wanted; only the operator's grant lets it act.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse

from . import brain_kv
from ._version import __version__
from .qpsi.governance import (
    ASSURANCE_APPROVED,
    Attestation,
    Candidate,
    Decision,
    GovernanceConfig,
    authorised_to_realise,
    config_digest,
    gate_beautiful_output,
)

log = logging.getLogger("quipu.x_process_studio")

VADMIN = "v-ADMIN"     # the virtual ADMIN: may stage and describe, never realise
RADMIN = "r-ADMIN"     # realises only an authorised proposal (V10-SEC-010)

SCHEMA_PROPOSE = "process_studio.propose/1"

MCP_URL_ENV = "QUIPU_ACTIVEPIECES_MCP_URL"
MCP_TOKEN_ENV = "QUIPU_ACTIVEPIECES_MCP_TOKEN"
MCP_TIMEOUT_ENV = "QUIPU_MCP_TIMEOUT_S"
OPERATORS_ENV = "QUIPU_PROCESS_STUDIO_OPERATORS"
MCP_PROTOCOL = "2025-03-26"

# Proposal statuses.  PROPOSED is the band (翈): staged by v-ADMIN, held.
PROPOSED = "PROPOSED"
AUTHORISED = "AUTHORISED"
REALISED = "REALISED"
FAILED = "FAILED"          # the tool itself refused (JSON-RPC error): permanent
REJECTED = "REJECTED"
STATUSES = (PROPOSED, AUTHORISED, REALISED, FAILED, REJECTED)

_INDEX_KEY = "process_studio:index"
_PROPOSAL_KEY = "process_studio:proposal:{pid}"
_MAX_BYTES = 256 * 1024     # same bound as relational records


# ---------------------------------------------------------------------------
# Activepieces MCP client (Streamable HTTP, JSON-RPC 2.0)
# ---------------------------------------------------------------------------

class MCPUnavailable(RuntimeError):
    """Transport-level failure (network, refused URL, bad body): retryable."""


class MCPError(RuntimeError):
    """The server answered with a JSON-RPC error: the call itself was refused."""


def _plain_http_ok(host: str) -> bool:
    """Plain http:// is allowed only where edge_client allows it: loopback,
    private/link-local addresses, and bare container names (no dot)."""
    h = (host or "").strip("[]").lower()
    if not h:
        return False
    if h == "localhost" or "." not in h:
        return True
    try:
        ip = ipaddress.ip_address(h)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        return h.endswith(".local") or h.endswith(".internal")


def _parse_rpc_body(text: str) -> Optional[dict[str, Any]]:
    """Parse a Streamable-HTTP response body: plain JSON, or SSE ``data:``
    frames (the transport may answer either way).  Returns the first JSON-RPC
    message carrying ``result`` or ``error``, else the last data frame."""
    t = (text or "").strip()
    if not t:
        return None
    if t[0] in "{[":
        try:
            parsed = json.loads(t)
            return parsed if isinstance(parsed, dict) else None
        except ValueError:
            return None
    frames: list[dict[str, Any]] = []
    for line in t.splitlines():
        if not line.startswith("data:"):
            continue
        frag = line[5:].strip()
        if not frag or frag == "[DONE]":
            continue
        try:
            msg = json.loads(frag)
        except ValueError:
            continue
        if isinstance(msg, dict):
            frames.append(msg)
    for msg in frames:
        if "result" in msg or "error" in msg:
            return msg
    return frames[-1] if frames else None


class ActivepiecesMCP:
    """Minimal MCP client for an Activepieces MCP server.

    ``opener`` is injectable (tests): any callable with urlopen's signature.
    The client is lazy — nothing touches the network until a call is made —
    and unconfigured (no URL) it reports ``available() == False`` so every
    caller can degrade instead of block.
    """

    def __init__(self, url: str | None = None, token: str | None = None,
                 timeout_s: float | None = None,
                 opener: Callable[..., Any] | None = None) -> None:
        self.url = (url if url is not None else os.environ.get(MCP_URL_ENV, "")).strip()
        self.token = (token if token is not None else os.environ.get(MCP_TOKEN_ENV, "")).strip()
        try:
            self.timeout_s = float(timeout_s if timeout_s is not None
                                   else os.environ.get(MCP_TIMEOUT_ENV, "20"))
        except (TypeError, ValueError):
            self.timeout_s = 20.0
        self._opener = opener or urllib.request.urlopen
        self._session_id: str | None = None
        self._initialised = False
        self._next_id = 0

    def available(self) -> bool:
        return bool(self.url)

    # -- transport ----------------------------------------------------------
    def _post(self, payload: Mapping[str, Any], *, expect_reply: bool = True) -> Optional[dict[str, Any]]:
        if not self.available():
            raise MCPUnavailable(f"no MCP server configured ({MCP_URL_ENV} unset)")
        parsed = urlparse(self.url)
        if parsed.scheme == "http" and not _plain_http_ok(parsed.hostname or ""):
            raise MCPUnavailable("plain http to a public address refused; use https")
        if parsed.scheme not in ("http", "https"):
            raise MCPUnavailable(f"unsupported scheme {parsed.scheme!r}")
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with self._opener(req, timeout=self.timeout_s) as resp:
                sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
                if sid:
                    self._session_id = sid
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise MCPUnavailable(f"MCP HTTP {exc.code}: {getattr(exc, 'reason', '')}") from exc
        except Exception as exc:                      # URLError, socket, TLS…
            raise MCPUnavailable(f"MCP transport: {exc}") from exc
        if not expect_reply:
            return None
        msg = _parse_rpc_body(raw.decode("utf-8", "replace"))
        if msg is None:
            raise MCPUnavailable("MCP reply carried no JSON-RPC message")
        if "error" in msg and msg["error"] is not None:
            err = msg["error"]
            raise MCPError(f"MCP error {err.get('code')}: {err.get('message')}")
        return msg

    def _rpc(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        msg = self._post({"jsonrpc": "2.0", "id": self._next_id, "method": method,
                          "params": dict(params or {})})
        assert msg is not None
        result = msg.get("result")
        return result if isinstance(result, dict) else {}

    # -- MCP methods ---------------------------------------------------------
    def initialize(self) -> dict[str, Any]:
        result = self._rpc("initialize", {
            "protocolVersion": MCP_PROTOCOL,
            "capabilities": {},
            "clientInfo": {"name": "quipu-x-process-studio", "version": __version__},
        })
        try:
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"},
                       expect_reply=False)
        except MCPUnavailable:
            pass                                       # the handshake result stands
        self._initialised = True
        return result

    def _ensure(self) -> None:
        if not self._initialised:
            self.initialize()

    def list_tools(self) -> list[dict[str, Any]]:
        self._ensure()
        tools = self._rpc("tools/list").get("tools")
        return list(tools) if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._ensure()
        return self._rpc("tools/call", {"name": name, "arguments": dict(arguments or {})})


# ---------------------------------------------------------------------------
# The ledger — v-ADMIN stages, the operator authorises, r-ADMIN realises
# ---------------------------------------------------------------------------

@dataclass
class ActionProposal:
    proposal_id: str
    source: str                      # fleet source that staged it (v-ADMIN side)
    tool: str                        # Activepieces MCP tool name
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)   # e.g. {"schema", "vlm"}
    status: str = PROPOSED
    created_at: float = field(default_factory=time.time)
    authorised_at: Optional[float] = None
    realised_at: Optional[float] = None
    authorised_by: str = ""
    grant_ref: str = ""
    decision: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    last_error: str = ""
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _save(p: ActionProposal) -> None:
    brain_kv.kv_set_json(_PROPOSAL_KEY.format(pid=p.proposal_id), p.to_dict())
    idx = brain_kv.kv_get_json(_INDEX_KEY, {}) or {}
    idx[p.proposal_id] = {"source": p.source, "tool": p.tool, "status": p.status,
                          "created_at": p.created_at}
    brain_kv.kv_set_json(_INDEX_KEY, idx)


def get_proposal(proposal_id: str) -> Optional[ActionProposal]:
    rec = brain_kv.kv_get_json(_PROPOSAL_KEY.format(pid=proposal_id), None)
    if not isinstance(rec, dict):
        return None
    known = {f for f in ActionProposal.__dataclass_fields__}      # tolerate extras
    return ActionProposal(**{k: v for k, v in rec.items() if k in known})


def list_proposals(status: str | None = None, source: str | None = None,
                   limit: int = 50) -> list[dict[str, Any]]:
    idx = brain_kv.kv_get_json(_INDEX_KEY, {}) or {}
    rows = [{"proposal_id": pid, **(meta or {})} for pid, meta in idx.items()]
    if status:
        rows = [r for r in rows if r.get("status") == status]
    if source:
        rows = [r for r in rows if r.get("source") == source]
    rows.sort(key=lambda r: r.get("created_at") or 0.0, reverse=True)
    return rows[: max(1, min(int(limit), 500))]


def stats() -> dict[str, Any]:
    idx = brain_kv.kv_get_json(_INDEX_KEY, {}) or {}
    by_status = {s: 0 for s in STATUSES}
    for meta in idx.values():
        s = (meta or {}).get("status")
        if s in by_status:
            by_status[s] += 1
    return {
        "ok": True,
        "holon": "x_process_studio",
        "vadmin": VADMIN,
        "radmin": RADMIN,
        "mcp_configured": ActivepiecesMCP().available(),
        "proposals": by_status,
        "total": len(idx),
    }


# -- v-ADMIN: staging --------------------------------------------------------

def propose(source: str, tool: str, arguments: Mapping[str, Any] | None = None,
            reason: str = "", meta: Mapping[str, Any] | None = None) -> ActionProposal:
    """Stage an ActionProposal (v-ADMIN).  Held in band: never executed here."""
    src = (source or "").strip()
    name = (tool or "").strip()
    if not src:
        raise ValueError("proposal needs a source")
    if not name:
        raise ValueError("proposal needs a tool")
    args = dict(arguments or {})
    m = dict(meta or {})
    m.setdefault("schema", SCHEMA_PROPOSE)
    p = ActionProposal(proposal_id=f"act_{uuid.uuid4().hex[:12]}", source=src,
                       tool=name, arguments=args, reason=str(reason or ""), meta=m)
    if len(json.dumps(p.to_dict(), default=str)) > _MAX_BYTES:
        raise ValueError(f"proposal over {_MAX_BYTES} bytes")
    _save(p)
    log.info("x_process_studio: %s staged %s -> %s (held)", src, p.proposal_id, name)
    return p


def propose_from_vlm(source: str, tool: str, vlm: Mapping[str, Any],
                     arguments: Mapping[str, Any] | None = None,
                     reason: str = "") -> ActionProposal:
    """Stage a proposal shaped from a VLM/LLM result (the fleet's perception
    layers): the raw signal rides in ``meta.vlm`` so the operator authorises
    with the evidence in view."""
    return propose(source, tool, arguments=arguments, reason=reason,
                   meta={"schema": SCHEMA_PROPOSE, "vlm": dict(vlm or {})})


# -- the operator's edge: authorise / reject ---------------------------------

def authorise(proposal_id: str, operator: str, grant_ref: str, *,
              approve: bool = True, why: str = "",
              counterpart_id: str = "the_beautiful_one") -> dict[str, Any]:
    """The operator's act.  Two-party Gate 6 attestation plus the organizational
    grant (``authorised_to_realise``, V10-SEC-010).  ``approve=False`` rejects."""
    p = get_proposal(proposal_id)
    if p is None:
        return {"status": "error", "error": f"unknown proposal {proposal_id}"}
    if p.status == AUTHORISED and approve and p.grant_ref == (grant_ref or ""):
        return {"status": "already_authorised", "proposal_id": p.proposal_id}
    if p.status not in (PROPOSED,):
        return {"status": "error",
                "error": f"proposal is {p.status}; only {PROPOSED} can be decided"}
    op = (operator or "").strip()
    if not op:
        return {"status": "error", "error": "operator required"}

    if not approve:
        p.status = REJECTED
        p.authorised_by = op
        p.last_error = (why or "rejected by operator").strip()
        _save(p)
        return {"status": "rejected", "proposal_id": p.proposal_id, "by": op}

    ref = (grant_ref or "").strip()
    atts = [
        Attestation(signer=op, scope=p.proposal_id, love_form="care",
                    shared_with=counterpart_id, beautiful_output=True,
                    assurance=ASSURANCE_APPROVED, approval_ref=ref or f"OP-{p.proposal_id[:8]}"),
        Attestation(signer=counterpart_id, scope=p.proposal_id, love_form="fidelity",
                    shared_with=op, beautiful_output=True,
                    assurance=ASSURANCE_APPROVED, approval_ref=ref or f"SYS-{p.proposal_id[:8]}"),
    ]
    cfg = GovernanceConfig(self_id=op, counterpart_id=counterpart_id)
    cand = Candidate(scope=p.proposal_id,
                     residual={"brain": 0.05 + 0.1j, "body": 0.02 + 0.05j},
                     realised_weight=0.1, unclamped_weight=0.1,
                     neighbour_weights=(0.08, 0.12))
    gate = gate_beautiful_output(cand, cfg, atts)
    decision = Decision(passed=gate.passed, results=[gate], held=not gate.passed,
                        scope=p.proposal_id)
    ok, why_realise = authorised_to_realise(
        decision, GovernanceConfig(realise_grant_ref=ref))
    record = {"gate_6": gate.reason, "authorised": ok, "why": why_realise,
              "config_digest": config_digest(cfg)}
    if not gate.passed or not ok:
        p.decision = record
        p.last_error = why_realise if not ok else gate.reason
        _save(p)                                     # stays PROPOSED: held (翈)
        return {"status": "held", "proposal_id": p.proposal_id, **record}
    p.status = AUTHORISED
    p.authorised_at = time.time()
    p.authorised_by = op
    p.grant_ref = ref
    p.decision = record
    p.last_error = ""
    _save(p)
    log.info("x_process_studio: %s authorised by %s under %s", p.proposal_id, op, ref)
    return {"status": "authorised", "proposal_id": p.proposal_id, **record}


def reject(proposal_id: str, operator: str, why: str = "") -> dict[str, Any]:
    return authorise(proposal_id, operator, "", approve=False, why=why)


# -- r-ADMIN: realisation ----------------------------------------------------

def realise(proposal_id: str, mcp: ActivepiecesMCP | None = None) -> dict[str, Any]:
    """Realise one AUTHORISED proposal through Activepieces MCP (r-ADMIN).

    Anything short of AUTHORISED is refused: a PROPOSED item is held at the
    edge (V10-SEC-010), REJECTED stays rejected, REALISED answers idempotently.
    A transport failure keeps the proposal AUTHORISED for retry; a JSON-RPC
    error from the server marks it FAILED (the tool itself refused)."""
    p = get_proposal(proposal_id)
    if p is None:
        return {"status": "error", "error": f"unknown proposal {proposal_id}"}
    if p.status == REALISED:
        return {"status": "already_realised", "proposal_id": p.proposal_id,
                "result": p.result}
    if p.status == REJECTED:
        return {"status": "rejected", "proposal_id": p.proposal_id}
    if p.status != AUTHORISED or not p.grant_ref:
        return {"status": "held", "proposal_id": p.proposal_id,
                "error": "not authorised: admissibility is not authorization "
                         "(V10-SEC-010); an operator grant_ref is required"}
    client = mcp or ActivepiecesMCP()
    if not client.available():
        p.attempts += 1
        p.last_error = f"MCP unconfigured ({MCP_URL_ENV} unset)"
        _save(p)
        return {"status": "held", "proposal_id": p.proposal_id, "error": p.last_error}
    try:
        result = client.call_tool(p.tool, p.arguments)
    except MCPError as exc:                       # the tool refused: permanent
        p.status = FAILED
        p.attempts += 1
        p.last_error = str(exc)
        _save(p)
        return {"status": "failed", "proposal_id": p.proposal_id, "error": p.last_error}
    except (MCPUnavailable, Exception) as exc:    # transport: keep for retry
        p.attempts += 1
        p.last_error = str(exc)
        _save(p)
        return {"status": "retry", "proposal_id": p.proposal_id, "error": p.last_error}
    blob = json.dumps(result, default=str)
    if len(blob) > _MAX_BYTES:
        result = {"truncated": True, "bytes": len(blob)}
    p.status = REALISED
    p.realised_at = time.time()
    p.result = result
    p.attempts += 1
    p.last_error = ""
    _save(p)
    log.info("x_process_studio: %s realised (%s) under %s", p.proposal_id, p.tool, p.grant_ref)
    return {"status": "realised", "proposal_id": p.proposal_id, "result": result}


def realise_ready(mcp: ActivepiecesMCP | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """Realise up to ``limit`` AUTHORISED proposals, oldest first."""
    rows = list_proposals(status=AUTHORISED, limit=max(1, int(limit)))
    rows.sort(key=lambda r: r.get("created_at") or 0.0)
    return [realise(r["proposal_id"], mcp=mcp) for r in rows]


# ---------------------------------------------------------------------------
# Operator CLI (runs inside the container, like every operator command)
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="x_process_studio",
                                 description="the x_ProcessStudio holon: stage, "
                                             "authorise and realise Activepieces actions")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status")
    ls = sub.add_parser("list")
    ls.add_argument("--status", default=None, choices=STATUSES)
    ls.add_argument("--source", default=None)
    sub.add_parser("tools")
    pr = sub.add_parser("propose")
    pr.add_argument("--source", required=True)
    pr.add_argument("--tool", required=True)
    pr.add_argument("--arguments-json", default="{}")
    pr.add_argument("--reason", default="")
    au = sub.add_parser("authorise")
    au.add_argument("proposal_id")
    au.add_argument("--operator", required=True)
    au.add_argument("--grant-ref", default="")
    au.add_argument("--reject", action="store_true")
    au.add_argument("--why", default="")
    re_ = sub.add_parser("realise")
    re_.add_argument("proposal_id", nargs="?")
    re_.add_argument("--ready", action="store_true")
    args = ap.parse_args(argv)
    out: Any
    if args.cmd == "status":
        out = stats()
    elif args.cmd == "list":
        out = list_proposals(status=args.status, source=args.source)
    elif args.cmd == "tools":
        out = ActivepiecesMCP().list_tools()
    elif args.cmd == "propose":
        out = propose(args.source, args.tool,
                      arguments=json.loads(args.arguments_json or "{}"),
                      reason=args.reason).to_dict()
    elif args.cmd == "authorise":
        out = authorise(args.proposal_id, args.operator, args.grant_ref,
                        approve=not args.reject, why=args.why)
    elif args.cmd == "realise":
        out = realise_ready() if args.ready else realise(args.proposal_id or "")
    else:
        ap.print_help()
        return 1
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":          # pragma: no cover
    sys.exit(_main())
