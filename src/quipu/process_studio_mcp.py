"""process_studio_mcp — the x_ProcessStudio holon as an MCP server, for IDE agents.

VS Code / Open VSX agents (and any MCP client) reach the holon here.  The
server speaks MCP over **Streamable HTTP** (``POST /mcp``) or **stdio**, and it
exposes only what v-ADMIN may do:

    process_studio_status     ledger counts, MCP availability
    process_studio_list       proposals, filtered by status / source
    activepieces_tools        what the operator's Activepieces can do (names + schemas)
    process_studio_propose    stage an ActionProposal — held for the operator

There is deliberately **no** authorise or realise tool.  An agent can want; only
the operator's grant (``grant_ref``, V10-SEC-010) lets r-ADMIN act, and that
happens in the QUIPU container (CLI) or from an operator-signed source — never
from an editor.

One brain, one writer: this process never opens the brain.  It is a thin client
of the observer (``QUIPU_URL``): reads are plain GETs, the one write is a POST
signed with QUIPU's edge canonical form as ``QUIPU_MCP_SOURCE`` (default
``vscode-agent``), so ``QUIPU_EDGE_AUTH=enforce`` and the operator's budget
apply to agents exactly as to every other writer.

Security posture:
    * ``QUIPU_MCP_TOKEN`` — bearer token every MCP request must carry
      (constant-time compared).  Unset ⇒ every request is answered 401: the
      server fails closed rather than open.  ``QUIPU_MCP_ALLOW_OPEN=1`` permits
      tokenless use, and only when bound to loopback.
    * ``Origin`` is validated (DNS-rebinding defence, as the MCP transport spec
      requires): a request carrying an Origin not in ``QUIPU_MCP_ORIGINS`` is 403.
      Editor clients send none.
    * Bodies are capped at 1 MB.

Run::

    python -m src.quipu.process_studio_mcp serve --host 127.0.0.1 --port 7110
    python -m src.quipu.process_studio_mcp stdio

Stdlib only.
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlencode, urlparse

from ._version import __version__
from .edge_client import idempotency_key, sign_headers

log = logging.getLogger("quipu.process_studio_mcp")

DEFAULT_PORT = 7110
MCP_PATH = "/mcp"
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL = PROTOCOL_VERSIONS[0]
_MAX_BODY = 1024 * 1024
_HTTP_TIMEOUT_S = 15.0

TOKEN_ENV = "QUIPU_MCP_TOKEN"
ALLOW_OPEN_ENV = "QUIPU_MCP_ALLOW_OPEN"
ORIGINS_ENV = "QUIPU_MCP_ORIGINS"
SOURCE_ENV = "QUIPU_MCP_SOURCE"

INSTRUCTIONS = (
    "x_ProcessStudio holon (QUIPU). You act as v-ADMIN: you may inspect the "
    "action ledger, list the operator's Activepieces tools, and STAGE action "
    "proposals with process_studio_propose. Staged proposals are held until the "
    "operator authorises them with an organizational grant reference; only then "
    "does r-ADMIN realise them. You cannot authorise or realise. Explain in "
    "`reason` why the action is wanted, and pass `arguments` that match the "
    "Activepieces tool's inputSchema from activepieces_tools."
)

_STATUSES = ["PROPOSED", "AUTHORISED", "REALISED", "FAILED", "REJECTED"]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "process_studio_status",
        "title": "x_ProcessStudio status",
        "description": "Ledger counts by status, whether Activepieces MCP is configured, "
                       "and the most recent proposals.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "process_studio_list",
        "title": "List action proposals",
        "description": "List staged/authorised/realised action proposals, newest first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": _STATUSES},
                "source": {"type": "string", "description": "fleet source, e.g. stockopoly"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
            },
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "activepieces_tools",
        "title": "Activepieces tools",
        "description": "The operator's Activepieces MCP tools (name, description, inputSchema) "
                       "that a proposal may name. Listing does not run anything.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True, "openWorldHint": True},
    },
    {
        "name": "process_studio_propose",
        "title": "Stage an action proposal",
        "description": "Stage an ActionProposal for an Activepieces tool. It is HELD for the "
                       "operator: nothing runs until the operator authorises it with a grant "
                       "reference. Identical proposals are de-duplicated.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "minLength": 1,
                         "description": "Activepieces tool name (see activepieces_tools)"},
                "arguments": {"type": "object", "description": "arguments for that tool"},
                "reason": {"type": "string", "description": "why this action is wanted"},
                "meta": {"type": "object", "description": "optional evidence, e.g. {vlm: {...}}"},
            },
            "required": ["tool", "reason"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False,
                        "idempotentHint": True, "openWorldHint": False},
    },
]


# ---------------------------------------------------------------------------
# Observer client (reads plain, the one write signed)
# ---------------------------------------------------------------------------

class ObserverClient:
    def __init__(self, base_url: str | None = None, source: str | None = None,
                 key_hex: str | None = None,
                 opener: Callable[..., Any] | None = None) -> None:
        self.base = (base_url or os.environ.get("QUIPU_URL") or "http://quipu:7100").rstrip("/")
        self.source = (source or os.environ.get(SOURCE_ENV) or "vscode-agent").strip()
        env_key = "QUIPU_EDGE_KEY_" + self.source.upper().replace("-", "_")
        raw = key_hex if key_hex is not None else (os.environ.get(env_key)
                                                  or os.environ.get("QUIPU_EDGE_KEY") or "")
        try:
            bytes.fromhex(raw.strip())
            self.key = raw.strip() or None
        except ValueError:
            self.key = None
        self._opener = opener or urllib.request.urlopen

    def _read(self, req: urllib.request.Request) -> tuple[int, Any]:
        try:
            with self._opener(req, timeout=_HTTP_TIMEOUT_S) as resp:
                code = getattr(resp, "status", 200)
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            code = exc.code
            try:
                raw = exc.read()
            except Exception:
                raw = b""
        try:
            return code, json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            return code, {"ok": False, "error": raw.decode("utf-8", "replace")[:300]}

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> tuple[int, Any]:
        q = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        url = self.base + path + (("?" + urlencode(q)) if q else "")
        return self._read(urllib.request.Request(url, headers={"Accept": "application/json"}))

    def post_signed(self, path: str, payload: Mapping[str, Any]) -> tuple[int, Any]:
        body = json.dumps(dict(payload), separators=(",", ":"), sort_keys=True).encode("utf-8")
        idem = idempotency_key(self.source, path, body)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.key:
            headers.update(sign_headers(self.key, self.source, "POST", path, body, idem=idem))
        else:
            headers.update({"X-Quipu-Source": self.source, "X-Quipu-Idempotency": idem})
        req = urllib.request.Request(self.base + path, data=body, headers=headers, method="POST")
        return self._read(req)


# ---------------------------------------------------------------------------
# MCP dispatch (transport-independent)
# ---------------------------------------------------------------------------

def _text_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}],
        "isError": bool(is_error),
    }
    if isinstance(payload, dict):
        out["structuredContent"] = payload
    return out


class ProcessStudioMCP:
    """JSON-RPC handler for the MCP methods this server implements."""

    def __init__(self, observer: ObserverClient | None = None) -> None:
        self.observer = observer or ObserverClient()

    # -- tools -------------------------------------------------------------
    def call_tool(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        args = dict(args or {})
        if name == "process_studio_status":
            code, data = self.observer.get("/process_studio")
            return _text_result(data, is_error=code != 200)
        if name == "process_studio_list":
            limit = args.get("limit", 20)
            try:
                limit = max(1, min(int(limit), 200))
            except (TypeError, ValueError):
                return _text_result({"error": "limit must be an integer"}, is_error=True)
            status = args.get("status")
            if status is not None and status not in _STATUSES:
                return _text_result({"error": f"status must be one of {_STATUSES}"}, is_error=True)
            code, data = self.observer.get("/process_studio", {
                "status": status, "source": args.get("source"), "limit": limit})
            if code != 200 or not isinstance(data, dict):
                return _text_result(data, is_error=True)
            return _text_result({"proposals": data.get("recent", []),
                                 "counts": data.get("proposals", {})})
        if name == "activepieces_tools":
            code, data = self.observer.get("/process_studio/tools")
            return _text_result(data, is_error=code != 200 or not (isinstance(data, dict) and data.get("ok")))
        if name == "process_studio_propose":
            tool = str(args.get("tool") or "").strip()
            reason = str(args.get("reason") or "").strip()
            if not tool or not reason:
                return _text_result({"error": "tool and reason are required"}, is_error=True)
            arguments = args.get("arguments") or {}
            meta = args.get("meta") or {}
            if not isinstance(arguments, dict) or not isinstance(meta, dict):
                return _text_result({"error": "arguments and meta must be objects"}, is_error=True)
            meta = {**meta, "via": "process_studio_mcp"}
            code, data = self.observer.post_signed("/process_studio/propose", {
                "source": self.observer.source, "tool": tool, "arguments": arguments,
                "reason": reason, "meta": meta})
            ok = code == 200 and isinstance(data, dict) and data.get("ok")
            if ok and data.get("duplicate"):
                data = {**data, "note": "identical proposal already staged; not staged twice"}
            elif ok:
                data = {**data, "note": "staged and HELD: the operator must authorise it "
                                        "with a grant reference before anything runs"}
            return _text_result(data, is_error=not ok)
        raise KeyError(name)

    # -- JSON-RPC ------------------------------------------------------------
    def handle(self, msg: Any) -> Optional[dict[str, Any]]:
        """One JSON-RPC message in, one response (or None for a notification) out."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return _rpc_error(None, -32600, "invalid request")
        mid = msg.get("id")
        method = msg.get("method")
        if not isinstance(method, str):
            # a response object from the client: nothing to answer
            return None
        is_notification = "id" not in msg
        params = msg.get("params") or {}
        try:
            result = self._dispatch(method, params)
        except _RpcError as exc:
            return None if is_notification else _rpc_error(mid, exc.code, exc.message)
        except Exception as exc:  # never leak a traceback to the client
            log.warning("process_studio_mcp: %s failed: %s", method, exc)
            return None if is_notification else _rpc_error(mid, -32603, "internal error")
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _dispatch(self, method: str, params: Mapping[str, Any]) -> Any:
        if method == "initialize":
            requested = str((params or {}).get("protocolVersion") or "")
            version = requested if requested in PROTOCOL_VERSIONS else LATEST_PROTOCOL
            return {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "quipu-x-process-studio", "title": "x_ProcessStudio",
                               "version": __version__},
                "instructions": INSTRUCTIONS,
            }
        if method in ("notifications/initialized", "notifications/cancelled"):
            return None
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": TOOLS}
        if method == "tools/call":
            name = str((params or {}).get("name") or "")
            if name not in {t["name"] for t in TOOLS}:
                raise _RpcError(-32602, f"unknown tool: {name}")
            arguments = (params or {}).get("arguments") or {}
            if not isinstance(arguments, dict):
                raise _RpcError(-32602, "arguments must be an object")
            return self.call_tool(name, arguments)
        raise _RpcError(-32601, f"method not found: {method}")


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def _rpc_error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# Streamable HTTP transport
# ---------------------------------------------------------------------------

def _expected_token() -> Optional[str]:
    tok = os.environ.get(TOKEN_ENV, "").strip()
    return tok or None


def _allowed_origins() -> set[str]:
    return {o.strip().rstrip("/") for o in os.environ.get(ORIGINS_ENV, "").split(",") if o.strip()}


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


class _McpHandler(BaseHTTPRequestHandler):
    server_version = f"quipu-process-studio-mcp/{__version__}"
    app: ProcessStudioMCP            # set on the server class
    sessions: dict[str, float]
    sessions_lock: threading.Lock
    allow_open: bool = False

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _send(self, code: int, payload: Any = None, extra: Mapping[str, str] | None = None) -> None:
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        self.send_response(code)
        if payload is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _guard(self) -> bool:
        """Origin (DNS rebinding) and bearer token.  True when the request may proceed."""
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if origin and origin not in _allowed_origins():
            self._send(403, {"error": "origin not allowed"})
            return False
        expected = _expected_token()
        if expected is None:
            if self.allow_open:
                return True
            self._send(401, {"error": f"{TOKEN_ENV} unset: the operator must set a token "
                                      "(or QUIPU_MCP_ALLOW_OPEN=1 on loopback)"},
                       {"WWW-Authenticate": 'Bearer realm="quipu-mcp"'})
            return False
        auth = self.headers.get("Authorization") or ""
        got = auth[7:].strip() if auth.startswith("Bearer ") else ""
        if not got or not hmac.compare_digest(got, expected):
            self._send(401, {"error": "unauthorized"},
                       {"WWW-Authenticate": 'Bearer realm="quipu-mcp"'})
            return False
        return True

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"ok": True, "service": "quipu-process-studio-mcp",
                             "version": __version__,
                             "auth": "token" if _expected_token() else
                                     ("open" if self.allow_open else "closed (token unset)")})
            return
        if path == MCP_PATH:
            # No server-initiated stream: this server only answers requests.
            self._send(405, {"error": "use POST"}, {"Allow": "POST, DELETE"})
            return
        self._send(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        if urlparse(self.path).path != MCP_PATH:
            self._send(404, {"error": "not found"})
            return
        if not self._guard():
            return
        sid = self.headers.get("Mcp-Session-Id") or ""
        with self.sessions_lock:
            self.sessions.pop(sid, None)
        self._send(200, {"ok": True})

    def do_POST(self) -> None:
        if urlparse(self.path).path != MCP_PATH:
            self._send(404, {"error": "not found"})
            return
        if not self._guard():
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0:
            self._send(400, _rpc_error(None, -32700, "empty body"))
            return
        if length > _MAX_BODY:
            self._send(413, _rpc_error(None, -32600, "body too large"))
            return
        try:
            msg = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send(400, _rpc_error(None, -32700, "parse error"))
            return

        sid = self.headers.get("Mcp-Session-Id")
        messages = msg if isinstance(msg, list) else [msg]
        is_init = any(isinstance(m, dict) and m.get("method") == "initialize" for m in messages)
        if sid and not is_init:
            with self.sessions_lock:
                known = sid in self.sessions
            if not known:
                self._send(404, _rpc_error(None, -32001, "unknown or expired session"))
                return
        extra: dict[str, str] = {}
        if is_init:
            sid = uuid.uuid4().hex
            with self.sessions_lock:
                self.sessions[sid] = time.time()
                if len(self.sessions) > 1024:            # bounded
                    for old in sorted(self.sessions, key=self.sessions.get)[:256]:
                        self.sessions.pop(old, None)
            extra["Mcp-Session-Id"] = sid

        replies = [r for r in (self.app.handle(m) for m in messages) if r is not None]
        if not replies:
            self._send(202, None, extra)                 # notifications / responses only
            return
        self._send(200, replies if isinstance(msg, list) else replies[0], extra)


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                app: ProcessStudioMCP | None = None,
                allow_open: bool | None = None) -> ThreadingHTTPServer:
    if allow_open is None:
        allow_open = os.environ.get(ALLOW_OPEN_ENV, "") in ("1", "true", "yes")
    if allow_open and not _is_loopback(host):
        log.warning("process_studio_mcp: %s ignored off loopback (%s)", ALLOW_OPEN_ENV, host)
        allow_open = False
    handler = type("McpHandler", (_McpHandler,), {
        "app": app or ProcessStudioMCP(),
        "sessions": {},
        "sessions_lock": threading.Lock(),
        "allow_open": bool(allow_open),
    })
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    httpd = make_server(host, port)
    auth = "token" if _expected_token() else ("OPEN (loopback)" if httpd.RequestHandlerClass.allow_open
                                              else f"CLOSED until {TOKEN_ENV} is set")
    print(f"[quipu-mcp] x_ProcessStudio MCP on http://{host}:{port}{MCP_PATH}  auth: {auth}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# stdio transport (newline-delimited JSON-RPC)
# ---------------------------------------------------------------------------

def run_stdio(app: ProcessStudioMCP | None = None, stdin=None, stdout=None) -> None:
    app = app or ProcessStudioMCP()
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            reply: Any = _rpc_error(None, -32700, "parse error")
        else:
            if isinstance(msg, list):
                replies = [r for r in (app.handle(m) for m in msg) if r is not None]
                reply = replies or None
            else:
                reply = app.handle(msg)
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="process_studio_mcp",
                                 description="x_ProcessStudio holon as an MCP server (v-ADMIN tools)")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve")
    s.add_argument("--host", default=os.environ.get("QUIPU_MCP_HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("QUIPU_MCP_PORT", DEFAULT_PORT)))
    sub.add_parser("stdio")
    args = ap.parse_args(argv)
    if args.cmd == "serve":
        serve(args.host, args.port)
        return 0
    if args.cmd == "stdio":
        run_stdio()
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":          # pragma: no cover
    sys.exit(_main())
