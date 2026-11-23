#!/usr/bin/env python3
"""quipu_mcp — use QUIPU's x_ProcessStudio MCP from an edge device (Termux).

This talks to the **same MCP server the VS Code / Open VSX extension registers**
(`quipu.x-process-studio`): the `quipu-mcp` service hosted on the Black Bulls fleet,
at its `/mcp` URL, with the same bearer token.  It is not a second MCP and adds no
capability: the tools are exactly the v-ADMIN ones (status, list, Activepieces tools,
the hub action catalogue, propose, and the Perceptopoly voice only while a human
Admin has it on).  There is no authorise or realise tool here or there: an edge
device can want things, never decide them.

    quipu_mcp.py doctor                      check URL, token, handshake and tool parity
    quipu_mcp.py tools                       list the tools the server offers
    quipu_mcp.py status                      the action ledger's counts
    quipu_mcp.py list [--status PROPOSED]    proposals
    quipu_mcp.py propose TOOL --reason WHY [--args JSON]    stage a proposal (held for the operator)
    quipu_mcp.py voice [--kind status|proposal|queue_entry] [--id ID]
    quipu_mcp.py call TOOL [JSON]            any tool, with JSON arguments

Configuration (environment, else ``~/.config/quipu/mcp.env`` as KEY=VALUE lines):
    QUIPU_MCP_URL          the server's /mcp address (the extension's ``xProcessStudio.serverUrl``)
    QUIPU_MCP_TOKEN        the bearer token (or QUIPU_MCP_TOKEN_FILE: a file holding it)

The token is never printed.  Plain ``http://`` is accepted only for loopback, private
and container addresses; anything else must be ``https://``.  Stdlib only.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse, urlunparse

CONFIG_PATH = Path.home() / ".config" / "quipu" / "mcp.env"
PROTOCOL = "2025-06-18"
TIMEOUT_S = 20.0
EXIT_OK, EXIT_FAIL, EXIT_SETUP = 0, 1, 3

#: The v-ADMIN surface the extension sees.  An edge device must see exactly this (plus the
#: voice while it is on), and nothing that decides or runs anything.
EXPECTED_TOOLS = {"process_studio_status", "process_studio_list", "activepieces_tools",
                  "hub_actions", "process_studio_propose"}
OPTIONAL_TOOLS = {"perceptopoly_voice"}
FORBIDDEN_FRAGMENTS = ("authoris", "authoriz", "realis", "realiz", "approve", "promote")


class EdgeError(RuntimeError):
    pass


def load_env_file(path: Path = CONFIG_PATH) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def plain_http_ok(host: str) -> bool:
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


def resolve_config(env: Mapping[str, str], file_env: Mapping[str, str]) -> tuple[str, str]:
    url = (env.get("QUIPU_MCP_URL") or file_env.get("QUIPU_MCP_URL") or "").strip()
    token = (env.get("QUIPU_MCP_TOKEN") or file_env.get("QUIPU_MCP_TOKEN") or "").strip()
    tfile = (env.get("QUIPU_MCP_TOKEN_FILE") or file_env.get("QUIPU_MCP_TOKEN_FILE") or "").strip()
    if not token and tfile:
        try:
            token = Path(tfile).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise EdgeError(f"cannot read QUIPU_MCP_TOKEN_FILE: {exc}") from exc
    if not url:
        raise EdgeError("QUIPU_MCP_URL is not set (the extension's xProcessStudio.serverUrl). "
                        "Put it in ~/.config/quipu/mcp.env or run edge_setup.sh.")
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise EdgeError("QUIPU_MCP_URL must be an http(s) URL")
    if p.scheme == "http" and not plain_http_ok(p.hostname):
        raise EdgeError("plain http to a public address is refused; use the https front door")
    if not token:
        raise EdgeError("no token: set QUIPU_MCP_TOKEN or QUIPU_MCP_TOKEN_FILE (the same token the extension stores)")
    return url, token


class McpEdge:
    """A minimal MCP client over Streamable HTTP (initialize -> notifications/initialized -> tools/*)."""

    def __init__(self, url: str, token: str, opener=None, timeout: float = TIMEOUT_S) -> None:
        self.url, self.token, self.timeout = url, token, timeout
        self._open = opener or urllib.request.urlopen
        self.session: Optional[str] = None
        self._id = 0
        self.initialised = False
        self.server_info: dict[str, Any] = {}

    def _post(self, payload: Mapping[str, Any], expect_reply: bool = True) -> Optional[dict[str, Any]]:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "Authorization": f"Bearer {self.token}"}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with self._open(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self.session = sid
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise EdgeError("401: the token is missing or wrong") from exc
            if exc.code == 403:
                raise EdgeError("403: the server refused this origin") from exc
            raise EdgeError(f"HTTP {exc.code} from the MCP server") from exc
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            raise EdgeError(f"cannot reach the MCP server: {getattr(exc, 'reason', exc)}") from exc
        if not expect_reply:
            return None
        msg = _parse(raw)
        if msg is None:
            raise EdgeError("the MCP server sent no JSON-RPC reply")
        if msg.get("error"):
            raise EdgeError(f"MCP error {msg['error'].get('code')}: {msg['error'].get('message')}")
        return msg

    def _rpc(self, method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._id += 1
        msg = self._post({"jsonrpc": "2.0", "id": self._id, "method": method, "params": dict(params or {})})
        result = (msg or {}).get("result")
        return result if isinstance(result, dict) else {}

    def initialize(self) -> dict[str, Any]:
        out = self._rpc("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                       "clientInfo": {"name": "quipu-edge", "version": "1"}})
        self.server_info = out.get("serverInfo") or {}
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, expect_reply=False)
        self.initialised = True
        return out

    def _ensure(self) -> None:
        if not self.initialised:
            self.initialize()

    def tools(self) -> list[dict[str, Any]]:
        self._ensure()
        t = self._rpc("tools/list").get("tools")
        return list(t) if isinstance(t, list) else []

    def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        self._ensure()
        return self._rpc("tools/call", {"name": name, "arguments": dict(arguments or {})})


def _parse(text: str) -> Optional[dict[str, Any]]:
    t = (text or "").strip()
    if not t:
        return None
    if t[0] in "{[":
        try:
            v = json.loads(t)
            return v if isinstance(v, dict) else None
        except ValueError:
            return None
    for line in t.splitlines():                      # SSE-framed body
        if line.startswith("data:"):
            try:
                m = json.loads(line[5:].strip())
            except ValueError:
                continue
            if isinstance(m, dict) and ("result" in m or "error" in m):
                return m
    return None


def health_url(mcp_url: str) -> str:
    p = urlparse(mcp_url)
    return urlunparse((p.scheme, p.netloc, "/health", "", "", ""))


def tool_text(result: Mapping[str, Any]) -> str:
    return "\n".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))


# --------------------------------------------------------------------------- doctor
def doctor(client: McpEdge, opener=None) -> tuple[list[str], bool]:
    """Check the edge sees the same v-ADMIN-only MCP the extension does.  Never prints the token."""
    lines: list[str] = []
    ok = True
    p = urlparse(client.url)
    lines.append(f"ok    url scheme {p.scheme} ({'private/loopback' if plain_http_ok(p.hostname or '') else 'public'} host)")
    try:
        req = urllib.request.Request(health_url(client.url), headers={"Accept": "application/json"})
        with (opener or urllib.request.urlopen)(req, timeout=client.timeout) as r:
            h = json.loads(r.read().decode())
        lines.append(f"ok    server reachable ({h.get('service')} {h.get('version')}; auth: {h.get('auth')})")
    except Exception as exc:
        lines.append(f"FAIL  cannot reach the server's /health: {exc}")
        return lines, False
    try:
        init = client.initialize()
        lines.append(f"ok    MCP handshake (protocol {init.get('protocolVersion')}, {client.server_info.get('name')})")
        names = {t["name"] for t in client.tools()}
    except EdgeError as exc:
        lines.append(f"FAIL  {exc}")
        return lines, False
    missing = EXPECTED_TOOLS - names
    extra = names - EXPECTED_TOOLS - OPTIONAL_TOOLS
    if missing:
        lines.append("FAIL  tools missing compared with the extension's: " + ", ".join(sorted(missing)))
        ok = False
    else:
        lines.append(f"ok    the same {len(EXPECTED_TOOLS)} v-ADMIN tools the extension lists")
    if extra:
        lines.append("WARN  tools this client does not know: " + ", ".join(sorted(extra)))
    bad = [n for n in names if any(f in n.lower() for f in FORBIDDEN_FRAGMENTS)]
    if bad:
        lines.append("FAIL  a tool that could decide or run something is exposed: " + ", ".join(sorted(bad)))
        ok = False
    else:
        lines.append("ok    no authorise/realise tool: this device can want, not decide")
    lines.append("ok    Perceptopoly voice is ON (a human Admin enabled it)" if "perceptopoly_voice" in names
                 else "ok    Perceptopoly voice is off (a human Admin has not enabled it)")
    return lines, ok


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[list] = None, env: Optional[Mapping[str, str]] = None, opener=None) -> int:
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(prog="quipu_mcp", description="QUIPU's x_ProcessStudio MCP from an edge device")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    sub.add_parser("tools")
    sub.add_parser("status")
    ls = sub.add_parser("list")
    ls.add_argument("--status", choices=("PROPOSED", "AUTHORISED", "REALISED", "FAILED", "REJECTED"))
    ls.add_argument("--source")
    ls.add_argument("--limit", type=int, default=20)
    pr = sub.add_parser("propose")
    pr.add_argument("tool")
    pr.add_argument("--reason", required=True)
    pr.add_argument("--args", default="{}", help="JSON arguments for the tool")
    pr.add_argument("--device", default=None, help="label recorded with the proposal (default: hostname)")
    vo = sub.add_parser("voice")
    vo.add_argument("--kind", default="status", choices=("status", "proposal", "queue_entry"))
    vo.add_argument("--id", default="")
    ca = sub.add_parser("call")
    ca.add_argument("tool")
    ca.add_argument("arguments", nargs="?", default="{}")
    args = ap.parse_args(argv)

    try:
        url, token = resolve_config(env, load_env_file())
    except EdgeError as exc:
        print(f"quipu_mcp: {exc}", file=sys.stderr)
        return EXIT_SETUP
    client = McpEdge(url, token, opener=opener)
    try:
        if args.cmd == "doctor":
            lines, ok = doctor(client, opener=opener)
            print("\n".join(lines))
            return EXIT_OK if ok else EXIT_FAIL
        if args.cmd == "tools":
            for t in client.tools():
                print(f"{t['name']}: {t.get('title') or ''}")
            return EXIT_OK
        if args.cmd == "status":
            res = client.call("process_studio_status", {})
        elif args.cmd == "list":
            a: dict[str, Any] = {"limit": args.limit}
            if args.status:
                a["status"] = args.status
            if args.source:
                a["source"] = args.source
            res = client.call("process_studio_list", a)
        elif args.cmd == "propose":
            try:
                targs = json.loads(args.args)
            except ValueError as exc:
                print(f"quipu_mcp: --args is not JSON: {exc}", file=sys.stderr)
                return EXIT_SETUP
            res = client.call("process_studio_propose", {
                "tool": args.tool, "arguments": targs, "reason": args.reason,
                "meta": {"edge_device": (args.device or socket.gethostname())[:64]}})
        elif args.cmd == "voice":
            a = {"kind": args.kind}
            if args.kind == "proposal":
                a["proposalId"] = args.id
            elif args.kind == "queue_entry":
                a["entryId"] = args.id
            res = client.call("perceptopoly_voice", a)
        else:
            try:
                res = client.call(args.tool, json.loads(args.arguments))
            except ValueError as exc:
                print(f"quipu_mcp: arguments are not JSON: {exc}", file=sys.stderr)
                return EXIT_SETUP
    except EdgeError as exc:
        print(f"quipu_mcp: {exc}", file=sys.stderr)
        return EXIT_FAIL
    print(tool_text(res))
    return EXIT_FAIL if res.get("isError") else EXIT_OK


if __name__ == "__main__":          # pragma: no cover
    sys.exit(main())
