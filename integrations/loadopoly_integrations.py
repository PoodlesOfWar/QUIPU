"""Loadopoly integrations SDK (Python, stdlib only).

Talks to QUIPU's x_ProcessStudio MCP over Streamable HTTP.  What it can do is exactly what the server
offers a v-ADMIN client: read status/ledger, list the Hub action catalogue, stage a proposal (held), and
hear the Perceptopoly voice while a human Admin has it on.  There is no authorise/realise method, because
the server has no such tool: a human decides, elsewhere.

    from loadopoly_integrations import Client
    c = Client("https://<host>/mcp", token=os.environ["QUIPU_MCP_TOKEN"])
    c.propose("hub.body_directive", "line 3 is quiet",
              {"title": "Check line 3", "do_this": "Walk it", "owner_role": "Ops"})
"""
from __future__ import annotations

import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Optional

__version__ = "0.1.0"
PROTOCOL = "2025-03-26"


class IntegrationError(RuntimeError):
    pass


def _private_host(host: str) -> bool:
    if host in ("localhost", "") or host.endswith(".local") or "." not in host:
        return host != ""
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False


def _parse(body: str) -> Optional[dict]:
    body = body.strip()
    if not body:
        return None
    if body.startswith("{"):
        try:
            return json.loads(body)
        except ValueError:
            return None
    for line in body.splitlines():
        if line.startswith("data:"):
            try:
                return json.loads(line[5:].strip())
            except ValueError:
                return None
    return None


class Client:
    def __init__(self, url: str, token: str, *, timeout: float = 15.0):
        u = urllib.parse.urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise IntegrationError("url must be http(s)")
        if u.scheme == "http" and not _private_host(u.hostname):
            raise IntegrationError("plain http to a public address is refused; use https")
        if not token:
            raise IntegrationError("a token is required")
        self.url, self._token, self.timeout = url, token, timeout
        self._session: Optional[str] = None
        self._id = 0

    def __repr__(self) -> str:                      # never show the token
        return f"Client({self.url!r})"

    def _rpc(self, method: str, params: Optional[dict] = None, notify: bool = False) -> Any:
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self._id += 1
            msg["id"] = self._id
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "Authorization": "Bearer " + self._token}
        if self._session:
            headers["Mcp-Session-Id"] = self._session
        req = urllib.request.Request(self.url, json.dumps(msg).encode(), headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self._session = r.headers.get("Mcp-Session-Id") or self._session
                out = _parse(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            raise IntegrationError(f"server answered {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise IntegrationError(f"cannot reach the server: {getattr(e, 'reason', e)}") from None
        if notify:
            return None
        if not out:
            raise IntegrationError("empty or unreadable reply")
        if "error" in out:
            raise IntegrationError(str(out["error"].get("message", out["error"])))
        return out.get("result")

    def connect(self) -> "Client":
        if self._session is None and self._id == 0:
            self._rpc("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                     "clientInfo": {"name": "loadopoly-integrations", "version": __version__}})
            self._rpc("notifications/initialized", notify=True)
        return self

    def tools(self) -> list:
        return self.connect()._rpc("tools/list", {}).get("tools", [])

    def call(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> Any:
        r = self.connect()._rpc("tools/call", {"name": name, "arguments": dict(arguments or {})})
        text = "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
        if r.get("isError"):
            raise IntegrationError(text or "tool error")
        try:
            return json.loads(text)
        except ValueError:
            return text

    def status(self) -> Any:
        return self.call("process_studio_status")

    def list(self, status: Optional[str] = None) -> Any:
        return self.call("process_studio_list", {"status": status} if status else {})

    def hub_actions(self) -> Any:
        return self.call("hub_actions")

    def propose(self, tool: str, reason: str, arguments: Optional[Mapping[str, Any]] = None) -> Any:
        """Stage a proposal.  It is held; a human authorises it elsewhere."""
        return self.call("process_studio_propose", {"tool": tool, "reason": reason, "arguments": dict(arguments or {})})

    def voice(self) -> Any:
        """The Perceptopoly voice; refused by the server unless a human Admin has it on."""
        return self.call("perceptopoly_voice")
