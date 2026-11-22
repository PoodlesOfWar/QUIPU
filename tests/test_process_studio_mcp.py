"""process_studio_mcp — the x_ProcessStudio holon as an MCP server: v-ADMIN
tools only, token + origin guard, Streamable HTTP and stdio, driven end to end
against a live in-process observer."""
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.quipu import observer_service as osvc
from src.quipu import process_studio_mcp as mcp
from src.quipu import x_process_studio as ps


@pytest.fixture()
def observer():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), osvc.ObserverHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture()
def mcp_server(observer, monkeypatch):
    monkeypatch.setenv(mcp.TOKEN_ENV, "s3cret-token")
    monkeypatch.delenv(mcp.ORIGINS_ENV, raising=False)
    app = mcp.ProcessStudioMCP(mcp.ObserverClient(base_url=observer, key_hex=""))
    httpd = mcp.make_server("127.0.0.1", 0, app=app)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(url, payload, *, token="s3cret-token", session=None, origin=None):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if session:
        headers["Mcp-Session-Id"] = session
    if origin:
        headers["Origin"] = origin
    req = urllib.request.Request(url + "/mcp", data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None), dict(r.headers)
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw else None), dict(e.headers)


def _init(url):
    code, body, hdr = _post(url, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                  "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "clientInfo": {"name": "vscode", "version": "1.101"}}})
    assert code == 200
    return body, hdr["Mcp-Session-Id"]


def test_handshake_lists_only_vadmin_tools(mcp_server):
    body, sid = _init(mcp_server)
    assert body["result"]["protocolVersion"] == "2025-06-18"
    assert "v-ADMIN" in body["result"]["instructions"]
    code, _, _ = _post(mcp_server, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session=sid)
    assert code == 202
    code, body, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session=sid)
    names = {t["name"] for t in body["result"]["tools"]}
    assert names == {"process_studio_status", "process_studio_list",
                     "activepieces_tools", "process_studio_propose"}
    assert not any("authori" in n or "realis" in n for n in names)   # v-ADMIN cannot act


def test_propose_over_mcp_stages_held_in_the_ledger(mcp_server):
    _, sid = _init(mcp_server)
    code, body, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "process_studio_propose",
        "arguments": {"tool": "send_slack_message", "arguments": {"channel": "#ops"},
                      "reason": "replenish rack A-12"}}}, session=sid)
    res = body["result"]
    assert res["isError"] is False
    sc = res["structuredContent"]
    assert sc["held"] is True and "operator must authorise" in sc["note"]
    pid = sc["proposal"]["proposal_id"]
    got = ps.get_proposal(pid)
    assert got.status == ps.PROPOSED and got.source == "vscode-agent"
    assert got.meta["via"] == "process_studio_mcp"

    # duplicate of the same body is de-duplicated by the edge, not staged twice
    code, body, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
        "name": "process_studio_propose",
        "arguments": {"tool": "send_slack_message", "arguments": {"channel": "#ops"},
                      "reason": "replenish rack A-12"}}}, session=sid)
    assert body["result"]["structuredContent"].get("duplicate") is True

    # list + status read back through the observer
    code, body, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
        "name": "process_studio_list", "arguments": {"status": "PROPOSED", "source": "vscode-agent"}}},
        session=sid)
    listed = body["result"]["structuredContent"]["proposals"]
    assert any(p["proposal_id"] == pid for p in listed)
    code, body, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                                       "params": {"name": "process_studio_status", "arguments": {}}},
                          session=sid)
    assert body["result"]["structuredContent"]["holon"] == "x_process_studio"


def test_propose_requires_reason_and_unknown_tool_is_rpc_error(mcp_server):
    _, sid = _init(mcp_server)
    _, body, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
        "name": "process_studio_propose", "arguments": {"tool": "t"}}}, session=sid)
    assert body["result"]["isError"] is True
    _, body, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {
        "name": "process_studio_realise", "arguments": {}}}, session=sid)
    assert body["error"]["code"] == -32602


def test_activepieces_tools_without_mcp_configured(mcp_server, monkeypatch):
    monkeypatch.delenv(ps.MCP_URL_ENV, raising=False)
    osvc._ps_tools_cache.update(at=0.0, payload=None)
    _, sid = _init(mcp_server)
    _, body, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                    "params": {"name": "activepieces_tools", "arguments": {}}}, session=sid)
    sc = body["result"]["structuredContent"]
    assert sc["mcp_configured"] is False and sc["tools"] == []


def test_token_origin_and_session_guards(mcp_server):
    code, _, hdr = _post(mcp_server, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, token=None)
    assert code == 401 and "Bearer" in hdr.get("WWW-Authenticate", "")
    code, _, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, token="wrong")
    assert code == 401
    code, _, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                       origin="https://evil.example")
    assert code == 403                                    # DNS-rebinding defence
    code, body, _ = _post(mcp_server, {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                          session="not-a-session")
    assert code == 404


def test_fails_closed_without_token(observer, monkeypatch):
    monkeypatch.delenv(mcp.TOKEN_ENV, raising=False)
    httpd = mcp.make_server("127.0.0.1", 0,
                            app=mcp.ProcessStudioMCP(mcp.ObserverClient(base_url=observer, key_hex="")),
                            allow_open=False)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        code, body, _ = _post(url, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, token=None)
        assert code == 401 and mcp.TOKEN_ENV in body["error"]
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            assert "closed" in json.loads(r.read())["auth"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_allow_open_is_refused_off_loopback():
    srv = mcp.make_server("0.0.0.0", 0, app=mcp.ProcessStudioMCP(mcp.ObserverClient(base_url="http://x")),
                          allow_open=True)
    try:
        assert srv.RequestHandlerClass.allow_open is False
    finally:
        srv.server_close()


def test_stdio_transport(observer):
    app = mcp.ProcessStudioMCP(mcp.ObserverClient(base_url=observer, key_hex=""))
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "nope"},
    ]
    out = io.StringIO()
    mcp.run_stdio(app, stdin=io.StringIO("\n".join(json.dumps(m) for m in lines) + "\n"), stdout=out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3]          # the notification got no reply
    assert replies[0]["result"]["protocolVersion"] == "2024-11-05"
    assert len(replies[1]["result"]["tools"]) == 4
    assert replies[2]["error"]["code"] == -32601


def test_signed_write_verifies_against_the_edge(monkeypatch):
    """The MCP server's one write carries QUIPU's edge canonical signature."""
    from src.quipu import edge_client
    seen = {}

    class _Resp(io.BytesIO):
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def opener(req, timeout=None):
        seen["headers"] = {k.title(): v for k, v in req.header_items()}
        seen["body"] = req.data
        return _Resp(b'{"ok": true, "held": true}')

    key = "cd" * 32
    c = mcp.ObserverClient(base_url="http://quipu:7100", source="vscode-agent", key_hex=key, opener=opener)
    code, data = c.post_signed("/process_studio/propose", {"tool": "t"})
    h = seen["headers"]
    want = edge_client.sign_headers(key, "vscode-agent", "POST", "/process_studio/propose", seen["body"],
                                    idem=h["X-Quipu-Idempotency"], ts=int(h["X-Quipu-Timestamp"]))
    assert h["X-Quipu-Signature"] == want["X-Quipu-Signature"] and data["held"] is True


def test_vscode_agent_source_profile():
    assert osvc._canonical_source("vscode") == "vscode-agent"
    assert osvc._canonical_source("open-vsx") == "vscode-agent"
    assert osvc.SOURCE_PROFILES["vscode-agent"]["kind"] == "agentic_ide"
