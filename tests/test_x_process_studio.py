"""x_ProcessStudio holon — v-ADMIN stages, the operator authorises, r-ADMIN
realises through Activepieces MCP; the observer routes sit behind the edge."""
import io
import json

import pytest

from src.quipu import observer_service as osvc
from src.quipu import x_process_studio as ps
from src.quipu.qpsi import edge_admission


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _FakeMCP:
    """Stands in for ActivepiecesMCP in realise(): records calls or raises."""

    def __init__(self, result=None, raises=None):
        self.result = result if result is not None else {"content": [{"type": "text", "text": "done"}]}
        self.raises = raises
        self.calls = []

    def available(self):
        return True

    def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments or {})))
        if self.raises is not None:
            raise self.raises
        return self.result


class _FakeResponse:
    def __init__(self, body: bytes, headers=None):
        self._body = body
        self.headers = {k.title(): v for k, v in (headers or {}).items()}
        self.headers = _H(self.headers)

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _H(dict):
    def get(self, k, d=None):
        for kk, vv in self.items():
            if kk.lower() == str(k).lower():
                return vv
        return d


def _rpc_result(id_, result):
    return json.dumps({"jsonrpc": "2.0", "id": id_, "result": result}).encode()


def _signed(source="supply-chain-brain"):
    return type("D", (), {"source": source, "assurance": "signed"})()


# ---------------------------------------------------------------------------
# v-ADMIN: staging is the whole of its authority
# ---------------------------------------------------------------------------

def test_vadmin_propose_is_held_and_never_realised():
    p = ps.propose("stockopoly", "send_slack_message", {"channel": "#ops"}, reason="restock")
    assert p.status == ps.PROPOSED and p.proposal_id.startswith("act_")
    assert p.meta["schema"] == ps.SCHEMA_PROPOSE
    out = ps.realise(p.proposal_id, mcp=_FakeMCP())
    assert out["status"] == "held" and "V10-SEC-010" in out["error"]
    assert ps.get_proposal(p.proposal_id).status == ps.PROPOSED


def test_propose_requires_source_and_tool_and_is_bounded():
    with pytest.raises(ValueError):
        ps.propose("", "tool")
    with pytest.raises(ValueError):
        ps.propose("stockopoly", "")
    with pytest.raises(ValueError):
        ps.propose("stockopoly", "t", {"blob": "x" * (ps._MAX_BYTES + 1)})


def test_propose_from_vlm_carries_the_signal():
    p = ps.propose_from_vlm("perceptopoly", "create_task",
                            {"ocrText": "PN 12-34", "confidence": 0.91},
                            reason="low-confidence part")
    assert p.meta["vlm"]["ocrText"] == "PN 12-34"
    assert p.meta["schema"] == ps.SCHEMA_PROPOSE


# ---------------------------------------------------------------------------
# the operator's edge: authorise / reject (V10-SEC-010)
# ---------------------------------------------------------------------------

def test_authorise_without_grant_ref_is_held_at_the_edge():
    p = ps.propose("burgerphone", "send_sms", reason="order ready")
    out = ps.authorise(p.proposal_id, "operator_admin", "")
    assert out["status"] == "held" and "V10-SEC-010" in out["why"]
    assert ps.get_proposal(p.proposal_id).status == ps.PROPOSED   # still in band


def test_authorise_then_realise_through_mcp():
    p = ps.propose("stockopoly", "start_flow", {"flow": "replenish"})
    out = ps.authorise(p.proposal_id, "operator_admin", "IT505-CR-0042")
    assert out["status"] == "authorised" and "IT505-CR-0042" in out["why"]
    assert out["config_digest"]
    got = ps.get_proposal(p.proposal_id)
    assert got.status == ps.AUTHORISED and got.grant_ref == "IT505-CR-0042"

    mcp = _FakeMCP(result={"content": [{"type": "text", "text": "flow started"}]})
    done = ps.realise(p.proposal_id, mcp=mcp)
    assert done["status"] == "realised"
    assert mcp.calls == [("start_flow", {"flow": "replenish"})]
    again = ps.realise(p.proposal_id, mcp=mcp)
    assert again["status"] == "already_realised" and len(mcp.calls) == 1


def test_reject_closes_the_proposal():
    p = ps.propose("perceptopoly", "open_door")
    out = ps.authorise(p.proposal_id, "operator_admin", "", approve=False, why="not now")
    assert out["status"] == "rejected"
    assert ps.get_proposal(p.proposal_id).status == ps.REJECTED
    assert ps.realise(p.proposal_id, mcp=_FakeMCP())["status"] == "rejected"


def test_transport_failure_keeps_authorised_for_retry():
    p = ps.propose("stockopoly", "t")
    ps.authorise(p.proposal_id, "op", "CR-1")
    out = ps.realise(p.proposal_id, mcp=_FakeMCP(raises=ps.MCPUnavailable("down")))
    assert out["status"] == "retry"
    got = ps.get_proposal(p.proposal_id)
    assert got.status == ps.AUTHORISED and got.attempts == 1 and "down" in got.last_error


def test_tool_refusal_marks_failed():
    p = ps.propose("stockopoly", "t")
    ps.authorise(p.proposal_id, "op", "CR-2")
    out = ps.realise(p.proposal_id, mcp=_FakeMCP(raises=ps.MCPError("MCP error -32602: bad args")))
    assert out["status"] == "failed"
    assert ps.get_proposal(p.proposal_id).status == ps.FAILED


def test_unconfigured_mcp_holds_realisation(monkeypatch):
    monkeypatch.delenv(ps.MCP_URL_ENV, raising=False)
    p = ps.propose("stockopoly", "t")
    ps.authorise(p.proposal_id, "op", "CR-3")
    out = ps.realise(p.proposal_id)                      # default client, no URL
    assert out["status"] == "held" and ps.MCP_URL_ENV in out["error"]
    assert ps.get_proposal(p.proposal_id).status == ps.AUTHORISED


def test_realise_ready_runs_oldest_first(tmp_path, monkeypatch):
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "studio.sqlite"))   # own ledger
    a = ps.propose("stockopoly", "t1"); ps.authorise(a.proposal_id, "op", "CR-4")
    b = ps.propose("stockopoly", "t2"); ps.authorise(b.proposal_id, "op", "CR-4")
    mcp = _FakeMCP()
    results = ps.realise_ready(mcp=mcp)
    assert [r["status"] for r in results] == ["realised", "realised"]
    assert [c[0] for c in mcp.calls] == ["t1", "t2"]
    s = ps.stats()
    assert s["proposals"][ps.REALISED] == 2 and s["holon"] == "x_process_studio"


# ---------------------------------------------------------------------------
# the MCP client (Streamable HTTP, JSON-RPC 2.0)
# ---------------------------------------------------------------------------

def test_mcp_client_handshake_tools_and_call():
    sent = []

    def opener(req, timeout=None):
        payload = json.loads(req.data.decode())
        sent.append((dict(req.headers), payload))
        method = payload.get("method")
        if method == "initialize":
            return _FakeResponse(_rpc_result(payload["id"], {"serverInfo": {"name": "ap"}}),
                                 headers={"Mcp-Session-Id": "s-123"})
        if method == "notifications/initialized":
            return _FakeResponse(b"")
        if method == "tools/list":
            return _FakeResponse(_rpc_result(payload["id"], {"tools": [{"name": "send_slack_message"}]}))
        if method == "tools/call":
            return _FakeResponse(_rpc_result(payload["id"], {"content": [{"type": "text", "text": "ok"}]}))
        raise AssertionError(method)

    c = ps.ActivepiecesMCP(url="https://cloud.activepieces.com/api/v1/mcp/abc/mcp",
                           token="tok", opener=opener)
    tools = c.list_tools()
    assert tools == [{"name": "send_slack_message"}]
    out = c.call_tool("send_slack_message", {"text": "hi"})
    assert out["content"][0]["text"] == "ok"
    init_headers = sent[0][0]
    assert init_headers.get("Authorization") == "Bearer tok"
    call_headers, call_payload = sent[-1]
    assert call_headers.get("Mcp-session-id") == "s-123"       # session carried
    assert call_payload["params"] == {"name": "send_slack_message", "arguments": {"text": "hi"}}


def test_mcp_client_parses_sse_frames():
    def opener(req, timeout=None):
        payload = json.loads(req.data.decode())
        if payload.get("method") == "notifications/initialized":
            return _FakeResponse(b"")
        body = ("event: message\n"
                f'data: {{"jsonrpc":"2.0","id":{payload["id"]},"result":{{"tools":[{{"name":"t"}}]}}}}\n\n')
        return _FakeResponse(body.encode())

    c = ps.ActivepiecesMCP(url="https://ap.example/mcp", opener=opener)
    assert c.list_tools() == [{"name": "t"}]


def test_mcp_rpc_error_raises_mcperror():
    def opener(req, timeout=None):
        payload = json.loads(req.data.decode())
        if payload.get("method") == "notifications/initialized":
            return _FakeResponse(b"")
        body = json.dumps({"jsonrpc": "2.0", "id": payload["id"],
                           "error": {"code": -32602, "message": "nope"}}).encode()
        return _FakeResponse(body)

    c = ps.ActivepiecesMCP(url="https://ap.example/mcp", opener=opener)
    with pytest.raises(ps.MCPError):
        c.list_tools()


def test_mcp_refuses_plain_http_to_public_hosts():
    c = ps.ActivepiecesMCP(url="http://ap.example.com/mcp", opener=lambda *a, **k: None)
    with pytest.raises(ps.MCPUnavailable):
        c.list_tools()
    assert ps._plain_http_ok("127.0.0.1") and ps._plain_http_ok("activepieces")
    assert ps._plain_http_ok("10.0.0.7") and not ps._plain_http_ok("93.184.216.34")
    assert c.available() and ps.ActivepiecesMCP(url="").available() is False


# ---------------------------------------------------------------------------
# observer routes: the edge in front, the operator gate behind
# ---------------------------------------------------------------------------

def test_endpoint_propose_requires_known_source():
    code, payload = osvc._ps_propose({"source": "nobody", "tool": "t"}, None)
    assert code == 400
    code, payload = osvc._ps_propose({"source": "stockopoly", "tool": "start_flow",
                                      "arguments": {"flow": "f"}}, None)
    assert code == 200 and payload["held"] is True
    assert payload["proposal"]["status"] == ps.PROPOSED


def test_endpoint_authorise_is_operator_gated(monkeypatch):
    p = ps.propose("stockopoly", "t")
    monkeypatch.delenv(ps.OPERATORS_ENV, raising=False)
    code, payload = osvc._ps_authorise({"source": "supply-chain-brain",
                                        "proposal_id": p.proposal_id,
                                        "grant_ref": "CR-9"}, _signed())
    assert code == 403 and "CLI" in payload["error"]

    monkeypatch.setenv(ps.OPERATORS_ENV, "supply-chain-brain")
    unsigned = type("D", (), {"source": "supply-chain-brain", "assurance": "recorded"})()
    code, _ = osvc._ps_authorise({"source": "supply-chain-brain",
                                  "proposal_id": p.proposal_id, "grant_ref": "CR-9"}, unsigned)
    assert code == 403

    code, payload = osvc._ps_authorise({"source": "supply-chain-brain",
                                        "proposal_id": p.proposal_id,
                                        "grant_ref": "CR-9"}, _signed())
    assert code == 200 and payload["status"] == "authorised"
    code, payload = osvc._ps_authorise({"source": "perceptopoly",
                                        "proposal_id": p.proposal_id,
                                        "grant_ref": "CR-9"}, _signed("perceptopoly"))
    assert code == 403                                      # not an operator source


def test_endpoint_realise_and_status(monkeypatch):
    monkeypatch.setenv(ps.OPERATORS_ENV, "supply-chain-brain")
    monkeypatch.delenv(ps.MCP_URL_ENV, raising=False)
    p = ps.propose("burgerphone", "send_sms", {"to": "+1"})
    osvc._ps_authorise({"source": "supply-chain-brain", "proposal_id": p.proposal_id,
                        "grant_ref": "CR-10"}, _signed())
    code, payload = osvc._ps_realise({"source": "supply-chain-brain",
                                      "proposal_id": p.proposal_id}, _signed())
    assert code == 200 and payload["status"] == "held"      # MCP unconfigured ⇒ held
    status = osvc._ps_status()
    assert status["holon"] == "x_process_studio" and status["mcp_configured"] is False
    assert any(r["proposal_id"] == p.proposal_id for r in status["recent"])
    assert status["operator_sources"] == ["supply-chain-brain"]


def test_process_studio_routes_are_behind_the_edge():
    assert ("POST", "/process_studio/propose") in edge_admission.MUTATING
    assert ("POST", "/process_studio/authorise") in edge_admission.MUTATING
    assert ("POST", "/process_studio/realise") in edge_admission.MUTATING


# ---------------------------------------------------------------------------
# burgerphone joins the fleet
# ---------------------------------------------------------------------------

def test_burgerphone_source_profile_and_routing():
    from src.quipu import mesh_slm
    prof = osvc.SOURCE_PROFILES["burgerphone"]
    assert prof["axis"] == "body" and prof["siblings"] == ["perceptopoly", "stockopoly"]
    assert osvc._canonical_source("BurgerPhone") == "burgerphone"
    assert osvc._canonical_source("burger phone") == "burgerphone"
    assert osvc._canonical_source("burger_phone") == "burgerphone"
    assert mesh_slm._axis_for_source(prof["axis_source"]) == 3   # ("wiki", 3): body
