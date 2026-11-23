"""The Holon Hub worker: VNN output -> staged hub action -> (human authorises) -> HubCore task,
and the v-ADMIN voice tool that exists only while a human Admin has it on.

HubCore is stood in for by a small in-test server that speaks its session / voice / task
endpoints (including its 401 and 403); the real HubCore is exercised separately.
"""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.quipu import hub_actions, hub_client, vnn_actions
from src.quipu import observer_service as osvc
from src.quipu import process_studio_mcp as mcp
from src.quipu import x_process_studio as ps
from src.quipu.qpsi import edge_admission


@pytest.fixture(autouse=True)
def _own_ledger(tmp_path, monkeypatch):
    """The suite's brain is session-wide; these tests write ledger rows and must not see each other's."""
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "brain.sqlite"))


# --------------------------------------------------------------------------- stand-in HubCore
class FakeHub:
    def __init__(self):
        self.sessions, self.tasks, self.voices = [], [], []
        self.voice_on, self.persona = False, "Perceptopoly"
        self.task_status = 201            # set to 403 / 500 to simulate a refusal / outage
        self.expire_next = False          # 401 once, to exercise token refresh
        self.requests = 0
        self.url = ""


@pytest.fixture()
def hub():
    state = FakeHub()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n).decode() or "{}") if n else {}

        def _authed(self):
            state.requests += 1
            if state.expire_next:
                state.expire_next = False
                self._send(401, {"error": "token expired"})
                return False
            if not (self.headers.get("Authorization") or "").startswith("Bearer tok-"):
                self._send(401, {"error": "token required"})
                return False
            return True

        def do_POST(self):
            body = self._body()
            if self.path == "/api/v1/session":
                state.sessions.append(body)
                self._send(201, {"token": f"tok-{len(state.sessions)}", "expiresAt": 4102444800})
            elif self.path == "/api/v1/tasks" and self._authed():
                if state.task_status != 201:
                    self._send(state.task_status, {"error": "refused" if state.task_status < 500 else "boom"})
                    return
                state.tasks.append(body)
                self._send(201, {"id": f"task-{len(state.tasks)}", "state": "CREATED", "kind": body["kind"]})
            elif self.path == "/api/v1/vadmin/voice" and self._authed():
                if not state.voice_on:
                    self._send(403, {"error": "the voice tool is off; a human Admin must turn it on"})
                    return
                state.voices.append(body)
                self._send(200, {"spoken": True, "persona": state.persona, "kind": body["kind"],
                                 "text": f"{state.persona}, a synthetic assistant, said {body['kind']}.",
                                 "disclosure": f"{state.persona} is a synthetic assistant."})
            else:
                self._send(404, {"error": "not found"})

        def do_GET(self):
            if self.path == "/api/v1/vadmin/tools" and self._authed():
                self._send(200, {"tools": {"voice": {"enabled": state.voice_on, "effective": state.voice_on,
                                                     "persona": state.persona if state.voice_on else None}}})
            else:
                self._send(404, {"error": "not found"})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()


def _client(hub, **kw):
    return hub_client.HubCoreClient(base_url=hub.url, org_id="org-test", **kw)


DIRECTIVE = {"title": "Inspect pump line 3", "do_this": "Walk the line and log any vibration",
             "owner_role": "Ops", "severity": "watch", "priority": 0.6}


# --------------------------------------------------------------------------- the catalogue
def test_catalogue_is_fixed_and_documented():
    pub = hub_actions.catalogue_public()
    assert [a["tool"] for a in pub] == ["hub.body_directive"] and "title" in pub[0]["fields"]
    assert "does not move or switch anything" in pub[0]["effect"]


def test_validate_cleans_and_bounds():
    clean, err = hub_actions.validate("hub.body_directive", {**DIRECTIVE, "title": "A\x00 title\nwith  breaks"})
    assert err == "" and clean["title"] == "A title with breaks" and clean["priority"] == 0.6
    assert clean["severity"] == "watch" and clean["why_it_matters"] == ""


@pytest.mark.parametrize("tool,args,why", [
    ("hub.open_the_gates", DIRECTIVE, "no such hub action"),                 # not in the closed catalogue
    ("hub.body_directive", {**DIRECTIVE, "shell": "rm -rf /"}, "unknown argument"),   # nothing rides along
    ("hub.body_directive", {**DIRECTIVE, "title": " "}, "title is required"),
    ("hub.body_directive", {**DIRECTIVE, "do_this": ""}, "do_this is required"),
    ("hub.body_directive", {**DIRECTIVE, "owner_role": "Root"}, "owner_role"),
    ("hub.body_directive", {**DIRECTIVE, "severity": "apocalyptic"}, "severity"),
    ("hub.body_directive", {**DIRECTIVE, "priority": 1.5}, "priority"),
    ("hub.body_directive", {**DIRECTIVE, "priority": True}, "priority"),
    ("hub.body_directive", {**DIRECTIVE, "fingerprint": "has space"}, "fingerprint"),
    ("hub.body_directive", {**DIRECTIVE, "evidence": {"blob": "x" * 5000}}, "evidence is over"),
    ("hub.body_directive", {**DIRECTIVE, "evidence": "text"}, "evidence must be an object"),
    ("hub.body_directive", "not an object", "arguments must be an object"),
])
def test_validate_refuses(tool, args, why):
    clean, err = hub_actions.validate(tool, args)
    assert clean is None and why in err


def test_propose_validates_hub_actions_and_stores_the_clean_form():
    with pytest.raises(ValueError, match="no such hub action"):
        ps.propose("vscode-agent", "hub.open_the_gates", {})
    with pytest.raises(ValueError, match="unknown argument"):
        ps.propose("vscode-agent", "hub.body_directive", {**DIRECTIVE, "x": 1})
    p = ps.propose("vscode-agent", "hub.body_directive", {**DIRECTIVE, "title": "T\nitle"}, reason="why")
    assert ps.get_proposal(p.proposal_id).arguments["title"] == "T itle" and p.status == ps.PROPOSED
    assert ps.propose("vscode-agent", "send_sms", {"anything": "goes"}).status == ps.PROPOSED   # non-hub: unchanged


# --------------------------------------------------------------------------- realising into HubCore
def _authorised_hub_proposal(**kw):
    p = ps.propose("quipu-vnn", "hub.body_directive", DIRECTIVE, reason="because", **kw)
    out = ps.authorise(p.proposal_id, "operator_admin", "IT505-CR-0042")
    assert out["status"] == "authorised"
    return p.proposal_id


def test_an_authorised_hub_action_becomes_a_hubcore_task_with_provenance(hub):
    pid = _authorised_hub_proposal()
    out = ps.realise(pid, hub=_client(hub))
    assert out["status"] == "realised" and out["result"]["hubcore_task"]["state"] == "CREATED"
    task = hub.tasks[0]
    assert task["kind"] == "body.directive" and task["orgId"] == "org-test"
    assert task["payload"]["directive"]["title"] == "Inspect pump line 3"
    assert task["payload"]["provenance"] == {"proposal_id": pid, "proposed_by": "quipu-vnn",
                                             "authorised_by": "operator_admin",
                                             "grant_ref": "IT505-CR-0042", "reason": "because"}
    assert ps.realise(pid, hub=_client(hub))["status"] == "already_realised" and len(hub.tasks) == 1


def test_the_bridge_only_ever_asks_for_the_self_service_role(hub):
    _client(hub).create_task("body.directive", {"x": 1})
    assert hub.sessions and all(s["role"] == "Requester" for s in hub.sessions)


def test_an_unauthorised_proposal_never_reaches_hubcore(hub):
    p = ps.propose("quipu-vnn", "hub.body_directive", DIRECTIVE, reason="r")
    out = ps.realise(p.proposal_id, hub=_client(hub))
    assert out["status"] == "held" and "V10-SEC-010" in out["error"] and hub.tasks == []
    ps.authorise(p.proposal_id, "op", "")                       # a missing grant_ref holds it at the edge
    assert ps.realise(p.proposal_id, hub=_client(hub))["status"] == "held" and hub.tasks == []


def test_unconfigured_hubcore_holds_and_keeps_the_authorisation(monkeypatch):
    monkeypatch.delenv(hub_client.URL_ENV, raising=False)
    pid = _authorised_hub_proposal()
    out = ps.realise(pid)
    assert out["status"] == "held" and hub_client.URL_ENV in out["error"]
    assert ps.get_proposal(pid).status == ps.AUTHORISED


def test_hubcore_refusal_is_permanent_and_an_outage_is_retried(hub):
    hub.task_status = 403
    pid = _authorised_hub_proposal()
    assert ps.realise(pid, hub=_client(hub))["status"] == "failed"
    assert ps.get_proposal(pid).status == ps.FAILED
    hub.task_status = 500
    pid2 = _authorised_hub_proposal()
    out = ps.realise(pid2, hub=_client(hub))
    assert out["status"] == "retry" and ps.get_proposal(pid2).status == ps.AUTHORISED
    hub.task_status = 201
    assert ps.realise(pid2, hub=_client(hub))["status"] == "realised"      # the retry succeeds


def test_realise_ready_routes_each_proposal_to_its_executor(hub):
    pid = _authorised_hub_proposal()
    results = ps.realise_ready(hub=_client(hub))
    assert [r["status"] for r in results] == ["realised"] and ps.get_proposal(pid).status == ps.REALISED


# --------------------------------------------------------------------------- the HubCore client
def test_token_is_cached_and_refreshed_once_on_401(hub):
    c = _client(hub)
    c.create_task("k", {})
    c.create_task("k", {})
    assert len(hub.sessions) == 1                                   # cached
    hub.expire_next = True
    c.create_task("k", {})
    assert len(hub.sessions) == 2 and len(hub.tasks) == 3           # one refresh, then it went through


def test_client_refuses_what_it_should():
    with pytest.raises(hub_client.HubCoreUnavailable, match="not configured"):
        hub_client.HubCoreClient(base_url="", org_id="").create_task("k", {})
    with pytest.raises(hub_client.HubCoreUnavailable, match="plain http to a public address"):
        hub_client.HubCoreClient(base_url="http://hub.example.com", org_id="o").vadmin_tools()
    assert hub_client.HubCoreClient(base_url="http://loadopoly-hubcore:8000", org_id="o").configured()


def test_voice_effective_fails_closed():
    assert hub_client.HubCoreClient(base_url="", org_id="").voice_effective() == (False, "")
    assert hub_client.HubCoreClient(base_url="http://127.0.0.1:1", org_id="o").voice_effective() == (False, "")


# --------------------------------------------------------------------------- VNN -> staged actions
REPORT_BAD = {"at": 1700000000.0,
              "physical": {"div_linf": 0.02, "kinetic_energy": 1.0},
              "ns_diagnostics": {"stable_single_step": False, "cfl_adv": 0.9, "fo_visc": 0.2, "recommended_substeps": 3},
              "vnn_diagnostics": {"stable": False, "graph_cfl": 2.4}}
REPORT_OK = {"at": 1700000001.0, "physical": {"div_linf": 1e-9},
             "ns_diagnostics": {"stable_single_step": True}, "vnn_diagnostics": {"stable": True}}


def test_default_rules_are_valid_and_cap_severity():
    rules, err = vnn_actions.validate_rules(vnn_actions.DEFAULT_RULES)
    assert err == "" and {r["directive"]["severity"] for r in rules} <= set(vnn_actions.RULE_SEVERITIES)


def test_a_healthy_report_fires_nothing_and_a_bad_one_fires_every_matching_rule():
    assert vnn_actions.evaluate(REPORT_OK) == []
    fired = {t["rule"]: t for t in vnn_actions.evaluate(REPORT_BAD)}
    assert set(fired) == {"vnn.graph.unstable", "vnn.ns.divergence", "vnn.ns.substeps"}
    a = fired["vnn.graph.unstable"]["arguments"]
    assert "graph CFL 2.4" in a["why_it_matters"] and a["fingerprint"] == "vnn:vnn.graph.unstable"
    assert a["evidence"]["observed"] is False and a["evidence"]["report_at"] == 1700000000.0
    assert "div_linf" in fired["vnn.ns.divergence"]["reason"] or "0.02" in fired["vnn.ns.divergence"]["reason"]


def test_a_missing_or_non_numeric_reading_never_fires_a_threshold():
    assert vnn_actions.evaluate({"physical": {}}) == []
    assert vnn_actions.evaluate({"physical": {"div_linf": "huge"}}) == []
    assert vnn_actions.evaluate({"physical": {"div_linf": True}}) == []


@pytest.mark.parametrize("mutate,why", [
    (lambda r: r.update(op="~="), "op must be"),
    (lambda r: r["directive"].update(severity="critical"), "never raises 'critical'"),
    (lambda r: r.update(path="Physical.DivLinf"), "dotted lower-case path"),
    (lambda r: r.update(value="big"), "value must be a number"),
    (lambda r: r["directive"].update(owner_role="Root"), "owner_role"),
    (lambda r: r["directive"].pop("do_this"), "do_this is required"),
    (lambda r: r.update(id="bad id"), "must be unique"),
])
def test_bad_rules_are_refused_at_load(mutate, why):
    rule = json.loads(json.dumps(vnn_actions.DEFAULT_RULES[1]))
    mutate(rule)
    rules, err = vnn_actions.validate_rules([rule])
    assert rules is None and why in err


def test_duplicate_ids_and_too_many_rules_are_refused():
    r = vnn_actions.DEFAULT_RULES[0]
    assert vnn_actions.validate_rules([r, r])[0] is None
    many = [{**json.loads(json.dumps(r)), "id": f"r{i}"} for i in range(vnn_actions.MAX_RULES + 1)]
    assert vnn_actions.validate_rules(many)[0] is None


def test_operator_rules_file_replaces_defaults_and_a_bad_file_is_loud(tmp_path, monkeypatch):
    good = {"rules": [{"id": "plant.ke", "path": "physical.kinetic_energy", "op": ">", "value": 0.5,
                       "directive": {"title": "Flow energy high ({physical.kinetic_energy})",
                                     "do_this": "Inspect line 3", "owner_role": "Ops", "severity": "watch"}}]}
    f = tmp_path / "rules.json"
    f.write_text(json.dumps(good))
    monkeypatch.setenv(vnn_actions.RULES_ENV, str(f))
    assert [r["id"] for r in vnn_actions.load_rules()] == ["plant.ke"]
    assert vnn_actions.evaluate(REPORT_BAD)[0]["arguments"]["title"] == "Flow energy high (1)"
    f.write_text(json.dumps({"rules": [{"id": "x"}]}))
    with pytest.raises(ValueError, match=vnn_actions.RULES_ENV):
        vnn_actions.load_rules()                                    # never silently falls back to defaults
    f.write_text("not json")
    with pytest.raises(ValueError, match="cannot read"):
        vnn_actions.load_rules()


def test_stage_holds_proposals_executes_nothing_and_dedupes(hub):
    out = vnn_actions.stage(REPORT_BAD)
    assert len(out["staged"]) == 3 and out["skipped"] == []
    for s in out["staged"]:
        p = ps.get_proposal(s["proposal_id"])
        assert p.status == ps.PROPOSED and p.source == "quipu-vnn" and p.tool == "hub.body_directive"
    assert hub.tasks == [] and hub.sessions == []                    # staging touched nothing in Hub Holon
    again = vnn_actions.stage(REPORT_BAD)
    assert again["staged"] == [] and {s["why"] for s in again["skipped"]} == {"already open"}
    # once an open one is rejected, the condition may stage again
    ps.authorise(out["staged"][0]["proposal_id"], "op", "", approve=False, why="seen")
    assert len(vnn_actions.stage(REPORT_BAD)["staged"]) == 1


def test_stage_and_dry_run_without_a_report():
    assert vnn_actions.stage(None)["staged"] == [] and "no VNN report" in vnn_actions.stage(None)["note"]
    assert vnn_actions.dry_run(None)["triggered"] == []


def test_full_path_vnn_to_hubcore_through_a_human(hub):
    staged = vnn_actions.stage(REPORT_BAD)["staged"]
    pid = [s["proposal_id"] for s in staged if s["rule"] == "vnn.graph.unstable"][0]
    assert hub.tasks == []
    assert ps.authorise(pid, "operator_admin", "IT505-CR-0099")["status"] == "authorised"
    assert ps.realise(pid, hub=_client(hub))["status"] == "realised"
    assert len(hub.tasks) == 1 and hub.tasks[0]["payload"]["directive"]["owner_role"] == "IT"


# --------------------------------------------------------------------------- observer routes
def test_vnn_stage_route_is_operator_gated(monkeypatch):
    monkeypatch.delenv(ps.OPERATORS_ENV, raising=False)
    signed = type("D", (), {"source": "supply-chain-brain", "assurance": "signed"})()
    code, out = osvc._vnn_actions_stage({"source": "supply-chain-brain"}, signed)
    assert code == 403 and "CLI" in out["error"]
    monkeypatch.setenv(ps.OPERATORS_ENV, "supply-chain-brain")
    unsigned = type("D", (), {"source": "supply-chain-brain", "assurance": "recorded"})()
    assert osvc._vnn_actions_stage({}, unsigned)[0] == 403
    assert osvc._vnn_actions_stage({}, signed)[0] == 200
    assert ("POST", "/vnn/actions/stage") in edge_admission.MUTATING


def test_status_can_be_filtered_to_one_proposal_and_carries_the_reason():
    a = ps.propose("stockopoly", "t1", {}, reason="first")
    ps.propose("stockopoly", "t2", {}, reason="second")
    rows = osvc._ps_status({"id": [a.proposal_id]})["recent"]
    assert [r["proposal_id"] for r in rows] == [a.proposal_id] and rows[0]["reason"] == "first"


# --------------------------------------------------------------------------- the voice tool in the MCP
@pytest.fixture()
def server(hub):
    app = mcp.ProcessStudioMCP(mcp.ObserverClient(base_url="http://127.0.0.1:9", key_hex=""), _client(hub))
    app.hub._STATE_TTL = 0          # not used; the per-call check is what is under test
    return app


def _names(app):
    return {t["name"] for t in app.tools_list()}


def test_the_voice_is_not_offered_until_a_human_turns_it_on_and_goes_when_turned_off(hub):
    app = mcp.ProcessStudioMCP(mcp.ObserverClient(base_url="http://127.0.0.1:9", key_hex=""), _client(hub))
    assert "perceptopoly_voice" not in _names(app) and "hub_actions" in _names(app)
    hub.voice_on = True
    app.hub._state = (0.0, None)                                    # past the few-second cache
    assert "perceptopoly_voice" in _names(app)
    hub.voice_on = False
    app.hub._state = (0.0, None)
    assert "perceptopoly_voice" not in _names(app)


def test_calling_the_voice_while_it_is_off_says_why_and_never_asks_hubcore_to_speak(hub):
    app = mcp.ProcessStudioMCP(mcp.ObserverClient(base_url="http://127.0.0.1:9", key_hex=""), _client(hub))
    res = app.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": "perceptopoly_voice", "arguments": {"kind": "status"}}})["result"]
    assert res["isError"] is True and "human Admin" in res["content"][0]["text"] and hub.voices == []


def test_the_voice_is_rechecked_on_every_call_not_trusted_from_the_listing(hub):
    hub.voice_on = True
    app = mcp.ProcessStudioMCP(mcp.ObserverClient(base_url="http://127.0.0.1:9", key_hex=""), _client(hub))
    assert "perceptopoly_voice" in _names(app)                       # listed, and cached for a few seconds
    hub.voice_on = False                                             # the Admin turns it off
    app.hub._state = (0.0, None)
    res = app._voice({"kind": "status"})
    assert res["isError"] is True and hub.voices == []


def test_the_voice_speaks_for_a_proposal_using_only_its_stated_facts(hub, tmp_path):
    hub.voice_on = True
    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), osvc.ObserverHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        p = ps.propose("vscode-agent", "send_sms", {"to": "+15555550100", "secret": "api-key-123"},
                       reason="order is ready")
        app = mcp.ProcessStudioMCP(
            mcp.ObserverClient(base_url=f"http://127.0.0.1:{httpd.server_address[1]}", key_hex=""), _client(hub))
        res = app._voice({"kind": "proposal", "proposalId": p.proposal_id})
        assert res["isError"] is False
        assert res["structuredContent"]["persona"] == "Perceptopoly"
        sent = hub.voices[0]
        assert sent["kind"] == "proposal"
        assert sent["facts"] == {"source": "vscode-agent", "tool": "send_sms",
                                 "reason": "order is ready", "status": "PROPOSED"}
        assert "api-key-123" not in json.dumps(hub.voices) and "+1555" not in json.dumps(hub.voices)   # arguments never leave
        assert app._voice({"kind": "proposal", "proposalId": "act_nope"})["isError"] is True
        assert app._voice({"kind": "queue_entry", "entryId": "vadm-1"})["isError"] is False
        assert hub.voices[-1]["facts"] == {"entryId": "vadm-1"}
        assert app._voice({"kind": "horoscope"})["isError"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_hubcore_refusing_or_down_is_reported_not_papered_over(hub):
    hub.voice_on = True
    app = mcp.ProcessStudioMCP(mcp.ObserverClient(base_url="http://127.0.0.1:9", key_hex=""), _client(hub))
    hub.voice_on = True
    ok = app._voice({"kind": "status"})
    assert ok["isError"] is False and ok["structuredContent"]["text"].startswith("Perceptopoly, a synthetic assistant")
    dead = mcp.ProcessStudioMCP(mcp.ObserverClient(base_url="http://127.0.0.1:9", key_hex=""),
                                hub_client.HubCoreClient(base_url="http://127.0.0.1:1", org_id="o"))
    assert dead._voice({"kind": "status"})["isError"] is True              # HubCore down: refused, not faked
    assert "perceptopoly_voice" not in {t["name"] for t in dead.tools_list()}   # and not listed (fail closed)
