"""Edge device (Termux) -> the same x_ProcessStudio MCP the extension registers.

Run against a real MCP server and a real observer; the setup script runs as a real subprocess.
"""
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from src.quipu import hub_client
from src.quipu import observer_service as osvc
from src.quipu import process_studio_mcp as mcp
from src.quipu import x_process_studio as ps

CLIENT = Path(__file__).resolve().parent.parent / "clients" / "termux" / "quipu_mcp.py"
SETUP = Path(__file__).resolve().parent.parent / "clients" / "termux" / "edge_setup.sh"
_spec = importlib.util.spec_from_file_location("quipu_mcp_edge", CLIENT)
edge = importlib.util.module_from_spec(_spec)
sys.modules["quipu_mcp_edge"] = edge
_spec.loader.exec_module(edge)

TOKEN = "edge-token-" + "z" * 24          # distinctive: it must never appear in any output


@pytest.fixture(autouse=True)
def _own_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "brain.sqlite"))
    monkeypatch.setenv(mcp.TOKEN_ENV, TOKEN)
    monkeypatch.delenv(mcp.ORIGINS_ENV, raising=False)


@pytest.fixture()
def server():
    obs = ThreadingHTTPServer(("127.0.0.1", 0), osvc.ObserverHandler)
    threading.Thread(target=obs.serve_forever, daemon=True).start()
    app = mcp.ProcessStudioMCP(
        mcp.ObserverClient(base_url=f"http://127.0.0.1:{obs.server_address[1]}", key_hex=""),
        hub_client.HubCoreClient(base_url="", org_id=""))               # HubCore off: the voice is not offered
    srv = mcp.make_server("127.0.0.1", 0, app=app)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/mcp"
    finally:
        for s in (srv, obs):
            s.shutdown()
            s.server_close()


def _env(url, token=TOKEN, **extra):
    return {"QUIPU_MCP_URL": url, "QUIPU_MCP_TOKEN": token, **extra}


def run(url, *args, token=TOKEN, capsys=None):
    rc = edge.main(list(args), env=_env(url, token))
    out = capsys.readouterr()
    return rc, out.out, out.err


# ------------------------------------------------------------------ doctor: the same surface
def test_doctor_sees_exactly_the_tools_the_extension_lists(server, capsys):
    rc, out, err = run(server, "doctor", capsys=capsys)
    assert rc == 0, out + err
    assert "the same 5 v-ADMIN tools the extension lists" in out
    assert "no authorise/realise tool" in out and "voice is off" in out
    names = {t["name"] for t in edge.McpEdge(server, TOKEN).tools()}
    assert names == edge.EXPECTED_TOOLS
    assert not [n for n in names if any(f in n for f in edge.FORBIDDEN_FRAGMENTS)]


def test_doctor_would_flag_a_tool_that_could_decide_or_run_something(server, capsys, monkeypatch):
    monkeypatch.setattr(mcp, "TOOLS", mcp.TOOLS + [{**mcp.TOOLS[0], "name": "process_studio_authorise"}])
    rc, out, _ = run(server, "doctor", capsys=capsys)
    assert rc == 1 and "FAIL  a tool that could decide or run something is exposed" in out


def test_the_token_is_never_printed(server, capsys, tmp_path):
    for cmd in (["doctor"], ["tools"], ["status"], ["list"], ["voice"], ["call", "hub_actions"]):
        rc, out, err = run(server, *cmd, capsys=capsys)
        assert TOKEN not in out + err, cmd
    rc, out, err = run(server, "doctor", token="wrong-" + TOKEN, capsys=capsys)
    assert rc == 1 and "401" in out and TOKEN not in out + err


# ------------------------------------------------------------------ configuration
def test_missing_or_unsafe_configuration_is_refused_with_the_fix(capsys):
    assert edge.main(["doctor"], env={}) == 3
    assert "QUIPU_MCP_URL is not set" in capsys.readouterr().err
    assert edge.main(["doctor"], env={"QUIPU_MCP_URL": "https://q.example.com/mcp"}) == 3
    assert "no token" in capsys.readouterr().err
    assert edge.main(["doctor"], env=_env("http://q.example.com/mcp")) == 3
    assert "plain http to a public address" in capsys.readouterr().err
    assert edge.main(["doctor"], env=_env("ftp://x/mcp")) == 3


def test_the_token_can_come_from_a_file(server, capsys, tmp_path):
    f = tmp_path / "tok"
    f.write_text(TOKEN + "\n")
    rc = edge.main(["status"], env={"QUIPU_MCP_URL": server, "QUIPU_MCP_TOKEN_FILE": str(f)})
    assert rc == 0 and "x_process_studio" in capsys.readouterr().out
    assert edge.main(["status"], env={"QUIPU_MCP_URL": server, "QUIPU_MCP_TOKEN_FILE": str(tmp_path / "nope")}) == 3


def test_plain_http_rule():
    for ok in ("127.0.0.1", "10.1.2.3", "192.168.1.9", "quipu", "hub.local"):
        assert edge.plain_http_ok(ok), ok
    for bad in ("93.184.216.34", "example.com", ""):
        assert not edge.plain_http_ok(bad), bad


# ------------------------------------------------------------------ using QUIPU from the edge
def test_status_list_and_call(server, capsys):
    rc, out, _ = run(server, "status", capsys=capsys)
    assert rc == 0 and json.loads(out)["holon"] == "x_process_studio"
    rc, out, _ = run(server, "call", "hub_actions", capsys=capsys)
    assert rc == 0 and "hub.body_directive" in out
    rc, out, err = run(server, "call", "process_studio_authorise", capsys=capsys)
    assert rc == 1 and "unknown tool" in err                         # there is nothing to authorise with, here or in the extension
    rc, out, err = run(server, "call", "hub_actions", "{not json", capsys=capsys)
    assert rc == 3 and "not JSON" in err


def test_an_edge_proposal_is_held_and_labelled(server, capsys):
    rc, out, _ = run(server, "propose", "send_sms", "--reason", "order is ready",
                     "--args", json.dumps({"to": "+15555550100"}), "--device", "pixel-edge", capsys=capsys)
    assert rc == 0
    pid = json.loads(out)["proposal"]["proposal_id"]
    p = ps.get_proposal(pid)
    assert p.status == ps.PROPOSED and p.source == "vscode-agent"      # held; signed by the MCP, as for the extension
    assert p.meta["via"] == "process_studio_mcp" and p.meta["edge_device"] == "pixel-edge"
    rc, out, _ = run(server, "list", "--status", "PROPOSED", capsys=capsys)
    assert pid in out


def test_an_edge_hub_action_is_validated_like_any_other(server, capsys):
    rc, out, _ = run(server, "propose", "hub.open_the_gates", "--reason", "x", capsys=capsys)
    assert rc == 1 and "no such hub action" in out
    ok = {"title": "Check line 3", "do_this": "Walk it", "owner_role": "Ops"}
    rc, out, _ = run(server, "propose", "hub.body_directive", "--reason", "r", "--args", json.dumps(ok), capsys=capsys)
    assert rc == 0 and json.loads(out)["held"] is True


def test_the_voice_is_refused_until_a_human_admin_turns_it_on(server, capsys):
    rc, out, err = run(server, "voice", capsys=capsys)
    assert rc == 1 and ("voice is off" in out + err or "unknown tool" in out + err)


def test_unreachable_server_is_reported_not_raised(capsys):
    rc, out, err = run("http://127.0.0.1:1/mcp", "status", capsys=capsys)
    assert rc == 1 and "cannot reach" in err


def test_sse_framed_replies_parse():
    body = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n\n'
    assert edge._parse(body)["result"] == {"tools": []}
    assert edge._parse("") is None and edge._parse("garbage") is None
    assert edge.health_url("https://q.example.com/mcp?x=1") == "https://q.example.com/health"


# ------------------------------------------------------------------ edge_setup.sh, as a subprocess
pytestmark_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")


def _setup(home, *args, extra=None):
    e = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]), "HOME": str(home)}
    e.update(extra or {})
    return subprocess.run(["sh", str(SETUP), *args], capture_output=True, text=True, env=e, timeout=60,
                          stdin=subprocess.DEVNULL)


@pytestmark_sh
def test_setup_stores_the_connection_privately_and_checks_it(server, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    tok = tmp_path / "t"
    tok.write_text(TOKEN + "\n")
    r = _setup(home, "--url", server, "--token-file", str(tok), "--no-prompt")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "the same 5 v-ADMIN tools" in r.stdout and "connected" in r.stdout
    envf, tokf = home / ".config/quipu/mcp.env", home / ".config/quipu/mcp.token"
    assert stat.S_IMODE(envf.stat().st_mode) == 0o600 and stat.S_IMODE(tokf.stat().st_mode) == 0o600
    assert TOKEN in tokf.read_text() and TOKEN not in envf.read_text()     # the token lives only in its own file
    assert f"QUIPU_MCP_URL={server}" in envf.read_text()
    assert TOKEN not in r.stdout + r.stderr
    # the stored config is what quipu_mcp.py now uses with nothing in the environment
    r2 = subprocess.run([sys.executable, str(CLIENT), "status"], capture_output=True, text=True, timeout=60,
                        env={"PATH": "/usr/bin:/bin", "HOME": str(home)})
    assert r2.returncode == 0 and "x_process_studio" in r2.stdout and TOKEN not in r2.stdout + r2.stderr


@pytestmark_sh
def test_setup_refuses_what_it_should(server, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    tok = tmp_path / "t"
    tok.write_text(TOKEN)
    assert "never taken on the command line" in _setup(home, "--url", server, "--token", "abc").stderr
    assert _setup(home, "--url", server, "--token=abc").returncode == 3
    r = _setup(home, "--url", "http://example.com/mcp", "--token-file", str(tok), "--no-prompt")
    assert r.returncode == 3 and "must be https" in r.stderr
    assert "no URL" in _setup(home, "--token-file", str(tok)).stderr
    r = _setup(home, "--url", server, "--no-prompt")
    assert r.returncode == 3 and "no token" in r.stderr
    assert _setup(home, "--url", server, "--token-file", str(tmp_path / "missing")).returncode == 3
    assert not (home / ".config/quipu/mcp.token").exists()                 # nothing half-written
    assert _setup(home, "--url", server, "--token-file", str(tok), "--no-prompt").returncode == 0
    r = _setup(home, "--url", server, "--token-file", str(tok), "--no-prompt")
    assert r.returncode == 3 and "already exists" in r.stderr              # no silent overwrite
    assert _setup(home, "--url", server, "--token-file", str(tok), "--no-prompt", "--force").returncode == 0


@pytestmark_sh
def test_setup_with_the_token_in_the_environment_and_a_wrong_token(server, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    ok = _setup(home, "--url", server, "--no-prompt", extra={"QUIPU_MCP_TOKEN": TOKEN})
    assert ok.returncode == 0 and TOKEN not in ok.stdout + ok.stderr
    bad = _setup(home, "--url", server, "--no-prompt", "--force", extra={"QUIPU_MCP_TOKEN": "wrong-token"})
    assert bad.returncode == 1 and "401" in bad.stdout and "saved, but the check failed" in bad.stderr
