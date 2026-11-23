"""The integrations SDKs (Python + JS) against a live MCP server, and the site server."""
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from src.quipu import hub_client
from src.quipu import observer_service as osvc
from src.quipu import process_studio_mcp as mcp
from src.quipu import x_process_studio as ps

ROOT = Path(__file__).resolve().parent.parent / "integrations"
TOKEN = "sdk-token-" + "q" * 24


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


sdk = _load("loadopoly_integrations", ROOT / "loadopoly_integrations.py")
site = _load("integrations_server", ROOT / "server.py")


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "brain.sqlite"))
    monkeypatch.setenv(mcp.TOKEN_ENV, TOKEN)
    monkeypatch.delenv(mcp.ORIGINS_ENV, raising=False)


@pytest.fixture()
def url():
    obs = ThreadingHTTPServer(("127.0.0.1", 0), osvc.ObserverHandler)
    threading.Thread(target=obs.serve_forever, daemon=True).start()
    app = mcp.ProcessStudioMCP(
        mcp.ObserverClient(base_url=f"http://127.0.0.1:{obs.server_address[1]}", key_hex=""),
        hub_client.HubCoreClient(base_url="", org_id=""))
    srv = mcp.make_server("127.0.0.1", 0, app=app)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/mcp"
    finally:
        for s in (srv, obs):
            s.shutdown()
            s.server_close()


def test_python_sdk_surface_and_held_proposal(url):
    c = sdk.Client(url, TOKEN)
    assert TOKEN not in repr(c)
    assert {t["name"] for t in c.tools()} == {"process_studio_status", "process_studio_list",
                                              "activepieces_tools", "hub_actions",
                                              "process_studio_propose"}
    assert "hub.body_directive" in json.dumps(c.hub_actions())
    r = c.propose("hub.body_directive", "r", {"title": "Check", "do_this": "Walk", "owner_role": "Ops"})
    p = ps.get_proposal(r["proposal"]["proposal_id"])
    assert p.status == ps.PROPOSED and p.source == "vscode-agent"
    assert not [n for n in dir(c) if any(f in n.lower() for f in ("authoris", "authoriz", "realis", "realiz", "approve"))]
    with pytest.raises(sdk.IntegrationError):
        c.voice()                                              # no Admin has turned it on


def test_python_sdk_refusals(url):
    with pytest.raises(sdk.IntegrationError, match="401"):
        sdk.Client(url, "wrong").tools()
    with pytest.raises(sdk.IntegrationError, match="plain http"):
        sdk.Client("http://example.com/mcp", "t")
    with pytest.raises(sdk.IntegrationError):
        sdk.Client(url, "")
    with pytest.raises(sdk.IntegrationError, match="cannot reach"):
        sdk.Client("http://127.0.0.1:1/mcp", "t").tools()


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_js_sdk_against_the_same_server(url):
    code = f"""
import {{ Client, IntegrationError }} from {json.dumps((ROOT / 'loadopoly_integrations.mjs').as_uri())};
const c = new Client({json.dumps(url)}, {json.dumps(TOKEN)});
const names = (await c.tools()).map(t => t.name).sort();
const r = await c.propose("hub.body_directive", "r", {{title: "Check from JS", do_this: "Walk", owner_role: "Ops"}});
let denied = "no";
try {{ await new Client({json.dumps(url)}, "wrong").tools(); }} catch (e) {{ denied = e instanceof IntegrationError && /401/.test(e.message) ? "401" : String(e); }}
let pub = "no";
try {{ new Client("http://example.com/mcp", "t"); }} catch (e) {{ pub = /plain http/.test(e.message) ? "refused" : String(e); }}
console.log(JSON.stringify({{names, id: r.proposal.proposal_id, denied, pub, shown: String(c)}}));
"""
    out = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["denied"] == "401" and got["pub"] == "refused" and TOKEN not in got["shown"]
    assert len(got["names"]) == 5 and "process_studio_propose" in got["names"]
    assert ps.get_proposal(got["id"]).status == ps.PROPOSED


def test_published_sdk_copies_match_the_sources():
    for f in ("loadopoly_integrations.py", "loadopoly_integrations.mjs", "loadopoly_integrations.d.mts"):
        assert (ROOT / f).read_bytes() == (ROOT / "site" / "sdk" / f).read_bytes(), f


@pytest.fixture()
def web():
    srv = site.make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _req(u, method="GET"):
    try:
        return urllib.request.urlopen(urllib.request.Request(u, method=method), timeout=10)
    except urllib.error.HTTPError as e:
        return e


def test_site_serves_with_strict_headers_and_no_listing(web):
    r = _req(web + "/health")
    assert r.status == 200 and json.loads(r.read()) == {"ok": True}
    r = _req(web + "/")
    assert r.status == 200 and b"cannot authorise" in r.read()
    assert "default-src 'none'" in r.headers["Content-Security-Policy"] and r.headers["X-Content-Type-Options"] == "nosniff"
    assert _req(web + "/sdk/loadopoly_integrations.py").status == 200
    assert _req(web + "/sdk/").status == 404                      # no directory listing
    assert _req(web + "/../server.py").status in (400, 404)
    for m in ("POST", "PUT", "DELETE"):
        assert _req(web + "/", m).status == 405
