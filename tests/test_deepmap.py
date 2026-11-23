"""deepmap — QUIPU's link map and the Termux node that feeds it.

The node is tested with an injected Termux:API runner (no phone needed), and
end to end: node -> signed edge client -> live in-process observer -> GET /deepmap.
"""
import importlib.util
import json
import socket
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from src.quipu import deepmap
from src.quipu import observer_service as osvc
from src.quipu.qpsi import edge_admission

_NODE_PATH = Path(__file__).resolve().parent.parent / "clients" / "termux" / "deepmap_node.py"
_spec = importlib.util.spec_from_file_location("deepmap_node", _NODE_PATH)
node = importlib.util.module_from_spec(_spec)
sys.modules["deepmap_node"] = node          # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(node)

SALT = "s" * 32
BSSID = "aa:bb:cc:dd:ee:ff"           # a made-up router address: must never appear in a record
WIFI = {"bssid": BSSID, "frequency_mhz": 5180, "ip": "192.168.1.23", "link_speed_mbps": 433,
        "mac_address": "02:00:00:00:00:00", "network_id": 5, "rssi": -54,
        "ssid": '"Home Net"', "ssid_hidden": False, "supplicant_state": "COMPLETED"}


def _runner(payload, code=0):
    def run(argv, timeout):
        assert argv == ["termux-wifi-connectioninfo"]
        return code, json.dumps(payload) if payload is not None else ""
    return run


def _collect(**kw):
    args = dict(ssids=["home net"], peers=[], label="my phone", salt=SALT, run=_runner(WIFI))
    args.update(kw)
    return node.collect(**args)


# ------------------------------------------------------------------ node: gating
def test_records_only_on_an_allow_listed_network_case_insensitively():
    out = _collect()
    assert out.status == "recorded" and out.record["schema"] == "deepmap.link/1"


def test_no_allow_list_is_fail_closed_and_makes_no_calls():
    called = []
    out = node.collect(ssids=[], peers=[node.Peer("a", "192.168.1.5", 80)], label="x", salt=SALT,
                       run=lambda *a: called.append(a) or (0, "{}"), prober=lambda p: called.append(p))
    assert out.status == "no-allowlist" and called == []


def test_off_network_probes_and_sends_nothing():
    probed = []
    out = _collect(ssids=["Some Other Net"], peers=[node.Peer("a", "192.168.1.5", 80)],
                   prober=lambda p: probed.append(p))
    assert out.status == "off-network" and out.record is None and probed == []


def test_not_connected_to_wifi_does_nothing():
    lte = dict(WIFI, supplicant_state="DISCONNECTED", ip="", ssid="<unknown ssid>")
    probed = []
    out = _collect(run=_runner(lte), peers=[node.Peer("a", "192.168.1.5", 80)], prober=lambda p: probed.append(p))
    assert out.status == "not-connected" and probed == []


def test_hidden_ssid_while_connected_is_a_setup_problem_with_a_fix():
    hidden = dict(WIFI, ssid="<unknown ssid>")
    with pytest.raises(node.SetupError, match="Nearby devices"):
        _collect(run=_runner(hidden))


def test_missing_termux_api_and_empty_output_are_setup_problems():
    with pytest.raises(node.SetupError, match="pkg install termux-api"):
        _collect(run=lambda argv, t: (127, ""))
    with pytest.raises(node.SetupError, match="Termux:API"):
        _collect(run=_runner(None))
    with pytest.raises(node.SetupError, match="error"):
        _collect(run=_runner({"error": "permission denied"}))


# ------------------------------------------------------------------ node: privacy
def test_record_carries_no_raw_network_identifier():
    rec = _collect(peers=[node.Peer("kitchen", "192.168.1.9", 8765)],
                   prober=lambda p: {"to": "peer-kitchen", "kind": "tcp", "port": 8765, "ok": True,
                                     "latency_ms": 4.2}).record
    blob = json.dumps(rec)
    assert BSSID not in blob and "aa:bb" not in blob and "02:00:00" not in blob
    assert "Home Net" not in blob and "192.168.1.23" not in blob
    assert rec["wifi"]["ap"].startswith("ap-") and rec["node"]["network_id"].startswith("net-")
    clean, err = deepmap.validate_link(rec)
    assert clean is not None, err                      # and QUIPU accepts what the node builds


def test_hashes_are_salted_stable_and_not_reversible_to_the_input():
    a = node.hash12("salt-one" * 4, "bssid:" + BSSID)
    assert a == node.hash12("salt-one" * 4, "bssid:" + BSSID)
    assert a != node.hash12("salt-two" * 4, "bssid:" + BSSID)
    assert BSSID.replace(":", "") not in a


def test_salt_is_created_once_and_private(tmp_path):
    p = tmp_path / "cfg" / "salt"
    s1 = node.load_salt(p)
    assert len(s1) == 32 and node.load_salt(p) == s1
    assert (p.stat().st_mode & 0o077) == 0


# ------------------------------------------------------------------ node: peers
def test_parse_peers_validates():
    peers = node.parse_peers("kitchen=192.168.1.9:8765, den=den.local:9000")
    assert [(p.name, p.host, p.port) for p in peers] == [("kitchen", "192.168.1.9", 8765), ("den", "den.local", 9000)]
    for bad in ("nohost", "a=host", "a=h:0", "a=h:70000", "bad name=h:1", "a=:80"):
        with pytest.raises(ValueError):
            node.parse_peers(bad)
    with pytest.raises(ValueError):
        node.parse_peers(",".join(f"p{i}=10.0.0.1:80" for i in range(33)))


class _Sock:
    closed = False

    def close(self):
        self.closed = True


def test_probe_times_a_private_peer_and_sends_no_data():
    s = _Sock()
    ticks = iter([1.0, 1.0042])
    edge = node.probe(node.Peer("kitchen", "kitchen.local", 8765),
                      resolver=lambda *a, **k: [(0, 0, 0, "", ("192.168.1.9", 8765))],
                      connect=lambda addr, timeout: s, clock=lambda: next(ticks))
    assert edge == {"to": "peer-kitchen", "kind": "tcp", "port": 8765, "ok": True, "latency_ms": 4.2}
    assert s.closed


def test_probe_never_contacts_a_public_address():
    dialled = []
    edge = node.probe(node.Peer("evil", "example.com", 443),
                      resolver=lambda *a, **k: [(0, 0, 0, "", ("93.184.216.34", 443))],
                      connect=lambda *a, **k: dialled.append(a))
    assert edge["ok"] is False and edge["refused"] and dialled == []
    # one public answer among private ones is also refused (DNS can lie)
    edge = node.probe(node.Peer("mixed", "x", 1),
                      resolver=lambda *a, **k: [(0, 0, 0, "", ("10.0.0.2", 1)), (0, 0, 0, "", ("8.8.8.8", 1))],
                      connect=lambda *a, **k: dialled.append(a))
    assert edge["ok"] is False and dialled == []


def test_probe_reports_unreachable_without_raising():
    def refuse(addr, timeout):
        raise ConnectionRefusedError

    edge = node.probe(node.Peer("down", "192.168.1.77", 80),
                      resolver=lambda *a, **k: [(0, 0, 0, "", ("192.168.1.77", 80))], connect=refuse)
    assert edge["ok"] is False and "latency_ms" not in edge
    edge = node.probe(node.Peer("nx", "nope.invalid", 80), resolver=lambda *a, **k: (_ for _ in ()).throw(socket.gaierror))
    assert edge["ok"] is False


# ------------------------------------------------------------------ node: CLI
def test_cli_check_names_the_fix(monkeypatch, capsys):
    monkeypatch.setattr(node, "run_command", lambda argv, timeout=10: (127, ""))
    assert node.main(["--check"]) == node.EXIT_SETUP
    assert "pkg install termux-api" in capsys.readouterr().out


def test_cli_defaults_to_a_dry_run_that_sends_nothing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(node, "CONFIG_PATH", tmp_path / "none.env")
    monkeypatch.setattr(node, "SALT_PATH", tmp_path / "salt")
    monkeypatch.setattr(node, "run_command", lambda argv, timeout=10: (0, json.dumps(WIFI)))
    made = []
    monkeypatch.setattr(node, "make_client", lambda *a, **k: made.append(a))
    rc = node.main(["--ssid", "Home Net"], env={})
    out = capsys.readouterr().out
    assert rc == 0 and "dry run; nothing was sent" in out and made == []
    assert BSSID not in out and "deepmap.link/1" in out


def test_cli_refuses_when_no_allow_list(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(node, "CONFIG_PATH", tmp_path / "none.env")
    monkeypatch.setattr(node, "SALT_PATH", tmp_path / "salt")
    monkeypatch.setattr(node, "run_command", lambda argv, timeout=10: (0, json.dumps(WIFI)))
    assert node.main([], env={}) == 0
    assert "no-allowlist" in capsys.readouterr().out


# ------------------------------------------------------------------ QUIPU: validation + graph
def _link(**over):
    rec = {"agent_id": "phone-aaa", "schema": "deepmap.link/1",
           "node": {"role": "phone", "label": "p", "network_id": "net-111"},
           "wifi": {"ap": "ap-bbb", "rssi_dbm": -50, "freq_mhz": 5180, "link_mbps": 400},
           "edges": [{"to": "peer-kitchen", "kind": "tcp", "port": 8765, "ok": True, "latency_ms": 5.0}],
           "observed_at": datetime.now(timezone.utc).isoformat()}
    rec.update(over)
    return rec


def test_validate_rebuilds_from_a_whitelist():
    rec = _link()
    rec["secret"] = "x"
    rec["node"]["extra"] = "y"
    rec["edges"][0]["note"] = "z"
    clean, err = deepmap.validate_link(rec)
    assert clean is not None and err == ""
    assert "secret" not in clean and "extra" not in clean["node"] and "note" not in clean["edges"][0]


def test_validate_refuses_mac_addresses_anywhere():
    for placed in ({"node": {"role": "phone", "label": "AA:BB:CC:DD:EE:FF"}},
                   {"edges": [{"to": "x", "kind": "tcp", "ok": True, "extra": "aa-bb-cc-dd-ee-ff"}]},
                   {"wifi": {"ap": "ap-1", "state": "02:00:00:00:00:00"}}):
        clean, err = deepmap.validate_link(_link(**placed))
        assert clean is None and "MAC address" in err


def test_validate_refuses_bad_shapes():
    bad = [_link(schema="other/1"), _link(agent_id="has space"), _link(node={"role": "toaster"}),
           _link(edges=[{"to": "x", "kind": "carrier-pigeon"}]),
           _link(edges=[{"to": "x", "kind": "tcp", "latency_ms": -1}]),
           _link(edges=[{"to": "x", "kind": "wifi", "rssi_dbm": 5}]),
           _link(edges=[{"to": "x", "kind": "tcp", "port": 0}]),
           _link(edges=[{"to": f"p{i}", "kind": "tcp"} for i in range(deepmap.MAX_EDGES + 1)]),
           _link(observed_at="yesterday-ish"), "not an object"]
    for b in bad:
        clean, err = deepmap.validate_link(b)
        assert clean is None and err, b


def test_graph_marks_silent_reporters_stale_and_groups_networks():
    now = datetime.now(timezone.utc)
    fresh = deepmap.validate_link(_link())[0]
    old = deepmap.validate_link(_link(agent_id="phone-old", observed_at=(now - timedelta(hours=3)).isoformat()))[0]
    g = deepmap.build_graph([fresh, old], now=now.timestamp(), max_age_s=3600)
    nodes = {n["id"]: n for n in g["nodes"]}
    assert nodes["phone-aaa"]["stale"] is False and nodes["phone-old"]["stale"] is True
    assert nodes["ap-bbb"]["role"] == "ap" and nodes["peer-kitchen"]["role"] == "peer"
    assert g["reporters"] == 2 and g["networks"] == {"net-111": 1}      # the stale phone is not counted
    e = [x for x in g["edges"] if x["from"] == "phone-aaa" and x["to"] == "peer-kitchen"][0]
    assert e["ok"] is True and e["latency_ms"] == 5.0 and e["stale"] is False
    assert any(x["kind"] == "wifi" and x["to"] == "ap-bbb" for x in g["edges"])   # implied by wifi.ap


# ------------------------------------------------------------------ end to end
@pytest.fixture()
def observer():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), osvc.ObserverHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def test_node_to_quipu_end_to_end(observer, tmp_path):
    rec = _collect(peers=[node.Peer("kitchen", "192.168.1.9", 8765)],
                   prober=lambda p: {"to": "peer-kitchen", "kind": "tcp", "port": 8765, "ok": True,
                                     "latency_ms": 7.5}).record
    from src.quipu import edge_client
    client = edge_client.QuipuEdgeClient("burgerphone", base_url=observer, key=None,
                                         outbox=tmp_path / "outbox.sqlite", background=False)
    res = node.send(rec, client)
    assert res["ok"] is True

    g = _get(observer + "/deepmap")
    ids = {n["id"] for n in g["nodes"]}
    assert rec["agent_id"] in ids and "peer-kitchen" in ids and rec["wifi"]["ap"] in ids
    assert BSSID not in json.dumps(g) and "Home Net" not in json.dumps(g)
    assert _get(observer + "/deepmap?source=burgerphone")["reporters"] >= 1
    assert _get(observer + "/deepmap?source=stockopoly")["reporters"] == 0


def test_the_route_refuses_what_the_validator_refuses(observer):
    def post(body):
        req = urllib.request.Request(observer + "/deepmap/link", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    code, out = post({"source": "burgerphone", "relational": _link(node={"role": "phone", "label": BSSID})})
    assert code == 400 and "MAC address" in out["error"]
    code, out = post({"source": "nobody", "relational": _link()})
    assert code == 400 and "unknown source" in out["error"]
    code, out = post({"source": "burgerphone", "relational": _link()})
    assert code == 200 and out["kept"]["schema"] == "deepmap.link/1"


def test_deepmap_route_is_behind_the_edge():
    assert ("POST", "/deepmap/link") in edge_admission.MUTATING
