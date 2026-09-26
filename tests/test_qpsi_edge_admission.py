"""qpsi.edge_admission — per-source identity, the operator's grant, and the
system's allocation inside it (Invariance #7: allocate, never widen)."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.quipu import observer_service as osvc
from src.quipu.qpsi import edge_admission as E

KEY = "11" * 32
GRANT = {"grant_ref": "TEST-GRANT", "total_tokens_per_hour": 100,
         "sources": {"hubcore": {"max_tokens_per_hour": 80},
                     "supply-chain-brain": {"max_tokens_per_hour": 80},
                     "loadopoly-ocr": {"max_tokens_per_hour": 50,
                                       "browser_origins": ["http://localhost:3000"]}}}


class Clock:
    def __init__(self, t=1_790_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _edge(clock=None, grant=GRANT, keys=None):
    writes = []
    e = E.Edge(clock=clock or Clock(), keys={"hubcore": KEY} if keys is None else keys, grant=grant,
               kv=lambda k, v: writes.append((k, v)))
    return e, writes


def _body(src="hubcore", text="alpha beta"):
    return json.dumps({"source": src, "text": text}).encode()


def _admit(e, raw, headers=None, tokens=2, path="/observe", method="POST"):
    return e.admit(method, path, headers or {}, raw, json.loads(raw or b"{}"), osvc._canonical_source, tokens)


@pytest.fixture(autouse=True)
def modes(monkeypatch):
    monkeypatch.setenv(E.AUTH_ENV, "enforce")
    monkeypatch.setenv(E.BUDGET_ENV, "enforce")
    monkeypatch.delenv(E.SKEW_ENV, raising=False)
    monkeypatch.delenv("QUIPU_CORS_ORIGINS", raising=False)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_a_signed_write_is_admitted_and_labelled():
    c = Clock()
    e, _ = _edge(c)
    raw = _body()
    d = _admit(e, raw, E.sign(KEY, "POST", "/observe", raw, "hubcore", ts=c.t))
    assert d.ok and d.assurance == "signed" and d.source == "hubcore"


def test_unsigned_tampered_replayed_stale_and_impersonating_writes_are_refused():
    c = Clock()
    e, _ = _edge(c)
    raw = _body()
    assert _admit(e, raw).code == 401                                        # unsigned
    h = E.sign(KEY, "POST", "/observe", raw, "hubcore", ts=c.t)
    assert _admit(e, _body(text="alpha gamma"), h).code == 401                # body changed after signing
    assert _admit(e, raw, h).ok and _admit(e, raw, h).code == 401            # second use of a signature
    old = E.sign(KEY, "POST", "/observe", raw, "hubcore", ts=c.t - 301)
    assert _admit(e, raw, old).code == 401                                   # outside the skew window
    other = _body(src="supply-chain-brain")
    h2 = E.sign(KEY, "POST", "/observe", other, "hubcore", ts=c.t)
    d = _admit(e, other, h2)
    assert d.code == 401 and "body says" in d.error                          # hubcore posing as SCA
    h3 = E.sign(KEY, "POST", "/observe", raw, "hubcore", ts=c.t + 1)
    assert _admit(e, raw, {**h3, E.H_SOURCE: "supply-chain-brain"}).code == 401   # no key for that source


def test_a_browser_origin_bound_in_the_grant_is_accepted_at_origin_assurance():
    e, _ = _edge()
    raw = _body(src="loadopoly-ocr")
    d = _admit(e, raw, {"Origin": "http://localhost:3000"})
    assert d.ok and d.assurance == "origin"
    assert _admit(e, raw, {"Origin": "https://evil.example"}).code == 401
    assert _admit(e, _body(src="hubcore"), {"Origin": "http://localhost:3000"}).code == 401   # origin bound to OCR only


def test_record_mode_refuses_nothing_and_says_what_it_would_have_done(monkeypatch):
    monkeypatch.setenv(E.AUTH_ENV, "record")
    monkeypatch.setenv(E.BUDGET_ENV, "record")
    e, _ = _edge()
    d = _admit(e, _body(src="bakugo"))
    assert d.ok and d.assurance == "unverified" and d.would_refuse
    assert e.status()["sources"]["bakugo"]["would_refuse"] == 1


def test_non_mutating_routes_pass_untouched():
    e, _ = _edge()
    assert _admit(e, b"{}", path="/slm").ok


# ---------------------------------------------------------------------------
# The grant and the allocation inside it
# ---------------------------------------------------------------------------

def test_allocation_never_exceeds_a_ceiling_or_the_total_and_names_only_granted_sources():
    g = E.Grant.from_json(GRANT)
    for info in ({}, {"hubcore": 0.9, "supply-chain-brain": 0.01}, {"hubcore": 0.5, "loadopoly-ocr": 0.5},
                 {"stranger": 1.0}, {"hubcore": 0.0, "supply-chain-brain": 0.0, "loadopoly-ocr": 0.0}):
        a = E.allocate(g, info)
        assert set(a) == set(g.ceilings)
        assert all(a[s] <= g.ceilings[s] + 1e-9 for s in a)
        assert sum(a.values()) <= g.total + 1e-9


def test_allocation_follows_information_and_refills_what_a_capped_source_frees():
    g = E.Grant.from_json(GRANT)
    even = E.allocate(g, {})
    assert even["hubcore"] == pytest.approx(100 / 3)
    rich = E.allocate(g, {"hubcore": 0.9, "supply-chain-brain": 0.05, "loadopoly-ocr": 0.05})
    assert rich["hubcore"] == pytest.approx(80.0)                  # capped at its ceiling
    assert rich["supply-chain-brain"] == pytest.approx(10.0)       # the freed 10 split by information
    new = E.allocate(g, {"hubcore": 0.2, "supply-chain-brain": 0.2})
    assert new["loadopoly-ocr"] == pytest.approx(new["hubcore"])   # no data yet: weighted at the mean


def test_a_total_above_the_sum_of_ceilings_is_cut_to_it():
    g = E.Grant.from_json({"total_tokens_per_hour": 10**9, "sources": {"a": {"max_tokens_per_hour": 5}}})
    assert g.total == 5


def test_writes_outside_the_grant_or_over_allocation_are_refused_then_readmitted():
    c = Clock()
    e, writes = _edge(c, keys={"hubcore": KEY, "jobhawk": KEY})
    raw = _body(src="jobhawk")
    d = _admit(e, raw, E.sign(KEY, "POST", "/observe", raw, "jobhawk", ts=c.t))
    assert d.code == 403                                                    # signed, but not granted
    for i in range(3):
        rb = _body(text=f"t{i}")
        assert _admit(e, rb, E.sign(KEY, "POST", "/observe", rb, "hubcore", ts=c.t), tokens=10).ok
    rb = _body(text="over")
    d = _admit(e, rb, E.sign(KEY, "POST", "/observe", rb, "hubcore", ts=c.t), tokens=10)
    assert d.code == 429 and d.retry_after and d.retry_after > 3000          # allocation ≈ 33/h
    c.t += 3601
    rb = _body(text="later")
    assert _admit(e, rb, E.sign(KEY, "POST", "/observe", rb, "hubcore", ts=c.t), tokens=10).ok
    assert any(k == E.KV_PLAN for k, _ in writes)


def test_settled_information_moves_the_plan_and_the_plan_is_written_only_when_it_moves():
    c = Clock()
    e, writes = _edge(c)
    e.plan()
    n0 = sum(1 for k, _ in writes if k == E.KV_PLAN)
    e.plan()
    assert sum(1 for k, _ in writes if k == E.KV_PLAN) == n0                 # nothing moved: no write
    d = E.Decision(ok=True, source="hubcore", tokens=10)
    e.settle(d, {"tokens": 10, "enacted": {"novel_tokens": ["a"] * 9}})
    e.settle(E.Decision(ok=True, source="supply-chain-brain", tokens=10),
             {"tokens": 10, "enacted": {"novel_tokens": []}})
    p = e.plan()
    assert p["hubcore"] > p["supply-chain-brain"]
    assert sum(1 for k, _ in writes if k == E.KV_PLAN) == n0 + 1


def test_the_edge_reads_its_grant_and_keys_and_never_writes_them(tmp_path, monkeypatch):
    kf, gf = tmp_path / "keys.json", tmp_path / "grant.json"
    kf.write_text(json.dumps({"sources": {"hubcore": KEY}}))
    gf.write_text(json.dumps(GRANT))
    monkeypatch.setenv(E.KEYS_ENV, str(kf))
    monkeypatch.setenv(E.GRANT_ENV, str(gf))
    before = (kf.read_bytes(), gf.read_bytes(), kf.stat().st_mtime_ns, gf.stat().st_mtime_ns)
    c = Clock()
    e = E.Edge(clock=c, kv=lambda k, v: None)
    raw = _body()
    for i in range(5):
        c.t += 1
        _admit(e, raw, E.sign(KEY, "POST", "/observe", raw, "hubcore", ts=c.t))
    e.plan()
    e.status()
    assert (kf.read_bytes(), gf.read_bytes(), kf.stat().st_mtime_ns, gf.stat().st_mtime_ns) == before


def test_mint_creates_keys_without_touching_existing_ones(tmp_path, capsys):
    kf = tmp_path / "k.json"
    E._main(["mint", "hubcore", "perceptopoly", "--keys-file", str(kf)])
    first = json.loads(kf.read_text())["sources"]
    E._main(["mint", "hubcore", "bakugo", "--keys-file", str(kf)])
    second = json.loads(kf.read_text())["sources"]
    assert second["hubcore"] == first["hubcore"] and set(second) == {"hubcore", "perceptopoly", "bakugo"}
    assert "QUIPU_EDGE_KEY_PERCEPTOPOLY=" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Through the observer's HTTP layer
# ---------------------------------------------------------------------------

@pytest.fixture
def server(monkeypatch):
    c = Clock()
    e, _ = _edge(c)
    monkeypatch.setattr(E, "_EDGE", e)
    seen = []

    def fake_observe(body):
        seen.append(body)
        return 200, {"ok": True, "tokens": 2, "enacted": {"novel_tokens": ["alpha"]}}
    monkeypatch.setattr(osvc, "_observe", fake_observe)
    monkeypatch.setattr(osvc.world_model, "annealing_cycle", lambda: {"ok": True, "annealed": True})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), osvc.ObserverHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", c, seen
    httpd.shutdown()


def _post(url, raw, headers):
    req = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json", **headers},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), dict(exc.headers)


def test_observe_over_http_refuses_unsigned_and_admits_signed(server):
    url, c, seen = server
    raw = _body(text="alpha beta")
    code, body, _ = _post(url + "/observe", raw, {})
    assert code == 401 and body["edge"]["assurance"] == "unverified" and not seen
    code, body, _ = _post(url + "/observe", raw, E.sign(KEY, "POST", "/observe", raw, "hubcore", ts=c.t))
    assert code == 200 and body["edge"]["assurance"] == "signed" and len(seen) == 1


def test_anneal_needs_identity_on_get_too(server):
    url, c, _ = server
    try:
        urllib.request.urlopen(url + "/anneal", timeout=5)
        assert False, "unsigned GET /anneal was served"
    except urllib.error.HTTPError as exc:
        assert exc.code == 401
    req = urllib.request.Request(url + "/anneal", headers=E.sign(KEY, "GET", "/anneal", b"", "hubcore", ts=c.t))
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200


def test_cors_echoes_bound_origins_only(server):
    url, _, _ = server
    for origin, want in (("http://localhost:3000", "http://localhost:3000"), ("https://evil.example", None)):
        req = urllib.request.Request(url + "/health", headers={"Origin": origin}, method="OPTIONS")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.headers.get("Access-Control-Allow-Origin") == want
