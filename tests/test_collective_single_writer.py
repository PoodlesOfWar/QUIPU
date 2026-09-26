"""v0.48.0 — one brain, one writer, every door leading to it.

edge_client (outbox, idempotency, signing, retry), the edge's persisted
idempotency and public-door rule, the single-writer guard, the Entirety
supervisor and the brain migration."""
from __future__ import annotations

import io
import json
import sqlite3
import threading
import time
import urllib.error
from http.server import ThreadingHTTPServer

import pytest

from src.quipu import edge_client as C
from src.quipu import entirety_service as ES
from src.quipu import local_store
from src.quipu import observer_service as osvc
from src.quipu.qpsi import edge_admission as E

KEY = "22" * 32


# ---------------------------------------------------------------------------
# A fake QUIPU for the client
# ---------------------------------------------------------------------------

class FakeResp(io.BytesIO):
    def __init__(self, status, payload):
        super().__init__(json.dumps(payload).encode())
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeQuipu:
    """Behaves like the observer behind the edge: verifies, dedupes, can fail."""

    def __init__(self):
        self.applied: list[dict] = []
        self.seen: set[str] = set()
        self.script: list = []          # queue of forced outcomes: int code or Exception
        self.headers: list[dict] = []

    def __call__(self, req, timeout=None):
        self.headers.append(dict(req.headers))
        if self.script:
            nxt = self.script.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            hdrs = {"Retry-After": "7"} if nxt == 429 else {}
            raise urllib.error.HTTPError(req.full_url, nxt, "forced", hdrs, io.BytesIO(b'{"error":"forced"}'))
        idem = req.headers.get("X-quipu-idempotency")
        if idem in self.seen:
            return FakeResp(200, {"ok": True, "duplicate": True})
        self.seen.add(idem)
        self.applied.append(json.loads(req.data))
        return FakeResp(200, {"ok": True})


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _client(tmp_path, fake, clock=None, **kw):
    return C.QuipuEdgeClient("perceptopoly", base_url="http://quipu:7100", key=KEY,
                             outbox=tmp_path / "ob.sqlite", background=False, opener=fake,
                             clock=clock or Clock(), **kw)


def test_a_write_is_kept_until_heard_and_sent_once(tmp_path):
    fake, clock = FakeQuipu(), Clock()
    q = _client(tmp_path, fake, clock)
    fake.script = [ConnectionRefusedError("quipu down"), 503]
    k = q.observe("standoff 2.4 m")
    q.flush()
    assert q.pending() == 1 and not fake.applied                      # network down: kept
    clock.t += 10_000
    q.flush()
    assert q.pending() == 1 and not fake.applied                      # 503: kept
    clock.t += 10_000
    q.flush()
    assert q.pending() == 0 and len(fake.applied) == 1
    assert fake.headers[-1]["X-quipu-idempotency"] == k


def test_the_outbox_survives_the_process(tmp_path):
    fake = FakeQuipu()
    fake.script = [ConnectionRefusedError("down")]
    q = _client(tmp_path, fake)
    q.observe("kept across restarts")
    q.flush()
    del q
    q2 = _client(tmp_path, fake, Clock(1_900_000_000.0))
    assert q2.pending() == 1
    q2.flush()
    assert q2.pending() == 0 and len(fake.applied) == 1


def test_retries_are_never_learned_twice(tmp_path):
    fake = FakeQuipu()
    q = _client(tmp_path, fake)
    k1 = q.observe("same words")
    k2 = q.observe("same words")                                      # identical write, same key
    assert k1 == k2 and q.pending() == 1
    q.flush()
    fake.seen.add("x" * 20)
    q.submit("/observe", {"source": "perceptopoly", "text": "same words"}, idem=k1)
    q.flush()                                                          # server says duplicate
    assert len(fake.applied) == 1 and q.stats()["duplicate"] == 1


def test_edge_refusals_hold_rate_limits_wait_and_bad_payloads_are_kept_aside(tmp_path):
    fake, clock = FakeQuipu(), Clock()
    q = _client(tmp_path, fake, clock)
    fake.script = [401, 429, 400]
    q.observe("a")
    q.flush()
    assert q.pending() == 1 and q.stats()["held_auth"] == 1          # no key yet: held, not dropped
    clock.t += C._AUTH_HOLD_S + 1
    q.flush()
    assert q.pending() == 1                                           # 429: waits Retry-After (7 s)
    clock.t += 8
    q.flush()
    s = q.stats()
    assert s["queued"] == 0 and s["rejected_kept"] == 1                # 400: kept for inspection


def test_a_full_outbox_refuses_new_writes_without_losing_old_ones(tmp_path):
    fake = FakeQuipu()
    fake.script = [ConnectionRefusedError("down")] * 5
    q = _client(tmp_path, fake)
    q.max_bytes = 200                                     # room for the first write only
    assert q.observe("x" * 50) is not None
    assert q.observe("y" * 500) is None and q.stats()["dropped_full"] >= 1
    assert q.pending() >= 1


def test_the_client_never_raises_into_its_caller(tmp_path):
    q = _client(tmp_path, FakeQuipu())
    assert q.submit("/observe", {"bad": object()}) is not None        # default=str serialises it
    q._db.close()
    assert q.observe("after the db is gone") is None                   # swallowed, counted
    assert q._counts["errors"] >= 1


def test_plain_http_to_a_public_host_is_queued_not_sent(tmp_path):
    fake = FakeQuipu()
    q = C.QuipuEdgeClient("perceptopoly", base_url="http://93.184.216.34:7100", key=KEY,
                          outbox=tmp_path / "o.sqlite", background=False, opener=fake)
    q.observe("secret-ish")
    q.flush()
    assert not fake.headers and q.pending() == 1 and not q.stats()["secure"]
    assert C._clear_text_ok("https://quipu.example.org") and C._clear_text_ok("http://quipu:7100")


def test_client_signature_verifies_at_the_edge(tmp_path):
    clock = Clock(time.time())
    edge = E.Edge(clock=clock, keys={"perceptopoly": KEY}, grant=None, kv=lambda k, v: None,
                  idempotency=E.IdempotencyStore(open_conn=lambda: sqlite3.connect(tmp_path / "i.sqlite")))
    body = b'{"source":"perceptopoly","text":"t"}'
    h = C.sign_headers(KEY, "perceptopoly", "POST", "/observe", body, idem="abcdef1234567890", ts=clock.t)
    d = edge.admit("POST", "/observe", h, body, json.loads(body), osvc._canonical_source, 1)
    assert d.ok and d.assurance == "signed" and d.idempotency == "abcdef1234567890"
    h2 = dict(h, **{E.H_IDEM: "zzzzzz1234567890"})                   # swap the key on a captured request
    edge2 = E.Edge(clock=clock, keys={"perceptopoly": KEY}, grant=None, kv=lambda k, v: None)
    assert edge2.admit("POST", "/observe", h2, body, json.loads(body), osvc._canonical_source, 1).assurance \
        == "unverified"


# ---------------------------------------------------------------------------
# The edge: persisted idempotency and the public door
# ---------------------------------------------------------------------------

def test_an_applied_write_is_recognised_after_a_restart(tmp_path, monkeypatch):
    monkeypatch.setenv(E.AUTH_ENV, "record")
    monkeypatch.setenv(E.BUDGET_ENV, "off")
    db = tmp_path / "brain.sqlite"
    store = lambda: E.IdempotencyStore(open_conn=lambda: sqlite3.connect(db))
    body = b'{"source":"bakugo","text":"t"}'
    hdr = {E.H_IDEM: "idem-0000000001"}
    e1 = E.Edge(keys={}, kv=lambda k, v: None, idempotency=store())
    d = e1.admit("POST", "/observe", hdr, body, json.loads(body), osvc._canonical_source, 1)
    assert d.ok and not d.duplicate
    e1.settle(d, {"tokens": 1, "enacted": {"novel_tokens": []}})
    e2 = E.Edge(keys={}, kv=lambda k, v: None, idempotency=store())      # a new process
    d2 = e2.admit("POST", "/observe", hdr, body, json.loads(body), osvc._canonical_source, 1)
    assert d2.ok and d2.duplicate


def test_a_public_front_door_only_admits_signed_writes(monkeypatch):
    monkeypatch.setenv(E.AUTH_ENV, "record")                            # local mode is lenient...
    e = E.Edge(keys={"hubcore": KEY}, grant={"sources": {"loadopoly-ocr": {
        "max_tokens_per_hour": 10, "browser_origins": ["http://localhost:3000"]}}},
        kv=lambda k, v: None, idempotency=E.IdempotencyStore(open_conn=lambda: sqlite3.connect(":memory:")))
    body = b'{"source":"loadopoly-ocr","text":"t"}'
    d = e.admit("POST", "/observe", {"CF-Connecting-IP": "203.0.113.9", "Origin": "http://localhost:3000"},
                body, json.loads(body), osvc._canonical_source, 1)
    assert not d.ok and d.code == 401                                   # ...but not for the internet
    b2 = b'{"source":"hubcore","text":"t"}'
    h = E.sign(KEY, "POST", "/observe", b2, "hubcore")
    d = e.admit("POST", "/observe", {**h, "X-Forwarded-For": "203.0.113.9"}, b2, json.loads(b2),
                osvc._canonical_source, 1)
    assert d.code == 403                     # signed, but hubcore is not in this grant: enforced too


def test_the_observer_answers_a_duplicate_without_learning_again(monkeypatch, tmp_path):
    monkeypatch.setenv(E.AUTH_ENV, "record")
    monkeypatch.setenv(E.BUDGET_ENV, "off")
    e = E.Edge(keys={}, kv=lambda k, v: None,
               idempotency=E.IdempotencyStore(open_conn=lambda: sqlite3.connect(tmp_path / "i.sqlite")))
    monkeypatch.setattr(E, "_EDGE", e)
    seen = []
    monkeypatch.setattr(osvc, "_observe", lambda b: (seen.append(b), (200, {"ok": True, "tokens": 1}))[1])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), osvc.ObserverHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        q = C.QuipuEdgeClient("bakugo", base_url=f"http://127.0.0.1:{httpd.server_address[1]}", key=None,
                              outbox=tmp_path / "o.sqlite", background=False)
        k = q.observe("card centering 58/42")
        q.flush()
        q.submit("/observe", {"source": "bakugo", "text": "card centering 58/42"}, idem=k)
        q.flush()
        assert len(seen) == 1 and q.stats()["sent"] == 1 and q.stats()["duplicate"] == 1
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# Single writer
# ---------------------------------------------------------------------------

def test_a_moved_brain_cannot_be_reopened_on_the_host(tmp_path, monkeypatch):
    marker = tmp_path / "local_brain.MOVED.json"
    monkeypatch.setattr(local_store, "MOVED_MARKER", marker)
    monkeypatch.setattr(local_store, "_DB_PATH", tmp_path / "local_brain.sqlite")
    monkeypatch.delenv("SCB_DB_PATH", raising=False)
    monkeypatch.delenv("QUIPU_BRAIN_OWNER", raising=False)
    assert local_store.db_path() == tmp_path / "local_brain.sqlite"
    marker.write_text("{}")
    with pytest.raises(local_store.BrainMovedError):
        local_store.db_path()
    monkeypatch.setenv("QUIPU_BRAIN_OWNER", "1")
    assert local_store.db_path() == tmp_path / "local_brain.sqlite"


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------

def test_a_failing_part_is_recorded_and_retried_without_stopping(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first run fails")
        return {"flipped": True}
    stop = threading.Event()
    t = threading.Thread(target=ES._loop, args=("flaky", 0.05, flaky), kwargs={"first_delay_s": 0.0, "stop": stop})
    t.start()
    time.sleep(0.4)
    stop.set()
    t.join(2)
    st = ES.status()["loops"]["flaky"]
    assert st["errors"] == 1 and st["runs"] >= 2 and "first run fails" in st["last_error"]
    assert st["last_result"] == {"flipped": True}


def test_the_pulse_is_the_operators_switch(monkeypatch):
    monkeypatch.delenv("QUIPU_PULSE_ROUTE", raising=False)
    monkeypatch.setenv("QUIPU_EXPANSION_INTERVAL_S", "3600")
    monkeypatch.setenv("QUIPU_DOC_ANNEAL_MINUTES", "600")
    monkeypatch.setattr(ES, "expansion_step", lambda: {})
    monkeypatch.setattr(ES, "doc_annealing", lambda: {})
    stop = threading.Event()
    names = [t.name for t in ES.start_background(stop)]
    stop.set()
    assert "entirety-pulse" not in names and "disabled" in ES.status()["loops"]["pulse"]
    assert not ES.settings()["pulse_route"]


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def _brain(path, feed=(), kv=None):
    cn = sqlite3.connect(path)
    cn.execute("CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute("CREATE TABLE mesh_corpus_feed(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, "
               "source TEXT NOT NULL DEFAULT '', created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)")
    for t, s in feed:
        cn.execute("INSERT INTO mesh_corpus_feed(text, source) VALUES(?,?)", (t, s))
    for k, v in (kv or {}).items():
        cn.execute("INSERT INTO brain_kv VALUES(?,?,?)", (k, json.dumps(v), ""))
    cn.commit()
    cn.close()


def test_migration_replays_the_observer_into_the_base_and_leaves_both_inputs_untouched(tmp_path, monkeypatch):
    from src.quipu import brain_migrate as BM
    base, obs, out = tmp_path / "host.sqlite", tmp_path / "obs.sqlite", tmp_path / "merged.sqlite"
    _brain(base, kv={"entirety:state": {"axes": 1}, "observer:bakugo:stats": {"observations": 2, "tokens": 10}})
    _brain(obs, feed=[("alpha beta gamma", "bakugo/code/hideout-mesh"), ("delta", "perceptopoly/percept/x")],
           kv={"entirety:state": {"axes": 2}, "observer:bakugo:stats": {"observations": 3, "tokens": 5},
               "world_model:phase": 0.4})
    before = (base.read_bytes(), obs.read_bytes())
    monkeypatch.setenv("SCB_DB_PATH", str(out))
    r = BM.migrate(base, obs, out, train=False)
    assert (base.read_bytes(), obs.read_bytes()) == before
    assert r["replayed"] == 2 and r["by_source"]["bakugo/code/hideout-mesh"] == 1
    cn = sqlite3.connect(out)
    kv = {k: json.loads(v) for k, v in cn.execute("SELECT key, value FROM brain_kv")}
    assert kv["entirety:state"] == {"axes": 1}                                  # the base's Entirety stands
    assert kv["migration:observer_entirety"]["entirety:state"] == {"axes": 2}   # the other is archived
    assert kv["observer:bakugo:stats"]["observations"] == 5                     # counters summed
    assert kv["world_model:phase"] == 0.4
    assert [t for (t,) in cn.execute("SELECT text FROM mesh_corpus_feed ORDER BY id")] == ["alpha beta gamma", "delta"]
    cn.close()
    assert BM.migrate(base, obs, out, train=False)["already"] is True           # idempotent


def test_a_relay_is_accepted_only_for_the_sources_the_grant_lets_it_carry(monkeypatch):
    monkeypatch.setenv(E.AUTH_ENV, "enforce")
    monkeypatch.setenv(E.BUDGET_ENV, "off")
    e = E.Edge(keys={"supply-chain-brain": KEY}, kv=lambda k, v: None,
               grant={"sources": {"supply-chain-brain": {"max_tokens_per_hour": 100, "relays_for": ["bakugo"]}}},
               idempotency=E.IdempotencyStore(open_conn=lambda: sqlite3.connect(":memory:")))
    for body_src, want in (("bakugo", "relayed"), ("perceptopoly", "unverified")):
        body = json.dumps({"source": body_src, "text": "t"}).encode()
        h = E.sign(KEY, "POST", "/observe", body, "supply-chain-brain")
        d = e.admit("POST", "/observe", h, body, json.loads(body), osvc._canonical_source, 1)
        assert d.assurance == want
        if want == "relayed":
            assert d.ok and d.source == "supply-chain-brain" and d.relayed_for == "bakugo"
        else:
            assert d.code == 401
