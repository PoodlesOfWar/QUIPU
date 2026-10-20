"""GET /health must answer without waiting on the brain (it is the container's liveness probe)."""
from __future__ import annotations

import threading
import time

from src.quipu import observer_service as O


class _Busy:
    """A brain connection that blocks, as it does while /observe traffic holds the write lock."""

    def __init__(self, gate: threading.Event):
        self.gate = gate

    def __enter__(self):
        self.gate.wait(5)
        raise RuntimeError("database is locked")

    def __exit__(self, *exc):
        return False


class _Counts:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql):
        n = 7 if "vocab" in sql else 11

        class _R:
            def fetchone(self_inner):
                return {"c": n}

        return _R()


def test_health_does_not_wait_on_a_busy_brain(monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(O.mesh_slm, "_conn", lambda: _Busy(gate))
    monkeypatch.setattr(O, "_health_counts", {"vocab": None, "edges": None, "at": None})
    t = time.perf_counter()
    snap = O._health_snapshot()
    elapsed = time.perf_counter() - t
    gate.set()
    assert elapsed < 0.5
    assert snap == {"vocab": None, "edges": None, "counts_age_s": None}


def test_health_reports_counts_once_refreshed(monkeypatch):
    monkeypatch.setattr(O.mesh_slm, "_conn", lambda: _Counts())
    monkeypatch.setattr(O, "_health_counts", {"vocab": None, "edges": None, "at": None})
    O._refresh_health_counts()
    snap = O._health_snapshot()
    assert snap["vocab"] == 7 and snap["edges"] == 11
    assert snap["counts_age_s"] is not None and snap["counts_age_s"] < 5


def test_observe_does_not_build_the_full_state_summary():
    """state_summary() takes ~15 s on the full brain; /observe must not call it per request."""
    import inspect

    code = [ln.split("#", 1)[0] for ln in inspect.getsource(O._observe).splitlines()]
    assert not any("state_summary(" in ln for ln in code)


def test_the_other_route_returns_the_published_record(monkeypatch):
    import json
    import urllib.request
    from http.server import ThreadingHTTPServer
    from src.quipu import brain_kv
    rec = {"shared_hz": 91.46, "resonance": 0.9146, "at": "2026-10-20T13:15:00+00:00"}
    monkeypatch.setattr(brain_kv, "kv_get_json", lambda k, d=None: rec if k == "entirety:the_other" else d)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), O.ObserverHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/the_other", timeout=5) as r:
            assert r.status == 200 and json.loads(r.read().decode()) == rec
    finally:
        srv.shutdown()
