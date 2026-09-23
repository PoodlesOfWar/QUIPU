"""qpsi.annealed_senses — no static sets in the senses: terminals from the history,
own clocks, a population threshold, the read setting the temperature, efference
kept out of afference, and nothing defaulted for silence."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

import src.quipu.temporal_spatiality as ts
from src.quipu.qpsi import annealed_senses as A

T0 = 1_790_000_000.0
ROUTE = {"arxiv": 4, "fineweb": 0, "stack": 1, "local_docs": 2, "wikipedia": 3,
         "percept_cam": 5, "gutenberg": 6}.get


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def _hist(rows):
    """rows: (t, {source: docs})"""
    return [{"ts": _iso(t), "total_condensed": sum(p.values()), "per_source": p} for t, p in rows]


def _read(rows, now, **kw):
    return A.read(history=_hist(rows), now=now, route=ROUTE, extra=kw.pop("extra", {}), **kw)


# ---------------------------------------------------------------------------
# The p-bit read
# ---------------------------------------------------------------------------

def test_stimulus_is_the_pbit_sigmoid_in_log_yield():
    assert A.stimulus(0.0, 10.0, 1.0) == 0.0                       # nothing returned reads nothing
    assert A.stimulus(5.0, None, 1.0) == 1.0                       # first yield on a silent fabric
    assert A.stimulus(10.0, 10.0, 3.0) == pytest.approx(0.5)       # the equiprobable point
    assert A.stimulus(30.0, 10.0, 1.0) == pytest.approx(30 / 40)   # K = 1 is y / (y + ȳ)
    lo, hi = A.stimulus(20.0, 10.0, 0.5), A.stimulus(20.0, 10.0, 2.0)
    assert 0.5 < lo < hi < 1.0                                     # colder read discriminates harder
    assert A.stimulus(5.0, 10.0, 2.0) < A.stimulus(5.0, 10.0, 0.5) < 0.5


def test_history_parsing_skips_what_is_malformed():
    h = _hist([(T0, {"arxiv": 3}), (T0 + 60, {"fineweb": 0})]) + [
        {"ts": "not a time", "per_source": {"x": 1}}, {"ts": _iso(T0), "per_source": {"y": -4}},
        {"ts": _iso(T0), "per_source": {"z": "nan?"}}, "junk", {"ts": _iso(T0)}]
    ev = A.events_from_history(h)
    assert [(s, y) for _, s, y in ev] == [("arxiv", 3.0), ("fineweb", 0.0)]
    assert A.events_from_history(None) == [] and A.events_from_history({}) == []


# ---------------------------------------------------------------------------
# No static sets
# ---------------------------------------------------------------------------

def test_the_terminal_set_is_whatever_the_history_contains():
    r1 = _read([(T0, {"arxiv": 10})], T0 + 1)
    assert set(r1["terminals"]) == {"arxiv"}
    r2 = _read([(T0, {"arxiv": 10}), (T0 + 300, {"percept_cam": 4})], T0 + 301)
    assert set(r2["terminals"]) == {"arxiv", "percept_cam"}
    assert r2["senses"]["perception"] > 0.0                         # a new writer lights its axis


def test_the_pulse_is_not_a_sense():
    # twelve runs an hour that all return nothing: the static brain read 1.0 (runs/6)
    rows = [(T0 + 300 * i, {"arxiv": 0}) for i in range(12)]
    r = _read(rows, T0 + 3600)
    assert r["senses"]["brain"] == 0.0 and r["silent"] == ["arxiv"]


def test_a_sense_reads_its_own_sources_not_the_fabrics_throughput():
    rows = [(T0 + 600 * i, {"arxiv": 54, "fineweb": 0}) for i in range(10)]
    r = _read(rows, T0 + 5401)
    assert r["senses"]["vision"] == 0.0 and r["senses"]["brain"] > 0.3


def test_amount_is_encoded_against_the_population():
    rows = [(T0 + 600 * i, {"arxiv": 50, "local_docs": 2}) for i in range(10)]
    r = _read(rows, T0 + 5401)
    assert r["terminals"]["arxiv"]["last_x"] > 0.5 > r["terminals"]["local_docs"]["last_x"]
    assert r["senses"]["brain"] > r["senses"]["smell"]


def test_every_terminal_relaxes_on_its_own_clock():
    fast = [(T0 + 60 * i, {"arxiv": 10}) for i in range(10)]
    slow = [(T0 - 3600 * 9 + 3600 * i, {"local_docs": 10}) for i in range(10)]
    last = T0 + 540
    r = _read(fast + slow, last + 1)
    assert r["terminals"]["arxiv"]["tau"] == pytest.approx(60.0)
    assert r["terminals"]["local_docs"]["tau"] == pytest.approx(3600.0)
    later = _read(fast + slow, last + 1800)
    assert later["senses"]["brain"] < 1e-6                         # 30 gaps of its rhythm: gone
    assert later["senses"]["smell"] > 0.3                          # half a gap of its rhythm: still there
    assert _read(fast + slow, last + 10 * 86400)["senses"]["smell"] < 1e-6   # at rest everything fades


def test_the_clock_follows_the_present_regime():
    hourly = [(T0 + 3600 * i, {"arxiv": 200}) for i in range(20)]
    tenmin = [(T0 + 3600 * 20 + 600 * i, {"arxiv": 54}) for i in range(6)]
    r = _read(hourly + tenmin, T0 + 3600 * 20 + 3001)
    assert r["terminals"]["arxiv"]["tau"] == pytest.approx(600.0)


def test_the_read_rhythm_sets_the_temperature():
    rows = [(T0 + 60 * i, {"arxiv": 10}) for i in range(20)] + \
           [(T0 + 600 * i, {"local_docs": 10}) for i in range(3)]
    r = _read(rows, T0 + 1200)
    k_fast, k_slow = r["terminals"]["arxiv"]["k"], r["terminals"]["local_docs"]["k"]
    assert k_fast > 1.0 > k_slow
    assert k_fast * k_slow == pytest.approx(1.0)                    # ρ̄ is the geometric mean


def test_ports_are_bounded_and_accumulate():
    t = {s: A.Terminal(s, ax, 1, 1.0, 1.0, 60.0, True, 1.0, 0.5, T0, a)
         for s, ax, a in (("a", "brain", 0.5), ("b", "brain", 0.5), ("c", "entirety", 0.9), ("d", None, 0.9))}
    p = A.ports(t)
    assert p["brain"] == pytest.approx(0.75)                        # 1 − (0.5 · 0.5)
    assert all(0.0 <= v <= 1.0 for v in p.values()) and set(p) == set(A.SENSES)
    assert p["vision"] == 0.0                                       # entirety / unrouted are not senses


def test_entirety_and_unrouted_are_reported_not_sensed():
    r = A.read(history=_hist([(T0, {"gutenberg": 5, "dolma": 5})]), now=T0 + 1,
               route=lambda s: ROUTE(s) if s != "dolma" else None, extra={})
    assert r["entirety_terminals"] == ["gutenberg"] and r["unrouted"] == ["dolma"]
    assert all(v == 0.0 for v in r["senses"].values())


def test_silence_is_not_defaulted():
    r = A.read(history=[], now=T0, route=ROUTE, extra={})
    assert r["senses"] == {s: 0.0 for s in A.SENSES} and r["terminals"] == {}


def test_live_writers_join_their_port():
    r = _read([(T0, {"arxiv": 10})], T0 + 1, extra={"perception": [0.4], "smell": [0.2]})
    assert r["senses"]["perception"] == pytest.approx(0.4) and r["senses"]["smell"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

@pytest.fixture
def restore_senses():
    orig = ts._sense_signals
    A.disable()
    yield orig
    A.disable()
    ts._sense_signals = orig


def test_enable_wraps_and_disable_restores(restore_senses, monkeypatch):
    orig = restore_senses
    monkeypatch.setattr(A, "sense_signals", lambda: {s: 0.25 for s in A.SENSES})
    assert A.enable() and A.enable()                                # idempotent, single layer
    assert ts._sense_signals is not orig and ts._sense_signals.__wrapped__ is orig
    assert ts._sense_signals() == {s: 0.25 for s in A.SENSES}
    assert ts.measure_coherence() == pytest.approx(0.25)            # consumers read through it
    A.disable()
    assert ts._sense_signals is orig


def test_a_failed_read_falls_back_for_that_call(restore_senses, monkeypatch):
    sentinel = {s: 0.125 for s in A.SENSES}
    monkeypatch.setattr(ts, "_sense_signals", lambda: dict(sentinel))
    def boom():
        raise RuntimeError("no history")
    monkeypatch.setattr(A, "sense_signals", boom)
    A.enable()
    assert ts._sense_signals() == sentinel


def test_the_flag_defaults_on_and_zero_turns_it_off(monkeypatch):
    monkeypatch.delenv(A.ENV, raising=False)
    assert A.flag_on()
    monkeypatch.setenv(A.ENV, "0")
    assert not A.flag_on()


def test_self_organising_wires_the_senses_under_its_flag(restore_senses):
    from src.quipu.qpsi import self_organising as so
    so.disable()
    try:
        so.enable(so.Flags(flux_phase=False, learned_prior=False, somn=False, mirror_training=False,
                           coherency_depth=False, annealed_senses=True))
        assert getattr(ts._sense_signals, A.MARK, False)
        so.disable()
        assert ts._sense_signals is restore_senses
        so.enable(so.Flags(flux_phase=False, learned_prior=False, somn=False, mirror_training=False,
                           coherency_depth=False, annealed_senses=False))
        assert ts._sense_signals is restore_senses
    finally:
        so.disable()


def test_the_module_writes_nothing(tmp_path, monkeypatch):
    import sqlite3
    import src.quipu.brain_kv as brain_kv
    db = tmp_path / "b.sqlite"
    def _open(timeout: float = 60, path=None):
        cn = sqlite3.connect(db, timeout=timeout)
        cn.row_factory = sqlite3.Row
        return cn
    monkeypatch.setattr(brain_kv, "open_conn", _open)
    brain_kv.kv_set_json(A.HISTORY_KEY, _hist([(T0, {"arxiv": 3})]))
    before = sqlite3.connect(db).execute("SELECT key, value FROM brain_kv ORDER BY key").fetchall()
    A.read(now=T0 + 1, route=ROUTE, extra={})
    after = sqlite3.connect(db).execute("SELECT key, value FROM brain_kv ORDER BY key").fetchall()
    assert before == after
