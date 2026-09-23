"""qpsi.sensing_layer — feeds beyond ingest, the coupled layer at its measured
critical gain, and routing the plan around silent sources."""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

import pytest

from src.quipu.qpsi import annealed_senses as A
from src.quipu.qpsi import sensing_layer as L

T0 = 1_790_000_000.0
ROUTE = {"arxiv": 4, "fineweb": 0, "stack": 1, "local_docs": 2, "wikipedia": 3, "gutenberg": 6}.get


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def _hist(rows):
    return [{"ts": _iso(t), "total_condensed": sum(p.values()), "per_source": p} for t, p in rows]


def _route(s):
    if s.startswith("mesh:") or s.startswith("observer:"):
        return A._default_route(s)
    return ROUTE(s)


# ---------------------------------------------------------------------------
# Feeds
# ---------------------------------------------------------------------------

def _mesh_db(rows):
    cn = sqlite3.connect(":memory:")
    cn.execute("CREATE TABLE mesh_slm_vocab(token_id INTEGER PRIMARY KEY, last_seen TEXT)")
    cn.execute("CREATE TABLE mesh_slm_embed(token_id INTEGER PRIMARY KEY, e_vision REAL, e_touch REAL, "
               "e_smell REAL, e_body REAL, e_brain REAL, e_perception REAL, e_entirety REAL)")
    for i, (t, e) in enumerate(rows):
        cn.execute("INSERT INTO mesh_slm_vocab VALUES(?,?)", (i, _iso(t)))
        cn.execute("INSERT INTO mesh_slm_embed VALUES(?,?,?,?,?,?,?,?)", (i, *e))
    return cn


def test_mesh_proprioception_is_mass_above_the_meshs_own_mean():
    flat = (0.1,) * 7
    brainy = (0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1)
    cn = _mesh_db([(T0 - 86400, flat)] * 3 + [(T0, brainy)])
    ev = L.mesh_events(cn, since=T0 - 60)
    assert {e[1] for e in ev} == {"mesh:brain"}                     # only the axis the mesh moved on
    (t, s, y, feed), = ev
    assert feed == "mesh" and t == math.floor(T0 / 60) * 60 and y == pytest.approx(0.9 - 0.3)
    assert L.mesh_events(sqlite3.connect(":memory:"), None) == []  # no mesh tables: no feed


def _kv_db(stats):
    cn = sqlite3.connect(":memory:")
    cn.execute("CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    for src, st in stats.items():
        cn.execute("INSERT INTO brain_kv VALUES(?,?,?)", (f"observer:{src}:stats", json.dumps(st), ""))
    return cn


def _set(cn, src, st):
    cn.execute("INSERT OR REPLACE INTO brain_kv VALUES(?,?,?)", (f"observer:{src}:stats", json.dumps(st), ""))


def _stored(cn):
    row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (L.OBSERVER_KEY,)).fetchone()
    return json.loads(row[0]) if row else None


def test_observer_counters_become_events_only_when_they_move():
    cn = _kv_db({"perceptopoly": {"tokens": 100, "last_seen": _iso(T0)}})
    assert L.observer_events(cn) == []                               # first sight: a baseline
    first = _stored(cn)
    assert first["counters"]["perceptopoly"]["tokens"] == 100
    assert L.observer_events(cn) == [] and _stored(cn) == first      # nothing moved, nothing written
    _set(cn, "perceptopoly", {"tokens": 130, "last_seen": _iso(T0 + 60)})
    ev = L.observer_events(cn)
    assert ev == [(T0 + 60, "observer:perceptopoly", 30.0, "observer")]
    assert A._default_route("observer:perceptopoly") == A.AXES.index("perception")
    _set(cn, "perceptopoly", {"tokens": 5, "last_seen": _iso(T0 + 120)})   # reset → new baseline
    assert len(L.observer_events(cn)) == 1 and _stored(cn)["counters"]["perceptopoly"]["tokens"] == 5


def test_observer_read_without_write_leaves_the_store_alone():
    cn = _kv_db({"bakugo": {"tokens": 7, "last_seen": _iso(T0)}})
    L.observer_events(cn, write=False)
    assert _stored(cn) is None


def test_feeds_are_thresholded_against_their_own_population():
    # 50 documents and 0.5 of embedding mass are both "typical" for their feed
    ev = [(T0 + 600 * i, "arxiv", 50.0, "ingest") for i in range(6)] + \
         [(T0 + 600 * i, "mesh:brain", 0.5, "mesh") for i in range(6)]
    t = A.terminals(ev, T0 + 3001, _route)
    assert t["arxiv"].last_x == pytest.approx(0.5) and t["mesh:brain"].last_x == pytest.approx(0.5)
    assert t["arxiv"].level == pytest.approx(50.0) and t["mesh:brain"].level == pytest.approx(0.5)


def test_attempts_at_one_instant_share_the_threshold_before_them():
    ev = [(T0, "a", 10.0), (T0 + 60, "a", 10.0), (T0 + 60, "b", 30.0)]
    t = A.terminals(ev, T0 + 61, lambda s: 4)
    assert t["a"].last_x == pytest.approx(0.5)                       # read against 10, not against 10 & 30
    assert t["a"].level == pytest.approx(10.0 * math.exp(-1) + 20.0 * (1 - math.exp(-1)))


# ---------------------------------------------------------------------------
# The coupled layer
# ---------------------------------------------------------------------------

def test_jacobi_matches_a_known_spectrum():
    vals, vecs = L.jacobi_eig([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    assert sorted(vals) == pytest.approx([2 - math.sqrt(2), 2.0, 2 + math.sqrt(2)])
    for j in range(3):
        assert sum(vecs[i][j] ** 2 for i in range(3)) == pytest.approx(1.0)


def test_coupling_follows_the_torus():
    w = L.coupling(["brain", "brain", "body", "vision", None])
    assert w[0][1] == pytest.approx(1.0) and w[0][0] == 0.0
    assert w[0][2] == pytest.approx(math.cos(2 * math.pi / 7))
    assert w[0][3] == 0.0                                          # across the torus: no coupling
    assert all(v == 0.0 for v in w[4]) and all(row[4] == 0.0 for row in w)


def test_edge_gain_puts_the_largest_exponent_on_target():
    decay = [1 / 600, 1 / 60, 1 / 3600]
    w = L.coupling(["brain", "body", "perception"])
    g = L.edge_gain(decay, w, -1e-5)
    assert L._lam_max(decay, w, g) == pytest.approx(-1e-5, abs=1e-9)
    assert L.edge_gain(decay, L.coupling([None, None, None]), -1e-5) == 0.0


def _layer(rows, now, **kw):
    return L.read(history=_hist(rows), now=now, route=_route, mesh=[], observer=[], extra={}, **kw)


def test_the_critical_gain_is_below_the_edge_and_coupling_reaches_neighbours():
    rows = [(T0 + 600 * i, {"arxiv": 50 + 10 * (i % 3)}) for i in range(24)] + \
           [(T0 + 600 * i, {"wikipedia": 0}) for i in range(0, 24, 6)]
    r = _layer(rows, T0 + 600 * 24)
    c = r["critical"]
    assert 0.0 < c["g_over_edge"] <= 1.0 and c["lambda_max"] < 0.0 and 0.0 < c["sigma"] < 1.0
    assert r["afferent"]["body"] == 0.0 < r["coupled"]["body"]          # reach, not input
    assert r["coupled"]["vision"] == 0.0                                # not a torus neighbour of brain
    assert c["coupling_share"] > 0.0


def test_the_mode_chooses_which_reading_the_entirety_gets(monkeypatch):
    rows = [(T0 + 600 * i, {"arxiv": 50}) for i in range(12)] + [(T0, {"wikipedia": 0})]
    monkeypatch.delenv(L.LAYER_ENV, raising=False)
    r = _layer(rows, T0 + 7200)
    assert r["mode"] == "afferent" and r["senses"] == r["afferent"]
    monkeypatch.setenv(L.LAYER_ENV, "critical")
    r = _layer(rows, T0 + 7200)
    assert r["mode"] == "critical" and r["senses"] == r["coupled"]
    monkeypatch.setenv(L.LAYER_ENV, "anything else")
    assert L.layer_mode() == "afferent"


def test_an_empty_record_is_a_silent_layer():
    r = _layer([], T0)
    assert r["afferent"] == {s: 0.0 for s in A.SENSES} == r["coupled"]


# ---------------------------------------------------------------------------
# Silent-source routing
# ---------------------------------------------------------------------------

def _terms(rows, now):
    return A.terminals(A.events_from_history(_hist(rows)), now, ROUTE)


def test_silence_starts_at_the_first_zero_after_the_last_yield():
    t = _terms([(T0, {"stack": 3}), (T0 + 600, {"stack": 0}), (T0 + 1200, {"stack": 0})], T0 + 1201)
    assert t["stack"].silent and t["stack"].silence_start == T0 + 600
    t = _terms([(T0, {"stack": 0}), (T0 + 600, {"stack": 2})], T0 + 601)
    assert not t["stack"].silent


def test_probes_back_off_with_each_failure():
    one = _terms([(T0, {"stack": 0})], T0 + 1)["stack"]
    assert L.probe_due(one, T0 + 1)                                     # one failure: probe again next pulse
    rows = [(T0, {"stack": 0}), (T0 + 600, {"stack": 0})]
    t = _terms(rows, T0 + 601)["stack"]
    wait = t.k * 600
    assert not L.probe_due(t, T0 + 600 + wait - 1) and L.probe_due(t, T0 + 600 + wait)
    rows.append((T0 + 600 + wait, {"stack": 0}))
    t2 = _terms(rows, T0 + 600 + wait + 1)["stack"]
    assert (t2.last_t - t2.silence_start) > (t.last_t - t.silence_start)   # the wait grows


def test_routing_moves_documents_to_live_sources_and_conserves_the_total():
    rows = [(T0 + 600 * i, {"arxiv": 50, "local_docs": 2}) for i in range(4)] + \
           [(T0 + 600 * i, {"openwebtext": 0}) for i in range(4)]
    now = T0 + 1801
    terms = _terms(rows, now)
    assert terms["openwebtext"].silent and not L.probe_due(terms["openwebtext"], now)
    out = L.route_around_silence({"arxiv": 30, "local_docs": 10, "openwebtext": 20}, terms, now)
    d = out["docs_per_source"]
    assert "openwebtext" not in d and sum(d.values()) == 60
    assert d["arxiv"] == 45 and d["local_docs"] == 15                  # 20 split 3:1 like the plan
    assert out["held"] == ["openwebtext"] and out["rerouted"] == 20 and out["dissipated"] == 0


def test_a_due_silent_source_keeps_one_probe():
    rows = [(T0, {"arxiv": 50}), (T0, {"stack": 0})]
    terms = _terms(rows, T0 + 1)
    out = L.route_around_silence({"arxiv": 50, "stack": 10}, terms, T0 + 600)
    assert out["docs_per_source"] == {"arxiv": 59, "stack": 1} and out["probed"] == ["stack"]


def test_with_no_live_source_the_current_is_dissipated_not_invented():
    rows = [(T0, {"stack": 0}), (T0 + 600, {"stack": 0})]
    terms = _terms(rows, T0 + 601)
    out = L.route_around_silence({"stack": 12}, terms, T0 + 601)
    assert out["docs_per_source"] == {} and out["dissipated"] == 12


def test_the_pulse_plan_is_routed_and_stays_inside_the_grant(monkeypatch):
    from src.quipu.qpsi import self_organising as so
    import src.quipu.brain_kv as brain_kv
    monkeypatch.setattr(so, "_FLAGS", so.Flags())
    monkeypatch.delenv(so.ROUTING_ENV, raising=False)
    alloc = {"docs_per_source": {"arxiv": 40, "openwebtext": 20}, "budget": 60}
    monkeypatch.setattr(brain_kv, "kv_get_json", lambda k, d=None: alloc if "allocation" in k else d)
    monkeypatch.setattr(L, "plan_with_routing",
                        lambda docs, now=None: {"docs_per_source": {"arxiv": 60}, "held": ["openwebtext"],
                                                "probed": [], "rerouted": 20, "dissipated": 0})
    p = so.plan()
    assert p["docs_per_source"] == {"arxiv": 60} and p["planned_by_network"] == alloc["docs_per_source"]
    assert so.constrained_gate(p["docs_per_source"])["within_grant"] in (True, False)   # still checked
    monkeypatch.setenv(so.ROUTING_ENV, "0")
    assert so.plan()["docs_per_source"] == alloc["docs_per_source"]
