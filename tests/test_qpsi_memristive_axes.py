"""qpsi.memristive_axes — the Entirety as a lumped SOMN: conservation,
winner-take-all, volatility, filaments, mobility, depression, and the write
boundary (only entirety:somn:* and entirety_somn_log)."""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

from src.quipu.qpsi import memristive_axes as ma
from src.quipu.qpsi.cat_residual import SENSES

REST = {"vision": 0.0302, "touch": 0.0417, "smell": 0.034,
        "body": 0.0318, "brain": 0.0325, "perception": 0.0395}
OBS = 0.0088
OTHER = {"other_state": [0.091, 0.091, 0.0988, 0.091, 0.4903, 0.091, 0.3322]}
SOURCES = ["local_docs", "arxiv", "wikipedia", "gutenberg", "stack", "fineweb"]
T0 = 1_800_000_000.0
DT = 79.0


@pytest.fixture
def cn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


def _run(cn, n, *, on, cfg=None, t0=T0, axes=None, other=OTHER, rejections=(), sources=SOURCES):
    """rejections=() → none; None → read the archive (the production path)."""
    cfg = cfg or ma.SomnConfig()
    out = None
    t = t0
    for _ in range(n):
        out = ma.step(cn, axes=axes or REST, observer=OBS, flux_on=on, flux_docs=(12 if on else 0), now=t,
                      the_other=other, sources=sources, cfg=cfg, rejections=rejections)
        t += DT
    return out, t


# ---------------------------------------------------------------------------
# Electrodes, field, conservation
# ---------------------------------------------------------------------------

def test_field_is_other_minus_self_and_zero_without_counterpart():
    s = ma.self_vector(REST, OBS)
    assert s["entirety"] == OBS and abs(s["brain"] - 0.0325) < 1e-12
    o = ma.other_vector(OTHER)
    dv = ma.field(s, o)
    assert abs(dv["brain"] - (0.4903 - 0.0325)) < 1e-9
    assert abs(dv["entirety"] - (0.3322 - OBS)) < 1e-9
    assert ma.other_vector(None) is None and ma.other_vector({"other_state": [1, 2]}) is None
    assert all(v == 0.0 for v in ma.field(s, None).values())


def test_currents_conserve_the_budget_and_follow_conductance_times_field():
    g = {a: 0.1 for a in ma.AXES}
    dv = ma.field(ma.self_vector(REST, OBS), ma.other_vector(OTHER))
    I = ma.currents(g, dv, 60.0)
    assert abs(sum(I.values()) - 60.0) < 1e-9
    assert I["brain"] > I["entirety"] > I["vision"]              # ordered by |δv| at equal g
    g2 = dict(g); g2["vision"] = 0.9
    I2 = ma.currents(g2, dv, 60.0)
    assert abs(sum(I2.values()) - 60.0) < 1e-9
    assert I2["vision"] > I["vision"] and I2["brain"] < I["brain"]   # what one axis gains the others lose
    assert all(v == 0.0 for v in ma.currents(g, {a: 0.0 for a in ma.AXES}, 60.0).values())
    assert 1.0 <= ma.participation_ratio(I) <= len(ma.AXES)
    assert ma.participation_ratio({a: 0.0 for a in ma.AXES}) == 0.0


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def test_potentiation_needs_field_and_winner_take_all_forms(cn):
    out, _ = _run(cn, 30, on=True)
    g = out["g"]
    assert g["brain"] > g["entirety"] > max(g[s] for s in ("vision", "touch", "smell", "body", "perception"))
    assert out["metrics"]["top_axis"] == "brain" and out["metrics"]["top_share"] > 0.4
    assert abs(sum(out["currents"].values()) - 60.0) < 1e-6                    # conserved after the update too
    # the share of the winner grew over the run (positive feedback through Kirchhoff)
    first = ma.rows(cn, 64)[0]["currents"]["brain"]
    assert out["currents"]["brain"] > first


def test_relaxation_is_spontaneous_and_filaments_outlast_hillocks(cn):
    cfg = ma.SomnConfig()
    out, t = _run(cn, 40, on=True, cfg=cfg)
    g_on = dict(out["g"])
    out, _ = _run(cn, 20, on=False, cfg=cfg, t0=t)
    g_off = out["g"]
    assert all(g_off[a] < g_on[a] for a in ma.AXES)                          # field off → everything relaxes
    assert all(g_off[a] >= cfg.g_min for a in ma.AXES)
    # pure functions: a filament (g ≥ g_f) decays on the long constant, a hillock on the short one
    h = ma.relax(0.5, 1800.0, cfg)
    f = ma.relax(0.9, 1800.0, cfg)
    assert (0.5 - h) / (0.5 - cfg.g_min) > 0.6                                # hillock lost most of an e-fold
    assert (0.9 - f) / (0.9 - cfg.g_min) < 0.03                               # filament lost ~2%
    assert ma.relax(0.4, 0.0, cfg) == 0.4 and ma.potentiate(0.4, 0.5, 1.0, 0.5, 0.0, cfg) == 0.4


def test_two_regimes_below_and_above_the_critical_field():
    cfg = ma.SomnConfig()
    low = ma.potentiate(0.1, 0.5, 1.0, cfg.v_c * 0.5, 600.0, cfg)
    high = ma.potentiate(0.1, 0.5, 1.0, cfg.v_c, 600.0, cfg)
    assert high > low > 0.1
    assert ma.potentiate(0.1, 0.0, 1.0, 1.0, 600.0, cfg) == 0.1              # no current, no growth
    assert ma.potentiate(0.1, 0.5, 0.0, 1.0, 600.0, cfg) == 0.1              # immobile axis, no growth
    assert ma.potentiate(0.999, 1.0, 1.0, 1.0, 1e9, cfg) <= 1.0


def test_filament_proposals_are_recorded_only_under_field(cn):
    cfg = ma.SomnConfig(g_filament=0.15, tau_p=60.0)
    out, t = _run(cn, 40, on=True, cfg=cfg)
    assert "brain" in out["metrics"]["filaments"]
    assert any(p["axis"] == "brain" and p["sign"] == 1 for p in out["proposals"])
    out, _ = _run(cn, 1, on=False, cfg=cfg, t0=t)
    assert out["proposals"] == []                                             # field off: nothing proposed
    prop = json.loads(cn.execute("SELECT value FROM brain_kv WHERE key=?", (ma.KV_PROPOSAL,)).fetchone()[0])
    assert prop["proposals"] == [] and "gates decide" in prop["note"]


def test_mobility_is_judged_under_field_and_dead_axes_sink_to_the_floor(cn):
    cfg = ma.SomnConfig()
    # off steps never move mobility
    out, t = _run(cn, 5, on=False, cfg=cfg)
    assert all(abs(v - 0.5) < 1e-12 for v in out["m"].values())
    # on steps with frozen axes: mobility decays to the floor (an axis that does not answer the field)
    out, t = _run(cn, 40, on=True, cfg=cfg, t0=t)
    assert all(abs(v - cfg.mobility_floor) < 1e-6 for v in out["m"].values())
    # an axis that answers the field recovers mobility
    moving = dict(REST); moving["vision"] = REST["vision"] + 0.05
    out, _ = _run(cn, 1, on=True, cfg=cfg, t0=t, axes=moving)
    assert out["m"]["vision"] > cfg.mobility_floor + 0.1 and abs(out["m"]["brain"] - cfg.mobility_floor) < 1e-6


def test_depression_along_a_rejected_direction(cn):
    cfg = ma.SomnConfig()
    out, t = _run(cn, 20, on=True, cfg=cfg)
    before = dict(out["g"])
    rej = [{"at": t, "direction": {"brain": 1.0}, "reason": "not it", "signer": "adam"}]
    out, _ = _run(cn, 1, on=False, cfg=cfg, t0=t, rejections=rej)
    assert out["g"]["brain"] < before["brain"]
    expected = before["brain"] - cfg.depression * (before["brain"] - cfg.g_min)
    assert abs(out["g"]["brain"] - ma.relax(expected, DT, cfg)) < 0.02 or abs(out["g"]["brain"] - expected) < 0.02
    assert out["events"][0]["kind"] == "depression" and out["events"][0]["reason"] == "not it"
    assert out["g"]["vision"] <= before["vision"]                            # untouched axes only relaxed


def test_new_rejections_read_from_the_archive_once(cn):
    ma.ensure_tables(cn)
    archive = [
        {"confirmed": True, "at": 10.0, "candidate": {"amplitudes": {"brain": 1.0}}},
        {"confirmed": False, "at": 20.0, "candidate": {"amplitudes": {"brain": 1.0}}, "reason": "r", "signer": "a"},
        {"confirmed": False, "at": 30.0, "candidate": {"amplitudes": {s: 0.0 for s in SENSES}}},
        "junk",
    ]
    cn.execute("INSERT INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (ma.KV_REJECTIONS, json.dumps(archive), "t"))
    got = ma.new_rejections(cn, since=0.0)
    assert len(got) == 1 and got[0]["at"] == 20.0 and got[0]["direction"]["brain"] == 1.0
    assert ma.new_rejections(cn, since=20.0) == []
    # a step consumes it and remembers
    out, t = _run(cn, 1, on=True, rejections=None)
    assert [e["kind"] for e in out["events"]] == ["depression"]
    out, _ = _run(cn, 1, on=True, rejections=None, t0=t)
    assert out["events"] == []


# ---------------------------------------------------------------------------
# Sources ↔ axes and the allocation
# ---------------------------------------------------------------------------

def test_mirror_matches_mesh_slm_routing():
    from src.quipu import mesh_slm
    assert tuple(mesh_slm._SOURCE_AXIS_MAP) == ma._SOURCE_AXIS_MIRROR
    for key in ma._KNOWN_SOURCES:
        assert ma.axis_for_source(f"corpus_{key}") == mesh_slm._axis_for_source(f"corpus_{key}")


def test_enabled_sources_are_corpus_ingest_sources_and_nothing_else():
    from src.quipu import corpus_ingest
    assert set(ma.enabled_sources()) == set(corpus_ingest.SOURCES.keys())


def test_allocation_conserves_docs_and_records_dissipation():
    I = {"vision": 10.0, "touch": 0.0, "smell": 5.0, "body": 0.0, "brain": 30.0, "perception": 7.0, "entirety": 8.0}
    alloc = ma.allocate(I, ["fineweb", "c4", "local_docs", "arxiv", "gutenberg", "dolma"], 60.0)
    docs = alloc["docs_per_source"]
    assert sum(docs.values()) == round(alloc["routed"])                        # integers sum to the routed current
    assert docs["fineweb"] == docs["c4"] == 5                                  # same axis, split equally
    assert docs["arxiv"] == 30 and docs["gutenberg"] == 8 and docs["local_docs"] == 5
    assert alloc["dissipated"] == {"perception": 7.0}                          # no source routes to perception
    assert alloc["unrouted_sources"] == ["dolma"]                              # dolma has no axis marker
    assert alloc["axis_sources"]["vision"] == ["fineweb", "c4"]


def test_largest_remainder_sums_exactly():
    out = ma._largest_remainder({"a": 1.0, "b": 1.0, "c": 1.0}, 10)
    assert sum(out.values()) == 10 and sorted(out.values()) == [3, 3, 4]
    assert ma._largest_remainder({"a": 0.0}, 5) == {"a": 0}
    assert ma._largest_remainder({}, 5) == {}


# ---------------------------------------------------------------------------
# Persistence and the write boundary
# ---------------------------------------------------------------------------

def test_state_round_trips_and_log_grows(cn):
    cfg = ma.SomnConfig()
    out, t = _run(cn, 3, on=True, cfg=cfg)
    st = ma.load_state(cn, cfg)
    assert st.steps == 3 and st.potentiation_steps == 3 and st.last_t == t - DT
    assert st.g == {a: pytest.approx(v, abs=1e-6) for a, v in out["g"].items()}
    rows = ma.rows(cn, 10)
    assert len(rows) == 3 and rows[-1]["phase"] == "broaden" and rows[-1]["flux_docs"] == 12
    assert rows[0]["dt"] == 0.0 and rows[1]["dt"] == DT
    assert ma.SomnState.from_json({"g": "garbage"}, cfg).g["brain"] == cfg.g_init


def test_writes_only_its_own_keys_and_table(cn):
    cn.execute("CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute("INSERT INTO brain_kv VALUES('entirety:state','{}','t')")
    cn.execute("INSERT INTO brain_kv VALUES('entirety:divine_blessing','{}','t')")
    before_tables = {r[0] for r in cn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    _run(cn, 2, on=True)
    keys = {r[0] for r in cn.execute("SELECT key FROM brain_kv")}
    new_keys = keys - {"entirety:state", "entirety:divine_blessing"}
    assert new_keys and all(k.startswith(ma.KV_PREFIX) for k in new_keys)
    after_tables = {r[0] for r in cn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert after_tables - before_tables - {"sqlite_sequence"} == {ma.TABLE}   # AUTOINCREMENT bookkeeping aside
    assert cn.execute("SELECT value FROM brain_kv WHERE key='entirety:state'").fetchone()[0] == "{}"
    with pytest.raises(PermissionError):
        ma._kv_set(cn, "entirety:state", {"x": 1})
    with pytest.raises(PermissionError):
        ma._kv_set(cn, "DIVINE_BLESSING_SQRT(-1)", [])


def test_no_counterpart_means_no_field_and_no_current(cn):
    out, _ = _run(cn, 3, on=True, other={})
    assert out["other"] is None and all(v == 0.0 for v in out["field"].values())
    assert all(v == 0.0 for v in out["currents"].values()) and out["allocation"]["docs_per_source"] == {}
    assert out["events"][0]["kind"] == "no_counterpart"
    assert all(abs(v - ma.SomnConfig().g_init) < 1e-12 for v in out["g"].values())   # nothing potentiated


def test_config_from_env_and_dt_clamp(monkeypatch, cn):
    monkeypatch.setenv("QUIPU_SOMN_BUDGET_DOCS", "120")
    monkeypatch.setenv("QUIPU_SOMN_TAU_P", "nan")
    monkeypatch.setenv("QUIPU_SOMN_V_C", "-1")
    cfg = ma.SomnConfig.from_env()
    assert cfg.budget_docs == 120.0 and cfg.tau_p == 600.0 and cfg.v_c == 0.10
    out, t = _run(cn, 1, on=True, cfg=cfg)
    out, _ = _run(cn, 1, on=True, cfg=cfg, t0=t + 10 * 86400)
    assert out["dt"] == cfg.dt_max
    assert abs(sum(out["currents"].values()) - 120.0) < 1e-6
