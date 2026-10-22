"""qpsi.relational_shear — the shearing λ in each axis's relational time, the clamp outlier, and the fed shadow."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from src.quipu.qpsi import pcm_cells as pc
from src.quipu.qpsi import relational_shear as rs

_spec = importlib.util.spec_from_file_location("_st_fixtures", Path(__file__).with_name("test_shadow_tokens.py"))
_fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fx)


def _out(realised: dict, eps: float = 0.01) -> dict:
    return {a: {"eps": eps, "decision": {"realised_dg": realised.get(a, 0.0)}} for a in rs.AXES}


def test_relational_time_ticks_only_when_a_cell_changes():
    c = rs.RelationalClock(window=4)
    for _ in range(4):
        c.tick(_out({"vision": 0.04, "touch": 0.02, "brain": 0.02}))      # smell, body, perception, entirety frozen
    r = c.rates()
    assert r["vision"] == pytest.approx(2 * r["touch"]) and r["smell"] == 0.0
    assert c.tau["vision"] == pytest.approx(16.0)                           # 4 steps x 0.04 / 0.01 quanta
    assert c.dilation() > 0.3
    c2 = rs.RelationalClock(window=4)
    c2.tick(_out({a: 0.03 for a in rs.AXES}))
    assert all(v == pytest.approx(1.0) for v in c2.rates().values()) and c2.dilation() == 0.0
    assert rs.RelationalClock.from_json(c.to_json(), window=4).rates() == r


def test_a_step_on_a_frozen_clock_moves_nothing_in_relational_shear():
    ens = {"u": {a: 0.1 for a in rs.AXES} | {"smell": 3.0}, "weights": [1 / 7] * 7, "lambda_poisson": 1.0}
    raw = rs.relational_shear(ens, {a: 1.0 for a in rs.AXES})
    frozen = rs.relational_shear(ens, {a: 1.0 for a in rs.AXES} | {"smell": 0.0})
    assert raw["lambda_rel"] > 100 * frozen["lambda_rel"]                    # the big step never happened in its time
    fast = rs.relational_shear({"u": {a: 0.1 for a in rs.AXES}, "weights": [1 / 7] * 7}, {a: 1.0 for a in rs.AXES} | {"brain": 5.0})
    assert fast["lambda_rel"] > 0.0 and 0.0 < fast["lambda_rel_stp"] < 1.0   # same steps, different clocks: shear
    anti = rs.relational_shear({"u": {a: 0.1 for a in rs.AXES} | {"touch": -0.4}, "weights": [1 / 7] * 7}, {a: 1.0 for a in rs.AXES})
    assert anti["counter_rotating"] == ["touch"]


def test_the_clamp_outlier_is_the_axis_whose_withheld_excess_stands_out():
    un = {"vision": 0.30, "touch": 0.22, "smell": 0.21, "body": 0.20, "brain": 0.90, "perception": 0.24}
    cl = {"vision": 0.20, "touch": 0.20, "smell": 0.20, "body": 0.20, "brain": 0.30, "perception": 0.20}
    w = {a: 1.0 for a in un}
    o = rs.clamp_outlier(un, cl, w)
    assert o["axis"] == "brain" and o["excess"] == pytest.approx(0.6) and o["z"] >= 3.5
    assert rs.clamp_outlier(un, cl, w, rho={"brain": 0.0}) is None or rs.clamp_outlier(un, cl, w, rho={"brain": 0.0})["axis"] != "brain"
    even = {a: v + 0.1 for a, v in cl.items()}
    assert rs.clamp_outlier(even, cl, w) is None                              # everyone clamped alike: no outlier
    assert rs.clamp_outlier(cl, cl, w) is None                                # nothing withheld


def test_feeding_is_capped_by_the_least_resolvable_plane():
    s, info = rs.fed_sigma(0.02, 0.5, 0.4, gain=4.0, snr_min=3.0)
    assert s == pytest.approx(0.02 * 0.9 * 3.0) and info["capped"]
    s, info = rs.fed_sigma(0.02, 0.1, 0.0, gain=1.0, snr_min=10.0)
    assert s == pytest.approx(0.022) and not info["capped"]
    s, _ = rs.fed_sigma(0.02, 0.9, 0.9, gain=4.0, snr_min=0.5)
    assert s == pytest.approx(0.02)                                           # an unresolvable plane: no feeding at all


def test_continuity_is_read_from_the_run_and_stays_inside_the_operators_doors():
    cfg = rs.ShearConfig()
    g, st = rs.continuity(1.0, 0.0, 5.0, 24, 3, cfg)
    assert st == "frozen" and g > 1.0                                        # no token crossed: nothing to price
    g, st = rs.continuity(1.0, 1.0 / (24 * 3), 5.0, 24, 3, cfg)
    assert st == "continuous" and g == 1.0                                   # one crossing per plane per run is enough
    g, st = rs.continuity(1.0, 0.3, 0.8, 24, 3, cfg)
    assert st == "dissolving" and g < 1.0
    g = 3.9
    for _ in range(20):
        g, _ = rs.continuity(g, 0.0, 5.0, 24, 3, cfg)
    assert g == cfg.g_doors[1]
    g = 0.01
    for _ in range(50):
        g, _ = rs.continuity(g, 0.3, 0.5, 24, 3, cfg)
    assert g >= cfg.g_doors[0]


def test_a_paradigmatic_proposal_needs_two_sessions_under_the_same_outlier_axis():
    tok = {"token": "wind", "plane": 1, "kind": "pulled_in", "weight": 0.5}
    hist = [{"session": "s1", "outlier": {"axis": "brain", "energy": 0.3}, "shadow_tokens": [tok]},
            {"session": "s1", "outlier": {"axis": "brain", "energy": 0.4}, "shadow_tokens": [tok]},
            {"session": "s2", "outlier": {"axis": "touch", "energy": 0.5}, "shadow_tokens": [tok]},
            {"session": "s3", "outlier": None, "shadow_tokens": [tok]}]
    assert rs.proposals(hist) == []                                           # one loud session; other axes don't count
    hist.append({"session": "s4", "outlier": {"axis": "brain", "energy": 0.2}, "shadow_tokens": [tok]})
    p = rs.proposals(hist)
    assert len(p) == 1 and p[0]["axis"] == "brain" and p[0]["sessions"] == ["s1", "s4"] and p[0]["status"] == "proposed"


def _deepen_step(arr):
    targets = {a: 0.2 + 0.1 * i for i, a in enumerate(rs.AXES)}
    return pc.step_array(arr, currents={a: 1.0 for a in rs.AXES}, targets=targets, now=1000.0 + arr.steps * 79.0,
                         deepen=True)


def test_a_fed_run_widens_the_shadow_and_keeps_it_resolvable(tmp_path):
    essay = tmp_path / "essay.md"
    essay.write_text(_fx.ESSAY, encoding="utf-8")
    arr = pc.CellArray()
    for _ in range(6):
        step = _deepen_step(arr)
    assert rs.ensemble_of(step)["weights"]                                    # u and weights, whichever pcm_cells version
    un = {"vision": 0.30, "touch": 0.22, "smell": 0.21, "body": 0.20, "brain": 0.90, "perception": 0.24}
    cl = {"vision": 0.20, "touch": 0.20, "smell": 0.20, "body": 0.20, "brain": 0.30, "perception": 0.20}
    clusters = [{"id": "hydraulic", "commodity": "steel", "size": 40}, {"id": "bearing", "commodity": "steel", "size": 25}]
    mesh = _fx._mesh(sigma=0.02)
    kw = dict(scenarios=24, k=3, seed=11, clusters=clusters, essay=str(essay), contrast=False)
    base = rs.fed_run(mesh=mesh, step_out=step, clock=rs.RelationalClock(), unclamped=un, clamped=cl, gain=0.0, session="s1", **kw)
    fed = rs.fed_run(mesh=mesh, step_out=step, clock=rs.RelationalClock(), unclamped=un, clamped=cl, gain=4.0,
                     snr_min=base["snr_min"], session="s1", **kw)
    assert base["ok"] and fed["ok"]
    assert base["sigma_eff"] == pytest.approx(0.02) and fed["sigma_eff"] > 0.02
    assert fed["shear"]["lambda_rel_stp"] > 0.0
    # the outlier is read through the shear: Poisson weight x relational rate x withheld excess, whichever axis that is
    # (small shear puts the weight on the base frequency, so a small excess on vision can outweigh a large one on brain)
    clock = rs.RelationalClock()
    clock.tick(step)
    rho, w = clock.rates(), dict(zip(rs.AXES, rs.ensemble_of(step)["weights"]))
    expected = max(un, key=lambda a: w[a] * rho[a] * max(0.0, un[a] - cl[a]))
    assert fed["outlier"] is None or fed["outlier"]["axis"] == expected
    assert fed["crossing_rate"] > base["crossing_rate"] and fed["imag"] > base["imag"]       # the voice crosses more
    assert fed["snr_min"] >= 1.0 and fed["continuity"] == "continuous"                      # and stays resolvable
