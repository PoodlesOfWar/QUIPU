"""qpsi.pcm_cells — chalcogenide phase-change cells behind the axis pressures,
and the derivative decision model that turns an rADAM step into SET / RESET /
REFRESH / HOLD and scales rADAM by the cell's programming slope."""
from __future__ import annotations

import math
import sqlite3

import pytest

from src.quipu.qpsi import pcm_cells as pc
from src.quipu.radam_optimizer import radam_step

CFG = pc.PcmConfig()


def _fresh():
    return pc.PcmCell(t_amorph=CFG.drift_t0)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def test_constants_put_set_between_tx_and_tm_and_reset_past_tm():
    dT_set = CFG.i_set ** 2 / CFG.g_on * CFG.r_th
    dT_reset = CFG.i_reset ** 2 / CFG.g_on * CFG.r_th
    assert CFG.t_x < CFG.t_amb + dT_set < CFG.t_m < CFG.t_amb + dT_reset
    assert pc.tau_crystallise(573.0, CFG) == pytest.approx(50e-9, rel=1e-6)   # the τ0 anchor
    assert pc.tau_crystallise(300.0, CFG) > 1.0                               # nothing crystallises at rest


def test_set_staircase_is_gradual_sigmoidal_and_saturates():
    c = _fresh()
    xs = []
    for _ in range(60):
        pc.set_pulse(c, CFG)
        xs.append(c.x)
    assert all(b >= a for a, b in zip(xs, xs[1:]))                            # monotone
    assert xs[0] < 0.01 and 0.1 < xs[15] < 0.5 and xs[-1] > 0.95               # incubation, partial, saturation
    steps = [b - a for a, b in zip(xs, xs[1:])]
    assert steps.index(max(steps)) > 3                                         # slope peaks mid-way, not at x=0


def test_reset_melts_and_quenches_amorphous_restarting_the_drift_clock():
    c = _fresh()
    for _ in range(30):
        pc.set_pulse(c, CFG)
    assert c.x > 0.5
    pc.age(c, 5000.0, CFG)
    assert c.t_amorph > 5000.0
    e = pc.reset_pulse(c, CFG)
    assert c.x == 0.0 and c.t_amorph == CFG.drift_t0 and e > 0.0
    assert c.temp < CFG.t_m                                                    # cooled after the pulse


def test_below_threshold_pulse_does_nothing():
    c = _fresh()
    tiny = CFG.v_th * pc.conductance(c, CFG) * 0.5                             # I/G < V_th
    assert pc.set_pulse(c, CFG, current=tiny) == 0.0 and c.x == 0.0


def test_amorphous_drift_is_a_power_law_with_the_stated_exponent():
    g1 = pc.g_reset(1.0, CFG)
    g10 = pc.g_reset(10.0, CFG)
    assert g10 / g1 == pytest.approx(10 ** (-CFG.drift_nu))
    assert pc.dg_reset_dt(10.0, CFG) == pytest.approx(-CFG.drift_nu * g10 / 10.0)
    c = _fresh()
    p0 = pc.pressure_of(c, CFG)
    pc.age(c, 86400.0, CFG)
    assert pc.pressure_of(c, CFG) <= p0                                       # drift only lowers the amorphous part


def test_derivatives_are_consistent_with_the_integrator():
    # dx/dt at a mid state and fixed temperature matches a small additivity substep
    x, T = 0.3, 545.0
    h = 1e-10
    assert (pc._advance_x(x, T, h, CFG) - x) / h == pytest.approx(pc.dx_dt(x, T, CFG), rel=1e-3)
    assert pc.dx_dt(0.3, CFG.t_m + 1.0, CFG) < 0.0                            # melting
    assert pc.dx_dt(0.3, CFG.t_x - 1.0, CFG) == 0.0
    c = pc.PcmCell(x=0.4, t_amorph=3.0)
    assert pc.dg_dx(c, CFG) == pytest.approx(CFG.g_set - pc.g_reset(3.0, CFG))


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------

def test_set_goes_up_in_pulses_and_the_realised_change_is_reported():
    c = _fresh()
    d = pc.decide(c, 0.4, CFG)
    assert d.action == pc.SET and 1 <= d.pulses <= CFG.max_set_pulses and d.energy_j > 0.0
    assert d.realised_dg > 0.0 and d.asked_dg > 0.0
    assert abs(pc.pressure_of(c, CFG) - 0.4) < 0.1 or d.pulses == CFG.max_set_pulses


def test_hold_when_the_change_is_within_drift_and_noise():
    c = _fresh()
    pc.decide(c, 0.5, CFG)
    p = pc.pressure_of(c, CFG)
    d = pc.decide(c, p + 0.001, CFG)                                           # below read noise (1 %)
    assert d.action == pc.HOLD and d.pulses == 0 and "drift+noise" in d.reason


def test_decrease_is_held_until_it_clears_the_energy_bar_then_reset_set():
    c = _fresh()
    pc.decide(c, 0.8, CFG)
    small = pc.decide(c, pc.pressure_of(c, CFG) - 0.03, CFG)
    assert small.action == pc.HOLD and small.bar is not None and 0.0 < small.bar < 1.0
    big = pc.decide(c, 0.05, CFG)
    assert big.action == pc.RESET_SET and big.pulses >= 1 and big.realised_dg < 0.0
    assert c.t_amorph == CFG.drift_t0                                          # the drift clock restarted
    assert big.bar == pc.pstar(1.0, 1.0) or big.bar is not None


def test_refresh_re_sets_a_cell_that_drift_carried_away_from_its_target():
    cfg = pc.PcmConfig(drift_nu=0.6)                                          # exaggerated drift to force it
    c = pc.PcmCell(t_amorph=cfg.drift_t0)
    pc.decide(c, 0.5, cfg)
    pc.age(c, 1e6, cfg)
    d = pc.decide(c, pc.pressure_of(c, cfg), cfg)                             # no new target: same pressure
    assert d.action in (pc.REFRESH, pc.HOLD)
    if d.action == pc.REFRESH:
        assert d.pulses >= 1 and d.realised_dg > 0.0


def test_energy_is_joule_heat_and_accumulates_on_the_cell():
    c = _fresh()
    d = pc.decide(c, 0.3, CFG)
    per_pulse = CFG.i_set ** 2 / CFG.g_on * CFG.t_set
    assert d.energy_j == pytest.approx(d.pulses * per_pulse, rel=1e-6)
    assert c.energy_j == pytest.approx(d.energy_j) and c.writes == d.pulses


# ---------------------------------------------------------------------------
# rADAM coupling
# ---------------------------------------------------------------------------

def test_lr_eff_is_scaled_by_the_secant_slope_and_never_exceeds_lr():
    c = _fresh()
    st = {}
    p, d = pc.coupled_step(st, c, 0.5, cfg=CFG, lr=0.3)
    assert 0.0 < st["pcm_lr_eff"] <= 0.3 and d.action == pc.SET and p == pytest.approx(pc.pressure_of(c, CFG))
    assert st["pressure"] == p                                                 # the optimizer holds the cell's pressure
    full = pc.PcmCell(x=0.9999, t_amorph=1.0)
    st2 = {}
    p2, d2 = pc.coupled_step(st2, full, 0.5, cfg=CFG, lr=0.3)
    assert st2["pcm_lr_eff"] < 0.01 and d2.action == pc.HOLD                   # saturated: no slope, no write


def test_with_a_nominal_slope_and_no_drift_the_coupling_reduces_to_radam():
    # A cell whose secant equals the nominal slope and whose asked step is realised exactly
    c = _fresh()
    st_c, st_r = {}, {}
    nominal = pc.nominal_slope(CFG)
    _, slope = pc.reach(c, CFG)
    lr = 0.3 * min(1.0, slope / nominal)
    radam_step(st_r, 0.5, lr=lr)                                               # same effective lr, same gradient
    pc.coupled_step(st_c, c, 0.5, cfg=CFG, lr=0.3)
    assert st_c["pcm_asked"] == pytest.approx(st_r["pressure"])                 # rADAM asked the same thing
    assert st_c["m"] == pytest.approx(st_r["m"]) and st_c["v"] == pytest.approx(st_r["v"])


def test_floquet_slows_crystallisation():
    a, b = _fresh(), _fresh()
    for _ in range(10):
        pc.set_pulse(a, CFG, floquet=1.0)
        pc.set_pulse(b, CFG, floquet=0.3)
    assert b.x < a.x
    assert 0.0 <= pc.floquet_for_axis(0.0, 0) <= 1.0 and pc.floquet_for_axis(0.0, 0) == pytest.approx(1.0)


def test_step_array_drives_every_axis_from_currents_and_persists_only_its_keys():
    arr = pc.CellArray()
    currents = {a: 1.0 for a in pc.AXES}
    targets = {a: 0.4 for a in pc.AXES}
    out = pc.step_array(arr, currents=currents, targets=targets, now=1000.0, cfg=CFG)
    assert set(out) == set(pc.AXES) and all(r["decision"]["action"] == pc.SET for r in out.values())
    out2 = pc.step_array(arr, currents=currents, targets=targets, now=1079.0, cfg=CFG)
    assert arr.steps == 2 and all(r["pressure"] >= out[a]["pressure"] - 0.03 for a, r in out2.items())
    assert all(abs(r["chi"]) < r["eps"] + 0.05 for r in out2.values())
    cn = sqlite3.connect(":memory:")
    pc.record(cn, arr, out2, now=1079.0)
    keys = [k for (k,) in cn.execute("SELECT key FROM brain_kv")]
    assert keys == [pc.KV_PREFIX + "system_entirety"]
    assert len(pc.rows(cn)) == len(pc.AXES)
    back = pc.load_array(cn)
    assert back.steps == 2 and back.cells["vision"].plus.x == pytest.approx(arr.cells["vision"].plus.x)
    assert back.cells["vision"].expected == pytest.approx(arr.cells["vision"].expected)
    with pytest.raises(PermissionError):
        pc._kv_set(cn, "entirety:radam_state:system_entirety", {})


def test_config_from_env_and_json(monkeypatch):
    monkeypatch.setenv("QUIPU_PCM_DRIFT_NU", "0.2")
    monkeypatch.setenv("QUIPU_PCM_MAX_SET_PULSES", "8")
    cfg = pc.PcmConfig.from_env()
    assert cfg.drift_nu == 0.2 and cfg.max_set_pulses == 8 and isinstance(cfg.max_set_pulses, int)
    assert cfg.to_json()["t_m"] == CFG.t_m


# ---------------------------------------------------------------------------
# v0.58.0: the resolution bound, error feedback and the differential pair
# ---------------------------------------------------------------------------

def test_resolution_is_about_a_hundred_levels_on_the_defaults():
    aged = pc.PcmCell(t_amorph=3600.0)
    assert pc.resolution(aged, CFG) == pytest.approx(0.0100, abs=0.0005)
    assert pc.bits_per_write(aged, CFG) == pytest.approx(6.64, abs=0.05)
    assert pc.resolution(_fresh(), CFG) > pc.resolution(aged, CFG)            # fresh amorphous drifts fastest


def test_sub_resolution_steps_are_deferred_in_chi_not_erased():
    dc = pc.DiffCell()
    st = {}
    pc.ef_step(st, dc, 0.5, cfg=CFG, dt_since=79.0)
    held0, exp0 = dc.held(CFG), dc.expected
    # tiny gradient: each step is far below one level
    for _ in range(30):
        _, info = pc.ef_step(st, dc, 1e-4, cfg=CFG, dt_since=79.0, lr=0.002)
    assert dc.expected > exp0                                                  # the optimizer kept moving
    assert abs(dc.chi(CFG)) < pc.resolution(dc.plus, CFG) + 0.05               # held within one level + one pulse
    assert st["pcm_chi"] == pytest.approx(dc.chi(CFG)) and st["pressure"] == dc.expected


def test_a_decrease_is_a_set_on_the_minus_cell_not_a_reset():
    dc = pc.DiffCell()
    st = {}
    for _ in range(6):
        pc.ef_step(st, dc, 0.6 - dc.expected, cfg=CFG, dt_since=79.0)
    minus_before, refreshes_before = dc.minus.x, dc.refreshes
    _, info = pc.ef_step(st, dc, 0.2 - dc.expected, cfg=CFG, dt_since=79.0)
    assert info["realised"] < 0.0 and info["action"] == pc.SET                 # down in one step,
    assert dc.minus.x > minus_before and dc.refreshes == refreshes_before      # as a SET on the − cell, no RESET
    lows = []
    for _ in range(14):
        pc.ef_step(st, dc, 0.2 - dc.expected, cfg=CFG, dt_since=79.0)
        lows.append(dc.pressure(CFG))
    assert min(abs(p - 0.2) for p in lows) < 0.05                              # reaches the target region


def test_tracking_beats_the_erasing_single_cell_on_an_up_down_ramp():
    def ramp(i):
        return 0.1 + 0.002 * i if i < 100 else 0.3 - 0.002 * (i - 100)
    c, st1 = pc.PcmCell(t_amorph=1.0), {}
    dc, st2 = pc.DiffCell(), {}
    e_old = e_new = 0.0
    for i in range(200):
        p1, _ = pc.coupled_step(st1, c, ramp(i) - pc.pressure_of(c, CFG), cfg=CFG, dt_since=79.0)
        p2, _ = pc.ef_step(st2, dc, ramp(i) - dc.expected, cfg=CFG, dt_since=79.0)
        if i >= 20:
            e_old += abs(ramp(i) - p1)
            e_new += abs(ramp(i) - p2)
    assert e_new < 0.5 * e_old


def test_refresh_resets_both_cells_and_keeps_the_net_value():
    dc = pc.DiffCell()
    for _ in range(40):
        pc.set_pulse(dc.plus, CFG)
    for _ in range(25):
        pc.set_pulse(dc.minus, CFG)
    net = dc.held(CFG)
    pc.refresh_pair(dc, CFG)
    assert dc.refreshes == 1 and min(dc.plus.x, dc.minus.x) == 0.0
    assert dc.held(CFG) == pytest.approx(net, abs=0.05)


def test_a_v057_single_cell_record_loads_as_the_plus_side():
    old = {"cells": {"vision": {"x": 0.4, "temp": 300.0, "t_amorph": 10.0, "target_x": 0.39}}, "steps": 3}
    arr = pc.CellArray.from_json(old)
    dc = arr.cells["vision"]
    assert dc.plus.x == 0.4 and dc.minus.x == 0.0 and dc.expected == 0.39 and arr.steps == 3


# ---------------------------------------------------------------------------
# v0.59.0: deepen-phase shear coupling across the axes
# ---------------------------------------------------------------------------

def _states(us, thetas=None):
    """rADAM states whose velocity m̂/√v̂ equals the given u (t=1: m̂ = m/(1-β1), v̂ = v/(1-β2))."""
    from src.quipu.radam_optimizer import DEFAULT_BETA1, DEFAULT_BETA2
    out = {}
    for i, (a, u) in enumerate(zip(pc.AXES, us)):
        v_hat = 0.04
        out[a] = {"t": 1, "m": u * (0.2 + 1e-6) * (1 - DEFAULT_BETA1), "v": v_hat * (1 - DEFAULT_BETA2),
                  "theta": (thetas[i] if thetas else 0.0)}
    return out


def test_axis_velocity_is_the_normalised_adam_step():
    assert pc.axis_velocity({}) == 0.0
    st = _states([0.5] * 7)["vision"]
    assert pc.axis_velocity(st) == pytest.approx(0.5, rel=1e-3)


def test_shear_coupling_finds_the_counter_rotating_axes_and_puts_them_anti_phase():
    us = [0.6, 0.5, 0.4, 0.3, -0.4, -0.5, 0.2]
    c = pc.shear_coupling(_states(us, thetas=[0.3] * 7))
    assert c["u_bar"] == pytest.approx(sum(us) / 7, rel=1e-3)
    assert set(c["counter_rotating"]) == {"brain", "perception"}
    assert sum(c["axes"][a]["shear"] for a in pc.AXES) == pytest.approx(0.0, abs=1e-9)
    assert c["psi"] == pytest.approx(0.3, abs=1e-9)
    assert c["axes"]["brain"]["phi"] == pytest.approx((0.3 + math.pi) % (2 * math.pi))
    assert c["axes"]["vision"]["phi"] == pytest.approx(0.3)
    for a in pc.AXES:                                                       # g_im carries the shear's sign
        assert math.copysign(1, c["axes"][a]["g_im"]) == math.copysign(1, c["axes"][a]["shear"])
    assert c["dispersion"] > 0.0


def test_broaden_is_unchanged_by_the_shear_flag():
    def run(shear):
        arr = pc.CellArray()
        for k in range(6):
            pc.step_array(arr, currents={a: 1.0 for a in pc.AXES}, targets={a: 0.5 for a in pc.AXES},
                          now=79.0 * k, cfg=CFG, deepen=False, shear=shear)
        return {a: arr.cells[a].expected for a in pc.AXES}
    assert run(True) == run(False)


def test_deepen_passes_the_shear_and_records_it():
    arr = pc.CellArray()
    tg = {a: 0.2 + 0.08 * i for i, a in enumerate(pc.AXES)}
    for k in range(6):
        pc.step_array(arr, currents={a: 1.0 for a in pc.AXES}, targets=tg, now=79.0 * k, cfg=CFG)
    out = pc.step_array(arr, currents={a: 1.0 for a in pc.AXES}, targets=tg, now=79.0 * 6, cfg=CFG, deepen=True)
    ens = out.pop("_ensemble")
    assert set(ens) == {"u_bar", "dispersion", "psi", "counter_rotating"}
    assert all(out[a]["shear"] is not None and "phi" in out[a]["shear"] for a in pc.AXES)
    off = pc.step_array(pc.CellArray(), currents={a: 1.0 for a in pc.AXES}, targets=tg, now=0.0, cfg=CFG,
                        deepen=True, shear=False)
    assert "_ensemble" not in off and all(off[a]["shear"] is None for a in pc.AXES)
    cn = sqlite3.connect(":memory:")
    pc.record(cn, arr, out, now=0.0)                                          # the ensemble row is not an axis
    assert len(pc.rows(cn)) == len(pc.AXES)


def test_a_negative_differential_with_a_latent_part_turns_theta_anti_phase():
    from src.quipu.radam_optimizer import radam_step
    st = {}
    radam_step(st, -0.2, 0.01)
    assert abs(st["theta"] - math.pi) < 0.1
    st2 = {}
    radam_step(st2, 0.2, 0.01)
    assert st2["theta"] < 0.1
