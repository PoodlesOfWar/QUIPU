"""v0.62.0: shear persistence, Poisson weighting of the axes, the rheology
reading of the CRAM axis array, and relative winding between axis phases."""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

from src.quipu.qpsi import pcm_cells as pc
from src.quipu.qpsi import rheology as rh

CFG = pc.PcmConfig()
TG = {a: 0.2 + 0.1 * i for i, a in enumerate(pc.AXES)}


def _run(n=6):
    arr = pc.CellArray()
    for k in range(n):
        pc.step_array(arr, currents={a: 1.0 for a in pc.AXES}, targets=TG, now=79.0 * k, cfg=CFG)
    out = pc.step_array(arr, currents={a: 1.0 for a in pc.AXES}, targets=TG, now=79.0 * n, cfg=CFG, deepen=True)
    return arr, out


def test_poisson_weights_normalised_and_mean_tracks_lambda():
    assert pc.poisson_weights(0.0) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for lam in (0.03, 0.5, 1.0):
        w = pc.poisson_weights(lam)
        assert sum(w) == pytest.approx(1.0)
        assert sum(k * x for k, x in enumerate(w)) == pytest.approx(lam, rel=2e-3)   # tail at k >= 7 is tiny


def test_stp_lambda_is_zero_without_shear_and_one_without_mean():
    w = pc.poisson_weights(1.0)
    assert pc.stp_lambda({a: 0.3 for a in pc.AXES}, w) == pytest.approx(0.0)
    u = {a: (1.0 if i % 2 else -1.0) for i, a in enumerate(pc.AXES)}
    assert 0.0 < pc.stp_lambda(u, w) <= 1.0


def test_deepen_step_reports_poisson_ensemble_and_record_persists_it():
    arr, out = _run()
    ens = out["_ensemble"]
    assert ens["lambda_poisson"] == pytest.approx(ens["dispersion"] ** 2)
    assert len(ens["weights"]) == 7 and sum(ens["weights"]) == pytest.approx(1.0)
    assert 0.0 <= ens["lambda_stp"] <= 1.0
    cn = sqlite3.connect(":memory:")
    pc.record(cn, arr, out, now=999.0)
    row = cn.execute(f"SELECT lambda_poisson, lambda_stp, weights FROM {pc.ENSEMBLE_TABLE}").fetchone()
    assert row[0] == pytest.approx(ens["lambda_poisson"]) and len(json.loads(row[2])) == 7
    th = cn.execute(f"SELECT theta, shear_u FROM {pc.TABLE} WHERE axis='touch'").fetchone()
    assert th[0] is not None and th[1] is not None
    r = json.loads(cn.execute("SELECT value FROM brain_kv WHERE key=?",
                              (rh.KV_KEY + "system_entirety",)).fetchone()[0])
    assert 0.0 <= r["edge_weight"] <= 0.25 and r["at"] == 999.0


def test_broaden_step_has_no_ensemble_row():
    arr = pc.CellArray()
    out = pc.step_array(arr, currents={a: 1.0 for a in pc.AXES}, targets=TG, now=0.0, cfg=CFG)
    cn = sqlite3.connect(":memory:")
    pc.record(cn, arr, out, now=0.0)
    assert cn.execute(f"SELECT COUNT(*) FROM {pc.ENSEMBLE_TABLE}").fetchone()[0] == 0


def _programmed(theta: float) -> pc.CellArray:
    arr = pc.CellArray()
    for a in pc.AXES:
        dc = arr.cell(a)
        for _ in range(20):
            pc.set_pulse(dc.plus, CFG)
        arr.radam[a] = {"theta": theta}
    return arr


def test_equal_realised_and_latent_is_the_critical_gel():
    r = rh.reading(_programmed(math.pi / 4), CFG)
    assert r["beta"] == pytest.approx(math.pi / 4)
    assert r["edge_weight"] == pytest.approx(0.125)
    assert r["eta_ratio"] == pytest.approx(1.0)
    assert r["bits"] > 0.0


def test_all_realised_is_viscous_and_all_latent_is_elastic():
    assert rh.reading(_programmed(0.0), CFG)["edge_weight"] == pytest.approx(0.25)
    lat = rh.reading(_programmed(math.pi / 2), CFG)
    assert lat["edge_weight"] == pytest.approx(0.0, abs=1e-9) and lat["eta_ratio"] is None


def test_quiet_array_holds_previous_angle():
    r = rh.reading(pc.CellArray(), CFG, prev={"beta": 0.3})
    assert r["held"] and r["beta"] == pytest.approx(0.3)


def test_weyl_twist_reports_acquisition_residual():
    r = rh.reading(_programmed(math.pi / 4), CFG, weyl_twist=math.pi / 4)
    assert r["acquisition"] == pytest.approx(0.0, abs=1e-9)


def test_relative_winding_counts_turns_and_crossings():
    n = 200
    a = [4.0 * math.pi * i / (n - 1) for i in range(n)]       # two full turns
    w = rh.relative_winding(a, [0.0] * n)
    assert w["winding"] == pytest.approx(2.0, rel=1e-6)
    assert w["crossings"] == 2
    assert rh.relative_winding([0.1] * n, [0.0] * n)["crossings"] == 0      # one cell cannot wrap itself


def test_logged_winding_reads_theta_from_the_log():
    arr = pc.CellArray()
    cn = sqlite3.connect(":memory:")
    for k in range(8):
        out = pc.step_array(arr, currents={a: 1.0 for a in pc.AXES}, targets=TG, now=79.0 * k, cfg=CFG,
                            deepen=bool(k % 2))
        pc.record(cn, arr, out, now=79.0 * k)
    w = rh.logged_winding(cn, "vision", "touch")
    assert w["steps"] == 7


# ---------------------------------------------------------------------------
# v0.62.1: fractional order, viscoelastic Weyl phase, capstan hold, beta1
# ---------------------------------------------------------------------------

def test_fractional_order_from_loss_angle():
    assert rh.fractional_order(math.pi / 4) == pytest.approx(0.5)           # the critical gel
    assert rh.fractional_order(0.0) == 0.0 and rh.fractional_order(math.pi) == 1.0
    r = rh.reading(_programmed(math.pi / 4), CFG)
    assert r["fractional_order"] == pytest.approx(0.5)
    assert r["loss_factor"] == pytest.approx(math.sin(math.pi / 4))
    assert r["g_ratio"] == pytest.approx(1.0)                                # first reading is its own reference


def test_viscoelastic_weyl_phase_is_dimensionless_product():
    v = rh.viscoelastic_weyl_phase(1, 1.0, 0.586, 0.5)
    assert v == pytest.approx(math.sqrt(3 * math.pi) * 0.586 * math.sin(math.pi / 4))
    with pytest.raises(ValueError):
        rh.viscoelastic_weyl_phase(-1, 1.0, 0.586, 0.5)


def test_capstan_threshold_matches_euler_eytelwein():
    assert rh.capstan_threshold(1.0, 0.40, 4 * math.pi) == pytest.approx(math.exp(0.4 * 4 * math.pi))
    assert rh.capstan_threshold(0.01, 0.40, 4 * math.pi) > 1.0              # above the cell's full range


def test_cells_step_with_published_adam_beta1():
    st = {}
    dc = pc.DiffCell()
    pc.ef_step(st, dc, 0.5, cfg=CFG)
    assert st["m"] == pytest.approx((1.0 - pc.PCM_BETA1) * 0.5)
    assert pc.PCM_BETA1 == 0.9


# ---------------------------------------------------------------------------
# v0.63.0: model-sized terminal SET pulse
# ---------------------------------------------------------------------------

def _cell_at(x):
    c = pc.PcmCell(t_amorph=CFG.drift_t0)
    while c.x < x:
        pc.set_pulse(c, CFG)
    return c


def test_linear_width_scaling_undershoots_and_solved_width_lands():
    c = _cell_at(0.3)
    full = pc.pulse_dg(c, CFG)
    assert pc.pulse_dg(c, CFG, duration=0.3 * CFG.t_set) < 0.05 * 0.3 * full     # 15 ns barely heats
    w = pc.pulse_width_for(c, 0.3 * full, CFG)
    assert 0.6 * CFG.t_set < w < CFG.t_set
    assert pc.pulse_dg(c, CFG, duration=w) == pytest.approx(0.3 * full, rel=1e-3)


def test_terminal_pulse_removes_single_pulse_overshoot():
    lo, hi = pc.g_bounds(CFG)
    for x in (0.1, 0.4, 0.7):
        base = _cell_at(x)
        g0 = pc.conductance(base, CFG)
        ask = 0.4 * pc.pulse_dg(base, CFG) + CFG.read_noise * (hi - lo)       # one pulse would overshoot
        off, on = pc.replace(base), pc.replace(base)
        pc.set_toward(off, g0 + ask, CFG, pwm=False)
        pc.set_toward(on, g0 + ask, CFG, pwm=True)
        r_off = (pc.conductance(off, CFG) - g0) / ask
        r_on = (pc.conductance(on, CFG) - g0) / ask
        assert r_off > 1.15 and r_on == pytest.approx(1.0, abs=1e-3)


def test_pwm_switch_restores_full_pulses(monkeypatch):
    monkeypatch.setenv("QUIPU_PCM_PWM", "0")
    assert not pc.pwm_enabled()
    c = _cell_at(0.3)
    g0 = pc.conductance(c, CFG)
    e, n = pc.set_toward(c, g0 + 1e-6 + 0.02 * (pc.g_bounds(CFG)[1] - pc.g_bounds(CFG)[0]), CFG)
    assert n == 1 and e == pytest.approx(CFG.i_set ** 2 / CFG.g_on * CFG.t_set, rel=0.5)


# ---------------------------------------------------------------------------
# v0.64.0: per-axis readings and the knot hold on the observation record
# ---------------------------------------------------------------------------

def test_axis_readings_follow_each_axis_phase():
    arr = _programmed(0.0)
    arr.radam["smell"] = {"theta": math.pi / 2}
    ax = rh.reading(arr, CFG)["axes"]
    assert ax["vision"]["edge_weight"] == pytest.approx(0.25)
    assert ax["smell"]["edge_weight"] == pytest.approx(0.0, abs=1e-9)
    assert all(v["bits"] > 0.0 for v in ax.values())


def test_pairs_accumulate_relative_winding_across_publishes():
    cn = sqlite3.connect(":memory:")
    pc.ensure_tables(cn)
    arr = _programmed(0.0)
    n = 40
    for i in range(n + 1):
        arr.radam["touch"] = {"theta": 2.0 * math.pi * i / n}          # one full turn against vision
        r = rh.publish(cn, arr, now=float(i))
    p = r["pairs"]["vision|touch"]
    assert p["winding"] == pytest.approx(-1.0, abs=1e-6) and p["crossings"] == 1
    assert p["hold_factor"] == pytest.approx(math.exp(0.2 * 2 * math.pi))
    assert r["pairs"]["vision|smell"]["winding"] == pytest.approx(0.0)
    assert not p["would_freeze"]
