"""viscous_interaction: stability number φ and φ-capped cantilevered muonic Bayesian weights."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.quipu import ueqgm_engine as ue
from src.quipu.qpsi import viscous_interaction as vi

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _graph(n=6):
    rng = np.random.default_rng(3)
    Z = rng.uniform(-1.0, 1.0, (n, 7))
    edges = [(i, (i + 1) % n, 0.9) for i in range(n)]
    return vi.VNNState(node_ids=list(range(n)), Z=Z, edges=edges)


def test_stability_number_is_phi_not_1_8():
    assert vi.PHI == pytest.approx(PHI)
    assert vi.VNNConfig().graph_cfl_max == pytest.approx(PHI)
    with pytest.raises(ValueError):
        vi.VNNConfig(graph_cfl_max=1.8).validate()


def test_substeps_keep_every_substep_at_or_below_phi():
    cfg = vi.VNNConfig(nu_s=4.0, kappa_s=2.0, dt=0.5)
    stab = vi.check_vnn_stability(_graph(), cfg)
    assert stab["graph_cfl"] > PHI and stab["substeps"] > 1
    assert stab["graph_cfl"] / stab["substeps"] <= PHI and stab["stable"] is True
    _, diag = vi.vnn_step(_graph(), cfg)
    assert diag["graph_cfl"] <= PHI + 1e-12 and diag["finite"] is True


def test_muonic_prior_is_the_antimony_decay_branch():
    total = ue.muonic_disappearance_rate()
    decay = ue._SB_HUFF_FACTOR / ue._MUON_FREE_LIFETIME_S
    assert vi.muonic_decay_prior() == pytest.approx(decay / total)
    assert 0.03 < vi.muonic_decay_prior() < 0.045


def test_cantilever_weight_is_bayesian_nonlinear_and_bounded():
    pi = vi.muonic_decay_prior()
    assert vi.cantilever_weight(0.0) == 0.0 and vi.cantilever_weight(1.0) == pytest.approx(1.0)
    s = 0.7
    d = s * s * (3 - s) / 2
    assert vi.cantilever_weight(s) == pytest.approx(pi * d / (pi * d + (1 - pi) * (1 - d)))
    xs = np.linspace(0, 1, 51)
    cs = [vi.cantilever_weight(x) for x in xs]
    assert all(b >= a for a, b in zip(cs, cs[1:]))            # monotone
    assert cs[25] < 0.5 * 0.5                                  # held back below the tip (nonlinear)
    assert vi.cantilever_weight(5.0) == vi.cantilever_weight(1.0)
    assert vi.cantilever_weight(0.5, prior=0.5) == pytest.approx(0.3125)  # flat prior → pure deflection
    with pytest.raises(ValueError):
        vi.cantilever_weight(float("nan"))
    with pytest.raises(TypeError):
        vi.cantilever_weight(1j)


def test_every_factor_is_capped_by_phi():
    for s in np.linspace(-1, 3, 41):
        f = vi.phi_capped_factor(s)
        assert 1.0 <= f <= PHI + 1e-12
    assert vi.phi_capped_factor(1.0) == pytest.approx(PHI)


def _coupled(**kw):
    coupler = vi.TypedViscousCoupler(
        ns_cfg=vi.NavierStokesConfig(dims=(8, 8), nu=0.03, dt=0.01),
        vnn_cfg=vi.VNNConfig(nu_s=0.10, kappa_s=0.05, forcing_gain=0.08, dt=0.10),
    )
    ns0 = vi.NavierStokesState.taylor_green(coupler.ns_cfg, amplitude=0.4)
    return coupler, coupler.step(ns0, _graph(), target_mesh7=[0.5] * 7, **kw)[2]


def test_coupler_saturated_inputs_stop_at_phi():
    coupler, rep = _coupled(
        rheology_reading={"edge_weight": 10.0, "loss_angle": math.pi / 2},
        shear_reading={"lambda_rel_stp": 1.0},
        bathymetry_features={"seabed_slope": 50.0},
    )
    assert rep.effective_nu_sem == pytest.approx(coupler.vnn_cfg.nu_s * PHI)
    assert rep.effective_kappa_sem == pytest.approx(coupler.vnn_cfg.kappa_s * PHI)
    assert 1.0 <= rep.forcing_scale <= PHI
    assert rep.effective_nu_phys <= coupler.ns_cfg.nu * PHI + 1e-15
    c = rep.cantilever
    assert c["phi"] == pytest.approx(PHI) and c["graph_cfl_max"] == pytest.approx(PHI)
    assert c["weights"]["kappa_sem"] == pytest.approx(1.0)


def test_coupler_quiet_inputs_are_held_near_one():
    coupler, rep = _coupled(rheology_reading={"edge_weight": 0.125}, shear_reading={"lambda_rel_stp": 0.12})
    assert rep.effective_kappa_sem / coupler.vnn_cfg.kappa_s < 1.01
    assert rep.effective_nu_sem / coupler.vnn_cfg.nu_s < 1.02


def test_coupler_refuses_imaginary_and_non_finite_readings():
    with pytest.raises(TypeError):
        _coupled(shear_reading={"lambda_rel_stp": 1j})
    with pytest.raises(ValueError):
        _coupled(rheology_reading={"edge_weight": float("inf")})
    with pytest.raises(ValueError):
        vi.TypedViscousCoupler(muonic_prior=1.0)


# ---- relational stability number: poADAM vs rADAM differential ----------------

STATE = {"m": 0.02, "v": 0.001, "t": 3, "pressure": 0.4, "pivot_ema": 0.4, "theta": 0.3}


def test_identity_knobs_give_zero_differential_and_phi():
    d = vi.poadam_radam_differential(STATE, 0.3)
    assert d["delta"] == 0.0 and d["p_radam"] == d["p_poadam"]
    assert d["stability_number"] == pytest.approx(PHI)


def test_differential_does_not_mutate_state():
    before = dict(STATE)
    vi.poadam_radam_differential(STATE, 0.3, 0.5, use_torus=True, external_phase=1.0)
    assert STATE == before


def test_relational_knobs_move_delta_and_tighten_the_number():
    d = vi.poadam_radam_differential(STATE, 0.3, 0.8, use_torus=True, external_phase=2.0)
    assert d["delta"] != 0.0
    assert d["delta"] == pytest.approx(d["p_radam"] - d["p_poadam"])
    assert 1.0 / PHI <= d["stability_number"] < PHI
    assert d["stability_number"] == pytest.approx(PHI ** (1 - 2 * abs(d["delta"])))


def test_explicit_potential_is_the_poadam_side():
    d = vi.poadam_radam_differential(STATE, 0.3, potential=0.9)
    assert d["p_poadam"] == 0.9
    assert d["delta"] == pytest.approx(d["p_radam"] - 0.9)
    with pytest.raises(ValueError):
        vi.poadam_radam_differential(STATE, 0.3, potential=float("nan"))


def test_relational_stability_number_bounds_and_symmetry():
    assert vi.relational_stability_number(0.0) == pytest.approx(PHI)
    assert vi.relational_stability_number(1.0) == pytest.approx(1.0 / PHI)
    assert vi.relational_stability_number(-1.0) == vi.relational_stability_number(1.0)
    assert vi.relational_stability_number(0.5) == pytest.approx(1.0)
    assert vi.relational_stability_number(7.0) == pytest.approx(1.0 / PHI)
    xs = np.linspace(0, 1, 21)
    ss = [vi.relational_stability_number(x) for x in xs]
    assert all(b < a for a, b in zip(ss, ss[1:]))
    with pytest.raises(ValueError):
        vi.relational_stability_number(float("inf"))


def test_coupler_uses_the_relational_stability_number():
    _, rep0 = _coupled()
    assert rep0.cantilever["graph_cfl_max"] == pytest.approx(PHI) and rep0.cantilever["radam_delta"] == 0.0
    _, rep = _coupled(radam_differential={"delta": 0.5})
    assert rep.cantilever["graph_cfl_max"] == pytest.approx(1.0)
    assert rep.vnn_diagnostics["graph_cfl"] <= 1.0 + 1e-12
    with pytest.raises(ValueError):
        _coupled(radam_differential={"delta": float("nan")})
