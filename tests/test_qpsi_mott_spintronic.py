"""mott_spintronic: entropically coupled LIF Mott neurons writing spikes into spintronic memory;
poissonic ADAM (potential) vs rADAM (realisation)."""
from __future__ import annotations

import math

import pytest

from src.quipu.qpsi import mott_spintronic as ms
from src.quipu.qpsi import rheology as rh

PHI = (1.0 + math.sqrt(5.0)) / 2.0
DRIVE = [0.08, 0.10, 0.12, 0.07, 0.09, 0.11, 0.06]


def test_shear_is_zero_when_planes_agree_and_bounded_otherwise():
    w = rh.axis_weights()
    assert ms.multiplanar_shear([0.4] * 7, w) == [0.0] * 7
    sh = ms.multiplanar_shear([0.0, 1.0, 0.2, 0.9, 0.5, 0.1, 0.7], w)
    assert all(-1.0 < x < 1.0 for x in sh) and sh[0] > 0 > sh[1]


def test_shadow_entropy_bounds():
    assert ms.shadow_entropy([0.0] * 7) == 0.0
    assert ms.shadow_entropy([1.0] + [0.0] * 6) == 0.0
    assert ms.shadow_entropy([0.3] * 7) == pytest.approx(1.0)


def test_neuron_integrates_leaks_fires_and_resets():
    cfg = ms.MottSpinConfig(coupling=0.0)
    st = ms.MottSpinState()
    below = [0.04] * 7                       # 0.04·τ = 0.8 < V_th: never fires
    for _ in range(400):
        ev = ms.step(st, below, cfg)
    assert sum(st.spikes) == 0 and st.v[0] == pytest.approx(0.8, abs=1e-3) and ev["fired"] == [0] * 7
    st = ms.MottSpinState()
    fired_at = [n for n in range(200) if ms.step(st, [0.1] * 7, cfg)["fired"][0]]
    assert fired_at and st.v[0] < cfg.v_threshold
    gaps = {b - a for a, b in zip(fired_at, fired_at[1:])}
    assert len(gaps) == 1                    # regular, event-driven firing
    assert 1.0 / gaps.pop() == ms.lif_rate(0.1, cfg)


def test_racetrack_counts_spikes_until_full_then_they_stay_latent():
    cfg = ms.MottSpinConfig(coupling=0.0, racetrack_sites=3)
    st = ms.MottSpinState()
    for _ in range(300):
        ms.step(st, [0.1] * 7, cfg)
    assert st.stored == [3] * 7 and st.written == [3] * 7 and st.spikes[0] > 3
    assert ms.read_and_clear(st) == [3] * 7 and st.stored == [0] * 7


def test_sot_mtj_holds_one_bit():
    cfg = ms.MottSpinConfig(coupling=0.0, memory="sot_mtj")
    st = ms.MottSpinState()
    for _ in range(100):
        ms.step(st, [0.1] * 7, cfg)
    assert st.stored == [1] * 7


def test_no_amplifier_a_weak_spike_is_not_written():
    cfg = ms.MottSpinConfig(coupling=0.0, spike_current=0.5, critical_current=0.8)
    st = ms.MottSpinState()
    for _ in range(100):
        ms.step(st, [0.1] * 7, cfg)
    assert sum(st.spikes) > 0 and sum(st.written) == 0 and st.stored == [0] * 7


def test_entropic_coupling_pulls_planes_together():
    def spread(coupling):
        cfg = ms.MottSpinConfig(coupling=coupling)
        st = ms.MottSpinState(v=[0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.5])
        ms.step(st, [0.0] * 7, cfg)
        return max(st.v) - min(st.v)
    assert spread(0.25) < spread(0.0)
    assert spread(0.02) < spread(0.0)


def test_poisson_score_and_poadam():
    assert ms.poisson_score(9, 9) == 0.0
    assert ms.poisson_score(13, 9) == pytest.approx(4 / 3)
    with pytest.raises(ValueError):
        ms.poisson_score(-1, 2)
    st = {"pressure": 0.5}
    p = ms.poissonic_adam_step(st, 13, 9)
    assert 0.5 < p <= 1.0 and st["t"] == 1


def test_fully_written_window_is_identifiable_delta_zero_phi():
    cfg = ms.MottSpinConfig(racetrack_sites=10_000)
    out = ms.run_window(ms.MottSpinState(), DRIVE, 300, cfg)
    assert out["spikes"] > 0 and out["latent"] == 0 and out["identifiable"] is True
    assert out["p_radam"] == out["p_poadam"] and out["radam_differential"]["delta"] == 0.0
    assert out["stability_number"] == pytest.approx(PHI)


def test_latent_spikes_open_a_differential_and_tighten_stability():
    cfg = ms.MottSpinConfig(racetrack_sites=2)
    po, r, st = {}, {}, ms.MottSpinState()
    ms.run_window(st, DRIVE, 300, cfg, po_state=po, r_state=r)
    ms.read_and_clear(st)
    out = ms.run_window(st, DRIVE, 300, cfg, po_state=po, r_state=r)
    assert out["latent"] > 0 and out["identifiable"] is False
    assert out["radam_differential"]["delta"] != 0.0
    assert 1.0 / PHI <= out["stability_number"] < PHI


def test_differential_feeds_the_viscous_coupler():
    np = pytest.importorskip("numpy")
    from src.quipu.qpsi import viscous_interaction as vi
    out = ms.run_window(ms.MottSpinState(), DRIVE, 300, ms.MottSpinConfig(racetrack_sites=2))
    coupler = vi.TypedViscousCoupler(ns_cfg=vi.NavierStokesConfig(dims=(8, 8), nu=0.03, dt=0.01))
    ns0 = vi.NavierStokesState.taylor_green(coupler.ns_cfg, amplitude=0.4)
    g = vi.VNNState(node_ids=[0, 1, 2], Z=np.zeros((3, 7)), edges=[(0, 1, 0.9), (1, 2, 0.9)])
    rep = coupler.step(ns0, g, radam_differential=out["radam_differential"])[2]
    assert rep.cantilever["graph_cfl_max"] == pytest.approx(out["stability_number"])


def test_config_and_inputs_are_validated():
    with pytest.raises(ValueError):
        ms.MottSpinConfig(memory="flash").validate()
    with pytest.raises(ValueError):
        ms.step(ms.MottSpinState(), [0.1] * 6)
    with pytest.raises(ValueError):
        ms.step(ms.MottSpinState(), [float("nan")] * 7)
    with pytest.raises(ValueError):
        ms.run_window(ms.MottSpinState(), DRIVE, 0)
