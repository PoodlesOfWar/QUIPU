"""v0.67.0: the LiTaSb bond-rupture force as a multiplanar shear on an
exciton-polariton condensate into the PC / CR-39 optics and asphalt pyrolysis
(qpsi/bond_shear.py)."""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

from src.quipu.qpsi import bond_shear as bs
from src.quipu.qpsi import pcm_cells as pc


def _db(coherence: float | None = 0.8) -> sqlite3.Connection:
    cn = sqlite3.connect(":memory:")
    cn.execute("CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    if coherence is not None:
        cn.execute("INSERT INTO brain_kv VALUES(?,?,?)",
                   (bs.ANALOG_COHERENCE_KEY, json.dumps({"coherence": coherence}), ""))
    return cn


# ---------------------------------------------------------------------------
# Morse rupture force
# ---------------------------------------------------------------------------

def test_morse_rupture_force_is_half_a_times_depth_in_nn():
    bond = bs.MorseBond(d_e_ev=2.0, r_e_angstrom=2.5, a_inv_angstrom=1.5)
    assert bond.rupture_force_nn() == pytest.approx(0.5 * 1.5 * 2.0 * bs.EV_PER_ANGSTROM_IN_NN)
    assert bond.rupture_length_angstrom() == pytest.approx(2.5 + math.log(2.0) / 1.5)


def test_litasb_breaks_at_its_weakest_bond_and_is_flagged_estimate():
    rup = bs.litasb_rupture(bs.BondShearConfig())
    assert rup["litasb_nn"] == pytest.approx(min(rup["li_sb_nn"], rup["ta_sb_nn"]))
    assert rup["weakest_link"] == "li_sb"
    assert 1.0 <= rup["litasb_nn"] <= 4.0        # single-bond AFM rupture order
    assert rup["estimate"] is True


def test_env_override_changes_the_default_bond(monkeypatch):
    monkeypatch.setenv(bs.ENV_PREFIX + "LI_SB_D_E_EV", "4.0")
    monkeypatch.setenv(bs.ENV_PREFIX + "FORCE_NN", "0.5")
    cfg = bs.BondShearConfig.from_env()
    assert cfg.li_sb_d_e_ev == 4.0 and cfg.force_nn == 0.5
    monkeypatch.setenv(bs.ENV_PREFIX + "LI_SB_D_E_EV", "not-a-number")
    assert bs.BondShearConfig.from_env().li_sb_d_e_ev == bs.BondShearConfig().li_sb_d_e_ev


# ---------------------------------------------------------------------------
# Multiplanar shear and the condensate
# ---------------------------------------------------------------------------

def test_multiplanar_shear_spreads_the_ratio_over_the_axis_planes():
    sh = bs.multiplanar_shear(2.0, 2.0)
    assert sh["shear_ratio"] == pytest.approx(1.0)
    assert set(sh["planes"]) == set(pc.AXES)
    assert sum(p["ratio"] for p in sh["planes"].values()) == pytest.approx(1.0, abs=1e-5)
    assert all(p["ratio"] < 1.0 for p in sh["planes"].values())
    assert bs.multiplanar_shear(5.0, 2.0)["shear_ratio"] == 1.0      # clipped
    assert bs.multiplanar_shear(1.0, 0.0)["shear_ratio"] == 0.0      # no rupture force → no ratio


def test_landau_survival_bounds():
    assert bs.landau_survival(0.8, 0.0) == pytest.approx(0.8)
    assert bs.landau_survival(0.8, 1.0) == 0.0
    assert 0.0 < bs.landau_survival(0.8, 0.5) < 0.8
    assert bs.landau_survival(1.5, 0.0) == 1.0


def test_condensate_reading_defaults_without_brain():
    assert bs.condensate_reading(None) == {"R": 0.5, "phi": 0.0, "amplitude": pytest.approx(math.sqrt(0.5))}
    cn = _db(0.64)
    cn.execute("INSERT INTO brain_kv VALUES(?,?,?)",
               (bs.RESONANCE_KEY, json.dumps({"realised": 0.0, "latent": 1.0}), ""))
    rd = bs.condensate_reading(cn)
    assert rd["R"] == pytest.approx(0.64) and rd["amplitude"] == pytest.approx(0.8)
    assert rd["phi"] == pytest.approx(math.pi / 2)


# ---------------------------------------------------------------------------
# Optics and pyrolysis
# ---------------------------------------------------------------------------

def test_fresnel_transmission_is_one_at_index_match_and_below_one_otherwise():
    assert bs.fresnel_transmission(1.5, 1.5) == 1.0
    t_pc = bs.fresnel_transmission(2.7, 1.586)
    t_cr = bs.fresnel_transmission(2.7, 1.498)
    assert 0.9 < t_cr < t_pc < 1.0


def test_arrhenius_rate_rises_with_temperature():
    k_cold = bs.arrhenius_pyrolysis_rate(1.55, 1e10, 600.0)
    k_hot = bs.arrhenius_pyrolysis_rate(1.55, 1e10, 723.0)
    assert 0.0 < k_cold < k_hot
    assert k_hot == pytest.approx(1e10 * math.exp(-1.55 / (pc.K_B * 723.0)))


def test_ns_forcing_profile_is_an_antisymmetric_couette_ramp():
    prof = bs.ns_forcing_profile(5, 0.5)
    assert prof[0] == pytest.approx(-0.5) and prof[-1] == pytest.approx(0.5) and prof[2] == pytest.approx(0.0)
    assert bs.ns_forcing_profile(1, 0.7) == [0.0]


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------

def test_evaluate_yield_is_bounded_and_composed():
    rec = bs.evaluate(_db(0.8), now=1.0)
    assert rec["applied_force_nn"] == pytest.approx(rec["rupture"]["litasb_nn"])
    assert rec["shear_ratio"] == pytest.approx(1.0)
    assert 0.0 < rec["condensate"]["survival"] <= 0.8
    assert 0.0 < rec["optical"]["throughput"] < 1.0
    assert 0.0 < rec["pyrolysis"]["remaining"] < 1.0
    assert rec["pyrolysis"]["textile_fraction"] == 0.0      # pure muonic / linear regime
    expected = rec["condensate"]["survival"] * rec["optical"]["throughput"] * rec["pyrolysis"]["remaining"]
    assert rec["yield"] == pytest.approx(expected, abs=1e-5)
    assert rec["at"] == 1.0


def test_zero_force_leaves_the_condensate_whole():
    rec = bs.evaluate(_db(0.8), cfg=bs.BondShearConfig(force_nn=0.0))
    assert rec["shear_ratio"] == 0.0
    assert rec["condensate"]["survival"] == pytest.approx(0.8, abs=1e-5)


def test_publish_writes_only_its_own_prefix_and_latest_reads_it_back():
    cn = _db(0.8)
    rec = bs.publish(cn, now=2.0)
    keys = [r[0] for r in cn.execute("SELECT key FROM brain_kv")]
    assert bs.KV_LATEST in keys
    assert all(k == bs.ANALOG_COHERENCE_KEY or k.startswith(bs.KV_PREFIX) for k in keys)
    assert bs.latest(cn)["yield"] == rec["yield"]
    assert bs.shear_reading(cn) == {"lambda_rel_stp": pytest.approx(rec["shear_ratio"])}
    with pytest.raises(PermissionError):
        bs._kv_set(cn, "entirety:pcm:rheology:x", {})


def test_shear_reading_without_a_published_record_evaluates_fresh():
    cn = _db(0.8)
    assert bs.shear_reading(cn)["lambda_rel_stp"] == pytest.approx(1.0)
    assert bs.latest(cn) == {}
