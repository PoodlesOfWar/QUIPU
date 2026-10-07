"""qpsi.weyl_reference — DES Weyl potential measurements against the GR prediction."""
from __future__ import annotations

import math
import sqlite3

import pytest

from src.quipu.qpsi import weyl_reference as w


def test_growth_is_exact_in_einstein_de_sitter():
    eds = w.Cosmology("EdS", omega_m=1.0, sigma8=0.8)
    for z in (0.0, 0.5, 1.0, 3.0):
        g = w.growth(z, eds)
        assert g.d_norm == pytest.approx(1.0 / (1.0 + z), rel=1e-6)
        assert g.f == pytest.approx(1.0, abs=1e-6)
        # Ω_m(a) = 1 always, D ∝ a: ĵ ∝ a and its log rate is +1
        assert w.weyl_decay_rate(z, eds) == pytest.approx(1.0, abs=1e-6)


def test_lcdm_growth_matches_the_standard_values():
    c = w.Cosmology("t", omega_m=0.3, sigma8=0.8)
    assert w.growth(0.0, c).f == pytest.approx(0.3 ** 0.55, abs=0.005)
    assert w.growth(1.0, c).d_norm == pytest.approx(0.612, abs=0.002)
    assert w.omega_m_of_a(1.0, c) == pytest.approx(0.3)


def test_j_hat_is_the_paper_definition():
    c = w.PLANCK18
    z = 0.5
    expect = w.omega_m_of_a(1 / 1.5, c) * w.growth(z, c).d_norm * c.sigma8
    assert w.j_hat(z, c) == pytest.approx(expect)
    assert w.j_hat(z, c, sigma_mg=1.1) == pytest.approx(1.1 * expect)
    assert w.j_hat(0.0, c) == pytest.approx(c.omega_m * c.sigma8)


def test_the_weyl_potential_decays_at_late_times_and_grew_earlier():
    rows = w.curve(w.PLANCK18)
    rates = {r["z"]: r["weyl_decay_rate"] for r in rows}
    assert rates[0.0] < -1.0 and rates[3.0] > 0.5
    # ĵ peaks where the rate changes sign, between z = 0.75 and 1.0 for Planck
    assert rates[0.75] < 0.0 < rates[1.0]


def test_des_y3_weyl_bins_sit_below_planck_lcdm():
    r = w.confirm(w.PLANCK18, w.DES_Y3_WEYL_CMB)
    assert [b.z for b in r.bins] == [0.295, 0.467, 0.626, 0.771]
    assert r.bins[1].pull < -2.0                 # the 0.467 bin, lowest relative to prediction
    assert abs(r.bins[2].pull) < 0.5             # 0.626 compatible
    assert r.amplitude < 1.0 and r.amplitude_pull < -2.0
    assert r.dof == 4 and r.chi2 == pytest.approx(sum(b.pull ** 2 for b in r.bins))


def test_lower_s8_cosmologies_and_evolving_dark_energy_reduce_the_mismatch():
    m = w.DES_Y3_WEYL_CMB
    planck = w.confirm(w.PLANCK18, m)
    for c in (w.DES_Y6_NLA, w.DES_Y6_TATT, w.W0WA_2026):
        r = w.confirm(c, m)
        assert r.chi2 < planck.chi2
        assert abs(r.amplitude - 1.0) < abs(planck.amplitude - 1.0)


def test_s8_point_value_and_recorded_results():
    assert w.DES_Y6_NLA.s8 == pytest.approx(0.763 * math.sqrt(0.332 / 0.3))
    names = [r.name for r in w.S8_RESULTS]
    assert "DES Y6 cosmic shear, NLA" in names and "Planck 2018 TT,TE,EE+lowE+lensing" in names
    for r in w.S8_RESULTS:
        assert r.source and r.err_lo > 0 and r.err_hi > 0


def test_reference_is_complete_and_json_serialisable():
    import json
    doc = w.reference()
    assert len(doc["confirmations"]) == len(w.COSMOLOGIES) * len(w.MEASUREMENTS)
    assert set(doc["curves"]) == {c.name for c in w.COSMOLOGIES}
    json.dumps(doc, ensure_ascii=False)
    s = w.summary(doc)
    assert len(s["confirmations"]) == len(doc["confirmations"])


def test_record_writes_only_its_own_key():
    cn = sqlite3.connect(":memory:")
    cn.execute("CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    w.record(cn)
    assert [k for (k,) in cn.execute("SELECT key FROM brain_kv")] == [w.KV_KEY]
    with pytest.raises(PermissionError):
        w._kv_set(cn, "learnings:weyl_tensor", [0.5] * 5)


def test_negative_redshift_is_rejected():
    with pytest.raises(ValueError):
        w.growth(-0.1, w.PLANCK18)
