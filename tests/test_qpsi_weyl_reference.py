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


# ---------------------------------------------------------------------------
# The reference in QUIPU's constant Weyl fallbacks
# ---------------------------------------------------------------------------

def test_amplitude_defaults_to_the_papers_comparison_and_is_selectable(monkeypatch):
    monkeypatch.delenv(w.COSMOLOGY_ENV, raising=False)
    a = w.amplitude()
    assert a.cosmology == w.PLANCK18.name and a.measurement == w.DES_Y3_WEYL_CMB.name
    assert a.amplitude == pytest.approx(0.926, abs=0.002)
    monkeypatch.setenv(w.COSMOLOGY_ENV, "DES Y6 cosmic shear (NLA)")
    assert w.amplitude().cosmology == w.DES_Y6_NLA.name
    monkeypatch.setenv(w.COSMOLOGY_ENV, "no such cosmology")
    assert w.amplitude().cosmology == w.PLANCK18.name


def test_reference_tensor_is_the_neutral_midpoint_scaled_by_a_w(monkeypatch):
    monkeypatch.delenv(w.COSMOLOGY_ENV, raising=False)
    t = w.reference_tensor()
    assert len(t) == 5 and len(set(t)) == 1
    assert t[0] == pytest.approx(0.5 * w.amplitude().amplitude)


def test_weyl_tensor_wrapper_defers_to_a_stored_tensor():
    orig = lambda: [0.1, 0.2, 0.3, 0.4, 0.9]          # noqa: E731
    stored = w.wrap_weyl_tensor(orig, lambda: "[0.1, 0.2, 0.3, 0.4, 0.9]")
    assert stored() == [0.1, 0.2, 0.3, 0.4, 0.9]
    for raw in (None, "", "not json", "[1, 2]", '{"a": 1}', "[1, 2, 3, 4, NaN]"):
        fn = w.wrap_weyl_tensor(lambda: [0.5] * 5, lambda raw=raw: raw)
        assert fn() == w.reference_tensor(), raw
    broken = w.wrap_weyl_tensor(lambda: [0.5] * 5, lambda: (_ for _ in ()).throw(RuntimeError("db")))
    assert broken() == [0.5] * 5                      # any failure → the original
    assert stored.__wrapped__ is orig


def test_runtime_wrapper_fills_the_boost_only_when_nothing_set_it():
    def orig(cn, source_key=None):
        return {"weyl_boost": 1.0, "source_payload": payload, "weyl_phase": 0.0}
    payload = {}
    fn = w.wrap_resuscitation_runtime(orig, lambda cn: {})
    out = fn(None)
    assert out["weyl_boost"] == pytest.approx(w.amplitude().amplitude)
    assert out["weyl_boost_source"] == w.SOURCE_LABEL and out["weyl_phase"] == 0.0
    for rhythm, pl in (({"boost": 1.3}, {}), ({"lr_factor": 0.8}, {}), ({}, {"weyl_boost": 1.2})):
        payload = pl

        def orig2(cn, source_key=None, pl=pl):
            return {"weyl_boost": 9.9, "source_payload": pl}
        out = w.wrap_resuscitation_runtime(orig2, lambda cn, r=rhythm: r)(None)
        assert out["weyl_boost"] == 9.9 and "weyl_boost_source" not in out


@pytest.fixture
def mesh_isolated(tmp_path, monkeypatch):
    from src.quipu import mesh_slm
    from src.quipu.qpsi import self_organising as so
    db_file = tmp_path / "brain.sqlite"

    def _open_conn(timeout: float = 30, path=None):
        cn = sqlite3.connect(db_file, timeout=timeout)
        cn.row_factory = sqlite3.Row
        return cn
    monkeypatch.setattr(mesh_slm, "_open_conn", _open_conn)
    with mesh_slm._conn() as cn:
        cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    monkeypatch.delenv(w.COSMOLOGY_ENV, raising=False)
    so.disable()
    yield mesh_slm, so, db_file
    so.disable()


def _flags(so, **on):
    base = dict(flux_phase=False, learned_prior=False, somn=False, mirror_training=False,
                coherency_depth=False, annealed_senses=False, resonance_phase=False, weyl_reference=False)
    base.update(on)
    return so.Flags(**base)


def test_enabled_through_self_organising_the_live_functions_use_the_reference(mesh_isolated):
    mesh_slm, so, db_file = mesh_isolated
    orig_tensor, orig_runtime = mesh_slm._weyl_tensor, mesh_slm._resuscitation_runtime
    assert orig_tensor() == [0.5] * 5                                    # the constant today
    so.enable(_flags(so, weyl_reference=True))
    assert mesh_slm._weyl_tensor() == w.reference_tensor()
    with mesh_slm._conn() as cn:
        rt = mesh_slm._resuscitation_runtime(cn)
    assert rt["weyl_boost"] == pytest.approx(w.amplitude().amplitude)
    # a stored tensor and a rhythm boost win
    with mesh_slm._conn() as cn:
        cn.execute("INSERT INTO brain_kv(key, value) VALUES(?, ?)", (w.TENSOR_KEY, "[0.2, 0.3, 0.4, 0.6, 0.7]"))
        cn.execute("INSERT INTO brain_kv(key, value) VALUES(?, ?)", (w.RHYTHM_KEY, '{"boost": 1.25}'))
    assert mesh_slm._weyl_tensor() == [0.2, 0.3, 0.4, 0.6, 0.7]
    with mesh_slm._conn() as cn:
        assert mesh_slm._resuscitation_runtime(cn)["weyl_boost"] == 1.25
    so.disable()
    assert mesh_slm._weyl_tensor is orig_tensor and mesh_slm._resuscitation_runtime is orig_runtime


def test_flag_off_leaves_mesh_slm_untouched(mesh_isolated):
    mesh_slm, so, _ = mesh_isolated
    orig_tensor, orig_runtime = mesh_slm._weyl_tensor, mesh_slm._resuscitation_runtime
    so.enable(_flags(so, weyl_reference=False))
    assert mesh_slm._weyl_tensor is orig_tensor and mesh_slm._resuscitation_runtime is orig_runtime
