"""qpsi.hidden_work: the bounds of Muruga et al. (Nature Physics 2026) and the Φ(δt) memory test."""
import json
import sqlite3

import pytest

from quipu.qpsi import hidden_work as H


def test_source_and_published_numbers_are_recorded():
    assert H.SOURCE["doi"] == "10.1038/s41567-026-03442-w"
    assert H.SOURCE["data_doi"] == "10.5281/zenodo.19881972"
    s = H.PUBLISHED["setup"]
    assert (s.structural_relaxation_s, s.threshold_nm, s.extraction_time_s) == (6.0, 70.0, 3.0)
    assert H.PUBLISHED["phi_peak_delta_t_s"] == 1.0
    assert {f.state for f in H.PUBLISHED["findings"]} == {"|10⟩", "|11⟩", "|01⟩", "|00⟩"}


def test_validation_on_the_papers_data_agrees_with_the_paper():
    v = H.VALIDATION
    curve = v["phi_by_delta_s"]
    assert max(curve, key=curve.get) == v["peak_delta_s"] == H.PUBLISHED["phi_peak_delta_t_s"]
    assert curve[200.0] < 0.05 * curve[1.0]                  # memory gone at δt ≫ τ_r
    s = v["state_phi_at_1s"]
    assert s["|10⟩"] > 0 and s["|11⟩"] > 0 and s["|01⟩"] < 0 and abs(s["|00⟩"]) < 0.1
    assert v["energy_10_at_1s_kT"]["max"] > H.PUBLISHED["equilibrium_energy_kT"] > v["energy_10_at_1s_kT"]["start"]


def test_markov_bound_is_reached_by_a_jump_and_not_exceeded_by_slower_protocols():
    k, m = 2.0, 0.7
    assert H.markov_energetic_bound(k, m) == pytest.approx(0.49)
    jump = H.best_jump(k, m)
    assert jump["work"] == pytest.approx(jump["bound"])
    for t_f in (0.5, 3.0, 10.0):                       # constant-velocity translations, as in the experiment
        for lam in (0.2, 0.7, 1.4):
            assert H.translation_work(k, m, [(t_f, lam)]) <= jump["bound"] + 1e-9
    assert H.translation_work(k, m, [(0.0, m), (5.0, m)]) == pytest.approx(jump["bound"])


def test_a_hidden_force_lets_translation_work_exceed_the_markov_bound():
    """The paper's |10⟩ result in miniature: a backflow from hidden degrees of freedom keeps pushing the
    particle outward, so following it extracts more than (κ/2)⟨X⟩²."""
    k, m = 1.0, 0.5
    push = lambda t, x: 0.6                            # the hidden network pulling outward
    w = H.translation_work(k, m, [(0.0, m), (3.0, m + 0.9)], drift=push)
    assert w > 1.5 * H.markov_energetic_bound(k, m)
    assert H.translation_work(k, m, [(0.0, m), (3.0, m + 0.9)]) < H.markov_energetic_bound(k, m)


def test_information_bound():
    assert H.information_bound(0.5, kT=2.0) == pytest.approx(1.0)
    assert H.information_bound(0.5, delta_free_energy=-0.25) == pytest.approx(0.75)


@pytest.fixture(scope="module")
def traces():
    markov = H.coupled_trace(30000, coupling=0.0, bath_friction=10.0, dt=0.05, seed=3)
    memory = H.coupled_trace(30000, coupling=1.0, bath_friction=10.0, dt=0.05, seed=3)
    return markov, memory


def test_phi_is_zero_without_hidden_memory_and_positive_with_it(traces):
    markov, memory = traces
    kw = dict(dt=0.05, horizon=50, stride=4, threshold=1.0, nulls=6)
    m = H.memory_order_parameter(markov, 20, **kw)
    h = H.memory_order_parameter(memory, 20, **kw)
    assert abs(m.z) < 3
    assert h.z > 5 and h.phi > 0
    # the paper's sign pattern with a threshold above the mean: |10⟩ gains, |01⟩ loses
    assert h.by_state["|10⟩"]["phi"] > 0 > h.by_state["|01⟩"]["phi"]


def test_memory_scan_finds_memory_only_where_it_is(traces):
    markov, memory = traces
    kw = dict(dt=0.05, horizon=40, stride=6, threshold=1.0, nulls=4)
    assert H.memory_scan(markov, [5, 30], **kw)["markovian"]
    scan = H.memory_scan(memory, [5, 30], **kw)
    assert scan["significant"] and scan["peak"]["delta"] in (5, 30)


def test_second_measurement_adds_information_only_with_memory(traces):
    markov, memory = traces
    assert H.second_measurement_information(memory, 20, 10) > 3 * H.second_measurement_information(markov, 20, 10)


def test_record_writes_only_its_own_key():
    cn = sqlite3.connect(":memory:")
    cn.execute("CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    doc = H.record(cn)
    assert doc["checks"]["jump_to_mean_reaches_bound"] is True
    stored = json.loads(cn.execute("SELECT value FROM brain_kv WHERE key=?", (H.KV_KEY,)).fetchone()[0])
    assert stored["source"]["doi"] == H.SOURCE["doi"]
    with pytest.raises(PermissionError):
        H._kv_set(cn, "entirety:prior", {})
