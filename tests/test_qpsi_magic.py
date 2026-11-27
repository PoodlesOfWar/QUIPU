"""magic: stabilizer Rényi entropy of a per-plane matter-antimatter pair."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.quipu.qpsi import magic as M

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)


def _exact(theta, phi):
    psi = np.zeros(4, dtype=complex)
    psi[0], psi[3] = math.cos(theta / 2), np.exp(1j * phi) * math.sin(theta / 2)
    ex = [float(np.real(psi.conj() @ np.kron(a, b) @ psi)) for a in (I2, X, Y, Z) for b in (I2, X, Y, Z)]
    return M.stabilizer_renyi_2(ex)


@pytest.mark.parametrize("p,phi", [(0.1, 0.0), (0.3, 0.4), (0.5, 1.1), (2 / 3, math.pi / 4), (0.9, 2.5)])
def test_closed_form_matches_the_full_16_pauli_computation(p, phi):
    theta = 2 * math.asin(math.sqrt(p))
    assert M.pair_magic(p, phi) == pytest.approx(_exact(theta, phi), abs=1e-9)


def test_stabilizer_states_have_no_magic():
    assert M.pair_magic(0.0, 0.3) == pytest.approx(0.0, abs=1e-12)     # |00>
    assert M.pair_magic(1.0, 0.3) == pytest.approx(0.0, abs=1e-12)     # |11>
    assert M.pair_magic(0.5, 0.0) == pytest.approx(0.0, abs=1e-12)     # Bell state
    assert M.pair_magic(0.5, math.pi / 2) == pytest.approx(0.0, abs=1e-12)


def test_maximum_is_log2_three_halves():
    best = max(M.pair_magic(p, ph) for p in np.linspace(0, 1, 101) for ph in np.linspace(0, math.pi, 41))
    assert best <= M.MAX_PAIR_MAGIC + 1e-9
    assert M.pair_magic(M.P_AT_MAX, math.pi / 4) == pytest.approx(math.log2(1.5))
    assert M.pair_magic(1 - M.P_AT_MAX, math.pi / 4) == pytest.approx(math.log2(1.5))


def test_antimatter_partner_carries_the_same_magic():
    for p, phi in [(0.2, 0.3), (0.6, 1.2), (0.8, 2.9)]:
        assert M.pair_magic(p, -phi) == pytest.approx(M.pair_magic(p, phi))


def test_reading_weights_planes_and_attributes_sources():
    pr = {a: 0.0 for a in M.AXES}
    pr["perception"] = M.P_AT_MAX
    r = M.reading(pr, {"perception": math.pi / 4}, weights=[1] * 7,
                  source_axes={"perceptopoly": "perception", "bakugo": "touch", "voice": "hearing"})
    assert r["planes"]["perception"]["magic"] == pytest.approx(math.log2(1.5), abs=1e-6)
    assert r["planes"]["perception"]["matter"] == r["planes"]["perception"]["antimatter"]
    assert r["total"] == pytest.approx(math.log2(1.5) / 7, abs=1e-6)
    assert r["by_source"] == {"perceptopoly": r["planes"]["perception"]["magic"], "bakugo": 0.0}
    assert 0.0 <= r["share_of_max"] <= 1.0


def test_bad_inputs():
    with pytest.raises(ValueError):
        M.pair_magic(float("nan"), 0.0)
    with pytest.raises(ValueError):
        M.stabilizer_renyi_2([])
    assert M.reading({"vision": float("inf")})["planes"]["vision"]["magic"] == 0.0


def test_one_positron_quantum_cyclotron_is_held_in_its_ground_state():
    c = M.quantum_cyclotron(5.36, 0.3)
    assert c["nu_c_hz"] == pytest.approx(1.5004e11, rel=1e-3)          # eB/(2 pi m_e)
    assert c["n_bar"] < 1e-10 and 1 - c["ground_probability"] < 1e-10
    warm = M.quantum_cyclotron(5.36, 20.0)
    assert warm["ground_probability"] < 0.5
    with pytest.raises(ValueError):
        M.quantum_cyclotron(0.0, 0.3)


def test_held_positron_carries_the_antimatter_half_and_cpt_holds_by_default():
    r = M.reading({"perception": 0.3}, {"perception": 0.8}, weights=[1] * 7)
    pl = r["planes"]["perception"]
    assert pl["antimatter"] == pytest.approx(pl["matter"]) and pl["cpt_asymmetry"] == 0.0
    assert r["cpt"] == {"ratio": 0.0, "max_asymmetry": 0.0, "conserved": True}
    assert "173001" in r["positron"]["source"]


def test_a_warm_trap_leaves_antimatter_magic_unheld():
    pl = M.reading({"perception": 0.3}, {"perception": 0.8}, weights=[1] * 7, temp_k=20.0)["planes"]["perception"]
    assert pl["antimatter"] + pl["unheld"] == pytest.approx(pl["matter"], abs=1e-6)
    assert pl["unheld"] > pl["antimatter"]


def test_a_cpt_violating_moment_ratio_shows_as_asymmetry(monkeypatch):
    monkeypatch.setenv("QUIPU_CPT_RATIO", "0.2")
    r = M.reading({"perception": 0.3}, {"perception": 0.8}, weights=[1] * 7)
    assert r["cpt"]["ratio"] == 0.2 and r["cpt"]["conserved"] is False
    assert r["planes"]["perception"]["cpt_asymmetry"] != 0.0
    assert r["planes"]["vision"]["cpt_asymmetry"] == 0.0               # no magic, nothing to break
    monkeypatch.setenv("QUIPU_CPT_RATIO", "nan")
    assert M.reading({"perception": 0.3}, {"perception": 0.8}, weights=[1] * 7)["cpt"]["conserved"] is True
