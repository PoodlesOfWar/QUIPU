"""The held ACRE tool candidate viscous_shear_chain: forge compatibility, the stage
math, and parity with the live modules it mirrors."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from src.quipu import tool_forge as tf
from src.quipu.radam_optimizer import radam_step

SRC = Path(__file__).resolve().parents[1] / "src/quipu/qpsi/tool_candidates/viscous_shear_chain.py"


def _module():
    spec = importlib.util.spec_from_file_location("viscous_shear_chain_candidate", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _module()
tool = M.tool_viscous_shear_chain


def test_passes_forge_validation_and_classifies_as_a_tool():
    code = SRC.read_text()
    assert tf._validate_code(code) == (True, "ok")
    assert tf._classify_artifact({"label": "viscous shear chain", "degree": 1}, code)["type"] == "Tool"
    assert [n for n in dir(M) if n.startswith("tool_")] == ["tool_viscous_shear_chain"]


def test_empty_input_runs_every_stage_with_zero_confidence():
    out = tool({})
    assert set(out) == {"result", "explanation", "confidence"}
    assert out["confidence"] == 0.0
    for s in M.STAGES:
        assert s in out["result"]


def test_unknown_stage_is_reported_not_raised():
    out = tool({"stage": "nonsense"})
    assert out["confidence"] == 0.0 and "unknown stage" in out["explanation"]


def test_output_is_deterministic_and_does_not_mutate_input():
    data = {"entropy_series": [0.1, 0.5, 0.2, 0.3, 0.9, 0.4, 0.41, 0.2], "now": 100.0, "broaden": 1,
            "actionable_potential": 1, "npcs": [{"magnitude": 0.7, "direction": 1.0, "t": 90.0}],
            "occupied": {"0,0": "a", "1,1": "b"}, "edges": [["a", "b", 0.5]]}
    before = copy.deepcopy(data)
    a, b = tool(data), tool(data)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    assert data == before


def test_decay_limits_match_the_muonic_and_textile_regimes():
    assert M._decay(1.0, 0.35, 0.0, 2.0) == pytest.approx(math.exp(-0.7))
    assert M._decay(2.0, 0.0, 0.5, 3.0) == pytest.approx(2.0 / 4.0)
    d = tool({"stage": "decay", "gamma": 0.0, "coherence": 1.0})["result"]["decay"]
    assert d["textile_fraction"] == 0.0 and d["half_life_ratio"] == pytest.approx(1.0, abs=1e-5)
    assert 1.0e7 < d["muonic_rate_s"] < 1.1e7


def test_proximity_equals_the_legacy_kernel_at_zero_drag():
    for d in (0, 8, 32, 64):
        assert M._proximity(d, 64, 0.0) == pytest.approx(1.0 - d / 64.0)
    assert M._proximity(32, 64, 1.0) == pytest.approx(0.75)


def test_knot_cells_close_and_the_mirror_reflects_one_axis():
    right, left = M._knot_cells(64, 2, 3), M._knot_cells(64, 2, -3)
    assert right[0] == (0, 0) and len(right) == len(left) > 64
    assert set(left) == {(i, (-j) % 64) for i, j in right}


def test_field_peaks_respect_the_gard_floor_and_monotone_series_gives_none():
    rising = tool({"stage": "field", "entropy_series": [0.1, 0.2, 0.3, 0.4, 0.5]})["result"]["field"]
    assert rising["peaks"] == [] and rising["cells"] == []
    short = tool({"stage": "field", "entropy_series": [0.1, 0.2]})["result"]["field"]
    assert "fewer than 3" in short["reason"]
    f = tool({"stage": "field", "entropy_series": [0.1, 0.5, 0.2, 0.3, 0.9, 0.4, 0.41, 0.2],
              "delta_star": 0.05})["result"]["field"]
    assert [p["idx"] for p in f["peaks"]] == [1, 4]
    assert all(0 <= c["amp"] <= 1 and -1 <= c["chi"] <= 1 for c in f["cells"])


def test_delta_star_matches_the_gard_crb_floor():
    from src.quipu.gard_info import cramer_rao_bound_gard
    series = [0.1, 0.5, 0.2, 0.3, 0.9, 0.4, 0.41, 0.2]
    mean = sum(series) / len(series)
    sigma = math.sqrt(sum((x - mean) ** 2 for x in series) / len(series))
    got = tool({"stage": "field", "entropy_series": series})["result"]["field"]["delta_star"]
    assert got == pytest.approx(math.sqrt(12 * cramer_rao_bound_gard(sigma, len(series))), abs=1e-6)


def test_plane_weights_match_rheology():
    from src.quipu.qpsi import rheology
    assert list(M.PLANE_WEIGHTS) == pytest.approx(rheology.axis_weights(), abs=1e-6)


def test_gate_matches_radam_step_with_fresh_state():
    g = tool({"stage": "gate", "now": 1000.0, "broaden": 1, "actionable_potential": 0.9,
              "npcs": [{"magnitude": 0.7, "direction": 0.3, "t": 990.0}]})["result"]["gate"]
    p = radam_step({}, 0.6, g["g_im"], external_phase=g["phase"], use_torus=True)
    assert g["pressure"] == pytest.approx(p, abs=1e-6)
    assert g["value"] == pytest.approx(min(1.0, p * 0.9), abs=1e-6)


def test_gate_is_zero_without_a_fresh_npc_and_decays_with_age():
    no = tool({"stage": "gate", "now": 1000.0, "broaden": 1, "liveness": 1, "actionable_potential": 1})
    assert no["result"]["gate"]["value"] == 0.0
    expired = tool({"stage": "gate", "now": 10_000.0, "broaden": 1, "actionable_potential": 1,
                    "npcs": [{"magnitude": 1, "t": 0.0}]})["result"]["gate"]
    assert expired["fresh"] == 0 and expired["ignored"] == 1
    young = tool({"stage": "gate", "now": 100.0, "npcs": [{"magnitude": 1, "t": 100.0}]})["result"]["gate"]
    old = tool({"stage": "gate", "now": 3000.0, "npcs": [{"magnitude": 1, "t": 100.0}]})["result"]["gate"]
    assert young["g_im"] > old["g_im"] > 0


def test_gate_skips_malformed_npcs_and_is_call_count_independent():
    data = {"stage": "gate", "now": 10.0, "broaden": 1, "actionable_potential": 1,
            "npcs": [{"magnitude": "x", "t": 5}, "junk", {"magnitude": 0.5, "t": 5}]}
    a, b = tool(data)["result"]["gate"], tool(data)["result"]["gate"]
    assert a == b and a["fresh"] == 1 and a["ignored"] == 2


def test_bond_chain_matches_bond_shear_evaluate():
    from src.quipu.qpsi import bond_shear as bs
    b = tool({"stage": "bond", "bond": {"r": 0.5}})["result"]["bond"]
    e = bs.evaluate(None, cfg=bs.BondShearConfig())
    assert b["rupture_nn"]["litasb"] == pytest.approx(e["rupture"]["litasb_nn"], abs=1e-6)
    assert b["survival"] == pytest.approx(e["condensate"]["survival"], abs=1e-6)
    assert b["yield"] == pytest.approx(e["yield"], abs=1e-6)
    assert b["estimate"] is True


def test_interactions_are_bounded_gated_and_limited_to_existing_edges():
    base = {"entropy_series": [0.1, 0.5, 0.2, 0.3, 0.9, 0.4, 0.41, 0.2], "now": 10.0, "broaden": 1,
            "actionable_potential": 1, "occupied": {"0,0": "a", "1,1": "b", "40,40": "c"},
            "edges": [["a", "b", 0.5], ["a", "c", 0.5]], "stage": "interactions"}
    closed = tool(base)["result"]["interactions"]
    assert closed["tokens"] == [] and closed["edges"] == []
    open_ = tool({**base, "npcs": [{"magnitude": 1, "t": 10.0}]})["result"]["interactions"]
    assert open_["tokens"] and all(0 < t["lift"] <= M.TOKEN_LIFT_MAX for t in open_["tokens"])
    assert [(e["src"], e["dst"]) for e in open_["edges"]] == [("a", "b")]
    assert all(0 < e["lift"] <= M.EDGE_LIFT_MAX for e in open_["edges"])


def test_worklink_hand_values_on_both_time_bases():
    utc = tool({"stage": "worklink", "research_hours": [13, 0.5]})["result"]["worklink"]
    assert utc["ratio"] == 26.0 and utc["ratio_bits"] == pytest.approx(4.7004, abs=1e-4)
    assert utc["gap_bits"] == pytest.approx(109.9155, abs=1e-3) and utc["fits_interstitial"] is True
    assert utc["v_knot_m_s"] == pytest.approx(1852 / 3600, abs=1e-6)
    planck = tool({"stage": "worklink", "time_base": "planck_time"})["result"]["worklink"]
    assert planck["gap_bits"] == pytest.approx(-33.8187, abs=1e-3) and planck["fits_interstitial"] is False


def test_worklink_overhand_capstan_and_work_bound():
    from src.quipu.qpsi.hidden_work import markov_energetic_bound
    geo = tool({"stage": "worklink"})["result"]["worklink"]["overhand"]
    tight = tool({"stage": "worklink", "theta": "tight"})["result"]["worklink"]["overhand"]
    assert geo["theta_rad"] == pytest.approx(3 * math.pi, abs=1e-6)
    assert tight["capstan_hold"] == pytest.approx(math.exp(0.2 * 4 * math.pi), abs=1e-5)
    w = tool({"stage": "worklink", "kappa_uN_per_m": 20.0, "mean_x": 1e-6, "windings": 2, "tau": 4.0})
    work = w["result"]["worklink"]["work"]
    assert work["bound_j"] == pytest.approx(markov_energetic_bound(20e-6, 1e-6))
    assert work["power_w"] == pytest.approx(work["bound_j"] * 2 / 4.0)


def test_pompeii_matches_the_repo_case_and_is_unresolved():
    from src.quipu.qpsi.vnn_bathymetry import build_pompeii_79ce_case
    case = build_pompeii_79ce_case()
    p = tool({"stage": "pompeii"})["result"]["pompeii"]
    assert M.POMPEII["burial_depth_m"] == pytest.approx(case["burial_depth_m"], abs=1e-3)
    td = case["temporal_depression"]
    assert M.POMPEII["temporal_depression_m"] == pytest.approx(td["temporal_depression_m"], abs=1e-3)
    assert M.POMPEII["uncertainty_m"] == pytest.approx(td["uncertainty_m"], abs=1e-2)
    assert p["accumulated_aggradation_m"] == pytest.approx(0.0022 * 1947, abs=1e-6)
    assert p["depression_z"] < 1.0 and p["resolved"] is False
    assert p["aggradation_gap_bits"] == pytest.approx(math.log2((1852 / 3600) / (0.0022 / 31557600)), abs=1e-4)


def test_confidence_scales_with_real_inputs():
    one = tool({"stage": "worklink", "research_hours": [13, 0.5]})["confidence"]
    bond_default = tool({"stage": "bond"})["confidence"]
    assert one == 0.9 and bond_default == 0.0
