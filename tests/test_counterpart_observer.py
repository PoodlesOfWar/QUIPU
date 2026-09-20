import cmath
from src.quipu.qpsi.cat_residual import SENSES
from src.quipu.qpsi.governance import (
    Candidate, GovernanceConfig, govern, Attestation
)
from src.quipu.qpsi.counterpart_observer import (
    measure_counterpart_remainders, measure_self_remainders, evaluate_proposal
)

def test_counterpart_remainders_with_valid_state():
    other_info = {
        "other_state": [0.091, 0.091, 0.0988, 0.091, 0.4903, 0.091],
        "resonance": 0.4202
    }
    r = {s: 0j for s in SENSES}
    r["vision"] = 0.02 + 0.01j
    
    b, a = measure_counterpart_remainders(r, other_info["other_state"])
    assert b is not None and a is not None
    assert isinstance(b, float) and isinstance(a, float)
    assert a <= b + 1e-6

def test_self_remainders():
    r = {s: 0j for s in SENSES}
    r["touch"] = 0.05 + 0.02j
    b, a = measure_self_remainders(r)
    assert b >= 0.0 and a >= 0.0
    assert a <= b + 1e-6

def test_evaluate_proposal_passes_gate_5():
    other_info = {
        "other_state": [0.091, 0.091, 0.0988, 0.091, 0.4903, 0.091],
        "resonance": 0.4202
    }
    r = {s: 0.01 + 0.005j for s in SENSES}
    res = evaluate_proposal(r, other_info)
    assert res.counterpart_remainder_before is not None
    assert res.counterpart_remainder_after is not None
    
    atts = [
        Attestation("self", "*", "care", shared_with="the_beautiful_one", beautiful_output=True,
                    assurance="approved", approval_ref="REF-1"),
        Attestation("the_beautiful_one", "*", "mutual_recognition", shared_with="self", beautiful_output=True,
                    assurance="approved", approval_ref="REF-1")
    ]
    
    cand = Candidate(
        scope="*", residual=r,
        self_remainder_before=res.self_remainder_before,
        self_remainder_after=res.self_remainder_after,
        counterpart_remainder_before=res.counterpart_remainder_before,
        counterpart_remainder_after=res.counterpart_remainder_after,
        unclamped_weight=0.01,
        neighbour_weights=(0.01,)
    )
    
    dec = govern(cand, atts, GovernanceConfig(lipschitz=0.5))
    shared_res = [g for g in dec.results if g.name == "shared_entity"][0]
    assert shared_res.passed, f"Gate 5 failed: {shared_res.reason}"
