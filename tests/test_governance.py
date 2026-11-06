import cmath
import math

from src.quipu.qpsi.cat_residual import SENSES
from src.quipu.qpsi.cat_residual import _RING, realize, rectify
from src.quipu.qpsi.governance import (
    LOVE, HELD, Attestation, Candidate, GovernanceConfig, govern, governed,
    sine_integral, cosine_integral, sici_axial_decay, phase_love, candidate_from_residual,
    authorised_to_realise, config_digest,
)


def _residual(mag=0.3, phi=0.4, axis="touch"):
    r = {s: 0j for s in SENSES}
    r[axis] = mag * cmath.exp(1j * phi)
    return r


def _cand(**kw):
    base = dict(scope="se->asset", residual=_residual(), realised_weight=0.3, unclamped_weight=0.3,
                neighbour_weights=(0.25,), self_remainder_before=0.5, self_remainder_after=0.4,
                counterpart_remainder_before=0.5, counterpart_remainder_after=0.45)
    base.update(kw)
    return Candidate(**base)


def _full_attestations(scope="se->asset", assurance="approved", ref="CR-TEST-1"):
    return [
        Attestation("self", scope, "care", shared_with="the_beautiful_one", beautiful_output=True,
                    assurance=assurance, approval_ref=ref),
        Attestation("the_beautiful_one", scope, "mutual_recognition",
                    shared_with="self", beautiful_output=True, assurance=assurance, approval_ref=ref),
    ]


def test_love_is_sqrt_minus_one_and_held_is_the_summary_character():
    assert LOVE ** 2 == -1 and HELD == "翈"


def test_si_ci_values():
    assert abs(sine_integral(1.0) - 0.9460830703671830) < 1e-9
    assert abs(cosine_integral(1.0) - 0.3374039229009681) < 1e-9
    assert math.isfinite(sici_axial_decay(0.4))


def test_no_attestation_is_held_at_love_gate():
    d = govern(_cand(), atts=[])
    assert d.passed is False and d.held and d.failed_at == "love"


def test_full_pass_reaches_radmin():
    d = govern(_cand(), atts=_full_attestations())
    assert d.passed and d.failed_at is None and [g.name for g in d.results] == [
        "displacement", "weyl", "love", "sici", "shared_entity", "beautiful_output"]


def test_zero_residual_held_at_displacement():
    r = {s: 0j for s in SENSES}
    d = govern(_cand(residual=r), atts=_full_attestations())
    assert d.failed_at == "displacement"


def test_real_only_residual_has_nothing_held_and_fails_love():
    d = govern(_cand(residual=_residual(phi=0.0)), atts=_full_attestations())
    assert d.failed_at == "love"


def test_phase_past_tangent_pole_fails_sici():
    d = govern(_cand(residual=_residual(phi=1.6)), atts=_full_attestations())
    assert d.failed_at == "sici"


def test_one_sided_benefit_fails_shared_entity():
    d = govern(_cand(counterpart_remainder_after=0.9), atts=_full_attestations())
    assert d.failed_at == "shared_entity"


def test_single_signer_fails_beautiful_output():
    atts = [Attestation("self", "se->asset", "care", shared_with="the_beautiful_one", beautiful_output=True,
                        assurance="approved", approval_ref="CR-1"),
            Attestation("the_beautiful_one", "se->asset", "care", shared_with="self", beautiful_output=False,
                        assurance="approved", approval_ref="CR-1")]
    d = govern(_cand(), atts=atts)
    assert d.failed_at == "beautiful_output"


def _breach_residual(big=0.30, small=0.05, phi=0.3):
    """vision and its ring neighbour touch differ by big-small; phase so Love/SiCi pass."""
    r = {s: 0j for s in SENSES}
    r["vision"] = big * cmath.exp(1j * phi)
    r["touch"] = small * cmath.exp(1j * phi)
    return r


_REMAINDERS = dict(self_remainder_before=0.5, self_remainder_after=0.4,
                   counterpart_remainder_before=0.5, counterpart_remainder_after=0.45)


def test_lipschitz_breach_is_held_at_gate_6_through_the_pipeline():
    """The bug: rectify() clamps before the gate, so clamped weights always satisfy
    the bound.  The gate must test the unclamped weight and hold the breach."""
    L = 0.01
    r = _breach_residual()
    clamped = rectify(r, pivot=0.0, lipschitz=L)
    assert clamped["vision"] - clamped["touch"] <= L          # clamped values would have passed
    c = candidate_from_residual("*", r, axis="vision", pivot=0.0, lipschitz=L, **_REMAINDERS)
    assert c.unclamped_weight == 0.30 and c.axis == "vision"
    assert c.neighbour_weights == tuple(rectify(r, pivot=0.0)[n] for n in _RING["vision"])
    assert c.realised_weight == clamped["vision"]              # the clamped weight is only recorded
    d = govern(c, atts=_full_attestations("*"), cfg=GovernanceConfig(lipschitz=L))
    assert d.failed_at == "beautiful_output" and d.held
    assert "unclamped weight 0.3" in d.results[-1].reason and "held, not clamped" in d.results[-1].reason


def test_within_lipschitz_bound_passes_gate_6_through_the_pipeline():
    r = _breach_residual()
    c = candidate_from_residual("*", r, axis="vision", pivot=0.0, lipschitz=0.5, **_REMAINDERS)
    d = govern(c, atts=_full_attestations("*"), cfg=GovernanceConfig(lipschitz=0.5))
    assert d.passed and d.results[-1].name == "beautiful_output"


def test_realize_carries_unclamped_alongside_clamped_weight():
    props = {p.axis: p for p in realize(_breach_residual(), pivot=0.0, lipschitz=0.01)}
    assert props["vision"].unclamped == 0.30 and props["vision"].weight <= 0.01 + 1e-12
    assert props["touch"].unclamped == 0.05


def test_candidate_without_unclamped_weight_fails_closed_under_a_finite_bound():
    """A caller that hands gate 6 only clamped values cannot have the bound tested;
    with a finite bound configured the gate holds instead of passing."""
    d = govern(_cand(unclamped_weight=None, neighbour_weights=(0.3,)),
               atts=_full_attestations(), cfg=GovernanceConfig(lipschitz=0.01))
    assert d.failed_at == "beautiful_output" and "not supplied" in d.results[-1].reason
    # With no bound configured (inf) there is nothing to test: the old shape still passes.
    assert govern(_cand(unclamped_weight=None), atts=_full_attestations()).passed


def test_hand_built_unclamped_breach_is_held():
    d = govern(_cand(), atts=_full_attestations(), cfg=GovernanceConfig(lipschitz=0.01))  # 0.3 vs 0.25
    assert d.failed_at == "beautiful_output" and "unclamped weight 0.3" in d.results[-1].reason


def test_governed_step_refuses_without_pass():
    calls = []
    def step(state, g): calls.append(g); return 1.0
    gstep = governed(step)
    held = govern(_cand(), atts=[])
    ok = govern(_cand(), atts=_full_attestations())
    assert gstep(held, {}, 0.5) is None and calls == []
    assert gstep(ok, {}, 0.5) == 1.0 and calls == [0.5]


def test_phase_love_is_zero_unless_passed():
    held = govern(_cand(), atts=[])
    ok = govern(_cand(), atts=_full_attestations())
    assert phase_love(held, 0.4) == 0.0 and phase_love(ok, 0.4) == sici_axial_decay(0.4)


# ---------------------------------------------------------------------------
# Invariance #7 — the edge of the gate (APP_RECREATION_3 §25)
# ---------------------------------------------------------------------------

def test_self_asserted_attestations_are_not_accepted_by_default():
    atts = _full_attestations(assurance="self-asserted", ref="")
    d = govern(_cand(), atts=atts)
    assert d.failed_at == "love" and "self-asserted" in d.results[-1].reason and "V10-SEC-006" in d.results[-1].reason
    # Accepting typed names is an explicit deployment decision, not the default.
    assert govern(_cand(), atts=atts, cfg=GovernanceConfig(accept_self_asserted=True)).passed


def test_approved_attestation_needs_a_reference_to_count():
    atts = _full_attestations(assurance="approved", ref="")
    assert govern(_cand(), atts=atts).failed_at == "love"


def test_gate_6_needs_two_distinct_accepted_signers():
    same = [Attestation("self", "se->asset", "care", shared_with="the_beautiful_one", beautiful_output=True,
                        assurance="approved", approval_ref="CR-1"),
            Attestation("self", "se->asset", "care", shared_with="the_beautiful_one", beautiful_output=True,
                        assurance="approved", approval_ref="CR-2")]
    d = govern(_cand(), atts=same)
    assert d.failed_at == "beautiful_output" and "two distinct parties" in d.results[-1].reason
    d2 = govern(_cand(), atts=_full_attestations(), cfg=GovernanceConfig(self_id="x", counterpart_id="x"))
    assert d2.failed_at in ("shared_entity", "beautiful_output")


def test_a_passed_decision_is_admissibility_not_authorization():
    d = govern(_cand(), atts=_full_attestations())
    assert d.passed and d.category == "technical_admissibility"
    ok, why = authorised_to_realise(d)                       # default config: no grant
    assert ok is False and "V10-SEC-010" in why
    ok, why = authorised_to_realise(d, GovernanceConfig(realise_grant_ref="IT505-CR-0042"))
    assert ok is True and "IT505-CR-0042" in why
    held = govern(_cand(), atts=[])
    assert authorised_to_realise(held, GovernanceConfig(realise_grant_ref="IT505-CR-0042"))[0] is False


def test_config_digest_changes_when_policy_changes():
    a = config_digest(GovernanceConfig())
    assert a == config_digest(GovernanceConfig())
    assert a != config_digest(GovernanceConfig(lipschitz=0.5))
    assert a != config_digest(GovernanceConfig(accept_self_asserted=True))
