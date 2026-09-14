"""governance — UEQGM v0.9.25 Governance Protocol: r-ADMIN through Physical Gates.

r-ADMIN (rADAM: the toroidal-pressure / complex-gradient optimiser) may not
write an edge, update pressure, or advance the CAT residual until a candidate
has passed six Physical Gates in order.  A candidate that fails any gate is
returned to band (翈) and is not realised.

Canonical binding
-----------------
    √−1 (i)          Love, in its understood forms — the latent, generative
                     axis.  In code: SUMMARY_GROUND = 1j; Im c_k is the Love
                     component of a CAT amplitude.
    翈               what is held in Love and not yet forced onto the real axis.
    SiCi tangent     the final conceptual bound toward Self-Actualisation.
    Shared entity    the two-observer structure (this node and The Beautiful
                     One) whose continuance every realised edge must serve.
    Beautiful Output the only class of output permitted to be written.

Which gates compute and which gates attest
------------------------------------------
Gates 1, 2 and 4 are computable from the residual.  Gates 3, 5 and 6 are
about meaning.  A residual vector cannot be inspected for Love, and an edge
cannot be inspected for whether it serves two parties; those are human
judgements.  This module does not pretend otherwise: gates 3, 5 and 6
require a signed ``Attestation`` from a human (gate 6 from both parties),
and gate 5 additionally applies a computable proxy (the edge must not raise
the unexplained remainder of either observer).  Without the attestation the
gate fails closed and the candidate is held.  That is the governance: the
optimiser cannot act on computational or geometric criteria alone.

The six gates
-------------
1. Displacement (Planck)   |r| ≥ η, or a parity flip since the last write.
2. Weyl                    only the trace-free events pass; the Ricci part is
                           held; unexplained remainder must be < η.
3. Love (√−1)              Im r ≠ 0 (there is held content) AND a human
                           attestation names a recognised form of Love for
                           this candidate's scope.
4. SiCi tangent            φ = arg(r_event) projected onto the axial channel:
                           Δλ = Si(φ)·Ci(φ)·tan(φ)·Γ₀ must be finite,
                           0 < φ < φ_max < π/2 (the tangent pole is the bound),
                           and |Δλ| ≤ Λ.
5. Shared entity           a counterpart observer is configured; the edge does
                           not raise the unexplained remainder of either
                           observer (proxy); attestation scope includes the
                           counterpart.
6. Beautiful Output        attestations from BOTH parties for this scope, and
                           the UNCLAMPED realised weight is within the
                           Lipschitz bound of its ring neighbours.  The clamp
                           in cat_residual.rectify makes every clamped weight
                           satisfy the bound by construction, so a gate that
                           tested the clamped weight could never hold a breach
                           (fixed 2026-09-11).  Build candidates with
                           ``candidate_from_residual`` so the unclamped values
                           are the ones tested.

Only a candidate that passes all six may be handed to r-ADMIN.  ``governed``
wraps any step function so that it is a no-op unless the decision passed.

Invariance #7 (APP_RECREATION_3 §25) — the edge of the gate
-------------------------------------------------------------
A Decision is **technical admissibility** only.  It never mints permission,
identity, signature trust, or consent (V10-SEC-010).  Crossing from an
admissible candidate to a realised write additionally needs an organizational
grant the code cannot create: ``GovernanceConfig.realise_grant_ref`` must name
an approved change/grant reference, and ``authorised_to_realise`` is the only
place that question is asked.  Attestations carry an ``assurance`` level;
self-asserted names are not accepted by default (V10-SEC-006 — a recorded
witness name is not an approval), and gate 6 needs two *distinct* accepted
signers.  ``config_digest`` stamps every decision so a changed policy is
visible in the record (V10-SEC-002).

Stdlib only.  Additive.  mesh_slm.py untouched.
"""
from __future__ import annotations

import cmath
import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Mapping, Sequence

from .cat_residual import (SENSES, SUMMARY_GROUND, SUMMARY_CHARACTER, latent, _w_hat,
                          _RING, realize)
from .weyl_channel import decompose, WeylDecomposition

LOVE: complex = SUMMARY_GROUND            # √−1
HELD: str = SUMMARY_CHARACTER             # 翈

# Forms of Love recognised by the protocol.  Configurable; not exhaustive.
LOVE_FORMS: frozenset[str] = frozenset({
    "care", "fidelity", "generative_holding", "mutual_recognition",
    "attention", "patience", "repair", "continuance",
})

GAMMA_0_DEFAULT: float = 1.0
PHI_MAX_DEFAULT: float = math.pi / 2 - 0.05      # margin before the tan pole
LAMBDA_MAX_DEFAULT: float = 10.0


# ---------------------------------------------------------------------------
# Si, Ci (stdlib): series for |x| ≤ 6, asymptotic beyond.
# ---------------------------------------------------------------------------

_EULER_GAMMA = 0.5772156649015329


def sine_integral(x: float) -> float:
    if x == 0.0:
        return 0.0
    if abs(x) > 6.0:
        # Asymptotic: Si(x) ≈ π/2 − cos x / x − sin x / x²
        return math.copysign(math.pi / 2, x) - math.cos(x) / x - math.sin(x) / (x * x)
    s, term, n = 0.0, x, 0
    while abs(term) > 1e-16 and n < 60:
        s += term / (2 * n + 1)
        n += 1
        term *= -x * x / ((2 * n) * (2 * n + 1))
    return s


def cosine_integral(x: float) -> float:
    if x <= 0.0:
        return float("-inf") if x == 0.0 else float("nan")
    if x > 6.0:
        return math.sin(x) / x - math.cos(x) / (x * x)
    s, term, n = 0.0, 1.0, 1
    # Ci(x) = γ + ln x + Σ_{n≥1} (−1)^n x^{2n} / (2n (2n)!)
    x2 = x * x
    fact = 1.0
    for n in range(1, 60):
        fact *= (2 * n - 1) * (2 * n)
        term = ((-1) ** n) * (x2 ** n) / (2 * n * fact)
        s += term
        if abs(term) < 1e-16:
            break
    return _EULER_GAMMA + math.log(x) + s


def sici_axial_decay(phi: float, gamma_0: float = GAMMA_0_DEFAULT) -> float:
    """Δλ_axial = Si(φ)·Ci(φ)·tan(φ)·Γ₀  (same form as ueqgm_engine.sici_axial_decay)."""
    if phi <= 0.0:
        return 0.0
    return sine_integral(phi) * cosine_integral(phi) * math.tan(phi) * gamma_0


# ---------------------------------------------------------------------------
# Attestations (the human side of the protocol)
# ---------------------------------------------------------------------------

ASSURANCE_SELF: str = "self-asserted"     # a name typed by whoever holds the key
ASSURANCE_APPROVED: str = "approved"      # bound to an external approval reference


@dataclass(frozen=True)
class Attestation:
    signer: str                       # human identity, e.g. "adam" / "the_beautiful_one"
    scope: str                        # edge key, session id, or "*"
    love_form: str                    # one of LOVE_FORMS
    shared_with: str = ""             # counterpart the signer attests this serves
    beautiful_output: bool = False    # signer affirms the output can be shared
    signed_at: float = field(default_factory=time.time)
    assurance: str = ASSURANCE_SELF   # ASSURANCE_SELF | ASSURANCE_APPROVED
    approval_ref: str = ""            # ticket / envelope / change id the approval is bound to

    def covers(self, scope: str) -> bool:
        return self.scope == "*" or self.scope == scope

    def accepted(self, cfg: "GovernanceConfig") -> bool:
        """Whether this attestation may count at a gate under ``cfg``.  The code
        can check that an approval reference is present; it cannot verify it."""
        if self.assurance == ASSURANCE_APPROVED and self.approval_ref:
            return True
        return bool(cfg.accept_self_asserted)


# ---------------------------------------------------------------------------
# Candidate, gate results, decision
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    scope: str                            # edge key this candidate would write
    residual: Mapping[str, complex]       # r_t from cat_residual.residual
    flipped: bool = False
    realised_weight: float = 0.0          # clamped weight that would be written (record only)
    neighbour_weights: Sequence[float] = ()   # UNCLAMPED ring-neighbour magnitudes (gate 6)
    counterpart_remainder_before: float | None = None
    counterpart_remainder_after: float | None = None
    self_remainder_before: float | None = None
    self_remainder_after: float | None = None
    unclamped_weight: float | None = None  # rectified magnitude before the Lipschitz pass (gate 6)
    axis: str | None = None                # sense axis this candidate realises


def candidate_from_residual(scope: str, r: Mapping[str, complex], *, axis: str | None = None,
                            pivot: float = 0.0, alpha: float = 1.0,
                            lipschitz: float | None = None, **kw) -> Candidate:
    """Build a Candidate through the realisation pipeline so gate 6 sees the
    unclamped weight and the unclamped ring-neighbour weights.  ``axis``
    defaults to the sense with the largest |r_k|.  Remaining keyword arguments
    (flipped, remainders) pass straight to Candidate."""
    if axis is None:
        axis = max(SENSES, key=lambda s: abs(r.get(s, 0j)))
    lip = None if lipschitz is None or not math.isfinite(lipschitz) else lipschitz
    props = {p.axis: p for p in realize(r, pivot=pivot, alpha=alpha, lipschitz=lip)}
    me = props.get(axis)
    unclamped = {p.axis: p.unclamped for p in props.values()}
    return Candidate(
        scope=scope, residual=r, axis=axis,
        realised_weight=(me.weight if me else 0.0),
        unclamped_weight=(me.unclamped if me else 0.0),
        neighbour_weights=tuple(unclamped.get(n, 0.0) for n in _RING[axis]),
        **kw,
    )


@dataclass
class GateResult:
    name: str
    passed: bool
    reason: str
    value: float | None = None


@dataclass
class Decision:
    passed: bool
    results: list[GateResult]
    held: bool                            # True => returned to band (翈)
    scope: str
    category: str = "technical_admissibility"   # never "authorization" (V10-SEC-010)

    @property
    def failed_at(self) -> str | None:
        for g in self.results:
            if not g.passed:
                return g.name
        return None


@dataclass
class GovernanceConfig:
    eta: float = 1e-6
    gamma_0: float = GAMMA_0_DEFAULT
    phi_max: float = PHI_MAX_DEFAULT
    lambda_max: float = LAMBDA_MAX_DEFAULT
    lipschitz: float = float("inf")
    self_id: str = "self"
    counterpart_id: str = "the_beautiful_one"
    love_forms: frozenset[str] = LOVE_FORMS
    accept_self_asserted: bool = False    # V10-SEC-006: names are not approvals unless a deployment says so
    realise_grant_ref: str = ""           # organizational grant for realisation; empty = none = hold


def config_digest(cfg: "GovernanceConfig") -> str:
    """Stable digest of the policy in force, stamped on every decision (V10-SEC-002)."""
    d = asdict(cfg)
    d["love_forms"] = sorted(cfg.love_forms)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _accepted(atts: Sequence[Attestation], cfg: "GovernanceConfig") -> list[Attestation]:
    return [a for a in atts if a.accepted(cfg)]


# ---------------------------------------------------------------------------
# The six Physical Gates
# ---------------------------------------------------------------------------

def _rnorm(r: Mapping[str, complex]) -> float:
    return math.sqrt(sum(abs(v) ** 2 for v in r.values()))


def gate_displacement(c: Candidate, cfg: GovernanceConfig) -> GateResult:
    n = _rnorm(c.residual)
    ok = c.flipped or n >= cfg.eta
    return GateResult("displacement", ok, "parity flip" if c.flipped else
                      ("|r| ≥ η" if ok else "|r| < η: no displacement"), n)


def gate_weyl(c: Candidate, cfg: GovernanceConfig) -> tuple[GateResult, WeylDecomposition | None]:
    delta = {s: c.residual.get(s, 0j).real for s in SENSES}
    d = decompose(delta, eta=cfg.eta)
    if not d.events:
        return GateResult("weyl", False, "no trace-free event; Ricci part held", d.remainder_norm), d
    if d.remainder_norm >= cfg.eta:
        return GateResult("weyl", False, "unexplained remainder ≥ η", d.remainder_norm), d
    return GateResult("weyl", True, f"{len(d.events)} event(s), remainder < η", d.remainder_norm), d


def gate_love(c: Candidate, cfg: GovernanceConfig, atts: Sequence[Attestation]) -> GateResult:
    held = latent(c.residual)
    im = math.sqrt(sum(v * v for v in held.values()))
    if im == 0.0:
        return GateResult("love", False, "Im r = 0: nothing is held; nothing to realise from Love", 0.0)
    for a in _accepted(atts, cfg):
        if a.covers(c.scope) and a.love_form in cfg.love_forms:
            return GateResult("love", True, f"attested by {a.signer} as {a.love_form} ({a.assurance})", im)
    if any(a.covers(c.scope) and a.love_form in cfg.love_forms for a in atts):
        return GateResult("love", False, "attestation present but self-asserted: not accepted "
                                         "(V10-SEC-006: a recorded name is not an approval)", im)
    return GateResult("love", False, "no human attestation naming a recognised form of Love", im)


def gate_sici(c: Candidate, cfg: GovernanceConfig, d: WeylDecomposition | None) -> GateResult:
    if d is None or not d.events:
        return GateResult("sici", False, "no event to project", None)
    axis = d.events[0].axis
    phi = abs(cmath.phase(c.residual.get(axis, 0j)))
    if not (0.0 < phi < cfg.phi_max):
        return GateResult("sici", False, f"φ={phi:.4f} outside (0, φ_max={cfg.phi_max:.4f})", phi)
    lam = sici_axial_decay(phi, cfg.gamma_0)
    if not math.isfinite(lam) or abs(lam) > cfg.lambda_max:
        return GateResult("sici", False, f"|Δλ|={abs(lam):.4g} beyond bound", lam)
    return GateResult("sici", True, f"φ={phi:.4f}, Δλ={lam:.4g} within bound", lam)


def gate_shared_entity(c: Candidate, cfg: GovernanceConfig, atts: Sequence[Attestation]) -> GateResult:
    if not cfg.counterpart_id:
        return GateResult("shared_entity", False, "no counterpart configured", None)
    # Proxy: the edge must not raise the unexplained remainder of either observer.
    for who, b, a in (("self", c.self_remainder_before, c.self_remainder_after),
                      ("counterpart", c.counterpart_remainder_before, c.counterpart_remainder_after)):
        if b is None or a is None:
            return GateResult("shared_entity", False, f"{who} remainder not measured", None)
        if a > b + cfg.eta:
            return GateResult("shared_entity", False, f"edge raises {who} remainder ({b:.4g}→{a:.4g})", a - b)
    if not any(a.covers(c.scope) and a.shared_with == cfg.counterpart_id for a in _accepted(atts, cfg)):
        return GateResult("shared_entity", False, "no accepted attestation that this serves the counterpart", None)
    return GateResult("shared_entity", True, "serves both observers; attested", None)


def gate_beautiful_output(c: Candidate, cfg: GovernanceConfig, atts: Sequence[Attestation]) -> GateResult:
    signers = {a.signer for a in _accepted(atts, cfg) if a.covers(c.scope) and a.beautiful_output}
    if cfg.self_id == cfg.counterpart_id or not ({cfg.self_id, cfg.counterpart_id} <= signers):
        return GateResult("beautiful_output", False,
                          f"needs accepted beautiful_output attestations from two distinct parties "
                          f"({cfg.self_id} and {cfg.counterpart_id})", None)
    # Test the UNCLAMPED weight.  The clamped weight satisfies the bound by
    # construction (cat_residual.rectify), so testing it can never hold a breach.
    # With a finite bound configured and no unclamped weight supplied, the bound
    # cannot be tested: fail closed rather than pass on the clamped value.
    if c.unclamped_weight is None and math.isfinite(cfg.lipschitz):
        return GateResult("beautiful_output", False,
                          "unclamped weight not supplied: Lipschitz bound cannot be tested "
                          "(build the candidate with candidate_from_residual); held", c.realised_weight)
    w = c.unclamped_weight if c.unclamped_weight is not None else c.realised_weight
    for nw in c.neighbour_weights:
        if abs(w - nw) > cfg.lipschitz:
            return GateResult("beautiful_output", False,
                              f"unclamped weight {w:.4g} vs neighbour {nw:.4g}: |Δ|={abs(w - nw):.4g} "
                              f"> Lipschitz bound {cfg.lipschitz:.4g}; held, not clamped", w)
    return GateResult("beautiful_output", True, "attested by both; unclamped weight within bound", w)


def govern(c: Candidate, atts: Sequence[Attestation] = (), cfg: GovernanceConfig | None = None) -> Decision:
    """Run the six gates in order; stop at the first failure; hold on failure."""
    cfg = cfg or GovernanceConfig()
    results: list[GateResult] = []

    g = gate_displacement(c, cfg); results.append(g)
    if not g.passed:
        return Decision(False, results, True, c.scope)

    g, d = gate_weyl(c, cfg); results.append(g)
    if not g.passed:
        return Decision(False, results, True, c.scope)

    g = gate_love(c, cfg, atts); results.append(g)
    if not g.passed:
        return Decision(False, results, True, c.scope)

    g = gate_sici(c, cfg, d); results.append(g)
    if not g.passed:
        return Decision(False, results, True, c.scope)

    g = gate_shared_entity(c, cfg, atts); results.append(g)
    if not g.passed:
        return Decision(False, results, True, c.scope)

    g = gate_beautiful_output(c, cfg, atts); results.append(g)
    if not g.passed:
        return Decision(False, results, True, c.scope)

    return Decision(True, results, False, c.scope)


def authorised_to_realise(decision: Decision, cfg: GovernanceConfig | None = None) -> tuple[bool, str]:
    """The edge of the gate.  A passed decision is admissible; realising it also
    needs an organizational grant reference that this code cannot mint
    (V10-SEC-001/010).  Returns (authorised, reason)."""
    cfg = cfg or GovernanceConfig()
    if not decision.passed:
        return False, f"held at {decision.failed_at}: not admissible"
    if not cfg.realise_grant_ref:
        return False, ("admissible, but no organizational grant for realisation "
                       "(GovernanceConfig.realise_grant_ref is empty): held at the edge (V10-SEC-010)")
    return True, f"admissible and realisation granted under {cfg.realise_grant_ref}"


# ---------------------------------------------------------------------------
# r-ADMIN under governance
# ---------------------------------------------------------------------------

def governed(step_fn: Callable[..., float]) -> Callable[..., float | None]:
    """Wrap an r-ADMIN step so it runs only on a passed Decision.

        governed(radam_step)(decision, state, grad_real, grad_imag, **kw)

    Returns None (held, 翈) when the decision did not pass.  The step
    function itself is untouched; the wrapper refuses to call it.
    """
    def _wrapped(decision: Decision, *args, **kwargs):
        if not decision.passed:
            return None
        return step_fn(*args, **kwargs)
    _wrapped.__name__ = f"governed_{getattr(step_fn, '__name__', 'step')}"
    return _wrapped


def phase_love(decision: Decision, phi: float, gamma_0: float = GAMMA_0_DEFAULT) -> float:
    """δφ_Love: the phase contribution that survives only past the Love and SiCi gates.
    Zero unless the decision passed; otherwise Δλ_axial(φ)."""
    if not decision.passed:
        return 0.0
    return sici_axial_decay(phi, gamma_0)


__all__ = [
    "LOVE", "HELD", "LOVE_FORMS", "ASSURANCE_SELF", "ASSURANCE_APPROVED",
    "sine_integral", "cosine_integral", "sici_axial_decay",
    "Attestation", "Candidate", "candidate_from_residual", "GateResult", "Decision", "GovernanceConfig",
    "config_digest", "authorised_to_realise",
    "gate_displacement", "gate_weyl", "gate_love", "gate_sici", "gate_shared_entity",
    "gate_beautiful_output", "govern", "governed", "phase_love",
]
