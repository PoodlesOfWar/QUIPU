"""The composed gate: agent signature -> grant -> essay stage -> allow.

This is the single place an action is authorized. The order follows the plan:

1. **request**  the agent's Ed25519 signature, its declaration that it is
                automated, freshness, and no replay;
2. **grant**    a live Warden grant naming this agent, this target and this
                action class;
3. **stage**    for game targets only, the Essay progression gate and the
                operator's second signature.

Failure modes, all closed
-------------------------
* Any exception inside a check becomes a denial (``gate:internal_error:<Type>``).
* If the decision cannot be written to the audit log, the action is denied
  (``gate:audit_unavailable``): an action nobody can later account for does not
  happen.
* A game target with no ``EssayGate`` configured is denied, not waved through.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from ._common import WardenError
from .audit import AuditLog
from .essay_gate import EssayGate
from .operator_attestation import OperatorAttestation
from .signed_agent import MAX_REQUEST_SKEW_S, AgentRegistry, NonceCache, SignedRequest, verify_request
from .warden_grant import GrantStore, WardenRegistry, parse_target, verify_grant


@dataclass(frozen=True)
class Decision:
    allow: bool
    layer: str
    reason: str
    agent_id: str | None = None
    target: str | None = None
    action: str | None = None
    grant_id: str | None = None
    stage: str | None = None

    @property
    def code(self) -> str:
        return f"{self.layer}:{self.reason}"


def _clip(value: Any, limit: int = 256) -> Any:
    return value[:limit] if isinstance(value, str) else value


class WardenGate:
    def __init__(
        self,
        *,
        grants: GrantStore,
        agents: AgentRegistry,
        wardens: WardenRegistry,
        audit: AuditLog,
        essay: EssayGate | None = None,
        nonces: NonceCache | None = None,
        clock: Callable[[], float] = time.time,
        secret: str | bytes | None = None,
        max_skew_s: int = MAX_REQUEST_SKEW_S,
    ) -> None:
        self._grants = grants
        self._agents = agents
        self._wardens = wardens
        self._audit = audit
        self._essay = essay
        self._nonces = nonces if nonces is not None else NonceCache()
        self._clock = clock
        self._secret = secret
        self._max_skew_s = max_skew_s

    @property
    def clock(self) -> Callable[[], float]:
        return self._clock

    @property
    def audit(self) -> AuditLog:
        return self._audit

    # ------------------------------------------------------------------ #

    def authorize(
        self,
        request: SignedRequest,
        *,
        attestation: OperatorAttestation | None = None,
    ) -> Decision:
        """Authorize one signed agent request. Always audited; never raises."""
        now = self._clock()
        try:
            decision = self._authorize(request, attestation, now)
        except Exception as exc:  # noqa: BLE001 - fail closed on anything unexpected
            decision = Decision(
                False, "gate", f"internal_error:{type(exc).__name__}",
                agent_id=_clip(getattr(request, "agent_id", None)),
                target=_clip(getattr(request, "target", None)),
                action=_clip(getattr(request, "action", None)),
                grant_id=_clip(getattr(request, "grant_id", None)),
            )
        extra: dict[str, Any] = {}
        if isinstance(request, SignedRequest):
            extra = {"nonce": _clip(request.nonce), "params_sha256": _safe_digest(request)}
        return self._record(decision, source="authorize", extra=extra)

    def check_grant(self, *, grant_id: str, agent_id: str, target: str, action: str) -> Decision:
        """Grant-only check for an already-authenticated agent. Audited, never raises.

        The embodiment uses this for the second grant (the robot's) behind a
        request whose signature the gate has already verified.
        """
        now = self._clock()
        try:
            decision = self._check_grant(grant_id, agent_id, target, action, now)
        except Exception as exc:  # noqa: BLE001
            decision = Decision(
                False, "gate", f"internal_error:{type(exc).__name__}",
                agent_id=_clip(agent_id), target=_clip(target), action=_clip(action), grant_id=_clip(grant_id),
            )
        return self._record(decision, source="check_grant")

    # ------------------------------------------------------------------ #

    def _authorize(
        self, request: SignedRequest, attestation: OperatorAttestation | None, now: float
    ) -> Decision:
        base = dict(
            agent_id=_clip(getattr(request, "agent_id", None)),
            target=_clip(getattr(request, "target", None)),
            action=_clip(getattr(request, "action", None)),
            grant_id=_clip(getattr(request, "grant_id", None)),
        )
        ok, reason = verify_request(
            request, agents=self._agents, nonces=self._nonces, now=now, max_skew_s=self._max_skew_s
        )
        if not ok:
            return Decision(False, "request", reason, **base)

        grant_decision = self._check_grant(request.grant_id, request.agent_id, request.target, request.action, now)
        if not grant_decision.allow:
            return grant_decision

        try:
            kind, _ = parse_target(request.target)
        except WardenError:
            return Decision(False, "grant", "target_mismatch", **base)
        if kind == "game":
            if self._essay is None:
                return Decision(False, "stage", "essay_gate_not_configured", **base)
            stage = self._essay.decide(
                target=request.target, agent_id=request.agent_id, now=now, attestation=attestation
            )
            if not stage.open:
                return Decision(False, "stage", stage.reason, stage=stage.stage, **base)
            return Decision(True, "stage", "ok", stage=stage.stage, **base)
        return Decision(True, "grant", "ok", **base)

    def _check_grant(self, grant_id: str, agent_id: str, target: str, action: str, now: float) -> Decision:
        base = dict(
            agent_id=_clip(agent_id), target=_clip(target), action=_clip(action), grant_id=_clip(grant_id)
        )
        grant = self._grants.get(grant_id)
        if grant is None:
            return Decision(False, "grant", "grant_not_found", **base)
        ok, reason = verify_grant(
            grant, agent_id=agent_id, target=target, action=action, now=now,
            store=self._grants, registry=self._wardens, secret=self._secret,
        )
        return Decision(ok, "grant", reason, **base)

    def _record(self, decision: Decision, *, source: str, extra: dict[str, Any] | None = None) -> Decision:
        details = {
            "source": source,
            "allow": decision.allow,
            "layer": decision.layer,
            "reason": decision.reason,
            "action": decision.action,
            "target": decision.target,
            "grant_id": decision.grant_id,
            "stage": decision.stage,
            **(extra or {}),
        }
        try:
            self._audit.append("decision", decision.agent_id or "unknown", details)
        except Exception:  # noqa: BLE001 - no audit record, no action
            return Decision(
                False, "gate", "audit_unavailable",
                agent_id=decision.agent_id, target=decision.target, action=decision.action,
                grant_id=decision.grant_id, stage=decision.stage,
            )
        return decision


def _safe_digest(request: SignedRequest) -> str | None:
    try:
        return request.params_digest()
    except Exception:  # noqa: BLE001 - e.g. non-JSON params; the digest is advisory
        return None
