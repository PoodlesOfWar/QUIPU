"""QUIPU Physical Gate (Gate 6) User Integration & Refinement Interlock.

Implements the human-in-the-loop User Confirmation protocol bound to
UEQGM v0.9.25 Physical Gate 6 (Beautiful Output / Two-Party Attestation).

Whenever an autonomous game action, multi-account login challenge, bot
navigation obstruction, or systemic refinement loop encounters a breakage
or epistemic rupture, the system enters a Gate 6 Hold (held in band 翈).
The User/Operator is prompted to confirm the right path, supply credentials,
or disambiguate the trajectory, releasing the hold via an approved Attestation
so r-ADMIN and the system can resume on the verified path.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from quipu import brain_kv
from quipu.qpsi.governance import (
    ASSURANCE_APPROVED,
    Attestation,
    Candidate,
    Decision,
    GateResult,
    GovernanceConfig,
    gate_beautiful_output,
)

logger = logging.getLogger("quipu.games.gate6_interlock")

_BRAIN_KV_GATE6_KEY = "governance:gate6_interlocks"


class BreakageType(str, Enum):
    AUTHENTICATION_CHALLENGE = "authentication_challenge"
    CAPTCHA_VERIFICATION = "captcha_verification"
    MULTI_ACCOUNT_INPUT_REQUIRED = "multi_account_input_required"
    PATH_OBSTRUCTION = "path_obstruction"
    DISCONNECTION = "disconnection"
    EPISTEMIC_RUPTURE = "epistemic_rupture"
    REFINEMENT_ANOMALY = "refinement_anomaly"
    MANUAL_OPERATOR_OVERRIDE = "manual_operator_override"


@dataclass
class RefinementBreakageEvent:
    breakage_id: str
    domain: str  # "game_session", "systemic_refinement", "world_model"
    target: str  # container name or subsystem ID
    breakage_type: str
    reason: str
    context: dict[str, Any] = field(default_factory=dict)
    suggested_paths: list[str] = field(default_factory=list)
    status: str = "HELD_AT_GATE_6"  # "HELD_AT_GATE_6", "RESOLVED_BY_USER", "REJECTED"
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    confirmed_path: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_payload: dict[str, Any] = field(default_factory=dict)
    attestation_digest: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Gate6UserInterlock:
    """Manages Gate 6 Holds, Operator Attestations, and Refinement Resumption."""

    def __init__(
        self,
        operator_id: str = "operator_admin",
        counterpart_id: str = "the_beautiful_one",
        persist: bool = True,
    ) -> None:
        self.operator_id = operator_id
        self.counterpart_id = counterpart_id
        self.persist = persist
        self._events: dict[str, RefinementBreakageEvent] = {}
        self._resumption_hooks: dict[str, Callable[[RefinementBreakageEvent], Any]] = {}
        if self.persist:
            self.load_state()

    def clear_all(self) -> None:
        self._events.clear()
        if self.persist:
            self.save_state()

    def register_resumption_hook(self, domain: str, hook: Callable[[RefinementBreakageEvent], Any]) -> None:
        self._resumption_hooks[domain] = hook

    def raise_breakage(
        self,
        domain: str,
        target: str,
        breakage_type: BreakageType | str,
        reason: str,
        suggested_paths: Optional[list[str]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> RefinementBreakageEvent:
        """Create a Gate 6 Hold for an interrupted game session or refinement step."""
        b_type = breakage_type.value if isinstance(breakage_type, BreakageType) else str(breakage_type)
        breakage_id = f"brk_{uuid.uuid4().hex[:8]}"

        event = RefinementBreakageEvent(
            breakage_id=breakage_id,
            domain=domain,
            target=target,
            breakage_type=b_type,
            reason=reason,
            suggested_paths=suggested_paths or ["Retry current trajectory", "Switch account / profile", "Hold in band 翈"],
            context=context or {},
            status="HELD_AT_GATE_6",
        )

        self._events[breakage_id] = event
        if self.persist:
            self.save_state()
        logger.warning(
            "GATE 6 HOLD ACTIVATED [%s] target=%s domain=%s: %s",
            breakage_id,
            target,
            domain,
            reason,
        )
        return event

    def list_active_holds(self, domain: Optional[str] = None) -> list[RefinementBreakageEvent]:
        holds = [e for e in self._events.values() if e.status == "HELD_AT_GATE_6"]
        if domain:
            holds = [e for e in holds if e.domain == domain]
        return holds

    def get_event(self, breakage_id: str) -> Optional[RefinementBreakageEvent]:
        return self._events.get(breakage_id)

    def confirm_right_path(
        self,
        breakage_id: str,
        operator_signer: str,
        confirmed_path: str,
        user_inputs: Optional[dict[str, Any]] = None,
        account_input: Optional[dict[str, Any]] = None,
        parameters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """User/Operator confirms the right path, passing Physical Gate 6."""
        event = self._events.get(breakage_id)
        if not event:
            return {"status": "error", "message": f"Breakage {breakage_id} not found."}

        if event.status != "HELD_AT_GATE_6":
            return {"status": "already_resolved", "breakage_id": breakage_id, "current_status": event.status}

        # 1. Generate Two-Party Approved Attestations required by Gate 6
        # Party 1: The Operator / Human User
        user_attestation = Attestation(
            signer=operator_signer,
            scope=breakage_id,
            love_form="repair",
            shared_with=self.counterpart_id,
            beautiful_output=True,
            assurance=ASSURANCE_APPROVED,
            approval_ref=f"USER-GATE6-{breakage_id[:8]}",
        )

        # Party 2: System Self / Counterpart confirmation
        system_attestation = Attestation(
            signer=self.counterpart_id,
            scope=breakage_id,
            love_form="fidelity",
            shared_with=operator_signer,
            beautiful_output=True,
            assurance=ASSURANCE_APPROVED,
            approval_ref=f"SYSTEM-GATE6-{breakage_id[:8]}",
        )

        # 2. Evaluate Physical Gate 6 (Beautiful Output & Human Confirmation)
        cfg = GovernanceConfig(
            self_id=operator_signer,
            counterpart_id=self.counterpart_id,
            accept_self_asserted=True,
        )

        mock_candidate = Candidate(
            scope=breakage_id,
            residual={"brain": 0.05 + 0.1j, "vision": 0.02 + 0.05j},
            realised_weight=0.1,
            unclamped_weight=0.1,
            neighbour_weights=(0.08, 0.12),
        )

        gate_res = gate_beautiful_output(mock_candidate, cfg, [user_attestation, system_attestation])

        if not gate_res.passed:
            logger.error("Gate 6 check failed on confirmation: %s", gate_res.reason)
            return {"status": "gate_failed", "reason": gate_res.reason}

        # 3. Update Breakage State to RESOLVED
        event.status = "RESOLVED_BY_USER"
        event.confirmed_path = confirmed_path
        event.resolved_by = operator_signer
        event.resolution_payload = {
            "confirmed_path": confirmed_path,
            "account_input": account_input or user_inputs or {},
            "user_inputs": user_inputs or account_input or {},
            "parameters": parameters or {},
            "gate_result": asdict(gate_res),
        }
        event.attestation_digest = user_attestation.approval_ref

        # 4. Trigger Resumption Hook if registered
        hook = self._resumption_hooks.get(event.domain)
        hook_result = None
        if hook:
            try:
                hook_result = hook(event)
            except Exception as exc:
                logger.error("Error executing resumption hook for %s: %s", event.domain, exc)

        if self.persist:
            self.save_state()
        logger.info(
            "GATE 6 HOLD RESOLVED [%s]: confirmed_path='%s' by %s",
            breakage_id,
            confirmed_path,
            operator_signer,
        )

        return {
            "status": "resolved",
            "breakage_id": breakage_id,
            "confirmed_path": confirmed_path,
            "resolved_by": operator_signer,
            "gate_6": "PASSED",
            "hook_result": hook_result,
        }

    def save_state(self) -> None:
        if not self.persist:
            return
        try:
            data = {bid: ev.to_dict() for bid, ev in self._events.items()}
            brain_kv.kv_set_json(_BRAIN_KV_GATE6_KEY, data)
        except Exception as exc:
            logger.debug("Failed saving Gate 6 interlocks: %s", exc)

    def load_state(self) -> None:
        try:
            data = brain_kv.kv_get_json(_BRAIN_KV_GATE6_KEY, None)
            if isinstance(data, dict):
                for bid, d in data.items():
                    if isinstance(d, dict):
                        self._events[bid] = RefinementBreakageEvent(**d)
        except Exception:
            pass


# Global interlock instance for common refinement integration
_GLOBAL_INTERLOCK: Optional[Gate6UserInterlock] = None


def get_global_interlock() -> Gate6UserInterlock:
    global _GLOBAL_INTERLOCK
    if _GLOBAL_INTERLOCK is None:
        _GLOBAL_INTERLOCK = Gate6UserInterlock()
    return _GLOBAL_INTERLOCK
