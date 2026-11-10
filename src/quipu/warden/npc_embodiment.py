"""World 4 NPC <-> Reachy 2 binding.

A Warden-authorized agent can bind an NPC to a Reachy 2 body and drive it. Four
rules keep that safe, and each is enforced in code rather than by convention:

1. **Binding needs two live grants.** One for ``npc:<id>`` (``npc.bind``) and one
   for ``reachy2:<host>`` (``robot.bind``), both naming the requesting agent, and
   the request itself must carry that agent's signature.
2. **No motion without a live grant.** Both grants are re-verified on *every*
   ``act``. There is no cached "still authorized"; a revoke made in another
   process stops the very next action.
3. **A bounded action set.** Only the intents in :data:`INTENT_KINDS` exist, each
   with numeric bounds validated before any driver is touched. Anything else is
   rejected, not clamped.
4. **A software e-stop that revoke triggers.** Revoking either grant halts the
   driver immediately through a listener on the :class:`GrantStore`. Expiry is
   caught on the next ``act`` and by :meth:`NpcEmbodiment.reap`.

The e-stop is a *software* stop. It cancels queued motion and relaxes the robot,
but it is not a substitute for the physical emergency stop on the hardware.

What is authorized is what is executed: the intent is read from the *signed*
request parameters, so the agent cannot be authorized for one thing and made to do
another.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from ._common import KVStore, MemoryStore, WardenError, WardenFormatError
from .gate import Decision, WardenGate
from .signed_agent import SignedRequest
from .warden_grant import GrantStore, parse_target

# --------------------------------------------------------------------------- #
# Bounded intents
# --------------------------------------------------------------------------- #

# intent kind -> the robot action class the Warden must have scoped.
INTENT_KINDS: dict[str, str] = {
    "look_at": "robot.move",
    "gesture": "robot.gesture",
    "speak": "robot.speak",
    "idle": "robot.observe",
}

# Torso-frame metres: x forward, y left, z up (the Reachy 2 SDK convention).
LOOK_X = (0.2, 2.0)
LOOK_Y = (-1.0, 1.0)
LOOK_Z = (-0.5, 1.0)
DURATION_S = (0.5, 10.0)
DEFAULT_DURATION_S = 2.0
GESTURES = frozenset({"neutral"})
SPEAK_MAX_CHARS = 280

# Grant-layer denials that mean the authority itself is gone, as opposed to a
# single out-of-scope action. These halt the robot; the others only refuse the act.
HALT_REASONS = frozenset(
    {"revoked", "expired", "warden_not_authorized", "bad_signature", "grant_not_found", "not_yet_valid"}
)


class IntentRejected(WardenError):
    """The requested intent is outside the bounded action set."""


class DriverRejected(WardenError):
    """The driver refused or could not carry out an authorized intent."""


class UnsupportedIntent(DriverRejected):
    """This driver has no verified implementation of the intent."""


@dataclass(frozen=True)
class BoundedIntent:
    kind: str
    params: Mapping[str, Any]

    @property
    def robot_action(self) -> str:
        return INTENT_KINDS[self.kind]


def _num(value: Any, field: str, bounds: tuple[float, float]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IntentRejected(f"{field} must be a number")
    if not math.isfinite(value):
        raise IntentRejected(f"{field} must be finite")
    low, high = bounds
    if not (low <= value <= high):
        raise IntentRejected(f"{field}={value} outside [{low}, {high}]")
    return float(value)


def parse_intent(params: Mapping[str, Any]) -> BoundedIntent:
    """Validate request parameters into a :class:`BoundedIntent` or raise."""
    kind = params.get("intent") if isinstance(params, Mapping) else None
    if kind not in INTENT_KINDS:
        raise IntentRejected(f"unknown intent: {kind!r}")
    if kind == "look_at":
        return BoundedIntent(kind, {
            "x": _num(params.get("x"), "x", LOOK_X),
            "y": _num(params.get("y"), "y", LOOK_Y),
            "z": _num(params.get("z"), "z", LOOK_Z),
            "duration": _num(params.get("duration", DEFAULT_DURATION_S), "duration", DURATION_S),
        })
    if kind == "gesture":
        name = params.get("name")
        if name not in GESTURES:
            raise IntentRejected(f"unknown gesture: {name!r}")
        return BoundedIntent(kind, {"name": name})
    if kind == "speak":
        text = params.get("text")
        if not isinstance(text, str) or not text or len(text) > SPEAK_MAX_CHARS:
            raise IntentRejected(f"text must be 1..{SPEAK_MAX_CHARS} characters")
        if any(ord(ch) < 32 and ch not in "\n\t" for ch in text):
            raise IntentRejected("text contains control characters")
        return BoundedIntent(kind, {"text": text})
    return BoundedIntent("idle", {})


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #

class RobotDriver(Protocol):
    def connect(self) -> None: ...
    def act(self, intent: BoundedIntent) -> None: ...
    def stop(self) -> bool: ...


class SimDriver:
    """Default driver and the one tests use. Records intents; touches nothing."""

    def __init__(self, host: str = "sim") -> None:
        self.host = host
        self.log: list[BoundedIntent] = []
        self.connected = False
        self.stopped = False

    def connect(self) -> None:
        if self.stopped:
            raise DriverRejected("driver was stopped; create a new one")
        self.connected = True

    def act(self, intent: BoundedIntent) -> None:
        if self.stopped or not self.connected:
            raise DriverRejected("driver is not live")
        self.log.append(intent)

    def stop(self) -> bool:
        self.stopped = True
        self.connected = False
        return True


class Reachy2Driver:
    """Drives a Pollen Robotics Reachy 2 through ``reachy2-sdk``.

    The SDK is imported lazily, so merely importing this module needs neither the
    package nor a robot. The calls used were checked against ``reachy2-sdk``
    1.0.15: ``ReachySDK(host=...)``, ``.connect()``, ``.is_connected()``,
    ``.turn_on()``, ``.head.look_at(x, y, z, duration, wait)``,
    ``.goto_posture("default", ...)``, ``.cancel_all_goto()``,
    ``.turn_off_smoothly()`` and ``.disconnect()``. Re-verify against the version
    you deploy, and **try it on Pollen's simulator before hardware**.

    Deliberately minimal: head and neutral posture only. No arms, no mobile base.
    ``speak`` is unsupported unless you inject ``speak_fn`` (the SDK has no
    verified text-to-speech path to wrap).
    """

    def __init__(
        self,
        host: str,
        *,
        sdk_factory: Callable[[str], Any] | None = None,
        speak_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.host = host
        self._sdk_factory = sdk_factory
        self._speak_fn = speak_fn
        self._reachy: Any = None
        self.last_stop_errors: list[str] = []

    def connect(self) -> None:
        if self._sdk_factory is not None:
            reachy = self._sdk_factory(self.host)
        else:
            try:
                from reachy2_sdk import ReachySDK  # lazy: not needed to import this module
            except ImportError as exc:
                raise DriverRejected("reachy2-sdk is not installed (pip install reachy2-sdk)") from exc
            reachy = ReachySDK(host=self.host)
        reachy.connect()
        if not reachy.is_connected():
            raise DriverRejected(f"could not connect to Reachy 2 at {self.host}")
        self._reachy = reachy
        if not reachy.turn_on():
            self._reachy = None
            raise DriverRejected("Reachy 2 refused to turn on")

    def act(self, intent: BoundedIntent) -> None:
        reachy = self._reachy
        if reachy is None:
            raise DriverRejected("driver is not live")
        if intent.kind == "look_at":
            goto = reachy.head.look_at(
                intent.params["x"], intent.params["y"], intent.params["z"],
                duration=intent.params["duration"], wait=False,
            )
            # The SDK signals "not reachable" or "neck off" with id == -1.
            if getattr(goto, "id", 0) == -1:
                raise DriverRejected("Reachy 2 rejected the look_at target")
        elif intent.kind == "gesture":
            reachy.goto_posture("default", duration=DEFAULT_DURATION_S, wait=False)
        elif intent.kind == "speak":
            if self._speak_fn is None:
                raise UnsupportedIntent("speak requires an injected speak_fn")
            self._speak_fn(intent.params["text"])
        # "idle" is a deliberate no-op.

    def stop(self) -> bool:
        """E-stop: cancel queued motion, relax gently, release. Never raises.

        Each step is attempted even if an earlier one fails, because partial
        failure is exactly when the remaining steps matter most.
        """
        reachy, self._reachy = self._reachy, None
        self.last_stop_errors = []
        if reachy is None:
            return True
        for name in ("cancel_all_goto", "turn_off_smoothly", "disconnect"):
            try:
                getattr(reachy, name)()
            except Exception as exc:  # noqa: BLE001
                self.last_stop_errors.append(f"{name}: {type(exc).__name__}")
        return not self.last_stop_errors


# --------------------------------------------------------------------------- #
# Binding
# --------------------------------------------------------------------------- #

class BindingDenied(WardenError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ActResult:
    ok: bool
    reason: str
    decision: Decision | None = None


@dataclass
class _Live:
    driver: RobotDriver
    npc_id: str
    host: str
    agent_id: str
    npc_grant_id: str
    reachy_grant_id: str
    expires_at: int


class NpcEmbodiment:
    """Binds World 4 NPCs to Reachy 2 bodies and gates every motion.

    One process owns each robot connection, so live bindings are in-memory; the
    store carries a record of each binding's status for visibility. After a
    restart nothing is live until ``bind_npc`` runs again and re-verifies both
    grants.
    """

    def __init__(
        self,
        *,
        gate: WardenGate,
        grants: GrantStore,
        store: KVStore | None = None,
        driver_factory: Callable[[str], RobotDriver] | None = None,
        audit_kind: str = "embodiment",
    ) -> None:
        self._gate = gate
        self._grants = grants
        self._store: KVStore = store if store is not None else MemoryStore()
        self._factory: Callable[[str], RobotDriver] = driver_factory or (lambda host: SimDriver(host))
        self._audit = gate.audit  # same chain as the gate's decisions
        self._audit_kind = audit_kind
        self._live: dict[str, _Live] = {}
        self._lock = threading.RLock()
        grants.add_revoke_listener(self._on_grant_revoked)

    # ---- audit / persistence helpers -------------------------------------- #

    def _log(self, actor: str, **details: Any) -> None:
        try:
            self._audit.append(self._audit_kind, actor, details)
        except Exception:  # noqa: BLE001 - never let logging block a stop
            pass

    def _persist(self, npc_id: str, record: dict[str, Any]) -> None:
        try:
            self._store.set(f"warden:binding:{npc_id}", record)
        except Exception:  # noqa: BLE001
            pass

    # ---- binding ---------------------------------------------------------- #

    def bind_npc(self, request: SignedRequest) -> dict[str, Any]:
        """Bind an NPC to a Reachy 2. ``request`` is a signed ``npc.bind``.

        Expected parameters: ``reachy_host`` and ``reachy_grant_id``. The request
        targets ``npc:<id>`` under the agent's NPC grant; the robot grant is
        checked here for ``robot.bind`` on ``reachy2:<host>``.
        """
        params = request.params if isinstance(request.params, Mapping) else {}
        host = params.get("reachy_host")
        reachy_grant_id = params.get("reachy_grant_id")
        try:
            kind, npc_id = parse_target(request.target)
            if kind != "npc" or request.action != "npc.bind":
                raise BindingDenied("not_an_npc_bind_request")
            if not isinstance(host, str) or not isinstance(reachy_grant_id, str):
                raise BindingDenied("bad_robot_parameters")
            robot_kind, _ = parse_target(f"reachy2:{host}")
            if robot_kind != "robot":
                raise BindingDenied("bad_robot_parameters")
        except WardenFormatError as exc:
            raise BindingDenied(f"bad_parameters:{exc}") from exc

        with self._lock:
            if npc_id in self._live:
                raise BindingDenied("npc_already_bound")
            if any(item.host == host for item in self._live.values()):
                raise BindingDenied("robot_already_bound")

            npc_decision = self._gate.authorize(request)
            if not npc_decision.allow:
                raise BindingDenied(f"npc_grant:{npc_decision.code}")
            robot_decision = self._gate.check_grant(
                grant_id=reachy_grant_id, agent_id=request.agent_id,
                target=f"reachy2:{host}", action="robot.bind",
            )
            if not robot_decision.allow:
                raise BindingDenied(f"robot_grant:{robot_decision.code}")

            npc_grant = self._grants.get(request.grant_id)
            robot_grant = self._grants.get(reachy_grant_id)
            if npc_grant is None or robot_grant is None:  # revoked/removed between checks
                raise BindingDenied("grant_not_found")

            driver = self._factory(host)
            try:
                driver.connect()
            except Exception as exc:  # noqa: BLE001
                self._safe_stop(driver)
                self._log(request.agent_id, event="bind_failed", npc_id=npc_id, host=host,
                          error=type(exc).__name__)
                raise BindingDenied(f"driver_connect_failed:{type(exc).__name__}") from exc

            live = _Live(
                driver=driver, npc_id=npc_id, host=host, agent_id=request.agent_id,
                npc_grant_id=request.grant_id, reachy_grant_id=reachy_grant_id,
                expires_at=min(npc_grant.expires_at, robot_grant.expires_at),
            )
            self._live[npc_id] = live
            record = {
                "npc_id": npc_id, "reachy_host": host, "agent_id": request.agent_id,
                "npc_grant_id": live.npc_grant_id, "reachy_grant_id": live.reachy_grant_id,
                "expires_at": live.expires_at, "status": "bound",
            }
            self._persist(npc_id, record)
            self._log(request.agent_id, event="bind", **{k: v for k, v in record.items() if k != "status"})
            return record

    def binding(self, npc_id: str) -> dict[str, Any] | None:
        with self._lock:
            live = self._live.get(npc_id)
            if live is None:
                return None
            return {
                "npc_id": live.npc_id, "reachy_host": live.host, "agent_id": live.agent_id,
                "npc_grant_id": live.npc_grant_id, "reachy_grant_id": live.reachy_grant_id,
                "expires_at": live.expires_at,
            }

    # ---- acting ----------------------------------------------------------- #

    def act(self, request: SignedRequest) -> ActResult:
        """Carry out one signed ``npc.act`` request, re-checking everything first."""
        try:
            kind, npc_id = parse_target(request.target)
        except WardenFormatError:
            return ActResult(False, "bad_target")
        if kind != "npc" or request.action != "npc.act":
            return ActResult(False, "not_an_npc_act_request")

        with self._lock:
            live = self._live.get(npc_id)
            if live is None:
                self._log(str(request.agent_id)[:128], event="act_denied", npc_id=npc_id, reason="not_bound")
                return ActResult(False, "not_bound")

            decision = self._gate.authorize(request)
            if not decision.allow:
                if decision.layer == "grant" and decision.reason in HALT_REASONS:
                    self.estop(npc_id, reason=f"npc_grant:{decision.code}")
                return ActResult(False, decision.code, decision)
            if request.agent_id != live.agent_id or request.grant_id != live.npc_grant_id:
                return ActResult(False, "binding_mismatch", decision)

            try:
                intent = parse_intent(request.params)
            except IntentRejected as exc:
                self._log(request.agent_id, event="act_denied", npc_id=npc_id, reason=f"intent:{exc}")
                return ActResult(False, f"intent_rejected:{exc}", decision)

            robot_decision = self._gate.check_grant(
                grant_id=live.reachy_grant_id, agent_id=request.agent_id,
                target=f"reachy2:{live.host}", action=intent.robot_action,
            )
            if not robot_decision.allow:
                if robot_decision.layer == "grant" and robot_decision.reason in HALT_REASONS:
                    self.estop(npc_id, reason=f"robot_grant:{robot_decision.code}")
                return ActResult(False, f"robot:{robot_decision.code}", robot_decision)

            try:
                live.driver.act(intent)
            except Exception as exc:  # noqa: BLE001
                self._log(request.agent_id, event="act_failed", npc_id=npc_id, intent=intent.kind,
                          error=type(exc).__name__)
                return ActResult(False, f"driver:{type(exc).__name__}", decision)
            self._log(request.agent_id, event="act", npc_id=npc_id, intent=intent.kind)
            return ActResult(True, "ok", decision)

    # ---- stopping --------------------------------------------------------- #

    @staticmethod
    def _safe_stop(driver: RobotDriver) -> bool:
        try:
            return bool(driver.stop())
        except Exception:  # noqa: BLE001
            return False

    def estop(self, npc_id: str, *, reason: str) -> bool:
        """Halt the driver and unbind. Idempotent; returns whether something was live."""
        with self._lock:
            live = self._live.pop(npc_id, None)
            if live is None:
                return False
            stopped = self._safe_stop(live.driver)
            self._persist(npc_id, {"npc_id": npc_id, "status": "unbound", "reason": reason[:256]})
            self._log(live.agent_id, event="estop", npc_id=npc_id, host=live.host,
                      reason=reason[:256], driver_stop_ok=stopped)
            return True

    def unbind(self, npc_id: str, *, reason: str = "unbind") -> bool:
        return self.estop(npc_id, reason=reason)

    def estop_all(self, *, reason: str = "estop_all") -> int:
        with self._lock:
            return sum(1 for npc_id in list(self._live) if self.estop(npc_id, reason=reason))

    def reap(self, now: float | None = None) -> list[str]:
        """Halt bindings whose grants have expired or been revoked.

        Call this on a timer. It exists for the idle case; any ``act`` already
        re-checks, so an *acting* NPC never depends on the timer.
        """
        stamp = self._gate.clock() if now is None else now
        halted: list[str] = []
        with self._lock:
            for npc_id, live in list(self._live.items()):
                if (
                    stamp >= live.expires_at
                    or self._grants.is_revoked(live.npc_grant_id)
                    or self._grants.is_revoked(live.reachy_grant_id)
                ):
                    self.estop(npc_id, reason="expired_or_revoked")
                    halted.append(npc_id)
        return halted

    def _on_grant_revoked(self, grant_id: str) -> None:
        with self._lock:
            for npc_id, live in list(self._live.items()):
                if grant_id in (live.npc_grant_id, live.reachy_grant_id):
                    self.estop(npc_id, reason=f"grant_revoked:{grant_id}")
