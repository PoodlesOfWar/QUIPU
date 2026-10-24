"""Game adapters: thin, behind the gate, and ship as stubs.

The interface is real; the wiring is deliberately not here. Connecting an agent to
a particular server is a separate step that only makes sense once the open
question in the plan is settled: *who operates the hosted server, and have they
agreed?* Until a :class:`~operator_attestation.OperatorRegistry` holds that
operator's key and they have signed an attestation for the agent, the gate will not
let a call reach an adapter at all.

Hard lines
----------
* **Do not point these at retail or third-party servers.** Their terms prohibit
  bots, and nothing in this package changes that. The operator attestation is
  evidence of consent, not a way around its absence.
* **Disclosure is mandatory.** Every adapter call receives the request's
  disclosure record (``automated: true`` plus the agent and grant ids). An adapter
  must forward it to the server or its operator and must refuse a call that does
  not carry it (:func:`require_disclosure`). Do not add anything that obscures
  that a client is automated: no input-timing randomization, no human-likeness
  tuning, no detection evasion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ._common import WardenError
from .gate import Decision, WardenGate
from .operator_attestation import OperatorAttestation
from .signed_agent import SignedRequest


class AdapterNotWired(WardenError):
    """The adapter is an interface stub; no server wiring exists yet."""


class DisclosureMissing(WardenError):
    """An adapter was called without the automation disclosure."""


def require_disclosure(disclosure: Mapping[str, Any] | None) -> None:
    """Refuse any call that does not declare itself automated."""
    if not isinstance(disclosure, Mapping) or disclosure.get("automated") is not True:
        raise DisclosureMissing("adapter calls must carry the automation disclosure")
    if not disclosure.get("agent_id") or not disclosure.get("grant_id"):
        raise DisclosureMissing("disclosure must identify the agent and grant")


class GameAdapter(Protocol):
    target: str

    def connect(self, disclosure: Mapping[str, Any]) -> None: ...
    def act(self, action: str, params: Mapping[str, Any], disclosure: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def disconnect(self) -> None: ...


class _StubAdapter:
    target = ""
    _label = ""

    def connect(self, disclosure: Mapping[str, Any]) -> None:
        require_disclosure(disclosure)
        raise AdapterNotWired(
            f"{self._label} is an interface stub. Wiring a specific server is a separate "
            "step that requires that server's operator to have agreed and attested."
        )

    def act(self, action: str, params: Mapping[str, Any], disclosure: Mapping[str, Any]) -> Mapping[str, Any]:
        require_disclosure(disclosure)
        raise AdapterNotWired(f"{self._label} is an interface stub; nothing was sent.")

    def disconnect(self) -> None:
        return None


class GuildWarsAdapter(_StubAdapter):
    """Stage 1. ``target`` is ``game:guild_wars``."""

    target = "game:guild_wars"
    _label = "GuildWarsAdapter"


class WowLegacyAdapter(_StubAdapter):
    """Stage 2. ``target`` is ``game:wow_legacy:<server>``, the operator-run host."""

    _label = "WowLegacyAdapter"

    def __init__(self, server: str) -> None:
        self.target = f"game:wow_legacy:{server}"


class RecordingAdapter:
    """A working adapter that talks to nothing: records calls for tests and dry runs."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.connected = False
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def connect(self, disclosure: Mapping[str, Any]) -> None:
        require_disclosure(disclosure)
        self.connected = True
        self.calls.append(("game.connect", {}, dict(disclosure)))

    def act(self, action: str, params: Mapping[str, Any], disclosure: Mapping[str, Any]) -> Mapping[str, Any]:
        require_disclosure(disclosure)
        self.calls.append((action, dict(params), dict(disclosure)))
        return {"recorded": True}

    def disconnect(self) -> None:
        self.connected = False


@dataclass(frozen=True)
class DispatchResult:
    ok: bool
    reason: str
    decision: Decision | None = None
    result: Mapping[str, Any] | None = None


class GameDispatcher:
    """Routes a signed request through the whole gate and then to an adapter."""

    def __init__(self, *, gate: WardenGate, adapters: Mapping[str, GameAdapter]) -> None:
        self._gate = gate
        self._adapters = dict(adapters)

    def dispatch(
        self, request: SignedRequest, *, attestation: OperatorAttestation | None = None
    ) -> DispatchResult:
        decision = self._gate.authorize(request, attestation=attestation)
        if not decision.allow:
            return DispatchResult(False, decision.code, decision)
        adapter = self._adapters.get(request.target)
        if adapter is None:
            return DispatchResult(False, "dispatch:no_adapter", decision)
        disclosure = request.disclosure()
        try:
            if request.action == "game.connect":
                adapter.connect(disclosure)
                return DispatchResult(True, "ok", decision, {"connected": True})
            return DispatchResult(True, "ok", decision, adapter.act(request.action, request.params, disclosure))
        except AdapterNotWired as exc:
            return DispatchResult(False, "dispatch:adapter_not_wired", decision, {"detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 - adapter failures must not escape
            return DispatchResult(False, f"dispatch:adapter_error:{type(exc).__name__}", decision)
