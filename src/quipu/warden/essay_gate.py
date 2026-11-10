"""Progression gate: Guild Wars, then WoW Legacy, unlocked by The Essay.

``EssayGate`` only ever *reads* recorded results. It has no method that opens a
stage, advances one, or writes a record, so it cannot talk itself forward. A stage
opens because a Warden-signed record says so, and for no other reason.

Stages
------
``guild_wars``
    opens when the required Essay test is recorded as passed.
``wow_legacy``
    opens only when the required test is passed **and** Guild Wars is recorded
    complete.

Operator consent
----------------
By default **every** game stage additionally requires the operator's own
attestation (see :mod:`operator_attestation`), not just WoW Legacy. The plan only
named the WoW server, but Guild Wars has the same property: it is a service run by
someone else, and a Warden's grant cannot stand in for their consent. Loosen this
deliberately with ``operator_required_stages`` if a stage genuinely has no
third-party operator.

Record shape (an assumption to confirm)
---------------------------------------
Where The Essay's results actually live is open item 1 in the plan. This module
therefore depends on the :class:`ProgressSource` protocol, not on a storage
layout. :class:`SignedRecordProgress` is the default: records signed by a Warden
and re-verified, including that the signer is *still* a Warden, on every read.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ._common import (
    KVStore,
    MemoryStore,
    WardenAuthorityError,
    WardenFormatError,
    WardenSecretError,
    hmac_sign,
    hmac_verify,
    load_secret,
)
from .operator_attestation import OperatorAttestation, OperatorRegistry, verify_attestation
from .warden_grant import WardenRegistry, _check_id, parse_target

RECORD_LABEL = "record"
STAGE_ORDER = ("guild_wars", "wow_legacy")


class ProgressSource(Protocol):
    """Read-only view of recorded progress. Implement this over the real Essay data."""

    def test_passed(self, test_id: str) -> bool: ...
    def stage_complete(self, stage: str) -> bool: ...


class SignedRecordProgress:
    """Default :class:`ProgressSource`: Warden-signed records in a key/value store."""

    def __init__(
        self,
        store: KVStore | None = None,
        *,
        registry: WardenRegistry,
        secret: str | bytes | None = None,
    ) -> None:
        self._store: KVStore = store if store is not None else MemoryStore()
        self._registry = registry
        self._secret = secret

    @staticmethod
    def _test_key(test_id: str) -> str:
        return f"warden:essay:test:{test_id}"

    @staticmethod
    def _stage_key(stage: str) -> str:
        return f"warden:essay:stage:{stage}"

    def _write(self, key: str, body: dict[str, Any], warden_id: str) -> dict[str, Any]:
        if not self._registry.is_warden(warden_id):
            raise WardenAuthorityError(f"{warden_id!r} is not a registered Warden")
        record = {**body, "recorded_by": warden_id, "recorded_at": int(time.time())}
        record["signature"] = hmac_sign(load_secret(self._secret), RECORD_LABEL, record)
        self._store.set(key, record)
        return record

    def record_test(self, test_id: str, passed: bool, *, warden_id: str) -> dict[str, Any]:
        _check_id(test_id, "test_id")
        return self._write(
            self._test_key(test_id),
            {"kind": "essay_test", "test_id": test_id, "passed": bool(passed)},
            warden_id,
        )

    def record_stage_complete(self, stage: str, *, warden_id: str) -> dict[str, Any]:
        if stage not in STAGE_ORDER:
            raise WardenFormatError(f"unknown stage: {stage!r}")
        return self._write(self._stage_key(stage), {"kind": "stage_complete", "stage": stage}, warden_id)

    def _read(self, key: str, kind: str) -> Mapping[str, Any] | None:
        record = self._store.get(key)
        if not isinstance(record, Mapping) or record.get("kind") != kind:
            return None
        body = {k: v for k, v in record.items() if k != "signature"}
        if not hmac_verify(load_secret(self._secret), RECORD_LABEL, body, record.get("signature")):
            return None
        signer = record.get("recorded_by")
        if not isinstance(signer, str) or not self._registry.is_warden(signer):
            return None
        return record

    def test_passed(self, test_id: str) -> bool:
        record = self._read(self._test_key(test_id), "essay_test")
        return bool(record and record.get("test_id") == test_id and record.get("passed") is True)

    def stage_complete(self, stage: str) -> bool:
        record = self._read(self._stage_key(stage), "stage_complete")
        return bool(record and record.get("stage") == stage)


@dataclass(frozen=True)
class StageDecision:
    open: bool
    stage: str | None
    reason: str


def stage_for_target(target: str) -> str | None:
    """Map a game target to its stage name, or ``None`` if it is not a game."""
    try:
        kind, ident = parse_target(target)
    except WardenFormatError:
        return None
    if kind != "game":
        return None
    return ident.split(":", 1)[0]


class EssayGate:
    def __init__(
        self,
        *,
        required_test: str,
        progress: ProgressSource,
        operators: OperatorRegistry,
        operator_required_stages: frozenset[str] = frozenset(STAGE_ORDER),
    ) -> None:
        self.required_test = _check_id(required_test, "required_test")
        self._progress = progress
        self._operators = operators
        self._operator_required = frozenset(operator_required_stages)

    def decide(
        self,
        *,
        target: str,
        agent_id: str,
        now: float,
        attestation: OperatorAttestation | None = None,
    ) -> StageDecision:
        stage = stage_for_target(target)
        if stage is None:
            return StageDecision(False, None, "not_a_game_target")
        try:
            if not self._progress.test_passed(self.required_test):
                return StageDecision(False, stage, "essay_test_not_passed")
            if stage == "wow_legacy" and not self._progress.stage_complete("guild_wars"):
                return StageDecision(False, stage, "guild_wars_not_complete")
        except WardenSecretError:
            return StageDecision(False, stage, "secret_unset")
        if stage in self._operator_required:
            ok, reason = verify_attestation(
                attestation, agent_id=agent_id, target=target, now=now, operators=self._operators
            )
            if not ok:
                return StageDecision(False, stage, reason)
        return StageDecision(True, stage, "ok")
