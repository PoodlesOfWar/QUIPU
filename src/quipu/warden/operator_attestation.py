"""Operator attestations: the game operator's own signature, as a required second key.

A Warden's grant says "this agent may do this". It cannot say "the people running
this server agree", because the Warden does not run the server. For any game
target that second consent must come from the operator, signed with a key the
operator holds and we do not.

That is why attestations are Ed25519 and not HMAC. With a shared secret we would
hold the operator's signing key, and an "operator signature" we could mint
ourselves would prove nothing. Here the registry stores only the operator's
*public* key.

Scope of this module
--------------------
It records and checks consent. It does not make a retail or third-party service
automatable: those services' terms prohibit bots, and no signature of ours changes
that. Only register an operator for a server whose operator has actually agreed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from ._common import (
    KVStore,
    MemoryStore,
    WardenAuthorityError,
    WardenFormatError,
    canonical_json,
)
from .signed_agent import _HEX_RE, _ed25519, ed25519_verify, is_small_order_key
from .warden_grant import CLOCK_SKEW_S, WardenRegistry, _check_epoch, _check_id, parse_target

ATTESTATION_LABEL = b"warden-operator/v1\n"


@dataclass(frozen=True)
class OperatorAttestation:
    attestation_id: str
    operator_id: str
    agent_id: str
    target: str
    issued_at: int
    expires_at: int
    signature: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "attestation_id": self.attestation_id,
            "operator_id": self.operator_id,
            "agent_id": self.agent_id,
            "target": self.target,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    def signing_bytes(self) -> bytes:
        return ATTESTATION_LABEL + canonical_json(self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, data: Any) -> "OperatorAttestation":
        if not isinstance(data, Mapping):
            raise WardenFormatError("attestation must be an object")
        kind, _ = parse_target(data.get("target"))
        if kind != "game":
            raise WardenFormatError("operator attestations apply to game targets only")
        return cls(
            attestation_id=_check_id(data.get("attestation_id"), "attestation_id"),
            operator_id=_check_id(data.get("operator_id"), "operator_id"),
            agent_id=_check_id(data.get("agent_id"), "agent_id"),
            target=data["target"],
            issued_at=_check_epoch(data.get("issued_at"), "issued_at"),
            expires_at=_check_epoch(data.get("expires_at"), "expires_at"),
            signature=data.get("signature") if isinstance(data.get("signature"), str) else "",
        )


class OperatorSigner:
    """The operator's side: holds the private key and signs attestations.

    Used by the operator's own tooling (and by tests). The gate never has one.
    """

    def __init__(self, operator_id: str, private_key: Any) -> None:
        self.operator_id = _check_id(operator_id, "operator_id")
        self._private_key = private_key

    @classmethod
    def generate(cls, operator_id: str) -> "OperatorSigner":
        private_cls, _, _ = _ed25519()
        return cls(operator_id, private_cls.generate())

    @property
    def public_key_hex(self) -> str:
        from cryptography.hazmat.primitives import serialization
        return self._private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ).hex()

    def attest(
        self,
        *,
        agent_id: str,
        target: str,
        ttl_s: float,
        now: float | None = None,
        attestation_id: str | None = None,
    ) -> OperatorAttestation:
        import secrets as _secrets
        kind, _ = parse_target(target)
        if kind != "game":
            raise WardenFormatError("operator attestations apply to game targets only")
        if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)) or ttl_s <= 0:
            raise WardenFormatError("ttl_s must be positive")
        issued = int(time.time() if now is None else now)
        unsigned = OperatorAttestation(
            attestation_id=attestation_id or _secrets.token_hex(16),
            operator_id=self.operator_id,
            agent_id=_check_id(agent_id, "agent_id"),
            target=target,
            issued_at=issued,
            expires_at=issued + int(ttl_s),
        )
        return replace(unsigned, signature=self._private_key.sign(unsigned.signing_bytes()).hex())


class OperatorRegistry:
    """Which operator runs which game target, and their public key.

    Registering an operator is an authority decision, so it needs a Warden. One
    target has exactly one operator: a second claim on the same target is refused
    rather than silently replacing the first.
    """

    def __init__(self, store: KVStore | None = None) -> None:
        self._store: KVStore = store if store is not None else MemoryStore()

    @staticmethod
    def _op_key(operator_id: str) -> str:
        return f"warden:operator:{operator_id}"

    @staticmethod
    def _target_key(target: str) -> str:
        return f"warden:operator_target:{target}"

    @staticmethod
    def _revoked_key(attestation_id: str) -> str:
        return f"warden:attestation_revoked:{attestation_id}"

    def register(
        self,
        operator_id: str,
        public_key_hex: str,
        targets: Iterable[str],
        *,
        warden_id: str,
        registry: WardenRegistry,
    ) -> None:
        _check_id(operator_id, "operator_id")
        if not registry.is_warden(warden_id):
            raise WardenAuthorityError(f"{warden_id!r} is not a registered Warden")
        if not isinstance(public_key_hex, str) or len(public_key_hex) != 64 or not _HEX_RE.fullmatch(public_key_hex):
            raise WardenFormatError("public key must be 32 bytes of lowercase hex")
        if is_small_order_key(public_key_hex):
            raise WardenFormatError("weak public key (small-order point) refused")
        target_list = sorted(set(targets))
        if not target_list:
            raise WardenFormatError("an operator must run at least one target")
        for target in target_list:
            kind, _ = parse_target(target)
            if kind != "game":
                raise WardenFormatError(f"{target!r} is not a game target")
            holder = self._store.get(self._target_key(target))
            if isinstance(holder, str) and holder != operator_id and self._active(holder):
                raise WardenFormatError(f"{target!r} already has operator {holder!r}")
        self._store.set(
            self._op_key(operator_id),
            {"public_key": public_key_hex, "targets": target_list, "removed": False, "registered_by": warden_id},
        )
        for target in target_list:
            self._store.set(self._target_key(target), operator_id)

    def _active(self, operator_id: str) -> bool:
        record = self._store.get(self._op_key(operator_id))
        return isinstance(record, Mapping) and record.get("removed") is False

    def remove(self, operator_id: str, *, warden_id: str, registry: WardenRegistry) -> None:
        if not registry.is_warden(warden_id):
            raise WardenAuthorityError(f"{warden_id!r} is not a registered Warden")
        record = self._store.get(self._op_key(operator_id))
        if isinstance(record, Mapping):
            self._store.set(self._op_key(operator_id), {**record, "removed": True})

    def operator_for(self, target: str) -> tuple[str | None, str | None, str]:
        """Return ``(operator_id, public_key_hex, reason)`` for a target."""
        holder = self._store.get(self._target_key(target)) if isinstance(target, str) else None
        if not isinstance(holder, str):
            return None, None, "no_operator_registered"
        record = self._store.get(self._op_key(holder))
        if not isinstance(record, Mapping):
            return None, None, "no_operator_registered"
        if record.get("removed") is not False:
            return None, None, "operator_removed"
        if target not in (record.get("targets") or []):
            return None, None, "no_operator_registered"
        key = record.get("public_key")
        return (holder, key, "ok") if isinstance(key, str) else (None, None, "no_operator_registered")

    def revoke_attestation(self, attestation_id: str, *, warden_id: str, registry: WardenRegistry) -> None:
        if not registry.is_warden(warden_id):
            raise WardenAuthorityError(f"{warden_id!r} is not a registered Warden")
        self._store.set(self._revoked_key(_check_id(attestation_id, "attestation_id")), {"by": warden_id})

    def is_attestation_revoked(self, attestation_id: str) -> bool:
        return self._store.get(self._revoked_key(attestation_id)) is not None


def verify_attestation(
    attestation: OperatorAttestation | None,
    *,
    agent_id: str,
    target: str,
    now: float,
    operators: OperatorRegistry,
) -> tuple[bool, str]:
    """Check the operator's second signature. Returns ``(ok, reason)``."""
    if attestation is None:
        return False, "operator_attestation_missing"
    if not isinstance(attestation, OperatorAttestation):
        return False, "operator_attestation_malformed"
    operator_id, public_key, reason = operators.operator_for(target)
    if public_key is None:
        return False, reason
    if attestation.operator_id != operator_id:
        return False, "operator_mismatch"
    if attestation.target != target:
        return False, "operator_target_mismatch"
    if attestation.agent_id != agent_id:
        return False, "operator_agent_mismatch"
    if not ed25519_verify(public_key, attestation.signing_bytes(), attestation.signature):
        return False, "operator_bad_signature"
    if operators.is_attestation_revoked(attestation.attestation_id):
        return False, "operator_attestation_revoked"
    if now < attestation.issued_at - CLOCK_SKEW_S:
        return False, "operator_attestation_not_yet_valid"
    if now >= attestation.expires_at:
        return False, "operator_attestation_expired"
    return True, "ok"
