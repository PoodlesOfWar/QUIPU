"""Agent identity: every action request is signed by the agent that makes it.

Each agent holds its own Ed25519 private key; the gate holds only the matching
*public* key. That asymmetry is the point. With a shared-secret HMAC the gate
would have to store every agent's secret, so it could forge any agent's requests
and a signature would prove nothing about who acted. With Ed25519 the gate can
verify a request but cannot make one.

Automation is declared, never concealed
---------------------------------------
Every request carries ``automated: true`` and a fixed declaration string *inside
the signed payload*. A request that lacks the declaration is denied, and the
declaration cannot be removed without invalidating the signature.
:meth:`SignedRequest.disclosure` returns the exact record a game adapter must
forward to the server or its operator. Nothing in this package hides that an
action is automated, and nothing should be added that does.
"""
from __future__ import annotations

import re
import secrets as _secrets
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping

from ._common import (
    KVStore,
    MemoryStore,
    WardenAuthorityError,
    WardenError,
    WardenFormatError,
    canonical_json,
    sha256_hex,
)
from .warden_grant import WardenRegistry, _check_id

REQUEST_LABEL = b"warden-request/v1\n"
AUTOMATED_DECLARATION = "automated-agent"
MAX_REQUEST_SKEW_S = 30
_NONCE_RE = re.compile(r"[A-Za-z0-9_-]{16,128}")
_HEX_RE = re.compile(r"[0-9a-f]+")


def _ed25519():
    """Import Ed25519 lazily and fail with a clear error if unavailable."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except Exception as exc:  # noqa: BLE001
        raise WardenError("the `cryptography` package is required for agent signatures") from exc
    return Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature


# Small-order Ed25519 points. A public key that is one of these is "weak": OpenSSL
# (and so ``cryptography``) does not reject them, and for such a key a signature can
# verify with no private key at all. Seen in practice: the all-zero key with an
# all-zero signature verifies for roughly one message in four. Registration is
# Warden-gated, but a Warden pasting ``00...00`` must not silently mint a forgeable
# identity, and the verifier refuses them regardless.
#
# These are the y-coordinates of the 8-element torsion subgroup (5 distinct values)
# plus the two non-canonical encodings of y=0 and y=1 (p and p+1). The test suite
# re-derives this set from curve arithmetic rather than trusting these constants.
_P25519 = 2**255 - 19
SMALL_ORDER_Y = frozenset({
    0,
    1,
    _P25519 - 1,
    _P25519,
    _P25519 + 1,
    0x05FC536D880238B13933C6D305ACDFD5F098EFF289F4C345B027B2C28F95E826,
    0x7A03AC9277FDC74EC6CC392CFA53202A0F67100D760B3CBA4FD84D3D706A17C7,
})


def is_small_order_key(public_key_hex: Any) -> bool:
    """True for malformed keys and for any small-order point (both unusable)."""
    if not isinstance(public_key_hex, str) or len(public_key_hex) != 64 or not _HEX_RE.fullmatch(public_key_hex):
        return True
    y = int.from_bytes(bytes.fromhex(public_key_hex), "little") & ((1 << 255) - 1)
    return y in SMALL_ORDER_Y


def ed25519_verify(public_key_hex: Any, message: bytes, signature_hex: Any) -> bool:
    """Verify an Ed25519 signature; any malformed input is simply ``False``."""
    if not isinstance(public_key_hex, str) or not isinstance(signature_hex, str):
        return False
    if not _HEX_RE.fullmatch(public_key_hex) or not _HEX_RE.fullmatch(signature_hex):
        return False
    if is_small_order_key(public_key_hex):
        return False
    _, public_cls, invalid_signature = _ed25519()
    try:
        public_cls.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(signature_hex), message
        )
        return True
    except (invalid_signature, ValueError):
        return False


@dataclass(frozen=True)
class SignedRequest:
    agent_id: str
    action: str
    target: str
    grant_id: str
    params: Mapping[str, Any]
    nonce: str
    issued_at: int
    automated: bool = True
    declaration: str = AUTOMATED_DECLARATION
    signature: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "action": self.action,
            "target": self.target,
            "grant_id": self.grant_id,
            "params": dict(self.params),
            "nonce": self.nonce,
            "issued_at": self.issued_at,
            "automated": self.automated,
            "declaration": self.declaration,
        }

    def signing_bytes(self) -> bytes:
        return REQUEST_LABEL + canonical_json(self.payload())

    def params_digest(self) -> str:
        return sha256_hex(canonical_json(dict(self.params)))

    def disclosure(self) -> dict[str, Any]:
        """What an adapter must forward to a game server or its operator."""
        return {
            "automated": True,
            "declaration": self.declaration,
            "agent_id": self.agent_id,
            "grant_id": self.grant_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, data: Any) -> "SignedRequest":
        if not isinstance(data, Mapping):
            raise WardenFormatError("request must be an object")
        params = data.get("params", {})
        if not isinstance(params, Mapping):
            raise WardenFormatError("params must be an object")
        if not isinstance(data.get("issued_at"), (int, float)) or isinstance(data.get("issued_at"), bool):
            raise WardenFormatError("invalid issued_at")
        return cls(
            agent_id=_check_id(data.get("agent_id"), "agent_id"),
            action=str(data.get("action", "")),
            target=str(data.get("target", "")),
            grant_id=_check_id(data.get("grant_id"), "grant_id"),
            params=dict(params),
            nonce=str(data.get("nonce", "")),
            issued_at=int(data["issued_at"]),
            automated=data.get("automated") is True,
            declaration=str(data.get("declaration", "")),
            signature=data.get("signature") if isinstance(data.get("signature"), str) else "",
        )


class AgentIdentity:
    """An agent's identity and private key. Held by the agent, never by the gate."""

    def __init__(self, agent_id: str, private_key: Any) -> None:
        self.agent_id = _check_id(agent_id, "agent_id")
        self._private_key = private_key

    @classmethod
    def generate(cls, agent_id: str) -> "AgentIdentity":
        private_cls, _, _ = _ed25519()
        return cls(agent_id, private_cls.generate())

    @classmethod
    def from_private_bytes(cls, agent_id: str, raw: bytes) -> "AgentIdentity":
        private_cls, _, _ = _ed25519()
        return cls(agent_id, private_cls.from_private_bytes(raw))

    @property
    def public_key_hex(self) -> str:
        from cryptography.hazmat.primitives import serialization
        return self._private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        ).hex()

    def sign_request(
        self,
        *,
        action: str,
        target: str,
        grant_id: str,
        params: Mapping[str, Any] | None = None,
        now: float | None = None,
        nonce: str | None = None,
    ) -> SignedRequest:
        unsigned = SignedRequest(
            agent_id=self.agent_id,
            action=action,
            target=target,
            grant_id=grant_id,
            params=dict(params or {}),
            nonce=nonce or _secrets.token_urlsafe(18),
            issued_at=int(time.time() if now is None else now),
        )
        signature = self._private_key.sign(unsigned.signing_bytes()).hex()
        return replace(unsigned, signature=signature)


class AgentRegistry:
    """Maps agent ids to public keys. Registration needs a Warden.

    Registration is gated because it is an authority question: if anyone could
    register a key for ``agent_id="x"``, they could then sign as ``x`` and spend any
    grant issued to ``x``. An existing key is never silently overwritten; a
    rotation must say so.
    """

    def __init__(self, store: KVStore | None = None) -> None:
        self._store: KVStore = store if store is not None else MemoryStore()

    @staticmethod
    def _key(agent_id: str) -> str:
        return f"warden:agent:{agent_id}"

    def register(
        self,
        agent_id: str,
        public_key_hex: str,
        *,
        warden_id: str,
        registry: WardenRegistry,
        rotate: bool = False,
    ) -> None:
        _check_id(agent_id, "agent_id")
        if not registry.is_warden(warden_id):
            raise WardenAuthorityError(f"{warden_id!r} is not a registered Warden")
        if not isinstance(public_key_hex, str) or len(public_key_hex) != 64 or not _HEX_RE.fullmatch(public_key_hex):
            raise WardenFormatError("public key must be 32 bytes of lowercase hex")
        if is_small_order_key(public_key_hex):
            raise WardenFormatError("weak public key (small-order point) refused")
        if self._store.get(self._key(agent_id)) is not None and not rotate:
            raise WardenFormatError(f"agent {agent_id!r} already registered; pass rotate=True to replace")
        self._store.set(
            self._key(agent_id),
            {"public_key": public_key_hex, "registered_by": warden_id, "revoked": False},
        )

    def revoke(self, agent_id: str, *, warden_id: str, registry: WardenRegistry) -> None:
        if not registry.is_warden(warden_id):
            raise WardenAuthorityError(f"{warden_id!r} is not a registered Warden")
        record = self._store.get(self._key(agent_id))
        if record is not None:
            self._store.set(self._key(agent_id), {**record, "revoked": True})

    def lookup(self, agent_id: str) -> tuple[str | None, str]:
        """Return ``(public_key_hex, reason)``; key is ``None`` unless usable."""
        record = self._store.get(self._key(agent_id)) if isinstance(agent_id, str) else None
        if not isinstance(record, Mapping):
            return None, "unknown_agent"
        if record.get("revoked") is not False:
            return None, "agent_revoked"
        key = record.get("public_key")
        if not isinstance(key, str):
            return None, "unknown_agent"
        return key, "ok"


class NonceCache:
    """Remembers recent nonces so a captured request cannot be replayed.

    In-process and bounded. If it ever fills with live entries it refuses new
    requests rather than evicting a live nonce, because evicting would reopen the
    replay window. A multi-process deployment needs a shared cache; the interface
    is two methods so one can be dropped in.
    """

    def __init__(self, max_entries: int = 100_000) -> None:
        self._seen: dict[str, float] = {}
        self._max = max_entries
        self._lock = threading.Lock()

    def check_and_add(self, key: str, expires_at: float, now: float) -> str:
        """Return ``"ok"``, ``"replay"`` or ``"nonce_cache_full"``."""
        with self._lock:
            if len(self._seen) >= self._max // 2:
                for stale in [k for k, exp in self._seen.items() if exp <= now]:
                    del self._seen[stale]
            if key in self._seen and self._seen[key] > now:
                return "replay"
            if len(self._seen) >= self._max:
                return "nonce_cache_full"
            self._seen[key] = expires_at
            return "ok"


def verify_request(
    request: SignedRequest,
    *,
    agents: AgentRegistry,
    nonces: NonceCache,
    now: float,
    max_skew_s: int = MAX_REQUEST_SKEW_S,
) -> tuple[bool, str]:
    """Authenticate one request. Returns ``(ok, reason)``.

    The nonce is consumed only *after* the signature checks out, so an
    unauthenticated sender cannot burn nonces to lock a real agent out.
    """
    if not isinstance(request, SignedRequest):
        return False, "malformed"
    if request.automated is not True or request.declaration != AUTOMATED_DECLARATION:
        return False, "not_declared_automated"
    public_key, reason = agents.lookup(request.agent_id)
    if public_key is None:
        return False, reason
    if not _NONCE_RE.fullmatch(request.nonce or ""):
        return False, "malformed"
    try:
        message = request.signing_bytes()
    except (ValueError, TypeError):
        # e.g. NaN/Infinity or a non-JSON value in params: canonical JSON refuses
        # them, so such a request can never have been validly signed.
        return False, "malformed"
    if not ed25519_verify(public_key, message, request.signature):
        return False, "bad_signature"
    if abs(now - request.issued_at) > max_skew_s:
        return False, "stale_request"
    status = nonces.check_and_add(
        f"{request.agent_id}:{request.nonce}", request.issued_at + 2 * max_skew_s, now
    )
    if status != "ok":
        return False, status
    return True, "ok"
