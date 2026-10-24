"""Warden grants: signed, expiring, revocable authorization records.

A *grant* is a Warden's statement that one agent may take a bounded set of action
classes against one target for a limited time. The code never mints authority for
itself: it verifies a record a registered Warden signed, and denies if anything
about that record is off.

Why revocation is a separate record
-----------------------------------
The signature covers the grant's content, and content that can change (a
``revoked`` flag) cannot be covered by a signature made in advance. So revocation
is its own signed record keyed by ``grant_id`` in the :class:`GrantStore`.
:func:`verify_grant` consults it on every call, and the presence of *any*
revocation record counts as revoked: a forged record can only deny, which is the
safe direction.

Targets
-------
``reachy2:<host>``, ``npc:<id>``, ``game:guild_wars``,
``game:wow_legacy:<server>``. Each target kind accepts only its own action
classes, so a robot grant can never be stretched into a game action.
"""
from __future__ import annotations

import re
import secrets as _secrets
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Protocol

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

GRANT_LABEL = "grant"
REVOCATION_LABEL = "revocation"

# Action classes per target kind. Exact match only: there are no wildcards, so a
# scope can be read off and audited by eye.
ACTION_CLASSES: dict[str, frozenset[str]] = {
    "robot": frozenset({"robot.bind", "robot.observe", "robot.move", "robot.gesture", "robot.speak"}),
    "npc": frozenset({"npc.bind", "npc.act"}),
    "game": frozenset({"game.connect", "game.act", "game.chat"}),
}
ALL_ACTION_CLASSES: frozenset[str] = frozenset().union(*ACTION_CLASSES.values())

MAX_GRANT_TTL_S = 7 * 24 * 3600
CLOCK_SKEW_S = 30

_ID_RE = re.compile(r"[A-Za-z0-9._:@-]{1,128}")
_TARGET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("robot", re.compile(r"reachy2:([A-Za-z0-9._-]{1,253})")),
    ("npc", re.compile(r"npc:([A-Za-z0-9._-]{1,128})")),
    ("game", re.compile(r"game:(guild_wars)")),
    ("game", re.compile(r"game:(wow_legacy:[A-Za-z0-9._-]{1,128})")),
)


def parse_target(target: Any) -> tuple[str, str]:
    """Return ``(kind, ident)`` for a valid target string, else raise."""
    if isinstance(target, str):
        for kind, pattern in _TARGET_PATTERNS:
            match = pattern.fullmatch(target)
            if match:
                return kind, match.group(1)
    raise WardenFormatError(f"invalid target: {target!r}")


def _check_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise WardenFormatError(f"invalid {field}: {value!r}")
    return value


def _check_epoch(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
        raise WardenFormatError(f"invalid {field}: {value!r}")
    return int(value)


def _check_scope(scope: Iterable[Any], kind: str) -> tuple[str, ...]:
    if isinstance(scope, (str, bytes)):
        raise WardenFormatError("scope must be a list of action classes")
    cleaned = sorted({str(s) for s in scope})
    if not cleaned:
        raise WardenFormatError("scope must not be empty")
    for item in cleaned:
        if item not in ALL_ACTION_CLASSES:
            raise WardenFormatError(f"unknown action class: {item!r}")
        if item not in ACTION_CLASSES[kind]:
            raise WardenFormatError(f"action class {item!r} does not apply to a {kind} target")
    return tuple(cleaned)


class WardenRegistry(Protocol):
    """Who currently holds the Warden role. Lives in the host repo (open item 1)."""

    def is_warden(self, warden_id: str) -> bool: ...


class StaticWardenRegistry:
    """A fixed set of Warden ids. An empty set means nobody can issue anything."""

    def __init__(self, warden_ids: Iterable[str] = ()) -> None:
        self._ids = frozenset(str(w) for w in warden_ids)

    @classmethod
    def from_env(cls, name: str = "WARDEN_IDS") -> "StaticWardenRegistry":
        import os
        raw = os.environ.get(name, "")
        return cls(part.strip() for part in raw.split(",") if part.strip())

    def is_warden(self, warden_id: str) -> bool:
        return isinstance(warden_id, str) and warden_id in self._ids


@dataclass(frozen=True)
class Grant:
    grant_id: str
    warden_id: str
    agent_id: str
    target: str
    scope: tuple[str, ...]
    issued_at: int
    expires_at: int
    signature: str = ""

    def payload(self) -> dict[str, Any]:
        """The exact content the signature covers (everything but the signature)."""
        return {
            "grant_id": self.grant_id,
            "warden_id": self.warden_id,
            "agent_id": self.agent_id,
            "target": self.target,
            "scope": sorted(self.scope),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    def to_dict(self) -> dict[str, Any]:
        out = self.payload()
        out["signature"] = self.signature
        return out

    @classmethod
    def from_dict(cls, data: Any) -> "Grant":
        if not isinstance(data, Mapping):
            raise WardenFormatError("grant must be an object")
        try:
            kind, _ = parse_target(data.get("target"))
            scope = data.get("scope")
            if not isinstance(scope, (list, tuple)):
                raise WardenFormatError("scope must be a list")
            return cls(
                grant_id=_check_id(data.get("grant_id"), "grant_id"),
                warden_id=_check_id(data.get("warden_id"), "warden_id"),
                agent_id=_check_id(data.get("agent_id"), "agent_id"),
                target=data["target"],
                scope=_check_scope(scope, kind),
                issued_at=_check_epoch(data.get("issued_at"), "issued_at"),
                expires_at=_check_epoch(data.get("expires_at"), "expires_at"),
                signature=data.get("signature") if isinstance(data.get("signature"), str) else "",
            )
        except WardenFormatError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise WardenFormatError(f"malformed grant: {exc}") from exc


def issue_grant(
    *,
    warden_id: str,
    agent_id: str,
    target: str,
    scope: Iterable[str],
    ttl_s: float,
    registry: WardenRegistry,
    secret: str | bytes | None = None,
    now: float | None = None,
    max_ttl_s: int = MAX_GRANT_TTL_S,
    grant_id: str | None = None,
) -> Grant:
    """Mint a signed grant. Refuses unless ``warden_id`` currently holds the role.

    ``registry`` is a required keyword with no default on purpose: there is no way
    to call this "just to get a grant" without naming who vouches for it.
    """
    _check_id(warden_id, "warden_id")
    if not registry.is_warden(warden_id):
        raise WardenAuthorityError(f"{warden_id!r} is not a registered Warden")
    kind, _ = parse_target(target)
    _check_id(agent_id, "agent_id")
    cleaned_scope = _check_scope(scope, kind)
    if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)) or not (0 < ttl_s <= max_ttl_s):
        raise WardenFormatError(f"ttl_s must be in (0, {max_ttl_s}]")
    key = load_secret(secret)
    issued = int(time.time() if now is None else now)
    grant = Grant(
        grant_id=_check_id(grant_id or _secrets.token_hex(16), "grant_id"),
        warden_id=warden_id,
        agent_id=agent_id,
        target=target,
        scope=cleaned_scope,
        issued_at=issued,
        expires_at=issued + int(ttl_s),
    )
    return replace(grant, signature=hmac_sign(key, GRANT_LABEL, grant.payload()))


class GrantStore:
    """Persists grants and revocation records; notifies listeners on revoke."""

    def __init__(self, store: KVStore | None = None) -> None:
        self._store: KVStore = store if store is not None else MemoryStore()
        self._listeners: list[Callable[[str], None]] = []

    @staticmethod
    def _grant_key(grant_id: str) -> str:
        return f"warden:grant:{grant_id}"

    @staticmethod
    def _revocation_key(grant_id: str) -> str:
        return f"warden:revocation:{grant_id}"

    def save(self, grant: Grant) -> None:
        self._store.set(self._grant_key(grant.grant_id), grant.to_dict())

    def get(self, grant_id: str) -> Grant | None:
        """Return the stored grant, or ``None`` if absent *or malformed*."""
        try:
            if not isinstance(grant_id, str) or not _ID_RE.fullmatch(grant_id):
                return None
            raw = self._store.get(self._grant_key(grant_id))
            return Grant.from_dict(raw) if raw is not None else None
        except WardenFormatError:
            return None

    def is_revoked(self, grant_id: str) -> bool:
        return self._store.get(self._revocation_key(grant_id)) is not None

    def get_revocation(self, grant_id: str) -> dict[str, Any] | None:
        record = self._store.get(self._revocation_key(grant_id))
        return dict(record) if isinstance(record, Mapping) else None

    def add_revoke_listener(self, listener: Callable[[str], None]) -> None:
        """Register a callback run immediately after a revocation is recorded.

        This is the in-process fast path (the NPC e-stop hooks in here). It is not
        the safety guarantee: every action re-checks the grant, so a revocation made
        in another process still stops the next action.
        """
        self._listeners.append(listener)

    def revoke(
        self,
        grant_id: str,
        *,
        warden_id: str,
        registry: WardenRegistry,
        reason: str = "",
        secret: str | bytes | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        _check_id(grant_id, "grant_id")
        _check_id(warden_id, "warden_id")
        if not registry.is_warden(warden_id):
            raise WardenAuthorityError(f"{warden_id!r} is not a registered Warden")
        key = load_secret(secret)
        body = {
            "grant_id": grant_id,
            "revoked_by": warden_id,
            "revoked_at": int(time.time() if now is None else now),
            "reason": str(reason)[:256],
        }
        record = {**body, "signature": hmac_sign(key, REVOCATION_LABEL, body)}
        # Write the record first. Listeners may fail; the revocation must not.
        self._store.set(self._revocation_key(grant_id), record)
        errors = 0
        for listener in list(self._listeners):
            try:
                listener(grant_id)
            except Exception:  # noqa: BLE001 - a broken listener must not undo a revoke
                errors += 1
        return {**record, "listener_errors": errors}


def verify_grant(
    grant: Grant,
    *,
    agent_id: str,
    target: str,
    action: str,
    now: float,
    store: GrantStore,
    registry: WardenRegistry,
    secret: str | bytes | None = None,
) -> tuple[bool, str]:
    """Check a grant for one proposed action. Returns ``(ok, reason)``.

    ``reason`` is ``"ok"`` or a stable code. Checks run cheapest-and-most-
    fundamental first, and the signature is checked before any field is trusted.
    """
    try:
        key = load_secret(secret)
    except WardenSecretError:
        return False, "secret_unset"
    if not isinstance(grant, Grant):
        return False, "malformed"
    if not hmac_verify(key, GRANT_LABEL, grant.payload(), grant.signature):
        return False, "bad_signature"
    try:
        grant_kind, _ = parse_target(grant.target)
        _check_scope(grant.scope, grant_kind)
    except WardenFormatError:
        return False, "malformed"
    if not registry.is_warden(grant.warden_id):
        return False, "warden_not_authorized"
    if store.is_revoked(grant.grant_id):
        return False, "revoked"
    if now < grant.issued_at - CLOCK_SKEW_S:
        return False, "not_yet_valid"
    if now >= grant.expires_at:
        return False, "expired"
    if grant.agent_id != agent_id:
        return False, "agent_mismatch"
    if grant.target != target:
        return False, "target_mismatch"
    if action not in ALL_ACTION_CLASSES:
        return False, "unknown_action"
    if action not in grant.scope:
        return False, "action_not_in_scope"
    return True, "ok"
