"""Shared primitives for the Warden package.

Canonical JSON, domain-separated HMAC signing, the key/value store protocol, and
the error types. Nothing here imports the rest of QUIPU at module load, so the
whole ``warden`` package can be lifted into another repo (the plan expects the
World 4 / Warden code to live outside QUIPU) by copying the directory and
swapping :class:`BrainKVStore` for a store that suits the host repo.

Design rule threaded through every module: the Warden's authorization is a
*record the code checks*, never something the code grants itself, and every
check fails closed. A missing secret, an unreadable store, a malformed record or
an unexpected exception all end in a denial, never in a default-allow.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from typing import Any, Mapping, Protocol

SECRET_ENV_PRIMARY = "WARDEN_GRANT_SECRET"
SECRET_ENV_FALLBACK = "SCBRAIN_GRID_SECRET"


class WardenError(Exception):
    """Base class for every error raised by the Warden package."""


class WardenSecretError(WardenError):
    """No signing secret is configured. There is deliberately no default.

    An earlier QUIPU component (the GARD Lite GUI, see ``_version.py`` 0.28.0)
    advertised authentication while keying its HMAC with a public constant, so it
    authenticated nothing. Refusing to run without a real secret is what makes the
    protection real.
    """


class WardenAuthorityError(WardenError):
    """The caller is not a registered Warden, so the operation is refused."""


class WardenFormatError(WardenError):
    """A record, target or scope is malformed."""


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, no NaN/Infinity."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_secret(secret: str | bytes | None = None) -> bytes:
    """Return the signing secret or raise :class:`WardenSecretError`.

    Precedence: explicit argument, ``WARDEN_GRANT_SECRET``, then
    ``SCBRAIN_GRID_SECRET`` (the grid secret QUIPU already uses for signed jobs).
    Whitespace-only counts as unset.
    """
    value: str | bytes | None = secret
    if value is None:
        value = os.environ.get(SECRET_ENV_PRIMARY) or os.environ.get(SECRET_ENV_FALLBACK)
    if isinstance(value, str):
        value = value.encode("utf-8")
    if not value or not bytes(value).strip():
        raise WardenSecretError(
            f"{SECRET_ENV_PRIMARY} (or {SECRET_ENV_FALLBACK}) must be set; "
            "there is no default"
        )
    return bytes(value)


def derive_key(secret: bytes, label: str) -> bytes:
    """Per-purpose subkey, so one configured secret never signs two protocols.

    Reusing ``SCBRAIN_GRID_SECRET`` directly would let a signature minted for one
    purpose (say, a GARD grid job) be replayed as another (a Warden grant). The
    label binds each signature to the purpose it was made for.
    """
    return hmac.new(secret, b"warden-key/v1\x00" + label.encode("utf-8"), hashlib.sha256).digest()


def hmac_sign(secret: bytes, label: str, payload: Mapping[str, Any]) -> str:
    key = derive_key(secret, label)
    return hmac.new(key, label.encode("utf-8") + b"\n" + canonical_json(dict(payload)), hashlib.sha256).hexdigest()


def hmac_verify(secret: bytes, label: str, payload: Mapping[str, Any], signature: Any) -> bool:
    """Constant-time check. Non-string or non-ASCII signatures are simply invalid."""
    if not isinstance(signature, str):
        return False
    try:
        expected = hmac_sign(secret, label, payload).encode("ascii")
        return hmac.compare_digest(expected, signature.encode("ascii"))
    except (UnicodeEncodeError, TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Key/value store
# --------------------------------------------------------------------------- #

class KVStore(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any) -> None: ...


class MemoryStore:
    """In-process store. The default for tests and for simulation runs."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            value = self._data.get(key)
            return json.loads(json.dumps(value)) if value is not None else None

    def set(self, key: str, value: Any) -> None:
        # Round-trip through JSON so a caller can never mutate stored state by
        # holding a reference, and so non-JSON values fail here rather than later.
        encoded = json.dumps(value)
        with self._lock:
            self._data[key] = json.loads(encoded)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class BrainKVStore:
    """Adapter over QUIPU's ``brain_kv`` (``kv_get_json`` / ``kv_set_json``).

    Imported lazily so the package stays portable. Pass this as the store in
    production inside QUIPU; the database location follows ``SCB_DB_PATH``.
    """

    def get(self, key: str) -> Any | None:
        from .. import brain_kv
        return brain_kv.kv_get_json(key, None)

    def set(self, key: str, value: Any) -> None:
        from .. import brain_kv
        brain_kv.kv_set_json(key, value)

    def delete(self, key: str) -> None:
        from .. import brain_kv
        brain_kv.kv_delete(key)
