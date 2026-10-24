"""Append-only, tamper-evident audit log.

Every grant, bind, revoke and action decision lands here. Entries are chained:
each carries the MAC of the one before it, and every MAC is an HMAC under a
dedicated audit subkey rather than a bare hash. A bare hash chain can be
recomputed end to end by anyone who can write the store; an HMAC chain cannot be
re-forged without the secret, so editing or deleting a past entry is detectable.

What this does *not* give you, stated plainly:

* **Tail truncation.** If an attacker rolls back both the newest entries and the
  head pointer together, the shortened chain still verifies. Anchor
  :meth:`AuditLog.head` somewhere the store's writers cannot reach (a ledger, a
  signed message to the Warden) to close that gap.
* **Multi-writer safety.** A lock serializes appends within one process. Two
  processes appending to the same store could fork the chain; run one writer.

The gate treats a failed append as a denial: no audit record, no action.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping

from ._common import KVStore, MemoryStore, WardenError, hmac_sign, hmac_verify, load_secret

AUDIT_LABEL = "audit"
GENESIS = "0" * 64
_HEAD_KEY = "warden:audit:head"


def _entry_key(seq: int) -> str:
    return f"warden:audit:{seq:012d}"


class AuditLog:
    def __init__(
        self,
        store: KVStore | None = None,
        *,
        secret: str | bytes | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store: KVStore = store if store is not None else MemoryStore()
        self._secret = secret
        self._clock = clock
        self._lock = threading.Lock()

    def head(self) -> dict[str, Any]:
        """``{"seq", "mac"}`` of the newest entry. Anchor this externally."""
        head = self._store.get(_HEAD_KEY)
        return dict(head) if isinstance(head, Mapping) else {"seq": 0, "mac": GENESIS}

    def append(self, kind: str, actor: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Append one entry. Raises if it cannot be written (e.g. no secret)."""
        key = load_secret(self._secret)
        with self._lock:
            head = self.head()
            body = {
                "seq": int(head["seq"]) + 1,
                "ts": int(self._clock()),
                "kind": str(kind),
                "actor": str(actor),
                "details": dict(details or {}),
                "prev": head["mac"],
            }
            entry = {**body, "mac": hmac_sign(key, AUDIT_LABEL, body)}
            # Entry first, head second: a crash between the two leaves an orphan
            # entry (harmless, overwritten next append) never a head with no entry.
            self._store.set(_entry_key(entry["seq"]), entry)
            self._store.set(_HEAD_KEY, {"seq": entry["seq"], "mac": entry["mac"]})
            return entry

    def entries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for seq in range(1, int(self.head()["seq"]) + 1):
            record = self._store.get(_entry_key(seq))
            if isinstance(record, Mapping):
                out.append(dict(record))
        return out

    def verify_chain(self) -> tuple[bool, str]:
        """Walk the whole chain. Returns ``(ok, reason)``."""
        try:
            key = load_secret(self._secret)
        except WardenError:
            return False, "secret_unset"
        head = self.head()
        prev = GENESIS
        for seq in range(1, int(head["seq"]) + 1):
            record = self._store.get(_entry_key(seq))
            if not isinstance(record, Mapping):
                return False, f"missing_entry:{seq}"
            body = {k: v for k, v in record.items() if k != "mac"}
            if body.get("seq") != seq:
                return False, f"seq_mismatch:{seq}"
            if body.get("prev") != prev:
                return False, f"chain_break:{seq}"
            if not hmac_verify(key, AUDIT_LABEL, body, record.get("mac")):
                return False, f"bad_mac:{seq}"
            prev = record["mac"]
        if prev != head["mac"]:
            return False, "head_mismatch"
        return True, "ok"
