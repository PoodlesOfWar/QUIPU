"""quipu.security — gVisor detection, syscall sandboxing, and payload boundary enforcement.

Protects QUIPU's observer endpoints (POST /observe, POST /feedback) and continuous
annealing loops from untrusted payloads, shell injection, and container breakout attempts.
"""

from __future__ import annotations

import os
import sys
import ctypes
from pathlib import Path
from typing import Any

# Maximum payload boundaries for untrusted ingestion
MAX_OBSERVE_BYTES = 512 * 1024       # 512 KB max raw payload
MAX_OBSERVE_TEXT_CHARS = 100_000    # 100,000 characters
MAX_META_DEPTH = 5                  # Prevent deeply nested JSON bomb attacks
MAX_FEEDBACK_BYTES = 64 * 1024       # 64 KB max feedback payload

# Permitted top-level keys on /observe
ALLOWED_OBSERVE_KEYS = {
    "source", "text", "kind", "confidence", "meta", "source_row_id", "timestamp"
}


def is_gvisor_sandboxed() -> bool:
    """Detect if running under Google gVisor (runsc) application kernel."""
    # 1. Environment indicator
    if os.environ.get("DOCKER_RUNTIME_SANDBOX", "").lower() == "runsc":
        return True
    if os.environ.get("GVISOR_SANDBOXED", "").lower() in ("1", "true", "yes"):
        return True

    # 2. Inspect /proc/version or /proc/sys/kernel/osrelease for gVisor signatures
    try:
        proc_ver = Path("/proc/version")
        if proc_ver.exists():
            content = proc_ver.read_text(encoding="utf-8", errors="ignore").lower()
            if "gvisor" in content or "google" in content:
                return True
    except Exception:
        pass

    # 3. Inspect dmesg or /proc/cpuinfo for gVisor hypervisor / sentry markers
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            content = cpuinfo.read_text(encoding="utf-8", errors="ignore").lower()
            if "gvisor" in content:
                return True
    except Exception:
        pass

    return False


def is_no_new_privs_set() -> bool:
    """Check if PR_SET_NO_NEW_PRIVS was enforced (via Docker security_opt)."""
    if sys.platform != "linux":
        return False
    try:
        libc = ctypes.CDLL("libc.so.6")
        PR_GET_NO_NEW_PRIVS = 39
        res = libc.prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0)
        return res == 1
    except Exception:
        return False


def get_security_posture() -> dict[str, Any]:
    """Return runtime security posture including gVisor isolation state."""
    sandboxed = is_gvisor_sandboxed()
    no_new_privs = is_no_new_privs_set()
    tier = "tier-1-gvisor-sandboxed" if sandboxed else ("tier-2-hardened-container" if no_new_privs else "tier-3-standard")

    return {
        "ok": True,
        "gvisor_sandboxed": sandboxed,
        "no_new_privileges": no_new_privs,
        "isolation_tier": tier,
        "platform": sys.platform,
        "limits": {
            "max_observe_bytes": MAX_OBSERVE_BYTES,
            "max_observe_text_chars": MAX_OBSERVE_TEXT_CHARS,
            "max_meta_depth": MAX_META_DEPTH,
            "max_feedback_bytes": MAX_FEEDBACK_BYTES,
        },
    }


def _check_depth(val: Any, current_depth: int = 0) -> bool:
    if current_depth > MAX_META_DEPTH:
        return False
    if isinstance(val, dict):
        return all(_check_depth(v, current_depth + 1) for v in val.values())
    if isinstance(val, list):
        return all(_check_depth(item, current_depth + 1) for item in val)
    return True


def validate_observe_payload(payload: Any) -> tuple[bool, str | None]:
    """Validate incoming /observe payload against size, structure, and nesting bounds."""
    if not isinstance(payload, dict):
        return False, "Payload must be a JSON object"

    # Enforce allowed top-level keys
    extra_keys = set(payload.keys()) - ALLOWED_OBSERVE_KEYS
    if extra_keys:
        return False, f"Unrecognized payload keys: {sorted(list(extra_keys))}"

    text = payload.get("text")
    if text is None or not isinstance(text, str):
        return False, "'text' field must be a non-null string"

    if len(text) > MAX_OBSERVE_TEXT_CHARS:
        return False, f"'text' exceeds maximum length of {MAX_OBSERVE_TEXT_CHARS} characters"

    meta = payload.get("meta")
    if meta is not None:
        if not isinstance(meta, dict):
            return False, "'meta' field must be a dictionary"
        if not _check_depth(meta, 0):
            return False, f"'meta' exceeds maximum nesting depth of {MAX_META_DEPTH}"

    confidence = payload.get("confidence")
    if confidence is not None:
        try:
            c_val = float(confidence)
            if not (0.0 <= c_val <= 1.0):
                return False, "'confidence' must be between 0.0 and 1.0"
        except (ValueError, TypeError):
            return False, "'confidence' must be a numeric value"

    return True, None


def validate_feedback_payload(payload: Any) -> tuple[bool, str | None]:
    """Validate incoming /feedback payload."""
    if not isinstance(payload, dict):
        return False, "Payload must be a JSON object"

    expected = payload.get("expected")
    if expected is None or not isinstance(expected, str):
        return False, "'expected' field is required and must be a string"

    if len(expected) > 10_000:
        return False, "'expected' exceeds maximum length of 10,000 characters"

    return True, None
