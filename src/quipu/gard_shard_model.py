"""GARD Shard model: authenticated encrypted JSON shards for the SCB mesh.

``gard-shard/v1`` is deliberately small and explicit:

* canonical UTF-8 JSON is compressed with zlib before it is partitioned;
* every partition derives independent AES/HMAC keys with HKDF-SHA256;
* AES-256-CBC provides confidentiality and HMAC-SHA256 provides
  Encrypt-then-MAC authentication;
* authentication is verified in constant time before any decrypt or
  decompression operation when ``verify_digest`` is enabled (the default).

The module never writes an external grid secret, a derived key, or a plaintext
payload to a manifest.  The public Loadopoly manifest contains only protocol
metadata and the non-secret MESH model identity.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import struct
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


# ---------------------------------------------------------------------------
# Protocol versions
#
# v2 (current) uses AES-256-GCM: one AEAD primitive providing confidentiality
# and integrity together, replacing v1's AES-256-CBC + PKCS#7 + separate
# HMAC-SHA256 encrypt-then-MAC construction. GCM removes the padding oracle
# surface CBC carries, halves the per-shard authentication overhead (a 16-byte
# tag versus a 32-byte HMAC), and drops the second HKDF-derived key.
#
# v1 remains fully readable. Its protocol string is bound into both the HKDF
# info and the MAC preimage, so it is threaded explicitly through the legacy
# code path rather than read from the module-level constant -- bumping the
# constant alone would silently change v1 key derivation and break every
# existing envelope.
# ---------------------------------------------------------------------------
GARD_PROTOCOL = "gard-shard/v2"
GARD_PROTOCOL_VERSION = 2
GARD_ENVELOPE_SCHEMA = "gard-shard-envelope/v2"

GARD_PROTOCOL_V1 = "gard-shard/v1"
GARD_PROTOCOL_VERSION_V1 = 1
GARD_ENVELOPE_SCHEMA_V1 = "gard-shard-envelope/v1"

GARD_MODEL_CONTEXT_SCHEMA = "gard-model-context/v1"
GARD_HUB_MANIFEST_SCHEMA = "loadopoly-gard-shard-status/v1"
GARD_ASSIGNMENT_PROOF_KIND = "catalog-assignment-proof/v1"
GARD_SHARD_DOMAIN = "SiCi_SQRT(-1)"
GRID_SECRET_ENV = "SCBRAIN_GRID_SECRET"
VERIFY_DIGEST_ENV = "SCBRAIN_GARD_VERIFY_DIGEST"

_AES_KEY_BYTES = 32
_HMAC_KEY_BYTES = 32
_HKDF_BYTES = _AES_KEY_BYTES + _HMAC_KEY_BYTES
_SALT_BYTES = 32
_IV_BYTES = 16
_HMAC_BYTES = 32
# AES-GCM: 96-bit nonce is the NIST SP 800-38D recommended size (the only one
# used directly without extra derivation), 128-bit tag is the full-strength tag.
_GCM_NONCE_BYTES = 12
_GCM_TAG_BYTES = 16
_MAX_SHARDS = 256
_MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_PLAINTEXT_BYTES = 32 * 1024 * 1024
_PIPELINE_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _PIPELINE_ROOT.parent


class GardShardError(ValueError):
    """Base error for malformed or unusable GARD shard material."""


class GardShardSecretError(GardShardError):
    """Raised when a GARD operation needs an unavailable grid secret."""


class GardShardVerificationError(GardShardError):
    """Raised when an authenticated GARD envelope or proof is invalid."""


@dataclass(frozen=True)
class GardShardConfig:
    """Non-secret protocol settings for a GARD shard operation.

    ``verify_digest`` is safe by default.  It may only be disabled through an
    explicit constructor argument or a deliberately false
    ``SCBRAIN_GARD_VERIFY_DIGEST`` environment value at the calling layer.
    The external grid secret itself is always required for encryption and
    decryption, irrespective of this verification switch.
    """

    shard_count: int = 1
    compression_level: int = 6
    verify_digest: bool = True
    secret_env: str = GRID_SECRET_ENV
    domain: str = GARD_SHARD_DOMAIN
    max_compressed_bytes: int = _MAX_COMPRESSED_BYTES
    max_plaintext_bytes: int = _MAX_PLAINTEXT_BYTES

    def __post_init__(self) -> None:
        if not 1 <= int(self.shard_count) <= _MAX_SHARDS:
            raise GardShardError(f"shard_count must be in [1, {_MAX_SHARDS}]")
        if not -1 <= int(self.compression_level) <= 9:
            raise GardShardError("compression_level must be in [-1, 9]")
        if not str(self.secret_env).strip():
            raise GardShardError("secret_env must name an environment variable")
        if not str(self.domain).strip():
            raise GardShardError("domain must be non-empty")
        if int(self.max_compressed_bytes) <= 0 or int(self.max_plaintext_bytes) <= 0:
            raise GardShardError("maximum payload sizes must be positive")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config(config: GardShardConfig | None = None) -> GardShardConfig:
    return config if config is not None else GardShardConfig()


def verification_enabled_from_env(default: bool = True) -> bool:
    """Return the explicit GARD verification policy from the environment.

    Unset means verification remains enabled.  Only the conventional explicit
    false values opt out, which prevents a missing secret from silently
    degrading a mesh-sharing operation.
    """

    raw = os.environ.get(VERIFY_DIGEST_ENV)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically for digests, KDF context, and proofs."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise GardShardError(f"value is not canonical JSON serializable: {exc}") from exc
    return rendered.encode("utf-8")


def canonical_json(value: Any) -> str:
    """Return canonical UTF-8 JSON as text for diagnostics and fixtures."""

    return canonical_json_bytes(value).decode("utf-8")


def sha256_hex(value: bytes | bytearray | memoryview) -> str:
    """Return a lower-case SHA-256 digest for public integrity metadata."""

    return hashlib.sha256(bytes(value)).hexdigest()


def _hex_digest(value: Any, field: str) -> bytes:
    text = str(value or "")
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text.lower()):
        raise GardShardVerificationError(f"invalid {field}")
    try:
        return bytes.fromhex(text)
    except ValueError as exc:
        raise GardShardVerificationError(f"invalid {field}") from exc


def _required_int(value: Any, field: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GardShardVerificationError(f"invalid {field}")
    parsed = int(value)
    if parsed < minimum or (maximum is not None and parsed > maximum):
        raise GardShardVerificationError(f"invalid {field}")
    return parsed


def _b64_encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64_decode(value: Any, field: str) -> bytes:
    if not isinstance(value, str):
        raise GardShardVerificationError(f"invalid {field}")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise GardShardVerificationError(f"invalid {field}") from exc


def _u32(value: int) -> bytes:
    if not 0 <= int(value) <= 0xFFFFFFFF:
        raise GardShardError("integer does not fit the GARD uint32 wire field")
    return struct.pack(">I", int(value))


def _binary_record(*fields: bytes) -> bytes:
    """Length-prefix binary fields to avoid ambiguous MAC/KDF concatenation."""

    out = bytearray(b"GARD\x00")
    for field in fields:
        raw = bytes(field)
        out.extend(_u32(len(raw)))
        out.extend(raw)
    return bytes(out)


def _secret_bytes(
    secret: str | bytes | bytearray | None,
    config: GardShardConfig,
) -> bytes:
    value: str | bytes | bytearray | None = secret
    if value is None:
        value = os.environ.get(config.secret_env)
    if isinstance(value, str):
        if not value.strip():
            raise GardShardSecretError(
                f"{config.secret_env} is required for GARD shard cryptography"
            )
        raw = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raise GardShardSecretError(
            f"{config.secret_env} is required for GARD shard cryptography"
        )
    if not raw:
        raise GardShardSecretError(
            f"{config.secret_env} is required for GARD shard cryptography"
        )
    return raw


def _derive_shard_keys(
    secret: bytes,
    *,
    salt: bytes,
    model_digest: bytes,
    shard_index: int,
    shard_count: int,
    config: GardShardConfig,
) -> tuple[bytes, bytes]:
    """Derive the legacy v1 pair of independent AES and HMAC keys (HKDF-SHA256).

    Retained for reading ``gard-shard/v1`` envelopes. The protocol string is
    pinned to the v1 constant because it is bound into the HKDF info: reading
    the module-level ``GARD_PROTOCOL`` here would change v1 derivation the
    moment the current version advances.
    """

    if len(salt) != _SALT_BYTES:
        raise GardShardVerificationError("invalid shard salt length")
    info = _binary_record(
        b"gard-shard-hkdf/v1",
        GARD_PROTOCOL_V1.encode("utf-8"),
        config.domain.encode("utf-8"),
        model_digest,
        _u32(shard_index),
        _u32(shard_count),
    )
    material = HKDF(
        algorithm=hashes.SHA256(),
        length=_HKDF_BYTES,
        salt=salt,
        info=info,
    ).derive(secret)
    return material[:_AES_KEY_BYTES], material[_AES_KEY_BYTES:]


def _derive_shard_key_gcm(
    secret: bytes,
    *,
    salt: bytes,
    model_digest: bytes,
    shard_index: int,
    shard_count: int,
    config: GardShardConfig,
) -> bytes:
    """Derive the single v2 AES-256-GCM key with HKDF-SHA256.

    GCM authenticates with the same key it encrypts under, so unlike v1 there
    is no second MAC key. The HKDF label and the bound protocol string are both
    version-distinct, so a v1 and a v2 shard sharing a salt, model digest and
    index still derive unrelated keys.
    """

    if len(salt) != _SALT_BYTES:
        raise GardShardVerificationError("invalid shard salt length")
    info = _binary_record(
        b"gard-shard-hkdf/v2",
        GARD_PROTOCOL.encode("utf-8"),
        config.domain.encode("utf-8"),
        model_digest,
        _u32(shard_index),
        _u32(shard_count),
    )
    return HKDF(
        algorithm=hashes.SHA256(),
        length=_AES_KEY_BYTES,
        salt=salt,
        info=info,
    ).derive(secret)


def _mac_preimage(
    *,
    model_digest: bytes,
    salt: bytes,
    shard_index: int,
    shard_count: int,
    iv: bytes,
    ciphertext: bytes,
    config: GardShardConfig,
    compression_level: int | None = None,
) -> bytes:
    """Return the fixed, length-prefixed v1 Encrypt-then-MAC preimage.

    Legacy read path only. As with the v1 KDF info, the protocol string and
    version are pinned to the v1 constants because they are bound into the MAC.
    """

    level = int(config.compression_level if compression_level is None else compression_level)
    return _binary_record(
        b"gard-shard-mac/v1",
        GARD_PROTOCOL_V1.encode("utf-8"),
        _u32(GARD_PROTOCOL_VERSION_V1),
        config.domain.encode("utf-8"),
        _u32(level + 1),
        model_digest,
        salt,
        _u32(shard_index),
        _u32(shard_count),
        iv,
        ciphertext,
    )


def _gcm_aad(
    *,
    model_digest: bytes,
    salt: bytes,
    shard_index: int,
    shard_count: int,
    nonce: bytes,
    config: GardShardConfig,
    compression_level: int | None = None,
) -> bytes:
    """Return the v2 AES-GCM additional authenticated data.

    Binds exactly the context v1 bound into its MAC preimage, minus the
    ciphertext (which GCM authenticates natively). Because this is AAD, a shard
    lifted into a different domain, reordered within the set, or replayed under
    a different model context fails the tag check rather than decrypting
    cleanly -- the same cross-context binding v1 achieved via the MAC.
    """

    level = int(config.compression_level if compression_level is None else compression_level)
    return _binary_record(
        b"gard-shard-aad/v2",
        GARD_PROTOCOL.encode("utf-8"),
        _u32(GARD_PROTOCOL_VERSION),
        config.domain.encode("utf-8"),
        _u32(level + 1),
        model_digest,
        salt,
        _u32(shard_index),
        _u32(shard_count),
        nonce,
    )


def _encrypt_cbc(plaintext: bytes, key: bytes, iv: bytes) -> bytes:
    if len(key) != _AES_KEY_BYTES or len(iv) != _IV_BYTES:
        raise GardShardError("invalid AES-256-CBC key material")
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded = padder.update(plaintext) + padder.finalize()
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return encryptor.update(padded) + encryptor.finalize()


def _decrypt_cbc(ciphertext: bytes, key: bytes, iv: bytes) -> bytes:
    if len(key) != _AES_KEY_BYTES or len(iv) != _IV_BYTES:
        raise GardShardVerificationError("invalid AES-256-CBC key material")
    try:
        decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(padded) + unpadder.finalize()
    except ValueError as exc:
        raise GardShardVerificationError("GARD ciphertext could not be decrypted") from exc


def _encrypt_gcm(plaintext: bytes, key: bytes, nonce: bytes, aad: bytes) -> bytes:
    """AES-256-GCM seal. Returns ciphertext with the 16-byte tag appended."""
    if len(key) != _AES_KEY_BYTES or len(nonce) != _GCM_NONCE_BYTES:
        raise GardShardError("invalid AES-256-GCM key material")
    return AESGCM(key).encrypt(nonce, plaintext, aad)


def _decrypt_gcm(ciphertext: bytes, key: bytes, nonce: bytes, aad: bytes) -> bytes:
    """AES-256-GCM open. Raises if the tag, nonce, key, or AAD do not match.

    There is no way to obtain plaintext from a tampered v2 shard: the tag check
    is inside the primitive, not a separate pass that could be skipped.
    """
    if len(key) != _AES_KEY_BYTES or len(nonce) != _GCM_NONCE_BYTES:
        raise GardShardVerificationError("invalid AES-256-GCM key material")
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise GardShardVerificationError("GARD envelope authentication failed") from exc


def _split_bytes(payload: bytes, count: int) -> list[bytes]:
    count = max(1, min(int(count), max(1, len(payload))))
    base, remainder = divmod(len(payload), count)
    chunks: list[bytes] = []
    cursor = 0
    for index in range(count):
        width = base + (1 if index < remainder else 0)
        chunks.append(payload[cursor:cursor + width])
        cursor += width
    return chunks


def _fixed_material(
    values: Sequence[bytes] | None,
    index: int,
    size: int,
    label: str,
) -> bytes:
    raw = secrets.token_bytes(size) if values is None else bytes(values[index])
    if len(raw) != size:
        raise GardShardError(f"test {label} at shard {index} must be {size} bytes")
    return raw


def _safe_vector(value: Any, length: int, fallback: float = 0.5) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return [fallback] * length
    out: list[float] = []
    for raw in list(value)[:length]:
        try:
            out.append(round(max(0.0, min(1.0, float(raw))), 6))
        except (TypeError, ValueError):
            out.append(fallback)
    return (out + [fallback] * length)[:length]


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_kv_json(key: str, default: Any) -> Any:
    try:
        from .brain_kv import kv_get_json

        return kv_get_json(key, default=default)
    except Exception:
        return default


def langevin_sigma_from_weyl(
    psi5_now: Sequence[Any],
    psi5_prev: Sequence[Any],
    *,
    kappa: float = 18.0,
    sigma_base: float = 0.05,
) -> float:
    """Return the corrected Weyl-only Langevin diffusion reference.

    This mirrors the self-contained Julia compression reference.  A malformed
    pair falls back to the base coefficient; both vectors must be complete
    before the delta-Lambda and disagreement terms are evaluated.
    """

    if not isinstance(psi5_now, (list, tuple)) or not isinstance(psi5_prev, (list, tuple)):
        return float(sigma_base)
    if len(psi5_now) < 5 or len(psi5_prev) < 5:
        return float(sigma_base)
    try:
        safe4 = lambda value: max(0.0, min(1.0 - 1e-9, float(value)))
        lambda_now = safe4(psi5_now[4]) * float(kappa) / (1.0 - safe4(psi5_now[4]))
        lambda_prev = safe4(psi5_prev[4]) * float(kappa) / (1.0 - safe4(psi5_prev[4]))
        delta_lambda = max(0.0, lambda_now - lambda_prev)
        psi3_now = max(1e-9, min(1.0, float(psi5_now[3])))
        psi3_prev = max(1e-9, min(1.0, float(psi5_prev[3])))
        disagreement = abs(psi3_now - psi3_prev) / (psi3_prev + 1e-9)
        token_entropy = max(0.0, min(1.0, float(psi5_now[1])))
        return max(
            0.0,
            min(
                1.0,
                float(sigma_base)
                * (1.0 + disagreement)
                * token_entropy
                * (1.0 + 0.1 * delta_lambda),
            ),
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return float(sigma_base)


# GARD information-science alias (same function, compression/information naming).
langevin_sigma_from_gard = langevin_sigma_from_weyl


def build_model_context(
    *,
    mesh_state7: Sequence[float] | None = None,
    weyl_psi5: Sequence[float] | None = None,
    scm_state: Mapping[str, Any] | None = None,
    slm_state: Mapping[str, Any] | None = None,
    config: GardShardConfig | None = None,
) -> dict[str, Any]:
    """Build a secret-free serializable identity for the current MESH model.

    The context is authenticated as metadata by the per-shard MAC and feeds
    HKDF as a public digest.  It is intentionally not key material and does
    not include raw corpus data, private payloads, or any grid secret.
    """

    cfg = _config(config)
    state = _safe_mapping(slm_state)
    if not state:
        try:
            from . import mesh_slm

            state = _safe_mapping(mesh_slm.state_summary())
        except Exception:
            state = {}

    mesh = _safe_vector(
        mesh_state7 if mesh_state7 is not None else state.get("mesh_state"),
        7,
    )
    weyl = _safe_vector(
        weyl_psi5 if weyl_psi5 is not None else _safe_kv_json("learnings:weyl_tensor", []),
        5,
    )
    runtime = _safe_mapping(_safe_kv_json("ueqgm:adaptive_runtime", {}))
    scm = _safe_mapping(scm_state)
    if not scm:
        scm = {
            "phase_weight": runtime.get("phase_weight", state.get("last_phase_weight", 1.0)),
            "coherence_depth": runtime.get("coherence_depth", 0),
            "phi": runtime.get("phi", 0.0),
            "last_emission": state.get("mcd") or state.get("scm_last_emission"),
        }
    previous_weyl = _safe_vector(scm.get("previous_weyl_psi5", weyl), 5)

    vocab_size = int(state.get("vocab_size") or 0)
    quipu_edges = int(state.get("quipu_edges") or 0)
    compaction: dict[str, Any] = {}
    try:
        from .ueqgm_engine import mesh_compaction_summary

        raw_compaction = mesh_compaction_summary(
            n_vocab=vocab_size or None,
            n_quipu_edges=quipu_edges or None,
        )
        compaction = {
            key: raw_compaction.get(key)
            for key in (
                "mesh_bytes_total",
                "compaction_ratio",
                "compaction_scalar",
                "hawking_remnant_score",
            )
        }
        if "gard_crb" in raw_compaction:
            compaction["gard_crb"] = raw_compaction["gard_crb"]
    except Exception:
        compaction = {}

    slm_id = "mesh-slm/torus-quipu-7d"
    slm_arch = "MESH-SLM-GLM-GNN"
    try:
        from . import mesh_slm

        slm_id = str(getattr(mesh_slm, "MODEL_ID", slm_id))
        slm_arch = str(getattr(mesh_slm, "MODEL_ARCH", slm_arch))
    except Exception:
        pass

    return {
        "schema": GARD_MODEL_CONTEXT_SCHEMA,
        "architecture": "MESH-SLM-SCM-GLM-Quipu-GNN",
        "mesh_state7": mesh,
        "weyl_psi5": weyl,
        "gard_state5": weyl,
        "scm": {
            "phase_weight": scm.get("phase_weight", 1.0),
            "coherence_depth": scm.get("coherence_depth", 0),
            "phi": scm.get("phi", 0.0),
            "last_emission": scm.get("last_emission"),
            "langevin_sigma_reference": round(langevin_sigma_from_weyl(weyl, previous_weyl), 6),
        },
        "models": {
            "slm": {"id": slm_id, "architecture": slm_arch},
            "glm": {"id": "mesh-glm/7d-alignment", "architecture": "MESH-GLM"},
            "gnn": {"id": "quipu-gnn/hebbian", "architecture": "Quipu-GNN"},
            "scm": {"id": "saturation-cycle-model", "architecture": "SCM"},
        },
        "quipu_gnn": {
            "vocab_size": vocab_size,
            "vocab_capacity": int(state.get("vocab_capacity") or 4096),
            "edge_count": quipu_edges,
            "rounds": int(state.get("rounds") or 0),
            "compaction": compaction,
        },
        "gard": {
            "protocol": GARD_PROTOCOL,
            "domain": cfg.domain,
            "verify_digest_default": bool(cfg.verify_digest),
        },
    }


def _normalise_envelope(
    envelope: Mapping[str, Any],
    config: GardShardConfig,
) -> dict[str, Any]:
    if not isinstance(envelope, Mapping):
        raise GardShardVerificationError("GARD envelope must be an object")
    out = dict(envelope)
    if out.get("schema") == GARD_ENVELOPE_SCHEMA and out.get("protocol") == GARD_PROTOCOL:
        version = GARD_PROTOCOL_VERSION
    elif out.get("schema") == GARD_ENVELOPE_SCHEMA_V1 and out.get("protocol") == GARD_PROTOCOL_V1:
        version = GARD_PROTOCOL_VERSION_V1
    else:
        raise GardShardVerificationError("unsupported GARD envelope protocol")
    if _required_int(out.get("protocol_version"), "protocol_version", minimum=version, maximum=version) != version:
        raise GardShardVerificationError("unsupported GARD envelope version")
    if out.get("domain") != config.domain:
        raise GardShardVerificationError("GARD envelope domain mismatch")
    if out.get("encoding") != "base64":
        raise GardShardVerificationError("unsupported GARD encoding")
    compression = out.get("compression")
    if not isinstance(compression, Mapping) or compression.get("algorithm") != "zlib":
        raise GardShardVerificationError("unsupported GARD compression configuration")
    _required_int(compression.get("level"), "compression level", minimum=-1, maximum=9)
    if version == GARD_PROTOCOL_VERSION_V1:
        if out.get("encryption") != {"algorithm": "AES-256-CBC", "padding": "PKCS#7"}:
            raise GardShardVerificationError("unsupported GARD encryption configuration")
        if out.get("authentication") != {"algorithm": "HMAC-SHA256", "mode": "encrypt-then-mac"}:
            raise GardShardVerificationError("unsupported GARD authentication configuration")
    else:
        if out.get("encryption") != {"algorithm": "AES-256-GCM", "padding": "none"}:
            raise GardShardVerificationError("unsupported GARD encryption configuration")
        if out.get("authentication") != {"algorithm": "AES-256-GCM", "mode": "aead"}:
            raise GardShardVerificationError("unsupported GARD authentication configuration")

    context = out.get("model_context")
    if not isinstance(context, Mapping):
        raise GardShardVerificationError("missing GARD model context")
    model_digest_text = out.get("model_digest")
    model_digest = _hex_digest(model_digest_text, "model_digest")
    if not isinstance(model_digest_text, str) or not hmac.compare_digest(sha256_hex(canonical_json_bytes(context)), model_digest_text):
        raise GardShardVerificationError("GARD model context digest mismatch")

    _hex_digest(out.get("plaintext_sha256"), "plaintext_sha256")
    _hex_digest(out.get("compressed_sha256"), "compressed_sha256")
    shard_count = _required_int(out.get("shard_count"), "shard_count", minimum=1, maximum=_MAX_SHARDS)
    shards = out.get("shards")
    if not isinstance(shards, list) or len(shards) != shard_count:
        raise GardShardVerificationError("invalid GARD shard collection")
    compressed_bytes = _required_int(
        out.get("compressed_bytes"),
        "compressed_bytes",
        maximum=int(config.max_compressed_bytes),
    )
    plaintext_bytes = _required_int(
        out.get("plaintext_bytes"),
        "plaintext_bytes",
        maximum=int(config.max_plaintext_bytes),
    )

    checked_shards: list[dict[str, Any]] = []
    for expected_index, raw_shard in enumerate(shards):
        if not isinstance(raw_shard, Mapping):
            raise GardShardVerificationError("invalid GARD shard")
        shard = dict(raw_shard)
        shard_index = _required_int(shard.get("index"), "GARD shard index", minimum=0, maximum=_MAX_SHARDS - 1)
        declared_count = _required_int(shard.get("count"), "GARD shard count", minimum=1, maximum=_MAX_SHARDS)
        if shard_index != expected_index or declared_count != shard_count:
            raise GardShardVerificationError("inconsistent GARD shard ordering")
        salt = _b64_decode(shard.get("salt_b64"), "salt_b64")
        ciphertext = _b64_decode(shard.get("ciphertext_b64"), "ciphertext_b64")
        if len(salt) != _SALT_BYTES:
            raise GardShardVerificationError("invalid GARD shard field length")
        if sha256_hex(ciphertext) != str(shard.get("ciphertext_sha256") or ""):
            raise GardShardVerificationError("GARD ciphertext digest mismatch")

        if version == GARD_PROTOCOL_VERSION_V1:
            iv = _b64_decode(shard.get("iv_b64"), "iv_b64")
            mac = _b64_decode(shard.get("hmac_sha256_b64"), "hmac_sha256_b64")
            if len(iv) != _IV_BYTES or len(mac) != _HMAC_BYTES:
                raise GardShardVerificationError("invalid GARD shard field length")
            # CBC ciphertext is always a whole number of blocks.
            if not ciphertext or len(ciphertext) % _IV_BYTES:
                raise GardShardVerificationError("invalid GARD ciphertext")
            checked_shards.append(
                {
                    "index": shard_index,
                    "count": declared_count,
                    "salt": salt,
                    "iv": iv,
                    "ciphertext": ciphertext,
                    "mac": mac,
                }
            )
        else:
            nonce = _b64_decode(shard.get("nonce_b64"), "nonce_b64")
            if len(nonce) != _GCM_NONCE_BYTES:
                raise GardShardVerificationError("invalid GARD shard field length")
            # GCM is a stream mode: no block alignment, but the tag is always
            # present, so the sealed length is at least the tag length.
            if len(ciphertext) < _GCM_TAG_BYTES:
                raise GardShardVerificationError("invalid GARD ciphertext")
            checked_shards.append(
                {
                    "index": shard_index,
                    "count": declared_count,
                    "salt": salt,
                    "nonce": nonce,
                    "ciphertext": ciphertext,
                }
            )

    return {
        "raw": out,
        "version": version,
        "model_digest": model_digest,
        "shard_count": shard_count,
        "shards": checked_shards,
        "compression_level": _required_int(compression.get("level"), "compression level", minimum=-1, maximum=9),
        "compressed_bytes": compressed_bytes,
        "plaintext_bytes": plaintext_bytes,
    }


def verify_envelope(
    envelope: Mapping[str, Any],
    *,
    secret: str | bytes | bytearray | None = None,
    config: GardShardConfig | None = None,
) -> dict[str, Any]:
    """Authenticate every GARD shard before use and return safe diagnostics.

    For v1 this checks each HMAC without decrypting. For v2 the AEAD tag can
    only be checked by opening the shard, so this performs the GCM open and
    discards the plaintext.

    ``verify_digest=False`` skips this pass. Note the asymmetry that makes v2
    stricter: for v1 that genuinely disabled integrity checking and a tampered
    envelope would still decrypt, whereas a v2 shard is authenticated inside
    the AEAD primitive on every open, so tampering is always rejected no matter
    how this flag is set.
    """

    cfg = _config(config)
    checked = _normalise_envelope(envelope, cfg)
    protocol = checked["raw"]["protocol"]
    if not cfg.verify_digest:
        return {
            "verified": False,
            "verification_disabled": True,
            "protocol": protocol,
            "shard_count": checked["shard_count"],
            "model_digest": checked["raw"]["model_digest"],
        }

    grid_secret = _secret_bytes(secret, cfg)
    for shard in checked["shards"]:
        if checked["version"] == GARD_PROTOCOL_VERSION_V1:
            _, mac_key = _derive_shard_keys(
                grid_secret,
                salt=shard["salt"],
                model_digest=checked["model_digest"],
                shard_index=shard["index"],
                shard_count=checked["shard_count"],
                config=cfg,
            )
            expected = hmac.new(
                mac_key,
                _mac_preimage(
                    model_digest=checked["model_digest"],
                    salt=shard["salt"],
                    shard_index=shard["index"],
                    shard_count=checked["shard_count"],
                    iv=shard["iv"],
                    ciphertext=shard["ciphertext"],
                    config=cfg,
                    compression_level=checked["compression_level"],
                ),
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(expected, shard["mac"]):
                raise GardShardVerificationError("GARD envelope authentication failed")
        else:
            key = _derive_shard_key_gcm(
                grid_secret,
                salt=shard["salt"],
                model_digest=checked["model_digest"],
                shard_index=shard["index"],
                shard_count=checked["shard_count"],
                config=cfg,
            )
            _decrypt_gcm(
                shard["ciphertext"],
                key,
                shard["nonce"],
                _gcm_aad(
                    model_digest=checked["model_digest"],
                    salt=shard["salt"],
                    shard_index=shard["index"],
                    shard_count=checked["shard_count"],
                    nonce=shard["nonce"],
                    config=cfg,
                    compression_level=checked["compression_level"],
                ),
            )

    return {
        "verified": True,
        "verification_disabled": False,
        "protocol": protocol,
        "shard_count": checked["shard_count"],
        "model_digest": checked["raw"]["model_digest"],
    }


def encrypt_json(
    payload: Any,
    *,
    secret: str | bytes | bytearray | None = None,
    model_context: Mapping[str, Any] | None = None,
    config: GardShardConfig | None = None,
    fixed_salts: Sequence[bytes] | None = None,
    fixed_ivs: Sequence[bytes] | None = None,
    fixed_nonces: Sequence[bytes] | None = None,
) -> dict[str, Any]:
    """Compress, split, encrypt, and authenticate JSON as a GARD envelope.

    Writes the current protocol (``gard-shard/v2``, AES-256-GCM). Reading
    ``gard-shard/v1`` envelopes remains supported via :func:`decrypt_json`.

    ``fixed_salts`` and ``fixed_nonces`` exist solely for deterministic public
    interoperability fixtures.  Production callers must leave them unset so
    each encrypted shard obtains fresh random material from ``secrets``.

    A GCM nonce must never repeat under the same key. Every shard here derives
    a distinct key (fresh random salt per shard through HKDF), so nonce reuse
    across shards is not a cross-shard risk -- but supplying ``fixed_nonces``
    together with ``fixed_salts`` and a fixed secret does pin a single
    (key, nonce) pair, which is why both are restricted to test fixtures.

    ``fixed_ivs`` is accepted as a deprecated alias for ``fixed_nonces``; v2
    uses a 12-byte GCM nonce rather than a 16-byte CBC IV, so entries are
    truncated to the nonce length.
    """

    cfg = _config(config)
    grid_secret = _secret_bytes(secret, cfg)
    context = dict(model_context) if isinstance(model_context, Mapping) else build_model_context(config=cfg)
    plain = canonical_json_bytes(payload)
    if len(plain) > int(cfg.max_plaintext_bytes):
        raise GardShardError("GARD plaintext exceeds configured limit")
    compressed = zlib.compress(plain, level=int(cfg.compression_level))
    if len(compressed) > int(cfg.max_compressed_bytes):
        raise GardShardError("GARD compressed payload exceeds configured limit")

    model_digest_text = sha256_hex(canonical_json_bytes(context))
    model_digest = bytes.fromhex(model_digest_text)
    chunks = _split_bytes(compressed, cfg.shard_count)
    if fixed_salts is not None and len(fixed_salts) < len(chunks):
        raise GardShardError("fixed_salts must provide one salt per shard")
    # fixed_ivs is the deprecated v1 spelling; v2 nonces are shorter, so the
    # supplied material is truncated to the GCM nonce length.
    if fixed_nonces is None and fixed_ivs is not None:
        fixed_nonces = [bytes(v)[:_GCM_NONCE_BYTES] for v in fixed_ivs]
    if fixed_nonces is not None and len(fixed_nonces) < len(chunks):
        raise GardShardError("fixed_nonces must provide one nonce per shard")

    encoded_shards: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        salt = _fixed_material(fixed_salts, index, _SALT_BYTES, "salt")
        nonce = _fixed_material(fixed_nonces, index, _GCM_NONCE_BYTES, "nonce")
        encryption_key = _derive_shard_key_gcm(
            grid_secret,
            salt=salt,
            model_digest=model_digest,
            shard_index=index,
            shard_count=len(chunks),
            config=cfg,
        )
        ciphertext = _encrypt_gcm(
            chunk,
            encryption_key,
            nonce,
            _gcm_aad(
                model_digest=model_digest,
                salt=salt,
                shard_index=index,
                shard_count=len(chunks),
                nonce=nonce,
                config=cfg,
                compression_level=cfg.compression_level,
            ),
        )
        encoded_shards.append(
            {
                "index": index,
                "count": len(chunks),
                "salt_b64": _b64_encode(salt),
                "nonce_b64": _b64_encode(nonce),
                "ciphertext_b64": _b64_encode(ciphertext),
                "ciphertext_sha256": sha256_hex(ciphertext),
            }
        )

    return {
        "schema": GARD_ENVELOPE_SCHEMA,
        "protocol": GARD_PROTOCOL,
        "protocol_version": GARD_PROTOCOL_VERSION,
        "domain": cfg.domain,
        "encoding": "base64",
        "compression": {"algorithm": "zlib", "level": int(cfg.compression_level)},
        "encryption": {"algorithm": "AES-256-GCM", "padding": "none"},
        "authentication": {"algorithm": "AES-256-GCM", "mode": "aead"},
        "model_context": context,
        "model_digest": model_digest_text,
        "plaintext_sha256": sha256_hex(plain),
        "compressed_sha256": sha256_hex(compressed),
        "plaintext_bytes": len(plain),
        "compressed_bytes": len(compressed),
        "shard_count": len(chunks),
        "shards": encoded_shards,
    }


def _bounded_decompress(compressed: bytes, max_plaintext_bytes: int) -> bytes:
    try:
        decoder = zlib.decompressobj()
        plaintext = decoder.decompress(compressed, max_plaintext_bytes + 1)
        if len(plaintext) > max_plaintext_bytes or decoder.unconsumed_tail:
            raise GardShardVerificationError("GARD decompressed payload exceeds configured limit")
        plaintext += decoder.flush(max_plaintext_bytes + 1 - len(plaintext))
        if len(plaintext) > max_plaintext_bytes or not decoder.eof or decoder.unused_data:
            raise GardShardVerificationError("invalid or oversized GARD compressed payload")
        return plaintext
    except zlib.error as exc:
        raise GardShardVerificationError("GARD compressed payload is invalid") from exc


def decrypt_json(
    envelope: Mapping[str, Any],
    *,
    secret: str | bytes | bytearray | None = None,
    config: GardShardConfig | None = None,
    expected_model_context: Mapping[str, Any] | None = None,
) -> Any:
    """Verify, decrypt, decompress, and deserialize a GARD JSON envelope.

    Accepts both the current ``gard-shard/v2`` (AES-256-GCM) and the legacy
    ``gard-shard/v1`` (AES-256-CBC + HMAC-SHA256) protocols.
    """

    cfg = _config(config)
    checked = _normalise_envelope(envelope, cfg)
    if expected_model_context is not None:
        expected = sha256_hex(canonical_json_bytes(dict(expected_model_context)))
        if not hmac.compare_digest(expected, checked["raw"]["model_digest"]):
            raise GardShardVerificationError("GARD model context does not match the envelope")

    is_v1 = checked["version"] == GARD_PROTOCOL_VERSION_V1
    if is_v1:
        # v1 has no integrity inside the cipher, so every MAC must be checked
        # before the CBC decrypt/decompression path is entered at all. v2 skips
        # this pass because the GCM open below authenticates natively -- running
        # it here would just decrypt each shard twice.
        verify_envelope(checked["raw"], secret=secret, config=cfg)

    grid_secret = _secret_bytes(secret, cfg)
    recovered_chunks: list[bytes] = []
    for shard in checked["shards"]:
        if is_v1:
            encryption_key, _ = _derive_shard_keys(
                grid_secret,
                salt=shard["salt"],
                model_digest=checked["model_digest"],
                shard_index=shard["index"],
                shard_count=checked["shard_count"],
                config=cfg,
            )
            recovered_chunks.append(_decrypt_cbc(shard["ciphertext"], encryption_key, shard["iv"]))
        else:
            encryption_key = _derive_shard_key_gcm(
                grid_secret,
                salt=shard["salt"],
                model_digest=checked["model_digest"],
                shard_index=shard["index"],
                shard_count=checked["shard_count"],
                config=cfg,
            )
            recovered_chunks.append(
                _decrypt_gcm(
                    shard["ciphertext"],
                    encryption_key,
                    shard["nonce"],
                    _gcm_aad(
                        model_digest=checked["model_digest"],
                        salt=shard["salt"],
                        shard_index=shard["index"],
                        shard_count=checked["shard_count"],
                        nonce=shard["nonce"],
                        config=cfg,
                        compression_level=checked["compression_level"],
                    ),
                )
            )

    compressed = b"".join(recovered_chunks)
    if len(compressed) != checked["compressed_bytes"]:
        raise GardShardVerificationError("GARD compressed payload length mismatch")
    if sha256_hex(compressed) != checked["raw"]["compressed_sha256"]:
        raise GardShardVerificationError("GARD compressed payload digest mismatch")
    plaintext = _bounded_decompress(compressed, int(cfg.max_plaintext_bytes))
    if len(plaintext) != checked["plaintext_bytes"]:
        raise GardShardVerificationError("GARD plaintext payload length mismatch")
    if sha256_hex(plaintext) != checked["raw"]["plaintext_sha256"]:
        raise GardShardVerificationError("GARD plaintext digest mismatch")
    try:
        return json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GardShardVerificationError("GARD plaintext is not valid UTF-8 JSON") from exc


def write_envelope_json(envelope: Mapping[str, Any], path: str | Path) -> Path:
    """Write an encrypted GARD envelope as readable JSON (never plaintext)."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(envelope), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_envelope_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON envelope without decrypting it."""

    source = Path(path).expanduser()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GardShardError(f"could not read GARD envelope {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GardShardError("GARD envelope JSON root must be an object")
    return payload


# Assignment proofs are an independent HMAC-SHA256 signature over catalog
# assignments -- they carry no ciphertext and are untouched by the envelope's
# CBC->GCM migration. Their bound protocol string stays pinned to v1 so proofs
# already issued continue to verify; only the envelope protocol advanced.
_ASSIGNMENT_PROOF_PROTOCOL = GARD_PROTOCOL_V1


def _assignment_proof_key(secret: bytes, assignment_digest: bytes, config: GardShardConfig) -> bytes:
    info = _binary_record(
        b"gard-assignment-proof-hkdf/v1",
        _ASSIGNMENT_PROOF_PROTOCOL.encode("utf-8"),
        config.domain.encode("utf-8"),
        assignment_digest,
    )
    return HKDF(
        algorithm=hashes.SHA256(),
        length=_HMAC_KEY_BYTES,
        salt=assignment_digest,
        info=info,
    ).derive(secret)


def _assignment_proof_preimage(assignment_digest: bytes, config: GardShardConfig) -> bytes:
    return _binary_record(
        b"gard-assignment-proof-mac/v1",
        _ASSIGNMENT_PROOF_PROTOCOL.encode("utf-8"),
        GARD_ASSIGNMENT_PROOF_KIND.encode("utf-8"),
        config.domain.encode("utf-8"),
        assignment_digest,
    )


def sign_assignment_proof(
    assignment: Mapping[str, Any],
    *,
    secret: str | bytes | bytearray | None = None,
    config: GardShardConfig | None = None,
) -> dict[str, Any]:
    """Create a compact, secret-free HMAC proof for a canonical assignment."""

    cfg = _config(config)
    if not isinstance(assignment, Mapping):
        raise GardShardError("GARD assignment must be an object")
    canonical = canonical_json_bytes(dict(assignment))
    normalized_assignment = json.loads(canonical.decode("utf-8"))
    digest_text = sha256_hex(canonical)
    digest = bytes.fromhex(digest_text)
    proof_key = _assignment_proof_key(_secret_bytes(secret, cfg), digest, cfg)
    proof = hmac.new(
        proof_key,
        _assignment_proof_preimage(digest, cfg),
        hashlib.sha256,
    ).digest()
    return {
        "protocol": _ASSIGNMENT_PROOF_PROTOCOL,
        "kind": GARD_ASSIGNMENT_PROOF_KIND,
        "domain": cfg.domain,
        "assignment": normalized_assignment,
        "assignment_sha256": digest_text,
        "proof_b64": _b64_encode(proof),
    }


def verify_assignment_proof(
    proof: Mapping[str, Any] | None,
    *,
    secret: str | bytes | bytearray | None = None,
    config: GardShardConfig | None = None,
    verify_digest: bool | None = None,
) -> bool:
    """Fail closed for missing, malformed, or unauthenticated assignment proofs."""

    cfg = _config(config)
    required = cfg.verify_digest if verify_digest is None else bool(verify_digest)
    if not required:
        return True
    try:
        if not isinstance(proof, Mapping):
            return False
        if proof.get("protocol") != _ASSIGNMENT_PROOF_PROTOCOL:
            return False
        if proof.get("kind") != GARD_ASSIGNMENT_PROOF_KIND:
            return False
        if proof.get("domain") != cfg.domain or not isinstance(proof.get("assignment"), Mapping):
            return False
        canonical = canonical_json_bytes(dict(proof["assignment"]))
        digest_text = sha256_hex(canonical)
        if not hmac.compare_digest(digest_text, str(proof.get("assignment_sha256") or "")):
            return False
        digest = bytes.fromhex(digest_text)
        supplied = _b64_decode(proof.get("proof_b64"), "proof_b64")
        if len(supplied) != _HMAC_BYTES:
            return False
        key = _assignment_proof_key(_secret_bytes(secret, cfg), digest, cfg)
        expected = hmac.new(
            key,
            _assignment_proof_preimage(digest, cfg),
            hashlib.sha256,
        ).digest()
        return hmac.compare_digest(expected, supplied)
    except (GardShardError, ValueError, TypeError):
        return False


def build_hub_manifest(
    *,
    model_context: Mapping[str, Any] | None = None,
    config: GardShardConfig | None = None,
) -> dict[str, Any]:
    """Build the secret-free status manifest served by the local Loadopoly Hub."""

    cfg = _config(config)
    context = dict(model_context) if isinstance(model_context, Mapping) else build_model_context(config=cfg)
    return {
        "schema": GARD_HUB_MANIFEST_SCHEMA,
        "generated_at": _utc_now(),
        "model": {
            "name": "MESH-SLM-SCM-GLM-Quipu-GNN GARD Shard",
            "protocol": GARD_PROTOCOL,
            "python_module": "src.quipu.gard_shard_model",
            "julia_module": "GardShardModel",
            "julia_source": "pipeline/src/quipu/gard_shard_model.jl",
        },
        "protocol": {
            "version": GARD_PROTOCOL_VERSION,
            "domain": cfg.domain,
            "canonical_payload": "UTF-8 JSON, sorted keys, compact separators",
            "compression": "zlib before sharding",
            "encryption": "AES-256-GCM (no padding)",
            "authentication": "AES-256-GCM AEAD, 128-bit tag",
            "kdf": "HKDF-SHA256 per shard",
            "reads_legacy": f"{GARD_PROTOCOL_V1} (AES-256-CBC + HMAC-SHA256)",
        },
        "security": {
            "grid_secret_env": cfg.secret_env,
            "verify_digest_default": bool(cfg.verify_digest),
            "verify_digest_env": VERIFY_DIGEST_ENV,
            "secret_configured": bool(os.environ.get(cfg.secret_env, "").strip()),
            "secrets_published": False,
            "plaintext_published": False,
        },
        "hub": {
            "route": "/gard-shard-model.json",
            "default_port": 9080,
        },
        "model_context_digest": sha256_hex(canonical_json_bytes(context)),
        "model_context": context,
        "interoperability_vector": "gard-shard-v1-fixed-001",
    }


def publish_loadopoly_manifest(
    path: str | Path | None = None,
    *,
    model_context: Mapping[str, Any] | None = None,
    config: GardShardConfig | None = None,
) -> Path:
    """Publish the public status manifest consumed by ``Loadopoly-Portal``."""

    target = Path(path) if path is not None else _WORKSPACE_ROOT / "Loadopoly-Portal" / "gard-shard-model.json"
    manifest = build_hub_manifest(model_context=model_context, config=config)
    return write_envelope_json(manifest, target)


def _selftest() -> dict[str, Any]:
    """Run a non-production protocol smoke test for the module CLI."""

    # Public fixture material only; this is never an operational grid secret.
    fixture_secret = b"gard-v1-public-test-fixture-key"
    payload = {
        "message": "GARD shard interoperability",
        "numbers": [1, 2, 3],
        "unicode": "MESH Psi",
    }
    context = {
        "architecture": "MESH-SLM-SCM-GLM-Quipu-GNN",
        "revision": 1,
    }
    envelope = encrypt_json(
        payload,
        secret=fixture_secret,
        model_context=context,
        config=GardShardConfig(shard_count=2),
        fixed_salts=[bytes(range(32)), bytes(range(32, 64))],
        fixed_nonces=[bytes(range(12)), bytes(range(12, 24))],
    )
    assert decrypt_json(envelope, secret=fixture_secret) == payload

    # Flipping a ciphertext bit must fail the AEAD tag. The ciphertext digest is
    # recomputed so the envelope stays internally consistent and the rejection
    # is demonstrably the GCM tag rather than the cheaper sha256 precheck.
    tampered = json.loads(json.dumps(envelope))
    raw = bytearray(base64.b64decode(tampered["shards"][0]["ciphertext_b64"]))
    raw[0] ^= 0x01
    tampered["shards"][0]["ciphertext_b64"] = _b64_encode(bytes(raw))
    tampered["shards"][0]["ciphertext_sha256"] = sha256_hex(bytes(raw))
    try:
        decrypt_json(tampered, secret=fixture_secret)
    except GardShardVerificationError:
        pass
    else:
        raise AssertionError("tampered GARD envelope was accepted")
    return {
        "ok": True,
        "protocol": GARD_PROTOCOL,
        "model_digest": envelope["model_digest"],
        "plaintext_sha256": envelope["plaintext_sha256"],
        "compressed_sha256": envelope["compressed_sha256"],
        "shard_count": envelope["shard_count"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``selftest`` validates protocol wiring; ``publish`` writes Hub status."""

    parser = argparse.ArgumentParser(description="GARD shard protocol and Hub manifest utility")
    parser.add_argument("command", choices=("selftest", "publish"))
    parser.add_argument("--output", help="Output path for the public publish manifest")
    args = parser.parse_args(argv)
    if args.command == "selftest":
        print(json.dumps(_selftest(), indent=2, sort_keys=True))
        return 0
    output = publish_loadopoly_manifest(args.output)
    print(json.dumps({"published": str(output), "protocol": GARD_PROTOCOL}, indent=2))
    return 0


__all__ = [
    "GARD_PROTOCOL",
    "GARD_PROTOCOL_VERSION",
    "GARD_PROTOCOL_V1",
    "GARD_PROTOCOL_VERSION_V1",
    "GARD_SHARD_DOMAIN",
    "GRID_SECRET_ENV",
    "VERIFY_DIGEST_ENV",
    "GardShardConfig",
    "GardShardError",
    "GardShardSecretError",
    "GardShardVerificationError",
    "canonical_json",
    "canonical_json_bytes",
    "sha256_hex",
    "langevin_sigma_from_weyl",
    "langevin_sigma_from_gard",
    "verification_enabled_from_env",
    "build_model_context",
    "encrypt_json",
    "decrypt_json",
    "verify_envelope",
    "write_envelope_json",
    "read_envelope_json",
    "sign_assignment_proof",
    "verify_assignment_proof",
    "build_hub_manifest",
    "publish_loadopoly_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())