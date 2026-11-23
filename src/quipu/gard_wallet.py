"""gard_wallet — GARD Shard marketplace wallet connectors: Solflare and MetaMask.

A wallet proves control of its address by signing a short sign-in message; QUIPU
verifies the signature and binds the address to a GARD assignment proof the
marketplace can check.  Nothing here touches a key, builds a transaction, or
moves funds: the only thing a wallet is ever asked to do is sign text.

    Solflare  (Solana)   ed25519 over the UTF-8 message (``signMessage``);
                         the address is the base58 public key itself.
    MetaMask  (EVM)      EIP-191 ``personal_sign`` over an EIP-4361
                         Sign-In-with-Ethereum message; the address is recovered
                         from the signature (secp256k1 + keccak-256, below) and
                         must equal the claimed address.

Flow (observer routes, see observer_service):
    POST /gard/wallet/challenge  {chain: "solana"|"evm", address, chain_id?, purpose?}
                                 -> {message, nonce, expires_at}   (in memory, single use)
    POST /gard/wallet/bind       {chain, address, nonce, signature}
                                 -> binding (+ GARD assignment proof when the grid
                                    secret is configured)
    POST /gard/wallet/unbind     same shape, a challenge issued with purpose "unbind"
    GET  /gard/wallet            index, or ?chain=&address= for one binding
    GET  /gard/wallet/connect    the connector page (Solflare + MetaMask)

Replay: every challenge is single-use, expires after ``CHALLENGE_TTL_S`` and is
bound to (chain, address, purpose, domain).  EIP-4361 framing lets MetaMask show
its own domain-mismatch warning if another site relays a message.

``QUIPU_GARD_WALLET_DOMAIN`` names the domain wallets sign for (default the
request's Host).  Writes only ``gard:wallet:*`` brain_kv keys; input to no gate.
"""
from __future__ import annotations

import base64
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from . import brain_kv

CHAINS = ("solana", "evm")
PURPOSES = ("bind", "unbind")
CHALLENGE_TTL_S = 300.0
_MAX_PENDING = 4096
DOMAIN_ENV = "QUIPU_GARD_WALLET_DOMAIN"

_KEY = "gard:wallet:{chain}:{address}"
_INDEX = "gard:wallet:index"
BINDING_KIND = "gard-wallet-binding/v1"


class WalletError(ValueError):
    """A challenge or signature that does not verify.  Always fail closed."""


# ---------------------------------------------------------------------------
# keccak-256 (the Ethereum hash; NOT hashlib.sha3_256, which pads differently)
# ---------------------------------------------------------------------------

_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]
_ROT = [
    [0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56], [27, 20, 39, 8, 14],
]
_M64 = (1 << 64) - 1


def _rol(v: int, s: int) -> int:
    s %= 64
    return ((v << s) | (v >> (64 - s))) & _M64 if s else v


def _keccak_f(a: list[list[int]]) -> None:
    for rc in _RC:
        c = [a[x][0] ^ a[x][1] ^ a[x][2] ^ a[x][3] ^ a[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ _rol(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                a[x][y] ^= d[x]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rol(a[x][y], _ROT[x][y])
        for x in range(5):
            for y in range(5):
                a[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])
        a[0][0] ^= rc


def keccak256(data: bytes) -> bytes:
    rate = 136
    msg = bytearray(data)
    msg.append(0x01)
    while len(msg) % rate:
        msg.append(0x00)
    msg[-1] |= 0x80
    a = [[0] * 5 for _ in range(5)]
    for off in range(0, len(msg), rate):
        block = msg[off:off + rate]
        for i in range(rate // 8):
            x, y = i % 5, i // 5
            a[x][y] ^= int.from_bytes(block[8 * i:8 * i + 8], "little")
        _keccak_f(a)
    out = bytearray()
    for i in range(4):
        x, y = i % 5, i // 5
        out += a[x][y].to_bytes(8, "little")
    return bytes(out)


# ---------------------------------------------------------------------------
# secp256k1 public-key recovery (verification only; no signing, no keys)
# ---------------------------------------------------------------------------

_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
      0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)


def _pt_add(p1: Optional[tuple[int, int]], p2: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    if p1[0] == p2[0] and (p1[1] + p2[1]) % _P == 0:
        return None
    if p1 == p2:
        lam = (3 * p1[0] * p1[0]) * pow(2 * p1[1], -1, _P) % _P
    else:
        lam = (p2[1] - p1[1]) * pow(p2[0] - p1[0], -1, _P) % _P
    x = (lam * lam - p1[0] - p2[0]) % _P
    return x, (lam * (p1[0] - x) - p1[1]) % _P


def _pt_mul(k: int, pt: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
    acc = None
    while k:
        if k & 1:
            acc = _pt_add(acc, pt)
        pt = _pt_add(pt, pt)
        k >>= 1
    return acc


def recover_evm_pubkey(msg_hash: bytes, signature: bytes) -> bytes:
    """Recover the 64-byte uncompressed public key (x||y) from a 65-byte r||s||v."""
    if len(signature) != 65 or len(msg_hash) != 32:
        raise WalletError("signature must be 65 bytes over a 32-byte hash")
    r = int.from_bytes(signature[:32], "big")
    s = int.from_bytes(signature[32:64], "big")
    v = signature[64]
    recid = v - 27 if v >= 27 else v
    if recid not in (0, 1) or not (0 < r < _N) or not (0 < s < _N):
        raise WalletError("malformed signature values")
    # EIP-2: only low-s signatures are canonical (wallets produce these).
    if s > _N // 2:
        raise WalletError("non-canonical (high-s) signature")
    x = r
    alpha = (pow(x, 3, _P) + 7) % _P
    y = pow(alpha, (_P + 1) // 4, _P)
    if (y * y) % _P != alpha:
        raise WalletError("r is not on the curve")
    if (y & 1) != recid:
        y = _P - y
    e = int.from_bytes(msg_hash, "big")
    r_inv = pow(r, -1, _N)
    q = _pt_add(_pt_mul(s, (x, y)), _pt_mul((-e) % _N, _G))
    q = _pt_mul(r_inv, q)
    if q is None:
        raise WalletError("recovered point at infinity")
    return q[0].to_bytes(32, "big") + q[1].to_bytes(32, "big")


def eip191_hash(message: str) -> bytes:
    raw = message.encode("utf-8")
    return keccak256(b"\x19Ethereum Signed Message:\n" + str(len(raw)).encode() + raw)


def to_checksum_address(address: str) -> str:
    """EIP-55 mixed-case checksum address."""
    a = address.lower().removeprefix("0x")
    if len(a) != 40 or any(c not in "0123456789abcdef" for c in a):
        raise WalletError("not an EVM address")
    h = keccak256(a.encode()).hex()
    return "0x" + "".join(c.upper() if int(h[i], 16) >= 8 else c for i, c in enumerate(a))


def recover_evm_address(message: str, signature_hex: str) -> str:
    sig = bytes.fromhex(signature_hex.removeprefix("0x"))
    pub = recover_evm_pubkey(eip191_hash(message), sig)
    return to_checksum_address("0x" + keccak256(pub)[-20:].hex())


# ---------------------------------------------------------------------------
# Solana (Solflare): base58 + ed25519
# ---------------------------------------------------------------------------

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_IDX = {c: i for i, c in enumerate(_B58)}


def b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        if ch not in _B58_IDX:
            raise WalletError("invalid base58")
        n = n * 58 + _B58_IDX[ch]
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + body


def b58encode(b: bytes) -> str:
    n = int.from_bytes(b, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    pad = len(b) - len(b.lstrip(b"\x00"))
    return "1" * pad + out


def _sig_bytes(sig: str, length: int) -> bytes:
    """Accept hex (0x optional), base64, or base58 — whichever decodes to ``length``."""
    s = (sig or "").strip()
    cands: list[bytes] = []
    try:
        cands.append(bytes.fromhex(s.removeprefix("0x")))
    except ValueError:
        pass
    try:
        cands.append(base64.b64decode(s, validate=True))
    except Exception:
        pass
    try:
        cands.append(b58decode(s))
    except WalletError:
        pass
    for c in cands:
        if len(c) == length:
            return c
    raise WalletError(f"signature must decode to {length} bytes")


def verify_solana(address: str, message: str, signature: str) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    pub = b58decode(address)
    if len(pub) != 32:
        raise WalletError("a Solana address is a 32-byte base58 public key")
    try:
        Ed25519PublicKey.from_public_bytes(pub).verify(_sig_bytes(signature, 64), message.encode("utf-8"))
        return True
    except InvalidSignature:
        return False


# ---------------------------------------------------------------------------
# Challenges (in memory, single use) and bindings (brain_kv)
# ---------------------------------------------------------------------------

@dataclass
class _Pending:
    chain: str
    address: str
    purpose: str
    domain: str
    message: str
    expires_at: float


_pending: dict[str, _Pending] = {}
_lock = threading.Lock()


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_address(chain: str, address: str) -> str:
    a = (address or "").strip()
    if chain == "evm":
        return to_checksum_address(a)
    if chain == "solana":
        if len(b58decode(a)) != 32:
            raise WalletError("a Solana address is a 32-byte base58 public key")
        return a
    raise WalletError(f"chain must be one of {CHAINS}")


def _statement(purpose: str) -> str:
    verb = "Bind this wallet to" if purpose == "bind" else "Remove this wallet's binding from"
    return (f"{verb} GARD Shard marketplace access. Signing costs nothing and "
            "authorises no transaction.")


def challenge(chain: str, address: str, *, domain: str, uri: str,
              chain_id: int | str | None = None, purpose: str = "bind",
              now: float | None = None) -> dict[str, Any]:
    if chain not in CHAINS:
        raise WalletError(f"chain must be one of {CHAINS}")
    if purpose not in PURPOSES:
        raise WalletError(f"purpose must be one of {PURPOSES}")
    addr = normalise_address(chain, address)
    t = time.time() if now is None else now
    nonce = secrets.token_hex(12)                 # EIP-4361: >= 8 alphanumerics
    exp = t + CHALLENGE_TTL_S
    account = "Ethereum" if chain == "evm" else "Solana"
    cid = str(int(chain_id)) if chain == "evm" and chain_id not in (None, "") else (
        "1" if chain == "evm" else "mainnet")
    message = (
        f"{domain} wants you to sign in with your {account} account:\n"
        f"{addr}\n\n"
        f"{_statement(purpose)}\n\n"
        f"URI: {uri}\n"
        f"Version: 1\n"
        f"Chain ID: {cid}\n"
        f"Nonce: {nonce}\n"
        f"Issued At: {_iso(t)}\n"
        f"Expiration Time: {_iso(exp)}"
    )
    with _lock:
        for k in [k for k, p in _pending.items() if p.expires_at < t]:
            _pending.pop(k, None)
        if len(_pending) >= _MAX_PENDING:
            for k in sorted(_pending, key=lambda k: _pending[k].expires_at)[: _MAX_PENDING // 4]:
                _pending.pop(k, None)
        _pending[nonce] = _Pending(chain, addr, purpose, domain, message, exp)
    return {"chain": chain, "address": addr, "purpose": purpose, "nonce": nonce,
            "message": message, "expires_at": _iso(exp)}


def _consume(chain: str, address: str, nonce: str, purpose: str, now: float | None) -> _Pending:
    with _lock:
        p = _pending.pop(str(nonce or ""), None)      # single use, verified or not
    if p is None:
        raise WalletError("unknown or already-used challenge")
    t = time.time() if now is None else now
    if p.expires_at < t:
        raise WalletError("challenge expired")
    if p.chain != chain or p.address != normalise_address(chain, address) or p.purpose != purpose:
        raise WalletError("challenge was issued for a different wallet or purpose")
    return p


def _verify(p: _Pending, signature: str) -> None:
    if p.chain == "solana":
        if not verify_solana(p.address, p.message, signature):
            raise WalletError("signature does not verify for this address")
        return
    try:
        recovered = recover_evm_address(p.message, signature)
    except ValueError as exc:
        raise WalletError(f"bad signature: {exc}") from exc
    if recovered != p.address:
        raise WalletError("signature was made by a different address")


def _index_set(chain: str, address: str, row: Optional[dict[str, Any]]) -> None:
    idx = brain_kv.kv_get_json(_INDEX, {}) or {}
    k = f"{chain}:{address}"
    if row is None:
        idx.pop(k, None)
    else:
        idx[k] = row
    brain_kv.kv_set_json(_INDEX, idx)


def bind(chain: str, address: str, nonce: str, signature: str, *,
         now: float | None = None) -> dict[str, Any]:
    p = _consume(chain, address, nonce, "bind", now)
    _verify(p, signature)
    t = time.time() if now is None else now
    assignment = {"kind": BINDING_KIND, "chain": p.chain, "address": p.address,
                  "domain": p.domain, "nonce": nonce, "bound_at": _iso(t)}
    proof: Any
    try:
        from . import gard_shard_model
        proof = gard_shard_model.sign_assignment_proof(assignment)
    except Exception as exc:      # grid secret unset: bind, but say the proof is absent
        proof = None
        assignment["proof_unavailable"] = type(exc).__name__
    record = {**assignment, "proof": proof}
    brain_kv.kv_set_json(_KEY.format(chain=p.chain, address=p.address), record)
    _index_set(p.chain, p.address, {"bound_at": record["bound_at"], "proof": proof is not None})
    return {"ok": True, "binding": record}


def unbind(chain: str, address: str, nonce: str, signature: str, *,
           now: float | None = None) -> dict[str, Any]:
    p = _consume(chain, address, nonce, "unbind", now)
    _verify(p, signature)
    key = _KEY.format(chain=p.chain, address=p.address)
    had = brain_kv.kv_get_json(key, None) is not None
    brain_kv.kv_set_json(key, None)
    _index_set(p.chain, p.address, None)
    return {"ok": True, "unbound": had, "chain": p.chain, "address": p.address}


def get_binding(chain: str, address: str) -> Optional[dict[str, Any]]:
    return brain_kv.kv_get_json(_KEY.format(chain=chain, address=normalise_address(chain, address)), None)


def index() -> dict[str, Any]:
    return brain_kv.kv_get_json(_INDEX, {}) or {}


def verify_binding_proof(record: dict[str, Any]) -> bool:
    """For the marketplace: does a binding carry a valid GARD assignment proof?"""
    proof = (record or {}).get("proof")
    if not proof or not isinstance(proof, dict):
        return False
    a = proof.get("assignment") or {}
    # A valid proof for some OTHER wallet must not vouch for this record.
    if (a.get("kind") != BINDING_KIND or a.get("chain") != record.get("chain")
            or a.get("address") != record.get("address")):
        return False
    try:
        from . import gard_shard_model
        return bool(gard_shard_model.verify_assignment_proof(proof))
    except Exception:
        return False
