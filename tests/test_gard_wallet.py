"""GARD Shard marketplace wallet connectors: Solflare (Solana ed25519) and
MetaMask (EIP-191 / EIP-4361).  EVM vectors were produced by eth_account
0.14.0 (not by this module), so recovery is checked against an independent
signer."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from src.quipu import gard_wallet as gw
from src.quipu import observer_service as osvc
from src.quipu.qpsi import edge_admission

# eth_account.Account.from_key("0x" + "11"*32)
EVM_ADDR = "0x19E7E376E7C213B7E7e7e46cc70A5dD086DAff2A"
FROZEN_MSG = "frozen vector for gard_wallet tests"
FROZEN_SIG = ("0x6ca1234c6d4524394d68659f568d017d288a08929bf6c77fab51d4225f54b582"
              "7e9fdd3a827925a2b6edaa49e553510f5a36dee102a71ccf6d0e826b1ba6a56b1c")
NOW = 1767225600.0          # 2026-01-01T00:00:00Z; challenges below are pinned to it
BIND_NONCE, BIND_SIG = "abcdef0123456789abcdef01", (
    "0xa5925b979f357fef09450e4e03fee6ff667afca2fa4da52a3d93fd40f81c2701"
    "05293b4959aa9ac1ba03f520fb258e2f4b4c41eaf9b5f2f994f0168556a737f71b")
UNBIND_NONCE, UNBIND_SIG = "fedcba9876543210fedcba98", (
    "0x7554de4fb02ea52378e0b37454e06977b6238986274a75512a2675860558081a"
    "67b7da4c2c4776debf44bf12d3c4850458306bb4b7d87495b882fd0e2de9fcf41b")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    gw._pending.clear()
    monkeypatch.delenv("SCBRAIN_GRID_SECRET", raising=False)
    yield
    gw._pending.clear()


def _evm_challenge(monkeypatch, purpose, nonce):
    monkeypatch.setattr(gw.secrets, "token_hex", lambda n=12: nonce)
    return gw.challenge("evm", EVM_ADDR.lower(), domain="127.0.0.1:7100",
                        uri="http://127.0.0.1:7100", chain_id=1, purpose=purpose, now=NOW)


def _sol_wallet():
    sk = Ed25519PrivateKey.generate()
    pub = sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return sk, gw.b58encode(pub)


# ---------------------------------------------------------------- primitives
def test_keccak256_known_vectors():
    assert gw.keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert gw.keccak256(b"abc").hex() == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"


def test_eip55_checksum():
    assert gw.to_checksum_address(EVM_ADDR.lower()) == EVM_ADDR
    # EIP-55 spec example
    assert gw.to_checksum_address("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed") == \
        "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
    with pytest.raises(gw.WalletError):
        gw.to_checksum_address("0x1234")


def test_evm_recovery_matches_independent_signer():
    assert gw.recover_evm_address(FROZEN_MSG, FROZEN_SIG) == EVM_ADDR
    assert gw.recover_evm_address(FROZEN_MSG + ".", FROZEN_SIG) != EVM_ADDR   # message bound


def test_evm_rejects_high_s_and_malformed():
    sig = bytearray.fromhex(FROZEN_SIG[2:])
    s = int.from_bytes(sig[32:64], "big")
    sig[32:64] = (gw._N - s).to_bytes(32, "big")       # the malleable twin
    sig[64] ^= 1
    with pytest.raises(gw.WalletError):
        gw.recover_evm_address(FROZEN_MSG, "0x" + sig.hex())
    with pytest.raises(gw.WalletError):
        gw.recover_evm_pubkey(b"\x00" * 32, b"\x00" * 65)


def test_base58_roundtrip():
    for raw in (b"\x00\x00\x01\x02", bytes(range(32)), b"\xff" * 32):
        assert gw.b58decode(gw.b58encode(raw)) == raw


# ---------------------------------------------------------------- Solflare
def test_solana_bind_and_unbind():
    sk, addr = _sol_wallet()
    ch = gw.challenge("solana", addr, domain="127.0.0.1:7100", uri="http://127.0.0.1:7100")
    assert "sign in with your Solana account" in ch["message"] and addr in ch["message"]
    assert "authorises no transaction" in ch["message"]
    sig = sk.sign(ch["message"].encode()).hex()
    out = gw.bind("solana", addr, ch["nonce"], sig)
    rec = out["binding"]
    assert rec["address"] == addr and rec["kind"] == gw.BINDING_KIND
    assert rec["proof"] is None and rec["proof_unavailable"]          # no grid secret
    assert gw.get_binding("solana", addr)["nonce"] == ch["nonce"]
    assert f"solana:{addr}" in gw.index()

    ch2 = gw.challenge("solana", addr, domain="d", uri="http://d", purpose="unbind")
    out = gw.unbind("solana", addr, ch2["nonce"], sk.sign(ch2["message"].encode()).hex())
    assert out["unbound"] is True and gw.get_binding("solana", addr) is None
    assert f"solana:{addr}" not in gw.index()


def test_solana_wrong_key_fails_and_nonce_is_single_use():
    sk, addr = _sol_wallet()
    other, _ = _sol_wallet()
    ch = gw.challenge("solana", addr, domain="d", uri="http://d")
    with pytest.raises(gw.WalletError):
        gw.bind("solana", addr, ch["nonce"], other.sign(ch["message"].encode()).hex())
    with pytest.raises(gw.WalletError, match="already-used"):        # burnt even on failure
        gw.bind("solana", addr, ch["nonce"], sk.sign(ch["message"].encode()).hex())


def test_challenge_is_bound_to_wallet_purpose_and_time():
    sk, addr = _sol_wallet()
    _, other_addr = _sol_wallet()
    ch = gw.challenge("solana", addr, domain="d", uri="http://d", now=NOW)
    with pytest.raises(gw.WalletError, match="different wallet"):
        gw.bind("solana", other_addr, ch["nonce"], sk.sign(ch["message"].encode()).hex(), now=NOW)
    ch = gw.challenge("solana", addr, domain="d", uri="http://d", now=NOW)
    with pytest.raises(gw.WalletError, match="different wallet or purpose"):
        gw.unbind("solana", addr, ch["nonce"], sk.sign(ch["message"].encode()).hex(), now=NOW)
    ch = gw.challenge("solana", addr, domain="d", uri="http://d", now=NOW)
    with pytest.raises(gw.WalletError, match="expired"):
        gw.bind("solana", addr, ch["nonce"], sk.sign(ch["message"].encode()).hex(),
                now=NOW + gw.CHALLENGE_TTL_S + 1)


# ---------------------------------------------------------------- MetaMask
def test_evm_bind_with_eip4361_message_and_grid_proof(monkeypatch):
    monkeypatch.setenv("SCBRAIN_GRID_SECRET", "test-grid-secret-for-gard-wallet-binding")
    ch = _evm_challenge(monkeypatch, "bind", BIND_NONCE)
    assert ch["message"].startswith("127.0.0.1:7100 wants you to sign in with your Ethereum account:\n"
                                    + EVM_ADDR + "\n")
    for line in ("URI: http://127.0.0.1:7100", "Version: 1", "Chain ID: 1",
                 f"Nonce: {BIND_NONCE}", "Issued At: 2026-01-01T00:00:00Z",
                 "Expiration Time: 2026-01-01T00:05:00Z"):
        assert line in ch["message"]
    out = gw.bind("evm", EVM_ADDR.lower(), BIND_NONCE, BIND_SIG, now=NOW + 10)
    rec = out["binding"]
    assert rec["address"] == EVM_ADDR and rec["proof"]["assignment_sha256"]
    assert gw.verify_binding_proof(rec) is True
    forged = {**rec, "address": "0x" + "00" * 20}               # proof must not vouch for another wallet
    assert gw.verify_binding_proof(forged) is False

    ch = _evm_challenge(monkeypatch, "unbind", UNBIND_NONCE)
    assert gw.unbind("evm", EVM_ADDR, UNBIND_NONCE, UNBIND_SIG, now=NOW + 20)["unbound"] is True


def test_evm_signature_from_another_address_is_refused(monkeypatch):
    _evm_challenge(monkeypatch, "bind", BIND_NONCE)
    with pytest.raises(gw.WalletError):
        gw.bind("evm", EVM_ADDR, BIND_NONCE, FROZEN_SIG, now=NOW + 1)   # valid sig, wrong message


# ---------------------------------------------------------------- observer routes
@pytest.fixture()
def observer(monkeypatch):
    monkeypatch.setenv(gw.DOMAIN_ENV, "127.0.0.1:7100")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), osvc.ObserverHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_flow_solflare(observer):
    sk, addr = _sol_wallet()
    code, ch = _post(observer + "/gard/wallet/challenge", {"chain": "solana", "address": addr})
    assert code == 200 and ch["message"].startswith("127.0.0.1:7100 wants you")
    code, out = _post(observer + "/gard/wallet/bind", {
        "source": "gard-marketplace", "chain": "solana", "address": addr,
        "nonce": ch["nonce"], "signature": sk.sign(ch["message"].encode()).hex()})
    assert code == 200 and out["binding"]["address"] == addr
    with urllib.request.urlopen(f"{observer}/gard/wallet?chain=solana&address={addr}", timeout=10) as r:
        got = json.loads(r.read())
    assert got["binding"]["address"] == addr and got["proof_valid"] is False
    code, out = _post(observer + "/gard/wallet/bind", {
        "source": "gard-marketplace", "chain": "solana", "address": addr,
        "nonce": ch["nonce"], "signature": "00" * 64})
    assert code == 401                                        # replay of a used nonce


def test_http_bad_inputs_and_page(observer):
    code, out = _post(observer + "/gard/wallet/challenge", {"chain": "btc", "address": "x"})
    assert code == 400
    code, out = _post(observer + "/gard/wallet/challenge", {"chain": "evm", "address": "0xnothex"})
    assert code == 400
    try:
        urllib.request.urlopen(f"{observer}/gard/wallet?chain=solana&address={gw.b58encode(bytes(32))}",
                               timeout=10)
        assert False, "expected 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404
    with urllib.request.urlopen(observer + "/gard/wallet/connect", timeout=10) as r:
        page = r.read().decode()
        csp = r.headers["Content-Security-Policy"]
    assert "Solflare" in page and "MetaMask" in page and "eip6963:requestProvider" in page
    assert "personal_sign" in page and "innerHTML" not in page
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp


def test_wallet_routes_are_behind_the_edge_and_source_profile():
    assert ("POST", "/gard/wallet/bind") in edge_admission.MUTATING
    assert ("POST", "/gard/wallet/unbind") in edge_admission.MUTATING
    assert ("POST", "/gard/wallet/challenge") not in edge_admission.MUTATING   # in memory only
    assert osvc._canonical_source("gard") == "gard-marketplace"
    assert osvc._canonical_source("marketplace") is None      # not the Internal Marketplace
