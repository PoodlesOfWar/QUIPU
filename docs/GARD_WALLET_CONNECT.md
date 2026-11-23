# GARD Shard marketplace wallet connectors — Solflare and MetaMask

`src/quipu/gard_wallet.py` lets a wallet holder bind their address to GARD
Shard marketplace access. The wallet proves control by **signing a sign-in
message**; QUIPU verifies it and, when the grid secret is configured, issues a
**GARD assignment proof** (`gard_shard_model.sign_assignment_proof`) the
marketplace can check.

No key ever leaves a wallet, no transaction is built, nothing is paid: signing
text is the only thing a wallet is asked to do.

| wallet | chain | what is signed | how QUIPU verifies |
|---|---|---|---|
| **Solflare** | Solana | UTF-8 message via `signMessage(bytes, "utf8")` | ed25519 against the base58 address (which *is* the public key) |
| **MetaMask** | Ethereum / EVM | EIP-4361 Sign-In-with-Ethereum text via `personal_sign` (EIP-191) | secp256k1 public-key recovery + keccak-256 → address must equal the claimed one; low-s only (EIP-2) |

keccak-256, EIP-55 and secp256k1 recovery are implemented in the module (stdlib
`sha3_256` is not keccak). They were cross-checked against `eth_account` 0.14.0:
keccak at every block-boundary length, 50/50 EIP-55 addresses, 25/25 recovered
SIWE signers; the repo tests carry frozen `eth_account` vectors so the check is
independent of this code's own maths.

## Flow

```
browser ── POST /gard/wallet/challenge {chain, address, chain_id?, purpose}
        ◄─ {message, nonce, expires_at}              single use · 5 min · in memory
wallet signs message
browser ── POST /gard/wallet/bind {source:"gard-marketplace", chain, address, nonce, signature}
        ◄─ {binding: {kind, chain, address, domain, bound_at, proof}}
```

- `POST /gard/wallet/unbind` — same, with a challenge issued for
  `purpose: "unbind"`. Only the wallet itself can remove its binding.
- `GET /gard/wallet` — index; `?chain=&address=` — one binding plus
  `proof_valid` (a proof only counts when its assignment names *this* wallet).
- `GET /gard/wallet/connect` — the connector page (Solflare + MetaMask, EIP-6963
  discovery so MetaMask is found beside other injected wallets), served
  same-origin with a strict CSP (`default-src 'none'`, `connect-src 'self'`,
  `frame-ancestors 'none'`), no third-party scripts.

Replay and relay: each nonce is burnt on first use (verified or not), expires,
and is bound to chain, address, purpose and domain. The EIP-4361 framing makes
MetaMask warn when a message's domain differs from the requesting site.

`bind`/`unbind` are edge writes (`edge_admission.MUTATING`) under the source
**`gard-marketplace`**: under `QUIPU_EDGE_AUTH=enforce` bind the page's origin
to that source in the grant's `browser_origins`. The challenge writes nothing to
the brain.

## Configuration

| variable | meaning |
|---|---|
| `QUIPU_GARD_WALLET_DOMAIN` | the domain wallets sign for (set it to the public host when tunnelled; default: request `Host`) |
| `SCBRAIN_GRID_SECRET` | enables GARD assignment proofs on bindings; unset ⇒ bindings are kept with `proof: null` and `proof_unavailable` |

Storage: `gard:wallet:<chain>:<address>` and `gard:wallet:index` in brain_kv.
Input to no gate.

## For the marketplace

```python
from src.quipu import gard_wallet
rec = gard_wallet.get_binding("evm", "0x…")      # or GET /gard/wallet?chain=evm&address=0x…
if rec and gard_wallet.verify_binding_proof(rec):
    ...  # wallet holds a QUIPU-issued GARD assignment
```
