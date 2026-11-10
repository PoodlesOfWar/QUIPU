# Warden-gated embodied agents

A World 4 NPC can be played by a Pollen Robotics **Reachy 2**, and signed agents can
take part in a hosted game (**Guild Wars first, then WoW Legacy**), but only on the
strength of a record a **Warden** signed, re-checked on every action.

The code lives in `src/quipu/warden/`. It is **self-contained**: nothing in it imports
the rest of QUIPU at load time, so the directory can be lifted into whichever repo owns
World 4 (QUIPU itself has no Warden, NPC, World or Essay concepts). Its only QUIPU
touchpoint is `BrainKVStore`, which imports `brain_kv` lazily.

> **The principle.** A Warden's authorization is a *record the code checks*, never
> something the code grants itself. Every check fails closed: a missing secret, an
> unreadable store, a malformed record, an audit write that fails, or an unexpected
> exception all end in a denial, never in a default-allow.

## How a request is judged

```
 agent ──signs──▶ SignedRequest ─▶ ┌────────────────────────────── WardenGate ─────────────────────────────┐
                                   │ 1 request  Ed25519 sig · declared automated · fresh · not replayed     │
                                   │ 2 grant    Warden-signed · unexpired · unrevoked · this agent · this   │
                                   │            target · this action class · signer still a Warden          │
                                   │ 3 stage    (game targets only) Essay record + operator's own signature │
                                   │ ─▶ audit    HMAC-chained; if it can't be written the action is denied  │
                                   └──────────────────────────────┬─────────────────────────────────────────┘
                              robot/NPC: NpcEmbodiment ◀──────────┴──────────▶ game: GameDispatcher ─▶ adapter
                              (re-checks BOTH grants every act)                 (carries the automation disclosure)
```

| Module | Role |
|---|---|
| `warden_grant.py` | Signed, expiring, revocable grants; `issue_grant`, `verify_grant`, `GrantStore` |
| `signed_agent.py` | Agent identity; Ed25519-signed requests; replay protection |
| `operator_attestation.py` | The game operator's own signature (the required second key) |
| `essay_gate.py` | Progression: Guild Wars → WoW Legacy, read-only over signed records |
| `npc_embodiment.py` | NPC ↔ Reachy 2 binding, bounded intents, e-stop, `SimDriver`, `Reachy2Driver` |
| `game_adapters.py` | `GameAdapter` interface, stub adapters, `GameDispatcher` |
| `audit.py` | Append-only, tamper-evident log |
| `gate.py` | The composed gate; the single place an action is authorized |

## Grants

Fields: `grant_id, warden_id, agent_id, target, scope, issued_at, expires_at`, plus an
HMAC-SHA256 `signature` over the canonical JSON of the rest.

**Targets** and the action classes each accepts (exact match only, no wildcards):

| Target | Action classes |
|---|---|
| `reachy2:<host>` | `robot.bind` `robot.observe` `robot.move` `robot.gesture` `robot.speak` |
| `npc:<id>` | `npc.bind` `npc.act` |
| `game:guild_wars` | `game.connect` `game.act` `game.chat` |
| `game:wow_legacy:<server>` | `game.connect` `game.act` `game.chat` |

A robot grant can never be stretched into a game action: scopes are validated against the
target kind at issue time *and* again at verify time.

**Revocation is a separate signed record**, because a signature made in advance cannot
cover a flag that changes later. `verify_grant` consults it on every call, and the
presence of *any* record under a grant's revocation key counts as revoked: a forged record
can only deny, which is the safe direction.

**Who may issue.** `issue_grant` requires a `WardenRegistry` and refuses unless the named
Warden is currently registered. Verification re-checks that the signer is *still* a
Warden, so demoting a Warden invalidates the grants they signed. Grants live at most 7
days (`MAX_GRANT_TTL_S`).

**Secrets.** `WARDEN_GRANT_SECRET`, falling back to `SCBRAIN_GRID_SECRET`. There is **no
default**: unset or blank means grants cannot be issued, verified or audited. Each purpose
(grant, revocation, audit, record) signs under its own derived subkey, so one configured
secret never signs two protocols.

## Agents

Each agent holds its own **Ed25519** private key; the gate holds only the public key.
Registration needs a Warden and never silently overwrites an existing key.

Every request carries `automated: true` and `declaration: "automated-agent"` *inside the
signed payload*. A request without them is denied, and they cannot be removed without
breaking the signature. `SignedRequest.disclosure()` is the exact record an adapter must
forward to the server or its operator. Nothing here conceals that an action is automated,
and nothing should be added that does.

Requests must be within 30 s of the gate's clock and are rejected on replay. The nonce is
consumed only *after* the signature verifies, so an unauthenticated sender cannot burn a
real agent's nonces.

### Weak-key defence

OpenSSL's Ed25519 does not reject small-order public keys, and for such a key a signature
can verify with **no private key at all** (the all-zero key with an all-zero signature
verifies for roughly one message in four). Registration refuses them and the verifier
rejects them regardless. The blocklist is checked in the test suite against an independent
re-derivation from curve arithmetic, not trusted from constants.

## Operator consent

A Warden's grant says "this agent may". It cannot say "the people running this server
agree", because the Warden does not run the server. For game targets that consent must be
the **operator's own signature**, made with a key the operator holds and we do not (we
store only the public key). One target has exactly one operator; an attestation names the
agent, the target and an expiry, and is checked on every game request.

> **Deliberate tightening versus the plan.** The plan required the operator signature for
> WoW Legacy only. By default it is required for **every** game stage, Guild Wars included,
> because Guild Wars is also a service run by someone else. To loosen it, pass
> `operator_required_stages` to `EssayGate`.

This records and checks consent. It does not make a retail or third-party service
automatable, and no signature of ours changes that.

## The Essay gate

`guild_wars` opens when the required Essay test is recorded as passed. `wow_legacy` opens
only when that test is passed **and** Guild Wars is recorded complete. `EssayGate` has no
method that opens, advances or writes anything; it only reads. A stage opens because a
Warden-signed record says so.

> **Assumption to confirm (plan item 1).** Where The Essay's results actually live is
> unresolved, so the gate depends on a `ProgressSource` protocol (`test_passed(test_id)`,
> `stage_complete(stage)`), not on a storage layout. `SignedRecordProgress` is the default:
> Warden-signed records, re-verified on every read, including that the signer is still a
> Warden. To use the real data, implement `ProgressSource` over it.

## NPC ↔ Reachy 2

`bind_npc(request)` takes a signed `npc.bind` request. It requires an NPC grant
(`npc.bind` on `npc:<id>`) **and** a robot grant (`robot.bind` on `reachy2:<host>`), both
naming the requesting agent. One NPC ↔ one robot.

* **Both grants are re-verified on every `act`.** There is no cached "still authorized";
  a revoke made in another process stops the very next action.
* **What is authorized is what is executed.** The intent is read from the *signed*
  parameters.
* **Bounded action set**, validated before any driver is touched; out-of-range values are
  rejected, never clamped:

| Intent | Robot action class | Bounds |
|---|---|---|
| `look_at` | `robot.move` | x∈[0.2, 2.0], y∈[−1, 1], z∈[−0.5, 1.0] m (torso frame); duration∈[0.5, 10] s; finite numbers only |
| `gesture` | `robot.gesture` | `neutral` only (the default posture) |
| `speak` | `robot.speak` | 1–280 chars, no control characters |
| `idle` | `robot.observe` | no-op |

* **E-stop on revoke.** Revoking either grant halts the driver immediately through a
  listener on the `GrantStore`. Expiry is caught on the next `act` and by `reap()`, which
  you should call on a timer for the idle case.

> **The e-stop is a software stop.** It cancels queued motion and relaxes the robot
> gently; it is **not** a substitute for the physical emergency stop on the hardware.

### Drivers

* `SimDriver` records intents and touches nothing. It is the default and what tests use.
* `Reachy2Driver` wraps `reachy2-sdk`, imported lazily so importing the package needs
  neither the SDK nor a robot. The calls it makes were checked against **`reachy2-sdk`
  1.0.15**: `ReachySDK(host=…)`, `.connect()`, `.is_connected()`, `.turn_on()`,
  `.head.look_at(x, y, z, duration, wait)`, `.goto_posture("default")`,
  `.cancel_all_goto()`, `.turn_off_smoothly()`, `.disconnect()`. It is deliberately
  minimal: **head and neutral posture only; no arms, no mobile base.** `speak` is
  unsupported unless you inject `speak_fn`. Its e-stop is cancel → relax → release, each
  step attempted even if an earlier one fails.

## Audit

Every grant decision, bind, act and e-stop is appended to a log in which each entry
carries the HMAC of the one before it, under a dedicated audit subkey. Editing, deleting,
reordering or splicing in an entry from another history is detected by `verify_chain()`,
and a plain recomputed hash chain does not fool it.

Request parameters are logged as a SHA-256 digest, not verbatim.

**Limits, stated plainly:**

* **Tail truncation** (rolling back the newest entries *and* the head pointer together) is
  not detectable from inside the store. Anchor `AuditLog.head()` somewhere the store's
  writers cannot reach.
* **One writer.** A lock serializes appends within a process; two processes appending to
  one store could fork the chain.

## Every denial code

`Decision.code` is `layer:reason`.

| Layer | Reasons |
|---|---|
| `request` | `not_declared_automated` `unknown_agent` `agent_revoked` `malformed` `bad_signature` `stale_request` `replay` `nonce_cache_full` |
| `grant` | `grant_not_found` `secret_unset` `bad_signature` `warden_not_authorized` `revoked` `not_yet_valid` `expired` `agent_mismatch` `target_mismatch` `unknown_action` `action_not_in_scope` |
| `stage` | `essay_gate_not_configured` `not_a_game_target` `essay_test_not_passed` `guild_wars_not_complete` `secret_unset` plus the operator reasons `operator_attestation_missing` `operator_attestation_malformed` `no_operator_registered` `operator_removed` `operator_mismatch` `operator_target_mismatch` `operator_agent_mismatch` `operator_bad_signature` `operator_attestation_revoked` `operator_attestation_not_yet_valid` `operator_attestation_expired` |
| `gate` | `audit_unavailable` `internal_error:<ExceptionType>` |

Grant-layer reasons `revoked`, `expired`, `warden_not_authorized`, `bad_signature`,
`grant_not_found` and `not_yet_valid` mean the authority itself is gone and **halt** a bound
robot. Others (such as `action_not_in_scope`) only refuse the one action.

## Wiring it up

```python
from src.quipu.warden import *

store = BrainKVStore()                        # persists via QUIPU's brain_kv (SCB_DB_PATH)
wardens = StaticWardenRegistry.from_env()     # WARDEN_IDS=alice,bob
grants, agents = GrantStore(store), AgentRegistry(store)
operators = OperatorRegistry(store)
audit = AuditLog(store)                       # reads WARDEN_GRANT_SECRET
essay = EssayGate(
    required_test="essay-final",                 # the id of the Essay test you named as the last one
    progress=SignedRecordProgress(store, registry=wardens),   # or your own ProgressSource
    operators=operators,
)
gate = WardenGate(grants=grants, agents=agents, wardens=wardens, audit=audit, essay=essay)
bodies = NpcEmbodiment(gate=gate, grants=grants, store=store)   # SimDriver by default
```

To drive real hardware, pass `driver_factory=lambda host: Reachy2Driver(host)`, and only
after the manual steps below.

## Verification

```bash
python -m pytest tests/test_warden_*.py     # no network, no hardware
```

The suite covers every denial path (tampered signature, expired, revoked, wrong
agent/target, out-of-scope action, missing secret), revoke-mid-session halting the driver,
the WoW stage staying closed until Guild Wars is complete *and* the operator has attested,
and the bounded intent set. Beyond the tests, the safety-critical checks were
**mutation-tested**: 45 deliberate breakages of the code (skip the signature check, skip
expiry, fail open on a missing secret, drop the e-stop listener, …) each had to be caught
by at least one test. The first pass caught 43; the two survivors exposed a missing
splice-detection test and were closed.

**Manual, in order — do not skip ahead:**

1. Run `SimDriver` end to end with a Warden-issued grant; confirm a mid-session revoke
   halts it.
2. Point `Reachy2Driver` at **Pollen's simulator** and repeat. Re-verify the SDK calls
   against the version you deploy.
3. Only then, with a Warden-issued grant, try real hardware, with the physical e-stop
   within reach.

## Known limitations

* **Grants are HMAC (as the plan specifies), so the verifier holds the signing secret.**
  A compromised gate could mint grants. Agents and operators do not have this weakness
  (Ed25519). If the gate must not be able to forge grants, move grants to Ed25519 too; the
  change is local to `warden_grant.py`.
* The nonce cache and revoke listeners are **in-process**. Across processes, the per-act
  re-verification is what guarantees a revoke takes effect; the listener is only the fast
  path. A multi-process deployment needs a shared nonce cache.
* A robot connection is process-local, so live bindings are in-memory. After a restart
  nothing is live until `bind_npc` runs again and re-verifies both grants.
* No motion rate limit beyond the bounded intent set and the SDK's own limits.
* `GuildWarsAdapter` and `WowLegacyAdapter` are **interface stubs**; they raise
  `AdapterNotWired`.

## Hard lines

* **Do not point the adapters at retail or third-party servers.** Their terms prohibit
  bots, and nothing in this package changes that. The operator attestation is evidence of
  consent, not a way around its absence.
* **Disclosure is mandatory.** Every adapter call carries the automation disclosure and
  must refuse a call that lacks it. Do not add input-timing randomization, human-likeness
  tuning, or anything else that obscures that a client is automated.

## Open items from the plan

| # | Item | Status |
|---|---|---|
| 1 | Where World 4, the Warden role and The Essay's results live | **Open.** Abstracted behind `WardenRegistry` and `ProgressSource`. |
| 2 | Who operates the hosted WoW server | **Open.** Gate requires that operator's attestation; no adapter is wired. |
| 3 | Reachy 2: hardware or simulation | **Open.** `SimDriver` is the default; `Reachy2Driver` is written against the verified 1.0.15 API but has not touched a robot. |
