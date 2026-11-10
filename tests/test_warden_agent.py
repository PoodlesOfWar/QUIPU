"""Signed agents: identity, declaration of automation, replay protection."""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.quipu.warden import (
    AUTOMATED_DECLARATION,
    AgentIdentity,
    AgentRegistry,
    NonceCache,
    SignedRequest,
    StaticWardenRegistry,
    WardenAuthorityError,
    WardenFormatError,
    verify_request,
)
from src.quipu.warden.signed_agent import SMALL_ORDER_Y, ed25519_verify, is_small_order_key
from warden_support import NOW, WARDEN

WARDENS = StaticWardenRegistry([WARDEN])


@pytest.fixture
def setup():
    agents = AgentRegistry()
    agent = AgentIdentity.generate("agent-1")
    agents.register("agent-1", agent.public_key_hex, warden_id=WARDEN, registry=WARDENS)
    return agents, agent, NonceCache()


def _req(agent, **kw):
    args = dict(action="npc.act", target="npc:n1", grant_id="g1", params={"intent": "idle"}, now=NOW)
    args.update(kw)
    return agent.sign_request(**args)


def _verify(agents, nonces, request, now=NOW):
    return verify_request(request, agents=agents, nonces=nonces, now=now)


def test_valid_request_verifies(setup):
    agents, agent, nonces = setup
    assert _verify(agents, nonces, _req(agent)) == (True, "ok")


# ── declaration: automation is declared and cannot be stripped ───────────────

def test_request_declares_automation_by_default(setup):
    _, agent, _ = setup
    request = _req(agent)
    assert request.automated is True and request.declaration == AUTOMATED_DECLARATION
    assert request.disclosure() == {
        "automated": True, "declaration": AUTOMATED_DECLARATION,
        "agent_id": "agent-1", "grant_id": "g1",
    }


@pytest.mark.parametrize("override", [
    {"automated": False},
    {"declaration": ""},
    {"declaration": "human"},
])
def test_a_request_that_does_not_declare_automation_is_denied(setup, override):
    agents, agent, nonces = setup
    request = replace(_req(agent), **override)
    assert _verify(agents, nonces, request) == (False, "not_declared_automated")


def test_the_declaration_is_inside_the_signature(setup):
    """Even a correctly re-signed 'automated: false' is refused; and flipping the
    field on a signed request without re-signing breaks the signature."""
    agents, agent, nonces = setup
    request = _req(agent)
    flipped = replace(request, declaration=AUTOMATED_DECLARATION + "!")
    assert _verify(agents, nonces, flipped) == (False, "not_declared_automated")
    # The payload really does include the fields (so they cannot be dropped silently).
    assert {"automated", "declaration"} <= set(request.payload())


# ── tampering and identity ───────────────────────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("action", "npc.bind"),
    ("target", "npc:other"),
    ("grant_id", "g2"),
    ("params", {"intent": "look_at", "x": 1, "y": 0, "z": 0}),
    ("issued_at", NOW + 1),
    ("nonce", "N" * 24),
])
def test_tampering_with_any_signed_field_breaks_the_signature(setup, field, value):
    agents, agent, nonces = setup
    forged = replace(_req(agent), **{field: value})
    assert _verify(agents, nonces, forged) == (False, "bad_signature")


def test_signature_from_a_different_key_is_rejected(setup):
    agents, _, nonces = setup
    impostor = AgentIdentity.generate("agent-1")  # same id, different key
    assert _verify(agents, nonces, _req(impostor)) == (False, "bad_signature")


def test_unknown_agent_is_denied(setup):
    agents, _, nonces = setup
    stranger = AgentIdentity.generate("agent-stranger")
    assert _verify(agents, nonces, _req(stranger)) == (False, "unknown_agent")


def test_revoked_agent_is_denied(setup):
    agents, agent, nonces = setup
    agents.revoke("agent-1", warden_id=WARDEN, registry=WARDENS)
    assert _verify(agents, nonces, _req(agent)) == (False, "agent_revoked")


@pytest.mark.parametrize("junk", ["", "zz", "0" * 64, None, 5, "é" * 128])
def test_garbage_signatures_deny_without_raising(setup, junk):
    agents, agent, nonces = setup
    assert _verify(agents, nonces, replace(_req(agent), signature=junk)) == (False, "bad_signature")


def test_ed25519_verify_never_raises_on_garbage():
    for pub, sig in [(None, None), ("", ""), ("zz", "zz"), ("00" * 32, "00" * 64), (5, 5), ("ab", "cd")]:
        assert ed25519_verify(pub, b"msg", sig) is False


# ── freshness and replay ─────────────────────────────────────────────────────

def test_stale_and_future_requests_are_denied(setup):
    agents, agent, nonces = setup
    assert _verify(agents, nonces, _req(agent, now=NOW - 31)) == (False, "stale_request")
    assert _verify(agents, nonces, _req(agent, now=NOW + 31)) == (False, "stale_request")
    assert _verify(agents, nonces, _req(agent, now=NOW - 30)) == (True, "ok")


def test_a_replayed_request_is_denied(setup):
    agents, agent, nonces = setup
    request = _req(agent)
    assert _verify(agents, nonces, request) == (True, "ok")
    assert _verify(agents, nonces, request) == (False, "replay")


def test_a_forged_request_cannot_burn_a_real_agents_nonce(setup):
    """The nonce is consumed only after the signature checks out."""
    agents, agent, nonces = setup
    real = _req(agent, nonce="N" * 24)
    forged = replace(real, params={"intent": "look_at"})  # bad signature, same nonce
    assert _verify(agents, nonces, forged) == (False, "bad_signature")
    assert _verify(agents, nonces, real) == (True, "ok")


def test_same_nonce_is_distinct_per_agent(setup):
    agents, agent, nonces = setup
    other = AgentIdentity.generate("agent-2")
    agents.register("agent-2", other.public_key_hex, warden_id=WARDEN, registry=WARDENS)
    assert _verify(agents, nonces, _req(agent, nonce="N" * 24)) == (True, "ok")
    assert _verify(agents, nonces, _req(other, nonce="N" * 24)) == (True, "ok")


def test_a_full_nonce_cache_refuses_rather_than_evicting(setup):
    """Evicting a live nonce would reopen the replay window."""
    agents, agent, _ = setup
    tiny = NonceCache(max_entries=2)
    assert _verify(agents, tiny, _req(agent, nonce="A" * 24)) == (True, "ok")
    assert _verify(agents, tiny, _req(agent, nonce="B" * 24)) == (True, "ok")
    assert _verify(agents, tiny, _req(agent, nonce="C" * 24)) == (False, "nonce_cache_full")
    # ...and the earlier nonces are still remembered, not evicted.
    assert _verify(agents, tiny, _req(agent, nonce="A" * 24)) == (False, "replay")


def test_expired_nonces_are_reclaimed(setup):
    agents, agent, _ = setup
    cache = NonceCache(max_entries=2)
    assert _verify(agents, cache, _req(agent, nonce="A" * 24)) == (True, "ok")
    assert _verify(agents, cache, _req(agent, nonce="B" * 24)) == (True, "ok")
    later = NOW + 1000
    assert _verify(agents, cache, _req(agent, nonce="C" * 24, now=later), now=later) == (True, "ok")


@pytest.mark.parametrize("nonce", ["", "short", "has space " * 3, "x" * 200, "bad/chars/in/nonce/xx"])
def test_malformed_nonces_are_denied(setup, nonce):
    """Hand-built and validly signed, so only the nonce format is at fault.

    (``sign_request`` itself replaces an empty nonce with a random one, so going
    through it would not exercise the malformed case.)
    """
    agents, agent, nonces = setup
    unsigned = replace(_req(agent), nonce=nonce, signature="")
    signed = replace(unsigned, signature=agent._private_key.sign(unsigned.signing_bytes()).hex())
    assert _verify(agents, nonces, signed) == (False, "malformed")


# ── registration is an authority decision ────────────────────────────────────

def test_only_a_warden_can_register_an_agent_key():
    agent = AgentIdentity.generate("agent-9")
    with pytest.raises(WardenAuthorityError):
        AgentRegistry().register("agent-9", agent.public_key_hex, warden_id="mallory", registry=WARDENS)


def test_registration_never_silently_overwrites_a_key():
    agents = AgentRegistry()
    first, second = AgentIdentity.generate("agent-1"), AgentIdentity.generate("agent-1")
    agents.register("agent-1", first.public_key_hex, warden_id=WARDEN, registry=WARDENS)
    with pytest.raises(WardenFormatError):
        agents.register("agent-1", second.public_key_hex, warden_id=WARDEN, registry=WARDENS)
    agents.register("agent-1", second.public_key_hex, warden_id=WARDEN, registry=WARDENS, rotate=True)
    nonces = NonceCache()
    assert _verify(agents, nonces, _req(first)) == (False, "bad_signature")
    assert _verify(agents, nonces, _req(second)) == (True, "ok")


@pytest.mark.parametrize("bad_key", ["", "abc", "zz" * 32, "AB" * 32, None, 12])
def test_malformed_public_keys_are_refused(bad_key):
    with pytest.raises(WardenFormatError):
        AgentRegistry().register("agent-1", bad_key, warden_id=WARDEN, registry=WARDENS)


def test_request_round_trips_through_dict(setup):
    agents, agent, nonces = setup
    request = _req(agent)
    restored = SignedRequest.from_dict(request.to_dict())
    assert restored == request
    assert _verify(agents, nonces, restored) == (True, "ok")


@pytest.mark.parametrize("data", [None, [], "x", {"agent_id": "a", "grant_id": "g", "params": []}])
def test_malformed_request_dicts_are_refused(data):
    with pytest.raises(WardenFormatError):
        SignedRequest.from_dict(data)


# ── small-order ("weak") public keys ─────────────────────────────────────────

def _torsion_y_values() -> set[int]:
    """Re-derive the small-order y-coordinates from Ed25519 curve arithmetic.

    Independent of the constants in the module under test: finds every point P with
    8P = identity by search over a candidate pool, then proves the result is a closed
    group of order 8. Non-canonical encodings (y+p for y in {0, 1}) are added.
    """
    p = 2**255 - 19
    d = (-121665 * pow(121666, -1, p)) % p

    def add(a, b):
        (x1, y1), (x2, y2) = a, b
        t = d * x1 * x2 * y1 * y2 % p
        return ((x1 * y2 + x2 * y1) * pow(1 + t, -1, p) % p, (y1 * y2 + x1 * x2) * pow(1 - t, -1, p) % p)

    def mul(point, n):
        acc = (0, 1)
        while n:
            if n & 1:
                acc = add(acc, point)
            point = add(point, point)
            n >>= 1
        return acc

    def lift(y, sign):
        x2 = (y * y - 1) * pow(d * y * y + 1, -1, p) % p
        x = pow(x2, (p + 3) // 8, p)
        if (x * x - x2) % p:
            x = x * pow(2, (p - 1) // 4, p) % p
        if (x * x - x2) % p:
            return None
        return (x if x & 1 == sign else (-x) % p, y)

    # Seed the pool with the order-8 points, found by clearing the prime-order part:
    # take arbitrary curve points and multiply by L (the prime subgroup order).
    L = 2**252 + 27742317777372353535851937790883648493
    torsion: set[tuple[int, int]] = set()
    y = 2
    while len(torsion) < 8 and y < 400:
        for sign in (0, 1):
            point = lift(y, sign)
            if point is not None:
                torsion.add(mul(point, L))
        y += 1
    assert len(torsion) == 8, "did not recover the full torsion subgroup"
    assert all(add(a, b) in torsion for a in torsion for b in torsion), "not closed under addition"
    assert all(mul(point, 8) == (0, 1) for point in torsion)
    ys = {point[1] for point in torsion}
    return ys | {p + 0, p + 1}


def test_the_small_order_blocklist_matches_the_curve():
    assert SMALL_ORDER_Y == _torsion_y_values()


@pytest.mark.parametrize("key", [
    "00" * 32,                               # y = 0, order 4
    "01" + "00" * 31,                        # y = 1, the identity
    "ec" + "ff" * 30 + "7f",                 # y = p - 1, order 2
    "ed" + "ff" * 30 + "7f",                 # y = p, non-canonical 0
    "ee" + "ff" * 30 + "7f",                 # y = p + 1, non-canonical 1
    "00" * 31 + "80",                        # y = 0 with the sign bit set
    "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",  # order 8
    "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",  # order 8
])
def test_small_order_keys_are_flagged(key):
    assert is_small_order_key(key)


def test_a_real_key_is_not_flagged():
    assert not is_small_order_key(AgentIdentity.generate("a").public_key_hex)


def test_the_all_zero_key_cannot_forge_a_signature():
    """Regression: with OpenSSL's Ed25519 an all-zero public key plus an all-zero
    signature verifies for about one message in four, with no private key at all."""
    forged_ok = sum(ed25519_verify("00" * 32, f"message-{i}".encode(), "00" * 64) for i in range(64))
    assert forged_ok == 0


def test_a_warden_cannot_register_a_weak_agent_key():
    with pytest.raises(WardenFormatError):
        AgentRegistry().register("agent-1", "00" * 32, warden_id=WARDEN, registry=WARDENS)
