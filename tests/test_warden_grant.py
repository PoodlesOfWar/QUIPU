"""Warden grants: signed, expiring, revocable. Every failure mode must DENY."""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.quipu.warden import (
    ACTION_CLASSES,
    Grant,
    GrantStore,
    MemoryStore,
    StaticWardenRegistry,
    WardenAuthorityError,
    WardenFormatError,
    WardenSecretError,
    issue_grant,
    parse_target,
    verify_grant,
)
from src.quipu.warden._common import derive_key, hmac_sign, load_secret
from warden_support import NOW, SECRET, WARDEN

TARGET = "npc:guard-7"
WARDENS = StaticWardenRegistry([WARDEN])


def _grant(**kw) -> Grant:
    args = dict(warden_id=WARDEN, agent_id="agent-1", target=TARGET, scope=["npc.act"],
                ttl_s=600, registry=WARDENS, secret=SECRET, now=NOW)
    args.update(kw)
    return issue_grant(**args)


def _verify(grant, store=None, **kw):
    args = dict(agent_id="agent-1", target=TARGET, action="npc.act", now=NOW + 1,
                store=store or GrantStore(), registry=WARDENS, secret=SECRET)
    args.update(kw)
    return verify_grant(grant, **args)


# ── the allow path ───────────────────────────────────────────────────────────

def test_valid_grant_allows():
    assert _verify(_grant()) == (True, "ok")


def test_grant_round_trips_through_dict():
    grant = _grant(scope=["npc.act", "npc.bind"])
    assert Grant.from_dict(grant.to_dict()) == grant
    assert _verify(Grant.from_dict(grant.to_dict())) == (True, "ok")


# ── tampering ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("scope", ("npc.act", "npc.bind")),          # widen the scope
    ("expires_at", NOW + 10_000_000),             # extend the lifetime
    ("agent_id", "agent-2"),                      # hand the grant to someone else
    ("target", "npc:other"),                      # retarget it
    ("warden_id", "warden-2"),
    ("grant_id", "forged-id"),
])
def test_any_tampered_field_breaks_the_signature(field, value):
    forged = replace(_grant(), **{field: value})
    ok, reason = _verify(forged, agent_id=forged.agent_id, target=forged.target)
    assert (ok, reason) == (False, "bad_signature")


@pytest.mark.parametrize("signature", ["", "0" * 64, "zz", None, 123, "é" * 64])
def test_bad_signatures_deny_without_raising(signature):
    forged = replace(_grant(), signature=signature)
    assert _verify(forged) == (False, "bad_signature")


def test_signature_from_a_different_secret_is_rejected():
    foreign = _grant(secret="a-completely-different-secret-value")
    assert _verify(foreign) == (False, "bad_signature")


# ── time ─────────────────────────────────────────────────────────────────────

def test_expired_grant_denies_at_and_after_expiry():
    grant = _grant(ttl_s=600)
    assert _verify(grant, now=NOW + 599) == (True, "ok")
    assert _verify(grant, now=NOW + 600) == (False, "expired")
    assert _verify(grant, now=NOW + 10_000) == (False, "expired")


def test_grant_from_the_future_denies():
    grant = _grant(now=NOW + 3600)
    assert _verify(grant, now=NOW) == (False, "not_yet_valid")


@pytest.mark.parametrize("ttl", [0, -1, 7 * 24 * 3600 + 1, True, "600", float("nan")])
def test_ttl_out_of_range_is_refused_at_issue(ttl):
    with pytest.raises(WardenFormatError):
        _grant(ttl_s=ttl)


# ── revocation ───────────────────────────────────────────────────────────────

def test_revoked_grant_denies_immediately():
    store = GrantStore()
    grant = _grant()
    assert _verify(grant, store) == (True, "ok")
    store.revoke(grant.grant_id, warden_id=WARDEN, registry=WARDENS, secret=SECRET, now=NOW)
    assert _verify(grant, store) == (False, "revoked")


def test_any_revocation_record_counts_even_if_unsigned():
    """A forged record can only deny, which is the safe direction."""
    backing = MemoryStore()
    store = GrantStore(backing)
    grant = _grant()
    backing.set(f"warden:revocation:{grant.grant_id}", {"junk": True})
    assert _verify(grant, store) == (False, "revoked")


def test_only_a_warden_can_revoke():
    with pytest.raises(WardenAuthorityError):
        GrantStore().revoke("g1", warden_id="mallory", registry=WARDENS, secret=SECRET)


def test_revoke_listener_fires_and_a_broken_listener_cannot_undo_it():
    store = GrantStore()
    seen: list[str] = []
    store.add_revoke_listener(lambda gid: (_ for _ in ()).throw(RuntimeError("boom")))
    store.add_revoke_listener(seen.append)
    result = store.revoke("g-listen", warden_id=WARDEN, registry=WARDENS, secret=SECRET, now=NOW)
    assert seen == ["g-listen"]
    assert result["listener_errors"] == 1
    assert store.is_revoked("g-listen")


# ── who may issue ────────────────────────────────────────────────────────────

def test_non_warden_cannot_issue():
    with pytest.raises(WardenAuthorityError):
        _grant(warden_id="mallory")


def test_empty_warden_registry_means_nothing_can_be_issued():
    with pytest.raises(WardenAuthorityError):
        _grant(registry=StaticWardenRegistry([]))


def test_demoted_warden_invalidates_the_grants_they_signed():
    grant = _grant()
    assert _verify(grant, registry=StaticWardenRegistry([])) == (False, "warden_not_authorized")


# ── binding to agent / target / action ───────────────────────────────────────

def test_wrong_agent_denies():
    assert _verify(_grant(), agent_id="agent-2") == (False, "agent_mismatch")


def test_wrong_target_denies():
    assert _verify(_grant(), target="npc:other") == (False, "target_mismatch")


def test_action_outside_scope_denies():
    grant = _grant(scope=["npc.act"])
    assert _verify(grant, action="npc.bind") == (False, "action_not_in_scope")


@pytest.mark.parametrize("action", ["", "npc.*", "*", "npc.act ", "game.act", "nonsense"])
def test_unknown_or_wildcard_actions_deny(action):
    ok, reason = _verify(_grant(), action=action)
    assert not ok and reason in {"unknown_action", "action_not_in_scope"}


# ── scope and target validation at issue time ────────────────────────────────

@pytest.mark.parametrize("scope", [[], "npc.act", ["not.an.action"], ["game.act"], ["robot.move"], ["npc.*"]])
def test_bad_scope_is_refused_at_issue(scope):
    with pytest.raises(WardenFormatError):
        _grant(scope=scope)


@pytest.mark.parametrize("target", [
    "", "npc:", "reachy2:", "game:", "game:wow_legacy", "game:wow_legacy:", "game:guild_wars:x",
    "game:retail_wow", "npc:a b", "npc:a\n", "reachy2:host/../x", "reachy2:ho st", "NPC:guard", None, 5,
])
def test_malformed_targets_are_refused(target):
    with pytest.raises(WardenFormatError):
        parse_target(target)


@pytest.mark.parametrize("target,expected", [
    ("reachy2:10.0.0.5", ("robot", "10.0.0.5")),
    ("reachy2:reachy.local", ("robot", "reachy.local")),
    ("npc:guard-7", ("npc", "guard-7")),
    ("game:guild_wars", ("game", "guild_wars")),
    ("game:wow_legacy:hideout-01", ("game", "wow_legacy:hideout-01")),
])
def test_valid_targets_parse(target, expected):
    assert parse_target(target) == expected


def test_each_target_kind_accepts_only_its_own_action_classes():
    """A robot grant can never be stretched into a game action."""
    for kind, target in [("robot", "reachy2:h"), ("npc", "npc:n"), ("game", "game:guild_wars")]:
        for other_kind, actions in ACTION_CLASSES.items():
            if other_kind == kind:
                continue
            with pytest.raises(WardenFormatError):
                _grant(target=target, scope=[sorted(actions)[0]])


def test_malformed_stored_grant_reads_as_missing():
    backing = MemoryStore()
    backing.set("warden:grant:g1", {"grant_id": "g1", "scope": "not-a-list"})
    assert GrantStore(backing).get("g1") is None
    assert GrantStore(backing).get("../../etc") is None


# ── secrets: fail closed, no dev default ─────────────────────────────────────

def test_missing_secret_denies_verification(monkeypatch):
    monkeypatch.delenv("WARDEN_GRANT_SECRET", raising=False)
    monkeypatch.delenv("SCBRAIN_GRID_SECRET", raising=False)
    grant = _grant()
    ok, reason = verify_grant(
        grant, agent_id="agent-1", target=TARGET, action="npc.act", now=NOW + 1,
        store=GrantStore(), registry=WARDENS, secret=None,
    )
    assert (ok, reason) == (False, "secret_unset")


def test_missing_secret_refuses_issue_and_revoke(monkeypatch):
    monkeypatch.delenv("WARDEN_GRANT_SECRET", raising=False)
    monkeypatch.delenv("SCBRAIN_GRID_SECRET", raising=False)
    with pytest.raises(WardenSecretError):
        issue_grant(warden_id=WARDEN, agent_id="agent-1", target=TARGET, scope=["npc.act"],
                    ttl_s=60, registry=WARDENS, secret=None, now=NOW)
    with pytest.raises(WardenSecretError):
        GrantStore().revoke("g1", warden_id=WARDEN, registry=WARDENS, secret=None)


@pytest.mark.parametrize("blank", ["", "   ", b"", b"  \n"])
def test_blank_secret_counts_as_unset(blank, monkeypatch):
    monkeypatch.delenv("WARDEN_GRANT_SECRET", raising=False)
    monkeypatch.delenv("SCBRAIN_GRID_SECRET", raising=False)
    with pytest.raises(WardenSecretError):
        load_secret(blank)


def test_secret_env_precedence(monkeypatch):
    monkeypatch.setenv("SCBRAIN_GRID_SECRET", "grid-secret")
    assert load_secret() == b"grid-secret"
    monkeypatch.setenv("WARDEN_GRANT_SECRET", "warden-secret")
    assert load_secret() == b"warden-secret"
    assert load_secret("explicit") == b"explicit"


def test_grants_work_from_the_environment_alone(monkeypatch):
    monkeypatch.setenv("WARDEN_GRANT_SECRET", SECRET)
    grant = issue_grant(warden_id=WARDEN, agent_id="agent-1", target=TARGET, scope=["npc.act"],
                        ttl_s=60, registry=WARDENS, now=NOW)
    assert verify_grant(grant, agent_id="agent-1", target=TARGET, action="npc.act", now=NOW + 1,
                        store=GrantStore(), registry=WARDENS) == (True, "ok")


# ── domain separation ────────────────────────────────────────────────────────

def test_signatures_are_bound_to_their_purpose():
    """One configured secret must not let a signature made for one purpose pass as another."""
    key = load_secret(SECRET)
    payload = _grant().payload()
    assert hmac_sign(key, "grant", payload) != hmac_sign(key, "revocation", payload)
    assert hmac_sign(key, "grant", payload) != hmac_sign(key, "audit", payload)
    assert hmac_sign(key, "grant", payload) != hmac_sign(key, "record", payload)
    # A grant's signature must not validate as a grant for a different purpose label.
    forged = replace(_grant(), signature=hmac_sign(key, "record", _grant().payload()))
    assert _verify(forged) == (False, "bad_signature")


def test_each_purpose_gets_its_own_subkey():
    """The configured secret is never used directly: a leaked or misused subkey for
    one purpose does not sign for another."""
    secret = load_secret(SECRET)
    keys = {label: derive_key(secret, label) for label in ("grant", "revocation", "audit", "record")}
    assert len(set(keys.values())) == len(keys)
    assert secret not in keys.values()
