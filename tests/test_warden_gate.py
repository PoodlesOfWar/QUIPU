"""The composed gate, its audit trail, the Essay progression gate and operator consent."""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.quipu.warden import (
    AdapterNotWired,
    AuditLog,
    BrainKVStore,
    DisclosureMissing,
    GameDispatcher,
    GrantStore,
    GuildWarsAdapter,
    MemoryStore,
    OperatorRegistry,
    OperatorSigner,
    RecordingAdapter,
    SignedRecordProgress,
    StaticWardenRegistry,
    WardenAuthorityError,
    WardenFormatError,
    WowLegacyAdapter,
    require_disclosure,
    verify_attestation,
)
from src.quipu.warden.audit import GENESIS
from warden_support import NOW, REQUIRED_TEST, SECRET, WARDEN, World

GW = "game:guild_wars"
WOW = "game:wow_legacy:hideout-01"


# ═══════════════════════════ non-game targets ═══════════════════════════════

def test_valid_request_is_allowed_and_audited():
    w = World()
    grant = w.grant("npc:n1", ["npc.act"])
    decision = w.gate.authorize(w.request(grant, "npc.act"))
    assert decision.allow and decision.code == "grant:ok"
    assert w.audit_codes() == ["grant:ok"]


@pytest.mark.parametrize("mutate,expected", [
    ("bad_sig", "request:bad_signature"),
    ("unknown_agent", "request:unknown_agent"),
    ("wrong_target", "grant:target_mismatch"),
    ("out_of_scope", "grant:action_not_in_scope"),
    ("expired", "grant:expired"),
    ("revoked", "grant:revoked"),
    ("no_grant", "grant:grant_not_found"),
])
def test_each_layer_denies_with_a_specific_code(mutate, expected):
    w = World()
    grant = w.grant("npc:n1", ["npc.act"])
    if mutate == "bad_sig":
        request = replace(w.request(grant, "npc.act"), params={"x": 1})
    elif mutate == "unknown_agent":
        request = w.request(grant, "npc.act", agent=w.new_agent("ghost", register=False))
    elif mutate == "wrong_target":
        request = w.request(grant, "npc.act", target="npc:other")
    elif mutate == "out_of_scope":
        request = w.request(grant, "npc.bind")
    elif mutate == "expired":
        w.now += 601
        request = w.request(grant, "npc.act")
    elif mutate == "revoked":
        w.revoke(grant)
        request = w.request(grant, "npc.act")
    else:
        request = replace(w.request(grant, "npc.act"), grant_id="does-not-exist")
        request = w.agent.sign_request(action="npc.act", target="npc:n1", grant_id="does-not-exist", now=w.now)
    decision = w.gate.authorize(request)
    assert not decision.allow and decision.code == expected
    assert w.audit_codes()[-1] == expected


def test_a_grant_for_another_agent_cannot_be_spent():
    w = World()
    other = w.new_agent("agent-2")
    grant = w.grant("npc:n1", ["npc.act"], agent=other)
    # agent-1 holds a valid signature of its own but presents agent-2's grant.
    decision = w.gate.authorize(w.request(grant, "npc.act", agent=w.agent))
    assert decision.code == "grant:agent_mismatch"


def test_a_replayed_request_is_denied_by_the_gate():
    w = World()
    grant = w.grant("npc:n1", ["npc.act"])
    request = w.request(grant, "npc.act")
    assert w.gate.authorize(request).allow
    assert w.gate.authorize(request).code == "request:replay"


def test_decisions_log_a_params_digest_not_the_raw_params():
    w = World()
    grant = w.grant("npc:n1", ["npc.act"])
    secret_text = "confidential-dialogue-line"
    w.gate.authorize(w.request(grant, "npc.act", {"intent": "speak", "text": secret_text}))
    dump = repr(w.audit.entries())
    assert secret_text not in dump
    assert "params_sha256" in dump


def test_every_decision_allow_or_deny_is_audited_in_order():
    w = World()
    grant = w.grant("npc:n1", ["npc.act"])
    w.gate.authorize(w.request(grant, "npc.act"))
    w.gate.authorize(w.request(grant, "npc.bind"))
    w.revoke(grant)
    w.gate.authorize(w.request(grant, "npc.act"))
    assert w.audit_codes() == ["grant:ok", "grant:action_not_in_scope", "grant:revoked"]


# ═══════════════════════════ failing closed ═════════════════════════════════

def test_no_audit_no_action(monkeypatch):
    """If the decision cannot be recorded, the action does not happen."""
    monkeypatch.delenv("WARDEN_GRANT_SECRET", raising=False)
    monkeypatch.delenv("SCBRAIN_GRID_SECRET", raising=False)
    w = World()
    grant = w.grant("npc:n1", ["npc.act"])
    w.gate._audit = AuditLog(w.store, secret=None, clock=w.clock)  # audit can't sign
    decision = w.gate.authorize(w.request(grant, "npc.act"))
    assert not decision.allow and decision.code == "gate:audit_unavailable"


def test_missing_secret_denies_the_whole_gate(monkeypatch):
    w = World()
    grant = w.grant("npc:n1", ["npc.act"])
    request = w.request(grant, "npc.act")
    monkeypatch.delenv("WARDEN_GRANT_SECRET", raising=False)
    monkeypatch.delenv("SCBRAIN_GRID_SECRET", raising=False)
    w.gate._secret = None
    assert w.gate.authorize(request).code in {"grant:secret_unset", "gate:audit_unavailable"}


def test_an_unexpected_exception_becomes_a_denial():
    class Exploding(GrantStore):
        def get(self, grant_id):  # noqa: D401
            raise RuntimeError("store on fire")

    w = World()
    grant = w.grant("npc:n1", ["npc.act"])
    w.gate._grants = Exploding(w.store)
    decision = w.gate.authorize(w.request(grant, "npc.act"))
    assert not decision.allow and decision.code == "gate:internal_error:RuntimeError"


def test_garbage_in_never_raises():
    w = World()
    for junk in (None, "x", 5, {}, object()):
        decision = w.gate.authorize(junk)  # type: ignore[arg-type]
        assert not decision.allow


# ═══════════════════════════ audit chain ════════════════════════════════════

def _filled_audit() -> World:
    w = World()
    grant = w.grant("npc:n1", ["npc.act"])
    for _ in range(4):
        w.gate.authorize(w.request(grant, "npc.act"))
    return w


def test_audit_chain_verifies():
    w = _filled_audit()
    assert w.audit.verify_chain() == (True, "ok")
    assert w.audit.head()["seq"] == 4


def test_editing_a_past_entry_is_detected():
    w = _filled_audit()
    entry = dict(w.store.get("warden:audit:000000000002"))
    entry["details"] = {**entry["details"], "allow": False}
    w.store.set("warden:audit:000000000002", entry)
    assert w.audit.verify_chain() == (False, "bad_mac:2")


def test_deleting_a_middle_entry_is_detected():
    w = _filled_audit()
    w.store._data.pop("warden:audit:000000000002")
    assert w.audit.verify_chain() == (False, "missing_entry:2")


def test_reordering_entries_is_detected():
    w = _filled_audit()
    a, b = w.store.get("warden:audit:000000000002"), w.store.get("warden:audit:000000000003")
    w.store.set("warden:audit:000000000002", b)
    w.store.set("warden:audit:000000000003", a)
    ok, reason = w.audit.verify_chain()
    assert not ok and reason.startswith(("seq_mismatch", "chain_break"))


def test_recomputing_a_plain_hash_chain_does_not_fool_it():
    """An attacker who can write the store but lacks the secret cannot re-forge MACs."""
    import hashlib
    import json
    w = _filled_audit()
    prev = GENESIS
    for seq in range(1, 5):
        key = f"warden:audit:{seq:012d}"
        entry = dict(w.store.get(key))
        if seq == 2:
            entry["details"] = {"allow": False}
        entry["prev"] = prev
        body = {k: v for k, v in entry.items() if k != "mac"}
        entry["mac"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        w.store.set(key, entry)
        prev = entry["mac"]
    assert w.audit.verify_chain()[0] is False


def test_splicing_in_an_entry_from_a_different_history_is_detected():
    """An entry with a valid MAC and the right seq, but from another log under the
    same secret, differs only in its `prev` link. Only the linkage check catches it."""
    a = _filled_audit()
    b = _filled_audit()
    # Give log B a different past so its entries carry different `prev` values.
    b_grant = b.grant("npc:other", ["npc.act"])
    b.gate.authorize(b.request(b_grant, "npc.bind"))
    b.audit.append("note", "someone", {"extra": True})
    donor = b.store.get("warden:audit:000000000003")
    assert donor["seq"] == 3 and donor["prev"] != a.store.get("warden:audit:000000000003")["prev"]
    a.store.set("warden:audit:000000000003", donor)
    assert a.audit.verify_chain() == (False, "chain_break:3")


def test_audit_refuses_to_append_or_verify_without_a_secret(monkeypatch):
    monkeypatch.delenv("WARDEN_GRANT_SECRET", raising=False)
    monkeypatch.delenv("SCBRAIN_GRID_SECRET", raising=False)
    log = AuditLog(MemoryStore(), secret=None)
    with pytest.raises(Exception):
        log.append("x", "y", {})
    assert log.verify_chain() == (False, "secret_unset")


def test_head_is_exposed_for_external_anchoring():
    w = _filled_audit()
    head = w.audit.head()
    assert set(head) == {"seq", "mac"} and head["mac"] == w.audit.entries()[-1]["mac"]


# ═══════════════════════════ BrainKV store ══════════════════════════════════

def test_state_persists_in_brain_kv(tmp_path, monkeypatch):
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "warden.sqlite"))
    w = World(store=BrainKVStore())
    grant = w.grant("npc:n1", ["npc.act"])
    assert w.gate.authorize(w.request(grant, "npc.act")).allow
    w.revoke(grant)

    # A fresh set of objects over the same database sees the same state.
    again = GrantStore(BrainKVStore())
    assert again.get(grant.grant_id) == grant
    assert again.is_revoked(grant.grant_id)
    assert AuditLog(BrainKVStore(), secret=SECRET).verify_chain() == (True, "ok")


# ═══════════════════════════ Essay progression ══════════════════════════════

def _game_request(w, target, action="game.act", **kw):
    grant = w.grant(target, ["game.connect", "game.act", "game.chat"])
    return grant, w.request(grant, action, {"cmd": "wave"}, **kw)


def test_game_target_without_an_essay_gate_is_denied():
    w = World()
    w.gate._essay = None
    _, request = _game_request(w, GW)
    assert w.gate.authorize(request).code == "stage:essay_gate_not_configured"


def test_guild_wars_is_closed_until_the_required_test_is_recorded():
    w = World()
    op = w.operator(GW)
    grant, request = _game_request(w, GW)
    attestation = w.attest(op, GW)
    assert w.gate.authorize(request, attestation=attestation).code == "stage:essay_test_not_passed"
    w.pass_essay()
    again = w.request(grant, "game.act", {"cmd": "wave"})
    decision = w.gate.authorize(again, attestation=attestation)
    assert decision.allow and decision.stage == "guild_wars"


def test_a_different_essay_test_does_not_unlock_the_required_one():
    w = World()
    op = w.operator(GW)
    w.pass_essay("some-other-test")
    _, request = _game_request(w, GW)
    assert w.gate.authorize(request, attestation=w.attest(op, GW)).code == "stage:essay_test_not_passed"


def test_a_recorded_failure_does_not_open_the_stage():
    w = World()
    op = w.operator(GW)
    w.progress.record_test(REQUIRED_TEST, False, warden_id=WARDEN)
    _, request = _game_request(w, GW)
    assert w.gate.authorize(request, attestation=w.attest(op, GW)).code == "stage:essay_test_not_passed"


def test_wow_stays_closed_until_guild_wars_is_complete_and_the_operator_attests():
    w = World()
    w.pass_essay()
    gw_op, wow_op = w.operator(GW, "op-gw"), w.operator(WOW, "op-wow")
    grant = w.grant(WOW, ["game.act"])

    def attempt(att):
        return w.gate.authorize(w.request(grant, "game.act", {"cmd": "wave"}), attestation=att).code

    wow_att = w.attest(wow_op, WOW)
    # test passed, operator attests — but Guild Wars is not recorded complete.
    assert attempt(wow_att) == "stage:guild_wars_not_complete"
    w.complete("guild_wars")
    # Guild Wars complete — but no operator attestation presented.
    assert attempt(None) == "stage:operator_attestation_missing"
    # Both present: finally open.
    assert attempt(wow_att) == "stage:ok"
    del gw_op


def test_completing_guild_wars_alone_is_not_enough_without_the_essay_test():
    w = World()
    w.complete("guild_wars")
    op = w.operator(WOW)
    grant = w.grant(WOW, ["game.act"])
    decision = w.gate.authorize(w.request(grant, "game.act"), attestation=w.attest(op, WOW))
    assert decision.code == "stage:essay_test_not_passed"


def test_forged_progress_records_are_ignored():
    """Writing the store directly, without the secret, does not advance anyone."""
    w = World()
    op = w.operator(GW)
    w.store.set(f"warden:essay:test:{REQUIRED_TEST}", {
        "kind": "essay_test", "test_id": REQUIRED_TEST, "passed": True,
        "recorded_by": WARDEN, "recorded_at": NOW, "signature": "0" * 64,
    })
    _, request = _game_request(w, GW)
    assert w.gate.authorize(request, attestation=w.attest(op, GW)).code == "stage:essay_test_not_passed"


def test_a_record_signed_by_someone_who_is_no_longer_a_warden_is_ignored():
    w = World()
    w.pass_essay()
    op = w.operator(GW)
    grant = w.grant(GW, ["game.act"])
    w.progress._registry = StaticWardenRegistry([])  # the signer lost the role
    w.essay._progress = w.progress
    decision = w.gate.authorize(w.request(grant, "game.act"), attestation=w.attest(op, GW))
    assert decision.code == "stage:essay_test_not_passed"


def test_only_a_warden_can_write_progress():
    w = World()
    with pytest.raises(WardenAuthorityError):
        w.progress.record_test(REQUIRED_TEST, True, warden_id="mallory")
    with pytest.raises(WardenAuthorityError):
        w.progress.record_stage_complete("guild_wars", warden_id="mallory")
    with pytest.raises(WardenFormatError):
        w.progress.record_stage_complete("retail_wow", warden_id=WARDEN)


def test_the_essay_gate_has_no_way_to_advance_itself():
    gate = World().essay
    forbidden = {"open", "advance", "unlock", "record_test", "record_stage_complete", "mark_complete", "set"}
    assert not forbidden & {name for name in dir(gate) if not name.startswith("_")}


def test_non_game_targets_are_not_stage_decisions():
    w = World()
    assert w.essay.decide(target="npc:n1", agent_id="agent-1", now=NOW).reason == "not_a_game_target"


def test_operator_requirement_can_be_narrowed_deliberately():
    w = World(operator_required_stages={"wow_legacy"})
    w.pass_essay()
    grant = w.grant(GW, ["game.act"])
    assert w.gate.authorize(w.request(grant, "game.act")).allow  # no operator needed for this stage


def test_by_default_every_game_stage_requires_the_operator():
    w = World()
    w.pass_essay()
    grant = w.grant(GW, ["game.act"])
    assert w.gate.authorize(w.request(grant, "game.act")).code == "stage:operator_attestation_missing"


# ═══════════════════════════ operator attestation ═══════════════════════════

def _stage_ready(target=GW):
    w = World()
    w.pass_essay()
    if target == WOW:
        w.complete("guild_wars")
    return w


def test_an_attestation_signed_by_the_wrong_key_is_rejected():
    w = _stage_ready()
    w.operator(GW)
    impostor = OperatorSigner.generate("operator-1")  # right id, key we never registered
    grant = w.grant(GW, ["game.act"])
    decision = w.gate.authorize(w.request(grant, "game.act"), attestation=w.attest(impostor, GW))
    assert decision.code == "stage:operator_bad_signature"


def test_another_operators_attestation_cannot_stand_in():
    w = _stage_ready(WOW)
    op_a = w.operator("game:wow_legacy:server-a", "op-a")
    w.operator(WOW, "op-b")
    grant = w.grant(WOW, ["game.act"])
    attestation = w.attest(op_a, "game:wow_legacy:server-a")
    decision = w.gate.authorize(w.request(grant, "game.act"), attestation=attestation)
    assert decision.code == "stage:operator_mismatch"


def test_consent_for_one_of_an_operators_servers_is_not_consent_for_another():
    """Same operator, two servers: a signature for server A must not unlock server B."""
    w = _stage_ready(WOW)
    server_a = "game:wow_legacy:server-a"
    signer = OperatorSigner.generate("op-both")
    w.operators.register("op-both", signer.public_key_hex, [server_a, WOW],
                         warden_id=WARDEN, registry=w.wardens)
    grant = w.grant(WOW, ["game.act"])
    for_a = signer.attest(agent_id="agent-1", target=server_a, ttl_s=600, now=w.now)
    assert w.gate.authorize(w.request(grant, "game.act"), attestation=for_a).code == "stage:operator_target_mismatch"
    for_b = signer.attest(agent_id="agent-1", target=WOW, ttl_s=600, now=w.now)
    assert w.gate.authorize(w.request(grant, "game.act"), attestation=for_b).allow


def test_an_attestation_for_another_agent_cannot_be_borrowed():
    w = _stage_ready()
    op = w.operator(GW)
    other = w.new_agent("agent-2")
    borrowed = w.attest(op, GW, agent=other)
    grant = w.grant(GW, ["game.act"])
    assert w.gate.authorize(w.request(grant, "game.act"), attestation=borrowed).code == "stage:operator_agent_mismatch"


def test_expired_and_revoked_attestations_are_rejected():
    w = _stage_ready()
    op = w.operator(GW)
    grant = w.grant(GW, ["game.act"], ttl=7200)
    short = w.attest(op, GW, ttl=100)
    assert w.gate.authorize(w.request(grant, "game.act"), attestation=short).allow
    w.now += 101
    assert w.gate.authorize(w.request(grant, "game.act"), attestation=short).code == "stage:operator_attestation_expired"
    w.now = NOW
    revoked = w.attest(op, GW)
    w.operators.revoke_attestation(revoked.attestation_id, warden_id=WARDEN, registry=w.wardens)
    assert w.gate.authorize(w.request(grant, "game.act"), attestation=revoked).code == "stage:operator_attestation_revoked"


def test_a_tampered_attestation_is_rejected():
    w = _stage_ready()
    op = w.operator(GW)
    att = w.attest(op, GW, ttl=100)
    stretched = replace(att, expires_at=att.expires_at + 10_000_000)
    assert verify_attestation(stretched, agent_id="agent-1", target=GW, now=NOW, operators=w.operators) == (
        False, "operator_bad_signature")


def test_a_removed_operator_invalidates_their_attestations():
    w = _stage_ready()
    op = w.operator(GW)
    att = w.attest(op, GW)
    w.operators.remove("operator-1", warden_id=WARDEN, registry=w.wardens)
    assert verify_attestation(att, agent_id="agent-1", target=GW, now=NOW, operators=w.operators) == (
        False, "operator_removed")


def test_a_target_has_exactly_one_operator():
    w = World()
    w.operator(GW, "op-1")
    with pytest.raises(WardenFormatError):
        w.operator(GW, "op-2")


def test_only_game_targets_can_have_operators_and_only_wardens_register_them():
    w = World()
    signer = OperatorSigner.generate("op-x")
    with pytest.raises(WardenFormatError):
        w.operators.register("op-x", signer.public_key_hex, ["npc:n1"], warden_id=WARDEN, registry=w.wardens)
    with pytest.raises(WardenAuthorityError):
        w.operators.register("op-x", signer.public_key_hex, [GW], warden_id="mallory", registry=w.wardens)
    with pytest.raises(WardenFormatError):
        w.operators.register("op-x", "00" * 32, [GW], warden_id=WARDEN, registry=w.wardens)  # weak key


def test_no_operator_registered_means_no_attestation_can_pass():
    w = _stage_ready()
    stray = OperatorSigner.generate("operator-1")
    att = stray.attest(agent_id="agent-1", target=GW, ttl_s=100, now=NOW)
    assert verify_attestation(att, agent_id="agent-1", target=GW, now=NOW, operators=w.operators) == (
        False, "no_operator_registered")


def test_missing_attestation_is_distinct_from_malformed():
    w = World()
    kw = dict(agent_id="agent-1", target=GW, now=NOW, operators=w.operators)
    assert verify_attestation(None, **kw) == (False, "operator_attestation_missing")
    assert verify_attestation("nope", **kw) == (False, "operator_attestation_malformed")  # type: ignore[arg-type]


# ═══════════════════════════ adapters ═══════════════════════════════════════

def _open_world():
    w = World()
    w.pass_essay()
    op = w.operator(GW)
    return w, w.attest(op, GW), w.grant(GW, ["game.connect", "game.act"])


def test_dispatch_reaches_the_adapter_only_after_the_whole_gate_passes():
    w, att, grant = _open_world()
    adapter = RecordingAdapter(GW)
    dispatcher = GameDispatcher(gate=w.gate, adapters={GW: adapter})
    result = dispatcher.dispatch(w.request(grant, "game.act", {"cmd": "wave"}), attestation=att)
    assert result.ok and adapter.calls[0][0] == "game.act"


def test_every_adapter_call_carries_the_automation_disclosure():
    w, att, grant = _open_world()
    adapter = RecordingAdapter(GW)
    dispatcher = GameDispatcher(gate=w.gate, adapters={GW: adapter})
    dispatcher.dispatch(w.request(grant, "game.connect"), attestation=att)
    dispatcher.dispatch(w.request(grant, "game.act", {"cmd": "wave"}), attestation=att)
    assert len(adapter.calls) == 2
    for _, _, disclosure in adapter.calls:
        assert disclosure["automated"] is True
        assert disclosure["agent_id"] == "agent-1" and disclosure["grant_id"] == grant.grant_id


def test_a_denied_request_never_reaches_the_adapter():
    w, _, grant = _open_world()
    adapter = RecordingAdapter(GW)
    result = GameDispatcher(gate=w.gate, adapters={GW: adapter}).dispatch(
        w.request(grant, "game.act"), attestation=None)  # operator has not attested
    assert not result.ok and result.reason == "stage:operator_attestation_missing"
    assert adapter.calls == []


def test_stub_adapters_are_not_wired_and_say_so():
    w, att, grant = _open_world()
    result = GameDispatcher(gate=w.gate, adapters={GW: GuildWarsAdapter()}).dispatch(
        w.request(grant, "game.connect"), attestation=att)
    assert not result.ok and result.reason == "dispatch:adapter_not_wired"


def test_wow_stub_targets_the_named_server_only():
    assert WowLegacyAdapter("hideout-01").target == WOW
    with pytest.raises(AdapterNotWired):
        WowLegacyAdapter("hideout-01").connect({"automated": True, "agent_id": "a", "grant_id": "g"})


def test_no_adapter_registered_is_a_clean_denial():
    w, att, grant = _open_world()
    result = GameDispatcher(gate=w.gate, adapters={}).dispatch(w.request(grant, "game.act"), attestation=att)
    assert result.reason == "dispatch:no_adapter"


@pytest.mark.parametrize("disclosure", [
    None, {}, {"automated": False, "agent_id": "a", "grant_id": "g"},
    {"automated": True}, {"automated": "yes", "agent_id": "a", "grant_id": "g"},
])
def test_adapters_refuse_calls_without_a_disclosure(disclosure):
    with pytest.raises(DisclosureMissing):
        require_disclosure(disclosure)
    with pytest.raises(DisclosureMissing):
        RecordingAdapter(GW).act("game.act", {}, disclosure)  # type: ignore[arg-type]


# ═══════════════════════════ documentation stays honest ═════════════════════

def test_the_documented_wiring_example_runs_as_written(tmp_path, monkeypatch):
    """Executes the 'Wiring it up' snippet from docs/WARDEN_EMBODIMENT.md verbatim,
    so the example cannot drift out of sync with the code."""
    import re
    from pathlib import Path

    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "doc.sqlite"))
    monkeypatch.setenv("WARDEN_GRANT_SECRET", SECRET)
    monkeypatch.setenv("WARDEN_IDS", "alice,bob")
    doc = (Path(__file__).resolve().parents[1] / "docs" / "WARDEN_EMBODIMENT.md").read_text(encoding="utf-8")
    snippet = re.search(r"## Wiring it up\n\n```python\n(.*?)```", doc, re.S).group(1)
    ns: dict = {}
    exec(snippet, ns)  # noqa: S102 - our own documentation, under test

    agent = ns["AgentIdentity"].generate("doc-agent")
    ns["agents"].register("doc-agent", agent.public_key_hex, warden_id="alice", registry=ns["wardens"])
    grant = ns["issue_grant"](warden_id="alice", agent_id="doc-agent", target="npc:n1",
                              scope=["npc.act"], ttl_s=600, registry=ns["wardens"])
    ns["grants"].save(grant)
    decision = ns["gate"].authorize(agent.sign_request(action="npc.act", target="npc:n1", grant_id=grant.grant_id))
    assert decision.allow and ns["audit"].verify_chain() == (True, "ok")

    monkeypatch.delenv("WARDEN_GRANT_SECRET")  # same wiring, secret removed: must fail closed
    again = ns["gate"].authorize(agent.sign_request(action="npc.act", target="npc:n1", grant_id=grant.grant_id))
    assert not again.allow


def test_star_import_exports_no_submodules():
    import types
    import src.quipu.warden as pkg
    assert not [n for n in pkg.__all__ if isinstance(getattr(pkg, n), types.ModuleType)]
    assert {"WardenGate", "NpcEmbodiment", "Reachy2Driver", "issue_grant"} <= set(pkg.__all__)
