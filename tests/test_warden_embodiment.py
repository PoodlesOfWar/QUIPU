"""NPC <-> Reachy 2 binding: two grants, bounded intents, e-stop on revoke.

No hardware and no network. ``SimDriver`` stands in for the robot, and
``Reachy2Driver`` is exercised against a fake SDK shaped like ``reachy2-sdk`` 1.0.15.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.quipu.warden import (
    BindingDenied,
    DriverRejected,
    Reachy2Driver,
    SimDriver,
    UnsupportedIntent,
    parse_intent,
)
from src.quipu.warden.npc_embodiment import BoundedIntent, IntentRejected
from warden_support import NOW, World

HOST = "10.0.0.5"
NPC = "npc:guard-7"
ROBOT = f"reachy2:{HOST}"
LOOK = {"intent": "look_at", "x": 1.0, "y": 0.1, "z": 0.2}


class Rig:
    """A world with an NPC grant, a robot grant, and one SimDriver per host."""

    def __init__(self, robot_scope=("robot.bind", "robot.move", "robot.observe", "robot.gesture"), ttl=600):
        self.w = World()
        self.drivers: dict[str, SimDriver] = {}
        self.emb = self.w.embodiment(driver_factory=self._make)
        self.npc_grant = self.w.grant(NPC, ["npc.bind", "npc.act"], ttl=ttl)
        self.robot_grant = self.w.grant(ROBOT, list(robot_scope), ttl=ttl)

    def _make(self, host):
        self.drivers[host] = SimDriver(host)
        return self.drivers[host]

    @property
    def driver(self) -> SimDriver:
        return self.drivers[HOST]

    def bind_request(self, **kw):
        params = {"reachy_host": HOST, "reachy_grant_id": self.robot_grant.grant_id}
        params.update(kw.pop("params", {}))
        return self.w.request(self.npc_grant, "npc.bind", params, **kw)

    def bind(self):
        return self.emb.bind_npc(self.bind_request())

    def act(self, params=None, **kw):
        return self.emb.act(self.w.request(self.npc_grant, "npc.act", LOOK if params is None else params, **kw))


@pytest.fixture
def rig():
    return Rig()


# ═══════════════════════════ binding needs two live grants ══════════════════

def test_bind_succeeds_with_both_grants(rig):
    record = rig.bind()
    assert record["npc_id"] == "guard-7" and record["reachy_host"] == HOST
    assert rig.driver.connected
    assert rig.emb.binding("guard-7")["agent_id"] == "agent-1"


def test_binding_expires_when_the_earlier_grant_does():
    rig = Rig()
    short = rig.w.grant(ROBOT, ["robot.bind", "robot.move"], ttl=60)
    rig.robot_grant = short
    assert rig.bind()["expires_at"] == NOW + 60


def test_bind_is_refused_without_a_robot_grant(rig):
    with pytest.raises(BindingDenied, match="grant_not_found"):
        rig.emb.bind_npc(rig.bind_request(params={"reachy_grant_id": "missing"}))
    assert rig.emb.binding("guard-7") is None and not rig.drivers


def test_bind_is_refused_when_the_robot_grant_is_for_another_host(rig):
    wrong = rig.w.grant("reachy2:10.9.9.9", ["robot.bind"])
    with pytest.raises(BindingDenied, match="target_mismatch"):
        rig.emb.bind_npc(rig.bind_request(params={"reachy_grant_id": wrong.grant_id}))


def test_bind_is_refused_when_the_robot_grant_does_not_scope_binding():
    rig = Rig(robot_scope=("robot.move",))
    with pytest.raises(BindingDenied, match="action_not_in_scope"):
        rig.bind()


def test_bind_is_refused_when_the_npc_grant_does_not_scope_binding(rig):
    only_act = rig.w.grant(NPC, ["npc.act"])
    request = rig.w.request(only_act, "npc.bind",
                            {"reachy_host": HOST, "reachy_grant_id": rig.robot_grant.grant_id})
    with pytest.raises(BindingDenied, match="npc_grant:grant:action_not_in_scope"):
        rig.emb.bind_npc(request)


def test_bind_is_refused_for_a_revoked_or_expired_grant(rig):
    rig.w.revoke(rig.robot_grant)
    with pytest.raises(BindingDenied, match="revoked"):
        rig.bind()
    rig2 = Rig()
    rig2.w.now += 601
    with pytest.raises(BindingDenied, match="expired"):
        rig2.bind()


def test_bind_requires_the_agents_signature(rig):
    forged = replace(rig.bind_request(), params={"reachy_host": "10.7.7.7",
                                                "reachy_grant_id": rig.robot_grant.grant_id})
    with pytest.raises(BindingDenied, match="request:bad_signature"):
        rig.emb.bind_npc(forged)


def test_another_agents_grants_cannot_be_used_to_bind(rig):
    other = rig.w.new_agent("agent-2")
    with pytest.raises(BindingDenied, match="agent_mismatch"):
        rig.emb.bind_npc(rig.w.request(rig.npc_grant, "npc.bind",
                                       {"reachy_host": HOST, "reachy_grant_id": rig.robot_grant.grant_id},
                                       agent=other))


@pytest.mark.parametrize("params", [
    {}, {"reachy_host": None}, {"reachy_host": 5}, {"reachy_host": "bad host"},
    {"reachy_host": HOST, "reachy_grant_id": 7},
])
def test_malformed_bind_parameters_are_refused(rig, params):
    request = rig.w.request(rig.npc_grant, "npc.bind", params)
    with pytest.raises(BindingDenied):
        rig.emb.bind_npc(request)


def test_wrong_action_or_target_kind_is_not_a_bind(rig):
    with pytest.raises(BindingDenied, match="not_an_npc_bind_request"):
        rig.emb.bind_npc(rig.w.request(rig.npc_grant, "npc.act", {}))
    with pytest.raises(BindingDenied):
        rig.emb.bind_npc(rig.w.request(rig.robot_grant, "robot.bind", {}))


def test_an_npc_and_a_robot_can_each_only_be_bound_once(rig):
    rig.bind()
    with pytest.raises(BindingDenied, match="npc_already_bound"):
        rig.bind()
    other_npc = rig.w.grant("npc:guard-8", ["npc.bind", "npc.act"])
    with pytest.raises(BindingDenied, match="robot_already_bound"):
        rig.emb.bind_npc(rig.w.request(other_npc, "npc.bind",
                                       {"reachy_host": HOST, "reachy_grant_id": rig.robot_grant.grant_id}))


def test_a_driver_that_fails_to_connect_leaves_nothing_bound():
    rig = Rig()

    class Broken(SimDriver):
        def connect(self):
            raise RuntimeError("no route to robot")

    rig.emb._factory = lambda host: rig.drivers.setdefault(host, Broken(host))
    with pytest.raises(BindingDenied, match="driver_connect_failed"):
        rig.bind()
    assert rig.emb.binding("guard-7") is None
    assert rig.drivers[HOST].stopped  # the half-open driver was released


# ═══════════════════════════ acting ═════════════════════════════════════════

def test_a_valid_act_reaches_the_driver(rig):
    rig.bind()
    result = rig.act()
    assert result.ok
    assert [(i.kind, dict(i.params)) for i in rig.driver.log] == [
        ("look_at", {"x": 1.0, "y": 0.1, "z": 0.2, "duration": 2.0})]


def test_act_on_an_unbound_npc_is_refused_and_touches_nothing(rig):
    result = rig.act()
    assert not result.ok and result.reason == "not_bound" and not rig.drivers


def test_what_is_authorized_is_what_is_executed(rig):
    """The intent comes from the signed parameters, so it cannot be swapped after signing."""
    rig.bind()
    signed = rig.w.request(rig.npc_grant, "npc.act", LOOK)
    swapped = replace(signed, params={"intent": "look_at", "x": 0.3, "y": 0.9, "z": 0.9})
    result = rig.emb.act(swapped)
    assert not result.ok and result.reason == "request:bad_signature"
    assert rig.driver.log == []


def test_a_replayed_act_is_refused(rig):
    rig.bind()
    request = rig.w.request(rig.npc_grant, "npc.act", LOOK)
    assert rig.emb.act(request).ok
    assert rig.emb.act(request).reason == "request:replay"
    assert len(rig.driver.log) == 1


def test_another_agent_cannot_drive_a_bound_npc(rig):
    rig.bind()
    other = rig.w.new_agent("agent-2")
    their_grant = rig.w.grant(NPC, ["npc.act"], agent=other)
    result = rig.emb.act(rig.w.request(their_grant, "npc.act", LOOK, agent=other))
    assert not result.ok and result.reason == "binding_mismatch"
    assert rig.driver.log == []


def test_an_out_of_scope_robot_action_is_refused_without_halting():
    rig = Rig(robot_scope=("robot.bind", "robot.move"))
    rig.bind()
    result = rig.act({"intent": "gesture", "name": "neutral"})  # robot.gesture not scoped
    assert not result.ok and result.reason == "robot:grant:action_not_in_scope"
    assert rig.emb.binding("guard-7") is not None and not rig.driver.stopped
    assert rig.act().ok  # the scoped action still works


def test_driver_errors_do_not_escape_or_unbind(rig):
    rig.bind()
    rig.driver.act = lambda intent: (_ for _ in ()).throw(DriverRejected("servo fault"))  # type: ignore[assignment]
    result = rig.act()
    assert not result.ok and result.reason == "driver:DriverRejected"
    assert rig.emb.binding("guard-7") is not None


# ═══════════════════════════ bounded action set ═════════════════════════════

@pytest.mark.parametrize("params", [
    {"intent": "look_at", "x": 5.0, "y": 0, "z": 0},          # too far
    {"intent": "look_at", "x": 0.0, "y": 0, "z": 0},          # inside the minimum
    {"intent": "look_at", "x": 1.0, "y": 3.0, "z": 0},
    {"intent": "look_at", "x": 1.0, "y": 0, "z": 9.0},
    {"intent": "look_at", "x": 1.0, "y": 0, "z": 0, "duration": 0},
    {"intent": "look_at", "x": 1.0, "y": 0, "z": 0, "duration": 99},
    {"intent": "look_at", "x": True, "y": 0, "z": 0},          # bool is not a number here
    {"intent": "look_at", "x": "1.0", "y": 0, "z": 0},
    {"intent": "look_at", "y": 0, "z": 0},                      # missing field
    {"intent": "gesture", "name": "backflip"},
    {"intent": "speak", "text": ""},
    {"intent": "speak", "text": "x" * 281},
    {"intent": "speak", "text": "bell\x07"},
    {"intent": "speak", "text": 123},
    {"intent": "arm_swing"}, {"intent": None}, {},
    {"intent": "turn_off"}, {"intent": "mobile_base"},
])
def test_out_of_bounds_intents_are_rejected_and_never_reach_the_driver(rig, params):
    rig.bind()
    result = rig.act(params)
    assert not result.ok and result.reason.startswith("intent_rejected")
    assert rig.driver.log == []


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_can_never_be_signed_or_accepted(rig, bad):
    """Canonical JSON refuses NaN/Infinity, so they cannot be signed; and a request
    that carries one on the wire (hand-built, signature irrelevant) is 'malformed',
    not an unexplained internal error."""
    rig.bind()
    params = {"intent": "look_at", "x": bad, "y": 0, "z": 0}
    with pytest.raises(ValueError):
        rig.w.request(rig.npc_grant, "npc.act", params)
    good = rig.w.request(rig.npc_grant, "npc.act", LOOK)
    result = rig.emb.act(replace(good, params=params))
    assert not result.ok and result.reason == "request:malformed"
    assert rig.driver.log == []
    with pytest.raises(IntentRejected):
        parse_intent(params)  # and the intent validator independently refuses it


def test_the_edges_of_the_bounds_are_accepted():
    for params in (
        {"intent": "look_at", "x": 0.2, "y": -1.0, "z": -0.5, "duration": 0.5},
        {"intent": "look_at", "x": 2.0, "y": 1.0, "z": 1.0, "duration": 10.0},
        {"intent": "speak", "text": "x" * 280},
        {"intent": "gesture", "name": "neutral"},
        {"intent": "idle"},
    ):
        assert isinstance(parse_intent(params), BoundedIntent)
    with pytest.raises(IntentRejected):
        parse_intent("not a mapping")  # type: ignore[arg-type]


# ═══════════════════════════ e-stop: revoke, expiry, reap ═══════════════════

@pytest.mark.parametrize("which", ["npc_grant", "robot_grant"])
def test_revoking_either_grant_mid_session_halts_the_driver_at_once(which):
    rig = Rig()
    rig.bind()
    assert rig.act().ok and not rig.driver.stopped
    rig.w.revoke(getattr(rig, which))
    # The listener stopped the robot immediately — before any further act arrives.
    assert rig.driver.stopped and not rig.driver.connected
    assert rig.emb.binding("guard-7") is None
    result = rig.act()
    assert not result.ok and result.reason == "not_bound"
    assert len(rig.driver.log) == 1  # nothing after the revoke


def test_a_revoke_in_another_process_is_caught_by_the_next_act():
    """The listener is only the fast path; per-act re-verification is the guarantee."""
    rig = Rig()
    rig.bind()
    rig.w.grants._listeners.clear()  # simulate a revoke made where our listener never ran
    rig.w.revoke(rig.robot_grant)
    assert not rig.driver.stopped
    result = rig.act()
    assert not result.ok and result.reason == "robot:grant:revoked"
    assert rig.driver.stopped and rig.emb.binding("guard-7") is None
    assert len(rig.driver.log) == 0


def test_no_motion_after_a_grant_expires_between_acts():
    rig = Rig(ttl=100)
    rig.bind()
    assert rig.act().ok
    rig.w.now += 101
    result = rig.act()
    assert not result.ok and result.reason.endswith("expired")
    assert rig.driver.stopped and len(rig.driver.log) == 1


def test_reap_halts_an_idle_binding_whose_grant_has_expired():
    rig = Rig(ttl=100)
    rig.bind()
    assert rig.emb.reap() == []
    rig.w.now += 100
    assert rig.emb.reap() == ["guard-7"]
    assert rig.driver.stopped and rig.emb.binding("guard-7") is None


def test_reap_halts_bindings_with_revoked_grants_even_if_the_listener_missed_it():
    rig = Rig()
    rig.bind()
    rig.w.grants._listeners.clear()
    rig.w.revoke(rig.npc_grant)
    assert rig.emb.reap() == ["guard-7"]


def test_a_demoted_warden_stops_the_robot_on_the_next_act():
    rig = Rig()
    rig.bind()
    rig.w.wardens._ids = frozenset()
    result = rig.act()
    assert not result.ok and "warden_not_authorized" in result.reason
    assert rig.driver.stopped


def test_estop_and_estop_all_are_idempotent():
    rig = Rig()
    rig.bind()
    assert rig.emb.estop_all() == 1 and rig.driver.stopped
    assert rig.emb.estop_all() == 0
    assert rig.emb.estop("guard-7", reason="again") is False


def test_a_robot_can_be_rebound_after_an_estop():
    rig = Rig()
    rig.bind()
    rig.emb.estop("guard-7", reason="manual")
    rig.npc_grant = rig.w.grant(NPC, ["npc.bind", "npc.act"])
    rig.robot_grant = rig.w.grant(ROBOT, ["robot.bind", "robot.move"])
    assert rig.bind()["reachy_host"] == HOST
    assert not rig.driver.stopped  # a fresh driver, not the stopped one


def test_a_failing_stop_still_completes_the_unbind():
    rig = Rig()
    rig.bind()
    rig.driver.stop = lambda: (_ for _ in ()).throw(RuntimeError("stuck"))  # type: ignore[assignment]
    assert rig.emb.estop("guard-7", reason="test") is True
    assert rig.emb.binding("guard-7") is None
    estop = [e for e in rig.w.audit.entries() if e["details"].get("event") == "estop"][0]
    assert estop["details"]["driver_stop_ok"] is False


def test_binds_acts_and_stops_are_all_audited():
    rig = Rig()
    rig.bind()
    rig.act()
    rig.w.revoke(rig.robot_grant)
    events = [e["details"].get("event") for e in rig.w.audit.entries() if e["kind"] == "embodiment"]
    assert events == ["bind", "act", "estop"]
    assert rig.w.audit.verify_chain() == (True, "ok")


# ═══════════════════════════ Reachy2Driver (fake SDK) ═══════════════════════

class FakeReachy:
    """Shaped like ``reachy2_sdk.ReachySDK`` 1.0.15 (verified against the wheel)."""

    def __init__(self, host, *, reachable=True, turn_on_ok=True, goto_id=1):
        self.host, self.calls = host, []
        self._reachable, self._turn_on_ok, self._goto_id = reachable, turn_on_ok, goto_id
        self.head = SimpleNamespace(look_at=self._look_at)
        self._connected = False

    def connect(self):
        self.calls.append("connect")
        self._connected = self._reachable

    def is_connected(self):
        return self._connected

    def turn_on(self):
        self.calls.append("turn_on")
        return self._turn_on_ok

    def _look_at(self, x, y, z, duration=2.0, wait=False, interpolation_mode="minimum_jerk"):
        self.calls.append(("look_at", x, y, z, duration, wait))
        return SimpleNamespace(id=self._goto_id)

    def goto_posture(self, common_posture="default", duration=2, wait=False, **kw):
        self.calls.append(("goto_posture", common_posture, duration, wait))

    def cancel_all_goto(self):
        self.calls.append("cancel_all_goto")

    def turn_off_smoothly(self):
        self.calls.append("turn_off_smoothly")

    def disconnect(self):
        self.calls.append("disconnect")


def _driver(**fake_kw):
    fakes = []

    def factory(host):
        fakes.append(FakeReachy(host, **fake_kw))
        return fakes[-1]

    return Reachy2Driver(HOST, sdk_factory=factory), fakes


def _intent(**params):
    return parse_intent(params)


def test_reachy2_driver_connects_then_powers_on():
    driver, fakes = _driver()
    driver.connect()
    assert fakes[0].calls == ["connect", "turn_on"]


def test_reachy2_driver_look_at_uses_the_verified_head_api():
    driver, fakes = _driver()
    driver.connect()
    driver.act(_intent(intent="look_at", x=1.0, y=0.1, z=0.2, duration=3.0))
    assert fakes[0].calls[-1] == ("look_at", 1.0, 0.1, 0.2, 3.0, False)  # non-blocking


def test_reachy2_driver_gesture_is_the_default_posture_only():
    driver, fakes = _driver()
    driver.connect()
    driver.act(_intent(intent="gesture", name="neutral"))
    assert fakes[0].calls[-1] == ("goto_posture", "default", 2.0, False)


def test_reachy2_driver_idle_sends_nothing():
    driver, fakes = _driver()
    driver.connect()
    before = list(fakes[0].calls)
    driver.act(_intent(intent="idle"))
    assert fakes[0].calls == before


def test_reachy2_driver_surfaces_an_unreachable_target():
    driver, _ = _driver(goto_id=-1)
    driver.connect()
    with pytest.raises(DriverRejected, match="rejected"):
        driver.act(_intent(intent="look_at", x=1.0, y=0, z=0))


def test_reachy2_driver_speak_needs_an_injected_speaker():
    driver, _ = _driver()
    driver.connect()
    with pytest.raises(UnsupportedIntent):
        driver.act(_intent(intent="speak", text="hello"))
    said = []
    voiced = Reachy2Driver(HOST, sdk_factory=lambda h: FakeReachy(h), speak_fn=said.append)
    voiced.connect()
    voiced.act(_intent(intent="speak", text="hello"))
    assert said == ["hello"]


@pytest.mark.parametrize("fake_kw,match", [
    ({"reachable": False}, "could not connect"),
    ({"turn_on_ok": False}, "refused to turn on"),
])
def test_reachy2_driver_connect_failures_are_clear(fake_kw, match):
    driver, _ = _driver(**fake_kw)
    with pytest.raises(DriverRejected, match=match):
        driver.connect()
    with pytest.raises(DriverRejected, match="not live"):
        driver.act(_intent(intent="idle"))


def test_reachy2_driver_estop_cancels_then_relaxes_then_releases_in_order():
    driver, fakes = _driver()
    driver.connect()
    assert driver.stop() is True
    assert fakes[0].calls[-3:] == ["cancel_all_goto", "turn_off_smoothly", "disconnect"]
    assert driver.stop() is True  # idempotent
    with pytest.raises(DriverRejected):
        driver.act(_intent(intent="idle"))


def test_reachy2_driver_estop_attempts_every_step_even_when_one_fails():
    driver, fakes = _driver()
    driver.connect()

    def boom():
        raise RuntimeError("grpc down")

    fakes[0].cancel_all_goto = boom
    assert driver.stop() is False
    assert fakes[0].calls[-2:] == ["turn_off_smoothly", "disconnect"]
    assert driver.last_stop_errors == ["cancel_all_goto: RuntimeError"]


def test_importing_the_driver_needs_neither_the_sdk_nor_a_robot(monkeypatch):
    monkeypatch.setitem(sys.modules, "reachy2_sdk", None)  # simulate "not installed"
    driver = Reachy2Driver(HOST)  # construction is free
    with pytest.raises(DriverRejected, match="not installed"):
        driver.connect()


def test_revoke_triggers_the_real_reachy2_driver_estop_end_to_end():
    w = World()
    fakes = []
    emb = w.embodiment(driver_factory=lambda host: Reachy2Driver(
        host, sdk_factory=lambda h: fakes.append(FakeReachy(h)) or fakes[-1]))
    npc = w.grant(NPC, ["npc.bind", "npc.act"])
    robot = w.grant(ROBOT, ["robot.bind", "robot.move"])
    emb.bind_npc(w.request(npc, "npc.bind", {"reachy_host": HOST, "reachy_grant_id": robot.grant_id}))
    assert emb.act(w.request(npc, "npc.act", LOOK)).ok
    w.revoke(robot)
    assert fakes[0].calls[-3:] == ["cancel_all_goto", "turn_off_smoothly", "disconnect"]
    assert not emb.act(w.request(npc, "npc.act", LOOK)).ok
