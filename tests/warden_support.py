"""Shared fixtures for the Warden test suites.

Pure in-process: no network, no hardware, no wall-clock dependence. Time is a
mutable ``World.now`` so tests can expire grants by advancing it.
"""
from __future__ import annotations

from src.quipu.warden import (
    AgentIdentity,
    AgentRegistry,
    AuditLog,
    EssayGate,
    Grant,
    GrantStore,
    MemoryStore,
    NpcEmbodiment,
    OperatorRegistry,
    OperatorSigner,
    SignedRecordProgress,
    SignedRequest,
    StaticWardenRegistry,
    WardenGate,
    issue_grant,
)

SECRET = "test-warden-secret-0123456789abcdef"
NOW = 1_800_000_000
REQUIRED_TEST = "essay-final"
WARDEN = "warden-1"


class World:
    def __init__(self, *, store=None, required_test: str = REQUIRED_TEST, operator_required_stages=None):
        self.now = NOW
        self.store = store if store is not None else MemoryStore()
        self.wardens = StaticWardenRegistry([WARDEN])
        self.grants = GrantStore(self.store)
        self.agents = AgentRegistry(self.store)
        self.audit = AuditLog(self.store, secret=SECRET, clock=self.clock)
        self.operators = OperatorRegistry(self.store)
        self.progress = SignedRecordProgress(self.store, registry=self.wardens, secret=SECRET)
        kwargs = {}
        if operator_required_stages is not None:
            kwargs["operator_required_stages"] = frozenset(operator_required_stages)
        self.essay = EssayGate(
            required_test=required_test, progress=self.progress, operators=self.operators, **kwargs
        )
        self.gate = WardenGate(
            grants=self.grants, agents=self.agents, wardens=self.wardens,
            audit=self.audit, essay=self.essay, secret=SECRET, clock=self.clock,
        )
        self.agent = self.new_agent("agent-1")

    def clock(self) -> float:
        return self.now

    def new_agent(self, agent_id: str, *, register: bool = True) -> AgentIdentity:
        identity = AgentIdentity.generate(agent_id)
        if register:
            self.agents.register(agent_id, identity.public_key_hex, warden_id=WARDEN, registry=self.wardens)
        return identity

    def grant(self, target: str, scope, *, ttl: int = 600, agent: AgentIdentity | None = None) -> Grant:
        grant = issue_grant(
            warden_id=WARDEN, agent_id=(agent or self.agent).agent_id, target=target,
            scope=scope, ttl_s=ttl, registry=self.wardens, secret=SECRET, now=self.now,
        )
        self.grants.save(grant)
        return grant

    def request(
        self, grant: Grant, action: str, params=None, *, agent: AgentIdentity | None = None,
        target: str | None = None, now: float | None = None,
    ) -> SignedRequest:
        return (agent or self.agent).sign_request(
            action=action, target=target or grant.target, grant_id=grant.grant_id,
            params=params or {}, now=self.now if now is None else now,
        )

    def revoke(self, grant: Grant, reason: str = "test") -> dict:
        return self.grants.revoke(
            grant.grant_id, warden_id=WARDEN, registry=self.wardens, reason=reason,
            secret=SECRET, now=self.now,
        )

    def operator(self, target: str, operator_id: str = "operator-1") -> OperatorSigner:
        signer = OperatorSigner.generate(operator_id)
        self.operators.register(
            operator_id, signer.public_key_hex, [target], warden_id=WARDEN, registry=self.wardens
        )
        return signer

    def attest(self, signer: OperatorSigner, target: str, *, ttl: int = 600, agent: AgentIdentity | None = None):
        return signer.attest(
            agent_id=(agent or self.agent).agent_id, target=target, ttl_s=ttl, now=self.now
        )

    def pass_essay(self, test_id: str = REQUIRED_TEST) -> None:
        self.progress.record_test(test_id, True, warden_id=WARDEN)

    def complete(self, stage: str) -> None:
        self.progress.record_stage_complete(stage, warden_id=WARDEN)

    def embodiment(self, **kwargs) -> NpcEmbodiment:
        return NpcEmbodiment(gate=self.gate, grants=self.grants, store=self.store, **kwargs)

    def audit_codes(self) -> list[str]:
        out = []
        for entry in self.audit.entries():
            d = entry["details"]
            out.append(f"{d.get('layer')}:{d.get('reason')}" if entry["kind"] == "decision" else entry["kind"])
        return out
