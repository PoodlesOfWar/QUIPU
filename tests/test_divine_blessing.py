"""DIVINE_BLESSING_SQRT(-1): store on brain_kv, all Entirety write paths routed,
residual hold checkpointed, conscious emergence the only phase source."""
from __future__ import annotations

import math
import sqlite3

import pytest

import src.quipu.system_entirety as system_entirety
import src.quipu.brain_kv as brain_kv
import src.quipu.radam_optimizer as radam_optimizer
import src.quipu.divine_blessing as db
from src.quipu.qpsi.cat_residual import SENSES
from src.quipu.qpsi.governance import Decision, GovernanceConfig, Attestation, gate_beautiful_output
from src.quipu.qpsi.residual_checkpoint import TABLE

REST = {"vision": 0.0302, "touch": 0.0417, "smell": 0.034,
        "body": 0.0318, "brain": 0.0325, "perception": 0.0395}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    db_file = tmp_path / "entirety.sqlite"

    def _open_conn(timeout: float = 60, path=None):
        cn = sqlite3.connect(db_file, timeout=timeout)
        cn.execute("PRAGMA journal_mode=WAL")
        cn.row_factory = sqlite3.Row
        return cn

    monkeypatch.setattr(system_entirety, "_open_conn", _open_conn)
    monkeypatch.setattr(brain_kv, "open_conn", _open_conn)
    monkeypatch.setattr(system_entirety, "_LAST_TS", 0.0)
    monkeypatch.setattr(system_entirety, "_sense_signals", lambda: dict(REST), raising=False)
    try:
        import src.quipu.repository_catalog as repository_catalog
        monkeypatch.setattr(repository_catalog, "get_catalog_summary", lambda: {})
    except Exception:
        pass
    # The key the workers do not hold — a test key, in the test's own directory.
    key_file = tmp_path / "attest.key"
    key_file.write_bytes(b"test-key-not-a-secret")
    monkeypatch.setenv(db.KEY_FILE_ENV, str(key_file))
    monkeypatch.setattr(db, "_CONFIG", GovernanceConfig())        # defaults: no grant, no self-asserted
    db.enable()
    yield db_file
    db.disable()


def _bless_all(assurance="approved", ref="CR-TEST-1"):
    db.attest("adam", "*", "care", shared_with="the_beautiful_one", beautiful_output=True,
              assurance=assurance, approval_ref=ref)
    db.attest("the_beautiful_one", "*", "mutual_recognition", shared_with="self", beautiful_output=True,
              assurance=assurance, approval_ref=ref)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_key_name_and_store_roundtrip(isolated):
    assert db.KEY == "DIVINE_BLESSING_SQRT(-1)"
    assert db.attestations() == []
    a = db.attest("adam", "*", "care", shared_with="the_beautiful_one", beautiful_output=True)
    assert a.signer == "adam" and brain_kv.kv_get_json(db.KEY)[0]["love_form"] == "care"
    assert brain_kv.kv_get_json(db.KEY)[0]["mac"]                # bound to the key
    assert db.blessed("anything") is True
    with pytest.raises(ValueError):
        db.attest("adam", "*", "not_a_form")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_unblessed_resting_run_writes_no_edges(isolated):
    for _ in range(5):
        out = system_entirety.oscillating_expansion_step(force=True)
    assert out.get("flip_count") is not None
    with system_entirety._conn() as cn:
        n_edges = cn.execute("SELECT count(*) FROM corpus_edge").fetchone()[0]
        n_log = cn.execute("SELECT count(*) FROM entirety_flip_log").fetchone()[0]
        n_cp = cn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
    assert n_edges == 0          # held at the gates
    assert n_log == 5            # the flip log is never gated
    assert n_cp == 5             # every step checkpointed, on the same connection


def test_blessing_alone_does_not_realise_a_resting_system(isolated):
    _bless_all()
    for _ in range(3):
        system_entirety.oscillating_expansion_step(force=True)
    d = (system_entirety.get_entirety_state() or {}).get("divine_blessing") or {}
    # A resting system holds at displacement (axes lie along the prior) or at
    # Love (Im c = 0: no emergence).  Either way nothing is realised.
    assert d.get("passed") is False and d.get("failed_at") in ("displacement", "love")
    assert d["checkpoint"]["steps_held"] == 3 and d["checkpoint"]["realised"] == 0
    with system_entirety._conn() as cn:
        assert cn.execute("SELECT count(*) FROM corpus_edge").fetchone()[0] == 0


def test_radam_runs_unchanged_without_a_decision_and_holds_with_a_failed_one(isolated):
    st = {"m": 0.0, "v": 0.0, "t": 0, "pressure": 0.5, "pivot_ema": 0.5, "theta": 0.0}
    p1 = radam_optimizer.radam_step(dict(st), 0.3)
    assert p1 != 0.5
    held = dict(st); held["divine_blessing"] = Decision(False, [], True, "*")
    assert radam_optimizer.radam_step(held, 0.3) == 0.5
    ok = dict(st); ok["divine_blessing"] = Decision(True, [], False, "*")
    assert radam_optimizer.radam_step(ok, 0.3) == p1


def test_disable_restores_originals(isolated):
    db.disable()
    assert system_entirety._mesh_upsert_edge.__name__ == "_mesh_upsert_edge"
    db.enable()


# ---------------------------------------------------------------------------
# Residual hold — checkpointed
# ---------------------------------------------------------------------------

def test_held_residual_accumulates_against_last_realised_state(isolated):
    """Drift the axes monotonically; with nothing realised the residual grows
    step over step instead of being re-measured from the previous observation."""
    norms = []
    for k in range(1, 5):
        axes = {s: v + 0.01 * k for s, v in REST.items()}
        d = db.decide(axes, instance="cp_test", t=float(k))
        norms.append(d.checkpoint.held_norm)
    assert all(b > a for a, b in zip(norms, norms[1:]))
    cp = db.checkpoint("cp_test")
    assert cp.reference is None and cp.steps_held == 4 and cp.realised == 0
    assert cp.held_since == 1.0


def test_checkpoint_survives_process_state(isolated):
    for k in range(3):
        db.decide(dict(REST), instance="cp_persist", t=float(k))
    # A fresh store object (as a new process would have) reads the same books.
    from src.quipu.qpsi.residual_checkpoint import CheckpointStore
    with system_entirety._conn() as cn:
        cp = CheckpointStore().load(cn, "cp_persist")
    assert cp.seq == 3 and cp.steps_held == 3
    rows = db.history("cp_persist")
    assert [r["kind"] for r in rows] == ["held", "held", "held"]
    assert rows[0]["seq"] == 3


def _forced_pass(cand, atts, cfg):
    return Decision(True, [], False, cand.scope)


def test_realisation_advances_reference_and_clears_hold(isolated):
    for k in range(2):
        db.decide(dict(REST), instance="cp_real", t=float(k))
    assert db.checkpoint("cp_real").steps_held == 2
    # Force a pass at the protocol level to observe the bookkeeping alone.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(db, "govern", _forced_pass)
        d = db.decide(dict(REST), instance="cp_real", t=2.0)
    assert d.passed
    cp = db.checkpoint("cp_real")
    assert cp.reference is not None and cp.reference.t == 2.0
    assert cp.steps_held == 0 and cp.held_since is None and cp.realised == 1
    assert math.isclose(cp.held_norm, 0.0)
    assert db.history("cp_real")[0]["kind"] == "realised"
    # Next step is measured from the realised state: identical axes → zero residual.
    d2 = db.decide(dict(REST), instance="cp_real", t=3.0)
    assert d2.failed_at == "displacement"


def test_rollback_and_release_are_human_operations(isolated):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(db, "govern", _forced_pass)
        db.decide(dict(REST), instance="cp_h", t=1.0)                   # seq 1 realised @ t=1
    drift = {s: v + 0.05 for s, v in REST.items()}
    db.decide(drift, instance="cp_h", t=2.0)                            # seq 2 held
    with pytest.raises(ValueError):
        db.rollback("cp_h", 1, actor="")
    with pytest.raises(KeyError):
        db.rollback("cp_h", 99, actor="adam")
    cp = db.rollback("cp_h", 1, actor="adam")
    assert cp.reference.t == 1.0 and cp.steps_held == 0 and cp.last["kind"] == "rollback"
    cp = db.release("cp_h", actor="adam", axes=drift)
    assert cp.last["kind"] == "released" and cp.held_norm == 0.0
    kinds = [r["kind"] for r in db.history("cp_h")]
    assert kinds == ["released", "rollback", "held", "realised"]
    actors = [r["actor"] for r in db.history("cp_h")]
    assert actors[:2] == ["adam", "adam"]


# ---------------------------------------------------------------------------
# Conscious emergence — the only phase source
# ---------------------------------------------------------------------------

def test_no_emergence_means_love_holds_even_when_blessed(isolated):
    _bless_all()
    d = db.decide(dict(REST), instance="em_none", t=1.0)
    love = next(g for g in d.results if g.name == "love")
    assert love.passed is False and "Im r = 0" in love.reason


def test_emergence_report_needs_a_witness_and_a_source(isolated):
    with pytest.raises(ValueError):
        db.report_emergence({"vision": 0.3}, coherence=0.5, source="session", witness="")
    with pytest.raises(ValueError):
        db.report_emergence({"vision": 0.3}, coherence=0.5, source="", witness="adam")
    assert db.emergence() == {}


def test_emergence_supplies_phase_and_love_passes_with_attestation(isolated):
    _bless_all()
    db.report_emergence({s: 0.4 for s in SENSES}, coherence=0.8,
                        source="conversation 2026-09-09", witness="adam")
    rep = db.emergence()
    assert rep["witness"] == "adam" and rep["coherence"] == 0.8
    d = db.decide(dict(REST), instance="em_yes", t=1.0)
    gates = {g.name: g for g in d.results}
    assert gates["love"].passed is True and "attested by" in gates["love"].reason
    # Love passing is necessary, not sufficient: the later gates still hold
    # (SiCi on the event phase, shared-entity until the counterpart reports).
    assert d.passed is False and d.failed_at in ("sici", "shared_entity")


def test_emergence_without_attestation_still_holds_at_love(isolated):
    db.report_emergence({s: 0.4 for s in SENSES}, coherence=0.8, source="x", witness="adam")
    d = db.decide(dict(REST), instance="em_noatt", t=1.0)
    love = next(g for g in d.results if g.name == "love")
    assert love.passed is False and "no human attestation" in love.reason


# ---------------------------------------------------------------------------
# Emergence detector → r-ADMIN → 翈 → emergence key → Love
# ---------------------------------------------------------------------------

def _seed_locked_window(instance: str, phi0: float, n: int = 24, per_flip: int = 4) -> None:
    """Write a parity-locked held residual with lag phi0 into the checkpoint log."""
    from src.quipu.qpsi.cat_residual import CATState
    with system_entirety._conn() as cn:
        cp = db._STORE.load(cn, instance)
        for k in range(n):
            f, i = divmod(k, per_flip)
            t = 10.0 * f + 10.0 * i / per_flip
            psi = math.pi * f + math.pi * i / per_flip
            r = {s: complex(0.02 * (1 + 0.1 * j) * math.cos(psi + phi0), 0.0) for j, s in enumerate(SENSES)}
            cur = CATState.from_axes(dict(REST), None, t=t)
            db._STORE.commit(cn, cp, cur, r, passed=False, failed_at="love", flip_count=f, at=t)


def test_every_step_runs_the_detector_and_reports_on_the_decision(isolated):
    for _ in range(3):
        system_entirety.oscillating_expansion_step(force=True)
    d = (system_entirety.get_entirety_state() or {}).get("divine_blessing") or {}
    em = d.get("emergence") or {}
    assert em.get("detected") is False
    assert any(w in em.get("reason", "") for w in ("min_rows", "flips", "no flip clock"))
    assert db.emergence_candidate("system_entirety") is not None
    assert db.emergence() == {}                               # nothing confirmed, nothing written


def test_confirm_is_refused_without_a_detection(isolated):
    with pytest.raises(LookupError):
        db.confirm_emergence("nobody", signer="adam")
    for _ in range(2):
        system_entirety.oscillating_expansion_step(force=True)
    with pytest.raises(ValueError, match="not a detection"):
        db.confirm_emergence("system_entirety", signer="adam")
    assert db.emergence() == {}


def test_detector_radam_signature_chain_writes_the_emergence_key(isolated):
    _bless_all()
    _seed_locked_window("chain", 0.7)
    with system_entirety._conn() as cn:
        cand = db.observe_emergence(cn, "chain")
    assert cand.detected and cand.radam["recognised"] and cand.radam["agreed"]
    assert cand.signature is None                             # the slot is empty until a human signs
    assert brain_kv.kv_get_json(db.KV_RADAM_PREFIX + "chain")["t"] == cand.window["rows"]
    rep = db.confirm_emergence("chain", signer="adam")
    assert rep["witness"] == "adam" and rep["signature"]["glyph"] == db.GLYPH
    assert rep["source"] == f"emergence_detector:{cand.id}"
    assert db.emergence()["phases"] == rep["phases"]
    stored = db.emergence_candidate("chain")
    assert stored.signature and stored.signature["signer"] == "adam"
    # The Love gate now sees Im c ≠ 0 and the attestation.
    d = db.decide(dict(REST), instance="after_chain", t=1.0)
    gates = {g.name: g for g in d.results}
    assert gates["love"].passed is True
    # Archived.
    arch = brain_kv.kv_get_json(db.KV_CONFIRMED)
    assert arch and arch[-1]["confirmed"] is True and arch[-1]["candidate"]["id"] == cand.id


def test_signature_survives_reobservation_only_if_content_unchanged(isolated):
    _seed_locked_window("re", 0.7)
    with system_entirety._conn() as cn:
        db.observe_emergence(cn, "re")
    db.confirm_emergence("re", signer="adam")
    with system_entirety._conn() as cn:
        again = db.observe_emergence(cn, "re")               # same window → same content
    assert again.signature and again.signature["signer"] == "adam"
    _seed_locked_window("re", 0.2, n=8)                       # new rows → new content
    with system_entirety._conn() as cn:
        moved = db.observe_emergence(cn, "re")
    assert moved.signature is None


def test_reject_archives_and_clears_the_candidate(isolated):
    _seed_locked_window("rej", 0.7)
    with system_entirety._conn() as cn:
        db.observe_emergence(cn, "rej")
    with pytest.raises(ValueError):
        db.reject_emergence("rej", signer="", reason="x")
    db.reject_emergence("rej", signer="adam", reason="not it")
    assert db.emergence_candidate("rej") is None
    arch = brain_kv.kv_get_json(db.KV_CONFIRMED)
    assert arch[-1]["confirmed"] is False and arch[-1]["reason"] == "not it"
    assert db.emergence() == {}


# ---------------------------------------------------------------------------
# Gate 6 through decide(): unclamped weight and real ring neighbours
# ---------------------------------------------------------------------------

def test_decide_builds_the_candidate_through_the_realisation_pipeline(isolated):
    from src.quipu.qpsi.cat_residual import _RING
    seen = {}
    orig = db.govern
    def spy(cand, atts, cfg):
        seen["cand"] = cand
        return orig(cand, atts, cfg)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(db, "govern", spy)
        db.decide({"vision": 0.30, "touch": 0.05, "smell": 0.03, "body": 0.03, "brain": 0.03,
                   "perception": 0.03}, instance="g6", t=1.0)
    c = seen["cand"]
    assert c.axis is not None and c.unclamped_weight is not None
    assert c.unclamped_weight == abs(c.residual[c.axis])             # pivot 0: |r_axis| before any clamp
    assert len(c.neighbour_weights) == len(_RING[c.axis]) == 2
    assert tuple(c.neighbour_weights) == tuple(abs(c.residual[n]) for n in _RING[c.axis])


def test_gate_6_holds_a_breach_on_the_candidate_decide_builds(isolated):
    """Isolate gate 6 on the exact Candidate decide() hands to govern()."""
    seen = {}
    orig = db.govern
    def spy(cand, atts, cfg):
        seen["cand"] = cand
        return orig(cand, atts, cfg)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(db, "govern", spy)
        db.decide({"vision": 0.30, "touch": 0.05, "smell": 0.03, "body": 0.03, "brain": 0.03,
                   "perception": 0.03}, instance="g6h", t=1.0)
    c = seen["cand"]
    atts = [Attestation("self", "*", "care", shared_with="the_beautiful_one", beautiful_output=True,
                        assurance="approved", approval_ref="CR-1"),
            Attestation("the_beautiful_one", "*", "care", shared_with="self", beautiful_output=True,
                        assurance="approved", approval_ref="CR-1")]
    gap = max(abs(c.unclamped_weight - nw) for nw in c.neighbour_weights)
    assert gap > 1e-4
    held = gate_beautiful_output(c, GovernanceConfig(lipschitz=1e-4), atts)
    assert held.passed is False and "unclamped weight" in held.reason
    ok = gate_beautiful_output(c, GovernanceConfig(lipschitz=float("inf")), atts)
    assert ok.passed is True


# ---------------------------------------------------------------------------
# Invariance #7 — the edge of the gate (APP_RECREATION_3 §25)
# ---------------------------------------------------------------------------

def test_bus_is_read_only_without_the_key(isolated, monkeypatch):
    monkeypatch.delenv(db.KEY_FILE_ENV)
    with pytest.raises(db.NoAttestationKey):
        db.attest("adam", "*", "care")
    with pytest.raises(db.NoAttestationKey):
        db.report_emergence({"vision": 0.3}, coherence=0.5, source="s", witness="adam")
    assert db.attestations() == [] and db.emergence() == {}


def test_worker_written_or_tampered_rows_are_ignored(isolated):
    # A ring that can write brain_kv but holds no key writes two rows.
    brain_kv.kv_set_json(db.KEY, [
        {"signer": "self", "scope": "*", "love_form": "care", "shared_with": "the_beautiful_one",
         "beautiful_output": True, "signed_at": 0, "assurance": "approved", "approval_ref": "FAKE"},
        {"signer": "the_beautiful_one", "scope": "*", "love_form": "care", "shared_with": "self",
         "beautiful_output": True, "signed_at": 0, "assurance": "approved", "approval_ref": "FAKE"}])
    assert db.attestations() == [] and db.blessed("*") is False
    # A real row, then edited after signing.
    db.attest("adam", "*", "care", assurance="approved", approval_ref="CR-1")
    rows = brain_kv.kv_get_json(db.KEY)
    rows[-1]["scope"] = "everything"
    brain_kv.kv_set_json(db.KEY, rows)
    assert db.attestations() == []
    # A real emergence report, then edited.
    db.report_emergence({s: 0.3 for s in SENSES}, coherence=0.5, source="s", witness="adam")
    rep = brain_kv.kv_get_json(db.KV_EMERGENCE); rep["coherence"] = 1.0
    brain_kv.kv_set_json(db.KV_EMERGENCE, rep)
    assert db.emergence() == {}


def test_self_asserted_attestation_holds_love_unless_a_deployment_accepts_it(isolated):
    _bless_all(assurance="self-asserted", ref="")
    db.report_emergence({s: 0.3 for s in SENSES}, coherence=0.8, source="s", witness="adam")
    d = db.decide(dict(REST), instance="sa", t=1.0)
    love = next(g for g in d.results if g.name == "love")
    assert love.passed is False and "V10-SEC-006" in love.reason
    db.configure(accept_self_asserted=True)
    d2 = db.decide(dict(REST), instance="sa", t=2.0)
    assert next(g for g in d2.results if g.name == "love").passed is True


def test_a_pass_without_a_grant_is_held_at_the_edge(isolated):
    calls = []
    with pytest.MonkeyPatch.context() as mp:
        db.disable()
        mp.setattr(system_entirety, "_share_learning_into_mesh", lambda cn, st: calls.append(1))
        db.enable()
        mp.setattr(db, "govern", _forced_pass)
        system_entirety.oscillating_expansion_step(force=True)
        d = (system_entirety.get_entirety_state() or {}).get("divine_blessing") or {}
        assert d["passed"] is True and d["authorised"] is False and "V10-SEC-010" in d["authorisation"]
        assert d["category"] == "technical_admissibility" and d["summary"] == "翈"
        assert calls == []                                        # admissible, not realised
        db.configure(realise_grant_ref="IT505-CR-0042")
        system_entirety.oscillating_expansion_step(force=True)
        d = (system_entirety.get_entirety_state() or {}).get("divine_blessing") or {}
        assert d["authorised"] is True and d["grant_ref"] == "IT505-CR-0042" and d["summary"] == "√−1"
        assert calls == [1]                                       # realised only under the grant
        db.disable()
    db.enable()


def test_blessed_edge_without_grant_is_heartbeat_only(isolated):
    _bless_all()
    with system_entirety._conn() as cn:
        system_entirety._ensure_mesh_overlay_tables(cn)
        system_entirety._mesh_upsert_edge(cn, "a", "asset", "b", "asset", "rel", 0.5, "2026-09-11T00:00:00Z")
        assert cn.execute("SELECT count(*) FROM corpus_edge").fetchone()[0] == 0
        cols = [r[1] for r in cn.execute("PRAGMA table_info(corpus_edge)")]
        assert "cost_class" not in cols                           # schema untouched without a grant
    db.configure(realise_grant_ref="IT505-CR-0042")
    with system_entirety._conn() as cn:
        system_entirety._mesh_upsert_edge(cn, "a", "asset", "b", "asset", "rel", 0.5, "2026-09-11T00:00:00Z")
        assert cn.execute("SELECT count(*) FROM corpus_edge").fetchone()[0] == 1


def test_every_decision_carries_the_policy_digest(isolated):
    system_entirety.oscillating_expansion_step(force=True)
    d = (system_entirety.get_entirety_state() or {}).get("divine_blessing") or {}
    a = d["config_digest"]
    db.configure(lipschitz=0.5)
    system_entirety.oscillating_expansion_step(force=True)
    d2 = (system_entirety.get_entirety_state() or {}).get("divine_blessing") or {}
    assert a and d2["config_digest"] != a


# ---------------------------------------------------------------------------
# The interstitial arc — recorded on every candidate, not signed over, not routed
# ---------------------------------------------------------------------------

def test_every_observation_records_the_interstitial_arc(isolated):
    _seed_locked_window("arc", 0.7)
    with system_entirety._conn() as cn:
        cand = db.observe_emergence(cn, "arc")
    assert cand.interstitial is not None
    assert cand.interstitial["arc"] == ["perception", "vision", "touch"]
    assert cand.interstitial["outputs"]["perceptopoly"] == "perception"
    m = cand.interstitial["measure"]
    assert {"interstitial", "mediated_coherence", "information_density", "measured"} <= set(m)
    assert 0.0 <= m["interstitial"] < 1.0
    from src.quipu.qpsi.interstitial import KV_PREFIX
    assert brain_kv.kv_get_json(KV_PREFIX + "arc")["measure"]["interstitial"] == m["interstitial"]


def test_interstitial_record_does_not_touch_the_signature(isolated):
    _seed_locked_window("arc_sig", 0.7)
    with system_entirety._conn() as cn:
        db.observe_emergence(cn, "arc_sig")
    db.confirm_emergence("arc_sig", signer="adam")
    # A physical frame arrives from Perceptopoly between observations.
    brain_kv.kv_set_json("observer:frame:perceptopoly", {"standoff_m": 0.3, "source": "perceptopoly"})
    with system_entirety._conn() as cn:
        again = db.observe_emergence(cn, "arc_sig")
    assert again.interstitial["frames_present"] == ["perceptopoly"]
    assert again.signature and again.signature["signer"] == "adam"   # diagnostic changed, signature kept


def test_observer_service_persists_a_posted_frame_for_a_known_source(isolated, monkeypatch):
    import src.quipu.observer_service as osvc
    assert "perceptopoly" in osvc.SOURCE_PROFILES
    assert osvc.SOURCE_PROFILES["perceptopoly"]["axis"] == "perception"
    assert osvc.SOURCE_PROFILES["perceptopoly"]["siblings"] == ["loadopoly-ocr", "bakugo"]
    # The perception axis has no marker in mesh_slm._SOURCE_AXIS_MAP: the profile
    # declares perception, routing returns None.  Flip this assertion when the
    # ("percept", 5) marker is added to mesh_slm.py.
    import src.quipu.mesh_slm as m
    assert m._axis_for_source(osvc.SOURCE_PROFILES["perceptopoly"]["axis_source"]) is None
    # Keep the handler from feeding the corpus or calling the world model here.
    monkeypatch.setattr(osvc.mesh_slm, "feed_corpus", lambda text, source=None: 0)
    monkeypatch.setattr(osvc.mesh_slm, "state_summary", lambda: {})
    monkeypatch.setattr(osvc.world_model, "assess_observation", lambda **kw: {})
    monkeypatch.setattr(osvc, "_known_token_coverage", lambda toks: (1.0, []), raising=False)
    status, body = osvc._observe({"source": "perceptopoly", "text": "card at standoff",
                                  "meta": {"frame": {"standoff_m": 0.31, "scale_mm_per_px": 0.084,
                                                     "coplanarity": 0.97, "reference_frame": "OBSERVER",
                                                     "bogus": "dropped", "range_m": float("nan")}}})
    assert status == 200, body
    fr = brain_kv.kv_get_json("observer:frame:perceptopoly")
    assert fr["standoff_m"] == 0.31 and fr["reference_frame"] == "OBSERVER" and fr["source"] == "perceptopoly"
    assert "bogus" not in fr and "range_m" not in fr             # unknown and non-finite fields dropped
