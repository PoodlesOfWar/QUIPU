"""Operator ruling 2026-09-27: the mirror aspect updates entities that already
exist behind the gates and never creates one; a new ACRE specialist passes
gate 6 like every other realisation."""
from __future__ import annotations

import sqlite3

import pytest

import src.quipu.brain_kv as brain_kv
import src.quipu.divine_blessing as db
import src.quipu.system_entirety as se
from src.quipu import entirety_service as ES
from src.quipu import mesh_slm
from src.quipu.qpsi import specialist_gate as SG
from src.quipu.qpsi.governance import GovernanceConfig


@pytest.fixture
def brain(tmp_path, monkeypatch):
    f = tmp_path / "brain.sqlite"
    monkeypatch.setenv("SCB_DB_PATH", str(f))

    def _open(timeout: float = 60, path=None):
        cn = sqlite3.connect(f, timeout=timeout)
        cn.row_factory = sqlite3.Row
        return cn
    monkeypatch.setattr(brain_kv, "open_conn", _open)
    key = tmp_path / "attest.key"
    key.write_bytes(b"test-key")
    monkeypatch.setenv(db.KEY_FILE_ENV, str(key))
    monkeypatch.setattr(db, "_CONFIG", GovernanceConfig(realise_grant_ref="GRANT-TEST"))
    db.enable()
    yield f
    db.disable()
    SG.disable()


def _edges(f):
    cn = sqlite3.connect(f)
    cn.execute("CREATE TABLE IF NOT EXISTS corpus_edge(src_id TEXT, src_type TEXT, dst_id TEXT, dst_type TEXT, "
               "rel TEXT, weight REAL, last_seen TEXT, samples INTEGER DEFAULT 1, "
               "PRIMARY KEY(src_id, src_type, dst_id, dst_type, rel))")
    return cn


def test_mirror_updates_an_existing_gated_edge_and_never_creates_one(brain, monkeypatch):
    cn = _edges(brain)
    cn.execute("INSERT INTO corpus_edge VALUES('a','A','b','B','R',0.2,'t0',3)")
    cn.commit()
    monkeypatch.setenv(db.MIRROR_ENV, "1")
    se._mesh_upsert_edge(cn, "a", "A", "b", "B", "R", 0.6, "t1")       # not blessed: mirror update
    se._mesh_upsert_edge(cn, "a", "A", "c", "C", "R", 0.9, "t1")       # new pair: held
    cn.commit()
    rows = {r[0]: r[1:] for r in cn.execute("SELECT dst_id, weight, samples FROM corpus_edge")}
    assert set(rows) == {"b"}                                          # nothing created
    assert rows["b"][1] == 4 and rows["b"][0] == pytest.approx((0.2 * 3 + 0.6) / 4)
    assert db.mirror_counts()["edge_creations_held"] >= 1


def test_without_the_ruling_an_unblessed_edge_only_gets_a_heartbeat(brain, monkeypatch):
    cn = _edges(brain)
    cn.execute("INSERT INTO corpus_edge VALUES('a','A','b','B','R',0.2,'t0',3)")
    cn.commit()
    monkeypatch.delenv(db.MIRROR_ENV, raising=False)
    se._mesh_upsert_edge(cn, "a", "A", "b", "B", "R", 0.9, "t1")
    cn.commit()
    w, n, seen = cn.execute("SELECT weight, samples, last_seen FROM corpus_edge").fetchone()
    assert (w, n, seen) == (0.2, 3, "t1")


def _fake_acre(name, bias):
    def acre_emerge(*a, **k):
        with mesh_slm._conn() as cn:
            specs = mesh_slm._load_emergent_specialists(cn)
            specs[name] = bias
            mesh_slm._meta_set(cn, mesh_slm._ACRE_META_SPECIALISTS, specs)
        return {"status": "created", "specialist": name, "bias": bias, "resonance": 0.9}
    return acre_emerge


def _attest_both(scope):
    for signer in ("self", "the_beautiful_one"):
        db.attest(signer, scope, "care", beautiful_output=True, assurance="approved", approval_ref="CR-1")


def test_a_new_specialist_is_held_at_gate6_until_both_parties_attest(brain, monkeypatch):
    mesh_slm._EMERGENT_BIASES.clear()
    monkeypatch.setattr(mesh_slm, "acre_emerge", _fake_acre("emergent_x_y", [0.25, 0.1, 0, 0, 0, 0, 0]))
    SG.enable()
    r = mesh_slm.acre_emerge()
    assert r["status"] == "held_at_gate6"
    with mesh_slm._conn() as cn:
        assert "emergent_x_y" not in mesh_slm._load_emergent_specialists(cn)       # not active
    assert SG.review()["held"] == ["emergent_x_y"]
    _attest_both("specialist:emergent_x_y")
    out = SG.review()
    assert out["realised"] == ["emergent_x_y"]
    with mesh_slm._conn() as cn:
        assert "emergent_x_y" in mesh_slm._load_emergent_specialists(cn)           # realised
    assert SG.status()["specialists"]["emergent_x_y"]["status"] == "realised"


def test_one_signer_or_no_grant_does_not_pass(brain, monkeypatch):
    monkeypatch.setattr(mesh_slm, "acre_emerge", _fake_acre("emergent_a_b", [0.25, 0, 0, 0, 0, 0, 0.1]))
    SG.enable()
    mesh_slm.acre_emerge()
    db.attest("self", "specialist:emergent_a_b", "care", beautiful_output=True, assurance="approved",
              approval_ref="CR-1")
    assert SG.review()["held"] == ["emergent_a_b"]                                 # one party only
    db.attest("the_beautiful_one", "specialist:emergent_a_b", "care", beautiful_output=True,
              assurance="approved", approval_ref="CR-1")
    monkeypatch.setattr(db, "_CONFIG", GovernanceConfig(realise_grant_ref=""))
    out = SG.review()
    assert out["held"] == ["emergent_a_b"]                                         # admissible, no grant
    assert SG.status()["specialists"]["emergent_a_b"]["gate6"]["passed"] is True


def test_an_admitted_specialist_keeps_working_while_its_pass_is_pending(brain):
    with mesh_slm._conn() as cn:
        mesh_slm._meta_set(cn, mesh_slm._ACRE_META_SPECIALISTS, {"emergent_brain_smell": [0, 0, 0.23, 0, 0.25, 0, 0.14]})
    e = SG.admit("emergent_brain_smell", note="operator ruling 2026-09-27")
    assert e["status"] == "active_pending" and e["scope"] == "specialist:emergent_brain_smell"
    assert SG.review()["held"] == ["emergent_brain_smell"]
    with mesh_slm._conn() as cn:
        assert "emergent_brain_smell" in mesh_slm._load_emergent_specialists(cn)   # still active
    _attest_both("specialist:emergent_brain_smell")
    assert SG.review()["passed"] == ["emergent_brain_smell"]


def test_the_mirror_loop_is_the_operators_switch_and_runs_every_part(monkeypatch):
    monkeypatch.delenv("QUIPU_MIRROR_UPDATE", raising=False)
    assert not ES.settings()["mirror_update"]
    calls = []
    monkeypatch.setattr(mesh_slm, "train_round", lambda **k: calls.append("train") or {"status": "ok"})
    monkeypatch.setattr(mesh_slm, "observe_interactions", lambda **k: calls.append("obs") or {})
    monkeypatch.setattr(mesh_slm, "acre_emerge", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(SG, "review", lambda: calls.append("review") or {})
    monkeypatch.setattr(ES, "_gard_manifest", lambda: calls.append("gard"))
    monkeypatch.setattr(ES, "_world_model_grounding", lambda: calls.append("wm"))
    monkeypatch.setattr(ES, "expansion_step", lambda: calls.append("step") or {})
    out = ES.mirror()
    assert out["acre"].startswith("error") and calls == ["train", "obs", "review", "gard", "wm", "step"]
