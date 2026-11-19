"""tool_gate: a forge-format ACRE tool candidate is held until both parties attest
to its exact scope and the grant is set."""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.quipu import brain_kv, divine_blessing as db, tool_forge as tf
from src.quipu.qpsi import tool_gate as TG
from src.quipu.qpsi.governance import GovernanceConfig

NAME = "viscous_shear_chain"


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
    monkeypatch.setattr(db, "_CONFIG", GovernanceConfig(realise_grant_ref="GRANT-TEST", lipschitz=0.5))
    tools = tmp_path / "generated"
    monkeypatch.setattr(tf, "_TOOLS_DIR", tools)
    monkeypatch.setattr(tf, "_MANIFEST", tools / "manifest.json")
    yield tmp_path


def _attest_both(scope, signers=("self", "the_beautiful_one"), **kw):
    for s in signers:
        db.attest(s, scope, "care", beautiful_output=True,
                  assurance=kw.get("assurance", "approved"), approval_ref=kw.get("approval_ref", "CR-1"))


def test_lint_accepts_the_committed_candidate():
    assert TG.lint((TG.CANDIDATE_DIR / f"{NAME}.py").read_text()) == (True, "ok")


@pytest.mark.parametrize("bad", [
    "import os\ndef tool_x(data):\n    return {}\n",
    "import json\ndef tool_x(data):\n    return {}\n",
    "def tool_x(data):\n    return open('f')\n",
    "def tool_x(data):\n    return getattr(data, 'k')\n",
    "def tool_x(data):\n    return data.__class__\n",
    "x = 1\ndef tool_x(data):\n    return {}\n",
    "def tool_x(data):\n    return {}\ndef tool_y(data):\n    return {}\n",
    "def tool_x(data):\n    for _ in range(3):\n        pass\n    return {}\n",
])
def test_lint_rejects_forbidden_constructs(bad):
    assert TG.lint(bad)[0] is False


def test_propose_holds_with_hash_scope_and_shadow_token(brain):
    e = TG.propose(NAME)
    assert e["status"] == "held"
    assert e["scope"] == f"tool:{NAME}@{e['sha256']}" and len(e["sha256"]) == 64
    assert e["shadow_token"] and e["evidence"]["lint"] == "ok"
    assert not tf._MANIFEST.exists()
    assert TG.propose(NAME)["created_at"] == e["created_at"]


def test_changed_source_supersedes_and_voids_earlier_approval(brain):
    e = TG.propose(NAME)
    _attest_both(e["scope"])
    src = e["source"] + "\n# revision\n"
    e2 = TG.propose(NAME, src)
    assert e2["scope"] != e["scope"] and e2["history"][0]["sha256"] == e["sha256"]
    assert TG.review(NAME)["held"] == [NAME]


def test_held_without_attestations_one_signer_or_self_assertion(brain, monkeypatch):
    e = TG.propose(NAME)
    assert TG.review()["held"] == [NAME]
    _attest_both(e["scope"], signers=("self",))
    assert TG.review()["held"] == [NAME]
    db.attest("the_beautiful_one", e["scope"], "care", beautiful_output=True)
    assert TG.review()["held"] == [NAME]


def test_exact_scope_pair_plus_grant_realises_and_the_forge_can_load_it(brain):
    e = TG.propose(NAME)
    _attest_both(e["scope"])
    assert TG.review()["realised"] == [NAME]
    rows = json.loads(tf._MANIFEST.read_text())["tools"]
    row = [r for r in rows if r["tool_name"] == f"tool_{NAME}"][0]
    assert row["origin"] == "tool_gate" and row["sha256"] == e["sha256"] and row["forged_at"]
    loaded = tf.load_generated_tools()
    assert f"tool_{NAME}" in loaded
    assert TG.realised_callable(NAME)({})["confidence"] == 0.0
    assert TG.status()[NAME]["live"] is True


def test_wildcard_attestations_do_not_pass(brain):
    TG.propose(NAME)
    _attest_both("*")
    assert TG.review()["held"] == [NAME]


def test_no_grant_stays_held(brain, monkeypatch):
    e = TG.propose(NAME)
    _attest_both(e["scope"])
    monkeypatch.setattr(db, "_CONFIG", GovernanceConfig(realise_grant_ref="", lipschitz=0.5))
    out = TG.review()
    assert out["held"] == [NAME]
    assert brain_kv.kv_get_json(TG.KV_GATE)[NAME]["gate6"]["grant"] is False


def test_reject_blocks_and_withdraws_a_realised_tool(brain):
    e = TG.propose(NAME)
    _attest_both(e["scope"])
    TG.review()
    TG.reject(NAME, "owner withdrew it")
    assert brain_kv.kv_get_json(TG.KV_GATE)[NAME]["status"] == "rejected"
    assert not (tf._TOOLS_DIR / f"tool_{NAME}.py").exists()
    assert TG.realised_callable(NAME) is None and TG.candidate_callable(NAME) is None


def test_realised_callable_rechecks_grant_and_source(brain, monkeypatch):
    e = TG.propose(NAME)
    assert TG.realised_callable(NAME) is None
    _attest_both(e["scope"])
    TG.review()
    assert TG.realised_callable(NAME) is not None
    monkeypatch.setattr(db, "_CONFIG", GovernanceConfig(realise_grant_ref="", lipschitz=0.5))
    assert TG.realised_callable(NAME) is None
    monkeypatch.setattr(db, "_CONFIG", GovernanceConfig(realise_grant_ref="G", lipschitz=0.5))
    d = brain_kv.kv_get_json(TG.KV_GATE)
    d[NAME]["source"] += "\n# tampered\n"
    brain_kv.kv_set_json(TG.KV_GATE, d)
    assert TG.realised_callable(NAME) is None


def test_materialise_restores_a_lost_file(brain):
    e = TG.propose(NAME)
    _attest_both(e["scope"])
    TG.review()
    (tf._TOOLS_DIR / f"tool_{NAME}.py").unlink()
    assert TG.materialise()["restored"] == [NAME]
    assert TG.materialise()["ok"] == [NAME]


def test_review_refuses_a_forge_tool_of_the_same_name(brain):
    e = TG.propose(NAME)
    _attest_both(e["scope"])
    tf._save_manifest({"tools": [{"tool_name": f"tool_{NAME}", "file": f"tool_{NAME}.py"}]})
    out = TG.review()
    assert out["held"] == [NAME] and NAME in out["errors"]


def test_prefix_guard():
    with pytest.raises(PermissionError):
        TG._kv_set("entirety:specialist_gate", {})


def test_show_drafts_commands_and_never_attests(brain, monkeypatch):
    e = TG.propose(NAME)
    monkeypatch.setattr(db, "attest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("attested")))
    text = TG.show(NAME)
    assert e["scope"] in text and "--signer the_beautiful_one" in text and "QUIPU_REALISE_GRANT_REF" in text
    assert TG.status()[NAME]["status"] == "held"
