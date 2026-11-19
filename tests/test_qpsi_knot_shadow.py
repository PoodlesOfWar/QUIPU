"""knot_shadow: read-only gathering, no execution while held, proposals only."""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.quipu import brain_kv, divine_blessing as db, tool_forge as tf
from src.quipu.qpsi import knot_shadow as KS, tool_gate as TG
from src.quipu.qpsi.governance import GovernanceConfig

SERIES = [0.1, 0.5, 0.2, 0.3, 0.9, 0.4, 0.41, 0.2]


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
    cn = sqlite3.connect(f)
    cn.executescript("""
        CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
        CREATE TABLE mesh_slm_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE mesh_slm_vocab(token_id INTEGER PRIMARY KEY, token TEXT UNIQUE, i INT, j INT, freq INT);
        CREATE TABLE mesh_slm_quipu(src INT, dst INT, weight REAL, samples INT, PRIMARY KEY(src, dst));
    """)
    cn.execute("INSERT INTO mesh_slm_meta VALUES('entropy_differential_history', ?)", (json.dumps(SERIES),))
    for tid, tok, i, j in [(1, "alpha", 0, 0), (2, "beta", 1, 1), (3, "gamma", 40, 40)]:
        cn.execute("INSERT INTO mesh_slm_vocab VALUES(?,?,?,?,1)", (tid, tok, i, j))
    cn.executemany("INSERT INTO mesh_slm_quipu VALUES(?,?,?,1)", [(1, 2, 0.5), (1, 3, 0.4)])
    for k, v in [("entirety:expansion_phase", "broaden"),
                 ("recursive_strengthening", {"actionable_potential": 0.9})]:
        cn.execute("INSERT INTO brain_kv(key, value) VALUES(?,?)", (k, json.dumps(v)))
    cn.commit()
    cn.close()
    yield f


def _npc(f, agent, magnitude, at, schema=KS.SCHEMA, received="2026-11-19T10:00:00+00:00"):
    cn = sqlite3.connect(f)
    rec = {"agent_id": agent, "schema": schema, "magnitude": magnitude, "direction": 1.0, "at": at,
           "source": "perceptopoly", "received_at": received}
    idx = json.loads((cn.execute("SELECT value FROM brain_kv WHERE key='relational:index'").fetchone() or ["{}"])[0])
    idx[f"perceptopoly:{agent}"] = {"schema": schema}
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value) VALUES(?,?)", (f"relational:perceptopoly:{agent}", json.dumps(rec)))
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value) VALUES('relational:index', ?)", (json.dumps(idx),))
    cn.commit()
    cn.close()


def _snapshot(f):
    cn = sqlite3.connect(f)
    out = {t: cn.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall()
           for t in ("mesh_slm_meta", "mesh_slm_vocab", "mesh_slm_quipu")}
    out["kv"] = cn.execute("SELECT key, value FROM brain_kv WHERE key NOT LIKE 'entirety:knot_shadow:%' "
                           "AND key NOT LIKE 'entirety:tool_gate%' AND key != 'DIVINE_BLESSING_SQRT(-1)' "
                           "ORDER BY key").fetchall()
    cn.close()
    return out


def _realise():
    e = TG.propose(KS.TOOL)
    for s in ("self", "the_beautiful_one"):
        db.attest(s, e["scope"], "care", beautiful_output=True, assurance="approved", approval_ref="CR-1")
    assert TG.review()["realised"] == [KS.TOOL]


NOW = 1795082400.0  # 2026-11-19T10:00:00Z


def test_gather_reads_inputs_and_supernova_records_by_schema_only(brain):
    _npc(brain, "npc-7", 0.7, "2026-11-19T09:59:00+00:00")
    _npc(brain, "other", 0.9, "2026-11-19T09:59:00+00:00", schema="perceptopoly.control/1")
    _npc(brain, "broken", "huge", "2026-11-19T09:59:00+00:00")
    cn = KS._ro_conn()
    d = KS.gather(cn, NOW)
    cn.close()
    assert d["entropy_series"] == SERIES and d["broaden"] == 1.0 and d["actionable_potential"] == 0.9
    assert [n["id"] for n in d["npcs"]] == ["npc-7"]


def test_effective_time_is_the_earlier_of_at_and_received(brain):
    _npc(brain, "liar", 0.5, "2030-01-01T00:00:00+00:00")
    cn = KS._ro_conn()
    d = KS.gather(cn, NOW)
    cn.close()
    assert d["npcs"][0]["t"] == pytest.approx(NOW)


def test_mesh_connection_is_read_only(brain):
    cn = KS._ro_conn()
    with pytest.raises(sqlite3.OperationalError):
        cn.execute("DELETE FROM mesh_slm_vocab")
    cn.close()


def test_held_tool_executes_nothing(brain):
    TG.propose(KS.TOOL)
    before = _snapshot(brain)
    out = KS.run(now=NOW)
    assert out["status"] == "held" and out["executed"] is False
    assert _snapshot(brain) == before and KS.latest() == {}


def test_dry_run_is_explicit_and_never_counts(brain):
    TG.propose(KS.TOOL)
    _npc(brain, "npc-7", 0.7, "2026-11-19T09:59:00+00:00")
    out = KS.run(now=NOW, dry_run=True)
    assert out["status"] == "dry_run" and out["mode"] == "dry_run"
    assert brain_kv.kv_get_json(KS.KV_PREFIX + "history", None) is None
    assert brain_kv.kv_get_json(KS.KV_PREFIX + "dry_run")["shadow_token"]


def test_live_run_publishes_and_needs_two_distinct_states(brain):
    _realise()
    _npc(brain, "npc-7", 0.7, "2026-11-19T09:59:00+00:00")
    before = _snapshot(brain)
    first = KS.run(now=NOW)
    assert first["status"] == "live" and first["shadow_token"] and first["token_proposals"] == []
    again = KS.run(now=NOW + 1)
    assert again["sessions"] == 1
    assert _snapshot(brain) == before
    _npc(brain, "npc-8", 0.4, "2026-11-19T09:59:30+00:00")
    before = _snapshot(brain)
    second = KS.run(now=NOW + 2)
    assert second["sessions"] == 2 and second["token_proposals"]
    assert all(p["realisation"].startswith("none") for p in second["token_proposals"])
    assert {(e["src"], e["dst"]) for e in second["edge_proposals"]} <= {("alpha", "beta")}
    assert _snapshot(brain) == before


def test_gate_zero_gives_no_proposals(brain):
    _realise()
    KS.run(now=NOW)
    out = KS.run(now=NOW + 5)
    assert out["gate"]["value"] == 0.0 and out["token_proposals"] == [] and out["edge_proposals"] == []


def test_recurrence_agrees_with_relational_shear_proposals():
    from src.quipu.qpsi.relational_shear import proposals
    hist = [{"session": s, "tokens": [{"token": "a", "lobe": "R", "lift": 0.1}, {"token": b, "lobe": "R", "lift": 0.1}]}
            for s, b in (("s1", "b"), ("s2", "c"))]
    ours = {t["token"] for t in KS.recurring(hist)["tokens"]}
    theirs = {p["token"] for p in proposals([{"session": h["session"], "outlier": {"axis": "R"},
                                              "shadow_tokens": [{"token": t["token"], "plane": 0}
                                                                for t in h["tokens"]]} for h in hist])}
    assert ours == theirs == {"a"}


def test_writes_only_its_prefix_and_history_is_capped(brain, monkeypatch):
    with pytest.raises(PermissionError):
        KS._kv_set("entirety:tool_gate", {})
    _realise()
    monkeypatch.setattr(KS, "HISTORY_CAP", 3)
    for k in range(5):
        _npc(brain, f"npc-{k}", 0.5, "2026-11-19T09:59:00+00:00")
        KS.run(now=NOW + k)
    assert len(brain_kv.kv_get_json(KS.KV_PREFIX + "history")) == 3
