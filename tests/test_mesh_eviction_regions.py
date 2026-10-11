"""Eviction on a saturated torus, the generated-text gate, and regions (v0.56.0)."""
from __future__ import annotations

import importlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def slm(tmp_path, monkeypatch):
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "brain.sqlite"))
    monkeypatch.delenv("QUIPU_TRAIN_GENERATED", raising=False)
    import src.quipu.local_store as ls
    importlib.reload(ls)
    import src.quipu.mesh_slm as m
    importlib.reload(m)
    m._LAST_TRAIN_TS = 0.0
    m._PROTECTED_CACHE.update(at=0.0, words=frozenset())
    return m


def _fill(m, n_tokens, born_days_ago, seen_days_ago, freq=5, prefix="old"):
    """Put tokens straight onto distinct cells with chosen ages and frequencies."""
    born = (datetime.now(timezone.utc) - timedelta(days=born_days_ago)).isoformat()
    seen = (datetime.now(timezone.utc) - timedelta(days=seen_days_ago)).isoformat()
    with m._conn() as cn:
        start = int(cn.execute("SELECT COUNT(*) FROM mesh_slm_vocab").fetchone()[0])
        for k in range(n_tokens):
            i, j = divmod(start + k, m._TORUS_N)
            cn.execute("INSERT INTO mesh_slm_vocab(token, i, j, freq, first_seen, last_seen) VALUES(?,?,?,?,?,?)",
                       (f"{prefix}{start + k}", i, j, freq, born, seen))


def test_a_newcomer_is_not_the_next_victim(slm, monkeypatch):
    m = slm
    monkeypatch.setattr(m, "_TORUS_N", 4)
    monkeypatch.setattr(m, "_VOCAB_LIMIT", 16)
    _fill(m, 15, born_days_ago=30, seen_days_ago=1, freq=5)        # established tokens
    _fill(m, 1, born_days_ago=0, seen_days_ago=0, freq=1, prefix="new")   # a newcomer at frequency 1
    with m._conn() as cn:
        v = m._eviction_victim(cn)
    assert not v["token"].startswith("new")                        # the old rule would have taken it


def test_long_silent_tokens_give_way_before_recent_ones(slm, monkeypatch):
    m = slm
    _fill(m, 1, born_days_ago=200, seen_days_ago=120, freq=40, prefix="silent")   # 40 halved ~8.6 times
    _fill(m, 1, born_days_ago=200, seen_days_ago=0, freq=3, prefix="recent")
    with m._conn() as cn:
        assert m._eviction_victim(cn)["token"].startswith("silent")


def test_the_shadow_words_are_never_evicted(slm, tmp_path, monkeypatch):
    m = slm
    essay = tmp_path / "essay.md"
    essay.write_text("# Canvas\n\n## I. Ground\n\nThe chthonic ground holds the river.\n", encoding="utf-8")
    monkeypatch.setenv("QUIPU_ESSAY_PATH", str(essay))
    _fill(m, 1, born_days_ago=300, seen_days_ago=300, freq=1, prefix="chthonic")
    with m._conn() as cn:
        cn.execute("UPDATE mesh_slm_vocab SET token='chthonic'")
        cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        cn.execute("INSERT OR REPLACE INTO brain_kv(key, value) VALUES('mesh:protected_tokens', ?)", (json.dumps(["keepme"]),))
    _fill(m, 1, born_days_ago=300, seen_days_ago=300, freq=1, prefix="keepme")
    with m._conn() as cn:
        cn.execute("UPDATE mesh_slm_vocab SET token='keepme' WHERE token LIKE 'keepme%'")
    _fill(m, 1, born_days_ago=300, seen_days_ago=10, freq=50, prefix="plain")
    with m._conn() as cn:
        words = m.protected_tokens(cn, refresh=True)
        assert {"chthonic", "ground", "river", "keepme"} <= words
        assert m._eviction_victim(cn)["token"].startswith("plain")   # a far stronger token, but unprotected


def test_forget_tokens_frees_cells_and_takes_edges_with_them(slm):
    m = slm
    m.feed_corpus("gard_desktop asset_resource_tunnel cores ram storage", source="corpus_arxiv")
    m.train_round(max_seconds=5, max_chunks=10)
    r = m.forget_tokens(["gard_desktop", "asset_resource_tunnel", "never_there"])
    assert r == {"removed": ["gard_desktop", "asset_resource_tunnel"], "absent": ["never_there"]}
    with m._conn() as cn:
        ids = {row[0] for row in cn.execute("SELECT token_id FROM mesh_slm_vocab")}
        for table in ("mesh_slm_quipu", "mesh_slm_region_edge"):
            assert all(s in ids and d in ids for s, d in cn.execute(f"SELECT src, dst FROM {table}"))
        assert cn.execute("SELECT COUNT(*) FROM mesh_slm_vocab WHERE token IN ('gard_desktop','asset_resource_tunnel')").fetchone()[0] == 0


def test_generated_text_does_not_train_unless_allowed(slm, monkeypatch):
    m = slm
    with m._conn() as cn:
        cn.executescript("""
            CREATE TABLE IF NOT EXISTS corpus_entity(entity_id TEXT, entity_type TEXT, label TEXT, props_json TEXT,
                first_seen TEXT, last_seen TEXT, samples INTEGER DEFAULT 1, PRIMARY KEY(entity_id, entity_type));
            INSERT INTO corpus_entity VALUES('asset:x','AssetResource','GARD_Desktop CORES asset','{}','2026','2026',1);
        """)
        assert m._corpus_stream(cn) == []
    monkeypatch.setenv("QUIPU_TRAIN_GENERATED", "1")
    with m._conn() as cn:
        assert ("AssetResource GARD_Desktop CORES asset", "corpus_entity") in m._corpus_stream(cn)


def test_empty_text_is_not_fed(slm):
    m = slm
    assert m.feed_corpus("{}", source="hubcore/self-docs/hideout-mesh") == 0
    with m._conn() as cn:
        assert cn.execute("SELECT COUNT(*) FROM mesh_corpus_feed").fetchone()[0] == 0


def test_regions_record_where_an_association_was_learned(slm):
    m = slm
    assert m.region_for_source("perceptopoly/percept/hideout-mesh") == "game"
    assert m.region_for_source("corpus_arxiv") == "reading"
    assert m.region_for_source("corpus_local_docs") == "self"
    assert m.region_for_source("stockopoly/code/hideout-mesh") == "operations"
    assert m.region_for_source("corpus_entity") == "generated"
    assert m.region_for_source("something-new") == "unclassed"
    m.feed_corpus("stray dog guard tension two regimes switching", source="perceptopoly/percept/hideout-mesh")
    m.feed_corpus("guard rotation shift warehouse dock", source="stockopoly/code/hideout-mesh")
    m.train_round(max_seconds=5, max_chunks=10)
    s = m.region_summary()
    assert set(s["regions"]) == {"game", "operations"}
    with m._conn() as cn:
        guard = cn.execute("SELECT token_id FROM mesh_slm_vocab WHERE token='guard'").fetchone()[0]
        regions = {r[0]: r[1] for r in cn.execute("SELECT region, freq FROM mesh_slm_region_vocab WHERE token_id=?", (guard,))}
        assert regions == {"game": 1, "operations": 1}                      # one token, one cell, two provenances
        g_edges = cn.execute("SELECT COUNT(*) FROM mesh_slm_region_edge WHERE region='game'").fetchone()[0]
        total = cn.execute("SELECT COUNT(*) FROM mesh_slm_quipu").fetchone()[0]
    assert g_edges == 6 and total == 10                                     # regions are a subset of the global mesh
