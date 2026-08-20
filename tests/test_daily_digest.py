"""Tests for daily_digest.py - the 'what has System Entirety learned today' report.

Seeds an isolated mesh DB directly (vocab rows with explicit first_seen/
last_seen dates, a mesh_slm_meta ACRE snapshot, and brain_kv history entries)
so the report's date-filtering logic can be checked deterministically against
two distinct days, independent of when the test actually runs.
"""
from __future__ import annotations

import importlib
import json
import sqlite3

import pytest

TODAY = "2026-07-20"
YESTERDAY = "2026-07-19"


@pytest.fixture
def seeded_digest_db(tmp_path, monkeypatch):
    db_file = tmp_path / "digest.sqlite"
    monkeypatch.setenv("SCB_DB_PATH", str(db_file))

    import src.quipu.local_store as ls
    importlib.reload(ls)
    import src.quipu.mesh_slm as mesh_slm
    importlib.reload(mesh_slm)
    import src.quipu.brain_kv as brain_kv
    importlib.reload(brain_kv)
    import src.quipu.tool_forge as tool_forge
    importlib.reload(tool_forge)
    import src.quipu.daily_digest as digest
    importlib.reload(digest)

    # _forged_tools_today reads tool_forge's manifest straight off disk (it's
    # not SCB_DB_PATH-scoped like everything else) - without this, these tests
    # would silently read whatever real generated-tools manifest already
    # exists on the machine running them. Redirect it into tmp_path.
    monkeypatch.setattr(tool_forge, "_TOOLS_DIR", tmp_path / "generated")
    monkeypatch.setattr(tool_forge, "_MANIFEST", tmp_path / "generated" / "manifest.json")
    tool_forge._save_manifest({"tools": [
        {"tool_name": "tool_holographic", "cluster": "holographic", "cluster_safe": "holographic",
         "file": "tool_holographic.py", "type": "Tool", "novelty": 0.8, "certainty": 0.7,
         "degree": 6, "forged_at": f"{TODAY}T10:12:00+00:00"},
        {"tool_name": "tool_static", "cluster": "static", "cluster_safe": "static",
         "file": "tool_static.py", "type": "Tool", "novelty": 0.6, "certainty": 0.5,
         "degree": 4, "forged_at": f"{YESTERDAY}T09:40:00+00:00"},
    ]})

    mesh_slm.state_summary()  # trigger DDL

    cn = sqlite3.connect(str(db_file))
    cn.row_factory = sqlite3.Row
    # Two tokens first seen yesterday (one reinforced today), one brand-new today.
    cn.execute(
        "INSERT INTO mesh_slm_vocab(token, i, j, freq, first_seen, last_seen) "
        "VALUES ('mesh', 0, 0, 10, ?, ?)",
        (f"{YESTERDAY}T09:00:00+00:00", f"{TODAY}T08:00:00+00:00"),
    )
    cn.execute(
        "INSERT INTO mesh_slm_vocab(token, i, j, freq, first_seen, last_seen) "
        "VALUES ('static', 1, 1, 3, ?, ?)",
        (f"{YESTERDAY}T09:00:00+00:00", f"{YESTERDAY}T09:00:00+00:00"),
    )
    cn.execute(
        "INSERT INTO mesh_slm_vocab(token, i, j, freq, first_seen, last_seen) "
        "VALUES ('holographic', 2, 2, 5, ?, ?)",
        (f"{TODAY}T10:00:00+00:00", f"{TODAY}T10:00:00+00:00"),
    )
    cn.execute(
        "INSERT INTO mesh_slm_meta(key, value) VALUES ('last_trained_at', ?)",
        (json.dumps(f"{TODAY}T10:05:00+00:00"),),
    )
    cn.execute(
        "INSERT INTO mesh_slm_meta(key, value) VALUES ('rounds', ?)",
        (json.dumps(7),),
    )
    cn.execute(
        "INSERT INTO mesh_slm_meta(key, value) VALUES ('acre_specialists', ?)",
        (json.dumps({"emergent_vision_touch": [0.5] * 7, "emergent_brain_entirety": [0.4] * 7}),),
    )
    cn.commit()
    cn.close()

    # brain_kv histories: one entry yesterday (baseline: only one specialist
    # existed then), one entry today (both specialists present -> one is "new").
    brain_kv.kv_set_json("systemic_refinement:history", [
        {"ts": f"{YESTERDAY}T09:30:00+00:00", "actions": ["no_action"],
         "acre_specialists": ["emergent_vision_touch"]},
        {"ts": f"{TODAY}T10:10:00+00:00", "actions": ["forged:1"],
         "acre_specialists": ["emergent_vision_touch", "emergent_brain_entirety"]},
    ])
    brain_kv.kv_set_json("corpus_ingest:history", [
        {"ts": f"{YESTERDAY}T09:35:00+00:00", "total_condensed": 40,
         "per_source": {"arxiv": 40}, "weyl_latest_psi5": [0.1, 0.1, 0.1, 0.1, 0.1],
         "compaction_ratio": 0, "compaction_scalar": 0.01},
        {"ts": f"{TODAY}T10:15:00+00:00", "total_condensed": 80,
         "per_source": {"arxiv": 40, "gutenberg": 40},
         "weyl_latest_psi5": [0.39, 0.93, 0.99, 0.0, 0.62],
         "compaction_ratio": 0, "compaction_scalar": 0.02},
    ])

    return digest


def test_build_digest_today_has_expected_activity(seeded_digest_db):
    digest = seeded_digest_db
    d = digest.build_digest(TODAY)

    assert d["db_exists"] is True
    assert d["nothing_happened"] is False

    assert d["vocab"]["new_count"] == 1
    assert d["vocab"]["new_tokens"][0]["token"] == "holographic"
    assert d["vocab"]["reinforced_count"] == 1
    assert d["vocab"]["reinforced_tokens"][0]["token"] == "mesh"
    assert d["vocab"]["vocab_size_now"] == 3

    assert d["training"]["trained_today"] is True
    assert d["training"]["rounds_total"] == 7

    assert d["ingest"]["cycles_today"] == 1
    assert d["ingest"]["documents_today"] == 80
    assert d["ingest"]["per_source_today"] == {"arxiv": 40, "gutenberg": 40}

    assert d["refinement"]["cycles_today"] == 1
    assert "forged:1" in d["refinement"]["actions_today"]

    assert d["acre"]["current"] == ["emergent_brain_entirety", "emergent_vision_touch"]
    assert d["acre"]["new_today"] == ["emergent_brain_entirety"]

    assert len(d["tools_forged_today"]) == 1
    assert d["tools_forged_today"][0]["tool_name"] == "tool_holographic"


def test_build_digest_yesterday_has_different_activity(seeded_digest_db):
    digest = seeded_digest_db
    d = digest.build_digest(YESTERDAY)

    # Both 'mesh' and 'static' have first_seen = yesterday -> both count as
    # "new" for a yesterday query, ordered by freq desc (mesh=10, static=3).
    assert d["vocab"]["new_count"] == 2
    assert [t["token"] for t in d["vocab"]["new_tokens"]] == ["mesh", "static"]
    assert d["vocab"]["reinforced_count"] == 0     # mesh's last_seen is TODAY, not yesterday -> not "reinforced yesterday"
    assert d["training"]["trained_today"] is False
    assert d["ingest"]["cycles_today"] == 1
    assert d["ingest"]["documents_today"] == 40
    # Baseline before yesterday is empty (no history before it) -> both
    # yesterday's own specialist is "new" relative to nothing.
    assert d["acre"]["new_today"] == ["emergent_vision_touch"]

    assert len(d["tools_forged_today"]) == 1
    assert d["tools_forged_today"][0]["tool_name"] == "tool_static"


def test_build_digest_empty_day_reports_nothing_happened(seeded_digest_db):
    digest = seeded_digest_db
    d = digest.build_digest("2020-01-01")
    assert d["nothing_happened"] is True
    assert "No learning activity" in d["headline"]


def test_build_digest_missing_db_is_handled_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "does_not_exist.sqlite"))
    import src.quipu.local_store as ls
    importlib.reload(ls)
    import src.quipu.daily_digest as digest
    importlib.reload(digest)

    d = digest.build_digest(TODAY)
    assert d["db_exists"] is False
    assert "No mesh database" in d["headline"]


def test_format_digest_produces_readable_report(seeded_digest_db):
    digest = seeded_digest_db
    d = digest.build_digest(TODAY)
    text = digest.format_digest(d)
    assert "System Entirety - daily digest" in text
    assert "holographic" in text
    assert "Specialists" in text
    assert "emergent" in text
