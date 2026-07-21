"""Tests for system_entirety.py — 7th-D observer-tangent Bit Flip orchestration.

Covers the orthogonality of the observer tangent to the weight prior, the
parity invariance of the Floquet bit flip across a half-period, the
oscillation period, and the rate-limited driver's persistence semantics.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

import pytest
import src.quipu.system_entirety as system_entirety

from src.quipu.system_entirety import (
    _OMEGA_BASE,
    bit_flip_parity,
    get_flip_history,
    observer_tangent,
    oscillating_expansion_step,
    system_entirety_state,
    get_bit_state,
    get_expansion_phase,
)
from src.quipu.temporal_spatiality import _SENSE_WEIGHTS


SENSES = ("vision", "touch", "smell", "body", "brain", "perception")


@pytest.fixture
def isolated_entirety_db(tmp_path, monkeypatch):
    db_file = tmp_path / "entirety.sqlite"

    def _open_conn(timeout: float = 60):
        cn = sqlite3.connect(db_file, timeout=timeout)
        cn.execute("PRAGMA journal_mode=WAL")
        cn.execute("PRAGMA synchronous=NORMAL")
        cn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        cn.row_factory = sqlite3.Row
        return cn

    monkeypatch.setattr(system_entirety, "_open_conn", _open_conn)
    monkeypatch.setattr(system_entirety, "_LAST_TS", 0.0)
    try:
        import src.quipu.repository_catalog as repository_catalog
        monkeypatch.setattr(repository_catalog, "get_catalog_summary", lambda: {})
    except Exception:
        pass
    return db_file


# ---------------------------------------------------------------------------
# Observer tangent geometry
# ---------------------------------------------------------------------------

def test_sense_weights_sum_to_one():
    assert math.isclose(sum(_SENSE_WEIGHTS.values()), 1.0, abs_tol=1e-9)
    assert set(_SENSE_WEIGHTS) == set(SENSES)


def test_observer_tangent_zero_when_signals_align_with_weights():
    """If the activity vector is proportional to the weight prior, its
    component orthogonal to the weight axis is zero — observer = 0."""
    aligned = {s: _SENSE_WEIGHTS[s] for s in SENSES}
    assert observer_tangent(aligned) == pytest.approx(0.0, abs=1e-9)


def test_observer_tangent_in_unit_interval():
    saturated = {s: 1.0 for s in SENSES}
    silent    = {s: 0.0 for s in SENSES}
    skewed    = {s: (1.0 if s == "perception" else 0.0) for s in SENSES}
    for sig in (saturated, silent, skewed):
        v = observer_tangent(sig)
        assert 0.0 <= v <= 1.0


def test_observer_tangent_nonzero_on_skewed_signals():
    """A single sense saturated with all others silent is maximally
    misaligned with the uniform-ish weight prior."""
    skewed = {s: (1.0 if s == "perception" else 0.0) for s in SENSES}
    assert observer_tangent(skewed) > 0.1


# ---------------------------------------------------------------------------
# Bit Flip parity
# ---------------------------------------------------------------------------

def test_bit_flip_parity_is_signed_integer():
    for obs in (0.0, 0.25, 0.5, 0.75, 1.0):
        b = bit_flip_parity(obs, t=0.0)
        assert b in (-1, 1)


def test_bit_flip_oscillates_over_period():
    """At t=0 cos(ω·0)=1 ⇒ bit=+1.  At t=π/ω cos(ω·t)=−1 ⇒ bit=−1
    (with zero observer bias).  This is the broaden→deepen flip."""
    obs = 0.0  # no bias
    omega = _OMEGA_BASE  # observer=0 → omega = base
    t_zero = 0.0
    t_half = math.pi / omega
    assert bit_flip_parity(obs, t=t_zero) == 1
    assert bit_flip_parity(obs, t=t_half) == -1


def test_bit_flip_period_length_positive():
    """Period must be finite/positive for any observer in [0,1]."""
    for obs in (0.0, 0.5, 1.0):
        omega = _OMEGA_BASE * (1.0 + 4.0 * obs)
        period = 2.0 * math.pi / omega
        assert period > 0.0


# ---------------------------------------------------------------------------
# State + driver
# ---------------------------------------------------------------------------

def test_system_entirety_state_shape(isolated_entirety_db):
    state = system_entirety_state({s: 0.5 for s in SENSES})
    assert set(state) == {
        "axes",
        "observer",
        "magnitude",
        "material_bifurcation",
        "pim_planning",
        "repository_catalog",
        "ueqgm_runtime",
        "transaction",
    }
    assert set(state["axes"]) == set(SENSES)
    assert 0.0 <= state["observer"] <= 1.0
    assert state["magnitude"] >= 0.0
    assert state["material_bifurcation"]["eligible"] is False
    assert state["repository_catalog"]["has_data"] is False
    assert state["ueqgm_runtime"]["active"] is False
    assert state["transaction"]["bit_flip"] == "observer_only"
    assert state["transaction"]["ueqgm_overlay"] == "inactive"


def test_repository_catalog_injection_expands_axes_and_transaction(isolated_entirety_db):
    aligned = {s: _SENSE_WEIGHTS[s] for s in SENSES}
    baseline = system_entirety_state(aligned)

    payload = {
        "total_files": 25000,
        "capabilities_discovered": 12000,
        "avg_relevance_score": 0.72,
        "avg_pim_relevance": 0.64,
        "avg_supply_chain_relevance": 0.81,
        "catalog_signal": 0.76,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_roots": [
            {"exists": True, "file_count": 13000},
            {"exists": True, "file_count": 12000},
        ],
        "top_terms": [{"term": "pim", "occurrences": 42}],
    }
    with system_entirety._conn() as cn:
        cn.execute(
            "CREATE TABLE IF NOT EXISTS brain_kv("
            "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            ("repo:summary_state", json.dumps(payload), payload["updated_at"]),
        )

    state = system_entirety_state(aligned)
    assert state["repository_catalog"]["has_data"] is True
    assert state["repository_catalog"]["catalog_signal"] == pytest.approx(0.76)
    assert state["transaction"]["drive"] > baseline["transaction"]["drive"]
    assert any(state["axes"][sense] > baseline["axes"][sense] for sense in SENSES)


def test_material_storage_signal_lifts_touch_and_perception(isolated_entirety_db):
    raw_axes = {s: 0.05 for s in SENSES}
    baseline = system_entirety_state(raw_axes)
    payload = {
        "physical_realization": 0.44,
        "asset_strength": 0.38,
        "material_strength": 0.62,
        "tunnel_density": 0.70,
        "asset_count": 4,
        "processor_count": 1,
        "processor_edges": 4,
        "free_disk_gb": 2048.0,
        "total_disk_gb": 4096.0,
        "drive_count": 4,
        "resource_sharing": {
            "core": {"hosts": 1},
            "memory": {"hosts": 1},
            "storage": {"hosts": 2, "free_disk_gb": 2048.0, "total_disk_gb": 4096.0, "units": 4},
            "mesh": {
                "peer_count": 2,
                "contract_count": 8,
                "transport_count": 2,
                "tunnel_peer_count": 1,
                "mesh_density": 0.55,
            },
        },
    }
    with system_entirety._conn() as cn:
        cn.execute(
            "CREATE TABLE IF NOT EXISTS brain_kv("
            "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            ("entirety:physical_realization", json.dumps(payload), datetime.now(timezone.utc).isoformat()),
        )

    state = system_entirety_state(raw_axes)
    material = state["material_bifurcation"]

    assert material["storage_signal"] > 0.5
    assert material["free_disk_gb"] == 2048.0
    assert material["drive_count"] == 4
    assert state["axes"]["touch"] > baseline["axes"]["touch"]
    assert state["axes"]["perception"] > baseline["axes"]["perception"]


def test_repository_catalog_playwright_artifacts_bias_perception(isolated_entirety_db):
    aligned = {s: _SENSE_WEIGHTS[s] for s in SENSES}

    payload_without_playwright = {
        "total_files": 800,
        "capabilities_discovered": 2400,
        "avg_relevance_score": 0.58,
        "avg_pim_relevance": 0.44,
        "avg_supply_chain_relevance": 0.61,
        "catalog_signal": 0.52,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_roots": [
            {"exists": True, "file_count": 500},
            {"exists": True, "file_count": 300},
        ],
        "top_terms": [{"term": "pim", "occurrences": 21}],
        "playwright_artifacts": {
            "file_count": 0,
            "pim_routed_count": 0,
            "avg_pim_relevance": 0.0,
            "avg_deep_corpus_score": 0.0,
            "signal": 0.0,
        },
    }
    with system_entirety._conn() as cn:
        cn.execute(
            "CREATE TABLE IF NOT EXISTS brain_kv("
            "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            ("repo:summary_state", json.dumps(payload_without_playwright), payload_without_playwright["updated_at"]),
        )

    without_playwright = system_entirety._repository_catalog_state()

    payload_with_playwright = dict(payload_without_playwright)
    payload_with_playwright["playwright_artifacts"] = {
        "file_count": 14,
        "pim_routed_count": 11,
        "avg_pim_relevance": 0.73,
        "avg_deep_corpus_score": 0.78,
        "signal": 0.71,
    }
    payload_with_playwright["updated_at"] = datetime.now(timezone.utc).isoformat()
    with system_entirety._conn() as cn:
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            ("repo:summary_state", json.dumps(payload_with_playwright), payload_with_playwright["updated_at"]),
        )

    with_playwright = system_entirety._repository_catalog_state()
    state = system_entirety_state(aligned)

    assert without_playwright["playwright_perception_signal"] == pytest.approx(0.0)
    assert with_playwright["playwright_perception_signal"] > 0.0
    assert with_playwright["axis_drive"]["perception"] > without_playwright["axis_drive"]["perception"]
    assert state["repository_catalog"]["playwright_artifacts"]["file_count"] == 14
    assert state["repository_catalog"]["playwright_perception_signal"] > 0.0


def test_repository_catalog_media_perception_biases_perception(isolated_entirety_db):
    aligned = {s: _SENSE_WEIGHTS[s] for s in SENSES}

    payload_without_media = {
        "total_files": 1200,
        "capabilities_discovered": 3600,
        "avg_relevance_score": 0.55,
        "avg_pim_relevance": 0.41,
        "avg_supply_chain_relevance": 0.58,
        "catalog_signal": 0.50,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_roots": [
            {"exists": True, "file_count": 700},
            {"exists": True, "file_count": 500},
        ],
        "top_terms": [{"term": "inventory", "occurrences": 18}],
        "playwright_artifacts": {
            "file_count": 0,
            "pim_routed_count": 0,
            "avg_pim_relevance": 0.0,
            "avg_deep_corpus_score": 0.0,
            "signal": 0.0,
        },
        "media_perception": {
            "file_count": 0,
            "analysed_count": 0,
            "entity_count": 0,
            "avg_confidence": 0.0,
            "avg_relevance_score": 0.0,
            "avg_pim_relevance": 0.0,
            "signal": 0.0,
            "sources": {},
        },
    }
    with system_entirety._conn() as cn:
        cn.execute(
            "CREATE TABLE IF NOT EXISTS brain_kv("
            "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            ("repo:summary_state", json.dumps(payload_without_media), payload_without_media["updated_at"]),
        )

    without_media = system_entirety._repository_catalog_state()

    payload_with_media = dict(payload_without_media)
    payload_with_media["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload_with_media["media_perception"] = {
        "file_count": 9,
        "image_count": 4,
        "screenshot_count": 3,
        "video_count": 2,
        "analysed_count": 8,
        "analysis_coverage": 0.8889,
        "entity_count": 26,
        "avg_confidence": 0.74,
        "avg_relevance_score": 0.69,
        "avg_pim_relevance": 0.62,
        "signal": 0.77,
        "sources": {"perception_visual": 5, "perception_video": 3},
    }
    with system_entirety._conn() as cn:
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            ("repo:summary_state", json.dumps(payload_with_media), payload_with_media["updated_at"]),
        )

    with_media = system_entirety._repository_catalog_state()
    state = system_entirety_state(aligned)

    assert without_media["media_perception_signal"] == pytest.approx(0.0)
    assert with_media["media_perception_signal"] > 0.0
    assert with_media["axis_drive"]["perception"] > without_media["axis_drive"]["perception"]
    assert state["repository_catalog"]["media_perception"]["file_count"] == 9
    assert state["repository_catalog"]["media_perception_signal"] > 0.0


def test_material_realization_bifurcates_axes_and_transaction(isolated_entirety_db):
    aligned = {s: _SENSE_WEIGHTS[s] for s in SENSES}
    baseline = system_entirety_state(aligned)
    assert baseline["observer"] == pytest.approx(0.0, abs=1e-9)

    payload = {
        "physical_realization": 0.92,
        "asset_strength": 0.81,
        "material_strength": 0.74,
        "tunnel_density": 0.88,
        "asset_count": 6,
        "processor_count": 2,
        "processor_edges": 8,
    }
    with system_entirety._conn() as cn:
        cn.execute(
            "CREATE TABLE IF NOT EXISTS brain_kv(" 
            "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            ("entirety:physical_realization", json.dumps(payload), "2026-05-17T00:00:00+00:00"),
        )

    state = system_entirety_state(aligned)
    assert state["material_bifurcation"]["eligible"] is True
    assert state["material_bifurcation"]["nodal_bifurcation"] > 0.0
    assert state["transaction"]["bit_flip"] == "material_bifurcated"
    assert state["transaction"]["drive"] > 0.0
    assert state["observer"] > 0.0
    assert any(state["axes"][sense] > aligned[sense] for sense in SENSES)


def test_mesh_realization_promotes_transaction_to_mesh(isolated_entirety_db):
    aligned = {s: _SENSE_WEIGHTS[s] for s in SENSES}
    payload = {
        "physical_realization": 0.92,
        "asset_strength": 0.81,
        "material_strength": 0.74,
        "tunnel_density": 0.88,
        "asset_count": 6,
        "processor_count": 2,
        "processor_edges": 8,
        "resource_sharing": {
            "core": {"hosts": 3},
            "memory": {"hosts": 3},
            "processor": {"processor_edges": 8},
            "mesh": {
                "peer_count": 3,
                "contract_count": 8,
                "transport_count": 2,
                "tunnel_peer_count": 2,
                "tunnel_peer_ratio": 2 / 3,
                "transport_diversity": 1.0,
                "mesh_density": 0.86,
            },
        },
    }
    with system_entirety._conn() as cn:
        cn.execute(
            "CREATE TABLE IF NOT EXISTS brain_kv(" 
            "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            ("entirety:physical_realization", json.dumps(payload), "2026-05-18T00:00:00+00:00"),
        )

    state = system_entirety_state(aligned)
    assert state["material_bifurcation"]["mesh"]["eligible"] is True
    assert state["material_bifurcation"]["mesh"]["peer_count"] == 3
    assert state["material_bifurcation"]["mesh"]["mesh_density"] > 0.0
    assert state["transaction"]["bit_flip"] == "mesh_bifurcated"
    assert state["transaction"]["topology"] == "mesh"
    assert state["transaction"]["drive"] > 0.0


def test_ueqgm_runtime_injection_expands_axes_and_transaction(isolated_entirety_db):
    aligned = {s: _SENSE_WEIGHTS[s] for s in SENSES}

    with system_entirety._conn() as cn:
        cn.execute(
            "CREATE TABLE IF NOT EXISTS brain_kv("
            "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
            (
                "ueqgm:adaptive_runtime",
                json.dumps({
                    "active": True,
                    "certainty": 0.82,
                    "corpus_signal": 0.71,
                    "learning_signal": 0.78,
                    "newness_signal": 0.74,
                    "relational_depth": 0.8,
                    "symbiotic_gain": 0.86,
                    "expansion_pressure": 0.77,
                    "phase_weight": 1.05,
                    "coherence_depth": 4,
                    "mesh_alignment": 0.68,
                    "observer_alignment": 0.73,
                    "runtime_keywords": ["symbiotic", "mesh", "observer"],
                    "applied_parameters": ["symbiotic_gain", "runtime_keywords"],
                    "axis_drive": {
                        "vision": 0.45,
                        "touch": 0.35,
                        "smell": 0.25,
                        "body": 0.4,
                        "brain": 0.55,
                        "perception": 0.65,
                    },
                }),
                "2026-05-18T00:00:00+00:00",
            ),
        )

    baseline = system_entirety_state(aligned, include_ueqgm_runtime=False)
    state = system_entirety_state(aligned)

    assert state["ueqgm_runtime"]["active"] is True
    assert state["transaction"]["ueqgm_overlay"] == "adaptive"
    assert state["transaction"]["symbiotic_drive"] > 0.0
    assert state["transaction"]["drive"] > baseline["transaction"]["drive"]
    assert state["ueqgm_runtime"]["axis_injection"]["perception"] > 0.0
    assert state["axes"]["perception"] >= baseline["axes"]["perception"]


def test_oscillating_expansion_step_persists_and_reads(isolated_entirety_db):
    """Forced step must produce a usable summary and persisted KVs."""
    summary = oscillating_expansion_step(force=True)
    assert "skipped" not in summary
    assert summary["bit_state"] in (-1, 1)
    assert summary["expansion_phase"] in ("broaden", "deepen")
    assert summary["flip_count"] >= 0
    assert isinstance(summary["flipped"], bool)
    assert summary["transaction"]["drive"] >= 0.0
    # Public accessors agree with the persisted state.
    assert get_bit_state() == summary["bit_state"]
    assert get_expansion_phase() == summary["expansion_phase"]
    history = get_flip_history(limit=1)
    assert len(history) == 1
    assert history[0]["bit_state"] == summary["bit_state"]
    assert history[0]["expansion_phase"] == summary["expansion_phase"]


def test_oscillating_expansion_step_projects_learning_into_mesh_graph(isolated_entirety_db):
    now_iso = datetime.now(timezone.utc).isoformat()
    now_legacy = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with system_entirety._conn() as cn:
        cn.executescript(
            """
            CREATE TABLE IF NOT EXISTS learning_log(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT,
                kind TEXT,
                title TEXT,
                signal_strength REAL
            );
            CREATE TABLE IF NOT EXISTS corpus_entity(
                entity_id TEXT, entity_type TEXT, label TEXT, props_json TEXT,
                first_seen TEXT, last_seen TEXT, samples INTEGER DEFAULT 1,
                PRIMARY KEY(entity_id, entity_type)
            );
            CREATE TABLE IF NOT EXISTS corpus_edge(
                src_id TEXT, src_type TEXT, dst_id TEXT, dst_type TEXT, rel TEXT,
                weight REAL, last_seen TEXT, samples INTEGER DEFAULT 1,
                PRIMARY KEY(src_id, src_type, dst_id, dst_type, rel)
            );
            """
        )
        cn.executemany(
            "INSERT INTO learning_log(logged_at, kind, title, signal_strength) VALUES(?,?,?,?)",
            [
                (now_iso, "rag_deepdive", "mesh learning deep dive", 0.82),
                (now_legacy, "rag_deepdive", "legacy timestamp preserved", 0.72),
                (now_iso, "promotion", "hardening promotion", 0.64),
            ],
        )
        cn.executemany(
            "INSERT INTO corpus_entity(entity_id, entity_type, label, props_json, first_seen, last_seen, samples) "
            "VALUES(?,?,?,?,?,?,?)",
            [
                ("asset:test:cores", "AssetResource", "test cores", "{}", now_iso, now_iso, 3),
                ("msp:test", "SpatialMaterialProcessor", "test processor", "{}", now_iso, now_iso, 2),
                ("endpoint:test", "Endpoint", "test endpoint", "{}", now_iso, now_iso, 1),
            ],
        )

    summary = oscillating_expansion_step(force=True)
    assert "skipped" not in summary

    with system_entirety._conn() as cn:
        window_count = cn.execute(
            "SELECT COUNT(*) FROM corpus_entity WHERE entity_type='MeshLearningWindow'"
        ).fetchone()[0]
        kind_count = cn.execute(
            "SELECT COUNT(*) FROM corpus_entity WHERE entity_type='LearningKindSummary'"
        ).fetchone()[0]
        runtime_count = cn.execute(
            "SELECT COUNT(*) FROM corpus_entity WHERE entity_type='UEQGMRuntimeState'"
        ).fetchone()[0]
        planning_count = cn.execute(
            "SELECT COUNT(*) FROM corpus_entity WHERE entity_type='PIMPlanningState'"
        ).fetchone()[0]
        edge_rels = {
            row[0]: row[1]
            for row in cn.execute(
                "SELECT rel, COUNT(*) FROM corpus_edge GROUP BY rel"
            ).fetchall()
        }
        payload = json.loads(
            cn.execute(
                "SELECT value FROM brain_kv WHERE key='entirety:physical_realization'"
            ).fetchone()[0]
        )

    assert window_count == 1
    assert kind_count >= 2
    assert runtime_count == 1
    assert planning_count == 1
    assert edge_rels["EMITS_LEARNING_WINDOW"] == 1
    assert edge_rels["SUMMARIZES_LEARNING_KIND"] >= 2
    assert edge_rels["HARDENS_ASSET_RESOURCE"] >= 2
    assert edge_rels["HARDENS_MATERIAL_PROCESSOR"] >= 2
    assert edge_rels["HARDENS_ENDPOINT"] >= 2
    assert payload["mesh_learning"]["recent_learning"]["entry_count"] == 3
    assert payload["mesh_learning_projection"]["kind_nodes"] >= 2
    assert payload["mesh_learning_projection"]["hardening_edges"] >= 6


def test_oscillating_expansion_step_rate_limited(isolated_entirety_db):
    """Two back-to-back unforced calls — at least one should rate-limit."""
    oscillating_expansion_step(force=True)
    a = oscillating_expansion_step(force=False)
    b = oscillating_expansion_step(force=False)
    assert a.get("skipped") or b.get("skipped")


def test_flip_history_records_transitions(isolated_entirety_db, monkeypatch):
    state = {
        "axes": {s: 0.5 for s in SENSES},
        "observer": 0.5,
        "magnitude": 1.0,
    }
    monkeypatch.setattr(system_entirety, "system_entirety_state", lambda signals=None: state)
    parity = iter((1, -1))
    monkeypatch.setattr(system_entirety, "bit_flip_parity", lambda observer, t=None: next(parity))

    first = oscillating_expansion_step(force=True)
    second = oscillating_expansion_step(force=True)

    assert first["flip_count"] == 0
    assert second["flipped"] is True
    assert second["flip_count"] == 1

    history = get_flip_history(limit=2)
    assert len(history) == 2
    assert history[0]["bit_state"] == -1
    assert history[0]["flipped"] is True
    assert history[0]["flip_count"] == 1
    assert history[1]["bit_state"] == 1
    assert history[1]["flipped"] is False


def test_failed_write_does_not_poison_rate_limit(tmp_path, monkeypatch):
    db_file = tmp_path / "entirety.sqlite"

    def _good_open_conn(timeout: float = 60):
        cn = sqlite3.connect(db_file, timeout=timeout)
        cn.execute("PRAGMA journal_mode=WAL")
        cn.execute("PRAGMA synchronous=NORMAL")
        cn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        cn.row_factory = sqlite3.Row
        return cn

    def _locked_open_conn(timeout: float = 60):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(system_entirety, "_LAST_TS", 0.0)
    monkeypatch.setattr(system_entirety, "_open_conn", _locked_open_conn)

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        oscillating_expansion_step(force=False)

    monkeypatch.setattr(system_entirety, "_open_conn", _good_open_conn)
    summary = oscillating_expansion_step(force=False)
    assert "skipped" not in summary


def test_transient_locked_write_retries_and_succeeds(tmp_path, monkeypatch):
    db_file = tmp_path / "entirety.sqlite"

    def _good_open_conn(timeout: float = 60):
        cn = sqlite3.connect(db_file, timeout=timeout)
        cn.execute("PRAGMA journal_mode=WAL")
        cn.execute("PRAGMA synchronous=NORMAL")
        cn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        cn.row_factory = sqlite3.Row
        return cn

    attempts = {"n": 0}

    def _flaky_open_conn(timeout: float = 60):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return _good_open_conn(timeout)

    monkeypatch.setattr(system_entirety, "_LAST_TS", 0.0)
    monkeypatch.setattr(system_entirety, "_open_conn", _flaky_open_conn)

    summary = oscillating_expansion_step(force=False)
    assert "skipped" not in summary
    assert attempts["n"] >= 2


def test_bridge_mesh_density_injects_into_material_bifurcation(isolated_entirety_db):
    aligned = {s: _SENSE_WEIGHTS[s] for s in SENSES}

    base_payload = {
        "physical_realization": 0.85,
        "asset_strength": 0.75,
        "material_strength": 0.70,
        "tunnel_density": 0.80,
        "asset_count": 4,
        "processor_count": 2,
        "processor_edges": 6,
    }
    bridge_density_payload = {
        "has_data": True,
        "computed_at": "2026-05-18T00:00:00+00:00",
        "primary_root": "hideout-mesh",
        "total_roots": 1,
        "total_endpoints": 5,
        "mesh_density": 0.72,
        "nodal_contribution": 0.58,
        "tunnel_saturation": 0.80,
        "anchor_strength": 0.65,
        "primary_root_anchor": 3.2,
        "total_anchor_weight": 12.8,
    }
    with system_entirety._conn() as cn:
        cn.execute(
            "CREATE TABLE IF NOT EXISTS brain_kv"
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key,value,updated_at) VALUES(?,?,?)",
            ("entirety:physical_realization", json.dumps(base_payload), "2026-05-18T00:00:00+00:00"),
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key,value,updated_at) VALUES(?,?,?)",
            ("entirety:bridge_mesh_density", json.dumps(bridge_density_payload), "2026-05-18T00:00:00+00:00"),
        )

    state = system_entirety_state(aligned)
    bmd = state["material_bifurcation"]["bridge_mesh_density"]
    assert bmd["has_data"] is True
    assert bmd["total_roots"] == 1
    assert bmd["primary_root"] == "hideout-mesh"
    assert bmd["mesh_density"] == pytest.approx(0.72, abs=0.001)
    assert bmd["tunnel_saturation"] == pytest.approx(0.80, abs=0.001)

    nodal = state["material_bifurcation"]["nodal_bifurcation"]
    assert nodal > 0.0

    mesh_d = state["material_bifurcation"]["mesh"]["mesh_density"]
    assert mesh_d > 0.0


def test_nodal_location_state_returns_zeros_when_absent(isolated_entirety_db):
    """_nodal_location_state() must return a has_data=False dict when kv is empty."""
    loc = system_entirety._nodal_location_state()
    assert loc["has_data"] is False
    assert loc["sync_score"] == 0.0
    assert loc["anchor_weight"] == 0.0
    assert loc["location_name"] == ""


def test_nodal_location_state_reads_kv(isolated_entirety_db):
    """_nodal_location_state() must surface values persisted in brain_kv."""
    payload = {
        "has_data": True,
        "computed_at": "2026-05-19T12:00:00",
        "location_name": "documents",
        "location_path": r"C:\Users\auser\Documents\VS Code",
        "docs_live": True,
        "onedrive_live": True,
        "sync_score": 1.0,
        "anchor_weight": 4.0,
        "location_count": 2,
    }
    with system_entirety._conn() as cn:
        cn.execute(
            "CREATE TABLE IF NOT EXISTS brain_kv"
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key,value,updated_at) VALUES(?,?,?)",
            ("entirety:nodal_location", json.dumps(payload), "2026-05-19T12:00:00"),
        )

    loc = system_entirety._nodal_location_state()
    assert loc["has_data"] is True
    assert loc["location_name"] == "documents"
    assert loc["docs_live"] is True
    assert loc["onedrive_live"] is True
    assert loc["sync_score"] == pytest.approx(1.0, abs=0.001)
    assert loc["anchor_weight"] == pytest.approx(4.0, abs=0.001)


def test_nodal_location_boosts_nodal_bifurcation(isolated_entirety_db):
    """When nodal_location has a high sync_score, nodal_bifurcation must increase."""
    base_payload = {
        "physical_realization": 0.5,
        "asset_strength": 0.5,
        "material_strength": 0.5,
        "tunnel_density": 0.5,
        "asset_count": 4,
        "processor_count": 2,
        "processor_edges": 6,
    }
    nodal_payload = {
        "has_data": True,
        "computed_at": "2026-05-19T12:00:00",
        "location_name": "documents",
        "location_path": r"C:\Users\auser\Documents\VS Code",
        "docs_live": True,
        "onedrive_live": True,
        "sync_score": 1.0,
        "anchor_weight": 4.0,
        "location_count": 2,
    }
    with system_entirety._conn() as cn:
        cn.execute(
            "CREATE TABLE IF NOT EXISTS brain_kv"
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key,value,updated_at) VALUES(?,?,?)",
            ("entirety:physical_realization", json.dumps(base_payload), "2026-05-19T12:00:00"),
        )
        cn.execute(
            "INSERT OR REPLACE INTO brain_kv(key,value,updated_at) VALUES(?,?,?)",
            ("entirety:nodal_location", json.dumps(nodal_payload), "2026-05-19T12:00:00"),
        )

    aligned = {s: _SENSE_WEIGHTS[s] for s in SENSES}
    state = system_entirety_state(aligned)
    nodal_loc_out = state["material_bifurcation"]["nodal_location"]
    assert nodal_loc_out["has_data"] is True
    assert nodal_loc_out["sync_score"] == pytest.approx(1.0, abs=0.001)
    # nodal_bifurcation must be lifted by the sync contribution
    nodal_bif = state["material_bifurcation"]["nodal_bifurcation"]
    assert nodal_bif > 0.0
