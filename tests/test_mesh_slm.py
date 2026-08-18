"""Tests for mesh_slm.py — toroidal quipu SLM trained by System Entirety."""
from __future__ import annotations

import json
import math
import random
import sqlite3
import pytest
@pytest.fixture
def isolated_slm_db(tmp_path, monkeypatch):
    db_file = tmp_path / "slm.sqlite"

    monkeypatch.setenv("SCB_DB_PATH", str(db_file))

    # Import after env var is set so local_store.db_path() picks it up.
    import importlib
    import src.quipu.local_store as ls
    importlib.reload(ls)
    import src.quipu.mesh_slm as mesh_slm
    importlib.reload(mesh_slm)

    # Seed corpus_entity + corpus_edge so train_round has text to chew on.
    cn = sqlite3.connect(str(db_file))
    cn.executescript("""
        CREATE TABLE IF NOT EXISTS corpus_entity(
            entity_id TEXT, entity_type TEXT, label TEXT, props_json TEXT,
            first_seen TEXT, last_seen TEXT, samples INTEGER DEFAULT 1,
            PRIMARY KEY(entity_id, entity_type));
        CREATE TABLE IF NOT EXISTS corpus_edge(
            src_id TEXT, src_type TEXT, dst_id TEXT, dst_type TEXT, rel TEXT,
            weight REAL, last_seen TEXT, samples INTEGER DEFAULT 1,
            PRIMARY KEY(src_id, src_type, dst_id, dst_type, rel));
    """)
    samples = [
        ("part_001", "Part",     "carbide insert assembly"),
        ("part_002", "Part",     "hydraulic cylinder piston rod"),
        ("vend_010", "Vendor",   "contoso3pl supply chain"),
        ("site_aa",  "Site",     "siteb plant warehouse"),
        ("po_55",    "PO",       "open purchase order pending"),
        ("oh_77",    "OnHand",   "stock balance available"),
    ]
    for eid, et, lab in samples:
        cn.execute(
            "INSERT INTO corpus_entity(entity_id, entity_type, label, "
            "first_seen, last_seen) VALUES(?,?,?,'2026-05-01','2026-05-20')",
            (eid, et, lab),
        )
    edges = [
        ("part_001", "Part", "vend_010", "Vendor", "SUPPLIED_BY"),
        ("part_002", "Part", "vend_010", "Vendor", "SUPPLIED_BY"),
        ("vend_010", "Vendor", "site_aa", "Site", "DELIVERS_TO"),
        ("po_55",   "PO",   "part_001", "Part", "ORDERS"),
        ("oh_77",   "OnHand","part_001","Part", "MEASURES"),
    ]
    for src_id, src_t, dst_id, dst_t, rel in edges:
        cn.execute(
            "INSERT INTO corpus_edge(src_id, src_type, dst_id, dst_type, rel, "
            "weight, last_seen) VALUES(?,?,?,?,?, 0.8, '2026-05-20')",
            (src_id, src_t, dst_id, dst_t, rel),
        )
    cn.commit()
    cn.close()

    yield mesh_slm


def test_train_round_creates_vocab_and_quipu(isolated_slm_db):
    slm = isolated_slm_db
    summary = slm.train_round(max_seconds=10.0, max_chunks=50)
    assert summary["status"] == "ok"
    assert summary["tokens"] > 0
    assert summary["pairs"] > 0
    state = slm.state_summary()
    assert state["vocab_size"] > 0
    assert state["quipu_edges"] > 0
    assert state["rounds"] == 1


def test_generate_emits_text_after_training(isolated_slm_db):
    slm = isolated_slm_db
    # Two rounds so quipu has real bigram mass.
    slm.train_round(max_seconds=10.0, max_chunks=50)
    # Reset rate limiter so a second round actually runs.
    slm._LAST_TRAIN_TS = 0.0
    slm.train_round(max_seconds=10.0, max_chunks=50)

    out = slm.generate("contoso3pl supply", max_new_tokens=8, seed=42)
    assert "text" in out
    assert out["tokens_emitted"] >= 1
    assert 0.0 <= out["confidence"] <= 1.0


def test_slm_caller_classify_returns_label(isolated_slm_db):
    slm = isolated_slm_db
    slm.train_round(max_seconds=10.0, max_chunks=50)
    fake_decision = type("D", (), {"model_id": "mesh-slm", "score": 0.7})()
    out = slm.slm_caller(fake_decision,
                         {"kind": "classify",
                          "labels": ["carbide insert", "hydraulic cylinder", "unknown"]},
                         {})
    assert out["source"] == "mesh_slm"
    assert out["label"] in ("carbide insert", "hydraulic cylinder", "unknown")
    assert 0.0 <= out["confidence"] <= 1.0


def test_slm_caller_low_confidence_raises(isolated_slm_db):
    """Empty/unseen prompt with cold vocab should raise so caller falls back."""
    slm = isolated_slm_db
    # Force tiny vocab — only one chunk
    slm.train_round(max_seconds=10.0, max_chunks=50)
    fake_decision = type("D", (), {"model_id": "mesh-slm", "score": 0.7})()
    # A totally OOV text prompt should produce low confidence → raise.
    with pytest.raises(slm.MeshSLMUnavailable):
        # Force confidence floor very high so we deterministically trip the fallback.
        original_floor = slm._CONF_FLOOR
        slm._CONF_FLOOR = 0.99
        try:
            slm.slm_caller(fake_decision, "xyzzy quux nonexistenttoken", {})
        finally:
            slm._CONF_FLOOR = original_floor


def test_map_resuscitation_quipu_populates_full_torus(isolated_slm_db):
    slm = isolated_slm_db

    from src.quipu.local_store import db_path

    cn = sqlite3.connect(str(db_path()))
    cn.row_factory = sqlite3.Row
    cn.executescript("""
        CREATE TABLE IF NOT EXISTS kv_store(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS brain_kv(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    cn.execute(
        "INSERT OR REPLACE INTO kv_store(key, value) VALUES(?, ?)",
        ("torus_amplify:bridge:hideout-mesh", "2.6"),
    )
    cn.execute(
        "INSERT OR REPLACE INTO kv_store(key, value) VALUES(?, ?)",
        ("torus_amplify:peer:DESKTOP-01", "1.8"),
    )
    cn.execute(
        "INSERT OR REPLACE INTO kv_store(key, value) VALUES(?, ?)",
        (
            "vram_rehydrate:asset:DESKTOP-01:vram",
            json.dumps(
                {
                    "target": "asset:DESKTOP-01:vram",
                    "rehydrate_weight": 3.13508,
                    "weyl": 2.253,
                    "weyl_boost": 1.2058,
                    "physical_realization": 0.60756,
                    "mesh_density": 0.475,
                    "harmonic_factor": 3.5636,
                    "polarity_weight": 1.25,
                }
            ),
        ),
    )
    cn.execute(
        "INSERT OR REPLACE INTO brain_kv(key, value) VALUES(?, ?)",
        (
            "temporal_spatiality_rhythm",
            json.dumps({"weyl": 2.253, "boost": 1.2058}),
        ),
    )
    cn.execute(
        "INSERT OR REPLACE INTO brain_kv(key, value) VALUES(?, ?)",
        (
            "entirety:physical_realization",
            json.dumps(
                {
                    "physical_realization": 0.60756,
                    "resource_sharing": {"mesh": {"mesh_density": 0.475}},
                }
            ),
        ),
    )
    cn.commit()
    cn.close()

    summary = slm.map_resuscitation_quipu()

    assert summary["node_count"] == 4096
    assert summary["source_label"] == "asset:DESKTOP-01:vram"
    assert summary["directed_stride"] > 0
    assert summary["peak_weight"] > 0.0

    cn = sqlite3.connect(str(db_path()))
    cn.row_factory = sqlite3.Row
    count = cn.execute(
        "SELECT COUNT(*) AS c FROM mesh_slm_quipu_node"
    ).fetchone()["c"]
    sample = cn.execute(
        "SELECT * FROM mesh_slm_quipu_node WHERE node_id=0"
    ).fetchone()
    cn.close()

    assert count == 4096
    assert sample["source_label"] == "asset:DESKTOP-01:vram"
    assert 0 <= sample["directed_target"] < 4096
    assert sample["resuscitation_weight"] > 0.0

    state = slm.state_summary()
    assert state["resuscitation_quipu"]["node_count"] == 4096


# ---------------------------------------------------------------------------
# 8th-D MESH field and UEQGM aspect integral tests
# ---------------------------------------------------------------------------

def test_train_round_reports_ueqgm_fields(isolated_slm_db):
    """train_round summary must include the three new UEQGM-derived fields."""
    slm = isolated_slm_db
    summary = slm.train_round(max_seconds=10.0, max_chunks=50)
    assert summary["status"] == "ok"
    # phase_weight must be a positive scalar (UEQGM adaptive runtime output)
    assert "phase_weight" in summary
    assert summary["phase_weight"] > 0.0
    # wavefunction_overlap must be a valid probability (squared cosine ∈ [0,1])
    assert "wavefunction_overlap" in summary
    assert 0.0 <= summary["wavefunction_overlap"] <= 1.0
    # mesh_field_8d must be a valid probability in [0, 1]
    assert "mesh_field_8d" in summary
    assert 0.0 <= summary["mesh_field_8d"] <= 1.0


def test_mesh_field_8d_is_in_unit_interval(isolated_slm_db):
    """_mesh_field_8d must always return a float in [0, 1]."""
    slm = isolated_slm_db
    slm.train_round(max_seconds=10.0, max_chunks=50)

    from src.quipu.local_store import db_path

    cn = sqlite3.connect(str(db_path()))
    cn.row_factory = sqlite3.Row
    mesh = slm._mesh_state_7d()
    field = slm._mesh_field_8d(cn, mesh)
    cn.close()
    assert 0.0 <= field <= 1.0, f"mesh_field_8d out of range: {field}"


def test_generate_exposes_mesh_field_8d(isolated_slm_db):
    """generate() return dict must include mesh_field_8d key."""
    slm = isolated_slm_db
    slm.train_round(max_seconds=10.0, max_chunks=50)
    slm._LAST_TRAIN_TS = 0.0
    slm.train_round(max_seconds=10.0, max_chunks=50)

    out = slm.generate("supply chain", max_new_tokens=8, seed=7)
    assert "mesh_field_8d" in out
    assert 0.0 <= out["mesh_field_8d"] <= 1.0


def test_state_summary_exposes_mesh_field_8d(isolated_slm_db):
    """state_summary() must expose mesh_field_8d, last_phase_weight, and
    last_wavefunction_overlap after at least one training round."""
    slm = isolated_slm_db
    slm.train_round(max_seconds=10.0, max_chunks=50)

    state = slm.state_summary()
    assert "mesh_field_8d" in state
    assert 0.0 <= state["mesh_field_8d"] <= 1.0
    assert "last_phase_weight" in state
    assert state["last_phase_weight"] is not None
    assert "last_wavefunction_overlap" in state
    assert state["last_wavefunction_overlap"] is not None


def test_slm_caller_text_path_exposes_mesh_field_8d(isolated_slm_db):
    """slm_caller text path must propagate mesh_field_8d from generate()."""
    slm = isolated_slm_db
    slm.train_round(max_seconds=10.0, max_chunks=50)
    slm._LAST_TRAIN_TS = 0.0
    slm.train_round(max_seconds=10.0, max_chunks=50)

    fake_decision = type("D", (), {"model_id": "mesh-slm", "score": 0.7})()
    # Use a known in-vocab phrase to maximise confidence
    try:
        out = slm.slm_caller(fake_decision, "contoso3pl supply chain", {})
    except slm.MeshSLMUnavailable:
        # Low confidence is acceptable in a cold fixture; just verify no crash
        return
    assert "mesh_field_8d" in out
    assert 0.0 <= out["mesh_field_8d"] <= 1.0


def test_mean_embed_7d_uses_highest_freq_tokens(isolated_slm_db):
    """_mean_embed_7d() must select by frequency, not insertion order.

    We insert two tokens with very different embeddings into the SLM tables
    directly: one token with high freq and one with low freq.  The high-freq
    token gets limit=1, so _mean_embed_7d with limit=1 should return *its*
    embedding, not the most-recently-inserted token's.
    """
    slm = isolated_slm_db
    # Ensure tables exist with a quick train.
    slm.train_round(max_seconds=5.0, max_chunks=20)

    from src.quipu.local_store import db_path

    cn = sqlite3.connect(str(db_path()))
    cn.row_factory = sqlite3.Row

    # Insert two tokens: high-freq (freq=1000) and low-freq (freq=1).
    # We insert low-freq *after* high-freq so token_id-DESC ordering would pick it.
    cn.execute(
        "INSERT OR IGNORE INTO mesh_slm_vocab(token, i, j, freq, first_seen, last_seen) "
        "VALUES('__highfreq__', 60, 60, 1000, '2026-01-01', '2026-06-01')"
    )
    hf_id = cn.execute(
        "SELECT token_id FROM mesh_slm_vocab WHERE token='__highfreq__'"
    ).fetchone()["token_id"]
    # Embed with a distinctive all-1.0 vector.
    cn.execute(
        "INSERT OR REPLACE INTO mesh_slm_embed(token_id, "
        "e_vision, e_touch, e_smell, e_body, e_brain, e_perception, e_entirety) "
        "VALUES(?, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)",
        (hf_id,),
    )

    cn.execute(
        "INSERT OR IGNORE INTO mesh_slm_vocab(token, i, j, freq, first_seen, last_seen) "
        "VALUES('__lowfreq__', 61, 61, 1, '2026-01-01', '2026-06-01')"
    )
    lf_id = cn.execute(
        "SELECT token_id FROM mesh_slm_vocab WHERE token='__lowfreq__'"
    ).fetchone()["token_id"]
    # Embed with a distinctive all-0.0 vector.
    cn.execute(
        "INSERT OR REPLACE INTO mesh_slm_embed(token_id, "
        "e_vision, e_touch, e_smell, e_body, e_brain, e_perception, e_entirety) "
        "VALUES(?, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)",
        (lf_id,),
    )
    cn.commit()

    # With limit=1 the function must pick the single highest-freq token (__highfreq__).
    result = slm._mean_embed_7d(cn, limit=1)
    cn.close()

    # All dims should be 1.0 (the high-freq token's embedding), not 0.0.
    assert all(v > 0.9 for v in result), (
        f"Expected high-freq embedding (≈1.0), got {result}"
    )


# ---------------------------------------------------------------------------
# STP-style geodesic diagnostic tests (docs/STP_DIAGNOSTIC_PLAN.md)
# ---------------------------------------------------------------------------

def test_train_round_reports_stp_diagnostic_fields(isolated_slm_db):
    """train_round summary must carry stp_embed_gap / stp_torus_gap; when not
    None each is a float in [0, 2] (cosine-distance range)."""
    slm = isolated_slm_db
    summary = slm.train_round(max_seconds=10.0, max_chunks=50)
    assert summary["status"] == "ok"
    assert "stp_embed_gap" in summary
    assert "stp_torus_gap" in summary
    for key in ("stp_embed_gap", "stp_torus_gap"):
        val = summary[key]
        if val is not None:
            assert isinstance(val, float)
            assert 0.0 <= val <= 2.0, f"{key} out of range: {val}"


def test_stp_cos_gap_collinear_is_zero(isolated_slm_db):
    """Three collinear equally-spaced points → gap ≈ 0 (same direction)."""
    slm = isolated_slm_db
    a = [0.0, 0.0, 0.0]
    b = [1.0, 1.0, 1.0]
    c = [2.0, 2.0, 2.0]
    gap = slm._stp_cos_gap(a, b, c)
    assert gap is not None
    assert abs(gap) < 1e-9


def test_stp_cos_gap_orthogonal_is_one(isolated_slm_db):
    """(c − b) ⊥ (b − a) → cos = 0 → gap ≈ 1.0."""
    slm = isolated_slm_db
    a = [0.0, 0.0]
    b = [1.0, 0.0]   # b - a = (1, 0)
    c = [1.0, 1.0]   # c - b = (0, 1), orthogonal to (1, 0)
    gap = slm._stp_cos_gap(a, b, c)
    assert gap is not None
    assert abs(gap - 1.0) < 1e-9


def test_stp_cos_gap_degenerate_returns_none(isolated_slm_db):
    """a == b → zero-length difference vector → None, not 0/1 or an exception."""
    slm = isolated_slm_db
    assert slm._stp_cos_gap([1.0, 2.0], [1.0, 2.0], [3.0, 4.0]) is None
    # also the other side: c == b
    assert slm._stp_cos_gap([0.0, 0.0], [1.0, 1.0], [1.0, 1.0]) is None


def test_torus_point_r4_wraps(isolated_slm_db):
    """Adjacent cells across the wrap boundary map to nearby R^4 points, unlike
    naive (i, j) subtraction which would show a jump of _TORUS_N - 1."""
    slm = isolated_slm_db
    p0 = slm._torus_point_r4((0, 0))
    pn = slm._torus_point_r4((slm._TORUS_N - 1, 0))
    r4_dist = math.dist(p0, pn)
    # One cell apart on a unit circle: chord = 2·sin(π/N) — small.
    assert r4_dist < 0.2, f"wrap neighbours too far in R^4: {r4_dist}"
    # Contrast: the naive coordinate jump the embedding is designed to avoid.
    naive_jump = abs(0 - (slm._TORUS_N - 1))
    assert naive_jump == slm._TORUS_N - 1


def test_stp_diagnostic_disabled_by_flag(isolated_slm_db, monkeypatch):
    """QUIPU_STP_DIAGNOSTIC=0 → round still ok, gap fields None (fail-to-legacy)."""
    slm = isolated_slm_db
    monkeypatch.setenv("QUIPU_STP_DIAGNOSTIC", "0")
    summary = slm.train_round(max_seconds=10.0, max_chunks=50)
    assert summary["status"] == "ok"
    assert summary.get("stp_embed_gap") is None
    assert summary.get("stp_torus_gap") is None


def test_stp_diagnostic_history_capped(isolated_slm_db):
    """Rolling STP history must never exceed _STP_HISTORY_CAP."""
    slm = isolated_slm_db
    random.seed(1234)
    cap = slm._STP_HISTORY_CAP
    # Pre-fill to the cap so any further append must be sliced back down.
    with slm._conn() as cn:
        slm._meta_set(cn, "stp_torus_gap_history", [0.5] * cap)
        slm._meta_set(cn, "stp_embed_gap_history", [0.5] * cap)
        slm._meta_set(cn, "loss_history", [0.1] * cap)
    for _ in range(5):
        slm._LAST_TRAIN_TS = 0.0
        slm.train_round(max_seconds=10.0, max_chunks=50)
    with slm._conn() as cn:
        hist_t = slm._meta_get(cn, "stp_torus_gap_history", [])
        hist_e = slm._meta_get(cn, "stp_embed_gap_history", [])
        hist_l = slm._meta_get(cn, "loss_history", [])
    assert len(hist_t) <= cap
    assert len(hist_e) <= cap
    assert len(hist_l) <= cap


# ---------------------------------------------------------------------------
# Real-vs-computational entropy differential (ΔS) tests
# ---------------------------------------------------------------------------

def test_entropy_differential_nonneg_and_present(isolated_slm_db):
    """After a training round the differential history exists, and ΔS >= 0
    (Cauchy–Schwarz: exact degree-pair sum <= mean-field closed form)."""
    slm = isolated_slm_db
    slm.train_round(max_seconds=10.0, max_chunks=50)
    with slm._conn() as cn:
        hist = slm._meta_get(cn, "entropy_differential_history", [])
        last = slm._meta_get(cn, "last_entropy_differential", None)
    assert hist, "entropy_differential_history not populated by train_round"
    assert last is not None
    for v in hist:
        assert v >= -1e-9, f"ΔS must be non-negative (Cauchy–Schwarz), got {v}"


def test_entropy_differential_empty_graph_is_none(isolated_slm_db):
    """No vocab / no edges → None (no differential to speak of), never raises."""
    slm = isolated_slm_db
    with slm._conn() as cn:
        assert slm._entropy_differential(cn) is None


def test_entropy_differential_star_beats_chain(isolated_slm_db):
    """Hub structure raises ΔS: a star graph's exact-vs-mean-field gap must
    exceed a regular cycle's (which is ≈ 0 by Cauchy–Schwarz equality)."""
    slm = isolated_slm_db
    with slm._conn() as cn:
        # Build vocab of 8 tokens.
        ids = []
        for k in range(8):
            ids.append(
                slm._upsert_token(cn, f"__dstar{k}__", "2026-08-18T00:00:00", mesh=[0.5] * slm._EMBED_DIM)
            )
        # Star: token 0 connected to all others.
        for k in range(1, 8):
            cn.execute(
                "INSERT OR REPLACE INTO mesh_slm_quipu(src, dst, weight, samples) "
                "VALUES(?, ?, 0.5, 1)",
                (ids[0], ids[k]),
            )
        star_ds = slm._entropy_differential(cn)
        # Replace with a cycle (2-regular): 0→1→…→7→0.
        cn.execute("DELETE FROM mesh_slm_quipu")
        for k in range(8):
            cn.execute(
                "INSERT INTO mesh_slm_quipu(src, dst, weight, samples) "
                "VALUES(?, ?, 0.5, 1)",
                (ids[k], ids[(k + 1) % 8]),
            )
        cycle_ds = slm._entropy_differential(cn)
    assert star_ds is not None and cycle_ds is not None
    assert abs(cycle_ds) < 1e-6, f"regular cycle should have ΔS ≈ 0, got {cycle_ds}"
    assert star_ds > 0.01, f"star should have clear positive ΔS, got {star_ds}"


def test_pearson_corr_basic(isolated_slm_db):
    """Perfect anti-correlation → −1; perfect correlation → +1; degenerate → None."""
    slm = isolated_slm_db
    assert slm._pearson_corr([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert slm._pearson_corr([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0
    assert slm._pearson_corr([1, 1, 1], [1, 2, 3]) is None   # zero variance
    assert slm._pearson_corr([1, 2], [1, 2]) is None          # too short


def test_stp_trend_reports_delta_s_signature(isolated_slm_db):
    """stp_diagnostic_trend must surface the ΔS↔STP-torus anti-correlation:
    rising ΔS against falling torus gap → negative corr → signature True."""
    slm = isolated_slm_db
    n = 40
    with slm._conn() as cn:
        # ΔS rising (hubs forming), torus gap falling (geodesics learned):
        # perfect anti-correlation, well past the −0.5·Ω_Λ threshold.
        slm._meta_set(cn, "entropy_differential_history",
                      [round(0.001 * k, 6) for k in range(n)])
        slm._meta_set(cn, "stp_torus_gap_history",
                      [round(0.5 - 0.01 * k, 6) for k in range(n)])
        slm._meta_set(cn, "stp_embed_gap_history",
                      [round(0.5 - 0.01 * k, 6) for k in range(n)])
        slm._meta_set(cn, "loss_history", [0.01] * n)  # plateaued
    trend = slm.stp_diagnostic_trend(window=10)
    assert trend["n_entropy_differential"] == n
    assert trend["entropy_differential_slope"] is not None
    assert trend["entropy_differential_slope"] > 0  # ΔS rising
    assert trend["delta_s_torus_corr"] is not None
    assert trend["delta_s_torus_corr"] <= -0.99
    assert trend["delta_s_signature"] is True


def test_stp_trend_delta_s_signature_absent_when_uncorrelated(isolated_slm_db):
    """Constant ΔS (no structural evolution) → corr None → signature False."""
    slm = isolated_slm_db
    n = 40
    with slm._conn() as cn:
        slm._meta_set(cn, "entropy_differential_history", [0.002] * n)
        slm._meta_set(cn, "stp_torus_gap_history",
                      [round(0.5 - 0.01 * k, 6) for k in range(n)])
        slm._meta_set(cn, "stp_embed_gap_history", [0.5] * n)
        slm._meta_set(cn, "loss_history", [0.01] * n)
    trend = slm.stp_diagnostic_trend(window=10)
    assert trend["delta_s_torus_corr"] is None
    assert trend["delta_s_signature"] is False
