"""Tests for the holographic corpus-ingest path (The Well mechanism).

Covers the pure, offline pieces: the 20-byte packed Weyl record (byte-exact
with the Julia mesh_compression_model peer), salience/entropy accounting, and
the cycle->Weyl compression + brain_kv persistence round-trip. No network.
"""
from __future__ import annotations

import base64
import os
from collections import Counter

import pytest

from src.quipu import corpus_ingest as ci


# ---------------------------------------------------------------------------
# Packed Weyl record — cross-language little-endian lock
# ---------------------------------------------------------------------------
def test_pack_weyl_is_20_bytes_le():
    psi = (0.0, 0.25, 0.5, 0.75, 1.0)          # all exact in Float32
    pk = ci.pack_weyl(psi)
    assert len(pk) == ci.MESH_BRAIN_KV_WEYL_PACKED_BYTES == 20
    # Exact little-endian byte layout (matches Julia pack_weyl's UInt8 loop):
    assert pk == bytes([0, 0, 0, 0,        # 0.0
                        0, 0, 128, 62,     # 0.25  -> 0x3E800000 LE
                        0, 0, 0, 63,       # 0.5   -> 0x3F000000 LE
                        0, 0, 64, 63,      # 0.75  -> 0x3F400000 LE
                        0, 0, 128, 63])    # 1.0   -> 0x3F800000 LE


def test_pack_weyl_golden_base64():
    # Golden vector printed by julia/mesh_compression_model.jl selftest().
    psi = (0.0, 0.25, 0.5, 0.75, 1.0)
    assert base64.b64encode(ci.pack_weyl(psi)).decode() == "AAAAAAAAgD4AAAA/AABAPwAAgD8="


def test_stored_bytes_is_base64_footprint_not_raw_pack():
    """brain_kv values are TEXT, so _persist_weyl base64-encodes the 20-byte
    packed record before storage. Real on-disk footprint is 28 bytes."""
    psi = (0.0, 0.25, 0.5, 0.75, 1.0)
    encoded_len = len(base64.b64encode(ci.pack_weyl(psi)).decode())
    assert encoded_len == ci.MESH_BRAIN_KV_WEYL_STORED_BYTES == 28
    assert ci.MESH_BRAIN_KV_WEYL_STORED_BYTES > ci.MESH_BRAIN_KV_WEYL_PACKED_BYTES


def test_pack_unpack_round_trip():
    psi = (0.1234, 0.9, 0.42, 0.0, 0.777)
    back = ci.unpack_weyl(ci.pack_weyl(psi))
    assert len(back) == 5
    for a, b in zip(psi, back):
        assert abs(a - b) < 1e-6                # <= Float32 eps


def test_unpack_weyl_rejects_short():
    with pytest.raises(ValueError):
        ci.unpack_weyl(b"\x00\x00")


# ---------------------------------------------------------------------------
# Salience + entropy accounting
# ---------------------------------------------------------------------------
def test_salience_bounds_and_trace():
    text = ("Supply chains optimize inventory. " * 6
            + "Holographic compression distills the corpus to a Weyl tensor. " * 4)
    trace, sal, terms = ci._salience(text)
    assert 0.0 <= sal <= 1.0
    assert isinstance(trace, str) and trace
    assert all(isinstance(t, str) for t in terms)


def test_salience_empty():
    trace, sal, terms = ci._salience("")
    assert trace == "" and sal == 0.0 and terms == []


def test_norm_entropy_uniform_vs_degenerate():
    assert ci._norm_entropy(Counter()) == 0.0
    assert ci._norm_entropy(Counter({"a": 5})) == 0.0          # single symbol -> 0
    uniform = ci._norm_entropy(Counter({"a": 1, "b": 1, "c": 1, "d": 1}))
    assert uniform > 0.99                                       # ~1.0 for uniform


# ---------------------------------------------------------------------------
# Cycle -> Weyl compression + brain_kv persistence (uses a temp DB)
# ---------------------------------------------------------------------------
@pytest.fixture
def _tmp_brain_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "weyl_test.sqlite"))
    yield


def test_compress_cycle_to_weyl_shape(_tmp_brain_db):
    psi5, meta = ci.compress_cycle_to_weyl(
        saliences=[0.3, 0.7, 0.5, 0.9],
        term_counts=Counter({"mesh": 4, "weyl": 3, "corpus": 2, "torus": 1}),
        source_counts=Counter({"arxiv": 2, "wikipedia": 2}),
        approx_source_bytes=4 * 4000,
    )
    assert len(psi5) == 5
    assert all(0.0 <= float(x) <= 1.0 for x in psi5)
    assert meta["corpus_volume"] == 4
    assert 0.0 <= meta["hawking_remnant_score"] <= 1.0


def test_persist_and_decompress_sigma(_tmp_brain_db):
    from src.quipu import brain_kv

    psi5 = (0.2, 0.6, 0.4, 0.5, 0.8)
    packed_b64 = ci._persist_weyl(psi5)
    assert packed_b64                                   # brain_kv present -> b64 returned

    stored = brain_kv.kv_get_json(ci._WEYL_TENSOR_KEY, None)
    assert stored and len(stored) == 5
    assert abs(stored[4] - 0.8) < 1e-6

    # Second write moves the first into the 'prev' slot; sigma is a real float.
    ci._persist_weyl((0.25, 0.65, 0.45, 0.55, 0.85))
    sigma = ci.decompress_sigma()
    assert isinstance(sigma, float) and 0.0 <= sigma <= 1.0
