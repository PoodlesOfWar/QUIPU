"""Tests for the QUIPU Stream Corpus Scanner, GARD Shard compression, and Vector Graph Integration."""

from pathlib import Path
import pytest

from quipu.stream_corpus_scanner import (
    CATALOG_SPECS,
    StreamCorpusScanner,
    StreamSessionSpec,
)
from quipu.gard_shard_model import GARD_PROTOCOL, read_envelope_json


def test_catalog_specs_multi_game_coverage():
    """Verify that catalog specs cover WoW, OSRS, and Guild Wars across Twitch and YouTube."""
    games = {spec.game for spec in CATALOG_SPECS}
    platforms = {spec.platform for spec in CATALOG_SPECS}
    assert games == {"wow", "osrs", "gw"}
    assert platforms == {"twitch", "youtube"}
    assert len(CATALOG_SPECS) >= 18


def test_video_and_transcript_generation(tmp_path: Path):
    """Verify MP4 video rendering with HUD vitals and synchronized text transcripts."""
    scanner = StreamCorpusScanner(recordings_dir=tmp_path, gard_dir=tmp_path)
    spec = CATALOG_SPECS[0]  # Scarlet Monastery Cathedral

    v_path, t_path = scanner.generate_synthetic_video_stream(spec)
    assert v_path.exists()
    assert t_path.exists()
    assert v_path.stat().st_size > 5000
    assert len(t_path.read_text(encoding="utf-8").splitlines()) == len(spec.speech_transcript)


def test_gard_shard_compression_and_decompression_roundtrip(tmp_path: Path):
    """Verify GARD Shard v2 (AES-256-GCM + zlib) encryption, sealing, and lossless decompression."""
    scanner = StreamCorpusScanner(recordings_dir=tmp_path, gard_dir=tmp_path)
    spec = CATALOG_SPECS[6]  # Theatre of Blood Verzik

    # 1. Compress to GARD Shard container
    shard_path, envelope = scanner.compress_session_to_gard_shard(spec)
    assert shard_path.exists()
    assert envelope["protocol"] == GARD_PROTOCOL
    assert envelope["compression"]["algorithm"] == "zlib"
    assert envelope["encryption"]["algorithm"] == "AES-256-GCM"

    # 2. Decompress and verify
    decrypted = scanner.decompress_gard_shard(shard_path)
    assert decrypted["session_id"] == spec.session_id
    assert decrypted["game"] == "osrs"
    assert len(decrypted["demonstrations"]) == len(spec.speech_transcript)
    assert decrypted["demonstrations"][0]["speech_intent"] is not None


def test_vector_graph_projection(tmp_path: Path):
    """Verify that session concepts feed into the toroidal Quipu vector graph."""
    scanner = StreamCorpusScanner(recordings_dir=tmp_path, gard_dir=tmp_path)
    spec = CATALOG_SPECS[12]  # Fissure of Woe Menzies

    fed_rows = scanner.project_into_vector_graph(spec)
    assert fed_rows > 0


def test_scanner_batch_execution(tmp_path: Path):
    """Run scanner across a limited batch (2 sessions) and verify full end-to-end output."""
    scanner = StreamCorpusScanner(recordings_dir=tmp_path, gard_dir=tmp_path)
    results = scanner.run_full_scan(limit=2)

    assert results["status"] == "success"
    assert results["sessions_scanned"] == 2
    assert len(results["generated_videos"]) == 2
    assert len(results["generated_shards"]) == 2
    assert "vector_graph_summary" in results
    assert results["vector_graph_summary"]["vocab_size"] is not None
