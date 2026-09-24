"""Tests for the QUIPU Video & Stream Continuous Trainer Daemon."""

import json
import threading
import time
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from quipu.video_trainer_daemon import (
    ContinuousVideoTrainer,
    TrainerHTTPHandler,
    TrainerState,
    _STATE,
)


def test_trainer_state_serialization():
    """Verify TrainerState serializes cleanly to dict with correct keys."""
    state = TrainerState()
    d = state.to_dict()
    assert d["status"] == "idle"
    assert "current_game" in d
    assert "training_round" in d
    assert "uptime_seconds" in d
    assert "total_samples_processed" in d
    assert "last_fidelity_score" in d


def test_process_synthetic_video_file(tmp_path: Path):
    """Generate a short OpenCV test clip and verify frame extraction and demonstration generation."""
    video_file = tmp_path / "test_stream_clip.mp4"
    transcript_file = tmp_path / "test_stream_clip.txt"
    transcript_file.write_text("Heal me up! Pulling next pack!\n", encoding="utf-8")

    # Generate a dummy 30-frame video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 10.0, (640, 480))
    for _ in range(30):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw green health bar area in standard top-left unit frame ROI (y: 19-28, x: 51-115)
        frame[15:35, 50:120] = (0, 220, 0)
        out.write(frame)
    out.release()

    trainer = ContinuousVideoTrainer()
    steps = trainer.process_real_video_file(video_file)

    assert len(steps) > 0
    assert steps[0].step_id.startswith("test_stream_clip")
    assert steps[0].agent_state.hp_pct > 0.0


def test_trainer_http_server_endpoints(tmp_path: Path):
    """Verify HTTP endpoints /status, /metrics, and /queue on a test port."""
    test_port = 7289
    server = HTTPServer(("127.0.0.1", test_port), TrainerHTTPHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    time.sleep(0.3)
    try:
        # 1. GET /status
        with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/status", timeout=3) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "status" in data
            assert "current_game" in data

        # 2. GET /metrics
        with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/metrics", timeout=3) as resp:
            assert resp.status == 200
            text = resp.read().decode("utf-8")
            assert "quipu_training_round" in text
            assert "quipu_fidelity_score" in text

        # 3. POST /queue
        req = urllib.request.Request(
            f"http://127.0.0.1:{test_port}/queue",
            data=json.dumps({"filename": "twitch_osrs_recording_01.mp4"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "queued"
            assert data["filename"] == "twitch_osrs_recording_01.mp4"

    finally:
        server.shutdown()
        server.server_close()
