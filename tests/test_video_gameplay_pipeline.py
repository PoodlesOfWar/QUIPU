"""Tests for Video-to-Quipu Pre-Training & Inverse Graph Tension Learning (IGTL) Pipeline."""

import math
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from quipu.video_gameplay_pipeline import (
    DemonstrationStep,
    HUDVisionExtractor,
    HUDVitals,
    InverseGraphTensionOptimizer,
    MinimapBlip,
    QuipuLearnedParameters,
    SpeechTranscriptSegment,
    StreamerAudioParser,
    StreamerSpeechIntent,
    VideoGameplayPipeline,
)
from quipu.quipu_game_mesh import TacticalActionType


def test_hud_vision_extractor_synthetic_frame():
    """Verify computer vision extraction of HP and MP bar percentages from a synthetic frame."""
    frame = HUDVisionExtractor.synthesize_test_frame(
        width=640,
        height=360,
        hp_pct=0.80,
        mp_pct=0.50,
        in_combat=True,
        hostile_blip_dist=25.0,
    )

    vitals = HUDVisionExtractor.extract_vitals_from_frame(frame)

    assert abs(vitals.hp_pct - 0.80) < 0.08
    assert abs(vitals.mp_pct - 0.50) < 0.08
    assert vitals.is_in_combat is True


def test_hud_minimap_blip_detection():
    """Verify detection of hostile red blip on the minimap radar circle."""
    frame = HUDVisionExtractor.synthesize_test_frame(
        width=640,
        height=360,
        hp_pct=1.0,
        mp_pct=1.0,
        in_combat=False,
        hostile_blip_dist=30.0,
    )

    blips = HUDVisionExtractor.extract_minimap_blips(frame, max_radar_yards=60.0)

    assert len(blips) >= 1
    hostile_blips = [b for b in blips if b.blip_type == "hostile"]
    assert len(hostile_blips) >= 1
    # Check that detected distance is approximately 30 yards
    assert abs(hostile_blips[0].distance_yards - 30.0) < 10.0


def test_streamer_speech_intent_parser():
    """Verify classification of streamer spoken audio cues into tactical intents."""
    cues = [
        ("I'm going to LoS behind the pillar so his fireballs don't hit me", StreamerSpeechIntent.LOS_TACTIC),
        ("Careful, he's a runner, don't let him pull adds! Stun him!", StreamerSpeechIntent.RUNNER_ALERT),
        ("Out of mana, sitting down to drink before this pull", StreamerSpeechIntent.REST_MANDATE),
        ("Too many adds, wipe it, run away!", StreamerSpeechIntent.EMERGENCY_FLEE),
        ("Pulling next mob, let's go", StreamerSpeechIntent.ENGAGE_PULL),
        ("Did you guys see that drop yesterday on stream?", StreamerSpeechIntent.NEUTRAL_CHAT),
    ]

    for text, expected_intent in cues:
        detected = StreamerAudioParser.classify_speech(text)
        assert detected == expected_intent, f"Failed on text: '{text}'"


def test_inverse_graph_tension_optimizer_training():
    """Verify that the IGTL optimizer minimizes NLL loss and achieves high top-1 accuracy on demonstrations."""
    pipeline = VideoGameplayPipeline()
    demonstrations = pipeline.build_synthetic_demonstrations(count=25)

    # Initialize with perturbed/suboptimal weights so initial loss > 0
    suboptimal_params = QuipuLearnedParameters(
        w_runner_urgency=0.1,
        w_escape_urgency=0.1,
        w_rest_reward=5.0,
        w_los_pull_reward=5.0,
    )
    optimizer = InverseGraphTensionOptimizer(demonstrations=demonstrations, initial_params=suboptimal_params)
    initial_loss = optimizer.compute_loss(optimizer.params.to_vector())

    # Train for 25 epochs
    optimized_params, stats = optimizer.train(epochs=25, learning_rate=0.08)
    final_loss = stats["final_loss"]

    # Loss should decrease substantially
    assert final_loss < initial_loss
    # Top-1 accuracy on demonstrations should be high (>= 80%)
    assert stats["final_accuracy"] >= 0.80

    # Weights must be positive and reasonable
    assert optimized_params.w_figure_eight > 0.0
    assert optimized_params.w_runner_urgency > 0.0
    assert optimized_params.w_rest_reward > 0.0


def test_end_to_end_pipeline_artifact_generation():
    """Verify end-to-end pipeline execution from demonstration generation to disk artifact saving."""
    temp_dir = tempfile.mkdtemp(prefix="quipu_test_pipeline_")
    try:
        pipeline = VideoGameplayPipeline(output_dir=temp_dir)
        params, stats = pipeline.run_training_pipeline(epochs=15, learning_rate=0.05)

        param_file = Path(temp_dir) / "quipu_learned_parameters.json"
        assert param_file.exists()
        assert param_file.stat().st_size > 0

        # Verify JSON is valid and matches params
        data = param_file.read_text(encoding="utf-8")
        assert "w_figure_eight" in data
        assert "w_runner_urgency" in data
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
