"""Tests for the Human Behavioral Fidelity & Social Indistinguishability Assessor."""

import pytest
from quipu.human_fidelity_assessor import (
    AssessmentVerdict,
    GAME_BENCHMARK_PROFILES,
    PeriodicFidelityAssessor,
    SupportedGame,
)


def test_human_mimic_passes_assessment():
    """Verify that natural human-mimic telemetry passes fidelity assessment across all dimensions."""
    assessor = PeriodicFidelityAssessor(game=SupportedGame.WOW, window_size=50)
    samples = assessor.generate_synthetic_samples(count=40, is_human_mimic=True)
    for s in samples:
        assessor.record_sample(s)

    report = assessor.evaluate_efficacy()

    assert report.verdict == AssessmentVerdict.PASS
    assert report.overall_fidelity_score >= 0.85
    assert report.sample_count == 40
    # No rigid bot pattern anomalies should be flagged
    assert not any("rigid" in a for a in report.flagged_anomalies)


def test_synthetic_bot_fails_assessment():
    """Verify that rigid, low-variance bot telemetry is detected and flagged as suspicious."""
    assessor = PeriodicFidelityAssessor(game=SupportedGame.WOW, window_size=50)
    samples = assessor.generate_synthetic_samples(count=40, is_human_mimic=False)
    for s in samples:
        assessor.record_sample(s)

    report = assessor.evaluate_efficacy()

    assert report.verdict == AssessmentVerdict.SUSPICIOUS
    assert report.overall_fidelity_score < 0.70
    # Must flag reaction time rigidity or straight-line pathing
    assert any("rigid" in a or "straight-line" in a for a in report.flagged_anomalies)


def test_osrs_and_guild_wars_profiles():
    """Verify fidelity assessment runs for OSRS and Guild Wars with game-specific benchmarks."""
    for game in [SupportedGame.OSRS, SupportedGame.GUILD_WARS]:
        assessor = PeriodicFidelityAssessor(game=game, window_size=30)
        samples = assessor.generate_synthetic_samples(count=25, is_human_mimic=True)
        for s in samples:
            assessor.record_sample(s)

        report = assessor.evaluate_efficacy()
        assert report.game == game.value
        assert report.verdict == AssessmentVerdict.PASS
        assert report.overall_fidelity_score >= 0.80


def test_insufficient_samples_fallback():
    """Verify that assessment with less than 5 samples gracefully reports borderline with notice."""
    assessor = PeriodicFidelityAssessor(game=SupportedGame.WOW)
    samples = assessor.generate_synthetic_samples(count=3, is_human_mimic=True)
    for s in samples:
        assessor.record_sample(s)

    report = assessor.evaluate_efficacy()
    assert report.verdict == AssessmentVerdict.BORDERLINE
    assert any("Insufficient" in a for a in report.flagged_anomalies)
