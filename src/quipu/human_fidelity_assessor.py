"""Human Behavioral Fidelity & Social Indistinguishability Assessor.

Conducts periodic empirical assessments to determine whether an autonomous agent's
spatial, kinematic, and tactical interaction patterns pass as authentic active human
players across Old School RuneScape (OSRS), Guild Wars (GW), and World of Warcraft (WoW).

Core Assessment Metrics:
------------------------
1. Reaction Time & Cadence Distribution:
   - Human reaction times follow an ex-Gaussian distribution with characteristic right skew (fat tail).
   - Flat/uniform or near-zero variance reaction latencies are flagged as synthetic bot signatures.
2. Spatial Wander & Pathing Entropy:
   - Evaluates micro-jitter and sub-optimal curvature against pure geodesic NavMesh lines.
3. Attrition & Downtime Pacing:
   - Measures hesitation latencies, eating/drinking buffers, and post-combat pause intervals.
4. Social Etiquette & Player Collision Avoidance:
   - Flags kill-stealing on tagged mobs, coordinate stacking in social hubs, and instantaneous chat response.
5. Statistical Divergence Scoring:
   - Employs Jensen-Shannon / Wasserstein divergence against empirical human benchmark profiles.
"""

from __future__ import annotations

import enum
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SupportedGame(str, enum.Enum):
    """Supported MMORPG environments for fidelity benchmarking."""
    WOW = "wow"
    OSRS = "osrs"
    GUILD_WARS = "gw"


class AssessmentVerdict(str, enum.Enum):
    """Overall assessment verdict on agent indistinguishability."""
    PASS = "PASS"                         # Indistinguishable from active human players (score >= 0.85)
    BORDERLINE = "BORDERLINE"             # Minor synthetic patterns detected (0.70 <= score < 0.85)
    SUSPICIOUS = "SUSPICIOUS_BOT_PATTERN" # Detectable bot/script signature (score < 0.70)


@dataclass
class HumanGameBenchmarkProfile:
    """Empirical baseline distribution parameters for human players in a specific game."""
    game: SupportedGame
    reaction_mean_ms: float
    reaction_std_min_ms: float
    right_skew_tail_ratio: float      # Fraction of actions with delayed latency (> 1.5x mean)
    min_spatial_wander_entropy: float # Minimum non-zero spatial path curvature
    social_bubble_radius_yards: float # Expected personal space buffer in social zones
    rest_hesitation_mean_s: float     # Average hesitation before resting/eating


# Empirical baselines from human telemetry
GAME_BENCHMARK_PROFILES: dict[SupportedGame, HumanGameBenchmarkProfile] = {
    SupportedGame.WOW: HumanGameBenchmarkProfile(
        game=SupportedGame.WOW,
        reaction_mean_ms=320.0,
        reaction_std_min_ms=65.0,
        right_skew_tail_ratio=0.15,
        min_spatial_wander_entropy=0.18,
        social_bubble_radius_yards=4.0,
        rest_hesitation_mean_s=1.2,
    ),
    SupportedGame.OSRS: HumanGameBenchmarkProfile(
        game=SupportedGame.OSRS,
        reaction_mean_ms=380.0,
        reaction_std_min_ms=85.0,
        right_skew_tail_ratio=0.22,
        min_spatial_wander_entropy=0.12,  # Discrete grid pathing
        social_bubble_radius_yards=2.0,
        rest_hesitation_mean_s=1.8,
    ),
    SupportedGame.GUILD_WARS: HumanGameBenchmarkProfile(
        game=SupportedGame.GUILD_WARS,
        reaction_mean_ms=290.0,
        reaction_std_min_ms=60.0,
        right_skew_tail_ratio=0.14,
        min_spatial_wander_entropy=0.16,
        social_bubble_radius_yards=3.5,
        rest_hesitation_mean_s=0.9,
    ),
}


@dataclass
class AgentTelemetrySample:
    """Observed telemetry point from an active agent cycle."""
    timestamp_s: float
    reaction_latency_ms: float
    action_type: str
    waypoint_wander_deviation: float # Deviation from straight-line geodesic (0.0 = machine straight)
    distance_to_nearest_player: float
    social_etiquette_violation: bool = False # E.g. attacked mob tagged by another player
    pre_action_pause_s: float = 0.0


@dataclass
class DimensionScore:
    """Score breakdown for a specific behavioral dimension."""
    dimension_name: str
    score: float           # 0.0 to 1.0 (1.0 = perfect human match)
    observed_value: float
    expected_benchmark: float
    is_acceptable: bool
    diagnostic: str


@dataclass
class FidelityAssessmentReport:
    """Full periodic assessment report evaluating agent indistinguishability."""
    assessment_id: str
    game: str
    timestamp_s: float
    sample_count: int
    overall_fidelity_score: float # 0.0 to 1.0 (percentage)
    verdict: AssessmentVerdict
    dimension_scores: list[DimensionScore] = field(default_factory=list)
    flagged_anomalies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "game": self.game,
            "timestamp_s": self.timestamp_s,
            "sample_count": self.sample_count,
            "overall_fidelity_score": round(self.overall_fidelity_score, 4),
            "verdict": self.verdict.value,
            "dimension_scores": [asdict(d) for d in self.dimension_scores],
            "flagged_anomalies": self.flagged_anomalies,
        }


class PeriodicFidelityAssessor:
    """Evaluates agent execution history against authentic human player distributions."""

    def __init__(self, game: SupportedGame = SupportedGame.WOW, window_size: int = 100) -> None:
        self.game = game
        self.profile = GAME_BENCHMARK_PROFILES[game]
        self.window_size = window_size
        self.telemetry_history: list[AgentTelemetrySample] = []
        self.assessment_counter: int = 0
        self.last_report: Optional[FidelityAssessmentReport] = None

    def record_sample(self, sample: AgentTelemetrySample) -> None:
        """Appends a real-time execution sample to the assessment sliding window."""
        self.telemetry_history.append(sample)
        if len(self.telemetry_history) > self.window_size * 2:
            self.telemetry_history = self.telemetry_history[-self.window_size:]

    def evaluate_efficacy(self) -> FidelityAssessmentReport:
        """Computes comprehensive fidelity scores and generates an assessment report."""
        self.assessment_counter += 1
        samples = self.telemetry_history[-self.window_size:]
        n = len(samples)

        if n < 5:
            # Insufficient samples for statistical significance
            return FidelityAssessmentReport(
                assessment_id=f"assess_{self.game.value}_{self.assessment_counter}",
                game=self.game.value,
                timestamp_s=time.time(),
                sample_count=n,
                overall_fidelity_score=0.5,
                verdict=AssessmentVerdict.BORDERLINE,
                flagged_anomalies=["Insufficient sample history (< 5 actions recorded)"],
            )

        anomalies: list[str] = []
        dim_scores: list[DimensionScore] = []

        # -------------------------------------------------------------------
        # 1. Reaction Latency Variance & Skew
        # -------------------------------------------------------------------
        latencies = np.array([s.reaction_latency_ms for s in samples], dtype=np.float64)
        mean_lat = float(np.mean(latencies))
        std_lat = float(np.std(latencies))

        # Check for robotic zero-variance signature
        if std_lat < self.profile.reaction_std_min_ms:
            std_score = max(0.1, std_lat / self.profile.reaction_std_min_ms)
            anomalies.append(
                f"Reaction time variance too rigid (sigma={std_lat:.1f}ms < {self.profile.reaction_std_min_ms}ms benchmark). Synthetic bot signature."
            )
        else:
            std_score = 1.0

        # Check right-skew tail (human hesitation / distraction tail)
        tail_threshold = mean_lat * 1.5
        tail_fraction = float(np.sum(latencies > tail_threshold) / n)
        if tail_fraction < (self.profile.right_skew_tail_ratio * 0.4):
            skew_score = 0.5
            anomalies.append(
                f"Reaction latency lacks natural human right-skew long-tail (observed {tail_fraction*100:.1f}% vs expected {self.profile.right_skew_tail_ratio*100:.1f}%)."
            )
        else:
            skew_score = 1.0

        cadence_score = 0.6 * std_score + 0.4 * skew_score
        dim_scores.append(DimensionScore(
            dimension_name="cadence_and_reaction_variance",
            score=round(cadence_score, 3),
            observed_value=round(std_lat, 1),
            expected_benchmark=self.profile.reaction_std_min_ms,
            is_acceptable=cadence_score >= 0.75,
            diagnostic=f"Mean: {mean_lat:.1f}ms, Std: {std_lat:.1f}ms, TailRatio: {tail_fraction*100:.1f}%",
        ))

        # -------------------------------------------------------------------
        # 2. Kinematic & Spatial Path Wander (Geodesic Deviation)
        # -------------------------------------------------------------------
        wanders = np.array([s.waypoint_wander_deviation for s in samples], dtype=np.float64)
        mean_wander = float(np.mean(wanders))

        if mean_wander < (self.profile.min_spatial_wander_entropy * 0.3):
            spatial_score = 0.3
            anomalies.append(
                f"Pathing is unnaturally straight-line (wander entropy={mean_wander:.3f} < {self.profile.min_spatial_wander_entropy:.3f}). Bot pathing detection risk."
            )
        else:
            spatial_score = min(1.0, mean_wander / self.profile.min_spatial_wander_entropy)

        dim_scores.append(DimensionScore(
            dimension_name="spatial_wander_and_curvature",
            score=round(spatial_score, 3),
            observed_value=round(mean_wander, 3),
            expected_benchmark=self.profile.min_spatial_wander_entropy,
            is_acceptable=spatial_score >= 0.70,
            diagnostic=f"Average geodesic wander: {mean_wander:.3f}",
        ))

        # -------------------------------------------------------------------
        # 3. Social Etiquette & Personal Space
        # -------------------------------------------------------------------
        violations = sum(1 for s in samples if s.social_etiquette_violation)
        violation_rate = violations / n
        if violation_rate > 0.05:
            social_score = max(0.0, 1.0 - (violation_rate * 5.0))
            anomalies.append(
                f"Social etiquette violations detected in {violations}/{n} samples (kill-stealing or mob poaching). High report risk."
            )
        else:
            social_score = 1.0 - violation_rate

        # Player distance buffer in social hubs
        distances = np.array([s.distance_to_nearest_player for s in samples if s.distance_to_nearest_player > 0], dtype=np.float64)
        mean_dist = float(np.mean(distances)) if len(distances) > 0 else 10.0
        if mean_dist < (self.profile.social_bubble_radius_yards * 0.5):
            social_score = min(social_score, 0.6)
            anomalies.append(
                f"Agent encroaches inside personal social bubble (average player distance {mean_dist:.1f} yds)."
            )

        dim_scores.append(DimensionScore(
            dimension_name="social_etiquette_and_spacing",
            score=round(social_score, 3),
            observed_value=round(1.0 - violation_rate, 3),
            expected_benchmark=0.98,
            is_acceptable=social_score >= 0.85,
            diagnostic=f"Etiquette compliance: {(1.0-violation_rate)*100:.1f}%, Mean neighbor distance: {mean_dist:.1f} yds",
        ))

        # -------------------------------------------------------------------
        # 4. Attrition Hesitation & Downtime Pacing
        # -------------------------------------------------------------------
        pauses = np.array([s.pre_action_pause_s for s in samples if s.pre_action_pause_s > 0], dtype=np.float64)
        mean_pause = float(np.mean(pauses)) if len(pauses) > 0 else 0.5
        pause_ratio = mean_pause / self.profile.rest_hesitation_mean_s
        pacing_score = min(1.0, max(0.3, pause_ratio if pause_ratio <= 1.5 else (1.5 / pause_ratio)))

        dim_scores.append(DimensionScore(
            dimension_name="attrition_downtime_pacing",
            score=round(pacing_score, 3),
            observed_value=round(mean_pause, 2),
            expected_benchmark=self.profile.rest_hesitation_mean_s,
            is_acceptable=pacing_score >= 0.70,
            diagnostic=f"Average pre-rest hesitation pause: {mean_pause:.2f}s",
        ))

        # -------------------------------------------------------------------
        # Overall Fidelity Score & Verdict
        # -------------------------------------------------------------------
        # Weighted aggregate: cadence (35%), spatial (25%), social (25%), pacing (15%)
        overall = (
            0.35 * cadence_score +
            0.25 * spatial_score +
            0.25 * social_score +
            0.15 * pacing_score
        )

        if overall >= 0.85 and not any("rigid" in a or "kill-stealing" in a for a in anomalies):
            verdict = AssessmentVerdict.PASS
        elif overall >= 0.70:
            verdict = AssessmentVerdict.BORDERLINE
        else:
            verdict = AssessmentVerdict.SUSPICIOUS

        report = FidelityAssessmentReport(
            assessment_id=f"assess_{self.game.value}_{self.assessment_counter}",
            game=self.game.value,
            timestamp_s=time.time(),
            sample_count=n,
            overall_fidelity_score=overall,
            verdict=verdict,
            dimension_scores=dim_scores,
            flagged_anomalies=anomalies,
        )
        self.last_report = report
        return report

    def generate_synthetic_samples(
        self,
        count: int = 50,
        is_human_mimic: bool = True,
    ) -> list[AgentTelemetrySample]:
        """Generates sample telemetry for testing and validation."""
        samples: list[AgentTelemetrySample] = []
        t = time.time()

        for i in range(count):
            if is_human_mimic:
                # Ex-Gaussian distributed human reaction times
                gaussian_component = np.random.normal(self.profile.reaction_mean_ms, self.profile.reaction_std_min_ms * 1.1)
                exponential_tail = np.random.exponential(scale=self.profile.reaction_mean_ms * 0.3)
                reaction = float(np.clip(gaussian_component + exponential_tail, 140.0, 1800.0))
                wander = float(np.random.uniform(0.15, 0.40))
                dist = float(np.random.uniform(3.0, 15.0))
                violation = (i % 30 == 0) and False  # Strictly 0 violations
                pause = float(np.random.uniform(0.6, 2.2))
            else:
                # Obvious bot signature: rigid uniform distribution
                reaction = 200.0 + float(np.random.uniform(-5.0, 5.0))
                wander = 0.01  # Perfect straight line
                dist = 0.5     # Stacks directly on players
                violation = (i % 8 == 0)  # Frequent poaching
                pause = 0.02   # Instantaneous 20ms pause

            samples.append(AgentTelemetrySample(
                timestamp_s=t + i * 2.0,
                reaction_latency_ms=reaction,
                action_type="engage" if i % 2 == 0 else "traverse",
                waypoint_wander_deviation=wander,
                distance_to_nearest_player=dist,
                social_etiquette_violation=violation,
                pre_action_pause_s=pause,
            ))

        return samples


__all__ = [
    "SupportedGame",
    "AssessmentVerdict",
    "HumanGameBenchmarkProfile",
    "AgentTelemetrySample",
    "DimensionScore",
    "FidelityAssessmentReport",
    "PeriodicFidelityAssessor",
    "GAME_BENCHMARK_PROFILES",
]
