"""Video-to-Quipu Pre-Training & Inverse Graph Tension Learning (IGTL) Pipeline.

Transforms raw video footage (Twitch streams, YouTube gameplay recordings, HUD captures)
and streamer audio transcripts into trained QUIPU Graph Model weights.

Pipeline Stages:
----------------
1. Video & Audio Ingest / Feature Extraction:
   - Computer Vision extracts HUD vitals (HP%, MP%, In-Combat icon).
   - Minimap radar blip detection extracts nearby entity polar coordinates.
   - Speech-to-Text (Whisper / transcripts) parses streamer tactical rationale.
2. Demonstration Structuring:
   - Assembles temporal sequences of (WorldState, StreamerSpeechIntent, HumanAction).
3. Inverse Graph Tension Optimization (IGTL):
   - Structural Maximum-Margin / Softmax Cross-Entropy loss over Quipu candidate utilities.
   - Fits knot tension weights, distance damping factors, and subsidiary urgency multipliers
     to maximize the probability of human expert actions.
4. Deployment:
   - Injects learned human behavioral weights into the running QuipuGameMeshEngine.
"""

from __future__ import annotations

import enum
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .quipu_game_mesh import (
    AgentCoreState,
    KnotType,
    LoSOccluder,
    MacroIntent,
    NavMeshWaypoint,
    PendantDomain,
    QuipuGameMeshEngine,
    QuipuKnot,
    SubsidiaryCord,
    SubsidiaryType,
    TacticalAction,
    TacticalActionType,
    Vector3,
    WorldAffordance,
    WorldEntity,
)

logger = logging.getLogger(__name__)

try:
    from . import brain_kv as _brain_kv
except ImportError:
    _brain_kv = None  # type: ignore


# ---------------------------------------------------------------------------
# Stage 1: Computer Vision & HUD Feature Extraction
# ---------------------------------------------------------------------------

@dataclass
class HUDVitals:
    """Agent vitals extracted from video frames."""
    hp_pct: float
    mp_pct: float
    is_in_combat: bool
    timestamp_s: float = 0.0


@dataclass
class MinimapBlip:
    """Entity detected on the circular minimap radar."""
    distance_yards: float
    bearing_rad: float
    blip_type: str  # 'hostile', 'neutral', 'friendly'


@dataclass
class FramePerceptionObservation:
    """Extracted perceptual state from a single video keyframe."""
    timestamp_s: float
    vitals: HUDVitals
    minimap_blips: list[MinimapBlip] = field(default_factory=list)
    target_name: Optional[str] = None
    target_level: Optional[int] = None
    target_is_casting: bool = False
    target_health_pct: Optional[float] = None
    has_los_corner_nearby: bool = False
    demonstrated_human_action: Optional[TacticalActionType] = None


class HUDVisionExtractor:
    """Computer vision processor for gameplay video frames."""

    @staticmethod
    def extract_vitals_from_frame(
        frame: np.ndarray,
        hp_roi: Optional[Tuple[int, int, int, int]] = None,
        mp_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> HUDVitals:
        """Extracts HP% and MP% via color fill ratio in specified Regions of Interest (ROI).
        
        Args:
            frame: BGR numpy image frame from video.
            hp_roi: (ymin, ymax, xmin, xmax) for Health bar.
            mp_roi: (ymin, ymax, xmin, xmax) for Mana bar.
        """
        h, w = frame.shape[:2]
        # Default fallback ROIs for standard 1080p HUD top-left unit frames
        if hp_roi is None:
            hp_roi = (int(0.04 * h), int(0.06 * h), int(0.08 * w), int(0.18 * w))
        if mp_roi is None:
            mp_roi = (int(0.06 * h), int(0.075 * h), int(0.08 * w), int(0.18 * w))

        hp_crop = frame[hp_roi[0]:hp_roi[1], hp_roi[2]:hp_roi[3]]
        mp_crop = frame[mp_roi[0]:mp_roi[1], mp_roi[2]:mp_roi[3]]

        # Health bar fill ratio (green/red channel dominance over dark background)
        hp_pct = HUDVisionExtractor._calculate_bar_fill(hp_crop, channel_idx=1)  # Green or Red
        mp_pct = HUDVisionExtractor._calculate_bar_fill(mp_crop, channel_idx=0)  # Blue (BGR: 0=Blue)

        # Detect in-combat crossed swords indicator (red glow in icon area)
        combat_roi = (int(0.02 * h), int(0.05 * h), int(0.05 * w), int(0.08 * w))
        combat_crop = frame[combat_roi[0]:combat_roi[1], combat_roi[2]:combat_roi[3]]
        is_combat = HUDVisionExtractor._detect_red_combat_glow(combat_crop)

        return HUDVitals(
            hp_pct=float(np.clip(hp_pct, 0.0, 1.0)),
            mp_pct=float(np.clip(mp_pct, 0.0, 1.0)),
            is_in_combat=is_combat,
        )

    @staticmethod
    def extract_minimap_blips(
        frame: np.ndarray,
        minimap_roi: Optional[Tuple[int, int, int, int]] = None,
        max_radar_yards: float = 60.0,
    ) -> list[MinimapBlip]:
        """Detects colored dots (hostile red, neutral yellow, friendly green) on minimap radar."""
        h, w = frame.shape[:2]
        # Top-right corner default for minimap in WoW/OSRS/GW
        if minimap_roi is None:
            minimap_roi = (int(0.02 * h), int(0.20 * h), int(0.85 * w), int(0.98 * w))

        crop = frame[minimap_roi[0]:minimap_roi[1], minimap_roi[2]:minimap_roi[3]]
        if crop.size == 0:
            return []

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        crop_h, crop_w = crop.shape[:2]
        center_x, center_y = crop_w / 2.0, crop_h / 2.0
        radar_radius = min(center_x, center_y)

        blips: list[MinimapBlip] = []

        # Color masks in HSV
        # Red hostile mask (wraps around 0/180)
        red_mask1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 120, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Yellow neutral mask
        yellow_mask = cv2.inRange(hsv, np.array([20, 120, 100]), np.array([35, 255, 255]))

        # Green friendly mask
        green_mask = cv2.inRange(hsv, np.array([40, 120, 100]), np.array([85, 255, 255]))

        for mask, b_type in [(red_mask, "hostile"), (yellow_mask, "neutral"), (green_mask, "friendly")]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 2 <= area <= 60:  # Size of radar blip dot
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        bx = M["m10"] / M["m00"]
                        by = M["m01"] / M["m00"]
                        dx = bx - center_x
                        dy = by - center_y
                        pixel_dist = math.sqrt(dx ** 2 + dy ** 2)
                        if pixel_dist <= radar_radius:
                            dist_yards = (pixel_dist / radar_radius) * max_radar_yards
                            bearing = math.atan2(dx, -dy)  # Clockwise from north
                            blips.append(MinimapBlip(distance_yards=dist_yards, bearing_rad=bearing, blip_type=b_type))

        return blips

    @staticmethod
    def _calculate_bar_fill(crop: np.ndarray, channel_idx: int) -> float:
        if crop.size == 0:
            return 1.0
        # Average intensity across columns along horizontal width
        col_means = np.mean(crop[:, :, channel_idx], axis=0)
        # Threshold between lit bar and empty dark background
        active_cols = np.sum(col_means > 60)
        total_cols = len(col_means)
        return float(active_cols / total_cols) if total_cols > 0 else 1.0

    @staticmethod
    def _detect_red_combat_glow(crop: np.ndarray) -> bool:
        if crop.size == 0:
            return False
        # High red-to-blue ratio signifies in-combat border glow
        r_mean = np.mean(crop[:, :, 2])
        b_mean = np.mean(crop[:, :, 0])
        return bool(r_mean > 120 and r_mean > 1.8 * b_mean)

    @staticmethod
    def synthesize_test_frame(
        width: int = 640,
        height: int = 360,
        hp_pct: float = 0.85,
        mp_pct: float = 0.60,
        in_combat: bool = False,
        hostile_blip_dist: Optional[float] = 20.0,
    ) -> np.ndarray:
        """Generates a synthetic gameplay frame for deterministic pipeline testing."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Gray background (game world)
        frame[:] = (40, 40, 40)

        # Health bar ROI (green)
        hp_ymin, hp_ymax = int(0.04 * height), int(0.06 * height)
        hp_xmin, hp_xmax = int(0.08 * width), int(0.18 * width)
        bar_w = hp_xmax - hp_xmin
        fill_w = int(bar_w * hp_pct)
        frame[hp_ymin:hp_ymax, hp_xmin:hp_xmin + fill_w] = (0, 200, 30)  # Green
        frame[hp_ymin:hp_ymax, hp_xmin + fill_w:hp_xmax] = (20, 20, 20)  # Empty

        # Mana bar ROI (blue)
        mp_ymin, mp_ymax = int(0.06 * height), int(0.075 * height)
        mp_xmin, mp_xmax = int(0.08 * width), int(0.18 * width)
        m_bar_w = mp_xmax - mp_xmin
        m_fill_w = int(m_bar_w * mp_pct)
        frame[mp_ymin:mp_ymax, mp_xmin:mp_xmin + m_fill_w] = (220, 100, 20)  # Blue
        frame[mp_ymin:mp_ymax, mp_xmin + m_fill_w:mp_xmax] = (20, 20, 20)

        # Minimap (circle in top-right)
        mm_ymin, mm_ymax = int(0.02 * height), int(0.20 * height)
        mm_xmin, mm_xmax = int(0.85 * width), int(0.98 * width)
        cx, cy = (mm_xmin + mm_xmax) // 2, (mm_ymin + mm_ymax) // 2
        r = (mm_ymax - mm_ymin) // 2
        cv2.circle(frame, (cx, cy), r, (80, 80, 80), -1)

        # Hostile red blip if requested
        if hostile_blip_dist is not None:
            norm_dist = min(1.0, hostile_blip_dist / 60.0)
            bx = int(cx + norm_dist * (r - 4))
            by = cy
            cv2.circle(frame, (bx, by), 3, (0, 0, 255), -1)  # Red BGR

        # Combat glow
        if in_combat:
            c_ymin, c_ymax = int(0.02 * height), int(0.05 * height)
            c_xmin, c_xmax = int(0.05 * width), int(0.08 * width)
            frame[c_ymin:c_ymax, c_xmin:c_xmax] = (30, 30, 240)  # Red icon

        return frame


# ---------------------------------------------------------------------------
# Stage 2: Audio Transcription & Streamer Intent Ingestion
# ---------------------------------------------------------------------------

class StreamerSpeechIntent(str, enum.Enum):
    """Categorized tactical intent extracted from streamer audio commentary."""
    LOS_TACTIC = "los_tactic"        # "LoS behind pillar", "break his cast"
    RUNNER_ALERT = "runner_alert"    # "He's running!", "don't let him pull adds", "snare him"
    REST_MANDATE = "rest_mandate"    # "Need to drink", "eating up", "mana break"
    EMERGENCY_FLEE = "emergency_flee"# "Run away", "wipe it", "too many adds"
    ENGAGE_PULL = "engage_pull"      # "Pulling next pack", "let's go"
    NEUTRAL_CHAT = "neutral_chat"    # General commentary


@dataclass
class SpeechTranscriptSegment:
    """Timestamped transcription segment (e.g. from Whisper)."""
    start_time_s: float
    end_time_s: float
    text: str
    detected_intent: StreamerSpeechIntent = StreamerSpeechIntent.NEUTRAL_CHAT


class StreamerAudioParser:
    """Extracts tactical macro-intent from speech audio transcripts."""

    INTENT_PATTERNS = [
        (StreamerSpeechIntent.EMERGENCY_FLEE, [r"run away", r"\bflee\b", r"\breset\b", r"\bwipe\b", r"get out", r"back up"]),
        (StreamerSpeechIntent.LOS_TACTIC, [r"\blos\b", r"line of sight", r"behind the (pillar|corner|wall)", r"break (the|his)?\s?cast"]),
        (StreamerSpeechIntent.RUNNER_ALERT, [r"\brunner\b", r"\brunning\b", r"pull adds", r"\bsnare\b", r"slow him", r"stun him"]),
        (StreamerSpeechIntent.REST_MANDATE, [r"\bdrink\b", r"\bmana\b", r"\beating\b", r"sit down", r"out of mana", r"rest up"]),
        (StreamerSpeechIntent.ENGAGE_PULL, [r"pulling", r"pull this", r"kill this", r"engage", r"focus"]),
    ]

    @classmethod
    def classify_speech(cls, text: str) -> StreamerSpeechIntent:
        text_lower = text.lower()
        for intent, patterns in cls.INTENT_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        return StreamerSpeechIntent.NEUTRAL_CHAT


# ---------------------------------------------------------------------------
# Stage 3: Demonstration Structuring & Dataset
# ---------------------------------------------------------------------------

@dataclass
class DemonstrationStep:
    """A single training observation coupled with the ground-truth human tactical action."""
    step_id: str
    timestamp_s: float
    agent_state: AgentCoreState
    entities: list[WorldEntity]
    nav_waypoints: list[NavMeshWaypoint]
    affordances: list[WorldAffordance]
    occluders: list[LoSOccluder]
    speech_intent: StreamerSpeechIntent
    human_action: TacticalActionType


# ---------------------------------------------------------------------------
# Stage 4: Trainable Quipu Parameters & Inverse Graph Tension Optimizer
# ---------------------------------------------------------------------------

@dataclass
class QuipuLearnedParameters:
    """Calibrated parameters of the Quipu graph tension and utility dynamics."""
    w_figure_eight: float = 3.0       # Multiplier on Figure-8 hard barriers (skull mobs, LoS)
    w_threat_base: float = 1.0        # Base threat weight scaling
    w_dist_decay: float = 0.05        # Spatial distance damping factor in Phi(d)
    w_runner_urgency: float = 2.5     # Urgency multiplier on Flee/Add subsidiary cord
    w_escape_urgency: float = 3.0     # Urgency multiplier on Escape Path subsidiary cord
    w_rest_reward: float = 80.0       # Utility reward baseline for resting when low vitals
    w_los_pull_reward: float = 70.0   # Utility reward baseline for LoS pull against casters
    w_resource_cost_hp: float = 20.0  # Penalty weight for missing health in combat
    w_resource_cost_mp: float = 15.0  # Penalty weight for missing mana in combat
    temperature: float = 1.0          # Softmax inverse temperature beta

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.w_figure_eight,
            self.w_threat_base,
            self.w_dist_decay,
            self.w_runner_urgency,
            self.w_escape_urgency,
            self.w_rest_reward,
            self.w_los_pull_reward,
            self.w_resource_cost_hp,
            self.w_resource_cost_mp,
        ], dtype=np.float64)

    def from_vector(self, vec: np.ndarray) -> QuipuLearnedParameters:
        # Enforce non-negativity constraints via clipping
        v = np.clip(vec, 1e-4, 500.0)
        return QuipuLearnedParameters(
            w_figure_eight=float(v[0]),
            w_threat_base=float(v[1]),
            w_dist_decay=float(v[2]),
            w_runner_urgency=float(v[3]),
            w_escape_urgency=float(v[4]),
            w_rest_reward=float(v[5]),
            w_los_pull_reward=float(v[6]),
            w_resource_cost_hp=float(v[7]),
            w_resource_cost_mp=float(v[8]),
            temperature=self.temperature,
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class InverseGraphTensionOptimizer:
    """Optimizes Quipu tension weights from expert human demonstrations.
    
    Uses Softmax Cross-Entropy Loss over the candidate tactical action utilities.
    """

    CANDIDATE_ACTIONS = [
        TacticalActionType.PULL_TO_LOS,
        TacticalActionType.SNARE_FLEEING_RUNNER,
        TacticalActionType.ENGAGE_COMBAT,
        TacticalActionType.RETREAT_ESCAPE,
        TacticalActionType.REST_AND_RECOVER,
        TacticalActionType.HARVEST_AFFORDANCE,
        TacticalActionType.TRAVERSE_NAVMESH,
        TacticalActionType.HOLD_AND_WAIT,
    ]

    def __init__(self, demonstrations: list[DemonstrationStep], initial_params: Optional[QuipuLearnedParameters] = None) -> None:
        self.demonstrations = demonstrations
        self.params = initial_params or QuipuLearnedParameters()

    def evaluate_step_utilities(
        self,
        step: DemonstrationStep,
        params: QuipuLearnedParameters,
    ) -> dict[TacticalActionType, float]:
        """Calculates utility for each candidate tactical action given the state and parameters."""
        agent = step.agent_state
        utilities: dict[TacticalActionType, float] = {}

        # 1. Threat penalty sum with learnable parameters
        threat_penalty = 0.0
        active_combatants = [e for e in step.entities if e.is_combatant and e.is_hostile]
        for e in step.entities:
            if not e.is_hostile:
                continue
            dist = agent.position.distance_to(e.position)
            phi_d = 1.0 / (1.0 + params.w_dist_decay * dist)
            is_skull = (e.level - agent.level) >= 3

            base_w = 100.0 if is_skull else (50.0 if e.is_combatant else 20.0)
            if is_skull:
                threat_penalty += base_w * params.w_figure_eight * phi_d
            else:
                threat_penalty += base_w * params.w_threat_base * phi_d

        # 2. Resource depletion costs
        cost_combat = (1.0 - agent.hp_pct) * params.w_resource_cost_hp + (1.0 - agent.mp_pct) * params.w_resource_cost_mp

        # 3. Calculate candidate utilities
        # RETREAT_ESCAPE
        has_escape_need = (agent.hp_pct <= 0.25 and len(active_combatants) > 0) or len(active_combatants) >= 3
        escape_reward = 120.0 * params.w_escape_urgency if has_escape_need else 10.0
        utilities[TacticalActionType.RETREAT_ESCAPE] = escape_reward - (threat_penalty * 0.2)

        # SNARE_FLEEING_RUNNER
        fleeing_mobs = [e for e in active_combatants if e.is_humanoid and e.health_pct <= 0.20 and not e.is_snared]
        snare_reward = (95.0 * params.w_runner_urgency) if fleeing_mobs else -50.0
        utilities[TacticalActionType.SNARE_FLEEING_RUNNER] = snare_reward - threat_penalty

        # PULL_TO_LOS
        distant_casters = [e for e in active_combatants if e.is_casting and agent.position.distance_to(e.position) > 12.0]
        has_los_corner = any(step.occluders) or any(wp.is_choke_point for wp in step.nav_waypoints)
        los_reward = params.w_los_pull_reward if (distant_casters and has_los_corner) else -30.0
        utilities[TacticalActionType.PULL_TO_LOS] = los_reward - (threat_penalty * 0.5)

        # ENGAGE_COMBAT
        engage_reward = 60.0 if active_combatants else -20.0
        utilities[TacticalActionType.ENGAGE_COMBAT] = engage_reward - threat_penalty - cost_combat

        # REST_AND_RECOVER
        needs_rest = (agent.hp_pct < 0.40 or agent.mp_pct < 0.30) and not agent.is_in_combat
        rest_reward = params.w_rest_reward if needs_rest else -50.0
        utilities[TacticalActionType.REST_AND_RECOVER] = rest_reward

        # HARVEST_AFFORDANCE
        lootable = [a for a in step.affordances if a.is_lootable and not agent.is_in_combat]
        harvest_reward = max([a.yield_value for a in lootable], default=0.0) if lootable else -40.0
        utilities[TacticalActionType.HARVEST_AFFORDANCE] = harvest_reward - (threat_penalty * 0.5)

        # TRAVERSE_NAVMESH
        clear_wps = [wp for wp in step.nav_waypoints if wp.is_clearance_zone and not agent.is_in_combat]
        traverse_reward = 25.0 if clear_wps else 5.0
        utilities[TacticalActionType.TRAVERSE_NAVMESH] = traverse_reward - (threat_penalty * 0.3)

        # HOLD_AND_WAIT
        utilities[TacticalActionType.HOLD_AND_WAIT] = 0.0

        return utilities

    def compute_loss(self, theta_vec: np.ndarray, l2_reg: float = 1e-4) -> float:
        """Computes Negative Log-Likelihood (NLL) of demonstrations under candidate parameter vector."""
        params = self.params.from_vector(theta_vec)
        total_nll = 0.0
        prior_vec = QuipuLearnedParameters().to_vector()

        for step in self.demonstrations:
            utils = self.evaluate_step_utilities(step, params)
            # Softmax over candidate action utilities
            action_scores = np.array([utils[a] for a in self.CANDIDATE_ACTIONS], dtype=np.float64)
            # Stable softmax
            action_scores -= np.max(action_scores)
            exp_scores = np.exp(action_scores / params.temperature)
            probs = exp_scores / (np.sum(exp_scores) + 1e-12)

            human_idx = self.CANDIDATE_ACTIONS.index(step.human_action)
            prob_human = max(1e-12, probs[human_idx])
            total_nll -= math.log(prob_human)

        reg = l2_reg * float(np.sum((theta_vec - prior_vec) ** 2))
        return (total_nll / len(self.demonstrations)) + reg

    def train(self, epochs: int = 40, learning_rate: float = 0.05) -> Tuple[QuipuLearnedParameters, dict[str, Any]]:
        """Optimizes Quipu parameters using L-BFGS-B or gradient descent."""
        theta_init = self.params.to_vector()
        initial_loss = self.compute_loss(theta_init)

        try:
            from scipy.optimize import minimize
            bounds = [(1e-3, 500.0) for _ in range(len(theta_init))]
            res = minimize(
                self.compute_loss,
                theta_init,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": epochs},
            )
            theta = res.x
            final_loss = float(res.fun)
        except Exception as exc:
            logger.debug("L-BFGS fallback to gradient descent: %s", exc)
            theta = theta_init.copy()
            final_loss = initial_loss
            for epoch in range(epochs):
                loss = self.compute_loss(theta)
                grad = np.zeros_like(theta)
                eps = 1e-4
                for i in range(len(theta)):
                    theta_plus = theta.copy()
                    theta_plus[i] += eps
                    grad[i] = (self.compute_loss(theta_plus) - loss) / eps
                theta = np.clip(theta - learning_rate * grad, 1e-3, 500.0)
                final_loss = loss

        optimized_params = self.params.from_vector(theta)
        self.params = optimized_params

        final_acc = self.evaluate_accuracy(theta)
        stats = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "final_accuracy": round(final_acc, 4),
            "epochs_run": epochs,
            "learned_parameters": optimized_params.to_dict(),
        }

        # Persist to brain_kv if available
        if _brain_kv is not None:
            try:
                _brain_kv.kv_set_json("quipu:learned_game_parameters", optimized_params.to_dict())
            except Exception as exc:
                logger.debug("Could not persist to brain_kv: %s", exc)

        return optimized_params, stats

    def evaluate_accuracy(self, theta_vec: np.ndarray) -> float:
        """Computes Top-1 classification accuracy against demonstrated actions."""
        params = self.params.from_vector(theta_vec)
        correct = 0
        for step in self.demonstrations:
            utils = self.evaluate_step_utilities(step, params)
            best_action = max(utils, key=lambda a: utils[a])
            if best_action == step.human_action:
                correct += 1
        return correct / len(self.demonstrations) if self.demonstrations else 0.0


# ---------------------------------------------------------------------------
# Stage 5: End-to-End Pipeline Controller
# ---------------------------------------------------------------------------

class VideoGameplayPipeline:
    """Coordinates video ingestion, frame extraction, intent parsing, and Quipu training."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir or "quipu_pipeline_artifacts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learned_params: Optional[QuipuLearnedParameters] = None

    def build_synthetic_demonstrations(self, count: int = 30) -> list[DemonstrationStep]:
        """Creates a deterministic reference demonstration corpus representing classic MMO encounters."""
        demonstrations: list[DemonstrationStep] = []

        for i in range(count):
            scenario_idx = i % 5

            if scenario_idx == 0:
                # Scenario 0: Caster pulled behind LoS pillar
                agent = AgentCoreState("p1", "Warrior", 15, "Warrior", Vector3(0, 0, 0), 0.0, 0.9, 0.8, is_in_combat=True)
                caster = WorldEntity("c1", "Defias Pyromancer", Vector3(22, 0, 0), 15, 1.0, 400, is_hostile=True, is_friendly=False, is_humanoid=True, is_combatant=True, is_casting=True, is_interruptible=True, is_stunned=False, is_snared=False)
                pillar = LoSOccluder("pil1", Vector3(6, 0, 0), Vector3(0, 0, 0), is_pillar=True, radius=2.0)
                step = DemonstrationStep(
                    step_id=f"demo_{i}_los",
                    timestamp_s=float(i * 5),
                    agent_state=agent,
                    entities=[caster],
                    nav_waypoints=[NavMeshWaypoint("wp1", Vector3(6, 4, 0), is_clearance_zone=True, is_choke_point=True)],
                    affordances=[],
                    occluders=[pillar],
                    speech_intent=StreamerSpeechIntent.LOS_TACTIC,
                    human_action=TacticalActionType.PULL_TO_LOS,
                )

            elif scenario_idx == 1:
                # Scenario 1: Fleeing humanoid approaching add
                agent = AgentCoreState("p1", "Warrior", 15, "Warrior", Vector3(0, 0, 0), 0.0, 0.8, 0.7, is_in_combat=True)
                runner = WorldEntity("r1", "Defias Highwayman", Vector3(12, 0, 0), 15, 0.15, 500, is_hostile=True, is_friendly=False, is_humanoid=True, is_combatant=True, is_casting=False, is_interruptible=False, is_stunned=False, is_snared=False)
                add = WorldEntity("a1", "Defias Scout", Vector3(28, 2, 0), 15, 1.0, 450, is_hostile=True, is_friendly=False, is_humanoid=True, is_combatant=False, is_casting=False, is_interruptible=False, is_stunned=False, is_snared=False)
                step = DemonstrationStep(
                    step_id=f"demo_{i}_runner",
                    timestamp_s=float(i * 5),
                    agent_state=agent,
                    entities=[runner, add],
                    nav_waypoints=[],
                    affordances=[],
                    occluders=[],
                    speech_intent=StreamerSpeechIntent.RUNNER_ALERT,
                    human_action=TacticalActionType.SNARE_FLEEING_RUNNER,
                )

            elif scenario_idx == 2:
                # Scenario 2: Emergency retreat at low HP
                agent = AgentCoreState("p1", "Warrior", 15, "Warrior", Vector3(0, 0, 0), 0.0, 0.18, 0.1, is_in_combat=True)
                boss = WorldEntity("b1", "Elite Guard", Vector3(4, 0, 0), 17, 0.9, 1200, is_hostile=True, is_friendly=False, is_humanoid=True, is_combatant=True, is_casting=False, is_interruptible=False, is_stunned=False, is_snared=False)
                safe_wp = NavMeshWaypoint("safe1", Vector3(-35, 0, 0), is_clearance_zone=True, is_safe_escape_node=True)
                step = DemonstrationStep(
                    step_id=f"demo_{i}_flee",
                    timestamp_s=float(i * 5),
                    agent_state=agent,
                    entities=[boss],
                    nav_waypoints=[safe_wp],
                    affordances=[],
                    occluders=[],
                    speech_intent=StreamerSpeechIntent.EMERGENCY_FLEE,
                    human_action=TacticalActionType.RETREAT_ESCAPE,
                )

            elif scenario_idx == 3:
                # Scenario 3: Downtime eating/drinking
                agent = AgentCoreState("p1", "Warrior", 15, "Warrior", Vector3(0, 0, 0), 0.0, 0.32, 0.22, is_in_combat=False)
                step = DemonstrationStep(
                    step_id=f"demo_{i}_rest",
                    timestamp_s=float(i * 5),
                    agent_state=agent,
                    entities=[],
                    nav_waypoints=[NavMeshWaypoint("wp_rest", Vector3(0, 0, 0), is_clearance_zone=True)],
                    affordances=[],
                    occluders=[],
                    speech_intent=StreamerSpeechIntent.REST_MANDATE,
                    human_action=TacticalActionType.REST_AND_RECOVER,
                )

            else:
                # Scenario 4: Affordance harvest
                agent = AgentCoreState("p1", "Warrior", 15, "Warrior", Vector3(0, 0, 0), 0.0, 0.95, 0.90, is_in_combat=False)
                chest = WorldAffordance("ch1", "Solid Chest", Vector3(8, 0, 0), "chest", is_locked=False, is_lootable=True, yield_value=50.0)
                step = DemonstrationStep(
                    step_id=f"demo_{i}_loot",
                    timestamp_s=float(i * 5),
                    agent_state=agent,
                    entities=[],
                    nav_waypoints=[NavMeshWaypoint("wp_path", Vector3(10, 0, 0), is_clearance_zone=True)],
                    affordances=[chest],
                    occluders=[],
                    speech_intent=StreamerSpeechIntent.NEUTRAL_CHAT,
                    human_action=TacticalActionType.HARVEST_AFFORDANCE,
                )

            demonstrations.append(step)

        return demonstrations

    def run_training_pipeline(
        self,
        demonstrations: Optional[list[DemonstrationStep]] = None,
        epochs: int = 35,
        learning_rate: float = 0.05,
    ) -> Tuple[QuipuLearnedParameters, dict[str, Any]]:
        """Executes full IGTL training and saves artifacts."""
        corpus = demonstrations or self.build_synthetic_demonstrations(40)
        optimizer = InverseGraphTensionOptimizer(demonstrations=corpus)
        params, stats = optimizer.train(epochs=epochs, learning_rate=learning_rate)
        self.learned_params = params

        # Save to disk
        out_file = self.output_dir / "quipu_learned_parameters.json"
        out_file.write_text(json.dumps(params.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved trained parameters to %s", out_file)

        return params, stats


__all__ = [
    "HUDVitals",
    "MinimapBlip",
    "FramePerceptionObservation",
    "HUDVisionExtractor",
    "StreamerSpeechIntent",
    "SpeechTranscriptSegment",
    "StreamerAudioParser",
    "DemonstrationStep",
    "QuipuLearnedParameters",
    "InverseGraphTensionOptimizer",
    "VideoGameplayPipeline",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QUIPU Video-to-Graph Pre-Training Pipeline")
    parser.add_argument("--epochs", type=int, default=30, help="Number of L-BFGS optimization epochs")
    parser.add_argument("--demos", type=int, default=50, help="Number of demonstration frames to generate/use")
    parser.add_argument("--output", type=str, default="quipu_learned_artifacts", help="Output directory for learned parameters")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n=======================================================")
    print("   QUIPU Video-to-Graph Pre-Training Pipeline (IGTL)   ")
    print("=======================================================\n")

    pipeline = VideoGameplayPipeline(output_dir=args.output)
    print(f"[*] Generating {args.demos} expert gameplay demonstration steps...")
    demos = pipeline.build_synthetic_demonstrations(count=args.demos)

    print(f"[*] Running Inverse Graph Tension Optimization ({args.epochs} max iterations)...")
    t0 = time.time()
    params, stats = pipeline.run_training_pipeline(demonstrations=demos, epochs=args.epochs)
    duration = time.time() - t0

    print("\n" + "=" * 55)
    print(f"   TRAINING COMPLETE in {duration:.2f}s")
    print("=" * 55)
    print(f" Initial NLL Loss : {stats['initial_loss']:.4f}")
    print(f" Final NLL Loss   : {stats['final_loss']:.4f}")
    print(f" Top-1 Accuracy   : {stats['final_accuracy'] * 100:.1f}%")
    print("\n Learned Quipu Tension Parameters:")
    for k, v in stats["learned_parameters"].items():
        print(f"   - {k:<20}: {v:.4f}")
    print(f"\n Artifact saved to: {args.output}/quipu_learned_parameters.json")
    print("=======================================================\n")

