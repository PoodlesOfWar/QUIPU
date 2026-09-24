"""QUIPU Game Mesh Autonomy Engine — Hierarchical Andean Graph Topology for In-Game Decision-Making.

Models autonomous perception, tactical threat geometry, spatial navigation, and
behavioral actuation for classic/ground-bound MMORPG environments (e.g., WoW: Forever).

Graph Architecture:
-------------------
                      [ PRIMARY TRUNK CORD ]
              (Agent Core State & Global Objective)
                        |         |         |
        +---------------+         |         +---------------+
        |                         |                         |
  [PENDANT: SPATIAL]      [PENDANT: ENTITY]       [PENDANT: AFFORDANCE]
   (NavMesh / Corridors)    (Threats / NPCs)        (Nodes / Containers)
        |                         |                         |
     o--Knot (Dist/Cost)       o--Knot (Threat/Level)    o--Knot (Yield/Req)
        |                         |
   [SUBSIDIARY: ESCAPE]      [SUBSIDIARY: ADDS]
     o--Knot (Safe Point)      o--Knot (Flee Trajectory)

Core Mechanical Invariants:
---------------------------
1. Ground-Bound Navigation & Terrain Friction:
   - NavMesh centroids, elevation barriers, choke points, line-of-sight (LoS) pillars.
2. Threat Geometry & Classic Aggro Physics:
   - Dynamic aggro radius with level-delta scaling: R = clamp(20 + 1.5 * delta_level, 5, 45).
   - Figure-8 Knot: Hard barrier for mobs >= 3 levels above (crushing blow / skull danger) or LoS obstruction.
   - Social Aggro & Link Pulls: Adjacent hostiles within social radius (5-10 yards) alert pack mates.
   - Runners & Flee Mechanics: Humanoid mobs flee at <= 20% HP; projected flee trajectory evaluated
     against surrounding unpulled hostile packs to sprout a "Flee / Add" subsidiary cord.
   - Leash Limits: Mobs dragged > 38-40 yards from spawn anchor evade and reset.
3. Resource Attrition & Downtime Pacing:
   - Non-instant HP/MP regeneration; safe rest/drink downtime cycles.
4. Decision & Traversal Cycle:
   - Perception Sampling & Graph Weaving
   - Graph Tension Evaluation & Utility Optimization
   - Actuation (Pull to LoS, Snare Runner, Engage, Escape, Rest, Harvest) & Knot Pruning
   - Distillation to Minimal Context Subgraph for low-latency execution.
"""

from __future__ import annotations

import enum
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Try to import brain_kv if available in QUIPU package; fallback gracefully for standalone use
try:
    from . import brain_kv as _brain_kv
except ImportError:
    _brain_kv = None  # type: ignore


# ---------------------------------------------------------------------------
# Physical Constants & Rule Thresholds (WoW: Forever Classic Mechanics)
# ---------------------------------------------------------------------------

DEFAULT_SENSORY_RADIUS: float = 80.0       # Perception horizon in yards
BASE_AGGRO_RADIUS: float = 20.0            # Base aggro radius against equal-level enemy (yards)
LEVEL_DELTA_AGGRO_SCALE: float = 1.5       # Yard delta per level difference
MIN_AGGRO_RADIUS: float = 5.0              # Minimum aggro radius floor (yards)
MAX_AGGRO_RADIUS: float = 45.0             # Maximum aggro radius ceiling (yards)
SKULL_LEVEL_THRESHOLD: int = 3             # Level delta at which enemy is a Figure-8 hard barrier
SOCIAL_AGGRO_RADIUS: float = 8.0           # Radius around pulled mob that triggers pack alert (yards)
RUNNER_HEALTH_THRESHOLD: float = 0.20      # Health fraction (<= 20%) where humanoids flee
PROJECTED_FLEE_DISTANCE: float = 25.0      # Forward trajectory length checked for flee path (yards)
LEASH_RESET_DISTANCE: float = 38.0         # Distance from spawn anchor before mob evades/resets (yards)

LOW_HEALTH_SURVIVAL_THRESHOLD: float = 0.25 # Agent HP% below which Escape Subsidiary cord sprouts
CRITICAL_TENSION_REROUTE: float = 75.0      # Tension threshold triggering emergency re-route
REST_HP_TRIGGER: float = 0.40              # Out-of-combat HP threshold to trigger eating
REST_MP_TRIGGER: float = 0.30              # Out-of-combat MP threshold to trigger drinking
REST_RECOVERY_TARGET: float = 0.85         # Target HP/MP fraction to finish resting


# ---------------------------------------------------------------------------
# Enums: Quipu Topology & Game Semantics
# ---------------------------------------------------------------------------

class KnotType(str, enum.Enum):
    """Quipu knot structural complexity classification."""
    FIGURE_EIGHT = "figure_eight"  # Hard Barrier: LoS occlusion, locked door, skull mob >= 3 levels
    LONG_KNOT = "long_knot"        # Scalar Intensity: N loops encoding continuous threat/DPS/HP%
    SINGLE_KNOT = "single_knot"    # Binary State: 1 loop = True / 0 loop = False (stunned, interruptible, lootable)


class PendantDomain(str, enum.Enum):
    """Perceptual domain hanging off the Primary Trunk Cord."""
    SPATIAL = "spatial"            # NavMesh, corridors, LoS occluders, elevation, clearance zones
    THREAT = "threat"              # Hostile/neutral/friendly entities, patrols, combatants
    AFFORDANCE = "affordance"      # Interactables, herbs, ore, quest chests, clickable doors


class SubsidiaryType(str, enum.Enum):
    """Contingency branch types sprouting from specific parent knots."""
    FLEE_ADD = "flee_add"          # Mob <= 20% HP fleeing toward adjacent packs
    ESCAPE_PATH = "escape_path"    # Agent health critical; safe corridor away from threat vectors
    REST_RECOVERY = "rest_recovery"# Out-of-combat attrition downtime (eating/drinking)
    CHOKE_AMBUSH = "choke_ambush"  # Narrow pass danger requiring pull back


class MacroIntent(str, enum.Enum):
    """Agent core vector global intent."""
    COMPLETE_QUEST = "complete_quest"
    FARM_RESOURCE = "farm_resource"
    TRAVEL_TO_TARGET = "travel_to_target"
    SURVIVE_AND_RESET = "survive_and_reset"
    REST_AND_MAINTAIN = "rest_and_maintain"


class TacticalActionType(str, enum.Enum):
    """Executable action resulting from graph tension evaluation."""
    PULL_TO_LOS = "pull_to_los"                  # Pull target, break line of sight around corner
    SNARE_FLEEING_RUNNER = "snare_fleeing_runner"# Apply slow/stun to humanoid mob fleeing toward adds
    ENGAGE_COMBAT = "engage_combat"              # Execute combat rotation in clearance zone
    RETREAT_ESCAPE = "retreat_escape"            # Follow safe escape subsidiary corridor
    REST_AND_RECOVER = "rest_and_recover"        # Eat/drink out of combat in safe clearance zone
    HARVEST_AFFORDANCE = "harvest_affordance"    # Gather node, open container, interact with door
    TRAVERSE_NAVMESH = "traverse_navmesh"        # Path towards next objective waypoint
    HOLD_AND_WAIT = "hold_and_wait"              # Hold position for patrol clearance


# ---------------------------------------------------------------------------
# Coordinate Geometry & 3D Spatial Types
# ---------------------------------------------------------------------------

@dataclass
class Vector3:
    """3D point / vector in world coordinates (yards)."""
    x: float
    y: float
    z: float

    def distance_to(self, other: Vector3) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def horizontal_distance_to(self, other: Vector3) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def heading_to(self, other: Vector3) -> float:
        """Yaw angle (radians) from self towards other."""
        return math.atan2(other.y - self.y, other.x - self.x)

    def normalized_2d(self) -> Vector3:
        mag = math.sqrt(self.x ** 2 + self.y ** 2)
        if mag < 1e-6:
            return Vector3(1.0, 0.0, 0.0)
        return Vector3(self.x / mag, self.y / mag, 0.0)

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "z": round(self.z, 3)}


@dataclass
class LoSOccluder:
    """Line-of-Sight barrier (pillar, wall, terrain ridge)."""
    id: str
    p1: Vector3  # Start of wall / center of pillar
    p2: Vector3  # End of wall (or radius in z for pillar)
    is_pillar: bool = False
    radius: float = 1.5

    def blocks_line(self, a: Vector3, b: Vector3) -> bool:
        """Determines if segment AB intersects this LoS occluder in the horizontal plane."""
        if self.is_pillar:
            px = self.p1.x
            py = self.p1.y
            abx = b.x - a.x
            aby = b.y - a.y
            ab_len_sq = abx ** 2 + aby ** 2
            if ab_len_sq < 1e-6:
                return a.horizontal_distance_to(self.p1) <= self.radius
            t = max(0.0, min(1.0, ((px - a.x) * abx + (py - a.y) * aby) / ab_len_sq))
            proj_x = a.x + t * abx
            proj_y = a.y + t * aby
            dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
            return dist_sq <= (self.radius ** 2)
        else:
            def ccw(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
                return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
            A = (a.x, a.y)
            B = (b.x, b.y)
            C = (self.p1.x, self.p1.y)
            D = (self.p2.x, self.p2.y)
            return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


# ---------------------------------------------------------------------------
# Quipu Topological Graph Data Structures
# ---------------------------------------------------------------------------

@dataclass
class QuipuKnot:
    """A weighted state attribute on a Quipu cord.
    
    Attributes:
        knot_id: Unique identifier.
        knot_type: Figure-8, Long Knot, or Single Knot.
        loops: Number of loops / scalar intensity (Long Knot: N; Single: 0 or 1; Fig-8: barrier severity).
        position_on_cord: Distance along cord from agent (in yards; depth from sensory core).
        weight: Raw numerical weight / threat value / yield value.
        label: Descriptive label (e.g. 'Defias Pillager', 'Peacebloom', 'LoS Pillar', 'Runner Vector').
        payload: Rich metadata dict (GUID, health, level, coords, flags).
        is_active: Whether knot is currently tied and contributing tension.
        created_at: Tick or unix timestamp.
    """
    knot_id: str
    knot_type: KnotType
    loops: float
    position_on_cord: float
    weight: float
    label: str
    payload: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: float = field(default_factory=time.time)

    def effective_tension(self) -> float:
        """Calculates tension exerted by this knot onto its cord.
        
        Tension scales with weight and knot loop count, with hyper-linear
        penalty for Figure-8 hard barriers and spatial proximity weighting.
        """
        if not self.is_active:
            return 0.0

        # Proximity weight: closer knots exert stronger tension (d in yards)
        proximity_factor = 1.0 / (1.0 + 0.05 * max(0.0, self.position_on_cord))

        if self.knot_type == KnotType.FIGURE_EIGHT:
            # Figure-8 knots represent hard barriers / lethal constraints
            return (self.weight * 3.0 + 100.0) * proximity_factor
        elif self.knot_type == KnotType.LONG_KNOT:
            # Scalar intensity based on loop count (e.g. DPS, aggro level)
            return self.weight * (1.0 + 0.2 * self.loops) * proximity_factor
        elif self.knot_type == KnotType.SINGLE_KNOT:
            # Binary indicator: loop count 1 or 0
            return (self.weight if self.loops >= 1.0 else 0.0) * proximity_factor
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "knot_id": self.knot_id,
            "knot_type": self.knot_type.value,
            "loops": round(self.loops, 2),
            "position_on_cord": round(self.position_on_cord, 2),
            "weight": round(self.weight, 2),
            "label": self.label,
            "effective_tension": round(self.effective_tension(), 2),
            "is_active": self.is_active,
            "payload": self.payload,
        }


@dataclass
class SubsidiaryCord:
    """Contingency branch sprouting from a parent knot under emergent conditions."""
    subsidiary_id: str
    parent_knot_id: str
    subsidiary_type: SubsidiaryType
    urgency_multiplier: float
    knots: list[QuipuKnot] = field(default_factory=list)
    description: str = ""

    def total_tension(self) -> float:
        base = sum(k.effective_tension() for k in self.knots if k.is_active)
        return base * self.urgency_multiplier

    def to_dict(self) -> dict[str, Any]:
        return {
            "subsidiary_id": self.subsidiary_id,
            "parent_knot_id": self.parent_knot_id,
            "subsidiary_type": self.subsidiary_type.value,
            "urgency_multiplier": round(self.urgency_multiplier, 2),
            "total_tension": round(self.total_tension(), 2),
            "description": self.description,
            "knots": [k.to_dict() for k in self.knots],
        }


@dataclass
class PendantCord:
    """Contextual perceptual cord hanging directly from the primary trunk."""
    domain: PendantDomain
    knots: list[QuipuKnot] = field(default_factory=list)
    subsidiary_cords: list[SubsidiaryCord] = field(default_factory=list)

    def total_tension(self) -> float:
        knot_tension = sum(k.effective_tension() for k in self.knots if k.is_active)
        sub_tension = sum(sub.total_tension() for sub in self.subsidiary_cords)
        return knot_tension + sub_tension

    def add_knot(self, knot: QuipuKnot) -> None:
        self.knots.append(knot)
        # Keep knots ordered by distance from agent (position on cord)
        self.knots.sort(key=lambda k: k.position_on_cord)

    def prune_knot(self, knot_id: str) -> bool:
        for k in self.knots:
            if k.knot_id == knot_id:
                k.is_active = False
                # Also prune any subsidiary cord sprouted from this knot
                self.subsidiary_cords = [s for s in self.subsidiary_cords if s.parent_knot_id != knot_id]
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "total_tension": round(self.total_tension(), 2),
            "knot_count": len([k for k in self.knots if k.is_active]),
            "subsidiary_count": len(self.subsidiary_cords),
            "knots": [k.to_dict() for k in self.knots if k.is_active],
            "subsidiaries": [s.to_dict() for s in self.subsidiary_cords],
        }


# ---------------------------------------------------------------------------
# Agent Core Vector (The Primary Trunk Cord)
# ---------------------------------------------------------------------------

@dataclass
class AgentCoreState:
    """Agent internal continuity, resources, and global intent."""
    agent_id: str
    name: str
    level: int
    character_class: str
    position: Vector3
    heading: float                          # Heading in radians
    hp_pct: float                           # 0.0 to 1.0
    mp_pct: float                           # 0.0 to 1.0
    stamina_pct: float = 1.0                # 0.0 to 1.0
    is_in_combat: bool = False
    bag_free_slots: int = 12
    bag_capacity: int = 16
    active_auras: list[str] = field(default_factory=list)
    global_intent: MacroIntent = MacroIntent.COMPLETE_QUEST
    objective_target_name: str = ""
    objective_count_current: int = 0
    objective_count_required: int = 10
    time_horizon_ticks: int = 100           # Ticks until next required state refresh

    def needs_rest(self) -> bool:
        """Determines if resource attrition mandates drinking/eating out of combat."""
        if self.is_in_combat:
            return False
        return (self.hp_pct < REST_HP_TRIGGER) or (self.mp_pct < REST_MP_TRIGGER)


@dataclass
class PrimaryTrunkCord:
    """The central Quipu spine linking Agent Core State with all Pendant Cords."""
    agent: AgentCoreState
    pendants: dict[PendantDomain, PendantCord] = field(default_factory=dict)
    last_cycle_timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.pendants:
            self.pendants = {
                PendantDomain.SPATIAL: PendantCord(domain=PendantDomain.SPATIAL),
                PendantDomain.THREAT: PendantCord(domain=PendantDomain.THREAT),
                PendantDomain.AFFORDANCE: PendantCord(domain=PendantDomain.AFFORDANCE),
            }

    @property
    def spatial(self) -> PendantCord:
        return self.pendants[PendantDomain.SPATIAL]

    @property
    def threat(self) -> PendantCord:
        return self.pendants[PendantDomain.THREAT]

    @property
    def affordance(self) -> PendantCord:
        return self.pendants[PendantDomain.AFFORDANCE]

    def total_system_tension(self) -> float:
        return sum(p.total_tension() for p in self.pendants.values())

    def reset_cords(self) -> None:
        """Resets all cords for the next perception tick."""
        for p in self.pendants.values():
            p.knots.clear()
            p.subsidiary_cords.clear()


# ---------------------------------------------------------------------------
# Perception Inputs: World State Entities & Sensor Telemetry
# ---------------------------------------------------------------------------

@dataclass
class WorldEntity:
    """Observed entity in the game environment."""
    guid: str
    name: str
    position: Vector3
    level: int
    health_pct: float
    max_health: int
    is_hostile: bool
    is_friendly: bool
    is_humanoid: bool
    is_combatant: bool
    is_casting: bool
    is_interruptible: bool
    is_stunned: bool
    is_snared: bool
    target_guid: Optional[str] = None
    spawn_anchor: Optional[Vector3] = None
    is_patrol: bool = False
    facing_angle: float = 0.0


@dataclass
class NavMeshWaypoint:
    """Centroid waypoint on the 3D NavMesh."""
    waypoint_id: str
    position: Vector3
    is_clearance_zone: bool = True     # Free of hostile patrols
    elevation_grade: float = 0.0       # Elevation slope
    is_choke_point: bool = False
    is_safe_escape_node: bool = False


@dataclass
class WorldAffordance:
    """Contextual interactable game object."""
    guid: str
    name: str
    position: Vector3
    affordance_type: str               # 'herb', 'ore', 'quest_chest', 'door', 'flight_master'
    is_locked: bool = False
    has_key_in_bag: bool = True
    is_lootable: bool = True
    yield_value: float = 10.0


# ---------------------------------------------------------------------------
# Tactical Action Output
# ---------------------------------------------------------------------------

@dataclass
class TacticalAction:
    """Action resulting from Graph Tension Evaluation."""
    action_type: TacticalActionType
    target_guid: Optional[str]
    target_position: Optional[Vector3]
    utility_score: float
    explanation: str
    associated_knot_id: Optional[str] = None
    subsidiary_cord_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target_guid": self.target_guid,
            "target_position": self.target_position.to_dict() if self.target_position else None,
            "utility_score": round(self.utility_score, 2),
            "explanation": self.explanation,
            "associated_knot_id": self.associated_knot_id,
            "subsidiary_cord_id": self.subsidiary_cord_id,
        }


# ---------------------------------------------------------------------------
# The QUIPU Game Mesh Autonomy Engine
# ---------------------------------------------------------------------------

class QuipuGameMeshEngine:
    """Hierarchical Quipu Graph Autonomy Engine for Classic MMORPGs.
    
    Executes the 3-phase cyclical loop:
    1. Perception Sampling & Graph Weaving
    2. Graph Tension Evaluation & Dynamic Re-Routing
    3. Actuation & Knot Pruning
    """

    def __init__(self, agent_state: AgentCoreState) -> None:
        self.trunk = PrimaryTrunkCord(agent=agent_state)
        self.occluders: list[LoSOccluder] = []
        self.active_combat_target_guid: Optional[str] = None
        self.cycle_count: int = 0
        self.last_action: Optional[TacticalAction] = None

    def register_los_occluder(self, occluder: LoSOccluder) -> None:
        """Registers a terrain barrier or LoS pillar."""
        self.occluders.append(occluder)

    def check_los(self, a: Vector3, b: Vector3) -> bool:
        """Returns True if clear line of sight exists between A and B."""
        for occluder in self.occluders:
            if occluder.blocks_line(a, b):
                return False
        return True

    # -----------------------------------------------------------------------
    # Phase 1: Perception Sampling & Graph Weaving
    # -----------------------------------------------------------------------

    def weave_perceptions(
        self,
        entities: list[WorldEntity],
        nav_waypoints: list[NavMeshWaypoint],
        affordances: list[WorldAffordance],
        sensory_radius: float = DEFAULT_SENSORY_RADIUS,
    ) -> None:
        """Constructs Quipu knots along Spatial, Threat, and Affordance cords."""
        self.trunk.reset_cords()
        agent = self.trunk.agent
        agent_pos = agent.position

        # -------------------------------------------------------------------
        # 1. Weave Spatial & Topological Cord
        # -------------------------------------------------------------------
        spatial_cord = self.trunk.spatial
        for wp in nav_waypoints:
            dist = agent_pos.distance_to(wp.position)
            if dist > sensory_radius:
                continue

            # Knot complexity for navmesh nodes:
            # Figure-8 if choke point or LoS occluded; Long knot with cost loops otherwise
            has_los = self.check_los(agent_pos, wp.position)
            if wp.is_choke_point or not has_los:
                k_type = KnotType.FIGURE_EIGHT
                loops = 2.0
                weight = 15.0 if wp.is_choke_point else 25.0
            else:
                k_type = KnotType.LONG_KNOT
                loops = max(1.0, wp.elevation_grade * 10.0)  # Loops encode elevation friction
                weight = 5.0 + dist * 0.1

            k_label = f"NavWP:{wp.waypoint_id}:{'Clear' if wp.is_clearance_zone else 'PatrolHazard'}"
            knot = QuipuKnot(
                knot_id=f"spatial_{wp.waypoint_id}",
                knot_type=k_type,
                loops=loops,
                position_on_cord=dist,
                weight=weight,
                label=k_label,
                payload={
                    "waypoint_id": wp.waypoint_id,
                    "position": wp.position.to_dict(),
                    "is_clearance_zone": wp.is_clearance_zone,
                    "is_safe_escape_node": wp.is_safe_escape_node,
                    "has_los": has_los,
                },
            )
            spatial_cord.add_knot(knot)

        # -------------------------------------------------------------------
        # 2. Weave Threat & Entity Cord
        # -------------------------------------------------------------------
        threat_cord = self.trunk.threat
        surrounding_hostiles: list[Tuple[WorldEntity, float]] = [
            (e, agent_pos.distance_to(e.position))
            for e in entities
            if e.is_hostile and agent_pos.distance_to(e.position) <= sensory_radius
        ]

        for entity in entities:
            dist = agent_pos.distance_to(entity.position)
            if dist > sensory_radius:
                continue

            if not entity.is_hostile:
                # Friendly or neutral NPC
                knot = QuipuKnot(
                    knot_id=f"entity_{entity.guid}",
                    knot_type=KnotType.SINGLE_KNOT,
                    loops=0.0,
                    position_on_cord=dist,
                    weight=0.0,
                    label=f"Friendly:{entity.name}",
                    payload={"guid": entity.guid, "level": entity.level},
                )
                threat_cord.add_knot(knot)
                continue

            # Dynamic Classic Threat Geometry
            delta_level = entity.level - agent.level
            aggro_radius = max(
                MIN_AGGRO_RADIUS,
                min(MAX_AGGRO_RADIUS, BASE_AGGRO_RADIUS + delta_level * LEVEL_DELTA_AGGRO_SCALE),
            )

            # Check leash reset status
            is_leashed = False
            if entity.spawn_anchor:
                leash_dist = entity.position.distance_to(entity.spawn_anchor)
                if leash_dist >= LEASH_RESET_DISTANCE:
                    is_leashed = True

            # Determine knot type based on danger rules
            if delta_level >= SKULL_LEVEL_THRESHOLD:
                # Figure-8 Knot: Hard Barrier (Skull Danger)
                k_type = KnotType.FIGURE_EIGHT
                loops = float(delta_level)
                base_weight = 100.0 + delta_level * 20.0
            elif entity.is_combatant:
                # Long Knot: High scalar intensity for active combatant
                k_type = KnotType.LONG_KNOT
                loops = 5.0 + max(0.0, float(delta_level))
                base_weight = 50.0 + (1.0 - entity.health_pct) * 20.0
            elif dist <= aggro_radius:
                # Long Knot: Hostile within aggro sphere
                k_type = KnotType.LONG_KNOT
                loops = 3.0
                base_weight = 35.0
            else:
                # Long Knot: Hostile outside aggro sphere
                k_type = KnotType.LONG_KNOT
                loops = 1.0
                base_weight = 10.0

            # Single Knots for binary combat modifiers
            if entity.is_stunned:
                base_weight *= 0.5
            if entity.is_casting and entity.is_interruptible:
                loops += 2.0  # High urgency to kick/counterspell

            knot = QuipuKnot(
                knot_id=f"threat_{entity.guid}",
                knot_type=k_type,
                loops=loops,
                position_on_cord=dist,
                weight=base_weight,
                label=f"Threat:{entity.name}(Lvl{entity.level})",
                payload={
                    "guid": entity.guid,
                    "name": entity.name,
                    "level": entity.level,
                    "delta_level": delta_level,
                    "health_pct": entity.health_pct,
                    "aggro_radius": aggro_radius,
                    "is_combatant": entity.is_combatant,
                    "is_humanoid": entity.is_humanoid,
                    "is_leashed": is_leashed,
                    "is_casting": entity.is_casting,
                    "is_interruptible": entity.is_interruptible,
                    "is_stunned": entity.is_stunned,
                    "is_snared": entity.is_snared,
                    "position": entity.position.to_dict(),
                    "has_los": self.check_los(agent_pos, entity.position),
                },
            )
            threat_cord.add_knot(knot)

            # ---------------------------------------------------------------
            # Emergent Subsidiary: Flee / Add Contingency Branch
            # ---------------------------------------------------------------
            if (
                entity.is_combatant
                and entity.is_humanoid
                and entity.health_pct <= RUNNER_HEALTH_THRESHOLD
                and not entity.is_snared
                and not entity.is_stunned
            ):
                # Humanoids flee directly away from agent
                flee_dir = Vector3(
                    entity.position.x - agent_pos.x,
                    entity.position.y - agent_pos.y,
                    0.0,
                ).normalized_2d()

                flee_target = Vector3(
                    entity.position.x + flee_dir.x * PROJECTED_FLEE_DISTANCE,
                    entity.position.y + flee_dir.y * PROJECTED_FLEE_DISTANCE,
                    entity.position.z,
                )

                # Check if projected flee trajectory approaches any other unpulled hostiles
                risk_knots: list[QuipuKnot] = []
                for other, other_dist in surrounding_hostiles:
                    if other.guid == entity.guid or other.is_combatant:
                        continue
                    # Distance from other mob to the flee line segment
                    dist_to_flee_path = self._point_to_segment_dist(
                        other.position, entity.position, flee_target
                    )
                    if dist_to_flee_path <= SOCIAL_AGGRO_RADIUS:
                        # Social aggro will trigger! Sprout a high-urgency risk knot
                        risk_knot = QuipuKnot(
                            knot_id=f"flee_add_risk_{other.guid}",
                            knot_type=KnotType.FIGURE_EIGHT,
                            loops=4.0,
                            position_on_cord=dist_to_flee_path,
                            weight=80.0,
                            label=f"AddLeashRisk:{other.name}",
                            payload={
                                "unpulled_add_guid": other.guid,
                                "dist_to_flee_path": dist_to_flee_path,
                            },
                        )
                        risk_knots.append(risk_knot)

                if risk_knots:
                    flee_sub = SubsidiaryCord(
                        subsidiary_id=f"flee_sub_{entity.guid}",
                        parent_knot_id=knot.knot_id,
                        subsidiary_type=SubsidiaryType.FLEE_ADD,
                        urgency_multiplier=2.5,  # Substantial tension multiplier
                        knots=risk_knots,
                        description=f"{entity.name} fleeing towards {len(risk_knots)} unpulled packs",
                    )
                    threat_cord.subsidiary_cords.append(flee_sub)

        # -------------------------------------------------------------------
        # Emergent Subsidiary: Escape Path Contingency Branch
        # -------------------------------------------------------------------
        active_combatants = [
            e for e, _ in surrounding_hostiles if e.is_combatant and not e.is_friendly
        ]
        if (
            (agent.hp_pct <= LOW_HEALTH_SURVIVAL_THRESHOLD and len(active_combatants) > 0)
            or len(active_combatants) >= 3  # Unplanned multi-pack pull
        ):
            # Trace safe escape path towards safe waypoints away from all threat vectors
            safe_escape_knots: list[QuipuKnot] = []
            for wp in nav_waypoints:
                if not wp.is_safe_escape_node and not wp.is_clearance_zone:
                    continue
                # Calculate minimum distance from this waypoint to any active combatant
                min_threat_dist = min(
                    (wp.position.distance_to(c.position) for c in active_combatants),
                    default=999.0,
                )
                if min_threat_dist >= 15.0:  # Waypoint provides safety clearance
                    escape_knot = QuipuKnot(
                        knot_id=f"escape_wp_{wp.waypoint_id}",
                        knot_type=KnotType.SINGLE_KNOT,
                        loops=1.0,
                        position_on_cord=agent_pos.distance_to(wp.position),
                        weight=50.0,
                        label=f"EscapeWaypoint:{wp.waypoint_id}",
                        payload={"waypoint_id": wp.waypoint_id, "position": wp.position.to_dict()},
                    )
                    safe_escape_knots.append(escape_knot)

            if safe_escape_knots:
                escape_sub = SubsidiaryCord(
                    subsidiary_id=f"escape_sub_{self.cycle_count}",
                    parent_knot_id="agent_trunk",
                    subsidiary_type=SubsidiaryType.ESCAPE_PATH,
                    urgency_multiplier=3.0,
                    knots=safe_escape_knots,
                    description=f"Agent critical HP ({round(agent.hp_pct*100)}%) with {len(active_combatants)} combatants",
                )
                threat_cord.subsidiary_cords.append(escape_sub)

        # -------------------------------------------------------------------
        # 3. Weave Affordance & Resource Cord
        # -------------------------------------------------------------------
        affordance_cord = self.trunk.affordance
        for aff in affordances:
            dist = agent_pos.distance_to(aff.position)
            if dist > sensory_radius:
                continue

            has_los = self.check_los(agent_pos, aff.position)
            if aff.is_locked and not aff.has_key_in_bag:
                # Locked door without key: Figure-8 Hard Barrier
                k_type = KnotType.FIGURE_EIGHT
                loops = 1.0
                weight = 0.0  # Cannot yield anything
            elif not has_los:
                k_type = KnotType.LONG_KNOT
                loops = 1.0
                weight = aff.yield_value * 0.5
            else:
                k_type = KnotType.SINGLE_KNOT
                loops = 1.0 if aff.is_lootable else 0.0
                weight = aff.yield_value

            knot = QuipuKnot(
                knot_id=f"affordance_{aff.guid}",
                knot_type=k_type,
                loops=loops,
                position_on_cord=dist,
                weight=weight,
                label=f"Affordance:{aff.name}",
                payload={
                    "guid": aff.guid,
                    "name": aff.name,
                    "affordance_type": aff.affordance_type,
                    "is_locked": aff.is_locked,
                    "is_lootable": aff.is_lootable,
                    "position": aff.position.to_dict(),
                },
            )
            affordance_cord.add_knot(knot)

        self.cycle_count += 1

    # -----------------------------------------------------------------------
    # Phase 2: Graph Tension Evaluation & Utility Optimization
    # -----------------------------------------------------------------------

    def evaluate_tensions(self) -> dict[str, Any]:
        """Calculates system tension across all cords and identifies dominant tension vector."""
        spatial_t = self.trunk.spatial.total_tension()
        threat_t = self.trunk.threat.total_tension()
        affordance_t = self.trunk.affordance.total_tension()
        total_t = self.trunk.total_system_tension()

        # Dominant cord determines primary mental focus
        cord_tensions = {
            "spatial": spatial_t,
            "threat": threat_t,
            "affordance": affordance_t,
        }
        dominant_cord = max(cord_tensions, key=lambda k: cord_tensions[k])

        is_critical = threat_t >= CRITICAL_TENSION_REROUTE or self.trunk.agent.hp_pct <= LOW_HEALTH_SURVIVAL_THRESHOLD

        return {
            "cycle": self.cycle_count,
            "total_tension": round(total_t, 2),
            "threat_tension": round(threat_t, 2),
            "spatial_tension": round(spatial_t, 2),
            "affordance_tension": round(affordance_t, 2),
            "dominant_cord": dominant_cord,
            "is_critical_tension": is_critical,
        }

    def compute_action_utility(
        self,
        candidate_action: TacticalActionType,
        target_knot: Optional[QuipuKnot] = None,
    ) -> float:
        """Utility Function:
        Action Utility = Objective Reward Knot - sum(Threat Knots * Distance Weight) - Resource Depletion Cost
        """
        agent = self.trunk.agent

        # Base objective reward
        reward = 0.0
        if candidate_action == TacticalActionType.RETREAT_ESCAPE:
            # Huge reward if low health
            reward = 120.0 if agent.hp_pct <= LOW_HEALTH_SURVIVAL_THRESHOLD else 20.0
        elif candidate_action == TacticalActionType.SNARE_FLEEING_RUNNER:
            reward = 95.0
        elif candidate_action == TacticalActionType.PULL_TO_LOS:
            reward = 70.0
        elif candidate_action == TacticalActionType.ENGAGE_COMBAT:
            reward = 50.0
        elif candidate_action == TacticalActionType.REST_AND_RECOVER:
            reward = 80.0 if agent.needs_rest() else 5.0
        elif candidate_action == TacticalActionType.HARVEST_AFFORDANCE:
            reward = (target_knot.weight if target_knot else 20.0)
        elif candidate_action == TacticalActionType.TRAVERSE_NAVMESH:
            reward = 25.0

        # Threat penalty sum
        threat_penalty = 0.0
        for knot in self.trunk.threat.knots:
            if not knot.is_active:
                continue
            dist = knot.position_on_cord
            dist_weight = 1.0 / (1.0 + 0.04 * dist)
            threat_penalty += knot.weight * dist_weight

        # Resource depletion cost
        resource_cost = 0.0
        if candidate_action == TacticalActionType.ENGAGE_COMBAT:
            resource_cost = (1.0 - agent.hp_pct) * 20.0 + (1.0 - agent.mp_pct) * 15.0
        elif candidate_action == TacticalActionType.PULL_TO_LOS:
            resource_cost = 5.0  # Cheap ranged spell / shot
        elif candidate_action == TacticalActionType.REST_AND_RECOVER:
            resource_cost = 0.0  # Consumes food/drink but restores resources

        return reward - threat_penalty - resource_cost

    # -----------------------------------------------------------------------
    # Phase 3: Actuation & Knot Pruning
    # -----------------------------------------------------------------------

    def decide_actuation(self) -> TacticalAction:
        """Determines the optimal tactical action based on graph tension and rules."""
        agent = self.trunk.agent
        threat_cord = self.trunk.threat
        spatial_cord = self.trunk.spatial
        affordance_cord = self.trunk.affordance

        # -------------------------------------------------------------------
        # Rule 1: Emergency Escape (Agent critical or overwhelming adds)
        # -------------------------------------------------------------------
        for sub in threat_cord.subsidiary_cords:
            if sub.subsidiary_type == SubsidiaryType.ESCAPE_PATH and sub.knots:
                # Pick the safest waypoint knot
                best_escape_knot = min(sub.knots, key=lambda k: k.position_on_cord)
                pos = Vector3(**best_escape_knot.payload["position"])
                action = TacticalAction(
                    action_type=TacticalActionType.RETREAT_ESCAPE,
                    target_guid=None,
                    target_position=pos,
                    utility_score=self.compute_action_utility(TacticalActionType.RETREAT_ESCAPE, best_escape_knot),
                    explanation=f"Critical escape triggered! Pathing to safe waypoint {best_escape_knot.payload['waypoint_id']}.",
                    associated_knot_id=best_escape_knot.knot_id,
                    subsidiary_cord_id=sub.subsidiary_id,
                )
                self.last_action = action
                return action

        # -------------------------------------------------------------------
        # Rule 2: Snare Runner Leash Prevention
        # -------------------------------------------------------------------
        for sub in threat_cord.subsidiary_cords:
            if sub.subsidiary_type == SubsidiaryType.FLEE_ADD and sub.knots:
                # Find parent fleeing mob
                parent_knot = next((k for k in threat_cord.knots if k.knot_id == sub.parent_knot_id), None)
                if parent_knot and parent_knot.payload.get("guid"):
                    pos = Vector3(**parent_knot.payload["position"])
                    action = TacticalAction(
                        action_type=TacticalActionType.SNARE_FLEEING_RUNNER,
                        target_guid=parent_knot.payload["guid"],
                        target_position=pos,
                        utility_score=self.compute_action_utility(TacticalActionType.SNARE_FLEEING_RUNNER, parent_knot),
                        explanation=f"Fleeing humanoid {parent_knot.label} will leash {len(sub.knots)} adds! Execute Snare/Stun immediately.",
                        associated_knot_id=parent_knot.knot_id,
                        subsidiary_cord_id=sub.subsidiary_id,
                    )
                    self.last_action = action
                    return action

        # -------------------------------------------------------------------
        # Rule 3: Out-of-Combat Resource Attrition (Rest / Drink)
        # -------------------------------------------------------------------
        if agent.needs_rest() and not agent.is_in_combat:
            action = TacticalAction(
                action_type=TacticalActionType.REST_AND_RECOVER,
                target_guid=None,
                target_position=agent.position,
                utility_score=self.compute_action_utility(TacticalActionType.REST_AND_RECOVER),
                explanation=f"Resource attrition downtime mandated: HP {round(agent.hp_pct*100)}%, MP {round(agent.mp_pct*100)}%. Rest in clearance zone.",
            )
            self.last_action = action
            return action

        # -------------------------------------------------------------------
        # Rule 4: Active Combat Engagement / Tactical LoS Pull
        # -------------------------------------------------------------------
        active_threat_knots = [
            k for k in threat_cord.knots
            if k.is_active and k.payload.get("is_combatant") and k.knot_type != KnotType.FIGURE_EIGHT
        ]

        if active_threat_knots:
            # Prioritize current combat target or closest combatant
            target_knot = min(active_threat_knots, key=lambda k: k.position_on_cord)
            pos = Vector3(**target_knot.payload["position"])

            # Check if target is a caster outside melee range and we have a LoS corner
            is_caster = target_knot.payload.get("is_casting", False)
            dist = target_knot.position_on_cord

            if is_caster and dist > 12.0:
                # Find adjacent LoS pillar/occluder to break spellcast
                for wp_knot in spatial_cord.knots:
                    if wp_knot.knot_type == KnotType.FIGURE_EIGHT and wp_knot.position_on_cord < 15.0:
                        los_pos = Vector3(**wp_knot.payload["position"])
                        action = TacticalAction(
                            action_type=TacticalActionType.PULL_TO_LOS,
                            target_guid=target_knot.payload["guid"],
                            target_position=los_pos,
                            utility_score=self.compute_action_utility(TacticalActionType.PULL_TO_LOS, target_knot),
                            explanation=f"Pulling caster {target_knot.label} behind LoS occluder {wp_knot.label} to force melee positioning.",
                            associated_knot_id=target_knot.knot_id,
                        )
                        self.last_action = action
                        return action

            # Normal combat engagement
            action = TacticalAction(
                action_type=TacticalActionType.ENGAGE_COMBAT,
                target_guid=target_knot.payload["guid"],
                target_position=pos,
                utility_score=self.compute_action_utility(TacticalActionType.ENGAGE_COMBAT, target_knot),
                explanation=f"Engaging primary combat target {target_knot.label}.",
                associated_knot_id=target_knot.knot_id,
            )
            self.last_action = action
            return action

        # -------------------------------------------------------------------
        # Rule 5: Affordance Harvest (Quest Chests, Ore, Herbs, Doors)
        # -------------------------------------------------------------------
        active_affordances = [
            k for k in affordance_cord.knots
            if k.is_active and k.knot_type != KnotType.FIGURE_EIGHT and k.payload.get("is_lootable")
        ]

        if active_affordances and not agent.is_in_combat:
            best_affordance = max(
                active_affordances,
                key=lambda k: k.weight / (1.0 + 0.05 * k.position_on_cord),
            )
            pos = Vector3(**best_affordance.payload["position"])
            action = TacticalAction(
                action_type=TacticalActionType.HARVEST_AFFORDANCE,
                target_guid=best_affordance.payload["guid"],
                target_position=pos,
                utility_score=self.compute_action_utility(TacticalActionType.HARVEST_AFFORDANCE, best_affordance),
                explanation=f"Harvesting high-yield affordance {best_affordance.label} (Yield: {best_affordance.weight}).",
                associated_knot_id=best_affordance.knot_id,
            )
            self.last_action = action
            return action

        # -------------------------------------------------------------------
        # Rule 6: NavMesh Traversal & Objective Exploration
        # -------------------------------------------------------------------
        clearance_wps = [
            k for k in spatial_cord.knots
            if k.is_active and k.payload.get("is_clearance_zone") and k.knot_type != KnotType.FIGURE_EIGHT
        ]

        if clearance_wps:
            # Progress forward along lowest tension clearance corridor
            next_wp = min(clearance_wps, key=lambda k: k.effective_tension())
            pos = Vector3(**next_wp.payload["position"])
            action = TacticalAction(
                action_type=TacticalActionType.TRAVERSE_NAVMESH,
                target_guid=None,
                target_position=pos,
                utility_score=self.compute_action_utility(TacticalActionType.TRAVERSE_NAVMESH, next_wp),
                explanation=f"Traversing navmesh corridor via clearance waypoint {next_wp.label}.",
                associated_knot_id=next_wp.knot_id,
            )
            self.last_action = action
            return action

        # Default fallback
        action = TacticalAction(
            action_type=TacticalActionType.HOLD_AND_WAIT,
            target_guid=None,
            target_position=agent.position,
            utility_score=0.0,
            explanation="Holding position in current zone, awaiting patrol clearance.",
        )
        self.last_action = action
        return action

    def prune_knot(self, knot_id: str) -> bool:
        """Unties/prunes a knot upon task completion, relieving graph tension."""
        pruned = False
        for p in self.trunk.pendants.values():
            if p.prune_knot(knot_id):
                pruned = True
        return pruned

    # -----------------------------------------------------------------------
    # Minimal Context Distillation (Quipu Mesh Principle)
    # -----------------------------------------------------------------------

    def select_minimal_context(self) -> dict[str, Any]:
        """Distills the Quipu graph into the smallest connected context subgraph
        satisfying the immediate tactical objective.
        
        Reduces token/computational burden to the immediate interaction horizon:
        Agent Core Vector + Immediate Threat Knot + Immediate Escape Waypoint.
        """
        agent = self.trunk.agent
        active_threats = [k for k in self.trunk.threat.knots if k.is_active and k.position_on_cord <= 35.0]
        active_threats.sort(key=lambda k: k.effective_tension(), reverse=True)

        immediate_threat = active_threats[0].to_dict() if active_threats else None

        # Nearest safe waypoint
        safe_wps = [
            k for k in self.trunk.spatial.knots
            if k.is_active and k.payload.get("is_clearance_zone")
        ]
        safe_wps.sort(key=lambda k: k.position_on_cord)
        nearest_safe_wp = safe_wps[0].to_dict() if safe_wps else None

        subgraph = {
            "agent_core": {
                "level": agent.level,
                "hp_pct": round(agent.hp_pct, 2),
                "mp_pct": round(agent.mp_pct, 2),
                "in_combat": agent.is_in_combat,
                "position": agent.position.to_dict(),
            },
            "immediate_threat_knot": immediate_threat,
            "immediate_safe_knot": nearest_safe_wp,
            "subsidiary_count": sum(len(p.subsidiary_cords) for p in self.trunk.pendants.values()),
            "total_system_tension": round(self.trunk.total_system_tension(), 2),
            "last_action": self.last_action.to_dict() if self.last_action else None,
        }

        # Persist to brain_kv if connected
        if _brain_kv is not None:
            try:
                _brain_kv.kv_set_json("quipu:game:minimal_context", subgraph)
            except Exception as exc:
                logger.debug("Failed to persist game minimal context: %s", exc)

        return subgraph

    # -----------------------------------------------------------------------
    # Helper Math
    # -----------------------------------------------------------------------

    @staticmethod
    def _point_to_segment_dist(p: Vector3, a: Vector3, b: Vector3) -> float:
        """2D distance from point P to line segment AB."""
        abx = b.x - a.x
        aby = b.y - a.y
        ab_len_sq = abx ** 2 + aby ** 2
        if ab_len_sq < 1e-6:
            return p.horizontal_distance_to(a)
        t = max(0.0, min(1.0, ((p.x - a.x) * abx + (p.y - a.y) * aby) / ab_len_sq))
        proj_x = a.x + t * abx
        proj_y = a.y + t * aby
        return math.sqrt((p.x - proj_x) ** 2 + (p.y - proj_y) ** 2)


# ---------------------------------------------------------------------------
# Public Helper API
# ---------------------------------------------------------------------------

def create_quipu_game_engine(
    agent_id: str = "agent_01",
    name: str = "ClassicWarrior",
    level: int = 15,
    character_class: str = "Warrior",
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
) -> QuipuGameMeshEngine:
    """Factory creating an initialized Quipu Game Mesh Autonomy Engine."""
    core = AgentCoreState(
        agent_id=agent_id,
        name=name,
        level=level,
        character_class=character_class,
        position=Vector3(x, y, z),
        heading=0.0,
        hp_pct=1.0,
        mp_pct=1.0,
        bag_free_slots=14,
        bag_capacity=16,
    )
    return QuipuGameMeshEngine(agent_state=core)


__all__ = [
    "KnotType",
    "PendantDomain",
    "SubsidiaryType",
    "MacroIntent",
    "TacticalActionType",
    "Vector3",
    "LoSOccluder",
    "QuipuKnot",
    "SubsidiaryCord",
    "PendantCord",
    "AgentCoreState",
    "PrimaryTrunkCord",
    "WorldEntity",
    "NavMeshWaypoint",
    "WorldAffordance",
    "TacticalAction",
    "QuipuGameMeshEngine",
    "create_quipu_game_engine",
]
