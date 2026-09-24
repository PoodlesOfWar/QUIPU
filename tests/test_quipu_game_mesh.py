"""Tests for the QUIPU Game Mesh Autonomy Engine (Classic MMORPG / WoW: Forever)."""

import math
import pytest

from quipu.quipu_game_mesh import (
    AgentCoreState,
    KnotType,
    LoSOccluder,
    MacroIntent,
    NavMeshWaypoint,
    PendantDomain,
    QuipuGameMeshEngine,
    QuipuKnot,
    SubsidiaryType,
    TacticalActionType,
    Vector3,
    WorldAffordance,
    WorldEntity,
    create_quipu_game_engine,
)


def test_knot_effective_tension():
    """Verify knot tensions for Figure-8, Long Knot, and Single Knot."""
    # Figure-8 hard barrier at 10 yards
    fig8 = QuipuKnot(
        knot_id="k1",
        knot_type=KnotType.FIGURE_EIGHT,
        loops=3.0,
        position_on_cord=10.0,
        weight=100.0,
        label="Skull Mob",
    )
    # Long knot with 5 loops at 10 yards
    long_k = QuipuKnot(
        knot_id="k2",
        knot_type=KnotType.LONG_KNOT,
        loops=5.0,
        position_on_cord=10.0,
        weight=50.0,
        label="Combatant Mob",
    )
    # Single knot (active = 1 loop) at 10 yards
    single_k = QuipuKnot(
        knot_id="k3",
        knot_type=KnotType.SINGLE_KNOT,
        loops=1.0,
        position_on_cord=10.0,
        weight=20.0,
        label="Lootable Node",
    )

    t_fig8 = fig8.effective_tension()
    t_long = long_k.effective_tension()
    t_single = single_k.effective_tension()

    # Figure 8 should carry greatest tension due to hard barrier penalty
    assert t_fig8 > t_long > t_single > 0.0

    # Pruned / inactive knot exerts zero tension
    fig8.is_active = False
    assert fig8.effective_tension() == 0.0


def test_level_delta_aggro_scaling_and_skull_mob():
    """Test Classic level delta aggro radius scaling and Figure-8 knot generation."""
    engine = create_quipu_game_engine(level=15, x=0.0, y=0.0, z=0.0)

    # 1. Equal level mob (Level 15)
    mob_equal = WorldEntity(
        guid="mob_equal",
        name="Defias Rogue",
        position=Vector3(18.0, 0.0, 0.0),
        level=15,
        health_pct=1.0,
        max_health=400,
        is_hostile=True,
        is_friendly=False,
        is_humanoid=True,
        is_combatant=False,
        is_casting=False,
        is_interruptible=False,
        is_stunned=False,
        is_snared=False,
    )

    # 2. Skull mob (+4 levels: Level 19)
    mob_skull = WorldEntity(
        guid="mob_skull",
        name="Defias Taskmaster",
        position=Vector3(25.0, 0.0, 0.0),
        level=19,
        health_pct=1.0,
        max_health=700,
        is_hostile=True,
        is_friendly=False,
        is_humanoid=True,
        is_combatant=False,
        is_casting=False,
        is_interruptible=False,
        is_stunned=False,
        is_snared=False,
    )

    engine.weave_perceptions(entities=[mob_equal, mob_skull], nav_waypoints=[], affordances=[])

    threat_knots = engine.trunk.threat.knots
    assert len(threat_knots) == 2

    skull_knot = next(k for k in threat_knots if k.payload["guid"] == "mob_skull")
    equal_knot = next(k for k in threat_knots if k.payload["guid"] == "mob_equal")

    # Skull mob must be classified as Figure-8 hard barrier
    assert skull_knot.knot_type == KnotType.FIGURE_EIGHT
    assert skull_knot.payload["delta_level"] == 4
    # Aggro radius for equal mob should be base 20.0 yds
    assert equal_knot.payload["aggro_radius"] == 20.0
    # Aggro radius for +4 mob should expand: 20 + 4*1.5 = 26.0 yds
    assert skull_knot.payload["aggro_radius"] == 26.0


def test_humanoid_runner_sprouts_flee_add_subsidiary():
    """Test that a humanoid mob at <= 20% health fleeing toward unpulled hostiles sprouts a Flee / Add subsidiary cord."""
    engine = create_quipu_game_engine(level=15, x=0.0, y=0.0, z=0.0)

    # Active combatant fleeing directly along +X axis
    runner_mob = WorldEntity(
        guid="runner_01",
        name="Defias Highwayman",
        position=Vector3(10.0, 0.0, 0.0),
        level=15,
        health_pct=0.15,  # <= 20% triggers fleeing
        max_health=500,
        is_hostile=True,
        is_friendly=False,
        is_humanoid=True,
        is_combatant=True,
        is_casting=False,
        is_interruptible=False,
        is_stunned=False,
        is_snared=False,
    )

    # An unpulled hostile mob sitting in the projected flee corridor (x=25, y=2)
    unpulled_add = WorldEntity(
        guid="unpulled_add_01",
        name="Defias Scout",
        position=Vector3(25.0, 2.0, 0.0),
        level=15,
        health_pct=1.0,
        max_health=450,
        is_hostile=True,
        is_friendly=False,
        is_humanoid=True,
        is_combatant=False,  # Not yet in combat
        is_casting=False,
        is_interruptible=False,
        is_stunned=False,
        is_snared=False,
    )

    engine.weave_perceptions(
        entities=[runner_mob, unpulled_add],
        nav_waypoints=[],
        affordances=[],
    )

    threat_cord = engine.trunk.threat
    subs = threat_cord.subsidiary_cords

    # Subsidiary cord should sprout
    assert len(subs) == 1
    flee_sub = subs[0]
    assert flee_sub.subsidiary_type == SubsidiaryType.FLEE_ADD
    assert flee_sub.urgency_multiplier == 2.5
    assert len(flee_sub.knots) == 1
    assert flee_sub.knots[0].payload["unpulled_add_guid"] == "unpulled_add_01"

    # Action decision should prioritize snaring or bursting the runner
    action = engine.decide_actuation()
    assert action.action_type == TacticalActionType.SNARE_FLEEING_RUNNER
    assert action.target_guid == "runner_01"


def test_critical_health_triggers_escape_path_subsidiary():
    """Verify that when agent health drops below threshold, an Escape Path subsidiary cord forms and directs retreat."""
    engine = create_quipu_game_engine(level=15, x=0.0, y=0.0, z=0.0)
    engine.trunk.agent.hp_pct = 0.18  # Critical HP (< 25%)
    engine.trunk.agent.is_in_combat = True

    combatant = WorldEntity(
        guid="elite_mob",
        name="Defias Blackguard",
        position=Vector3(3.0, 0.0, 0.0),
        level=16,
        health_pct=0.8,
        max_health=800,
        is_hostile=True,
        is_friendly=False,
        is_humanoid=True,
        is_combatant=True,
        is_casting=False,
        is_interruptible=False,
        is_stunned=False,
        is_snared=False,
    )

    # Safe escape waypoint far behind player
    safe_wp = NavMeshWaypoint(
        waypoint_id="wp_safe_exit",
        position=Vector3(-30.0, 0.0, 0.0),
        is_clearance_zone=True,
        is_safe_escape_node=True,
    )

    engine.weave_perceptions(
        entities=[combatant],
        nav_waypoints=[safe_wp],
        affordances=[],
    )

    threat_cord = engine.trunk.threat
    subs = threat_cord.subsidiary_cords
    escape_subs = [s for s in subs if s.subsidiary_type == SubsidiaryType.ESCAPE_PATH]

    assert len(escape_subs) == 1
    action = engine.decide_actuation()
    assert action.action_type == TacticalActionType.RETREAT_ESCAPE
    assert action.target_position.x == -30.0


def test_resource_attrition_rest_and_recover():
    """Verify that out-of-combat health/mana depletion triggers Rest and Recover action."""
    engine = create_quipu_game_engine(level=15, x=0.0, y=0.0, z=0.0)
    engine.trunk.agent.is_in_combat = False
    engine.trunk.agent.hp_pct = 0.30  # Below 40% rest trigger
    engine.trunk.agent.mp_pct = 0.20  # Below 30% rest trigger

    engine.weave_perceptions(entities=[], nav_waypoints=[], affordances=[])

    action = engine.decide_actuation()
    assert action.action_type == TacticalActionType.REST_AND_RECOVER
    assert "downtime mandated" in action.explanation


def test_pull_to_los_against_distant_caster():
    """Verify that a distant casting mob paired with a nearby LoS pillar triggers Pull to LoS."""
    engine = create_quipu_game_engine(level=15, x=0.0, y=0.0, z=0.0)

    # Caster mob 20 yards away
    caster = WorldEntity(
        guid="caster_01",
        name="Defias Firemage",
        position=Vector3(20.0, 0.0, 0.0),
        level=15,
        health_pct=1.0,
        max_health=350,
        is_hostile=True,
        is_friendly=False,
        is_humanoid=True,
        is_combatant=True,
        is_casting=True,
        is_interruptible=True,
        is_stunned=False,
        is_snared=False,
    )

    # LoS occluder pillar at (5, 0)
    pillar = LoSOccluder(id="pillar_01", p1=Vector3(5.0, 0.0, 0.0), p2=Vector3(0.0, 0.0, 0.0), is_pillar=True, radius=2.0)
    engine.register_los_occluder(pillar)

    # Choke/occluded NavMesh waypoint behind pillar
    wp_los = NavMeshWaypoint(
        waypoint_id="los_corner",
        position=Vector3(5.0, 4.0, 0.0),
        is_clearance_zone=True,
        is_choke_point=True,
    )

    engine.weave_perceptions(
        entities=[caster],
        nav_waypoints=[wp_los],
        affordances=[],
    )

    action = engine.decide_actuation()
    assert action.action_type == TacticalActionType.PULL_TO_LOS
    assert action.target_guid == "caster_01"


def test_affordance_harvest_and_knot_pruning():
    """Verify that out of combat with no immediate threats, highest-yield affordance is harvested and pruned."""
    engine = create_quipu_game_engine(level=15, x=0.0, y=0.0, z=0.0)

    chest = WorldAffordance(
        guid="chest_01",
        name="Battered Chest",
        position=Vector3(8.0, 0.0, 0.0),
        affordance_type="quest_chest",
        is_locked=False,
        is_lootable=True,
        yield_value=40.0,
    )

    herb = WorldAffordance(
        guid="herb_01",
        name="Peacebloom",
        position=Vector3(15.0, 5.0, 0.0),
        affordance_type="herb",
        is_locked=False,
        is_lootable=True,
        yield_value=15.0,
    )

    engine.weave_perceptions(entities=[], nav_waypoints=[], affordances=[chest, herb])

    action = engine.decide_actuation()
    assert action.action_type == TacticalActionType.HARVEST_AFFORDANCE
    assert action.target_guid == "chest_01"

    # Now simulate task completion and pruning of the knot
    pruned = engine.prune_knot(action.associated_knot_id)
    assert pruned is True

    # Tension should drop and remaining affordance becomes next target
    action_next = engine.decide_actuation()
    assert action_next.target_guid == "herb_01"


def test_minimal_context_distillation():
    """Verify that Quipu Minimal Context correctly distills the graph down to immediate essentials."""
    engine = create_quipu_game_engine(level=15, x=0.0, y=0.0, z=0.0)

    mob = WorldEntity(
        guid="mob_01",
        name="Riverpaw Gnoll",
        position=Vector3(12.0, 0.0, 0.0),
        level=15,
        health_pct=0.9,
        max_health=400,
        is_hostile=True,
        is_friendly=False,
        is_humanoid=True,
        is_combatant=True,
        is_casting=False,
        is_interruptible=False,
        is_stunned=False,
        is_snared=False,
    )

    wp = NavMeshWaypoint(
        waypoint_id="wp_clear_1",
        position=Vector3(5.0, 0.0, 0.0),
        is_clearance_zone=True,
    )

    engine.weave_perceptions(entities=[mob], nav_waypoints=[wp], affordances=[])
    _ = engine.decide_actuation()

    minimal = engine.select_minimal_context()

    assert "agent_core" in minimal
    assert minimal["agent_core"]["level"] == 15
    assert minimal["immediate_threat_knot"] is not None
    assert minimal["immediate_threat_knot"]["label"] == "Threat:Riverpaw Gnoll(Lvl15)"
    assert minimal["immediate_safe_knot"] is not None
    assert minimal["immediate_safe_knot"]["label"].startswith("NavWP:wp_clear_1")
    assert minimal["total_system_tension"] > 0.0
