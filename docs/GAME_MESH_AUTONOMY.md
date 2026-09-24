# QUIPU Game Mesh Autonomy & Video Pre-Training Pipeline

## 1. Executive Summary

The **QUIPU Game Mesh** implements a hierarchical, Andean-inspired topological graph model for autonomous game perception, threat geometry, and real-time decision-making in classic, ground-bound virtual worlds (e.g. *World of Warcraft: Forever*, *Old School RuneScape*, and *Guild Wars*).

To solve the cold-start problem and teach the agent authentic human reflexes and social etiquette without triggering live anti-cheat systems, the **Video Pre-Training Pipeline** extracts perceptual features directly from Twitch/YouTube video recordings and optimizes knot tension weights via **Inverse Graph Tension Learning (IGTL)**.

---

## 2. Graph Architecture

```
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
```

### 2.1 The Four Quipu Cords
1. **Primary Trunk Cord**: Agent vitals ($HP\%$, $MP\%$), stamina, active auras, inventory constraints, and macro-intent.
2. **Spatial & Topological Pendant**: NavMesh centroids, elevation gradient, clearance zones, choke points, and Line-of-Sight (LoS) occluders.
3. **Threat & Entity Pendant**: Hostiles, patrols, casting status, target facing angle, and level delta scaling.
4. **Affordance & Resource Pendant**: Mining/herb nodes, chests, doors, flight masters.
5. **Social Dynamics Pendant**: Other players, crowd density, etiquette barriers (avoiding kill stealing).

### 2.2 Knot Structural Complexity
- **Figure-8 Knot (Hard Barrier)**: LoS pillars, locked gates, skull danger mobs ($\ge 3$ levels higher).
- **Long Knot with $N$ Loops (Scalar Intensity)**: Incoming DPS, remaining mob health %, aggro threat.
- **Single Knot (Binary State)**: 1 loop = True / 0 = False (stunned, interruptible, lootable).

### 2.3 Emergent Subsidiary Cords
- **"Flee / Add" Subsidiary**: Sprouted when a humanoid combatant drops to $\le 20\%$ HP. The mob's forward flee vector ($25\text{ yds}$) is raycast against adjacent unpulled hostile packs ($R_{\text{social}} \le 8\text{ yds}$). If collision occurs, high-urgency snare/stun knots trigger immediate actuation.
- **"Escape Path" Subsidiary**: Sprouted when agent HP drops below $25\%$ or multi-pack adds exceed throughput. Navigates along safe clearance corridors to reset boundaries.
- **"Rest / Recovery" Subsidiary**: Out-of-combat downtime eating/drinking until reaching safe threshold ($\ge 85\%$).

---

## 3. Video Pre-Training & Inverse Graph Tension Learning (IGTL)

```
                       [ RAW VIDEO & STREAM STREAM ]
                (Twitch / YouTube: 1080p60 Video + Audio)
                                   |
            +----------------------+----------------------+
            |                                             |
            v                                             v
   [ COMPUTER VISION / HUD ]                     [ AUDIO TRANSCRIPTION ]
 (Spatial Depth, Entity Detection,                 (Whisper / STT:
   Minimap, HP/MP Bar OCR)                    Streamer Rationale & Chat)
            |                                             |
            +----------------------+----------------------+
                                   v
             [ PERCEPTION PARSER -> WEAVE QUIPU GRAPH ]
           +---------------------------------------------+
           | - Trunk Cord: HP/MP/Buffs (from HUD OCR)    |
           | - Spatial Cord: NavMesh Path (from Viewport)|
           | - Threat Cord: Mobs, Levels, Cast Bars      |
           | - Social Cord: Adjacent Players, Chat Box   |
           +---------------------------------------------+
                                   |
                                   v
             [ INVERSE GRAPH TENSION OPTIMIZATION (IGTL) ]
            Fits Knot Tension Weights & Utility Functions:
            "What tension configuration caused the human
             to pull behind the LoS pillar right now?"
```

Given human expert demonstrations $D = \{(s_i, a_i^*)\}$, the optimizer minimizes the Negative Log-Likelihood (NLL) of the human actions under the Quipu Softmax policy using quasi-Newton L-BFGS-B:

$$\mathcal{L}(\vec{\theta}) = -\frac{1}{N} \sum_{i=1}^N \log \left( \frac{\exp(\beta \cdot U(a_i^*; \vec{\theta}))}{\sum_{a \in \mathcal{A}} \exp(\beta \cdot U(a; \vec{\theta}))} \right) + \lambda \|\vec{\theta} - \vec{\theta}_0\|_2^2$$

---

## 4. Containerized Microservice

The engine runs as an active Docker container exposing port `7200`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Container liveness, uptime, and last training accuracy |
| `GET` | `/parameters` | Active calibrated Quipu tension parameters |
| `POST` | `/train` | Triggers an on-demand training cycle from demonstrations |
| `POST` | `/tick` | Real-time state evaluation returning Quipu tactical actuation |

### Quickstart Running in Docker
```powershell
# Build and run container
docker compose up -d quipu-game-pipeline

# Check health
curl http://127.0.0.1:7200/health

# Trigger real-time tactical tick
curl -X POST http://127.0.0.1:7200/tick -H "Content-Type: application/json" -d '{"agent_core": {"hp_pct": 0.18, "is_in_combat": true}, "entities": [{"guid": "m1", "name": "Defias Rogue", "level": 17, "health_pct": 0.9, "is_hostile": true, "is_combatant": true}]}'
```
