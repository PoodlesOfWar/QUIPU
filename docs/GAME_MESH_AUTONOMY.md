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

## 4. Multi-Game Human Behavioral Fidelity Assessor

To determine if the autonomous agent's actions pass as authentic active human players across Old School RuneScape (OSRS), Guild Wars (GW), and World of Warcraft (WoW), `human_fidelity_assessor.py` performs periodic evaluations across four empirical dimensions:

1. **Cadence & Reaction Latency Variance**: Tests for natural ex-Gaussian distributions with long right-skew tails; flags rigid, flat, or zero-variance reaction times.
2. **Spatial Wander & Curvature**: Measures non-zero geodesic path wander; flags straight-line machine pathing.
3. **Social Etiquette & Bubble Buffers**: Tracks personal space distances and flags anti-social behaviors (e.g. mob poaching / kill-stealing).
4. **Attrition & Hesitation Pacing**: Measures natural hesitation pauses before eating, drinking, or engaging.

Reports output a categorical verdict: `PASS` ($\ge 85\%$), `BORDERLINE` ($70\text{--}84\%$), or `SUSPICIOUS_BOT_PATTERN` ($< 70\%$).

---

## 5. Containerized Microservice

The engine runs as an active Docker container exposing port `7200`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Container liveness, uptime, and last training accuracy |
| `GET` | `/parameters` | Active calibrated Quipu tension parameters |
| `GET` | `/assessments` | Live multi-game fidelity assessment reports (WoW, OSRS, GW) |
| `POST` | `/assess` | Triggers an on-demand fidelity check for a specific game |
| `POST` | `/train` | Triggers an on-demand training cycle from demonstrations |
| `POST` | `/tick` | Real-time state evaluation returning Quipu tactical actuation |

### Quickstart Running in Docker
```powershell
# Build and run container
docker compose up -d quipu-game-pipeline

# Check health
curl http://127.0.0.1:7200/health

# Check multi-game fidelity assessments
curl http://127.0.0.1:7200/assessments

# Trigger real-time tactical tick
curl -X POST http://127.0.0.1:7200/tick -H "Content-Type: application/json" -d '{"agent_core": {"hp_pct": 0.18, "is_in_combat": true}, "entities": [{"guid": "m1", "name": "Defias Rogue", "level": 17, "health_pct": 0.9, "is_hostile": true, "is_combatant": true}]}'
```

---

## 6. Continuous Video & Stream Recording Trainer Container

The **QUIPU Video Trainer Daemon** runs as an active, continuous background worker (`quipu-video-trainer:dev`) exposing port `7250`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/status` | Active training round, current game, total frames processed, last loss, top-1 accuracy, and human fidelity score |
| `GET` | `/metrics` | Prometheus-formatted metrics (`quipu_training_round`, `quipu_total_samples`, `quipu_top1_accuracy`, `quipu_fidelity_score`) |
| `POST` | `/queue` | Manually queues or flags an uploaded video recording for priority processing |

### 6.1 Recording Ingestion & Replay Synthesis
- **Directory Watcher**: Continuously monitors `/app/recordings` for `.mp4`, `.mkv`, `.webm`, and `.avi` files alongside paired `.txt` transcripts.
- **Vision & Speech Processing**: Samples frames at 2 FPS, passes images to `HUDVisionExtractor` for HP/MP and minimap radar blip extraction, and maps speech cues into tactical action intents.
- **Continuous Multi-Game Replay**: If no offline video files are present, automatically generates synthetic stream replay batches rotating across World of Warcraft, Old School RuneScape, and Guild Wars.
- **Tension Optimization & Fidelity Benchmarking**: Runs quasi-Newton L-BFGS-B IGTL optimization, benchmarks each round via `PeriodicFidelityAssessor`, and persists learned parameters to `/app/quipu_learned_artifacts/quipu_learned_parameters.json`.

### 6.2 Running the Trainer Container
```powershell
# Build and run the video trainer container
docker compose up -d quipu-video-trainer

# Check live trainer status
curl http://127.0.0.1:7250/status

# Check Prometheus metrics
curl http://127.0.0.1:7250/metrics
```

---

## 7. Stream Corpus Scanner, GARD Shard Packaging & Vector Graph Integration

### 7.1 High-Volume Streaming Corpus Ingestion
The **Stream Corpus Scanner** (`src/quipu/stream_corpus_scanner.py`) scans and generates 18 rich streaming sessions across Twitch and YouTube, generating 60+ files in `recordings/` across World of Warcraft, Old School RuneScape, and Guild Wars:
- **Dungeons & Raids**: Scarlet Monastery, Molten Core, Theatre of Blood, Blackrock Depths.
- **Bossing & Hardcore**: Zulrah rotation switching, TzTok-Jad prayer flicking, Elwynn Fargodeep mine survival.
- **Wilderness & PvP**: Stranglethorn Vale ganking, Deep Wilderness Revenant Caves multi-PK escape, Warsong Gulch flag carrying.
- **Guild Wars Campaigns**: Fissure of Woe Menzies temple defense, Underworld reaper escort, Heroes' Ascent 8v8, Droknar's Forge mountain traverse, Pre-Searing Ascalon North Gate exploration.

### 7.2 GARD Shard Compression & Lossless Decompression (`gard-shard/v2`)
Every demonstration session is sealed and authenticated into a `.gard.json` / `.gard.store` container using the existing `gard_shard_model.py` architecture:
- **Compression**: zlib level 6 with canonical deterministic JSON formatting.
- **AEAD Encryption**: AES-256-GCM authenticated encryption with 12-byte random nonce and 16-byte MAC tag bound to domain `SiCi_SQRT(-1)`.
- **Zero Data Dissociation**: Lossless bit-for-bit reconstruction via `decrypt_json()` with constant-time AEAD verification.
- **Holographic Compaction**: Computes Page-curve Hawking information remnant score and compaction ratio via `ueqgm_engine.weyl_scalar_tensor()`.

### 7.3 Toroidal Quipu Vector Graph Integration (`mesh_slm`)
Session concept-dense tokens and tactical action tuples are projected directly into `mesh_slm` (`local_brain.sqlite`):
1. **Corpus Ingestion**: Tokens are appended to `mesh_corpus_feed` under platform/game source tags.
2. **7-D Embedding Updates**: Token positions in `mesh_slm_embed` are adjusted across the seven sense axes (`vision`, `touch`, `smell`, `body`, `brain`, `perception`, `entirety`).
3. **Directed Hebbian GNN Quipu Edges**: Bigram transitions update message-passing channel weights in `mesh_slm_quipu` ($w \leftarrow w + \eta_q \cdot (1 - w)$).
4. **Co-Potentiation**: The containerized `video_trainer_daemon` automatically decompresses GARD Shard containers and executes online `train_round()` cycles, keeping the video agent and the vector graph co-potentiating in real time.

---

## 8. Dedicated Game Client Containers & Automated Asset Downloaders

Each of the three games runs within an isolated container environment equipped with virtual framebuffers, compatibility layers (Wine / OpenJDK), and dedicated HTTP daemons that automatically fetch official game assets and manage execution state.

### 8.1 Multi-Game Asset Architecture

```
                    +------------------------------------+
                    |  QUIPU Multi-Game Asset Downloader |
                    | (src/quipu/game_asset_downloader)  |
                    +-----------------+------------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
+-----------------------+     +-----------------------+     +-----------------------+
|  quipu-game-osrs      |     |  quipu-game-wow       |     |  quipu-game-gw        |
|  (Port 7310)          |     |  (Port 7320)          |     |  (Port 7330)          |
|  Debian + OpenJDK 21  |     |  Debian + Wine + Xvfb |     |  Debian + Wine + Xvfb |
|  RuneLite.jar (2.5MB) |     |  WoW 1.12.1 + MPQs    |     |  GwSetup.exe (5.66MB) |
|  Volume: osrs_game_data|    |  Volume: wow_game_data|    |  Volume: gw_game_data |
+-----------------------+     +-----------------------+     +-----------------------+
```

### 8.2 Client Specifications & Asset Provenance
1. **Old School RuneScape (`quipu-game-osrs`, Port 7310)**:
   - **Engine**: Headless OpenJDK 21 with Xvfb virtual display (`:99`).
   - **Binary Source**: Official GitHub Releases `https://github.com/runelite/launcher/releases/download/2.7.3/RuneLite.jar` (2,499,996 bytes).
   - **Verification**: SHA-256 integrity hash verification (`d22a33eaa43b2f859a76824f3352cf3051bf1ff60b4fc405455ecbe8717bb827`).
   - **Configuration**: Standard `runelite.properties` and `/games/osrs/launch_osrs.sh` wrapper.
   - **Volume Mount**: Named volume `osrs_game_data` mounted at `/games/osrs`.

2. **World of Warcraft 1.12.1 Classic (`quipu-game-wow`, Port 7320)**:
   - **Engine**: Wine x86_64 environment with Xvfb display (`:99`).
   - **Client Layout**: Standardized 1.12.1 Classic client directory tree with `WoW.exe`, `realmlist.wtf` pointing to private community realm (`logon.turtle-wow.org`), and `WTF/Config.wtf`.
   - **Data Archives**: Initialized standard MPQ archives (`patch.mpq`, `dbc.mpq`, `terrain.mpq`, `wmo.mpq`).
   - **Volume Mount**: Named volume `wow_game_data` mounted at `/games/wow`.

3. **Guild Wars 1 (`quipu-game-gw`, Port 7330)**:
   - **Engine**: Wine x86_64 environment with Xvfb display (`:99`).
   - **Binary Source**: Official ArenaNet CloudFront CDN `https://cloudfront.guildwars2.com/client/GwSetup.exe` (5,660,840 bytes).
   - **Verification**: SHA-256 integrity verification (`9152740c834eef57db881b51b46149e0bedebc65deef0900050abd0c41ce1c4b`).
   - **Execution**: Installs `Gw.exe` and generates Wine launch script `/games/gw/launch_gw.sh`.
   - **Volume Mount**: Named volume `gw_game_data` mounted at `/games/gw`.

### 8.3 Container HTTP Endpoints
Each container daemon exposes standardized HTTP management routes:
- `GET /status`: Inspects directory manifest, asset file sizes, client status (`idle` or `running`), session ID, VPN attachment, and active player state telemetry.
- `GET /health`: Healthcheck probe for Docker Compose orchestrator.
- `GET /session`: Inspects active authenticated session and VPN attachment.
- `POST /login`: Authenticates an account into the container environment.
- `POST /logout`: Terminates active session.
- `POST /challenge`: Sets challenge status requiring human confirmation.
- `POST /resolve_challenge`: Clears challenge upon Gate 6 User Attestation.
- `POST /download`: Explicitly triggers fresh asset download with hash verification.
- `POST /launch`: Initiates headless client execution on display `:99`.

---

## 9. Mass Session Handler, Tailscale Mesh VPN & Physical Gate (Gate 6) User Integration

```
                         [ r-ADMIN CONTROLLER ]
               (rADAM: Complex Gradient & Pressure Tensor z)
                                  |
                                  v
                  [ MASS SESSION HANDLER (src/quipu/games) ]
                                  |
       +--------------------------+--------------------------+
       |                          |                          |
       v                          v                          v
[MULTI-ACCOUNT STORE]     [TAILSCALE MESH VPN]      [PHYSICAL GATE 6 INTERLOCK]
(AES-256-GCM Vault)       (100.64.0.0/10 CGNAT)     (Two-Party Human Attestation)
       |                          |                          |
       +--------------------------+--------------------------+
                                  |
            +---------------------+---------------------+
            |                     |                     |
            v                     v                     v
   [quipu-game-osrs]     [quipu-game-wow]      [quipu-game-gw]
   (Port 7310 / .10)     (Port 7320 / .20)     (Port 7330 / .30)
```

### 9.1 Multi-Account Credential Store (`src/quipu/games/account_store.py`)
- **Confidentiality & Integrity**: Credentials encrypted using AES-256-GCM with 12-byte random nonces and authenticated tag bindings (`QUIPU_ACCOUNT_STORE_KEY`).
- **Rotation Engine**: Automated round-robin selection among idle accounts per game, tracking last-login timestamps and cooldown intervals.
- **Tagging & Partitioning**: Manages profiles across roles (e.g. `main`, `tank`, `dps`, `pvp_pure`, `skilling`), realm targets, and VPN exit nodes.

### 9.2 Tailscale-Style Mesh VPN Overlay (`src/quipu/games/vpn_mesh_integration.py`)
- **CGNAT Overlay Topology**:
  - `quipu-r-admin`: `100.64.0.5` (`radmin.quipu.mesh`)
  - `quipu-game-osrs`: `100.64.0.10` (`osrs.quipu.mesh`)
  - `quipu-game-wow`: `100.64.0.20` (`wow.quipu.mesh`)
  - `quipu-game-gw`: `100.64.0.30` (`gw.quipu.mesh`)
  - `quipu-video-trainer`: `100.64.0.40` (`trainer.quipu.mesh`)
- **Encrypted Peer WireGuard Routing**: End-to-end encrypted packet delivery without open public ports.
- **DERP Relays & Exit Nodes**: Configured regionally (`nyc`, `ord`, `fra`, `dal`) with dedicated egress nodes (`exit-us-east`, `exit-eu-west`, `exit-us-central`).

### 9.3 Physical Gate (Gate 6) Level of User Integration (`src/quipu/games/gate6_user_interlock.py`)
Rooted in UEQGM v0.9.25 Physical Gate 6 (*Beautiful Output*):
- **Universal Breakage Interlock**: When an autonomous agent encounters a captcha, 2FA prompt, dead-end navigation obstacle, or when Ring 5 refinement/world model encounters an epistemic rupture:
  1. The affected container/refinement loop immediately halts and is held in band $\text{翈}$ (`HELD_AT_GATE_6`).
  2. A `RefinementBreakageEvent` is published to the `brain_kv` bus (`governance:gate6_interlocks`).
  3. r-ADMIN detects the hold and suspends automated write gradients on that coordinate.
- **Human User Confirmation**:
  - The operator inspects the challenge via CLI or API.
  - The operator confirms the right path or provides credentials.
  - Generates an authenticated two-party `Attestation(signer=operator_id, love_form="repair", beautiful_output=True, assurance="approved")`.
  - Evaluates Gate 6 via `governance.gate_beautiful_output`. Once passed, the hold is released (`RESOLVED_BY_USER`), notifying the game daemon or refinement loop to resume on the user-verified trajectory.

### 9.4 Common Refinement Protocol
This Gate 6 User Integration serves as the unified confirmation interface across the entire QUIPU system:
- **Game Autonomy**: Re-routes bot paths, confirms visual captchas, provisions multi-account inputs.
- **Systemic Refinement (`src/quipu/systemic_refinement_agent.py`)**: Traps specialist emergence anomalies and tool forging divergences.
- **World Model (`src/quipu/world_model.py`)**: Intercepts high-surprise epistemic rupture transitions.



