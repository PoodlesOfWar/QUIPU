# QUIPU Gameplay Video & Stream Recordings Ingestion Directory

This directory is monitored continuously by the `quipu-video-trainer` container daemon.

### Supported File Formats:
- **Video Streams**: `.mp4`, `.mkv`, `.webm`, `.avi`
- **Speech Transcripts**: Corresponding `.txt` files with matching filename stem (e.g., `clip_01.mp4` paired with `clip_01.txt`).

### Pipeline Behavior:
1. Video files dropped here are sampled at 2 FPS by the `HUDVisionExtractor`.
2. Unit frame vitals (HP, Mana/Energy/Action Points, In-Combat status) and minimap radar blips are extracted.
3. Streamer spoken commentary is mapped to strategic intent cues (`engage_pull`, `flee_retreat`, `heal_support`, `wait_ooc`, `loot_gather`).
4. Demonstrations are fed into the Inverse Graph Tension Learning (IGTL) quasi-Newton L-BFGS-B optimizer to calibrate Quipu knot tension weights.
5. In the absence of offline video files, the replay engine continuously synthesizes stream batches across World of Warcraft, Old School RuneScape, and Guild Wars to maintain active potentiating loops.
