"""QUIPU Game Mesh & Video Pipeline Daemon Service.

Runs an active HTTP service exposing:
- GET  /health      -> Health status, container uptime, last training metrics
- GET  /parameters  -> Current calibrated Quipu tension parameters
- POST /train       -> Triggers an on-demand training cycle from demonstrations
- POST /tick        -> Real-time state evaluation returning Quipu tactical actuation
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Optional

from .quipu_game_mesh import (
    AgentCoreState,
    QuipuGameMeshEngine,
    TacticalActionType,
    Vector3,
    WorldAffordance,
    WorldEntity,
    create_quipu_game_engine,
)
from .video_gameplay_pipeline import (
    QuipuLearnedParameters,
    VideoGameplayPipeline,
)

logger = logging.getLogger("quipu.game_pipeline_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PORT = int(os.environ.get("GAME_PIPELINE_PORT", "7200"))
_HOST = os.environ.get("GAME_PIPELINE_HOST", "0.0.0.0")
_ARTIFACT_DIR = os.environ.get("QUIPU_ARTIFACT_DIR", "/app/quipu_learned_artifacts")

_PIPELINE: Optional[VideoGameplayPipeline] = None
_LATEST_PARAMS: Optional[QuipuLearnedParameters] = None
_LAST_TRAIN_STATS: dict[str, Any] = {}
_SERVICE_START_TIME: float = time.time()
_ENGINE: Optional[QuipuGameMeshEngine] = None


def init_service() -> None:
    """Initializes the pipeline, trains baseline parameters, and mounts the active engine."""
    global _PIPELINE, _LATEST_PARAMS, _LAST_TRAIN_STATS, _ENGINE
    logger.info("Initializing QUIPU Game Pipeline Service...")
    _PIPELINE = VideoGameplayPipeline(output_dir=_ARTIFACT_DIR)
    _LATEST_PARAMS, _LAST_TRAIN_STATS = _PIPELINE.run_training_pipeline(epochs=25)
    _ENGINE = create_quipu_game_engine(name="QuipuAutonomousAgent", level=15, character_class="Warrior")
    logger.info("Service initialized. Baseline training Top-1 Accuracy: %.1f%%", _LAST_TRAIN_STATS["final_accuracy"] * 100)


class GamePipelineHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the containerized game pipeline."""

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond_json(200, {
                "status": "healthy",
                "service": "quipu-game-pipeline",
                "uptime_seconds": round(time.time() - _SERVICE_START_TIME, 1),
                "last_train_stats": _LAST_TRAIN_STATS,
            })
        elif self.path == "/parameters":
            params_dict = _LATEST_PARAMS.to_dict() if _LATEST_PARAMS else {}
            self._respond_json(200, {
                "status": "ok",
                "parameters": params_dict,
            })
        else:
            self._respond_json(404, {"error": "Not Found", "valid_endpoints": ["/health", "/parameters", "/train", "/tick"]})

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        if self.path == "/train":
            epochs = int(payload.get("epochs", 30))
            demos = int(payload.get("demos", 50))
            logger.info("Received /train trigger: epochs=%d, demos=%d", epochs, demos)
            assert _PIPELINE is not None
            demo_steps = _PIPELINE.build_synthetic_demonstrations(count=demos)
            params, stats = _PIPELINE.run_training_pipeline(demonstrations=demo_steps, epochs=epochs)
            global _LATEST_PARAMS, _LAST_TRAIN_STATS
            _LATEST_PARAMS = params
            _LAST_TRAIN_STATS = stats
            self._respond_json(200, {
                "status": "success",
                "stats": stats,
            })

        elif self.path == "/tick":
            assert _ENGINE is not None
            # Update agent core if provided
            core = payload.get("agent_core", {})
            if "hp_pct" in core:
                _ENGINE.trunk.agent.hp_pct = float(core["hp_pct"])
            if "mp_pct" in core:
                _ENGINE.trunk.agent.mp_pct = float(core["mp_pct"])
            if "is_in_combat" in core:
                _ENGINE.trunk.agent.is_in_combat = bool(core["is_in_combat"])
            if "position" in core:
                _ENGINE.trunk.agent.position = Vector3(**core["position"])

            # Entities
            entities = []
            for e in payload.get("entities", []):
                entities.append(WorldEntity(
                    guid=e.get("guid", "mob_1"),
                    name=e.get("name", "Hostile"),
                    position=Vector3(**e.get("position", {"x": 10.0, "y": 0.0, "z": 0.0})),
                    level=int(e.get("level", 15)),
                    health_pct=float(e.get("health_pct", 1.0)),
                    max_health=int(e.get("max_health", 500)),
                    is_hostile=bool(e.get("is_hostile", True)),
                    is_friendly=bool(e.get("is_friendly", False)),
                    is_humanoid=bool(e.get("is_humanoid", True)),
                    is_combatant=bool(e.get("is_combatant", False)),
                    is_casting=bool(e.get("is_casting", False)),
                    is_interruptible=bool(e.get("is_interruptible", True)),
                    is_stunned=bool(e.get("is_stunned", False)),
                    is_snared=bool(e.get("is_snared", False)),
                ))

            _ENGINE.weave_perceptions(entities=entities, nav_waypoints=[], affordances=[])
            action = _ENGINE.decide_actuation()
            minimal = _ENGINE.select_minimal_context()

            self._respond_json(200, {
                "action": action.to_dict(),
                "tensions": _ENGINE.evaluate_tensions(),
                "minimal_context": minimal,
            })
        else:
            self._respond_json(404, {"error": "Not Found"})

    def _respond_json(self, status_code: int, data: dict[str, Any]) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        # Route to standard logger instead of stderr spam
        logger.debug("%s - %s", self.address_string(), format % args)


def run_server() -> None:
    init_service()
    server = HTTPServer((_HOST, _PORT), GamePipelineHTTPHandler)
    logger.info("QUIPU Game Pipeline Service actively listening on http://%s:%d", _HOST, _PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.server_close()


if __name__ == "__main__":
    run_server()
