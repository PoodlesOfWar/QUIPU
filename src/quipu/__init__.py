"""QUIPU - extracted Supply Chain Architect model core.

This package replaces the parent application's ``src.brain`` package for the
modules carried into QUIPU. Modules use relative imports internally, so the
package name change is transparent to them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ._version import PHASES, __build_date__, __release__, __version__

_CONFIG_CACHE: dict | None = None


def load_config() -> dict:
    """Config shim for the extracted core.

    The parent application loaded ``pipeline/config/brain.yaml``. QUIPU modules
    only consult optional tuning keys via ``.get()``, so an empty mapping means
    "use module defaults". Point ``QUIPU_CONFIG_JSON`` at a JSON file to
    override specific keys without reintroducing YAML or app config.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = {}
        p = os.environ.get("QUIPU_CONFIG_JSON")
        if p and Path(p).is_file():
            try:
                loaded = json.loads(Path(p).read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    _CONFIG_CACHE = loaded
            except Exception:
                _CONFIG_CACHE = {}
    return _CONFIG_CACHE

# DIVINE_BLESSING_SQRT(-1): route every Entirety write path through the six
# Physical Gates (UEQGM v0.9.25).  Set QUIPU_DIVINE_BLESSING=0 to leave the
# paths unrouted for a session.
if os.environ.get("QUIPU_DIVINE_BLESSING", "1") != "0":
    try:
        from . import divine_blessing as _divine_blessing
        _divine_blessing.enable()
    except Exception as _exc:  # pragma: no cover
        import logging as _logging
        _logging.getLogger(__name__).warning("DIVINE_BLESSING_SQRT(-1) not enabled: %s", _exc)

# Self-organising loop (qpsi.self_organising): phase read from the ingest flux,
# a lumped memristive model on the seven axes, the prior advanced only on
# realisation.  Default OFF: set QUIPU_SELF_ORGANISING=1 to wire it for a
# session.  The gates above are untouched either way.
if os.environ.get("QUIPU_SELF_ORGANISING", "0") == "1":
    try:
        from .qpsi import self_organising as _self_organising
        _self_organising.enable()
    except Exception as _exc:  # pragma: no cover
        import logging as _logging
        _logging.getLogger(__name__).warning("self_organising not enabled: %s", _exc)

from . import quipu_game_mesh
from . import video_gameplay_pipeline
from . import game_pipeline_service
from . import human_fidelity_assessor


