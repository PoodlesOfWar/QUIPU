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
