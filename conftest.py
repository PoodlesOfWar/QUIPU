"""pytest conftest for QUIPU - local-only, no live DB, no LLM endpoints."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Path bootstrap: make `from src.quipu import ...` work from the repo root.
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    """Stub any LLM dispatch if an ensemble module is ever added back.

    In the parent suite this forced intent parsing onto its deterministic
    fallback. QUIPU carries no LLM caller, so this is a guarded no-op kept for
    forward compatibility.
    """
    try:
        from src.quipu import llm_ensemble  # type: ignore

        monkeypatch.setattr(
            llm_ensemble,
            "dispatch_parallel",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("LLM stubbed in tests")),
        )
    except Exception:
        pass
    yield


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """Return a fresh temporary SQLite file path (auto-cleaned)."""
    return tmp_path / "test_brain.sqlite"
