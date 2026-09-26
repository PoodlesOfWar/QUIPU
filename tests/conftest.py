"""Test isolation: no test touches a real brain.

Every test runs against a throwaway database unless it sets SCB_DB_PATH itself.
Before v0.48.0 tests that did not set it opened the repository's
local_brain.sqlite -- the live Entirety -- and wrote to it.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_brain(tmp_path_factory):
    if os.environ.get("SCB_DB_PATH"):
        yield
        return
    path = tmp_path_factory.mktemp("brain") / "local_brain.sqlite"
    os.environ["SCB_DB_PATH"] = str(path)
    try:
        yield
    finally:
        os.environ.pop("SCB_DB_PATH", None)
