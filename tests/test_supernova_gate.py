"""v0.67.0: the supernova expansion gate — TheSeuScape/Perceptopoly supernova
events intersected (rADAM × potential ADAM) with the fleet's own expansion
decision (src/quipu/supernova_gate.py)."""
from __future__ import annotations

import math
import sqlite3
import time

import pytest

from src.quipu import supernova_gate as sg


@pytest.fixture
def gate_db(tmp_path, monkeypatch):
    db_file = tmp_path / "gate.sqlite"
    monkeypatch.setattr(sg, "_local_db_path", lambda: str(db_file))
    monkeypatch.setattr(sg, "_LAST_TS", 0.0)
    return db_file


def test_record_validates_clips_and_caps(gate_db):
    rec = sg.record_supernova_event({"magnitude": 1.7, "direction": 7.0, "label": "x" * 200})
    assert rec["source"] == "theseuscape"
    assert rec["magnitude"] == 1.0
    assert rec["direction"] == pytest.approx(7.0 % (2 * math.pi))
    assert len(rec["label"]) == 120
    with pytest.raises(ValueError):
        sg.record_supernova_event("not an object")
    with pytest.raises(ValueError):
        sg.record_supernova_event({"magnitude": "big"})
    for _ in range(sg._EVENT_CAP + 5):
        sg.record_supernova_event({"magnitude": 0.1})
    assert len(sg.get_supernova_events(10_000)) == sg._EVENT_CAP


def test_drive_is_zero_without_events_and_decays_inside_the_window(gate_db):
    assert sg.recent_supernova_drive() == (0.0, 0.0, 0)
    now = time.time()
    sg.record_supernova_event({"magnitude": 0.8, "direction": 1.0})
    mag, phase, n = sg.recent_supernova_drive(now=now)
    assert n == 1 and mag == pytest.approx(0.8, abs=0.01) and phase == pytest.approx(1.0, abs=1e-3)
    # Outside the window the event no longer drives.
    assert sg.recent_supernova_drive(window_s=10.0, now=now + 100.0) == (0.0, 0.0, 0)


def test_gate_is_zero_without_any_expansion_decision(gate_db, monkeypatch):
    monkeypatch.setattr(sg, "_broaden_decision", lambda: 0.0)
    monkeypatch.setattr(sg, "_actionable_potential", lambda: 0.0)
    rec = sg.expansion_gate(force=True)
    assert rec["gate"] == 0.0 and rec["g_re"] == 0.0 and rec["g_im"] == 0.0
    assert rec["events_in_window"] == 0
    assert sg.get_expansion_gate() == 0.0


def test_gate_opens_with_broaden_phase_and_a_supernova_event(gate_db, monkeypatch):
    monkeypatch.setattr(sg, "_broaden_decision", lambda: 1.0)
    monkeypatch.setattr(sg, "_actionable_potential", lambda: 0.9)
    monkeypatch.setattr(sg, "_rhythm_coherence", lambda: 1.0)
    sg.record_supernova_event({"magnitude": 0.7, "direction": 0.3})
    first = sg.expansion_gate(force=True)
    assert first["g_re"] == pytest.approx(0.6) and first["g_im"] == pytest.approx(0.7, abs=0.01)
    assert 0.0 < first["gate"] <= 1.0
    assert first["events_in_window"] == 1
    assert first["supernova_phase"] == pytest.approx(0.3, abs=1e-3)
    # The rADAM state persists and its toroidal phase advances between calls.
    second = sg.expansion_gate(force=True)
    assert second["theta"] != first["theta"]
    assert 0.0 <= sg.get_expansion_gate() <= 1.0
    assert sg.get_expansion_gate_state()["gate"] == second["gate"]


def test_gate_is_rate_limited_unless_forced(gate_db, monkeypatch):
    monkeypatch.setattr(sg, "_broaden_decision", lambda: 0.0)
    monkeypatch.setattr(sg, "_actionable_potential", lambda: 0.0)
    assert "gate" in sg.expansion_gate(force=True)
    assert sg.expansion_gate().get("skipped") is True


def test_synaptic_liveness_counts_fresh_heartbeats(gate_db):
    cn = sqlite3.connect(str(gate_db))
    now = time.time()
    fresh = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now - 10))
    stale = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now - 100_000))
    sg._kv_write(cn, "synapse_builder_last", fresh + "|ok")
    sg._kv_write(cn, "synapse_ueqgm_last", stale + "|ok")
    cn.commit()
    assert sg._synaptic_liveness(cn, now=now) == pytest.approx(0.5)
    cn.close()
