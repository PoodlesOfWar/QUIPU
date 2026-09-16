"""The interstitial arc: perception–vision–touch coherence on the held residual,
the unmediated chord, information density, and the physical frame attachment."""
from __future__ import annotations

import cmath
import math
import random
import sqlite3

import pytest

from src.quipu.qpsi.cat_residual import SENSES, _RING
from src.quipu.qpsi.interstitial import (
    ARC, EDGES, CHORD, OUTPUT_AXES, FRAME_KEY_PREFIX, KV_PREFIX,
    arc_is_contiguous, coherence, information_density, arc_measure, physical_frames, record,
)


def rows_with(*, lag_pv: float, lag_vt: float, direct_pt: float = 0.0, n: int = 24,
              noise: float = 0.0, seed: int = 3) -> list[dict]:
    """A vision carrier; perception lags it by lag_pv, touch by lag_vt.  direct_pt
    adds a perception–touch component that does not pass through vision."""
    rnd = random.Random(seed)
    rows = []
    for k in range(n):
        psi = 2 * math.pi * k / 12
        v = cmath.exp(1j * psi)
        p = cmath.exp(1j * (psi + lag_pv))
        t = cmath.exp(1j * (psi + lag_vt))
        if direct_pt:
            shared = cmath.exp(1j * (7 * psi))            # a second rhythm shared by p and t only
            p += direct_pt * shared
            t += direct_pt * shared
        held = {s: 0j for s in SENSES}
        held["vision"] = 0.02 * v + noise * complex(rnd.gauss(0, 1), rnd.gauss(0, 1))
        held["perception"] = 0.02 * p + noise * complex(rnd.gauss(0, 1), rnd.gauss(0, 1))
        held["touch"] = 0.02 * t + noise * complex(rnd.gauss(0, 1), rnd.gauss(0, 1))
        rows.append({"seq": k + 1, "at": float(k), "flip_count": k // 6, "held": held})
    return rows


# ---------------------------------------------------------------------------
# Geometry: the arc is a fact about the ring, checked rather than assumed
# ---------------------------------------------------------------------------

def test_arc_is_contiguous_on_the_ring_and_the_chord_is_not_an_edge():
    assert ARC == ("perception", "vision", "touch")
    assert arc_is_contiguous()
    assert "vision" in _RING["perception"] and "touch" in _RING["vision"]
    assert "touch" not in _RING["perception"]
    assert OUTPUT_AXES == {"perceptopoly": "perception", "loadopoly-ocr": "vision", "bakugo": "touch"}


# ---------------------------------------------------------------------------
# Coherence and the unmediated chord
# ---------------------------------------------------------------------------

def test_coherence_recovers_lag_and_is_unit_when_locked():
    rows = rows_with(lag_pv=0.4, lag_vt=-0.3)
    c = coherence(rows, "perception", "vision")
    assert abs(abs(c) - 1.0) < 1e-9 and abs(cmath.phase(c) - 0.4) < 1e-9
    c2 = coherence(rows, "vision", "touch")
    assert abs(cmath.phase(c2) - 0.3) < 1e-9          # touch lags vision by 0.3 → vision→touch = +0.3


def test_fully_mediated_arc_has_zero_interstitial():
    """Perception and touch each locked to vision, nothing shared directly: the
    chord is exactly what the two edges predict."""
    m = arc_measure(rows_with(lag_pv=0.4, lag_vt=-0.3))
    assert m.measured
    assert m.pairs["pv"].kind == "edge" and m.pairs["pt"].kind == "chord"
    assert abs(m.pairs["pv"].coherence - 1.0) < 1e-9 and abs(m.pairs["pt"].coherence - 1.0) < 1e-9
    assert abs(m.mediated_coherence - 1.0) < 1e-9
    assert m.interstitial < 1e-9


def test_direct_perception_touch_content_raises_the_interstitial():
    base = arc_measure(rows_with(lag_pv=0.4, lag_vt=-0.3)).interstitial
    direct = arc_measure(rows_with(lag_pv=0.4, lag_vt=-0.3, direct_pt=0.8)).interstitial
    assert direct > base + 0.05
    # and the chord stays well correlated while the vision edges weaken
    m = arc_measure(rows_with(lag_pv=0.4, lag_vt=-0.3, direct_pt=0.8))
    assert m.pairs["pt"].coherence > m.mediated_coherence


def test_interstitial_is_bounded_and_silent_axes_are_named():
    rows = rows_with(lag_pv=0.1, lag_vt=0.2)
    for r in rows:
        r["held"]["perception"] = 0j
    m = arc_measure(rows)
    assert not m.measured and any("no held content on perception" in x for x in m.reasons)
    assert 0.0 <= m.interstitial < 1.0


def test_short_window_is_refused():
    m = arc_measure(rows_with(lag_pv=0.1, lag_vt=0.2)[:5])
    assert not m.measured and any("min_rows" in x for x in m.reasons)


# ---------------------------------------------------------------------------
# Information density (the paper's quantity)
# ---------------------------------------------------------------------------

def test_patterned_trajectory_is_denser_than_noise():
    patterned = information_density(rows_with(lag_pv=0.4, lag_vt=-0.3, n=64))
    noisy = information_density(rows_with(lag_pv=0.4, lag_vt=-0.3, n=64, noise=0.05))
    assert patterned > noisy
    assert 0.0 <= noisy <= 1.0 and 0.0 <= patterned <= 1.0
    assert information_density([]) == 0.0


# ---------------------------------------------------------------------------
# Physical frames and the record
# ---------------------------------------------------------------------------

def _mem_db():
    cn = sqlite3.connect(":memory:")
    cn.execute("CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    return cn


def test_record_without_frames_says_so_and_persists():
    cn = _mem_db()
    rec = record(cn, "arc_test", rows_with(lag_pv=0.4, lag_vt=-0.3), now=5.0)
    assert rec["frames_present"] == [] and all(v is None for v in rec["frames"].values())
    assert rec["measure"]["measured"] and rec["at"] == 5.0
    stored = cn.execute("SELECT value FROM brain_kv WHERE key=?", (KV_PREFIX + "arc_test",)).fetchone()
    assert stored and '"interstitial"' in stored[0]


def test_record_attaches_the_outputs_latest_frames():
    import json
    cn = _mem_db()
    cn.execute("INSERT INTO brain_kv VALUES(?,?,?)", (FRAME_KEY_PREFIX + "perceptopoly",
               json.dumps({"standoff_m": 0.31, "scale_mm_per_px": 0.084, "coplanarity": 0.97, "source": "perceptopoly"}), ""))
    cn.execute("INSERT INTO brain_kv VALUES(?,?,?)", (FRAME_KEY_PREFIX + "loadopoly-ocr",
               json.dumps({"bearing_deg": 212.5, "range_m": 3.1, "reference_frame": "ENU_METER_WORLD_V1", "source": "loadopoly-ocr"}), ""))
    frames = physical_frames(cn)
    assert frames["perceptopoly"]["standoff_m"] == 0.31 and frames["bakugo"] is None
    rec = record(cn, "arc_f", rows_with(lag_pv=0.4, lag_vt=-0.3))
    assert sorted(rec["frames_present"]) == ["loadopoly-ocr", "perceptopoly"]
    assert rec["frames"]["loadopoly-ocr"]["reference_frame"] == "ENU_METER_WORLD_V1"
