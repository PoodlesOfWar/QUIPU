"""qpsi.resonance_phase — the parity bit is the realised part of one phasor whose
latent part is the 翈 axis; the resonance forms under the field, decays without
it, and the bit crosses only by turning through the latent quarter."""
from __future__ import annotations

import math
import random
import sqlite3

import pytest

from src.quipu.qpsi import flux_phase as fp
from src.quipu.qpsi import resonance_phase as rp
from src.quipu.qpsi.cat_residual import parity_from_turns

T0 = 1_800_000_000.0
CFG = rp.ResonanceConfig()           # θ_on 0.6, θ_off 0.3, τ and κ derived


def _sessions(n: int = 3, runs: int = 12, gap=(200.0, 1500.0), docs=(3, 9), seed: int = 7):
    """n sessions a day apart; gaps inside each burst often exceed flux_phase's 300 s."""
    rnd = random.Random(seed)
    out = []
    for s in range(n):
        t = T0 + s * 86400.0
        for _ in range(runs):
            t += rnd.uniform(*gap)
            out.append({"ts": t, "total_condensed": rnd.randint(*docs)})
    return out


# ---------------------------------------------------------------------------
# Rest and the session clock
# ---------------------------------------------------------------------------

def test_a_network_that_never_saw_the_field_rests_in_deepen_on_the_real_axis():
    r = rp.replay([], now=T0, cfg=CFG)
    assert r.state == "deepen" and r.bit == rp.DEEPEN and r.turns == 2
    assert r.R == 0.0 and r.realised == -1.0 and r.latent == 0.0 and not r.in_band


def test_one_dusk_and_one_dawn_per_session_where_the_window_chatters():
    h = _sessions(n=3)
    r = rp.replay(h, now=T0 + 3 * 86400.0, cfg=CFG)
    assert r.derivation.source == "history" and r.derivation.separable
    assert r.dawns == 3 and r.dusks == 3
    assert r.state == "deepen" and r.bit == rp.DEEPEN
    # The same field read through flux_phase's 300 s window, sampled each minute.
    prev, flips = None, 0
    for t in range(int(T0), int(T0 + 3 * 86400), 60):
        b = fp.ingest_flux(h, now=t, window_s=300).parity
        flips += prev is not None and b != prev
        prev = b
    assert flips > 2 * 3 * 3                     # the window splits each session many times


def test_four_quarter_turns_make_a_session():
    h = _sessions(n=2)
    r = rp.replay(h, now=T0 + 2 * 86400.0, cfg=CFG)
    net = sum(e["quarter"] for e in r.transitions)
    assert r.turns == 2 + net
    assert net == 4 * r.dawns                    # i⁴ = 1 per session: back to deepen


def test_saturation_bounds_dusk_after_the_field_stops():
    """A burst six times louder cannot hold broaden longer than τ·ln(1/θ_off)."""
    quiet = _sessions(n=1, docs=(3, 9), seed=1)
    loud = [dict(e, total_condensed=6 * e["total_condensed"]) for e in quiet]
    last = quiet[-1]["ts"]
    for h in (quiet, loud):
        # Fix τ and κ so loudness is the only difference between the two runs.
        cfg = rp.ResonanceConfig(tau_s=1200.0, kappa=0.25)
        bound = cfg.tau_s * math.log(1.0 / cfg.theta_off)
        r = rp.replay(h, now=last + bound + 1.0, cfg=cfg)
        assert r.state == "deepen"
        dusk_t = [e["t"] for e in r.transitions if e["to"] == "deepen"][-1]
        assert dusk_t - last <= bound + 1e-6


# ---------------------------------------------------------------------------
# Entanglement: the bit crosses only through the latent axis
# ---------------------------------------------------------------------------

def test_the_bit_never_flips_without_passing_through_the_latent_quarter():
    r = rp.replay(_sessions(n=3), now=T0 + 3 * 86400.0, cfg=CFG)
    turns, bit = 2, rp.DEEPEN
    for e in r.transitions:
        assert abs(e["quarter"]) == 1            # one quarter turn at a time, always
        before = rp.STATES.index(e["from"])
        assert before == turns % 4
        turns += e["quarter"]
        new_bit = rp.committed_bit(turns)
        if new_bit != bit:
            assert before % 2 == 1               # every flip leaves an odd (翈) state
        bit = new_bit
    assert turns == r.turns and bit == r.bit


def test_a_pulse_too_small_to_cross_the_band_is_a_latent_excursion_not_a_flip():
    cfg = rp.ResonanceConfig(tau_s=600.0, kappa=0.4)
    h = [{"ts": T0, "total_condensed": 1}]       # R = 1 − e^-0.4 ≈ 0.33: inside the band
    mid = rp.replay(h, now=T0 + 1.0, cfg=cfg)
    assert mid.state == "dawn" and mid.in_band and mid.bit == rp.DEEPEN
    assert mid.latent < 0.0                      # rising: −i side
    after = rp.replay(h, now=T0 + 3600.0, cfg=cfg)
    assert after.state == "deepen" and after.excursions == 1 and after.dawns == 0
    assert after.latent == 0.0 and after.bit == rp.DEEPEN


def test_the_phasor_is_unit_real_on_a_bit_and_pure_latent_at_the_band_midpoint():
    on, off = CFG.theta_on, CFG.theta_off
    mid = 0.5 * (on + off)
    assert rp.phase_angle(1, mid, on, off) == pytest.approx(math.pi / 2)        # dusk: +i
    assert rp.phase_angle(3, mid, on, off) == pytest.approx(3 * math.pi / 2)    # dawn: −i
    assert rp.phase_angle(0, 0.9, on, off) == 0.0
    assert rp.phase_angle(2, 0.1, on, off) == math.pi
    for turns in range(8):
        if turns % 2 == 0:
            assert rp.committed_bit(turns) == parity_from_turns(turns)          # i^turns itself
        else:
            assert parity_from_turns(turns) == 0                                # 翈: no bit of its own
            assert rp.committed_bit(turns) == parity_from_turns(turns - 1)      # held from entry
    assert rp.committed_bit(2) == -1                                            # i² = −1: deepen


def test_reading_z_has_unit_modulus_and_the_latent_sign_names_the_direction():
    cfg = rp.ResonanceConfig(tau_s=1000.0, kappa=1.0)
    h = [{"ts": T0, "total_condensed": 3}]       # R ≈ 0.95: broaden
    # find a moment inside the dusk band
    t_in = T0 + 1000.0 * math.log(0.95 / 0.45)
    r = rp.replay(h, now=t_in, cfg=cfg)
    assert r.state == "dusk" and r.bit == rp.BROADEN
    assert abs(r.z) == pytest.approx(1.0) and r.latent > 0.0


# ---------------------------------------------------------------------------
# Derivation from the field
# ---------------------------------------------------------------------------

def test_tau_and_kappa_are_derived_from_the_history():
    h = _sessions(n=3)
    events = rp.events_from_history(h)
    d = rp.derive(events, CFG)
    assert d.source == "history" and d.break_ratio >= rp.MIN_BREAK_RATIO
    assert d.tau_s == pytest.approx(d.within_gap_max_s / math.log(CFG.theta_on / CFG.theta_off))
    lift = math.log(1.0 / (1.0 - rp.target_lift(CFG)))
    assert d.kappa == pytest.approx(lift / d.median_docs)
    assert d.within_gap_max_s < 1500.0 < 3600.0 < d.between_gap_min_s


def test_too_little_history_falls_back_and_says_so():
    d = rp.derive(rp.events_from_history([{"ts": T0, "total_condensed": 2}]), CFG)
    # one run gives κ (median docs) but no gap, so τ alone falls back
    assert d.source == "mixed" and d.within_gap_max_s is None and d.separable is None
    assert rp.derive([], CFG).source == "fallback"
    assert d.tau_s == pytest.approx(rp.FALLBACK_WITHIN_GAP_S / math.log(2.0))


def test_env_overrides_are_recorded(monkeypatch):
    monkeypatch.setenv(rp.TAU_ENV, "900")
    monkeypatch.setenv(rp.KAPPA_ENV, "0.5")
    monkeypatch.setenv(rp.THETA_ON_ENV, "0.7")
    monkeypatch.setenv(rp.THETA_OFF_ENV, "0.2")
    cfg = rp.ResonanceConfig.from_env()
    r = rp.replay(_sessions(n=1), now=T0 + 86400.0, cfg=cfg)
    assert r.derivation.source == "override" and r.derivation.tau_s == 900.0
    assert r.to_json()["config"] == {"theta_on": 0.7, "theta_off": 0.2, "tau_s": 900.0, "kappa": 0.5}
    monkeypatch.setenv(rp.THETA_OFF_ENV, "0.9")                      # off ≥ on → defaults
    assert rp.ResonanceConfig.from_env().theta_off == rp.DEFAULT_THETA_OFF
    with pytest.raises(ValueError):
        rp.ResonanceConfig(theta_on=0.3, theta_off=0.6)


def test_reading_is_causal_and_malformed_history_is_skipped():
    junk = ["x", None, 3, {"ts": "not a date", "total_condensed": 9}, {"total_condensed": 9},
            {"ts": T0, "total_condensed": "seven"}, {"ts": T0, "total_condensed": 0}]
    assert rp.events_from_history(junk) == []
    future = [{"ts": T0 + 10_000.0, "total_condensed": 50}]
    r = rp.replay(future, now=T0, cfg=CFG)
    assert r.runs == 0 and r.state == "deepen"


# ---------------------------------------------------------------------------
# Wiring and the write boundary
# ---------------------------------------------------------------------------

def test_wrapper_returns_the_committed_bit_and_falls_back_on_failure(monkeypatch):
    calls = []

    def fallback(observer, t=None):
        calls.append((observer, t))
        return 1

    fn = rp.wrap_bit_flip_parity(fallback)
    assert fn.__name__ == "bit_flip_parity" and fn._qpsi_fallback is fallback
    monkeypatch.setattr(rp, "read", lambda now=None: rp.replay(_sessions(n=1), now=T0 + 86400.0, cfg=CFG))
    assert fn(0.5, T0 + 86400.0) == rp.DEEPEN and not calls
    monkeypatch.setattr(rp, "read", lambda now=None: (_ for _ in ()).throw(RuntimeError("kv down")))
    assert fn(0.5, 123.0) == 1 and calls == [(0.5, 123.0)]


def test_record_writes_only_its_own_key():
    cn = sqlite3.connect(":memory:")
    cn.execute("CREATE TABLE brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    r = rp.replay(_sessions(n=1), now=T0 + 86400.0, cfg=CFG)
    rp.record(cn, r, "system_entirety")
    keys = [k for (k,) in cn.execute("SELECT key FROM brain_kv")]
    assert keys == [rp.KV_PREFIX + "system_entirety"]
    with pytest.raises(PermissionError):
        rp._kv_set(cn, "entirety:conscious_emergence", {})
