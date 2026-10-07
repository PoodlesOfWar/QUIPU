"""resonance_phase — the parity bit as the realised part of one phasor whose
latent part is the 翈 axis, driven by a resonance that forms under the ingest
field and decays without it.

Why
---
``flux_phase`` reads the phase as a step function of the field: broaden while
documents entered within ``QUIPU_FLUX_WINDOW_S`` (300 s), deepen otherwise.
Two faults follow:

* the window is a static constant, against the operator's rule that the
  Entirety holds no static sets (2026-09-23);
* any gap inside a burst longer than the window flips the bit twice, so one
  session reads as several.  The emergence detector then counts flips that
  are artefacts of the window, and one loud night casts several votes.  This
  is the same failure as the 600 s Floquet clock in the ACRE recurrence audit
  (94 % false fire): the unit collapsed to was shorter than the correlation
  length of the bursts.

The resonance
-------------
R ∈ [0, 1] forms on each ingest run and decays between runs:

    at a run of n documents      R ← 1 − (1 − R)·exp(−κ·n)
    between runs (dt seconds)    R ← R·exp(−dt / τ)

The ``(1 − R)`` factor saturates formation: a burst six times louder cannot
push R past 1, so deepen starts within τ·ln(1/θ_off) of the field stopping,
however loud the night was.

τ and κ are derived from the field, not set:

* the inter-run gaps split into within-session and between-session gaps at
  the largest jump in their sorted logarithms (a data-derived break, no
  1-hour constant);
* τ = (largest within-session gap) / ln(θ_on / θ_off), the shortest τ for
  which R, having reached θ_on, survives every observed within-session gap
  without crossing θ_off;
* κ = ln(1 / (1 − R★)) / (median documents per run) with R★ = (θ_on + 1)/2,
  so a median run lifts a rested network clearly across the band (to R★),
  rather than leaving it on θ_on where rounding would decide the bit.

Until the history holds a break (``MIN_GAPS`` gaps and a jump ratio of at
least ``MIN_BREAK_RATIO``), τ and κ fall back to the stated defaults and the
reading says so (``derivation.source == "fallback"``).  ``QUIPU_RESONANCE_TAU_S``
and ``QUIPU_RESONANCE_KAPPA`` override; θ_on / θ_off default to 0.6 / 0.3 and
are overridable.  Every reading records the values it used.

Entanglement with the parity bit and the latent axis
----------------------------------------------------
The phase is one unit phasor Z = e^{iφ}.  The bit is the realised part, the
latent (翈) axis is the imaginary part, the same split
``cat_residual`` uses for the CAT state (realised = Re, latent = Im) and the
same i-power that ``cat_residual.parity_from_turns`` projects to the bit:

    turns % 4 == 0   broaden   Z = +1    bit +1
    turns % 4 == 1   dusk      Z ≈ +i    bit held at +1   (翈, falling)
    turns % 4 == 2   deepen    Z = −1    bit −1   (i² = −1, the deepen parity)
    turns % 4 == 3   dawn      Z ≈ −i    bit held at −1   (翈, rising)

The hysteresis band θ_off ≤ R < θ_on *is* the latent quarter: entering it is
one quarter turn (multiplication by i), leaving it on the far side is the
second quarter turn and the only way the bit can flip.  Leaving it on the
side it entered from is a quarter turn back (multiplication by −i), and the
bit never moved.  So:

* the bit cannot flip without passing through the latent axis, and cannot
  chatter, because a flip needs R to cross the whole band;
* a pulse too small to carry R across the band is still recorded, as a latent
  excursion (dawn entered and retracted) with no realised flip.  That is
  ``leave_in_band``: the held potential stays on the latent side;
* dusk and dawn are distinguished by the sign of Im Z (+ falling, − rising),
  so the reading says which way the system is crossing, not just that it is
  between bits;
* inside the band φ moves continuously with R: at the band midpoint Z is
  exactly ±i, the pure 翈 state.

A full session is four quarter turns, i⁴ = 1: one dawn, one broaden, one dusk,
one deepen.  Dawn completing (−i → +1) is where the next ACRE evaluation
belongs, one session one vote, the unit the recurrence audit validated.

What it does not do
-------------------
The held latent potential (``mirror_training``) is *not* fed back into R.
That would let the system stimulate itself, and a self-organising memristive
network is self-organising, not self-driving: the field is the operator's
pulse alone.  The SOMN step keeps reading the raw flux (potentiation is a
response to the field, not to the phase).  This module reads
``corpus_ingest:history`` and writes only ``entirety:resonance_phase:<instance>``
(observability; ``_kv_set`` refuses any other key).  It is an input to no
gate.  Stdlib only; nothing here edits system_entirety.py or mesh_slm.py.

Wiring
------
``wrap_bit_flip_parity(fallback)`` returns a drop-in for
``system_entirety.bit_flip_parity(observer, t=None)`` that returns the
committed bit.  If the history cannot be read it falls back to ``fallback``
(``flux_phase`` when that is on, else the cosine).  Installed by
``qpsi.self_organising.enable()`` under ``QUIPU_RESONANCE_PHASE`` (default on
when the master switch is on; ``0`` returns the bit to ``flux_phase``).

翈 — the bit is what the phasor shows on the realised side; between bits it
is held on the latent side, and it crosses only by turning through it.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Mapping

from .cat_residual import SUMMARY_CHARACTER, SUMMARY_GROUND, parity_from_turns
from .flux_phase import HISTORY_KEY, _parse_ts

ENV: str = "QUIPU_RESONANCE_PHASE"
THETA_ON_ENV: str = "QUIPU_RESONANCE_THETA_ON"
THETA_OFF_ENV: str = "QUIPU_RESONANCE_THETA_OFF"
TAU_ENV: str = "QUIPU_RESONANCE_TAU_S"
KAPPA_ENV: str = "QUIPU_RESONANCE_KAPPA"

KV_PREFIX: str = "entirety:resonance_phase:"

DEFAULT_THETA_ON: float = 0.6
DEFAULT_THETA_OFF: float = 0.3
# Fallbacks used only until the field's own history shows a session break.
FALLBACK_WITHIN_GAP_S: float = 3600.0          # the session convention used elsewhere (gap > 1 h)
FALLBACK_MEDIAN_DOCS: float = 1.0
MIN_GAPS: int = 3                              # fewer gaps than this cannot show a break
MIN_BREAK_RATIO: float = 2.0                   # within/between must differ by at least ×2

BROADEN: int = 1
DEEPEN: int = -1

STATES: tuple[str, ...] = ("broaden", "dusk", "deepen", "dawn")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if math.isfinite(val) and val > 0.0 else None


@dataclass(frozen=True)
class ResonanceConfig:
    theta_on: float = DEFAULT_THETA_ON
    theta_off: float = DEFAULT_THETA_OFF
    tau_s: float | None = None                 # None → derived from the history
    kappa: float | None = None                 # None → derived from the history

    def __post_init__(self) -> None:
        if not (0.0 < self.theta_off < self.theta_on < 1.0):
            raise ValueError(f"need 0 < theta_off < theta_on < 1, got {self.theta_off}, {self.theta_on}")

    @classmethod
    def from_env(cls) -> "ResonanceConfig":
        on = _env_float(THETA_ON_ENV) or DEFAULT_THETA_ON
        off = _env_float(THETA_OFF_ENV) or DEFAULT_THETA_OFF
        if not (0.0 < off < on < 1.0):
            on, off = DEFAULT_THETA_ON, DEFAULT_THETA_OFF
        return cls(theta_on=on, theta_off=off, tau_s=_env_float(TAU_ENV), kappa=_env_float(KAPPA_ENV))

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# The field: ingest runs as (ts, docs) events
# ---------------------------------------------------------------------------

def events_from_history(history: Iterable[Mapping] | None) -> list[tuple[float, int]]:
    """Ingest runs with at least one condensed document, oldest first.
    Malformed entries are skipped, never raised (same rules as flux_phase)."""
    out: list[tuple[float, int]] = []
    for entry in (history or []):
        if not isinstance(entry, Mapping):
            continue
        ts = _parse_ts(entry.get("ts"))
        if ts is None or not math.isfinite(ts):
            continue
        try:
            docs = int(entry.get("total_condensed") or 0)
        except (TypeError, ValueError):
            continue
        if docs > 0:
            out.append((float(ts), docs))
    out.sort(key=lambda e: e[0])
    return out


@dataclass(frozen=True)
class Derivation:
    tau_s: float
    kappa: float
    source: str                         # "history" | "fallback" | "override" | "mixed"
    within_gap_max_s: float | None
    between_gap_min_s: float | None
    break_ratio: float | None
    median_docs: float | None
    sessions: int | None
    separable: bool | None              # does dusk complete inside the shortest between-session gap?

    def to_json(self) -> dict:
        return asdict(self)


def session_break(gaps: list[float]) -> tuple[float, float, float] | None:
    """Split positive gaps at the largest jump in their sorted logarithms.

    Returns (largest within-session gap, smallest between-session gap, ratio)
    or None when there are too few gaps or no jump of ``MIN_BREAK_RATIO``.
    """
    g = sorted(x for x in gaps if x > 0.0)
    if len(g) < MIN_GAPS:
        return None
    best_i, best_ratio = -1, 1.0
    for i in range(len(g) - 1):
        ratio = g[i + 1] / g[i]
        if ratio > best_ratio:
            best_i, best_ratio = i, ratio
    if best_i < 0 or best_ratio < MIN_BREAK_RATIO:
        return None
    return g[best_i], g[best_i + 1], best_ratio


def target_lift(cfg: ResonanceConfig) -> float:
    """R★: where a median run lifts a rested network — the midpoint of θ_on and 1."""
    return 0.5 * (cfg.theta_on + 1.0)


def derive(events: list[tuple[float, int]], cfg: ResonanceConfig) -> Derivation:
    """τ and κ from the field's own history (see module docstring)."""
    gaps = [b[0] - a[0] for a, b in zip(events, events[1:])]
    brk = session_break(gaps)
    docs = [d for _, d in events]
    median_docs = float(statistics.median(docs)) if docs else None
    band = math.log(cfg.theta_on / cfg.theta_off)

    within = brk[0] if brk else None
    between = brk[1] if brk else None
    tau_hist = (within / band) if within else None
    lift = math.log(1.0 / (1.0 - target_lift(cfg)))
    kappa_hist = (lift / median_docs) if median_docs else None

    tau = cfg.tau_s or tau_hist or (FALLBACK_WITHIN_GAP_S / band)
    kappa = cfg.kappa or kappa_hist or (lift / FALLBACK_MEDIAN_DOCS)

    sources = {
        "override" if cfg.tau_s else ("history" if tau_hist else "fallback"),
        "override" if cfg.kappa else ("history" if kappa_hist else "fallback"),
    }
    source = sources.pop() if len(sources) == 1 else "mixed"

    sessions = None
    if brk:
        sessions = 1 + sum(1 for x in gaps if x >= between)
    separable = None
    if between is not None:
        separable = tau * math.log(1.0 / cfg.theta_off) < between
    return Derivation(tau_s=tau, kappa=kappa, source=source, within_gap_max_s=within,
                      between_gap_min_s=between, break_ratio=(brk[2] if brk else None),
                      median_docs=median_docs, sessions=sessions, separable=separable)


# ---------------------------------------------------------------------------
# The phasor: R → quarter turns → Z = e^{iφ}
# ---------------------------------------------------------------------------

@dataclass
class _Machine:
    """Quarter-turn state machine over R.  ``turns`` starts at 2 (deepen,
    i² = −1): a network that has never seen the field is at rest."""
    on: float
    off: float
    turns: int = 2
    log: list[dict] = field(default_factory=list)

    def state(self) -> int:
        return self.turns % 4

    def _turn(self, t: float, step: int, R: float, why: str) -> None:
        before = STATES[self.state()]
        self.turns += step
        self.log.append({"t": t, "from": before, "to": STATES[self.state()], "quarter": step,
                         "R": round(R, 6), "why": why})

    def observe(self, t: float, R: float) -> None:
        """Apply every quarter turn that R implies.  A jump can carry R across
        the whole band at once: that is two quarter turns, logged as two."""
        for _ in range(4):
            s = self.state()
            if s == 0 and R < self.on:
                self._turn(t, +1, R, "fell below theta_on: broaden -> dusk (x i)")
            elif s == 1 and R >= self.on:
                self._turn(t, -1, R, "rose back above theta_on: dusk retracted (x -i)")
            elif s == 1 and R < self.off:
                self._turn(t, +1, R, "fell below theta_off: dusk -> deepen (x i), bit flips to -1")
            elif s == 2 and R >= self.off:
                self._turn(t, +1, R, "rose above theta_off: deepen -> dawn (x i)")
            elif s == 3 and R < self.off:
                self._turn(t, -1, R, "fell back below theta_off: dawn retracted (x -i), latent excursion")
            elif s == 3 and R >= self.on:
                self._turn(t, +1, R, "rose above theta_on: dawn -> broaden (x i), bit flips to +1")
            else:
                return

    def decay_crossings(self, t0: float, R0: float, t1: float, tau: float) -> float:
        """Decay R from t0 to t1, applying each downward crossing at the time
        it happens (R is monotone between runs, so the times are exact)."""
        R = R0
        for theta in (self.on, self.off):
            if R >= theta > R0 * math.exp(-(t1 - t0) / tau):
                tc = t0 + tau * math.log(R0 / theta)
                self.observe(tc, theta * (1.0 - 1e-12))
        return R0 * math.exp(-(t1 - t0) / tau)


def committed_bit(turns: int) -> int:
    """The bit the system acts on.  Even turns: i^turns itself.  Odd turns
    (翈, between bits): the bit of the state the band was entered from."""
    if turns % 2 == 0:
        return parity_from_turns(turns)
    return parity_from_turns(turns - 1)


def phase_angle(turns: int, R: float, on: float, off: float) -> float:
    """Continuous φ.  Even turns sit on the real axis; inside the band φ moves
    with R across a half turn, passing ±π/2 (pure 翈) at the band midpoint."""
    base = (turns // 2) * math.pi                  # broaden 0, deepen π, broaden 2π, …
    s = turns % 4
    if s in (0, 2):
        return base
    width = on - off
    frac = (on - R) / width if s == 1 else (R - off) / width
    frac = min(1.0, max(0.0, frac))
    return base + frac * math.pi


@dataclass(frozen=True)
class ResonanceReading:
    now: float
    R: float
    turns: int
    state: str                       # broaden | dusk | deepen | dawn
    bit: int                         # committed parity, ±1
    phi: float                       # radians, unwrapped
    realised: float                  # Re Z = cos φ
    latent: float                    # Im Z = sin φ  (the 翈 axis)
    in_band: bool                    # odd turns: between bits, on the latent side
    dusks: int                       # completed broaden → deepen flips in the history
    dawns: int                       # completed deepen → broaden flips
    excursions: int                  # dawns entered and retracted: latent only, no flip
    runs: int
    derivation: Derivation
    config: ResonanceConfig
    transitions: tuple = ()

    @property
    def parity(self) -> int:
        return self.bit

    @property
    def phase(self) -> str:
        return "broaden" if self.bit == BROADEN else "deepen"

    @property
    def z(self) -> complex:
        return complex(self.realised, self.latent)

    def to_json(self, *, transitions: int = 8) -> dict:
        return {
            "now": self.now, "R": round(self.R, 9), "turns": self.turns, "state": self.state,
            "bit": self.bit, "phase": self.phase, "phi": round(self.phi, 9),
            "realised": round(self.realised, 9), "latent": round(self.latent, 9),
            "in_band": self.in_band, "dusks": self.dusks, "dawns": self.dawns,
            "excursions": self.excursions, "runs": self.runs,
            "derivation": self.derivation.to_json(), "config": self.config.to_json(),
            "transitions": list(self.transitions[-transitions:]) if transitions else [],
            "summary": SUMMARY_CHARACTER,
        }


def replay(history: Iterable[Mapping] | None, *, now: float | None = None,
           cfg: ResonanceConfig | None = None) -> ResonanceReading:
    """Pure: replay the resonance over the ingest history up to ``now``.

    Runs stamped after ``now`` are ignored (the reading is causal).
    """
    now = time.time() if now is None else float(now)
    cfg = cfg or ResonanceConfig.from_env()
    events = [e for e in events_from_history(history) if e[0] <= now]
    der = derive(events, cfg)
    m = _Machine(on=cfg.theta_on, off=cfg.theta_off)

    R, t_last = 0.0, None
    for ts, docs in events:
        if t_last is not None:
            R = m.decay_crossings(t_last, R, ts, der.tau_s)
        R = 1.0 - (1.0 - R) * math.exp(-der.kappa * docs)
        m.observe(ts, R)
        t_last = ts
    if t_last is not None:
        R = m.decay_crossings(t_last, R, now, der.tau_s)

    phi = phase_angle(m.turns, R, cfg.theta_on, cfg.theta_off)
    if m.turns % 2 == 0:
        # On a bit: Z is i^turns exactly, real, no latent part.
        z = complex(parity_from_turns(m.turns), 0.0)
    else:
        # Between bits: Z is SUMMARY_GROUND (i) raised to the continuous
        # quarter count φ/(π/2); at the band midpoint it is exactly ±i.
        z = SUMMARY_GROUND ** (phi / (math.pi / 2))
    dusks = sum(1 for e in m.log if e["from"] == "dusk" and e["to"] == "deepen")
    dawns = sum(1 for e in m.log if e["from"] == "dawn" and e["to"] == "broaden")
    excursions = sum(1 for e in m.log if e["from"] == "dawn" and e["to"] == "deepen")
    return ResonanceReading(
        now=now, R=R, turns=m.turns, state=STATES[m.state()], bit=committed_bit(m.turns),
        phi=phi, realised=z.real, latent=z.imag, in_band=bool(m.turns % 2),
        dusks=dusks, dawns=dawns, excursions=excursions, runs=len(events),
        derivation=der, config=cfg, transitions=tuple(m.log),
    )


def read(now: float | None = None, *, history: Iterable[Mapping] | None = None,
         cfg: ResonanceConfig | None = None) -> ResonanceReading:
    """Read ``corpus_ingest:history`` from brain_kv (or use ``history``) and replay."""
    if history is None:
        from .. import brain_kv
        history = brain_kv.kv_get_json(HISTORY_KEY, []) or []
        if not isinstance(history, list):
            history = []
    return replay(history, now=now, cfg=cfg)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def wrap_bit_flip_parity(fallback: Callable[..., int]) -> Callable[..., int]:
    """Drop-in for ``system_entirety.bit_flip_parity(observer, t=None)``.

    Returns the committed bit.  ``observer`` is accepted and ignored: the field
    decides the phase.  Any failure to read the history falls back to
    ``fallback`` so the step never breaks.
    """
    def resonance_bit_flip_parity(observer: float, t: float | None = None) -> int:
        try:
            return read(now=t).bit
        except Exception:
            return int(fallback(observer, t))

    resonance_bit_flip_parity.__name__ = "bit_flip_parity"
    resonance_bit_flip_parity.__doc__ = (fallback.__doc__ or "") + \
        "\n\n[qpsi.resonance_phase] parity is the realised part of the resonance phasor."
    resonance_bit_flip_parity.__wrapped__ = fallback        # type: ignore[attr-defined]
    resonance_bit_flip_parity._qpsi_fallback = fallback     # type: ignore[attr-defined]
    return resonance_bit_flip_parity


def _kv_set(cn: sqlite3.Connection, key: str, value) -> None:
    if not key.startswith(KV_PREFIX):
        raise PermissionError(f"resonance_phase writes only {KV_PREFIX}* keys, not {key!r}")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def record(cn: sqlite3.Connection, reading: ResonanceReading, instance: str = "system_entirety") -> dict:
    """Persist the latest reading for observability.  Read by nothing that gates."""
    payload = reading.to_json()
    _kv_set(cn, KV_PREFIX + instance, payload)
    return payload


__all__ = [
    "ENV", "THETA_ON_ENV", "THETA_OFF_ENV", "TAU_ENV", "KAPPA_ENV", "KV_PREFIX",
    "DEFAULT_THETA_ON", "DEFAULT_THETA_OFF", "BROADEN", "DEEPEN", "STATES",
    "ResonanceConfig", "Derivation", "ResonanceReading",
    "events_from_history", "session_break", "target_lift", "derive", "committed_bit", "phase_angle",
    "replay", "read", "wrap_bit_flip_parity", "record",
]
