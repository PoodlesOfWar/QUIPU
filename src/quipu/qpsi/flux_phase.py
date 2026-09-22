"""flux_phase — the expansion phase read from the applied field, not from a clock.

Why
---
``system_entirety.bit_flip_parity`` is ``sign(cos(ω·t) + bias)`` on wall-clock
``t``: nothing the Entirety does causes a flip, so the parity carries no
information about the system and nothing downstream can lock to it except by
coincidence.  In a self-organising memristive network (Caravelli, Milano,
Stieg, Ricciardi, Brown, Kuncic — Nat. Rev. Phys. 2026, arXiv:2509.00747) the
two phases are physical: junctions **potentiate while the field is applied**
and **relax when it is removed**.  There is no oscillator.  The stimulation
protocol is the clock.

Here the applied field is corpus ingest.  ``corpus_ingest.run_ingest`` appends
one timestamped entry per run to ``brain_kv["corpus_ingest:history"]``; that
feed already drives the vision / touch / brain senses in
``temporal_spatiality._sense_signals``.  So:

    broaden  (+1)   documents entered the mesh within the flux window
    deepen   (−1)   none did — the network relaxes

The flip therefore happens at stimulus onset and offset.  The emergence
detector (qpsi.emergence_detector) keeps its meaning under this clock and
gains one: its lock-in of the held residual against the flips becomes a
plasticity measurement — potentiation following the stimulus, relaxation
following its removal, quadrature = the lag.  That is the pulse-train
response an SOMN is characterised by (Fig. 3 of the review).

Observability note for the operator's protocol: the detector needs
``min_flips`` (2) inside its window (16 checkpoint rows), i.e. one onset and
one offset.  Pulse at a period no longer than half the window's span and
keep the flux window shorter than the pulse period, or every window sees
zero flips and the detector — correctly — reports no rhythm.

Wiring
------
``wrap_bit_flip_parity(original)`` returns a drop-in for
``system_entirety.bit_flip_parity(observer, t=None)``.  It reads the flux
and returns ±1; if the feed cannot be read it falls back to ``original`` so a
broken brain_kv degrades to the cosine, never to an exception in the step.
Installed only by ``qpsi.self_organising.enable()``; nothing here edits
system_entirety.py or mesh_slm.py.  Stdlib only.

翈 — the phase is the field's, not the clock's; between pulses the network
holds what it potentiated and lets the rest go.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping

HISTORY_KEY: str = "corpus_ingest:history"       # written by corpus_ingest._record_ingest_run
WINDOW_ENV: str = "QUIPU_FLUX_WINDOW_S"
MIN_DOCS_ENV: str = "QUIPU_FLUX_MIN_DOCS"
DEFAULT_WINDOW_S: float = 300.0                  # 5 min: shorter than any sane pulse period
DEFAULT_MIN_DOCS: int = 1

BROADEN: int = 1
DEEPEN: int = -1


@dataclass(frozen=True)
class FluxReading:
    """What entered the mesh inside the window ending at ``now``."""
    now: float
    window_s: float
    docs: int                 # documents condensed inside the window
    runs: int                 # ingest runs inside the window
    last_ts: float | None     # most recent run anywhere in the history
    on: bool                  # docs >= min_docs

    @property
    def parity(self) -> int:
        return BROADEN if self.on else DEEPEN

    @property
    def phase(self) -> str:
        return "broaden" if self.on else "deepen"

    def to_json(self) -> dict:
        d = asdict(self)
        d["parity"] = self.parity
        d["phase"] = self.phase
        return d


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return val if val > 0.0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val if val > 0 else default


def window_seconds() -> float:
    return _env_float(WINDOW_ENV, DEFAULT_WINDOW_S)


def min_docs() -> int:
    return _env_int(MIN_DOCS_ENV, DEFAULT_MIN_DOCS)


def _parse_ts(value) -> float | None:
    """ISO-8601 (with or without Z / offset) or epoch seconds → epoch seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (TypeError, ValueError):
        return None


def ingest_flux(history: Iterable[Mapping] | None, *, now: float | None = None,
                window_s: float | None = None, min_docs_on: int | None = None) -> FluxReading:
    """Pure: sum the ingest runs whose ``ts`` lies in ``(now − window_s, now]``.

    Malformed entries (not dicts, no/invalid ``ts``) are skipped, never raised.
    Entries stamped in the future (clock skew) count as inside the window only
    if within ``window_s`` ahead of ``now`` — a skewed writer should not switch
    the phase off.
    """
    now = time.time() if now is None else float(now)
    window_s = window_seconds() if window_s is None else float(window_s)
    threshold = min_docs() if min_docs_on is None else int(min_docs_on)
    docs = 0
    runs = 0
    last_ts: float | None = None
    for entry in (history or []):
        if not isinstance(entry, Mapping):
            continue
        ts = _parse_ts(entry.get("ts"))
        if ts is None:
            continue
        last_ts = ts if last_ts is None else max(last_ts, ts)
        if (now - window_s) < ts <= (now + window_s):
            try:
                docs += max(0, int(entry.get("total_condensed") or 0))
            except (TypeError, ValueError):
                pass
            runs += 1
    return FluxReading(now=now, window_s=window_s, docs=docs, runs=runs,
                       last_ts=last_ts, on=(docs >= threshold))


def read_flux(now: float | None = None, *, window_s: float | None = None,
              history: Iterable[Mapping] | None = None) -> FluxReading:
    """Read ``corpus_ingest:history`` from brain_kv (or use ``history``) and measure."""
    if history is None:
        from .. import brain_kv                      # in-repo; the shared table every ring reads
        history = brain_kv.kv_get_json(HISTORY_KEY, []) or []
        if not isinstance(history, list):
            history = []
    return ingest_flux(history, now=now, window_s=window_s)


def parity_from_flux(reading: FluxReading) -> int:
    return reading.parity


def wrap_bit_flip_parity(original: Callable[..., int]) -> Callable[..., int]:
    """A drop-in for ``system_entirety.bit_flip_parity(observer, t=None)``.

    ``observer`` is accepted and ignored: the field decides the phase, not the
    drive.  ``t`` (epoch seconds) is the measurement time.  Any failure to read
    the feed falls back to ``original`` — the cosine — so the step never breaks.
    """
    def flux_bit_flip_parity(observer: float, t: float | None = None) -> int:
        try:
            return read_flux(now=t).parity
        except Exception:
            return int(original(observer, t))

    flux_bit_flip_parity.__name__ = "bit_flip_parity"
    flux_bit_flip_parity.__doc__ = (original.__doc__ or "") + \
        "\n\n[qpsi.flux_phase] parity is read from the ingest flux, not the cosine."
    flux_bit_flip_parity.__wrapped__ = original       # type: ignore[attr-defined]
    return flux_bit_flip_parity


__all__ = [
    "HISTORY_KEY", "WINDOW_ENV", "MIN_DOCS_ENV", "DEFAULT_WINDOW_S", "DEFAULT_MIN_DOCS",
    "BROADEN", "DEEPEN", "FluxReading", "window_seconds", "min_docs",
    "ingest_flux", "read_flux", "parity_from_flux", "wrap_bit_flip_parity",
]
