"""interstitial — the arc between the three outputs, measured on the held residual.

The outputs and the sense axes they route to (observer_service.SOURCE_PROFILES):

    perceptopoly   → perception   spatial coordination — relational measurements
                                  taken from the observer's perspective
    loadopoly-ocr  → vision       unstructured observation (archival scans)
    bakugo         → touch        structured construction (cardcenter)

On the CAT ring (cat_residual._RING, SENSES order vision, touch, smell, body,
brain, perception) those three axes are contiguous:

        brain — perception — vision — touch — smell

so the outputs form an arc with vision as the hinge.  perception–vision and
vision–touch are ring edges; perception–touch is the chord.  Perceptopoly is
required to coincide with both Loadopoly-OCR and Bakugo.  Its coincidence with
OCR is a ring edge.  Its coincidence with Bakugo either passes through OCR
(mediated by vision) or exists directly across the chord.  The part that exists
directly — what the two edges do not explain — is the interstitial content.
That is the quantity this module measures.

Quantities, per checkpoint window, on the held residual h_s(k) per sense
--------------------------------------------------------------------------
    C_ab = Σ_k h_a(k)·conj(h_b(k)) / √(Σ_k|h_a(k)|² · Σ_k|h_b(k)|²)

is the complex coherence of a pair: |C_ab| ∈ [0, 1] is how locked the two axes
are to each other over the window, arg C_ab is the phase lag a→b.  It is
computed for the two edges (pv, vt) and the chord (pt).  Then

    mediated      = C_pv · C_vt                    the chord the two edges predict
    interstitial  = |C_pt − C_pv·C_vt| / (1 + |C_pt|)   ∈ [0, 1)

is the unmediated chord: zero when Perceptopoly's coincidence with Bakugo is
fully carried through OCR, larger when the two coincide directly, beyond what
passing through OCR explains.  ``arc_information_density`` is
1 − compressed/raw over the quantised arc trajectory — the paper's quantity
(Boger & Firestone, NHB 2026: cognition represents a type-independent
information density) — so a record also says whether the movement across the
three outputs was patterned or diffuse.

Lineage.  The third-order form |C₀₁·C₁₂ − C₀₂| / (1 + |C₀₂|) is
``ueqgm_engine.interstitial_entanglement_score``'s, and is credited to it.  It is
applied here to the outputs' residuals rather than to Weyl compression cycles,
and with no claim of entanglement — the quantity is an unmediated coherence, and
that is what it is called.

Physical frame
--------------
"Real physical space" enters from the outputs, not from the residual.  QUIPU
already grounds OCR detections in ENU metres (geospatial_relation) and turns
that geometry into Touch pressure.  Perceptopoly's relational measurements —
standoff, scale, coplanarity, bearing/range — are the observer's frame.  When a
client posts them on ``/observe`` as ``meta.frame``, observer_service persists
the latest frame per source at ``brain_kv["observer:frame:<source>"]`` and this
module attaches the three sources' latest frames to every arc record.  The
record then sits next to the physical coordinates it was measured under; with
no frames posted it says so rather than inventing any.

Recorded, not routed
--------------------
Nothing downstream reads these numbers.  The existing
``ueqgm:interstitial_entanglement → ie_multiplier`` path into mesh_slm's
learning rate is untouched.  Switching that path to this measurement would be
the first time qpsi influenced what the predictor learns; it is a realisation
and needs the grant like any other.  Stdlib only.
"""
from __future__ import annotations

import cmath
import json
import math
import time
import zlib
from dataclasses import dataclass, field, asdict
from typing import Mapping, Sequence

from .cat_residual import SENSES, _RING

# The three outputs, the axes they route to, and the arc they form on the ring.
OUTPUT_AXES: dict[str, str] = {
    "perceptopoly": "perception",
    "loadopoly-ocr": "vision",
    "bakugo": "touch",
}
ARC: tuple[str, str, str] = ("perception", "vision", "touch")
EDGES: tuple[tuple[str, str], tuple[str, str]] = (("perception", "vision"), ("vision", "touch"))
CHORD: tuple[str, str] = ("perception", "touch")
FRAME_KEY_PREFIX: str = "observer:frame:"
KV_PREFIX: str = "entirety:interstitial_arc:"
LINEAGE: str = ("third-order form from ueqgm_engine.interstitial_entanglement_score, "
                "applied to the outputs' held residuals; no entanglement claim")


def arc_is_contiguous() -> bool:
    """True iff the three output axes are consecutive on the CAT ring.
    Checked, not assumed: the edges must be ring edges and the chord must not."""
    edges_ok = all(b in _RING[a] for a, b in EDGES)
    chord_ok = CHORD[1] not in _RING[CHORD[0]]
    return edges_ok and chord_ok


@dataclass
class PairCoherence:
    a: str
    b: str
    coherence: float          # |C_ab|
    lag: float                # arg C_ab, radians (a → b)
    kind: str                 # "edge" | "chord"
    re: float = 0.0
    im: float = 0.0


@dataclass
class ArcMeasure:
    rows: int
    pairs: dict[str, PairCoherence]          # keys "pv", "vt", "pt"
    mediated_coherence: float                # |C_pv · C_vt|
    mediated_lag: float                      # arg(C_pv · C_vt)
    interstitial: float                      # |C_pt − C_pv·C_vt| / (1 + |C_pt|)
    information_density: float               # 1 − compressed/raw, clamped [0, 1]
    measured: bool
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["pairs"] = {k: asdict(v) for k, v in self.pairs.items()}
        return d


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

def coherence(rows: Sequence[Mapping], a: str, b: str) -> complex:
    """Complex coherence of the held residual on axes a and b over the window."""
    num = 0j
    pa = 0.0
    pb = 0.0
    for r in rows:
        held = r.get("held") or {}
        ha = complex(held.get(a, 0j))
        hb = complex(held.get(b, 0j))
        num += ha * hb.conjugate()
        pa += abs(ha) ** 2
        pb += abs(hb) ** 2
    denom = math.sqrt(pa * pb)
    return num / denom if denom > 0 else 0j


def _pair(rows: Sequence[Mapping], a: str, b: str, kind: str) -> PairCoherence:
    c = coherence(rows, a, b)
    return PairCoherence(a, b, abs(c), cmath.phase(c) if abs(c) > 0 else 0.0, kind, c.real, c.imag)


# ---------------------------------------------------------------------------
# Information density (the paper's quantity, on the arc trajectory)
# ---------------------------------------------------------------------------

def information_density(rows: Sequence[Mapping], senses: Sequence[str] = ARC) -> float:
    """1 − compressed/raw over the int8-quantised (Re, Im) trajectory of the given
    senses.  0 = incompressible (diffuse or too short to say); higher = patterned.
    Windows of a few dozen rows give a coarse number; it is a relative measure."""
    vals: list[float] = []
    for r in rows:
        held = r.get("held") or {}
        for s in senses:
            h = complex(held.get(s, 0j))
            vals.extend((h.real, h.imag))
    if not vals:
        return 0.0
    scale = max(abs(v) for v in vals) or 1.0
    raw = bytes(int(round(127 * v / scale)) & 0xFF for v in vals)
    comp = zlib.compress(raw, 9)
    # zlib carries a fixed ~11-byte envelope; discount it so short windows are not
    # penalised for the header alone.
    effective = max(1, len(comp) - 11)
    return max(0.0, min(1.0, 1.0 - effective / len(raw)))


# ---------------------------------------------------------------------------
# The arc measure
# ---------------------------------------------------------------------------

def arc_measure(rows: Sequence[Mapping], *, min_rows: int = 8, eta: float = 1e-6) -> ArcMeasure:
    rows = list(rows)
    reasons: list[str] = []
    empty = {k: PairCoherence(a, b, 0.0, 0.0, kind) for k, (a, b, kind) in
             {"pv": (*EDGES[0], "edge"), "vt": (*EDGES[1], "edge"), "pt": (*CHORD, "chord")}.items()}
    if not arc_is_contiguous():
        reasons.append("arc is not contiguous on the ring: SENSES order changed; measure withheld")
        return ArcMeasure(len(rows), empty, 0.0, 0.0, 0.0, 0.0, False, reasons)
    if len(rows) < min_rows:
        reasons.append(f"{len(rows)} rows < min_rows {min_rows}")
        return ArcMeasure(len(rows), empty, 0.0, 0.0, 0.0, 0.0, False, reasons)
    energy = {s: sum(abs(complex((r.get("held") or {}).get(s, 0j))) ** 2 for r in rows) for s in ARC}
    silent = [s for s in ARC if energy[s] < eta * eta]
    if silent:
        reasons.append(f"no held content on {', '.join(silent)}: coincidence undefined there")
    c_pv = coherence(rows, *EDGES[0])
    c_vt = coherence(rows, *EDGES[1])
    c_pt = coherence(rows, *CHORD)
    med = c_pv * c_vt
    inter = abs(c_pt - med) / (1.0 + abs(c_pt))
    pairs = {"pv": _pair(rows, *EDGES[0], "edge"), "vt": _pair(rows, *EDGES[1], "edge"),
             "pt": _pair(rows, *CHORD, "chord")}
    dens = information_density(rows)
    measured = not silent
    if measured:
        reasons.append(f"|C_pv|={abs(c_pv):.3f} |C_vt|={abs(c_vt):.3f} |C_pt|={abs(c_pt):.3f} "
                       f"mediated={abs(med):.3f} interstitial={inter:.3f} density={dens:.3f}")
    return ArcMeasure(len(rows), pairs, abs(med), cmath.phase(med) if abs(med) > 0 else 0.0,
                      inter, dens, measured, reasons)


# ---------------------------------------------------------------------------
# Physical frames (from the outputs, via observer_service) and the record
# ---------------------------------------------------------------------------

def physical_frames(cn) -> dict[str, dict | None]:
    """Latest observer frame per output, or None where none has been posted."""
    out: dict[str, dict | None] = {}
    for source in OUTPUT_AXES:
        try:
            row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (FRAME_KEY_PREFIX + source,)).fetchone()
            out[source] = json.loads(row[0]) if row and row[0] else None
        except Exception:
            out[source] = None
    return out


def record(cn, instance: str, rows: Sequence[Mapping], *, now: float | None = None) -> dict:
    """Measure the arc on this window, attach the outputs' latest physical frames,
    persist under entirety:interstitial_arc:<instance>, and return the record.
    Nothing reads it back but people and the CLI."""
    m = arc_measure(rows)
    frames = physical_frames(cn)
    rec = {"instance": instance, "at": time.time() if now is None else float(now),
           "arc": list(ARC), "outputs": dict(OUTPUT_AXES), "measure": m.to_json(),
           "frames": frames, "frames_present": [s for s, f in frames.items() if f],
           "lineage": LINEAGE}
    try:
        cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
        cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
                   (KV_PREFIX + instance, json.dumps(rec, default=str),
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    except Exception:
        pass
    return rec


__all__ = ["OUTPUT_AXES", "ARC", "EDGES", "CHORD", "FRAME_KEY_PREFIX", "KV_PREFIX", "LINEAGE",
           "arc_is_contiguous", "PairCoherence", "ArcMeasure", "coherence", "information_density",
           "arc_measure", "physical_frames", "record"]
