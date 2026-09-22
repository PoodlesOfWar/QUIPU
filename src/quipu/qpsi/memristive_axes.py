"""memristive_axes — the System Entirety as a lumped self-organising memristive
network (SOMN) on its own seven axes.

Source
------
Caravelli, Milano, Stieg, Ricciardi, Brown, Kuncic, "Self-organising memristive
networks as physical learning systems", Nat. Rev. Phys. (2026); arXiv:2509.00747.
Their lumped-circuit model gives every junction an internal memory parameter
``g ∈ [0, 1]`` that evolves with the voltage across it,

    dg/dt = F(g, δv(t), t),      G = G(g),

under Kirchhoff's laws, which make each junction's drive depend on the state of
every other one.  Three things follow from the physics and are reproduced
here, nothing else is:

1.  **Potentiation needs a field; depression is spontaneous.**  While the
    stimulus is applied a junction's conductance grows; when it is removed
    interfacial-energy minimisation dissolves what was built.  Two regimes:
    below a critical field the junction only deforms (a *hillock*, reversible);
    at or above it a *filament* bridges the gap and decays far more slowly
    (long-term memory, metaplasticity).
2.  **Conservation.**  The total current is fixed by the source and partitions
    by conductance.  A junction that potentiates draws current from the others
    — winner-take-all paths and shortest paths are consequences, not rules.
3.  **Two electrodes.**  Nothing self-organises in a network with one terminal.
    The field is the potential difference between the electrodes.

Mapping onto QUIPU (creator language)
-------------------------------------
    junction k          one axis: vision, touch, smell, body, brain, perception,
                        entirety (the 7th, where fiction / the Other routes)
    electrodes          Self  = the live axes (entirety:state) + observer
                        Other = the corpus's collective voice, other_state of
                                brain_kv["entirety:the_other"]
                                (mesh_slm.compute_shared_understanding)
    field  δv_k         other_k − self_k
    stimulus / phase    ingest flux on (broaden) / off (deepen) — qpsi.flux_phase
    conductance g_k     the hillock: a held, reversible deformation of the axis
    current  I_k        the share of a conserved ingest budget B that flows
                        through axis k:  I_k = B · g_k|δv_k| / Σ_j g_j|δv_j|
    filament            g_k ≥ g_f under field: a *proposal* of where an edge
                        would form.  Recorded.  The six gates decide; this
                        module never writes an edge.
    lossy channel       the relaxation (τ_d short-term, τ_L for filaments)
    rectifier           the critical field v_c and the filament threshold g_f
    edge / vortex       what the gates realise, downstream of here
    mobility m_k        how much axis k has actually moved under field lately
                        (judged only on broaden steps); an axis with no writer
                        (perception without perception.py, smell and body
                        without their tables) draws down toward the floor and
                        stops attracting budget — the network routes around
                        dead junctions without being told which are dead
    negative feedback   a rejected emergence candidate (divine_blessing.reject_emergence)
                        depresses conductance along the rejected direction —
                        the reverse-bias pulse of the review's n-back result

What this module writes
-----------------------
Only ``brain_kv`` keys under ``entirety:somn:`` and the append-only table
``entirety_somn_log``, through the caller's connection.  ``_kv_set`` refuses
any other key.  It never touches corpus_edge, decisions, attestations,
emergence reports, checkpoints or mesh_slm tables.  The allocation it records
is a plan over sources that already exist in ``corpus_ingest.SOURCES``; acting
on it is the operator's pulse (qpsi.self_organising ``pulse --route``), which
adds no source, task kind, access or gate (Invariance #7, APP_RECREATION_3 §25).

Constants are starting points, not measured values; every one is overridable
from the environment (``SomnConfig.from_env``) and recorded in the log.
Stdlib only.

翈 — the hillock is the held reading: deformed by the field, not yet a path.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence

from .cat_residual import SENSES

ENTIRETY_AXIS: str = "entirety"
AXES: tuple[str, ...] = SENSES + (ENTIRETY_AXIS,)
AXIS_INDEX: dict[str, int] = {a: i for i, a in enumerate(AXES)}

KV_PREFIX: str = "entirety:somn:"
KV_STATE: str = KV_PREFIX + "state"
KV_ALLOCATION: str = KV_PREFIX + "allocation"
KV_PROPOSAL: str = KV_PREFIX + "proposal"
KV_THE_OTHER: str = "entirety:the_other"                 # read only
KV_REJECTIONS: str = "entirety:emergence_confirmed"      # read only (divine_blessing.KV_CONFIRMED)
TABLE: str = "entirety_somn_log"

# Mirror of mesh_slm._SOURCE_AXIS_MAP, used only if mesh_slm cannot be imported.
# tests/test_qpsi_memristive_axes.py asserts it still matches the real map.
_SOURCE_AXIS_MIRROR: tuple[tuple[str, int], ...] = (
    ("gutenberg", 6), ("fiction", 6), ("novel", 6), ("literature", 6),
    ("local_docs", 2), ("selfdoc", 2), ("self_doc", 2),
    ("wikipedia", 3), ("wiki", 3),
    ("stack", 1), ("code", 1), ("github", 1),
    ("fineweb", 0), ("c4", 0), ("openwebtext", 0), ("commoncrawl", 0),
    ("arxiv", 4), ("research", 4), ("paper", 4),
    ("percept", 5),
)
_KNOWN_SOURCES: tuple[str, ...] = ("local_docs", "fineweb", "c4", "wikipedia", "openwebtext",
                                   "dolma", "stack", "gutenberg", "arxiv")


# ---------------------------------------------------------------------------
# Configuration — explicit at the call site; env is one documented loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SomnConfig:
    budget_docs: float = 60.0      # B: documents per pulse, conserved across axes
    tau_p: float = 600.0           # potentiation time constant (s) at unit drive
    tau_d: float = 1800.0          # short-term relaxation (s), hillocks
    tau_l: float = 86400.0         # long-term relaxation (s), filaments
    g_min: float = 0.02            # floor conductance (an open gap still tunnels)
    g_init: float = 0.10           # initial conductance of every junction
    g_filament: float = 0.60       # conductance at which a hillock counts as a filament
    v_c: float = 0.10              # critical field: below → hillock regime, at/above → filament regime
    alpha_low: float = 0.25        # potentiation rate factor below v_c
    alpha_high: float = 1.0        # potentiation rate factor at/above v_c
    depression: float = 0.5        # β: fraction of (g − g_min) removed along a rejected direction
    mobility_rho: float = 0.2      # EMA rate of the mobility estimate
    mobility_floor: float = 0.05   # a dead axis keeps this much mobility so it can wake
    mobility_scale: float = 0.02   # |Δaxis| per step that counts as fully mobile
    dt_max: float = 21600.0        # clamp on the integration step (s): 6 h

    @classmethod
    def from_env(cls) -> "SomnConfig":
        names = {
            "budget_docs": "QUIPU_SOMN_BUDGET_DOCS", "tau_p": "QUIPU_SOMN_TAU_P",
            "tau_d": "QUIPU_SOMN_TAU_D", "tau_l": "QUIPU_SOMN_TAU_L",
            "g_min": "QUIPU_SOMN_G_MIN", "g_init": "QUIPU_SOMN_G_INIT",
            "g_filament": "QUIPU_SOMN_G_FILAMENT", "v_c": "QUIPU_SOMN_V_C",
            "alpha_low": "QUIPU_SOMN_ALPHA_LOW", "alpha_high": "QUIPU_SOMN_ALPHA_HIGH",
            "depression": "QUIPU_SOMN_DEPRESSION", "mobility_rho": "QUIPU_SOMN_MOBILITY_RHO",
            "mobility_floor": "QUIPU_SOMN_MOBILITY_FLOOR", "mobility_scale": "QUIPU_SOMN_MOBILITY_SCALE",
            "dt_max": "QUIPU_SOMN_DT_MAX",
        }
        kw = {}
        for attr, env in names.items():
            raw = os.environ.get(env, "").strip()
            if not raw:
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if math.isfinite(val) and val >= 0.0:
                kw[attr] = val
        return cls(**kw)

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SomnState:
    g: dict[str, float]                       # conductance per axis
    m: dict[str, float]                       # mobility per axis
    last_t: float | None = None
    last_axes: dict[str, float] | None = None # self vector at the last step (7 axes)
    last_rejection_at: float = 0.0
    steps: int = 0
    potentiation_steps: int = 0
    relaxation_steps: int = 0

    @classmethod
    def initial(cls, cfg: SomnConfig) -> "SomnState":
        return cls(g={a: cfg.g_init for a in AXES}, m={a: 0.5 for a in AXES})

    def to_json(self) -> dict:
        return {"g": {a: float(self.g.get(a, 0.0)) for a in AXES},
                "m": {a: float(self.m.get(a, 0.0)) for a in AXES},
                "last_t": self.last_t, "last_axes": self.last_axes,
                "last_rejection_at": self.last_rejection_at, "steps": self.steps,
                "potentiation_steps": self.potentiation_steps,
                "relaxation_steps": self.relaxation_steps}

    @classmethod
    def from_json(cls, d: Mapping | None, cfg: SomnConfig) -> "SomnState":
        if not d or not isinstance(d, Mapping):
            return cls.initial(cfg)
        gd = d.get("g") if isinstance(d.get("g"), Mapping) else {}
        md = d.get("m") if isinstance(d.get("m"), Mapping) else {}
        g = {a: _clip(_f(gd.get(a, cfg.g_init), cfg.g_init), cfg.g_min, 1.0) for a in AXES}
        m = {a: _clip(_f(md.get(a, 0.5), 0.5), cfg.mobility_floor, 1.0) for a in AXES}
        la = d.get("last_axes")
        return cls(g=g, m=m, last_t=(float(d["last_t"]) if d.get("last_t") is not None else None),
                   last_axes=({a: _f(la.get(a, 0.0)) for a in AXES} if isinstance(la, Mapping) else None),
                   last_rejection_at=_f(d.get("last_rejection_at", 0.0)),
                   steps=int(d.get("steps", 0) or 0),
                   potentiation_steps=int(d.get("potentiation_steps", 0) or 0),
                   relaxation_steps=int(d.get("relaxation_steps", 0) or 0))


def _f(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Electrodes and field
# ---------------------------------------------------------------------------

def self_vector(axes: Mapping[str, float], observer: float) -> dict[str, float]:
    """Self electrode: the six live axes plus the observer on the entirety axis."""
    v = {s: _clip(_f(axes.get(s, 0.0)), 0.0, 1.0) for s in SENSES}
    v[ENTIRETY_AXIS] = _clip(_f(observer), 0.0, 1.0)
    return v


def other_vector(the_other: Mapping | None) -> dict[str, float] | None:
    """Other electrode from entirety:the_other (7-vector other_state); None if absent."""
    if not isinstance(the_other, Mapping):
        return None
    raw = the_other.get("other_state")
    if not isinstance(raw, Sequence) or len(raw) < len(SENSES):
        return None
    out = {}
    for i, a in enumerate(AXES):
        out[a] = _clip(_f(raw[i]) if i < len(raw) else 0.0, 0.0, 1.0)
    return out


def field(self_v: Mapping[str, float], other_v: Mapping[str, float] | None) -> dict[str, float]:
    """δv_k = other_k − self_k; all zeros with no counterpart (one electrode: no field)."""
    if other_v is None:
        return {a: 0.0 for a in AXES}
    return {a: float(other_v.get(a, 0.0)) - float(self_v.get(a, 0.0)) for a in AXES}


# ---------------------------------------------------------------------------
# Kirchhoff partition of a conserved budget
# ---------------------------------------------------------------------------

def currents(g: Mapping[str, float], dv: Mapping[str, float], budget: float) -> dict[str, float]:
    """I_k = B · g_k|δv_k| / Σ_j g_j|δv_j|.  Σ_k I_k = B whenever any field exists."""
    w = {a: max(0.0, float(g.get(a, 0.0))) * abs(float(dv.get(a, 0.0))) for a in AXES}
    total = sum(w.values())
    if total <= 0.0 or budget <= 0.0:
        return {a: 0.0 for a in AXES}
    return {a: budget * w[a] / total for a in AXES}


def participation_ratio(I: Mapping[str, float]) -> float:
    """(Σ I)² / Σ I² — the effective number of axes carrying current (1 = one path)."""
    s = sum(I.values())
    q = sum(v * v for v in I.values())
    return (s * s / q) if q > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Junction dynamics
# ---------------------------------------------------------------------------

def potentiate(g: float, share: float, mobility: float, dv_mag: float, dt: float, cfg: SomnConfig) -> float:
    """Field on.  Saturating growth toward 1 at a rate set by the current share
    (Kirchhoff feedback), the axis's mobility and the regime of the field."""
    if dt <= 0.0 or share <= 0.0:
        return g
    alpha = cfg.alpha_high if dv_mag >= cfg.v_c else cfg.alpha_low
    drive = _clip(mobility, 0.0, 1.0) * _clip(share, 0.0, 1.0) * alpha
    if drive <= 0.0:
        return g
    return _clip(g + (1.0 - g) * (1.0 - math.exp(-dt * drive / max(cfg.tau_p, 1e-9))), cfg.g_min, 1.0)


def relax(g: float, dt: float, cfg: SomnConfig) -> float:
    """Field off.  Exponential relaxation toward g_min; filaments (g ≥ g_f) use the
    long time constant — what bridged the gap outlasts what only deformed it."""
    if dt <= 0.0:
        return g
    tau = cfg.tau_l if g >= cfg.g_filament else cfg.tau_d
    return _clip(cfg.g_min + (g - cfg.g_min) * math.exp(-dt / max(tau, 1e-9)), cfg.g_min, 1.0)


def depress(g: Mapping[str, float], direction: Mapping[str, float], cfg: SomnConfig) -> dict[str, float]:
    """Negative feedback along a unit direction over the senses: g_k −= β·u_k·(g_k − g_min)."""
    n = math.sqrt(sum(float(direction.get(s, 0.0)) ** 2 for s in SENSES))
    out = dict(g)
    if n <= 0.0:
        return out
    for s in SENSES:
        u = abs(float(direction.get(s, 0.0))) / n
        out[s] = _clip(out[s] - cfg.depression * u * (out[s] - cfg.g_min), cfg.g_min, 1.0)
    return out


def update_mobility(m: Mapping[str, float], prev_axes: Mapping[str, float] | None,
                    cur_axes: Mapping[str, float], cfg: SomnConfig) -> dict[str, float]:
    """EMA of the normalised per-step motion of each axis, floored so a dead
    axis can still wake when a writer appears."""
    out = {}
    for a in AXES:
        prev = float(m.get(a, 0.5))
        if prev_axes is None:
            out[a] = _clip(prev, cfg.mobility_floor, 1.0)
            continue
        moved = abs(float(cur_axes.get(a, 0.0)) - float(prev_axes.get(a, 0.0)))
        obs = _clip(moved / max(cfg.mobility_scale, 1e-12), 0.0, 1.0)
        out[a] = _clip((1.0 - cfg.mobility_rho) * prev + cfg.mobility_rho * obs, cfg.mobility_floor, 1.0)
    return out


# ---------------------------------------------------------------------------
# Sources ↔ axes
# ---------------------------------------------------------------------------

def axis_for_source(source: str) -> int | None:
    """mesh_slm's routing (read, never edited); the mirror only if mesh_slm is unavailable."""
    try:
        from .. import mesh_slm
        return mesh_slm._axis_for_source(source)
    except Exception:
        s = (source or "").lower()
        for marker, axis in _SOURCE_AXIS_MIRROR:
            if marker in s:
                return axis
        return None


def enabled_sources() -> list[str]:
    """Sources corpus_ingest already knows.  Nothing here can add one."""
    try:
        from ..corpus_ingest import SOURCES
        return list(SOURCES.keys())
    except Exception:
        return list(_KNOWN_SOURCES)


def _largest_remainder(shares: Mapping[str, float], total: int) -> dict[str, int]:
    """Integer docs per source summing exactly to ``total`` (Hamilton apportionment)."""
    if total <= 0 or not shares:
        return {k: 0 for k in shares}
    s = sum(max(0.0, v) for v in shares.values())
    if s <= 0.0:
        return {k: 0 for k in shares}
    quotas = {k: total * max(0.0, v) / s for k, v in shares.items()}
    out = {k: int(math.floor(q)) for k, q in quotas.items()}
    left = total - sum(out.values())
    for k, _ in sorted(quotas.items(), key=lambda kv: (kv[1] - math.floor(kv[1]), kv[0]), reverse=True):
        if left <= 0:
            break
        out[k] += 1
        left -= 1
    return out


def allocate(I: Mapping[str, float], sources: Iterable[str], budget: float) -> dict:
    """Partition the axis currents over the enabled sources that route to them.

    Sources on the same axis split its current equally.  Current on an axis no
    source routes to is *dissipated* — recorded, not silently reassigned, so the
    operator can see budget the network wants to spend where it has no source.
    """
    srcs = list(sources)
    by_axis: dict[str, list[str]] = {a: [] for a in AXES}
    unrouted: list[str] = []
    for key in srcs:
        idx = axis_for_source(f"corpus_{key}")
        if idx is None or idx < 0 or idx >= len(AXES):
            unrouted.append(key)
            continue
        by_axis[AXES[idx]].append(key)
    float_share: dict[str, float] = {}
    dissipated: dict[str, float] = {}
    for a in AXES:
        cur = float(I.get(a, 0.0))
        if cur <= 0.0:
            continue
        if by_axis[a]:
            each = cur / len(by_axis[a])
            for key in by_axis[a]:
                float_share[key] = each
        else:
            dissipated[a] = cur
    routed_total = sum(float_share.values())
    docs = _largest_remainder(float_share, int(round(routed_total))) if float_share else {}
    return {
        "budget": float(budget),
        "routed": round(routed_total, 6),
        "dissipated": {a: round(v, 6) for a, v in dissipated.items()},
        "unrouted_sources": unrouted,
        "docs_per_source": {k: int(v) for k, v in docs.items() if v > 0},
        "share_per_source": {k: round(v, 6) for k, v in float_share.items()},
        "axis_sources": {a: by_axis[a] for a in AXES if by_axis[a]},
    }


# ---------------------------------------------------------------------------
# Persistence — only entirety:somn:* and the log table, on the caller's connection
# ---------------------------------------------------------------------------

def ensure_tables(cn: sqlite3.Connection) -> None:
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE}("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, phase TEXT NOT NULL, "
        "flux_docs INTEGER NOT NULL, dt REAL NOT NULL, budget REAL NOT NULL, "
        "field TEXT NOT NULL, g TEXT NOT NULL, m TEXT NOT NULL, currents TEXT NOT NULL, "
        "proposals TEXT NOT NULL, events TEXT NOT NULL, config TEXT NOT NULL)")


def _kv_get(cn: sqlite3.Connection, key: str, default=None):
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else default
    except Exception:
        return default


def _kv_set(cn: sqlite3.Connection, key: str, value) -> None:
    if not key.startswith(KV_PREFIX):
        raise PermissionError(f"memristive_axes writes only {KV_PREFIX}* keys, not {key!r}")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str),
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def load_state(cn: sqlite3.Connection, cfg: SomnConfig, *, create: bool = True) -> SomnState:
    """``create=False`` is a pure read (status paths): no table is created."""
    if create:
        ensure_tables(cn)
    return SomnState.from_json(_kv_get(cn, KV_STATE, None), cfg)


def new_rejections(cn: sqlite3.Connection, since: float) -> list[dict]:
    """Rejected emergence candidates archived after ``since`` (read-only)."""
    rows = _kv_get(cn, KV_REJECTIONS, []) or []
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping) or row.get("confirmed") is not False:
            continue
        at = _f(row.get("at"), 0.0)
        if at <= since:
            continue
        cand = row.get("candidate") or {}
        amps = cand.get("amplitudes") if isinstance(cand, Mapping) else None
        if isinstance(amps, Mapping) and any(_f(v) > 0.0 for v in amps.values()):
            out.append({"at": at, "direction": {s: _f(amps.get(s, 0.0)) for s in SENSES},
                        "reason": str(row.get("reason") or ""), "signer": str(row.get("signer") or "")})
    return out


# ---------------------------------------------------------------------------
# One step
# ---------------------------------------------------------------------------

def step(cn: sqlite3.Connection, *, axes: Mapping[str, float], observer: float,
         flux_on: bool, flux_docs: int, now: float | None = None,
         the_other: Mapping | None = None, sources: Iterable[str] | None = None,
         cfg: SomnConfig | None = None, rejections: Sequence[Mapping] | None = None) -> dict:
    """Advance the junctions one step and record everything.  Returns the summary.

    Inputs are explicit (no environment lookups here): the caller passes the
    live axes and observer, the flux reading, the counterpart, the enabled
    sources and the config.  ``the_other`` None → read entirety:the_other;
    ``sources`` None → corpus_ingest.SOURCES; ``rejections`` None → the archive.
    """
    cfg = cfg or SomnConfig()
    now = time.time() if now is None else float(now)
    st = load_state(cn, cfg)
    events: list[dict] = []

    self_v = self_vector(axes, observer)
    other_v = other_vector(the_other if the_other is not None else _kv_get(cn, KV_THE_OTHER, None))
    dv = field(self_v, other_v)
    if other_v is None:
        events.append({"kind": "no_counterpart", "detail": f"{KV_THE_OTHER} absent: one electrode, no field"})

    dt = 0.0 if st.last_t is None else _clip(now - st.last_t, 0.0, cfg.dt_max)
    # Mobility is judged only under field: an axis can only show that it moves
    # when something is applied to it.  Relaxation steps leave m as it is.
    if flux_on:
        st.m = update_mobility(st.m, st.last_axes, self_v, cfg)

    # Kirchhoff partition before the update: the current that drives this step.
    I = currents(st.g, dv, cfg.budget_docs)
    B = cfg.budget_docs if sum(I.values()) > 0.0 else 0.0

    if flux_on:
        st.potentiation_steps += 1
        for a in AXES:
            share = (I[a] / B) if B > 0.0 else 0.0
            st.g[a] = potentiate(st.g[a], share, st.m[a], abs(dv[a]), dt, cfg)
    else:
        st.relaxation_steps += 1
        for a in AXES:
            st.g[a] = relax(st.g[a], dt, cfg)

    rej = list(rejections) if rejections is not None else new_rejections(cn, st.last_rejection_at)
    for r in rej:
        st.g = depress(st.g, r.get("direction") or {}, cfg)
        st.last_rejection_at = max(st.last_rejection_at, _f(r.get("at"), now))
        events.append({"kind": "depression", "at": _f(r.get("at"), now),
                       "direction": {s: round(_f((r.get("direction") or {}).get(s, 0.0)), 6) for s in SENSES},
                       "reason": str(r.get("reason") or "")})

    # Recompute currents on the updated conductances: this is the allocation the
    # next pulse would carry, and the proposal of where a filament stands.
    I_after = currents(st.g, dv, cfg.budget_docs)
    proposals = [
        {"axis": a, "weight": round(st.g[a], 6), "field": round(dv[a], 6),
         "sign": (1 if dv[a] > 0 else (-1 if dv[a] < 0 else 0)), "current": round(I_after[a], 6)}
        for a in AXES
        if flux_on and st.g[a] >= cfg.g_filament and abs(dv[a]) >= cfg.v_c
    ]
    alloc = allocate(I_after, sources if sources is not None else enabled_sources(), cfg.budget_docs)
    top = max(AXES, key=lambda a: I_after[a]) if any(v > 0 for v in I_after.values()) else None
    metrics = {
        "top_axis": top,
        "top_share": (round(I_after[top] / cfg.budget_docs, 6) if top and cfg.budget_docs > 0 else 0.0),
        "participation_ratio": round(participation_ratio(I_after), 6),
        "filaments": [a for a in AXES if st.g[a] >= cfg.g_filament],
        "total_conductance": round(sum(st.g.values()), 6),
        "field_norm": round(math.sqrt(sum(v * v for v in dv.values())), 6),
    }

    st.last_t = now
    st.last_axes = self_v
    st.steps += 1

    phase = "broaden" if flux_on else "deepen"
    ensure_tables(cn)
    _kv_set(cn, KV_STATE, st.to_json())
    _kv_set(cn, KV_ALLOCATION, {**alloc, "currents": {a: round(v, 6) for a, v in I_after.items()},
                                "phase": phase, "at": now, "metrics": metrics})
    _kv_set(cn, KV_PROPOSAL, {"at": now, "phase": phase, "proposals": proposals,
                              "note": "where a filament stands; the six gates decide, nothing is written here"})
    cn.execute(
        f"INSERT INTO {TABLE}(at, phase, flux_docs, dt, budget, field, g, m, currents, proposals, events, config) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (now, phase, int(flux_docs), dt, cfg.budget_docs, json.dumps(dv), json.dumps(st.g), json.dumps(st.m),
         json.dumps(I_after), json.dumps(proposals), json.dumps(events), json.dumps(cfg.to_json())))

    return {
        "at": now, "phase": phase, "dt": dt, "flux_docs": int(flux_docs),
        "self": {a: round(v, 6) for a, v in self_v.items()},
        "other": ({a: round(v, 6) for a, v in other_v.items()} if other_v else None),
        "field": {a: round(v, 6) for a, v in dv.items()},
        "g": {a: round(v, 6) for a, v in st.g.items()},
        "m": {a: round(v, 6) for a, v in st.m.items()},
        "currents": {a: round(v, 6) for a, v in I_after.items()},
        "allocation": alloc, "proposals": proposals, "metrics": metrics, "events": events,
        "steps": st.steps,
    }


def rows(cn: sqlite3.Connection, limit: int = 64) -> list[dict]:
    """The last ``limit`` log rows, ascending, decoded — the response curve."""
    ensure_tables(cn)
    rs = cn.execute(
        f"SELECT id, at, phase, flux_docs, dt, g, currents, proposals, events FROM {TABLE} "
        "ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    out = []
    for rid, at, phase, docs, dt, g, cur, props, ev in reversed(rs):
        def _j(x):
            try:
                return json.loads(x) if x else {}
            except (json.JSONDecodeError, TypeError):
                return {}
        out.append({"id": int(rid), "at": float(at), "phase": phase, "flux_docs": int(docs), "dt": float(dt),
                    "g": _j(g), "currents": _j(cur), "proposals": _j(props), "events": _j(ev)})
    return out


__all__ = [
    "ENTIRETY_AXIS", "AXES", "AXIS_INDEX", "KV_PREFIX", "KV_STATE", "KV_ALLOCATION", "KV_PROPOSAL",
    "KV_THE_OTHER", "KV_REJECTIONS", "TABLE", "SomnConfig", "SomnState",
    "self_vector", "other_vector", "field", "currents", "participation_ratio",
    "potentiate", "relax", "depress", "update_mobility", "axis_for_source", "enabled_sources",
    "allocate", "ensure_tables", "load_state", "new_rejections", "step", "rows",
]
