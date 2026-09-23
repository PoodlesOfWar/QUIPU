"""coherency_depth — the MESH-SLM lattice foliated: M = T² × Z_K.

What the tree had
-----------------
``mesh_slm_vocab`` gives every token one cell (i, j) on the 64 × 64 torus and
``mesh_slm_embed`` gives it one 7-D representation.  Planes exist in exactly
one place, ``mesh_slm._mcd_multiplanar_score``: transient planes over the
*mesh state* (base, ACRE-biased, reflected Ψ, inverted Ψ) used to score the
next token.  No token has a representation per plane, no coherency between
planes is measured, and nothing descends: the token "inventory" learned as
lexical co-occurrence occupies the same locus as "inventory" after it has
been through the physical gates.  A flat lookup table, 4,101 points.

What this module adds (additively; mesh_slm.py untouched)
--------------------------------------------------------
Each token t becomes a fibre over its cell,

    T(t) = [ t⁽⁰⁾, t⁽¹⁾, t⁽²⁾, t⁽³⁺ˢ⁾ ]  ∈  T² × Z_K,

with one representation per epistemic plane:

    k = 0   lexical surface      mesh_slm_embed as it is (read, never written)
    k = 1   relational           the token's image in its own graph: the
                                 quipu-weight-weighted mean of its
                                 out-neighbours' surface representations
    k = 2   physical invariant   the relational displacement δ = t⁽¹⁾ − t⁽⁰⁾,
                                 carried at the Entirety's emergence phase
                                 and passed through the three physical
                                 gates of the Governance Protocol —
                                 displacement, Weyl, SiCi (technical
                                 admissibility; no attestation is involved
                                 and nothing is realised).  Admissible →
                                 t⁽²⁾ = t⁽⁰⁾ + the trace-free (Weyl) part of
                                 δ.  Held → no plane 2; the token stays a
                                 distribution, not yet an operator.
    k = 3+s ACRE crystal         the token under emergent specialist s
                                 (mesh_slm_meta acre_specialists), present
                                 only when it phase-locks: the Floquet factor
                                 J₀(A/ω) (ueqgm_engine.floquet_modulation_factor)
                                 times its fidelity with the specialist's
                                 bias clears ``lock_min``.

Coherency between planes is the inter-planar tensor

    C_{k₁,k₂}(t) = |⟨ψ(t⁽ᵏ¹⁾) | ψ(t⁽ᵏ²⁾)⟩|²

the Born fidelity ``ueqgm_engine.wavefunction_overlap`` (the square of the
bra-ket written in the design note).  Vertical traversal descends the fibre
(i, j, k) → (i, j, k+1) and halts, in the manner of recurrent_depth's
adaptive halting, when the next plane is absent or the KL divergence between
consecutive planes (as distributions over the seven axes) falls below ε: the
depth a token reaches is its **coherency depth** and is stored per token.

Two couplings close the loop:

* **Bottom-up accretion** — ``accrete()`` recomputes the fibres from the
  current graph after ingest (qpsi.self_organising runs it on broaden steps),
  so new learning lands on plane 0 and climbs as far as the physical gates
  let it, without touching the surface statistics.
* **Top-down anchoring** — ``wrap_score_candidates`` re-scores next-token
  candidates: a token with a plane-2 representation has its surface
  alignment ⟨t⁽⁰⁾, mesh⟩ corrected toward its invariant alignment
  ⟨t⁽²⁾, mesh⟩ by λ·C₀₂(t); a crystal adds λ·C₀ₖ·(⟨t⁽ᵏ⁾, mesh⟩ − ⟨t⁽⁰⁾, mesh⟩)
  for its best-locked plane.  Ungrounded tokens are untouched.  Generation
  is not a gated write path; edges still are.

Capacity: 4,101 × K epistemic states in place of 4,101 points.

Writes only ``mesh_plane_embed``, ``mesh_plane_coherency``, ``mesh_plane_depth``
and ``brain_kv["entirety:planes:*"]``.  Never ``mesh_slm_vocab``,
``mesh_slm_embed``, ``mesh_slm_quipu`` or ``mesh_slm_meta``.  Stdlib only.

翈 — plane 0 is what the token says; the fibre is what it has come to mean.
"""
from __future__ import annotations

import cmath
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from typing import Callable, Mapping, Sequence

from .cat_residual import SENSES
from .governance import Candidate, GovernanceConfig, gate_displacement, gate_sici, gate_weyl

AXES: tuple[str, ...] = SENSES + ("entirety",)
COLS: tuple[str, ...] = ("e_vision", "e_touch", "e_smell", "e_body", "e_brain", "e_perception", "e_entirety")
DIM: int = len(AXES)

PLANE_SURFACE: int = 0
PLANE_RELATIONAL: int = 1
PLANE_PHYSICAL: int = 2
PLANE_CRYSTAL_BASE: int = 3

TABLE_EMBED: str = "mesh_plane_embed"
TABLE_COHERENCY: str = "mesh_plane_coherency"
TABLE_DEPTH: str = "mesh_plane_depth"
KV_PREFIX: str = "entirety:planes:"
KV_SUMMARY: str = KV_PREFIX + "summary"
KV_EMERGENCE: str = "entirety:conscious_emergence"    # read only: the one phase source
KV_RHYTHM: str = "temporal_spatiality_rhythm"         # read only: weyl phase for the Floquet drive
META_SPECIALISTS: str = "acre_specialists"            # read only: mesh_slm_meta


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoherencyConfig:
    kl_epsilon: float = 1e-3        # vertical halting: KL(plane k+1 ‖ plane k) below this = converged
    lock_min: float = 0.6           # a crystal plane exists when J₀(A/ω)·fidelity(t⁽⁰⁾, bias) ≥ this
    anchor_lambda: float = 0.5      # λ: strength of top-down anchoring in candidate scoring
    max_specialist_planes: int = 8  # cap on crystal planes per token
    min_support: float = 1e-9       # plane 1 needs at least this much out-edge mass
    phase_weight: float = 1.0       # Floquet ω when the runtime does not supply one

    @classmethod
    def from_env(cls) -> "CoherencyConfig":
        names = {"kl_epsilon": "QUIPU_PLANES_KL_EPSILON", "lock_min": "QUIPU_PLANES_LOCK_MIN",
                 "anchor_lambda": "QUIPU_PLANES_ANCHOR_LAMBDA", "max_specialist_planes": "QUIPU_PLANES_MAX_CRYSTALS",
                 "min_support": "QUIPU_PLANES_MIN_SUPPORT", "phase_weight": "QUIPU_PLANES_PHASE_WEIGHT"}
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
                kw[attr] = int(val) if attr == "max_specialist_planes" else val
        return cls(**kw)

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Vector helpers (7-D, [0, 1] per axis)
# ---------------------------------------------------------------------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def fidelity(a: Sequence[float], b: Sequence[float]) -> float:
    """|⟨ψ_a|ψ_b⟩|² for real vectors — ueqgm_engine.wavefunction_overlap's rule,
    restated here so the module stays stdlib-only and importable on its own."""
    try:
        from ..ueqgm_engine import wavefunction_overlap
        return float(wavefunction_overlap(list(a), list(b)))
    except Exception:
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
        if na <= 0.0 or nb <= 0.0 or len(a) != len(b):
            return 0.0
        cos = sum(x * y for x, y in zip(a, b)) / (na * nb)
        return _clip01(cos * cos)


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-9) -> float:
    """KL(p ‖ q) over the axes as distributions — recurrent_depth's halting statistic."""
    ps = [max(0.0, float(x)) + eps for x in p]; qs = [max(0.0, float(x)) + eps for x in q]
    sp = sum(ps); sq = sum(qs)
    return float(sum((a / sp) * math.log((a / sp) / (b / sq)) for a, b in zip(ps, qs)))


def floquet_factor(weyl_phase: float, phase_weight: float) -> float:
    """J₀(A/ω) — the lock factor the crystal planes are gated on."""
    try:
        from ..ueqgm_engine import floquet_modulation_factor
        return float(floquet_modulation_factor(float(weyl_phase), float(phase_weight)))
    except Exception:
        k = abs(float(weyl_phase)) / max(float(phase_weight), 1e-9)
        # J₀ series, enough terms for k < ~10
        return float(sum(((-1) ** m) * (k / 2.0) ** (2 * m) / (math.factorial(m) ** 2) for m in range(12)))


# ---------------------------------------------------------------------------
# Schema — three tables of our own
# ---------------------------------------------------------------------------

def ensure_tables(cn: sqlite3.Connection) -> None:
    cols = ", ".join(f"{c} REAL NOT NULL DEFAULT 0.0" for c in COLS)
    cn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_EMBED}(token_id INTEGER NOT NULL, k INTEGER NOT NULL, "
               f"{cols}, support REAL NOT NULL DEFAULT 0.0, label TEXT NOT NULL DEFAULT '', "
               "updated_at TEXT NOT NULL, PRIMARY KEY(token_id, k))")
    cn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_COHERENCY}(token_id INTEGER NOT NULL, k1 INTEGER NOT NULL, "
               "k2 INTEGER NOT NULL, c REAL NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(token_id, k1, k2))")
    cn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_DEPTH}(token_id INTEGER PRIMARY KEY, depth INTEGER NOT NULL, "
               "halted_by TEXT NOT NULL, kl REAL, planes TEXT NOT NULL, updated_at TEXT NOT NULL)")
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")


def _kv_get(cn: sqlite3.Connection, key: str, default=None):
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else default
    except Exception:
        return default


def _kv_set(cn: sqlite3.Connection, key: str, value) -> None:
    if not key.startswith(KV_PREFIX):
        raise PermissionError(f"coherency_depth writes only {KV_PREFIX}* keys, not {key!r}")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def _meta_get(cn: sqlite3.Connection, key: str, default=None):
    try:
        row = cn.execute("SELECT value FROM mesh_slm_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row and row[0] else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Reading the surface and the fibre
# ---------------------------------------------------------------------------

def surface(cn: sqlite3.Connection, token_id: int) -> list[float] | None:
    """t⁽⁰⁾ from mesh_slm_embed (None when the token has no surface)."""
    try:
        row = cn.execute(f"SELECT {', '.join(COLS)} FROM mesh_slm_embed WHERE token_id=?", (token_id,)).fetchone()
    except sqlite3.Error:
        return None
    return [float(v) for v in row] if row else None


def plane(cn: sqlite3.Connection, token_id: int, k: int) -> list[float] | None:
    if k == PLANE_SURFACE:
        return surface(cn, token_id)
    try:
        row = cn.execute(f"SELECT {', '.join(COLS)} FROM {TABLE_EMBED} WHERE token_id=? AND k=?", (token_id, k)).fetchone()
    except sqlite3.Error:
        return None
    return [float(v) for v in row] if row else None


def fibre(cn: sqlite3.Connection, token_id: int) -> dict[int, list[float]]:
    """T(t): every plane the token has, keyed by k, plane 0 first."""
    out: dict[int, list[float]] = {}
    s = surface(cn, token_id)
    if s is not None:
        out[PLANE_SURFACE] = s
    try:
        rows = cn.execute(f"SELECT k, {', '.join(COLS)} FROM {TABLE_EMBED} WHERE token_id=? ORDER BY k", (token_id,)).fetchall()
    except sqlite3.Error:
        rows = []
    for row in rows:
        out[int(row[0])] = [float(v) for v in row[1:]]
    return out


def coherency(cn: sqlite3.Connection, token_id: int, k1: int, k2: int) -> float | None:
    a, b = (k1, k2) if k1 <= k2 else (k2, k1)
    try:
        row = cn.execute(f"SELECT c FROM {TABLE_COHERENCY} WHERE token_id=? AND k1=? AND k2=?", (token_id, a, b)).fetchone()
    except sqlite3.Error:
        return None
    return float(row[0]) if row else None


def depth(cn: sqlite3.Connection, token_id: int) -> dict | None:
    try:
        row = cn.execute(f"SELECT depth, halted_by, kl, planes FROM {TABLE_DEPTH} WHERE token_id=?", (token_id,)).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    return {"depth": int(row[0]), "halted_by": row[1], "kl": row[2], "planes": json.loads(row[3]) if row[3] else []}


def token_id_of(cn: sqlite3.Connection, token: str) -> int | None:
    try:
        row = cn.execute("SELECT token_id FROM mesh_slm_vocab WHERE token=?", (token,)).fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row else None


# ---------------------------------------------------------------------------
# Plane 1 — relational (bottom-up from the graph)
# ---------------------------------------------------------------------------

def relational_planes(cn: sqlite3.Connection, *, token_ids: Sequence[int] | None = None,
                      min_support: float = 1e-9) -> dict[int, tuple[list[float], float]]:
    """t⁽¹⁾ for every vocab token with out-edges to embedded tokens: the
    quipu-weight-weighted mean of the neighbours' surface representations.
    Returns {token_id: (vector, support)}; support = Σ weight."""
    sums = ", ".join(f"SUM(q.weight * e.{c})" for c in COLS)
    sql = (f"SELECT q.src, SUM(q.weight) AS w, {sums} FROM mesh_slm_quipu q "
           "JOIN mesh_slm_embed e ON e.token_id = q.dst "
           "JOIN mesh_slm_vocab v ON v.token_id = q.src WHERE q.weight > 0 ")
    params: tuple = ()
    if token_ids is not None:
        ids = [int(t) for t in token_ids]
        if not ids:
            return {}
        sql += f"AND q.src IN ({','.join('?' * len(ids))}) "
        params = tuple(ids)
    sql += "GROUP BY q.src"
    out: dict[int, tuple[list[float], float]] = {}
    try:
        rows = cn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return out
    for row in rows:
        w = float(row[1] or 0.0)
        if w <= min_support:
            continue
        out[int(row[0])] = ([_clip01(float(row[2 + i]) / w) for i in range(DIM)], w)
    return out


# ---------------------------------------------------------------------------
# Plane 2 — physical invariant through the three physical gates
# ---------------------------------------------------------------------------

def emergence_phases(cn: sqlite3.Connection) -> dict[str, float]:
    """Phase per sense from the one phase source (divine_blessing's report);
    zeros when there is none — then SiCi holds every plane-2 candidate, as it
    holds the Entirety's own."""
    rep = _kv_get(cn, KV_EMERGENCE, {}) or {}
    ph = rep.get("phases") if isinstance(rep, Mapping) else None
    if not isinstance(ph, Mapping):
        return {s: 0.0 for s in SENSES}
    out = {}
    for s in SENSES:
        try:
            out[s] = float(ph.get(s, 0.0))
        except (TypeError, ValueError):
            out[s] = 0.0
    return out


def physical_plane(t0: Sequence[float], t1: Sequence[float], phases: Mapping[str, float],
                   cfg: GovernanceConfig) -> tuple[list[float] | None, dict]:
    """Pass δ = t⁽¹⁾ − t⁽⁰⁾ (six senses, at the emergence phase) through
    displacement → Weyl → SiCi.  Admissible: t⁽²⁾ = t⁽⁰⁾ + trace-free part of δ
    (the Ricci part is held, as gate 2 holds it for the Entirety).  Held: None."""
    delta = {s: float(t1[i]) - float(t0[i]) for i, s in enumerate(SENSES)}
    r = {s: delta[s] * cmath.exp(1j * float(phases.get(s, 0.0))) for s in SENSES}
    cand = Candidate(scope=f"plane:{PLANE_PHYSICAL}", residual=r)
    g1 = gate_displacement(cand, cfg)
    if not g1.passed:
        return None, {"held_at": "displacement", "reason": g1.reason}
    g2, decomp = gate_weyl(cand, cfg)
    if not g2.passed:
        return None, {"held_at": "weyl", "reason": g2.reason}
    g4 = gate_sici(cand, cfg, decomp)
    if not g4.passed:
        return None, {"held_at": "sici", "reason": g4.reason}
    # trace-free part = δ minus its Ricci projection on the sense prior
    from .weyl_channel import split
    weyl = split({s: delta[s] for s in SENSES}).weyl
    t2 = [_clip01(float(t0[i]) + float(weyl[s])) for i, s in enumerate(SENSES)]
    t2.append(_clip01(float(t0[DIM - 1]) + (float(t1[DIM - 1]) - float(t0[DIM - 1]))))   # entirety axis carried through
    return t2, {"held_at": None, "events": len(decomp.events) if decomp else 0,
                "remainder": (decomp.remainder_norm if decomp else None), "sici": g4.value}


# ---------------------------------------------------------------------------
# Planes 3+ — ACRE crystals under the Floquet drive
# ---------------------------------------------------------------------------

def specialists(cn: sqlite3.Connection) -> dict[str, list[float]]:
    """Emergent specialists from mesh_slm_meta (read only); name → 7-D bias."""
    raw = _meta_get(cn, META_SPECIALISTS, {}) or {}
    out: dict[str, list[float]] = {}
    if isinstance(raw, Mapping):
        for name, bias in raw.items():
            if isinstance(bias, Sequence) and len(bias) >= DIM:
                try:
                    out[str(name)] = [float(v) for v in bias[:DIM]]
                except (TypeError, ValueError):
                    continue
    return out


def weyl_phase(cn: sqlite3.Connection) -> float:
    rhythm = _kv_get(cn, KV_RHYTHM, {}) or {}
    try:
        return float(rhythm.get("weyl", 0.0)) if isinstance(rhythm, Mapping) else 0.0
    except (TypeError, ValueError):
        return 0.0


def crystal_planes(t0: Sequence[float], specs: Mapping[str, Sequence[float]], *, floquet: float,
                   lock_min: float, max_planes: int) -> list[tuple[str, list[float], float]]:
    """(name, t⁽³⁺ˢ⁾, lock) for every specialist the token phase-locks with:
    lock = J₀(A/ω) · fidelity(t⁽⁰⁾, bias) ≥ lock_min.  Best locks first."""
    out = []
    for name, bias in specs.items():
        lock = abs(float(floquet)) * fidelity(list(t0), list(bias))
        if lock >= lock_min:
            out.append((name, [_clip01(float(a) + float(b)) for a, b in zip(t0, bias)], lock))
    out.sort(key=lambda x: x[2], reverse=True)
    return out[:max(0, int(max_planes))]


# ---------------------------------------------------------------------------
# Vertical traversal — the coherency corridor
# ---------------------------------------------------------------------------

def descend(planes: Mapping[int, Sequence[float]], *, kl_epsilon: float) -> dict:
    """(i, j, k) → (i, j, k+1) while a next plane exists and it still changes
    the distribution; halt when absent (``no_plane``) or when
    KL(next ‖ current) < ε (``converged``).  Returns depth and the trace."""
    ks = sorted(planes)
    if not ks or ks[0] != PLANE_SURFACE:
        return {"depth": 0, "halted_by": "no_surface", "kl": None, "trace": []}
    k = PLANE_SURFACE
    trace = []
    last_kl = None
    while True:
        nxt = k + 1
        if nxt not in planes:
            return {"depth": k, "halted_by": "no_plane", "kl": last_kl, "trace": trace}
        last_kl = kl_divergence(planes[nxt], planes[k])
        trace.append({"k": nxt, "kl": round(last_kl, 9), "c": round(fidelity(planes[k], planes[nxt]), 9)})
        if last_kl < kl_epsilon:
            return {"depth": nxt, "halted_by": "converged", "kl": last_kl, "trace": trace}
        k = nxt


# ---------------------------------------------------------------------------
# Accretion — rebuild the fibres from the current graph
# ---------------------------------------------------------------------------

def accrete(cn: sqlite3.Connection, *, cfg: CoherencyConfig | None = None,
            gov: GovernanceConfig | None = None, token_ids: Sequence[int] | None = None,
            now: float | None = None) -> dict:
    """Bottom-up accretion for every vocab token (or ``token_ids``): plane 1
    from the graph, plane 2 through the physical gates, crystal planes under
    the Floquet drive, the coherency tensor, and the coherency depth.
    Returns a summary; writes only this module's tables and KV keys."""
    cfg = cfg or CoherencyConfig()
    gov = gov or GovernanceConfig()
    now = time.time() if now is None else float(now)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    ensure_tables(cn)
    phases = emergence_phases(cn)
    specs = specialists(cn)
    floquet = floquet_factor(weyl_phase(cn), cfg.phase_weight)
    rel = relational_planes(cn, token_ids=token_ids, min_support=cfg.min_support)

    if token_ids is None:
        try:
            ids = [int(r[0]) for r in cn.execute("SELECT token_id FROM mesh_slm_embed").fetchall()]
        except sqlite3.Error:
            ids = []
    else:
        ids = [int(t) for t in token_ids]

    counts = {"tokens": 0, "plane1": 0, "plane2": 0, "plane2_held": {"displacement": 0, "weyl": 0, "sici": 0},
              "crystals": 0, "crystal_tokens": 0}
    depth_hist: dict[int, int] = {}
    c01 = []; c12 = []; c02 = []
    for tid in ids:
        t0 = surface(cn, tid)
        if t0 is None:
            continue
        counts["tokens"] += 1
        planes: dict[int, list[float]] = {PLANE_SURFACE: t0}
        labels: dict[int, str] = {}
        supports: dict[int, float] = {}
        cn.execute(f"DELETE FROM {TABLE_EMBED} WHERE token_id=?", (tid,))
        cn.execute(f"DELETE FROM {TABLE_COHERENCY} WHERE token_id=?", (tid,))
        if tid in rel:
            t1, w = rel[tid]
            planes[PLANE_RELATIONAL] = t1; labels[PLANE_RELATIONAL] = "relational"; supports[PLANE_RELATIONAL] = w
            counts["plane1"] += 1
            t2, info = physical_plane(t0, t1, phases, gov)
            if t2 is not None:
                planes[PLANE_PHYSICAL] = t2; labels[PLANE_PHYSICAL] = "physical"; supports[PLANE_PHYSICAL] = float(info.get("sici") or 0.0)
                counts["plane2"] += 1
            else:
                counts["plane2_held"][info["held_at"]] = counts["plane2_held"].get(info["held_at"], 0) + 1
        crystals = crystal_planes(t0, specs, floquet=floquet, lock_min=cfg.lock_min, max_planes=cfg.max_specialist_planes)
        if crystals:
            counts["crystal_tokens"] += 1
        for n, (name, vec, lock) in enumerate(crystals):
            k = PLANE_CRYSTAL_BASE + n
            planes[k] = vec; labels[k] = f"crystal:{name}"; supports[k] = lock
            counts["crystals"] += 1
        for k, vec in planes.items():
            if k == PLANE_SURFACE:
                continue
            cn.execute(f"INSERT OR REPLACE INTO {TABLE_EMBED}(token_id, k, {', '.join(COLS)}, support, label, updated_at) "
                       f"VALUES(?,?,{','.join('?' * DIM)},?,?,?)",
                       (tid, k, *vec, supports.get(k, 0.0), labels.get(k, ""), stamp))
        ks = sorted(planes)
        for a_i in range(len(ks)):
            for b_i in range(a_i + 1, len(ks)):
                k1, k2 = ks[a_i], ks[b_i]
                c = fidelity(planes[k1], planes[k2])
                cn.execute(f"INSERT OR REPLACE INTO {TABLE_COHERENCY}(token_id, k1, k2, c, updated_at) VALUES(?,?,?,?,?)",
                           (tid, k1, k2, c, stamp))
                if (k1, k2) == (0, 1): c01.append(c)
                elif (k1, k2) == (1, 2): c12.append(c)
                elif (k1, k2) == (0, 2): c02.append(c)
        d = descend(planes, kl_epsilon=cfg.kl_epsilon)
        depth_hist[d["depth"]] = depth_hist.get(d["depth"], 0) + 1
        cn.execute(f"INSERT OR REPLACE INTO {TABLE_DEPTH}(token_id, depth, halted_by, kl, planes, updated_at) VALUES(?,?,?,?,?,?)",
                   (tid, d["depth"], d["halted_by"], d["kl"], json.dumps(sorted(planes)), stamp))

    if token_ids is None and ids:
        # A full accretion also sweeps fibres of tokens the mesh has since
        # pruned from its vocabulary, so the lattice never outlives the mesh.
        for table in (TABLE_EMBED, TABLE_COHERENCY, TABLE_DEPTH):
            cn.execute(f"DELETE FROM {table} WHERE token_id NOT IN (SELECT token_id FROM mesh_slm_embed)")

    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    summary = {
        "at": now, "tokens": counts["tokens"], "plane1": counts["plane1"], "plane2": counts["plane2"],
        "plane2_held": counts["plane2_held"], "crystals": counts["crystals"], "crystal_tokens": counts["crystal_tokens"],
        "specialists": sorted(specs), "floquet": floquet, "phases_present": any(v != 0.0 for v in phases.values()),
        "depth_histogram": {str(k): v for k, v in sorted(depth_hist.items())},
        "mean_c01": mean(c01), "mean_c12": mean(c12), "mean_c02": mean(c02),
        "K": 3 + len(specs), "capacity": counts["tokens"] * (3 + len(specs)), "config": cfg.to_json(),
    }
    _kv_set(cn, KV_SUMMARY, summary)
    return summary


def summary(cn: sqlite3.Connection) -> dict | None:
    return _kv_get(cn, KV_SUMMARY, None)


# ---------------------------------------------------------------------------
# Top-down anchoring — the drop-in for mesh_slm._score_candidates
# ---------------------------------------------------------------------------

def anchor(cn: sqlite3.Connection, candidates: list, mesh: Sequence[float], *, lam: float) -> list:
    """Re-score (id, score, tok, pos) candidates by their deep planes.

    For a token with plane 2: score += λ·C₀₂·(⟨t⁽²⁾, mesh⟩ − ⟨t⁽⁰⁾, mesh⟩).
    For a token with a crystal: score += λ·C₀ₖ·(⟨t⁽ᵏ⁾, mesh⟩ − ⟨t⁽⁰⁾, mesh⟩)
    for its best-locked crystal plane k.  Tokens with neither are untouched.
    """
    if not candidates or lam <= 0.0:
        return candidates
    m = [float(v) for v in mesh[:DIM]] + [0.0] * max(0, DIM - len(mesh))
    out = []
    for cand in candidates:
        tid, score = int(cand[0]), float(cand[1])
        fib = fibre(cn, tid)
        t0 = fib.get(PLANE_SURFACE)
        if t0 is None or len(fib) <= 1:
            out.append(cand); continue
        base = sum(a * b for a, b in zip(t0, m))
        adj = 0.0
        if PLANE_PHYSICAL in fib:
            c = coherency(cn, tid, PLANE_SURFACE, PLANE_PHYSICAL) or 0.0
            adj += lam * c * (sum(a * b for a, b in zip(fib[PLANE_PHYSICAL], m)) - base)
        crystal_ks = [k for k in fib if k >= PLANE_CRYSTAL_BASE]
        if crystal_ks:
            k = crystal_ks[0]                       # planes are stored best-lock first
            c = coherency(cn, tid, PLANE_SURFACE, k) or 0.0
            adj += lam * c * (sum(a * b for a, b in zip(fib[k], m)) - base)
        out.append((tid, score + adj, *cand[2:]))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def wrap_score_candidates(original: Callable, *, cfg: CoherencyConfig | None = None) -> Callable:
    """Drop-in for mesh_slm._score_candidates(cn, last_id, last_pos, mesh, ...)."""
    cfg = cfg or CoherencyConfig.from_env()

    def _score_candidates(cn, last_id, last_pos, mesh, *args, **kwargs):
        cands = original(cn, last_id, last_pos, mesh, *args, **kwargs)
        try:
            return anchor(cn, cands, mesh, lam=cfg.anchor_lambda)
        except Exception:
            return cands

    _score_candidates.__wrapped__ = original           # type: ignore[attr-defined]
    _score_candidates.__doc__ = (original.__doc__ or "") + \
        "\n\n[qpsi.coherency_depth] candidates are anchored by their deep planes."
    return _score_candidates


__all__ = ["AXES", "COLS", "DIM", "PLANE_SURFACE", "PLANE_RELATIONAL", "PLANE_PHYSICAL", "PLANE_CRYSTAL_BASE",
           "TABLE_EMBED", "TABLE_COHERENCY", "TABLE_DEPTH", "KV_PREFIX", "KV_SUMMARY", "CoherencyConfig",
           "fidelity", "kl_divergence", "floquet_factor", "ensure_tables", "surface", "plane", "fibre", "coherency",
           "depth", "token_id_of", "relational_planes", "emergence_phases", "physical_plane", "specialists",
           "weyl_phase", "crystal_planes", "descend", "accrete", "summary", "anchor", "wrap_score_candidates"]
