"""deepmap — information theory along deep (multi-hop) MESH pathways.

A *DeepMap pathway* is a walk of up to ``MAX_DEPTH`` hops through the Quipu
graph (``mesh_slm_quipu``: the GNN message edges), seeded at the proposed
shadow tokens (or, with none yet, the most frequent tokens), chosen by beam
search on the GNN message distribution.  Proposed ``new`` shadow edges may be
walked too; they are flagged and counted, never treated as real.

Every hop u → v carries the system's information-theoretic aspects, and each
aspect composes along the path by the rule its theory gives:

| aspect | per hop | along the path |
|---|---|---|
| Shannon channel | surprisal −log₂ p(v\\|u), p = w_uv / Σ_out w_u; branching entropy H(out u) | surprisal adds; entropy averaged |
| GARD resolvability | resolvable bits log₂(√n / 2√3) from the CRB step δ* = 2√3σ/√n (σ = 1, n = the edge's samples); the surprisal above it is *interstitial* | both add |
| magic | plane mix π ∝ \\|e_v − e_u\\| (the direction travelled), M₂ = Σ π_k M₂,k from ``entirety:mesh_flow:magic`` | M₂ is additive over tensor products, so it adds; stabilizer purity 2^{−M₂} multiplies; the monotone floor is the smallest hop (Leone & Bittel) |
| design | — | D = (1 − Π 2^{−M₂}) / (1 − (7/16)^L), normalised at L hops of the proved two-qubit maximum |
| positron / CPT | held share and asymmetry of the hop's planes | held multiplies (every hop must be held); asymmetry is the largest |
| spikes (poADAM vs rADAM) | — | delivered bits = surprisal · S(Δ)/φ, the mesh flow's stability number |
| infodynamics | — | Landauer energy k_B T ln2 per bit and its mass-energy equivalent E/c² (Vopson's mass–energy–information principle, contested) |

Path gain g = (1 + (φ − 1)·D) · held ∈ [0, φ].

**Perceptopoly's supernova events.**  Each TheSeuScape NPC posts its latest
``perceptopoly.supernova/1`` record through ``/relational``: a magnitude
m ∈ [0, 1] and a direction θ ∈ [0, 2π) for its local expansion.  The DeepMap
connects them both ways:

* *Events → pathways.*  A fresh event (age −60 s … 3600 s) weighs
  w = e^{−age/1800}·m (the same freshness rule as the held ACRE tool's gate).
  Its direction is read on the seven axis planes placed evenly round the
  circle, ν_k ∝ exp(κ cos(θ − 2πk/7)) with κ = 2 (a declared von Mises
  mapping).  A path's alignment with an event is the cosine between its plane
  mix and ν, its supernova pressure is s = Σ w·a / max(1, Σ w), and its
  expansion gain is min(φ, g·(1 + (φ − 1)·s)).  Paths are ranked by expansion
  gain, which equals g when no event is fresh.
* *Pathways → events.*  Each fresh NPC gets the three pathways its expansion
  would travel (largest w·a·g), with their information account: the answer
  Perceptopoly reads back on ``/deepmap`` and ``/guidance`` → ``deepmap.supernova``.

Reads the MESH tables and ``entirety:mesh_flow:*``; writes only
``brain_kv["entirety:deepmap:*"]``.  Never writes ``mesh_slm_*``.  The HubCore
``x_DeepMap`` registry was not available to the session that wrote this; this
module is the QUIPU side of the pathway map, served read-only at ``GET /deepmap``.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Any, Mapping, Optional, Sequence

KV_PREFIX: str = "entirety:deepmap:"
KV_LATEST: str = KV_PREFIX + "latest"

AXES: tuple[str, ...] = ("vision", "touch", "smell", "body", "brain", "perception", "entirety")
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
MAX_DEPTH: int = 4
BEAM: int = 3
FANOUT: int = 16
MAX_SEEDS: int = 8
MAX_PATHS: int = 16
K_BOLTZMANN: float = 1.380649e-23
C_LIGHT: float = 299_792_458.0
TEMPERATURE_K: float = 300.0
PURITY_AT_MAX: float = 7.0 / 16.0
SUPERNOVA_TAU_S: float = 1800.0
SUPERNOVA_WINDOW_S: float = 3600.0
SUPERNOVA_KAPPA: float = 2.0
PATHS_PER_NPC: int = 3
MAX_NPCS_OUT: int = 32
_TINY: frozenset = frozenset({"landauer_j", "info_mass_kg", "probability"})


def _kv_set(cn: sqlite3.Connection, key: str, value: Any) -> None:
    if not key.startswith(KV_PREFIX):
        raise PermissionError(f"deepmap writes only {KV_PREFIX}* keys, not {key!r}")
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def _kv_get(cn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return default
    try:
        return json.loads(row[0]) if row else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- per-hop aspects

def resolvable_bits(samples: float) -> float:
    """log₂(1/δ*) with δ* = 2√3/√n: the bits a CRB-limited estimator resolves from n observations."""
    n = max(1.0, float(samples or 1.0))
    delta = 2.0 * math.sqrt(3.0) / math.sqrt(n)
    return max(0.0, math.log2(1.0 / delta)) if delta < 1.0 else 0.0


def branching_entropy(weights: Sequence[float]) -> float:
    w = [max(0.0, float(x)) for x in weights]
    s = sum(w)
    if s <= 0.0:
        return 0.0
    return -sum((x / s) * math.log2(x / s) for x in w if x > 0.0)


def hop_planes(e_u: Optional[Sequence[float]], e_v: Optional[Sequence[float]]) -> list[float]:
    """π ∝ |e_v − e_u| (the direction travelled), else |e_u| + |e_v|, else uniform."""
    if e_u is None or e_v is None:
        return [1.0 / 7.0] * 7
    d = [abs(float(b) - float(a)) for a, b in zip(e_u, e_v)]
    if sum(d) <= 0.0 or not all(math.isfinite(x) for x in d):
        d = [abs(float(a)) + abs(float(b)) for a, b in zip(e_u, e_v)]
    s = sum(d)
    return [x / s for x in d] if s > 0.0 and math.isfinite(s) else [1.0 / 7.0] * 7


def hop_magic(planes: Sequence[float], plane_magic: Mapping[str, Mapping]) -> dict[str, float]:
    out = {"magic": 0.0, "held_num": 0.0, "held_den": 0.0, "cpt_asymmetry": 0.0}
    for a, pk in zip(AXES, planes):
        rec = plane_magic.get(a) or {}
        out["magic"] += pk * float(rec.get("magic", 0.0) or 0.0)
        mat, anti, un = (float(rec.get(k, 0.0) or 0.0) for k in ("matter", "antimatter", "unheld"))
        out["held_num"] += pk * (mat + anti)
        out["held_den"] += pk * (mat + anti + un)
        out["cpt_asymmetry"] = max(out["cpt_asymmetry"], abs(float(rec.get("cpt_asymmetry", 0.0) or 0.0)))
    held = out["held_num"] / out["held_den"] if out["held_den"] > 1e-15 else 1.0
    return {"magic": out["magic"], "held": held, "cpt_asymmetry": out["cpt_asymmetry"]}


# ---------------------------------------------------------------- path composition

def compose(hops: Sequence[Mapping[str, Any]], *, stability_number: float = PHI,
            temperature_k: float = TEMPERATURE_K) -> dict[str, Any]:
    """Fold per-hop aspects into the path's information account (see module table)."""
    L = len(hops)
    if L == 0:
        return {"hops": 0}
    surprisal = sum(h["surprisal"] for h in hops)
    resolvable = sum(min(h["surprisal"], h["resolvable"]) for h in hops)
    interstitial = sum(max(0.0, h["surprisal"] - h["resolvable"]) for h in hops)
    magic = sum(h["magic"] for h in hops)
    purity = 1.0
    held = 1.0
    for h in hops:
        purity *= 2.0 ** (-max(0.0, h["magic"]))
        held *= h["held"]
    design = min(1.0, (1.0 - purity) / (1.0 - PURITY_AT_MAX ** L))
    s = min(PHI, max(0.0, float(stability_number)))
    energy = K_BOLTZMANN * float(temperature_k) * math.log(2.0) * surprisal
    planes = [sum(h["planes"][k] for h in hops) / L for k in range(7)]
    return {
        "hops": L,
        "surprisal_bits": surprisal,
        "probability": 2.0 ** (-surprisal),
        "branching_entropy_bits": sum(h["entropy"] for h in hops) / L,
        "resolvable_bits": resolvable,
        "interstitial_bits": interstitial,
        "magic": magic,
        "magic_floor": min(h["magic"] for h in hops),
        "stabilizer_purity": purity,
        "design": design,
        "held": held,
        "cpt_asymmetry": max(h["cpt_asymmetry"] for h in hops),
        "delivered_bits": surprisal * s / PHI,
        "landauer_j": energy,
        "info_mass_kg": energy / (C_LIGHT * C_LIGHT),
        "gain": (1.0 + (PHI - 1.0) * design) * held,
        "plane": AXES[max(range(7), key=lambda k: planes[k])],
        "planes": [round(x, 6) for x in planes],
        "shadow_hops": sum(1 for h in hops if h.get("shadow")),
    }


# ---------------------------------------------------------------- graph access

class _Graph:
    def __init__(self, cn: sqlite3.Connection, shadow_edges: Sequence[Mapping[str, Any]]):
        self.cn = cn
        self._out: dict[int, tuple[list[tuple[int, float, float, bool]], float]] = {}
        self._emb: dict[int, Optional[list[float]]] = {}
        self._tok: dict[int, str] = {}
        self._id: dict[str, int] = {}
        self.shadow: dict[int, list[tuple[int, float]]] = {}
        for e in shadow_edges:
            if e.get("kind") != "new":
                continue
            a, b = self.token_id(e.get("src")), self.token_id(e.get("dst"))
            if a is None or b is None:
                continue
            lift = float(e.get("lift", 0.0) or 0.0)
            self.shadow.setdefault(a, []).append((b, lift))
            self.shadow.setdefault(b, []).append((a, lift))

    def token_id(self, token: Any) -> Optional[int]:
        if token is None:
            return None
        t = str(token)
        if t not in self._id:
            row = self.cn.execute("SELECT token_id FROM mesh_slm_vocab WHERE token=?", (t,)).fetchone()
            self._id[t] = int(row[0]) if row else None
        return self._id[t]

    def token(self, tid: int) -> str:
        if tid not in self._tok:
            row = self.cn.execute("SELECT token FROM mesh_slm_vocab WHERE token_id=?", (tid,)).fetchone()
            self._tok[tid] = str(row[0]) if row else str(tid)
        return self._tok[tid]

    def embed(self, tid: int) -> Optional[list[float]]:
        if tid not in self._emb:
            try:
                row = self.cn.execute(
                    "SELECT e_vision, e_touch, e_smell, e_body, e_brain, e_perception, e_entirety "
                    "FROM mesh_slm_embed WHERE token_id=?", (tid,)).fetchone()
            except sqlite3.OperationalError:
                row = None
            self._emb[tid] = [float(x or 0.0) for x in row] if row else None
        return self._emb[tid]

    def out(self, tid: int) -> tuple[list[tuple[int, float, float, bool]], float]:
        """Top out-edges (dst, weight, samples, shadow) and the full out-weight mass (shadow included)."""
        if tid not in self._out:
            rows = self.cn.execute(
                "SELECT dst, weight, samples FROM mesh_slm_quipu WHERE src=? AND weight > 0 "
                "ORDER BY weight DESC LIMIT ?", (tid, FANOUT)).fetchall()
            total = self.cn.execute("SELECT COALESCE(SUM(weight), 0) FROM mesh_slm_quipu "
                                    "WHERE src=? AND weight > 0", (tid,)).fetchone()[0]
            edges = [(int(d), float(w), float(n or 1), False) for d, w, n in rows if int(d) != tid]
            for d, lift in self.shadow.get(tid, []):
                if lift > 0.0 and d != tid:
                    edges.append((d, lift, 1.0, True))
                    total = float(total or 0.0) + lift
            self._out[tid] = (edges, float(total or 0.0))
        return self._out[tid]


def _hop(g: _Graph, u: int, v: int, w: float, n: float, shadow: bool, total: float,
         weights: Sequence[float], plane_magic: Mapping[str, Mapping]) -> dict[str, Any]:
    p = w / total if total > 0.0 else 0.0
    planes = hop_planes(g.embed(u), g.embed(v))
    m = hop_magic(planes, plane_magic)
    return {"src": g.token(u), "dst": g.token(v), "shadow": shadow,
            "p": p, "surprisal": -math.log2(p) if p > 0.0 else 0.0,
            "entropy": branching_entropy(weights), "resolvable": resolvable_bits(n),
            "planes": planes, **m}


def pathways(cn: sqlite3.Connection, seeds: Sequence[int], plane_magic: Mapping[str, Mapping], *,
             shadow_edges: Sequence[Mapping[str, Any]] = (), max_depth: int = MAX_DEPTH,
             beam: int = BEAM) -> list[list[dict[str, Any]]]:
    """Most probable simple walks (≤ max_depth hops) from each seed, by beam search on Σ log₂ p."""
    g = _Graph(cn, shadow_edges)
    found: list[list[dict[str, Any]]] = []
    for s in seeds:
        frontier: list[tuple[float, list[int], list[dict[str, Any]]]] = [(0.0, [s], [])]
        for _ in range(max(1, int(max_depth))):
            nxt = []
            for score, nodes, hops in frontier:
                u = nodes[-1]
                edges, total = g.out(u)
                weights = [e[1] for e in edges]
                for v, w, n, shadow in edges:
                    if v in nodes:
                        continue
                    h = _hop(g, u, v, w, n, shadow, total, weights, plane_magic)
                    nxt.append((score + h["surprisal"], nodes + [v], hops + [h]))
            if not nxt:
                break
            nxt.sort(key=lambda t: t[0])
            frontier = nxt[:max(1, int(beam))]
        found.extend(h for _, _, h in frontier if h)
    return found


# ---------------------------------------------------------------- supernova events

def npc_planes(direction: float, kappa: float = SUPERNOVA_KAPPA) -> list[float]:
    """ν_k ∝ exp(κ cos(θ − 2πk/7)): an NPC's expansion direction read on the seven axis planes."""
    th = float(direction)
    if not math.isfinite(th):
        raise ValueError("direction must be finite")
    w = [math.exp(float(kappa) * math.cos(th - 2.0 * math.pi * k / 7.0)) for k in range(7)]
    s = sum(w)
    return [x / s for x in w]


def alignment(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of two non-negative plane mixes, in [0, 1]."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))


def supernova_events(cn: sqlite3.Connection, now: float) -> tuple[list[dict], int]:
    """Fresh Perceptopoly supernova decisions with their weight and plane reading; and how many were ignored."""
    from .knot_shadow import _npcs
    try:
        raw = _npcs(cn)
    except sqlite3.OperationalError:
        return [], 0
    fresh, ignored = [], 0
    for n in raw:
        age = float(now) - float(n["t"])
        if not (-60.0 <= age <= SUPERNOVA_WINDOW_S):
            ignored += 1
            continue
        m = max(0.0, min(1.0, float(n["magnitude"])))
        nu = npc_planes(n["direction"])
        fresh.append({"agent_id": n["id"], "magnitude": m, "direction": float(n["direction"]),
                      "age_s": age, "weight": math.exp(-max(0.0, age) / SUPERNOVA_TAU_S) * m,
                      "plane": AXES[max(range(7), key=lambda k: nu[k])], "planes": nu})
    return fresh, ignored


def connect_supernova(paths: list[dict], npcs: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Score paths by the fresh events (in place) and return each NPC's pathways."""
    sw = sum(n["weight"] for n in npcs)
    for p in paths:
        a = [alignment(p["planes"], n["planes"]) for n in npcs]
        s = sum(n["weight"] * x for n, x in zip(npcs, a)) / max(1.0, sw)
        p["supernova"] = round(s, 9)
        p["expansion_gain"] = math.floor(min(PHI, p["gain"] * (1.0 + (PHI - 1.0) * s)) * 1e9) / 1e9
    out = []
    for n in sorted(npcs, key=lambda n: -n["weight"])[:MAX_NPCS_OUT]:
        scored = sorted(((n["weight"] * alignment(p["planes"], n["planes"]) * p["gain"],
                          alignment(p["planes"], n["planes"]), p) for p in paths), key=lambda t: -t[0])
        out.append({
            "agent_id": n["agent_id"], "magnitude": n["magnitude"], "direction": n["direction"],
            "age_s": round(n["age_s"], 3), "weight": round(n["weight"], 9), "plane": n["plane"],
            "paths": [{"tokens": p["tokens"], "alignment": round(a, 9), "score": round(sc, 9),
                       "plane": p["plane"], "gain": p["gain"], "expansion_gain": p["expansion_gain"],
                       "surprisal_bits": p["surprisal_bits"], "delivered_bits": p["delivered_bits"],
                       "magic": p["magic"], "held": p["held"], "shadow_hops": p["shadow_hops"]}
                      for sc, a, p in scored[:PATHS_PER_NPC] if sc > 0.0],
        })
    return out


# ---------------------------------------------------------------- the map

def build(cn: sqlite3.Connection, *, now: Optional[float] = None,
          temperature_k: float = TEMPERATURE_K, max_paths: int = MAX_PATHS) -> dict[str, Any]:
    """Build and publish the DeepMap from the MESH graph and the mesh flow's latest readings."""
    now = time.time() if now is None else float(now)
    plane_magic = (_kv_get(cn, "entirety:mesh_flow:magic", {}) or {}).get("planes") or {}
    flow = _kv_get(cn, "entirety:mesh_flow:latest", {}) or {}
    props = _kv_get(cn, "entirety:mesh_flow:proposals", {}) or {}
    stability = float(flow.get("stability_number", PHI) or PHI)
    try:
        g = _Graph(cn, ())
        seeds = [tid for tid in (g.token_id(t.get("token")) for t in (props.get("tokens") or [])) if tid is not None]
        seed_from = "shadow_tokens"
        if not seeds:
            seeds = [int(r[0]) for r in cn.execute(
                "SELECT token_id FROM mesh_slm_vocab ORDER BY freq DESC LIMIT ?", (MAX_SEEDS,)).fetchall()]
            seed_from = "top_tokens"
        seeds = list(dict.fromkeys(seeds))[:MAX_SEEDS]
        walks = pathways(cn, seeds, plane_magic, shadow_edges=props.get("edges") or [])
    except sqlite3.OperationalError:
        seeds, walks, seed_from = [], [], "no_mesh"
    paths = []
    seen = set()
    for hops in walks:
        key = tuple([hops[0]["src"]] + [h["dst"] for h in hops])
        if key in seen:
            continue
        seen.add(key)
        acc = compose(hops, stability_number=stability, temperature_k=temperature_k)
        paths.append({"tokens": list(key), **{k: (round(v, 9) if isinstance(v, float) and k not in _TINY else v)
                                              for k, v in acc.items()}})
    npcs, ignored = supernova_events(cn, now)
    per_npc = connect_supernova(paths, npcs)
    paths.sort(key=lambda p: (-p["expansion_gain"], p["surprisal_bits"]))
    paths = paths[:max(1, int(max_paths))]
    by_plane: dict[str, int] = {}
    for p in paths:
        by_plane[p["plane"]] = by_plane.get(p["plane"], 0) + 1
    rec = {
        "at": now,
        "seed_from": seed_from,
        "seeds": len(seeds),
        "stability_number": stability,
        "temperature_k": temperature_k,
        "paths": paths,
        "by_plane": by_plane,
        "totals": {
            "surprisal_bits": round(sum(p["surprisal_bits"] for p in paths), 9),
            "interstitial_bits": round(sum(p["interstitial_bits"] for p in paths), 9),
            "magic": round(sum(p["magic"] for p in paths), 9),
            "info_mass_kg": sum(p["info_mass_kg"] for p in paths),
        },
        "supernova": {"fresh": len(npcs), "ignored": ignored,
                      "pressure": round(sum(p["supernova"] for p in paths) / len(paths), 9) if paths else 0.0,
                      "npcs": per_npc},
        "realisation": "map only: pathways are read, never written",
    }
    _kv_set(cn, KV_LATEST, rec)
    return rec


def latest(cn: sqlite3.Connection) -> dict:
    rec = _kv_get(cn, KV_LATEST, {})
    return rec if isinstance(rec, dict) else {}


__all__ = ["KV_PREFIX", "KV_LATEST", "AXES", "npc_planes", "alignment", "supernova_events",
           "connect_supernova", "resolvable_bits", "branching_entropy", "hop_planes",
           "hop_magic", "compose", "pathways", "build", "latest"]
