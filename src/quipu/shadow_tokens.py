"""shadow_tokens — Monte Carlo Lagrangian λ tokens over the Essay's planes.

The Essay (*The Aero-Chthonic Canvas*, Perceptopoly ``docs/essays``) has six
movements, I–VI.  Each one is a plane: a direction in QUIPU's 7-D mesh, built
from the movement's own words as QUIPU has embedded them, minus the Essay's
mean direction (``contrast``).  Without the contrast the planes are nearly one
plane: on the host brain (2026-10-01) ~95 % of every movement's direction is
what the whole Essay shares, and the six directions have pairwise cosines of
0.73–0.997; contrasted, −0.74 to 0.59.  Voicing the Essay
through the mesh is a mixed-integer linear problem (one scenario ω):

    maximise   Σ_t Σ_p  s_tp(ω) · x_tp
    subject to Σ_t x_tp      = K          for every plane p      (each movement is voiced by K tokens)
               Σ_p x_tp      ≤ 1          for every token t      (a token voices one movement)
               Σ_{t∈T_c,p} x_tp ≥ q_c     for every cluster c    (the Internal Marketplace's equitable floor)
               x_tp ∈ {0, 1}

``s_tp(ω)`` is the centred cosine between token t's embedding and plane p's
direction under scenario ω — non-linear in the plane, which is why the planes
are called non-linear.  ``T_c`` is the set of tokens nearest to marketplace
cluster c's anchor (its category and commodity words, as QUIPU embeds them);
tokens near no anchor are the commons and carry no floor.

A MILP has no LP shadow prices.  Its prices come from Lagrangian relaxation:
the two coupling constraints (token-once, cluster floor) are moved into the
objective with multipliers μ_t ≥ 0 and λ_c ≥ 0, and the problem that is left
splits by plane (take the K best reduced scores).  The dual
``min_{μ,λ≥0} L(μ, λ)`` is solved by subgradient descent with a Polyak step; it
is an upper bound on the MILP optimum.  Because the split problem has the
integrality property, that bound equals the LP-relaxation bound (Geoffrion,
1974), and the tests check both.  The MILP itself is solved exactly with HiGHS
(``scipy.optimize.milp``) when SciPy is present, otherwise by repairing the
Lagrangian solution; the gap between the dual bound and that primal is
reported for every scenario.

**λ_c is the price of equity for cluster c**: how much of the Essay's alignment
score the emission gives up per token to give that cluster its floor, in units
of centred cosine, not dollars.  A high λ_c means QUIPU's mesh holds little
that sits near that cluster; it is a signal of where QUIPU knows the
marketplace least.

Monte Carlo: each of N scenarios perturbs every plane direction with QUIPU's
own Langevin emission noise (σ from ``mesh_slm._langevin_compute_sigma``, i.e.
corpus freshness) and re-draws the plane from a bootstrap of the movement's
sentences.  λ is therefore a distribution (mean, p10, p50, p90, how often the
floor binds), not a point.

Outputs (``brain_kv["entirety:lambda:latest"]``, ``GET /lambda``):

* ``lambda_tokens`` — one per marketplace cluster: ``λ:<cluster>`` with its
  distribution, floor, capacity and expected share;
* ``shadow_tokens`` — tokens the coupling constraints move, against the free
  emission (each movement takes its K best tokens, no coupling): *held back*
  (free emission picks them, the MILP does not: a token-once conflict or a floor
  displaces them) and *pulled in* (the MILP picks them, free emission does not:
  a floor pays for them); weight = the difference in selection frequency
  across scenarios, with the token's mean reduced score at the dual optimum;
* ``emission`` — per movement, the tokens the MILP picks in most scenarios,
  each movement's top-K selection frequencies, and its signal-to-noise ratio
  (mean direction over the spread of its scenario draws); below 1 the mesh
  cannot resolve that movement under its own noise, and its voice is empty;
* ``projections`` — what the Internal Marketplace reads: per cluster, the λ
  distribution and its share of the total price of equity (``attention``).
  A projection proposes nothing and changes no price.
* ``potential`` — the pruned residual potential, per movement and in total, in
  Perceptopoly's form: real = the pruned excess (what the coupling pruned, plus
  what the pool cut before the problem started), imaginary = crossing rate ×
  separation (the share of a voice that changes between two draws, times how far
  apart its two regimes sit);
* ``k_profile`` — each movement's crossing rate at every voice size: where its
  scores have a gap the noise does not cross.

Each movement may voice its own number of tokens (``k`` = int or
``{movement: K}``).  ``qpsi.open_doors`` tunes K per movement, the pool and the
scenario count from the potential, inside the bounds the operator grants.

Clusters arrive from the marketplace (``POST /lambda/clusters``, signed:
supply-chain-brain, hubcore, hub-floor) and are kept in
``brain_kv["entirety:lambda:clusters"]``.  Nothing here writes to the mesh:
no vocabulary, no edges, no specialists.

    python -m src.quipu.shadow_tokens run [--scenarios 64] [--k 8] [--equity 0.5]
    python -m src.quipu.shadow_tokens status

翈 — the feather vane prices the wind it turns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

_LOG = logging.getLogger(__name__)

KV_CLUSTERS = "entirety:lambda:clusters"
KV_LATEST = "entirety:lambda:latest"
ESSAY_ENV = "QUIPU_ESSAY_PATH"
ESSAY_CANDIDATES = (
    "/app/essays/aero_chthonic_canvas.md",                     # fleet compose mount
    "docs/essays/aero_chthonic_canvas.md",
    "../Perceptopoly/docs/essays/aero_chthonic_canvas.md",      # fleet checkout
)
MARKET_SOURCES = ("supply-chain-brain", "hubcore", "hub-floor")
MAX_CLUSTERS = 64
ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,30}")
_SENT = re.compile(r"(?<=[.!?])\s+")
_NUMERIC = re.compile(r"^[\d.\-]+$")


# ---------------------------------------------------------------------------
# The Essay's planes
# ---------------------------------------------------------------------------

@dataclass
class Plane:
    n: str
    title: str
    text: str
    sentences: list[str] = field(default_factory=list)


def essay_path(explicit: Optional[str] = None) -> Optional[Path]:
    for cand in ([explicit] if explicit else []) + [os.environ.get(ESSAY_ENV, "")] + list(ESSAY_CANDIDATES):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def essay_planes(path: Path) -> tuple[list[Plane], dict[str, Any]]:
    """Split the Essay at its ``## <roman>.`` headings: one plane per movement."""
    raw = path.read_bytes()
    text = raw.decode("utf-8", "replace")
    planes: list[Plane] = []
    cur: Optional[Plane] = None
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code            # diagrams are not the movement's words
            continue
        if in_code:
            continue
        m = re.match(r"^##\s+([IVX]+)\.\s*(.+?)\s*$", line)
        if m and m.group(1) in ROMAN:
            cur = Plane(m.group(1), m.group(2), "")
            planes.append(cur)
        elif cur is not None:
            cur.text += line + "\n"
    for p in planes:
        body = re.sub(r"[*_`>#|─│┌┐└┘┴┼►◄▼▲\[\]]", " ", p.text)
        p.sentences = [s.strip() for s in _SENT.split(body) if len(_WORD.findall(s)) >= 3]
    title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), None)
    return planes, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "title": title,
                    "planes": [p.n for p in planes]}


# ---------------------------------------------------------------------------
# QUIPU's mesh: candidate tokens and embeddings
# ---------------------------------------------------------------------------

_EMB_COLS = "e_vision, e_touch, e_smell, e_body, e_brain, e_perception, e_entirety"


@dataclass
class Mesh:
    tokens: list[str]
    freq: np.ndarray            # (T,)
    E: np.ndarray               # (T, 7) raw embeddings of the candidate pool
    centre: np.ndarray          # (7,) the reference every score is centred on
    lookup: dict[str, np.ndarray]   # every embedded token -> raw embedding (for plane words, anchors)
    sigma: float                # QUIPU's Langevin emission noise
    ext_tokens: list[str] = field(default_factory=list)   # the next block past the pool, by frequency
    ext_E: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    ext_freq: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def pool(self, n: int, m: int = 0) -> "Mesh":
        """The first ``n`` ranked candidates as the pool and the next ``m`` as the block past
        it (the ranking is pool then block, as loaded).  The centre does not move with the
        pool, so a token scores the same whatever pool it sits in."""
        toks = list(self.tokens) + list(self.ext_tokens)
        E = np.vstack([self.E.reshape(-1, 7), self.ext_E.reshape(-1, 7)])
        freq = np.concatenate([np.asarray(self.freq, float), np.asarray(self.ext_freq, float)])
        n = max(0, min(int(n), len(toks)))
        m = max(0, min(int(m), len(toks) - n))
        return Mesh(toks[:n], freq[:n], E[:n], self.centre, self.lookup, self.sigma,
                    toks[n:n + m], E[n:n + m], freq[n:n + m])


def _candidate(tok: str) -> bool:
    return (len(tok) >= 3 and not _NUMERIC.match(tok) and ":" not in tok
            and tok.count("-") < 2 and not tok.endswith("-"))


def load_mesh(n_candidates: int = 384, cn=None, n_extension: int = 256) -> Mesh:
    """The pool (the ``n_candidates`` most frequent embedded tokens) and the next
    ``n_extension`` past it.  Centre: the frequency-weighted mean of every embedded token,
    so it does not depend on where the pool is cut."""
    from . import mesh_slm

    def _read(c):
        rows = c.execute(
            f"SELECT v.token, v.freq, {_EMB_COLS} FROM mesh_slm_vocab v "
            "JOIN mesh_slm_embed e ON e.token_id = v.token_id ORDER BY v.freq DESC").fetchall()
        try:
            h = mesh_slm._langevin_token_entropy(c)
        except Exception:
            h = 0.5
        return rows, h

    if cn is None:
        with mesh_slm._conn() as c:
            rows, h = _read(c)
    else:
        rows, h = _read(cn)
    lookup: dict[str, np.ndarray] = {}
    tot_w, acc = 0.0, np.zeros(7)
    ranked_t, ranked_f, ranked_e = [], [], []
    limit = max(0, int(n_candidates)) + max(0, int(n_extension))
    for r in rows:
        tok = str(r[0])
        vec = np.array([float(r[k]) for k in range(2, 9)], dtype=float)
        if not np.any(vec):
            continue
        lookup[tok] = vec
        w = max(float(r[1] or 0.0), 0.0)
        acc += w * vec
        tot_w += w
        if len(ranked_t) < limit and _candidate(tok):
            ranked_t.append(tok)
            ranked_f.append(float(r[1] or 0.0))
            ranked_e.append(vec)
    centre = acc / tot_w if tot_w > 0 else (np.mean(list(lookup.values()), axis=0) if lookup else np.zeros(7))
    sigma = float(mesh_slm._langevin_compute_sigma([], 0.0, h))
    full = Mesh(ranked_t, np.array(ranked_f), np.array(ranked_e, dtype=float).reshape(-1, 7),
                centre, lookup, sigma)
    return full.pool(n_candidates, n_extension)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-12)


def plane_direction(words: Sequence[str], mesh: Mesh) -> tuple[Optional[np.ndarray], int]:
    """Frequency-weighted mean of the movement's words as QUIPU embeds them, centred."""
    vecs = [mesh.lookup[w] for w in words if w in mesh.lookup]
    if not vecs:
        return None, 0
    return np.mean(vecs, axis=0) - mesh.centre, len(vecs)


def _words(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


# ---------------------------------------------------------------------------
# Marketplace clusters
# ---------------------------------------------------------------------------

def sanitize_clusters(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for c in (raw or [])[:MAX_CLUSTERS] if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        cid = re.sub(r"[^A-Za-z0-9 _\-./]", "", str(c.get("id") or c.get("cluster_id") or ""))[:64].strip()
        if not cid or cid.lower() in seen:
            continue
        seen.add(cid.lower())
        anchors = c.get("anchors") or [cid, c.get("commodity") or ""]
        out.append({"id": cid, "commodity": re.sub(r"[^A-Za-z0-9 _\-]", "", str(c.get("commodity") or ""))[:32],
                    "size": max(0, int(c.get("size") or c.get("cluster_size") or 0)),
                    "anchors": [a for a in (re.sub(r"[^A-Za-z0-9 _\-]", " ", str(x))[:64] for x in anchors) if a.strip()][:8]})
    return out


def store_clusters(raw: Any, source: str) -> dict[str, Any]:
    from . import brain_kv
    clusters = sanitize_clusters(raw)
    rec = {"clusters": clusters, "source": source, "at": time.time()}
    brain_kv.kv_set_json(KV_CLUSTERS, rec)
    return {"ok": True, "clusters": len(clusters)}


def assign_clusters(mesh: Mesh, clusters: list[dict[str, Any]], threshold: float = 0.3
                    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Each candidate token's nearest cluster anchor (centred cosine ≥ threshold), else the commons (-1)."""
    anchored, A = [], []
    for c in clusters:
        words = [w for a in c["anchors"] for w in _words(a)]
        vecs = [mesh.lookup[w] for w in words if w in mesh.lookup]
        info = {**c, "anchors_found": sorted({w for w in words if w in mesh.lookup})}
        if vecs:
            A.append(np.mean(vecs, axis=0) - mesh.centre)
            info["index"] = len(A) - 1
        else:
            info["index"] = None
        anchored.append(info)
    if not A or not len(mesh.E):
        return np.full(len(mesh.tokens), -1), anchored
    cos = _unit(mesh.E - mesh.centre) @ _unit(np.array(A)).T          # (T, C)
    best = cos.argmax(axis=1)
    member = np.where(cos[np.arange(len(best)), best] >= threshold, best, -1)
    return member, anchored


# ---------------------------------------------------------------------------
# One scenario: Lagrangian dual (subgradient) and exact primal (HiGHS)
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    S: np.ndarray          # (T, P) scores
    member: np.ndarray     # (T,) cluster index or -1
    q: np.ndarray          # (C,) floors
    K: Any                 # tokens per plane: one int for every plane, or one per plane

    def __post_init__(self) -> None:
        self.K = _per_plane(self.K, self.S.shape[1])


def _per_plane(K: Any, P: int) -> np.ndarray:
    return np.broadcast_to(np.asarray(K, dtype=int), (P,)).copy()


def _subproblem(R: np.ndarray, K: Any) -> np.ndarray:
    T, P = R.shape
    Ks = _per_plane(K, P)
    x = np.zeros_like(R)
    for p in range(P):
        k = int(min(Ks[p], T))
        if k <= 0:
            continue
        idx = np.argpartition(-R[:, p], k - 1)[:k]
        x[idx, p] = 1.0
    return x


def _cluster_counts(x: np.ndarray, member: np.ndarray, C: int) -> np.ndarray:
    per_token = x.sum(axis=1)
    m = member >= 0
    return np.bincount(member[m], weights=per_token[m], minlength=C)[:C]


def lagrangian_dual(inst: Instance, iters: int = 200, lower: float | None = None
                    ) -> dict[str, Any]:
    """min over μ, λ ≥ 0 of L(μ, λ); returns the best bound, its multipliers and subproblem solution."""
    S, member, q, K = inst.S, inst.member, inst.q, inst.K
    T, P = S.shape
    C = len(q)
    mu, lam = np.zeros(T), np.zeros(C)
    lam_t = np.zeros(T)
    best = {"bound": math.inf}
    lb = lower if lower is not None else _greedy_value(inst)
    for it in range(iters):
        lam_t[:] = 0.0
        m = member >= 0
        lam_t[m] = lam[member[m]]
        R = S - mu[:, None] + lam_t[:, None]
        x = _subproblem(R, K)
        L = float((R * x).sum() + mu.sum() - lam @ q)
        if L < best["bound"]:
            best = {"bound": L, "mu": mu.copy(), "lam": lam.copy(), "x": x.copy(), "iter": it}
        g_mu = x.sum(axis=1) - 1.0                           # ≤ 0 when feasible
        g_lam = q - _cluster_counts(x, member, C)            # ≤ 0 when the floors hold
        g_mu = np.where((mu <= 0) & (g_mu < 0), 0.0, g_mu)   # projected subgradient
        g_lam = np.where((lam <= 0) & (g_lam < 0), 0.0, g_lam)
        norm = float(g_mu @ g_mu + g_lam @ g_lam)
        if norm < 1e-12:
            break                                            # x is optimal for the MILP
        step = max(L - lb, 1e-3 * max(abs(L), 1.0)) / norm
        mu = np.maximum(0.0, mu + step * g_mu)
        lam = np.maximum(0.0, lam + step * g_lam)
    return best


def _greedy_value(inst: Instance) -> float:
    return float(_repair(inst, _subproblem(inst.S, inst.K))[1])


def _repair(inst: Instance, x0: np.ndarray) -> tuple[np.ndarray, float]:
    """A feasible MILP point from a relaxed one: one plane per token, then floors, then refill."""
    S, member, q, K = inst.S, inst.member, inst.q, inst.K
    T, P = S.shape
    x = np.zeros_like(S)
    used = np.zeros(T, bool)
    for t in np.argsort(-(S * x0).max(axis=1)):
        if x0[t].any():
            p = int(np.argmax(np.where(x0[t] > 0, S[t], -np.inf)))
            if x[:, p].sum() < K[p]:
                x[t, p], used[t] = 1.0, True
    for c in range(len(q)):
        need = int(q[c] - _cluster_counts(x, member, len(q))[c])
        for t in np.argsort(-S.max(axis=1)):
            if need <= 0:
                break
            if member[t] != c or used[t]:
                continue
            open_p = [p for p in range(P) if x[:, p].sum() < K[p]]
            p = max(open_p, key=lambda p: S[t, p]) if open_p else int(np.argmax(S[t]))
            if not open_p:      # swap out the plane's worst token that no floor needs
                counts = _cluster_counts(x, member, len(q))
                cands = [u for u in np.where(x[:, p] > 0)[0]
                         if member[u] < 0 or counts[member[u]] > q[member[u]]]
                if not cands:
                    continue
                u = min(cands, key=lambda u: S[u, p])
                x[u, p], used[u] = 0.0, False
            x[t, p], used[t] = 1.0, True
            need -= 1
    for p in range(P):
        for t in np.argsort(-S[:, p]):
            if x[:, p].sum() >= K[p]:
                break
            if not used[t]:
                x[t, p], used[t] = 1.0, True
    return x, float((S * x).sum())


def exact_milp(inst: Instance, time_limit: float = 10.0) -> Optional[tuple[np.ndarray, float]]:
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception:
        return None
    S, member, q, K = inst.S, inst.member, inst.q, inst.K
    T, P = S.shape
    C = len(q)
    n = T * P                                           # x[t, p] -> t * P + p
    A = lil_matrix((P + T + C, n))
    for p in range(P):
        A[p, [t * P + p for t in range(T)]] = 1.0
    for t in range(T):
        A[P + t, t * P:(t + 1) * P] = 1.0
        if member[t] >= 0:
            A[P + T + member[t], t * P:(t + 1) * P] = 1.0
    lo = np.concatenate([K.astype(float), np.full(T, -np.inf), q])
    hi = np.concatenate([K.astype(float), np.ones(T), np.full(C, np.inf)])
    res = milp(-S.ravel(), constraints=LinearConstraint(A.tocsr(), lo, hi),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        return None
    x = np.round(res.x).reshape(T, P)
    return x, float((S * x).sum())


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def _q(vals: list[float], pct: float) -> float:
    return float(np.percentile(vals, pct)) if vals else 0.0


@dataclass
class Simulation:
    """Every scenario's record of one Monte Carlo run, kept so the pruned residual
    potential can be computed, bootstrapped and compared with a mirror run."""
    planes: list[Plane]
    voiced: list[dict[str, Any]]
    essay: dict[str, Any]
    mesh: Mesh
    K: np.ndarray                  # (P,) tokens per voiced plane
    seed: int
    equity: float
    contrast: bool
    member: np.ndarray             # (T,) cluster of each pool token, -1 = commons
    anchored: list[dict[str, Any]]
    live: list[dict[str, Any]]
    capacity: np.ndarray
    q: np.ndarray
    floor: int
    S: np.ndarray                  # (n, T, P) scores
    milp: np.ndarray               # (n, T, P) the MILP's selection
    free: np.ndarray               # (n, T, P) free emission: each plane's K best, no coupling
    cut_milp: np.ndarray           # (n, P) lowest score the MILP voiced on each plane
    cut_free: np.ndarray           # (n, P) each plane's K-th best score
    ext_S: np.ndarray              # (n, M, P) scores of the block past the pool
    lam: np.ndarray                # (n, C)
    red: np.ndarray                # (T, P) mean reduced score at the dual optimum
    draws: np.ndarray              # (n, P, 7)
    bounds: list[float]
    primals: list[float]
    gaps: list[float]
    exact_n: int
    seconds: float

    @property
    def n(self) -> int:
        return int(self.S.shape[0])


def _resolve_k(k: Any, planes: list[Plane]) -> np.ndarray:
    if isinstance(k, dict):
        default = int(k.get("*", k.get("default", 8)))
        return np.array([int(k.get(p.n, default)) for p in planes], dtype=int)
    if isinstance(k, (list, tuple, np.ndarray)):
        return np.array([int(v) for v in k][:len(planes)], dtype=int)
    return np.full(len(planes), int(k), dtype=int)


def simulate(*, scenarios: int = 64, k: Any = 8, equity: float = 0.5, n_candidates: int = 384,
             n_extension: int = 256, iters: int = 200, seed: int | None = None, threshold: float = 0.3,
             clusters: Optional[list[dict[str, Any]]] = None, essay: Optional[str] = None,
             mesh: Optional[Mesh] = None, contrast: bool = True
             ) -> tuple[Optional[Simulation], Optional[dict[str, Any]]]:
    """One Monte Carlo run.  Returns (simulation, None) or (None, error)."""
    started = time.time()
    path = essay_path(essay)
    if path is None:
        return None, {"ok": False, "error": f"the Essay was not found (set {ESSAY_ENV})"}
    planes, essay_info = essay_planes(path)
    mesh = mesh or load_mesh(n_candidates, n_extension=n_extension)
    if clusters is None:
        from . import brain_kv
        clusters = (brain_kv.kv_get_json(KV_CLUSTERS, {}) or {}).get("clusters") or []
    clusters = sanitize_clusters(clusters)

    voiced = []
    for p in planes:
        d, n_words = plane_direction(_words(p.text), mesh)
        voiced.append({"plane": p.n, "title": p.title, "words_in_mesh": n_words, "voiced": d is not None})
    live_planes = [p for p, v in zip(planes, voiced) if v["voiced"]]
    if not live_planes:
        return None, {"ok": False, "error": "none of the Essay's words are in QUIPU's mesh yet"}
    P = len(live_planes)
    K = np.maximum(_resolve_k(k, live_planes), 1)
    T = len(mesh.tokens)
    if T < int(K.sum()):
        return None, {"ok": False, "error": f"the mesh has {T} candidate tokens; {int(K.sum())} are needed"}

    member, anchored = assign_clusters(mesh, clusters, threshold)
    live = [c for c in anchored if c["index"] is not None]
    C = len(live)
    budget = int(K.sum())
    capacity = np.bincount(member[member >= 0], minlength=C)[:C] if C else np.zeros(0, int)
    floor = int(math.floor(equity * budget / C)) if C else 0
    q = np.minimum(floor, capacity).astype(float)

    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    rng = np.random.default_rng(seed)
    Eu = _unit(mesh.E - mesh.centre)
    Xu = _unit(mesh.ext_E.reshape(-1, 7) - mesh.centre)
    M = len(Xu)
    n = int(scenarios)
    S_all = np.zeros((n, T, P), dtype=np.float32)
    X_all = np.zeros((n, M, P), dtype=np.float32)
    milp_all = np.zeros((n, T, P), dtype=bool)
    free_all = np.zeros((n, T, P), dtype=bool)
    cut_m = np.zeros((n, P))
    cut_f = np.zeros((n, P))
    lam_samples = np.zeros((n, C))
    red_sum = np.zeros((T, P))
    draws = np.zeros((n, P, 7))
    gaps, bounds, primals, exact_n = [], [], [], 0
    for s in range(n):
        # The draws use the rng in a fixed order that does not depend on the pool or on
        # K, so the same seed gives the same planes whatever those are (the mirror's basis).
        D = []
        for p in live_planes:
            sents = p.sentences or [p.text]
            draw = rng.choice(len(sents), size=len(sents), replace=True)
            d, _ = plane_direction(_words(" ".join(sents[i] for i in draw)), mesh)
            if d is None:
                d, _ = plane_direction(_words(p.text), mesh)
            D.append(d + rng.normal(0.0, mesh.sigma, 7))
        D = np.array(D)
        if contrast and P > 1:
            D = D - D.mean(axis=0)              # what each movement adds to the Essay
        draws[s] = D
        Du = _unit(D)
        S = Eu @ Du.T                                                # (T, P) centred cosine
        inst = Instance(S, member, q, K)
        ex = exact_milp(inst)
        if ex is not None:
            xp, zp = ex
            exact_n += 1
        dual = lagrangian_dual(inst, iters=iters, lower=(zp if ex is not None else None))
        if ex is None:
            xp, zp = _repair(inst, dual["x"])
        xf = _subproblem(S, K)
        S_all[s] = S
        if M:
            X_all[s] = Xu @ Du.T
        milp_all[s] = xp > 0.5
        free_all[s] = xf > 0.5
        cut_m[s] = [S[milp_all[s][:, j], j].min() if milp_all[s][:, j].any() else 0.0 for j in range(P)]
        cut_f[s] = [S[free_all[s][:, j], j].min() if free_all[s][:, j].any() else 0.0 for j in range(P)]
        lam_samples[s] = dual["lam"]
        lam_t = np.zeros(T)
        lam_t[member >= 0] = dual["lam"][member[member >= 0]]
        red_sum += S - dual["mu"][:, None] + lam_t[:, None]
        bounds.append(dual["bound"])
        primals.append(zp)
        gaps.append((dual["bound"] - zp) / max(abs(zp), 1e-9))
    sim = Simulation(live_planes, voiced, essay_info, mesh, K, int(seed), float(equity), bool(contrast),
                     member, anchored, live, capacity, q, floor, S_all, milp_all, free_all, cut_m, cut_f,
                     X_all, lam_samples, red_sum / n, draws, bounds, primals, gaps, exact_n,
                     round(time.time() - started, 2))
    return sim, None


# ---------------------------------------------------------------------------
# The pruned residual potential
# ---------------------------------------------------------------------------

def _complex(real: float, imag: float, rate: float, sep: float) -> dict[str, Any]:
    return {"real": round(real, 5), "imag": round(imag, 5), "magnitude": round(math.hypot(real, imag), 5),
            "phase_deg": round(math.degrees(math.atan2(imag, real)), 1),
            "crossing_rate": round(rate, 4), "separation": round(sep, 5)}


def potential(sim: Simulation, idx: Optional[np.ndarray] = None) -> dict[str, Any]:
    """The pruned residual potential, per plane and in total, in Perceptopoly's form
    (``theseuscape.bonding.residual_potential``): real = surprise, imaginary = crossing
    rate × separation.  Every (token, plane) is a two-regime record across the scenarios —
    voiced or pruned.

    real        the pruned excess: alignment above the lowest voiced score that the coupling
                constraints pruned (free emission voices it, the MILP does not), per voiced
                slot and scenario
    rate        the share of the plane's voice that differs between two scenarios:
                Σ_t 2 f_t (1 − f_t) / (2 K), f_t the share of scenarios voicing t
    separation  the score distance between a token's two regimes (voiced vs pruned),
                weighted by how often it crosses
    imag        rate × separation
    pool        the excess the pool cut before the problem started: tokens past the pool
                whose score clears each plane's K-th best (free emission), per slot

    Units: centred cosine.  ``idx`` resamples scenarios (bootstrap)."""
    sel = sim.milp if idx is None else sim.milp[idx]
    free = sim.free if idx is None else sim.free[idx]
    S = (sim.S if idx is None else sim.S[idx]).astype(float)
    cm = sim.cut_milp if idx is None else sim.cut_milp[idx]
    cf = sim.cut_free if idx is None else sim.cut_free[idx]
    X = (sim.ext_S if idx is None else sim.ext_S[idx]).astype(float)
    n = max(1, S.shape[0])
    K = sim.K.astype(float)
    f = sel.mean(axis=0)                                          # (T, P)
    r = S - cm[:, None, :]
    cnt = sel.sum(axis=0)
    mean_v = (r * sel).sum(axis=0) / np.maximum(cnt, 1)
    mean_p = (r * ~sel).sum(axis=0) / np.maximum(n - cnt, 1)
    sep = np.where((cnt > 0) & (cnt < n), np.abs(mean_v - mean_p), 0.0)
    cross = 2.0 * f * (1.0 - f)
    csum = cross.sum(axis=0)                                      # (P,)
    rate = csum / (2.0 * K)
    separation = (cross * sep).sum(axis=0) / np.maximum(csum, 1e-12)
    imag = rate * separation
    real = (np.clip(r, 0.0, None) * (free & ~sel)).sum(axis=(0, 1)) / n / K
    if X.shape[1]:
        over = X - cf[:, None, :]
        pool = np.clip(over, 0.0, None).sum(axis=(0, 1)) / n / K
        enter = (over > 0).sum(axis=1).mean(axis=0)
    else:
        pool = np.zeros(len(K))
        enter = np.zeros(len(K))
    planes = []
    for j, p in enumerate(sim.planes):
        d = _complex(float(real[j] + pool[j]), float(imag[j]), float(rate[j]), float(separation[j]))
        d.update({"plane": p.n, "k": int(sim.K[j]), "pruned_excess": round(float(real[j]), 5),
                  "pool_excess": round(float(pool[j]), 5), "pool_would_enter": round(float(enter[j]), 2)})
        planes.append(d)
    tr, ti = float(np.mean(real + pool)), float(np.mean(imag))
    tot = _complex(tr, ti, float(np.mean(rate)), float(np.mean(separation)))
    tot.update({"pruned_excess": round(float(np.mean(real)), 5), "pool_excess": round(float(np.mean(pool)), 5),
                "pool_would_enter": round(float(enter.sum()), 2)})
    tot["reading"] = (f"crossing rate {tot['crossing_rate']:.2f} x separation {tot['separation']:.3f} = "
                      f"potential {ti:.4f}i; pruned excess {tot['pruned_excess']:.4f} + pool "
                      f"{tot['pool_excess']:.4f} = real {tr:.4f}; phase {tot['phase_deg']:.0f} deg")
    return {"planes": planes, "total": tot}


def k_profile(sim: Simulation, lo: int = 1, hi: Optional[int] = None) -> dict[str, dict[int, float]]:
    """For each plane, the crossing rate of free emission at every voice size K' in [lo, hi]:
    the share of the voice that changes between two scenarios if the plane voiced its K'
    best tokens.  Its minimum is where the plane's scores have a gap the noise does not
    cross.  Free emission ignores the coupling, so this is the estimate the mirror checks."""
    T = sim.S.shape[1]
    hi = int(min(hi if hi is not None else max(2 * int(sim.K.max()), 2), T))
    lo = int(max(1, min(lo, hi)))
    out: dict[str, dict[int, float]] = {}
    n = sim.S.shape[0]
    for j, p in enumerate(sim.planes):
        Sp = sim.S[:, :, j].astype(float)                        # (n, T)
        rank = np.argsort(-Sp, axis=1, kind="stable")            # exactly K per scenario, ties by rank
        prof = {}
        for kk in range(lo, hi + 1):
            sel = np.zeros_like(Sp, dtype=bool)
            np.put_along_axis(sel, rank[:, :kk], True, axis=1)
            f = sel.mean(axis=0)
            prof[kk] = round(float((2.0 * f * (1.0 - f)).sum() / (2.0 * kk)), 5)
        out[p.n] = prof
    return out


def report(sim: Simulation, persist: bool = False) -> dict[str, Any]:
    """The JSON a run publishes (GET /lambda)."""
    mesh, T, P, n = sim.mesh, sim.S.shape[1], len(sim.planes), sim.n
    sel_pri = sim.milp.mean(axis=0)
    sel_free = sim.free.mean(axis=0)
    C = len(sim.live)
    lambda_tokens, total_lam = [], float(sim.lam.mean(axis=0).sum()) if C else 0.0
    for j, c in enumerate(sim.live):
        vals = sim.lam[:, j].tolist()
        share = float(sel_pri[sim.member == j].sum())
        lambda_tokens.append({
            "token": f"λ:{c['id']}", "cluster": c["id"], "commodity": c["commodity"],
            "mean": round(float(np.mean(vals)), 6), "std": round(float(np.std(vals)), 6),
            "p10": round(_q(vals, 10), 6), "p50": round(_q(vals, 50), 6), "p90": round(_q(vals, 90), 6),
            "binding_rate": round(float(np.mean(np.array(vals) > 1e-6)), 3),
            "floor": int(sim.q[j]), "capacity": int(sim.capacity[j]), "expected_share": round(share, 3),
            "anchors_found": c["anchors_found"]})
    unanchored = [c["id"] for c in sim.anchored if c["index"] is None]

    shadow = []
    for t in range(T):
        for pi, p in enumerate(sim.planes):
            diff = float(sel_free[t, pi] - sel_pri[t, pi])
            if abs(diff) >= 0.25:
                shadow.append({"token": mesh.tokens[t], "plane": p.n,
                               "kind": "held_back" if diff > 0 else "pulled_in",
                               "weight": round(abs(diff), 3),
                               "cluster": sim.live[sim.member[t]]["id"] if sim.member[t] >= 0 else None,
                               "reduced_score": round(float(sim.red[t, pi]), 4)})
    shadow.sort(key=lambda r: -r["weight"])

    # Signal-to-noise per plane across the scenarios: the plane's mean direction against
    # the spread of its draws (sentence bootstrap + Langevin noise).  Below 1 the mesh
    # cannot tell this movement from the others under QUIPU's own noise.
    snr = np.linalg.norm(sim.draws.mean(axis=0), axis=1) / np.maximum(
        np.linalg.norm(sim.draws.std(axis=0), axis=1), 1e-12)
    voiced = [dict(v) for v in sim.voiced]
    by_plane = {p.n: i for i, p in enumerate(sim.planes)}
    for v in voiced:
        if v["plane"] in by_plane:
            v["snr"] = round(float(snr[by_plane[v["plane"]]]), 3)
            v["resolvable"] = bool(snr[by_plane[v["plane"]]] >= 1.0)
    emission = []
    for pi, p in enumerate(sim.planes):
        kp = int(sim.K[pi])
        order = np.argsort(-sel_pri[:, pi])
        toks = [mesh.tokens[t] for t in order[:kp] if sel_pri[t, pi] >= 0.5]
        emission.append({"plane": p.n, "title": p.title, "k": kp, "tokens": toks, "text": " ".join(toks),
                         "snr": round(float(snr[pi]), 3),
                         "frequencies": [[mesh.tokens[t], round(float(sel_pri[t, pi]), 3)] for t in order[:kp]]})

    projections = [{"cluster": lt["cluster"], "commodity": lt["commodity"],
                    "lambda_p10": lt["p10"], "lambda_p50": lt["p50"], "lambda_p90": lt["p90"],
                    "binding_rate": lt["binding_rate"],
                    "attention": round(lt["mean"] / total_lam, 4) if total_lam > 0 else 0.0,
                    "expected_share": lt["expected_share"], "floor": lt["floor"]}
                   for lt in lambda_tokens]
    exact_n = sim.exact_n
    out = {
        "ok": True, "at": time.time(), "seconds": sim.seconds, "seed": sim.seed,
        "essay": sim.essay, "planes": voiced,
        "problem": {"scenarios": n, "k_per_plane": {p.n: int(sim.K[i]) for i, p in enumerate(sim.planes)},
                    "budget": int(sim.K.sum()), "equity": sim.equity, "candidates": T,
                    "beyond_pool_checked": int(sim.ext_S.shape[1]), "clusters": C, "floor": sim.floor,
                    "sigma": round(mesh.sigma, 5),
                    "planes": "contrasted" if sim.contrast and P > 1 else "centred",
                    "plane_cosine_max": _max_offdiag(planes_dirs(sim.planes, mesh, sim.contrast)),
                    "primal": "exact (HiGHS)" if exact_n == n else
                              ("Lagrangian repair" if exact_n == 0 else f"exact in {exact_n}/{n}")},
        "bounds": {"dual_mean": round(float(np.mean(sim.bounds)), 5),
                   "primal_mean": round(float(np.mean(sim.primals)), 5),
                   "gap_mean": round(float(np.mean(sim.gaps)), 6), "gap_max": round(float(np.max(sim.gaps)), 6)},
        "potential": potential(sim),
        "k_profile": k_profile(sim, 1, min(24, max(2 * int(sim.K.max()), 2))),
        "lambda_tokens": lambda_tokens, "unanchored_clusters": unanchored,
        "shadow_tokens": shadow[:200], "emission": emission, "projections": projections,
        "note": "λ and the potential are in units of centred cosine (alignment with the Essay), not "
                "currency. Projections propose nothing and change no price.",
    }
    if persist:
        from . import brain_kv
        brain_kv.kv_set_json(KV_LATEST, out)
    return out


def run(*, scenarios: int = 64, k: Any = 8, equity: float = 0.5, n_candidates: int = 384,
        n_extension: int = 256, iters: int = 200, seed: int | None = None, threshold: float = 0.3,
        clusters: Optional[list[dict[str, Any]]] = None, essay: Optional[str] = None,
        mesh: Optional[Mesh] = None, persist: bool = True, contrast: bool = True) -> dict[str, Any]:
    sim, err = simulate(scenarios=scenarios, k=k, equity=equity, n_candidates=n_candidates,
                        n_extension=n_extension, iters=iters, seed=seed, threshold=threshold,
                        clusters=clusters, essay=essay, mesh=mesh, contrast=contrast)
    if err is not None:
        return err
    return report(sim, persist=persist)


def planes_dirs(planes: list[Plane], mesh: Mesh, contrast: bool) -> np.ndarray:
    D = np.array([plane_direction(_words(p.text), mesh)[0] for p in planes])
    if contrast and len(D) > 1:
        D = D - D.mean(axis=0)
    return _unit(D)


def _max_offdiag(U: np.ndarray) -> float:
    if len(U) < 2:
        return 0.0
    G = U @ U.T
    np.fill_diagonal(G, -np.inf)
    return round(float(G.max()), 4)


def latest() -> dict[str, Any]:
    from . import brain_kv
    return brain_kv.kv_get_json(KV_LATEST, None) or {"ok": False, "error": "no run yet"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shadow_tokens", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--scenarios", type=int, default=64)
    r.add_argument("--k", type=int, default=8)
    r.add_argument("--equity", type=float, default=0.5)
    r.add_argument("--candidates", type=int, default=384)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--essay", default=None)
    sub.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        out = run(scenarios=a.scenarios, k=a.k, equity=a.equity, n_candidates=a.candidates,
                  seed=a.seed, essay=a.essay)
    else:
        out = latest()
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
