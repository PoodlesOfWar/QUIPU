"""mesh_flow — one pass from the axis cells through the VNN to proposed shadow edges.

The chain (run once per self-organising step, after the PCM cells and bond shear):

    PCM axis pressures p_k                       (qpsi.pcm_cells, rADAM-programmed)
      → Mott drive  I_k = I_rheo + span·p_k      (qpsi.mott_spintronic, 7 LIF neurons)
      → spikes written to the racetrack / latent
      → poADAM (poissonic) vs rADAM (realised)   Δ = p_rADAM − p_poADAM
      → TypedViscousCoupler: stability S(Δ) = φ^(1−2|Δ|), rheology and bond-shear readings,
        target = the live 7-D MESH state (axes, observer as entirety)
      → VNN step over a read-only copy of the MESH graph
        (mesh_slm_vocab / mesh_slm_embed / mesh_slm_quipu: tokens → nodes, Quipu edges → graph)
      → shadow edges: pairs whose 7-D states the flow brought together
          * ``lift``: an existing Quipu edge whose two ends converged
          * ``new``:  a pair with no edge that the flow made near-parallel (similarity ≥ 0.9)
        each lift damped by S(Δ)/φ, so a realisation that diverges from its potential
        proposes less
      → magic on the pathways (``qpsi.magic``): each plane's pressure and rADAM phase as a
        matter–antimatter pair, the antimatter half held by the one-positron quantum cyclotron.
        A shadow edge's *pathway* is the plane mix the flow moved its two tokens along,
        π_k ∝ |ΔZ_u,k| + |ΔZ_v,k|; its magic is Σ π_k·magic_k (matter, held antimatter,
        unheld, CPT asymmetry likewise), and its lift is scaled by
            g = (1 + (φ − 1)·D) · held                  ∈ [0, φ], then re-capped,
        D = Σ π_k (1 − 2^{−M₂,k})/(9/16): the pathway's mean Clifford-orbit design
        indistinguishability, normalised at the proved two-qubit maximum
        (``magic.operational_gain``; Bittel & Leone; log₂(16/7) bound),
        held = (matter + antimatter)/(matter + antimatter + unheld).  Non-stabilizer
        pathways strengthen, unheld antimatter weakens.
      → shadow tokens: every endpoint of a shadow edge, on the plane its own state moved
        most, with its own pathway magic and the largest lift among its edges.
      → recurrence: an edge or token is *proposed* only once it appears in ≥ 2 distinct
        input states.

Nothing here writes ``mesh_slm_*``: proposals carry
``realisation: "none: through the six gates, two signers and the grant"``, the same
contract as ``knot_shadow``.  Writes go only to ``brain_kv["entirety:mesh_flow:*"]``
(and the coupler's own ``entirety:vnn:latest`` report), on the step's connection.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from typing import Any, Mapping, Optional, Sequence

from . import mott_spintronic as ms
from .knot_shadow import MIN_SESSIONS, REALISATION, recurring

KV_PREFIX: str = "entirety:mesh_flow:"
KV_STATE: str = KV_PREFIX + "state"
KV_HISTORY: str = KV_PREFIX + "history"
KV_PROPOSALS: str = KV_PREFIX + "proposals"
KV_LATEST: str = KV_PREFIX + "latest"
KV_MAGIC: str = KV_PREFIX + "magic"

AXES: tuple[str, ...] = ("vision", "touch", "smell", "body", "brain", "perception", "entirety")
WINDOW_STEPS: int = 64          # Mott steps per self-organising step
DRIVE_SPAN: float = 0.10        # drive above rheobase at pressure 1
MAX_NODES: int = 128
EDGE_LIFT_MAX: float = 0.05     # same ceiling as the held ACRE tool's edge proposals
NEW_EDGE_WEIGHT: float = 0.05
NEW_EDGE_MIN_SIM: float = 0.90
MIN_GAIN: float = 1e-6
MAX_PER_KIND: int = 32
HISTORY_CAP: int = 64
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0


def _kv_set(cn: sqlite3.Connection, key: str, value: Any) -> None:
    if not key.startswith(KV_PREFIX):
        raise PermissionError(f"mesh_flow writes only {KV_PREFIX}* keys, not {key!r}")
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def _kv_get(cn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return default
    if not row:
        return default
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return default


def mott_drive(pressures: Mapping[str, float], cfg: ms.MottSpinConfig = ms.MottSpinConfig()) -> list[float]:
    """I_k = 1/τ + span·p_k: a cell at pressure 0 sits exactly at rheobase and stays silent."""
    rheobase = cfg.v_threshold / cfg.tau_leak
    out = []
    for a in AXES:
        p = float(pressures.get(a, 0.0) or 0.0)
        p = min(1.0, max(0.0, p)) if math.isfinite(p) else 0.0
        out.append(rheobase + DRIVE_SPAN * p)
    return out


def _cos_matrix(Z):
    import numpy as np
    n = np.linalg.norm(Z, axis=1)
    ok = n > 1e-9
    U = np.zeros_like(Z)
    U[ok] = Z[ok] / n[ok, None]
    return U @ U.T, ok


def pathway_planes(Z0, Z1, nodes: Sequence[int]) -> list[float]:
    """Plane mix π_k ∝ Σ_nodes |Z1 − Z0|_k (falls back to |Z1|_k when nothing moved)."""
    import numpy as np
    Z0, Z1 = np.asarray(Z0, dtype=float), np.asarray(Z1, dtype=float)
    w = np.zeros(Z1.shape[1])
    for u in nodes:
        w += np.abs(Z1[u] - Z0[u])
    if not np.isfinite(w).all() or w.sum() <= 0.0:
        w = np.zeros(Z1.shape[1])
        for u in nodes:
            w += np.abs(Z1[u])
    s = float(w.sum())
    return [float(x) / s for x in w] if s > 0.0 else [1.0 / Z1.shape[1]] * Z1.shape[1]


def pathway_magic(planes: Sequence[float], plane_magic: Optional[Mapping[str, Mapping]]) -> dict[str, Any]:
    """Magic along a pathway: Σ π_k · (magic, matter, antimatter, unheld, cpt_asymmetry, design)_k.

    gain = (1 + (φ − 1)·design) · held, with design = Σ π_k·(1 − 2^{−M₂,k})/(9/16) the
    pathway's mean design indistinguishability (``magic.design_indistinguishability``).
    """
    from .magic import design_indistinguishability
    keys = ("magic", "matter", "antimatter", "unheld", "cpt_asymmetry", "design")
    acc = {k: 0.0 for k in keys}
    for a, pk in zip(AXES, planes):
        rec = (plane_magic or {}).get(a) or {}
        for k in keys:
            if k == "design":
                acc[k] += pk * design_indistinguishability(float(rec.get("magic", 0.0) or 0.0))
            else:
                acc[k] += pk * float(rec.get(k, 0.0) or 0.0)
    carried = acc["matter"] + acc["antimatter"] + acc["unheld"]
    held = (acc["matter"] + acc["antimatter"]) / carried if carried > 1e-15 else 1.0
    gain = (1.0 + (PHI - 1.0) * min(1.0, acc["design"])) * held
    top = max(range(len(planes)), key=lambda k: planes[k]) if planes else 0
    return {**{k: round(v, 9) for k, v in acc.items()}, "held": round(held, 9),
            "gain": round(gain, 9), "plane": AXES[top], "planes": [round(x, 6) for x in planes]}


def shadow_edges(Z0, Z1, node_ids: Sequence[int], edges: Sequence[tuple[int, int, float]],
                 damping: float = 1.0,
                 plane_magic: Optional[Mapping[str, Mapping]] = None) -> dict[str, list[dict]]:
    """Edges the VNN flow is pulling together, between the pre- and post-step states.

    ``lift``: existing edge (u, v) with cos₁ > cos₀; lift = EDGE_LIFT_MAX·damping·(cos₁−cos₀)/(1−cos₀).
    ``new``: no edge either way, cos₁ ≥ NEW_EDGE_MIN_SIM and cos₁ > cos₀;
             weight = NEW_EDGE_WEIGHT·damping·cos₁.

    With ``plane_magic`` (``magic.reading(...)["planes"]``) each edge carries its
    ``pathway`` and its lift is multiplied by the pathway gain, then re-capped at
    the kind's ceiling.  Without it the gain is 1 and the result is unchanged.
    """
    import numpy as np
    C0, ok0 = _cos_matrix(np.asarray(Z0, dtype=float))
    C1, ok1 = _cos_matrix(np.asarray(Z1, dtype=float))
    ok = ok0 & ok1
    d = min(1.0, max(0.0, float(damping)))
    linked = set()
    lifts = []
    for u, v, _w in edges:
        linked.add((u, v))
        linked.add((v, u))
        if u == v or not (ok[u] and ok[v]):
            continue
        c0, c1 = float(C0[u, v]), float(C1[u, v])
        if c1 - c0 > MIN_GAIN:
            lift = EDGE_LIFT_MAX * d * (c1 - c0) / max(1e-9, 1.0 - c0)
            pw = pathway_magic(pathway_planes(Z0, Z1, (u, v)), plane_magic) if plane_magic else None
            g = pw["gain"] if pw else 1.0
            lifts.append({"src": node_ids[u], "dst": node_ids[v], "kind": "lift", "u": u, "v": v,
                          "cos0": c0, "cos1": c1, "lift": min(EDGE_LIFT_MAX, lift * g), "pathway": pw})
    news = []
    n = len(node_ids)
    for u in range(n):
        if not ok[u]:
            continue
        for v in range(u + 1, n):
            if not ok[v] or (u, v) in linked:
                continue
            c0, c1 = float(C0[u, v]), float(C1[u, v])
            if c1 >= NEW_EDGE_MIN_SIM and c1 - c0 > MIN_GAIN:
                news.append({"src": node_ids[u], "dst": node_ids[v], "kind": "new", "u": u, "v": v,
                             "cos0": c0, "cos1": c1, "lift": NEW_EDGE_WEIGHT * d * c1})
    news.sort(key=lambda e: (-(e["cos1"] - e["cos0"]), -e["cos1"]))
    news = news[:MAX_PER_KIND]
    for e in news:
        pw = pathway_magic(pathway_planes(Z0, Z1, (e["u"], e["v"])), plane_magic) if plane_magic else None
        e["pathway"] = pw
        if pw:
            e["lift"] = min(NEW_EDGE_WEIGHT, e["lift"] * pw["gain"])
    lifts.sort(key=lambda e: -e["lift"])
    return {"lift": lifts[:MAX_PER_KIND], "new": news}


def shadow_tokens(Z0, Z1, node_ids: Sequence[int], shadow: Mapping[str, Sequence[Mapping]],
                  plane_magic: Optional[Mapping[str, Mapping]] = None) -> list[dict]:
    """Every endpoint of a shadow edge: its own plane (where its state moved most), its own
    pathway magic, and the largest lift among the shadow edges it sits on."""
    best: dict[int, float] = {}
    for kind in ("lift", "new"):
        for e in shadow.get(kind) or []:
            for k in ("u", "v"):
                best[e[k]] = max(best.get(e[k], 0.0), float(e["lift"]))
    out = []
    for u, lift in best.items():
        pw = pathway_magic(pathway_planes(Z0, Z1, (u,)), plane_magic)
        out.append({"node": node_ids[u], "plane": pw["plane"], "lift": lift, "pathway": pw})
    out.sort(key=lambda t: -t["lift"])
    return out


def _tokens(cn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, str]:
    ids = sorted({int(i) for i in ids})
    if not ids:
        return {}
    q = ",".join("?" * len(ids))
    return {int(r[0]): str(r[1]) for r in cn.execute(
        f"SELECT token_id, token FROM mesh_slm_vocab WHERE token_id IN ({q})", ids).fetchall()}


def run(cn: sqlite3.Connection, *, pressures: Mapping[str, float], axes: Mapping[str, float],
        observer: float = 0.0, rheology_reading: Optional[Mapping[str, Any]] = None,
        shear_reading: Optional[Mapping[str, Any]] = None, now: Optional[float] = None,
        max_nodes: int = MAX_NODES, phases: Optional[Mapping[str, float]] = None,
        source_axes: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """One pass of the chain on the step's connection; returns the published summary."""
    from . import viscous_interaction as vi

    now = time.time() if now is None else float(now)
    cfg = ms.MottSpinConfig()
    saved = _kv_get(cn, KV_STATE, {}) or {}
    st = ms.MottSpinState()
    if isinstance(saved.get("v"), list) and len(saved["v"]) == ms.PLANES:
        st.v = [min(cfg.v_threshold, max(0.0, float(x))) for x in saved["v"]]
    po, rr = dict(saved.get("poadam") or {}), dict(saved.get("radam") or {})

    # 1. Axis cells → Mott core → poADAM vs rADAM
    drive = mott_drive(pressures, cfg)
    win = ms.run_window(st, drive, WINDOW_STEPS, cfg, po_state=po, r_state=rr)
    ms.read_and_clear(st)        # the window's spikes are read out; memory frees for the next step
    diff = win["radam_differential"]

    # 2. MESH graph → VNN under the relational stability number
    try:
        vnn0 = vi.load_vnn_state_from_conn(cn, max_nodes=max_nodes)
    except sqlite3.OperationalError:          # a brain without MESH tables yet: no graph to flow over
        import numpy as np
        vnn0 = vi.VNNState(node_ids=[], Z=np.zeros((0, 7)), edges=[])
    target = [float(axes.get(a, 0.0) or 0.0) for a in AXES[:-1]] + [float(observer or 0.0)]
    coupler = vi.TypedViscousCoupler(ns_cfg=vi.NavierStokesConfig(dims=(8, 8), nu=0.03, dt=0.01))
    ns0 = vi.NavierStokesState.taylor_green(coupler.ns_cfg, amplitude=0.4)
    # 3. Information theory → magic: each plane's pressure and rADAM phase as a matter–antimatter
    #    pair (antimatter half held by the one-positron quantum cyclotron); it weights the pathways.
    from . import magic as _magic
    mg = _magic.reading(pressures, phases, source_axes=source_axes)
    mg["at"] = now

    shadow = {"lift": [], "new": []}
    stoks: list[dict] = []
    report = None
    if vnn0.node_ids:
        _, vnn1, report = coupler.step(ns0, vnn0, rheology_reading=rheology_reading,
                                       shear_reading=shear_reading, radam_differential=diff,
                                       target_mesh7=target)
        vi.publish_vnn_report(cn, report)
        damping = win["stability_number"] / PHI
        shadow = shadow_edges(vnn0.Z, vnn1.Z, vnn0.node_ids, vnn0.edges, damping, mg["planes"])
        stoks = shadow_tokens(vnn0.Z, vnn1.Z, vnn0.node_ids, shadow, mg["planes"])

    # 4. Name the edges and tokens and fold them into the recurrence history
    toks = _tokens(cn, [e[k] for kind in shadow.values() for e in kind for k in ("src", "dst")])

    def _pw(pw):
        return {k: pw[k] for k in ("plane", "magic", "matter", "antimatter", "unheld", "cpt_asymmetry",
                                   "design", "held", "gain")} if pw else None

    named = [{"src": toks.get(e["src"], str(e["src"])), "dst": toks.get(e["dst"], str(e["dst"])),
              "lobe": e["kind"], "lift": round(e["lift"], 6), "cos0": round(e["cos0"], 6),
              "cos1": round(e["cos1"], 6), "pathway": _pw(e.get("pathway"))}
             for kind in ("lift", "new") for e in shadow[kind]]
    named_tokens = [{"token": toks.get(t["node"], str(t["node"])), "lobe": t["plane"],
                     "lift": round(t["lift"], 6), "pathway": _pw(t["pathway"])} for t in stoks]
    session = hashlib.sha256(json.dumps({
        "drive": [round(x, 6) for x in drive], "target": [round(x, 6) for x in target],
        "nodes": vnn0.node_ids, "delta": round(diff["delta"], 6),
        "edges": sorted((e["src"], e["dst"], e["lobe"]) for e in named),
    }, sort_keys=True).encode()).hexdigest()[:16]
    hist = [h for h in (_kv_get(cn, KV_HISTORY, []) or []) if isinstance(h, dict)]
    if not any(h.get("session") == session for h in hist):
        hist.append({"session": session, "at": now, "tokens": named_tokens, "edges": named})
        hist = hist[-HISTORY_CAP:]
    rec = recurring(hist, MIN_SESSIONS)
    proposals = rec["edges"]
    for p in proposals:
        p["kind"] = p.pop("lobe")
    token_proposals = rec["tokens"]
    for p in token_proposals:
        p["plane"] = p.pop("lobe")
    _latest_pw = {}
    for h in hist:
        for e in h.get("edges") or []:
            _latest_pw[(e["src"], e["dst"], e["lobe"])] = e.get("pathway")
        for t in h.get("tokens") or []:
            _latest_pw[(t["token"], t["lobe"])] = t.get("pathway")
    for p in proposals:
        p["pathway"] = _latest_pw.get((p["src"], p["dst"], p["kind"]))
    for p in token_proposals:
        p["pathway"] = _latest_pw.get((p["token"], p["plane"]))

    summary = {
        "at": now,
        "magic": mg["total"],
        "session": session,
        "drive": [round(x, 6) for x in drive],
        "spikes": win["spikes"], "written": win["written"], "latent": win["latent"],
        "p_poadam": win["p_poadam"], "p_radam": win["p_radam"],
        "delta": diff["delta"], "stability_number": win["stability_number"],
        "nodes": len(vnn0.node_ids), "graph_edges": len(vnn0.edges),
        "vnn_graph_cfl": (report.vnn_diagnostics.get("graph_cfl") if report else None),
        "shadow_this_step": {"lift": len(shadow["lift"]), "new": len(shadow["new"]), "tokens": len(stoks)},
        "proposed": len(proposals),
        "proposed_tokens": len(token_proposals),
        "pathway_magic": (round(sum(e["pathway"]["magic"] for e in named if e["pathway"]) /
                                max(1, sum(1 for e in named if e["pathway"])), 9) if named else 0.0),
        "realisation": REALISATION,
    }
    _kv_set(cn, KV_STATE, {"v": st.v, "poadam": po, "radam": rr})
    _kv_set(cn, KV_HISTORY, hist)
    _kv_set(cn, KV_PROPOSALS, {"at": now, "min_sessions": MIN_SESSIONS, "edges": proposals,
                               "tokens": token_proposals})
    _kv_set(cn, KV_MAGIC, mg)
    _kv_set(cn, KV_LATEST, summary)
    return summary


def latest(cn: sqlite3.Connection) -> dict:
    rec = _kv_get(cn, KV_LATEST, {})
    return rec if isinstance(rec, dict) else {}


def proposals(cn: sqlite3.Connection) -> dict:
    rec = _kv_get(cn, KV_PROPOSALS, {})
    return rec if isinstance(rec, dict) else {}


__all__ = ["KV_PREFIX", "KV_STATE", "KV_HISTORY", "KV_PROPOSALS", "KV_LATEST", "KV_MAGIC", "AXES",
           "mott_drive", "pathway_planes", "pathway_magic", "shadow_edges", "shadow_tokens",
           "run", "latest", "proposals"]
