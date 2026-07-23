"""tool_forge — Ring 5 (Refinement -> Toolforge) for the QUIPU System Entirety.

Autonomous synthesis of digital Tools/Modules/Nodes from knowledge clusters
emerging in the QUIPU mesh corpus.

This is a clean-room, QUIPU-native adaptation of the parent Supply Chain
Brain's ``tool_forge.py``. Two structural differences from the parent, both
deliberate:

1. **Corpus source.** The parent scanned ``corpus_entity`` / ``corpus_edge`` /
   ``learning_log`` — Ring 4 application tables (ERP-linked knowledge graph)
   that were never extracted into QUIPU. This module scans QUIPU's own Ring 4
   analog instead: ``mesh_slm_vocab`` (tokens = concepts) and ``mesh_slm_quipu``
   (Hebbian edges between them). A vocab token with high degree, high
   frequency, and recent ``last_seen`` plays the same role a high-degree
   corpus entity played in the parent.

2. **Digital-only.** The parent's PHYSICAL and ERP synthesis paths produced
   ``PhysicalToolSpec`` / ERP-rectification HITL items using Astec-specific
   prompts and an Oracle/Azure-linked HITL review page — none of which exist
   in QUIPU (no ERP tables, no Streamlit, no employer-specific prompts). Only
   the DIGITAL auto-implementation path is carried over: it requires zero
   external LLM calls in either version (the parent's OpenRouter step was
   always a guarded *fallback*, never primary).

Synthesis mechanism (unchanged from parent)
--------------------------------------------
    novelty_score x certainty_score >= AUTO_IMPL_THRESHOLD (default 0.55)

    novelty_score   = fraction of the cluster's neighbour tokens not yet
                       operationalised as tools
    certainty_score = log(degree) x recency x source-diversity proxy

Generated artifacts land in ``src/quipu/tools/generated/<name>.py`` with a
``manifest.json`` alongside, and are validated (AST-only, no unsafe imports,
no exec/eval) before being written to disk.

Public API
----------
    forge_round(max_tools=3)   -> dict   scan + synthesize this cycle
    load_generated_tools()     -> dict   name -> callable
    forged_tool_names()        -> set    safe-names already operationalised
"""
from __future__ import annotations

import ast
import json
import logging
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .local_store import open_conn
from . import brain_kv
from . import mesh_slm

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_TOOLS_DIR = _HERE / "tools" / "generated"
_MANIFEST = _TOOLS_DIR / "manifest.json"

_SEEN_KEY = "tool_forge:seen"
_FORGE_LOG_KEY = "tool_forge:last_round"
_THE_OTHER_KEY = "entirety:the_other"

# Auto-implementation thresholds (identical to the parent's tri-path gate).
AUTO_IMPL_THRESHOLD = 0.55
NOVELTY_THRESHOLD = 0.35
CERTAINTY_THRESHOLD = 0.40

# Resonance gate — the "shared Hz wavelength" requirement, expressed on the
# 52-hertz-whale band (see mesh_slm.LONELY_WHALE_HZ). A concept is forged only
# when it resonates at or above 52 Hz: woven into the shared concept graph
# (mutual centrality) AND sitting on the shared analytical wavelength rather
# than the 7th/entirety "Other" axis. This replaces the parent's typed-entity
# eligibility (Concept/Mechanism/MLPaper) that QUIPU lacks. Concepts below the
# line — lonely callers, single-source nouns, fiction embedded on the entirety
# axis — are kept in the mesh and shared through the 8th-D field, but not forged.
RESONANCE_FLOOR_HZ = 52.0

_NODE_KEYWORDS = {
    "monitor", "detect", "alert", "watch", "stream", "continuous",
    "optimize", "track", "observe", "anomaly", "drift", "velocity",
    "forecast", "predict", "trend", "signal", "pattern",
}


# ===========================================================================
# QUIPU-native cluster scan — mesh_slm_vocab / mesh_slm_quipu
# ===========================================================================
def _ensure_mesh_tables() -> None:
    """Trigger mesh_slm's own DDL so vocab/quipu tables exist before we query them."""
    try:
        mesh_slm.state_summary()
    except Exception as exc:
        log.debug("tool_forge: mesh_slm not ready yet: %s", exc)


def _scan_vocab_clusters(limit: int = 20) -> list[dict]:
    """Return top mesh tokens ranked by synthesis priority, QUIPU's Ring-4 analog
    of the parent's corpus-entity cluster scan.

    score = log(degree + 1) x recency_weight x novelty_factor
        degree         = distinct neighbour tokens in mesh_slm_quipu (src or dst)
        recency_weight = normalised recency of last_seen among candidates
        novelty_factor = 0.0 if already forged (in tool_forge:seen), else 1.0
    """
    _ensure_mesh_tables()
    seen = _load_seen()

    cn = open_conn()
    try:
        rows = cn.execute(
            """
            WITH deg AS (
                SELECT token_id, COUNT(*) AS degree FROM (
                    SELECT src AS token_id FROM mesh_slm_quipu
                    UNION ALL
                    SELECT dst AS token_id FROM mesh_slm_quipu
                ) GROUP BY token_id
            )
            SELECT v.token_id, v.token, v.freq, v.last_seen,
                   COALESCE(d.degree, 0) AS degree
            FROM mesh_slm_vocab v
            LEFT JOIN deg d ON v.token_id = d.token_id
            WHERE COALESCE(d.degree, 0) > 1
            ORDER BY degree DESC
            LIMIT 400
            """
        ).fetchall()
    except sqlite3.OperationalError as exc:
        log.debug("tool_forge: vocab scan unavailable (%s)", exc)
        cn.close()
        return []

    if not rows:
        cn.close()
        return []

    # Recency normalisation across the candidate set.
    def _ts(v: str | None) -> float:
        if not v:
            return 0.0
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    times = [_ts(r["last_seen"]) for r in rows]
    t_min, t_max = min(times), max(times)
    span = max(t_max - t_min, 1.0)

    clusters: list[dict] = []
    for row in rows:
        tid = row["token_id"]
        eid = f"tok:{tid}"
        degree = int(row["degree"])
        recency_w = ((_ts(row["last_seen"]) - t_min) / span) + 0.1
        novelty = 0.0 if eid in seen else 1.0
        score = math.log(degree + 1) * recency_w * novelty
        if score <= 0:
            continue
        clusters.append({
            "entity_id": eid,
            "token_id": tid,
            "label": row["token"],
            "degree": degree,
            "freq": int(row["freq"]),
            "score": score,
        })

    clusters.sort(key=lambda x: x["score"], reverse=True)
    top = clusters[:limit]

    # Enrich with neighbour tokens (the "related concepts" a tool's docstring cites).
    for cl in top:
        tid = cl["token_id"]
        neighbors = cn.execute(
            """
            SELECT DISTINCT vb.token AS label, q.weight
            FROM mesh_slm_quipu q
            JOIN mesh_slm_vocab vb ON (
                (q.src = ? AND vb.token_id = q.dst)
                OR (q.dst = ? AND vb.token_id = q.src)
            )
            LIMIT 20
            """,
            (tid, tid),
        ).fetchall()
        cl["neighbors"] = [{"label": r["label"], "rel": "quipu_edge"} for r in neighbors]

    cn.close()
    return top


def _load_seen() -> set[str]:
    raw = brain_kv.kv_get_json(_SEEN_KEY, [])
    return set(raw) if isinstance(raw, list) else set()


def _save_seen(seen: set[str]) -> None:
    brain_kv.kv_set_json(_SEEN_KEY, sorted(seen))


# ===========================================================================
# Tool-name safety + code validation (identical logic to the parent)
# ===========================================================================
def _safe_name(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (label or "").lower())
    s = s.strip("_")[:40]
    return s or "concept"


def _validate_code(code: str) -> tuple[bool, str]:
    """AST-only validation: one tool_* function, no unsafe imports, no exec/eval."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    funcs = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name.startswith("tool_")]
    if not funcs:
        return False, "No top-level function named tool_* found"

    banned = {"os", "sys", "subprocess", "socket", "requests", "shutil", "pathlib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in banned:
                    return False, f"Unsafe import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in banned:
                return False, f"Unsafe from-import: {node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("exec", "eval", "compile", "__import__"):
                return False, f"Disallowed call: {node.func.id}()"
    return True, "ok"


def _classify_artifact(cluster: dict, code: str) -> dict:
    label_lower = cluster["label"].lower()
    code_lower = code.lower()
    is_node = (
        "# node_eligible" in code_lower
        or any(kw in label_lower for kw in _NODE_KEYWORDS)
        or any(kw in code_lower for kw in ("while true", "for _ in range", "sleep("))
    )
    is_module_anchor = cluster["degree"] >= 20 and len(cluster.get("neighbors", [])) >= 6
    artifact_type = "Node" if is_node else ("Module" if is_module_anchor else "Tool")
    return {"type": artifact_type, "is_node": is_node}


def _novelty_certainty_scores(cluster: dict, existing_tool_labels: set[str]) -> tuple[float, float]:
    """(novelty, certainty) for a mesh-token cluster — same formula as the parent,
    substituting mesh-native degree/freq for corpus-entity degree/recency_cnt.
    """
    neighbors = cluster.get("neighbors", [])
    if neighbors:
        novel = sum(1 for n in neighbors if _safe_name(n.get("label", "")) not in existing_tool_labels)
        novelty = novel / len(neighbors)
    else:
        novelty = 0.5

    degree = cluster.get("degree", 1)
    freq = cluster.get("freq", 1)
    deg_norm = math.log(degree + 1) / math.log(60)
    freq_norm = min(freq / 20.0, 1.0)
    certainty = min(deg_norm * 0.6 + freq_norm * 0.4, 1.0)
    return float(novelty), float(certainty)


def _should_auto_implement(novelty: float, certainty: float) -> bool:
    return (
        novelty >= NOVELTY_THRESHOLD
        and certainty >= CERTAINTY_THRESHOLD
        and novelty * certainty >= AUTO_IMPL_THRESHOLD
    )


# ===========================================================================
# Internal synthesis — pure stdlib templates, zero external calls
# ===========================================================================
def _corpus_header(cluster: dict) -> str:
    label = cluster["label"]
    degree = cluster.get("degree", 0)
    neighbors = cluster.get("neighbors", [])
    nb_labels = [n.get("label", "") for n in neighbors[:6] if n.get("label")]
    lines = [f"    # -- Mesh source: {label} ({degree} quipu edges) --"]
    if nb_labels:
        lines.append(f"    # Related tokens: {', '.join(nb_labels)}")
    return "\n".join(lines) + "\n"


def _synthesize_internal(cluster: dict) -> str:
    """Route to the best-fit computation family; always returns valid Python."""
    label = cluster["label"]
    safe = _safe_name(label)
    neighbors = [n.get("label", "").lower() for n in cluster.get("neighbors", [])]
    all_text = (label + " " + " ".join(neighbors)).lower()
    hdr = _corpus_header(cluster)

    if any(k in all_text for k in ("entropy", "bound", "holographic", "compress")):
        return _tpl_entropy_bounds(safe, hdr)
    if any(k in all_text for k in ("agent", "emergent", "coalition", "swarm")):
        return _tpl_agent_simulation(safe, hdr)
    if any(k in all_text for k in ("interface", "module", "decomposition", "coupling")):
        return _tpl_systems_decomp(safe, hdr)
    if any(k in all_text for k in ("constraint", "feasib", "flow", "allocation")):
        return _tpl_operations_research(safe, hdr)
    return _tpl_composite_score(safe, hdr)


def _tpl_entropy_bounds(safe: str, hdr: str = "") -> str:
    head = f"def tool_{safe}(data: dict) -> dict:\n{hdr}"
    body = '''\
    """Information-theoretic capacity bound for a mesh sub-network.
    Bekenstein-Hawking-inspired: max encodable information scales with the
    boundary (edges), not the bulk (nodes). Excess data beyond that bound
    yields diminishing returns.
    Inputs: n_nodes (int), n_edges (int), avg_signal (float 0-1), horizon (int).
    """
    import math
    n_nodes = max(1, int(data.get("n_nodes", 1)))
    n_edges = max(0, int(data.get("n_edges", 0)))
    signal  = min(1.0, max(0.0, float(data.get("avg_signal", 0.5))))
    horizon = max(1, int(data.get("horizon", 30)))
    if n_nodes < 2:
        return {"result": None, "explanation": "Need at least 2 nodes.",
                "confidence": 0.0, "holographic_bound_bits": None}
    max_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0
    bound = math.log2(n_edges + 1) * signal * math.log2(horizon + 1)
    actual = math.log2(n_nodes + 1) * signal
    efficiency = round(min(actual / (bound + 1e-9), 1.0), 4)
    return {
        "result": {"holographic_bound_bits": round(bound, 4),
                   "actual_info_bits": round(actual, 4),
                   "info_efficiency": efficiency, "edge_density": round(density, 4)},
        "explanation": f"Bound={round(bound,3)} bits, efficiency={efficiency}.",
        "confidence": round(min(0.7 + n_edges / 500, 0.94), 3),
    }
'''
    return head + body


def _tpl_agent_simulation(safe: str, hdr: str = "") -> str:
    head = f"def tool_{safe}(data: dict) -> dict:\n{hdr}"
    body = '''\
    """Emergent multi-agent interaction metrics over a mesh signal series.
    Computes amplification ratio and coalition stability among coupled nodes.
    Inputs: signal_series (list[float]), n_agents (int), coupling (float 0-1).
    """
    import statistics, math
    series = data.get("signal_series") or []
    n_ag = max(2, int(data.get("n_agents", 2)))
    coupling = min(1.0, max(0.0, float(data.get("coupling", 0.0))))
    if len(series) < 2:
        return {"result": None, "explanation": "Need >= 2 signal points.", "confidence": 0.0}
    mean_s = statistics.mean(series)
    std_s = statistics.stdev(series) if len(series) > 1 else 0
    window = max(2, len(series) // 4)
    smooth = [statistics.mean(series[max(0, i - window):i + 1]) for i in range(len(series))]
    std_smooth = statistics.stdev(smooth) if len(smooth) > 1 else 1e-9
    amp_ratio = round((std_s ** 2) / max(std_smooth ** 2, 1e-9), 4)
    damp = round(amp_ratio * (1 - coupling * 0.7), 4)
    coalition = round(1.0 / (1.0 + math.log(max(damp, 1.0))), 4)
    return {
        "result": {"amplification_ratio": amp_ratio, "dampened_ratio": damp,
                   "coalition_stability": coalition},
        "explanation": f"Amplification={amp_ratio}, coalition stability={coalition}.",
        "confidence": round(min(0.65 + len(series) / 80, 0.93), 3),
    }
# NODE_ELIGIBLE
'''
    return head + body


def _tpl_systems_decomp(safe: str, hdr: str = "") -> str:
    head = f"def tool_{safe}(data: dict) -> dict:\n{hdr}"
    body = '''\
    """Systems-decomposition complexity index over a mesh sub-graph.
    Complexity scales with interface density and coupling between components.
    Inputs: n_components (int), n_interfaces (int), coupling_ratio (float 0-1).
    """
    import math
    n = max(2, int(data.get("n_components", 2)))
    ni = max(0, int(data.get("n_interfaces", 0)))
    cr = min(1.0, max(0.0, float(data.get("coupling_ratio", 0.3))))
    max_ifaces = n * (n - 1) / 2
    density = ni / max_ifaces if max_ifaces > 0 else 0
    seci = round(density * (1.0 + cr * math.log(n + 1)), 4)
    modularity = round(1.0 / (1.0 + seci), 4)
    return {
        "result": {"seci": seci, "modularity_index": modularity,
                   "interface_density": round(density, 4)},
        "explanation": f"SECI={seci}, modularity={modularity}.",
        "confidence": round(min(0.75 + n / 100, 0.95), 3),
    }
'''
    return head + body


def _tpl_operations_research(safe: str, hdr: str = "") -> str:
    head = f"def tool_{safe}(data: dict) -> dict:\n{hdr}"
    body = '''\
    """Feasibility check for a mesh resource-allocation sub-problem.
    Inputs: supply (list[float]), demand (list[float]).
    """
    supply = data.get("supply") or []
    demand = data.get("demand") or []
    if not supply or not demand:
        return {"result": None, "explanation": "Provide supply and demand lists.", "confidence": 0.0}
    total_supply, total_demand = sum(supply), sum(demand)
    feas = round(min(total_supply / max(total_demand, 1e-9), 1.0), 4)
    return {
        "result": {"feasibility_ratio": feas, "supply_slack": round(total_supply - total_demand, 4)},
        "explanation": f"Feasibility ratio={feas}.",
        "confidence": round(min(0.7 + len(supply) / 50, 0.95), 3),
    }
'''
    return head + body


def _tpl_composite_score(safe: str, hdr: str = "") -> str:
    head = f"def tool_{safe}(data: dict) -> dict:\n{hdr}"
    body = '''\
    """Composite mesh-signal score: normalised weighted blend of input metrics.
    Generic fallback tool for clusters without a more specific template match.
    Inputs: metrics (dict[str, float]), weights (dict[str, float], optional).
    """
    metrics = data.get("metrics") or {}
    weights = data.get("weights") or {}
    if not metrics:
        return {"result": None, "explanation": "Provide a metrics dict.", "confidence": 0.0}
    total_w = sum(weights.get(k, 1.0) for k in metrics) or 1.0
    score = sum(v * weights.get(k, 1.0) for k, v in metrics.items()) / total_w
    return {
        "result": {"composite_score": round(score, 4), "n_metrics": len(metrics)},
        "explanation": f"Composite score={round(score, 4)} over {len(metrics)} metrics.",
        "confidence": round(min(0.5 + len(metrics) / 20, 0.9), 3),
    }
'''
    return head + body


# ===========================================================================
# Manifest + tool loading
# ===========================================================================
def _load_manifest() -> dict:
    if _MANIFEST.exists():
        try:
            return json.loads(_MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"tools": []}


def _save_manifest(manifest: dict) -> None:
    _TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    _MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _ensure_init_py() -> None:
    init_f = _TOOLS_DIR / "__init__.py"
    if not init_f.exists():
        _TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        init_f.write_text('"""Auto-generated QUIPU tools (tool_forge). Do not hand-edit."""\n',
                          encoding="utf-8")


def forged_tool_names() -> set[str]:
    manifest = _load_manifest()
    return {t.get("cluster_safe", "") for t in manifest.get("tools", [])}


def load_generated_tools() -> dict[str, Any]:
    """Dynamically import every forged tool file; return {tool_name: callable}."""
    import importlib.util
    manifest = _load_manifest()
    result: dict[str, Any] = {}
    for entry in manifest.get("tools", []):
        fpath = _TOOLS_DIR / entry.get("file", "")
        if not fpath.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location(entry["tool_name"], fpath)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            for attr in dir(mod):
                if attr.startswith("tool_") and callable(getattr(mod, attr)):
                    result[attr] = getattr(mod, attr)
        except Exception as exc:
            log.debug("load_generated_tools: skip %s - %s", fpath.name, exc)
    return result


# ===========================================================================
# Resonance — the shared-wavelength gate
# ===========================================================================
def _shared_reference() -> list[float] | None:
    """The self-state to resonate concepts against, from the shared 'the Other'
    signal on the brain_kv bus (falls back to None -> live mesh state)."""
    other = brain_kv.kv_get_json(_THE_OTHER_KEY, None)
    if isinstance(other, dict):
        ss = other.get("self_state")
        if isinstance(ss, list) and len(ss) == 7:
            return [float(x) for x in ss]
    return None


def _cluster_resonance_hz(cn: sqlite3.Connection, cluster: dict,
                          reference: list[float] | None) -> float:
    """The cluster concept's resonance on the 0-100 Hz band (see
    mesh_slm.LONELY_WHALE_HZ) — structural centrality x shared-wavelength
    directionality, scaled to Hz."""
    tid = cluster.get("token_id")
    if tid is None:
        return 0.0
    try:
        return mesh_slm.concept_resonance_hz(cn, int(tid), reference)
    except Exception:
        return 0.0


# ===========================================================================
# Main forge round
# ===========================================================================
def forge_round(max_tools: int = 3) -> dict:
    """Scan the mesh corpus, synthesize + validate digital tools.

    Digital-only (see module docstring for why physical/ERP paths are absent).
    Returns a summary dict compatible with the systemic_refinement_agent loop.
    """
    ts_start = datetime.now(timezone.utc).isoformat()
    summary = {
        "ts": ts_start, "tools_forged": 0, "skipped": 0,
        "skipped_low_resonance": 0, "errors": [], "candidates": 0,
    }

    _ensure_init_py()
    seen = _load_seen()
    manifest = _load_manifest()
    existing_labels = {t.get("cluster_safe", "") for t in manifest.get("tools", [])}

    clusters = _scan_vocab_clusters(limit=max(40, max_tools * 10))
    summary["candidates"] = len(clusters)
    if not clusters:
        log.info("tool_forge: no clusters eligible for synthesis")
        return summary

    # Resonance gate: score every candidate on the 52-Hz band and forge only the
    # concepts above the lonely-whale line, highest-resonance first. This is the
    # mutual-understanding filter — a concept must be answered by the shared
    # graph AND sit on the shared analytical wavelength, not merely be frequent.
    reference = _shared_reference()
    scored: list[tuple[float, dict]] = []
    cn = open_conn()
    try:
        for cluster in clusters:
            hz = _cluster_resonance_hz(cn, cluster, reference)
            cluster["resonance_hz"] = round(hz, 2)
            scored.append((hz, cluster))
    finally:
        cn.close()
    scored.sort(key=lambda t: t[0], reverse=True)

    forged = 0
    for resonance_hz, cluster in scored:
        if forged >= max_tools:
            break
        eid = cluster["entity_id"]
        if eid in seen:
            summary["skipped"] += 1
            continue

        if resonance_hz < RESONANCE_FLOOR_HZ:
            summary["skipped"] += 1
            summary["skipped_low_resonance"] += 1
            continue    # below the 52 Hz line - a lonely caller, not forged

        novelty, certainty = _novelty_certainty_scores(cluster, existing_labels)
        if not _should_auto_implement(novelty, certainty):
            summary["skipped"] += 1
            continue    # don't mark seen - scores improve as the mesh grows

        try:
            code = _synthesize_internal(cluster)
            ok, err = _validate_code(code)
            if not ok:
                summary["errors"].append(f"{cluster['label']}: {err}")
                continue

            safe = _safe_name(cluster["label"])
            artifact = _classify_artifact(cluster, code)
            fname = f"tool_{safe}.py"
            (_TOOLS_DIR / fname).parent.mkdir(parents=True, exist_ok=True)
            (_TOOLS_DIR / fname).write_text(code, encoding="utf-8")

            manifest.setdefault("tools", []).append({
                "tool_name": f"tool_{safe}",
                "cluster": cluster["label"],
                "cluster_safe": safe,
                "file": fname,
                "type": artifact["type"],
                "novelty": round(novelty, 4),
                "certainty": round(certainty, 4),
                "resonance_hz": round(resonance_hz, 2),
                "degree": cluster.get("degree"),
                "forged_at": ts_start,
            })
            existing_labels.add(safe)
            seen.add(eid)
            forged += 1
            summary["tools_forged"] += 1
            log.info("tool_forge: forged tool_%s (%.1fHz novelty=%.2f certainty=%.2f)",
                     safe, resonance_hz, novelty, certainty)
        except Exception as exc:
            summary["errors"].append(f"{cluster['label']}: {exc}")

    _save_manifest(manifest)
    _save_seen(seen)
    brain_kv.kv_set_json(_FORGE_LOG_KEY, summary)
    return summary


def clean_generated_tools() -> dict:
    """Delete all forged tool files + manifest and clear the forge seen-set.

    Use after changing the synthesis gate (e.g. the resonance floor) to flush
    tools forged under the old, looser policy so the mesh re-forges only
    on-wavelength concepts. Does not touch learned mesh state.
    """
    removed = 0
    if _TOOLS_DIR.exists():
        for p in _TOOLS_DIR.glob("tool_*.py"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    if _MANIFEST.exists():
        try:
            _MANIFEST.unlink()
        except OSError:
            pass
    try:
        brain_kv.kv_set_json(_SEEN_KEY, [])
    except Exception:
        pass
    return {"status": "cleaned", "removed": removed}


__all__ = [
    "forge_round", "load_generated_tools", "forged_tool_names",
    "clean_generated_tools", "RESONANCE_FLOOR_HZ",
    "AUTO_IMPL_THRESHOLD", "NOVELTY_THRESHOLD", "CERTAINTY_THRESHOLD",
]
