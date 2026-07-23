"""Self-Realization Closed-Loop — gap detector + actuator router.

On every ~5-minute cycle:

1. **DETECT** — scan synaptic worker heartbeats, brain_kv signals, and corpus
   density to find gaps the Brain cannot close through normal operations.

2. **ROUTE**  — map each gap to the best actuator:

   +-----------------+---------------------------------------------+
   | Gap kind        | Primary actuator                            |
   +=================+=============================================+
   | worker_gap      | Log; mark for HITL if worker is zombie      |
   | capability_gap  | tool_forge via forge_for_gap()              |
   | corpus_gap      | doc_rag_reindex                             |
   | rag_gap         | doc_rag_reindex                             |
   | resilience_gap  | body_directive + ACRE if patch template     |
   | external_gap    | body_directive (human action needed)        |
   +-----------------+---------------------------------------------+

3. **ACT**    — invoke the actuator with bounded parameters (no more than
   ``_MAX_ACTUATIONS_PER_ROUND`` side-effects per round).

4. **LOG**    — write to brain_kv ``self_realization:last_round`` and append
   a ``self_realization`` entry to the learning_log.

Design constraints
──────────────────
- Effect-bounded  — max ``_MAX_ACTUATIONS_PER_ROUND`` actuations per round.
- Auditable       — every action written to learning_log.
- No circular dep — imports tool_forge / code_restructure lazily (local).
- Rate-limited    — ``_MIN_ROUND_INTERVAL_S`` hard floor between rounds.

Public API
──────────
    detect_self_realization_gaps() -> list[dict]
    route_gap(gap: dict) -> dict
    self_realization_round() -> dict
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
_MAX_ACTUATIONS_PER_ROUND: int   = 3
_MIN_ROUND_INTERVAL_S:     float = 240.0   # 4-minute hard floor
_WORKER_STALE_MULTIPLIER:  int   = 4       # age > N × interval → stale
_LAST_ROUND_TS: float = 0.0

# ── Paths ─────────────────────────────────────────────────────────────────────
_BRAIN_DIR    = Path(__file__).parent
_PIPELINE_DIR = _BRAIN_DIR.parent.parent


def _db_path() -> Path:
    try:
        from .local_store import db_path  # type: ignore[import]
        return Path(db_path())
    except Exception:
        return _PIPELINE_DIR / "local_brain.sqlite"


def _conn() -> sqlite3.Connection:
    try:
        from .local_store import open_conn
        return open_conn(timeout=30)
    except Exception:
        cn = sqlite3.connect(str(_db_path()), timeout=30)
        cn.row_factory = sqlite3.Row
        cn.execute("PRAGMA journal_mode=WAL")
        cn.execute("PRAGMA busy_timeout=30000")
        return cn


def _kv_read(key: str, default: str = "") -> str:
    try:
        from .brain_kv import kv_get  # type: ignore[import]
        v = kv_get(key)
        return v if v is not None else default
    except Exception:
        return default


def _kv_write(key: str, value: str) -> bool:
    try:
        from .brain_kv import kv_set  # type: ignore[import]
        return bool(kv_set(key, value))
    except Exception as exc:
        log.warning("[self_realization] KV write failed for %s: %s", key, exc)
        return False


def request_worker_recovery(reason: str) -> bool:
    """Persist an explicit request for the next bounded recovery round."""
    payload = {
        "reason": str(reason)[:240],
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    return _kv_write("self_realization:worker_recovery_request", json.dumps(payload))


def _log_to_learning(kind: str, payload: dict) -> None:
    """Append a self_realization entry to learning_log (best-effort)."""
    try:
        cn = _conn()
        now = datetime.now(timezone.utc).isoformat()
        columns = {row[1] for row in cn.execute("PRAGMA table_info(learning_log)").fetchall()}
        detail = json.dumps(payload, default=str)
        if {"logged_at", "title", "detail"}.issubset(columns):
            cn.execute(
                "INSERT INTO learning_log(logged_at, kind, title, detail, signal_strength) "
                "VALUES (?, ?, ?, ?, ?)",
                (now, kind, "Self-realization recovery round", detail, 0.75),
            )
        else:
            cn.execute(
                "INSERT OR IGNORE INTO learning_log"
                "(kind, content, source_url, source_row_id, created_at)"
                "VALUES (?, ?, '', NULL, ?)",
                (kind, detail, now),
            )
        cn.commit()
        cn.close()
    except Exception as exc:
        log.warning("[self_realization] learning-log write failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Gap detection
# ─────────────────────────────────────────────────────────────────────────────

def _heartbeat_timestamp(raw: str) -> str:
    """Read a timestamp from legacy pipe text or structured health JSON."""
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return str(payload.get("ran_at") or payload.get("updated_at") or "")
    except Exception:
        pass
    return str(raw).split("|", 1)[0]

def _detect_worker_gaps() -> list[dict]:
    """Return gaps for synaptic workers that have never run or are stale."""
    gaps: list[dict] = []
    worker_intervals = {
        "synapse_builder_last":          600,
        "synapse_lookahead_7d_last":     900,
        "synapse_convergence_last":     1800,
        "synapse_vision_last":           300,
        "synapse_torus_last":             30,
        "synapse_ueqgm_last":            420,
        "synapse_doc_anneal_last":      1800,
        "synapse_tool_forge_last":     14400,
        "synapse_acre_last":           21600,
        "synapse_specialists_last":      900,
        "synapse_self_realization_last":  300,
        "synapse_ingest_last":          1200,
        "llm_self_train:last_round":    1800,
    }
    critical = {
        "synapse_specialists_last",
        "synapse_self_realization_last",
        "synapse_ingest_last",
        "llm_self_train:last_round",
    }
    now = datetime.now(timezone.utc)
    for kv_key, interval_s in worker_intervals.items():
        raw = _kv_read(kv_key, "")
        ts_iso = _heartbeat_timestamp(raw)
        if not ts_iso:
            gaps.append({
                "kind":    "worker_gap",
                "worker":  kv_key,
                "reason":  "never_ran",
                "age_s":   None,
                "severity": "medium" if kv_key in critical else "low",
            })
            continue
        try:
            last_ts = datetime.fromisoformat(ts_iso)
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            age_s = (now - last_ts).total_seconds()
            if age_s > _WORKER_STALE_MULTIPLIER * interval_s:
                gaps.append({
                    "kind":    "worker_gap",
                    "worker":  kv_key,
                    "reason":  "stale",
                    "age_s":   int(age_s),
                    "threshold_s": _WORKER_STALE_MULTIPLIER * interval_s,
                    "severity": "medium",
                })
        except Exception:
            pass
    return gaps


def _detect_capability_gaps() -> list[dict]:
    """Return gaps for corpus clusters that have no operationalised tool."""
    gaps: list[dict] = []
    try:
        cn = _conn()
        rows = cn.execute(
            """
            SELECT e.entity_id, e.label, e.entity_type,
                   CAST(json_extract(e.props_json, '$.certainty') AS REAL) AS certainty,
                   CAST(json_extract(e.props_json, '$.novelty')   AS REAL) AS novelty
            FROM corpus_entity e
            WHERE e.entity_type IN ('Capability','ToolCapability','KnowledgeCluster')
              AND (certainty IS NULL OR certainty >= 0.50)
              AND e.entity_id NOT IN (
                  SELECT json_extract(props_json,'$.source_entity_id')
                  FROM corpus_entity
                  WHERE entity_type = 'GeneratedTool'
              )
            ORDER BY certainty DESC, novelty DESC
            LIMIT 5
            """,
        ).fetchall()
        cn.close()
        for row in rows:
            gaps.append({
                "kind":       "capability_gap",
                "entity_id":  row["entity_id"],
                "label":      row["label"],
                "entity_type": row["entity_type"],
                "certainty":  row["certainty"],
                "novelty":    row["novelty"],
                "severity":   "medium",
            })
    except Exception:
        pass
    return gaps


def _detect_corpus_gaps() -> list[dict]:
    """Return gaps when corpus entity density is below minimum threshold."""
    gaps: list[dict] = []
    try:
        cn = _conn()
        count = cn.execute(
            "SELECT COUNT(*) FROM corpus_entity"
            " WHERE last_seen >= datetime('now','-7 days')",
        ).fetchone()[0]
        cn.close()
        if count < 10:
            gaps.append({
                "kind":     "corpus_gap",
                "reason":   "low_entity_density",
                "count_7d": count,
                "severity": "high" if count < 3 else "medium",
            })
    except Exception:
        pass
    return gaps


def _detect_rag_gaps() -> list[dict]:
    """Return gaps when RAG index is stale (no reindex in last 48 h)."""
    gaps: list[dict] = []
    raw = _kv_read("doc_rag:last_index_ts", "")
    if not raw:
        gaps.append({
            "kind":     "rag_gap",
            "reason":   "never_indexed",
            "severity": "medium",
        })
        return gaps
    try:
        last_ts = datetime.fromisoformat(raw.split("|", 1)[0])
        age_h   = (datetime.now(timezone.utc) - last_ts.replace(
            tzinfo=timezone.utc if last_ts.tzinfo is None else last_ts.tzinfo,
        )).total_seconds() / 3600
        if age_h > 48:
            gaps.append({
                "kind":     "rag_gap",
                "reason":   "stale_index",
                "age_h":    round(age_h, 1),
                "severity": "low",
            })
    except Exception:
        pass
    return gaps


def _detect_resilience_gaps() -> list[dict]:
    """Return gaps from unresolved resilience_event entries in learning_log."""
    gaps: list[dict] = []
    try:
        cn = _conn()
        columns = {row[1] for row in cn.execute("PRAGMA table_info(learning_log)").fetchall()}
        if {"detail", "logged_at"}.issubset(columns):
            rows = cn.execute(
                "SELECT detail AS payload FROM learning_log "
                "WHERE kind='resilience_event' "
                "AND logged_at >= datetime('now','-24 hours') LIMIT 10"
            ).fetchall()
        else:
            rows = cn.execute(
                "SELECT content AS payload FROM learning_log "
                "WHERE kind='resilience_event' "
                "AND created_at >= datetime('now','-24 hours') LIMIT 10"
            ).fetchall()
        cn.close()
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except Exception:
                payload = {"raw": row["payload"]}
            gaps.append({
                "kind":     "resilience_gap",
                "payload":  payload,
                "severity": "medium",
            })
    except Exception:
        pass
    return gaps


def detect_self_realization_gaps() -> list[dict]:
    """Return all detected self-realization gaps, sorted by severity."""
    _SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
    gaps: list[dict] = []
    gaps.extend(_detect_worker_gaps())
    gaps.extend(_detect_capability_gaps())
    gaps.extend(_detect_corpus_gaps())
    gaps.extend(_detect_rag_gaps())
    gaps.extend(_detect_resilience_gaps())
    gaps.sort(key=lambda g: _SEVERITY_ORDER.get(g.get("severity", "low"), 2))
    return gaps


# ─────────────────────────────────────────────────────────────────────────────
# Gap routing
# ─────────────────────────────────────────────────────────────────────────────

def route_gap(gap: dict) -> dict:
    """Map *gap* to an actuator and execute it (bounded).

    Returns a routing record::

        {
            "gap_kind":  str,
            "actuator":  str,
            "status":    "ok" | "skipped" | "error",
            "detail":    str,
        }
    """
    kind = gap.get("kind", "")
    record: dict[str, Any] = {
        "gap_kind": kind,
        "actuator": "none",
        "status":   "skipped",
        "detail":   "",
    }

    try:
        if kind == "capability_gap":
            record["actuator"] = "tool_forge"
            from src.quipu.tool_forge import forge_for_gap  # type: ignore[import]
            result = forge_for_gap(gap, max_tools=1)
            record["status"] = "ok"
            record["detail"] = json.dumps(result, default=str)

        elif kind in ("corpus_gap", "rag_gap"):
            record["actuator"] = "doc_rag_reindex"
            from src.quipu.doc_rag import index_documents  # type: ignore[import]
            index_documents(fresh=False)
            record["status"] = "ok"
            record["detail"] = f"incremental reindex triggered for {kind}"

        elif kind == "resilience_gap":
            record["actuator"] = "body_directive"
            from src.quipu.brain_body_signals import queue_directive  # type: ignore[import]
            payload = gap.get("payload", {})
            queue_directive(
                kind      = "resilience_alert",
                message   = f"Unresolved resilience event: {str(payload)[:200]}",
                priority  = 2,
                source    = "self_realization_loop",
            )
            record["status"] = "ok"
            record["detail"] = "body directive queued"

        elif kind == "worker_gap":
            worker = str(gap.get("worker") or "")
            if worker == "llm_self_train:last_round":
                record["actuator"] = "self_train_rearm"
                from src.quipu.llm_self_train import ensure_scheduled  # type: ignore[import]
                thread = ensure_scheduled(request_now=True)
                record["status"] = "ok"
                record["detail"] = f"self-train scheduler armed (alive={thread.is_alive()})"
            elif worker == "synapse_self_realization_last":
                record["actuator"] = "synaptic_cohort_health_check"
                from src.quipu.synaptic_workers import ensure_continuous_synaptic_agents  # type: ignore[import]
                status = ensure_continuous_synaptic_agents()
                record["status"] = "ok"
                record["detail"] = (
                    f"synaptic cohort health check: alive={status.get('alive', 0)}/"
                    f"{status.get('total', 0)}"
                )
            elif worker == "synapse_ingest_last":
                record["actuator"] = "worker_restart"
                from src.quipu.synapse_ingest import ensure_scheduled as _ingest_arm  # type: ignore[import]
                thread = _ingest_arm(request_now=True)
                record["status"] = "ok"
                record["detail"] = (
                    f"corpus-ingest worker re-armed (alive={thread.is_alive()}); "
                    f"forced immediate round to refill the 1h sense window"
                )
            elif worker == "synapse_tool_forge_last":
                record["actuator"] = "worker_restart"
                from src.quipu.tool_forge import forge_round  # type: ignore[import]
                result = forge_round(max_tools=1)
                record["status"] = "ok"
                record["detail"] = f"tool_forge remediation round triggered: {json.dumps(result, default=str)[:220]}"
            elif worker == "synapse_acre_last":
                record["actuator"] = "worker_restart"
                try:
                    from src.quipu.self_modification import autonomous_self_modify_round  # type: ignore[import]
                    result = autonomous_self_modify_round(force=True)
                    engine = "self_modification"
                except Exception:
                    from src.quipu.code_restructure import restructure_round  # type: ignore[import]
                    result = restructure_round(force=True)
                    engine = "acre"
                record["status"] = "ok"
                record["detail"] = f"{engine} remediation round triggered: {json.dumps(result, default=str)[:220]}"
            elif worker == "synapse_specialists_last":
                record["actuator"] = "worker_restart"
                from src.quipu.expert_orchestrator import run_expert_orchestration  # type: ignore[import]
                result = run_expert_orchestration(
                    "continue specialist orchestration through Quipu GNN",
                    context={
                        "source": "self_realization_worker_gap",
                        "orchestration_mode": "quipu_gnn_multidimensional",
                        "last_experts": ["supply_chain_optimizer", "research_specialist", "operations_resuscitator"],
                    },
                )
                record["status"] = "ok"
                record["detail"] = (
                    f"specialist remediation round triggered: "
                    f"experts={json.dumps(result.get('experts_used', []) if isinstance(result, dict) else [], default=str)}"
                )
            elif gap.get("severity") == "high":
                record["actuator"] = "hitl_queue"
                _kv_write(
                    "self_realization:hitl",
                    json.dumps({"gap": gap, "ts": datetime.now(timezone.utc).isoformat()},
                               default=str),
                )
                record["status"] = "ok"
                record["detail"] = f"worker {gap.get('worker')} queued for HITL review"
            else:
                record["status"] = "skipped"
                record["detail"] = f"worker gap severity={gap.get('severity')} — no action"

        else:
            record["status"] = "skipped"
            record["detail"] = f"no actuator registered for gap kind '{kind}'"

    except Exception as exc:
        record["status"] = "error"
        record["detail"] = str(exc)
        log.warning("[self_realization] route_gap error for %s: %s", kind, exc)

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main round entry point
# ─────────────────────────────────────────────────────────────────────────────

def self_realization_round() -> dict:
    """Run one full self-realization cycle.

    Returns::

        {
            "gaps_detected": int,
            "gaps_routed":   int,
            "gaps_closed":   int,
            "routes":        list[dict],
            "ts":            str,
        }
    """
    global _LAST_ROUND_TS

    now_ts = time.time()
    if now_ts - _LAST_ROUND_TS < _MIN_ROUND_INTERVAL_S:
        log.debug("[self_realization] throttled — last round was recent")
        return {
            "gaps_detected": 0,
            "gaps_routed":   0,
            "gaps_closed":   0,
            "routes":        [],
            "ts":            datetime.now(timezone.utc).isoformat(),
        }

    _LAST_ROUND_TS = now_ts
    ts             = datetime.now(timezone.utc).isoformat()

    gaps      = detect_self_realization_gaps()
    routes:   list[dict] = []
    actuated  = 0

    for gap in gaps:
        if actuated >= _MAX_ACTUATIONS_PER_ROUND:
            break
        # Skip worker gaps with severity=low — don't burn actuation budget
        if gap.get("kind") == "worker_gap" and gap.get("severity") == "low":
            continue
        route = route_gap(gap)
        routes.append(route)
        if route["status"] == "ok":
            actuated += 1

    gaps_closed = sum(1 for r in routes if r["status"] == "ok")

    summary = {
        "gaps_detected": len(gaps),
        "gaps_routed":   len(routes),
        "gaps_closed":   gaps_closed,
        "routes":        routes,
        "ts":            ts,
    }

    _kv_write("self_realization:last_round", json.dumps(summary, default=str))
    _log_to_learning("self_realization", summary)
    log.info(
        "[self_realization] detected=%d routed=%d closed=%d",
        len(gaps), len(routes), gaps_closed,
    )
    return summary
