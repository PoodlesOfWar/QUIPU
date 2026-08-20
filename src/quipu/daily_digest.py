"""daily_digest — "what has the System Entirety learned today?"

A read-only report over QUIPU's own mesh database (never Supply-Chain-Architect's
— see the module-separation note in local_store.py: each repo computes its
`local_brain.sqlite` path relative to its own file location, so this always
reads QUIPU's corpus even if both repos happen to be checked out side by side).

Every timestamp this codebase writes — `mesh_slm_vocab.first_seen/last_seen`
(SQLite `CURRENT_TIMESTAMP`, UTC), `mesh_slm_meta.last_trained_at`, the
`corpus_ingest`/`systemic_refinement_agent` history entries, and the
`tool_forge` manifest's `forged_at` — is UTC. "Today" here means the current
UTC calendar day, not local time; the report header says so explicitly so a
run late at night in a UTC-behind timezone doesn't look like it's missing
activity that's actually still "today" in UTC.

What counts as "learned today"
-------------------------------
* new concepts   — vocab tokens whose first_seen falls today (never seen before)
* reinforced     — tokens seen before, touched again today (last_seen today,
                    first_seen earlier)
* Weyl cycles    — corpus_ingest holographic-compression cycles run today
                    (from the bounded ``corpus_ingest:history`` in brain_kv)
* Ring 5 cycles  — systemic_refinement_agent passes run today
* tools forged   — tool_forge manifest entries with forged_at == today
* ACRE           — current emergent specialists, and which ones are new
                    relative to the last known snapshot from before today

Public API
----------
    build_digest(date: str | None = None) -> dict
    format_digest(digest: dict) -> str

CLI
---
    python -m src.quipu.daily_digest
    python -m src.quipu.daily_digest --date 2026-07-19
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .local_store import db_path, open_conn
from . import brain_kv


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _is_on(ts: str | None, day: str) -> bool:
    """True if ISO-ish timestamp *ts* (space- or 'T'-separated, UTC) falls on *day*."""
    return bool(ts) and ts[:10] == day


def _vocab_activity(cn: sqlite3.Connection, day: str) -> dict:
    new_rows = cn.execute(
        "SELECT token, freq, first_seen FROM mesh_slm_vocab "
        "WHERE substr(first_seen, 1, 10) = ? ORDER BY freq DESC LIMIT 50",
        (day,),
    ).fetchall()
    reinforced_rows = cn.execute(
        "SELECT token, freq, last_seen FROM mesh_slm_vocab "
        "WHERE substr(last_seen, 1, 10) = ? AND substr(first_seen, 1, 10) != ? "
        "ORDER BY freq DESC LIMIT 50",
        (day, day),
    ).fetchall()
    totals = cn.execute(
        "SELECT COUNT(*) AS vocab, "
        "(SELECT COUNT(*) FROM mesh_slm_quipu) AS edges FROM mesh_slm_vocab"
    ).fetchone()
    return {
        "new_tokens": [{"token": r["token"], "freq": r["freq"]} for r in new_rows],
        "new_count": len(new_rows),
        "reinforced_tokens": [{"token": r["token"], "freq": r["freq"]} for r in reinforced_rows],
        "reinforced_count": len(reinforced_rows),
        "vocab_size_now": int(totals["vocab"]),
        "quipu_edges_now": int(totals["edges"]),
    }


def _training_activity(cn: sqlite3.Connection, day: str) -> dict:
    def _meta(key: str) -> Any:
        row = cn.execute("SELECT value FROM mesh_slm_meta WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            return None

    last_trained = _meta("last_trained_at")
    return {
        "rounds_total": _meta("rounds") or 0,
        "last_trained_at": last_trained,
        "trained_today": _is_on(last_trained, day),
    }


def _live_acre_specialists(cn: sqlite3.Connection) -> list[str]:
    row = cn.execute("SELECT value FROM mesh_slm_meta WHERE key='acre_specialists'").fetchone()
    try:
        return sorted(json.loads(row["value"]).keys()) if row else []
    except (TypeError, json.JSONDecodeError):
        return []


def _acre_activity(cn: sqlite3.Connection, day: str, refinement_history: list[dict]) -> dict:
    """Specialist list "as of *day*", and which ones are new relative to the
    last known snapshot from strictly before *day*.

    Deliberately snapshot-based (not the live ``mesh_slm_meta`` state) so a
    query for a past day reflects what existed *then*, not what exists *now* -
    ``mesh_slm_meta`` has no timestamp on the specialists dict itself, only the
    per-cycle history entries carry a point-in-time snapshot. Falls back to the
    live state only when there is no history at all yet (e.g. immediately
    after the very first ingest, before any Ring 5 cycle has been recorded).
    """
    asof: list[str] | None = None
    baseline: list[str] = []
    for entry in reversed(refinement_history):
        ts = entry.get("ts", "")
        if asof is None and ts[:10] <= day:
            asof = entry.get("acre_specialists") or []
        if ts[:10] < day:
            baseline = entry.get("acre_specialists") or []
            break

    if not refinement_history:
        # No historical evidence exists anywhere yet - live state is the best
        # available answer (there is nothing else to fall back to).
        asof = _live_acre_specialists(cn)
    elif asof is None:
        # History exists, but every entry is dated AFTER *day* - we genuinely
        # have no evidence of what existed as of that day.
        asof = []

    new_today = sorted(set(asof) - set(baseline))
    return {"current": sorted(asof), "new_today": new_today}


def _ingest_activity(day: str) -> dict:
    history = brain_kv.kv_get_json("corpus_ingest:history", [])
    if not isinstance(history, list):
        history = []
    todays = [e for e in history if _is_on(e.get("ts"), day)]
    docs_total = sum(e.get("total_condensed", 0) for e in todays)
    per_source: dict[str, int] = {}
    for e in todays:
        for src, cnt in (e.get("per_source") or {}).items():
            per_source[src] = per_source.get(src, 0) + cnt
    latest = todays[-1] if todays else {}
    return {
        "cycles_today": len(todays),
        "documents_today": docs_total,
        "per_source_today": per_source,
        "latest_psi5": latest.get("weyl_latest_psi5"),
        "latest_compaction_ratio": latest.get("compaction_ratio"),
        "latest_compaction_scalar": latest.get("compaction_scalar"),
    }


def _refinement_activity(day: str) -> tuple[dict, list[dict]]:
    history = brain_kv.kv_get_json("systemic_refinement:history", [])
    if not isinstance(history, list):
        history = []
    todays = [e for e in history if _is_on(e.get("ts"), day)]
    actions: list[str] = []
    for e in todays:
        actions.extend(e.get("actions") or [])
    return {"cycles_today": len(todays), "actions_today": actions}, history


def _forged_tools_today(day: str) -> list[dict]:
    try:
        from . import tool_forge
        manifest = tool_forge._load_manifest()  # noqa: SLF001 - sibling-module, same package
    except Exception:
        return []
    tools = manifest.get("tools", [])
    return [t for t in tools if _is_on(t.get("forged_at"), day)]


def build_digest(date: str | None = None) -> dict:
    """Assemble the full day's activity report. *date* defaults to today (UTC)."""
    day = date or _today_utc()
    db = db_path()
    if not db.exists():
        return {
            "date": day, "db_path": str(db), "db_exists": False,
            "headline": f"No mesh database found yet at {db} - nothing has run.",
        }

    cn = open_conn()
    try:
        vocab = _vocab_activity(cn, day)
        training = _training_activity(cn, day)
        refinement_summary, refinement_history = _refinement_activity(day)
        acre = _acre_activity(cn, day, refinement_history)
    finally:
        cn.close()

    ingest = _ingest_activity(day)
    tools_today = _forged_tools_today(day)
    the_other = brain_kv.kv_get_json("entirety:the_other", {}) or {}
    try:
        from . import mesh_slm
        base_specs = mesh_slm.base_specialists()
        active_spec = mesh_slm.select_resonant_specialist()
        consensus = mesh_slm.resonant_specialists(k=3)
    except Exception:
        base_specs, active_spec, consensus = [], None, []

    nothing_happened = (
        vocab["new_count"] == 0 and vocab["reinforced_count"] == 0
        and ingest["cycles_today"] == 0 and refinement_summary["cycles_today"] == 0
        and not tools_today and not training["trained_today"]
    )

    headline = (
        f"No learning activity recorded for {day} (UTC) yet."
        if nothing_happened else
        f"{vocab['new_count']} new concepts, {vocab['reinforced_count']} reinforced, "
        f"{ingest['cycles_today']} Weyl cycle(s) over {ingest['documents_today']} documents, "
        f"{len(tools_today)} tool(s) forged, {len(acre['new_today'])} new ACRE specialist(s)."
    )

    return {
        "date": day,
        "db_path": str(db),
        "db_exists": True,
        "nothing_happened": nothing_happened,
        "headline": headline,
        "vocab": vocab,
        "training": training,
        "ingest": ingest,
        "refinement": refinement_summary,
        "acre": acre,
        "base_specialists": base_specs,
        "active_specialist": active_spec,
        "consensus": consensus,
        "tools_forged_today": tools_today,
        "the_other": the_other,
    }


def format_digest(d: dict) -> str:
    """Render build_digest()'s output as a readable terminal report."""
    lines: list[str] = []
    lines.append(f"System Entirety - daily digest for {d['date']} (UTC)")
    lines.append("=" * 60)

    if not d.get("db_exists"):
        lines.append("")
        lines.append(d["headline"])
        lines.append(f"(expected database at: {d['db_path']})")
        return "\n".join(lines)

    lines.append("")
    lines.append(d["headline"])
    lines.append("")

    v = d["vocab"]
    lines.append(f"Corpus (vocab now: {v['vocab_size_now']}, quipu edges now: {v['quipu_edges_now']})")
    if v["new_count"]:
        top_new = ", ".join(f"{t['token']}({t['freq']})" for t in v["new_tokens"][:10])
        more = f" (+{v['new_count'] - 10} more)" if v["new_count"] > 10 else ""
        lines.append(f"  new concepts today   : {v['new_count']} - {top_new}{more}")
    else:
        lines.append("  new concepts today   : 0")
    if v["reinforced_count"]:
        top_re = ", ".join(f"{t['token']}({t['freq']})" for t in v["reinforced_tokens"][:10])
        more = f" (+{v['reinforced_count'] - 10} more)" if v["reinforced_count"] > 10 else ""
        lines.append(f"  reinforced today     : {v['reinforced_count']} - {top_re}{more}")
    else:
        lines.append("  reinforced today     : 0")

    t = d["training"]
    trained = f"yes (last at {t['last_trained_at']})" if t["trained_today"] else "no"
    lines.append(f"  trained today        : {trained}  (lifetime rounds: {t['rounds_total']})")

    lines.append("")
    i = d["ingest"]
    lines.append(f"The Well - holographic compression (corpus_ingest)")
    lines.append(f"  Weyl cycles today    : {i['cycles_today']}")
    lines.append(f"  documents condensed  : {i['documents_today']}")
    if i["per_source_today"]:
        src_str = ", ".join(f"{k}={v}" for k, v in i["per_source_today"].items())
        lines.append(f"  by source            : {src_str}")
    if i["latest_psi5"]:
        lines.append(f"  latest Weyl tensor   : {i['latest_psi5']}")
        lines.append(f"  compaction ratio/sc  : {i['latest_compaction_ratio']} / {i['latest_compaction_scalar']}")

    lines.append("")
    r = d["refinement"]
    lines.append(f"Ring 5 - refinement / toolforge")
    lines.append(f"  refinement cycles    : {r['cycles_today']}")
    if r["actions_today"]:
        lines.append(f"  actions              : {', '.join(r['actions_today'])}")
    tf = d["tools_forged_today"]
    if tf:
        for entry in tf:
            lines.append(
                f"  forged tool          : {entry.get('tool_name')} "
                f"(novelty={entry.get('novelty')}, certainty={entry.get('certainty')})"
            )
    else:
        lines.append("  tools forged today   : 0")

    a = d["acre"]
    lines.append("")
    lines.append(f"Specialists")
    lines.append(f"  base (always present): {d.get('base_specialists') or '(none)'}")
    lines.append(f"  emergent             : {a['current'] or '(none yet)'}")
    lines.append(f"  new emergent today   : {a['new_today'] or '(none)'}")
    cons = d.get("consensus") or []
    if cons:
        cons_str = ", ".join(f"{n} {w:.0%}" for n, w in cons)
        lines.append(f"  active consensus     : {cons_str}")
    else:
        lines.append(f"  active (resonant)    : {d.get('active_specialist') or '(none)'}")

    o = d.get("the_other") or {}
    if o:
        lines.append("")
        lines.append("Shared understanding of the Other")
        hz = o.get("shared_hz")
        if hz is not None:
            line52 = "above 52 Hz line" if hz >= 52 else "below 52 Hz (the lonely-whale line)"
            lines.append(f"  shared wavelength    : {hz} Hz  ({line52})")
        else:
            lines.append(f"  resonance            : {o.get('resonance')}")
        if o.get("other_axis_strength") is not None:
            lines.append(f"  entirety/Other axis  : {o.get('other_axis_strength')}  (fiction/experiential mass)")
        if o.get("acre_axis"):
            lines.append(f"  interaction axis     : {o.get('acre_axis')}")

    return "\n".join(lines)


def _cli() -> None:
    ap = argparse.ArgumentParser(description='"What has the System Entirety learned today?"')
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (UTC); defaults to today")
    ap.add_argument("--json", action="store_true", help="print raw JSON instead of the formatted report")
    args = ap.parse_args()

    digest = build_digest(args.date)
    if args.json:
        print(json.dumps(digest, indent=2, default=str))
    else:
        print(format_digest(digest))


if __name__ == "__main__":
    _cli()
