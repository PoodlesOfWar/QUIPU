"""export_entirety_gui_data — dump live System Entirety data for the HTML GUI.

Writes ``entirety_gui_data.json`` next to ``entirety_gui.html``.  The GUI loads
that file if present (via fetch) and otherwise falls back to its embedded sample
data, so this exporter is optional but gives the drill-down real numbers.

Run from the repo root::

    python gui/export_entirety_gui_data.py
    python gui/export_entirety_gui_data.py --windows 1,3,6,12,24,72 --flips 200

The JSON schema (consumed by entirety_gui.html):

    {
      "generated_at": iso,
      "current": <system_entirety_state() dict>,
      "senses": <temporal_spatiality._sense_signals() dict>,
      "bit_state": int, "expansion_phase": str,
      "ingest_heartbeat": <synapse_ingest.heartbeat() dict>,
      "windows": [ {"hours": h, "learning": <_recent_learning_state>}, ... ],
      "flip_history": [ <get_flip_history() rows, oldest→newest> ],
      "ingest_history": [ {ts, total_condensed, per_source, weyl_cycles}, ... ]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _safe(fn, default):
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - exporter is best-effort
        print(f"  ! {getattr(fn, '__name__', fn)} failed: {exc}", file=sys.stderr)
        return default


def build_payload(window_hours: list[float], flip_limit: int) -> dict:
    from src.quipu import system_entirety as se
    from src.quipu.temporal_spatiality import _sense_signals

    senses = _safe(_sense_signals, {})
    current = _safe(lambda: se.system_entirety_state(), {})

    # Learnings sliced across several windows so the GUI's time slicer has real
    # data to interpolate between.
    windows: list[dict] = []
    try:
        from src.quipu.system_entirety import _recent_learning_state  # type: ignore
        with se._conn() as cn:  # type: ignore[attr-defined]
            for h in window_hours:
                learning = _safe(lambda h=h: _recent_learning_state(cn, window_hours=h, limit=25), {})
                windows.append({"hours": h, "learning": learning})
    except Exception as exc:
        print(f"  ! learning windows failed: {exc}", file=sys.stderr)

    flips = _safe(lambda: se.get_flip_history(limit=flip_limit), [])
    flips = list(reversed(flips))  # oldest → newest for timeline scrubbing

    ingest_hist = []
    try:
        from src.quipu import corpus_ingest as ci
        ingest_hist = _safe(ci.ingest_history, []) or []
    except Exception:
        pass

    ingest_hb = {}
    try:
        from src.quipu import synapse_ingest
        ingest_hb = _safe(synapse_ingest.heartbeat, {}) or {}
    except Exception:
        pass

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "senses": senses,
        "current": current,
        "bit_state": _safe(se.get_bit_state, 1),
        "expansion_phase": _safe(se.get_expansion_phase, "broaden"),
        "ingest_heartbeat": ingest_hb,
        "windows": windows,
        "flip_history": flips,
        "ingest_history": ingest_hist[-200:],
    }


def _build_standalone_html(payload: dict, template: Path, out_html: Path) -> bool:
    """Bake *payload* into a copy of the template so it works on double-click.

    Injects ``window.__ENTIRETY_DATA__`` just before the template's main
    ``<script>`` block; the GUI prefers that embedded object over fetch/sample.
    """
    if not template.exists():
        print(f"  ! template not found: {template} (skipping standalone HTML)", file=sys.stderr)
        return False
    html = template.read_text(encoding="utf-8")
    # Escape "</" so an embedded string like "</script>" can't close the block;
    # "<\/" is still valid JSON/JS and parses back to "</".
    blob = json.dumps(payload, default=str).replace("</", "<\\/")
    inject = (
        "<script>window.__ENTIRETY_DATA__ = "
        + blob
        + ";</script>\n<script>"
    )
    marker = "<script>\n/* ═"  # start of the main script block
    if marker in html:
        new_html = html.replace(marker, inject + "\n/* ═", 1)
    else:
        # Fallback: inject before the first <script> that isn't already ours.
        idx = html.find("<script>")
        new_html = html[:idx] + inject[:-8] + html[idx:] if idx != -1 else html + inject
    out_html.write_text(new_html, encoding="utf-8")
    return True


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="Export live System Entirety data for the HTML GUI.")
    ap.add_argument("--windows", default="1,3,6,12,24,72",
                    help="comma list of learning windows in hours")
    ap.add_argument("--flips", type=int, default=200, help="max flip-history rows")
    ap.add_argument("--out", default=str(here / "entirety_gui_data.json"),
                    help="output JSON path")
    ap.add_argument("--template", default=str(here / "entirety_gui.html"),
                    help="GUI template to bake data into")
    ap.add_argument("--html-out", default=str(here / "entirety_gui_live.html"),
                    help="standalone HTML output (double-click ready)")
    ap.add_argument("--no-html", action="store_true", help="only write JSON, skip standalone HTML")
    args = ap.parse_args()

    window_hours = [float(x) for x in args.windows.split(",") if x.strip()]
    payload = build_payload(window_hours, args.flips)

    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    n_flips = len(payload.get("flip_history") or [])
    n_learn = 0
    for w in payload.get("windows") or []:
        n_learn = max(n_learn, int((w.get("learning") or {}).get("entry_count") or 0))
    print(f"Wrote {out}  ({n_flips} flip snapshots, {len(window_hours)} windows, "
          f"max {n_learn} learnings/window)")

    if not args.no_html:
        html_out = Path(args.html_out)
        if _build_standalone_html(payload, Path(args.template), html_out):
            print(f"Wrote {html_out}  — open this file directly for live data")


if __name__ == "__main__":
    main()
