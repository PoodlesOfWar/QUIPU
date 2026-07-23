"""doc_annealing — regenerate the QUIPU Entirety living system map.

This is QUIPU's clean-room reimplementation of the Supply-Chain-Brain doc
annealing worker. On each cycle it:

1. reads the live 7+1-D System Entirety state (``system_entirety``), the
   MESH-SLM predictor snapshot (``mesh_slm.state_summary``), and the STP
   geodesic P1 trend (``mesh_slm.stp_diagnostic_trend``);
2. computes a SHA-256 **structural fingerprint** over the version + bridge
   root + mesh density + UEQGM certainty + nodal bifurcation (each rounded to
   2 dp, so only meaningful moves count);
3. if the fingerprint differs from ``brain_kv["doc:system_map_hash"]`` (or when
   ``force``), rewrites ``docs/system_entirety_map.md``, bumps
   ``brain_kv["doc:system_map_version"]``, and appends a bounded entry to
   ``brain_kv["doc_annealing:history"]``.

Unlike the SCB parent, QUIPU ships no RAG index, so no re-index is triggered;
the regenerated map is the deliverable. Everything is read-only against the
mesh except the three ``brain_kv`` bookkeeping keys and the map file itself.

Public API
----------
    anneal_docs(*, force=False, out_path=None, collected=None) -> dict
    structural_change_review(collected, prev_hash=None) -> dict
    render_system_map(collected, map_version, fingerprint) -> str
    run_loop(interval_s=1800, max_cycles=None) -> None

CLI
---
    python -m src.quipu.doc_annealing            # one anneal cycle
    python -m src.quipu.doc_annealing --force    # rewrite even if unchanged
    python -m src.quipu.doc_annealing --loop --interval 1800   # run forever
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import brain_kv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MAP = _REPO_ROOT / "docs" / "system_entirety_map.md"

_HASH_KEY = "doc:system_map_hash"
_VERSION_KEY = "doc:system_map_version"
_HISTORY_KEY = "doc_annealing:history"
_HISTORY_CAP = 200
_DEFAULT_INTERVAL_S = 1800  # 30 min, matching the SCB anneal cadence


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_state() -> dict:
    """Gather everything the map needs. Every source is defensively optional so
    a cold or partially-initialised mesh still anneals a valid (sparser) map."""
    try:
        from . import _version
        version = str(getattr(_version, "__version__", "0.0.0"))
    except Exception:
        version = "0.0.0"

    entirety: dict = {}
    try:
        from .system_entirety import system_entirety_state
        entirety = system_entirety_state() or {}
    except Exception:
        try:
            from .system_entirety import get_entirety_state
            entirety = get_entirety_state() or {}
        except Exception:
            entirety = {}

    slm: dict = {}
    stp: dict = {}
    try:
        from . import mesh_slm
        try:
            slm = mesh_slm.state_summary() or {}
        except Exception:
            slm = {}
        try:
            stp = mesh_slm.stp_diagnostic_trend() or {}
        except Exception:
            stp = {}
    except Exception:
        pass

    return {
        "version": version,
        "annealed_at": _now_iso(),
        "entirety": entirety,
        "slm": slm,
        "stp": stp,
    }


# ---------------------------------------------------------------------------
# Structural fingerprint
# ---------------------------------------------------------------------------
def _fingerprint_fields(collected: dict) -> dict:
    """Extract the coarse structural quantities the fingerprint hashes."""
    e = collected.get("entirety", {}) or {}
    mat = e.get("material_bifurcation", {}) or {}
    mesh = mat.get("mesh", {}) or {}
    bridge = mat.get("bridge_mesh_density", {}) or {}
    ueqgm = e.get("ueqgm_runtime", {}) or {}
    return {
        "version": str(collected.get("version", "0.0.0")),
        "primary_root": str(bridge.get("primary_root") or "none"),
        "total_roots": int(bridge.get("total_roots", 0) or 0),
        "mesh_density": float(mesh.get("mesh_density", 0.0) or 0.0),
        "certainty": float(ueqgm.get("certainty", 0.0) or 0.0),
        "nodal_bifurcation": float(mat.get("nodal_bifurcation", 0.0) or 0.0),
    }


def _fingerprint(collected: dict) -> str:
    """SHA-256 (16 hex chars) over the coarse structural quantities.

    Each float is rounded to 2 dp before hashing, so sub-1% jitter in the live
    state does not churn the map — only structural moves trip a rewrite.
    """
    f = _fingerprint_fields(collected)
    payload = (
        f"{f['version']}|{f['primary_root']}|{f['total_roots']}"
        f"|{f['mesh_density']:.2f}|{f['certainty']:.2f}|{f['nodal_bifurcation']:.2f}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def structural_change_review(collected: dict, prev_hash: str | None = None) -> dict:
    """Compare the current structural fingerprint against the stored one."""
    fp = _fingerprint(collected)
    if prev_hash is None:
        prev_hash = brain_kv.kv_get(_HASH_KEY, None)
    return {
        "fingerprint": fp,
        "prev_hash": prev_hash,
        "changed": fp != prev_hash,
        "fields": _fingerprint_fields(collected),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_system_map(collected: dict, map_version: int, fingerprint: str) -> str:
    """Render the full QUIPU Entirety living system map as markdown."""
    version = collected.get("version", "0.0.0")
    annealed = collected.get("annealed_at", _now_iso())
    e = collected.get("entirety", {}) or {}
    axes = e.get("axes", {}) or {}
    mat = e.get("material_bifurcation", {}) or {}
    mesh = mat.get("mesh", {}) or {}
    bridge = mat.get("bridge_mesh_density", {}) or {}
    ueqgm = e.get("ueqgm_runtime", {}) or {}
    txn = e.get("transaction", {}) or {}
    slm = collected.get("slm", {}) or {}
    stp = collected.get("stp", {}) or {}

    L: list[str] = []
    L.append("# QUIPU Entirety — Living System Map")
    L.append("")
    L.append(f"> **Version**: {version}  ")
    L.append(f"> **Annealed**: {annealed}  ")
    L.append(f"> **Map revision**: v{map_version} · fingerprint `{fingerprint}`  ")
    L.append("> **Generator**: `src/quipu/doc_annealing.py` — QUIPU's own annealer. "
             "Regenerated whenever the structural fingerprint (version + bridge root + "
             "mesh density + UEQGM certainty + nodal bifurcation) changes.")
    L.append("")
    L.append("---")
    L.append("")

    # 1. Architecture overview
    L.append("## 1. Architecture Overview")
    L.append("")
    L.append("The QUIPU Entirety (descended from the Supply-Chain-Brain) is a "
             "**closed-loop attractor system**. The bridge/material/entirety/bit-flip "
             "stack densifies the torus; at the centre the **MESH-SLM predictor** reads "
             "the resulting 7+1-D state and returns a ranked continuation, writing its "
             "Hebbian updates back onto the same torus — the dense-memory read head of "
             "the loop:")
    L.append("")
    L.append("```")
    L.append("System Entirety 7+1-D state  [src/quipu/system_entirety.py]")
    L.append("  └─ axes[6] (vision·touch·smell·body·brain·perception) + observer + mesh_field_8d")
    L.append("       ↓  persisted as entirety:state")
    L.append("MESH-SLM predictor           [src/quipu/mesh_slm.py]")
    L.append("  └─ score(t) = 0.55·quipu + (0.25 + 0.06·warp)·prox")
    L.append("                + ⟨embed7(t), mesh_state_7d⟩ + 0.18·mesh_field_8d")
    L.append("  └─ identity-style predictor (no learned head) — STP paper P5")
    L.append("  └─ Hebbian write-back: weight += η_q·(1 − w)  → same torus")
    L.append("  └─ STP geodesic diagnostic (v0.25.0): 1 − cos(h_t−h_r, h_r−h_s)")
    L.append("       on the 7-D embedding and the ℝ⁴ torus embedding (passive)")
    L.append("       ↑  loop closes: denser torus → richer state → sharper prediction")
    L.append("```")
    L.append("")
    L.append("Full architecture: `docs/SYSTEM_ENTIRETY_ANALYSIS.md`, `docs/SYSTEM_DYNAMICS.md`, "
             "`docs/MESH_SLM_GLM_CLASSIFIER.md`.")
    L.append("")
    L.append("---")
    L.append("")

    # 2. Live System Entirety state
    L.append("## 2. System Entirety — Live State")
    L.append("")
    L.append(f"> Snapshot at annealing time `{annealed}`")
    L.append("")
    L.append("### 2.1 Sense Axes (6-D)")
    L.append("")
    L.append("| Sense | Value |")
    L.append("|---|---|")
    for sense in ("vision", "touch", "smell", "body", "brain", "perception"):
        L.append(f"| {sense} | {_pct(axes.get(sense))} |")
    L.append("")
    L.append(f"**Observer tangent** (7th-D orthogonal excitation): {_pct(e.get('observer'))}  ")
    L.append(f"**7-D magnitude**: {_num(e.get('magnitude'))}")
    L.append("")
    L.append("### 2.2 Material Bifurcation")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Physical realization | {_pct(mat.get('physical_realization'))} |")
    L.append(f"| Nodal bifurcation | {_pct(mat.get('nodal_bifurcation'))} |")
    L.append(f"| Mesh density | {_pct(mesh.get('mesh_density'))} |")
    L.append(f"| Topology | `{mesh.get('topology', 'n/a')}` |")
    L.append(f"| Material eligible | {'yes' if mat.get('eligible') else 'no'} |")
    L.append("")
    L.append("### 2.3 Transaction")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Transaction drive | {_pct(txn.get('drive'))} |")
    L.append(f"| Transaction kind | `{txn.get('kind', 'n/a')}` |")
    L.append("")
    L.append("### 2.4 UEQGM Runtime")
    L.append("")
    L.append("| Parameter | Value |")
    L.append("|---|---|")
    L.append(f"| Certainty | {_pct(ueqgm.get('certainty'))} |")
    L.append(f"| Symbiotic gain | {_pct(ueqgm.get('symbiotic_gain'))} |")
    L.append(f"| Expansion pressure | {_pct(ueqgm.get('expansion_pressure'))} |")
    L.append(f"| Mesh alignment | {_pct(ueqgm.get('mesh_alignment'))} |")
    L.append("")
    L.append("### 2.5 Bridge Gravity Well")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Primary root | `{bridge.get('primary_root', 'n/a')}` |")
    L.append(f"| Total roots | {bridge.get('total_roots', 'n/a')} |")
    L.append(f"| Mesh density (well depth) | {_pct(bridge.get('mesh_density'))} |")
    L.append(f"| Tunnel saturation | {_pct(bridge.get('tunnel_saturation'))} |")
    L.append(f"| Anchor strength | {_pct(bridge.get('anchor_strength'))} |")
    L.append("")
    L.append("---")
    L.append("")

    # 3. Predictor + STP
    L.append("## 3. MESH-SLM Predictor & STP Diagnostic")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Vocab size | {slm.get('vocab_size', 'n/a')} / {slm.get('vocab_capacity', 4096)} "
             f"({slm.get('vocab_fill_pct', 'n/a')}%) |")
    L.append(f"| Quipu edges (GNN) | {slm.get('quipu_edges', 'n/a')} |")
    L.append(f"| Training rounds | {slm.get('rounds', 'n/a')} |")
    L.append(f"| Last loss | {slm.get('last_loss', 'n/a')} |")
    L.append(f"| 8th-D MESH field | {slm.get('mesh_field_8d', 'n/a')} |")
    L.append(f"| Last STP embed gap | {slm.get('last_stp_embed_gap', 'n/a')} |")
    L.append(f"| Last STP torus gap | {slm.get('last_stp_torus_gap', 'n/a')} |")
    L.append("")
    L.append("**STP P1 trend** (loss plateau while the geodesic gap keeps falling):")
    L.append("")
    L.append("| Series | Slope | P1 |")
    L.append("|---|---|---|")
    L.append(f"| loss | {stp.get('loss_slope', 'n/a')} | plateaued: {stp.get('loss_plateaued', 'n/a')} |")
    L.append(f"| STP embed gap | {stp.get('stp_embed_slope', 'n/a')} | {stp.get('p1_embed', 'n/a')} |")
    L.append(f"| STP torus gap | {stp.get('stp_torus_slope', 'n/a')} | {stp.get('p1_torus', 'n/a')} |")
    if stp.get("insufficient_data"):
        L.append("")
        L.append(f"> Not enough rounds yet for a P1 verdict "
                 f"(window {stp.get('window', 'n/a')}; need 2× window per series).")
    L.append("")
    L.append("---")
    L.append("")

    # 4. Structural changelog
    L.append("## 4. Structural Changelog")
    L.append("")
    history = brain_kv.kv_get_json(_HISTORY_KEY, []) or []
    if history:
        for entry in reversed(history[-12:]):
            L.append(f"- **{entry.get('ts', '?')}** map v{entry.get('map_version', '?')} "
                     f"({entry.get('version', '?')}) — hash `{entry.get('hash', '?')}`"
                     f"{' (forced)' if entry.get('forced') else ''}")
    else:
        L.append("- (no prior anneal cycles recorded yet)")
    L.append("")
    L.append("---")
    L.append("")

    # 5. Module inventory
    L.append("## 5. Module Inventory (QUIPU core)")
    L.append("")
    L.append("| Module | Role |")
    L.append("|---|---|")
    L.append("| `mesh_slm.py` | MESH-SLM predictor, training, STP geodesic diagnostic |")
    L.append("| `system_entirety.py` | 7+1-D state, material bifurcation, bit flip, observer tangent |")
    L.append("| `ueqgm_engine.py` | SiCi axial decay, adaptive runtime, coherence, Floquet |")
    L.append("| `asset_resource_mesh.py` | Physical realization, compute asset mesh, tunnel density |")
    L.append("| `temporal_spatiality.py` | Sense signals, weight prior, torus boundary |")
    L.append("| `brain_kv.py` | Canonical key/value persistence (shared SQLite) |")
    L.append("| `doc_annealing.py` | **This map's generator** — structural-change review + regeneration |")
    L.append("| `local_store.py` | SQLite connection manager |")
    L.append("")
    L.append("*End of living system map — auto-generated by the QUIPU Entirety annealer. "
             "Do not edit by hand; edit `src/quipu/doc_annealing.py` instead.*")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Anneal cycle
# ---------------------------------------------------------------------------
def anneal_docs(
    *,
    force: bool = False,
    out_path: str | Path | None = None,
    collected: dict | None = None,
) -> dict:
    """Run one anneal cycle. Rewrites the map only on structural change or *force*.

    *collected* may be injected (tests); otherwise live state is gathered.
    """
    if collected is None:
        collected = _collect_state()
    review = structural_change_review(collected)
    out = Path(out_path) if out_path else _DEFAULT_MAP
    map_version = int(brain_kv.kv_get_json(_VERSION_KEY, 0) or 0)

    if not review["changed"] and not force and out.exists():
        return {
            "status": "unchanged",
            "fingerprint": review["fingerprint"],
            "map_version": map_version,
            "path": str(out),
        }

    map_version += 1
    text = render_system_map(collected, map_version, review["fingerprint"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    brain_kv.kv_set(_HASH_KEY, review["fingerprint"])
    brain_kv.kv_set_json(_VERSION_KEY, map_version)
    hist = brain_kv.kv_get_json(_HISTORY_KEY, []) or []
    hist.append({
        "ts": collected.get("annealed_at", _now_iso()),
        "version": collected.get("version"),
        "map_version": map_version,
        "hash": review["fingerprint"],
        "forced": bool(force and not review["changed"]),
    })
    brain_kv.kv_set_json(_HISTORY_KEY, hist[-_HISTORY_CAP:])

    return {
        "status": "annealed",
        "fingerprint": review["fingerprint"],
        "prev_hash": review["prev_hash"],
        "map_version": map_version,
        "path": str(out),
        "bytes": len(text),
    }


def run_loop(interval_s: int = _DEFAULT_INTERVAL_S, max_cycles: int | None = None) -> None:
    """Anneal every *interval_s* seconds until interrupted (or *max_cycles*)."""
    cycles = 0
    while True:
        try:
            result = anneal_docs()
            print(f"[{_now_iso()}] {result['status']} — map v{result['map_version']} "
                  f"({result['fingerprint']})", flush=True)
        except Exception as exc:  # a bad cycle must not kill the worker
            print(f"[{_now_iso()}] anneal error: {exc}", flush=True)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            return
        time.sleep(max(1, int(interval_s)))


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Regenerate the QUIPU Entirety living system map.")
    ap.add_argument("--loop", action="store_true", help="run continuously every --interval seconds")
    ap.add_argument("--interval", type=int, default=_DEFAULT_INTERVAL_S,
                    help="seconds between anneal cycles in --loop mode (default 1800)")
    ap.add_argument("--force", action="store_true", help="rewrite the map even if unchanged")
    ap.add_argument("--out", default=None, help="output path (default docs/system_entirety_map.md)")
    ap.add_argument("--json", action="store_true", help="print the result dict as JSON")
    args = ap.parse_args()

    if args.loop:
        run_loop(interval_s=args.interval)
        return

    result = anneal_docs(force=args.force, out_path=args.out)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"{result['status']}: map v{result['map_version']} "
              f"({result['fingerprint']}) → {result['path']}")


if __name__ == "__main__":
    _cli()
