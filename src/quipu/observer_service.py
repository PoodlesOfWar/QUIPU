"""QUIPU Observer — the bi-directional learning hub for the OCR fleet.

QUIPU sits between the two capture applications as the Observer:

    Loadopoly-OCR  (major UNSTRUCTURED images: archival scans, scenery,
                    open-vocabulary documents)  → vision axis (broad observation)
    Bakugo         (relatively STRUCTURED images: trading cards, closed
                    catalog vocabulary, collector numbers)  → touch axis
                    (construction / structured form)

Both applications FEED observations in (``POST /observe``), and both RECEIVE
the mesh's accumulated learnings back (``GET /guidance``) to drive their OCR:
Loadopoly-OCR injects the learned lexicon into its extraction prompt as
disambiguation priors; Bakugo uses cross-corpus token frequencies to break
ties between otherwise-ambiguous catalog collector numbers.

The Observer never blocks a client: everything degrades to "no guidance".
Learning happens in a background trainer thread that folds fresh observations
into the MESH SLM (vocab + quipu bigram edges) on a fixed cadence, so the
enacted guidance strengthens as both corpora accumulate.

Endpoints (all JSON unless noted):
    GET  /health              liveness + mesh counters + hideout identity
    GET  /state               mesh_slm.state_summary()
    GET  /hideout             Black Bulls Hideout realization (scbrain-hideout)
    POST /observe             {source, text, kind?, confidence?, meta?}
    GET  /guidance?source=X   lexicon, pairs, calibration, per-source stats
    POST /feedback            {source, expected, observed?, confidence?}
    GET  /digest              today's learning digest (text/plain)

Environment:
    QUIPU_HOST            bind address (default 0.0.0.0)
    QUIPU_PORT            port (default 7100)
    QUIPU_TRAIN_INTERVAL  seconds between trainer wake-ups (default 45)
    QUIPU_TRAIN_BUDGET    max seconds per train_round (default 8)
    SCB_DB_PATH           SQLite brain location (honoured by local_store)
    QUIPU_HIDEOUT_HOST    MESH peer host (default scbrain-hideout)
    QUIPU_HIDEOUT_MESH    mesh target (default hideout-mesh)
    QUIPU_HIDEOUT_TUNNEL  Dev Tunnel id (default scbrain-hideout.use2)

Stdlib only — no new dependencies.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import brain_kv, hideout_mesh, mesh_slm, world_model
from ._version import __version__
from .local_store import db_path

# ---------------------------------------------------------------------------
# Source profiles — how each application routes onto the mesh sense-axes.
# Marker substrings are chosen so mesh_slm._axis_for_source() routes them:
#   "commoncrawl" → axis 0 (vision: broad unstructured observation)
#   "code"        → axis 1 (touch: structured construction)
# ---------------------------------------------------------------------------
SOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "loadopoly-ocr": {
        "kind": "unstructured",
        "axis": "vision",
        "axis_source": "loadopoly-ocr/commoncrawl/hideout-mesh",
        "sibling": "bakugo",
    },
    "bakugo": {
        "kind": "structured",
        "axis": "touch",
        "axis_source": "bakugo/code/hideout-mesh",
        "sibling": "loadopoly-ocr",
    },
}

_ALIASES = {
    "loadopoly": "loadopoly-ocr",
    "loadopoly_ocr": "loadopoly-ocr",
    "loadopolyocr": "loadopoly-ocr",
    "geograph": "loadopoly-ocr",
    "cardcenter": "bakugo",
}

_STATS_KEY = "observer:{source}:stats"
_FEEDBACK_KEY = "observer:{source}:feedback"
_LAST_TRAIN_KEY = "observer:last_train"

_pending_lock = threading.Lock()
_pending_observations = 0


def _canonical_source(raw: str | None) -> str | None:
    s = (raw or "").strip().lower().replace(" ", "-")
    if s in SOURCE_PROFILES:
        return s
    return _ALIASES.get(s.replace("-", "_"), _ALIASES.get(s))


def _bump_pending(n: int = 1) -> None:
    global _pending_observations
    with _pending_lock:
        _pending_observations += n


def _drain_pending() -> int:
    global _pending_observations
    with _pending_lock:
        n = _pending_observations
        _pending_observations = 0
        return n


def _load_stats(source: str) -> dict[str, Any]:
    return brain_kv.kv_get_json(_STATS_KEY.format(source=source), {}) or {}


def _save_stats(source: str, stats: dict[str, Any]) -> None:
    brain_kv.kv_set_json(_STATS_KEY.format(source=source), stats)


def _record_observation(source: str, tokens: int, confidence: float | None) -> dict[str, Any]:
    stats = _load_stats(source)
    stats["observations"] = int(stats.get("observations", 0)) + 1
    stats["tokens"] = int(stats.get("tokens", 0)) + tokens
    if confidence is not None:
        prev = float(stats.get("confidence_ema", confidence))
        stats["confidence_ema"] = round(0.9 * prev + 0.1 * float(confidence), 4)
    stats["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_stats(source, stats)
    return stats


def _known_token_coverage(tokens: list[str]) -> tuple[float, list[str]]:
    """Fraction of tokens already in the mesh vocab, plus the unknown ones."""
    if not tokens:
        return 0.0, []
    unknown: list[str] = []
    known = 0
    with mesh_slm._conn() as cn:
        for tok in dict.fromkeys(tokens):
            row = cn.execute(
                "SELECT 1 FROM mesh_slm_vocab WHERE token = ?", (tok,)
            ).fetchone()
            if row:
                known += 1
            else:
                unknown.append(tok)
    distinct = len(dict.fromkeys(tokens))
    return (known / distinct if distinct else 0.0), unknown[:12]


def _lexicon(limit: int = 60, numeric_only: bool = False) -> list[dict[str, Any]]:
    """Top mesh tokens by frequency — the shared cross-corpus lexicon."""
    where = "WHERE token GLOB '[0-9]*'" if numeric_only else ""
    with mesh_slm._conn() as cn:
        rows = cn.execute(
            f"SELECT token, freq FROM mesh_slm_vocab {where} "
            "ORDER BY freq DESC, last_seen DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [{"token": r["token"], "freq": int(r["freq"])} for r in rows]


def _strong_pairs(limit: int = 40) -> list[dict[str, Any]]:
    """Strongest quipu bigram edges — context for post-OCR correction."""
    with mesh_slm._conn() as cn:
        rows = cn.execute(
            "SELECT a.token AS src, b.token AS dst, q.weight, q.samples "
            "FROM mesh_slm_quipu q "
            "JOIN mesh_slm_vocab a ON a.token_id = q.src "
            "JOIN mesh_slm_vocab b ON b.token_id = q.dst "
            "ORDER BY q.weight DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [
        {
            "src": r["src"],
            "dst": r["dst"],
            "weight": round(float(r["weight"]), 5),
            "samples": int(r["samples"]),
        }
        for r in rows
    ]


def _calibration(source: str) -> dict[str, Any]:
    """Confidence guidance derived from what the Observer has seen."""
    stats = _load_stats(source)
    ema = stats.get("confidence_ema")
    if ema is None:
        return {"suggested_min_confidence": 0.5, "basis": "default (no observations yet)"}
    suggested = max(0.35, min(0.9, round(float(ema) - 0.15, 3)))
    return {
        "suggested_min_confidence": suggested,
        "basis": f"EMA of {stats.get('observations', 0)} observed confidences ({ema})",
    }


def _guidance(source: str, limit: int) -> dict[str, Any]:
    profile = SOURCE_PROFILES[source]
    sibling = profile["sibling"]
    summary = mesh_slm.state_summary()
    return {
        "ok": True,
        "source": source,
        "kind": profile["kind"],
        "axis": profile["axis"],
        "lexicon": _lexicon(limit),
        "numeric_lexicon": _lexicon(30, numeric_only=True),
        "pairs": _strong_pairs(),
        "calibration": _calibration(source),
        "world_model": world_model.world_model_state(),
        "retrieval_directive": world_model.retrieval_directive(),
        "sources": {
            source: _load_stats(source),
            sibling: _load_stats(sibling),
        },
        "mesh": {
            "vocab": summary.get("vocab_size"),
            "edges": summary.get("quipu_edges"),
            "weyl_tensor": brain_kv.kv_get_json("learnings:weyl_tensor", None),
        },
        "last_train": brain_kv.kv_get_json(_LAST_TRAIN_KEY, None),
    }


def _observe(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    source = _canonical_source(body.get("source"))
    if source is None:
        return 400, {
            "ok": False,
            "error": "unknown source; expected one of "
            + ", ".join(sorted(SOURCE_PROFILES)),
        }
    text = str(body.get("text") or "").strip()
    if not text:
        return 400, {"ok": False, "error": "text is required"}

    profile = SOURCE_PROFILES[source]
    confidence = body.get("confidence")
    try:
        confidence = None if confidence is None else float(confidence)
    except (TypeError, ValueError):
        confidence = None

    tokens = mesh_slm._tokenize(text)
    fed = mesh_slm.feed_corpus(text, source=profile["axis_source"])
    coverage, unknown = _known_token_coverage(tokens)
    stats = _record_observation(source, len(tokens), confidence)
    _bump_pending()

    world_model_result = world_model.assess_observation(
        source=source,
        tokens=tokens,
        confidence=confidence,
        coverage=coverage,
        novel_tokens=unknown,
    )

    summary = mesh_slm.state_summary()
    return 200, {
        "ok": True,
        "source": source,
        "kind": body.get("kind") or profile["kind"],
        "axis": profile["axis"],
        "fed_rows": fed,
        "tokens": len(tokens),
        "enacted": {
            "known_token_coverage": round(coverage, 4),
            "novel_tokens": unknown,
            "calibration": _calibration(source),
        },
        "world_model": world_model_result,
        "totals": {
            "observations": stats.get("observations"),
            "tokens": stats.get("tokens"),
        },
        "mesh": {"vocab": summary.get("vocab_size"), "edges": summary.get("quipu_edges")},
        "hideout": hideout_mesh.hideout_identity(),
    }


def _feedback(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Ground-truth reinforcement: the app tells the Observer what was right."""
    source = _canonical_source(body.get("source"))
    if source is None:
        return 400, {"ok": False, "error": "unknown source"}
    expected = str(body.get("expected") or "").strip()
    if not expected:
        return 400, {"ok": False, "error": "expected is required"}
    observed = str(body.get("observed") or "").strip()

    profile = SOURCE_PROFILES[source]
    # Ground truth is fed with extra weight (twice) so corrections outweigh
    # the original misreading in the bigram statistics.
    mesh_slm.feed_corpus(expected, source=profile["axis_source"])
    mesh_slm.feed_corpus(expected, source=profile["axis_source"])
    _bump_pending(2)

    key = _FEEDBACK_KEY.format(source=source)
    log = brain_kv.kv_get_json(key, []) or []
    log.append(
        {
            "expected": expected[:200],
            "observed": observed[:200],
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    brain_kv.kv_set_json(key, log[-200:])
    return 200, {"ok": True, "source": source, "recorded": True, "log_size": len(log)}


# ---------------------------------------------------------------------------
# Background trainer — the Observer's learning loop.
# ---------------------------------------------------------------------------

def _trainer_loop(interval: float, budget: float) -> None:
    while True:
        time.sleep(interval)
        pending = _drain_pending()
        if pending <= 0:
            continue
        try:
            started = time.time()
            summary = mesh_slm.train_round(max_seconds=budget, max_chunks=200)
            brain_kv.kv_set_json(
                _LAST_TRAIN_KEY,
                {
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "pending_consumed": pending,
                    "seconds": round(time.time() - started, 2),
                    "summary": {
                        k: v
                        for k, v in (summary or {}).items()
                        if isinstance(v, (int, float, str, bool))
                    },
                },
            )
            print(f"[observer] train_round: {pending} pending folded in "
                  f"({time.time() - started:.1f}s)", flush=True)
            hideout_mesh.realize_hideout_fleet()
        except Exception:
            traceback.print_exc()


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class ObserverHandler(BaseHTTPRequestHandler):
    server_version = f"quipu-observer/{__version__}"

    def log_message(self, fmt: str, *args: Any) -> None:  # keep stdout clean
        pass

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self._send_raw(code, body, "application/json")

    def _send_raw(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # CORS preflight for the browser clients
        self._send_raw(204, b"", "text/plain")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                summary = mesh_slm.state_summary()
                self._send_json(200, {
                    "ok": True,
                    "service": "quipu-observer",
                    "version": __version__,
                    "db": str(db_path()),
                    "vocab": summary.get("vocab_size"),
                    "edges": summary.get("quipu_edges"),
                    "pending": _pending_observations,
                    "hideout": hideout_mesh.hideout_identity(),
                })
            elif parsed.path == "/state":
                self._send_json(200, mesh_slm.state_summary())
            elif parsed.path == "/hideout":
                self._send_json(200, hideout_mesh.hideout_status())
            elif parsed.path == "/guidance":
                qs = parse_qs(parsed.query)
                source = _canonical_source((qs.get("source") or [None])[0])
                if source is None:
                    self._send_json(400, {
                        "ok": False,
                        "error": "guidance requires ?source=loadopoly-ocr|bakugo",
                    })
                    return
                limit = int((qs.get("limit") or ["60"])[0])
                self._send_json(200, _guidance(source, max(1, min(limit, 500))))
            elif parsed.path == "/world-model":
                self._send_json(200, world_model.world_model_state())
            elif parsed.path == "/digest":
                try:
                    from . import daily_digest

                    text = daily_digest.format_digest(daily_digest.build_digest())
                except Exception as exc:
                    text = f"digest unavailable: {exc}"
                self._send_raw(200, text.encode(), "text/plain; charset=utf-8")
            else:
                self._send_json(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._send_json(400, {"ok": False, "error": "invalid JSON body"})
                return
            if not isinstance(body, dict):
                self._send_json(400, {"ok": False, "error": "body must be an object"})
                return

            if parsed.path == "/observe":
                code, payload = _observe(body)
            elif parsed.path == "/feedback":
                code, payload = _feedback(body)
            else:
                code, payload = 404, {"ok": False, "error": "not found"}
            self._send_json(code, payload)
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"ok": False, "error": str(exc)})


def run() -> None:
    host = os.environ.get("QUIPU_HOST", "0.0.0.0")
    port = int(os.environ.get("QUIPU_PORT", "7100"))
    interval = float(os.environ.get("QUIPU_TRAIN_INTERVAL", "45"))
    budget = float(os.environ.get("QUIPU_TRAIN_BUDGET", "8"))

    trainer = threading.Thread(
        target=_trainer_loop, args=(interval, budget), daemon=True,
        name="quipu-observer-trainer",
    )
    trainer.start()

    hideout = hideout_mesh.realize_hideout_fleet()
    identity = hideout_mesh.hideout_identity()
    httpd = ThreadingHTTPServer((host, port), ObserverHandler)
    print(f"[observer] QUIPU Observer v{__version__} on http://{host}:{port}", flush=True)
    print(f"[observer] brain: {db_path()}", flush=True)
    print(f"[observer] trainer: every {interval:.0f}s, budget {budget:.0f}s", flush=True)
    print(
        f"[observer] hideout: {identity['host']} / {identity['mesh']} "
        f"via {identity['tunnel_id']} realized={hideout.get('ok')}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run()
