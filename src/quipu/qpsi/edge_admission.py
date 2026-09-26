"""edge_admission — who may write into QUIPU from outside, and how much.

QUIPU's only door from the fleet is observer_service (:7100).  Before v0.47.0
anything that could reach the port could post as any source, trigger an
annealing cycle, or — through ``Access-Control-Allow-Origin: *`` — be driven by
any web page the operator had open.  This module is the edge in front of every
mutating route (``POST /observe``, ``POST /feedback``, ``POST|GET /anneal``).

Two parts, each with its own mode (``off`` | ``record`` | ``enforce``):

1. Identity  (``QUIPU_EDGE_AUTH``, default ``record``)
------------------------------------------------------
A source proves who it is in one of two ways:

    signed    X-Quipu-Source, X-Quipu-Timestamp, X-Quipu-Signature, where
              signature = hex HMAC-SHA256(key_source,
                                          METHOD \\n PATH \\n TS \\n sha256(body))
              with the source's own key.  The timestamp must be within the skew
              window and a signature is accepted once (replay cache).  The body's
              ``source`` must be the signing source.
    origin    a browser client that cannot hold a key: the request's Origin must
              be one the operator bound to that source in the grant
              (``browser_origins``).  Weaker; recorded as such.

Keys come from ``QUIPU_EDGE_KEYS_FILE`` (JSON ``{"sources": {name: hex}}``),
minted by the operator (``python -m src.quipu.qpsi.edge_admission mint``) and
kept outside the repo.  ``record`` verifies and labels every write (verified /
origin / unverified) without refusing any; ``enforce`` refuses unverified
writes with 401.

2. Admission budget  (``QUIPU_EDGE_BUDGET``, default ``record``)
----------------------------------------------------------------
The operator's grant (``QUIPU_EDGE_GRANT_FILE``) names the sources allowed in,
each with a ceiling in tokens per hour, and a total:

    {"grant_ref": "...", "total_tokens_per_hour": 60000,
     "sources": {"hubcore": {"max_tokens_per_hour": 20000},
                 "loadopoly-ocr": {"max_tokens_per_hour": 20000,
                                   "browser_origins": ["http://localhost:3000"]}}}

Within it the system optimises: each source's admitted rate is its share of the
total by the information it has been delivering — novel tokens per token over
the trailing hour (the observer's ``novel_tokens``) — water-filled so no source
exceeds its ceiling and budget freed by a capped source goes to the others.  A
source without data yet is weighted at the mean of the others, so a new
granted source gets a fair first share; with no data at all the split is even.
Admission is a sliding hour: a write is admitted while the source's admitted
tokens in the last hour plus this write fit its allocation.

This is the constrained gate of the 2026-09-22 Invariance #7 ruling applied to
the edge: the system allocates the operator's grant among the operator's
sources.  It cannot add a source, raise a ceiling or the total, mint or read
out a key, or change a mode — the grant and keys files are read, never
written, and every allocation is checked against the grant before it is used.
``enforce`` refuses writes from sources outside the grant (403) and writes over
allocation (429, Retry-After).  ``record`` computes the same decisions, reports
what it would have refused, and refuses nothing.

Persistence: the plan is written to ``brain_kv["entirety:edge:plan"]`` only
when an allocation moves by at least one token per hour, and per-source
counters to ``entirety:edge:stats`` at most once a minute.

Stdlib only.

翈 — the door knows who is knocking; how wide it opens is the grant's, not ours.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

AUTH_ENV = "QUIPU_EDGE_AUTH"
BUDGET_ENV = "QUIPU_EDGE_BUDGET"
KEYS_ENV = "QUIPU_EDGE_KEYS_FILE"
GRANT_ENV = "QUIPU_EDGE_GRANT_FILE"
SKEW_ENV = "QUIPU_EDGE_SKEW_S"
MODES = ("off", "record", "enforce")
HOUR = 3600.0
DEFAULT_SKEW_S = 300.0
KV_PLAN = "entirety:edge:plan"
KV_STATS = "entirety:edge:stats"
STATS_FLUSH_S = 60.0
MUTATING = {("POST", "/observe"), ("POST", "/feedback"), ("POST", "/anneal"), ("GET", "/anneal")}
H_SOURCE, H_TS, H_SIG = "X-Quipu-Source", "X-Quipu-Timestamp", "X-Quipu-Signature"
# The browser surfaces hub_workspace.json declares on this host (Loadopoly-OCR
# :3000, the portal :9080).  Set QUIPU_CORS_ORIGINS to replace the list.
DEFAULT_CORS_ORIGINS = ("http://localhost:3000,http://127.0.0.1:3000,"
                        "http://localhost:9080,http://127.0.0.1:9080")


def _mode(env: str) -> str:
    m = os.environ.get(env, "record").strip().lower()
    return m if m in MODES else "record"


def _default_file(name: str) -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return (Path(base) / "QUIPU" / name) if base else (Path.home() / ".quipu" / name)


def keys_path() -> Path:
    return Path(os.environ.get(KEYS_ENV) or _default_file("edge_keys.json"))


def grant_path() -> Path:
    return Path(os.environ.get(GRANT_ENV) or _default_file("edge_grant.json"))


def _load_json(p: Path) -> dict:
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# ===========================================================================
# Signing (clients use this)
# ===========================================================================

def canonical(method: str, path: str, ts: str, body: bytes) -> bytes:
    return f"{method.upper()}\n{path}\n{ts}\n{hashlib.sha256(body).hexdigest()}".encode()


def sign(key_hex: str, method: str, path: str, body: bytes, source: str, ts: float | None = None) -> dict:
    """Headers a client adds to a request.  ``body`` must be the exact bytes sent."""
    t = str(int(time.time() if ts is None else ts))
    mac = hmac.new(bytes.fromhex(key_hex), canonical(method, path, t, body), hashlib.sha256).hexdigest()
    return {H_SOURCE: source, H_TS: t, H_SIG: mac}


# ===========================================================================
# Grant and allocation (pure)
# ===========================================================================

@dataclass(frozen=True)
class Grant:
    ref: str
    total: float
    ceilings: dict[str, float]
    origins: dict[str, tuple[str, ...]]

    @classmethod
    def from_json(cls, d: Mapping) -> "Grant":
        srcs = d.get("sources") if isinstance(d.get("sources"), Mapping) else {}
        ceil, orig = {}, {}
        for name, spec in srcs.items():
            spec = spec if isinstance(spec, Mapping) else {}
            try:
                c = float(spec.get("max_tokens_per_hour", 0.0))
            except (TypeError, ValueError):
                c = 0.0
            if math.isfinite(c) and c > 0.0:
                ceil[str(name)] = c
            o = spec.get("browser_origins") or []
            orig[str(name)] = tuple(str(x).rstrip("/") for x in o if isinstance(x, str))
        try:
            total = float(d.get("total_tokens_per_hour", sum(ceil.values())))
        except (TypeError, ValueError):
            total = sum(ceil.values())
        total = max(0.0, min(total, sum(ceil.values()))) if math.isfinite(total) else 0.0
        return cls(ref=str(d.get("grant_ref") or ""), total=total, ceilings=ceil, origins=orig)

    def allows(self, source: str) -> bool:
        return source in self.ceilings


def allocate(grant: Grant, info: Mapping[str, float | None]) -> dict[str, float]:
    """Water-fill the grant's total over its sources by measured information.

    ``info[s]`` is novel tokens per token (None: no data yet).  Never exceeds a
    ceiling; the sum never exceeds the total; only granted sources appear.
    """
    srcs = sorted(grant.ceilings)
    if not srcs or grant.total <= 0.0:
        return {s: 0.0 for s in srcs}
    seen = [float(info[s]) for s in srcs if info.get(s) is not None and float(info[s]) > 0.0]
    prior = (sum(seen) / len(seen)) if seen else 1.0
    w = {s: (float(info[s]) if info.get(s) is not None else prior) for s in srcs}
    if all(v <= 0.0 for v in w.values()):
        w = {s: 1.0 for s in srcs}
    out = {s: 0.0 for s in srcs}
    left, free = grant.total, set(srcs)
    while free and left > 1e-9:
        wsum = sum(w[s] for s in free)
        if wsum <= 0.0:
            share = {s: left / len(free) for s in free}
        else:
            share = {s: left * w[s] / wsum for s in free}
        capped = {s for s in free if out[s] + share[s] >= grant.ceilings[s] - 1e-9}
        if not capped:
            for s in free:
                out[s] += share[s]
            left = 0.0
            break
        for s in capped:
            left -= grant.ceilings[s] - out[s]
            out[s] = grant.ceilings[s]
        free -= capped
    return out


# ===========================================================================
# The edge
# ===========================================================================

@dataclass
class Decision:
    ok: bool
    code: int = 200
    error: str | None = None
    source: str | None = None
    assurance: str = "unverified"          # signed | origin | unverified
    tokens: int = 0
    would_refuse: str | None = None        # what enforce would have done, in record mode
    retry_after: int | None = None

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class _Src:
    admitted: deque = field(default_factory=deque)     # (t, tokens) admitted, trailing hour
    results: deque = field(default_factory=deque)      # (t, tokens, novel) trailing hour
    counts: dict = field(default_factory=lambda: {"signed": 0, "origin": 0, "unverified": 0,
                                                  "refused": 0, "would_refuse": 0})


class Edge:
    def __init__(self, *, clock: Callable[[], float] = time.time,
                 keys: Mapping[str, str] | None = None, grant: Mapping | None = None,
                 kv: Any = None):
        self._clock = clock
        self._lock = threading.Lock()
        self._keys_override = dict(keys) if keys is not None else None
        self._grant_override = grant
        self._kv = kv
        self._src: dict[str, _Src] = {}
        self._seen_sigs: dict[str, float] = {}
        self._plan: dict[str, float] = {}
        self._last_flush = 0.0
        self._cache: dict[str, tuple[float, Any]] = {}

    # --- configuration (read only; re-read when the file changes) ----------
    def _file(self, key: str, p: Path) -> dict:
        try:
            m = p.stat().st_mtime
        except OSError:
            return {}
        hit = self._cache.get(key)
        if hit and hit[0] == m:
            return hit[1]
        data = _load_json(p)
        self._cache[key] = (m, data)
        return data

    def keys(self) -> dict[str, str]:
        if self._keys_override is not None:
            return self._keys_override
        d = self._file("keys", keys_path()).get("sources") or {}
        return {str(k): str(v) for k, v in d.items() if isinstance(v, str)}

    def grant(self) -> Grant | None:
        if self._grant_override is not None:
            return Grant.from_json(self._grant_override)
        d = self._file("grant", grant_path())
        return Grant.from_json(d) if d else None

    # --- identity ------------------------------------------------------------
    def _identify(self, method: str, path: str, headers: Mapping, raw: bytes, body_source: str | None,
                  canon: Callable[[str | None], str | None]) -> tuple[str, str | None, str | None]:
        """(assurance, source, error)."""
        h = {str(k).lower(): v for k, v in headers.items()}
        hs, ts, sig = h.get(H_SOURCE.lower()), h.get(H_TS.lower()), h.get(H_SIG.lower())
        if hs or sig:
            src = canon(hs)
            key = self.keys().get(src or "")
            if not (src and ts and sig and key):
                return "unverified", body_source, "signature incomplete or source has no key"
            try:
                t = float(ts)
            except (TypeError, ValueError):
                return "unverified", body_source, "bad timestamp"
            skew = float(os.environ.get(SKEW_ENV, DEFAULT_SKEW_S))
            now = self._clock()
            if abs(now - t) > skew:
                return "unverified", body_source, "timestamp outside the skew window"
            want = hmac.new(bytes.fromhex(key), canonical(method, path, str(ts), raw), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(want, str(sig)):
                return "unverified", body_source, "signature mismatch"
            for s_, t_ in list(self._seen_sigs.items()):
                if now - t_ > skew:
                    del self._seen_sigs[s_]
            if sig in self._seen_sigs:
                return "unverified", body_source, "replayed signature"
            self._seen_sigs[str(sig)] = now
            if body_source is not None and body_source != src:
                return "unverified", body_source, f"signed as {src} but body says {body_source}"
            return "signed", src, None
        origin = str(h.get("origin") or "").rstrip("/")
        g = self.grant()
        if origin and g and body_source and origin in g.origins.get(body_source, ()):
            return "origin", body_source, None
        return "unverified", body_source, "unsigned"

    # --- information and plan ------------------------------------------------
    def _trim(self, now: float) -> None:
        for st in self._src.values():
            while st.admitted and now - st.admitted[0][0] > HOUR:
                st.admitted.popleft()
            while st.results and now - st.results[0][0] > HOUR:
                st.results.popleft()

    def information(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for s, st in self._src.items():
            tok = sum(r[1] for r in st.results)
            out[s] = (sum(r[2] for r in st.results) / tok) if tok > 0 else None
        return out

    def plan(self) -> dict[str, float]:
        g = self.grant()
        if not g:
            return {}
        alloc = allocate(g, self.information())
        moved = set(alloc) != set(self._plan) or any(abs(alloc[s] - self._plan.get(s, 0.0)) >= 1.0 for s in alloc)
        if moved:
            self._plan = alloc
            self._kv_set(KV_PLAN, {"grant_ref": g.ref, "total": g.total, "ceilings": g.ceilings,
                                   "allocation": {s: round(v, 3) for s, v in alloc.items()},
                                   "information": {s: (round(v, 6) if v is not None else None)
                                                   for s, v in self.information().items() if s in alloc},
                                   "at": self._clock()})
        return dict(self._plan)

    # --- the gate -----------------------------------------------------------
    def admit(self, method: str, path: str, headers: Mapping, raw: bytes, body: Mapping,
              canon: Callable[[str | None], str | None], tokens: int) -> Decision:
        if (method.upper(), path) not in MUTATING:
            return Decision(ok=True, assurance="n/a")
        auth, budget = _mode(AUTH_ENV), _mode(BUDGET_ENV)
        with self._lock:
            now = self._clock()
            self._trim(now)
            body_source = canon(body.get("source")) if isinstance(body, Mapping) else None
            assurance, source, err = ("unverified", body_source, None) if auth == "off" else \
                self._identify(method, path, headers, raw, body_source, canon)
            st = self._src.setdefault(source or "?", _Src())
            d = Decision(ok=True, source=source, assurance=assurance, tokens=int(tokens))
            refusals: list[tuple[int, str, int | None]] = []
            if auth != "off" and assurance == "unverified":
                refusals.append((401, f"unverified: {err}", None))
            g = self.grant() if budget != "off" else None
            if g is not None and path in ("/observe", "/feedback"):
                if not source or not g.allows(source):
                    refusals.append((403, f"source {source!r} is not in the operator's grant", None))
                else:
                    alloc = self.plan().get(source, 0.0)
                    used = sum(t for _, t in st.admitted)
                    if used + tokens > alloc:
                        wait = 60
                        if st.admitted:
                            need, acc = used + tokens - alloc, 0.0
                            for t0, n in st.admitted:
                                acc += n
                                if acc >= need:
                                    wait = max(1, int(t0 + HOUR - now) + 1)
                                    break
                        refusals.append((429, f"over allocation ({used:.0f}+{tokens} > {alloc:.0f} tokens/h)", wait))
            enforce_codes = {401} if auth == "enforce" else set()
            if budget == "enforce":
                enforce_codes |= {403, 429}
            hard = [r for r in refusals if r[0] in enforce_codes]
            soft = [r for r in refusals if r[0] not in enforce_codes]
            if hard:
                code, msg, retry = hard[0]
                d.ok, d.code, d.error, d.retry_after = False, code, msg, retry
                st.counts["refused"] += 1
            else:
                if soft:
                    d.would_refuse = soft[0][1]
                    st.counts["would_refuse"] += 1
                st.admitted.append((now, int(tokens)))
                st.counts[assurance] = st.counts.get(assurance, 0) + 1
            self._maybe_flush(now)
            return d

    def settle(self, decision: Decision, result: Mapping | None) -> None:
        """Feed the outcome back: how many of the admitted tokens were new to the mesh."""
        if not decision.ok or not decision.source or not isinstance(result, Mapping):
            return
        enacted = result.get("enacted") if isinstance(result.get("enacted"), Mapping) else {}
        novel = enacted.get("novel_tokens")
        n_novel = len(novel) if isinstance(novel, (list, tuple)) else (int(novel) if isinstance(novel, int) else None)
        tok = result.get("tokens", decision.tokens)
        try:
            tok = int(tok)
        except (TypeError, ValueError):
            tok = decision.tokens
        if n_novel is None or tok <= 0:
            return
        with self._lock:
            self._src.setdefault(decision.source, _Src()).results.append((self._clock(), tok, min(n_novel, tok)))

    # --- persistence ----------------------------------------------------------
    def _kv_set(self, key: str, value: Any) -> None:
        try:
            if self._kv is not None:
                self._kv(key, value)
            else:
                from .. import brain_kv
                brain_kv.kv_set_json(key, value)
        except Exception:
            pass

    def _maybe_flush(self, now: float) -> None:
        if now - self._last_flush < STATS_FLUSH_S:
            return
        self._last_flush = now
        self._kv_set(KV_STATS, self.status(locked=True))

    def status(self, *, locked: bool = False) -> dict:
        def build() -> dict:
            g = self.grant()
            return {
                "auth": _mode(AUTH_ENV), "budget": _mode(BUDGET_ENV),
                "keys_file": str(keys_path()), "keyed_sources": sorted(self.keys()),
                "grant_file": str(grant_path()), "grant_ref": g.ref if g else None,
                "total_tokens_per_hour": g.total if g else None,
                "ceilings": g.ceilings if g else {},
                "allocation": {s: round(v, 3) for s, v in self._plan.items()},
                "sources": {s: {"admitted_last_hour": sum(t for _, t in st.admitted),
                                "information": self.information().get(s), **st.counts}
                            for s, st in sorted(self._src.items())},
            }
        if locked:
            return build()
        with self._lock:
            self._trim(self._clock())
            return build()


_EDGE: Edge | None = None


def edge() -> Edge:
    global _EDGE
    if _EDGE is None:
        _EDGE = Edge()
    return _EDGE


def cors_origin(origin: str | None) -> str | None:
    """The Origin to echo, or None.  Only origins bound in the grant, plus
    ``QUIPU_CORS_ORIGINS`` (comma list, operator-set), are echoed."""
    o = (origin or "").rstrip("/")
    if not o:
        return None
    raw = os.environ.get("QUIPU_CORS_ORIGINS")
    if raw is None:
        raw = DEFAULT_CORS_ORIGINS
    allowed = {x.strip().rstrip("/") for x in raw.split(",") if x.strip()}
    g = edge().grant()
    if g:
        for v in g.origins.values():
            allowed.update(v)
    return o if o in allowed else None


# ===========================================================================
# Operator CLI (minting is the operator's act)
# ===========================================================================

def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="edge_admission", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mint", help="create a key for each named source; prints env lines for the clients")
    m.add_argument("sources", nargs="+")
    m.add_argument("--keys-file", default=None)
    m.add_argument("--rotate", action="store_true", help="replace existing keys")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    if args.cmd == "status":
        print(json.dumps(edge().status(), indent=2, default=str))
        return 0
    p = Path(args.keys_file) if args.keys_file else keys_path()
    data = _load_json(p)
    srcs = dict(data.get("sources") or {})
    for s in args.sources:
        if s in srcs and not args.rotate:
            continue
        srcs[s] = secrets.token_hex(32)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"sources": srcs}, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    print(f"# keys file: {p}")
    for s in args.sources:
        print(f"QUIPU_EDGE_KEY_{s.upper().replace('-', '_')}={srcs[s]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = ["AUTH_ENV", "BUDGET_ENV", "KEYS_ENV", "GRANT_ENV", "MUTATING", "H_SOURCE", "H_TS", "H_SIG",
           "canonical", "sign", "Grant", "allocate", "Decision", "Edge", "edge", "cors_origin",
           "keys_path", "grant_path"]
