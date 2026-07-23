# Implementation Plan — STP-style Geodesic Diagnostic in `train_round()`

> **Status — Phase 1 landed in v0.25.0 (2026-07-23).** Sections 1–4 are implemented in
> `src/quipu/mesh_slm.py` (helpers near `_proximity`, wiring in `train_round()`,
> `stp_diagnostic_trend()` near `state_summary()`) and `tests/test_mesh_slm.py` (7 tests),
> behind `QUIPU_STP_DIAGNOSTIC` (default on). Phase 2 (§5.4, η coupling) remains deferred
> until `stp_diagnostic_trend()` confirms the P1 signature on real corpus data.

**Companion to**: `docs/STP_TORUS_QUIPU.md` (§5, Follow-up).
**Target module**: `src/quipu/mesh_slm.py`.
**Goal**: add a passive, flagged diagnostic that samples `s < r < t` from each round's token trajectories and measures whether QUIPU's own torus geometry exhibits the paper's **P1 signature** — the STP-style geometric loss keeps falling after the ordinary training loss plateaus. If it does, that's empirical evidence the torus placement is doing real geodesic work, not just serving as a convenient index. Only *after* that's confirmed does folding the signal into `η_eff` become a justified follow-up, not before.

This plan is scoped to be a **single, low-risk, additive PR**: no change to existing training outcomes, no new required dependencies, fails toward legacy behavior on any error, matches existing code conventions exactly.

---

## 0. Grounding — what already exists

Read directly from the current QUIPU source, so the plan below wires into real functions rather than assumed ones:

| Thing | Location |
|---|---|
| `train_round()` | `src/quipu/mesh_slm.py:2143-2347` |
| Per-chunk token-id trajectory `ids` | built at `mesh_slm.py:2223-2226` inside the `for chunk, source in chunks:` loop (line 2208) — **this is the "recent trajectory"** the diagnostic samples from |
| 7-D embedding table | `mesh_slm_embed(token_id, e_vision, e_touch, e_smell, e_body, e_brain, e_perception, e_entirety)` — updated in place at `mesh_slm.py:2229-2244` |
| Vocab torus-cell table | `mesh_slm_vocab(token_id, i, j, freq, ...)`, `i, j ∈ [0, _TORUS_N)` |
| `_TORUS_N` | `mesh_slm.py:711` (`= 64`) |
| Wrap-aware distance | `_torus_dist(a, b)` — `mesh_slm.py:2023-2031`, L1 wrap distance on `(i, j)` |
| `_proximity(a, b)` | `mesh_slm.py:2034-2043` — scalar decay from `_torus_dist` |
| `_mean_embed_7d(cn, limit)` | `mesh_slm.py:1769-1795` — pattern to copy for a single-token embed fetch |
| `mesh_slm_meta` key/value store | schema `mesh_slm.py:815`; `_meta_get`/`_meta_set` at `mesh_slm.py:866`, `880` |
| Existing flag pattern | `_acre_enabled()`, `mesh_slm.py:1597-1601` — reads an env var, defaults **on**, falsy strings disable it |
| `η`/`η_q` computation | `mesh_slm.py:2185-2186`, already modulated by `phase_weight` and `overlap` — the eventual hook point for Phase 2 |
| Existing summary fields added the same way (`phase_weight`, `wavefunction_overlap`, `mesh_field_8d`) | computed once per round, stored to meta (`mesh_slm.py:2282-2284`), returned in the summary dict (`mesh_slm.py:2341-2343`) — **this diagnostic follows the identical pattern** |
| Test fixture | `isolated_slm_db` in `tests/test_mesh_slm.py:7-62`; existing precedent test `test_train_round_reports_ueqgm_fields` at `tests/test_mesh_slm.py:216-229` |

**One design correction to the original note.** The suggestion was to "compute 1 − cos on 7-D embeddings using wrap-aware `torus_dist` differences." `_torus_dist` returns a **scalar** (an L1 wrap distance); a cosine formula needs **vectors** to difference. Reusing it literally isn't well-defined. Two separate, well-defined diagnostics instead:

1. **Embedding-space gap** — paper-faithful. QUIPU's nearest analogue to a "hidden state" is the 7-D embedding row. Compute `1 − cos(h_t − h_r, h_r − h_s)` exactly as the paper does, no wrap needed (these are plain bounded scalars in `[0,1]`, not angles).
2. **Torus-space gap** — the actually novel, QUIPU-native test, since this is what has no analogue in the paper at all. Embed each sampled token's `(i, j)` cell isometrically onto the flat torus in `ℝ⁴`: `u = (cos θ, sin θ, cos φ, sin φ)` with `θ = 2π·i/_TORUS_N`, `φ = 2π·j/_TORUS_N`. This is wrap-aware *by construction* (sin/cos are already periodic — cell `(63,0)` and `(0,0)` map to adjacent points in `ℝ⁴`, unlike raw coordinate subtraction, which would show a jump of 63). Apply the identical `1 − cos(u_t − u_r, u_r − u_s)` formula in this `ℝ⁴` space.

Reporting both lets us tell whether any geodesic signal is coming from the *learned* embedding, the *graph topology* (torus placement), or both — which is exactly the question "is the torus placement empirically validated" is asking.

---

## 1. New helpers (pure functions, colocated with `_torus_dist`/`_proximity`, `mesh_slm.py:~2044`)

```python
# ---------------------------------------------------------------------------
# STP-style geodesic diagnostic (docs/STP_DIAGNOSTIC_PLAN.md)
# ---------------------------------------------------------------------------
_STP_DIAG_ENV: str = "QUIPU_STP_DIAGNOSTIC"
_STP_HISTORY_CAP: int = 200          # rolling window kept in mesh_slm_meta


def _stp_diagnostic_enabled() -> bool:
    """True unless QUIPU_STP_DIAGNOSTIC is explicitly falsy (default: enabled).

    Purely observational — computing it cannot change training outcomes —
    so it defaults on like phase_weight/overlap/mesh_field_8d. Mirrors
    _acre_enabled()'s flag convention exactly.
    """
    return str(os.environ.get(_STP_DIAG_ENV, "1")).strip().lower() not in (
        "0", "false", "no", "off",
    )


def _stp_cos_gap(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float | None:
    """1 − cos(c − b, b − a) — the Semantic Tube Prediction loss, generic over
    any equal-length vector triple (works for 7-D embeddings and for the R^4
    torus embedding below). Returns None if either difference vector is ~0
    (degenerate/duplicate points — undefined direction, not a signal)."""
    bx = [ci - bi for ci, bi in zip(c, b)]
    ab = [bi - ai for bi, ai in zip(b, a)]
    nb = math.sqrt(sum(v * v for v in bx))
    na = math.sqrt(sum(v * v for v in ab))
    if nb < 1e-9 or na < 1e-9:
        return None
    dot = sum(x * y for x, y in zip(bx, ab))
    cos = max(-1.0, min(1.0, dot / (nb * na)))
    return 1.0 - cos


def _torus_point_r4(cell: tuple[int, int]) -> list[float]:
    """Isometric flat-torus embedding of a vocab cell into R^4: (cosθ, sinθ,
    cosφ, sinφ). Wrap-aware by construction — this is what makes the torus
    diagnostic well-defined across the wrap boundary, unlike raw (i, j)
    differencing."""
    theta = 2.0 * math.pi * cell[0] / float(_TORUS_N)
    phi = 2.0 * math.pi * cell[1] / float(_TORUS_N)
    return [math.cos(theta), math.sin(theta), math.cos(phi), math.sin(phi)]


def _sample_stp_triplet(n: int) -> tuple[int, int, int] | None:
    """Random 0 <= s < r < t < n. None if n < 3 (chunk too short to sample)."""
    if n < 3:
        return None
    s, r, t = sorted(random.sample(range(n), 3))
    return s, r, t


def _embed7_for_token(cn: sqlite3.Connection, token_id: int) -> list[float] | None:
    row = cn.execute(
        "SELECT e_vision, e_touch, e_smell, e_body, e_brain, e_perception, e_entirety "
        "FROM mesh_slm_embed WHERE token_id=?",
        (token_id,),
    ).fetchone()
    if row is None:
        return None
    return [float(row[k] or 0.0) for k in
            ("e_vision", "e_touch", "e_smell", "e_body", "e_brain", "e_perception", "e_entirety")]


def _torus_cell_for_token(cn: sqlite3.Connection, token_id: int) -> tuple[int, int] | None:
    row = cn.execute("SELECT i, j FROM mesh_slm_vocab WHERE token_id=?", (token_id,)).fetchone()
    if row is None:
        return None
    return int(row["i"]), int(row["j"])
```

All pure stdlib (`math`, `random`, already imported), no numpy — consistent with the rest of the module and with `tool_forge`'s "pure stdlib, no LLM call" ethos.

---

## 2. Wire into `train_round()`

**2a. Initialize accumulators** before the chunk loop, next to `n_tokens = 0` etc. (`mesh_slm.py:2192-2195`):

```python
stp_embed_samples: list[float] = []
stp_torus_samples: list[float] = []
stp_diag_on = _stp_diagnostic_enabled()
```

**2b. Sample once per chunk**, immediately after Step 1 finishes populating `ids` and updating embeddings (i.e., right after line 2244, before the Step 2 bigram loop begins). Sampling *after* the embedding update means `h_r`/`h_t` reflect the same post-update state the paper's teacher-forced `h_t` represents:

```python
if stp_diag_on:
    triplet = _sample_stp_triplet(len(ids))
    if triplet is not None:
        si, ri, ti = triplet
        try:
            h_s = _embed7_for_token(cn, ids[si])
            h_r = _embed7_for_token(cn, ids[ri])
            h_t = _embed7_for_token(cn, ids[ti])
            if h_s is not None and h_r is not None and h_t is not None:
                gap = _stp_cos_gap(h_s, h_r, h_t)
                if gap is not None:
                    stp_embed_samples.append(gap)
        except Exception:
            pass  # diagnostic must never fail a training round
        try:
            cell_s = _torus_cell_for_token(cn, ids[si])
            cell_r = _torus_cell_for_token(cn, ids[ri])
            cell_t = _torus_cell_for_token(cn, ids[ti])
            if cell_s and cell_r and cell_t:
                gap_t = _stp_cos_gap(
                    _torus_point_r4(cell_s), _torus_point_r4(cell_r), _torus_point_r4(cell_t)
                )
                if gap_t is not None:
                    stp_torus_samples.append(gap_t)
        except Exception:
            pass
```

Both `try/except` blocks are unconditional pass-throughs by design — this mirrors the existing ACRE step's discipline at `mesh_slm.py:2286-2292` ("ACRE must never fail a training round").

**2c. Aggregate + persist**, next to the existing `_meta_set` calls (`mesh_slm.py:2277-2284`):

```python
avg_stp_embed = (sum(stp_embed_samples) / len(stp_embed_samples)) if stp_embed_samples else None
avg_stp_torus = (sum(stp_torus_samples) / len(stp_torus_samples)) if stp_torus_samples else None

if avg_stp_embed is not None:
    _meta_set(cn, "last_stp_embed_gap", round(avg_stp_embed, 6))
    hist = _meta_get(cn, "stp_embed_gap_history", []) or []
    hist.append(round(avg_stp_embed, 6))
    _meta_set(cn, "stp_embed_gap_history", hist[-_STP_HISTORY_CAP:])

if avg_stp_torus is not None:
    _meta_set(cn, "last_stp_torus_gap", round(avg_stp_torus, 6))
    hist_t = _meta_get(cn, "stp_torus_gap_history", []) or []
    hist_t.append(round(avg_stp_torus, 6))
    _meta_set(cn, "stp_torus_gap_history", hist_t[-_STP_HISTORY_CAP:])

# Needed so P1 (loss plateaus while STP gap keeps falling) is checkable at
# all — last_loss today is overwritten every round with no history.
loss_hist = _meta_get(cn, "loss_history", []) or []
loss_hist.append(round(avg_loss, 6))
_meta_set(cn, "loss_history", loss_hist[-_STP_HISTORY_CAP:])
```

The `loss_history` addition is small but necessary: `last_loss` (`mesh_slm.py:2278`) is a single overwritten scalar today, and you cannot detect a *plateau* from one number. This is the only change to existing behavior in Phase 1, and it's additive (a new meta key, nothing removed or altered).

**2d. Add to the returned summary dict** (`mesh_slm.py:2333-2347`), next to `phase_weight`/`wavefunction_overlap`/`mesh_field_8d`:

```python
"stp_embed_gap": round(avg_stp_embed, 6) if avg_stp_embed is not None else None,
"stp_torus_gap": round(avg_stp_torus, 6) if avg_stp_torus is not None else None,
```

---

## 3. Trend-check utility (analysis only, not wired into the training loop)

A small standalone function, e.g. placed near `state_summary()`:

```python
def stp_diagnostic_trend(cn: sqlite3.Connection | None = None, window: int = 20) -> dict:
    """Compare trailing-window slope of loss_history vs. stp_*_gap_history to
    check the paper's P1 signature: loss flat/plateaued while the STP gap
    keeps falling. Read-only; used for manual/notebook validation, not
    during training."""
    ...
    # slope := (last window mean) - (previous window mean), for each series
    # P1 signature: |loss_slope| below a small epsilon AND stp_slope < 0
```

This is intentionally **not** called from `train_round()`. It's the tool used in Step 4 (validation) below, run manually or from a notebook/daily-digest extension once enough rounds have accumulated.

---

## 4. Tests — `tests/test_mesh_slm.py`

Following the existing `isolated_slm_db` fixture and the precedent of `test_train_round_reports_ueqgm_fields` (`tests/test_mesh_slm.py:216-229`):

1. **`test_train_round_reports_stp_diagnostic_fields`** — after `train_round()`, summary contains `stp_embed_gap`/`stp_torus_gap` keys; if not `None`, each is a float in `[0, 2]` (cosine-distance range).
2. **`test_stp_cos_gap_collinear_is_zero`** — three collinear equally-spaced points → gap `≈ 0`.
3. **`test_stp_cos_gap_orthogonal_is_one`** — construct `a, b, c` so `(c−b) ⊥ (b−a)` → gap `≈ 1.0`.
4. **`test_stp_cos_gap_degenerate_returns_none`** — `a == b` → `None`, not an exception or a fabricated `0`/`1`.
5. **`test_torus_point_r4_wraps`** — `_torus_point_r4((0, 0))` and `_torus_point_r4((_TORUS_N - 1, 0))` are close in Euclidean `ℝ⁴` distance (proves wrap-awareness); contrast with naive `(i, j)` subtraction, which would show a jump of `_TORUS_N - 1`.
6. **`test_stp_diagnostic_disabled_by_flag`** — `monkeypatch.setenv("QUIPU_STP_DIAGNOSTIC", "0")`; `train_round()` still returns `status == "ok"`, and `stp_embed_gap`/`stp_torus_gap` are `None`/absent. Confirms "fails toward legacy."
7. **`test_stp_diagnostic_history_capped`** — run `train_round()` enough times (resetting `slm._LAST_TRAIN_TS = 0.0` between calls, per the existing pattern at `tests/test_mesh_slm.py:82`) to exceed `_STP_HISTORY_CAP`; confirm `stp_torus_gap_history` length stays capped.

---

## 5. Rollout / empirical validation sequence

1. **Land Phase 1** (sections 1–4 above) behind `QUIPU_STP_DIAGNOSTIC`, default **on** — it's read-only and cheap (2 extra indexed `SELECT`s per chunk, not per token; the paper itself predicts negligible overhead from computing cosine similarity, §3.2).
2. **Accumulate data**: run training rounds against a real or seeded corpus so `stp_embed_gap_history`, `stp_torus_gap_history`, and `loss_history` build up (dozens of rounds minimum — early rounds are expected to be noisy: newly-seeded tokens start at `0.05·mesh + noise`, per `mesh_slm.py:2015`, so their triplets will show high, unconverged gaps almost by construction; don't read early noise as disproof).
3. **Check P1** with `stp_diagnostic_trend()`: does `loss_history` flatten (slope near zero) while `stp_torus_gap_history` and/or `stp_embed_gap_history` keep decreasing over the same trailing window? Record the result — confirming or falsifying — in `LEARNINGS.md`, matching the repo's verification culture regardless of outcome.
4. **Only if confirmed**, open a **separate** Phase 2 PR: fold `stp_torus_gap` into `η`/`η_q` the same way `phase_weight`/`overlap` already modulate them (`mesh_slm.py:2185-2186`), behind a **new**, independently-defaulted-off flag (higher bar than a passive diagnostic, since this one changes learning dynamics):

   ```python
   def _stp_eta_coupling_enabled() -> bool:
       """Off by default — flip only after stp_diagnostic_trend() confirms P1
       on real corpus data. See docs/STP_DIAGNOSTIC_PLAN.md §5."""
       return str(os.environ.get("QUIPU_STP_ETA_COUPLING", "0")).strip().lower() \
           in ("1", "true", "yes", "on")

   stp_gain = 1.0
   if _stp_eta_coupling_enabled() and avg_stp_torus is not None:
       # Widen effective LR when the tube gap is high (trajectory is noisy
       # relative to the geodesic) — same signal/noise framing as the paper.
       stp_gain = 0.85 + 0.30 * min(1.0, avg_stp_torus)
   eta = _LR_BASE * (1.0 - progress) * phase_weight * (0.70 + 0.30 * overlap) * stp_gain
   ```

   Phase 2 needs its own test asserting `eta` actually moves with `stp_torus_gap`, plus a before/after `avg_loss` comparison over a fixed corpus fixture as a regression guard. Not scoped further here — deliberately deferred until Phase 1 data justifies it.

---

## 6. Explicit risks / edge cases

- **Chunks shorter than 3 tokens.** Already partly handled — `len(tokens) < 2: continue` at `mesh_slm.py:2212` skips 0/1-token chunks; `_sample_stp_triplet` returns `None` for the remaining 2-token case, and the diagnostic is skipped for that chunk while quipu-edge training proceeds unaffected.
- **Cold-start noise.** Freshly seeded tokens' embeddings start near `0.05 · mesh_state` (`mesh_slm.py:2015`) — early-round triplets touching new vocabulary will show inflated gaps that are an artifact of seeding, not a geodesic violation. The trend check (Step 5.3) should window over a stabilized period, not round 1.
- **Vocab/embed row existence.** Both lookups require the token already present in `mesh_slm_vocab`/`mesh_slm_embed`; guaranteed within a chunk because Step 1 (`_upsert_token`, lines 2223-2226) runs before the diagnostic samples the same `ids` list.
- **Performance.** At most 6 extra indexed point-lookups per *chunk* (not per token): 3× `mesh_slm_embed` + 3× `mesh_slm_vocab`. Negligible next to the existing per-token `UPDATE` in Step 1.
- **Concurrency.** Runs inside the existing `with _conn() as cn:` block already held for the whole round (`mesh_slm.py:2197`) — no new locking required.
- **Test determinism.** `_sample_stp_triplet` uses the same `random` module already used for `random.shuffle(chunks)` (`mesh_slm.py:2206`). Tests that need deterministic indices should `random.seed(...)`; otherwise assert on type/range, not exact values, matching how the rest of the suite treats randomized internals.

---

## 7. Summary of files touched (Phase 1)

| File | Change |
|---|---|
| `src/quipu/mesh_slm.py` | Add §1 helpers near `_proximity` (~line 2044); wire §2a–2d into `train_round()`; add `stp_diagnostic_trend()` near `state_summary()` |
| `tests/test_mesh_slm.py` | Add the 7 tests in §4 |
| `LEARNINGS.md` | One entry once §5.3's trend check produces a result (confirm or falsify) |

No change to `julia/*`, no new dependencies, no schema migration (uses the existing generic `mesh_slm_meta` key/value store — consistent with the "diagnostics and pair state belong in the existing `brain_kv` JSON state; schema migrations for observability data were never worth it" learning in `LEARNINGS.md:23`).
