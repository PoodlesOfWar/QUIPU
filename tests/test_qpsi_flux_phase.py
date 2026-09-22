"""qpsi.flux_phase — the expansion phase read from the ingest flux, not a clock."""
from __future__ import annotations

import pytest

from src.quipu.qpsi import flux_phase as fp

NOW = 1_800_000_000.0


def _h(*entries):
    return [{"ts": NOW - age, "total_condensed": docs} for age, docs in entries]


def test_field_on_inside_window_off_outside():
    on = fp.ingest_flux(_h((100, 12)), now=NOW, window_s=300)
    off = fp.ingest_flux(_h((4000, 500)), now=NOW, window_s=300)
    assert on.on and on.parity == fp.BROADEN and on.phase == "broaden" and on.docs == 12
    assert not off.on and off.parity == fp.DEEPEN and off.phase == "deepen" and off.docs == 0
    assert off.last_ts == NOW - 4000              # history remembered even when off


def test_window_boundary_is_half_open_and_skew_tolerant():
    edge = fp.ingest_flux(_h((300, 5)), now=NOW, window_s=300)        # exactly window_s ago → out
    assert not edge.on
    inside = fp.ingest_flux(_h((299, 5)), now=NOW, window_s=300)
    assert inside.on
    ahead = fp.ingest_flux([{"ts": NOW + 60, "total_condensed": 5}], now=NOW, window_s=300)
    assert ahead.on                              # a skewed writer does not switch the phase off


def test_min_docs_threshold_and_iso_timestamps():
    iso = [{"ts": "2027-01-15T10:00:00+00:00", "total_condensed": 1},
           {"ts": "2027-01-15T10:01:00Z", "total_condensed": 1}]
    from datetime import datetime, timezone
    now = datetime(2027, 1, 15, 10, 2, tzinfo=timezone.utc).timestamp()
    r = fp.ingest_flux(iso, now=now, window_s=300, min_docs_on=2)
    assert r.docs == 2 and r.runs == 2 and r.on
    assert not fp.ingest_flux(iso, now=now, window_s=300, min_docs_on=3).on


def test_malformed_history_is_skipped_not_raised():
    junk = ["x", None, 3, {"ts": "not a date", "total_condensed": 9}, {"total_condensed": 9},
            {"ts": NOW - 10, "total_condensed": "seven"}, {"ts": NOW - 10, "total_condensed": 4}]
    r = fp.ingest_flux(junk, now=NOW, window_s=300)
    assert r.docs == 4 and r.runs == 2 and r.on
    assert not fp.ingest_flux(None, now=NOW).on
    assert not fp.ingest_flux([], now=NOW).on


def test_env_window_and_min_docs(monkeypatch):
    monkeypatch.setenv(fp.WINDOW_ENV, "42")
    monkeypatch.setenv(fp.MIN_DOCS_ENV, "3")
    assert fp.window_seconds() == 42.0 and fp.min_docs() == 3
    monkeypatch.setenv(fp.WINDOW_ENV, "-5")
    monkeypatch.setenv(fp.MIN_DOCS_ENV, "zero")
    assert fp.window_seconds() == fp.DEFAULT_WINDOW_S and fp.min_docs() == fp.DEFAULT_MIN_DOCS


def test_wrapper_reads_flux_and_falls_back_to_original(monkeypatch):
    calls = []

    def original(observer, t=None):
        calls.append((observer, t))
        return -1

    monkeypatch.setattr(fp, "read_flux", lambda now=None, **kw: fp.ingest_flux(_h((10, 2)), now=NOW))
    wrapped = fp.wrap_bit_flip_parity(original)
    assert wrapped(0.9, NOW) == fp.BROADEN and calls == []          # the drive is ignored, the field decides
    assert wrapped.__wrapped__ is original and wrapped.__name__ == "bit_flip_parity"

    def boom(now=None, **kw):
        raise RuntimeError("brain_kv unreachable")

    monkeypatch.setattr(fp, "read_flux", boom)
    assert wrapped(0.9, NOW) == -1 and calls == [(0.9, NOW)]        # fell back to the cosine


def test_read_flux_uses_brain_kv_feed(monkeypatch):
    import src.quipu.brain_kv as brain_kv
    monkeypatch.setattr(brain_kv, "kv_get_json", lambda key, default=None: _h((5, 3)) if key == fp.HISTORY_KEY else default)
    r = fp.read_flux(now=NOW, window_s=300)
    assert r.on and r.docs == 3
    monkeypatch.setattr(brain_kv, "kv_get_json", lambda key, default=None: {"not": "a list"})
    assert not fp.read_flux(now=NOW).on
