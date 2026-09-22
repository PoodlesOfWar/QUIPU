"""qpsi.self_organising — default off, wraps three attributes of system_entirety,
closes the loop through the ingest flux, never breaks the step, never routes
without --route."""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone

import pytest

import src.quipu.system_entirety as system_entirety
import src.quipu.brain_kv as brain_kv
import src.quipu.divine_blessing as db
from src.quipu.qpsi import self_organising as so
from src.quipu.qpsi import flux_phase, memristive_axes as ma, learned_prior as lp
from src.quipu.qpsi.governance import GovernanceConfig

REST = {"vision": 0.0302, "touch": 0.0417, "smell": 0.034,
        "body": 0.0318, "brain": 0.0325, "perception": 0.0395}
OTHER = {"self_state": [0.0302, 0.0417, 0.034, 0.0318, 0.0325, 0.0395, 0.0088],
         "other_state": [0.091, 0.091, 0.0988, 0.091, 0.4903, 0.091, 0.3322],
         "resonance": 0.4202, "shared_hz": 42.02}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    db_file = tmp_path / "entirety.sqlite"

    def _open_conn(timeout: float = 60, path=None):
        cn = sqlite3.connect(db_file, timeout=timeout)
        cn.execute("PRAGMA journal_mode=WAL")
        cn.row_factory = sqlite3.Row
        return cn

    monkeypatch.setattr(system_entirety, "_open_conn", _open_conn)
    monkeypatch.setattr(brain_kv, "open_conn", _open_conn)
    monkeypatch.setattr(system_entirety, "_LAST_TS", 0.0)
    monkeypatch.setattr(system_entirety, "_sense_signals", lambda: dict(REST), raising=False)
    try:
        import src.quipu.repository_catalog as repository_catalog
        monkeypatch.setattr(repository_catalog, "get_catalog_summary", lambda: {})
    except Exception:
        pass
    key_file = tmp_path / "attest.key"
    key_file.write_bytes(b"test-key-not-a-secret")
    monkeypatch.setenv(db.KEY_FILE_ENV, str(key_file))
    monkeypatch.setattr(db, "_CONFIG", GovernanceConfig())
    monkeypatch.delenv(so.ENV, raising=False)
    for name in (so.FLUX_ENV, so.PRIOR_ENV, so.SOMN_ENV):
        monkeypatch.delenv(name, raising=False)
    db.enable()
    so.disable()
    yield db_file
    so.disable()
    db.disable()


def _flux_on(age_s: float = 10.0, docs: int = 12):
    ts = datetime.fromtimestamp(time.time() - age_s, tz=timezone.utc).isoformat()
    brain_kv.kv_set_json(flux_phase.HISTORY_KEY, [{"ts": ts, "total_condensed": docs}])


def _flux_off():
    brain_kv.kv_set_json(flux_phase.HISTORY_KEY, [])


# ---------------------------------------------------------------------------
# Default off
# ---------------------------------------------------------------------------

def test_default_off_leaves_system_entirety_untouched(isolated):
    assert not so.is_enabled() and not so.master_switch_on()
    for name in ("bit_flip_parity", "observer_tangent", "oscillating_expansion_step"):
        fn = getattr(system_entirety, name)
        assert fn.__module__ == "src.quipu.system_entirety" and not hasattr(fn, "__wrapped__")


def test_enable_wraps_and_disable_restores_identically(isolated):
    orig = {n: getattr(system_entirety, n) for n in ("bit_flip_parity", "observer_tangent", "oscillating_expansion_step")}
    assert so.enable() and so.enable()                       # idempotent
    assert so.is_enabled()
    for n, f in orig.items():
        assert getattr(system_entirety, n) is not f
        assert getattr(system_entirety, n).__wrapped__ is f
    so.disable(); so.disable()
    for n, f in orig.items():
        assert getattr(system_entirety, n) is f


def test_flags_select_what_is_wrapped(isolated):
    orig_parity = system_entirety.bit_flip_parity
    orig_obs = system_entirety.observer_tangent
    orig_step = system_entirety.oscillating_expansion_step
    so.enable(so.Flags(flux_phase=False, learned_prior=False, somn=True))
    assert system_entirety.bit_flip_parity is orig_parity
    assert system_entirety.observer_tangent is orig_obs
    assert system_entirety.oscillating_expansion_step is not orig_step
    so.disable()
    so.enable(so.Flags(flux_phase=True, learned_prior=False, somn=False))
    assert system_entirety.bit_flip_parity is not orig_parity
    assert system_entirety.oscillating_expansion_step is orig_step        # nothing to hook


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def test_phase_follows_the_field_and_the_somn_records_each_step(isolated):
    brain_kv.kv_set_json(ma.KV_THE_OTHER, OTHER)
    so.enable()
    _flux_on()
    a = system_entirety.oscillating_expansion_step(force=True)
    assert "skipped" not in a
    assert a["expansion_phase"] == "broaden" and a["bit_state"] == 1
    s = a["self_organising"]
    assert s["flux"]["on"] and s["step_phase"] == "broaden"
    assert s["somn"]["phase"] == "broaden" and s["somn"]["top_axis"] == "brain"
    assert s["somn"]["docs_per_source"] and s["prior_advanced"] is False

    _flux_off()
    b = system_entirety.oscillating_expansion_step(force=True)
    assert b["expansion_phase"] == "deepen" and b["bit_state"] == -1 and b["flipped"] is True
    assert b["self_organising"]["somn"]["phase"] == "deepen"
    assert b["flip_count"] == a["flip_count"] + 1                    # parity flips still ungated

    with system_entirety._conn() as cn:
        st = ma.load_state(cn, ma.SomnConfig())
        assert st.steps == 2 and st.potentiation_steps == 1 and st.relaxation_steps == 1
        assert len(ma.rows(cn, 10)) == 2
        flips = cn.execute("SELECT COUNT(*) FROM entirety_flip_log").fetchone()[0]
        assert flips == 2
        # the step's own record is intact and carries the blessing decision, not our summary
        state = system_entirety.get_entirety_state()
        assert state["axes"] == b["axes"] and "divine_blessing" in state and "self_organising" not in state
        assert brain_kv.kv_get_json(ma.KV_ALLOCATION)["phase"] == "deepen"
        assert lp.stored_prior(cn) is None                            # held → the prior did not move
    # the loop closed through the senses: the ingest history moved vision and brain
    # while the field was on (docs/1500 and runs/6 in temporal_spatiality._sense_signals)
    assert a["axes"]["brain"] > b["axes"]["brain"] and a["axes"]["vision"] > b["axes"]["vision"]


def test_without_a_counterpart_the_loop_records_no_field(isolated):
    so.enable()
    _flux_on()
    out = system_entirety.oscillating_expansion_step(force=True)
    s = out["self_organising"]["somn"]
    assert s["field_norm"] == 0.0 and s["docs_per_source"] == {} and "no_counterpart" in s["events"]


def test_after_step_failure_never_breaks_the_step(isolated, monkeypatch):
    so.enable()

    def boom(*a, **k):
        raise RuntimeError("somn exploded")

    monkeypatch.setattr(ma, "step", boom)
    out = system_entirety.oscillating_expansion_step(force=True)
    assert "skipped" not in out and out["flip_count"] is not None
    assert out["self_organising"] == {"error": "somn exploded"}


def test_rate_limited_step_gets_no_summary(isolated):
    so.enable()
    system_entirety.oscillating_expansion_step(force=True)
    second = system_entirety.oscillating_expansion_step()
    assert second.get("skipped") and "self_organising" not in second


# ---------------------------------------------------------------------------
# The operator's pulse
# ---------------------------------------------------------------------------

def test_pulse_without_route_is_a_plan_on_paper(isolated, monkeypatch):
    import src.quipu.corpus_ingest as corpus_ingest
    brain_kv.kv_set_json(ma.KV_THE_OTHER, OTHER)
    so.enable()
    _flux_on()
    system_entirety.oscillating_expansion_step(force=True)
    monkeypatch.setattr(corpus_ingest, "run_ingest", lambda *a, **k: pytest.fail("run_ingest must not be called"))
    p = so.pulse(route=False)
    assert p["routed"] is False and p["plan"]["docs_per_source"]
    assert sum(p["plan"]["docs_per_source"].values()) <= ma.SomnConfig().budget_docs


def test_pulse_with_route_runs_exactly_the_plan(isolated, monkeypatch):
    import src.quipu.corpus_ingest as corpus_ingest
    brain_kv.kv_set_json(ma.KV_THE_OTHER, OTHER)
    so.enable()
    _flux_on()
    system_entirety.oscillating_expansion_step(force=True)
    plan = so.plan()["docs_per_source"]
    calls = []

    def fake_run(sources, *, docs_per_source, **kw):
        calls.append((tuple(sources), docs_per_source, kw.get("refine_every")))
        return {"per_source": {sources[0]: {"ingested": docs_per_source}}, "elapsed_s": 0.1}

    monkeypatch.setattr(corpus_ingest, "run_ingest", fake_run)
    monkeypatch.setattr(system_entirety, "_LAST_TS", 0.0)
    out = so.pulse(route=True)
    assert out["routed"] is True
    assert {c[0][0]: c[1] for c in calls} == plan
    assert all(c[2] == 0 for c in calls)                            # refinement off unless asked
    assert set(c[0][0] for c in calls) <= set(corpus_ingest.SOURCES)
    assert "flip_count" in out["step"]


def test_status_reports_flags_flux_state_and_prior(isolated):
    so.enable()
    st = so.status()
    assert st["enabled"] and st["flags"]["somn"] and "g" in st["somn"] and st["prior"]["realised_seen"] == 0
    assert st["flux"]["phase"] in ("broaden", "deepen") and "budget_docs" in st["config"]


def test_a_second_module_copy_adopts_the_existing_wrapping_instead_of_stacking(isolated):
    """``python -m …self_organising`` executes the file twice (package copy + __main__).
    The second copy must not wrap on top of the first: one after_step per step."""
    import importlib.util
    import pathlib
    so.enable()
    spec = importlib.util.spec_from_file_location(
        "src.quipu.qpsi._self_organising_copy", pathlib.Path(so.__file__))
    copy = importlib.util.module_from_spec(spec)
    copy.__package__ = "src.quipu.qpsi"
    import sys
    sys.modules[spec.name] = copy                       # dataclasses resolve the module by name
    try:
        spec.loader.exec_module(copy)
        assert copy.enable() and copy.is_enabled()
    finally:
        sys.modules.pop(spec.name, None)
    step = system_entirety.oscillating_expansion_step
    assert getattr(step, so.MARK, False) and not hasattr(step.__wrapped__, so.MARK)   # single layer
    assert step.__wrapped__.__module__ == "src.quipu.system_entirety"
    _flux_on()
    out = system_entirety.oscillating_expansion_step(force=True)
    with system_entirety._conn() as cn:
        assert ma.load_state(cn, ma.SomnConfig()).steps == 1                         # one SOMN step, not two
    copy.disable()
