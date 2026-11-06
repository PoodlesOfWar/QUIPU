"""qpsi.open_doors — the λ run's own settings tuned by the pruned residual potential inside
the operator's doors: Rogue absorbs, Gambit fires, Gauche mirrors, Grey transforms."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.quipu import shadow_tokens as ST
from src.quipu.qpsi import open_doors as OD

from test_shadow_tokens import ESSAY, PLANE_WORDS

BASE = 0.05


def gap_mesh(seed: int = 0, sigma: float = 0.08, beyond: int = 0) -> ST.Mesh:
    """Movement I has three tokens pointing straight at it and ten equally good ones at an
    angle, each leaning a different way: its voice is stable at K ≤ 3 and not above.  The
    other movements have six near-equal tokens each.  ``beyond`` tokens past the pool point
    straight at movement II, better than anything in the pool for it."""
    rng = np.random.default_rng(seed)
    lookup = {}
    for ax, words in PLANE_WORDS.items():
        for w in words.split():
            v = np.full(7, BASE)
            v[ax] += 0.95
            lookup.setdefault(w, v)
    toks, E = [], []

    def add(t, v):
        toks.append(t)
        E.append(v)
        lookup[t] = v
    for i in range(3):
        v = np.full(7, BASE)
        v[0] += 1.0
        add(f"strong{i}", v)
    for i in range(10):
        v = np.full(7, BASE)
        v[0] += 0.75
        v[6] += 0.45
        v[1 + i % 5] += 0.25
        add(f"weak{i}", v)
    for ax in range(1, 6):
        for i in range(6):
            v = np.full(7, BASE)
            v[ax] += 0.8
            v[6] += 0.45
            add(f"p{ax}_{i}", v + rng.normal(0, 0.01, 7))
    ext_t, ext_E = [], []
    for i in range(beyond):
        v = np.full(7, BASE)
        v[1] += 1.0
        ext_t.append(f"past{i}")
        ext_E.append(v)
        lookup[f"past{i}"] = v
    return ST.Mesh(toks, np.ones(len(toks)), np.array(E), np.full(7, BASE), lookup, sigma,
                   ext_t, np.array(ext_E).reshape(-1, 7), np.ones(len(ext_t)))


@pytest.fixture
def essay(tmp_path):
    p = tmp_path / "essay.md"
    p.write_text(ESSAY, encoding="utf-8")
    return str(p)


@pytest.fixture
def kv(monkeypatch):
    from src.quipu import brain_kv
    store = {}
    monkeypatch.setattr(brain_kv, "kv_set_json", lambda k, v: store.__setitem__(k, json.loads(json.dumps(v))))
    monkeypatch.setattr(brain_kv, "kv_get_json", lambda k, d=None: json.loads(json.dumps(store[k])) if k in store else d)
    return store


def _sim(essay, k=5, seed=3, mesh=None, scenarios=32):
    sim, err = ST.simulate(scenarios=scenarios, k=k, seed=seed, clusters=[], essay=essay, mesh=mesh or gap_mesh())
    assert err is None, err
    return sim


BASE_SETTINGS = {"k_per_plane": 5, "candidates": 43, "scenarios": 32, "extension": 6, "equity": 0.5}


def _fresh():
    return {"values": {"k": {}}, "charges": {}, "pending": None, "clock": [], "history": [], "mirrors": []}


# ---------------------------------------------------------------------------
# The measure
# ---------------------------------------------------------------------------

def test_the_profile_selects_exactly_k_and_finds_the_gap(essay):
    sim = _sim(essay)
    prof = ST.k_profile(sim, 1, 8)
    assert all(0.0 <= r <= 1.0 for p in prof.values() for r in p.values())     # ties never over-count
    assert prof["I"][3] == 0.0 and prof["I"][5] > 0.2                           # stable at 3, not at 5
    pot = ST.potential(sim)
    planes = {p["plane"]: p for p in pot["planes"]}
    assert planes["I"]["crossing_rate"] == pytest.approx(OD.plane_rate(sim, 0), abs=1e-4)
    assert planes["I"]["imag"] == pytest.approx(planes["I"]["crossing_rate"] * planes["I"]["separation"], abs=1e-5)
    for key in ("real", "imag", "magnitude", "phase_deg", "crossing_rate", "separation", "reading"):
        assert key in pot["total"]                                             # Perceptopoly's form


def test_tokens_past_the_pool_are_pruned_potential_until_the_pool_takes_them(essay):
    full = gap_mesh(beyond=6)
    sim = _sim(essay, mesh=full)
    pot = ST.potential(sim)["total"]
    assert pot["pool_excess"] > 0.01 and pot["pool_would_enter"] == pytest.approx(6.0)
    grown = _sim(essay, mesh=full.pool(len(full.tokens) + 6, 0))
    assert OD.pool_excess(grown) == 0.0


# ---------------------------------------------------------------------------
# The operator's grant
# ---------------------------------------------------------------------------

def test_the_grant_opens_only_measurement_doors_inside_hard_caps(tmp_path):
    g = OD.grant('{"k_per_plane": [0, 99], "candidates": [100, 9000], "scenarios": [16, 64], '
                 '"equity": [0, 1], "threshold": [0, 1], "speed": 3, "grant_ref": "OPS-7"}')
    assert g["doors"] == {"k_per_plane": (1, 64), "candidates": (100, 4096), "scenarios": (16, 64)}
    assert "operator's alone: equity, threshold" in g["error"] and "unknown: speed" in g["error"]
    assert g["grant_ref"] == "OPS-7"
    assert OD.grant("")["doors"] == {} and OD.grant("{not json")["doors"] == {}
    f = tmp_path / "doors.json"
    f.write_text('{"scenarios": [40, 80]}')
    assert OD.grant(str(f))["doors"] == {"scenarios": (40, 80)}


def test_a_closed_door_holds_the_operators_value_and_a_narrowed_one_clamps():
    st = _fresh()
    st["values"].update({"k": {"I": 12}, "candidates": 900, "scenarios": 200})
    open_ = OD.grant('{"k_per_plane": [2, 8], "candidates": [384, 1024]}')
    v = OD.live_values(st, BASE_SETTINGS, open_)
    assert v["k"]["I"] == 8 and v["candidates"] == 900 and v["scenarios"] == 32      # scenarios closed
    closed = OD.live_values(st, BASE_SETTINGS, OD.grant(""))
    assert closed["k"] == {} and closed["k_default"] == 5 and closed["candidates"] == 43


# ---------------------------------------------------------------------------
# Rogue, Gambit, Gauche, Grey
# ---------------------------------------------------------------------------

def test_rogue_absorbs_what_a_door_could_discharge_and_the_charge_fades(essay):
    sim = _sim(essay)
    st = _fresh()
    g = OD.grant('{"k_per_plane": [2, 8]}')
    OD.rogue_absorb(st, sim, {}, g, now=1000.0, tau=600.0)
    ch = st["charges"]["k:I"]
    assert ch["target"] == 3 and ch["x"] == pytest.approx(1.0) and ch["a"] == pytest.approx(1.0)
    st["charges"]["k:I"]["a"] = 0.8
    # same evidence an hour later: relaxed by exp(-Δt/τ) before the new stimulus lands
    st["charges"]["k:I"]["at"] = 1000.0
    st2 = json.loads(json.dumps(st))
    st2["charges"]["k:I"]["x"] = 0.0
    sim_flat = _sim(essay, k={"*": 5, "I": 3})                     # at K=3 nothing to discharge
    OD.rogue_absorb(st2, sim_flat, {}, g, now=1600.0, tau=600.0)
    assert st2["charges"]["k:I"]["a"] == pytest.approx(0.8 * math.exp(-1.0), rel=1e-6)


def test_gambit_fires_the_most_charged_open_door_inside_its_bounds():
    st = _fresh()
    st["charges"] = {"k:I": {"a": 1.0, "target": 3}, "k:II": {"a": 0.4, "target": 8},
                     "scenarios": {"a": 0.9, "target": 48}}
    values = {"k": {"I": 5, "II": 5}, "k_default": 5, "candidates": 43, "scenarios": 32}
    assert OD.gambit_fire(json.loads(json.dumps(st)), values, OD.grant("")) is None         # every door shut
    p = OD.gambit_fire(st, values, OD.grant('{"k_per_plane": [2, 8], "scenarios": [16, 64]}'))
    assert p == {"door": "k:I", "old": 5, "new": 3, "target": 3, "charge": 1.0}
    assert st["charges"]["k:I"]["a"] == pytest.approx(0.0)                                   # spent
    st["charges"]["k:I"] = {"a": 0.6, "target": 1}
    p = OD.gambit_fire(st, values, OD.grant('{"k_per_plane": [4, 8]}'))
    assert p["new"] == 4 and p["target"] == 4                    # the door's edge, never past it


def test_gauche_mirrors_on_the_same_draws_and_grey_keeps_a_discharged_move(essay, kv):
    full = gap_mesh()
    live = _sim(essay, mesh=full)
    values = {"k": {p.n: 5 for p in live.planes}, "k_default": 5, "candidates": len(full.tokens), "scenarios": 32}
    base = dict(BASE_SETTINGS, extension=0)
    prop = {"door": "k:I", "old": 5, "new": 3, "target": 3, "charge": 1.0}
    m = OD.gauche_mirror(live, prop, values, base, full, {"essay": essay, "clusters": []})
    assert m["verdict"] == "discharged" and m["mirror"] == 0.0 and m["diff"] < -2 * m["se"]
    st = _fresh()
    step = OD.grey_transform(st, prop, m, live, now=10.0)
    assert step["kind"] == "transformed" and st["values"]["k"]["I"] == 3
    assert st["pending"]["old"] == 5 and st["pending"]["baseline"] == pytest.approx(m["live"])
    # a move inside the noise: plane I from 3 to 2 changes nothing it can measure
    live3 = _sim(essay, mesh=full, k={"*": 5, "I": 3})
    values["k"]["I"] = 3
    m2 = OD.gauche_mirror(live3, {"door": "k:I", "old": 3, "new": 2, "target": 2, "charge": 0.6},
                          values, base, full, {"essay": essay, "clusters": []})
    assert m2["verdict"] == "inconclusive"
    assert OD.grey_transform(st, {"door": "k:I", "old": 3, "new": 2}, m2, live3, 11.0)["kind"] == "held"
    assert st["values"]["k"]["I"] == 3


def test_grey_returns_to_the_previous_value_when_the_next_run_does_not_confirm(essay):
    st = _fresh()
    st["values"]["k"]["I"] = 5
    st["pending"] = {"door": "k:I", "old": 3, "new": 5, "measure": "crossing_rate:I",
                     "baseline": 0.0, "baseline_se": 0.0, "at": 1.0}
    out = OD.grey_confirm(st, _sim(essay), now=2.0)                  # live at K=5: rate ≈ 0.3
    assert out["kind"] == "reverted" and st["values"]["k"]["I"] == 3 and st["pending"] is None
    st["pending"] = {"door": "k:I", "old": 5, "new": 3, "measure": "crossing_rate:I",
                     "baseline": 0.3, "baseline_se": 0.02, "at": 3.0}
    st["values"]["k"]["I"] = 3
    out = OD.grey_confirm(st, _sim(essay, k={"*": 5, "I": 3}), now=4.0)
    assert out["kind"] == "confirmed" and st["values"]["k"]["I"] == 3


def test_the_pool_door_grows_over_what_it_cut_and_scenarios_follow_an_undecided_mirror(essay, kv):
    full = gap_mesh(beyond=6)
    live = _sim(essay, mesh=full.pool(len(full.tokens), 6))
    st = _fresh()
    g = OD.grant('{"candidates": [10, 200], "scenarios": [16, 64]}')
    OD.rogue_absorb(st, live, {}, g, now=5.0, tau=600.0)
    assert st["charges"]["candidates"]["target"] == len(full.tokens) + 6
    assert st["charges"]["candidates"]["a"] > 0.5
    values = {"k": {p.n: 5 for p in live.planes}, "k_default": 5, "candidates": len(full.tokens), "scenarios": 32}
    prop = OD.gambit_fire(st, values, g)
    assert prop["door"] == "candidates" and prop["new"] > prop["old"]
    m = OD.gauche_mirror(live, prop, values, dict(BASE_SETTINGS, extension=6), full, {"essay": essay, "clusters": []})
    assert m["measure"] == "pool_excess" and m["verdict"] == "discharged"
    assert m["crossing_rate_guard"]["diff"] <= 2 * m["crossing_rate_guard"]["se"] + 1e-9
    # an undecided mirror charges the scenarios door; the move is applied as precision
    st["mirrors"].append({"verdict": "inconclusive"})
    OD.rogue_absorb(st, live, {}, g, now=6.0, tau=600.0)
    assert st["charges"]["scenarios"]["a"] == pytest.approx(1.0) and st["charges"]["scenarios"]["target"] == 48
    st["charges"] = {"scenarios": st["charges"]["scenarios"]}
    p = OD.gambit_fire(st, values, g)
    assert p["door"] == "scenarios" and p["new"] == 48
    assert OD.grey_transform(st, p, None, live, 7.0)["kind"] == "precision" and st["values"]["scenarios"] == 48


# ---------------------------------------------------------------------------
# One λ period, end to end
# ---------------------------------------------------------------------------

def test_with_every_door_shut_the_cycle_measures_and_points_but_moves_nothing(essay, kv):
    out = OD.cycle(now=1000.0, full_mesh=gap_mesh(), essay=essay, clusters=[], seed=3,
                   base=BASE_SETTINGS, grant_raw="")
    d = out["doors"]
    assert d["fired"] is None and d["grant"] == {}
    assert d["charges"]["k:I"]["target"] == 3 and d["charges"]["k:I"]["open"] is False
    assert "k:I" in d["closed_pressure"]
    assert out["problem"]["k_per_plane"]["I"] == 5 and kv[ST.KV_LATEST]["doors"]["fired"] is None


def test_an_open_k_door_finds_the_gap_keeps_it_and_the_next_run_confirms_it(essay, kv):
    grant = '{"k_per_plane": [2, 8]}'
    kinds = []
    for i in range(4):
        out = OD.cycle(now=1000.0 + 1800 * i, full_mesh=gap_mesh(), essay=essay, clusters=[], seed=10 + i,
                       base=BASE_SETTINGS, grant_raw=grant)
        if out["doors"]["step"]:
            kinds.append((out["doors"]["step"]["door"], out["doors"]["step"]["kind"]))
        if out["doors"]["confirmed"]:
            kinds.append((out["doors"]["confirmed"]["door"], out["doors"]["confirmed"]["kind"]))
    st = kv[OD.KV_DOORS]
    assert st["values"]["k"]["I"] == 3
    assert ("k:I", "transformed") in kinds and ("k:I", "confirmed") in kinds
    assert ("k:I", "reverted") not in kinds
    assert kv[ST.KV_LATEST]["problem"]["k_per_plane"]["I"] == 3
    hist = [h for h in st["history"] if h["door"] == "k:I"]
    assert hist[0]["kind"] == "transformed" and hist[0]["old"] == 5 and hist[0]["new"] == 3


def test_the_lambda_loop_runs_one_open_doors_period(monkeypatch):
    from src.quipu import entirety_service as ES
    seen = []
    monkeypatch.setattr(OD, "cycle", lambda **kw: seen.append(kw) or {"ok": True, "doors": {}})
    assert ES.lambda_tokens() == {"ok": True, "doors": {}} and seen
