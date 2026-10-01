"""Monte Carlo Lagrangian λ tokens over the Essay's planes (src/quipu/shadow_tokens.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.quipu import shadow_tokens as ST

ESSAY = """# A Test Canvas

## I. The Feather
Feather vane wind lift sky. The feather meets the wind. Lift rises from the vane.

```
  [ diagram ] ──► ignored
```

## II. The Scale
Scale stone ground cave weight. The scale holds the ground. Weight sinks to stone.

## III. The Fall
Fall sideways history time path. The fall moves sideways. History bends time.

## IV. The Keeper
Keeper boundary grid lock order. The keeper locks the grid. Order holds the boundary.

## V. The Brothers
Brothers friction mirror twin pair. The twin faces the mirror. Friction splits the pair.

## VI. The Axis
Axis union wing serpent bond. The serpent grows a wing. The bond is the axis.
"""

PLANE_WORDS = {
    0: "feather vane wind lift sky rises meets",
    1: "scale stone ground cave weight holds sinks",
    2: "fall sideways history time path moves bends",
    3: "keeper boundary grid lock order locks holds",
    4: "brothers friction mirror twin pair faces splits",
    5: "axis union wing serpent bond grows",
}


def _mesh(seed: int = 0, sigma: float = 0.02) -> ST.Mesh:
    """A 7-D mesh where each plane's words sit on their own axis, filler tokens near the
    planes, and two marketplace anchors: 'hydraulic' near plane I, 'bearing' on axis 6,
    which no plane points at (so its floor costs the Essay something: λ > 0)."""
    rng = np.random.default_rng(seed)
    lookup = {}
    for ax, words in PLANE_WORDS.items():
        for w in words.split():
            v = np.full(7, 0.1)
            v[ax] = 0.9
            lookup.setdefault(w, v + rng.normal(0, 0.01, 7))
    for w in ("the", "a", "is", "from", "to"):
        lookup[w] = np.full(7, 0.3)
    toks, E = [], []
    for i in range(60):                       # 10 filler tokens per plane axis
        ax = i % 6
        v = np.full(7, 0.1)
        v[ax] = 0.8
        toks.append(f"tok{ax}_{i}")
        E.append(v + rng.normal(0, 0.03, 7))
    for i in range(12):                       # bearing-like tokens on axis 6
        v = np.full(7, 0.1)
        v[6] = 0.9
        toks.append(f"brg_{i}")
        E.append(v + rng.normal(0, 0.03, 7))
    for t, e in zip(toks, E):
        lookup[t] = e
    hyd = np.full(7, 0.1)
    hyd[0] = 0.85
    brg = np.full(7, 0.1)
    brg[6] = 0.9
    lookup["hydraulic"], lookup["bearing"] = hyd, brg
    E = np.array(E)
    return ST.Mesh(toks, np.ones(len(toks)), E, E.mean(axis=0), lookup, sigma)


@pytest.fixture
def essay(tmp_path):
    p = tmp_path / "essay.md"
    p.write_text(ESSAY, encoding="utf-8")
    return str(p)


def _random_instance(rng, T=40, P=4, C=3, K=4, floor=3):
    S = rng.normal(size=(T, P))
    member = rng.integers(-1, C, size=T)
    cap = np.bincount(member[member >= 0], minlength=C)
    floor = min(floor, (P * K) // C)          # as run() sizes them: the floors fit the budget
    return ST.Instance(S, member, np.minimum(floor, cap).astype(float), K)


def _feasible(inst, x):
    T, P = inst.S.shape
    assert np.allclose(x.sum(axis=0), inst.K)
    assert (x.sum(axis=1) <= 1 + 1e-9).all()
    counts = ST._cluster_counts(x, inst.member, len(inst.q))
    assert (counts >= inst.q - 1e-9).all()


def test_the_essay_splits_into_its_six_movements(essay):
    planes, info = ST.essay_planes(ST.essay_path(essay))
    assert [p.n for p in planes] == ["I", "II", "III", "IV", "V", "VI"]
    assert planes[0].title == "The Feather" and "diagram" not in planes[0].text
    assert all(len(p.sentences) >= 2 for p in planes)
    assert len(info["sha256"]) == 64 and info["title"] == "A Test Canvas"


def test_dual_bound_is_above_the_milp_and_equals_the_lp_relaxation():
    pytest.importorskip("scipy")
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix
    rng = np.random.default_rng(7)
    for _ in range(5):
        inst = _random_instance(rng)
        x_star, z_star = ST.exact_milp(inst)
        _feasible(inst, x_star)
        dual = ST.lagrangian_dual(inst, iters=400, lower=z_star)
        assert dual["bound"] >= z_star - 1e-6                       # weak duality
        # LP relaxation of the same MILP: the split problem has the integrality property,
        # so the Lagrangian dual and the LP bound coincide (Geoffrion 1974).
        T, P = inst.S.shape
        C = len(inst.q)
        A_eq = lil_matrix((P, T * P))
        for p in range(P):
            A_eq[p, [t * P + p for t in range(T)]] = 1
        A_ub = lil_matrix((T + C, T * P))
        for t in range(T):
            A_ub[t, t * P:(t + 1) * P] = 1
            if inst.member[t] >= 0:
                A_ub[T + inst.member[t], t * P:(t + 1) * P] = -1
        b_ub = np.concatenate([np.ones(T), -inst.q])
        lp = linprog(-inst.S.ravel(), A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(),
                     b_eq=np.full(P, inst.K), bounds=(0, 1), method="highs")
        assert dual["bound"] == pytest.approx(-lp.fun, rel=2e-3, abs=2e-3)


def test_repair_always_returns_a_feasible_point():
    rng = np.random.default_rng(3)
    for _ in range(25):
        inst = _random_instance(rng, T=int(rng.integers(30, 60)), P=int(rng.integers(2, 6)),
                                C=int(rng.integers(1, 4)), K=int(rng.integers(2, 5)), floor=2)
        x, z = ST._repair(inst, ST._subproblem(inst.S, inst.K))
        _feasible(inst, x)
        assert z == pytest.approx(float((inst.S * x).sum()))


def test_a_binding_floor_has_a_positive_price_and_a_slack_one_has_none():
    rng = np.random.default_rng(1)
    T, P, K = 30, 3, 4
    S = rng.uniform(0.5, 1.0, size=(T, P))
    member = np.full(T, -1)
    member[:5] = 0                       # cluster 0: poorly aligned with every plane
    S[:5] = -0.5
    member[5:10] = 1                     # cluster 1: as good as anything
    inst = ST.Instance(S, member, np.array([3.0, 0.0]), K)
    dual = ST.lagrangian_dual(inst, iters=400, lower=ST._greedy_value(inst))
    assert dual["lam"][0] > 0.5          # the Essay gives up ~1.0-1.5 per forced token
    assert dual["lam"][1] == pytest.approx(0.0, abs=1e-6)


def test_monte_carlo_run_prices_equity_per_cluster(essay):
    mesh = _mesh()
    clusters = [{"id": "hydraulic", "commodity": "steel", "size": 40},
                {"id": "bearing", "commodity": "steel", "size": 25},
                {"id": "nothing-here", "commodity": "unobtainium", "size": 3}]
    # the synthetic planes are already orthogonal axes: no shared component to contrast away
    out = ST.run(scenarios=12, k=3, equity=0.5, seed=5, clusters=clusters, essay=essay,
                 mesh=mesh, persist=False, threshold=0.3, contrast=False)
    assert out["ok"], out
    lt = {r["cluster"]: r for r in out["lambda_tokens"]}
    assert set(lt) == {"hydraulic", "bearing"} and out["unanchored_clusters"] == ["nothing-here"]
    assert lt["bearing"]["token"] == "λ:bearing"
    # bearing sits on the axis no plane points at: its floor binds, and costs more than hydraulic's
    assert lt["bearing"]["binding_rate"] == 1.0
    assert lt["bearing"]["p50"] > lt["hydraulic"]["p50"]
    assert lt["bearing"]["expected_share"] >= lt["bearing"]["floor"] - 1e-9
    assert lt["bearing"]["p10"] <= lt["bearing"]["p50"] <= lt["bearing"]["p90"]
    # shadow tokens: the floor pulls bearing tokens into the Essay's voice (free emission never
    # picks them); token-once conflicts pull in a few plane tokens too, with less weight
    pulled = [s for s in out["shadow_tokens"] if s["kind"] == "pulled_in"]
    brg = [s for s in pulled if s["cluster"] == "bearing"]
    assert brg and pulled[0]["cluster"] == "bearing"
    assert not [s for s in out["shadow_tokens"] if s["cluster"] == "bearing" and s["kind"] == "held_back"]
    held = [s for s in out["shadow_tokens"] if s["kind"] == "held_back"]
    assert held and all(s["cluster"] != "bearing" for s in held)
    # the emission voices each movement; plane I speaks with plane-I tokens
    em = {e["plane"]: e for e in out["emission"]}
    assert set(em) == {"I", "II", "III", "IV", "V", "VI"}
    assert any(t.startswith("tok0_") for t in em["I"]["tokens"])
    assert out["bounds"]["gap_mean"] >= -1e-6
    assert sum(p["attention"] for p in out["projections"]) == pytest.approx(1.0, abs=1e-3)
    assert "not currency" in out["note"]


def test_contrast_separates_planes_that_share_the_essays_common_direction(essay):
    """On the host brain ~95 % of every movement's direction is the Essay's shared part.
    Here each plane word also carries a large common component; contrast removes it."""
    mesh = _mesh()
    common = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    for ax, words in PLANE_WORDS.items():
        for w in words.split():
            mesh.lookup[w] = mesh.lookup[w] + common
    planes, _ = ST.essay_planes(ST.essay_path(essay))
    flat = ST._max_offdiag(ST.planes_dirs(planes, mesh, contrast=False))
    sharp = ST._max_offdiag(ST.planes_dirs(planes, mesh, contrast=True))
    assert flat > 0.9 and sharp < 0.2
    out = ST.run(scenarios=6, k=3, seed=4, clusters=[], essay=essay, mesh=mesh, persist=False)
    assert out["problem"]["planes"] == "contrasted" and out["problem"]["plane_cosine_max"] < 0.2
    em = {e["plane"]: e["tokens"] for e in out["emission"]}
    assert any(t.startswith("tok0_") for t in em["I"]) and any(t.startswith("tok4_") for t in em["V"])


def test_without_clusters_the_essay_is_voiced_with_no_prices(essay):
    out = ST.run(scenarios=4, k=3, seed=2, clusters=[], essay=essay, mesh=_mesh(), persist=False)
    assert out["ok"] and out["lambda_tokens"] == [] and out["projections"] == []
    assert len(out["emission"]) == 6


def test_clusters_are_sanitised_and_capped():
    raw = [{"id": "hy<dr>aulic';--", "commodity": "steel", "size": "12"}, {"id": ""}, "junk",
           {"cluster_id": "Hydraulic", "commodity": "x"}] + [{"id": f"c{i}"} for i in range(100)]
    out = ST.sanitize_clusters(raw)
    assert out[0]["id"] == "hydraulic--" or out[0]["id"].startswith("hydr")
    assert all("<" not in c["id"] and ";" not in c["id"] for c in out)
    assert len(out) <= ST.MAX_CLUSTERS
    assert ST.sanitize_clusters("not a list") == []


def test_missing_essay_or_thin_mesh_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv(ST.ESSAY_ENV, str(tmp_path / "missing.md"))
    monkeypatch.setattr(ST, "ESSAY_CANDIDATES", ())
    assert ST.run(scenarios=2, mesh=_mesh(), clusters=[], persist=False)["ok"] is False


# ---------------------------------------------------------------------------
# Through the observer: the marketplace sets clusters (signed), anyone reads GET /lambda
# ---------------------------------------------------------------------------

def test_clusters_need_a_signed_marketplace_source_and_lambda_is_readable(monkeypatch):
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from src.quipu import brain_kv
    from src.quipu import observer_service as osvc
    from src.quipu.qpsi import edge_admission as E

    kv = {}
    monkeypatch.setattr(brain_kv, "kv_set_json", lambda k, v: kv.__setitem__(k, json.loads(json.dumps(v))))
    monkeypatch.setattr(brain_kv, "kv_get_json", lambda k, d=None: kv.get(k, d))
    monkeypatch.setenv(E.AUTH_ENV, "enforce")
    monkeypatch.setenv(E.BUDGET_ENV, "enforce")
    key = "22" * 32
    t0 = 1_790_000_000.0
    grant = {"grant_ref": "T", "total_tokens_per_hour": 100,
             "sources": {"supply-chain-brain": {"max_tokens_per_hour": 50},
                         "perceptopoly": {"max_tokens_per_hour": 50}}}
    monkeypatch.setattr(E, "_EDGE", E.Edge(clock=lambda: t0, grant=grant, kv=lambda k, v: None,
                                           keys={"supply-chain-brain": key, "perceptopoly": key}))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), osvc.ObserverHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(src, signed=True):
        raw = json.dumps({"source": src, "clusters": [{"id": "hydraulic", "commodity": "steel", "size": 4}]}).encode()
        hdr = E.sign(key, "POST", "/lambda/clusters", raw, src, ts=t0) if signed else {}
        req = urllib.request.Request(url + "/lambda/clusters", data=raw, method="POST",
                                     headers={"Content-Type": "application/json", **hdr})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())
    try:
        assert post("supply-chain-brain", signed=False)[0] == 401
        assert post("perceptopoly")[0] == 403                     # signed, but not the marketplace
        assert ST.KV_CLUSTERS not in kv
        code, body = post("supply-chain-brain")
        assert code == 200 and body["clusters"] == 1
        assert kv[ST.KV_CLUSTERS]["source"] == "supply-chain-brain"
        with urllib.request.urlopen(url + "/lambda", timeout=5) as r:
            assert json.loads(r.read())["ok"] is False             # no run yet, still readable
        kv[ST.KV_LATEST] = {"ok": True, "projections": [{"cluster": "hydraulic"}]}
        with urllib.request.urlopen(url + "/lambda", timeout=5) as r:
            assert json.loads(r.read())["projections"][0]["cluster"] == "hydraulic"
    finally:
        httpd.shutdown()


def test_the_lambda_loop_is_the_operators_switch(monkeypatch):
    from src.quipu import entirety_service as ES
    monkeypatch.delenv("QUIPU_LAMBDA_TOKENS", raising=False)
    assert ES.settings()["lambda_tokens"] is False
    monkeypatch.setenv("QUIPU_LAMBDA_TOKENS", "1")
    monkeypatch.setenv("QUIPU_LAMBDA_SCENARIOS", "3")
    seen = {}
    monkeypatch.setattr(ST, "run", lambda **kw: seen.update(kw) or {"ok": True})
    assert ES.settings()["lambda_tokens"] is True and ES.lambda_tokens() == {"ok": True}
    assert seen["scenarios"] == 3 and seen["k"] == 8 and seen["equity"] == 0.5
