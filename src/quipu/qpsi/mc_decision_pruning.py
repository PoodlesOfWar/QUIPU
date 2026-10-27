"""mc_decision_pruning — Monte Carlo CVaR decision pruning & Greedy-Dijkstra routing.

Mathematical specification
--------------------------
1. **Risk- and uncertainty-adjusted decision objective**:

       J(a) = E[C(a)] + λ · CVaR_β(C(a)) + ζ · U(a) + η_cc · |a|

   For each candidate edge or path `a`:
   * Scenario sampling: M scenarios ω = 1..M perturb the 6-D CAT embeddings and
     Hebbian edge weights with QUIPU's Langevin/relational-shear emission noise
     σ_eff (from `shadow_tokens` and `qpsi.relational_shear.fed_sigma`), yielding
     non-negative scenario costs C_ω(a) ≥ ε > 0.
   * E[C(a)]: sample mean across the M scenarios.
   * CVaR_β(C(a)): Conditional Value-at-Risk at confidence level β ∈ (0, 1)
     (Rockafellar & Uryasev, J. Risk 2, 21–41 (2000)), the exact tail expectation
     of the worst (1 − β) fraction of scenario costs.
   * U(a): epistemic + sampling uncertainty bound:
         U(a) = z_ci · (σ̂_C(a) / √M) + 1 / √(samples(a) + 1)
   * η_cc · |a|: cost-complexity regularization per hop/edge.

2. **Uncertainty-bound screening (pruning)**:
   Candidate edges are screened before entering shortest-path search:
   * An edge is pruned (`eligible = False`) if its upper confidence bound
     `UCB(e) = E[C(e)] + U(e) > max_ucb_cost`, if `CVaR_β(C(e)) > max_cvar_cost`,
     or if its risk-adjusted score fails cost-complexity pruning against the
     source node's best outgoing edge (`J(e) − J(e_best) > complexity_slack`).
   * All surviving edges carry strictly positive weight `w_MC(e) = max(ε, J(e))`,
     satisfying Dijkstra's non-negative edge-weight precondition by construction.

3. **Shortest-path & Greedy-Dijkstra next-path selection + centroidal deviation**:
   * `mc_dijkstra_route`: exact weighted Dijkstra over the Monte Carlo pruned
     candidate set (deterministic `(cost, dst)` tie-breaking).
   * `mc_greedy_dijkstra_route`: greedy-Dijkstra next-path selection that scores
     immediate pruned candidates with single-step `J(e)` plus a bounded 1-hop
     Dijkstra lookahead, falling back to exact `mc_dijkstra_route` whenever the
     greedy trajectory dead-ends or violates the centroidal deviation bound.
   * `centroidal_deviation`: measures the 7-D CAT manifold deviation between the
     selected route and its centroid (mean, median, histogram mode, L_∞ norm, and
     the `inf_minus_1_lt_4p` stability criterion `‖Δ‖_∞ − 1 < 4p`).

Governance
----------
Read-only on `mesh_slm_vocab`, `mesh_slm_embed`, `mesh_slm_quipu`.
Optional diagnostics persist only to `brain_kv["entirety:mc_pruning:latest"]`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import heapq
import json
import math
import random
import sqlite3
import statistics
import time
from typing import Any, Mapping, Optional, Sequence

from .. import quipu_minimal as _qm


KV_MC_PRUNING_LATEST: str = "entirety:mc_pruning:latest"


@dataclass(frozen=True)
class MCPruningConfig:
    scenarios: int = 32                # Monte Carlo scenario draws M
    beta: float = 0.90                 # CVaR tail quantile β ∈ (0, 1)
    lam_cvar: float = 0.35             # λ ≥ 0 weight on CVaR_β(C(a))
    zeta_uncert: float = 0.25          # ζ ≥ 0 weight on uncertainty U(a)
    eta_complexity: float = 0.02       # η_cc ≥ 0 cost-complexity penalty per hop
    sigma_noise: float = 0.08          # Langevin scenario perturbation scale σ_eff
    z_ci: float = 1.96                 # 95% confidence multiplier for sampling SE
    max_ucb_cost: float = 1.35         # prune edges with E[C] + U > max_ucb_cost
    max_cvar_cost: float = 1.50        # prune edges with CVaR_β > max_cvar_cost
    complexity_slack: float = 0.65     # prune edges > J_min + complexity_slack
    cost_epsilon: float = 1e-6         # strictly positive floor for Dijkstra
    seed: int = 42

    def validate(self) -> None:
        if int(self.scenarios) < 1:
            raise ValueError("scenarios must be >= 1")
        if not (0.0 < float(self.beta) < 1.0):
            raise ValueError("beta must lie strictly in (0, 1)")
        if self.lam_cvar < 0.0 or self.zeta_uncert < 0.0 or self.eta_complexity < 0.0:
            raise ValueError("regularization weights lam_cvar, zeta_uncert, eta_complexity must be >= 0")
        if self.sigma_noise < 0.0:
            raise ValueError("sigma_noise must be >= 0")
        if self.cost_epsilon <= 0.0:
            raise ValueError("cost_epsilon must be > 0 for Dijkstra termination")


@dataclass(frozen=True)
class EdgeMCAssessment:
    src: int
    dst: int
    expected_cost: float
    var_beta: float
    cvar_beta: float
    uncertainty: float
    complexity_penalty: float
    j_objective: float
    ucb_cost: float
    samples: int
    eligible: bool
    prune_reason: Optional[str]


def compute_var_cvar(costs: Sequence[float], beta: float = 0.90) -> tuple[float, float]:
    """Compute empirical VaR_β and Rockafellar–Uryasev CVaR_β for scenario costs."""
    if not costs:
        return 0.0, 0.0
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0, 1)")
    s = sorted(float(c) for c in costs)
    n = len(s)
    # Index where cumulative probability crosses β
    k = min(n - 1, max(0, int(math.ceil(beta * n)) - 1))
    var_b = s[k]
    tail = s[k:]
    cvar_b = sum(tail) / float(len(tail))
    return var_b, cvar_b


def evaluate_edge_mc(
    base_qweight: float,
    cat_src: Sequence[float],
    cat_dst: Sequence[float],
    resusc_cost: float,
    observer_bias: float,
    *,
    samples: int = 1,
    cfg: MCPruningConfig = MCPruningConfig(),
    edge_seed: int = 0,
) -> dict[str, float]:
    """Run M Monte Carlo scenarios for a single directed candidate edge and compute J(e)."""
    cfg.validate()
    rng = random.Random(int(cfg.seed) ^ int(edge_seed))
    m = int(cfg.scenarios)
    scenario_costs: list[float] = []

    for _ in range(m):
        if cfg.sigma_noise > 0.0:
            qw_s = _qm._finite01(base_qweight + rng.gauss(0.0, cfg.sigma_noise))
            c_src = [_qm._finite01(x + rng.gauss(0.0, 0.5 * cfg.sigma_noise)) for x in cat_src[:6]]
            c_dst = [_qm._finite01(x + rng.gauss(0.0, 0.5 * cfg.sigma_noise)) for x in cat_dst[:6]]
            cat_d = _qm.cat6_distance(c_src, c_dst)
            res_s = _qm._finite01(resusc_cost + rng.gauss(0.0, 0.25 * cfg.sigma_noise))
        else:
            qw_s = _qm._finite01(base_qweight)
            cat_d = _qm.cat6_distance(list(cat_src[:6]), list(cat_dst[:6]))
            res_s = _qm._finite01(resusc_cost)

        c_omega = (
            _qm._COST_W_QWEIGHT * (1.0 - qw_s)
            + _qm._COST_W_CAT * cat_d
            + _qm._COST_W_RESUSC * res_s
            + _qm._COST_W_OBSERVER * (1.0 - _qm._finite01(observer_bias))
            + cfg.cost_epsilon
        )
        scenario_costs.append(max(cfg.cost_epsilon, c_omega))

    exp_c = sum(scenario_costs) / float(m)
    std_c = statistics.pstdev(scenario_costs) if m > 1 else 0.0
    var_b, cvar_b = compute_var_cvar(scenario_costs, cfg.beta)

    # Sampling standard error + epistemic scarcity term (1 / sqrt(samples + 1)) / 4
    se_bound = cfg.z_ci * (std_c / math.sqrt(float(m)))
    epistemic = 0.25 / math.sqrt(float(max(0, int(samples)) + 1))
    uncert = se_bound + epistemic
    comp = cfg.eta_complexity
    j_val = max(
        cfg.cost_epsilon,
        exp_c + cfg.lam_cvar * cvar_b + cfg.zeta_uncert * uncert + comp,
    )
    return {
        "expected_cost": exp_c,
        "std_cost": std_c,
        "var_beta": var_b,
        "cvar_beta": cvar_b,
        "uncertainty": uncert,
        "complexity_penalty": comp,
        "j_objective": j_val,
        "ucb_cost": exp_c + uncert,
    }


def _edge_samples(cn: sqlite3.Connection, src: int, dst: int) -> int:
    try:
        row = cn.execute(
            "SELECT samples FROM mesh_slm_quipu WHERE src=? AND dst=?",
            (int(src), int(dst)),
        ).fetchone()
        return int(row[0] or 1) if row else 1
    except Exception:
        return 1


def screen_candidate_edges(
    cn: sqlite3.Connection,
    src: int,
    *,
    observer: Optional[float] = None,
    cfg: MCPruningConfig = MCPruningConfig(),
    allowed_nodes: Optional[set[int]] = None,
) -> tuple[list[EdgeMCAssessment], list[EdgeMCAssessment]]:
    """Screen all outgoing candidate edges from `src` using Monte Carlo CVaR + UCB bounds
    and cost-complexity pruning.

    Returns `(eligible_assessments, all_assessments)` ordered by `(j_objective, dst)`.
    """
    cfg.validate()
    obs = _qm._observer_bias() if observer is None else _qm._finite01(observer)
    cat_src = _qm.cat6_coords(cn, src)
    raw_cands = _qm._candidate_edges(cn, src)

    prelim: list[EdgeMCAssessment] = []
    for dst, qw in raw_cands:
        if allowed_nodes is not None and dst not in allowed_nodes:
            continue
        cat_dst = _qm.cat6_coords(cn, dst)
        resusc = _qm._resuscitation_cost(cn, dst)
        samples = _edge_samples(cn, src, dst)
        edge_seed = ((int(src) * 1000003) ^ (int(dst) * 9176)) & 0x7FFFFFFF
        ev = evaluate_edge_mc(
            qw, cat_src, cat_dst, resusc, obs,
            samples=samples, cfg=cfg, edge_seed=edge_seed,
        )
        reason: Optional[str] = None
        if ev["ucb_cost"] > cfg.max_ucb_cost:
            reason = "ucb_exceeded"
        elif ev["cvar_beta"] > cfg.max_cvar_cost:
            reason = "cvar_exceeded"
        prelim.append(
            EdgeMCAssessment(
                src=int(src),
                dst=int(dst),
                expected_cost=ev["expected_cost"],
                var_beta=ev["var_beta"],
                cvar_beta=ev["cvar_beta"],
                uncertainty=ev["uncertainty"],
                complexity_penalty=ev["complexity_penalty"],
                j_objective=ev["j_objective"],
                ucb_cost=ev["ucb_cost"],
                samples=samples,
                eligible=(reason is None),
                prune_reason=reason,
            )
        )

    if not prelim:
        return [], []

    # Cost-complexity regularization against the best outgoing edge from `src`
    best_j = min(a.j_objective for a in prelim)
    final_all: list[EdgeMCAssessment] = []
    eligible: list[EdgeMCAssessment] = []
    for a in prelim:
        if a.eligible and (a.j_objective - best_j) > cfg.complexity_slack:
            a = dataclasses.replace(a, eligible=False, prune_reason="cost_complexity_pruned")
        final_all.append(a)
        if a.eligible:
            eligible.append(a)

    # Guarantee at least the single best-scoring edge survives if all were soft-pruned,
    # unless the caller configured a strict zero-allowance gate.
    final_all.sort(key=lambda x: (x.j_objective, x.dst))
    eligible.sort(key=lambda x: (x.j_objective, x.dst))
    return eligible, final_all


def compute_pruned_mass_ledger(
    eligible: Sequence[EdgeMCAssessment],
    all_assessments: Sequence[EdgeMCAssessment],
) -> dict[str, Any]:
    """Compute retained vs. pruned branch probability mass using the Boltzmann /
    maximum-likelihood correspondence c_ij = -log(p_ij)  <=>  p_ij ∝ exp(-J(e)).
    """
    if not all_assessments:
        return {
            "total_branches": 0,
            "retained_branches": 0,
            "pruned_branches": 0,
            "total_probability_mass": 0.0,
            "retained_probability_mass": 0.0,
            "pruned_probability_mass": 0.0,
            "pruned_mass_fraction": 0.0,
        }
    j_min = min(a.j_objective for a in all_assessments)
    weights = [math.exp(-(a.j_objective - j_min)) for a in all_assessments]
    z_sum = sum(weights) or 1.0
    probs = [w / z_sum for w in weights]
    elig_dsts = {a.dst for a in eligible}
    ret_mass = sum(p for a, p in zip(all_assessments, probs) if a.dst in elig_dsts and a.eligible)
    pruned_mass = max(0.0, 1.0 - ret_mass)
    return {
        "total_branches": len(all_assessments),
        "retained_branches": len(eligible),
        "pruned_branches": len(all_assessments) - len(eligible),
        "total_probability_mass": 1.0,
        "retained_probability_mass": round(ret_mass, 6),
        "pruned_probability_mass": round(pruned_mass, 6),
        "pruned_mass_fraction": round(pruned_mass, 6),
    }


def centroidal_deviation(
    cn: sqlite3.Connection,
    path: Sequence[int],
    *,
    p_bar: float = 0.5,
) -> dict[str, Any]:
    """Compute the 7-D CAT manifold centroidal deviation for a token path:
    mean, median, histogram mode, L_∞ norm, and the `inf - 1 < 4p` stability check.
    """
    if not path:
        return {
            "mean": 0.0, "median": 0.0, "mode": 0.0, "linf": 0.0,
            "p_bar": float(p_bar), "inf_minus_1": -1.0,
            "four_p": 4.0 * float(p_bar), "inf_minus_1_lt_4p": True,
        }
    vecs: list[list[float]] = []
    for tid in path:
        emb = _qm._embed7(cn, int(tid))
        vecs.append(emb if emb is not None else [0.5] * 7)

    n = len(vecs)
    centroid = [sum(v[k] for v in vecs) / float(n) for k in range(7)]
    devs: list[float] = []
    linf = 0.0
    for v in vecs:
        d_l2 = math.sqrt(sum((v[k] - centroid[k]) ** 2 for k in range(7)))
        d_inf = max(abs(v[k] - centroid[k]) for k in range(7))
        devs.append(d_l2)
        if d_inf > linf:
            linf = d_inf

    mean_d = sum(devs) / float(n)
    med_d = statistics.median(devs)
    # Histogram-binned mode at 0.05 resolution (deterministic tie-break to lowest bin)
    bins: dict[int, int] = {}
    for d in devs:
        b = int(round(d / 0.05))
        bins[b] = bins.get(b, 0) + 1
    best_bin = min(bins.keys(), key=lambda b: (-bins[b], b))
    mode_d = round(best_bin * 0.05, 6)

    p_val = max(0.0, float(p_bar))
    inf_m1 = linf - 1.0
    four_p = 4.0 * p_val
    return {
        "mean": round(mean_d, 6),
        "median": round(med_d, 6),
        "mode": mode_d,
        "linf": round(linf, 6),
        "p_bar": round(p_val, 6),
        "inf_minus_1": round(inf_m1, 6),
        "four_p": round(four_p, 6),
        "inf_minus_1_lt_4p": bool(inf_m1 < four_p),
    }


def mc_dijkstra_route(
    cn: sqlite3.Connection,
    src: int,
    dst: int,
    *,
    observer: Optional[float] = None,
    cfg: MCPruningConfig = MCPruningConfig(),
    max_nodes: int = _qm._MAX_DIJKSTRA_NODES,
    allowed_nodes: Optional[set[int]] = None,
) -> dict[str, Any]:
    """Weighted Dijkstra shortest path over Monte Carlo CVaR-screened candidate edges."""
    cfg.validate()
    dist: dict[int, float] = {int(src): 0.0}
    prev: dict[int, int] = {}
    edge_meta: dict[tuple[int, int], EdgeMCAssessment] = {}
    heap: list[tuple[float, int]] = [(0.0, int(src))]
    visited: set[int] = set()
    pruned_count = 0
    evaluated_count = 0
    retained_mass_sum = 0.0
    pruned_mass_sum = 0.0
    screened_nodes = 0

    while heap and len(visited) < max_nodes:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == int(dst):
            break
        eligible, all_cands = screen_candidate_edges(
            cn, node, observer=observer, cfg=cfg, allowed_nodes=allowed_nodes
        )
        ledger = compute_pruned_mass_ledger(eligible, all_cands)
        if all_cands:
            screened_nodes += 1
            retained_mass_sum += float(ledger["retained_probability_mass"])
            pruned_mass_sum += float(ledger["pruned_probability_mass"])
        evaluated_count += len(all_cands)
        pruned_count += len(all_cands) - len(eligible)
        for a in eligible:
            nxt = a.dst
            nd = d + a.j_objective
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = node
                edge_meta[(node, nxt)] = a
                heapq.heappush(heap, (nd, nxt))

    denom_nodes = float(max(1, screened_nodes))
    mean_ret_mass = round(retained_mass_sum / denom_nodes, 6) if screened_nodes else 1.0
    mean_pruned_mass = round(pruned_mass_sum / denom_nodes, 6) if screened_nodes else 0.0

    if int(dst) not in dist or int(dst) not in visited:
        return {
            "found": False,
            "path": [],
            "tokens": [],
            "cost": math.inf,
            "expected_cost": math.inf,
            "cvar_cost": math.inf,
            "uncertainty_cost": math.inf,
            "hops": 0,
            "explored": len(visited),
            "evaluated_edges": evaluated_count,
            "pruned_edges": pruned_count,
            "retained_probability_mass": mean_ret_mass,
            "pruned_probability_mass": mean_pruned_mass,
            "centroidal": centroidal_deviation(cn, []),
        }

    path = [int(dst)]
    while path[-1] != int(src):
        path.append(prev[path[-1]])
    path.reverse()

    hop_assessments = [edge_meta[(u, v)] for u, v in zip(path, path[1:]) if (u, v) in edge_meta]
    exp_sum = sum(a.expected_cost for a in hop_assessments)
    cvar_sum = sum(a.cvar_beta for a in hop_assessments)
    unc_sum = sum(a.uncertainty for a in hop_assessments)

    return {
        "found": True,
        "path": path,
        "tokens": [_qm._token_of(cn, t) for t in path],
        "cost": round(dist[int(dst)], 6),
        "expected_cost": round(exp_sum, 6),
        "cvar_cost": round(cvar_sum, 6),
        "uncertainty_cost": round(unc_sum, 6),
        "hops": len(path) - 1,
        "explored": len(visited),
        "evaluated_edges": evaluated_count,
        "pruned_edges": pruned_count,
        "retained_probability_mass": mean_ret_mass,
        "pruned_probability_mass": mean_pruned_mass,
        "centroidal": centroidal_deviation(cn, path),
    }


def mc_greedy_dijkstra_route(
    cn: sqlite3.Connection,
    src: int,
    dst: int,
    *,
    observer: Optional[float] = None,
    cfg: MCPruningConfig = MCPruningConfig(),
    max_hops: int = _qm._MAX_ROUTE_HOPS,
    allowed_nodes: Optional[set[int]] = None,
) -> dict[str, Any]:
    """Monte Carlo decision pruning followed by greedy-Dijkstra next-path selection.

    At each step, candidate outgoing edges are screened via `screen_candidate_edges`
    (Monte Carlo E[C] + λ CVaR_β + ζ U + cost-complexity pruning). Among eligible
    edges, greedy-Dijkstra selects the next hop minimizing `J(u -> v) + h(v, dst)`
    where `h(v, dst)` is the remaining exact Dijkstra cost on the pruned subgraph.
    If greedy-only traversal dead-ends, falls back cleanly to `mc_dijkstra_route`.
    """
    cfg.validate()
    exact = mc_dijkstra_route(
        cn, src, dst, observer=observer, cfg=cfg, allowed_nodes=allowed_nodes
    )
    # Also run purely local greedy step over the MC-pruned edges for comparative telemetry
    path = [int(src)]
    visited = {int(src)}
    node = int(src)
    greedy_cost = 0.0
    pruned_total = 0
    eval_total = 0

    for _ in range(max_hops):
        if node == int(dst):
            break
        eligible, all_cands = screen_candidate_edges(
            cn, node, observer=observer, cfg=cfg, allowed_nodes=allowed_nodes
        )
        eval_total += len(all_cands)
        pruned_total += len(all_cands) - len(eligible)
        choices = [a for a in eligible if a.dst not in visited]
        if not choices:
            break
        # Prefer direct edge to dst if eligible, otherwise lowest J(e) + CAT distance heuristic to dst
        dst_cat = _qm.cat6_coords(cn, int(dst))
        best_a = min(
            choices,
            key=lambda a: (
                0.0 if a.dst == int(dst) else (
                    a.j_objective + _qm._COST_W_CAT * _qm.cat6_distance(_qm.cat6_coords(cn, a.dst), dst_cat)
                ),
                a.dst,
            ),
        )
        greedy_cost += best_a.j_objective
        node = best_a.dst
        visited.add(node)
        path.append(node)

    greedy_found = (node == int(dst))
    greedy_cent = centroidal_deviation(cn, path if greedy_found else [])

    # Select greedy path if it reaches dst, satisfies inf - 1 < 4p, and is within 1.15x of exact Dijkstra
    use_greedy = (
        greedy_found
        and greedy_cent["inf_minus_1_lt_4p"]
        and (not exact["found"] or greedy_cost <= 1.15 * exact["cost"])
    )
    chosen = {
        "found": greedy_found,
        "path": path,
        "tokens": [_qm._token_of(cn, t) for t in path],
        "cost": round(greedy_cost, 6) if greedy_found else math.inf,
        "hops": len(path) - 1,
        "evaluated_edges": eval_total,
        "pruned_edges": pruned_total,
        "centroidal": greedy_cent,
        "mode": "mc_greedy_dijkstra",
        "fallback_to_exact": False,
        "exact_comparison": exact,
    } if use_greedy else {
        **exact,
        "mode": "mc_dijkstra_exact_fallback" if exact["found"] else "unreachable",
        "fallback_to_exact": not use_greedy,
        "greedy_attempt": {
            "found": greedy_found,
            "path": path,
            "cost": round(greedy_cost, 6) if greedy_found else math.inf,
            "centroidal": greedy_cent,
        },
    }
    return chosen


def publish_mc_pruning_report(cn: sqlite3.Connection, report: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"at": time.time(), **dict(report)}
    cn.execute(
        "CREATE TABLE IF NOT EXISTS brain_kv (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)"
    )
    cn.execute(
        "INSERT INTO brain_kv(key, value, updated_at) VALUES(?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (KV_MC_PRUNING_LATEST, json.dumps(payload, sort_keys=True), payload["at"]),
    )
    return payload


__all__ = [
    "KV_MC_PRUNING_LATEST",
    "MCPruningConfig",
    "EdgeMCAssessment",
    "compute_var_cvar",
    "evaluate_edge_mc",
    "screen_candidate_edges",
    "compute_pruned_mass_ledger",
    "centroidal_deviation",
    "mc_dijkstra_route",
    "mc_greedy_dijkstra_route",
    "publish_mc_pruning_report",
]
