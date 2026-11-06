"""Acceptance and unit tests for QUIPU's viscous Navier-Stokes + VNN coupling,
Monte Carlo CVaR decision pruning + Greedy-Dijkstra routing, and 5-term semantic
distillation student.
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np
import pytest

from src.quipu import quipu_minimal as qm
from src.quipu.qpsi import (
    mc_decision_pruning as mcp,
    pcm_cells as pc,
    relational_shear as rs,
    rheology as rh,
    semantic_distillation as sd,
    viscous_interaction as vi,
)


# ---------------------------------------------------------------------------
# 1. Multidimensional Navier-Stokes & VNN Graph Evolution
# ---------------------------------------------------------------------------

def test_navier_stokes_2d_and_3d_incompressibility_and_viscous_decay():
    # 2D Taylor-Green vortex
    cfg2 = vi.NavierStokesConfig(dims=(16, 16), nu=0.05, dt=0.01)
    st0 = vi.NavierStokesState.taylor_green(cfg2, amplitude=0.6)
    ke0 = 0.5 * float(np.mean(np.sum(st0.u * st0.u, axis=0)))

    st1, diag1 = vi.ns_step(st0, cfg2)
    assert diag1["finite"] is True
    # Discrete central-difference divergence after Helmholtz-Hodge projection is ~0
    assert diag1["div_linf"] < 1e-12
    # Unforced viscous flow dissipates kinetic energy
    assert diag1["kinetic_energy"] < ke0

    # 3D incompressible flow with non-zero initial divergence is projected to div=0
    cfg3 = vi.NavierStokesConfig(dims=(8, 8, 8), domain_lengths=(1.0, 1.0, 1.0), nu=0.02, dt=0.005)
    rng = np.random.default_rng(7)
    u_divergent = rng.normal(0.0, 0.3, size=(3, 8, 8, 8))
    st3_0 = vi.NavierStokesState(u=u_divergent, p=np.zeros((8, 8, 8)))
    st3_1, diag3 = vi.ns_step(st3_0, cfg3)
    assert diag3["finite"] is True
    assert diag3["div_linf"] < 1e-12


def test_navier_stokes_stability_substepping():
    # High velocity / large dt triggers automatic substepping to preserve CFL and Fo bounds
    cfg = vi.NavierStokesConfig(dims=(16, 16), nu=0.25, dt=0.20, cfl_max=0.45, fo_max=0.45)
    st = vi.NavierStokesState.taylor_green(cfg, amplitude=1.5)
    stab = vi.check_ns_stability(st, cfg)
    assert stab["stable_single_step"] is False
    assert stab["recommended_substeps"] > 1

    st_next, diag = vi.ns_step(st, cfg)
    assert diag["substeps"] == stab["recommended_substeps"]
    assert diag["cfl_adv"] <= cfg.cfl_max + 1e-9
    assert diag["fo_visc"] <= cfg.fo_max + 1e-9
    assert diag["div_linf"] < 1e-12
    assert diag["finite"] is True


def test_vnn_graph_convection_diffusion_and_typed_coupling(tmp_path):
    rng = np.random.default_rng(42)
    Z0 = rng.uniform(0.0, 1.0, size=(6, 7))
    edges = [
        (0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7),
        (3, 4, 0.85), (4, 5, 0.75), (5, 0, 0.65),
    ]
    vnn0 = vi.VNNState(node_ids=[10, 11, 12, 13, 14, 15], Z=Z0, edges=edges)

    # Pure diffusion (kappa_s=0, forcing_gain=0) strictly reduces Dirichlet graph energy
    cfg_diff = vi.VNNConfig(nu_s=0.20, kappa_s=0.0, forcing_gain=0.0, dt=0.10)
    _, diag0 = vi.vnn_step(vnn0, vi.VNNConfig(nu_s=0.0, kappa_s=0.0, forcing_gain=0.0, dt=0.01))
    vnn1, diag1 = vi.vnn_step(vnn0, cfg_diff)
    assert diag1["dirichlet_energy"] < diag0["dirichlet_energy"]
    assert diag1["finite"] is True

    # Coupled step with rheology and relational shear readings
    arr = pc.CellArray()
    rheo = rh.reading(arr)
    coupler = vi.TypedViscousCoupler(
        ns_cfg=vi.NavierStokesConfig(dims=(12, 12), nu=0.03, dt=0.01),
        vnn_cfg=vi.VNNConfig(nu_s=0.10, kappa_s=0.05, forcing_gain=0.08, dt=0.10),
    )
    ns0 = vi.NavierStokesState.taylor_green(coupler.ns_cfg, amplitude=0.4)
    ns1, vnn2, rep = coupler.step(
        ns0,
        vnn0,
        rheology_reading=rheo,
        shear_reading={"lambda_rel_stp": 0.12},
        target_mesh7=[0.5] * 7,
    )
    assert rep.physical.spatial_dims == 2
    assert rep.semantic.semantic_dims == 7
    assert rep.physical.div_linf < 1e-12
    assert vnn2.Z.shape == (6, 7)

    # Dimensional mismatch is rejected by the typed interface
    bad_vnn = vi.VNNState(node_ids=[1, 2], Z=np.zeros((2, 3)), edges=[(0, 1, 0.5)])
    with pytest.raises(TypeError):
        coupler.step(ns1, bad_vnn)

    # Persist telemetry to brain_kv only
    cn = sqlite3.connect(str(tmp_path / "vnn.sqlite"))
    saved = vi.publish_vnn_report(cn, rep)
    assert saved["semantic"]["semantic_dims"] == 7
    row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (vi.KV_VNN_LATEST,)).fetchone()
    assert row is not None
    cn.close()


# ---------------------------------------------------------------------------
# 2. Monte Carlo CVaR Decision Pruning & Greedy-Dijkstra
# ---------------------------------------------------------------------------

def _build_routing_db(tmp_path) -> sqlite3.Connection:
    cn = sqlite3.connect(str(tmp_path / "mc_route.sqlite"))
    cn.row_factory = sqlite3.Row
    cn.executescript(
        """
        CREATE TABLE mesh_slm_vocab (
            token_id INTEGER PRIMARY KEY,
            token TEXT NOT NULL,
            i INTEGER NOT NULL DEFAULT 0,
            j INTEGER NOT NULL DEFAULT 0,
            freq INTEGER NOT NULL DEFAULT 10
        );
        CREATE TABLE mesh_slm_embed (
            token_id INTEGER PRIMARY KEY,
            e_vision REAL, e_touch REAL, e_smell REAL, e_body REAL,
            e_brain REAL, e_perception REAL, e_entirety REAL
        );
        CREATE TABLE mesh_slm_quipu (
            src INTEGER NOT NULL,
            dst INTEGER NOT NULL,
            weight REAL NOT NULL,
            samples INTEGER NOT NULL DEFAULT 10,
            PRIMARY KEY (src, dst)
        );
        CREATE TABLE mesh_slm_quipu_node (
            node_id INTEGER PRIMARY KEY,
            resuscitation_weight REAL NOT NULL DEFAULT 0.8,
            directed_target INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    # 5 tokens:
    # Path A: 1 -> 2 -> 5 (high sample count, low uncertainty, steady cost)
    # Path B: 1 -> 3 -> 5 (1->3 has only 0 samples and poor resuscitation -> high U(a) & CVaR)
    # Edge 1 -> 4: very weak weight (0.01) -> pruned by cost-complexity / UCB
    tokens = [
        (1, "anchor_start", 0, 0, 50, (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)),
        (2, "safe_relay",   1, 1, 45, (0.52, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)),
        (3, "risky_hop",    8, 8, 10, (0.95, 0.05, 0.95, 0.05, 0.95, 0.05, 0.5)),
        (4, "pruned_dead",  30, 30, 2, (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.1)),
        (5, "target_goal",  2, 2, 50, (0.54, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)),
    ]
    for tid, tok, i, j, freq, emb in tokens:
        cn.execute(
            "INSERT INTO mesh_slm_vocab(token_id, token, i, j, freq) VALUES(?,?,?,?,?)",
            (tid, tok, i, j, freq),
        )
        cn.execute(
            "INSERT INTO mesh_slm_embed VALUES(?,?,?,?,?,?,?,?)",
            (tid, *emb),
        )
        flat = i * 64 + j
        res_w = 0.90 if tid in (1, 2, 5) else 0.05
        cn.execute(
            "INSERT INTO mesh_slm_quipu_node(node_id, resuscitation_weight, directed_target) VALUES(?,?,0)",
            (flat, res_w),
        )

    edges = [
        (1, 2, 0.88, 100),
        (2, 5, 0.90, 100),
        (1, 3, 0.55, 0),
        (3, 5, 0.55, 0),
        (1, 4, 0.01, 0),
    ]
    for s, d, w, smp in edges:
        cn.execute(
            "INSERT INTO mesh_slm_quipu(src, dst, weight, samples) VALUES(?,?,?,?)",
            (s, d, w, smp),
        )
    cn.commit()
    return cn


def test_mc_cvar_screening_and_greedy_dijkstra(tmp_path):
    var_90, cvar_90 = mcp.compute_var_cvar([0.1, 0.2, 0.3, 0.4, 1.0], beta=0.80)
    assert var_90 == pytest.approx(0.4)
    assert cvar_90 == pytest.approx(0.7)

    cn = _build_routing_db(tmp_path)
    cfg = mcp.MCPruningConfig(
        scenarios=32,
        beta=0.90,
        lam_cvar=0.40,
        zeta_uncert=0.35,
        eta_complexity=0.02,
        sigma_noise=0.06,
        max_ucb_cost=0.75,
        complexity_slack=0.35,
        seed=99,
    )
    eligible, all_cands = mcp.screen_candidate_edges(cn, 1, observer=0.8, cfg=cfg)
    assert len(all_cands) == 3
    # Edge 1->4 (and/or 1->3) is pruned before entering Dijkstra
    elig_dsts = [a.dst for a in eligible]
    assert 2 in elig_dsts
    assert 4 not in elig_dsts

    # Both mc_dijkstra_route and mc_greedy_dijkstra_route select 1 -> 2 -> 5
    exact = mcp.mc_dijkstra_route(cn, 1, 5, observer=0.8, cfg=cfg)
    assert exact["found"] is True
    assert exact["path"] == [1, 2, 5]
    assert exact["pruned_edges"] >= 1
    assert exact["centroidal"]["inf_minus_1_lt_4p"] is True

    greedy = mcp.mc_greedy_dijkstra_route(cn, 1, 5, observer=0.8, cfg=cfg)
    assert greedy["found"] is True
    assert greedy["path"] == [1, 2, 5]
    assert greedy["centroidal"]["inf_minus_1_lt_4p"] is True

    via_qm = qm.mc_pruned_dijkstra_route(cn, 1, 5, observer=0.8, greedy_next_hop=True)
    assert via_qm["found"] is True
    assert via_qm["path"] == [1, 2, 5]
    assert via_qm["centroidal"]["inf_minus_1_lt_4p"] is True
    cn.close()


# ---------------------------------------------------------------------------
# 3. QUIPU-Native 5-Term Semantic Distillation
# ---------------------------------------------------------------------------

def test_semantic_distillation_5term_loss_and_student_convergence():
    teacher_logits = [2.4, 1.1, -0.5, -1.2]
    student_init = [0.1, 0.2, 0.0, -0.1]
    perturbed = [0.15, 0.18, -0.05, -0.12]

    cfg = sd.DistillationConfig(
        alpha_ce=0.25,
        beta_kl=0.35,
        gamma_consistency=0.15,
        eta_complexity=0.10,
        xi_abstention=0.15,
        temperature=2.0,
        learning_rate=0.25,
    )
    loss0 = sd.compute_distillation_loss(
        teacher_logits,
        student_init,
        perturbed_student_logits=perturbed,
        student_weights=[0.5] * 7,
        cfg=cfg,
    )
    assert loss0.total_loss > 0.0
    assert loss0.t2_l_kl == pytest.approx(4.0 * loss0.l_kl)
    assert 0.0 <= loss0.abstention_prob <= 1.0

    # Train OpenSourceDecisionStudent over 12 steps via native rADAM and check KL reduction
    student = sd.OpenSourceDecisionStudent()
    cand_embeds = [
        [0.9, 0.8, 0.1, 0.2, 0.7, 0.8, 0.6],
        [0.5, 0.4, 0.3, 0.3, 0.4, 0.5, 0.4],
        [0.1, 0.2, 0.8, 0.7, 0.2, 0.1, 0.2],
        [0.05, 0.1, 0.9, 0.8, 0.1, 0.05, 0.1],
    ]
    pert_embeds = [
        [x + 0.02 for x in row] for row in cand_embeds
    ]
    mesh7 = [0.8, 0.7, 0.2, 0.3, 0.7, 0.8, 0.6]
    base_qw = [0.85, 0.50, 0.15, 0.05]

    first_loss, _ = student.distill_step(
        teacher_logits, cand_embeds, mesh7,
        perturbed_embeds=pert_embeds, base_quipu_weights=base_qw, cfg=cfg,
    )
    last_loss = first_loss
    for _ in range(11):
        last_loss, info = student.distill_step(
            teacher_logits, cand_embeds, mesh7,
            perturbed_embeds=pert_embeds, base_quipu_weights=base_qw, cfg=cfg,
        )
    assert last_loss.l_kl <= first_loss.l_kl
    assert len(info["weyl_tensor"]) == 5
    assert all(0.0 <= w <= 1.0 for w in info["weyl_tensor"])


def test_shadow_comparison_abstention_and_gate6_hold():
    cfg = sd.DistillationConfig(min_confidence=0.40, max_kl_divergence=0.30)
    # Case 1: High-confidence aligned student -> accepted
    comp_ok = sd.compare_teacher_student(
        teacher_logits=[3.0, 0.2, -1.0],
        student_logits=[2.8, 0.3, -0.9],
        proposes_new_structure=False,
        cfg=cfg,
    )
    assert comp_ok["abstained"] is False
    assert comp_ok["selected_source"] == "student"
    assert comp_ok["top1_agreement"] is True
    assert comp_ok["gate6_hold_required"] is False

    # Case 2: Divergent / flat student -> abstains and falls back to teacher; structural proposal triggers Gate 6 hold
    comp_abstain = sd.compare_teacher_student(
        teacher_logits=[3.5, -1.0, -2.0],
        student_logits=[0.0, 0.01, -0.01],
        proposes_new_structure=True,
        cfg=cfg,
    )
    assert comp_abstain["abstained"] is True
    assert comp_abstain["selected_source"] == "teacher_fallback"
    assert comp_abstain["selected_index"] == 0
    assert comp_abstain["gate6_hold_required"] is True
    assert len(comp_abstain["abstention_reasons"]) >= 1
