"""semantic_distillation — QUIPU-native 5-term semantic distillation & shadow comparison.

Mathematical specification
--------------------------
1. **Student distillation objective**:

       L = α L_CE + β T² L_KL + γ L_consistency + η L_complexity + ξ L_abstention

   * `L_CE`: cross-entropy between verified target distribution y and student
     probabilities p_S(· | x) at T = 1.
   * `T² L_KL`: temperature-scaled Kullback–Leibler divergence
         T² · KL( softmax(z_T / T) ‖ softmax(z_S / T) )
     distilling dark knowledge from QUIPU's MESH-SLM + specialist teacher logits z_T
     into the open-source decision student logits z_S.
   * `L_consistency`: invariance penalty under Viscous Neural Network / Langevin
     perturbation of the 7-D sense embedding (symmetric KL between p_S(· | x) and
     p_S(· | x̃) plus Weyl-tensor alignment residual).
   * `L_complexity`: cost-complexity regularization on the student's projection
     weights and active subgraph footprint (L1/L2 spectral norm + active-route ratio).
   * `L_abstention`: selective-prediction risk with unwind-cost bar
     `p* = C_wrong / (C_wrong + C_missed)` (`cat_residual.pstar`):
         L_abstention = (1 − a(x)) · err(x) + a(x) · p*
     where `a(x) ∈ [0, 1]` is the student's abstention probability.

2. **Preservation of QUIPU learning & Weyl compression**:
   * The student's adaptive parameters step through `radam_optimizer.radam_step`
     (complex gradient `g_re + i·g_im`, toroidal phase coupling) without modifying
     `mesh_slm.py`'s Hebbian tables.
   * Distillation cycles emit the canonical 5-float Newman–Penrose Weyl scalar
     tensor (`ueqgm_engine.weyl_scalar_tensor`, 20-byte Float32 LE packed record).

3. **Comparative shadow evaluation, abstention & governance gates**:
   * `compare_teacher_student`: runs the open-source decision student in shadow
     mode against the MESH-SLM teacher, reporting KL divergence, top-1/top-k
     agreement, epistemic uncertainty, and abstention decisions.
   * When `abstained = True` (student confidence < min_confidence, KL > max_kl,
     or `a(x) ≥ p*`), execution falls back to the MESH-SLM teacher; any action
     proposing a new edge or specialist remains held at Physical Gate 6
     (`governance.authorised_to_realise`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import sqlite3
import time
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from ..radam_optimizer import DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_LR, radam_step
from ..ueqgm_engine import weyl_scalar_tensor
from .cat_residual import pstar


KV_DISTILL_LATEST: str = "entirety:distillation:latest"


@dataclass(frozen=True)
class DistillationConfig:
    alpha_ce: float = 0.25             # α ≥ 0 weight on L_CE
    beta_kl: float = 0.35              # β ≥ 0 weight on T² L_KL
    gamma_consistency: float = 0.15    # γ ≥ 0 weight on L_consistency
    eta_complexity: float = 0.10       # η ≥ 0 weight on L_complexity
    xi_abstention: float = 0.15        # ξ ≥ 0 weight on L_abstention
    temperature: float = 2.0           # distillation temperature T > 0
    cost_wrong: float = 0.65           # C_wrong for unwind-cost bar p*
    cost_missed: float = 0.35          # C_missed for unwind-cost bar p*
    min_confidence: float = 0.28       # fallback floor (matches CONF_FLOOR = 0.22 + margin)
    max_kl_divergence: float = 0.45    # maximum teacher-student KL before shadow fallback
    learning_rate: float = DEFAULT_LR

    def validate(self) -> None:
        for name in ("alpha_ce", "beta_kl", "gamma_consistency", "eta_complexity", "xi_abstention"):
            val = getattr(self, name)
            if not (math.isfinite(val) and val >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0")
        if not (math.isfinite(self.temperature) and self.temperature > 0.0):
            raise ValueError("temperature must be finite and > 0")
        if self.cost_wrong < 0.0 or self.cost_missed < 0.0 or (self.cost_wrong + self.cost_missed) <= 0.0:
            raise ValueError("cost_wrong and cost_missed must be non-negative with positive sum")

    @property
    def abstention_bar(self) -> float:
        """Unwind-cost bar p* = C_wrong / (C_wrong + C_missed)."""
        return pstar(self.cost_wrong, self.cost_missed)


@dataclass(frozen=True)
class DistillationLossBreakdown:
    total_loss: float
    l_ce: float
    l_kl: float
    t2_l_kl: float
    l_consistency: float
    l_complexity: float
    l_abstention: float
    abstention_prob: float
    abstention_bar: float


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Numerically stable softmax(z / T)."""
    if not logits:
        return []
    t = max(1e-8, float(temperature))
    scaled = [float(z) / t for z in logits]
    m = max(scaled)
    exps = [math.exp(z - m) for z in scaled]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """KL(p ‖ q) = Σ_i p_i log(p_i / q_i), non-negative and finite."""
    if len(p) != len(q) or not p:
        raise ValueError("p and q must be non-empty distributions of equal length")
    total = 0.0
    for pi, qi in zip(p, q):
        pc = max(eps, float(pi))
        qc = max(eps, float(qi))
        if float(pi) > 0.0:
            total += float(pi) * math.log(pc / qc)
    return max(0.0, total)


def cross_entropy(target_probs: Sequence[float], student_probs: Sequence[float], eps: float = 1e-12) -> float:
    """H(y, p_S) = −Σ_i y_i log(p_S,i)."""
    if len(target_probs) != len(student_probs) or not target_probs:
        raise ValueError("target_probs and student_probs must be non-empty of equal length")
    return max(
        0.0,
        -sum(float(y) * math.log(max(eps, float(ps))) for y, ps in zip(target_probs, student_probs)),
    )


def compute_distillation_loss(
    teacher_logits: Sequence[float],
    student_logits: Sequence[float],
    *,
    perturbed_student_logits: Optional[Sequence[float]] = None,
    target_probs: Optional[Sequence[float]] = None,
    student_weights: Optional[Sequence[float]] = None,
    active_node_ratio: float = 0.25,
    weyl_alignment_gap: float = 0.0,
    cfg: DistillationConfig = DistillationConfig(),
) -> DistillationLossBreakdown:
    """Evaluate the 5-term QUIPU-native semantic distillation objective:
        L = α L_CE + β T² L_KL + γ L_consistency + η L_complexity + ξ L_abstention
    """
    cfg.validate()
    if len(teacher_logits) != len(student_logits) or not teacher_logits:
        raise ValueError("teacher_logits and student_logits must have matching non-zero length")

    p_t1 = softmax(teacher_logits, temperature=1.0)
    p_s1 = softmax(student_logits, temperature=1.0)
    y = list(target_probs) if target_probs is not None else p_t1

    # 1. Cross-entropy loss L_CE
    l_ce = cross_entropy(y, p_s1)

    # 2. Temperature-scaled KL divergence T² L_KL
    T = cfg.temperature
    p_t_temp = softmax(teacher_logits, temperature=T)
    p_s_temp = softmax(student_logits, temperature=T)
    l_kl = kl_divergence(p_t_temp, p_s_temp)
    t2_l_kl = (T * T) * l_kl

    # 3. Consistency loss L_consistency across VNN / Langevin perturbation + Weyl gap
    if perturbed_student_logits is not None:
        p_s_pert = softmax(perturbed_student_logits, temperature=1.0)
        sym_kl = 0.5 * (kl_divergence(p_s1, p_s_pert) + kl_divergence(p_s_pert, p_s1))
    else:
        sym_kl = 0.0
    l_consistency = sym_kl + max(0.0, float(weyl_alignment_gap))

    # 4. Complexity regularization L_complexity (weight norm + subgraph ratio)
    if student_weights is not None and len(student_weights) > 0:
        d = float(len(student_weights))
        l2_term = sum(float(w) * float(w) for w in student_weights) / d
        l1_term = sum(abs(float(w)) for w in student_weights) / d
        weight_comp = 0.5 * (l2_term + l1_term)
    else:
        weight_comp = 0.0
    l_complexity = weight_comp + max(0.0, min(1.0, float(active_node_ratio)))

    # 5. Selective-prediction abstention loss L_abstention
    # Student confidence = max_k p_S1(k); normalized entropy drives abstention probability a(x)
    n_classes = len(p_s1)
    max_ent = math.log(float(max(2, n_classes)))
    ent = -sum(p * math.log(max(1e-12, p)) for p in p_s1)
    norm_ent = min(1.0, max(0.0, ent / max_ent))
    kl_1 = kl_divergence(p_t1, p_s1)
    abstain_prob = min(1.0, max(0.0, 0.6 * norm_ent + 0.4 * math.tanh(kl_1)))
    p_bar = cfg.abstention_bar
    # Expected error proxy = 1 - sum_k y_k p_S1(k)
    err_proxy = max(0.0, 1.0 - sum(float(yk) * float(pk) for yk, pk in zip(y, p_s1)))
    l_abstention = (1.0 - abstain_prob) * err_proxy + abstain_prob * p_bar

    total = (
        cfg.alpha_ce * l_ce
        + cfg.beta_kl * t2_l_kl
        + cfg.gamma_consistency * l_consistency
        + cfg.eta_complexity * l_complexity
        + cfg.xi_abstention * l_abstention
    )
    return DistillationLossBreakdown(
        total_loss=total,
        l_ce=l_ce,
        l_kl=l_kl,
        t2_l_kl=t2_l_kl,
        l_consistency=l_consistency,
        l_complexity=l_complexity,
        l_abstention=l_abstention,
        abstention_prob=abstain_prob,
        abstention_bar=p_bar,
    )


@dataclass
class OpenSourceDecisionStudent:
    """Lightweight open-source decision student coupled to QUIPU's 7-D sense axes
    via native `radam_step` plasticity and Weyl compression.
    """
    axis_gains: list[float] = field(default_factory=lambda: [0.5] * 7)
    radam_states: dict[int, dict[str, float]] = field(default_factory=dict)
    steps: int = 0
    last_weyl_5: tuple[float, float, float, float, float] = (0.5, 0.5, 0.5, 0.5, 0.5)

    def score_logits(
        self,
        candidate_embeds: Sequence[Sequence[float]],
        mesh7: Sequence[float],
        base_quipu_weights: Optional[Sequence[float]] = None,
    ) -> list[float]:
        """Compute student logits over candidate tokens from 7-D embeddings and axis gains."""
        logits: list[float] = []
        for idx, emb in enumerate(candidate_embeds):
            dot = sum(
                float(self.axis_gains[d]) * float(emb[d]) * float(mesh7[d])
                for d in range(min(7, len(emb), len(mesh7)))
            )
            qw = float(base_quipu_weights[idx]) if base_quipu_weights is not None and idx < len(base_quipu_weights) else 0.0
            logits.append(0.55 * qw + dot)
        return logits

    def distill_step(
        self,
        teacher_logits: Sequence[float],
        candidate_embeds: Sequence[Sequence[float]],
        mesh7: Sequence[float],
        *,
        perturbed_embeds: Optional[Sequence[Sequence[float]]] = None,
        base_quipu_weights: Optional[Sequence[float]] = None,
        active_node_ratio: float = 0.20,
        cfg: DistillationConfig = DistillationConfig(),
    ) -> tuple[DistillationLossBreakdown, dict[str, Any]]:
        """Run one teacher-student distillation step, updating `axis_gains` via `radam_step`
        (real gradient = distillation alignment error, imaginary gradient = consistency residual)
        and compressing the cycle into the 5-float Weyl scalar tensor.
        """
        cfg.validate()
        student_logits = self.score_logits(candidate_embeds, mesh7, base_quipu_weights)
        pert_logits = (
            self.score_logits(perturbed_embeds, mesh7, base_quipu_weights)
            if perturbed_embeds is not None
            else None
        )
        loss = compute_distillation_loss(
            teacher_logits,
            student_logits,
            perturbed_student_logits=pert_logits,
            student_weights=self.axis_gains,
            active_node_ratio=active_node_ratio,
            cfg=cfg,
        )

        p_t = softmax(teacher_logits, temperature=cfg.temperature)
        p_s = softmax(student_logits, temperature=cfg.temperature)
        p_pert = softmax(pert_logits, temperature=1.0) if pert_logits is not None else p_s

        # Update each of the 7 axis gains with QUIPU's native rADAM optimizer
        new_gains: list[float] = []
        for d in range(7):
            st = self.radam_states.setdefault(
                d, {"m": 0.0, "v": 0.0, "t": 0, "pressure": float(self.axis_gains[d]), "pivot_ema": float(self.axis_gains[d]), "theta": 0.0}
            )
            # Real gradient pushes student probability p_S toward teacher p_T along axis d
            g_re = sum(
                (pt_i - ps_i) * float(candidate_embeds[i][d]) * float(mesh7[d])
                for i, (pt_i, ps_i) in enumerate(zip(p_t, p_s))
            )
            # Imaginary gradient captures latent perturbation discrepancy along axis d
            g_im = sum(
                (ps_i - pp_i) * float(candidate_embeds[i][d])
                for i, (ps_i, pp_i) in enumerate(zip(p_s, p_pert))
            )
            p_new = radam_step(
                st,
                grad_real=g_re,
                grad_imag=g_im,
                lr=cfg.learning_rate,
                beta1=DEFAULT_BETA1,
                beta2=DEFAULT_BETA2,
            )
            new_gains.append(p_new)

        self.axis_gains = new_gains
        self.steps += 1

        # Compress cycle telemetry into the 5-float Newman–Penrose Weyl scalar tensor
        max_ent = math.log(float(max(2, len(p_s))))
        ent_s = -sum(p * math.log(max(1e-12, p)) for p in p_s) / max_ent
        alignment = max(0.0, 1.0 - min(1.0, loss.l_kl))
        remnant = max(0.0, 1.0 - loss.abstention_prob)
        self.last_weyl_5 = weyl_scalar_tensor(
            signal_flux=max(p_s) if p_s else 0.0,
            topic_entropy=ent_s,
            corpus_volume=float(len(candidate_embeds)),
            mesh_alignment=alignment,
            remnant_score=remnant,
        )
        return loss, {
            "steps": self.steps,
            "axis_gains": list(self.axis_gains),
            "weyl_tensor": list(self.last_weyl_5),
        }


def compare_teacher_student(
    teacher_logits: Sequence[float],
    student_logits: Sequence[float],
    *,
    perturbed_student_logits: Optional[Sequence[float]] = None,
    student_weights: Optional[Sequence[float]] = None,
    active_node_ratio: float = 0.20,
    proposes_new_structure: bool = False,
    cfg: DistillationConfig = DistillationConfig(),
) -> dict[str, Any]:
    """SLM-MESH shadow evaluation comparing teacher vs. open-source student with
    explicit abstention, fallback, and Physical Gate 6 governance rules.
    """
    loss = compute_distillation_loss(
        teacher_logits,
        student_logits,
        perturbed_student_logits=perturbed_student_logits,
        student_weights=student_weights,
        active_node_ratio=active_node_ratio,
        cfg=cfg,
    )
    p_t = softmax(teacher_logits, temperature=1.0)
    p_s = softmax(student_logits, temperature=1.0)

    top_t = int(np.argmax(p_t)) if p_t else -1
    top_s = int(np.argmax(p_s)) if p_s else -1
    student_conf = float(p_s[top_s]) if top_s >= 0 else 0.0
    teacher_conf = float(p_t[top_t]) if top_t >= 0 else 0.0
    kl_1 = kl_divergence(p_t, p_s)

    reasons: list[str] = []
    if student_conf < cfg.min_confidence:
        reasons.append("low_student_confidence")
    if kl_1 > cfg.max_kl_divergence:
        reasons.append("high_teacher_student_kl")
    if loss.abstention_prob >= loss.abstention_bar:
        reasons.append("abstention_exceeds_pstar_bar")

    abstained = len(reasons) > 0
    selected_source = "teacher_fallback" if abstained else "student"
    selected_index = top_t if abstained else top_s

    return {
        "teacher_top_index": top_t,
        "student_top_index": top_s,
        "top1_agreement": (top_t == top_s),
        "teacher_confidence": round(teacher_conf, 6),
        "student_confidence": round(student_conf, 6),
        "kl_divergence": round(kl_1, 6),
        "abstained": abstained,
        "abstention_reasons": reasons,
        "selected_source": selected_source,
        "selected_index": selected_index,
        "gate6_hold_required": bool(proposes_new_structure),
        "loss": asdict(loss),
    }


def publish_distillation_report(cn: sqlite3.Connection, report: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"at": time.time(), **dict(report)}
    cn.execute(
        "CREATE TABLE IF NOT EXISTS brain_kv (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)"
    )
    cn.execute(
        "INSERT INTO brain_kv(key, value, updated_at) VALUES(?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (KV_DISTILL_LATEST, json.dumps(payload, sort_keys=True), payload["at"]),
    )
    return payload


__all__ = [
    "KV_DISTILL_LATEST",
    "DistillationConfig",
    "DistillationLossBreakdown",
    "OpenSourceDecisionStudent",
    "softmax",
    "kl_divergence",
    "cross_entropy",
    "compute_distillation_loss",
    "compare_teacher_student",
    "publish_distillation_report",
]
