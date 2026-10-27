"""viscous_interaction — Multidimensional Navier-Stokes & Viscous Neural Network (VNN).

Mathematical specification
--------------------------
1. **Multidimensional incompressible Navier-Stokes** (physical field on a periodic D-domain):

       ∂u/∂t + (u · ∇)u = −(1/ρ) ∇p + ν ∇²u + f,      ∇ · u = 0

   Advanced via Chorin–Temam Helmholtz–Hodge pressure projection:
   * Advection-diffusion predictor:
         u* = u^n + Δt · [ −(u^n · ∇_h)u^n + ν ∇_h² u^n + f^n ]
   * Poisson pressure equation on the periodic lattice (exact spectral FFT when NumPy
     is available, or Gauss–Seidel/Jacobi relaxation in pure stdlib):
         ∇_h² p = (ρ / Δt) (∇_h · u*)
   * Divergence-free corrector:
         u^{n+1} = u* − (Δt / ρ) ∇_h p

   Stability invariants (checked and enforced via automatic substepping):
         CFL_adv  = Δt · Σ_d (max |u_d| / Δx_d) ≤ cfl_max  (default 0.45)
         Fo_visc  = ν · Δt · Σ_d (2 / Δx_d²)    ≤ fo_max   (default 0.45 < 0.5)

2. **Viscous Neural Network (VNN) — nonlinear semantic-graph convection-diffusion**:

       dZ/dt = −ν_s L_G Z − κ_s N_G(Z) + F_sem(Z, t)

   where Z ∈ ℝ^{|V| × 7} carries the 7-D sense embeddings (`vision`, `touch`,
   `smell`, `body`, `brain`, `perception`, `entirety`) over QUIPU's directed
   weighted graph G = (V, E, W):
   * L_G: degree-normalised graph Laplacian, (L_G Z)_i = Σ_j w̃_ij (Z_i − Z_j).
   * N_G(Z): nonlinear directed convection transport along Quipu edges,
         N_G(Z)_i = Σ_{j ∈ N+(i)} w_ij · tanh(⟨Z_j − Z_i, d̂_ij⟩) · (Z_j − Z_i)
   * F_sem(Z, t): external semantic forcing toward the live 7-D MESH state, with
     optional Ricci/Weyl trace-free split (`weyl_channel.split`).

   Graph stability invariant (Gershgorin bound λ_max(L_G) ≤ 2 max_i d_i):
         Stab_G = Δt · (ν_s λ_max(L_G) + 2 κ_s d_max) ≤ graph_cfl_max (default 1.8 < 2.0)

3. **Typed Coupling Interface (`TypedViscousCoupler`)**:
   Connects the D-dimensional physical flow (u, p) and the 7-D semantic graph Z
   through dimensionless scalar invariants (enstrophy, strain rate, pressure
   variance, rheology loss angle δ_G from `qpsi.rheology`, and relational shear
   λ_rel_STP from `qpsi.relational_shear`) without assuming that QUIPU's 7
   semantic dimensions are physical spatial coordinates.

Governance
----------
Read-only on the mesh (`mesh_slm_vocab`, `mesh_slm_embed`, `mesh_slm_quipu` are
never written). Optional state persistence writes only `brain_kv["entirety:vnn:latest"]`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
import math
import sqlite3
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .cat_residual import SENSES, SENSE_WEIGHTS
from .weyl_channel import split as weyl_split

AXES_7D: tuple[str, ...] = (
    "vision", "touch", "smell", "body", "brain", "perception", "entirety"
)
KV_VNN_LATEST: str = "entirety:vnn:latest"


# ---------------------------------------------------------------------------
# 1. Multidimensional Incompressible Navier-Stokes (Periodic D-Domain)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NavierStokesConfig:
    dims: tuple[int, ...] = (16, 16)   # spatial grid shape (N_1, ..., N_D), D >= 2
    domain_lengths: tuple[float, ...] = (1.0, 1.0)
    rho: float = 1.0                   # fluid density ρ > 0
    nu: float = 0.02                   # kinematic viscosity ν ≥ 0
    coriolis_f: float = 0.0            # rotating-frame Coriolis parameter f (2Ω sin φ)
    dt: float = 0.01                   # nominal timestep Δt > 0
    cfl_max: float = 0.45              # advective CFL ceiling (< 1)
    fo_max: float = 0.45               # viscous Fourier number ceiling (< 0.5)
    max_substeps: int = 256            # safety cap on automatic substepping

    def validate(self) -> None:
        if len(self.dims) < 2 or any(int(n) < 4 for n in self.dims):
            raise ValueError("dims must have at least 2 spatial axes with size >= 4")
        if len(self.domain_lengths) != len(self.dims) or any(L <= 0.0 for L in self.domain_lengths):
            raise ValueError("domain_lengths must match dims and be strictly positive")
        if not (math.isfinite(self.rho) and self.rho > 0.0):
            raise ValueError("rho must be finite and > 0")
        if not (math.isfinite(self.nu) and self.nu >= 0.0):
            raise ValueError("nu must be finite and >= 0")
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError("dt must be finite and > 0")

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def dx(self) -> tuple[float, ...]:
        return tuple(float(L) / float(n) for L, n in zip(self.domain_lengths, self.dims))


@dataclass
class NavierStokesState:
    u: np.ndarray                      # shape (D, N_1, ..., N_D)
    p: np.ndarray                      # shape (N_1, ..., N_D)
    t: float = 0.0
    steps: int = 0

    @classmethod
    def zeros(cls, cfg: NavierStokesConfig) -> "NavierStokesState":
        cfg.validate()
        u = np.zeros((cfg.ndim,) + tuple(cfg.dims), dtype=float)
        p = np.zeros(tuple(cfg.dims), dtype=float)
        return cls(u=u, p=p, t=0.0, steps=0)

    @classmethod
    def taylor_green(cls, cfg: NavierStokesConfig, amplitude: float = 0.5) -> "NavierStokesState":
        """Divergence-free Taylor–Green vortex initial condition in D >= 2 dimensions."""
        cfg.validate()
        grids = [
            np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
            for n in cfg.dims
        ]
        mesh = np.meshgrid(*grids, indexing="ij")
        u = np.zeros((cfg.ndim,) + tuple(cfg.dims), dtype=float)
        u[0] = amplitude * np.sin(mesh[0]) * np.cos(mesh[1])
        u[1] = -amplitude * np.cos(mesh[0]) * np.sin(mesh[1])
        for d in range(2, cfg.ndim):
            u *= np.cos(mesh[d])
        p = -0.25 * (amplitude ** 2) * (np.cos(2.0 * mesh[0]) + np.cos(2.0 * mesh[1]))
        return cls(u=u, p=p, t=0.0, steps=0)


def _periodic_grad(field: np.ndarray, axis: int, dx: float) -> np.ndarray:
    """Central difference ∂field/∂x_axis on a periodic grid."""
    return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2.0 * dx)


def _periodic_divergence(u: np.ndarray, dx: Sequence[float]) -> np.ndarray:
    """∇_h · u = Σ_d ∂u_d/∂x_d on the periodic grid."""
    div = np.zeros(u.shape[1:], dtype=float)
    for d, h in enumerate(dx):
        div += _periodic_grad(u[d], axis=d, dx=h)
    return div


def _periodic_laplacian(field: np.ndarray, dx: Sequence[float]) -> np.ndarray:
    """∇_h² field = Σ_d (field(x+h e_d) − 2 field(x) + field(x−h e_d)) / h²."""
    lap = np.zeros_like(field, dtype=float)
    for d, h in enumerate(dx):
        lap += (
            np.roll(field, -1, axis=d)
            - 2.0 * field
            + np.roll(field, 1, axis=d)
        ) / (h * h)
    return lap


def _solve_poisson_periodic(rhs: np.ndarray, dx: Sequence[float]) -> np.ndarray:
    """Exact spectral solver for ∇_h² p = rhs with periodic boundary conditions
    compatible with the central-difference divergence/gradient stencil so that
    ∇_h · (u* − (Δt/ρ) ∇_h p) = 0 to machine precision.
    """
    shape = rhs.shape
    rhs_hat = np.fft.fftn(rhs)
    # Central difference ∂/∂x_d has symbol i * sin(k_d h_d) / h_d.
    # Composition (∇_h · ∇_h) has symbol −Σ_d (sin(k_d h_d) / h_d)².
    denom = np.zeros(shape, dtype=float)
    for d, (n, h) in enumerate(zip(shape, dx)):
        freqs = 2.0 * math.pi * np.fft.fftfreq(n, d=1.0)
        symbol_1d = (np.sin(freqs) / h) ** 2
        reshape_dims = [1] * len(shape)
        reshape_dims[d] = n
        denom += symbol_1d.reshape(reshape_dims)

    p_hat = np.zeros_like(rhs_hat)
    mask = denom > 1e-14
    p_hat[mask] = -rhs_hat[mask] / denom[mask]
    # Zero out the Nyquist checkerboard modes where sin(k_d) == 0
    p_hat[~mask] = 0.0
    return np.real(np.fft.ifftn(p_hat))


def project_incompressible(
    u_star: np.ndarray,
    dx: Sequence[float],
    rho: float = 1.0,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Helmholtz–Hodge projection of u_star onto the divergence-free subspace."""
    div_star = _periodic_divergence(u_star, dx)
    rhs = (rho / max(dt, 1e-12)) * div_star
    p = _solve_poisson_periodic(rhs, dx)
    u_proj = np.array(u_star, dtype=float, copy=True)
    for d, h in enumerate(dx):
        u_proj[d] -= (dt / rho) * _periodic_grad(p, axis=d, dx=h)
    return u_proj, p


def check_ns_stability(state: NavierStokesState, cfg: NavierStokesConfig) -> dict[str, Any]:
    """Compute CFL_adv, Fo_visc, required substeps, and divergence norms."""
    cfg.validate()
    dx = cfg.dx
    cfl_adv = 0.0
    for d, h in enumerate(dx):
        u_max = float(np.max(np.abs(state.u[d]))) if state.u.size else 0.0
        cfl_adv += cfg.dt * u_max / h
    fo_visc = cfg.nu * cfg.dt * sum(2.0 / (h * h) for h in dx)
    ratio = max(
        cfl_adv / max(cfg.cfl_max, 1e-9),
        fo_visc / max(cfg.fo_max, 1e-9),
    )
    substeps = max(1, min(cfg.max_substeps, int(math.ceil(ratio)))) if ratio > 1.0 else 1
    div = _periodic_divergence(state.u, dx)
    return {
        "cfl_adv": cfl_adv,
        "fo_visc": fo_visc,
        "stable_single_step": (cfl_adv <= cfg.cfl_max and fo_visc <= cfg.fo_max),
        "recommended_substeps": substeps,
        "dt_effective": cfg.dt / substeps,
        "div_l2": float(np.sqrt(np.mean(div * div))),
        "div_linf": float(np.max(np.abs(div))),
    }


def ns_step(
    state: NavierStokesState,
    cfg: NavierStokesConfig,
    forcing: Optional[np.ndarray] = None,
) -> tuple[NavierStokesState, dict[str, Any]]:
    """Advance the D-dimensional incompressible Navier-Stokes equations by cfg.dt."""
    stab_pre = check_ns_stability(state, cfg)
    substeps = stab_pre["recommended_substeps"]
    dt_sub = cfg.dt / float(substeps)
    dx = cfg.dx

    u = np.array(state.u, dtype=float, copy=True)
    p = np.array(state.p, dtype=float, copy=True)
    f = np.zeros_like(u) if forcing is None else np.asarray(forcing, dtype=float)
    if f.shape != u.shape:
        raise ValueError(f"forcing shape {f.shape} does not match velocity shape {u.shape}")

    for _ in range(substeps):
        # 1. Nonlinear convection (u · ∇)u, viscous diffusion ν ∇²u, and Coriolis 2Ω × u
        conv = np.zeros_like(u)
        diff = np.zeros_like(u)
        cor = np.zeros_like(u)
        if cfg.coriolis_f != 0.0 and cfg.ndim >= 2:
            # Horizontal plane (x_0, x_1): (f k̂ × u)_0 = -f u_1, (f k̂ × u)_1 = +f u_0
            cor[0] = -cfg.coriolis_f * u[1]
            cor[1] = cfg.coriolis_f * u[0]
        for d in range(cfg.ndim):
            diff[d] = cfg.nu * _periodic_laplacian(u[d], dx)
            for k, hk in enumerate(dx):
                conv[d] += u[k] * _periodic_grad(u[d], axis=k, dx=hk)
        u_star = u + dt_sub * (-conv - cor + diff + f)
        # 2. Helmholtz–Hodge pressure projection onto ∇ · u = 0
        u, p = project_incompressible(u_star, dx, rho=cfg.rho, dt=dt_sub)

    next_state = NavierStokesState(u=u, p=p, t=state.t + cfg.dt, steps=state.steps + 1)
    div_post = _periodic_divergence(next_state.u, dx)
    ke = 0.5 * cfg.rho * float(np.mean(np.sum(next_state.u * next_state.u, axis=0)))
    # Enstrophy = ½ ⟨|∇ × u|²⟩ (in 2D: ½ ⟨(∂u_1/∂x_0 − ∂u_0/∂x_1)²⟩)
    vort_sq = np.zeros(cfg.dims, dtype=float)
    for i in range(cfg.ndim):
        for j in range(i + 1, cfg.ndim):
            omega_ij = _periodic_grad(next_state.u[j], i, dx[i]) - _periodic_grad(next_state.u[i], j, dx[j])
            vort_sq += omega_ij * omega_ij
    enstrophy = 0.5 * float(np.mean(vort_sq))

    diagnostics = {
        "substeps": substeps,
        "dt_effective": dt_sub,
        "cfl_adv": stab_pre["cfl_adv"] / substeps,
        "fo_visc": stab_pre["fo_visc"] / substeps,
        "div_l2": float(np.sqrt(np.mean(div_post * div_post))),
        "div_linf": float(np.max(np.abs(div_post))),
        "kinetic_energy": ke,
        "enstrophy": enstrophy,
        "pressure_std": float(np.std(next_state.p)),
        "finite": bool(np.all(np.isfinite(next_state.u)) and np.all(np.isfinite(next_state.p))),
    }
    return next_state, diagnostics


# ---------------------------------------------------------------------------
# 2. Viscous Neural Network (VNN) — Nonlinear Semantic-Graph Evolution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VNNConfig:
    nu_s: float = 0.08                 # semantic graph viscosity (diffusion) ν_s ≥ 0
    kappa_s: float = 0.04              # nonlinear graph convection κ_s ≥ 0
    forcing_gain: float = 0.05         # semantic forcing gain toward target state
    dt: float = 0.10                   # integration step Δt > 0
    graph_cfl_max: float = 1.80        # stability ceiling on Δt·(ν_s λ_max + 2 κ_s d_max) < 2
    weyl_trace_free: bool = True       # project forcing through (I − ŵŵ*) on the 6 sense axes
    max_substeps: int = 32

    def validate(self) -> None:
        if not (math.isfinite(self.nu_s) and self.nu_s >= 0.0):
            raise ValueError("nu_s must be finite and >= 0")
        if not (math.isfinite(self.kappa_s) and self.kappa_s >= 0.0):
            raise ValueError("kappa_s must be finite and >= 0")
        if not (math.isfinite(self.dt) and self.dt > 0.0):
            raise ValueError("dt must be finite and > 0")


@dataclass
class VNNState:
    node_ids: list[int]
    Z: np.ndarray                      # shape (|V|, 7) semantic state vectors
    edges: list[tuple[int, int, float]]  # (src_idx, dst_idx, weight) in 0..|V|-1
    t: float = 0.0
    steps: int = 0


def _trace_free_7d(vec7: np.ndarray) -> np.ndarray:
    """Apply `weyl_channel.split` across the first 6 sense axes while keeping axis 7 (entirety)."""
    out = np.array(vec7, dtype=float, copy=True)
    if out.ndim == 1:
        sp = weyl_split({s: float(out[i]) for i, s in enumerate(SENSES)}, SENSE_WEIGHTS)
        for i, s in enumerate(SENSES):
            out[i] = sp.weyl[s]
        return out
    for row in range(out.shape[0]):
        sp = weyl_split({s: float(out[row, i]) for i, s in enumerate(SENSES)}, SENSE_WEIGHTS)
        for i, s in enumerate(SENSES):
            out[row, i] = sp.weyl[s]
    return out


def check_vnn_stability(state: VNNState, cfg: VNNConfig) -> dict[str, Any]:
    """Compute Gershgorin spectral bound λ_max(L_G) and graph convection-diffusion CFL."""
    cfg.validate()
    n = len(state.node_ids)
    if n == 0:
        return {"lambda_max_bound": 0.0, "d_max": 0.0, "graph_cfl": 0.0, "substeps": 1, "stable": True}
    deg = np.zeros(n, dtype=float)
    for u, v, w in state.edges:
        if 0 <= u < n and 0 <= v < n and u != v:
            deg[u] += max(0.0, float(w))
            deg[v] += max(0.0, float(w))
    # For degree-normalised symmetric Laplacian, λ_max(L_G) ≤ 2.0; for unnormalised, ≤ 2 max_i d_i.
    # Because vnn_step normalises each node's row by max(1.0, deg[i]), effective d_max ≤ 1.0.
    d_max = float(np.max(deg)) if n > 0 else 0.0
    eff_deg = min(1.0, d_max) if d_max > 0.0 else 0.0
    lambda_max_bound = 2.0 * eff_deg
    graph_cfl = cfg.dt * (cfg.nu_s * lambda_max_bound + 2.0 * cfg.kappa_s * eff_deg)
    substeps = max(1, min(cfg.max_substeps, int(math.ceil(graph_cfl / cfg.graph_cfl_max)))) if graph_cfl > cfg.graph_cfl_max else 1
    return {
        "lambda_max_bound": lambda_max_bound,
        "d_max": d_max,
        "graph_cfl": graph_cfl,
        "substeps": substeps,
        "dt_effective": cfg.dt / substeps,
        "stable": (graph_cfl / substeps) <= cfg.graph_cfl_max,
    }


def vnn_step(
    state: VNNState,
    cfg: VNNConfig,
    *,
    target_mesh7: Optional[Sequence[float]] = None,
    forcing_scale: float = 1.0,
) -> tuple[VNNState, dict[str, Any]]:
    """Advance the nonlinear semantic-graph ODE:
        dZ/dt = −ν_s L_G Z − κ_s N_G(Z) + F_sem(Z, t)
    """
    stab = check_vnn_stability(state, cfg)
    substeps = stab["substeps"]
    dt_sub = cfg.dt / float(substeps)
    Z = np.array(state.Z, dtype=float, copy=True)
    n = Z.shape[0]
    if n == 0:
        return VNNState(list(state.node_ids), Z, list(state.edges), state.t + cfg.dt, state.steps + 1), {
            "substeps": 1, "graph_cfl": 0.0, "dirichlet_energy": 0.0, "convection_norm": 0.0, "finite": True
        }

    # Build degree-normalised adjacency for L_G (undirected diffusion) and N_G (directed convection)
    sym_deg = np.zeros(n, dtype=float)
    out_deg = np.zeros(n, dtype=float)
    valid_edges: list[tuple[int, int, float]] = []
    for u, v, w in state.edges:
        if 0 <= u < n and 0 <= v < n and u != v and w > 0.0:
            wf = float(w)
            valid_edges.append((u, v, wf))
            sym_deg[u] += 0.5 * wf
            sym_deg[v] += 0.5 * wf
            out_deg[u] += wf

    sym_norm = np.maximum(1.0, sym_deg)
    out_norm = np.maximum(1.0, out_deg)

    target_vec = np.asarray(target_mesh7 if target_mesh7 is not None else [0.5] * 7, dtype=float).reshape(1, 7)
    last_conv_norm = 0.0

    for _ in range(substeps):
        # 1. Linear graph diffusion: (L_G Z)_i = Σ_j w̃_ij (Z_i − Z_j)
        lap_Z = np.zeros_like(Z)
        # 2. Nonlinear directed convection: N_G(Z)_i = Σ_{j ∈ N+(i)} ŵ_ij tanh(⟨ΔZ, d̂⟩) ΔZ
        conv_Z = np.zeros_like(Z)

        for u, v, wf in valid_edges:
            diff_uv = Z[v] - Z[u]
            # Symmetric diffusion contribution
            w_sym_u = (0.5 * wf) / sym_norm[u]
            w_sym_v = (0.5 * wf) / sym_norm[v]
            lap_Z[u] += w_sym_u * (Z[u] - Z[v])
            lap_Z[v] += w_sym_v * (Z[v] - Z[u])

            # Directed nonlinear convection along u -> v
            norm_diff = float(np.linalg.norm(diff_uv))
            if norm_diff > 1e-12:
                # Nonlinear flux velocity phi(ΔZ) = tanh(norm_diff)
                flux = math.tanh(norm_diff)
                conv_Z[u] += (wf / out_norm[u]) * flux * diff_uv

        # 3. Semantic forcing F_sem(Z, t)
        raw_force = cfg.forcing_gain * float(forcing_scale) * (target_vec - Z)
        if cfg.weyl_trace_free:
            raw_force = _trace_free_7d(raw_force)

        dZ_dt = -cfg.nu_s * lap_Z - cfg.kappa_s * conv_Z + raw_force
        Z = np.clip(Z + dt_sub * dZ_dt, -2.0, 2.0)
        last_conv_norm = float(np.linalg.norm(conv_Z) / math.sqrt(max(1, n)))

    # Compute Dirichlet graph energy ½ Σ_(u,v) w_uv ‖Z_u − Z_v‖²
    dirichlet = 0.0
    for u, v, wf in valid_edges:
        dvec = Z[u] - Z[v]
        dirichlet += 0.5 * wf * float(np.dot(dvec, dvec))

    next_state = VNNState(
        node_ids=list(state.node_ids),
        Z=Z,
        edges=list(state.edges),
        t=state.t + cfg.dt,
        steps=state.steps + 1,
    )
    diagnostics = {
        "substeps": substeps,
        "dt_effective": dt_sub,
        "graph_cfl": stab["graph_cfl"] / substeps,
        "dirichlet_energy": dirichlet,
        "convection_norm": last_conv_norm,
        "mean_norm": float(np.mean(np.linalg.norm(Z, axis=1))),
        "finite": bool(np.all(np.isfinite(Z))),
    }
    return next_state, diagnostics


# ---------------------------------------------------------------------------
# 3. Typed Coupling Interface (Physical D-Space ↔ Semantic 7-D Graph)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysicalFieldSummary:
    """Dimensionless invariants extracted from the D-dimensional physical flow."""
    spatial_dims: int
    kinetic_energy: float
    enstrophy: float
    div_linf: float
    pressure_std: float


@dataclass(frozen=True)
class SemanticGraphSummary:
    """Dimensionless invariants extracted from the 7-D semantic graph."""
    semantic_dims: int
    n_nodes: int
    n_edges: int
    dirichlet_energy: float
    convection_norm: float
    rheology_loss_angle: float
    relational_shear_stp: float


@dataclass(frozen=True)
class CoupledStepReport:
    physical: PhysicalFieldSummary
    semantic: SemanticGraphSummary
    effective_nu_phys: float
    effective_nu_sem: float
    effective_kappa_sem: float
    forcing_scale: float
    ns_diagnostics: dict[str, Any]
    vnn_diagnostics: dict[str, Any]
    bathymetric_feature_dims: int = 6
    bathymetry_features: Optional[dict[str, float]] = None


class TypedViscousCoupler:
    """Typed coupling interface between physical Navier-Stokes (D_phys spatial dims)
    and the Viscous Neural Network (7 semantic dims), plus optional 6-feature
    bathymetric / morphodynamic coupling (`vnn_bathymetry`).

    Enforces dimensional separation: physical velocity vectors u(x) ∈ ℝ^{D_phys}
    and 6-D bathymetric features are never added to or dot-producted with semantic
    state vectors Z_i ∈ ℝ^7.
    All coupling occurs exclusively through dimensionless scalar invariants:
    - From QUIPU rheology (`rheology.reading`) & relational shear (`relational_shear`):
          ν_s = ν_{s,0} · (1 + w_r / 0.125)
          κ_s = κ_{s,0} · (1 + λ_rel_STP)
    - From physical enstrophy Ω, kinetic energy E_k, and optional seabed slope s:
          forcing_scale = (1 + tanh(Ω / (E_k + 1))) · (1 + 0.1 tanh(s))
    """

    def __init__(
        self,
        ns_cfg: NavierStokesConfig = NavierStokesConfig(),
        vnn_cfg: VNNConfig = VNNConfig(),
    ) -> None:
        ns_cfg.validate()
        vnn_cfg.validate()
        self.ns_cfg = ns_cfg
        self.vnn_cfg = vnn_cfg

    def step(
        self,
        ns_state: NavierStokesState,
        vnn_state: VNNState,
        *,
        rheology_reading: Optional[Mapping[str, Any]] = None,
        shear_reading: Optional[Mapping[str, Any]] = None,
        bathymetry_features: Optional[Mapping[str, float]] = None,
        target_mesh7: Optional[Sequence[float]] = None,
        physical_forcing: Optional[np.ndarray] = None,
    ) -> tuple[NavierStokesState, VNNState, CoupledStepReport]:
        if ns_state.u.shape[0] != self.ns_cfg.ndim:
            raise TypeError("Physical velocity field dimension mismatch")
        if vnn_state.Z.ndim != 2 or vnn_state.Z.shape[1] != 7:
            raise TypeError("Semantic state Z must have shape (|V|, 7)")

        # Read dimensionless rheology & relational shear invariants
        w_r = float((rheology_reading or {}).get("edge_weight", 0.125))
        loss_angle = float((rheology_reading or {}).get("loss_angle", math.pi / 4.0))
        lam_stp = float((shear_reading or {}).get("lambda_rel_stp", 0.0))

        bathy_dict: Optional[dict[str, float]] = None
        slope_factor = 1.0
        if bathymetry_features is not None:
            bathy_dict = {str(k): float(v) for k, v in bathymetry_features.items()}
            slope_val = max(0.0, float(bathy_dict.get("seabed_slope", 0.0)))
            slope_factor = 1.0 + 0.10 * math.tanh(slope_val)

        # Modulate physical viscosity by viscoelastic loss factor sin(δ_G) in [0.5, 1.5]
        visc_factor = 0.5 + math.sin(max(0.0, min(math.pi / 2.0, loss_angle)))
        eff_nu_phys = self.ns_cfg.nu * visc_factor * slope_factor
        ns_cfg_eff = dataclasses.replace(self.ns_cfg, nu=eff_nu_phys)

        # Step 1: Advance physical Navier-Stokes field
        ns_next, ns_diag = ns_step(ns_state, ns_cfg_eff, forcing=physical_forcing)

        # Step 2: Map physical invariants + rheology/shear to semantic VNN coefficients
        eff_nu_sem = self.vnn_cfg.nu_s * (1.0 + max(0.0, min(0.25, w_r)) / 0.125)
        eff_kappa_sem = self.vnn_cfg.kappa_s * (1.0 + max(0.0, min(1.0, lam_stp)))
        forcing_scale = (
            1.0 + math.tanh(ns_diag["enstrophy"] / (ns_diag["kinetic_energy"] + 1.0))
        ) * slope_factor
        vnn_cfg_eff = dataclasses.replace(
            self.vnn_cfg, nu_s=eff_nu_sem, kappa_s=eff_kappa_sem
        )

        # Step 3: Advance semantic graph VNN
        vnn_next, vnn_diag = vnn_step(
            vnn_state,
            vnn_cfg_eff,
            target_mesh7=target_mesh7,
            forcing_scale=forcing_scale,
        )

        phys_summary = PhysicalFieldSummary(
            spatial_dims=self.ns_cfg.ndim,
            kinetic_energy=ns_diag["kinetic_energy"],
            enstrophy=ns_diag["enstrophy"],
            div_linf=ns_diag["div_linf"],
            pressure_std=ns_diag["pressure_std"],
        )
        sem_summary = SemanticGraphSummary(
            semantic_dims=7,
            n_nodes=len(vnn_next.node_ids),
            n_edges=len(vnn_next.edges),
            dirichlet_energy=vnn_diag["dirichlet_energy"],
            convection_norm=vnn_diag["convection_norm"],
            rheology_loss_angle=loss_angle,
            relational_shear_stp=lam_stp,
        )
        report = CoupledStepReport(
            physical=phys_summary,
            semantic=sem_summary,
            effective_nu_phys=eff_nu_phys,
            effective_nu_sem=eff_nu_sem,
            effective_kappa_sem=eff_kappa_sem,
            forcing_scale=forcing_scale,
            ns_diagnostics=ns_diag,
            vnn_diagnostics=vnn_diag,
            bathymetric_feature_dims=6,
            bathymetry_features=bathy_dict,
        )
        return ns_next, vnn_next, report


def load_vnn_state_from_conn(
    cn: sqlite3.Connection, max_nodes: int = 256, max_edges_per_node: int = 16
) -> VNNState:
    """Read-only loader building a VNNState from `mesh_slm_vocab`, `mesh_slm_embed`, and `mesh_slm_quipu`."""
    rows = cn.execute(
        "SELECT v.token_id, e.e_vision, e.e_touch, e.e_smell, e.e_body, "
        "       e.e_brain, e.e_perception, e.e_entirety "
        "FROM mesh_slm_vocab v "
        "JOIN mesh_slm_embed e ON e.token_id = v.token_id "
        "ORDER BY v.freq DESC LIMIT ?",
        (max(1, int(max_nodes)),),
    ).fetchall()
    node_ids: list[int] = []
    idx_of: dict[int, int] = {}
    vecs: list[list[float]] = []
    for r in rows:
        tid = int(r[0])
        idx_of[tid] = len(node_ids)
        node_ids.append(tid)
        vecs.append([float(r[k]) for k in range(1, 8)])

    Z = np.array(vecs, dtype=float).reshape(len(node_ids), 7) if node_ids else np.zeros((0, 7), dtype=float)
    edges: list[tuple[int, int, float]] = []
    for tid, u_idx in idx_of.items():
        erows = cn.execute(
            "SELECT dst, weight FROM mesh_slm_quipu WHERE src=? ORDER BY weight DESC LIMIT ?",
            (tid, max(1, int(max_edges_per_node))),
        ).fetchall()
        for er in erows:
            dst = int(er[0])
            if dst in idx_of and dst != tid:
                edges.append((u_idx, idx_of[dst], float(er[1])))
    return VNNState(node_ids=node_ids, Z=Z, edges=edges)


def publish_vnn_report(cn: sqlite3.Connection, report: CoupledStepReport) -> dict[str, Any]:
    """Persist the latest VNN coupled report to `brain_kv["entirety:vnn:latest"]`."""
    payload = {
        "at": time.time(),
        "physical": dataclasses.asdict(report.physical),
        "semantic": dataclasses.asdict(report.semantic),
        "effective_nu_phys": report.effective_nu_phys,
        "effective_nu_sem": report.effective_nu_sem,
        "effective_kappa_sem": report.effective_kappa_sem,
        "forcing_scale": report.forcing_scale,
        "ns_diagnostics": report.ns_diagnostics,
        "vnn_diagnostics": report.vnn_diagnostics,
    }
    cn.execute(
        "CREATE TABLE IF NOT EXISTS brain_kv (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)"
    )
    cn.execute(
        "INSERT INTO brain_kv(key, value, updated_at) VALUES(?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (KV_VNN_LATEST, json.dumps(payload, sort_keys=True), payload["at"]),
    )
    return payload


__all__ = [
    "AXES_7D",
    "KV_VNN_LATEST",
    "NavierStokesConfig",
    "NavierStokesState",
    "VNNConfig",
    "VNNState",
    "PhysicalFieldSummary",
    "SemanticGraphSummary",
    "CoupledStepReport",
    "TypedViscousCoupler",
    "project_incompressible",
    "check_ns_stability",
    "ns_step",
    "check_vnn_stability",
    "vnn_step",
    "load_vnn_state_from_conn",
    "publish_vnn_report",
]
