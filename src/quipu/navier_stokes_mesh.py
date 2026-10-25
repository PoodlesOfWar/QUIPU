"""navier_stokes_mesh — 2-D incompressible Navier–Stokes on the mesh's own torus.

The QUIPU MESH places every token on a 64×64 periodic grid (``mesh_slm._TORUS_N``,
cell ``(i, j)`` at angle ``2π·i/N``). That grid is exactly the domain a Fourier
pseudo-spectral Navier–Stokes solver wants: the 2-torus ``[0, 2π)²``. This module
solves the vorticity form of the incompressible equations on it,

    ∂ω/∂t + u·∇ω = ν∇²ω − αω + f(x, y),       ∇²ψ = −ω,
    u = ∂ψ/∂y,   v = −∂ψ/∂x,   ω = ∂v/∂x − ∂u/∂y,

with the mesh supplying the forcing ``f`` — one way only (see below).

Where √−1 enters, and what it does
----------------------------------
In Fourier space every derivative is multiplication by ``i·k``::

    ∂/∂x  →  i·kₓ        ∂/∂y  →  i·k_y        ∇²  →  −|k|²

so the imaginary unit is the *derivative operator*, and the nonlinear term
``u·∇ω`` is a convolution sum over wavevector triads ``p + q = k`` with ``i·q`` in
every term. That is the real mathematical role of √−1 in this equation.

It is also why the repository's earlier framing — the √−1 "fantasy" *resolved to
the science it intersects, 1* — holds here as a checkable fact rather than a
slogan. For a real velocity field the Fourier coefficients are Hermitian
(``ĝ(−k) = conj ĝ(k)``), so every derivative field is real once transformed back:
the imaginary part, the **shadow**, is zero up to round-off. This module does not
assume that: :meth:`SpectralNS2D.imaginary_residue` computes the shadow from the
full complex inverse transform *before* any real part is taken. It is ≈1e-16 on a
dealiased state, and demonstrably **not** zero if the Nyquist mode is differentiated
(see :func:`spectral_derivative`), which is the one place i does not drop out.

Numerics
--------
* **Dealiasing:** Orszag 2/3 rule with cutoff ``(N − 1) // 3`` — strictly below
  ``N/3`` so that no product of two retained modes can alias back into the retained
  band. (``N // 3`` is *not* enough when ``N`` is divisible by 3.)
* **Time stepping:** integrating-factor RK4 — viscosity/drag exactly, the nonlinear
  and forcing terms to fourth order.
* **Verified, not asserted:** the Taylor–Green vortex has an exact solution
  (decay at ``e^{−2νt}``) the solver must reproduce; the pseudo-spectral nonlinear
  term must equal a direct triadic convolution sum; inviscid energy and enstrophy
  must be conserved; ``dE/dt = −2νZ`` must hold. See ``tests/test_navier_stokes_mesh.py``.

The mesh coupling is one-way and read-only
------------------------------------------
:func:`mesh_forcing_field` only *reads* ``mesh_slm_vocab`` and ``mesh_slm_quipu``
(on a connection it opens in ``query_only`` mode), turns incident quipu-edge weight
per torus cell into a zero-mean forcing field, and hands it to the solver. Nothing
flows back: the flow does not touch training, the learning rate, or any table. The
repository's own rule for new signals is *instrument before you couple*; this follows it.

What this is not
----------------
* The mapping "edge weight → forcing vorticity" is a **modelling choice**, not a
  law of nature. The mesh is not a fluid; this gives its connectivity a flow to be
  read through, nothing more.
* This is the **two-dimensional** problem, for which global existence and
  smoothness are classical (Ladyzhenskaya, 1969). It says nothing about the
  three-dimensional regularity question.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

TWO_PI = 2.0 * math.pi
IMAGINARY_UNIT = 1j

#: Side of the mesh's torus. Mirrors ``mesh_slm._TORUS_N``; a test pins the equality
#: so the two cannot drift apart, without importing the (heavy) mesh module here.
MESH_TORUS_N = 64
DEFAULT_N = MESH_TORUS_N


class NavierStokesMeshError(ValueError):
    """Invalid grid, parameter or field."""


# --------------------------------------------------------------------------- #
# Spectral building blocks
# --------------------------------------------------------------------------- #

def wavenumbers(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer wavenumber grids ``(kx, ky)``, shape ``(n, n)``, axis 0 = x, axis 1 = y."""
    k = np.fft.fftfreq(n, d=1.0 / n)  # 0, 1, …, n/2−1, −n/2, …, −1 (exact integers)
    return np.meshgrid(k, k, indexing="ij")


def dealias_cutoff(n: int) -> int:
    """Largest retained |k| per axis: ``(n − 1) // 3``.

    Two retained modes sum to at most ``2·kmax``; that must stay below the first
    wavenumber ``n − kmax`` that aliases back into the band, i.e. ``3·kmax < n``.
    """
    return (int(n) - 1) // 3


def dealias_mask(n: int) -> np.ndarray:
    kx, ky = wavenumbers(n)
    kmax = dealias_cutoff(n)
    return ((np.abs(kx) <= kmax) & (np.abs(ky) <= kmax)).astype(float)


def spectral_derivative(f_hat: np.ndarray, axis: int, *, zero_nyquist: bool = True) -> np.ndarray:
    """Apply ``i·k`` along ``axis`` (0 = x, 1 = y) to Fourier coefficients ``f_hat``.

    For even ``n`` the Nyquist wavenumber ``k = −n/2`` has no conjugate partner, so
    ``i·k`` applied to a real field's Nyquist mode yields a purely imaginary
    physical contribution: the shadow does *not* vanish there. The standard remedy,
    on by default, is to zero the operator at Nyquist. Pass ``zero_nyquist=False``
    to see the shadow survive.
    """
    f_hat = np.asarray(f_hat)
    n = f_hat.shape[axis]
    k = np.fft.fftfreq(n, d=1.0 / n)
    op = IMAGINARY_UNIT * k
    if zero_nyquist and n % 2 == 0:
        op[n // 2] = 0.0
    shape = [1] * f_hat.ndim
    shape[axis] = n
    return f_hat * op.reshape(shape)


def resample_field(field: np.ndarray, n_out: int) -> np.ndarray:
    """Band-limited (Fourier) resampling of a periodic real field to ``n_out`` points.

    Keeps only modes that the target grid can hold *and* dealias; physical values
    are preserved (a constant field stays that constant).
    """
    field = np.asarray(field, dtype=float)
    n_in = field.shape[0]
    if field.shape != (n_in, n_in):
        raise NavierStokesMeshError("field must be square")
    kmax = min(dealias_cutoff(n_out), n_in // 2 - 1)
    ks = np.arange(-kmax, kmax + 1)
    coeff = np.fft.fft2(field) / n_in**2
    out_hat = np.zeros((n_out, n_out), dtype=complex)
    out_hat[np.ix_(ks % n_out, ks % n_out)] = coeff[np.ix_(ks % n_in, ks % n_in)]
    return np.fft.ifft2(out_hat).real * n_out**2


# --------------------------------------------------------------------------- #
# The solver
# --------------------------------------------------------------------------- #

class SpectralNS2D:
    """Pseudo-spectral 2-D incompressible Navier–Stokes on the periodic square.

    State is the vorticity's Fourier coefficients ``ω̂`` (numpy's unnormalized
    ``fft2`` convention). The domain is ``[0, 2π)²`` with integer wavenumbers.
    """

    def __init__(
        self,
        n: int = DEFAULT_N,
        *,
        nu: float = 1e-3,
        drag: float = 0.0,
        forcing: np.ndarray | None = None,
    ) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 8:
            raise NavierStokesMeshError("n must be an integer ≥ 8")
        if not (np.isfinite(nu) and nu >= 0.0):
            raise NavierStokesMeshError("nu must be finite and ≥ 0")
        if not (np.isfinite(drag) and drag >= 0.0):
            raise NavierStokesMeshError("drag must be finite and ≥ 0")
        self.n = n
        self.nu = float(nu)
        self.drag = float(drag)
        self.dx = TWO_PI / n

        self._kx, self._ky = wavenumbers(n)
        self._k2 = self._kx**2 + self._ky**2
        self._k2_safe = np.where(self._k2 == 0.0, 1.0, self._k2)  # ψ̂ = ω̂/k², k=0 → 0
        self._mask = dealias_mask(n)
        self._linear = -(self.nu * self._k2 + self.drag)
        self._cached_dt: float | None = None
        self._E = self._E2 = None

        self._f_hat = np.zeros((n, n), dtype=complex)
        if forcing is not None:
            self.set_forcing(forcing)
        self._w_hat = np.zeros((n, n), dtype=complex)
        self.time = 0.0
        self.steps = 0

    # ---- state in / out -------------------------------------------------- #

    def _check_field(self, field: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(field, dtype=float)
        if arr.shape != (self.n, self.n):
            raise NavierStokesMeshError(f"{name} must have shape ({self.n}, {self.n})")
        if not np.all(np.isfinite(arr)):
            raise NavierStokesMeshError(f"{name} must be finite")
        return arr

    def _to_state(self, field: np.ndarray) -> np.ndarray:
        hat = np.fft.fft2(field) * self._mask
        hat[0, 0] = 0.0  # a periodic velocity field has zero-mean vorticity
        return hat

    def set_vorticity(self, omega: np.ndarray) -> None:
        self._w_hat = self._to_state(self._check_field(omega, "omega"))

    def set_forcing(self, forcing: np.ndarray) -> None:
        """Steady forcing on the vorticity equation (zero-mean, band-limited)."""
        self._f_hat = self._to_state(self._check_field(forcing, "forcing"))

    def vorticity(self) -> np.ndarray:
        return np.fft.ifft2(self._w_hat).real

    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        psi_hat = self._w_hat / self._k2_safe
        u = np.fft.ifft2(spectral_derivative(psi_hat, 1)).real
        v = -np.fft.ifft2(spectral_derivative(psi_hat, 0)).real
        return u, v

    @property
    def state_hat(self) -> np.ndarray:
        return self._w_hat.copy()

    # ---- the nonlinear "integral": a convolution over wavevector triads --- #

    def nonlinear_hat(self, w_hat: np.ndarray) -> np.ndarray:
        """``N̂ = −fft2(u·∇ω) + f̂``, dealiased.

        ``u·∇ω`` is a convolution in Fourier space (``Σ_{p+q=k} û_p·(i q) ω̂_q``).
        Evaluating it in physical space and transforming back is the pseudo-spectral
        route; with the 2/3 rule it equals the exact retained-band convolution,
        which the test suite checks against a direct triadic sum.
        """
        psi_hat = w_hat / self._k2_safe
        fields_hat = np.stack(
            (
                spectral_derivative(psi_hat, 1),    #  u  = ∂ψ/∂y
                -spectral_derivative(psi_hat, 0),   #  v  = −∂ψ/∂x
                spectral_derivative(w_hat, 0),      #  ω_x
                spectral_derivative(w_hat, 1),      #  ω_y
            )
        )
        u, v, wx, wy = np.fft.ifft2(fields_hat, axes=(-2, -1)).real
        return -np.fft.fft2(u * wx + v * wy) * self._mask + self._f_hat

    # ---- time stepping ---------------------------------------------------- #

    def _factors(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_dt != dt:
            self._E = np.exp(self._linear * dt / 2.0)
            self._E2 = self._E**2
            self._cached_dt = dt
        return self._E, self._E2

    def step(self, dt: float) -> None:
        """Advance by ``dt`` with integrating-factor RK4."""
        if not (np.isfinite(dt) and dt > 0.0):
            raise NavierStokesMeshError("dt must be finite and > 0")
        E, E2 = self._factors(dt)
        v = self._w_hat
        k1 = self.nonlinear_hat(v)
        k2 = self.nonlinear_hat(E * (v + 0.5 * dt * k1))
        k3 = self.nonlinear_hat(E * v + 0.5 * dt * k2)
        k4 = self.nonlinear_hat(E2 * v + dt * E * k3)
        # No mask here: the state starts in-band (``_to_state``), ``E`` is diagonal and
        # every ``k`` is in-band, so the update cannot leave the band.
        self._w_hat = E2 * v + (dt / 6.0) * (E2 * k1 + 2.0 * E * (k2 + k3) + k4)
        self.time += dt
        self.steps += 1

    def max_speed(self) -> float:
        u, v = self.velocity()
        return float(np.max(np.hypot(u, v)))

    def stable_dt(self, cfl: float = 0.5) -> float:
        """Advective time-step bound ``cfl · dx / max|u|`` (RK4 is stable to ≈2.8)."""
        speed = self.max_speed()
        return cfl * self.dx / speed if speed > 0.0 else math.inf

    def run(self, t_end: float, dt: float | None = None, *, cfl: float = 0.5, max_steps: int = 1_000_000) -> int:
        """Integrate to ``t_end``. ``dt=None`` picks a CFL-limited step each iteration."""
        if not (np.isfinite(t_end) and t_end >= self.time):
            raise NavierStokesMeshError("t_end must be finite and ≥ the current time")
        taken = 0
        while self.time < t_end - 1e-14 and taken < max_steps:
            h = dt if dt is not None else min(self.stable_dt(cfl), 0.1)
            h = min(h, t_end - self.time)
            self.step(h)
            taken += 1
        if self.time < t_end - 1e-14:
            raise NavierStokesMeshError(f"max_steps={max_steps} reached before t_end")
        return taken

    # ---- diagnostics ------------------------------------------------------ #

    def energy(self) -> float:
        """Domain-mean kinetic energy ``½⟨|u|²⟩``."""
        psi_hat = self._w_hat / self._k2_safe
        return float(0.5 * np.sum(self._k2 * np.abs(psi_hat) ** 2) / self.n**4)

    def enstrophy(self) -> float:
        """Domain-mean enstrophy ``½⟨ω²⟩``."""
        return float(0.5 * np.sum(np.abs(self._w_hat) ** 2) / self.n**4)

    def max_divergence(self) -> float:
        """``max|∇·u|`` computed spectrally from the *physical-space* velocity."""
        u, v = self.velocity()
        div = np.fft.ifft2(spectral_derivative(np.fft.fft2(u), 0) + spectral_derivative(np.fft.fft2(v), 1))
        return float(np.max(np.abs(div)))

    def imaginary_residue(self) -> float:
        """The shadow: largest imaginary part of any derivative field, relative to
        the largest real part, from the full complex inverse transform.

        ≈1e-16 means √−1 has resolved to 1 — the imaginary parts cancelled exactly
        because the state is Hermitian. A value that is *not* tiny would mean the
        state, or an operator, has broken that symmetry.
        """
        psi_hat = self._w_hat / self._k2_safe
        stack = np.stack(
            (
                spectral_derivative(psi_hat, 1),
                spectral_derivative(psi_hat, 0),
                spectral_derivative(self._w_hat, 0),
                spectral_derivative(self._w_hat, 1),
            )
        )
        phys = np.fft.ifft2(stack, axes=(-2, -1))
        scale = float(np.max(np.abs(phys.real)))
        return float(np.max(np.abs(phys.imag)) / scale) if scale > 0.0 else 0.0

    def tail_fraction(self, shell: float = 0.8) -> float:
        """Share of enstrophy in the top spectral shell ``max(|kx|, |ky|) ∈ (shell·kmax, kmax]``.

        The retained band is the *square* ``|kx|, |ky| ≤ kmax`` (corners reach
        ``√2·kmax``), so the shell is measured with the max-norm. A circular
        ``|k| ≤ kmax`` shell would miss every mode in the corners — exactly where
        diagonal under-resolution hides.
        """
        kmax = dealias_cutoff(self.n)
        kinf = np.maximum(np.abs(self._kx), np.abs(self._ky))
        total = float(np.sum(np.abs(self._w_hat) ** 2))
        if total == 0.0:
            return 0.0
        top = (kinf > shell * kmax) & (kinf <= kmax)
        return float(np.sum(np.abs(self._w_hat[top]) ** 2) / total)

    def is_resolved(self, tol: float = 1e-6) -> bool:
        """Spectral-tail test: the smallest retained scales must carry negligible
        enstrophy, otherwise the flow has structure the grid cannot hold."""
        return self.tail_fraction() <= tol

    def diagnostics(self) -> dict[str, float | bool | int]:
        return {
            "time": self.time,
            "steps": self.steps,
            "energy": self.energy(),
            "enstrophy": self.enstrophy(),
            "max_speed": self.max_speed(),
            "max_divergence": self.max_divergence(),
            "imaginary_residue": self.imaginary_residue(),
            "tail_fraction": self.tail_fraction(),
            "resolved": self.is_resolved(),
        }


# --------------------------------------------------------------------------- #
# Mesh -> forcing (one way, read only)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MeshForcing:
    field: np.ndarray        # (n, n), zero-mean, rms == amplitude (or all zero)
    source: str              # "mesh" | "empty"
    n_tokens: int
    n_edges: int
    n_skipped: int           # tokens whose (i, j) lay outside the torus


#: Default vortex-blob radius, in torus cells. Measured, not guessed: with the
#: 4-token reference mesh at n=64, ν=0.02, α=0.2, amplitude 1, t=40, the share of
#: enstrophy in the top spectral shell falls from 1.4e-2 (a point source) to 1.0e-7
#: at 2 cells, inside the 1e-6 resolution tolerance. It is not a licence: at
#: amplitude 2 the same blob leaves 1.7e-6 and ``is_resolved()`` correctly says no.
#: Stronger forcing needs a finer grid; check ``diagnostics()["resolved"]``.
DEFAULT_BLOB_RADIUS = 2.0


def _gaussian_blob(grid: np.ndarray, radius_cells: float) -> np.ndarray:
    """Convolve with a Gaussian of standard deviation ``radius_cells`` torus cells.

    Done in Fourier space (where the Gaussian is ``exp(−σ²k²/2)``, and where i·k
    again does the work), so it is exact on the periodic grid.
    """
    if radius_cells == 0.0:
        return grid
    n = grid.shape[0]
    kx, ky = wavenumbers(n)
    sigma = radius_cells * TWO_PI / n
    return np.fft.ifft2(np.fft.fft2(grid) * np.exp(-0.5 * sigma**2 * (kx**2 + ky**2))).real


def forcing_from_edges(
    cells: Mapping[int, tuple[int, int]],
    edges: Iterable[tuple[int, int, float]],
    *,
    n: int = DEFAULT_N,
    amplitude: float = 1.0,
    torus_n: int = MESH_TORUS_N,
    blob_radius: float = DEFAULT_BLOB_RADIUS,
) -> MeshForcing:
    """Build a forcing field from the quipu graph, with no database involved.

    Each edge's weight is credited to the torus cell of *both* its endpoints, giving
    the weighted degree of the token occupying each cell. Each token then becomes a
    Gaussian **vortex blob** of ``blob_radius`` cells rather than a point: a point
    source is singular and forces the grid scale directly, which leaves the flow
    under-resolved (see :data:`DEFAULT_BLOB_RADIUS`). The field is made zero-mean,
    scaled so its root-mean-square is ``amplitude``, and resampled (band-limited) to
    the solver's grid. Weights that are not finite are ignored.
    """
    if not (np.isfinite(amplitude) and amplitude >= 0.0):
        raise NavierStokesMeshError("amplitude must be finite and ≥ 0")
    if not (np.isfinite(blob_radius) and blob_radius >= 0.0):
        raise NavierStokesMeshError("blob_radius must be finite and ≥ 0")
    grid = np.zeros((torus_n, torus_n))
    skipped = 0
    for i, j in cells.values():
        if not (0 <= i < torus_n and 0 <= j < torus_n):
            skipped += 1
    n_edges = 0
    for src, dst, weight in edges:
        w = float(weight)
        if not math.isfinite(w):
            continue
        n_edges += 1
        for token in (src, dst):
            cell = cells.get(token)
            if cell is not None and 0 <= cell[0] < torus_n and 0 <= cell[1] < torus_n:
                grid[cell[0], cell[1]] += w
    grid = _gaussian_blob(grid, float(blob_radius))
    raw_scale = float(np.max(np.abs(grid)))
    grid -= grid.mean()
    rms = float(np.sqrt(np.mean(grid**2)))
    # A uniform mesh leaves only round-off after mean removal. Judge "is there any
    # structure left" relative to the field's own scale, so that a mesh whose weights
    # are all tiny is still a mesh (the forcing is rescaled to a fixed rms anyway).
    if rms <= 1e-12 * raw_scale:
        rms = 0.0
    if rms == 0.0:
        return MeshForcing(np.zeros((n, n)), "empty", len(cells), n_edges, skipped)
    field = resample_field(grid * (amplitude / rms), n) if n != torus_n else grid * (amplitude / rms)
    return MeshForcing(field - field.mean(), "mesh", len(cells), n_edges, skipped)


def mesh_forcing_field(
    cn: Any = None, *, n: int = DEFAULT_N, amplitude: float = 1.0, blob_radius: float = DEFAULT_BLOB_RADIUS
) -> MeshForcing:
    """Read the quipu graph from the mesh database and turn it into a forcing field.

    **Read only.** When this function opens the connection itself it sets
    ``PRAGMA query_only = ON``, so a write is impossible even by mistake; a
    connection passed in by the caller is used with ``SELECT`` statements only. A
    missing or empty mesh yields a zero forcing (``source == "empty"``), not an error.
    """
    import sqlite3

    owned = cn is None
    if owned:
        from .local_store import open_conn  # lazy: keeps this module light to import
        cn = open_conn()
        cn.execute("PRAGMA query_only = ON")
    try:
        try:
            cells = {int(r[0]): (int(r[1]), int(r[2])) for r in cn.execute(
                "SELECT token_id, i, j FROM mesh_slm_vocab")}
            edges = [(int(r[0]), int(r[1]), float(r[2])) for r in cn.execute(
                "SELECT src, dst, weight FROM mesh_slm_quipu")]
        except sqlite3.OperationalError:  # tables not created yet: an empty mesh
            return MeshForcing(np.zeros((n, n)), "empty", 0, 0, 0)
        return forcing_from_edges(cells, edges, n=n, amplitude=amplitude, blob_radius=blob_radius)
    finally:
        if owned:
            cn.close()


def simulate_from_mesh(
    t_end: float,
    *,
    cn: Any = None,
    n: int = DEFAULT_N,
    nu: float = 1e-2,
    drag: float = 0.1,
    amplitude: float = 1.0,
    blob_radius: float = DEFAULT_BLOB_RADIUS,
    dt: float | None = None,
    cfl: float = 0.5,
) -> dict[str, Any]:
    """Drive the flow from rest with the mesh's forcing and report what it did.

    ``drag`` is a linear (Ekman) friction that lets the forced flow settle instead
    of accumulating energy without bound; it is a modelling parameter, set it to 0
    to remove it. The mesh is only read; the returned dict is the sole output.
    """
    forcing = mesh_forcing_field(cn, n=n, amplitude=amplitude, blob_radius=blob_radius)
    solver = SpectralNS2D(n, nu=nu, drag=drag, forcing=forcing.field)
    solver.run(t_end, dt=dt, cfl=cfl)
    return {
        "forcing_source": forcing.source,
        "n_tokens": forcing.n_tokens,
        "n_edges": forcing.n_edges,
        "diagnostics": solver.diagnostics(),
        "solver": solver,
    }


__all__ = [
    "DEFAULT_BLOB_RADIUS",
    "DEFAULT_N",
    "IMAGINARY_UNIT",
    "MESH_TORUS_N",
    "MeshForcing",
    "NavierStokesMeshError",
    "SpectralNS2D",
    "dealias_cutoff",
    "dealias_mask",
    "forcing_from_edges",
    "mesh_forcing_field",
    "resample_field",
    "simulate_from_mesh",
    "spectral_derivative",
    "wavenumbers",
]
