"""navier_stokes_3d — forced 3-D incompressible Navier–Stokes on the periodic cube,
and the instrument that watches any flow for the loss of its own resolution.

Same method as :mod:`navier_stokes_mesh`, one dimension up. On ``[0, 2π)³`` the
velocity form is, with the pressure folded into the Leray projection ``P``::

    ∂u/∂t = P[ u × ω ] + ν∇²u + f(x, t),     ω = ∇×u = (i k) × û,     ∇·u = 0

(``(u·∇)u = ω × u + ∇(|u|²/2)``; the gradient is absorbed by the pressure, so the
rotation form needs only ``u`` and ``ω``.) √−1 is again the derivative: the curl
is ``i k ×``, the Laplacian ``−|k|²``, the projector ``I − k kᵀ/|k|²``. The shadow —
the imaginary part left in a derivative field — is measured, not assumed, exactly as
in 2-D: :meth:`SpectralNS3D.imaginary_residue`.

What is verified (``tests/test_navier_stokes_3d.py``), against independent truths
rather than itself:

* a Beltrami (ABC) flow decays exactly as ``e^{−νt}``, with a time-dependent force
  that keeps it exactly on a prescribed amplitude;
* a 2-D flow embedded in 3-D must agree, to round-off, with the 2-D solver
  :class:`~quipu.navier_stokes_mesh.SpectralNS2D` — an independently verified code;
* the nonlinear term equals a direct triadic convolution sum;
* energy and helicity are conserved inviscidly, ``dE/dt = −2νZ``, fourth-order time
  convergence, ``∇·u = 0``.

The resolution instrument
-------------------------
A blow-up, if there is one, is from the numerics' point of view a collapse of scale:
whatever grid you pick, the solution eventually has structure finer than it. What can
be done is to *watch the spectrum and say when that will happen*. For a solution that
is analytic in a strip of half-width ``δ(t)``, the energy spectrum decays like
``E(k) ~ k^b e^{−2δk}`` (Sulem–Sulem–Frisch), so an ``n``³ grid is resolved
only while ``δ`` stays above a threshold set by its cutoff, and a singularity at ``T*``
is a strip that closes: ``δ(t) → 0``. :func:`analyticity_width` fits ``δ``; :func:`required_width` is the
``δ`` a grid needs; :func:`resolution_forecast` extrapolates a ``δ(t)`` series to the
time the grid is lost and to the singularity time.

What this does not do
---------------------
* It does not decide whether a singularity exists. A shrinking ``δ`` is consistent
  with blow-up and with a sharp but finite event; only a proof settles it.
* Its model is wrong for **smooth, non-analytic** forcing. A compactly supported
  ``C^∞`` force cannot be analytic, so the spectrum of the response is Gevrey, not
  exponential; :func:`decay_exponent` fits the exponent so that mismatch is visible
  rather than hidden inside a misleading ``δ``.
* It says nothing about the 2-D mesh; the mesh lives on a 2-D torus and
  :mod:`navier_stokes_mesh` is its solver.
"""
from __future__ import annotations

import math
from typing import Callable, Union

import numpy as np

from .navier_stokes_mesh import NavierStokesMeshError, TWO_PI, dealias_cutoff

Forcing = Union[np.ndarray, Callable[[float], np.ndarray]]


# --------------------------------------------------------------------------- #
# Fourier grid
# --------------------------------------------------------------------------- #

def wavenumbers3(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer wavenumber grids ``(kx, ky, kz)``; axis 0 is x, 1 is y, 2 is z."""
    k = np.fft.fftfreq(n, d=1.0 / n)
    return tuple(np.meshgrid(k, k, k, indexing="ij"))  # type: ignore[return-value]


def dealias_mask3(n: int) -> np.ndarray:
    """2/3-rule box mask: ``max(|kx|, |ky|, |kz|) ≤ (n − 1) // 3``."""
    kmax = dealias_cutoff(n)
    kx, ky, kz = wavenumbers3(n)
    return ((np.abs(kx) <= kmax) & (np.abs(ky) <= kmax) & (np.abs(kz) <= kmax)).astype(float)


# --------------------------------------------------------------------------- #
# The solver
# --------------------------------------------------------------------------- #

class SpectralNS3D:
    """Pseudo-spectral 3-D incompressible Navier–Stokes on ``[0, 2π)³``.

    State is the velocity's Fourier coefficients ``û`` of shape ``(3, n, n, n)``
    (numpy's unnormalized ``fft`` convention), always dealiased, divergence-free and
    zero-mean. ``forcing`` is either a steady ``(3, n, n, n)`` array or a callable
    ``f(t) -> (3, n, n, n)`` evaluated at each Runge–Kutta stage time.
    """

    def __init__(self, n: int = 32, *, nu: float = 1e-2, forcing: Forcing | None = None) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 8:
            raise NavierStokesMeshError("n must be an integer ≥ 8")
        if not (np.isfinite(nu) and nu >= 0.0):
            raise NavierStokesMeshError("nu must be finite and ≥ 0")
        self.n = n
        self.nu = float(nu)
        self.dx = TWO_PI / n

        self._k = np.stack(wavenumbers3(n))                       # (3, n, n, n)
        self._k2 = np.sum(self._k**2, axis=0)
        self._k2_safe = np.where(self._k2 == 0.0, 1.0, self._k2)
        self._mask = dealias_mask3(n)
        self._linear = -self.nu * self._k2
        self._cached_dt: float | None = None
        self._E = self._E2 = None

        self._f_steady = np.zeros((3, n, n, n), dtype=complex)
        self._f_fn: Callable[[float], np.ndarray] | None = None
        if forcing is not None:
            self.set_forcing(forcing)
        self._u_hat = np.zeros((3, n, n, n), dtype=complex)
        self.time = 0.0
        self.steps = 0

    # ---- state in / out -------------------------------------------------- #

    def _check_vector_field(self, field: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(field, dtype=float)
        if arr.shape != (3, self.n, self.n, self.n):
            raise NavierStokesMeshError(f"{name} must have shape (3, {self.n}, {self.n}, {self.n})")
        if not np.all(np.isfinite(arr)):
            raise NavierStokesMeshError(f"{name} must be finite")
        return arr

    def _project(self, v_hat: np.ndarray) -> np.ndarray:
        """Leray projection onto divergence-free fields: ``v̂ − k (k·v̂)/|k|²``."""
        return v_hat - self._k * (np.sum(self._k * v_hat, axis=0) / self._k2_safe)

    def _to_state(self, field: np.ndarray) -> np.ndarray:
        hat = self._project(np.fft.fftn(field, axes=(1, 2, 3)) * self._mask)
        hat[:, 0, 0, 0] = 0.0                     # no mean flow
        return hat

    def set_velocity(self, u: np.ndarray) -> None:
        """Set the velocity; the part that is not divergence-free is projected out."""
        self._u_hat = self._to_state(self._check_vector_field(u, "u"))

    def set_forcing(self, forcing: Forcing) -> None:
        """Steady array, or ``f(t)`` returning one. Either is projected and dealiased."""
        if callable(forcing):
            self._f_fn = forcing
            self._f_steady = np.zeros((3, self.n, self.n, self.n), dtype=complex)
        else:
            self._f_fn = None
            self._f_steady = self._to_state(self._check_vector_field(forcing, "forcing"))

    def _forcing_hat(self, t: float) -> np.ndarray:
        if self._f_fn is None:
            return self._f_steady
        return self._to_state(self._check_vector_field(self._f_fn(t), "forcing(t)"))

    def velocity(self) -> np.ndarray:
        return np.fft.ifftn(self._u_hat, axes=(1, 2, 3)).real

    def vorticity_hat(self, u_hat: np.ndarray | None = None) -> np.ndarray:
        """``ω̂ = (i k) × û`` — the curl, where √−1 does its work."""
        u_hat = self._u_hat if u_hat is None else u_hat
        return np.cross(1j * self._k, u_hat, axis=0)

    def vorticity(self) -> np.ndarray:
        return np.fft.ifftn(self.vorticity_hat(), axes=(1, 2, 3)).real

    @property
    def state_hat(self) -> np.ndarray:
        return self._u_hat.copy()

    # ---- the nonlinear term ---------------------------------------------- #

    def nonlinear_hat(self, u_hat: np.ndarray, t: float = 0.0) -> np.ndarray:
        """``N̂ = P[fft(u × ω)] + f̂(t)``, dealiased."""
        both = np.stack((u_hat, self.vorticity_hat(u_hat)))
        u, w = np.fft.ifftn(both, axes=(2, 3, 4)).real
        cross_hat = np.fft.fftn(np.cross(u, w, axis=0), axes=(1, 2, 3)) * self._mask
        return self._project(cross_hat) + self._forcing_hat(t)

    # ---- time stepping ---------------------------------------------------- #

    def _factors(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_dt != dt:
            self._E = np.exp(self._linear * dt / 2.0)
            self._E2 = self._E**2
            self._cached_dt = dt
        return self._E, self._E2

    def step(self, dt: float) -> None:
        """Advance by ``dt`` with integrating-factor RK4 (forcing sampled at stage times)."""
        if not (np.isfinite(dt) and dt > 0.0):
            raise NavierStokesMeshError("dt must be finite and > 0")
        E, E2 = self._factors(dt)
        v, t = self._u_hat, self.time
        k1 = self.nonlinear_hat(v, t)
        k2 = self.nonlinear_hat(E * (v + 0.5 * dt * k1), t + 0.5 * dt)
        k3 = self.nonlinear_hat(E * v + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.nonlinear_hat(E2 * v + dt * E * k3, t + dt)
        self._u_hat = E2 * v + (dt / 6.0) * (E2 * k1 + 2.0 * E * (k2 + k3) + k4)
        self.time += dt
        self.steps += 1

    def max_speed(self) -> float:
        """Largest ``|u₁| + |u₂| + |u₃|`` — the quantity the 3-D CFL bound uses."""
        return float(np.max(np.sum(np.abs(self.velocity()), axis=0)))

    def stable_dt(self, cfl: float = 0.5) -> float:
        speed = self.max_speed()
        return cfl * self.dx / speed if speed > 0.0 else math.inf

    def run(self, t_end: float, dt: float | None = None, *, cfl: float = 0.5, max_steps: int = 1_000_000) -> int:
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
        return float(0.5 * np.sum(np.abs(self._u_hat) ** 2) / self.n**6)

    def enstrophy(self) -> float:
        """Domain-mean enstrophy ``½⟨|ω|²⟩`` (so that ``dE/dt = −2νZ``)."""
        return float(0.5 * np.sum(self._k2 * np.sum(np.abs(self._u_hat) ** 2, axis=0)) / self.n**6)

    def helicity(self) -> float:
        """Domain-mean helicity ``⟨u·ω⟩`` — conserved by the inviscid, unforced flow."""
        w_hat = self.vorticity_hat()
        return float(np.sum(np.real(np.conj(self._u_hat) * w_hat)) / self.n**6)

    def max_divergence(self) -> float:
        """``max|∇·u|`` computed spectrally from the *physical-space* velocity."""
        u_hat = np.fft.fftn(self.velocity(), axes=(1, 2, 3))
        div = np.fft.ifftn(np.sum(1j * self._k * u_hat, axis=0))
        return float(np.max(np.abs(div)))

    def imaginary_residue(self) -> float:
        """The shadow: largest imaginary part of any velocity-gradient component,
        relative to the largest real part, from the full complex inverse transform.
        Round-off (~1e-16) when the state is Hermitian; large if it is not."""
        grad_hat = (1j * self._k)[None, :] * self._u_hat[:, None]          # ∂_j u_i, (3,3,n,n,n)
        phys = np.fft.ifftn(grad_hat, axes=(2, 3, 4))
        scale = float(np.max(np.abs(phys.real)))
        return float(np.max(np.abs(phys.imag)) / scale) if scale > 0.0 else 0.0

    def tail_fraction(self, shell: float = 0.8) -> float:
        """Share of enstrophy in the top shell ``max|k_i| ∈ (shell·kmax, kmax]``
        (max-norm: the retained band is a cube whose corners reach ``√3·kmax``)."""
        kmax = dealias_cutoff(self.n)
        weight = self._k2 * np.sum(np.abs(self._u_hat) ** 2, axis=0)
        total = float(np.sum(weight))
        if total == 0.0:
            return 0.0
        kinf = np.max(np.abs(self._k), axis=0)
        return float(np.sum(weight[(kinf > shell * kmax) & (kinf <= kmax)]) / total)

    def is_resolved(self, tol: float = 1e-6) -> bool:
        return self.tail_fraction() <= tol

    def energy_spectrum(self, *, per_mode: bool = False) -> np.ndarray:
        """Energy in spherical shells ``round(|k|) = 0, 1, …, n // 2``.

        ``per_mode=True`` divides each shell by its number of modes, which removes the
        lattice-point count (≈ 4πk²) from the spectrum.
        """
        shell = np.rint(np.sqrt(self._k2)).astype(int).ravel()
        power = (0.5 * np.sum(np.abs(self._u_hat) ** 2, axis=0) / self.n**6).ravel()
        size = self.n // 2 + 1
        keep = shell < size
        spec = np.bincount(shell[keep], weights=power[keep], minlength=size)
        if per_mode:
            count = np.bincount(shell[keep], minlength=size)
            spec = np.divide(spec, count, out=np.zeros_like(spec), where=count > 0)
        return spec

    def analyticity_width(self, k_lo: int | None = None, k_hi: int | None = None) -> float:
        """Strip half-width ``δ`` fitted to this state's **per-mode** energy spectrum.

        Per-mode (shell sum ÷ number of modes) rather than shell sum, because the sum
        carries the lattice-point count ``≈ 4πk²`` that a ``b ln k`` term only
        approximately absorbs (a ~6% bias on a test spectrum); per-mode is exact for an
        isotropic envelope. Default window: ``kmax//3 … kmax``.
        """
        kmax = dealias_cutoff(self.n)
        return analyticity_width(self.energy_spectrum(per_mode=True), k_lo if k_lo is not None else max(2, kmax // 3),
                                 k_hi if k_hi is not None else kmax)

    def diagnostics(self) -> dict[str, float | bool | int]:
        return {
            "time": self.time,
            "steps": self.steps,
            "energy": self.energy(),
            "enstrophy": self.enstrophy(),
            "helicity": self.helicity(),
            "max_speed": self.max_speed(),
            "max_divergence": self.max_divergence(),
            "imaginary_residue": self.imaginary_residue(),
            "tail_fraction": self.tail_fraction(),
            "resolved": self.is_resolved(),
        }


# --------------------------------------------------------------------------- #
# The resolution instrument: how fast is the spectrum's decay closing?
# --------------------------------------------------------------------------- #

def _fit_window(spectrum: np.ndarray, k_lo: int, k_hi: int) -> tuple[np.ndarray, np.ndarray]:
    spec = np.asarray(spectrum, dtype=float)
    if spec.ndim != 1 or not np.all(np.isfinite(spec)):
        raise NavierStokesMeshError("spectrum must be a finite 1-D array indexed by k")
    if not (isinstance(k_lo, (int, np.integer)) and isinstance(k_hi, (int, np.integer))) or isinstance(k_lo, bool):
        raise NavierStokesMeshError("k_lo and k_hi must be integers")
    if not (1 <= k_lo < k_hi < len(spec)):
        raise NavierStokesMeshError("need 1 ≤ k_lo < k_hi < len(spectrum)")
    k = np.arange(k_lo, k_hi + 1, dtype=float)
    e = spec[k_lo:k_hi + 1]
    ok = e > 0.0
    if int(ok.sum()) < 4:
        raise NavierStokesMeshError("need at least 4 positive spectrum values in the fit window")
    return k[ok], np.log(e[ok])


def analyticity_fit(spectrum: np.ndarray, k_lo: int, k_hi: int) -> tuple[float, float]:
    """Fit ``ln E(k) = a + b ln k − 2δ k`` on ``k ∈ [k_lo, k_hi]``; return ``(δ, b)``."""
    k, y = _fit_window(spectrum, k_lo, k_hi)
    design = np.column_stack((np.ones_like(k), np.log(k), -2.0 * k))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[2]), float(coef[1])


def analyticity_width(spectrum: np.ndarray, k_lo: int, k_hi: int) -> float:
    """The strip half-width ``δ`` of :func:`analyticity_fit`.

    Negative or ~0 means the spectrum is not exponentially decaying in the window
    (see :func:`decay_exponent`).
    """
    return analyticity_fit(spectrum, k_lo, k_hi)[0]


def decay_exponent(spectrum: np.ndarray, k_lo: int, k_hi: int) -> tuple[float, float]:
    """Fit ``ln E(k) = a + b ln k − c·k^θ`` over ``θ ∈ [0.2, 1]``; return ``(θ, c)``.

    ``θ = 1`` is analytic (exponential decay); ``θ < 1`` is Gevrey, the class a smooth
    but non-analytic force produces. If ``θ`` is well below 1 the strip width from
    :func:`analyticity_width` is the wrong model and should not be extrapolated.
    """
    k, y = _fit_window(spectrum, k_lo, k_hi)
    best: tuple[float, float, float] | None = None
    for theta in np.linspace(0.2, 1.0, 81):
        design = np.column_stack((np.ones_like(k), np.log(k), -(k**theta)))
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = float(np.sum((design @ coef - y) ** 2))
        if best is None or resid < best[0]:
            best = (resid, float(theta), float(coef[2]))
    assert best is not None
    return best[1], best[2]


def _model_tail_share(delta: float, kk: np.ndarray, top: np.ndarray, b: float) -> float:
    """Top-shell share of the model state's enstrophy, mode by mode: amplitude²
    ``|k|^b e^{−2δ|k|}`` per mode, enstrophy weight ``|k|²`` times that."""
    w = np.exp((b + 2.0) * np.log(kk) - 2.0 * delta * kk)
    return float(np.sum(w[top]) / np.sum(w))


def required_width(n: int, tol: float = 1e-6, *, b: float = -2.0, shell: float = 0.8) -> float:
    """The strip half-width ``δ`` at which an ``n``³ grid stops being resolved.

    Evaluated on the solver's own lattice, with its own test: the retained modes
    ``max|k_i| ≤ kmax = (n−1)//3`` carry per-mode energy ``|k|^b e^{−2δ|k|}``, and
    ``δ`` is found by bisection so that the share of enstrophy with
    ``max|k_i| ∈ (shell·kmax, kmax]`` equals ``tol`` — exactly what
    :meth:`SpectralNS3D.tail_fraction` / :meth:`~SpectralNS3D.is_resolved` measure.
    The share falls monotonically as ``δ`` grows, so the answer is unique. Pass the
    ``b`` that :func:`analyticity_fit` reports (per-mode spectrum); the default is a
    typical value.
    """
    if not (isinstance(n, (int, np.integer)) and n >= 8) or isinstance(n, bool):
        raise NavierStokesMeshError("n must be an integer ≥ 8")
    if not (0.0 < tol < 1.0) or not (0.0 < shell < 1.0) or not np.isfinite(b):
        raise NavierStokesMeshError("need tol, shell in (0, 1) and a finite b")
    kmax = dealias_cutoff(int(n))
    kx, ky, kz = wavenumbers3(int(n))
    kinf = np.maximum(np.maximum(np.abs(kx), np.abs(ky)), np.abs(kz))
    keep = (kinf <= kmax) & ((kx != 0) | (ky != 0) | (kz != 0))
    kk = np.sqrt(kx**2 + ky**2 + kz**2)[keep]
    top = (kinf[keep] > shell * kmax)
    if not top.any() or top.all():
        raise NavierStokesMeshError("shell leaves no modes on one side of the cut")
    lo, hi = 0.0, 1.0
    for _ in range(80):
        if _model_tail_share(hi, kk, top, b) <= tol:
            break
        hi *= 2.0
    else:
        raise NavierStokesMeshError("no finite width reaches that tolerance")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _model_tail_share(mid, kk, top, b) > tol:
            lo = mid
        else:
            hi = mid
    return hi


def resolution_forecast(
    times: np.ndarray, widths: np.ndarray, n: int, tol: float = 1e-6, *, b: float = -2.0
) -> dict[str, float | bool | None]:
    """Extrapolate a measured ``δ(t)`` series.

    Fits a straight line ``δ(t) = s·(T* − t)`` (the standard practice for a closing
    strip) and reports

    * ``collapsing`` — the slope is negative;
    * ``t_singularity`` — where the line reaches 0 (``None`` if not collapsing);
    * ``t_resolution_lost`` — where it falls to :func:`required_width` (give the fitted ``b``);
    * ``already_lost`` — the latest measured ``δ`` is already below the requirement;
    * ``r_squared`` — how straight the series is; a poor value means do not trust it.

    This extrapolates a trend. It does not prove anything about the equations.
    """
    t = np.asarray(times, dtype=float)
    d = np.asarray(widths, dtype=float)
    if t.ndim != 1 or t.shape != d.shape or len(t) < 3:
        raise NavierStokesMeshError("times and widths must be 1-D, equal length, with at least 3 points")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(d))) or np.any(np.diff(t) <= 0.0):
        raise NavierStokesMeshError("times must be finite and strictly increasing; widths finite")
    need = required_width(n, tol, b=b)
    slope, intercept = np.polyfit(t, d, 1)
    fitted = slope * t + intercept
    ss_tot = float(np.sum((d - d.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - float(np.sum((d - fitted) ** 2)) / ss_tot
    collapsing = bool(slope < 0.0)
    return {
        "collapsing": collapsing,
        "slope": float(slope),
        "t_singularity": float(-intercept / slope) if collapsing else None,
        "required_width": need,
        "t_resolution_lost": float((need - intercept) / slope) if collapsing else None,
        "already_lost": bool(d[-1] < need),
        "r_squared": r2,
    }


__all__ = [
    "SpectralNS3D",
    "analyticity_fit",
    "analyticity_width",
    "dealias_mask3",
    "decay_exponent",
    "required_width",
    "resolution_forecast",
    "wavenumbers3",
]
