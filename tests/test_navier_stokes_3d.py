"""3-D Navier–Stokes and the resolution instrument: verified against exact answers.

Independent ground truths only: the Beltrami (ABC) flow's exact decay, a time-dependent
force that pins a prescribed amplitude, a steady shear whose nonlinear term is a pure
gradient, a 2-D flow embedded in 3-D (which must match the separately verified 2-D
solver), a direct triadic convolution, conservation laws, the energy budget, and the
order of the integrator. The instrument is checked on spectra built with known strip
width, and on a synthetic collapse with a known singularity time. No network.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.quipu import navier_stokes_3d as ns3
from src.quipu.navier_stokes_3d import (
    SpectralNS3D,
    analyticity_fit,
    analyticity_width,
    decay_exponent,
    dealias_mask3,
    required_width,
    resolution_forecast,
    wavenumbers3,
)
from src.quipu.navier_stokes_mesh import NavierStokesMeshError, SpectralNS2D, dealias_cutoff


def grid3(n):
    x = np.arange(n) * 2 * math.pi / n
    return np.meshgrid(x, x, x, indexing="ij")


def abc_flow(n, A=1.0, B=0.7, C=0.43):
    """Beltrami flow: ∇×u = u exactly, so u×ω ≡ 0 and the flow only decays."""
    X, Y, Z = grid3(n)
    return np.stack((A * np.sin(Z) + C * np.cos(Y), B * np.sin(X) + A * np.cos(Z), C * np.sin(Y) + B * np.cos(X)))


# First in the file on purpose: if time stopped advancing, every ``run`` below would
# spin to its step budget, so this must fail before any of them is reached.
def test_a_step_advances_time_and_the_step_count():
    s = SpectralNS3D(8, nu=0.01)
    s.step(0.25)
    s.step(0.125)
    assert s.time == pytest.approx(0.375, abs=1e-15) and s.steps == 2


# ═══════════════════════════ the grid ═══════════════════════════════════════

def test_wavenumbers_are_exact_integers_with_x_on_axis_0():
    kx, ky, kz = wavenumbers3(8)
    assert np.array_equal(kx[:, 0, 0], np.array([0, 1, 2, 3, -4, -3, -2, -1]))
    assert np.array_equal(ky[0, :, 0], kx[:, 0, 0]) and np.array_equal(kz[0, 0, :], kx[:, 0, 0])
    assert np.all(kx == np.rint(kx))


@pytest.mark.parametrize("n", [8, 12, 16, 32])
def test_the_dealias_mask_is_the_box_below_n_over_3_and_symmetric(n):
    kmax = dealias_cutoff(n)
    m = dealias_mask3(n)
    assert m.sum() == (2 * kmax + 1) ** 3
    assert m[0, 0, 0] == 1.0
    kx, ky, kz = wavenumbers3(n)
    assert np.all((m == 1.0) == ((np.abs(kx) <= kmax) & (np.abs(ky) <= kmax) & (np.abs(kz) <= kmax)))
    flipped = m[(-np.arange(n)) % n][:, (-np.arange(n)) % n][:, :, (-np.arange(n)) % n]
    assert np.array_equal(m, flipped)


# ═══════════════════════════ √−1 as the curl ════════════════════════════════

def test_the_curl_is_i_k_cross_u_and_an_abc_flow_is_its_own_curl():
    n = 16
    s = SpectralNS3D(n)
    u = abc_flow(n)
    s.set_velocity(u)
    assert np.max(np.abs(s.vorticity() - u)) < 1e-13


def test_the_curl_has_the_right_orientation():
    """u = (−sin y, 0, 0) has ω_z = ∂x u_y − ∂y u_x = +cos y."""
    n = 16
    X, Y, Z = grid3(n)
    s = SpectralNS3D(n)
    s.set_velocity(np.stack((-np.sin(Y), 0 * X, 0 * X)))
    w = s.vorticity()
    assert np.max(np.abs(w[2] - np.cos(Y))) < 1e-13 and np.max(np.abs(w[:2])) < 1e-13


# ═══════════════════════════ exact solutions ════════════════════════════════

@pytest.mark.parametrize("nu", [0.0, 0.05, 0.3])
def test_an_abc_flow_decays_exactly_as_exp_minus_nu_t(nu):
    n = 16
    u0 = abc_flow(n)
    s = SpectralNS3D(n, nu=nu)
    s.set_velocity(u0)
    s.run(1.0, dt=0.01)
    assert np.max(np.abs(s.velocity() - u0 * math.exp(-nu * s.time))) < 1e-13


def test_a_time_dependent_force_holds_an_abc_flow_on_a_prescribed_amplitude():
    """u = a(t)·ABC with a = 1 + t² solves the equation exactly when f = (a' + νa)·ABC.
    The integrator is fourth order, not exact, so the error is O(dt⁴): ~1e-10 at
    dt = 0.05 and 16× smaller per halving. A force sampled at the wrong times would be
    O(dt) — five orders of magnitude worse."""
    n, nu = 16, 0.1
    u0 = abc_flow(n)

    def err(dt):
        s = SpectralNS3D(n, nu=nu, forcing=lambda t: (2 * t + nu * (1 + t * t)) * u0)
        s.set_velocity(u0)
        s.run(1.0, dt=dt)
        return np.max(np.abs(s.velocity() - 2.0 * u0))

    e1, e2 = err(0.1), err(0.05)
    assert e2 < 1e-9 and e1 / e2 > 12


def test_the_force_is_sampled_at_the_runge_kutta_stage_times():
    seen: list[float] = []

    def f(t):
        seen.append(t)
        return np.zeros((3, 8, 8, 8))

    s = SpectralNS3D(8, forcing=f)
    s.step(0.2)
    s.step(0.2)
    assert seen == pytest.approx([0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4], abs=1e-15)


def test_a_steady_shear_is_an_exact_steady_state_though_its_nonlinear_term_is_nonzero():
    """u = (sin y, 0, 0), f = νu. Here u×ω = (0, sin y cos y, 0) ≠ 0 — a pure gradient the
    pressure must absorb. Only a correct Leray projection leaves the flow steady."""
    n, nu = 16, 0.05
    X, Y, Z = grid3(n)
    u0 = np.stack((np.sin(Y), 0 * X, 0 * X))
    s = SpectralNS3D(n, nu=nu, forcing=nu * u0)
    s.set_velocity(u0)
    raw = np.cross(s.velocity(), s.vorticity(), axis=0)
    assert np.max(np.abs(raw)) > 0.4                        # the gradient is really there
    s.run(2.0, dt=0.02)
    assert np.max(np.abs(s.velocity() - u0)) < 1e-13


@pytest.mark.parametrize("nu", [0.0, 0.02])
def test_a_2d_flow_embedded_in_3d_matches_the_verified_2d_solver(nu):
    """An x-y flow (u, v, 0) independent of z must reproduce the 2-D solver's vorticity:
    this drives the 3-D nonlinear term on a genuinely nonlinear, non-Beltrami flow."""
    n = 16
    rng = np.random.default_rng(3)
    x = np.arange(n) * 2 * math.pi / n
    X2, Y2 = np.meshgrid(x, x, indexing="ij")
    w2 = np.zeros((n, n))
    for kx in range(-2, 3):
        for ky in range(-2, 3):
            if kx or ky:
                w2 += rng.standard_normal() * np.cos(kx * X2 + ky * Y2 + rng.uniform(0, 6.28)) / (1 + kx * kx + ky * ky)
    s2 = SpectralNS2D(n, nu=nu)
    s2.set_vorticity(w2)
    u2, v2 = s2.velocity()
    u3 = np.zeros((3, n, n, n))
    u3[0], u3[1] = u2[:, :, None], v2[:, :, None]
    s3 = SpectralNS3D(n, nu=nu)
    s3.set_velocity(u3)
    s2.run(0.5, dt=0.01)
    s3.run(0.5, dt=0.01)
    w3 = s3.vorticity()
    assert np.max(np.abs(w3[2] - s2.vorticity()[:, :, None])) < 1e-12
    assert np.max(np.abs(w3[:2])) < 1e-13 and np.max(np.abs(s3.velocity()[2])) < 1e-13


def test_the_embedding_holds_in_another_plane_too():
    """The same flow in the y-z plane (components 1, 2; coordinates y, z) gives ω_x."""
    n = 16
    rng = np.random.default_rng(5)
    x = np.arange(n) * 2 * math.pi / n
    Y2, Z2 = np.meshgrid(x, x, indexing="ij")
    w2 = sum(rng.standard_normal() * np.cos(k * Y2 + l * Z2 + rng.uniform(0, 6.28))
             for k in (-1, 0, 1, 2) for l in (-2, -1, 1, 2)) / 4
    s2 = SpectralNS2D(n, nu=0.02)
    s2.set_vorticity(w2)
    u2, v2 = s2.velocity()
    u3 = np.zeros((3, n, n, n))
    u3[1], u3[2] = u2[None, :, :], v2[None, :, :]
    s3 = SpectralNS3D(n, nu=0.02)
    s3.set_velocity(u3)
    s2.run(0.4, dt=0.01)
    s3.run(0.4, dt=0.01)
    assert np.max(np.abs(s3.vorticity()[0] - s2.vorticity()[None, :, :])) < 1e-12


def test_the_nonlinear_term_equals_the_direct_triadic_convolution():
    """N̂(k) = (1/n³) P_k Σ_{p+q=k} û_p × (i q × û_q), summed directly over retained
    modes — the 3-D 'non-linear integral', with √−1 inside the curl."""
    n = 8
    kmax = dealias_cutoff(n)
    rng = np.random.default_rng(7)
    s = SpectralNS3D(n)
    s.set_velocity(rng.standard_normal((3, n, n, n)))
    u_hat = s.state_hat
    idx = [(a, b, c) for a in range(-kmax, kmax + 1) for b in range(-kmax, kmax + 1) for c in range(-kmax, kmax + 1)]

    def at(arr, k):
        return arr[:, k[0] % n, k[1] % n, k[2] % n]

    direct = np.zeros((3, n, n, n), dtype=complex)
    for k in idx:
        total = np.zeros(3, dtype=complex)
        for p in idx:
            q = (k[0] - p[0], k[1] - p[1], k[2] - p[2])
            if max(abs(q[0]), abs(q[1]), abs(q[2])) > kmax:
                continue
            total += np.cross(at(u_hat, p), np.cross(1j * np.array(q, dtype=float), at(u_hat, q)))
        kv = np.array(k, dtype=float)
        k2 = kv @ kv
        total = total / n**3
        if k2 > 0:
            total = total - kv * (kv @ total) / k2          # Leray projection
        direct[:, k[0] % n, k[1] % n, k[2] % n] = total
    assert np.max(np.abs(s.nonlinear_hat(u_hat) - direct)) < 1e-11
    assert np.max(np.abs(direct)) > 1e-3                      # a real, non-trivial comparison


def test_the_state_never_leaves_the_dealiased_band_and_is_divergence_free_and_mean_free():
    n = 16
    rng = np.random.default_rng(11)
    s = SpectralNS3D(n, nu=0.01, forcing=rng.normal(size=(3, n, n, n)))     # white-noise force
    s.set_velocity(rng.normal(size=(3, n, n, n)) + 5.0)                     # and a mean flow
    m = dealias_mask3(n)
    assert np.all(s.state_hat[:, m == 0] == 0) and np.all(s.state_hat[:, 0, 0, 0] == 0)
    s.run(0.02, dt=0.002)
    assert np.all(s.state_hat[:, m == 0] == 0)
    assert s.max_divergence() < 1e-11 and abs(s.velocity().mean()) < 1e-13


def test_a_gradient_field_is_projected_out_entirely():
    n = 16
    X, Y, Z = grid3(n)
    s = SpectralNS3D(n)
    s.set_velocity(np.stack((np.cos(X) * np.sin(Y), np.sin(X) * np.cos(Y), 0 * X)))   # = ∇(sin x sin y)
    assert np.max(np.abs(s.state_hat)) < 1e-12
    base = abc_flow(n)
    a, b = SpectralNS3D(n), SpectralNS3D(n)
    a.set_velocity(base)
    b.set_velocity(base + np.stack((np.cos(X) * np.sin(Y), np.sin(X) * np.cos(Y), 0 * X)))
    assert np.max(np.abs(a.state_hat - b.state_hat)) < 1e-12


# ═══════════════════════════ conservation laws ══════════════════════════════

def _smooth_random_velocity(n, seed=1, modes=2):
    rng = np.random.default_rng(seed)
    X, Y, Z = grid3(n)
    u = np.zeros((3, n, n, n))
    for kx in range(-modes, modes + 1):
        for ky in range(-modes, modes + 1):
            for kz in range(-modes, modes + 1):
                if kx == ky == kz == 0:
                    continue
                phase = rng.uniform(0, 6.28)
                amp = rng.standard_normal(3) / (1 + kx * kx + ky * ky + kz * kz)
                for c in range(3):
                    u[c] += amp[c] * np.cos(kx * X + ky * Y + kz * Z + phase)
    return u


def test_inviscid_unforced_flow_conserves_energy_and_helicity_up_to_the_integrator():
    """The Galerkin-truncated system conserves both exactly; what is left is the RK4 error,
    so the drift must shrink by far more than 4× when dt halves — that is what makes it the
    integrator's."""
    n = 16
    u0 = _smooth_random_velocity(n)

    def drift(dt):
        s = SpectralNS3D(n, nu=0.0)
        s.set_velocity(u0)
        e0, h0 = s.energy(), s.helicity()
        assert abs(h0) > 1e-3 * e0                            # helicity is genuinely non-zero
        s.run(0.3, dt=dt)
        return abs(s.energy() - e0) / e0, abs(s.helicity() - h0) / abs(h0)

    (e1, h1), (e2, h2) = drift(0.005), drift(0.0025)
    assert e2 < 1e-9 and h2 < 1e-9
    assert e1 / e2 > 8 and h1 / h2 > 8          # > 4 rules out second order; measured ≈ 22 and 14


def test_the_energy_budget_dE_dt_equals_minus_two_nu_Z():
    nu, h = 0.02, 1e-4
    s = SpectralNS3D(16, nu=nu)
    s.set_velocity(_smooth_random_velocity(16, seed=5))
    s.run(0.3, dt=0.005)
    e0, z0 = s.energy(), s.enstrophy()
    s.step(h)
    e1, z1 = s.energy(), s.enstrophy()
    assert (e1 - e0) / h == pytest.approx(-2 * nu * 0.5 * (z0 + z1), rel=1e-4)


def test_energy_and_enstrophy_are_the_parseval_sums_of_the_physical_fields():
    n = 16
    s = SpectralNS3D(n)
    s.set_velocity(_smooth_random_velocity(n, seed=9))
    u, w = s.velocity(), s.vorticity()
    assert s.energy() == pytest.approx(0.5 * np.mean(np.sum(u**2, axis=0)), rel=1e-12)
    assert s.enstrophy() == pytest.approx(0.5 * np.mean(np.sum(w**2, axis=0)), rel=1e-12)
    assert s.helicity() == pytest.approx(np.mean(np.sum(u * w, axis=0)), rel=1e-10)


# ═══════════════════════════ the time integrator ════════════════════════════

def test_time_stepping_is_fourth_order():
    n, nu, T = 16, 0.01, 0.5
    u0 = _smooth_random_velocity(n, seed=2)

    def final(dt):
        s = SpectralNS3D(n, nu=nu)
        s.set_velocity(u0)
        s.run(T, dt=dt)
        return s.velocity()

    ref = final(T / 320)
    errs = [np.max(np.abs(final(T / m) - ref)) for m in (5, 10, 20)]
    assert errs[0] / errs[1] > 12 and errs[1] / errs[2] > 12     # ≈ 16 for 4th order
    assert errs[2] > 0 and errs[0] / errs[2] > 150


def test_the_cfl_step_is_cfl_times_dx_over_the_peak_speed():
    n = 16
    X, Y, Z = grid3(n)
    s = SpectralNS3D(n)
    assert s.stable_dt() == math.inf
    s.set_velocity(np.stack((np.sin(Y), 0 * X, 0 * X)))
    assert s.max_speed() == pytest.approx(1.0, abs=1e-12)
    assert s.stable_dt(0.5) == pytest.approx(0.5 * 2 * math.pi / n, rel=1e-12)
    s.set_velocity(np.stack((3 * np.sin(Y), 0 * X, 0 * X)))
    assert s.stable_dt(0.5) == pytest.approx(0.5 * 2 * math.pi / n / 3, rel=1e-12)
    s.set_velocity(np.stack((np.sin(Y), np.sin(Z), np.sin(X))))        # ∇·u = 0; CFL counts every component
    assert s.max_speed() == pytest.approx(3.0, abs=1e-12)


def test_the_cfl_limited_run_reaches_t_end_exactly_and_stays_finite():
    s = SpectralNS3D(16, nu=0.02)
    s.set_velocity(_smooth_random_velocity(16, seed=4) * 2)
    steps = s.run(0.5)
    assert s.time == pytest.approx(0.5, abs=1e-12) and steps > 1
    assert np.all(np.isfinite(s.velocity()))


# ═══════════════════════════ the shadow ═════════════════════════════════════

def test_the_shadow_is_round_off_after_a_nonlinear_run():
    s = SpectralNS3D(16, nu=0.02)
    s.set_velocity(_smooth_random_velocity(16, seed=6))
    s.run(0.3, dt=0.01)
    assert s.imaginary_residue() < 1e-13


def test_the_shadow_measurement_has_teeth():
    s = SpectralNS3D(16, nu=0.02)
    s.set_velocity(_smooth_random_velocity(16, seed=6))
    assert s.imaginary_residue() < 1e-13
    s._u_hat[0, 2, 1, 3] += 5.0                      # a mode with no conjugate partner
    assert s.imaginary_residue() > 1e-4


def test_the_divergence_diagnostic_has_teeth():
    n = 16
    X, _, _ = grid3(n)
    s = SpectralNS3D(n)
    s.velocity = lambda: np.stack((np.sin(X), 0 * X, 0 * X))        # ∇·u = cos x
    assert s.max_divergence() == pytest.approx(1.0, abs=1e-12)


# ═══════════════════════════ resolution ═════════════════════════════════════

def test_a_smooth_flow_is_resolved_and_top_shell_energy_is_not():
    n = 32
    s = SpectralNS3D(n)
    s.set_velocity(abc_flow(n))
    assert s.tail_fraction() == pytest.approx(0.0, abs=1e-20) and s.is_resolved()
    kmax = dealias_cutoff(n)
    X, Y, Z = grid3(n)
    s.set_velocity(abc_flow(n) + 0.5 * np.stack((np.sin(kmax * Y), 0 * X, 0 * X)))
    assert s.tail_fraction() > 0.1 and not s.is_resolved()


def test_under_resolution_hiding_in_a_cube_corner_is_caught():
    """The retained band is a cube; its corner (kmax, kmax, kmax) has |k| = √3·kmax, far
    outside any spherical shell. Energy only there must still register in the tail."""
    n = 32
    kmax = dealias_cutoff(n)
    X, Y, Z = grid3(n)
    s = SpectralNS3D(n)
    phase = kmax * (X + Y + Z)
    u = np.stack((np.cos(phase), -np.cos(phase), 0 * X))        # k·u = kmax(1−1) = 0: divergence-free
    s.set_velocity(u)
    assert s.tail_fraction() == pytest.approx(1.0, abs=1e-12)


def test_the_energy_spectrum_accounts_for_all_the_energy():
    n = 16
    s = SpectralNS3D(n)
    s.set_velocity(_smooth_random_velocity(n, seed=8))
    spec = s.energy_spectrum()
    assert spec.shape == (n // 2 + 1,) and spec[0] == 0.0
    assert spec.sum() == pytest.approx(s.energy(), rel=1e-12)


# ═══════════════════════════ the instrument ═════════════════════════════════

def spectral_state(n, delta, b=-3.0, *, exact_k=False):
    """A divergence-free state with per-mode energy ``k^b e^{−2δk}`` — ``k`` the shell
    index (``exact_k=False``) or the exact ``|k|`` — and a known strip width ``δ``.
    û(k) = i·env·(a×k)/|a×k| is Hermitian and perpendicular to k with |û| = env."""
    kx, ky, kz = wavenumbers3(n)
    K = np.stack((kx, ky, kz))
    kk = np.sqrt((K**2).sum(0))
    shell = kk if exact_k else np.rint(kk)
    kmax = dealias_cutoff(n)
    inband = (np.abs(K).max(0) <= kmax) & (kk > 0)
    env = np.where(inband, np.power(np.where(shell > 0, shell, 1.0), b / 2) * np.exp(-delta * shell), 0.0)
    a = np.array([1.0, 0.37, 0.21]).reshape(3, 1, 1, 1) * np.ones_like(K)
    c = np.cross(a, K, axis=0)
    norm = np.sqrt((c**2).sum(0))
    norm[norm == 0] = 1.0
    return np.fft.ifftn(1j * env * c / norm, axes=(1, 2, 3)).real


@pytest.mark.parametrize("delta,b", [(0.25, -3.0), (0.6, -5 / 3), (1.1, -4.0)])
def test_the_strip_width_and_prefactor_are_recovered_exactly_from_a_state(delta, b):
    n = 32
    s = SpectralNS3D(n)
    s.set_velocity(spectral_state(n, delta, b))
    kmax = dealias_cutoff(n)
    d, bb = analyticity_fit(s.energy_spectrum(per_mode=True), 2, kmax)
    assert d == pytest.approx(delta, abs=1e-9) and bb == pytest.approx(b, abs=1e-9)
    assert s.analyticity_width() == pytest.approx(analyticity_width(s.energy_spectrum(per_mode=True), max(2, kmax // 3), kmax))


def test_the_shell_sum_spectrum_is_biased_by_the_lattice_count_which_is_why_per_mode_is_used():
    """Shell sums carry ≈4πk² modes per shell; a b·ln k term absorbs that only
    approximately. The bias is real (~6%) and bounded, so the method fits per-mode."""
    n = 32
    s = SpectralNS3D(n)
    s.set_velocity(spectral_state(n, 0.6, -3.0))
    kmax = dealias_cutoff(n)
    biased = analyticity_width(s.energy_spectrum(), 2, kmax)
    assert 0.01 < abs(biased - 0.6) / 0.6 < 0.10
    assert s.analyticity_width(2, kmax) == pytest.approx(0.6, abs=1e-9)


@pytest.mark.parametrize("theta", [1.0, 0.7, 0.5])
def test_the_decay_exponent_tells_analytic_from_gevrey(theta):
    k = np.arange(0, 41)
    spec = np.zeros(41)
    spec[1:] = k[1:] ** -2.0 * np.exp(-0.9 * k[1:] ** theta)
    th, c = decay_exponent(spec, 4, 40)
    assert th == pytest.approx(theta, abs=0.02) and c == pytest.approx(0.9, rel=0.1)


def test_fit_windows_are_validated():
    spec = np.exp(-0.3 * np.arange(20.0))
    for lo, hi in [(0, 10), (5, 5), (8, 4), (3, 20), (3, 5)]:
        with pytest.raises(NavierStokesMeshError):
            analyticity_width(spec, lo, hi)
    with pytest.raises(NavierStokesMeshError):
        analyticity_width(np.array([1.0, np.nan, 1.0, 1.0, 1.0, 1.0]), 1, 5)
    with pytest.raises(NavierStokesMeshError):
        analyticity_width(spec, 2.5, 10)
    with pytest.raises(NavierStokesMeshError):
        analyticity_width(np.zeros(20), 2, 10)               # nothing positive to fit


def test_required_width_is_the_width_at_which_the_solver_itself_stops_being_resolved():
    n, tol, b = 32, 1e-6, -3.0
    req = required_width(n, tol, b=b)
    for factor, resolved in [(0.9, False), (0.98, False), (1.02, True), (1.1, True)]:
        s = SpectralNS3D(n)
        s.set_velocity(spectral_state(n, factor * req, b, exact_k=True))
        assert s.is_resolved(tol) is resolved
    s.set_velocity(spectral_state(n, req, b, exact_k=True))
    assert s.tail_fraction() == pytest.approx(tol, rel=1e-3)


def test_required_width_falls_with_a_finer_grid_and_rises_with_a_stricter_tolerance():
    widths = [required_width(n, 1e-6, b=-3.0) for n in (16, 32, 48)]
    assert widths[0] > widths[1] > widths[2] > 0
    assert required_width(32, 1e-9, b=-3.0) > required_width(32, 1e-6, b=-3.0) > required_width(32, 1e-3, b=-3.0)


@pytest.mark.parametrize("kwargs", [dict(n=4), dict(n=16.0), dict(n=True), dict(n=16, tol=0.0),
                                    dict(n=16, tol=1.0), dict(n=16, shell=1.0), dict(n=16, b=float("nan"))])
def test_required_width_refuses_bad_arguments(kwargs):
    with pytest.raises(NavierStokesMeshError):
        required_width(**kwargs)


def test_a_synthetic_collapse_is_forecast_to_its_singularity_time():
    """δ(t) = δ0 (1 − t/T*): measured from real spectral states, not handed in."""
    n, b, delta0, t_star = 32, -3.0, 1.6, 0.9
    ts = np.linspace(0.0, 0.3, 6)          # still above the requirement at the last sample
    widths = []
    for t in ts:
        s = SpectralNS3D(n)
        s.set_velocity(spectral_state(n, delta0 * (1 - t / t_star), b))
        widths.append(s.analyticity_width(2, dealias_cutoff(n)))
    f = resolution_forecast(ts, widths, n, 1e-6, b=b)
    assert f["collapsing"] and f["r_squared"] > 1 - 1e-9
    assert f["t_singularity"] == pytest.approx(t_star, rel=1e-6)
    assert not f["already_lost"]
    req = required_width(n, 1e-6, b=b)
    assert f["t_resolution_lost"] == pytest.approx(t_star * (1 - req / delta0), rel=1e-6)
    assert f["t_resolution_lost"] < f["t_singularity"]


def test_the_grid_really_loses_resolution_when_the_forecast_says():
    """Same collapse, states built with the exact |k| envelope: the solver's own
    is_resolved must flip at the forecast time."""
    n, b, delta0, t_star = 32, -3.0, 1.6, 0.9
    req = required_width(n, 1e-6, b=b)
    t_lost = t_star * (1 - req / delta0)
    for t, resolved in [(0.9 * t_lost, True), (0.98 * t_lost, True), (1.02 * t_lost, False), (1.1 * t_lost, False)]:
        s = SpectralNS3D(n)
        s.set_velocity(spectral_state(n, delta0 * (1 - t / t_star), b, exact_k=True))
        assert s.is_resolved() is resolved


def test_a_widening_strip_is_not_a_collapse():
    f = resolution_forecast([0.0, 1.0, 2.0, 3.0], [0.5, 0.6, 0.7, 0.8], 16)
    assert not f["collapsing"] and f["t_singularity"] is None and f["t_resolution_lost"] is None


def test_a_flow_already_below_the_requirement_is_flagged_by_its_latest_width():
    req = required_width(16, 1e-6)
    shrinking = resolution_forecast([0.0, 1.0, 2.0], [req * 1.5, req * 1.0, req * 0.5], 16)
    assert shrinking["already_lost"] and shrinking["collapsing"]            # was fine, now below
    recovering = resolution_forecast([0.0, 1.0, 2.0], [req * 0.5, req * 1.0, req * 1.5], 16)
    assert not recovering["already_lost"] and not recovering["collapsing"]  # was below, now fine


def test_a_noisy_series_reports_a_poor_fit():
    rng = np.random.default_rng(2)
    ts = np.linspace(0, 1, 12)
    noisy = 1.0 - 0.1 * ts + rng.normal(0, 0.3, size=12)
    assert resolution_forecast(ts, noisy, 16)["r_squared"] < 0.5


@pytest.mark.parametrize("times,widths", [
    ([0, 1], [1, 0.5]), ([0, 1, 1], [1, 0.5, 0.2]), ([0, 2, 1], [1, 0.5, 0.2]),
    ([0, 1, 2], [1, np.nan, 0.2]), ([0, 1, 2], [1, 0.5]), ([[0, 1, 2]], [[1, 0.5, 0.2]]),
])
def test_forecast_inputs_are_validated(times, widths):
    with pytest.raises(NavierStokesMeshError):
        resolution_forecast(times, widths, 16)


# ═══════════════════════════ validation ═════════════════════════════════════

@pytest.mark.parametrize("kwargs", [dict(n=4), dict(n=16.0), dict(n=True), dict(nu=-1.0), dict(nu=float("nan"))])
def test_bad_parameters_are_refused(kwargs):
    with pytest.raises(NavierStokesMeshError):
        SpectralNS3D(**kwargs)


def test_bad_fields_are_refused():
    s = SpectralNS3D(8)
    for bad in (np.zeros((8, 8, 8)), np.zeros((3, 8, 8)), np.full((3, 8, 8, 8), np.nan)):
        with pytest.raises(NavierStokesMeshError):
            s.set_velocity(bad)
        with pytest.raises(NavierStokesMeshError):
            s.set_forcing(bad)


def test_a_force_that_returns_the_wrong_shape_fails_loudly_at_the_step():
    s = SpectralNS3D(8, forcing=lambda t: np.zeros((3, 4, 4, 4)))
    with pytest.raises(NavierStokesMeshError):
        s.step(0.1)


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_time_steps_are_refused(dt):
    with pytest.raises(NavierStokesMeshError):
        SpectralNS3D(8).step(dt)


def test_run_refuses_a_target_in_the_past_and_a_step_budget_overrun():
    s = SpectralNS3D(8, nu=0.01)
    s.set_velocity(abc_flow(8))
    s.run(0.2, dt=0.1)
    with pytest.raises(NavierStokesMeshError):
        s.run(0.1)
    with pytest.raises(NavierStokesMeshError, match="max_steps"):
        s.run(10.0, dt=0.01, max_steps=3)


def test_the_diagnostics_dict_is_complete_and_consistent():
    s = SpectralNS3D(16, nu=0.02)
    s.set_velocity(_smooth_random_velocity(16, seed=3))
    s.run(0.1, dt=0.01)
    d = s.diagnostics()
    assert set(d) == {"time", "steps", "energy", "enstrophy", "helicity", "max_speed",
                      "max_divergence", "imaginary_residue", "tail_fraction", "resolved"}
    assert d["energy"] == s.energy() and d["resolved"] is s.is_resolved() and d["steps"] == 10


def test_module_exports_resolve():
    for name in ns3.__all__:
        assert hasattr(ns3, name)
