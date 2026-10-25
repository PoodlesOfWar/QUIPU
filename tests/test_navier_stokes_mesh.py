"""2-D Navier–Stokes on the mesh torus: verified against exact answers, not asserted.

Every numerical claim here has an independent ground truth: the Taylor–Green
vortex's exact decay, a direct triadic convolution sum, a steady Kolmogorov flow,
conservation laws, the energy budget dE/dt = −2νZ, and the order of the time
integrator. No network, no database outside ``tmp_path``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.quipu import navier_stokes_mesh as nsm
from src.quipu.navier_stokes_mesh import (
    DEFAULT_BLOB_RADIUS,
    MESH_TORUS_N,
    NavierStokesMeshError,
    SpectralNS2D,
    dealias_cutoff,
    dealias_mask,
    forcing_from_edges,
    mesh_forcing_field,
    resample_field,
    simulate_from_mesh,
    spectral_derivative,
    wavenumbers,
)


def grid(n):
    x = np.arange(n) * 2 * math.pi / n
    return np.meshgrid(x, x, indexing="ij")


def taylor_green(n, nu, t_end, dt=0.01):
    X, Y = grid(n)
    s = SpectralNS2D(n, nu=nu)
    s.set_vorticity(2 * np.cos(X) * np.cos(Y))
    s.run(t_end, dt=dt)
    return s, 2 * np.cos(X) * np.cos(Y) * math.exp(-2 * nu * s.time)


# First in the file on purpose: if time stopped advancing, every ``run`` below would
# spin to its step budget, so this must fail before any of them is reached.
def test_a_step_advances_time_and_the_step_count():
    s = SpectralNS2D(16, nu=0.01)
    s.step(0.25)
    s.step(0.125)
    assert s.time == pytest.approx(0.375, abs=1e-15) and s.steps == 2


# ═══════════════════════════ √−1 as the derivative operator ═════════════════

def test_wavenumbers_are_exact_integers_with_x_on_axis_0():
    kx, ky = wavenumbers(8)
    assert kx[:, 0].tolist() == [0, 1, 2, 3, -4, -3, -2, -1]
    assert np.all(kx[:, 0:1] == kx[:, 3:4]) and np.all(ky[0:1, :] == ky[3:4, :])


@pytest.mark.parametrize("n", [16, 32, 64])
def test_spectral_derivative_is_exact_for_resolved_modes(n):
    """∂/∂x sin(3x) = 3cos(3x): i·k applied in Fourier space, to round-off."""
    x = np.arange(n) * 2 * math.pi / n
    d = np.fft.ifft(spectral_derivative(np.fft.fft(np.sin(3 * x)), 0))
    assert np.max(np.abs(d.real - 3 * np.cos(3 * x))) < 1e-12
    assert np.max(np.abs(d.imag)) < 1e-12  # the shadow vanishes


def test_the_shadow_survives_only_at_the_nyquist_mode():
    """The one place i does NOT drop out: the Nyquist wavenumber has no conjugate
    partner, so i·k leaves a purely imaginary physical field. Zeroing it removes it."""
    n = 32
    x = np.arange(n) * 2 * math.pi / n
    nyq = np.fft.fft(np.cos(x * (n // 2)))
    assert np.max(np.abs(np.fft.ifft(spectral_derivative(nyq, 0, zero_nyquist=False)).imag)) > 10
    assert np.max(np.abs(np.fft.ifft(spectral_derivative(nyq, 0, zero_nyquist=True)).imag)) < 1e-12


def test_the_nyquist_mode_is_zeroed_by_default():
    n = 32
    x = np.arange(n) * 2 * math.pi / n
    nyq = np.fft.fft(np.cos(x * (n // 2)))
    assert np.array_equal(spectral_derivative(nyq, 0), spectral_derivative(nyq, 0, zero_nyquist=True))
    assert np.max(np.abs(spectral_derivative(nyq, 0))) == 0.0


def test_odd_grids_have_no_nyquist_mode_to_worry_about():
    n = 15
    x = np.arange(n) * 2 * math.pi / n
    d = np.fft.ifft(spectral_derivative(np.fft.fft(np.sin(2 * x)), 0, zero_nyquist=False))
    assert np.max(np.abs(d.imag)) < 1e-12


# ═══════════════════════════ dealiasing ═════════════════════════════════════

@pytest.mark.parametrize("n,expected", [(8, 2), (12, 3), (32, 10), (64, 21), (96, 31)])
def test_dealias_cutoff_is_strictly_below_n_over_3(n, expected):
    assert dealias_cutoff(n) == expected
    assert 3 * dealias_cutoff(n) < n


def test_n_over_3_would_alias_when_n_is_divisible_by_3():
    """Why the cutoff is (n−1)//3 and not n//3: for n=12, kmax=4 lets 4+4=8 alias to
    8−12=−4, which is inside the retained band."""
    n = 12
    assert n // 3 == 4 and (n // 3) * 3 == n          # the naive choice sits exactly on the edge
    assert (2 * (n // 3)) - n == -(n // 3)           # …and the sum aliases onto −kmax
    assert 2 * dealias_cutoff(n) - n < -dealias_cutoff(n)  # ours does not (−6 < −3)


def test_mask_is_symmetric_under_k_to_minus_k_and_keeps_the_mean_mode():
    """The retained band must be closed under k → −k, or the state stops being Hermitian."""
    n = 32
    m = dealias_mask(n)
    neg = (-np.arange(n)) % n
    assert m[0, 0] == 1
    assert np.array_equal(m, m[np.ix_(neg, neg)])
    kmax = dealias_cutoff(n)
    kx, ky = wavenumbers(n)
    assert np.array_equal(m.astype(bool), (np.abs(kx) <= kmax) & (np.abs(ky) <= kmax))


# ═══════════════════════════ exact solutions ════════════════════════════════

@pytest.mark.parametrize("nu", [0.0, 0.01, 0.1])
def test_taylor_green_vortex_matches_the_exact_solution(nu):
    """ω = 2cos x cos y is an eigenfunction (|k|²=2) so the nonlinear term vanishes and
    the exact solution is e^{−2νt}·ω₀. A very sharp test of the whole stepper."""
    s, exact = taylor_green(32, nu, 1.0)
    assert np.max(np.abs(s.vorticity() - exact)) < 1e-12


def test_taylor_green_energy_and_enstrophy_are_exact():
    nu = 0.05
    s, _ = taylor_green(32, nu, 1.0)
    assert s.energy() == pytest.approx(0.25 * math.exp(-4 * nu * s.time), rel=1e-12)
    assert s.enstrophy() == pytest.approx(0.5 * math.exp(-4 * nu * s.time), rel=1e-12)


def test_forced_kolmogorov_flow_converges_to_its_exact_steady_state():
    """u = U sin(ky) is a steady solution when f = (νk²+α)·ω. Checks the forcing and
    drag paths, and the sign convention ω = ∂v/∂x − ∂u/∂y = −Uk·cos(ky).

    The *discrete* fixed point of a 4th-order scheme differs from the exact steady
    state by O(dt⁴), so the honest assertion is that the error is small and shrinks
    16× when dt is halved — not that it is zero."""
    n, nu, drag, k, U = 32, 0.05, 0.2, 3, 1.0
    X, Y = grid(n)
    omega = -U * k * np.cos(k * Y)

    def settle(dt):
        s = SpectralNS2D(n, nu=nu, drag=drag, forcing=(nu * k**2 + drag) * omega)
        s.run(60.0, dt=dt)                    # e^{−0.65·60} ≈ 1e-17: fully converged in time
        return s

    coarse, fine = settle(0.05), settle(0.025)
    e_coarse = np.max(np.abs(coarse.vorticity() - omega))
    e_fine = np.max(np.abs(fine.vorticity() - omega))
    assert e_coarse < 1e-8 and e_fine < 1e-9
    assert e_coarse / e_fine > 12                              # ≈ 2⁴ = 16
    u, v = fine.velocity()
    assert np.max(np.abs(u - U * np.sin(k * Y))) < 1e-9 and np.max(np.abs(v)) < 1e-9


def test_velocity_sign_convention():
    X, Y = grid(32)
    s = SpectralNS2D(32)
    s.set_vorticity(2 * np.cos(X) * np.cos(Y))   # ψ = cos x cos y
    u, v = s.velocity()
    assert np.max(np.abs(u - (-np.cos(X) * np.sin(Y)))) < 1e-12   # u = ∂ψ/∂y
    assert np.max(np.abs(v - (np.sin(X) * np.cos(Y)))) < 1e-12    # v = −∂ψ/∂x


# ═══════════════════════════ the nonlinear "integral" ═══════════════════════

def test_pseudo_spectral_nonlinear_term_equals_the_direct_triadic_convolution():
    """N̂(k) = −(1/n²) Σ_{p+q=k} [û_p·(i q_x)ω̂_q + v̂_p·(i q_y)ω̂_q], summed directly over
    retained modes, must equal the FFT route. This is the 'non-linear integral'."""
    n = 12
    kmax = dealias_cutoff(n)
    rng = np.random.default_rng(7)
    s = SpectralNS2D(n)
    s.set_vorticity(rng.standard_normal((n, n)))
    w_hat = s.state_hat
    kx, ky = wavenumbers(n)
    psi_hat = w_hat / np.where(kx**2 + ky**2 == 0, 1, kx**2 + ky**2)
    u_hat, v_hat = 1j * ky * psi_hat, -1j * kx * psi_hat
    idx = [(a, b) for a in range(-kmax, kmax + 1) for b in range(-kmax, kmax + 1)]
    direct = np.zeros((n, n), dtype=complex)
    for (kx0, ky0) in idx:
        total = 0j
        for (px, py) in idx:
            qx, qy = kx0 - px, ky0 - py
            if abs(qx) > kmax or abs(qy) > kmax:
                continue
            total += (u_hat[px % n, py % n] * (1j * qx) * w_hat[qx % n, qy % n]
                      + v_hat[px % n, py % n] * (1j * qy) * w_hat[qx % n, qy % n])
        direct[kx0 % n, ky0 % n] = -total / n**2
    assert np.max(np.abs(s.nonlinear_hat(w_hat) - direct)) < 1e-11


def test_without_the_two_thirds_rule_the_same_sum_would_be_wrong():
    """Control for the test above: evaluate the product on a full-band field and the
    retained modes are polluted by aliasing — so the agreement above is meaningful."""
    n = 12
    rng = np.random.default_rng(3)
    f = rng.standard_normal((n, n))
    g = rng.standard_normal((n, n))
    full = np.fft.fft2(f * g)                      # aliased product of full-band fields
    kmax = dealias_cutoff(n)
    F, G = np.fft.fft2(f), np.fft.fft2(g)
    exact = np.zeros((n, n), dtype=complex)        # exact product of the band-limited parts
    Fm, Gm = F * dealias_mask(n), G * dealias_mask(n)
    exact = np.fft.fft2(np.fft.ifft2(Fm).real * np.fft.ifft2(Gm).real) * dealias_mask(n)
    assert np.max(np.abs(full * dealias_mask(n) - exact)) > 1e-3
    assert kmax == 3


# ═══════════════════════════ conservation laws ══════════════════════════════

def _smooth_random_state(n, seed=1, modes=3):
    rng = np.random.default_rng(seed)
    X, Y = grid(n)
    w = np.zeros((n, n))
    for kx in range(-modes, modes + 1):
        for ky in range(-modes, modes + 1):
            if kx == ky == 0:
                continue
            w += rng.standard_normal() * np.cos(kx * X + ky * Y + rng.uniform(0, 6.28)) / (1 + kx * kx + ky * ky)
    return w


def test_inviscid_unforced_flow_conserves_energy_and_enstrophy():
    s = SpectralNS2D(32, nu=0.0)
    s.set_vorticity(_smooth_random_state(32))
    e0, z0 = s.energy(), s.enstrophy()
    s.run(1.0, dt=0.002)
    assert abs(s.energy() - e0) / e0 < 1e-8
    assert abs(s.enstrophy() - z0) / z0 < 1e-8


def test_the_energy_budget_dE_dt_equals_minus_two_nu_Z():
    """Unforced: dE/dt = −2νZ exactly (the dissipation is enstrophy-weighted)."""
    nu, h = 0.02, 1e-4
    s = SpectralNS2D(32, nu=nu)
    s.set_vorticity(_smooth_random_state(32, seed=5))
    s.run(0.5, dt=0.005)
    e0, z0 = s.energy(), s.enstrophy()
    s.step(h)
    e1, z1 = s.energy(), s.enstrophy()
    assert (e1 - e0) / h == pytest.approx(-2 * nu * 0.5 * (z0 + z1), rel=1e-4)


def test_enstrophy_never_increases_without_forcing():
    s = SpectralNS2D(32, nu=0.01)
    s.set_vorticity(_smooth_random_state(32, seed=9))
    previous = s.enstrophy()
    for _ in range(40):
        s.run(s.time + 0.05, dt=0.01)
        assert s.enstrophy() <= previous * (1 + 1e-12)
        previous = s.enstrophy()


def test_the_state_never_leaves_the_dealiased_band():
    """Out-of-band content in the vorticity or the forcing is cut on the way in, and
    the integrator cannot put it back: every masked-off mode stays exactly zero."""
    n = 32
    rng = np.random.default_rng(11)
    s = SpectralNS2D(n, nu=0.01, forcing=rng.normal(size=(n, n)))      # white noise: all modes
    s.set_vorticity(rng.normal(size=(n, n)))
    assert np.all(s.state_hat[dealias_mask(n) == 0] == 0)
    s.run(0.05, dt=0.005)
    assert np.all(s.state_hat[dealias_mask(n) == 0] == 0)
    assert np.any(s.state_hat != 0)


def test_the_divergence_diagnostic_has_teeth():
    """Velocity from a streamfunction is divergence-free by construction, so the
    diagnostic can only be trusted if it reports a field that is not."""
    n = 32
    X, _ = grid(n)
    s = SpectralNS2D(n)
    s.velocity = lambda: (np.sin(X), np.zeros((n, n)))     # ∇·u = cos x
    assert s.max_divergence() == pytest.approx(1.0, abs=1e-12)


def test_the_flow_is_divergence_free_and_zero_mean():
    s = SpectralNS2D(32, nu=0.01)
    s.set_vorticity(_smooth_random_state(32) + 5.0)    # a constant offset must be removed
    s.run(0.3, dt=0.01)
    assert s.max_divergence() < 1e-12
    assert abs(s.vorticity().mean()) < 1e-13


# ═══════════════════════════ the time integrator ════════════════════════════

def test_time_stepping_is_fourth_order():
    """Halving dt must cut the error ~16× on a genuinely nonlinear problem."""
    n, nu, T = 32, 0.01, 1.0
    w0 = _smooth_random_state(n, seed=2)

    def final(dt):
        s = SpectralNS2D(n, nu=nu)
        s.set_vorticity(w0)
        s.run(T, dt=dt)
        return s.vorticity()

    ref = final(T / 320)
    errs = [np.max(np.abs(final(T / m) - ref)) for m in (10, 20, 40)]
    assert errs[0] / errs[1] > 12 and errs[1] / errs[2] > 12     # ≈ 16 for 4th order
    assert errs[2] < 1e-5


def test_the_cfl_step_is_cfl_times_dx_over_the_peak_speed():
    n = 32
    X, Y = grid(n)
    s = SpectralNS2D(n)
    assert s.stable_dt() == math.inf                          # at rest: no advective limit
    s.set_vorticity(2 * np.cos(X) * np.cos(Y))                # peak speed exactly 1
    assert s.max_speed() == pytest.approx(1.0, abs=1e-12)
    assert s.stable_dt(0.5) == pytest.approx(0.5 * 2 * math.pi / n, rel=1e-12)
    s.set_vorticity(6 * np.cos(X) * np.cos(Y))                # three times faster
    assert s.stable_dt(0.5) == pytest.approx(0.5 * 2 * math.pi / n / 3, rel=1e-12)


def test_the_cfl_limited_run_is_stable_and_reaches_t_end_exactly():
    s = SpectralNS2D(32, nu=0.01)
    s.set_vorticity(_smooth_random_state(32, seed=4) * 3)
    steps = s.run(1.0)                       # dt chosen automatically
    assert s.time == pytest.approx(1.0, abs=1e-12) and steps > 1
    assert np.all(np.isfinite(s.vorticity())) and s.enstrophy() < 1e3


# ═══════════════════════════ the shadow is measured, not assumed ════════════

def test_the_shadow_is_round_off_after_a_nonlinear_run():
    s = SpectralNS2D(32, nu=0.01)
    s.set_vorticity(_smooth_random_state(32, seed=6))
    s.run(1.0, dt=0.01)
    assert s.imaginary_residue() < 1e-13


def test_the_measurement_has_teeth_a_broken_state_has_a_large_shadow():
    """If the √−1 → 1 resolution failed, imaginary_residue must show it. Plant a
    mode with no conjugate partner (a non-Hermitian state) and it does."""
    s = SpectralNS2D(32, nu=0.01)
    s.set_vorticity(_smooth_random_state(32))
    assert s.imaginary_residue() < 1e-13
    s._w_hat[3, 5] += 5.0                    # break Hermitian symmetry on purpose
    assert s.imaginary_residue() > 1e-4          # ≥ 9 orders above the 1e-13 healthy floor


# ═══════════════════════════ resolution ═════════════════════════════════════

def test_under_resolution_hiding_in_the_square_corners_is_caught():
    """Regression. The retained band is a square, so its corners reach √2·kmax. A
    circular |k| ≤ kmax shell never saw energy there, and the resolution check
    reported 'resolved' for a flow concentrated entirely at the grid scale."""
    n = 32
    kmax = dealias_cutoff(n)
    X, Y = grid(n)
    s = SpectralNS2D(n, nu=0.01)
    s.set_vorticity(np.cos(kmax * (X + Y)))                    # |k| = √2·kmax, in the corner
    assert math.hypot(kmax, kmax) > kmax                       # outside the old circular shell
    assert s.tail_fraction() > 0.99 and not s.is_resolved()


def test_a_smooth_flow_is_resolved_and_one_with_top_shell_energy_is_not():
    s = SpectralNS2D(32, nu=0.01)
    s.set_vorticity(_smooth_random_state(32))
    assert s.is_resolved() and s.tail_fraction() < 1e-6
    X, Y = grid(32)
    kmax = dealias_cutoff(32)
    s.set_vorticity(np.cos(kmax * X + (kmax - 1) * Y))     # all energy in the top shell
    assert not s.is_resolved() and s.tail_fraction() > 0.99


# ═══════════════════════════ validation ═════════════════════════════════════

@pytest.mark.parametrize("kwargs", [
    {"n": 4}, {"n": 8.0}, {"n": True}, {"nu": -1.0}, {"nu": float("nan")},
    {"drag": -0.1}, {"drag": float("inf")},
])
def test_bad_parameters_are_refused(kwargs):
    with pytest.raises(NavierStokesMeshError):
        SpectralNS2D(**{"n": 16, **kwargs})


@pytest.mark.parametrize("bad", [np.zeros((8, 8)), np.full((16, 16), np.nan), np.full((16, 16), np.inf)])
def test_bad_fields_are_refused(bad):
    s = SpectralNS2D(16)
    with pytest.raises(NavierStokesMeshError):
        s.set_vorticity(bad)
    with pytest.raises(NavierStokesMeshError):
        s.set_forcing(bad)


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_time_steps_are_refused(dt):
    with pytest.raises(NavierStokesMeshError):
        SpectralNS2D(16).step(dt)


def test_run_refuses_a_target_in_the_past_and_a_step_budget_overrun():
    s = SpectralNS2D(16)
    s.set_vorticity(_smooth_random_state(16))
    s.run(0.1, dt=0.01)
    with pytest.raises(NavierStokesMeshError):
        s.run(0.0)
    with pytest.raises(NavierStokesMeshError, match="max_steps"):
        s.run(10.0, dt=0.01, max_steps=3)


# ═══════════════════════════ mesh -> forcing (one way, read only) ═══════════

def test_the_module_grid_is_the_meshs_torus():
    from src.quipu import mesh_slm
    assert MESH_TORUS_N == mesh_slm._TORUS_N == nsm.DEFAULT_N


def test_resampling_preserves_values_and_low_modes():
    X, Y = grid(32)
    f = 3.0 + np.cos(X) * np.sin(2 * Y)
    X2, Y2 = grid(64)
    up = resample_field(f, 64)
    # the 2/3 band-limit removes nothing here: modes ≤ 2 are far below the cutoff
    assert np.max(np.abs(up - (3.0 + np.cos(X2) * np.sin(2 * Y2)))) < 1e-12
    assert np.allclose(resample_field(np.full((16, 16), 7.0), 32), 7.0)


def test_forcing_is_zero_mean_with_the_requested_rms():
    cells = {1: (3, 4), 2: (10, 20), 3: (40, 7)}
    edges = [(1, 2, 2.0), (2, 3, 5.0), (1, 3, 1.0)]
    f = forcing_from_edges(cells, edges, n=64, amplitude=0.7)
    assert f.source == "mesh" and f.n_edges == 3 and f.n_tokens == 3
    assert abs(f.field.mean()) < 1e-12
    assert math.sqrt(np.mean(f.field**2)) == pytest.approx(0.7, rel=1e-9)


def test_each_edge_credits_both_endpoints():
    f = forcing_from_edges({1: (5, 5), 2: (20, 30)}, [(1, 2, 3.0)], n=64)
    assert f.field[5, 5] == pytest.approx(f.field[20, 30])      # symmetric
    assert f.field[5, 5] > f.field[40, 40]                       # above the empty cells


def test_heavier_edges_force_harder_where_they_land():
    cells = {1: (5, 5), 2: (20, 20), 3: (50, 50)}
    f = forcing_from_edges(cells, [(1, 2, 1.0), (2, 3, 9.0)], n=64)
    assert f.field[20, 20] > f.field[50, 50] > f.field[5, 5]


def test_an_empty_or_degenerate_mesh_gives_zero_forcing_not_an_error():
    assert forcing_from_edges({}, [], n=64).source == "empty"
    assert not forcing_from_edges({}, [], n=64).field.any()
    # equal weighted degree on every one of the 64×64 cells ⇒ nothing left after mean
    # removal ⇒ 'empty', not a field of round-off noise rescaled to the target rms
    cells = {t: (t // 64, t % 64) for t in range(4096)}
    uniform = forcing_from_edges(cells, [(t, (t + 1) % 4096, 1.0) for t in range(4096)], n=64)
    assert uniform.source == "empty" and not uniform.field.any()


def test_forcing_is_invariant_to_the_overall_scale_of_the_weights():
    cells = {1: (5, 5), 2: (20, 31), 3: (47, 12)}
    base = forcing_from_edges(cells, [(1, 2, 2.0), (2, 3, 5.0)], n=64).field
    for scale in (1e-20, 1e20):
        scaled = forcing_from_edges(cells, [(1, 2, 2.0 * scale), (2, 3, 5.0 * scale)], n=64)
        assert scaled.source == "mesh" and np.allclose(scaled.field, base, atol=1e-9)


def test_out_of_range_cells_and_nonfinite_weights_are_ignored():
    cells = {1: (5, 5), 2: (200, 7), 3: (20, 20)}
    f = forcing_from_edges(cells, [(1, 3, 2.0), (1, 2, float("nan")), (2, 3, 4.0)], n=64)
    assert f.n_skipped == 1 and np.all(np.isfinite(f.field))
    assert f.n_edges == 2          # the NaN edge is not counted


def test_forcing_can_be_resampled_to_a_coarser_solver_grid():
    f = forcing_from_edges({1: (5, 5), 2: (20, 30)}, [(1, 2, 3.0)], n=32)
    assert f.field.shape == (32, 32) and abs(f.field.mean()) < 1e-12


@pytest.mark.parametrize("amp", [-1.0, float("nan")])
def test_bad_amplitude_is_refused(amp):
    with pytest.raises(NavierStokesMeshError):
        forcing_from_edges({}, [], amplitude=amp)


def _spectral_tail(field, shell=0.8):
    n = field.shape[0]
    kmax = dealias_cutoff(n)
    kx, ky = wavenumbers(n)
    kinf = np.maximum(np.abs(kx), np.abs(ky))
    p = np.abs(np.fft.fft2(field)) ** 2
    return float(p[(kinf > shell * kmax) & (kinf <= kmax)].sum() / p.sum())


def test_a_vortex_blob_removes_grid_scale_content_without_changing_the_rms():
    cells = {1: (5, 5), 2: (20, 31), 3: (47, 12), 4: (9, 58)}
    edges = [(1, 2, 4.0), (2, 3, 1.5), (3, 4, 0.5), (4, 1, 3.0)]
    tails = []
    for radius in (0.0, 1.0, 2.0, 3.0):
        f = forcing_from_edges(cells, edges, n=64, amplitude=0.5, blob_radius=radius)
        assert math.sqrt(np.mean(f.field**2)) == pytest.approx(0.5, rel=1e-9)
        assert abs(f.field.mean()) < 1e-12
        tails.append(_spectral_tail(f.field))
    assert tails == sorted(tails, reverse=True) and tails[0] / tails[-1] > 1e4   # strictly smoother


@pytest.mark.parametrize("radius,rel", [(1.0, 1e-2), (2.0, 1e-6), (3.5, 1e-6)])
def test_the_vortex_blob_is_a_gaussian_of_the_stated_width(radius, rel):
    """A delta spread by ``radius`` cells must fall off as exp(−d²/2r²) in both
    directions and keep its total (the blob redistributes, it does not scale).
    At one cell the grid itself truncates the Gaussian's spectrum (~0.4%), hence
    the looser tolerance there and the shorter reach (≤ 2 radii)."""
    n = 64
    delta = np.zeros((n, n))
    delta[20, 30] = 1.0
    blob = nsm._gaussian_blob(delta, radius)
    assert blob.sum() == pytest.approx(1.0, abs=1e-12)
    for d in (d for d in (1, 2, 4) if d <= 2 * radius):
        expected = math.exp(-d * d / (2 * radius * radius))
        assert blob[20 + d, 30] / blob[20, 30] == pytest.approx(expected, rel=rel)
        assert blob[20, 30 - d] / blob[20, 30] == pytest.approx(expected, rel=rel)


def test_the_default_blob_radius_is_the_measured_one():
    assert DEFAULT_BLOB_RADIUS == 2.0
    cells = {1: (5, 5), 2: (20, 31)}
    default = forcing_from_edges(cells, [(1, 2, 1.0)], n=64)
    explicit = forcing_from_edges(cells, [(1, 2, 1.0)], n=64, blob_radius=2.0)
    assert np.array_equal(default.field, explicit.field)


@pytest.mark.parametrize("radius", [-1.0, float("nan"), float("inf")])
def test_bad_blob_radius_is_refused(radius):
    with pytest.raises(NavierStokesMeshError):
        forcing_from_edges({}, [], blob_radius=radius)


# ---- against a real SQLite mesh database ----------------------------------

def _seed_mesh(tmp_path, monkeypatch):
    from src.quipu import mesh_slm
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "mesh.sqlite"))
    with mesh_slm._conn() as cn:                # creates the real mesh schema
        cn.executemany("INSERT INTO mesh_slm_vocab(token_id, token, i, j) VALUES(?,?,?,?)",
                       [(1, "supply", 5, 5), (2, "chain", 20, 31), (3, "mesh", 47, 12), (4, "node", 9, 58)])
        cn.executemany("INSERT INTO mesh_slm_quipu(src, dst, weight, samples) VALUES(?,?,?,?)",
                       [(1, 2, 4.0, 4), (2, 3, 1.5, 2), (3, 4, 0.5, 1), (4, 1, 3.0, 3)])
        cn.commit()
    return mesh_slm


def _table_checksum(mesh_slm):
    with mesh_slm._conn() as cn:
        return [list(map(tuple, cn.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()))
                for t in ("mesh_slm_vocab", "mesh_slm_quipu", "mesh_slm_meta")]


def test_forcing_is_read_from_the_real_mesh_tables(tmp_path, monkeypatch):
    _seed_mesh(tmp_path, monkeypatch)
    f = mesh_forcing_field(n=64)
    assert f.source == "mesh" and f.n_tokens == 4 and f.n_edges == 4
    # token 2 ('chain') touches the two heaviest edges' endpoint at (20, 31)
    assert f.field[20, 31] > f.field[47, 12]


def test_the_mesh_is_only_read_never_written(tmp_path, monkeypatch):
    mesh_slm = _seed_mesh(tmp_path, monkeypatch)
    before = _table_checksum(mesh_slm)
    out = simulate_from_mesh(0.5, n=32, dt=0.05)
    assert out["forcing_source"] == "mesh" and out["diagnostics"]["energy"] > 0
    assert _table_checksum(mesh_slm) == before                 # byte-for-byte unchanged


def test_the_connection_it_opens_is_query_only_and_runs_only_selects(tmp_path, monkeypatch):
    """One-way by construction: a connection this module opens itself is set to
    query_only, and every statement it runs is a SELECT."""
    import sqlite3
    from src.quipu import local_store
    _seed_mesh(tmp_path, monkeypatch)
    statements: list[str] = []

    real_open_conn = local_store.open_conn

    class Spy:
        def __init__(self):
            self._cn = real_open_conn()
            self.closed = False

        def execute(self, sql, *a):
            statements.append(" ".join(sql.split()))
            return self._cn.execute(sql, *a)

        def close(self):
            self.closed = True
            self._cn.close()

    spy_holder: list[Spy] = []
    monkeypatch.setattr(local_store, "open_conn", lambda: spy_holder.append(Spy()) or spy_holder[-1])
    mesh_forcing_field(n=32)
    assert statements[0] == "PRAGMA query_only = ON"
    assert all(s.upper().startswith("SELECT") for s in statements[1:]) and len(statements) == 3
    assert spy_holder[0].closed

    # and sqlite itself refuses writes on such a connection
    cn = sqlite3.connect(str(tmp_path / "mesh.sqlite"))
    cn.execute("PRAGMA query_only = ON")
    with pytest.raises(sqlite3.OperationalError):
        cn.execute("INSERT INTO mesh_slm_meta(key, value) VALUES('x', 'y')")
    cn.close()


def test_a_missing_mesh_is_an_empty_forcing_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "never_created.sqlite"))
    f = mesh_forcing_field(n=32)
    assert f.source == "empty" and not f.field.any()


def test_an_empty_mesh_drives_no_flow_and_a_populated_one_does(tmp_path, monkeypatch):
    monkeypatch.setenv("SCB_DB_PATH", str(tmp_path / "empty.sqlite"))
    quiet = simulate_from_mesh(0.5, n=32, dt=0.05)
    assert quiet["diagnostics"]["energy"] == 0.0
    mesh_slm = _seed_mesh(tmp_path, monkeypatch)
    del mesh_slm
    moving = simulate_from_mesh(0.5, n=32, dt=0.05)
    assert moving["diagnostics"]["energy"] > 0.0


def test_mesh_driven_runs_are_deterministic(tmp_path, monkeypatch):
    _seed_mesh(tmp_path, monkeypatch)
    a = simulate_from_mesh(0.5, n=32, dt=0.05)["solver"].vorticity()
    b = simulate_from_mesh(0.5, n=32, dt=0.05)["solver"].vorticity()
    assert np.array_equal(a, b)


def test_the_mesh_driven_flow_settles_to_a_resolved_divergence_free_state(tmp_path, monkeypatch):
    """With friction the forced flow equilibrates; it stays resolved, divergence-free,
    and its shadow stays at round-off the whole way."""
    _seed_mesh(tmp_path, monkeypatch)
    out = simulate_from_mesh(40.0, n=64, nu=0.02, drag=0.2, amplitude=0.5, dt=0.05)   # default blob
    s = out["solver"]
    d = out["diagnostics"]
    e_before = s.energy()
    s.run(s.time + 2.0, dt=0.05)
    assert abs(s.energy() - e_before) / e_before < 1e-3          # near steady state
    assert d["max_divergence"] < 1e-11 and d["imaginary_residue"] < 1e-12
    assert d["resolved"] and np.all(np.isfinite(s.vorticity()))
