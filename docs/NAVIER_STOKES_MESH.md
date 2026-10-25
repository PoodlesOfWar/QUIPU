# Navier–Stokes on the mesh torus (√−1 as the derivative)

`src/quipu/navier_stokes_mesh.py` solves the 2-D incompressible Navier–Stokes
equations on the same 64×64 periodic grid the QUIPU MESH already uses
(`mesh_slm._TORUS_N`), with a Fourier pseudo-spectral method, and lets the mesh's
quipu edges **drive the forcing, one way**.

```
∂ω/∂t + u·∇ω = ν∇²ω − αω + f(x, y)        ∇²ψ = −ω        u = ∂ψ/∂y,  v = −∂ψ/∂x
```

> **How the request was read.** The task said *"integrate the SQRT(-1) for the iteration
> of for within the synoidal QUIPU MESH interaction for its resolution in the Navier
> Stokes model"*, and chose *"spectral solver for the shadow derivative non-linear
> integrals"*. That was taken to mean: **i = √−1** (the "iteration of for" read as the
> imaginary unit), **synoidal** ≈ sinusoidal/periodic (the mesh's torus), the **shadow
> derivative** = the imaginary part a spectral derivative leaves behind, and the
> **non-linear integrals** = the convolution sum that is `u·∇ω`. If a different
> reading was meant, the solver is unaffected but the framing here should change.

## Where √−1 does real work

In Fourier space every derivative is multiplication by `i·k`:

```
∂/∂x → i·kₓ        ∂/∂y → i·k_y        ∇² → −|k|²
```

So the imaginary unit **is** the derivative operator, and the nonlinear term is a
convolution over wavevector triads `p + q = k`, with an `i·q` inside every term:

```
(u·∇ω)^(k) = Σ_{p+q=k}  û(p) · (i q) ω̂(q)
```

It is evaluated the pseudo-spectral way (derivatives in Fourier space, the product on
the grid, back again), and the test suite checks that this equals the direct triadic
sum exactly (to round-off) on the retained band.

### The shadow

For a real field the Fourier coefficients are Hermitian, `ĝ(−k) = conj ĝ(k)`, so every
`i·k`-derivative is real after the inverse transform: the imaginary part — the
**shadow** — cancels. That is the repository's "√−1 resolves to 1" framing, made
checkable. `SpectralNS2D.imaginary_residue()` measures it from the full complex inverse
transform *before* any real part is taken:

| State | Shadow (relative) |
|---|---|
| after a nonlinear run (`n=32`, `t=1`) | 1.8 × 10⁻¹⁶ |
| after a mesh-driven run, any blob radius | 7 × 10⁻¹⁷ – 3 × 10⁻¹⁶ |
| one planted mode with no conjugate partner | > 10⁻⁴ (the measurement has teeth) |
| differentiating the **Nyquist** mode | 16 (an absolute value on `n=32`) |

The Nyquist mode is the one place `i` does **not** drop out: it has no conjugate
partner, so `i·k` leaves a purely imaginary field. `spectral_derivative` zeroes it by
default (`zero_nyquist=True`); inside the solver the dealiasing mask removes it anyway.

## Numerics

* **Dealiasing:** Orszag 2/3 rule with cutoff `(N − 1) // 3`. `N // 3` is not enough
  when 3 divides `N` (a test builds the `N = 12` counter-example).
* **Time stepping:** integrating-factor RK4 — viscosity and drag are exact, the
  nonlinear and forcing terms are fourth order. Halving `dt` cuts the error 16.3× and
  16.5× on a genuinely nonlinear case.
* **Step size:** `stable_dt = cfl · dx / max|u|`.
* **State** is always in-band: vorticity and forcing are masked on the way in and the
  integrator cannot put anything back (a test checks every masked-off mode stays
  exactly zero).

## What was verified, and against what

| Claim | Independent ground truth | Result |
|---|---|---|
| Solver is correct | Taylor–Green vortex, exact decay `e^{−2νt}`, ν ∈ {0, 0.01, 0.1} | error ≤ 2.2 × 10⁻¹⁵ |
| Forced flow is correct | Kolmogorov flow, exact steady state | error < 10⁻⁸, shrinks 16× per `dt` halving (the residual is the integrator's own O(dt⁴) fixed-point bias) |
| Nonlinear term is the convolution | direct triadic sum over `p + q = k` | equal to round-off |
| Dealiasing matters | same sum without the mask | differs > 10⁻³ |
| Inviscid, unforced flow | energy and enstrophy conserved | conserved |
| Energy budget | `dE/dt = −2νZ` | holds |
| Enstrophy never grows unforced | monotone | holds |
| Incompressible | `max|∇·u|` | 2 × 10⁻¹⁵ |
| Order | error ratio on halving `dt` | 16.3, 16.5 |

77 tests, about 8 s, no network. Test quality was checked by **mutation**: 34 deliberate
bugs (flipped signs, wrong RK4 weights, dropped forcing, the 2/3 rule off by one, the
read-only guard removed, …) were applied one at a time. **All 34 are caught.** The first
run caught 29 and showed 5 gaps; each was closed with a test that fails for the
specific bug (default Nyquist handling, the divergence diagnostic, the CFL step, the
blob width, time bookkeeping). A sixth mutant — the post-step mask — was *equivalent*
(the update cannot leave the band), so the redundant line was removed and the invariant
is tested instead.

Two real defects were found by this process in the module's own diagnostics, and fixed:

1. `tail_fraction` measured the top spectral shell with a **circle**, but the retained
   band is a **square** with corners at `√2·k_max`, so under-resolution in the diagonal
   corners was invisible. It now uses the max-norm; a regression test hides energy in
   a corner and requires it to be seen.
2. The "is the forcing empty?" test used an absolute round-off floor, so a mesh whose
   edge weights were all tiny was treated as empty. It is now relative to the field's own
   scale (scale-invariance is tested).

## The mesh → forcing map

`mesh_forcing_field()` reads `mesh_slm_vocab` (token → torus cell `(i, j)`) and
`mesh_slm_quipu` (edges `src → dst, weight`), then:

1. credits each edge's weight to the cell of **both** endpoints (weighted degree);
2. turns each occupied cell into a Gaussian **vortex blob** of `blob_radius` cells
   (default 2);
3. removes the mean (a periodic flow has zero-mean vorticity forcing);
4. scales the field to a requested rms `amplitude`; resamples band-limited if the
   solver grid is not 64.

Non-finite weights and cells outside the torus are skipped and counted. An empty,
uniform, or missing mesh gives a zero forcing (`source == "empty"`), not an error.

**Why the blob.** A point source is singular and forces the grid scale directly. On the
four-token reference mesh (`n=64, ν=0.02, α=0.2`, amplitude 1, `t=40`) the share of
enstrophy in the top spectral shell is:

| Blob radius (cells) | Top-shell enstrophy | Resolved (≤ 10⁻⁶)? |
|---|---|---|
| 0 (point) | 1.4 × 10⁻² | no |
| 1 | 3.5 × 10⁻⁴ | no |
| **2 (default)** | **1.0 × 10⁻⁷** | yes |
| 3 | 5.5 × 10⁻⁸ | yes |

This is not a licence. At amplitude 2 the same blob leaves 1.7 × 10⁻⁶ and
`is_resolved()` correctly reports **False**. Stronger forcing needs a finer grid. Always
check `diagnostics()["resolved"]` — a result with `resolved == False` is not a
converged answer. (An earlier version of this comment quoted a single figure without its
amplitude; it was re-measured and corrected.)

## One-way and read-only

Nothing flows from the fluid back to the mesh: no training signal, learning rate or
table is touched. This follows the repository's rule for new signals, *instrument before
you couple*. Enforcement, tested rather than promised:

* when the module opens the connection itself it runs `PRAGMA query_only = ON`, so a
  write is impossible even by mistake;
* a connection passed in is used with `SELECT` only — a spy test records every statement
  executed and asserts the set is `{PRAGMA query_only = ON, SELECT…}`;
* a checksum of the mesh tables is unchanged after a run.

## Using it

```python
from src.quipu.navier_stokes_mesh import simulate_from_mesh

out = simulate_from_mesh(t_end=20.0)            # reads the local mesh, read-only
d = out["diagnostics"]
d["resolved"], d["imaginary_residue"], d["max_divergence"], d["energy"]
```

`simulate_from_mesh(t_end, *, cn=None, n=64, nu=1e-2, drag=0.1, amplitude=1.0,
blob_radius=2.0, dt=None, cfl=0.5)`. `drag` is a linear (Ekman) friction that lets a
forced flow settle instead of accumulating energy without bound; set it to 0 to remove it.

## What this is not

* **The map is a modelling choice.** "Edge weight → forcing vorticity" is not a law of
  nature. The mesh is not a fluid; this gives its connectivity a flow to be read through,
  and the flow's energy, enstrophy and spectrum are diagnostics *of that choice*.
* **Two dimensions only.** Global existence and smoothness for 2-D Navier–Stokes are
  classical (Ladyzhenskaya, 1969). Nothing here bears on the three-dimensional regularity
  problem, and nothing here should be read as progress on it.
* **Resolution is checked, not guaranteed** (see above).
* **No version bump.** This adds a module; it changes no existing file.
