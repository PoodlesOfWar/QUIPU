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

98 tests, about 10 s, no network. Test quality was checked by **mutation**: 46 deliberate
bugs (flipped signs, wrong RK4 weights, dropped forcing, the 2/3 rule off by one, the
read-only guard removed, …) were applied one at a time. **All 46 are caught** (34 for
the solver and mesh coupling, 12 for `simulate_resolved`). The first run of the 34 caught 29 and
showed 5 gaps; each was closed with a test that fails for the specific bug (default Nyquist
handling, the divergence diagnostic, the CFL step, the blob width, time bookkeeping). The
first run of the 12 caught 9; the 3 survivors (tail test not required, enstrophy ignored in
the grid comparison, a flow at rest counted as changed) were closed the same way. A sixth mutant — the post-step mask — was *equivalent*
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

## When one grid is not enough: `simulate_resolved`

`simulate_from_mesh` says whether a grid resolved the flow; it cannot do anything when it
did not. `simulate_resolved` walks a grid ladder (default 64, 96, 128, 192) and stops at the
first grid where

1. the top spectral shell is negligible (`is_resolved()`), **and**
2. energy and enstrophy agree with the previous grid to `conv_tol` (default 1e-4).

Condition 1 alone is evidence from one grid; condition 2 makes it a comparison between two.
The mesh is read **once**, at its own 64×64, and resampled band-limited to each rung, so every
rung sees the same forcing even if the mesh is trained meanwhile (a test counts the SELECTs).
A structureless mesh drives no flow and is converged at the first grid. If the ladder runs
out, the finest result comes back with `converged == False` (or `NavierStokesMeshError` with
`strict=True`). The grid must start at ≥ 64 so the forcing is never coarser than the mesh
defines it.

Measured on the amplitude-2 case that `simulate_from_mesh` could only flag (4-token
reference mesh, ν = 0.02, α = 0.2, t = 40):

| n | top-shell enstrophy | resolved | energy |
|---|---|---|---|
| 64 | 1.7 × 10⁻⁶ | no | 1.576912 |
| **96** | **1.1 × 10⁻⁸** | **yes** | **1.576904** |
| 128 | 1.3 × 10⁻¹¹ | yes | 1.576904 |
| 192 | 8.9 × 10⁻¹⁹ | yes | 1.576904 |

The coarse grid's energy was already within 5 × 10⁻⁶ of the converged value, so the 10⁻⁶
tail tolerance is conservative; it is a resolution test, not an error bound. The 192 run
took four minutes; refinement costs roughly n³ (n² points × n steps).

For a *point-source* forcing (blob radius 0) the ladder has to climb further: 64, 96 and 128
all fail the tail test (3.0 × 10⁻², 1.3 × 10⁻², 8.8 × 10⁻⁶ at t = 2) and 192 passes
(2.7 × 10⁻⁹).

This cures **numerical** under-resolution of a 2-D flow, which cannot blow up. A flow whose
scales collapse in finite time would defeat every finite ladder; that is a different problem,
and [NAVIER_STOKES_3D.md](NAVIER_STOKES_3D.md) has the instrument that watches for it.

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
blob_radius=2.0, dt=None, cfl=0.5)`.
`simulate_resolved(t_end, *, cn=None, ladder=(64, 96, 128, 192), …, conv_tol=1e-4, strict=False)`
takes the same physics arguments and returns the per-grid record in `out["runs"]`. `drag` is a linear (Ekman) friction that lets a
forced flow settle instead of accumulating energy without bound; set it to 0 to remove it.

## What this is not

* **The map is a modelling choice.** "Edge weight → forcing vorticity" is not a law of
  nature. The mesh is not a fluid; this gives its connectivity a flow to be read through,
  and the flow's energy, enstrophy and spectrum are diagnostics *of that choice*.
* **Two dimensions only.** Global existence and smoothness for 2-D Navier–Stokes are
  classical (Ladyzhenskaya, 1969), so nothing in this module can blow up. The three-dimensional
  question is a different problem; see [NAVIER_STOKES_3D.md](NAVIER_STOKES_3D.md) for the 3-D
  solver, for what is and is not established about the 2026 forced-blow-up announcement
  (announced and machine-checked by its authors, disputed by an independent audit, not
  independently built here), and for why the unforced equation is untouched by it. Nothing
  in this module is progress on any of it.
* **Resolution is checked, and now acted on, but not guaranteed.** `simulate_resolved`
  refines until two grids agree; if the ladder is too short it says `converged == False`.
* **No version bump.** This adds a module; it changes no existing file.
