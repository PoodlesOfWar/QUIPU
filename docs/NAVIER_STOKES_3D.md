# 3-D forced Navier–Stokes, and an instrument for the resolution limit

`src/quipu/navier_stokes_3d.py` is the three-dimensional counterpart of
[`navier_stokes_mesh.py`](NAVIER_STOKES_MESH.md): a Fourier pseudo-spectral solver for
the forced incompressible equations on the periodic cube `[0, 2π)³`, plus a small
instrument that watches a flow's spectrum and says **when a given grid will stop being
able to hold it**.

It does **not** reproduce, test or rely on any blow-up result. Section 3 says why, and
what is and is not known about the 2026 claim it was asked to be read against.

## 1. The solver

```
∂u/∂t = P[ u × ω ] + ν∇²u + f(x, t),      ω = ∇×u = (i k) × û,      ∇·u = 0
```

`(u·∇)u = ω × u + ∇(|u|²/2)`, and the gradient is absorbed by the pressure, so the rotation
form needs only `u` and `ω`. √−1 does the same work as in 2-D: the curl is `i k ×`, the
Laplacian `−|k|²`, the Leray projector `P = I − k kᵀ/|k|²`. Orszag 2/3 dealiasing
(`max|k_i| ≤ (n−1)//3`), integrating-factor RK4, CFL step on `max(|u₁|+|u₂|+|u₃|)`. The
force is either steady or a function `f(t)` sampled at each Runge–Kutta stage time, so a
time-dependent force can be hosted.

The shadow is measured the same way as in 2-D (`imaginary_residue`, from the full complex
transform of all nine velocity-gradient components): 1.4 × 10⁻¹⁶ after a nonlinear run,
and > 10⁻⁴ if a mode without a conjugate partner is planted.

### Verified against independent truths

| Claim | Ground truth | Result |
|---|---|---|
| Curl and decay | ABC (Beltrami) flow, `∇×u = u`, decays as `e^{−νt}` exactly | error ≤ 1e-13 for ν ∈ {0, 0.05, 0.3} |
| Projection | steady shear `u = (sin y, 0, 0)`, `f = νu`: `u×ω` is a *non-zero* pure gradient the pressure must absorb | stays steady to 1e-13 |
| Nonlinear term (non-Beltrami) | a 2-D flow embedded in 3-D must match the separately verified 2-D solver, in two different planes | ≤ 1e-12 |
| Nonlinear term (independent) | direct triadic sum `Σ_{p+q=k} û_p × (i q × û_q)`, `n = 8` | ≤ 1e-11 |
| Time-dependent force | `u = (1+t²)·ABC` for `f = (a′+νa)·ABC`; wrong stage times would be O(dt) | O(dt⁴): 7e-11 at dt = 0.05, ÷16 per halving |
| Conservation | inviscid energy and helicity | drift ≤ 1e-9, shrinking > 8× per dt-halving (measured 22×, 14×) |
| Energy budget | `dE/dt = −2νZ`, `Z = ½⟨|ω|²⟩` | holds to 1e-4 |
| Order | error ratio on halving dt | 15.0, 18.9 (→ 16) |

(Enstrophy is **not** monotone in 3-D — vortex stretching — so, unlike the 2-D module, it
is not tested for that.)

74 tests, about 8 s, no network. Test quality was checked by mutation: see §5.

## 2. The instrument

A blow-up, if there is one, looks to a numerical method like a collapse of scale:
whatever grid you choose, the solution eventually has structure finer than it. What can
be done is to measure how fast the spectrum is losing its decay, and extrapolate to the
time a given grid is lost.

For a solution analytic in a strip of half-width `δ(t)` the energy spectrum decays like
`E(k) ~ k^b e^{−2δk}` (Sulem–Sulem–Frisch). Then:

* `analyticity_fit(spectrum, k_lo, k_hi)` → `(δ, b)` by least squares on
  `ln E = a + b ln k − 2δk`. On spectra built with known `δ` it is exact to 1e-9. It is
  applied to the **per-mode** spectrum: shell sums carry the lattice count `≈ 4πk²`, which
  `b ln k` absorbs only approximately (a measured 6% bias, pinned by a test).
* `required_width(n, tol, b=…)` → the `δ` at which an `n³` grid stops being resolved. It
  is computed on the solver's own lattice with the solver's own test, so
  `is_resolved()` flips exactly there: the tail share at that `δ` equals `tol` to 1e-3
  relative, and `δ` ±2% either side lands on either side of `is_resolved`. It falls as the
  grid is refined (1.75, 0.90, 0.60 for `n` = 16, 32, 48 at `b = −3`, `tol = 1e-6`).
* `resolution_forecast(times, widths, n, tol, b=…)` fits `δ(t) = s (T* − t)` and reports
  `t_singularity`, `t_resolution_lost`, `already_lost`, and `r_squared` (a poor value means
  don't trust the extrapolation).
* `decay_exponent(spectrum, k_lo, k_hi)` → `(θ, c)` for `ln E = a + b ln k − c k^θ`;
  `θ = 1` analytic, `θ < 1` Gevrey. See §4.

**Stress test (synthetic).** Spectral states with `δ(t) = 1.6 (1 − t/0.9)` on `n = 32`:
measured from the states, `δ` is recovered exactly; the forecast gives `T* = 0.9` and the
grid lost at `t = 0.3965`; and the solver's own `is_resolved()` flips at that time
(true at 0.9×, 0.98×, false at 1.02×, 1.1×). This tests the instrument, not any claim about
the equations: those states are not solutions of Navier–Stokes.

## 3. The 2026 forced-blow-up claim

**What exists.** OpenAI published a manuscript and a Lean 4 formalization of finite-time
blow-up for the *forced* 3-D incompressible equations, for every ν > 0, on ℝ³ and on the
torus. I cloned the repository (`github.com/openai/NavierStokesAndEuler`, commit
`f9e8bc5`, 2026-09-10, Lean 4.34.0-rc2 + Mathlib, about 640 000 lines) and read its
statements rather than a description of them.

* The README states: on ℝ³, *"smooth initial data and forcing for which no global smooth
  solution with uniformly bounded kinetic energy exists"*; on the torus, *"no global smooth
  solution exists."* It presents these as alternatives (C) and (D) of the Clay problem
  description. I have not assessed whether the Clay Institute accepts that framing, and
  nothing here depends on it.
* The final theorem, as stated:
  ```
  theorem navier_stokes_breakdown_R3 (nu : ℝ) (hnu : nu > 0) :
      ∃ (u₀ : ℝ³ → ℝ³) (f : ℝ³ → ℝ → ℝ³),
        InitialVelocityConditionDecay u₀ ∧ ForceConditionDecay f ∧
        ¬ (∃ v p, NavierStokesExistenceAndSmoothnessRn nu u₀ f v p)
  ```
  against definitions copied from Google DeepMind's `formal-conjectures`. The whole-space
  proof takes **zero initial data** and a force that is smooth, spatially compactly
  supported and compactly supported in time; the companion theorem
  `theorem_1_1_with_maximalH3` adds that the maximal classical H³ lifespan is exactly 1.
* A textual scan of the proof tree finds no `sorry`, `axiom`, `opaque`, `native_decide`
  or `unsafe`. (The `ComparatorChallenges/` reference file keeps its `sorry` placeholders
  on purpose.) **I did not build it**: that needs a Lean + Mathlib toolchain and the
  Mathlib cache, which this sandbox cannot fetch. A grep is not a proof check.

**What I could not verify.** The sandbox cannot reach the OpenAI PDFs, arXiv or Zenodo, so
I have read none of the papers. Search summaries (secondary, and one auto-generated) report
that an October 2026 audit, arXiv:2610.08144 (Bastounis, Circelli, Hansen), argues the
compiled repository does not faithfully translate the manuscript (a "Lemma 8.6 derivative
mismatch"), that a pressure-periodicity erratum was involved, and that as of mid-September
there was no refereed independent verification. I cannot tell you whether that audit is
right. Treat the claim as **announced, machine-checked by its authors, and disputed**, not
as settled either way.

**What it would not settle even if it holds.** It concerns a specifically constructed
smooth force. The unforced equation (Clay's (A) and (B)) is untouched, and so is the
behaviour of *this repository's* forcing, which comes from the quipu graph, not from that
construction.

**Why it is not used as a benchmark.** A benchmark needs an explicit `f(x, t)` to sample.
The Lean development proves existence: at least some of its constants are `Classical.choose`d
(e.g. `mainBound`), and the construction runs through a schedule of infinitely many scales.
I could not extract a force from it, and I will not invent one and call it theirs.

**What follows for the resolution limit, conditionally.** On ℝ³ the docstring of
`ComparatorR3Theorem.lean` says the theorem *"constructs one compactly supported flow with
a uniform energy bound up to time one"*, call the bound `E`; and `theorem_1_1_with_maximalH3`
gives a maximal classical H³ lifespan of exactly 1, where (I read the definition)
`IsMaximalClassicalH3Solution` means a classical H³ solution on `[0, 1]` that has no
classical H³ extension. By the standard H³ continuation criterion (not proved in that
file, and not checked by me), `‖u(t)‖_{H³} → ∞` as `t → 1`. Modes at or below any fixed
cutoff `K` contribute at most `(1+K²)³ · 2E` to that norm, which is bounded; so the
divergence must come from modes above `K`, **for every `K`**. No fixed grid could resolve
that solution up to `t = 1`, and refining `n` would only delay the loss. That is the sense
in which, *if the claim holds*, the resolution limit is a property of the object rather than a
numerical inconvenience. It is also why [`simulate_resolved`](NAVIER_STOKES_MESH.md) can only
cure the 2-D under-resolution it was built for (2-D flows do not blow up), while this
module can only *forecast* a 3-D one.

## 4. Where the instrument is the wrong tool

* It does not decide whether a singularity exists. A shrinking `δ` is consistent with
  blow-up and with a sharp but finite event.
* **Its model fails for smooth, non-analytic forcing.** A compactly supported `C^∞` force
  cannot be analytic, so the response is Gevrey: the spectrum decays like `e^{−c k^θ}` with
  `θ < 1`, and the strip width is zero from the start. `decay_exponent` fits `θ` so that
  mismatch is visible rather than hidden inside a misleading `δ`, and it separates the two
  classes on exact synthetic spectra (θ = 1, 0.7, 0.5 recovered to 0.02). For a forced
  construction of the kind in §3, quote `θ`, not `δ`.
* Extrapolating a straight line in `δ(t)` is practice, not theorem. Check `r_squared`.

## 5. Verification of the verifier

51 mutants of the module (flipped curl and nonlinear signs, a dropped or unprojected
nonlinear term, wrong stage times, RK4 weights, the cutoff, the dealiasing of input and
output, the energy / enstrophy / helicity formulas, the CFL speed, the fit's design matrix,
the bisection direction, the forecast formulas, input validation, …) were applied one at a
time. The first run caught 50; the survivor (`already_lost` read the first width instead of
the latest) was closed with a test that fails for it, so all 51 are caught.

## 6. Notes

* The cube is `2π`-periodic; the Lean statements use period 1. Navier–Stokes rescales
  between the two by changing ν and the time unit, so this is cosmetic.
* The mesh lives on a 2-D torus, so there is no mesh-driven 3-D forcing here.
* No version bump; the change adds files only.
