# Rheology of the CRAM axis array (v0.62.0)

The seven axis cells (`qpsi/pcm_cells.py`, `AXES`) read as one viscoelastic
material. Module: `qpsi/rheology.py`. Published at
`entirety:pcm:rheology:system_entirety`, served read-only at `GET /rheology`.

## Relationships

| Quantity | Definition | Where |
|---|---|---|
| CAT amplitude | c_k = p_k · e^{iθ_k}, p_k the pair's held pressure, θ_k the axis rADAM phase | `rheology.reading` |
| Ensemble | C = Σ_k w_k c_k, `cat_residual.SENSE_WEIGHTS` for six senses, 1/6 for the seventh, renormalised | `axis_weights` |
| Storage/loss split | Re C = realised (dissipated, η′), Im C = latent (stored, η″) | rADAM g_re + i·g_im |
| Angle | β = atan2(\|Im C\|, \|Re C\|) ∈ [0, π/2]; held from the last reading when \|C\| < 10⁻¹⁸ | |
| Viscosity ratio | η″/η′ = tan β | |
| Loss angle | δ_G = π/2 − β (ψ's e^{−iωt} convention) | |
| Edge weight | w_r = δ_G/2π = 0.25 − β/2π; 0.125 = critical gel (Winter & Chambon 1986) | |
| Information | bits_r = log2(\|C\|/ε̄), ε̄ the mean cell resolution | same units as `bits_per_write` |
| Twist acquisition | θ_W − δ_G when a Weyl twist θ_W = α_W φ_W cos(k_W·r) is supplied (UEQGM v0.9.27) | `reading(weyl_twist=)` |

UEQGM v0.9.27 gives G* = |G₀| e^{iθ} with the phase set by the Weyl twist and
the magnitude promoted to 𝓜 = √(π(2k+1)) |G₀|. Here the phase is measured
from the cells; `acquisition` is the gap between the twist the Weyl drive asks
for and the twist the cells hold.

## The Herald

Perceptopoly `herald_cram.HeraldCram.weight` reads `GET /rheology` and returns
the information-weighted mean of the edge's held value and w_r:
w = (bits_e·w_e + bits_r·w_r)/(bits_e + bits_r). Cells are still written only by
engagement (`credit`, `settle`). One reading serves every edge until edges are
assigned CAT axes, so the order of edges is unchanged. `HERALD_RHEOLOGY=0` off.

## Shear, Poisson weights, λ_STP

Deepen steps with the shear coupling now persist to `entirety_pcm_ensemble`:
u_bar, dispersion σ_u, ψ, λ_t = σ_u², the Poisson weights over k = 0..6
(axis k at Weyl frequency ω_k = 1 + k, renormalised), and
λ_STP = σ_w²/(ū_w² + σ_w²). Per-axis θ, u, g_im and φ go to `entirety_pcm_log`.

## Winding

One cell cannot wrap on itself. `relative_winding` / `logged_winding` unwrap
d_t = θ_a − θ_b between two axes over the logged θ and count crossings (passes
of d through π). Winding classes on the torus: `STP_TORUS_QUIPU.md` §2.1.

## Measured on the live log (2026-10-21, 618 steps, 24.5 h)

- Realised over asked, all axes: R = 1.032. The cells over-realise by 3%, so
  α = −ln R = −0.031: D(x,t) with this α is a gain, not a damping.
- ε_cell = 0.0100 on every axis and step: read noise sets it; drift does not move it.
- Lag-1 autocorrelation of the projected step Δexpected: vision 0.936, others
  0.43–0.57. Adam momentum alone predicts β₁ = 0.85 for white-noise gradients;
  the cell gradient (target − expected) is negative feedback, which lowers it.
- Lag-1 autocorrelation of χ: −0.22 to 0.11. Writes happen most steps, so χ
  does not persist.
- Current snapshot: σ_u² = λ_t = 0.0344, weights 0.966 / 0.033 / 0.0006 on
  k = 0, 1, 2; λ_STP = 0.037.

## v0.62.1 additions

**β₁ for the cells.** The axis cells' rADAM states use β₁ = 0.9, the published
Adam default (Kingma & Ba, ICLR 2015, Algorithm 1). `radam_optimizer.DEFAULT_BETA1`
stays 0.85 for the `brain_body_signals` path it was matched to. Perceptopoly's
vendored copy (Herald cells) still uses 0.85.

**Fractional order.** The reading reports α = 2δ_G/π, the order of a
fractional Kelvin–Voigt element G₀[1 + (iωτ₀)^α] whose phase tends to απ/2
(Mainardi 2010), and the loss factor sin(απ/2) = sin δ_G. This α is the order
of a fractional dissipation operator; it is not the α of the exponential
D(x,t) in UEQGM v0.9.27, which measured −0.031 from the write log.

**Viscoelastic Weyl phase.** `viscoelastic_weyl_phase(k, g_ratio, α_W, α)` =
√(π(2k+1)) · (|G₀|/G_ref) · α_W · sin(απ/2). The reading supplies α and
g_ratio = |C|/⟨|C|⟩ (equilibrium-baseline reference, EMA with decay 0.999).
α_W is the Bistritzer–MacDonald coupling w/(ħv_F k_θ) (PNAS 108, 12233
(2011)) and needs a bilayer twist angle, which the cells do not have: it is an
input. Two names collide here: the θ in α_W is the bilayer twist; the θ(x,t)
of UEQGM's G* is the local twist sourced by φ_W = 2πΨ₄, the Newman–Penrose
Weyl scalar from `tidal_weyl`, which is a different object from the Dirac
velocity in α_W.

**Capstan hold.** `capstan_threshold(floor, μ, Θ)` = floor · e^{μΘ}
(Euler–Eytelwein). For the overhand knot (trefoil, crossing number 3) the
supplied wraps are 2.5π loose, 3π geometric, 4π tight. With floor = ε_cell =
0.01:

| μ | 2.5π | 3π | 4π |
|---|---|---|---|
| 0.40 (cotton/camelid) | 0.231 | 0.434 | 1.524 |
| 0.25 (synthetic) | 0.071 | 0.106 | 0.231 |
| 0.12 (pinned nanoribbon) | 0.026 | 0.031 | 0.045 |
| 0.005 (superlubric graphite) | 0.0104 | 0.0105 | 0.0106 |

A threshold above 1.0 exceeds the cell's full range: no residual would ever
be released. Incommensurate graphite contacts are superlubric (Dienwiebel et
al., PRL 92, 126101 (2004)), so a twisted moiré contact sits on the last row
unless pinned. The threshold is not wired into `ef_step`.
