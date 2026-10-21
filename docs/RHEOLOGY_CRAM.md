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

## v0.63.0: model-sized terminal pulse

The write-log overshoot (R = 1.383 on single-pulse writes) is quantization: a
write starts at |χ| ≥ ε ≈ 0.010 but one full SET pulse moves a mid-range cell
0.024–0.036. `set_toward` now sizes the pulse that would pass the target by
bisection on a probe copy, so its predicted step equals what remains.

Linear width scaling does not work in this model: heating takes τ_th = 10 ns
and crystallisation is Arrhenius (E_a = 2 eV). Asked fraction → realised/asked
with a linear width: 0.3 → 0.003, 0.5 → 0.10–0.12, 0.7 → 0.40–0.45,
0.9 → 0.80–0.84. Solved widths are 35–48 ns.

Replay of the live log's expected trajectories (4,928 rows, fresh pairs,
Floquet factor 1 in both arms):

| | Full pulses | Sized terminal pulse |
|---|---|---|
| Realised/asked, all writes | 1.066 | 0.990 |
| 1-pulse writes | 1.378 | 0.999 |
| 2–15-pulse writes | 0.986–1.097 | 0.997–1.000 |
| 16–17-pulse writes (cap) | 0.776–0.833 | 0.855–0.880 |
| Mean \|χ\| | 0.0082 | 0.0020 |
| Pulses / energy | 11,441 / 35.5 nJ | 12,748 / 36.7 nJ |

The rolling α_k = −ln R_k over 32 writes stays an audit metric only.

## v0.64.0: per-axis readings and the knot hold

**Herald edges on the CAT axes** (Perceptopoly `herald_cram.NODE_AXIS`):

| Axis | Nodes |
|---|---|
| vision | analog_cognition_waves |
| touch | dynamic_localization, guarded_row_locks |
| smell | non_markovian_memory |
| body | companion_society_mesh, grand_exchange_market |
| brain | topological_network_entropy, infodynamic_compression |
| perception | wireguard_ed25519_gateway, gard_shard_reserve |
| entirety | unitary_ledger_conservation, sqrt_minus_one_orthogonality, entropic_gravity_geometry |

thermodynamic_erasure has no axis yet. An edge reads its endpoint's axis; an
edge across two axes combines them by bits; bibliography-to-bibliography edges
read the ensemble value. Per axis: β_k = atan2(|sin θ_k|, |cos θ_k|),
w_r,k = 0.25 − β_k/2π, bits_k = log2(p_k/ε_k).

**Knot hold.** On the observation record only. Each publish advances the
relative winding W and crossings of all 21 axis pairs and records
hold_factor = e^{μ·2π|W|}, hold_threshold = ε̄·hold_factor and would_freeze
(threshold ≥ 1). μ = `QUIPU_KNOT_MU` (default 0.2). It is never applied to χ.
