"""
    SCMPredictor

**SCM — Saturation-Cycle Model**: the technology adoption → tangential
atrophy → modular resuscitation predictor. Julia port of
`pipeline/src/quipu/research/tech_adoption_resuscitation.py`, numerically
verified against that module's reference run (see [`selftest`](@ref)).

Core equations
--------------
1. **Bass diffusion** (closed form):
   `F(t) = (1 − e^{−(p+q)t}) / (1 + (q/p)e^{−(p+q)t})`, `A(t) = m·F(t)`,
   `dA/dt = p(m−A) + (q/m)A(m−A)`.
2. **Tangential atrophy**: `dS/dt = −γ(A/m)S` ⇒ `S(t) = S0·e^{−γG(t)}` with
   the closed form `G(t) = ∫0^t F = t + ((1+c)/(bc))·log((1+c·e^{−bt})/(1+c))`,
   `b = p+q`, `c = q/p`.
3. **NHPP resuscitation intensity**:
   `ln λ = β0 + β1·sat + β2·μ·max(0, sat−θ)·(S0−S)/S0 + β3·D(t)`,
   `Λ(T) = ∫0^T λ dt`.
4. **Unified viability — Weyl 1-D contraction form**:
   `V(t) = A·(S/S0 + φμ·Λ/(Λ+κ)) = A·⟨w, Ψ⟩` with `w = (0,0,0,1,φμ)` under
   the Weyl scalar mapping (Ψ0 = adoption flux, Ψ1 = innovation breadth,
   Ψ2 = Λ bulk mass, Ψ3 = S/S0, Ψ4 = Λ/(Λ+κ)) — the same 5-scalar tensor as
   `ueqgm_engine.weyl_scalar_tensor` (local port; formulas identical).

Imaginary time (`t ± n·√(-1)`)
------------------------------
All constituents are analytic in the strip `|Im z| < π/(2(p+q))` (Bass poles
at `±π/(p+q)`; the half-strip keeps `Re(1+c·e^{−bz}) ≥ 1`, so the principal
log branch *is* the continuation and `|F| ≤ 2`). `max(0,·)` is replaced by
the analytic surrogate `softplus_k` for the Wick-rotated intensity;
`Λ̃(t+in)` is computed by Cauchy contour (real leg + vertical leg). Schwarz
reflection `Ṽ(t−in) = conj(Ṽ(t+in))` and the contraction identity
`Ṽ = Ã·⟨w, Ψ̃⟩` are verified on every prediction record.

Variations: Ogata-thinning event simulation, Norton–Bass multi-generation
substitution with density spillover, one-at-a-time elasticities, and an
MRP-style time-phased module/service demand hook.
"""
module SCMPredictor

using Random

export
    SCMParams,
    reparam,
    bass_fraction,
    adoption,
    adoption_rate,
    bass_cumulative_fraction,
    tangential_health,
    log_intensity,
    intensity,
    intensity_analytic,
    simulate,
    simulate_resuscitation_events,
    multi_generation,
    sensitivity_elasticities,
    mrp_module_demand,
    weyl_scalar_tensor,
    wavefunction_overlap,
    weyl_psi_trajectory,
    max_imag_offset,
    viability_weights,
    wick_viability,
    imaginary_time_predictions,
    emission_mode,
    langevin_scm_step,
    selftest

const SOFTPLUS_K = 48.0

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
"""Rest-of-Ecosystem innovation density D(t) (complex-safe, analytic)."""
default_innovation_density(t) =
    0.35 + 0.55 * (1 - exp(-0.07 * t)) * (1 + 0.25 * sin(2pi * t / 7))

"""
    SCMParams(; m=1000.0, p=0.025, q=0.38, gamma=0.12, mu=0.78, s0=1.0,
              beta0=-2.8, beta1=2.1, beta2=3.2, beta3=1.4,
              phi=0.55, kappa=18.0, theta=0.72,
              density=default_innovation_density)

Framework parameters; defaults are the specification's reference run.
"""
Base.@kwdef struct SCMParams
    m::Float64 = 1000.0       # market potential
    p::Float64 = 0.025        # innovation coefficient
    q::Float64 = 0.38         # imitation coefficient
    gamma::Float64 = 0.12     # atrophy sensitivity
    mu::Float64 = 0.78        # modularity index in [0, 1]
    s0::Float64 = 1.0         # initial tangential health
    beta0::Float64 = -2.8     # NHPP intercept
    beta1::Float64 = 2.1      # NHPP saturation gain
    beta2::Float64 = 3.2      # NHPP modular-atrophy interaction gain
    beta3::Float64 = 1.4      # NHPP external-density gain
    phi::Float64 = 0.55       # max viability boost from resuscitation
    kappa::Float64 = 18.0     # resuscitation benefit half-saturation
    theta::Float64 = 0.72     # atrophy onset threshold on sat(t)
    density::Function = default_innovation_density
end

"""
    reparam(prm; kwargs...) -> SCMParams

Copy `prm` with the given fields replaced (Julia analogue of
`dataclasses.replace`).
"""
function reparam(prm::SCMParams; kwargs...)
    props = NamedTuple{fieldnames(SCMParams)}(
        ntuple(i -> getfield(prm, i), fieldcount(SCMParams)))
    return SCMParams(; merge(props, values(kwargs))...)
end

bass_b(prm::SCMParams) = prm.p + prm.q
bass_c(prm::SCMParams) = prm.q / prm.p

"""Safe analytic half-strip bound `π/(2(p+q))` for Wick offsets |n|."""
max_imag_offset(prm::SCMParams) = pi / (2.0 * bass_b(prm))

"""Weyl contraction weights `w` with `V = A·⟨w, Ψ⟩`."""
viability_weights(prm::SCMParams) = (0.0, 0.0, 0.0, 1.0, prm.phi * prm.mu)

"""
    clip01(x) -> Float64

Clamp real `x` to `[0, 1]`; non-finite maps to `0.0`.
"""
function clip01(x)::Float64
    v = try
        Float64(x)
    catch
        return 0.0
    end
    isfinite(v) || return 0.0
    return clamp(v, 0.0, 1.0)
end

# ---------------------------------------------------------------------------
# 1-2. Closed forms — Bass diffusion + tangential atrophy (complex-safe)
# ---------------------------------------------------------------------------
"""Cumulative adoption fraction F(z) — analytic in the safe strip."""
function bass_fraction(z, prm::SCMParams)
    e = exp(-bass_b(prm) * z)
    return (1 - e) / (1 + bass_c(prm) * e)
end

"""Cumulative adoption A(z) = m·F(z)."""
adoption(z, prm::SCMParams) = prm.m * bass_fraction(z, prm)

"""Instantaneous adoption rate dA/dt from the Bass ODE (closed form)."""
function adoption_rate(z, prm::SCMParams)
    a = adoption(z, prm)
    return prm.p * (prm.m - a) + (prm.q / prm.m) * a * (prm.m - a)
end

"""G(z) = ∫0^z F(s) ds, closed form (see module docstring derivation)."""
function bass_cumulative_fraction(z, prm::SCMParams)
    b, c = bass_b(prm), bass_c(prm)
    return z + ((1 + c) / (b * c)) * (log(1 + c * exp(-b * z)) - log(1 + c))
end

"""S(z) = S0·exp(−γ·G(z)) — exact solution of dS/dt = −γ(A/m)S."""
tangential_health(z, prm::SCMParams) =
    prm.s0 * exp(-prm.gamma * bass_cumulative_fraction(z, prm))

# ---------------------------------------------------------------------------
# 3. NHPP resuscitation intensity
# ---------------------------------------------------------------------------
"""Analytic surrogate of max(0, x); exact as k → ∞. Complex-safe."""
softplus(x, k::Real = SOFTPLUS_K) = log(1 + exp(k * x)) / k

"""ln λ(z); `analytic=true` swaps max(0,·) → softplus (for Wick rotation)."""
function log_intensity(z, prm::SCMParams; analytic::Bool = false)
    sat = bass_fraction(z, prm)
    s = tangential_health(z, prm)
    atrophy = analytic ? softplus(sat - prm.theta) : max(0.0, real(sat) - prm.theta)
    return (prm.beta0 + prm.beta1 * sat
            + prm.beta2 * prm.mu * atrophy * (prm.s0 - s) / prm.s0
            + prm.beta3 * prm.density(z))
end

"""λ(z) = exp(ln λ(z)) — exact max(0,·) form (real axis)."""
intensity(z, prm::SCMParams) = exp(log_intensity(z, prm))

"""λ̃(z) — holomorphic softplus form used for complex-time contours."""
intensity_analytic(z, prm::SCMParams) = exp(log_intensity(z, prm; analytic = true))

"""Cumulative trapezoid with leading 0 (complex-safe)."""
function cumtrapz(y::AbstractVector, x::AbstractVector)
    out = zeros(promote_type(eltype(y), Float64), length(y))
    @inbounds for i in 2:length(y)
        out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    end
    return out
end

"""Linear interpolation of (xs, ys) at scalar x (xs ascending)."""
function linterp(x::Real, xs::AbstractVector, ys::AbstractVector)
    x <= xs[1] && return float(ys[1])
    x >= xs[end] && return float(ys[end])
    j = searchsortedlast(xs, x)
    frac = (x - xs[j]) / (xs[j + 1] - xs[j])
    return ys[j] + frac * (ys[j + 1] - ys[j])
end

# ---------------------------------------------------------------------------
# 4. Real-time simulation — the reference prediction run
# ---------------------------------------------------------------------------
"""
    simulate(prm=SCMParams(); horizon=40.0, n_steps=481) -> NamedTuple

Deterministic reference run of equations 1-4 on [0, horizon] via the closed
forms. Returns arrays `t, A, rate, sat, S, D, lam, Lambda, resusc_benefit,
V` plus headline metrics.
"""
function simulate(prm::SCMParams = SCMParams(); horizon::Real = 40.0,
                  n_steps::Int = 481)
    t = collect(range(0.0, horizon; length = n_steps))
    a = adoption.(t, Ref(prm))
    sat = clamp.(a ./ prm.m, 0.0, 1.0)
    s = tangential_health.(t, Ref(prm))
    lam = intensity.(t, Ref(prm))
    big_lambda = cumtrapz(lam, t)
    benefit = prm.phi .* prm.mu .* big_lambda ./ (big_lambda .+ prm.kappa)
    v = a .* (s ./ prm.s0 .+ benefit)

    sat_idx = something(findfirst(>=(0.90), sat), n_steps)
    peak_idx = argmax(lam)
    return (
        t = t, A = a, rate = adoption_rate.(t, Ref(prm)), sat = sat, S = s,
        D = prm.density.(t), lam = lam, Lambda = big_lambda,
        resusc_benefit = benefit, V = v,
        t_sat90 = t[sat_idx], t_peak_lambda = t[peak_idx],
        peak_lambda = lam[peak_idx], S_final = s[end],
        Lambda_final = big_lambda[end], V_final = v[end],
    )
end

# ---------------------------------------------------------------------------
# Variation A — stochastic resuscitation events via Ogata thinning
# ---------------------------------------------------------------------------
"""
    simulate_resuscitation_events(prm; horizon=40.0, seed=42) -> NamedTuple

One NHPP realisation of resuscitation event times by Ogata thinning:
homogeneous λ_max proposals accepted w.p. λ(t)/λ_max. `E[count] = Λ(horizon)`.
"""
function simulate_resuscitation_events(prm::SCMParams = SCMParams();
                                       horizon::Real = 40.0, seed::Int = 42)
    rng = MersenneTwister(seed)
    grid = collect(range(0.0, horizon; length = 2049))
    lam_max = maximum(intensity.(grid, Ref(prm))) * 1.05
    events = Float64[]
    t = 0.0
    while true
        t += -log(rand(rng)) / lam_max      # Exponential(1/λ_max) arrival
        t > horizon && break
        rand(rng) < intensity(t, prm) / lam_max && push!(events, t)
    end
    expected = cumtrapz(intensity.(grid, Ref(prm)), grid)[end]
    return (events = events, count = length(events), expected = expected,
            lambda_max = lam_max)
end

# ---------------------------------------------------------------------------
# Variation B — Norton–Bass multi-generation substitution
# ---------------------------------------------------------------------------
"""
    multi_generation(generations, base=SCMParams(); horizon=40.0,
                     n_steps=481, spillover=0.25) -> NamedTuple

Norton–Bass successive generations, each `(m, p, q, launch[, label])`
NamedTuple. Recursive in-use form `pot_g = m_g + F_{g-1}·pot_{g-1}`,
`inuse_g = F_g·pot_g·(1 − F_{g+1})`; every generation runs its own
atrophy/NHPP/viability pipeline with density lifted by sibling saturation
`D_g = D·(1 + spillover·Σ_{h≠g} sat_h)`. Returns per-gen arrays + `V_total`.
"""
function multi_generation(generations::Vector{<:NamedTuple},
                          base::SCMParams = SCMParams();
                          horizon::Real = 40.0, n_steps::Int = 481,
                          spillover::Real = 0.25)
    t = collect(range(0.0, horizon; length = n_steps))
    fracs = Vector{Vector{Float64}}()
    for gen in generations
        gp = reparam(base; m = gen.m, p = gen.p, q = gen.q)
        launch = hasproperty(gen, :launch) ? gen.launch : 0.0
        push!(fracs, [ti > launch ? bass_fraction(ti - launch, gp) : 0.0 for ti in t])
    end

    pots = Vector{Vector{Float64}}()
    for (g, gen) in enumerate(generations)
        pot = g == 1 ? fill(float(gen.m), n_steps) :
              gen.m .+ fracs[g - 1] .* pots[g - 1]
        push!(pots, pot)
    end
    inuse = [fracs[g] .* pots[g] .*
             (g < length(generations) ? 1.0 .- fracs[g + 1] : fill(1.0, n_steps))
             for g in 1:length(generations)]

    gens_out = NamedTuple[]
    v_total = zeros(n_steps)
    for (g, gen) in enumerate(generations)
        sat_g = fracs[g]
        sibling = sum(fracs[h] for h in 1:length(generations) if h != g;
                      init = zeros(n_steps))
        d_g = base.density.(t) .* (1.0 .+ spillover .* sibling)
        s_g = base.s0 .* exp.(-base.gamma .* cumtrapz(sat_g, t))
        atrophy = max.(0.0, sat_g .- base.theta)
        lam_g = exp.(base.beta0 .+ base.beta1 .* sat_g
                     .+ base.beta2 .* base.mu .* atrophy .* (base.s0 .- s_g) ./ base.s0
                     .+ base.beta3 .* d_g)
        big_l = cumtrapz(lam_g, t)
        v_g = inuse[g] .* (s_g ./ base.s0
                           .+ base.phi .* base.mu .* big_l ./ (big_l .+ base.kappa))
        v_total .+= v_g
        label = hasproperty(gen, :label) ? gen.label : "gen$(g)"
        push!(gens_out, (label = label, in_use = inuse[g], sat = sat_g,
                         S = s_g, lam = lam_g, Lambda = big_l, V = v_g))
    end
    return (t = t, generations = gens_out, V_total = v_total)
end

# ---------------------------------------------------------------------------
# Variation C — sensitivity analysis (elasticities of V at the horizon)
# ---------------------------------------------------------------------------
const SENS_FIELDS = (:p, :q, :gamma, :mu, :beta0, :beta1, :beta2, :beta3,
                     :phi, :kappa, :theta)

"""
    sensitivity_elasticities(prm; horizon=40.0, rel_step=1e-3)
        -> Vector{Pair{Symbol,Float64}}

One-at-a-time elasticities `ε_x = (∂V(T)/∂x)·(x/V(T))` (central difference),
sorted descending by |ε|.
"""
function sensitivity_elasticities(prm::SCMParams = SCMParams();
                                  horizon::Real = 40.0, rel_step::Real = 1e-3)
    v_base = simulate(prm; horizon = horizon).V_final
    out = Pair{Symbol,Float64}[]
    for name in SENS_FIELDS
        x = getfield(prm, name)
        h = x != 0 ? abs(x) * rel_step : rel_step
        v_hi = simulate(reparam(prm; (name => x + h,)...); horizon = horizon).V_final
        v_lo = simulate(reparam(prm; (name => x - h,)...); horizon = horizon).V_final
        push!(out, name => v_base != 0 ? ((v_hi - v_lo) / (2h)) * (x / v_base) : 0.0)
    end
    return sort(out; by = kv -> -abs(kv.second))
end

# ---------------------------------------------------------------------------
# Variation D — MRP-style time-phased module/service demand hook
# ---------------------------------------------------------------------------
"""
    mrp_module_demand(prm; horizon=40.0, service_rate=0.02,
                      modules_per_event=6.0, lead_time=2, on_hand=40.0)
        -> Vector{NamedTuple}

MRP hook: per yearly period `gross_k = service_rate·Ā_k +
modules_per_event·ΔΛ_k` (installed-base service + resuscitation-triggered
module demand), lot-for-lot planned releases offset by `lead_time`.
"""
function mrp_module_demand(prm::SCMParams = SCMParams(); horizon::Real = 40.0,
                           service_rate::Real = 0.02,
                           modules_per_event::Real = 6.0,
                           lead_time::Int = 2, on_hand::Real = 40.0)
    sim = simulate(prm; horizon = horizon, n_steps = Int(horizon) * 12 + 1)
    edges = collect(0.0:1.0:horizon)
    rows = NamedTuple[]
    projected = float(on_hand)
    for k in 1:(length(edges) - 1)
        a_mid = linterp(edges[k] + 0.5, sim.t, sim.A)
        dlam = linterp(edges[k + 1], sim.t, sim.Lambda) -
               linterp(edges[k], sim.t, sim.Lambda)
        gross = service_rate * a_mid + modules_per_event * dlam
        net = max(0.0, gross - max(projected, 0.0))
        projected = projected - gross + net
        push!(rows, (period = k - 1, gross_req = round(gross; digits = 2),
                     net_req = round(net; digits = 2),
                     projected_on_hand = round(projected; digits = 2),
                     planned_release_period = max(0, k - 1 - lead_time),
                     planned_release_qty = round(net; digits = 2)))
    end
    return rows
end

# ---------------------------------------------------------------------------
# Weyl Ψ 1-D tensor (local ports of ueqgm_engine formulas)
# ---------------------------------------------------------------------------
"""
    weyl_scalar_tensor(signal_flux, topic_entropy, corpus_volume,
                       mesh_alignment, remnant_score) -> NTuple{5,Float64}

Newman-Penrose Weyl scalars (Ψ0…Ψ4), each clipped to [0, 1]; Ψ2 uses the
saturating `ρ/(ρ+1)` transform. Port of `ueqgm_engine.weyl_scalar_tensor`.
"""
function weyl_scalar_tensor(signal_flux, topic_entropy, corpus_volume,
                            mesh_alignment, remnant_score)
    rho = max(0.0, float(corpus_volume))
    return (clip01(signal_flux), clip01(topic_entropy),
            clip01(rho / (rho + 1.0)), clip01(mesh_alignment),
            clip01(remnant_score))
end

"""
    wavefunction_overlap(a, b) -> Float64

`|⟨ψ_a|ψ_b⟩|² = cos²θ(a, b)`. Port of `ueqgm_engine.wavefunction_overlap`.
"""
function wavefunction_overlap(a, b)::Float64
    (isempty(a) || length(a) != length(b)) && return 0.0
    dot = sum(ai * bi for (ai, bi) in zip(a, b))
    na = sqrt(sum(ai * ai for ai in a))
    nb = sqrt(sum(bi * bi for bi in b))
    (na == 0.0 || nb == 0.0) && return 0.0
    return round((dot / (na * nb))^2; digits = 6)
end

"""
    weyl_psi_trajectory(sim, prm) -> Matrix{Float64}  (n_steps × 5)

Map a [`simulate`](@ref) run onto the Weyl 5-vector per step:
Ψ0 = rate/max, Ψ1 = D/max, Ψ2 = Λ (ρ/(ρ+1) inside), Ψ3 = S/S0,
Ψ4 = Λ/(Λ+κ). Note `V(t) = A·(Ψ3 + φμ·Ψ4)` — the unified equation is the
contraction with [`viability_weights`](@ref).
"""
function weyl_psi_trajectory(sim::NamedTuple, prm::SCMParams = SCMParams())
    rate_max = max(maximum(sim.rate), eps())
    d_max = max(maximum(sim.D), eps())
    n = length(sim.t)
    psi = zeros(n, 5)
    for i in 1:n
        psi[i, :] .= weyl_scalar_tensor(
            sim.rate[i] / rate_max, sim.D[i] / d_max, sim.Lambda[i],
            sim.S[i] / prm.s0, sim.Lambda[i] / (sim.Lambda[i] + prm.kappa))
    end
    return psi
end

"""Analytic (unclipped) continuation of the Ψ mapping at complex z."""
function analytic_weyl_psi(z::Complex, prm::SCMParams, rate_max::Real,
                           d_max::Real, big_lambda::Complex)
    return ComplexF64[
        adoption_rate(z, prm) / rate_max,
        prm.density(z) / d_max,
        big_lambda / (big_lambda + 1.0),
        tangential_health(z, prm) / prm.s0,
        big_lambda / (big_lambda + prm.kappa),
    ]
end

# ---------------------------------------------------------------------------
# Imaginary-time predictions — t_{±n}·√(-1) against the Weyl 1-D tensor
# ---------------------------------------------------------------------------
"""Λ̃(anchor + i·offset) by contour: real-axis leg + vertical leg."""
function lambda_contour(anchor::Real, offset::Real, prm::SCMParams;
                        n_real::Int = 801, n_imag::Int = 201)
    tr = collect(range(0.0, anchor; length = n_real))
    leg_real = cumtrapz(intensity_analytic.(tr, Ref(prm)), tr)[end]
    offset == 0.0 && return ComplexF64(leg_real)
    s = collect(range(0.0, offset; length = n_imag))
    leg_imag = cumtrapz(intensity_analytic.(anchor .+ 1im .* s, Ref(prm)), s)[end]
    return ComplexF64(leg_real) + 1im * leg_imag
end

"""
    wick_viability(z, prm=SCMParams()) -> ComplexF64

Analytically continued viability Ṽ(z) at complex time `z = t + i·n`.
Throws outside the safe strip `|Im z| < π/(2(p+q))`.
"""
function wick_viability(z::Complex, prm::SCMParams = SCMParams())
    abs(imag(z)) >= max_imag_offset(prm) && throw(ArgumentError(
        "|Im z|=$(abs(imag(z))) outside analytic strip (< $(max_imag_offset(prm)))"))
    big_lambda = lambda_contour(real(z), imag(z), prm)
    a = adoption(z, prm)
    s = tangential_health(z, prm)
    return a * (s / prm.s0 + prm.phi * prm.mu * big_lambda / (big_lambda + prm.kappa))
end

"""
    imaginary_time_predictions(prm=SCMParams(); anchors=(10.,20.,30.,40.),
                               offsets=(1.,2.,3.), horizon=nothing)
        -> Vector{NamedTuple}

Predictions at `z = t_a ± n·√(-1)` projected on the Weyl 1-D tensor.
Per record: Ṽ (Re/Im/|·|/arg) vs real-axis `V(t_a)`, |Ψ̃| component
magnitudes, `psi_overlap = |⟨|Ψ̃|, Ψ(t_a)⟩|²` (1.0 ⇒ shape-preserving pure
gauge; decay ⇒ genuine spectral drift at rotation depth n), the contraction
residual `|Ṽ − Ã·⟨w, Ψ̃⟩|` (≈ 0), and a numeric Schwarz-reflection check.
"""
function imaginary_time_predictions(prm::SCMParams = SCMParams();
                                    anchors = (10.0, 20.0, 30.0, 40.0),
                                    offsets = (1.0, 2.0, 3.0),
                                    horizon::Union{Nothing,Real} = nothing)
    hz = horizon === nothing ? maximum(anchors) + 1.0 : horizon
    sim = simulate(prm; horizon = hz)
    rate_max = max(maximum(sim.rate), eps())
    d_max = max(maximum(sim.D), eps())
    psi_real_traj = weyl_psi_trajectory(sim, prm)
    w = collect(viability_weights(prm))

    records = NamedTuple[]
    for t_a in anchors
        idx = argmin(abs.(sim.t .- t_a))
        psi_real = psi_real_traj[idx, :]
        for n in offsets
            n > 0 || throw(ArgumentError("offsets must be positive (t_{±n}, n > 0)"))
            z_plus = complex(t_a, n)
            v_plus = wick_viability(z_plus, prm)
            v_minus = wick_viability(complex(t_a, -n), prm)
            schwarz_ok = isapprox(v_minus, conj(v_plus); rtol = 1e-6, atol = 1e-6)
            big_lambda = lambda_contour(t_a, n, prm)
            psi_c = analytic_weyl_psi(z_plus, prm, rate_max, d_max, big_lambda)
            v_contract = adoption(z_plus, prm) * sum(w .* psi_c)
            overlap = wavefunction_overlap(abs.(psi_c), psi_real)
            for (sign, v_z) in (("+", v_plus), ("-", v_minus))
                push!(records, (
                    anchor_t = t_a, offset_n = n, sign = sign,
                    z = complex(t_a, sign == "+" ? n : -n),
                    V_re = round(real(v_z); digits = 4),
                    V_im = round(imag(v_z); digits = 4),
                    V_abs = round(abs(v_z); digits = 4),
                    V_arg = round(angle(v_z); digits = 6),
                    V_real_axis = round(sim.V[idx]; digits = 4),
                    psi_abs = round.(abs.(psi_c); digits = 6),
                    psi_real = round.(psi_real; digits = 6),
                    psi_overlap = overlap,
                    contract_residual = round(abs(v_plus - v_contract); digits = 8),
                    schwarz_ok = schwarz_ok,
                ))
            end
        end
    end
    return records
end

# ---------------------------------------------------------------------------
# Self-check — asserts against the Python reference run
# ---------------------------------------------------------------------------
# MCD emission mode adapter (conceptual; no Quipu writes)
# ---------------------------------------------------------------------------
"""
    emission_mode(d_lambda, prm=SCMParams(); t=nothing) -> Symbol

Pure-function adapter returning the MCD emission mode for a given NHPP
increment `d_lambda` and optional SCM parameter set `prm`.

When `d_lambda < 0.50` (below threshold) returns `:conservative` — the
sub-threshold case that triggers the traditional quiet revival.

Above threshold, the mode is determined by the SCM saturation phase:

| phase | indicator | mode |
|---|---|---|
| pre-saturation (`sat < θ`)     | tangential health S/S₀ still high | `:integrate` |
| near-peak λ (`sat ∈ [θ, 0.9)`) | transition era with rising Λ      | `:emit_tangential` |
| post-peak (`sat ≥ 0.9`)        | atrophied ecosystem, leave intact | `:leave_in_band` |

The optional keyword `t` (clock time in years) allows an exact phase
computation via `bass_fraction`; when omitted, `sat` is estimated from
`d_lambda` via `d_lambda / (d_lambda + 1)` (a monotone proxy that maps
0→0, ∞→1).

This function does NOT write to Quipu or modify any model state — it is
purely diagnostic, exposing the SCM's current emission phase as a Symbol.
"""
function emission_mode(d_lambda::Real, prm::SCMParams = SCMParams();
                       t::Union{Nothing,Real} = nothing)::Symbol
    d_lambda < 0.50 && return :conservative

    # Estimate saturation phase
    sat = if t !== nothing
        Float64(bass_fraction(t, prm))
    else
        Float64(d_lambda) / (Float64(d_lambda) + 1.0)
    end

    if sat < prm.theta
        return :integrate
    elseif sat < 0.9
        return :emit_tangential
    else
        return :leave_in_band
    end
end

# ---------------------------------------------------------------------------
# Reference self-test against Python numerical anchors
# (tech_adoption_resuscitation.py, executed 2026-07-15 in the pipeline venv)
# ---------------------------------------------------------------------------
"""
    selftest() -> Bool

Asserts this port reproduces the Python reference run: 90 % saturation @
12.3 yr, peak λ = 3.887 @ 36.8 yr, S(40) = 0.020, Λ(40) = 81.0,
V(40) = 370.8, and the t=10, n=1 Wick record Ṽ ≈ 581.7 + 14.2i with
overlap ≈ 0.9999, zero contraction residual, Schwarz reflection OK.
"""
function selftest()::Bool
    sim = simulate()
    @assert isapprox(sim.t_sat90, 12.33; atol = 0.1)
    @assert isapprox(sim.t_peak_lambda, 36.8; atol = 0.15)
    @assert isapprox(sim.peak_lambda, 3.887; atol = 0.01)
    @assert isapprox(sim.S_final, 0.020; atol = 0.002)
    @assert isapprox(sim.Lambda_final, 81.0; atol = 0.5)
    @assert isapprox(sim.V_final, 370.8; atol = 0.5)

    recs = imaginary_time_predictions(; anchors = (10.0,), offsets = (1.0,))
    r = first(rec for rec in recs if rec.sign == "+")
    @assert isapprox(r.V_re, 581.7; atol = 1.0)
    @assert isapprox(r.V_im, 14.2; atol = 1.0)
    @assert isapprox(r.psi_overlap, 0.9999; atol = 0.005)
    @assert r.contract_residual < 1e-6
    @assert r.schwarz_ok
    println("SCMPredictor.selftest(): OK — matches tech_adoption_resuscitation.py reference run")
    return true
end

# ---------------------------------------------------------------------------
# Langevin / biased random walk — pure function, no state writes
# ---------------------------------------------------------------------------
"""
    langevin_scm_step(prm, t, mesh7, psi5, d_lambda;
                     direction=fill(0.5,7), plane_scores=Float64[],
                     phase_weight=1.0)
        -> NamedTuple(delta_x, drift, sigma, viability_proxy, mode)

Pure-function Langevin step for the SCM-only predictor.  No Quipu graph or
embedding table is required — the function is purely analytic, using the
SCM state vectors.

    Δx = μ·dt + σ·dW·√dt

where:
- **μ** = α·(Ψ⁷ − m) + β·(mean_specialist − m) + γ·direction,
  with **Ψ⁷** derived from `psi5` via the Weyl 5→7-D lift map.
- **σ** = σ₀·(1+D)·H·(1+0.1·ΔΛ), with *D* from `plane_scores` std-dev
  and *H* from `psi5[2]` (Weyl Ψ1 = topic_entropy proxy).
- **viability_proxy** = |cosθ(Ψ⁷, m+Δx)| — cosine between the Weyl attractor
  and the projected mesh; used as a proxy for wavefunction overlap without
  a live embedding table.
- **mode** is from `emission_mode(d_lambda, prm; t=t)` — the SCM emission
  phase that would select `:integrate|:emit_tangential|:leave_in_band`.

This function does NOT modify `prm` or any model state.
"""
function langevin_scm_step(
    prm::SCMParams, t::Real, mesh7, psi5, d_lambda::Real;
    direction = fill(0.5, 7),
    plane_scores::Vector{Float64} = Float64[],
    phase_weight::Real = 1.0,
)
    # Langevin hyper-parameters (mirror training models)
    LDT    = 0.01;  SIGMA_BASE = 0.05
    ALPHA  = 0.30;  BETA  = 0.20;  GAMMA  = 0.50
    VT     = 0.60;  DS    = 0.30

    # Weyl Ψ-vector lifted to 7-D
    psi5_v = collect(Float64, psi5)
    weyl7 = zeros(7)
    length(psi5_v) >= 1 && (weyl7[1] = clamp(psi5_v[1], 0.0, 1.0))   # Ψ0 → vision
    length(psi5_v) >= 2 && (weyl7[3] = clamp(psi5_v[2], 0.0, 1.0))   # Ψ1 → touch(2)
    length(psi5_v) >= 3 && (weyl7[2] = clamp(psi5_v[3], 0.0, 1.0))   # Ψ2 → touch(1)
    if length(psi5_v) >= 4
        weyl7[5] = clamp(0.6 * psi5_v[4], 0.0, 1.0)                  # Ψ3 → brain
        weyl7[7] = clamp(0.4 * psi5_v[4], 0.0, 1.0)                  # Ψ3 → entirety
    end
    length(psi5_v) >= 5 && (weyl7[6] = clamp(psi5_v[5], 0.0, 1.0))   # Ψ4 → perception

    m  = collect(Float64, mesh7)
    dv = collect(Float64, direction)

    # No emergent specialist biases available in pure SCM mode; mean_force = Weyl7
    mean_force = weyl7

    drift = [
        ALPHA * (weyl7[k] - m[k])
        + BETA  * (mean_force[k] - m[k])
        + GAMMA * dv[k]
        for k in 1:7
    ]

    # Diffusion coefficient
    disagreement = if length(plane_scores) > 1
        ms = sum(plane_scores) / length(plane_scores)
        sqrt(sum((s - ms)^2 for s in plane_scores)) / (abs(ms) + 1e-9)
    else
        0.0
    end
    token_entropy = length(psi5_v) >= 2 ? clamp(psi5_v[2], 0.0, 1.0) : 0.5
    sigma = clamp(
        SIGMA_BASE * (1.0 + disagreement) * token_entropy * (1.0 + 0.1 * d_lambda),
        0.0, 1.0,
    )

    dt = LDT
    delta_x = [drift[k] * dt + sigma * randn() * sqrt(dt) for k in 1:7]

    # Viability proxy: cosine(Ψ7, mesh+Δx)
    proj = [clamp(m[k] + delta_x[k], 0.0, 1.0) for k in 1:7]
    dot_pw = sum(weyl7[k] * proj[k] for k in 1:7)
    norm_w = max(sqrt(sum(weyl7 .^ 2)), 1e-9)
    norm_p = max(sqrt(sum(proj  .^ 2)), 1e-9)
    viability_proxy = clamp(dot_pw / (norm_w * norm_p), 0.0, 1.0)

    apply_scale = viability_proxy >= VT ? 1.0 : DS
    effective_dx = delta_x .* apply_scale

    mode = emission_mode(d_lambda, prm; t = t)
    return (delta_x         = round.(effective_dx; digits = 6),
            drift           = round.(drift;        digits = 6),
            sigma           = round(sigma;         digits = 4),
            viability_proxy = round(viability_proxy; digits = 4),
            mode            = mode)
end

end # module SCMPredictor

# ---------------------------------------------------------------------------
# Demo entry point — `julia scm_predictor.jl`
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    using .SCMPredictor

    SCMPredictor.selftest()
    prm = SCMParams()
    sim = simulate(prm)
    println("\n=== SCM reference run ===")
    println("Time to ~90% saturation: $(round(sim.t_sat90; digits=1)) years")
    println("Peak resuscitation rate at t=$(round(sim.t_peak_lambda; digits=1)) yr, ",
            "lambda=$(round(sim.peak_lambda; digits=3))")
    println("Final S(40) = $(round(sim.S_final; digits=3)) | ",
            "Total expected resuscitations = $(round(sim.Lambda_final; digits=1))")
    println("Final predicted viability V(40) = $(round(sim.V_final; digits=1))")

    ev = simulate_resuscitation_events(prm)
    println("\n=== Ogata thinning === events=$(ev.count) ",
            "(expected Lambda(40)=$(round(ev.expected; digits=1)))")

    gens = multi_generation([
        (label = "gen1", m = 1000.0, p = 0.025, q = 0.38, launch = 0.0),
        (label = "gen2", m = 600.0, p = 0.03, q = 0.42, launch = 12.0),
        (label = "gen3", m = 400.0, p = 0.035, q = 0.45, launch = 24.0),
    ], prm)
    println("=== Norton-Bass 3-generation === V_total(40) = ",
            "$(round(gens.V_total[end]; digits=1))")

    sens = sensitivity_elasticities(prm)
    println("=== Elasticities of V(40) (top 4) === ",
            join(("$(kv.first)=$(round(kv.second; digits=3))" for kv in sens[1:4]), ", "))

    println("=== MRP hook (first 3 periods) ===")
    for row in mrp_module_demand(prm)[1:3]
        println("  ", row)
    end

    println("\n=== Imaginary-time predictions t +/- n*sqrt(-1) vs Weyl 1-D tensor ===")
    println("(safe strip |n| < $(round(max_imag_offset(prm); digits=2)); ",
            "V = A*<w,Psi>, w = $(round.(viability_weights(prm); digits=3)))")
    for rec in imaginary_time_predictions(prm)
        rec.sign == "+" || continue
        println("  t=$(lpad(Int(rec.anchor_t), 3)) n=$(Int(rec.offset_n)) ",
                "V(t)=$(lpad(round(rec.V_real_axis; digits=1), 7)) | ",
                "Vtil=$(round(rec.V_re; digits=1))$(rec.V_im >= 0 ? "+" : "")$(round(rec.V_im; digits=1))i ",
                "|Vtil|=$(round(rec.V_abs; digits=1)) arg=$(round(rec.V_arg; digits=3)) ",
                "overlap=$(round(rec.psi_overlap; digits=4)) ",
                "schwarz=$(rec.schwarz_ok ? "ok" : "FAIL")")
    end
end
