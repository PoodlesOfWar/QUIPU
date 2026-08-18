"""
    MeshSLMSCMGLMGNN

**MESH-SLM-SCM-GLM-GNN** — the holistic Supply Chain Brain model
(`mesh_slm.py`, `MODEL_ARCH = "MESH-SLM-GLM-GNN"`) extended with **SCM**,
the *Saturation-Cycle Model* predictor (technology adoption → tangential
atrophy → modular resuscitation; see the standalone port `scm_predictor.jl`
and the Python source `research/tech_adoption_resuscitation.py`).

Single-file and self-contained (no `include`), following the convention of
the companion `mesh_slm_glm_gnn_model.jl` — shared functions are duplicated
inline so this file stands alone.

Five cooperating aspects:

  * **MESH** — 7-D local + 8th-D global state field (UEQGM aspect
    integrals), here *SCM-extended* to a six-integral blend
    ([`mesh_field_scm`](@ref)).
  * **SLM**  — next-token prediction with temperature sampling
    ([`generate`](@ref)).
  * **SCM**  — Saturation-Cycle Model: the Bass + atrophy + NHPP
    resuscitation predictor driving the ecosystem clock
    ([`scm_state`](@ref)).
  * **GLM**  — graph-anchored 7-D token embeddings ([`embed_dot`](@ref)).
  * **GNN**  — Hebbian Quipu message-passing edges ([`train_round!`](@ref)).

## The five SCM coupling points

1. **End-State slot** — the Bass saturation `sat(τ)` at the model's
   ecosystem clock replaces `mesh_slm._end_state_progress` in the effective
   learning rates: `η = LR_BASE·(1−sat)·phase_weight·(0.70+0.30·overlap)`,
   `η_q = QUIPU_LR·(1−sat)·phase_weight` — learning cools exactly as the
   adoption S-curve saturates.
2. **Tangential atrophy of graph knowledge** — past the SCM threshold
   (`sat > θ`), every Quipu weight decays multiplicatively by
   `1 − ATROPHY_COUPLING·((sat−θ)/(1−θ))·(1−S/S0)` (clamped ≥ 0.5),
   mirroring `dS/dt = −γ·sat·S` at the edge level.
3. **Modular resuscitation of atrophied edges** — the expected NHPP event
   mass `ΔΛ` accrued over the round revives the
   `round(ΔΛ·REVIVE_EDGES_PER_EVENT)` weakest edges via the same Hebbian
   form `w ← w + RESUSC_LR·(1−w)` (the modularity μ acts inside λ through
   β2, so more modular ecosystems revive more knowledge).
4. **SCM-extended MESH field** — the 8th-D blend gains a sixth integral
   `R = clip01(V/(m·(1+φμ)))` (viability normalised by its theoretical
   ceiling), re-normalised weights
   `0.28·H + 0.22·F + 0.22·O + 0.09·P + 0.09·W + 0.10·R`
   (extension of the source blend 0.30/0.25/0.25/0.10/0.10).
5. **Shared Weyl 1-D tensor** — [`weyl_psi`](@ref) maps the SCM state onto
   the same Ψ0…Ψ4 scalars persisted by `ueqgm_engine.weyl_scalar_tensor`,
   with the contraction identity `V = A·⟨(0,0,0,1,φμ), Ψ⟩`.

The ecosystem clock advances by `scm_dt` years per training round, so the
model's own life-cycle (grow → saturate → atrophy → resuscitate) unfolds
across training epochs.
"""
module MeshSLMSCMGLMGNN

using Random

export
    MODEL_ARCH,
    SCMParams,
    MeshSLMModel,
    tokenize,
    upsert_token!,
    torus_dist,
    proximity,
    embed_dot,
    wavefunction_overlap,
    holographic_entropy,
    metric_perturbation,
    weyl_scalar_tensor,
    mean_embed7,
    mesh_field_scm,
    apply_specialist_bias,
    acre_observe!,
    acre_principal_axis,
    acre_emerge!,
    scm_state,
    weyl_psi,
    mcd_lift_weyl_psi,
    mcd_multiplanar_score,
    mcd_tangential_direction,
    mcd_emit_along_direction!,
    mcd_revive_weakest!,
    mcd_dispatch!,
    langevin_compute_drift,
    langevin_compute_sigma,
    langevin_token_entropy,
    langevin_step!,
    train_round!,
    score_candidates,
    softmax_pick,
    generate,
    state_summary

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
const MODEL_ARCH = "MESH-SLM-SCM-GLM-GNN"

# ---------------------------------------------------------------------------
# Shared constants (mesh_slm.py / ueqgm_engine.py values)
# ---------------------------------------------------------------------------
const EMBED_DIM         = 7
const LR_BASE           = 0.04
const QUIPU_LR          = 0.06
const QUIPU_GAIN        = 0.55
const PROX_GAIN         = 0.25
const MESH_FIELD_GAIN   = 0.18
const WARP_GAIN         = 0.06
const METRIC_MASS_SCALE = 6.7e26
const CONF_FLOOR        = 0.22
const MIN_VOCAB_FOR_GENERATE = 16
const G_CONST = 6.674e-11
const C_CONST = 2.998e8
const GAMMA_0_DEFAULT = 1.0
const TAN_CLAMP = 1.0e3

# SCM coupling constants (this integration's tunables)
const ATROPHY_COUPLING       = 0.05   # max per-round quipu decay strength
const RESUSC_LR              = 0.10   # Hebbian revival nudge per event
const REVIVE_EDGES_PER_EVENT = 8      # weakest edges revived per NHPP event
# H/F/O/P/W carry the Planck 2018 cosmological census proportions
# (Ω_Λ, Ω_c, Ω_b, Ω_ν, Ω_γ — Aghanim et al. 2020, A&A 641, A6), scaled
# into the (1 − R) envelope.  R = 0.10 is this integration's ONE declared
# modeling choice: the SCM viability coupling, which has no cosmological
# counterpart and is documented as a tunable, not a measurement.
const _CENSUS_TOTAL = 0.6847 + 0.2645 + 0.0493 + 0.0014 + 5.4e-5
const FIELD_W = (
    H = 0.90 * 0.6847 / _CENSUS_TOTAL,   # dark energy
    F = 0.90 * 0.2645 / _CENSUS_TOTAL,   # cold dark matter
    O = 0.90 * 0.0493 / _CENSUS_TOTAL,   # baryons
    P = 0.90 * 0.0014 / _CENSUS_TOTAL,   # neutrinos
    W = 0.90 * 5.4e-5 / _CENSUS_TOTAL,   # photons
    R = 0.10,                            # SCM viability (declared tunable)
)

# ACRE — Axial Cross-Resonance Emergence (ports of mesh_slm.acre_*)
const AXES = ("vision", "touch", "smell", "body", "brain", "perception", "entirety")
const ACRE_EMA_RHO          = 0.15
const ACRE_MIN_RESONANCE    = 0.35
const ACRE_NOVELTY_CEILING  = 0.60
const ACRE_BIAS_SCALE       = 0.25
const ACRE_MIN_COMPONENT    = 0.02
const ACRE_MAX_EMERGENT     = 4

# MCD — Multiplanar Comparative Decompression emission dispatcher
const MCD_EMISSION_THRESHOLD    = 0.50   # ΔΛ threshold to activate mode selection
const MCD_INTEGRATE_RATIO       = 1.50f0 # edge-revival multiplier for :integrate
const MCD_RESONANCE_INTEGRATE   = 1.15   # resonance_ratio → :integrate
const MCD_RESONANCE_TANGENTIAL  = 0.90   # resonance_ratio → :emit_tangential
const MCD_TANGENTIAL_LR         = 0.12   # Hebbian nudge along tangential direction
const MCD_ACRE_LOW_RESONANCE    = 0.25   # relaxed ACRE gate for :emit_tangential

# Langevin dynamics — stochastic MESH manifold evolution
# Δx = drift·dt + σ·dW·√dt  (biased random walk over the quipu manifold)
const LANGEVIN_DT                  = 0.01    # integration step size
const LANGEVIN_SIGMA_BASE          = 0.05    # base noise amplitude
const LANGEVIN_DRIFT_ALPHA         = 0.30    # Weyl-remnant pull coefficient
const LANGEVIN_DRIFT_BETA          = 0.20    # specialist-force coefficient
const LANGEVIN_DRIFT_GAMMA         = 0.50    # emission-direction coefficient
const LANGEVIN_VIABILITY_THRESHOLD = 0.60    # reinforce vs diffuse gate
const LANGEVIN_DIFFUSE_SCALE       = 0.30    # dampened step when viability is low
const LANGEVIN_MAX_NUDGE           = 0.05    # per-token embedding nudge cap

const SPECIALIST_BIASES = Dict{String,NTuple{7,Float64}}(
    "supply_chain_optimizer" => (0.0, 0.0, 0.0, 0.15, 0.25, 0.20, 0.10),
    "research_specialist"    => (0.10, 0.05, 0.05, 0.05, 0.20, 0.15, 0.15),
    "mesh_historian"         => (0.05, 0.05, 0.0, 0.0, 0.0, 0.10, 0.30),
)
const NUMERIC_BOOST_SUBSTRINGS = ("eoq", "safety", "demand", "stock", "lead", "cost", "hold")
const NUMERIC_BOOST_TOKENS = Set([
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",",
    "eoq", "sqrt", "demand", "stock", "safety", "lead", "hold", "cost",
])
const OPT_BOOST = 0.35
const TOKEN_RE = r"[A-Za-z][A-Za-z0-9_\-]{1,30}|\d+(?:\.\d+)?|[+\-*/=<>^]|\b(?:sqrt|eoq|safety|stock|demand|lead|cost|hold|order)\b"

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
# SCM — Saturation-Cycle Model (compact port; full docs in scm_predictor.jl)
# ---------------------------------------------------------------------------
"""SCM parameters — defaults are the predictor's reference run."""
Base.@kwdef struct SCMParams
    m::Float64 = 1000.0
    p::Float64 = 0.025
    q::Float64 = 0.38
    gamma::Float64 = 0.12
    mu::Float64 = 0.78
    s0::Float64 = 1.0
    beta0::Float64 = -2.8
    beta1::Float64 = 2.1
    beta2::Float64 = 3.2
    beta3::Float64 = 1.4
    phi::Float64 = 0.55
    kappa::Float64 = 18.0
    theta::Float64 = 0.72
end

scm_density(t) = 0.35 + 0.55 * (1 - exp(-0.07 * t)) * (1 + 0.25 * sin(2pi * t / 7))

function scm_fraction(t::Real, prm::SCMParams)
    b, c = prm.p + prm.q, prm.q / prm.p
    e = exp(-b * t)
    return (1 - e) / (1 + c * e)
end

function scm_cumfraction(t::Real, prm::SCMParams)
    b, c = prm.p + prm.q, prm.q / prm.p
    return t + ((1 + c) / (b * c)) * (log(1 + c * exp(-b * t)) - log(1 + c))
end

scm_health(t::Real, prm::SCMParams) = prm.s0 * exp(-prm.gamma * scm_cumfraction(t, prm))

function scm_intensity(t::Real, prm::SCMParams)
    sat = scm_fraction(t, prm)
    s = scm_health(t, prm)
    return exp(prm.beta0 + prm.beta1 * sat
               + prm.beta2 * prm.mu * max(0.0, sat - prm.theta) * (prm.s0 - s) / prm.s0
               + prm.beta3 * scm_density(t))
end

"""Λ(t) = ∫0^t λ ds by trapezoid on an n-point grid."""
function scm_cum_intensity(t::Real, prm::SCMParams; n::Int = 401)
    t <= 0 && return 0.0
    grid = range(0.0, t; length = n)
    vals = [scm_intensity(g, prm) for g in grid]
    acc = 0.0
    for i in 2:n
        acc += 0.5 * (vals[i] + vals[i - 1]) * step(grid)
    end
    return acc
end

# ---------------------------------------------------------------------------
# The model — SLM-GLM-GNN state + SCM ecosystem clock
# ---------------------------------------------------------------------------
"""
    MeshSLMModel(; torus_n=64, embed_dim=7, scm=SCMParams(), scm_dt=0.5)

Mutable in-memory MESH-SLM-SCM-GLM-GNN. Graph fields mirror the
`mesh_slm_vocab/_embed/_quipu/_meta` SQLite tables (see
`mesh_slm_glm_gnn_model.jl` for the table map); the SCM block adds the
ecosystem clock (`scm_clock`, advanced by `scm_dt` years per
[`train_round!`](@ref)) and life-cycle diagnostics.
"""
mutable struct MeshSLMModel
    torus_n::Int
    vocab_limit::Int
    embed_dim::Int
    next_id::Int
    vocab_id::Dict{String,Int}
    tokens::Dict{Int,String}
    positions::Dict{Int,Tuple{Int,Int}}
    occupied::Dict{Tuple{Int,Int},Int}
    freq::Dict{Int,Int}
    last_seen::Dict{Int,Float64}
    embeddings::Dict{Int,Vector{Float64}}
    quipu_weight::Dict{Tuple{Int,Int},Float64}
    quipu_samples::Dict{Tuple{Int,Int},Int}
    rounds::Int
    last_loss::Float64
    last_eta::Float64
    last_mesh_field::Float64
    # SCM block
    scm::SCMParams
    scm_clock::Float64
    scm_dt::Float64
    scm_events_total::Float64
    last_sat::Float64
    last_eco_health::Float64
    last_atrophy_factor::Float64
    last_revived::Int
    last_viability::Float64
    acre_interaction::Matrix{Float64}                 # ACRE 7×7 multi-axial EMA
    emergent_biases::Dict{String,NTuple{7,Float64}}   # ACRE emergent specialists
end

function MeshSLMModel(; torus_n::Int = 64, embed_dim::Int = 7,
                      scm::SCMParams = SCMParams(), scm_dt::Float64 = 0.5)
    return MeshSLMModel(
        torus_n, torus_n * torus_n, embed_dim, 1,
        Dict{String,Int}(), Dict{Int,String}(), Dict{Int,Tuple{Int,Int}}(),
        Dict{Tuple{Int,Int},Int}(), Dict{Int,Int}(), Dict{Int,Float64}(),
        Dict{Int,Vector{Float64}}(),
        Dict{Tuple{Int,Int},Float64}(), Dict{Tuple{Int,Int},Int}(),
        0, 0.0, 0.0, 0.0,
        scm, 0.0, scm_dt, 0.0, 0.0, 1.0, 1.0, 0, 0.0,
        zeros(7, 7), Dict{String,NTuple{7,Float64}}(),
    )
end

"""
    scm_state(model) -> NamedTuple

Evaluate the SCM predictor at the model's ecosystem clock:
`(tau, A, sat, S, lam, Lambda, benefit, V, R)` where `R` is the
viability-ceiling normalisation `clip01(V/(m·(1+φμ)))` used as the sixth
MESH-field integral.
"""
function scm_state(model::MeshSLMModel)
    prm, tau = model.scm, model.scm_clock
    sat = scm_fraction(tau, prm)
    a = prm.m * sat
    s = scm_health(tau, prm)
    lam = scm_intensity(tau, prm)
    big_l = scm_cum_intensity(tau, prm)
    benefit = prm.phi * prm.mu * big_l / (big_l + prm.kappa)
    v = a * (s / prm.s0 + benefit)
    r = clip01(v / (prm.m * (1.0 + prm.phi * prm.mu)))
    return (tau = tau, A = a, sat = sat, S = s, lam = lam, Lambda = big_l,
            benefit = benefit, V = v, R = r)
end

# ---------------------------------------------------------------------------
# Tokenizer, torus allocation, geometry, GLM term (as in the -GLM-GNN port)
# ---------------------------------------------------------------------------
"""Split text into lowercase tokens (max 256). Port of `mesh_slm._tokenize`."""
function tokenize(text::AbstractString)::Vector{String}
    isempty(text) && return String[]
    matches = [lowercase(m.match) for m in eachmatch(TOKEN_RE, text)]
    return matches[1:min(256, length(matches))]
end

function evict!(model::MeshSLMModel, token_id::Int)
    filter!(p -> p.first[1] != token_id && p.first[2] != token_id, model.quipu_weight)
    filter!(p -> p.first[1] != token_id && p.first[2] != token_id, model.quipu_samples)
    delete!(model.embeddings, token_id)
    pos = get(model.positions, token_id, nothing)
    pos === nothing || delete!(model.occupied, pos)
    delete!(model.positions, token_id)
    tok = get(model.tokens, token_id, nothing)
    tok === nothing || delete!(model.vocab_id, tok)
    delete!(model.tokens, token_id)
    delete!(model.freq, token_id)
    delete!(model.last_seen, token_id)
    return nothing
end

function allocate_coord!(model::MeshSLMModel, token::AbstractString)::Tuple{Int,Int}
    base = Int(hash(token) % UInt(model.vocab_limit))
    for stepn in 0:(model.vocab_limit - 1)
        idx = (base + stepn) % model.vocab_limit
        i, j = divrem(idx, model.torus_n)
        haskey(model.occupied, (i, j)) || return (i, j)
    end
    victim, vfreq, vseen = nothing, typemax(Int), Inf
    for tid in keys(model.positions)
        f = get(model.freq, tid, 0)
        seen = get(model.last_seen, tid, 0.0)
        if f < vfreq || (f == vfreq && seen < vseen)
            victim, vfreq, vseen = tid, f, seen
        end
    end
    victim === nothing && return (0, 0)
    vpos = model.positions[victim]
    evict!(model, victim)
    return vpos
end

"""Insert/refresh a token; seeds a MESH-aligned 7-D embedding on first sight."""
function upsert_token!(model::MeshSLMModel, token::AbstractString, mesh_state7)::Int
    now = time()
    if haskey(model.vocab_id, token)
        tid = model.vocab_id[token]
        model.freq[tid] = get(model.freq, tid, 0) + 1
        model.last_seen[tid] = now
        return tid
    end
    i, j = allocate_coord!(model, token)
    tid = model.next_id
    model.next_id += 1
    model.vocab_id[token] = tid
    model.tokens[tid] = String(token)
    model.positions[tid] = (i, j)
    model.occupied[(i, j)] = tid
    model.freq[tid] = 1
    model.last_seen[tid] = now
    model.embeddings[tid] = [0.05 * Float64(v) + (rand() - 0.5) * 0.02 for v in mesh_state7]
    return tid
end

function torus_dist(a::Tuple{Int,Int}, b::Tuple{Int,Int}, torus_n::Int)::Int
    di, dj = abs(a[1] - b[1]), abs(a[2] - b[2])
    return min(di, torus_n - di) + min(dj, torus_n - dj)
end

proximity(a::Tuple{Int,Int}, b::Tuple{Int,Int}, torus_n::Int) =
    max(0.0, 1.0 - torus_dist(a, b, torus_n) / Float64(torus_n))

"""GLM alignment term — dot of a token's 7-D embedding with the MESH state."""
embed_dot(embed7, mesh7) = sum(a * b for (a, b) in zip(embed7, mesh7))

# ---------------------------------------------------------------------------
# UEQGM helpers (ports; full docs in mesh_slm_glm_gnn_model.jl)
# ---------------------------------------------------------------------------
coherence_to_phi(coherence::Integer) = pi / 4 + coherence * pi

function _raw_sici(phi::Real)
    x = phi != 0.0 ? abs(phi) : 1.0e-12
    si = x - x^3 / 18.0 + x^5 / 600.0 - x^7 / 35280.0
    ci = 0.5772156649 + log(x) - x^2 / 4.0 + x^4 / 96.0
    return (phi >= 0 ? si : -si), ci
end

function sici_axial_decay(phi::Real, gamma_0::Real = GAMMA_0_DEFAULT)
    si, ci = _raw_sici(phi)
    return si * ci * clamp(tan(phi), -TAN_CLAMP, TAN_CLAMP) * gamma_0
end

function wavefunction_overlap(a, b)::Float64
    (isempty(a) || length(a) != length(b)) && return 0.0
    dot = sum(ai * bi for (ai, bi) in zip(a, b))
    na = sqrt(sum(ai * ai for ai in a))
    nb = sqrt(sum(bi * bi for bi in b))
    (na == 0.0 || nb == 0.0) && return 0.0
    return round((dot / (na * nb))^2; digits = 6)
end

function besselj0_as(x::Real)::Float64
    ax = abs(Float64(x))
    if ax <= 3.0
        y = (ax / 3.0)^2
        return 1.0 + y * (-2.2499997 + y * (1.2656208 + y * (-0.3163866 +
            y * (0.0444479 + y * (-0.0039444 + y * 0.0002100)))))
    end
    u = 3.0 / ax
    f0 = 0.79788456 + u * (-0.00000077 + u * (-0.00552740 + u * (-0.00009512 +
        u * (0.00137237 + u * (-0.00072805 + u * 0.00014476)))))
    theta0 = ax - 0.78539816 + u * (-0.04166397 + u * (-0.00003954 +
        u * (0.00262573 + u * (-0.00054125 + u * (-0.00029333 +
        u * 0.00013558)))))
    return f0 * cos(theta0) / sqrt(ax)
end

# Floquet J₀ renormalization (dynamic localization, Dunlap-Kenkre 1986).
floquet_modulation_factor(drive::Real, omega::Real) =
    omega == 0.0 ? 1.0 : besselj0_as(drive / omega)

# Von Neumann graph entropy, HEHW (2012) quadratic approximation.
function holographic_entropy(
    n_edges::Real, n_nodes::Real, degree_pair_sum::Union{Nothing,Real} = nothing,
)::Float64
    (n_nodes <= 0 || n_edges <= 0) && return 0.0
    n = float(n_nodes)
    dps = degree_pair_sum === nothing ? (n * n) / (4.0 * float(n_edges)) : float(degree_pair_sum)
    return clip01(1.0 - (1.0 / n) - (dps / (n * n)))
end

function metric_perturbation(mass_eff::Real, r::Real)::Float64
    r <= 0.0 && return 0.0
    return 2.0 * G_CONST * mass_eff / (C_CONST^2 * r)
end

function phase_evolution_total(phi::Real; gamma_eff::Real = GAMMA_0_DEFAULT)
    return sici_axial_decay(phi) * (2.0 * pi / max(gamma_eff, 1.0e-30))
end

"""Weyl scalars Ψ0…Ψ4 — port of `ueqgm_engine.weyl_scalar_tensor`."""
function weyl_scalar_tensor(signal_flux, topic_entropy, corpus_volume,
                            mesh_alignment, remnant_score)
    rho = max(0.0, float(corpus_volume))
    return (clip01(signal_flux), clip01(topic_entropy),
            clip01(rho / (rho + 1.0)), clip01(mesh_alignment),
            clip01(remnant_score))
end

"""
    weyl_psi(model) -> NTuple{5,Float64}

**Weyl bridge**: map the current SCM state onto the shared Ψ 1-D tensor
(Ψ0 = adoption flux vs Bass peak `m(p+q)²/(4q)·(1+q/p... )` proxy, Ψ1 =
innovation breadth, Ψ2 = Λ bulk mass, Ψ3 = S/S0, Ψ4 = Λ/(Λ+κ)); note the
contraction identity `V = A·(Ψ3 + φμ·Ψ4)`.
"""
function weyl_psi(model::MeshSLMModel)
    st = scm_state(model)
    prm = model.scm
    rate = prm.p * (prm.m - st.A) + (prm.q / prm.m) * st.A * (prm.m - st.A)
    rate_peak = prm.m * (prm.p + prm.q)^2 / (4.0 * prm.q)   # Bass peak rate
    return weyl_scalar_tensor(rate / rate_peak,
                              scm_density(st.tau) / 0.90,
                              st.Lambda, st.S / prm.s0,
                              st.Lambda / (st.Lambda + prm.kappa))
end

"""Mean 7-D embedding across the most-frequent tokens (neutral 0.5 fallback)."""
function mean_embed7(model::MeshSLMModel; limit::Int = 128)::NTuple{7,Float64}
    isempty(model.embeddings) && return ntuple(_ -> 0.5, 7)
    ranked = sort(collect(keys(model.embeddings)); by = tid -> get(model.freq, tid, 0), rev = true)
    top_n = min(limit, length(ranked))
    acc = zeros(Float64, model.embed_dim)
    for k in 1:top_n
        acc .+= model.embeddings[ranked[k]]
    end
    acc ./= top_n
    return ntuple(k -> clip01(acc[k]), 7)
end

# ---------------------------------------------------------------------------
# SCM-extended MESH field (coupling point 4)
# ---------------------------------------------------------------------------
"""
    mesh_field_scm(model, mesh_state7; phase_weight=1.0, coherence_depth=0,
                   weyl_phase=0.0) -> Float64

The SCM-extended global field:
`FIELD_W.H·H + FIELD_W.F·F + FIELD_W.O·O + FIELD_W.P·P + FIELD_W.W·W + 0.10·R`
where H/F/O/P/W carry the Planck 2018 census proportions in the 0.90
envelope, and `R = clip01(V/(m(1+φμ)))` is the SCM viability integral
(the one declared tunable).
"""
function mesh_field_scm(model::MeshSLMModel, mesh_state7;
                        phase_weight::Real = 1.0, coherence_depth::Integer = 0,
                        weyl_phase::Real = 0.0)::Float64
    n_vocab = length(model.positions)
    n_edges = length(model.quipu_weight)
    H = holographic_entropy(n_edges, n_vocab)
    phi = coherence_to_phi(coherence_depth)
    omega = max(0.1, phase_weight)
    F = clip01(0.5 * (1.0 + floquet_modulation_factor(mod(weyl_phase, 2pi), omega)))
    O = wavefunction_overlap(mean_embed7(model), mesh_state7)
    P = clip01(abs(phase_evolution_total(phi)) / (2.0 * pi))
    vocab_fill = n_vocab / max(model.vocab_limit, 1)
    r = max(1.0 - 0.5 * vocab_fill, 0.01)
    W = clip01(metric_perturbation(vocab_fill * METRIC_MASS_SCALE, r) * r)
    R = scm_state(model).R
    return clip01(FIELD_W.H * H + FIELD_W.F * F + FIELD_W.O * O +
                  FIELD_W.P * P + FIELD_W.W * W + FIELD_W.R * R)
end

function apply_specialist_bias(mesh7, specialist::Union{Nothing,AbstractString},
                                emergent::Dict{String,NTuple{7,Float64}} =
                                    Dict{String,NTuple{7,Float64}}())
    specialist === nothing && return mesh7
    bias = haskey(SPECIALIST_BIASES, specialist) ? SPECIALIST_BIASES[specialist] :
           haskey(emergent, specialist)          ? emergent[specialist] : nothing
    bias === nothing && return mesh7
    return ntuple(k -> clip01(mesh7[k] + bias[k]), 7)
end

# ---------------------------------------------------------------------------
# ACRE — Axial Cross-Resonance Emergence (ports of mesh_slm.acre_*; full
# gate-by-gate documentation lives in mesh_slm_glm_gnn_model.jl)
# ---------------------------------------------------------------------------
"""Hebbian EMA of the symmetrized MESH × mean-embed outer product."""
function acre_observe!(model::MeshSLMModel, mesh_state7, mean_embed;
                        rho::Real = ACRE_EMA_RHO)
    m = [clip01(mesh_state7[k]) for k in 1:7]
    e = [clip01(mean_embed[k]) for k in 1:7]
    for i in 1:7, j in 1:7
        inter = 0.5 * (m[i] * e[j] + e[i] * m[j])
        model.acre_interaction[i, j] =
            (1.0 - rho) * model.acre_interaction[i, j] + rho * inter
    end
    return model.acre_interaction
end

"""Dominant eigenpair by power iteration; non-negative Perron axis."""
function acre_principal_axis(C::AbstractMatrix)
    n = size(C, 1)
    v = fill(1.0 / sqrt(n), n)
    tr = sum(C[i, i] for i in 1:n)
    for _ in 1:32
        w = C * v
        nrm = sqrt(sum(abs2, w))
        nrm <= 1e-12 && return zeros(n), 0.0, tr
        v = w ./ nrm
    end
    eigval = sum(v .* (C * v))
    k = argmax(abs.(v))
    v[k] < 0 && (v = -v)
    return max.(v, 0.0), eigval, tr
end

"""Signal → capacity → resonance → novelty gates; registers `emergent_*`."""
function acre_emerge!(model::MeshSLMModel;
                       min_resonance::Real = ACRE_MIN_RESONANCE,
                       novelty_ceiling::Real = ACRE_NOVELTY_CEILING,
                       bias_scale::Real = ACRE_BIAS_SCALE,
                       max_emergent::Int = ACRE_MAX_EMERGENT)
    axis, eigval, tr = acre_principal_axis(model.acre_interaction)
    (tr <= 1e-6 || eigval <= 1e-9) && return (status = "no_signal",)
    length(model.emergent_biases) >= max_emergent && return (status = "at_capacity",)
    resonance = eigval / tr
    resonance < min_resonance &&
        return (status = "no_resonance", resonance = round(resonance; digits = 4))

    max_overlap = wavefunction_overlap(axis, ones(7))
    nearest = "__uniform__"
    for (name, bias) in merge(SPECIALIST_BIASES, model.emergent_biases)
        ov = wavefunction_overlap(axis, collect(bias))
        ov > max_overlap && ((max_overlap, nearest) = (ov, name))
    end
    max_overlap >= novelty_ceiling &&
        return (status = "not_novel", nearest = nearest,
                overlap = round(max_overlap; digits = 4),
                resonance = round(resonance; digits = 4))

    peak = max(maximum(axis), eps())
    bias_vec = [round(bias_scale * a / peak; digits = 4) for a in axis]
    bias_vec = [b >= ACRE_MIN_COMPONENT ? b : 0.0 for b in bias_vec]
    order = sortperm(bias_vec; rev = true)
    name = "emergent_$(AXES[order[1]])_$(AXES[order[2]])"
    (haskey(model.emergent_biases, name) || haskey(SPECIALIST_BIASES, name)) &&
        (name = "$(name)_$(length(model.emergent_biases) + 1)")
    model.emergent_biases[name] = ntuple(k -> bias_vec[k], 7)
    return (status = "created", specialist = name, bias = bias_vec,
            resonance = round(resonance; digits = 4),
            max_overlap = round(max_overlap; digits = 4),
            emergent_count = length(model.emergent_biases))
end

# ---------------------------------------------------------------------------
# MCD — Multiplanar Comparative Decompression helpers
# ---------------------------------------------------------------------------
"""
    mcd_lift_weyl_psi(psi5) -> Vector{Float64}

Project the 5-scalar Weyl Ψ vector into 7-D mesh space.
Ψ0→vision, Ψ1→smell, Ψ2→touch, Ψ3→brain(0.6)+entirety(0.4), Ψ4→perception.
"""
function mcd_lift_weyl_psi(psi5)::Vector{Float64}
    out = zeros(7)
    if length(psi5) >= 1; out[1] = clip01(psi5[1]) end                # Ψ0 → vision
    if length(psi5) >= 2; out[2] = clip01(psi5[2]) end                # Ψ1 → touch
    if length(psi5) >= 3; out[3] = clip01(psi5[3]) end                # Ψ2 → smell
    if length(psi5) >= 4
        out[5] = clip01(0.6 * psi5[4])                                # Ψ3 → brain
        out[7] = clip01(0.4 * psi5[4])                                # Ψ3 → entirety
    end
    if length(psi5) >= 5; out[6] = clip01(psi5[5]) end                # Ψ4 → perception
    return out
end

"""
    mcd_multiplanar_score(model, last_id, mesh7, psi5; mesh_field=0.0)
        -> (base_score, resonance_ratio)

Run `score_candidates` across base mesh7, each emergent-bias plane,
reflected Ψ (1−lift), and inverted Ψ (lift). Returns the base plane score
and `max_score / (base_score + ε)`.
"""
function mcd_multiplanar_score(
    model::MeshSLMModel, last_id::Int, mesh7, psi5;
    mesh_field::Real = 0.0,
)
    planes = [collect(Float64, mesh7)]
    for bias in values(model.emergent_biases)
        push!(planes, [clip01(mesh7[k] + bias[k]) for k in 1:7])
    end
    lifted = mcd_lift_weyl_psi(psi5)
    push!(planes, [clip01(1.0 - v) for v in lifted])   # reflected
    push!(planes, [clip01(v)       for v in lifted])   # inverted (straight lift)

    scores = Float64[]
    for plane in planes
        cands = score_candidates(model, last_id, plane; top_k = 4, mesh_field = mesh_field)
        isempty(cands) || push!(scores, cands[1].score)
    end
    isempty(scores) && return (0.0, 1.0, Float64[])
    base = scores[1]
    best = maximum(scores)
    return (base, best / (base + 1e-9), scores)
end

"""
    mcd_tangential_direction(mesh7, psi5, phase_weight) -> Vector{Float64}

Compute the 7-D tangential emission direction via the Lagrangian
photon/neutrino phase interaction: `d[k] = 0.5·cos(ω·m[k]+lift[k]) + 0.5·sin(ω·lift[k]−m[k])`,
normalised to max-magnitude 1.
"""
function mcd_tangential_direction(mesh7, psi5, phase_weight::Real)::Vector{Float64}
    lifted = mcd_lift_weyl_psi(psi5)
    omega = max(0.1, Float64(phase_weight)) * π
    m = collect(Float64, mesh7)
    dir = [clip01(0.5 * cos(omega * m[k] + lifted[k]) +
                  0.5 * sin(omega * lifted[k] - m[k])) for k in 1:7]
    peak = max(maximum(abs.(dir)), 1e-9)
    return dir ./ peak
end

"""
    mcd_revive_weakest!(model, d_lambda; multiplier=1.0) -> Int

Revive `round(ΔΛ·REVIVE_EDGES_PER_EVENT·multiplier)` weakest quipu edges.
"""
function mcd_revive_weakest!(model::MeshSLMModel, d_lambda::Real;
                              multiplier::Real = 1.0)::Int
    n = min(round(Int, d_lambda * REVIVE_EDGES_PER_EVENT * multiplier),
            length(model.quipu_weight))
    n == 0 && return 0
    weakest = sort(collect(model.quipu_weight); by = kv -> kv.second)[1:n]
    for (key, w) in weakest
        model.quipu_weight[key] = w + RESUSC_LR * (1.0 - w)
    end
    return n
end

"""
    mcd_emit_along_direction!(model, last_id, direction, d_lambda; mesh_field=0.0)
        -> Int

Boost/create Quipu edges aligned with `direction`; returns count modified.
"""
function mcd_emit_along_direction!(
    model::MeshSLMModel, last_id::Int, direction::Vector{Float64}, d_lambda::Real;
    mesh_field::Real = 0.0,
)::Int
    cands = score_candidates(model, last_id, direction; top_k = 16, mesh_field = mesh_field)
    n_emit = max(1, round(Int, d_lambda * REVIVE_EDGES_PER_EVENT))
    modified = 0
    for c in cands[1:min(n_emit, length(cands))]
        key = (last_id, c.token_id)
        if haskey(model.quipu_weight, key)
            w = model.quipu_weight[key]
            model.quipu_weight[key] = min(1.0, w + MCD_TANGENTIAL_LR * (1.0 - w))
            model.quipu_samples[key] = get(model.quipu_samples, key, 0) + 1
        else
            model.quipu_weight[key] = MCD_TANGENTIAL_LR
            model.quipu_samples[key] = 1
        end
        modified += 1
    end
    return modified
end

"""
    mcd_dispatch!(model, last_id, mesh7, d_lambda, psi5; phase_weight=1.0,
                  mesh_field=0.0) -> NamedTuple

Multiplanar Comparative Decompression emission dispatcher.

When `d_lambda >= MCD_EMISSION_THRESHOLD`:
- `:integrate`       — resonance_ratio ≥ MCD_RESONANCE_INTEGRATE → 1.5× revival.
- `:emit_tangential` — resonance_ratio ≥ MCD_RESONANCE_TANGENTIAL → tangential edge boost +
                       relaxed ACRE emergence.
- `:leave_in_band`   — below both thresholds → no quipu writes.

Below threshold: conservative 1× revival (original resuscitation pass).
"""
# ---------------------------------------------------------------------------
# Langevin dynamics helpers
# ---------------------------------------------------------------------------
"""
    langevin_token_entropy(model; limit=128) -> Float64

Normalised Shannon entropy of the top-`limit` token frequencies in the model.
Used as the `new_token_entropy` (corpus freshness) term in `σ`.
"""
function langevin_token_entropy(model::MeshSLMModel; limit::Int = 128)::Float64
    isempty(model.freq) && return 0.5
    top_freqs = sort(collect(values(model.freq)); rev = true)[1:min(limit, length(model.freq))]
    total = max(sum(top_freqs), 1)
    entropy = -sum((f / total) * log(f / total + 1e-12) for f in top_freqs)
    max_entropy = log(max(length(top_freqs), 1) + 1)
    return clip01(entropy / max(max_entropy, 1.0))
end

"""
    langevin_compute_drift(mesh7, weyl7, specialist_forces, direction) -> Vector{Float64}

Structured drift: μ_k = α·(Ψ_k⁽⁷⁾ − m_k) + β·(f̄_k − m_k) + γ·d_k
"""
function langevin_compute_drift(mesh7, weyl7, specialist_forces, direction)::Vector{Float64}
    m = collect(Float64, mesh7)
    w = collect(Float64, weyl7)
    d = collect(Float64, direction)
    mean_force = if isempty(specialist_forces)
        zeros(7)
    else
        [sum(sf[k] for sf in specialist_forces) / length(specialist_forces) for k in 1:7]
    end
    return [
        LANGEVIN_DRIFT_ALPHA * (clip01(w[k]) - m[k])
        + LANGEVIN_DRIFT_BETA  * (mean_force[k] - m[k])
        + LANGEVIN_DRIFT_GAMMA * d[k]
        for k in 1:7
    ]
end

"""
    langevin_compute_sigma(plane_scores, d_lambda, token_entropy) -> Float64

Diffusion coefficient: σ = σ₀·(1+D)·H·(1+0.1·ΔΛ) where D = normalised score std-dev.
"""
function langevin_compute_sigma(
    plane_scores::Vector{Float64}, d_lambda::Real, token_entropy::Real
)::Float64
    disagreement = if length(plane_scores) > 1
        mean_s = sum(plane_scores) / length(plane_scores)
        sqrt(sum((s - mean_s)^2 for s in plane_scores)) / (abs(mean_s) + 1e-9)
    else
        0.0
    end
    sigma = LANGEVIN_SIGMA_BASE * (1.0 + disagreement) *
            clip01(Float64(token_entropy)) * (1.0 + 0.1 * d_lambda)
    return clip01(sigma)
end

"""
    langevin_step!(model, last_id, mesh7, d_lambda, psi5, direction, plane_scores;
                  phase_weight=1.0) -> NamedTuple

Langevin / biased random walk update for the MESH manifold.

    Δx = μ·dt + σ·dW·√dt

Applied to `last_id`'s embedding and up to 8 outgoing Quipu neighbours.
If `viability = overlap(mesh+Δx, mean_embed7) ≥ LANGEVIN_VIABILITY_THRESHOLD`
the step is **reinforced** (full Δx); otherwise **dampened** by
`LANGEVIN_DIFFUSE_SCALE` (diffusion continues in the imaginary/Weyl layer).
"""
function langevin_step!(
    model::MeshSLMModel, last_id::Int, mesh7, d_lambda::Real,
    psi5, direction::Vector{Float64}, plane_scores::Vector{Float64};
    phase_weight::Real = 1.0,
)
    weyl7 = mcd_lift_weyl_psi(psi5)
    specialist_forces = Vector{Float64}[
        [clip01(mesh7[k] + bias[k]) for k in 1:7]
        for bias in values(model.emergent_biases)
    ]
    drift = langevin_compute_drift(mesh7, weyl7, specialist_forces, direction)
    token_ent = langevin_token_entropy(model)
    sigma = langevin_compute_sigma(plane_scores, d_lambda, token_ent)

    dt = LANGEVIN_DT
    delta_x = [drift[k] * dt + sigma * randn() * sqrt(dt) for k in 1:7]

    # Viability of projected t+
    proj_mesh = [clip01(mesh7[k] + delta_x[k]) for k in 1:7]
    me = mean_embed7(model)
    viability = wavefunction_overlap(proj_mesh, me)

    apply_scale = viability >= LANGEVIN_VIABILITY_THRESHOLD ? 1.0 : LANGEVIN_DIFFUSE_SCALE
    step_mode   = viability >= LANGEVIN_VIABILITY_THRESHOLD ? :reinforce : :diffuse
    effective_dx = delta_x .* apply_scale
    nudge_lr = min(LANGEVIN_MAX_NUDGE, abs(apply_scale) * sigma)

    # Collect affected token IDs: last_id + top outgoing neighbours
    neighbours = [k for (k, w) in
        sort([(key[2], w) for (key, w) in model.quipu_weight if key[1] == last_id];
             by = x -> x[2], rev = true)[1:min(8, length(model.quipu_weight))]]
    affected_ids = unique([last_id; neighbours])
    n_affected = 0
    for tid in affected_ids[1:min(8, length(affected_ids))]
        haskey(model.embeddings, tid) || continue
        e = model.embeddings[tid]
        for k in 1:7
            e[k] = clip01(e[k] + nudge_lr * effective_dx[k])
        end
        n_affected += 1
    end

    return (mode = step_mode,
            delta_x = round.(effective_dx; digits = 6),
            viability = round(viability; digits = 4),
            sigma = round(sigma; digits = 4),
            drift_magnitude = round(sqrt(sum(drift .^ 2)); digits = 4),
            token_entropy = round(token_ent; digits = 4),
            affected_tokens = n_affected)
end

function mcd_dispatch!(
    model::MeshSLMModel, last_id::Int, mesh7, d_lambda::Real, psi5;
    phase_weight::Real = 1.0, mesh_field::Real = 0.0,
)
    if d_lambda < MCD_EMISSION_THRESHOLD
        revived = mcd_revive_weakest!(model, d_lambda; multiplier = 1.0)
        return (mode = :conservative, revived = revived, emitted = 0,
                acre = nothing, resonance_ratio = 1.0, langevin = nothing)
    end

    base_score, resonance_ratio, plane_scores = mcd_multiplanar_score(
        model, last_id, mesh7, psi5; mesh_field = mesh_field
    )
    acre = nothing
    direction = mcd_tangential_direction(mesh7, psi5, phase_weight)  # always compute

    if resonance_ratio >= MCD_RESONANCE_INTEGRATE
        revived = mcd_revive_weakest!(model, d_lambda; multiplier = MCD_INTEGRATE_RATIO)
        lv = langevin_step!(model, last_id, mesh7, d_lambda, psi5, direction, plane_scores;
                            phase_weight = phase_weight)
        return (mode = :integrate, revived = revived, emitted = 0,
                acre = nothing, resonance_ratio = round(resonance_ratio; digits = 4),
                langevin = lv)

    elseif resonance_ratio >= MCD_RESONANCE_TANGENTIAL
        emitted = mcd_emit_along_direction!(model, last_id, direction, d_lambda;
                                            mesh_field = mesh_field)
        acre = acre_emerge!(model; min_resonance = MCD_ACRE_LOW_RESONANCE)
        lv = langevin_step!(model, last_id, mesh7, d_lambda, psi5, direction, plane_scores;
                            phase_weight = phase_weight)
        return (mode = :emit_tangential, revived = 0, emitted = emitted,
                acre = acre, resonance_ratio = round(resonance_ratio; digits = 4),
                langevin = lv)

    else
        # :leave_in_band — Langevin runs in imaginary/Weyl layer (no quipu writes above)
        lv = langevin_step!(model, last_id, mesh7, d_lambda, psi5,
                            mcd_lift_weyl_psi(psi5), plane_scores;
                            phase_weight = phase_weight)
        return (mode = :leave_in_band, revived = 0, emitted = 0,
                acre = nothing, resonance_ratio = round(resonance_ratio; digits = 4),
                langevin = lv)
    end
end

# ---------------------------------------------------------------------------
# Training — Hebbian pass + SCM atrophy/resuscitation (coupling points 1-3)
# ---------------------------------------------------------------------------
"""
    train_round!(model, corpus; mesh_state7=(0.5,...), max_seconds=30.0,
                 phase_weight=1.0) -> NamedTuple

One online training pass with the three SCM live couplings:

1. `progress ← sat(τ)` (SCM saturation fills the End-State slot):
   `η = LR_BASE·(1−sat)·pw·(0.70+0.30·overlap)`, `η_q = QUIPU_LR·(1−sat)·pw`.
2. Post-threshold **atrophy pass**: all Quipu weights ×
   `clamp(1 − ATROPHY_COUPLING·((sat−θ)/(1−θ))·(1−S/S0), 0.5, 1)`.
3. **MCD emission dispatcher** (replaces flat resuscitation): when
   `ΔΛ >= MCD_EMISSION_THRESHOLD` selects `:integrate|:emit_tangential|:leave_in_band`
   based on multiplanar resonance; below threshold falls back to conservative revival.

The ecosystem clock then advances `τ ← τ + scm_dt`. Standard Hebbian
updates are identical to `mesh_slm.train_round`.
"""
function train_round!(
    model::MeshSLMModel, corpus::Vector{<:AbstractString};
    mesh_state7 = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    max_seconds::Real = 30.0, phase_weight::Real = 1.0,
)
    st = scm_state(model)
    progress = clip01(st.sat)                        # coupling point 1
    pw = clamp(Float64(phase_weight), 0.1, 2.0)
    overlap = wavefunction_overlap(mean_embed7(model), mesh_state7)
    mesh_field = mesh_field_scm(model, mesh_state7; phase_weight = pw)
    eta = LR_BASE * (1.0 - progress) * pw * (0.70 + 0.30 * overlap)
    eta_q = QUIPU_LR * (1.0 - progress) * pw

    isempty(corpus) && return (status = "no_corpus", scm_sat = st.sat)

    started = time()
    n_tokens, n_pairs, loss_acc, n_loss = 0, 0, 0.0, 0
    for chunk in shuffle(corpus)
        (time() - started) >= max_seconds && break
        toks = tokenize(chunk)
        length(toks) < 2 && continue
        ids = Int[]
        for tok in toks
            tid = upsert_token!(model, tok, mesh_state7)
            push!(ids, tid)
            e = model.embeddings[tid]
            for k in 1:model.embed_dim
                e[k] += eta * (Float64(mesh_state7[k]) - e[k])
            end
            n_tokens += 1
        end
        for k in 1:(length(ids) - 1)
            key = (ids[k], ids[k + 1])
            if haskey(model.quipu_weight, key)
                w = model.quipu_weight[key]
                loss_acc += (1.0 - w)^2
                n_loss += 1
                model.quipu_weight[key] = w + eta_q * (1.0 - w)
                model.quipu_samples[key] = get(model.quipu_samples, key, 0) + 1
            else
                model.quipu_weight[key] = eta_q
                model.quipu_samples[key] = 1
            end
            n_pairs += 1
        end
    end

    # Coupling point 2 — tangential atrophy of graph knowledge
    atr = max(0.0, st.sat - model.scm.theta)
    factor = clamp(1.0 - ATROPHY_COUPLING * (atr / (1.0 - model.scm.theta)) *
                   (1.0 - st.S / model.scm.s0), 0.5, 1.0)
    if factor < 1.0
        for key in collect(keys(model.quipu_weight))
            model.quipu_weight[key] *= factor
        end
    end

    # Coupling point 3 — MCD emission dispatcher (replaces flat resuscitation)
    d_lambda = scm_cum_intensity(model.scm_clock + model.scm_dt, model.scm) - st.Lambda
    d_lambda = max(0.0, d_lambda)
    # Load stored Weyl psi5 from model meta (falls back to SCM weyl_psi)
    psi5_stored = get(model.slm_meta, "mcd_weyl_psi5", nothing)
    psi5_now = psi5_stored !== nothing ? psi5_stored : collect(weyl_psi(model.scm, model.scm_clock))
    # Pick a representative last_id anchor (most-frequent token)
    anchor_id = isempty(model.freq) ? 0 :
        argmax(Dict(tid => f for (tid, f) in model.freq))
    mcd = if anchor_id > 0
        mcd_dispatch!(model, anchor_id, mesh_state7, d_lambda, psi5_now;
                      phase_weight = pw, mesh_field = mesh_field)
    else
        # No vocab yet — conservative revival
        n_r = mcd_revive_weakest!(model, d_lambda; multiplier = 1.0)
        (mode = :conservative, revived = n_r, emitted = 0, acre = nothing, resonance_ratio = 1.0)
    end
    # For backward compat: n_revive exposed to state_summary
    n_revive = mcd.revived

    model.scm_events_total += d_lambda
    model.scm_clock += model.scm_dt
    model.rounds += 1
    avg_loss = n_loss > 0 ? loss_acc / n_loss : 0.0
    model.last_loss = avg_loss
    model.last_eta = eta
    model.last_mesh_field = mesh_field
    model.last_sat = st.sat
    model.last_eco_health = st.S
    model.last_atrophy_factor = factor
    model.last_revived = n_revive
    model.last_viability = st.V

    # ACRE — accumulate multi-axial interaction; attempt emergence
    acre_observe!(model, mesh_state7, mean_embed7(model))
    acre = if mcd.mode == :emit_tangential
        # Already called with relaxed thresholds inside mcd_dispatch!; skip double-call
        get(mcd, :acre, nothing)
    else
        acre_emerge!(model)
    end

    return (status = "ok", rounds = model.rounds, tokens = n_tokens,
            pairs = n_pairs, avg_loss = round(avg_loss; digits = 6),
            eta = round(eta; digits = 5),
            mesh_field = round(mesh_field; digits = 4),
            scm_sat = round(st.sat; digits = 4),
            scm_health = round(st.S; digits = 4),
            scm_lambda = round(st.lam; digits = 4),
            scm_viability = round(st.V; digits = 2),
            atrophy_factor = round(factor; digits = 4),
            revived_edges = n_revive,
            emission_mode = mcd.mode,
            mcd = mcd,
            acre = acre)
end

# ---------------------------------------------------------------------------
# Scoring + generation (as in the -GLM-GNN port, on the SCM-extended field)
# ---------------------------------------------------------------------------
"""Top-K candidates: `QUIPU_GAIN·w + (PROX+WARP·warp)·prox + embed·mesh + FIELD·field`."""
function score_candidates(
    model::MeshSLMModel, last_id::Int, mesh7;
    top_k::Int = 32, mesh_field::Real = 0.0,
    specialist::Union{Nothing,AbstractString} = nothing,
)
    last_pos = get(model.positions, last_id, (0, 0))
    boost_active = specialist === nothing || specialist == "supply_chain_optimizer"
    outgoing = Tuple{Int,Float64}[]
    for (key, w) in model.quipu_weight
        key[1] == last_id && push!(outgoing, (key[2], w))
    end
    sort!(outgoing; by = x -> x[2], rev = true)
    outgoing = outgoing[1:min(top_k * 3, length(outgoing))]

    scored = NamedTuple[]
    for (dst_id, w) in outgoing
        haskey(model.positions, dst_id) || continue
        dst_pos = model.positions[dst_id]
        tok = get(model.tokens, dst_id, "")
        prox = proximity(last_pos, dst_pos, model.torus_n)
        eff = max(prox, 0.01)
        warp = clip01(metric_perturbation(w * METRIC_MASS_SCALE, eff) * eff)
        embed = get(model.embeddings, dst_id, zeros(model.embed_dim))
        s = QUIPU_GAIN * w + (PROX_GAIN + WARP_GAIN * warp) * prox +
            embed_dot(embed, mesh7) + MESH_FIELD_GAIN * mesh_field
        if boost_active && (tok in NUMERIC_BOOST_TOKENS ||
                             any(k -> occursin(k, tok), NUMERIC_BOOST_SUBSTRINGS))
            s += OPT_BOOST
        end
        push!(scored, (token_id = dst_id, score = s, token = tok, pos = dst_pos))
    end
    if isempty(scored)
        ranked = sort(collect(keys(model.positions)); by = tid -> get(model.freq, tid, 0), rev = true)
        for tid in ranked[1:min(top_k, length(ranked))]
            pos = model.positions[tid]
            embed = get(model.embeddings, tid, zeros(model.embed_dim))
            s = embed_dot(embed, mesh7) + PROX_GAIN * proximity(last_pos, pos, model.torus_n) +
                MESH_FIELD_GAIN * mesh_field
            push!(scored, (token_id = tid, score = s, token = get(model.tokens, tid, ""), pos = pos))
        end
    end
    sort!(scored; by = c -> c.score, rev = true)
    return scored[1:min(top_k, length(scored))]
end

"""Temperature softmax over scored candidates (greedy at T ≤ 0)."""
function softmax_pick(cands, temperature::Real; rng = Random.default_rng())
    isempty(cands) && error("softmax_pick: cands must be non-empty")
    temperature <= 0.0 && return cands[1]
    scores = [c.score / max(temperature, 1.0e-3) for c in cands]
    m = maximum(scores)
    exps = [exp(s - m) for s in scores]
    total = max(sum(exps), eps())
    r = rand(rng) * total
    acc = 0.0
    for (c, e) in zip(cands, exps)
        acc += e
        acc >= r && return c
    end
    return cands[end]
end

"""
    generate(model, prompt; max_new_tokens=24, temperature=0.7, seed=nothing,
             specialist=nothing, mesh_state7=(0.5,...), phase_weight=1.0)
        -> NamedTuple

SLM generation on the SCM-extended MESH field; result includes the live
SCM diagnostics `(scm_sat, scm_viability)` so callers can gate fallback on
ecosystem state as well as `confidence < CONF_FLOOR`.
"""
function generate(
    model::MeshSLMModel, prompt::AbstractString;
    max_new_tokens::Int = 24, temperature::Real = 0.7,
    seed::Union{Nothing,Integer} = nothing,
    specialist::Union{Nothing,AbstractString} = nothing,
    mesh_state7 = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    phase_weight::Real = 1.0,
)
    seed === nothing || Random.seed!(seed)
    mesh = apply_specialist_bias(mesh_state7, specialist, model.emergent_biases)
    prompt_tokens = tokenize(prompt)
    st = scm_state(model)

    n_vocab = length(model.positions)
    if n_vocab < MIN_VOCAB_FOR_GENERATE
        return (text = "", confidence = 0.0, tokens_emitted = 0,
                vocab_hit_rate = 0.0, vocab_size = n_vocab,
                mesh_field = 0.0, scm_sat = st.sat, scm_viability = st.V,
                reason = "vocab_too_small")
    end

    anchor = nothing
    hits = 0
    for tok in reverse(prompt_tokens)
        tid = get(model.vocab_id, tok, nothing)
        if tid !== nothing
            hits += 1
            anchor === nothing && (anchor = (tid, model.positions[tid]))
        end
    end
    for tok in prompt_tokens
        haskey(model.vocab_id, tok) && (hits += 1)
    end
    vocab_hit_rate = isempty(prompt_tokens) ? 0.0 : hits / max(1, length(prompt_tokens) * 2)

    if anchor === nothing
        best, best_freq = nothing, -1
        for tid in keys(model.positions)
            f = get(model.freq, tid, 0)
            f > best_freq && ((best, best_freq) = (tid, f))
        end
        best === nothing && return (text = "", confidence = 0.0,
                tokens_emitted = 0, vocab_hit_rate = 0.0, vocab_size = n_vocab,
                mesh_field = 0.0, scm_sat = st.sat, scm_viability = st.V,
                reason = "empty_vocab")
        anchor = (best, model.positions[best])
    end

    mesh_field = mesh_field_scm(model, mesh; phase_weight = phase_weight)
    last_id, _ = anchor
    emitted = String[]
    cumulative = 0.0
    for _ in 1:max_new_tokens
        cands = score_candidates(model, last_id, mesh; top_k = 12,
                                  mesh_field = mesh_field, specialist = specialist)
        isempty(cands) && break
        pick = softmax_pick(cands, temperature)
        last_id = pick.token_id
        push!(emitted, pick.token)
        cumulative += pick.score
    end
    avg_score = isempty(emitted) ? 0.0 : cumulative / length(emitted)
    confidence = clip01(tanh(0.5 * avg_score) * 0.7 + vocab_hit_rate * 0.3)

    return (
        text = join(emitted, " "),
        confidence = round(confidence; digits = 4),
        tokens_emitted = length(emitted),
        vocab_hit_rate = round(vocab_hit_rate; digits = 4),
        vocab_size = n_vocab,
        avg_score = round(avg_score; digits = 4),
        mesh_field = round(mesh_field; digits = 4),
        scm_sat = round(st.sat; digits = 4),
        scm_viability = round(st.V; digits = 2),
    )
end

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
"""
    state_summary(model) -> NamedTuple

Snapshot: graph size, rounds, last LR/field, and the SCM life-cycle block
(clock, saturation, ecosystem health, cumulative expected events, last
atrophy factor / revived edges, viability, Weyl Ψ bridge).
"""
function state_summary(model::MeshSLMModel)
    return (
        model_arch = MODEL_ARCH,
        vocab_size = length(model.positions),
        quipu_edges = length(model.quipu_weight),
        rounds = model.rounds,
        last_loss = model.last_loss,
        last_eta = model.last_eta,
        last_mesh_field = model.last_mesh_field,
        scm_clock_years = model.scm_clock,
        scm_sat = round(model.last_sat; digits = 4),
        scm_health = round(model.last_eco_health; digits = 4),
        scm_events_total = round(model.scm_events_total; digits = 2),
        scm_atrophy_factor = round(model.last_atrophy_factor; digits = 4),
        scm_revived_edges = model.last_revived,
        scm_viability = round(model.last_viability; digits = 2),
        weyl_psi = round.(weyl_psi(model); digits = 4),
        emergent_specialists = sort(collect(keys(model.emergent_biases))),
    )
end

end # module MeshSLMSCMGLMGNN

# ---------------------------------------------------------------------------
# Demo entry point — `julia mesh_slm_scm_glm_gnn_model.jl`
# Trains across the SCM life cycle (grow → saturate → atrophy → resuscitate)
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    using .MeshSLMSCMGLMGNN

    model = MeshSLMModel(scm_dt = 0.5)   # 0.5 ecosystem-years per round
    corpus = [
        "supply chain optimization eoq formula sqrt demand cost",
        "eoq demand safety stock lead cost hold order",
        "hierarchical eoq empirical bayes shrinkage multi echelon",
        "safety stock protects against demand and lead time variability",
        "holding cost and ordering cost trade off in the eoq model",
    ]

    println("MESH-SLM-SCM-GLM-GNN — training across the SCM life cycle:")
    for round_i in 1:24   # 12 ecosystem-years: crosses theta = 0.72 at ~9.3 yr
        r = train_round!(model, corpus; max_seconds = 5.0)
        if round_i in (1, 8, 16, 24)
            println("  round $(lpad(round_i, 2)): eta=$(r.eta) sat=$(r.scm_sat) ",
                    "S=$(r.scm_health) atrophy_factor=$(r.atrophy_factor) ",
                    "revived=$(r.revived_edges) V=$(r.scm_viability)")
        end
    end

    println("\nstate_summary():")
    println("  ", state_summary(model))

    result = generate(model, "eoq demand"; max_new_tokens = 8,
                       temperature = 0.7, seed = 42)
    println("\ngenerate():")
    println("  text          = ", result.text)
    println("  confidence    = ", result.confidence)
    println("  mesh_field    = ", result.mesh_field, "  (SCM-extended blend)")
    println("  scm_sat       = ", result.scm_sat)
    println("  scm_viability = ", result.scm_viability)
end
