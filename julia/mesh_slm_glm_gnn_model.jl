"""
    MeshSLMGLMGNN

Julia port of the **holistic** Supply Chain Brain MESH-SLM-GLM-GNN
(`pipeline/src/quipu/mesh_slm.py`, `MODEL_ARCH = "MESH-SLM-GLM-GNN"`),
single-file and self-contained (no `include` of the companion
`mesh_inference_model.jl` — some functions below intentionally duplicate
that file so this one stands alone).

The architecture label, as defined in the Python source, names four
cooperating aspects — all four are modeled here:

  * **MESH** — a 7-D local + 8th-D global state field (UEQGM aspect
    integrals), see [`mesh_field_8d`](@ref).
  * **SLM**  — Small Language Model next-token prediction with
    temperature sampling, see [`generate`](@ref).
  * **GLM**  — Graph Language Model: graph-anchored 7-D token embeddings,
    see [`MeshSLMModel.embeddings`](@ref) and [`embed_dot`](@ref).
  * **GNN**  — Graph Neural Network: Hebbian message-passing edges
    (Quipu bigrams), see [`train_round!`](@ref).

## What's modeled vs. treated as external input

This file models the SLM-GLM-GNN's *own* state machine: tokenizer → torus
vocab allocation (with saturation eviction) → 7-D embeddings → Quipu
bigram edges → Hebbian training → 8th-D MESH field → scoring → generation.

Exactly as in the Python source, three categories of input are supplied
by *other* Brain subsystems and are therefore modeled here as external
parameters with the same neutral fallbacks, not recomputed:

  * `mesh_state7` — the live 7-D MESH activation vector (from
    `system_entirety`); fallback `(0.5,...,0.5)`.
  * `phase_weight` / `coherence_depth` / `weyl_phase` — UEQGM
    adaptive-runtime scalars (from `ueqgm_engine.refresh_adaptive_runtime`
    / `temporal_spatiality`); fallbacks `1.0` / `0` / `0.0`.
  * the training corpus text itself (from `corpus_entity` / `corpus_edge`
    / `llm_dispatch_log`, i.e. `mesh_slm._corpus_stream`); the caller
    passes a `Vector{String}` of chunks directly to [`train_round!`](@ref).

Also out of scope, as separate auxiliary subsystems layered on top of the
core SLM-GLM-GNN rather than part of it: the photon/neutrino resuscitation
overlay (`map_resuscitation_quipu`), the reversible local-minima
distillation overlay (`quipu_minimal.py` — see the companion
`mesh_compression_model.jl` for the WRITE/READ compression accounting),
and the domain-specific EOQ "hybrid grounding" hook inside Python's
`generate()`.

# Quick start

```julia
model = MeshSLMModel()
corpus = [
    "eoq demand safety stock lead cost hold order",
    "supply chain optimization eoq formula sqrt demand cost",
]
train_round!(model, corpus)
result = generate(model, "eoq demand")
println(result.text, "  confidence=", result.confidence)
```
"""
module MeshSLMGLMGNN

using Random

export
    MODEL_ARCH,
    MeshSLMModel,
    tokenize,
    upsert_token!,
    allocate_coord!,
    torus_dist,
    proximity,
    embed_dot,
    coherence_to_phi,
    sici_axial_decay,
    wavefunction_overlap,
    floquet_modulation_factor,
    holographic_entropy,
    metric_perturbation,
    phase_evolution_total,
    mean_embed7,
    mesh_field_8d,
    apply_specialist_bias,
    acre_observe!,
    acre_principal_axis,
    acre_emerge!,
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
# Model identity (mirrors mesh_slm.py module-level constants)
# ---------------------------------------------------------------------------
const MODEL_ARCH = "MESH-SLM-GLM-GNN"

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
const EMBED_DIM         = 7
const LR_BASE           = 0.04   # base embedding learning rate
const QUIPU_LR          = 0.06   # bigram edge Hebbian update rate

const QUIPU_GAIN        = 0.55   # bigram weight in scoring
const PROX_GAIN         = 0.25   # toroidal proximity weight in scoring
const MESH_FIELD_GAIN   = 0.18   # 8th-D MESH field contribution to scoring
const WARP_GAIN         = 0.06   # per-candidate metric-warp amplification
const METRIC_MASS_SCALE = 6.7e26
const CONF_FLOOR        = 0.22
const MIN_VOCAB_FOR_GENERATE = 16

const G_CONST = 6.674e-11   # gravitational constant, m^3 kg^-1 s^-2
const C_CONST = 2.998e8     # speed of light,        m s^-1
const GAMMA_0_DEFAULT = 1.0
const TAN_CLAMP = 1.0e3

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

# ACRE — Axial Cross-Resonance Emergence (ports of mesh_slm.acre_*)
const AXES = ("vision", "touch", "smell", "body", "brain", "perception", "entirety")
const ACRE_EMA_RHO          = 0.15   # interaction-matrix EMA update rate
const ACRE_MIN_RESONANCE    = 0.35   # min dominant-eigenvalue share of trace
const ACRE_NOVELTY_CEILING  = 0.60   # max overlap vs uniform + existing biases
const ACRE_BIAS_SCALE       = 0.25   # max bias component (hand-tuned scale)
const ACRE_MIN_COMPONENT    = 0.02   # sparsify components below this
const ACRE_MAX_EMERGENT     = 4      # capacity cap on emergent specialists

# MCD — Multiplanar Comparative Decompression (proxy ΔΛ via holographic entropy)
const MCD_EMISSION_THRESHOLD    = 0.50   # ΔH threshold to activate mode selection
const MCD_INTEGRATE_RATIO       = 1.50   # edge-revival multiplier for :integrate
const MCD_RESONANCE_INTEGRATE   = 1.15   # resonance_ratio → :integrate
const MCD_RESONANCE_TANGENTIAL  = 0.90   # resonance_ratio → :emit_tangential
const MCD_TANGENTIAL_LR         = 0.12   # Hebbian nudge along tangential direction
const MCD_ACRE_LOW_RESONANCE    = 0.25   # relaxed ACRE gate for :emit_tangential

# Langevin dynamics — stochastic MESH manifold evolution
const LANGEVIN_DT                  = 0.01
const LANGEVIN_SIGMA_BASE          = 0.05
const LANGEVIN_DRIFT_ALPHA         = 0.30
const LANGEVIN_DRIFT_BETA          = 0.20
const LANGEVIN_DRIFT_GAMMA         = 0.50
const LANGEVIN_VIABILITY_THRESHOLD = 0.60
const LANGEVIN_DIFFUSE_SCALE       = 0.30
const LANGEVIN_MAX_NUDGE           = 0.05
const REVIVE_EDGES_PER_EVENT    = 8      # base edges revived per emission event
const RESUSC_LR                 = 0.10   # Hebbian revival nudge

# Tokenizer — port of mesh_slm._TOKEN_RE (word-initial alpha runs, bare
# numbers, math operators, and a small supply-chain keyword allowlist).
const TOKEN_RE = r"[A-Za-z][A-Za-z0-9_\-]{1,30}|\d+(?:\.\d+)?|[+\-*/=<>^]|\b(?:sqrt|eoq|safety|stock|demand|lead|cost|hold|order)\b"

"""
    clip01(x) -> Float64

Clamp `x` to `[0.0, 1.0]`; non-finite input maps to `0.0`.
"""
function clip01(x)::Float64
    v = try
        Float64(x)
    catch
        return 0.0
    end
    isfinite(v) || return 0.0
    return max(0.0, min(1.0, v))
end

# ---------------------------------------------------------------------------
# The model — mutable state mirroring mesh_slm_vocab / _embed / _quipu / _meta
# ---------------------------------------------------------------------------
"""
    MeshSLMModel(; torus_n=64, embed_dim=7)

Mutable in-memory MESH-SLM-GLM-GNN. Fields mirror the SQLite tables in
`STRUCTURAL_MAP` (`mesh_slm.py`):

| field                  | mirrors table / column                         |
|------------------------|-------------------------------------------------|
| `vocab_id`, `tokens`   | `mesh_slm_vocab.token` (both directions)        |
| `positions`, `occupied`| `mesh_slm_vocab.(i,j)` (both directions)        |
| `freq`, `first_seen`, `last_seen` | `mesh_slm_vocab.*`                    |
| `embeddings`           | `mesh_slm_embed.e_vision .. e_entirety`         |
| `quipu_weight`, `quipu_samples` | `mesh_slm_quipu.(weight, samples)`     |
| `rounds`, `last_*`     | `mesh_slm_meta` key/value diagnostics           |

`torus_n` is a per-instance field here (the Python source fixes it at the
module-level constant 64) so the model can be experimented with at other
vocabulary ceilings.
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
    first_seen::Dict{Int,Float64}
    last_seen::Dict{Int,Float64}
    embeddings::Dict{Int,Vector{Float64}}
    quipu_weight::Dict{Tuple{Int,Int},Float64}
    quipu_samples::Dict{Tuple{Int,Int},Int}
    rounds::Int
    last_loss::Float64
    last_eta::Float64
    last_phase_weight::Float64
    last_wavefunction_overlap::Float64
    last_mesh_field_8d::Float64
    acre_interaction::Matrix{Float64}                 # ACRE 7×7 multi-axial EMA
    emergent_biases::Dict{String,NTuple{7,Float64}}   # ACRE emergent specialists
    prev_holographic_h::Float64                       # MCD: previous holographic entropy H
end

function MeshSLMModel(; torus_n::Int = 64, embed_dim::Int = 7)
    return MeshSLMModel(
        torus_n, torus_n * torus_n, embed_dim, 1,
        Dict{String,Int}(), Dict{Int,String}(), Dict{Int,Tuple{Int,Int}}(),
        Dict{Tuple{Int,Int},Int}(), Dict{Int,Int}(),
        Dict{Int,Float64}(), Dict{Int,Float64}(),
        Dict{Int,Vector{Float64}}(),
        Dict{Tuple{Int,Int},Float64}(), Dict{Tuple{Int,Int},Int}(),
        0, 0.0, 0.0, 1.0, 0.5, 0.0,
        zeros(7, 7), Dict{String,NTuple{7,Float64}}(), 0.0,
    )
end

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
"""
    tokenize(text) -> Vector{String}

Split `text` into lowercase tokens (max 256 per call). Port of
`mesh_slm._tokenize`.
"""
function tokenize(text::AbstractString)::Vector{String}
    isempty(text) && return String[]
    matches = [lowercase(m.match) for m in eachmatch(TOKEN_RE, text)]
    return matches[1:min(256, length(matches))]
end

# ---------------------------------------------------------------------------
# Torus vocab allocation (with saturation eviction)
# ---------------------------------------------------------------------------
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
    delete!(model.first_seen, token_id)
    delete!(model.last_seen, token_id)
    return nothing
end

"""
    allocate_coord!(model, token) -> (i, j)

Hash-then-linear-probe a free torus cell for `token`. When the torus is
saturated, evicts the lowest-frequency, oldest-`last_seen` token (freeing
its cell for reuse) — mirrors `ORDER BY freq ASC, last_seen ASC LIMIT 1`
in `mesh_slm._allocate_coord`.

Note: Julia's `hash` and Python's (hash-randomized by default) `hash` will
not agree numerically — only the *algorithm* (hash → probe → evict LRU) is
ported; torus placement is an internal addressing detail, not semantically
meaningful beyond relative proximity structure.
"""
function allocate_coord!(model::MeshSLMModel, token::AbstractString)::Tuple{Int,Int}
    base = Int(hash(token) % UInt(model.vocab_limit))
    for step in 0:(model.vocab_limit - 1)
        idx = (base + step) % model.vocab_limit
        i, j = divrem(idx, model.torus_n)
        haskey(model.occupied, (i, j)) || return (i, j)
    end

    victim = nothing
    victim_freq = typemax(Int)
    victim_seen = Inf
    for tid in keys(model.positions)
        f = get(model.freq, tid, 0)
        seen = get(model.last_seen, tid, 0.0)
        if f < victim_freq || (f == victim_freq && seen < victim_seen)
            victim = tid
            victim_freq = f
            victim_seen = seen
        end
    end
    victim === nothing && return (0, 0)
    victim_pos = model.positions[victim]
    evict!(model, victim)
    return victim_pos
end

"""
    upsert_token!(model, token, mesh_state7) -> Int

Insert `token` into the vocab torus (seeding its 7-D embedding with a
small MESH-state-aligned bias) or increment its frequency counter when
already known. Port of `mesh_slm._upsert_token`.
"""
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
    model.first_seen[tid] = now
    model.last_seen[tid] = now
    model.embeddings[tid] = [0.05 * Float64(v) + (rand() - 0.5) * 0.02 for v in mesh_state7]
    return tid
end

# ---------------------------------------------------------------------------
# Toroidal geometry
# ---------------------------------------------------------------------------
"""
    torus_dist(a, b, torus_n) -> Int

Wrap-aware L1 distance between torus cells `a` and `b`. Port of
`mesh_slm._torus_dist`.
"""
function torus_dist(a::Tuple{Int,Int}, b::Tuple{Int,Int}, torus_n::Int)::Int
    di = abs(a[1] - b[1])
    dj = abs(a[2] - b[2])
    return min(di, torus_n - di) + min(dj, torus_n - dj)
end

"""
    proximity(a, b, torus_n) -> Float64

Linear proximity in `[0, 1]`, decaying from `1.0` at `a == b` to `0.0` at
the maximum torus distance. Port of `mesh_slm._proximity`.
"""
function proximity(a::Tuple{Int,Int}, b::Tuple{Int,Int}, torus_n::Int)::Float64
    d = torus_dist(a, b, torus_n)
    return max(0.0, 1.0 - d / Float64(torus_n))
end

# ---------------------------------------------------------------------------
# GLM alignment term
# ---------------------------------------------------------------------------
"""
    embed_dot(embed7, mesh7) -> Float64

Unnormalised dot product of a token's 7-D embedding with the live MESH
state. Port of `mesh_slm._embed_dot`.
"""
embed_dot(embed7, mesh7) = sum(a * b for (a, b) in zip(embed7, mesh7))

# ---------------------------------------------------------------------------
# UEQGM physics helpers needed by mesh_field_8d
# (ports of pipeline/src/quipu/ueqgm_engine.py)
# ---------------------------------------------------------------------------
coherence_to_phi(coherence::Integer) = pi / 4 + coherence * pi

function _raw_sici(phi::Real)::Tuple{Float64,Float64}
    x = phi != 0.0 ? abs(phi) : 1.0e-12
    si_val = x - x^3 / 18.0 + x^5 / 600.0 - x^7 / 35280.0
    euler_mascheroni = 0.5772156649
    ci_val = euler_mascheroni + log(x) - x^2 / 4.0 + x^4 / 96.0
    return (phi >= 0 ? si_val : -si_val), ci_val
end

"""
    sici_axial_decay(phi, gamma_0=GAMMA_0_DEFAULT) -> Float64

Port of `ueqgm_engine.sici_axial_decay`: `Si(φ)·Ci(φ)·tan(φ)·Γ0`, with
`tan(φ)` clamped to `±TAN_CLAMP`.
"""
function sici_axial_decay(phi::Real, gamma_0::Real = GAMMA_0_DEFAULT)::Float64
    si, ci = _raw_sici(phi)
    tan_phi = clamp(tan(phi), -TAN_CLAMP, TAN_CLAMP)
    return si * ci * tan_phi * gamma_0
end

"""
    wavefunction_overlap(a, b) -> Float64

Port of `ueqgm_engine.wavefunction_overlap`: `|⟨ψ_a|ψ_b⟩|² = cos²θ(a, b)`.
"""
function wavefunction_overlap(a, b)::Float64
    (isempty(a) || isempty(b) || length(a) != length(b)) && return 0.0
    dot = sum(ai * bi for (ai, bi) in zip(a, b))
    norm_a = sqrt(sum(ai * ai for ai in a))
    norm_b = sqrt(sum(bi * bi for bi in b))
    (norm_a == 0.0 || norm_b == 0.0) && return 0.0
    cos_theta = dot / (norm_a * norm_b)
    return round(cos_theta^2, digits = 6)
end

"""
    floquet_modulation_factor(t, omega) -> Float64

Port of `ueqgm_engine.floquet_modulation_factor`: `cos(ω·t)`.
"""
floquet_modulation_factor(t::Real, omega::Real) = cos(omega * t)

"""
    holographic_entropy(n_edges, n_nodes) -> Float64

Port of `ueqgm_engine.holographic_entropy`: `n_edges / (n_nodes + 1)`.
"""
holographic_entropy(n_edges::Real, n_nodes::Real) = n_edges / (n_nodes + 1)

"""
    metric_perturbation(mass_eff, r) -> Float64

Port of `ueqgm_engine.metric_perturbation`: `2·G·mass_eff / (c²·r)`,
`0.0` for `r ≤ 0`.
"""
function metric_perturbation(mass_eff::Real, r::Real)::Float64
    r <= 0.0 && return 0.0
    return 2.0 * G_CONST * mass_eff / (C_CONST^2 * r)
end

"""
    phase_evolution_total(phi; delta_mu=0.0, delta_q=0.0, delta_gamma=0.0,
                           gamma_0=GAMMA_0_DEFAULT, gamma_eff=GAMMA_0_DEFAULT)
        -> Float64

Port of `ueqgm_engine.phase_evolution_total`:
`δφ_μ + δφ_q + δφ_γ + Δλ_axial·2π/Γ_eff`.
"""
function phase_evolution_total(
    phi::Real; delta_mu::Real = 0.0, delta_q::Real = 0.0, delta_gamma::Real = 0.0,
    gamma_0::Real = GAMMA_0_DEFAULT, gamma_eff::Real = GAMMA_0_DEFAULT,
)::Float64
    axial = sici_axial_decay(phi, gamma_0)
    axial_phase = axial * (2.0 * pi / max(gamma_eff, 1.0e-30))
    return delta_mu + delta_q + delta_gamma + axial_phase
end

"""
    mean_embed7(model; limit=128) -> NTuple{7,Float64}

Mean 7-D embedding across the `limit` most-frequent tokens; neutral
midpoint when `model` has no embeddings yet. Port of
`mesh_slm._mean_embed_7d`.
"""
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
# 8th-D MESH field
# ---------------------------------------------------------------------------
"""
    mesh_field_8d(; n_vocab, n_quipu_edges, mean_embed7, mesh_state7,
                  phase_weight=1.0, coherence_depth=0, weyl_phase=0.0,
                  vocab_limit, metric_mass_scale=METRIC_MASS_SCALE) -> Float64

`field = 0.30·H + 0.25·F + 0.25·O + 0.10·P + 0.10·W` — see the module
docstring's architecture table. Port of `mesh_slm._mesh_field_8d`.
"""
function mesh_field_8d(;
    n_vocab::Integer, n_quipu_edges::Integer, mean_embed7, mesh_state7,
    phase_weight::Real = 1.0, coherence_depth::Integer = 0, weyl_phase::Real = 0.0,
    vocab_limit::Integer, metric_mass_scale::Real = METRIC_MASS_SCALE,
)::Float64
    H = clip01(holographic_entropy(n_quipu_edges, n_vocab) / 8.0)

    phi = coherence_to_phi(coherence_depth)

    omega = max(0.1, phase_weight)
    raw_F = floquet_modulation_factor(mod(weyl_phase, 2.0 * pi), omega)
    F = clip01(0.5 * (1.0 + raw_F))

    O = wavefunction_overlap(mean_embed7, mesh_state7)

    raw_P = abs(phase_evolution_total(phi))
    P = clip01(raw_P / (2.0 * pi))

    vocab_fill = n_vocab / max(vocab_limit, 1)
    r = max(1.0 - 0.5 * vocab_fill, 0.01)
    raw_W = metric_perturbation(vocab_fill * metric_mass_scale, r)
    W = clip01(raw_W * r)

    return clip01(0.30 * H + 0.25 * F + 0.25 * O + 0.10 * P + 0.10 * W)
end

# ---------------------------------------------------------------------------
# Specialist bias
# ---------------------------------------------------------------------------
"""
    apply_specialist_bias(mesh7, specialist, emergent=Dict()) -> NTuple{7,Float64}

Port of `mesh_slm._apply_specialist_bias`. Resolution order: hand-tuned
`SPECIALIST_BIASES` first, then the model's ACRE `emergent` dict; unknown
specialists leave the mesh unchanged.
"""
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
# ACRE — Axial Cross-Resonance Emergence (emergent specialist creation)
# ---------------------------------------------------------------------------
"""
    acre_observe!(model, mesh_state7, mean_embed; rho=ACRE_EMA_RHO) -> Matrix

Accumulate one multi-axial interaction observation into the model's ACRE
matrix — Hebbian EMA of the symmetrized outer product
`C ← (1−ρ)C + ρ·½(m e' + e m')` with both vectors clipped to [0, 1] so C
stays entrywise non-negative (Perron–Frobenius guarantee for
[`acre_emerge!`](@ref)). Port of `mesh_slm.acre_observe`.
"""
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

"""
    acre_principal_axis(C) -> (axis, eigenvalue, trace)

Dominant eigenpair by power iteration (32 iterations; ample for 7×7).
Canonical sign then clamped non-negative — exact for a non-negative
symmetric matrix. Port of `mesh_slm._acre_principal_axis`.
"""
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

"""
    acre_emerge!(model; min_resonance=ACRE_MIN_RESONANCE,
                 novelty_ceiling=ACRE_NOVELTY_CEILING,
                 bias_scale=ACRE_BIAS_SCALE, max_emergent=ACRE_MAX_EMERGENT)
        -> NamedTuple

Attempt emergent-specialist creation from the ACRE interaction matrix.
Gates (in order): **signal** (trace > 1e-6), **capacity**
(< max_emergent), **resonance** (dominant eigenvalue share of trace ≥
min_resonance — one coherent multi-axial mode), **novelty**
(`wavefunction_overlap` < novelty_ceiling vs the uniform direction AND
every existing bias, hand-tuned + emergent). On success the Perron axis is
rescaled to hand-tuned magnitude, sparsified, named
`emergent_<axis1>_<axis2>` after its two strongest axes, and registered in
`model.emergent_biases` — immediately addressable via
`generate(…; specialist=…)`. Port of `mesh_slm.acre_emerge`.
"""
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
Ψ0→vision, Ψ1→touch, Ψ2→smell, Ψ3→brain(0.6)+entirety(0.4), Ψ4→perception.
"""
function mcd_lift_weyl_psi(psi5)::Vector{Float64}
    out = zeros(7)
    if length(psi5) >= 1; out[1] = clip01(psi5[1]) end
    if length(psi5) >= 2; out[2] = clip01(psi5[2]) end
    if length(psi5) >= 3; out[3] = clip01(psi5[3]) end
    if length(psi5) >= 4
        out[5] = clip01(0.6 * psi5[4])
        out[7] = clip01(0.4 * psi5[4])
    end
    if length(psi5) >= 5; out[6] = clip01(psi5[5]) end
    return out
end

"""
    mcd_multiplanar_score(model, last_id, mesh7, psi5; mesh_field=0.0)
        -> (base_score, resonance_ratio)

Score across base mesh, each emergent-bias plane, reflected Ψ, and inverted Ψ.
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
    push!(planes, [clip01(1.0 - v) for v in lifted])
    push!(planes, [clip01(v)       for v in lifted])

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

Lagrangian photon/neutrino phase interaction direction, normalised to max 1.
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

Revive the `round(ΔΛ·REVIVE_EDGES_PER_EVENT·multiplier)` weakest quipu edges.
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

# ---------------------------------------------------------------------------
# Langevin dynamics helpers
# ---------------------------------------------------------------------------
function langevin_token_entropy(model::MeshSLMModel; limit::Int = 128)::Float64
    isempty(model.freq) && return 0.5
    top_freqs = sort(collect(values(model.freq)); rev = true)[1:min(limit, length(model.freq))]
    total = max(sum(top_freqs), 1)
    entropy = -sum((f / total) * log(f / total + 1e-12) for f in top_freqs)
    max_entropy = log(max(length(top_freqs), 1) + 1)
    return clip01(entropy / max(max_entropy, 1.0))
end

function langevin_compute_drift(mesh7, weyl7, specialist_forces, direction)::Vector{Float64}
    m = collect(Float64, mesh7)
    w = collect(Float64, weyl7)
    d = collect(Float64, direction)
    mean_force = isempty(specialist_forces) ? zeros(7) :
        [sum(sf[k] for sf in specialist_forces) / length(specialist_forces) for k in 1:7]
    return [
        LANGEVIN_DRIFT_ALPHA * (clip01(w[k]) - m[k])
        + LANGEVIN_DRIFT_BETA  * (mean_force[k] - m[k])
        + LANGEVIN_DRIFT_GAMMA * d[k]
        for k in 1:7
    ]
end

function langevin_compute_sigma(
    plane_scores::Vector{Float64}, d_lambda::Real, token_entropy::Real
)::Float64
    disagreement = length(plane_scores) > 1 ? begin
        mean_s = sum(plane_scores) / length(plane_scores)
        sqrt(sum((s - mean_s)^2 for s in plane_scores)) / (abs(mean_s) + 1e-9)
    end : 0.0
    sigma = LANGEVIN_SIGMA_BASE * (1.0 + disagreement) *
            clip01(Float64(token_entropy)) * (1.0 + 0.1 * d_lambda)
    return clip01(sigma)
end

"""
    langevin_step!(model, last_id, mesh7, d_lambda, psi5, direction, plane_scores;
                  phase_weight=1.0) -> NamedTuple
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
    drift     = langevin_compute_drift(mesh7, weyl7, specialist_forces, direction)
    token_ent = langevin_token_entropy(model)
    sigma     = langevin_compute_sigma(plane_scores, d_lambda, token_ent)

    dt = LANGEVIN_DT
    delta_x = [drift[k] * dt + sigma * randn() * sqrt(dt) for k in 1:7]

    proj_mesh  = [clip01(mesh7[k] + delta_x[k]) for k in 1:7]
    viability  = wavefunction_overlap(proj_mesh, mean_embed7(model))
    apply_scale = viability >= LANGEVIN_VIABILITY_THRESHOLD ? 1.0 : LANGEVIN_DIFFUSE_SCALE
    step_mode   = viability >= LANGEVIN_VIABILITY_THRESHOLD ? :reinforce : :diffuse
    effective_dx = delta_x .* apply_scale
    nudge_lr = min(LANGEVIN_MAX_NUDGE, abs(apply_scale) * sigma)

    neighbours = [k for (k, _) in
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

"""
    mcd_dispatch!(model, last_id, mesh7, d_lambda_proxy, psi5=(0.5,...);
                  phase_weight=1.0, mesh_field=0.0, weyl_phase=0.0) -> NamedTuple

Multiplanar Comparative Decompression dispatcher for the non-SCM model.
`d_lambda_proxy` is derived from holographic-entropy growth (ΔH = H_now − H_prev).
"""
function mcd_dispatch!(
    model::MeshSLMModel, last_id::Int, mesh7, d_lambda_proxy::Real,
    psi5 = fill(0.5, 5);
    phase_weight::Real = 1.0, mesh_field::Real = 0.0,
)
    if d_lambda_proxy < MCD_EMISSION_THRESHOLD
        revived = mcd_revive_weakest!(model, d_lambda_proxy; multiplier = 1.0)
        return (mode = :conservative, revived = revived, emitted = 0,
                acre = nothing, resonance_ratio = 1.0, langevin = nothing)
    end

    base_score, resonance_ratio, plane_scores = mcd_multiplanar_score(
        model, last_id, mesh7, psi5; mesh_field = mesh_field
    )
    direction = mcd_tangential_direction(mesh7, psi5, phase_weight)

    if resonance_ratio >= MCD_RESONANCE_INTEGRATE
        revived = mcd_revive_weakest!(model, d_lambda_proxy; multiplier = MCD_INTEGRATE_RATIO)
        lv = langevin_step!(model, last_id, mesh7, d_lambda_proxy, psi5,
                            direction, plane_scores; phase_weight = phase_weight)
        return (mode = :integrate, revived = revived, emitted = 0,
                acre = nothing, resonance_ratio = round(resonance_ratio; digits = 4),
                langevin = lv)

    elseif resonance_ratio >= MCD_RESONANCE_TANGENTIAL
        emitted = mcd_emit_along_direction!(model, last_id, direction, d_lambda_proxy;
                                            mesh_field = mesh_field)
        acre = acre_emerge!(model; min_resonance = MCD_ACRE_LOW_RESONANCE)
        lv = langevin_step!(model, last_id, mesh7, d_lambda_proxy, psi5,
                            direction, plane_scores; phase_weight = phase_weight)
        return (mode = :emit_tangential, revived = 0, emitted = emitted,
                acre = acre, resonance_ratio = round(resonance_ratio; digits = 4),
                langevin = lv)

    else
        # :leave_in_band — Langevin step in imaginary/Weyl layer
        lv = langevin_step!(model, last_id, mesh7, d_lambda_proxy, psi5,
                            mcd_lift_weyl_psi(psi5), plane_scores;
                            phase_weight = phase_weight)
        return (mode = :leave_in_band, revived = 0, emitted = 0,
                acre = nothing, resonance_ratio = round(resonance_ratio; digits = 4),
                langevin = lv)
    end
end

# ---------------------------------------------------------------------------
# GNN training — Hebbian embedding nudge + Quipu bigram update
# ---------------------------------------------------------------------------
"""
    train_round!(model, corpus; mesh_state7=(0.5,...), max_seconds=30.0,
                 end_state_progress=0.0, phase_weight=1.0,
                 coherence_depth=0, weyl_phase=0.0) -> NamedTuple

One online training pass over `corpus` (a `Vector` of text chunks — the
caller-supplied equivalent of `mesh_slm._corpus_stream`'s output).
Updates, in order:

 1. `mesh_field_8d` and the mean-embedding/MESH wavefunction `overlap`,
    both computed from the **pre-round** graph state;
 2. the effective learning rates
    `η = LR_BASE·(1-progress)·phase_weight·(0.70+0.30·overlap)` and
    `η_q = QUIPU_LR·(1-progress)·phase_weight`;
 3. for each shuffled chunk (until `max_seconds` elapses): tokenize, skip
    if fewer than 2 tokens, else `upsert_token!` every token (nudging its
    embedding `e += η·(mesh_state7 - e)` per axis) and Hebbian-update every
    consecutive Quipu pair `w += η_q·(1 - w)` (cold-start weight `η_q`).

Port of `mesh_slm.train_round` (the wall-clock rate limiter
`_MIN_TRAIN_GAP_S` guarding concurrent daemon callers is intentionally
omitted — irrelevant to a single-threaded batch/offline model call).
"""
function train_round!(
    model::MeshSLMModel, corpus::Vector{<:AbstractString};
    mesh_state7 = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    max_seconds::Real = 30.0, end_state_progress::Real = 0.0,
    phase_weight::Real = 1.0, coherence_depth::Integer = 0, weyl_phase::Real = 0.0,
)
    n_vocab0 = length(model.positions)
    n_quipu0 = length(model.quipu_weight)
    mean_embed = mean_embed7(model)
    progress = clip01(end_state_progress)
    phase_weight = clamp(Float64(phase_weight), 0.1, 2.0)

    mesh_field = mesh_field_8d(
        n_vocab = n_vocab0, n_quipu_edges = n_quipu0,
        mean_embed7 = mean_embed, mesh_state7 = mesh_state7,
        phase_weight = phase_weight, coherence_depth = coherence_depth,
        weyl_phase = weyl_phase, vocab_limit = model.vocab_limit,
    )
    overlap = wavefunction_overlap(mean_embed, mesh_state7)

    eta = LR_BASE * (1.0 - progress) * phase_weight * (0.70 + 0.30 * overlap)
    eta_q = QUIPU_LR * (1.0 - progress) * phase_weight

    if isempty(corpus)
        return (status = "no_corpus", elapsed_ms = 0, end_state_progress = progress)
    end

    shuffled = shuffle(corpus)
    started = time()
    n_tokens = 0
    n_pairs = 0
    loss_acc = 0.0
    n_loss = 0

    for chunk in shuffled
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

    model.rounds += 1
    avg_loss = n_loss > 0 ? loss_acc / n_loss : 0.0
    model.last_loss = avg_loss
    model.last_eta = eta
    model.last_phase_weight = phase_weight
    model.last_wavefunction_overlap = round(overlap, digits = 4)
    model.last_mesh_field_8d = round(mesh_field, digits = 4)

    # ACRE — accumulate multi-axial interaction; attempt emergence
    acre_observe!(model, mesh_state7, mean_embed7(model))

    # MCD — Multiplanar Comparative Decompression (proxy ΔΛ from holographic entropy growth)
    n_quipu_now = length(model.quipu_weight)
    n_vocab_now = length(model.positions)
    H_now = n_quipu_now / max(n_vocab_now + 1, 1)
    d_lambda_proxy = max(0.0, H_now - model.prev_holographic_h)
    model.prev_holographic_h = H_now
    # Neutral Ψ5 proxy: use current mesh state as Weyl stand-in
    psi5_proxy = collect(Float64, mesh_state7)[1:min(5, length(mesh_state7))]
    while length(psi5_proxy) < 5; push!(psi5_proxy, 0.5); end
    anchor_id = isempty(model.freq) ? 0 :
        argmax(Dict(tid => f for (tid, f) in model.freq))
    mcd = if anchor_id > 0
        mcd_dispatch!(model, anchor_id, mesh_state7, d_lambda_proxy, psi5_proxy;
                      phase_weight = phase_weight, mesh_field = mesh_field)
    else
        (mode = :conservative, revived = 0, emitted = 0, acre = nothing, resonance_ratio = 1.0)
    end
    acre = if mcd.mode == :emit_tangential
        get(mcd, :acre, nothing)
    else
        acre_emerge!(model)
    end

    elapsed_ms = round(Int, (time() - started) * 1000)
    return (
        status = "ok", rounds = model.rounds, tokens = n_tokens, pairs = n_pairs,
        avg_loss = round(avg_loss, digits = 6), eta = round(eta, digits = 5),
        end_state_progress = round(progress, digits = 4),
        phase_weight = round(phase_weight, digits = 4),
        wavefunction_overlap = round(overlap, digits = 4),
        mesh_field_8d = round(mesh_field, digits = 4),
        emission_mode = mcd.mode,
        mcd = mcd,
        acre = acre,
        elapsed_ms = elapsed_ms,
    )
end

# ---------------------------------------------------------------------------
# Candidate scoring — the GNN + GLM + MESH composite score
# ---------------------------------------------------------------------------
"""
    score_candidates(model, last_id, mesh7; top_k=32, mesh_field=0.0,
                      specialist=nothing) -> Vector{NamedTuple}

Port of `mesh_slm._score_candidates` — see `mesh_inference_model.jl` for
the line-by-line formula documentation (identical here, reading `model`'s
own fields instead of a separate read-only graph view).
"""
function score_candidates(
    model::MeshSLMModel, last_id::Int, mesh7;
    top_k::Int = 32, mesh_field::Real = 0.0, specialist::Union{Nothing,AbstractString} = nothing,
)
    last_pos = get(model.positions, last_id, (0, 0))
    boost_active = specialist === nothing || specialist == "supply_chain_optimizer"

    outgoing = Tuple{Int,Float64}[]
    for (key, w) in model.quipu_weight
        src, dst = key
        src == last_id && push!(outgoing, (dst, w))
    end
    sort!(outgoing; by = x -> x[2], rev = true)
    outgoing = outgoing[1:min(top_k * 3, length(outgoing))]

    scored = NamedTuple[]
    for (dst_id, w) in outgoing
        haskey(model.positions, dst_id) || continue
        dst_pos = model.positions[dst_id]
        tok = get(model.tokens, dst_id, "")
        prox = proximity(last_pos, dst_pos, model.torus_n)
        effective_dist = max(prox, 0.01)
        warp = clip01(metric_perturbation(w * METRIC_MASS_SCALE, effective_dist) * effective_dist)
        embed = get(model.embeddings, dst_id, zeros(Float64, model.embed_dim))
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
            embed = get(model.embeddings, tid, zeros(Float64, model.embed_dim))
            s = embed_dot(embed, mesh7) + PROX_GAIN * proximity(last_pos, pos, model.torus_n) +
                MESH_FIELD_GAIN * mesh_field
            push!(scored, (token_id = tid, score = s, token = get(model.tokens, tid, ""), pos = pos))
        end
    end

    sort!(scored; by = c -> c.score, rev = true)
    return scored[1:min(top_k, length(scored))]
end

"""
    softmax_pick(cands, temperature; rng=Random.default_rng())

Temperature-scaled softmax sample over scored candidates; greedy arg-max
when `temperature <= 0`. Port of the sampling loop inside
`mesh_slm.generate`.
"""
function softmax_pick(cands, temperature::Real; rng = Random.default_rng())
    isempty(cands) && error("softmax_pick: cands must be non-empty")
    temperature <= 0.0 && return cands[1]
    scores = [c.score / max(temperature, 1.0e-3) for c in cands]
    m = maximum(scores)
    exps = [exp(s - m) for s in scores]
    total = sum(exps)
    total = total > 0 ? total : 1.0
    r = rand(rng) * total
    acc = 0.0
    for (c, e) in zip(cands, exps)
        acc += e
        acc >= r && return c
    end
    return cands[end]
end

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
"""
    generate(model, prompt; max_new_tokens=24, temperature=0.7, seed=nothing,
             specialist=nothing, mesh_state7=(0.5,...), phase_weight=1.0,
             coherence_depth=0, weyl_phase=0.0) -> NamedTuple

Generate a token sequence conditioned on `prompt`. Port of
`mesh_slm.generate` (the domain-specific EOQ "hybrid grounding" hook is
intentionally out of scope — see module docstring).
"""
function generate(
    model::MeshSLMModel, prompt::AbstractString;
    max_new_tokens::Int = 24, temperature::Real = 0.7,
    seed::Union{Nothing,Integer} = nothing,
    specialist::Union{Nothing,AbstractString} = nothing,
    mesh_state7 = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    phase_weight::Real = 1.0, coherence_depth::Integer = 0, weyl_phase::Real = 0.0,
)
    seed === nothing || Random.seed!(seed)
    mesh = apply_specialist_bias(mesh_state7, specialist, model.emergent_biases)
    prompt_tokens = tokenize(prompt)

    n_vocab = length(model.positions)
    if n_vocab < MIN_VOCAB_FOR_GENERATE
        return (text = "", confidence = 0.0, tokens_emitted = 0, vocab_hit_rate = 0.0,
                vocab_size = n_vocab, avg_score = 0.0, mesh_state = mesh,
                mesh_field_8d = 0.0, reason = "vocab_too_small")
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
        best = nothing
        best_freq = -1
        for tid in keys(model.positions)
            f = get(model.freq, tid, 0)
            if f > best_freq
                best = tid
                best_freq = f
            end
        end
        best === nothing && return (text = "", confidence = 0.0, tokens_emitted = 0,
                vocab_hit_rate = 0.0, vocab_size = n_vocab, avg_score = 0.0,
                mesh_state = mesh, mesh_field_8d = 0.0, reason = "empty_vocab")
        anchor = (best, model.positions[best])
    end

    mesh_field = mesh_field_8d(
        n_vocab = n_vocab, n_quipu_edges = length(model.quipu_weight),
        mean_embed7 = mean_embed7(model), mesh_state7 = mesh,
        phase_weight = phase_weight, coherence_depth = coherence_depth,
        weyl_phase = weyl_phase, vocab_limit = model.vocab_limit,
    )

    last_id, last_pos = anchor
    emitted = String[]
    cumulative_score = 0.0
    for _ in 1:max_new_tokens
        cands = score_candidates(model, last_id, mesh; top_k = 12,
                                  mesh_field = mesh_field, specialist = specialist)
        isempty(cands) && break
        pick = softmax_pick(cands, temperature)
        last_id, last_pos = pick.token_id, pick.pos
        push!(emitted, pick.token)
        cumulative_score += pick.score
    end

    avg_score = isempty(emitted) ? 0.0 : cumulative_score / length(emitted)
    confidence = clip01(tanh(0.5 * avg_score) * 0.7 + vocab_hit_rate * 0.3)

    return (
        text = join(emitted, " "),
        confidence = round(confidence, digits = 4),
        tokens_emitted = length(emitted),
        vocab_hit_rate = round(vocab_hit_rate, digits = 4),
        vocab_size = n_vocab,
        avg_score = round(avg_score, digits = 4),
        mesh_state = mesh,
        mesh_field_8d = round(mesh_field, digits = 4),
    )
end

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
"""
    state_summary(model) -> NamedTuple

Diagnostics snapshot: vocab size, Quipu edge count, training rounds, and
the last recorded loss / eta / phase-weight / overlap / mesh-field.
Port of `mesh_slm.state_summary` (trimmed to the fields this Julia model
tracks).
"""
function state_summary(model::MeshSLMModel)
    return (
        model_arch = MODEL_ARCH,
        vocab_size = length(model.positions),
        quipu_edges = length(model.quipu_weight),
        rounds = model.rounds,
        last_loss = model.last_loss,
        last_eta = model.last_eta,
        last_phase_weight = model.last_phase_weight,
        last_wavefunction_overlap = model.last_wavefunction_overlap,
        last_mesh_field_8d = model.last_mesh_field_8d,
        emergent_specialists = sort(collect(keys(model.emergent_biases))),
    )
end

end # module MeshSLMGLMGNN

# ---------------------------------------------------------------------------
# Demo entry point — `julia mesh_slm_glm_gnn_model.jl` trains a few rounds
# on a tiny synthetic supply-chain corpus and generates from a prompt;
# `include(...)` from elsewhere only defines the module.
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    using .MeshSLMGLMGNN

    model = MeshSLMModel()
    corpus = [
        "supply chain optimization eoq formula sqrt demand cost",
        "eoq demand safety stock lead cost hold order",
        "hierarchical eoq empirical bayes shrinkage multi echelon",
        "safety stock protects against demand and lead time variability",
        "holding cost and ordering cost trade off in the eoq model",
    ]

    for _ in 1:20
        train_round!(model, corpus; max_seconds = 5.0)
    end

    println("MeshSLMGLMGNN demo state_summary():")
    println("  ", state_summary(model))
    println()

    result = generate(model, "eoq demand"; max_new_tokens = 8, temperature = 0.7, seed = 42)
    println("MeshSLMGLMGNN demo generate():")
    println("  text       = ", result.text)
    println("  confidence = ", result.confidence)
    println("  mesh_field_8d = ", result.mesh_field_8d)
end
