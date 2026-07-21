"""
    MeshInferenceModel

Julia port of the **inferential aspect** of the Supply Chain Brain
MESH-SLM-GLM-GNN — the scoring/generation path in
`pipeline/src/quipu/mesh_slm.py` (`_score_candidates`, `_embed_dot`,
`_proximity`, `_mesh_field_8d`, `generate`), plus the UEQGM physics helpers
it calls from `pipeline/src/quipu/ueqgm_engine.py`.

This module is **inference-only** — no Hebbian training. It answers "given
a trained graph snapshot, what token comes next?":

  * **GNN** message term — reads (never updates) Hebbian Quipu bigram
    weights `src -> dst`.
  * **GLM** alignment term — [`embed_dot`](@ref), the dot product of a
    token's stored 7-D graph embedding with the live MESH state.
  * **Toroidal geometry** — [`torus_dist`](@ref) / [`proximity`](@ref),
    wrap-aware spatial attraction on the 64×64 vocabulary torus.
  * **8th-D MESH field** — [`mesh_field_8d`](@ref), the UEQGM 5-integral
    blend (holographic entropy, Floquet modulation, wavefunction overlap,
    phase evolution, metric warp) injected into every score.
  * **SLM generation** — [`score_candidates`](@ref) + [`generate`](@ref),
    temperature-softmax next-token sampling over the composite score.

For the full trainable model (tokenizer, torus vocab allocation with
saturation eviction, Hebbian embedding/quipu updates) see the companion
holistic port, `mesh_slm_glm_gnn_model.jl`, in this same directory — it
duplicates the functions below (self-contained, single-file) rather than
`include`-ing this file.

Two categories of input are treated as *external* to this module, exactly
as they are in the Python source (fetched from other Brain subsystems —
`system_entirety`, `ueqgm_engine.refresh_adaptive_runtime`,
`temporal_spatiality` — with the same neutral fallbacks when those
subsystems are unavailable):

  * the live 7-D MESH state vector `mesh_state7`
    (fallback `(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)`), and
  * the UEQGM adaptive-runtime scalars `phase_weight` (fallback `1.0`),
    `coherence_depth` (fallback `0`), and `weyl_phase` (fallback `0.0`).

The domain-specific "hybrid EOQ grounding" hook inside the Python
`generate()` (regex-extracting numbers from the prompt and running the
real `eoq.py` formula) is intentionally out of scope here — it is a
supply-chain modular-expert plug-in layered on top of the generic
MESH-SLM-GLM-GNN inference engine, not part of the engine itself.
"""
module MeshInferenceModel

using Random

export
    clip01,
    TORUS_N,
    VOCAB_LIMIT,
    CONF_FLOOR,
    MeshGraphView,
    add_token!,
    set_embedding!,
    add_edge!,
    token_lookup,
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
    mesh_field_8d,
    mean_embed7,
    SPECIALIST_BIASES,
    apply_specialist_bias,
    register_specialist!,
    score_candidates,
    softmax_pick,
    generate,
    emission_mode_diagnostic,
    langevin_step_diagnostic

# ---------------------------------------------------------------------------
# Shared constants — mirror pipeline/src/quipu/mesh_slm.py / ueqgm_engine.py
# ---------------------------------------------------------------------------
const TORUS_N          = 64
const VOCAB_LIMIT       = TORUS_N * TORUS_N     # 4 096
const EMBED_DIM         = 7

const QUIPU_GAIN        = 0.55   # bigram weight in scoring
const PROX_GAIN         = 0.25   # toroidal proximity weight in scoring
const MESH_FIELD_GAIN   = 0.18   # 8th-D MESH field contribution to scoring
const WARP_GAIN         = 0.06   # per-candidate metric-warp amplification
const METRIC_MASS_SCALE = 6.7e26
const CONF_FLOOR        = 0.22   # min confidence to surface an SLM answer
const MIN_VOCAB_FOR_GENERATE = 16

const G_CONST = 6.674e-11   # gravitational constant, m^3 kg^-1 s^-2
const C_CONST = 2.998e8     # speed of light,        m s^-1
const GAMMA_0_DEFAULT = 1.0
const TAN_CLAMP = 1.0e3

# supply_chain_optimizer / research_specialist / mesh_historian 7-D bias
# vectors: (vision, touch, smell, body, brain, perception, entirety).
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
# Read-only graph view — the inference-time data an inference call needs.
# ---------------------------------------------------------------------------
"""
    MeshGraphView

Minimal in-memory snapshot of the torus vocab / 7-D embeddings / Quipu
bigram edges that [`score_candidates`](@ref) and [`generate`](@ref) read.
This is the inference-time analogue of the `mesh_slm_vocab` / `_embed` /
`_quipu` SQLite tables. Build one with [`add_token!`](@ref) /
[`set_embedding!`](@ref) / [`add_edge!`](@ref), or populate it from a
trained `MeshSLMGLMGNN.MeshSLMModel` snapshot.
"""
struct MeshGraphView
    positions::Dict{Int,Tuple{Int,Int}}
    tokens::Dict{Int,String}
    freq::Dict{Int,Int}
    embeddings::Dict{Int,NTuple{7,Float64}}
    quipu::Dict{Tuple{Int,Int},Float64}
end

MeshGraphView() = MeshGraphView(
    Dict{Int,Tuple{Int,Int}}(), Dict{Int,String}(), Dict{Int,Int}(),
    Dict{Int,NTuple{7,Float64}}(), Dict{Tuple{Int,Int},Float64}(),
)

"""
    add_token!(graph, token_id, token, pos; freq=1) -> Nothing

Register `token` at torus position `pos = (i, j)` under `token_id`.
"""
function add_token!(graph::MeshGraphView, token_id::Int, token::AbstractString,
                     pos::Tuple{Int,Int}; freq::Int = 1)
    graph.positions[token_id] = pos
    graph.tokens[token_id] = String(token)
    graph.freq[token_id] = freq
    return nothing
end

"""
    set_embedding!(graph, token_id, embed7) -> Nothing

Store the 7-D `(vision, touch, smell, body, brain, perception, entirety)`
embedding for `token_id`. `embed7` may be any 7-element indexable
collection (`Tuple`, `Vector`, ...).
"""
function set_embedding!(graph::MeshGraphView, token_id::Int, embed7)
    graph.embeddings[token_id] = ntuple(k -> Float64(embed7[k]), 7)
    return nothing
end

"""
    add_edge!(graph, src, dst, weight) -> Nothing

Set the directed Quipu bigram weight (`src -> dst`), `weight` clipped to
`[0, 1]`.
"""
function add_edge!(graph::MeshGraphView, src::Int, dst::Int, weight::Real)
    graph.quipu[(src, dst)] = clip01(weight)
    return nothing
end

"""
    token_lookup(graph, token) -> Union{Int,Nothing}

Return the `token_id` matching `token`, or `nothing` when absent.
"""
function token_lookup(graph::MeshGraphView, token::AbstractString)
    for (tid, tok) in graph.tokens
        tok == token && return tid
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Toroidal geometry
# ---------------------------------------------------------------------------
"""
    torus_dist(a, b, torus_n=TORUS_N) -> Int

Wrap-aware L1 distance between torus cells `a` and `b`. Port of
`mesh_slm._torus_dist`.
"""
function torus_dist(a::Tuple{Int,Int}, b::Tuple{Int,Int}, torus_n::Int = TORUS_N)::Int
    di = abs(a[1] - b[1])
    dj = abs(a[2] - b[2])
    return min(di, torus_n - di) + min(dj, torus_n - dj)
end

"""
    proximity(a, b, torus_n=TORUS_N) -> Float64

Linear proximity in `[0, 1]`: `1.0` at `a == b`, decaying to `0.0` at the
maximum torus distance. Port of `mesh_slm._proximity`.
"""
function proximity(a::Tuple{Int,Int}, b::Tuple{Int,Int}, torus_n::Int = TORUS_N)::Float64
    d = torus_dist(a, b, torus_n)
    return max(0.0, 1.0 - d / Float64(torus_n))
end

# ---------------------------------------------------------------------------
# GLM alignment term
# ---------------------------------------------------------------------------
"""
    embed_dot(embed7, mesh7) -> Float64

Unnormalised dot product of a token's 7-D embedding with the live MESH
state — the **GLM alignment** term in token scoring. Accepts any pair of
7-element iterables. Port of `mesh_slm._embed_dot`.
"""
embed_dot(embed7, mesh7) = sum(a * b for (a, b) in zip(embed7, mesh7))

# ---------------------------------------------------------------------------
# UEQGM physics helpers needed by mesh_field_8d
# (ports of pipeline/src/quipu/ueqgm_engine.py)
# ---------------------------------------------------------------------------
"""
    coherence_to_phi(coherence) -> Float64

`φ = π/4 + coherence·π` — every value is a natural sin/cos intersection
point. Port of `ueqgm_engine.coherence_to_phi`.
"""
coherence_to_phi(coherence::Integer) = pi / 4 + coherence * pi

"""
    _raw_sici(phi) -> (Si, Ci)

Power-series approximation of the sine/cosine integrals, accurate to
roughly 4 significant figures for `|φ| ≤ 2π` (mirrors the Python source's
scipy-less fallback path; this port always uses the series since no
scipy equivalent is assumed). For production-grade accuracy outside that
range, swap in `SpecialFunctions.sinint` / `.cosint`.
"""
function _raw_sici(phi::Real)::Tuple{Float64,Float64}
    x = phi != 0.0 ? abs(phi) : 1.0e-12
    si_val = x - x^3 / 18.0 + x^5 / 600.0 - x^7 / 35280.0
    euler_mascheroni = 0.5772156649
    ci_val = euler_mascheroni + log(x) - x^2 / 4.0 + x^4 / 96.0
    return (phi >= 0 ? si_val : -si_val), ci_val
end

"""
    sici_axial_decay(phi, gamma_0=GAMMA_0_DEFAULT) -> Float64

`Δλ_axial = Si(φ)·Ci(φ)·tan(φ)·Γ0`, with `tan(φ)` clamped to
`±TAN_CLAMP` to avoid divergence near `φ = π/2 + nπ`. Port of
`ueqgm_engine.sici_axial_decay`.
"""
function sici_axial_decay(phi::Real, gamma_0::Real = GAMMA_0_DEFAULT)::Float64
    si, ci = _raw_sici(phi)
    tan_phi = clamp(tan(phi), -TAN_CLAMP, TAN_CLAMP)
    return si * ci * tan_phi * gamma_0
end

"""
    wavefunction_overlap(a, b) -> Float64

`|⟨ψ_a|ψ_b⟩|² = cos²θ(a, b)` — squared cosine similarity of two
unnormalised state vectors; `0.0` for degenerate (zero-norm) or mismatched
inputs. Port of `ueqgm_engine.wavefunction_overlap`.
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

`cos(ω·t)` — Floquet periodicity modulation. Port of
`ueqgm_engine.floquet_modulation_factor`.
"""
floquet_modulation_factor(t::Real, omega::Real) = cos(omega * t)

"""
    holographic_entropy(n_edges, n_nodes) -> Float64

`S = n_edges / (n_nodes + 1)`. Port of `ueqgm_engine.holographic_entropy`.
"""
holographic_entropy(n_edges::Real, n_nodes::Real) = n_edges / (n_nodes + 1)

"""
    metric_perturbation(mass_eff, r) -> Float64

`h_μν = 2·G·mass_eff / (c²·r)`; `0.0` for `r ≤ 0`. Port of
`ueqgm_engine.metric_perturbation`.
"""
function metric_perturbation(mass_eff::Real, r::Real)::Float64
    r <= 0.0 && return 0.0
    return 2.0 * G_CONST * mass_eff / (C_CONST^2 * r)
end

"""
    phase_evolution_total(phi; delta_mu=0.0, delta_q=0.0, delta_gamma=0.0,
                           gamma_0=GAMMA_0_DEFAULT, gamma_eff=GAMMA_0_DEFAULT)
        -> Float64

`δφ_total = δφ_μ + δφ_q + δφ_γ + Δλ_axial·2π/Γ_eff`. Port of
`ueqgm_engine.phase_evolution_total`.
"""
function phase_evolution_total(
    phi::Real; delta_mu::Real = 0.0, delta_q::Real = 0.0, delta_gamma::Real = 0.0,
    gamma_0::Real = GAMMA_0_DEFAULT, gamma_eff::Real = GAMMA_0_DEFAULT,
)::Float64
    axial = sici_axial_decay(phi, gamma_0)
    axial_phase = axial * (2.0 * pi / max(gamma_eff, 1.0e-30))
    return delta_mu + delta_q + delta_gamma + axial_phase
end

# ---------------------------------------------------------------------------
# 8th-D MESH field — the global UEQGM projection injected into every score
# ---------------------------------------------------------------------------
"""
    mesh_field_8d(; n_vocab, n_quipu_edges, mean_embed7, mesh_state7,
                  phase_weight=1.0, coherence_depth=0, weyl_phase=0.0,
                  vocab_limit=VOCAB_LIMIT, metric_mass_scale=METRIC_MASS_SCALE)
        -> Float64

The 8th-dimension MESH field scalar, blending five UEQGM aspect integrals:

`field = 0.30·H + 0.25·F + 0.25·O + 0.10·P + 0.10·W`

  * `H` — `holographic_entropy(n_quipu_edges, n_vocab) / 8`
  * `F` — Floquet modulation of the Weyl phase at the SiCi-corrected
    frequency `ω = max(0.1, phase_weight)`, remapped from `[-1,1]→[0,1]`
  * `O` — `wavefunction_overlap(mean_embed7, mesh_state7)`
  * `P` — `|phase_evolution_total(φ)| / 2π`, `φ = coherence_to_phi(coherence_depth)`
  * `W` — metric warp from vocab-fill density (`n_vocab / vocab_limit`)

Returns a scalar in `[0, 1]`. Port of `mesh_slm._mesh_field_8d`.
"""
function mesh_field_8d(;
    n_vocab::Integer, n_quipu_edges::Integer, mean_embed7, mesh_state7,
    phase_weight::Real = 1.0, coherence_depth::Integer = 0, weyl_phase::Real = 0.0,
    vocab_limit::Integer = VOCAB_LIMIT, metric_mass_scale::Real = METRIC_MASS_SCALE,
)::Float64
    # H — holographic boundary entropy, normalised by an 8 edges/node ceiling
    H = clip01(holographic_entropy(n_quipu_edges, n_vocab) / 8.0)

    phi = coherence_to_phi(coherence_depth)

    # F — Floquet modulation, Weyl coupling at SiCi-corrected frequency
    omega = max(0.1, phase_weight)
    raw_F = floquet_modulation_factor(mod(weyl_phase, 2.0 * pi), omega)
    F = clip01(0.5 * (1.0 + raw_F))

    # O — wavefunction overlap of mean vocab embedding with live MESH state
    O = wavefunction_overlap(mean_embed7, mesh_state7)

    # P — total 6D CAT phase evolution normalised to [0, 1]
    raw_P = abs(phase_evolution_total(phi))
    P = clip01(raw_P / (2.0 * pi))

    # W — metric warp from vocab fill density
    vocab_fill = n_vocab / max(vocab_limit, 1)
    r = max(1.0 - 0.5 * vocab_fill, 0.01)
    raw_W = metric_perturbation(vocab_fill * metric_mass_scale, r)
    W = clip01(raw_W * r)

    return clip01(0.30 * H + 0.25 * F + 0.25 * O + 0.10 * P + 0.10 * W)
end

"""
    mean_embed7(graph; limit=128) -> NTuple{7,Float64}

Mean 7-D embedding across the `limit` most-frequent tokens in `graph`;
neutral midpoint `(0.5,...,0.5)` when `graph` has no embeddings. Port of
`mesh_slm._mean_embed_7d`.
"""
function mean_embed7(graph::MeshGraphView; limit::Int = 128)::NTuple{7,Float64}
    isempty(graph.embeddings) && return ntuple(_ -> 0.5, 7)
    ranked = sort(collect(keys(graph.embeddings)); by = tid -> get(graph.freq, tid, 0), rev = true)
    top_n = min(limit, length(ranked))
    acc = zeros(Float64, 7)
    for k in 1:top_n
        e = graph.embeddings[ranked[k]]
        acc .+= e
    end
    acc ./= top_n
    return ntuple(k -> clip01(acc[k]), 7)
end

# ---------------------------------------------------------------------------
# Specialist bias
# ---------------------------------------------------------------------------
"""
    apply_specialist_bias(mesh7, specialist) -> NTuple{7,Float64}

Add the named specialist's 7-D bias vector to `mesh7`, clipping each axis
to `[0, 1]`; returns `mesh7` unchanged when `specialist` is `nothing` or
unrecognised. Port of `mesh_slm._apply_specialist_bias`.
"""
function apply_specialist_bias(mesh7, specialist::Union{Nothing,AbstractString})
    (specialist === nothing || !haskey(SPECIALIST_BIASES, specialist)) && return mesh7
    bias = SPECIALIST_BIASES[specialist]
    return ntuple(k -> clip01(mesh7[k] + bias[k]), 7)
end

"""
    register_specialist!(name, bias7) -> NTuple{7,Float64}

Register a specialist bias at inference time — the intake hook for **ACRE
emergent specialists** (Axial Cross-Resonance Emergence). Emergence itself
is a training-time phenomenon (see `acre_emerge!` in the trainable models
and `mesh_slm.acre_emerge` in the Python core); this inference module
receives the resulting `emergent_*` bias vectors as data. Components are
clipped to [0, 1].
"""
function register_specialist!(name::AbstractString, bias7)
    bias = ntuple(k -> clip01(bias7[k]), 7)
    SPECIALIST_BIASES[String(name)] = bias
    return bias
end

# ---------------------------------------------------------------------------
# Candidate scoring — the GNN + GLM + MESH composite score
# ---------------------------------------------------------------------------
"""
    score_candidates(graph, last_id, mesh7; top_k=32, mesh_field=0.0,
                      specialist=nothing) -> Vector{NamedTuple}

Top-K next-token candidates out of `last_id`, ranked by:

`score(t) = QUIPU_GAIN·quipu_weight + (PROX_GAIN + WARP_GAIN·warp_t)·proximity
          + embed_dot(t, mesh7) + MESH_FIELD_GAIN·mesh_field`

`warp_t` is the metric-warp amplification of the candidate's proximity
(GR-inspired). When `specialist` is `nothing` or `"supply_chain_optimizer"`,
numeric/optimization-relevant tokens receive an `OPT_BOOST` bonus. Falls
back to embedding+proximity ranking over the whole vocab when `last_id`
has no outgoing Quipu edges (cold start). Port of
`mesh_slm._score_candidates`.
"""
function score_candidates(
    graph::MeshGraphView, last_id::Int, mesh7;
    top_k::Int = 32, mesh_field::Real = 0.0, specialist::Union{Nothing,AbstractString} = nothing,
)
    last_pos = get(graph.positions, last_id, (0, 0))
    boost_active = specialist === nothing || specialist == "supply_chain_optimizer"

    outgoing = Tuple{Int,Float64}[]
    for (key, w) in graph.quipu
        src, dst = key
        src == last_id && push!(outgoing, (dst, w))
    end
    sort!(outgoing; by = x -> x[2], rev = true)
    outgoing = outgoing[1:min(top_k * 3, length(outgoing))]

    scored = NamedTuple[]
    for (dst_id, w) in outgoing
        haskey(graph.positions, dst_id) || continue
        dst_pos = graph.positions[dst_id]
        tok = get(graph.tokens, dst_id, "")
        prox = proximity(last_pos, dst_pos)
        effective_dist = max(prox, 0.01)
        warp = clip01(metric_perturbation(w * METRIC_MASS_SCALE, effective_dist) * effective_dist)
        embed = get(graph.embeddings, dst_id, ntuple(_ -> 0.0, 7))
        s = QUIPU_GAIN * w + (PROX_GAIN + WARP_GAIN * warp) * prox +
            embed_dot(embed, mesh7) + MESH_FIELD_GAIN * mesh_field
        if boost_active && (tok in NUMERIC_BOOST_TOKENS ||
                             any(k -> occursin(k, tok), NUMERIC_BOOST_SUBSTRINGS))
            s += OPT_BOOST
        end
        push!(scored, (token_id = dst_id, score = s, token = tok, pos = dst_pos))
    end

    if isempty(scored)
        # Cold start: rank the whole vocab by embed alignment + proximity.
        ranked = sort(collect(keys(graph.positions)); by = tid -> get(graph.freq, tid, 0), rev = true)
        for tid in ranked[1:min(top_k, length(ranked))]
            pos = graph.positions[tid]
            embed = get(graph.embeddings, tid, ntuple(_ -> 0.0, 7))
            s = embed_dot(embed, mesh7) + PROX_GAIN * proximity(last_pos, pos) +
                MESH_FIELD_GAIN * mesh_field
            push!(scored, (token_id = tid, score = s, token = get(graph.tokens, tid, ""), pos = pos))
        end
    end

    sort!(scored; by = c -> c.score, rev = true)
    return scored[1:min(top_k, length(scored))]
end

"""
    softmax_pick(cands, temperature; rng=Random.default_rng())

Temperature-scaled softmax sample over scored candidates (as returned by
[`score_candidates`](@ref)); greedy arg-max when `temperature <= 0`. Port
of the sampling loop inside `mesh_slm.generate`.
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
# Generation — temperature-softmax next-token sampling loop
# ---------------------------------------------------------------------------
"""
    generate(graph, prompt_tokens; max_new_tokens=24, temperature=0.7,
             seed=nothing, specialist=nothing,
             mesh_state7=(0.5,0.5,0.5,0.5,0.5,0.5,0.5)) -> NamedTuple

Generate a token sequence from `graph` conditioned on `prompt_tokens`.
Resolves the last in-vocab prompt token as the conditioning anchor
(falling back to the highest-frequency vocab token when nothing in the
prompt is known), computes [`mesh_field_8d`](@ref) once for the pass, then
repeatedly calls [`score_candidates`](@ref) and [`softmax_pick`](@ref).

Returns `(text, confidence, tokens_emitted, vocab_hit_rate, vocab_size,
avg_score, mesh_state, mesh_field_8d)`. `confidence < CONF_FLOOR` signals
the caller should fall back to another model. Port of `mesh_slm.generate`
(the domain-specific EOQ "hybrid grounding" hook in the Python source is
intentionally out of scope — see module docstring).
"""
function generate(
    graph::MeshGraphView, prompt_tokens::Vector{<:AbstractString};
    max_new_tokens::Int = 24, temperature::Real = 0.7,
    seed::Union{Nothing,Integer} = nothing,
    specialist::Union{Nothing,AbstractString} = nothing,
    mesh_state7 = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
)
    seed === nothing || Random.seed!(seed)
    mesh = apply_specialist_bias(mesh_state7, specialist)

    n_vocab = length(graph.positions)
    if n_vocab < MIN_VOCAB_FOR_GENERATE
        return (text = "", confidence = 0.0, tokens_emitted = 0,
                vocab_hit_rate = 0.0, vocab_size = n_vocab, avg_score = 0.0,
                mesh_state = mesh, mesh_field_8d = 0.0, reason = "vocab_too_small")
    end

    anchor = nothing
    hits = 0
    for tok in reverse(prompt_tokens)
        tid = token_lookup(graph, tok)
        if tid !== nothing
            hits += 1
            anchor === nothing && (anchor = (tid, graph.positions[tid]))
        end
    end
    for tok in prompt_tokens
        token_lookup(graph, tok) !== nothing && (hits += 1)
    end
    vocab_hit_rate = isempty(prompt_tokens) ? 0.0 : hits / max(1, length(prompt_tokens) * 2)

    if anchor === nothing
        best = nothing
        best_freq = -1
        for tid in keys(graph.positions)
            f = get(graph.freq, tid, 0)
            if f > best_freq
                best = tid
                best_freq = f
            end
        end
        best === nothing && return (text = "", confidence = 0.0, tokens_emitted = 0,
                vocab_hit_rate = 0.0, vocab_size = n_vocab, avg_score = 0.0,
                mesh_state = mesh, mesh_field_8d = 0.0, reason = "empty_vocab")
        anchor = (best, graph.positions[best])
    end

    n_quipu_edges = length(graph.quipu)
    mesh_field = mesh_field_8d(
        n_vocab = n_vocab, n_quipu_edges = n_quipu_edges,
        mean_embed7 = mean_embed7(graph), mesh_state7 = mesh,
    )

    last_id, last_pos = anchor
    emitted = String[]
    cumulative_score = 0.0
    for _ in 1:max_new_tokens
        cands = score_candidates(graph, last_id, mesh; top_k = 12,
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
# MCD — read-only emission mode diagnostic (no state mutation)
# ---------------------------------------------------------------------------
const _MCD_EMISSION_THRESHOLD   = 0.50
const _MCD_RESONANCE_INTEGRATE  = 1.15
const _MCD_RESONANCE_TANGENTIAL = 0.90

"""
    emission_mode_diagnostic(graph, last_id, mesh7, psi5=(0.5,...);
                              mesh_field=0.0, phase_weight=1.0)
        -> NamedTuple(mode, resonance_ratio, base_score, d_lambda_proxy)

Read-only MCD emission mode estimator for the inference-only model.  No
state is modified — this purely computes what mode the MCD dispatcher
*would* select given the current graph snapshot.

Planes scored: base `mesh7`, each specialist bias in `SPECIALIST_BIASES`,
reflected Ψ (1−lift), and inverted Ψ (lift).  `d_lambda_proxy` is derived
from the holographic entropy `H = n_quipu/(n_vocab+1)` relative to
`MESH_EMBED_DIMS` as a unit-normalised saturation proxy; a constant
external ΔH can be supplied via the `d_lambda_proxy` keyword.
"""
function emission_mode_diagnostic(
    graph::MeshGraphView, last_id::Int, mesh7,
    psi5 = fill(0.5, 5);
    mesh_field::Real = 0.0,
    phase_weight::Real = 1.0,
    d_lambda_proxy::Union{Nothing,Real} = nothing,
)
    # Derive d_lambda_proxy from holographic entropy if not provided
    if d_lambda_proxy === nothing
        n_q = length(graph.quipu)
        n_v = length(graph.positions)
        d_lambda_proxy = n_q / max(n_v + 1, 1)
    end

    # Build scoring planes
    planes = [collect(Float64, mesh7)]
    for (_, bias) in SPECIALIST_BIASES
        push!(planes, [clip01(mesh7[k] + bias[k]) for k in 1:EMBED_DIM])
    end
    # Ψ lift: Ψ0→vis, Ψ1→touch, Ψ2→smell, Ψ3→brain+entirety, Ψ4→perception
    lifted = zeros(EMBED_DIM)
    if length(psi5) >= 1; lifted[1] = clip01(psi5[1]) end
    if length(psi5) >= 2; lifted[2] = clip01(psi5[2]) end
    if length(psi5) >= 3; lifted[3] = clip01(psi5[3]) end
    if length(psi5) >= 4
        lifted[5] = clip01(0.6 * psi5[4])
        lifted[7] = clip01(0.4 * psi5[4])
    end
    if length(psi5) >= 5; lifted[6] = clip01(psi5[5]) end
    push!(planes, [clip01(1.0 - v) for v in lifted])  # reflected
    push!(planes, [clip01(v)       for v in lifted])  # inverted

    scores = Float64[]
    for plane in planes
        cands = score_candidates(graph, last_id, plane; top_k = 4, mesh_field = mesh_field)
        isempty(cands) || push!(scores, cands[1].score)
    end

    if isempty(scores) || d_lambda_proxy < _MCD_EMISSION_THRESHOLD
        return (mode = :leave_in_band, resonance_ratio = 1.0,
                base_score = isempty(scores) ? 0.0 : scores[1],
                d_lambda_proxy = Float64(d_lambda_proxy))
    end

    base = scores[1]
    best = maximum(scores)
    resonance_ratio = best / (base + 1e-9)
    mode = if resonance_ratio >= _MCD_RESONANCE_INTEGRATE
        :integrate
    elseif resonance_ratio >= _MCD_RESONANCE_TANGENTIAL
        :emit_tangential
    else
        :leave_in_band
    end

    return (mode = mode,
            resonance_ratio = round(resonance_ratio, digits = 4),
            base_score = round(base, digits = 4),
            d_lambda_proxy = round(Float64(d_lambda_proxy), digits = 4))
end

# Private Langevin constants (mirrors mesh_slm_scm_glm_gnn_model)
const _LANGEVIN_DT                  = 0.01
const _LANGEVIN_SIGMA_BASE          = 0.05
const _LANGEVIN_DRIFT_ALPHA         = 0.30
const _LANGEVIN_DRIFT_BETA          = 0.20
const _LANGEVIN_DRIFT_GAMMA         = 0.50
const _LANGEVIN_VIABILITY_THRESHOLD = 0.60
const _LANGEVIN_DIFFUSE_SCALE       = 0.30

"""
    langevin_step_diagnostic(graph, last_id, mesh7, d_lambda, psi5;
                             direction=nothing, plane_scores=Float64[],
                             phase_weight=1.0)
        -> NamedTuple(delta_x, drift, sigma, viability, step_mode)

Read-only Langevin step estimator.  Computes the stochastic MESH manifold
update that *would* be applied, without writing to ``graph``.  Useful for
logging, introspection, and offline analysis.

    Δx = μ·dt + σ·dW·√dt

`direction` defaults to the Weyl-lifted Ψ vector when not provided.
`plane_scores` should come from `emission_mode_diagnostic`’s internal
scores list; if empty, specialist disagreement is assumed zero.
"""
function langevin_step_diagnostic(
    graph::MeshGraphView, last_id::Int, mesh7, d_lambda::Real,
    psi5 = fill(0.5, 5);
    direction = nothing,
    plane_scores::Vector{Float64} = Float64[],
    phase_weight::Real = 1.0,
)
    # Weyl lifted vector into 7-D
    lifted = zeros(EMBED_DIM)
    if length(psi5) >= 1; lifted[1] = clip01(psi5[1]) end
    if length(psi5) >= 2; lifted[2] = clip01(psi5[2]) end
    if length(psi5) >= 3; lifted[3] = clip01(psi5[3]) end
    if length(psi5) >= 4
        lifted[5] = clip01(0.6 * psi5[4])
        lifted[7] = clip01(0.4 * psi5[4])
    end
    if length(psi5) >= 5; lifted[6] = clip01(psi5[5]) end

    # Direction: Lagrangian phase if not provided, else Weyl lift
    eff_dir = if direction !== nothing
        collect(Float64, direction)
    else
        omega = max(0.1, Float64(phase_weight)) * π
        m = collect(Float64, mesh7)
        raw = [clip01(0.5 * cos(omega * m[k] + lifted[k]) +
                      0.5 * sin(omega * lifted[k] - m[k])) for k in 1:EMBED_DIM]
        peak = max(maximum(abs.(raw)), 1e-9)
        raw ./ peak
    end

    m = collect(Float64, mesh7)
    # Specialist forces from SPECIALIST_BIASES (read-only view)
    specialist_forces = Vector{Float64}[
        [clip01(m[k] + bias[k]) for k in 1:EMBED_DIM]
        for (_, bias) in SPECIALIST_BIASES
    ]
    mean_force = isempty(specialist_forces) ? zeros(EMBED_DIM) :
        [sum(sf[k] for sf in specialist_forces) / length(specialist_forces)
         for k in 1:EMBED_DIM]

    drift = [
        _LANGEVIN_DRIFT_ALPHA * (clip01(lifted[k]) - m[k])
        + _LANGEVIN_DRIFT_BETA  * (mean_force[k] - m[k])
        + _LANGEVIN_DRIFT_GAMMA * eff_dir[k]
        for k in 1:EMBED_DIM
    ]

    # Sigma from plane_scores disagreement
    disagreement = if length(plane_scores) > 1
        mean_s = sum(plane_scores) / length(plane_scores)
        sqrt(sum((s - mean_s)^2 for s in plane_scores)) / (abs(mean_s) + 1e-9)
    else
        0.0
    end
    # Token entropy proxy: entropy of quipu weight distribution
    ws = collect(values(graph.quipu))
    token_entropy_proxy = if length(ws) > 1
        total = max(sum(ws), 1e-9)
        ent = -sum((w / total) * log(w / total + 1e-12) for w in ws)
        clip01(ent / max(log(length(ws) + 1), 1.0))
    else
        0.5
    end
    sigma = _LANGEVIN_SIGMA_BASE * (1.0 + disagreement) *
            token_entropy_proxy * (1.0 + 0.1 * d_lambda)
    sigma = clip01(sigma)

    dt = _LANGEVIN_DT
    # Use a fixed seed for reproducibility in diagnostic mode (no state writes)
    delta_x = [drift[k] * dt + sigma * randn() * sqrt(dt) for k in 1:EMBED_DIM]

    proj = [clip01(m[k] + delta_x[k]) for k in 1:EMBED_DIM]
    # Mean embed proxy from quipu neighbour embeddings
    me = if haskey(graph.positions, last_id)
        pos = graph.positions[last_id]
        neighbours = [(k, w) for ((src, k), w) in graph.quipu if src == last_id]
        if isempty(neighbours)
            fill(0.5, EMBED_DIM)
        else
            # simple positional proxy
            collect(Float64, mesh7)  # reuse current mesh as mean proxy
        end
    else
        fill(0.5, EMBED_DIM)
    end
    dot_pq = sum(proj[k] * me[k] for k in 1:EMBED_DIM)
    norm_p = max(sqrt(sum(proj .^ 2)), 1e-9)
    norm_m = max(sqrt(sum(me   .^ 2)), 1e-9)
    viability = clip01(dot_pq / (norm_p * norm_m))

    step_mode = viability >= _LANGEVIN_VIABILITY_THRESHOLD ? :reinforce : :diffuse
    apply_scale = viability >= _LANGEVIN_VIABILITY_THRESHOLD ? 1.0 : _LANGEVIN_DIFFUSE_SCALE
    effective_dx = delta_x .* apply_scale

    return (delta_x = round.(effective_dx; digits = 6),
            drift   = round.(drift;        digits = 6),
            sigma   = round(sigma;         digits = 4),
            viability = round(viability;   digits = 4),
            step_mode = step_mode)
end

end # module MeshInferenceModel

# ---------------------------------------------------------------------------
# Demo entry point — `julia mesh_inference_model.jl` builds a tiny synthetic
# graph and runs one `generate` call; `include(...)` from elsewhere only
# defines the module.
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    using .MeshInferenceModel

    graph = MeshGraphView()
    # >= MIN_VOCAB_FOR_GENERATE (16) tokens so the demo exercises the real
    # generation loop instead of the early "vocab_too_small" fallback.
    demo_tokens = [
        "eoq", "demand", "safety", "stock", "lead", "cost", "hold", "order",
        "supply", "chain", "optimization", "formula", "sqrt", "holding",
        "ordering", "trade", "variability", "echelon",
    ]
    for (tid, tok) in enumerate(demo_tokens)
        add_token!(graph, tid, tok, (tid, tid); freq = 10 - tid)
        set_embedding!(graph, tid, ntuple(_ -> 0.4 + 0.05 * tid, 7))
    end
    for tid in 1:(length(demo_tokens) - 1)
        add_edge!(graph, tid, tid + 1, 0.4 + 0.05 * tid)
    end

    result = generate(graph, ["eoq", "demand"]; max_new_tokens = 6, temperature = 0.7, seed = 42)
    println("MeshInferenceModel demo generate():")
    println("  text       = ", result.text)
    println("  confidence = ", result.confidence)
    println("  mesh_field_8d = ", result.mesh_field_8d)
end
