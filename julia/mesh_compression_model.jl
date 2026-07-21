"""
    MeshCompressionModel

Julia port of the **read/write compression accounting** layer of the
Supply Chain Brain MESH-SLM-GLM-GNN
(`pipeline/src/quipu/ueqgm_engine.py` + `pipeline/src/quipu/quipu_minimal.py`).

This is a *model-output* prototype for future (e.g. native/accelerated)
implementation — it is not wired into the running Python brain. Constants
and formulas mirror the Python source exactly so results are numerically
comparable; every function's Python origin is noted in its docstring.

Two independent, already-implemented compression directions are modeled:

  * **WRITE compression** — how the raw learning corpus (bytes) is
    distilled, cycle over cycle, into the MESH's own on-disk footprint
    (`mesh_slm_vocab` + `_embed` + `_quipu` + `_quipu_node` + `_meta` +
    `brain_kv`), and ultimately into the irreducible 5-float
    Newman-Penrose Weyl scalar tensor (`Ψ0..Ψ4`, ≈50 bytes) persisted at
    `brain_kv["learnings:weyl_tensor"]`. See [`mesh_compaction_summary`](@ref)
    (port of `ueqgm_engine.mesh_compaction_summary`). [`pack_weyl`](@ref) /
    [`unpack_weyl`](@ref) additionally provide the 20-byte v2.2
    float-dimension `brain_kv` record (5 x Float32 LE) shared byte-for-byte
    with the ContosoHub JS plane.

  * **READ compression** — how a single inference call avoids reading the
    *full* Quipu token/edge graph by first selecting the smallest connected
    "local-minima" subgraph that still satisfies a GLOBAL-MAX
    sufficiency/confidence objective. See [`minimal_context_compression`](@ref)
    (port of the compression/resource block inside
    `quipu_minimal.minimal_infer`).

[`mesh_read_write_compression`](@ref) bundles both into one model-output
record — the top-level "proc" for future implementation.
"""
module MeshCompressionModel

using Base64: base64encode

export
    clip01,
    holographic_entropy,
    hawking_information_remnant_score,
    weyl_scalar_tensor,
    pack_weyl,
    unpack_weyl,
    information_compaction_scalar,
    MeshByteBudget,
    mesh_compaction_summary,
    resource_norm,
    MinimalContextCompression,
    minimal_context_compression,
    ReadWriteCompressionModel,
    mesh_read_write_compression,
    emission_mode_from_weyl,
    langevin_sigma_from_weyl,
    selftest

# ---------------------------------------------------------------------------
# Shared constants — mirror pipeline/src/quipu/ueqgm_engine.py exactly
# ---------------------------------------------------------------------------
const THE_WELL_TB_REF        = 15.0     # reference corpus size (TB): Polymathic AI "The Well"
const THE_WELL_N_DATASETS    = 16       # canonical 16 PDE simulation datasets in The Well
const THE_WELL_SPATIAL_DIMS  = 4        # 3 spatial + 1 temporal dims per dataset

const MESH_TORUS_N           = 64
const MESH_VOCAB_LIMIT       = MESH_TORUS_N * MESH_TORUS_N   # 4 096 token ceiling
const MESH_EMBED_DIMS        = 7
const MESH_BYTES_PER_FLOAT   = 8

const MESH_VOCAB_ROW_BYTES       = 100    # token + (i,j) + freq + 2 timestamps
const MESH_QUIPU_ROW_BYTES       = 32     # src + dst + weight + samples
const MESH_QUIPU_NODE_ROW_BYTES  = 80     # phase/gain fields + timestamp + 2 labels
const MESH_META_BYTES            = 1_024  # mesh_slm_meta: ~9 KV pairs

const MESH_BRAIN_KV_WEYL_BYTES     = 50    # Weyl tensor 5-float JSON
const MESH_BRAIN_KV_WEYL_PACKED_BYTES = 20 # v2.2 float-dimension KV record: 5 x Float32 LE
const MESH_BRAIN_KV_REMNANT_BYTES  = 10    # remnant score float string
const MESH_BRAIN_KV_RUNTIME_BYTES  = 3_072 # adaptive runtime state dict

# quipu_minimal.py read-side resource ceilings (asset_resource_mesh._asset_capacity)
const NORM_CORES       = 16.0
const NORM_RAM_GB      = 64.0
const NORM_VRAM_GB     = 24.0
const NORM_STORAGE_GB  = 1024.0

# ---------------------------------------------------------------------------
# clip01 — shared numeric guard
# ---------------------------------------------------------------------------
"""
    clip01(x) -> Float64

Clamp `x` to the closed interval `[0.0, 1.0]`. Non-finite input (`NaN`,
`Inf`, or anything that fails to convert) maps to `0.0`, mirroring the
Python brain's defensive `ueqgm_engine._clip01` / `quipu_minimal._finite01`
helpers.
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
# WRITE compression — corpus → MESH footprint → Weyl tensor
# ---------------------------------------------------------------------------
"""
    holographic_entropy(n_edges, n_nodes) -> Float64

Bekenstein-Hawking-inspired boundary/bulk entropy `S = n_edges / (n_nodes + 1)`.
Always finite and non-negative; returns `n_edges` when `n_nodes == 0`.
Port of `ueqgm_engine.holographic_entropy`.
"""
holographic_entropy(n_edges::Real, n_nodes::Real) = n_edges / (n_nodes + 1)

"""
    hawking_information_remnant_score(n_datasets, n_spatial_dims, total_tb=0.0) -> Float64

Bekenstein-Hawking-inspired information-remnant score for a dataset
collection, in `[0, 1]`.

`area = n_datasets * max(1, n_spatial_dims)`, `i_remnant = area / (area + 1)`.
When `total_tb > 0` a log-normalised Bekenstein-bound mass term (against the
15 TB "The Well" reference corpus) is blended in 70/30:

`score = 0.70 * i_remnant + 0.30 * min(1, log1p(total_tb) / log1p(THE_WELL_TB_REF))`

Port of `ueqgm_engine.hawking_information_remnant_score`.
"""
function hawking_information_remnant_score(
    n_datasets::Integer, n_spatial_dims::Integer, total_tb::Real = 0.0,
)::Float64
    n_datasets <= 0 && return 0.0
    n_dims = max(1, n_spatial_dims)
    area = float(n_datasets * n_dims)
    i_remnant = area / (area + 1.0)
    score = if total_tb > 0.0
        log_mass = log1p(total_tb) / log1p(THE_WELL_TB_REF)
        0.70 * i_remnant + 0.30 * min(1.0, log_mass)
    else
        i_remnant
    end
    return clip01(score)
end

"""
    weyl_scalar_tensor(signal_flux, topic_entropy, corpus_volume, mesh_alignment, remnant_score)
        -> NTuple{5,Float64}

Compress one learning-corpus cycle into the 5 Newman-Penrose Weyl scalars
`(Ψ0, Ψ1, Ψ2, Ψ3, Ψ4)`, each clipped to `[0, 1]`:

  * `Ψ0` = `clip01(signal_flux)`                    — ingoing signal flux
  * `Ψ1` = `clip01(topic_entropy)`                   — ingoing topic diversity
  * `Ψ2` = `clip01(ρ / (ρ + 1))`, `ρ = max(0, corpus_volume)` — Coulomb bulk mass
  * `Ψ3` = `clip01(mesh_alignment)`                  — outgoing MESH alignment
  * `Ψ4` = `clip01(remnant_score)`                   — Hawking remnant radiation

Serializing this 5-tuple as JSON is the entire persisted "write" of a
research cycle (≈50 bytes vs. kilobytes/terabytes of source corpus).
Port of `ueqgm_engine.weyl_scalar_tensor`.
"""
function weyl_scalar_tensor(
    signal_flux::Real, topic_entropy::Real, corpus_volume::Real,
    mesh_alignment::Real, remnant_score::Real,
)::NTuple{5,Float64}
    rho = max(0.0, float(corpus_volume))
    psi2_bulk = rho / (rho + 1.0)
    return (
        clip01(signal_flux), clip01(topic_entropy), clip01(psi2_bulk),
        clip01(mesh_alignment), clip01(remnant_score),
    )
end

"""
    pack_weyl(psi5) -> Vector{UInt8}

Serialize a 5-scalar Weyl tuple to the **float-dimension KV record**: 20
bytes, 5 x Float32, explicit little-endian (endian-safe on any host). This
is the canonical machine form shared byte-for-byte with the ContosoHub v2.2
JS plane (`packWeyl`); `_meta.weyl` JSON in data files remains the readable
boundary. Round-trip error vs Float64 inputs is <= Float32 eps (~6e-8 rel).
"""
function pack_weyl(psi5)::Vector{UInt8}
    length(psi5) >= 5 || throw(ArgumentError("need 5 Weyl scalars"))
    out = Vector{UInt8}(undef, MESH_BRAIN_KV_WEYL_PACKED_BYTES)
    for i in 1:5
        u = reinterpret(UInt32, Float32(psi5[i]))
        o = 4 * (i - 1)
        out[o + 1] =  u        % UInt8
        out[o + 2] = (u >> 8)  % UInt8
        out[o + 3] = (u >> 16) % UInt8
        out[o + 4] = (u >> 24) % UInt8
    end
    return out
end

"""
    unpack_weyl(bytes) -> NTuple{5,Float64}

Inverse of [`pack_weyl`](@ref): read 5 x Float32 LE from the 20-byte KV
record back into a Float64 5-tuple.
"""
function unpack_weyl(bytes::AbstractVector{UInt8})::NTuple{5,Float64}
    length(bytes) >= MESH_BRAIN_KV_WEYL_PACKED_BYTES ||
        throw(ArgumentError("need >= 20 bytes (5 x Float32 LE)"))
    return ntuple(5) do i
        o = 4 * (i - 1)
        u = UInt32(bytes[o + 1])       | UInt32(bytes[o + 2]) << 8 |
            UInt32(bytes[o + 3]) << 16 | UInt32(bytes[o + 4]) << 24
        Float64(reinterpret(Float32, u))
    end
end

"""
    information_compaction_scalar(source_bytes, mesh_bytes) -> Float64

Log-normalised scalar `κ ∈ [0, 1]` measuring how efficiently the MESH
compacts a raw source of `source_bytes` into `mesh_bytes`:

`κ = clip01(log1p(R) / log1p(R_max))`, `R = source_bytes / mesh_bytes`,
`R_max = source_bytes / MESH_BRAIN_KV_WEYL_BYTES` (the 50-byte Weyl-tensor
floor). `κ = 1.0` means the corpus is distilled all the way to its
irreducible Weyl-tensor boundary encoding; `κ = 0.0` means no compression.
Port of `ueqgm_engine.information_compaction_scalar`.
"""
function information_compaction_scalar(source_bytes::Real, mesh_bytes::Real)::Float64
    (source_bytes <= 0.0 || mesh_bytes <= 0.0) && return 0.0
    ratio = source_bytes / mesh_bytes
    max_ratio = source_bytes / max(Float64(MESH_BRAIN_KV_WEYL_BYTES), 1.0)
    max_ratio <= 1.0 && return 0.0
    return clip01(log1p(ratio) / log1p(max_ratio))
end

"""
    MeshByteBudget

Per-component byte footprint of the full MESH-SLM-GLM-GNN persistence layer
for one [`mesh_compaction_summary`](@ref) call — mirrors `STRUCTURAL_MAP` in
`mesh_slm.py` (tables `mesh_slm_vocab` / `_embed` / `_quipu` / `_quipu_node`
/ `_meta`) plus the three `brain_kv` scalars (Weyl tensor, remnant score,
adaptive runtime).
"""
struct MeshByteBudget
    vocab_bytes::Int
    embed_bytes::Int
    quipu_bytes::Int
    quipu_node_bytes::Int
    meta_bytes::Int
    brain_kv_weyl_bytes::Int
    brain_kv_remnant_bytes::Int
    brain_kv_runtime_bytes::Int
end

_total_bytes(b::MeshByteBudget) = b.vocab_bytes + b.embed_bytes + b.quipu_bytes +
    b.quipu_node_bytes + b.meta_bytes + b.brain_kv_weyl_bytes +
    b.brain_kv_remnant_bytes + b.brain_kv_runtime_bytes

"""
    mesh_compaction_summary(; source_tb=THE_WELL_TB_REF, n_vocab=nothing, n_quipu_edges=nothing)
        -> NamedTuple

**WRITE-side compression proc.** Computes the full in-memory/on-disk
footprint of the MESH (all SQLite tables + `brain_kv` persistence) and
returns the compaction ratio/scalar against a raw corpus of `source_tb`
terabytes. Direct port of `ueqgm_engine.mesh_compaction_summary`.

`n_vocab` / `n_quipu_edges` default to the worst-case full 4 096-token torus
(`MESH_VOCAB_LIMIT`) when omitted, matching the Python default.

Canonical invocation — The Well (15 TB, 16 datasets, 4-D):

```julia
julia> mesh_compaction_summary()
(source_tb = 15.0, source_bytes = 16492674416640, ..., mesh_bytes_total = 1101884,
 hawking_remnant_score = 0.989231, compaction_ratio = 14967705, compaction_scalar = 0.622934)

The historical Python docstring cited `14_967_032` and `≈0.618`; those values
are not reproducible from the byte constants above. The active Python
`ueqgm_engine.mesh_compaction_summary()` implementation returns the values
shown here, which this Julia reference model uses as its regression baseline.
```
"""
function mesh_compaction_summary(;
    source_tb::Real = THE_WELL_TB_REF,
    n_vocab::Union{Nothing,Integer} = nothing,
    n_quipu_edges::Union{Nothing,Integer} = nothing,
)
    source_bytes = source_tb * (1024.0^4)   # TB → bytes (binary)

    vocab_n = something(n_vocab, MESH_VOCAB_LIMIT)
    quipu_n = something(n_quipu_edges, vocab_n)

    budget = MeshByteBudget(
        vocab_n * MESH_VOCAB_ROW_BYTES,
        vocab_n * MESH_EMBED_DIMS * MESH_BYTES_PER_FLOAT,
        quipu_n * MESH_QUIPU_ROW_BYTES,
        MESH_VOCAB_LIMIT * MESH_QUIPU_NODE_ROW_BYTES,   # always full torus
        MESH_META_BYTES,
        MESH_BRAIN_KV_WEYL_BYTES,
        MESH_BRAIN_KV_REMNANT_BYTES,
        MESH_BRAIN_KV_RUNTIME_BYTES,
    )
    mesh_bytes_total = _total_bytes(budget)

    compaction_ratio  = mesh_bytes_total > 0 ? source_bytes / mesh_bytes_total : 0.0
    compaction_scalar = information_compaction_scalar(source_bytes, Float64(mesh_bytes_total))
    remnant_score = hawking_information_remnant_score(
        THE_WELL_N_DATASETS, THE_WELL_SPATIAL_DIMS, source_tb,
    )

    return (
        source_tb               = Float64(source_tb),
        source_bytes             = round(Int, source_bytes),
        vocab_bytes              = budget.vocab_bytes,
        embed_bytes              = budget.embed_bytes,
        quipu_bytes              = budget.quipu_bytes,
        quipu_node_bytes         = budget.quipu_node_bytes,
        meta_bytes               = budget.meta_bytes,
        brain_kv_weyl_bytes      = budget.brain_kv_weyl_bytes,
        brain_kv_remnant_bytes   = budget.brain_kv_remnant_bytes,
        brain_kv_runtime_bytes   = budget.brain_kv_runtime_bytes,
        mesh_bytes_total         = mesh_bytes_total,
        mesh_kb_total            = round(mesh_bytes_total / 1024, digits = 2),
        mesh_mb_total            = round(mesh_bytes_total / 1024^2, digits = 4),
        hawking_remnant_score    = round(remnant_score, digits = 6),
        compaction_ratio         = round(Int, compaction_ratio),
        compaction_scalar        = round(compaction_scalar, digits = 6),
    )
end

# ---------------------------------------------------------------------------
# READ compression — full Quipu graph → local-minima inference subgraph
# ---------------------------------------------------------------------------
"""
    resource_norm(n_tokens, n_edges) -> Float64

Normalised mechanical/electrical requirement of a (sub)graph in `[0, 1]`,
against the same CPU/RAM/VRAM/storage ceilings used by
`asset_resource_mesh._asset_capacity` (16 cores / 64 GB RAM / 24 GB VRAM /
1024 GB storage). Monotonically increasing in graph size.
Port of `quipu_minimal._resource_norm`.
"""
function resource_norm(n_tokens::Integer, n_edges::Integer)::Float64
    bytes_est = n_tokens * (MESH_VOCAB_ROW_BYTES + MESH_EMBED_DIMS * MESH_BYTES_PER_FLOAT) +
        n_edges * MESH_QUIPU_ROW_BYTES
    storage_gb = bytes_est / 1.0e9
    ram_gb = 3.0 * storage_gb     # working set ≈ 3× on-disk
    cores = n_edges / 2048.0      # candidate-scoring CPU pressure
    vram_gb = 0.0                 # no GPU requirement
    return clip01((cores / NORM_CORES + ram_gb / NORM_RAM_GB +
                    vram_gb / NORM_VRAM_GB + storage_gb / NORM_STORAGE_GB) / 4.0)
end

"""
    MinimalContextCompression

**READ-side compression model output.** Bundles the byte-level
`compression_ratio` and normalised `resource_reduction` achieved by
answering a query from the selected local-minima subgraph
(`n_minimal_*`) instead of the full Quipu graph (`n_full_*`).
Mirrors the diagnostics block returned by `quipu_minimal.minimal_infer`.
"""
struct MinimalContextCompression
    full_bytes::Int
    minimal_bytes::Int
    compression_ratio::Float64
    full_resource_norm::Float64
    minimal_resource_norm::Float64
    resource_reduction::Float64
end

"""
    minimal_context_compression(n_full_vocab, n_full_edges, n_minimal_nodes, n_minimal_edges)
        -> MinimalContextCompression

Compute the **READ-side compression proc**: how much smaller (bytes) and
cheaper (normalised CPU/RAM/VRAM/storage) it is to serve one inference call
from the selected local-minima subgraph versus the full persisted Quipu
graph. Direct port of the compression/resource block inside
`quipu_minimal.minimal_infer`.
"""
function minimal_context_compression(
    n_full_vocab::Integer, n_full_edges::Integer,
    n_minimal_nodes::Integer, n_minimal_edges::Integer,
)::MinimalContextCompression
    full_bytes = n_full_vocab * (MESH_VOCAB_ROW_BYTES + MESH_EMBED_DIMS * MESH_BYTES_PER_FLOAT) +
        n_full_edges * MESH_QUIPU_ROW_BYTES
    minimal_bytes = n_minimal_nodes * (MESH_VOCAB_ROW_BYTES + MESH_EMBED_DIMS * MESH_BYTES_PER_FLOAT) +
        n_minimal_edges * MESH_QUIPU_ROW_BYTES
    compression_ratio = round(full_bytes / max(1, minimal_bytes), digits = 4)

    full_norm = resource_norm(n_full_vocab, n_full_edges)
    minimal_norm = resource_norm(n_minimal_nodes, n_minimal_edges)
    resource_reduction = full_norm > 0 ? clip01(1.0 - minimal_norm / full_norm) : 0.0

    return MinimalContextCompression(
        full_bytes, minimal_bytes, compression_ratio,
        full_norm, minimal_norm, resource_reduction,
    )
end

# ---------------------------------------------------------------------------
# Combined model output — WRITE + READ compression in one proc
# ---------------------------------------------------------------------------
"""
    ReadWriteCompressionModel

Top-level "model output" bundling both compression directions of the
MESH-SLM-GLM-GNN:

  * `write_compaction` — corpus → MESH → Weyl-tensor accounting
    ([`mesh_compaction_summary`](@ref))
  * `read_compression`  — full Quipu graph → minimal inference subgraph
    accounting ([`minimal_context_compression`](@ref))
"""
struct ReadWriteCompressionModel
    write_compaction::NamedTuple
    read_compression::MinimalContextCompression
end

"""
    mesh_read_write_compression(;
        source_tb=THE_WELL_TB_REF, n_vocab=nothing, n_quipu_edges=nothing,
        n_minimal_nodes=24, n_minimal_edges=32,
    ) -> ReadWriteCompressionModel

**The combined read/write compression proc** for the MESH-SLM-GLM-GNN,
returned as a single model-output record for future implementation.

* WRITE side: distills a `source_tb`-terabyte corpus into the MESH's own
  byte footprint (`n_vocab` / `n_quipu_edges` default to the full torus),
  down to the irreducible 5-float Weyl tensor.
* READ side: compresses that same full graph down to a `n_minimal_nodes` /
  `n_minimal_edges` local-minima subgraph sized for one inference call.

# Example

```julia
julia> m = mesh_read_write_compression(source_tb = 15.0, n_minimal_nodes = 24, n_minimal_edges = 40);
julia> m.write_compaction.compaction_ratio    # ≈ 14_967_705×  (corpus → MESH)
julia> m.read_compression.compression_ratio   # e.g. ≈ 34×     (full graph → minimal subgraph)
julia> m.read_compression.resource_reduction  # fraction of CPU/RAM/VRAM/storage avoided
```
"""
function mesh_read_write_compression(;
    source_tb::Real = THE_WELL_TB_REF,
    n_vocab::Union{Nothing,Integer} = nothing,
    n_quipu_edges::Union{Nothing,Integer} = nothing,
    n_minimal_nodes::Integer = 24,
    n_minimal_edges::Integer = 32,
)::ReadWriteCompressionModel
    write_compaction = mesh_compaction_summary(
        source_tb = source_tb, n_vocab = n_vocab, n_quipu_edges = n_quipu_edges,
    )
    full_vocab = something(n_vocab, MESH_VOCAB_LIMIT)
    full_edges = something(n_quipu_edges, full_vocab)
    read_compression = minimal_context_compression(
        full_vocab, full_edges, n_minimal_nodes, n_minimal_edges,
    )
    return ReadWriteCompressionModel(write_compaction, read_compression)
end

# ---------------------------------------------------------------------------
# Self-check — reproduces the canonical "The Well" example documented in
# ueqgm_engine.mesh_compaction_summary's Python docstring, so a numeric
# regression against the source implementation is easy to catch.
# ---------------------------------------------------------------------------
"""
    emission_mode_from_weyl(psi5_now, psi5_prev; kappa=18.0) -> Symbol

Compression-friendly emission mode estimation from two consecutive Weyl Ψ
tensor snapshots — no SCM clock or Quipu graph required.

`Ψ4 = Λ/(Λ+κ)` is the Hawking-remnant/NHPP cumulative-intensity scalar.
Solving for Λ: `Λ = Ψ4·κ/(1−Ψ4)`. The delta is an estimate of the NHPP
increment `ΔΛ` between the two snapshots.  The decision thresholds mirror
the live MCD dispatcher:

| ΔΛ estimate | resonance_proxy | mode |
|---|---|---|
| < 0.50 | — | `:leave_in_band` |
| ≥ 0.50 | Ψ3_now/Ψ3_prev ≥ 1.15 | `:integrate` |
| ≥ 0.50 | ≥ 0.90 | `:emit_tangential` |
| ≥ 0.50 | < 0.90 | `:leave_in_band` |

`Ψ3 = S/S0` (tangential health) serves as a compression-proxy for the
multiplanar resonance ratio: rising tangential health ↔ coherent emergent
structure ready for full integration.

Returns a Symbol: `:integrate`, `:emit_tangential`, or `:leave_in_band`.
Port-analogue of `mcd_dispatch!` for the distilled Weyl-only representation.
"""
function emission_mode_from_weyl(psi5_now, psi5_prev; kappa::Real = 18.0)::Symbol
    (length(psi5_now) < 5 || length(psi5_prev) < 5) &&
        return :leave_in_band

    # Reconstruct Λ from Ψ4 = Λ/(Λ+κ) → Λ = Ψ4·κ/(1−Ψ4)
    safe_psi4(v) = clamp(Float64(v), 0.0, 1.0 - 1e-9)
    lambda_now  = safe_psi4(psi5_now[5])  * kappa / (1.0 - safe_psi4(psi5_now[5]))
    lambda_prev = safe_psi4(psi5_prev[5]) * kappa / (1.0 - safe_psi4(psi5_prev[5]))
    d_lambda = max(0.0, lambda_now - lambda_prev)

    d_lambda < 0.50 && return :leave_in_band

    # Resonance proxy: Ψ3 = S/S0 ratio between snapshots
    psi3_now  = clamp(Float64(psi5_now[4]),  1e-9, 1.0)
    psi3_prev = clamp(Float64(psi5_prev[4]), 1e-9, 1.0)
    resonance_proxy = psi3_now / (psi3_prev + 1e-9)

    if resonance_proxy >= 1.15
        return :integrate
    elseif resonance_proxy >= 0.90
        return :emit_tangential
    else
        return :leave_in_band
    end
end

"""
    langevin_sigma_from_weyl(psi5_now, psi5_prev; kappa=18.0, sigma_base=0.05)
        -> Float64

Estimate the Langevin diffusion coefficient σ from two consecutive Weyl Ψ
tensor snapshots, without requiring access to a Quipu graph or SCM state.
Used by compression/distillation pipelines that store only the Weyl tensor.

Formula mirrors `langevin_compute_sigma` in the live training models:

    σ = σ₀ · (1 + D) · H · (1 + 0.1 · ΔΛ)

where:
- **ΔΛ** is reconstructed from Ψ4 delta (Hawking-remnant scalar):
  `Λ = Ψ4·κ/(1−Ψ4)`, `ΔΛ = max(0, Λ_now − Λ_prev)`
- **D** = |(Ψ3_now − Ψ3_prev)|/(Ψ3_prev + ε)  (tangential health delta as
  specialist disagreement proxy)
- **H** = Ψ1_now  (Ψ1 = topic_entropy = token-entropy proxy, already normalised)
"""
function langevin_sigma_from_weyl(
    psi5_now, psi5_prev;
    kappa::Real = 18.0,
    sigma_base::Real = 0.05,
)::Float64
    length(psi5_now)  < 5 && return sigma_base
    length(psi5_prev) < 5 && return sigma_base

    # ΔΛ from Ψ4 delta
    safe4(v) = clamp(Float64(v), 0.0, 1.0 - 1e-9)
    lambda_now  = safe4(psi5_now[5])  * kappa / (1.0 - safe4(psi5_now[5]))
    lambda_prev = safe4(psi5_prev[5]) * kappa / (1.0 - safe4(psi5_prev[5]))
    d_lambda = max(0.0, lambda_now - lambda_prev)

    # Specialist disagreement proxy: tangential health delta (Ψ3 = S/S0)
    psi3_now  = clamp(Float64(psi5_now[4]),  1e-9, 1.0)
    psi3_prev = clamp(Float64(psi5_prev[4]), 1e-9, 1.0)
    disagreement = abs(psi3_now - psi3_prev) / (psi3_prev + 1e-9)

    # Token entropy proxy: Ψ1 = topic_entropy (already [0,1])
    token_entropy = clamp(Float64(psi5_now[2]), 0.0, 1.0)

    sigma = Float64(sigma_base) * (1.0 + disagreement) * token_entropy *
            (1.0 + 0.1 * d_lambda)
    return clamp(sigma, 0.0, 1.0)
end

"""
    selftest() -> Bool

Asserts this module reproduces the canonical "The Well" (15 TB / 16
datasets / 4-D) example documented in `ueqgm_engine.mesh_compaction_summary`.
Returns `true` and prints a confirmation when every check passes.
"""
function selftest()::Bool
    s = mesh_compaction_summary()
    @assert s.source_bytes == 16_492_674_416_640
    @assert s.vocab_bytes == 409_600
    @assert s.embed_bytes == 229_376
    @assert s.quipu_bytes == 131_072
    @assert s.quipu_node_bytes == 327_680
    @assert s.mesh_bytes_total == 1_101_884
    @assert s.compaction_ratio == 14_967_705
    @assert isapprox(s.hawking_remnant_score, 0.989231; atol = 1e-6)
    @assert isapprox(s.compaction_scalar, 0.622934; atol = 1e-6)

    # v2.2 float-dimension KV record — LE canaries + round-trips + golden b64
    psi_g = (0.0, 0.25, 0.5, 0.75, 1.0)          # exactly representable in Float32
    pk = pack_weyl(psi_g)
    @assert length(pk) == MESH_BRAIN_KV_WEYL_PACKED_BYTES
    @assert pk[5:8]   == UInt8[0x00, 0x00, 0x80, 0x3e]   # 0.25f0 little-endian
    @assert pk[17:20] == UInt8[0x00, 0x00, 0x80, 0x3f]   # 1.00f0 little-endian
    @assert unpack_weyl(pk) == psi_g
    psi_r = (0.123456, 0.606066, 0.289474, 0.987654, 0.646154)
    @assert all(abs.(collect(unpack_weyl(pack_weyl(psi_r))) .- collect(psi_r)) .< 1e-6)
    println("  weyl KV golden b64 = ", base64encode(pk))

    println("MeshCompressionModel.selftest(): OK — matches ueqgm_engine.mesh_compaction_summary() ",
            "(v2.2 packed Weyl KV round-trip verified)")
    return true
end

end # module MeshCompressionModel

# ---------------------------------------------------------------------------
# Demo entry point — `julia mesh_compression_model.jl` prints one model
# output; `include("mesh_compression_model.jl")` from elsewhere only
# defines the module.
# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    using .MeshCompressionModel

    MeshCompressionModel.selftest()
    println()

    m = mesh_read_write_compression(n_minimal_nodes = 24, n_minimal_edges = 40)
    println("WRITE compression (corpus → MESH → Weyl tensor):")
    for (k, v) in pairs(m.write_compaction)
        println("  ", k, " = ", v)
    end
    println()
    println("READ compression (full Quipu graph → minimal inference subgraph):")
    println("  ", m.read_compression)
end
