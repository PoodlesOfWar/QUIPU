"""
    GardShardModel

Julia protocol peer for the Supply Chain Brain **GARD Shard** model.

`gard-shard/v1` turns a canonical UTF-8 JSON value into one or more encrypted,
authenticated shards:

1. Canonical JSON (sorted object keys, compact separators) is zlib-compressed.
2. The compressed bytes are partitioned deterministically.
3. Each shard derives AES/HMAC key material via RFC 5869 HKDF-SHA256 from the
   external `SCBRAIN_GRID_SECRET`, random salt, model-context digest, shard
   index/count, and public `SiCi_SQRT(-1)` domain context.
4. AES-256-CBC + PKCS#7 encrypts the shard.
5. HMAC-SHA256 uses Encrypt-then-MAC over a fixed length-prefixed binary
   preimage. Authentication is checked before any decrypt/decompress path when
   `verify_digest=true` (the default).

The implementation uses mature Julia packages for JSON (`JSON3`), zlib
(`CodecZlib`), AES-CBC/HMAC (`Nettle`), plus Julia standard libraries. It never
serializes a grid secret, derived key, or plaintext into a Hub manifest.
"""
module GardShardModel

using Base64
using CodecZlib
using Dates
using JSON3
using Nettle
using Random
using SHA

export
    GARD_PROTOCOL,
    GARD_PROTOCOL_VERSION,
    GARD_SHARD_DOMAIN,
    GRID_SECRET_ENV,
    VERIFY_DIGEST_ENV,
    GardShardConfig,
    GardShardError,
    GardShardSecretError,
    GardShardVerificationError,
    canonical_json,
    canonical_json_bytes,
    sha256_hex,
    mesh_compaction_summary,
    langevin_sigma_from_weyl,
    verification_enabled_from_env,
    build_model_context,
    encrypt_json,
    decrypt_json,
    verify_envelope,
    write_envelope_json,
    read_envelope_json,
    sign_assignment_proof,
    verify_assignment_proof,
    build_hub_manifest,
    publish_loadopoly_manifest,
    selftest

const GARD_PROTOCOL = "gard-shard/v1"
const GARD_PROTOCOL_VERSION = 1
const GARD_ENVELOPE_SCHEMA = "gard-shard-envelope/v1"
const GARD_MODEL_CONTEXT_SCHEMA = "gard-model-context/v1"
const GARD_HUB_MANIFEST_SCHEMA = "loadopoly-gard-shard-status/v1"
const GARD_ASSIGNMENT_PROOF_KIND = "catalog-assignment-proof/v1"
const GARD_SHARD_DOMAIN = "SiCi_SQRT(-1)"
const GRID_SECRET_ENV = "SCBRAIN_GRID_SECRET"
const VERIFY_DIGEST_ENV = "SCBRAIN_GARD_VERIFY_DIGEST"

const AES_KEY_BYTES = 32
const HMAC_KEY_BYTES = 32
const HKDF_BYTES = AES_KEY_BYTES + HMAC_KEY_BYTES
const SALT_BYTES = 32
const IV_BYTES = 16
const HMAC_BYTES = 32
const MAX_SHARDS = 256
const MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
const MAX_PLAINTEXT_BYTES = 32 * 1024 * 1024

# Corrected read/write compression reference values from mesh_compression_model.jl
# and the active Python ueqgm_engine implementation. These are duplicated here
# intentionally: GardShardModel remains self-contained and can be validated on a
# Julia host without relying on the repository's untracked model prototypes.
const THE_WELL_TB_REF = 15.0
const THE_WELL_N_DATASETS = 16
const THE_WELL_SPATIAL_DIMS = 4
const MESH_TORUS_N = 64
const MESH_VOCAB_LIMIT = MESH_TORUS_N * MESH_TORUS_N
const MESH_EMBED_DIMS = 7
const MESH_BYTES_PER_FLOAT = 8
const MESH_VOCAB_ROW_BYTES = 100
const MESH_QUIPU_ROW_BYTES = 32
const MESH_QUIPU_NODE_ROW_BYTES = 80
const MESH_META_BYTES = 1_024
const MESH_BRAIN_KV_WEYL_BYTES = 50
const MESH_BRAIN_KV_REMNANT_BYTES = 10
const MESH_BRAIN_KV_RUNTIME_BYTES = 3_072
const LANGEVIN_SIGMA_BASE = 0.05

"""Base exception for malformed or unusable GARD material."""
struct GardShardError <: Exception
    message::String
end
Base.showerror(io::IO, err::GardShardError) = print(io, err.message)

"""Raised when a GARD operation requires an unavailable grid secret."""
struct GardShardSecretError <: Exception
    message::String
end
Base.showerror(io::IO, err::GardShardSecretError) = print(io, err.message)

"""Raised when authenticated GARD data is invalid or has been tampered with."""
struct GardShardVerificationError <: Exception
    message::String
end
Base.showerror(io::IO, err::GardShardVerificationError) = print(io, err.message)

"""Non-secret GARD protocol settings; verification is fail-closed by default."""
Base.@kwdef struct GardShardConfig
    shard_count::Int = 1
    compression_level::Int = 6
    verify_digest::Bool = true
    secret_env::String = GRID_SECRET_ENV
    domain::String = GARD_SHARD_DOMAIN
    max_compressed_bytes::Int = MAX_COMPRESSED_BYTES
    max_plaintext_bytes::Int = MAX_PLAINTEXT_BYTES
end

function _validate_config(cfg::GardShardConfig)::GardShardConfig
    1 <= cfg.shard_count <= MAX_SHARDS ||
        throw(GardShardError("shard_count must be in [1, $(MAX_SHARDS)]"))
    -1 <= cfg.compression_level <= 9 ||
        throw(GardShardError("compression_level must be in [-1, 9]"))
    !isempty(strip(cfg.secret_env)) ||
        throw(GardShardError("secret_env must name an environment variable"))
    !isempty(strip(cfg.domain)) || throw(GardShardError("domain must be non-empty"))
    cfg.max_compressed_bytes > 0 && cfg.max_plaintext_bytes > 0 ||
        throw(GardShardError("maximum payload sizes must be positive"))
    return cfg
end

_cfg(config::Nothing) = _validate_config(GardShardConfig())
_cfg(config::GardShardConfig) = _validate_config(config)

"""Return whether the explicit environment policy leaves verification enabled."""
function verification_enabled_from_env(default::Bool = true)::Bool
    raw = get(ENV, VERIFY_DIGEST_ENV, nothing)
    raw === nothing && return default
    return !(lowercase(strip(String(raw))) in ("0", "false", "no", "off"))
end

_utc_now() = Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z"

# ---------------------------------------------------------------------------
# Self-contained compression reference (mirrors corrected mesh_compression_model.jl)
# ---------------------------------------------------------------------------
_clip01(value::Real) = clamp(Float64(value), 0.0, 1.0)

function _hawking_information_remnant_score(
    n_datasets::Integer, n_spatial_dims::Integer, total_tb::Real,
)::Float64
    # Unitary Page curve with island prescription (Page 1993; Almheiri et
    # al., RMP 93, 035002 (2021)): score = min(f,1−f) + max(0,2f−1) = f,
    # the evaporated fraction against the Page reference area (The Well
    # 16×4 = 64), with a Bekenstein mass-energy factor for total_tb.
    n_datasets <= 0 && return 0.0
    area = Float64(n_datasets * max(1, n_spatial_dims))
    if total_tb > 0.0
        area *= 1.0 + (log1p(Float64(total_tb)) / log1p(THE_WELL_TB_REF))
    end
    page_area = Float64(THE_WELL_N_DATASETS * THE_WELL_SPATIAL_DIMS)
    f = area / (area + page_area)
    return _clip01(min(f, 1.0 - f) + max(0.0, 2.0 * f - 1.0))
end

function _information_compaction_scalar(source_bytes::Real, mesh_bytes::Real)::Float64
    (source_bytes <= 0.0 || mesh_bytes <= 0.0) && return 0.0
    ratio = source_bytes / mesh_bytes
    maximum_ratio = source_bytes / max(Float64(MESH_BRAIN_KV_WEYL_BYTES), 1.0)
    maximum_ratio <= 1.0 && return 0.0
    return _clip01(log1p(ratio) / log1p(maximum_ratio))
end

"""
    mesh_compaction_summary(; source_tb=15.0, n_vocab=nothing, n_quipu_edges=nothing)

Return the corrected MESH footprint and Weyl-compaction accounting used by
`ueqgm_engine.mesh_compaction_summary()`. The canonical 15 TB/full-torus result
is `mesh_bytes_total=1_101_884`, `compaction_ratio=14_967_705`, and
`compaction_scalar=0.622934`.
"""
function mesh_compaction_summary(
    ; source_tb::Real = THE_WELL_TB_REF,
    n_vocab::Union{Nothing,Integer} = nothing,
    n_quipu_edges::Union{Nothing,Integer} = nothing,
)::Dict{String,Any}
    source_bytes = Float64(source_tb) * (1024.0 ^ 4)
    vocab_n = something(n_vocab, MESH_VOCAB_LIMIT)
    quipu_n = something(n_quipu_edges, vocab_n)
    vocab_bytes = vocab_n * MESH_VOCAB_ROW_BYTES
    embed_bytes = vocab_n * MESH_EMBED_DIMS * MESH_BYTES_PER_FLOAT
    quipu_bytes = quipu_n * MESH_QUIPU_ROW_BYTES
    quipu_node_bytes = MESH_VOCAB_LIMIT * MESH_QUIPU_NODE_ROW_BYTES
    mesh_bytes_total = vocab_bytes + embed_bytes + quipu_bytes + quipu_node_bytes +
        MESH_META_BYTES + MESH_BRAIN_KV_WEYL_BYTES +
        MESH_BRAIN_KV_REMNANT_BYTES + MESH_BRAIN_KV_RUNTIME_BYTES
    compaction_ratio = mesh_bytes_total > 0 ? source_bytes / mesh_bytes_total : 0.0
    return Dict(
        "source_tb" => Float64(source_tb),
        "source_bytes" => round(Int, source_bytes),
        "vocab_bytes" => vocab_bytes,
        "embed_bytes" => embed_bytes,
        "quipu_bytes" => quipu_bytes,
        "quipu_node_bytes" => quipu_node_bytes,
        "meta_bytes" => MESH_META_BYTES,
        "brain_kv_weyl_bytes" => MESH_BRAIN_KV_WEYL_BYTES,
        "brain_kv_remnant_bytes" => MESH_BRAIN_KV_REMNANT_BYTES,
        "brain_kv_runtime_bytes" => MESH_BRAIN_KV_RUNTIME_BYTES,
        "mesh_bytes_total" => mesh_bytes_total,
        "compaction_ratio" => round(Int, compaction_ratio),
        "compaction_scalar" => round(_information_compaction_scalar(source_bytes, mesh_bytes_total); digits = 6),
        "hawking_remnant_score" => round(
            _hawking_information_remnant_score(
                THE_WELL_N_DATASETS,
                THE_WELL_SPATIAL_DIMS,
                source_tb,
            );
            digits = 6,
        ),
    )
end

"""
    langevin_sigma_from_weyl(psi5_now, psi5_prev; kappa=18.0, sigma_base=0.05)

Reference Langevin diffusion coefficient from the corrected compression model.
Invalid Weyl vectors use the safe base coefficient; the length guard is
explicitly conjunctive so a valid pair never returns prematurely.
"""
function langevin_sigma_from_weyl(
    psi5_now, psi5_prev; kappa::Real = 18.0, sigma_base::Real = LANGEVIN_SIGMA_BASE,
)::Float64
    (length(psi5_now) < 5 || length(psi5_prev) < 5) && return Float64(sigma_base)
    safe4(value) = clamp(Float64(value), 0.0, 1.0 - 1e-9)
    lambda_now = safe4(psi5_now[5]) * kappa / (1.0 - safe4(psi5_now[5]))
    lambda_prev = safe4(psi5_prev[5]) * kappa / (1.0 - safe4(psi5_prev[5]))
    delta_lambda = max(0.0, lambda_now - lambda_prev)
    psi3_now = clamp(Float64(psi5_now[4]), 1e-9, 1.0)
    psi3_prev = clamp(Float64(psi5_prev[4]), 1e-9, 1.0)
    disagreement = abs(psi3_now - psi3_prev) / (psi3_prev + 1e-9)
    token_entropy = clamp(Float64(psi5_now[2]), 0.0, 1.0)
    return _clip01(Float64(sigma_base) * (1.0 + disagreement) * token_entropy *
                   (1.0 + 0.1 * delta_lambda))
end

# ---------------------------------------------------------------------------
# Canonical JSON + wire primitives
# ---------------------------------------------------------------------------
"""Convert JSON3 decoded values to ordinary Dict/Vector values recursively."""
function _native_json(value)
    if value isa JSON3.Object || value isa AbstractDict
        out = Dict{String,Any}()
        for (key, child) in pairs(value)
            out[String(key)] = _native_json(child)
        end
        return out
    elseif value isa JSON3.Array || value isa AbstractVector || value isa Tuple
        return [_native_json(child) for child in value]
    elseif value isa Symbol
        return String(value)
    end
    return value
end

function _json_scalar(value)::String
    try
        return String(JSON3.write(value))
    catch err
        throw(GardShardError("value is not JSON serializable: $(sprint(showerror, err))"))
    end
end

function _write_canonical(io::IO, value)::Nothing
    value = _native_json(value)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa AbstractString
        print(io, _json_scalar(String(value)))
    elseif value isa Integer
        print(io, value)
    elseif value isa AbstractFloat
        isfinite(value) || throw(GardShardError("canonical JSON rejects non-finite floats"))
        print(io, _json_scalar(Float64(value)))
    elseif value isa AbstractDict
        entries = Tuple{String,Any}[]
        for (key, child) in pairs(value)
            (key isa AbstractString || key isa Symbol) ||
                throw(GardShardError("canonical JSON object keys must be strings"))
            push!(entries, (String(key), child))
        end
        sort!(entries; by = first)
        print(io, "{")
        for (index, (key, child)) in enumerate(entries)
            index > 1 && print(io, ",")
            print(io, _json_scalar(key))
            print(io, ":")
            _write_canonical(io, child)
        end
        print(io, "}")
    elseif value isa NamedTuple
        _write_canonical(io, Dict(String(key) => getfield(value, key) for key in keys(value)))
    elseif value isa AbstractVector || value isa Tuple
        print(io, "[")
        for (index, child) in enumerate(value)
            index > 1 && print(io, ",")
            _write_canonical(io, child)
        end
        print(io, "]")
    else
        throw(GardShardError("value is not canonical JSON serializable"))
    end
    return nothing
end

"""Encode a JSON-compatible value as stable, compact, sorted-key UTF-8 bytes."""
function canonical_json_bytes(value)::Vector{UInt8}
    io = IOBuffer()
    _write_canonical(io, value)
    return take!(io)
end

"""Return canonical UTF-8 JSON text."""
canonical_json(value)::String = String(canonical_json_bytes(value))

"""Return a lower-case SHA-256 hexadecimal digest."""
sha256_hex(value::AbstractVector{UInt8})::String = bytes2hex(sha256(value))

function _hex_digest(value, field::String)::Vector{UInt8}
    text = try
        String(value)
    catch
        throw(GardShardVerificationError("invalid $(field)"))
    end
    if length(text) != 64 || text != lowercase(text) ||
       any(ch -> !(isletter(ch) || isdigit(ch)) || (isletter(ch) && !(ch in 'a':'f')), text)
        throw(GardShardVerificationError("invalid $(field)"))
    end
    try
        return hex2bytes(text)
    catch
        throw(GardShardVerificationError("invalid $(field)"))
    end
end

_b64_encode(value::AbstractVector{UInt8}) = base64encode(value)

function _b64_decode(value, field::String)::Vector{UInt8}
    value isa AbstractString || throw(GardShardVerificationError("invalid $(field)"))
    try
        return base64decode(String(value))
    catch
        throw(GardShardVerificationError("invalid $(field)"))
    end
end

function _u32(value::Integer)::Vector{UInt8}
    0 <= value <= typemax(UInt32) ||
        throw(GardShardError("integer does not fit the GARD uint32 wire field"))
    n = UInt64(value)
    return UInt8[
        (n >> 24) & 0xff,
        (n >> 16) & 0xff,
        (n >> 8) & 0xff,
        n & 0xff,
    ]
end

"""Build an unambiguous, length-prefixed binary record shared with Python."""
function _binary_record(fields::AbstractVector{UInt8}...)::Vector{UInt8}
    out = Vector{UInt8}(codeunits("GARD\0"))
    for field in fields
        append!(out, _u32(length(field)))
        append!(out, field)
    end
    return out
end

function _constant_time_equal(left::AbstractVector{UInt8}, right::AbstractVector{UInt8})::Bool
    length(left) == length(right) || return false
    difference = UInt8(0)
    @inbounds for index in eachindex(left, right)
        difference |= left[index] ⊻ right[index]
    end
    return iszero(difference)
end

function _secret_bytes(secret, cfg::GardShardConfig)::Vector{UInt8}
    value = secret === nothing ? get(ENV, cfg.secret_env, "") : secret
    if value isa AbstractString
        isempty(strip(String(value))) &&
            throw(GardShardSecretError("$(cfg.secret_env) is required for GARD shard cryptography"))
        return Vector{UInt8}(codeunits(String(value)))
    elseif value isa AbstractVector{UInt8}
        isempty(value) &&
            throw(GardShardSecretError("$(cfg.secret_env) is required for GARD shard cryptography"))
        return Vector{UInt8}(value)
    end
    throw(GardShardSecretError("$(cfg.secret_env) is required for GARD shard cryptography"))
end

_hmac_sha256(key::AbstractVector{UInt8}, data::AbstractVector{UInt8}) =
    Nettle.digest("sha256", Vector{UInt8}(key), Vector{UInt8}(data))

"""RFC 5869 HKDF-SHA256, using Nettle HMAC-SHA256 for extract and expand."""
function _hkdf_sha256(
    ikm::AbstractVector{UInt8}, salt::AbstractVector{UInt8},
    info::AbstractVector{UInt8}, output_length::Integer,
)::Vector{UInt8}
    0 < output_length <= 255 * HMAC_BYTES ||
        throw(GardShardError("invalid HKDF output length"))
    prk = _hmac_sha256(salt, ikm)
    output = UInt8[]
    previous = UInt8[]
    counter = 1
    while length(output) < output_length
        previous = _hmac_sha256(prk, vcat(previous, info, UInt8[counter]))
        append!(output, previous)
        counter += 1
    end
    return output[1:output_length]
end

function _derive_shard_keys(
    secret::AbstractVector{UInt8}; salt::AbstractVector{UInt8},
    model_digest::AbstractVector{UInt8}, shard_index::Integer,
    shard_count::Integer, config::GardShardConfig,
)::Tuple{Vector{UInt8},Vector{UInt8}}
    length(salt) == SALT_BYTES || throw(GardShardVerificationError("invalid shard salt length"))
    info = _binary_record(
        Vector{UInt8}(codeunits("gard-shard-hkdf/v1")),
        Vector{UInt8}(codeunits(GARD_PROTOCOL)),
        Vector{UInt8}(codeunits(config.domain)),
        Vector{UInt8}(model_digest),
        _u32(shard_index),
        _u32(shard_count),
    )
    material = _hkdf_sha256(secret, salt, info, HKDF_BYTES)
    return material[1:AES_KEY_BYTES], material[AES_KEY_BYTES + 1:end]
end

function _mac_preimage(
    ; model_digest::AbstractVector{UInt8}, salt::AbstractVector{UInt8},
    shard_index::Integer, shard_count::Integer, iv::AbstractVector{UInt8},
    ciphertext::AbstractVector{UInt8}, config::GardShardConfig,
    compression_level::Integer = config.compression_level,
)::Vector{UInt8}
    return _binary_record(
        Vector{UInt8}(codeunits("gard-shard-mac/v1")),
        Vector{UInt8}(codeunits(GARD_PROTOCOL)),
        _u32(GARD_PROTOCOL_VERSION),
        Vector{UInt8}(codeunits(config.domain)),
        _u32(compression_level + 1),
        Vector{UInt8}(model_digest),
        Vector{UInt8}(salt),
        _u32(shard_index),
        _u32(shard_count),
        Vector{UInt8}(iv),
        Vector{UInt8}(ciphertext),
    )
end

function _encrypt_cbc(plaintext::AbstractVector{UInt8}, key::AbstractVector{UInt8}, iv::AbstractVector{UInt8})
    length(key) == AES_KEY_BYTES && length(iv) == IV_BYTES ||
        throw(GardShardError("invalid AES-256-CBC key material"))
    padded = Nettle.add_padding_PKCS5(Vector{UInt8}(plaintext), IV_BYTES)
    return Nettle.encrypt("AES256", :CBC, Vector{UInt8}(iv), Vector{UInt8}(key), padded)
end

function _decrypt_cbc(ciphertext::AbstractVector{UInt8}, key::AbstractVector{UInt8}, iv::AbstractVector{UInt8})
    length(key) == AES_KEY_BYTES && length(iv) == IV_BYTES ||
        throw(GardShardVerificationError("invalid AES-256-CBC key material"))
    try
        padded = Nettle.decrypt("AES256", :CBC, Vector{UInt8}(iv), Vector{UInt8}(key), Vector{UInt8}(ciphertext))
        return Nettle.trim_padding_PKCS5(padded)
    catch err
        throw(GardShardVerificationError("GARD ciphertext could not be decrypted: $(sprint(showerror, err))"))
    end
end

function _zlib_compress(data::AbstractVector{UInt8}, level::Integer)::Vector{UInt8}
    buffer = IOBuffer()
    stream = ZlibCompressorStream(buffer; level = level)
    try
        write(stream, data)
    finally
        close(stream)
    end
    return take!(buffer)
end

function _zlib_decompress_bounded(data::AbstractVector{UInt8}, maximum::Integer)::Vector{UInt8}
    source = IOBuffer(data)
    stream = ZlibDecompressorStream(source)
    output = IOBuffer()
    buffer = Vector{UInt8}(undef, 8192)
    total = 0
    try
        while !eof(stream)
            count = readbytes!(stream, buffer)
            total += count
            total <= maximum || throw(GardShardVerificationError("GARD decompressed payload exceeds configured limit"))
            count > 0 && write(output, view(buffer, 1:count))
        end
    catch err
        err isa GardShardVerificationError && rethrow()
        throw(GardShardVerificationError("GARD compressed payload is invalid: $(sprint(showerror, err))"))
    finally
        close(stream)
    end
    return take!(output)
end

function _split_bytes(payload::AbstractVector{UInt8}, count::Integer)::Vector{Vector{UInt8}}
    n = max(1, min(Int(count), max(1, length(payload))))
    base, remainder = divrem(length(payload), n)
    chunks = Vector{Vector{UInt8}}()
    cursor = 1
    for index in 1:n
        width = base + (index <= remainder ? 1 : 0)
        push!(chunks, Vector{UInt8}(payload[cursor:cursor + width - 1]))
        cursor += width
    end
    return chunks
end

function _fixed_material(values, index::Integer, size::Integer, label::String)::Vector{UInt8}
    raw = values === nothing ? rand(RandomDevice(), UInt8, size) : Vector{UInt8}(values[index])
    length(raw) == size || throw(GardShardError("test $(label) at shard $(index) must be $(size) bytes"))
    return raw
end

function _as_dict(value, message::String)::Dict{String,Any}
    native = _native_json(value)
    native isa AbstractDict || throw(GardShardVerificationError(message))
    return Dict{String,Any}(String(key) => child for (key, child) in pairs(native))
end

function _as_int(
    value,
    field::String;
    minimum::Int = typemin(Int),
    maximum::Int = typemax(Int),
)::Int
    (value isa Integer && !(value isa Bool)) ||
        throw(GardShardVerificationError("invalid $(field)"))
    parsed = Int(value)
    minimum <= parsed <= maximum ||
        throw(GardShardVerificationError("invalid $(field)"))
    return parsed
end

function _safe_vector(value, length_needed::Int; fallback::Float64 = 0.5)::Vector{Float64}
    values = value isa AbstractVector || value isa Tuple ? collect(value) : Any[]
    out = Float64[]
    limit = min(length(values), length_needed)
    for raw in values[1:limit]
        parsed = try
            Float64(raw)
        catch
            fallback
        end
        push!(out, clamp(parsed, 0.0, 1.0))
    end
    append!(out, fill(fallback, max(0, length_needed - length(out))))
    return round.(out; digits = 6)
end

# ---------------------------------------------------------------------------
# Model context + envelope encoding
# ---------------------------------------------------------------------------
"""
    build_model_context(; mesh_state7=nothing, weyl_psi5=nothing, scm_state=nothing,
                         slm_state=nothing, config=nothing) -> Dict

Build the secret-free MESH-SLM-SCM-GLM-Quipu-GNN model identity authenticated
by each GARD envelope. It is public metadata, never replacement key material.
"""
function build_model_context(
    ; mesh_state7 = nothing, weyl_psi5 = nothing, scm_state = nothing,
    slm_state = nothing, config::Union{Nothing,GardShardConfig} = nothing,
)::Dict{String,Any}
    cfg = _cfg(config)
    state = slm_state === nothing ? Dict{String,Any}() : _as_dict(slm_state, "invalid SLM state")
    scm = scm_state === nothing ? Dict{String,Any}() : _as_dict(scm_state, "invalid SCM state")
    mesh = _safe_vector(mesh_state7 === nothing ? get(state, "mesh_state", nothing) : mesh_state7, 7)
    weyl = _safe_vector(weyl_psi5, 5)
    previous_weyl = _safe_vector(get(scm, "previous_weyl_psi5", weyl), 5)
    vocab_size = Int(get(state, "vocab_size", 0))
    quipu_edges = Int(get(state, "quipu_edges", 0))
    compaction = mesh_compaction_summary(
        n_vocab = vocab_size > 0 ? vocab_size : nothing,
        n_quipu_edges = quipu_edges > 0 ? quipu_edges : nothing,
    )
    return Dict(
        "schema" => GARD_MODEL_CONTEXT_SCHEMA,
        "architecture" => "MESH-SLM-SCM-GLM-Quipu-GNN",
        "mesh_state7" => mesh,
        "weyl_psi5" => weyl,
        "scm" => Dict(
            "phase_weight" => get(scm, "phase_weight", get(state, "last_phase_weight", 1.0)),
            "coherence_depth" => get(scm, "coherence_depth", 0),
            "phi" => get(scm, "phi", 0.0),
            "last_emission" => get(scm, "last_emission", nothing),
            "langevin_sigma_reference" => round(
                langevin_sigma_from_weyl(weyl, previous_weyl);
                digits = 6,
            ),
        ),
        "models" => Dict(
            "slm" => Dict("id" => "mesh-slm/torus-quipu-7d", "architecture" => "MESH-SLM-GLM-GNN"),
            "glm" => Dict("id" => "mesh-glm/7d-alignment", "architecture" => "MESH-GLM"),
            "gnn" => Dict("id" => "quipu-gnn/hebbian", "architecture" => "Quipu-GNN"),
            "scm" => Dict("id" => "saturation-cycle-model", "architecture" => "SCM"),
        ),
        "quipu_gnn" => Dict(
            "vocab_size" => vocab_size,
            "vocab_capacity" => Int(get(state, "vocab_capacity", 4096)),
            "edge_count" => quipu_edges,
            "rounds" => Int(get(state, "rounds", 0)),
            "compaction" => compaction,
        ),
        "gard" => Dict(
            "protocol" => GARD_PROTOCOL,
            "domain" => cfg.domain,
            "verify_digest_default" => cfg.verify_digest,
        ),
    )
end

function _normalise_envelope(envelope, cfg::GardShardConfig)
    raw = _as_dict(envelope, "GARD envelope must be an object")
    get(raw, "schema", nothing) == GARD_ENVELOPE_SCHEMA ||
        throw(GardShardVerificationError("unsupported GARD envelope schema"))
    get(raw, "protocol", nothing) == GARD_PROTOCOL ||
        throw(GardShardVerificationError("unsupported GARD envelope protocol"))
    _as_int(
        get(raw, "protocol_version", nothing),
        "protocol_version";
        minimum = GARD_PROTOCOL_VERSION,
        maximum = GARD_PROTOCOL_VERSION,
    ) == GARD_PROTOCOL_VERSION ||
        throw(GardShardVerificationError("unsupported GARD envelope version"))
    get(raw, "domain", nothing) == cfg.domain ||
        throw(GardShardVerificationError("GARD envelope domain mismatch"))
    get(raw, "encoding", nothing) == "base64" ||
        throw(GardShardVerificationError("unsupported GARD encoding"))

    compression = _as_dict(get(raw, "compression", nothing), "invalid GARD compression configuration")
    get(compression, "algorithm", nothing) == "zlib" ||
        throw(GardShardVerificationError("unsupported GARD compression configuration"))
    compression_level = _as_int(
        get(compression, "level", nothing),
        "compression level";
        minimum = -1,
        maximum = 9,
    )

    encryption = _as_dict(get(raw, "encryption", nothing), "invalid GARD encryption configuration")
    encryption == Dict("algorithm" => "AES-256-CBC", "padding" => "PKCS#7") ||
        throw(GardShardVerificationError("unsupported GARD encryption configuration"))
    authentication = _as_dict(get(raw, "authentication", nothing), "invalid GARD authentication configuration")
    authentication == Dict("algorithm" => "HMAC-SHA256", "mode" => "encrypt-then-mac") ||
        throw(GardShardVerificationError("unsupported GARD authentication configuration"))

    context = _as_dict(get(raw, "model_context", nothing), "missing GARD model context")
    model_digest_text = get(raw, "model_digest", nothing)
    model_digest_text isa AbstractString ||
        throw(GardShardVerificationError("invalid model_digest"))
    model_digest_text = String(model_digest_text)
    model_digest = _hex_digest(model_digest_text, "model_digest")
    _constant_time_equal(sha256(canonical_json_bytes(context)), model_digest) ||
        throw(GardShardVerificationError("GARD model context digest mismatch"))
    _hex_digest(get(raw, "plaintext_sha256", nothing), "plaintext_sha256")
    _hex_digest(get(raw, "compressed_sha256", nothing), "compressed_sha256")

    shard_count = _as_int(
        get(raw, "shard_count", nothing),
        "shard_count";
        minimum = 1,
        maximum = MAX_SHARDS,
    )
    shards = _native_json(get(raw, "shards", nothing))
    1 <= shard_count <= MAX_SHARDS && shards isa AbstractVector && length(shards) == shard_count ||
        throw(GardShardVerificationError("invalid GARD shard collection"))
    compressed_bytes = _as_int(
        get(raw, "compressed_bytes", nothing),
        "compressed_bytes";
        minimum = 0,
        maximum = cfg.max_compressed_bytes,
    )
    plaintext_bytes = _as_int(
        get(raw, "plaintext_bytes", nothing),
        "plaintext_bytes";
        minimum = 0,
        maximum = cfg.max_plaintext_bytes,
    )

    checked = Dict{String,Any}[]
    for expected_index in 0:(shard_count - 1)
        shard = _as_dict(shards[expected_index + 1], "invalid GARD shard")
        index = _as_int(
            get(shard, "index", nothing),
            "GARD shard index";
            minimum = 0,
            maximum = MAX_SHARDS - 1,
        )
        count = _as_int(
            get(shard, "count", nothing),
            "GARD shard count";
            minimum = 1,
            maximum = MAX_SHARDS,
        )
        index == expected_index && count == shard_count ||
            throw(GardShardVerificationError("inconsistent GARD shard ordering"))
        salt = _b64_decode(get(shard, "salt_b64", nothing), "salt_b64")
        iv = _b64_decode(get(shard, "iv_b64", nothing), "iv_b64")
        ciphertext = _b64_decode(get(shard, "ciphertext_b64", nothing), "ciphertext_b64")
        mac = _b64_decode(get(shard, "hmac_sha256_b64", nothing), "hmac_sha256_b64")
        length(salt) == SALT_BYTES && length(iv) == IV_BYTES && length(mac) == HMAC_BYTES ||
            throw(GardShardVerificationError("invalid GARD shard field length"))
        (!isempty(ciphertext) && length(ciphertext) % IV_BYTES == 0) ||
            throw(GardShardVerificationError("invalid GARD ciphertext"))
        sha256_hex(ciphertext) == String(get(shard, "ciphertext_sha256", "")) ||
            throw(GardShardVerificationError("GARD ciphertext digest mismatch"))
        push!(checked, Dict(
            "index" => index,
            "count" => count,
            "salt" => salt,
            "iv" => iv,
            "ciphertext" => ciphertext,
            "mac" => mac,
        ))
    end
    return (
        raw = raw,
        model_digest = model_digest,
        shard_count = shard_count,
        shards = checked,
        compression_level = compression_level,
        compressed_bytes = compressed_bytes,
        plaintext_bytes = plaintext_bytes,
    )
end

"""
    verify_envelope(envelope; secret=nothing, config=nothing) -> Dict

Verify every GARD Encrypt-then-MAC tag in constant time before any CBC decrypt
or zlib decompression. `verify_digest=false` is an explicit opt-out only.
"""
function verify_envelope(
    envelope; secret = nothing, config::Union{Nothing,GardShardConfig} = nothing,
)::Dict{String,Any}
    cfg = _cfg(config)
    checked = _normalise_envelope(envelope, cfg)
    if !cfg.verify_digest
        return Dict(
            "verified" => false,
            "verification_disabled" => true,
            "protocol" => GARD_PROTOCOL,
            "shard_count" => checked.shard_count,
            "model_digest" => checked.raw["model_digest"],
        )
    end

    grid_secret = _secret_bytes(secret, cfg)
    for shard in checked.shards
        _, mac_key = _derive_shard_keys(
            grid_secret;
            salt = shard["salt"], model_digest = checked.model_digest,
            shard_index = shard["index"], shard_count = checked.shard_count,
            config = cfg,
        )
        expected = _hmac_sha256(
            mac_key,
            _mac_preimage(
                model_digest = checked.model_digest, salt = shard["salt"],
                shard_index = shard["index"], shard_count = checked.shard_count,
                iv = shard["iv"], ciphertext = shard["ciphertext"], config = cfg,
                compression_level = checked.compression_level,
            ),
        )
        _constant_time_equal(expected, shard["mac"]) ||
            throw(GardShardVerificationError("GARD envelope authentication failed"))
    end
    return Dict(
        "verified" => true,
        "verification_disabled" => false,
        "protocol" => GARD_PROTOCOL,
        "shard_count" => checked.shard_count,
        "model_digest" => checked.raw["model_digest"],
    )
end

"""
    encrypt_json(payload; secret=nothing, model_context=nothing, config=nothing,
                 fixed_salts=nothing, fixed_ivs=nothing) -> Dict

Canonicalize, zlib-compress, shard, AES-256-CBC encrypt, and HMAC authenticate
a JSON-compatible payload. Fixed salts/IVs exist only for public deterministic
interoperability fixtures; production callers leave both unset.
"""
function encrypt_json(
    payload; secret = nothing, model_context = nothing,
    config::Union{Nothing,GardShardConfig} = nothing,
    fixed_salts = nothing, fixed_ivs = nothing,
)::Dict{String,Any}
    cfg = _cfg(config)
    grid_secret = _secret_bytes(secret, cfg)
    context = model_context === nothing ? build_model_context(config = cfg) : _as_dict(model_context, "invalid GARD model context")
    plaintext = canonical_json_bytes(payload)
    length(plaintext) <= cfg.max_plaintext_bytes ||
        throw(GardShardError("GARD plaintext exceeds configured limit"))
    compressed = _zlib_compress(plaintext, cfg.compression_level)
    length(compressed) <= cfg.max_compressed_bytes ||
        throw(GardShardError("GARD compressed payload exceeds configured limit"))

    model_digest = sha256(canonical_json_bytes(context))
    chunks = _split_bytes(compressed, cfg.shard_count)
    fixed_salts !== nothing && length(fixed_salts) < length(chunks) &&
        throw(GardShardError("fixed_salts must provide one salt per shard"))
    fixed_ivs !== nothing && length(fixed_ivs) < length(chunks) &&
        throw(GardShardError("fixed_ivs must provide one IV per shard"))

    shards = Dict{String,Any}[]
    for (julia_index, chunk) in enumerate(chunks)
        index = julia_index - 1
        salt = _fixed_material(fixed_salts, julia_index, SALT_BYTES, "salt")
        iv = _fixed_material(fixed_ivs, julia_index, IV_BYTES, "IV")
        encryption_key, mac_key = _derive_shard_keys(
            grid_secret;
            salt = salt, model_digest = model_digest, shard_index = index,
            shard_count = length(chunks), config = cfg,
        )
        ciphertext = _encrypt_cbc(chunk, encryption_key, iv)
        mac = _hmac_sha256(
            mac_key,
            _mac_preimage(
                model_digest = model_digest, salt = salt, shard_index = index,
                shard_count = length(chunks), iv = iv, ciphertext = ciphertext, config = cfg,
                compression_level = cfg.compression_level,
            ),
        )
        push!(shards, Dict(
            "index" => index,
            "count" => length(chunks),
            "salt_b64" => _b64_encode(salt),
            "iv_b64" => _b64_encode(iv),
            "ciphertext_b64" => _b64_encode(ciphertext),
            "ciphertext_sha256" => sha256_hex(ciphertext),
            "hmac_sha256_b64" => _b64_encode(mac),
        ))
    end
    return Dict(
        "schema" => GARD_ENVELOPE_SCHEMA,
        "protocol" => GARD_PROTOCOL,
        "protocol_version" => GARD_PROTOCOL_VERSION,
        "domain" => cfg.domain,
        "encoding" => "base64",
        "compression" => Dict("algorithm" => "zlib", "level" => cfg.compression_level),
        "encryption" => Dict("algorithm" => "AES-256-CBC", "padding" => "PKCS#7"),
        "authentication" => Dict("algorithm" => "HMAC-SHA256", "mode" => "encrypt-then-mac"),
        "model_context" => context,
        "model_digest" => bytes2hex(model_digest),
        "plaintext_sha256" => sha256_hex(plaintext),
        "compressed_sha256" => sha256_hex(compressed),
        "plaintext_bytes" => length(plaintext),
        "compressed_bytes" => length(compressed),
        "shard_count" => length(chunks),
        "shards" => shards,
    )
end

"""Verify, decrypt, decompress, and parse a GARD JSON envelope."""
function decrypt_json(
    envelope; secret = nothing, config::Union{Nothing,GardShardConfig} = nothing,
    expected_model_context = nothing,
)
    cfg = _cfg(config)
    checked = _normalise_envelope(envelope, cfg)
    if expected_model_context !== nothing
        expected = sha256(canonical_json_bytes(_as_dict(expected_model_context, "invalid expected model context")))
        _constant_time_equal(expected, checked.model_digest) ||
            throw(GardShardVerificationError("GARD model context does not match the envelope"))
    end

    # Must finish all MAC checks before CBC decryption or zlib decompression.
    verify_envelope(checked.raw; secret = secret, config = cfg)
    grid_secret = _secret_bytes(secret, cfg)
    recovered = UInt8[]
    for shard in checked.shards
        encryption_key, _ = _derive_shard_keys(
            grid_secret;
            salt = shard["salt"], model_digest = checked.model_digest,
            shard_index = shard["index"], shard_count = checked.shard_count,
            config = cfg,
        )
        append!(recovered, _decrypt_cbc(shard["ciphertext"], encryption_key, shard["iv"]))
    end
    sha256_hex(recovered) == checked.raw["compressed_sha256"] ||
        throw(GardShardVerificationError("GARD compressed payload digest mismatch"))
    length(recovered) == checked.compressed_bytes ||
        throw(GardShardVerificationError("GARD compressed payload length mismatch"))
    plaintext = _zlib_decompress_bounded(recovered, cfg.max_plaintext_bytes)
    sha256_hex(plaintext) == checked.raw["plaintext_sha256"] ||
        throw(GardShardVerificationError("GARD plaintext digest mismatch"))
    length(plaintext) == checked.plaintext_bytes ||
        throw(GardShardVerificationError("GARD plaintext payload length mismatch"))
    try
        return _native_json(JSON3.read(String(plaintext)))
    catch err
        throw(GardShardVerificationError("GARD plaintext is not valid UTF-8 JSON: $(sprint(showerror, err))"))
    end
end

# ---------------------------------------------------------------------------
# Canonical signed repository assignments
# ---------------------------------------------------------------------------
function _assignment_proof_key(secret::AbstractVector{UInt8}, assignment_digest::AbstractVector{UInt8}, cfg::GardShardConfig)
    info = _binary_record(
        Vector{UInt8}(codeunits("gard-assignment-proof-hkdf/v1")),
        Vector{UInt8}(codeunits(GARD_PROTOCOL)),
        Vector{UInt8}(codeunits(cfg.domain)),
        Vector{UInt8}(assignment_digest),
    )
    return _hkdf_sha256(secret, assignment_digest, info, HMAC_KEY_BYTES)
end

function _assignment_proof_preimage(assignment_digest::AbstractVector{UInt8}, cfg::GardShardConfig)
    return _binary_record(
        Vector{UInt8}(codeunits("gard-assignment-proof-mac/v1")),
        Vector{UInt8}(codeunits(GARD_PROTOCOL)),
        Vector{UInt8}(codeunits(GARD_ASSIGNMENT_PROOF_KIND)),
        Vector{UInt8}(codeunits(cfg.domain)),
        Vector{UInt8}(assignment_digest),
    )
end

"""Create a compact GARD HMAC proof for a versioned canonical assignment object."""
function sign_assignment_proof(
    assignment; secret = nothing, config::Union{Nothing,GardShardConfig} = nothing,
)::Dict{String,Any}
    cfg = _cfg(config)
    normalized = _as_dict(assignment, "GARD assignment must be an object")
    canonical = canonical_json_bytes(normalized)
    digest = sha256(canonical)
    key = _assignment_proof_key(_secret_bytes(secret, cfg), digest, cfg)
    proof = _hmac_sha256(key, _assignment_proof_preimage(digest, cfg))
    return Dict(
        "protocol" => GARD_PROTOCOL,
        "kind" => GARD_ASSIGNMENT_PROOF_KIND,
        "domain" => cfg.domain,
        "assignment" => _native_json(JSON3.read(String(canonical))),
        "assignment_sha256" => bytes2hex(digest),
        "proof_b64" => _b64_encode(proof),
    )
end

"""Fail closed for absent, malformed, or unauthenticated GARD assignment proofs."""
function verify_assignment_proof(
    proof; secret = nothing, config::Union{Nothing,GardShardConfig} = nothing,
    verify_digest::Union{Nothing,Bool} = nothing,
)::Bool
    cfg = _cfg(config)
    required = verify_digest === nothing ? cfg.verify_digest : verify_digest
    !required && return true
    try
        parsed = _as_dict(proof, "invalid GARD assignment proof")
        get(parsed, "protocol", nothing) == GARD_PROTOCOL || return false
        get(parsed, "kind", nothing) == GARD_ASSIGNMENT_PROOF_KIND || return false
        get(parsed, "domain", nothing) == cfg.domain || return false
        assignment = _as_dict(get(parsed, "assignment", nothing), "invalid GARD assignment")
        digest = sha256(canonical_json_bytes(assignment))
        bytes2hex(digest) == get(parsed, "assignment_sha256", nothing) || return false
        supplied = _b64_decode(get(parsed, "proof_b64", nothing), "proof_b64")
        length(supplied) == HMAC_BYTES || return false
        key = _assignment_proof_key(_secret_bytes(secret, cfg), digest, cfg)
        expected = _hmac_sha256(key, _assignment_proof_preimage(digest, cfg))
        return _constant_time_equal(expected, supplied)
    catch
        return false
    end
end

# ---------------------------------------------------------------------------
# Secret-free public manifests
# ---------------------------------------------------------------------------
"""Write a readable JSON envelope or public status manifest to disk."""
function write_envelope_json(value, path::AbstractString)::String
    target = abspath(expanduser(path))
    mkpath(dirname(target))
    open(target, "w") do io
        JSON3.pretty(io, JSON3.write(value))
        println(io)
    end
    return target
end

"""Read an envelope or public manifest as ordinary Julia Dict/Vector values."""
function read_envelope_json(path::AbstractString)
    try
        return _native_json(JSON3.read(read(path, String)))
    catch err
        throw(GardShardError("could not read GARD JSON $(path): $(sprint(showerror, err))"))
    end
end

"""Build the public, secret-free status model consumed by the Loadopoly Hub."""
function build_hub_manifest(
    ; model_context = nothing, config::Union{Nothing,GardShardConfig} = nothing,
)::Dict{String,Any}
    cfg = _cfg(config)
    context = model_context === nothing ? build_model_context(config = cfg) : _as_dict(model_context, "invalid GARD model context")
    return Dict(
        "schema" => GARD_HUB_MANIFEST_SCHEMA,
        "generated_at" => _utc_now(),
        "model" => Dict(
            "name" => "MESH-SLM-SCM-GLM-Quipu-GNN GARD Shard",
            "protocol" => GARD_PROTOCOL,
            "python_module" => "src.quipu.gard_shard_model",
            "julia_module" => "GardShardModel",
            "julia_source" => "pipeline/src/quipu/gard_shard_model.jl",
        ),
        "protocol" => Dict(
            "version" => GARD_PROTOCOL_VERSION,
            "domain" => cfg.domain,
            "canonical_payload" => "UTF-8 JSON, sorted keys, compact separators",
            "compression" => "zlib before sharding",
            "encryption" => "AES-256-CBC with PKCS#7 padding",
            "authentication" => "HMAC-SHA256 Encrypt-then-MAC",
            "kdf" => "HKDF-SHA256 per shard",
        ),
        "security" => Dict(
            "grid_secret_env" => cfg.secret_env,
            "verify_digest_default" => cfg.verify_digest,
            "verify_digest_env" => VERIFY_DIGEST_ENV,
            "secret_configured" => !isempty(strip(get(ENV, cfg.secret_env, ""))),
            "secrets_published" => false,
            "plaintext_published" => false,
        ),
        "hub" => Dict("route" => "/gard-shard-model.json", "default_port" => 9080),
        "model_context_digest" => sha256_hex(canonical_json_bytes(context)),
        "model_context" => context,
        "interoperability_vector" => "gard-shard-v1-fixed-001",
    )
end

"""Publish the status manifest at the local Loadopoly Portal's static route."""
function publish_loadopoly_manifest(
    path::Union{Nothing,AbstractString} = nothing;
    model_context = nothing, config::Union{Nothing,GardShardConfig} = nothing,
)::String
    target = path === nothing ? normpath(joinpath(@__DIR__, "..", "..", "..", "Loadopoly-Portal", "gard-shard-model.json")) : String(path)
    return write_envelope_json(build_hub_manifest(model_context = model_context, config = config), target)
end

# ---------------------------------------------------------------------------
# Public deterministic compatibility fixture
# ---------------------------------------------------------------------------
"""
    selftest() -> Bool

Run the shared `gard-shard-v1-fixed-001` fixture. It checks canonical JSON,
zlib bytes, AES-CBC ciphertext, HMAC tags, recovery, and a tampered MAC.
The expected values are generated by `gard_shard_model.py` using the same
public non-production fixture secret and fixed salt/IV inputs.
"""
function selftest()::Bool
    fixture_secret = Vector{UInt8}(codeunits("gard-v1-public-test-fixture-key"))
    payload = Dict(
        "message" => "GARD shard interoperability",
        "numbers" => [1, 2, 3],
        "unicode" => "MESH Psi",
    )
    context = Dict(
        "architecture" => "MESH-SLM-SCM-GLM-Quipu-GNN",
        "revision" => 1,
    )
    envelope = encrypt_json(
        payload;
        secret = fixture_secret, model_context = context,
        config = GardShardConfig(shard_count = 2),
        fixed_salts = [UInt8.(0:31), UInt8.(32:63)],
        fixed_ivs = [UInt8.(0:15), UInt8.(16:31)],
    )
    @assert envelope["model_digest"] == "5bbba9598191e8e5622ffce857bf7f70af38fa964d7a0eac5be79e7c4f8bc804"
    @assert envelope["plaintext_sha256"] == "3a9d41f63f18d0ef61b631842420eff808fa97830788442c08e364c11f416d7a"
    @assert envelope["compressed_sha256"] == "cfc7db87da6cdd7347f2b1becef891a5f5b01cda5a3b983c896b6482cf2f40cb"
    @assert envelope["shards"][1]["ciphertext_b64"] == "KMkV3Vgqx06peRg9miaOhM8obZR4sirNYiyXOxL/ZUhiqrLQGMlxcHFJhQkFgge7"
    @assert envelope["shards"][1]["hmac_sha256_b64"] == "Sjp9VwTqUNalaistPssRIx9WRveRMxkM21kQ9A6fvyQ="
    @assert envelope["shards"][2]["ciphertext_b64"] == "BGYLiI7tAEDiWQ2c3a46NM9iRP157TYRukKa3cGMd6NOIQ2fyJk9lrGS7z2YE+Ey"
    @assert envelope["shards"][2]["hmac_sha256_b64"] == "vw+2tmiJgMZoXrzwQmgVDEKszrFqBkFhvQq0uh11dHo="
    @assert decrypt_json(envelope; secret = fixture_secret) == payload

    tampered = deepcopy(envelope)
    tampered["shards"][1]["hmac_sha256_b64"] = _b64_encode(fill(UInt8(0), HMAC_BYTES))
    rejected = false
    try
        decrypt_json(tampered; secret = fixture_secret)
    catch err
        rejected = err isa GardShardVerificationError
    end
    @assert rejected
    println("GardShardModel.selftest(): OK — gard-shard/v1 Python fixture parity")
    return true
end

end # module GardShardModel

if abspath(PROGRAM_FILE) == @__FILE__
    using .GardShardModel

    GardShardModel.selftest()
    println(GardShardModel.publish_loadopoly_manifest())
end