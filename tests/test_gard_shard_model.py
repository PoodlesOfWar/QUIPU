from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.quipu.gard_shard_model import (
    GARD_PROTOCOL,
    GARD_PROTOCOL_V1,
    GardShardConfig,
    GardShardVerificationError,
    build_hub_manifest,
    build_model_context,
    canonical_json_bytes,
    decrypt_json,
    encrypt_json,
    langevin_sigma_from_weyl,
    publish_loadopoly_manifest,
    read_envelope_json,
    verify_envelope,
    write_envelope_json,
)


_PUBLIC_FIXTURE_SECRET = b"gard-v1-public-test-fixture-key"
_VECTOR_PAYLOAD = {
    "message": "GARD shard interoperability",
    "numbers": [1, 2, 3],
    "unicode": "MESH Psi",
}
_VECTOR_CONTEXT = {
    "architecture": "MESH-SLM-SCM-GLM-Quipu-GNN",
    "revision": 1,
}
_VECTOR_SALTS = [bytes(range(32)), bytes(range(32, 64))]
_VECTOR_NONCES = [bytes(range(12)), bytes(range(12, 24))]
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# A real gard-shard/v1 envelope produced by the pre-GCM implementation, frozen
# here so the legacy read path stays exercised after the v2 migration. Do not
# regenerate this: its whole value is that nothing in the current code produced
# it. Payload/context/secret match the v2 vector above.
_LEGACY_V1_ENVELOPE = {
    "schema": "gard-shard-envelope/v1",
    "protocol": "gard-shard/v1",
    "protocol_version": 1,
    "domain": "SiCi_SQRT(-1)",
    "encoding": "base64",
    "compression": {"algorithm": "zlib", "level": 6},
    "encryption": {"algorithm": "AES-256-CBC", "padding": "PKCS#7"},
    "authentication": {"algorithm": "HMAC-SHA256", "mode": "encrypt-then-mac"},
    "model_context": {"architecture": "MESH-SLM-SCM-GLM-Quipu-GNN", "revision": 1},
    "model_digest": "5bbba9598191e8e5622ffce857bf7f70af38fa964d7a0eac5be79e7c4f8bc804",
    "plaintext_sha256": "3a9d41f63f18d0ef61b631842420eff808fa97830788442c08e364c11f416d7a",
    "compressed_sha256": "cfc7db87da6cdd7347f2b1becef891a5f5b01cda5a3b983c896b6482cf2f40cb",
    "plaintext_bytes": 80,
    "compressed_bytes": 86,
    "shard_count": 2,
    "shards": [
        {
            "index": 0,
            "count": 2,
            "salt_b64": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
            "iv_b64": "AAECAwQFBgcICQoLDA0ODw==",
            "ciphertext_b64": "KMkV3Vgqx06peRg9miaOhM8obZR4sirNYiyXOxL/ZUhiqrLQGMlxcHFJhQkFgge7",
            "ciphertext_sha256": "43180488e24480ebc6ae21dd2db75399af10405eb2c10e11463dfd989f8c0a2c",
            "hmac_sha256_b64": "Sjp9VwTqUNalaistPssRIx9WRveRMxkM21kQ9A6fvyQ=",
        },
        {
            "index": 1,
            "count": 2,
            "salt_b64": "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8=",
            "iv_b64": "EBESExQVFhcYGRobHB0eHw==",
            "ciphertext_b64": "BGYLiI7tAEDiWQ2c3a46NM9iRP157TYRukKa3cGMd6NOIQ2fyJk9lrGS7z2YE+Ey",
            "ciphertext_sha256": "9a679ad6e78414bc74655433655e38b407432d7603b754263d34333478e2ca7f",
            "hmac_sha256_b64": "vw+2tmiJgMZoXrzwQmgVDEKszrFqBkFhvQq0uh11dHo=",
        },
    ],
}


def _vector_envelope():
    return encrypt_json(
        _VECTOR_PAYLOAD,
        secret=_PUBLIC_FIXTURE_SECRET,
        model_context=_VECTOR_CONTEXT,
        config=GardShardConfig(shard_count=2),
        fixed_salts=_VECTOR_SALTS,
        fixed_nonces=_VECTOR_NONCES,
    )


def test_canonical_json_is_stable_utf8_and_sorted():
    assert canonical_json_bytes({"z": "MESH Ψ", "a": [True, None, 2]}) == (
        b'{"a":[true,null,2],"z":"MESH \xce\xa8"}'
    )


def test_build_model_context_is_serializable_and_secret_free():
    context = build_model_context(
        mesh_state7=[0.1] * 7,
        weyl_psi5=[0.2] * 5,
        scm_state={"phase_weight": 1.1, "coherence_depth": 2, "phi": 3.0},
        slm_state={"vocab_size": 12, "quipu_edges": 34, "rounds": 5},
    )

    rendered = canonical_json_bytes(context).decode("utf-8")
    assert context["architecture"] == "MESH-SLM-SCM-GLM-Quipu-GNN"
    assert context["mesh_state7"] == [0.1] * 7
    assert context["weyl_psi5"] == [0.2] * 5
    assert context["quipu_gnn"]["edge_count"] == 34
    assert context["quipu_gnn"]["compaction"]["mesh_bytes_total"] > 0
    assert 0.0 <= context["scm"]["langevin_sigma_reference"] <= 1.0
    assert "SCBRAIN_GRID_SECRET" not in rendered
    assert "secret" not in rendered.lower()


def test_langevin_sigma_reference_handles_invalid_vectors_and_valid_weyl_pairs():
    assert langevin_sigma_from_weyl([0.4] * 4, [0.4] * 5) == 0.05

    sigma = langevin_sigma_from_weyl(
        [0.5, 0.8, 0.7, 0.9, 0.8],
        [0.5, 0.4, 0.7, 0.6, 0.3],
    )
    assert 0.05 < sigma <= 1.0


def test_multishard_round_trip_and_readable_envelope_json(tmp_path):
    payload = {"blob": "replenishment " * 400, "part": "80446-04", "min": 5}
    envelope = encrypt_json(
        payload,
        secret=_PUBLIC_FIXTURE_SECRET,
        model_context=_VECTOR_CONTEXT,
        config=GardShardConfig(shard_count=3),
    )

    assert envelope["protocol"] == GARD_PROTOCOL
    assert envelope["compression"]["algorithm"] == "zlib"
    assert envelope["encryption"]["algorithm"] == "AES-256-GCM"
    assert envelope["authentication"]["mode"] == "aead"
    assert envelope["shard_count"] == 3
    assert decrypt_json(envelope, secret=_PUBLIC_FIXTURE_SECRET) == payload

    path = write_envelope_json(envelope, tmp_path / "payload.gard.json")
    reloaded = read_envelope_json(path)
    assert reloaded == envelope
    assert decrypt_json(reloaded, secret=_PUBLIC_FIXTURE_SECRET) == payload


def test_recovery_accepts_any_valid_self_described_compression_level():
    payload = {"part": "80446-04", "notes": "compression-level coverage " * 80}
    envelope = encrypt_json(
        payload,
        secret=_PUBLIC_FIXTURE_SECRET,
        model_context=_VECTOR_CONTEXT,
        config=GardShardConfig(shard_count=2, compression_level=9),
    )

    assert envelope["compression"]["level"] == 9
    assert decrypt_json(envelope, secret=_PUBLIC_FIXTURE_SECRET) == payload


def test_random_salts_and_nonces_are_unique_between_envelopes():
    """Fresh random salt and nonce per shard.

    This is what keeps AES-GCM safe here: a repeated (key, nonce) pair is
    catastrophic for GCM, and a fresh salt per shard means every shard derives
    a distinct key even if a nonce ever did repeat.
    """
    first = encrypt_json(_VECTOR_PAYLOAD, secret=_PUBLIC_FIXTURE_SECRET, model_context=_VECTOR_CONTEXT)
    second = encrypt_json(_VECTOR_PAYLOAD, secret=_PUBLIC_FIXTURE_SECRET, model_context=_VECTOR_CONTEXT)

    assert first["shards"][0]["salt_b64"] != second["shards"][0]["salt_b64"]
    assert first["shards"][0]["nonce_b64"] != second["shards"][0]["nonce_b64"]
    assert first["shards"][0]["ciphertext_b64"] != second["shards"][0]["ciphertext_b64"]


def _flip_ciphertext_bit(envelope, offset=0):
    """Flip one ciphertext bit AND repair ciphertext_sha256.

    Keeping the envelope self-consistent means the cheap digest precheck cannot
    be what rejects it, so the rejection demonstrably comes from the AES-GCM
    authentication tag itself.
    """
    shard = envelope["shards"][0]
    raw = bytearray(base64.b64decode(shard["ciphertext_b64"]))
    raw[offset] ^= 0x01
    shard["ciphertext_b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    shard["ciphertext_sha256"] = hashlib.sha256(bytes(raw)).hexdigest()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda envelope: envelope["shards"][0].__setitem__("ciphertext_b64", base64.b64encode(b"x" * 32).decode("ascii")),
        lambda envelope: envelope["shards"][0].__setitem__("nonce_b64", base64.b64encode(b"z" * 12).decode("ascii")),
        # Body bit and tag bit, both with the digest repaired.
        lambda envelope: _flip_ciphertext_bit(envelope, 0),
        lambda envelope: _flip_ciphertext_bit(envelope, -1),
        lambda envelope: envelope.__setitem__("model_digest", "0" * 64),
        lambda envelope: envelope.__setitem__("plaintext_sha256", "0" * 64),
        lambda envelope: envelope.__setitem__("compressed_sha256", "0" * 64),
        lambda envelope: envelope.__setitem__("protocol", "gard-shard/v0"),
        lambda envelope: envelope.__setitem__("protocol_version", 0),
        lambda envelope: envelope["compression"].__setitem__("level", 9),
        lambda envelope: envelope.__setitem__("compressed_bytes", envelope["compressed_bytes"] + 1),
        # Reordering shards must fail: the shard index is bound into the AAD.
        lambda envelope: envelope["shards"].reverse(),
    ],
)
def test_tampering_is_rejected_before_or_during_verified_recovery(mutate):
    envelope = deepcopy(_vector_envelope())
    mutate(envelope)

    with pytest.raises(GardShardVerificationError):
        decrypt_json(envelope, secret=_PUBLIC_FIXTURE_SECRET)


def test_wrong_secret_is_rejected():
    envelope = _vector_envelope()
    with pytest.raises(GardShardVerificationError):
        decrypt_json(envelope, secret=b"wrong-grid-secret")


def test_verify_digest_opt_out_cannot_bypass_the_aead_tag():
    """v2 tightens what ``verify_digest=False`` can give away.

    Under v1 that flag disabled the separate HMAC pass, and a tampered envelope
    would still decrypt to its payload. Under v2 authentication happens inside
    AES-GCM on every open, so there is no code path that yields plaintext from
    a tampered shard. The flag now only skips the redundant pre-pass.
    """
    envelope = _vector_envelope()
    disabled = GardShardConfig(verify_digest=False)

    # The opt-out still reports itself, and still round-trips intact data.
    status = verify_envelope(envelope, secret=_PUBLIC_FIXTURE_SECRET, config=disabled)
    assert status["verification_disabled"] is True
    assert decrypt_json(envelope, secret=_PUBLIC_FIXTURE_SECRET, config=disabled) == _VECTOR_PAYLOAD

    # But tampering is rejected regardless of the flag.
    tampered = deepcopy(envelope)
    _flip_ciphertext_bit(tampered)
    with pytest.raises(GardShardVerificationError):
        decrypt_json(tampered, secret=_PUBLIC_FIXTURE_SECRET, config=disabled)


# ── gard-shard/v1 backward compatibility ─────────────────────────────────────

def test_legacy_v1_envelope_still_decrypts():
    """The frozen pre-migration v1 envelope must still round-trip.

    Guards the whole legacy path at once: the v1 branch of the validator, the
    v1 HKDF info (whose bound protocol string must stay pinned to v1), the v1
    MAC preimage, and CBC/PKCS#7 decryption.
    """
    assert decrypt_json(_LEGACY_V1_ENVELOPE, secret=_PUBLIC_FIXTURE_SECRET) == _VECTOR_PAYLOAD


def test_legacy_v1_envelope_reports_its_own_protocol():
    status = verify_envelope(_LEGACY_V1_ENVELOPE, secret=_PUBLIC_FIXTURE_SECRET)
    assert status["verified"] is True
    assert status["protocol"] == GARD_PROTOCOL_V1
    assert status["protocol"] != GARD_PROTOCOL


def test_legacy_v1_tampering_is_still_rejected():
    tampered = deepcopy(_LEGACY_V1_ENVELOPE)
    tampered["shards"][0]["hmac_sha256_b64"] = base64.b64encode(b"x" * 32).decode("ascii")
    with pytest.raises(GardShardVerificationError):
        decrypt_json(tampered, secret=_PUBLIC_FIXTURE_SECRET)

    wrong_secret = pytest.raises(GardShardVerificationError)
    with wrong_secret:
        decrypt_json(_LEGACY_V1_ENVELOPE, secret=b"wrong-grid-secret")


def test_v1_and_v2_derive_unrelated_keys_for_identical_shard_context():
    """Same secret, salt, model digest and index must not share key material.

    The HKDF label and bound protocol string are both version-distinct, so the
    two protocols' ciphertexts for the same plaintext must differ.
    """
    v2 = _vector_envelope()
    assert v2["shards"][0]["salt_b64"] == _LEGACY_V1_ENVELOPE["shards"][0]["salt_b64"]
    assert v2["model_digest"] == _LEGACY_V1_ENVELOPE["model_digest"]
    assert v2["shards"][0]["ciphertext_b64"] != _LEGACY_V1_ENVELOPE["shards"][0]["ciphertext_b64"]


def test_envelope_does_not_publish_secret_or_derived_key_material():
    envelope = _vector_envelope()
    rendered = json.dumps(envelope, sort_keys=True)

    assert _PUBLIC_FIXTURE_SECRET.decode("utf-8") not in rendered
    assert base64.b64encode(_PUBLIC_FIXTURE_SECRET).decode("ascii") not in rendered
    assert "encryption_key" not in rendered
    assert "derived_key" not in rendered


def test_deterministic_public_interoperability_vector():
    envelope = _vector_envelope()

    assert envelope["model_digest"] == "5bbba9598191e8e5622ffce857bf7f70af38fa964d7a0eac5be79e7c4f8bc804"
    assert envelope["plaintext_sha256"] == "3a9d41f63f18d0ef61b631842420eff808fa97830788442c08e364c11f416d7a"
    assert envelope["compressed_sha256"] == "cfc7db87da6cdd7347f2b1becef891a5f5b01cda5a3b983c896b6482cf2f40cb"
    assert envelope["shards"][0]["nonce_b64"] == "AAECAwQFBgcICQoL"
    assert envelope["shards"][0]["ciphertext_b64"] == (
        "ZpwXOZ/ebpcwuJSWM7R/kIw+G0ykXBdwEviR8FdOLWBoahqygkGDYQaPR2ZFePySRc00FPyQEXKgUsA="
    )
    assert envelope["shards"][1]["nonce_b64"] == "DA0ODxAREhMUFRYX"
    assert envelope["shards"][1]["ciphertext_b64"] == (
        "aZTxOKmJs/34MMAAWFxrxBLNHtw3BcwIh9TIPtIgs3jN70fCLHsY/XMzu1ZuQNOisHNL9N4HFY9RwCo="
    )


def test_hub_manifest_is_public_and_publishable(tmp_path, monkeypatch):
    monkeypatch.delenv("SCBRAIN_GRID_SECRET", raising=False)
    path = publish_loadopoly_manifest(tmp_path / "gard-shard-model.json", model_context=_VECTOR_CONTEXT)
    manifest = read_envelope_json(path)
    direct = build_hub_manifest(model_context=_VECTOR_CONTEXT)

    assert manifest["model"]["protocol"] == GARD_PROTOCOL
    assert manifest["security"]["verify_digest_default"] is True
    assert manifest["security"]["secrets_published"] is False
    assert manifest["security"]["secret_configured"] is False
    assert manifest["hub"]["route"] == "/gard-shard-model.json"
    assert direct["model_context_digest"] == manifest["model_context_digest"]
    assert _PUBLIC_FIXTURE_SECRET.decode("utf-8") not in json.dumps(manifest)


def test_committed_hub_assets_and_launcher_wiring_are_consistent():
    portal_manifest = read_envelope_json(
        _WORKSPACE_ROOT / "Loadopoly-Portal" / "gard-shard-model.json"
    )
    repo_root = Path(__file__).resolve().parents[1]
    doc_path = repo_root / "docs" / "GARD_SHARD_MODEL_MANIFEST.json"
    if not doc_path.exists():
        doc_path = _WORKSPACE_ROOT / "pipeline" / "docs" / "GARD_SHARD_MODEL_MANIFEST.json"
    documentation = json.loads(doc_path.read_text(encoding="utf-8"))
    portal = (_WORKSPACE_ROOT / "Loadopoly-Portal" / "index.html").read_text(encoding="utf-8")
    launcher = (_WORKSPACE_ROOT / "Launch-Loadopoly.ps1").read_text(encoding="utf-8")

    assert portal_manifest["model"]["protocol"] == GARD_PROTOCOL
    assert portal_manifest["security"]["secrets_published"] is False
    assert portal_manifest["security"]["plaintext_published"] is False
    assert build_hub_manifest(model_context=portal_manifest["model_context"])[
        "model_context_digest"
    ] == portal_manifest["model_context_digest"]
    assert documentation["wire_protocol"]["name"] == GARD_PROTOCOL
    assert documentation["compression_reference"]["canonical_15tb_full_torus"][
        "compaction_ratio"
    ] == 14967705
    assert "gard-shard-model.json" in portal
    assert "gard-card" in portal
    assert "src.quipu.gard_shard_model publish" in launcher


def test_julia_peer_and_corrected_compression_reference_are_declared():
    repo_root = Path(__file__).resolve().parents[1]
    gard_path = repo_root / "julia" / "gard_shard_model.jl"
    if not gard_path.exists():
        gard_path = _WORKSPACE_ROOT / "pipeline" / "src" / "brain" / "gard_shard_model.jl"
    gard_julia = gard_path.read_text(encoding="utf-8")

    comp_path = repo_root / "julia" / "mesh_compression_model.jl"
    if not comp_path.exists():
        comp_path = _WORKSPACE_ROOT / "pipeline" / "src" / "brain" / "mesh_compression_model.jl"
    compression_julia = comp_path.read_text(encoding="utf-8")

    proj_path = repo_root / "julia" / "Project.toml"
    if not proj_path.exists():
        proj_path = _WORKSPACE_ROOT / "pipeline" / "Project.toml"
    project = proj_path.read_text(encoding="utf-8")

    assert "module GardShardModel" in gard_julia
    assert "using CodecZlib" in gard_julia
    assert "using Nettle" in gard_julia
    assert "function mesh_compaction_summary" in gard_julia
    assert "function langevin_sigma_from_weyl" in gard_julia
    assert "(length(psi5_now) < 5 || length(psi5_prev) < 5) &&" in gard_julia
    assert "mesh_read_write_compression," in compression_julia
    assert "@assert s.compaction_ratio == 14_967_705" in compression_julia
    assert "(length(psi5_now) < 5 || length(psi5_prev) < 5) &&" in compression_julia
    assert "CodecZlib" in project
    assert "JSON3" in project
    assert "Nettle" in project


def test_mesh_compression_julia_declares_packed_weyl_kv_record():
    repo_root = Path(__file__).resolve().parents[1]
    comp_path = repo_root / "julia" / "mesh_compression_model.jl"
    if not comp_path.exists():
        comp_path = _WORKSPACE_ROOT / "pipeline" / "src" / "brain" / "mesh_compression_model.jl"
    compression_julia = comp_path.read_text(encoding="utf-8")

    assert "using Base64: base64encode" in compression_julia
    assert "MESH_BRAIN_KV_WEYL_PACKED_BYTES = 20" in compression_julia
    assert "function pack_weyl(psi5)::Vector{UInt8}" in compression_julia
    assert (
        "function unpack_weyl(bytes::AbstractVector{UInt8})::NTuple{5,Float64}"
        in compression_julia
    )
    assert "pk[5:8]   == UInt8[0x00, 0x00, 0x80, 0x3e]" in compression_julia
    assert "pk[17:20] == UInt8[0x00, 0x00, 0x80, 0x3f]" in compression_julia
    assert "@assert unpack_weyl(pk) == psi_g" in compression_julia
    # New pack/unpack pair is additive; the corrected compaction reference
    # this module already guarantees stays byte-identical.
    assert "@assert s.compaction_ratio == 14_967_705" in compression_julia

# ── GARD rename additive tests ────────────────────────────────────────────────

def test_langevin_sigma_from_gard_alias_matches_weyl():
    """langevin_sigma_from_gard is an alias for langevin_sigma_from_weyl."""
    from src.quipu.gard_shard_model import langevin_sigma_from_gard

    psi5_now = [0.5, 0.8, 0.7, 0.9, 0.8]
    psi5_prev = [0.5, 0.4, 0.7, 0.6, 0.3]

    assert langevin_sigma_from_gard(psi5_now, psi5_prev) == langevin_sigma_from_weyl(
        psi5_now, psi5_prev
    )
    assert langevin_sigma_from_gard is langevin_sigma_from_weyl


def test_build_model_context_gard_state5_equals_weyl_psi5():
    """gard_state5 twin key must equal weyl_psi5 in build_model_context output."""
    weyl = [0.1, 0.2, 0.3, 0.4, 0.5]
    context = build_model_context(
        mesh_state7=[0.0] * 7,
        weyl_psi5=weyl,
    )
    assert context["gard_state5"] == context["weyl_psi5"]
    assert context["weyl_psi5"] == weyl  # original key still present


def test_build_model_context_gard_crb_present_and_populated():
    """build_model_context should include gard_crb in compaction when available."""
    context = build_model_context(
        mesh_state7=[0.0] * 7,
        weyl_psi5=[0.1, 0.2, 0.3, 0.4, 0.5],
        slm_state={"vocab_size": 100, "quipu_edges": 50},
    )
    compaction = context["quipu_gnn"]["compaction"]
    assert "gard_crb" in compaction
    gard_crb = compaction["gard_crb"]
    assert gard_crb.get("sigma") == 0.05
    assert "floor_bytes" in gard_crb
    assert "compaction_scalar_at_crb_floor" in gard_crb


def test_langevin_sigma_from_gard_in_all():
    from src.quipu import gard_shard_model
    assert "langevin_sigma_from_gard" in gard_shard_model.__all__
