"""GARD information floor: Fisher / Cramér–Rao bounds for the lossy tier.

"GARD" is the information-science name for what was previously called "weyl
compression".  The compression/decompression chain (canonical JSON -> zlib ->
AES-256-CBC -> HMAC-SHA256, sharded) is LOSSLESS; this module defines the
bounded LOSSY floor that the imaging-style channel

    true state -> blur -> sampling -> Langevin noise -> data

imposes on any GARD record.  The NP-Weyl scalar names remain only at the
physics boundary (ueqgm_engine); everything about *compression* is GARD.

Interstitial bit space
----------------------
Between the CRB floor and the actual GARD encoding sits the *interstitial* gap:
bits of precision the record stores beyond what the noisy channel can justify.

This is over-provisioning, not a side channel. Information below the CRB floor
was never resolvable from the measurement — it was not diverted somewhere else
to be recovered later. A large interstitial gap is a reason to spend fewer bits,
not a reason to look for signal in them.

The helpers below make that gap measurable so encoding width can be chosen
against the channel rather than by habit.
"""
from __future__ import annotations

import math

# GARD record floors (information science naming).
GARD_STATE_JSON_BYTES = 50     # 5-scalar state as canonical JSON
GARD_STATE_PACKED_BYTES = 20   # 5 x Float32 LE, raw pre-encoding pack (struct.pack("<5f", ...))

# corpus_ingest._persist_weyl never writes the 20 raw bytes above to brain_kv —
# brain_kv values are TEXT, so the packed record is base64-encoded first
# (learnings:weyl_tensor_packed_b64). That encoding inflates 20 bytes to
# ceil(20/3)*4 = 28 bytes of stored text; GARD_STATE_STORED_BYTES is that
# actual on-disk footprint, not the pre-encoding pack size.
GARD_STATE_STORED_BYTES = 4 * -(-GARD_STATE_PACKED_BYTES // 3)  # ceil(20/3)*4 = 28


def fisher_information_gard(sigma: float, n_samples: int = 1) -> float:
    """Fisher information of a scalar GARD component under Gaussian noise.

    I(theta) = n / sigma^2  (sensitivity ~ 1 in the interior of clip01).
    """
    sigma = max(1e-12, float(sigma))
    return max(1, int(n_samples)) / (sigma * sigma)


def cramer_rao_bound_gard(sigma: float, n_samples: int = 1) -> float:
    """Hard lower bound on Var(theta_hat) for ANY unbiased estimator."""
    return 1.0 / fisher_information_gard(sigma, n_samples)


def gard_floor_bytes(sigma: float, n_components: int = 5, n_samples: int = 1) -> dict:
    """CRB-justified minimum byte footprint of a GARD record.

    Sets uniform-quantizer noise equal to the CRB (never spend bits resolving
    below channel noise):  delta* = sqrt(12 * CRB) = 2*sqrt(3)*sigma/sqrt(n).
    """
    crb = cramer_rao_bound_gard(sigma, n_samples)
    delta_star = math.sqrt(12.0 * crb)
    # When delta_star >= 1.0 (large sigma, high noise) the channel cannot resolve
    # any sub-unit interval, so 1 bit per component is the minimum justified
    # allocation.  The max(1.0, ...) clamp captures this limit explicitly.
    if delta_star >= 1.0:
        bits_per = 1.0
    else:
        bits_per = max(1.0, math.log2(1.0 / max(delta_star, 1e-9)))
    floor = max(1, int(n_components)) * math.ceil(bits_per / 8.0)
    return {
        "sigma": sigma,
        "cramer_rao_var": crb,
        "quant_step": delta_star,
        "bits_per_component": bits_per,
        "floor_bytes": floor,
    }


def interstitial_bits(
    sigma: float,
    n_components: int = 5,
    n_samples: int = 1,
    *,
    actual_bits_per_component: float | None = None,
) -> dict:
    """Bits of precision the encoding spends beyond the Cramér-Rao floor.

    The *interstitial* gap is the difference between what the GARD record
    actually stores and the minimum justified by the CRB. It quantifies
    over-provisioning: precision spent resolving distinctions the channel noise
    cannot support.

    This is a budgeting diagnostic, not a channel. Bits below the floor do not
    hold recoverable structure — the information they would encode was never
    resolvable from the measurement in the first place. A large gap means the
    encoding can be narrowed without loss.

    The actual bits per component default to ``GARD_STATE_STORED_BYTES × 8 /
    n_components`` when *actual_bits_per_component* is not supplied — the
    base64-encoded text actually written to ``brain_kv``, not the smaller
    pre-encoding ``GARD_STATE_PACKED_BYTES`` (Float32) pack size.

    Parameters
    ----------
    sigma:
        Langevin noise σ for the current GARD channel.
    n_components:
        Number of scalar components in the GARD record (default 5 — the 5-scalar
        Weyl state).
    n_samples:
        Number of independent samples (default 1).
    actual_bits_per_component:
        Override for the actual encoding resolution in bits per component.
        Defaults to ``GARD_STATE_STORED_BYTES × 8 / n_components`` (the stored
        base64 text, not the raw Float32 pack).

    Returns
    -------
    Dict with keys:

    * ``floor_bits``         — CRB-justified minimum bits (total across components).
    * ``actual_bits``        — Actual encoded bits (total across components).
    * ``interstitial_bits``  — Gap = actual − floor, clamped ≥ 0.
    * ``interstitial_fraction`` — interstitial / actual ∈ [0, 1].
    """
    n_comp = max(1, int(n_components))
    floor_info = gard_floor_bytes(sigma, n_comp, n_samples)
    floor_bits = float(floor_info["floor_bytes"]) * 8.0

    if actual_bits_per_component is not None:
        actual_bpc = float(actual_bits_per_component)
    else:
        # Base64-encoded packed record, as actually written to brain_kv.
        actual_bpc = (GARD_STATE_STORED_BYTES * 8.0) / max(1, int(n_components))
    actual_bits = actual_bpc * n_comp

    gap = max(0.0, actual_bits - floor_bits)
    fraction = gap / actual_bits if actual_bits > 0.0 else 0.0
    return {
        "sigma": sigma,
        "floor_bits": floor_bits,
        "actual_bits": actual_bits,
        "interstitial_bits": gap,
        "interstitial_fraction": round(fraction, 6),
    }


def state_correlation(
    psi5_a: list[float] | tuple[float, ...],
    psi5_b: list[float] | tuple[float, ...],
) -> float:
    """Pearson correlation between two 5-scalar GARD states.

    Measures whether two states co-vary across their components. Useful for
    detecting drift, spotting cycles that have collapsed onto a degenerate
    state, and as the input to ``correlation_transitivity_score``.

    NOT a photon-number correlation and NOT an entanglement measure. These are
    five real floats, not a quantum state: there is no Hilbert space, no tensor
    product, and so nothing that could be separable or entangled. Detecting
    entanglement requires coincidence measurement across separated detectors
    with independent basis choice and a Bell violation; a correlation
    coefficient between classical vectors cannot do it, however large.

    Returns
    -------
    Pearson correlation C_ab in [-1, 1]. Returns 0.0 when either vector is
    constant (zero variance) or the lengths differ.
    """
    if len(psi5_a) != len(psi5_b) or not psi5_a:
        return 0.0
    n = len(psi5_a)
    a = [float(v) for v in psi5_a]
    b = [float(v) for v in psi5_b]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b)) / n
    var_a = sum((ai - mean_a) ** 2 for ai in a) / n
    var_b = sum((bi - mean_b) ** 2 for bi in b) / n
    denom = math.sqrt(var_a * var_b)
    if denom == 0.0:
        return 0.0
    return max(-1.0, min(1.0, cov / denom))


# Deprecated alias. Retained one release for import compatibility; the name
# asserted a photon-number/entanglement interpretation the function does not
# support. Use state_correlation.
photon_covariance_proxy = state_correlation


def gard_floor_bytes_from_state(
    psi5_now,
    psi5_prev,
    *,
    n_samples: int = 1,
) -> dict:
    """Derive σ from the existing Langevin estimator then compute the CRB floor."""
    from .gard_shard_model import langevin_sigma_from_gard

    return gard_floor_bytes(
        langevin_sigma_from_gard(psi5_now, psi5_prev),
        n_samples=n_samples,
    )


__all__ = [
    "GARD_STATE_JSON_BYTES",
    "GARD_STATE_PACKED_BYTES",
    "GARD_STATE_STORED_BYTES",
    "fisher_information_gard",
    "cramer_rao_bound_gard",
    "gard_floor_bytes",
    "interstitial_bits",
    "state_correlation",
    "photon_covariance_proxy",  # deprecated alias
    "gard_floor_bytes_from_state",
]
