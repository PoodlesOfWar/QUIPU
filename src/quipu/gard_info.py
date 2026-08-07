"""GARD information floor: Fisher / Cramér–Rao bounds for the lossy tier.

"GARD" is the information-science name for what was previously called "weyl
compression".  The compression/decompression chain (canonical JSON -> zlib ->
AES-256-CBC -> HMAC-SHA256, sharded) is LOSSLESS; this module defines the
bounded LOSSY floor that the imaging-style channel

    true state -> blur -> sampling -> Langevin noise -> data

imposes on any GARD record.  The NP-Weyl scalar names remain only at the
physics boundary (ueqgm_engine); everything about *compression* is GARD.
"""
from __future__ import annotations

import math

# GARD record floors (information science naming).
GARD_STATE_JSON_BYTES = 50     # 5-scalar state as canonical JSON
GARD_STATE_PACKED_BYTES = 20   # 5 x Float32 LE packed record


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
    "fisher_information_gard",
    "cramer_rao_bound_gard",
    "gard_floor_bytes",
    "gard_floor_bytes_from_state",
]
