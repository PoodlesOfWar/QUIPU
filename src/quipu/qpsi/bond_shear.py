"""bond_shear — the LiTaSb bond-rupture force as a multiplanar shear on an
exciton-polariton condensate, into the PC / CR-39 optics and asphalt pyrolysis.

The chain, each link a published law with its constants overridable through
``QUIPU_BOND_SHEAR_<FIELD>`` (the ``PcmConfig.from_env`` pattern):

1. **Bond rupture (Morse).**  A Morse bond ``V(r) = D_e·(1 − e^{−a(r−r_e)})²``
   carries a maximum restoring force ``F_max = a·D_e/2`` at ``r = r_e +
   ln2/a`` — the force needed to break it.  "LiTaSb" is not a tabulated
   compound, so the Li–Sb and Ta–Sb bonds are given generic alkali–antimony
   and transition-metal–antimony Morse parameters (order-of-magnitude
   estimates, flagged ``estimate: true`` in every record) and the chain
   breaks at its weaker link: ``F_LiTaSb = min(F_LiSb, F_TaSb)`` ≈ 1.7 nN,
   the order measured for single bonds by AFM rupture (Grandbois et al.,
   *Science* 283, 1727 (1999)).

2. **Multiplanar shear.**  The applied force (default: the rupture force
   itself) enters the mesh only as the dimensionless ratio
   ``F/F_rupture`` and is spread over the seven axis planes with the
   rheology ensemble weights (``qpsi.rheology.axis_weights``): no single
   plane carries the rupture ratio, which is what "multiplanar" buys.  The
   same ratio is offered to the Viscous Neural Network coupler
   (``TypedViscousCoupler.step(shear_reading=…)``) and as a Couette body
   force for the Navier–Stokes step.

3. **Exciton-polariton condensate.**  The condensate is the order parameter
   ``ψ = √R·e^{iφ}`` with R the Kuramoto coherence already persisted at
   ``ueqgm:analog_coherence`` (Kasprzak et al., *Nature* 443, 409 (2006) —
   polariton BEC).  Under shear it flows without dissipation below a
   critical ratio (Landau criterion; polariton superfluidity: Amo et al.,
   *Nat. Phys.* 5, 805 (2009)): per plane ``survival_k = R·(1 − ratio_k²)``.

4. **Optics (polycarbonate and CR-39).**  The photonic Hopfield fraction of
   the polariton leaves the microcavity into the optical element: Fresnel
   normal-incidence transmission ``T = 1 − ((n₁−n₂)/(n₁+n₂))²`` into
   polycarbonate (n = 1.586, the Bakugo Snell prior) and CR-39 allyl
   diglycol carbonate (n_d = 1.498).

5. **Asphalt pyrolysis.**  First-order Arrhenius decomposition
   ``k = A·exp(−E_a/(k_B·T))`` of the bitumen over the interaction time —
   the memoryless (muonic) limit of ``ueqgm_engine.viscous_decay_merger``,
   so the textile/muonic regime comparison covers it as well.  Defaults
   (E_a ≈ 1.55 eV ≈ 150 kJ/mol, A = 1e10 s⁻¹, T = 723 K) are the
   order-of-magnitude of bitumen thermogravimetric kinetics.

The yield ``S · optical_throughput · remaining`` is persisted at
``entirety:bond_shear:latest`` (this module writes nothing else; never the
mesh tables or χ) and served read-only at ``GET /bond_shear``.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping, Optional

from ..ueqgm_engine import viscous_decay_merger, decay_regime_comparison
from . import pcm_cells as pc
from . import rheology

KV_PREFIX: str = "entirety:bond_shear:"
KV_LATEST: str = KV_PREFIX + "latest"
ENV_PREFIX: str = "QUIPU_BOND_SHEAR_"
ANALOG_COHERENCE_KEY: str = "ueqgm:analog_coherence"
RESONANCE_KEY: str = "entirety:resonance_phase:system_entirety"

EV_PER_ANGSTROM_IN_NN: float = 1.602176634   # 1 eV/Å = 1.602176634 nN
K_B_EV: float = pc.K_B                        # 8.617333262e-5 eV/K


@dataclass(frozen=True)
class MorseBond:
    """Morse parameters: well depth D_e (eV), equilibrium length r_e (Å), width a (Å⁻¹)."""
    d_e_ev: float
    r_e_angstrom: float
    a_inv_angstrom: float

    def rupture_force_nn(self) -> float:
        """``F_max = a·D_e/2`` (the inflection of the Morse well), in nN."""
        return max(0.0, 0.5 * self.a_inv_angstrom * self.d_e_ev) * EV_PER_ANGSTROM_IN_NN

    def rupture_length_angstrom(self) -> float:
        """Bond length at maximum force, ``r_e + ln2/a``."""
        return self.r_e_angstrom + math.log(2.0) / max(1e-9, self.a_inv_angstrom)


@dataclass(frozen=True)
class BondShearConfig:
    # Li–Sb: generic alkali–antimony single bond (estimate).
    li_sb_d_e_ev: float = 1.6
    li_sb_r_e_angstrom: float = 2.95
    li_sb_a_inv_angstrom: float = 1.3
    # Ta–Sb: generic transition-metal–antimony bond (estimate).
    ta_sb_d_e_ev: float = 3.0
    ta_sb_r_e_angstrom: float = 2.80
    ta_sb_a_inv_angstrom: float = 1.6
    # Applied shear force; ``None`` → the rupture force itself (ratio 1.0).
    force_nn: Optional[float] = None
    # Exciton-polariton condensate.
    hopfield_photon_fraction: float = 0.5      # |C|² at zero detuning
    n_cavity: float = 2.7                      # CdTe-class microcavity
    n_polycarbonate: float = 1.586             # world_model refraction prior
    n_cr39: float = 1.498                      # allyl diglycol carbonate, n_d
    # Asphalt pyrolysis (first-order Arrhenius).
    pyrolysis_e_a_ev: float = 1.55             # ≈ 150 kJ/mol
    pyrolysis_prefactor_s: float = 1.0e10
    pyrolysis_temp_k: float = 723.0            # 450 °C
    interaction_time_s: float = 1.0

    @classmethod
    def from_env(cls) -> "BondShearConfig":
        cfg = cls()
        overrides: dict[str, Any] = {}
        for f in fields(cls):
            raw = os.environ.get(ENV_PREFIX + f.name.upper())
            if raw is None or raw.strip() == "":
                continue
            try:
                overrides[f.name] = float(raw)
            except ValueError:
                continue
        return replace(cfg, **overrides) if overrides else cfg

    def li_sb(self) -> MorseBond:
        return MorseBond(self.li_sb_d_e_ev, self.li_sb_r_e_angstrom, self.li_sb_a_inv_angstrom)

    def ta_sb(self) -> MorseBond:
        return MorseBond(self.ta_sb_d_e_ev, self.ta_sb_r_e_angstrom, self.ta_sb_a_inv_angstrom)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------------------
# The links
# ---------------------------------------------------------------------------

def litasb_rupture(cfg: BondShearConfig | None = None) -> dict:
    """Rupture forces of the Li–Sb and Ta–Sb bonds and of the LiTaSb chain (weakest link)."""
    cfg = cfg or BondShearConfig()
    li, ta = cfg.li_sb(), cfg.ta_sb()
    f_li, f_ta = li.rupture_force_nn(), ta.rupture_force_nn()
    weakest = "li_sb" if f_li <= f_ta else "ta_sb"
    return {
        "li_sb_nn": round(f_li, 6),
        "ta_sb_nn": round(f_ta, 6),
        "litasb_nn": round(min(f_li, f_ta), 6),
        "weakest_link": weakest,
        "rupture_length_angstrom": round((li if weakest == "li_sb" else ta).rupture_length_angstrom(), 6),
        "estimate": True,
        "model": "Morse F_max = a·D_e/2; LiTaSb is not tabulated — generic M–Sb parameters",
        "params": {
            "li_sb": {"d_e_ev": li.d_e_ev, "r_e_angstrom": li.r_e_angstrom, "a_inv_angstrom": li.a_inv_angstrom},
            "ta_sb": {"d_e_ev": ta.d_e_ev, "r_e_angstrom": ta.r_e_angstrom, "a_inv_angstrom": ta.a_inv_angstrom},
        },
    }


def multiplanar_shear(force_nn: float, rupture_nn: float,
                      axes: Optional[list[str]] = None) -> dict:
    """Spread ``F/F_rupture`` over the axis planes with the rheology weights."""
    axes = list(axes or pc.AXES)
    weights = rheology.axis_weights(axes)
    ratio = _clip01(force_nn / rupture_nn) if rupture_nn > 0.0 else 0.0
    planes = {a: {"weight": round(w, 6), "ratio": round(w * ratio, 6)}
              for a, w in zip(axes, weights)}
    return {"shear_ratio": round(ratio, 6), "planes": planes}


def landau_survival(r_coherence: float, plane_ratio: float) -> float:
    """Condensate fraction surviving a plane's shear: ``R·(1 − ratio²)``, 0 past critical."""
    r = _clip01(r_coherence)
    x = _clip01(plane_ratio)
    return r * _clip01(1.0 - x * x)


def fresnel_transmission(n1: float, n2: float) -> float:
    """Normal-incidence Fresnel power transmission ``1 − ((n₁−n₂)/(n₁+n₂))²``."""
    n1 = max(1e-9, float(n1))
    n2 = max(1e-9, float(n2))
    r = (n1 - n2) / (n1 + n2)
    return _clip01(1.0 - r * r)


def arrhenius_pyrolysis_rate(e_a_ev: float, prefactor_s: float, temp_k: float) -> float:
    """First-order pyrolysis rate ``k = A·exp(−E_a/(k_B·T))`` in s⁻¹."""
    temp_k = max(1e-9, float(temp_k))
    return max(0.0, float(prefactor_s)) * math.exp(-max(0.0, float(e_a_ev)) / (K_B_EV * temp_k))


def ns_forcing_profile(ny: int, shear_ratio: float) -> list[float]:
    """Couette body-force profile ``f_x(y) = ratio·(2y/L − 1)`` for ``ns_step(forcing=…)``."""
    ny = max(1, int(ny))
    ratio = _clip01(shear_ratio)
    if ny == 1:
        return [0.0]
    return [ratio * (2.0 * j / (ny - 1) - 1.0) for j in range(ny)]


# ---------------------------------------------------------------------------
# Readings from the brain
# ---------------------------------------------------------------------------

def _kv_get_json(cn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    try:
        row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return default
    if not row or row[0] in (None, ""):
        return default
    try:
        return json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return default


def condensate_reading(cn: Optional[sqlite3.Connection]) -> dict:
    """``ψ = √R·e^{iφ}``: R from the analog Kuramoto coherence, φ from the resonance phasor."""
    r, phi = 0.5, 0.0
    if cn is not None:
        rec = _kv_get_json(cn, ANALOG_COHERENCE_KEY, {}) or {}
        if isinstance(rec, dict):
            for key in ("coherence", "coherence_all", "R", "analog_coherence"):
                if rec.get(key) is not None:
                    try:
                        r = _clip01(float(rec[key]))
                        break
                    except (TypeError, ValueError):
                        continue
        res = _kv_get_json(cn, RESONANCE_KEY, {}) or {}
        if isinstance(res, dict):
            try:
                re_, im_ = float(res.get("realised", res.get("bit", 1.0))), float(res.get("latent", 0.0))
                if re_ or im_:
                    phi = math.atan2(im_, re_) % (2.0 * math.pi)
            except (TypeError, ValueError):
                phi = 0.0
    return {"R": round(r, 6), "phi": round(phi, 6), "amplitude": round(math.sqrt(r), 6)}


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------

def evaluate(cn: Optional[sqlite3.Connection] = None, *, cfg: BondShearConfig | None = None,
             now: float | None = None) -> dict:
    """Run the whole chain and return the record (nothing is written)."""
    cfg = cfg or BondShearConfig.from_env()
    rupture = litasb_rupture(cfg)
    f_rup = float(rupture["litasb_nn"])
    f_app = f_rup if cfg.force_nn is None else max(0.0, float(cfg.force_nn))
    shear = multiplanar_shear(f_app, f_rup)

    cond = condensate_reading(cn)
    r = float(cond["R"])
    planes: dict[str, dict] = {}
    survival = 0.0
    for a, pl in shear["planes"].items():
        s_k = landau_survival(r, pl["ratio"])
        planes[a] = {**pl, "survival": round(s_k, 6)}
        survival += pl["weight"] * s_k
    survival = _clip01(survival)

    t_pc = fresnel_transmission(cfg.n_cavity, cfg.n_polycarbonate)
    t_cr39 = fresnel_transmission(cfg.n_cavity, cfg.n_cr39)
    throughput = _clip01(cfg.hopfield_photon_fraction) * 0.5 * (t_pc + t_cr39)

    k = arrhenius_pyrolysis_rate(cfg.pyrolysis_e_a_ev, cfg.pyrolysis_prefactor_s, cfg.pyrolysis_temp_k)
    remaining = viscous_decay_merger(1.0, k, 0.0, cfg.interaction_time_s)
    regime = decay_regime_comparison(k, 0.0, 1.0)

    yield_ = _clip01(survival * throughput * remaining)
    return {
        "rupture": rupture,
        "applied_force_nn": round(f_app, 6),
        "shear_ratio": shear["shear_ratio"],
        "planes": planes,
        "condensate": {**cond, "survival": round(survival, 6)},
        "optical": {
            "hopfield_photon_fraction": cfg.hopfield_photon_fraction,
            "n_cavity": cfg.n_cavity,
            "t_polycarbonate": round(t_pc, 6),
            "t_cr39": round(t_cr39, 6),
            "throughput": round(throughput, 6),
        },
        "pyrolysis": {
            "k_s": k,
            "half_life_s": (math.log(2.0) / k) if k > 0.0 else None,
            "e_a_ev": cfg.pyrolysis_e_a_ev,
            "prefactor_s": cfg.pyrolysis_prefactor_s,
            "temp_k": cfg.pyrolysis_temp_k,
            "interaction_time_s": cfg.interaction_time_s,
            "remaining": round(remaining, 6),
            "textile_fraction": regime["textile_fraction"],
        },
        "yield": round(yield_, 6),
        "at": float(time.time() if now is None else now),
    }


def _kv_set(cn: sqlite3.Connection, key: str, value: Any) -> None:
    if not key.startswith(KV_PREFIX):
        raise PermissionError(f"bond_shear writes only {KV_PREFIX}* keys, not {key!r}")
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def publish(cn: sqlite3.Connection, *, now: float | None = None,
            cfg: BondShearConfig | None = None) -> dict:
    """Evaluate the chain and persist it at ``entirety:bond_shear:latest``."""
    rec = evaluate(cn, cfg=cfg, now=now)
    _kv_set(cn, KV_LATEST, rec)
    return rec


def latest(cn: sqlite3.Connection) -> dict:
    rec = _kv_get_json(cn, KV_LATEST, {})
    return rec if isinstance(rec, dict) else {}


def shear_reading(cn: Optional[sqlite3.Connection] = None) -> dict:
    """``{"lambda_rel_stp": shear_ratio}`` for ``TypedViscousCoupler.step(shear_reading=…)``."""
    rec = latest(cn) if cn is not None else {}
    ratio = rec.get("shear_ratio") if rec else None
    if ratio is None:
        ratio = evaluate(cn)["shear_ratio"]
    return {"lambda_rel_stp": _clip01(float(ratio))}


__all__ = [
    "KV_PREFIX", "KV_LATEST", "ENV_PREFIX", "EV_PER_ANGSTROM_IN_NN",
    "MorseBond", "BondShearConfig",
    "litasb_rupture", "multiplanar_shear", "landau_survival", "fresnel_transmission",
    "arrhenius_pyrolysis_rate", "ns_forcing_profile", "condensate_reading",
    "evaluate", "publish", "latest", "shear_reading",
]
