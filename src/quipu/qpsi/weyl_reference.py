"""weyl_reference — the measured Weyl potential from the Dark Energy Survey, as a
physical reference: published measurements with their sources, and the values
that follow from them computed on demand.

What cosmic shear measures
--------------------------
Light bundles are sheared by the Weyl (trace-free) part of curvature; in the
Sachs optical equations the shear is sourced by Ψ₀, the transverse radiative
scalar, and the convergence by the Ricci part.  On cosmological scales weak
lensing therefore measures the Weyl potential

    Ψ_W = (Φ + Ψ) / 2                       (the two metric potentials)

and its evolution with redshift.  Tutusaus et al. (DES Y3) measure the
dimensionless evolution

    ĵ(z) = Σ(z) · Ω_m(z) · D₁(z)/D₁(0) · σ₈(0)

(their eqs. 4, 10, 12), where Σ = 1 in General Relativity, Ω_m(z) is the
matter fraction at z and D₁ the linear growth factor.  This is the measured
quantity closest to QUIPU's Weyl channel: a physical Weyl signal, measured,
with uncertainties.

What is recorded (MEASUREMENTS, with sources)
---------------------------------------------
* DES Y3 Weyl potential ĵ at z = 0.295, 0.467, 0.626, 0.771
  (Tutusaus et al., Nat. Commun. 15, 9295 (2024), arXiv:2312.06434), with a
  Planck CMB prior and without, standard scale cuts;
* the same bins re-measured under w₀w_aCDM (arXiv:2605.22599, 2026), which
  reports w₀ = −0.855 (+0.054 −0.047), w_a = −0.48 ± 0.22;
* DES Y3 cosmic shear S₈ (Amon, Gruen, Secco et al., PRD 105, 023514 (2022),
  arXiv:2105.13544 / 2105.13543);
* DES Y6 cosmic shear S₈, Ω_m, σ₈ (arXiv:2602.10065, 2026), NLA and TATT;
* Planck 2018 TT,TE,EE+lowE+lensing (arXiv:1807.06209, Table 2) as the CMB
  anchor the DES papers compare against.

These are observations, not tuned constants: each carries its source and
uncertainty, and none is a threshold anywhere in QUIPU.

What is computed (dynamic values)
---------------------------------
* ``growth(z, cosmo)`` — D₁(z)/D₁(0) and f = d ln D₁/d ln a from the linear
  growth equation, integrated in ln a, for flat w₀w_aCDM (ΛCDM when w₀ = −1,
  w_a = 0);
* ``j_hat(z, cosmo)`` — the GR prediction of ĵ;
* ``weyl_decay_rate(z, cosmo)`` — d ln ĵ / d ln a = f + d ln Ω_m(a)/d ln a.
  Negative at late times: the Weyl potential grew during matter domination and
  decays as dark energy takes over;
* ``confirm(cosmo, measurement)`` — per-bin Σ_eff = ĵ_measured / ĵ_GR (the Σ
  that would explain each bin if growth is as assumed), per-bin pull, χ², and
  the inverse-variance amplitude A_W = Σ w·Σ_eff / Σ w (GR with that cosmology
  predicts A_W = 1);
* ``reference()`` — all of the above for the recorded cosmologies, as one
  JSON document.

The comparison uses each cosmology's point values and the measurement's
errors only; it does not propagate the cosmology's own uncertainty.  So the
pulls against Planck 2018 here (−1.45, −2.25, −0.07, −1.10 with the CMB-prior
bins) are smaller than the tensions the paper reports (2σ, 2.8σ, compatible,
1.6σ), whose ΛCDM prediction came from its own chains (σ₈ = 0.849 ± 0.030).
The paper's figures are kept in ``WeylMeasurement.note``; this module does not
claim to reproduce them.

Validation: the growth solver gives D = a and f = 1 exactly in
Einstein–de Sitter, and f(0) = 0.513, D(z=1)/D(0) = 0.612 for Ω_m = 0.3 ΛCDM
(f ≈ Ω_m^0.55 = 0.516).

Relation to QUIPU's Weyl channel, and what this module does not do
------------------------------------------------------------------
``learnings:weyl_tensor`` (Ψ₀–Ψ₄ from corpus compression) and the CAT
``weyl_channel`` split are QUIPU's own Weyl objects; they are not physical
curvature, and no mapping from Mpc-scale lensing into them is asserted here.
This module supplies a measured reference beside them.  It writes only
``brain_kv["entirety:weyl_reference"]`` (``_kv_set`` refuses any other key),
is an input to no gate, and is not read by mesh_slm.py, which is untouched.
Stdlib only.  No network: the measurements are the published numbers.

翈 — the reference is what was measured; how far QUIPU leans on it stays the
operator's decision.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

KV_KEY: str = "entirety:weyl_reference"


# ---------------------------------------------------------------------------
# Measurements (published numbers, with sources)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cosmology:
    """Flat w₀w_aCDM background.  ``sigma8`` is σ₈ at z = 0."""
    name: str
    omega_m: float
    sigma8: float
    w0: float = -1.0
    wa: float = 0.0
    source: str = ""

    @property
    def s8(self) -> float:
        """S₈ ≡ σ₈ (Ω_m/0.3)^½ of these point values (marginal means of the
        posterior differ slightly from this; the reported S₈ is kept separately)."""
        return self.sigma8 * math.sqrt(self.omega_m / 0.3)


@dataclass(frozen=True)
class JBin:
    z: float
    j: float
    err_lo: float          # 1σ below
    err_hi: float          # 1σ above


@dataclass(frozen=True)
class WeylMeasurement:
    name: str
    bins: tuple[JBin, ...]
    source: str
    note: str = ""


@dataclass(frozen=True)
class S8Result:
    name: str
    s8: float
    err_lo: float
    err_hi: float
    source: str
    omega_m: float | None = None
    omega_m_err: tuple[float, float] | None = None     # (lo, hi)
    sigma8: float | None = None
    sigma8_err: tuple[float, float] | None = None


PLANCK18 = Cosmology(
    "Planck 2018 TT,TE,EE+lowE+lensing", omega_m=0.3153, sigma8=0.8111,
    source="Planck Collaboration VI, A&A 641, A6 (2020), arXiv:1807.06209, Table 2 (Ωm 0.3153±0.0073, σ8 0.8111±0.0060)")
DES_Y6_NLA = Cosmology(
    "DES Y6 cosmic shear (NLA)", omega_m=0.332, sigma8=0.763,
    source="DES Collaboration, arXiv:2602.10065 (2026), Table III")
DES_Y6_TATT = Cosmology(
    "DES Y6 cosmic shear (TATT)", omega_m=0.321, sigma8=0.763,
    source="DES Collaboration, arXiv:2602.10065 (2026), Table III")
# Dark-energy parameters from the 2026 Weyl re-analysis.  Ω_m and σ₈ are not
# re-fitted here; Planck's are used so the w₀w_a change is the only change.
W0WA_2026 = Cosmology(
    "Planck 2018 background with DES Y3 Weyl w0waCDM (w0 -0.855, wa -0.48)",
    omega_m=0.3153, sigma8=0.8111, w0=-0.855, wa=-0.48,
    source="arXiv:2605.22599 (2026): w0 = -0.855 +0.054/-0.047, wa = -0.48 ± 0.22; Ωm, σ8 from Planck 2018")

COSMOLOGIES: tuple[Cosmology, ...] = (PLANCK18, DES_Y6_NLA, DES_Y6_TATT, W0WA_2026)

_TUTUSAUS = "Tutusaus et al. (DES), Nat. Commun. 15, 9295 (2024), arXiv:2312.06434, MagLim lenses, standard scale cuts"

DES_Y3_WEYL_CMB = WeylMeasurement(
    "DES Y3 Weyl potential ĵ(z), Planck CMB prior",
    bins=(JBin(0.295, 0.325, 0.015, 0.015), JBin(0.467, 0.333, 0.019, 0.017),
          JBin(0.626, 0.387, 0.029, 0.026), JBin(0.771, 0.354, 0.035, 0.035)),
    source=_TUTUSAUS,
    note="reported tensions with ΛCDM+Planck: 2σ, 2.8σ, compatible, 1.6σ; σ8 fit 0.743±0.039 vs 0.849±0.030 (2.2σ)")
DES_Y3_WEYL_NOCMB = WeylMeasurement(
    "DES Y3 Weyl potential ĵ(z), no CMB prior",
    bins=(JBin(0.295, 0.327, 0.017, 0.017), JBin(0.467, 0.328, 0.023, 0.019),
          JBin(0.626, 0.388, 0.032, 0.032), JBin(0.771, 0.345, 0.040, 0.040)),
    source=_TUTUSAUS,
    note="reported tensions with ΛCDM: 2.1σ, 2.9σ, 1.7σ, 2.4σ")

MEASUREMENTS: tuple[WeylMeasurement, ...] = (DES_Y3_WEYL_CMB, DES_Y3_WEYL_NOCMB)

S8_RESULTS: tuple[S8Result, ...] = (
    S8Result("DES Y3 cosmic shear, fiducial", 0.759, 0.023, 0.025,
             "Amon et al. / Secco et al. (DES), PRD 105, 023514/023515 (2022), arXiv:2105.13544"),
    S8Result("DES Y3 cosmic shear, optimised ΛCDM scale cuts", 0.772, 0.017, 0.018,
             "Amon et al. / Secco et al. (DES), PRD 105, 023514/023515 (2022), arXiv:2105.13544"),
    S8Result("DES Y6 cosmic shear, NLA", 0.798, 0.015, 0.014,
             "DES Collaboration, arXiv:2602.10065 (2026)",
             omega_m=0.332, omega_m_err=(0.044, 0.035), sigma8=0.763, sigma8_err=(0.057, 0.050)),
    S8Result("DES Y6 cosmic shear, TATT", 0.783, 0.015, 0.019,
             "DES Collaboration, arXiv:2602.10065 (2026)",
             omega_m=0.321, omega_m_err=(0.047, 0.036), sigma8=0.763, sigma8_err=(0.062, 0.053)),
    S8Result("Planck 2018 TT,TE,EE+lowE+lensing", 0.832, 0.013, 0.013,
             "Planck Collaboration VI, A&A 641, A6 (2020), arXiv:1807.06209"),
)


# ---------------------------------------------------------------------------
# Background and growth (flat w₀w_aCDM, radiation neglected)
# ---------------------------------------------------------------------------

def e2(a: float, c: Cosmology) -> float:
    """H²(a)/H₀² with CPL dark energy w(a) = w₀ + w_a(1 − a)."""
    ode = 1.0 - c.omega_m
    de = ode * a ** (-3.0 * (1.0 + c.w0 + c.wa)) * math.exp(-3.0 * c.wa * (1.0 - a))
    return c.omega_m * a ** -3.0 + de


def omega_m_of_a(a: float, c: Cosmology) -> float:
    return c.omega_m * a ** -3.0 / e2(a, c)


def _dln_e2(a: float, c: Cosmology) -> float:
    """d ln E² / d ln a."""
    ode = 1.0 - c.omega_m
    m = c.omega_m * a ** -3.0
    de = ode * a ** (-3.0 * (1.0 + c.w0 + c.wa)) * math.exp(-3.0 * c.wa * (1.0 - a))
    w = c.w0 + c.wa * (1.0 - a)
    return (-3.0 * m - 3.0 * (1.0 + w) * de) / (m + de)


def _growth_table(c: Cosmology, a_init: float = 1e-3, steps: int = 4000) -> list[tuple[float, float, float]]:
    """RK4 in x = ln a of  D'' + (2 + ½ d ln E²/d ln a) D' − (3/2) Ω_m(a) D = 0,
    starting in matter domination (D = a, D' = a).  Returns (a, D, D')."""
    x0, x1 = math.log(a_init), 0.0
    h = (x1 - x0) / steps

    def rhs(x: float, d: float, dp: float) -> tuple[float, float]:
        a = math.exp(x)
        return dp, -(2.0 + 0.5 * _dln_e2(a, c)) * dp + 1.5 * omega_m_of_a(a, c) * d

    d, dp, x = a_init, a_init, x0
    out = [(a_init, d, dp)]
    for _ in range(steps):
        k1 = rhs(x, d, dp)
        k2 = rhs(x + h / 2, d + h / 2 * k1[0], dp + h / 2 * k1[1])
        k3 = rhs(x + h / 2, d + h / 2 * k2[0], dp + h / 2 * k2[1])
        k4 = rhs(x + h, d + h * k3[0], dp + h * k3[1])
        d += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        dp += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        x += h
        out.append((math.exp(x), d, dp))
    return out


_TABLES: dict[Cosmology, list[tuple[float, float, float]]] = {}


def _table(c: Cosmology) -> list[tuple[float, float, float]]:
    t = _TABLES.get(c)
    if t is None:
        t = _TABLES[c] = _growth_table(c)
    return t


@dataclass(frozen=True)
class Growth:
    z: float
    d_norm: float          # D₁(z)/D₁(0)
    f: float               # d ln D₁ / d ln a


def growth(z: float, c: Cosmology) -> Growth:
    if z < 0.0:
        raise ValueError("z must be ≥ 0")
    t = _table(c)
    a = 1.0 / (1.0 + z)
    d0 = t[-1][1]
    # t is uniform in ln a; linear interpolation in ln a
    x0, x1 = math.log(t[0][0]), 0.0
    pos = (math.log(a) - x0) / (x1 - x0) * (len(t) - 1)
    i = min(len(t) - 2, max(0, int(pos)))
    u = pos - i
    d = t[i][1] * (1 - u) + t[i + 1][1] * u
    dp = t[i][2] * (1 - u) + t[i + 1][2] * u
    return Growth(z=z, d_norm=d / d0, f=dp / d)


def j_hat(z: float, c: Cosmology, sigma_mg: float = 1.0) -> float:
    """ĵ(z) = Σ · Ω_m(z) · D₁(z)/D₁(0) · σ₈(0)  (Tutusaus et al. eqs. 10, 12)."""
    a = 1.0 / (1.0 + z)
    return sigma_mg * omega_m_of_a(a, c) * growth(z, c).d_norm * c.sigma8


def weyl_decay_rate(z: float, c: Cosmology) -> float:
    """d ln ĵ / d ln a = f + d ln Ω_m(a)/d ln a = f − 3 − d ln E²/d ln a."""
    a = 1.0 / (1.0 + z)
    return growth(z, c).f - 3.0 - _dln_e2(a, c)


# ---------------------------------------------------------------------------
# Confirmation: measurement against prediction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinCheck:
    z: float
    measured: float
    predicted: float
    sigma_eff: float        # measured / predicted
    pull: float             # (measured − predicted) / σ on the side of the deviation


@dataclass(frozen=True)
class Confirmation:
    cosmology: str
    measurement: str
    bins: tuple[BinCheck, ...]
    chi2: float
    dof: int
    amplitude: float        # A_W, inverse-variance mean of Σ_eff; GR predicts 1
    amplitude_err: float
    amplitude_pull: float   # (A_W − 1) / err

    def to_json(self) -> dict:
        d = asdict(self)
        d["bins"] = [asdict(b) for b in self.bins]
        return d


def confirm(c: Cosmology, m: WeylMeasurement) -> Confirmation:
    checks: list[BinCheck] = []
    chi2 = 0.0
    wsum = wa_sum = 0.0
    for b in m.bins:
        pred = j_hat(b.z, c)
        err = b.err_hi if b.j > pred else b.err_lo
        pull = (b.j - pred) / err
        chi2 += pull * pull
        s_eff = b.j / pred
        s_err = 0.5 * (b.err_lo + b.err_hi) / pred
        w = 1.0 / (s_err * s_err)
        wsum += w
        wa_sum += w * s_eff
        checks.append(BinCheck(z=b.z, measured=b.j, predicted=pred, sigma_eff=s_eff, pull=pull))
    amp = wa_sum / wsum
    amp_err = 1.0 / math.sqrt(wsum)
    return Confirmation(cosmology=c.name, measurement=m.name, bins=tuple(checks), chi2=chi2,
                        dof=len(m.bins), amplitude=amp, amplitude_err=amp_err,
                        amplitude_pull=(amp - 1.0) / amp_err)


def curve(c: Cosmology, zs: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)) -> list[dict]:
    """ĵ(z), growth and the Weyl decay rate along z for one cosmology."""
    out = []
    for z in zs:
        g = growth(z, c)
        out.append({"z": z, "j_hat": j_hat(z, c), "d_norm": g.d_norm, "f": g.f,
                    "omega_m_z": omega_m_of_a(1.0 / (1.0 + z), c),
                    "weyl_decay_rate": weyl_decay_rate(z, c)})
    return out


def reference(cosmologies: Sequence[Cosmology] = COSMOLOGIES,
              measurements: Sequence[WeylMeasurement] = MEASUREMENTS) -> dict:
    """The whole reference as one JSON-able document."""
    return {
        "what": "measured Weyl potential evolution ĵ(z) (DES) against the GR prediction",
        "definition": "j_hat(z) = Sigma * Omega_m(z) * D1(z)/D1(0) * sigma8(0); Sigma = 1 in GR",
        "cosmologies": [dict(asdict(c), s8_point=c.s8) for c in cosmologies],
        "s8_results": [asdict(r) for r in S8_RESULTS],
        "measurements": [{"name": m.name, "source": m.source, "note": m.note,
                          "bins": [asdict(b) for b in m.bins]} for m in measurements],
        "confirmations": [confirm(c, m).to_json() for c in cosmologies for m in measurements],
        "curves": {c.name: curve(c) for c in cosmologies},
    }


# ---------------------------------------------------------------------------
# Persistence (observability only)
# ---------------------------------------------------------------------------

def _kv_set(cn: sqlite3.Connection, key: str, value) -> None:
    if key != KV_KEY:
        raise PermissionError(f"weyl_reference writes only {KV_KEY!r}, not {key!r}")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def record(cn: sqlite3.Connection, doc: Mapping | None = None) -> dict:
    doc = dict(doc) if doc is not None else reference()
    _kv_set(cn, KV_KEY, doc)
    return doc


def summary(doc: Mapping | None = None) -> dict:
    """The few numbers worth showing in a status line."""
    doc = doc if doc is not None else reference()
    rows = []
    for c in doc["confirmations"]:
        rows.append({"cosmology": c["cosmology"], "measurement": c["measurement"],
                     "A_W": round(c["amplitude"], 4), "A_W_err": round(c["amplitude_err"], 4),
                     "A_W_pull": round(c["amplitude_pull"], 2), "chi2": round(c["chi2"], 2),
                     "dof": c["dof"]})
    return {"confirmations": rows}


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m src.quipu.qpsi.weyl_reference",
                                description="DES Weyl potential reference: measurements vs GR prediction.")
    p.add_argument("--full", action="store_true", help="print the whole reference, not the summary")
    p.add_argument("--record", action="store_true", help=f"write it to brain_kv[{KV_KEY!r}]")
    args = p.parse_args(argv)
    doc = reference()
    if args.record:
        from .. import system_entirety as se
        with se._conn() as cn:
            record(cn, doc)
    print(json.dumps(doc if args.full else summary(doc), indent=2, ensure_ascii=False))
    return 0


__all__ = [
    "KV_KEY", "Cosmology", "JBin", "WeylMeasurement", "S8Result", "PLANCK18", "DES_Y6_NLA",
    "DES_Y6_TATT", "W0WA_2026", "COSMOLOGIES", "DES_Y3_WEYL_CMB", "DES_Y3_WEYL_NOCMB",
    "MEASUREMENTS", "S8_RESULTS", "e2", "omega_m_of_a", "Growth", "growth", "j_hat",
    "weyl_decay_rate", "BinCheck", "Confirmation", "confirm", "curve", "reference",
    "record", "summary",
]

if __name__ == "__main__":
    raise SystemExit(_main())
