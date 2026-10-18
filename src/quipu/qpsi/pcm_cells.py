"""pcm_cells — chalcogenide phase-change memory cells as the cellular RAM behind
r-ADMIN's pressures, with the derivative decision model that turns an rADAM
step into a programming action and feeds the cell's slope back into rADAM.

The physics (what a cell is)
----------------------------
A phase-change cell is a chalcogenide (Ge₂Sb₂Te₅, "GST") volume between two
electrodes.  It stores information in the fraction x ∈ [0, 1] of the volume
that is crystalline:

    conductance   G(x, t) = x·G_set + (1 − x)·G_reset(t)

* SET (crystallise): a moderate current heats the cell above the
  crystallisation temperature T_x (≈ 150 °C for GST) and below the melting
  point T_m (≈ 620 °C); x grows by nucleation and growth, JMAK kinetics
  x(t) = 1 − exp(−(t/τ(T))ⁿ) with Avrami exponent n and an Arrhenius τ(T).
  Partial SET gives intermediate x: this is the analogue/multilevel use of PCM
  (Burr et al., J. Vac. Sci. Technol. B 28, 223 (2010); Sebastian et al.,
  J. Appl. Phys. 124, 111101 (2018)).
* RESET (amorphise): a short, high current melts the volume (T > T_m); the
  quench is fast enough (≳10⁹ K/s) that it freezes amorphous, x → 0.
  RESET is abrupt; there is no gradual decrease of x.
* Drift: the amorphous phase relaxes structurally, so its resistance rises as
  a power law, R(t) = R₀ (t/t₀)^ν with ν ≈ 0.1 (Ielmini, Lacaita & Mantegazza,
  IEEE Trans. Electron Devices 54, 308 (2007)).  So G_reset(t) = G_reset,0
  (t/t₀)^(−ν), and a programmed state moves on its own after it is written.
* Threshold switching: the amorphous phase conducts above a threshold
  voltage V_th without crystallising (ovonic threshold switching, Ovshinsky,
  Phys. Rev. Lett. 21, 1450 (1968)); it is what lets a RESET cell be
  programmed at all.  Here it sets the minimum pulse that does anything.
* Heat: the cell is a thermal RC: dT/dt = I²R/C_th − (T − T_amb)/τ_th.

Every constant above is a published order of magnitude for GST, recorded in
``PcmConfig`` with its source and overridable (``QUIPU_PCM_*``).  No hardware
is claimed: this is a model of the cell a pressure would live in.

The derivatives (what the decision model reads)
-----------------------------------------------
    dx/dt     = (n/τ(T)) (1 − x) [−ln(1 − x)]^((n−1)/n)      T_x < T < T_m
              = −x/τ_melt                                   T ≥ T_m
              = 0                                           T ≤ T_x
    dG/dx     = G_set − G_reset(t)
    dG_r/dt   = −ν G_reset(t) / t                            drift, amorphous part
    dG/dpulse = (dG/dx)·Δx(one SET pulse)                     the programming slope
    dT/dt     = I²R/C_th − (T − T_amb)/τ_th

Those five derivatives decide the action.  They are not tuned thresholds:
each is the cell's own rate at its current state.

The decision model (one rADAM step → one action)
------------------------------------------------
rADAM proposes a pressure p_new.  The cell holds pressure as normalised
conductance p = (G − G_min)/(G_max − G_min).  Let ΔG = G(p_new) − G(now).

    HOLD        |ΔG| ≤ drift over the next interval (|dG_r/dt|·Δt_next) plus the
                read noise: the write would be undone by drift before it was
                read.  (The Planck-gate rule again: earn the write.)
    SET         ΔG > 0: partial-SET pulses until G reaches the target or the
                per-step cap; the realised ΔG is what the pulses gave, not what
                was asked.
    RESET_SET   ΔG < 0: the only way down is melt-and-quench then re-SET to the
                target.  Cost = RESET energy + k SET pulses + the drift clock
                restarting at t₀ (a fresh amorphous phase drifts fastest).  It
                is taken only when the asked decrease clears the unwind-cost bar
                p* = C_wrong/(C_wrong + C_missed) (cat_residual.pstar), with
                C_wrong the RESET_SET energy and C_missed the energy of
                carrying the error forward (|ΔG| worth of SET later).
    REFRESH     the programmed target is unchanged but drift has carried G
                more than one programming slope from it: re-SET toward target.

    Energy per action is Joule heat ∫I²R dt over the pulse train; it is
    recorded, and it is what the bar is set from.

Coupling into rADAM (the "derivative" in the name)
--------------------------------------------------
rADAM's step is lr·m̂/(√v̂ + ε) in pressure units.  On a PCM cell the achievable
step is the programming slope, which is ~0 in JMAK incubation (x ≈ 0), largest
mid-way and ~0 at saturation (x → 1), and is zero on the way down.  The slope
used is the secant over one full step (max_set_pulses), so incubation does not
deadlock the optimizer and saturation does stop it.  ``coupled_step`` runs
``radam_step`` with

    lr_eff = lr · min(1, (dG/dpulse)_secant / (dG/dpulse)_nominal)

so the optimizer asks for what the cell can give, then applies the decision
above and writes the *realised* pressure (the cell's conductance) back into
the rADAM state.  [v0.58.0: superseded for the axis array by ``ef_step`` —
error feedback on a differential pair, lr unscaled; see the section "The
resolution bound, the differential pair and error feedback" below.  Writing
the realised pressure back erased the sub-resolution part of every step, which
is what stalled the optimizer.]  The optimizer never carries a pressure the cell does not
hold.  With a cell whose slope is nominal everywhere and no drift, this
reduces to ``radam_step`` alone.

Coupling into the UEQGM MESH
----------------------------
* The heating current of axis k is the SOMN Kirchhoff current I_k
  (memristive_axes.currents): the field partition the MESH already computes
  becomes the programming current.  Hot axes crystallise; cold axes drift.
* The Floquet renormalisation J₀(K) of ueqgm_engine.floquet_modulation_factor
  (drive = Weyl phase, ω = the axis's characteristic frequency) multiplies the
  SET rate: a strongly driven coupling crystallises slower, exactly as the
  driven lattice's bandwidth is renormalised.  This is a modelled coupling,
  stated as such; the Weyl phase comes from the stored tensor or, with none,
  from the DES reference (qpsi.weyl_reference).

What this is not
----------------
Cells hold pressures.  They are not edges, not attestations, not decisions.
The module writes only ``entirety:pcm:<instance>`` and the append-only table
``entirety_pcm_log`` (``_kv_set`` refuses any other key); it is an input to no
gate; mesh_slm.py and radam_optimizer.py are not edited.  Stdlib only.

翈 — the cell remembers what the current wrote and forgets at the drift rate;
the optimizer learns only what the cell can hold.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Mapping, MutableMapping, Sequence

from ..radam_optimizer import DEFAULT_LR, radam_step
from .cat_residual import pstar

ENV: str = "QUIPU_PCM_CELLS"
KV_PREFIX: str = "entirety:pcm:"
TABLE: str = "entirety_pcm_log"
K_B: float = 8.617333262e-5          # eV/K

HOLD, SET, RESET_SET, REFRESH = "hold", "set", "reset_set", "refresh"
ACTIONS: tuple[str, ...] = (HOLD, SET, RESET_SET, REFRESH)


# ---------------------------------------------------------------------------
# Configuration: GST order-of-magnitude constants, each with its source
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PcmConfig:
    # Phase boundaries (K).  GST: T_x ≈ 150 °C, T_m ≈ 620 °C (Sebastian et al. 2018).
    t_amb: float = 300.0
    t_x: float = 423.0
    t_m: float = 893.0
    # JMAK crystallisation: Avrami n (2–3 for nucleation-and-growth), Arrhenius
    # τ(T) = τ0·exp(Ea/kT) with Ea ≈ 2 eV (GST, Ielmini & Lacaita, Mater. Today 14, 600 (2011));
    # τ0 chosen so τ(T_x + 150 K) ≈ 50 ns, the usual SET time scale.
    avrami_n: float = 2.5
    e_a: float = 2.0
    tau0: float = 50e-9 / math.exp(2.0 / (K_B * 573.0))
    tau_melt: float = 5e-9           # melting is fast once T ≥ T_m
    # Conductance (S): crystalline/amorphous ratio ≈ 10²–10³; drift ν ≈ 0.1
    # (Ielmini, Lacaita & Mantegazza 2007); t0 = 1 s reference.
    g_set: float = 1e-4
    g_reset0: float = 1e-7
    drift_nu: float = 0.1
    drift_t0: float = 1.0
    # Thermal model: ΔT = P·R_th with R_th ~ 10⁶–10⁷ K/W for a nanoscale cell
    # (Sebastian et al. 2018); τ_th ~ 10 ns.  C_th = τ_th / R_th.
    r_th: float = 5.6e6
    tau_th: float = 10e-9
    # Threshold switching (Ovshinsky 1968): above V_th the amorphous phase conducts
    # in an ON state g_on (~5 kΩ) without crystallising; below it nothing happens.
    v_th: float = 1.0
    g_on: float = 2e-4
    # Pulses (A, s).  SET 93 µA × 50 ns (5 τ_th, so the cell reaches its steady
    # temperature): ΔT = I²/g_on·R_th ≈ 242 K → T ≈ 542 K, between T_x and T_m,
    # where τ ≈ 0.5 µs, so one pulse is a partial SET and ~12–16 pulses saturate.
    # RESET 300 µA × 10 ns: ΔT ≈ 2500 K, well past T_m, then quench.
    i_set: float = 93e-6
    t_set: float = 50e-9
    i_reset: float = 300e-6
    t_reset: float = 10e-9
    # Decision model
    max_set_pulses: int = 16          # per step
    read_noise: float = 0.01          # fraction of (G_max − G_min) the read cannot resolve
    dt_next: float = 79.0             # expected interval to the next read (s): the Entirety's step
    cost_carry_per_pulse: float = 1.0 # weight of a deferred SET pulse vs a spent one in p*

    @classmethod
    def from_env(cls) -> "PcmConfig":
        kw = {}
        for f in cls.__dataclass_fields__:
            raw = os.environ.get("QUIPU_PCM_" + f.upper(), "").strip()
            if not raw:
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if math.isfinite(val) and val >= 0.0:
                kw[f] = int(val) if f == "max_set_pulses" else val
        return cls(**kw)

    def to_json(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cell state and physics
# ---------------------------------------------------------------------------

@dataclass
class PcmCell:
    x: float = 0.0                 # crystalline fraction
    temp: float = 300.0            # K
    t_amorph: float = 1.0          # s since the last RESET (drift clock), ≥ drift_t0
    target_x: float = 0.0          # the pressure the last programming reached (drift is measured from it)
    writes: int = 0
    energy_j: float = 0.0          # cumulative Joule energy spent
    last_action: str = HOLD

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping | None) -> "PcmCell":
        if not isinstance(d, Mapping):
            return cls()
        c = cls()
        for k in cls.__dataclass_fields__:
            if k in d:
                try:
                    setattr(c, k, type(getattr(c, k))(d[k]))
                except (TypeError, ValueError):
                    pass
        return c


def tau_crystallise(temp: float, cfg: PcmConfig) -> float:
    """Arrhenius crystallisation time constant (s)."""
    return cfg.tau0 * math.exp(cfg.e_a / (K_B * max(temp, 1.0)))


def dx_dt(x: float, temp: float, cfg: PcmConfig) -> float:
    """JMAK rate form dx/dt = (n/τ)(1−x)[−ln(1−x)]^((n−1)/n); melt above T_m;
    nothing below T_x.  Reported as a derivative; the integrator below uses the
    additivity rule (equivalent time), which has no singularity at x = 0."""
    x = min(max(x, 0.0), 1.0 - 1e-12)
    if temp >= cfg.t_m:
        return -x / cfg.tau_melt
    if temp <= cfg.t_x:
        return 0.0
    n = cfg.avrami_n
    tau = tau_crystallise(temp, cfg)
    y = -math.log(1.0 - x)
    return (n / tau) * (1.0 - x) * (y ** ((n - 1.0) / n) if y > 0.0 else 0.0)


def _advance_x(x: float, temp: float, h: float, cfg: PcmConfig, floquet: float = 1.0) -> float:
    """One integration substep of JMAK under the additivity rule: the state is
    the equivalent isothermal time t_eq = τ(T)·[−ln(1−x)]^(1/n); the substep
    adds h·floquet to it at the current temperature."""
    x = min(max(x, 0.0), 1.0 - 1e-12)
    if temp >= cfg.t_m:
        return x * math.exp(-h / cfg.tau_melt)
    if temp <= cfg.t_x or floquet <= 0.0:
        return x
    n = cfg.avrami_n
    tau = tau_crystallise(temp, cfg)
    t_eq = tau * ((-math.log(1.0 - x)) ** (1.0 / n)) + h * floquet
    return 1.0 - math.exp(-((t_eq / tau) ** n))


def g_reset(t_amorph: float, cfg: PcmConfig) -> float:
    """Amorphous conductance with drift."""
    return cfg.g_reset0 * (max(t_amorph, cfg.drift_t0) / cfg.drift_t0) ** (-cfg.drift_nu)


def dg_reset_dt(t_amorph: float, cfg: PcmConfig) -> float:
    return -cfg.drift_nu * g_reset(t_amorph, cfg) / max(t_amorph, cfg.drift_t0)


def conductance(cell: PcmCell, cfg: PcmConfig) -> float:
    return cell.x * cfg.g_set + (1.0 - cell.x) * g_reset(cell.t_amorph, cfg)


def dg_dx(cell: PcmCell, cfg: PcmConfig) -> float:
    return cfg.g_set - g_reset(cell.t_amorph, cfg)


def g_bounds(cfg: PcmConfig) -> tuple[float, float]:
    """(G_min, G_max): fresh amorphous at t₀ to fully crystalline.  Fixed, so a
    pressure means the same conductance on every read."""
    return cfg.g_reset0, cfg.g_set


def pressure_of(cell: PcmCell, cfg: PcmConfig) -> float:
    lo, hi = g_bounds(cfg)
    return min(1.0, max(0.0, (conductance(cell, cfg) - lo) / (hi - lo)))


def conductance_of_pressure(p: float, cfg: PcmConfig) -> float:
    lo, hi = g_bounds(cfg)
    return lo + min(1.0, max(0.0, p)) * (hi - lo)


def c_th(cfg: PcmConfig) -> float:
    return cfg.tau_th / cfg.r_th


def switched(cell: PcmCell, current: float, cfg: PcmConfig) -> bool:
    """Threshold switching: the OFF-state voltage I/G exceeds V_th, or the cell is
    crystalline enough to conduct anyway."""
    return current / max(conductance(cell, cfg), 1e-18) >= cfg.v_th or cell.x >= 0.5


def g_conducting(cell: PcmCell, cfg: PcmConfig) -> float:
    """Conductance seen by a pulse once switched: the crystalline path or the ON state."""
    return max(conductance(cell, cfg), cfg.g_on)


def dT_dt(temp: float, current: float, g: float, cfg: PcmConfig) -> float:
    """Thermal RC: heating by I²/g against the thermal resistance, relaxation on τ_th."""
    return (current * current / max(g, 1e-18)) / c_th(cfg) - (temp - cfg.t_amb) / cfg.tau_th


def _integrate_pulse(cell: PcmCell, current: float, duration: float, cfg: PcmConfig,
                     floquet: float = 1.0, steps: int = 40) -> float:
    """Apply one current pulse: thermal RC and JMAK integrated together.
    Returns the Joule energy.  ``floquet`` multiplies the crystallisation rate."""
    if not switched(cell, current, cfg):
        return 0.0                               # below threshold switching: nothing happens
    h = duration / steps
    energy = 0.0
    for _ in range(steps):
        g = g_conducting(cell, cfg)
        energy += current * current / g * h
        cell.temp += h * dT_dt(cell.temp, current, g, cfg)
        cell.x = _advance_x(cell.x, cell.temp, h, cfg, floquet)
    return energy


def cool(cell: PcmCell, duration: float, cfg: PcmConfig) -> None:
    """Field off: thermal relaxation to ambient; a melted cell quenches amorphous."""
    melted = cell.temp >= cfg.t_m
    cell.temp = cfg.t_amb + (cell.temp - cfg.t_amb) * math.exp(-duration / cfg.tau_th)
    if melted:
        cell.x = 0.0
        cell.t_amorph = cfg.drift_t0


def age(cell: PcmCell, dt: float, cfg: PcmConfig) -> None:
    """Time passes without programming: drift clock advances, cell cools."""
    if dt <= 0.0:
        return
    cell.t_amorph += dt
    cool(cell, dt, cfg)


def set_pulse(cell: PcmCell, cfg: PcmConfig, floquet: float = 1.0, current: float | None = None) -> float:
    e = _integrate_pulse(cell, current if current is not None else cfg.i_set, cfg.t_set, cfg, floquet)
    cool(cell, 10.0 * cfg.tau_th, cfg)
    return e


def reset_pulse(cell: PcmCell, cfg: PcmConfig) -> float:
    e = _integrate_pulse(cell, cfg.i_reset, cfg.t_reset, cfg)
    cell.temp = max(cell.temp, cfg.t_m)         # a RESET pulse is defined by reaching the melt
    cool(cell, 10.0 * cfg.tau_th, cfg)
    return e


def dg_dpulse(cell: PcmCell, cfg: PcmConfig, floquet: float = 1.0) -> float:
    """The tangent programming slope: ΔG one SET pulse gives from the current state."""
    probe = replace(cell)
    g0 = conductance(probe, cfg)
    set_pulse(probe, cfg, floquet)
    return conductance(probe, cfg) - g0


def reach(cell: PcmCell, cfg: PcmConfig, floquet: float = 1.0) -> tuple[float, float]:
    """(ΔG over a full step of max_set_pulses, the secant slope ΔG/pulses).  The
    secant is the derivative the optimizer is scaled by: the tangent is ~0 in
    JMAK incubation (x ≈ 0) and ~0 at saturation, the secant is ~0 only at
    saturation, so a fresh cell can be programmed and a full one cannot."""
    probe = replace(cell)
    g0 = conductance(probe, cfg)
    for _ in range(cfg.max_set_pulses):
        set_pulse(probe, cfg, floquet)
    dg = conductance(probe, cfg) - g0
    return dg, dg / cfg.max_set_pulses


_NOMINAL: dict[PcmConfig, float] = {}


def nominal_slope(cfg: PcmConfig) -> float:
    """The largest dG/dpulse on a fresh cell's SET staircase: the slope rADAM's
    lr assumes, so lr_eff = lr·slope/nominal never exceeds lr."""
    v = _NOMINAL.get(cfg)
    if v is None:
        probe = PcmCell(t_amorph=cfg.drift_t0)
        best = 0.0
        for _ in range(4 * cfg.max_set_pulses):
            best = max(best, reach(probe, cfg)[1])
            set_pulse(probe, cfg)
            if probe.x > 0.999:
                break
        v = _NOMINAL[cfg] = best
    return v


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    action: str
    asked_dg: float
    realised_dg: float
    pulses: int
    energy_j: float
    slope: float                  # secant dG/dpulse over a full step, before acting
    drift_next: float             # |dG_r/dt|·dt_next·(1 − x): what drift will do before the next read
    bar: float | None             # p* for RESET_SET, else None
    reason: str

    def to_json(self) -> dict:
        return asdict(self)


def decide(cell: PcmCell, p_target: float, cfg: PcmConfig, *, floquet: float = 1.0,
           dt_next: float | None = None) -> Decision:
    """Choose and apply the action that brings the cell toward p_target.  Mutates cell."""
    dt_next = cfg.dt_next if dt_next is None else dt_next
    g_now = conductance(cell, cfg)
    g_want = conductance_of_pressure(p_target, cfg)
    dg = g_want - g_now
    _, slope = reach(cell, cfg, floquet)
    lo, hi = g_bounds(cfg)
    noise = cfg.read_noise * (hi - lo)
    drift_next = abs(dg_reset_dt(cell.t_amorph, cfg)) * dt_next * (1.0 - cell.x)
    tol = drift_next + noise

    def finish(action, realised, pulses, energy, bar, reason):
        cell.writes += pulses
        cell.energy_j += energy
        cell.last_action = action
        return Decision(action, dg, realised, pulses, energy, slope, drift_next, bar, reason)

    # REFRESH: target unchanged, drift carried the cell more than one slope away.
    g_prog = conductance_of_pressure(cell.target_x, cfg)
    if abs(dg) <= tol and (g_prog - g_now) > max(slope, noise) and cell.x > 0.0:
        pulses, energy = 0, 0.0
        while pulses < cfg.max_set_pulses and conductance(cell, cfg) < g_prog - noise:
            energy += set_pulse(cell, cfg, floquet)
            pulses += 1
        if pulses:
            return finish(REFRESH, conductance(cell, cfg) - g_now, pulses, energy, None,
                          "drift exceeded one programming slope; re-SET toward the programmed target")

    if abs(dg) <= tol:
        return finish(HOLD, 0.0, 0, 0.0, None,
                      f"|dG| {abs(dg):.3g} within drift+noise {tol:.3g} before the next read")

    if dg > 0.0:
        if slope <= 0.0:
            return finish(HOLD, 0.0, 0, 0.0, None, "cell saturated: no SET slope left")
        energy, pulses = 0.0, 0
        while pulses < cfg.max_set_pulses and conductance(cell, cfg) < g_want - noise:
            energy += set_pulse(cell, cfg, floquet)
            pulses += 1
        cell.target_x = pressure_of(cell, cfg)          # the programmed state is what was reached
        return finish(SET, conductance(cell, cfg) - g_now, pulses, energy, None,
                      f"{pulses} partial-SET pulse(s) at slope {slope:.3g}")

    # dg < 0: only RESET then re-SET goes down.  Price it.
    probe = replace(cell)
    e_reset = reset_pulse(probe, cfg)
    k = min(cfg.max_set_pulses, math.ceil(max(0.0, g_want - conductance(probe, cfg)) / max(nominal_slope(cfg), 1e-18)))
    e_set_nom = k * cfg.i_set * cfg.i_set / cfg.g_on * cfg.t_set
    c_wrong = e_reset + e_set_nom
    # carrying |dg| forward costs the SET pulses it would later take to undo it
    c_missed = cfg.cost_carry_per_pulse * (abs(dg) / max(nominal_slope(cfg), 1e-18)) \
        * cfg.i_set * cfg.i_set / cfg.g_on * cfg.t_set
    bar = pstar(c_wrong, c_missed)
    # evidence that the decrease is real: the fraction of it that survives drift
    # and read noise before the next read
    evidence = max(0.0, 1.0 - tol / abs(dg))
    if evidence < bar:
        return finish(HOLD, 0.0, 0, 0.0, bar,
                      f"asked decrease {abs(dg):.3g} does not clear the RESET bar p*={bar:.3f} (evidence {evidence:.3f})")
    energy = reset_pulse(cell, cfg)
    pulses = 1
    while pulses - 1 < cfg.max_set_pulses and conductance(cell, cfg) < g_want - noise:
        energy += set_pulse(cell, cfg, floquet)
        pulses += 1
    cell.target_x = pressure_of(cell, cfg)          # the programmed state is what was reached
    return finish(RESET_SET, conductance(cell, cfg) - g_now, pulses, energy, bar,
                  f"melt-quench then {pulses - 1} SET pulse(s); bar p*={bar:.3f}")


# ---------------------------------------------------------------------------
# rADAM coupling
# ---------------------------------------------------------------------------

def coupled_step(state: MutableMapping[str, float], cell: PcmCell, grad_real: float,
                 grad_imag: float = 0.0, *, cfg: PcmConfig | None = None, lr: float = DEFAULT_LR,
                 floquet: float = 1.0, dt_since: float = 0.0, **radam_kwargs) -> tuple[float, Decision]:
    """One rADAM step whose learning rate is scaled by the cell's programming
    slope, followed by the cell's decision, with the realised pressure written
    back into the rADAM state.  Returns (realised pressure, decision)."""
    cfg = cfg or PcmConfig()
    age(cell, dt_since, cfg)
    nominal = nominal_slope(cfg)
    _, slope = reach(cell, cfg, floquet)
    lr_eff = lr * (min(1.0, slope / nominal) if nominal > 0.0 else 1.0)
    # the optimizer's view of pressure is the cell's, not its own last write
    state["pressure"] = pressure_of(cell, cfg)
    p_asked = radam_step(state, grad_real, grad_imag, lr=lr_eff, **radam_kwargs)
    d = decide(cell, p_asked, cfg, floquet=floquet)
    p_real = pressure_of(cell, cfg)
    state["pressure"] = p_real
    state["pcm_lr_eff"] = lr_eff
    state["pcm_asked"] = p_asked
    return p_real, d


def floquet_for_axis(weyl_phase: float, axis_index: int) -> float:
    """J₀(K) with K = weyl_phase/ω, ω = 1 + axis index (ueqgm_engine's convention),
    folded to [0, 1] as mesh_slm does: a strongly driven axis crystallises slower."""
    try:
        from ..ueqgm_engine import floquet_modulation_factor
        f = floquet_modulation_factor(weyl_phase, 1.0 + axis_index)
    except Exception:
        return 1.0
    return min(1.0, max(0.0, 0.5 * (1.0 + f)))



# ---------------------------------------------------------------------------
# The resolution bound, the differential pair and error feedback (v0.58.0)
# ---------------------------------------------------------------------------
#
# What held the optimizer (v0.57.0)
#   A cell can distinguish ~1/ε levels, ε = (drift before the next read + read
#   noise)/(G_max − G_min): about 100 levels, log2(100) ≈ 6.6 bits per write on
#   the defaults.  An rADAM step smaller than ε carries information the cell
#   cannot represent in one write.  v0.57.0 answered with HOLD and then set the
#   rADAM pressure to the cell's value, so the sub-ε part was erased, not
#   deferred: once the gradient was small every step was below ε and the
#   optimizer stopped.  Decreases were worse: the only way down was RESET,
#   priced by p*, so they were held until the debt grew (and lr was also cut by
#   the slope ratio, shrinking the steps further below ε).  The bound itself is
#   not removable — one write cannot carry more than log2(1/ε) bits — but
#   erasing what does not fit is a choice, not the bound.
#
# What lets it continue
#   1. Error feedback.  The expected location (rADAM's high-precision
#      pressure) and the held location (the cells) are kept separately, and
#      their difference χ = expected − held is stored, not discarded.  The
#      cells are programmed only when |χ| ≥ ε, by the amount that fits; the
#      remainder stays in χ.  Invariant after every step: |χ| < ε + one pulse,
#      so the held value tracks the expected one to within one level, however
#      small each step is.  Steps below ε are delayed, never lost: the
#      optimizer moves on the expected location every step.
#      (Mixed-precision training on PCM: Nandakumar, Le Gallo, Boybat,
#      Rajendran, Sebastian & Eleftheriou, arXiv:1712.01192, Front. Neurosci.
#      14, 406 (2020); error feedback in compressed SGD: Karimireddy et al.,
#      ICML 2019.)
#   2. The differential pair.  The value is held as p = p₊ − p₋ on two cells.
#      Up is a partial SET on the + cell, down is a partial SET on the − cell:
#      both directions are gradual, and RESET is no longer the price of every
#      decrease.  Two devices per synapse: Burr et al., IEDM 2014.
#   3. Refresh.  When the cell that must move has no SET headroom left (its
#      reach over a full step is below what χ asks), both cells are RESET and
#      the net value is re-programmed onto one of them.  This is the only
#      RESET; it is forced by saturation, so it carries no p* bar.
#
# The SQL location: brain_kv entirety:pcm:<instance> stores, per axis, the
# two cells, the expected pressure and χ.  The gap between what rADAM wants and
# what the cells hold is therefore recorded and carried, not overwritten.


def resolution(cell: PcmCell, cfg: PcmConfig, dt_next: float | None = None) -> float:
    """ε in pressure units: the smallest change one write can make that survives
    drift and read noise until the next read."""
    dt_next = cfg.dt_next if dt_next is None else dt_next
    lo, hi = g_bounds(cfg)
    drift = abs(dg_reset_dt(cell.t_amorph, cfg)) * dt_next * (1.0 - cell.x)
    return (drift + cfg.read_noise * (hi - lo)) / (hi - lo)


def bits_per_write(cell: PcmCell, cfg: PcmConfig) -> float:
    """log2(1/ε): the information one write to this cell can carry."""
    return math.log2(1.0 / max(resolution(cell, cfg), 1e-12))


@dataclass
class DiffCell:
    plus: PcmCell = field(default_factory=PcmCell)
    minus: PcmCell = field(default_factory=PcmCell)
    expected: float = 0.0          # rADAM's high-precision pressure: where the value should be
    refreshes: int = 0

    def held(self, cfg: PcmConfig) -> float:
        """The value the cells hold: p₊ − p₋ (may be negative internally)."""
        return pressure_of(self.plus, cfg) - pressure_of(self.minus, cfg)

    def pressure(self, cfg: PcmConfig) -> float:
        return min(1.0, max(0.0, self.held(cfg)))

    def chi(self, cfg: PcmConfig) -> float:
        return self.expected - self.held(cfg)

    def to_json(self) -> dict:
        return {"plus": self.plus.to_json(), "minus": self.minus.to_json(),
                "expected": self.expected, "refreshes": self.refreshes}

    @classmethod
    def from_json(cls, d: Mapping | None) -> "DiffCell":
        if not isinstance(d, Mapping):
            return cls()
        if "plus" not in d and "x" in d:               # a v0.57.0 single cell becomes the + side
            c = cls(plus=PcmCell.from_json(d))
            c.expected = c.plus.target_x
            return c
        return cls(plus=PcmCell.from_json(d.get("plus")), minus=PcmCell.from_json(d.get("minus")),
                   expected=float(d.get("expected") or 0.0), refreshes=int(d.get("refreshes") or 0))


def refresh_pair(dc: DiffCell, cfg: PcmConfig, floquet: float = 1.0) -> float:
    """RESET both cells and re-program the net value onto one of them."""
    net = dc.held(cfg)
    energy = reset_pulse(dc.plus, cfg) + reset_pulse(dc.minus, cfg)
    side = dc.plus if net >= 0.0 else dc.minus
    target = conductance_of_pressure(abs(net), cfg)
    lo, hi = g_bounds(cfg)
    noise = cfg.read_noise * (hi - lo)
    for _ in range(4 * cfg.max_set_pulses):
        if conductance(side, cfg) >= target - noise:
            break
        energy += set_pulse(side, cfg, floquet)
    for c in (dc.plus, dc.minus):
        c.target_x = pressure_of(c, cfg)
        c.last_action = REFRESH
    dc.refreshes += 1
    return energy


def ef_step(state: MutableMapping[str, float], dc: DiffCell, grad_real: float, grad_imag: float = 0.0, *,
            cfg: PcmConfig | None = None, lr: float = DEFAULT_LR, floquet: float = 1.0,
            dt_since: float = 0.0, **radam_kwargs) -> tuple[float, dict]:
    """One rADAM step on the expected location, then error-feedback programming
    of the differential pair.  Returns (held pressure, info)."""
    cfg = cfg or PcmConfig()
    age(dc.plus, dt_since, cfg)
    age(dc.minus, dt_since, cfg)
    # rADAM moves the expected (high-precision) location, at its own lr
    state["pressure"] = min(1.0, max(0.0, dc.expected))
    dc.expected = radam_step(state, grad_real, grad_imag, lr=lr, **radam_kwargs)
    held0 = dc.held(cfg)
    chi0 = dc.expected - held0
    side = dc.plus if chi0 > 0.0 else dc.minus
    eps = resolution(side, cfg)
    action, pulses, energy, reason = HOLD, 0, 0.0, ""
    if abs(chi0) < eps:
        reason = f"|chi| {abs(chi0):.4f} < eps {eps:.4f}: deferred in chi, not lost"
    else:
        reach_g, _ = reach(side, cfg, floquet)
        lo, hi = g_bounds(cfg)
        if reach_g / (hi - lo) < min(abs(chi0), 1.0) - eps and pressure_of(side, cfg) > 0.5:
            energy += refresh_pair(dc, cfg, floquet)
            action, pulses = REFRESH, 1
            side = dc.plus if (dc.expected - dc.held(cfg)) > 0.0 else dc.minus
            reason = "SET headroom exhausted on the moving cell: both RESET, net re-programmed"
        want = dc.expected - dc.held(cfg)
        if abs(want) >= resolution(side, cfg):
            d = decide(side, pressure_of(side, cfg) + abs(want), cfg, floquet=floquet)
            energy += d.energy_j
            pulses += d.pulses
            if d.action != HOLD:
                action = SET if action == HOLD else action
                reason = (reason + "; " if reason else "") + \
                    f"{d.pulses} SET pulse(s) on the {'+' if side is dc.plus else '-'} cell"
            elif not reason:
                reason = d.reason
    held1 = dc.held(cfg)
    chi1 = dc.expected - held1
    state["pressure"] = dc.expected
    state["pcm_chi"] = chi1
    state["pcm_held"] = held1
    state["pcm_lr_eff"] = lr
    state["pcm_asked"] = dc.expected
    info = {"action": action, "pulses": pulses, "energy_j": energy, "eps": eps,
            "bits_per_write": math.log2(1.0 / max(eps, 1e-12)),
            "expected": dc.expected, "held": held1, "chi_before": chi0, "chi": chi1,
            "realised": held1 - held0, "plus_x": dc.plus.x, "minus_x": dc.minus.x,
            "refreshes": dc.refreshes, "reason": reason}
    return min(1.0, max(0.0, held1)), info


# ---------------------------------------------------------------------------
# The cellular RAM: one cell per MESH axis (and any named cell set)
# ---------------------------------------------------------------------------

AXES: tuple[str, ...] = ("vision", "touch", "smell", "body", "brain", "perception", "entirety")


@dataclass
class CellArray:
    cells: dict[str, DiffCell] = field(default_factory=dict)
    radam: dict[str, dict] = field(default_factory=dict)
    last_t: float | None = None
    steps: int = 0

    def cell(self, name: str) -> DiffCell:
        if name not in self.cells:
            self.cells[name] = DiffCell()
        return self.cells[name]

    def to_json(self) -> dict:
        return {"cells": {k: c.to_json() for k, c in self.cells.items()},
                "radam": self.radam, "last_t": self.last_t, "steps": self.steps}

    @classmethod
    def from_json(cls, d: Mapping | None) -> "CellArray":
        a = cls()
        if isinstance(d, Mapping):
            for k, v in (d.get("cells") or {}).items():
                a.cells[k] = DiffCell.from_json(v)
            a.radam = {k: dict(v) for k, v in (d.get("radam") or {}).items() if isinstance(v, Mapping)}
            a.last_t = d.get("last_t")
            a.steps = int(d.get("steps") or 0)
        return a


def step_array(arr: CellArray, *, currents: Mapping[str, float], targets: Mapping[str, float],
               grads: Mapping[str, float] | None = None, weyl_phase: float = 0.0, now: float,
               cfg: PcmConfig | None = None) -> dict:
    """Advance every axis cell one Entirety step.

    ``currents``: SOMN Kirchhoff currents (documents per pulse) → programming
    current scale; ``targets``: the pressure each axis should hold (the live
    axis value); ``grads``: optional gradient per axis for rADAM (default:
    target − held).  Returns per-axis decisions.
    """
    cfg = cfg or PcmConfig()
    dt = 0.0 if arr.last_t is None else max(0.0, now - float(arr.last_t))
    total_i = sum(max(0.0, float(v)) for v in currents.values()) or 1.0
    out: dict[str, dict] = {}
    for i, a in enumerate(AXES):
        dc = arr.cell(a)
        st = arr.radam.setdefault(a, {})
        share = max(0.0, float(currents.get(a, 0.0))) / total_i
        fl = floquet_for_axis(weyl_phase, i)
        # the axis's share of the field sets how hard it is driven (lr), not the
        # pulse current: a pulse below V_th does nothing, so current stays physical
        target = float(targets.get(a, 0.0))
        g = float(grads[a]) if grads and a in grads else (target - dc.expected)
        p, info = ef_step(st, dc, g, cfg=cfg, lr=DEFAULT_LR * max(share * len(AXES), 0.05),
                          floquet=fl, dt_since=dt)
        out[a] = {"pressure": round(p, 6), "expected": round(info["expected"], 6),
                  "chi": round(info["chi"], 6), "eps": round(info["eps"], 6),
                  "bits_per_write": round(info["bits_per_write"], 3),
                  "plus_x": round(dc.plus.x, 6), "minus_x": round(dc.minus.x, 6),
                  "x": round(dc.plus.x - dc.minus.x, 6),
                  "share": round(share, 6), "floquet": round(fl, 6),
                  "lr_eff": round(float(st.get("pcm_lr_eff", 0.0)), 6),
                  "decision": {"action": info["action"], "asked_dg": info["chi_before"],
                               "realised_dg": info["realised"], "pulses": info["pulses"],
                               "energy_j": info["energy_j"], "slope": None, "drift_next": None,
                               "bar": None, "reason": info["reason"]}}
    arr.last_t = now
    arr.steps += 1
    return out


# ---------------------------------------------------------------------------
# Persistence (observability only)
# ---------------------------------------------------------------------------

def ensure_tables(cn: sqlite3.Connection) -> None:
    cn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE}("
               "id INTEGER PRIMARY KEY AUTOINCREMENT, instance TEXT NOT NULL, at REAL NOT NULL, "
               "step INTEGER NOT NULL, axis TEXT NOT NULL, action TEXT NOT NULL, asked_dg REAL, "
               "realised_dg REAL, pulses INTEGER, energy_j REAL, slope REAL, drift_next REAL, "
               "bar REAL, x REAL, pressure REAL, reason TEXT)")
    have = {r[1] for r in cn.execute(f"PRAGMA table_info({TABLE})")}
    for col in ("expected", "chi", "eps"):
        if col not in have:
            cn.execute(f"ALTER TABLE {TABLE} ADD COLUMN {col} REAL")
    cn.execute("CREATE TABLE IF NOT EXISTS brain_kv(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")


def _kv_set(cn: sqlite3.Connection, key: str, value) -> None:
    if not key.startswith(KV_PREFIX):
        raise PermissionError(f"pcm_cells writes only {KV_PREFIX}* keys, not {key!r}")
    cn.execute("INSERT OR REPLACE INTO brain_kv(key, value, updated_at) VALUES(?,?,?)",
               (key, json.dumps(value, default=str), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def load_array(cn: sqlite3.Connection, instance: str = "system_entirety") -> CellArray:
    ensure_tables(cn)
    row = cn.execute("SELECT value FROM brain_kv WHERE key=?", (KV_PREFIX + instance,)).fetchone()
    if not row:
        return CellArray()
    try:
        return CellArray.from_json(json.loads(row[0]))
    except (TypeError, ValueError):
        return CellArray()


def record(cn: sqlite3.Connection, arr: CellArray, result: Mapping[str, Mapping], *,
           now: float, instance: str = "system_entirety") -> None:
    ensure_tables(cn)
    _kv_set(cn, KV_PREFIX + instance, arr.to_json())
    for a, r in result.items():
        d = r["decision"]
        cn.execute(f"INSERT INTO {TABLE}(instance, at, step, axis, action, asked_dg, realised_dg, pulses, "
                   "energy_j, slope, drift_next, bar, x, pressure, reason, expected, chi, eps) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (instance, now, arr.steps, a, d["action"], d["asked_dg"], d["realised_dg"], d["pulses"],
                    d["energy_j"], d["slope"], d["drift_next"], d["bar"], r["x"], r["pressure"], d["reason"],
                    r.get("expected"), r.get("chi"), r.get("eps")))


def rows(cn: sqlite3.Connection, limit: int = 64, instance: str = "system_entirety") -> list[dict]:
    ensure_tables(cn)
    cols = ["at", "step", "axis", "action", "asked_dg", "realised_dg", "pulses", "energy_j", "slope",
            "drift_next", "bar", "x", "pressure", "reason", "expected", "chi", "eps"]
    q = cn.execute(f"SELECT {', '.join(cols)} FROM {TABLE} WHERE instance=? ORDER BY id DESC LIMIT ?",
                   (instance, int(limit)))
    return [dict(zip(cols, r)) for r in q.fetchall()]


__all__ = ["ENV", "KV_PREFIX", "TABLE", "ACTIONS", "HOLD", "SET", "RESET_SET", "REFRESH", "AXES",
           "PcmConfig", "PcmCell", "Decision", "CellArray", "tau_crystallise", "dx_dt", "g_reset",
           "dg_reset_dt", "conductance", "dg_dx", "pressure_of", "conductance_of_pressure", "dT_dt",
           "c_th", "switched", "g_conducting", "age", "cool", "set_pulse", "reset_pulse", "dg_dpulse", "reach", "nominal_slope", "decide",
           "coupled_step", "resolution", "bits_per_write", "DiffCell", "refresh_pair", "ef_step", "floquet_for_axis", "step_array", "ensure_tables", "load_array", "record", "rows"]
