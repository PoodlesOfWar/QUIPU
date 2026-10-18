# QPSI — Chalcogenide phase-change cells (the cellular RAM behind r-ADMIN)

`src/quipu/qpsi/pcm_cells.py`, v0.57.0. Flag `QUIPU_PCM_CELLS` (default on with
`QUIPU_SELF_ORGANISING=1` and the SOMN on). Writes only `entirety:pcm:<instance>`
and `entirety_pcm_log`. Input to no gate.

## The cell

A phase-change memory cell is a chalcogenide volume (Ge₂Sb₂Te₅) whose conductance
is set by the crystalline fraction x:

    G(x, t) = x·G_set + (1 − x)·G_reset(t),   G_reset(t) = G_reset,0 (t/t₀)^(−ν)

| Mechanism | Model | Source |
|---|---|---|
| SET (crystallise) | JMAK, x = 1 − exp(−(t/τ(T))ⁿ), Arrhenius τ(T), integrated by the additivity rule | Sebastian et al., J. Appl. Phys. 124, 111101 (2018); Ielmini & Lacaita, Mater. Today 14, 600 (2011) |
| RESET (amorphise) | melt above T_m, quench to x = 0, drift clock restarts | same |
| Drift | R ∝ t^ν, ν ≈ 0.1 | Ielmini, Lacaita & Mantegazza, IEEE TED 54, 308 (2007) |
| Threshold switching | conducts in an ON state above V_th, else nothing | Ovshinsky, PRL 21, 1450 (1968) |
| Heat | thermal RC, ΔT = I²/g·R_th, R_th ~ 10⁶–10⁷ K/W | Sebastian et al. 2018 |

Defaults (`PcmConfig`, every field overridable as `QUIPU_PCM_<FIELD>`): T_x 423 K,
T_m 893 K, n 2.5, E_a 2 eV, τ(573 K) = 50 ns, G_set 10⁻⁴ S, G_reset,0 10⁻⁷ S, ν 0.1,
R_th 5.6·10⁶ K/W, τ_th 10 ns, V_th 1 V, g_on 2·10⁻⁴ S, SET 93 µA × 50 ns (steady
T ≈ 542 K, τ ≈ 0.5 µs: one pulse is a partial SET, 12–16 saturate), RESET 300 µA × 10 ns.

## The derivatives

    dx/dt   = (n/τ(T))(1 − x)[−ln(1 − x)]^((n−1)/n)   T_x < T < T_m;  −x/τ_melt above T_m;  0 below T_x
    dG/dx   = G_set − G_reset(t)
    dG_r/dt = −ν G_reset(t)/t
    dT/dt   = I²/g / C_th − (T − T_amb)/τ_th
    dG/dpulse: tangent (one pulse) and secant (one full step of max_set_pulses)

## The decision model (one rADAM step → one action)

rADAM proposes p_new; the cell holds p = (G − G_min)/(G_max − G_min); ΔG = G(p_new) − G.

| Action | When | What |
|---|---|---|
| HOLD | \|ΔG\| ≤ drift over the next interval + read noise | nothing: the write would be undone before it was read |
| SET | ΔG > 0 | partial-SET pulses until the target or the per-step cap; the realised ΔG is reported |
| RESET_SET | ΔG < 0 and the decrease clears p* = C_wrong/(C_wrong + C_missed) | melt-quench, then re-SET toward the target; C_wrong = RESET + re-SET energy, C_missed = the SET energy of carrying the error forward; evidence = the fraction of ΔG that survives drift + noise |
| REFRESH | target unchanged, drift carried G more than one slope below it | re-SET toward the programmed state |

Energy is ∫I²R dt over the pulse train; it is what the bar is set from.

## Coupling to rADAM

`coupled_step` runs `radam_step` with `lr_eff = lr · min(1, secant/nominal)` (nominal =
the largest secant on a fresh cell's staircase), applies the decision, and writes the
*realised* pressure back into the rADAM state. The optimizer never carries a pressure
the cell does not hold. The secant, not the tangent, is used so that JMAK incubation
(x ≈ 0) does not deadlock the optimizer while saturation (x → 1) does stop it.

## Coupling to the UEQGM MESH

* Heating current of axis k = the SOMN Kirchhoff current I_k (`memristive_axes.currents`),
  as a share of the field: the partition the MESH already computes becomes the
  programming drive.
* The Floquet factor J₀(K) (`ueqgm_engine.floquet_modulation_factor`, K = Weyl phase/ω,
  ω = 1 + axis index) multiplies the SET rate: a strongly driven axis crystallises slower.
  The Weyl phase is 2π·Ψ₄ from `mesh_slm._weyl_tensor()` (the DES reference when nothing
  is stored). This is a modelled coupling, stated as such.
* Targets: the live axis values and the observer (entirety).

## Boundaries

Cells hold pressures; they are not edges, attestations or decisions. No hardware is
claimed. `mesh_slm.py` and `radam_optimizer.py` are not edited.

翈 — the cell remembers what the current wrote and forgets at the drift rate; the
optimizer learns only what the cell can hold.

## v0.58.0 — the resolution bound, error feedback and the differential pair

### What held the optimizer

One write can make a change of no less than ε = (drift before the next read + read noise)
/ (G_max − G_min) and still have it read back: about 0.010 in pressure, so ~100 levels,
log2(100) ≈ 6.6 bits per write on the defaults (≈ 5.8 bits on a fresh amorphous cell,
which drifts fastest). That limit is physical and stays. What v0.57.0 added on top of it
was erasure: a step below ε was HELD and rADAM's pressure was then overwritten with the
cell's value, so the sub-ε part of every step was thrown away. Once the gradient was small,
every step was below ε and the optimizer stopped. Decreases were worse, because the only
way down was RESET behind the p* bar, and lr was also cut by the slope ratio, pushing steps
further below ε. Measured: 34 of 40 small steps held, and the cell stuck at 0.944 with a target of 0.6.

### What lets it continue

1. **Error feedback.** rADAM steps on the *expected* location (high precision). The cells
   hold the *held* location. χ = expected − held is stored with them in
   `entirety:pcm:<instance>` and in `entirety_pcm_log`, and nothing overwrites it. The cells
   are programmed only when |χ| ≥ ε, by the amount that fits; the rest stays in χ. After
   every step |χ| < ε plus one pulse, so the held value tracks the expected one to within
   one level, however small the steps are. Steps below ε are delayed, not lost.
   (Nandakumar, Le Gallo, Boybat, Rajendran, Sebastian & Eleftheriou, arXiv:1712.01192 /
   Front. Neurosci. 14, 406 (2020); Karimireddy et al., ICML 2019.)
2. **The differential pair.** p = p₊ − p₋ on two cells. Up is a partial SET on +, down is a
   partial SET on −: both directions are gradual and a decrease no longer needs RESET.
   (Two devices per synapse: Burr et al., IEDM 2014.)
3. **Refresh.** When the cell that has to move has no SET headroom left, both cells are
   RESET and the net value is re-programmed onto one of them. This is the only RESET. It
   is forced by saturation, so no bar applies to it.
4. **No slope scaling of lr.** With χ carrying the remainder, lr_eff = lr. The device's
   granularity is absorbed by χ, not by shrinking the optimizer's steps.

### Measured on the defaults

| | v0.57.0 single cell | v0.58.0 pair + χ |
|---|---|---|
| Ramp 0.1 → 0.3 → 0.1 over 200 steps, mean \|target − held\| | 0.0265 | 0.0064 |
| End of ramp, held vs target 0.102 | 0.059 | 0.097 |
| \|χ\| after warm-up | not kept | < 0.026 (one level + one pulse) |

`coupled_step` (single cell, slope-scaled) is kept for reference. The axis array uses `ef_step`.
