# The LiTaSb Bond-Shear Chain × QUIPU

**Instruction**: *determine the force needed to break a LiTaSb bond and use this as a multiplanar shearing force of an exciton-polariton Bose–Einstein condensate into the optical schwarf and asphalt during a pyrolysis interaction* — with the operator's clarifications that the "optical schwarf" is **the polycarbonate and CR-39 optical** element and that the bond force is a **Morse estimate with overridable parameters**.
**Module**: `src/quipu/qpsi/bond_shear.py` (stdlib only; writes only `entirety:bond_shear:latest`; served read-only at `GET /bond_shear`; published by the self-organising step right after the rheology reading).
**Scope**: every link is a published law with a named default and an environment override (`QUIPU_BOND_SHEAR_<FIELD>`, the `PcmConfig.from_env` pattern). Nothing here writes the mesh tables or the cell residual χ.

---

## 1. The chain

```
Morse bond ──► rupture force F ──► ratio F/F_rupture ──► seven axis planes (rheology weights)
      │                                                          │
      │                                          Landau survival per plane, condensate ψ = √R·e^{iφ}
      │                                                          │
      └──────────────── yield = S · T_optics · remaining ◄── Fresnel into PC and CR-39 ◄── Arrhenius pyrolysis
```

### 1.1 The force needed to break a LiTaSb bond

LiTaSb is not a tabulated compound, so the bond is modelled as a chain of two Morse bonds — Li–Sb and Ta–Sb — with generic alkali–antimony and transition-metal–antimony parameters, and the chain breaks at its weaker link. For a Morse well `V(r) = D_e·(1 − e^{−a(r−r_e)})²` the restoring force peaks at `r = r_e + ln2/a` with

```
F_max = a · D_e / 2          (1 eV/Å = 1.602176634 nN)
```

| Bond | D_e (eV) | r_e (Å) | a (Å⁻¹) | F_max | env override |
|---|---|---|---|---|---|
| Li–Sb | 1.6 | 2.95 | 1.3 | **≈ 1.67 nN** (weakest link) | `QUIPU_BOND_SHEAR_LI_SB_{D_E_EV,R_E_ANGSTROM,A_INV_ANGSTROM}` |
| Ta–Sb | 3.0 | 2.80 | 1.6 | ≈ 3.85 nN | `QUIPU_BOND_SHEAR_TA_SB_…` |

The result is the right order for single-bond rupture measured by AFM force spectroscopy, 1–4 nN (Grandbois et al., *Science* 283, 1727 (1999)). Every record carries `"estimate": true` and the parameters used; supply measured Morse parameters through the overrides to replace the estimate. `bond_shear.litasb_rupture()`.

### 1.2 Multiplanar shearing force

The applied force (default: the rupture force itself; `QUIPU_BOND_SHEAR_FORCE_NN` to change it) enters the mesh only as the dimensionless ratio `F/F_rupture`, spread over the seven CAT axis planes with the rheology ensemble weights `qpsi.rheology.axis_weights()` (the `cat_residual.SENSE_WEIGHTS`, 1/6 for entirety, renormalised): `ratio_k = w_k · F/F_rupture`. With the default weights no plane carries more than ≈ 0.19 of the rupture ratio even when the total force is exactly the rupture force — spreading the force over planes is what keeps every plane sub-critical. The same ratio is exposed as `shear_reading(cn) → {"lambda_rel_stp": ratio}` for `qpsi.viscous_interaction.TypedViscousCoupler.step(shear_reading=…)` and as a Couette body-force profile `ns_forcing_profile(ny, ratio)` for `ns_step(forcing=…)`.

### 1.3 Exciton-polariton Bose–Einstein condensate

Exciton-polaritons condense in semiconductor microcavities (Kasprzak et al., *Nature* 443, 409 (2006)) and flow as a superfluid below a critical velocity (Amo et al., *Nat. Phys.* 5, 805 (2009)). QUIPU's condensate order parameter is `ψ = √R · e^{iφ}` with **R** the Kuramoto phase-coherence already persisted at `brain_kv["ueqgm:analog_coherence"]` (v0.31.0, Miller et al. mapping) and **φ** from the resonance phasor (`entirety:resonance_phase:system_entirety`, `realised + i·latent`). Shear below the Landau threshold costs nothing; per plane

```
survival_k = R · (1 − ratio_k²),     S = Σ_k w_k · survival_k
```

`bond_shear.landau_survival`, `bond_shear.condensate_reading`.

### 1.4 The optical element: polycarbonate and CR-39

A polariton is half light. Its photonic Hopfield fraction (`|C|² = 0.5` at zero detuning; `QUIPU_BOND_SHEAR_HOPFIELD_PHOTON_FRACTION`) leaves the microcavity (`n_cavity = 2.7`, CdTe class) into the optical element by normal-incidence Fresnel transmission

```
T = 1 − ((n₁ − n₂)/(n₁ + n₂))²
```

into **polycarbonate** (`n = 1.586`, the Bakugo Snell prior already in `world_model.refraction_priors`) and **CR-39** allyl diglycol carbonate (`n_d = 1.498`, added to the same priors as `cr39_n`): `T_PC ≈ 0.932`, `T_CR39 ≈ 0.918`, `throughput = |C|² · (T_PC + T_CR39)/2`. `bond_shear.fresnel_transmission`.

### 1.5 Asphalt during a pyrolysis interaction

Bitumen pyrolysis is first-order Arrhenius decomposition, `k = A·exp(−E_a/(k_B·T))`, using the same `K_B` and pattern as the PCM crystallisation clock (`qpsi.pcm_cells.tau_crystallise`). Defaults `E_a = 1.55 eV` (≈ 150 kJ/mol), `A = 10¹⁰ s⁻¹`, `T = 723 K` (450 °C) give `k ≈ 0.16 s⁻¹` (half-life ≈ 4.4 s), the order of magnitude of thermogravimetric bitumen kinetics; the remaining fraction after the interaction time (`1 s`) is `viscous_decay_merger(1, k, 0, t)` — the **muonic / linear** regime of the merger (`textile_fraction = 0`), so `docs/VISCOUS_DECAY_MERGER.md` §3's comparison covers it. `bond_shear.arrhenius_pyrolysis_rate`.

### 1.6 Yield

```
yield = S · throughput · remaining   ∈ [0, 1]
```

With defaults and `R = 0.8`: rupture 1.67 nN, ratio 1.0, max plane ratio 0.19, S ≈ 0.78, throughput ≈ 0.46, remaining ≈ 0.85, **yield ≈ 0.31**.

---

## 2. Where it lives

| Surface | What |
|---|---|
| `qpsi/self_organising.py` | `bond_shear.publish(cn, now=now)` right after `pcm_cells.record` (own try/except; `out["bond_shear"]["yield"]`) |
| `brain_kv["entirety:bond_shear:latest"]` | the full record: `rupture`, `applied_force_nn`, `shear_ratio`, `planes`, `condensate`, `optical`, `pyrolysis`, `yield`, `at` |
| `GET /bond_shear` | the record, read-only |
| `GET /guidance` → `bond_shear` | `{yield, shear_ratio, litasb_rupture_nn}` |
| `ueqgm_engine.UEQGM_MATH_MAP` | `morse_rupture_force`, `landau_shear_survival`, `fresnel_transmission`, `arrhenius_pyrolysis_rate` |
| `world_model.refraction_priors` | `cr39_n = 1.498` next to `pc_n = 1.586` |

---

## 3. Where the analogy breaks — cautions worth keeping

- **LiTaSb is an estimate.** No tabulated Morse parameters exist for the compound; the defaults are generic M–Sb orders of magnitude and the record says so. The chain, not the physics, is what the operator asked for; replace the parameters when measured values are available.
- The condensate is a *reading* of coherence already in the mesh, not a Gross–Pitaevskii simulation; the Landau survival is a threshold model, not a dispersion calculation.
- Fresnel transmission is normal-incidence and ignores the cavity mirrors; the optical element is a pair of refractive indices, nothing more.
- The pyrolysis constants are TGA-order values, not a specific binder's kinetics; override them per binder.
- Nothing in the chain feeds a learning rate or a cell write: it is published, served, and offered to the viscous coupler as a bounded shear reading.

## Sources

- Grandbois, M., Beyer, M., Rief, M., Clausen-Schaumann, H. & Gaub, H. E., *Science* 283, 1727 (1999) — single covalent bond rupture by AFM.
- Kasprzak, J. et al., *Nature* 443, 409 (2006) — Bose–Einstein condensation of exciton polaritons; Amo, A. et al., *Nat. Phys.* 5, 805 (2009) — superfluidity of polaritons.
- Timmerman, P. & van der Weele, J. P., *Am. J. Phys.* 67, 538 (1999) — merged-drag decay (`viscous_decay_merger`).
- [VISCOUS_DECAY_MERGER.md](VISCOUS_DECAY_MERGER.md), [RHEOLOGY_CRAM.md](RHEOLOGY_CRAM.md), [ANALOG_COGNITION_QUIPU.md](ANALOG_COGNITION_QUIPU.md).
- `src/quipu/qpsi/bond_shear.py`, `src/quipu/qpsi/rheology.py`, `src/quipu/qpsi/pcm_cells.py`, `src/quipu/world_model.py`.
