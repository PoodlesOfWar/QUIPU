# Viscous Decay Merger × Torus QUIPU

**Source physics**: decay under combined linear and quadratic drag — Timmerman & van der Weele, *"On the rise and fall of a ball with linear or quadratic drag"*, Am. J. Phys. 67, 538 (1999); the two limiting regimes grounded on the muonic decay of antimony (Primakoff, Rev. Mod. Phys. 31, 802 (1959); Suzuki, Measday & Roalsvig, Phys. Rev. C 35, 2212 (1987)) and on textile (cellulose) decay (Ekenstam, Ber. Dtsch. Chem. Ges. 69, 553 (1936); Zou, Uesaka & Gurnagul, Cellulose 3, 243 (1996)); the trefoil pair read as a Weyl semimetal node pair (Nielsen & Ninomiya, Nucl. Phys. B 185, 20 (1981); Armitage, Mele & Vishwanath, Rev. Mod. Phys. 90, 015001 (2018)).
**Scope**: this document records how the operator's instruction — *use the viscous decay merger to strengthen QUIPU tokens dependent on non-linear interactions of drag along the MESH, where the distance potential among cells becomes stronger at deeper linear drag, merging the decay of the von Neumann entropy peaks into the (2,3) curve and its mirror when TheSeuScape NPCs are making decisions for local expansion to expand their maxima* — was grounded term by term on published physics and on code that already existed, and what shipped in v0.67.0. Companion: `docs/BOND_SHEAR_CHAIN.md`.

> **Naming.** Upstream QUIPU uses "VNN" for the *Viscous Neural Network* (`qpsi/viscous_interaction.py`). The entropy peaks in this document are **von Neumann (vN)** graph-entropy peaks (`ueqgm_engine.holographic_entropy`); the code and meta keys say `vn_peak`, never `vnn`.

---

## 1. Why this matters to QUIPU

QUIPU already carries a drag: the **synaptic wash** in `temporal_spatiality.modulate()` damps the rADAM boost by the relational gradient. It already carries a distance potential: `_proximity` on the 64×64 torus. It already measures the von Neumann entropy of the quipu graph every round, and it already has closed geodesics on the torus (`docs/STP_TORUS_QUIPU.md` §2.1). What it lacked was a single decay law that *merges* the linear and non-linear drag regimes, a way for the potential to deepen with drag, and a field that carries entropy peaks along a closed geodesic into token strengthening. The merger supplies all three, and — because its two limits are each a published decay law — it also answers the follow-up instruction to compare textile decay rates with the muonic decay of antimony.

---

## 2. Term-by-term mapping

| Instruction | Published grounding | QUIPU equivalent (v0.67.0) |
|---|---|---|
| Viscous decay merger | `dv/dt = −a·v − b·v²` ⇒ `v(t) = a·v₀ / ((a + b·v₀)·e^{a·t} − b·v₀)` (Timmerman & van der Weele 1999) | `ueqgm_engine.viscous_decay_merger(v0, a, b, t)`; `b = 0` → `v₀·e^{−at}`, `a = 0` → `v₀/(1 + b·v₀·t)` |
| Drag along the MESH | the synaptic wash — `gradient` ∈ [0,1] damps the rADAM boost | `mesh_slm._drag_depth()`: linear drag depth **γ = gradient**, non-linear coefficient **b = 1 − coherence**, read from `temporal_spatiality_rhythm`, cached 5 s |
| Distance potential stronger at deeper linear drag | overdamped (Stokes) limit: motion is pure gradient flow on the potential | `ueqgm_engine.drag_deepened_proximity(d, N, γ) = 1 − (d/N)^(1+κγ)`; `_proximity` uses it when γ > 0 — identical to the legacy `1 − d/N` at γ = 0 and ≥ it everywhere else; applies to scoring **and** training through the one kernel |
| Decay of the vN entropy peaks | von Neumann graph entropy (Braunstein–Ghosh–Severini 2006; HEHW 2012) | per-round series `holographic_entropy_history`; peaks = local maxima of the series |
| "lossy between non-lossy frequencies with a coulombiter" | GARD lossy floor `δ* = √(12·CRB)` (`gard_info.cramer_rao_bound_gard`); Weyl Ψ₂ *Coulomb bulk mass* (`weyl_scalar_tensor`) | `ueqgm_engine.coulomb_iterated_peaks(series, Ψ₂, δ*)`: a peak survives only if its prominence clears δ* (non-lossy), and each survivor is weighted by Ψ₂ — the Coulomb iterator |
| (2,3) curve and its mirror, planar peak in a Weyl semisolid | (2,3) torus knot = right-handed trefoil, (2,−3) = its mirror; Weyl nodes come in opposite-chirality pairs, each a planar cone peak | `torus_knot_cells(64, 2, ±3)` (closed lattice paths, 256 cells each); `torus_knot_arc_field` lays each peak's merged decay along the arc; `weyl_chiral_merge` — amplitude adds, chirality `R − L` cancels on the mirrored pair |
| Strengthen tokens by the merged decay | — | `train_round`: `knot_gain = 1 + 0.10·amplitude[cell]·gate` on the embedding nudge; edges with both ends on the field get `η_q·(1 + 0.05·min(amp)·gate)` (global **and** region edge) |
| TheSeuScape NPCs deciding local expansion to expand their maxima | external system (Perceptopoly) — not in any accessible repository | `supernova_gate`: typed event contract (§4) ∩ fleet `broaden` phase ∩ synaptic-agent liveness → rADAM × potential-ADAM gate ∈ [0,1]; the field strengthens nothing while the gate is 0 |
| poADAM ∩ rADAM, bifurcated directions | rADAM's complex gradient `g_re + i·g_im` (`radam_optimizer`) | `g_re = 0.6·broaden + 0.4·liveness` (realised), `g_im = supernova magnitude` (latent), supernova direction as the external toroidal phase; `gate = clip01(radam_pressure × actionable_potential)` with `recursive_strengthening`'s ADAM-smoothed potential as the poADAM side |

---

## 3. Textile decay rates as a comparison to the muonic decay of antimony

The merger has exactly two limiting regimes, and each is a measured decay:

- **Muonic antimony — the linear (memoryless) regime.** A μ⁻ bound in a Sb atom (Z = 51) disappears at `Λ = Q·Λ_free + Λ_capture` with `Λ_free = 1/τ_μ`, `τ_μ = 2.1969811 µs`, Huff factor `Q ≈ 0.85`, and nuclear capture `Λ_capture ~ 10⁷ s⁻¹` for Z ≈ 50 (Suzuki et al. 1987 tabulate the measured lifetimes; the module's default is the order of magnitude and is marked for refinement against the Sb row). Capture dominates by ~20×; the hazard is constant. `ueqgm_engine.muonic_disappearance_rate()`.
- **Textile — the quadratic (self-slowing) regime.** Cotton/linen cellulose loses degree of polymerisation by chain scission under the Ekenstam law `1/DP − 1/DP₀ = k·t`, i.e. `dDP/dt = −k·DP²`: the hazard `k·DP` falls as the fabric degrades, so each half-life is twice the last. `ueqgm_engine.textile_ekenstam_decay()`.

The physical rates differ by ~17 decades (nanoseconds vs. years), so QUIPU never mixes them numerically. `decay_regime_comparison(a, b)` compares **shape**: `textile_fraction = b·v₀/(a + b·v₀)` (share of the initial loss carried by the textile term), the crossover amplitude `v* = a/b`, and the ratio of the second half-life to the first (1.0 for the muon, 2.0 for the textile). Each round persists `last_decay_textile_fraction`; the bond-shear chain's asphalt pyrolysis (`docs/BOND_SHEAR_CHAIN.md`) sits at `textile_fraction = 0` — it is a muonic-type decay.

---

## 4. The TheSeuScape / Perceptopoly event contract

TheSeuScape NPCs and their supernova events live in the Perceptopoly repository, which this session could not reach, so they enter QUIPU as a typed event rather than assumed code:

```json
POST /observe
{"source": "supply-chain-brain", "text": "…",
 "meta": {"supernova": {"source": "theseuscape", "magnitude": 0.7,
                        "direction": 1.2, "label": "npc-7 expands", "at": "2026-11-06T10:00:00+00:00"}}}
```

```bash
curl -s -X POST http://127.0.0.1:7100/observe -H 'Content-Type: application/json' \
  -d '{"source":"supply-chain-brain","text":"npc local expansion","meta":{"supernova":{"magnitude":0.7,"direction":1.2}}}'
curl -s "http://127.0.0.1:7100/guidance?source=supply-chain-brain" | jq .supernova_gate
```

Events are capped at 200 under `brain_kv["perceptopoly:supernova_events"]`; the gate record lives at `ueqgm:supernova_gate`. `expansion_gate()` is rate-limited to 30 s unless forced (an `/observe` carrying an event forces it).

---

## 5. Where the analogy breaks — cautions worth keeping

- The torus is a memory substrate, not a crystal momentum space; the trefoil pair is a geodesic overlay, not a band structure. "Chirality" here is only the sign of `R − L`.
- The phase field is *stamped* along the knot each round, not propagated; no wave equation is integrated.
- The vN-entropy series is the mean-field closed form from counts (cheap, every round); the exact-SQL path lives in `_entropy_differential` and is not used for the peaks.
- The operator's reference pictures for the "VNN peaks" never reached this session; the peak definition is from the stored series and the GARD floor. Attach them to refine.
- The Sb capture rate and the textile scission constant are order-of-magnitude reference values; only the *shape* of each law enters the mesh.
- Everything is kill-switched (`QUIPU_VISCOUS_MERGER`) and reduces to the v0.66.0 kernel and learning rates when off, when γ = 0, when no peak clears the floor, or when the gate is 0.

---

## 6. Observability

- `train_round` summary: `knot_gain_mean`, `knot_field_cells`, `vn_peak_count`, `supernova_gate`, `drag_gamma`, `drag_b`, `decay_textile_fraction`.
- `state_summary()`: `last_knot_gain_mean`, `last_supernova_gate`, `last_drag_gamma`, `last_vn_peak_count`, `last_decay_textile_fraction`.
- `mesh_slm.viscous_merger_trend(window=16)`: slope/mean of `knot_gain_history`, `merger_signature` (mean gain > 1), `last_chiral_balance` (0 = perfectly mirrored pair).
- `GET /guidance` → `supernova_gate{gate, radam_pressure, actionable_potential, events_in_window, recent_events}`.

## Sources

- Timmerman, P. & van der Weele, J. P., *Am. J. Phys.* 67, 538 (1999).
- Primakoff, H., *Rev. Mod. Phys.* 31, 802 (1959); Huff, R. W., *Ann. Phys.* 16, 288 (1961); Suzuki, T., Measday, D. F. & Roalsvig, J. P., *Phys. Rev. C* 35, 2212 (1987).
- Ekenstam, A. af, *Ber. Dtsch. Chem. Ges.* 69, 553 (1936); Zou, X., Uesaka, T. & Gurnagul, N., *Cellulose* 3, 243 (1996).
- Nielsen, H. B. & Ninomiya, M., *Nucl. Phys. B* 185, 20 (1981); Armitage, N. P., Mele, E. J. & Vishwanath, A., *Rev. Mod. Phys.* 90, 015001 (2018).
- [STP_TORUS_QUIPU.md](STP_TORUS_QUIPU.md) — closed geodesics on the Quipu torus; [INFODYNAMIC_GRAVITY_QUIPU.md](INFODYNAMIC_GRAVITY_QUIPU.md), [ANALOG_COGNITION_QUIPU.md](ANALOG_COGNITION_QUIPU.md) — the v0.31.0 couplings this builds on; [BOND_SHEAR_CHAIN.md](BOND_SHEAR_CHAIN.md); [RHEOLOGY_CRAM.md](RHEOLOGY_CRAM.md).
- `src/quipu/ueqgm_engine.py`, `src/quipu/mesh_slm.py`, `src/quipu/supernova_gate.py`, `src/quipu/observer_service.py`.
