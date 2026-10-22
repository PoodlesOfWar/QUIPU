# Relational shear: the shearing λ in each axis's own time, feeding the shadow

`src/quipu/qpsi/relational_shear.py` (v0.65.0). Tests: `tests/test_qpsi_relational_shear.py`.

This module defines how the cross-axis shear of the CRAM axis array feeds the shadow tokens
(`shadow_tokens.py`). The aim is to keep the shadow's non-linear emission continuous, and to propose the
shadow tokens that recur under one shear source as paradigmatic events. Every quantity below is one QUIPU already
computes. The module only reads them and writes nothing to the mesh.

## Definitions

| Term | Definition | Read from |
|---|---|---|
| Relational time τ_k | an axis's own clock: dτ_k = \|realised_k\| / eps_k, the resolvable quanta its cell actually wrote that step. A step whose change stayed deferred in χ is not time for that axis. | `pcm_cells.step_array`: `decision.realised_dg`, `eps` |
| Relational rate ρ_k | Δτ_k / mean_j Δτ_j over a 16-step window (the emergence detector's window); 0 is frozen | `RelationalClock.rates()` |
| Differentiation in relational time | std of log ρ over the axes that moved | `RelationalClock.dilation()` |
| Relational velocity v_k | u_k · ρ_k: the axis's normalised Adam step on the ensemble's clock | `shear.u` per axis, or `_ensemble.u` (v0.62.0+) |
| λ_rel | Σ_k w_k (v_k − v̄_w)², with the deepen step's Poisson weights (mean σ_u²) | `relational_shear()` |
| λ_rel_STP | λ_rel / (v̄_w² + λ_rel), the shear's share of step energy | `relational_shear()` |
| Withheld excess e_k | unclamped_k − weight_k: what the Lipschitz clamp removed | `cat_residual.realize`: `unclamped`, `weight` |
| Clamp outlier k* | argmax of s_k = w_k · ρ_k · e_k, when its modified z-score is at least 3.5 (Iglewicz & Hoaglin 1993) | `clamp_outlier()` |
| Energy E* | s_k* | `clamp_outlier()` |
| Fed noise σ_eff | σ_L (1 + g (λ_rel_STP + E*)), capped at σ_L · max(1, 0.9 · SNR_min) | `fed_sigma()` |
| Gain g | inside the operator's doors [g_min, g_max] (default [0, 4]); moved by the continuity controller | `continuity()` |

## Continuity

The shadow's emission is continuous when:
- every voiced plane is resolvable (SNR ≥ 1), and
- its voice still crosses: the potential's crossing rate is at least one token changing per plane per run,
  1 / (n · K̄) for n scenarios and K̄ tokens per plane.

Below that crossing rate the emission is **frozen**: the same tokens in every scenario, leaving nothing to price.
Under SNR 1 it **dissolves**: the mesh cannot tell the movements apart.

The controller acts after each fed run:
- frozen: g goes up (×1.25, or +0.05 from near zero)
- dissolving: g goes down (×0.8)
- otherwise: g is held

Both bars are read from the run itself. A first draft used a fixed band on the potential's imaginary part; on the
test mesh it called a run with SNR 4 "dissolving", so it was replaced.

## Paradigmatic proposals

A shadow token (held back or pulled in) becomes a proposal when it recurs in two or more sessions while the same
axis is the clamp outlier. That makes it a two-regime record driven by one shear source. The rule is one vote per
session, so one loud session cannot make a proposal.

`proposals()` returns the proposals. A caller that persists them uses `entirety:shear:proposal`. A proposal is not
an edge.

## Governance

- Never writes the mesh, the shared embedding (`mesh_slm_embed`) or the axis cells.
- The clamp's withheld excess goes only into the shadow's scenario noise, never into a weight; putting it into a
  weight would cross gate 6.
- Proposals reach the torus only through the six gates, two distinct signers and the grant.
- Shadow tokens are existing mesh tokens: no shadow token takes a torus cell (ruling of 2026-09-28).
- The gain's range is the operator's door. The module moves g only inside it.
- σ_L itself (`mesh_slm._langevin_compute_sigma`) is not changed; a fed run uses a copy of the mesh with σ_eff.

## Use

```python
from src.quipu import shadow_tokens as ST
from src.quipu.qpsi import pcm_cells as pc, relational_shear as rs

clock = rs.RelationalClock()
step = pc.step_array(arr, currents=..., targets=..., now=..., deepen=True)
props = cat_residual.realize(residual, pivot=p_star, lipschitz=L)
unclamped = {p.axis: p.unclamped for p in props}
clamped = {p.axis: p.weight for p in props}
out = rs.fed_run(mesh=ST.load_mesh(), step_out=step, clock=clock, unclamped=unclamped, clamped=clamped,
                 gain=g, snr_min=last_snr_min, session=session_id, scenarios=64, k=8)
g, last_snr_min = out["gain_next"], out["snr_min"]
history.append(out)
proposed = rs.proposals(history)
```

## Measured (synthetic test mesh, 24 scenarios, K = 3)

- Feeding raised the crossing rate and the potential's imaginary part, with every plane still at SNR ≥ 1.
- **The outlier is read through the shear.** When the shear is small, the Poisson weight sits on the base
  frequency (vision, k = 0). A small clamp excess on vision then outweighs a large one on brain. This follows from
  the v0.62.0 weighting. Reading the outlier from the clamp alone would mean dropping w_k from s_k.

## Not yet wired

`fed_run` is a library call. Nothing in the Entirety step calls it yet. Wiring it into the deepen step, and
persisting proposals and the gain, are left for the operator to rule on.

## Where it fits

Perceptopoly's `docs/plans/theseuscape_fun_roadmap.md`, item 1 (content QUIPU generates), uses these proposals as
its source of candidate content: tokens QUIPU's own dynamics single out, tied to one axis and recurring across
sessions. Rollout to a world still goes through Gate 6.

翈 — the vane measures the wind by the time each feather took to turn.
