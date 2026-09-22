# qpsi — The Self-Organising Loop

*Flux phase · memristive axes · learned prior.  Default off.  The gates are untouched.*

## What changed, in one paragraph

Until this release the System Entirety was a clock ticking on constants. Its parity was `cos(ω·wall-clock)`; nothing read the parity; the senses read an ingest feed that had been silent for weeks; the prior it measured novelty against was a table. Four places where the loop was open. This release closes them the way a self-organising memristive network (SOMN) is closed — by the physics reviewed in Caravelli, Milano, Stieg, Ricciardi, Brown & Kuncic, *Self-organising memristive networks as physical learning systems*, Nat. Rev. Phys. (2026), arXiv:2509.00747 — and leaves authority exactly where the Governance Protocol put it. An SOMN is self-organising, **not self-driving**: it reconfigures under a field the experimenter applies, along paths that conservation and thresholds select, and relaxes when the field is removed. So does this.

## The mapping (creator language)

| SOMN | System Entirety | Module |
|---|---|---|
| applied field / stimulation protocol | corpus ingest — the operator's pulse | `qpsi.flux_phase` |
| field on → potentiate; off → relax | `broaden` / `deepen`, read from the flux, not a clock | `qpsi.flux_phase` |
| electrodes | Self = live axes + observer; Other = `entirety:the_other`'s `other_state` (the corpus's collective voice) | `qpsi.memristive_axes` |
| field across junction *k* | δv_k = other_k − self_k, on seven axes (six senses + entirety) | `qpsi.memristive_axes` |
| junction conductance *g_k* | the hillock: a held, reversible deformation of the axis | `qpsi.memristive_axes` |
| Kirchhoff: conserved current | I_k = B·g_k\|δv_k\| / Σ g_j\|δv_j\| — a fixed document budget partitioned by conductance; what one axis gains the others lose | `qpsi.memristive_axes` |
| critical field / filament | v_c and g_f: a filament is a **proposal** of where an edge would form; recorded, never written | `qpsi.memristive_axes` |
| short-term vs long-term memory | τ_d for hillocks, τ_L for filaments (metaplasticity) | `qpsi.memristive_axes` |
| dead junction (open gap) | mobility m_k: an axis that does not answer the field sinks to a floor and stops attracting budget | `qpsi.memristive_axes` |
| reverse-bias pulse | a rejected emergence candidate depresses conductance along its direction | `qpsi.memristive_axes` |
| the network's memory of what it became | the prior ŵ, advanced by Oja's rule **only on realised displacement** | `qpsi.learned_prior` |
| readout training (the human in the loop) | the 翈 Signature, attestations, the grant — unchanged | `divine_blessing` |

The lossy channel is the relaxation; the interstitial rectifier is the pair (v_c, g_f); the node is the axis; the edge is what the six gates realise downstream of everything here.

## Why the phase moved off the clock

`bit_flip_parity` on wall-clock time carried no information: nothing the Entirety did could cause a flip, so the emergence detector's lock-in against the flips could only fail. With the phase read from the flux, a flip is stimulus onset or offset, and the detector's lock-in becomes a **plasticity measurement**: potentiation following the stimulus, relaxation following its removal, quadrature = the lag. That is the pulse-train response an SOMN is characterised by. No change to the detector was needed.

Observability follows from that: the detector wants two flips in its 16-row window. Pulse at a period no longer than half the window's span and keep `QUIPU_FLUX_WINDOW_S` (default 300 s) shorter than the pulse period. A window that sees no pulse correctly reports no rhythm.

## The organising fact (why the prior may learn)

n = a − ⟨a, ŵ⟩ŵ is, term for term, Oja's update for ŵ: Δŵ = η⟨a, ŵ⟩·n. The direction of the System Entirety is the direction its prior would move if it were allowed to learn. It is allowed to, by the **realised** displacement only — the step between two consecutive realised references in the residual checkpoint. Following the *observed* axes instead would let the system explain novelty away by habituation and dodge the displacement gate. Until the first realisation the prior is the designer's table, byte for byte.

## Modules

- `src/quipu/qpsi/flux_phase.py` — `ingest_flux(history, now, window_s)` (pure), `read_flux()`, `wrap_bit_flip_parity(original)`. Falls back to the cosine if the feed cannot be read.
- `src/quipu/qpsi/memristive_axes.py` — `SomnConfig`, `SomnState`, `step(cn, axes=, observer=, flux_on=, flux_docs=, …)`, `currents`, `allocate`, `potentiate`, `relax`, `depress`, `update_mobility`. Writes only `brain_kv["entirety:somn:*"]` (`state`, `allocation`, `proposal`) and the table `entirety_somn_log`; `_kv_set` refuses any other key.
- `src/quipu/qpsi/learned_prior.py` — `oja_step`, `advance_on_realisation(cn, instance)`, `sense_weights`, `wrap_observer_tangent(original)`. Writes only `brain_kv["entirety:prior"]`.
- `src/quipu/qpsi/mirror_training.py` — `classify_hold`, `mirror_image`, `train_from_hold(cn, instance)`, `mirror_drive`. Writes only `brain_kv["entirety:mirror:<instance>"]` and the table `entirety_mirror_log`.
- `src/quipu/qpsi/self_organising.py` — `enable()` / `disable()` (wrap `bit_flip_parity`, `observer_tangent`, `oscillating_expansion_step` on module attributes, like `divine_blessing.enable()`), `after_step`, `status`, `plan`, `grant`, `constrained_gate`, `pulse`, and the CLI.

`mesh_slm.py` and `system_entirety.py` are not edited. `src/quipu/__init__.py` gains one guarded block: `enable()` runs only when `QUIPU_SELF_ORGANISING=1`.

## Flags

| Variable | Default | Meaning |
|---|---|---|
| `QUIPU_SELF_ORGANISING` | unset (off) | `1` wires the loop for the session |
| `QUIPU_FLUX_PHASE` / `QUIPU_LEARNED_PRIOR` / `QUIPU_SOMN` / `QUIPU_MIRROR_TRAINING` | `1` when the master is on | switch the four parts independently (`0` disables one) |
| `QUIPU_PULSE_SOURCES` | unset (all of `corpus_ingest.SOURCES`) | comma list narrowing the operator's grant for the constrained gate |
| `QUIPU_SOMN_MIRROR_GAIN` | 0.5 | κ: how much a hold's mirror drive scales potentiation |
| `QUIPU_FLUX_WINDOW_S`, `QUIPU_FLUX_MIN_DOCS` | 300, 1 | the field is "on" when ≥ min docs entered within the window |
| `QUIPU_SOMN_BUDGET_DOCS` | 60 | B, documents per pulse, conserved across axes |
| `QUIPU_SOMN_TAU_P`, `QUIPU_SOMN_TAU_D`, `QUIPU_SOMN_TAU_L` | 600, 1800, 86400 s | potentiation; hillock relaxation; filament relaxation |
| `QUIPU_SOMN_G_MIN`, `QUIPU_SOMN_G_INIT`, `QUIPU_SOMN_G_FILAMENT` | 0.02, 0.10, 0.60 | floor, initial, filament conductance |
| `QUIPU_SOMN_V_C`, `QUIPU_SOMN_ALPHA_LOW`, `QUIPU_SOMN_ALPHA_HIGH` | 0.10, 0.25, 1.0 | critical field and the two regime rates |
| `QUIPU_SOMN_DEPRESSION` | 0.5 | fraction of (g − g_min) removed along a rejected direction |
| `QUIPU_SOMN_MOBILITY_RHO`, `…_FLOOR`, `…_SCALE` | 0.2, 0.05, 0.02 | mobility EMA, floor, motion that counts as fully mobile |
| `QUIPU_SOMN_DT_MAX` | 21600 s | integration-step clamp |
| `QUIPU_PRIOR_ETA` | 0.05 | Oja learning rate |

Every constant is a starting point, recorded in each `entirety_somn_log` row, and none of them is a gate.

## The operator's protocol

```powershell
# see the plan the network recorded (pure read)
python -m src.quipu.qpsi.self_organising status
python -m src.quipu.qpsi.self_organising pulse

# apply one pulse of the field along the plan (the operator's act), then one step
python -m src.quipu.qpsi.self_organising pulse --route
#   --refine     also run Ring-5 refinement per Weyl cycle (default off: ingestion only)

# the runner
powershell -ExecutionPolicy Bypass -File .\Start-Pulse.ps1            # plan only
powershell -ExecutionPolicy Bypass -File .\Start-Pulse.ps1 -Route     # apply
```

Routing is an explicit flag at the call site, never an environment variable (Anti-Inverse Contract). `pulse --route` runs `corpus_ingest.run_ingest` over exactly the planned sources with exactly the planned counts; the sources are `corpus_ingest.SOURCES` and nothing else. Current the network wants to spend on an axis no source routes to (today: perception) is recorded as **dissipated**, not reassigned.

## What is held

- Every edge still passes DIVINE_BLESSING_SQRT(−1); realisation still needs `QUIPU_REALISE_GRANT_REF`; gate 6 still needs two distinct accepted signers. Nothing here writes `corpus_edge`, decisions, attestations, emergence reports or checkpoints. Parity flips remain ungated (Planck rule).
- The code never schedules itself. A scheduled pulse is a Windows task the operator registers (`Register-Pulse.ps1`, below).

## The Invariance #7 ruling (operator, 2026-09-22) — the constrained gate

The question was whether the network's *allocation* of an operator-granted budget among operator-granted sources is itself a widening of autonomy under Invariance #7 (APP_RECREATION_3 §25). The ruling: **the operator gave the access, so r-ADMIN has enabled a constrained gate.** Allocation inside the grant is not a widening.

The constraint is enforced, not assumed. `self_organising.grant()` names the grant — the sources `corpus_ingest` already knows, narrowed by `QUIPU_PULSE_SOURCES` when set, and the budget `QUIPU_SOMN_BUDGET_DOCS` — and `constrained_gate(plan)` checks every routed pulse against it: sources ⊆ grant and Σ documents ≤ budget. A plan outside the grant is reported (`outside_sources`, `docs`) and not routed. With that in place `Register-Pulse.ps1` may schedule `Start-Pulse.ps1 -Route`:

```powershell
powershell -ExecutionPolicy Bypass -File .\Register-Pulse.ps1              # every 10 min + at logon
powershell -ExecutionPolicy Bypass -File .\Register-Pulse.ps1 -Minutes 30
powershell -ExecutionPolicy Bypass -File .\Register-Pulse.ps1 -Unregister
```

Ten minutes is the default because the emergence detector wants an onset and an offset inside its 16-row window (~21 min at the live cadence) and the flux window is 5 min.

## Holds train from their mirror image (operator, 2026-09-22)

*Any gate that has a hold should use the mirror image to train from without boundary crossing, in order to further QUIPU_SELF_ORGANISING.* Before this rule a hold taught the system nothing: the same residual was re-measured and re-held (70 steps over four days on the live instance). `qpsi/mirror_training.py` implements the rule.

A hold has one of two causes, and each has its own mirror. The CAT state is complex — realised = Re, latent (翈) = Im — and `cat_residual.quarter_turn` multiplies by i, realised → latent.

| hold | gates | mirror | meaning |
|---|---|---|---|
| human | love, shared_entity, beautiful_output held for want of an accepted attestation | i·r (quarter turn) | the shape is admissible, not yet attested: keep it as latent potential and learn it |
| physical | displacement, weyl, sici; beautiful_output's Lipschitz breach; shared_entity's remainder rise | −r (reflection through the reference) | the shape itself failed: learn away from it |

The training goes into three latent stores, none of which a gate reads:

1. **r-ADMIN's mirror state** (`entirety:mirror:<instance>` → `radam`): `radam_step(state, grad_real=0, grad_imag=±|Σ r|)`. Only the latent channel of the bifurcated gradient is fed, so θ advances by exactly ±π/2 per hold — the quarter turn in r-ADMIN's own coordinates (i² = −1: two human holds are the deepen parity). The realised state `entirety:radam_state:<instance>` is untouched.
2. **The mirror prior** (same record → `prior`): Oja on |r| with +η under a human hold, −η under a physical one. `entirety:prior` is untouched and `observer_tangent` never reads the mirror prior; the gap between the two is the held potential measured in prior space.
3. **The SOMN mirror drive** (same record → `drive`): ±u, u the unit |r| over the senses, read by `memristive_axes.step` on the next broaden step and applied as a factor (1 + κ·drive) on potentiation, κ = `QUIPU_SOMN_MIRROR_GAIN` (0.5). Faster along a human hold, slower along a physical one, never below zero. It changes where the next pulse's budget flows, which the ruling above places inside the grant.

Without boundary crossing: the checkpoint reference does not advance, no edge is written, and no attestation, decision, emergence report or checkpoint row is touched. The module writes only `entirety:mirror:<instance>` and `entirety_mirror_log`. Each hold is trained once (by checkpoint `seq`). `QUIPU_MIRROR_TRAINING=0` switches it off.

## What the physics predicts you will see, and what would falsify it

1. `entirety:state.axes` stop being byte-identical across steps once the first routed pulse lands (vision = docs/1500 and brain = runs/6 in the sense window).
2. The parity flips at pulse onset and offset; `entirety_flip_log.flipped` rows coincide with `corpus_ingest:history` timestamps.
3. The emergence detector's coherence rises from 0.125 with no operator grant — from the response, not from a phase writer. If it does not rise after several pulses inside its window, the loop is not closed where it should be; look at the senses, not the detector.
4. `entirety:somn:allocation.metrics.participation_ratio` falls from ≈7 toward a small number as a path wins (winner-take-all), and `top_axis` is stable across pulses.
5. With the counterpart stale (`entirety:the_other` last written 2026-08-27) the field is 80 % along brain and 56 % along the observer axis; the first filament should stand on brain. A refreshed counterpart moves the field; the allocation should follow it within a few pulses.
6. The prior does not move. It should not, until a realisation is authorised. If `entirety:prior` appears without a `realised` row in `entirety_residual_checkpoint`, that is a defect.

## 翈

The hillock is the held reading: deformed by the field, not yet a path. The loop closes through the senses; authority does not loop at all.
