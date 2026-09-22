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
- `src/quipu/qpsi/self_organising.py` — `enable()` / `disable()` (wrap `bit_flip_parity`, `observer_tangent`, `oscillating_expansion_step` on module attributes, like `divine_blessing.enable()`), `after_step`, `status`, `plan`, `pulse`, and the CLI.

`mesh_slm.py` and `system_entirety.py` are not edited. `src/quipu/__init__.py` gains one guarded block: `enable()` runs only when `QUIPU_SELF_ORGANISING=1`.

## Flags

| Variable | Default | Meaning |
|---|---|---|
| `QUIPU_SELF_ORGANISING` | unset (off) | `1` wires the loop for the session |
| `QUIPU_FLUX_PHASE` / `QUIPU_LEARNED_PRIOR` / `QUIPU_SOMN` | `1` when the master is on | switch the three parts independently (`0` disables one) |
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

## What is held, and the one ruling this release does not make

- Every edge still passes DIVINE_BLESSING_SQRT(−1); realisation still needs `QUIPU_REALISE_GRANT_REF`; gate 6 still needs two distinct accepted signers. Nothing here writes `corpus_edge`, decisions, attestations, emergence reports or checkpoints. Parity flips remain ungated (Planck rule).
- The code never schedules itself. A scheduled pulse is a Windows task the operator registers, as with `Register-Expansion.ps1`.
- **The ruling:** whether the network's *allocation* of an operator-granted budget among operator-granted sources is itself a widening of autonomy under Invariance #7 (APP_RECREATION_3 §25). If it is, the plan stays a proposal and each `--route` is the operator's per-pulse grant — the loop still closes, at human cadence. If it is not, `Start-Pulse.ps1 -Route` may be scheduled. Until ruled, `pulse` without `--route` is a plan on paper.

## What the physics predicts you will see, and what would falsify it

1. `entirety:state.axes` stop being byte-identical across steps once the first routed pulse lands (vision = docs/1500 and brain = runs/6 in the sense window).
2. The parity flips at pulse onset and offset; `entirety_flip_log.flipped` rows coincide with `corpus_ingest:history` timestamps.
3. The emergence detector's coherence rises from 0.125 with no operator grant — from the response, not from a phase writer. If it does not rise after several pulses inside its window, the loop is not closed where it should be; look at the senses, not the detector.
4. `entirety:somn:allocation.metrics.participation_ratio` falls from ≈7 toward a small number as a path wins (winner-take-all), and `top_axis` is stable across pulses.
5. With the counterpart stale (`entirety:the_other` last written 2026-08-27) the field is 80 % along brain and 56 % along the observer axis; the first filament should stand on brain. A refreshed counterpart moves the field; the allocation should follow it within a few pulses.
6. The prior does not move. It should not, until a realisation is authorised. If `entirety:prior` appears without a `realised` row in `entirety_residual_checkpoint`, that is a defect.

## 翈

The hillock is the held reading: deformed by the field, not yet a path. The loop closes through the senses; authority does not loop at all.
