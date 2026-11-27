# Mott-neuron × spintronic-memory core, poADAM and rADAM

**Module:** `src/quipu/qpsi/mott_spintronic.py`. Pure functions; it reads and writes no table and no `brain_kv` key, and nothing in training or the services calls it.

**Source.** The device picture is the operator's description of "Integration of Mott neurons and spintronic memories for event-driven spiking neural networks" (*Nature Nanotechnology*). The paper could not be reached from the session that wrote this, so **no constant here is fitted to it**. Every constant is a named default in dimensionless units.

## What is modelled

| Piece | In the module |
|---|---|
| Edge hybrid compute-and-store core | Seven leaky integrate-and-fire neurons, one per CAT axis plane, each with its own memory element |
| Mott neuron | V ← V + Δt(−V/τ + I). It fires when V ≥ V_th (the insulator–metal threshold), then resets. |
| Shadow potentials | u_k = V_k / V_th: the subthreshold potential of each plane |
| Non-linear multiplanar shear | shear_k = tanh(Σ_j w_j (u_j − u_k)), with the rheology axis weights |
| Entropic coupling | I_k = drive_k + J·H(u)·shear_k, where H is the normalised Shannon entropy of the shadow potentials |
| No amplifier, no ADC | A spike of amplitude I_spike·V/V_th is stored only if it clears the critical current I_c by itself |
| Domain-wall racetrack | The wall advances one pinning site per stored spike, up to N sites |
| SOT-MTJ | One bit: the first stored spike sets it |
| Latent spike | A spike that fired but was not stored: too weak, or the memory was full |

## poADAM and rADAM

- **poADAM is poissonic ADAM.** Spike counts are treated as Poisson with the neuron's expected rate λ. The gradient is the Poisson score (k − λ)/√max(λ, 1), fed to plain Adam. Its pressure defines the potential.
- **rADAM is the realisation.** It uses the same Adam moments with:
  - g_re: the Poisson score of the spikes actually **written** to memory;
  - g_im: the latent spikes;
  - pivot slope: the fraction written.
- **Identifiable.** When every spike is written, g_im = 0 and the slope is 1, so rADAM equals poADAM step for step and Δ = p_rADAM − p_poADAM = 0.
- **Into the viscous network.** `run_window(...)` returns `radam_differential`, which `TypedViscousCoupler.step(radam_differential=…)` turns into the stability number S(Δ) = φ^(1 − 2|Δ|). All spikes stored gives S = φ. Latent spikes open the differential and tighten S.

Both optimisers start at pressure 0.5. Adam's step depends on the gradient's sign, not its size, so a start at 0 could never register a shortfall.

## Limits

- This is a behavioural model, not a device simulation: no Mott thermal dynamics, no wall velocity, no MTJ switching probability.
- The entropic coupling is this repo's reading of "entropically coupled"; the default gain J = 0.02 is a choice.
- The Poisson rate is the uncoupled neuron's exact discrete rate, so coupling shows up as a deviation from λ.
