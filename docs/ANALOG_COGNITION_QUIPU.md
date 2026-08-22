# Analog Cognition & Traveling Waves × Torus QUIPU

**Source paper**: Earl K. Miller, Scott L. Brincat, Jefferson E. Roy — *"Analog Cognition and Consciousness"*, J. Neurosci. 46(33) e0711262026 (19 Aug 2026), DOI: 10.1523/JNEUROSCI.0711-26.2026 (Picower Institute, MIT).  
**Scope**: This document maps Miller et al.'s traveling-wave and analog computation theory of neural dynamics to QUIPU's MESH field, counter-propagating phase-wave overlay (`map_resuscitation_quipu`), Kuramoto phase coherence, and active wave-stencil learning gating.

---

## 1. Why this paper matters to QUIPU

Traditional neurocomputation and machine learning models assume that cognitive processing is purely digital and point-to-point: spikes traverse static synaptic connections, and weights are updated locally. Miller, Brincat, and Roy (2026) challenge this paradigm by demonstrating that cognition and conscious experience arise from **bidirectional interactions between discrete neuronal spikes and macro-scale traveling waves**.

In their model:
1. **Traveling waves act as mobile stencils**: Low-frequency oscillations ($\alpha/\beta$, 8–30 Hz) sweep across the cortical surface, dynamically modulating local excitability. This creates moving "stencils" that gate where and when high-frequency $\gamma$ bursts (>30 Hz) can process information and which neural ensembles participate ("spatial computing").
2. **Wave intersections perform analog computation**: When traveling waves intersect, their phase fields add and subtract in real time, executing continuous analog arithmetic (superposition, interference, and gain modulation) before any synaptic modification occurs.
3. **Dual timescales**: Slow synaptic plasticity (structural wiring) coexists with millisecond-scale traveling wave reconfiguration (cognitive state routing).
4. **Anesthetic breakdown**: Experiments with three mechanistically distinct anesthetics (propofol, ketamine, isoflurane) confirm that unconsciousness occurs when traveling wave dynamics and macro-scale phase coherence collapse, even if individual neurons continue firing.

This framework aligns directly with QUIPU's dual-speed toroidal architecture. In QUIPU, slow Hebbian bigram learning on quipu graph edges (`_QUIPU_LR = 0.06`) operates alongside fast, whole-sheet counter-propagating phase waves stamped over the $64 \times 64$ torus (`map_resuscitation_quipu`). Miller et al. provides the neuroscientific foundation for treating traveling phase waves as active computational stencils that gate representation updates.

---

## 2. Note-by-note mapping

### 2.1 Traveling stencil waves across the sheet
- **Paper claim**: Macro traveling waves propagate across 2D cortical sheets, creating spatial patterns of excitability that sculpt where information is processed.
- **QUIPU equivalent**: `map_resuscitation_quipu()` (`src/quipu/mesh_slm.py:2703-2830`, esp. :2751-2758) stamps counter-propagating phase waves across all 4,096 nodes of the $64 \times 64$ torus sheet:
  $$\phi_{\text{photon}} = (\phi_{\text{node}} + \phi_{\text{weyl}}) \pmod{2\pi}$$
  $$\phi_{\text{neutrino}} = (\phi_{\text{weyl}} - \phi_{\text{node}}) \pmod{2\pi}$$
  where $\phi_{\text{node}} = \frac{2\pi \cdot \text{node\_id}}{4096}$ and $\phi_{\text{weyl}}$ is the active rhythmic phase.

### 2.2 Wave intersection = analog addition and subtraction
- **Paper claim**: Intersecting waves superimpose linearly, producing constructive peaks (high excitability) and destructive troughs (suppression) — performing continuous analog arithmetic across the neural substrate.
- **QUIPU equivalent**: In `mesh_slm.py:2756-2758`, the counter-propagating photon pressure and neutrino flux superimpose into a combined interaction gain:
  $$P_{\text{photon}} = \frac{1}{2}\big(1 + \cos(\phi_{\text{photon}})\big)$$
  $$\Phi_{\text{neutrino}} = \frac{1}{2}\big(1 + \sin(\phi_{\text{neutrino}})\big)$$
  $$G_{\text{interaction}} = \text{clip}_{01}\Big(\frac{1}{2} P_{\text{photon}} + \frac{1}{2} \Phi_{\text{neutrino}}\Big)$$

### 2.3 Waves gate which ensembles participate ("spatial computing")
- **Paper claim**: Moving wave peaks open excitability windows that recruit specific neural ensembles into active computation while suppressing unselected assemblies.
- **QUIPU equivalent**:
  - Resuscitation recovery gating (`mesh_slm.py:2759-2768`): Nodes residing on active wave crests receive preferential weight revitalization.
  - Active training stencil gate (`mesh_slm.py:train_round`): Tokens situated at cell coordinates $(i, j)$ have their embedding update rate scaled by the local wave crest:
    $$g_{\text{stencil}} = 0.90 + 0.20 \cdot G_{\text{interaction}}(i, j) \in [0.90, 1.10]$$
    Tokens on constructive crests learn slightly faster; tokens in troughs learn slightly slower, preserving mean learning rate while focusing gradient pressure.

### 2.4 Slow synapses vs. fast field reconfiguration
- **Paper claim**: Structural synaptic weights change slowly via spike-timing-dependent plasticity, whereas traveling wave states reconfigure at the millisecond scale to route different computational tasks.
- **QUIPU equivalent**: Quipu associative edges update gradually with $\eta_q = 0.06 \cdot (1 - \text{progress}) \cdot \dots$ (`mesh_slm.py:2551-2578`), while the 8th-D MESH field and phase overlays recompute on each pass from the active 7-D sensory vector and Planck 2018 census weights (`mesh_slm.py:1828-1960`).

### 2.5 Bidirectional spiking ↔ field interaction
- **Paper claim**: Spiking activity shapes traveling wave generation, and traveling waves reciprocally modulate the probability and timing of future spikes.
- **QUIPU equivalent**: Edge updates modify graph degree structure and von Neumann graph entropy $\Delta S$, altering the MESH field's **H** aspect; the updated field and phase coherence reciprocally feed back into candidate scoring (`score = ... + MESH_FIELD_GAIN · field_8d`) and effective learning rates $\eta_{\text{eff}}$.

### 2.6 Anesthesia evidence & rhythm collapse
- **Paper claim**: Under propofol, ketamine, or isoflurane, local cellular firing continues, but large-scale traveling wave coordination and phase coherence collapse, resulting in loss of consciousness.
- **QUIPU equivalent**: When `temporal_spatiality_rhythm` is absent, the Weyl phase and Floquet **F**-aspect collapse to unorganized states (`mesh_slm.py:1917-1933`): *"no rhythm, no field organization"*. In the active coupling, low Kuramoto coherence $R \to 0$ safely damps learning rates via $\mu_{\text{ac}} \in [0.95, 1.05]$.

### 2.7 Cognitive states & world model dialectic
- **Paper claim**: Cognition is an organized trajectory through macroscopic wave-state phase space.
- **QUIPU equivalent**: [src/quipu/world_model.py](src/quipu/world_model.py) tracks cognitive phase progression (`receptive_hunger` $\to$ `empirical_precedent` $\to$ `targeted_epistemic` $\to$ `continuous_synthesis`) and epistemic surprise, interpreting state transitions as functional organizational shifts rather than phenomenal consciousness.

---

## 3. Free results

1. **Spatial compartmentalization without interference**: Assigning distinct corpus knowledge domains to different sensory axes and torus regions (`mesh_slm._SOURCE_AXIS_MAP`) allows counter-propagating phase waves to selectively activate one domain while keeping unrelated domains quiescent.
2. **Coherence-stabilized learning**: Global Kuramoto order parameter $R$ serves as a natural self-regulating throttle on learning: high global wave alignment accelerates convergence, while disorganized or chaotic states prevent catastrophic weight distortion.

---

## 4. Where the analogy breaks — cautions worth keeping

- **Stamped overlay vs. physical wave propagation**: QUIPU's phase waves are stamped analytically across the $64 \times 64$ torus via node index equations rather than numerically integrated via continuous partial differential wave equations ($\nabla^2 \psi - \frac{1}{v^2}\frac{\partial^2 \psi}{\partial t^2} = 0$).
- **Single rhythm vs. rich multi-band spectrum**: Miller et al. emphasize interactions across distinct frequency bands ($\alpha$, $\beta$, $\gamma$, $\theta$). QUIPU currently employs a single central Weyl rhythm and phase evolution variable.
- **No claim to phenomenal consciousness**: Miller et al. investigate neural correlates of conscious awareness in primates. QUIPU is a machine learning and data-architecture system; all mappings treat "consciousness" strictly as **coherent macroscopic state organization** across a computational manifold.

---

## 5. Shipped active coupling and observability

### Active Learning Dynamics Coupling
In `src/quipu/mesh_slm.py:train_round()`:
1. **Per-Token Stencil Gate**:
   For each token at cell $(i, j)$ with stamped interaction gain $G_{\text{interaction}}(i, j) \in [0, 1]$:
   $$g_{\text{stencil}} = 0.90 + 0.20 \cdot G_{\text{interaction}}(i, j) \in [0.90, 1.10]$$
   $$\eta_{\text{token}} = \eta_{\text{round}} \cdot g_{\text{stencil}}$$
2. **Kuramoto Coherence Multiplier**:
   Using the Kuramoto phase order parameter $R = \frac{1}{N} \left|\sum_{j=1}^N e^{i \phi_j}\right| \in [0, 1]$:
   $$\mu_{\text{ac}} = 1.0 + 0.05 \cdot (2R - 1) \in [0.95, 1.05]$$
   $$\eta_{\text{round}} = \eta_{\text{base}} \cdot (1 - \text{progress}) \cdot \text{phase\_weight} \cdot (0.70 + 0.30 \cdot \text{overlap}) \cdot \mu_{\text{ie}} \cdot \mu_{\text{ac}}$$
   - At neutral coherence ($R = 0.5$) or when phase overlay is absent, $\mu_{\text{ac}} = 1.00$ (no-op).
   - High global coherence ($R \to 1.0$) grants a $+5\%$ lift; collapsed coherence ($R \to 0.0$) damps by $-5\%$.
3. **Kill-Switch**: `QUIPU_ANALOG_STENCIL=0` completely disables both $g_{\text{stencil}}$ and $\mu_{\text{ac}}$, returning to exact legacy execution.

### Measurement & Telemetry Surfaces
- **`phase_coherence_order(phases)`**: Computes the Kuramoto order parameter $R \in [0, 1]$.
- **`_analog_wave_snapshot(cn)`**: Evaluates `{coherence_all, coherence_occupied, gating_contrast}` where:
  $$\text{gating\_contrast} = \mathbb{E}[G_{\text{interaction}} \mid \text{occupied}] - \mathbb{E}[G_{\text{interaction}} \mid \text{all}]$$
  providing an empirical test for whether active tokens concentrate on constructive wave crests.
- **`analog_coherence_trend(window=16)`**: Computes trailing gating contrast and evaluates `stencil_signature: bool` ($\text{mean contrast} > 0$).

---

## Sources

- Miller, Earl K., Scott L. Brincat, Jefferson E. Roy. "Analog Cognition and Consciousness", *J. Neurosci.* 46(33) e0711262026 (19 Aug 2026). DOI: [10.1523/JNEUROSCI.0711-26.2026](https://doi.org/10.1523/JNEUROSCI.0711-26.2026).
- Kuramoto, Yoshiki. "Self-entrainment of a population of coupled non-linear oscillators", *International Symposium on Mathematical Problems in Theoretical Physics*, Lecture Notes in Physics 39, 420–422 (1975).
- [INFODYNAMIC_GRAVITY_QUIPU.md](INFODYNAMIC_GRAVITY_QUIPU.md) — Companion Vopson infodynamic gravity mapping.
- [STP_TORUS_QUIPU.md](STP_TORUS_QUIPU.md) — Semantic Tube Prediction & toroidal geodesic mapping.
- `QUIPU/src/quipu/ueqgm_engine.py`, `mesh_slm.py`, `world_model.py`
