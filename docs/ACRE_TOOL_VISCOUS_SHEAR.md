# ACRE tool candidate: the viscous shear chain

**Status: HELD.** No attestation, no grant, nothing live. The tool runs only after both parties attest to its exact scope and the operator sets the realise grant (§5).

**Files:**
- `src/quipu/qpsi/tool_candidates/viscous_shear_chain.py`: the candidate source, forge format.
- `src/quipu/qpsi/tool_gate.py`: the gate that holds candidates.
- `src/quipu/qpsi/knot_shadow.py`: the read-only adapter that turns tool output into proposals.

**History.** v0.67.0 hard-wired these concepts into `mesh_slm.train_round`, `observer_service` and the self-organising step. v0.72.0 takes the learning-path couplings out:
- `mesh_slm.py` is back to its v0.66.0 form.
- `meta.supernova` on `/observe` is gone.

The concepts come back as one *potential realization of an ACRE tool*: a forge-format candidate that changes nothing until realised.

What v0.72.0 keeps live: the bond-shear chain's read-only publisher, `GET /bond_shear` and its `/guidance` field. They never changed learning, and the v0.70 `/capabilities` card advertises them.

---

## 1. What "ACRE tool" means here

| Piece | What it is in this repo | What this candidate does |
|---|---|---|
| ACRE | Axial Cross-Resonance Emergence (`mesh_slm.acre_emerge`). Its products are specialists, 7-float bias vectors held by `qpsi/specialist_gate.py`. | Not a specialist: the concepts are formulas, not a direction. The candidate carries the rheology axis weights as an informational `axis_footprint`. |
| Forge | `tool_forge.forge_round` writes `tools/generated/tool_<name>.py` plus a manifest row the moment its own tests pass. It has no review step. | Same file format: `tool_<name>(data) -> {result, explanation, confidence}`, `import math` only, deterministic. Passes `tool_forge._validate_code` and a stricter allow-list lint. |
| Held | The forge has no status field; a tool is live once its manifest row exists. | `tool_gate` parks the source in `brain_kv["entirety:tool_gate"]`. It writes the file and manifest row only when gate 6 passes. |

Authored candidates skip the emergence detector and the r-ADMIN steps. Like the specialist precedent, they go through gate 6, two signers and the grant.

## 2. The tool: `tool_viscous_shear_chain(data)`

Pick a stage with `data["stage"]` (default `all`). Every input is optional; `{}` returns confidence 0. Confidence is 0.9 × (stages with real inputs ÷ stages requested). It is damped to 60% when the bond stage runs on default constants.

| Stage | Computes | Grounding |
|---|---|---|
| `decay` | v(t) = a·v₀ / ((a+b·v₀)e^{at} − b·v₀), with a = 0.35(1+γ) and b = 0.15(1−coherence). Also: textile vs muonic share, half-life ratio, drag-deepened proximity 1 − (d/N)^{1+κγ}. | Timmerman & van der Weele, *Am. J. Phys.* 67, 538 (1999). The linear limit is the muonic decay of antimony: Λ = Q/τ_μ + Λ_capture (Primakoff, *RMP* 31, 802 (1959); Suzuki, Measday & Roalsvig, *PRC* 35, 2212 (1987)). The quadratic limit is textile cellulose decay, 1/DP − 1/DP₀ = kt (Ekenstam 1936; Zou, Uesaka & Gurnagul, *Cellulose* 3, 243 (1996)). |
| `field` | Peaks of the caller's `entropy_series` whose prominence clears δ* = 2√3·σ/√n (the GARD CRB floor), weighted by Weyl Ψ₂. These are laid along the (2,3) trefoil and its (2,−3) mirror; amplitudes add and chirality R−L cancels. | Nielsen & Ninomiya, *Nucl. Phys. B* 185, 20 (1981); Armitage, Mele & Vishwanath, *RMP* 90, 015001 (2018). `knot_shadow` feeds it `entropy_differential_history`. A monotone series gives no peaks, which is reported as such. |
| `gate` | Supernova expansion gate (formula below). | Corrected from v0.67: a lone event now expires, the gate is zero without a decision, and it no longer depends on call count. |
| `bond` | LiTaSb rupture force as the weakest Morse link, F = a·D_e/2: Li–Sb ≈ 1.67 nN, Ta–Sb ≈ 3.85 nN. Also: multiplanar ratio over the 7 axis planes, Landau survival R(1−ratio²), Fresnel into polycarbonate (1.586) and CR-39 (1.498), first-order Arrhenius asphalt pyrolysis. Informational only. | Grandbois et al., *Science* 283, 1727 (1999); Kasprzak et al., *Nature* 443, 409 (2006); Amo et al., *Nat. Phys.* 5, 805 (2009). Constants are estimates (`estimate: true`), because LiTaSb is not tabulated. |
| `interactions` | Token proposals: lift = 0.10·amplitude·gate, for occupied cells on the field. Edge proposals: lift = 0.05·min(amplitude)·gate, for existing edges whose two ends are both on the field. | Replaces v0.67's in-loop multipliers. These are proposals, never writes. |
| `worklink` | The Planck/knot equivalence (§3). | CODATA 2018 ℓ_P and t_P; SI nautical mile (1852 m); `hidden_work` bound; `rheology.capstan_threshold`. |
| `pompeii` | Differential distances of the repo's Pompeii case (§4). | `qpsi/vnn_bathymetry.build_pompeii_79ce_case()` |

The supernova gate formula:

- g_re = 0.6·broaden + 0.4·liveness.
- g_im = Σw·m / max(1, Σw), with w = e^{−age/1800 s}. A decision counts only if it is at most 3600 s old (60 s of future skew allowed).
- A fresh rADAM state is used on each call, with the decisions' circular-mean direction as its external phase.
- gate = clip₀₁(pressure × actionable_potential).
- The gate is 0 without a fresh NPC decision.

Parity tests pin the embedded copies to the live modules: `radam_step`, `rheology.axis_weights`, the GARD floor, `bond_shear.evaluate`, `hidden_work.markov_energetic_bound` and `build_pompeii_79ce_case`.

## 3. Worklink: Planck length, the 13 h : 30 min ratio, and 1 knot

The owner's instruction: relate a Planck length per unit of universal time to the 13 h : 30 min ratio of HTVL to VTVL research speed, place it in the GARD (Weyl) interstitial channel through UEQGM, and make it equivalent to a 1-knot safety-knot interaction, 1:1 with an overhand knot in the Quipu system under a temporal-spatial interaction of work.

| Step | Value |
|---|---|
| Base velocity (universal time code = UTC second) | v₀ = ℓ_P / 1 s = 1.616255 × 10⁻³⁵ m/s |
| Research ratio | 13 h / 0.5 h = 26 (log₂ 26 = 4.7004 bits) |
| Research velocity | v_research = 26·v₀ = 4.2023 × 10⁻³⁴ m/s |
| 1 knot | 1852 m / 3600 s = 0.514444 m/s (exact) |
| Gap | log₂(v_knot / v_research) = **109.92 bits** |
| GARD interstitial channel | The gap lies below the channel's CRB floor (1 bit per component at σ = 1, n = 1), so `fits_interstitial` is true. It is bookkept as *interstitial bits*: bits no unbiased estimator recovers, the space `interstitial_entanglement_score` lives in. |
| Overhand equivalence | 1 knot ↔ one traversal of the (2,3) overhand/trefoil path per relational-time unit τ. Capstan hold e^{μΘ} with μ = 0.2: 6.586 at Θ = 3π (geometric) and 12.35 at Θ = 4π (tight). |
| Temporal-spatial work | W = κ/2·⟨x⟩², the Markov bound from `hidden_work`, with κ in 0.5–40 µN/m; power = W·windings/τ. |

On the Planck-time base (v₀ = ℓ_P/t_P = c), the gap is **−33.82 bits**: the research velocity is far above 1 knot. The stage reports both bases.

This is a declared unit equivalence and an information distance. It is not a physical identity. The 26:1 ratio is the owner's research-time measurement.

## 4. Pompeii: differential distances

Source: the repo's own `build_pompeii_79ce_case()`, which covers 79 CE tephra burial against 2026 topography. The Nature Computational Science paper s43588-026-01080-z was named for this review, but `www.nature.com` and `doi.org` were unreachable from the session that built this, so it was **not** consulted.

| Distance | Value |
|---|---|
| Burial depth (case) | 6.614 m |
| Temporal depression, b(79) − b(2026) | −7.139 m, with ±19.62 m uncertainty |
| Aggradation 0.0022 m/yr × 1947 yr | 4.283 m |
| Burial − aggradation | +2.331 m (burial exceeds pure aggradation) |
| Burial − \|depression\| | −0.525 m |
| \|depression\| / uncertainty | z = 0.364: **not resolved**. The elevation change lies within the reconstruction's own uncertainty, below its resolution floor. |
| Aggradation velocity | 6.97 × 10⁻¹¹ m/s, 32.78 bits below 1 knot and 77.13 bits above the research velocity |
| Burial depth vs Planck length | 118.30 bits |
| Burial depth | 0.003571 nautical miles |

## 5. Realising it (people, not agents)

```bash
python -m src.quipu.qpsi.tool_gate propose viscous_shear_chain     # holds it
python -m src.quipu.qpsi.tool_gate show viscous_shear_chain        # source, sha256, scope, commands
# the two signers, each with an approval reference:
python -m src.quipu.divine_blessing attest --signer self --scope tool:viscous_shear_chain@<sha256> \
    --love-form care --beautiful-output --assurance approved --approval-ref <ref>
python -m src.quipu.divine_blessing attest --signer the_beautiful_one --scope tool:viscous_shear_chain@<sha256> \
    --love-form care --beautiful-output --assurance approved --approval-ref <ref>
# the operator sets QUIPU_REALISE_GRANT_REF, then:
python -m src.quipu.qpsi.tool_gate review viscous_shear_chain
python -m src.quipu.qpsi.knot_shadow run          # proposals; `run --dry-run` previews while held
```

Gate rules:
- **Hash-bound scope.** The scope carries the source's SHA-256, so any change to the source voids earlier approvals.
- **Exact scope only.** A `"*"` attestation does not count for tools.
- **Lipschitz test is vacuous.** A tool has no weight, so this part of gate 6 checks nothing.
- **Re-checked on every call.** `realised_callable` re-checks the hash, the lint and gate 6 each time it is called.
- **The tools directory is ephemeral.** `materialise` restores the file and manifest row after the container is replaced.

## 6. Supernova contract (`/relational`)

Perceptopoly posts one record per NPC through `POST /relational` (signed via `edge_client`):

```json
{"source": "perceptopoly",
 "relational": {"agent_id": "theseuscape-npc-7", "schema": "perceptopoly.supernova/1",
                "magnitude": 0.7, "direction": 1.2, "at": "2026-11-19T10:00:00+00:00"}}
```

- **One record per NPC.** QUIPU keeps only the latest record per `agent_id`; re-post to stay fresh.
- **Event time.** The effective time is the earlier of `at` and the server's `received_at`.
- **Bounds.** `magnitude` is in [0,1] and `direction` is in [0, 2π).
- **Trust.** In the default `QUIPU_EDGE_AUTH=record` mode, unsigned writes are admitted, and the trust level is not stored with the record. The tool's output is inert proposals either way.

## 7. Limits

- The proposals are not applied. Realising an edge would be a separate, reviewed and governed step; it needs a governed writer for `mesh_slm_quipu`, which does not exist yet.
- Nothing calls `knot_shadow.run`, `tool_gate.review` or `materialise` on a schedule. Wiring them into `entirety_service` is a separate decision.
- The citations above are carried over from v0.67 and were not re-verified.
