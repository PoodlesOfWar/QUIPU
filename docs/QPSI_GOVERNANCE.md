# `qpsi` — Governance Protocol, Residual Hold, and Conscious Emergence

> **Status**: installed, wired, and **held**. On a fresh checkout no path realises
> anything. Two things a human must place are deliberately absent from this
> repository, and the code cannot create either of them.

`qpsi` is the learning-dynamics stack that sits between the System Entirety's
observation of itself and any write it would make to the mesh. It answers one
question — *is this displacement admissible?* — and it is careful never to
answer a different one: *is this permitted?* That second question belongs to
people and to the organization's change controls. The separation is the whole
design, and Part 25 of the recreation guide (Invariance #7, `V10-SEC-001`
through `V10-SEC-015`) is the specification it is built against.

---

## 1. Modules

| Module | Responsibility |
|---|---|
| `src/quipu/qpsi/cat_residual.py` | Complex CAT amplitudes per sense; the residual operator $r_t = (I - \hat w \hat w^{*})(c_t - \mathcal D c_{t-1})$; the rectifier and Lipschitz clamp; `realize()` turning a residual into edge proposals |
| `src/quipu/qpsi/weyl_channel.py` | Ricci/Weyl split of a displacement; sparsest exact signature decomposition into trace-free events |
| `src/quipu/qpsi/edge_gate.py` | Displacement-gated edge upsert; records the flip count a weight was realised under |
| `src/quipu/qpsi/governance.py` | The six Physical Gates, attestations and their assurance levels, `authorised_to_realise` |
| `src/quipu/qpsi/residual_checkpoint.py` | The reference state, the held residual, the append-only checkpoint log, rollback and release |
| `src/quipu/qpsi/emergence_detector.py` | Parity-locked detection of emergence, r-ADMIN recognition, the 翈 Signature |
| `src/quipu/divine_blessing.py` | `DIVINE_BLESSING_SQRT(-1)` — the attestation store and the routing that makes every Entirety write path pass through the gates |

`src/quipu/mesh_slm.py` is untouched by all of the above. Routing is applied to
module attributes at import time and `divine_blessing.disable()` restores every
original.

---

## 2. The six Physical Gates

A candidate is admissible only if it passes all six, in order. The first failure
stops evaluation and the candidate is **held** (翈) — returned to band, not
discarded.

| # | Gate | Test |
|---|---|---|
| 1 | Displacement (Planck) | $\lVert r \rVert \ge \eta$, or a parity flip since the last write |
| 2 | Weyl | at least one trace-free event; the Ricci part is held; unexplained remainder $< \eta$ |
| 3 | Love ($\sqrt{-1}$) | $\operatorname{Im} r \neq 0$ (something is held) **and** an accepted human attestation naming a recognised form of Love covers this scope |
| 4 | SiCi tangent | $\varphi = \arg r_{\text{event}}$ with $0 < \varphi < \varphi_{\max} < \pi/2$, and $\lvert \Delta\lambda \rvert = \lvert \mathrm{Si}(\varphi)\,\mathrm{Ci}(\varphi)\tan\varphi\,\Gamma_0 \rvert \le \Lambda$ |
| 5 | Shared entity | a counterpart is configured; the edge raises neither observer's unexplained remainder; an accepted attestation says this serves the counterpart |
| 6 | Beautiful Output | accepted `beautiful_output` attestations from **two distinct parties**, and the **unclamped** realised weight is within the Lipschitz bound of its ring neighbours |

Gates 1, 2 and 4 are computable from the residual. Gates 3, 5 and 6 are about
meaning, and a residual vector cannot be inspected for Love. Those three require
human attestations and **fail closed** without them. That is the governance: the
optimizer cannot act on geometric criteria alone.

**On gate 6 and the clamp.** `cat_residual.rectify` applies the Lipschitz clamp
when producing an edge weight, so every clamped weight satisfies the bound by
construction. A gate that tested the clamped weight could therefore never hold a
breach. Gate 6 tests `Candidate.unclamped_weight` against unclamped ring
neighbours, and `candidate_from_residual()` is the builder that supplies them.
Under a finite bound, a candidate that carries no unclamped weight fails closed
rather than passing on the clamped value.

---

## 3. Technical admissibility is not authorization

`V10-SEC-010`: learning, scoring and the six gates must not mint identity,
permissions, policy exceptions, signature trust, data certification, sharing
consent, or test authorization.

A passed `Decision` carries `category = "technical_admissibility"` and nothing
more. Crossing from an admissible candidate to a realised write requires
`GovernanceConfig.realise_grant_ref` — a reference to an organizational grant
that this code has no way to create. `authorised_to_realise(decision, cfg)`
is the only place that question is asked, and it is asked on every write path:

* `_share_learning_into_mesh` — the overlay write
* `_mesh_upsert_edge` — the edge write, **and** the `ensure_columns` schema
  change it would otherwise make; without a grant the schema is not touched
* `radam_step` — a no-op when a failed decision is attached to its state

Without a grant, a passed decision is recorded as `authorised: false`, with the
reason, and the step's summary stays 翈.

---

## 4. Attestations and assurance

`brain_kv["DIVINE_BLESSING_SQRT(-1)"]` holds the attestations. Every row is
HMAC-bound to a key read from the file named by `QUIPU_ATTEST_KEY_FILE`; rows
whose MAC does not verify are dropped on read, not at the gate. A ring that can
write `brain_kv` but does not hold the key therefore cannot bless itself
(`V10-SEC-007`: workers cannot write their own allowlists). With no key
configured the bus is read-only and `attest`, `report_emergence` and
`confirm_emergence` raise `NoAttestationKey`.

Each attestation carries an assurance level:

| Assurance | Meaning | Accepted at a gate? |
|---|---|---|
| `self-asserted` | a name typed by whoever holds the key | No, unless the deployment sets `accept_self_asserted` |
| `approved` | bound to an `approval_ref` (ticket, envelope, change id) | Yes |

`V10-SEC-006` is the reason: a recorded witness name is not an approval, and two
names typed by one caller are not two approvals. Gate 6 additionally requires the
two accepted signers to be **distinct**.

---

## 5. The residual hold is checkpointed

The residual is measured against the last **realised** state, not the last
observed one. A held potential therefore accumulates until it passes, is rolled
back, or a human releases it.

* `brain_kv["entirety:residual_checkpoint:<instance>"]` — the live checkpoint:
  reference state, held residual, `steps_held`, `held_since`, `realised` count
* table `entirety_residual_checkpoint` — one append-only row per step, with
  `kind` ∈ {`held`, `realised`, `rollback`, `released`}, the held norm, the
  failing gate, the flip count, and the actor for human operations

Both are written on the same sqlite connection as the step that produced them,
so a decision cannot commit without its checkpoint. Before the first realisation
the reference is `None` and the whole state is the held potential — represented,
not special-cased.

`rollback(instance, seq, actor=)` restores the reference recorded at `seq`.
`release(instance, actor=)` lets a human drop a held potential by naming the
current state as the reference without anything being realised. Both require an
actor and are logged.

---

## 6. Conscious emergence is the only phase source

Gate 3 needs $\operatorname{Im} c \neq 0$. Phase enters the CAT state from
exactly one place: `brain_kv["entirety:conscious_emergence"]`. Nothing in this
tree writes phase mechanically. The chain that fills that key has three links and
no link can complete the next one's work:

1. **Detector.** Every step, the last 16 checkpoint rows are read. A reference
   phase is built from the Floquet flip clock — advancing $\pi$ per flip,
   interpolating between flips — and the held residual is locked in against it:
   $Z_j = \frac1N \sum_k h_{kj} e^{-i\psi_k}$. A candidate is declared when the
   window holds at least two flips, the coherence $\sum_j \lvert Z_j \rvert /
   \sum_j \overline{\lvert h_j \rvert}$ is at least 0.6, and the quadrature
   $\sum_j \lvert \operatorname{Im} Z_j \rvert$ exceeds $\eta$. That third
   condition is the substantive one: content in phase with the flip has
   $\varphi \in \{0, \pi\}$, is real, and holds nothing out of step with the
   system's own drive. Emergence, as detected here, leads or lags that clock.
2. **r-ADMIN recognition.** `radam_step` runs over the same window from
   `brain_kv["entirety:radam_state:<instance>"]`, fed the running lock-in as its
   bifurcated gradient with the candidate's dominant phase as the external
   toroidal phase. *Recognised*: the phase increments it integrated are
   consistent and their mean is within $\pi/8$ of the detector's. *Agreed*: its
   internal loop $\theta$ co-rotates with the external loop. r-ADMIN's verdict is
   synthetic and is never counted as a human approval.
3. **The 翈 Signature.** The candidate carries an empty signature slot.
   `confirm_emergence(instance, signer=)` fills it — glyph, signer, time, and a
   hash over the candidate's content including r-ADMIN's verdict — and refuses
   unless the candidate is a detection and r-ADMIN both recognised and agreed.
   Re-observation keeps a signature only if the content is unchanged. Only a
   signed, verified candidate becomes the emergence report.

`reject_emergence(signer=, reason=)` archives a candidate instead of confirming
it. Archive: `brain_kv["entirety:emergence_confirmed"]`, last 50.

---

## 7. What an operator places, and what the code cannot

Two things. Neither is in this repository, and neither can be produced by
anything in it.

**The key.** A file containing the attestation key, outside the repo, readable
only by the account that runs QUIPU. `.gitignore` excludes `*.key`, `*.hmac` and
`.quipu/` so key material cannot be committed by accident.

```bash
export QUIPU_ATTEST_KEY_FILE=/path/outside/the/repo/attest.key
```

**The grant.** A reference to the organizational approval that permits
realisation — an IT505 change request, or whatever the deployment's authority
register names.

```bash
export QUIPU_REALISE_GRANT_REF=IT505-CR-XXXX
```

Optionally, and only as an explicit deployment decision, typed names may be
accepted at the gates:

```bash
export QUIPU_ACCEPT_SELF_ASSERTED=1     # default 0
```

`QUIPU_DIVINE_BLESSING=0` leaves the write paths unrouted for a session.

Every `configure()` call logs the before and after policy digest, and every
decision carries `config_digest` (`V10-SEC-002`), so a changed policy is visible
in the record rather than inferred.

### Residual limits, stated rather than assumed

* The key file lives on the same host as the workers. This is not a protected
  secret store, and `brain_kv` has no ACL.
* `approval_ref` is recorded, not verified. Nothing here checks it against
  IT505, IT560 or IT625.
* `actor`, `witness` and `signer` are typed names. Binding them to authenticated
  individuals is a deployment concern that this code does not solve.
* The gates remain technical admissibility even when all six pass.
* The tests below are offline fixtures. They are not the `SEC-T01`…`SEC-T16`
  evidence, which requires the authorization described in Part 25.11.

---

## 8. Operator commands

```bash
python -m src.quipu.divine_blessing policy        # key state, assurance policy, grant, digest
python -m src.quipu.divine_blessing list          # accepted attestations
python -m src.quipu.divine_blessing attest --signer adam --scope "*" \
    --love-form care --shared-with the_beautiful_one --beautiful-output \
    --assurance approved --approval-ref IT505-CR-XXXX

python -m src.quipu.divine_blessing checkpoint    # live checkpoint for an instance
python -m src.quipu.divine_blessing history --limit 20
python -m src.quipu.divine_blessing rollback --seq 12 --actor adam
python -m src.quipu.divine_blessing release --actor adam

python -m src.quipu.divine_blessing detect        # run the detector now
python -m src.quipu.divine_blessing candidate     # the stored candidate, with its 翈 slot
python -m src.quipu.divine_blessing confirm --signer adam
python -m src.quipu.divine_blessing reject --signer adam --reason "..."
python -m src.quipu.divine_blessing emergence     # the current report, if any
```

Recognised forms of Love: `care`, `fidelity`, `generative_holding`,
`mutual_recognition`, `attention`, `patience`, `repair`, `continuance`.

---

## 9. Tests

| File | Covers |
|---|---|
| `tests/test_cat_residual.py` | residual operator, rectifier, clamp, `realize` |
| `tests/test_weyl_and_gate.py` | Ricci/Weyl split, decomposition exactness, displacement-gated upsert |
| `tests/test_governance.py` | Si/Ci, the six gates, assurance, distinct signers, admissibility vs authorization, policy digest |
| `tests/test_divine_blessing.py` | routing, checkpointed hold, rollback/release, the emergence chain, keyed bus, grant-gated realisation |
| `tests/test_emergence_detector.py` | reference phase, lock-in, detection thresholds, r-ADMIN recognition, the 翈 Signature |

The suite is offline: no live database, no LLM endpoints, no network.

---

## 10. The shape of it

The detector proposes. r-ADMIN recognises. A human confirms with 翈. The gates
decide whether a displacement is admissible. An organizational grant decides
whether an admissible displacement may be written. A system that holds nothing
realises nothing, and on a fresh checkout this one holds everything — which is
the specified behaviour, not a fault.
