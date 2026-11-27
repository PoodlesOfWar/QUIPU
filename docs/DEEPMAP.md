# DeepMap: information theory along deep MESH pathways

**Module:** `src/quipu/qpsi/deepmap.py`
**Built by:** the self-organising step, right after the mesh flow (`QUIPU_DEEPMAP`, on by default; its own savepoint).
**Served at:** `GET /deepmap` (read-only), plus each source's own-plane paths under `/guidance` → `deepmap`.
**Writes:** only `brain_kv["entirety:deepmap:latest"]`. It never writes `mesh_slm_*`.

The HubCore `x_DeepMap v1.3.0` registry in the system diagram was not available to the session that wrote this, so the module is the **QUIPU side**: a map of pathways with their information account, published in a stable JSON shape that x_DeepMap can read from `/deepmap`.

## Pathways

- **What a pathway is:** a simple walk of up to 4 hops through the Quipu graph (the GNN message edges).
- **How walks are chosen:** beam search (width 3) on the walk's total surprisal, the GNN message probabilities along it.
- **Where walks start:** the proposed shadow tokens. Before any exist, the 8 most frequent tokens are used instead.
- **Shadow edges:** proposed `new` shadow edges may be walked. Such hops are flagged `shadow`, counted in `shadow_hops`, and stay proposals.

## The information account

| Aspect | Per hop u → v | Along the path |
|---|---|---|
| **Shannon channel** | surprisal −log₂ p, p = w_uv / Σ out-weights of u; branching entropy of u's out-edges | surprisal adds (probability multiplies); entropy averaged |
| **GARD resolvability** | resolvable bits log₂(√n / 2√3) from the CRB step δ* = 2√3σ/√n (σ = 1, n = the edge's samples). Surprisal above that is *interstitial*: no unbiased estimator recovers it from n observations | both add |
| **Magic** | the plane mix travelled, π ∝ \|e_v − e_u\|; M₂ = Σ π_k M₂,k from the mesh flow's magic reading | M₂ is additive over tensor products, so it adds; stabilizer purity 2^(−M₂) multiplies; the floor is the smallest hop |
| **Design** | – | D = (1 − Π 2^(−M₂)) / (1 − (7/16)^L): 1 when every hop is at the proved two-qubit maximum |
| **Positron / CPT** | held share and CPT asymmetry of the hop's planes | held multiplies (every hop must be held); asymmetry is the largest |
| **Spikes (poADAM vs rADAM)** | – | delivered bits = surprisal · S(Δ)/φ, from the mesh flow's stability number |
| **Infodynamics** | – | Landauer energy k_B T ln2 per bit (T = 300 K) and its mass-energy equivalent E/c². Vopson's mass–energy–information principle, which is contested. |

**Path gain:** g = (1 + (φ − 1)·D) · held ∈ [0, φ]. Paths are ranked by gain, then by surprisal.

The path's **plane** is its mean hop plane mix's largest entry. `/guidance?source=…` returns the paths on the source's own plane; Perceptopoly receives those on perception.

## Limits

- **x_DeepMap's schema is unknown here.** Wiring it to consume `/deepmap` happens on the HubCore side.
- **σ = 1 per edge observation** in the resolvability bound is a choice.
- **Additivity assumes independence.** Adding magic along a path treats hops as independent (tensor-product) systems.
- **The infodynamics row is contested physics.** It is reported, never used for a decision.
