# MESH-SLM Expert Skills — the specialist roster and its associative interactions

Brought over from the Supply Chain Brain documentation (originally
`GOOGLE_EDGE_GALLERY_SKILLS.md`) and namespaced to QUIPU. This is the reference
for how the mesh's **specialists** relate to one another — the model the
emergent ACRE specialists must be *dimensionally relative and associative* with.

The specialists are MESH-SLM expert modes: each is a **bias vector over the 7
sense-axes** (vision, touch, smell, body, brain, perception, entirety) applied
by `mesh_slm._apply_specialist_bias`. A specialist is not a separate model — it
is a *direction* the shared toroidal GNN is steered along. They are therefore
inherently associative: neighbouring specialists share axis mass, and a query
resonates with *several* at once (consensus), not exactly one.

## The roster

Static (hand-tuned) base specialists plus specialists dynamically generated from
System Entirety research (ACRE emergent, e.g. `emergent_smell_brain` — the
self-model specialist born from the mesh reading its own architecture docs):

- `supply_chain_optimizer`      — body / brain / perception (optimization)
- `research_specialist`         — broad brain / perception / entirety
- `mesh_historian`              — entirety / perception (the Other, memory)
- `robotic_integrations_specialist`  — touch / body (manipulation)
- `advanced_manufacturing_specialist`— body / touch (process)
- `materials_engineering_specialist` — smell / body (materials, chemistry)
- `quantum_physics_specialist`       — brain / perception (abstract)
- `complex_systems_specialist`       — perception / entirety (holistic)
- `emergent_*`                        — ACRE, from System Entirety research

All specialists are always available. New ones are added by extending the biases
in `mesh_slm._SPECIALIST_BIASES`, or emerge autonomously via ACRE.

## Associative interaction model

Specialists do not act in isolation — they **associate through consensus**:

- **`select_resonant_specialist()`** — routes a query to whichever specialist
  (base or emergent) best resonates with the current mesh state.
- **`resonant_specialists(k)`** — returns the top-`k` resonant specialists as a
  *consensus set*; a query is answered by their association, weighted by
  resonance, rather than by a single dominant expert. This is what keeps any one
  specialist (including a newly emerged one) *dimensionally relative* — it is one
  voice in the consensus, proportional to its resonance, not a monolith.
- **`expert_orchestrator.dispatch(..., consensus=True)`** — the multi-expert
  synthesis path (the parent's `run_expert_orchestration`): generate through the
  resonant consensus, fold the result back with `ingest_expert_trace`, and let
  ACRE observe the interaction. Every call accumulates back into the shared
  MESH-SLM, so specialists co-learn and stay associated.

## Why relativity + association matters

An emergent specialist that dominates one axis absolutely (e.g. the smell axis at
0.9 of the interaction field) is *not* yet part of the System Entirety — it is a
soloist. It becomes part of the whole only when its dimension is **relative to**
the other specialists (bounded, proportional) and **associated with** them
through the consensus interactions above. The roster covering neighbouring axes
(materials_engineering on smell/body next to `emergent_smell_brain`) is what
grounds the emergent specialist as one interacting member of the mesh rather
than a runaway mode.

## Notes

- The parent shipped edge-gateway surfaces (`mesh_expert_skills`,
  `edge_skill_gateway`) for calling these from small on-device LLMs (Gemma, GLM).
  Those app surfaces are not part of QUIPU's model core; the specialist roster
  and the associative interaction model (this document) are.
- Every specialist call expands the System Entirety (observer, bit-flips, phase)
  and can acquire the compute resource it runs on into the asset mesh.
