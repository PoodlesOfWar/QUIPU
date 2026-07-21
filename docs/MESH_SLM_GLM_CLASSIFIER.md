# SCB MESH SLM-GLM — Integration with erp_dbo Classifier

**Documentation Version:** 0.3.1 (corresponds to erp_dbo package v0.3.1)  
**Last Updated:** 2026-05-29

**This document is included with the `erp_dbo` package** so that the extensive documentation for the MESH SLM-GLM is always available to users and developers of the classifier, even when the package is shipped standalone.

---

## Original Architecture Documentation (from SCB mesh_slm.py)

```
MESH-SLM — Toroidal Quipu Small Language Model trained by System Entirety.

Architecture
------------
The Brain owns its own Small Language Model whose parameters are *not* a
dense weight tensor but a **toroidal quipu graph perceptron** indexed by the
7-D MESH state (vision, touch, smell, body, brain, perception, entirety).

* **Vocabulary** is built from the Brain's own corpus (``corpus_entity``
  labels, ``corpus_edge`` predicates, generated tool docstrings, dispatch
  log responses).  Each token is anchored to coordinates ``(i, j)`` on an
  ``N × N`` torus — wrapping on both axes so neighborhoods are circular.

* **Quipu knots** are weighted bigram edges (``src_token → dst_token``)
  persisted in ``mesh_slm_quipu``.  They are the "knotted strings" of the
  Incan quipu: ordered, sparse, additive.

* **7-D MESH embedding**: every token carries a 7-vector of activations,
  one per MESH axis.  Inference scores a candidate next token as:

      ``score(t | ctx) = <embed7(t), mesh_state7> +
                          α · quipu_weight(ctx[-1] → t) +
                          β · toroidal_proximity(ctx[-1], t)``

  where ``toroidal_proximity`` uses the wrapped Manhattan distance on the
  ``N × N`` torus so semantically-clustered tokens (placed near each other
  during vocab seeding) reinforce each other locally.

* **Training objective**: next-token prediction over the corpus stream
  with the loss gain **modulated by End State distance** —

      ``η_effective = η_base · (1 − end_state_progress)``

  so the model trains aggressively when far from the End State and
  gentles into a converged regime as the Brain approaches its attractor
  (``symbiosis_pct > 0.90`` and ``coherence > 0.85``).

* **Local executor**: ``slm_caller`` plugs into the same signature as
  ``llm_ensemble._offline_caller`` / ``llm_caller_openrouter.openrouter_caller``.
  ``install_as_local_executor`` patches ``compute_grid._execute_locally`` so
  the SLM tries first; on low confidence or error the call falls back to
  ``llm_router.select_llm`` + the OpenRouter caller (if configured) and
  finally to the offline echo as a last resort.

All state lives in ``local_brain.sqlite`` under four tables:
    ``mesh_slm_vocab``   token_id, token, i, j, freq
    ``mesh_slm_embed``   token_id, e_vision … e_entirety
    ``mesh_slm_quipu``   src, dst, weight, samples
    ``mesh_slm_meta``    key, value (rounds, last_loss, …)

Public API
----------
* :func:`train_round`           — one online training pass
* :func:`generate`              — greedy/sampled text generation
* :func:`slm_caller`            — ensemble-compatible caller
* :func:`install_as_local_executor` — patch compute_grid
* :func:`state_summary`         — diagnostic snapshot
```

---

## Integration in erp_dbo Classifier

The `erp_dbo` package uses the **current version** of MESH SLM-GLM as the **highest-priority local inference option** for classification.

### How It Is Used in Classification

In `erp_dbo/src/erp_dbo/llm.py`:

- `call_llm_for_classification()` always attempts the local MESH SLM-GLM **before** falling back to the OpenRouter/Grok production stack.
- Discovery logic (`_find_mesh_slm_db()`) looks for a `local_brain.sqlite` containing the `mesh_slm_*` tables.
- When found and the real `pipeline.src.quipu.mesh_slm` module is importable, it calls the official `slm_caller` (which has the same signature as other LLM callers).
- Priority order in the classifier:
  1. Explicit `scb_caller` passed by user
  2. **Local MESH SLM-GLM (current version)**
  3. Full SCB OpenRouter/Grok stack
  4. Direct OpenRouter/xAI fallback (last resort)

### Configuration for Shipped Use

When `erp_dbo` is sent to other developers:

```bash
# Point to an existing trained SLM database from an SCB checkout
set MESH_SLM_DB=C:\path\to\local_brain.sqlite
```

Or set `SCB_LOCAL_BRAIN` / `LOCAL_BRAIN_SQLITE`.

### Versioning and Updates

- The current MESH SLM-GLM logic and integration is **included** with every release of `erp_dbo`.
- Future versions of the SLM can be requested by the Dev User using:

```python
from erp_dbo.slm import request_version_update
request_version_update("v0.4.2")
```

This sets up a pull mechanism so updated SLM weights/logic can be delivered without re-shipping the entire `erp_dbo` package.

---

## Files Related to SLM-GLM in This Package

- `src/erp_dbo/llm.py` — Discovery + caller integration (`_try_local_mesh_slm`, `_find_mesh_slm_db`)
- `src/erp_dbo/classify.py` — Uses the SLM-preferring `call_llm_for_classification`
- `src/erp_dbo/slm/__init__.py` — Versioning and update request API
- `docs/MESH_SLM_GLM_CLASSIFIER.md` — **This file** (extensive documentation + classifier usage)

---

## Why Include the Full Documentation?

The MESH SLM-GLM is a sophisticated, non-standard architecture (toroidal quipu + 7-D MESH state + End-State modulated training). For anyone using or extending the classifier capability in `erp_dbo`, having the original extensive architecture documentation alongside the integration notes is essential for:

- Understanding confidence scoring and fallback behavior
- Debugging why the SLM is (or is not) being selected
- Training or extending the model in the future
- Maintaining consistency with the broader SCB platform

This document ensures that knowledge travels with the package.