"""qpsi — residual operator, Weyl channel, displacement-gated edges, and the
v0.9.25 Governance Protocol (six Physical Gates).  Stdlib only; additive."""
from .cat_residual import *   # noqa: F401,F403
from .weyl_channel import *   # noqa: F401,F403
from .edge_gate import *      # noqa: F401,F403
from .governance import *     # noqa: F401,F403
from .residual_checkpoint import *  # noqa: F401,F403
from .emergence_detector import *  # noqa: F401,F403
from . import interstitial  # noqa: F401  (submodule; its names KV_PREFIX/LINEAGE/record would shadow others)
from . import flux_phase, memristive_axes, learned_prior, mirror_training  # noqa: F401  (submodules; generic names like step/rows/field stay namespaced; self_organising is imported by src/quipu/__init__ under its flag)
