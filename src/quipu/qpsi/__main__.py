"""``python -m src.quipu.qpsi <command>`` — the self-organising CLI's entry point.

With QUIPU_SELF_ORGANISING=1 the package import (src/quipu/__init__.py)
already imports ``qpsi.self_organising`` to wire the loop.  Running that
module itself with ``-m`` then makes runpy execute a second copy of it as
``__main__`` and warn:

    RuntimeWarning: 'src.quipu.qpsi.self_organising' found in sys.modules
    after import of package 'src.quipu.qpsi', but prior to execution of
    'src.quipu.qpsi.self_organising'; this may result in unpredictable behaviour

The second copy dispatches to the first (``self_organising._main``), so the
behaviour was defined — but the warning was printed on every pulse.  This
module is imported by nothing, so runpy has nothing to warn about, and the
command is shorter:

    python -m src.quipu.qpsi status
    python -m src.quipu.qpsi pulse --route
    python -m src.quipu.qpsi planes | accrete [--limit N] | fibre TOKEN

``python -m src.quipu.qpsi.self_organising …`` still works as before.
"""
from .self_organising import _main

if __name__ == "__main__":
    raise SystemExit(_main())
