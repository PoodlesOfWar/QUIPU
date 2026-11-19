"""Source text for held ACRE tool candidates (qpsi/tool_gate.py).

Modules here are never imported at runtime. ``tool_gate`` reads a file as text,
lints it, hashes it, and holds it at gate 6. A tool becomes callable only after
both parties attest to its exact scope and the realise grant is set.
"""
