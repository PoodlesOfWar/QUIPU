"""Warden-gated embodied agents.

A World 4 NPC can be played by a Pollen Robotics Reachy 2, and signed agents can
take part in a hosted game, but only on the strength of a record a Warden signed,
re-checked on every action. See ``docs/WARDEN_EMBODIMENT.md``.

The package is self-contained and does not import the rest of QUIPU at load time,
so it can be lifted into whichever repo owns World 4. Its only QUIPU touchpoint is
:class:`BrainKVStore`, which imports ``brain_kv`` lazily.
"""
from ._common import (
    BrainKVStore,
    KVStore,
    MemoryStore,
    WardenAuthorityError,
    WardenError,
    WardenFormatError,
    WardenSecretError,
)
from .audit import AuditLog
from .essay_gate import EssayGate, ProgressSource, SignedRecordProgress, StageDecision
from .game_adapters import (
    AdapterNotWired,
    DispatchResult,
    DisclosureMissing,
    GameAdapter,
    GameDispatcher,
    GuildWarsAdapter,
    RecordingAdapter,
    WowLegacyAdapter,
    require_disclosure,
)
from .gate import Decision, WardenGate
from .npc_embodiment import (
    ActResult,
    BindingDenied,
    BoundedIntent,
    DriverRejected,
    IntentRejected,
    NpcEmbodiment,
    Reachy2Driver,
    RobotDriver,
    SimDriver,
    UnsupportedIntent,
    parse_intent,
)
from .operator_attestation import (
    OperatorAttestation,
    OperatorRegistry,
    OperatorSigner,
    verify_attestation,
)
from .signed_agent import (
    AUTOMATED_DECLARATION,
    AgentIdentity,
    AgentRegistry,
    NonceCache,
    SignedRequest,
    verify_request,
)
from .warden_grant import (
    ACTION_CLASSES,
    Grant,
    GrantStore,
    StaticWardenRegistry,
    WardenRegistry,
    issue_grant,
    parse_target,
    verify_grant,
)

from types import ModuleType as _ModuleType

# Public names only: classes, functions and constants. Submodule objects (``audit``,
# ``gate`` ...) are excluded so ``from ... import *`` cannot shadow a caller's variables.
__all__ = sorted(
    name for name, value in dict(globals()).items()
    if not name.startswith("_") and not isinstance(value, _ModuleType)
)
