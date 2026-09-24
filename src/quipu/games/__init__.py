"""QUIPU Dedicated Game Client Containers & Daemons.

Contains client runners, session handlers, and monitoring daemons for:
- Old School RuneScape (OSRS): RuneLite Java client on port 7310
- World of Warcraft (WoW): Classic client on port 7320
- Guild Wars 1 (GW): ArenaNet client on port 7330

Also provides:
- MassSessionHandler: Multi-container mass login & r-ADMIN coordination
- AccountStore: Multi-account credential management with AES-256-GCM encryption
- TailscaleMeshIntegration: Tailscale-style mesh VPN overlay (100.64.0.0/10)
- Gate6UserInterlock: Physical Gate 6 user confirmation & refinement interlock
"""

from quipu.games.account_store import AccountProfile, AccountStore
from quipu.games.gate6_user_interlock import (
    BreakageType,
    Gate6UserInterlock,
    RefinementBreakageEvent,
    get_global_interlock,
)
from quipu.games.mass_session_handler import MassSessionHandler
from quipu.games.vpn_mesh_integration import MeshNode, TailscaleMeshIntegration

__all__ = [
    "AccountProfile",
    "AccountStore",
    "BreakageType",
    "Gate6UserInterlock",
    "MassSessionHandler",
    "MeshNode",
    "RefinementBreakageEvent",
    "TailscaleMeshIntegration",
    "get_global_interlock",
]

