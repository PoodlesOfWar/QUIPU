"""QUIPU Multi-Account Credential and Profile Store.

Manages multi-account inputs across Old School RuneScape, World of Warcraft,
and Guild Wars. Stores credentials securely with AES-256-GCM encryption,
tracks account status, rotation schedules, and VPN node associations.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from quipu import brain_kv

logger = logging.getLogger("quipu.games.account_store")

_BRAIN_KV_ACCOUNT_KEY = "games:multi_accounts:store"
_DEFAULT_KEY_HEX = "4172636869746563747572655365637265744b65795155495055323032362121"  # 32-byte key


def _get_encryption_key() -> bytes:
    key_hex = os.environ.get("QUIPU_ACCOUNT_STORE_KEY", _DEFAULT_KEY_HEX)
    try:
        raw = bytes.fromhex(key_hex)
        if len(raw) == 32:
            return raw
    except Exception:
        pass
    return bytes.fromhex(_DEFAULT_KEY_HEX)


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret string using AES-256-GCM returning base64 string."""
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), b"QUIPU_ACCOUNT_STORE_V1")
    payload = nonce + ciphertext
    return base64.b64encode(payload).decode("ascii")


def decrypt_secret(encrypted_b64: str) -> str:
    """Decrypt a base64 AES-256-GCM string back to plaintext."""
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    raw = base64.b64decode(encrypted_b64.encode("ascii"))
    nonce = raw[:12]
    ciphertext = raw[12:]
    plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, b"QUIPU_ACCOUNT_STORE_V1")
    return plaintext_bytes.decode("utf-8")


@dataclass
class AccountProfile:
    account_id: str
    game: str  # "osrs", "wow", "gw"
    username: str
    character_name: str
    secret_enc: str  # AES-256-GCM encrypted
    realm_or_world: str = ""
    pin: str = ""
    assigned_vpn_node: str = ""
    status: str = "idle"  # "idle", "authenticated", "in_game", "challenge_required", "cooldown"
    last_login: Optional[float] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_secret: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_secret:
            d["secret_enc"] = "[ENCRYPTED]"
        return d

    def get_decrypted_secret(self) -> str:
        return decrypt_secret(self.secret_enc)


class AccountStore:
    """Persistent multi-account credential manager."""

    def __init__(self, storage_file: Optional[Path] = None, load_from_kv: bool = True) -> None:
        self.storage_file = storage_file or Path("quipu_learned_artifacts/accounts_store.json")
        self.load_from_kv = load_from_kv
        self._accounts: dict[str, AccountProfile] = {}
        self.load()
        if not self._accounts:
            self._seed_default_accounts()

    def add_account(
        self,
        account_id: str,
        game: str,
        username: str,
        password: str,
        character_name: str,
        realm_or_world: str = "",
        pin: str = "",
        assigned_vpn_node: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AccountProfile:
        """Add or update an account with automatic secret encryption."""
        enc_secret = encrypt_secret(password)
        profile = AccountProfile(
            account_id=account_id,
            game=game.lower(),
            username=username,
            character_name=character_name,
            secret_enc=enc_secret,
            realm_or_world=realm_or_world,
            pin=pin,
            assigned_vpn_node=assigned_vpn_node,
            status="idle",
            tags=tags or [],
            metadata=metadata or {},
        )
        self._accounts[account_id] = profile
        self.save()
        logger.info("Saved account %s for game %s", account_id, game)
        return profile

    def get_account(self, account_id: str) -> Optional[AccountProfile]:
        return self._accounts.get(account_id)

    def list_accounts(self, game: Optional[str] = None) -> list[AccountProfile]:
        if game is None or game.lower() == "all":
            return list(self._accounts.values())
        return [acc for acc in self._accounts.values() if acc.game.lower() == game.lower()]

    def rotate_account(self, game: str) -> Optional[AccountProfile]:
        """Select next idle account for a given game."""
        candidates = [acc for acc in self.list_accounts(game) if acc.status == "idle"]
        if not candidates:
            # Fall back to any non-challenge account
            candidates = [acc for acc in self.list_accounts(game) if acc.status != "challenge_required"]
        if not candidates:
            return None
        # Sort by least recently logged in
        candidates.sort(key=lambda a: a.last_login or 0.0)
        return candidates[0]

    def update_status(self, account_id: str, status: str, last_login: Optional[float] = None) -> bool:
        acc = self.get_account(account_id)
        if not acc:
            return False
        acc.status = status
        if last_login is not None:
            acc.last_login = last_login
        self.save()
        return True

    def _seed_default_accounts(self) -> None:
        """Seed initial multi-account profiles across OSRS, WoW, and GW."""
        self.add_account(
            account_id="osrs_quipubot_main",
            game="osrs",
            username="quipu_explorer@osrs.local",
            password="GARD_Hash_OSRS_2026!",
            character_name="QuipuBot",
            realm_or_world="World 301 (Free)",
            pin="1984",
            assigned_vpn_node="quipu-osrs-node1",
            tags=["main", "skilling", "combat75"],
        )
        self.add_account(
            account_id="osrs_pure_alt",
            game="osrs",
            username="quipu_pure@osrs.local",
            password="GARD_Hash_OSRS_Pure!",
            character_name="QuipuPure",
            realm_or_world="World 302 (Trade)",
            pin="2048",
            assigned_vpn_node="quipu-osrs-node2",
            tags=["alt", "pvp_pure", "combat50"],
        )
        self.add_account(
            account_id="wow_warrior_main",
            game="wow",
            username="quipu_warrior",
            password="WoW_Classic_Turtle_2026!",
            character_name="QuipuWarrior",
            realm_or_world="logon.turtle-wow.org",
            assigned_vpn_node="quipu-wow-node1",
            tags=["main", "tank", "level15", "westfall"],
        )
        self.add_account(
            account_id="wow_mage_alt",
            game="wow",
            username="quipu_mage",
            password="WoW_Classic_Mage_2026!",
            character_name="QuipuMage",
            realm_or_world="logon.turtle-wow.org",
            assigned_vpn_node="quipu-wow-node2",
            tags=["alt", "dps", "aoe_frost", "level20"],
        )
        self.add_account(
            account_id="gw_ele_main",
            game="gw",
            username="quipu_elementalist@guildwars.local",
            password="ArenaNet_Prophecies_2026!",
            character_name="Quipu Elementalist",
            realm_or_world="America-1 (Droknar's Forge)",
            assigned_vpn_node="quipu-gw-node1",
            tags=["main", "fire_magic", "level20", "droknars"],
        )
        self.add_account(
            account_id="gw_monk_alt",
            game="gw",
            username="quipu_monk@guildwars.local",
            password="ArenaNet_Factions_2026!",
            character_name="Quipu Monk",
            realm_or_world="America-1 (Kaineng)",
            assigned_vpn_node="quipu-gw-node2",
            tags=["alt", "healer", "level20", "factions"],
        )

    def save(self) -> None:
        """Persist to brain_kv and file cache."""
        data = {acc_id: acc.to_dict(include_secret=True) for acc_id, acc in self._accounts.items()}
        if self.load_from_kv:
            try:
                brain_kv.kv_set_json(_BRAIN_KV_ACCOUNT_KEY, data)
            except Exception as exc:
                logger.debug("Failed to write to brain_kv: %s", exc)

        try:
            self.storage_file.parent.mkdir(parents=True, exist_ok=True)
            self.storage_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to write account file: %s", exc)

    def load(self) -> None:
        """Load from brain_kv or fallback to file."""
        data = None
        if self.load_from_kv:
            try:
                data = brain_kv.kv_get_json(_BRAIN_KV_ACCOUNT_KEY, None)
            except Exception:
                pass

        if not data and self.storage_file.exists():
            try:
                data = json.loads(self.storage_file.read_text(encoding="utf-8"))
            except Exception:
                data = None

        if isinstance(data, dict):
            for acc_id, d in data.items():
                if isinstance(d, dict):
                    self._accounts[acc_id] = AccountProfile(
                        account_id=d["account_id"],
                        game=d["game"],
                        username=d["username"],
                        character_name=d["character_name"],
                        secret_enc=d["secret_enc"],
                        realm_or_world=d.get("realm_or_world", ""),
                        pin=d.get("pin", ""),
                        assigned_vpn_node=d.get("assigned_vpn_node", ""),
                        status=d.get("status", "idle"),
                        last_login=d.get("last_login"),
                        tags=d.get("tags", []),
                        metadata=d.get("metadata", {}),
                    )
