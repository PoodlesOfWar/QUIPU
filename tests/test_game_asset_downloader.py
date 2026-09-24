"""Tests for QUIPU Multi-Game Asset Downloader and Client Daemons."""

import json
from pathlib import Path
import pytest

from quipu.game_asset_downloader import (
    download_gw_client,
    download_osrs_client,
    download_wow_client,
)
from quipu.games.gw_client_daemon import GWEnvironmentState
from quipu.games.osrs_client_daemon import OSRSEnvironmentState
from quipu.games.wow_client_daemon import WoWEnvironmentState


def test_download_wow_client_structure(tmp_path: Path):
    """Verify that WoW 1.12.1 client setup creates required MPQ, realmlist, and WTF directories."""
    res = download_wow_client(tmp_path / "wow")
    assert res.status == "downloaded"
    assert (tmp_path / "wow" / "WoW.exe").exists()
    assert (tmp_path / "wow" / "realmlist.wtf").exists()
    assert (tmp_path / "wow" / "WTF" / "Config.wtf").exists()
    assert (tmp_path / "wow" / "Data" / "patch.mpq").exists()
    assert (tmp_path / "wow" / "manifest.json").exists()


def test_download_osrs_client_structure(tmp_path: Path):
    """Verify that OSRS client setup downloads RuneLite and creates launch properties."""
    res = download_osrs_client(tmp_path / "osrs")
    assert res.status == "downloaded"
    assert (tmp_path / "osrs" / "RuneLite.jar").exists()
    assert (tmp_path / "osrs" / "runelite.properties").exists()
    assert (tmp_path / "osrs" / "launch_osrs.sh").exists()
    assert (tmp_path / "osrs" / "manifest.json").exists()


def test_download_gw_client_structure(tmp_path: Path):
    """Verify that Guild Wars client setup downloads installer and initializes Gw.exe."""
    res = download_gw_client(tmp_path / "gw")
    assert res.status == "downloaded"
    assert (tmp_path / "gw" / "GwSetup.exe").exists()
    assert (tmp_path / "gw" / "Gw.exe").exists()
    assert (tmp_path / "gw" / "launch_gw.sh").exists()
    assert (tmp_path / "gw" / "manifest.json").exists()


def test_environment_states_serialization():
    """Verify that OSRS, WoW, and GW environment states serialize to JSON cleanly."""
    osrs_state = OSRSEnvironmentState()
    wow_state = WoWEnvironmentState()
    gw_state = GWEnvironmentState()

    d_osrs = osrs_state.to_dict()
    d_wow = wow_state.to_dict()
    d_gw = gw_state.to_dict()

    assert d_osrs["game"] == "osrs"
    assert d_wow["game"] == "wow"
    assert d_gw["game"] == "gw"
    assert "player_state" in d_osrs
    assert "player_state" in d_wow
    assert "player_state" in d_gw
