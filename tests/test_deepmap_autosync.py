"""autosync.sh: schedules the deepmap node through Termux:API's job scheduler -- and refuses to
unless setup is complete.  Run for real as a subprocess against stand-in Termux commands."""
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "clients" / "termux" / "autosync.sh"
ALL_FLAGS = "--job-id --script --period-ms --network --persisted --cancel --pending"

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")

WIFI = """#!/usr/bin/env python3
import json, os
print(json.dumps({"bssid": "aa:bb:cc:dd:ee:ff", "frequency_mhz": 5180, "ip": "192.168.1.23",
    "link_speed_mbps": 433, "rssi": -54, "ssid": os.environ.get("FAKE_SSID", '"Test Net"'),
    "supplicant_state": "COMPLETED"}))
"""

SCHEDULER = """#!/bin/sh
# stand-in termux-job-scheduler: logs every call, answers --help with FAKE_FLAGS
echo "$@" >> "$FAKE_LOG"
case "$1" in --help) echo "usage: termux-job-scheduler $FAKE_FLAGS"; exit 0;; esac
exit 0
"""


@pytest.fixture()
def env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("termux-wifi-connectioninfo", WIFI), ("termux-job-scheduler", SCHEDULER)):
        f = bin_dir / name
        f.write_text(body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    home = tmp_path / "home"
    (home / ".config" / "quipu").mkdir(parents=True)
    log = tmp_path / "scheduler.log"
    log.write_text("")
    return {"bin": bin_dir, "home": home, "log": log, "conf": home / ".config" / "quipu" / "deepmap.env"}


def _write_conf(env, ssids="Test Net", url="https://quipu.example.com"):
    lines = []
    if ssids:
        lines.append(f"DEEPMAP_SSIDS={ssids}")
    if url:
        lines.append(f"QUIPU_URL={url}")
    env["conf"].write_text("\n".join(lines) + "\n")


def _run(env, *args, flags=ALL_FLAGS, extra=None, path_dirs=None):
    # python's own directory must stay on PATH; the rest is deliberately minimal
    base = [str(env["bin"])] if path_dirs is None else path_dirs
    e = {"PATH": os.pathsep.join(base + [str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
         "HOME": str(env["home"]), "FAKE_LOG": str(env["log"]), "FAKE_FLAGS": flags}
    e.update(extra or {})
    return subprocess.run(["sh", str(SCRIPT), *args], capture_output=True, text=True, env=e, timeout=60)


def _calls(env):
    return [ln for ln in env["log"].read_text().splitlines() if ln.strip()]


def _scheduled(env):
    return [c for c in _calls(env) if "--script" in c]


def test_enable_schedules_with_the_right_constraints(env):
    _write_conf(env)
    r = _run(env, "enable")
    assert r.returncode == 0, r.stderr
    (call,) = _scheduled(env)
    wrap = env["home"] / ".config" / "quipu" / "deepmap_job.sh"
    assert call == f"--job-id 4242 --script {wrap} --period-ms 900000 --network unmetered --persisted true"
    body = wrap.read_text()
    assert "deepmap_node.py --send" in body and str(ROOT) in body and "deepmap.log" in body
    assert stat.S_IMODE(wrap.stat().st_mode) == 0o700
    assert "every 15 min" in r.stdout


def test_enable_respects_overrides(env):
    _write_conf(env)
    r = _run(env, "enable", extra={"DEEPMAP_JOB_ID": "77", "DEEPMAP_PERIOD_MS": "1800000"})
    assert r.returncode == 0 and "--job-id 77 " in _scheduled(env)[0] and "--period-ms 1800000" in _scheduled(env)[0]


def test_disable_and_status(env):
    assert _run(env, "disable").returncode == 0
    assert "--cancel --job-id 4242" in _calls(env)[-1]
    assert _run(env, "status").returncode == 0
    assert _calls(env)[-1] == "--pending"


def test_refuses_without_the_scheduler(env):
    _write_conf(env)
    os.remove(env["bin"] / "termux-job-scheduler")
    r = _run(env, "enable")
    assert r.returncode == 3 and "termux-job-scheduler not found" in r.stderr and "Termux:API" in r.stderr


@pytest.mark.parametrize("missing", ["--persisted", "--network", "--period-ms"])
def test_refuses_when_a_flag_it_needs_is_not_there(env, missing):
    _write_conf(env)
    flags = " ".join(f for f in ALL_FLAGS.split() if f != missing)
    r = _run(env, "enable", flags=flags)
    assert r.returncode == 3 and f"does not list {missing}" in r.stderr
    assert _scheduled(env) == []                                   # it guessed nothing


def test_refuses_when_the_node_is_not_set_up(env):
    _write_conf(env)
    r = _run(env, "enable", extra={"FAKE_SSID": "<unknown ssid>"})     # Android is hiding the network name
    assert r.returncode == 3 and "--check failed" in r.stderr and _scheduled(env) == []


def test_refuses_with_an_empty_allow_list(env):
    _write_conf(env, ssids="")
    r = _run(env, "enable")
    assert r.returncode == 3 and "DEEPMAP_SSIDS is empty" in r.stderr and _scheduled(env) == []


@pytest.mark.parametrize("url", ["", "http://quipu.example.com", "quipu.example.com"])
def test_refuses_without_an_https_front_door(env, url):
    _write_conf(env, url=url)
    r = _run(env, "enable")
    assert r.returncode == 3 and "QUIPU_URL=https://" in r.stderr and _scheduled(env) == []


def test_unknown_subcommand(env):
    r = _run(env, "launch-missiles")
    assert r.returncode == 3 and "usage" in r.stderr
