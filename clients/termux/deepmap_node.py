#!/usr/bin/env python3
"""deepmap_node — a Termux node for QUIPU's deepmap.

Runs on an Android phone under Termux (with the Termux:API app).  It reports the
Wi-Fi link the phone is on and whether the BurgerPhone peers you list answer,
as one ``deepmap.link/1`` record per run, to QUIPU's ``POST /deepmap/link``
through the signed edge client.  QUIPU assembles every node's record into one
graph at ``GET /deepmap``.

What it will and will not do
----------------------------
* **Dry run by default.**  Without ``--send`` it prints the record and stops, so
  you can read exactly what would leave the phone.
* **Fail closed on the network.**  It acts only while the phone is connected to
  an SSID in ``DEEPMAP_SSIDS`` (a list you set: there is no default).  Off that
  network, or with the list empty, it does nothing at all -- no probe, no send.
* **No scanning.**  It does not sweep the LAN.  It connects (and immediately
  closes, sending no data) to the ``DEEPMAP_PEERS`` you list, to time them, and
  only when the peer resolves to a private, link-local or loopback address.
* **No raw network identifiers leave the phone.**  The access point and the
  network are sent as salted hashes (the salt is generated on this phone and
  stays here), so the map can say "the same access point" without saying which
  one.  QUIPU refuses any record containing a MAC address.

Configuration (environment, or ``~/.config/quipu/deepmap.env`` as KEY=VALUE lines)
    DEEPMAP_SSIDS     comma-separated SSIDs this node may act on (required)
    DEEPMAP_PEERS     comma-separated ``name=host:port`` BurgerPhone peers to time
    DEEPMAP_LABEL     a human label for this phone (default: the Termux hostname)
    QUIPU_URL         the https address of QUIPU's front door (operator's tunnel)
    QUIPU_EDGE_KEY_BURGERPHONE / QUIPU_EDGE_KEY   the edge key for source burgerphone

Usage
    python clients/termux/deepmap_node.py --check            # setup + permission report
    python clients/termux/deepmap_node.py                    # dry run: print the record
    python clients/termux/deepmap_node.py --send             # send once
    python clients/termux/deepmap_node.py --send --loop 300  # every 5 minutes

Stdlib only.  See docs/DEEPMAP_TERMUX.md.
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

SCHEMA = "deepmap.link/1"
SOURCE = "burgerphone"
MAX_PEERS = 32
PROBE_TIMEOUT_S = 1.5
TERMUX_TIMEOUT_S = 10.0
CONFIG_PATH = Path.home() / ".config" / "quipu" / "deepmap.env"
SALT_PATH = Path.home() / ".config" / "quipu" / "deepmap_salt"

EXIT_OK, EXIT_SEND_FAILED, EXIT_SETUP = 0, 1, 3

_NAME = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
_UNKNOWN_SSID = {"", "<unknown ssid>", "unknown ssid", "0x"}

PERMISSION_HINT = (
    "Android hides the Wi-Fi name from an app that lacks permission. In Android Settings > Apps > "
    "Termux:API > Permissions, allow Location (and, on Android 13 or later, Nearby devices), "
    "keep Location switched on, and make sure Wi-Fi is connected."
)

Runner = Callable[[list, float], "tuple[int, str]"]


class SetupError(RuntimeError):
    """Something the user must fix on the phone; the message says what."""


def run_command(argv: list, timeout: float = TERMUX_TIMEOUT_S) -> tuple[int, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""


# --------------------------------------------------------------------------- config
def load_env_file(path: Path = CONFIG_PATH) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def setting(name: str, env: Mapping[str, str], file_env: Mapping[str, str], default: str = "") -> str:
    return (env.get(name) or file_env.get(name) or default).strip()


def load_salt(path: Path = SALT_PATH) -> str:
    """A per-phone random salt, created on first use and never sent anywhere."""
    try:
        s = path.read_text(encoding="utf-8").strip()
        if len(s) >= 32:
            return s
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    s = secrets.token_hex(16)
    path.write_text(s + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return s


def hash12(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\n{value}".encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- Wi-Fi
def norm_ssid(raw: Any) -> str:
    s = str(raw or "").strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        s = s[1:-1]
    return "" if s.strip().lower() in _UNKNOWN_SSID else s


def read_wifi(run: Optional[Runner] = None) -> dict[str, Any]:
    run = run or run_command
    code, out = run(["termux-wifi-connectioninfo"], TERMUX_TIMEOUT_S)
    if code == 127:
        raise SetupError("termux-wifi-connectioninfo not found. Install the Termux:API app (from the same "
                         "source as Termux) and run: pkg install termux-api")
    if code != 0 or not out.strip():
        raise SetupError("termux-wifi-connectioninfo returned nothing. Is the Termux:API app installed and "
                         "allowed to run in the background? " + PERMISSION_HINT)
    try:
        data = json.loads(out)
    except ValueError as exc:
        raise SetupError(f"could not read termux-wifi-connectioninfo output: {exc}") from exc
    if not isinstance(data, dict):
        raise SetupError("unexpected termux-wifi-connectioninfo output")
    if "error" in data:
        raise SetupError(f"termux-wifi-connectioninfo error: {data['error']}. {PERMISSION_HINT}")
    return data


# --------------------------------------------------------------------------- peers
@dataclass(frozen=True)
class Peer:
    name: str
    host: str
    port: int


def parse_peers(spec: str) -> list[Peer]:
    peers: list[Peer] = []
    for part in [p for p in (spec or "").split(",") if p.strip()]:
        if "=" not in part or ":" not in part.split("=", 1)[1]:
            raise ValueError(f"peer {part!r} must look like name=host:port")
        name, hostport = part.split("=", 1)
        host, port_s = hostport.strip().rsplit(":", 1)
        name, host = name.strip(), host.strip().strip("[]")
        if not _NAME.match(name):
            raise ValueError(f"peer name {name!r} must be 1-32 letters, digits, . _ -")
        if not port_s.isdigit() or not 1 <= int(port_s) <= 65535:
            raise ValueError(f"peer {name!r}: port must be 1-65535")
        if not host:
            raise ValueError(f"peer {name!r}: host is empty")
        peers.append(Peer(name, host, int(port_s)))
    if len(peers) > MAX_PEERS:
        raise ValueError(f"at most {MAX_PEERS} peers")
    return peers


def _is_local(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def probe(peer: Peer, *, timeout: float = PROBE_TIMEOUT_S,
          resolver: Callable[..., Any] = socket.getaddrinfo,
          connect: Callable[..., Any] = socket.create_connection,
          clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """Time a bare TCP connect to one listed peer.  No data is sent; the socket is closed at once.
    A peer that resolves to anything but a private, link-local or loopback address is not contacted."""
    edge: dict[str, Any] = {"to": f"peer-{peer.name}", "kind": "tcp", "port": peer.port}
    try:
        infos = resolver(peer.host, peer.port, type=socket.SOCK_STREAM)
    except OSError:
        edge["ok"] = False
        return edge
    addrs = [i[4][0] for i in infos]
    if not addrs or not all(_is_local(a) for a in addrs):
        edge["ok"] = False
        edge["refused"] = "not a local address"
        return edge
    t0 = clock()
    try:
        s = connect((addrs[0], peer.port), timeout=timeout)
    except OSError:
        edge["ok"] = False
        return edge
    edge["latency_ms"] = round((clock() - t0) * 1000.0, 1)
    edge["ok"] = True
    try:
        s.close()
    except OSError:
        pass
    return edge


# --------------------------------------------------------------------------- the record
def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def build_record(wifi: Mapping[str, Any], peer_edges: list, *, salt: str, label: str,
                 now: Optional[datetime] = None) -> dict[str, Any]:
    ssid = norm_ssid(wifi.get("ssid"))
    bssid = str(wifi.get("bssid") or "").strip().lower()
    node_id = "phone-" + hash12(salt, "self")
    ap_id = "ap-" + hash12(salt, "bssid:" + bssid) if bssid else None
    w: dict[str, Any] = {}
    if ap_id:
        w["ap"] = ap_id
    for src, dst in (("rssi", "rssi_dbm"), ("frequency_mhz", "freq_mhz"), ("link_speed_mbps", "link_mbps")):
        n = _num(wifi.get(src))
        if n is not None:
            w[dst] = n
    if wifi.get("supplicant_state"):
        w["state"] = str(wifi["supplicant_state"])
    edges = []
    if ap_id:
        e = {"to": ap_id, "kind": "wifi", "ok": True}
        for k in ("rssi_dbm", "link_mbps"):
            if k in w:
                e[k] = w[k]
        edges.append(e)
    edges.extend({k: v for k, v in pe.items() if k != "refused"} for pe in peer_edges)
    when = (now or datetime.now(timezone.utc)).isoformat()
    return {"agent_id": node_id, "schema": SCHEMA,
            "node": {"role": "phone", "label": label[:64], "network_id": "net-" + hash12(salt, "ssid:" + ssid)},
            "wifi": w, "edges": edges, "observed_at": when}


# --------------------------------------------------------------------------- one pass
@dataclass
class Outcome:
    status: str                      # recorded | off-network | no-allowlist | not-connected
    detail: str = ""
    record: Optional[dict] = None


def collect(*, ssids: list, peers: list, label: str, salt: str,
            run: Optional[Runner] = None, prober: Optional[Callable[[Peer], dict]] = None,
            now: Optional[datetime] = None) -> Outcome:
    prober = prober or probe
    if not ssids:
        return Outcome("no-allowlist", "DEEPMAP_SSIDS is empty: set the network(s) this node may act on")
    wifi = read_wifi(run)
    state = str(wifi.get("supplicant_state") or "COMPLETED").upper()
    ssid = norm_ssid(wifi.get("ssid"))
    if state not in ("COMPLETED", "ASSOCIATED") or not wifi.get("ip"):
        return Outcome("not-connected", "the phone is not connected to Wi-Fi")
    if not ssid:
        raise SetupError("connected to Wi-Fi, but Android is not telling Termux:API the network name. " + PERMISSION_HINT)
    allowed = {s.strip().lower() for s in ssids if s.strip()}
    if ssid.lower() not in allowed:
        return Outcome("off-network", "connected to a network that is not in DEEPMAP_SSIDS; nothing was probed or sent")
    peer_edges = [prober(p) for p in peers]
    return Outcome("recorded", "", build_record(wifi, peer_edges, salt=salt, label=label, now=now))


# --------------------------------------------------------------------------- sending
def make_client(url: str, key: str):
    here = Path(__file__).resolve()
    for cand in (here.parents[2] / "src" / "quipu", here.parent):
        if (cand / "edge_client.py").exists():
            sys.path.insert(0, str(cand))
            break
    import edge_client                      # single-file stdlib client
    return edge_client.QuipuEdgeClient(SOURCE, base_url=url or None, key=key or None, background=False)


def send(record: Mapping[str, Any], client) -> dict[str, Any]:
    """Queue, flush, and report whether *this* record was acknowledged (counters are
    cumulative, so success is the change in acknowledged writes, not their total)."""
    def acked(st: Mapping[str, Any]) -> int:
        return int(st.get("sent", 0)) + int(st.get("duplicate", 0))

    before = acked(client.stats())
    k = client.submit("/deepmap/link", {"source": SOURCE, "relational": dict(record)})
    if k is None:
        return {"ok": False, "error": "could not queue the record"}
    st = client.flush()
    return {"ok": acked(st) > before and client.pending() == 0, "stats": st}


# --------------------------------------------------------------------------- CLI
def check_report(run: Optional[Runner] = None) -> tuple[list, bool]:
    lines, ok = [], True
    try:
        w = read_wifi(run)
        ssid = norm_ssid(w.get("ssid"))
        lines.append("ok    Termux:API answered")
        lines.append("ok    Wi-Fi info readable" if w.get("ip") else "WARN  not connected to Wi-Fi")
        if w.get("ip") and not ssid:
            lines.append("FAIL  network name hidden: " + PERMISSION_HINT)
            ok = False
        elif ssid:
            lines.append("ok    network name visible (not printed)")
    except SetupError as exc:
        lines.append(f"FAIL  {exc}")
        ok = False
    return lines, ok


def main(argv: Optional[list] = None, env: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(prog="deepmap_node", description="Termux node for QUIPU's deepmap")
    ap.add_argument("--check", action="store_true", help="report setup and permission status, then exit")
    ap.add_argument("--send", action="store_true", help="send the record to QUIPU (default: print only)")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS", help="repeat every SECONDS (minimum 30)")
    ap.add_argument("--ssid", action="append", default=[], help="allowed SSID (repeatable; or DEEPMAP_SSIDS)")
    ap.add_argument("--peers", default=None, help="name=host:port,... (or DEEPMAP_PEERS)")
    ap.add_argument("--label", default=None)
    args = ap.parse_args(argv)

    if args.check:
        lines, ok = check_report()
        print("\n".join(lines))
        return EXIT_OK if ok else EXIT_SETUP

    file_env = load_env_file()
    ssids = args.ssid or [s for s in setting("DEEPMAP_SSIDS", env, file_env).split(",") if s.strip()]
    try:
        peers = parse_peers(args.peers if args.peers is not None else setting("DEEPMAP_PEERS", env, file_env))
    except ValueError as exc:
        print(f"deepmap_node: {exc}", file=sys.stderr)
        return EXIT_SETUP
    label = args.label or setting("DEEPMAP_LABEL", env, file_env) or socket.gethostname()
    url = setting("QUIPU_URL", env, file_env)
    key = setting("QUIPU_EDGE_KEY_BURGERPHONE", env, file_env) or setting("QUIPU_EDGE_KEY", env, file_env)
    interval = max(30, args.loop) if args.loop else 0
    client = make_client(url, key) if args.send else None
    rc = EXIT_OK
    while True:
        try:
            out = collect(ssids=ssids, peers=peers, label=label, salt=load_salt())
        except SetupError as exc:
            print(f"deepmap_node: {exc}", file=sys.stderr)
            return EXIT_SETUP
        if out.status != "recorded":
            print(f"deepmap_node: {out.status}: {out.detail}")
        elif client is None:
            print(json.dumps(out.record, indent=2, sort_keys=True))
            print("deepmap_node: dry run; nothing was sent. Re-run with --send to send it.")
        else:
            res = send(out.record, client)
            print("deepmap_node: sent" if res["ok"] else f"deepmap_node: not sent ({res})")
            rc = EXIT_OK if res["ok"] else EXIT_SEND_FAILED
        if not interval:
            return rc
        time.sleep(interval)


if __name__ == "__main__":          # pragma: no cover
    sys.exit(main())
