# deepmap on Termux — mapping BurgerPhone connections from a phone

`clients/termux/deepmap_node.py` runs on an Android phone under Termux and
reports two things to QUIPU's **deepmap**: the Wi-Fi link the phone is on, and
whether the BurgerPhone peers you list answer. QUIPU assembles every reporting
phone into one graph at `GET /deepmap`.

It is deliberately small and does only what you tell it to:

| it does | it does not |
|---|---|
| read the phone's own Wi-Fi link via Termux:API | scan or sweep your LAN |
| time a bare TCP connect to **peers you list** (no data sent) | contact anything that is not a private, link-local or loopback address |
| act only while connected to an SSID **you allow-list** | do anything with no allow-list, or on any other network |
| send salted hashes of the access point and network | send a MAC, BSSID, SSID or IP — QUIPU refuses any record containing a MAC |
| print what it would send, and **send nothing until you pass `--send`** | run in the background unless you start it |

## What I verified, and what I could not

Verified here: the node, its safeguards and the QUIPU side (23 tests, including
a run from the node through the signed edge client into a live observer), and
the command-line flow against a stand-in `termux-wifi-connectioninfo`.

**Not verified:** anything on a real phone. This was built without access to the
device or to Termux's own documentation, so the exact fields `termux-wifi-connectioninfo`
returns on your Android version, and which Android permission each Termux:API
command needs, are untested. `--check` is there to tell you; it reports a hidden
network name and names the fix rather than guessing.

## 1. Install (once)

Install **Termux** and the **Termux:API** app from the *same* source (both from
F-Droid, or both from GitHub releases; apps signed by different sources cannot
talk to each other), then in Termux:

```sh
pkg update && pkg install python git termux-api
git clone https://github.com/PoodlesOfWar/QUIPU
cd QUIPU
```

## 2. The permission prompt ("connect to nearby devices")

Android asks per app. Grant it to **Termux:API** (the app that answers the
commands), in *Android Settings → Apps → Termux:API → Permissions*:

- **Nearby devices** (Android 13+; this is the group Android's Wi-Fi APIs use for
  "nearby Wi-Fi devices") — allow it.
- **Location** — allow it and keep Location switched on. Android hides the Wi-Fi
  network name from an app that has no location access, and the node needs the
  name to know whether it is on your network.

This node uses the Wi-Fi link information only. It does **not** use Bluetooth:
I could not confirm a Termux:API command for Bluetooth scanning, so nothing
here pretends to.

Then check:

```sh
python clients/termux/deepmap_node.py --check
```

It prints `ok`/`WARN`/`FAIL` lines and never prints the network name.

## 3. Configure

Create `~/.config/quipu/deepmap.env` (or export the same variables):

```sh
DEEPMAP_SSIDS=Your Network Name          # comma-separated; required. No default.
DEEPMAP_PEERS=kitchen=192.168.1.9:8765,den=den.local:8765   # name=host:port, BurgerPhone peers to time
DEEPMAP_LABEL=my phone
QUIPU_URL=https://quipu.example.com      # the operator's https front door (see below)
QUIPU_EDGE_KEY_BURGERPHONE=<hex key>     # minted by the operator for source "burgerphone"
```

`DEEPMAP_SSIDS` has no default on purpose: with it empty the node does nothing.
The SSID is a convenience guard against acting on the wrong network, not a
security boundary (an SSID can be spoofed).

## 4. Run

```sh
python clients/termux/deepmap_node.py                  # dry run: prints the record, sends nothing
python clients/termux/deepmap_node.py --send           # send once
termux-wake-lock
python clients/termux/deepmap_node.py --send --loop 300   # every 5 minutes (minimum 30 s)
```

Android may stop Termux in the background: exempt Termux and Termux:API from
battery optimisation if you want the loop to keep running.

Check the result from any machine that can reach QUIPU:

```sh
curl -s "$QUIPU_URL/deepmap" | python -m json.tool
```

## Automatic sync

```sh
sh clients/termux/autosync.sh enable     # schedule it
sh clients/termux/autosync.sh status     # pending Termux jobs
sh clients/termux/autosync.sh disable    # cancel it
```

`enable` uses Termux:API's job scheduler: Android's JobScheduler runs the node
about every 15 minutes (the minimum Android allows; `DEEPMAP_PERIOD_MS` to
change) while the phone is on an unmetered network, and keeps the job across
reboots. The node still acts only on an SSID in `DEEPMAP_SSIDS`, so a job that
fires on some other Wi-Fi does nothing. Android may delay jobs to save battery;
if it runs late, exempt Termux and Termux:API from battery optimisation.

`enable` **refuses to schedule** unless: Termux:API answers and the network name
is visible (`--check`), `DEEPMAP_SSIDS` is set, `QUIPU_URL` is an `https://`
address, and this Termux:API's `termux-job-scheduler` lists every flag the
script uses (it checks rather than assumes, because I could not read Termux's own
docs). Put the configuration in `~/.config/quipu/deepmap.env`; a scheduled job
does not see your shell's exports. Output goes to `~/.config/quipu/deepmap.log`
(trimmed when it grows).

## About "Android Switch" and "All files access"

Android's **All files access** screen lists only apps that *declare* that
permission, and nothing — no app, no script, no Claude — can switch it on for
another app: Android makes the person do it in Settings, per app. On the
screen as I saw it, the apps listed were Android Auto, Android Switch, Copilot,
Cross-Device Services, F-Droid, Files by Google, Hub, Link to Windows and
Termux; there was no BurgerPhone Android app, so there was nothing to set to
Allowed.

Android Switch is Google's one-time device-setup transfer tool, not a sync
service: it has no hook another app can use to sync through it. The automatic
sync above is the supported equivalent, and it does not need All files access at
all (the node reads and writes only Termux's own private
`~/.config/quipu`). I would leave All files access off for anything that does not
need it; Termux already has it, and F-Droid and Hub being "Not allowed" is a
sound default.

## 5. Reaching QUIPU from the phone

QUIPU listens on `127.0.0.1:7100` on the fleet host, so the phone needs a front
door. The fleet compose has a named Cloudflare tunnel for it
(`docker compose --profile tunnel-quipu up -d` with `QUIPU_TUNNEL_TOKEN`); set
`QUIPU_URL` to its `https://` address. The edge client refuses plain `http://` to
anything public, so the key never crosses the internet in clear. Anything
arriving through a public front door must be signed, whatever
`QUIPU_EDGE_AUTH` says.

Under `QUIPU_EDGE_AUTH=enforce` the operator must mint a key for source
`burgerphone` and list it in the grant (`ops/New-QuipuEdgeKeys.ps1`). Link
records carry no text, so they cost none of the source's token budget.

## The record

One `deepmap.link/1` record per run, replacing that phone's previous one:

```json
{"agent_id": "phone-<hash>", "schema": "deepmap.link/1",
 "node":  {"role": "phone", "label": "my phone", "network_id": "net-<hash>"},
 "wifi":  {"ap": "ap-<hash>", "rssi_dbm": -54, "freq_mhz": 5180, "link_mbps": 433, "state": "COMPLETED"},
 "edges": [{"to": "ap-<hash>", "kind": "wifi", "ok": true, "rssi_dbm": -54},
           {"to": "peer-kitchen", "kind": "tcp", "port": 8765, "ok": true, "latency_ms": 4.2}],
 "observed_at": "2026-11-22T20:26:00+00:00"}
```

`GET /deepmap` returns `nodes`, `edges` and `networks`. A phone that has not
reported within `max_age_s` (default one hour, `?max_age_s=`) is marked `stale`
rather than shown as connected, and does not count toward its network.

Hashes use a random salt created on first run in `~/.config/quipu/deepmap_salt`
(mode 600) and never sent anywhere, so the same access point has the same id on
this phone without the id revealing, or being reversible to, the address.
Delete the file to rotate every id.
