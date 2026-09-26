# QUIPU edge — identity and admission

QUIPU's one door from the fleet is the observer (`:7100`). `qpsi.edge_admission`
stands in front of every write: `POST /observe`, `POST /feedback`, and `/anneal`
(both methods). Status: `GET /edge`, or `python -m src.quipu.qpsi.edge_admission status`.

## Relational structure (fleet compose, 2026-09-26)

QUIPU is on no shared network. Each writer has an internal network holding only
QUIPU and itself:

| writer | container | network | QUIPU source |
|---|---|---|---|
| HubCore annealing forwarder | `hubcore` | `edge-hubcore` | `hubcore` |
| Supply Chain Architect | `scb-brain` | `edge-scb` | `supply-chain-brain` |
| JobHawk | `jobhawk` | `edge-jobhawk` | `jobhawk` |
| Bakugo CardCenter | `bakugo-app` | `touch` | `bakugo` |
| Perceptopoly | `perceptopoly-game-pipeline`, `perceptopoly-video-trainer` | `hideout_edge_perceptopoly` | `perceptopoly` |
| Loadopoly-OCR (browser) | — | host loopback | `loadopoly-ocr` (origin) |

The Marketplace reaches HubCore only and never writes to QUIPU. `quipu-net` is
QUIPU's own outbound network. The same links are declared in
`hub/hub_workspace.json` under `edges`.

## Identity (`QUIPU_EDGE_AUTH`: off | record | enforce, default record)

Signed: `X-Quipu-Source`, `X-Quipu-Timestamp`, `X-Quipu-Signature` =
hex HMAC-SHA256(key, `METHOD\nPATH\nTS\nsha256(body)`), within the skew window
(`QUIPU_EDGE_SKEW_S`, 300 s), each signature once, body `source` = signer.
Origin: a browser client whose `Origin` the grant binds to that source (weaker;
recorded as `origin`). `record` labels every write and refuses none; `enforce`
refuses unverified writes (401). CORS echoes only bound origins (and
`QUIPU_CORS_ORIGINS`), never `*`.

## Admission (`QUIPU_EDGE_BUDGET`: off | record | enforce, default record)

The operator's grant (`edge_grant.json`) names the sources and a ceiling per
source in tokens per hour, plus a total. Inside it QUIPU allocates: each source's
share of the total follows the information it delivers (novel tokens per token,
trailing hour), water-filled under the ceilings. `enforce` refuses sources
outside the grant (403) and writes over allocation (429 + Retry-After).

This is the constrained gate of the 2026-09-22 Invariance #7 ruling: QUIPU
allocates the operator's grant among the operator's sources. It cannot add a
source, raise a ceiling or the total, mint or read out a key, or change a mode;
the grant and key files are only read.

## One brain, one writer (v0.48.0)

The `quipu` container runs the whole Entirety (`src/quipu/entirety_service.py`):
the observer behind the edge, the expansion step, the operator's pulse
(`QUIPU_PULSE_ROUTE`) and doc annealing, against one brain in the
`vscode_quipu_brain` volume. The host no longer opens a QUIPU brain: after
`ops/Move-QuipuBrain.ps1` the repository copy is renamed and
`local_brain.MOVED.json` makes any host process that tries to open it fail
loudly (`BrainMovedError`) instead of starting a second brain. Operator commands
run inside the container (`docker exec quipu python -m ...`); `Start-Pulse.ps1`,
`Start-Expansion.ps1` and `Start-DocAnnealing.ps1` now do exactly that.

## Every write reaches it, once (`src/quipu/edge_client.py`)

Every client writes through the same single-file, stdlib client, vendored into
each codebase as `quipu_edge_client.py`: a local SQLite outbox keeps each write
until QUIPU answers 2xx; each write carries `X-Quipu-Idempotency` (inside the
signature), and QUIPU records applied keys in the brain (`edge_idempotency`,
30 days) and answers a repeat with `duplicate: true` without learning again.
401/403 hold the write (no key or grant yet), 429 waits for Retry-After, 400 is
kept aside as rejected, everything else backs off up to an hour. Loadopoly-OCR's
browser client keeps the same outbox in localStorage.

Clients: HubCore annealing forwarder (its durable cursor plus idempotency),
Perceptopoly (`quipu_client`, and its own mesh's `feed_corpus` is teed to QUIPU),
Bakugo CardCenter, SCA (`erp_dbo`, both copies), the JobHawk/SCA integrator,
the tri-repo feedback loop (both copies), Loadopoly-OCR.

Relays: a client that forwards another source's writes signs as itself; the
grant must list the originals under its `relays_for` (assurance `relayed`).

External servers: `docker compose --profile tunnel-quipu up -d` publishes QUIPU
through a named Cloudflare tunnel (`QUIPU_TUNNEL_TOKEN`). Anything arriving
through a public front door (Cf-Connecting-IP, X-Forwarded-For, Forwarded) must
be signed, whatever `QUIPU_EDGE_AUTH` says; browser-origin assurance is not
accepted from outside. Remote clients use `edge_client` with
`QUIPU_URL=https://…`; it refuses to send over plain http to a public address.

## Operator steps

1. `.\ops\New-QuipuEdgeKeys.ps1 -WriteEnv` (fleet repo): mints keys into
   `%LOCALAPPDATA%\QUIPU\edge`, writes the grant template, sets the `.env` lines.
2. Edit the grant's ceilings.
3. `docker compose up -d` in the fleet root, then in `Perceptopoly/`.
4. Watch `GET /edge` until every writer shows `signed`; then set
   `QUIPU_EDGE_AUTH=enforce` and `QUIPU_EDGE_BUDGET=enforce`.

Clients that sign today: HubCore (`services/hubcore/annealing.py`) and
Perceptopoly (`quipu_client.py`). Supply Chain Architect, JobHawk and Bakugo
receive their key in the environment but do not sign yet; under `enforce` their
writes are refused until they do.

翈 — the door knows who is knocking; how wide it opens is the grant's.
