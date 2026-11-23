# x_ProcessStudio — authorizable action through Activepieces MCP (r-ADMIN via v-ADMIN)

`src/quipu/x_process_studio.py` is the holon where the fleet's VLM/LLM wanting
meets the operator's process automation. Activepieces exposes the operator's
connected pieces as MCP tools; the holon is the one governed door to them.

```
Perceptopoly ─┐  (quipu_client, signed)
StockOpoly  ──┼─ POST /process_studio/propose ──► ActionProposal, held (翈)   v-ADMIN
BurgerPhone ──┘                                        │
                                   operator authorises │  grant_ref  (V10-SEC-010)
                                                       ▼
                               POST /process_studio/realise / CLI realise      r-ADMIN
                                                       │
                                        Activepieces MCP  tools/call
```

## The authority model

| side | who | may | may not |
|---|---|---|---|
| **v-ADMIN** (virtual) | any admitted fleet source — its VLM/LLM layer included | stage an `ActionProposal`, list, inspect | execute anything |
| operator | a human with the grant | authorise (Gate 6 two-party attestation + `authorised_to_realise` with a `grant_ref`) or reject | — |
| **r-ADMIN** (realised) | the holon, only on an AUTHORISED proposal | dispatch `tools/call` to Activepieces MCP | act on admissibility alone |

A passed gate is **technical admissibility**, never authorization: realisation
additionally needs the organizational `grant_ref` the code cannot mint
(`qpsi.governance.authorised_to_realise`, V10-SEC-010). Every decision records
the gate reason, the grant, and `config_digest` (V10-SEC-002).

Proposal statuses: `PROPOSED` (staged, held in band) → `AUTHORISED` →
`REALISED`, or `FAILED` (the tool itself refused — permanent), or `REJECTED`.
A transport failure keeps the proposal `AUTHORISED` and retryable; nothing is
silently dropped.

## Wire contract (observer, `:7100`)

| route | body | answer |
|---|---|---|
| `GET /process_studio` | — | stats: MCP configured, counts by status, recent, operator sources |
| `POST /process_studio/propose` | `{source, tool, arguments?, reason?, meta?}` | `200 {ok, held: true, proposal}` |
| `POST /process_studio/authorise` | `{source, proposal_id, operator?, grant_ref, approve?, why?}` | `authorised` / `rejected` / `held` |
| `POST /process_studio/realise` | `{source, proposal_id}` or `{source, ready: true}` | `realised` / `retry` / `held` / `failed` |

All three POSTs are in `edge_admission.MUTATING`: signed identity, idempotency
and the operator's token budget apply exactly as for `/observe`. `meta.vlm`
carries the perception evidence (`propose_from_vlm`) so the operator authorises
with the signal in view; `meta.schema` is `process_studio.propose/1`.

**Operator routes are off by default.** `/authorise` and `/realise` answer 403
unless the write is *signed* and its source is named in
`QUIPU_PROCESS_STUDIO_OPERATORS`. The default operator path is the CLI inside
the container, like every operator command:

```bash
docker exec quipu python -m src.quipu.x_process_studio status
docker exec quipu python -m src.quipu.x_process_studio list --status PROPOSED
docker exec quipu python -m src.quipu.x_process_studio tools
docker exec quipu python -m src.quipu.x_process_studio authorise act_… \
    --operator operator_admin --grant-ref IT505-CR-0042
docker exec quipu python -m src.quipu.x_process_studio realise --ready
```

## Activepieces MCP

Turn the MCP server on in Activepieces (Settings → MCP Server), copy the
server URL and set it on the `quipu` container:

```
QUIPU_ACTIVEPIECES_MCP_URL=https://cloud.activepieces.com/api/v1/mcp/<id>/mcp
QUIPU_ACTIVEPIECES_MCP_TOKEN=…        # optional bearer, self-hosted fronts
QUIPU_MCP_TIMEOUT_S=20
```

The client speaks MCP JSON-RPC 2.0 over **Streamable HTTP** (`initialize` →
`notifications/initialized` → `tools/list` / `tools/call`), accepts plain-JSON
and SSE-framed bodies, and carries the server's `Mcp-Session-Id`. Plain
`http://` is accepted only for loopback, private and container addresses (as in
`edge_client`). Unset URL ⇒ `available() == False`: proposing still works,
realisation answers `held` — the studio is opt-in on the acting side and the
fleet never blocks on it.

## The fleet connection (Perceptopoly ↔ StockOpoly ↔ BurgerPhone)

Each sibling connects the way every writer connects: a minted key and a grant
line, then signed writes through its vendored `quipu_edge_client.py` (or any
client that sends `sign_headers`). `burgerphone` is now a first-class source
profile (body axis, siblings perceptopoly + stockopoly), beside the existing
`perceptopoly` and `stockopoly` profiles:

```powershell
.\ops\New-QuipuEdgeKeys.ps1 -WriteEnv     # mint burgerphone / stockopoly keys
# add the sources and ceilings to edge_grant.json, then docker compose up -d
```

- **Perceptopoly** already signs (`quipu_client.py`): point its agentic layer
  at `submit("/process_studio/propose", {...})`.
- **StockOpoly** ships `engine/stockopoly/process_studio.py` — its grouping
  cascade (tier-3 VLM) stages regroup/slotting actions.
- **BurgerPhone** vendors the same single-file client with
  `QUIPU_EDGE_KEY_BURGERPHONE`; its assistant stages comms/order actions.

The studio writes only `process_studio:*` brain_kv keys, feeds no mesh, and is
input to no gate.

## For IDE agents: the MCP server, hosted on the fleet (v0.68.0)

`src/quipu/process_studio_mcp.py` serves the holon to agents as an **MCP
server** — Streamable HTTP at `POST /mcp` (or `stdio`). It exposes exactly the
v-ADMIN surface:

| MCP tool | does |
|---|---|
| `process_studio_status` | ledger counts, Activepieces availability |
| `process_studio_list` | proposals by status / source |
| `activepieces_tools` | the operator's Activepieces tools + input schemas (`GET /process_studio/tools`, cached 5 min; listing never runs a tool) |
| `process_studio_propose` | stage a proposal, held for the operator |

There is **no** authorise or realise tool: an editor agent can want, never act.

**Hosting (Hub Holon docker fleet).** `docker-compose.yml` runs it as the
`quipu-mcp` service: same image as `quipu`, its own process, published on
`127.0.0.1:7110`. It never opens the brain (one brain, one writer); it reads
the observer and signs its single write as the **`vscode-agent`** source over
the internal `edge-mcp` network, so edge identity, idempotency (identical
proposals de-duplicate) and the operator's budget apply to agents like to every
writer.

```bash
export QUIPU_MCP_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
docker compose up -d quipu quipu-mcp
curl -s http://127.0.0.1:7110/health
# remote agents: a named Cloudflare tunnel in front (TLS), token still required
QUIPU_MCP_TUNNEL_TOKEN=… docker compose --profile tunnel-mcp up -d
```

Security: every request needs `Authorization: Bearer $QUIPU_MCP_TOKEN`
(constant-time compare). **Unset ⇒ every request is 401** — the server fails
closed; `QUIPU_MCP_ALLOW_OPEN=1` permits tokenless use on loopback only. A
browser `Origin` not listed in `QUIPU_MCP_ORIGINS` is refused (DNS-rebinding
defence); editors send none. Mint the `vscode-agent` edge key and add it to the
grant to run under `QUIPU_EDGE_AUTH=enforce`.

**In a fleet-root compose** (where QUIPU sits on per-writer edge networks), add
the same service on its own edge:

```yaml
  quipu-mcp:
    image: quipu-observer:dev
    command: ["python", "-m", "src.quipu.process_studio_mcp", "serve", "--host", "0.0.0.0", "--port", "7110"]
    ports: ["127.0.0.1:7110:7110"]
    environment:
      QUIPU_URL: http://quipu:7100
      QUIPU_MCP_TOKEN: ${QUIPU_MCP_TOKEN}
      QUIPU_EDGE_KEY_VSCODE_AGENT: ${QUIPU_EDGE_KEY_VSCODE_AGENT}
    networks: [edge-mcp]        # and add edge-mcp to the quipu service's networks
```

## VS Code and Open VSX

`extensions/vscode-x-process-studio/` is a VS Code extension (engine
`^1.101.0`, the stable MCP provider API) that contributes
`mcpServerDefinitionProviders` and registers the server via
`vscode.lm.registerMcpServerDefinitionProvider` as an `McpHttpServerDefinition`.
Agent mode then lists **x_ProcessStudio (QUIPU)** among its MCP servers.

- `xProcessStudio.serverUrl` (default `http://127.0.0.1:7110/mcp`): plain http
  only for this machine or a private network; `https://` otherwise.
- The token is asked for once, when the server first starts (never while
  merely listing servers), and kept in VS Code **secret storage**;
  `x_ProcessStudio: Set Server Token` / `Clear Server Token` manage it.
- Open VSX serves the same `.vsix` to VSCodium, Cursor, Windsurf, Gitpod and
  other VS Code-compatible editors; agent use there depends on the editor
  implementing the VS Code MCP API.

**Publishing** (`.github/workflows/publish-extension.yml`): every PR touching
the extension compiles and packages it; tagging `ext-v<version>` publishes to
the **VS Code Marketplace** when the `VSCE_PAT` secret is set and to **Open
VSX** when `OVSX_PAT` is set, and attaches the `.vsix` to the GitHub release.
One-time owner steps: create publisher `poodlesofwar` on the Marketplace and
namespace `poodlesofwar` on open-vsx.org (`npx ovsx create-namespace
poodlesofwar -p $OVSX_PAT`), add both PATs as repository secrets, choose a
license (the manifest says `UNLICENSED` until one is added), then
`git tag ext-v0.1.0 && git push origin ext-v0.1.0`.

翈 — the studio stages what is wanted; only the operator's grant lets it act.
