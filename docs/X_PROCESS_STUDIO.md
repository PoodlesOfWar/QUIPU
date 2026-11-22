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

翈 — the studio stages what is wanted; only the operator's grant lets it act.
