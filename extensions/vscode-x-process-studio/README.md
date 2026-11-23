# x_ProcessStudio (QUIPU) — MCP for VS Code agents

Registers QUIPU's **x_ProcessStudio holon** as an MCP server, so agent mode in
VS Code (and any editor that installs from Open VSX and implements the VS Code
MCP API) can work with the operator's Activepieces automations, under
governance.

## What an agent can do

| tool | effect |
|---|---|
| `process_studio_status` | ledger counts, whether Activepieces is connected |
| `process_studio_list` | proposals by status / source |
| `activepieces_tools` | the Activepieces tools a proposal may name (read-only) |
| `process_studio_propose` | **stage** an action proposal: held until the operator authorises it |

An agent is **v-ADMIN**: it can stage, never authorise or run. Nothing executes
until the operator authorises the proposal with an organizational grant
reference, after which QUIPU's r-ADMIN realises it through Activepieces.

## Setup

1. Run the fleet's `quipu-mcp` service (QUIPU `docker compose up -d`), with
   `QUIPU_MCP_TOKEN` set.
2. Install this extension (VS Code Marketplace or Open VSX), VS Code ≥ 1.101.
3. The server appears in the MCP server list as **x_ProcessStudio (QUIPU)**.
   On first start you are asked for the token; it is kept in VS Code secret
   storage. `x_ProcessStudio: Set Server Token` / `Clear Server Token` manage it.

## Settings

- `xProcessStudio.serverUrl` — default `http://127.0.0.1:7110/mcp`. Plain
  `http://` is accepted only for this machine or a private network; use
  `https://` (e.g. the fleet's Cloudflare tunnel) for anything else.
- `xProcessStudio.enabled` — offer the server to agents (default on).

See `docs/X_PROCESS_STUDIO.md` in the QUIPU repository for the authority model.
