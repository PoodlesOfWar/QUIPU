# Holon Desk, the integrations site and the SDKs

## Holon Desk: not found, so not built

You asked for Holon Desk to manage actions and interweavings. No repository I
could reach contains it (Perceptopoly's changelog mentions `HolonDesk/holons.json`;
the file is not there). I did not invent a Desk. What exists instead, and what a
Desk would plug into:

- the **x_ProcessStudio MCP** (`quipu-mcp`, :7110): the IDE/agent surface, v-ADMIN tools only;
- the **SDKs** below: the same surface for your own tooling;
- the human decision (authorise with a `grant_ref`) stays in QUIPU's operator
  endpoints, signed, and in HubCore for turning the voice on. A Desk that wants
  to show a queue can read `process_studio_list`; deciding is not an SDK call.

If you point me at the Desk's repo or its `holons.json` format, registering
`quipu-mcp` (:7110) and `integrations` (:8090) is a small follow-up. I would
rather not guess that format.

## integrations site and SDKs (`integrations/`)

- `loadopoly_integrations.py` (stdlib) and `loadopoly_integrations.mjs` +
  `.d.mts` (no dependencies): `connect/tools/status/list/hubActions/propose/voice`.
  No authorise or realise method exists, because the server has no such tool.
  Plain `http` to a public host is refused; the token is never in `repr`/`toString`.
- `server.py`: stdlib static server on :8090, GET/HEAD only, strict headers
  (CSP `default-src 'none'`, nosniff, no framing), no directory listings,
  `/health`. `Dockerfile`: python:3.12-slim, non-root. The fleet compose
  (`vs-code`, `--profile integrations`) builds it, read-only, bound to 127.0.0.1.
- `site/sdk/` holds the published copies; a test keeps them identical to the sources.

**`integrations.loadopoly.com`** needs the owner: DNS, and a route in the
Cloudflare named tunnel to `http://integrations:8090`. Nothing here creates it.

Verified: 5 tests (Python and Node SDKs against a live MCP server and observer,
type declarations checked with `tsc --strict`, site headers and refusals). Not
verified: building the image (no Docker daemon in the session that built it),
a deployed host, or any Desk.
