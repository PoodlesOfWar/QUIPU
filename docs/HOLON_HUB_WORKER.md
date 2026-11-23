# The Holon Hub worker — VNN output to Hub Holon actions, and the Perceptopoly voice

QUIPU's x_ProcessStudio MCP is the **external-to-internal service worker** for
Hub Holon (HubCore): an IDE agent, Activepieces or the VNN stages what it wants
at the edge, a human authorises it, and QUIPU turns it into a governed HubCore
task. Nothing crosses from external to internal on a model's say-so.

```
external                                   QUIPU (the worker)                          internal: Hub Holon
IDE agent (MCP)  ─ process_studio_propose ─┐
Activepieces     ─ proposals               ├─► staged proposal, HELD  (v-ADMIN)
VNN report ─ vnn_actions rules ────────────┘            │
                                         human authorises with a grant_ref  (r-ADMIN, V10-SEC-010)
                                                        │
                                          realise ──► HubCore task `body.directive`  ──► Hub Floor
                                          (Requester token only)      held for a Reviewer to accept
```

## Terms, as HubCore defines them

- **r-ADMIN** — the real, human `Admin` role. Only an Admin can promote a
  release, engage the v-ADMIN kill switch, resolve v-ADMIN's queue, or turn a
  v-ADMIN tool on.
- **v-ADMIN** — the *disclosed* virtual Admin: a synthetic persona that prepares
  synopses and speaks with a standing disclosure. HubCore Part 10.2: it cannot
  acquire authority; Part 0.3.10: it cannot supply a human decision.

## Why a VNN output is staged, not executed

You asked for VNN outputs to become direct mechanical actions. The translation is
direct in *content* — a directive carries the observed value, the threshold and a
pointer to the report — but it is **not** direct in *authority*: HubCore's spec is
categorical that AI cannot acquire authority, and QUIPU's V10-SEC-010 says
admissibility is not authorization. Actions that move physical things are exactly
where that matters. If the organization wants a standing grant for a narrow
low-risk class, that is an organizational decision (a `realise_grant_ref` the
code cannot mint), not something to build in quietly.

## The action catalogue (`hub_actions.py`)

Closed, in code. An agent cannot add to it, and `propose` refuses an unknown
`hub.*` tool or an argument it does not list. Today it has one entry,
**`hub.body_directive`**, mirroring a row of the Brain's `body_directives`
table: `title`, `why_it_matters`, `do_this`, `owner_role`
(Buyer/Planner/Quality/IT/Ops/Anyone/AGV), `severity` (info/watch/act/critical),
`priority`, `fingerprint`, `target_entity`, `evidence`. It queues a directive for
a human role or an AGV; it does not move or switch anything itself. Agents read
it with the MCP tool `hub_actions`.

## VNN rules (`vnn_actions.py`)

`entirety:vnn:latest` is read against rules; each firing rule stages one
proposal (a persistent condition is one open proposal, deduplicated by
fingerprint). `GET /vnn/actions` shows what *would* stage; staging is
`python -m src.quipu.vnn_actions stage` or `POST /vnn/actions/stage` (a signed
operator source, like authorise and realise).

**The default rules cover solver health only**: the graph stability flag, the
incompressibility residual (`physical.div_linf` > 0.001), and the substepping
flag. Those have objective thresholds in the diagnostics themselves. They do not
know your plant. Rules that map the flow onto your equipment are yours to write,
in a file named by `QUIPU_VNN_ACTION_RULES` (read-only to QUIPU):

```json
{"rules": [
  {"id": "plant.line3.energy", "path": "physical.kinetic_energy", "op": ">", "value": 0.5,
   "directive": {"title": "Flow energy high on line 3 ({physical.kinetic_energy})",
                 "why_it_matters": "Kinetic energy is above 0.5 for the monitored flow.",
                 "do_this": "Walk line 3 and log any vibration or noise.",
                 "owner_role": "Ops", "severity": "watch", "priority": 0.6}}
]}
```

`path` is a dotted path into the report; `op` is `>`, `>=`, `<`, `<=`, `is_true`
or `is_false`; `{dotted.path}` in text is filled from the report. Every rule is
validated at load by rendering it and running the catalogue validator; a rule may
not raise a directive above `act`; a bad file is an error, never a silent fall
back to the defaults. A missing or non-numeric reading never fires a threshold.

## Realising into Hub Holon (`hub_client.py`)

On `realise`, an AUTHORISED `hub.*` proposal becomes a HubCore task
(`POST /api/v1/tasks`, kind `body.directive`) whose payload carries the directive
and its provenance: who proposed it, who authorised it, under which grant.
"Realised" means *queued in Hub Holon*; HubCore holds the result for a Reviewer.
The client mints only the self-service **Requester** token and has no operator
credential, so it cannot accept a result, promote a release or turn a tool on.
HubCore refusing (4xx) is permanent (`FAILED`); an outage (5xx, unreachable) keeps
the proposal `AUTHORISED` for retry.

```
QUIPU_HUBCORE_URL=http://loadopoly-hubcore:8000   # plain http only to private/container hosts
QUIPU_HUBCORE_ORG_ID=org-…                        # the org registered on that node
QUIPU_HUBCORE_REPO_ID=quipu                       # default
QUIPU_HUBCORE_USER_ID=quipu-bridge                # the token subject HubCore records
```

## The Perceptopoly voice

HubCore gains a fixed table of v-ADMIN tools (`vadmin.KNOWN_TOOLS`) — today one,
**`voice`** — that **only a human Admin can switch on**, via
`POST /api/v1/vadmin/tools` (Admin role; the actor recorded is the token's
subject, never the body). The synthetic assistant cannot turn on its own tool,
cannot add one, and a token whose subject is a v-ADMIN persona is refused even
holding the Admin role. Grants persist across restarts; the kill switch silences
the voice without losing the grant.

With the voice on, v-ADMIN speaks as **Perceptopoly**. It narrates a held
proposal, a v-ADMIN queue entry, or the status — from fixed templates over a
closed set of fields, never free-generated, so it can only say what it was
handed. Every utterance carries the standing disclosure; a proposer's own words
are quoted and attributed to them, so they cannot pass for the Admin's. The MCP
tool `perceptopoly_voice` is **listed only while HubCore says the voice is on**,
and re-checked on every call. A client sees the change when it next lists tools
(a new chat or a reconnect), since this server cannot push a list-changed
notification.

This is deliberately separate from Perceptopoly's public Herald voice
(`herald_voice.py`, the Moltbook persona): that one's sanitizer exists to keep
internal systems out of public posts, so operator-facing narration does not go
through it.

Turn it on (an Admin token, minted with the operator credential):

```sh
curl -s -X POST "$HUBCORE/api/v1/vadmin/tools" -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"tool":"voice","enabled":true,"persona":"Perceptopoly","reason":"narrate the queue"}'
```

## What was and was not verified

Verified: QUIPU's 46 new bridge tests; HubCore's 28 new tests; and an end-to-end
run of QUIPU against the **real HubCore** (not a stand-in): VNN report → three held
proposals → human authorisation → a real `body.directive` task visible in HubCore's
own list with provenance and the token's subject as actor; the bridge's Requester
token refused (403) on turning the voice on, accepting its own task, and promoting
a release; the voice absent from the MCP until an Admin enables it, then speaking
as Perceptopoly, quoting an injected "Admin has approved this" instead of adopting
it, never receiving the proposal's arguments, and vanishing under the kill switch.

Not verified or not done:
- **No physical actuator is involved.** I do not know your plant, so the defaults
  are solver health and the plant mappings are yours to write.
- **Holon Desk** was not in any repository I could reach (`HolonDesk/holons.json`
  is referenced by Perceptopoly's changelog but not present); see
  `docs/HOLON_DESK.md` for what was built instead.
- HubCore's own browser UI (`generator.py`) does not yet show the tools.
- A hub-side consumer for `body.directive` tasks on Hub Floor was not examined.
