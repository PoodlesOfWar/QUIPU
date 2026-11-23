# QUIPU on an edge device — the same MCP the extension uses

`clients/termux/quipu_mcp.py` connects an edge device (a phone under Termux, or
any small Linux box) to the **same x_ProcessStudio MCP server** that the VS Code /
Open VSX extension registers (`process_studio_mcp.py`, hosted as `quipu-mcp` on
the Hub Holon docker fleet). It is not a second implementation of the policy:
the server decides what a client may do, and the edge client only speaks to it.

| what the edge can do | what it cannot do |
|---|---|
| read status and the ledger | authorise or realise anything (the server has no such tool) |
| stage a proposal — **held** as `PROPOSED`, source `vscode-agent`, labelled with `meta.edge_device` | decide for a human, or widen its own authority |
| list the closed `hub.*` action catalogue | add an action; unknown `hub.*` tools/arguments are refused at propose |
| hear the Perceptopoly voice, **only while a human Admin has it on** | turn the voice on |

`doctor` checks this parity from the device: it fails if the server lists
anything other than the five base tools (plus the optional voice), and fails
loudly if any tool name suggests authorising, realising, approving or promoting.

## The URL you need

The hosted location is the fleet's `quipu-mcp` service. It listens on
`127.0.0.1:7110` on the fleet host, so an edge device needs a front door:

```sh
# on the fleet host (vs-code repo):
docker compose --profile mcp --profile tunnel-mcp up -d
docker compose logs quipu-mcp-tunnel | grep trycloudflare      # quick tunnel: a temporary https URL
```

or a named Cloudflare tunnel on a hostname you own. Your edge URL is
`https://<that host>/mcp`, and it is the same value as the extension's
`xProcessStudio.serverUrl`. The client refuses plain `http://` to anything that
is not a private/loopback/`.local` host. The bearer token is `QUIPU_MCP_TOKEN`
from the fleet's environment.

## Setup — type the token in Termux, never in chat

```sh
cd QUIPU
sh clients/termux/edge_setup.sh --url https://<your-host>/mcp      # prompts for the token, hidden
python clients/termux/quipu_mcp.py doctor
```

`edge_setup.sh` writes `~/.config/quipu/mcp.env` (URL) and
`~/.config/quipu/mcp.token` (token) with mode 0600, then runs `doctor`. The token
is refused on the command line (it would land in shell history and `ps`); it may
come from `--token-file`, `$QUIPU_MCP_TOKEN` or the prompt. An existing config is
not overwritten without `--force`. If the check fails the files are kept and the
script says so.

## Use

```sh
python clients/termux/quipu_mcp.py status
python clients/termux/quipu_mcp.py list --status PROPOSED
python clients/termux/quipu_mcp.py propose hub.body_directive --reason "line 3 is quiet" \
    --args '{"title":"Check line 3","do_this":"Walk it","owner_role":"Ops"}' --device my-phone
python clients/termux/quipu_mcp.py voice          # refused until a human Admin enables it
```

Exit codes: 0 ok, 1 server/check failure, 3 configuration problem.

## Via Desktop Commander Remote

If you drive Termux through Desktop Commander Remote, it is the relay that runs
the commands above in your Termux session; it adds no capability to QUIPU. Two
cautions:

- Desktop Commander Remote is a **third-party relay**. Whatever it runs and prints
  passes through it. So run `edge_setup.sh` with the token typed at Termux's own
  prompt, and do not `cat` the token file or print the environment through it.
  `quipu_mcp.py` never prints the token.
- Its device ID and verification code are credentials for your device. Keep them
  out of commits and chats.

The Desktop Commander tools were **not available in the session that built this**,
so nothing here was run through it. What was verified: 15 tests that run the
client against a real MCP server and observer and run `edge_setup.sh` as a real
subprocess (permissions, no token in any output, refusal paths, wrong token).
Not verified: a real phone, a real tunnel, or Desktop Commander itself.
