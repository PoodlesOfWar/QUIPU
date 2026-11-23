#!/usr/bin/env sh
# edge_setup.sh -- point this edge device (Termux) at QUIPU's hosted x_ProcessStudio MCP.
#
# It is the SAME MCP server the VS Code / Open VSX extension registers (the `quipu-mcp`
# service on the Black Bulls fleet, at its /mcp address, with the same bearer token).
# This script stores that address and token for quipu_mcp.py and checks the connection.
# It changes nothing on the server and adds no capability: the edge sees the v-ADMIN tools
# only (no authorise/realise), exactly as the extension does.
#
#   sh clients/termux/edge_setup.sh --url https://<front-door>/mcp [--token-file FILE] [--no-prompt] [--force]
#
# The token is NEVER taken as a command-line argument (it would sit in the process list and in
# any remote-shell transcript, Desktop Commander's included). It comes from, in order:
#   --token-file FILE, then $QUIPU_MCP_TOKEN, then a hidden prompt (unless --no-prompt).
set -eu

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONF="${HOME}/.config/quipu"
ENVF="${CONF}/mcp.env"
TOKF="${CONF}/mcp.token"

url="${QUIPU_MCP_URL:-}"
tokfile=""
prompt=1
force=0

die() { printf '%s\n' "edge_setup: $*" >&2; exit 3; }

while [ $# -gt 0 ]; do
  case "$1" in
    --url) [ $# -ge 2 ] || die "--url needs a value"; url="$2"; shift 2 ;;
    --token-file) [ $# -ge 2 ] || die "--token-file needs a value"; tokfile="$2"; shift 2 ;;
    --no-prompt) prompt=0; shift ;;
    --force) force=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    --token|--token=*) die "the token is never taken on the command line: use --token-file, \$QUIPU_MCP_TOKEN, or the prompt" ;;
    *) die "unknown option: $1" ;;
  esac
done

PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
[ -n "$PY" ] || die "python not found. Run: pkg install python"
[ -n "$url" ] || die "no URL. Pass --url https://<front-door>/mcp (the extension's xProcessStudio.serverUrl)"
case "$url" in
  https://*) ;;
  http://127.0.0.1*|http://localhost*|http://10.*|http://192.168.*) ;;
  *) die "the URL must be https:// (plain http only for this device or a private network)" ;;
esac

mkdir -p "$CONF"
chmod 700 "$CONF" 2>/dev/null || true

if [ -e "$ENVF" ] && [ "$force" -ne 1 ]; then
  die "$ENVF already exists; pass --force to replace it"
fi

# Obtain the token without ever echoing it.
token=""
if [ -n "$tokfile" ]; then
  [ -r "$tokfile" ] || die "cannot read --token-file $tokfile"
  token="$(tr -d '\r\n' < "$tokfile")"
elif [ -n "${QUIPU_MCP_TOKEN:-}" ]; then
  token="$QUIPU_MCP_TOKEN"
elif [ "$prompt" -eq 1 ] && [ -t 0 ]; then
  printf 'QUIPU MCP token (hidden): ' >&2
  stty -echo 2>/dev/null || true
  IFS= read -r token || true
  stty echo 2>/dev/null || true
  printf '\n' >&2
fi
[ -n "$token" ] || die "no token. Use --token-file FILE, set QUIPU_MCP_TOKEN, or run interactively (not with --no-prompt)"

umask 077
printf '%s\n' "$token" > "$TOKF"
{
  printf '# written by edge_setup.sh; the token is in mcp.token (mode 600), not here\n'
  printf 'QUIPU_MCP_URL=%s\n' "$url"
  printf 'QUIPU_MCP_TOKEN_FILE=%s\n' "$TOKF"
} > "$ENVF"
chmod 600 "$ENVF" "$TOKF"

printf '%s\n' "edge_setup: saved $ENVF and $TOKF (mode 600)"
printf '%s\n' "edge_setup: checking the connection..."
if "$PY" "$ROOT/clients/termux/quipu_mcp.py" doctor; then
  printf '\n%s\n' "edge_setup: connected. Try:  python clients/termux/quipu_mcp.py status"
  printf '%s\n' "In VS Code / an Open VSX editor, set xProcessStudio.serverUrl to the same URL and run"
  printf '%s\n' "'x_ProcessStudio: Set Server Token' (the extension stores the token in secret storage)."
else
  printf '\n%s\n' "edge_setup: saved, but the check failed. Fix what it says and re-run: python clients/termux/quipu_mcp.py doctor" >&2
  exit 1
fi
