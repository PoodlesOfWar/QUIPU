#!/usr/bin/env sh
# autosync.sh -- run the deepmap node automatically, through Termux:API's job scheduler.
#
#   sh clients/termux/autosync.sh enable    schedule it (refuses unless setup is complete)
#   sh clients/termux/autosync.sh disable   cancel it
#   sh clients/termux/autosync.sh status    show pending Termux jobs
#
# What "automatic" means here: Android's JobScheduler runs the job about every
# DEEPMAP_PERIOD_MS (default 15 minutes, the minimum Android allows) while the phone
# is on an unmetered network (Wi-Fi), and keeps it across reboots. Android may delay a
# job for battery reasons; exempt Termux and Termux:API from battery optimisation if it
# runs late. The node itself still only acts on an SSID in DEEPMAP_SSIDS, so a job that
# fires on some other Wi-Fi does nothing.
#
# This does not need "All files access": the node reads and writes only Termux's own
# private ~/.config/quipu. Configure it in ~/.config/quipu/deepmap.env, not with shell
# exports: a scheduled job does not see your shell's environment.
set -eu

JOB_ID="${DEEPMAP_JOB_ID:-4242}"
PERIOD_MS="${DEEPMAP_PERIOD_MS:-900000}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONF="${HOME}/.config/quipu"
WRAP="${CONF}/deepmap_job.sh"
LOG="${CONF}/deepmap.log"

say() { printf '%s\n' "autosync: $*"; }
die() { printf '%s\n' "autosync: $*" >&2; exit 3; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 not found. $2"; }

scheduler_ready() {
  need termux-job-scheduler "Install the Termux:API app (same source as Termux) and run: pkg install termux-api"
  # Flags are checked, not assumed: if this Termux:API lacks one, schedule nothing rather than guess.
  help="$(termux-job-scheduler --help 2>&1 || true)"
  for flag in --job-id --script --period-ms --network --persisted --cancel; do
    printf '%s' "$help" | grep -q -- "$flag" \
      || die "this termux-job-scheduler does not list $flag; refusing to schedule. Run 'termux-job-scheduler --help' and tell whoever maintains this script."
  done
}

python_bin() { command -v python3 2>/dev/null || command -v python 2>/dev/null || die "python not found. Run: pkg install python"; }

cmd="${1:-status}"
case "$cmd" in
  status)
    need termux-job-scheduler "Install the Termux:API app and run: pkg install termux-api"
    termux-job-scheduler --pending
    ;;

  disable)
    need termux-job-scheduler "Install the Termux:API app and run: pkg install termux-api"
    termux-job-scheduler --cancel --job-id "$JOB_ID"
    say "cancelled job $JOB_ID"
    ;;

  enable)
    scheduler_ready
    PY="$(python_bin)"

    # 1. The node must be set up: Termux:API answering, network name visible.
    "$PY" "$ROOT/clients/termux/deepmap_node.py" --check >/dev/null 2>&1 \
      || die "deepmap_node --check failed. Run: python clients/termux/deepmap_node.py --check   and fix what it says first."

    # 2. A dry run must not say the allow-list is empty (the node does nothing without one).
    out="$("$PY" "$ROOT/clients/termux/deepmap_node.py" 2>&1 || true)"
    printf '%s' "$out" | grep -q "no-allowlist" \
      && die "DEEPMAP_SSIDS is empty. Set it in $CONF/deepmap.env (the network(s) this node may act on)."

    # 3. A phone needs QUIPU's https front door; the default (127.0.0.1) is the phone itself.
    grep -Eq '^QUIPU_URL=https://' "$CONF/deepmap.env" 2>/dev/null \
      || die "QUIPU_URL=https://... is not set in $CONF/deepmap.env (the operator's tunnel address)."

    mkdir -p "$CONF"
    cat > "$WRAP" <<EOF
#!/usr/bin/env sh
cd '$ROOT' || exit 1
PY="\$(command -v python3 || command -v python)"
"\$PY" clients/termux/deepmap_node.py --send >> '$LOG' 2>&1
if [ "\$(wc -c < '$LOG' 2>/dev/null || echo 0)" -gt 262144 ]; then
  tail -n 200 '$LOG' > '$LOG.tmp' && mv '$LOG.tmp' '$LOG'
fi
EOF
    chmod 700 "$WRAP"

    termux-job-scheduler --job-id "$JOB_ID" --script "$WRAP" \
      --period-ms "$PERIOD_MS" --network unmetered --persisted true
    say "scheduled job $JOB_ID every $((PERIOD_MS / 60000)) min on unmetered networks; log: $LOG"
    say "stop it with: sh clients/termux/autosync.sh disable"
    ;;

  *)
    die "usage: autosync.sh enable | disable | status"
    ;;
esac
