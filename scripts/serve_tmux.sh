#!/usr/bin/env bash
# Launch the supervisor inside a tmux session so it survives this shell.
#
#   ./scripts/serve_tmux.sh          # start (idempotent)
#   tmux attach -t predictor         # watch the logs
#   ./scripts/serve_tmux.sh stop     # stop it
set -eu

SESSION="${SESSION:-predictor}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ "${1:-start}" = "stop" ]; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  echo "stopped $SESSION"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session $SESSION already running; attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$DIR" \
  "PORT=${PORT:-12000} HOST=${HOST:-0.0.0.0} ./scripts/run_server.sh; exec bash"
echo "started tmux session $SESSION (attach: tmux attach -t $SESSION)"
