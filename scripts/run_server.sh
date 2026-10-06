#!/usr/bin/env bash
# Keep the predictor web app running in this sandbox.
#
# The runtime has no init/supervisor (PID 1 is the agent) and recycles its VM,
# which kills every process. This script restarts the app whenever it dies and
# survives the shell that launched it. Run it inside tmux for the sandbox, or
# as a systemd unit / container entrypoint on a real host.
#
#   ./scripts/run_server.sh              # foreground supervisor
#   PORT=12000 ./scripts/run_server.sh   # pick the port
set -u

cd "$(dirname "$0")/.."

PORT="${PORT:-12000}"
HOST="${HOST:-0.0.0.0}"
PYTHON="${PYTHON:-python}"
LOG="${LOG:-/tmp/webapp.log}"
MIN_BACKOFF=5
MAX_BACKOFF=60

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

install_deps() {
  # The sandbox wipes site-packages on recycle; recover instead of crash-looping.
  if ! "$PYTHON" -c "import flask, pandas, sklearn, websockets" >/dev/null 2>&1; then
    log "dependencies missing; installing from requirements.txt"
    "$PYTHON" -m pip install -q -r requirements.txt >>"$LOG" 2>&1 || log "pip install failed"
  fi
}

backoff=$MIN_BACKOFF
while true; do
  install_deps
  log "starting web app on ${HOST}:${PORT}"
  PORT="$PORT" HOST="$HOST" "$PYTHON" -m predictor.webapp >>"$LOG" 2>&1
  code=$?
  log "web app exited (code ${code}); restarting in ${backoff}s"
  sleep "$backoff"
  backoff=$(( backoff * 2 ))
  [ "$backoff" -gt "$MAX_BACKOFF" ] && backoff=$MAX_BACKOFF
done
