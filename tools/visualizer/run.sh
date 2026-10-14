#!/usr/bin/env bash
# Start (or restart) the tablebase visualizer in the background.
# Usage: tools/visualizer/run.sh [PORT] [HOST...]   (defaults: 8642, 127.0.0.1)
# Tailnet (IPv4 and IPv6, so MagicDNS names work):
#   tools/visualizer/run.sh 8000 $(tailscale ip -4) $(tailscale ip -6)
set -euo pipefail
TB_ROOT=${TB_ROOT:-/home/supersketchy/git/chess-solver/tb}
PORT=${1:-8642}
shift || true
HOSTS=("${@:-127.0.0.1}")
HOST_ARGS=()
for h in "${HOSTS[@]}"; do HOST_ARGS+=(--host "$h"); done
HOST=${HOSTS[0]}
PIDFILE=${TMPDIR:-/tmp}/tablebase-visualizer-$PORT.pid
HERE=$(cd "$(dirname "$0")" && pwd)

if [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")"
  sleep 0.5
fi
setsid "$TB_ROOT/venv/bin/python" -I "$HERE/server.py" \
  --tables "$TB_ROOT/results/all5" \
  --tables "$TB_ROOT/results/solve-KQBNvKR" \
  --syzygy "$TB_ROOT/references-345" \
  --syzygy "$TB_ROOT/references-six" \
  --logs "$TB_ROOT/logs" \
  "${HOST_ARGS[@]}" --port "$PORT" > "$TB_ROOT/logs/visualizer.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
for _ in $(seq 100); do
  curl -s -o /dev/null "http://$HOST:$PORT/" && break
  sleep 0.2
done
cat "$TB_ROOT/logs/visualizer.log"
