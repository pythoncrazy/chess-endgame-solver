#!/usr/bin/env bash
# Start (or restart) the tablebase visualizer in the background.
# Usage: tools/visualizer/run.sh [PORT] [HOST]   (defaults: 8642, 127.0.0.1; KQBNvKR tree)
# Example for the tailnet: tools/visualizer/run.sh 8000 "$(tailscale ip -4)"
set -euo pipefail
TB_ROOT=${TB_ROOT:-/home/supersketchy/git/chess-solver/tb}
PORT=${1:-8642}
HOST=${2:-127.0.0.1}
PIDFILE=${TMPDIR:-/tmp}/tablebase-visualizer-$PORT.pid
HERE=$(cd "$(dirname "$0")" && pwd)

if [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")"
  sleep 0.5
fi
setsid "$TB_ROOT/venv/bin/python" -I "$HERE/server.py" \
  --tables "$TB_ROOT/results/solve-KQBNvKR" \
  --syzygy "$TB_ROOT/references-six" \
  --logs "$TB_ROOT/logs" \
  --host "$HOST" --port "$PORT" > "$TB_ROOT/logs/visualizer.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
for _ in $(seq 100); do
  curl -s -o /dev/null "http://$HOST:$PORT/" && break
  sleep 0.2
done
cat "$TB_ROOT/logs/visualizer.log"
