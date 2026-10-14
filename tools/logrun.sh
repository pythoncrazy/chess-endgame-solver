#!/usr/bin/env bash
# Usage: logrun.sh NAME CMD... ; saves stdout, stderr, exit status and command to $TB_ROOT/logs/NAME.*
set -u
TB_ROOT=${TB_ROOT:-/home/supersketchy/git/chess-solver/tb}
name=$1; shift
log=$TB_ROOT/logs/$name
printf '%q ' "$@" > "$log.cmd"; echo >> "$log.cmd"
date -u +%FT%TZ >> "$log.cmd"
start=$(date +%s.%N)
"$@" > "$log.stdout" 2> "$log.stderr"
rc=$?
end=$(date +%s.%N)
echo "$rc" > "$log.exit"
printf '%s exit=%s wall=%.1fs\n' "$name" "$rc" "$(echo "$end - $start" | bc)" | tee -a "$TB_ROOT/logs/SUMMARY.txt"
exit $rc
