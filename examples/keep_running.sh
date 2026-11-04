#!/bin/zsh
# Rerun a resumable example until it exits cleanly, keeping the Mac awake.
#   examples/keep_running.sh examples/math500_qwen_mlx.py
# Run it from the repository root. Logs go to <script>.log next to the run.
set -u
script=${1:?usage: keep_running.sh <example.py> [max_restarts]}
max_restarts=${2:-20}
root=${0:A:h:h}
cd "$root/examples"
[[ -f ../.env ]] && { set -a; . ../.env; set +a; }
log="runs/${script:t:r}.log"
mkdir -p runs
for attempt in $(seq 0 $max_restarts); do
  echo "=== $(date) start attempt $attempt" >> "$log"
  caffeinate -i ../.venv/bin/python "${script:t}" >> "$log" 2>&1
  code=$?
  echo "=== $(date) exit $code" >> "$log"
  [[ $code -eq 0 ]] && exit 0
  sleep 30
done
echo "=== $(date) gave up after $max_restarts restarts" >> "$log"
exit 1
