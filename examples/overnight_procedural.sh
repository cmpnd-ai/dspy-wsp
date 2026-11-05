#!/bin/zsh
# Run the Procedural Typed Decisions suite for each model in turn: two baselines, a WeightSpace run,
# and the ReAnchor comparison. A watchdog stops any step whose process grows above 20 GB.
#   examples/overnight_procedural.sh d1 nanojev
# Run it from the repository root. Logs go to examples/runs/procedural-decisions/.
set -u
root=${0:A:h:h}
cd "$root/examples"
[[ -f ../.env ]] && export $(grep '^OPENAI_API_KEY=' ../.env | xargs)
logs=runs/procedural-decisions
mkdir -p $logs

watch_memory() {
  local pid=$1 name=$2
  while kill -0 $pid 2>/dev/null; do
    local gb=$(footprint $pid 2>/dev/null | awk '/phys_footprint:/ {if ($3=="GB") print int($2); else print 0; exit}')
    if [[ ${gb:-0} -ge 20 ]]; then
      echo "=== $(date) WATCHDOG stopped $name at ${gb} GB" >> $logs/overnight.log
      kill $pid
    fi
    sleep 15
  done
}

for model in "$@"; do
  for step in baseline-protocol baseline-program optimize compare; do
    echo "=== $(date) start $model $step" >> $logs/overnight.log
    caffeinate -i ../.venv/bin/python procedural_decisions.py --model $model --step $step > $logs/${model}_${step}.log 2>&1 &
    pid=$!
    watch_memory $pid "$model $step" &
    wait $pid
    code=$?
    echo "=== $(date) end $model $step exit $code" >> $logs/overnight.log
    if [[ $code -ne 0 ]]; then
      echo "=== $(date) stopping: $model $step failed" >> $logs/overnight.log
      exit $code
    fi
  done
done
echo "=== $(date) all done" >> $logs/overnight.log
