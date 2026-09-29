#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p analysis_outputs continuation_runs

setsid python3 scripts/run_bifurcation_continuation.py \
  --lo 0.15 \
  --hi 0.18 \
  --tol 0.002 \
  --seed-case CHM_015 \
  --seed-tu 200000 \
  --extension-stages 40000,80000 \
  --gpus "${GPUS:-0,1}" \
  --max-parallel 2 \
  > analysis_outputs/continuation_scheduler.nohup.log 2>&1 < /dev/null &

echo "$!" > analysis_outputs/continuation_scheduler.pid
echo "started continuation bifurcation scheduler pid=$(cat analysis_outputs/continuation_scheduler.pid)"
