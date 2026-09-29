#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p analysis_outputs runs

setsid python3 scripts/run_bifurcation_refine.py \
  --lo 0.15 \
  --hi 0.18 \
  --tol 0.002 \
  --final-target 200000 \
  --stages 120000,160000,200000 \
  --gpus "${GPUS:-0,1}" \
  --max-parallel 2 \
  > analysis_outputs/bifurcation_scheduler.nohup.log 2>&1 < /dev/null &

echo "$!" > analysis_outputs/bifurcation_scheduler.pid
echo "started bifurcation refinement scheduler pid=$(cat analysis_outputs/bifurcation_scheduler.pid)"
