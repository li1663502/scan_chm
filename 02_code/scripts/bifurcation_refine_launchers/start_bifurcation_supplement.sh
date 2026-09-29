#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p analysis_outputs supplement_runs continuation_runs

setsid python3 scripts/run_bifurcation_supplement.py \
  --gpus "${GPUS:-0,1}" \
  --max-parallel 2 \
  > analysis_outputs/supplement_scheduler.nohup.log 2>&1 < /dev/null &

echo "$!" > analysis_outputs/supplement_scheduler.pid
echo "started bifurcation supplement scheduler pid=$(cat analysis_outputs/supplement_scheduler.pid)"
