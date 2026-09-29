#!/usr/bin/env bash
set -u

BASE="/media/xb/F963E8A1EEEE8BFC/liyang/li-gpu/BOTTOM FEEDBACK/phi_feedback/changeU/experiment2_pair_attractor/sweep_long_d60_fixed_120M"
BIN="$BASE/source_snapshot/experiment2_fixed"
ANALYSIS="$BASE/source_snapshot/nonlinear_analysis.py"
CSV="$BASE/smallCHM_run_status.csv"
LOG="$BASE/smallCHM_pipeline.log"
RUNSTEP=120000000

FOLDERS=(CHM_00025 CHM_0005 CHM_001 CHM_003)
CHMS=(0.0025 0.005 0.01 0.03)
GPUS=(0 1 0 1)

exec >> "$LOG" 2>&1

timestamp() {
  date '+%F %T'
}

state_file() {
  local folder="$1"
  local key="$2"
  printf '%s/%s/.%s' "$BASE" "$folder" "$key"
}

set_state() {
  local folder="$1"
  local key="$2"
  local value="$3"
  printf '%s\n' "$value" > "$(state_file "$folder" "$key")"
}

get_state() {
  local folder="$1"
  local key="$2"
  local default="$3"
  local path
  path="$(state_file "$folder" "$key")"
  if [ -f "$path" ]; then
    head -n 1 "$path"
  else
    printf '%s\n' "$default"
  fi
}

final_time() {
  local folder="$1"
  local data_dir="$BASE/$folder/0"
  find "$data_dir" -maxdepth 1 -type f -name 'ungrid*.dat' -printf '%f\n' 2>/dev/null \
    | sed -E 's/^ungrid([0-9]+)\.dat$/\1/' \
    | sort -n \
    | tail -1
}

data_complete() {
  local folder="$1"
  local data_dir="$BASE/$folder/0"
  [ -f "$data_dir/ungrid120000.dat" ] || return 1
  [ -f "$data_dir/gel0rn120000.dat" ] || return 1
  [ -f "$data_dir/gel1rn120000.dat" ] || return 1
  [ -f "$data_dir/gel0bodycenter0.dat" ] || return 1
  [ -f "$data_dir/gel1bodycenter0.dat" ] || return 1
  [ "$(wc -l < "$data_dir/gel0bodycenter0.dat")" -eq 120000 ] || return 1
  [ "$(wc -l < "$data_dir/gel1bodycenter0.dat")" -eq 120000 ] || return 1
}

figs_complete() {
  local folder="$1"
  local figs_dir="$BASE/$folder/figs"
  for name in rc_timeseries.png rc_trajectory.png speed_timeseries.png chemistry_final.png memory_final.png dynamics.mp4; do
    [ -s "$figs_dir/$name" ] || return 1
  done
}

write_status() {
  {
    printf 'CHM,folder,gpu,data_dir,figs_dir,final_time,run_status,analyze_status,notes\n'
    local i folder chm gpu ft run_status analyze_status notes
    for i in "${!FOLDERS[@]}"; do
      folder="${FOLDERS[$i]}"
      chm="${CHMS[$i]}"
      gpu="${GPUS[$i]}"
      ft="$(final_time "$folder")"
      run_status="$(get_state "$folder" run_status queued)"
      analyze_status="$(get_state "$folder" analyze_status queued)"
      notes="$(get_state "$folder" notes '')"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$chm" "$folder" "$gpu" "$BASE/$folder/0" "$BASE/$folder/figs" \
        "${ft:-}" "$run_status" "$analyze_status" "$notes"
    done
  } > "$CSV.tmp"
  mv "$CSV.tmp" "$CSV"
}

prepare_case() {
  local folder="$1"
  local gpu="$2"
  local path="$BASE/$folder"
  if [ -e "$path" ]; then
    if data_complete "$folder"; then
      mkdir -p "$path/figs"
      set_state "$folder" run_status skipped_complete_data
      set_state "$folder" analyze_status queued
      set_state "$folder" notes data_preexisted_complete
      set_state "$folder" gpu "$gpu"
      return 0
    fi
    local ts backup
    ts="$(date '+%Y%m%d_%H%M%S')"
    backup="$path.backup_$ts"
    mv "$path" "$backup"
    mkdir -p "$path/0" "$path/figs"
    : > "$path/run.log"
    : > "$path/analyze.log"
    set_state "$folder" run_status queued
    set_state "$folder" analyze_status queued
    set_state "$folder" notes "incomplete_backup_${ts}"
    set_state "$folder" gpu "$gpu"
    printf '[%s] %s incomplete; backed up to %s\n' "$(timestamp)" "$folder" "$backup"
  else
    mkdir -p "$path/0" "$path/figs"
    : > "$path/run.log"
    : > "$path/analyze.log"
    set_state "$folder" run_status queued
    set_state "$folder" analyze_status queued
    set_state "$folder" notes created
    set_state "$folder" gpu "$gpu"
    printf '[%s] %s created\n' "$(timestamp)" "$folder"
  fi
}

run_case() {
  local folder="$1"
  local chm="$2"
  local gpu="$3"
  if data_complete "$folder"; then
    set_state "$folder" run_status skipped_complete_data
    return 0
  fi
  set_state "$folder" run_status running
  printf '[%s] START %s CHM=%s GPU=%s\n' "$(timestamp)" "$folder" "$chm" "$gpu"
  (
    cd "$BASE/$folder" || exit 97
    CUDA_VISIBLE_DEVICES="$gpu" RUNSTEP="$RUNSTEP" CHM_VAL="$chm" \
      D_INIT_X=60 D_INIT_Y=60 GEL_BASE_X=175 GEL_BASE_Y=170 \
      "$BIN" > run.log 2>&1
  )
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    set_state "$folder" run_status "failed_rc_${rc}"
    set_state "$folder" notes "run_failed_rc_${rc}"
    printf '[%s] FAIL %s rc=%s\n' "$(timestamp)" "$folder" "$rc"
    return 0
  fi
  if data_complete "$folder"; then
    set_state "$folder" run_status complete
    printf '[%s] DONE %s complete\n' "$(timestamp)" "$folder"
  else
    set_state "$folder" run_status incomplete_after_run
    set_state "$folder" notes incomplete_after_run
    printf '[%s] DONE %s but data incomplete\n' "$(timestamp)" "$folder"
  fi
  return 0
}

run_batch() {
  local i1="$1"
  local i2="$2"
  run_case "${FOLDERS[$i1]}" "${CHMS[$i1]}" "${GPUS[$i1]}" &
  local p1=$!
  run_case "${FOLDERS[$i2]}" "${CHMS[$i2]}" "${GPUS[$i2]}" &
  local p2=$!
  sleep 1
  write_status
  wait "$p1"
  wait "$p2"
  write_status
}

analyze_case() {
  local folder="$1"
  if ! data_complete "$folder"; then
    set_state "$folder" analyze_status skipped_no_complete_data
    return 0
  fi
  set_state "$folder" analyze_status running
  printf '[%s] ANALYZE START %s\n' "$(timestamp)" "$folder"
  python3 "$ANALYSIS" --run-dir "$BASE/$folder" --figs-dir "$BASE/$folder/figs" > "$BASE/$folder/analyze.log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    set_state "$folder" analyze_status "failed_rc_${rc}"
    set_state "$folder" notes "analyze_failed_rc_${rc}"
    printf '[%s] ANALYZE FAIL %s rc=%s\n' "$(timestamp)" "$folder" "$rc"
    return 0
  fi
  if figs_complete "$folder"; then
    set_state "$folder" analyze_status complete
    printf '[%s] ANALYZE DONE %s complete\n' "$(timestamp)" "$folder"
  else
    set_state "$folder" analyze_status incomplete_figs
    set_state "$folder" notes incomplete_figs
    printf '[%s] ANALYZE DONE %s but figs incomplete\n' "$(timestamp)" "$folder"
  fi
  write_status
}

printf '[%s] small CHM pipeline started\n' "$(timestamp)"
for i in "${!FOLDERS[@]}"; do
  prepare_case "${FOLDERS[$i]}" "${GPUS[$i]}"
done
write_status

run_batch 0 1
run_batch 2 3

for folder in "${FOLDERS[@]}"; do
  analyze_case "$folder"
done
write_status
printf '[%s] small CHM pipeline finished\n' "$(timestamp)"
