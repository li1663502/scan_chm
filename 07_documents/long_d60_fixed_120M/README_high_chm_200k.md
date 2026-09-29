# Experiment 2 high-CHM extension to 200000 TU

Purpose: extend and supplement the `d_ini=60` pair-attractor sweep to inspect
whether high `CHM` produces richer nonlinear synchronization/capture dynamics.

## Source

The modified source is kept in:

```bash
source_snapshot/
```

The new binary is:

```bash
source_snapshot/experiment2_resume
```

Main source changes:

- `TARGET_TU` controls the absolute target simulation time.
- The program scans `0/ungrid*.dat` plus `gel0/gel1 {rn,un,um,vm,wm,wmo}` files.
- It resumes from the latest complete saved state.
- `CHM_020` therefore continues from `120000 TU` toward `200000 TU`.

Rebuild:

```bash
cd source_snapshot
./build_resume.sh
```

## Run plan

Stage 1:

- `CHM_020`, `CHM=0.20`, `d_ini=60`, continue `120000 -> 200000 TU`

Stage 2, started automatically only after stage 1 completes:

- `CHM_015`, `CHM=0.15`, continue to `200000 TU`
- `CHM_018`, `CHM=0.18`, new run to `200000 TU`
- `CHM_023`, `CHM=0.23`, new run to `200000 TU`
- `CHM_025`, `CHM=0.25`, new run to `200000 TU`
- `CHM_028`, `CHM=0.28`, new run to `200000 TU`
- `CHM_030`, `CHM=0.30`, new run to `200000 TU`

Stage 2 uses two GPUs in parallel (`GPUS=0,1`, `MAX_PARALLEL=2`).

## Status

Scheduler PID:

```bash
analysis_outputs/high_chm_200k_scheduler.pid
```

Status CSV:

```bash
analysis_outputs/high_chm_200k_status.csv
```

Scheduler log:

```bash
analysis_outputs/high_chm_200k_scheduler.log
```

Per-case run log:

```bash
CHM_*/run_200k.log
```

