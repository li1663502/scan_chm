#!/usr/bin/env python3
"""Fixed-point supplemental continuation runs for the Experiment 2 bifurcation.

This scheduler complements the adaptive critical-point search.  It runs a
curated set of chi_M values so the final bifurcation diagram has enough points
near the threshold and on the post-0.18 nonlinear envelope.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bifurcation_common import (  # noqa: E402
    BASELINE,
    BIN,
    CONTINUATION_RUNS,
    DEFAULT_D_INIT,
    OUT,
    SUPPLEMENT_RUNS,
    data_complete,
    format_chm,
    latest_snapshot_tu,
    metrics_for_run,
)


STATUS_CSV = OUT / "supplement_scheduler_status.csv"
PLAN_CSV = OUT / "supplement_run_plan.csv"
LOG = OUT / "supplement_scheduler.log"
STATE_JSON = OUT / "supplement_search_state.json"

STATE_FIELDS = ("rn", "un", "um", "vm", "wm", "wmo")


@dataclass(frozen=True)
class Job:
    branch: str
    chm: float
    seed_case: str
    seed_tu: int
    target_tu: int
    root_name: str

    @property
    def root(self) -> Path:
        if self.root_name == "continuation":
            return CONTINUATION_RUNS
        if self.root_name == "supplement":
            return SUPPLEMENT_RUNS
        raise ValueError(f"Unknown root_name: {self.root_name}")

    @property
    def path(self) -> Path:
        return self.root / format_chm(self.chm)


@dataclass
class RunningJob:
    job: Job
    gpu: str
    proc: subprocess.Popen
    log_handle: object


PLAN = [
    # Finish the two partial threshold probes from the previous adaptive search.
    Job("critical-complete", 0.17703704, "CHM_015", 200000, 280000, "continuation"),
    Job("critical-complete", 0.17740741, "CHM_015", 200000, 280000, "continuation"),
    # Add denser threshold points for a visually smoother onset.
    Job("critical-dense", 0.17720000, "CHM_015", 200000, 280000, "continuation"),
    Job("post-0.18", 0.18100000, "CHM_018", 200000, 240000, "supplement"),
    Job("critical-dense", 0.17750000, "CHM_015", 200000, 280000, "continuation"),
    Job("post-0.18", 0.18250000, "CHM_018", 200000, 240000, "supplement"),
    Job("critical-dense", 0.17760000, "CHM_015", 200000, 280000, "continuation"),
    Job("post-0.18", 0.18500000, "CHM_018", 200000, 240000, "supplement"),
    Job("critical-dense", 0.17770000, "CHM_015", 200000, 280000, "continuation"),
    Job("post-0.18", 0.18750000, "CHM_018", 200000, 240000, "supplement"),
    # Fill the nonlinear envelope between the existing coarse baseline points.
    Job("post-0.18", 0.19000000, "CHM_018", 200000, 240000, "supplement"),
    Job("post-0.18", 0.19500000, "CHM_018", 200000, 240000, "supplement"),
    Job("post-0.18", 0.20500000, "CHM_018", 200000, 240000, "supplement"),
    Job("post-0.18", 0.21500000, "CHM_018", 200000, 240000, "supplement"),
    Job("post-0.18", 0.22000000, "CHM_018", 200000, 240000, "supplement"),
    Job("post-0.18", 0.24000000, "CHM_018", 200000, 240000, "supplement"),
    Job("post-0.18", 0.26000000, "CHM_018", 200000, 240000, "supplement"),
    Job("post-0.18", 0.29000000, "CHM_018", 200000, 240000, "supplement"),
]


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(f"[{time.strftime('%F %T')}] {message}\n")
        handle.flush()


def read_text(path: Path, default: str = "") -> str:
    return path.read_text().strip() if path.exists() else default


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")


def seed_files(seed_case: str, seed_tu: int) -> list[tuple[Path, str]]:
    source = BASELINE / seed_case / "0"
    files = [(source / f"ungrid{seed_tu}.dat", f"ungrid{seed_tu}.dat")]
    for gel in (0, 1):
        for field in STATE_FIELDS:
            files.append((source / f"gel{gel}{field}{seed_tu}.dat", f"gel{gel}{field}{seed_tu}.dat"))
        files.append((source / f"gel{gel}bodycenter0.dat", f"gel{gel}bodycenter0.dat"))
    return files


def ensure_seed(job: Job) -> None:
    data = job.path / "0"
    data.mkdir(parents=True, exist_ok=True)
    write_text(job.path / ".chm", f"{job.chm:.8f}")
    write_text(job.path / ".branch", job.branch)
    write_text(job.path / ".seed_case", job.seed_case)
    write_text(job.path / ".seed_tu", str(job.seed_tu))
    write_text(job.path / ".metric_min_time", str(job.seed_tu))
    if read_text(job.path / ".seed_ready") == "yes":
        return
    for source, name in seed_files(job.seed_case, job.seed_tu):
        if not source.exists():
            raise FileNotFoundError(f"Missing seed file: {source}")
        shutil.copy2(source, data / name)
    write_text(job.path / ".seed_ready", "yes")
    write_text(job.path / ".run_status", "seeded")
    log(f"SEEDED {format_chm(job.chm)} branch={job.branch} from {job.seed_case}@{job.seed_tu}")


def job_complete(job: Job) -> bool:
    return data_complete(job.path, job.target_tu)


def launch(job: Job, gpu: str) -> RunningJob:
    ensure_seed(job)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TARGET_TU": str(job.target_tu),
            "CHM_VAL": f"{job.chm:.8f}",
            "D_INIT_X": str(DEFAULT_D_INIT),
            "D_INIT_Y": str(DEFAULT_D_INIT),
            "GEL_BASE_X": "175",
            "GEL_BASE_Y": "170",
        }
    )
    write_text(job.path / ".gpu", str(gpu))
    write_text(job.path / ".run_status", f"running_to_{job.target_tu}")
    log_file = (job.path / f"supplement_to_{job.target_tu}.log").open("a")
    log_file.write(
        f"\n===== supplement {time.strftime('%F %T')} branch={job.branch} "
        f"chi_M={job.chm:.8f} gpu={gpu} target={job.target_tu} =====\n"
    )
    log_file.flush()
    log(f"START {format_chm(job.chm)} branch={job.branch} GPU={gpu} target={job.target_tu}")
    proc = subprocess.Popen([str(BIN)], cwd=str(job.path), env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return RunningJob(job=job, gpu=str(gpu), proc=proc, log_handle=log_file)


def write_plan() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with PLAN_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["branch", "CHM", "seed_case", "seed_tu", "target_tu", "root", "path"],
        )
        writer.writeheader()
        for job in PLAN:
            writer.writerow(
                {
                    "branch": job.branch,
                    "CHM": f"{job.chm:.8f}",
                    "seed_case": job.seed_case,
                    "seed_tu": job.seed_tu,
                    "target_tu": job.target_tu,
                    "root": job.root_name,
                    "path": str(job.path),
                }
            )


def write_status(running: list[RunningJob]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for job in PLAN:
        metrics = metrics_for_run(job.path, min_time=job.seed_tu)
        has_metrics = int(metrics["n"]) > 0
        rows.append(
            {
                "branch": job.branch,
                "case": format_chm(job.chm),
                "CHM": f"{job.chm:.8f}",
                "seed_case": job.seed_case,
                "seed_tu": job.seed_tu,
                "target_tu": job.target_tu,
                "latest_snapshot_tu": latest_snapshot_tu(job.path),
                "post_seed_rows": int(metrics["n"]),
                "d_mean": f"{float(metrics['d_mean']):.6f}" if has_metrics else "",
                "d_std": f"{float(metrics['d_std']):.6f}" if has_metrics else "",
                "d_amp90": f"{float(metrics['d_amp90']):.6f}" if has_metrics else "",
                "bound_fraction": f"{float(metrics['bound_fraction']):.6f}" if has_metrics else "",
                "run_status": read_text(job.path / ".run_status", "queued"),
                "gpu": read_text(job.path / ".gpu", ""),
                "path": str(job.path),
            }
        )
    tmp = STATUS_CSV.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(STATUS_CSV)

    STATE_JSON.write_text(
        json.dumps(
            {
                "updated_at": time.strftime("%F %T"),
                "active_running": [
                    {
                        "branch": item.job.branch,
                        "chm": item.job.chm,
                        "target_tu": item.job.target_tu,
                        "gpu": item.gpu,
                        "pid": item.proc.pid,
                    }
                    for item in running
                ],
                "total_jobs": len(PLAN),
                "completed_jobs": sum(1 for job in PLAN if job_complete(job)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def refresh_outputs() -> None:
    subprocess.run([sys.executable, str(SCRIPT_DIR / "analyze_bifurcation_refine.py")], check=False)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "plot_bifurcation_refine.py")], check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0,1"))
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    if not BIN.exists():
        print(f"Missing binary: {BIN}", file=sys.stderr)
        return 2
    gpus = [gpu for gpu in args.gpus.replace(" ", "").split(",") if gpu]
    if not gpus:
        print("No GPUs configured", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT_RUNS.mkdir(parents=True, exist_ok=True)
    CONTINUATION_RUNS.mkdir(parents=True, exist_ok=True)
    write_plan()
    log(f"Supplement scheduler started jobs={len(PLAN)} gpus={gpus} max_parallel={args.max_parallel}")

    running: list[RunningJob] = []
    gpu_cursor = 0
    pending = [job for job in PLAN if not job_complete(job)]
    for job in PLAN:
        if job_complete(job):
            write_text(job.path / ".run_status", f"complete_to_{job.target_tu}")
        elif read_text(job.path / ".run_status") != f"running_to_{job.target_tu}":
            write_text(job.path / ".run_status", "queued")
    write_status(running)

    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            if job_complete(job):
                write_text(job.path / ".run_status", f"complete_to_{job.target_tu}")
                continue
            gpu = gpus[gpu_cursor % len(gpus)]
            gpu_cursor += 1
            running.append(launch(job, gpu))
            write_status(running)

        time.sleep(max(5, args.poll_seconds))
        still_running: list[RunningJob] = []
        for item in running:
            rc = item.proc.poll()
            if rc is None:
                still_running.append(item)
                continue
            item.log_handle.close()
            if rc == 0 and job_complete(item.job):
                write_text(item.job.path / ".run_status", f"complete_to_{item.job.target_tu}")
                log(f"DONE {format_chm(item.job.chm)} branch={item.job.branch} target={item.job.target_tu} GPU={item.gpu}")
            elif rc == 0:
                write_text(item.job.path / ".run_status", f"incomplete_rc0_to_{item.job.target_tu}")
                log(f"INCOMPLETE {format_chm(item.job.chm)} branch={item.job.branch} target={item.job.target_tu}")
                pending.insert(0, item.job)
            else:
                write_text(item.job.path / ".run_status", f"failed_rc_{rc}_to_{item.job.target_tu}")
                log(f"FAILED {format_chm(item.job.chm)} branch={item.job.branch} target={item.job.target_tu} rc={rc}")
                pending.insert(0, item.job)
        running = still_running
        write_status(running)

    log("Supplement scheduler complete; refreshing outputs")
    refresh_outputs()
    write_status(running)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
