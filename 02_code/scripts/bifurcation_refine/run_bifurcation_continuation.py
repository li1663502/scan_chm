#!/usr/bin/env python3
"""Continuation search for the Experiment 2 high chi_M bifurcation.

This is the fast path: start every trial from the completed quiet locked branch
at chi_M = 0.15, t = 200000 TU, switch CHM_VAL to the candidate value, and test
whether the locked distance remains quiet over an added continuation window.
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
    data_complete,
    format_chm,
    high_branch_decision,
    latest_snapshot_tu,
    metrics_for_run,
    unique_sorted,
)


STATUS_CSV = OUT / "continuation_scheduler_status.csv"
MANIFEST_CSV = OUT / "continuation_run_manifest.csv"
LOG = OUT / "continuation_scheduler.log"
STATE_JSON = OUT / "continuation_search_state.json"

STATE_FIELDS = ("rn", "un", "um", "vm", "wm", "wmo")


@dataclass
class RunningCase:
    chm: float
    target_tu: int
    gpu: str
    proc: subprocess.Popen
    log_handle: object


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%F %T")
    with LOG.open("a") as handle:
        handle.write(f"[{stamp}] {message}\n")
        handle.flush()


def read_text(path: Path, default: str = "") -> str:
    return path.read_text().strip() if path.exists() else default


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n")


def run_dir(chm: float) -> Path:
    return CONTINUATION_RUNS / format_chm(chm)


def decision_path(chm: float) -> Path:
    return run_dir(chm) / ".decision"


def decision(chm: float) -> str | None:
    path = decision_path(chm)
    if not path.exists():
        return None
    value = path.read_text().strip()
    return value or None


def seed_files(seed_case: str, seed_tu: int) -> list[tuple[Path, str]]:
    source = BASELINE / seed_case / "0"
    files = [(source / f"ungrid{seed_tu}.dat", f"ungrid{seed_tu}.dat")]
    for gel in (0, 1):
        for field in STATE_FIELDS:
            files.append((source / f"gel{gel}{field}{seed_tu}.dat", f"gel{gel}{field}{seed_tu}.dat"))
        files.append((source / f"gel{gel}bodycenter0.dat", f"gel{gel}bodycenter0.dat"))
    return files


def ensure_seed(chm: float, seed_case: str, seed_tu: int) -> Path:
    path = run_dir(chm)
    data = path / "0"
    data.mkdir(parents=True, exist_ok=True)
    write_text(path / ".chm", f"{chm:.8f}")
    write_text(path / ".seed_case", seed_case)
    write_text(path / ".seed_tu", str(seed_tu))
    write_text(path / ".metric_min_time", str(seed_tu))
    if read_text(path / ".seed_ready") == "yes":
        return path
    for source, name in seed_files(seed_case, seed_tu):
        if not source.exists():
            raise FileNotFoundError(f"Missing seed file: {source}")
        target = data / name
        shutil.copy2(source, target)
    write_text(path / ".seed_ready", "yes")
    write_text(path / ".run_status", "seeded")
    log(f"SEEDED {format_chm(chm)} from {seed_case} t={seed_tu}")
    return path


def write_decision(chm: float, value: str, target_tu: int, seed_tu: int, metrics: dict[str, object]) -> None:
    path = run_dir(chm)
    write_text(path / ".decision", value)
    payload = {
        "chm": chm,
        "decision": value,
        "target_tu": target_tu,
        "seed_tu": seed_tu,
        "metrics": metrics,
        "updated_at": time.strftime("%F %T"),
    }
    (path / ".decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log(f"DECISION {format_chm(chm)} chi_M={chm:.8f} target={target_tu} decision={value}")


def maybe_finalize(chm: float, final_target: int, stages: list[int], seed_tu: int) -> str | None:
    current = decision(chm)
    if current:
        return current
    path = run_dir(chm)
    for target in stages:
        if not data_complete(path, target):
            return None
        metrics = metrics_for_run(path, min_time=seed_tu)
        verdict = high_branch_decision(metrics, final=target >= final_target)
        if verdict == "nonlinear":
            write_decision(chm, "nonlinear", target, seed_tu, metrics)
            return "nonlinear"
        if verdict == "locked":
            write_decision(chm, "locked", target, seed_tu, metrics)
            return "locked"
    return None


def next_target(chm: float, final_target: int, stages: list[int], seed_tu: int) -> int | None:
    if decision(chm):
        return None
    path = run_dir(chm)
    for target in stages:
        if not data_complete(path, target):
            return target
        metrics = metrics_for_run(path, min_time=seed_tu)
        verdict = high_branch_decision(metrics, final=target >= final_target)
        if verdict == "nonlinear":
            write_decision(chm, "nonlinear", target, seed_tu, metrics)
            return None
        if verdict == "locked":
            write_decision(chm, "locked", target, seed_tu, metrics)
            return None
    return None


def launch(chm: float, target_tu: int, gpu: str) -> RunningCase:
    path = run_dir(chm)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TARGET_TU": str(target_tu),
            "CHM_VAL": f"{chm:.8f}",
            "D_INIT_X": str(DEFAULT_D_INIT),
            "D_INIT_Y": str(DEFAULT_D_INIT),
            "GEL_BASE_X": "175",
            "GEL_BASE_Y": "170",
        }
    )
    write_text(path / ".gpu", str(gpu))
    write_text(path / ".run_status", f"running_to_{target_tu}")
    log_file = (path / f"continuation_to_{target_tu}.log").open("a")
    log_file.write(f"\n===== continuation {time.strftime('%F %T')} chi_M={chm:.8f} gpu={gpu} target={target_tu} =====\n")
    log_file.flush()
    log(f"START {format_chm(chm)} chi_M={chm:.8f} GPU={gpu} target={target_tu}")
    proc = subprocess.Popen([str(BIN)], cwd=str(path), env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return RunningCase(chm=chm, target_tu=target_tu, gpu=str(gpu), proc=proc, log_handle=log_file)


def known_decisions(final_target: int, stages: list[int], seed_tu: int) -> dict[float, str]:
    known = {0.15: "locked", 0.18: "nonlinear"}
    if CONTINUATION_RUNS.exists():
        for path in CONTINUATION_RUNS.iterdir():
            if not path.is_dir():
                continue
            try:
                chm = float((path / ".chm").read_text().strip())
            except (FileNotFoundError, ValueError):
                continue
            value = maybe_finalize(chm, final_target, stages, seed_tu)
            if value in {"locked", "nonlinear"}:
                known[round(chm, 8)] = value
    return known


def current_bracket(known: dict[float, str], lo_seed: float, hi_seed: float) -> tuple[float, float]:
    lo = lo_seed
    hi = hi_seed
    for chm, value in sorted(known.items()):
        if value == "locked" and lo <= chm < hi:
            lo = max(lo, chm)
        elif value == "nonlinear" and lo < chm <= hi:
            hi = min(hi, chm)
    return lo, hi


def proposed_points(lo: float, hi: float) -> list[float]:
    width = hi - lo
    return unique_sorted([lo + width / 3.0, lo + 2.0 * width / 3.0])


def write_status(running: list[RunningCase], final_target: int, stages: list[int], seed_tu: int, lo: float, hi: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    chms: list[float] = []
    if CONTINUATION_RUNS.exists():
        for path in CONTINUATION_RUNS.iterdir():
            if not path.is_dir():
                continue
            try:
                chms.append(float((path / ".chm").read_text().strip()))
            except (FileNotFoundError, ValueError):
                continue
    for chm in sorted(set(chms)):
        path = run_dir(chm)
        metrics = metrics_for_run(path, min_time=seed_tu)
        has_metrics = int(metrics["n"]) > 0
        rows.append(
            {
                "case": format_chm(chm),
                "CHM": f"{chm:.8f}",
                "seed_tu": seed_tu,
                "final_target": final_target,
                "latest_snapshot_tu": latest_snapshot_tu(path),
                "post_seed_rows": int(metrics["n"]),
                "d_mean": f"{float(metrics['d_mean']):.6f}" if has_metrics else "",
                "d_std": f"{float(metrics['d_std']):.6f}" if has_metrics else "",
                "d_amp90": f"{float(metrics['d_amp90']):.6f}" if has_metrics else "",
                "bound_fraction": f"{float(metrics['bound_fraction']):.6f}" if has_metrics else "",
                "decision": decision(chm) or "",
                "run_status": read_text(path / ".run_status", "queued"),
                "gpu": read_text(path / ".gpu", ""),
                "path": str(path),
            }
        )
    tmp = STATUS_CSV.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        fieldnames = [
            "case",
            "CHM",
            "seed_tu",
            "final_target",
            "latest_snapshot_tu",
            "post_seed_rows",
            "d_mean",
            "d_std",
            "d_amp90",
            "bound_fraction",
            "decision",
            "run_status",
            "gpu",
            "path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(STATUS_CSV)

    with MANIFEST_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "CHM", "seed_tu", "path"])
        writer.writeheader()
        for chm in sorted(set(chms)):
            writer.writerow({"case": format_chm(chm), "CHM": f"{chm:.8f}", "seed_tu": seed_tu, "path": str(run_dir(chm))})

    payload = {
        "updated_at": time.strftime("%F %T"),
        "active_running": [
            {"chm": item.chm, "target_tu": item.target_tu, "gpu": item.gpu, "pid": item.proc.pid}
            for item in running
        ],
        "upper_transition_continuation": {
            "lo_locked": lo,
            "hi_nonlinear": hi,
            "midpoint": 0.5 * (lo + hi),
            "width": hi - lo,
        },
        "seed_tu": seed_tu,
        "stages": stages,
    }
    STATE_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def refresh_outputs() -> None:
    subprocess.run([sys.executable, str(SCRIPT_DIR / "analyze_bifurcation_refine.py")], check=False)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "plot_bifurcation_refine.py")], check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lo", type=float, default=0.15)
    parser.add_argument("--hi", type=float, default=0.18)
    parser.add_argument("--tol", type=float, default=0.002)
    parser.add_argument("--seed-case", default="CHM_015")
    parser.add_argument("--seed-tu", type=int, default=200000)
    parser.add_argument("--extension-stages", default="40000,80000")
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0,1"))
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CONTINUATION_RUNS.mkdir(parents=True, exist_ok=True)
    if not BIN.exists():
        print(f"Missing binary: {BIN}", file=sys.stderr)
        return 2
    gpus = [gpu for gpu in args.gpus.replace(" ", "").split(",") if gpu]
    if not gpus:
        print("No GPUs configured", file=sys.stderr)
        return 2

    extension_stages = sorted({int(value) for value in args.extension_stages.split(",") if value.strip()})
    stages = [args.seed_tu + extension for extension in extension_stages]
    final_target = max(stages)

    log(
        "Continuation scheduler started "
        f"lo={args.lo:.8f} hi={args.hi:.8f} tol={args.tol:.8f} "
        f"seed={args.seed_case}@{args.seed_tu} stages={stages} gpus={gpus}"
    )

    running: list[RunningCase] = []
    queued: list[float] = []
    gpu_cursor = 0

    while True:
        still_running: list[RunningCase] = []
        for item in running:
            rc = item.proc.poll()
            if rc is None:
                still_running.append(item)
                continue
            item.log_handle.close()
            path = run_dir(item.chm)
            if rc == 0 and data_complete(path, item.target_tu):
                write_text(path / ".run_status", f"complete_to_{item.target_tu}")
                log(f"DONE {format_chm(item.chm)} target={item.target_tu} GPU={item.gpu}")
                maybe_finalize(item.chm, final_target, stages, args.seed_tu)
            elif rc == 0:
                write_text(path / ".run_status", f"incomplete_rc0_to_{item.target_tu}")
                log(f"INCOMPLETE {format_chm(item.chm)} target={item.target_tu}")
            else:
                write_text(path / ".run_status", f"failed_rc_{rc}_to_{item.target_tu}")
                log(f"FAILED {format_chm(item.chm)} target={item.target_tu} rc={rc}")
        running = still_running

        known = known_decisions(final_target, stages, args.seed_tu)
        lo, hi = current_bracket(known, args.lo, args.hi)
        width = hi - lo
        write_status(running, final_target, stages, args.seed_tu, lo, hi)

        if width <= args.tol and not running:
            log(f"Continuation search complete lo={lo:.8f} hi={hi:.8f} width={width:.8f}")
            refresh_outputs()
            return 0

        for chm in proposed_points(lo, hi):
            if chm <= lo + 1e-10 or chm >= hi - 1e-10:
                continue
            if decision(chm):
                continue
            if any(abs(item.chm - chm) < 1e-10 for item in running):
                continue
            if all(abs(item - chm) >= 1e-10 for item in queued):
                queued.append(chm)

        launched = False
        while queued and len(running) < args.max_parallel:
            chm = queued.pop(0)
            ensure_seed(chm, args.seed_case, args.seed_tu)
            target = next_target(chm, final_target, stages, args.seed_tu)
            if target is None:
                continue
            gpu = gpus[gpu_cursor % len(gpus)]
            gpu_cursor += 1
            running.append(launch(chm, target, gpu))
            launched = True

        if launched:
            write_status(running, final_target, stages, args.seed_tu, lo, hi)
            time.sleep(2)
            continue
        if running:
            time.sleep(max(5, args.poll_seconds))
            continue
        if not queued:
            log("No queued continuation work remains before tolerance was reached")
            refresh_outputs()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
