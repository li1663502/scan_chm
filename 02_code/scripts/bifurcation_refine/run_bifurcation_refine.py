#!/usr/bin/env python3
"""Two-GPU adaptive search for the high chi_M bifurcation point.

The search brackets the first loss of the quiet locked-pair branch between
chi_M = 0.15 and 0.18.  Each iteration launches two trisection points, so both
GPUs can be used while keeping the number of expensive simulations small.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bifurcation_common import (  # noqa: E402
    BIN,
    DEFAULT_D_INIT,
    OUT,
    RUNS,
    case_dir,
    data_complete,
    format_chm,
    high_branch_decision,
    latest_snapshot_tu,
    metrics_for_run,
    unique_sorted,
)


STATUS_CSV = OUT / "bifurcation_scheduler_status.csv"
MANIFEST_CSV = OUT / "bifurcation_run_manifest.csv"
LOG = OUT / "bifurcation_scheduler.log"
ESTIMATE_JSON = OUT / "bifurcation_search_state.json"


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


def ensure_case(chm: float) -> Path:
    path = case_dir(chm)
    (path / "0").mkdir(parents=True, exist_ok=True)
    write_text(path / ".chm", f"{chm:.8f}")
    return path


def decision_path(chm: float) -> Path:
    return case_dir(chm) / ".decision"


def decision(chm: float) -> str | None:
    path = decision_path(chm)
    if path.exists():
        value = path.read_text().strip()
        return value or None
    return None


def write_decision(chm: float, value: str, target_tu: int, metrics: dict[str, object]) -> None:
    path = ensure_case(chm)
    write_text(path / ".decision", value)
    payload = {
        "chm": chm,
        "decision": value,
        "target_tu": target_tu,
        "metrics": metrics,
        "updated_at": time.strftime("%F %T"),
    }
    (path / ".decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log(f"DECISION {format_chm(chm)} chi_M={chm:.8f} target={target_tu} decision={value}")


def maybe_finalize(chm: float, final_target: int, stages: list[int]) -> str | None:
    current = decision(chm)
    if current:
        return current
    path = ensure_case(chm)
    for target in stages:
        if not data_complete(path, target):
            return None
        metrics = metrics_for_run(path)
        is_final = target >= final_target
        verdict = high_branch_decision(metrics, final=is_final)
        if verdict == "nonlinear":
            write_decision(chm, "nonlinear", target, metrics)
            return "nonlinear"
        if verdict == "locked":
            write_decision(chm, "locked", target, metrics)
            return "locked"
    return None


def next_target(chm: float, final_target: int, stages: list[int]) -> int | None:
    if decision(chm):
        return None
    path = ensure_case(chm)
    for target in stages:
        if not data_complete(path, target):
            return target
        metrics = metrics_for_run(path)
        verdict = high_branch_decision(metrics, final=target >= final_target)
        if verdict == "nonlinear":
            write_decision(chm, "nonlinear", target, metrics)
            return None
        if verdict == "locked":
            write_decision(chm, "locked", target, metrics)
            return None
    return None


def launch(chm: float, target_tu: int, gpu: str) -> RunningCase:
    path = ensure_case(chm)
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
    log_file = (path / f"run_to_{target_tu}.log").open("a")
    log_file.write(f"\n===== launch {time.strftime('%F %T')} chi_M={chm:.8f} gpu={gpu} target={target_tu} =====\n")
    log_file.flush()
    log(f"START {format_chm(chm)} chi_M={chm:.8f} GPU={gpu} target={target_tu}")
    proc = subprocess.Popen([str(BIN)], cwd=str(path), env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return RunningCase(chm=chm, target_tu=target_tu, gpu=str(gpu), proc=proc, log_handle=log_file)


def known_decisions(final_target: int, stages: list[int]) -> dict[float, str]:
    known = {0.15: "locked", 0.18: "nonlinear"}
    if RUNS.exists():
        for path in RUNS.iterdir():
            if not path.is_dir():
                continue
            chm_file = path / ".chm"
            try:
                chm = float(chm_file.read_text().strip()) if chm_file.exists() else float(path.name[4:].replace("p", "."))
            except ValueError:
                continue
            value = maybe_finalize(chm, final_target, stages)
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


def write_status(running: list[RunningCase], final_target: int, stages: list[int], lo: float, hi: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    chms = set()
    if RUNS.exists():
        for path in RUNS.iterdir():
            if not path.is_dir():
                continue
            chm_file = path / ".chm"
            try:
                chms.add(float(chm_file.read_text().strip()) if chm_file.exists() else float(path.name[4:].replace("p", ".")))
            except ValueError:
                continue
    for chm in sorted(chms):
        path = ensure_case(chm)
        metrics = metrics_for_run(path)
        rows.append(
            {
                "case": format_chm(chm),
                "CHM": f"{chm:.8f}",
                "final_target": final_target,
                "latest_snapshot_tu": latest_snapshot_tu(path),
                "bodycenter_rows": int(metrics["n"]),
                "d_mean": f"{float(metrics['d_mean']):.6f}" if int(metrics["n"]) else "",
                "d_std": f"{float(metrics['d_std']):.6f}" if int(metrics["n"]) else "",
                "d_amp90": f"{float(metrics['d_amp90']):.6f}" if int(metrics["n"]) else "",
                "bound_fraction": f"{float(metrics['bound_fraction']):.6f}" if int(metrics["n"]) else "",
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
            "final_target",
            "latest_snapshot_tu",
            "bodycenter_rows",
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
        writer = csv.DictWriter(handle, fieldnames=["case", "CHM", "path"])
        writer.writeheader()
        for chm in sorted(chms):
            writer.writerow({"case": format_chm(chm), "CHM": f"{chm:.8f}", "path": str(case_dir(chm))})

    payload = {
        "updated_at": time.strftime("%F %T"),
        "active_running": [
            {"chm": item.chm, "target_tu": item.target_tu, "gpu": item.gpu, "pid": item.proc.pid}
            for item in running
        ],
        "upper_transition": {
            "lo_locked": lo,
            "hi_nonlinear": hi,
            "midpoint": 0.5 * (lo + hi),
            "width": hi - lo,
        },
        "stages": stages,
    }
    ESTIMATE_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_analysis_and_plot() -> None:
    analyze = SCRIPT_DIR / "analyze_bifurcation_refine.py"
    plot = SCRIPT_DIR / "plot_bifurcation_refine.py"
    subprocess.run([sys.executable, str(analyze)], check=False)
    subprocess.run([sys.executable, str(plot)], check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lo", type=float, default=0.15)
    parser.add_argument("--hi", type=float, default=0.18)
    parser.add_argument("--tol", type=float, default=0.002)
    parser.add_argument("--final-target", type=int, default=200000)
    parser.add_argument("--stages", default="120000,160000,200000")
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0,1"))
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    if not BIN.exists():
        print(f"Missing binary: {BIN}", file=sys.stderr)
        return 2
    gpus = [gpu for gpu in args.gpus.replace(" ", "").split(",") if gpu]
    if not gpus:
        print("No GPUs configured", file=sys.stderr)
        return 2
    stages = [int(value) for value in args.stages.split(",") if value.strip()]
    if stages[-1] != args.final_target:
        stages.append(args.final_target)
    stages = sorted(set(stages))

    log(
        "Scheduler started "
        f"lo={args.lo:.8f} hi={args.hi:.8f} tol={args.tol:.8f} "
        f"stages={stages} gpus={gpus} max_parallel={args.max_parallel}"
    )

    running: list[RunningCase] = []
    gpu_cursor = 0
    queued: list[float] = []

    while True:
        still_running: list[RunningCase] = []
        for item in running:
            rc = item.proc.poll()
            if rc is None:
                still_running.append(item)
                continue
            item.log_handle.close()
            path = ensure_case(item.chm)
            if rc == 0 and data_complete(path, item.target_tu):
                write_text(path / ".run_status", f"complete_to_{item.target_tu}")
                log(f"DONE {format_chm(item.chm)} target={item.target_tu} GPU={item.gpu}")
                maybe_finalize(item.chm, args.final_target, stages)
            elif rc == 0:
                write_text(path / ".run_status", f"incomplete_rc0_to_{item.target_tu}")
                log(f"INCOMPLETE {format_chm(item.chm)} target={item.target_tu}")
            else:
                write_text(path / ".run_status", f"failed_rc_{rc}_to_{item.target_tu}")
                log(f"FAILED {format_chm(item.chm)} target={item.target_tu} rc={rc}")
        running = still_running

        known = known_decisions(args.final_target, stages)
        lo, hi = current_bracket(known, args.lo, args.hi)
        width = hi - lo
        write_status(running, args.final_target, stages, lo, hi)

        if width <= args.tol and not running:
            log(f"Search complete lo={lo:.8f} hi={hi:.8f} width={width:.8f}")
            run_analysis_and_plot()
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
            target = next_target(chm, args.final_target, stages)
            if target is None:
                continue
            gpu = gpus[gpu_cursor % len(gpus)]
            gpu_cursor += 1
            running.append(launch(chm, target, gpu))
            launched = True

        if launched:
            write_status(running, args.final_target, stages, lo, hi)
            time.sleep(2)
            continue

        if running:
            time.sleep(max(5, args.poll_seconds))
            continue

        if not queued and width <= args.tol:
            log(f"Search complete lo={lo:.8f} hi={hi:.8f} width={width:.8f}")
            run_analysis_and_plot()
            return 0

        if not queued:
            log("No queued or running work remains before tolerance was reached")
            run_analysis_and_plot()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
