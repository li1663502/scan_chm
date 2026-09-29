#!/usr/bin/env python3
"""Extend Experiment 2 high-CHM pair runs to 200000 TU.

Phase 1 extends the existing CHM=0.20, d_ini=60 run to 200000 TU. Only after
that case is complete does phase 2 start the supplemental CHM sweep with up to
two GPUs in parallel.
"""

from __future__ import annotations

import csv
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
BIN = BASE / "source_snapshot" / "experiment2_resume"
OUT = BASE / "analysis_outputs"
STATUS_CSV = OUT / "high_chm_200k_status.csv"
LOG = OUT / "high_chm_200k_scheduler.log"

TARGET_TU = int(os.environ.get("TARGET_TU", "200000"))
D_INIT = 60
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "2"))
GPUS = [g for g in os.environ.get("GPUS", "0,1").replace(" ", "").split(",") if g]

PHASE1 = [0.20]
PHASE2 = [0.15, 0.18, 0.23, 0.25, 0.28, 0.30]


@dataclass(frozen=True)
class Case:
    chm: float
    phase: int

    @property
    def name(self) -> str:
        return f"CHM_{round(self.chm * 100):03d}"

    @property
    def path(self) -> Path:
        return BASE / self.name


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%F %T")
    with LOG.open("a") as f:
        f.write(f"[{stamp}] {message}\n")
        f.flush()


def final_time(case: Case) -> int:
    data = case.path / "0"
    vals: list[int] = []
    for p in data.glob("ungrid*.dat"):
        suffix = p.stem.replace("ungrid", "")
        if suffix.isdigit():
            vals.append(int(suffix))
    return max(vals) if vals else 0


def state_complete(case: Case, tu: int) -> bool:
    data = case.path / "0"
    if not (data / f"ungrid{tu}.dat").exists():
        return False
    for ig in (0, 1):
        for mid in ("rn", "un", "um", "vm", "wm", "wmo"):
            p = data / f"gel{ig}{mid}{tu}.dat"
            if not p.exists() or p.stat().st_size == 0:
                return False
    return True


def bodycenter_rows(case: Case, ig: int) -> int:
    p = case.path / "0" / f"gel{ig}bodycenter0.dat"
    if not p.exists():
        return 0
    with p.open("rb") as f:
        return sum(1 for _ in f)


def data_complete(case: Case) -> bool:
    return (
        state_complete(case, TARGET_TU)
        and bodycenter_rows(case, 0) >= TARGET_TU
        and bodycenter_rows(case, 1) >= TARGET_TU
    )


def read_state(case: Case, key: str, default: str = "queued") -> str:
    p = case.path / f".{key}"
    return p.read_text().strip() if p.exists() else default


def write_state(case: Case, key: str, value: str) -> None:
    case.path.mkdir(parents=True, exist_ok=True)
    (case.path / f".{key}").write_text(value + "\n")


def write_status(cases: list[Case]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_CSV.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "phase",
                "CHM",
                "d_init",
                "target_tu",
                "final_time",
                "gel0_bodycenter_rows",
                "gel1_bodycenter_rows",
                "gpu",
                "run_status",
                "path",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case": case.name,
                    "phase": case.phase,
                    "CHM": case.chm,
                    "d_init": D_INIT,
                    "target_tu": TARGET_TU,
                    "final_time": final_time(case),
                    "gel0_bodycenter_rows": bodycenter_rows(case, 0),
                    "gel1_bodycenter_rows": bodycenter_rows(case, 1),
                    "gpu": read_state(case, "gpu", ""),
                    "run_status": read_state(case, "run_status"),
                    "path": str(case.path),
                }
            )
    tmp.replace(STATUS_CSV)


def launch_case(case: Case, gpu: str) -> subprocess.Popen:
    case.path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TARGET_TU": str(TARGET_TU),
            "CHM_VAL": str(case.chm),
            "D_INIT_X": str(D_INIT),
            "D_INIT_Y": str(D_INIT),
            "GEL_BASE_X": "175",
            "GEL_BASE_Y": "170",
        }
    )
    write_state(case, "gpu", str(gpu))
    write_state(case, "run_status", "running")
    log(f"START phase={case.phase} {case.name} CHM={case.chm} GPU={gpu} target={TARGET_TU}")
    f = (case.path / "run_200k.log").open("a")
    f.write(f"\n===== high_chm_200k launch {time.strftime('%F %T')} gpu={gpu} target_tu={TARGET_TU} =====\n")
    f.flush()
    proc = subprocess.Popen([str(BIN)], cwd=str(case.path), env=env, stdout=f, stderr=subprocess.STDOUT)
    proc._log_file = f  # type: ignore[attr-defined]
    return proc


def run_phase(cases: list[Case], max_parallel: int, all_cases: list[Case]) -> bool:
    pending = [case for case in cases if not data_complete(case)]
    for case in cases:
        if data_complete(case):
            write_state(case, "run_status", "skipped_complete")
        elif read_state(case, "run_status") != "running":
            write_state(case, "run_status", "queued")
    write_status(all_cases)

    running: dict[subprocess.Popen, Case] = {}
    gpu_for_proc: dict[subprocess.Popen, str] = {}
    next_gpu = 0

    while pending or running:
        while pending and len(running) < max_parallel:
            case = pending.pop(0)
            if data_complete(case):
                write_state(case, "run_status", "skipped_complete")
                continue
            gpu = GPUS[next_gpu % len(GPUS)]
            next_gpu += 1
            proc = launch_case(case, gpu)
            running[proc] = case
            gpu_for_proc[proc] = gpu
            write_status(all_cases)

        time.sleep(60)
        finished: list[subprocess.Popen] = []
        for proc, case in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            finished.append(proc)
            log_file = getattr(proc, "_log_file", None)
            if log_file is not None:
                log_file.close()
            if rc == 0 and data_complete(case):
                write_state(case, "run_status", "complete")
                log(f"DONE {case.name} GPU={gpu_for_proc[proc]}")
            elif rc == 0:
                write_state(case, "run_status", f"incomplete_rc0_final_{final_time(case)}")
                log(f"INCOMPLETE {case.name} rc=0 final_time={final_time(case)}")
                return False
            else:
                write_state(case, "run_status", f"failed_rc_{rc}_final_{final_time(case)}")
                log(f"FAILED {case.name} rc={rc} final_time={final_time(case)}")
                return False

        for proc in finished:
            del running[proc]
            del gpu_for_proc[proc]
        write_status(all_cases)

    write_status(all_cases)
    return True


def main() -> int:
    if not GPUS:
        print("No GPUs configured")
        return 2
    if not BIN.exists():
        print(f"Missing binary: {BIN}")
        return 2

    phase1_cases = [Case(chm, 1) for chm in PHASE1]
    phase2_cases = [Case(chm, 2) for chm in PHASE2]
    all_cases = phase1_cases + phase2_cases

    for case in all_cases:
        if data_complete(case):
            write_state(case, "run_status", "skipped_complete")
        elif read_state(case, "run_status") != "running":
            write_state(case, "run_status", "queued")

    log(f"Scheduler started target={TARGET_TU} GPUs={GPUS} max_parallel={MAX_PARALLEL}")
    ok = run_phase(phase1_cases, 1, all_cases)
    if not ok:
        log("Scheduler stopped before phase 2 because phase 1 did not complete")
        return 1
    log("Phase 1 complete; starting phase 2")
    ok = run_phase(phase2_cases, MAX_PARALLEL, all_cases)
    log("Scheduler finished" if ok else "Scheduler stopped with errors")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
