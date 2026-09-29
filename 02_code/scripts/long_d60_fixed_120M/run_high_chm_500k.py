#!/usr/bin/env python3
"""Run the d_ini=60 high-CHM cases to an absolute target of 500000 TU.

The two worker queues are pinned to physical GPUs so that only one simulation
uses each GPU at a time:

    GPU0: CHM=0.35, then CHM=0.45
    GPU1: CHM=0.40, then CHM=0.50

Each case is restartable through experiment2_resume, which scans the case's
``0`` directory for its latest complete checkpoint.
"""

from __future__ import annotations

import csv
import concurrent.futures
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
BIN = BASE / "source_snapshot" / "experiment2_resume"
OUT = BASE / "analysis_outputs"
STATUS_CSV = OUT / "high_chm_500k_status.csv"
SCHEDULER_LOG = OUT / "high_chm_500k_scheduler.log"

TARGET_TU = int(os.environ.get("TARGET_TU", "500000"))
D_INIT = 60
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "60"))


@dataclass(frozen=True)
class Case:
    name: str
    chm: float
    gpu: int
    queue_position: int

    @property
    def path(self) -> Path:
        return BASE / self.name

    @property
    def data_path(self) -> Path:
        return self.path / "0"

    @property
    def run_log(self) -> Path:
        return self.path / f"run_to_{TARGET_TU}_gpu{self.gpu}.log"


CASES = [
    Case("CHM_035", 0.35, 0, 1),
    Case("CHM_040", 0.40, 1, 1),
    Case("CHM_045", 0.45, 0, 2),
    Case("CHM_050", 0.50, 1, 2),
]

STATE_MIDS = ("rn", "un", "um", "vm", "wm", "wmo")
io_lock = threading.Lock()


def timestamp() -> str:
    return time.strftime("%F %T")


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{timestamp()}] {message}"
    with io_lock:
        with SCHEDULER_LOG.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
    print(line, flush=True)


def state_file(case: Case, key: str) -> Path:
    return case.path / f".{key}"


def read_state(case: Case, key: str, default: str = "") -> str:
    path = state_file(case, key)
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return default


def write_state(case: Case, key: str, value: str | int | float) -> None:
    case.path.mkdir(parents=True, exist_ok=True)
    path = state_file(case, key)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"{value}\n")
    tmp.replace(path)


def state_complete(case: Case, tu: int) -> bool:
    if tu < 0:
        return False
    paths = [case.data_path / f"ungrid{tu}.dat"]
    paths.extend(
        case.data_path / f"gel{gel}{mid}{tu}.dat"
        for gel in (0, 1)
        for mid in STATE_MIDS
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def final_complete_time(case: Case) -> int:
    if not case.data_path.is_dir():
        return 0
    candidates: list[int] = []
    for path in case.data_path.glob("ungrid*.dat"):
        suffix = path.stem.removeprefix("ungrid")
        if suffix.isdigit():
            candidates.append(int(suffix))
    for tu in sorted(candidates, reverse=True):
        if state_complete(case, tu):
            return tu
    return 0


def bodycenter_rows(case: Case, gel: int) -> int:
    path = case.data_path / f"gel{gel}bodycenter0.dat"
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def data_complete(case: Case) -> bool:
    return (
        state_complete(case, TARGET_TU)
        and bodycenter_rows(case, 0) >= TARGET_TU
        and bodycenter_rows(case, 1) >= TARGET_TU
    )


def write_status() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "CHM",
        "d_init",
        "target_tu",
        "gpu",
        "queue_position",
        "final_complete_tu",
        "gel0_bodycenter_rows",
        "gel1_bodycenter_rows",
        "run_status",
        "pid",
        "path",
    ]
    with io_lock:
        tmp = STATUS_CSV.with_suffix(".tmp")
        with tmp.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for case in CASES:
                writer.writerow(
                    {
                        "case": case.name,
                        "CHM": f"{case.chm:.2f}",
                        "d_init": D_INIT,
                        "target_tu": TARGET_TU,
                        "gpu": case.gpu,
                        "queue_position": case.queue_position,
                        "final_complete_tu": final_complete_time(case),
                        "gel0_bodycenter_rows": bodycenter_rows(case, 0),
                        "gel1_bodycenter_rows": bodycenter_rows(case, 1),
                        "run_status": read_state(case, "run_status", "queued"),
                        "pid": read_state(case, "pid"),
                        "path": str(case.path),
                    }
                )
        tmp.replace(STATUS_CSV)


def prepare_case(case: Case) -> None:
    case.data_path.mkdir(parents=True, exist_ok=True)
    (case.path / "figs").mkdir(exist_ok=True)
    write_state(case, "CHM", f"{case.chm:.2f}")
    write_state(case, "d_init", D_INIT)
    write_state(case, "gpu", case.gpu)
    write_state(case, "target_tu", TARGET_TU)
    if data_complete(case):
        write_state(case, "run_status", "complete")
        write_state(case, "notes", "target_data_preexisted_complete")
    else:
        write_state(case, "run_status", "queued")
        write_state(case, "notes", f"resume_from_complete_tu_{final_complete_time(case)}")


def run_case(case: Case) -> bool:
    if data_complete(case):
        write_state(case, "run_status", "skipped_complete")
        log(f"SKIP {case.name}: target {TARGET_TU} already complete")
        write_status()
        return True

    restart_tu = final_complete_time(case)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(case.gpu),
            "TARGET_TU": str(TARGET_TU),
            "CHM_VAL": f"{case.chm:.2f}",
            "D_INIT_X": str(D_INIT),
            "D_INIT_Y": str(D_INIT),
            "GEL_BASE_X": "175",
            "GEL_BASE_Y": "170",
        }
    )
    if restart_tu > 0:
        env["RESTART_TU"] = str(restart_tu)

    write_state(case, "run_status", "starting")
    with case.run_log.open("a") as stream:
        stream.write(
            f"\n===== launch {timestamp()} CHM={case.chm:.2f} d_init={D_INIT} "
            f"gpu={case.gpu} restart_tu={restart_tu} target_tu={TARGET_TU} =====\n"
        )
        stream.flush()
        proc = subprocess.Popen(
            [str(BIN)],
            cwd=str(case.path),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        write_state(case, "pid", proc.pid)
        write_state(case, "run_status", "running")
        log(
            f"START {case.name} CHM={case.chm:.2f} d_ini={D_INIT} "
            f"GPU={case.gpu} pid={proc.pid} restart={restart_tu} target={TARGET_TU}"
        )
        write_status()

        while proc.poll() is None:
            time.sleep(POLL_SECONDS)
            write_status()

        rc = proc.returncode

    write_state(case, "pid", "")
    if rc == 0 and data_complete(case):
        write_state(case, "run_status", "complete")
        write_state(case, "notes", f"completed_at_{timestamp().replace(' ', '_')}")
        log(f"DONE {case.name} GPU={case.gpu} target={TARGET_TU}")
        write_status()
        return True

    final_tu = final_complete_time(case)
    if rc == 0:
        status = f"incomplete_rc0_final_{final_tu}"
    else:
        status = f"failed_rc_{rc}_final_{final_tu}"
    write_state(case, "run_status", status)
    write_state(case, "notes", status)
    log(f"FAIL {case.name} GPU={case.gpu} rc={rc} final_complete_tu={final_tu}")
    write_status()
    return False


def gpu_worker(gpu: int) -> list[tuple[str, bool]]:
    results: list[tuple[str, bool]] = []
    queue = sorted(
        (case for case in CASES if case.gpu == gpu),
        key=lambda case: case.queue_position,
    )
    for case in queue:
        results.append((case.name, run_case(case)))
    return results


def main() -> int:
    if not BIN.is_file() or not os.access(BIN, os.X_OK):
        raise SystemExit(f"Missing executable: {BIN}")
    if TARGET_TU <= 0:
        raise SystemExit(f"Invalid TARGET_TU={TARGET_TU}")

    for case in CASES:
        prepare_case(case)
    write_status()
    log(
        f"Scheduler started target={TARGET_TU}; "
        "GPU0=[CHM_035,CHM_045], GPU1=[CHM_040,CHM_050]"
    )

    results: list[tuple[str, bool]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(gpu_worker, gpu) for gpu in (0, 1)]
        for future in futures:
            results.extend(future.result())

    failures = [name for name, ok in results if not ok]
    if failures:
        log("Scheduler finished with failures: " + ",".join(failures))
        return 1
    log("Scheduler finished: all four cases complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
