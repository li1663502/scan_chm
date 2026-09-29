#!/usr/bin/env python3
"""Shared helpers for Experiment 2 bifurcation refinement.

The refinement focuses on the quiet locked-pair to nonlinear/breathing
transition near chi_M = 0.15-0.18.  It uses robust late-time distance metrics so
that one noisy spike does not dominate the decision.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
REFINE_ROOT = ROOT / "bifurcation_refine"
RUNS = REFINE_ROOT / "runs"
CONTINUATION_RUNS = REFINE_ROOT / "continuation_runs"
SUPPLEMENT_RUNS = REFINE_ROOT / "supplement_runs"
OUT = REFINE_ROOT / "analysis_outputs"
BASELINE = ROOT / "sweep_long_d60_fixed_120M"
BASELINE_TABLE = ROOT / "experiment2_final_outputs" / "experiment2_d60_full_CHM_ladder.csv"
BIN = BASELINE / "source_snapshot" / "experiment2_resume"

DEFAULT_D_INIT = 60
QUIET_STD_MAX = 4.5
QUIET_AMP90_MAX = 12.0
QUIET_MEAN_MAX = 62.0
QUIET_BOUND_MIN = 0.90
QUIET_SLOPE_ABS_MAX = 25.0

EARLY_NONLINEAR_STD_MIN = 8.0
EARLY_NONLINEAR_AMP90_MIN = 25.0
EARLY_NONLINEAR_P95_MIN = 80.0
EARLY_NONLINEAR_BOUND_MAX = 0.75


def format_chm(chm: float) -> str:
    token = f"{chm:.5f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"CHM_{token}"


def parse_chm_token(token: str) -> float:
    if token.startswith("CHM_"):
        token = token[4:]
    return float(token.replace("p", "."))


def case_dir(chm: float) -> Path:
    return RUNS / format_chm(chm)


def percentile(sorted_values: list[float], percent: float) -> float:
    if not sorted_values:
        return math.nan
    k = (len(sorted_values) - 1) * percent / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - k) + sorted_values[hi] * (k - lo)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(var)


def linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return math.nan
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0.0:
        return math.nan
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom


def read_bodycenter(path: Path, max_time: float | None = None) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                time = float(parts[0])
                if max_time is not None and time >= max_time:
                    break
                rows.append((time, float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    return rows


def load_distances(run_dir: Path, max_time: float | None = None) -> tuple[list[float], list[float]]:
    gel0 = read_bodycenter(run_dir / "0" / "gel0bodycenter0.dat", max_time=max_time)
    gel1 = read_bodycenter(run_dir / "0" / "gel1bodycenter0.dat", max_time=max_time)
    n_rows = min(len(gel0), len(gel1))
    times: list[float] = []
    distances: list[float] = []
    for row0, row1 in zip(gel0[:n_rows], gel1[:n_rows]):
        times.append(row0[0])
        distances.append(math.hypot(row0[1] - row1[1], row0[2] - row1[2]))
    return times, distances


def bodycenter_rows(run_dir: Path, gel: int) -> int:
    path = run_dir / "0" / f"gel{gel}bodycenter0.dat"
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def state_complete(run_dir: Path, target_tu: int) -> bool:
    data = run_dir / "0"
    if not (data / f"ungrid{target_tu}.dat").exists():
        return False
    for gel in (0, 1):
        for field in ("rn", "un", "um", "vm", "wm", "wmo"):
            path = data / f"gel{gel}{field}{target_tu}.dat"
            if not path.exists() or path.stat().st_size == 0:
                return False
    return True


def data_complete(run_dir: Path, target_tu: int) -> bool:
    return (
        state_complete(run_dir, target_tu)
        and bodycenter_rows(run_dir, 0) >= target_tu
        and bodycenter_rows(run_dir, 1) >= target_tu
    )


def latest_snapshot_tu(run_dir: Path) -> int:
    data = run_dir / "0"
    if not data.exists():
        return 0
    latest = 0
    for path in data.glob("ungrid*.dat"):
        suffix = path.stem.replace("ungrid", "")
        if suffix.isdigit():
            latest = max(latest, int(suffix))
    return latest


def metrics_for_run(
    run_dir: Path,
    late_fraction: float = 0.25,
    min_time: float | None = None,
    max_time: float | None = None,
) -> dict[str, float | int | str]:
    times, distances = load_distances(run_dir, max_time=max_time)
    if min_time is not None or max_time is not None:
        filtered = [
            (time, distance)
            for time, distance in zip(times, distances)
            if (min_time is None or time >= min_time)
            and (max_time is None or time < max_time)
        ]
        times = [time for time, _distance in filtered]
        distances = [distance for _time, distance in filtered]
    if not distances:
        return {
            "n": 0,
            "final_time": 0,
            "late_start_time": 0,
            "d_mean": math.nan,
            "d_std": math.nan,
            "d_min": math.nan,
            "d_max": math.nan,
            "d_p05": math.nan,
            "d_p95": math.nan,
            "d_amp90": math.nan,
            "bound_fraction": math.nan,
            "contact_fraction": math.nan,
            "slope_per_100k": math.nan,
            "basis": "missing bodycenter data",
        }

    start = int((1.0 - late_fraction) * len(distances))
    late = distances[start:]
    late_times = times[start:]
    d_mean, d_std = mean_std(late)
    sorted_late = sorted(late)
    d_p05 = percentile(sorted_late, 5.0)
    d_p95 = percentile(sorted_late, 95.0)
    slope = linear_slope(late_times, late) * 100000.0
    return {
        "n": len(distances),
        "final_time": int(times[-1]) if times else 0,
        "late_start_time": int(late_times[0]) if late_times else 0,
        "d_mean": d_mean,
        "d_std": d_std,
        "d_min": min(late),
        "d_max": max(late),
        "d_p05": d_p05,
        "d_p95": d_p95,
        "d_amp90": d_p95 - d_p05,
        "bound_fraction": sum(value < 60.0 for value in late) / len(late),
        "contact_fraction": sum(value < 56.0 for value in late) / len(late),
        "slope_per_100k": slope,
        "basis": (
            f"last {late_fraction:.0%} of trajectory"
            + (f" after t >= {min_time:g}" if min_time is not None else "")
            + (f" before t < {max_time:g}" if max_time is not None else "")
        ),
    }


def is_quiet_locked(metrics: dict[str, float | int | str]) -> bool:
    return (
        float(metrics["d_mean"]) <= QUIET_MEAN_MAX
        and float(metrics["d_std"]) <= QUIET_STD_MAX
        and float(metrics["d_amp90"]) <= QUIET_AMP90_MAX
        and float(metrics["bound_fraction"]) >= QUIET_BOUND_MIN
        and abs(float(metrics["slope_per_100k"])) <= QUIET_SLOPE_ABS_MAX
    )


def is_early_nonlinear(metrics: dict[str, float | int | str]) -> bool:
    return (
        float(metrics["d_std"]) >= EARLY_NONLINEAR_STD_MIN
        or float(metrics["d_amp90"]) >= EARLY_NONLINEAR_AMP90_MIN
        or float(metrics["d_p95"]) >= EARLY_NONLINEAR_P95_MIN
        or float(metrics["bound_fraction"]) <= EARLY_NONLINEAR_BOUND_MAX
    )


def high_branch_decision(metrics: dict[str, float | int | str], final: bool) -> str:
    if int(metrics["n"]) <= 0:
        return "missing"
    if is_quiet_locked(metrics):
        return "locked" if final else "extend"
    if is_early_nonlinear(metrics):
        return "nonlinear"
    return "nonlinear" if final else "extend"


def classify_for_plot(
    metrics: dict[str, float | int | str],
    manual_state: str = "",
) -> str:
    manual = manual_state.lower()
    if "strong nonlinear" in manual:
        return "nonlinear"
    if is_early_nonlinear(metrics) and float(metrics["d_p95"]) >= 75.0:
        return "nonlinear"
    if "unbound" in manual or "escape" in manual:
        return "unbound"
    if "locked rotating pair" in manual:
        return "locked"
    if is_quiet_locked(metrics):
        return "locked"
    if is_early_nonlinear(metrics):
        return "nonlinear"
    return "transition"


def unique_sorted(values: Iterable[float]) -> list[float]:
    rounded = {round(value, 8) for value in values}
    return sorted(rounded)
