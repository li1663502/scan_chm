#!/usr/bin/env python3
"""Diagnose time-window bias and build a phase-aligned bifurcation branch."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_bifurcation_refine import (  # noqa: E402
    METRICS_CSV,
    baseline_analysis_end,
    main as analyze_main,
)
from bifurcation_common import OUT, load_distances, mean_std, percentile  # noqa: E402


SENSITIVITY_CSV = OUT / "sampling_window_sensitivity.csv"
BRANCH_CSV = OUT / "phase_aligned_bifurcation_branch.csv"
DIAGNOSIS_MD = OUT / "sampling_window_diagnosis.md"

SMOOTHING_POINTS = 501
WINDOW_LENGTHS = (10000, 20000, 40000, 50000, 100000)


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return math.nan


def read_metric_rows() -> list[dict[str, str]]:
    analyze_main()
    with METRICS_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def marker_time(path: Path, name: str) -> float | None:
    marker = path / name
    if not marker.exists():
        return None
    try:
        return float(marker.read_text().strip())
    except ValueError:
        return None


def analysis_bounds(row: dict[str, str]) -> tuple[float, float]:
    path = Path(row["path"])
    chm = as_float(row, "CHM")
    source = row["source"]
    start = marker_time(path, ".metric_min_time")
    if start is None:
        start = 0.0
    elif source in {"continuation", "supplement"}:
        start += 1.0
    if source == "baseline":
        end = float(baseline_analysis_end(chm))
    else:
        end = as_float(row, "final_time") + 1.0
    return start, end


def filtered_series(row: dict[str, str]) -> tuple[list[float], list[float]]:
    start, end = analysis_bounds(row)
    times, distances = load_distances(Path(row["path"]), max_time=end)
    filtered = [
        (time, distance)
        for time, distance in zip(times, distances)
        if time >= start and math.isfinite(distance)
    ]
    return [time for time, _distance in filtered], [distance for _time, distance in filtered]


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    mean, std = mean_std(values)
    return {
        "mean": mean,
        "std": std,
        "p05": percentile(ordered, 5.0),
        "p95": percentile(ordered, 95.0),
        "min": ordered[0],
        "max": ordered[-1],
    }


def moving_average(values: list[float], width: int) -> list[float]:
    if not values:
        return []
    half = max(0, width // 2)
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    smoothed: list[float] = []
    for index in range(len(values)):
        lo = max(0, index - half)
        hi = min(len(values), index + half + 1)
        smoothed.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return smoothed


def smoothed_extrema(
    path: Path,
    start: float,
    end: float,
) -> tuple[float, float, float, float, int]:
    times, distances = load_distances(path, max_time=end)
    selected = [
        (time, distance)
        for time, distance in zip(times, distances)
        if time >= start and math.isfinite(distance)
    ]
    if not selected:
        return math.nan, math.nan, math.nan, math.nan, 0
    selected_times = [time for time, _distance in selected]
    selected_distances = [distance for _time, distance in selected]
    smoothed = moving_average(selected_distances, SMOOTHING_POINTS)
    lower_index = min(range(len(smoothed)), key=smoothed.__getitem__)
    upper_index = max(range(len(smoothed)), key=smoothed.__getitem__)
    return (
        smoothed[lower_index],
        smoothed[upper_index],
        selected_times[lower_index],
        selected_times[upper_index],
        len(smoothed),
    )


def sensitivity_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        chm = as_float(row, "CHM")
        if chm < 0.177:
            continue
        times, distances = filtered_series(row)
        if not distances:
            continue
        data_start = times[0]
        data_end = times[-1] + 1.0
        duration = data_end - data_start
        lengths = sorted({length for length in WINDOW_LENGTHS if length <= duration + 1e-9})
        if int(duration) not in lengths:
            lengths.append(int(duration))
        for length in lengths:
            window_start = max(data_start, data_end - length)
            values = [
                distance
                for time, distance in zip(times, distances)
                if time >= window_start
            ]
            stats = distribution(values)
            output.append(
                {
                    "source": row["source"],
                    "case": row["case"],
                    "CHM": f"{chm:.8f}",
                    "window_length": int(data_end - window_start),
                    "window_start": int(window_start),
                    "window_end": int(data_end),
                    "n": len(values),
                    "d_mean": f"{stats['mean']:.6f}",
                    "d_std": f"{stats['std']:.6f}",
                    "d_p05": f"{stats['p05']:.6f}",
                    "d_p95": f"{stats['p95']:.6f}",
                    "d_min": f"{stats['min']:.6f}",
                    "d_max": f"{stats['max']:.6f}",
                    "basis": "fixed elapsed-time window ending at the run analysis cutoff",
                }
            )
    return output


def select_state_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        chm = as_float(row, "CHM")
        state = row["state"]
        source = row["source"]
        include = (
            (state == "unbound" and source == "baseline" and chm <= 0.03)
            or (state == "locked" and source == "baseline" and chm <= 0.15)
            or (state == "locked" and source == "continuation" and 0.15 < chm <= 0.1777)
            or (state in {"transition", "nonlinear"} and source == "continuation" and 0.177 < chm < 0.18)
        )
        if not include:
            continue
        branch = "transition" if 0.177 < chm < 0.18 and state != "locked" else state
        center = as_float(row, "d_mean")
        lower = as_float(row, "d_p05") if branch == "transition" else center
        upper = as_float(row, "d_p95") if branch == "transition" else center
        output.append(
            {
                "branch": branch,
                "source": source,
                "case": row["case"],
                "CHM": f"{chm:.8f}",
                "d_center": f"{center:.6f}",
                "d_lower": f"{lower:.6f}",
                "d_upper": f"{upper:.6f}",
                "t_lower": "",
                "t_upper": "",
                "window_start": row["late_start_time"],
                "window_end": row["final_time"],
                "smoothing_points": 0,
                "sample_basis": row["basis"],
                "cycle_status": "stationary mean" if branch != "transition" else "critical-band spread",
            }
        )
    return output


def breathing_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    anchor = next(
        row
        for row in rows
        if row["source"] == "baseline" and abs(as_float(row, "CHM") - 0.18) < 1e-9
    )
    breathing_cases = [(anchor, 160000.0, 200000.0, "complete-cycle anchor")]
    breathing_cases.extend(
        (row, 200001.0, min(240000.0, as_float(row, "final_time") + 1.0), "first phase-aligned excursion")
        for row in rows
        if row["source"] == "supplement" and as_float(row, "CHM") > 0.18
    )
    for row, start, end, status in breathing_cases:
        path = Path(row["path"])
        lower, upper, t_lower, t_upper, count = smoothed_extrema(path, start, end)
        center = 0.5 * (lower + upper)
        output.append(
            {
                "branch": "breathing",
                "source": row["source"],
                "case": row["case"],
                "CHM": f"{as_float(row, 'CHM'):.8f}",
                "d_center": f"{center:.6f}",
                "d_lower": f"{lower:.6f}",
                "d_upper": f"{upper:.6f}",
                "t_lower": f"{t_lower:.0f}",
                "t_upper": f"{t_upper:.0f}",
                "window_start": f"{start:.0f}",
                "window_end": f"{end:.0f}",
                "smoothing_points": SMOOTHING_POINTS,
                "sample_basis": "global extrema of the 501-point moving-average distance in a fixed phase window",
                "cycle_status": status if count else "missing",
            }
        )
    return sorted(output, key=lambda item: float(item["CHM"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def values_for_case(
    rows: list[dict[str, object]],
    source: str,
    chm: float,
) -> dict[int, dict[str, object]]:
    return {
        int(row["window_length"]): row
        for row in rows
        if row["source"] == source and abs(float(row["CHM"]) - chm) < 1e-9
    }


def polyline_variation(points: list[tuple[float, float]]) -> float:
    ordered = sorted(points)
    return sum(abs(right[1] - left[1]) for left, right in zip(ordered, ordered[1:]))


def write_diagnosis(
    metric_rows: list[dict[str, str]],
    sensitivity: list[dict[str, object]],
    branch: list[dict[str, object]],
) -> None:
    current_nonlinear = [row for row in metric_rows if row["state"] == "nonlinear"]
    current_lower = [(as_float(row, "CHM"), as_float(row, "d_p05")) for row in current_nonlinear]
    current_upper = [(as_float(row, "CHM"), as_float(row, "d_p95")) for row in current_nonlinear]
    aligned = [row for row in branch if row["branch"] == "breathing"]
    aligned_lower = [(float(row["CHM"]), float(row["d_lower"])) for row in aligned]
    aligned_upper = [(float(row["CHM"]), float(row["d_upper"])) for row in aligned]

    examples: list[str] = []
    for chm in (0.181, 0.29):
        case_rows = values_for_case(sensitivity, "supplement", chm)
        if not case_rows:
            continue
        shortest = case_rows[min(case_rows)]
        longest = case_rows[max(case_rows)]
        examples.append(
            f"- chi_M={chm:.3f}: P5 changes from {float(shortest['d_p05']):.2f} "
            f"(last {int(shortest['window_length'])} steps) to {float(longest['d_p05']):.2f} "
            f"(full {int(longest['window_length'])}-step post-switch record)."
        )

    with DIAGNOSIS_MD.open("w") as handle:
        handle.write("# Sampling-window diagnosis\n\n")
        handle.write("## Result\n\n")
        handle.write(
            "The large zig-zag in the previous orange polyline is not a single numerical branch. "
            "It joins independent baseline histories and fixed-seed continuation runs after sorting only by chi_M. "
            "Time-window phase bias then changes the reported lower/upper distances further.\n\n"
        )
        handle.write("## Window sensitivity examples\n\n")
        handle.write("\n".join(examples) + "\n\n")
        handle.write("## Physical variability that remains\n\n")
        handle.write(
            "The baseline chi_M=0.23, 0.25, and 0.28 records contain capture, large escape, or incomplete-cycle episodes. "
            "Those differences are history-sensitive dynamics and must remain as unconnected validation observations; "
            "choosing prettier samples would hide real behavior.\n\n"
        )
        handle.write("## Plotting rule used\n\n")
        handle.write(
            "The clean branch uses only one fixed-seed continuation protocol. For chi_M>0.18 it takes the global lower "
            "and upper values of a 501-point moving-average distance over the same elapsed-time window t=200001-239999. "
            "The chi_M=0.18 anchor uses t=160000-199999. Critical-band states are shown but not connected to the breathing envelope.\n\n"
        )
        handle.write("## Roughness check\n\n")
        handle.write(
            f"- Previous mixed polyline total variation: lower={polyline_variation(current_lower):.2f}, "
            f"upper={polyline_variation(current_upper):.2f}.\n"
        )
        handle.write(
            f"- Phase-aligned branch total variation: lower={polyline_variation(aligned_lower):.2f}, "
            f"upper={polyline_variation(aligned_upper):.2f}.\n"
        )
        handle.write(
            "- The phase-aligned supplement records contain one resolved maximum but do not yet contain the following full minimum; "
            "the lower edge is therefore the common locked starting phase, not a proven asymptotic minimum branch.\n"
        )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    metric_rows = read_metric_rows()
    sensitivity = sensitivity_rows(metric_rows)
    branch = select_state_rows(metric_rows) + breathing_rows(metric_rows)
    branch.sort(key=lambda item: (float(item["CHM"]), str(item["branch"])))
    write_csv(SENSITIVITY_CSV, sensitivity)
    write_csv(BRANCH_CSV, branch)
    write_diagnosis(metric_rows, sensitivity, branch)
    print(f"Wrote {SENSITIVITY_CSV}")
    print(f"Wrote {BRANCH_CSV}")
    print(f"Wrote {DIAGNOSIS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
