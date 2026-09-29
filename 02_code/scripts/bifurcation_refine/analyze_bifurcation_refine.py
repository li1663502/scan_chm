#!/usr/bin/env python3
"""Summarize Experiment 2 bifurcation-refinement runs."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bifurcation_common import (  # noqa: E402
    BASELINE,
    BASELINE_TABLE,
    CONTINUATION_RUNS,
    OUT,
    REFINE_ROOT,
    RUNS,
    SUPPLEMENT_RUNS,
    classify_for_plot,
    format_chm,
    metrics_for_run,
    parse_chm_token,
    unique_sorted,
)


METRICS_CSV = OUT / "bifurcation_refine_metrics.csv"
SUMMARY_MD = OUT / "bifurcation_refine_summary.md"
ESTIMATE_CSV = OUT / "bifurcation_estimates.csv"

BASELINE_CASE_BY_CHM = {
    0.0: "CHM_000",
    0.0025: "CHM_00025",
    0.005: "CHM_0005",
    0.01: "CHM_001",
    0.03: "CHM_003",
    0.05: "CHM_005",
    0.08: "CHM_008",
    0.10: "CHM_010",
    0.15: "CHM_015",
    0.18: "CHM_018",
    0.20: "CHM_020",
    0.23: "CHM_023",
    0.25: "CHM_025",
    0.28: "CHM_028",
    0.30: "CHM_030",
}


def baseline_analysis_end(chm: float) -> int:
    return 200000 if chm >= 0.15 else 120000


def read_baseline_states() -> dict[float, str]:
    states: dict[float, str] = {}
    if not BASELINE_TABLE.exists():
        return states
    with BASELINE_TABLE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                states[round(float(row["CHM"]), 8)] = row.get("state", "")
            except (KeyError, ValueError):
                continue
    return states


def baseline_cases() -> list[tuple[str, float, Path, str]]:
    rows: list[tuple[str, float, Path, str]] = []
    states = read_baseline_states()
    seen_paths: set[Path] = set()
    if BASELINE_TABLE.exists():
        with BASELINE_TABLE.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    chm = float(row["CHM"])
                except (KeyError, ValueError):
                    continue
                case = row.get("case") or BASELINE_CASE_BY_CHM.get(round(chm, 8), format_chm(chm))
                path = BASELINE / case
                if path.exists() and path not in seen_paths:
                    rows.append((case, chm, path, states.get(round(chm, 8), "")))
                    seen_paths.add(path)
    return rows


def scan_run_root(run_root: Path, source: str) -> list[tuple[str, float, Path, str, str]]:
    rows: list[tuple[str, float, Path, str, str]] = []
    if not run_root.exists():
        return rows
    for path in sorted(run_root.iterdir()):
        if not path.is_dir():
            continue
        chm_file = path / ".chm"
        try:
            chm = float(chm_file.read_text().strip()) if chm_file.exists() else parse_chm_token(path.name)
        except ValueError:
            continue
        decision_file = path / ".decision"
        decision = decision_file.read_text().strip() if decision_file.exists() else ""
        rows.append((source, path.name, chm, path, decision))
    return rows


def refine_cases() -> list[tuple[str, str, float, Path, str]]:
    return (
        scan_run_root(RUNS, "refine")
        + scan_run_root(CONTINUATION_RUNS, "continuation")
        + scan_run_root(SUPPLEMENT_RUNS, "supplement")
    )


def metric_min_time(path: Path) -> float | None:
    marker = path / ".metric_min_time"
    if not marker.exists():
        return None
    try:
        return float(marker.read_text().strip())
    except ValueError:
        return None


def row_for_case(source: str, case: str, chm: float, path: Path, manual_state: str) -> dict[str, object]:
    max_time = baseline_analysis_end(chm) if source == "baseline" else None
    metrics = metrics_for_run(path, min_time=metric_min_time(path), max_time=max_time)
    state = classify_for_plot(metrics, manual_state)
    decision = manual_state if source == "baseline" else manual_state
    return {
        "source": source,
        "case": case,
        "CHM": f"{chm:.8f}",
        "path": str(path),
        "n": int(metrics["n"]),
        "final_time": int(metrics["final_time"]),
        "latest_snapshot_tu": latest_snapshot(path),
        "late_start_time": int(metrics["late_start_time"]),
        "d_mean": fmt(metrics["d_mean"]),
        "d_std": fmt(metrics["d_std"]),
        "d_min": fmt(metrics["d_min"]),
        "d_max": fmt(metrics["d_max"]),
        "d_p05": fmt(metrics["d_p05"]),
        "d_p95": fmt(metrics["d_p95"]),
        "d_amp90": fmt(metrics["d_amp90"]),
        "bound_fraction": fmt(metrics["bound_fraction"]),
        "contact_fraction": fmt(metrics["contact_fraction"]),
        "slope_per_100k": fmt(metrics["slope_per_100k"]),
        "state": state,
        "decision": decision,
        "basis": metrics["basis"],
    }


def latest_snapshot(path: Path) -> int:
    data = path / "0"
    latest = 0
    for item in data.glob("ungrid*.dat"):
        suffix = item.stem.replace("ungrid", "")
        if suffix.isdigit():
            latest = max(latest, int(suffix))
    return latest


def fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number:.6f}"


def estimate_bracket(rows: list[dict[str, object]], low: float, high: float) -> tuple[float, float] | None:
    points: list[tuple[float, str]] = []
    for row in rows:
        try:
            chm = float(row["CHM"])
        except (KeyError, ValueError):
            continue
        if chm < low - 1e-12 or chm > high + 1e-12:
            continue
        state = str(row["state"])
        decision = str(row.get("decision", ""))
        if state == "locked" or decision == "locked":
            points.append((chm, "locked"))
        elif state in {"nonlinear", "unbound", "transition"} or decision == "nonlinear":
            points.append((chm, "nonlinear"))
    points = sorted(points)
    if not points:
        return None
    locked = [chm for chm, state in points if state == "locked"]
    nonlinear = [chm for chm, state in points if state == "nonlinear"]
    brackets = [(lo, hi) for lo in locked for hi in nonlinear if lo < hi]
    if not brackets:
        return None
    return min(brackets, key=lambda item: item[1] - item[0])


def estimate_lower_capture(rows: list[dict[str, object]]) -> tuple[float, float] | None:
    unbound: list[float] = []
    locked: list[float] = []
    for row in rows:
        try:
            chm = float(row["CHM"])
        except (KeyError, ValueError):
            continue
        if chm < -1e-12 or chm > 0.061:
            continue
        state = str(row["state"])
        if state == "locked":
            locked.append(chm)
        elif state == "unbound":
            unbound.append(chm)
    brackets = [(lo, hi) for lo in unbound for hi in locked if lo < hi]
    if not brackets:
        return None
    return min(brackets, key=lambda item: item[1] - item[0])


def write_outputs(rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "case",
        "CHM",
        "path",
        "n",
        "final_time",
        "latest_snapshot_tu",
        "late_start_time",
        "d_mean",
        "d_std",
        "d_min",
        "d_max",
        "d_p05",
        "d_p95",
        "d_amp90",
        "bound_fraction",
        "contact_fraction",
        "slope_per_100k",
        "state",
        "decision",
        "basis",
    ]
    tmp = METRICS_CSV.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(METRICS_CSV)

    lower = estimate_lower_capture(rows)
    upper = estimate_bracket(rows, 0.145, 0.185)
    with ESTIMATE_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["transition", "lo_locked_or_unbound", "hi_locked_or_nonlinear", "midpoint", "width"])
        writer.writeheader()
        if lower:
            writer.writerow(
                {
                    "transition": "capture_lower",
                    "lo_locked_or_unbound": f"{lower[0]:.8f}",
                    "hi_locked_or_nonlinear": f"{lower[1]:.8f}",
                    "midpoint": f"{0.5 * (lower[0] + lower[1]):.8f}",
                    "width": f"{lower[1] - lower[0]:.8f}",
                }
            )
        if upper:
            writer.writerow(
                {
                    "transition": "locked_to_nonlinear_upper",
                    "lo_locked_or_unbound": f"{upper[0]:.8f}",
                    "hi_locked_or_nonlinear": f"{upper[1]:.8f}",
                    "midpoint": f"{0.5 * (upper[0] + upper[1]):.8f}",
                    "width": f"{upper[1] - upper[0]:.8f}",
                }
            )

    with SUMMARY_MD.open("w") as handle:
        handle.write("# Experiment 2 bifurcation refinement\n\n")
        handle.write(f"Root: `{REFINE_ROOT}`\n\n")
        handle.write("## Current critical brackets\n\n")
        if lower:
            handle.write(
                f"- Lower capture threshold: `{lower[0]:.5f} < chi_M,c1 < {lower[1]:.5f}` "
                f"(midpoint `{0.5 * (lower[0] + lower[1]):.5f}`, width `{lower[1] - lower[0]:.5f}`).\n"
            )
        else:
            handle.write("- Lower capture threshold: not bracketed yet.\n")
        if upper:
            handle.write(
                f"- Upper locked-to-nonlinear threshold: `{upper[0]:.5f} < chi_M,c2 < {upper[1]:.5f}` "
                f"(midpoint `{0.5 * (upper[0] + upper[1]):.5f}`, width `{upper[1] - upper[0]:.5f}`).\n"
            )
        else:
            handle.write("- Upper locked-to-nonlinear threshold: not bracketed yet.\n")
        handle.write("\n## Distance metrics\n\n")
        handle.write("| source | case | chi_M | n | mean | std | P5-P95 amp | bound% | state |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in sorted(rows, key=lambda item: (float(item["CHM"]), str(item["source"]))):
            bound = float(row["bound_fraction"]) * 100.0 if row["bound_fraction"] else math.nan
            handle.write(
                f"| {row['source']} | {row['case']} | {float(row['CHM']):.5f} | {row['n']} | "
                f"{float(row['d_mean']):.2f} | {float(row['d_std']):.2f} | "
                f"{float(row['d_amp90']):.2f} | {bound:.1f} | {row['state']} |\n"
            )


def main() -> int:
    all_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for case, chm, path, state in baseline_cases():
        row = row_for_case("baseline", case, chm, path, state)
        all_rows.append(row)
        seen.add(("baseline", f"{chm:.8f}"))
    for source, case, chm, path, decision in refine_cases():
        row = row_for_case("refine", case, chm, path, decision)
        row["source"] = source
        all_rows.append(row)
    all_rows = [row for row in all_rows if int(row["n"]) > 0]
    all_rows.sort(key=lambda item: (float(item["CHM"]), str(item["source"])))
    write_outputs(all_rows)
    print(f"Wrote {METRICS_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
