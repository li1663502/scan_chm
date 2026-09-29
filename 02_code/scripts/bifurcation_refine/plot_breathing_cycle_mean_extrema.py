#!/usr/bin/env python3
"""Plot means of detected breathing-cycle maxima and minima."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import cairo

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_bifurcation_refine import METRICS_CSV  # noqa: E402
from analyze_sampling_sensitivity import moving_average  # noqa: E402
from bifurcation_common import OUT, load_distances, mean_std  # noqa: E402
from plot_bifurcation_existing_data import (  # noqa: E402
    draw_circle,
    draw_polyline,
    draw_text,
    read_csv,
)
from plot_breathing_centroid_extrema import (  # noqa: E402
    SELECTED_CSV,
    read_threshold,
    selected_rows,
    write_selected,
)


PDF_OUT = OUT / "chiM_breathing_cycle_mean_extrema.pdf"
METRICS_OUT = OUT / "breathing_cycle_extrema_metrics.csv"

SMOOTHING_POINTS = 501
BASELINE_COMPLETE_WINDOW = (100000.0, 200000.0)


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return math.nan


def collapse_candidates(candidates: list[tuple[str, int, float]]) -> list[tuple[str, int, float]]:
    output: list[tuple[str, int, float]] = []
    for candidate in candidates:
        if output and candidate[0] == output[-1][0]:
            previous = output[-1]
            more_extreme = (
                candidate[2] > previous[2]
                if candidate[0] == "max"
                else candidate[2] < previous[2]
            )
            if more_extreme:
                output[-1] = candidate
            continue
        output.append(candidate)
    return output


def extract_major_extrema(
    times: list[float],
    distances: list[float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    smoothed = moving_average(distances, SMOOTHING_POINTS)
    if len(smoothed) < 3:
        return [], []
    minimum = min(smoothed)
    maximum = max(smoothed)
    span = maximum - minimum
    lower_gate = minimum + 0.40 * span
    upper_gate = minimum + 0.60 * span
    required_turn = max(1.0, 0.03 * span)
    candidates: list[tuple[str, int, float]] = []

    probe = min(2000, len(smoothed) - 1)
    initial_trend = smoothed[probe] - smoothed[0]
    if smoothed[0] <= lower_gate and initial_trend > 1.0:
        candidates.append(("min", 0, smoothed[0]))
    elif smoothed[0] >= upper_gate and initial_trend < -1.0:
        candidates.append(("max", 0, smoothed[0]))

    previous_sign = 0
    for index in range(1, len(smoothed)):
        delta = smoothed[index] - smoothed[index - 1]
        sign = 1 if delta > 0.0 else -1 if delta < 0.0 else previous_sign
        if previous_sign and sign != previous_sign:
            extremum_index = index - 1
            if extremum_index >= len(smoothed) - SMOOTHING_POINTS // 2:
                previous_sign = sign
                continue
            extremum_type = "max" if previous_sign > 0 else "min"
            value = smoothed[extremum_index]
            tail = smoothed[extremum_index:]
            resolved_turn = (
                value - min(tail) >= required_turn
                if extremum_type == "max"
                else max(tail) - value >= required_turn
            )
            if not resolved_turn:
                previous_sign = sign
                continue
            if (extremum_type == "max" and value >= upper_gate) or (
                extremum_type == "min" and value <= lower_gate
            ):
                candidates.append((extremum_type, extremum_index, value))
        previous_sign = sign

    candidates = collapse_candidates(candidates)
    maxima = [(times[index], value) for kind, index, value in candidates if kind == "max"]
    minima = [(times[index], value) for kind, index, value in candidates if kind == "min"]
    return maxima, minima


def metric_path_map() -> dict[tuple[str, str], Path]:
    return {
        (row["source"], row["case"]): Path(row["path"])
        for row in read_csv(METRICS_CSV)
    }


def fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


def summarize_extrema(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    paths = metric_path_map()
    output: list[dict[str, str]] = []
    for row in rows:
        if row["role"] not in {"connected_branch", "independent_validation"}:
            continue
        chm = as_float(row, "CHM")
        start = as_float(row, "window_start")
        end = as_float(row, "window_end")
        if row["source"] == "baseline" and chm in {0.18, 0.20, 0.30}:
            start, end = BASELINE_COMPLETE_WINDOW
        path = paths[(row["source"], row["case"])]
        times, distances = load_distances(path, max_time=end)
        selected = [
            (time, distance)
            for time, distance in zip(times, distances)
            if time >= start and math.isfinite(distance)
        ]
        selected_times = [time for time, _distance in selected]
        selected_distances = [distance for _time, distance in selected]
        maxima, minima = extract_major_extrema(selected_times, selected_distances)
        max_values = [value for _time, value in maxima]
        min_values = [value for _time, value in minima]
        max_mean, max_std = mean_std(max_values) if max_values else (math.nan, math.nan)
        min_mean, min_std = mean_std(min_values) if min_values else (math.nan, math.nan)
        n_effective = min(len(max_values), len(min_values))
        output.append(
            {
                "role": row["role"],
                "source": row["source"],
                "case": row["case"],
                "CHM": row["CHM"],
                "window_start": f"{start:.0f}",
                "window_end": f"{end:.0f}",
                "n_max": str(len(max_values)),
                "n_min": str(len(min_values)),
                "d_max_mean": fmt(max_mean),
                "d_max_std": fmt(max_std),
                "d_min_mean": fmt(min_mean),
                "d_min_std": fmt(min_std),
                "max_times": ";".join(f"{time:.0f}" for time, _value in maxima),
                "min_times": ";".join(f"{time:.0f}" for time, _value in minima),
                "adequacy": "multi-extrema" if n_effective >= 2 else "provisional N=1",
            }
        )
    return sorted(output, key=lambda item: (as_float(item, "CHM"), item["role"]))


def write_metrics(rows: list[dict[str, str]]) -> None:
    with METRICS_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_error_bar(
    ctx: cairo.Context,
    x: float,
    y0: float,
    y1: float,
    color: tuple[float, float, float],
) -> None:
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(1.1)
    ctx.move_to(x, y0)
    ctx.line_to(x, y1)
    ctx.move_to(x - 3.0, y0)
    ctx.line_to(x + 3.0, y0)
    ctx.move_to(x - 3.0, y1)
    ctx.line_to(x + 3.0, y1)
    ctx.stroke()
    ctx.restore()


def draw(
    selected: list[dict[str, str]],
    extrema: list[dict[str, str]],
) -> None:
    locked = [row for row in selected if row["role"] == "locked_branch"]
    connected = [row for row in extrema if row["role"] == "connected_branch"]
    locked.sort(key=lambda row: as_float(row, "CHM"))
    connected.sort(key=lambda row: as_float(row, "CHM"))

    width, height = 690.0, 455.0
    left, right = 76.0, 28.0
    top, bottom = 70.0, 63.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = 0.045, 0.305
    y_min, y_max = 40.0, 145.0
    black = (0.02, 0.02, 0.02)
    orange = (0.89, 0.34, 0.07)
    purple = (0.45, 0.25, 0.65)

    def xmap(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def ymap(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    upper = [(as_float(row, "CHM"), as_float(row, "d_max_mean")) for row in connected]
    lower = [(as_float(row, "CHM"), as_float(row, "d_min_mean")) for row in connected]

    surface = cairo.PDFSurface(str(PDF_OUT), width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()

    ctx.set_source_rgb(0.86, 0.86, 0.86)
    ctx.set_line_width(0.45)
    for ytick in (40, 60, 80, 100, 120, 140):
        ctx.move_to(left, ymap(ytick))
        ctx.line_to(left + plot_width, ymap(ytick))
    for xtick in (0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30):
        ctx.move_to(xmap(xtick), top)
        ctx.line_to(xmap(xtick), top + plot_height)
    ctx.stroke()
    ctx.set_source_rgb(0.08, 0.08, 0.08)
    ctx.set_line_width(1.0)
    ctx.rectangle(left, top, plot_width, plot_height)
    ctx.stroke()

    draw_polyline(
        ctx,
        xmap,
        ymap,
        [(as_float(row, "CHM"), as_float(row, "d_min")) for row in locked],
        black,
        2.0,
    )
    for row in locked:
        draw_circle(ctx, xmap(as_float(row, "CHM")), ymap(as_float(row, "d_min")), black, 3.0)

    ctx.set_source_rgba(0.93, 0.44, 0.12, 0.18)
    ctx.move_to(xmap(upper[0][0]), ymap(upper[0][1]))
    for x, y in upper[1:]:
        ctx.line_to(xmap(x), ymap(y))
    for x, y in reversed(lower):
        ctx.line_to(xmap(x), ymap(y))
    ctx.close_path()
    ctx.fill()
    draw_polyline(ctx, xmap, ymap, upper, orange, 2.0)
    draw_polyline(ctx, xmap, ymap, lower, orange, 1.8, dash=(5.0, 3.0))

    for row in connected:
        x = xmap(as_float(row, "CHM"))
        values = (
            ("d_max_mean", "d_max_std", "n_max"),
            ("d_min_mean", "d_min_std", "n_min"),
        )
        for mean_key, std_key, count_key in values:
            mean = as_float(row, mean_key)
            std = as_float(row, std_key)
            if not math.isnan(std) and std > 0.0:
                draw_error_bar(ctx, x, ymap(mean - std), ymap(mean + std), orange)
            draw_circle(
                ctx,
                x,
                ymap(mean),
                orange,
                3.1,
                open_marker=int(row[count_key]) < 2,
            )

    threshold = read_threshold()
    if threshold is not None:
        ctx.save()
        ctx.set_source_rgb(*purple)
        ctx.set_line_width(1.0)
        ctx.set_dash((3.0, 3.0))
        ctx.move_to(xmap(threshold), top)
        ctx.line_to(xmap(threshold), top + plot_height)
        ctx.stroke()
        ctx.restore()
        draw_text(ctx, xmap(threshold), top - 7.0, f"chi_M,c2 ~ {threshold:.5f}", 8.2, purple, align="center")

    for ytick in (40, 60, 80, 100, 120, 140):
        draw_text(ctx, left - 8.0, ymap(ytick) + 3.7, str(ytick), 8.5, align="right")
    for xtick in (0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30):
        draw_text(ctx, xmap(xtick), top + plot_height + 18.0, f"{xtick:.2f}", 8.2, align="center")

    draw_text(ctx, width / 2.0, 25.0, "Cycle-averaged locked-to-breathing bifurcation", 12.4, bold=True, align="center")
    draw_text(ctx, width / 2.0, 43.0, "markers: mean extrema; error bars: +/- one standard deviation", 8.7, (0.25, 0.25, 0.25), align="center")
    draw_text(ctx, left + plot_width / 2.0, height - 17.0, "chi_M", 10.8, align="center")
    ctx.save()
    ctx.translate(21.0, top + plot_height / 2.0)
    ctx.rotate(-math.pi / 2.0)
    draw_text(ctx, 0.0, 0.0, "centroid distance d", 10.4, align="center")
    ctx.restore()

    legend_x, legend_y = left + 20.0, top + 22.0
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.86)
    ctx.rectangle(legend_x - 9.0, legend_y - 15.0, 188.0, 55.0)
    ctx.fill()
    legend = (
        ("locked mean", black, None),
        ("mean of cycle maxima", orange, None),
        ("mean of cycle minima", orange, (5.0, 3.0)),
    )
    for index, (label, color, dash) in enumerate(legend):
        y = legend_y + 17.0 * index
        ctx.save()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(1.8)
        if dash:
            ctx.set_dash(dash)
        ctx.move_to(legend_x, y)
        ctx.line_to(legend_x + 21.0, y)
        ctx.stroke()
        ctx.restore()
        draw_text(ctx, legend_x + 29.0, y + 3.5, label, 8.0)

    surface.finish()
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", "180", str(PDF_OUT), str(PDF_OUT.with_suffix(""))],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = selected_rows()
    write_selected(selected)
    extrema = summarize_extrema(selected)
    write_metrics(extrema)
    draw(selected, extrema)
    print(f"Wrote {METRICS_OUT}")
    print(f"Wrote {PDF_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
