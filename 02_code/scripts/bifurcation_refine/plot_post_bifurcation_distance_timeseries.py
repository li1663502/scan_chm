#!/usr/bin/env python3
"""Plot one two-gel centroid-distance time series per post-bifurcation CHM."""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cairo

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_bifurcation_refine import (  # noqa: E402
    ESTIMATE_CSV,
    METRICS_CSV,
    baseline_analysis_end,
)
from analyze_sampling_sensitivity import moving_average  # noqa: E402
from bifurcation_common import OUT, load_distances  # noqa: E402
from plot_bifurcation_existing_data import draw_text  # noqa: E402


OUTPUT_DIR = OUT / "post_bifurcation_distance_timeseries"
MANIFEST_CSV = OUTPUT_DIR / "post_bifurcation_distance_timeseries_manifest.csv"
README_OUT = OUTPUT_DIR / "README.md"
OVERVIEW_PNG = OUTPUT_DIR / "post_bifurcation_distance_timeseries_overview.png"
OVERVIEW_PDF = OUTPUT_DIR / "post_bifurcation_distance_timeseries_overview.pdf"
SMOOTHING_POINTS = 501
MAX_DRAW_POINTS = 20000

BLACK = (0.05, 0.05, 0.05)
ORANGE = (0.89, 0.34, 0.07)
GRAY = (0.48, 0.48, 0.48)
LIGHT_GRID = (0.86, 0.86, 0.86)


@dataclass
class Series:
    source: str
    case: str
    chm: float
    path: Path
    original_start: float
    original_end: float
    elapsed: list[float]
    distance: list[float]
    smoothed: list[float]
    protocol: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def upper_threshold() -> tuple[float, float, float]:
    for row in read_csv(ESTIMATE_CSV):
        if row.get("transition") != "locked_to_nonlinear_upper":
            continue
        lower = float(row["lo_locked_or_unbound"])
        upper = float(row["hi_locked_or_nonlinear"])
        return lower, upper, 0.5 * (lower + upper)
    raise RuntimeError("Missing locked-to-nonlinear threshold estimate")


def marker_time(path: Path, name: str) -> float | None:
    marker = path / name
    if not marker.exists():
        return None
    try:
        return float(marker.read_text().strip())
    except ValueError:
        return None


def protocol_label(source: str) -> str:
    if source == "baseline":
        return "independent fixed-D+ run from t = 0"
    if source == "continuation":
        return "switched from locked chi_M = 0.15 at t = 200000 TU"
    if source == "supplement":
        return "switched from chi_M = 0.18 at t = 200000 TU"
    return source


def bounds(row: dict[str, str]) -> tuple[float, float]:
    source = row["source"]
    chm = float(row["CHM"])
    path = Path(row["path"])
    if source == "baseline":
        return 0.0, float(baseline_analysis_end(chm))
    start = marker_time(path, ".metric_min_time")
    if start is None:
        start = marker_time(path, ".seed_tu") or 0.0
    return start + 1.0, float(row["final_time"]) + 1.0


def load_series(row: dict[str, str]) -> Series:
    start, end = bounds(row)
    path = Path(row["path"])
    times, distances = load_distances(path, max_time=end)
    selected = [
        (time, distance)
        for time, distance in zip(times, distances)
        if time >= start and math.isfinite(distance)
    ]
    if not selected:
        raise RuntimeError(f"No finite distance data for {row['source']} {row['case']}")
    original_times = [time for time, _distance in selected]
    values = [distance for _time, distance in selected]
    elapsed = [time - original_times[0] for time in original_times]
    return Series(
        source=row["source"],
        case=row["case"],
        chm=float(row["CHM"]),
        path=path,
        original_start=original_times[0],
        original_end=original_times[-1],
        elapsed=elapsed,
        distance=values,
        smoothed=moving_average(values, SMOOTHING_POINTS),
        protocol=protocol_label(row["source"]),
    )


def post_bifurcation_series() -> tuple[tuple[float, float, float], list[Series]]:
    threshold = upper_threshold()
    rows = [
        row
        for row in read_csv(METRICS_CSV)
        if row["source"] in {"baseline", "continuation", "supplement"}
        and float(row["CHM"]) > threshold[2]
    ]
    rows.sort(key=lambda row: (float(row["CHM"]), row["source"]))
    return threshold, [load_series(row) for row in rows]


def filename_token(chm: float) -> str:
    return f"{chm:.8f}".rstrip("0").rstrip(".").replace(".", "p")


def nice_step(span: float, target_ticks: int = 6) -> float:
    if span <= 0.0:
        return 1.0
    rough = span / target_ticks
    power = 10.0 ** math.floor(math.log10(rough))
    fraction = rough / power
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * power


def axis_ticks(lower: float, upper: float, target_ticks: int = 6) -> list[float]:
    step = nice_step(upper - lower, target_ticks)
    first = math.ceil(lower / step) * step
    output: list[float] = []
    value = first
    while value <= upper + 1e-9 * max(1.0, abs(upper)):
        output.append(value)
        value += step
    return output


def format_tick(value: float) -> str:
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:g}k"
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:g}"


def decimated_pairs(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    stride = max(1, math.ceil(len(xs) / MAX_DRAW_POINTS))
    points = list(zip(xs[::stride], ys[::stride]))
    if points[-1][0] != xs[-1]:
        points.append((xs[-1], ys[-1]))
    return points


def draw_polyline(
    ctx: cairo.Context,
    points: list[tuple[float, float]],
    xmap,
    ymap,
    color: tuple[float, float, float],
    width: float,
    alpha: float = 1.0,
) -> None:
    if not points:
        return
    ctx.save()
    ctx.set_source_rgba(*color, alpha)
    ctx.set_line_width(width)
    ctx.move_to(xmap(points[0][0]), ymap(points[0][1]))
    for x, y in points[1:]:
        ctx.line_to(xmap(x), ymap(y))
    ctx.stroke()
    ctx.restore()


def distance_domain(values: list[float]) -> tuple[float, float]:
    minimum = min(values)
    maximum = max(values)
    span = max(maximum - minimum, 2.0)
    lower = max(0.0, minimum - 0.09 * span)
    upper = maximum + 0.09 * span
    step = nice_step(upper - lower)
    return math.floor(lower / step) * step, math.ceil(upper / step) * step


def draw_figure(ctx: cairo.Context, width: float, height: float, series: Series) -> None:
    left, right = 88.0, 35.0
    top, bottom = 92.0, 70.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = 0.0, max(series.elapsed)
    y_min, y_max = distance_domain(series.distance)

    def xmap(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1.0) * plot_width

    def ymap(value: float) -> float:
        return top + (y_max - value) / max(y_max - y_min, 1.0) * plot_height

    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()

    x_ticks = axis_ticks(x_min, x_max, 5)
    y_ticks = axis_ticks(y_min, y_max, 6)
    ctx.set_source_rgb(*LIGHT_GRID)
    ctx.set_line_width(0.7)
    for value in x_ticks:
        ctx.move_to(xmap(value), top)
        ctx.line_to(xmap(value), top + plot_height)
    for value in y_ticks:
        ctx.move_to(left, ymap(value))
        ctx.line_to(left + plot_width, ymap(value))
    ctx.stroke()

    for value, dash, label in ((50.0, (2.0, 3.0), "gel size d=50"), (60.0, (6.0, 4.0), "locked cutoff d=60")):
        if not (y_min <= value <= y_max):
            continue
        ctx.save()
        ctx.set_source_rgba(*GRAY, 0.8)
        ctx.set_line_width(1.0)
        ctx.set_dash(dash)
        ctx.move_to(left, ymap(value))
        ctx.line_to(left + plot_width, ymap(value))
        ctx.stroke()
        ctx.restore()
        draw_text(ctx, left + plot_width - 6.0, ymap(value) - 5.0, label, 8.0, GRAY, align="right")

    raw = decimated_pairs(series.elapsed, series.distance)
    smooth = decimated_pairs(series.elapsed, series.smoothed)
    draw_polyline(ctx, raw, xmap, ymap, ORANGE, 0.75, alpha=0.28)
    draw_polyline(ctx, smooth, xmap, ymap, ORANGE, 2.0)

    ctx.set_source_rgb(*BLACK)
    ctx.set_line_width(1.2)
    ctx.rectangle(left, top, plot_width, plot_height)
    ctx.stroke()

    for value in x_ticks:
        draw_text(ctx, xmap(value), top + plot_height + 20.0, format_tick(value), 8.8, BLACK, align="center")
    for value in y_ticks:
        draw_text(ctx, left - 10.0, ymap(value) + 3.5, format_tick(value), 8.8, BLACK, align="right")

    draw_text(
        ctx,
        width / 2.0,
        29.0,
        f"Two-gel centroid distance: chi_M = {series.chm:.8f}".rstrip("0").rstrip("."),
        14.0,
        BLACK,
        bold=True,
        align="center",
    )
    draw_text(ctx, width / 2.0, 51.0, series.protocol, 9.4, GRAY, align="center")
    draw_text(
        ctx,
        width / 2.0,
        height - 18.0,
        "elapsed time at displayed chi_M (TU)",
        10.5,
        BLACK,
        align="center",
    )
    ctx.save()
    ctx.translate(24.0, top + plot_height / 2.0)
    ctx.rotate(-math.pi / 2.0)
    draw_text(ctx, 0.0, 0.0, "centroid distance d", 10.5, BLACK, align="center")
    ctx.restore()

    legend_x, legend_y = left + 15.0, top + 18.0
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.84)
    ctx.rectangle(legend_x - 8.0, legend_y - 13.0, 188.0, 43.0)
    ctx.fill()
    ctx.set_source_rgba(*ORANGE, 0.32)
    ctx.set_line_width(1.0)
    ctx.move_to(legend_x, legend_y)
    ctx.line_to(legend_x + 28.0, legend_y)
    ctx.stroke()
    draw_text(ctx, legend_x + 36.0, legend_y + 3.5, "raw d(t)", 8.5, BLACK)
    ctx.set_source_rgb(*ORANGE)
    ctx.set_line_width(2.0)
    ctx.move_to(legend_x, legend_y + 18.0)
    ctx.line_to(legend_x + 28.0, legend_y + 18.0)
    ctx.stroke()
    draw_text(ctx, legend_x + 36.0, legend_y + 21.5, "501-point moving mean", 8.5, BLACK)

    stats = (
        f"min={min(series.distance):.2f}   mean={sum(series.distance) / len(series.distance):.2f}   "
        f"max={max(series.distance):.2f}"
    )
    draw_text(ctx, left + plot_width - 8.0, top + 18.0, stats, 8.7, BLACK, align="right")


def write_plot(series: Series) -> tuple[Path, Path]:
    token = filename_token(series.chm)
    stem = OUTPUT_DIR / f"chiM_{token}_centroid_distance_vs_time"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    width, height = 1000, 600

    image_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    image_ctx = cairo.Context(image_surface)
    draw_figure(image_ctx, float(width), float(height), series)
    image_surface.write_to_png(str(png_path))
    image_surface.finish()

    pdf_surface = cairo.PDFSurface(str(pdf_path), float(width), float(height))
    pdf_ctx = cairo.Context(pdf_surface)
    draw_figure(pdf_ctx, float(width), float(height), series)
    pdf_surface.finish()
    return png_path, pdf_path


def draw_overview(ctx: cairo.Context, png_paths: list[Path]) -> None:
    width, height = 1800.0, 2520.0
    columns, cell_width, cell_height = 3, 600.0, 360.0
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()
    for index, png_path in enumerate(png_paths):
        row, column = divmod(index, columns)
        source = cairo.ImageSurface.create_from_png(str(png_path))
        scale = 0.58
        x = column * cell_width + 10.0
        y = row * cell_height + 6.0
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(scale, scale)
        ctx.set_source_surface(source, 0.0, 0.0)
        ctx.paint()
        ctx.restore()
        source.finish()
        ctx.set_source_rgb(0.78, 0.78, 0.78)
        ctx.set_line_width(0.8)
        ctx.rectangle(x, y, 580.0, 348.0)
        ctx.stroke()

    draw_text(
        ctx,
        width - 18.0,
        height - 11.0,
        f"{len(png_paths)} valid post-bifurcation parameter records",
        9.0,
        GRAY,
        align="right",
    )


def write_overview(png_paths: list[Path]) -> None:
    width, height = 1800, 2520
    image_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    draw_overview(cairo.Context(image_surface), png_paths)
    image_surface.write_to_png(str(OVERVIEW_PNG))
    image_surface.finish()

    pdf_surface = cairo.PDFSurface(str(OVERVIEW_PDF), float(width), float(height))
    draw_overview(cairo.Context(pdf_surface), png_paths)
    pdf_surface.finish()


def write_outputs(threshold: tuple[float, float, float], series_list: list[Series]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    readme_lines = [
        "# 分岔点后双凝胶质心距离时间序列",
        "",
        (
            f"筛选条件：`CHM > {threshold[2]:.8f}`；临界括区 "
            f"`{threshold[0]:.8f} < CHM_c2 < {threshold[1]:.8f}`。"
        ),
        "",
        "每个参数分别输出 PNG 和 PDF。橙色细线为原始 `d(t)`，粗线为 501 点移动平均。",
        "续算和补点图只显示参数切换后的有效时间段，横轴为该参数作用后的经过时间。",
        "总览图：`post_bifurcation_distance_timeseries_overview.png`。",
        "",
        "| CHM | 来源 | 原始时间窗 (TU) | 点数 | d_min | d_mean | d_max | PNG |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    png_paths: list[Path] = []
    for series in series_list:
        png_path, pdf_path = write_plot(series)
        png_paths.append(png_path)
        mean_distance = sum(series.distance) / len(series.distance)
        manifest.append(
            {
                "source": series.source,
                "case": series.case,
                "CHM": f"{series.chm:.8f}",
                "protocol": series.protocol,
                "original_time_start": f"{series.original_start:.0f}",
                "original_time_end": f"{series.original_end:.0f}",
                "elapsed_time_end": f"{series.elapsed[-1]:.0f}",
                "n_points": str(len(series.distance)),
                "d_min": f"{min(series.distance):.6f}",
                "d_mean": f"{mean_distance:.6f}",
                "d_max": f"{max(series.distance):.6f}",
                "png": str(png_path),
                "pdf": str(pdf_path),
            }
        )
        readme_lines.append(
            f"| {series.chm:.8f} | {series.source} | "
            f"{series.original_start:.0f}–{series.original_end:.0f} | {len(series.distance)} | "
            f"{min(series.distance):.2f} | {mean_distance:.2f} | {max(series.distance):.2f} | "
            f"[{png_path.name}]({png_path.name}) |"
        )

    with MANIFEST_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    write_overview(png_paths)
    README_OUT.write_text("\n".join(readme_lines) + "\n")


def main() -> int:
    threshold, series_list = post_bifurcation_series()
    write_outputs(threshold, series_list)
    print(f"Wrote {len(series_list)} PNG and {len(series_list)} PDF plots to {OUTPUT_DIR}")
    print(f"Wrote {MANIFEST_CSV}")
    print(f"Wrote {README_OUT}")
    print(f"Wrote {OVERVIEW_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
