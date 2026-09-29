#!/usr/bin/env python3
"""Draw an evidence-complete bifurcation figure from existing trajectories only."""

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

from analyze_bifurcation_refine import ESTIMATE_CSV, METRICS_CSV  # noqa: E402
from analyze_sampling_sensitivity import (  # noqa: E402
    BRANCH_CSV,
    distribution,
    main as sampling_main,
    smoothed_extrema,
)
from bifurcation_common import OUT, load_distances  # noqa: E402


PDF_OUT = OUT / "chiM_and_d_bifurcation_existing_data.pdf"
POINTS_CSV = OUT / "existing_data_bifurcation_points.csv"

COMPLETE_CYCLE_CHM = {0.18, 0.20, 0.30}
INTERMITTENT_CHM = {0.23, 0.25, 0.28}
BASELINE_WINDOW_START = 100000.0
BASELINE_WINDOW_END = 200000.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return math.nan


def fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return "" if math.isnan(value) else f"{value:.6f}"


def select_baseline(metrics: list[dict[str, str]], chm: float) -> dict[str, str]:
    return next(
        row
        for row in metrics
        if row["source"] == "baseline" and abs(as_float(row, "CHM") - chm) < 1e-9
    )


def baseline_distribution(row: dict[str, str]) -> dict[str, float]:
    times, distances = load_distances(Path(row["path"]), max_time=BASELINE_WINDOW_END)
    values = [
        distance
        for time, distance in zip(times, distances)
        if time >= BASELINE_WINDOW_START and math.isfinite(distance)
    ]
    return distribution(values)


def build_points(metrics: list[dict[str, str]], branch: list[dict[str, str]]) -> list[dict[str, str]]:
    points: list[dict[str, str]] = []
    for row in branch:
        role = row["branch"]
        if role not in {"unbound", "locked", "transition"}:
            continue
        points.append(
            {
                "role": role,
                "source": row["source"],
                "case": row["case"],
                "CHM": row["CHM"],
                "d_center": row["d_center"],
                "d_lower": row["d_lower"],
                "d_upper": row["d_upper"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "status": row["cycle_status"],
                "panel": "A",
            }
        )

    for row in branch:
        if row["branch"] != "breathing":
            continue
        points.append(
            {
                "role": "first_excursion_max",
                "source": row["source"],
                "case": row["case"],
                "CHM": row["CHM"],
                "d_center": row["d_upper"],
                "d_lower": "",
                "d_upper": row["d_upper"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "status": "resolved first maximum; following minimum unresolved" if row["source"] == "supplement" else "complete-cycle anchor",
                "panel": "A",
            }
        )

    for chm in sorted(COMPLETE_CYCLE_CHM):
        row = select_baseline(metrics, chm)
        lower, upper, t_lower, t_upper, _count = smoothed_extrema(
            Path(row["path"]),
            BASELINE_WINDOW_START,
            BASELINE_WINDOW_END,
        )
        points.append(
            {
                "role": "complete_cycle_extrema",
                "source": "baseline",
                "case": row["case"],
                "CHM": f"{chm:.8f}",
                "d_center": fmt(0.5 * (lower + upper)),
                "d_lower": fmt(lower),
                "d_upper": fmt(upper),
                "window_start": f"{BASELINE_WINDOW_START:.0f}",
                "window_end": f"{BASELINE_WINDOW_END:.0f}",
                "status": f"resolved extrema at t={t_lower:.0f}/{t_upper:.0f}",
                "panel": "A",
            }
        )

    for chm in sorted(COMPLETE_CYCLE_CHM | INTERMITTENT_CHM):
        row = select_baseline(metrics, chm)
        stats = baseline_distribution(row)
        points.append(
            {
                "role": "baseline_history_complete" if chm in COMPLETE_CYCLE_CHM else "baseline_history_intermittent",
                "source": "baseline",
                "case": row["case"],
                "CHM": f"{chm:.8f}",
                "d_center": fmt(stats["mean"]),
                "d_lower": fmt(stats["p05"]),
                "d_upper": fmt(stats["p95"]),
                "window_start": f"{BASELINE_WINDOW_START:.0f}",
                "window_end": f"{BASELINE_WINDOW_END:.0f}",
                "status": "complete periodic evidence" if chm in COMPLETE_CYCLE_CHM else "capture/escape or incomplete-cycle history",
                "panel": "B",
            }
        )
    return sorted(points, key=lambda row: (row["panel"], as_float(row, "CHM"), row["role"]))


def write_points(points: list[dict[str, str]]) -> None:
    with POINTS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)


def read_upper_threshold() -> float | None:
    if not ESTIMATE_CSV.exists():
        return None
    for row in read_csv(ESTIMATE_CSV):
        if row.get("transition") != "locked_to_nonlinear_upper":
            continue
        try:
            return 0.5 * (float(row["lo_locked_or_unbound"]) + float(row["hi_locked_or_nonlinear"]))
        except (KeyError, ValueError):
            return None
    return None


def draw_text(
    ctx: cairo.Context,
    x: float,
    y: float,
    text: str,
    size: float = 10.0,
    color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bold: bool = False,
    align: str = "left",
) -> None:
    ctx.save()
    ctx.select_font_face(
        "Noto Sans",
        cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
    )
    ctx.set_font_size(size)
    xb, _yb, width, _height, _xa, _ya = ctx.text_extents(text)
    if align == "center":
        x -= width / 2.0 + xb
    elif align == "right":
        x -= width + xb
    ctx.set_source_rgb(*color)
    ctx.move_to(x, y)
    ctx.show_text(text)
    ctx.restore()


def draw_polyline(
    ctx: cairo.Context,
    xmap,
    ymap,
    points: list[tuple[float, float]],
    color: tuple[float, float, float],
    width: float,
    dash: tuple[float, ...] | None = None,
) -> None:
    if not points:
        return
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(width)
    if dash:
        ctx.set_dash(dash)
    ctx.move_to(xmap(points[0][0]), ymap(points[0][1]))
    for x, y in points[1:]:
        ctx.line_to(xmap(x), ymap(y))
    ctx.stroke()
    ctx.restore()


def draw_circle(
    ctx: cairo.Context,
    x: float,
    y: float,
    color: tuple[float, float, float],
    radius: float = 3.0,
    open_marker: bool = False,
) -> None:
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.arc(x, y, radius, 0.0, 2.0 * math.pi)
    if open_marker:
        ctx.set_line_width(1.2)
        ctx.stroke()
    else:
        ctx.fill()
    ctx.restore()


def draw_triangle(
    ctx: cairo.Context,
    x: float,
    y: float,
    color: tuple[float, float, float],
    up: bool,
    size: float = 4.0,
) -> None:
    direction = -1.0 if up else 1.0
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.move_to(x, y + direction * size)
    ctx.line_to(x - size, y - direction * size)
    ctx.line_to(x + size, y - direction * size)
    ctx.close_path()
    ctx.fill()
    ctx.restore()


def draw_diamond(
    ctx: cairo.Context,
    x: float,
    y: float,
    color: tuple[float, float, float],
    size: float = 3.5,
) -> None:
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.move_to(x, y - size)
    ctx.line_to(x + size, y)
    ctx.line_to(x, y + size)
    ctx.line_to(x - size, y)
    ctx.close_path()
    ctx.fill()
    ctx.restore()


def draw_whisker(
    ctx: cairo.Context,
    x: float,
    y0: float,
    y1: float,
    color: tuple[float, float, float],
    cap: float = 3.5,
) -> None:
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(1.2)
    ctx.move_to(x, y0)
    ctx.line_to(x, y1)
    ctx.move_to(x - cap, y0)
    ctx.line_to(x + cap, y0)
    ctx.move_to(x - cap, y1)
    ctx.line_to(x + cap, y1)
    ctx.stroke()
    ctx.restore()


def draw_axes(
    ctx: cairo.Context,
    left: float,
    top: float,
    width: float,
    height: float,
    xmap,
    ymap,
    xticks: tuple[float, ...],
    yticks: tuple[int, ...],
) -> None:
    ctx.save()
    ctx.set_source_rgb(0.86, 0.86, 0.86)
    ctx.set_line_width(0.45)
    for value in yticks:
        ctx.move_to(left, ymap(value))
        ctx.line_to(left + width, ymap(value))
    for value in xticks:
        ctx.move_to(xmap(value), top)
        ctx.line_to(xmap(value), top + height)
    ctx.stroke()
    ctx.set_source_rgb(0.08, 0.08, 0.08)
    ctx.set_line_width(1.0)
    ctx.rectangle(left, top, width, height)
    ctx.stroke()
    ctx.restore()

    for value in yticks:
        draw_text(ctx, left - 8.0, ymap(value) + 3.7, str(value), 8.2, align="right")
    for value in xticks:
        draw_text(ctx, xmap(value), top + height + 17.0, f"{value:.2f}", 8.0, align="center")


def draw_figure(points: list[dict[str, str]]) -> None:
    width, height = 760.0, 690.0
    left, right = 78.0, 28.0
    plot_width = width - left - right
    top_a, height_a = 82.0, 270.0
    top_b, height_b = 432.0, 190.0
    orange = (0.89, 0.34, 0.07)
    black = (0.02, 0.02, 0.02)
    gray = (0.36, 0.39, 0.44)
    purple = (0.45, 0.25, 0.65)

    def xmap_a(value: float) -> float:
        return left + (value + 0.002) / 0.307 * plot_width

    def ymap_a(value: float) -> float:
        return top_a + (145.0 - value) / 105.0 * height_a

    def xmap_b(value: float) -> float:
        return left + (value - 0.175) / 0.13 * plot_width

    def ymap_b(value: float) -> float:
        return top_b + (240.0 - value) / 205.0 * height_b

    surface = cairo.PDFSurface(str(PDF_OUT), width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()

    draw_text(ctx, width / 2.0, 25.0, "Experiment 2 bifurcation evidence from existing data", 13.0, bold=True, align="center")
    draw_text(ctx, width / 2.0, 44.0, "resolved extrema are separated from finite-time and history-sensitive observations", 8.8, (0.25, 0.25, 0.25), align="center")
    draw_text(ctx, left, 68.0, "A  Stable branches and currently resolved extrema", 9.5, bold=True)

    draw_axes(
        ctx,
        left,
        top_a,
        plot_width,
        height_a,
        xmap_a,
        ymap_a,
        (0.00, 0.03, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30),
        (40, 60, 80, 100, 120, 140),
    )
    ctx.set_source_rgba(0.93, 0.44, 0.12, 0.07)
    ctx.rectangle(xmap_a(0.18), top_a, xmap_a(0.305) - xmap_a(0.18), height_a)
    ctx.fill()
    ctx.set_source_rgba(0.45, 0.25, 0.65, 0.12)
    ctx.rectangle(xmap_a(0.1770), top_a, xmap_a(0.1800) - xmap_a(0.1770), height_a)
    ctx.fill()

    unbound = [row for row in points if row["role"] == "unbound"]
    locked = [row for row in points if row["role"] == "locked"]
    transition = [row for row in points if row["role"] == "transition"]
    first_max = [row for row in points if row["role"] == "first_excursion_max"]
    complete = [row for row in points if row["role"] == "complete_cycle_extrema"]
    unbound.sort(key=lambda row: as_float(row, "CHM"))
    locked.sort(key=lambda row: as_float(row, "CHM"))
    first_max.sort(key=lambda row: as_float(row, "CHM"))

    draw_polyline(ctx, xmap_a, ymap_a, [(as_float(row, "CHM"), as_float(row, "d_center")) for row in unbound], gray, 1.4, dash=(4.0, 3.0))
    draw_polyline(ctx, xmap_a, ymap_a, [(as_float(row, "CHM"), as_float(row, "d_center")) for row in locked], black, 2.0)
    draw_polyline(ctx, xmap_a, ymap_a, [(as_float(row, "CHM"), as_float(row, "d_upper")) for row in first_max], orange, 1.9)

    for row in unbound:
        draw_circle(ctx, xmap_a(as_float(row, "CHM")), ymap_a(as_float(row, "d_center")), gray)
    for row in locked:
        draw_circle(ctx, xmap_a(as_float(row, "CHM")), ymap_a(as_float(row, "d_center")), black)
    for row in transition:
        x = xmap_a(as_float(row, "CHM"))
        draw_whisker(ctx, x, ymap_a(as_float(row, "d_lower")), ymap_a(as_float(row, "d_upper")), purple)
        draw_circle(ctx, x, ymap_a(as_float(row, "d_center")), purple, 2.7, open_marker=True)
    for row in first_max:
        draw_circle(
            ctx,
            xmap_a(as_float(row, "CHM")),
            ymap_a(as_float(row, "d_upper")),
            orange,
            3.0,
            open_marker=row["source"] == "supplement",
        )
    complete.sort(key=lambda row: as_float(row, "CHM"))
    draw_polyline(
        ctx,
        xmap_a,
        ymap_a,
        [(as_float(row, "CHM"), as_float(row, "d_lower")) for row in complete],
        orange,
        1.7,
        dash=(5.0, 3.0),
    )
    for row in complete:
        x = xmap_a(as_float(row, "CHM"))
        draw_triangle(ctx, x, ymap_a(as_float(row, "d_upper")), orange, up=True)
        draw_triangle(ctx, x, ymap_a(as_float(row, "d_lower")), orange, up=False)

    threshold = read_upper_threshold()
    if threshold is not None:
        ctx.save()
        ctx.set_source_rgb(*purple)
        ctx.set_line_width(1.0)
        ctx.set_dash((3.0, 3.0))
        ctx.move_to(xmap_a(threshold), top_a)
        ctx.line_to(xmap_a(threshold), top_a + height_a)
        ctx.stroke()
        ctx.restore()
        draw_text(ctx, xmap_a(threshold), top_a - 7.0, f"chi_M,c2 ~ {threshold:.5f}", 8.2, purple, align="center")

    draw_text(ctx, xmap_a(0.225), ymap_a(64.0), "lower extrema unresolved at open-circle points", 8.0, orange, align="center")
    draw_text(ctx, left + plot_width / 2.0, top_a + height_a + 38.0, "chi_M", 10.2, align="center")
    ctx.save()
    ctx.translate(21.0, top_a + height_a / 2.0)
    ctx.rotate(-math.pi / 2.0)
    draw_text(ctx, 0.0, 0.0, "inter-gel distance d", 10.0, align="center")
    ctx.restore()

    legend_x, legend_y = left + 18.0, top_a + 20.0
    legend = (
        ("unbound / locked mean", gray),
        ("first excursion maximum", orange),
        ("resolved lower branch (3 points; dashed)", orange),
        ("critical-band spread", purple),
    )
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.86)
    ctx.rectangle(legend_x - 8.0, legend_y - 14.0, 213.0, 69.0)
    ctx.fill()
    for index, (label, color) in enumerate(legend):
        y = legend_y + 16.0 * index
        ctx.set_source_rgb(*color)
        ctx.set_line_width(1.7)
        ctx.move_to(legend_x, y)
        ctx.line_to(legend_x + 20.0, y)
        ctx.stroke()
        draw_text(ctx, legend_x + 28.0, y + 3.4, label, 7.9)

    draw_text(ctx, left, 416.0, "B  Independent high-chi_M baseline histories (P5-P95; not connected)", 9.5, bold=True)
    draw_axes(
        ctx,
        left,
        top_b,
        plot_width,
        height_b,
        xmap_b,
        ymap_b,
        (0.18, 0.20, 0.23, 0.25, 0.28, 0.30),
        (40, 80, 120, 160, 200, 240),
    )
    histories = [row for row in points if row["panel"] == "B"]
    for row in histories:
        chm = as_float(row, "CHM")
        color = orange if row["role"] == "baseline_history_complete" else purple
        x = xmap_b(chm)
        draw_whisker(ctx, x, ymap_b(as_float(row, "d_lower")), ymap_b(as_float(row, "d_upper")), color, cap=5.0)
        if row["role"] == "baseline_history_complete":
            draw_circle(ctx, x, ymap_b(as_float(row, "d_center")), color, 3.2)
        else:
            draw_diamond(ctx, x, ymap_b(as_float(row, "d_center")), color, 3.8)

    labels = {0.23: "capture", 0.25: "large escape", 0.28: "incomplete escape"}
    for chm, label in labels.items():
        row = next(item for item in histories if abs(as_float(item, "CHM") - chm) < 1e-9)
        y = min(234.0, as_float(row, "d_upper") + 10.0)
        draw_text(ctx, xmap_b(chm), ymap_b(y), label, 7.8, purple, align="center")

    draw_text(ctx, left + plot_width / 2.0, top_b + height_b + 37.0, "chi_M", 10.2, align="center")
    ctx.save()
    ctx.translate(21.0, top_b + height_b / 2.0)
    ctx.rotate(-math.pi / 2.0)
    draw_text(ctx, 0.0, 0.0, "distance distribution", 10.0, align="center")
    ctx.restore()
    draw_text(ctx, left + 16.0, top_b + 17.0, "circle: complete-cycle evidence", 7.8, orange)
    draw_text(ctx, left + 174.0, top_b + 17.0, "diamond: intermittent/incomplete", 7.8, purple)

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
    sampling_main()
    metrics = read_csv(METRICS_CSV)
    branch = read_csv(BRANCH_CSV)
    points = build_points(metrics, branch)
    write_points(points)
    draw_figure(points)
    print(f"Wrote {POINTS_CSV}")
    print(f"Wrote {PDF_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
