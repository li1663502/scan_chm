#!/usr/bin/env python3
"""Plot the protocol-consistent, phase-aligned Experiment 2 branch."""

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

from analyze_bifurcation_refine import ESTIMATE_CSV  # noqa: E402
from analyze_sampling_sensitivity import (  # noqa: E402
    BRANCH_CSV,
    SENSITIVITY_CSV,
    main as sampling_main,
)
from bifurcation_common import OUT  # noqa: E402


PDF_OUT = OUT / "chiM_and_d_bifurcation_phase_aligned.pdf"
DIAGNOSTIC_PDF_OUT = OUT / "sampling_window_comparison.pdf"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return math.nan


def read_upper_threshold() -> tuple[float, float] | None:
    if not ESTIMATE_CSV.exists():
        return None
    for row in read_csv(ESTIMATE_CSV):
        if row.get("transition") != "locked_to_nonlinear_upper":
            continue
        try:
            return float(row["lo_locked_or_unbound"]), float(row["hi_locked_or_nonlinear"])
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


def draw_marker(
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


def draw_whisker(
    ctx: cairo.Context,
    x: float,
    y0: float,
    y1: float,
    color: tuple[float, float, float],
    width: float = 1.0,
) -> None:
    ctx.save()
    ctx.set_source_rgb(*color)
    ctx.set_line_width(width)
    ctx.move_to(x, y0)
    ctx.line_to(x, y1)
    ctx.move_to(x - 3.0, y0)
    ctx.line_to(x + 3.0, y0)
    ctx.move_to(x - 3.0, y1)
    ctx.line_to(x + 3.0, y1)
    ctx.stroke()
    ctx.restore()


def convert_pdf(pdf_path: Path) -> None:
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", "180", str(pdf_path), str(pdf_path.with_suffix(""))],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


def draw_main(rows: list[dict[str, str]]) -> None:
    width, height = 680.0, 440.0
    margin_l, margin_r = 75.0, 28.0
    margin_t, margin_b = 58.0, 64.0
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x_min, x_max = -0.002, 0.305
    y_min, y_max = 40.0, 140.0

    def xmap(value: float) -> float:
        return margin_l + (value - x_min) / (x_max - x_min) * plot_w

    def ymap(value: float) -> float:
        return margin_t + (y_max - value) / (y_max - y_min) * plot_h

    colors = {
        "unbound": (0.36, 0.39, 0.44),
        "locked": (0.02, 0.02, 0.02),
        "transition": (0.45, 0.25, 0.65),
        "breathing": (0.89, 0.34, 0.07),
    }
    by_branch: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_branch.setdefault(row["branch"], []).append(row)
    for branch_rows in by_branch.values():
        branch_rows.sort(key=lambda row: as_float(row, "CHM"))

    surface = cairo.PDFSurface(str(PDF_OUT), width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()

    ctx.set_source_rgba(0.93, 0.44, 0.12, 0.08)
    ctx.rectangle(xmap(0.18), margin_t, xmap(x_max) - xmap(0.18), plot_h)
    ctx.fill()
    ctx.set_source_rgba(0.45, 0.25, 0.65, 0.12)
    ctx.rectangle(xmap(0.1770), margin_t, xmap(0.1800) - xmap(0.1770), plot_h)
    ctx.fill()

    ctx.set_line_width(0.45)
    ctx.set_source_rgb(0.86, 0.86, 0.86)
    for ytick in range(40, 141, 20):
        ctx.move_to(margin_l, ymap(ytick))
        ctx.line_to(margin_l + plot_w, ymap(ytick))
    for xtick in (0.0, 0.03, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30):
        ctx.move_to(xmap(xtick), margin_t)
        ctx.line_to(xmap(xtick), margin_t + plot_h)
    ctx.stroke()

    ctx.set_source_rgb(0.08, 0.08, 0.08)
    ctx.set_line_width(1.0)
    ctx.rectangle(margin_l, margin_t, plot_w, plot_h)
    ctx.stroke()

    unbound = [(as_float(row, "CHM"), as_float(row, "d_center")) for row in by_branch.get("unbound", [])]
    locked = [(as_float(row, "CHM"), as_float(row, "d_center")) for row in by_branch.get("locked", [])]
    breathing_lower = [(as_float(row, "CHM"), as_float(row, "d_lower")) for row in by_branch.get("breathing", [])]
    breathing_upper = [(as_float(row, "CHM"), as_float(row, "d_upper")) for row in by_branch.get("breathing", [])]

    draw_polyline(ctx, xmap, ymap, unbound, colors["unbound"], 1.4, dash=(4.0, 3.0))
    draw_polyline(ctx, xmap, ymap, locked, colors["locked"], 2.0)
    if breathing_lower and breathing_upper:
        ctx.set_source_rgba(0.93, 0.44, 0.12, 0.19)
        ctx.move_to(xmap(breathing_upper[0][0]), ymap(breathing_upper[0][1]))
        for x, y in breathing_upper[1:]:
            ctx.line_to(xmap(x), ymap(y))
        for x, y in reversed(breathing_lower):
            ctx.line_to(xmap(x), ymap(y))
        ctx.close_path()
        ctx.fill()
        draw_polyline(ctx, xmap, ymap, breathing_lower, colors["breathing"], 1.9)
        draw_polyline(ctx, xmap, ymap, breathing_upper, colors["breathing"], 1.9)

    for x, y in unbound:
        draw_marker(ctx, xmap(x), ymap(y), colors["unbound"])
    for x, y in locked:
        draw_marker(ctx, xmap(x), ymap(y), colors["locked"])
    for x, y in breathing_lower + breathing_upper:
        draw_marker(ctx, xmap(x), ymap(y), colors["breathing"], open_marker=True)
    for row in by_branch.get("transition", []):
        x = xmap(as_float(row, "CHM"))
        y0 = ymap(as_float(row, "d_lower"))
        y1 = ymap(as_float(row, "d_upper"))
        draw_whisker(ctx, x, y0, y1, colors["transition"], width=1.0)
        draw_marker(ctx, x, ymap(as_float(row, "d_center")), colors["transition"], 2.7, open_marker=True)

    upper_threshold = read_upper_threshold()
    if upper_threshold:
        threshold = 0.5 * (upper_threshold[0] + upper_threshold[1])
        ctx.save()
        ctx.set_source_rgb(*colors["transition"])
        ctx.set_line_width(1.0)
        ctx.set_dash((3.0, 3.0))
        ctx.move_to(xmap(threshold), margin_t)
        ctx.line_to(xmap(threshold), margin_t + plot_h)
        ctx.stroke()
        ctx.restore()
        draw_text(ctx, xmap(threshold), margin_t - 8.0, f"chi_M,c2 ~ {threshold:.5f}", 8.4, colors["transition"], align="center")

    for ytick in range(40, 141, 20):
        y = ymap(ytick)
        ctx.move_to(margin_l - 4.0, y)
        ctx.line_to(margin_l, y)
        ctx.set_source_rgb(0.08, 0.08, 0.08)
        ctx.set_line_width(0.75)
        ctx.stroke()
        draw_text(ctx, margin_l - 8.0, y + 3.8, str(ytick), 8.5, align="right")
    for xtick in (0.00, 0.03, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30):
        x = xmap(xtick)
        ctx.move_to(x, margin_t + plot_h)
        ctx.line_to(x, margin_t + plot_h + 4.0)
        ctx.set_source_rgb(0.08, 0.08, 0.08)
        ctx.set_line_width(0.75)
        ctx.stroke()
        draw_text(ctx, x, margin_t + plot_h + 18.0, f"{xtick:.2f}", 8.2, align="center")

    draw_text(ctx, width / 2.0, 24.0, "Experiment 2 phase-aligned pair-distance bifurcation", 12.0, bold=True, align="center")
    draw_text(ctx, width / 2.0, 42.0, "fixed-seed protocol; first-excursion extrema from equal elapsed-time windows", 8.8, (0.25, 0.25, 0.25), align="center")
    draw_text(ctx, margin_l + plot_w / 2.0, height - 17.0, "chi_M", 11.0, align="center")
    ctx.save()
    ctx.translate(21.0, margin_t + plot_h / 2.0)
    ctx.rotate(-math.pi / 2.0)
    draw_text(ctx, 0.0, 0.0, "inter-gel distance d (lattice)", 10.5, align="center")
    ctx.restore()

    legend_x, legend_y = margin_l + 18.0, margin_t + 22.0
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.86)
    ctx.rectangle(legend_x - 9.0, legend_y - 15.0, 218.0, 72.0)
    ctx.fill()
    legend = (
        ("unbound mean", "unbound", False),
        ("quiet locked mean", "locked", False),
        ("critical-band spread", "transition", True),
        ("phase-aligned excursion envelope", "breathing", False),
    )
    for index, (label, branch, marker_only) in enumerate(legend):
        y = legend_y + 17.0 * index
        color = colors[branch]
        if marker_only:
            draw_whisker(ctx, legend_x + 10.0, y - 7.0, y + 7.0, color)
            draw_marker(ctx, legend_x + 10.0, y, color, 2.7, open_marker=True)
        else:
            ctx.set_source_rgb(*color)
            ctx.set_line_width(1.8)
            ctx.move_to(legend_x, y)
            ctx.line_to(legend_x + 20.0, y)
            ctx.stroke()
        draw_text(ctx, legend_x + 29.0, y + 3.5, label, 8.2)

    surface.finish()
    convert_pdf(PDF_OUT)


def draw_diagnostic(rows: list[dict[str, str]]) -> None:
    supplement = [row for row in rows if row["source"] == "supplement"]
    grouped: dict[float, dict[int, dict[str, str]]] = {}
    for row in supplement:
        grouped.setdefault(as_float(row, "CHM"), {})[int(row["window_length"])] = row
    full_rows = [grouped[chm][max(grouped[chm])] for chm in sorted(grouped)]
    short_rows = [grouped[chm][min(grouped[chm])] for chm in sorted(grouped)]

    width, height = 650.0, 400.0
    margin_l, margin_r = 72.0, 28.0
    margin_t, margin_b = 58.0, 62.0
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x_min, x_max = 0.178, 0.295
    y_min, y_max = 45.0, 135.0

    def xmap(value: float) -> float:
        return margin_l + (value - x_min) / (x_max - x_min) * plot_w

    def ymap(value: float) -> float:
        return margin_t + (y_max - value) / (y_max - y_min) * plot_h

    short_color = (0.89, 0.34, 0.07)
    full_color = (0.08, 0.44, 0.55)
    surface = cairo.PDFSurface(str(DIAGNOSTIC_PDF_OUT), width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()

    ctx.set_source_rgb(0.86, 0.86, 0.86)
    ctx.set_line_width(0.45)
    for ytick in range(50, 131, 20):
        ctx.move_to(margin_l, ymap(ytick))
        ctx.line_to(margin_l + plot_w, ymap(ytick))
    for xtick in (0.18, 0.20, 0.22, 0.24, 0.26, 0.28):
        ctx.move_to(xmap(xtick), margin_t)
        ctx.line_to(xmap(xtick), margin_t + plot_h)
    ctx.stroke()
    ctx.set_source_rgb(0.08, 0.08, 0.08)
    ctx.set_line_width(1.0)
    ctx.rectangle(margin_l, margin_t, plot_w, plot_h)
    ctx.stroke()

    for data_rows, color, dash in ((short_rows, short_color, (4.0, 3.0)), (full_rows, full_color, None)):
        lower = [(as_float(row, "CHM"), as_float(row, "d_p05")) for row in data_rows]
        upper = [(as_float(row, "CHM"), as_float(row, "d_p95")) for row in data_rows]
        draw_polyline(ctx, xmap, ymap, lower, color, 1.7, dash=dash)
        draw_polyline(ctx, xmap, ymap, upper, color, 1.7, dash=dash)
        for x, y in lower + upper:
            draw_marker(ctx, xmap(x), ymap(y), color, 2.7, open_marker=True)

    for ytick in range(50, 131, 20):
        draw_text(ctx, margin_l - 8.0, ymap(ytick) + 3.8, str(ytick), 8.5, align="right")
    for xtick in (0.18, 0.20, 0.22, 0.24, 0.26, 0.28):
        draw_text(ctx, xmap(xtick), margin_t + plot_h + 18.0, f"{xtick:.2f}", 8.2, align="center")

    draw_text(ctx, width / 2.0, 24.0, "Time-window phase bias in the continuation branch", 12.0, bold=True, align="center")
    draw_text(ctx, width / 2.0, 42.0, "same trajectories; only the analysis window changes", 8.8, (0.25, 0.25, 0.25), align="center")
    draw_text(ctx, margin_l + plot_w / 2.0, height - 17.0, "chi_M", 11.0, align="center")
    ctx.save()
    ctx.translate(21.0, margin_t + plot_h / 2.0)
    ctx.rotate(-math.pi / 2.0)
    draw_text(ctx, 0.0, 0.0, "distance envelope P5/P95", 10.5, align="center")
    ctx.restore()

    legend_x, legend_y = margin_l + 18.0, margin_t + 22.0
    for index, (label, color, dash) in enumerate(
        (("last 10,000 steps", short_color, (4.0, 3.0)), ("full 39,999 post-switch steps", full_color, None))
    ):
        y = legend_y + 18.0 * index
        ctx.save()
        ctx.set_source_rgb(*color)
        ctx.set_line_width(1.7)
        if dash:
            ctx.set_dash(dash)
        ctx.move_to(legend_x, y)
        ctx.line_to(legend_x + 22.0, y)
        ctx.stroke()
        ctx.restore()
        draw_text(ctx, legend_x + 31.0, y + 3.5, label, 8.4)

    surface.finish()
    convert_pdf(DIAGNOSTIC_PDF_OUT)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not BRANCH_CSV.exists() or not SENSITIVITY_CSV.exists():
        sampling_main()
    draw_main(read_csv(BRANCH_CSV))
    draw_diagnostic(read_csv(SENSITIVITY_CSV))
    print(f"Wrote {PDF_OUT}")
    print(f"Wrote {DIAGNOSTIC_PDF_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
