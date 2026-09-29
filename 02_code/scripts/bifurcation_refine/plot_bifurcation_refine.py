#!/usr/bin/env python3
"""Draw the refined Experiment 2 chi_M-distance bifurcation diagram."""

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

from analyze_bifurcation_refine import ESTIMATE_CSV, METRICS_CSV, main as analyze_main  # noqa: E402
from bifurcation_common import OUT  # noqa: E402


PDF_OUT = OUT / "chiM_and_d_bifurcation_refined.pdf"


def read_rows() -> list[dict[str, str]]:
    if not METRICS_CSV.exists():
        analyze_main()
    with METRICS_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_estimates() -> dict[str, tuple[float, float]]:
    estimates: dict[str, tuple[float, float]] = {}
    if not ESTIMATE_CSV.exists():
        return estimates
    with ESTIMATE_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                estimates[row["transition"]] = (float(row["lo_locked_or_unbound"]), float(row["hi_locked_or_nonlinear"]))
            except (KeyError, ValueError):
                continue
    return estimates


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
    radius: float,
    open_marker: bool,
) -> None:
    ctx.save()
    ctx.new_path()
    ctx.set_source_rgb(*color)
    if open_marker:
        ctx.set_line_width(1.25)
        ctx.arc(x, y, radius, 0, 2 * math.pi)
        ctx.stroke()
    else:
        ctx.arc(x, y, radius, 0, 2 * math.pi)
        ctx.fill()
    ctx.new_path()
    ctx.restore()


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return math.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_rows() if as_float(row, "CHM") <= 0.305]
    estimates = read_estimates()

    width, height = 640.0, 420.0
    margin_l, margin_r = 78.0, 28.0
    margin_t, margin_b = 58.0, 66.0
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    x_min, x_max = -0.002, 0.305
    y_min, y_max = 35.0, 225.0

    def xmap(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w

    def ymap(y: float) -> float:
        return margin_t + (y_max - y) / (y_max - y_min) * plot_h

    surface = cairo.PDFSurface(str(PDF_OUT), width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.paint()

    lower = estimates.get("capture_lower")
    upper = estimates.get("locked_to_nonlinear_upper")
    if lower:
        ctx.set_source_rgba(0.2, 0.35, 0.55, 0.08)
        ctx.rectangle(xmap(x_min), margin_t, xmap(lower[1]) - xmap(x_min), plot_h)
        ctx.fill()
    if upper:
        ctx.set_source_rgba(0.93, 0.44, 0.12, 0.11)
        ctx.rectangle(xmap(upper[0]), margin_t, xmap(x_max) - xmap(upper[0]), plot_h)
        ctx.fill()

    ctx.set_line_width(0.45)
    ctx.set_source_rgb(0.86, 0.86, 0.86)
    for yt in range(40, 221, 20):
        ctx.move_to(margin_l, ymap(yt))
        ctx.line_to(margin_l + plot_w, ymap(yt))
    for xt in [0.0, 0.03, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30]:
        ctx.move_to(xmap(xt), margin_t)
        ctx.line_to(xmap(xt), margin_t + plot_h)
    ctx.stroke()

    ctx.set_source_rgb(0.08, 0.08, 0.08)
    ctx.set_line_width(1.05)
    ctx.rectangle(margin_l, margin_t, plot_w, plot_h)
    ctx.stroke()

    colors = {
        "unbound": (0.35, 0.39, 0.44),
        "locked": (0.02, 0.02, 0.02),
        "nonlinear": (0.89, 0.34, 0.07),
        "transition": (0.45, 0.25, 0.65),
    }

    by_state: dict[str, list[dict[str, str]]] = {"unbound": [], "locked": [], "nonlinear": [], "transition": []}
    for row in rows:
        by_state.setdefault(row["state"], []).append(row)
    for state_rows in by_state.values():
        state_rows.sort(key=lambda row: as_float(row, "CHM"))

    unbound_points = [(as_float(row, "CHM"), as_float(row, "d_mean")) for row in by_state["unbound"]]
    locked_points = [(as_float(row, "CHM"), as_float(row, "d_mean")) for row in by_state["locked"]]
    nonlinear_min = [(as_float(row, "CHM"), as_float(row, "d_p05")) for row in by_state["nonlinear"]]
    nonlinear_max = [(as_float(row, "CHM"), as_float(row, "d_p95")) for row in by_state["nonlinear"]]

    draw_polyline(ctx, xmap, ymap, unbound_points, colors["unbound"], 1.4, dash=(4.0, 3.0))
    draw_polyline(ctx, xmap, ymap, locked_points, colors["locked"], 2.0)

    if nonlinear_min and nonlinear_max:
        ctx.set_source_rgba(0.93, 0.44, 0.12, 0.18)
        ctx.move_to(xmap(nonlinear_max[0][0]), ymap(nonlinear_max[0][1]))
        for x, y in nonlinear_max[1:]:
            ctx.line_to(xmap(x), ymap(y))
        for x, y in reversed(nonlinear_min):
            ctx.line_to(xmap(x), ymap(y))
        ctx.close_path()
        ctx.fill()
        draw_polyline(ctx, xmap, ymap, nonlinear_max, colors["nonlinear"], 1.9)
        draw_polyline(ctx, xmap, ymap, nonlinear_min, colors["nonlinear"], 1.9)

    for row in rows:
        chm = as_float(row, "CHM")
        state = row["state"]
        color = colors.get(state, colors["transition"])
        open_marker = row["source"] in {"refine", "continuation", "supplement"}
        if state == "nonlinear":
            draw_marker(ctx, xmap(chm), ymap(as_float(row, "d_p05")), color, 3.2, open_marker)
            draw_marker(ctx, xmap(chm), ymap(as_float(row, "d_p95")), color, 3.2, open_marker)
        else:
            draw_marker(ctx, xmap(chm), ymap(as_float(row, "d_mean")), color, 3.2, open_marker)

    for yt in range(40, 221, 20):
        y = ymap(yt)
        ctx.set_source_rgb(0.08, 0.08, 0.08)
        ctx.set_line_width(0.75)
        ctx.move_to(margin_l - 4, y)
        ctx.line_to(margin_l, y)
        ctx.stroke()
        draw_text(ctx, margin_l - 8, y + 3.8, str(yt), 8.5, align="right")

    for xt in [0.00, 0.03, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30]:
        x = xmap(xt)
        ctx.set_source_rgb(0.08, 0.08, 0.08)
        ctx.set_line_width(0.75)
        ctx.move_to(x, margin_t + plot_h)
        ctx.line_to(x, margin_t + plot_h + 4)
        ctx.stroke()
        draw_text(ctx, x, margin_t + plot_h + 18, f"{xt:.2f}", 8.0, align="center")

    if upper:
        mid = 0.5 * (upper[0] + upper[1])
        ctx.save()
        ctx.set_source_rgb(0.75, 0.18, 0.02)
        ctx.set_line_width(1.0)
        ctx.set_dash((3.0, 3.0))
        ctx.move_to(xmap(mid), margin_t)
        ctx.line_to(xmap(mid), margin_t + plot_h)
        ctx.stroke()
        ctx.restore()
        draw_text(ctx, xmap(mid), margin_t - 8, f"chi_M,c2~{mid:.4f}", 8.3, colors["nonlinear"], align="center")
    if lower:
        mid = 0.5 * (lower[0] + lower[1])
        ctx.save()
        ctx.set_source_rgb(*colors["unbound"])
        ctx.set_line_width(1.0)
        ctx.set_dash((3.0, 3.0))
        ctx.move_to(xmap(mid), margin_t)
        ctx.line_to(xmap(mid), margin_t + plot_h)
        ctx.stroke()
        ctx.restore()
        draw_text(ctx, xmap(mid), margin_t - 22, f"chi_M,c1~{mid:.4f}", 8.3, colors["unbound"], align="center")

    draw_text(ctx, width / 2.0, 24.0, "Experiment 2 refined pair-distance bifurcation", 12.0, bold=True, align="center")
    draw_text(ctx, width / 2.0, 41.0, "d_ini = 60; filled markers baseline, open markers refinement", 8.7, (0.25, 0.25, 0.25), align="center")
    draw_text(ctx, margin_l + plot_w / 2.0, height - 18.0, "chi_M", 11.0, align="center")
    ctx.save()
    ctx.translate(22.0, margin_t + plot_h / 2.0)
    ctx.rotate(-math.pi / 2.0)
    draw_text(ctx, 0.0, 0.0, "inter-gel distance d (lattice)", 10.5, align="center")
    ctx.restore()

    lx, ly = margin_l + 22.0, margin_t + 23.0
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.82)
    ctx.rectangle(lx - 9.0, ly - 15.0, 206.0, 72.0)
    ctx.fill()
    legend = [
        ("unbound mean", "unbound"),
        ("quiet locked mean", "locked"),
        ("nonlinear P5/P95 envelope", "nonlinear"),
        ("new runs are open markers", "transition"),
    ]
    for idx, (label, state) in enumerate(legend):
        yy = ly + idx * 17.0
        color = colors[state]
        if state == "transition":
            draw_marker(ctx, lx + 9.0, yy - 2.5, color, 3.2, True)
        else:
            ctx.set_source_rgb(*color)
            ctx.set_line_width(1.8)
            ctx.move_to(lx, yy)
            ctx.line_to(lx + 20.0, yy)
            ctx.stroke()
        draw_text(ctx, lx + 29.0, yy + 3.5, label, 8.2)

    surface.finish()
    print(f"Wrote {PDF_OUT}")
    png_prefix = PDF_OUT.with_suffix("")
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", "180", str(PDF_OUT), str(png_prefix)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
