#!/usr/bin/env python3
"""Plot only breathing-state centroid-distance extrema from existing data."""

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
from analyze_sampling_sensitivity import BRANCH_CSV, main as sampling_main  # noqa: E402
from bifurcation_common import OUT  # noqa: E402
from plot_bifurcation_existing_data import (  # noqa: E402
    POINTS_CSV,
    draw_circle,
    draw_polyline,
    draw_text,
    draw_triangle,
    read_csv,
)


PDF_OUT = OUT / "chiM_breathing_centroid_extrema.pdf"
SELECTED_CSV = OUT / "breathing_centroid_extrema_selected.csv"


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return math.nan


def read_threshold() -> float | None:
    if not ESTIMATE_CSV.exists():
        return None
    for row in read_csv(ESTIMATE_CSV):
        if row.get("transition") == "locked_to_nonlinear_upper":
            return 0.5 * (float(row["lo_locked_or_unbound"]) + float(row["hi_locked_or_nonlinear"]))
    return None


def selected_rows() -> list[dict[str, str]]:
    if not BRANCH_CSV.exists() or not POINTS_CSV.exists():
        sampling_main()
    branch = read_csv(BRANCH_CSV)
    points = read_csv(POINTS_CSV)
    selected: list[dict[str, str]] = []
    for row in branch:
        if row["branch"] != "locked":
            continue
        selected.append(
            {
                "role": "locked_branch",
                "source": row["source"],
                "case": row["case"],
                "CHM": row["CHM"],
                "d_min": row["d_center"],
                "d_max": row["d_center"],
                "t_min": "",
                "t_max": "",
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "selection": "late-time locked mean",
            }
        )
    for row in branch:
        if row["branch"] != "breathing":
            continue
        selected.append(
            {
                "role": "connected_branch",
                "source": row["source"],
                "case": row["case"],
                "CHM": row["CHM"],
                "d_min": row["d_lower"],
                "d_max": row["d_upper"],
                "t_min": row["t_lower"],
                "t_max": row["t_upper"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "selection": "complete-cycle anchor" if row["source"] == "baseline" else "phase-aligned first excursion",
            }
        )

    complete = [row for row in points if row["role"] == "complete_cycle_extrema"]
    row_030 = next(row for row in complete if abs(as_float(row, "CHM") - 0.30) < 1e-9)
    selected.append(
        {
            "role": "connected_branch",
            "source": row_030["source"],
            "case": row_030["case"],
            "CHM": row_030["CHM"],
            "d_min": row_030["d_lower"],
            "d_max": row_030["d_upper"],
            "t_min": "138371",
            "t_max": "172075",
            "window_start": row_030["window_start"],
            "window_end": row_030["window_end"],
            "selection": "complete-cycle anchor",
        }
    )

    row_020 = next(row for row in complete if abs(as_float(row, "CHM") - 0.20) < 1e-9)
    selected.append(
        {
            "role": "independent_validation",
            "source": row_020["source"],
            "case": row_020["case"],
            "CHM": row_020["CHM"],
            "d_min": row_020["d_lower"],
            "d_max": row_020["d_upper"],
            "t_min": "175648",
            "t_max": "145237",
            "window_start": row_020["window_start"],
            "window_end": row_020["window_end"],
            "selection": "independent complete-cycle validation; not connected",
        }
    )
    return sorted(selected, key=lambda row: (as_float(row, "CHM"), row["role"]))


def write_selected(rows: list[dict[str, str]]) -> None:
    with SELECTED_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(rows: list[dict[str, str]]) -> None:
    locked = [row for row in rows if row["role"] == "locked_branch"]
    connected = [row for row in rows if row["role"] == "connected_branch"]
    validation = [row for row in rows if row["role"] == "independent_validation"]
    locked.sort(key=lambda row: as_float(row, "CHM"))
    connected.sort(key=lambda row: as_float(row, "CHM"))

    width, height = 680.0, 445.0
    left, right = 76.0, 28.0
    top, bottom = 67.0, 62.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = 0.045, 0.305
    y_min, y_max = 40.0, 145.0
    orange = (0.89, 0.34, 0.07)
    purple = (0.45, 0.25, 0.65)
    black = (0.02, 0.02, 0.02)

    def xmap(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def ymap(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lower = [(as_float(row, "CHM"), as_float(row, "d_min")) for row in connected]
    upper = [(as_float(row, "CHM"), as_float(row, "d_max")) for row in connected]

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
        if row["selection"] == "complete-cycle anchor":
            draw_triangle(ctx, x, ymap(as_float(row, "d_max")), orange, up=True)
            draw_triangle(ctx, x, ymap(as_float(row, "d_min")), orange, up=False)
        else:
            draw_circle(ctx, x, ymap(as_float(row, "d_max")), orange, 3.0, open_marker=True)
            draw_circle(ctx, x, ymap(as_float(row, "d_min")), orange, 3.0, open_marker=True)

    for row in validation:
        x = xmap(as_float(row, "CHM"))
        draw_triangle(ctx, x, ymap(as_float(row, "d_max")), purple, up=True, size=4.3)
        draw_triangle(ctx, x, ymap(as_float(row, "d_min")), purple, up=False, size=4.3)

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
        draw_text(ctx, xmap(threshold), top - 7.0, f"chi_M,c2 ~ {threshold:.5f}", 8.3, purple, align="center")

    for ytick in (40, 60, 80, 100, 120, 140):
        draw_text(ctx, left - 8.0, ymap(ytick) + 3.7, str(ytick), 8.5, align="right")
    for xtick in (0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30):
        draw_text(ctx, xmap(xtick), top + plot_height + 18.0, f"{xtick:.2f}", 8.2, align="center")

    draw_text(ctx, width / 2.0, 25.0, "Locked-to-breathing centroid-distance bifurcation", 12.5, bold=True, align="center")
    draw_text(ctx, width / 2.0, 43.0, "locked mean followed by phase-aligned breathing extrema", 8.7, (0.25, 0.25, 0.25), align="center")
    draw_text(ctx, left + plot_width / 2.0, height - 17.0, "chi_M", 10.8, align="center")
    ctx.save()
    ctx.translate(21.0, top + plot_height / 2.0)
    ctx.rotate(-math.pi / 2.0)
    draw_text(ctx, 0.0, 0.0, "centroid distance d", 10.4, align="center")
    ctx.restore()

    draw_text(ctx, xmap(0.292), ymap(131.0), "d_max", 8.5, orange, align="right")
    draw_text(ctx, xmap(0.292), ymap(58.0), "d_min", 8.5, orange, align="right")
    draw_text(ctx, xmap(0.145), ymap(61.0), "locked", 8.5, black, align="center")

    legend_x, legend_y = left + 20.0, top + 22.0
    ctx.set_source_rgba(1.0, 1.0, 1.0, 0.86)
    ctx.rectangle(legend_x - 9.0, legend_y - 15.0, 224.0, 89.0)
    ctx.fill()
    legend = (
        ("locked mean", black, None),
        ("maximum branch", orange, None),
        ("minimum branch", orange, (5.0, 3.0)),
        ("open circles: first-excursion extrema", orange, None),
        ("purple triangles: independent cycle check", purple, None),
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
    rows = selected_rows()
    write_selected(rows)
    draw(rows)
    print(f"Wrote {SELECTED_CSV}")
    print(f"Wrote {PDF_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
