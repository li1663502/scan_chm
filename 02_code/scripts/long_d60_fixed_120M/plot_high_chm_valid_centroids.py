#!/usr/bin/env python3
"""Plot the finite, pre-divergence centroid trajectories for high-CHM runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np


CASES = {
    "CHM_035": 0.35,
    "CHM_040": 0.40,
}
COLORS = ("#1764b0", "#e07a1f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    return parser.parse_args()


def physical_rows(bodycenter: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(bodycenter[:, 1])
        & np.isfinite(bodycenter[:, 2])
        & (bodycenter[:, 1] >= 0.0)
        & (bodycenter[:, 1] <= 400.0)
        & (bodycenter[:, 2] >= 0.0)
        & (bodycenter[:, 2] <= 400.0)
    )


def first_sustained_invalid(mask: np.ndarray, width: int = 20) -> int:
    invalid = (~mask).astype(np.int32)
    hits = np.convolve(invalid, np.ones(width, dtype=np.int32), mode="valid")
    starts = np.flatnonzero(hits == width)
    return int(starts[0]) if starts.size else len(mask)


def load_valid_pair(data_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    b0 = np.loadtxt(data_dir / "gel0bodycenter0.dat")
    b1 = np.loadtxt(data_dir / "gel1bodycenter0.dat")
    count = min(len(b0), len(b1))
    b0 = b0[:count]
    b1 = b1[:count]
    common_physical = physical_rows(b0) & physical_rows(b1)
    stop = first_sustained_invalid(common_physical)
    keep = common_physical[:stop]
    return b0[:stop][keep], b1[:stop][keep], stop


def colored_line(
    ax: plt.Axes,
    bodycenter: np.ndarray,
    norm: Normalize,
    color_values: np.ndarray | None = None,
) -> LineCollection:
    points = bodycenter[:, 1:3]
    segments = np.stack((points[:-1], points[1:]), axis=1)
    line = LineCollection(segments, cmap="viridis", norm=norm, linewidth=1.45)
    if color_values is None:
        color_values = bodycenter[:, 0]
    line.set_array(color_values[:-1])
    ax.add_collection(line)
    return line


def padded_limits(pair: tuple[np.ndarray, np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.concatenate([pair[0][:, 1], pair[1][:, 1]])
    y = np.concatenate([pair[0][:, 2], pair[1][:, 2]])
    xpad = max(4.0, 0.07 * float(np.ptp(x)))
    ypad = max(4.0, 0.07 * float(np.ptp(y)))
    return (float(x.min() - xpad), float(x.max() + xpad)), (float(y.min() - ypad), float(y.max() + ypad))


def style_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("laboratory x")
    ax.set_ylabel("laboratory y")
    ax.grid(alpha=0.22)


def draw_panel(
    ax: plt.Axes,
    bodycenter: np.ndarray,
    gel: int,
    norm: Normalize,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    normalized_time: bool = False,
) -> LineCollection:
    color_values = None
    if normalized_time:
        color_values = bodycenter[:, 0] / float(bodycenter[-1, 0])
    line = colored_line(ax, bodycenter, norm, color_values)
    ax.scatter(
        bodycenter[0, 1], bodycenter[0, 2], marker="*", s=230,
        c="#50e050", edgecolors="black", linewidths=1.2, zorder=5,
    )
    ax.scatter(
        bodycenter[-1, 1], bodycenter[-1, 2], marker="o", s=85,
        c="#e53935", edgecolors="black", linewidths=1.0, zorder=5,
    )
    displacement = float(np.hypot(*(bodycenter[-1, 1:3] - bodycenter[0, 1:3])))
    ax.set_title(f"Gel {gel}  |ΔRc|={displacement:.2f}")
    style_axis(ax, xlim, ylim)
    return line


def save_case_figure(case_dir: Path, chm: float, pair: tuple[np.ndarray, np.ndarray]) -> Path:
    figs = case_dir / "figs"
    figs.mkdir(exist_ok=True)
    final_tu = int(min(pair[0][-1, 0], pair[1][-1, 0]))
    norm = Normalize(0.0, float(final_tu))
    xlim, ylim = padded_limits(pair)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), dpi=170)
    line = None
    for gel, (ax, bodycenter) in enumerate(zip(axes, pair)):
        line = draw_panel(ax, bodycenter, gel, norm, xlim, ylim)
    fig.suptitle(f"CHM={chm:.2f}, d_ini=60 — valid centroid motion (0–{final_tu} TU)")
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.11, top=0.86, wspace=0.22)
    color_ax = fig.add_axes([0.935, 0.14, 0.018, 0.67])
    colorbar = fig.colorbar(line, cax=color_ax)
    colorbar.set_label("time (TU)")
    output = figs / "centroid_motion_valid.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def save_comparison(
    base: Path,
    loaded: dict[str, tuple[float, tuple[np.ndarray, np.ndarray]]],
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5), dpi=180)
    line = None
    for row, (case, (chm, pair)) in enumerate(loaded.items()):
        final_tu = int(min(pair[0][-1, 0], pair[1][-1, 0]))
        norm = Normalize(0.0, 1.0)
        xlim, ylim = padded_limits(pair)
        for gel in (0, 1):
            line = draw_panel(
                axes[row, gel], pair[gel], gel, norm, xlim, ylim,
                normalized_time=True,
            )
            axes[row, gel].set_title(f"CHM={chm:.2f} · Gel {gel} · 0–{final_tu} TU")
    fig.suptitle("High-CHM gel centroid motion before numerical divergence", fontsize=15)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.07, top=0.92, hspace=0.28, wspace=0.24)
    color_ax = fig.add_axes([0.93, 0.12, 0.017, 0.76])
    colorbar = fig.colorbar(line, cax=color_ax)
    colorbar.set_label("normalized time within each valid interval")
    output = base / "analysis_outputs" / "chm035_chm040_centroid_motion_valid.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def compact_points(bodycenter: np.ndarray, max_points: int = 900) -> list[dict[str, float | int]]:
    stride = max(1, int(np.ceil(len(bodycenter) / max_points)))
    sampled = bodycenter[::stride]
    if sampled[-1, 0] != bodycenter[-1, 0]:
        sampled = np.vstack((sampled, bodycenter[-1]))
    return [
        {"t": int(row[0]), "x": round(float(row[1]), 4), "y": round(float(row[2]), 4)}
        for row in sampled
    ]


def main() -> None:
    args = parse_args()
    loaded: dict[str, tuple[float, tuple[np.ndarray, np.ndarray]]] = {}
    payload = {"cases": []}
    for case, chm in CASES.items():
        case_dir = args.base / case
        b0, b1, divergence_index = load_valid_pair(case_dir / "0")
        pair = (b0, b1)
        loaded[case] = (chm, pair)
        output = save_case_figure(case_dir, chm, pair)
        final_tu = int(min(b0[-1, 0], b1[-1, 0]))
        payload["cases"].append(
            {
                "case": case,
                "chm": chm,
                "final_tu": final_tu,
                "divergence_index": divergence_index,
                "image": str(output),
                "gels": [compact_points(b0), compact_points(b1)],
            }
        )
    comparison = save_comparison(args.base, loaded)
    payload["comparison_image"] = str(comparison)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, separators=(",", ":")))
    print(json.dumps({"comparison": str(comparison), "cases": payload["cases"]}, indent=2))


if __name__ == "__main__":
    main()
