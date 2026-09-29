#!/usr/bin/env python3
"""Render every saved high-CHM motion frame that contains no NaN or Inf.

The visual style follows CHM_020/figs/0.2dynamics.mp4.  Once finite gel
coordinates leave the 0..400 laboratory domain, the frame remains in the
movie and carries an explicit out-of-domain annotation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


FIG_DPI = 150
U_VMIN = 0.0
U_VMAX = 0.40


@dataclass(frozen=True)
class Case:
    folder: str
    chm_label: str
    filename: str


CASES = (
    Case("CHM_035", "0.35", "0.35dynamics_all_finite.mp4"),
    Case("CHM_040", "0.40", "0.4dynamics_all_finite.mp4"),
    Case("CHM_045", "0.45", "0.45dynamics_all_finite.mp4"),
    Case("CHM_050", "0.50", "0.5dynamics_all_finite.mp4"),
)


def file_time(path: Path, gel: int) -> int | None:
    match = re.fullmatch(rf"gel{gel}motion(\d+)\.dat", path.name)
    return int(match.group(1)) if match else None


def common_motion_times(data_dir: Path) -> list[int]:
    by_gel: list[set[int]] = []
    for gel in (0, 1):
        times = {
            time
            for path in data_dir.glob(f"gel{gel}motion*.dat")
            if (time := file_time(path, gel)) is not None and time >= 1000
        }
        by_gel.append(times)
    return sorted(by_gel[0] & by_gel[1])


def load_finite_frames(data_dir: Path) -> tuple[list[int], dict[tuple[int, int], np.ndarray], list[int]]:
    included: list[int] = []
    skipped: list[int] = []
    cache: dict[tuple[int, int], np.ndarray] = {}
    for time in common_motion_times(data_dir):
        frames = []
        valid = True
        for gel in (0, 1):
            motion = np.loadtxt(data_dir / f"gel{gel}motion{time}.dat")
            if motion.shape != (2500, 5) or not np.isfinite(motion).all():
                valid = False
                break
            frames.append(motion)
        if not valid:
            skipped.append(time)
            continue
        included.append(time)
        cache[(0, time)] = frames[0]
        cache[(1, time)] = frames[1]
    if not included:
        raise RuntimeError(f"No finite motion frames in {data_dir}")
    return included, cache, skipped


def load_bodycenters(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.loadtxt(data_dir / "gel0bodycenter0.dat"),
        np.loadtxt(data_dir / "gel1bodycenter0.dat"),
    )


def in_domain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x <= 400.0) & (y >= 0.0) & (y <= 400.0)


def draw_motion(ax: plt.Axes, motion: np.ndarray):
    x = motion[:, 0]
    y = motion[:, 1]
    u = motion[:, 3]
    max_coordinate = float(np.max(np.abs(motion[:, :2])))
    if max_coordinate < 1.0e6:
        x_grid = x.reshape(50, 50)
        y_grid = y.reshape(50, 50)
        u_grid = u.reshape(50, 50)
        return ax.pcolormesh(
            x_grid,
            y_grid,
            u_grid,
            cmap="jet",
            vmin=U_VMIN,
            vmax=U_VMAX,
            shading="nearest",
            edgecolors="none",
            linewidth=0.0,
            rasterized=True,
            zorder=3,
        ), max_coordinate

    visible = in_domain(x, y)
    artist = None
    if np.any(visible):
        artist = ax.scatter(
            x[visible], y[visible], c=u[visible], cmap="jet",
            vmin=U_VMIN, vmax=U_VMAX, marker="s", s=14,
            linewidths=0, rasterized=True, zorder=3,
        )
    return artist, max_coordinate


def draw_trajectory(ax: plt.Axes, bodycenter: np.ndarray, time: int) -> None:
    count = min(time, len(bodycenter))
    history = bodycenter[:count]
    x = history[:, 1]
    y = history[:, 2]
    visible = in_domain(x, y)
    # NaN separators are a display-only clipping device.  Every finite source
    # frame remains present in the animation and is reported in the title.
    plot_x = np.where(visible, x, np.nan)
    plot_y = np.where(visible, y, np.nan)
    ax.plot(
        plot_x, plot_y, color="black", lw=3.6, alpha=0.88,
        solid_capstyle="round", zorder=5,
    )
    if in_domain(np.asarray([x[0]]), np.asarray([y[0]]))[0]:
        ax.scatter(
            x[0], y[0], marker="*", s=300, c="lime",
            edgecolors="black", linewidths=1.5, zorder=8,
        )
    if visible[-1]:
        ax.scatter(
            x[-1], y[-1], marker="o", s=58, facecolors="white",
            edgecolors="black", linewidths=1.5, zorder=9,
        )


def setup_grid(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, 400.0)
    ax.set_ylim(0.0, 400.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.22)
    ax.set_aspect("equal", adjustable="box")


def render_case(base: Path, case: Case) -> dict[str, object]:
    case_dir = base / case.folder
    data_dir = case_dir / "0"
    figs_dir = case_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    times, frame_cache, skipped = load_finite_frames(data_dir)
    b0, b1 = load_bodycenters(data_dir)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIG_DPI)
    scalar = ScalarMappable(norm=Normalize(vmin=U_VMIN, vmax=U_VMAX), cmap="jet")
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.047, pad=0.045)
    colorbar.set_label("u")
    fig.suptitle(f"{case.folder} — all finite saved frames")

    def draw(frame_index: int) -> None:
        time = times[frame_index]
        ax.clear()
        max_coordinates = []
        for gel in (0, 1):
            _, max_coordinate = draw_motion(ax, frame_cache[(gel, time)])
            max_coordinates.append(max_coordinate)
        draw_trajectory(ax, b0, time)
        draw_trajectory(ax, b1, time)
        setup_grid(ax)
        ax.set_title(f"Gels & COM trajectory, time = {time}")
        max_coordinate = max(max_coordinates)
        if max_coordinate > 400.0:
            ax.text(
                0.5,
                0.025,
                f"finite values; gel coordinates outside 0–400 domain  (max |r| = {max_coordinate:.2e})",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                color="crimson",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "crimson", "alpha": 0.88, "pad": 3},
                zorder=20,
            )

    output = figs_dir / case.filename
    movie = animation.FuncAnimation(fig, draw, frames=len(times), interval=333, repeat=False)
    movie.save(output, writer=animation.FFMpegWriter(fps=3, bitrate=1800), dpi=FIG_DPI)
    plt.close(fig)
    return {
        "case": case.folder,
        "output": str(output.resolve()),
        "first_tu": times[0],
        "last_tu": times[-1],
        "frames": len(times),
        "skipped_nonfinite": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument(
        "--folders",
        nargs="+",
        choices=[case.folder for case in CASES],
        default=[case.folder for case in CASES],
        help="case folders to render (default: all)",
    )
    args = parser.parse_args()
    base = args.base.resolve()
    selected_cases = [case for case in CASES if case.folder in args.folders]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(render_case, base, case) for case in selected_cases]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
