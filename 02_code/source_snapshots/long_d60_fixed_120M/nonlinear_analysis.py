"""
Redraw pair-attractor figures for a single CHM run directory.

Usage:
    python3 nonlinear_analysis.py --run-dir CHM_005 --figs-dir CHM_005/figs

The per-run output format intentionally follows CHM_005/figs:
    memory_final.png, chemistry_final.png, rc_timeseries.png,
    rc_trajectory.png, speed_timeseries.png, dynamics.mp4
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


FIG_DPI = 150
U_VMIN = 0.0
U_VMAX = 0.40
GREY = "#808080"
BLUE = "#1f5fb3"
RUN_TAG = "(unparsed run tag)"


def load_required(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.loadtxt(path)


def final_time(data_dir):
    times = []
    for path in Path(data_dir).glob("ungrid*.dat"):
        m = re.match(r"ungrid(\d+)\.dat$", path.name)
        if m:
            times.append(int(m.group(1)))
    return max(times) if times else None


def square_field(path, label):
    arr = load_required(path)
    side = int(round(np.sqrt(arr.size)))
    if side * side != arr.size:
        raise ValueError(f"{label} length {arr.size} is not square")
    return arr.reshape(side, side)


def load_bodycenters(data_dir):
    b0 = load_required(Path(data_dir) / "gel0bodycenter0.dat")
    b1 = load_required(Path(data_dir) / "gel1bodycenter0.dat")
    return b0, b1


def load_motion(data_dir, ig, time):
    arr = load_required(Path(data_dir) / f"gel{ig}motion{time}.dat")
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"gel{ig}motion{time}.dat has unexpected shape {arr.shape}")
    return arr


def motion_grid(motion):
    n = int(round(np.sqrt(motion.shape[0])))
    if n * n != motion.shape[0]:
        return None
    return motion[:, 0].reshape(n, n), motion[:, 1].reshape(n, n), motion[:, 3].reshape(n, n)


def draw_u_field(ax, data_dir, time, mesh_edges=False):
    last = None
    for ig in (0, 1):
        motion = load_motion(data_dir, ig, time)
        grid = motion_grid(motion)
        if grid is None:
            last = ax.scatter(
                motion[:, 0], motion[:, 1], c=motion[:, 3], cmap="jet",
                vmin=U_VMIN, vmax=U_VMAX, marker="s", s=14, linewidths=0,
                rasterized=True, zorder=3,
            )
        else:
            xg, yg, ug = grid
            edge = (0.0, 0.0, 0.0, 0.18) if mesh_edges else "none"
            lw = 0.04 if mesh_edges else 0.0
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The input coordinates to pcolormesh.*",
                    category=UserWarning,
                )
                last = ax.pcolormesh(
                    xg, yg, ug, cmap="jet", vmin=U_VMIN, vmax=U_VMAX,
                    shading="nearest", edgecolors=edge, linewidth=lw,
                    rasterized=True, zorder=3,
                )
    return last


def draw_trajectory_overlay(ax, b0, b1, end_idx=None, final_markers=False, current_markers=False):
    if end_idx is None:
        end_idx = min(len(b0), len(b1)) - 1
    end_idx = max(0, min(end_idx, min(len(b0), len(b1)) - 1))
    sl = slice(0, end_idx + 1)

    for b in (b0, b1):
        ax.plot(b[sl, 1], b[sl, 2], color="black", lw=3.6, alpha=0.88,
                solid_capstyle="round", zorder=5)
        ax.scatter(b[0, 1], b[0, 2], marker="*", s=300, c="lime",
                   edgecolors="black", linewidths=1.5, zorder=8)
        if final_markers:
            ax.scatter(b[-1, 1], b[-1, 2], marker="o", s=150, c="red",
                       edgecolors="black", linewidths=1.0, zorder=9)
        if current_markers:
            ax.scatter(b[end_idx, 1], b[end_idx, 2], marker="o", s=58,
                       facecolors="white", edgecolors="black", linewidths=1.5,
                       zorder=9)


def setup_grid(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.22)
    ax.set_aspect("equal", adjustable="box")


def save_memory_final(data_dir, figs_dir, time, b0, b1):
    mem = square_field(Path(data_dir) / f"ungrid{time}.dat", "ungrid")
    fig = plt.figure(figsize=(11, 8), dpi=FIG_DPI)
    ax = fig.add_axes([0.232, 0.12, 0.628, 0.80])
    cax_left = fig.add_axes([0.080, 0.12, 0.032, 0.80])
    cax_right = fig.add_axes([0.890, 0.12, 0.032, 0.80])
    mem_im = ax.imshow(
        mem, origin="lower", extent=[0, 400, 0, 400], cmap="Greys",
        vmin=0.0, vmax=float(np.nanmax(mem)), zorder=1,
    )
    u_im = draw_u_field(ax, data_dir, time, mesh_edges=False)
    draw_trajectory_overlay(ax, b0, b1, final_markers=True)
    setup_grid(ax, (0, 400), (0, 400))
    ax.set_title(f"{RUN_TAG} — final memory (M) + gel chemistry (u) at t={time} TU")

    cbar_left = fig.colorbar(mem_im, cax=cax_left)
    cbar_left.set_label("M (memory)")
    cbar_left.ax.yaxis.set_label_position("left")
    cbar_left.ax.yaxis.set_ticks_position("left")
    cbar_right = fig.colorbar(u_im, cax=cax_right)
    cbar_right.set_label("u (BZ activator)")
    fig.savefig(Path(figs_dir) / "memory_final.png")
    plt.close(fig)


def save_chemistry_final(data_dir, figs_dir, time, b0, b1):
    fig = plt.figure(figsize=(8, 7), dpi=FIG_DPI)
    ax = fig.add_axes([0.134, 0.105, 0.682, 0.838])
    cax = fig.add_axes([0.850, 0.105, 0.038, 0.838])
    u_im = draw_u_field(ax, data_dir, time, mesh_edges=True)
    draw_trajectory_overlay(ax, b0, b1, final_markers=True)
    setup_grid(ax, (118, 282), (131, 309))
    ax.set_title(f"Final BZ activator (u) field — t={time} TU")
    cbar = fig.colorbar(u_im, cax=cax)
    cbar.set_label("u")
    fig.savefig(Path(figs_dir) / "chemistry_final.png")
    plt.close(fig)


def save_rc_timeseries(figs_dir, b0, b1):
    t = b0[:, 0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=FIG_DPI, sharex=True)
    fig.suptitle(f"{RUN_TAG} — Centroid components vs time")

    axes[0].plot(t, b0[:, 1], color=GREY, lw=0.9, label="ig=0")
    axes[0].plot(t, b1[:, 1], color=BLUE, lw=0.9, label="ig=1")
    axes[0].set_ylabel("Rc.x")
    axes[0].legend(loc="upper left")

    axes[1].plot(t, b0[:, 2], color=GREY, lw=0.9, label="ig=0")
    axes[1].plot(t, b1[:, 2], color=BLUE, lw=0.9, label="ig=1")
    axes[1].set_ylabel("Rc.y")
    axes[1].set_xlabel("t (TU)")
    axes[1].legend(loc="center right")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.margins(x=0.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(Path(figs_dir) / "rc_timeseries.png")
    plt.close(fig)


def save_rc_trajectory(figs_dir, b0, b1):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), dpi=FIG_DPI)
    fig.suptitle(f"{RUN_TAG} — Centroid trajectory")
    all_x = np.r_[b0[:, 1], b1[:, 1]]
    all_y = np.r_[b0[:, 2], b1[:, 2]]
    xlim = (float(np.nanmin(all_x) - 4.0), float(np.nanmax(all_x) + 4.0))
    ylim = (float(np.nanmin(all_y) - 4.0), float(np.nanmax(all_y) + 4.0))

    for ig, (ax, b) in enumerate(zip(axes, (b0, b1))):
        disp = float(np.hypot(b[-1, 1] - b[0, 1], b[-1, 2] - b[0, 2]))
        ax.plot(b[:, 1], b[:, 2], color="#222222", lw=3.4,
                solid_capstyle="round", zorder=3)
        ax.scatter(b[0, 1], b[0, 2], marker="*", s=360, c="lime",
                   edgecolors="black", linewidths=1.5, zorder=8)
        ax.scatter(b[-1, 1], b[-1, 2], marker="o", s=160, c="red",
                   edgecolors="black", linewidths=1.0, zorder=9)

        sample_times = np.arange(5000, min(120000, len(b)), 5000)
        idx = np.clip(sample_times, 0, len(b) - 1)
        ax.scatter(b[idx, 1], b[idx, 2], marker="o", s=24, c=BLUE,
                   edgecolors="white", linewidths=0.6, zorder=7)
        for tt, ii in zip(sample_times, idx):
            ax.text(b[ii, 1] + 1.2, b[ii, 2] + 1.2, f"{int(tt)}",
                    color=BLUE, fontsize=8, zorder=10)

        ax.set_title(f"ig={ig}, |ΔRc| = {disp:.2f}")
        setup_grid(ax, xlim, ylim)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(Path(figs_dir) / "rc_trajectory.png")
    plt.close(fig)


def moving_average_same(values, width=50):
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def centroid_speed(bodycenter):
    if bodycenter.shape[1] >= 7:
        speed = np.hypot(bodycenter[:, 5], bodycenter[:, 6])
    else:
        t = bodycenter[:, 0]
        dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
        speed = np.hypot(np.gradient(bodycenter[:, 1], dt), np.gradient(bodycenter[:, 2], dt))
    return moving_average_same(speed, 50)


def save_speed_timeseries(figs_dir, b0, b1):
    t = b0[:, 0]
    v0 = centroid_speed(b0)
    v1 = centroid_speed(b1)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIG_DPI)
    ax.set_title(f"{RUN_TAG} — Centroid speed")
    ax.fill_between(t, 0, v0, color=GREY, alpha=0.25, linewidth=0)
    ax.plot(t, v0, color=GREY, lw=1.6, label="ig=0 (50-pt MA)")
    ax.plot(t, v1, color=BLUE, lw=1.6, label="ig=1 (50-pt MA)")
    ax.set_xlabel("t (TU)")
    ax.set_ylabel("|v|")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax.margins(x=0.05)
    ax.set_ylim(-0.015, max(0.30, float(np.nanmax([v0.max(), v1.max()])) * 1.05))
    fig.tight_layout()
    fig.savefig(Path(figs_dir) / "speed_timeseries.png")
    plt.close(fig)


def save_dynamics(data_dir, figs_dir, folder_name, final_t, b0, b1):
    times = list(range(1000, final_t + 1, 1000))
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg writer is not available")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=FIG_DPI)
    sm = ScalarMappable(norm=Normalize(vmin=U_VMIN, vmax=U_VMAX), cmap="jet")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.047, pad=0.045)
    cbar.set_label("u")

    fig.suptitle(folder_name)

    def draw(frame_index):
        tu = times[frame_index]
        idx = min(tu - 1, len(b0) - 1, len(b1) - 1)
        if frame_index == 0 or (frame_index + 1) % 5 == 1 or frame_index == len(times) - 1:
            print(f"[anim]   frame {frame_index + 1}/{len(times)}  TU={tu}")
        ax.clear()
        draw_u_field(ax, data_dir, tu, mesh_edges=False)
        draw_trajectory_overlay(ax, b0, b1, end_idx=idx, current_markers=True)
        setup_grid(ax, (0, 400), (0, 400))
        ax.set_title(f"Gels & COM trajectory, time = {tu}")

    print(f"[anim] preparing animation ({len(times)} frames @ 3 fps)")
    print("[anim] ffmpeg available = True")
    ani = animation.FuncAnimation(fig, draw, frames=len(times), interval=333, repeat=False)
    out = Path(figs_dir) / "dynamics.mp4"
    ani.save(out, writer=animation.FFMpegWriter(fps=3, bitrate=1800), dpi=FIG_DPI)
    plt.close(fig)
    print(f"[anim] saved {out}")


def run_per_case(run_dir, figs_dir):
    run_dir = Path(run_dir).resolve()
    data_dir = run_dir / "0"
    figs_dir = Path(figs_dir).resolve()
    figs_dir.mkdir(parents=True, exist_ok=True)
    ft = final_time(data_dir)
    if ft is None:
        raise RuntimeError(f"no ungrid*.dat files found in {data_dir}")

    print("[setup] gel dims = LX=50 LY=50  (nodes 52x52, elems 51x51)")
    print(f"[setup] run_dir   = {run_dir}")
    print(f"[setup] data_dir  = {data_dir}")
    print(f"[setup] figs_dir  = {figs_dir}")
    print("[setup] params (s, κ, φ) = None")
    print("[setup] detected 2 gel(s), igs=[0, 1]")

    b0, b1 = load_bodycenters(data_dir)
    print(f"[15.0] gel0 bc shape={b0.shape}, idx range [{int(b0[:, 0].min())}, {int(b0[:, 0].max())}], unique={np.unique(b0[:, 0]).size}")
    if len(b0) > 1:
        print(f"[15.0] idx_to_TU = {np.median(np.diff(b0[:, 0])):.1f}")

    tasks = [
        ("rc_timeseries", "rc_timeseries.png", lambda: save_rc_timeseries(figs_dir, b0, b1)),
        ("rc_trajectory", "rc_trajectory.png", lambda: save_rc_trajectory(figs_dir, b0, b1)),
        ("speed_timeseries", "speed_timeseries.png", lambda: save_speed_timeseries(figs_dir, b0, b1)),
        ("chemistry_final", "chemistry_final.png", lambda: save_chemistry_final(data_dir, figs_dir, ft, b0, b1)),
        ("memory_final", "memory_final.png", lambda: save_memory_final(data_dir, figs_dir, ft, b0, b1)),
        ("dynamics", "dynamics.mp4", lambda: save_dynamics(data_dir, figs_dir, run_dir.name, ft, b0, b1)),
    ]

    failures = []
    labels = ["[a]", "[b]", "[c]", "[d]", "[e]", "[anim]"]
    for idx, (name, filename, func) in enumerate(tasks):
        print(f"{labels[idx]} {filename} ...")
        try:
            func()
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"[ERROR] {name}: {exc}", file=sys.stderr)
    if failures:
        print("[done] with failures")
        for name, msg in failures:
            print(f"FAILED {name}: {msg}", file=sys.stderr)
        return 1
    print("[done]")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--figs-dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    figs_dir = Path(args.figs_dir) if args.figs_dir else run_dir / "figs"
    raise SystemExit(run_per_case(run_dir, figs_dir))


if __name__ == "__main__":
    main()
