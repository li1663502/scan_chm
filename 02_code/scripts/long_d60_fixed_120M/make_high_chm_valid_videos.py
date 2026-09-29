#!/usr/bin/env python3
"""Render pre-divergence high-CHM videos in the CHM=0.20 video style."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Case:
    folder: str
    label: str
    final_frame_tu: int
    filename: str


CASES = (
    Case("CHM_035", "CHM_035", 23000, "0.35dynamics.mp4"),
    Case("CHM_040", "CHM_040", 13000, "0.4dynamics.mp4"),
)


def load_analysis(source: Path):
    spec = importlib.util.spec_from_file_location("high_chm_nonlinear_analysis", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_motion_frames(data_dir: Path, final_tu: int) -> None:
    for tu in range(1000, final_tu + 1, 1000):
        for gel in (0, 1):
            path = data_dir / f"gel{gel}motion{tu}.dat"
            motion = np.loadtxt(path)
            if motion.shape != (2500, 5):
                raise ValueError(f"Unexpected shape {motion.shape}: {path}")
            if not np.isfinite(motion).all():
                raise ValueError(f"Non-finite motion frame: {path}")
            if np.max(np.abs(motion[:, :2])) > 1.0e6:
                raise ValueError(f"Diverged position in motion frame: {path}")


def render_case(base: Path, case: Case) -> str:
    case_dir = base / case.folder
    data_dir = case_dir / "0"
    figs_dir = case_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    verify_motion_frames(data_dir, case.final_frame_tu)

    source = base / "source_snapshot" / "nonlinear_analysis.py"
    analysis = load_analysis(source)
    b0, b1 = analysis.load_bodycenters(data_dir)
    # save_dynamics indexes centroid history by TU-1.  Keep exactly the finite
    # rows needed by the final frame and omit the first corrupted boundary row.
    if len(b0) < case.final_frame_tu or len(b1) < case.final_frame_tu:
        raise ValueError(f"Insufficient centroid history for {case.folder}")
    b0 = b0[: case.final_frame_tu]
    b1 = b1[: case.final_frame_tu]
    if not np.isfinite(b0).all() or not np.isfinite(b1).all():
        raise ValueError(f"Non-finite centroid history before final frame for {case.folder}")
    if np.max(np.abs(b0[:, 1:3])) > 1.0e6 or np.max(np.abs(b1[:, 1:3])) > 1.0e6:
        raise ValueError(f"Diverged centroid history before final frame for {case.folder}")

    temp_dir = Path(tempfile.mkdtemp(prefix=".video-build-", dir=figs_dir))
    try:
        analysis.save_dynamics(
            data_dir,
            temp_dir,
            case.label,
            case.final_frame_tu,
            b0,
            b1,
        )
        source_video = temp_dir / "dynamics.mp4"
        output = figs_dir / case.filename
        os.replace(source_video, output)
    finally:
        temp_dir.rmdir()
    return str(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(render_case, base, case) for case in CASES]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
