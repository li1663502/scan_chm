#!/usr/bin/env python3
"""Plot CHM=0.30 gel-centroid and spiral-tip trajectories.

The spiral tip is reconstructed as the unique phase singularity of
arg[(u-u_ref) + i(v-v_ref)] on the deforming material grid.  A bilinear
Newton solve provides sub-cell coordinates for each saved chemistry frame.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--html-output", type=Path)
    return parser.parse_args()


def motion_time(path: Path) -> int:
    match = re.search(r"motion(\d+)\.dat$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse motion time from {path}")
    return int(match.group(1))


def phase_reference(data_dir: Path) -> tuple[float, float]:
    refs = []
    for gel in (0, 1):
        motion = np.loadtxt(data_dir / f"gel{gel}motion0.dat")
        refs.append((float(np.median(motion[:, 3])), float(np.median(motion[:, 2]))))
    # Rounding moves the phase origin infinitesimally away from the large set of
    # nodes that are exactly at the initialized steady state in the t=0 file.
    # This removes a degenerate 46-cell winding detection while preserving the
    # unique physical singularity in every saved frame.
    return round(float(np.mean([r[0] for r in refs])), 4), round(
        float(np.mean([r[1] for r in refs])), 4
    )


def bilinear_value(values: np.ndarray, x: float, y: float) -> float:
    cross = values[3] - values[1] - values[2] + values[0]
    return float(
        values[0]
        + (values[1] - values[0]) * x
        + (values[2] - values[0]) * y
        + cross * x * y
    )


def solve_tip_in_cell(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    row: int,
    col: int,
    u_ref: float,
    v_ref: float,
) -> tuple[float, float, float, float, float]:
    def corners(field: np.ndarray) -> np.ndarray:
        return np.array(
            [
                field[row, col],
                field[row, col + 1],
                field[row + 1, col],
                field[row + 1, col + 1],
            ],
            dtype=float,
        )

    uf = corners(u_grid) - u_ref
    vf = corners(v_grid) - v_ref
    local_x = 0.5
    local_y = 0.5
    for _ in range(12):
        def derivatives(values: np.ndarray) -> tuple[float, float]:
            cross = values[3] - values[1] - values[2] + values[0]
            dx = values[1] - values[0] + cross * local_y
            dy = values[2] - values[0] + cross * local_x
            return float(dx), float(dy)

        residual = np.array(
            [
                bilinear_value(uf, local_x, local_y),
                bilinear_value(vf, local_x, local_y),
            ]
        )
        du_dx, du_dy = derivatives(uf)
        dv_dx, dv_dy = derivatives(vf)
        jacobian = np.array([[du_dx, du_dy], [dv_dx, dv_dy]])
        try:
            step = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            break
        local_x, local_y = np.clip(
            np.array([local_x, local_y]) - step,
            0.0,
            1.0,
        )

    physical_x = bilinear_value(corners(x_grid), local_x, local_y)
    physical_y = bilinear_value(corners(y_grid), local_x, local_y)
    residual = float(
        np.hypot(
            bilinear_value(uf, local_x, local_y),
            bilinear_value(vf, local_x, local_y),
        )
    )
    return physical_x, physical_y, col + local_x, row + local_y, residual


def extract_phase_tips(
    data_dir: Path,
    gel: int,
    u_ref: float,
    v_ref: float,
) -> tuple[np.ndarray, dict[int, int]]:
    files = sorted(data_dir.glob(f"gel{gel}motion*.dat"), key=motion_time)
    rows = []
    count_histogram: dict[int, int] = {}
    previous_material: np.ndarray | None = None

    for path in files:
        time = motion_time(path)
        motion = np.loadtxt(path)
        if motion.shape != (2500, 5):
            raise ValueError(f"Unexpected motion shape in {path}: {motion.shape}")

        x_grid = motion[:, 0].reshape(50, 50)
        y_grid = motion[:, 1].reshape(50, 50)
        v_grid = motion[:, 2].reshape(50, 50)
        u_grid = motion[:, 3].reshape(50, 50)
        phase = np.arctan2(v_grid - v_ref, u_grid - u_ref)
        vertices = [
            phase[:-1, :-1],
            phase[:-1, 1:],
            phase[1:, 1:],
            phase[1:, :-1],
            phase[:-1, :-1],
        ]
        winding = np.zeros((49, 49), dtype=float)
        for first, second in zip(vertices[:-1], vertices[1:]):
            winding += (second - first + np.pi) % (2.0 * np.pi) - np.pi
        candidates = np.argwhere(np.abs(winding) > np.pi)
        count_histogram[len(candidates)] = count_histogram.get(len(candidates), 0) + 1
        if len(candidates) == 0:
            rows.append([time, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue

        solved = []
        for row, col in candidates:
            tip = solve_tip_in_cell(
                x_grid,
                y_grid,
                u_grid,
                v_grid,
                int(row),
                int(col),
                u_ref,
                v_ref,
            )
            solved.append(tip)
        if previous_material is None or len(solved) == 1:
            chosen = solved[0]
        else:
            distances = [
                (candidate[2] - previous_material[0]) ** 2
                + (candidate[3] - previous_material[1]) ** 2
                for candidate in solved
            ]
            chosen = solved[int(np.argmin(distances))]
        previous_material = np.array(chosen[2:4])
        rows.append([time, *chosen])

    return np.asarray(rows, dtype=float), count_histogram


def colored_trajectory(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    time: np.ndarray,
    norm: Normalize,
    stride: int = 1,
    linewidth: float = 1.0,
) -> LineCollection:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(time)
    x = x[valid][::stride]
    y = y[valid][::stride]
    time = time[valid][::stride]
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=norm,
        linewidth=linewidth,
        rasterized=True,
    )
    collection.set_array(time[:-1])
    ax.add_collection(collection)
    ax.update_datalim(points)
    ax.autoscale_view()
    ax.scatter(x[0], y[0], marker="*", s=150, c="#55dd55", edgecolors="black", zorder=5)
    ax.scatter(x[-1], y[-1], marker="o", s=65, c="#e53935", edgecolors="black", zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    return collection


def save_centroid_figure(
    bodycenter: np.ndarray,
    gel: int,
    output_dir: Path,
) -> None:
    time = bodycenter[:, 0]
    norm = Normalize(float(time.min()), float(time.max()))
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    line = colored_trajectory(
        ax,
        bodycenter[:, 1],
        bodycenter[:, 2],
        time,
        norm,
        stride=10,
        linewidth=1.25,
    )
    ax.set_title(f"CHM=0.30 — Gel {gel} centroid trajectory")
    ax.set_xlabel("laboratory x")
    ax.set_ylabel("laboratory y")
    colorbar = fig.colorbar(line, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("time (TU)")
    fig.tight_layout()
    stem = output_dir / f"chm030_gel{gel}_centroid_trajectory"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_tip_figure(
    bodycenter: np.ndarray,
    tips: np.ndarray,
    gel: int,
    output_dir: Path,
) -> None:
    time = tips[:, 0]
    indices = np.minimum(time.astype(int), len(bodycenter) - 1)
    relative_x = tips[:, 1] - bodycenter[indices, 1]
    relative_y = tips[:, 2] - bodycenter[indices, 2]
    norm = Normalize(float(np.nanmin(time)), float(np.nanmax(time)))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5))
    panels = [
        (tips[:, 1], tips[:, 2], "Laboratory frame", "laboratory x", "laboratory y"),
        (relative_x, relative_y, "Relative to gel centroid", "x − Rc,x", "y − Rc,y"),
        (tips[:, 3], tips[:, 4], "Material frame", "material x", "material y"),
    ]
    line = None
    for ax, (x, y, title, xlabel, ylabel) in zip(axes, panels):
        line = colored_trajectory(ax, x, y, time, norm, linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle(f"CHM=0.30 — Gel {gel} spiral-tip trajectory (phase singularity)")
    fig.text(
        0.5,
        0.012,
        "Chemistry snapshots are separated by 100 TU; fine-scale lobe counts are temporally aliased.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    assert line is not None
    colorbar = fig.colorbar(line, ax=axes, fraction=0.018, pad=0.025)
    colorbar.set_label("time (TU)")
    fig.subplots_adjust(left=0.055, right=0.92, bottom=0.13, top=0.84, wspace=0.28)
    stem = output_dir / f"chm030_gel{gel}_spiral_tip_trajectory"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def rounded_pairs(x: np.ndarray, y: np.ndarray, stride: int) -> tuple[list[float], list[float]]:
    valid = np.isfinite(x) & np.isfinite(y)
    return (
        np.round(x[valid][::stride], 4).tolist(),
        np.round(y[valid][::stride], 4).tolist(),
    )


def save_inline_html(
    html_output: Path,
    bodycenters: list[np.ndarray],
    tips: list[np.ndarray],
) -> None:
    datasets = []
    for gel in (0, 1):
        body = bodycenters[gel]
        tip = tips[gel]
        datasets.append(
            {
                "gel": gel,
                "centroid": rounded_pairs(body[:, 1], body[:, 2], 25),
                "centroid_t": np.round(body[::25, 0], 1).tolist(),
                "tip_lab": rounded_pairs(tip[:, 1], tip[:, 2], 1),
                "tip_material": rounded_pairs(tip[:, 3], tip[:, 4], 1),
                "tip_t": np.round(tip[:, 0], 1).tolist(),
            }
        )

    payload = json.dumps(datasets, separators=(",", ":"))
    fragment = f"""<div id="chm03-trajectory-root">
  <div id="chm03-trajectory-plot" role="img" aria-label="Six trajectory plots showing the centroid, laboratory-frame spiral tip, and material-frame spiral tip for gels zero and one at CHM 0.30."></div>
</div>
<style>
  #chm03-trajectory-root {{ width: 100%; color: var(--foreground); }}
  #chm03-trajectory-plot {{ width: 100%; height: 820px; }}
  @media (max-width: 560px) {{ #chm03-trajectory-plot {{ height: 1040px; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('chm03-trajectory-root');
  const plot = document.getElementById('chm03-trajectory-plot');
  const datasets = {payload};
  const styles = getComputedStyle(root);
  const foreground = styles.getPropertyValue('--foreground').trim() || 'currentColor';
  const muted = styles.getPropertyValue('--muted-foreground').trim() || foreground;
  const border = styles.getPropertyValue('--border').trim() || muted;
  const traces = [];
  const titles = [];
  const axes = ['x', 'x2', 'x3', 'x4', 'x5', 'x6'];
  const yaxes = ['y', 'y2', 'y3', 'y4', 'y5', 'y6'];
  datasets.forEach((d, row) => {{
    const panels = [
      [d.centroid[0], d.centroid[1], d.centroid_t, `Gel ${{d.gel}} centroid — lab frame`],
      [d.tip_lab[0], d.tip_lab[1], d.tip_t, `Gel ${{d.gel}} spiral tip — lab frame`],
      [d.tip_material[0], d.tip_material[1], d.tip_t, `Gel ${{d.gel}} spiral tip — material frame`]
    ];
    panels.forEach((p, col) => {{
      const index = row * 3 + col;
      traces.push({{
        type: 'scattergl', mode: 'lines+markers', x: p[0], y: p[1],
        xaxis: axes[index], yaxis: yaxes[index], name: p[3], showlegend: false,
        line: {{color: muted, width: 0.7}},
        marker: {{size: col === 0 ? 2.4 : 3.2, color: p[2], colorscale: 'Viridis',
          cmin: 0, cmax: 200000, showscale: index === 0,
          colorbar: index === 0 ? {{title: {{text: 'time (TU)'}}, thickness: 12, len: 0.55, x: 1.02}} : undefined}}
      }});
      titles.push(p[3]);
    }});
  }});
  const axisStyle = {{showgrid: true, gridcolor: border, zeroline: false,
    linecolor: border, tickfont: {{color: muted}}, title: {{font: {{color: foreground}}}}}};
  const layout = {{
    margin: {{l: 58, r: 52, t: 54, b: 50}}, paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)', font: {{color: foreground}},
    grid: {{rows: 2, columns: 3, pattern: 'independent', xgap: 0.12, ygap: 0.16}},
    annotations: titles.map((title, index) => ({{
      text: title, showarrow: false, xref: axes[index] + ' domain', yref: yaxes[index] + ' domain',
      x: 0.5, y: 1.08, font: {{color: foreground, size: 13}}
    }}))
  }};
  for (let i = 1; i <= 6; i++) {{
    const suffix = i === 1 ? '' : String(i);
    layout['xaxis' + suffix] = {{...axisStyle, title: {{text: i % 3 === 0 ? 'material x' : 'laboratory x'}}, scaleanchor: 'y' + suffix, scaleratio: 1}};
    layout['yaxis' + suffix] = {{...axisStyle, title: {{text: i % 3 === 0 ? 'material y' : 'laboratory y'}}}};
  }}
  Plotly.newPlot(plot, traces, layout, {{responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d']}});
  new ResizeObserver(() => Plotly.Plots.resize(plot)).observe(root);
}})();
</script>
"""
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(fragment, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.run_dir / "0"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    u_ref, v_ref = phase_reference(data_dir)
    print(f"phase reference: u_ref={u_ref:.8f}, v_ref={v_ref:.8f}")

    bodycenters = [
        np.loadtxt(data_dir / f"gel{gel}bodycenter0.dat")
        for gel in (0, 1)
    ]
    tips = []
    for gel in (0, 1):
        tip, histogram = extract_phase_tips(data_dir, gel, u_ref, v_ref)
        print(f"gel {gel}: phase-singularity count histogram {histogram}")
        print(f"gel {gel}: maximum bilinear residual {np.nanmax(tip[:, 5]):.3e}")
        tips.append(tip)
        np.savetxt(
            args.output_dir / f"chm030_gel{gel}_phase_tip_trajectory.csv",
            tip,
            delimiter=",",
            header="time_TU,x_lab,y_lab,x_material,y_material,bilinear_residual",
            comments="",
        )
        save_centroid_figure(bodycenters[gel], gel, args.output_dir)
        save_tip_figure(bodycenters[gel], tip, gel, args.output_dir)

    if args.html_output is not None:
        save_inline_html(args.html_output, bodycenters, tips)
        print(f"inline HTML: {args.html_output}")
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
