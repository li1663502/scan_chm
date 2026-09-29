# Experiment 2 bifurcation refinement

This folder refines the high-`chi_M` transition in the `d_ini = 60` pair-attractor
data without changing the original sweep outputs.

## Method

- Baseline bracket: `0.15` is a quiet locked pair and `0.18` is nonlinear/breathing.
- Search target: first loss of the quiet locked-distance branch.
- Order parameter: late-time robust distance amplitude `A90 = P95(d) - P5(d)`.
- Quiet locked decision: low late `std`, low `A90`, high fraction with `d < 60`,
  and small late trend.
- Fast path: every trial is run in stages (`120k`, `160k`, `200k` TU). A clearly
  nonlinear trial can stop early; a quiet trial is extended to `200k` to avoid
  long-transient false positives.
- GPU usage: each search round launches two trisection points, so GPU 0 and GPU 1
  can run at the same time.

## Commands

Start or resume the recommended continuation scheduler:

```bash
cd experiment2_pair_attractor/bifurcation_refine
./start_bifurcation_continuation.sh
```

The slower fresh-start scheduler is also available when the basin from
`d_ini = 60` is the quantity of interest:

```bash
cd experiment2_pair_attractor/bifurcation_refine
./start_bifurcation_refine.sh
```

Refresh analysis and the refined PDF:

```bash
python3 scripts/analyze_bifurcation_refine.py
python3 scripts/plot_bifurcation_refine.py
```

Diagnose time-window bias and draw the protocol-consistent branch:

```bash
python3 scripts/analyze_sampling_sensitivity.py
python3 scripts/plot_bifurcation_phase_aligned.py
```

Draw the evidence-complete finite-time figure using existing trajectories only:

```bash
python3 scripts/plot_bifurcation_existing_data.py
```

Draw only the breathing centroid-distance maximum/minimum branches:

```bash
python3 scripts/plot_breathing_centroid_extrema.py
```

Average all resolved major maxima/minima in each available breathing record:

```bash
python3 scripts/plot_breathing_cycle_mean_extrema.py
```

Open markers have only one resolved extremum (`N=1`); filled markers contain
at least two extrema, with error bars showing one standard deviation.

This two-panel figure connects only the stationary branches and resolved first
maxima. Complete-cycle minima/maxima are separate triangles, while independent
capture/escape histories remain unconnected P5-P95 observations.

The phase-aligned plot does not connect independent baseline histories to the
fixed-seed continuation branch. For `chi_M > 0.18`, all plotted extrema use the
same post-switch interval (`t = 200001-239999`) after smoothing the distance
with a 501-point moving average. The current supplemental records resolve the
first maximum but not the following full minimum, so this is a finite-time
continuation envelope rather than a completed asymptotic limit-cycle branch.

Run fixed supplemental points for a smoother figure:

```bash
cd experiment2_pair_attractor/bifurcation_refine
./start_bifurcation_supplement.sh
```

Primary outputs:

- `analysis_outputs/bifurcation_scheduler_status.csv`
- `analysis_outputs/continuation_scheduler_status.csv`
- `analysis_outputs/supplement_scheduler_status.csv`
- `analysis_outputs/bifurcation_refine_metrics.csv`
- `analysis_outputs/bifurcation_estimates.csv`
- `analysis_outputs/chiM_and_d_bifurcation_refined.pdf`
- `analysis_outputs/sampling_window_sensitivity.csv`
- `analysis_outputs/phase_aligned_bifurcation_branch.csv`
- `analysis_outputs/sampling_window_diagnosis.md`
- `analysis_outputs/chiM_and_d_bifurcation_phase_aligned.pdf`
- `analysis_outputs/sampling_window_comparison.pdf`
- `analysis_outputs/existing_data_bifurcation_points.csv`
- `analysis_outputs/chiM_and_d_bifurcation_existing_data.pdf`
- `analysis_outputs/breathing_centroid_extrema_selected.csv`
- `analysis_outputs/chiM_breathing_centroid_extrema.pdf`
- `analysis_outputs/breathing_cycle_extrema_metrics.csv`
- `analysis_outputs/chiM_breathing_cycle_mean_extrema.pdf`
