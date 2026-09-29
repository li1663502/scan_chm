# Sampling-window diagnosis

## Result

The large zig-zag in the previous orange polyline is not a single numerical branch. It joins independent baseline histories and fixed-seed continuation runs after sorting only by chi_M. Time-window phase bias then changes the reported lower/upper distances further.

## Window sensitivity examples

- chi_M=0.181: P5 changes from 84.77 (last 10000 steps) to 56.36 (full 39999-step post-switch record).
- chi_M=0.290: P5 changes from 113.08 (last 10000 steps) to 59.99 (full 39999-step post-switch record).

## Physical variability that remains

The baseline chi_M=0.23, 0.25, and 0.28 records contain capture, large escape, or incomplete-cycle episodes. Those differences are history-sensitive dynamics and must remain as unconnected validation observations; choosing prettier samples would hide real behavior.

## Plotting rule used

The clean branch uses only one fixed-seed continuation protocol. For chi_M>0.18 it takes the global lower and upper values of a 501-point moving-average distance over the same elapsed-time window t=200001-239999. The chi_M=0.18 anchor uses t=160000-199999. Critical-band states are shown but not connected to the breathing envelope.

## Roughness check

- Previous mixed polyline total variation: lower=526.35, upper=363.19.
- Phase-aligned branch total variation: lower=2.72, upper=35.49.
- The phase-aligned supplement records contain one resolved maximum but do not yet contain the following full minimum; the lower edge is therefore the common locked starting phase, not a proven asymptotic minimum branch.
