# Experiment 2 bifurcation refinement

Root: `/media/xb/F963E8A1EEEE8BFC/liyang/li-gpu/BOTTOM FEEDBACK/phi_feedback/changeU/experiment2_pair_attractor/bifurcation_refine`

## Current critical brackets

- Lower capture threshold: `0.03000 < chi_M,c1 < 0.05000` (midpoint `0.04000`, width `0.02000`).
- Upper locked-to-nonlinear threshold: `0.17770 < chi_M,c2 < 0.17778` (midpoint `0.17774`, width `0.00008`).

## Distance metrics

| source | case | chi_M | n | mean | std | P5-P95 amp | bound% | state |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | CHM_000 | 0.00000 | 120000 | 71.77 | 0.06 | 0.24 | 0.0 | unbound |
| baseline | CHM_00025 | 0.00250 | 120000 | 71.85 | 0.06 | 0.23 | 0.0 | unbound |
| baseline | CHM_0005 | 0.00500 | 120000 | 71.94 | 0.06 | 0.23 | 0.0 | unbound |
| baseline | CHM_001 | 0.01000 | 120000 | 72.10 | 0.06 | 0.19 | 0.0 | unbound |
| baseline | CHM_003 | 0.03000 | 120000 | 72.84 | 0.05 | 0.17 | 0.0 | unbound |
| baseline | CHM_005 | 0.05000 | 120000 | 53.05 | 1.36 | 3.77 | 100.0 | locked |
| baseline | CHM_008 | 0.08000 | 120000 | 53.05 | 1.38 | 3.81 | 100.0 | locked |
| baseline | CHM_010 | 0.10000 | 120000 | 49.86 | 2.04 | 6.57 | 100.0 | locked |
| baseline | CHM_015 | 0.15000 | 200000 | 55.64 | 1.53 | 4.20 | 100.0 | locked |
| continuation | CHM_0p16 | 0.16000 | 80000 | 56.56 | 1.55 | 4.28 | 100.0 | locked |
| continuation | CHM_0p16667 | 0.16667 | 40000 | 57.11 | 1.56 | 4.54 | 100.0 | locked |
| continuation | CHM_0p17 | 0.17000 | 80000 | 57.29 | 1.55 | 4.58 | 98.6 | locked |
| continuation | CHM_0p17333 | 0.17333 | 80000 | 57.90 | 1.27 | 3.71 | 95.8 | locked |
| continuation | CHM_0p17556 | 0.17556 | 40000 | 57.39 | 1.64 | 5.12 | 93.3 | locked |
| continuation | CHM_0p17667 | 0.17667 | 80000 | 57.59 | 1.39 | 4.02 | 98.9 | locked |
| continuation | CHM_0p17704 | 0.17704 | 80000 | 57.52 | 1.54 | 4.41 | 97.8 | locked |
| continuation | CHM_0p1772 | 0.17720 | 80000 | 62.34 | 3.50 | 10.28 | 34.2 | nonlinear |
| continuation | CHM_0p17741 | 0.17741 | 80000 | 61.19 | 3.68 | 10.93 | 47.7 | nonlinear |
| continuation | CHM_0p1775 | 0.17750 | 80000 | 58.14 | 1.96 | 6.63 | 85.3 | transition |
| continuation | CHM_0p1776 | 0.17760 | 80000 | 58.23 | 1.54 | 5.00 | 88.2 | transition |
| continuation | CHM_0p1777 | 0.17770 | 80000 | 57.60 | 1.42 | 4.00 | 98.5 | locked |
| continuation | CHM_0p17778 | 0.17778 | 80000 | 59.93 | 3.97 | 12.39 | 52.8 | nonlinear |
| continuation | CHM_0p17889 | 0.17889 | 80000 | 65.66 | 3.61 | 11.70 | 9.0 | nonlinear |
| baseline | CHM_018 | 0.18000 | 200000 | 76.13 | 14.04 | 39.34 | 22.2 | nonlinear |
| supplement | CHM_0p181 | 0.18100 | 40000 | 89.43 | 2.55 | 8.43 | 0.0 | nonlinear |
| supplement | CHM_0p1825 | 0.18250 | 40000 | 89.87 | 2.53 | 8.36 | 0.0 | nonlinear |
| supplement | CHM_0p185 | 0.18500 | 40000 | 90.91 | 2.50 | 8.25 | 0.0 | nonlinear |
| supplement | CHM_0p1875 | 0.18750 | 40000 | 91.43 | 2.63 | 8.64 | 0.0 | nonlinear |
| supplement | CHM_0p19 | 0.19000 | 40000 | 92.32 | 2.67 | 8.74 | 0.0 | nonlinear |
| supplement | CHM_0p195 | 0.19500 | 40000 | 93.81 | 2.59 | 8.48 | 0.0 | nonlinear |
| baseline | CHM_020 | 0.20000 | 200000 | 74.64 | 16.54 | 49.12 | 26.0 | nonlinear |
| supplement | CHM_0p205 | 0.20500 | 40000 | 96.54 | 2.67 | 8.70 | 0.0 | nonlinear |
| supplement | CHM_0p215 | 0.21500 | 40000 | 99.35 | 2.70 | 8.75 | 0.0 | nonlinear |
| supplement | CHM_0p22 | 0.22000 | 40000 | 100.81 | 2.92 | 9.36 | 0.0 | nonlinear |
| baseline | CHM_023 | 0.23000 | 200000 | 50.65 | 2.67 | 9.13 | 100.0 | nonlinear |
| supplement | CHM_0p24 | 0.24000 | 40000 | 107.23 | 3.08 | 9.84 | 0.0 | nonlinear |
| baseline | CHM_025 | 0.25000 | 200000 | 91.43 | 53.24 | 153.17 | 57.7 | nonlinear |
| supplement | CHM_0p26 | 0.26000 | 40000 | 113.02 | 3.61 | 11.39 | 0.0 | nonlinear |
| baseline | CHM_028 | 0.28000 | 200000 | 68.16 | 17.18 | 55.91 | 57.4 | nonlinear |
| supplement | CHM_0p29 | 0.29000 | 40000 | 121.39 | 4.54 | 14.09 | 0.0 | nonlinear |
| baseline | CHM_030 | 0.30000 | 200000 | 106.18 | 25.31 | 76.86 | 5.5 | nonlinear |
