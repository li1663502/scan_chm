Source code snapshot for sweep_long_d60
- First case computed: CHM=0.0 (CHM_000)
- All 4 cases (CHM=0.0, 0.05, 0.1, 0.2) used the SAME binary, this code reproduces it
- Date: 2026年 05月 18日 星期一 10:01:08 CST
- Branch: changeU_Fchemo + Patches P1-P7 + d_init/base env vars
- Compile: nvcc -std=c++14 -O2 -arch=native -lcudart ligel.cu ligelSystem.cu ligel_kernel.cu libottomgridSystem.cu -o experiment2
- Known issue (diagnosed 2026-05-18): calRepulsion uses perimeter-only LJ with POSI=1e-4 and is called every 5 iter
  - At CHM>=0.1, F_chemo total can equal or exceed LJ contact wall, causing edge interpenetration
  - At CHM=0.2 the gels partially pass through each other; figures show d going below ~50 (gel size 50 wide)
  - Fix candidates: raise POSI; remove % 5 gating; use full-node LJ instead of perimeter-only
