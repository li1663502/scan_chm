Source snapshot for sweep_long_d60_fixed
- First case computed: CHM=0.0 (CHM_000)
- All 4 cases used the same binary 'experiment2_fixed'
- Repulsion fix applied:
  * POSI 1e-4 -> 1e-3 (10x LJ depth)         [ligel_kernel.cu:1762]
  * Source loop: perimeter -> all nodes      [ligel_kernel.cu:1771-1784]
  * Removed 5-step gating on calRepulsion    [ligel.cu:251-255]
- Date: 2026年 05月 19日 星期二 10:07:49 CST
- Compile: nvcc -std=c++14 -O2 -arch=native -lcudart \
    ligel.cu ligelSystem.cu ligel_kernel.cu libottomgridSystem.cu -o experiment2_fixed
- Performance: ~676 s/M iter (6.3x slower than pre-fix 107 s/M iter)
