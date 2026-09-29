#include "ligelSystem.h"
#include "libottomgridSystem.h"
#include "ligel_kernel.cuh"
#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdlib>   // [Exp1] for std::getenv / atoll

// [Exp1] runstep moved into main() as a runtime variable (RUNSTEP env var)

// ============================================================
// Steering Feedback Parameters  (edit per scan variant)
// ============================================================
#define STEER_K_TURN    0.50f
#define STEER_ALPHA_DELAY     0.03f
#define STEER_T_ON      5000
#define STEER_RHO_MEM      0.9995    // substrate memory decay per grid update
#define STEER_D_AHEAD   20.0f
#define STEER_D_SIDE    15.0f
#define STEER_INTERVAL     10000     // how often (iters) to update heading & steer_delay
// B2: sinusoidal k_turn modulation
// k_turn(t) = K0 * sin(2*pi * t / T_period)
// T_period = 10000 TU = 10000000 iters
#define STEER_B2_PERIOD   10000000LL  // 10000 TU full period

// ============================================================

#define CUDA_OK(call) do {                                      \
  cudaError_t _e = (call);                                      \
  if (_e != cudaSuccess) {                                      \
    fprintf(stderr, "CUDA ERR %s:%d: %s\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(_e));        \
    abort();                                                    \
  }                                                             \
} while(0)

int main(int argc, char** argv) {
    CUDA_OK(cudaSetDevice(0));

    // [Exp1] runstep configurable via RUNSTEP env var (default 5000000 iter)
    const long long runstep = std::getenv("RUNSTEP") ? atoll(std::getenv("RUNSTEP")) : 5000000;
    printf("[Exp1] runstep = %lld iter\n", runstep);

    int time = 0;
    int2 gelSize = make_int2(50, 50);   // ��ߴ�
    int2 bottomgridSize = make_int2(400, 400);
    const int ng = 2;                         // [Exp2] pair of gels

    int count = 0;

    char originalDirectory[FILENAME_MAX];
    if (!getcwd(originalDirectory, sizeof(originalDirectory))) {
        std::cerr << "Error getting current directory\n";
        return EXIT_FAILURE;
    }

    BottomgridSystem bottomGrid(bottomgridSize, time);


    for (int i = 15; i <= 15; i++) {
        for (double j = 10; j <= 10; j++) {
            for (double k = 0.3; k <= 0.3; k = k + 0.05) {

                std::string directoryName = std::to_string(count);
                mkdir(directoryName.c_str(), 0755);
                chdir(directoryName.c_str());

                // ---------------- ������������ ----------------
                std::vector<std::unique_ptr<GelSystem>> gels;
                gels.reserve(ng);

                double dx = 1.0, suitlocX = 150.0, suitlocY = 150.0, LAMP = 1.1;
                // 与Fortran一致: radius2=60, 凝胶中心间距=60
                // gelSize*LAMP*dx = 50*1.1*1.0 = 55, 加5使间距=60
                double intervalx = gelSize.x * LAMP * dx + 5.0;
                double intervaly = gelSize.y * LAMP * dx + 5.0;

                // [Exp2.5] override gel spacing via env vars (physical lattice units).
                // For ng=2 vertical pair, intervaly = d_init between gel0/gel1 startY.
                if (const char* ev = std::getenv("D_INIT_X")) intervalx = atof(ev);
                if (const char* ev = std::getenv("D_INIT_Y")) intervaly = atof(ev);

                // [Exp2.5] override gel base position to auto-center the pair on the 400x400
                // substrate (=200, 200). Defaults preserve original placement (150, 150).
                // For ng=2 vertical pair with given intervaly, set suitlocY = 200 - intervaly/2
                // so that the pair midpoint sits at y=200.
                if (const char* ev = std::getenv("GEL_BASE_X")) suitlocX = atof(ev);
                if (const char* ev = std::getenv("GEL_BASE_Y")) suitlocY = atof(ev);

                printf("[Exp2.5] gel spacing: intervalx=%.2f intervaly=%.2f, base=(%.1f, %.1f)\n",
                       intervalx, intervaly, suitlocX, suitlocY);

                for (int ig = 0; ig < ng; ++ig) {
                    double startX = 0, startY = 0;
                    if (ig < 2) { startX = suitlocX;                startY = suitlocY + intervaly * ig; }
                    else if (ig < 8) { startX = suitlocX + intervalx; startY = suitlocY + intervaly * (ig - 2); }
                    else if (ig < 12) { startX = suitlocX + 2 * intervalx; startY = suitlocY + intervaly * (ig - 8); }
                    // else if (ig < 4) { startX = suitlocX + intervalx1; startY = suitlocY + intervaly * (ig - 2); }
                    // else { startX = suitlocX + 3 * intervalx;         startY = suitlocY + intervaly * (ig - 12); }
                    gels.emplace_back(std::make_unique<GelSystem>(ng, ig, gelSize, time, i, j, k, startX, startY));
                }


                // rn ָ������豸�ࣩ
                double2** d_gels_rn = nullptr;
                CUDA_OK(cudaMalloc(&d_gels_rn, sizeof(double2*) * ng));

                // ŷ���Ĳ�����Ķ������豸��
                double2** d_gels_rm = nullptr;
                double** d_gels_um = nullptr;
                SimParams* d_params = nullptr;
                {
                    std::vector<double2*>  h_rm(ng);
                    std::vector<double*>   h_um(ng);
                    std::vector<SimParams> h_params(ng);
                    for (int ig = 0; ig < ng; ++ig) {
                        h_rm[ig] = gels[ig]->m_drm;
                        h_um[ig] = gels[ig]->m_dum;
                        h_params[ig] = gels[ig]->m_params;
                    }
                    bottomGrid.configureHolepointAuto(h_params.data(), ng);
                    CUDA_OK(cudaMalloc(&d_gels_rm, sizeof(double2*) * ng));
                    CUDA_OK(cudaMalloc(&d_gels_um, sizeof(double*) * ng));
                    CUDA_OK(cudaMalloc(&d_params, sizeof(SimParams) * ng));
                    CUDA_OK(cudaMemcpy(d_gels_rm, h_rm.data(), sizeof(double2*) * ng, cudaMemcpyHostToDevice));
                    CUDA_OK(cudaMemcpy(d_gels_um, h_um.data(), sizeof(double*) * ng, cudaMemcpyHostToDevice));
                    CUDA_OK(cudaMemcpy(d_params, h_params.data(), sizeof(SimParams) * ng, cudaMemcpyHostToDevice));
                }

                // ����ͬ���¼���ÿ�� gel ���һ֡
                std::vector<cudaEvent_t> ev_gel_ready(ng);
                for (int ig = 0; ig < ng; ++ig)
                    CUDA_OK(cudaEventCreateWithFlags(&ev_gel_ready[ig], cudaEventDisableTiming));

                // ��ӡ
                printf("CHS = %f, f = %f, ep = %f, I = %f\n",
                    gels[0]->m_params.CHS, gels[0]->m_params.f,
                    gels[0]->m_params.ep, gels[0]->m_params.phi);
                gels[0]->m_count = count;
                printf("count = %d\n", count);

                const int LX = gelSize.x;
                const int LY = gelSize.y;
                const int perim = 2 * (LX - 1) + 2 * (LY - 1); // ���� kernel �ġ��߽������
                const int repBlock = 128;
                const int repGrid = (perim + repBlock - 1) / repBlock;
                // const int repGrid = perim;

                int maxC = 0;
                for (int ig = 0; ig < ng; ++ig) {
                    int LX = gels[ig]->m_params.LX;
                    int LY = gels[ig]->m_params.LY;
                    maxC = max(maxC, 2 * (LX - 1) + 2 * (LY - 1));    // �ܽ綥�����Ͻ�
                }
                int MAX_C = max(256, maxC);     // ��������
                int MAX_Y = MAX_C;              // �������������������Ͻ�ȡ��ͬ

                // ---- warmup: ÿ�� gel ֻ��һ�� ----
                for (int ig = 0; ig < ng; ++ig) {
                    auto& G = *gels[ig];
                    const size_t bytes_xy =
                        sizeof(double2) * (size_t)((G.m_params.LX + 2) * (G.m_params.LY + 2));

                    CUDA_OK(cudaMemcpyAsync(G.m_drnn, G.m_drn, bytes_xy,
                        cudaMemcpyDeviceToDevice, G.m_gel_stream));
                    CUDA_OK(cudaMemsetAsync(G.m_dVeln0, 0, bytes_xy, G.m_gel_stream));
                    CUDA_OK(cudaMemsetAsync(G.m_dVeln, 0, bytes_xy, G.m_gel_stream));
                    CUDA_OK(cudaMemsetAsync(G.m_dFn, 0, bytes_xy, G.m_gel_stream));

                    if (time == 0) {
                        // �� wm/pm ���Ƕ��ļ��õ����Ȱ���ǰλ����һ��
                        calElementsvolumefraction << <gels[ig]->m_gridDim1, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream >> >
                            (gels[ig]->m_dwm, gels[ig]->m_dwmo, gels[ig]->m_dwn, gels[ig]->m_drn, gels[ig]->m_params);
                        // 6) �ڵ��������
                        calNodesvolumefraction << <gels[ig]->m_gridDim2, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream >> >
                            (gels[ig]->m_dwm, gels[ig]->m_dwn, gels[ig]->m_drn, gels[ig]->m_params);
                        // [FchemoCoupling] sample substrate-M at element centers before pm,
                        // so calPressureD can include the CHM*w*M term. With CHM=0 (default)
                        // this is a no-op numerically and preserves original Y-B warmup.
                        calMatElem << <gels[ig]->m_gridDim1, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream >> >
                            (gels[ig]->m_dM_at_elem, gels[ig]->m_drm, bottomGrid.m_dungrid, gels[ig]->m_params, bottomGrid.m_gridparams);
                        calPressureD << <gels[ig]->m_gridDim1, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream >> >
                            (gels[ig]->m_dPrem, gels[ig]->m_dvm, gels[ig]->m_dwm, gels[ig]->m_dM_at_elem, gels[ig]->m_params);
                    }
                }
                // ������ warmup ���
                for (int ig = 0; ig < ng; ++ig)
                    CUDA_OK(cudaStreamSynchronize(gels[ig]->m_gel_stream));

                // ────── Steering Feedback: host-side state ──────
                // h_Rc: gel centroid; h_ehat: forward unit vector; h_nhat: lateral unit vector
                // h_steer_delay: low-pass filtered left-right differential signal
                double2 h_Rc[ng], h_ehat[ng], h_nhat[ng];
                double  h_steer_delay[ng];
                for (int ig = 0; ig < ng; ++ig) {
                    h_Rc[ig]          = make_double2(0.0, 0.0);
                    h_ehat[ig]        = make_double2(1.0, 0.0);  // default: heading +x
                    h_nhat[ig]        = make_double2(0.0, 1.0);  // default: lateral +y
                    h_steer_delay[ig] = 0.0;
                }
                // Device buffer for sampled substrate values: [M_left_0, M_right_0, M_left_1, ...]
                double* d_Mlr = nullptr;
                CUDA_OK(cudaMalloc(&d_Mlr, sizeof(double) * 2 * ng));
                CUDA_OK(cudaMemset(d_Mlr, 0, sizeof(double) * 2 * ng));
                // Host buffer for reading element positions to compute centroid
                const int nElems = gels[0]->m_numGelElements;
                double2* h_rm_buf = new double2[nElems];
                // ─────────────────────────────────────────────────


                for (long long solverIterations = time * gels[0]->m_df; solverIterations <= runstep; ++solverIterations)
                {
                    // ÿ 5 ����һ���ų⣨�����㡢ָ���ˢ�¡�ŷ���Ĳ���
                    if (solverIterations % 5 == 0) {

                        // 0) �����ų⻺��
                        for (int ig = 0; ig < ng; ++ig) {
                            size_t bytes = sizeof(double2) * (size_t)((LX + 2) * (LY + 2));
                            CUDA_OK(cudaMemsetAsync(gels[ig]->m_dFFRepulsion1, 0, bytes, gels[ig]->m_gel_stream));
                            CUDA_OK(cudaMemsetAsync(gels[ig]->m_dFFRepulsion2, 0, bytes, gels[ig]->m_gel_stream));
                        }

                        // 1) ͳһˢ�� rn ָ�����Host->Device����������������ͬ��
                        std::vector<double2*> h_rn(ng);
                        for (int t = 0; t < ng; ++t) {
                            h_rn[t] = gels[t]->m_drn;
                            if (!h_rn[t]) printf("ERROR: gel[%d] m_drn is null!\n", t);
                        }
                        // ��������С��Ҳ����������������������ʾ�첽 + �¼�
                        CUDA_OK(cudaMemcpyAsync(d_gels_rn, h_rn.data(),
                            sizeof(double2*) * ng,
                            cudaMemcpyHostToDevice,
                            bottomGrid.m_grid_stream));

                        cudaEvent_t ev_rn_updated;
                        CUDA_OK(cudaEventCreateWithFlags(&ev_rn_updated, cudaEventDisableTiming));
                        CUDA_OK(cudaEventRecord(ev_rn_updated, bottomGrid.m_grid_stream));
                        for (int ig = 0; ig < ng; ++ig)
                            CUDA_OK(cudaStreamWaitEvent(gels[ig]->m_gel_stream, ev_rn_updated, 0));
                        CUDA_OK(cudaEventDestroy(ev_rn_updated));
                    }

                    // 2) ÿ�� gel���ų⣨ÿ 5 ���� + update
                    for (int ig = 0; ig < ng; ++ig) {
                        if (!gels[ig]->result) { printf("[Debug] gel[%d] result=false\n", ig); break; }

                        if (solverIterations % 5 == 0) {
                            calRepulsion << <repGrid, repBlock, 0, gels[ig]->m_gel_stream >> > (gels[ig]->m_dFFRepulsion1, gels[ig]->m_dFFRepulsion2,
                                gels[ig]->m_drn, d_gels_rn, ng, LX, LY, gels[ig]->m_params, bottomGrid.m_gridparams);
                            CUDA_OK(cudaPeekAtLastError());

                            // projectToLegalDomain << <repGrid, repBlock, 0, gels[ig]->m_gel_stream >> > (gels[ig]->m_drn, LX, LY, gels[ig]->m_params, bottomGrid.m_gridparams);

                        }

                        gels[ig]->update(solverIterations,
                            bottomGrid.m_dungrid, bottomGrid.m_gridparams);   // [FchemoCoupling]
                        CUDA_OK(cudaEventRecord(ev_gel_ready[ig], gels[ig]->m_gel_stream));
                    }

                    // 3) ÿ 5 ����ŷ���Ĳ����ȴ����� gel ��֡�¼���
                    if (solverIterations % 5 == 0) {
                        // ָ��� rm/um/params Ҳˢ�£��Է����������ط��䣩
                        std::vector<double2*>  h_rm(ng);
                        std::vector<double*>   h_um(ng);
                        std::vector<SimParams> h_params(ng);
                        for (int ig = 0; ig < ng; ++ig) {
                            h_rm[ig] = gels[ig]->m_drm;
                            h_um[ig] = gels[ig]->m_dum;
                            h_params[ig] = gels[ig]->m_params;
                        }
                       
                        CUDA_OK(cudaMemcpyAsync(d_gels_rm, h_rm.data(), sizeof(double2*) * ng,
                            cudaMemcpyHostToDevice, bottomGrid.m_grid_stream));
                        CUDA_OK(cudaMemcpyAsync(d_gels_um, h_um.data(), sizeof(double*) * ng,
                            cudaMemcpyHostToDevice, bottomGrid.m_grid_stream));
                        CUDA_OK(cudaMemcpyAsync(d_params, h_params.data(), sizeof(SimParams) * ng,
                            cudaMemcpyHostToDevice, bottomGrid.m_grid_stream));

                        for (int ig = 0; ig < ng; ++ig)
                            CUDA_OK(cudaStreamWaitEvent(bottomGrid.m_grid_stream, ev_gel_ready[ig], 0));
                        // ────── Steering: update heading and steer_delay every STEER_INTERVAL iters ──────
#if 0  // [Exp1] disable steering feedback to isolate F_chemo
                        if (solverIterations % STEER_INTERVAL == 0) {
                            CUDA_OK(cudaStreamSynchronize(bottomGrid.m_grid_stream));
                            for (int ig = 0; ig < ng; ++ig) {
                                CUDA_OK(cudaStreamSynchronize(gels[ig]->m_gel_stream));
                                // Read element positions to compute centroid
                                CUDA_OK(cudaMemcpy(h_rm_buf, gels[ig]->m_drm,
                                    sizeof(double2) * nElems, cudaMemcpyDeviceToHost));
                                double cx = 0.0, cy = 0.0;
                                for (int e = 0; e < nElems; ++e) { cx += h_rm_buf[e].x; cy += h_rm_buf[e].y; }
                                cx /= nElems; cy /= nElems;
                                double2 Rc_new = make_double2(cx, cy);

                                // Update heading via EMA if gel has moved
                                double dvx = Rc_new.x - h_Rc[ig].x;
                                double dvy = Rc_new.y - h_Rc[ig].y;
                                double dist = sqrt(dvx*dvx + dvy*dvy);
                                if (dist > 1e-3) {
                                    const double alpha_h = 0.3;
                                    double ex = dvx/dist, ey = dvy/dist;
                                    h_ehat[ig].x = (1-alpha_h)*h_ehat[ig].x + alpha_h*ex;
                                    h_ehat[ig].y = (1-alpha_h)*h_ehat[ig].y + alpha_h*ey;
                                    double elen = sqrt(h_ehat[ig].x*h_ehat[ig].x + h_ehat[ig].y*h_ehat[ig].y);
                                    if (elen > 1e-10) { h_ehat[ig].x /= elen; h_ehat[ig].y /= elen; }
                                    h_nhat[ig].x = -h_ehat[ig].y;
                                    h_nhat[ig].y =  h_ehat[ig].x;
                                }
                                h_Rc[ig] = Rc_new;

                                // Sample substrate memory at left/right forward points
                                double px_L = Rc_new.x + STEER_D_AHEAD*h_ehat[ig].x - STEER_D_SIDE*h_nhat[ig].x;
                                double py_L = Rc_new.y + STEER_D_AHEAD*h_ehat[ig].y - STEER_D_SIDE*h_nhat[ig].y;
                                double px_R = Rc_new.x + STEER_D_AHEAD*h_ehat[ig].x + STEER_D_SIDE*h_nhat[ig].x;
                                double py_R = Rc_new.y + STEER_D_AHEAD*h_ehat[ig].y + STEER_D_SIDE*h_nhat[ig].y;

                                calSampleUngrid<<<1, 2, 0, gels[ig]->m_gel_stream>>>(
                                    bottomGrid.m_dungrid, bottomGrid.m_gridparams,
                                    px_L, py_L, px_R, py_R,
                                    d_Mlr + ig*2, d_Mlr + ig*2 + 1);
                                CUDA_OK(cudaStreamSynchronize(gels[ig]->m_gel_stream));

                                double M_left, M_right;
                                CUDA_OK(cudaMemcpy(&M_left,  d_Mlr + ig*2,     sizeof(double), cudaMemcpyDeviceToHost));
                                CUDA_OK(cudaMemcpy(&M_right, d_Mlr + ig*2 + 1, sizeof(double), cudaMemcpyDeviceToHost));

                                double delta_raw = M_right - M_left;
                                h_steer_delay[ig] = (1.0 - STEER_ALPHA_DELAY)*h_steer_delay[ig]
                                                  + STEER_ALPHA_DELAY * delta_raw;

                                // ── B2: sinusoidal k_turn modulation ──
                                // k_turn(t) = K0 * sin(2*pi*(t-t_on)/T_period)
                                if (solverIterations >= STEER_T_ON) {
                                    long long t_since_on = solverIterations - (long long)(STEER_T_ON * gels[0]->m_df);
                                    double phase = 2.0 * M_PI * (double)t_since_on / (double)STEER_B2_PERIOD;
                                    double k_sine = (double)STEER_K_TURN * sin(phase);
                                    applySteeringPulseD<<<gels[ig]->m_gridDim1, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream>>>(
                                        gels[ig]->m_dum, gels[ig]->m_drm, gels[ig]->m_dmap_element,
                                        gels[ig]->m_params,
                                        h_Rc[ig], h_ehat[ig], h_nhat[ig],
                                        h_steer_delay[ig], k_sine);
                                    if (ig == 0 && solverIterations % 5000000 == 0) {
                                        printf("[B2-SINE t=%.0f] k_sine=%.3f phase=%.2f\n",
                                               solverIterations*0.001, k_sine, phase);
                                        fflush(stdout);
                                    }
                                    CUDA_OK(cudaPeekAtLastError());
                                }

                                if (solverIterations % 100000 == 0) {
                                    printf("[Steer ig=%d t=%.0f] Rc=(%.1f,%.1f) ehat=(%.3f,%.3f) M_L=%.4f M_R=%.4f delta=%.4f steer_d=%.4f\n",
                                        ig, solverIterations*0.001,
                                        cx, cy, h_ehat[ig].x, h_ehat[ig].y,
                                        M_left, M_right, delta_raw, h_steer_delay[ig]);
                                }
                            }
                        }
#endif  // [Exp1] end disable steering
                        // ──────────────────────────────────────────────────────────────────────────

                        // ────
                        calClearLab << <bottomGrid.m_ggridDim, bottomGrid.m_gblockDim, 0, bottomGrid.m_grid_stream >> > (
                            bottomGrid.m_dlab, bottomGrid.m_dungrid, bottomGrid.m_dungriddiffusion,
                            bottomGrid.m_gridparams, STEER_RHO_MEM);
                        CUDA_OK(cudaPeekAtLastError());
                      
                        calHolepoint << < bottomGrid.HolgridDim,
                            bottomGrid.HolblockDim,
                            bottomGrid.smem_bytes,
                            bottomGrid.m_grid_stream >> > (
                                d_gels_rm, d_params, ng,
                                bottomGrid.m_dlab, bottomGrid.m_gridparams,
                                bottomGrid.Hol_maxC, bottomGrid.Hol_sortN
                                );
                        CUDA_OK(cudaPeekAtLastError());

                        // ��ֵ
                        calGridInterpOnly << <bottomGrid.m_ggridDim, bottomGrid.m_gblockDim, 0, bottomGrid.m_grid_stream >> > (bottomGrid.m_dungriddiffusion,
                            d_gels_rm, d_gels_um, d_params, ng, bottomGrid.m_dlocgridx, bottomGrid.m_dlocgridy, bottomGrid.m_dlab, bottomGrid.m_gridparams);
                        CUDA_OK(cudaPeekAtLastError());

                        // ��ɢ/������
                        calGridDiffuseOnly << <bottomGrid.m_ggridDim, bottomGrid.m_gblockDim, 0, bottomGrid.m_grid_stream >> > (
                            bottomGrid.m_dungrid, bottomGrid.m_dungriddiffusion, bottomGrid.m_dlab, bottomGrid.m_gridparams, gels[0]->m_params.dt);
                        CUDA_OK(cudaPeekAtLastError());

                        // �߽�
                        calgridBoundary << <bottomGrid.m_ggridDim, bottomGrid.m_gblockDim, 0, bottomGrid.m_grid_stream >> > (
                            bottomGrid.m_dungrid, bottomGrid.m_dgridmap_node, bottomGrid.m_gridparams);
                        CUDA_OK(cudaPeekAtLastError());

                        CUDA_OK(cudaStreamSynchronize(bottomGrid.m_grid_stream));
                    }
                    int ig = 0;
                    // 4) ����ÿ����
                    for (ig = 0; ig < ng; ++ig) {
                        // Bug fix: after swap, m_drnn holds NEW positions (post-mechanics),
                        // m_drn holds OLD positions. Use m_drnn to match Fortran.
                        calTermsD << <gels[ig]->m_gridDim_1, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream >> > (
                            gels[ig]->m_djsum, gels[ig]->m_djsvm, gels[ig]->m_djspm,
                            gels[ig]->m_dwm, gels[ig]->m_dwmo, gels[ig]->m_drnn, gels[ig]->m_dVeln,
                            gels[ig]->m_dwn, gels[ig]->m_dun, gels[ig]->m_dum_norm, gels[ig]->m_dum,
                            gels[ig]->m_dvm_norm, gels[ig]->m_dvm, gels[ig]->m_drm, gels[ig]->m_params);
                        CUDA_OK(cudaPeekAtLastError());

                        /*   if ( ig == 0 || ig == 3) {
                               calChemD2 << <gels[ig]->m_gridDim_1, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream >> > (
                                   gels[ig]->m_dvm_norm, gels[ig]->m_dum_norm, gels[ig]->m_dvm, gels[ig]->m_dum,
                                   gels[ig]->m_dwm, gels[ig]->m_djsum, gels[ig]->m_djsvm, gels[ig]->m_djspm,
                                   gels[ig]->m_drm, gels[ig]->m_params);
                               CUDA_OK(cudaPeekAtLastError());
                           }*/
                           // if (ig == 1 || ig == 2) {
                        calChemD1 << <gels[ig]->m_gridDim_1, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream >> > (
                            gels[ig]->m_dvm_norm, gels[ig]->m_dum_norm, gels[ig]->m_dvm, gels[ig]->m_dum,
                            gels[ig]->m_dwm, gels[ig]->m_djsum, gels[ig]->m_djsvm, gels[ig]->m_djspm,
                            gels[ig]->m_drm, gels[ig]->m_params);
                        CUDA_OK(cudaPeekAtLastError());
                        //  }

                        calChemBoundaryD << <gels[ig]->m_gridDim1, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream >> > (
                            gels[ig]->m_dum, gels[ig]->m_drm, gels[ig]->m_dwm, gels[ig]->m_dmap_element,
                            bottomGrid.m_dungrid, gels[ig]->m_params, bottomGrid.m_gridparams);
                        CUDA_OK(cudaPeekAtLastError());

                        calUVnnormD << <gels[ig]->m_gridDim2, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream >> > (
                            gels[ig]->m_dun, gels[ig]->m_dum, gels[ig]->m_dvm, gels[ig]->m_dvn,
                            gels[ig]->m_dwm, gels[ig]->m_dwmo, gels[ig]->m_params);
                        CUDA_OK(cudaPeekAtLastError());

                        if (solverIterations % 1000 == 0) {

                            calrecordCenterElementD << <dim3(1, 1, 1), dim3(1, 1, 1), 0, gels[ig]->m_gel_stream >> > (
                                gels[ig]->m_dum_center, gels[ig]->m_dvm_center, gels[ig]->m_dwm_center, gels[ig]->m_drm_center,
                                gels[ig]->m_dFn_center, gels[ig]->m_dVeln_center,
                                gels[ig]->m_dum, gels[ig]->m_dvm, gels[ig]->m_dwm,
                                gels[ig]->m_drn, gels[ig]->m_dFn, gels[ig]->m_dVeln,
                                (solverIterations / 1000) % 1000, gels[ig]->m_params);
                            CUDA_OK(cudaPeekAtLastError());

                            cudaMemset(gels[ig]->d_hitCnt, 0, sizeof(unsigned int));
                            calFilamentD << <gels[ig]->m_gridDim_1, gels[ig]->m_blockDim, 0, gels[ig]->m_gel_stream >> > (gels[ig]->m_dvn, gels[ig]->m_dun, gels[ig]->m_dfilament, (solverIterations / 1000) % 1000, gels[ig]->d_hitCnt, gels[ig]->m_params);


                        }
                    }

                    // 5) 写文件（保留原逻辑）
                    for (int ig = 0; ig < ng; ++ig) {
                        if (solverIterations % 100000 == 0) {
                            if (gels[ig]->m_file_writer_thread.joinable())
                                gels[ig]->m_file_writer_thread.join();
                            gels[ig]->copyDataToHost();
                            double current_time = solverIterations * gels[ig]->m_params.dt;
                            gels[ig]->m_file_writer_thread =
                                std::thread(&GelSystem::writeFiles, gels[ig].get(), ig, current_time);
                        }
                    }
                    if (solverIterations % 100000 == 0) {
                        if (bottomGrid.m_gridfile_writer_thread.joinable())
                            bottomGrid.m_gridfile_writer_thread.join();
                        CUDA_OK(cudaStreamSynchronize(bottomGrid.m_grid_stream));
                        bottomGrid.copygridDataToHost();
                        double sim_time = solverIterations * bottomGrid.m_dt;
                        bottomGrid.m_gridfile_writer_thread =
                            std::thread(&BottomgridSystem::writebottomgridFiles, &bottomGrid, sim_time);
                    }
                } // end time loop

                // 强制最终保存（用于对比）
                for (int ig = 0; ig < ng; ++ig) {
                    if (gels[ig]->m_file_writer_thread.joinable())
                        gels[ig]->m_file_writer_thread.join();
                    CUDA_OK(cudaStreamSynchronize(gels[ig]->m_gel_stream));
                    gels[ig]->copyDataToHost();
                    // Bug fix: after swap, m_drnn has the latest node positions.
                    // Override the rn copied by copyDataToHost() (which used m_drn=old).
                    size_t rn_bytes = sizeof(double2) * (size_t)((gels[ig]->m_params.LX + 2) * (gels[ig]->m_params.LY + 2));
                    CUDA_OK(cudaMemcpy(gels[ig]->m_hrn, gels[ig]->m_drnn, rn_bytes, cudaMemcpyDeviceToHost));
                    gels[ig]->recordData(ig, 1);
                }

                // ----------------释放资源 ----------------
                for (int ig = 0; ig < ng; ++ig) CUDA_OK(cudaEventDestroy(ev_gel_ready[ig]));
                CUDA_OK(cudaFree(d_gels_rn));
                CUDA_OK(cudaFree(d_gels_rm));
                CUDA_OK(cudaFree(d_gels_um));
                CUDA_OK(cudaFree(d_params));
                // Steering feedback cleanup
                CUDA_OK(cudaFree(d_Mlr));
                delete[] h_rm_buf;

                chdir(originalDirectory);
                count++;
            }
        }
    }

    return 0;
}
