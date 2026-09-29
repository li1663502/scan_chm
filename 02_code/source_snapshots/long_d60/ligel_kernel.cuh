#ifndef LIGEL_KERNEL_CUH
#define LIGEL_KERNEL_CUH

#include "libottomgridParams.h" 
#include "ligelParams.h"

__global__ void calElementsvolumefraction(double* wm, double* wmo, double* wn, double2* rn, SimParams params);

__global__ void calNodesvolumefraction(double* wm, double* wn, double2* rn, SimParams params);

__global__ void calPressureD(double* pm, double* vm, double* wm,
    const double* __restrict__ M_at_elem, SimParams params);

// [FchemoCoupling] Sample substrate memory M at each gel element's center
// position. Writes M_at_elem[gi] for each interior element. Used by calPressureD
// to add the substrate-M osmotic coupling term  +CHM * w * M  to pm.
__global__ void calMatElem(double* __restrict__ M_at_elem,
    const double2* __restrict__ rm,
    const double* __restrict__ ungrid,
    SimParams params, SimgridParams gridparams);

__global__ void calNodesVelocityD(const double2* __restrict__ r_prev, double2* __restrict__ r_next, double2* __restrict__ Veln, double2* __restrict__ Veln0, double2* __restrict__ Fn,
	const double* __restrict__ pm, const double* __restrict__ wm, const double2* __restrict__ FFRepulsion1, const double2* __restrict__ FFRepulsion2, SimParams params);

__global__ void calGelBoundaryNodesPositionD(double2* rn, int* map_node, SimParams params);

__global__ void calGelElementsPositionD(double2* rn, double2* rm, SimParams params);

__global__ void calTermsD(double* jsu, double* jsv, double* jsp, double* wm, double* wmo, double2* rn, double2* Veln, double* wn, double* un, double* um_norm, double* um, double* vm_norm, double* vm, double2* rm, SimParams params);

__global__ void calChemD1(double* vm_norm, double* um_norm, double* vm, double* um, double* wm, double* jsu, double* jsv, double* jsp, double2* rm, SimParams params);

__global__ void calChemD2(double* vm_norm, double* um_norm, double* vm, double* um, double* wm, double* jsu, double* jsv, double* jsp, double2* rm, SimParams params);

__global__ void calChemBoundaryD(double* um, double2* rm, double* wm, int* map_element,
    double* ungrid, SimParams params, SimgridParams gridparams);

__global__ void applySteeringPulseD(double* um, double2* rm, int* map_element,
    SimParams params,
    double2 Rc_g, double2 ehat_g, double2 nhat_g,
    double steer_delay_g, double k_pulse);

__global__ void calClearLab(int* lab, const double* ungrid, double* ungriddiffusion, SimgridParams gp, double rho_mem);

__global__ void calSampleUngrid(const double* ungrid, SimgridParams gridparams,
    double px_left, double py_left, double px_right, double py_right,
    double* M_left_out, double* M_right_out);

__global__ void calHolepoint(const double2* const* __restrict__ all_rm,
    const SimParams* __restrict__ all_params,
    int ng,
    int* __restrict__ lab,
    SimgridParams gridparams,
    int maxC,
    int sortN);

__global__ void calgridBoundary(double* __restrict__ ungrid, const int* __restrict__ gridmap_node, SimgridParams gridparams);

__global__ void calUVnnormD(double* un, double* um, double* vm, double* vn, double* wm, double* wmo, SimParams params);

__global__ void applyPulseBC(double* um, SimParams params);

__global__ void calrecordCenterElementD(double* um_center, double* vm_center, double* wm_center, double2* rm_center, double2* Fn_center, double2* Veln_center, double* um, double* vm, double* wm, double2* rn, double2* Fn, double2* Veln, int time, SimParams params);

__global__ void calGridInterpOnly(double* __restrict__ ungriddiffusion, const double2* const* __restrict__ all_rm, const double* const* __restrict__ all_um, const SimParams* __restrict__ all_params,
	int ng, const double* __restrict__ m_locgridx, const double* __restrict__ m_locgridy, int* __restrict__ lab, SimgridParams gridparams);

__global__ void calGridDiffuseOnly(double* __restrict__ ungrid, const double* __restrict__ ungriddiffusion, const int* __restrict__ lab, SimgridParams gridparams, double dt);

__global__ void calRepulsion(double2* __restrict__ FFRepulsion1, double2* __restrict__ FFRepulsion2, const double2* __restrict__ my_gel_rn,
	double2* const* __restrict__ gels_rn, int ng, int LX, int LY, SimParams params, SimgridParams gridparams);

__global__ void calFilamentD(double* vn, double* un, double2* filament, int time, unsigned int* hitCnt, SimParams params);



#endif