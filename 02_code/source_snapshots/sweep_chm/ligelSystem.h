#ifndef LIGELSYSTEM_H
#define LIGELSYSTEM_H

#include <thread>
#include <cuda_runtime.h>

#include "ligelParams.h"
#include "ligel_kernel.cuh"
//#include"libottomgridParams.h"

using namespace std;
//Gel system class
class GelSystem {
public:
	GelSystem(int ng, int ig, int2 gelSize, int time, int i, double j, double k, double startX, double startY);
	~GelSystem();
	void _initialize(int time, double startX, double startY);
	// [FchemoCoupling] update now also receives the substrate ungrid pointer
	// and grid params so calMatElem can sample M at element positions before
	// calPressureD adds the CHM * w * M osmotic-coupling term to pm.
	void update(long long int solverIterations,
		const double* d_ungrid, const SimgridParams& gridparams);
	void _finalize();
	void writeFiles(int ig, double time);
	void allocateHostStorage();
	void allocateDeviceStorage();
	void copyDataToDevice();
	void copyDataToHost();
	void freeHostMemory();
	void freeDeviceMemory();
	// �������б���
	cudaStream_t m_gel_stream;
	cudaEvent_t  m_ev_after_update;  // <<< �����¼�


protected:  // methods
	void setInitValue(int ig, double startX, double startY);
	void setGoonValue(int ig, int time);
	double fu_h(double u, double v, double w, double phi);
	void steadyStateValue(double& u, double& v, double& w, double phi);
	int get_index(int xi, int yi, int size);
	void setType(int* a, int size);
public:
	void recordData(int ig, int time);
	void recordmotionData(int ig, int time);
	//void recordSpiralWave(int time);
	void recordCenterElement(int ig, int time);//&&&&&&&&&&&  double time
	void setChemicalWave(int type);


protected:  // data
	// CPU data
	//chemical variables


	//dynamics variables
	double2* m_inihrncenter;

	double2* m_hcir;
	double2* m_hinterpa;
	double2* m_hedge;


	int* m_hmap_node;
	int* m_hmap_element;


	double m_htime = 0;

	// GPU data
	//chemical variables

	double* m_dun_norm;
	double* m_dvn_norm;

	double* m_dtime;


	double* m_dwmp;

	//dynamics variables
	double2* m_inidrncenter;

	double2* m_drm_loc;
	double2* m_dnmSm;
	double* m_dVolm;
	//double* m_dPrem;


	int* m_dmap_node;
	//int* m_dmap_element;

public:
	// params
	double* m_dPrem;
	double* m_dM_at_elem;   // [FchemoCoupling] substrate-M sampled at each element center
	int* m_dmap_element;
	double* m_djsum;
	double* m_djsvm;
	double* m_djspm;
	double* m_dvm_norm;

	double* m_hum;
	double* m_hvm;
	double* m_hun;
	double* m_hvn;
	double* m_hwm;
	double* m_hwn;
	double* m_hwmo;
	double* m_hum_center;
	double* m_hvm_center;
	double* m_hwm_center;
	double2* m_hrm_center;
	double2* m_hFn_center;
	double2* m_hVeln_center;
	double2* m_hVeln;
	double2* m_hVeln0;
	double2* m_hFn;
	double2* m_hfilament;


	double* m_dwm;
	double* m_dwn;
	double* m_dwmo;
	double* m_dvm;
	double* m_dun;
	double* m_dvn;
	double* m_dum;
	double2* m_dVeln;
	double2* m_dVeln0;
	double2* m_dFn;
	double* m_dum_center;
	double* m_dvm_center;
	double* m_dwm_center;
	double2* m_drm_center;
	double2* m_dFn_center;
	double2* m_dVeln_center;

	double2* m_dinterpa;
	double2* m_dedge;
	double* m_dedge_y_values;
	double* m_dedge_x_values;
	double2* m_dcir;
	double* m_dxx;
	unsigned int* d_hitCnt;
	double2* m_dfilament;


	int m_ng;
	int m_ig;
	bool m_bInitialized;
	int2 m_gelSize;
	double2* m_hFFRepulsion;
	double2* m_hFFRepulsion1;
	double2* m_hFFRepulsion2;
	double2* m_dFFRepulsion;
	double2* m_dFFRepulsion1;
	double2* m_dFFRepulsion2;
	double2* m_drn;
	double2* m_drnn;
	double2* m_hrn;
	double2* m_hrnn;
	double2* m_hrm;
	double* m_hum_norm;


	double2* m_drm;
	double* m_dum_norm;
	//int* m_dlab;
	double m_dt;
	int m_df;
	int m_numGelElements;
	int m_numGelNodes;
	SimParams m_params;
	//cudaStream_t m_gel_stream;
	thread m_file_writer_thread;
	dim3 m_blockDim;
	dim3 m_gridDim_1;
	dim3 m_gridDim0;
	dim3 m_gridDim1;
	dim3 m_gridDim2;
	dim3 m_gridDim3;
	bool flag = false;
	bool result = true;
	int m_count;
	double ep[30];
	double f[5];
	double CHS[9];
	double m_inirmcenter;

};

#endif  // __GELSYSTEM_H__