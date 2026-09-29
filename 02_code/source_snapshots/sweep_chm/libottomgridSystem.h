#ifndef LIBOTTOMGRIDSYSTEM_H
#define LIBOTTOMGRIDSYSTEM_H

#include <thread>
#include <cuda_runtime.h>
//#include <vector>

#include "libottomgridParams.h"
#include "ligel_kernel.cuh"


using namespace std;
//Bottomgrid system class
class BottomgridSystem {
public:
	BottomgridSystem(int2 bottomgridSize, int time);
	~BottomgridSystem();
	void _gridinitialize(int time);
	void updategrid(long long int solverIterations);
	void _gridfinalize();
	void writebottomgridFiles(double time);
	void copygridDataToDevice();
	void copygridDataToHost();
	// libottomgridSystem.h 里 class BottomgridSystem public: 增加
	void configureHolepointAuto(const SimParams* h_params, int ng);



protected:  // methods
	void allocategridHostStorage();
	void allocategridDeviceStorage();
	void freeHostgridMemory();
	void freeDevicegridMemory();
	void setgridInitValue();
	void setgridGoonValue(int time);
	int get_gnindex(int xi, int yi, int size);
	void setgridType(int* a, int size);
	void recordbottomgridData(int time);
	void configureGridDimensions();


protected:  // data
	// CPU data
	//chemical variables



	//dynamics variables
	double2* m_hrngrid;
	double2* m_hrmgrid;
	double2* m_hlocgrid;

	//double2* m_hlocgrid;

	int* m_hgridmap_node;
	int* m_hgridmap_element;
	// GPU data
	//chemical variables

	double2* m_dlocgrid;

	//dynamics variables
	double2* m_drngrid;
	double2* m_drmgrid;


public:
	// params
	bool m_gridInitialized;
	int2 m_bottomgridSize;
	double* m_dungrid;
	double* m_dungriddiffusion;
	double* m_hungrid;
	double* m_hungriddiffusion;
	double* m_dlocgridx;
	double* m_dlocgridy;
	double* m_hlocgridx;
	double* m_hlocgridy;
	int* m_dlab;
	double m_dt;
	double m_df;
	//double ind;
	int multind;
	double* m_sizelocgridx;
	double* m_sizelocgridy;
	int GLX_;
	int GLY_;
	//int m_numGelElements;
	int m_numBottomgridNodes;
	int m_numBottomgridElements;

	SimgridParams m_gridparams;
	cudaStream_t m_grid_stream;
	thread m_gridfile_writer_thread;
	dim3 m_gblockDim;
	dim3 m_ggridDim;

	int HolblockDim;
	int HolgridDim;
	size_t smem_bytes;

	bool flag = false;
	bool result = true;
	int m_count;
	int* m_dgridmap_node;
	int* m_dgridmap_element;

	// 并在 public: 数据区增加
	int Hol_maxC = 0;
	int Hol_sortN = 0;


};

#endif  // __BOTTOMGRIDSYSTEM_H__




