#ifndef LIBOTTOMGRIDPARAMS_H
#define LIBOTTOMGRIDPARAMS_H

#include <thread>
#include <cuda_runtime.h>
//#include < vector_types.h >

// simulation parameters

struct SimgridParams {
	int dlx;
	int dly;
	int bottomlx;
	int bottomly;
	int2 rngrid_offset[9];
	int2 ungrid_offset_noflux[9];
	int2 ungrid_offset_periodic[9];
	int TargetWave_y;
	double ind;
};

#endif
