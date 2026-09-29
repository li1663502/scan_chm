#include <assert.h>
#include <memory.h>
#include <cmath>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <thread>
#include <functional>
#include <cstdlib>   // [Exp1] for std::getenv / atof
#include <cuda_runtime.h>
#include <algorithm>

#include "ligel_kernel.cuh"
#include"libottomgridSystem.h"


extern __device__ double* ungrid;
extern __device__ SimgridParams gridparams;


void BottomgridSystem::allocategridHostStorage()
{

	//chemical variables
	m_hungrid = new double[m_numBottomgridNodes];
	memset(m_hungrid, 0, m_numBottomgridNodes * sizeof(double));

	//dynamics variables
	m_hrngrid = new double2[m_numBottomgridNodes];
	memset(m_hrngrid, 0, m_numBottomgridNodes * sizeof(double2));

	m_hrmgrid = new double2[m_numBottomgridElements];
	memset(m_hrmgrid, 0, m_numBottomgridElements * sizeof(double2));

	m_hgridmap_element = new int[m_numBottomgridElements];
	memset(m_hgridmap_element, 0, m_numBottomgridElements * sizeof(int));

	m_hgridmap_node = new int[m_numBottomgridNodes];
	memset(m_hgridmap_node, 0, m_numBottomgridNodes * sizeof(int));

	m_hlocgridx = new double[2 * m_bottomgridSize.x + 1];
	m_hlocgridy = new double[2 * m_bottomgridSize.y + 1];
	memset(m_hlocgridx, 0, (2 * m_bottomgridSize.x + 1) * sizeof(double));
	memset(m_hlocgridy, 0, (2 * m_bottomgridSize.y + 1) * sizeof(double));

	assert(m_numBottomgridNodes > 0 && "Grid node count must be positive!");
	assert(m_numBottomgridElements > 0 && "Grid element count must be positive!");

	std::cout << "[allocateBottomGridHostStorage] EulerNodes: " << m_numBottomgridNodes << ", EulerElements: " << m_numBottomgridElements << std::endl;

}

void BottomgridSystem::allocategridDeviceStorage()
{
	//chemical variables
	cudaMalloc((void**)&m_dungrid, m_numBottomgridNodes * sizeof(double));
	cudaMalloc((void**)&m_dungriddiffusion, m_numBottomgridNodes * sizeof(double));
	//dynamics variables
	cudaMalloc((void**)&m_drngrid, m_numBottomgridNodes * sizeof(double2));
	cudaMalloc((void**)&m_drmgrid, m_numBottomgridElements * sizeof(double2));
	cudaMalloc((void**)&m_dlocgridx, (2 * m_bottomgridSize.x + 1) * sizeof(double));
	cudaMalloc((void**)&m_dlocgridy, (2 * m_bottomgridSize.y + 1) * sizeof(double));

	cudaMalloc((void**)&m_dgridmap_element, m_numBottomgridElements * sizeof(int));
	cudaMalloc((void**)&m_dgridmap_node, m_numBottomgridNodes * sizeof(int));
	cudaMalloc((void**)&m_dlab, (1001 * 1001) * sizeof(int));
}

int BottomgridSystem::get_gnindex(int xi, int yi, int gsize)//0-1,0元素，1点
{
	int multind = 2;
	return xi + yi * (multind * m_bottomgridSize.x + gsize);
}

void BottomgridSystem::setgridInitValue()
{
	// 初始化化学变量的值
	double ind = 0.5;
	multind = int(1 / ind);
	for (int yi = 0; yi < multind * m_bottomgridSize.y + 1; yi++) { // 遍历y轴方向，从0到1000
		for (int xi = 0; xi < multind * m_bottomgridSize.x + 1; xi++) { // 遍历x轴方向，从0到1000
			int ggi = get_gnindex(xi, yi, 1); // 调用get_index函数获取当前点的索引，这里的1可能表示某种类型的索引
			m_hungrid[ggi] = 0; // 随机初始化m_hum数组的值
		}
	}

#if 0  // [Exp2] disable static Gaussian injection; substrate M evolves from 0 (gel-deposited trail)
	// [Exp1] Static Gaussian M field injection (overrides the zero-fill above).
	// M(x,y) = M0 * exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
	// cx/cy/sigma are PHYSICAL coords (same scale as m_hlocgridx = xi*ind).
	// Tunable via environment:
	//   GAUSS_M0 (default 0.3), GAUSS_CX (200), GAUSS_CY (200), GAUSS_SIGMA (30)
	{
		double M0    = std::getenv("GAUSS_M0")    ? atof(std::getenv("GAUSS_M0"))    : 0.3;
		double cx    = std::getenv("GAUSS_CX")    ? atof(std::getenv("GAUSS_CX"))    : 200.0;
		double cy    = std::getenv("GAUSS_CY")    ? atof(std::getenv("GAUSS_CY"))    : 200.0;
		double sigma = std::getenv("GAUSS_SIGMA") ? atof(std::getenv("GAUSS_SIGMA")) : 30.0;
		printf("[Exp1] Injecting static Gaussian M: M0=%.3f, center=(%.1f,%.1f), sigma=%.1f\n",
		       M0, cx, cy, sigma);
		for (int yi = 0; yi < multind * m_bottomgridSize.y + 1; yi++) {
			for (int xi = 0; xi < multind * m_bottomgridSize.x + 1; xi++) {
				double px = xi * ind;   // same physical mapping as m_hlocgridx
				double py = yi * ind;
				double r2 = (px - cx) * (px - cx) + (py - cy) * (py - cy);
				int ggi = get_gnindex(xi, yi, 1);
				m_hungrid[ggi] = M0 * exp(-r2 / (2.0 * sigma * sigma));
			}
		}
	}
#endif  // [Exp2] end disable static Gaussian injection

	for (int xi = 1; xi < multind * m_bottomgridSize.x + 1; xi++) {
		m_hlocgridx[xi] = xi * ind;
	}
	for (int yi = 1; yi < multind * m_bottomgridSize.y + 1; yi++) {
		m_hlocgridy[yi] = yi * ind;
	}

	setgridType(m_hgridmap_node, 1);
	setgridType(m_hgridmap_element, 0);
}

void BottomgridSystem::setgridGoonValue(int time)
{
	ifstream frngrid, fungrid, flocgridx, flocgridy;//定义了四个输入文件流（ifstream）和四个字符串，用于存储要打开的数据文件的路径
	string str_rngrid, str_ungrid, str_locgridx, str_locgridy;
	//str_rngrid = "rngrid" + to_string(time) + ".dat";//文件名"um" + 时间 + ".dat"
	str_ungrid = "ungrid" + to_string(time) + ".dat";
	//str_locgridx = "locgridx" + to_string(time) + ".dat";
	//str_locgridy = "locgridy" + to_string(time) + ".dat";
	//frngrid.open(str_rngrid, ios::in);//打开四个数据文件
	fungrid.open(str_ungrid, ios::in);
	//flocgridx.open(str_locgridx, ios::in);
	//flocgridy.open(str_locgridy, ios::in);

	string line;
	for (int yi = 0; yi < multind * m_bottomgridSize.y + 1; yi++) {
		for (int xi = 0; xi < multind * m_bottomgridSize.x + 1; xi++) {
			int ggi = get_gnindex(xi, yi, 1);
			double ungrid;
			getline(fungrid, line);
			istringstream iss1(line);
			iss1 >> ungrid;
			m_hungrid[ggi] = ungrid;

		}
	}

	for (int xi = 1; xi < multind * m_bottomgridSize.x + 1; xi++) {
		m_hlocgridx[xi] = xi * m_gridparams.ind;
	}
	for (int yi = 1; yi < multind * m_bottomgridSize.y + 1; yi++) {
		m_hlocgridy[yi] = yi * m_gridparams.ind;
	}

	/*for (int xi = 0; xi < multind * m_bottomgridSize.x + 1; ++xi) {
		double locgridx;
		getline(flocgridx, line);
		istringstream iss2(line);
		iss2 >> locgridx;
		m_hlocgridx[xi] = locgridx;
	}
	for (int yi = 0; yi < multind * m_bottomgridSize.y + 1; ++yi) {
		double locgridy;
		getline(flocgridy, line);
		istringstream iss3(line);
		iss3 >> locgridy;
		m_hlocgridy[yi] = locgridy;
	}*/


	/*for (int yi = 0; yi < multind * m_bottomgridSize.y + 1; yi++) {
		for (int xi = 0; xi < multind * m_bottomgridSize.x + 1; xi++) {
			int ggi = get_gnindex(xi, yi, 1);
			double rngridx, rngridy;
			getline(frngrid, line);
			istringstream iss2(line);
			iss2 >> rngridx >> rngridy;
			m_hrngrid[ggi] = make_double2(rngridx, rngridy);
		}
	}*/
	//frngrid.close();
	fungrid.close();
	//flocgridx.close();
	//flocgridy.close();


	setgridType(m_hgridmap_node, 1);
	setgridType(m_hgridmap_element, 0);
}

void BottomgridSystem::setgridType(int* a, int gsize)
{//点0-10000，元素0-999
	int xi, yi;
	int DLX = multind * m_bottomgridSize.x + gsize - 1;//999,1000
	int DLY = multind * m_bottomgridSize.y + gsize - 1;
	for (xi = 1; xi < DLX; xi++) {//1-998,1-999      
		for (yi = 1; yi < DLY; yi++) {
			a[get_gnindex(xi, yi, gsize)] = 0;
		}
	}//除边界外其他位置

	for (xi = 1; xi < DLX; xi++) {
		a[get_gnindex(xi, 0, gsize)] = 1;//（1-998/999，0）
		a[get_gnindex(xi, DLY, gsize)] = 2;//（1-998/999，999/1000）
	}
	for (yi = 1; yi < DLY; yi++) {
		a[get_gnindex(0, yi, gsize)] = 3;//（0，1-998/999）
		a[get_gnindex(DLX, yi, gsize)] = 4;//（999/1000，1-998/999）
	}//4条棱

	a[get_gnindex(0, 0, gsize)] = 5;
	a[get_gnindex(0, DLY, gsize)] = 6;
	a[get_gnindex(DLX, 0, gsize)] = 7;
	a[get_gnindex(DLX, DLY, gsize)] = 8;
}//4个点

void BottomgridSystem::copygridDataToDevice()
{
	cudaMemcpy(m_dungrid, m_hungrid, sizeof(double) * m_numBottomgridNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_drngrid, m_hrngrid, sizeof(double2) * m_numBottomgridNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_drmgrid, m_hrmgrid, sizeof(double2) * m_numBottomgridElements, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dlocgridx, m_hlocgridx, sizeof(double) * static_cast<size_t>(2 * m_bottomgridSize.x + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(m_dlocgridy, m_hlocgridy, sizeof(double) * static_cast<size_t>(2 * m_bottomgridSize.y + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(m_dgridmap_element, m_hgridmap_element, sizeof(int) * m_numBottomgridElements, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dgridmap_node, m_hgridmap_node, sizeof(int) * m_numBottomgridNodes, cudaMemcpyHostToDevice);
}

void BottomgridSystem::copygridDataToHost()
{
	cudaMemcpyAsync(m_hungrid, m_dungrid, sizeof(double) * m_numBottomgridNodes, cudaMemcpyDeviceToHost, m_grid_stream);
	cudaMemcpyAsync(m_hrngrid, m_drngrid, sizeof(double2) * m_numBottomgridNodes, cudaMemcpyDeviceToHost, m_grid_stream);
	cudaMemcpyAsync(m_hrmgrid, m_drmgrid, sizeof(double2) * m_numBottomgridElements, cudaMemcpyDeviceToHost, m_grid_stream);
	cudaMemcpyAsync(m_hlocgridx, m_dlocgridx, sizeof(double) * static_cast<size_t>(2 * m_bottomgridSize.x + 1), cudaMemcpyDeviceToHost, m_grid_stream);
	cudaMemcpyAsync(m_hlocgridy, m_dlocgridy, sizeof(double) * static_cast<size_t>(2 * m_bottomgridSize.y + 1), cudaMemcpyDeviceToHost, m_grid_stream);
	cudaStreamSynchronize(m_grid_stream);//同步
}

void BottomgridSystem::freeHostgridMemory()
{
	delete[] m_hungrid;//释放m_hum指向的数组的内存
	delete[] m_hrngrid;
	delete[] m_hrmgrid;
	delete[] m_hgridmap_element;
	delete[] m_hgridmap_node;
	delete[] m_hlocgridx;
	delete[] m_hlocgridy;


}

void BottomgridSystem::freeDevicegridMemory()//释放之前在GPU（设备）上分配的内存
{

	cudaFree(m_dungrid);//释放m_dum指向的GPU内存块
	cudaFree(m_dungriddiffusion);//
	cudaFree(m_drngrid);
	cudaFree(m_drmgrid);
	cudaFree(m_dgridmap_element);
	cudaFree(m_dgridmap_node);
	cudaFree(m_dlocgridx);
	cudaFree(m_dlocgridy);
	cudaFree(m_dlab);
}

void BottomgridSystem::recordbottomgridData(int time)
{
	if (time % 1 == 0) {
		std::ofstream fungrid, flocgridx, flocgridy;
		fungrid.open("ungrid" + std::to_string(time) + ".dat",
			std::ios::out | std::ios::trunc);
		/*	flocgridx.open("locgridx" + std::to_string(time) + ".dat",
				std::ios::out | std::ios::trunc);
			flocgridy.open("locgridy" + std::to_string(time) + ".dat",
				std::ios::out | std::ios::trunc);*/

		fungrid.setf(std::ios::fixed);
		//flocgridx.setf(std::ios::fixed);
		//flocgridy.setf(std::ios::fixed);

		fungrid << std::setprecision(12);
		//	flocgridx << std::setprecision(12);
			//flocgridy << std::setprecision(12);

		for (int yi = 0; yi < multind * m_bottomgridSize.y + 1; ++yi) {
			for (int xi = 0; xi < multind * m_bottomgridSize.x + 1; ++xi) {
				int ggi = get_gnindex(xi, yi, 1);
				fungrid << m_hungrid[ggi] << '\n';
			}
		}

		/*for (int xi = 0; xi < multind * m_bottomgridSize.x + 1; ++xi)
			flocgridx << m_hlocgridx[xi] << '\n';

		for (int yi = 0; yi < multind * m_bottomgridSize.y + 1; ++yi)
			flocgridy << m_hlocgridy[yi] << '\n';*/
	}
}


void BottomgridSystem::writebottomgridFiles(double time)
{
	if (int(time) % 1000 == 0) {
		recordbottomgridData(time);
	}
}

void BottomgridSystem::configureGridDimensions() {
	m_gblockDim = dim3(16, 16);  // 固定每块 16x16 个线程
	//m_ggridDim=dim3(63, 63);
	m_ggridDim.x = (2 * m_bottomgridSize.x + 1 + m_gblockDim.x - 1) / m_gblockDim.x;//63//32
	m_ggridDim.y = (2 * m_bottomgridSize.y + 1 + m_gblockDim.y - 1) / m_gblockDim.y;
}


BottomgridSystem::BottomgridSystem(int2 bottomgridSize, int time) :
	m_gridInitialized(false),
	m_bottomgridSize(bottomgridSize),
	//CPU data
	m_hungrid(0),
	m_hungriddiffusion(0),
	m_hrngrid(0),
	m_hrmgrid(0),
	m_hlocgrid(0),
	m_hgridmap_node(0),
	m_hgridmap_element(0),
	m_hlocgridx(0),
	m_hlocgridy(0),

	//GPU data
	m_dungrid(0),
	m_dungriddiffusion(0),
	m_dlocgridx(0),
	m_dlocgridy(0),
	m_drngrid(0),
	m_drmgrid(0),
	m_dlocgrid(0),
	m_dlab(0),
	m_dgridmap_node(0),
	m_dgridmap_element(0)
{
	m_dt = 0.001;//e-3; //时间步长
	m_df = int(1 / m_dt);//计数次数
	m_gridparams.ind = 0.5;
	multind = int(1 / m_gridparams.ind);

	int totalX = multind * m_bottomgridSize.x;
	int totalY = multind * m_bottomgridSize.y;

	m_gridparams.dlx = multind * m_bottomgridSize.x;//1000
	m_gridparams.dly = multind * m_bottomgridSize.y;

	m_numBottomgridElements = (multind * m_bottomgridSize.x) * (multind * m_bottomgridSize.y);//凝胶元素数量1000*1000
	m_numBottomgridNodes = (multind * m_bottomgridSize.x + 1) * (multind * m_bottomgridSize.y + 1);//网格节点数量1001*1001

	m_gblockDim = dim3(16, 16);//8x8x8的线程块

	m_ggridDim.x = (m_gridparams.dlx + 1 + m_gblockDim.x - 1) / m_gblockDim.x;//63
	m_ggridDim.y = (m_gridparams.dly + 1 + m_gblockDim.y - 1) / m_gblockDim.y;//
	//Holpoint
	int cols = m_gridparams.dlx + 1;
	//HolblockDim = 128;                      // blockDim.x
	HolblockDim = 1;
	HolgridDim = cols;                   // gridDim.x = dlx + 1
	smem_bytes =                  // 动态共享内存大小
		sizeof(double) * 2048 +          // MAX_Y
		sizeof(double2) * 2048;          // MAX_C
	//*****************
	//configureGridDimensions();

	GLX_ = multind * m_bottomgridSize.x - 2;
	GLY_ = multind * m_bottomgridSize.y - 2;

	m_gridparams.bottomlx = m_bottomgridSize.x;
	m_gridparams.bottomly = m_bottomgridSize.y;

	m_gridparams.rngrid_offset[0] = { 0, 0 }; m_gridparams.ungrid_offset_noflux[0] = { 0, 0 }; m_gridparams.ungrid_offset_periodic[0] = { 0, 0 };//两种边界条件：零流边界和周期性边界，凝胶中心点
	//4条棱
	m_gridparams.rngrid_offset[1] = { 0, 1 }; m_gridparams.ungrid_offset_noflux[1] = { 0, 1 }; m_gridparams.ungrid_offset_periodic[1] = { 0, GLY_ };
	m_gridparams.rngrid_offset[2] = { 0, -1 }; m_gridparams.ungrid_offset_noflux[2] = { 0, -1 }; m_gridparams.ungrid_offset_periodic[2] = { 0, -GLY_ };
	m_gridparams.rngrid_offset[3] = { 1, 0 }; m_gridparams.ungrid_offset_noflux[3] = { 1, 0 }; m_gridparams.ungrid_offset_periodic[3] = { GLX_, 0 };
	m_gridparams.rngrid_offset[4] = { -1, 0 }; m_gridparams.ungrid_offset_noflux[4] = { -1, 0 }; m_gridparams.ungrid_offset_periodic[4] = { -GLX_, 0 };
	//4个点
	m_gridparams.rngrid_offset[5] = { 1, 1 }; m_gridparams.ungrid_offset_noflux[5] = { 1, 1 }; m_gridparams.ungrid_offset_periodic[5] = { GLX_, GLY_ };
	m_gridparams.rngrid_offset[6] = { 1, -1 }; m_gridparams.ungrid_offset_noflux[6] = { 1, -1 }; m_gridparams.ungrid_offset_periodic[6] = { GLX_, -GLY_ };
	m_gridparams.rngrid_offset[7] = { -1, 1 }; m_gridparams.ungrid_offset_noflux[7] = { -1, 1 }; m_gridparams.ungrid_offset_periodic[7] = { -GLX_, GLY_ };
	m_gridparams.rngrid_offset[8] = { -1, -1 }; m_gridparams.ungrid_offset_noflux[8] = { -1, -1 }; m_gridparams.ungrid_offset_periodic[8] = { -GLX_, -GLY_ };


	_gridinitialize(time);
}

BottomgridSystem::~BottomgridSystem()// 定义类的析构函数  
{
	_gridfinalize();// 调用_finalize函数，可能用于释放资源或执行清理工作  
}

void BottomgridSystem::_gridinitialize(int time)
{
	assert(!m_gridInitialized);// 断言，确保m_bInitialized为false，即对象尚未初始化

	allocategridHostStorage();
	allocategridDeviceStorage();
	if (time)
		setgridGoonValue(time);
	else
		setgridInitValue();
	copygridDataToDevice();
	//cudaMemcpyToSymbol(ungrid, &m_dungrid, sizeof(double*));
	//cudaMemcpyToSymbol(gridparams, &m_gridparams, sizeof(SimgridParams));
	cudaDeviceSynchronize();

	cudaStreamCreate(&m_grid_stream);

	m_gridInitialized = true;
	//printf("[Host Check] m_gridparams.dlx = %d, m_gridparams.dly = %d\n", m_gridparams.dlx, m_gridparams.dly);


}

void BottomgridSystem::updategrid(long long int gridIterations)//迭代次数
{
	assert(m_gridInitialized);
	double time = gridIterations * m_dt;
	if (gridIterations % 5 == 0) {
		if (m_gridfile_writer_thread.joinable()) {// 如果文件写入线程是可连接的（即正在运行）  
			m_gridfile_writer_thread.join();// 等待文件写入线程完成  
		}
		copygridDataToHost();
		m_gridfile_writer_thread = thread(mem_fn(&BottomgridSystem::writebottomgridFiles), this, time);// 创建一个新的线程来写入文件，并将当前时间作为参数传递
	}

}

void BottomgridSystem::_gridfinalize()//释放内存
{
	assert(m_gridInitialized);
	cudaStreamDestroy(m_grid_stream);
	if (m_gridfile_writer_thread.joinable()) {//程序能执行
		m_gridfile_writer_thread.join();
	}
	freeHostgridMemory();
	freeDevicegridMemory();
}

static inline int nextPow2(int x) {
	int p = 1;
	while (p < x) p <<= 1;
	return p;
}

void BottomgridSystem::configureHolepointAuto(const SimParams* h_params, int ng)
{
	// 1) 计算所有 gel 的 maxC
	int maxC = 0;
	for (int ig = 0; ig < ng; ++ig) {
		const int LX = h_params[ig].LX;
		const int LY = h_params[ig].LY;
		const int C = 2 * (LX + LY) - 8;
		maxC = std::max(maxC, C);
	}
	if (maxC < 2) maxC = 2;

	// 2) sortN 取 >=maxC 的 2 次幂（bitonic 用）
	int sortN = nextPow2(maxC);

	// 3) 动态共享内存大小：ys[sortN] + 对齐 + cirL[maxC]
	size_t smem = (size_t)sortN * sizeof(double)
		+ (size_t)maxC * sizeof(double2)
		+ 16; // 给 double2 对齐余量

	// 4) 自动选 blockDim（简单稳健：256，别让它超过设备上限）
	cudaDeviceProp prop{};
	cudaGetDeviceProperties(&prop, 0);
	int block = 256;
	block = std::min(block, prop.maxThreadsPerBlock);
	if (block < 64) block = 64; // 太小反而慢

	// 5) 一列一个 block：gridDim = dlx+1
	int cols = m_gridparams.dlx + 1;

	Hol_maxC = maxC;
	Hol_sortN = sortN;
	HolblockDim = block;
	HolgridDim = cols;
	smem_bytes = smem;

	// （可选）如果你担心共享内存上限，可加保护：
	// if (smem_bytes > prop.sharedMemPerBlock) { ...fallback... }
}