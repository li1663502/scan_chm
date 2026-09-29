#include <assert.h>
#include <memory.h>
#include <cmath>
#include <algorithm>
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

#include "ligel_kernel.cuh"
#include"ligelSystem.h"
//#include "ligel_kernel.cu"
//#include "ligelParams.h"


void GelSystem::allocateHostStorage()
{

	//chemical variables
	m_hum = new double[m_numGelElements];
	memset(m_hum, 0, m_numGelElements * sizeof(double));

	m_hun = new double[m_numGelNodes];
	memset(m_hun, 0, m_numGelNodes * sizeof(double));

	m_hum_norm = new double[m_numGelElements];
	memset(m_hum_norm, 0, m_numGelElements * sizeof(double));

	m_hvm = new double[m_numGelElements];
	memset(m_hvm, 0, m_numGelElements * sizeof(double));

	m_hvn = new double[m_numGelNodes];
	memset(m_hvn, 0, m_numGelNodes * sizeof(double));

	m_hwm = new double[m_numGelElements];
	memset(m_hwm, 0, m_numGelElements * sizeof(double));

	m_hwn = new double[m_numGelNodes];
	memset(m_hwn, 0, m_numGelNodes * sizeof(double));

	m_hwmo = new double[m_numGelElements];
	memset(m_hwmo, 0, m_numGelElements * sizeof(double));

	//dynamics variables
	m_hrn = new double2[m_numGelNodes];
	memset(m_hrn, 0, m_numGelNodes * sizeof(double2));

	m_hrnn = new double2[m_numGelNodes];
	memset(m_hrnn, 0, m_numGelNodes * sizeof(double2));


	m_hrm = new double2[m_numGelElements];
	memset(m_hrm, 0, m_numGelElements * sizeof(double2));

	m_hFn = new double2[m_numGelNodes];
	memset(m_hFn, 0, m_numGelNodes * sizeof(double2));


	m_hFFRepulsion1 = new double2[m_numGelNodes];
	memset(m_hFFRepulsion1, 0, m_numGelNodes * sizeof(double2));
	m_hFFRepulsion2 = new double2[m_numGelNodes];
	memset(m_hFFRepulsion2, 0, m_numGelNodes * sizeof(double2));


	m_hVeln = new double2[m_numGelNodes];
	memset(m_hVeln, 0, m_numGelNodes * sizeof(double2));

	m_hVeln0 = new double2[m_numGelNodes];
	memset(m_hVeln0, 0, m_numGelNodes * sizeof(double2));

	m_hmap_element = new int[m_numGelElements];
	memset(m_hmap_element, 0, m_numGelElements * sizeof(int));

	m_hmap_node = new int[m_numGelNodes];
	memset(m_hmap_node, 0, m_numGelNodes * sizeof(int));

	m_hvm_center = new double[1000];
	memset(m_hvm_center, 0, 1000 * sizeof(double));

	m_hum_center = new double[1000];
	memset(m_hum_center, 0, 1000 * sizeof(double));

	m_hwm_center = new double[1000];
	memset(m_hwm_center, 0, 1000 * sizeof(double));

	m_hrm_center = new double2[1000];
	memset(m_hrm_center, 0, 1000 * sizeof(double2));

	m_hFn_center = new double2[1000];
	memset(m_hFn_center, 0, 1000 * sizeof(double2));

	m_hVeln_center = new double2[1000];
	memset(m_hVeln_center, 0, 1000 * sizeof(double2));

	m_hfilament = new double2[1000];
	memset(m_hfilament, 0, 1000 * m_params.maxFilamentlen * sizeof(bool));

	std::cout << "[allocateGelGridHostStorage] GelNodes: " << m_numGelNodes << ", GelElements: " << m_numGelElements << std::endl;

}

void GelSystem::allocateDeviceStorage()
{
	//chemical variables
	cudaMalloc((void**)&m_dum, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dun, m_numGelNodes * sizeof(double));
	cudaMalloc((void**)&m_dum_norm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dun_norm, m_numGelNodes * sizeof(double));

	cudaMalloc((void**)&m_dvm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dvn, m_numGelNodes * sizeof(double));
	cudaMalloc((void**)&m_dvm_norm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dvn_norm, m_numGelNodes * sizeof(double));

	cudaMalloc((void**)&m_dwm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dwn, m_numGelNodes * sizeof(double));
	cudaMalloc((void**)&m_dwmo, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_drn, m_numGelNodes * sizeof(double2));
	cudaMalloc((void**)&m_drnn, m_numGelNodes * sizeof(double2));

	cudaMalloc((void**)&m_djsum, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_djsvm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_djspm, m_numGelElements * sizeof(double));
	//dynamics variables


	cudaMalloc((void**)&m_inidrncenter, m_numGelNodes * sizeof(double2));
	cudaMalloc((void**)&m_drm, m_numGelElements * sizeof(double2));

	cudaMalloc((void**)&m_dFn, m_numGelNodes * sizeof(double2));
	cudaMalloc((void**)&m_dVeln, m_numGelNodes * sizeof(double2));
	cudaMalloc((void**)&m_dVeln0, m_numGelNodes * sizeof(double2));


	//cudaMalloc((void**)&m_dFFRepulsion, m_numGelNodes * sizeof(double2));
	cudaMalloc((void**)&m_dFFRepulsion1, m_numGelNodes * sizeof(double2));
	cudaMalloc((void**)&m_dFFRepulsion2, m_numGelNodes * sizeof(double2));

	cudaMalloc((void**)&m_dPrem, m_numGelElements * sizeof(double));
	// [FchemoCoupling] allocate substrate-M sampling array (same layout as pm),
	// initialized to zero so a no-substrate run is identical to original Y-B.
	cudaMalloc((void**)&m_dM_at_elem, m_numGelElements * sizeof(double));
	cudaMemset(m_dM_at_elem, 0, m_numGelElements * sizeof(double));

	cudaMalloc((void**)&m_dmap_element, m_numGelElements * sizeof(int));
	cudaMalloc((void**)&m_dmap_node, m_numGelNodes * sizeof(int));

	cudaMalloc((void**)&m_dvm_center, 1000 * sizeof(double));
	cudaMalloc((void**)&m_dum_center, 1000 * sizeof(double));
	cudaMalloc((void**)&m_dwm_center, 1000 * sizeof(double));
	cudaMalloc((void**)&m_drm_center, 1000 * sizeof(double2));
	cudaMalloc((void**)&m_dFn_center, 1000 * sizeof(double2));
	cudaMalloc((void**)&m_dVeln_center, 1000 * sizeof(double2));

	cudaMalloc((void**)&m_dcir, (2 * (m_gelSize.x + m_gelSize.y) - 7) * sizeof(double2));
	cudaMalloc((void**)&m_dinterpa, static_cast<size_t>(2 * 4) * (m_gelSize.x + m_gelSize.y) * sizeof(double2));
	cudaMalloc((void**)&m_dedge, static_cast<size_t>(2 * 4) * (m_gelSize.x + m_gelSize.y) * sizeof(double2));
	cudaMalloc((void**)&m_dedge_y_values, static_cast<size_t>(2 * 4) * (m_gelSize.x + m_gelSize.y) * sizeof(double));
	cudaMalloc((void**)&m_dedge_x_values, static_cast<size_t>(2 * 4) * (m_gelSize.x + m_gelSize.y) * sizeof(double));
	cudaMalloc((void**)&m_dxx, static_cast<size_t>(2 * 4 * 2) * (m_gelSize.x + m_gelSize.y) * sizeof(double));

	cudaMalloc((void**)&m_dfilament, 1000 * m_params.maxFilamentlen * sizeof(double2));
	cudaMalloc((void**)&m_dtime, sizeof(double));
	cudaMalloc(&d_hitCnt, sizeof(unsigned int));

}

int GelSystem::get_index(int xi, int yi, int size)
{
	return xi + yi * (m_gelSize.x + size);
}

void GelSystem::setChemicalWave(int wave_type)
{
	int LX = m_gelSize.x;
	int LY = m_gelSize.y;
	double uss_max = m_params.uss;
	double vss_max = m_params.vss;
	double cuty = 0.4;
	double band = 0.2;
	int y0 = 1;                      // 底边起点
	int H = (int)(0.35 * LY);       // 激发矩形高度(按图改，比如0.2~0.5)

	int x0u = 15;                    // u 激发矩形离左边的起点(你原来就是15)
	int Wu = (int)(0.15 * LX);      // u 激发矩形宽度(按图改)

	int x0v = 20;                    // v 激发矩形离左边的起点(你原来就是20)
	int Wv = (int)(0.15 * LX);      // v 激发矩形宽度(按图改)

	switch (wave_type) {
	case 0:      //pulse waves initialization
		//for (int yi = 0; yi < 1; yi++) {
		for (int xi = 0; xi <= LX; xi++) {
			int gi = get_index(xi, 0, 1);
			m_hum[gi] = 5.0 * uss_max;
		}
		//}
		break;
		//case 1: { // 生成顺时针(CW)螺旋“种子”
		//	const int H = (int)(LY * (1 - cuty));   // 凝胶有效高度(沿用你原来的)
		//	const double PI = 3.141592653589793;

		//	// ===== 你要改的位置参数 =====
		//	double xc = 0.20 * LX;      // 目标螺旋核x（比如靠左就0.2*LX，靠右就0.8*LX）
		//	double yc = 0.15 * H;       // 目标螺旋核y（如果yi=1在底边，这就是靠底；若yi=1在顶边改成0.85*H）
		//	// ==========================

		//	// ===== 形状/尺度参数（决定成功率）=====
		//	double R = 0.18 * std::min((double)LX, (double)H);   // patch半径：至少覆盖~1个波长
		//	double lambda = 0.14 * std::min((double)LX, (double)H);   // 螺旋臂间距≈波长(不知道就先用0.12~0.18)
		//	double k = 2.0 * PI / lambda;                        // 相位径向增长率
		//	double w = 3.0;                                      // 边缘融合宽度(网格单位) 2~6
		//	double eps = 0.20;                                     // 条纹“锋利度” 0.1~0.3
		//	double r0 = 2.0 * w;                                  // 核心去奇点半径
		//	// ======================================

		//	int xL = std::max(1, (int)std::floor(xc - R - 6 * w));
		//	int xR = std::min(LX - 1, (int)std::ceil(xc + R + 6 * w));
		//	int yL = std::max(1, (int)std::floor(yc - R - 6 * w));
		//	int yR = std::min(H - 1, (int)std::ceil(yc + R + 6 * w));

		//	for (int yi = yL; yi <= yR; yi++) {
		//		for (int xi = xL; xi <= xR; xi++) {
		//			double dx = xi - xc, dy = yi - yc;
		//			double r = std::sqrt(dx * dx + dy * dy);
		//			if (r > R + 6 * w) continue;

		//			double theta = std::atan2(dy, dx);
		//			double rr = (r < r0) ? r0 : r;             // 避免r=0奇点
		//			double phi = theta - k * rr;               // ★ CW：用 “theta - k*r”
		//			double s = 0.5 * (1.0 + std::tanh(std::sin(phi) / eps)); // 0~1条纹
		//			double env = 0.5 * (1.0 - std::tanh((r - R) / w));         // 圆patch平滑边界

		//			int gi = get_index(xi, yi, 1);

		//			// 用“互补”让u像激发态、v像不应期（比单纯给u更稳）
		//			double u_target = uss_max * s;
		//			double v_target = vss_max * (1.0 - s);

		//			// 平滑融合到背景，避免硬切导致反射/湮灭
		//			m_hum[gi] = (1.0 - env) * m_hum[gi] + env * u_target;
		//			m_hvm[gi] = (1.0 - env) * m_hvm[gi] + env * v_target;
		//		}
		//	}
		//	break;
		//}

		//case 2: { // 生成逆时针(CCW)螺旋“种子”
		//	const int H = (int)(LY * (1 - cuty));
		//	const double PI = 3.141592653589793;

		//	double xc = 0.80 * LX;      // 想放右边就0.8*LX（或用 xc = LX - (case1的xc)）
		//	double yc = 0.15 * H;

		//	double R = 0.18 * std::min((double)LX, (double)H);
		//	double lambda = 0.14 * std::min((double)LX, (double)H);
		//	double k = 2.0 * PI / lambda;
		//	double w = 3.0;
		//	double eps = 0.20;
		//	double r0 = 2.0 * w;

		//	int xL = std::max(1, (int)std::floor(xc - R - 6 * w));
		//	int xR = std::min(LX - 1, (int)std::ceil(xc + R + 6 * w));
		//	int yL = std::max(1, (int)std::floor(yc - R - 6 * w));
		//	int yR = std::min(H - 1, (int)std::ceil(yc + R + 6 * w));

		//	for (int yi = yL; yi <= yR; yi++) {
		//		for (int xi = xL; xi <= xR; xi++) {
		//			double dx = xi - xc, dy = yi - yc;
		//			double r = std::sqrt(dx * dx + dy * dy);
		//			if (r > R + 6 * w) continue;

		//			double theta = std::atan2(dy, dx);
		//			double rr = (r < r0) ? r0 : r;
		//			double phi = theta + k * rr;               // ★ CCW：用 “theta + k*r”
		//			double s = 0.5 * (1.0 + std::tanh(std::sin(phi) / eps));
		//			double env = 0.5 * (1.0 - std::tanh((r - R) / w));

		//			int gi = get_index(xi, yi, 1);

		//			double u_target = uss_max * s;
		//			double v_target = vss_max * (1.0 - s);

		//			m_hum[gi] = (1.0 - env) * m_hum[gi] + env * u_target;
		//			m_hvm[gi] = (1.0 - env) * m_hvm[gi] + env * v_target;
		//		}
		//	}
		//	break;
		//}
	case 1: // original band-style CW (顺时针条带式激发 — 恢复原始版本)
		for (int yi = 1; yi <= LY * (1 - cuty); yi++) {
			for (int xi = 15; xi <= 0.5 * LX; xi++) {
				int gi = get_index(xi, yi, 1);
				m_hum[gi] = uss_max * (LX * (cuty + 0.5 * band) - xi) / (LX * band);
			}
		}
		for (int yi = 1; yi < LY * (1 - cuty); yi++) {
			for (int xi = 20; xi < 0.6 * LX; xi++) {
				int gi = get_index(xi, yi, 1);
				m_hvm[gi] = vss_max * (LX * (cuty + band) - xi) / (LX * band);
			}
		}
		break;
	case 2: // original band-style CCW (逆时针条带式激发 — 恢复原始版本，镜像)
		for (int yi = 1; yi <= LY * (1 - cuty); yi++) {
			for (int xi = 15; xi <= 0.5 * LX; xi++) {
				int gi = get_index(LX - xi, yi, 1);   // ★ mirror to right side for CCW
				m_hum[gi] = uss_max * (LX * (cuty + 0.5 * band) - xi) / (LX * band);
			}
		}
		for (int yi = 1; yi < LY * (1 - cuty); yi++) {
			for (int xi = 20; xi < 0.6 * LX; xi++) {
				int gi = get_index(LX - xi, yi, 1);   // ★ mirror
				m_hvm[gi] = vss_max * (LX * (cuty + band) - xi) / (LX * band);
			}
		}
		break;
	case 3:    //target waves initialization
		for (int yi = int(LY / 2) - 2; yi <= int(LY / 2) + 2; yi++) {
			for (int xi = int(LX / 2) - 2; xi <= int(LX / 2) + 2; xi++) {
				int gi = get_index(xi, yi, 1);
				m_hum[gi] = 5.0 * uss_max;
			}
		}
		break;
	case 5: { // [TestC] true CCW atan2 spiral seed (single-arm, off-center)
		const double PI = 3.141592653589793;
		// Eccentric spiral core placed at (0.35*LX, 0.35*LY) — off-center to break symmetry
		double xc = 0.35 * LX;
		double yc = 0.35 * LY;
		double R  = 0.40 * std::min((double)LX, (double)LY); // patch radius
		double lambda = 0.16 * std::min((double)LX, (double)LY);
		double k  = 2.0 * PI / lambda;
		double w  = 3.0;   // edge blending width
		double eps = 0.18; // stripe sharpness
		double r0  = 2.0 * w; // desingularize r=0
		int xL = std::max(1, (int)std::floor(xc - R - 6*w));
		int xR = std::min(LX-1, (int)std::ceil(xc  + R + 6*w));
		int yL = std::max(1, (int)std::floor(yc - R - 6*w));
		int yR = std::min(LY-1, (int)std::ceil(yc  + R + 6*w));
		for (int yi = yL; yi <= yR; yi++) {
			for (int xi = xL; xi <= xR; xi++) {
				double dx = xi - xc, dy = yi - yc;
				double r = std::sqrt(dx*dx + dy*dy);
				if (r > R + 6*w) continue;
				double theta = std::atan2(dy, dx);
				double rr = (r < r0) ? r0 : r;
				double phi = theta + k * rr;  // ★ CCW: theta + k*r
				double s   = 0.5 * (1.0 + std::tanh(std::sin(phi) / eps));
				double env = 0.5 * (1.0 - std::tanh((r - R) / w));
				int gi = get_index(xi, yi, 1);
				m_hum[gi] = (1.0-env)*m_hum[gi] + env * uss_max * s;
				m_hvm[gi] = (1.0-env)*m_hvm[gi] + env * vss_max * (1.0 - s);
			}
		}
		break;
	}
	case 4:    //stable-state waves initialization
		srand(static_cast<unsigned>(time(nullptr)));  //   ʼ       
		for (int yi = 0; yi <= LY; yi++) {
			for (int xi = 0; xi <= LX; xi++) {
				int gi = get_index(xi, yi, 1);
				float random_perturbation = (rand() / (float)RAND_MAX) * 0.001f - 0.0005f;  // [-0.0005, 0.0005]
				m_hum[gi] = uss_max * (1.0f + random_perturbation);  //     ֵ   [0.9995*uss_max, 1.0005*uss_max]

				if (yi > 0 && yi < LY && xi>0 && xi < LX) {
					m_hvm[gi] = vss_max;
				}
			}
		}
		break;
	}
}

void GelSystem::setInitValue(int ig, double startX, double startY)
{
	//   ʼ    ѧ      ֵ
	bool random = false;
	for (int yi = 1; yi < m_gelSize.y; yi++) {
		for (int xi = 1; xi < m_gelSize.x; xi++) {
			int gi = get_index(xi, yi, 1);

			if (random) {
				m_hvm[gi] = m_params.vss * (1 + 0.5 * (2 * double(rand()) / RAND_MAX - 1));
			}
			else {
				m_hvm[gi] = m_params.vss;
			}

		}
	}

	for (int yi = 0; yi <= m_gelSize.y; yi++) {
		for (int xi = 0; xi <= m_gelSize.x; xi++) {
			int gii = get_index(xi, yi, 2);
			int gi = get_index(xi, yi, 1);
			if (random) {
				m_hum[gi] = m_params.uss * (1 + 0.5 * (2 * double(rand()) / RAND_MAX - 1));
				m_hvn[gii] = m_params.vss * (1 + 0.5 * (2 * double(rand()) / RAND_MAX - 1));
				m_hwm[gi] = m_params.wss;
				m_hwmo[gi] = m_params.wss;
			}
			else {
				m_hum[gi] = m_params.uss;
				m_hvn[gii] = m_params.vss;
				m_hwm[gi] = m_params.wss;
				m_hwmo[gi] = m_params.wss;

			}
		}
	}

	for (int yi = 0; yi <= m_gelSize.y + 1; yi++) {
		for (int xi = 0; xi <= m_gelSize.x + 1; xi++) {
			int gi = get_index(xi, yi, 2);

			if (random) {
				m_hun[gi] = m_params.uss * (1 + 0.5 * (2 * double(rand()) / RAND_MAX - 1));
				m_hwn[gi] = m_params.wss;
			}
			else {
				m_hun[gi] = m_params.uss;
				m_hwn[gi] = m_params.wss;
			}
		}
	}

	double dx = 1.0;
	double LAMP = 1.1;
	double intervalx = m_gelSize.x * LAMP * dx + 3;
	double intervaly = m_gelSize.y * LAMP * dx + 3;

	for (int yi = 0; yi < m_gelSize.y + 2; yi++) { //     y ᷽ 򣬴 0  m_gelSize.y+1
		for (int xi = 0; xi < m_gelSize.x + 2; xi++) { //     x ᷽ 򣬴 0  m_gelSize.x+1
			int hi = get_index(xi, yi, 2);

			m_hrn[hi].x = double(xi) * LAMP * dx + startX;
			m_hrn[hi].y = double(yi) * LAMP * dx + startY;

		}
	}


	/*if (ig == 0) {
		m_params.drep = 0.05;
	}*/
	//if (ig == 0 || ig == 1) {
	//setChemicalWave(1);
	//}
	//if (ig == 2) {
	setChemicalWave(2); // [Baseline] CCW band wave
	//}


	//    ýڵ     
	setType(m_hmap_node, 2);
	setType(m_hmap_element, 1);

	//setChemicalWave(1);

}

void GelSystem::setGoonValue(int ig, int time)
{
	std::ifstream frn, fun, fum, fvm, fwm, fwmo; //fwmo, frnn, fVeln0, fVeln, fFFRepulsion1, fFFRepulsion2, fFn;
	//frm, fvn,fwmo, frnn, fwn, fVeln0, fVeln,
	//fFFRepulsion1, fFFRepulsion2, fFn;

	auto fname = [&](const char* mid) {
		return std::string("gel") + std::to_string(ig) + mid + std::to_string(time) + ".dat";
		};

	frn.open(fname("rn"), std::ios::in);
	//frnn.open(fname("rnn"), std::ios::in);
	fun.open(fname("un"), std::ios::in);
	//fvn.open(fname("vn"), std::ios::in);
	//fwn.open(fname("wn"), std::ios::in);
	//frm.open(fname("rm"), std::ios::in);
	fum.open(fname("um"), std::ios::in);
	fvm.open(fname("vm"), std::ios::in);
	fwm.open(fname("wm"), std::ios::in);
	fwmo.open(fname("wmo"), std::ios::in);
	//fFFRepulsion1.open(fname("FFRepulsion1"), std::ios::in);
	//fFFRepulsion2.open(fname("FFRepulsion2"), std::ios::in);
	//fVeln.open(fname("Veln"), std::ios::in);
	//fVeln0.open(fname("Veln0"), std::ios::in);
	//fFn.open(fname("Fn"), std::ios::in);            // ← 新增


	// 读单元量 layer=1，范围 [1, m_gelSize-1]
	for (int yi = 0; yi <= m_gelSize.y; ++yi) {
		for (int xi = 0; xi <= m_gelSize.x; ++xi) {
			int gi = get_index(xi, yi, 1);
			double um, vm, wm, rmx, rmy, wmo;

			fum >> um;            m_hum[gi] = um;
			fvm >> vm;            m_hvm[gi] = vm;
			fwm >> wm;            m_hwm[gi] = wm;
			fwmo >> wmo;            m_hwmo[gi] = wmo;
			//	frm >> rmx >> rmy;    m_hrm[gi] = make_double2(rmx, rmy);
		}
	}
	//printf("[calNodesVelocityD]wm **(%d,%d)(%.15f)\n", 1, 1, m_hwm[get_index(1, 1, 1)]);
	//printf("[calNodesVelocityD]uuuummm **(%d,%d)(%.15f)\n", 1, 1, m_hum[get_index(1, 1, 1)]);
	//printf("[calNodesVelocityD]vvmvvvvvm **(%d,%d)(%.15f)\n", 1, 1, m_hvm[get_index(1, 1, 1)]);
	//printf("[calNodesVelocityD]22222 **(%d,%d)(%.15f)\n", 1, 1, m_hwmo[get_index(1, 1, 1)]);

	// 读节点量 layer=2，范围 [1, m_gelSize]
	for (int yi = 0; yi <= m_gelSize.y + 1; ++yi) {
		for (int xi = 0; xi <= m_gelSize.x + 1; ++xi) {
			int gi = get_index(xi, yi, 2);
			double rnx, rny, rnnx, rnny;
			double un, wn, velx, vely, vel0x, vel0y, vn;
			double r1x, r1y, r2x, r2y;
			double fnx, fny;

			frn >> rnx >> rny;              m_hrn[gi] = make_double2(rnx, rny);
			//	frnn >> rnnx >> rnny;             m_hrnn[gi] = make_double2(rnnx, rnny);
			fun >> un;                       m_hun[gi] = un;
			//	fwn >> wn;                       m_hwn[gi] = wn;
			//	fvn >> vn;                       m_hvn[gi] = vn;
				//fVeln >> velx >> vely;            m_hVeln[gi] = make_double2(velx, vely);
				//fVeln0 >> vel0x >> vel0y;            m_hVeln0[gi] = make_double2(vel0x, vel0y);
				//fFFRepulsion1 >> r1x >> r1y;      m_hFFRepulsion1[gi] = make_double2(r1x, r1y);
				//fFFRepulsion2 >> r2x >> r2y;      m_hFFRepulsion2[gi] = make_double2(r2x, r2y);
				//fFn >> fnx >> fny;               m_hFn[gi] = make_double2(fnx, fny);   // ← 新增
		}
	}
	//printf("[calNodesVelocityD]unnnnuuuu **(%d,%d)(%.15f)\n", 1, 1, m_hun[get_index(1, 1, 2)]);
	frn.close();   fun.close();  fum.close();  fvm.close();  fwm.close();
	fwmo.close(); //frnn.close(); //fwn.close(); fvn.close(); frm.close();
	//fFFRepulsion1.close(); fFFRepulsion2.close();
	//fVeln.close(); fVeln0.close(); fFn.close();

	// set node type
	setType(m_hmap_node, 2);
	setType(m_hmap_element, 1);
}



void GelSystem::setType(int* a, int size)
{
	int xi, yi;
	int LX = m_gelSize.x + size - 1;
	int LY = m_gelSize.y + size - 1;

	for (xi = 1; xi < LX; xi++) {
		for (yi = 1; yi < LY; yi++) {
			a[get_index(xi, yi, size)] = 0;
		}
	}

	for (xi = 1; xi < LX; xi++) {
		a[get_index(xi, 0, size)] = 1;
		a[get_index(xi, LY, size)] = 2;
	}
	for (yi = 1; yi < LY; yi++) {
		a[get_index(0, yi, size)] = 3;
		a[get_index(LX, yi, size)] = 4;
	}

	a[get_index(0, 0, size)] = 5;
	a[get_index(0, LY, size)] = 6;
	a[get_index(LX, 0, size)] = 7;
	a[get_index(LX, LY, size)] = 8;
}

void GelSystem::copyDataToDevice()
{
	cudaMemcpy(m_dum, m_hum, sizeof(double) * m_numGelElements, cudaMemcpyHostToDevice);

	cudaMemcpy(m_dun, m_hun, sizeof(double) * m_numGelNodes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_dvm, m_hvm, sizeof(double) * m_numGelElements, cudaMemcpyHostToDevice);

	cudaMemcpy(m_dvn, m_hvn, sizeof(double) * m_numGelNodes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_dwm, m_hwm, sizeof(double) * m_numGelElements, cudaMemcpyHostToDevice);

	cudaMemcpy(m_dwn, m_hwn, sizeof(double) * m_numGelNodes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_dwmo, m_hwmo, sizeof(double) * m_numGelElements, cudaMemcpyHostToDevice);

	cudaMemcpy(m_drn, m_hrn, sizeof(double2) * m_numGelNodes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_drnn, m_hrnn, sizeof(double2) * m_numGelNodes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_dVeln, m_hVeln, sizeof(double2) * m_numGelNodes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_dVeln0, m_hVeln0, sizeof(double2) * m_numGelNodes, cudaMemcpyHostToDevice);

	cudaMemcpy(m_dmap_element, m_hmap_element, sizeof(int) * m_numGelElements, cudaMemcpyHostToDevice);

	cudaMemcpy(m_dmap_node, m_hmap_node, sizeof(int) * m_numGelNodes, cudaMemcpyHostToDevice);

}

void GelSystem::copyDataToHost()
{
	cudaMemcpyAsync(m_hum, m_dum, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hun, m_dun, sizeof(double) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hum_norm, m_dum_norm, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hvm, m_dvm, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hvn, m_dvn, sizeof(double) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hwm, m_dwm, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hwn, m_dwn, sizeof(double) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hwmo, m_dwmo, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hrm, m_drm, sizeof(double2) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hrn, m_drn, sizeof(double2) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hrnn, m_drnn, sizeof(double2) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hFn, m_dFn, sizeof(double2) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hVeln, m_dVeln, sizeof(double2) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hVeln0, m_dVeln0, sizeof(double2) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hvm_center, m_dvm_center, sizeof(double) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hum_center, m_dum_center, sizeof(double) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hwm_center, m_dwm_center, sizeof(double) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hrm_center, m_drm_center, sizeof(double2) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hFn_center, m_dFn_center, sizeof(double2) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hVeln_center, m_dVeln_center, sizeof(double2) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaMemcpyAsync(m_hfilament, m_dfilament, sizeof(double2) * 1000 * m_params.maxFilamentlen, cudaMemcpyDeviceToHost, m_gel_stream);

	cudaStreamSynchronize(m_gel_stream);//ͬ  

}

void GelSystem::freeHostMemory()
{
	delete[] m_hum;// ͷ m_humָ         ڴ 
	delete[] m_hum_norm;
	delete[] m_hvm;
	delete[] m_hvn;
	delete[] m_hun;
	delete[] m_hwm;
	delete[] m_hwn;
	delete[] m_hwmo;
	delete[] m_hrn;
	delete[] m_hrnn;
	delete[] m_hrm;
	delete[] m_hFn;
	delete[] m_hFFRepulsion1;//*************
	delete[] m_hFFRepulsion2;//*************
	delete[] m_hVeln;
	delete[] m_hVeln0;
	delete[] m_hmap_element;
	delete[] m_hmap_node;
	delete[]m_hvm_center;
	delete[]m_hum_center;
	delete[]m_hwm_center;
	delete[]m_hrm_center;
	delete[]m_hFn_center;
	delete[]m_hVeln_center;
	delete[] m_hfilament;

}

void GelSystem::freeDeviceMemory()
{

	cudaFree(m_dum);
	cudaFree(m_dun);
	cudaFree(m_dum_norm);
	cudaFree(m_dun_norm);
	cudaFree(m_dvm);
	cudaFree(m_dvn);
	cudaFree(m_dvm_norm);
	cudaFree(m_dvn_norm);
	cudaFree(m_dwm);
	cudaFree(m_dwn);
	cudaFree(m_dwmo);
	cudaFree(m_djsum);
	cudaFree(m_djsvm);
	cudaFree(m_djspm);
	cudaFree(m_drn);
	cudaFree(m_drnn);
	cudaFree(m_drm);
	cudaFree(m_dFn);
	cudaFree(m_dFFRepulsion1);
	cudaFree(m_dFFRepulsion2);
	cudaFree(m_dVeln);
	cudaFree(m_dVeln0);
	cudaFree(m_drm_loc);
	cudaFree(m_dnmSm);
	cudaFree(m_dPrem);
	cudaFree(m_dM_at_elem);   // [FchemoCoupling]
	cudaFree(m_dmap_element);
	cudaFree(m_dmap_node);
	cudaFree(m_dcir);
	cudaFree(m_dinterpa);
	cudaFree(m_dedge);
	cudaFree(m_dedge_y_values);
	cudaFree(m_dedge_x_values);
	cudaFree(m_dxx);
	cudaFree(m_dvm_center);
	cudaFree(m_dum_center);
	cudaFree(m_dwm_center);
	cudaFree(m_drm_center);
	cudaFree(m_dFn_center);
	cudaFree(m_dVeln_center);
	cudaFree(m_dfilament);

}

void GelSystem::steadyStateValue(double& um, double& vm, double& wm, double phi)
{

	double vsm, usm, usa;
	double vsa = 1.0;
	double vsb = 0.0;
	double LAMP = 1.1;
	wm = m_params.FA0 / (m_params.LAMDAV * LAMP * LAMP);
	while (abs(vsa - vsb) > 1e-10)
	{
		vsm = 0.5 * (vsa + vsb);
		usm = vsm / (1 - wm);
		usa = vsa / (1 - wm);
		if (fu_h(usm, vsm, wm, phi) * fu_h(usa, vsa, wm, phi) < 0) {
			vsb = vsm;
		}
		else
			vsa = vsm;
	}
	um = vsb / (1.0 - wm);
	vm = vsb;
}

double GelSystem::fu_h(double u, double v, double w, double phi)
{
	double ww = (1.0 - w) * (1.0 - w);
	return ww * u - u * u - (1.0 - w) * m_params.f * v * (u - m_params.q * ww) / (u + m_params.q * ww) + phi * m_params.P2;
}

void GelSystem::recordCenterElement(int ig, int time)
{
	ofstream fbodycenter;
	ofstream ffilament;
	string   str_filament;
	fbodycenter.open("gel" + to_string(ig) + "bodycenter" + to_string(m_count) + ".dat", ios::app);
	str_filament = "gel" + to_string(ig) + "filament" + to_string(m_count) + ".dat";
	ffilament.open(str_filament, ios::app);
	for (int i = 0; i < 1000; i++) {
		fbodycenter
			<< setw(9) << to_string(time - 1000 + i * 1) << "      "
			<< setw(9) << to_string(m_hrm_center[i].x) << "      "
			<< setw(9) << to_string(m_hrm_center[i].y) << "      "
			<< setw(9) << to_string(m_hFn_center[i].x) << "      "
			<< setw(9) << to_string(m_hFn_center[i].y) << "      "
			<< setw(9) << to_string(m_hVeln_center[i].x) << "      "
			<< setw(9) << to_string(m_hVeln_center[i].y) << "      "
			<< setw(9) << to_string(m_hum_center[i]) << "      "
			<< setw(9) << to_string(m_hvm_center[i]) << "      "
			<< setw(9) << to_string(m_hwm_center[i]) << "\n";
		//}
		//for (int i = 0; i < 1000; i++) {
			//str_filament = "gel" + to_string(ig) + "filament" + to_string(int(time) - 1000 + i * 1) + ".dat";
			//ffilament.open(str_filament);
			//for (int j = 0; j < m_params.maxFilamentlen; j++) {
				//int gi = m_params.maxFilamentlen * i + j;
				//if (m_hfilament[gi].x != 0) {
		ffilament
			<< setw(9) << to_string(time - 1000 + i * 1) << "      "
			<< setw(9) << to_string(m_hfilament[i].x) << "      "
			<< setw(9) << to_string(m_hfilament[i].y) << "\n";
		//}
	//}
	//ffilament.close();

	}
	ffilament.close();

	fbodycenter.close();

}


void GelSystem::recordData(int ig, int time)
{
	if (time % 1 == 0) {
		std::ofstream frn, fum, fun, fvm, fwm, fwmo;// , fFn, fFFRepulsion1, fFFRepulsion2, fVeln, fVeln0, frnn;//, fwmo, fwn, fFn, fFFRepulsion1, fFFRepulsion2, fVeln, fVeln0, frnn, frm fvn;
		auto open_out = [](std::ofstream& f, const std::string& name) {
			f.open(name, std::ios::out | std::ios::trunc);
			f.setf(std::ios::fixed);                  // 固定小数位
			f << std::setprecision(12);               // 小数点后 12 位
			};

		open_out(frn, "gel" + std::to_string(ig) + "rn" + std::to_string(time) + ".dat");
		//open_out(frnn, "gel" + std::to_string(ig) + "rnn" + std::to_string(time) + ".dat");
		open_out(fun, "gel" + std::to_string(ig) + "un" + std::to_string(time) + ".dat");
		//open_out(fvn, "gel" + std::to_string(ig) + "vn" + std::to_string(time) + ".dat");
		//open_out(fwn, "gel" + std::to_string(ig) + "wn" + std::to_string(time) + ".dat");
		//open_out(frm, "gel" + std::to_string(ig) + "rm" + std::to_string(time) + ".dat");
		open_out(fum, "gel" + std::to_string(ig) + "um" + std::to_string(time) + ".dat");
		open_out(fvm, "gel" + std::to_string(ig) + "vm" + std::to_string(time) + ".dat");
		open_out(fwm, "gel" + std::to_string(ig) + "wm" + std::to_string(time) + ".dat");
		open_out(fwmo, "gel" + std::to_string(ig) + "wmo" + std::to_string(time) + ".dat");
		//open_out(fFn, "gel" + std::to_string(ig) + "Fn" + std::to_string(time) + ".dat");
		//open_out(fFFRepulsion1, "gel" + std::to_string(ig) + "FFRepulsion1" + std::to_string(time) + ".dat");
		//open_out(fFFRepulsion2, "gel" + std::to_string(ig) + "FFRepulsion2" + std::to_string(time) + ".dat");
		//open_out(fVeln, "gel" + std::to_string(ig) + "Veln" + std::to_string(time) + ".dat");
		//open_out(fVeln0, "gel" + std::to_string(ig) + "Veln0" + std::to_string(time) + ".dat");

		int gi;
		// 节点数据 layer=2
		for (int yi = 0; yi <= m_gelSize.y + 1; ++yi) {
			for (int xi = 0; xi <= m_gelSize.x + 1; ++xi) {
				gi = get_index(xi, yi, 2);
				frn << m_hrn[gi].x << ' ' << m_hrn[gi].y << '\n';
				//frnn << m_hrnn[gi].x << ' ' << m_hrnn[gi].y << '\n';
				fun << m_hun[gi] << '\n';
				//fvn << m_hvn[gi] << '\n';
				//fwn << m_hwn[gi] << '\n';
				//fFn << m_hFn[gi].x << ' ' << m_hFn[gi].y << '\n';
				//fFFRepulsion1 << m_hFFRepulsion1[gi].x << ' ' << m_hFFRepulsion1[gi].y << '\n';
				//fFFRepulsion2 << m_hFFRepulsion2[gi].x << ' ' << m_hFFRepulsion2[gi].y << '\n';
				//fVeln << m_hVeln[gi].x << ' ' << m_hVeln[gi].y << '\n';
				//fVeln0 << m_hVeln0[gi].x << ' ' << m_hVeln0[gi].y << '\n';
			}
		}

		// 单元数据 layer=1
		for (int yi = 0; yi <= m_gelSize.y; ++yi) {
			for (int xi = 0; xi <= m_gelSize.x; ++xi) {
				gi = get_index(xi, yi, 1);
				//	frm << m_hrm[gi].x << ' ' << m_hrm[gi].y << '\n';
				fum << m_hum[gi] << '\n';
				fvm << m_hvm[gi] << '\n';
				fwm << m_hwm[gi] << '\n';
				fwmo << m_hwmo[gi] << '\n';
			}
		}
	}
}

void GelSystem::recordmotionData(int ig, int time)
{
	if (time % 1 == 0) {
		ofstream fmotion;
		string str_motion;

		str_motion = "gel" + to_string(ig) + "motion" + to_string(time) + ".dat";
		fmotion.open(str_motion);
		int gi, gii;
		for (int yi = 1; yi < m_gelSize.y + 1; yi++) {
			for (int xi = 1; xi < m_gelSize.x + 1; xi++) {
				gi = get_index(xi, yi, 2);
				gii = get_index(xi, yi, 1);

				fmotion//// 
					<< setw(9) << to_string(m_hrn[gi].x) << "      "
					<< setw(9) << to_string(m_hrn[gi].y) << "      "
					<< setw(9) << to_string(m_hvn[gi]) << "       "
					<< setw(9) << to_string(m_hun[gi]) << "     "
					<< setw(9) << to_string(m_hwn[gi]) << "\n";
			}
		}

		fmotion.close();

	}

}

void GelSystem::writeFiles(int ig, double time)
{
	if (int(time) % 1000 == 0) {
		recordData(ig, time);
	}
	if (int(time) % 100 == 0) {
		recordmotionData(ig, time);
	}

	if (flag && int(time) % 1000 == 0) {
		recordCenterElement(ig, time);
	}
	else {
		flag = true;
	}
}

GelSystem::GelSystem(int ng, int ig, int2 gelSize, int time, int i, double j, double k, double startX, double startY) :
	m_bInitialized(false),
	m_gelSize(gelSize),
	m_ng(ng),
	m_ig(ig),
	//CPU data
	m_hum(0),
	m_hvm(0),
	m_hwm(0),
	m_hwn(0),
	m_hwmo(0),
	m_hrn(0),
	m_hrnn(0),
	m_hrm(0),
	m_hFn(0),
	m_hVeln(0),
	m_hVeln0(0),
	m_hvm_center(0),
	m_hum_center(0),
	m_hwm_center(0),
	m_hrm_center(0),
	m_hFn_center(0),
	m_hVeln_center(0),
	m_hmap_element(0),
	m_hmap_node(0),
	m_hfilament(0),
	//GPU data
	m_dfilament(0),
	m_dum(0),
	m_dum_norm(0),
	m_dun_norm(0),
	m_dvm(0),
	m_dvm_norm(0),
	m_dvn_norm(0),
	m_dwm(0),
	m_dwmo(0),
	m_dwn(0),
	m_djsum(0),
	m_djsvm(0),
	m_djspm(0),
	m_drn(0),
	m_drnn(0),
	m_drm(0),
	m_dFn(0),
	m_dFFRepulsion1(0),
	m_dFFRepulsion2(0),
	m_dVeln(0),
	m_dVeln0(0),
	m_drm_loc(0),
	m_dnmSm(0),
	m_dPrem(0),
	m_dM_at_elem(0),   // [FchemoCoupling]
	m_dvm_center(0),
	m_dum_center(0),
	m_dwm_center(0),
	m_drm_center(0),
	m_dFn_center(0),
	m_dVeln_center(0),
	m_dmap_element(0),
	m_dmap_node(0),
	m_dcir(0),
	m_dinterpa(0),
	m_dedge(0),
	m_dedge_y_values(0),
	m_dedge_x_values(0),
	m_dxx(0)

{
	m_dt = 0.001;
	m_df = int(1 / m_dt);//        
	m_numGelElements = (gelSize.x + 1) * (gelSize.y + 1);
	m_numGelNodes = (gelSize.x + 2) * (gelSize.y + 2);

	m_blockDim = dim3(8, 8, 1);//8x8x8  

	m_gridDim_1.x = (m_gelSize.x - 1 + m_blockDim.x - 1) / m_blockDim.x;   //4.25
	m_gridDim_1.y = (m_gelSize.y - 1 + m_blockDim.y - 1) / m_blockDim.y;

	m_gridDim0.x = (m_gelSize.x + m_blockDim.x - 1) / m_blockDim.x;//4.5
	m_gridDim0.y = (m_gelSize.y + m_blockDim.y - 1) / m_blockDim.y;

	m_gridDim1.x = (m_gelSize.x + 1 + m_blockDim.x - 1) / m_blockDim.x;//4 15+1Ԫ  
	m_gridDim1.y = (m_gelSize.y + 1 + m_blockDim.y - 1) / m_blockDim.y;

	m_gridDim2.x = (m_gelSize.x + 2 + m_blockDim.x - 1) / m_blockDim.x;//4.2   15+2 ڵ 
	m_gridDim2.y = (m_gelSize.y + 2 + m_blockDim.y - 1) / m_blockDim.y;

	// set simulation parameters
	/*for (int a = 0; a < sizeof(f) / sizeof(f[0]); a++) {
		f[a] = 1.1 + a * 0.1;
	}
	for (int b = 0; b < sizeof(ep) / sizeof(ep[0]); b++) {
		ep[b] = (b + 30) * 0.01;
	}
	for (int c = 0; c < sizeof(CHS) / sizeof(CHS[0]); c++) {
		CHS[c] = 0.01 + c * 0.01;
	}*/
	m_params.LX = m_gelSize.x;
	m_params.LY = m_gelSize.y;

	m_params.ig = m_ig;
	m_params.phi = 0.0;
	m_params.f = 0.9;
	//m_params.f = f[i];
	m_params.q = 1e-4;
	m_params.epini = 0.3;
	m_params.drep = 0.0;
	m_params.ep = m_params.epini + m_params.drep;
	m_params.P1 = 0.0124;
	m_params.P2 = 0.77;
	m_params.dt = m_dt;//0.001
	m_params.dtx = 5 * m_dt;//0.005
	m_params.dx = 1.0;
	m_params.dy = 1.0;
	m_params.CH0 = 0.338;
	m_params.CH1 = 0.518;
	m_params.CHS = 0.25;
	// [Exp1] Read CHM from environment variable CHM_VAL, default 0.01.
	// Usage: CHM_VAL=0.005 ./ligel
	{
		const char* chm_env = std::getenv("CHM_VAL");
		m_params.CHM = chm_env ? atof(chm_env) : 0.01;
		printf("[Exp1] m_params.CHM = %.6f (from %s)\n",
		       m_params.CHM, chm_env ? "CHM_VAL env" : "default");
	}
	m_params.C0 = 1.3e-3;
	m_params.LAMDAV = 1.1;
	m_params.AZ0 = 100.0;
	m_params.FA0 = 0.139;
	m_params.distance = 5.0;
	m_params.Repulsion_wall = 8.0;
	m_params.maxFilamentlen = 1;

	steadyStateValue(m_params.uss, m_params.vss, m_params.wss, 0);
	int LX_ = m_gelSize.x - 1;
	int LY_ = m_gelSize.y - 1;
	m_params.rn_offset[0] = { 0, 0 }; m_params.um_offset_noflux[0] = { 0, 0 }; m_params.um_offset_periodic[0] = { 0, 0 };

	m_params.rn_offset[1] = { 0, 1 }; m_params.um_offset_noflux[1] = { 0, 1 }; m_params.um_offset_periodic[1] = { 0, LY_ };
	m_params.rn_offset[2] = { 0, -1 }; m_params.um_offset_noflux[2] = { 0, -1 }; m_params.um_offset_periodic[2] = { 0,-LY_ };
	m_params.rn_offset[3] = { 1, 0 }; m_params.um_offset_noflux[3] = { 1, 0 }; m_params.um_offset_periodic[3] = { LX_, 0 };
	m_params.rn_offset[4] = { -1, 0 }; m_params.um_offset_noflux[4] = { -1, 0 }; m_params.um_offset_periodic[4] = { -LX_, 0 };

	m_params.rn_offset[5] = { 1, 1 }; m_params.um_offset_noflux[5] = { 1, 1 }; m_params.um_offset_periodic[5] = { LX_, LY_ };
	m_params.rn_offset[6] = { 1, -1 }; m_params.um_offset_noflux[6] = { 1, -1 }; m_params.um_offset_periodic[6] = { LX_, -LY_ };
	m_params.rn_offset[7] = { -1, 1 }; m_params.um_offset_noflux[7] = { -1, 1 }; m_params.um_offset_periodic[7] = { -LX_, LY_ };
	m_params.rn_offset[8] = { -1, -1 }; m_params.um_offset_noflux[8] = { -1, -1 }; m_params.um_offset_periodic[8] = { -LX_, -LY_ };

	_initialize(time, startX, startY);

}

GelSystem::~GelSystem()
{
	_finalize();
}

void GelSystem::_initialize(int time, double startX, double startY)
{
	assert(!m_bInitialized);

	allocateHostStorage();
	allocateDeviceStorage();

	if (time)
		setGoonValue(m_ig, time);
	else
		setInitValue(m_ig, startX, startY);
	copyDataToDevice();

	cudaStreamCreate(&m_gel_stream);

	m_bInitialized = true;

}

#define KCHK(where) do{ \
  cudaError_t _e = cudaGetLastError(); \
  if(_e != cudaSuccess){ \
    fprintf(stderr,"KERNEL %s: %s\n", where, cudaGetErrorString(_e)); \
    abort(); \
  } \
}while(0)

void GelSystem::update(long long int solverIterations,
	const double* d_ungrid, const SimgridParams& gridparams)
{
	assert(m_bInitialized);

	if (solverIterations % 5 == 0) {
		// [FchemoCoupling] First sample substrate M at each element center.
		// When CHM = 0 the result is read but multiplied by zero in calPressureD,
		// so behavior is bit-for-bit identical to original Y-B.
		calMatElem << <m_gridDim1, m_blockDim, 0, m_gel_stream >> >
			(m_dM_at_elem, m_drm, d_ungrid, m_params, gridparams);
		KCHK("calMatElem");

		//printf("[Step %lld] solverIterations = %lld\n", solverIterations, solverIterations);
		calPressureD << <m_gridDim1, m_blockDim, 0, m_gel_stream >> >
			(m_dPrem, m_dvm, m_dwm, m_dM_at_elem, m_params);
		KCHK("calPressureD");

		calNodesVelocityD << <m_gridDim0, m_blockDim, 0, m_gel_stream >> > (
			m_drnn, m_drn, m_dVeln, m_dVeln0, m_dFn,
			m_dPrem, m_dwm, m_dFFRepulsion1, m_dFFRepulsion2, m_params);
		KCHK("calNodesVelocityD_safe");

		calGelBoundaryNodesPositionD << <m_gridDim2, m_blockDim, 0, m_gel_stream >> >
			(m_drn, m_dmap_node, m_params);
		KCHK("calGelBoundaryNodesPositionD");

		calGelElementsPositionD << <m_gridDim1, m_blockDim, 0, m_gel_stream >> >
			(m_drn, m_drm, m_params);
		KCHK("calGelElementsPositionD");

		calElementsvolumefraction << <m_gridDim1, m_blockDim, 0, m_gel_stream >> >
			(m_dwm, m_dwmo, m_dwn, m_drn, m_params);
		KCHK("calElementsvolumefraction");

		calNodesvolumefraction << <m_gridDim2, m_blockDim, 0, m_gel_stream >> >
			(m_dwm, m_dwn, m_drn, m_params);
		KCHK("calNodesvolumefraction");

		std::swap(m_drn, m_drnn);
	}
}

void GelSystem::_finalize()
{

	assert(m_bInitialized);
	cudaStreamDestroy(m_gel_stream);

	if (m_file_writer_thread.joinable()) { //      ִ  
		m_file_writer_thread.join();
	}
	freeHostMemory();
	freeDeviceMemory();

	m_bInitialized = false;
}
























