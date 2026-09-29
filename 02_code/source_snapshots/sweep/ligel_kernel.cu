#include "stdio.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm> // 包含 std::max_element 和 std::min_element
#include <climits> 
#include <cstdio>  // 如果你用 printf()
#include "ligelParams.h"
#include"libottomgridParams.h"

// [MemoryFeedback] configurable feedback sign: +1.0 = positive feedback, -1.0 = negative feedback
// Change this value to switch between positive and negative feedback
constexpr double FEEDBACK_SIGN = -1.0;  // default: negative feedback

__constant__ SimParams params;
__constant__ SimgridParams gridparams;


__device__ double2 operator+(double2 a, double2 b)
{
	return make_double2(a.x + b.x, a.y + b.y);
}

__device__ double2 operator-(double2 a, double2 b)
{
	return make_double2(a.x - b.x, a.y - b.y);
}

__device__ double2 operator--(double2 a)
{
	return make_double2(-a.x, -a.y);
}

__device__ double2 operator*(double a, double2 b)
{
	return make_double2(a * b.x, a * b.y);
}

//__device__ double2 operator*(double2 a, double b)
//{
//	return make_double2(a.x * b, a.y * b);
//}

__device__ double operator*(double2 a, double2 b)
{
	return a.x * b.x + a.y * b.y;
}

__device__ double2 operator/(double2 a, double2 b)
{
	return make_double2(a.x / b.x, a.y / b.y);
}

__device__ double2 operator^(double2 a, double2 b)
{
	return make_double2(a.x * b.x, a.y * b.y);
}

__device__ void operator+=(double2& a, double2 b)
{
	a.x += b.x;
	a.y += b.y;
}

__device__ void operator-=(double2& a, double2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}

__device__ void operator*=(double2& a, double b)
{
	a.x *= b;
	a.y *= b;
}

__device__ void operator/=(double2& a, double b)
{
	a.x /= b;
	a.y /= b;
}

__device__ double cross(double2 a, double2 b)
{

	return a.x * b.y - b.x * a.y;
}

__device__ double distance(double2 a, double2 b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));//两个点a,b之间的距离
}

__device__ double2 pow(double2 a, int b)
{
	return make_double2(pow(a.x, b), pow(a.y, b));
}

__device__ int get_index(int xi, int yi, int size, SimParams params)//索引第几个元素
{
	return xi + yi * (params.LX + size);
}


__device__ int get_gnindex(int xi, int yi, int mode, SimgridParams gridparams)
{
	int pitch = 0;
	if (mode == 0) {
		pitch = gridparams.dlx + 0;
	}
	else if (mode == 1) {
		pitch = gridparams.dlx + 1;
	}
	else {
		return -1;
	}

	return xi + yi * pitch;
}

__device__ __forceinline__ int safe_get_index(int x, int y, int layer, const SimParams& params, int LX, int LY)
{
	int gi = get_index(x, y, layer, params);
	int max_index = (LX + 2) * (LY + 2);
	return (gi >= 0 && gi < max_index) ? gi : -1;
}


__device__ double maxval(double* vec, int size)
{
	double max = vec[0];
	for (int i = 1; i < size; ++i) {
		if (vec[i] > max) {
			max = vec[i];
		}
	}
	return max;
}

__device__ double minval(double* vec, int size)
{
	double min = vec[0]; // 假设第一个元素是最小值
	for (int i = 1; i < size; ++i) {
		if (vec[i] < min) {
			min = vec[i]; // 更新最小值
		}
	}
	return min;
}

__device__ double fv(double u, double v, double w, double phi, SimParams params)
{
	return params.ep * ((1.0 - w) * (1.0 - w) * u - (1.0 - w) * v + phi * (0.5 * params.P1 + params.P2));

}

__device__ double fu(double u, double v, double w, double phi, SimParams params)
{
	double ww = (1.0 - w) * (1.0 - w);
	return ww * u - u * u - (1.0 - w) * (params.f * v + phi * params.P1) * (u - params.q * ww) / (u + params.q * ww) + phi * params.P2;
}


__device__ double linear_interp(double a, double b, double t) {
	return a + (b - a) * t;
}

__device__ __host__ inline double clamp_d(double v, double lo, double hi)
{
	return (v < lo) ? lo : (v > hi ? hi : v);
}

//// ==== 工具 =====
__device__ __forceinline__ double clampi_device(double v, double lo, double hi) {
	v = (v < lo) ? lo : v;
	return (v > hi) ? hi : v;
}
__device__ __forceinline__ double safe_val(double v, double fallback) {
	return (isfinite(v) ? v : fallback);
}
__device__ __forceinline__ double2 limit_vec(double2 v, double vmax) {
	double n2 = v.x * v.x + v.y * v.y;
	if (n2 > vmax * vmax) {
		double n = sqrt(n2);
		double s = vmax / (n + 1e-12);
		v.x *= s; v.y *= s;
	}
	return v;
}


__device__ __forceinline__ bool finite2(double2 a) {
	return isfinite(a.x) && isfinite(a.y);
}

__device__ __forceinline__ double dmin_repul(double a, double b) { return (a < b) ? a : b; }
__device__ __forceinline__ double dmax_repul(double a, double b) { return (a > b) ? a : b; }

// 排斥7777777777777777777777777777777777胶-胶 LJ（保持不变）
__device__ __forceinline__ void add_lj_from_pos(const double2& p, const double2& my_pos, double rsi, double rc2, double eps2, double F_shift, double POSI, double& fx, double& fy)
{
	double dx = my_pos.x - p.x, dy = my_pos.y - p.y;
	double d2 = dx * dx + dy * dy;
	if (d2 < rc2 && d2 > eps2) {
		double d = sqrt(d2), invd = 1.0 / d;
		double r = rsi * invd;
		double r2 = r * r, r4 = r2 * r2, r6 = r4 * r2, r12 = r6 * r6;
		double F = -4.0 * POSI * invd * (-12.0 * r12 + 6.0 * r6) + F_shift;
		fx += (dx * invd) * F;  fy += (dy * invd) * F;
	}
}

// 平滑弹簧墙（纯排斥，C¹ 连续）
__device__ __forceinline__ double wall_force_smooth(double d, double R, double k, double Fmax)
{
	if (d >= R) return 0.0;
	double x = 1.0 - d / R;                          // x∈[0,1]
	double F = k * R * x * x * (3.0 - 2.0 * x);      // C¹
	return (F > Fmax) ? Fmax : F;
}

__device__ __forceinline__ void add_wall_point_smooth(double xw, double yw, const double2& p, double RC, double RC2, double k, double Fmax, double& fx, double& fy)
{

	double dx = p.x - xw, dy = p.y - yw;
	double d2 = dx * dx + dy * dy;
	if (d2 < RC2 && d2 > 0.0) {
		//double d = sqrt(d2), invd = 1.0 / d;
		double d = sqrt(d2);
		if (d < 1e-12) d = 1e-12;
		double invd = 1.0 / d;
		double F = wall_force_smooth(d, RC, k, Fmax);
		fx += (dx * invd) * F;  fy += (dy * invd) * F;
	}
}

__device__ __forceinline__ void repel_rect_SDF(const double2& p, double xL, double xR, double yB, double yT, double RC_in, double RC_out, double k, double Fmax, double2& Fout)
{
	if (!(xR > xL && yT > yB)) return;

	const bool inside = (p.x >= xL && p.x <= xR && p.y >= yB && p.y <= yT);

	if (inside) {
		// 内部：对最近边施力
		double dB = p.y - yB, dT = yT - p.y, dL = p.x - xL, dR = xR - p.x;
		double dist = dB; int which = 0;
		if (dT < dist) { dist = dT; which = 1; }
		if (dL < dist) { dist = dL; which = 2; }
		if (dR < dist) { dist = dR; which = 3; }
		if (dist < RC_in) {
			double F = wall_force_smooth(dist, RC_in, k, Fmax);
			if (which == 0) Fout.y -= F;   // 近下边
			else if (which == 1) Fout.y += F;   // 近上边
			else if (which == 2) Fout.x += F;   // 近左边
			else                  Fout.x -= F;  // 近右边
		}
	}
	else {
		// 外部：对“矩形最近点”（覆盖四角）施力
		double qx = clampi_device(p.x, xL, xR);
		double qy = clampi_device(p.y, yB, yT);
		double dx = p.x - qx, dy = p.y - qy;
		double d2 = dx * dx + dy * dy;
		if (d2 > 0.0 && RC_out > 0.0) {
			double d = sqrt(d2);
			if (d < 1e-12) d = 1e-12;          // 防除零
			if (d < RC_out) {
				double F = wall_force_smooth(d, RC_out, k, Fmax);
				Fout.x += dx / d * F;
				Fout.y += dy / d * F;
			}
		}
	}
}


// 便捷：warp 归约（double）
__inline__ __device__ double warp_sum(double v)
{
	for (int off = 16; off > 0; off >>= 1)
		v += __shfl_down_sync(0xffffffff, v, off);
	return v;
}

__inline__ __device__ void block_sum(double& fx, double& fy)
{
	fx = warp_sum(fx);
	fy = warp_sum(fy);
	__shared__ double sfx[32], sfy[32];        // 至多 1024/32 = 32 个 warp
	if ((threadIdx.x & 31) == 0) {             // 每个 warp 的 lane0 写入
		sfx[threadIdx.x >> 5] = fx;
		sfy[threadIdx.x >> 5] = fy;
	}
	__syncthreads();
	int nwarp = (blockDim.x + 31) >> 5;
	fx = (threadIdx.x < nwarp) ? sfx[threadIdx.x] : 0.0;
	fy = (threadIdx.x < nwarp) ? sfy[threadIdx.x] : 0.0;
	if (threadIdx.x < 32) {                    // 用第 0 个 warp 汇总
		fx = warp_sum(fx);
		fy = warp_sum(fy);
	}
}
//排斥*****************************************
__device__ double cal_IBInterp(double xaa[4], double yaa[4], double uaa[4], double m_locgridx, double m_locgridy)
{
	const double EPS = 1e-10;
	double HH1, HH2, EE1, EE2, FF1, FF2, GG1, GG2;
	double ka0, ka1, ka2, wa;
	double vat, uat, vaa1, vaa2;
	double fp, fq;
	double IBInterp = 0.0;

	HH1 = m_locgridx - xaa[3];
	HH2 = m_locgridy - yaa[3];
	EE1 = xaa[2] - xaa[3];
	EE2 = yaa[2] - yaa[3];
	FF1 = xaa[0] - xaa[3];
	FF2 = yaa[0] - yaa[3];
	GG1 = xaa[3] - xaa[2] + xaa[1] - xaa[0];
	GG2 = yaa[3] - yaa[2] + yaa[1] - yaa[0];

	ka0 = HH1 * EE2 - HH2 * EE1;
	ka1 = EE1 * FF2 - EE2 * FF1 + HH1 * GG2 - HH2 * GG1;
	ka2 = GG1 * FF2 - GG2 * FF1;

	// 插值主路径
	if (fabs(GG1) < EPS) {
		// 退化为线性插值
		if (fabs(ka1) < EPS) return uaa[3];
		vat = -ka0 / ka1;

		double denom = EE1 + GG1 * vat;
		if (fabs(denom) < EPS) return uaa[3];
		uat = (HH1 - FF1 * vat) / denom;

		fq = uaa[0] + (uaa[1] - uaa[0]) * uat;
		fp = uaa[3] + (uaa[2] - uaa[3]) * uat;
		IBInterp = fp + (fq - fp) * vat;
	}
	else {
		wa = ka1 * ka1 - 4.0 * ka0 * ka2;
		if (wa < 0.0 || fabs(ka2) < EPS) return uaa[3];

		wa = sqrt(wa);
		vaa1 = (-ka1 - wa) / (2.0 * ka2);
		vaa2 = (-ka1 + wa) / (2.0 * ka2);

		if (vaa1 >= 0.0 && vaa1 <= 1.0) {
			vat = vaa1;
		}
		else if (vaa2 >= 0.0 && vaa2 <= 1.0) {
			vat = vaa2;
		}
		else {
			return uaa[3];
		}

		double denom = EE1 + GG1 * vat;
		if (fabs(denom) < EPS) return uaa[3];
		uat = (HH1 - FF1 * vat) / denom;

		fq = uaa[0] + (uaa[1] - uaa[0]) * uat;
		fp = uaa[3] + (uaa[2] - uaa[3]) * uat;
		IBInterp = fp + (fq - fp) * vat;
	}

	if (!isfinite(IBInterp)) return uaa[3];
	return IBInterp;
}

__device__ static bool solveBilinear(double A, double B, double C, double D, double E, double F, double G, double H, double vIso, double& xb, double& yb, double eps = 1e-12)
{
	// ---------- ϵ   ----------
	const double a00 = A - vIso;
	const double a10 = B - A;
	const double a01 = C - A;
	const double a11 = A - B + D - C;

	const double b00 = E - vIso;
	const double b10 = F - E;
	const double b01 = G - E;
	const double b11 = E - F + H - G;

	const double c2 = b10 * a11 - b11 * a10;
	const double c1 = b00 * a11 + b10 * a01 - b01 * a10 - b11 * a00;
	const double c0 = b00 * a01 - b01 * a00;

	double roots[2];
	int n = 0;

	// ----------       / һ η    ----------
	if (fabs(c2) < eps) {               //     
		if (fabs(c1) < eps) return false;
		roots[n++] = -c0 / c1;
	}
	else {
		double disc = c1 * c1 - 4.0 * c2 * c0;
		if (disc < 0.0) return false;    //   ʵ  
		double s = sqrt(disc);
		roots[n++] = (-c1 + s) / (2.0 * c2);
		roots[n++] = (-c1 - s) / (2.0 * c2);
	}

	// ----------  ش    y    ɸѡ Ϸ    ----------
	for (int i = 0; i < n; ++i) {
		double x = roots[i];
		if (x < -eps || x > 1.0 + eps) continue;

		double denom = a01 + a11 * x;
		if (fabs(denom) < eps) continue;

		double y = -(a00 + a10 * x) / denom;
		if (y < -eps || y > 1.0 + eps) continue;

		xb = clamp_d(x, 0.0, 1.0);
		yb = clamp_d(y, 0.0, 1.0);
		return true;
	}
	return false;                        // û н      ڵ Ԫ  
}

__global__ void calElementsvolumefraction(double* wm, double* wmo, double* wn, double2* rn, SimParams params)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;//theradId0-4,blockId0-3,0-19
	int yi = threadIdx.y + blockIdx.y * blockDim.y;

	int LX = params.LX;
	int LY = params.LY;
	if (xi > LX || yi > LY) return;
	int gi = get_index(xi, yi, 1, params);

	//wmo[gi] = wm[gi];
	double node1 = rn[get_index(xi + 1, yi + 1, 2, params)].x - rn[get_index(xi, yi, 2, params)].x;
	double node2 = rn[get_index(xi, yi + 1, 2, params)].y - rn[get_index(xi + 1, yi, 2, params)].y;
	double node3 = rn[get_index(xi + 1, yi + 1, 2, params)].y - rn[get_index(xi, yi, 2, params)].y;
	double node4 = rn[get_index(xi, yi + 1, 2, params)].x - rn[get_index(xi + 1, yi, 2, params)].x;
	double area = 0.0;
	area = params.LAMDAV * 0.5 * (node1 * node2 - node3 * node4);
	// [NaNguard] prevent wm singularity: degenerate/inverted elements get high pressure restoring force
	if (!isfinite(area) || area <= 0.0) area = params.FA0; // fallback to reference area
	wm[gi] = params.FA0 / area;
	if (!isfinite(wm[gi]) || wm[gi] < 0.0) wm[gi] = 1.0; // degenerate catch
	wm[gi] = fmin(wm[gi], 0.99); // [NaNguard] hard cap: wm<1 ensures 1/(1-wm) finite, log(1-wm) finite
}

__global__ void calNodesvolumefraction(double* wm, double* wn, double2* rn, SimParams params)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;//0-19
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	int LX = params.LX;
	int LY = params.LY;

	if (xi > LX + 1 || yi > LY + 1) return;
	int gi = get_index(xi, yi, 2, params);

	if (xi >= 1 && xi <= LX && yi >= 1 && yi <= LY) {
		wn[gi] = 0.25 * (wm[get_index(xi, yi, 1, params)] + wm[get_index(xi, yi - 1, 1, params)] + wm[get_index(xi - 1, yi, 1, params)] + wm[get_index(xi - 1, yi - 1, 1, params)]);
	}
	// 角点处理
	if (xi == 0 && yi == 0) {
		wn[gi] = wm[get_index(0, 0, 1, params)];
	}
	else if (xi == 0 && yi == LY + 1) {
		wn[gi] = wm[get_index(0, LY, 1, params)];
	}
	else if (xi == LX + 1 && yi == 0) {
		wn[gi] = wm[get_index(LX, 0, 1, params)];
	}
	else if (xi == LX + 1 && yi == LY + 1) {
		wn[gi] = wm[get_index(LX, LY, 1, params)];
	}

	if (xi == 0 && yi > 0 && yi < LY + 1) {
		wn[gi] = wn[get_index(1, yi, 2, params)];
	}
	else if (xi == LX + 1 && yi > 0 && yi < LY + 1) {
		wn[gi] = wn[get_index(LX, yi, 2, params)];
	}
	else if (yi == 0 && xi > 0 && xi < LX + 1) {
		wn[gi] = wn[get_index(xi, 1, 2, params)];
	}
	else if (yi == LY + 1 && xi > 0 && xi < LX + 1) {
		wn[gi] = wn[get_index(xi, LY, 2, params)];
	}

}


__global__ void calPressureD(double* pm, double* vm, double* wm,
	const double* __restrict__ M_at_elem, SimParams params)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;//0-19
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int LX = params.LX;
	int LY = params.LY;

	if (xi > params.LX - 1 || yi > params.LY - 1) {//1-14
		return;
	}
	int gi = get_index(xi, yi, 1, params);
	double wmt = wm[gi];
	wmt = fmax(1e-8, fmin(wmt, 1.0 - 1e-6)); // [NaNguard] protect log(1-wmt) from singularity

	// [FchemoCoupling] Read pre-sampled substrate M at this element's center.
	// (Sampling done in a separate kernel calMatElem before calPressureD,
	//  so this kernel's signature stays minimal and Y-B Eq. 48 in
	//  calNodesVelocityD requires no changes — it sees the augmented pm
	//  through the standard pressure-gradient assembly.)
	const double M_local = M_at_elem[gi];

	double ini_pm = -(params.wss + log(1.0 - params.wss) + (params.CH0 + params.CH1 * params.wss) * params.wss * params.wss) + params.C0 * params.wss / (2.0 * params.FA0) + params.CHS * params.wss * params.vss;
	// note: ini_pm uses M_ref = 0 (gel before any deposition), so no CHM term in ini_pm.
	pm[gi] = -(wmt + log(1.0 - wmt) + (params.CH0 + params.CH1 * wmt) * wmt * wmt) + params.C0 * wmt / (2.0 * params.FA0) + params.CHS * wmt * vm[gi] + params.CHM * wmt * M_local - ini_pm;
}


// [FchemoCoupling] Sample substrate memory M at every gel-element center
// position rm[gi], write to M_at_elem[gi]. Uses the same bilinear scheme as
// calSampleUngrid (with multind^2 normalization). Out-of-domain positions are
// clamped to the grid interior. Called from GelSystem::update before calPressureD.
__global__ void calMatElem(double* __restrict__ M_at_elem,
	const double2* __restrict__ rm,
	const double* __restrict__ ungrid,
	SimParams params, SimgridParams gridparams)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;

	if (xi > params.LX - 1 || yi > params.LY - 1) return;
	int gi = get_index(xi, yi, 1, params);

	double px = rm[gi].x;
	double py = rm[gi].y;

	double ind = gridparams.ind;
	int dlx = gridparams.dlx, dly = gridparams.dly;

	// Clamp sampling point to valid grid domain
	px = fmax(0.0, fmin(px, ind * (double)dlx - 1e-9));
	py = fmax(0.0, fmin(py, ind * (double)dly - 1e-9));

	int gx = (int)(px / ind);
	int gy = (int)(py / ind);
	gx = max(0, min(gx, dlx - 1));
	gy = max(0, min(gy, dly - 1));

	double lox = fmod(px, ind);
	double loy = fmod(py, ind);
	int multind = (int)(1.0 / ind + 0.5);

	double d00 = ungrid[get_gnindex(gx,   gy,   1, gridparams)];
	double d10 = ungrid[get_gnindex(gx+1, gy,   1, gridparams)];
	double d01 = ungrid[get_gnindex(gx,   gy+1, 1, gridparams)];
	double d11 = ungrid[get_gnindex(gx+1, gy+1, 1, gridparams)];

	M_at_elem[gi] = (double)(multind * multind) * (
		d00 * (ind - lox) * (ind - loy) +
		d10 * lox         * (ind - loy) +
		d01 * (ind - lox) * loy         +
		d11 * lox         * loy
	);
}


__global__ void calNodesVelocityD(const double2* __restrict__ r_prev, double2* __restrict__ r_next, double2* __restrict__ Veln, double2* __restrict__ Veln0, double2* __restrict__ Fn,
	const double* __restrict__ pm, const double* __restrict__ wm, const double2* __restrict__ FFRepulsion1, const double2* __restrict__ FFRepulsion2, SimParams params)
{
	const int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	const int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	if (xi > params.LX || yi > params.LY) return;

	const int gi = get_index(xi, yi, 2, params);
	const int gii = get_index(xi, yi, 1, params);
	const int cap2 = (params.LX + 2) * (params.LY + 2);
	const int cap1 = (params.LX + 1) * (params.LY + 1);
	if ((unsigned)gi >= (unsigned)cap2 || (unsigned)gii >= (unsigned)cap1) return;

	const double2 rc = r_prev[gi];
	if (!isfinite(rc.x) || !isfinite(rc.y) ||
		!isfinite(Veln0[gi].x) || !isfinite(Veln0[gi].y)) {
		Fn[gi] = make_double2(0, 0);
		Veln[gi] = make_double2(0, 0);
		r_next[gi] = rc;
		Veln0[gi] = make_double2(0, 0);
		return;
	}

	auto add = [](double2 a, double2 b) { return make_double2(a.x + b.x, a.y + b.y); };
	auto mul = [](double s, double2 a) { return make_double2(s * a.x, s * a.y); };

	double2 F = make_double2(0, 0);
	// w_mean = average of 4 surrounding element volume fractions (matches Fortran wn)
	double  w_mean = 0.25 * (wm[get_index(xi - 1, yi - 1, 1, params)] +
	                         wm[get_index(xi,     yi - 1, 1, params)] +
	                         wm[get_index(xi - 1, yi,     1, params)] +
	                         wm[get_index(xi,     yi,     1, params)]);

	// ---- w 的数值保护（防止 w=0/1 或 NaN）
	w_mean = safe_val(w_mean, 0.5);
	const double w_eps = 1e-6;                  // 比 eps=1e-12 稍大，数值更稳
	double w = clampi_device(w_mean, w_eps, 1.0 - w_eps);

	// ================= 内部点 =================
	if (xi > 1 && xi < params.LX && yi > 1 && yi < params.LY) {
		const int xm1 = get_index(xi - 1, yi, 2, params);
		const int xp1 = get_index(xi + 1, yi, 2, params);
		const int ym1 = get_index(xi, yi - 1, 2, params);
		const int yp1 = get_index(xi, yi + 1, 2, params);

		const double2 rxm = r_prev[xm1], rxp = r_prev[xp1];
		const double2 rym = r_prev[ym1], ryp = r_prev[yp1];

		double2 r_sum = make_double2(0, 0);
		{
			int base = get_index(xi - 1, yi - 1, 2, params);
			const double2 a0 = r_prev[base + 0], a1 = r_prev[base + 1], a2 = r_prev[base + 2];
			r_sum.x += (a0.x - rc.x) + (a1.x - rc.x) + (a2.x - rc.x);
			r_sum.y += (a0.y - rc.y) + (a1.y - rc.y) + (a2.y - rc.y);
		}
		{
			int base = get_index(xi - 1, yi, 2, params);
			const double2 a0 = r_prev[base + 0], a1 = r_prev[base + 1], a2 = r_prev[base + 2];
			r_sum.x += (a0.x - rc.x) + (a1.x - rc.x) + (a2.x - rc.x);
			r_sum.y += (a0.y - rc.y) + (a1.y - rc.y) + (a2.y - rc.y);
		}
		{
			int base = get_index(xi - 1, yi + 1, 2, params);
			const double2 a0 = r_prev[base + 0], a1 = r_prev[base + 1], a2 = r_prev[base + 2];
			r_sum.x += (a0.x - rc.x) + (a1.x - rc.x) + (a2.x - rc.x);
			r_sum.y += (a0.y - rc.y) + (a1.y - rc.y) + (a2.y - rc.y);
		}

		const double ra1 = ryp.x - rxp.x, rb1 = -(ryp.y - rxp.y);
		const double ra2 = rxm.x - ryp.x, rb2 = -(rxm.y - ryp.y);
		const double ra3 = rym.x - rxm.x, rb3 = -(rym.y - rxm.y);
		const double ra4 = rxp.x - rym.x, rb4 = -(rxp.y - rym.y);

		w_mean = 0.25 * (
			wm[get_index(xi, yi, 1, params)] +
			wm[get_index(xi - 1, yi, 1, params)] +
			wm[get_index(xi, yi - 1, 1, params)] +
			wm[get_index(xi - 1, yi - 1, 1, params)]
			);

		double2 p_sum;
		p_sum.x = rb1 * pm[gii] + rb2 * pm[get_index(xi - 1, yi, 1, params)]
			+ rb3 * pm[get_index(xi - 1, yi - 1, 1, params)]
			+ rb4 * pm[get_index(xi, yi - 1, 1, params)];
		p_sum.y = ra1 * pm[gii] + ra2 * pm[get_index(xi - 1, yi, 1, params)]
			+ ra3 * pm[get_index(xi - 1, yi - 1, 1, params)]
			+ ra4 * pm[get_index(xi, yi - 1, 1, params)];

		F = add(mul(params.C0 / 3.0, r_sum), mul(0.5 * params.LAMDAV, p_sum));
		//***********************
				Fn[gi] = F;
	}

	// ================= 左边界 =================
	if (xi == 1 && yi > 1 && yi < params.LY) {
		double2 r_sum = make_double2(0, 0);
		for (int jj = -1; jj <= 1; ++jj)
			for (int ii = 0; ii <= 1; ++ii) {
				const double2 n = r_prev[get_index(xi + ii, yi + jj, 2, params)];
				r_sum.x += (n.x - rc.x); r_sum.y += (n.y - rc.y);
			}
		const double2 ryp = r_prev[get_index(xi, yi + 1, 2, params)];
		const double2 rxp = r_prev[get_index(xi + 1, yi, 2, params)];
		const double2 rym = r_prev[get_index(xi, yi - 1, 2, params)];
		const double ra1 = ryp.x - rxp.x, rb1 = -(ryp.y - rxp.y);
		const double ra4 = rxp.x - rym.x, rb4 = -(rxp.y - rym.y);

		w_mean = 0.5 * (wm[get_index(xi, yi, 1, params)] + wm[get_index(xi, yi - 1, 1, params)]);

		double2 ps; ps.x = rb1 * pm[gii] + rb4 * pm[get_index(xi, yi - 1, 1, params)];
		ps.y = ra1 * pm[gii] + ra4 * pm[get_index(xi, yi - 1, 1, params)];

		F = add(mul(params.C0 / 3.0, r_sum), mul(0.5 * params.LAMDAV, ps));
		F = add(F, FFRepulsion2[get_index(1, yi, 2, params)]);
		//***********************
				Fn[gi] = F;
	}

	// ================= 右边界 =================
	if (xi == params.LX && yi > 1 && yi < params.LY) {
		double2 r_sum = make_double2(0, 0);
		for (int jj = -1; jj <= 1; ++jj)
			for (int ii = -1; ii <= 0; ++ii) {
				const double2 n = r_prev[get_index(xi + ii, yi + jj, 2, params)];
				r_sum.x += (n.x - rc.x); r_sum.y += (n.y - rc.y);
			}
		const double2 ryp = r_prev[get_index(xi, yi + 1, 2, params)];
		const double2 rxm = r_prev[get_index(xi - 1, yi, 2, params)];
		const double2 rym = r_prev[get_index(xi, yi - 1, 2, params)];
		const double ra2 = rxm.x - ryp.x, rb2 = -(rxm.y - ryp.y);
		const double ra3 = rym.x - rxm.x, rb3 = -(rym.y - rxm.y);

		w_mean = 0.5 * (wm[get_index(xi - 1, yi, 1, params)] + wm[get_index(xi - 1, yi - 1, 1, params)]);

		double2 ps; ps.x = rb2 * pm[get_index(xi - 1, yi, 1, params)]
			+ rb3 * pm[get_index(xi - 1, yi - 1, 1, params)];
		ps.y = ra2 * pm[get_index(xi - 1, yi, 1, params)]
			+ ra3 * pm[get_index(xi - 1, yi - 1, 1, params)];

		F = add(mul(params.C0 / 3.0, r_sum), mul(0.5 * params.LAMDAV, ps));
		F = add(F, FFRepulsion2[get_index(params.LX, yi, 2, params)]);
		//***********************
				Fn[gi] = F;
	}

	// ================= 下边界 =================
	if (yi == 1 && xi > 1 && xi < params.LX) {
		double2 r_sum = make_double2(0, 0);
		for (int jj = 0; jj <= 1; ++jj)
			for (int ii = -1; ii <= 1; ++ii) {
				const double2 n = r_prev[get_index(xi + ii, yi + jj, 2, params)];
				r_sum.x += (n.x - rc.x); r_sum.y += (n.y - rc.y);
			}
		const double2 ryp = r_prev[get_index(xi, yi + 1, 2, params)];
		const double2 rxm = r_prev[get_index(xi - 1, yi, 2, params)];
		const double ra1 = ryp.x - r_prev[get_index(xi + 1, yi, 2, params)].x;
		const double rb1 = -(ryp.y - r_prev[get_index(xi + 1, yi, 2, params)].y);
		const double ra2 = rxm.x - ryp.x, rb2 = -(rxm.y - ryp.y);

		w_mean = 0.5 * (wm[get_index(xi, yi, 1, params)] + wm[get_index(xi - 1, yi, 1, params)]);

		double2 ps; ps.x = rb1 * pm[gii] + rb2 * pm[get_index(xi - 1, yi, 1, params)];
		ps.y = ra1 * pm[gii] + ra2 * pm[get_index(xi - 1, yi, 1, params)];

		F = add(mul(params.C0 / 3.0, r_sum), mul(0.5 * params.LAMDAV, ps));
		F = add(F, FFRepulsion1[get_index(xi, 1, 2, params)]);
		//***********************
				Fn[gi] = F;
	}

	// ================= 上边界 =================
	if (yi == params.LY && xi > 1 && xi < params.LX) {
		double2 r_sum = make_double2(0, 0);
		for (int jj = -1; jj <= 0; ++jj)
			for (int ii = -1; ii <= 1; ++ii) {
				const double2 n = r_prev[get_index(xi + ii, yi + jj, 2, params)];
				r_sum.x += (n.x - rc.x); r_sum.y += (n.y - rc.y);
			}
		const double2 rym = r_prev[get_index(xi, yi - 1, 2, params)];
		const double2 rxm = r_prev[get_index(xi - 1, yi, 2, params)];
		const double2 rxp = r_prev[get_index(xi + 1, yi, 2, params)];
		const double ra3 = rym.x - rxm.x, rb3 = -(rym.y - rxm.y);
		const double ra4 = rxp.x - rym.x, rb4 = -(rxp.y - rym.y);

		w_mean = 0.5 * (wm[get_index(xi - 1, yi - 1, 1, params)] + wm[get_index(xi, yi - 1, 1, params)]);

		double2 ps; ps.x = rb3 * pm[get_index(xi - 1, yi - 1, 1, params)]
			+ rb4 * pm[get_index(xi, yi - 1, 1, params)];
		ps.y = ra3 * pm[get_index(xi - 1, yi - 1, 1, params)]
			+ ra4 * pm[get_index(xi, yi - 1, 1, params)];

		F = add(mul(params.C0 / 3.0, r_sum), mul(0.5 * params.LAMDAV, ps));
		F = add(F, FFRepulsion1[get_index(xi, params.LY, 2, params)]);
		//***********************
				Fn[gi] = F;
	}

	// ================= 四个角 =================
	// 左下
	if (xi == 1 && yi == 1) {
		double2 r_sum = make_double2(0, 0);
		for (int jj = 0; jj <= 1; ++jj)
			for (int ii = 0; ii <= 1; ++ii) {
				const double2 n = r_prev[get_index(xi + ii, yi + jj, 2, params)];
				r_sum.x += (n.x - rc.x); r_sum.y += (n.y - rc.y);
			}
		const double2 ryp = r_prev[get_index(xi, yi + 1, 2, params)];
		const double2 rxp = r_prev[get_index(xi + 1, yi, 2, params)];
		const double ra1 = ryp.x - rxp.x, rb1 = -(ryp.y - rxp.y);

		w_mean = wm[get_index(xi, yi, 1, params)];
		double2 ps; ps.x = rb1 * pm[gii]; ps.y = ra1 * pm[gii];

		F = add(mul(params.C0 / 3.0, r_sum), mul(0.5 * params.LAMDAV, ps));
		F = add(F, FFRepulsion2[get_index(1, 1, 2, params)]);  // 左下(1,1): LEFT edge → FFRepulsion2
		//***********************
				Fn[gi] = F;
	}
	// 右下
	if (xi == params.LX && yi == 1) {
		double2 r_sum = make_double2(0, 0);
		for (int jj = 0; jj <= 1; ++jj)
			for (int ii = -1; ii <= 0; ++ii) {
				const double2 n = r_prev[get_index(xi + ii, yi + jj, 2, params)];
				r_sum.x += (n.x - rc.x); r_sum.y += (n.y - rc.y);
			}
		const double2 ryp = r_prev[get_index(xi, yi + 1, 2, params)];
		const double2 rxm = r_prev[get_index(xi - 1, yi, 2, params)];
		const double ra2 = rxm.x - ryp.x, rb2 = -(rxm.y - ryp.y);

		w_mean = wm[get_index(xi - 1, yi, 1, params)];
		double2 ps; ps.x = rb2 * pm[get_index(xi - 1, yi, 1, params)];
		ps.y = ra2 * pm[get_index(xi - 1, yi, 1, params)];

		F = add(mul(params.C0 / 3.0, r_sum), mul(0.5 * params.LAMDAV, ps));
		F = add(F, FFRepulsion1[get_index(params.LX, 1, 2, params)]);  // 右下(LX,1): BOTTOM edge → FFRepulsion1
		//***********************
				Fn[gi] = F;
	}
	// 左上
	if (xi == 1 && yi == params.LY) {
		double2 r_sum = make_double2(0, 0);
		for (int jj = -1; jj <= 0; ++jj)
			for (int ii = 0; ii <= 1; ++ii) {
				const double2 n = r_prev[get_index(xi + ii, yi + jj, 2, params)];
				r_sum.x += (n.x - rc.x); r_sum.y += (n.y - rc.y);
			}
		const double2 rym = r_prev[get_index(xi, yi - 1, 2, params)];
		const double2 rxp = r_prev[get_index(xi + 1, yi, 2, params)];
		const double ra4 = rxp.x - rym.x, rb4 = -(rxp.y - rym.y);

		w_mean = wm[get_index(xi, yi - 1, 1, params)];
		double2 ps; ps.x = rb4 * pm[get_index(xi, yi - 1, 1, params)];
		ps.y = ra4 * pm[get_index(xi, yi - 1, 1, params)];

		F = add(mul(params.C0 / 3.0, r_sum), mul(0.5 * params.LAMDAV, ps));
		F = add(F, FFRepulsion1[get_index(1, params.LY, 2, params)]);  // 左上(1,LY): TOP edge → FFRepulsion1
		//***********************
				Fn[gi] = F;
	}
	// 右上
	if (xi == params.LX && yi == params.LY) {
		double2 r_sum = make_double2(0, 0);
		for (int jj = -1; jj <= 0; ++jj)
			for (int ii = -1; ii <= 0; ++ii) {
				const double2 n = r_prev[get_index(xi + ii, yi + jj, 2, params)];
				r_sum.x += (n.x - rc.x); r_sum.y += (n.y - rc.y);
			}
		const double2 rym = r_prev[get_index(xi, yi - 1, 2, params)];
		const double2 rxm = r_prev[get_index(xi - 1, yi, 2, params)];
		const double ra3 = rym.x - rxm.x, rb3 = -(rym.y - rxm.y);

		w_mean = wm[get_index(xi - 1, yi - 1, 1, params)];
		double2 ps; ps.x = rb3 * pm[get_index(xi - 1, yi - 1, 1, params)];
		ps.y = ra3 * pm[get_index(xi - 1, yi - 1, 1, params)];

		F = add(mul(params.C0 / 3.0, r_sum), mul(0.5 * params.LAMDAV, ps));
		F = add(F, FFRepulsion2[get_index(params.LX, params.LY, 2, params)]);  // 右上(LX,LY): RIGHT edge → FFRepulsion2
		//***********************
				Fn[gi] = F;
	}

	// ====== 统一的速度/位置更新（数值保护） ======
	if (!isfinite(F.x) || !isfinite(F.y)) {
		Fn[gi] = make_double2(0, 0);
		Veln[gi] = make_double2(0, 0);
		r_next[gi] = rc;
		Veln0[gi] = make_double2(0, 0);
		return;
	}

	w = clampi_device(safe_val(w_mean, 0.5), 1e-6, 1.0 - 1e-6);
	double FA0 = safe_val(params.FA0, 0.0);
	if (FA0 < 0.0) FA0 = 0.0;           // 防负
	double area = params.dx * params.dy;
	area = (isfinite(area) && area > 1e-12) ? area : 1e-12;

	double sqrt_arg = FA0 / w;
	if (!(sqrt_arg > 0.0) || !isfinite(sqrt_arg)) sqrt_arg = 0.0;
	//printf("w_mean=%.6f w=%.6f\n", w_mean, w);

	double Mn = 4.0 * safe_val(params.AZ0, 0.0) * sqrt(sqrt_arg) * (1.0 - w) / area;

	//printf("F1=%.4f ,F2=%.4f ,vel1=%.4f,vel2=%.4f Mn=%.4f\n", Fn[1325].x, Fn[1325].y, Veln[1325].x, Veln[1325].y, Mn);

	// 与Fortran一致：不限制Mn上限
	if (!isfinite(Mn) || Mn < 0.0) Mn = 0.0;

	// 原始速度
	double2 veln = make_double2(Mn * F.x, Mn * F.y);


	Veln[gi] = veln;
	//************************************8888888888888888

	// 与Fortran一致: DISPLACEMENT每5步调用一次, 位移 = 5*0.5*dt*(vel_new+vel_old)
	const double2 vhalf = make_double2(5.0 * 0.5 * params.dt * (veln.x + Veln0[gi].x),
		5.0 * 0.5 * params.dt * (veln.y + Veln0[gi].y));
	if (!isfinite(vhalf.x) || !isfinite(vhalf.y)) {
		r_next[gi] = rc;
		Veln0[gi] = make_double2(0, 0);
		Fn[gi] = make_double2(0, 0);
		Veln[gi] = make_double2(0, 0);
		return;
	}
	////*******************
	double2 pre_loc = make_double2(rc.x + vhalf.x, rc.y + vhalf.y);

	// 与Fortran一致：不对位移做截断
	// 全节点：最后一道兜底，防止单个坏值扩散
	if (!isfinite(pre_loc.x) || !isfinite(pre_loc.y)) {
		pre_loc = rc;
		veln = make_double2(0.0, 0.0);
		F = make_double2(0.0, 0.0);
		Veln[gi] = veln;
		Fn[gi] = F;
	}
	/*printf("w=%.6f, |F|=%.4e, Mn=%.3f\n",
		w, sqrt(F.x* F.x + F.y * F.y), Mn);*/
		// 写回
	r_next[gi] = pre_loc;
	Veln0[gi] = veln;
}

__global__ void calGelBoundaryNodesPositionD(double2* rn, int* map_node, SimParams params)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;//blockidId0-4   0-19
	int yi = threadIdx.y + blockIdx.y * blockDim.y;

	int LX = params.LX;
	int LY = params.LY;
	int gi = get_index(xi, yi, 2, params);
	int node_type = map_node[gi];
	if (node_type == 0 || xi > params.LX + 1 || yi > params.LY + 1) {
		return;//0-16
	}
	if (node_type != 0) {
		int gi1 = get_index(xi + params.rn_offset[node_type].x, yi + params.rn_offset[node_type].y, 2, params);
		int gi2 = get_index(xi + 2 * params.rn_offset[node_type].x, yi + 2 * params.rn_offset[node_type].y, 2, params);
		rn[gi] = 2 * rn[gi1] - rn[gi2];
	}
}

__global__ void calGelElementsPositionD(double2* rn, double2* rm, SimParams params)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;//blockidId0-4   0-19
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	int LX = params.LX;
	int LY = params.LY;
	int gi = get_index(xi, yi, 1, params);
	if (xi > params.LX || yi > params.LY) {
		return;//0-15
	}
	rm[gi] = 0.25 * (rn[get_index(xi, yi, 2, params)] + rn[get_index(xi + 1, yi, 2, params)] + rn[get_index(xi, yi + 1, 2, params)] + rn[get_index(xi + 1, yi + 1, 2, params)]);

}

__global__ void calTermsD(double* jsu, double* jsv, double* jsp, double* wm, double* wmo, double2* rn, double2* Veln, double* wn, double* un, double* um_norm, double* um, double* vm_norm, double* vm, double2* rm, SimParams params)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;//1-20
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	if (xi > params.LX || yi > params.LY) {
		return;//1-14
	}
	int gi = get_index(xi, yi, 1, params);
	um_norm[gi] = um[gi];
	vm_norm[gi] = vm[gi];
	double ax1 = 0, ax2 = 0, ax3 = 0, ax4 = 0;
	double ay1 = 0, ay2 = 0, ay3 = 0, ay4 = 0;
	double bx1 = 0, bx2 = 0, bx3 = 0, bx4 = 0;
	double by1 = 0, by2 = 0, by3 = 0, by4 = 0;
	//*********************************
	double cx1 = 0, cx2 = 0, cx3 = 0, cx4 = 0;
	double cy1 = 0, cy2 = 0, cy3 = 0, cy4 = 0;
	//*************************************
	double jsu1 = 0, jsu2 = 0, jsu3 = 0, jsu4 = 0;
	//88
	double rr1 = 0, rr2 = 0, rr3 = 0, rr4 = 0;
	if (xi >= 1 && xi < params.LX && yi >= 1 && yi < params.LY) {
		ax1 = (Veln[get_index(xi, yi, 2, params)].x * un[get_index(xi, yi, 2, params)] / (1.0 - wn[get_index(xi, yi, 2, params)]) + Veln[get_index(xi + 1, yi, 2, params)].x * un[get_index(xi + 1, yi, 2, params)] / (1.0 - wn[get_index(xi + 1, yi, 2, params)]));
		ay1 = (Veln[get_index(xi, yi, 2, params)].y * un[get_index(xi, yi, 2, params)] / (1.0 - wn[get_index(xi, yi, 2, params)]) + Veln[get_index(xi + 1, yi, 2, params)].y * un[get_index(xi + 1, yi, 2, params)] / (1.0 - wn[get_index(xi + 1, yi, 2, params)]));
		ax2 = (Veln[get_index(xi + 1, yi, 2, params)].x * un[get_index(xi + 1, yi, 2, params)] / (1.0 - wn[get_index(xi + 1, yi, 2, params)]) + Veln[get_index(xi + 1, yi + 1, 2, params)].x * un[get_index(xi + 1, yi + 1, 2, params)] / (1.0 - wn[get_index(xi + 1, yi + 1, 2, params)]));
		ay2 = (Veln[get_index(xi + 1, yi, 2, params)].y * un[get_index(xi + 1, yi, 2, params)] / (1.0 - wn[get_index(xi + 1, yi, 2, params)]) + Veln[get_index(xi + 1, yi + 1, 2, params)].y * un[get_index(xi + 1, yi + 1, 2, params)] / (1.0 - wn[get_index(xi + 1, yi + 1, 2, params)]));
		ax3 = (Veln[get_index(xi + 1, yi + 1, 2, params)].x * un[get_index(xi + 1, yi + 1, 2, params)] / (1.0 - wn[get_index(xi + 1, yi + 1, 2, params)]) + Veln[get_index(xi, yi + 1, 2, params)].x * un[get_index(xi, yi + 1, 2, params)] / (1.0 - wn[get_index(xi, yi + 1, 2, params)]));
		ay3 = (Veln[get_index(xi + 1, yi + 1, 2, params)].y * un[get_index(xi + 1, yi + 1, 2, params)] / (1.0 - wn[get_index(xi + 1, yi + 1, 2, params)]) + Veln[get_index(xi, yi + 1, 2, params)].y * un[get_index(xi, yi + 1, 2, params)] / (1.0 - wn[get_index(xi, yi + 1, 2, params)]));
		ax4 = (Veln[get_index(xi, yi + 1, 2, params)].x * un[get_index(xi, yi + 1, 2, params)] / (1.0 - wn[get_index(xi, yi + 1, 2, params)]) + Veln[get_index(xi, yi, 2, params)].x * un[get_index(xi, yi, 2, params)] / (1.0 - wn[get_index(xi, yi, 2, params)]));
		ay4 = (Veln[get_index(xi, yi + 1, 2, params)].y * un[get_index(xi, yi + 1, 2, params)] / (1.0 - wn[get_index(xi, yi + 1, 2, params)]) + Veln[get_index(xi, yi, 2, params)].y * un[get_index(xi, yi, 2, params)] / (1.0 - wn[get_index(xi, yi, 2, params)]));
		//*************
		bx1 = (rn[get_index(xi + 1, yi, 2, params)].x - rn[get_index(xi, yi, 2, params)].x);
		by1 = (rn[get_index(xi + 1, yi, 2, params)].y - rn[get_index(xi, yi, 2, params)].y);
		bx2 = (rn[get_index(xi + 1, yi + 1, 2, params)].x - rn[get_index(xi + 1, yi, 2, params)].x);
		by2 = (rn[get_index(xi + 1, yi + 1, 2, params)].y - rn[get_index(xi + 1, yi, 2, params)].y);
		bx3 = (rn[get_index(xi, yi + 1, 2, params)].x - rn[get_index(xi + 1, yi + 1, 2, params)].x);
		by3 = (rn[get_index(xi, yi + 1, 2, params)].y - rn[get_index(xi + 1, yi + 1, 2, params)].y);
		bx4 = (rn[get_index(xi, yi, 2, params)].x - rn[get_index(xi, yi + 1, 2, params)].x);
		by4 = (rn[get_index(xi, yi, 2, params)].y - rn[get_index(xi, yi + 1, 2, params)].y);
		//****************
		jsp[gi] = ax1 * by1 - ay1 * bx1 + ax2 * by2 - ay2 * bx2 + ax3 * by3 - ay3 * bx3 + ax4 * by4 - ay4 * bx4;//*polymer - solvent diffusion
		jsp[gi] = -0.5 * jsp[gi];
		cx1 = (rm[get_index(xi, yi - 1, 1, params)].x - rm[get_index(xi, yi, 1, params)].x);
		cy1 = (rm[get_index(xi, yi - 1, 1, params)].y - rm[get_index(xi, yi, 1, params)].y);
		cx2 = (rm[get_index(xi + 1, yi, 1, params)].x - rm[get_index(xi, yi, 1, params)].x);
		cy2 = (rm[get_index(xi + 1, yi, 1, params)].y - rm[get_index(xi, yi, 1, params)].y);
		cx3 = (rm[get_index(xi, yi + 1, 1, params)].x - rm[get_index(xi, yi, 1, params)].x);
		cy3 = (rm[get_index(xi, yi + 1, 1, params)].y - rm[get_index(xi, yi, 1, params)].y);
		cx4 = (rm[get_index(xi - 1, yi, 1, params)].x - rm[get_index(xi, yi, 1, params)].x);
		cy4 = (rm[get_index(xi - 1, yi, 1, params)].y - rm[get_index(xi, yi, 1, params)].y);

		// [NaNguard] safe wm values: wm is already capped at 0.99, but guard anyway
		const double WM_SAFE = 0.99;
		double wm_c = fmin(wm[get_index(xi,   yi,   1, params)], WM_SAFE);
		double wm_s = fmin(wm[get_index(xi,   yi-1, 1, params)], WM_SAFE);
		double wm_n = fmin(wm[get_index(xi,   yi+1, 1, params)], WM_SAFE);
		double wm_e = fmin(wm[get_index(xi+1, yi,   1, params)], WM_SAFE);
		double wm_w = fmin(wm[get_index(xi-1, yi,   1, params)], WM_SAFE);
		jsu1 = (1.0 - 0.5 * (wm_c + wm_s)) * (um[get_index(xi, yi - 1, 1, params)] / (1.0 - wm_s) - um[get_index(xi, yi, 1, params)] / (1.0 - wm_c));
		jsu2 = (1.0 - 0.5 * (wm_c + wm_e)) * (um[get_index(xi + 1, yi, 1, params)] / (1.0 - wm_e) - um[get_index(xi, yi, 1, params)] / (1.0 - wm_c));
		jsu3 = (1.0 - 0.5 * (wm_c + wm_n)) * (um[get_index(xi, yi + 1, 1, params)] / (1.0 - wm_n) - um[get_index(xi, yi, 1, params)] / (1.0 - wm_c));
		jsu4 = (1.0 - 0.5 * (wm_c + wm_w)) * (um[get_index(xi - 1, yi, 1, params)] / (1.0 - wm_w) - um[get_index(xi, yi, 1, params)] / (1.0 - wm_c));
		//**************
		rr1 = (cx1 * cx1 + cy1 * cy1);
		rr2 = (cx2 * cx2 + cy2 * cy2);
		rr3 = (cx3 * cx3 + cy3 * cy3);
		rr4 = (cx4 * cx4 + cy4 * cy4);

		//**********
		const double RR_MIN = 1e-10; // [NaNguard] prevent divide-by-zero when element centers coincide
		jsu[gi] = -(jsu1 * (cx1 * by1 - cy1 * bx1) / fmax(rr1, RR_MIN) + jsu2 * (cx2 * by2 - cy2 * bx2) / fmax(rr2, RR_MIN) + jsu3 * (cx3 * by3 - cy3 * bx3) / fmax(rr3, RR_MIN) + jsu4 * (cx4 * by4 - cy4 * bx4) / fmax(rr4, RR_MIN));
		//***************************************************************************************************************************************
		double wmo_safe = fmax(wmo[get_index(xi, yi, 1, params)], 1e-8); // [NaNguard]
		double jsv_raw = (1.0 - wm[get_index(xi, yi, 1, params)] / wmo_safe);
		jsv[gi] = fmax(-3.0, fmin(jsv_raw, 1.0)); // [NaNguard] clamp jsv: prevents chemical blowup when wm suddenly spikes
	}
}

__global__ void calChemD1(double* vm_norm, double* um_norm, double* vm, double* um, double* wm, double* jsu, double* jsv, double* jsp, double2* rm, SimParams params)
{
	int dx = 1;
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;//1-20
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	if (xi > params.LX - 1 || yi > params.LY - 1) {
		return;//1-14
	}
	int gi = get_index(xi, yi, 1, params);

	double dvm = vm[gi];
	double dum = um[gi];
	double dwm = wm[gi];
	double I = params.phi;
	/*if (xi >= 5&&xi<=10&& yi >=5&&yi<=10) {
		I = 0.2;
	}*/
	//Fourth order Runge Kutta method
	double k1_vm, k2_vm, k3_vm, k4_vm;
	double k1_um, k2_um, k3_um, k4_um;

	k1_vm = fv(dum, dvm, dwm, I, params);
	k1_um = fu(dum, dvm, dwm, I, params);

	// k2
	k2_vm = fv(dum + 0.5 * params.dt * k1_um, dvm + 0.5 * params.dt * k1_vm, dwm, I, params);
	k2_um = fu(dum + 0.5 * params.dt * k1_um, dvm + 0.5 * params.dt * k1_vm, dwm, I, params);
	// k3
	k3_vm = fv(dum + 0.5 * params.dt * k2_um, dvm + 0.5 * params.dt * k2_vm, dwm, I, params);
	k3_um = fu(dum + 0.5 * params.dt * k2_um, dvm + 0.5 * params.dt * k2_vm, dwm, I, params);
	// k4
	k4_vm = fv(dum + params.dt * k3_um, dvm + params.dt * k3_vm, dwm, I, params);
	k4_um = fu(dum + params.dt * k3_um, dvm + params.dt * k3_vm, dwm, I, params);

	um[gi] += (k1_um + (2.0 * (k2_um + k3_um)) + k4_um) * params.dt / 6.0 - dum * jsv[gi] - params.dt * (params.LAMDAV / (pow(dx, 2) * params.FA0)) * (jsp[gi] + jsu[gi]) * dwm;
	vm[gi] += (k1_vm + (2.0 * (k2_vm + k3_vm)) + k4_vm) * params.dt / 6.0 - dvm * jsv[gi];
	// [NaNguard] clamp chemical variables to finite physical range
	if (!isfinite(um[gi])) um[gi] = params.uss;
	if (!isfinite(vm[gi])) vm[gi] = params.vss;
	um[gi] = fmax(-0.5, fmin(um[gi], 3.0));
	vm[gi] = fmax(-0.5, fmin(vm[gi], 3.0));
}

__global__ void calChemD2(double* vm_norm, double* um_norm, double* vm, double* um, double* wm, double* jsu, double* jsv, double* jsp, double2* rm, SimParams params)
{
	int dx = 1;
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;//1-20
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	if (xi > params.LX - 1 || yi > params.LY - 1) {
		return;//1-14
	}
	int gi = get_index(xi, yi, 1, params);

	double dvm = vm[gi];
	double dum = um[gi];
	double dwm = wm[gi];
	double I = params.phi;
	if (xi >= 40 && xi <= 45 && yi >= 5 && yi <= 10) {
		I = 0.2;
	}
	//Fourth order Runge Kutta method
	double k1_vm, k2_vm, k3_vm, k4_vm;
	double k1_um, k2_um, k3_um, k4_um;

	k1_vm = fv(dum, dvm, dwm, I, params);
	k1_um = fu(dum, dvm, dwm, I, params);

	// k2
	k2_vm = fv(dum + 0.5 * params.dt * k1_um, dvm + 0.5 * params.dt * k1_vm, dwm, I, params);
	k2_um = fu(dum + 0.5 * params.dt * k1_um, dvm + 0.5 * params.dt * k1_vm, dwm, I, params);
	// k3
	k3_vm = fv(dum + 0.5 * params.dt * k2_um, dvm + 0.5 * params.dt * k2_vm, dwm, I, params);
	k3_um = fu(dum + 0.5 * params.dt * k2_um, dvm + 0.5 * params.dt * k2_vm, dwm, I, params);
	// k4
	k4_vm = fv(dum + params.dt * k3_um, dvm + params.dt * k3_vm, dwm, I, params);
	k4_um = fu(dum + params.dt * k3_um, dvm + params.dt * k3_vm, dwm, I, params);

	um[gi] += (k1_um + (2.0 * (k2_um + k3_um)) + k4_um) * params.dt / 6.0 - dum * jsv[gi] - params.dt * (params.LAMDAV / (pow(dx, 2) * params.FA0)) * (jsp[gi] + jsu[gi]) * dwm;
	vm[gi] += (k1_vm + (2.0 * (k2_vm + k3_vm)) + k4_vm) * params.dt / 6.0 - dvm * jsv[gi];
	// [NaNguard] clamp chemical variables to finite physical range
	if (!isfinite(um[gi])) um[gi] = params.uss;
	if (!isfinite(vm[gi])) vm[gi] = params.vss;
	um[gi] = fmax(-0.5, fmin(um[gi], 3.0));
	vm[gi] = fmax(-0.5, fmin(vm[gi], 3.0));
}


// [SteeringFeedback] calChemBoundaryD: Test A baseline - NO per-iter modification.
// Steering is applied as one-shot pulses via applySteeringPulseD (called at STEER_INTERVAL).
__global__ void calChemBoundaryD(double* um, double2* rm, double* wm, int* map_element,
    double* ungrid, SimParams params, SimgridParams gridparams)
{
    // No-op: Test A baseline. Steering applied via separate applySteeringPulseD kernel.
    return;
}

// [SteeringFeedback] applySteeringPulseD: one-shot left-right asymmetric um pulse.
// Called once per STEER_INTERVAL (e.g., every 10 TU), only when sim_iter >= t_on.
// Applies:  um[gi] += k_pulse * steer_delay * side_i * front_i
// where side_i=+1 right / -1 left, front_i ramps 0→1 from center to front edge.
// This is NOT called every iter → no compounding. The chemistry PDE equilibrates between pulses.
__global__ void applySteeringPulseD(double* um, double2* rm, int* map_element,
    SimParams params,
    double2 Rc_g, double2 ehat_g, double2 nhat_g,
    double steer_delay_g, double k_pulse)
{
    int xi = threadIdx.x + blockIdx.x * blockDim.x;
    int yi = threadIdx.y + blockIdx.y * blockDim.y;
    if (xi > params.LX || yi > params.LY) return;

    int gi = get_index(xi, yi, 1, params);
    if (map_element[gi] == 0) return;

    double relx = rm[gi].x - Rc_g.x;
    double rely = rm[gi].y - Rc_g.y;

    // side_i: +1 = right side, -1 = left side (guaranteed opposite)
    double dot_n = relx * nhat_g.x + rely * nhat_g.y;
    double side_i = (dot_n >= 0.0) ? 1.0 : -1.0;

    // front_i: front half of boundary gets higher weight, back half gets 0
    double proj_fwd = relx * ehat_g.x + rely * ehat_g.y;
    double front_i = fmax(0.0, fmin(proj_fwd / 25.0, 1.0));

    // One-shot asymmetric pulse: right front +bias, left front -bias
    double bias_i = k_pulse * steer_delay_g * side_i * front_i;

    // Clamp to physical range; additive to current chemistry value
    um[gi] = fmax(0.0001, fmin(um[gi] + bias_i, 1.5));
}


__global__ void calUVnnormD(double* un, double* um, double* vm, double* vn, double* wm, double* wmo, SimParams params)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;//1-20
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	if (xi > params.LX || yi > params.LY) {
		return;//1-15
	}
	int gi = get_index(xi, yi, 2, params);
	int gii = get_index(xi, yi, 1, params);
	if (xi > 0 && xi <= params.LX && yi > 0 && yi <= params.LY) {
		un[gi] = 0.25 * (um[get_index(xi - 1, yi - 1, 1, params)] + um[get_index(xi, yi - 1, 1, params)] + um[get_index(xi - 1, yi, 1, params)] + um[get_index(xi, yi, 1, params)]);
	}
	if (xi > 1 && xi < params.LX && yi>1 && yi < params.LY) {
		vn[gi] = 0.25 * (vm[get_index(xi - 1, yi - 1, 1, params)] + vm[get_index(xi - 1, yi, 1, params)] + vm[get_index(xi, yi - 1, 1, params)] + vm[get_index(xi, yi, 1, params)]);
	}
	if (xi == 1 && yi > 1 && yi < params.LY) {
		vn[gi] = 0.5 * (vm[get_index(1, yi, 1, params)] + vm[get_index(1, yi - 1, 1, params)]);
	}
	if (xi == params.LX && yi > 1 && yi < params.LY) {
		vn[gi] = 0.5 * (vm[get_index(xi - 1, yi, 1, params)] + vm[get_index(xi - 1, yi - 1, 1, params)]);
	}
	if (xi > 1 && xi < params.LX && yi == 1) {
		vn[gi] = 0.5 * (vm[get_index(xi, yi, 1, params)] + vm[get_index(xi - 1, yi, 1, params)]);
	}
	if (xi > 1 && xi < params.LX && yi == params.LY) {
		vn[gi] = 0.5 * (vm[get_index(xi, yi - 1, 1, params)] + vm[get_index(xi - 1, yi - 1, 1, params)]);
	}
	vn[get_index(1, 1, 2, params)] = vm[get_index(1, 1, 1, params)];
	vn[get_index(1, params.LY, 2, params)] = vm[get_index(1, params.LY - 1, 1, params)];
	vn[get_index(params.LX, 1, 2, params)] = vm[get_index(params.LX - 1, 1, 1, params)];
	vn[get_index(params.LX, params.LY, 2, params)] = vm[get_index(params.LX - 1, params.LY - 1, 1, params)];

	wmo[gii] = wm[gii];
	//um[get_index(xi, 0, 1, params)] = 5.0 * params.uss; 
}

__global__ void applyPulseBC(double* um, SimParams params)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;//1-20
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	if (xi > params.LX || yi > params.LY) {
		return;//1-15
	}

	um[get_index(xi, 0, 1, params)] = 5.0 * params.uss;//*****************************
}

// [SteeringFeedback] calClearLab: preserve substrate memory via rho_mem decay
// ungrid is READ-ONLY here (memory from previous step); ungriddiffusion is seeded.
__global__ void calClearLab(int* lab, const double* ungrid, double* ungriddiffusion, SimgridParams gp, double rho_mem) {
	int pitch = gp.dlx + 1;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x > gp.dlx || y > gp.dly) return;
	int gi = x + y * pitch;
	lab[gi] = 0;
	// Seed ungriddiffusion with decayed previous memory; ungrid NOT zeroed (it's the memory field)
	ungriddiffusion[gi] = rho_mem * ungrid[gi];
}

// [SteeringFeedback] calSampleUngrid: sample substrate memory at two points (left/right)
// Thread 0 → M_left_out;  Thread 1 → M_right_out
__global__ void calSampleUngrid(const double* ungrid, SimgridParams gridparams,
    double px_left, double py_left, double px_right, double py_right,
    double* M_left_out, double* M_right_out)
{
    double px = (threadIdx.x == 0) ? px_left  : px_right;
    double py = (threadIdx.x == 0) ? py_left  : py_right;
    double* out = (threadIdx.x == 0) ? M_left_out : M_right_out;

    double ind = gridparams.ind;
    int dlx = gridparams.dlx, dly = gridparams.dly;

    // Clamp sampling point to valid grid domain
    px = fmax(0.0, fmin(px, ind * (double)dlx - 1e-9));
    py = fmax(0.0, fmin(py, ind * (double)dly - 1e-9));

    int gx = (int)(px / ind);
    int gy = (int)(py / ind);
    gx = max(0, min(gx, dlx - 1));
    gy = max(0, min(gy, dly - 1));

    double lox = fmod(px, ind);
    double loy = fmod(py, ind);
    int multind = (int)(1.0 / ind + 0.5);

    double d00 = ungrid[get_gnindex(gx,   gy,   1, gridparams)];
    double d10 = ungrid[get_gnindex(gx+1, gy,   1, gridparams)];
    double d01 = ungrid[get_gnindex(gx,   gy+1, 1, gridparams)];
    double d11 = ungrid[get_gnindex(gx+1, gy+1, 1, gridparams)];

    *out = (double)(multind * multind) * (
        d00 * (ind - lox) * (ind - loy) +
        d10 * lox         * (ind - loy) +
        d01 * (ind - lox) * loy         +
        d11 * lox         * loy
    );
}


__host__ __device__ __forceinline__ int nextPow2_int(int x) {
	int p = 1;
	while (p < x) p <<= 1;
	return p;
}

// perimeter 点序号 p -> (x,y) -> rm 索引（与 Fortran/test.f90 的 cir 构造顺序一致）
__device__ __forceinline__ int perim_rm_index(int p, const SimParams& sp)
{
	const int LX = sp.LX;
	const int LY = sp.LY;

	const int n0 = (LX - 1);              // bottom: x=1..LX-1, y=1
	const int n1 = n0 + (LY - 2);         // right : x=LX-1, y=2..LY-1
	const int n2 = n1 + (LX - 2);         // top   : x=LX-2..1, y=LY-1
	// left  : x=1, y=LY-2..2  (count LY-3)

	int x, y;
	if (p < n0) {
		x = 1 + p;  y = 1;
	}
	else if (p < n1) {
		x = LX - 1; y = 2 + (p - n0);
	}
	else if (p < n2) {
		x = (LX - 2) - (p - n1); y = LY - 1;
	}
	else {
		x = 1; y = (LY - 2) - (p - n2);
	}
	return get_index(x, y, 1, sp);
}

__global__ void calHolepoint(const double2* const* __restrict__ all_rm,
	const SimParams* __restrict__ all_params,
	int ng,
	int* __restrict__ lab,
	SimgridParams gridparams,
	int maxC,
	int sortN)
{
	const int col = (int)blockIdx.x;                 // 一列一个 block
	if (col < 0 || col > gridparams.dlx) return;

	const double ind = gridparams.ind;
	const double xline = col * ind;
	const int gy_max = gridparams.dly;
	const int pitch = gridparams.dlx + 1;

	// 动态共享内存布局：ys[sortN] + (对齐) + cirL[maxC]
	extern __shared__ unsigned char smem[];
	size_t p = (size_t)smem;

	double* ys = (double*)p;
	p += (size_t)sortN * sizeof(double);

	// 16 字节对齐给 double2
	p = (p + 15u) & ~(size_t)15u;
	double2* cirL = (double2*)p;

	__shared__ int sh_m;
	__shared__ int sh_gy0, sh_gy1;

	const int tid = (int)threadIdx.x;
	const int nthreads = (int)blockDim.x;

	const double INF = 1e300;
	const double EPSX = 1e-12;
	const double EPSY = 1e-10;

	for (int ig = 0; ig < ng; ++ig) {
		const double2* __restrict__ rm = all_rm[ig];
		const SimParams params = all_params[ig];
		const int LX = params.LX;
		const int LY = params.LY;

		// 你的周界构造对应的段长度
		const int n0 = (LX - 1);      // y=1, x=1..LX-1
		const int n1 = (LY - 2);      // x=LX-1, y=2..LY-1
		const int n2 = (LX - 2);      // y=LY-1, x=LX-2..1
		const int n3 = (LY - 3);      // x=1, y=LY-2..2
		const int C = n0 + n1 + n2 + n3;   // = 2*(LX+LY)-8

		if (C < 2) continue;
		// host 端应保证 maxC >= 所有 gel 的 C；这里做个兜底
		if (C > maxC) continue;

		// 1) 并行加载周界点到 cirL[0..C-1]
		for (int k = tid; k < C; k += nthreads) {
			int x, y;
			if (k < n0) {
				x = 1 + k; y = 1;
			}
			else if (k < n0 + n1) {
				int t = k - n0;
				x = LX - 1; y = 2 + t;
			}
			else if (k < n0 + n1 + n2) {
				int t = k - (n0 + n1);
				x = (LX - 2) - t; y = LY - 1;
			}
			else {
				int t = k - (n0 + n1 + n2);
				x = 1; y = (LY - 2) - t;
			}
			cirL[k] = rm[get_index(x, y, 1, params)];
		}
		__syncthreads();

		// 2) 并行求交点写入 ys，用 shared atomic 累计个数
		if (tid == 0) sh_m = 0;
		__syncthreads();

		for (int k = tid; k < C; k += nthreads) {
			const double2 a = cirL[k];
			const double2 b = cirL[(k + 1) % C];
			const double dx = b.x - a.x;
			const double dy = b.y - a.y;

			double yhit;
			bool hit = false;

			if (fabs(dx) < EPSX) {
				// 近似竖线：若与 xline 重合，取中点
				if (fabs(a.x - xline) < EPSX) {
					yhit = 0.5 * (a.y + b.y);
					hit = true;
				}
			}
			else {
				const double t = (xline - a.x) / dx;
				if (t >= 0.0 && t <= 1.0) {
					yhit = a.y + t * dy;
					hit = true;
				}
			}

			if (hit) {
				int pos = atomicAdd(&sh_m, 1);
				if (pos < maxC) ys[pos] = yhit;
			}
		}
		__syncthreads();

		int m = sh_m;
		if (m < 2) { __syncthreads(); continue; }

		// 3) pad 到 sortN，保证能 bitonic 排序
		for (int i = tid; i < sortN; i += nthreads) {
			if (i >= m) ys[i] = INF;
		}
		__syncthreads();

		// 4) bitonic sort：排序 ys[0..sortN-1]
		for (int k = 2; k <= sortN; k <<= 1) {
			for (int j = k >> 1; j > 0; j >>= 1) {
				for (int i = tid; i < sortN; i += nthreads) {
					int ixj = i ^ j;
					if (ixj > i) {
						const bool up = ((i & k) == 0);
						double a = ys[i], b = ys[ixj];
						if ((a > b) == up) { ys[i] = b; ys[ixj] = a; }
					}
				}
				__syncthreads();
			}
		}

		// 5) 去重（交点数量通常很小，用 thread0 串行压缩足够快且稳定）
		if (tid == 0) {
			int mm = 0;
			double prev = -INF;
			for (int i = 0; i < m; ++i) {
				double v = ys[i];
				if (v >= INF * 0.5) break;
				if (mm == 0 || fabs(v - prev) > EPSY) {
					ys[mm++] = v;
					prev = v;
				}
			}
			sh_m = mm;
		}
		__syncthreads();

		m = sh_m;
		if (m < 2) { __syncthreads(); continue; }

		// 6) 两两配对填充 lab（并行写入）
		const int npairs = m / 2;
		for (int pidx = 0; pidx < npairs; ++pidx) {
			if (tid == 0) {
				double y0 = ys[2 * pidx];
				double y1 = ys[2 * pidx + 1];
				if (y0 > y1) { double t = y0; y0 = y1; y1 = t; }

				int gy0 = (int)(y0 / ind) + 1;
				int gy1 = (int)(y1 / ind);
				if (fabs(fmod(y1, ind)) < EPSY) gy1 -= 1;

				if (gy0 < 0) gy0 = 0;
				if (gy1 > gy_max) gy1 = gy_max;

				sh_gy0 = gy0;
				sh_gy1 = gy1;
			}
			__syncthreads();

			for (int gy = sh_gy0 + tid; gy <= sh_gy1; gy += nthreads) {
				lab[col + gy * pitch] = 1;  // 并集
			}
			__syncthreads();
		}
		}
	}

#if __CUDA_ARCH__ >= 350
#define RDG(ptr) __ldg(ptr)
#else
#define RDG(ptr) (*(ptr))
#endif

// 仅做插值：只在 lab==1 的点做，命中四边形才写 ungriddiffusion；其它位置保持（calClearLab 已清零）
__global__ void calGridInterpOnly(double* __restrict__ ungriddiffusion, const double2* const* __restrict__ all_rm, const double* const* __restrict__ all_um, const SimParams* __restrict__ all_params, int ng,
	const double* __restrict__ m_locgridx, const double* __restrict__ m_locgridy, int* __restrict__ lab, SimgridParams gridparams)
{
	// 与对照核一致：+1 偏移，排除外边界
	const int gxi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	const int gyi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	const int dlx = gridparams.dlx, dly = gridparams.dly;
	if (gxi >= dlx || gyi >= dly) return;

	const int ggi = get_gnindex(gxi, gyi, 1, gridparams);
	if (lab[ggi] != 1) return; // 只在包络内插

	const double gx = RDG(&m_locgridx[gxi]);
	const double gy = RDG(&m_locgridy[gyi]);

	bool   hit = false;
	double best = 0.0;

	// 对每个 gel 做与对照核一致的四边带判定与插值；汇聚策略：max
	for (int ig = 0; ig < ng; ++ig) {
		const double2* __restrict__ rm = all_rm[ig];
		const double* __restrict__ um = all_um[ig];
		const SimParams sp = all_params[ig];

		// ------- 底边带 -------
		for (int xi = 1; xi < sp.LX - 1; ++xi) {
			const int i00 = get_index(xi, 1, 1, sp);
			const int i10 = get_index(xi + 1, 1, 1, sp);
			const int i11 = get_index(xi + 1, 2, 1, sp);
			const int i01 = get_index(xi, 2, 1, sp);

			const double2 p00 = RDG(&rm[i00]), p10 = RDG(&rm[i10]);
			const double2 p11 = RDG(&rm[i11]), p01 = RDG(&rm[i01]);

			double xaa[4] = { p00.x, p10.x, p11.x, p01.x };
			double yaa[4] = { p00.y, p10.y, p11.y, p01.y };
			double uaa[4] = { RDG(&um[i00]), RDG(&um[i10]), RDG(&um[i11]), RDG(&um[i01]) };

			double tx = xaa[0], bx = xaa[0], ty = yaa[0], by = yaa[0];
#pragma unroll
			for (int k = 1; k < 4; ++k) { tx = fmax(tx, xaa[k]); bx = fmin(bx, xaa[k]); ty = fmax(ty, yaa[k]); by = fmin(by, yaa[k]); }
			if (gx < bx || gx > tx || gy < by || gy > ty) continue;

			const double h1 = (xaa[1] - xaa[0]) * (gy - yaa[0]) - (yaa[1] - yaa[0]) * (gx - xaa[0]);
			const double h2 = (xaa[2] - xaa[1]) * (gy - yaa[1]) - (yaa[2] - yaa[1]) * (gx - xaa[1]);
			const double h3 = (xaa[3] - xaa[2]) * (gy - yaa[2]) - (yaa[3] - yaa[2]) * (gx - xaa[2]);
			const double h4 = (xaa[0] - xaa[3]) * (gy - yaa[3]) - (yaa[0] - yaa[3]) * (gx - xaa[3]);

			auto commit = [&](double v) { best = hit ? fmax(best, v) : v; hit = true; };
			//auto commit = [&](double v) { best = v; hit = true; };

			if (fabs(h1) < 1e-10) { // 边上插值
				if (fabs(xaa[1] - xaa[0]) > 1e-10) commit(uaa[1] - (uaa[1] - uaa[0]) * (xaa[1] - gx) / (xaa[1] - xaa[0]));
				else                               commit(uaa[1] - (uaa[1] - uaa[0]) * (yaa[1] - gy) / (yaa[1] - yaa[0]));
			}
			if (fabs(h2) < 1e-10) {
				if (xaa[2] != xaa[1])              commit(uaa[2] - (uaa[2] - uaa[1]) * (xaa[2] - gx) / (xaa[2] - xaa[1]));
				else                               commit(uaa[2] - (uaa[2] - uaa[1]) * (yaa[2] - gy) / (yaa[2] - yaa[1]));
			}
			if (fabs(h3) < 1e-10) {
				if (fabs(xaa[3] - xaa[2]) > 1e-10) commit(uaa[3] - (uaa[3] - uaa[2]) * (xaa[3] - gx) / (xaa[3] - xaa[2]));
				else                               commit(uaa[3] - (uaa[3] - uaa[2]) * (yaa[3] - gy) / (yaa[3] - yaa[2]));
			}
			if (fabs(h4) < 1e-10) {
				if (fabs(xaa[0] - xaa[3]) > 1e-10) commit(uaa[0] - (uaa[0] - uaa[3]) * (xaa[0] - gx) / (xaa[0] - xaa[3]));
				else                               commit(uaa[0] - (uaa[0] - uaa[3]) * (yaa[0] - gy) / (yaa[0] - yaa[3]));
			}
			if ((h1 > 0 && h2 > 0 && h3 > 0 && h4 > 0) || (h1 < 0 && h2 < 0 && h3 < 0 && h4 < 0))
				commit(cal_IBInterp(xaa, yaa, uaa, gx, gy));
		}

		// ------- 顶边带 -------
		for (int xi = 1; xi < sp.LX - 1; ++xi) {
			const int i00 = get_index(xi, sp.LY - 2, 1, sp);
			const int i10 = get_index(xi + 1, sp.LY - 2, 1, sp);
			const int i11 = get_index(xi + 1, sp.LY - 1, 1, sp);
			const int i01 = get_index(xi, sp.LY - 1, 1, sp);

			const double2 p00 = RDG(&rm[i00]), p10 = RDG(&rm[i10]);
			const double2 p11 = RDG(&rm[i11]), p01 = RDG(&rm[i01]);

			double xaa[4] = { p00.x, p10.x, p11.x, p01.x };
			double yaa[4] = { p00.y, p10.y, p11.y, p01.y };
			double uaa[4] = { RDG(&um[i00]), RDG(&um[i10]), RDG(&um[i11]), RDG(&um[i01]) };

			double tx = xaa[0], bx = xaa[0], ty = yaa[0], by = yaa[0];
#pragma unroll
			for (int k = 1; k < 4; ++k) { tx = fmax(tx, xaa[k]); bx = fmin(bx, xaa[k]); ty = fmax(ty, yaa[k]); by = fmin(by, yaa[k]); }
			if (gx < bx || gx > tx || gy < by || gy > ty) continue;

			const double h1 = (xaa[1] - xaa[0]) * (gy - yaa[0]) - (yaa[1] - yaa[0]) * (gx - xaa[0]);
			const double h2 = (xaa[2] - xaa[1]) * (gy - yaa[1]) - (yaa[2] - yaa[1]) * (gx - xaa[1]);
			const double h3 = (xaa[3] - xaa[2]) * (gy - yaa[2]) - (yaa[3] - yaa[2]) * (gx - xaa[2]);
			const double h4 = (xaa[0] - xaa[3]) * (gy - yaa[3]) - (yaa[0] - yaa[3]) * (gx - xaa[3]);

			auto commit = [&](double v) { best = hit ? fmax(best, v) : v; hit = true; };
			//auto commit = [&](double v) { best = v; hit = true; };

			if (fabs(h1) < 1e-10) {
				if (fabs(xaa[1] - xaa[0]) > 1e-10) commit(uaa[1] - (uaa[1] - uaa[0]) * (xaa[1] - gx) / (xaa[1] - xaa[0]));
				else                               commit(uaa[1] - (uaa[1] - uaa[0]) * (yaa[1] - gy) / (yaa[1] - yaa[0]));
			}
			if (fabs(h2) < 1e-10) {
				if (xaa[2] != xaa[1])              commit(uaa[2] - (uaa[2] - uaa[1]) * (xaa[2] - gx) / (xaa[2] - xaa[1]));
				else                               commit(uaa[2] - (uaa[2] - uaa[1]) * (yaa[2] - gy) / (yaa[2] - yaa[1]));
			}
			if (fabs(h3) < 1e-10) {
				if (fabs(xaa[3] - xaa[2]) > 1e-10) commit(uaa[3] - (uaa[3] - uaa[2]) * (xaa[3] - gx) / (xaa[3] - xaa[2]));
				else                               commit(uaa[3] - (uaa[3] - uaa[2]) * (yaa[3] - gy) / (yaa[3] - yaa[2]));
			}
			if (fabs(h4) < 1e-10) {
				if (fabs(xaa[0] - xaa[3]) > 1e-10) commit(uaa[0] - (uaa[0] - uaa[3]) * (xaa[0] - gx) / (xaa[0] - xaa[3]));
				else                               commit(uaa[0] - (uaa[0] - uaa[3]) * (yaa[0] - gy) / (yaa[0] - yaa[3]));
			}
			if ((h1 > 0 && h2 > 0 && h3 > 0 && h4 > 0) || (h1 < 0 && h2 < 0 && h3 < 0 && h4 < 0))
				commit(cal_IBInterp(xaa, yaa, uaa, gx, gy));
		}

		// ------- 左边带 -------
		for (int yi = 1; yi < sp.LY - 1; ++yi) {
			const int i00 = get_index(1, yi, 1, sp);
			const int i10 = get_index(2, yi, 1, sp);
			const int i11 = get_index(2, yi + 1, 1, sp);
			const int i01 = get_index(1, yi + 1, 1, sp);

			const double2 p00 = RDG(&rm[i00]), p10 = RDG(&rm[i10]);
			const double2 p11 = RDG(&rm[i11]), p01 = RDG(&rm[i01]);

			double xaa[4] = { p00.x, p10.x, p11.x, p01.x };
			double yaa[4] = { p00.y, p10.y, p11.y, p01.y };
			double uaa[4] = { RDG(&um[i00]), RDG(&um[i10]), RDG(&um[i11]), RDG(&um[i01]) };

			double tx = xaa[0], bx = xaa[0], ty = yaa[0], by = yaa[0];
#pragma unroll
			for (int k = 1; k < 4; ++k) { tx = fmax(tx, xaa[k]); bx = fmin(bx, xaa[k]); ty = fmax(ty, yaa[k]); by = fmin(by, yaa[k]); }
			if (gx < bx || gx > tx || gy < by || gy > ty) continue;

			const double h1 = (xaa[1] - xaa[0]) * (gy - yaa[0]) - (yaa[1] - yaa[0]) * (gx - xaa[0]);
			const double h2 = (xaa[2] - xaa[1]) * (gy - yaa[1]) - (yaa[2] - yaa[1]) * (gx - xaa[1]);
			const double h3 = (xaa[3] - xaa[2]) * (gy - yaa[2]) - (yaa[3] - yaa[2]) * (gx - xaa[2]);
			const double h4 = (xaa[0] - xaa[3]) * (gy - yaa[3]) - (yaa[0] - yaa[3]) * (gx - xaa[3]);

			auto commit = [&](double v) { best = hit ? fmax(best, v) : v; hit = true; };
			//auto commit = [&](double v) { best = v; hit = true; };

			if (fabs(h1) < 1e-10) {
				if (fabs(xaa[1] - xaa[0]) > 1e-10) commit(uaa[1] - (uaa[1] - uaa[0]) * (xaa[1] - gx) / (xaa[1] - xaa[0]));
				else                               commit(uaa[1] - (uaa[1] - uaa[0]) * (yaa[1] - gy) / (yaa[1] - yaa[0]));
			}
			if (fabs(h2) < 1e-10) {
				if (xaa[2] != xaa[1])              commit(uaa[2] - (uaa[2] - uaa[1]) * (xaa[2] - gx) / (xaa[2] - xaa[1]));
				else                               commit(uaa[2] - (uaa[2] - uaa[1]) * (yaa[2] - gy) / (yaa[2] - yaa[1]));
			}
			if (fabs(h3) < 1e-10) {
				if (fabs(xaa[3] - xaa[2]) > 1e-10) commit(uaa[3] - (uaa[3] - uaa[2]) * (xaa[3] - gx) / (xaa[3] - xaa[2]));
				else                               commit(uaa[3] - (uaa[3] - uaa[2]) * (yaa[3] - gy) / (yaa[3] - yaa[2]));
			}
			if (fabs(h4) < 1e-10) {
				if (fabs(xaa[0] - xaa[3]) > 1e-10) commit(uaa[0] - (uaa[0] - uaa[3]) * (xaa[0] - gx) / (xaa[0] - xaa[3]));
				else                               commit(uaa[0] - (uaa[0] - uaa[3]) * (yaa[0] - gy) / (yaa[0] - yaa[3]));
			}
			if ((h1 > 0 && h2 > 0 && h3 > 0 && h4 > 0) || (h1 < 0 && h2 < 0 && h3 < 0 && h4 < 0))
				commit(cal_IBInterp(xaa, yaa, uaa, gx, gy));
		}

		// ------- 右边带 -------
		for (int yi = 1; yi < sp.LY - 1; ++yi) {
			const int i00 = get_index(sp.LX - 2, yi, 1, sp);
			const int i10 = get_index(sp.LX - 1, yi, 1, sp);
			const int i11 = get_index(sp.LX - 1, yi + 1, 1, sp);
			const int i01 = get_index(sp.LX - 2, yi + 1, 1, sp);

			const double2 p00 = RDG(&rm[i00]), p10 = RDG(&rm[i10]);
			const double2 p11 = RDG(&rm[i11]), p01 = RDG(&rm[i01]);

			double xaa[4] = { p00.x, p10.x, p11.x, p01.x };
			double yaa[4] = { p00.y, p10.y, p11.y, p01.y };
			double uaa[4] = { RDG(&um[i00]), RDG(&um[i10]), RDG(&um[i11]), RDG(&um[i01]) };

			double tx = xaa[0], bx = xaa[0], ty = yaa[0], by = yaa[0];
#pragma unroll
			for (int k = 1; k < 4; ++k) { tx = fmax(tx, xaa[k]); bx = fmin(bx, xaa[k]); ty = fmax(ty, yaa[k]); by = fmin(by, yaa[k]); }
			if (gx < bx || gx > tx || gy < by || gy > ty) continue;

			const double h1 = (xaa[1] - xaa[0]) * (gy - yaa[0]) - (yaa[1] - yaa[0]) * (gx - xaa[0]);
			const double h2 = (xaa[2] - xaa[1]) * (gy - yaa[1]) - (yaa[2] - yaa[1]) * (gx - xaa[1]);
			const double h3 = (xaa[3] - xaa[2]) * (gy - yaa[2]) - (yaa[3] - yaa[2]) * (gx - xaa[2]);
			const double h4 = (xaa[0] - xaa[3]) * (gy - yaa[3]) - (yaa[0] - yaa[3]) * (gx - xaa[3]);

			auto commit = [&](double v) { best = hit ? fmax(best, v) : v; hit = true; };
			//auto commit = [&](double v) { best = v; hit = true; };

			if (fabs(h1) < 1e-10) { // 边上插值
				if (fabs(xaa[1] - xaa[0]) > 1e-10) commit(uaa[1] - (uaa[1] - uaa[0]) * (xaa[1] - gx) / (xaa[1] - xaa[0]));
				else                               commit(uaa[1] - (uaa[1] - uaa[0]) * (yaa[1] - gy) / (yaa[1] - yaa[0]));
			}
			if (fabs(h2) < 1e-10) {
				if (xaa[2] != xaa[1])              commit(uaa[2] - (uaa[2] - uaa[1]) * (xaa[2] - gx) / (xaa[2] - xaa[1]));
				else                               commit(uaa[2] - (uaa[2] - uaa[1]) * (yaa[2] - gy) / (yaa[2] - yaa[1]));
			}
			if (fabs(h3) < 1e-10) {
				if (fabs(xaa[3] - xaa[2]) > 1e-10) commit(uaa[3] - (uaa[3] - uaa[2]) * (xaa[3] - gx) / (xaa[3] - xaa[2]));
				else                               commit(uaa[3] - (uaa[3] - uaa[2]) * (yaa[3] - gy) / (yaa[3] - yaa[2]));
			}
			if (fabs(h4) < 1e-10) {
				if (fabs(xaa[0] - xaa[3]) > 1e-10) commit(uaa[0] - (uaa[0] - uaa[3]) * (xaa[0] - gx) / (xaa[0] - xaa[3]));
				else                               commit(uaa[0] - (uaa[0] - uaa[3]) * (yaa[0] - gy) / (yaa[0] - yaa[3]));
			}
			if ((h1 > 0 && h2 > 0 && h3 > 0 && h4 > 0) || (h1 < 0 && h2 < 0 && h3 < 0 && h4 < 0))
				commit(cal_IBInterp(xaa, yaa, uaa, gx, gy));
		}
	} // ig
	//printf("hit: %d  %d  %f\n", (int)hit, ggi, best);
	//if (hit) ungriddiffusion[ggi] = best; // 只在命中时覆盖
	//ungriddiffusion[ggi] = hit ? best : 0.0;
	if (hit) {
		ungriddiffusion[ggi] = best; // [SteeringFeedback] deposit gel chemistry into substrate
		lab[ggi] = 2;
	}
	// no-hit: retain rho_mem seed from calClearLab (substrate memory preserved)
	// else { ungriddiffusion[ggi] = 0.0; }  // [Removed] was clearing memory on no-hit
}

// 仅做扩散/非线性：lab==0 才做演化；否则把插值值直通到输出（关键改动）
__global__ void calGridDiffuseOnly(double* __restrict__ ungrid, const double* __restrict__ ungriddiffusion, const int* __restrict__ lab, SimgridParams gridparams, double dt)
{
	const int gxi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	const int gyi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	const int dlx = gridparams.dlx, dly = gridparams.dly;
	if (gxi >= dlx || gyi >= dly) return;

	const int ggi = get_gnindex(gxi, gyi, 1, gridparams);
	// [Baseline] true original: no contact-zone early return; diffuse all cells uniformly

	const int ggi1 = get_gnindex(gxi + 1, gyi, 1, gridparams);
	const int ggi2 = get_gnindex(gxi - 1, gyi, 1, gridparams);
	const int ggi3 = get_gnindex(gxi, gyi + 1, 1, gridparams);
	const int ggi4 = get_gnindex(gxi, gyi - 1, 1, gridparams);

	const double ind = gridparams.ind;
	const double ind2 = ind * ind;

	const double c = ungriddiffusion[ggi];
	const double d1 = (5.0 * dt / ind2) * (ungriddiffusion[ggi1] + ungriddiffusion[ggi2] - 2.0 * c);
	const double d2 = (5.0 * dt / ind2) * (ungriddiffusion[ggi3] + ungriddiffusion[ggi4] - 2.0 * c);
	const double nl = -5.0 * dt * c * c;

	ungrid[ggi] = c + d1 + d2 + nl;
}



// ========== 4) 欧拉边界：仅把边界的 ungrid 清零 ==========
__global__ void calgridBoundary(double* __restrict__ ungrid, const int* __restrict__ gridmap_node, SimgridParams gridparams)
{
	int gxi = blockIdx.x * blockDim.x + threadIdx.x;
	int gyi = blockIdx.y * blockDim.y + threadIdx.y;
	if (gxi > gridparams.dlx || gyi > gridparams.dly) return;
	const int ggi = get_gnindex(gxi, gyi, 1, gridparams);
	const int node_type = gridmap_node[ggi];
	if (node_type == 0) return;
	ungrid[ggi] = 0.0;
}


// =================== 计算排斥力核 ===================
__global__ void calRepulsion(double2* __restrict__ FFRepulsion1, double2* __restrict__ FFRepulsion2, const double2* __restrict__ my_gel_rn, double2* const* __restrict__ gels_rn,
	int ng, int LX, int LY, SimParams params, SimgridParams gridparams)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int nB = (LX - 1), nR = (LY - 1), nT = (LX - 1), nL = (LY - 1);
	const int perim = nB + nR + nT + nL;
	if (tid >= perim) return;

	enum Edge { BOTTOM = 0, RIGHT = 1, TOP = 2, LEFT = 3 } edge;
	int x = -1, y = -1, t = tid;
	if (t < nB) { x = 2 + t;        y = 1;           edge = BOTTOM; }
	else if ((t -= nB) < nR) { x = LX;           y = 2 + t;       edge = RIGHT; }
	else if ((t -= nR) < nT) { x = (LX - 1) - t;   y = LY;          edge = TOP; }
	else { t -= nT;          x = 1;           y = (LY - 1) - t;  edge = LEFT; }

	const int gi = safe_get_index(x, y, 2, params, LX, LY);
	if (gi < 0) return;

	const double2 my_pos = my_gel_rn[gi];
	double2 force = make_double2(0.0, 0.0);

	// 胶-胶 LJ
	const double www = 1.244455, POSI = 1e-4, rc = 2.0;
	const double rsi = rc / www, rc2 = rc * rc, eps2 = 1e-12;
	const double s = rsi / rc, s2 = s * s, s4 = s2 * s2, s6 = s4 * s2, s12 = s6 * s6;
	const double F_shift = 4.0 * POSI * (1.0 / rc) * (-12.0 * s12 + 6.0 * s6);

	// 几何
	const int Blx = gridparams.bottomlx;
	const int Bly = gridparams.bottomly;

	// —— 胶-胶
	for (int g = 0; g < ng; ++g) {
		const double2* other = gels_rn[g];
		if (other == nullptr || other == my_gel_rn) continue;
		for (int xi = 1; xi <= LX; ++xi) {
			int gi_b = safe_get_index(xi, 1, 2, params, LX, LY);
			if (gi_b >= 0) add_lj_from_pos(other[gi_b], my_pos, rsi, rc2, eps2, F_shift, POSI, force.x, force.y);
			int gi_t = safe_get_index(xi, LY, 2, params, LX, LY);
			if (gi_t >= 0) add_lj_from_pos(other[gi_t], my_pos, rsi, rc2, eps2, F_shift, POSI, force.x, force.y);
		}
		for (int yi = 2; yi <= LY - 1; ++yi) {
			int gi_l = safe_get_index(1, yi, 2, params, LX, LY);
			if (gi_l >= 0) add_lj_from_pos(other[gi_l], my_pos, rsi, rc2, eps2, F_shift, POSI, force.x, force.y);
			int gi_r = safe_get_index(LX, yi, 2, params, LX, LY);
			if (gi_r >= 0) add_lj_from_pos(other[gi_r], my_pos, rsi, rc2, eps2, F_shift, POSI, force.x, force.y);
		}
	}

	// —— 欧拉外墙（四边）：与Fortran一致，使用LJ，rc=2.0*dx=2.0，POSI=1e-4
	// Fortran: rc=2.0d0*dx, POSI=1d-4，Fun()即标准LJ
	{
		const double rc2_wall = rc2;  // rc=2.0，与Fortran rc=2.0*dx相同
		// y=0 底壁
		{
			double dy = my_pos.y, r2w = rc2_wall - dy * dy;
			if (r2w > 0.0) {
				double dxw = sqrt(r2w);
				int x0 = clampi_device((int)floor(my_pos.x - dxw), 0, Blx);
				int x1 = clampi_device((int)ceil(my_pos.x + dxw), 0, Blx);
				for (int gx = x0; gx <= x1; ++gx)
					add_lj_from_pos(make_double2((double)gx, 0.0), my_pos, rsi, rc2_wall, eps2, F_shift, POSI, force.x, force.y);
			}
		}
		// y=Bly 顶壁
		{
			double dy = my_pos.y - (double)Bly, r2w = rc2_wall - dy * dy;
			if (r2w > 0.0) {
				double dxw = sqrt(r2w);
				int x0 = clampi_device((int)floor(my_pos.x - dxw), 0, Blx);
				int x1 = clampi_device((int)ceil(my_pos.x + dxw), 0, Blx);
				for (int gx = x0; gx <= x1; ++gx)
					add_lj_from_pos(make_double2((double)gx, (double)Bly), my_pos, rsi, rc2_wall, eps2, F_shift, POSI, force.x, force.y);
			}
		}
		// x=0 左壁
		{
			double dxw = my_pos.x, r2w = rc2_wall - dxw * dxw;
			if (r2w > 0.0) {
				double dyw = sqrt(r2w);
				int y0 = clampi_device((int)floor(my_pos.y - dyw), 0, Bly);
				int y1 = clampi_device((int)ceil(my_pos.y + dyw), 0, Bly);
				for (int gy = y0; gy <= y1; ++gy)
					add_lj_from_pos(make_double2(0.0, (double)gy), my_pos, rsi, rc2_wall, eps2, F_shift, POSI, force.x, force.y);
			}
		}
		// x=Blx 右壁
		{
			double dxw = my_pos.x - (double)Blx, r2w = rc2_wall - dxw * dxw;
			if (r2w > 0.0) {
				double dyw = sqrt(r2w);
				int y0 = clampi_device((int)floor(my_pos.y - dyw), 0, Bly);
				int y1 = clampi_device((int)ceil(my_pos.y + dyw), 0, Bly);
				for (int gy = y0; gy <= y1; ++gy)
					add_lj_from_pos(make_double2((double)Blx, (double)gy), my_pos, rsi, rc2_wall, eps2, F_shift, POSI, force.x, force.y);
			}
		}
	}

	// 写回
	if (edge == BOTTOM || edge == TOP)  FFRepulsion1[gi] = force;
	else                            FFRepulsion2[gi] = force;
}


__global__ void calrecordCenterElementD(double* um_center, double* vm_center, double* wm_center, double2* rm_center, double2* Fn_center, double2* Veln_center, double* um, double* vm, double* wm, double2* rn, double2* Fn, double2* Veln, int time, SimParams params)
{
	int gi = get_index(params.LX / 2, params.LY / 2, 1, params);
	int hi = get_index((params.LX + 1) / 2, (params.LY + 1) / 2, 2, params);
	um_center[time] = um[gi];
	vm_center[time] = vm[gi];
	wm_center[time] = wm[gi];
	rm_center[time] = rn[hi];
	Fn_center[time] = Fn[hi];
	Veln_center[time] = Veln[hi];
}

__global__ void calFilamentD(double* vn, double* un, double2* filament, int time, unsigned int* hitCnt, SimParams params)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	if (xi > params.LX - 1 || yi > params.LY - 1) {
		return;
	}
	double Viso = 0.15;
	//   ȡ 8    ֵ
	double u00 = un[get_index(xi, yi, 2, params)];
	double u10 = un[get_index(xi + 1, yi, 2, params)];
	double u01 = un[get_index(xi, yi + 1, 2, params)];
	double u11 = un[get_index(xi + 1, yi + 1, 2, params)];
	double v00 = vn[get_index(xi, yi, 2, params)];
	double v10 = vn[get_index(xi + 1, yi, 2, params)];
	double v01 = vn[get_index(xi, yi + 1, 2, params)];
	double v11 = vn[get_index(xi + 1, yi + 1, 2, params)];

	double xb, yb;
	bool ok = solveBilinear(u00, u10, u01, u11, v00, v10, v01, v11, Viso, xb, yb);
	if (ok) {
		//    
		unsigned int gi = atomicAdd(hitCnt, 1u);
		filament[time * params.maxFilamentlen + gi] = make_double2(xi + xb, yi + yb);
		//printf("filax=%.10f,filay=%.10f\n", filament[time * params.maxFilamentlen + gi].x, filament[time * params.maxFilamentlen + gi].y);
	}
}
