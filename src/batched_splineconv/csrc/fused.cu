// Fused fixed-mesh SplineConv kernels.
//
// Notation: rows m = i * B + b enumerate (destination node, sample). The operator is stored
// as a CSR over rows r = i * K + k (node-major, then kernel index), entries (j, coef):
//     Z_k[(i, b), c] = sum_{(j, coef) in row(i, k)} coef * X[j, b, c]
// Forward:   out[(i, b), :] = sum_k Z_k[(i, b), :] @ W_k          (Z is never stored)
// dX:        same kernel on the transposed operator with W_k^T
// dW_k:      sum_{(i, b)} Z_k[(i, b), :]^T dOut[(i, b), :]       (split-K, deterministic)
//
// Tiles of Z_k are gathered into shared memory, then multiplied with a register-tiled
// SGEMM micro-kernel. Tile shapes are template parameters chosen from the channel counts so
// that narrow layers (C = 16, 32) do not waste FMAs on padding.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

namespace {

// ---------------------------------------------------------------------------------------
// gather_gemm: block = BM rows (i, b) x BN output channels, BK input channels per step,
// thread tile TM x TN, (BM/TM) * (BN/TN) threads.
// X: [N_src, B, Cin]  W: [K, Cin, Cout]  out: [N_dst, B, Cout]
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) gather_gemm_kernel(
    const float* __restrict__ X, const float* __restrict__ W,
    const int* __restrict__ ptr, const int* __restrict__ col, const float* __restrict__ val,
    float* __restrict__ out, int N_dst, int B, int Cin, int Cout, int K) {
  constexpr int NT = (BM / TM) * (BN / TN);
  constexpr int PAD = 4;  // keeps float4 alignment, limits bank conflicts on the gather writes
  __shared__ __align__(16) float Zs[BK][BM + PAD];
  __shared__ __align__(16) float Ws[BK][BN];

  const int M = N_dst * B;
  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;
  const int tid = threadIdx.x;
  const int tr = tid / (BN / TN);
  const int tc = tid % (BN / TN);

  float acc[TM][TN];
#pragma unroll
  for (int x = 0; x < TM; ++x)
#pragma unroll
    for (int y = 0; y < TN; ++y) acc[x][y] = 0.f;

  for (int k = 0; k < K; ++k) {
    for (int c0 = 0; c0 < Cin; c0 += BK) {
      // gather Z_k tile [BM, BK] (input channel fastest: coalesced reads of X)
      int nonempty = 0;
#pragma unroll 4
      for (int e = tid; e < BM * BK; e += NT) {
        const int c = e % BK, ml = e / BK, m = m0 + ml;
        float z = 0.f;
        if (m < M && c0 + c < Cin) {
          const int i = m / B, b = m - i * B;
          const int r = i * K + k;
          const int p1 = __ldg(ptr + r + 1);
          for (int p = __ldg(ptr + r); p < p1; ++p) {
            z = fmaf(__ldg(val + p),
                     __ldg(X + ((long long)__ldg(col + p) * B + b) * Cin + c0 + c), z);
            nonempty = 1;
          }
        }
        Zs[c][ml] = z;
      }
      // skip the GEMM when no row of the tile has a neighbour in kernel cell k
      if (!__syncthreads_or(nonempty)) continue;

      for (int e = tid; e < BK * BN; e += NT) {
        const int n = e % BN, c = e / BN;
        Ws[c][n] = (c0 + c < Cin && n0 + n < Cout)
                       ? __ldg(W + ((long long)k * Cin + c0 + c) * Cout + n0 + n) : 0.f;
      }
      __syncthreads();
#pragma unroll
      for (int c = 0; c < BK; ++c) {
        float a[TM], w[TN];
#pragma unroll
        for (int x = 0; x < TM; x += 4)
          *reinterpret_cast<float4*>(a + x) = *reinterpret_cast<const float4*>(&Zs[c][tr * TM + x]);
#pragma unroll
        for (int y = 0; y < TN; y += 4)
          *reinterpret_cast<float4*>(w + y) = *reinterpret_cast<const float4*>(&Ws[c][tc * TN + y]);
#pragma unroll
        for (int x = 0; x < TM; ++x)
#pragma unroll
          for (int y = 0; y < TN; ++y) acc[x][y] = fmaf(a[x], w[y], acc[x][y]);
      }
      __syncthreads();
    }
  }
#pragma unroll
  for (int x = 0; x < TM; ++x) {
    const int m = m0 + tr * TM + x;
    if (m >= M) continue;
#pragma unroll
    for (int y = 0; y < TN; ++y) {
      const int n = n0 + tc * TN + y;
      if (n < Cout) out[(long long)m * Cout + n] = acc[x][y];
    }
  }
}

// ---------------------------------------------------------------------------------------
// wgrad: partial[s, k, Cin, Cout] = sum over rows m in split s of Z_k[m, :]^T G[m, :]
// block = BC input channels x BN output channels for one (k, split), RK rows per step.
template <int BC, int BN, int RK, int TM, int TN>
__global__ void __launch_bounds__((BC / TM) * (BN / TN)) wgrad_kernel(
    const float* __restrict__ X, const float* __restrict__ G,
    const int* __restrict__ ptr, const int* __restrict__ col, const float* __restrict__ val,
    float* __restrict__ partial, int N_dst, int B, int Cin, int Cout, int K, int splits) {
  constexpr int NT = (BC / TM) * (BN / TN);
  __shared__ __align__(16) float Zs[RK][BC];
  __shared__ __align__(16) float Gs[RK][BN];

  const int M = N_dst * B;
  const int c0 = blockIdx.x * BC;
  const int n0 = blockIdx.y * BN;
  const int k = blockIdx.z % K;
  const int s = blockIdx.z / K;
  const int rows_per_split = ((M + splits - 1) / splits + RK - 1) / RK * RK;
  const int mbeg = s * rows_per_split;
  const int mend = min(M, mbeg + rows_per_split);

  const int tid = threadIdx.x;
  const int tr = tid / (BN / TN);
  const int tc = tid % (BN / TN);
  float acc[TM][TN];
#pragma unroll
  for (int x = 0; x < TM; ++x)
#pragma unroll
    for (int y = 0; y < TN; ++y) acc[x][y] = 0.f;

  for (int mb = mbeg; mb < mend; mb += RK) {
    int nonempty = 0;
#pragma unroll 4
    for (int e = tid; e < RK * BC; e += NT) {
      const int c = e % BC, rl = e / BC, m = mb + rl;
      float z = 0.f;
      if (m < mend && c0 + c < Cin) {
        const int i = m / B, b = m - i * B;
        const int r = i * K + k;
        const int p1 = __ldg(ptr + r + 1);
        for (int p = __ldg(ptr + r); p < p1; ++p) {
          z = fmaf(__ldg(val + p),
                   __ldg(X + ((long long)__ldg(col + p) * B + b) * Cin + c0 + c), z);
          nonempty = 1;
        }
      }
      Zs[rl][c] = z;
    }
    if (!__syncthreads_or(nonempty)) continue;
    for (int e = tid; e < RK * BN; e += NT) {
      const int n = e % BN, rl = e / BN, m = mb + rl;
      Gs[rl][n] = (m < mend && n0 + n < Cout) ? __ldg(G + (long long)m * Cout + n0 + n) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int rl = 0; rl < RK; ++rl) {
      float a[TM], g[TN];
#pragma unroll
      for (int x = 0; x < TM; x += 4)
        *reinterpret_cast<float4*>(a + x) = *reinterpret_cast<const float4*>(&Zs[rl][tr * TM + x]);
#pragma unroll
      for (int y = 0; y < TN; y += 4)
        *reinterpret_cast<float4*>(g + y) = *reinterpret_cast<const float4*>(&Gs[rl][tc * TN + y]);
#pragma unroll
      for (int x = 0; x < TM; ++x)
#pragma unroll
        for (int y = 0; y < TN; ++y) acc[x][y] = fmaf(a[x], g[y], acc[x][y]);
    }
    __syncthreads();
  }
  float* P = partial + ((long long)s * K + k) * Cin * Cout;
#pragma unroll
  for (int x = 0; x < TM; ++x) {
    const int c = c0 + tr * TM + x;
    if (c >= Cin) continue;
#pragma unroll
    for (int y = 0; y < TN; ++y) {
      const int n = n0 + tc * TN + y;
      if (n < Cout) P[(long long)c * Cout + n] = acc[x][y];
    }
  }
}

void check(const torch::Tensor& t, const char* name, torch::ScalarType dt) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dt, name, " has the wrong dtype");
}

template <int BM, int BN, int BK, int TM, int TN>
void launch_gather_gemm(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& ptr,
                        const torch::Tensor& col, const torch::Tensor& val, torch::Tensor& out,
                        int n_dst, int B, int Cin, int Cout, int K) {
  const long long M = (long long)n_dst * B;
  dim3 grid((unsigned)((M + BM - 1) / BM), (unsigned)((Cout + BN - 1) / BN));
  gather_gemm_kernel<BM, BN, BK, TM, TN>
      <<<grid, (BM / TM) * (BN / TN), 0, at::cuda::getCurrentCUDAStream()>>>(
          x.data_ptr<float>(), w.data_ptr<float>(), ptr.data_ptr<int>(), col.data_ptr<int>(),
          val.data_ptr<float>(), out.data_ptr<float>(), n_dst, B, Cin, Cout, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BC, int BN, int RK, int TM, int TN>
torch::Tensor launch_wgrad(const torch::Tensor& x, const torch::Tensor& g, const torch::Tensor& ptr,
                           const torch::Tensor& col, const torch::Tensor& val,
                           int n_dst, int B, int Cin, int Cout, int K) {
  const long long M = (long long)n_dst * B;
  const long long tiles = (long long)((Cin + BC - 1) / BC) * ((Cout + BN - 1) / BN) * K;
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int splits = (int)std::max<long long>(
      1, std::min<long long>((8LL * sms + tiles - 1) / tiles, (M + 4 * RK - 1) / (4 * RK)));
  auto partial = torch::empty({splits, K, Cin, Cout}, x.options());
  dim3 grid((Cin + BC - 1) / BC, (Cout + BN - 1) / BN, K * splits);
  wgrad_kernel<BC, BN, RK, TM, TN>
      <<<grid, (BC / TM) * (BN / TN), 0, at::cuda::getCurrentCUDAStream()>>>(
          x.data_ptr<float>(), g.data_ptr<float>(), ptr.data_ptr<int>(), col.data_ptr<int>(),
          val.data_ptr<float>(), partial.data_ptr<float>(), n_dst, B, Cin, Cout, K, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return partial.sum(0);
}

}  // namespace

// x: [N_src, B, Cin], weight: [K, Cin, Cout] -> [N_dst, B, Cout]
torch::Tensor gather_gemm(torch::Tensor x, torch::Tensor weight, torch::Tensor ptr,
                          torch::Tensor col, torch::Tensor val, int64_t n_dst) {
  check(x, "x", torch::kFloat32); check(weight, "weight", torch::kFloat32);
  check(ptr, "ptr", torch::kInt32); check(col, "col", torch::kInt32);
  check(val, "val", torch::kFloat32);
  const c10::cuda::CUDAGuard guard(x.device());
  const int B = x.size(1), Cin = x.size(2), K = weight.size(0), Cout = weight.size(2);
  TORCH_CHECK(weight.size(1) == Cin, "channel mismatch");
  TORCH_CHECK(ptr.numel() == n_dst * K + 1, "operator / kernel size mismatch");
  auto out = torch::empty({n_dst, B, Cout}, x.options());
  if ((long long)n_dst * B == 0) return out;
  const int nd = (int)n_dst;
#define GG(BM, BN, BK) launch_gather_gemm<BM, BN, BK, 4, 4>(x, weight, ptr, col, val, out, nd, B, Cin, Cout, K)
#define GG_BK(BM, BN)                      \
  if (Cin <= 8) GG(BM, BN, 8);             \
  else if (Cin <= 16) GG(BM, BN, 16);      \
  else GG(BM, BN, 32);
  if (Cout <= 16) { GG_BK(256, 16) }
  else if (Cout <= 32) { GG_BK(128, 32) }
  else { GG_BK(64, 64) }
#undef GG_BK
#undef GG
  return out;
}

// x: [N_src, B, Cin], grad_out: [N_dst, B, Cout] -> dW [K, Cin, Cout]
torch::Tensor wgrad(torch::Tensor x, torch::Tensor grad_out, torch::Tensor ptr,
                    torch::Tensor col, torch::Tensor val, int64_t K) {
  check(x, "x", torch::kFloat32); check(grad_out, "grad_out", torch::kFloat32);
  check(ptr, "ptr", torch::kInt32); check(col, "col", torch::kInt32);
  check(val, "val", torch::kFloat32);
  const c10::cuda::CUDAGuard guard(x.device());
  const int n_dst = grad_out.size(0), B = x.size(1), Cin = x.size(2), Cout = grad_out.size(2);
  const int k = (int)K;
#define WG(BC, BN) return launch_wgrad<BC, BN, 32, 4, 4>(x, grad_out, ptr, col, val, n_dst, B, Cin, Cout, k)
  if (Cin <= 16 && Cout <= 16) WG(16, 16);
  if (Cin <= 16) WG(16, 64);
  if (Cout <= 16) WG(64, 16);
  if (Cin <= 32 && Cout <= 32) WG(32, 32);
  WG(64, 64);
#undef WG
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_gemm", &gather_gemm, "fused gather + GEMM over kernel cells");
  m.def("wgrad", &wgrad, "weight gradient, split-K");
}
