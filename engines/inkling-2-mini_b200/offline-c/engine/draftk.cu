// The DSpark draft head (mtp.safetensors) as a fixed sequence of CUDA kernels: one draft-and-verify cycle's draft
// side (acceptance, the draft's context K/V of the verify rows, the block of 8 rows through 3 layers, the Markov
// recurrence -> 7 drafts). Same function as engine/draft.py (Draft, torch), up to summation order: BF16 storage and
// GEMM operands, FP32 accumulation / norms / RoPE / softmax / gate math. Weights in m16n8k16 A-fragment order
// (draft.py frag_bf16), GEMVs on the tensor cores with up to 8 activation columns. Compiled without fast math.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

typedef __nv_bfloat16 bf16;
#define H 2304
#define NH 24
#define NKV 8
#define HD 128
#define NB 8
#define RK 256
#define VOC 200058
#define VSTRIPS 12504
#define VP (VSTRIPS * 16)
#define QKVN 5120
#define FF 9216
#ifndef NSPLIT
#define NSPLIT 32
#endif
#define FOLD_BLOCKS ((VSTRIPS + 7) / 8)
#define EPS 1e-6f

struct DkL { const uint8_t *qkv, *kvc, *o, *gu, *dn; const float *in_ln, *post_ln, *dflare, *q_norm, *k_norm, *k_norm_ctx; bf16 *kc, *vc;
             const uint8_t *qkv8, *kvc8, *o8, *gu8, *dn8; const float *qkv_s, *kvc_s, *o_s, *gu_s, *dn_s; };   // FP8 copies (f8 bit 2)
struct Dk {
  DkL L[3];
  const uint8_t *wctx, *wrec, *un, *fold;
  const bf16 *fold_b, *w1, *embed, *mask_embed;
  const float *rnn_b, *embed_norm, *norm, *m_norm, *alpha, *inv_freq;
  int *vtok, *vpos; const int* tgt; const bf16* feats; int* emit;
  long long *c_new, *anchor, *frow; const long long* pos64;
  int* st;   // [0] c, [1] a, [2] c_new, [3] anchor, [4] frow, [5] first skipped Markov step, [6] n-gram proposal used,
             // [8 + r] context position of row r
  bf16 *u, *f, *xres, *h, *kvraw, *qkv, *q, *att, *act, *xm, *base, *state;
  float *part, *opart, *ctxo, *raw;
  unsigned long long* amax;
  const uint2* pbt; const bf16* fout;   // mode 0 with pbt set: the features straight from the verify pass (pbt rows 5, 23, 41 + fout)
  float* lse;     // [7][FOLD_BLOCKS] per-block sums of exp(lg - 30) of each Markov step (fixed-order sums in k_end)
  float* pconf;   // [7] the draft's top-1 probability per step
  unsigned* acnt; // [8] attention splits finished per kv head (the last one merges; it resets the counter)
  float* cum;     // [7] cumulative products (written by the recurrent GEMV of the next step); st[5] = first skipped step
  long long T, mode, nrows, pad;
  const uint8_t *un8, *fold8; const float *un_s, *fold_s;   // FP8 E4M3 copies (fragment order) + per-row scales
  int* hist;      // [hT] committed tokens by position (n-gram drafts; -1 = none)
  long long f8;   // bit 0: base unembedding from un8, bit 1: fold from fold8
  long long ngmin, hT;   // n-gram drafts: minimum match length (> 8: off), history length
  double theta;   // draft length: the largest k with prod_{i<=k} p_i >= theta (k >= 1); next verify rows = k + 1
};

// programmatic dependent launch: let the next kernel start (its blocks wait in gdc_wait for this grid's completion
// and memory); every kernel waits before touching data of earlier kernels (so completion is transitive)
__device__ __forceinline__ void gdc_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void gdc_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
// L2 prefetch of slice `part` of `nparts` of a weight matrix (bytes a multiple of 2048), issued by one thread
__device__ __forceinline__ void pf_slice(const uint8_t* w, long bytes, int part, int nparts) {
  const long u = bytes / 2048, b0 = u * part / nparts * 2048, b1 = u * (part + 1) / nparts * 2048;
  for (long o = b0; o < b1; o += 65536) {
    const unsigned n = (unsigned)min(65536L, b1 - o);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(w + o), "r"(n) : "memory");
  }
}
#define O_BYTES (144L * 48 * 2048)
#define GU_BYTES (1152L * 36 * 2048)
#define DN_BYTES (144L * 144 * 2048)
#define QKV_BYTES (320L * 36 * 2048)
#ifndef FUSE_POST
#define FUSE_POST 0   // q / k norm + RoPE of the block rows inside k_attn: +8 us per cycle (redundant per split), off
#endif
#ifndef FUSE_MERGE
#define FUSE_MERGE 0  // last split merges (atomic counter): +36 us per cycle (serial merge on one block), off
#endif
#ifndef PF_A
#define PF_A 0
#endif
#ifndef PF_B
#define PF_B 0
#endif
__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 tobf(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rbf(float x) { return bf(tobf(x)); }
__device__ __forceinline__ uint4 ldg_nc(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// two E4M3 codes (low byte = first element) -> f16x2
__device__ __forceinline__ uint32_t e4m3x2(uint32_t w16) {
  uint32_t r; unsigned short h = (unsigned short)w16;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(h));
  return r;
}
// one FP8 fragment word pair (8 codes of a k16 tile) on the f16 tensor cores
__device__ __forceinline__ void mma_e4m3(float* acc, uint32_t wa, uint32_t wb, uint32_t b0, uint32_t b1) {
  mma_f16(acc, e4m3x2(wa & 0xffffu), e4m3x2(wa >> 16), e4m3x2(wb & 0xffffu), e4m3x2(wb >> 16), b0, b1);
}
__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t v) {
  const __half2 h = __floats2half2_rn(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ unsigned long long fkey(float z, int idx) {   // larger value, then lower index, wins
  unsigned u = __float_as_uint(z);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((unsigned long long)u << 32) | (unsigned)(0xFFFFFFFFu - (unsigned)idx);
}
__device__ __forceinline__ int kidx(unsigned long long k) { return (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFu)); }

template <int NT>
__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; i++) s += red[i];
  return s;
}

// (one warp) the draft's top-1 probability of Markov step i: exp(zmax - 30) / sum exp(lg - 30), the sum in a fixed order
__device__ __forceinline__ float step_p(const Dk& d, int i) {
  const int lane = threadIdx.x & 31;
  float s = 0.f;
  for (int k = lane; k < FOLD_BLOCKS; k += 32) s += d.lse[i * FOLD_BLOCKS + k];
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const unsigned u = (unsigned)(d.amax[i] >> 32);
  const float zmax = __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
  return expf(zmax - 30.0f) / s;
}

// ---------------------------------------------------------------------------------------------- cycle state
// every block: c, a, c_new, anchor, frow of this cycle
struct CS { int c, a, cn, anchor, frow; };
__device__ __forceinline__ CS cycle_state(const Dk& d) {
  CS s;
  if (d.mode == 0) {
    s.c = d.vpos[0];
    const int kp = min(max(d.vpos[1], 1), NB);   // rows of the verify pass that just ran: rows >= kp are stale
    int a = 0;
    while (a < kp - 1 && d.vtok[1 + a] == d.tgt[a]) a++;
    s.a = a; s.cn = s.c + a + 1; s.anchor = d.tgt[a]; s.frow = a;
  } else {
    s.c = 0; s.a = 0; s.cn = (int)d.c_new[0]; s.anchor = (int)d.anchor[0]; s.frow = (int)d.frow[0];
  }
  return s;
}

// feature j (0..3) of verify row r, elements k, k + 1 (k even)
__device__ __forceinline__ float2 feat2(const Dk& d, int r, int j, int k) {
  if (d.mode == 0 && d.pbt) {
    if (j == 3) return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(d.fout + (long)r * H + k));
    const int idx = j == 0 ? 5 : (j == 1 ? 23 : 41);
    const unsigned v = d.pbt[((long)idx * NB + r) * (H / 2) + k / 2].x;
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
  }
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(d.feats + ((long)r * 4 + j) * H + k));
}
// K1: blocks 0..23 context mixes u[l][r]; block 24 the Markov feature mix f; blocks 25..32 the block's input rows
#define K1_NT 256
#define EPT1 9   // 2304 / 256
#define MIXP 5   // element pairs per thread (1152 pairs / 256, rounded up)
__device__ __forceinline__ float2 feat2(const Dk& d, int r, int j, int k);
// m = sum_j alpha_j F_j (FP32, j in order) for the element pairs of this thread; ss += sum m^2
__device__ __forceinline__ void mix_row(const Dk& d, int r, const float* al, float* m, float& ss) {
#pragma unroll
  for (int e = 0; e < MIXP; e++) {
    const int k = 2 * (threadIdx.x + e * K1_NT);
    m[2 * e] = m[2 * e + 1] = 0.f;
    if (k >= H) continue;
    float2 f0 = feat2(d, r, 0, k), f1 = feat2(d, r, 1, k), f2 = feat2(d, r, 2, k), f3 = feat2(d, r, 3, k);
    float v0 = f0.x * al[0], v1 = f0.y * al[0];
    v0 += f1.x * al[1]; v1 += f1.y * al[1];
    v0 += f2.x * al[2]; v1 += f2.y * al[2];
    v0 += f3.x * al[3]; v1 += f3.y * al[3];
    m[2 * e] = v0; m[2 * e + 1] = v1; ss += v0 * v0 + v1 * v1;
  }
}
__global__ void __launch_bounds__(K1_NT) k_start(const Dk d) {
  gdc_launch(); gdc_wait();
  __shared__ float red[32];
  const CS s = cycle_state(d);
  const int b = blockIdx.x, tid = threadIdx.x;
  if (b == 0 && tid == 0) {
    if (d.mode == 0) {
      d.emit[0] = s.a + 1;
      for (int i = 0; i < NB; i++) d.emit[1 + i] = d.tgt[i];
      d.c_new[0] = s.cn; d.anchor[0] = s.anchor; d.frow[0] = s.frow;
    }
    d.st[0] = s.c; d.st[1] = s.a; d.st[2] = s.cn; d.st[3] = s.anchor; d.st[4] = s.frow; d.st[5] = NB - 1; d.st[6] = 0;
    for (int r = 0; r < NB; r++) d.st[8 + r] = d.mode == 1 ? (int)d.pos64[r < d.nrows ? r : 0] : s.c + r;
  }
  if (b == 0 && tid < NB && d.mode != 1) d.amax[tid] = 0ull;
  if (b == 0 && d.mode != 1) d.state[tid] = tobf(0.f);   // the Markov state s_0 = 0 (K1_NT = RK threads)
  if (b < 24) {
    if (d.mode == 2) return;
    const int l = b / NB, r = b % NB;
    if (r >= d.nrows) return;
    float m[2 * MIXP], ss = 0.f;
    mix_row(d, r, d.alpha + l * 4, m, ss);
    const float rs = rsqrtf(block_sum<K1_NT>(ss, red) / (float)H + EPS);
    const float* w = d.L[l].dflare;
#pragma unroll
    for (int e = 0; e < MIXP; e++) {
      const int k = 2 * (tid + e * K1_NT);
      if (k < H) { d.u[((long)l * NB + r) * H + k] = tobf(m[2 * e] * rs * w[k]); d.u[((long)l * NB + r) * H + k + 1] = tobf(m[2 * e + 1] * rs * w[k + 1]); }
    }
    return;
  }
  if (d.mode == 1) return;
  if (b == 24) {
    float m[2 * MIXP], ss = 0.f;
    mix_row(d, s.frow, d.alpha + 3 * 4, m, ss);
    const float rs = rsqrtf(block_sum<K1_NT>(ss, red) / (float)H + EPS);
#pragma unroll
    for (int e = 0; e < MIXP; e++) {
      const int k = 2 * (tid + e * K1_NT);
      if (k < H) { d.f[k] = tobf(m[2 * e] * rs * d.m_norm[k]); d.f[k + 1] = tobf(m[2 * e + 1] * rs * d.m_norm[k + 1]); }
    }
    return;
  }
  const int r = b - 25;
  float x[EPT1];
  if (r == 0) {
    float ss = 0.f;
    const bf16* er = d.embed + (long)s.anchor * H;
#pragma unroll
    for (int e = 0; e < EPT1; e++) { x[e] = bf(er[tid + e * K1_NT]); ss += x[e] * x[e]; }
    const float rs = rsqrtf(block_sum<K1_NT>(ss, red) / (float)H + EPS);
#pragma unroll
    for (int e = 0; e < EPT1; e++) x[e] = rbf(x[e] * rs * d.embed_norm[tid + e * K1_NT]);
  } else {
#pragma unroll
    for (int e = 0; e < EPT1; e++) x[e] = bf(d.mask_embed[tid + e * K1_NT]);
  }
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < EPT1; e++) ss += x[e] * x[e];
  const float rs = rsqrtf(block_sum<K1_NT>(ss, red) / (float)H + EPS);
  const float* w = d.L[0].in_ln;
#pragma unroll
  for (int e = 0; e < EPT1; e++) {
    const int k = tid + e * K1_NT;
    d.xres[(long)r * H + k] = tobf(x[e]);
    d.h[(long)r * H + k] = tobf(x[e] * rs * w[k]);
  }
}

// ---------------------------------------------------------------------------------------------- GEMV
#define E_BF16 0
#define E_PART 1
#define E_ACT 2
#define E_F32 3
struct GJ {
  const uint8_t* W; const bf16* X; const bf16* X2; void* out; const float* bias; const float* bp;   // E_F32: + (sum of nbp partials bp[q * NB * 768]) + bias
  const float* wsc; int fp8, ng;   // ng: nothing to do when the cycle's proposal is the n-gram one (st[6])   // fp8: W holds E4M3 codes in FP8 fragment order, wsc the per-row scales (B operand as f16)
  int nbp, xs, ksplit_at, ncol, strips, U, ks, epi, os, rec, sb;   // rec >= 0: the Markov step's recurrent GEMV; sb strips per block
};
struct GJs { GJ j[3]; int n; };
__host__ __device__ __forceinline__ int gj_blocks(const GJ& j) { return (j.strips + j.sb - 1) / j.sb * j.ks; }

__device__ __forceinline__ int n0w(int strip) { return strip * 16 + ((threadIdx.x & 31) >> 2); }
__device__ __forceinline__ void gemv_epi(const GJ& j, int strip, int split, const float* v, const float* bpre = nullptr) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n0 = strip * 16 + g;
  if (j.epi == E_ACT) {
    const int idx = strip * 8 + g;
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int m = 2 * t + q;
      if (m >= j.ncol) continue;
      const float ga = rbf(v[q]), up = rbf(v[2 + q]);
      const float si = __fdiv_rn(ga, __fadd_rn(1.0f, expf(-ga)));
      reinterpret_cast<bf16*>(j.out)[(long)m * j.os + idx] = tobf(__fmul_rn(si, up));
    }
    return;
  }
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const int m = 2 * t + (q & 1), n = n0 + 8 * (q >> 1);
    if (m >= j.ncol) continue;
    if (j.epi == E_BF16) reinterpret_cast<bf16*>(j.out)[(long)m * j.os + n] = tobf(v[q]);   // (FP8 row scales applied by the caller)
    else if (j.epi == E_PART) reinterpret_cast<float*>(j.out)[((long)split * NB + m) * j.os + n] = v[q];
    else {
      float b = 0.f;
      if (j.bp) { if (bpre) b = bpre[q >> 1]; else { for (int k = 0; k < j.nbp; k++) b += j.bp[(long)k * NB * 768 + n]; b = __fadd_rn(b, j.bias[n]); } }
      else if (j.bias) b = j.bias[n];
      reinterpret_cast<float*>(j.out)[(long)m * j.os + n] = (j.bp || j.bias) ? __fadd_rn(v[q], b) : v[q];
    }
  }
}

// block: sb strips x one k-split; the split's columns of X (8 rows) staged in smem (rows padded by 8 elements: the
// B-fragment loads are bank-conflict free); warps split the k-units and reuse each B fragment for the sb strips
#ifndef G_NW
#define G_NW 8
#endif
#ifndef G_SBMAX
#define G_SBMAX 4
#endif
#ifndef SM_MINB
#define SM_MINB 2   // <= 128 registers: 2 blocks per SM (129+ registers halve the occupancy: +36 us per cycle)
#endif
template <bool F8>
__global__ void __launch_bounds__(G_NW * 32, SM_MINB) k_gemv(const Dk d, const GJs js) {
  extern __shared__ __align__(16) unsigned char gsm[];
  __shared__ float4 red[G_SBMAX][G_NW][32];
  int bid = blockIdx.x, ji = 0;
  while (ji < js.n - 1 && bid >= gj_blocks(js.j[ji])) { bid -= gj_blocks(js.j[ji]); ji++; }
  GJ j = js.j[ji];
  gdc_launch();
  if (bid >= gj_blocks(j)) { gdc_wait(); return; }
  const bool f8j = F8 && j.fp8;
  const int sg = bid / j.ks, split = bid % j.ks;
  const int s0 = sg * j.sb, ns = min(j.sb, j.strips - s0);
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int ub0 = j.U * split / j.ks, ub1 = j.U * (split + 1) / j.ks, nu = ub1 - ub0;
  const int kr = nu * 64, RS = kr + 8;
  const int u0 = nu * w / G_NW, u1 = nu * (w + 1) / G_NW;
  const uint8_t* wbase = j.W + ((long)s0 * j.U + ub0) * 2048 + lane * 16;
  uint4 wv[G_SBMAX][4];   // the first unit's weights before waiting for the previous kernel
  const uint8_t* wbase8 = j.W + ((long)s0 * j.U + ub0) * 1024 + lane * 16;
  if (u0 < u1)
#pragma unroll
    for (int sb = 0; sb < G_SBMAX; sb++)
      if (sb < ns) {
        if (f8j) {
#pragma unroll
          for (int q = 0; q < 2; q++) wv[sb][q] = ldg_nc(wbase8 + ((long)sb * j.U + u0) * 1024 + q * 512);
        } else {
#pragma unroll
          for (int q = 0; q < 4; q++) wv[sb][q] = ldg_nc(wbase + ((long)sb * j.U + u0) * 2048 + q * 512);
        }
      }
  gdc_wait();
  if (j.ng && d.st[6]) return;   // the n-gram proposal is used: nothing to compute
  if (j.rec >= 0) {
    const int i = j.rec;
    const int prev = i == 0 ? d.st[3] : kidx(d.amax[i - 1]);
    j.X = d.state + (i & 1) * RK;
    j.X2 = d.w1 + (long)prev * RK;
  }
  bf16* xsm = reinterpret_cast<bf16*>(gsm);
  for (int i = tid; i < NB * (kr / 8); i += G_NW * 32) {
    const int m = i / (kr / 8), c8 = (i % (kr / 8)) * 8, k = ub0 * 64 + c8;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (m < j.ncol) {
      const bf16* src = (j.X2 && k >= j.ksplit_at) ? j.X2 + (long)m * j.xs + (k - j.ksplit_at) : j.X + (long)m * j.xs + k;
      v = *reinterpret_cast<const uint4*>(src);
      if (f8j) { v.x = bf2_to_h2(v.x); v.y = bf2_to_h2(v.y); v.z = bf2_to_h2(v.z); v.w = bf2_to_h2(v.w); }
    }
    *reinterpret_cast<uint4*>(xsm + m * RS + c8) = v;
  }
  __syncthreads();
  float acc[G_SBMAX][4];
#pragma unroll
  for (int sb = 0; sb < G_SBMAX; sb++) acc[sb][0] = acc[sb][1] = acc[sb][2] = acc[sb][3] = 0.f;
  for (int u = u0; u < u1; u++) {
    uint32_t bfr[4][2];
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const bf16* pb = xsm + g * RS + u * 64 + q * 16 + 2 * t;
      bfr[q][0] = *reinterpret_cast<const uint32_t*>(pb);
      bfr[q][1] = *reinterpret_cast<const uint32_t*>(pb + 8);
    }
    if (f8j) {
      if (u > u0)
#pragma unroll
        for (int sb = 0; sb < G_SBMAX; sb++)
          if (sb < ns)
#pragma unroll
            for (int q = 0; q < 2; q++) wv[sb][q] = ldg_nc(wbase8 + ((long)sb * j.U + u) * 1024 + q * 512);
#pragma unroll
      for (int sb = 0; sb < G_SBMAX; sb++)
        if (sb < ns)
#pragma unroll
          for (int q = 0; q < 2; q++) {
            mma_e4m3(acc[sb], wv[sb][q].x, wv[sb][q].y, bfr[2 * q][0], bfr[2 * q][1]);
            mma_e4m3(acc[sb], wv[sb][q].z, wv[sb][q].w, bfr[2 * q + 1][0], bfr[2 * q + 1][1]);
          }
      continue;
    }
    if (u > u0)
#pragma unroll
      for (int sb = 0; sb < G_SBMAX; sb++)
        if (sb < ns)
#pragma unroll
          for (int q = 0; q < 4; q++) wv[sb][q] = ldg_nc(wbase + ((long)sb * j.U + u) * 2048 + q * 512);
#pragma unroll
    for (int sb = 0; sb < G_SBMAX; sb++)
      if (sb < ns)
#pragma unroll
        for (int q = 0; q < 4; q++) mma_bf16(acc[sb], wv[sb][q].x, wv[sb][q].y, wv[sb][q].z, wv[sb][q].w, bfr[q][0], bfr[q][1]);
  }
#pragma unroll
  for (int sb = 0; sb < G_SBMAX; sb++) if (sb < ns) red[sb][w][lane] = make_float4(acc[sb][0], acc[sb][1], acc[sb][2], acc[sb][3]);
  __syncthreads();
  if (w >= ns) return;
  const int strip = s0 + w;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < G_NW; k++) { const float4 r = red[w][k][lane]; v[0] += r.x; v[1] += r.y; v[2] += r.z; v[3] += r.w; }
  if (f8j && j.wsc) { const float s0 = j.wsc[n0w(strip)], s1 = j.wsc[n0w(strip) + 8]; v[0] *= s0; v[1] *= s0; v[2] *= s1; v[3] *= s1; }
  gemv_epi(j, strip, split, v);
}

// L1 variant (sb = 1): the B fragments straight from global memory (L1-resident X), warps split the strip's k-units
#define G_UNR 4
#ifndef L1_NW
#define L1_NW 8
#endif
#ifndef L1_MINB
#define L1_MINB (32 / L1_NW)
#endif
__global__ void __launch_bounds__(L1_NW * 32, L1_MINB) k_gemv_l1(const Dk d, const GJs js) {
  __shared__ float4 red[L1_NW][32];
  int bid = blockIdx.x, ji = 0;
  while (ji < js.n - 1 && bid >= js.j[ji].strips * js.j[ji].ks) { bid -= js.j[ji].strips * js.j[ji].ks; ji++; }
  GJ j = js.j[ji];
  gdc_launch();
  if (bid >= j.strips * j.ks) { gdc_wait(); return; }
  const int strip = bid / j.ks, split = bid % j.ks;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int ub0 = j.U * split / j.ks, ub1 = j.U * (split + 1) / j.ks;
  const int u0 = ub0 + (ub1 - ub0) * w / L1_NW, u1 = ub0 + (ub1 - ub0) * (w + 1) / L1_NW;
  const uint8_t* wst = j.W + (long)strip * j.U * 2048;
  uint4 wv[4 * G_UNR];   // the first batch of weights before waiting for the previous kernel
#pragma unroll
  for (int q = 0; q < 4 * G_UNR; q++) if (u0 + (q >> 2) < u1) wv[q] = ldg_nc(wst + ((long)(u0 + (q >> 2)) * 4 + (q & 3)) * 512 + lane * 16);
  gdc_wait();
  if (j.ng && d.st[6]) return;   // the n-gram proposal is used: nothing to compute
  if (j.rec >= 1) {   // Markov step i >= 1: stop once the cumulative top-1 probability of steps < i fell below theta
    __shared__ int skip;
    if (threadIdx.x < 32) {
      const int i = j.rec;
      int sk = d.st[5] <= i - 1;   // stopped earlier (st[5] is written by block 0 only; every block decides alike)
      if (!sk) {
        const float p = step_p(d, i - 1);
        const float cm = i - 1 == 0 ? p : d.cum[i - 2] * p;
        sk = cm < (float)d.theta;
        if (blockIdx.x == 0 && threadIdx.x == 0) { if (sk) d.st[5] = i; else d.cum[i - 1] = cm; }
      }
      if (threadIdx.x == 0) skip = sk;
    }
    __syncthreads();
    if (skip) return;
  }
  if (j.rec >= 0) {
    const int i = j.rec;
    const int prev = i == 0 ? d.st[3] : kidx(d.amax[i - 1]);
    j.X = d.state + (i & 1) * RK;
    j.X2 = d.w1 + (long)prev * RK;
  }
  float bpre[2] = {0.f, 0.f};
  if (j.bp && w == 0) {
#pragma unroll
    for (int hh = 0; hh < 2; hh++) {
      const int n = strip * 16 + g + 8 * hh;
      float b = 0.f;
      for (int k = 0; k < j.nbp; k++) b += j.bp[(long)k * NB * 768 + n];
      bpre[hh] = __fadd_rn(b, j.bias[n]);
    }
  }
  const bool colok = g < j.ncol;
  const bf16* xr = j.X + (long)g * j.xs;
  const bf16* xr2 = j.X2 ? j.X2 + (long)g * j.xs - j.ksplit_at : nullptr;
  auto bget = [&](int k0, uint32_t& b0, uint32_t& b1) {
    if (!colok) { b0 = b1 = 0u; return; }
    const bf16* p = (xr2 && k0 >= j.ksplit_at) ? xr2 + k0 + 2 * t : xr + k0 + 2 * t;
    b0 = __ldg(reinterpret_cast<const unsigned*>(p));
    b1 = __ldg(reinterpret_cast<const unsigned*>(p + 8));
  };
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  // batches of G_UNR units (the first one loaded above); same unit order as one unit at a time
  for (int u = u0; u < u1; u += G_UNR) {
    if (u > u0) {
#pragma unroll
      for (int q = 0; q < 4 * G_UNR; q++) if (u + (q >> 2) < u1) wv[q] = ldg_nc(wst + ((long)(u + (q >> 2)) * 4 + (q & 3)) * 512 + lane * 16);
    }
#pragma unroll
    for (int q = 0; q < 4 * G_UNR; q++) {
      if (u + (q >> 2) >= u1) break;
      uint32_t b0, b1;
      bget((u + (q >> 2)) * 64 + (q & 3) * 16, b0, b1);
      mma_bf16(acc, wv[q].x, wv[q].y, wv[q].z, wv[q].w, b0, b1);
    }
  }
  red[w][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __syncthreads();
  if (w != 0) return;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < L1_NW; k++) { const float4 r = red[k][lane]; v[0] += r.x; v[1] += r.y; v[2] += r.z; v[3] += r.w; }
  gemv_epi(j, strip, split, v, bpre);
}

// ---------------------------------------------------------------------------------------------- q / k / v post
// one warp per 128-vector: context rows (all layers): k = rope(rms(k) * k_norm_ctx), v; block rows of `layer`:
// q = rope(rms(q) * q_norm), k = rope(rms(k) * k_norm), v -> the context buffers' rows T..T+7 and q
// (one warp) y = bf16-rounded rope(rms(src) * nw) at position pos (or without rope) for elements lane * 4 .. + 3
__device__ __forceinline__ void rms_rope4(const Dk& d, const bf16* src, const float* nw, int pos, bool rope, float* y) {
  const int lane = threadIdx.x & 31;
  float x[4], ss = 0.f;
#pragma unroll
  for (int e = 0; e < 4; e++) { x[e] = bf(src[lane * 4 + e]); ss += x[e] * x[e]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float rs = rsqrtf(ss / 128.f + EPS);
#pragma unroll
  for (int e = 0; e < 4; e++) x[e] = rbf(x[e] * rs * nw[lane * 4 + e]);
  if (!rope) {
#pragma unroll
    for (int e = 0; e < 4; e++) y[e] = x[e];
    return;
  }
  const bool lo = lane < 16;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const float other = __shfl_xor_sync(0xffffffffu, x[e], 16);
    const int fi = (lane & 15) * 4 + e;
    const float fr = __fmul_rn((float)pos, d.inv_freq[fi]);
    const float cs = cosf(fr), sn = sinf(fr);
    y[e] = rbf(lo ? __fsub_rn(__fmul_rn(x[e], cs), __fmul_rn(other, sn)) : __fadd_rn(__fmul_rn(x[e], cs), __fmul_rn(other, sn)));
  }
}
__device__ __forceinline__ void rms_rope_store(const Dk& d, const bf16* src, const float* nw, int pos, bf16* dst, bool rope) {
  const int lane = threadIdx.x & 31;
  float x[4], ss = 0.f;
#pragma unroll
  for (int e = 0; e < 4; e++) { x[e] = bf(src[lane * 4 + e]); ss += x[e] * x[e]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float rs = rsqrtf(ss / 128.f + EPS);
#pragma unroll
  for (int e = 0; e < 4; e++) x[e] = rbf(x[e] * rs * nw[lane * 4 + e]);
  if (!rope) {
#pragma unroll
    for (int e = 0; e < 4; e++) dst[lane * 4 + e] = tobf(x[e]);
    return;
  }
  const bool lo = lane < 16;
#pragma unroll
  for (int e = 0; e < 4; e++) {
    const float other = __shfl_xor_sync(0xffffffffu, x[e], 16);
    const int fi = (lane & 15) * 4 + e;
    const float fr = __fmul_rn((float)pos, d.inv_freq[fi]);
    const float cs = cosf(fr), sn = sinf(fr);
    float y;
    if (lo) y = __fsub_rn(__fmul_rn(x[e], cs), __fmul_rn(other, sn));
    else y = __fadd_rn(__fmul_rn(x[e], cs), __fmul_rn(other, sn));
    dst[lane * 4 + e] = tobf(y);
  }
}
__global__ void __launch_bounds__(256) k_post(const Dk d, int layer, int do_ctx, int do_blk) {
  gdc_launch(); gdc_wait();
  const int vec = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const long T = d.T;
  int v = vec;
  const int nctx = do_ctx ? 3 * NB * NKV * 2 : 0;
  if (v < nctx) {
    const int isv = v & 1, rest = v >> 1, g = rest % NKV, r = (rest / NKV) % NB, l = rest / (NKV * NB);
    if (r >= d.nrows) return;
    const int p = d.st[8 + r];
    if (p < 0 || p >= T) return;
    const bf16* src = d.kvraw + ((long)l * NB + r) * 2048 + isv * 1024 + g * HD;
    if (isv) {
      bf16* dst = d.L[l].vc + ((long)p * NKV + g) * HD;
#pragma unroll
      for (int e = 0; e < 4; e++) dst[lane * 4 + e] = src[lane * 4 + e];
    } else {
      rms_rope_store(d, src, d.L[l].k_norm_ctx, p, d.L[l].kc + ((long)p * NKV + g) * HD, true);
    }
    return;
  }
  v -= nctx;
  if (d.st[6]) return;
  if (!do_blk || v >= NB * (NH + 2 * NKV)) return;
  const int r = v / (NH + 2 * NKV), j = v % (NH + 2 * NKV);
  const int pos = d.st[2] + r;
  const bf16* row = d.qkv + (long)r * QKVN;
  const DkL& L = d.L[layer];
  if (j < NH) rms_rope_store(d, row + j * HD, L.q_norm, pos, d.q + (long)r * NH * HD + j * HD, true);
  else if (j < NH + NKV) { const int g = j - NH; rms_rope_store(d, row + NH * HD + g * HD, L.k_norm, pos, L.kc + ((T + r) * NKV + g) * HD, true); }
  else {
    const int g = j - NH - NKV;
    const bf16* src = row + (NH + NKV) * HD + g * HD;
    bf16* dst = L.vc + ((T + r) * NKV + g) * HD;
#pragma unroll
    for (int e = 0; e < 4; e++) dst[lane * 4 + e] = src[lane * 4 + e];
  }
}

// ---------------------------------------------------------------------------------------------- attention
// block (kv head g, split s): the 24 queries of the group (8 rows x 3 heads) over keys [lo, c_new) of the context and
// the 8 block rows; online softmax over 64-key tiles; partial (m, l, o[128]) per query
#define AT 64
struct AttS { float K[AT][HD + 1]; float V[AT][HD]; float Q[24][HD]; float P[24][AT]; float BK[NB][HD]; float BV[NB][HD];
              float al[24]; float m[24]; float l[24]; int last; };
__global__ void __launch_bounds__(256) k_attn(const Dk d, int layer, int local) {
  extern __shared__ __align__(16) unsigned char smraw[];
  AttS& S = *reinterpret_cast<AttS*>(smraw);
  gdc_launch();
  if (PF_A && threadIdx.x == 0) {   // HBM is idle during attention / merge / norms: stage o_proj and gate_up in L2
    pf_slice(d.L[layer].o, O_BYTES, blockIdx.x, gridDim.x);
    if (PF_A >= 2) pf_slice(d.L[layer].gu, GU_BYTES, blockIdx.x, gridDim.x);
  }
  gdc_wait();
  if (d.st[6]) return;   // the n-gram proposal is used
  const int g = blockIdx.x / NSPLIT, sp = blockIdx.x % NSPLIT, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int cn = d.st[2];
  const int lo = local ? max(0, cn - 1024) : 0;
  const int n = cn - lo, nk = n + NB;
  const int k0 = (int)((long)nk * sp / NSPLIT), k1 = (int)((long)nk * (sp + 1) / NSPLIT);
  const long T = d.T;
  const DkL& L = d.L[layer];
#if !FUSE_POST
  for (int i = tid; i < 24 * HD / 8; i += 256) {
    const int qq = i / (HD / 8), dd = (i % (HD / 8)) * 8, r = qq / 3, hh = 3 * g + qq % 3;
    const uint4 v = *reinterpret_cast<const uint4*>(d.q + (long)r * NH * HD + hh * HD + dd);
    const bf16* e = reinterpret_cast<const bf16*>(&v);
#pragma unroll
    for (int k = 0; k < 8; k++) S.Q[qq][dd + k] = bf(e[k]);
  }
  if (k1 > n) {
    const int r = w;
#pragma unroll
    for (int e = 0; e < 4; e++) {
      S.BK[r][lane * 4 + e] = bf(L.kc[((T + r) * NKV + g) * HD + lane * 4 + e]);
      S.BV[r][lane * 4 + e] = bf(L.vc[((T + r) * NKV + g) * HD + lane * 4 + e]);
    }
  }
#else
  // the group's 24 queries (8 rows x 3 heads): rope(rms(q) * q_norm) at c_new + r, from the raw qkv rows
  for (int qq = w; qq < 24; qq += 8) {
    const int r = qq / 3, hh = 3 * g + qq % 3;
    float y[4];
    rms_rope4(d, d.qkv + (long)r * QKVN + hh * HD, L.q_norm, cn + r, true, y);
#pragma unroll
    for (int e = 0; e < 4; e++) S.Q[qq][lane * 4 + e] = y[e];
  }
  if (k1 > n) {   // this split covers block rows: their k (rope(rms(k) * k_norm)) and v of head g
    const int r = w;
    float y[4];
    rms_rope4(d, d.qkv + (long)r * QKVN + NH * HD + g * HD, L.k_norm, cn + r, true, y);
#pragma unroll
    for (int e = 0; e < 4; e++) {
      S.BK[r][lane * 4 + e] = y[e];
      S.BV[r][lane * 4 + e] = bf(d.qkv[(long)r * QKVN + (NH + NKV) * HD + g * HD + lane * 4 + e]);
    }
  }
#endif
  if (tid < 24) { S.m[tid] = -INFINITY; S.l[tid] = 0.f; }
  float o[12];
#pragma unroll
  for (int i = 0; i < 12; i++) o[i] = 0.f;
  const int dcol = tid & 127, qh = tid >> 7;
  for (int t0 = k0; t0 < k1; t0 += AT) {
    const int nt = min(AT, k1 - t0);
    __syncthreads();
#pragma unroll
    for (int it = 0; it < AT * HD / 8 / 256; it++) {
      const int i = tid + it * 256;
      const int jj = i / (HD / 8), dd = (i % (HD / 8)) * 8;
      const int vj = t0 + jj;
      if (jj < nt && vj >= n) {
#pragma unroll
        for (int k = 0; k < 8; k++) { S.K[jj][dd + k] = S.BK[vj - n][dd + k]; S.V[jj][dd + k] = S.BV[vj - n][dd + k]; }
        continue;
      }
      uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = make_uint4(0u, 0u, 0u, 0u);
      if (jj < nt) {
        const long row = lo + vj;
        kv = *reinterpret_cast<const uint4*>(L.kc + (row * NKV + g) * HD + dd);
        vv = *reinterpret_cast<const uint4*>(L.vc + (row * NKV + g) * HD + dd);
      }
      const bf16* ke = reinterpret_cast<const bf16*>(&kv);
      const bf16* ve = reinterpret_cast<const bf16*>(&vv);
#pragma unroll
      for (int k = 0; k < 8; k++) { S.K[jj][dd + k] = bf(ke[k]); S.V[jj][dd + k] = bf(ve[k]); }
    }
    __syncthreads();
    {
      const int jj = tid & 63, qb = tid >> 6;
      float sc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (jj < nt) {
#pragma unroll 4
        for (int dd = 0; dd < HD; dd++) {
          const float kv = S.K[jj][dd];
#pragma unroll
          for (int i = 0; i < 6; i++) sc[i] = fmaf(S.Q[qb + 4 * i][dd], kv, sc[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < 6; i++) S.P[qb + 4 * i][jj] = jj < nt ? sc[i] * 0.08838834764831845f : -INFINITY;
    }
    __syncthreads();
    for (int qq = w; qq < 24; qq += 8) {
      const float s0 = S.P[qq][lane], s1 = S.P[qq][lane + 32];
      float tm = fmaxf(s0, s1);
#pragma unroll
      for (int of = 16; of; of >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, of));
      const float mo = S.m[qq], mn = fmaxf(mo, tm);
      const float p0 = expf(s0 - mn), p1 = expf(s1 - mn);
      float ps = p0 + p1;
#pragma unroll
      for (int of = 16; of; of >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, of);
      __syncwarp();
      S.P[qq][lane] = p0; S.P[qq][lane + 32] = p1;
      if (lane == 0) {
        const float a = expf(mo - mn);   // mo = -inf at the first tile: 0
        S.al[qq] = a; S.l[qq] = S.l[qq] * a + ps; S.m[qq] = mn;
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 12; i++) o[i] *= S.al[qh + 2 * i];
    for (int jj = 0; jj < nt; jj++) {
      const float vv = S.V[jj][dcol];
#pragma unroll
      for (int i = 0; i < 12; i++) o[i] = fmaf(S.P[qh + 2 * i][jj], vv, o[i]);
    }
  }
  __syncthreads();
  float* pp = d.part + ((long)(g * NSPLIT + sp) * 24) * 132;
#pragma unroll
  for (int i = 0; i < 12; i++) { const int qq = qh + 2 * i; pp[qq * 132 + 4 + dcol] = o[i]; }
  if (tid < 24) { pp[tid * 132] = S.m[tid]; pp[tid * 132 + 1] = S.l[tid]; }
#if FUSE_MERGE
  // the last split of head g to finish merges the group's 24 queries (fixed split order: deterministic)
  __threadfence();
  __syncthreads();
  if (tid == 0) S.last = atomicAdd(d.acnt + g, 1u) == NSPLIT - 1;
  __syncthreads();
  if (!S.last) return;
  __threadfence();
  for (int qq = w; qq < 24; qq += 8) {
    const int r = qq / 3, hh = 3 * g + qq % 3;
    const float* p0 = d.part + ((long)(g * NSPLIT) * 24 + qq) * 132;
    const long sst = 24L * 132;
    float ms = -INFINITY, ls = 0.f;
    if (lane < NSPLIT) { ms = __ldcg(p0 + lane * sst); ls = __ldcg(p0 + lane * sst + 1); }
    float M = ms;
#pragma unroll
    for (int of = 16; of; of >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, of));
    const float e = ls > 0.f ? expf(ms - M) : 0.f;
    float Ssum = ls * e;
#pragma unroll
    for (int of = 16; of; of >>= 1) Ssum += __shfl_xor_sync(0xffffffffu, Ssum, of);
    float O[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
    for (int s2 = 0; s2 < NSPLIT; s2++) {
      const float es = __shfl_sync(0xffffffffu, e, s2);
      if (es != 0.f) {
        const float4 ov = __ldcg(reinterpret_cast<const float4*>(p0 + s2 * sst + 4 + lane * 4));
        O[0] += ov.x * es; O[1] += ov.y * es; O[2] += ov.z * es; O[3] += ov.w * es;
      }
    }
#pragma unroll
    for (int k = 0; k < 4; k++) d.att[(long)r * NH * HD + hh * HD + lane * 4 + k] = tobf(O[k] / Ssum);
  }
  if (tid == 0) d.acnt[g] = 0u;
#endif
}
// one warp per (row, head): merge the splits -> att (bf16)
__global__ void __launch_bounds__(256) k_merge(const Dk d) {
  gdc_launch(); gdc_wait();
  if (d.st[6]) return;   // the n-gram proposal is used
  const int wv = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (wv >= NB * NH) return;
  const int r = wv / NH, hh = wv % NH, g = hh / 3, qq = r * 3 + hh % 3;
  const float* p0 = d.part + ((long)(g * NSPLIT) * 24 + qq) * 132;
  const long sst = 24L * 132;
  float ms = -INFINITY, ls = 0.f;
  if (lane < NSPLIT) { ms = p0[lane * sst]; ls = p0[lane * sst + 1]; }
  float M = ms;
#pragma unroll
  for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  const float e = ls > 0.f ? expf(ms - M) : 0.f;
  float Ssum = ls * e;
#pragma unroll
  for (int o = 16; o; o >>= 1) Ssum += __shfl_xor_sync(0xffffffffu, Ssum, o);
  float4 ov[NSPLIT];
#pragma unroll
  for (int sp = 0; sp < NSPLIT; sp++) ov[sp] = *reinterpret_cast<const float4*>(p0 + sp * sst + 4 + lane * 4);
  float O[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int sp = 0; sp < NSPLIT; sp++) {
    const float es = __shfl_sync(0xffffffffu, e, sp);
    if (es != 0.f) { O[0] += ov[sp].x * es; O[1] += ov[sp].y * es; O[2] += ov[sp].z * es; O[3] += ov[sp].w * es; }
  }
#pragma unroll
  for (int k = 0; k < 4; k++) d.att[(long)r * NH * HD + hh * HD + lane * 4 + k] = tobf(O[k] / Ssum);
}

// ---------------------------------------------------------------------------------------------- add + rmsnorm rows
// x = bf16(sum of ks partials); s = x + res; h = bf16(rms(s) * w); res = bf16(s). final: rows 1..7 -> xm (with f)
#define AR_NT 768
__global__ void __launch_bounds__(AR_NT) k_addrms(const Dk d, const float* part, int ks, const float* w, int final_) {
  gdc_launch(); gdc_wait();
  if (d.st[6]) return;   // the n-gram proposal is used
  __shared__ float red[32];
  const int r = blockIdx.x, tid = threadIdx.x;
  float s[3], ss = 0.f;
  float pv[3][4];
#pragma unroll
  for (int e = 0; e < 3; e++)
#pragma unroll
    for (int q = 0; q < 4; q++) pv[e][q] = q < ks ? part[((long)q * NB + r) * H + tid + e * AR_NT] : 0.f;
#pragma unroll
  for (int e = 0; e < 3; e++) {
    const int k = tid + e * AR_NT;
    float x = 0.f;
#pragma unroll
    for (int q = 0; q < 4; q++) if (q < ks) x += pv[e][q];
    x = rbf(x);
    s[e] = x + bf(d.xres[(long)r * H + k]);
    ss += s[e] * s[e];
  }
  const float rs = rsqrtf(block_sum<AR_NT>(ss, red) / (float)H + EPS);
#pragma unroll
  for (int e = 0; e < 3; e++) {
    const int k = tid + e * AR_NT;
    const bf16 hv = tobf(s[e] * rs * w[k]);
    if (!final_) { d.h[(long)r * H + k] = hv; d.xres[(long)r * H + k] = tobf(s[e]); }
    else if (r >= 1) { d.xm[(long)(r - 1) * 2 * H + k] = hv; d.xm[(long)(r - 1) * 2 * H + H + k] = d.f[k]; }
  }
}

// ---------------------------------------------------------------------------------------------- Markov step
// every block: gates from raw (context[i] + recurrent GEMV), the new state (block 0 stores it), latent; then one
// warp per 16 vocabulary rows: bias = bf16(fold_w latent + fold_b), z = bf16(base + bias), lg = bf16(30 tanh(z / 30))
template <bool F8>
__global__ void __launch_bounds__(256) k_fold(const Dk d, int i) {
  __shared__ __align__(16) bf16 lat[RK];
  __shared__ __align__(16) __half lath[RK];
  constexpr bool f8 = F8;
  __shared__ unsigned long long bk[8];
  __shared__ float bs[8];
  const int tid = threadIdx.x;
  gdc_launch();
  const int w = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int strip = blockIdx.x * 8 + w;
  gdc_wait();
  if (d.st[5] <= i) return;   // early exit: this step's draft is never verified (no weight loads before knowing it)
  uint4 wv[16];
  if (strip < VSTRIPS) {
    if (f8) {
      const uint8_t* wst = d.fold8 + (long)strip * 4 * 1024;
#pragma unroll
      for (int q = 0; q < 8; q++) wv[q] = ldg_nc(wst + (long)q * 512 + lane * 16);
    } else {
      const uint8_t* wst = d.fold + (long)strip * 4 * 2048;
#pragma unroll
      for (int q = 0; q < 16; q++) wv[q] = ldg_nc(wst + (long)q * 512 + lane * 16);
    }
  }
  {
    const float gt0 = rbf(d.raw[tid]), cd0 = rbf(d.raw[RK + tid]), ou = rbf(d.raw[2 * RK + tid]);
    const float gt = rbf(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gt0))));
    const float cd = rbf(tanhf(cd0));
    const float st = bf(d.state[(i & 1) * RK + tid]);
    const bf16 ns = tobf(__fadd_rn(rbf(__fmul_rn(gt, st)), rbf(__fmul_rn(rbf(__fsub_rn(1.0f, gt)), cd))));
    if (blockIdx.x == 0) d.state[((i + 1) & 1) * RK + tid] = ns;
    lat[tid] = tobf(tanhf(ou));
    if (F8) lath[tid] = __float2half_rn(bf(lat[tid]));
  }
  __syncthreads();
  unsigned long long best = 0ull;
  float esum = 0.f;
  if (strip < VSTRIPS) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (f8) {
#pragma unroll
      for (int q = 0; q < 16; q++) {   // tile q: word pair (q & 1) of uint4 q / 2
        uint32_t b0 = 0u, b1 = 0u;
        if (g == 0) { b0 = *reinterpret_cast<const uint32_t*>(&lath[q * 16 + 2 * t]); b1 = *reinterpret_cast<const uint32_t*>(&lath[q * 16 + 8 + 2 * t]); }
        const uint4 v = wv[q >> 1];
        if (q & 1) mma_e4m3(acc, v.z, v.w, b0, b1); else mma_e4m3(acc, v.x, v.y, b0, b1);
      }
      if (t == 0) {
#pragma unroll
        for (int hh = 0; hh < 2; hh++) { const int n = strip * 16 + g + 8 * hh; acc[2 * hh] *= d.fold_s[n]; }
      }
    } else {
#pragma unroll
    for (int q = 0; q < 16; q++) {
      uint32_t b0 = 0u, b1 = 0u;
      if (g == 0) { b0 = *reinterpret_cast<const uint32_t*>(&lat[q * 16 + 2 * t]); b1 = *reinterpret_cast<const uint32_t*>(&lat[q * 16 + 8 + 2 * t]); }
      mma_bf16(acc, wv[q].x, wv[q].y, wv[q].z, wv[q].w, b0, b1);
    }
    }
    if (t == 0) {
#pragma unroll
      for (int hh = 0; hh < 2; hh++) {
        const int n = strip * 16 + g + 8 * hh;
        if (n < VOC) {
          const float bias = rbf(__fadd_rn(acc[2 * hh], bf(d.fold_b[n])));
          const float z = rbf(__fadd_rn(bf(d.base[(long)i * VP + n]), bias));
          const float lg = rbf(__fmul_rn(30.0f, tanhf(__fdiv_rn(z, 30.0f))));
          const unsigned long long k = fkey(lg, n);
          if (k > best) best = k;
          esum += expf(lg - 30.0f);   // lg <= 30 (soft cap): no max pass needed
        }
      }
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { const unsigned long long ok = __shfl_xor_sync(0xffffffffu, best, o); if (ok > best) best = ok; }
#pragma unroll
  for (int o = 16; o; o >>= 1) esum += __shfl_xor_sync(0xffffffffu, esum, o);
  if (lane == 0) { bk[w] = best; bs[w] = esum; }
  __syncthreads();
  if (tid == 0) {
    unsigned long long b = bk[0];
    float es = bs[0];
    for (int k = 1; k < 8; k++) { if (bk[k] > b) b = bk[k]; es += bs[k]; }
    atomicMax(d.amax + i, b);
    d.lse[i * FOLD_BLOCKS + blockIdx.x] = es;
  }
}
// drafts, next verify position and row count: warp i < (executed steps) recomputes step i's probability
__global__ void __launch_bounds__(256) k_end(const Dk d) {
  gdc_launch(); gdc_wait();
  if (d.st[6]) return;   // the n-gram proposal is used
  __shared__ float pc[NB];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int ns = min(d.st[5], NB - 1);   // steps that ran
  if (w < ns) {
    const float p = step_p(d, w);
    if (lane == 0) { pc[w] = p; d.pconf[w] = p; d.vtok[1 + w] = kidx(d.amax[w]); }
  }
  __syncthreads();
  if (tid == 0) {
    float cum = 1.f;
    int kd = 1;
    for (int k = 0; k < ns; k++) {
      cum *= pc[k];
      if (cum >= (float)d.theta) kd = k + 1; else break;
    }
    const int last = kidx(d.amax[ns - 1]);
    for (int k = ns; k < NB - 1; k++) { d.vtok[1 + k] = last; d.pconf[k] = 0.f; }   // never verified
    d.vtok[0] = d.st[3]; d.vpos[0] = d.st[2]; d.vpos[1] = kd + 1;
  }
}
// n-gram drafts (engine/ngram.py, mk.cu mk_ngram_k: the same rule): record the verify pass's target tokens at
// positions c_old + 1 .. c_old + 8 (mode 0); the longest earlier occurrence (<= 8 tokens, ties: most recent) of the
// committed suffix ending at the anchor hist[c_new]; if >= ngmin tokens: its continuation (periodic past c_new) is
// the proposal, K = 8, and the DSpark block / Markov kernels of this cycle return at once (st[6] = 1, st[5] = 0)
__global__ void __launch_bounds__(1024) k_ngram(const Dk d) {
  __shared__ int suf[8];
  __shared__ unsigned long long red[32];
  gdc_launch(); gdc_wait();
  int* hist = d.hist;
  const int T = (int)d.hT, c = d.st[2], minn = (int)d.ngmin;
  if (d.mode == 0 && threadIdx.x < 8) {
    const int p = d.st[0] + 1 + threadIdx.x;
    if (p >= 0 && p < T) hist[p] = d.tgt[threadIdx.x];
  }
  __syncthreads();
  if (threadIdx.x < 8) suf[threadIdx.x] = (c - (int)threadIdx.x >= 0) ? hist[c - threadIdx.x] : -2;
  __syncthreads();
  unsigned long long best = 0ull;
  for (int i = threadIdx.x; i < c && i < T; i += blockDim.x) {
    int L = 0;
    while (L < 8 && i - L >= 0 && c - L >= 0 && hist[i - L] == suf[L]) L++;
    const unsigned long long key = ((unsigned long long)L << 32) | (unsigned)i;
    if (key > best) best = key;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { const unsigned long long x = __shfl_xor_sync(0xffffffffu, best, o); if (x > best) best = x; }
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = best;
  __syncthreads();
  if (threadIdx.x < 32) {
    best = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0ull;
#pragma unroll
    for (int o = 16; o; o >>= 1) { const unsigned long long x = __shfl_xor_sync(0xffffffffu, best, o); if (x > best) best = x; }
    const int L = (int)(best >> 32), i = (int)(best & 0xffffffffu);
    if (L >= minn) {
      if (threadIdx.x < 7) {
        const int per = max(c - i, 1);
        d.vtok[1 + threadIdx.x] = hist[min(i + 1 + (int)threadIdx.x % per, T - 1)];
      }
      if (threadIdx.x == 0) { d.vtok[0] = d.st[3]; d.vpos[0] = c; d.vpos[1] = NB; d.st[6] = 1; d.st[5] = 0; }
    }
  }
}
__global__ void k_zero_state(const Dk d) { gdc_launch(); gdc_wait(); d.state[threadIdx.x] = tobf(0.f); }

// ---------------------------------------------------------------------------------------------- host
static bool g_pdl = true;
// launches with programmatic stream serialization (PDL) except the first kernel of a cycle
template <typename... KA, typename... AA>
static void launch(void (*k)(KA...), int grid, int block, size_t smem, cudaStream_t s, bool pdl, AA... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(block); cfg.dynamicSmemBytes = smem; cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = (pdl && g_pdl) ? 1 : 0;
  cudaLaunchKernelEx(&cfg, k, args...);
}
static GJ job(const uint8_t* W, const bf16* X, int xs, int ncol, int strips, int U, int ks, int epi, void* out, int os, const float* bias = nullptr) {
  GJ j; j.W = W; j.X = X; j.X2 = nullptr; j.out = out; j.bias = bias; j.bp = nullptr; j.nbp = 0; j.wsc = nullptr; j.fp8 = 0; j.ng = 1; j.xs = xs; j.ksplit_at = 1 << 30; j.ncol = ncol;
  j.strips = strips; j.U = U; j.ks = ks; j.epi = epi; j.os = os; j.rec = -1;
  const int nb = strips * ks;
#ifndef SB_MID
#define SB_MID 4
#endif
#ifndef SB_LO
#define SB_LO 300
#endif
  j.sb = nb <= SB_LO ? 1 : (nb <= 2400 ? SB_MID : G_SBMAX);
  return j;
}
// a layer weight: the FP8 copy when f8 bit 2 is set
static GJ f8w(GJ j, const Dk& d, const uint8_t* w8, const float* sc) {
  if ((d.f8 & 4) && w8) { j.W = w8; j.wsc = sc; j.fp8 = 1; }
  return j;
}
static void gemv(const Dk& d, cudaStream_t s, GJ a, GJ* b = nullptr, GJ* c = nullptr) {
  GJs js; js.j[0] = a; js.n = 1;
  if (b) js.j[js.n++] = *b;
  if (c) js.j[js.n++] = *c;
  int nb = 0, kr = 0;
  for (int k = 0; k < js.n; k++) {
    nb += gj_blocks(js.j[k]);
    const int u = (js.j[k].U + js.j[k].ks - 1) / js.j[k].ks;
    kr = u * 64 > kr ? u * 64 : kr;
  }
  bool any8 = false;
  for (int k = 0; k < js.n; k++) any8 |= js.j[k].fp8 != 0;
  if (a.sb > 1 || any8) { if (any8) launch(k_gemv<true>, nb, G_NW * 32, NB * (kr + 8) * 2, s, true, d, js);
                  else launch(k_gemv<false>, nb, G_NW * 32, NB * (kr + 8) * 2, s, true, d, js); }
  else {
    for (int k = 0; k < js.n; k++) js.j[k].sb = 1;
    nb = 0;
    for (int k = 0; k < js.n; k++) nb += gj_blocks(js.j[k]);
    launch(k_gemv_l1, nb, L1_NW * 32, 0, s, true, d, js);
  }
}

extern "C" {
int dk_struct_bytes() { return (int)sizeof(Dk); }
int dk_setup() {
  { const char* e = getenv("MK_DK_PDL"); g_pdl = !(e && e[0] == '0'); }
  cudaError_t e = cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(AttS));
  if (e == cudaSuccess) e = cudaFuncSetAttribute(k_gemv<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(k_gemv<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 * 1024);
  if (e != cudaSuccess) { fprintf(stderr, "dk_setup: %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
// mode 0: a full cycle (acceptance, context rows of the 8 verify rows, propose); mode 1: context rows only (nrows
// rows at positions pos64[r]: the prompt); mode 2: propose only (c_new, anchor, frow set by the host)
int dk_run(const Dk* dp, void* stream) {
  const Dk d = *dp;
  cudaStream_t s = (cudaStream_t)stream;
  const int mode = (int)d.mode, nr = mode == 0 ? NB : (int)d.nrows;
  Dk dd = d; dd.nrows = nr;
  launch(k_start, 33, K1_NT, 0, s, false, dd);
  if (mode != 1 && d.ngmin <= 8) launch(k_ngram, 1, 1024, 0, s, true, dd);
  // context K/V projections (mode 0, 1) and layer 0's q/k/v (mode 0, 2)
  {
    GJ jc[3], jq = f8w(job(d.L[0].qkv, d.h, H, NB, QKVN / 16, 36, 1, E_BF16, d.qkv, QKVN), d, d.L[0].qkv8, d.L[0].qkv_s);
    for (int l = 0; l < 3; l++) {
      jc[l] = f8w(job(d.L[l].kvc, d.u + (long)l * NB * H, H, nr, 2048 / 16, 36, 1, E_BF16, d.kvraw + (long)l * NB * 2048, 2048), d, d.L[l].kvc8, d.L[l].kvc_s);
      jc[l].ng = 0;
    }
    if (mode != 2) gemv(dd, s, jc[0], &jc[1], &jc[2]);
    if (mode != 1) gemv(dd, s, jq);
  }
  if (!FUSE_POST) {
    const int npost_ctx = mode != 2 ? 3 * NB * NKV * 2 : 0, npost_blk = mode != 1 ? NB * (NH + 2 * NKV) : 0;
    launch(k_post, (npost_ctx + npost_blk + 7) / 8, 256, 0, s, true, dd, 0, (int)(mode != 2), (int)(mode != 1));
  } else if (mode != 2) launch(k_post, (3 * NB * NKV * 2 + 7) / 8, 256, 0, s, true, dd, 0, 1, 0);   // context rows; block rows: in k_attn
  if (mode == 1) return (int)cudaGetLastError();
  for (int l = 0; l < 3; l++) {
    const DkL& L = d.L[l];
    if (l > 0) {
      gemv(dd, s, f8w(job(L.qkv, d.h, H, NB, QKVN / 16, 36, 1, E_BF16, d.qkv, QKVN), d, L.qkv8, L.qkv_s));
      if (!FUSE_POST) launch(k_post, (NB * (NH + 2 * NKV) + 7) / 8, 256, 0, s, true, dd, l, 0, 1);
    }
    launch(k_attn, NKV * NSPLIT, 256, sizeof(AttS), s, true, dd, l, (int)(l != 1));
    if (!FUSE_MERGE) launch(k_merge, NB * NH / 8, 256, 0, s, true, dd);
    gemv(dd, s, f8w(job(L.o, d.att, NH * HD, NB, H / 16, 48, 4, E_PART, d.opart, H), d, L.o8, L.o_s));
    launch(k_addrms, NB, AR_NT, 0, s, true, dd, (const float*)d.opart, 4, L.post_ln, 0);
    gemv(dd, s, f8w(job(L.gu, d.h, H, NB, 2 * FF / 16, 36, 1, E_ACT, d.act, FF), d, L.gu8, L.gu_s));
    gemv(dd, s, f8w(job(L.dn, d.act, FF, NB, H / 16, 144, 4, E_PART, d.opart, H), d, L.dn8, L.dn_s));
    launch(k_addrms, NB, AR_NT, 0, s, true, dd, (const float*)d.opart, 4, l < 2 ? d.L[l + 1].in_ln : d.norm, (int)(l == 2));
  }
  // Markov head: context projection [h_i, f] (FP32 out + bias) and the base logits of the 7 rows
  {
    GJ jb = job(d.un, d.xm, 2 * H, NB - 1, VSTRIPS, 36, 1, E_BF16, d.base, VP);
    if (d.f8 & 1) { jb.W = d.un8; jb.wsc = d.un_s; jb.fp8 = 1; }
    GJ jc = job(d.wctx, d.xm, 2 * H, NB - 1, 768 / 16, 72, 2, E_PART, d.ctxo, 768);   // partials [2][8][768]
    jc.sb = 1;
    gemv(dd, s, jb, &jc);
  }
  for (int i = 0; i < NB - 1; i++) {
    GJ jr = job(d.wrec, d.state, RK, 1, 768 / 16, 8, 1, E_F32, d.raw, 768, d.rnn_b);
    jr.bp = d.ctxo + (long)i * 768; jr.nbp = 2;
    jr.ksplit_at = RK; jr.rec = i;
    gemv(dd, s, jr);
    if (d.f8 & 2) launch(k_fold<true>, (VSTRIPS + 7) / 8, 256, 0, s, true, dd, i);
    else launch(k_fold<false>, (VSTRIPS + 7) / 8, 256, 0, s, true, dd, i);
  }
  launch(k_end, 1, 256, 0, s, true, dd);
  return (int)cudaGetLastError();
}
}
