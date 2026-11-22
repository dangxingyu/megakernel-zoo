// Inkling-2-Mini decode engine: one persistent cooperative kernel per step on one B200.
// See notes/plan.md. Phases hand over through tagged data (value + tag in one 8-byte store, the consumer polls
// its own elements); no grid barrier inside a step. All reductions in a fixed order.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "mkprof.cuh"
#ifndef W13_UNR16
#define W13_UNR16 12   // s18-c4: w13 NVFP4 units per warp batch (12 with W2V16 0: V(16) -2.2%; 9 worse)
#endif
#ifndef W2S_RS
#define W2S_RS (IE + 8)
#endif

typedef __nv_bfloat16 bf16;

#define H 2304
#define HP (H + 8)
#define NQH 24
#define NKV 8
#define HD 128
#define QW 3072
#define QWP (QW + 8)
#define NE 128
#define TOPK 8
#define IE 1152
#define NLW 24
#define NAPP 48
#define MAXB 8
#define MAXR 16   // rows of the global per-row buffers (verify passes up to 16 rows; step kernels use <= 8)
#ifndef NTH
#define NTH 768
#endif
#define NWARP (NTH / 32)
#ifndef GRP_EXP
#define GRP_EXP true
#endif
#define SMB MAXB
#define CH 128           // attention positions per item (at most)
#define MAXCH_S MAXCH
#define MAXCH 64         // chunks per (token, kv head)
#define MAXU 160         // union entries per layer (16-row passes: an expert with > 8 of the rows' tokens takes two entries)
#define VOCAB_REAL 200058
#define VOCAB_ROWS 201024
#define UN_STRIPS 12504  // 200064 rows
// fused qkv matrix rows (output layout fixed; weight strips: 410 + 2 * nsk)
#define R_G 3072
#define R_R 6144
#define R_S 6528
#define QG_STRIPS 410
#define K0 6560
#define V0 7616
#define QKVR 8672
// pieces (units of 64 k per item)
#define QKV_PU 6
#define QKV_NP 6
#define O_PU 6
#define O_NP 8
#define RT_PU 2
#define RT_NP 18
#define D_PU 6
#define D_NP 12

// ------------------------------------------------------------------------------------------------ parameters
struct W12M { const uint8_t* data; const int* base; const int* hdr; const int* esc; };   // engine/w12.py
struct LayerW {
  const uint8_t* wqkv; const uint8_t* wo; const uint8_t* wr; const float* rbias;
  const uint8_t* w13; const uint8_t* w2; const float* f13; const float* f2;
  const bf16* norm[3]; const bf16* mlp_norm;
  const bf16* ksc; const bf16* vsc; const bf16* attn_sc; const bf16* mlp_sc;
  const bf16* qscale; const bf16* rscale; const bf16* relp;
  const float* aq; const float* mq;
  const uint8_t* cD; const bf16* cFG;
  W12M qkv12, wo12;
  int nsk, mx, glob, naq;
};
struct App { int layer, pass, slot, bnd, nblk, blk[4], pin, nmsg, msg[2], loopl, out, pad; };
struct Ctx {
  LayerW L[NLW];
  App A[NAPP];
  const bf16* embed; const bf16* embed_norm; const float* oq; const bf16* fnorm; const uint8_t* unembed;
  const float* loopq;
  W12M un12;
  int nfin, fin[4], fin_pin, maxctx;
  uint2* pbt; bf16* kc; bf16* vc;          // pbt [49][MAXB][H/2]: (bf16x2, tag) partial after each application
  bf16* hk; bf16* hv; bf16* ha; bf16* hm;
  uint2* qkvt; uint2* arec; uint2* aoutt; uint2* attst; uint2* rlogt; uint2* act16t;
  float* part; float* w2p; unsigned* cnt; unsigned* bar;
  uint2* xgt; uint2* mgt; uint2* ubt; uint2* gvt; uint2* xpt;
  uint2* xst; uint2* xrt; uint2* sht; uint2* w2ct; uint2* ntt;   // ntt: next token per sequence (device loop)     // B >= 2: normalized rows published by designated blocks
  unsigned long long* amax; int* next_tok;
  long long* prof;
  bf16* fout;                              // [MAXB][H] final-norm output per row (verify passes: draft features)
  uint2* xft;                              // [MAXB][NPAIR] verify passes: final-norm rows published by designated blocks
};
struct Hdr { int B, pos, flags, nsteps; const int* tokens; float* logits; int* out_tok; int out_stride; unsigned tag0;
             const int* prompt; int plen; unsigned tok_tag;
             const int* vpos; int ro;
             const int* mt; };   // mt != null (verify passes): rows of several sequences, mt[0] = rows, mt[1 + b] = sequence (cache slot), mt[17 + b] = position, mt[33 + b] = earlier rows of its sequence in this pass (rows of a sequence consecutive)   // ro: verify passes of > 8 rows: first row of the half a phase works on   // vpos != null: verify pass, rows b < vpos[1] = one sequence (cache slot 0) at positions vpos[0] + b
#ifndef RING
#ifndef SHARED_ATTR
#define SHARED_ATTR
#endif
#ifndef RAWSM
#define RAWSM 1      // attention: the current position's raw k / v for the conv ring from smem (no re-poll)
#endif
#ifndef RAWSM_STEP
#define RAWSM_STEP 1   // ... also in the step kernels (track A: B=1 -25 us, B=8 -0.2%; bit-identical to be checked)
#endif
#ifndef KV4
#define KV4 0        // (measured slower in the verify kernel: +150 us at K=2) k / v convolutions: the 4 channels of a lane with one round trip of loads (kv_conv4)
#endif
#ifndef KV4_STEP
#define KV4_STEP 0   // ... also in the step kernel (a track-A change)
#endif
#define RING 32   // short-convolution history: raw inputs of the last positions, slot = position % RING
#endif
// row b of a step: position, cache slot (sequence) and how many earlier rows of the same pass precede it in its sequence
// (V: the verify kernel; the step kernel is compiled with V = false, so its code is the plain one-position code)
#define MT_SEQ 1
#define MT_POS 17
#define MT_NIN 33
#ifndef MSEQ
#define MSEQ 0   // multi-sequence passes (Hdr.mt row tables): built and bit-identical (tests/c3/vmulti.py), unused; the
                 // table checks in the row accessors cost ~70 us per 16-row pass, so they are compiled only with MSEQ 1
#endif
#if MSEQ
template <bool V> __device__ __forceinline__ int row_t(const Hdr& hd, int b) { return V ? (hd.mt ? __ldg(hd.mt + MT_POS + b) : hd.pos + b) : hd.pos; }
template <bool V> __device__ __forceinline__ int row_seq(const Hdr& hd, int b) { return V ? (hd.mt ? __ldg(hd.mt + MT_SEQ + b) : 0) : b; }
template <bool V> __device__ __forceinline__ int row_nin(const Hdr& hd, int b) { return V ? (hd.mt ? __ldg(hd.mt + MT_NIN + b) : b) : 0; }
#define HD_MT(hd) ((hd).mt)
#else
template <bool V> __device__ __forceinline__ int row_t(const Hdr& hd, int b) { return hd.pos + (V ? b : 0); }
template <bool V> __device__ __forceinline__ int row_seq(const Hdr& hd, int b) { return V ? 0 : b; }
template <bool V> __device__ __forceinline__ int row_nin(const Hdr& hd, int b) { return V ? b : 0; }
#define HD_MT(hd) ((const int*)nullptr)
#endif
template <bool V> __device__ __forceinline__ int row_tmax(const Hdr& hd) {   // the largest position of the pass's rows
  if (!(V && HD_MT(hd))) return row_t<V>(hd, hd.B - 1);
  int m = 0;
  for (int b = 0; b < hd.B; b++) m = max(m, row_t<V>(hd, b));
  return m;
}   // nsteps > 0: device loop over nsteps steps
#define F_LOGITS 1

// counter regions
#define CN_QKV 0
#define CN_ATT 1024
#define CN_O 1280
#define CN_RT 1536
#define CN_D 1600
#define CN_W13 2048          // MAXU * 144
#define CN_W2 (2048 + 9216)  // MAXB * 144
#define CN_TOTAL (CN_W2 + 1152 + 64)

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ uint4 ldg_nc(const void* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ uint2 ldg_nc2(const void* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ldg_nc1(const void* p) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ uint32_t ldcg_u32(const void* p) {
  uint32_t r;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}
__device__ __forceinline__ float ldcg_f(const float* p) { return __ldcg(p); }
__device__ __forceinline__ float4 ldcg_f4(const float4* p) { return __ldcg(p); }
__device__ __forceinline__ uint32_t cvt_fp4x2(uint32_t b) {
  uint32_t r; unsigned short bs = (unsigned short)b;
  asm("{ .reg .b8 t; cvt.u8.u16 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(r) : "h"(bs));
  return r;
}
__device__ __forceinline__ uint32_t cvt_e4m3x2(uint32_t w16) {
  uint32_t r; unsigned short h = (unsigned short)w16;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(h));
  return r;
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.f16x2 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ void mma_f16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 tobf(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rbf(float x) { return bf(tobf(x)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void fence_acqrel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void red_release_gpu(unsigned* p, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int lane_id() { int l; asm("mov.u32 %0, %%laneid;" : "=r"(l)); return l; }

// ------------------------------------------------------------------------------------------------ shared memory
// the router input of 8 rows (bf16) lives on x16 rows 12..15 (written last: a 16-row pass converts row 8 + b after the
// router, element by element in place, in row order, every element by the same thread: tests/c3/vcheck16.py)
#define EXR(smm, b) (reinterpret_cast<bf16*>(&(smm).u.e.x16[12][0][0]) + (long)(b) * HP)
// ROUT1: the router input of all 16 rows on x16 rows 8..15 (converted in place after the router, rows in order)
#define EXR16(smm, b) (reinterpret_cast<bf16*>(&(smm).u.e.x16[8][0][0]) + (long)(b) * HP)
struct Smem {
  union {
    bf16 xs[3][SMB][HP];                 // B operands (bf16), streams q / k / v (router, unembed: stream 0)
    bf16 xo[SMB][QWP];                   // o_proj input
    bf16 xs16[2][MAXR][HP];               // 16-row qkv in one pass: stream 0 and the block's other stream (QKV1)
    struct { __half x16[MAXR][2][HP]; } e;   // expert inputs of all 16 rows (f16 hi / lo); the router input of 8 rows: EXR
   // router input + expert input (f16 hi / lo)
    struct {                               // attention item
      bf16 kk[CH][HD + 8]; bf16 vv[CH][HD + 8]; bf16 qb[8][HD + 8]; bf16 ph[16][CH + 8]; bf16 pl[16][CH + 8];
      float sc[3][CH]; float pr[16][CH]; float rr[3][16];
      float kt[MAXB][HD]; float vt[MAXB][HD]; float ml[2][3][2]; float po[3][HD]; float raw[2][HD];
    } at;
    struct {                               // verify passes: attention record item shared by all rows (64 keys)
      bf16 kk[64][HD + 8]; bf16 vv[64][HD + 8]; bf16 qb[3 * MAXR][HD + 8]; bf16 ph[3 * MAXR][64 + 8]; bf16 pl[3 * MAXR][64 + 8];
      float sc[3 * MAXR][64]; float prw[16][64 + MAXR]; float rr[3 * MAXR][16];
      bf16 kt[MAXR][HD]; bf16 vt[MAXR][HD]; float ml[3 * MAXR][2]; bf16 raw[2][MAXR][HD];   // kt / vt / raw: bf16-exact values
    } as;
    float mrg[MAXCH_S * 3 * 130 + 3 * MAXCH];      // attention merge staging
    __half w2s[24][2][IE];                       // w2 inputs (expert, token) pairs of a round, hi / lo
  } u;
  float4 red4[NWARP][32];                  // block GEMV: partial fragments per warp
  float w2o[MAXB][TOPK][16];               // w2: per (token, expert rank) output strip
  float red[NWARP][32];
  float bcast[64];
  float sc16[MAXR];                        // 2^-s of the expert input f16 scaling per token
  // routing tables (identical in every block)
  int nU; int uexp[MAXU]; int uncnt[MAXU]; int utok[MAXU][MAXB];
  int ue0[NE]; int uc0[NE];               // 16-row routing: experts and their token counts before the split
  int tex[MAXR][TOPK]; float twt[MAXR][TOPK];   // per token: experts sorted by id, weights
  unsigned umask[4];
  int flag;
  unsigned subn;
};

// ------------------------------------------------------------------------------------------------ barrier
__device__ __forceinline__ void grid_bar(const Ctx& c, unsigned& nbar, unsigned base) {
  __syncthreads();
  nbar += 1;
  if (threadIdx.x == 0) {
    unsigned target = base + nbar * gridDim.x;
    __threadfence();
    red_release_gpu(&c.bar[0], 1);
    while ((int)(ld_relaxed_gpu(&c.bar[0]) - target) < 0) { }
    fence_acqrel_gpu();
  }
  __syncthreads();
}

// block-wide sum of N values per thread (all threads get the result)
template <int N> struct P2 { static constexpr int v = N <= 1 ? 1 : N <= 2 ? 2 : N <= 4 ? 4 : N <= 8 ? 8 : N <= 16 ? 16 : 32; };
// warp reduce-scatter: after it, lane l holds the warp sum of value index ridx(l) (N padded to a power of two <= 32)
template <int N>
__device__ __forceinline__ float warp_rs(const float* v, int& ridx) {
  constexpr int M = P2<N>::v;
  float x[M];
#pragma unroll
  for (int i = 0; i < M; i++) x[i] = i < N ? v[i] : 0.f;
  const int l = threadIdx.x & 31;
  int idx = 0, off = 16;
#pragma unroll
  for (int n = M; n > 1; n >>= 1) {
    const int half = n >> 1;
    const bool up = (l & off) != 0;
#pragma unroll
    for (int i = 0; i < half; i++) {
      const float send = up ? x[i] : x[i + half];
      const float keep = up ? x[i + half] : x[i];
      x[i] = keep + __shfl_xor_sync(0xffffffff, send, off);
    }
    if (up) idx += half;
    off >>= 1;
  }
  for (; off; off >>= 1) x[0] += __shfl_xor_sync(0xffffffff, x[0], off);
  ridx = idx;
  return x[0];
}
template <int N>
__device__ __forceinline__ void block_sum(float* v, Smem& sm) {
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  int ri;
  const float r = warp_rs<N>(v, ri);
  constexpr int M = P2<N>::v;
  if (ri < N && (l & (32 / M - 1)) == 0) sm.red[w][ri] = r;    // lanes l = idx * (32 / M) hold distinct indices
  __syncthreads();
  if (threadIdx.x < N) {
    float s = 0.f;
#pragma unroll 4
    for (int k = 0; k < NWARP; k++) s += sm.red[k][threadIdx.x];
    sm.bcast[threadIdx.x] = s;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; i++) v[i] = sm.bcast[i];
  // no third barrier: the next writers of bcast (threads < N of the next call) are behind that call's first barrier,
  // which every thread reaches only after reading bcast here
}
// sums of N values and maxima of M values in one reduction
template <int N, int M>
__device__ __forceinline__ void block_sum_max(float* v, float* mx, Smem& sm) {
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < N; i++) { float x = v[i];
#pragma unroll
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
    v[i] = x; }
#pragma unroll
  for (int i = 0; i < M; i++) { float x = mx[i];
#pragma unroll
    for (int o = 16; o; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o));
    mx[i] = x; }
  if (l == 0) {
#pragma unroll
    for (int i = 0; i < N; i++) sm.red[w][i] = v[i];
#pragma unroll
    for (int i = 0; i < M; i++) sm.red[w][N + i] = mx[i];
  }
  __syncthreads();
  if (threadIdx.x < N + M) {
    float s = threadIdx.x < N ? 0.f : -1e30f;
    for (int k = 0; k < NWARP; k++) s = threadIdx.x < N ? s + sm.red[k][threadIdx.x] : fmaxf(s, sm.red[k][threadIdx.x]);
    sm.bcast[threadIdx.x] = s;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; i++) v[i] = sm.bcast[i];
#pragma unroll
  for (int i = 0; i < M; i++) mx[i] = sm.bcast[N + i];
}
__device__ __forceinline__ float block_max1(float x, Smem& sm) {
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
#pragma unroll
  for (int o = 16; o; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o));
  if (l == 0) sm.red[w][0] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = sm.red[0][0];
    for (int k = 1; k < NWARP; k++) s = fmaxf(s, sm.red[k][0]);
    sm.bcast[0] = s;
  }
  __syncthreads();
  return sm.bcast[0];
}


// Block GEMV: the block computes up to NWARP / WG strips at once; warp group q (WG warps) runs strip q, warp wl of the
// group covers units [U*wl/WG, U*(wl+1)/WG) (a split fixed per matrix: the same order for any batch); the fragments
// are summed in warp order through shared memory and epi(q, acc) runs on the group's first warp.
// GRP: the WG warps of a strip sync among themselves (named barrier 1 + group) instead of the whole block, so groups
// run their rounds independently (the round no longer waits for the slowest warp of the block); same arithmetic
__device__ __forceinline__ void grp_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
template <int WG, int NG, bool GRP = false, class RUN, class EPI>
__device__ __forceinline__ void block_strips(Smem& sm, int ns, int U, RUN run, EPI epi) {
  const int w = threadIdx.x >> 5, lane = lane_id();
  const int q = w / WG, wl = w - q * WG;
  float acc[NG][4];
#pragma unroll
  for (int ng = 0; ng < NG; ng++) acc[ng][0] = acc[ng][1] = acc[ng][2] = acc[ng][3] = 0.f;
  const bool act = q < ns && q < NWARP / WG;
  if (act) { const int u0 = U * wl / WG, u1 = U * (wl + 1) / WG; if (u1 > u0) run(q, u0, u1 - u0, acc); }
  float tot[NG][4];
#pragma unroll
  for (int ng = 0; ng < NG; ng++) {
    if (WG > 1) {
      sm.red4[w][lane] = make_float4(acc[ng][0], acc[ng][1], acc[ng][2], acc[ng][3]);
      if (GRP) grp_sync(1 + q, WG * 32); else __syncthreads();
      tot[ng][0] = tot[ng][1] = tot[ng][2] = tot[ng][3] = 0.f;
      if (act && wl == 0) {
#pragma unroll
        for (int k = 0; k < WG; k++) {
          const float4 v = sm.red4[w + k][lane];
          tot[ng][0] += v.x; tot[ng][1] += v.y; tot[ng][2] += v.z; tot[ng][3] += v.w;
        }
      }
      if (GRP) grp_sync(1 + q, WG * 32); else __syncthreads();
    } else {
      tot[ng][0] = acc[ng][0]; tot[ng][1] = acc[ng][1]; tot[ng][2] = acc[ng][2]; tot[ng][3] = acc[ng][3];
    }
  }
  if (act && wl == 0) epi(q, tot);
}

// L2 prefetch of a contiguous weight range (one thread; bulk prefetches of <= 64 KB)
__device__ __forceinline__ void pf_l2(const void* p, long bytes) {
#ifndef NO_PF
  const char* q = reinterpret_cast<const char*>(p);
  for (long o = 0; o < bytes; o += 65536) {
    const unsigned n = (unsigned)min(65536L, bytes - o);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(q + o), "r"(n) : "memory");
  }
#endif
}

// ------------------------------------------------------------------------------------------------ tagged data
// value and tag travel as ONE 64-bit scalar access (single-copy atomic; a v2.u32 access may tear)
__device__ __forceinline__ void st_tag(uint2* p, uint32_t v, unsigned tag) {
  const unsigned long long w = ((unsigned long long)tag << 32) | v;
  asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" :: "l"(p), "l"(w) : "memory");
}
__device__ __forceinline__ void st_tagf(uint2* p, float v, unsigned tag) { st_tag(p, __float_as_uint(v), tag); }
__device__ __forceinline__ uint2 ld_tag(const uint2* p) {
#ifdef FIRST_CG
  { unsigned long long w; asm volatile("ld.global.cg.b64 %0, [%1];" : "=l"(w) : "l"(p) : "memory"); return make_uint2((uint32_t)w, (uint32_t)(w >> 32)); }
#endif
  // relaxed (coherent) load: ld.global.cg returned current tags with stale values on this card (here and in the
  // Inkling-2-Small engine, e0064-e0067 there); relaxed loads cost ~0.45 us vs ~0.15 us but overlap within a thread
  unsigned long long w;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(w) : "l"(p) : "memory");
  return make_uint2((uint32_t)w, (uint32_t)(w >> 32));
}
// two consecutive tagged words (16-byte aligned), each 64-bit element single-copy atomic
__device__ __forceinline__ void ld_tag2(const uint2* p, uint2& x, uint2& y) {
  unsigned long long a, b;
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(p) : "memory");
  x = make_uint2((uint32_t)a, (uint32_t)(a >> 32)); y = make_uint2((uint32_t)b, (uint32_t)(b >> 32));
}
__device__ __forceinline__ uint2 ld_tag_r(const uint2* p) {
  unsigned long long w;
  asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(w) : "l"(p) : "memory");
  return make_uint2((uint32_t)w, (uint32_t)(w >> 32));
}
#ifndef POLL_NS
#define POLL_NS 0
#endif
__device__ __forceinline__ uint32_t wait_tag(const uint2* p, uint2 v, unsigned tag) {
  while (v.y != tag) { if (POLL_NS) __nanosleep(POLL_NS); v = ld_tag_r(p); }
  return v.x;
}
__device__ __forceinline__ uint32_t poll(const uint2* p, unsigned tag) { return wait_tag(p, ld_tag(p), tag); }
__device__ __forceinline__ float pollf(const uint2* p, unsigned tag) { return __uint_as_float(poll(p, tag)); }
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  return (uint32_t)__bfloat16_as_ushort(tobf(lo)) | ((uint32_t)__bfloat16_as_ushort(tobf(hi)) << 16);
}
__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }
// tags: applications tag0 + a + 1; pbt[idx] tag0 + idx; the final phase tag0 + 50
#define TAGA(a) (hd.tag0 + (unsigned)(a) + 1u)
#define TAGP(i) (hd.tag0 + (unsigned)(i))

// ------------------------------------------------------------------------------------------------ mixes
// Every thread holds element pairs p = threadIdx.x + j * NTH (j < 2, p < H / 2): slot k -> element EL(k).
#define EPT 4
#define NPAIR (H / 2)
static_assert(NPAIR > NTH && NPAIR <= 2 * NTH, "pair layout");
__device__ __forceinline__ int EL(int k) { return 2 * (threadIdx.x + (k >> 1) * NTH) + (k & 1); }
__device__ __forceinline__ bool EV(int k) { return k < 2 || threadIdx.x < NPAIR - NTH; }

// softmax over depth: sources v[s][k] (f32) for the bits s of msk, queries qs[q] (f32 [H]); out[q][k] (f32). normalize_values = true.
template <int NQ_MAX>
__device__ __forceinline__ void load_queries(const float* const* qs, int nq, float (*qv)[EPT]) {
#pragma unroll
  for (int q = 0; q < NQ_MAX; q++)
#pragma unroll
    for (int j = 0; j < 2; j++) {
      float2 f = make_float2(0.f, 0.f);
      if (q < nq) f = __ldg(reinterpret_cast<const float2*>(qs[q]) + (EV(2 * j) ? threadIdx.x + j * NTH : threadIdx.x));
      qv[q][2 * j] = f.x; qv[q][2 * j + 1] = f.y;
    }
}
template <int NS_MAX, int NQ_MAX>
__device__ __forceinline__ void depth_mix_q(float (*v)[EPT], unsigned msk, const float (*qv)[EPT], int nq, float (*out)[EPT], Smem& sm) {
  float red[NS_MAX + NQ_MAX * NS_MAX];
#pragma unroll
  for (int i = 0; i < NS_MAX + NQ_MAX * NS_MAX; i++) red[i] = 0.f;
#pragma unroll
  for (int s = 0; s < NS_MAX; s++) {
    if ((msk >> s) & 1u) {
#pragma unroll
      for (int k = 0; k < EPT; k++) red[s] += v[s][k] * v[s][k];
#pragma unroll
      for (int q = 0; q < NQ_MAX; q++)
#pragma unroll
        for (int k = 0; k < EPT; k++) red[NS_MAX + q * NS_MAX + s] += v[s][k] * qv[q][k];
    }
  }
  block_sum<NS_MAX + NQ_MAX * NS_MAX>(red, sm);
  float inv[NS_MAX];
#pragma unroll
  for (int s = 0; s < NS_MAX; s++) inv[s] = rsqrtf(red[s] / (float)H + 1e-6f);
  const float hs = rsqrtf((float)H);
#pragma unroll
  for (int q = 0; q < NQ_MAX; q++) {
    if (q >= nq) continue;
    float lg[NS_MAX], mx = -1e30f;
#pragma unroll
    for (int s = 0; s < NS_MAX; s++) {
      lg[s] = ((msk >> s) & 1u) ? red[NS_MAX + q * NS_MAX + s] * hs * inv[s] : -1e30f;
      mx = fmaxf(mx, lg[s]);
    }
    float den = 0.f;
#pragma unroll
    for (int s = 0; s < NS_MAX; s++) { lg[s] = ((msk >> s) & 1u) ? expf(lg[s] - mx) : 0.f; den += lg[s]; }
    float rd = 1.0f / den;
#pragma unroll
    for (int k = 0; k < EPT; k++) {
      float o = 0.f;
#pragma unroll
      for (int s = 0; s < NS_MAX; s++)
        if ((msk >> s) & 1u) o += (lg[s] * rd) * (v[s][k] * inv[s]);
      out[q][k] = o;
    }
  }
}
template <int NS_MAX, int NQ_MAX>
__device__ __forceinline__ void depth_mix(float (*v)[EPT], unsigned msk, const float* const* qs, int nq, float (*out)[EPT], Smem& sm) {
  float qv[NQ_MAX][EPT];
  load_queries<NQ_MAX>(qs, nq, qv);
  depth_mix_q<NS_MAX, NQ_MAX>(v, msk, qv, nq, out, sm);
}
// norm weights of one stream (bf16 pairs) as floats
__device__ __forceinline__ void load_w(const bf16* w, float* wv) {
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const __nv_bfloat162 x = *reinterpret_cast<const __nv_bfloat162*>(w + (EV(2 * j) ? EL(2 * j) : EL(0)));
    wv[2 * j] = bf(x.x); wv[2 * j + 1] = bf(x.y);
  }
}
// rmsnorm of nq streams at once (one reduction) with prefetched weights -> bf16 smem rows
template <int NQ>
__device__ __forceinline__ void rmsnorm_store_n(const float (*x)[EPT], const float (*wv)[EPT], int nq, bf16* const* dst, Smem& sm) {
  float s[NQ];
#pragma unroll
  for (int q = 0; q < NQ; q++) { s[q] = 0.f;
#pragma unroll
    for (int k = 0; k < EPT; k++) s[q] += x[q][k] * x[q][k]; }
  block_sum<NQ>(s, sm);
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    if (q >= nq) continue;
    const float r = rsqrtf(s[q] / (float)H + 1e-6f);
#pragma unroll
    for (int j = 0; j < 2; j++) {
      if (!EV(2 * j)) continue;
      *reinterpret_cast<uint32_t*>(dst[q] + EL(2 * j)) = pack_bf2(x[q][2 * j] * r * wv[q][2 * j], x[q][2 * j + 1] * r * wv[q][2 * j + 1]);
    }
  }
}
// rmsnorm with weight -> bf16 into dst (smem row, length H)
__device__ __forceinline__ void rmsnorm_store(const float* x, const bf16* w, bf16* dst, Smem& sm) {
  float s[1] = {0.f};
#pragma unroll
  for (int k = 0; k < EPT; k++) s[0] += x[k] * x[k];
  block_sum<1>(s, sm);
  float r = rsqrtf(s[0] / (float)H + 1e-6f);
#pragma unroll
  for (int j = 0; j < 2; j++) {
    if (!EV(2 * j)) continue;
    const int e = EL(2 * j);
    float y0 = x[2 * j] * r, y1 = x[2 * j + 1] * r;
    if (w) { const __nv_bfloat162 wv = *reinterpret_cast<const __nv_bfloat162*>(w + e); y0 = y0 * bf(wv.x); y1 = y1 * bf(wv.y); }
    *reinterpret_cast<uint32_t*>(dst + e) = pack_bf2(y0, y1);
  }
}

__device__ __forceinline__ float ldcg_bf(const bf16* p) { return bf(__ushort_as_bfloat16(__ldcg(reinterpret_cast<const unsigned short*>(p)))); }
// rows of pbt (polled): idx[s] for s < ns, token b
template <int NS>
__device__ __forceinline__ void load_srcs(const Ctx& c, const Hdr& hd, const int* idx, unsigned msk, int b, float (*v)[EPT]) {
  // loads issued unconditionally (clamped addresses): arrays assigned under a condition would live in local memory
  const int p1 = EV(2) ? threadIdx.x + NTH : threadIdx.x;
  uint2 r[NS][2];
#pragma unroll
  for (int s = 0; s < NS; s++) {
    const uint2* base = c.pbt + ((long)idx[s] * MAXR + b) * NPAIR;
    r[s][0] = ld_tag(base + threadIdx.x);
    r[s][1] = ld_tag(base + p1);
  }
#pragma unroll
  for (int s = 0; s < NS; s++) {
    const uint2* base = c.pbt + ((long)idx[s] * MAXR + b) * NPAIR;
    const unsigned tg = TAGP(idx[s]);
    const uint32_t x0 = ((msk >> s) & 1u) ? wait_tag(base + threadIdx.x, r[s][0], tg) : 0u;
    const uint32_t x1 = (((msk >> s) & 1u) && EV(2)) ? wait_tag(base + p1, r[s][1], tg) : 0u;
    v[s][0] = bf_lo(x0); v[s][1] = bf_hi(x0); v[s][2] = bf_lo(x1); v[s][3] = bf_hi(x1);
  }
}
// a tagged f32 vector [H] (pairs of elements as consecutive entries)
__device__ __forceinline__ void load_vec_f(const uint2* p, unsigned tag, float* v) {
  uint2 r[EPT];
#pragma unroll
  for (int k = 0; k < EPT; k++) r[k] = ld_tag(p + (EV(k) ? EL(k) : EL(k & 1)));
#pragma unroll
  for (int k = 0; k < EPT; k++) v[k] = EV(k) ? __uint_as_float(wait_tag(p + EL(k), r[k], tag)) : 0.f;
}
// a tagged bf16 vector [H] (pairs)
__device__ __forceinline__ void load_vec_bf(const uint2* p, unsigned tag, float* v) {
  const int p1 = EV(2) ? threadIdx.x + NTH : threadIdx.x;
  const uint2 r0 = ld_tag(p + threadIdx.x), r1 = ld_tag(p + p1);
  const uint32_t x0 = wait_tag(p + threadIdx.x, r0, tag);
  const uint32_t x1 = EV(2) ? wait_tag(p + p1, r1, tag) : 0u;
  v[0] = bf_lo(x0); v[1] = bf_hi(x0); v[2] = bf_lo(x1); v[3] = bf_hi(x1);
}

// ------------------------------------------------------------------------------------------------ GEMV pieces
// B operand loaders. bf16 smem rows: element (k, col n) at base[n * stride + k].
struct BSm {
  const bf16* base; int stride; int ncol;   // columns >= ncol read as zero
  __device__ __forceinline__ uint2 get(int k0) const {
    const int l = lane_id(), g = l >> 2, t = l & 3;
    if (g >= ncol) return make_uint2(0, 0);
    const bf16* p = base + g * stride + k0 + 2 * t;
    return make_uint2(*reinterpret_cast<const uint32_t*>(p), *reinterpret_cast<const uint32_t*>(p + 8));
  }
};
struct BTag {   // tagged bf16 pair rows in global memory (polled): element pair (k / 2) of column n at base[n * stride + k / 2]
  const uint2* base; int stride; int ncol; unsigned tag;
  __device__ __forceinline__ uint2 get(int k0) const {
    const int l = lane_id(), g = l >> 2, t = l & 3;
    if (g >= ncol) return make_uint2(0, 0);
    const uint2* p = base + (long)g * stride + (k0 >> 1) + t;
    return make_uint2(poll(p, tag), poll(p + 4, tag));
  }
};

#ifndef BF_UNR
#define BF_UNR 2
#endif
template <class BL>
__device__ __forceinline__ void piece_bf16(const uint8_t* wst, int u0, int nu, const BL& bl, float* acc) {
  const int lane = lane_id();
  int u = u0;
  const int ue = u0 + nu;
  for (; u + BF_UNR <= ue; u += BF_UNR) {
    uint4 w[4 * BF_UNR];
#pragma unroll
    for (int q = 0; q < 4 * BF_UNR; q++) w[q] = ldg_nc(wst + ((long)(u + (q >> 2)) * 4 + (q & 3)) * 512 + lane * 16);
#pragma unroll
    for (int q = 0; q < 4 * BF_UNR; q++) {
      uint2 b = bl.get((u + (q >> 2)) * 64 + (q & 3) * 16);
      mma_bf16(acc, w[q].x, w[q].y, w[q].z, w[q].w, b.x, b.y);
    }
  }
  for (; u < ue; u++) {
    uint4 w[4];
#pragma unroll
    for (int q = 0; q < 4; q++) w[q] = ldg_nc(wst + ((long)u * 4 + q) * 512 + lane * 16);
#pragma unroll
    for (int q = 0; q < 4; q++) {
      uint2 b = bl.get(u * 64 + q * 16);
      mma_bf16(acc, w[q].x, w[q].y, w[q].z, w[q].w, b.x, b.y);
    }
  }
}

// piece_bf16 for two column groups (rows 0..7 and 8..15 of a 16-row pass): each weight fragment feeds two MMAs; per
// column the same units in the same order as piece_bf16 (columns of an mma are independent): bit-identical
template <class BL>
__device__ __forceinline__ void piece_bf16_2(const uint8_t* wst, int u0, int nu, const BL& b0, const BL& b1, float (*acc)[4]) {
  const int lane = lane_id();
  int u = u0;
  const int ue = u0 + nu;
  for (; u + BF_UNR <= ue; u += BF_UNR) {
    uint4 w[4 * BF_UNR];
#pragma unroll
    for (int q = 0; q < 4 * BF_UNR; q++) w[q] = ldg_nc(wst + ((long)(u + (q >> 2)) * 4 + (q & 3)) * 512 + lane * 16);
#pragma unroll
    for (int q = 0; q < 4 * BF_UNR; q++) {
      const int k = (u + (q >> 2)) * 64 + (q & 3) * 16;
      const uint2 x0 = b0.get(k), x1 = b1.get(k);
      mma_bf16(acc[0], w[q].x, w[q].y, w[q].z, w[q].w, x0.x, x0.y);
      mma_bf16(acc[1], w[q].x, w[q].y, w[q].z, w[q].w, x1.x, x1.y);
    }
  }
  for (; u < ue; u++) {
    uint4 w[4];
#pragma unroll
    for (int q = 0; q < 4; q++) w[q] = ldg_nc(wst + ((long)u * 4 + q) * 512 + lane * 16);
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint2 x0 = b0.get(u * 64 + q * 16), x1 = b1.get(u * 64 + q * 16);
      mma_bf16(acc[0], w[q].x, w[q].y, w[q].z, w[q].w, x0.x, x0.y);
      mma_bf16(acc[1], w[q].x, w[q].y, w[q].z, w[q].w, x1.x, x1.y);
    }
  }
}

// ---- exact 12-bit BF16 code (track B, engine/w12.py): decode one unit (4 tiles x 4 registers per lane)
__device__ __forceinline__ uint32_t prmt0(uint32_t a, uint32_t sel) { uint32_t r; asm("prmt.b32 %0,%1,0,%2;" : "=r"(r) : "r"(a), "r"(sel)); return r; }
// registers a[4 j + r] of tile j: value bytes of tile j (8, value order 2r + half), nibble word nw[j]; E0 / E8 = the
// rows' window bases << 7 in both halves (rows g and g + 8)
__device__ __forceinline__ void w12_unit(const uint4& d0, const uint4& d1, const uint4& dn, uint32_t E0, uint32_t E8, uint32_t* a) {
  const uint32_t by[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
  const uint32_t nw[4] = {dn.x, dn.y, dn.z, dn.w};
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      const uint32_t B = by[2 * j + (r >> 1)];
      const uint32_t i = (r & 1) * 2;
      const uint32_t p = prmt0(B, i | ((8u | i) << 4) | ((i + 1) << 8) | ((8u | (i + 1)) << 12));   // byte, sign-replicated byte
      const uint32_t n = ((nw[j] >> (4 * r)) & 0x000F000Fu) << 7;
      a[4 * j + r] = (p & 0x807F807Fu) | (n + ((r & 1) ? E8 : E0));
    }
}
__device__ __forceinline__ void w12_patch(const W12M& M, int hd, uint32_t* a) {
  const int cnt = hd & 0xFF;
  const int lane = lane_id();
  const int* e = M.esc + (hd >> 8);
  for (int q = 0; q < cnt; q++) {
    const int en = __ldg(e + q);
    if (((en >> 27) & 31) == lane) {
      const int rg = (en >> 23) & 15, hf = (en >> 22) & 1;
      const uint32_t val = en & 0xFFFF;
#pragma unroll
      for (int k = 0; k < 16; k++)
        if (k == rg) a[k] = hf ? ((a[k] & 0x0000FFFFu) | (val << 16)) : ((a[k] & 0xFFFF0000u) | val);
    }
  }
}
struct W12Pre { uint4 d0, d1, dn; int hd; };
__device__ __forceinline__ W12Pre w12_load(const W12M& M, long strip, int U, int u) {
  const int lane = lane_id();
  const uint8_t* p = M.data + ((strip * U + u) * 3) * 512 + lane * 16;
  W12Pre w;
  w.d0 = ldg_nc(p); w.d1 = ldg_nc(p + 512); w.dn = ldg_nc(p + 1024);
  w.hd = __ldg(M.hdr + strip * U + u);
  return w;
}
__device__ __forceinline__ void w12_bases(const W12M& M, long strip, uint32_t& E0, uint32_t& E8) {
  const int g = lane_id() >> 2;
  const uint32_t e0 = (uint32_t)__ldg(M.base + strip * 16 + g), e8 = (uint32_t)__ldg(M.base + strip * 16 + g + 8);
  E0 = (e0 | (e0 << 16)) << 7; E8 = (e8 | (e8 << 16)) << 7;
}
template <class BL>
__device__ __forceinline__ void w12_mma(const W12M& M, const W12Pre& w, uint32_t E0, uint32_t E8, int u, const BL& bl, float* acc) {
  uint32_t a[16];
  w12_unit(w.d0, w.d1, w.dn, E0, E8, a);
  if (w.hd & 0xFF) w12_patch(M, w.hd, a);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const uint2 b = bl.get(u * 64 + j * 16);
    mma_bf16(acc, a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3], b.x, b.y);
  }
}
// units [u0, u0 + nu) of a W12 strip, two units in flight (MMA order identical to piece_bf16: bit-identical results)
template <class BL>
__device__ __forceinline__ void piece_w12(const W12M& M, long strip, int U, int u0, int nu, const BL& bl, float* acc) {
  uint32_t E0, E8;
  w12_bases(M, strip, E0, E8);
  int u = u0;
  const int ue = u0 + nu;
  for (; u + 2 <= ue; u += 2) {
    const W12Pre w0 = w12_load(M, strip, U, u), w1 = w12_load(M, strip, U, u + 1);
    w12_mma(M, w0, E0, E8, u, bl, acc);
    w12_mma(M, w1, E0, E8, u + 1, bl, acc);
  }
  for (; u < ue; u++) { const W12Pre w0 = w12_load(M, strip, U, u); w12_mma(M, w0, E0, E8, u, bl, acc); }
}

// f16 B operand for experts: columns n = 2 * li + part (part 0: hi, 1: lo), li = 4 * ng + t-th token of the expert
__device__ __forceinline__ void scl4(uint32_t w, uint32_t* o) {  // bytes (g,g8) j0, (g,g8) j1 -> {g,g},{g8,g8},{g,g},{g8,g8}
  uint32_t lo, hi;
  asm("{ .reg .b16 l,h; mov.b32 {l,h}, %2; cvt.rn.f16x2.e4m3x2 %0, l; cvt.rn.f16x2.e4m3x2 %1, h; }" : "=r"(lo), "=r"(hi) : "r"(w));
  asm("{ .reg .b16 a,b; mov.b32 {a,b}, %4; mov.b32 %0, {a,a}; mov.b32 %1, {b,b}; mov.b32 {a,b}, %5; mov.b32 %2, {a,a}; mov.b32 %3, {b,b}; }"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(lo), "r"(hi));
}
__device__ __forceinline__ void cvt4(uint32_t w, uint32_t* o) {
  asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;\n"
      " cvt.rn.f16x2.e2m1x2 %0, b0; cvt.rn.f16x2.e2m1x2 %1, b1; cvt.rn.f16x2.e2m1x2 %2, b2; cvt.rn.f16x2.e2m1x2 %3, b3; }"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(w));
}
// B fragments of one unit (64 k) for NG n-groups: bfr[ng][j] = (b0, b1)
template <int NG>
__device__ __forceinline__ void load_b16_unit(const __half* const* colp, int u, uint2 (*bfr)[4]) {
  const int t = lane_id() & 3;
#pragma unroll
  for (int ng = 0; ng < NG; ng++) {
    const __half* p = colp[ng];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (p) {
        const __half* q = p + u * 64 + j * 16 + 2 * t;
        bfr[ng][j] = make_uint2(*reinterpret_cast<const uint32_t*>(q), *reinterpret_cast<const uint32_t*>(q + 8));
      } else bfr[ng][j] = make_uint2(0, 0);
    }
  }
}
// tagged f16 pairs in global memory: colp points at uint2 rows (pair index k / 2), polled with tag; null columns
// read the first non-null column's address (loads unconditional) and give zeros
template <int NG>
__device__ __forceinline__ void load_b16_unit_gl(const __half* const* colp, int u, uint2 (*bfr)[4], unsigned tag) {
  const int t = lane_id() & 3;
  uint2 r[NG][4][2];
  const uint2* p0 = reinterpret_cast<const uint2*>(colp[2]);   // fallback (always valid)
#pragma unroll
  for (int ng = 0; ng < NG; ng++) {
    const uint2* p = colp[ng] ? reinterpret_cast<const uint2*>(colp[ng]) : p0;
#pragma unroll
    for (int j = 0; j < 4; j++) { r[ng][j][0] = ld_tag(p + u * 32 + j * 8 + t); r[ng][j][1] = ld_tag(p + u * 32 + j * 8 + 4 + t); }
  }
#pragma unroll
  for (int ng = 0; ng < NG; ng++) {
    const uint2* p = reinterpret_cast<const uint2*>(colp[ng]);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      if (p) bfr[ng][j] = make_uint2(wait_tag(p + u * 32 + j * 8 + t, r[ng][j][0], tag), wait_tag(p + u * 32 + j * 8 + 4 + t, r[ng][j][1], tag));
      else bfr[ng][j] = make_uint2(0, 0);
    }
  }
}

// NVFP4: per unit 576 B: codes [lane][16 B], scales [g][8 B]
template <int NG>
__device__ __forceinline__ void unit_fp4(const uint4& cw, const uint2& sw, const uint2 (*bfr)[4], float (*acc)[4]) {
  uint32_t cc[4] = {cw.x, cw.y, cw.z, cw.w};
  uint32_t ss[2] = {sw.x, sw.y};
#pragma unroll
  for (int jj = 0; jj < 2; jj++) {
    uint32_t sc[4];
    scl4(ss[jj], sc);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int j = jj * 2 + h;
      uint32_t a[4];
      cvt4(cc[j], a);
      const uint32_t sg = sc[2 * h], sg8 = sc[2 * h + 1];
      const uint32_t a0 = hmul2(a[0], sg), a1 = hmul2(a[1], sg8), a2 = hmul2(a[2], sg), a3 = hmul2(a[3], sg8);
#pragma unroll
      for (int ng = 0; ng < NG; ng++) mma_f16(acc[ng], a0, a1, a2, a3, bfr[ng][j].x, bfr[ng][j].y);
    }
  }
}
template <int NG, bool GL, int UNR_ = 0>
__device__ __forceinline__ void piece_fp4(const uint8_t* wst, int u0, int nu, const __half* const* colp, float (*acc)[4], unsigned tag = 0) {
  const int lane = lane_id(), g = lane >> 2;
#ifndef FP4_UNR
#define FP4_UNR 6
#endif
  constexpr int UNR = UNR_ ? UNR_ : FP4_UNR;   // s18-c4: per call (W13_UNR16 for w13)
  int u = u0;
  const int ue = u0 + nu;
  for (; u + UNR <= ue; u += UNR) {
    uint4 cw[UNR]; uint2 sw[UNR];
#pragma unroll
    for (int q = 0; q < UNR; q++) {
      cw[q] = ldg_nc(wst + (long)(u + q) * 576 + lane * 16);
      sw[q] = ldg_nc2(wst + (long)(u + q) * 576 + 512 + g * 8);
    }
#pragma unroll
    for (int q = 0; q < UNR; q++) {
      uint2 bfr[NG][4];
      if (GL) load_b16_unit_gl<NG>(colp, u + q, bfr, tag); else load_b16_unit<NG>(colp, u + q, bfr);
      unit_fp4<NG>(cw[q], sw[q], bfr, acc);
    }
  }
  for (; u < ue; u++) {
    const uint4 cw = ldg_nc(wst + (long)u * 576 + lane * 16);
    const uint2 sw = ldg_nc2(wst + (long)u * 576 + 512 + g * 8);
    uint2 bfr[NG][4];
    if (GL) load_b16_unit_gl<NG>(colp, u, bfr, tag); else load_b16_unit<NG>(colp, u, bfr);
    unit_fp4<NG>(cw, sw, bfr, acc);
  }
}
// rolling double buffer of RB-unit batches: batch k + 1 loads while batch k is decoded (verify passes); the units are
// accumulated in increasing order exactly as piece_fp4
template <int NG, int RB>
__device__ __forceinline__ void piece_fp4_roll(const uint8_t* wst, int u0, int nu, const __half* const* colp, float (*acc)[4]) {
  const int lane = lane_id(), g = lane >> 2;
  uint4 cA[RB], cB[RB]; uint2 sA[RB], sB[RB];
  const int nb = nu / RB;
  auto ld = [&](int b, uint4* cw, uint2* sw) {
#pragma unroll
    for (int q = 0; q < RB; q++) {
      const int u = u0 + b * RB + q;
      cw[q] = ldg_nc(wst + (long)u * 576 + lane * 16);
      sw[q] = ldg_nc2(wst + (long)u * 576 + 512 + g * 8);
    }
  };
  auto run = [&](int b, const uint4* cw, const uint2* sw) {
#pragma unroll
    for (int q = 0; q < RB; q++) {
      uint2 bfr[NG][4];
      load_b16_unit<NG>(colp, u0 + b * RB + q, bfr);
      unit_fp4<NG>(cw[q], sw[q], bfr, acc);
    }
  };
  ld(0, cA, sA);
#pragma unroll 1
  for (int b = 0; b < nb; b += 2) {
    if (b + 1 < nb) ld(b + 1, cB, sB);
    run(b, cA, sA);
    if (b + 2 < nb) ld(b + 2, cA, sA);
    if (b + 1 < nb) run(b + 1, cB, sB);
  }
}
// MXFP8: per unit 1056 B: codes [lane][32 B] ((j, kh) u32 = rows (g, g8) pairs), scales [g][blk][r] 4 B
template <int NG>
__device__ __forceinline__ void unit_fp8(const uint4& c0, const uint4& c1, uint32_t sw, const uint2 (*bfr)[4], float (*acc)[4]) {
  uint32_t cc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
  for (int blk = 0; blk < 2; blk++) {
    float d[NG][4];
#pragma unroll
    for (int ng = 0; ng < NG; ng++) { d[ng][0] = d[ng][1] = d[ng][2] = d[ng][3] = 0.f; }
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int j = blk * 2 + h;
      const uint32_t x0 = cc[2 * j], x1 = cc[2 * j + 1];
      const uint32_t a0 = cvt_e4m3x2(x0 & 0xffff), a1 = cvt_e4m3x2(x0 >> 16), a2 = cvt_e4m3x2(x1 & 0xffff), a3 = cvt_e4m3x2(x1 >> 16);
#pragma unroll
      for (int ng = 0; ng < NG; ng++) mma_f16(d[ng], a0, a1, a2, a3, bfr[ng][j].x, bfr[ng][j].y);
    }
    const uint32_t e0 = (sw >> (16 * blk)) & 0xff, e1 = (sw >> (16 * blk + 8)) & 0xff;
    const float s0 = e0 ? __int_as_float(e0 << 23) : __int_as_float(0x00400000), s1 = e1 ? __int_as_float(e1 << 23) : __int_as_float(0x00400000);
#pragma unroll
    for (int ng = 0; ng < NG; ng++) {
      acc[ng][0] = fmaf(d[ng][0], s0, acc[ng][0]); acc[ng][1] = fmaf(d[ng][1], s0, acc[ng][1]);
      acc[ng][2] = fmaf(d[ng][2], s1, acc[ng][2]); acc[ng][3] = fmaf(d[ng][3], s1, acc[ng][3]);
    }
  }
}
template <int NG, bool GL>
__device__ __forceinline__ void piece_fp8(const uint8_t* wst, int u0, int nu, const __half* const* colp, float (*acc)[4], unsigned tag = 0) {
  const int lane = lane_id(), g = lane >> 2;
#ifndef FP8_UNR
#define FP8_UNR 3
#endif
  constexpr int UNR = FP8_UNR;
  int u = u0;
  const int ue = u0 + nu;
  for (; u + UNR <= ue; u += UNR) {
    uint4 c0[UNR], c1[UNR]; uint32_t sw[UNR];
#pragma unroll
    for (int q = 0; q < UNR; q++) {
      c0[q] = ldg_nc(wst + (long)(u + q) * 1056 + lane * 32);
      c1[q] = ldg_nc(wst + (long)(u + q) * 1056 + lane * 32 + 16);
      sw[q] = ldg_nc1(wst + (long)(u + q) * 1056 + 1024 + g * 4);
    }
#pragma unroll
    for (int q = 0; q < UNR; q++) {
      uint2 bfr[NG][4];
      if (GL) load_b16_unit_gl<NG>(colp, u + q, bfr, tag); else load_b16_unit<NG>(colp, u + q, bfr);
      unit_fp8<NG>(c0[q], c1[q], sw[q], bfr, acc);
    }
  }
  for (; u < ue; u++) {
    const uint4 c0 = ldg_nc(wst + (long)u * 1056 + lane * 32);
    const uint4 c1 = ldg_nc(wst + (long)u * 1056 + lane * 32 + 16);
    const uint32_t sw = ldg_nc1(wst + (long)u * 1056 + 1024 + g * 4);
    uint2 bfr[NG][4];
    if (GL) load_b16_unit_gl<NG>(colp, u, bfr, tag); else load_b16_unit<NG>(colp, u, bfr);
    unit_fp8<NG>(c0, c1, sw, bfr, acc);
  }
}

// three chains over units [6k, 6k + 6) (k = 0, 1, 2) of an 18-unit row strip, interleaved so that six units are in
// flight; each chain accumulates its units in increasing order exactly as piece_fp4 / piece_fp8 over its six units
template <int NG, bool MX>
__device__ __forceinline__ void piece3(const uint8_t* wst, const __half* const* colp, float (*acc)[NG][4]) {
  const int lane = lane_id(), g = lane >> 2;
  constexpr int UB = MX ? 1056 : 576;
#pragma unroll 1
  for (int j = 0; j < 6; j += 2) {
    uint4 c0[6], c1[6]; uint2 s4[6]; uint32_t s8[6];
#pragma unroll
    for (int q = 0; q < 6; q++) {
      const int u = 6 * (q >> 1) + j + (q & 1);
      if (MX) {
        c0[q] = ldg_nc(wst + (long)u * UB + lane * 32);
        c1[q] = ldg_nc(wst + (long)u * UB + lane * 32 + 16);
        s8[q] = ldg_nc1(wst + (long)u * UB + 1024 + g * 4);
      } else {
        c0[q] = ldg_nc(wst + (long)u * UB + lane * 16);
        s4[q] = ldg_nc2(wst + (long)u * UB + 512 + g * 8);
      }
    }
#pragma unroll
    for (int q = 0; q < 6; q++) {
      const int u = 6 * (q >> 1) + j + (q & 1);
      uint2 bfr[NG][4];
      load_b16_unit<NG>(colp, u, bfr);
      if (MX) unit_fp8<NG>(c0[q], c1[q], s8[q], bfr, acc[q >> 1]);
      else unit_fp4<NG>(c0[q], s4[q], bfr, acc[q >> 1]);
    }
  }
}

// verify passes: one warp runs a whole row strip as three chains over units [UPC k, UPC k + UPC) (k = 0, 1, 2), the
// split block_strips<3> gives its three warps; each chain accumulates its units in increasing order exactly as
// piece_fp4 / piece_fp8 over them; BPC units of every chain in flight per batch
template <int NG, bool MX, int UPC, int BPC>
__device__ __forceinline__ void piece3c(const uint8_t* wst, const __half* const* colp, float (*acc)[NG][4]) {
  const int lane = lane_id(), g = lane >> 2;
  constexpr int UB = MX ? 1056 : 576;
  constexpr int NQ = 3 * BPC;
#pragma unroll 1
  for (int j = 0; j < UPC; j += BPC) {
    uint4 c0[NQ], c1[MX ? NQ : 1]; uint2 s4[MX ? 1 : NQ]; uint32_t s8[MX ? NQ : 1];
#pragma unroll
    for (int q = 0; q < NQ; q++) {
      const int u = UPC * (q / BPC) + j + (q % BPC);
      if (MX) {
        c0[q] = ldg_nc(wst + (long)u * UB + lane * 32);
        c1[MX ? q : 0] = ldg_nc(wst + (long)u * UB + lane * 32 + 16);
        s8[MX ? q : 0] = ldg_nc1(wst + (long)u * UB + 1024 + g * 4);
      } else {
        c0[q] = ldg_nc(wst + (long)u * UB + lane * 16);
        s4[MX ? 0 : q] = ldg_nc2(wst + (long)u * UB + 512 + g * 8);
      }
    }
#pragma unroll
    for (int q = 0; q < NQ; q++) {
      const int u = UPC * (q / BPC) + j + (q % BPC);
      uint2 bfr[NG][4];
      load_b16_unit<NG>(colp, u, bfr);
      if (MX) unit_fp8<NG>(c0[q], c1[MX ? q : 0], s8[MX ? q : 0], bfr, acc[q / BPC]);
      else unit_fp4<NG>(c0[q], s4[MX ? 0 : q], bfr, acc[q / BPC]);
    }
  }
}
// sum of the three chains in block_strips' order (0 + chain 0 + chain 1 + chain 2)
template <int NG>
__device__ __forceinline__ void sum3(const float (*acc)[NG][4], float (*tot)[4]) {
#pragma unroll
  for (int ng = 0; ng < NG; ng++)
#pragma unroll
    for (int i = 0; i < 4; i++) { float x = 0.f; x += acc[0][ng][i]; x += acc[1][ng][i]; x += acc[2][ng][i]; tot[ng][i] = x; }
}

// write a warp's piece, return true (in all lanes) if this warp is the last of `npieces` for counter `cnt`;
// then a[] holds the sum of all pieces in piece order.
__device__ __forceinline__ bool piece_reduce(float* part, unsigned* cnt, int piece, int npieces, float* acc) {
  const int lane = lane_id();
  float4* slot = reinterpret_cast<float4*>(part);
  slot[piece * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  __threadfence();
  __syncwarp();
  unsigned old = 0;
  if (lane == 0) old = atomicAdd(cnt, 1u);
  old = __shfl_sync(0xffffffff, old, 0);
  if (old != (unsigned)(npieces - 1)) return false;
  __threadfence();
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  for (int p = 0; p < npieces; p++) {
    float4 v = ldcg_f4(slot + p * 32 + lane);
    a0 += v.x; a1 += v.y; a2 += v.z; a3 += v.w;
  }
  acc[0] = a0; acc[1] = a1; acc[2] = a2; acc[3] = a3;
  if (lane == 0) { atomicExch(cnt, 0u); __threadfence(); }
  __syncwarp();
  return true;
}

// ------------------------------------------------------------------------------------------------ phases
#define PH_ARGS const Ctx& c, const Hdr& hd, Smem& sm
#ifdef MKPROF
#define SUBMARK(tag) do { if (c.prof && threadIdx.x == 0) { unsigned n_ = sm.subn++; if (n_ < 1024) { unsigned long long* r_ = (unsigned long long*)c.prof + 2ull * ((unsigned long long)blockIdx.x * 4096 + 3072 + n_); r_[0] = mkprof_gtime(); r_[1] = ((unsigned long long)clock64() << 16) | (tag); } } } while (0)
#else
#define SUBMARK(tag) do {} while (0)
#endif

// static short convolution (kernel 4, tap 3 = current), identity residual, x 5^-1/2; ring of the last 3 raw inputs
__device__ __forceinline__ float sconv_static(float x, float x1, float x2, float x3, const bf16* w4) {
  float acc = __fmul_rn(x3, bf(w4[0]));
  acc = __fadd_rn(acc, __fmul_rn(x2, bf(w4[1])));
  acc = __fadd_rn(acc, __fmul_rn(x1, bf(w4[2])));
  acc = __fadd_rn(acc, __fmul_rn(x, bf(w4[3])));
  return __fmul_rn(__fadd_rn(acc, x), 0.44721359549995793f);
}

// ---- B >= 2: designated blocks publish normalized rows (tagged bf16 pairs); every block polls them
// x (f32, nq streams) -> rmsnorm with weights -> published pairs dst[q] (row of NPAIR words)
template <int NQ>
__device__ __forceinline__ void norm_publish(const float (*x)[EPT], const float (*wv)[EPT], int nq, uint2* const* dst, unsigned tag, Smem& sm) {
  float s[NQ];
#pragma unroll
  for (int q = 0; q < NQ; q++) { s[q] = 0.f;
#pragma unroll
    for (int k = 0; k < EPT; k++) s[q] += x[q][k] * x[q][k]; }
  block_sum<NQ>(s, sm);
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    if (q >= nq) continue;
    const float r = rsqrtf(s[q] / (float)H + 1e-6f);
#pragma unroll
    for (int j = 0; j < 2; j++)
      if (EV(2 * j)) st_tag(dst[q] + threadIdx.x + j * NTH, pack_bf2(x[q][2 * j] * r * wv[q][2 * j], x[q][2 * j + 1] * r * wv[q][2 * j + 1]), tag);
  }
}
// poll nrows published rows (row r at src + r * NPAIR) into smem rows dst(r) (bf16, pairs)
template <class DST>
#ifndef PR16_UNR
#define PR16_UNR 8   // tagged row polls in flight per thread
#endif
__device__ __forceinline__ void poll_rows(const uint2* src, int nrows, DST dst, unsigned tag) {
  const int n = nrows * NPAIR;   // words, two per load
  for (int i0 = 2 * threadIdx.x; i0 < n; i0 += 2 * NTH * PR16_UNR) {
    uint2 r[PR16_UNR][2];
#pragma unroll
    for (int q = 0; q < PR16_UNR; q++) ld_tag2(src + min(i0 + q * 2 * NTH, n - 2), r[q][0], r[q][1]);
#pragma unroll
    for (int q = 0; q < PR16_UNR; q++) {
      const int i = i0 + q * 2 * NTH;
      if (i >= n) continue;
      const int row = i / NPAIR, w = i % NPAIR;
      uint32_t* d = reinterpret_cast<uint32_t*>(dst(row)) + w;
      d[0] = wait_tag(src + i, r[q][0], tag);
      d[1] = wait_tag(src + i + 1, r[q][1], tag);
    }
  }
}

#ifndef VDES_MIN
#define VDES_MIN 3   // verify passes: designated-block mixes (publish + poll) from this many rows, else every block mixes all rows
#endif
// the qkv input streams this block's qkv strips read (ph_qkv's strip -> stream map): bit s set for stream s
__device__ __forceinline__ unsigned qkv_streams(const LayerW& L, int nstreams) {
  const int strips = QG_STRIPS + 2 * L.nsk;
  unsigned m = 0u;
  for (int strip = blockIdx.x; strip < strips; strip += gridDim.x) {
    int st = strip < QG_STRIPS ? 0 : (strip < QG_STRIPS + L.nsk ? 1 : 2);
    if (st >= nstreams) st = 0;
    m |= 1u << st;
  }
  return m;
}
#ifndef POLL_STEP
#define POLL_STEP 0   // ... also in the step kernels (track A)
#endif
#ifndef POLL_NEEDED
#define POLL_NEEDED 1   // verify passes: poll only the published input streams this block's qkv strips use
#endif
#ifndef HYB2
#define HYB2 1   // verify passes of 2 .. HYB_MAX rows: row 0 mixed in every block, rows >= 1 by the last blocks (published)
#endif
#ifndef HYB_MAX
#define HYB_MAX 2
#endif
template <bool V> __device__ __forceinline__ bool hyb2(const Hdr& hd) { return V && HYB2 && hd.B >= 2 && hd.B <= HYB_MAX; }
#ifndef DREV
#define DREV 1   // verify passes: designated rows on the last blocks (no w2b / o_proj strips) instead of the first
#endif
#ifndef DREV_MAX
#define DREV_MAX 4   // ... for passes of at most this many rows (K=8: +220 us on the last blocks)
#endif
template <bool V> __device__ __forceinline__ int dsel(const Hdr& hd) { return (V && DREV && hd.B <= DREV_MAX) ? (int)(gridDim.x - 1 - blockIdx.x) : (int)blockIdx.x; }
// verify passes of more than 8 rows: the smem-row phases (qkv, router .. w2a, final unembedding) run per 8-row half
template <bool V> __device__ __forceinline__ bool big16(const Hdr& hd) { return V && hd.B > SMB; }
template <bool V> __device__ __forceinline__ bool desig(const Hdr& hd) { return hd.B >= (V ? VDES_MIN : 2) && !hyb2<V>(hd); }
// The attention input streams of application a -> normalized bf16 smem streams (returns their number), or for the
// gated applications (loop passes 2 and 3) G1: blocks (token, controller) write x, m and [unit(x), unit(m)] (returns -1)
template <bool V> __device__ int ph_attn_in(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = hd.B;
  const bool dz = desig<V>(hd);
#ifndef PF_MIN
#define PF_MIN 3   // L2 prefetch of the qkv / router strips from this many rows (the step kernel: from 2)
#endif
  const bool pfq = B >= (V ? PF_MIN : 2);
  if (pfq && threadIdx.x < 4) {   // this block's qkv strips into L2 while the mix waits for its sources (B >= 2: the mix is long enough)
    const int strip = blockIdx.x + threadIdx.x * gridDim.x;
    if (strip < QG_STRIPS + 2 * L.nsk) pf_l2(L.wqkv + (long)strip * 36 * 2048, 36 * 2048);
  } else if (pfq && threadIdx.x == 32 && blockIdx.x < 8) pf_l2(L.wr + (long)blockIdx.x * 36 * 2048, 36 * 2048);   // router strip
  const int nq = L.glob ? 3 : 1;
  const int par = a & 1;
  const bool hy = hyb2<V>(hd);
  if (ap.nmsg == 0 && (dz || hy)) {
    uint2* xs_pub = c.xst + (long)par * 3 * MAXR * NPAIR;   // [3][MAXR][NPAIR]
    // hybrid (2 rows): jobs of row 1 on the last blocks; row 0 below in every block
    for (int job = hy ? nq + (int)(gridDim.x - 1 - blockIdx.x) : dsel<V>(hd); job < B * nq; job += gridDim.x) {
      const int b = job / nq, q = job % nq;
      float x[1][EPT];
      if (a == 0) {
        const int tok = min(max(hd.tokens ? hd.tokens[b] : (int)poll(c.ntt + b, hd.tok_tag), 0), VOCAB_ROWS - 1);   // memory-safe; ids are checked on the host only where that costs no sync
        const bf16* er = c.embed + (long)tok * H;
        float e[EPT], s[1] = {0.f};
#pragma unroll
        for (int j = 0; j < 2; j++) {
          __nv_bfloat162 w2 = EV(2 * j) ? *reinterpret_cast<const __nv_bfloat162*>(er + EL(2 * j)) : __floats2bfloat162_rn(0.f, 0.f);
          e[2 * j] = bf(w2.x); e[2 * j + 1] = bf(w2.y);
        }
#pragma unroll
        for (int k = 0; k < EPT; k++) s[0] += e[k] * e[k];
        block_sum<1>(s, sm);
        const float r = rsqrtf(s[0] / (float)H + 1e-6f);
#pragma unroll
        for (int j = 0; j < 2; j++) {
          if (!EV(2 * j)) { x[0][2 * j] = x[0][2 * j + 1] = 0.f; continue; }
          const int el = EL(2 * j);
          const __nv_bfloat162 nw = *reinterpret_cast<const __nv_bfloat162*>(c.embed_norm + el);
          const uint32_t hv = pack_bf2(e[2 * j] * r * bf(nw.x), e[2 * j + 1] * r * bf(nw.y));
          st_tag(c.pbt + (long)b * NPAIR + threadIdx.x + j * NTH, hv, TAGP(0));
          x[0][2 * j] = bf_lo(hv); x[0][2 * j + 1] = bf_hi(hv);
        }
      } else {
        float v[5][EPT];
        const int idx[5] = {ap.blk[0], ap.blk[1], ap.blk[2], ap.blk[3], ap.pin};
        const unsigned msk = ((1u << ap.nblk) - 1u) | 16u;
        const float* qs[1] = {L.aq + q * H};
        float qv[1][EPT];
        load_queries<1>(qs, 1, qv);
        load_srcs<5>(c, hd, idx, msk, b, v);
        depth_mix_q<5, 1>(v, msk, qv, 1, x, sm);
      }
      float wv[1][EPT];
      load_w(L.norm[q], wv[0]);
      uint2* dst[1] = {xs_pub + ((long)q * MAXR + b) * NPAIR};
      norm_publish<1>(x, wv, 1, dst, TAGA(a), sm);
    }
    if (!hy) {
      if (big16<V>(hd)) return nq;   // rows polled per half in the qkv phase
      const unsigned need = ((V || POLL_STEP) && POLL_NEEDED) ? qkv_streams(L, nq) : 7u;
      for (int q = 0; q < nq; q++)
        if ((need >> q) & 1u) poll_rows(xs_pub + (long)q * MAXR * NPAIR, B, [&](int r) { return &sm.u.xs[q][r][0]; }, TAGA(a));
      __syncthreads();
      return nq;
    }
  }
  if (ap.nmsg == 0) {
    for (int b = 0; b < (hy ? 1 : B); b++) {
      float v[5][EPT];
      float x[3][EPT];
      if (a == 0) {
        // h0 = rmsnorm(embed[tok]) * embed_norm (bf16); block 0 publishes it as pbt[0]
        const int tok = min(max(hd.tokens ? hd.tokens[b] : (int)poll(c.ntt + b, hd.tok_tag), 0), VOCAB_ROWS - 1);   // memory-safe; ids are checked on the host only where that costs no sync
        const bf16* er = c.embed + (long)tok * H;
        float e[EPT], s[1] = {0.f};
#pragma unroll
        for (int j = 0; j < 2; j++) {
          __nv_bfloat162 w2 = EV(2 * j) ? *reinterpret_cast<const __nv_bfloat162*>(er + EL(2 * j)) : __floats2bfloat162_rn(0.f, 0.f);
          e[2 * j] = bf(w2.x); e[2 * j + 1] = bf(w2.y);
        }
#pragma unroll
        for (int k = 0; k < EPT; k++) s[0] += e[k] * e[k];
        block_sum<1>(s, sm);
        const float r = rsqrtf(s[0] / (float)H + 1e-6f);
#pragma unroll
        for (int j = 0; j < 2; j++) {
          if (!EV(2 * j)) { x[0][2 * j] = x[0][2 * j + 1] = 0.f; continue; }
          const int el = EL(2 * j);
          const __nv_bfloat162 nw = *reinterpret_cast<const __nv_bfloat162*>(c.embed_norm + el);
          const uint32_t hv = pack_bf2(e[2 * j] * r * bf(nw.x), e[2 * j + 1] * r * bf(nw.y));
          if (blockIdx.x == 0) st_tag(c.pbt + (long)b * NPAIR + threadIdx.x + j * NTH, hv, TAGP(0));
          x[0][2 * j] = bf_lo(hv); x[0][2 * j + 1] = bf_hi(hv);
        }
      } else {
        const int idx[5] = {ap.blk[0], ap.blk[1], ap.blk[2], ap.blk[3], ap.pin};
        const unsigned msk = ((1u << ap.nblk) - 1u) | 16u;
        const float* qs[3] = {L.aq, L.aq + H, L.aq + 2 * H};
        float qv[3][EPT];
        SUBMARK(30);
        load_queries<3>(qs, nq, qv);                 // weights first: independent of the polled sources
        load_srcs<5>(c, hd, idx, msk, b, v);
        SUBMARK(31);
        if (nq == 1) depth_mix_q<5, 1>(v, msk, qv, 1, x, sm); else depth_mix_q<5, 3>(v, msk, qv, 3, x, sm);
        SUBMARK(32);
      }
      float wv[3][EPT];
      for (int q = 0; q < 3; q++) load_w(L.norm[q < nq ? q : 0], wv[q]);
      bf16* dst[3] = {&sm.u.xs[0][b][0], &sm.u.xs[1][b][0], &sm.u.xs[2][b][0]};
      if (nq == 1) rmsnorm_store_n<1>(x, wv, 1, dst, sm); else rmsnorm_store_n<3>(x, wv, 3, dst, sm);
      SUBMARK(33);
    }
    if (hy) {   // row 1 from the last blocks
      const uint2* xs_pub = c.xst + (long)par * 3 * MAXR * NPAIR;
      const unsigned need = (POLL_NEEDED) ? qkv_streams(L, nq) : 7u;
      for (int q = 0; q < nq; q++)
        if ((need >> q) & 1u) poll_rows(xs_pub + ((long)q * MAXR + 1) * NPAIR, B - 1, [&](int r) { return &sm.u.xs[q][1 + r][0]; }, TAGA(a));
    }
    __syncthreads();
    return nq;
  }
  return -1;
}
// mlp-gate G1: block per token (controller 3)
__device__ void mlp_mix(PH_ARGS, int a, int b, float (*x)[EPT]);
__device__ void ph_mlp_g1(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const int B = hd.B, par = a & 1;
  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    float x[1][EPT], m[1][EPT], ms[2][EPT];
    mlp_mix(c, hd, sm, a, b, x);
    load_srcs<2>(c, hd, ap.msg, 3u, b, ms);
    const float* mq[1] = {c.loopq + ((long)ap.loopl * 4 + 3) * H};
    depth_mix<2, 1>(ms, 3u, mq, 1, m, sm);
    float s2[2] = {0.f, 0.f};
#pragma unroll
    for (int k = 0; k < EPT; k++) { s2[0] += x[0][k] * x[0][k]; s2[1] += m[0][k] * m[0][k]; }
    block_sum<2>(s2, sm);
    const float rx = rsqrtf(s2[0] / (float)H + 1e-6f), rm = rsqrtf(s2[1] / (float)H + 1e-6f);
    const long o = ((long)(par * MAXR + b) * 4 + 3) * H;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      if (!EV(2 * j)) continue;
      const int el = EL(2 * j);
      st_tagf(c.xgt + o + el, x[0][2 * j], TAGA(a)); st_tagf(c.xgt + o + el + 1, x[0][2 * j + 1], TAGA(a));
      st_tagf(c.mgt + o + el, m[0][2 * j], TAGA(a)); st_tagf(c.mgt + o + el + 1, m[0][2 * j + 1], TAGA(a));
      st_tag(c.ubt + o + el / 2, pack_bf2(x[0][2 * j] * rx, x[0][2 * j + 1] * rx), TAGA(a));
      st_tag(c.ubt + o + NPAIR + el / 2, pack_bf2(m[0][2 * j] * rm, m[0][2 * j + 1] * rm), TAGA(a));
    }
  }
}

// Gate down-projection D [64, 4608] of controllers [c0, c0 + nc) (c0 = 3: the MLP input gate). One block per
// (controller, strip of 16 rows, token): it computes the token's stream x (attention-input or MLP-input mix) and the
// message m itself, the unit vectors [x / rms(x), m / rms(m)] as the B column in shared memory, 24 warps x 3 units;
// strip 0 also publishes x and m (f32) for the F / G phase.
#ifndef GATE_MFIRST_STEP
#define GATE_MFIRST_STEP 0   // ... also in the step kernels (track A)
#endif
#ifndef GATE_MFIRST
#define GATE_MFIRST 1   // gate D items: the loop-message mix (ready since the previous pass) before the x mix
#endif
#ifndef GD2
#define GD2 0   // (measured slower: 8620 vs 8548 us at K=16; V(12) +240 us from code generation)
#endif
template <bool V> __device__ void ph_gate_d(PH_ARGS, int a, int c0, int nc) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = hd.B, par = a & 1;
  // GD2 (16-row passes with more D jobs than blocks): a job does JP of its row's four D strips in turn after one mix
  const int JP = (GD2 && V && B > SMB && nc * 4 * B > (int)gridDim.x) ? 2 : 1, NQ = 4 / JP;
  const int N = nc * NQ * B;
  constexpr int ST = 2 * H + 8;
  bf16* xb = &sm.u.xs[0][0][0];                                   // [1][ST]
  for (int it = blockIdx.x; it < N; it += gridDim.x) {
    const int b = it % B, sj = it / B, ctrl = c0 + sj / NQ, jq = sj % NQ;
    const int j0 = jq * JP;
    float x[1][EPT], m[1][EPT], ms[2][EPT];
    {
      const float* mq[1] = {c.loopq + ((long)ap.loopl * 4 + ctrl) * H};
      float mqv[1][EPT];
      load_queries<1>(mq, 1, mqv);
      const bool mfirst = (V || GATE_MFIRST_STEP) && GATE_MFIRST;
      if (mfirst) {
        load_srcs<2>(c, hd, ap.msg, 3u, b, ms);
        depth_mix_q<2, 1>(ms, 3u, mqv, 1, m, sm);
      }
      if (ctrl < 3) {
        float v[5][EPT];
        const int idx[5] = {ap.blk[0], ap.blk[1], ap.blk[2], ap.blk[3], ap.pin};
        const unsigned msk = ((1u << ap.nblk) - 1u) | 16u;
        const float* qs[1] = {L.aq + (L.glob ? ctrl : 0) * H};
        float qv[1][EPT];
        load_queries<1>(qs, 1, qv);
        load_srcs<5>(c, hd, idx, msk, b, v);
        depth_mix_q<5, 1>(v, msk, qv, 1, x, sm);
      } else {
        mlp_mix(c, hd, sm, a, b, x);
      }
      if (!mfirst) {
        load_srcs<2>(c, hd, ap.msg, 3u, b, ms);
        depth_mix_q<2, 1>(ms, 3u, mqv, 1, m, sm);
      }
    }
    float s2[2] = {0.f, 0.f};
#pragma unroll
    for (int k = 0; k < EPT; k++) { s2[0] += x[0][k] * x[0][k]; s2[1] += m[0][k] * m[0][k]; }
    block_sum<2>(s2, sm);
    const float rx = rsqrtf(s2[0] / (float)H + 1e-6f), rm = rsqrtf(s2[1] / (float)H + 1e-6f);
    const long o = ((long)(par * MAXR + b) * 4 + ctrl) * H;
#pragma unroll
    for (int jj = 0; jj < 2; jj++) {
      if (!EV(2 * jj)) continue;
      const int el = EL(2 * jj);
      *reinterpret_cast<uint32_t*>(&xb[el]) = pack_bf2(x[0][2 * jj] * rx, x[0][2 * jj + 1] * rx);
      *reinterpret_cast<uint32_t*>(&xb[H + el]) = pack_bf2(m[0][2 * jj] * rm, m[0][2 * jj + 1] * rm);
      if (j0 == 0) {
        st_tagf(c.xgt + o + el, x[0][2 * jj], TAGA(a)); st_tagf(c.xgt + o + el + 1, x[0][2 * jj + 1], TAGA(a));
        st_tagf(c.mgt + o + el, m[0][2 * jj], TAGA(a)); st_tagf(c.mgt + o + el + 1, m[0][2 * jj + 1], TAGA(a));
      }
    }
    __syncthreads();
    BSm bl{xb, ST, 1};
    for (int j = j0; j < j0 + JP; j++) {
    const int strip = ctrl * 4 + j;
    block_strips<NWARP, 1>(sm, 1, 72,
      [&](int, int u0, int nu, float (*acc)[4]) { piece_bf16(L.cD + (long)strip * 72 * 2048, u0, nu, bl, acc[0]); },
      [&](int, float (*acc)[4]) {
        const int lane = lane_id(), g = lane >> 2, t = lane & 3;
        if (t == 0) {   // column 0 = this token
          st_tagf(c.gvt + ((long)(par * MAXR + b) * 4 + ctrl) * 64 + j * 16 + g, rbf(acc[0][0]), TAGA(a));
          st_tagf(c.gvt + ((long)(par * MAXR + b) * 4 + ctrl) * 64 + j * 16 + g + 8, rbf(acc[0][2]), TAGA(a));
        }
      });
    }
    __syncthreads();
  }
}
// F / G elementwise for controllers [c0, c0 + nc): xpt[b][ctrl][i]; one thread per (ctrl, b, i)
#ifndef FG16
#define FG16 0
#endif
__device__ void ph_gate_fg(PH_ARGS, int a, int c0, int nc) {
  const LayerW& L = c.L[c.A[a].layer];
  const int B = hd.B, par = a & 1;
  if (FG16 > 0 && B > SMB) {   // 16-row passes: one task per (controller, channel, RB rows): the weight row loaded once (mk.cu FG_ROWS)
    constexpr int RB = FG16 > 0 ? FG16 : 1;   // rows per task
    const int NG_ = (B + RB - 1) / RB;
    const int N = nc * H * NG_;
    const int nd = nc * 4 * B;
    const int f0 = (nd < (int)gridDim.x && ((int)gridDim.x - nd) * (NTH / 2) >= N) ? nd : 0, nf = (int)gridDim.x - f0;
    if ((int)blockIdx.x < f0) return;
    const int fb = blockIdx.x - f0;
    bool any = false;
    for (int it = threadIdx.x * nf + fb; it < N; it += nf * NTH) any = true;
    if (!__syncthreads_or(any)) return;
    float* gs = reinterpret_cast<float*>(&sm.u.xs[0][0][0]);   // [B][4][64]
    const int tsk0 = (threadIdx.x >> 1) * nf + fb, odd = threadIdx.x & 1;
    for (int k = threadIdx.x; k < B * nc * 64; k += NTH) {
      const int r = k % 64, bc = k / 64, bb = bc / nc, cc = c0 + bc % nc;
      gs[(bb * 4 + cc) * 64 + r] = pollf(c.gvt + ((long)(par * MAXR + bb) * 4 + cc) * 64 + r, TAGA(a));
    }
    __syncthreads();
    for (int it0 = 0; it0 < N; it0 += nf * (NTH / 2)) {
      const int it = it0 + tsk0;
      const bool ok = it < N;
      const int itc = ok ? it : 0;
      const int i = itc % H, rest = itc / H, bg = rest % NG_, ctrl = c0 + rest / NG_;
      const int bb0 = bg * RB, bb1 = min(B, bb0 + RB);
      const uint4* R = reinterpret_cast<const uint4*>(L.cFG + ((long)ctrl * 2 * H + odd * H + i) * 64);
      uint4 fw[8];
#pragma unroll
      for (int q = 0; q < 8; q++) fw[q] = ldg_nc(R + q);
      uint2 rx[RB];
#pragma unroll
      for (int j = 0; j < RB; j++) {
        const long o = ((long)(par * MAXR + min(bb0 + j, bb1 - 1)) * 4 + ctrl) * H + i;
        rx[j] = ld_tag(odd ? c.mgt + o : c.xgt + o);
      }
#pragma unroll
      for (int j = 0; j < RB; j++) {
        const int b = min(bb0 + j, bb1 - 1);
        const bool okj = ok && bb0 + j < bb1;   // warp-uniform loop: no early exit around the shuffle
        const long o = ((long)(par * MAXR + b) * 4 + ctrl) * H + i;
        const float* gvv = gs + (b * 4 + ctrl) * 64;
        float f = 0.f;
#pragma unroll
        for (int q = 0; q < 8; q++) {
          const bf16* fp = reinterpret_cast<const bf16*>(&fw[q]);
#pragma unroll
          for (int r = 0; r < 8; r++) f = fmaf(bf(fp[r]), gvv[q * 8 + r], f);
        }
        f = tanhf(f);
        const float val = okj ? __uint_as_float(wait_tag(odd ? c.mgt + o : c.xgt + o, rx[j], TAGA(a))) : 0.f;
        const float mine = odd ? f * val : (1.f - f) * val;
        const float other = __shfl_xor_sync(0xffffffff, mine, 1);
        if (okj && !odd) st_tagf(c.xpt + o, mine + other, TAGA(a));
      }
    }
    return;
  }
  const int N = nc * B * H;
  // tasks only on the blocks without a D job (those start late, after their own job), unless every block has one
  const int nd = nc * 4 * B;
  const int f0 = (nd < (int)gridDim.x && ((int)gridDim.x - nd) * (NTH / 2) >= N) ? nd : 0, nf = (int)gridDim.x - f0;
  if ((int)blockIdx.x < f0) return;
  const int fb = blockIdx.x - f0;
  bool any = false;
  for (int it = threadIdx.x * nf + fb; it < N; it += nf * NTH) any = true;
  if (!__syncthreads_or(any)) return;
  float* gs = reinterpret_cast<float*>(&sm.u.xs[0][0][0]);   // [B][4][64]
  // two lanes per (ctrl, b, i): even lane the forget row, odd lane the gain row
  const int tsk0 = (threadIdx.x >> 1) * nf + fb, odd = threadIdx.x & 1;
  for (int it0 = 0; it0 < N; it0 += nf * (NTH / 2)) {
    const int it = it0 + tsk0;
    const bool ok = it < N;
    const int itc = ok ? it : 0;
    const int i = itc % H, rest = itc / H, b = rest % B, ctrl = c0 + rest / B;
    // independent loads first: the weight row and x (or m), then the gate vector g
    const uint4* R = reinterpret_cast<const uint4*>(L.cFG + ((long)ctrl * 2 * H + odd * H + i) * 64);
    uint4 fw[8];
#pragma unroll
    for (int q = 0; q < 8; q++) fw[q] = ldg_nc(R + q);
    const long o = ((long)(par * MAXR + b) * 4 + ctrl) * H + i;
    const uint2 rx = ld_tag(odd ? c.mgt + o : c.xgt + o);
    if (it0 == 0) {
      for (int k = threadIdx.x; k < B * nc * 64; k += NTH) {
        const int r = k % 64, bc = k / 64, bb = bc / nc, cc = c0 + bc % nc;
        gs[(bb * 4 + cc) * 64 + r] = pollf(c.gvt + ((long)(par * MAXR + bb) * 4 + cc) * 64 + r, TAGA(a));
      }
      __syncthreads();
    }
    const float* gvv = gs + (b * 4 + ctrl) * 64;
    float f = 0.f;
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const bf16* fp = reinterpret_cast<const bf16*>(&fw[q]);
#pragma unroll
      for (int r = 0; r < 8; r++) f = fmaf(bf(fp[r]), gvv[q * 8 + r], f);
    }
    f = tanhf(f);
    const float val = ok ? __uint_as_float(wait_tag(odd ? c.mgt + o : c.xgt + o, rx, TAGA(a))) : 0.f;
    const float mine = odd ? f * val : (1.f - f) * val;   // gain * m  |  (1 - forget) * x
    const float other = __shfl_xor_sync(0xffffffff, mine, 1);
    if (ok && !odd) st_tagf(c.xpt + o, mine + other, TAGA(a));
  }
}

// QKV GEMV + epilogue: qkvt[par][b][row] (bf16-rounded f32). Block b: strips b, b + G, b + 2G, b + 3G; 6 warps per strip.
template <bool V> __device__ void ph_qkv(PH_ARGS, int a, int nstreams) {
  const int ro = V ? hd.ro : 0;
  const LayerW& L = c.L[c.A[a].layer];
  const int B = hd.B, par = a & 1;
  const int nsk = L.nsk;
  const int strips = QG_STRIPS + 2 * nsk;
  const int G = gridDim.x;
  const int ns = (strips - (int)blockIdx.x + G - 1) / G;
  block_strips<6, 1>(sm, ns, 36,
    [&](int q, int u0, int nu, float (*acc)[4]) {
      const int strip = blockIdx.x + q * G;
      int st = strip < QG_STRIPS ? 0 : (strip < QG_STRIPS + nsk ? 1 : 2);
      if (st >= nstreams) st = 0;
      BSm bl{&sm.u.xs[st][0][0], HP, B};
#ifdef W12
      piece_w12(L.qkv12, strip, 36, u0, nu, bl, acc[0]);
#else
      piece_bf16(L.wqkv + (long)strip * 36 * 2048, u0, nu, bl, acc[0]);
#endif
    },
    [&](int q, float (*acc)[4]) {
      const int strip = blockIdx.x + q * G;
      const int st = strip < QG_STRIPS ? 0 : (strip < QG_STRIPS + nsk ? 1 : 2);
      const int orow = st == 0 ? strip * 16 : (st == 1 ? K0 + (strip - QG_STRIPS) * 16 : V0 + (strip - QG_STRIPS - nsk) * 16);
      const int lane = lane_id(), g = lane >> 2, t = lane & 3;
      for (int h = 0; h < 2; h++)
        for (int cc = 0; cc < 2; cc++) {
          const int tok = 2 * t + cc;
          if (tok < B) st_tagf(c.qkvt + (long)(par * MAXR + ro + tok) * QKVR + orow + g + 8 * h, rbf(acc[0][2 * h + cc]), TAGA(a));
        }
    });
}

#ifndef QKV1
#define QKV1 0   // 16-row passes: qkv strips in one pass (both 8-row column groups per weight fragment): measured no gain (8786 vs 8772 us), off
#endif
__device__ __forceinline__ int qkv_stream_of(int strip, int nsk, int nstreams) {
  const int st = strip < QG_STRIPS ? 0 : (strip < QG_STRIPS + nsk ? 1 : 2);
  return st >= nstreams ? 0 : st;
}
// one block's qkv strips of a 16-row pass in one pass: inputs xs16[0] = stream 0, xs16[1] = the block's other stream
__device__ void ph_qkv16(PH_ARGS, int a, int nstreams, int other) {
  const LayerW& L = c.L[c.A[a].layer];
  const int B = hd.B, par = a & 1;
  const int nsk = L.nsk;
  const int strips = QG_STRIPS + 2 * nsk;
  const int G = gridDim.x;
  const int ns = (strips - (int)blockIdx.x + G - 1) / G;
  block_strips<6, 2>(sm, ns, 36,
    [&](int q, int u0, int nu, float (*acc)[4]) {
      const int strip = blockIdx.x + q * G;
      const int slot = qkv_stream_of(strip, nsk, nstreams) == 0 ? 0 : 1;
      BSm b0{&sm.u.xs16[slot][0][0], HP, SMB}, b1{&sm.u.xs16[slot][SMB][0], HP, B - SMB};
      piece_bf16_2(L.wqkv + (long)strip * 36 * 2048, u0, nu, b0, b1, acc);
    },
    [&](int q, float (*acc)[4]) {
      const int strip = blockIdx.x + q * G;
      const int st = strip < QG_STRIPS ? 0 : (strip < QG_STRIPS + nsk ? 1 : 2);
      const int orow = st == 0 ? strip * 16 : (st == 1 ? K0 + (strip - QG_STRIPS) * 16 : V0 + (strip - QG_STRIPS - nsk) * 16);
      const int lane = lane_id(), g = lane >> 2, t = lane & 3;
      for (int ng = 0; ng < 2; ng++)
        for (int h = 0; h < 2; h++)
          for (int cc = 0; cc < 2; cc++) {
            const int tok = SMB * ng + 2 * t + cc;
            if (tok < B) st_tagf(c.qkvt + (long)(par * MAXR + tok) * QKVR + orow + g + 8 * h, rbf(acc[ng][2 * h + cc]), TAGA(a));
          }
    });
  (void)other;
}

// attention: items (b, kv head g, chunk of csz positions); the last chunk of (b, g) to finish merges
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

// k / v of row r (position t, nin earlier rows of its sequence in this pass): raw current value, the raw values of the
// three previous positions (earlier rows of this pass: their tagged qkv output; before: the history ring), conv
// (static for local layers, dynamic taps for global)
__device__ __forceinline__ float kv_conv(const LayerW& L, const uint2* qkvp, int r, unsigned tag, const bf16* hist, const bf16* sc, int base_row, int ch2, int g, int t, int nin, float* xraw = nullptr) {
  const uint2* qr = qkvp + (long)r * QKVR;
  uint2 rd[5];
  rd[0] = ld_tag(qr + base_row + ch2);
#pragma unroll
  for (int lag = 0; lag < 4; lag++) rd[1 + lag] = ld_tag(qr + base_row + 1024 + (L.glob ? g * 4 + lag : 0));
  const bf16* hb = hist + ch2;
  float xl[3];
  uint2 rl[3];
#pragma unroll
  for (int l = 1; l <= 3; l++)
    if (l <= nin) rl[l - 1] = ld_tag(qkvp + (long)(r - l) * QKVR + base_row + ch2);
    else xl[l - 1] = ldcg_bf(hb + ((t - l) & (RING - 1)) * 1024);
  const float x = __uint_as_float(wait_tag(qr + base_row + ch2, rd[0], tag));
  if (xraw) *xraw = x;
#pragma unroll
  for (int l = 1; l <= 3; l++)
    if (l <= nin) xl[l - 1] = __uint_as_float(wait_tag(qkvp + (long)(r - l) * QKVR + base_row + ch2, rl[l - 1], tag));
  const float x1 = xl[0], x2 = xl[1], x3 = xl[2];
  if (!L.glob) {
    const uint2 wv = *reinterpret_cast<const uint2*>(sc + ch2 * 4);
    const bf16* w4 = reinterpret_cast<const bf16*>(&wv);
    float acc = __fmul_rn(x3, bf(w4[0]));
    acc = __fadd_rn(acc, __fmul_rn(x2, bf(w4[1])));
    acc = __fadd_rn(acc, __fmul_rn(x1, bf(w4[2])));
    acc = __fadd_rn(acc, __fmul_rn(x, bf(w4[3])));
    return rbf(__fmul_rn(__fadd_rn(acc, x), 0.44721359549995793f));
  }
  const uint2* dyn = qr + base_row + 1024 + g * 4;
  const float xs4[4] = {x, x1, x2, x3};
  float acc = 0.f;
#pragma unroll
  for (int lag = 0; lag < 4; lag++) acc = __fadd_rn(acc, __fmul_rn(xs4[lag], __fadd_rn(bf(sc[lag * 1024 + ch2]), __uint_as_float(wait_tag(dyn + lag, rd[1 + lag], tag)))));
  return rbf(__fmul_rn(__fadd_rn(acc, x), 0.33333333333333333f));
}

// kv_conv of the 4 channels ch0 .. ch0 + 3 of one lane with every load issued before the first wait (one round trip
// instead of four); the same arithmetic per channel as kv_conv
__device__ __forceinline__ void kv_conv4(const LayerW& L, const uint2* qkvp, int r, unsigned tag, const bf16* hist, const bf16* sc, int base_row, int ch0, int g, int t, int nin, float* out) {
  const uint2* qr = qkvp + (long)r * QKVR;
  uint2 rx[4], rd[4], rl[3][4];
  float xl[3][4];
#pragma unroll
  for (int k = 0; k < 4; k++) rx[k] = ld_tag(qr + base_row + ch0 + k);
#pragma unroll
  for (int lag = 0; lag < 4; lag++) rd[lag] = ld_tag(qr + base_row + 1024 + (L.glob ? g * 4 + lag : 0));
#pragma unroll
  for (int l = 1; l <= 3; l++)
#pragma unroll
    for (int k = 0; k < 4; k++)
      if (l <= nin) rl[l - 1][k] = ld_tag(qkvp + (long)(r - l) * QKVR + base_row + ch0 + k);
      else xl[l - 1][k] = ldcg_bf(hist + ch0 + k + ((t - l) & (RING - 1)) * 1024);
  float x[4];
#pragma unroll
  for (int k = 0; k < 4; k++) x[k] = __uint_as_float(wait_tag(qr + base_row + ch0 + k, rx[k], tag));
#pragma unroll
  for (int l = 1; l <= 3; l++)
#pragma unroll
    for (int k = 0; k < 4; k++)
      if (l <= nin) xl[l - 1][k] = __uint_as_float(wait_tag(qkvp + (long)(r - l) * QKVR + base_row + ch0 + k, rl[l - 1][k], tag));
  if (!L.glob) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const uint2 wv = *reinterpret_cast<const uint2*>(sc + (ch0 + k) * 4);
      const bf16* w4 = reinterpret_cast<const bf16*>(&wv);
      float acc = __fmul_rn(xl[2][k], bf(w4[0]));
      acc = __fadd_rn(acc, __fmul_rn(xl[1][k], bf(w4[1])));
      acc = __fadd_rn(acc, __fmul_rn(xl[0][k], bf(w4[2])));
      acc = __fadd_rn(acc, __fmul_rn(x[k], bf(w4[3])));
      out[k] = rbf(__fmul_rn(__fadd_rn(acc, x[k]), 0.44721359549995793f));
    }
    return;
  }
  const uint2* dyn = qr + base_row + 1024 + g * 4;
  float dv[4];
#pragma unroll
  for (int lag = 0; lag < 4; lag++) dv[lag] = __uint_as_float(wait_tag(dyn + lag, rd[lag], tag));
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const float xs4[4] = {x[k], xl[0][k], xl[1][k], xl[2][k]};
    float acc = 0.f;
#pragma unroll
    for (int lag = 0; lag < 4; lag++) acc = __fadd_rn(acc, __fmul_rn(xs4[lag], __fadd_rn(bf(sc[lag * 1024 + ch0 + k]), dv[lag])));
    out[k] = rbf(__fmul_rn(__fadd_rn(acc, x[k]), 0.33333333333333333f));
  }
}

__device__ __forceinline__ void ldsm_x4(uint32_t* r, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
SHARED_ATTR __device__ void ph_attn_shared(PH_ARGS, int a);
template <bool V> __device__ void ph_attn(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = hd.B;
#ifndef SHARED_MIN
#define SHARED_MIN 2   // verify passes of at least this many rows use the shared-record attention
#endif
  if (V && !HD_MT(hd) && B >= SHARED_MIN && (L.glob || row_t<V>(hd, B - 1) + 1 <= 1024)) { ph_attn_shared(c, hd, sm, a); return; }
  // records (online-softmax partials) always cover 64 keys, so that the merge is the same for every batch size;
  // for B >= 3 an item loads two records' keys (fewer items) and computes the two records separately
  const int sub = B <= 2 ? 1 : 2, csz = 64 * sub;
  const int tmax = row_tmax<V>(hd);              // the largest row position (multi-sequence passes: any order)
  const int lomax = L.glob ? 0 : max(0, tmax + 1 - 1024);
  const int nchmax = (tmax - lomax + csz) / csz;
  const int N = B * NKV * nchmax;
  const unsigned tag = TAGA(a);
  const int par = a & 1;
  const bool use_tau = L.glob;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
  auto& at = sm.u.at;
  const long cstride = (long)c.maxctx * HD;   // per (slot, b, head)
  // items: chunk-major from the last chunk down; chunk 0 of (b, g) merges and comes last in every block's order
  for (int it = blockIdx.x; it < N; it += gridDim.x) {
    const int ch = nchmax - 1 - it / (B * NKV), bg = it % (B * NKV), g = bg % NKV, b = bg / NKV;
    const int t = row_t<V>(hd, b), nin = row_nin<V>(hd, b), seq = row_seq<V>(hd, b);
    const int lo = L.glob ? 0 : max(0, t + 1 - 1024);
    const int nch = (t - lo + csz) / csz;      // items per (token, kv head)
    const int nrc = (t - lo + 64) / 64;        // records per (token, kv head)
    if (ch >= nch) continue;                   // block-uniform
    const float tau = (L.glob && t + 1 > 8192) ? (1.0f + 0.1f * logf((float)(t + 1) / 8192.0f)) : 1.0f;   // = the max(1, .) form
    const int j0 = lo + ch * csz, j1 = min(j0 + csz, t + 1), nj = j1 - j0;
    const bool cur = (j1 == t + 1);
    // positions >= tp0 belong to rows of this pass: their k / v are computed here from the tagged qkv outputs
    const int tp0 = t - nin, P0 = V ? min(max(j0, tp0), j1) : (cur ? t : j1), np = V ? j1 - P0 : (cur ? 1 : 0);
    const uint2* qkvp = c.qkvt + (long)(par * MAXR) * QKVR;
    const uint2* qr = qkvp + (long)b * QKVR;
    SUBMARK(20);
    // ---- one wave of loads: K / V chunk (async), relative-bias rows, q / r, current k / v
    {
      const bf16* kb = c.kc + (((long)ap.slot * MAXB + seq) * NKV + g) * cstride + (long)j0 * HD;
      const bf16* vb = c.vc + (((long)ap.slot * MAXB + seq) * NKV + g) * cstride + (long)j0 * HD;
      const int nld = (P0 - j0) * 16;
      for (int i = tid; i < nld; i += NTH) {
        const int r = i >> 4, cc = (i & 15) * 8;
        cp_async16(&at.kk[r][cc], reinterpret_cast<const uint4*>(kb) + i);
        cp_async16(&at.vv[r][cc], reinterpret_cast<const uint4*>(vb) + i);
      }
    }
    for (int i = tid; i < 16 * nj; i += NTH) {
      const int r = i / nj, jj = i % nj, dist = t - (j0 + jj);
      at.pr[r][jj] = dist < 1024 ? bf(L.relp[r * 1024 + min(dist, 1023)]) : 0.f;   // index clamped: the select may load
    }
    // zero: P rows 3..15, q rows 3..7, V rows beyond nj up to the next 16
    for (int i = tid; i < 13 * (CH + 8) / 2; i += NTH) {
      reinterpret_cast<uint32_t*>(&at.ph[3][0])[i] = 0u;
      reinterpret_cast<uint32_t*>(&at.pl[3][0])[i] = 0u;
    }
    for (int i = tid; i < 5 * (HD + 8) / 2; i += NTH) reinterpret_cast<uint32_t*>(&at.qb[3][0])[i] = 0u;
    {
      const int nj16 = (nj + 15) & ~15;
      for (int i = tid; i < (nj16 - nj) * (HD / 2); i += NTH) reinterpret_cast<uint32_t*>(&at.vv[nj + i / (HD / 2)][0])[i % (HD / 2)] = 0u;
    }
    if (w < 3) {
      const int h = 3 * g + w;
      float qv[4], ss = 0.f;
      uint2 rq[4];
#pragma unroll
      for (int k = 0; k < 4; k++) rq[k] = ld_tag(qr + h * HD + lane * 4 + k);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const int d = lane * 4 + k;
        qv[k] = __uint_as_float(wait_tag(qr + h * HD + d, rq[k], tag)) * bf(L.qscale[h * HD + d]);
        ss += qv[k] * qv[k];
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      const float f = fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        float y = rbf(qv[k] * f);
        if (use_tau) y = rbf(y * tau);
        at.qb[w][lane * 4 + k] = tobf(y);
      }
    } else if (w < 6) {
      const int hh = w - 3, h = 3 * g + hh;
      float rv = 0.f, ss = 0.f;
      if (lane < 16) { rv = pollf(qr + R_R + h * 16 + lane, tag) * bf(L.rscale[h * 16 + lane]); ss = rv * rv; }
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      const float f = fminf(8.0f / (sqrtf(ss) + 1e-6f), 1.0f);
      if (lane < 16) {
        float y = rbf(rv * f);
        if (use_tau) y = rbf(y * tau);
        at.rr[hh][lane] = y;
      }
    } else if (w >= 6 && w < 6 + 2 * np) {
      // position P0 + i of row b - d (d = t - P0 - i): k (even warp) or v (odd warp)
      const bool isv = (w & 1);
      const int i = V ? (w - 6) >> 1 : 0, d = V ? t - (P0 + i) : 0;
      const bf16* hist = (isv ? c.hv : c.hk) + ((long)ap.slot * MAXB + seq) * RING * 1024;
      float kv[4], ss = 0.f;
      if (KV4 && (V || KV4_STEP)) {
        kv_conv4(L, qkvp, b - d, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, g * HD + lane * 4, g, t - d, nin - d, kv);
#pragma unroll
        for (int k = 0; k < 4; k++) ss += kv[k] * kv[k];
      } else {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          const int ch2 = g * HD + lane * 4 + k;
          kv[k] = kv_conv(L, qkvp, b - d, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, ch2, g, t - d, nin - d,
                          (RAWSM && (V || RAWSM_STEP) && d == 0) ? &at.raw[isv][lane * 4 + k] : nullptr);
          ss += kv[k] * kv[k];
        }
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      const float f = isv ? rsqrtf(ss / 128.f + 1e-6f) : fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
      float* dst = isv ? at.vt[i] : at.kt[i];
#pragma unroll
      for (int k = 0; k < 4; k++) dst[lane * 4 + k] = rbf(kv[k] * f);
    }
    cp_async_wait();
    __syncthreads();
    SUBMARK(21);
    if (tid < HD) {
      // positions of this pass: from the values computed above; the current one: write the cache and the conv history
      for (int i = 0; i < np; i++) {
        at.kk[P0 - j0 + i][tid] = tobf(at.kt[i][tid]);
        at.vv[P0 - j0 + i][tid] = tobf(at.vt[i][tid]);
      }
    }
    if (cur && tid < HD) {
      const long co = (((long)ap.slot * MAXB + seq) * NKV + g) * cstride + (long)t * HD + tid;
      c.kc[co] = tobf(at.kt[np - 1][tid]);
      c.vc[co] = tobf(at.vt[np - 1][tid]);
      const long hrow = ((long)ap.slot * MAXB + seq) * RING * 1024 + (t & (RING - 1)) * 1024 + g * HD + tid;
      if (RAWSM && (V || RAWSM_STEP)) {   // the raw current k / v as the k / v warps loaded them
        c.hk[hrow] = tobf(at.raw[0][tid]);
        c.hv[hrow] = tobf(at.raw[1][tid]);
      } else {
        c.hk[hrow] = tobf(pollf(qr + K0 + g * HD + tid, tag));
        c.hv[hrow] = tobf(pollf(qr + V0 + g * HD + tid, tag));
      }
    }
    // relative bias: bias[h][jj] = bf16(sum_r rr[h][r] * P[r][t - j]) (0 beyond 1023); kept in sc until the scores
    if (tid < 3 * CH) {
      const int hh = tid / CH, jj = tid % CH;
      if (jj < nj) {
        float bsum = 0.f;
#pragma unroll
        for (int r = 0; r < 16; r++) bsum += at.rr[hh][r] * at.pr[r][jj];
        at.sc[hh][jj] = (t - (j0 + jj) < 1024) ? rbf(bsum) : 0.f;
      }
    }
    __syncthreads();
    SUBMARK(22);
    // scores on the tensor cores: S^T[key][head] = K[key][:] . q[head][:]; one warp per 16 keys
    if (w < (nj + 15) / 16) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < HD / 16; ks++) {
        uint32_t af[4];
        ldsm_x4(af, &at.kk[16 * w + (lane & 15)][16 * ks + (lane >> 4) * 8]);
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&at.qb[g8][16 * ks + 2 * t4]);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&at.qb[g8][16 * ks + 8 + 2 * t4]);
        mma_bf16(acc, af[0], af[1], af[2], af[3], b0, b1);
      }
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int key = 16 * w + g8 + (q >> 1) * 8, head = 2 * t4 + (q & 1);
        if (head < 3 && key < nj) at.sc[head][key] = acc[q] * 0.08838834764831845f + at.sc[head][key];
      }
    }
    __syncthreads();
    if (w < 3 * sub) {
      // softmax of record sc (keys [64 sc, 64 sc + 64) of the item) for head w % 3
      const int hh = w % 3, sc = w / 3, k0 = 64 * sc, k1 = min(k0 + 64, nj);
      float m = -1e30f;
      for (int jj = k0 + lane; jj < k1; jj += 32) m = fmaxf(m, at.sc[hh][jj]);
#pragma unroll
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
      float l = 0.f;
      for (int jj = k0 + lane; jj < k0 + 64; jj += 32) {
        const float e = jj < k1 ? expf(at.sc[hh][jj] - m) : 0.f;
        const bf16 eh = tobf(e);
        at.ph[hh][jj] = eh;
        at.pl[hh][jj] = tobf(e - bf(eh));
        l += e;
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffff, l, o);
      if (lane == 0) { at.ml[sc][hh][0] = m; at.ml[sc][hh][1] = l; }
    }
    __syncthreads();
    SUBMARK(23);
    // PV on the tensor cores per record: O[head][d] = sum_j (Phi + Plo)[head][j] V[j][d]; warp w < 8: dims 16w .. +15
    const bool merger = (ch == 0) && nrc > 1;
    {
      // warps 8 sc .. 8 sc + 7: record sc (dims 16 (w % 8) .. +15); warp 8 sub: the records' (m, l)
      const int sc = w >> 3, wd = w & 7;
      const int k0 = 64 * sc, nk = min(64, nj - k0);
      const int rc = sub * ch + sc;                  // record index
      const bool local = rc == 0 && (merger || nrc == 1);
      if (sc < sub && nk > 0) {
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int ks = k0 / 16; ks < (k0 + nk + 15) / 16; ks++) {
          uint32_t ah[4], al[4], bv[4];
          ldsm_x4(ah, &at.ph[lane & 15][16 * ks + (lane >> 4) * 8]);
          ldsm_x4(al, &at.pl[lane & 15][16 * ks + (lane >> 4) * 8]);
          ldsm_x4_t(bv, &at.vv[16 * ks + (lane & 15)][16 * wd + (lane >> 4) * 8]);
          mma_bf16(acc[0], ah[0], ah[1], ah[2], ah[3], bv[0], bv[1]);
          mma_bf16(acc[0], al[0], al[1], al[2], al[3], bv[0], bv[1]);
          mma_bf16(acc[1], ah[0], ah[1], ah[2], ah[3], bv[2], bv[3]);
          mma_bf16(acc[1], al[0], al[1], al[2], al[3], bv[2], bv[3]);
        }
        if (g8 < 3) {
          const int hh = g8;
#pragma unroll
          for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int q = 0; q < 2; q++) {
              const int d = 16 * wd + 8 * nt + 2 * t4 + q;
              if (local) at.po[hh][d] = acc[nt][q];
              else st_tagf(c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH + rc) * 130 + 2 + d, acc[nt][q], tag);
            }
        }
      } else if (w == 8 * sub && lane < 3 * sub) {
        const int s2 = lane / 3, hh = lane % 3, r2 = sub * ch + s2;
        const bool loc2 = r2 == 0 && (merger || nrc == 1);
        if (64 * s2 < nj && !loc2) {
          uint2* rec = c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH + r2) * 130;
          st_tagf(rec, at.ml[s2][hh][0], tag);
          st_tagf(rec + 1, at.ml[s2][hh][1], tag);
        }
      }
    }
    SUBMARK(24);
    if (merger || nrc == 1) {
      __syncthreads();
      float gate = 0.f, sink = 0.f;
      if (tid < 3 * HD) {
        const int hh = tid / HD, d = tid % HD, h = 3 * g + hh;
        const uint2 rg = ld_tag(qr + R_G + h * HD + d), rs = ld_tag(qr + R_S + h);
        gate = __uint_as_float(wait_tag(qr + R_G + h * HD + d, rg, tag));
        sink = __uint_as_float(wait_tag(qr + R_S + h, rs, tag));
      }
      float* st = sm.u.mrg;   // [3][nrc][130]; record 0 from this block
      // the own record first: the staging of the other records grows over at.po / at.ml beyond 57 records
      // (same union); the own record's slots (head hh: [hh nrc 130, +130), < 67 KB) never reach them
      {
      const float pv = tid < 3 * HD ? at.po[tid / HD][tid % HD] : 0.f;
      const float m0 = tid < 3 * HD ? at.ml[0][tid / HD][0] : 0.f, l0 = tid < 3 * HD ? at.ml[0][tid / HD][1] : 0.f;
      __syncthreads();
      if (tid < 3 * HD) {
        const int hh = tid / HD, d = tid % HD;
        st[hh * nrc * 130 + 2 + d] = pv;
        if (d == 0) { st[hh * nrc * 130] = m0; st[hh * nrc * 130 + 1] = l0; }
      }
      }
      __syncthreads();
      const int nrec = 3 * (nrc - 1) * 130;
      for (int i0 = tid; i0 < nrec; i0 += NTH * 8) {
        uint2 v[8];
#pragma unroll
        for (int q = 0; q < 8; q++) {
          const int i = min(i0 + q * NTH, nrec - 1);
          const int hh = i / ((nrc - 1) * 130), rest = i % ((nrc - 1) * 130);
          v[q] = ld_tag(c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH + 1) * 130 + rest);
        }
#pragma unroll
        for (int q = 0; q < 8; q++) {
          const int i = i0 + q * NTH;
          if (i < nrec) {
            const int hh = i / ((nrc - 1) * 130), rest = i % ((nrc - 1) * 130);
            const uint2* p = c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH + 1) * 130 + rest;
            while (v[q].y != tag) v[q] = ld_tag(p);
            st[hh * nrc * 130 + 130 + rest] = __uint_as_float(v[q].x);
          }
        }
      }
      const int ewt = 3 * nrc * 130;
      __syncthreads();
      if (w < 3) {
        const float* r0 = st + w * nrc * 130;
        float M = -1e30f;
        for (int k = lane; k < nrc; k += 32) M = fmaxf(M, r0[k * 130]);
#pragma unroll
        for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o));
        float S = 0.f;
        for (int k = lane; k < nrc; k += 32) {
          const float e = expf(r0[k * 130] - M);
          st[ewt + w * MAXCH + k] = e;
          S += r0[k * 130 + 1] * e;
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) S += __shfl_xor_sync(0xffffffff, S, o);
        if (lane == 0) { sm.bcast[2 * w] = M; sm.bcast[2 * w + 1] = S; }
      }
      __syncthreads();
      if (tid < 3 * HD) {
        const int hh = tid / HD, d = tid % HD;
        const float* r0 = st + hh * nrc * 130;
        float O = 0.f;
        for (int k = 0; k < nrc; k++) O += r0[k * 130 + 2 + d] * st[ewt + hh * MAXCH + k];
        const float S = sm.bcast[2 * hh + 1];
        const float lse = sm.bcast[2 * hh] + logf(S);
        const float out = (O / S) * sigm(gate) * sigm(lse - sink);
        const float o1 = __shfl_down_sync(0xffffffff, out, 1);
        if ((d & 1) == 0) st_tag(c.aoutt + (long)(par * MAXR + b) * (QW / 2) + ((3 * g + hh) * HD + d) / 2, pack_bf2(out, o1), tag);
      }
      SUBMARK(26);
    }
    __syncthreads();
    SUBMARK(27);
  }
}

// Verify passes whose rows all start their window at position 0 (global layers, or every row before position 1023):
// rows b = positions t0 + b of one sequence share their keys. Record items (kv head g, 64-key record ch) load the
// record's K / V once and compute the record of every row b that has it (3 heads each); merge items (row b, kv head g)
// then merge each row's records. Every record, and every merge, is the same arithmetic in the same order as the
// per-row items of ph_attn (scores, softmax and PV columns are independent of the other rows of an mma), so the
// result is bit-identical to step() (tests/c/vcheck.py).
SHARED_ATTR __device__ void ph_attn_shared(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = hd.B, t0 = hd.pos, nq = 3 * B;
  const int tmax = t0 + B - 1;
  const int nrcmax = (tmax + 64) / 64;
#ifndef Q2
#define Q2 1
#endif
#ifndef MG16_UNR
#define MG16_UNR 8   // merge record polls in flight per thread
#endif
#ifndef MRG2
#define MRG2 0   // (measured slower: 8852 vs 8612 us at K=16)
#endif
  const int mslot = 3 * nrcmax * 130 + 3 * MAXCH;   // merge staging per row slot (floats)
  const bool pair = MRG2 && B > SMB && 2 * mslot <= (int)(sizeof(sm.u.mrg) / sizeof(float));
  const int nb2 = pair ? (B + 1) / 2 : B;
#ifndef REC2
#define REC2 0   // (measured slower: 8980 vs 8592 us at K=16, ctx 600)
#endif
  const bool rec2 = REC2 && B > SMB;
  const int NR = NKV * (rec2 ? (nrcmax + 1) / 2 : nrcmax), NM = NKV * nb2;
  const unsigned tag = TAGA(a);
  const int par = a & 1;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
  auto& at = sm.u.as;
  const long cstride = (long)c.maxctx * HD;
  const uint2* qkvp = c.qkvt + (long)(par * MAXR) * QKVR;
  for (int it = blockIdx.x; it < NR + NM; it += gridDim.x) {
    if (it < NR) {
      // ---------------------------------------------------------------- record item (g, ch)
      SUBMARK(60);
      const int g = it % NKV;
      // REC2 (16-row passes): an item computes two records of the kv head in turn (ch, ch - 1); the q / r of every
      // (row, head) once for both; each record's arithmetic as before
      for (int rk = 0; rk < (rec2 ? 2 : 1); rk++) {
      const int ch = rec2 ? nrcmax - 1 - 2 * (it / NKV) - rk : nrcmax - 1 - it / NKV;
      if (ch < 0) break;
      const int j0 = 64 * ch;
      const int b0 = max(0, j0 - t0);                       // rows b >= b0 have this record (t_b >= j0)
      const int jend = min(j0 + 64, tmax + 1), njm = jend - j0;
      const int P0 = min(max(j0, t0), jend), np = jend - P0; // positions >= t0 are rows of this pass
      {
        const bf16* kb = c.kc + (((long)ap.slot * MAXB + 0) * NKV + g) * cstride + (long)j0 * HD;
        const bf16* vb = c.vc + (((long)ap.slot * MAXB + 0) * NKV + g) * cstride + (long)j0 * HD;
        const int nld = (P0 - j0) * 16;
        for (int i = tid; i < nld; i += NTH) {
          const int r = i >> 4, cc = (i & 15) * 8;
          cp_async16(&at.kk[r][cc], reinterpret_cast<const uint4*>(kb) + i);
          cp_async16(&at.vv[r][cc], reinterpret_cast<const uint4*>(vb) + i);
        }
      }
      // relative-bias rows: prw[r][x] = P[r][dist], dist = t0 - j0 - 63 + x (row b, key jj: x = b - jj + 63)
      for (int i = tid; i < 16 * (63 + B); i += NTH) {
        const int r = i / (63 + B), x = i % (63 + B), dist = t0 - j0 - 63 + x;
        at.prw[r][x] = (dist >= 0 && dist < 1024) ? bf(L.relp[r * 1024 + dist]) : 0.f;
      }
      for (int i = tid; i < 3 * MAXR * (64 + 8) / 2; i += NTH) {
        reinterpret_cast<uint32_t*>(&at.ph[0][0])[i] = 0u;
        reinterpret_cast<uint32_t*>(&at.pl[0][0])[i] = 0u;
      }
      if (rk == 0) for (int i = tid; i < (3 * MAXR - nq) * (HD + 8) / 2; i += NTH) reinterpret_cast<uint32_t*>(&at.qb[nq][0])[i] = 0u;
      {
        const int nj16 = (njm + 15) & ~15;
        for (int i = tid; i < (nj16 - njm) * (HD / 2); i += NTH) reinterpret_cast<uint32_t*>(&at.vv[njm + i / (HD / 2)][0])[i % (HD / 2)] = 0u;
      }
      // warp tasks: k / v of the pass's positions in the record, then q and r of every (row, head)
      const bool q2 = Q2 && B > SMB;   // two (row, head) queries per warp task (16 lanes each), the same sums
      const int nqt = q2 ? (nq + 1) / 2 : nq;
      for (int task = w; task < 2 * np + (rk == 0 ? nqt : 0); task += NWARP) {
        if (task < 2 * np) {
          const bool isv = task & 1;
          const int i = task >> 1, p = P0 + i, r = p - t0;
          const bf16* hist = (isv ? c.hv : c.hk) + ((long)ap.slot * MAXB + 0) * RING * 1024;
          float kv[4], ss = 0.f;
#if KV4
          kv_conv4(L, qkvp, r, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, g * HD + lane * 4, g, p, r, kv);
#pragma unroll
          for (int k = 0; k < 4; k++) ss += kv[k] * kv[k];
#else
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const int ch2 = g * HD + lane * 4 + k;
            float xr_;
            kv[k] = kv_conv(L, qkvp, r, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, ch2, g, p, r, &xr_);
            at.raw[isv][i][lane * 4 + k] = tobf(xr_);
            ss += kv[k] * kv[k];
          }
#endif
#pragma unroll
          for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
          const float f = isv ? rsqrtf(ss / 128.f + 1e-6f) : fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
          bf16* dst = isv ? at.vt[i] : at.kt[i];
#pragma unroll
          for (int k = 0; k < 4; k++) dst[lane * 4 + k] = tobf(kv[k] * f);
        } else if (q2) {
          // two queries per warp: lanes 16 x + l hold query 2 (task - 2 np) + x; lane l carries the 4-element sums
          // of the original lanes l and l + 16 (the same order), whose level-16 shuffle add is s_l + s_(l+16);
          // levels 8 .. 1 stay within the half: the clip factors are the original ones bit for bit
          const int half = lane >> 4, l16 = lane & 15;
          const int qi0 = 2 * (task - 2 * np) + half, okq = qi0 < nq, qi = okq ? qi0 : nq - 1;
          const int b = qi / 3, hh = qi % 3, h = 3 * g + hh;
          const uint2* qr = qkvp + (long)b * QKVR;
          const int t = t0 + b;
          const float tau = (L.glob && t + 1 > 8192) ? (1.0f + 0.1f * logf((float)(t + 1) / 8192.0f)) : 1.0f;
          float qv[2][4], sa = 0.f, sb = 0.f;
          uint2 rq[2][4];
#pragma unroll
          for (int u = 0; u < 2; u++)
#pragma unroll
            for (int k = 0; k < 4; k++) rq[u][k] = ld_tag(qr + h * HD + (l16 + 16 * u) * 4 + k);
          const uint2 rrv = ld_tag(qr + R_R + h * 16 + l16);
#pragma unroll
          for (int u = 0; u < 2; u++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
              const int d = (l16 + 16 * u) * 4 + k;
              qv[u][k] = __uint_as_float(wait_tag(qr + h * HD + d, rq[u][k], tag)) * bf(L.qscale[h * HD + d]);
              if (u == 0) sa += qv[u][k] * qv[u][k]; else sb += qv[u][k] * qv[u][k];
            }
          const float rv = __uint_as_float(wait_tag(qr + R_R + h * 16 + l16, rrv, tag)) * bf(L.rscale[h * 16 + l16]);
          float ss = sa + sb, sr = rv * rv;
#pragma unroll
          for (int o = 8; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
#pragma unroll
          for (int o = 8; o; o >>= 1) sr += __shfl_xor_sync(0xffffffff, sr, o);
          const float f = fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
          const float fr = fminf(8.0f / (sqrtf(sr) + 1e-6f), 1.0f);
          if (okq) {
#pragma unroll
            for (int u = 0; u < 2; u++)
#pragma unroll
              for (int k = 0; k < 4; k++) {
                float y = rbf(qv[u][k] * f);
                if (L.glob) y = rbf(y * tau);
                at.qb[qi][(l16 + 16 * u) * 4 + k] = tobf(y);
              }
            float y = rbf(rv * fr);
            if (L.glob) y = rbf(y * tau);
            at.rr[qi][l16] = y;
          }
        } else {
          // q (all lanes, 4 dims each) and r (lanes < 16) of (row b, head hh): loads first, then the waits
          const int qi = task - 2 * np, b = qi / 3, hh = qi % 3, h = 3 * g + hh;
          const uint2* qr = qkvp + (long)b * QKVR;
          const int t = t0 + b;
          const float tau = (L.glob && t + 1 > 8192) ? (1.0f + 0.1f * logf((float)(t + 1) / 8192.0f)) : 1.0f;
          float qv[4], ss = 0.f;
          uint2 rq[4];
#pragma unroll
          for (int k = 0; k < 4; k++) rq[k] = ld_tag(qr + h * HD + lane * 4 + k);
          const uint2 rrv = ld_tag(qr + R_R + h * 16 + (lane & 15));
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const int d = lane * 4 + k;
            qv[k] = __uint_as_float(wait_tag(qr + h * HD + d, rq[k], tag)) * bf(L.qscale[h * HD + d]);
            ss += qv[k] * qv[k];
          }
          float rv = 0.f, sr = 0.f;
          if (lane < 16) { rv = __uint_as_float(wait_tag(qr + R_R + h * 16 + lane, rrv, tag)) * bf(L.rscale[h * 16 + lane]); sr = rv * rv; }
#pragma unroll
          for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
#pragma unroll
          for (int o = 16; o; o >>= 1) sr += __shfl_xor_sync(0xffffffff, sr, o);
          const float f = fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
#pragma unroll
          for (int k = 0; k < 4; k++) {
            float y = rbf(qv[k] * f);
            if (L.glob) y = rbf(y * tau);
            at.qb[qi][lane * 4 + k] = tobf(y);
          }
          const float fr = fminf(8.0f / (sqrtf(sr) + 1e-6f), 1.0f);
          if (lane < 16) {
            float y = rbf(rv * fr);
            if (L.glob) y = rbf(y * tau);
            at.rr[qi][lane] = y;
          }
        }
      }
      SUBMARK(61);
      cp_async_wait();
      __syncthreads();
      SUBMARK(62);
      if (tid < HD) {
        for (int i = 0; i < np; i++) {
          at.kk[P0 - j0 + i][tid] = tobf(at.kt[i][tid]);
          at.vv[P0 - j0 + i][tid] = tobf(at.vt[i][tid]);
          // position P0 + i is row r's current position: the cache and the conv history (as its own step would)
          const int p = P0 + i, r = p - t0;
          const long co = (((long)ap.slot * MAXB + 0) * NKV + g) * cstride + (long)p * HD + tid;
          c.kc[co] = tobf(at.kt[i][tid]);
          c.vc[co] = tobf(at.vt[i][tid]);
          const long hrow = ((long)ap.slot * MAXB + 0) * RING * 1024 + (p & (RING - 1)) * 1024 + g * HD + tid;
          (void)r;
          c.hk[hrow] = tobf(at.raw[0][i][tid]);
          c.hv[hrow] = tobf(at.raw[1][i][tid]);
        }
      }
      // relative bias per (row, head): sc[qi][jj] = bf16(sum_r rr[qi][r] * P[r][t_b - j]) (0 beyond 1023)
      for (int i = tid; i < nq * 64; i += NTH) {
        const int qi = i / 64, jj = i % 64, b = qi / 3, t = t0 + b;
        if (jj < njm && j0 + jj <= t) {
          float bsum = 0.f;
#pragma unroll
          for (int r = 0; r < 16; r++) bsum += at.rr[qi][r] * at.prw[r][b - jj + 63];
          at.sc[qi][jj] = (t - (j0 + jj) < 1024) ? rbf(bsum) : 0.f;
        }
      }
      __syncthreads();
      SUBMARK(63);
      // scores: S^T[key][query] = K[key][:] . q[query][:]; warp (key tile mt, query tile nt)
      {
        const int nmt = (njm + 15) / 16, nnt = (nq + 7) / 8;
        if (w < nmt * nnt) {
          const int mt = w / nnt, nt = w % nnt;
          float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int ks = 0; ks < HD / 16; ks++) {
            uint32_t af[4];
            ldsm_x4(af, &at.kk[16 * mt + (lane & 15)][16 * ks + (lane >> 4) * 8]);
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&at.qb[8 * nt + g8][16 * ks + 2 * t4]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&at.qb[8 * nt + g8][16 * ks + 8 + 2 * t4]);
            mma_bf16(acc, af[0], af[1], af[2], af[3], b0, b1);
          }
#pragma unroll
          for (int q = 0; q < 4; q++) {
            const int key = 16 * mt + g8 + (q >> 1) * 8, qi = 8 * nt + 2 * t4 + (q & 1);
            if (qi < nq && key < njm && j0 + key <= t0 + qi / 3) at.sc[qi][key] = acc[q] * 0.08838834764831845f + at.sc[qi][key];
          }
        }
      }
      __syncthreads();
      SUBMARK(64);
      // softmax of the record per (row, head): keys [j0, min(j0 + 64, t_b + 1))
      for (int qi = w; qi < nq; qi += NWARP) {
        const int b = qi / 3, t = t0 + b;
        if (t >= j0) {
          const int k1 = min(64, t + 1 - j0);
          float m = -1e30f;
          for (int jj = lane; jj < k1; jj += 32) m = fmaxf(m, at.sc[qi][jj]);
#pragma unroll
          for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
          float l = 0.f;
          for (int jj = lane; jj < 64; jj += 32) {
            const float e = jj < k1 ? expf(at.sc[qi][jj] - m) : 0.f;
            const bf16 eh = tobf(e);
            at.ph[qi][jj] = eh;
            at.pl[qi][jj] = tobf(e - bf(eh));
            l += e;
          }
#pragma unroll
          for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffff, l, o);
          if (lane == 0) {
            at.ml[qi][0] = m; at.ml[qi][1] = l;
            uint2* rec = c.arec + ((long)(b * NQH + 3 * g + qi % 3) * MAXCH + ch) * 130;   // the record's (m, l)
            st_tagf(rec, m, tag);
            st_tagf(rec + 1, l, tag);
          }
        }
      }
      __syncthreads();
      SUBMARK(65);
      // PV per (row, head): O[qi][d] = sum_j (Phi + Plo)[qi][j] V[j][d]; warp (query tile mt, dims 16 wd .. +15)
      {
        const int nmt = (nq + 15) / 16;
        if (w < 8 * nmt) {
          const int mt = w >> 3, wd = w & 7;
          float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
          for (int ks = 0; ks < (njm + 15) / 16; ks++) {
            uint32_t ah[4], al[4], bv[4];
            ldsm_x4(ah, &at.ph[16 * mt + (lane & 15)][16 * ks + (lane >> 4) * 8]);
            ldsm_x4(al, &at.pl[16 * mt + (lane & 15)][16 * ks + (lane >> 4) * 8]);
            ldsm_x4_t(bv, &at.vv[16 * ks + (lane & 15)][16 * wd + (lane >> 4) * 8]);
            mma_bf16(acc[0], ah[0], ah[1], ah[2], ah[3], bv[0], bv[1]);
            mma_bf16(acc[0], al[0], al[1], al[2], al[3], bv[0], bv[1]);
            mma_bf16(acc[1], ah[0], ah[1], ah[2], ah[3], bv[2], bv[3]);
            mma_bf16(acc[1], al[0], al[1], al[2], al[3], bv[2], bv[3]);
          }
#pragma unroll
          for (int hr = 0; hr < 2; hr++) {
            const int qi = 16 * mt + g8 + 8 * hr, b = qi / 3, hh = qi % 3;
            if (qi < nq && t0 + b >= j0) {
              uint2* rec = c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH + ch) * 130;
#pragma unroll
              for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int q = 0; q < 2; q++) st_tagf(rec + 2 + 16 * wd + 8 * nt + 2 * t4 + q, acc[nt][2 * hr + q], tag);
            }
          }
        }
      }
      (void)b0;
      SUBMARK(66);
      if (rec2) __syncthreads();   // smem of the record is reused by the next one
      }
    } else {
      // ---------------------------------------------------------------- merge item (b, g)
      // 16-row passes (MRG2): one item merges rows p and p + nb2 (threads 0..383 and 384..767: 3 x 128 each, as
      // the single-row item uses), so that record and merge items fit one round of the grid; per row the same code
      SUBMARK(70);
      const int sub = pair ? (tid >= 3 * HD) : 0, ltid = pair ? tid - sub * 3 * HD : tid, lw = ltid >> 5;
      const int mi = it - NR, bp = mi / NKV, g = mi % NKV;
      const int b = pair ? bp + sub * nb2 : bp;
      const bool rv = b < B;
      const int t = t0 + (rv ? b : 0);
      const int nrc = (t + 64) / 64;
      const uint2* qr = qkvp + (long)(rv ? b : 0) * QKVR;
      float gate = 0.f, sink = 0.f;
      uint2 rg = make_uint2(0u, 0u), rs = make_uint2(0u, 0u);
      if (rv && ltid < 3 * HD) {
        const int hh = ltid / HD, d = ltid % HD, h = 3 * g + hh;
        rg = ld_tag(qr + R_G + h * HD + d); rs = ld_tag(qr + R_S + h);
      }
      float* st = sm.u.mrg + (long)sub * mslot;   // [3][nrc][130]
      const int nrec = rv ? 3 * nrc * 130 : 0;
      const int nth = pair ? 3 * HD : NTH;
      for (int i0 = ltid; i0 < nrec; i0 += nth * MG16_UNR) {
        uint2 v[MG16_UNR];
#pragma unroll
        for (int q = 0; q < MG16_UNR; q++) {
          const int i = min(i0 + q * nth, nrec - 1);
          const int hh = i / (nrc * 130), rest = i % (nrc * 130);
          v[q] = ld_tag(c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH) * 130 + rest);
        }
#pragma unroll
        for (int q = 0; q < MG16_UNR; q++) {
          const int i = i0 + q * nth;
          if (i < nrec) {
            const int hh = i / (nrc * 130), rest = i % (nrc * 130);
            const uint2* p = c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH) * 130 + rest;
            while (v[q].y != tag) v[q] = ld_tag(p);
            st[hh * nrc * 130 + rest] = __uint_as_float(v[q].x);
          }
        }
      }
      if (rv && ltid < 3 * HD) {
        const int hh = ltid / HD, d = ltid % HD, h = 3 * g + hh;
        gate = __uint_as_float(wait_tag(qr + R_G + h * HD + d, rg, tag));
        sink = __uint_as_float(wait_tag(qr + R_S + h, rs, tag));
      }
      const int ewt = 3 * nrc * 130;
      __syncthreads();
      SUBMARK(71);
      if (rv && lw < 3) {
        const float* r0 = st + lw * nrc * 130;
        float M = -1e30f;
        for (int k = lane; k < nrc; k += 32) M = fmaxf(M, r0[k * 130]);
#pragma unroll
        for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o));
        float S = 0.f;
        for (int k = lane; k < nrc; k += 32) {
          const float e = expf(r0[k * 130] - M);
          st[ewt + lw * MAXCH + k] = e;
          S += r0[k * 130 + 1] * e;
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) S += __shfl_xor_sync(0xffffffff, S, o);
        if (lane == 0) { sm.bcast[6 * sub + 2 * lw] = M; sm.bcast[6 * sub + 2 * lw + 1] = S; }
      }
      __syncthreads();
      if (rv && ltid < 3 * HD) {
        const int hh = ltid / HD, d = ltid % HD;
        const float* r0 = st + hh * nrc * 130;
        float O = 0.f;
        for (int k = 0; k < nrc; k++) O += r0[k * 130 + 2 + d] * st[ewt + hh * MAXCH + k];
        const float S = sm.bcast[6 * sub + 2 * hh + 1];
        const float lse = sm.bcast[6 * sub + 2 * hh] + logf(S);
        const float out = (O / S) * sigm(gate) * sigm(lse - sink);
        const float o1 = __shfl_down_sync(0xffffffff, out, 1);
        if ((d & 1) == 0) st_tag(c.aoutt + (long)(par * MAXR + b) * (QW / 2) + ((3 * g + hh) * HD + d) / 2, pack_bf2(out, o1), tag);
      }
      SUBMARK(72);
    }
    __syncthreads();
  }
}

// o_proj + epilogue: attn_raw -> sconv -> attst
// verify passes of > 8 rows: o_proj in halves of 8 rows; the preloaded weights serve both halves; the conv lags of the
// second half's first rows come from the first half's raw values (kept by the epilogue warp)
__device__ void ph_oproj16(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int par = a & 1;
  if (blockIdx.x >= 144) return;
  const int wq = threadIdx.x >> 5, lq = lane_id();
  uint4 wpre[8];
  {
    const uint8_t* wst = L.wo + (long)blockIdx.x * 48 * 2048;
#pragma unroll
    for (int q = 0; q < 8; q++) wpre[q] = ldg_nc(wst + ((long)(2 * wq + (q >> 2)) * 4 + (q & 3)) * 512 + lq * 16);
  }
  float xrp[4] = {0.f, 0.f, 0.f, 0.f};
  for (int ro = 0; ro < hd.B; ro += SMB) {
    const int B = min(SMB, hd.B - ro);
    {
      const int n = B * QW / 2;
      const uint2* src = c.aoutt + ((long)par * MAXR + ro) * (QW / 2);
      for (int i0 = threadIdx.x; i0 < n; i0 += NTH * 4) {
        uint2 r[4];
#pragma unroll
        for (int q = 0; q < 4; q++) r[q] = ld_tag(src + min(i0 + q * NTH, n - 1));
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const int i = i0 + q * NTH;
          if (i < n) {
            const int b = i / (QW / 2), k = (i % (QW / 2)) * 2;
            *reinterpret_cast<uint32_t*>(&sm.u.xo[b][k]) = wait_tag(src + i, r[q], TAGA(a));
          }
        }
      }
    }
    __syncthreads();
    for (int strip = blockIdx.x; strip < 144; strip += gridDim.x) {
      BSm bl{&sm.u.xo[0][0], QWP, B};
      block_strips<NWARP, 1>(sm, 1, 48,
        [&](int, int u0, int nu, float (*acc)[4]) {
#pragma unroll
          for (int q = 0; q < 8; q++) {
            const uint2 b = bl.get((u0 + (q >> 2)) * 64 + (q & 3) * 16);
            mma_bf16(acc[0], wpre[q].x, wpre[q].y, wpre[q].z, wpre[q].w, b.x, b.y);
          }
        },
        [&](int, float (*acc4)[4]) {
        const float* acc = acc4[0];
        const int lane = lane_id(), g = lane >> 2, tt = lane & 3;
        float y[4], xr[4], xin[3][4], xinp[3][4];
#pragma unroll
        for (int q = 0; q < 4; q++) xr[q] = rbf(acc[q]);
#pragma unroll
        for (int l = 1; l <= 3; l++)
#pragma unroll
          for (int q = 0; q < 4; q++) {
            const int cc = q & 1, st = 2 * tt + cc - l, src = max(st, 0) >> 1, srcp = min((SMB + st) >> 1, 3);
            xin[l - 1][q] = __shfl_sync(0xffffffffu, xr[(q & 2) | ((cc - l) & 1)], (lane & ~3) | src);
            xinp[l - 1][q] = __shfl_sync(0xffffffffu, xrp[(q & 2) | ((cc - l) & 1)], (lane & ~3) | srcp);
          }
        for (int h = 0; h < 2; h++)
          for (int cc = 0; cc < 2; cc++) {
            const int tok = 2 * tt + cc;
            y[2 * h + cc] = 0.f;
            if (tok >= B) continue;
            const int row = strip * 16 + g + 8 * h;
            const float x = xr[2 * h + cc];
            const int t = row_t<true>(hd, ro + tok), nin = row_nin<true>(hd, ro + tok), sq = row_seq<true>(hd, ro + tok);
            bf16* hb = c.ha + ((long)ap.slot * MAXB + sq) * RING * H + row;
            float xl[3];
#pragma unroll
            for (int l = 1; l <= 3; l++)
              xl[l - 1] = l <= nin ? (tok - l >= 0 ? xin[l - 1][2 * h + cc] : xinp[l - 1][2 * h + cc]) : ldcg_bf(hb + ((t - l) & (RING - 1)) * H);
            y[2 * h + cc] = sconv_static(x, xl[0], xl[1], xl[2], L.attn_sc + row * 4);
            hb[(t & (RING - 1)) * H] = tobf(x);
          }
#pragma unroll
        for (int q = 0; q < 4; q++) xrp[q] = xr[q];
        float yo[4];
#pragma unroll
        for (int q = 0; q < 4; q++) yo[q] = __shfl_down_sync(0xffffffff, y[q], 4);
        if ((g & 1) == 0)
          for (int h = 0; h < 2; h++)
            for (int cc = 0; cc < 2; cc++) {
              const int tok = 2 * tt + cc;
              if (tok >= B) continue;
              const int row = strip * 16 + g + 8 * h;
              st_tag(c.attst + (long)(par * MAXR + ro + tok) * NPAIR + row / 2, pack_bf2(y[2 * h + cc], yo[2 * h + cc]), TAGA(a));
            }
        });
    }
    __syncthreads();
  }
}
template <bool V> __device__ void ph_oproj(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = hd.B, par = a & 1;
  if (blockIdx.x >= 144) return;
  // this warp's two units of the block's strip, loaded before waiting for the attention output (same order as
  // piece_bf16 over units 2w, 2w + 1, and block_strips' warp-order sum: bit-identical to the generic path)
  const int wq = threadIdx.x >> 5, lq = lane_id();
#ifdef W12
  const W12Pre wp0 = w12_load(L.wo12, blockIdx.x, 48, 2 * wq), wp1 = w12_load(L.wo12, blockIdx.x, 48, 2 * wq + 1);
  uint32_t oE0, oE8;
  w12_bases(L.wo12, blockIdx.x, oE0, oE8);
  (void)lq;
#else
  uint4 wpre[8];
  {
    const uint8_t* wst = L.wo + (long)blockIdx.x * 48 * 2048;
#pragma unroll
    for (int q = 0; q < 8; q++) wpre[q] = ldg_nc(wst + ((long)(2 * wq + (q >> 2)) * 4 + (q & 3)) * 512 + lq * 16);
  }
#endif
  {
    const int n = B * QW / 2;
    const uint2* src = c.aoutt + (long)par * MAXR * (QW / 2);
    for (int i0 = threadIdx.x; i0 < n; i0 += NTH * 4) {
      uint2 r[4];
#pragma unroll
      for (int q = 0; q < 4; q++) r[q] = ld_tag(src + min(i0 + q * NTH, n - 1));
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int i = i0 + q * NTH;
        if (i < n) {
          const int b = i / (QW / 2), k = (i % (QW / 2)) * 2;
          *reinterpret_cast<uint32_t*>(&sm.u.xo[b][k]) = wait_tag(src + i, r[q], TAGA(a));
        }
      }
    }
  }
  __syncthreads();
  // one block per output strip (144), 24 warps x 2 units (preloaded)
  for (int strip = blockIdx.x; strip < 144; strip += gridDim.x) {
    BSm bl{&sm.u.xo[0][0], QWP, B};
    block_strips<NWARP, 1>(sm, 1, 48,
      [&](int, int u0, int nu, float (*acc)[4]) {
#ifdef W12
        w12_mma(L.wo12, wp0, oE0, oE8, u0, bl, acc[0]);
        w12_mma(L.wo12, wp1, oE0, oE8, u0 + 1, bl, acc[0]);
#else
#pragma unroll
        for (int q = 0; q < 8; q++) {
          const uint2 b = bl.get((u0 + (q >> 2)) * 64 + (q & 3) * 16);
          mma_bf16(acc[0], wpre[q].x, wpre[q].y, wpre[q].z, wpre[q].w, b.x, b.y);
        }
#endif
      },
      [&](int, float (*acc4)[4]) {
      const float* acc = acc4[0];
      const int lane = lane_id(), g = lane >> 2, tt = lane & 3;
      float y[4], xr[4], xin[3][4];
#pragma unroll
      for (int q = 0; q < 4; q++) xr[q] = rbf(acc[q]);
      // raw values of the rows tok - l (verify passes: the same sequence's previous positions): lane (g, (tok - l) / 2)
#pragma unroll
      for (int l = 1; l <= 3; l++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const int cc = q & 1, src = max(2 * tt + cc - l, 0) >> 1;
          xin[l - 1][q] = __shfl_sync(0xffffffffu, xr[(q & 2) | ((cc - l) & 1)], (lane & ~3) | src);
        }
      for (int h = 0; h < 2; h++)
        for (int cc = 0; cc < 2; cc++) {
          const int tok = 2 * tt + cc;
          y[2 * h + cc] = 0.f;
          if (tok >= B) continue;
          const int row = strip * 16 + g + 8 * h;
          const float x = xr[2 * h + cc];
          const int t = row_t<V>(hd, tok), nin = row_nin<V>(hd, tok), seq = row_seq<V>(hd, tok);
          bf16* hb = c.ha + ((long)ap.slot * MAXB + seq) * RING * H + row;
          float xl[3];
#pragma unroll
          for (int l = 1; l <= 3; l++) xl[l - 1] = l <= nin ? xin[l - 1][2 * h + cc] : ldcg_bf(hb + ((t - l) & (RING - 1)) * H);
          y[2 * h + cc] = sconv_static(x, xl[0], xl[1], xl[2], L.attn_sc + row * 4);
          hb[(t & (RING - 1)) * H] = tobf(x);
        }
      // rows (g, g + 1) are lanes (l, l + 4): pack pairs on even g
      float yo[4];
#pragma unroll
      for (int q = 0; q < 4; q++) yo[q] = __shfl_down_sync(0xffffffff, y[q], 4);
      if ((g & 1) == 0)
        for (int h = 0; h < 2; h++)
          for (int cc = 0; cc < 2; cc++) {
            const int tok = 2 * tt + cc;
            if (tok >= B) continue;
            const int row = strip * 16 + g + 8 * h;
            st_tag(c.attst + (long)(par * MAXR + tok) * NPAIR + row / 2, pack_bf2(y[2 * h + cc], yo[2 * h + cc]), TAGA(a));
          }
      });
  }
}

// mlp mix of application a, token b -> x (f32). Sources: blocks in slots 0..3, the boundary layer's old partial in
// slot 4, the partial (partial + attention output in f32, or the attention output alone) in slot 5.
__device__ void mlp_mix(PH_ARGS, int a, int b, float (*x)[EPT]) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  float v[6][EPT];
  const bool add_pin = !(a == 0 || ap.bnd);
  const int idx[6] = {a == 0 ? 0 : ap.blk[0], ap.blk[1], ap.blk[2], ap.blk[3], ap.pin, ap.pin};
  unsigned msk = (a == 0) ? 1u : (((1u << ap.nblk) - 1u) | (ap.bnd ? 16u : 0u));
  const float* qs[1] = {L.mq};
  float qv[1][EPT];
  load_queries<1>(qs, 1, qv);
  load_srcs<6>(c, hd, idx, msk | (add_pin ? 32u : 0u), b, v);
  float as[EPT];
  load_vec_bf(c.attst + (long)((a & 1) * MAXR + b) * NPAIR, TAGA(a), as);
#pragma unroll
  for (int k = 0; k < EPT; k++) v[5][k] = add_pin ? v[5][k] + as[k] : as[k];
  depth_mix_q<6, 1>(v, msk | 32u, qv, 1, x, sm);
}

// (gated) mlp input of token b (f32) -> router input xr (bf16) + expert input x16 (f16 hi / lo) in smem
__device__ void mlp_in_x(PH_ARGS, int a, float (*xin)[EPT], int b) {
  const LayerW& L = c.L[c.A[a].layer];
  float wv[EPT];
  load_w(L.mlp_norm, wv);
  float s[1] = {0.f}, mx[1] = {0.f};
#pragma unroll
  for (int k = 0; k < EPT; k++) { s[0] += xin[0][k] * xin[0][k]; mx[0] = fmaxf(mx[0], fabsf(xin[0][k] * wv[k])); }
  block_sum_max<1, 1>(s, mx, sm);
  const float r = rsqrtf(s[0] / (float)H + 1e-6f);
  // scale: max |y| <= bf16 rounding of r * max |x w|: y * 2^sh stays below 2^15 (f16 exact for the hi part)
  int e2 = 0;
  const float bound = mx[0] * r;
  frexpf(bound, &e2);                    // bound in [2^(e2-1), 2^e2)
  const int sh = (bound > 0.f) ? 15 - e2 : 0;
  const float scl = ldexpf(1.0f, sh);
#pragma unroll
  for (int j = 0; j < 2; j++) {
    if (!EV(2 * j)) continue;
    const int el = EL(2 * j);
    const uint32_t yb = pack_bf2(xin[0][2 * j] * r * wv[2 * j], xin[0][2 * j + 1] * r * wv[2 * j + 1]);
    *reinterpret_cast<uint32_t*>((EXR(sm, b) + el)) = yb;
    const float y0 = bf_lo(yb) * scl, y1 = bf_hi(yb) * scl;
    const __half h0 = __float2half_rn(y0), h1 = __float2half_rn(y1);
    sm.u.e.x16[b][0][el] = h0; sm.u.e.x16[b][0][el + 1] = h1;
    sm.u.e.x16[b][1][el] = __float2half_rn((y0 - __half2float(h0)) * 16777216.0f);
    sm.u.e.x16[b][1][el + 1] = __float2half_rn((y1 - __half2float(h1)) * 16777216.0f);
  }
  if (threadIdx.x == 0) sm.sc16[b] = ldexpf(1.0f, -sh);
}

// B >= 2: the designated block of token b publishes the router input row (bf16 pairs) and the f16 scale exponent
__device__ void mlp_publish(PH_ARGS, int a, float (*xin)[EPT], int b) {
  const LayerW& L = c.L[c.A[a].layer];
  const int par = a & 1;
  float wv[EPT];
  load_w(L.mlp_norm, wv);
  float s[1] = {0.f}, mx[1] = {0.f};
#pragma unroll
  for (int k = 0; k < EPT; k++) { s[0] += xin[0][k] * xin[0][k]; mx[0] = fmaxf(mx[0], fabsf(xin[0][k] * wv[k])); }
  block_sum_max<1, 1>(s, mx, sm);
  const float r = rsqrtf(s[0] / (float)H + 1e-6f);
  int e2 = 0;
  const float bound = mx[0] * r;
  frexpf(bound, &e2);
  const int sh = (bound > 0.f) ? 15 - e2 : 0;
  uint2* dst = c.xrt + (long)(par * MAXR + b) * NPAIR;
#pragma unroll
  for (int j = 0; j < 2; j++)
    if (EV(2 * j)) st_tag(dst + threadIdx.x + j * NTH, pack_bf2(xin[0][2 * j] * r * wv[2 * j], xin[0][2 * j + 1] * r * wv[2 * j + 1]), TAGA(a));
  if (threadIdx.x == 0) st_tag(c.sht + par * MAXR + b, (uint32_t)sh, TAGA(a));
}
// every block: router input rows + the expert input (f16 hi / lo, the same arithmetic as mlp_in_x)
__device__ void mlp_poll(PH_ARGS, int a, int r0 = 0) {
  const int B = hd.B, par = a & 1;
  poll_rows(c.xrt + ((long)par * MAXR + r0) * NPAIR, B - r0, [&](int r) { return EXR(sm, r0 + r); }, TAGA(a));
  if (threadIdx.x >= r0 && threadIdx.x < B) {
    const int sh = (int)poll(c.sht + par * MAXR + threadIdx.x, TAGA(a));
    sm.bcast[32 + threadIdx.x] = (float)sh;
    sm.sc16[threadIdx.x] = ldexpf(1.0f, -sh);
  }
  __syncthreads();
  for (int b = r0; b < B; b++) {
    const float scl = ldexpf(1.0f, (int)sm.bcast[32 + b]);
#pragma unroll
    for (int j = 0; j < 2; j++) {
      if (!EV(2 * j)) continue;
      const int el = EL(2 * j);
      const uint32_t yb = *reinterpret_cast<const uint32_t*>((EXR(sm, b) + el));
      const float y0 = bf_lo(yb) * scl, y1 = bf_hi(yb) * scl;
      const __half h0 = __float2half_rn(y0), h1 = __float2half_rn(y1);
      sm.u.e.x16[b][0][el] = h0; sm.u.e.x16[b][0][el + 1] = h1;
      sm.u.e.x16[b][1][el] = __float2half_rn((y0 - __half2float(h0)) * 16777216.0f);
      sm.u.e.x16[b][1][el + 1] = __float2half_rn((y1 - __half2float(h1)) * 16777216.0f);
    }
  }
  __syncthreads();
}
// verify passes of > 8 rows: rows [hd.ro, hd.ro + hd.B) of the published router inputs into smem slots 0 .. hd.B - 1
// (the same arithmetic as mlp_poll per row)
#ifndef MOE1
#define MOE1 1   // 16-row passes: one MoE over all rows (router per 8-row half, then one routing / w13 / w2a), not per half
#endif
__device__ void mlp_conv_half(PH_ARGS, int a);
__device__ void mlp_poll_half(PH_ARGS, int a, bool conv = true) {
  const int B = hd.B, par = a & 1, ro = hd.ro;
  poll_rows(c.xrt + ((long)par * MAXR + ro) * NPAIR, B, [&](int r) { return EXR(sm, r); }, TAGA(a));
  if (threadIdx.x < B) {
    const int sh = (int)poll(c.sht + par * MAXR + ro + threadIdx.x, TAGA(a));
    sm.bcast[32 + threadIdx.x] = (float)sh;
    sm.sc16[MOE1 ? ro + threadIdx.x : threadIdx.x] = ldexpf(1.0f, -sh);
  }
  __syncthreads();
  if (conv) mlp_conv_half(c, hd, sm, a);
}
// router input rows (EXR) -> expert input rows ro + b (one-pass MoE) or b, f16 hi / lo with the row's 2^sh scaling
__device__ void mlp_conv_half(PH_ARGS, int a) {
  const int B = hd.B, ro = hd.ro;
  for (int b = 0; b < B; b++) {
    const float scl = ldexpf(1.0f, (int)sm.bcast[32 + b]);
    const int xb = MOE1 ? ro + b : b;   // one-pass MoE: the expert inputs of all 16 rows stay in smem
#pragma unroll
    for (int j = 0; j < 2; j++) {
      if (!EV(2 * j)) continue;
      const int el = EL(2 * j);
      const uint32_t yb = *reinterpret_cast<const uint32_t*>((EXR(sm, b) + el));
      const float y0 = bf_lo(yb) * scl, y1 = bf_hi(yb) * scl;
      const __half h0 = __float2half_rn(y0), h1 = __float2half_rn(y1);
      sm.u.e.x16[xb][0][el] = h0; sm.u.e.x16[xb][0][el + 1] = h1;
      sm.u.e.x16[xb][1][el] = __float2half_rn((y0 - __half2float(h0)) * 16777216.0f);
      sm.u.e.x16[xb][1][el + 1] = __float2half_rn((y1 - __half2float(h1)) * 16777216.0f);
    }
  }
  __syncthreads();
}

#ifndef RTR2
#define RTR2 1
#endif
#ifndef ROUT1
#define ROUT1 0   // (measured slower: 8635 vs 8603 us; the per-half polls pipeline) 16-row passes: one poll of the 16 router-input rows, one router GEMV (two column groups), then x16
#endif
// 16-row passes: the router inputs of all rows -> smem (EXR16), the router logits of the 16 rows (blocks 0..7), then
// the expert inputs x16[b] (in place, rows in order; every element by the same thread: see EXR16)
__device__ void moe_in16(PH_ARGS, int a) {
  const LayerW& L = c.L[c.A[a].layer];
  const int B = hd.B, par = a & 1;
  poll_rows(c.xrt + ((long)par * MAXR) * NPAIR, B, [&](int r) { return EXR16(sm, r); }, TAGA(a));
  if (threadIdx.x < B) {
    const int sh = (int)poll(c.sht + par * MAXR + threadIdx.x, TAGA(a));
    sm.bcast[32 + threadIdx.x] = (float)sh;
    sm.sc16[threadIdx.x] = ldexpf(1.0f, -sh);
  }
  __syncthreads();
  for (int strip = blockIdx.x; strip < 8; strip += gridDim.x) {
    BSm b0{EXR16(sm, 0), HP, SMB}, b1{EXR16(sm, SMB), HP, B - SMB};
    block_strips<NWARP, 2>(sm, 1, 36,
      [&](int, int u0, int nu, float (*acc)[4]) { piece_bf16_2(L.wr + (long)strip * 36 * 2048, u0, nu, b0, b1, acc); },
      [&](int, float (*acc)[4]) {
        const int lane = lane_id(), g = lane >> 2, t = lane & 3;
        for (int ng = 0; ng < 2; ng++)
          for (int h = 0; h < 2; h++)
            for (int cc = 0; cc < 2; cc++) {
              const int tok = SMB * ng + 2 * t + cc;
              if (tok < B) st_tagf(c.rlogt + (par * MAXR + tok) * NE + strip * 16 + g + 8 * h, acc[ng][2 * h + cc], TAGA(a));
            }
      });
  }
  __syncthreads();
  for (int b = 0; b < B; b++) {
    const float scl = ldexpf(1.0f, (int)sm.bcast[32 + b]);
#pragma unroll
    for (int j = 0; j < 2; j++) {
      if (!EV(2 * j)) continue;
      const int el = EL(2 * j);
      const uint32_t yb = *reinterpret_cast<const uint32_t*>(EXR16(sm, b) + el);
      const float y0 = bf_lo(yb) * scl, y1 = bf_hi(yb) * scl;
      const __half h0 = __float2half_rn(y0), h1 = __float2half_rn(y1);
      sm.u.e.x16[b][0][el] = h0; sm.u.e.x16[b][0][el + 1] = h1;
      sm.u.e.x16[b][1][el] = __float2half_rn((y0 - __half2float(h0)) * 16777216.0f);
      sm.u.e.x16[b][1][el + 1] = __float2half_rn((y1 - __half2float(h1)) * 16777216.0f);
    }
  }
  __syncthreads();
}

template <bool V> __device__ int ph_mlp_in(PH_ARGS, int a) {
  const App& ap = c.A[a];
  if (ap.nmsg) return -1;   // the gate's D blocks compute the mix themselves
  if (hyb2<V>(hd)) {
    for (int b = 1 + (int)(gridDim.x - 1 - blockIdx.x); b < hd.B; b += gridDim.x) {
      float x[1][EPT];
      mlp_mix(c, hd, sm, a, b, x);
      mlp_publish(c, hd, sm, a, x, b);
    }
    {
      float x[1][EPT];
      mlp_mix(c, hd, sm, a, 0, x);
      mlp_in_x(c, hd, sm, a, x, 0);
    }
    mlp_poll(c, hd, sm, a, 1);
    return 0;
  }
  if (desig<V>(hd)) {
    for (int b = dsel<V>(hd); b < hd.B; b += gridDim.x) {
      float x[1][EPT];
      mlp_mix(c, hd, sm, a, b, x);
      mlp_publish(c, hd, sm, a, x, b);
    }
    if (big16<V>(hd)) return 0;   // rows polled per half before the router
    mlp_poll(c, hd, sm, a);
    return 0;
  }
  for (int b = 0; b < hd.B; b++) {
    float x[1][EPT];
    mlp_mix(c, hd, sm, a, b, x);
    mlp_in_x(c, hd, sm, a, x, b);
  }
  __syncthreads();
  return 0;
}

template <bool V> __device__ void ph_router(PH_ARGS, int a, int boff = 0) {
  const int ro = V ? hd.ro : 0;
  const LayerW& L = c.L[c.A[a].layer];
  const int B = hd.B, par = a & 1;
  for (int strip = (int)blockIdx.x - boff; strip >= 0 && strip < 8; strip += gridDim.x) {   // boff: blocks boff .. boff + 7
    BSm bl{EXR(sm, 0), HP, B};
    block_strips<NWARP, 1>(sm, 1, 36,
      [&](int, int u0, int nu, float (*acc)[4]) { piece_bf16(L.wr + (long)strip * 36 * 2048, u0, nu, bl, acc[0]); },
      [&](int, float (*acc)[4]) {
        const int lane = lane_id(), g = lane >> 2, t = lane & 3;
        for (int h = 0; h < 2; h++)
          for (int cc = 0; cc < 2; cc++) {
            const int tok = 2 * t + cc;
            if (tok < B) st_tagf(c.rlogt + (par * MAXR + ro + tok) * NE + strip * 16 + g + 8 * h, acc[0][2 * h + cc], TAGA(a));
          }
      });
  }
}

// top-8 + union (every block, identical)
template <bool V> __device__ void routing(const Ctx& c, const Hdr& hd, Smem& sm, int a) {
  const int ro = V ? hd.ro : 0;
  const LayerW& L = c.L[c.A[a].layer];
  const int B = hd.B;
  const int w = threadIdx.x >> 5, lane = lane_id();
  if (threadIdx.x < 4) sm.umask[threadIdx.x] = 0u;
  if (B > MAXB && threadIdx.x < NE) sm.uc0[threadIdx.x] = 0;   // 16-row passes: rows choosing expert e (bit mask)
  __syncthreads();
  if (w < B) {
    float lg[4], sel[4];
    uint2 rl[4];
#pragma unroll
    for (int k = 0; k < 4; k++) rl[k] = ld_tag(c.rlogt + ((a & 1) * MAXR + ro + w) * NE + lane + 32 * k);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const int e = lane + 32 * k;
      lg[k] = __uint_as_float(wait_tag(c.rlogt + ((a & 1) * MAXR + ro + w) * NE + e, rl[k], TAGA(a)));
      sel[k] = sigm(lg[k]) + L.rbias[e];
      if (!(sel[k] == sel[k])) { sel[k] = -1e30f; atomicOr(c.bar + 2, 1u << (a & 31)); }   // NaN: never chosen, flagged
    }
    unsigned taken = 0;
    int myid = 0; float mylg = 0.f;
    // order-preserving integer keys of the selection scores (finite: NaN was replaced above)
    unsigned key[4];
#pragma unroll
    for (int k = 0; k < 4; k++) { const unsigned u = __float_as_uint(sel[k]); key[k] = (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
    for (int r = 0; r < TOPK; r++) {
      // this lane's best untaken entry (ties: the lower index), then the warp's largest key and, among equal keys,
      // the lowest index: the same choice as a comparison tree with (value, lower index) ordering
      unsigned bk = 0u; int bi_l = 0x7fffffff; float bl_l = 0.f;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const bool ok = !(taken & (1u << k));
        if (ok && (key[k] > bk || bi_l == 0x7fffffff)) { bk = key[k]; bi_l = lane + 32 * k; bl_l = lg[k]; }
      }
      const unsigned mk = __reduce_max_sync(0xffffffffu, bk);
      int bi = (int)__reduce_min_sync(0xffffffffu, (bk == mk && bi_l != 0x7fffffff) ? (unsigned)bi_l : 0x7fffffffu);
      bi = bi < NE ? bi : 0;   // cannot happen; keeps every index in range
      const float bl = __shfl_sync(0xffffffffu, bl_l, bi & 31);
      if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
      if (lane == r) { myid = bi; mylg = bl; }
    }
    // lanes 0..7 hold (id, logit) in rank order: weights sigmoid / sum * 8 (sum in rank order)
    const float sg = sigm(mylg);
    float den = 0.f;
#pragma unroll
    for (int r = 0; r < TOPK; r++) den += __shfl_sync(0xffffffff, sg, r);
    const float wt = sg / den * 8.0f;
    int rank = 0;
#pragma unroll
    for (int r = 0; r < TOPK; r++) rank += (__shfl_sync(0xffffffff, myid, r) < myid);
    if (lane < TOPK) {
      sm.tex[w][rank] = myid; sm.twt[w][rank] = wt;
      atomicOr(&sm.umask[myid >> 5], 1u << (myid & 31));
      if (B > MAXB) atomicOr(&sm.uc0[myid], 1 << w);
    }
  }
  __syncthreads();
  if (w == 0) {
    int n = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const bool used = (sm.umask[k] >> lane) & 1u;
      const unsigned bal = __ballot_sync(0xffffffff, used);
      if (used) sm.uexp[n + __popc(bal & ((1u << lane) - 1u))] = 32 * k + lane;
      n += __popc(bal);
    }
    if (lane == 0) sm.nU = n;
    __syncwarp();
#ifdef MKPROF
    if (blockIdx.x == 0) SUBMARK(1000 + n);
#endif
    if (B <= MAXB) {
      for (int u = lane; u < n; u += 32) {
        const int e = sm.uexp[u];
        int cnt = 0;
        for (int b = 0; b < B; b++)
          for (int r = 0; r < TOPK; r++)
            if (sm.tex[b][r] == e) sm.utok[u][cnt++] = b;
        sm.uncnt[u] = cnt;
      }
    } else {
      // 16-row passes: an expert chosen by more than 8 rows takes two entries (rows in order: the first 8, the rest),
      // so that every entry has <= 8 tokens (the w13 / w2a item code); lane 0 builds the list in expert order
      // 16-row passes: an expert chosen by more than 8 rows takes two entries (rows in order: the first 8, the rest),
      // so that every entry has <= 8 tokens (the w13 / w2a item code). Warp 0: counts per expert, a prefix sum of the
      // entries (1 or 2 per expert, in expert order), then each lane fills its experts' entries.
      for (int u = lane; u < n; u += 32) sm.ue0[u] = sm.uexp[u];
      __syncwarp();
      int base = 0;
      for (int u0 = 0; u0 < n; u0 += 32) {
        const int u = u0 + lane;
        const unsigned msk = u < n ? (unsigned)sm.uc0[sm.ue0[u]] : 0u;
        const int ne = u < n ? (__popc(msk) > MAXB ? 2 : 1) : 0;
        int incl = ne;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += y; }
        if (u < n) {
          int mm = base + incl - ne, cnt = 0;
          const int e = sm.ue0[u];
          for (unsigned mb = msk; mb; mb &= mb - 1) {   // rows in order
            if (cnt == MAXB) { sm.uexp[mm] = e; sm.uncnt[mm] = cnt; mm++; cnt = 0; }
            sm.utok[mm][cnt++] = __ffs(mb) - 1;
          }
          sm.uexp[mm] = e; sm.uncnt[mm] = cnt;
        }
        base += __shfl_sync(0xffffffffu, incl, 31);
      }
      if (lane == 0) sm.nU = base;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ float act_fn(float g, float u) {
  float a = 4.0f * tanhf(g / 4.0f) * sigm(g);
  return a * (25.0f * tanhf(u / 25.0f));
}

// w13: work items (union slot u, strip of 16 rows = 8 (gate, up) pairs), round robin over blocks, 8 per round,
// 3 warps x 12 units each; the epilogue computes act * routing weight -> act16t (f16 hi / lo, x16)
#ifndef W13_ROLL
#define W13_ROLL 0   // (measured slower at 4 and 3: +430 / +320 us at K=2) verify passes: w13 NVFP4 chains as rolling batches of this many units (0: piece_fp4)
#endif
template <int NG, bool ROLL = false>
__device__ __forceinline__ void w13_run(const Ctx& c, Smem& sm, const LayerW& L, int u, int strip, int u0, int nu, float (*acc)[4]) {
  const int g = lane_id() >> 2;
  const int e = sm.uexp[u], nt = sm.uncnt[u];
  const __half* colp[3];
#pragma unroll
  for (int ng = 0; ng < 2; ng++) {
    const int li = 4 * ng + (g >> 1), part = g & 1;
    colp[ng] = li < nt ? &sm.u.e.x16[sm.utok[u][li]][part][0] : nullptr;
  }
  colp[2] = nullptr;
  if (!L.mx) {
    if (ROLL && W13_ROLL > 0 && nu % W13_ROLL == 0) piece_fp4_roll<NG, (W13_ROLL > 0 ? W13_ROLL : 1)>(L.w13 + ((long)e * 144 + strip) * 36 * 576, u0, nu, colp, acc);
    else piece_fp4<NG, false, W13_UNR16>(L.w13 + ((long)e * 144 + strip) * 36 * 576, u0, nu, colp, acc);
  } else piece_fp8<NG, false>(L.w13 + ((long)e * 144 + strip) * 36 * 1056, u0, nu, colp, acc);
}
template <bool V = false>
__device__ void w13_epi(const Ctx& c, const Hdr& hd, Smem& sm, const LayerW& L, int a, int u, int strip, int NGN, float (*acc)[4]) {
  const int lane = lane_id(), g = lane >> 2, t = lane & 3, par = (a & 1) + (V ? 2 * (hd.ro >> 3) : 0);   // act16t set (half)
  const int e = sm.uexp[u], nt = sm.uncnt[u];
  const float fac = L.mx ? 1.0f : L.f13[e];
  for (int ng = 0; ng < NGN; ng++) {
    const float a0 = acc[ng][0], a1 = acc[ng][1], a2 = acc[ng][2], a3 = acc[ng][3];
    // lane (g, t): rows g, g+8 of token li = 4 ng + t: (hi, lo) = (a0, a1) row g; (a2, a3) row g+8
    const int li = 4 * ng + t;
    const bool ok = li < nt;
    const int tok = ok ? sm.utok[u][li] : 0;
    const float s16 = sm.sc16[tok];
    const float h0 = rbf((a0 + a1 * 5.9604644775390625e-08f) * s16 * fac);
    const float h1 = rbf((a2 + a3 * 5.9604644775390625e-08f) * s16 * fac);
    // pair i: rows 2i (gate), 2i+1 (up); partner row g ^ 1 is lane ^ 4
    const float o0 = __shfl_xor_sync(0xffffffff, h0, 4), o1 = __shfl_xor_sync(0xffffffff, h1, 4);
    float wgt = 0.f;
    if (ok) for (int r = 0; r < TOPK; r++) if (sm.tex[tok][r] == e) wgt = sm.twt[tok][r];
    const float act0 = rbf(act_fn(h0, o0) * wgt), act1 = rbf(act_fn(h1, o1) * wgt);
    const float v0 = act0 * 16.0f, v1 = act1 * 16.0f;
    const __half hi0 = __float2half_rn(v0), hi1 = __float2half_rn(v1);
    const __half lo0 = __float2half_rn((v0 - __half2float(hi0)) * 16777216.0f), lo1 = __float2half_rn((v1 - __half2float(hi1)) * 16777216.0f);
    const uint32_t q0 = __half_as_ushort(hi0), q1 = __half_as_ushort(lo0), q2 = __half_as_ushort(hi1), q3 = __half_as_ushort(lo1);
    // pairs i0 = strip*8 + g/2 (g even) and i0 + 1 (g + 2): lanes l and l + 8
    const uint32_t p0 = __shfl_down_sync(0xffffffff, q0, 8), p1 = __shfl_down_sync(0xffffffff, q1, 8);
    const uint32_t p2 = __shfl_down_sync(0xffffffff, q2, 8), p3 = __shfl_down_sync(0xffffffff, q3, 8);
    if (ok && (g & 3) == 0) {
      uint2* dst = c.act16t + (((long)(par * MAXU + u) * MAXB + li) * 2) * (IE / 2);
      const int i0 = strip * 8 + (g >> 1), i1 = strip * 8 + 4 + (g >> 1);
      st_tag(dst + i0 / 2, q0 | (p0 << 16), TAGA(a));
      st_tag(dst + IE / 2 + i0 / 2, q1 | (p1 << 16), TAGA(a));
      st_tag(dst + i1 / 2, q2 | (p2 << 16), TAGA(a));
      st_tag(dst + IE / 2 + i1 / 2, q3 | (p3 << 16), TAGA(a));
    }
  }
}
#ifndef W13_ONEWARP
#define W13_ONEWARP 0   // measured slower (w13 work 25.5 vs 19.4 us at 8 rows): 6 serial batches per warp
#endif
template <int NG>
__device__ __forceinline__ void w13_item_v(const Ctx& c, const Hdr& hd, Smem& sm, const LayerW& L, int a, int u, int strip) {
  const int g = lane_id() >> 2;
  const int e = sm.uexp[u], nt = sm.uncnt[u];
  const __half* colp[3];
#pragma unroll
  for (int ng = 0; ng < 2; ng++) {
    const int li = 4 * ng + (g >> 1), part = g & 1;
    colp[ng] = li < nt ? &sm.u.e.x16[sm.utok[u][li]][part][0] : nullptr;
  }
  colp[2] = nullptr;
  float acc[3][NG][4];
#pragma unroll
  for (int k = 0; k < 3; k++)
#pragma unroll
    for (int ng = 0; ng < NG; ng++) acc[k][ng][0] = acc[k][ng][1] = acc[k][ng][2] = acc[k][ng][3] = 0.f;
  if (!L.mx) piece3c<NG, false, 12, 2>(L.w13 + ((long)e * 144 + strip) * 36 * 576, colp, acc);
  else piece3c<NG, true, 12, 1>(L.w13 + ((long)e * 144 + strip) * 36 * 1056, colp, acc);
  float tot[2][4];
  sum3<NG>(acc, tot);
  w13_epi(c, hd, sm, L, a, u, strip, NG, tot);
}
// verify passes, NVFP4 layers: the w13 weights of each warp's chains stream through a per-warp ring of W13_D units in
// shared memory (cp.async, continuous: the next units load while this one is decoded), the same items per group of
// 3 warps, the same chains (units [12 wl, 12 wl + 12) of each strip), the same group reduction and epilogue as
// block_strips<3, NG, GRP>: bit-identical
#ifndef W13_ASYNC
#define W13_ASYNC 0   // measured SLOWER (K=2 +560 us, K=8 +1.27 ms): per-unit wait/sync overhead, smaller L1
#endif
#define W13_D 7
#define W13_RING (W13_D * 576)
__device__ void w13_async(const Ctx& c, const Hdr& hd, Smem& sm, const LayerW& L, int a) {
  const int nU = sm.nU, T = nU * 144, G = gridDim.x;
  const int w = threadIdx.x >> 5, lane = lane_id(), q = w / 3, wl = w - 3 * q, g = lane >> 2;
  const int first = blockIdx.x + G * q, step = 8 * G;
  const int nit = first < T ? (T - first + step - 1) / step : 0;
  const int N = nit * 12;
  unsigned char* ring = w < 9 ? reinterpret_cast<unsigned char*>(EXR(sm, 0)) + w * W13_RING
                              : reinterpret_cast<unsigned char*>(&sm) + sizeof(Smem) + (w - 9) * W13_RING;
  auto issue = [&](int n) {
    const int it = first + step * (n / 12), u = it / 144, strip = it - 144 * u;
    const uint8_t* p = L.w13 + ((long)sm.uexp[u] * 144 + strip) * 36 * 576 + (long)(12 * wl + n % 12) * 576;
    unsigned char* d = ring + (n % W13_D) * 576;
    cp_async16(d + lane * 16, p + lane * 16);
    if (lane < 4) cp_async16(d + 512 + lane * 16, p + 512 + lane * 16);
  };
#pragma unroll 1
  for (int n = 0; n < W13_D; n++) { if (n < N) issue(n); cp_async_commit(); }
  float acc[2][4];
#pragma unroll
  for (int ng = 0; ng < 2; ng++) acc[ng][0] = acc[ng][1] = acc[ng][2] = acc[ng][3] = 0.f;
#pragma unroll 1
  for (int n = 0; n < N; n++) {
    cp_async_wait_group<W13_D - 1>();
    __syncwarp();
    const int it = first + step * (n / 12), u = it / 144, strip = it - 144 * u, j = n % 12;
    const int nt = sm.uncnt[u];
    const unsigned char* sl = ring + (n % W13_D) * 576;
    const uint4 cw = *reinterpret_cast<const uint4*>(sl + lane * 16);
    const uint2 sw = *reinterpret_cast<const uint2*>(sl + 512 + g * 8);
    __syncwarp();
    if (n + W13_D < N) issue(n + W13_D);
    cp_async_commit();
    const __half* colp[3];
#pragma unroll
    for (int ng = 0; ng < 2; ng++) {
      const int li = 4 * ng + (g >> 1), part = g & 1;
      colp[ng] = li < nt ? &sm.u.e.x16[sm.utok[u][li]][part][0] : nullptr;
    }
    colp[2] = nullptr;
    const int uu = 12 * wl + j;
    if (nt > 4) { uint2 bfr[2][4]; load_b16_unit<2>(colp, uu, bfr); unit_fp4<2>(cw, sw, bfr, acc); }
    else { uint2 bfr[1][4]; load_b16_unit<1>(colp, uu, bfr); unit_fp4<1>(cw, sw, bfr, acc); }
    if (j == 11) {
      const int NGN = nt > 4 ? 2 : 1;
      float tot[2][4];
      for (int ng = 0; ng < NGN; ng++) {
        sm.red4[w][lane] = make_float4(acc[ng][0], acc[ng][1], acc[ng][2], acc[ng][3]);
        grp_sync(1 + q, 96);
        tot[ng][0] = tot[ng][1] = tot[ng][2] = tot[ng][3] = 0.f;
        if (wl == 0) {
#pragma unroll
          for (int k = 0; k < 3; k++) {
            const float4 v = sm.red4[w + k][lane];
            tot[ng][0] += v.x; tot[ng][1] += v.y; tot[ng][2] += v.z; tot[ng][3] += v.w;
          }
        }
        grp_sync(1 + q, 96);
      }
      if (wl == 0) w13_epi(c, hd, sm, L, a, u, strip, NGN, tot);
#pragma unroll
      for (int ng = 0; ng < 2; ng++) acc[ng][0] = acc[ng][1] = acc[ng][2] = acc[ng][3] = 0.f;
    }
  }
  cp_async_wait();
}
template <bool V> __device__ void ph_w13(PH_ARGS, int a) {
  const LayerW& L = c.L[c.A[a].layer];
  const int nU = sm.nU;
  if (V && W13_ASYNC && !L.mx) { w13_async(c, hd, sm, L, a); return; }
  if (V && W13_ONEWARP) {   // verify passes: one warp per item (item it on block it % G, warp it / G), whole strip
    const int T = nU * 144, G = gridDim.x, w = threadIdx.x >> 5;
    for (int it = blockIdx.x + G * w; it < T; it += G * NWARP) {
      if (sm.uncnt[it / 144] > 4) w13_item_v<2>(c, hd, sm, L, a, it / 144, it % 144);
      else w13_item_v<1>(c, hd, sm, L, a, it / 144, it % 144);
    }
    return;
  }

  const int T = nU * 144, G = gridDim.x;
  constexpr int WG = 3, PER = NWARP / WG;
  const int mine = (T - (int)blockIdx.x + G - 1) / G;      // items blockIdx.x + k G
#ifdef MKPROF
  const bool w13m = V && (threadIdx.x & 31) == 0 && (threadIdx.x >> 5) == 0;   // warp 0 marks its rounds
#endif
  for (int r0 = 0; r0 < mine; r0 += PER) {
#ifdef MKPROF
    if (w13m && c.prof) { unsigned n_ = atomicAdd(&sm.subn, 1u); if (n_ < 1024) { unsigned long long* r_ = (unsigned long long*)c.prof + 2ull * ((unsigned long long)blockIdx.x * 4096 + 3072 + n_); r_[0] = mkprof_gtime(); r_[1] = ((unsigned long long)clock64() << 16) | (90 + r0 / PER); } }
#endif
    const int ns = min(PER, mine - r0);
    // one n-group unless an expert of this round has more than 4 tokens
    bool two = false;
    for (int q = 0; q < ns; q++) two |= sm.uncnt[(blockIdx.x + (r0 + q) * G) / 144] > 4;
    auto run1 = [&](int q, int u0, int nu, float (*acc)[4]) { const int it = blockIdx.x + (r0 + q) * G; w13_run<1, V>(c, sm, L, it / 144, it % 144, u0, nu, acc); };
    auto run2 = [&](int q, int u0, int nu, float (*acc)[4]) {
      const int it = blockIdx.x + (r0 + q) * G;
      if (sm.uncnt[it / 144] > 4) w13_run<2, V>(c, sm, L, it / 144, it % 144, u0, nu, acc);
      else w13_run<1, V>(c, sm, L, it / 144, it % 144, u0, nu, acc);
    };
    auto epi = [&](int q, float (*acc)[4]) { const int it = blockIdx.x + (r0 + q) * G; w13_epi<V>(c, hd, sm, L, a, it / 144, it % 144, sm.uncnt[it / 144] > 4 ? 2 : 1, acc); };
    if (!two) block_strips<WG, 1, GRP_EXP>(sm, ns, 36, run1, epi);
    else block_strips<WG, 2, GRP_EXP>(sm, ns, 36, run2, epi);
  }
#ifdef MKPROF
  if (w13m && c.prof) { unsigned n_ = atomicAdd(&sm.subn, 1u); if (n_ < 1024) { unsigned long long* r_ = (unsigned long long*)c.prof + 2ull * ((unsigned long long)blockIdx.x * 4096 + 3072 + n_); r_[0] = mkprof_gtime(); r_[1] = ((unsigned long long)clock64() << 16) | 99; } }
#endif
}

// w2, part 1: work items (union expert u, output strip) expert-major in block-contiguous ranges (at most two experts
// per block, their inputs staged in shared memory once); 8 items per round, 3 warps x 6 units each (a fixed split).
// Each item publishes its contribution per token as tagged f32 rows w2ct[par][token][rank of the expert][row].
#ifndef W2_BPC16
#define W2_BPC16 2   // w2a NVFP4 chains: units loaded per batch
#endif
template <bool V, bool M> __device__ void ph_w2a(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int par = a & 1;
  const int nU = sm.nU;
  constexpr int WG = 3, PER = NWARP / WG;
  const int lane = lane_id();
  const int T = nU * 144;
  int ib = (int)((long)T * blockIdx.x / gridDim.x), ie = (int)((long)T * (blockIdx.x + 1) / gridDim.x);
#ifndef W2_BAL
#define W2_BAL 1
#endif
  bool big = false;
#ifndef W2_STEP
#define W2_STEP 1   // 1: the step kernel for B >= 2 (mk_step<false, true>) uses the one-warp w2a items and the work balance
#endif
#ifndef W2V16
#define W2V16 0   // s18-c4 (V(16) -1.5%): 0 = the step's 3-warp w2a items (faster in the 8-row kernel once the bank conflicts were fixed)
#endif
  const bool w2v = (V && W2V16) || (W2_STEP && M);
  if (w2v && W2_BAL) for (int u = 0; u < nU; u++) big |= sm.uncnt[u] > 4;
  if (w2v && W2_BAL && big && (!V || hd.B <= MAXB || [&] { long C2 = 0; for (int u = 0; u < nU; u++) C2 += 144L * (sm.uncnt[u] > 4 ? 3 : 2); return C2 <= 288L * gridDim.x; }())) {
    // verify passes: block ranges of equal estimated work (an item of an expert with more than 4 tokens runs two
    // n-groups: cost 3 vs 2), mapped back to item indices; the same assignment in every block (routing tables)
    long C = 0;
    for (int u = 0; u < nU; u++) C += 144L * (sm.uncnt[u] > 4 ? 3 : 2);
    const long lo = C * blockIdx.x / gridDim.x, hi = C * (blockIdx.x + 1) / gridDim.x;
    // first item whose cost start is >= x
    auto first_at = [&](long x) {
      long P = 0;
      for (int u = 0; u < nU; u++) {
        const int cu = sm.uncnt[u] > 4 ? 3 : 2;
        if (x <= P + 143L * cu) { const long s2 = (x - P + cu - 1) / cu; return (int)(144L * u + s2); }
        P += 144L * cu;
      }
      return T;
    };
    ib = first_at(lo); ie = blockIdx.x + 1 == gridDim.x ? T : first_at(hi);
  }
  if (ib >= ie) return;
  SUBMARK(40);
  const int u0 = ib / 144, u1 = (ie - 1) / 144;
  const int n0 = sm.uncnt[u0];
  SUBMARK(200 + (L.mx ? 100 : 0) + 10 * max(n0, u1 > u0 ? sm.uncnt[u1] : 0) + (u1 > u0 ? 1 : 0));
  {
    const int np = n0 + (u1 > u0 ? sm.uncnt[u1] : 0);
    const int nw = np * 2 * (IE / 2);   // tagged words
#ifndef W2S_UNR_V
#define W2S_UNR_V 2   // (2: -28 us at K=16 vs 8) w2a input staging: tagged loads in flight per thread (verify passes)
#endif
    constexpr int SB = V ? W2S_UNR_V : 8;
    for (int i0 = 2 * threadIdx.x; i0 < nw; i0 += 2 * NTH * SB) {
      uint2 r[SB][2]; const uint2* pp[SB];
#pragma unroll
      for (int q = 0; q < SB; q++) {
        const int i = min(i0 + q * 2 * NTH, nw - 2);
        const int j = i / IE, w0 = i % IE;
        const int uu = j < n0 ? u0 : u1, jj = j < n0 ? j : j - n0;
        pp[q] = c.act16t + (((long)((par + (V ? 2 * (hd.ro >> 3) : 0)) * MAXU + uu) * MAXB + jj) * 2) * (IE / 2) + w0;
        ld_tag2(pp[q], r[q][0], r[q][1]);
      }
#pragma unroll
      for (int q = 0; q < SB; q++) {
        const int i = i0 + q * 2 * NTH;
        if (i >= nw) continue;
        const int jw = i / IE, ww = i % IE;   // s18-c4: padded rows (no 8-way bank conflicts)
        uint32_t* d = reinterpret_cast<uint32_t*>(&sm.u.w2s[0][0][0]) + (jw * 2 + ww / (IE / 2)) * (W2S_RS / 2) + ww % (IE / 2);
        d[0] = wait_tag(pp[q], r[q][0], TAGA(a));
        d[1] = wait_tag(pp[q] + 1, r[q][1], TAGA(a));
      }
    }
  }
  __syncthreads();
  SUBMARK(41);
  const __half* w2sb = &sm.u.w2s[0][0][0];
  auto epi = [&](int it, float (*acc)[4]) {
    const int u = it / 144, strip = it % 144;
    const int e = sm.uexp[u], nt = sm.uncnt[u];
    const int g = lane >> 2, tt = lane & 3;
    const float fac = (L.mx ? 1.0f : L.f2[e]) * 0.0625f;
    for (int ng = 0; ng < (nt > 4 ? 2 : 1); ng++) {
      const int li = 4 * ng + tt;
      if (li >= nt) continue;
      const int tok = sm.utok[u][li];
      int k = 0;
      for (int r = 0; r < TOPK; r++) if (sm.tex[tok][r] == e) k = r;
      uint2* dst = c.w2ct + (((long)(par * MAXR + (V ? hd.ro : 0) + tok) * TOPK + k) * H) + strip * 16;
      st_tagf(dst + g, (acc[ng][0] + acc[ng][1] * 5.9604644775390625e-08f) * fac, TAGA(a));
      st_tagf(dst + g + 8, (acc[ng][2] + acc[ng][3] * 5.9604644775390625e-08f) * fac, TAGA(a));
    }
  };
#ifndef W2_1W_MIN
#define W2_1W_MIN 0   // one warp per item only from this many items per block (12 / 16 / 24 measured slower at every K >= 2)
#endif
  if (w2v && ie - ib >= W2_1W_MIN) {   // verify passes: one warp per item of the block's range, the whole strip as three chains
    for (int it = ib + (threadIdx.x >> 5); it < ie; it += NWARP) {
      const int u = it / 144, strip = it % 144;
      const int e = sm.uexp[u], nt = sm.uncnt[u], j0 = (u == u0) ? 0 : n0;
      const int g = lane >> 2;
      const __half* colp[3];
#pragma unroll
      for (int ng = 0; ng < 2; ng++) {
        const int li = 4 * ng + (g >> 1), part = g & 1;
        colp[ng] = li < nt ? w2sb + ((long)(j0 + li) * 2 + part) * W2S_RS : nullptr;
      }
      colp[2] = nullptr;
      float tot[2][4];
      if (nt > 4) {
        float acc[3][2][4] = {};
        if (!L.mx) piece3c<2, false, 6, W2_BPC16>(L.w2 + ((long)e * 144 + strip) * 18 * 576, colp, acc);
        else piece3c<2, true, 6, 1>(L.w2 + ((long)e * 144 + strip) * 18 * 1056, colp, acc);
        sum3<2>(acc, tot);
      } else {
        float acc[3][1][4] = {};
        if (!L.mx) piece3c<1, false, 6, W2_BPC16>(L.w2 + ((long)e * 144 + strip) * 18 * 576, colp, acc);
        else piece3c<1, true, 6, 1>(L.w2 + ((long)e * 144 + strip) * 18 * 1056, colp, acc);
        sum3<1>(acc, tot);
      }
      epi(it, tot);
    }
    SUBMARK(43);
    return;
  }
  for (int r0 = ib; r0 < ie; r0 += PER) {
    const int ns = min(PER, ie - r0);
    auto run = [&](int q, int uu0, int nu, float (*acc)[4]) {
      const int it = r0 + q, u = it / 144, strip = it % 144;
      const int e = sm.uexp[u], nt = sm.uncnt[u], j0 = (u == u0) ? 0 : n0;
      const int g = lane >> 2;
      const __half* colp[3];
#pragma unroll
      for (int ng = 0; ng < 2; ng++) {
        const int li = 4 * ng + (g >> 1), part = g & 1;
        colp[ng] = li < nt ? w2sb + ((long)(j0 + li) * 2 + part) * W2S_RS : nullptr;
      }
      colp[2] = nullptr;
      if (nt > 4) {
        if (!L.mx) piece_fp4<2, false>(L.w2 + ((long)e * 144 + strip) * 18 * 576, uu0, nu, colp, acc);
        else piece_fp8<2, false>(L.w2 + ((long)e * 144 + strip) * 18 * 1056, uu0, nu, colp, acc);
      } else {
        if (!L.mx) piece_fp4<1, false>(L.w2 + ((long)e * 144 + strip) * 18 * 576, uu0, nu, colp, acc);
        else piece_fp8<1, false>(L.w2 + ((long)e * 144 + strip) * 18 * 1056, uu0, nu, colp, acc);
      }
    };
    auto epiq = [&](int q, float (*acc)[4]) { epi(r0 + q, acc); };
    bool two = false;
    for (int q = 0; q < ns; q++) two |= sm.uncnt[(r0 + q) / 144] > 4;
    if (!two) block_strips<WG, 1, GRP_EXP>(sm, ns, 18, run, epiq);
    else block_strips<WG, 2, GRP_EXP>(sm, ns, 18, run, epiq);
    SUBMARK(42);
  }
}

// w2, part 2: one block per output strip: per token the 8 expert rows summed in expert-id order, the short
// convolution and the new partial (pbt)
template <bool V> __device__ void ph_w2b(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = hd.B, par = a & 1;
  for (int strip = blockIdx.x; strip < 144; strip += gridDim.x) {
    const bool addp = !(a == 0 || ap.bnd);
    const int et = threadIdx.x >> 4, er = threadIdx.x & 15;    // token et, row er
    const int row = strip * 16 + er;
    float np_ = 0.f;
    const int t = row_t<V>(hd, min(et, MAXR - 1)), nin = row_nin<V>(hd, min(et, MAXR - 1)), seq = row_seq<V>(hd, min(et, MAXR - 1));
    bf16* hb = c.hm + ((long)ap.slot * MAXB + seq) * RING * H + row;
    float x = 0.f, xl[3] = {0.f, 0.f, 0.f};
    uint2 ra = make_uint2(0u, 0u), rp = make_uint2(0u, 0u), tw = make_uint2(0u, 0u);
    if (et < B) {
      // independent loads first: history, taps, attention output, old partial; then the expert rows
#pragma unroll
      for (int l = 1; l <= 3; l++) if (l > nin) xl[l - 1] = ldcg_bf(hb + ((t - l) & (RING - 1)) * H);
      tw = *reinterpret_cast<const uint2*>(L.mlp_sc + row * 4);
      ra = ld_tag(c.attst + (long)(par * MAXR + et) * NPAIR + row / 2);
      rp = ld_tag(c.pbt + ((long)(addp ? ap.pin : 0) * MAXR + et) * NPAIR + row / 2);
      uint2 rk[TOPK];
      const uint2* src = c.w2ct + ((long)(par * MAXR + et) * TOPK) * H + row;
#pragma unroll
      for (int k = 0; k < TOPK; k++) rk[k] = ld_tag(src + (long)k * H);
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; k++) s += __uint_as_float(wait_tag(src + (long)k * H, rk[k], TAGA(a)));
      x = rbf(s);
    }
    if (V) {   // verify pass: the previous positions of this pass are the rows et - l of this block
      float (*cx)[16] = reinterpret_cast<float (*)[16]>(&sm.red[0][0]);   // [MAXB][16] (smem size kept: L1 carve-out)
      __syncthreads();
      if (et < B) cx[et][er] = x;
      __syncthreads();
#pragma unroll
      for (int l = 1; l <= 3; l++) if (et < B && l <= nin) xl[l - 1] = cx[et - l][er];
    }
    if (et < B) {
      const bf16* w4 = reinterpret_cast<const bf16*>(&tw);
      const float x1 = xl[0], x2 = xl[1], x3 = xl[2];
      float acc = __fmul_rn(x3, bf(w4[0]));
      acc = __fadd_rn(acc, __fmul_rn(x2, bf(w4[1])));
      acc = __fadd_rn(acc, __fmul_rn(x1, bf(w4[2])));
      acc = __fadd_rn(acc, __fmul_rn(x, bf(w4[3])));
      const float ms = rbf(__fmul_rn(__fadd_rn(acc, x), 0.44721359549995793f));
      hb[(t & (RING - 1)) * H] = tobf(x);
      const uint32_t av = wait_tag(c.attst + (long)(par * MAXR + et) * NPAIR + row / 2, ra, TAGA(a));
      const float as = (row & 1) ? bf_hi(av) : bf_lo(av);
      if (!addp) np_ = as + ms;
      else {
        const uint32_t pv = wait_tag(c.pbt + ((long)ap.pin * MAXR + et) * NPAIR + row / 2, rp, TAGP(ap.pin));
        np_ = (((row & 1) ? bf_hi(pv) : bf_lo(pv)) + as) + ms;
      }
    }
    const float other = __shfl_down_sync(0xffffffff, np_, 1);
    if (et < B && (er & 1) == 0) st_tag(c.pbt + ((long)ap.out * MAXR + et) * NPAIR + row / 2, pack_bf2(np_, other), TAGP(ap.out));
  }
}

// final: output mix -> norm -> unembed -> logits, argmax
#ifndef FIN_DES
#define FIN_DES 3   // verify passes of at least this many rows: designated blocks mix the final rows and publish them
#endif
template <bool V> __device__ void ph_final(PH_ARGS) {
  const int B = hd.B;
  if (V && B >= FIN_DES) {
    for (int b = dsel<V>(hd); b < B; b += gridDim.x) {
      float v[5][EPT], x[1][EPT], wv[1][EPT];
      const int idx[5] = {c.fin[0], c.fin[1], c.fin[2], c.fin[3], c.fin_pin};
      const unsigned msk = ((1u << c.nfin) - 1u) | 16u;
      load_srcs<5>(c, hd, idx, msk, b, v);
      const float* qs[1] = {c.oq};
      depth_mix<5, 1>(v, msk, qs, 1, x, sm);
      load_w(c.fnorm, wv[0]);
      uint2* dst[1] = {c.xft + (long)b * NPAIR};
      norm_publish<1>(x, wv, 1, dst, TAGP(60), sm);   // the same arithmetic as rmsnorm_store
    }
    poll_rows(c.xft, B, [&](int r) { return &sm.u.xs[0][r][0]; }, TAGP(60));
  } else
  for (int b = 0; b < B; b++) {
    float v[5][EPT], x[1][EPT];
    const int idx[5] = {c.fin[0], c.fin[1], c.fin[2], c.fin[3], c.fin_pin};
    const unsigned msk = ((1u << c.nfin) - 1u) | 16u;
    load_srcs<5>(c, hd, idx, msk, b, v);
    const float* qs[1] = {c.oq};
    depth_mix<5, 1>(v, msk, qs, 1, x, sm);
    rmsnorm_store(x[0], c.fnorm, &sm.u.xs[0][b][0], sm);
  }
  __syncthreads();
  if (V && blockIdx.x == 0)   // verify pass: the final-norm rows are a draft feature
    for (int i = threadIdx.x; i < B * (H / 8); i += NTH)
      reinterpret_cast<uint4*>(c.fout)[i] = *reinterpret_cast<const uint4*>(&sm.u.xs[0][i / (H / 8)][(i % (H / 8)) * 8]);
  const int w = threadIdx.x >> 5, lane = lane_id(), g = lane >> 2, t = lane & 3;
#ifndef FIN1
#define FIN1 1   // 16-row passes: the unembedding in one pass (two column groups per weight fragment: weights read once)
#endif
#ifndef W12
  if (V && FIN1 && B > SMB) {
    unsigned long long best[2][2] = {{0ull, 0ull}, {0ull, 0ull}};
    for (int it = blockIdx.x + gridDim.x * w; it < UN_STRIPS; it += gridDim.x * NWARP) {
      BSm b0{&sm.u.xs[0][0][0], HP, SMB}, b1{&sm.u.xs[0][0][0] + (long)SMB * HP, HP, B - SMB};
      float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      piece_bf16_2(c.unembed + (long)it * 36 * 2048, 0, 36, b0, b1, acc);
      for (int ng = 0; ng < 2; ng++)
        for (int h = 0; h < 2; h++) {
          const int row = it * 16 + g + 8 * h;
          for (int cc = 0; cc < 2; cc++) {
            const int tok = SMB * ng + 2 * t + cc;
            if (tok >= B || row >= VOCAB_REAL) continue;
            const float z = 30.0f * tanhf(acc[ng][2 * h + cc] / 30.0f);
            if (hd.logits) hd.logits[(long)tok * VOCAB_REAL + row] = z;
            unsigned u = __float_as_uint(z);
            u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            const unsigned long long key = ((unsigned long long)u << 32) | (unsigned)(0xFFFFFFFFu - row);
            if (key > best[ng][cc]) best[ng][cc] = key;
          }
        }
    }
    for (int ng = 0; ng < 2; ng++)
      for (int cc = 0; cc < 2; cc++) {
        unsigned long long k = best[ng][cc];
  #pragma unroll
        for (int o = 4; o < 32; o <<= 1) { const unsigned long long ok = __shfl_xor_sync(0xffffffff, k, o); if (ok > k) k = ok; }
        const int tok = SMB * ng + 2 * t + cc;
        if (g == 0 && tok < B && k) atomicMax(c.amax + tok, k);
      }
    return;
  }
#endif
  // verify passes of > 8 rows: the unembedding per half (the 16 final rows lie contiguously in the xs region)
  for (int ro = 0; ro < (V ? B : 1); ro += SMB) {
    const int Bh = V ? min(SMB, B - ro) : B;
    unsigned long long best[2] = {0ull, 0ull};
    for (int it = blockIdx.x + gridDim.x * w; it < UN_STRIPS; it += gridDim.x * NWARP) {
      BSm bl{&sm.u.xs[0][0][0] + (long)ro * HP, HP, Bh};
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
  #ifdef W12
      piece_w12(c.un12, it, 36, 0, 36, bl, acc);
  #else
      piece_bf16(c.unembed + (long)it * 36 * 2048, 0, 36, bl, acc);
  #endif
      for (int h = 0; h < 2; h++) {
        const int row = it * 16 + g + 8 * h;
        for (int cc = 0; cc < 2; cc++) {
          const int tok = 2 * t + cc;
          if (tok >= Bh || row >= VOCAB_REAL) continue;
          const float z = 30.0f * tanhf(acc[2 * h + cc] / 30.0f);
          if (hd.logits) hd.logits[(long)(ro + tok) * VOCAB_REAL + row] = z;
          unsigned u = __float_as_uint(z);
          u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
          const unsigned long long key = ((unsigned long long)u << 32) | (unsigned)(0xFFFFFFFFu - row);
          if (key > best[cc]) best[cc] = key;
        }
      }
    }
    for (int cc = 0; cc < 2; cc++) {
      unsigned long long k = best[cc];
  #pragma unroll
      for (int o = 4; o < 32; o <<= 1) { const unsigned long long ok = __shfl_xor_sync(0xffffffff, k, o); if (ok > k) k = ok; }
      const int tok = 2 * t + cc;
      if (g == 0 && tok < Bh && k) atomicMax(c.amax + ro + tok, k);
    }
  }
}

// ------------------------------------------------------------------------------------------------ kernel
// V: verify pass (rows = positions of one sequence); M: the step kernel for B >= 2 (the B = 1 kernel keeps its code)
template <bool V, bool M = false> __global__ void __launch_bounds__(NTH, 1) mk_step(const Ctx* __restrict__ cp, Hdr hd0) {
  extern __shared__ __align__(16) unsigned char smraw[];
  Smem& sm = *reinterpret_cast<Smem*>(smraw);
  const Ctx& c = *cp;
  unsigned nbar = 0;
  const unsigned base = c.bar[1];
  MKPROF_DECL(pn);
  MkProf P{(unsigned long long*)c.prof, 4096};
#define MARK(tag) do { if (c.prof && threadIdx.x == 0) MKPROF_MARK(P, pn, tag); } while (0)
#ifdef USE_BAR
#define GBAR() do { MARK(MKPROF_WAIT); grid_bar(c, nbar, base); } while (0)
#else
#define GBAR() do { __syncthreads(); } while (0)
#endif
  // track C: both verify kernels are queued for every cycle; flags 0x100 = run only if the device row count vpos[1] > 8
  // (the 16-row library), 0x200 = run only if vpos[1] <= 8: the other one returns at once (block-uniform, before any
  // barrier or tagged store)
  if (V && (hd0.flags & 0x300)) { const int kb_ = hd0.vpos[1]; if (((hd0.flags & 0x100) && kb_ <= 8) || ((hd0.flags & 0x200) && kb_ > 8)) return; }
#ifndef KB16
#define KB16 0
#endif
  if (KB16 && V && hd0.vpos[1] != MAXR) return;   // (never: 16-row passes have exactly 16 rows)
  const int nst = hd0.nsteps > 0 ? hd0.nsteps : 1;
  for (int st = 0; st < nst; st++) {
  Hdr hd = hd0;
  if (V) {   // verify pass: position and number of rows from the device (written by the draft of the previous cycle)
    hd.pos = hd0.vpos[0];
    const int kb = hd0.vpos[1];
    hd.B = (kb >= 1 && kb <= MAXR) ? kb : hd0.B;
#if KB16
    hd.B = MAXR;   // s18-c4: compiled for exactly 16 rows (the n-gram rule proposes 15 drafts; other counts return below)
#endif
    if (HD_MT(hd0)) { const int nr = hd0.mt[0]; hd.B = (nr >= 1 && nr <= MAXR) ? nr : 1; hd.pos = hd0.mt[MT_POS]; }
  }
  if (hd0.nsteps > 0) {   // device loop: step st at position pos + st, its own tag range, token from the prompt or fed back
    hd.pos = hd0.pos + st;
    hd.tag0 = (hd0.tag0 + 64u * st) & 0x7fffffffu;
    hd.tokens = st < hd0.plen ? hd0.prompt + st * hd0.B : nullptr;
    hd.tok_tag = (hd0.tag0 + 64u * (st - 1) + 60u) & 0x7fffffffu;
    hd.out_tok = (hd0.out_tok && st >= hd0.plen - 1) ? hd0.out_tok + (long)(st - (hd0.plen - 1)) * MAXB : nullptr;
    hd.logits = nullptr;
  }
  MARK(MKPROF_STEP);
#ifdef MKPROF
  if (threadIdx.x == 0) sm.subn = 0;
#endif
  for (int a = 0; a < NAPP; a++) {
    const App& ap = c.A[a];
    MARK(1);
    int ns = ph_attn_in<V>(c, hd, sm, a);
    if (ns < 0) {
      GBAR(); MARK(7);
      ph_gate_d<V>(c, hd, sm, a, 0, 3);
      GBAR(); MARK(8);
      ph_gate_fg(c, hd, sm, a, 0, 3);
      GBAR(); MARK(9);
      const LayerW& L = c.L[ap.layer];
      float wv[3][EPT];
      for (int q = 0; q < 3; q++) load_w(L.norm[L.glob ? q : 0], wv[q]);
      const bool bc = desig<V>(hd), hy = hyb2<V>(hd);
      auto row = [&](int b, bool pub) {
        float x[3][EPT];
        uint2 r[3][EPT];
        const uint2* p0 = c.xpt + ((long)((a & 1) * MAXR + b) * 4) * H;
#pragma unroll
        for (int q = 0; q < 3; q++)
#pragma unroll
          for (int k = 0; k < EPT; k++) r[q][k] = ld_tag(p0 + q * H + (EV(k) ? EL(k) : EL(k & 1)));
#pragma unroll
        for (int q = 0; q < 3; q++)
#pragma unroll
          for (int k = 0; k < EPT; k++) x[q][k] = EV(k) ? __uint_as_float(wait_tag(p0 + q * H + EL(k), r[q][k], TAGA(a))) : 0.f;
        if (pub) {
          uint2* xs_pub = c.xst + (long)(a & 1) * 3 * MAXR * NPAIR;
          uint2* dst[3] = {xs_pub + (long)b * NPAIR, xs_pub + (long)(MAXR + b) * NPAIR, xs_pub + (long)(2 * MAXR + b) * NPAIR};
          norm_publish<3>(x, wv, 3, dst, TAGA(a), sm);
        } else {
          bf16* dst[3] = {&sm.u.xs[0][b][0], &sm.u.xs[1][b][0], &sm.u.xs[2][b][0]};
          rmsnorm_store_n<3>(x, wv, 3, dst, sm);
        }
      };
      if (bc) { for (int b = dsel<V>(hd); b < hd.B; b += gridDim.x) row(b, true); }
      else if (hy) { for (int b = 1 + (int)(gridDim.x - 1 - blockIdx.x); b < hd.B; b += gridDim.x) row(b, true); row(0, false); }
      else { for (int b = 0; b < hd.B; b++) row(b, false); }
      if ((bc || hy) && !big16<V>(hd)) {
        const int r0 = hy ? 1 : 0;
        const uint2* xs_pub = c.xst + (long)(a & 1) * 3 * MAXR * NPAIR;
        const unsigned need = ((V || POLL_STEP) && POLL_NEEDED) ? qkv_streams(L, 3) : 7u;
        for (int q = 0; q < 3; q++)
          if ((need >> q) & 1u) poll_rows(xs_pub + ((long)q * MAXR + r0) * NPAIR, hd.B - r0, [&](int rr) { return &sm.u.xs[q][r0 + rr][0]; }, TAGA(a));
      }
      __syncthreads();
      ns = 3;
    }
    MARK(10);
    if (big16<V>(hd)) {
      // rows in halves of 8: the published normalized rows of the half -> smem -> the qkv strips (rows ro + b)
      const LayerW& L = c.L[ap.layer];
      const uint2* xs_pub = c.xst + (long)(a & 1) * 3 * MAXR * NPAIR;
      const unsigned need = POLL_NEEDED ? qkv_streams(L, ns) : 7u;
#ifndef W12
      // the streams of this block's strips: stream 0 and at most one other (148 blocks, <= 4 strips each), else halves
      int other = -1; bool one = QKV1;
      {
        const int strips = QG_STRIPS + 2 * L.nsk, G = gridDim.x;
        for (int strip = blockIdx.x; strip < strips; strip += G) {
          const int st = qkv_stream_of(strip, L.nsk, ns);
          if (st != 0) { if (other >= 0 && other != st) one = false; other = st; }
        }
      }
      if (one) {
        poll_rows(xs_pub + (long)0 * NPAIR, hd.B, [&](int rr) { return &sm.u.xs16[0][rr][0]; }, TAGA(a));
        if (other > 0) poll_rows(xs_pub + ((long)other * MAXR) * NPAIR, hd.B, [&](int rr) { return &sm.u.xs16[1][rr][0]; }, TAGA(a));
        __syncthreads();
        ph_qkv16(c, hd, sm, a, ns, other);
        __syncthreads();
      } else
#endif
      for (int h = 0; h * SMB < hd.B; h++) {
        Hdr hh = hd; hh.ro = h * SMB; hh.B = min(SMB, hd.B - hh.ro);
        for (int q = 0; q < ns; q++)
          if ((need >> q) & 1u) poll_rows(xs_pub + ((long)q * MAXR + hh.ro) * NPAIR, hh.B, [&](int rr) { return &sm.u.xs[q][rr][0]; }, TAGA(a));
        __syncthreads();
        ph_qkv<V>(c, hh, sm, a, ns);
        __syncthreads();
      }
    } else
    ph_qkv<V>(c, hd, sm, a, ns);
    GBAR(); MARK(2);
    ph_attn<V>(c, hd, sm, a);
    GBAR(); MARK(3);
    if (big16<V>(hd)) ph_oproj16(c, hd, sm, a); else ph_oproj<V>(c, hd, sm, a);
    GBAR(); MARK(4);
    int r = ph_mlp_in<V>(c, hd, sm, a);
    if (r < 0) {
      GBAR(); MARK(14);
      ph_gate_d<V>(c, hd, sm, a, 3, 1);
      GBAR(); MARK(15);
      ph_gate_fg(c, hd, sm, a, 3, 1);
      GBAR(); MARK(16);
      if (hyb2<V>(hd)) {
        for (int b = 1 + (int)(gridDim.x - 1 - blockIdx.x); b < hd.B; b += gridDim.x) {
          float x[1][EPT];
          load_vec_f(c.xpt + ((long)((a & 1) * MAXR + b) * 4 + 3) * H, TAGA(a), x[0]);
          mlp_publish(c, hd, sm, a, x, b);
        }
        {
          float x[1][EPT];
          load_vec_f(c.xpt + ((long)((a & 1) * MAXR + 0) * 4 + 3) * H, TAGA(a), x[0]);
          mlp_in_x(c, hd, sm, a, x, 0);
        }
        mlp_poll(c, hd, sm, a, 1);
      } else if (desig<V>(hd)) {
        for (int b = dsel<V>(hd); b < hd.B; b += gridDim.x) {
          float x[1][EPT];
          load_vec_f(c.xpt + ((long)((a & 1) * MAXR + b) * 4 + 3) * H, TAGA(a), x[0]);
          mlp_publish(c, hd, sm, a, x, b);
        }
        if (!big16<V>(hd)) mlp_poll(c, hd, sm, a);
      } else {
        for (int b = 0; b < hd.B; b++) {
          float x[1][EPT];
          load_vec_f(c.xpt + ((long)((a & 1) * MAXR + b) * 4 + 3) * H, TAGA(a), x[0]);
          mlp_in_x(c, hd, sm, a, x, b);
        }
      }
      __syncthreads();
    }
    MARK(11);
    if (big16<V>(hd)) {
      // the MoE per half of 8 rows (router inputs of the half -> router -> routing -> w13 -> w2a; act16t set per half)
      if (MOE1) {   // router per half (8-row smem input), then one routing / w13 / w2a over the 16 rows
        if (ROUT1) moe_in16(c, hd, sm, a); else
        for (int h = 0; h * SMB < hd.B; h++) {
          Hdr hh = hd; hh.ro = h * SMB; hh.B = min(SMB, hd.B - hh.ro);
          mlp_poll_half(c, hh, sm, a, false);
          ph_router<V>(c, hh, sm, a, RTR2 ? h * 8 : 0);   // RTR2: half h on blocks 8h .. 8h + 7 (the halves in parallel)
          __syncthreads();
          mlp_conv_half(c, hh, sm, a);   // after the router: rows 8 + b overwrite the router input in place
        }
        MARK(5);
        routing<V>(c, hd, sm, a);
        MARK(12);
        ph_w13<V>(c, hd, sm, a);
        __syncthreads(); MARK(6);
        ph_w2a<V, M>(c, hd, sm, a);
        __syncthreads(); MARK(17);
      } else
      for (int h = 0; h * SMB < hd.B; h++) {
        Hdr hh = hd; hh.ro = h * SMB; hh.B = min(SMB, hd.B - hh.ro);
        mlp_poll_half(c, hh, sm, a);
        ph_router<V>(c, hh, sm, a);
        __syncthreads(); if (h) MARK(5);
        routing<V>(c, hh, sm, a);
        if (h) MARK(12);
        ph_w13<V>(c, hh, sm, a);
        __syncthreads(); if (h) MARK(6);
        ph_w2a<V, M>(c, hh, sm, a);
        __syncthreads(); if (h) MARK(17);
      }
    } else {
    ph_router<V>(c, hd, sm, a);
    GBAR(); MARK(5);
    routing<V>(c, hd, sm, a);
    MARK(12);
    ph_w13<V>(c, hd, sm, a);
    GBAR(); MARK(6);
    ph_w2a<V, M>(c, hd, sm, a);
    GBAR(); MARK(17);
    }
    ph_w2b<V>(c, hd, sm, a);
    GBAR();
  }
  MARK(13);
  ph_final<V>(c, hd, sm);
  MARK(MKPROF_WAIT);
  grid_bar(c, nbar, base);
  if (blockIdx.x == 0) {
    if (threadIdx.x < hd.B) {
      const unsigned long long k = c.amax[threadIdx.x];
      const int tok = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFu));
      c.next_tok[threadIdx.x] = tok;
      c.amax[threadIdx.x] = 0ull;
      st_tag(c.ntt + threadIdx.x, (uint32_t)tok, (hd.tag0 + 60u) & 0x7fffffffu);
      if (hd.out_tok) hd.out_tok[threadIdx.x] = tok + 1;   // 0 = not yet written (host polls)
    }
  }
  }   // steps
  if (blockIdx.x == 0 && threadIdx.x == 0) c.bar[1] = base + nbar * gridDim.x;
}

// ------------------------------------------------------------------------------------------------ n-gram drafts
// One block (engine/ngram.py describes the rule): record the verify pass's target tokens at positions c_old + 1 ..
// c_old + 8 (record), then the longest (<= 8) earlier occurrence of the committed suffix ending at the anchor
// (c = vpos[0]); ties: the most recent; if at least minn long its continuation (periodic past c) replaces the drafts
// vtok[1..7] and the verify pass gets nb rows (vpos[1]).
extern "C" __global__ void __launch_bounds__(1024) mk_ngram_k(int* hist, int T, const int* emit, int* vpos, int* vtok, int record, int minn, int nb) {
  __shared__ int suf[8];
  __shared__ unsigned long long red[32];
  const int c = vpos[0];
  if (record && threadIdx.x < 8) {
    const int p = c - emit[0] + 1 + threadIdx.x;
    if (p >= 0 && p < T) hist[p] = emit[1 + threadIdx.x];
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
    if (L >= minn && threadIdx.x < 7) {
      const int per = max(c - i, 1);
      vtok[1 + threadIdx.x] = hist[min(i + 1 + (int)threadIdx.x % per, T - 1)];
    }
    if (L >= minn && threadIdx.x == 0) vpos[1] = nb;
  }
}

// ------------------------------------------------------------------------------------------------ host API
extern "C" {
int mk_ngram(int* hist, int T, const int* emit, int* vpos, int* vtok, int record, int minn, int nb, void* stream) {
  mk_ngram_k<<<1, 1024, 0, (cudaStream_t)stream>>>(hist, T, emit, vpos, vtok, record, minn, nb);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
int mk_ctx_bytes() { return (int)sizeof(Ctx); }
int mk_layer_bytes() { return (int)sizeof(LayerW); }
int mk_app_bytes() { return (int)sizeof(App); }
int mk_hdr_bytes() { return (int)sizeof(Hdr); }
int mk_smem_bytes() { return (int)sizeof(Smem); }
int mk_nth() { return NTH; }
int mk_cn_total() { return CN_TOTAL; }
int mk_ring() { return RING; }
int mk_maxu() { return MAXU; }
#ifdef W12
int mk_w12() { return 1; }
#else
int mk_w12() { return 0; }
#endif
#ifndef VSMEM_EXTRA
#define VSMEM_EXTRA (W13_ASYNC ? 15 * W13_RING : 0)   // extra dynamic shared memory of the verify kernel (bytes)
#endif
int mk_setup() {
  cudaError_t e = cudaFuncSetAttribute(mk_step<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem));
  if (e == cudaSuccess) e = cudaFuncSetAttribute(mk_step<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem));
  if (e == cudaSuccess) e = cudaFuncSetAttribute(mk_step<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem) + VSMEM_EXTRA);
  if (e != cudaSuccess) { fprintf(stderr, "mk_setup: %s\n", cudaGetErrorString(e)); return 1; }
  int nb = 0, nb2 = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mk_step<false>, NTH, sizeof(Smem));
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb2, mk_step<true>, NTH, sizeof(Smem) + VSMEM_EXTRA);
  if (nb2 < nb) nb = nb2;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb2, mk_step<false, true>, NTH, sizeof(Smem));
  if (nb2 < nb) nb = nb2;
  if (nb < 1) { fprintf(stderr, "mk_setup: occupancy %d\n", nb); return 2; }
  return 0;
}
int mk_launch(const void* ctx_dev, const Hdr* hd, int nblk, void* stream) {
  Hdr h = *hd;
  void* args[] = {(void*)&ctx_dev, (void*)&h};
  const void* kern = h.vpos ? (const void*)mk_step<true> : (h.B >= 2 ? (const void*)mk_step<false, true> : (const void*)mk_step<false>);
  cudaError_t e = cudaLaunchCooperativeKernel(kern, dim3(nblk), dim3(NTH), args, sizeof(Smem) + (h.vpos ? VSMEM_EXTRA : 0), (cudaStream_t)stream);
  if (e != cudaSuccess) { fprintf(stderr, "mk_launch: %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
void* mk_host_alloc(size_t n) { void* p = nullptr; if (cudaHostAlloc(&p, n, cudaHostAllocMapped) != cudaSuccess) return nullptr; memset(p, 0, n); return p; }
}
