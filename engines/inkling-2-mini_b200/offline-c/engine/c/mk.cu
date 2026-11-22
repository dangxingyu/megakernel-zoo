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
#ifndef NTH
#define NTH 768
#endif
#define NWARP (NTH / 32)
// merge run (idea of s15-mini-b200-b2 s0020/s0021 "K1"): this kernel only serves batch 1 (B >= 2 runs on engine/a1),
// so the batch size is a compile-time constant and every B >= 2 path is dead code (KB1=0: the generic kernel)
#ifndef KB1
#define KB1 0   // KB1=1: B=1 +375 us (fewer spills, slower): off
#endif
#if KB1
#define HB(h) 1
#else
#define HB(h) ((h).B)
#endif
#ifndef W13_UNR
#define W13_UNR 12
#endif
#ifndef W2S_RS
#define W2S_RS (IE + 8)   // w2a staged input rows: stride in halves (IE: every column on one bank, 8-way conflicts at 4 tokens)
#endif
#ifndef VSMEM_EXTRA_V
#define VSMEM_EXTRA_V (4 * 36 * 576)   // W13_TMA: slots beyond the Smem struct (bytes)
#endif
#ifndef D_UNR
#define D_UNR 3
#endif
#ifndef QKV_UNR
#define QKV_UNR BF_UNR
#endif
#ifndef RTPF
#define RTPF 1
#endif
#ifndef GDPF
#define GDPF 0
#endif
#ifndef QPF
#define QPF 0
#endif
#ifndef GRP_EXP
#define GRP_EXP true
#endif
#define SMB MAXB
#define CH 128           // attention positions per item (at most)
#define MAXCH_S MAXCH
#define MAXCH 64         // chunks per (token, kv head)
#define MAXU 64          // union experts per layer
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
  const uint8_t* un8;                      // verify passes: INT8 copy of the unembedding (per-row scale), fragment order
  const float* un8s; const float* un8m;    // per row: scale, error coefficient (max |w - s q| + 6e-4 max |w|) * (1 + 1e-3)
  float* zhi; unsigned* zlo;               // [MAXB][UN_STRIPS * 16] upper bounds of z; per row max lower bound (ordered)
};
#define FLAG_DEVTAG 0x20000000   // Hdr.flags: verify pass tags from the device counter (KVONLY kernels)
struct Hdr { int B, pos, flags, nsteps; const int* tokens; float* logits; int* out_tok; int out_stride; unsigned tag0;
             const int* prompt; int plen; unsigned tok_tag;
             const int* vpos; };   // vpos != null: verify pass, rows b < vpos[1] = one sequence (cache slot 0) at positions vpos[0] + b
#ifndef QSEQ
#define QSEQ 0   // (1 measured +5..15 us) shared attention (verify): q polls one at a time per lane
#endif
#ifndef LS_SEQ_S
#define LS_SEQ_S 0   // step kernels: mix sources polled one source at a time
#endif
#ifndef LS_SEQ
#define LS_SEQ 0   // (1 measured +10..25 us) verify kernel: mix sources polled one source at a time
#endif
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
#ifndef RING
#define RING 16   // short-convolution history: raw inputs of the last positions, slot = position % RING
#endif
// row b of a step: position, cache slot (sequence) and how many earlier rows of the same pass precede it in its sequence
// (V: the verify kernel; the step kernel is compiled with V = false, so its code is the plain one-position code)
template <bool V> __device__ __forceinline__ int row_t(const Hdr& hd, int b) { return hd.pos + (V ? b : 0); }
template <bool V> __device__ __forceinline__ int row_seq(const Hdr& hd, int b) { return V ? 0 : b; }
template <bool V> __device__ __forceinline__ int row_nin(const Hdr& hd, int b) { return V ? b : 0; }   // nsteps > 0: device loop over nsteps steps
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
struct Smem {
  union {
    bf16 xs[3][SMB][HP];                 // B operands (bf16), streams q / k / v (router, unembed: stream 0)
    bf16 xo[SMB][QWP];                   // o_proj input
    struct { bf16 xr[SMB][HP]; __half x16[SMB][2][HP]; } e;   // router input + expert input (f16 hi / lo)
    struct {                               // attention item
      bf16 kk[CH][HD + 8]; bf16 vv[CH][HD + 8]; bf16 qb[8][HD + 8]; bf16 ph[16][CH + 8]; bf16 pl[16][CH + 8];
      float sc[3][CH]; float pr[16][CH]; float rr[3][16];
      float kt[MAXB][HD]; float vt[MAXB][HD]; float ml[2][3][2]; float po[3][HD]; float raw[2][HD];
    } at;
    struct {                               // verify passes: attention record item shared by all rows (64 keys)
      bf16 kk[64][HD + 8]; bf16 vv[64][HD + 8]; bf16 qb[3 * MAXB][HD + 8]; bf16 ph[32][64 + 8]; bf16 pl[32][64 + 8];
      float sc[3 * MAXB][64]; float prw[16][64 + MAXB]; float rr[3 * MAXB][16];
      float kt[MAXB][HD]; float vt[MAXB][HD]; float ml[3 * MAXB][2]; float raw[2][MAXB][HD];
    } as;
    float mrg[MAXCH_S * 3 * 130 + 3 * MAXCH];      // attention merge staging
    __half w2s[24][2][IE];                       // w2 inputs (expert, token) pairs of a round, hi / lo
  } u;
  float4 red4[NWARP][32];                  // block GEMV: partial fragments per warp
  float w2o[MAXB][TOPK][16];               // w2: per (token, expert rank) output strip
  float red[NWARP][32];
  float bcast[64];
  float sc16[MAXB];                        // 2^-s of the expert input f16 scaling per token
  // routing tables (identical in every block)
  int nU; int uexp[MAXU]; int uncnt[MAXU]; int utok[MAXU][4];   // union entries hold <= 4 tokens
  float ef13[NE], ef2[NE];                 // per expert: w13 / w2 scale factors of the layer (loaded before the routing polls)
  int tex[MAXB][TOPK]; float twt[MAXB][TOPK];   // per token: experts sorted by id, weights
  unsigned umask[4];
  unsigned tokm[NE / 4];                   // routing: per expert the bit mask of its tokens (8 bits each)
  int flag;
  unsigned subn;
  unsigned long long tbar[8];              // verify passes (W13_TMA): mbarriers of the w13 overflow slots
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
template <int NS, bool SEQ = false>
__device__ __forceinline__ void load_srcs(const Ctx& c, const Hdr& hd, const int* idx, unsigned msk, int b, float (*v)[EPT]) {
  // loads issued unconditionally (clamped addresses): arrays assigned under a condition would live in local memory
  const int p1 = EV(2) ? threadIdx.x + NTH : threadIdx.x;
  uint2 r[NS][2];
  if (!SEQ) {
#pragma unroll
    for (int s = 0; s < NS; s++) {
      const uint2* base = c.pbt + ((long)idx[s] * MAXB + b) * NPAIR;
      r[s][0] = ld_tag(base + threadIdx.x);
      r[s][1] = ld_tag(base + p1);
    }
  }
#pragma unroll
  for (int s = 0; s < NS; s++) {
    const uint2* base = c.pbt + ((long)idx[s] * MAXB + b) * NPAIR;
    if (SEQ) { r[s][0] = ld_tag(base + threadIdx.x); r[s][1] = ld_tag(base + p1); }
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
#define BF_UNR 3   // bf16 GEMV: units of loads in flight per warp (3: step B=1 -30 us, B=8 -26..-115, verify -20..-45 vs 2)
#endif
template <int UNR_ = BF_UNR, class BL>
__device__ __forceinline__ void piece_bf16(const uint8_t* wst, int u0, int nu, const BL& bl, float* acc) {
  constexpr int BU = UNR_;
  const int lane = lane_id();
  int u = u0;
  const int ue = u0 + nu;
  for (; u + BU <= ue; u += BU) {
    uint4 w[4 * BU];
#pragma unroll
    for (int q = 0; q < 4 * BU; q++) w[q] = ldg_nc(wst + ((long)(u + (q >> 2)) * 4 + (q & 3)) * 512 + lane * 16);
#pragma unroll
    for (int q = 0; q < 4 * BU; q++) {
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
// INT8 unembedding strip (per lane 32 B per unit: tiles 0..3 x 8 int8 in the bf16 fragment's element order), exact
// int8 -> bf16, products with the bf16 rows exact, fp32 sums
__device__ __forceinline__ uint32_t i8x2_bf16(uint32_t w, int sh) {
  const float a = (float)(signed char)((w >> sh) & 0xff), b = (float)(signed char)((w >> (sh + 8)) & 0xff);
  const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<const uint32_t*>(&v);
}
template <class BL>
__device__ __forceinline__ void piece_i8(const uint8_t* wst, const BL& bl, float* acc) {
  const int lane = lane_id();
#pragma unroll 1
  for (int u = 0; u < 36; u += 6) {
    uint4 w[12];
#pragma unroll
    for (int q = 0; q < 12; q++) w[q] = ldg_nc(wst + (long)(u + (q >> 1)) * 1024 + lane * 32 + (q & 1) * 16);
#pragma unroll
    for (int q = 0; q < 12; q++) {
#pragma unroll
      for (int tt = 0; tt < 2; tt++) {
        const uint32_t lo = tt ? w[q].z : w[q].x, hi = tt ? w[q].w : w[q].y;
        const uint2 b = bl.get((u + (q >> 1)) * 64 + (2 * (q & 1) + tt) * 16);
        mma_bf16(acc, i8x2_bf16(lo, 0), i8x2_bf16(lo, 16), i8x2_bf16(hi, 0), i8x2_bf16(hi, 16), b.x, b.y);
      }
    }
  }
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
// ---- bulk copies global -> shared with an mbarrier (verify passes, W13_TMA)
__device__ __forceinline__ unsigned smaddr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(unsigned long long* b, unsigned cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smaddr(b)), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void mbar_expect_tx(unsigned long long* b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smaddr(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void tma_g2s(void* dst, const void* src, unsigned bytes, unsigned long long* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smaddr(dst)), "l"(src), "r"(bytes), "r"(smaddr(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n MBW%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra MBW%=;\n}"
               :: "r"(smaddr(b)), "r"(parity) : "memory");
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
__device__ __forceinline__ void piece_fp4(const uint8_t* wst, int u0, int nu, const __half* const* colp, float (*acc)[4], unsigned tag = 0, bool fake = false) {
  const int lane = lane_id(), g = lane >> 2;
#ifndef FP4_UNR
#define FP4_UNR 6
#endif
#ifndef DIAG_MOE
#define DIAG_MOE 0
#endif
#ifndef W13_LDA
#define W13_LDA 0
#endif
  constexpr int UNR = UNR_ ? UNR_ : FP4_UNR;
  int u = u0;
  const int ue = u0 + nu;
  for (; u + UNR <= ue; u += UNR) {
    uint4 cw[UNR]; uint2 sw[UNR];
#pragma unroll
    for (int q = 0; q < UNR; q++) {
#if DIAG_MOE == 1 || DIAG_MOE == 3   // timing only: no weight loads (3: only where fake)
      if (DIAG_MOE == 1 || fake) {
        const unsigned z = (unsigned)(uintptr_t)(wst + (long)(u + q) * 576) ^ lane;
        cw[q] = make_uint4(z, z * 3u, z * 5u, z * 7u); sw[q] = make_uint2(0x38383838u ^ (z & 0x01010101u), 0x38383838u);
      } else {
        cw[q] = ldg_nc(wst + (long)(u + q) * 576 + lane * 16);
        sw[q] = ldg_nc2(wst + (long)(u + q) * 576 + 512 + g * 8);
      }
#elif W13_LDA   // allocating loads (hits on lines prefetched into L1)
      cw[q] = __ldg(reinterpret_cast<const uint4*>(wst + (long)(u + q) * 576 + lane * 16));
      sw[q] = __ldg(reinterpret_cast<const uint2*>(wst + (long)(u + q) * 576 + 512 + g * 8));
#else
      cw[q] = ldg_nc(wst + (long)(u + q) * 576 + lane * 16);
      sw[q] = ldg_nc2(wst + (long)(u + q) * 576 + 512 + g * 8);
#endif
    }
#pragma unroll
    for (int q = 0; q < UNR; q++) {
#if DIAG_MOE == 2   // timing only: no decode / mma
      acc[0][0] += __uint_as_float((cw[q].x ^ cw[q].y ^ cw[q].z ^ cw[q].w ^ sw[q].x ^ sw[q].y) & 0x3fffffffu);
#else
      uint2 bfr[NG][4];
      if (GL) load_b16_unit_gl<NG>(colp, u + q, bfr, tag); else load_b16_unit<NG>(colp, u + q, bfr);
      unit_fp4<NG>(cw[q], sw[q], bfr, acc);
#endif
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
// piece_fp4 with the item's units in shared memory (a W13_TMA slot): the same batches and MMA order
template <int NG, int UNR>
__device__ __forceinline__ void piece_fp4s(const uint8_t* sst, int u0, int nu, const __half* const* colp, float (*acc)[4]) {
  const int lane = lane_id(), g = lane >> 2;
  int u = u0;
  const int ue = u0 + nu;
  for (; u + UNR <= ue; u += UNR) {
    uint4 cw[UNR]; uint2 sw[UNR];
#pragma unroll
    for (int q = 0; q < UNR; q++) {
      cw[q] = *reinterpret_cast<const uint4*>(sst + (u + q) * 576 + lane * 16);
      sw[q] = *reinterpret_cast<const uint2*>(sst + (u + q) * 576 + 512 + g * 8);
    }
#pragma unroll
    for (int q = 0; q < UNR; q++) {
      uint2 bfr[NG][4];
      load_b16_unit<NG>(colp, u + q, bfr);
      unit_fp4<NG>(cw[q], sw[q], bfr, acc);
    }
  }
  for (; u < ue; u++) {
    const uint4 cw = *reinterpret_cast<const uint4*>(sst + u * 576 + lane * 16);
    const uint2 sw = *reinterpret_cast<const uint2*>(sst + u * 576 + 512 + g * 8);
    uint2 bfr[NG][4];
    load_b16_unit<NG>(colp, u, bfr);
    unit_fp4<NG>(cw, sw, bfr, acc);
  }
}
// two items' chains of CU units in one batch (verify w2a, W2X2): loads of both first, then each item's units in order
// (each accumulator sees exactly piece_fp4's MMA sequence)
#ifndef X2_SHAREB
#define X2_SHAREB 0   // piece_fp4_x2: items of the same expert share the B-fragment loads
#endif
template <int CU>
__device__ __forceinline__ void piece_fp4_x2(const uint8_t* wa, const uint8_t* wb, bool hb, int u0, const __half* const* ca, const __half* const* cb, float (*accA)[4], float (*accB)[4]) {
  const int lane = lane_id(), g = lane >> 2;
  uint4 cwa[CU], cwb[CU]; uint2 swa[CU], swb[CU];
  const uint8_t* wb_ = hb ? wb : wa;
#pragma unroll
  for (int q = 0; q < CU; q++) {
    cwa[q] = ldg_nc(wa + (long)(u0 + q) * 576 + lane * 16);
    swa[q] = ldg_nc2(wa + (long)(u0 + q) * 576 + 512 + g * 8);
  }
#pragma unroll
  for (int q = 0; q < CU; q++) {
    cwb[q] = ldg_nc(wb_ + (long)(u0 + q) * 576 + lane * 16);
    swb[q] = ldg_nc2(wb_ + (long)(u0 + q) * 576 + 512 + g * 8);
  }
#if X2_SHAREB
  // both items of one union entry (same tokens): one B-fragment load per unit; warp-uniform (mma.sync needs every lane)
  if (__all_sync(0xffffffffu, hb && ca[0] == cb[0] && ca[1] == cb[1])) {
#pragma unroll
    for (int q = 0; q < CU; q++) {
      uint2 bfr[1][4];
      load_b16_unit<1>(ca, u0 + q, bfr);
      unit_fp4<1>(cwa[q], swa[q], bfr, accA);
      unit_fp4<1>(cwb[q], swb[q], bfr, accB);
    }
    return;
  }
#endif
#pragma unroll
  for (int q = 0; q < CU; q++) {
    uint2 bfr[1][4];
    load_b16_unit<1>(ca, u0 + q, bfr);
    unit_fp4<1>(cwa[q], swa[q], bfr, accA);
  }
  if (hb) {
#pragma unroll
    for (int q = 0; q < CU; q++) {
      uint2 bfr[1][4];
      load_b16_unit<1>(cb, u0 + q, bfr);
      unit_fp4<1>(cwb[q], swb[q], bfr, accB);
    }
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
#define FP8_UNR 3   // verify kernel (s18): MXFP8 expert pieces 3 units deep (4: +10..30 us per pass)
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
#if DIAG_MOE == 1
        const unsigned z = (unsigned)(uintptr_t)(wst + (long)u * UB) ^ lane;
        c0[q] = make_uint4(z, z * 3u, z * 5u, z * 7u); s4[MX ? 0 : q] = make_uint2(0x38383838u ^ (z & 0x01010101u), 0x38383838u);
#else
        c0[q] = ldg_nc(wst + (long)u * UB + lane * 16);
        s4[MX ? 0 : q] = ldg_nc2(wst + (long)u * UB + 512 + g * 8);
#endif
      }
    }
#pragma unroll
    for (int q = 0; q < NQ; q++) {
      const int u = UPC * (q / BPC) + j + (q % BPC);
#if DIAG_MOE == 2
      if (!MX) { acc[q / BPC][0][0] += __uint_as_float((c0[q].x ^ c0[q].y ^ c0[q].z ^ c0[q].w ^ s4[MX ? 0 : q].x) & 0x3fffffffu); continue; }
#endif
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
#elif defined(MARK_BAR)
#define SUBMARK(tag) asm volatile("" ::: "memory")
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
#ifndef MG_UNR
#define MG_UNR 1     // verify kernel: shared-attention merge record polls in flight per thread
#endif
#ifndef OP_UNR_S
#define OP_UNR_S 4   // step kernels: o_proj input polls in flight per thread
#endif
#ifndef OP_UNR_V
#define OP_UNR_V 4   // verify kernel: o_proj input polls in flight per thread
#endif
#ifndef W2S_UNR_V
#define W2S_UNR_V 2  // verify kernel: w2a input staging polls in flight per thread
#endif
#ifndef W2B_UNR_V
#define W2B_UNR_V 8   // verify kernel: w2b expert-contribution polls per batch (1, 2, 4, 8)
#endif
#ifndef PR_UNR_S
#define PR_UNR_S 2   // step kernels (B >= 2): tagged row polls in flight per thread
#endif
#ifndef W2S_UNR_S
#define W2S_UNR_S 2  // step kernels: w2a input staging polls in flight per thread
#endif
#ifndef PR_UNR_V
#define PR_UNR_V 2   // verify kernel: tagged row polls in flight per thread (8 / 4 / 3 / 2: K=8 5640 / 5598 / 5494 / 5486 us)
#endif
template <int PRU = 8, class DST>
__device__ __forceinline__ void poll_rows(const uint2* src, int nrows, DST dst, unsigned tag) {
  const int n = nrows * NPAIR;   // words, two per load
  for (int i0 = 2 * threadIdx.x; i0 < n; i0 += 2 * NTH * PRU) {
    uint2 r[PRU][2];
#pragma unroll
    for (int q = 0; q < PRU; q++) ld_tag2(src + min(i0 + q * 2 * NTH, n - 2), r[q][0], r[q][1]);
#pragma unroll
    for (int q = 0; q < PRU; q++) {
      const int i = i0 + q * 2 * NTH;
      if (i >= n) continue;
      const int row = i / NPAIR, w = i % NPAIR;
      uint32_t* d = reinterpret_cast<uint32_t*>(dst(row)) + w;
      d[0] = wait_tag(src + i, r[q][0], tag);
      d[1] = wait_tag(src + i + 1, r[q][1], tag);
    }
  }
}
#ifndef PR1_MIN
#define PR1_MIN 3
#endif
#ifndef PR1_MAX
#define PR1_MAX 4   // verify passes of at most this many rows poll published rows 1 deep (else PR_UNR_V)
#endif
template <bool V, class DST>
__device__ __forceinline__ void pollr(int B, const uint2* src, int nrows, DST dst, unsigned tag) {
  if (V && B >= PR1_MIN && B <= PR1_MAX) poll_rows<1>(src, nrows, dst, tag);
  else poll_rows<V ? PR_UNR_V : PR_UNR_S>(src, nrows, dst, tag);
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
template <bool V> __device__ __forceinline__ bool desig(const Hdr& hd) { return hd.B >= (V ? VDES_MIN : 2) && !hyb2<V>(hd); }
// The attention input streams of application a -> normalized bf16 smem streams (returns their number), or for the
// gated applications (loop passes 2 and 3) G1: blocks (token, controller) write x, m and [unit(x), unit(m)] (returns -1)
template <bool V> __device__ int ph_attn_in(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = HB(hd);
  const bool dz = desig<V>(hd);
#ifndef PF_MIN_S
#define PF_MIN_S 2   // step kernels: L2 prefetch of the qkv / router strips from this many rows
#endif
#ifndef PF_MIN
#define PF_MIN 9   // L2 prefetch of the qkv / router strips from this many rows (the step kernel: from 2)
#endif
  const bool pfq = B >= (V ? PF_MIN : PF_MIN_S);
  if (pfq && threadIdx.x < 4) {   // this block's qkv strips into L2 while the mix waits for its sources (B >= 2: the mix is long enough)
    const int strip = blockIdx.x + threadIdx.x * gridDim.x;
    if (strip < QG_STRIPS + 2 * L.nsk) pf_l2(L.wqkv + (long)strip * 36 * 2048, 36 * 2048);
  } else if (pfq && threadIdx.x == 32 && blockIdx.x < 8) pf_l2(L.wr + (long)blockIdx.x * 36 * 2048, 36 * 2048);   // router strip
#if QPF > 0
  if (B == 1) {
    // qkv: block_strips<6> strips blockIdx + q G (q < 4 groups), warp w: units [6 (w % 6), +6) of strip q = w / 6
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, strip = blockIdx.x + (w / 6) * gridDim.x;
    if (strip < QG_STRIPS + 2 * L.nsk) {
      const uint8_t* p = L.wqkv + (long)strip * 36 * 2048 + (long)(6 * (w % 6)) * 2048 + l * 64;
#pragma unroll
      for (int q = 0; q < QPF; q++) asm volatile("prefetch.global.L1 [%0];" :: "l"(p + q * 2048));
    }
  }
#endif
  const int nq = L.glob ? 3 : 1;
  const int par = a & 1;
  const bool hy = hyb2<V>(hd);
  if (ap.nmsg == 0 && (dz || hy)) {
    uint2* xs_pub = c.xst + (long)par * 3 * MAXB * NPAIR;   // [3][MAXB][NPAIR]
    // hybrid (2 rows): jobs of row 1 on the last blocks; row 0 below in every block
    for (int job = hy ? nq + (int)(gridDim.x - 1 - blockIdx.x) : dsel<V>(hd); job < B * nq; job += gridDim.x) {
      const int b = job / nq, q = job % nq;
      float x[1][EPT];
      float wv[1][EPT];
      load_w(L.norm[q], wv[0]);
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
        load_srcs<5, (V ? LS_SEQ : LS_SEQ_S)>(c, hd, idx, msk, b, v);
        depth_mix_q<5, 1>(v, msk, qv, 1, x, sm);
      }
      uint2* dst[1] = {xs_pub + ((long)q * MAXB + b) * NPAIR};
      norm_publish<1>(x, wv, 1, dst, TAGA(a), sm);
    }
    if (!hy) {
      const unsigned need = ((V || POLL_STEP) && POLL_NEEDED) ? qkv_streams(L, nq) : 7u;
      for (int q = 0; q < nq; q++)
        if ((need >> q) & 1u) pollr<V>(hd.B, xs_pub + (long)q * MAXB * NPAIR, B, [&](int r) { return &sm.u.xs[q][r][0]; }, TAGA(a));
      __syncthreads();
      return nq;
    }
  }
  if (ap.nmsg == 0) {
    for (int b = 0; b < (hy ? 1 : B); b++) {
      float v[5][EPT];
      float x[3][EPT];
      float wv[3][EPT];   // norm weights first (an ordinary load after a poll would wait for it)
      for (int q = 0; q < 3; q++) load_w(L.norm[q < nq ? q : 0], wv[q]);
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
        load_srcs<5, (V ? LS_SEQ : LS_SEQ_S)>(c, hd, idx, msk, b, v);
        SUBMARK(31);
        if (nq == 1) depth_mix_q<5, 1>(v, msk, qv, 1, x, sm); else depth_mix_q<5, 3>(v, msk, qv, 3, x, sm);
        SUBMARK(32);
      }
      bf16* dst[3] = {&sm.u.xs[0][b][0], &sm.u.xs[1][b][0], &sm.u.xs[2][b][0]};
      if (nq == 1) rmsnorm_store_n<1>(x, wv, 1, dst, sm); else rmsnorm_store_n<3>(x, wv, 3, dst, sm);
      SUBMARK(33);
    }
    if (hy) {   // row 1 from the last blocks
      const uint2* xs_pub = c.xst + (long)par * 3 * MAXB * NPAIR;
      const unsigned need = (POLL_NEEDED) ? qkv_streams(L, nq) : 7u;
      for (int q = 0; q < nq; q++)
        if ((need >> q) & 1u) pollr<V>(hd.B, xs_pub + ((long)q * MAXB + 1) * NPAIR, B - 1, [&](int r) { return &sm.u.xs[q][1 + r][0]; }, TAGA(a));
    }
    __syncthreads();
    return nq;
  }
  return -1;
}
// mlp-gate G1: block per token (controller 3)
template <bool V> __device__ void mlp_mix(PH_ARGS, int a, int b, float (*x)[EPT]);
template <bool V> __device__ void ph_mlp_g1(PH_ARGS, int a) {   // (unused)
  const App& ap = c.A[a];
  const int B = HB(hd), par = a & 1;
  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    float x[1][EPT], m[1][EPT], ms[2][EPT];
    mlp_mix<V>(c, hd, sm, a, b, x);
    const float* mq[1] = {c.loopq + ((long)ap.loopl * 4 + 3) * H};
    float mqv[1][EPT];
    load_queries<1>(mq, 1, mqv);
    load_srcs<2>(c, hd, ap.msg, 3u, b, ms);
    depth_mix_q<2, 1>(ms, 3u, mqv, 1, m, sm);
    float s2[2] = {0.f, 0.f};
#pragma unroll
    for (int k = 0; k < EPT; k++) { s2[0] += x[0][k] * x[0][k]; s2[1] += m[0][k] * m[0][k]; }
    block_sum<2>(s2, sm);
    const float rx = rsqrtf(s2[0] / (float)H + 1e-6f), rm = rsqrtf(s2[1] / (float)H + 1e-6f);
    const long o = ((long)(par * MAXB + b) * 4 + 3) * H;
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
template <bool V> __device__ void ph_gate_d(PH_ARGS, int a, int c0, int nc) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = HB(hd), par = a & 1;
  const int N = nc * 4 * B;
  constexpr int ST = 2 * H + 8;
  bf16* xb = &sm.u.xs[0][0][0];                                   // [1][ST]
  for (int it = blockIdx.x; it < N; it += gridDim.x) {
    const int b = it % B, sj = it / B, ctrl = c0 + sj / 4, j = sj % 4, strip = ctrl * 4 + j;
#if GDPF > 0
    if (B == 1) {
      // the first GDPF units of every warp's share (units [3 w, 3 w + 3)) into this SM's L1 while the mixes run
      const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
      const uint8_t* p = L.cD + (long)strip * 72 * 2048 + (long)(3 * w) * 2048 + l * 64;
#pragma unroll
      for (int q = 0; q < GDPF; q++) asm volatile("prefetch.global.L1 [%0];" :: "l"(p + q * 2048));
    }
#endif
    float x[1][EPT], m[1][EPT], ms[2][EPT];
    {
      const float* mq[1] = {c.loopq + ((long)ap.loopl * 4 + ctrl) * H};
      float mqv[1][EPT];
      load_queries<1>(mq, 1, mqv);
      const bool mfirst = (V || GATE_MFIRST_STEP) && GATE_MFIRST;
      if (mfirst) {
        load_srcs<2, (V ? LS_SEQ : LS_SEQ_S)>(c, hd, ap.msg, 3u, b, ms);
        depth_mix_q<2, 1>(ms, 3u, mqv, 1, m, sm);
      }
      if (ctrl < 3) {
        float v[5][EPT];
        const int idx[5] = {ap.blk[0], ap.blk[1], ap.blk[2], ap.blk[3], ap.pin};
        const unsigned msk = ((1u << ap.nblk) - 1u) | 16u;
        const float* qs[1] = {L.aq + (L.glob ? ctrl : 0) * H};
        float qv[1][EPT];
        load_queries<1>(qs, 1, qv);
        load_srcs<5, (V ? LS_SEQ : LS_SEQ_S)>(c, hd, idx, msk, b, v);
        depth_mix_q<5, 1>(v, msk, qv, 1, x, sm);
      } else {
        mlp_mix<V>(c, hd, sm, a, b, x);
      }
      if (!mfirst) {
        load_srcs<2, (V ? LS_SEQ : LS_SEQ_S)>(c, hd, ap.msg, 3u, b, ms);
        depth_mix_q<2, 1>(ms, 3u, mqv, 1, m, sm);
      }
    }
    float s2[2] = {0.f, 0.f};
#pragma unroll
    for (int k = 0; k < EPT; k++) { s2[0] += x[0][k] * x[0][k]; s2[1] += m[0][k] * m[0][k]; }
    block_sum<2>(s2, sm);
    const float rx = rsqrtf(s2[0] / (float)H + 1e-6f), rm = rsqrtf(s2[1] / (float)H + 1e-6f);
    const long o = ((long)(par * MAXB + b) * 4 + ctrl) * H;
#pragma unroll
    for (int jj = 0; jj < 2; jj++) {
      if (!EV(2 * jj)) continue;
      const int el = EL(2 * jj);
      *reinterpret_cast<uint32_t*>(&xb[el]) = pack_bf2(x[0][2 * jj] * rx, x[0][2 * jj + 1] * rx);
      *reinterpret_cast<uint32_t*>(&xb[H + el]) = pack_bf2(m[0][2 * jj] * rm, m[0][2 * jj + 1] * rm);
      if (j == 0) {
        st_tagf(c.xgt + o + el, x[0][2 * jj], TAGA(a)); st_tagf(c.xgt + o + el + 1, x[0][2 * jj + 1], TAGA(a));
        st_tagf(c.mgt + o + el, m[0][2 * jj], TAGA(a)); st_tagf(c.mgt + o + el + 1, m[0][2 * jj + 1], TAGA(a));
      }
    }
    __syncthreads();
    BSm bl{xb, ST, 1};
    block_strips<NWARP, 1>(sm, 1, 72,
      [&](int, int u0, int nu, float (*acc)[4]) { piece_bf16<D_UNR>(L.cD + (long)strip * 72 * 2048, u0, nu, bl, acc[0]); },
      [&](int, float (*acc)[4]) {
        const int lane = lane_id(), g = lane >> 2, t = lane & 3;
        if (t == 0) {   // column 0 = this token
          st_tagf(c.gvt + ((long)(par * MAXB + b) * 4 + ctrl) * 64 + j * 16 + g, rbf(acc[0][0]), TAGA(a));
          st_tagf(c.gvt + ((long)(par * MAXB + b) * 4 + ctrl) * 64 + j * 16 + g + 8, rbf(acc[0][2]), TAGA(a));
        }
      });
    __syncthreads();
  }
}
// F / G elementwise for controllers [c0, c0 + nc): xpt[b][ctrl][i]; one thread per (ctrl, b, i)
#ifndef FG_ROWS
#define FG_ROWS 0   // verify passes: one F/G task per (controller, channel) for all rows (weight row loaded once)
#endif
template <bool V> __device__ void ph_gate_fg(PH_ARGS, int a, int c0, int nc) {
  const LayerW& L = c.L[c.A[a].layer];
  const int B = hd.B, par = a & 1;
  if (V && FG_ROWS && B >= 2) {
    constexpr int RB = FG_ROWS > 0 ? FG_ROWS : 1;   // rows per task
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
      gs[(bb * 4 + cc) * 64 + r] = pollf(c.gvt + ((long)(par * MAXB + bb) * 4 + cc) * 64 + r, TAGA(a));
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
        const long o = ((long)(par * MAXB + min(bb0 + j, bb1 - 1)) * 4 + ctrl) * H + i;
        rx[j] = ld_tag(odd ? c.mgt + o : c.xgt + o);
      }
#pragma unroll
      for (int j = 0; j < RB; j++) {
        const int b = min(bb0 + j, bb1 - 1);
        const bool okj = ok && bb0 + j < bb1;   // warp-uniform loop: no early exit around the shuffle
        const long o = ((long)(par * MAXB + b) * 4 + ctrl) * H + i;
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
    const long o = ((long)(par * MAXB + b) * 4 + ctrl) * H + i;
    const uint2 rx = ld_tag(odd ? c.mgt + o : c.xgt + o);
    if (it0 == 0) {
      for (int k = threadIdx.x; k < B * nc * 64; k += NTH) {
        const int r = k % 64, bc = k / 64, bb = bc / nc, cc = c0 + bc % nc;
        gs[(bb * 4 + cc) * 64 + r] = pollf(c.gvt + ((long)(par * MAXB + bb) * 4 + cc) * 64 + r, TAGA(a));
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
__device__ void ph_qkv(PH_ARGS, int a, int nstreams) {
  const LayerW& L = c.L[c.A[a].layer];
  const int B = HB(hd), par = a & 1;
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
      piece_bf16<QKV_UNR>(L.wqkv + (long)strip * 36 * 2048, u0, nu, bl, acc[0]);
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
          if (tok < B) st_tagf(c.qkvt + (long)(par * MAXB + tok) * QKVR + orow + g + 8 * h, rbf(acc[0][2 * h + cc]), TAGA(a));
        }
    });
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
__device__ __forceinline__ void kv_conv4(const LayerW& L, const uint2* qkvp, int r, unsigned tag, const bf16* hist, const bf16* sc, int base_row, int ch0, int g, int t, int nin, float* out, float* xraw4 = nullptr) {
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
  if (xraw4) {
#pragma unroll
    for (int k = 0; k < 4; k++) xraw4[k] = x[k];   // the raw current values (conv ring, RAWSM)
  }
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
  const int B = HB(hd);
#ifndef SHARED_MIN
#define SHARED_MIN 2   // verify passes of at least this many rows use the shared-record attention
#endif
  if (V && B >= SHARED_MIN && (L.glob || row_t<V>(hd, B - 1) + 1 <= 1024)) { ph_attn_shared(c, hd, sm, a); return; }
  // records (online-softmax partials) always cover 64 keys, so that the merge is the same for every batch size;
  // for B >= 3 an item loads two records' keys (fewer items) and computes the two records separately
  const int sub = B <= 2 ? 1 : 2, csz = 64 * sub;
  const int tmax = row_t<V>(hd, B - 1);          // rows' positions grow with b (verify passes) or are equal
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
    const uint2* qkvp = c.qkvt + (long)(par * MAXB) * QKVR;
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
      const uint2 qsc = *reinterpret_cast<const uint2*>(L.qscale + h * HD + lane * 4);   // before the waits
      const bf16* qs4 = reinterpret_cast<const bf16*>(&qsc);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        const int d = lane * 4 + k;
        qv[k] = __uint_as_float(wait_tag(qr + h * HD + d, rq[k], tag)) * bf(qs4[k]);
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
      const float rsc = lane < 16 ? bf(L.rscale[h * 16 + lane]) : 0.f;   // before the poll
      if (lane < 16) { rv = pollf(qr + R_R + h * 16 + lane, tag) * rsc; ss = rv * rv; }
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
        kv_conv4(L, qkvp, b - d, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, g * HD + lane * 4, g, t - d, nin - d, kv,
                 (RAWSM && (V || RAWSM_STEP) && d == 0) ? &at.raw[isv][lane * 4] : nullptr);
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
      constexpr int MGR = 8;   // (MG_UNR here measured slower at ctx 1100)
      for (int i0 = tid; i0 < nrec; i0 += NTH * MGR) {
        uint2 v[MGR];
#pragma unroll
        for (int q = 0; q < MGR; q++) {
          const int i = min(i0 + q * NTH, nrec - 1);
          const int hh = i / ((nrc - 1) * 130), rest = i % ((nrc - 1) * 130);
          v[q] = ld_tag(c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH + 1) * 130 + rest);
        }
#pragma unroll
        for (int q = 0; q < MGR; q++) {
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
        if ((d & 1) == 0) st_tag(c.aoutt + (long)(par * MAXB + b) * (QW / 2) + ((3 * g + hh) * HD + d) / 2, pack_bf2(out, o1), tag);
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
  const int NR = NKV * nrcmax, NM = NKV * B;
  const unsigned tag = TAGA(a);
  const int par = a & 1;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
  auto& at = sm.u.as;
  const long cstride = (long)c.maxctx * HD;
  const uint2* qkvp = c.qkvt + (long)(par * MAXB) * QKVR;
  for (int it = blockIdx.x; it < NR + NM; it += gridDim.x) {
    if (it < NR) {
      // ---------------------------------------------------------------- record item (g, ch)
      const int ch = nrcmax - 1 - it / NKV, g = it % NKV;
      SUBMARK(64 * ch + 64 > t0 ? 59 : 60);   // 59: a record holding positions of this pass
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
      for (int i = tid; i < 32 * (64 + 8) / 2; i += NTH) {
        reinterpret_cast<uint32_t*>(&at.ph[0][0])[i] = 0u;
        reinterpret_cast<uint32_t*>(&at.pl[0][0])[i] = 0u;
      }
      for (int i = tid; i < (3 * MAXB - nq) * (HD + 8) / 2; i += NTH) reinterpret_cast<uint32_t*>(&at.qb[nq][0])[i] = 0u;
      {
        const int nj16 = (njm + 15) & ~15;
        for (int i = tid; i < (nj16 - njm) * (HD / 2); i += NTH) reinterpret_cast<uint32_t*>(&at.vv[njm + i / (HD / 2)][0])[i % (HD / 2)] = 0u;
      }
      // warp tasks: k / v of the pass's positions in the record, then q and r of every (row, head)
#ifndef KVSPLIT
#define KVSPLIT 0   // the k / v conv of a pass position on S warps (128 / S channels per lane; S = 4 if it all fits one
                    // round of warps, else 2), raw values staged, then the norm by one warp in the original lane order
#endif
      const int S = !KVSPLIT || np == 0 ? 1 : (8 * np + nq <= NWARP ? 4 : (4 * np + nq <= 2 * NWARP ? 2 : 1));
      for (int task = w; task < 2 * np * S + nq; task += NWARP) {
        if (S > 1 && task < 2 * np * S) {
          const int kt = task / S, part = task % S;   // k / v task kt, channel part
          const bool isv = kt & 1;
          const int i = kt >> 1, p = P0 + i, r = p - t0;
          const bf16* hist = (isv ? c.hv : c.hk) + ((long)ap.slot * MAXB + 0) * RING * 1024;
          float* dst = isv ? at.vt[i] : at.kt[i];
          for (int j = 0; j < 4 / S; j++) {
            const int cl = part * (128 / S) + j * 32 + lane;   // channel within the head
            const int ch2 = g * HD + cl;
            dst[cl] = kv_conv(L, qkvp, r, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, ch2, g, p, r, &at.raw[isv][i][cl]);
          }
          continue;
        }
        const int task1 = task - (S > 1 ? 2 * np * (S - 1) : 0);   // q / r tasks keep their numbering
        if (S == 1 && task1 < 2 * np) {
          const int task = task1;
          const bool isv = task & 1;
          const int i = task >> 1, p = P0 + i, r = p - t0;
          const bf16* hist = (isv ? c.hv : c.hk) + ((long)ap.slot * MAXB + 0) * RING * 1024;
          float kv[4], ss = 0.f;
#ifndef KV4_SH
#define KV4_SH KV4   // shared-record attention: the 4 channels' loads before the first wait (kv_conv4)
#endif
#if KV4_SH == 2   // kv_conv4, with the per-channel path kept compiled behind a never-true runtime condition (register allocation)
          if (!(hd.flags & 0x40000000)) {
            kv_conv4(L, qkvp, r, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, g * HD + lane * 4, g, p, r, kv, &at.raw[isv][i][lane * 4]);
#pragma unroll
            for (int k = 0; k < 4; k++) ss += kv[k] * kv[k];
          } else {
#pragma unroll
            for (int k = 0; k < 4; k++) {
              const int ch2 = g * HD + lane * 4 + k;
              kv[k] = kv_conv(L, qkvp, r, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, ch2, g, p, r, &at.raw[isv][i][lane * 4 + k]);
              ss += kv[k] * kv[k];
            }
          }
#elif KV4_SH
          kv_conv4(L, qkvp, r, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, g * HD + lane * 4, g, p, r, kv, &at.raw[isv][i][lane * 4]);
#pragma unroll
          for (int k = 0; k < 4; k++) ss += kv[k] * kv[k];
#else
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const int ch2 = g * HD + lane * 4 + k;
            kv[k] = kv_conv(L, qkvp, r, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, ch2, g, p, r, &at.raw[isv][i][lane * 4 + k]);
            ss += kv[k] * kv[k];
          }
#endif
#pragma unroll
          for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
          const float f = isv ? rsqrtf(ss / 128.f + 1e-6f) : fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
          float* dst = isv ? at.vt[i] : at.kt[i];
#pragma unroll
          for (int k = 0; k < 4; k++) dst[lane * 4 + k] = rbf(kv[k] * f);
        } else {
          // q (all lanes, 4 dims each) and r (lanes < 16) of (row b, head hh): loads first, then the waits
          const int qi = task1 - 2 * np, b = qi / 3, hh = qi % 3, h = 3 * g + hh;
          const uint2* qr = qkvp + (long)b * QKVR;
          const int t = t0 + b;
          const float tau = (L.glob && t + 1 > 8192) ? (1.0f + 0.1f * logf((float)(t + 1) / 8192.0f)) : 1.0f;
          float qv[4], ss = 0.f;
          uint2 rq[4];
          if (!QSEQ) {
#pragma unroll
            for (int k = 0; k < 4; k++) rq[k] = ld_tag(qr + h * HD + lane * 4 + k);
          }
          const uint2 rrv = ld_tag(qr + R_R + h * 16 + (lane & 15));
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const int d = lane * 4 + k;
            if (QSEQ) rq[k] = ld_tag(qr + h * HD + d);
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
      if (S > 1) {   // norms of the split k / v tasks: lane sums its channels 4 lane .. + 3 in order, then the butterfly
        __syncthreads();
        for (int kt = w; kt < 2 * np; kt += NWARP) {
          const bool isv = kt & 1;
          float* dst = isv ? at.vt[kt >> 1] : at.kt[kt >> 1];
          float kv[4], ss = 0.f;
#pragma unroll
          for (int k = 0; k < 4; k++) { kv[k] = dst[lane * 4 + k]; ss += kv[k] * kv[k]; }
#pragma unroll
          for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
          const float f = isv ? rsqrtf(ss / 128.f + 1e-6f) : fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
          __syncwarp();
#pragma unroll
          for (int k = 0; k < 4; k++) dst[lane * 4 + k] = rbf(kv[k] * f);
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
      if (w < nq) {
        const int qi = w, b = qi / 3, t = t0 + b;
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
          if (lane == 0) { at.ml[qi][0] = m; at.ml[qi][1] = l; }
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
        } else if (w == 8 * nmt && lane < nq) {
          const int qi = lane, b = qi / 3, hh = qi % 3;
          if (t0 + b >= j0) {
            uint2* rec = c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH + ch) * 130;
            st_tagf(rec, at.ml[qi][0], tag);
            st_tagf(rec + 1, at.ml[qi][1], tag);
          }
        }
      }
      (void)b0;
      SUBMARK(66);
    } else {
      // ---------------------------------------------------------------- merge item (b, g)
      SUBMARK(70);
      const int mi = it - NR, b = mi / NKV, g = mi % NKV, t = t0 + b;
      const int nrc = (t + 64) / 64;
      const uint2* qr = qkvp + (long)b * QKVR;
      float gate = 0.f, sink = 0.f;
      uint2 rg = make_uint2(0u, 0u), rs = make_uint2(0u, 0u);
      if (tid < 3 * HD) {
        const int hh = tid / HD, d = tid % HD, h = 3 * g + hh;
        rg = ld_tag(qr + R_G + h * HD + d); rs = ld_tag(qr + R_S + h);
      }
      float* st = sm.u.mrg;   // [3][nrc][130]
      const int nrec = 3 * nrc * 130;
      for (int i0 = tid; i0 < nrec; i0 += NTH * MG_UNR) {
        uint2 v[MG_UNR];
#pragma unroll
        for (int q = 0; q < MG_UNR; q++) {
          const int i = min(i0 + q * NTH, nrec - 1);
          const int hh = i / (nrc * 130), rest = i % (nrc * 130);
          v[q] = ld_tag(c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH) * 130 + rest);
        }
#pragma unroll
        for (int q = 0; q < MG_UNR; q++) {
          const int i = i0 + q * NTH;
          if (i < nrec) {
            const int hh = i / (nrc * 130), rest = i % (nrc * 130);
            const uint2* p = c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH) * 130 + rest;
            while (v[q].y != tag) v[q] = ld_tag(p);
            st[hh * nrc * 130 + rest] = __uint_as_float(v[q].x);
          }
        }
      }
      if (tid < 3 * HD) {
        const int hh = tid / HD, d = tid % HD, h = 3 * g + hh;
        gate = __uint_as_float(wait_tag(qr + R_G + h * HD + d, rg, tag));
        sink = __uint_as_float(wait_tag(qr + R_S + h, rs, tag));
      }
      const int ewt = 3 * nrc * 130;
      __syncthreads();
      SUBMARK(71);
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
        if ((d & 1) == 0) st_tag(c.aoutt + (long)(par * MAXB + b) * (QW / 2) + ((3 * g + hh) * HD + d) / 2, pack_bf2(out, o1), tag);
      }
      SUBMARK(72);
    }
    __syncthreads();
  }
}

// o_proj + epilogue: attn_raw -> sconv -> attst
template <bool V> __device__ void ph_oproj(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = HB(hd), par = a & 1;
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
  // the epilogue's short-conv history (3 lags) and taps of this block's 16 rows, loaded before the poll below
  float* ost = reinterpret_cast<float*>(&sm.u.xs[2][0][0]);   // beyond the xo staging; [MAXB][16][3] then taps
  uint2* otp = reinterpret_cast<uint2*>(ost + MAXB * 16 * 3);
  float oh0 = 0.f, oh1 = 0.f, oh2 = 0.f;
  uint2 otap = make_uint2(0, 0);
  const int oht = threadIdx.x >> 4, ohr = threadIdx.x & 15;    // token, row of the strip
  if (blockIdx.x < 144 && oht < B) {
    // lags older than this pass's rows from the ring (c1 layout: slot = position % RING); in-pass lags by shuffles
    const int t = row_t<V>(hd, oht), nin = row_nin<V>(hd, oht), seq = row_seq<V>(hd, oht);
    const bf16* hb = c.ha + ((long)ap.slot * MAXB + seq) * RING * H + blockIdx.x * 16 + ohr;
    if (nin < 1) oh0 = ldcg_bf(hb + ((t - 1) & (RING - 1)) * H);
    if (nin < 2) oh1 = ldcg_bf(hb + ((t - 2) & (RING - 1)) * H);
    if (nin < 3) oh2 = ldcg_bf(hb + ((t - 3) & (RING - 1)) * H);
  }
  if (blockIdx.x < 144 && threadIdx.x < 16) otap = *reinterpret_cast<const uint2*>(L.attn_sc + (blockIdx.x * 16 + threadIdx.x) * 4);
  {
    const int n = B * QW / 2;
    const uint2* src = c.aoutt + (long)par * MAXB * (QW / 2);
    constexpr int OPU = V ? OP_UNR_V : OP_UNR_S;
    for (int i0 = threadIdx.x; i0 < n; i0 += NTH * OPU) {
      uint2 r[OPU];
#pragma unroll
      for (int q = 0; q < OPU; q++) r[q] = ld_tag(src + min(i0 + q * NTH, n - 1));
#pragma unroll
      for (int q = 0; q < OPU; q++) {
        const int i = i0 + q * NTH;
        if (i < n) {
          const int b = i / (QW / 2), k = (i % (QW / 2)) * 2;
          *reinterpret_cast<uint32_t*>(&sm.u.xo[b][k]) = wait_tag(src + i, r[q], TAGA(a));
        }
      }
    }
  }
  if (oht < B) { float* d = ost + (oht * 16 + ohr) * 3; d[0] = oh0; d[1] = oh1; d[2] = oh2; }
  if (threadIdx.x < 16) otp[threadIdx.x] = otap;
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
          // = sconv_static(x, lags, taps): lags of this pass's earlier rows by shuffles, older ones staged above
          const float* hv = ost + (tok * 16 + g + 8 * h) * 3;
          const bf16* w4 = reinterpret_cast<const bf16*>(&otp[g + 8 * h]);
          const float x1 = 1 <= nin ? xin[0][2 * h + cc] : hv[0];
          const float x2 = 2 <= nin ? xin[1][2 * h + cc] : hv[1];
          const float x3 = 3 <= nin ? xin[2][2 * h + cc] : hv[2];
          float sa = __fmul_rn(x3, bf(w4[0]));
          sa = __fadd_rn(sa, __fmul_rn(x2, bf(w4[1])));
          sa = __fadd_rn(sa, __fmul_rn(x1, bf(w4[2])));
          sa = __fadd_rn(sa, __fmul_rn(x, bf(w4[3])));
          y[2 * h + cc] = __fmul_rn(__fadd_rn(sa, x), 0.44721359549995793f);
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
            st_tag(c.attst + (long)(par * MAXB + tok) * NPAIR + row / 2, pack_bf2(y[2 * h + cc], yo[2 * h + cc]), TAGA(a));
          }
      });
  }
}

// mlp mix of application a, token b -> x (f32). Sources: blocks in slots 0..3, the boundary layer's old partial in
// slot 4, the partial (partial + attention output in f32, or the attention output alone) in slot 5.
template <bool V> __device__ void mlp_mix(PH_ARGS, int a, int b, float (*x)[EPT]) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  float v[6][EPT];
  const bool add_pin = !(a == 0 || ap.bnd);
  const int idx[6] = {a == 0 ? 0 : ap.blk[0], ap.blk[1], ap.blk[2], ap.blk[3], ap.pin, ap.pin};
  unsigned msk = (a == 0) ? 1u : (((1u << ap.nblk) - 1u) | (ap.bnd ? 16u : 0u));
  const float* qs[1] = {L.mq};
  float qv[1][EPT];
  load_queries<1>(qs, 1, qv);
  // the attention partial (newest source) is requested before waiting for the older ones (as load_vec_bf)
  const uint2* pa = c.attst + (long)((a & 1) * MAXB + b) * NPAIR;
  const int pa1 = EV(2) ? threadIdx.x + NTH : threadIdx.x;
  const uint2 ra0 = ld_tag(pa + threadIdx.x), ra1 = ld_tag(pa + pa1);
  load_srcs<6, (V ? LS_SEQ : LS_SEQ_S)>(c, hd, idx, msk | (add_pin ? 32u : 0u), b, v);
  float as[EPT];
  {
    const uint32_t x0 = wait_tag(pa + threadIdx.x, ra0, TAGA(a));
    const uint32_t x1 = EV(2) ? wait_tag(pa + pa1, ra1, TAGA(a)) : 0u;
    as[0] = bf_lo(x0); as[1] = bf_hi(x0); as[2] = bf_lo(x1); as[3] = bf_hi(x1);
  }
#pragma unroll
  for (int k = 0; k < EPT; k++) v[5][k] = add_pin ? v[5][k] + as[k] : as[k];
  depth_mix_q<6, 1>(v, msk | 32u, qv, 1, x, sm);
}

// (gated) mlp input of token b (f32) -> router input xr (bf16) + expert input x16 (f16 hi / lo) in smem
__device__ void mlp_in_x(PH_ARGS, int a, float (*xin)[EPT], int b, const float* wv) {   // wv: L.mlp_norm (loaded early)
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
    *reinterpret_cast<uint32_t*>(&sm.u.e.xr[b][el]) = yb;
    const float y0 = bf_lo(yb) * scl, y1 = bf_hi(yb) * scl;
    const __half h0 = __float2half_rn(y0), h1 = __float2half_rn(y1);
    sm.u.e.x16[b][0][el] = h0; sm.u.e.x16[b][0][el + 1] = h1;
    sm.u.e.x16[b][1][el] = __float2half_rn((y0 - __half2float(h0)) * 16777216.0f);
    sm.u.e.x16[b][1][el + 1] = __float2half_rn((y1 - __half2float(h1)) * 16777216.0f);
  }
  if (threadIdx.x == 0) sm.sc16[b] = ldexpf(1.0f, -sh);
}

// B >= 2: the designated block of token b publishes the router input row (bf16 pairs) and the f16 scale exponent
__device__ void mlp_publish(PH_ARGS, int a, float (*xin)[EPT], int b, const float* wv) {   // wv: L.mlp_norm (loaded early)
  const int par = a & 1;
  float s[1] = {0.f}, mx[1] = {0.f};
#pragma unroll
  for (int k = 0; k < EPT; k++) { s[0] += xin[0][k] * xin[0][k]; mx[0] = fmaxf(mx[0], fabsf(xin[0][k] * wv[k])); }
  block_sum_max<1, 1>(s, mx, sm);
  const float r = rsqrtf(s[0] / (float)H + 1e-6f);
  int e2 = 0;
  const float bound = mx[0] * r;
  frexpf(bound, &e2);
  const int sh = (bound > 0.f) ? 15 - e2 : 0;
  uint2* dst = c.xrt + (long)(par * MAXB + b) * NPAIR;
#pragma unroll
  for (int j = 0; j < 2; j++)
    if (EV(2 * j)) st_tag(dst + threadIdx.x + j * NTH, pack_bf2(xin[0][2 * j] * r * wv[2 * j], xin[0][2 * j + 1] * r * wv[2 * j + 1]), TAGA(a));
  if (threadIdx.x == 0) st_tag(c.sht + par * MAXB + b, (uint32_t)sh, TAGA(a));
}
// every block: router input rows + the expert input (f16 hi / lo, the same arithmetic as mlp_in_x)
template <bool VV> __device__ void mlp_poll(PH_ARGS, int a, int r0 = 0) {
  const int B = hd.B, par = a & 1;
  pollr<VV>(hd.B, c.xrt + ((long)par * MAXB + r0) * NPAIR, B - r0, [&](int r) { return &sm.u.e.xr[r0 + r][0]; }, TAGA(a));
  if (threadIdx.x >= r0 && threadIdx.x < B) {
    const int sh = (int)poll(c.sht + par * MAXB + threadIdx.x, TAGA(a));
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
      const uint32_t yb = *reinterpret_cast<const uint32_t*>(&sm.u.e.xr[b][el]);
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
#if RTPF
  if (HB(hd) == 1 && blockIdx.x < 8) {
    // this block's router strip (74 KB) into its L1 while the MLP-input mix waits for the attention output
    const uint8_t* p = c.L[ap.layer].wr + (long)blockIdx.x * 36 * 2048;
    for (int i = threadIdx.x; i < 36 * 2048 / 128; i += NTH) asm volatile("prefetch.global.L1 [%0];" :: "l"(p + i * 128));
  }
#endif
  if (ap.nmsg) return -1;   // the gate's D blocks compute the mix themselves
  if (hyb2<V>(hd)) {
    for (int b = 1 + (int)(gridDim.x - 1 - blockIdx.x); b < hd.B; b += gridDim.x) {
      float x[1][EPT], wv[EPT];
      load_w(c.L[ap.layer].mlp_norm, wv);
      mlp_mix<V>(c, hd, sm, a, b, x);
      mlp_publish(c, hd, sm, a, x, b, wv);
    }
    {
      float x[1][EPT], wv[EPT];
      load_w(c.L[ap.layer].mlp_norm, wv);
      mlp_mix<V>(c, hd, sm, a, 0, x);
      mlp_in_x(c, hd, sm, a, x, 0, wv);
    }
    mlp_poll<V>(c, hd, sm, a, 1);
    return 0;
  }
  if (desig<V>(hd)) {
    for (int b = dsel<V>(hd); b < hd.B; b += gridDim.x) {
      float x[1][EPT], wv[EPT];
      load_w(c.L[ap.layer].mlp_norm, wv);
      mlp_mix<V>(c, hd, sm, a, b, x);
      mlp_publish(c, hd, sm, a, x, b, wv);
    }
    mlp_poll<V>(c, hd, sm, a);
    return 0;
  }
  for (int b = 0; b < hd.B; b++) {
    float x[1][EPT], wv[EPT];
    load_w(c.L[ap.layer].mlp_norm, wv);
    mlp_mix<V>(c, hd, sm, a, b, x);
    mlp_in_x(c, hd, sm, a, x, b, wv);
  }
  __syncthreads();
  return 0;
}

__device__ void ph_router(PH_ARGS, int a) {
  const LayerW& L = c.L[c.A[a].layer];
  const int B = HB(hd), par = a & 1;
  for (int strip = blockIdx.x; strip < 8; strip += gridDim.x) {
    BSm bl{&sm.u.e.xr[0][0], HP, B};
    block_strips<NWARP, 1>(sm, 1, 36,
      [&](int, int u0, int nu, float (*acc)[4]) { piece_bf16(L.wr + (long)strip * 36 * 2048, u0, nu, bl, acc[0]); },
      [&](int, float (*acc)[4]) {
        const int lane = lane_id(), g = lane >> 2, t = lane & 3;
        for (int h = 0; h < 2; h++)
          for (int cc = 0; cc < 2; cc++) {
            const int tok = 2 * t + cc;
            if (tok < B) st_tagf(c.rlogt + (par * MAXB + tok) * NE + strip * 16 + g + 8 * h, acc[0][2 * h + cc], TAGA(a));
          }
      });
  }
}

// top-8 + union (every block, identical)
#ifndef RT_SEQ
#define RT_SEQ 0   // (s18: 0 is -10..-20 us per pass) verify kernel: router-logit polls one at a time per lane (all blocks poll the same logits)
#endif
template <bool V> __device__ void routing(const Ctx& c, const Hdr& hd, Smem& sm, int a) {
  const LayerW& L = c.L[c.A[a].layer];
  const int B = HB(hd);
  const int w = threadIdx.x >> 5, lane = lane_id();
  if (threadIdx.x < NE) { sm.ef13[threadIdx.x] = L.mx ? 1.0f : L.f13[threadIdx.x]; sm.ef2[threadIdx.x] = (L.mx ? 1.0f : L.f2[threadIdx.x]) * 0.0625f; }
  if (threadIdx.x < 4) sm.umask[threadIdx.x] = 0u;
  if (threadIdx.x < NE / 4) sm.tokm[threadIdx.x] = 0u;
  __syncthreads();
  if (w < B) {
    float lg[4], sel[4];
    uint2 rl[4];
    const bool seq = V && RT_SEQ;
    if (!seq) {
#pragma unroll
      for (int k = 0; k < 4; k++) rl[k] = ld_tag(c.rlogt + ((a & 1) * MAXB + w) * NE + lane + 32 * k);
    }
    float rb[4];
#pragma unroll
    for (int k = 0; k < 4; k++) rb[k] = L.rbias[lane + 32 * k];   // before the waits
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const int e = lane + 32 * k;
      if (seq) rl[k] = ld_tag(c.rlogt + ((a & 1) * MAXB + w) * NE + e);
      lg[k] = __uint_as_float(wait_tag(c.rlogt + ((a & 1) * MAXB + w) * NE + e, rl[k], TAGA(a)));
      sel[k] = sigm(lg[k]) + rb[k];
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
      atomicOr(&sm.tokm[myid >> 2], 1u << (8 * (myid & 3) + w));
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
    __syncwarp();
#ifdef MKPROF
    if (blockIdx.x == 0) SUBMARK(1000 + n);
#endif
    if (B <= 4) {   // no expert can have more than 4 tokens
      if (lane == 0) sm.nU = n;
      for (int u = lane; u < n; u += 32) {
        const int e = sm.uexp[u];
        unsigned mk = (sm.tokm[e >> 2] >> (8 * (e & 3))) & 0xffu;   // tokens in ascending order
        int cnt = 0;
        while (mk) { sm.utok[u][cnt++] = __ffs(mk) - 1; mk &= mk - 1; }
        sm.uncnt[u] = cnt;
      }
    } else {
    // union entries: an expert with more than 4 tokens becomes two entries (tokens 0-3, 4-7 in token order), so
    // every expert GEMV runs one n-group (the per-token arithmetic is the same: n-groups are independent MMAs)
    int e2[2], cnt2[2]; unsigned tk2[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int u = lane + 32 * h;
      e2[h] = u < n ? sm.uexp[u] : -1; cnt2[h] = 0; tk2[h] = 0u;
      if (u < n) {
        unsigned mk = (sm.tokm[e2[h] >> 2] >> (8 * (e2[h] & 3))) & 0xffu;   // tokens in ascending order
        while (mk) { tk2[h] |= (unsigned)(__ffs(mk) - 1) << (4 * cnt2[h]); cnt2[h]++; mk &= mk - 1; }
      }
    }
    __syncwarp();
    const unsigned lt = (1u << lane) - 1u;
    const unsigned s0 = __ballot_sync(0xffffffff, cnt2[0] > 4), s1 = __ballot_sync(0xffffffff, cnt2[1] > 4);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      if (e2[h] < 0) continue;
      const int pos = lane + 32 * h + (h == 0 ? __popc(s0 & lt) : __popc(s0) + __popc(s1 & lt));
      const int c0 = min(cnt2[h], 4);
      sm.uexp[pos] = e2[h]; sm.uncnt[pos] = c0;
      for (int i = 0; i < c0; i++) sm.utok[pos][i] = (tk2[h] >> (4 * i)) & 15u;
      if (cnt2[h] > 4) {
        sm.uexp[pos + 1] = e2[h]; sm.uncnt[pos + 1] = cnt2[h] - 4;
        for (int i = 4; i < cnt2[h]; i++) sm.utok[pos + 1][i - 4] = (tk2[h] >> (4 * i)) & 15u;
      }
    }
    if (lane == 0) sm.nU = n + __popc(s0) + __popc(s1);
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
__device__ __forceinline__ void w13_run(const Ctx& c, Smem& sm, const LayerW& L, int u, int strip, int u0, int nu, float (*acc)[4], bool fake = false, const uint8_t* slot = nullptr) {
  const int g = lane_id() >> 2;
  const int e = sm.uexp[u], nt = sm.uncnt[u];
  const __half* colp[3];
#pragma unroll
  for (int ng = 0; ng < 2; ng++) {
    const int li = 4 * ng + (g >> 1), part = g & 1;
    colp[ng] = li < nt ? &sm.u.e.x16[sm.utok[u][li]][part][0] : nullptr;
  }
  colp[2] = nullptr;
  if (slot) { piece_fp4s<NG, W13_UNR>(slot, u0, nu, colp, acc); return; }
  if (!L.mx) {
    if (ROLL && W13_ROLL > 0 && nu % W13_ROLL == 0) piece_fp4_roll<NG, (W13_ROLL > 0 ? W13_ROLL : 1)>(L.w13 + ((long)e * 144 + strip) * 36 * 576, u0, nu, colp, acc);
    else piece_fp4<NG, false, W13_UNR>(L.w13 + ((long)e * 144 + strip) * 36 * 576, u0, nu, colp, acc, 0u, fake);
  } else piece_fp8<NG, false>(L.w13 + ((long)e * 144 + strip) * 36 * 1056, u0, nu, colp, acc);
}
__device__ void w13_epi(const Ctx& c, const Hdr& hd, Smem& sm, const LayerW& L, int a, int u, int strip, int NGN, float (*acc)[4]) {
  const int lane = lane_id(), g = lane >> 2, t = lane & 3, par = a & 1;
  const int e = sm.uexp[u], nt = sm.uncnt[u];
  const float fac = sm.ef13[e];
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
  unsigned char* ring = w < 9 ? reinterpret_cast<unsigned char*>(&sm.u.e.xr[0][0]) + w * W13_RING
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
#ifndef MOE_PF
#define MOE_PF 0   // verify passes: bulk L2 prefetch of this block's expert items right after routing (1: w13 from
                   // round MOE_PF_R0 on + w2, 2: w13 only, 3: w2 only)
#endif
#ifndef MOE_PF_R0
#define MOE_PF_R0 1
#endif
#ifndef MOE_PF_MIN
#define MOE_PF_MIN 2   // ... from this many rows
#endif
  if (V && MOE_PF && hd.B >= MOE_PF_MIN) {
    const int nw13 = (T - (int)blockIdx.x + G - 1) / G;
    if (MOE_PF != 3)
      for (int k = MOE_PF_R0 * PER + (int)threadIdx.x; k < nw13; k += NTH) {
        const int it = (int)blockIdx.x + k * G, e = sm.uexp[it / 144], strip = it % 144;
        pf_l2(L.w13 + ((long)e * 144 + strip) * 36 * (L.mx ? 1056 : 576), 36L * (L.mx ? 1056 : 576));
      }
    if (MOE_PF != 2) {
      // w2 items of this block: the same contiguous range ph_w2a takes without the work balance (all entries <= 4 tokens)
      const int ib = (int)((long)T * blockIdx.x / G), ie = (int)((long)T * (blockIdx.x + 1) / G);
      for (int it = ib + (int)threadIdx.x; it < ie; it += NTH) {
        const int e = sm.uexp[it / 144], strip = it % 144;
        pf_l2(L.w2 + ((long)e * 144 + strip) * 18 * (L.mx ? 1056 : 576), 18L * (L.mx ? 1056 : 576));
      }
    }
  }
#ifndef W13_CONTIG
#define W13_CONTIG 0   // verify passes: each block takes a contiguous range of (expert, strip) items (contiguous weights)
#endif
  const bool contig = V && W13_CONTIG;
  const int cb = contig ? (int)((long)T * blockIdx.x / G) : 0, ce = contig ? (int)((long)T * (blockIdx.x + 1) / G) : 0;
  const int mine = contig ? ce - cb : (T - (int)blockIdx.x + G - 1) / G;      // items blockIdx.x + k G (or cb + k)
  auto itm = [&](int k) { return contig ? cb + k : (int)blockIdx.x + k * G; };
#ifndef W13_TMA
#define W13_TMA 0   // verify passes, NVFP4 layers: items after the first round (k >= PER) are bulk-copied into free shared
                    // memory at the start of the phase (router input rows, expert-input rows >= K, VSMEM_EXTRA beyond Smem)
#endif
  // slot table: every thread computes the same; slot j holds item PER + j
  constexpr int SLOTB = 36 * 576;
  int nsl = 0;
  const uint8_t* slots[8];
  if (V && W13_TMA && !L.mx && mine > PER) {
    auto add = [&](const uint8_t* p0, int bytes) { for (int o = 0; o + SLOTB <= bytes && nsl < 8; o += SLOTB) slots[nsl++] = p0 + o; };
    add(reinterpret_cast<const uint8_t*>(&sm) + ((sizeof(Smem) + 127) & ~(size_t)127), VSMEM_EXTRA_V);
    add(reinterpret_cast<const uint8_t*>(&sm.u.e.xr[0][0]), (int)sizeof(sm.u.e.xr));
    add(reinterpret_cast<const uint8_t*>(&sm.u.e.x16[hd.B][0][0]), (MAXB - hd.B) * (int)sizeof(sm.u.e.x16[0]));
    nsl = min(nsl, mine - PER);
    if (nsl > 0) {
      fence_proxy_async();   // earlier generic accesses of these bytes before the async-proxy writes
      __syncthreads();
      if (threadIdx.x == 0) {
        for (int j = 0; j < nsl; j++) mbar_init(&sm.tbar[j], 1);
        mbar_fence_init();
        for (int j = 0; j < nsl; j++) {
          const int it = itm(PER + j);
          mbar_expect_tx(&sm.tbar[j], SLOTB);
          tma_g2s(const_cast<uint8_t*>(slots[j]), L.w13 + ((long)sm.uexp[it / 144] * 144 + it % 144) * 36 * 576, SLOTB, &sm.tbar[j]);
        }
      }
      __syncthreads();
    }
  }
#ifdef MKPROF
  const bool w13m = V && (threadIdx.x & 31) == 0 && (threadIdx.x >> 5) == 0;   // warp 0 marks its rounds
  if (V && c.prof && blockIdx.x == 0 && threadIdx.x == 0) {   // routing statistics of this application (record 2900 + a)
    int nd = 0, mx = 0;
    int run = 0;
    for (int u = 0; u < nU; u++) { if (u == 0 || sm.uexp[u] != sm.uexp[u - 1]) { nd++; run = 0; } run += sm.uncnt[u]; mx = max(mx, run); }
    unsigned long long* r_ = (unsigned long long*)c.prof + 2ull * (2900 + a);
    r_[0] = 0; r_[1] = (unsigned long long)(nU | (nd << 8) | (mx << 16));
  }
#endif
#ifndef W13_STAG
#define W13_STAG 0   // verify passes with more than STAG_MIN rounds: groups >= PER / 2 start W13_STAG ns later (stagger loads)
#endif
#ifndef STAG_MIN
#define STAG_MIN 2
#endif
  if (V && W13_STAG > 0 && mine > (STAG_MIN - 1) * PER && (threadIdx.x >> 5) / WG >= PER / 2) __nanosleep(W13_STAG);
#ifndef W13X2
#define W13X2 0   // verify passes, NVFP4, more than one round: every group takes two items per round, each warp its
                  // 12-unit chain of both in two batches of 6 + 6 units (two independent accumulator chains)
#endif
  if (V && W13X2 && !L.mx && mine > PER) {
    const int w = threadIdx.x >> 5, q = w / WG, wl = w - q * WG, lane = lane_id();
    auto cols = [&](int it, const __half** colp) {
      const int u = it / 144, nt = sm.uncnt[u], g = lane >> 2;
#pragma unroll
      for (int ng = 0; ng < 2; ng++) {
        const int li = 4 * ng + (g >> 1), part = g & 1;
        colp[ng] = li < nt ? &sm.u.e.x16[sm.utok[u][li]][part][0] : nullptr;
      }
      colp[2] = nullptr;
    };
    for (int r0 = 0; r0 < mine; r0 += 2 * PER) {
      const int kA = contig ? r0 + 2 * q : r0 + q, kB = contig ? r0 + 2 * q + 1 : r0 + q + PER;   // contiguous: adjacent strips
      const bool hA = kA < mine, hB = kB < mine;
      const int itA = itm(hA ? kA : 0), itB = itm(hB ? kB : (hA ? kA : 0));
      float accA[1][4] = {{0.f, 0.f, 0.f, 0.f}}, accB[1][4] = {{0.f, 0.f, 0.f, 0.f}};
      if (hA) {
        const __half* ca[3]; const __half* cb[3];
        cols(itA, ca); cols(itB, cb);
        const uint8_t* wa = L.w13 + ((long)sm.uexp[itA / 144] * 144 + itA % 144) * 36 * 576;
        const uint8_t* wb = L.w13 + ((long)sm.uexp[itB / 144] * 144 + itB % 144) * 36 * 576;
        piece_fp4_x2<6>(wa, wb, hB, 12 * wl, ca, cb, accA, accB);
        piece_fp4_x2<6>(wa, wb, hB, 12 * wl + 6, ca, cb, accA, accB);
      }
      for (int k2 = 0; k2 < 2; k2++) {   // block_strips' reduction (0 + warp 0 + warp 1 + warp 2), item A then B
        const float* ac = k2 ? accB[0] : accA[0];
        sm.red4[w][lane] = make_float4(ac[0], ac[1], ac[2], ac[3]);
        grp_sync(1 + q, WG * 32);
        float tot[1][4] = {{0.f, 0.f, 0.f, 0.f}};
        const bool has = k2 ? hB : hA;
        if (wl == 0 && has) {
#pragma unroll
          for (int k = 0; k < WG; k++) {
            const float4 v = sm.red4[w + k][lane];
            tot[0][0] += v.x; tot[0][1] += v.y; tot[0][2] += v.z; tot[0][3] += v.w;
          }
        }
        grp_sync(1 + q, WG * 32);
        if (wl == 0 && has) { const int it = k2 ? itB : itA; w13_epi(c, hd, sm, L, a, it / 144, it % 144, 1, tot); }
      }
    }
    return;
  }
  for (int r0 = 0; r0 < mine; r0 += PER) {
#ifdef MKPROF
    if (w13m && c.prof) { unsigned n_ = atomicAdd(&sm.subn, 1u); if (n_ < 1024) { unsigned long long* r_ = (unsigned long long*)c.prof + 2ull * ((unsigned long long)blockIdx.x * 4096 + 3072 + n_); r_[0] = mkprof_gtime(); r_[1] = ((unsigned long long)clock64() << 16) | (90 + r0 / PER); } }
#endif
    const int ns = min(PER, mine - r0);
#ifndef W13_L1PF
#define W13_L1PF 0   // verify passes: each warp prefetches the first W13_L1PF units of its slice of the next round into L1
#endif
    if (V && W13_L1PF > 0 && r0 + PER < mine) {
      const int w = threadIdx.x >> 5, q = w / WG, wl = w - q * WG, k = r0 + PER + q;
      if (q < PER && k < mine) {
        const int it = itm(k), ub = L.mx ? 1056 : 576;
        const uint8_t* pb = L.w13 + (((long)sm.uexp[it / 144] * 144 + it % 144) * 36 + 12 * wl) * ub;
        const int bytes = W13_L1PF * ub;
        for (int o = lane_id() * 128; o < bytes; o += 32 * 128) asm volatile("prefetch.global.L1 [%0];" :: "l"(pb + o));
      }
    }
    // one n-group: union entries have at most 4 tokens (routing splits larger experts)
    auto run1 = [&](int q, int u0, int nu, float (*acc)[4]) {
      const int k = r0 + q, it = itm(k), j = k - PER;
      const uint8_t* sl = nullptr;
      if (V && W13_TMA && j >= 0 && j < nsl) { mbar_wait(&sm.tbar[j], 0); sl = slots[j]; }
      w13_run<1, V>(c, sm, L, it / 144, it % 144, u0, nu, acc, V && r0 >= PER, sl);
    };
    auto epi = [&](int q, float (*acc)[4]) { const int it = itm(r0 + q); w13_epi(c, hd, sm, L, a, it / 144, it % 144, 1, acc); };
    block_strips<WG, 1, GRP_EXP>(sm, ns, 36, run1, epi);
  }
#ifdef MKPROF
  if (w13m && c.prof) { unsigned n_ = atomicAdd(&sm.subn, 1u); if (n_ < 1024) { unsigned long long* r_ = (unsigned long long*)c.prof + 2ull * ((unsigned long long)blockIdx.x * 4096 + 3072 + n_); r_[0] = mkprof_gtime(); r_[1] = ((unsigned long long)clock64() << 16) | 99; } }
#endif
}

// w2, part 1: work items (union expert u, output strip) expert-major in block-contiguous ranges (at most two experts
// per block, their inputs staged in shared memory once); 8 items per round, 3 warps x 6 units each (a fixed split).
// Each item publishes its contribution per token as tagged f32 rows w2ct[par][token][rank of the expert][row].
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
#ifndef W2V
#define W2V 0   // verify passes: one warp per w2a item (0: the step kernel's 3-warp items, 8 per round; faster at every K since W2S_RS)
#endif
#ifndef W2V_MIN
#define W2V_MIN 1   // ... from this many rows
#endif
  const bool w2v = (V && W2V && hd.B >= W2V_MIN) || (W2_STEP && M);
  if (w2v && W2_BAL) for (int u = 0; u < nU; u++) big |= sm.uncnt[u] > 4;
  if (w2v && W2_BAL && big) {
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
    constexpr int SB = V ? W2S_UNR_V : W2S_UNR_S;
    for (int i0 = 2 * threadIdx.x; i0 < nw; i0 += 2 * NTH * SB) {
      uint2 r[SB][2]; const uint2* pp[SB];
#pragma unroll
      for (int q = 0; q < SB; q++) {
        const int i = min(i0 + q * 2 * NTH, nw - 2);
        const int j = i / IE, w0 = i % IE;
        const int uu = j < n0 ? u0 : u1, jj = j < n0 ? j : j - n0;
        pp[q] = c.act16t + (((long)(par * MAXU + uu) * MAXB + jj) * 2) * (IE / 2) + w0;
        ld_tag2(pp[q], r[q][0], r[q][1]);
      }
#pragma unroll
      for (int q = 0; q < SB; q++) {
        const int i = i0 + q * 2 * NTH;
        if (i >= nw) continue;
        const int jw = i / IE, ww = i % IE;   // pair, word within the pair's (hi, lo) rows
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
      uint2* dst = c.w2ct + (((long)(par * MAXB + tok) * TOPK + k) * H) + strip * 16;
      st_tagf(dst + g, (acc[ng][0] + acc[ng][1] * 5.9604644775390625e-08f) * fac, TAGA(a));
      st_tagf(dst + g + 8, (acc[ng][2] + acc[ng][3] * 5.9604644775390625e-08f) * fac, TAGA(a));
    }
  };
#ifndef W2_BPC
#define W2_BPC 2   // verify w2a (NVFP4): units of each chain loaded per batch (1, 2, 3, 6)
#endif
#ifndef W2_1W_MIN
#define W2_1W_MIN 0   // one warp per item only from this many items per block (12 / 16 / 24 measured slower at every K >= 2)
#endif
  if (w2v && ie - ib >= W2_1W_MIN) {   // verify passes: one warp per item of the block's range, the whole strip as three chains
    for (int it = ib + (threadIdx.x >> 5); it < ie; it += NWARP) {
#ifndef W2_L1PF
#define W2_L1PF 0   // verify passes: each warp prefetches its next w2a item into L1 before streaming this one
#endif
      if (W2_L1PF && it + NWARP < ie) {
        const int ub = L.mx ? 1056 : 576, itn = it + NWARP;
        const uint8_t* pb = L.w2 + ((long)sm.uexp[itn / 144] * 144 + itn % 144) * 18 * ub;
        for (int o = lane_id() * 128; o < 18 * ub; o += 32 * 128) asm volatile("prefetch.global.L1 [%0];" :: "l"(pb + o));
      }
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
        if (!L.mx) piece3c<2, false, 6, W2_BPC>(L.w2 + ((long)e * 144 + strip) * 18 * 576, colp, acc);
        else piece3c<2, true, 6, 1>(L.w2 + ((long)e * 144 + strip) * 18 * 1056, colp, acc);
        sum3<2>(acc, tot);
      } else {
        float acc[3][1][4] = {};
        if (!L.mx) piece3c<1, false, 6, W2_BPC>(L.w2 + ((long)e * 144 + strip) * 18 * 576, colp, acc);
        else piece3c<1, true, 6, 1>(L.w2 + ((long)e * 144 + strip) * 18 * 1056, colp, acc);
        sum3<1>(acc, tot);
      }
      epi(it, tot);
    }
    SUBMARK(43);
    return;
  }
#ifndef W2X2
#define W2X2 1   // verify passes, NVFP4: every 3-warp group takes two items per round (16 per round), each warp the same
                 // 6-unit chain of both in one batch; reductions and epilogues as block_strips, item by item
#endif
  if (V && W2X2 && !L.mx) {
    const int w = threadIdx.x >> 5, q = w / WG, wl = w - q * WG;
    auto cols = [&](int it, const __half** colp) {
      const int u = it / 144, nt = sm.uncnt[u], j0 = (u == u0) ? 0 : n0, g = lane >> 2;
#pragma unroll
      for (int ng = 0; ng < 2; ng++) {
        const int li = 4 * ng + (g >> 1), part = g & 1;
        colp[ng] = li < nt ? w2sb + ((long)(j0 + li) * 2 + part) * W2S_RS : nullptr;
      }
      colp[2] = nullptr;
    };
    for (int r0 = ib; r0 < ie; r0 += 2 * PER) {
      const int itA = r0 + q, itB = r0 + q + PER;
      const bool hA = itA < ie, hB = itB < ie;
      float accA[1][4] = {{0.f, 0.f, 0.f, 0.f}}, accB[1][4] = {{0.f, 0.f, 0.f, 0.f}};
      if (hA) {
        const __half* ca[3]; const __half* cb[3];
        cols(itA, ca); cols(hB ? itB : itA, cb);
        const uint8_t* wa = L.w2 + ((long)sm.uexp[itA / 144] * 144 + itA % 144) * 18 * 576;
        const uint8_t* wb = L.w2 + ((long)sm.uexp[(hB ? itB : itA) / 144] * 144 + (hB ? itB : itA) % 144) * 18 * 576;
        piece_fp4_x2<6>(wa, wb, hB, 6 * wl, ca, cb, accA, accB);
      }
      // block_strips' reduction, item A then item B (tot = 0 + warp 0 + warp 1 + warp 2 of the group)
      for (int k2 = 0; k2 < 2; k2++) {
        const float* ac = k2 ? accB[0] : accA[0];
        sm.red4[w][lane] = make_float4(ac[0], ac[1], ac[2], ac[3]);
        grp_sync(1 + q, WG * 32);
        float tot[1][4] = {{0.f, 0.f, 0.f, 0.f}};
        if (wl == 0 && (k2 ? hB : hA)) {
#pragma unroll
          for (int k = 0; k < WG; k++) {
            const float4 v = sm.red4[w + k][lane];
            tot[0][0] += v.x; tot[0][1] += v.y; tot[0][2] += v.z; tot[0][3] += v.w;
          }
        }
        grp_sync(1 + q, WG * 32);
        if (wl == 0 && (k2 ? hB : hA)) epi(k2 ? itB : itA, tot);
      }
    }
    SUBMARK(42);
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
      if (!L.mx) piece_fp4<1, false>(L.w2 + ((long)e * 144 + strip) * 18 * 576, uu0, nu, colp, acc);
      else piece_fp8<1, false>(L.w2 + ((long)e * 144 + strip) * 18 * 1056, uu0, nu, colp, acc);
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
  const int B = HB(hd), par = a & 1;
  for (int strip = blockIdx.x; strip < 144; strip += gridDim.x) {
    const bool addp = !(a == 0 || ap.bnd);
    const int et = threadIdx.x >> 4, er = threadIdx.x & 15;    // token et, row er
    const int row = strip * 16 + er;
    float np_ = 0.f;
    const int t = row_t<V>(hd, min(et, MAXB - 1)), nin = row_nin<V>(hd, min(et, MAXB - 1)), seq = row_seq<V>(hd, min(et, MAXB - 1));
    bf16* hb = c.hm + ((long)ap.slot * MAXB + seq) * RING * H + row;
    float x = 0.f, xl[3] = {0.f, 0.f, 0.f};
    uint2 ra = make_uint2(0u, 0u), rp = make_uint2(0u, 0u), tw = make_uint2(0u, 0u);
    if (et < B) {
      // independent loads first: history, taps, attention output, old partial; then the expert rows
#pragma unroll
      for (int l = 1; l <= 3; l++) if (l > nin) xl[l - 1] = ldcg_bf(hb + ((t - l) & (RING - 1)) * H);
      tw = *reinterpret_cast<const uint2*>(L.mlp_sc + row * 4);
      ra = ld_tag(c.attst + (long)(par * MAXB + et) * NPAIR + row / 2);
      rp = ld_tag(c.pbt + ((long)(addp ? ap.pin : 0) * MAXB + et) * NPAIR + row / 2);
      uint2 rk[TOPK];
      const uint2* src = c.w2ct + ((long)(par * MAXB + et) * TOPK) * H + row;
      constexpr int WB = V ? W2B_UNR_V : TOPK;   // loads issued per batch (the sum stays in expert-rank order)
      float s = 0.f;
#pragma unroll
      for (int k0 = 0; k0 < TOPK; k0 += WB) {
#pragma unroll
        for (int k = k0; k < k0 + WB; k++) rk[k] = ld_tag(src + (long)k * H);
#pragma unroll
        for (int k = k0; k < k0 + WB; k++) s += __uint_as_float(wait_tag(src + (long)k * H, rk[k], TAGA(a)));
      }
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
      const uint32_t av = wait_tag(c.attst + (long)(par * MAXB + et) * NPAIR + row / 2, ra, TAGA(a));
      const float as = (row & 1) ? bf_hi(av) : bf_lo(av);
      if (!addp) np_ = as + ms;
      else {
        const uint32_t pv = wait_tag(c.pbt + ((long)ap.pin * MAXB + et) * NPAIR + row / 2, rp, TAGP(ap.pin));
        np_ = (((row & 1) ? bf_hi(pv) : bf_lo(pv)) + as) + ms;
      }
    }
    const float other = __shfl_down_sync(0xffffffff, np_, 1);
    if (et < B && (er & 1) == 0) st_tag(c.pbt + ((long)ap.out * MAXB + et) * NPAIR + row / 2, pack_bf2(np_, other), TAGP(ap.out));
  }
}

// final: output mix -> norm -> unembed -> logits, argmax
#ifndef UN8_PRE
#define UN8_PRE 1   // certified pass 1: per-row parameters loaded before the strip's weight loads
#endif
#ifndef FINAL_ATTR
#define FINAL_ATTR   // the final phase compiled out of line: its code does not move the other phases' registers
#endif
#ifndef FIN_DES
#define FIN_DES 3   // verify passes of at least this many rows: designated blocks mix the final rows and publish them
#endif
#ifndef UN8_DIAG
#define UN8_DIAG 0   // timing only: skip the exact recompute (wrong tokens)
#endif
#ifndef UN8
#define UN8 1   // verify passes without logits output: certified argmax (2: int8 tensor-core pass 1 (s18): bit-exact tokens, final phase -50 us but needs FINAL_ATTR __noinline__ (+30..40 us elsewhere): net 0; 1: c1's INT8 copy
                // decoded to bf16, no gain; the engine enables it with MK_UN8)
#endif
template <bool V> FINAL_ATTR __device__ void ph_final(PH_ARGS, unsigned& nbar, unsigned base) {
  const int B = HB(hd);
  if (V && B >= FIN_DES) {
    for (int b = dsel<V>(hd); b < B; b += gridDim.x) {
      float v[5][EPT], x[1][EPT], wv[1][EPT];
      const int idx[5] = {c.fin[0], c.fin[1], c.fin[2], c.fin[3], c.fin_pin};
      const unsigned msk = ((1u << c.nfin) - 1u) | 16u;
      load_w(c.fnorm, wv[0]);
      const float* qs[1] = {c.oq};
      float qv[1][EPT];
      load_queries<1>(qs, 1, qv);
      load_srcs<5, (V ? LS_SEQ : LS_SEQ_S)>(c, hd, idx, msk, b, v);
      depth_mix_q<5, 1>(v, msk, qv, 1, x, sm);
      uint2* dst[1] = {c.xft + (long)b * NPAIR};
      norm_publish<1>(x, wv, 1, dst, TAGP(60), sm);   // the same arithmetic as rmsnorm_store
    }
    pollr<V>(hd.B, c.xft, B, [&](int r) { return &sm.u.xs[0][r][0]; }, TAGP(60));
  } else
  for (int b = 0; b < B; b++) {
    float v[5][EPT], x[1][EPT];
    const int idx[5] = {c.fin[0], c.fin[1], c.fin[2], c.fin[3], c.fin_pin};
    const unsigned msk = ((1u << c.nfin) - 1u) | 16u;
    const float* qs[1] = {c.oq};
    float qv[1][EPT], wv[EPT];
    load_queries<1>(qs, 1, qv);
    load_w(c.fnorm, wv);
    load_srcs<5>(c, hd, idx, msk, b, v);
    depth_mix_q<5, 1>(v, msk, qv, 1, x, sm);
    {   // = rmsnorm_store(x, fnorm) with the weights loaded above
      float s1[1] = {0.f};
#pragma unroll
      for (int k = 0; k < EPT; k++) s1[0] += x[0][k] * x[0][k];
      block_sum<1>(s1, sm);
      const float r = rsqrtf(s1[0] / (float)H + 1e-6f);
#pragma unroll
      for (int j = 0; j < 2; j++) {
        if (!EV(2 * j)) continue;
        const int e = EL(2 * j);
        float y0 = x[0][2 * j] * r, y1 = x[0][2 * j + 1] * r;
        y0 = y0 * wv[2 * j]; y1 = y1 * wv[2 * j + 1];
        *reinterpret_cast<uint32_t*>(&sm.u.xs[0][b][0] + e) = pack_bf2(y0, y1);
      }
    }
  }
  __syncthreads();
  if (V && blockIdx.x == 0)   // verify pass: the final-norm rows are a draft feature
    for (int i = threadIdx.x; i < B * (H / 8); i += NTH)
      reinterpret_cast<uint4*>(c.fout)[i] = *reinterpret_cast<const uint4*>(&sm.u.xs[0][i / (H / 8)][(i % (H / 8)) * 8]);
  const int w = threadIdx.x >> 5, lane = lane_id(), g = lane >> 2, t = lane & 3;
  unsigned long long best[2] = {0ull, 0ull};
  const bool cert = V && UN8 && c.un8 && !hd.logits;
  float zl0[2] = {-1e30f, -1e30f};
#if UN8 == 2
  if (cert) {
    // pass 1 (certified IMMA argmax): per token b, h = t_b v + e (v = rint(h / t_b), |v| <= 32767, |e| <= 0.51 t_b),
    // v = 256 hi + lo (hi s8, lo u8); per vocabulary row r, w = s_r q + d (q int8). Exact int32 tensor-core sums
    // acc_hi = q . hi, acc_lo = q . lo give A = s_r t_b (256 acc_hi + acc_lo) (double), and
    // |acc_exact - A| <= s_r |q|_1 0.51 t_b + min(max|d| |h|_1, ||d||_2 ||h||_2) + 1e-4 |w|_1 max|h| + 1e-9 |A|
    // (the last-but-one term: the fp32 tensor-core accumulation of the exact path; Cauchy-Schwarz for d . h).
    // z bounds through 30 tanh(./30) with 0.06 for the approximate tanh; per strip the largest upper bound per token
    // goes to zhi[tok][strip], per token the largest lower bound to zlo (ordered integers, atomicMax).
    constexpr int ARS = H + 16;
    int8_t* ahi = reinterpret_cast<int8_t*>(&sm.u.xs[1][0][0]);
    uint8_t* alo = reinterpret_cast<uint8_t*>(&sm.u.xs[2][0][0]);
    float hs[2 * MAXB], hm[MAXB];
#pragma unroll
    for (int b = 0; b < MAXB; b++) {
      float s1 = 0.f, s2 = 0.f, mx = 0.f;
      if (b < B) for (int i = threadIdx.x; i < H; i += NTH) { const float x = fabsf(bf(sm.u.xs[0][b][i])); s1 += x; s2 += x * x; mx = fmaxf(mx, x); }
      hs[b] = s1; hs[MAXB + b] = s2; hm[b] = mx;
    }
    block_sum_max<2 * MAXB, MAXB>(hs, hm, sm);
    for (int i = threadIdx.x; i < MAXB * H; i += NTH) {
      const int b = i / H, kk = i % H;
      int v = 0;
      if (b < B && hm[b] > 0.f) v = __float2int_rn(bf(sm.u.xs[0][b][kk]) * (32767.0f / hm[b]));
      v = max(-32767, min(32767, v));
      ahi[b * ARS + kk] = (int8_t)(v >> 8);
      alo[b * ARS + kk] = (uint8_t)(v & 255);
    }
    __syncthreads();
    double tb[2], h1[2], h2[2], hmx[2];
#pragma unroll
    for (int cc = 0; cc < 2; cc++) {
      const int tok = min(2 * t + cc, MAXB - 1);
      tb[cc] = (double)hm[tok] / 32767.0;
      h1[cc] = (double)hs[tok] * 1.001;
      h2[cc] = sqrt((double)hs[MAXB + tok]) * 1.001;
      hmx[cc] = (double)hm[tok];
    }
    const int nrow = UN_STRIPS * 16;
    for (int it = blockIdx.x + gridDim.x * w; it < UN_STRIPS; it += gridDim.x * NWARP) {
      int dh[4] = {0, 0, 0, 0}, dl[4] = {0, 0, 0, 0};
      const uint4* wq = reinterpret_cast<const uint4*>(c.un8) + (long)it * 72 * 32 + lane;
      const int8_t* bh = ahi + g * ARS + 4 * t;
      const uint8_t* bl8 = alo + g * ARS + 4 * t;
#ifndef UN8_B
#define UN8_B 12   // certified pass 1: k32 steps of loads in flight per warp
#endif
      // the strip's per-row parameters before the weight loads (independent of them)
      float prm[2][5];
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int j = 0; j < 5; j++) prm[h][j] = UN8_PRE ? __ldg(c.un8s + (long)j * nrow + it * 16 + g + 8 * h) : 0.f;
#pragma unroll 1
      for (int ks0 = 0; ks0 < 72; ks0 += UN8_B) {
        uint4 a4[UN8_B];
#pragma unroll
        for (int q = 0; q < UN8_B; q++) a4[q] = ldg_nc(wq + (ks0 + q) * 32);
#pragma unroll
        for (int q = 0; q < UN8_B; q++) {
          const int k0 = 32 * (ks0 + q);
          const uint32_t b0h = *reinterpret_cast<const uint32_t*>(bh + k0), b1h = *reinterpret_cast<const uint32_t*>(bh + k0 + 16);
          const uint32_t b0l = *reinterpret_cast<const uint32_t*>(bl8 + k0), b1l = *reinterpret_cast<const uint32_t*>(bl8 + k0 + 16);
          asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       : "+r"(dh[0]), "+r"(dh[1]), "+r"(dh[2]), "+r"(dh[3]) : "r"(a4[q].x), "r"(a4[q].y), "r"(a4[q].z), "r"(a4[q].w), "r"(b0h), "r"(b1h));
          asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       : "+r"(dl[0]), "+r"(dl[1]), "+r"(dl[2]), "+r"(dl[3]) : "r"(a4[q].x), "r"(a4[q].y), "r"(a4[q].z), "r"(a4[q].w), "r"(b0l), "r"(b1l));
        }
      }
      float up[2] = {-1e30f, -1e30f};
      for (int h = 0; h < 2; h++) {
        const int row = it * 16 + g + 8 * h;
        const bool real = row < VOCAB_REAL;
        const float* pr = c.un8s;
        const double sr = UN8_PRE ? prm[h][0] : pr[row], q1 = UN8_PRE ? prm[h][1] : pr[nrow + row], einf = UN8_PRE ? prm[h][2] : pr[2 * nrow + row],
                     e2 = UN8_PRE ? prm[h][3] : pr[3 * nrow + row], w1 = UN8_PRE ? prm[h][4] : pr[4 * nrow + row];
        for (int cc = 0; cc < 2; cc++) {
          const int tok = 2 * t + cc;
          if (tok >= B || !real) continue;
          const double V = 256.0 * (double)dh[2 * h + cc] + (double)dl[2 * h + cc];
          const double A = sr * tb[cc] * V;
          const double M = sr * q1 * 0.51 * tb[cc] + fmin(einf * h1[cc], e2 * h2[cc]) + 1e-4 * w1 * hmx[cc] + 1e-9 * fabs(A) + 1e-6;
          const float zhi = 30.0f * tanhf((float)((A + M) / 30.0)) + 0.06f;
          const float zlo = 30.0f * tanhf((float)((A - M) / 30.0)) - 0.06f;
          up[cc] = fmaxf(up[cc], zhi);
          zl0[cc] = fmaxf(zl0[cc], zlo);
        }
      }
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) up[cc] = fmaxf(up[cc], __shfl_xor_sync(0xffffffff, up[cc], o));
        const int tok = 2 * t + cc;
        if (g == 0 && tok < B) c.zhi[(long)tok * UN_STRIPS + it] = up[cc];
      }
    }
    for (int cc = 0; cc < 2; cc++) {
      float m = zl0[cc];
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
      const int tok = 2 * t + cc;
      if (g == 0 && tok < B) {
        unsigned u = __float_as_uint(m);
        u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        atomicMax(c.zlo + tok, u);
      }
    }
    grid_bar(c, nbar, base);
  }
#else
  if (cert) {
    // pass 1: approximate logits A = s * sum q h with a rigorous bound M on |exact acc - A| (quantization error x |h|_1,
    // fp32 summation error of both sums, the scale's rounding); z bounds with a 0.06 allowance for the approximate tanh
    float h1[MAXB];
#pragma unroll
    for (int b = 0; b < MAXB; b++) {
      float sacc = 0.f;
      if (b < B) for (int i = threadIdx.x; i < H; i += NTH) sacc += fabsf(bf(sm.u.xs[0][b][i]));
      h1[b] = sacc;
    }
    block_sum<MAXB>(h1, sm);
    for (int it = blockIdx.x + gridDim.x * w; it < UN_STRIPS; it += gridDim.x * NWARP) {
      BSm bl{&sm.u.xs[0][0][0], HP, B};
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      piece_i8(c.un8 + (long)it * 36 * 1024, bl, acc);
      for (int h = 0; h < 2; h++) {
        const int row = it * 16 + g + 8 * h;
        if (row >= VOCAB_REAL) continue;
        const float sc = c.un8s[row], mc = c.un8m[row];
        for (int cc = 0; cc < 2; cc++) {
          const int tok = 2 * t + cc;
          if (tok >= B) continue;
          const float A = sc * acc[2 * h + cc];
          const float M = mc * h1[tok] * 1.001f + 1e-7f * fabsf(A) + 1e-6f;
          const float zhi = 30.0f * tanhf((A + M) / 30.0f) + 0.06f;
          const float zlo = 30.0f * tanhf((A - M) / 30.0f) - 0.06f;
          c.zhi[(long)tok * (UN_STRIPS * 16) + row] = zhi;
          zl0[cc] = fmaxf(zl0[cc], zlo);
        }
      }
    }
    for (int cc = 0; cc < 2; cc++) {
      float m = zl0[cc];
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
      const int tok = 2 * t + cc;
      if (g == 0 && tok < B) {
        unsigned u = __float_as_uint(m);
        u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        atomicMax(c.zlo + tok, u);
      }
    }
    grid_bar(c, nbar, base);
  }
#endif
  float zmin[2] = {0.f, 0.f};
  if (cert)
    for (int cc = 0; cc < 2; cc++) {
      const int tok = 2 * t + cc;
      const unsigned u = tok < B ? ldcg_u32(c.zlo + tok) : 0u;
      zmin[cc] = __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
    }
  for (int it = blockIdx.x + gridDim.x * w; it < UN_STRIPS; it += gridDim.x * NWARP) {
    if (cert) {   // pass 2: only strips holding a row whose upper bound reaches a row's best lower bound, exactly
      bool any = false;
      for (int h = 0; h < 2; h++) {
        const int row = it * 16 + g + 8 * h;
        for (int cc = 0; cc < 2; cc++) {
          const int tok = 2 * t + cc;
#if UN8 == 2
          if (tok < B && __ldcg(c.zhi + (long)tok * UN_STRIPS + it) >= zmin[cc]) any = true;
#else
          if (tok < B && row < VOCAB_REAL && __ldcg(c.zhi + (long)tok * (UN_STRIPS * 16) + row) >= zmin[cc]) any = true;
#endif
        }
      }
      if (!__any_sync(0xffffffffu, any) || UN8_DIAG) continue;
#ifdef UN8_COUNT
      if (lane == 0) atomicAdd(reinterpret_cast<unsigned*>(c.bar) + 3, 1u);   // debug: certified strips recomputed
#endif
    }
    BSm bl{&sm.u.xs[0][0][0], HP, B};
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
        if (tok >= B || row >= VOCAB_REAL) continue;
        const float z = 30.0f * tanhf(acc[2 * h + cc] / 30.0f);
        if (hd.logits) hd.logits[(long)tok * VOCAB_REAL + row] = z;
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
    if (g == 0 && tok < B && k) atomicMax(c.amax + tok, k);
  }
}

// ------------------------------------------------------------------------------------------------ kernel
// V: verify pass (rows = positions of one sequence); M: the step kernel for B >= 2 (the B = 1 kernel keeps its code)
#ifndef KSPEC
#define KSPEC 0   // verify passes: the pass body compiled once per row count 1..8 (rows a compile-time constant: register
                  // allocation and loops per K; measured K=2 -200 us, K=8 -120 us), chosen at kernel entry, out of line
#endif
template <bool V, bool M = false, int KF = 0> __device__ __forceinline__ void mk_body(const Ctx* __restrict__ cp, Hdr hd0) {
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
  const int nst = hd0.nsteps > 0 ? hd0.nsteps : 1;
  for (int st = 0; st < nst; st++) {
  Hdr hd = hd0;
  if (V) {   // verify pass: position and number of rows from the device (written by the draft of the previous cycle)
    hd.pos = hd0.vpos[0];
    const int kb = hd0.vpos[1];
    hd.B = (kb >= 1 && kb <= MAXB) ? kb : hd0.B;
    if (KF > 0) hd.B = KF;   // the dispatcher chose this instance for exactly KF rows
#ifdef KBV
    hd.B = KBV;   // experiment: rows fixed at compile time (timing only unless the pass has KBV rows)
#endif
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
      ph_gate_fg<V>(c, hd, sm, a, 0, 3);
      GBAR(); MARK(9);
      const LayerW& L = c.L[ap.layer];
      float wv[3][EPT];
      for (int q = 0; q < 3; q++) load_w(L.norm[L.glob ? q : 0], wv[q]);
#ifndef HYB_GN_MAX
#define HYB_GN_MAX 2   // gate norm: hybrid up to this many rows
#endif
      const bool hy = V && HYB2 && hd.B >= 2 && hd.B <= (HYB_GN_MAX > HYB_MAX ? HYB_GN_MAX : HYB_MAX);
      const bool bc = desig<V>(hd) && !hy;
      auto row = [&](int b, bool pub) {
        float x[3][EPT];
        uint2 r[3][EPT];
        const uint2* p0 = c.xpt + ((long)((a & 1) * MAXB + b) * 4) * H;
#pragma unroll
        for (int q = 0; q < 3; q++)
#pragma unroll
          for (int k = 0; k < EPT; k++) r[q][k] = ld_tag(p0 + q * H + (EV(k) ? EL(k) : EL(k & 1)));
#pragma unroll
        for (int q = 0; q < 3; q++)
#pragma unroll
          for (int k = 0; k < EPT; k++) x[q][k] = EV(k) ? __uint_as_float(wait_tag(p0 + q * H + EL(k), r[q][k], TAGA(a))) : 0.f;
        if (pub) {
          uint2* xs_pub = c.xst + (long)(a & 1) * 3 * MAXB * NPAIR;
          uint2* dst[3] = {xs_pub + (long)b * NPAIR, xs_pub + (long)(MAXB + b) * NPAIR, xs_pub + (long)(2 * MAXB + b) * NPAIR};
          norm_publish<3>(x, wv, 3, dst, TAGA(a), sm);
        } else {
          bf16* dst[3] = {&sm.u.xs[0][b][0], &sm.u.xs[1][b][0], &sm.u.xs[2][b][0]};
          rmsnorm_store_n<3>(x, wv, 3, dst, sm);
        }
      };
      if (bc) { for (int b = dsel<V>(hd); b < hd.B; b += gridDim.x) row(b, true); }
      else if (hy) { for (int b = 1 + (int)(gridDim.x - 1 - blockIdx.x); b < hd.B; b += gridDim.x) row(b, true); row(0, false); }
      else { for (int b = 0; b < hd.B; b++) row(b, false); }
      if (bc || hy) {
        const int r0 = hy ? 1 : 0;
        const uint2* xs_pub = c.xst + (long)(a & 1) * 3 * MAXB * NPAIR;
        const unsigned need = ((V || POLL_STEP) && POLL_NEEDED) ? qkv_streams(L, 3) : 7u;
        for (int q = 0; q < 3; q++)
          if ((need >> q) & 1u) pollr<V>(hd.B, xs_pub + ((long)q * MAXB + r0) * NPAIR, hd.B - r0, [&](int rr) { return &sm.u.xs[q][r0 + rr][0]; }, TAGA(a));
      }
      __syncthreads();
      ns = 3;
    }
    MARK(10);
    ph_qkv(c, hd, sm, a, ns);
    GBAR(); MARK(2);
    ph_attn<V>(c, hd, sm, a);
    GBAR(); MARK(3);
    ph_oproj<V>(c, hd, sm, a);
    GBAR(); MARK(4);
    int r = ph_mlp_in<V>(c, hd, sm, a);
    if (r < 0) {
      GBAR(); MARK(14);
      ph_gate_d<V>(c, hd, sm, a, 3, 1);
      GBAR(); MARK(15);
      ph_gate_fg<V>(c, hd, sm, a, 3, 1);
      GBAR(); MARK(16);
      if (hyb2<V>(hd)) {
        for (int b = 1 + (int)(gridDim.x - 1 - blockIdx.x); b < hd.B; b += gridDim.x) {
          float x[1][EPT], wv[EPT];
          load_w(c.L[ap.layer].mlp_norm, wv);
          load_vec_f(c.xpt + ((long)((a & 1) * MAXB + b) * 4 + 3) * H, TAGA(a), x[0]);
          mlp_publish(c, hd, sm, a, x, b, wv);
        }
        {
          float x[1][EPT], wv[EPT];
          load_w(c.L[ap.layer].mlp_norm, wv);
          load_vec_f(c.xpt + ((long)((a & 1) * MAXB + 0) * 4 + 3) * H, TAGA(a), x[0]);
          mlp_in_x(c, hd, sm, a, x, 0, wv);
        }
        mlp_poll<V>(c, hd, sm, a, 1);
      } else if (desig<V>(hd)) {
        for (int b = dsel<V>(hd); b < hd.B; b += gridDim.x) {
          float x[1][EPT], wv[EPT];
          load_w(c.L[ap.layer].mlp_norm, wv);
          load_vec_f(c.xpt + ((long)((a & 1) * MAXB + b) * 4 + 3) * H, TAGA(a), x[0]);
          mlp_publish(c, hd, sm, a, x, b, wv);
        }
        mlp_poll<V>(c, hd, sm, a);
      } else {
        for (int b = 0; b < HB(hd); b++) {
          float x[1][EPT], wv[EPT];
          load_w(c.L[ap.layer].mlp_norm, wv);
          load_vec_f(c.xpt + ((long)((a & 1) * MAXB + b) * 4 + 3) * H, TAGA(a), x[0]);
          mlp_in_x(c, hd, sm, a, x, b, wv);
        }
      }
      __syncthreads();
    }
    MARK(11);
    ph_router(c, hd, sm, a);
    GBAR(); MARK(5);
    routing<V>(c, hd, sm, a);
    MARK(12);
    ph_w13<V>(c, hd, sm, a);
    GBAR(); MARK(6);
    ph_w2a<V, M>(c, hd, sm, a);
    GBAR(); MARK(17);
    ph_w2b<V>(c, hd, sm, a);
    GBAR();
  }
  MARK(13);
  ph_final<V>(c, hd, sm, nbar, base);
  MARK(MKPROF_WAIT);
  grid_bar(c, nbar, base);
  if (blockIdx.x == 0) {
    if (threadIdx.x < HB(hd)) {
      const unsigned long long k = c.amax[threadIdx.x];
      const int tok = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFu));
      c.next_tok[threadIdx.x] = tok;
      c.amax[threadIdx.x] = 0ull;
      if (V && c.zlo) c.zlo[threadIdx.x] = 0u;
      st_tag(c.ntt + threadIdx.x, (uint32_t)tok, (hd.tag0 + 60u) & 0x7fffffffu);
      if (hd.out_tok) hd.out_tok[threadIdx.x] = tok + 1;   // 0 = not yet written (host polls)
    }
  }
  }   // steps
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    c.bar[1] = base + nbar * gridDim.x;
    if (V && (hd0.flags & FLAG_DEVTAG)) c.bar[3] = c.bar[3] + 1u;
  }
}
#ifndef VBODY_ATTR
#define VBODY_ATTR __forceinline__   // (__noinline__: 1.3 KB spills per body, much slower)
#endif
template <int KF> __device__ VBODY_ATTR void mk_vbody(const Ctx* __restrict__ cp, Hdr hd0) { mk_body<true, false, KF>(cp, hd0); }
template <bool V, bool M = false> __global__ void __launch_bounds__(NTH, 1) mk_step(const Ctx* __restrict__ cp, Hdr hd0) {
  // passes of more than 8 rows belong to the 16-row library (engine/c/mk16.cu, queued next to this one): return at once
  if (V && hd0.vpos[1] > MAXB) return;
#ifdef KVONLY
  // a library of one verify kernel for exactly KVONLY rows (engine/build/libmkv<K>.so): any other pass returns at once
  if (V) {
    const int kb = hd0.vpos[1];
    if (((kb >= 1 && kb <= MAXB) ? kb : hd0.B) != KVONLY) return;
    // FLAG_DEVTAG: the pass's tag range from the device counter bar[3] (launches captured in a graph keep their
    // parameters): every block reads it here, block 0 advances it after the pass's last grid barrier
    if (hd0.flags & FLAG_DEVTAG) hd0.tag0 = ((__ldcg(cp->bar + 3) + 1u) * 64u) & 0x7fffffffu;
    mk_body<true, false, KVONLY>(cp, hd0);
    return;
  }
#endif
  if (V && KSPEC) {
    const int kb = hd0.vpos[1];
    switch ((kb >= 1 && kb <= MAXB) ? kb : hd0.B) {
      case 1: mk_vbody<1>(cp, hd0); return;
      case 2: mk_vbody<2>(cp, hd0); return;
      case 3: mk_vbody<3>(cp, hd0); return;
      case 4: mk_vbody<4>(cp, hd0); return;
      case 5: mk_vbody<5>(cp, hd0); return;
      case 6: mk_vbody<6>(cp, hd0); return;
      case 7: mk_vbody<7>(cp, hd0); return;
      default: mk_vbody<8>(cp, hd0); return;
    }
  }
  mk_body<V, M, 0>(cp, hd0);
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
#ifndef KVONLY
__global__ void k_vsel(const int* vpos, int defB, cudaGraphConditionalHandle h) {
  const int kb = vpos[1];
  cudaGraphSetConditional(h, (unsigned)(((kb >= 1 && kb <= MAXB) ? kb : defB) - 1));
}
#endif
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
int mk_un8_mode() { return UN8; }
int mk_ring() { return RING; }
int mk_nth() { return NTH; }
int mk_cn_total() { return CN_TOTAL; }
#ifdef W12
int mk_w12() { return 1; }
#else
int mk_w12() { return 0; }
#endif
#ifndef VSMEM_EXTRA
#define VSMEM_EXTRA (W13_ASYNC ? 15 * W13_RING : (W13_TMA ? 128 + VSMEM_EXTRA_V : 0))   // extra dynamic shared memory of the verify kernel (bytes)
#endif
int mk_setup() {
#ifdef KVONLY
  cudaError_t e = cudaFuncSetAttribute(mk_step<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem) + VSMEM_EXTRA);
#else
  cudaError_t e = cudaFuncSetAttribute(mk_step<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem));
  if (e == cudaSuccess) e = cudaFuncSetAttribute(mk_step<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem));
  if (e == cudaSuccess) e = cudaFuncSetAttribute(mk_step<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem) + VSMEM_EXTRA);
#endif
  if (e != cudaSuccess) { fprintf(stderr, "mk_setup: %s\n", cudaGetErrorString(e)); return 1; }
  int nb = 0, nb2 = 0;
#ifdef KVONLY
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mk_step<true>, NTH, sizeof(Smem) + VSMEM_EXTRA);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb2, mk_step<true>, NTH, sizeof(Smem) + VSMEM_EXTRA);
#else
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mk_step<false>, NTH, sizeof(Smem));
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb2, mk_step<true>, NTH, sizeof(Smem) + VSMEM_EXTRA);
#endif
  if (nb2 < nb) nb = nb2;
#ifndef KVONLY
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb2, mk_step<false, true>, NTH, sizeof(Smem));
  if (nb2 < nb) nb = nb2;
#endif
  if (nb < 1) { fprintf(stderr, "mk_setup: occupancy %d\n", nb); return 2; }
  return 0;
}
int mk_launch(const void* ctx_dev, const Hdr* hd, int nblk, void* stream) {
  Hdr h = *hd;
  if (KB1 && h.B != 1) { fprintf(stderr, "mk_launch: this build serves batch 1 only (KB1)\n"); return 2; }
  void* args[] = {(void*)&ctx_dev, (void*)&h};
#ifdef KVONLY
  if (!h.vpos) { fprintf(stderr, "mk_launch: a KVONLY library runs verify passes only\n"); return 2; }
  const void* kern = (const void*)mk_step<true>;
#else
  const void* kern = h.vpos ? (const void*)mk_step<true> : (h.B >= 2 ? (const void*)mk_step<false, true> : (const void*)mk_step<false>);
#endif
  cudaError_t e = cudaLaunchCooperativeKernel(kern, dim3(nblk), dim3(NTH), args, sizeof(Smem) + (h.vpos ? VSMEM_EXTRA : 0), (cudaStream_t)stream);
  if (e != cudaSuccess) { fprintf(stderr, "mk_launch: %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
#ifndef KVONLY
// the cycle graph's verify (s18): a one-thread kernel sets a SWITCH node to vpos[1] - 1 and the node's eight bodies
// each hold one specialized verify kernel (captured from the KVONLY libraries' mk_launch): one launch per pass
typedef int (*mk_launch_fn)(const void*, const Hdr*, int, void*);
int mk_capture_vswitch(void* stream, void* const* fns, const void* ctx_dev, const Hdr* hd, int nblk) {
  cudaStream_t s = (cudaStream_t)stream;
  cudaStreamCaptureStatus cst = cudaStreamCaptureStatusNone;
  cudaGraph_t g = nullptr;
  if (cudaStreamGetCaptureInfo(s, &cst, nullptr, &g, nullptr, nullptr, nullptr) != cudaSuccess || cst != cudaStreamCaptureStatusActive) return 11;
  cudaGraphConditionalHandle h;
  if (cudaGraphConditionalHandleCreate(&h, g, 7u, 0u) != cudaSuccess) return 12;
  k_vsel<<<1, 1, 0, s>>>(hd->vpos, hd->B, h);
  const cudaGraphNode_t* deps = nullptr; const cudaGraphEdgeData* ed = nullptr; size_t nd = 0;
  if (cudaStreamGetCaptureInfo(s, &cst, nullptr, &g, &deps, &ed, &nd) != cudaSuccess) return 13;
  cudaGraphNodeParams np = {};
  np.type = cudaGraphNodeTypeConditional;
  np.conditional.handle = h;
  np.conditional.type = cudaGraphCondTypeSwitch;
  np.conditional.size = MAXB;
  cudaGraphNode_t node;
  if (cudaGraphAddNode(&node, g, deps, nullptr, nd, &np) != cudaSuccess) return 14;
  static cudaStream_t s2 = nullptr;
  if (!s2 && cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking) != cudaSuccess) return 15;
  for (int k = 0; k < MAXB; k++) {
    if (cudaStreamBeginCaptureToGraph(s2, np.conditional.phGraph_out[k], nullptr, nullptr, 0, cudaStreamCaptureModeRelaxed) != cudaSuccess) return 16;
    const int r = ((mk_launch_fn)fns[k])(ctx_dev, hd, nblk, (void*)s2);
    cudaGraph_t body;
    if (cudaStreamEndCapture(s2, &body) != cudaSuccess || r != 0) return 17;
  }
  if (cudaStreamUpdateCaptureDependencies(s, &node, nullptr, 1, cudaStreamSetCaptureDependencies) != cudaSuccess) return 18;
  return (int)cudaGetLastError();
}
#endif
void* mk_host_alloc(size_t n) { void* p = nullptr; if (cudaHostAlloc(&p, n, cudaHostAllocMapped) != cudaSuccess) return nullptr; memset(p, 0, n); return p; }
}
