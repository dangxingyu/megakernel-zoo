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
#ifndef W13_UNR
#define W13_UNR 12
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
  // track B (SKH, appended: no offset of the fields above changes): certified int8 sketch of the unembedding
  const uint8_t* skq; const float* sks; const float* ske; const float* skn; unsigned* skl; float skeps; int skpad;
};
struct Hdr { int B, pos, flags, nsteps; const int* tokens; float* logits; int* out_tok; int out_stride; unsigned tag0;
             const int* prompt; int plen; unsigned tok_tag; };   // nsteps > 0: device loop over nsteps steps
// track A (K1, port of s15-mini-b200-b2 s0020): the same source compiled twice; KB1 = 1 gives a kernel for batch 1
// only (every B >= 2 path is dead and removed, registers allocated for the B = 1 paths alone). api.py launches the
// one that fits the batch.
#ifndef KB1
#define KB1 0
#endif
#define HDB (KB1 ? 1 : hd.B)
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
      float kt[HD]; float vt[HD]; float ml[2][3][2]; float po[3][HD];
    } at;
    float mrg[MAXCH_S * 3 * 130 + 3 * MAXCH];      // attention merge staging
    struct {                               // ATT3: three teams' 64-key records (K / V rows unpadded, XOR-swizzled)
      bf16 kk[3][64 * HD]; bf16 vv[3][64 * HD]; float sc[3][3][64]; bf16 qb[3][3][HD]; bf16 ph[3][3][64]; bf16 pl[3][3][64];
      float rr[3][3][16]; float kt[3][HD]; float vt[3][HD]; float ml[3][3][2];
    } a3;
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
};

static_assert(sizeof(((Smem*)0)->u.a3) <= sizeof(((Smem*)0)->u.xs), "ATT3 tiles must not grow the shared-memory union");

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
    const uint2* base = c.pbt + ((long)idx[s] * MAXB + b) * NPAIR;
    r[s][0] = ld_tag(base + threadIdx.x);
    r[s][1] = ld_tag(base + p1);
  }
#pragma unroll
  for (int s = 0; s < NS; s++) {
    const uint2* base = c.pbt + ((long)idx[s] * MAXB + b) * NPAIR;
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
  constexpr int UNR = UNR_ ? UNR_ : FP4_UNR;
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
#define FP8_UNR 4
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
__device__ __forceinline__ float sconv_static(float x, const bf16* hist_row_base, long ring_stride, int pos, const bf16* w4) {
  const float x1 = ldcg_bf(hist_row_base + ((pos + 2) % 3) * ring_stride);
  const float x2 = ldcg_bf(hist_row_base + ((pos + 1) % 3) * ring_stride);
  const float x3 = ldcg_bf(hist_row_base + (pos % 3) * ring_stride);
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
__device__ __forceinline__ void poll_rows(const uint2* src, int nrows, DST dst, unsigned tag) {
  const int n = nrows * NPAIR;   // words, two per load
  for (int i0 = 2 * threadIdx.x; i0 < n; i0 += 2 * NTH * 8) {
    uint2 r[8][2];
#pragma unroll
    for (int q = 0; q < 8; q++) ld_tag2(src + min(i0 + q * 2 * NTH, n - 2), r[q][0], r[q][1]);
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const int i = i0 + q * 2 * NTH;
      if (i >= n) continue;
      const int row = i / NPAIR, w = i % NPAIR;
      uint32_t* d = reinterpret_cast<uint32_t*>(dst(row)) + w;
      d[0] = wait_tag(src + i, r[q][0], tag);
      d[1] = wait_tag(src + i + 1, r[q][1], tag);
    }
  }
}

// The attention input streams of application a -> normalized bf16 smem streams (returns their number), or for the
// gated applications (loop passes 2 and 3) G1: blocks (token, controller) write x, m and [unit(x), unit(m)] (returns -1)
__device__ int ph_attn_in(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = HDB;
  if (B >= 2 && threadIdx.x < 4) {   // this block's qkv strips into L2 while the mix waits for its sources (B >= 2: the mix is long enough)
    const int strip = blockIdx.x + threadIdx.x * gridDim.x;
    if (strip < QG_STRIPS + 2 * L.nsk) pf_l2(L.wqkv + (long)strip * 36 * 2048, 36 * 2048);
  } else if (B >= 2 && threadIdx.x == 32 && blockIdx.x < 8) pf_l2(L.wr + (long)blockIdx.x * 36 * 2048, 36 * 2048);   // router strip
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
  if (ap.nmsg == 0 && B >= 2) {
    uint2* xs_pub = c.xst + (long)par * 3 * MAXB * NPAIR;   // [3][MAXB][NPAIR]
    for (int job = blockIdx.x; job < B * nq; job += gridDim.x) {
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
        load_srcs<5>(c, hd, idx, msk, b, v);
        depth_mix_q<5, 1>(v, msk, qv, 1, x, sm);
      }
      uint2* dst[1] = {xs_pub + ((long)q * MAXB + b) * NPAIR};
      norm_publish<1>(x, wv, 1, dst, TAGA(a), sm);
    }
    for (int q = 0; q < nq; q++)
      poll_rows(xs_pub + (long)q * MAXB * NPAIR, B, [&](int r) { return &sm.u.xs[q][r][0]; }, TAGA(a));
    __syncthreads();
    return nq;
  }
  if (ap.nmsg == 0) {
    for (int b = 0; b < B; b++) {
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
        load_srcs<5>(c, hd, idx, msk, b, v);
        SUBMARK(31);
        if (nq == 1) depth_mix_q<5, 1>(v, msk, qv, 1, x, sm); else depth_mix_q<5, 3>(v, msk, qv, 3, x, sm);
        SUBMARK(32);
      }
      bf16* dst[3] = {&sm.u.xs[0][b][0], &sm.u.xs[1][b][0], &sm.u.xs[2][b][0]};
      if (nq == 1) rmsnorm_store_n<1>(x, wv, 1, dst, sm); else rmsnorm_store_n<3>(x, wv, 3, dst, sm);
      SUBMARK(33);
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
  const int B = HDB, par = a & 1;
  for (int b = blockIdx.x; b < B; b += gridDim.x) {
    float x[1][EPT], m[1][EPT], ms[2][EPT];
    mlp_mix(c, hd, sm, a, b, x);
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
__device__ void ph_gate_d(PH_ARGS, int a, int c0, int nc) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = HDB, par = a & 1;
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
      load_srcs<2>(c, hd, ap.msg, 3u, b, ms);
      depth_mix_q<2, 1>(ms, 3u, mqv, 1, m, sm);
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
__device__ void ph_gate_fg(PH_ARGS, int a, int c0, int nc) {
  const LayerW& L = c.L[c.A[a].layer];
  const int B = HDB, par = a & 1;
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
  const int B = HDB, par = a & 1;
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
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }

// raw current values, history ring, conv (static for local layers, dynamic taps for global) of 4 consecutive channels
// ch0 .. ch0 + 3; every independent load is issued before the first wait (per channel the same arithmetic as before)
__device__ __forceinline__ void kv_conv4(const LayerW& L, const uint2* qr, unsigned tag, const bf16* hist, const bf16* sc, int base_row, int ch0, int g, int t, float* out) {
  uint2 rd[4], rt[4];
#pragma unroll
  for (int k = 0; k < 4; k++) rd[k] = ld_tag(qr + base_row + ch0 + k);
#pragma unroll
  for (int lag = 0; lag < 4; lag++) rt[lag] = ld_tag(qr + base_row + 1024 + (L.glob ? g * 4 + lag : 0));
  // history rows (t + 2) % 3 = x1 (newest), (t + 1) % 3 = x2, t % 3 = x3, 4 channels each (8-byte loads)
  const uint2 h1 = __ldcg(reinterpret_cast<const uint2*>(hist + ((t + 2) % 3) * 1024 + ch0));
  const uint2 h2 = __ldcg(reinterpret_cast<const uint2*>(hist + ((t + 1) % 3) * 1024 + ch0));
  const uint2 h3 = __ldcg(reinterpret_cast<const uint2*>(hist + (t % 3) * 1024 + ch0));
  uint2 wsc[4];
  if (!L.glob) {
#pragma unroll
    for (int k = 0; k < 4; k++) wsc[k] = *reinterpret_cast<const uint2*>(sc + (ch0 + k) * 4);   // taps of channel k
  } else {
#pragma unroll
    for (int lag = 0; lag < 4; lag++) wsc[lag] = *reinterpret_cast<const uint2*>(sc + lag * 1024 + ch0);   // lag-major taps
  }
  const bf16* p1 = reinterpret_cast<const bf16*>(&h1);
  const bf16* p2 = reinterpret_cast<const bf16*>(&h2);
  const bf16* p3 = reinterpret_cast<const bf16*>(&h3);
  float dyn[4] = {0.f, 0.f, 0.f, 0.f};
  if (L.glob) {
#pragma unroll
    for (int lag = 0; lag < 4; lag++) dyn[lag] = __uint_as_float(wait_tag(qr + base_row + 1024 + g * 4 + lag, rt[lag], tag));
  }
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const float x = __uint_as_float(wait_tag(qr + base_row + ch0 + k, rd[k], tag));
    const float x1 = bf(p1[k]), x2 = bf(p2[k]), x3 = bf(p3[k]);
    if (!L.glob) {
      const bf16* w4 = reinterpret_cast<const bf16*>(&wsc[k]);
      float acc = __fmul_rn(x3, bf(w4[0]));
      acc = __fadd_rn(acc, __fmul_rn(x2, bf(w4[1])));
      acc = __fadd_rn(acc, __fmul_rn(x1, bf(w4[2])));
      acc = __fadd_rn(acc, __fmul_rn(x, bf(w4[3])));
      out[k] = rbf(__fmul_rn(__fadd_rn(acc, x), 0.44721359549995793f));
    } else {
      const float xs4[4] = {x, x1, x2, x3};
      float acc = 0.f;
#pragma unroll
      for (int lag = 0; lag < 4; lag++) acc = __fadd_rn(acc, __fmul_rn(xs4[lag], __fadd_rn(bf(reinterpret_cast<const bf16*>(&wsc[lag])[k]), dyn[lag])));
      out[k] = rbf(__fmul_rn(__fadd_rn(acc, x), 0.33333333333333333f));
    }
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
__device__ void ph_attn(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = HDB, t = hd.pos;
  // records (online-softmax partials) always cover 64 keys, so that the merge is the same for every batch size;
  // for B >= 3 an item loads two records' keys (fewer items) and computes the two records separately
  const int sub = B <= 2 ? 1 : 2, csz = 64 * sub;
  const int lo = L.glob ? 0 : max(0, t + 1 - 1024);
  const int nch = (t - lo + csz) / csz;      // items per (token, kv head)
  const int nrc = (t - lo + 64) / 64;        // records per (token, kv head)
  const int N = B * NKV * nch;
  const unsigned tag = TAGA(a);
  const int par = a & 1;
  const float tau = L.glob ? (1.0f + 0.1f * logf(fmaxf(1.0f, (float)(t + 1) / 8192.0f))) : 1.0f;
  const bool use_tau = L.glob;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
  auto& at = sm.u.at;
  const long cstride = (long)c.maxctx * HD;   // per (slot, b, head)
  // items: chunk-major from the last chunk down; chunk 0 of (b, g) merges and comes last in every block's order
  for (int it = blockIdx.x; it < N; it += gridDim.x) {
    const int ch = nch - 1 - it / (B * NKV), bg = it % (B * NKV), g = bg % NKV, b = bg / NKV;
    const int j0 = lo + ch * csz, j1 = min(j0 + csz, t + 1), nj = j1 - j0;
    const bool cur = (j1 == t + 1);
    const uint2* qr = c.qkvt + (long)(par * MAXB + b) * QKVR;
    SUBMARK(20);
    // ---- one wave of loads: K / V chunk (async), relative-bias rows, q / r, current k / v
    {
      const bf16* kb = c.kc + (((long)ap.slot * MAXB + b) * NKV + g) * cstride + (long)j0 * HD;
      const bf16* vb = c.vc + (((long)ap.slot * MAXB + b) * NKV + g) * cstride + (long)j0 * HD;
      const int nld = (cur ? nj - 1 : nj) * 16;
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
    } else if ((w == 6 || w == 7) && cur) {
      const bool isv = (w == 7);
      const bf16* hist = (isv ? c.hv : c.hk) + ((long)ap.slot * MAXB + b) * 3 * 1024;
      float kv[4], ss = 0.f;
      kv_conv4(L, qr, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, g * HD + lane * 4, g, t, kv);
#pragma unroll
      for (int k = 0; k < 4; k++) ss += kv[k] * kv[k];
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      const float f = isv ? rsqrtf(ss / 128.f + 1e-6f) : fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
      float* dst = isv ? at.vt : at.kt;
#pragma unroll
      for (int k = 0; k < 4; k++) dst[lane * 4 + k] = rbf(kv[k] * f);
    }
    cp_async_wait();
    __syncthreads();
    SUBMARK(21);
    if (cur && tid < HD) {
      // current position: from the values computed above; write the cache and the conv history
      at.kk[nj - 1][tid] = tobf(at.kt[tid]);
      at.vv[nj - 1][tid] = tobf(at.vt[tid]);
      const long co = (((long)ap.slot * MAXB + b) * NKV + g) * cstride + (long)t * HD + tid;
      c.kc[co] = tobf(at.kt[tid]);
      c.vc[co] = tobf(at.vt[tid]);
      const long hrow = ((long)ap.slot * MAXB + b) * 3 * 1024 + (t % 3) * 1024 + g * HD + tid;
      c.hk[hrow] = tobf(pollf(qr + K0 + g * HD + tid, tag));
      c.hv[hrow] = tobf(pollf(qr + V0 + g * HD + tid, tag));
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
        if ((d & 1) == 0) st_tag(c.aoutt + (long)(par * MAXB + b) * (QW / 2) + ((3 * g + hh) * HD + d) / 2, pack_bf2(out, o1), tag);
      }
      SUBMARK(26);
    }
    __syncthreads();
    SUBMARK(27);
  }
}

// ---- ATT3 (track A, this run): attention for B >= 3 by three independent 8-warp teams per block, one 64-key record
// per team at a time (named barriers instead of block syncs; K / V tiles without padding, chunks XOR-swizzled by row),
// then a block-level merge per (token, kv head). The record arithmetic is the B <= 2 path's, operation for operation,
// so the records and the logits are bit-identical to ph_attn's at every batch size.
#ifndef ATT3
#define ATT3 0
#endif
#ifndef ATT3_MIN
#define ATT3_MIN 0       // use ATT3 when B * NKV * records > ATT3_MIN (else ph_attn)
#endif
__device__ __forceinline__ int swz(int r, int d) { return r * HD + ((((d >> 3) ^ (r & 7))) << 3) + (d & 7); }
#ifndef NI_ATT3
#define NI_ATT3 __forceinline__
#endif
__device__ NI_ATT3 void ph_attn3(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = HDB, t = hd.pos;
  const int lo = L.glob ? 0 : max(0, t + 1 - 1024);
  const int nrc = (t - lo + 64) / 64;        // records per (token, kv head)
  const int R = B * NKV * nrc;
  const unsigned tag = TAGA(a);
  const int par = a & 1;
  const float tau = L.glob ? (1.0f + 0.1f * logf(fmaxf(1.0f, (float)(t + 1) / 8192.0f))) : 1.0f;
  const bool use_tau = L.glob;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
  const int tm = w >> 3, tw = w & 7, T = tid & 255;
  auto& A3 = sm.u.a3;
  bf16* kk = A3.kk[tm]; bf16* vv = A3.vv[tm];
  float (*sc)[64] = A3.sc[tm]; bf16 (*qb)[HD] = A3.qb[tm]; bf16 (*ph)[64] = A3.ph[tm]; bf16 (*pl)[64] = A3.pl[tm];
  float (*rr)[16] = A3.rr[tm]; float* kt = A3.kt[tm]; float* vt = A3.vt[tm]; float (*ml)[2] = A3.ml[tm];
  const long cstride = (long)c.maxctx * HD;
  // software pipeline per team: record r+1's K tile loads during r's softmax / PV, its V tile during r's PV tail and
  // r+1's polls (cp.async groups: K then V per record; wait_group 1 = all but the newest group complete)
  auto kvbase = [&](int r, bool isv) -> const uint4* {
    const int rc = r / (B * NKV), bg = r % (B * NKV), g = bg % NKV, b = bg / NKV;
    return reinterpret_cast<const uint4*>((isv ? c.vc : c.kc) + (((long)ap.slot * MAXB + b) * NKV + g) * cstride + (long)(lo + rc * 64) * HD);
  };
  auto nld_of = [&](int r) -> int {
    const int rc = r / (B * NKV), j0 = lo + rc * 64, j1 = min(j0 + 64, t + 1);
    return ((j1 == t + 1) ? j1 - j0 - 1 : j1 - j0) * 16;
  };
  auto issue = [&](int r, bool isv) {
    if (r < R) {
      const uint4* src = kvbase(r, isv);
      bf16* dst = isv ? vv : kk;
      const int nld = nld_of(r);
      for (int i = T; i < nld; i += 256) { const int rw = i >> 4; cp_async16(dst + rw * HD + (((i & 15) ^ (rw & 7)) << 3), src + i); }
      if (isv) {
        const int rc = r / (B * NKV), j0 = lo + rc * 64, nj = min(j0 + 64, t + 1) - j0, nj16 = (nj + 15) & ~15;
        for (int i = T; i < (nj16 - nj) * (HD / 2); i += 256) reinterpret_cast<uint32_t*>(vv + nj * HD)[i] = 0u;
      }
    }
    cp_async_commit();
  };
  const int r0 = tm * gridDim.x + blockIdx.x;
  issue(r0, false); issue(r0, true);
  for (int r = r0; r < R; r += 3 * gridDim.x) {
    const int rc = r / (B * NKV), bg = r % (B * NKV), g = bg % NKV, b = bg / NKV;
    const int j0 = lo + rc * 64, j1 = min(j0 + 64, t + 1), nj = j1 - j0;
    const bool cur = (j1 == t + 1);
    const uint2* qr = c.qkvt + (long)(par * MAXB + b) * QKVR;
    const int bh = T >> 6, bj = T & 63;
    float prv[16];
    {
      const int dist = t - (j0 + bj);
#pragma unroll
      for (int q = 0; q < 16; q++) prv[q] = (T < 192 && bj < nj && dist < 1024) ? bf(L.relp[q * 1024 + min(dist, 1023)]) : 0.f;
    }
    if (tw < 3) {
      const int h = 3 * g + tw;
      float qv[4], ss = 0.f;
      uint2 rq[4];
#pragma unroll
      for (int k = 0; k < 4; k++) rq[k] = ld_tag(qr + h * HD + lane * 4 + k);
      const uint2 qsc = *reinterpret_cast<const uint2*>(L.qscale + h * HD + lane * 4);
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
        qb[tw][lane * 4 + k] = tobf(y);
      }
    } else if (tw < 6) {
      const int hh = tw - 3, h = 3 * g + hh;
      float rv = 0.f, ss = 0.f;
      const float rsc = lane < 16 ? bf(L.rscale[h * 16 + lane]) : 0.f;
      if (lane < 16) { rv = pollf(qr + R_R + h * 16 + lane, tag) * rsc; ss = rv * rv; }
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      const float f = fminf(8.0f / (sqrtf(ss) + 1e-6f), 1.0f);
      if (lane < 16) {
        float y = rbf(rv * f);
        if (use_tau) y = rbf(y * tau);
        rr[hh][lane] = y;
      }
    } else if (cur) {
      const bool isv = (tw == 7);
      const bf16* hist = (isv ? c.hv : c.hk) + ((long)ap.slot * MAXB + b) * 3 * 1024;
      float kv[4], ss = 0.f;
      kv_conv4(L, qr, tag, hist, isv ? L.vsc : L.ksc, isv ? V0 : K0, g * HD + lane * 4, g, t, kv);
#pragma unroll
      for (int k = 0; k < 4; k++) ss += kv[k] * kv[k];
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      const float f = isv ? rsqrtf(ss / 128.f + 1e-6f) : fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
      float* dst = isv ? vt : kt;
#pragma unroll
      for (int k = 0; k < 4; k++) dst[lane * 4 + k] = rbf(kv[k] * f);
    }
    cp_async_wait1();          // this record's K tile
    grp_sync(1 + tm, 256);
    if (cur && T < HD) {
      kk[swz(nj - 1, T)] = tobf(kt[T]);
      const long co = (((long)ap.slot * MAXB + b) * NKV + g) * cstride + (long)t * HD + T;
      c.kc[co] = tobf(kt[T]);
      c.vc[co] = tobf(vt[T]);
      const long hrow = ((long)ap.slot * MAXB + b) * 3 * 1024 + (t % 3) * 1024 + g * HD + T;
      c.hk[hrow] = tobf(pollf(qr + K0 + g * HD + T, tag));
      c.hv[hrow] = tobf(pollf(qr + V0 + g * HD + T, tag));
    }
    if (T < 192 && bj < nj) {
      float bsum = 0.f;
#pragma unroll
      for (int q = 0; q < 16; q++) bsum += rr[bh][q] * prv[q];
      sc[bh][bj] = (t - (j0 + bj) < 1024) ? rbf(bsum) : 0.f;
    }
    grp_sync(1 + tm, 256);
    if (tw < (nj + 15) / 16) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < HD / 16; ks++) {
        uint32_t af[4];
        ldsm_x4(af, kk + swz(16 * tw + (lane & 15), 16 * ks + (lane >> 4) * 8));
        const uint32_t b0 = g8 < 3 ? *reinterpret_cast<const uint32_t*>(&qb[g8][16 * ks + 2 * t4]) : 0u;
        const uint32_t b1 = g8 < 3 ? *reinterpret_cast<const uint32_t*>(&qb[g8][16 * ks + 8 + 2 * t4]) : 0u;
        mma_bf16(acc, af[0], af[1], af[2], af[3], b0, b1);
      }
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int key = 16 * tw + g8 + (q >> 1) * 8, head = 2 * t4 + (q & 1);
        if (head < 3 && key < nj) sc[head][key] = acc[q] * 0.08838834764831845f + sc[head][key];
      }
    }
    grp_sync(1 + tm, 256);
    issue(r + 3 * gridDim.x, false);   // the K tile is free: next record's keys
    if (tw < 3) {
      const int hh = tw, k1 = min(64, nj);
      float m = -1e30f;
      for (int jj = lane; jj < k1; jj += 32) m = fmaxf(m, sc[hh][jj]);
#pragma unroll
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
      float l = 0.f;
      for (int jj = lane; jj < 64; jj += 32) {
        const float e = jj < k1 ? expf(sc[hh][jj] - m) : 0.f;
        const bf16 eh = tobf(e);
        ph[hh][jj] = eh;
        pl[hh][jj] = tobf(e - bf(eh));
        l += e;
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffff, l, o);
      if (lane == 0) { ml[hh][0] = m; ml[hh][1] = l; }
    }
    cp_async_wait1();          // this record's V tile (the next K tile may still be in flight)
    if (cur && T < HD) vv[swz(nj - 1, T)] = tobf(vt[T]);
    grp_sync(1 + tm, 256);
    {
      const int wd = tw;
      float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      for (int ks = 0; ks < (nj + 15) / 16; ks++) {
        uint32_t ah[4], al[4], bv[4];
        ah[0] = g8 < 3 ? *reinterpret_cast<const uint32_t*>(&ph[g8][16 * ks + 2 * t4]) : 0u; ah[1] = 0u;
        ah[2] = g8 < 3 ? *reinterpret_cast<const uint32_t*>(&ph[g8][16 * ks + 8 + 2 * t4]) : 0u; ah[3] = 0u;
        al[0] = g8 < 3 ? *reinterpret_cast<const uint32_t*>(&pl[g8][16 * ks + 2 * t4]) : 0u; al[1] = 0u;
        al[2] = g8 < 3 ? *reinterpret_cast<const uint32_t*>(&pl[g8][16 * ks + 8 + 2 * t4]) : 0u; al[3] = 0u;
        ldsm_x4_t(bv, vv + swz(16 * ks + (lane & 15), 16 * wd + (lane >> 4) * 8));
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
            st_tagf(c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH + rc) * 130 + 2 + d, acc[nt][q], tag);
          }
      }
      if (tw == 0 && lane < 3) {
        uint2* rec = c.arec + ((long)(b * NQH + 3 * g + lane) * MAXCH + rc) * 130;
        st_tagf(rec, ml[lane][0], tag);
        st_tagf(rec + 1, ml[lane][1], tag);
      }
    }
    grp_sync(1 + tm, 256);
    issue(r + 3 * gridDim.x, true);    // the V tile is free: next record's values
  }
  cp_async_wait();
  __syncthreads();
  // ---- merge per (token, kv head): every record from arec (the B <= 2 path's merge arithmetic)
  for (int m = blockIdx.x; m < B * NKV; m += gridDim.x) {
    const int b = m / NKV, g = m % NKV;
    const uint2* qr = c.qkvt + (long)(par * MAXB + b) * QKVR;
    float gate = 0.f, sink = 0.f;
    if (tid < 3 * HD) {
      const int hh = tid / HD, d = tid % HD, h = 3 * g + hh;
      const uint2 rg = ld_tag(qr + R_G + h * HD + d), rs = ld_tag(qr + R_S + h);
      gate = __uint_as_float(wait_tag(qr + R_G + h * HD + d, rg, tag));
      sink = __uint_as_float(wait_tag(qr + R_S + h, rs, tag));
    }
    float* st = sm.u.mrg;   // [3][nrc][130]
    const int nrec = 3 * nrc * 130;
    for (int i0 = tid; i0 < nrec; i0 += NTH * 8) {
      uint2 v[8];
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const int i = min(i0 + q * NTH, nrec - 1);
        const int hh = i / (nrc * 130), rest = i % (nrc * 130);
        v[q] = ld_tag(c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH) * 130 + rest);
      }
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const int i = i0 + q * NTH;
        if (i < nrec) {
          const int hh = i / (nrc * 130), rest = i % (nrc * 130);
          const uint2* p = c.arec + ((long)(b * NQH + 3 * g + hh) * MAXCH) * 130 + rest;
          while (v[q].y != tag) v[q] = ld_tag(p);
          st[hh * nrc * 130 + rest] = __uint_as_float(v[q].x);
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
    __syncthreads();
  }
}

// o_proj + epilogue: attn_raw -> sconv -> attst
__device__ void ph_oproj(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = HDB, t = hd.pos, par = a & 1;
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
    const bf16* hb = c.ha + ((long)ap.slot * MAXB + oht) * 3 * H + blockIdx.x * 16 + ohr;
    oh0 = ldcg_bf(hb + ((t + 2) % 3) * H); oh1 = ldcg_bf(hb + ((t + 1) % 3) * H); oh2 = ldcg_bf(hb + (t % 3) * H);
  }
  if (blockIdx.x < 144 && threadIdx.x < 16) otap = *reinterpret_cast<const uint2*>(L.attn_sc + (blockIdx.x * 16 + threadIdx.x) * 4);
  {
    const int n = B * QW / 2;
    const uint2* src = c.aoutt + (long)par * MAXB * (QW / 2);
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
      float y[4];
      for (int h = 0; h < 2; h++)
        for (int cc = 0; cc < 2; cc++) {
          const int tok = 2 * tt + cc;
          y[2 * h + cc] = 0.f;
          if (tok >= B) continue;
          const int row = strip * 16 + g + 8 * h;
          const float x = rbf(acc[2 * h + cc]);
          // = sconv_static(x, history, taps) with the values staged above
          const float* hv = ost + (tok * 16 + g + 8 * h) * 3;
          const bf16* w4 = reinterpret_cast<const bf16*>(&otp[g + 8 * h]);
          const float x1 = hv[0], x2 = hv[1], x3 = hv[2];
          float sa = __fmul_rn(x3, bf(w4[0]));
          sa = __fadd_rn(sa, __fmul_rn(x2, bf(w4[1])));
          sa = __fadd_rn(sa, __fmul_rn(x1, bf(w4[2])));
          sa = __fadd_rn(sa, __fmul_rn(x, bf(w4[3])));
          y[2 * h + cc] = __fmul_rn(__fadd_rn(sa, x), 0.44721359549995793f);
          c.ha[((long)ap.slot * MAXB + tok) * 3 * H + (t % 3) * H + row] = tobf(x);
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
  // the attention partial (newest source) is requested before waiting for the older ones (as load_vec_bf)
  const uint2* pa = c.attst + (long)((a & 1) * MAXB + b) * NPAIR;
  const int pa1 = EV(2) ? threadIdx.x + NTH : threadIdx.x;
  const uint2 ra0 = ld_tag(pa + threadIdx.x), ra1 = ld_tag(pa + pa1);
  load_srcs<6>(c, hd, idx, msk | (add_pin ? 32u : 0u), b, v);
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
__device__ void mlp_poll(PH_ARGS, int a) {
  const int B = HDB, par = a & 1;
  poll_rows(c.xrt + (long)par * MAXB * NPAIR, B, [&](int r) { return &sm.u.e.xr[r][0]; }, TAGA(a));
  if (threadIdx.x < B) {
    const int sh = (int)poll(c.sht + par * MAXB + threadIdx.x, TAGA(a));
    sm.bcast[32 + threadIdx.x] = (float)sh;
    sm.sc16[threadIdx.x] = ldexpf(1.0f, -sh);
  }
  __syncthreads();
  for (int b = 0; b < B; b++) {
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

__device__ int ph_mlp_in(PH_ARGS, int a) {
  const App& ap = c.A[a];
#if RTPF
  if (HDB == 1 && blockIdx.x < 8) {
    // this block's router strip (74 KB) into its L1 while the MLP-input mix waits for the attention output
    const uint8_t* p = c.L[ap.layer].wr + (long)blockIdx.x * 36 * 2048;
    for (int i = threadIdx.x; i < 36 * 2048 / 128; i += NTH) asm volatile("prefetch.global.L1 [%0];" :: "l"(p + i * 128));
  }
#endif
  if (ap.nmsg) return -1;   // the gate's D blocks compute the mix themselves
  if (HDB >= 2) {
    for (int b = blockIdx.x; b < HDB; b += gridDim.x) {
      float x[1][EPT], wv[EPT];
      load_w(c.L[ap.layer].mlp_norm, wv);
      mlp_mix(c, hd, sm, a, b, x);
      mlp_publish(c, hd, sm, a, x, b, wv);
    }
    mlp_poll(c, hd, sm, a);
    return 0;
  }
  for (int b = 0; b < HDB; b++) {
    float x[1][EPT], wv[EPT];
    load_w(c.L[ap.layer].mlp_norm, wv);
    mlp_mix(c, hd, sm, a, b, x);
    mlp_in_x(c, hd, sm, a, x, b, wv);
  }
  __syncthreads();
  return 0;
}

__device__ void ph_router(PH_ARGS, int a) {
  const LayerW& L = c.L[c.A[a].layer];
  const int B = HDB, par = a & 1;
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
__device__ void routing(const Ctx& c, const Hdr& hd, Smem& sm, int a) {
  const LayerW& L = c.L[c.A[a].layer];
  const int B = HDB;
  const int w = threadIdx.x >> 5, lane = lane_id();
  if (threadIdx.x < NE) { sm.ef13[threadIdx.x] = L.mx ? 1.0f : L.f13[threadIdx.x]; sm.ef2[threadIdx.x] = (L.mx ? 1.0f : L.f2[threadIdx.x]) * 0.0625f; }
  if (threadIdx.x < 4) sm.umask[threadIdx.x] = 0u;
  if (threadIdx.x < NE / 4) sm.tokm[threadIdx.x] = 0u;
  __syncthreads();
  if (w < B) {
    float lg[4], sel[4];
    uint2 rl[4];
#pragma unroll
    for (int k = 0; k < 4; k++) rl[k] = ld_tag(c.rlogt + ((a & 1) * MAXB + w) * NE + lane + 32 * k);
    float rb[4];
#pragma unroll
    for (int k = 0; k < 4; k++) rb[k] = L.rbias[lane + 32 * k];   // before the waits
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const int e = lane + 32 * k;
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
template <int NG>
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
  if (!L.mx) piece_fp4<NG, false, W13_UNR>(L.w13 + ((long)e * 144 + strip) * 36 * 576, u0, nu, colp, acc);
  else piece_fp8<NG, false>(L.w13 + ((long)e * 144 + strip) * 36 * 1056, u0, nu, colp, acc);
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
__device__ void ph_w13(PH_ARGS, int a) {
  const LayerW& L = c.L[c.A[a].layer];
  const int nU = sm.nU;

  const int T = nU * 144, G = gridDim.x;
  constexpr int WG = 3, PER = NWARP / WG;
  const int mine = (T - (int)blockIdx.x + G - 1) / G;      // items blockIdx.x + k G
  for (int r0 = 0; r0 < mine; r0 += PER) {
    const int ns = min(PER, mine - r0);
    // one n-group: union entries have at most 4 tokens (routing splits larger experts)
    auto run1 = [&](int q, int u0, int nu, float (*acc)[4]) { const int it = blockIdx.x + (r0 + q) * G; w13_run<1>(c, sm, L, it / 144, it % 144, u0, nu, acc); };
    auto epi = [&](int q, float (*acc)[4]) { const int it = blockIdx.x + (r0 + q) * G; w13_epi(c, hd, sm, L, a, it / 144, it % 144, 1, acc); };
    block_strips<WG, 1, GRP_EXP>(sm, ns, 36, run1, epi);
  }
}

// w2, part 1: work items (union expert u, output strip) expert-major in block-contiguous ranges (at most two experts
// per block, their inputs staged in shared memory once); 8 items per round, 3 warps x 6 units each (a fixed split).
// Each item publishes its contribution per token as tagged f32 rows w2ct[par][token][rank of the expert][row].
__device__ void ph_w2a(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int par = a & 1;
  const int nU = sm.nU;
  constexpr int WG = 3, PER = NWARP / WG;
  const int lane = lane_id();
  const int T = nU * 144;
  const int ib = (int)((long)T * blockIdx.x / gridDim.x), ie = (int)((long)T * (blockIdx.x + 1) / gridDim.x);
  if (ib >= ie) return;
  SUBMARK(40);
  const int u0 = ib / 144, u1 = (ie - 1) / 144;
  const int n0 = sm.uncnt[u0];
  {
    const int np = n0 + (u1 > u0 ? sm.uncnt[u1] : 0);
    const int nw = np * 2 * (IE / 2);   // tagged words
    constexpr int SB = 8;
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
        uint32_t* d = reinterpret_cast<uint32_t*>(&sm.u.w2s[0][0][0]) + i;
        d[0] = wait_tag(pp[q], r[q][0], TAGA(a));
        d[1] = wait_tag(pp[q] + 1, r[q][1], TAGA(a));
      }
    }
  }
  __syncthreads();
  SUBMARK(41);
  const __half* w2sb = &sm.u.w2s[0][0][0];
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
        colp[ng] = li < nt ? w2sb + ((long)(j0 + li) * 2 + part) * IE : nullptr;
      }
      colp[2] = nullptr;
      if (!L.mx) piece_fp4<1, false>(L.w2 + ((long)e * 144 + strip) * 18 * 576, uu0, nu, colp, acc);
      else piece_fp8<1, false>(L.w2 + ((long)e * 144 + strip) * 18 * 1056, uu0, nu, colp, acc);
    };
    auto epi = [&](int q, float (*acc)[4]) {
      const int it = r0 + q, u = it / 144, strip = it % 144;
      const int e = sm.uexp[u], nt = sm.uncnt[u];
      const int g = lane >> 2, tt = lane & 3;
      const float fac = sm.ef2[e];
      for (int ng = 0; ng < 1; ng++) {
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
    block_strips<WG, 1, GRP_EXP>(sm, ns, 18, run, epi);
    SUBMARK(42);
  }
}

// w2, part 2: one block per output strip: per token the 8 expert rows summed in expert-id order, the short
// convolution and the new partial (pbt)
__device__ void ph_w2b(PH_ARGS, int a) {
  const App& ap = c.A[a];
  const LayerW& L = c.L[ap.layer];
  const int B = HDB, t = hd.pos, par = a & 1;
  for (int strip = blockIdx.x; strip < 144; strip += gridDim.x) {
    const bool addp = !(a == 0 || ap.bnd);
    const int et = threadIdx.x >> 4, er = threadIdx.x & 15;    // token et, row er
    const int row = strip * 16 + er;
    float np_ = 0.f;
    if (et < B) {
      // independent loads first: history, taps, attention output, old partial; then the expert rows
      const bf16* hb = c.hm + ((long)ap.slot * MAXB + et) * 3 * H + row;
      const float x1 = ldcg_bf(hb + ((t + 2) % 3) * H), x2 = ldcg_bf(hb + ((t + 1) % 3) * H), x3 = ldcg_bf(hb + (t % 3) * H);
      const uint2 tw = *reinterpret_cast<const uint2*>(L.mlp_sc + row * 4);
      const uint2 ra = ld_tag(c.attst + (long)(par * MAXB + et) * NPAIR + row / 2);
      const uint2 rp = ld_tag(c.pbt + ((long)(addp ? ap.pin : 0) * MAXB + et) * NPAIR + row / 2);
      uint2 rk[TOPK];
      const uint2* src = c.w2ct + ((long)(par * MAXB + et) * TOPK) * H + row;
#pragma unroll
      for (int k = 0; k < TOPK; k++) rk[k] = ld_tag(src + (long)k * H);
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; k++) s += __uint_as_float(wait_tag(src + (long)k * H, rk[k], TAGA(a)));
      const float x = rbf(s);
      const bf16* w4 = reinterpret_cast<const bf16*>(&tw);
      float acc = __fmul_rn(x3, bf(w4[0]));
      acc = __fadd_rn(acc, __fmul_rn(x2, bf(w4[1])));
      acc = __fadd_rn(acc, __fmul_rn(x1, bf(w4[2])));
      acc = __fadd_rn(acc, __fmul_rn(x, bf(w4[3])));
      const float ms = rbf(__fmul_rn(__fadd_rn(acc, x), 0.44721359549995793f));
      c.hm[((long)ap.slot * MAXB + et) * 3 * H + (t % 3) * H + row] = tobf(x);
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

// the softcap of the head, 30 tanh(z / 30), as the track-A exact path computes it (under --use_fast_math: an approximate
// tanh; the SKH certificate uses a measured bound on its error, see mk_cap_eps)
__device__ __forceinline__ float head_cap(float a) { return 30.0f * tanhf(a / 30.0f); }

// ------------------------------------------------------------------------------------------------ SKH (track B)
// Certified int8 sketch of the unembedding for the greedy path (hd.logits null: generate() and the serving loop; step()
// with logits keeps the exact full head). Ported from s15-mini-b200-b2 (s0009 / s0047). Part 1 (sk_bound): per row r an
// int8 code s_r q_r ~ w_r, x carried as sigma (d1 + d2 / 254) with int8 digits, s8 MMA -> an interval [lo_r, up_r] that
// contains the fp32 value a_r the exact path computes; per strip the max of up, grid-wide per token L = max_r lo_r.
// A grid barrier. Part 2 (sk_exact): with eps >= |head_cap(a) - 30 tanh(a / 30)| over every finite float a (measured
// exhaustively at load, mk_cap_eps) a strip is skipped only if 30 tanh(u / 30) + 2 eps + 1e-6 < 30 tanh(L / 30) (double):
// then every row of it has head_cap(a_r) <= G(a_r) + eps <= G(u) + eps < G(L) - eps <= head_cap(a_rL), strictly below
// the row that gave L, so it cannot hold the argmax (keys tie-break on the row; a strict inequality never ties).
// The other strips are recomputed with the track-A MMA chain (same operands, same order) -> the same keys.
// State: c.skl (unsigned [64]): [8 par + tok] L keys of step parity par (reset by block 0 after the step's last
// barrier, reused two steps later); [32] exact strips, [33] sketch steps, [34] strips without a stored bound,
// [35] max exact strips in a step, [36] steps with no usable L (all strips exact), [40 + par] per-step count,
// [48 + min(n, 15)] histogram of exact strips per step.
#ifndef SKH
#define SKH 0
#endif
#if SKH
#ifndef SK_UNR
#define SK_UNR 12
#endif
#ifndef SK_NI
#define SK_NI __noinline__
#endif
#ifndef SK_FP
#define SK_FP 6.0e-4f   // |a_r - z_r| of the fp32 MMA chain <= SK_FP n_r |x| (2304 terms x 2^-23 = 2.75e-4, x2 margin)
#endif
#define SK_SPW 4        // strips per warp with a stored bound (12504 strips / (148 blocks x 24 warps) = 3.5)
struct SkScr {          // overlays sm.red4 (block GEMV scratch, unused between ph_final and the end of the step)
  float u[NWARP * SK_SPW][MAXB];
  float sig[MAXB], xn[MAXB];
  int cand[NWARP * SK_SPW]; int ncand;
};
static_assert(sizeof(SkScr) <= sizeof(float4) * NWARP * 32, "SkScr must fit in sm.red4");
__device__ __forceinline__ void mma_s8(int* d, uint4 a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float sk_wmax(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}
__device__ __forceinline__ unsigned sk_fsort(float z) { unsigned u = __float_as_uint(z); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
__device__ __forceinline__ float sk_funsort(unsigned u) { return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u); }

// part 1: digits of x (sm.u.xs[0], the exact path's bf16 operand), sketch GEMV, bounds
__device__ SK_NI void sk_bound(const Ctx& c, Smem& sm, const int B, const int par) {
  const int w = threadIdx.x >> 5, lane = lane_id(), g = lane >> 2, t = lane & 3;
  SkScr& sk = *reinterpret_cast<SkScr*>(&sm.red4[0][0]);
  constexpr int HP2 = 2 * H + 8;   // row stride = 4 words mod 32: the 8 x 4 lanes of a B fragment load hit 32 banks
  int8_t* dig = reinterpret_cast<int8_t*>(&sm.u.xs[1][0][0]);   // [MAXB][2][HP2] = 73856 B (in xs[1], xs[2])
  // all tokens in one sweep: max |x| and |x|^2 per token, one block reduction
  constexpr int NB = KB1 ? 1 : MAXB;
  float mx[NB], ss[NB];
#pragma unroll
  for (int b = 0; b < NB; b++) {
    mx[b] = 0.f; ss[b] = 0.f;
    if (b < B)
      for (int k = threadIdx.x; k < H; k += NTH) { const float v = bf(sm.u.xs[0][b][k]); mx[b] = fmaxf(mx[b], fabsf(v)); ss[b] = fmaf(v, v, ss[b]); }
  }
  block_sum_max<NB, NB>(ss, mx, sm);
#pragma unroll
  for (int b = 0; b < NB; b++) {
    if (b >= B) break;
    // sigma = the smallest power of two with 127 sigma >= max|x| (exact comparisons): |x / sigma| <= 127, no digit
    // clamps, and y = x * (1 / sigma) is exact (a power-of-two scaling; fast math does not enter)
    float sig = 1.0f, rs = 1.0f;
    if (mx[b] > 1e-30f && mx[b] < 1e30f) {   // (else sigma = 1: tiny x is within the bound; huge x overflows |x|: all exact)
      sig = __uint_as_float(__float_as_uint(mx[b] * (1.0f / 127.0f)) & 0x7f800000u);   // 2^floor(log2(~max / 127))
      while (127.0f * sig < mx[b]) sig *= 2.0f;
      while (127.0f * (0.5f * sig) >= mx[b]) sig *= 0.5f;
      rs = __uint_as_float((254u - (__float_as_uint(sig) >> 23)) << 23);                  // 1 / sig, exact
    }
    if (threadIdx.x == 0) { sk.sig[b] = sig; sk.xn[b] = __fsqrt_ru(ss[b]) * 1.00001f + 1e-30f; }
    for (int k = threadIdx.x; k < H; k += NTH) {
      const float y = bf(sm.u.xs[0][b][k]) * rs;
      const float d1 = fminf(fmaxf(rintf(y), -127.f), 127.f);
      const float d2 = fminf(fmaxf(rintf((y - d1) * 254.0f), -127.f), 127.f);
      dig[(b * 2 + 0) * HP2 + k] = (int8_t)d1; dig[(b * 2 + 1) * HP2 + k] = (int8_t)d2;
    }
  }
  __syncthreads();
#ifdef MKPROF
  SUBMARK(62);   // (was an unguarded write: null c.prof in MKPROF builds without a profile buffer)
#endif
  float lmax[2] = {-INFINITY, -INFINITY};
  int ns = 0;
  for (int it = blockIdx.x + gridDim.x * w; it < UN_STRIPS; it += gridDim.x * NWARP, ns++) {
    int a1[4] = {0, 0, 0, 0}, a2[4] = {0, 0, 0, 0};
    const uint8_t* qp = c.skq + (long)it * 72 * 512 + lane * 16;
    const int8_t* d1p = dig + (g * 2 + 0) * HP2 + 4 * t;
    const int8_t* d2p = dig + (g * 2 + 1) * HP2 + 4 * t;
    const bool gok = g < B;
#pragma unroll 1
    for (int kk = 0; kk < 72; kk += SK_UNR) {
      uint4 q[SK_UNR];
#pragma unroll
      for (int j = 0; j < SK_UNR; j++) q[j] = ldg_nc(qp + (long)(kk + j) * 512);
#pragma unroll
      for (int j = 0; j < SK_UNR; j++) {
        const int k0 = (kk + j) * 32;
        const uint32_t b10 = gok ? *reinterpret_cast<const uint32_t*>(d1p + k0) : 0u, b11 = gok ? *reinterpret_cast<const uint32_t*>(d1p + k0 + 16) : 0u;
        const uint32_t b20 = gok ? *reinterpret_cast<const uint32_t*>(d2p + k0) : 0u, b21 = gok ? *reinterpret_cast<const uint32_t*>(d2p + k0 + 16) : 0u;
        mma_s8(a1, q[j], b10, b11); mma_s8(a2, q[j], b20, b21);
      }
    }
    float up[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = it * 16 + g + 8 * h;
      const float sr = __ldg(c.sks + row), er = __ldg(c.ske + row), nr = __ldg(c.skn + row);
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        const int tok = 2 * t + cc;
        if (tok >= B || row >= VOCAB_REAL) continue;
        // z_r = s_r q_r.x^ + (w_r - s_r q_r).x^ + w_r.(x - x^): |z_r - av| <= e_r |x^| + n_r |x - x^| (+ rounding of av);
        // |x - x^| <= 48 sigma (1 / 508 + 1e-5); |a_r - z_r| <= SK_FP n_r |x|; 1.01 and 1e-6 cover the fp32 rounding of E, av
        const float sig = sk.sig[tok], xn = sk.xn[tok];
        const float dx = 48.0f * sig * (1.0f / 508.0f + 1e-5f);
        const float S = (float)a1[2 * h + cc] + (float)a2[2 * h + cc] * (1.0f / 254.0f);
        const float av = sr * sig * S;
        const float E = (er * (xn + dx) + nr * dx + SK_FP * nr * xn + 1e-6f * fabsf(av) + 1e-6f) * 1.01f;
        up[cc] = fmaxf(up[cc], av + E);
        lmax[cc] = fmaxf(lmax[cc], av - E);
      }
    }
#pragma unroll
    for (int cc = 0; cc < 2; cc++) {
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) up[cc] = fmaxf(up[cc], __shfl_xor_sync(0xffffffff, up[cc], o));
      if (g == 0 && ns < SK_SPW) sk.u[w * SK_SPW + ns][2 * t + cc] = up[cc];
    }
  }
#pragma unroll
  for (int cc = 0; cc < 2; cc++) {
    float l = lmax[cc];
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) l = fmaxf(l, __shfl_xor_sync(0xffffffff, l, o));
    const int tok = 2 * t + cc;
    if (g == 0 && tok < B && l > -INFINITY) atomicMax(c.skl + par * 8 + tok, sk_fsort(l));
  }
}

// part 2 (after a grid barrier): candidate strips of this block, exact recompute, argmax keys
__device__ SK_NI void sk_exact(const Ctx& c, Smem& sm, const int B, const int par) {
  const int w = threadIdx.x >> 5, lane = lane_id(), g = lane >> 2, t = lane & 3;
  SkScr& sk = *reinterpret_cast<SkScr*>(&sm.red4[0][0]);
  double T[2];   // per token: strips with u < T are skipped; G(T) = G(L) - 2 eps - 1e-6 (G = 30 tanh(. / 30), in double)
  bool nol = false;
#pragma unroll
  for (int cc = 0; cc < 2; cc++) {
    const int tok = 2 * t + cc;
    T[cc] = -INFINITY;
    if (tok >= B) continue;
    const double L = (double)sk_funsort(ld_relaxed_gpu(c.skl + par * 8 + tok));   // never set: NaN
    const double cl = 30.0 * tanh(L / 30.0) - 2.0 * (double)c.skeps - 1e-6;
    if (cl > -30.0) T[cc] = 30.0 * atanh(cl / 30.0); else nol = true;              // (NaN: no strip skipped)
  }
  if (threadIdx.x == 0) sk.ncand = 0;
  __syncthreads();
  int ns = 0, nfb = 0;
  for (int it = blockIdx.x + gridDim.x * w; it < UN_STRIPS; it += gridDim.x * NWARP, ns++) {
    bool cand = false;
#pragma unroll
    for (int cc = 0; cc < 2; cc++) {
      const int tok = 2 * t + cc;
      if (tok >= B) continue;
      const float u = ns < SK_SPW ? sk.u[w * SK_SPW + ns][tok] : INFINITY;
      if (!((double)u < T[cc])) cand = true;   // (NaN u: exact)
    }
    if (ns >= SK_SPW) nfb++;
    if (__any_sync(0xffffffff, cand) && lane == 0) { const int q = atomicAdd(&sk.ncand, 1); if (q < NWARP * SK_SPW) sk.cand[q] = it; }
  }
  __syncthreads();
  const int ncand = sk.ncand > NWARP * SK_SPW ? 0 : sk.ncand;
  if (threadIdx.x == 0 && sk.ncand) { atomicAdd(c.skl + 32, (unsigned)sk.ncand); atomicAdd(c.skl + 40 + par, (unsigned)sk.ncand); }
  if (lane == 0 && nfb) atomicAdd(c.skl + 34, (unsigned)nfb);
  if (blockIdx.x == 0 && __any_sync(0xffffffff, nol) && threadIdx.x == 0) atomicAdd(c.skl + 36, 1u);
  unsigned long long best[2] = {0ull, 0ull};
  // one strip on one warp with the track-A piece (the same function, the same order): W12 storage, and the fallback
  // when a block has more candidates than it can list (grids below 131 blocks): then all its strips are exact
  auto warp_strip = [&](int it) {
    BSm bl{&sm.u.xs[0][0][0], HP, B};
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#ifdef W12
    piece_w12(c.un12, it, 36, 0, 36, bl, acc);
#else
    piece_bf16(c.unembed + (long)it * 36 * 2048, 0, 36, bl, acc);
#endif
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int row = it * 16 + g + 8 * h;
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        const int tok = 2 * t + cc;
        if (tok >= B || row >= VOCAB_REAL) continue;
        const unsigned long long key = ((unsigned long long)sk_fsort(head_cap(acc[2 * h + cc])) << 32) | (unsigned)(0xFFFFFFFFu - row);
        if (key > best[cc]) best[cc] = key;
      }
    }
  };
  if (sk.ncand > NWARP * SK_SPW) {
    for (int it = blockIdx.x + gridDim.x * w; it < UN_STRIPS; it += gridDim.x * NWARP) warp_strip(it);
  } else {
#ifdef W12
  for (int q = w; q < ncand; q += NWARP) warp_strip(sk.cand[q]);
#else
  // each candidate strip staged into shared memory by the whole block in one round of loads, then the track-A MMA
  // chain of piece_bf16 (same operands, units u = 0..35, quarters 0..3 in order) by warp 0
  uint8_t* stg = reinterpret_cast<uint8_t*>(&sm.u.xs[1][0][0]);   // 73984 B >= 73728 (the digits are no longer needed)
  for (int q = 0; q < ncand; q++) {
    const int it = sk.cand[q];
    const uint8_t* src = c.unembed + (long)it * 36 * 2048;
    for (int k = threadIdx.x; k < 36 * 2048 / 16; k += NTH)
      *reinterpret_cast<uint4*>(stg + k * 16) = ldg_nc(src + (long)k * 16);
    __syncthreads();
    if (w == 0) {
      BSm bl{&sm.u.xs[0][0][0], HP, B};
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      const uint32_t sb = (uint32_t)__cvta_generic_to_shared(stg) + lane * 16;
#pragma unroll 4
      for (int u = 0; u < 36; u++) {
#pragma unroll
        for (int qq = 0; qq < 4; qq++) {
          uint4 wv;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w) : "r"(sb + (u * 4 + qq) * 512));
          const uint2 bb = bl.get(u * 64 + qq * 16);
          mma_bf16(acc, wv.x, wv.y, wv.z, wv.w, bb.x, bb.y);
        }
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int row = it * 16 + g + 8 * h;
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
          const int tok = 2 * t + cc;
          if (tok >= B || row >= VOCAB_REAL) continue;
          const unsigned long long key = ((unsigned long long)sk_fsort(head_cap(acc[2 * h + cc])) << 32) | (unsigned)(0xFFFFFFFFu - row);
          if (key > best[cc]) best[cc] = key;
        }
      }
    }
    __syncthreads();
  }
#endif
  }
#pragma unroll
  for (int cc = 0; cc < 2; cc++) {
    unsigned long long k = best[cc];
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) { const unsigned long long ok = __shfl_xor_sync(0xffffffff, k, o); if (ok > k) k = ok; }
    const int tok = 2 * t + cc;
    if (g == 0 && tok < B && k) atomicMax(c.amax + tok, k);
  }
}
#endif

// final: output mix -> norm -> unembed -> logits, argmax
__device__ void ph_final(PH_ARGS) {
  const int B = HDB;
  SUBMARK(60);
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
#if SKH
  if (!hd.logits) { SUBMARK(61); sk_bound(c, sm, B, (int)((hd.tag0 >> 6) & 1u)); SUBMARK(63); return; }   // part 2 after a grid barrier
#endif
  const int w = threadIdx.x >> 5, lane = lane_id(), g = lane >> 2, t = lane & 3;
  unsigned long long best[2] = {0ull, 0ull};
  for (int it = blockIdx.x + gridDim.x * w; it < UN_STRIPS; it += gridDim.x * NWARP) {
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
        const float z = head_cap(acc[2 * h + cc]);
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
extern "C" __global__ void __launch_bounds__(NTH, 1) mk_step(const Ctx* __restrict__ cp, Hdr hd0) {
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
    int ns = ph_attn_in(c, hd, sm, a);
    if (ns < 0) {
      GBAR(); MARK(7);
      ph_gate_d(c, hd, sm, a, 0, 3);
      GBAR(); MARK(8);
      ph_gate_fg(c, hd, sm, a, 0, 3);
      GBAR(); MARK(9);
      const LayerW& L = c.L[ap.layer];
      float wv[3][EPT];
      for (int q = 0; q < 3; q++) load_w(L.norm[L.glob ? q : 0], wv[q]);
      const bool bc = HDB >= 2;
      for (int b = bc ? (int)blockIdx.x : 0; b < HDB; b += bc ? (int)gridDim.x : 1) {
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
        if (bc) {
          uint2* xs_pub = c.xst + (long)(a & 1) * 3 * MAXB * NPAIR;
          uint2* dst[3] = {xs_pub + (long)b * NPAIR, xs_pub + (long)(MAXB + b) * NPAIR, xs_pub + (long)(2 * MAXB + b) * NPAIR};
          norm_publish<3>(x, wv, 3, dst, TAGA(a), sm);
        } else {
          bf16* dst[3] = {&sm.u.xs[0][b][0], &sm.u.xs[1][b][0], &sm.u.xs[2][b][0]};
          rmsnorm_store_n<3>(x, wv, 3, dst, sm);
        }
      }
      if (bc) {
        const uint2* xs_pub = c.xst + (long)(a & 1) * 3 * MAXB * NPAIR;
        for (int q = 0; q < 3; q++)
          poll_rows(xs_pub + (long)q * MAXB * NPAIR, HDB, [&](int r) { return &sm.u.xs[q][r][0]; }, TAGA(a));
      }
      __syncthreads();
      ns = 3;
    }
    MARK(10);
    ph_qkv(c, hd, sm, a, ns);
    GBAR(); MARK(2);
#if ATT3
    if (HDB >= 3 && HDB * NKV * ((hd.pos - (c.L[ap.layer].glob ? 0 : max(0, hd.pos + 1 - 1024)) + 64) / 64) > ATT3_MIN) ph_attn3(c, hd, sm, a);
    else
#endif
    ph_attn(c, hd, sm, a);
    GBAR(); MARK(3);
    ph_oproj(c, hd, sm, a);
    GBAR(); MARK(4);
    int r = ph_mlp_in(c, hd, sm, a);
    if (r < 0) {
      GBAR(); MARK(14);
      ph_gate_d(c, hd, sm, a, 3, 1);
      GBAR(); MARK(15);
      ph_gate_fg(c, hd, sm, a, 3, 1);
      GBAR(); MARK(16);
      if (HDB >= 2) {
        for (int b = blockIdx.x; b < HDB; b += gridDim.x) {
          float x[1][EPT], wv[EPT];
          load_w(c.L[ap.layer].mlp_norm, wv);
          load_vec_f(c.xpt + ((long)((a & 1) * MAXB + b) * 4 + 3) * H, TAGA(a), x[0]);
          mlp_publish(c, hd, sm, a, x, b, wv);
        }
        mlp_poll(c, hd, sm, a);
      } else {
        for (int b = 0; b < HDB; b++) {
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
    routing(c, hd, sm, a);
    MARK(12);
    ph_w13(c, hd, sm, a);
    GBAR(); MARK(6);
    ph_w2a(c, hd, sm, a);
    GBAR(); MARK(17);
    ph_w2b(c, hd, sm, a);
    GBAR();
  }
  MARK(13);
  ph_final(c, hd, sm);
#if SKH
  const int skp = (int)((hd.tag0 >> 6) & 1u);   // step parity: the sketch's per-step words alternate between two sets
  if (!hd.logits) { MARK(19); grid_bar(c, nbar, base); MARK(20); sk_exact(c, sm, HDB, skp); }
#endif
  MARK(MKPROF_WAIT);
  grid_bar(c, nbar, base);
  if (blockIdx.x == 0) {
    if (threadIdx.x < HDB) {
      const unsigned long long k = c.amax[threadIdx.x];
      const int tok = (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFu));
      c.next_tok[threadIdx.x] = tok;
      c.amax[threadIdx.x] = 0ull;
#if SKH
      if (!hd.logits) c.skl[skp * 8 + threadIdx.x] = 0u;   // next used two steps later, after the next step's barriers
#endif
      st_tag(c.ntt + threadIdx.x, (uint32_t)tok, (hd.tag0 + 60u) & 0x7fffffffu);
      if (hd.out_tok) hd.out_tok[threadIdx.x] = tok + 1;   // 0 = not yet written (host polls)
    }
#if SKH
    if (!hd.logits && threadIdx.x == 32) {   // statistics of this step (every block's count is in: after the barrier)
      const unsigned n = atomicExch(c.skl + 40 + skp, 0u);
      atomicAdd(c.skl + 33, 1u); atomicMax(c.skl + 35, n); atomicAdd(c.skl + 48 + min(n, 15u), 1u);
    }
#endif
  }
  }   // steps
  if (blockIdx.x == 0 && threadIdx.x == 0) c.bar[1] = base + nbar * gridDim.x;
}

#if SKH
// every finite float32 bit pattern (all 2^32 minus Inf / NaN): the error of the exact path's softcap
extern "C" __global__ void mk_capcheck(unsigned long long* out) {
  double em = 0.0;
  for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < (1ull << 32); i += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned bits = (unsigned)i;
    if (((bits >> 23) & 0xFFu) == 0xFFu) continue;
    const float a = __uint_as_float(bits);
    const double e = fabs((double)head_cap(a) - 30.0 * tanh((double)a / 30.0));
    em = e > em ? e : em;
  }
  em = fmax(em, __shfl_xor_sync(0xffffffff, em, 16)); em = fmax(em, __shfl_xor_sync(0xffffffff, em, 8));
  em = fmax(em, __shfl_xor_sync(0xffffffff, em, 4)); em = fmax(em, __shfl_xor_sync(0xffffffff, em, 2));
  em = fmax(em, __shfl_xor_sync(0xffffffff, em, 1));
  if ((threadIdx.x & 31) == 0) atomicMax(out, (unsigned long long)__double_as_longlong(em));
}
#endif

// ------------------------------------------------------------------------------------------------ host API
extern "C" {
int mk_ctx_bytes() { return (int)sizeof(Ctx); }
int mk_layer_bytes() { return (int)sizeof(LayerW); }
int mk_app_bytes() { return (int)sizeof(App); }
int mk_hdr_bytes() { return (int)sizeof(Hdr); }
int mk_smem_bytes() { return (int)sizeof(Smem); }
int mk_nth() { return NTH; }
int mk_kb1() { return KB1; }
int mk_cn_total() { return CN_TOTAL; }
int mk_skh() { return SKH; }
// SKH: max over every finite float a of |head_cap(a) - 30 tanh(a / 30)| (double reference), exhaustive, on the device
double mk_cap_eps() {
#if SKH
  unsigned long long* d = nullptr;
  if (cudaMalloc(&d, 8) != cudaSuccess) return -1.0;
  cudaMemset(d, 0, 8);
  mk_capcheck<<<1184, 512>>>(d);
  unsigned long long h = 0;
  const cudaError_t e = cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost);
  cudaFree(d);
  if (e != cudaSuccess) { fprintf(stderr, "mk_cap_eps: %s\n", cudaGetErrorString(e)); return -1.0; }
  double r; memcpy(&r, &h, 8);
  return r;
#else
  return -1.0;
#endif
}
#ifdef W12
int mk_w12() { return 1; }
#else
int mk_w12() { return 0; }
#endif
int mk_setup() {
  cudaError_t e = cudaFuncSetAttribute(mk_step, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(Smem));
  if (e != cudaSuccess) { fprintf(stderr, "mk_setup: %s\n", cudaGetErrorString(e)); return 1; }
  int nb = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mk_step, NTH, sizeof(Smem));
  if (nb < 1) { fprintf(stderr, "mk_setup: occupancy %d\n", nb); return 2; }
  return 0;
}
int mk_launch(const void* ctx_dev, const Hdr* hd, int nblk, void* stream) {
  Hdr h = *hd;
  void* args[] = {(void*)&ctx_dev, (void*)&h};
  cudaError_t e = cudaLaunchCooperativeKernel((const void*)mk_step, dim3(nblk), dim3(NTH), args, sizeof(Smem), (cudaStream_t)stream);
  if (e != cudaSuccess) { fprintf(stderr, "mk_launch: %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
void* mk_host_alloc(size_t n) { void* p = nullptr; if (cudaHostAlloc(&p, n, cudaHostAllocMapped) != cudaSuccess) return nullptr; memset(p, 0, n); return p; }
}
