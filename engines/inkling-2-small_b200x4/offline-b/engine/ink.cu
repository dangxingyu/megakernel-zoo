// Inkling-2-Small decode engine: one persistent kernel per card per step, four cards.
// Attention TP4 by heads, experts TP4 inside every expert (intermediate 576 per card), everything else replicated.
// See notes/plan.md for the design and the layouts.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

typedef __nv_bfloat16 bf16;
// The attention, routing and w2-quantization phases are compiled out of line: the persistent kernel sits at the
// register cap (REG:80 at 768 threads) and these bodies, inlined, took registers from the GEMV loops. Measured on
// fixed tokens (e0125/e0126): B=8 4858 -> 4770 us, B=1 3270 -> 3261; with the team phases out of line as well the
// kernel spills (STACK:104) and B=1 gets slower.
#ifndef PHASES_INLINE
#define PHASE_NI __noinline__
#else
#define PHASE_NI
#endif

#define H 4608
#define NL 48
#define NE 384
#define TOPK 8
#define ILOC 576
#define HD 128
#define NQL 12          // local q heads
#define NKVL 2          // local kv heads
#define DREL 16
#define RELEXT 1024
#define WIN 1024
#define NCARD 4
#define MAXB 8
#ifndef NTH
#define NTH 768   // 85 registers per thread (1024 threads: 64 and heavy spills; measured 5.95 -> 5.40 ms at B=1, e0056)
#endif
#define NWARP (NTH / 32)
#define CHUNK 128       // attention positions per block
#define MAXCHUNK 17     // up to 2176 positions
#define MAXU 64         // max union experts per layer (B<=8)
#define VOCAB_REAL 200058
#ifndef MAXSLOT
#define MAXSLOT 24
#endif
#define NSL 18           // team slices per token: 256 elements (one warp, 8 per lane) each
#define NRID 10
#ifndef SM_TOK
#define SM_TOK 768        // elements of the token state per block (H for the old one-block-per-token code)
#endif
#ifndef SM_OPART
#define SM_OPART 3072
#endif

// ------------------------------------------------------------------------------------------------ parameters
struct W12M {               // exact 12-bit code of a BF16 matrix (engine/w12.py)
  const uint8_t* data;      // [strips][units][3][32][16]
  const uint8_t* base;      // [rows] window base exponent
  const int* hdr;           // [strips*units] escape count | offset << 8
  const int* esc;           // entries: lane<<27 | reg<<23 | half<<22 | bf16 bits
};
struct LayerW {
  const uint8_t* wqkv;      // bf16 frag [strips][U=72][4 j][32 lanes][16 B]
  const uint8_t* wo;        // bf16 frag [288][24]...
  const uint8_t* wr;        // router bf16 frag [24][72]
  const float* rbias;       // [384]
  const uint8_t* w13;       // [384][72 strips][72 units][UB]
  const uint8_t* w2;        // [384][288 strips][9 units][UB]
  const float* f13;         // [384] weight outer factor (nvfp4), unused for mxfp8
  const float* f2;
  const uint8_t* w13t;      // nvfp4 w13 in the tcgen05 layout [384][9 row groups][72 K steps][4608 B] (batches >= TC_MINB)
  const uint8_t* w2t;       // nvfp4 w2 in the tcgen05 layout [384][36 row groups][9 K steps][4608 B]
  const bf16* norm[3];      // attn_norm (local layers: all three the same) or q/k/v_input_norm
  const bf16* mlp_norm;
  const bf16* ksc;          // local: [256][4] (local channels); global: static_w [4][256]
  const bf16* vsc;
  const bf16* attn_sc;      // [4608][4]
  const bf16* mlp_sc;
  const bf16* qscale;       // [12][128]
  const bf16* rscale;       // [12][16]
  const bf16* relp;         // [1024][16] (distance-major: a row is one A-fragment row of the bias MMA)
  const float* aq;          // attention residual queries [nq][4608] (layer 0: null)
  const float* mq;          // [4608]
  W12M qkv12, wo12, wr12;
  int glob;                 // global layer
  int mx;                   // experts are mxfp8
  int qkv_strips;           // 237 or 239 (layer 0 with its token tables, track B: 192, the query and gate strips)
  int qkv_skip;             // leading strips of wqkv that are neither prefetched nor computed beside the gate split
                            // (0; layer 0 with its token tables: the 96 query strips)
};

struct Comm {                       // lives on each card, written by all cards
  // low-latency exchange: every value travels with its tag in one 8-byte store; buffers alternate by layer parity
  uint2 lla[2][NCARD][MAXB][H];       // o_proj partial per source card, token-major: (float bits, tag)
  uint2 llm[2][NCARD][MAXB][H];       // w2 partial
  uint2 llx[NL][NCARD][MAXU][MAXB];   // per source card: max |act| of its 576-slice per (union expert, token)
  float cand_v[NCARD][MAXB][256];   // argmax candidates per source card per block
  int cand_i[NCARD][MAXB][256];
  unsigned cnt[8];                  // 0 attn, 1 mlp, 2 amax, 3 final
};

struct Hdr {
  int card, B, nblk, pos, parity, want_logits, gen_idx, gen_n;
  unsigned xbase[4];
  unsigned tag0, pad1;                // tags of this step: tag0 + layer + 1; pad1: first layer | last layer << 8 | mode << 16
  float* logits; int* tokens; int* gen_out;
};
struct Ctx {
  Hdr h;
  LayerW L[NL];
  const bf16* embed; const bf16* embed_norm; const float* oq; const bf16* fnorm;
  const uint8_t* unembed; int un_strips; int un_row0;
  W12M un12;
  bf16* blocks;     // [9][MAXB][H] finished blocks (block 0 = h0)
  bf16* partial;    // [MAXB][H]
  bf16* attn_s;     // [MAXB][H]  attention output after its sconv (current layer)
  bf16* hist_attn;  // [NL][MAXB][3][H]
  bf16* hist_mlp;   // [NL][MAXB][3][H]
  bf16* hist_k;     // [NL][MAXB][3][256]
  bf16* hist_v;
  bf16* kc; bf16* vc; int maxctx; int pad0;  // [NL][MAXB][2][maxctx][128]
  bf16* xfrag;      // [3][72 units][32][4][2][2] bf16 frags of xq/xk/xv
  bf16* x2frag;     // router frag of the mlp input (bf16)
  uint8_t* w13frag; // quantized mlp input: f16 frag (nvfp4) or e4m3 frag (mxfp8)
  float* w13sc;     // [MAXB] outer (nvfp4) or [144][MAXB] block scales (mxfp8)
  bf16* qkvo;       // [MAXB][3840]
  float* rlog;      // [MAXB][384]
  int* ids; float* wts;   // [MAXB][8]
  int* uni; int* nuni;
  bf16* act;        // [MAXU][MAXB][576]
  uint8_t* w2frag;  // [MAXU][9 units] frags
  float* w2sc;      // [MAXU][MAXB] / [MAXU][18][MAXB]
  float* achunk;    // [MAXB][2][MAXCHUNK][6][2+128]
  unsigned* acnt;   // [MAXB][2]
  bf16* ofrag;      // o_proj B frag
  bf16* hfrag;      // final hidden frag
  float* part;      // gemv partial slots
  unsigned* scnt;   // per-strip arrival counters
  int* next_tok;    // [MAXB]
  unsigned* bar;    // [0] counter, [1] base
  Comm* comm[NCARD];
  unsigned long long* tprof;  // optional [NL+1][16] globaltimer stamps of block 0
  bf16* dbg;        // optional [NL][2][MAXB][H]: attention and mlp outputs after their sconv
  int* dbg_ids;     // optional [NL][MAXB][8]
  unsigned* amax_loc;  // [MAXU][MAXB] local running max (float bits), reset by the last strip of each expert
  unsigned* ucnt;      // [MAXU] strips finished per union expert
  uint2* rll;          // [MAXB][384] router logits as (float bits, tag): routing polls them (no grid barrier)
  uint2* tred;         // [NRID][MAXB][32][NSL] team reductions: (float bits, tag) per (value, slice)
  uint2* gll;          // [MAXB][1536] sigmoid(output gate) as (float bits, tag), when the gate GEMV runs beside attention
  uint2* ctag;         // [MAXB][NKVL][MAXCHUNK] "this chunk's record is written" tags of the attention
  int* prog;           // host-mapped progress words of the tensor-core kernel (the host's stall report reads them);
                       // here and not in the per-launch header: 8 more bytes of kernel parameters showed as STACK:104
  // track B, layer-0 token tables (engine/api.py _build_l0_tables). The launch mode travels in h.pad1 >> 16:
  // 0 = compute everything; 1 = compute everything and store the normed embedding of each token at l0emb (the
  // build); 2 = `embed` holds the normed embedding and l0tab the token's layer-0 projection rows.
  const bf16* l0tab;   // [vocab][2304]: this card's q rows (1536), then r, sink, padding, k, v rows (768) of layer 0
  bf16* l0emb;         // [vocab][H], written in mode 1
};

// ------------------------------------------------------------------------------------------------ small helpers
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
__device__ __forceinline__ uint4 ld4(const void* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint32_t cvt_fp4x2(uint32_t b) {
  uint32_t r; unsigned short bs = (unsigned short)b;
  asm("{ .reg .b8 t; cvt.u8.u16 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(r) : "h"(bs));
  return r;
}
__device__ __forceinline__ uint32_t cvt_e4m3x2_f16x2(uint32_t w16) {
  uint32_t r; unsigned short h = (unsigned short)w16;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(h));
  return r;
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.f16x2 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t r; asm("prmt.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ void mma_f16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_e4m3(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float ue8m0(uint32_t e) { return e == 0 ? __int_as_float(0x00400000) : __int_as_float(e << 23); }
__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 tobf(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float rbf(float x) { return bf(tobf(x)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ unsigned ld_relaxed_gpu(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_relaxed_sys(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void fence_acqrel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_acqrel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) { unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_release_gpu(unsigned* p, unsigned v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_relaxed_sys(unsigned* p, unsigned v) { asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_release_sys(unsigned* p, unsigned v) { asm volatile("red.release.sys.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }

// bf16/f16 B-fragment index (m16n8k16), unit-major: element (token g, k) -> index in 16-bit elements
__device__ __forceinline__ int fidx16(int g, int k) {
  int u = k >> 6, j = (k >> 4) & 3, kk = k & 15, r = kk >> 3, t = (kk & 7) >> 1, h = kk & 1;
  return ((((u * 32 + g * 4 + t) * 4 + j) * 2 + r) * 2 + h);
}
// e4m3 B-fragment index (m16n8k32), unit = 64 k: element (g, k) -> byte index
__device__ __forceinline__ int fidx8(int g, int k) {
  int u = k >> 6, s = (k >> 5) & 1, kk = k & 31, r = kk >> 4, t = (kk & 15) >> 2, e = kk & 3;
  return ((((u * 32 + g * 4 + t) * 2 + s) * 2 + r) * 4 + e);
}

// compact layouts for B == 1 (column 0 only): lanes t = 0..3 of g = 0
__device__ __forceinline__ int fidx16c(int k) {
  int u = k >> 6, j = (k >> 4) & 3, kk = k & 15, r = kk >> 3, t = (kk & 7) >> 1, h = kk & 1;
  return ((((u * 4 + t) * 4 + j) * 2 + r) * 2 + h);
}
__device__ __forceinline__ int fidx8c(int k) {
  int u = k >> 6, s = (k >> 5) & 1, kk = k & 31, r = kk >> 4, t = (kk & 15) >> 2, e = kk & 3;
  return ((((u * 4 + t) * 2 + s) * 2 + r) * 4 + e);
}
template <bool CPT> __device__ __forceinline__ int fi16(int g, int k) { return CPT ? fidx16c(k) : fidx16(g, k); }
template <bool CPT> __device__ __forceinline__ int fi8(int g, int k) { return CPT ? fidx8c(k) : fidx8(g, k); }
#define LSZ (CPT ? 4 : 32)
#define LSL (CPT ? (lane & 3) : lane)
// ------------------------------------------------------------------------------------------------ barriers
__device__ unsigned g_barbase_dummy;
__device__ __forceinline__ void grid_bar(const Ctx& c, unsigned& nbar, unsigned base) {
  __syncthreads();
  nbar += 1;
  if (threadIdx.x == 0) {
    unsigned target = base + nbar * gridDim.x;
    red_release_gpu(&c.bar[0], 1);
#ifndef GB_BACKOFF
#define GB_BACKOFF 0
#endif
    while ((int)(ld_relaxed_gpu(&c.bar[0]) - target) < 0) { if (GB_BACKOFF) __nanosleep(GB_BACKOFF); }
    fence_acqrel_gpu();
  }
  __syncthreads();
}
// cross-card events. Host keeps xbase[k] = events of kind k before this launch.
#ifndef XSIG_FLAT
#define XSIG_HIER
#endif
#ifdef XSIG_HIER
// local grid barrier, then block 0 signals every card once (measured: 152 per-block system fences cost 5-10 us)
#define XPER 1u
#else
// every block of every card signals counter k on every card
#define XPER gridDim.x
#endif
__device__ __forceinline__ void xsignal(const Ctx& c, int k, unsigned& nbar, unsigned base);
__device__ __forceinline__ void xwait(const Ctx& c, int k, unsigned events) {
  if (threadIdx.x == 0) {
    unsigned target = (c.h.xbase[k] + events) * NCARD * XPER;
    const unsigned* p = &c.comm[c.h.card]->cnt[k];
    while ((int)(ld_relaxed_sys(p) - target) < 0) { }
    fence_acqrel_sys();
  }
  __syncthreads();
}

__device__ __forceinline__ void xsignal(const Ctx& c, int k, unsigned& nbar, unsigned base) {
#ifdef XSIG_HIER
  grid_bar(c, nbar, base);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    for (int d = 0; d < NCARD; d++) red_relaxed_sys(&c.comm[d]->cnt[k], 1);   // ordered by the fence above
  }
#else
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    for (int d = 0; d < NCARD; d++) red_relaxed_sys(&c.comm[d]->cnt[k], 1);   // ordered by the fence above
  }
#endif
}
// Same-card exchange through L2: stores and loads that bypass L1 (a plain store stays in the writer's L1, a plain load
// may be served from a stale L1 line; tools/bench/red.cu). Unlike ld.relaxed these loads overlap (37 ns from L2;
// a relaxed load costs 0.1-0.3 us and a thread's relaxed loads run one after the other).
__device__ __forceinline__ void put_l2(uint2* p, float v, unsigned tag) {
#ifdef L2X_RELAXED_ST
  asm volatile("st.relaxed.gpu.global.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(__float_as_uint(v)), "r"(tag) : "memory");
#else
  asm volatile("st.global.L1::no_allocate.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(__float_as_uint(v)), "r"(tag) : "memory");
#endif
}
__device__ __forceinline__ uint2 ld_l2_v2(const void* p) {
  uint2 r;
  // (the L1-bypassing plain loads faulted here with an illegal address, e0056/e0057: relaxed loads)
  asm volatile("ld.relaxed.gpu.global.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p) : "memory");
  return r;
}
// Stores that go to L2 at once (not into the writer's L1): data another SM reads right after an atomic or a tag,
// without a fence in between (a fence costs ~0.5 us; the w13 strip hand-off and epilogue had four of them).
#ifndef FENCED
__device__ __forceinline__ void st_l2(float4* p, float4 v) {
  asm volatile("st.global.L1::no_allocate.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void st_l2(float2* p, float2 v) {
  asm volatile("st.global.L1::no_allocate.v2.f32 [%0], {%1, %2};" :: "l"(p), "f"(v.x), "f"(v.y) : "memory");
}
__device__ __forceinline__ void st_l2(bf16* p, bf16 v) {
  unsigned short b = __bfloat16_as_ushort(v);
  asm volatile("st.global.L1::no_allocate.u16 [%0], %1;" :: "l"(p), "h"(b) : "memory");
}
__device__ __forceinline__ void st_l2(float* p, float v) {
  asm volatile("st.global.L1::no_allocate.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
}
__device__ __forceinline__ void st_l2(uint4* p, uint4 v) {
  asm volatile("st.global.L1::no_allocate.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
#else
template <class T> __device__ __forceinline__ void st_l2(T* p, T v) { *p = v; }
#endif
// Fences that stay: before the per-expert strip counter and the amax publish (w13 epilogue) and before the chunk
// counter (attention). Without them a consumer saw the count before the data: atomics and stores to different
// lines are not ordered (B=8 runs differed after 30-70 steps, e0061/e0062). The strip hand-off of the flat GEMV
// needs no fence: every slot word carries its tag and the reader spins until the tags are current.
#define FENCE_E() __threadfence()
#define FENCE_A() __threadfence()
__device__ __forceinline__ void put_gpu(uint2* p, float v, unsigned tag) {
  asm volatile("st.relaxed.gpu.global.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(__float_as_uint(v)), "r"(tag) : "memory");
}
__device__ __forceinline__ uint2 ld_gpu_v2(const void* p) {
  uint2 r;
  asm volatile("ld.relaxed.gpu.global.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p) : "memory");
  return r;
}
// Tag-checked reads use relaxed loads, one instruction per (value, tag) word. Cache-global loads (ld.global.cg)
// returned current tags with wrong values after L1-bypassing or relaxed stores (e0064-e0067): not usable here.
__device__ __forceinline__ uint4 ld_gpu_v4(const void* p) {
  uint4 r;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p) : "memory");
  return r;
}
// LL protocol: producer stores (value, tag) as one 8-byte store; consumer re-reads until the tag matches
__device__ __forceinline__ void ll_put(uint2* p, float v, unsigned tag) {
  asm volatile("st.relaxed.sys.global.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(__float_as_uint(v)), "r"(tag) : "memory");
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const void* p) {
  uint4 r;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p) : "memory");
  return r;
}
__device__ __forceinline__ uint2 ld_relaxed_sys_v2(const void* p) {
  uint2 r;
  asm volatile("ld.relaxed.sys.global.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p) : "memory");
  return r;
}
// two (value, tag) pairs; spins until both tags are current
#ifndef LL_BACKOFF
#define LL_BACKOFF 32
#endif
__device__ __forceinline__ float2 ll_get2(const uint2* p, unsigned tag) {
  uint4 r = ld_relaxed_sys_v4(p);
  while (r.y != tag || r.w != tag) { __nanosleep(LL_BACKOFF); r = ld_relaxed_sys_v4(p); }
  return make_float2(__uint_as_float(r.x), __uint_as_float(r.z));
}
__device__ __forceinline__ float ll_get1(const uint2* p, unsigned tag) {
  uint2 r = ld_relaxed_sys_v2(p);
  while (r.y != tag) { __nanosleep(LL_BACKOFF); r = ld_relaxed_sys_v2(p); }
  return __uint_as_float(r.x);
}
// ------------------------------------------------------------------------------------------------ L2 prefetch
// part `part` of `nparts` of [base, base+bytes): warp 0's lanes issue bulk L2 prefetches of 32 KB.
// 32-bit arithmetic only: a 64-bit division costs ~0.1 us and this runs on critical paths (measured: 0.2 us per call
// with the 64-bit version, 59 calls per layer per block for the w2 slabs at B=8).
__device__ __forceinline__ void pf_l2(const void* p, unsigned n) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(n) : "memory");
}
__device__ __forceinline__ unsigned share16(unsigned n16, unsigned part, unsigned nparts) {   // floor(n16 * part / nparts)
  return (n16 / nparts) * part + ((n16 % nparts) * part) / nparts;
}
__device__ __forceinline__ void prefetch_l2(const void* base, long bytes, int part, int nparts, int pw = 0) {
#ifndef NO_PREFETCH
  if ((int)(threadIdx.x >> 5) != pw || nparts <= 0) return;
  const unsigned n16 = (unsigned)(bytes >> 4);
  const unsigned a = share16(n16, part, nparts), b = (part == nparts - 1) ? n16 : share16(n16, part + 1, nparts);
  const char* p = (const char*)base + (long)a * 16;
  const unsigned len = (b - a) * 16;
  for (unsigned o = (threadIdx.x & 31) * 32768u; o < len; o += 32u * 32768u) pf_l2(p + o, min(32768u, len - o));
#endif
}
// thread `item` (< nitems) prefetches this block's share of item's region [base + item * stride, +bytes); bytes/nparts <= 32 KB
__device__ __forceinline__ void prefetch_items(const void* p, unsigned bytes, int part, int nparts) {
#ifndef NO_PREFETCH
  const unsigned n16 = bytes >> 4;
  const unsigned a = share16(n16, part, nparts), b = (part == nparts - 1) ? n16 : share16(n16, part + 1, nparts);
  if (b > a) pf_l2((const char*)p + (long)a * 16, min(32768u, (b - a) * 16));
#endif
}
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define TMB(id) do { if (c.tprof && (threadIdx.x & 31) == 0 && threadIdx.x < 32) c.tprof[49 * 64 + 1184 + blockIdx.x * 16 + (id)] = gtime(); } while (0)
#define TM(li, id) do { if (c.tprof && blockIdx.x == 0 && threadIdx.x == 0) c.tprof[(li) * 64 + (id)] = gtime(); } while (0)
// ------------------------------------------------------------------------------------------------ block reductions
template <int N>
__device__ __forceinline__ void block_sum(float* v, float* red) {  // red: smem [NWARP][N]
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < N; i++) {
    float x = v[i];
#pragma unroll
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
    v[i] = x;
  }
  __syncthreads();
  if (lane == 0)
#pragma unroll
    for (int i = 0; i < N; i++) red[w * N + i] = v[i];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; i++) {
    float s = 0.f;
    for (int j = 0; j < NWARP; j++) s += red[j * N + i];
    v[i] = s;
  }
  __syncthreads();
}
__device__ __forceinline__ float block_max1(float x, float* red) {
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  for (int o = 16; o; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o));
  __syncthreads();
  if (lane == 0) red[w] = x;
  __syncthreads();
  float m = red[0];
  for (int j = 1; j < NWARP; j++) m = fmaxf(m, red[j]);
  __syncthreads();
  return m;
}

#define PER ((H + NTH - 1) / NTH)   // 5 elements per thread (the last one only for tid < 512)
#define VALID(e) (threadIdx.x + (e) * NTH < H)
// element e of thread: idx = threadIdx.x + e * NTH

// rms_norm of fp32 vector (per-thread regs) with bf16 weight -> bf16 values (as float)
__device__ __forceinline__ void rmsnorm_regs(const float* u, const bf16* w, float* y, float* red) {
  float s[1] = {0.f};
#pragma unroll
  for (int e = 0; e < PER; e++) if (VALID(e)) s[0] += u[e] * u[e];
  block_sum<1>(s, red);
  float r = rsqrtf(s[0] / (float)H + 1e-6f);
#pragma unroll
  for (int e = 0; e < PER; e++) {
    if (!VALID(e)) { y[e] = 0.f; continue; }
    float v = u[e] * r;
    if (w) v = v * bf(w[threadIdx.x + e * NTH]);
    y[e] = rbf(v);
  }
}

__device__ __forceinline__ float sconv4f(float h0, float h1, float h2, float cur, const float* w4) {
  float acc = 0.f;
  acc = acc + h0 * w4[0];
  acc = acc + h1 * w4[1];
  acc = acc + h2 * w4[2];
  acc = acc + cur * w4[3];
  return rbf((acc + cur) * 0.4472135954999579f);
}
// static short conv over [hist0, hist1, hist2, cur]; weight [C][4]; returns bf16-rounded float
__device__ __forceinline__ float sconv4(float h0, float h1, float h2, float cur, const bf16* w4) {
  float acc = 0.f;
  acc = acc + h0 * bf(w4[0]);
  acc = acc + h1 * bf(w4[1]);
  acc = acc + h2 * bf(w4[2]);
  acc = acc + cur * bf(w4[3]);
  return rbf((acc + cur) * 0.4472135954999579f);
}

// ------------------------------------------------------------------------------------------------ GEMV skeleton
// Flattened (strip, unit) space split evenly over all warps of the grid. A strip covered by several warps is
// summed in slot order by the last warp to arrive (per-strip counter). Epilogue gets per-lane d0..d3
// (row g, col 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1).
__device__ __forceinline__ long wstart(long T, long w, long NW) { return T * w / NW; }
__device__ __forceinline__ long warp_of(long x, long T, long NW) { return ((x + 1) * NW - 1) / T; }

#ifndef MINU
#define MINU 4
#endif
template <class OP>
__device__ void gemv(const Ctx& c, OP& op, long strips, int U) {
  const long T = strips * U;
  const long NWALL = (long)gridDim.x * NWARP;
  const long NW = max(1L, min(NWALL, T / MINU));
  // logical warp index. Ops that use every warp: block-contiguous ranges (a block streams one contiguous region;
  // the strided mapping put a block's 32 streams 729 KB apart in lockstep and some blocks ran 2x slower).
  // Small ops: consecutive logical warps on different SMs.
#ifndef GW_SPREAD_ALL
  const long gw = (NW == NWALL) ? (long)blockIdx.x * NWARP + (threadIdx.x >> 5) : (long)(threadIdx.x >> 5) * gridDim.x + blockIdx.x;
#else
  const long gw = (long)(threadIdx.x >> 5) * gridDim.x + blockIdx.x;
#endif
  if (gw >= NW) return;
  const int lane = threadIdx.x & 31;
  long beg = wstart(T, gw, NW), end = wstart(T, gw + 1, NW);
  long x = beg;
  const bool tmw = c.tprof && op.tm && gw == 0;   // block 0 warp 0 stamps its first strip
  if (tmw && lane == 0) c.tprof[op.layer * 64 + op.tm] = gtime();
  while (x < end) {
    long s = x / U;
    int u0 = (int)(x - s * U);
    int u1 = (int)min((long)U, end - s * U);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    op.run(c, s, u0, u1, acc);
    if (tmw && lane == 0 && x == beg) c.tprof[op.layer * 64 + op.tm + 1] = gtime();
    long fw = warp_of(s * U, T, NW), lw = warp_of(s * U + U - 1, T, NW);
    if (fw == lw) {
      op.epi(c, s, acc);
    } else {
#ifndef HANDOFF_COUNTER
      // A strip split over several warps is finished by the warp that holds its FIRST piece: that piece is the last
      // one of that warp's range, while the other warps compute their part of the strip first, so their slots are
      // (almost always) there already. No counter, no atomics: the other warps store their tagged slot and go on.
      if (gw != fw) {
        // slot of this warp's piece: per lane {v0, tag, v1, tag}, {v2, tag, v3, tag}
        uint4* slot = reinterpret_cast<uint4*>(c.part) + ((s * MAXSLOT + (gw - fw)) * 32 + lane) * 2;
        st_l2(slot, make_uint4(__float_as_uint(acc[0]), op.tag, __float_as_uint(acc[1]), op.tag));
        st_l2(slot + 1, make_uint4(__float_as_uint(acc[2]), op.tag, __float_as_uint(acc[3]), op.tag));
      } else {
        float a[4] = {acc[0], acc[1], acc[2], acc[3]};   // sum in slot order: this warp's piece is slot 0
        const int ns = (int)(lw - fw) + 1;
        for (int k = 1; k < ns; k++) {
          const uint4* q = reinterpret_cast<const uint4*>(c.part) + ((s * MAXSLOT + k) * 32 + lane) * 2;
          uint4 p0 = ld_gpu_v4(q), p1 = ld_gpu_v4(q + 1);
          while (p0.y != op.tag || p0.w != op.tag || p1.y != op.tag || p1.w != op.tag) { p0 = ld_gpu_v4(q); p1 = ld_gpu_v4(q + 1); }
          a[0] += __uint_as_float(p0.x); a[1] += __uint_as_float(p0.z); a[2] += __uint_as_float(p1.x); a[3] += __uint_as_float(p1.z);
        }
        op.epi(c, s, a);
      }
    }
#else
      // slot of this warp's piece: per lane {v0, tag, v1, tag}, {v2, tag, v3, tag}
      uint4* slot = reinterpret_cast<uint4*>(c.part) + ((s * MAXSLOT + (gw - fw)) * 32 + lane) * 2;
      st_l2(slot, make_uint4(__float_as_uint(acc[0]), op.tag, __float_as_uint(acc[1]), op.tag));
      st_l2(slot + 1, make_uint4(__float_as_uint(acc[2]), op.tag, __float_as_uint(acc[3]), op.tag));
      __syncwarp();
      unsigned old = 0;
      if (lane == 0) old = atomicAdd(&c.scnt[s], 1u);
      old = __shfl_sync(0xffffffff, old, 0);
      if (old == (unsigned)(lw - fw)) {   // last to arrive: sum the pieces in slot order
        float a[4] = {0.f, 0.f, 0.f, 0.f};
        const int ns = (int)(lw - fw) + 1;
        for (int k = 0; k < ns; k++) {
          if (k == (int)(gw - fw)) {   // this warp's own piece: from the registers (two relaxed loads less)
            a[0] += acc[0]; a[1] += acc[1]; a[2] += acc[2]; a[3] += acc[3];
            continue;
          }
          const uint4* q = reinterpret_cast<const uint4*>(c.part) + ((s * MAXSLOT + k) * 32 + lane) * 2;
          uint4 p0 = ld_gpu_v4(q), p1 = ld_gpu_v4(q + 1);
          while (p0.y != op.tag || p0.w != op.tag || p1.y != op.tag || p1.w != op.tag) { p0 = ld_gpu_v4(q); p1 = ld_gpu_v4(q + 1); }
          a[0] += __uint_as_float(p0.x); a[1] += __uint_as_float(p0.z); a[2] += __uint_as_float(p1.x); a[3] += __uint_as_float(p1.z);
        }
        if (lane == 0) atomicExch(&c.scnt[s], 0u);
        op.epi(c, s, a);
      }
    }
#endif
    if (tmw && lane == 0 && x == beg) c.tprof[op.layer * 64 + op.tm + 2] = gtime();
    x = s * U + u1;
  }
}

// Alternative schedule: block b owns strips b, b+nblk, ...; its 32 warps split the strip's units; partials reduced in
// shared memory in warp order (deterministic, no global hand-off). red: 32 warps x 32 lanes x float4 (16 KB).
template <class OP>
__device__ void gemv_blk(const Ctx& c, OP& op, int strips, int U, float* red, int npar = 2, int b0 = 0, int nb = 0) {
  // (b0, nb): the blocks b0 .. b0 + nb - 1 share the strips (default: the whole grid)
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (nb == 0) nb = gridDim.x;
  const int bi = (int)blockIdx.x - b0;
  if (bi < 0 || bi >= nb) return;
  const int ns = (strips - bi + nb - 1) / nb;   // strips of this block
  // two strips at a time, half of the warps each: one synchronisation, two reducer warps in parallel (the strips one
  // after the other cost a second barrier + reduction + epilogue round; B=8 fixed tokens: w2 22.6 -> 19.4 us,
  // qkv 10.0 -> 9.5, o_proj 3.7 -> 3.4; e0042)
  // (npar strips at a time: 4 for the w13 GEMV of one token, whose 576 strips are 3-4 per block; the flat schedule
  //  splits each of those strips over ~6 warps of different blocks and pays the tagged hand-off for every piece)
  for (int r = 0; r < ns; r += npar) {
    const int n2 = min(npar, ns - r), wg = NWARP / n2;
    const int q = w / wg, wl = w % wg;
    const int s = bi + (r + q) * nb;
    const int u0 = U * wl / wg, u1 = U * (wl + 1) / wg;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (q < n2 && u1 > u0) op.run(c, s, u0, u1, acc);
    *reinterpret_cast<float4*>(red + (w * 32 + lane) * 4) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();
    if (q < n2 && wl == 0) {
      float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
      for (int k = w; k < w + wg; k++) {
        float4 p = *reinterpret_cast<const float4*>(red + (k * 32 + lane) * 4);
        a[0] += p.x; a[1] += p.y; a[2] += p.z; a[3] += p.w;
      }
      op.epi(c, s, a);
    }
    __syncthreads();
  }
}
#ifndef GEMV_BLK
#define GEMV_BLK 23   // qkv, o_proj, router, w2 by block-per-strip (measured best, e0053, e0062)
#endif
template <class OP>
__device__ __forceinline__ void gemv_sel(const Ctx& c, OP& op, long strips, int U, int bit, float* red) {
  if (GEMV_BLK & bit) gemv_blk(c, op, (int)strips, U, red);
  else gemv(c, op, strips, U);
}

// BF16 dense: weights [strip][unit][j][lane][16B]; B frags (bf16) [unit][lane][j][2 u32]
template <bool CPT>
__device__ __forceinline__ void run_bf16(const uint8_t* wstrip, const uint32_t* xf, int u0, int u1, float* acc, int B) {
  const int lane = threadIdx.x & 31, g = lane >> 2;
  const bool act = CPT ? (g < B) : true;   // 8-column layout: unused columns compute garbage nobody reads
  int u = u0;
  for (; u + 2 <= u1; u += 2) {
    uint4 w[8];
#pragma unroll
    for (int q = 0; q < 8; q++) w[q] = ldg_nc(wstrip + ((long)(u + (q >> 2)) * 4 + (q & 3)) * 512 + lane * 16);
    uint4 b[4];
#pragma unroll
    for (int q = 0; q < 4; q++) b[q] = act ? ld4(xf + ((u + (q >> 1)) * LSZ + LSL) * 8 + (q & 1) * 4) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const uint4& bb = b[q >> 1];
      uint32_t b0 = (q & 1) ? bb.z : bb.x, b1 = (q & 1) ? bb.w : bb.y;
      mma_bf16(acc, w[q].x, w[q].y, w[q].z, w[q].w, b0, b1);
    }
  }
  for (; u < u1; u++) {
    uint4 w[4];
#pragma unroll
    for (int q = 0; q < 4; q++) w[q] = ldg_nc(wstrip + ((long)u * 4 + q) * 512 + lane * 16);
    uint4 b[2];
#pragma unroll
    for (int q = 0; q < 2; q++) b[q] = act ? ld4(xf + (u * LSZ + LSL) * 8 + q * 4) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const uint4& bb = b[q >> 1];
      uint32_t b0 = (q & 1) ? bb.z : bb.x, b1 = (q & 1) ? bb.w : bb.y;
      mma_bf16(acc, w[q].x, w[q].y, w[q].z, w[q].w, b0, b1);
    }
  }
}

// 12-bit exact BF16 code (track B): decode into bf16x2 registers, patch escapes, MMA.
__device__ __forceinline__ uint32_t prmt0(uint32_t a, uint32_t sel) { uint32_t r; asm("prmt.b32 %0,%1,0,%2;" : "=r"(r) : "r"(a), "r"(sel)); return r; }
__device__ __forceinline__ uint32_t w12dec(uint32_t bytes, int half, uint32_t nib, int a, uint32_t E128) {
  uint32_t y = prmt0(bytes, half ? 0xB3A2u : 0x9180u);
  uint32_t t = (nib >> (4 * a)) & 0x000F000Fu;
  return (y & 0x807F807Fu) | (t * 128u + E128);
}
template <bool CPT>
__device__ __forceinline__ void run_w12(const W12M& M, long strip, const uint32_t* xf, int u0, int u1, float* acc, int B, int U) {
  const int lane = threadIdx.x & 31, g = lane >> 2;
  const bool act = CPT ? (g < B) : true;   // 8-column layout: unused columns compute garbage nobody reads
  uint32_t eg = M.base[strip * 16 + g], eg8 = M.base[strip * 16 + g + 8];
  const uint32_t E0 = (eg | (eg << 16)) << 7, E1 = (eg8 | (eg8 << 16)) << 7;
  const uint8_t* wbase = M.data + strip * (long)U * 1536;
  const int* hdr = M.hdr + strip * (long)U;
  for (int u = u0; u < u1; u++) {
    uint4 d0 = ldg_nc(wbase + (long)u * 1536 + lane * 16);
    uint4 d1 = ldg_nc(wbase + (long)u * 1536 + 512 + lane * 16);
    uint4 dn = ldg_nc(wbase + (long)u * 1536 + 1024 + lane * 16);
    int hd = __ldg(hdr + u);
    uint4 b0 = act ? ld4(xf + (u * LSZ + LSL) * 8) : make_uint4(0, 0, 0, 0);
    uint4 b1 = act ? ld4(xf + (u * LSZ + LSL) * 8 + 4) : make_uint4(0, 0, 0, 0);
    uint32_t by[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
    uint32_t nb[4] = {dn.x, dn.y, dn.z, dn.w};
    uint32_t a[16];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      a[4 * j + 0] = w12dec(by[2 * j], 0, nb[j], 0, E0);
      a[4 * j + 1] = w12dec(by[2 * j], 1, nb[j], 1, E1);
      a[4 * j + 2] = w12dec(by[2 * j + 1], 0, nb[j], 2, E0);
      a[4 * j + 3] = w12dec(by[2 * j + 1], 1, nb[j], 3, E1);
    }
    int cnt = hd & 0xFF;
    if (cnt) {
      const int* e = M.esc + (hd >> 8);
      for (int q = 0; q < cnt; q++) {
        int en = __ldg(e + q);
        if ((en >> 27 & 31) == lane) {
          int r = (en >> 23) & 15, hf = (en >> 22) & 1;
          uint32_t val = en & 0xFFFF;
#pragma unroll
          for (int k = 0; k < 16; k++)
            if (k == r) a[k] = hf ? ((a[k] & 0x0000FFFFu) | (val << 16)) : ((a[k] & 0xFFFF0000u) | val);
        }
      }
    }
    uint32_t bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
    for (int j = 0; j < 4; j++) mma_bf16(acc, a[4 * j], a[4 * j + 1], a[4 * j + 2], a[4 * j + 3], bb[2 * j], bb[2 * j + 1]);
  }
}

// NVFP4 unit (64 k): codes [lane][16B] (512 B) + scales [g][8B] (64 B). B frags f16 [unit][lane][j][2].
#define UB4 576
__device__ __forceinline__ void cvt4(uint32_t w, uint32_t* o) {
  asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;\n"
      " cvt.rn.f16x2.e2m1x2 %0, b0; cvt.rn.f16x2.e2m1x2 %1, b1; cvt.rn.f16x2.e2m1x2 %2, b2; cvt.rn.f16x2.e2m1x2 %3, b3; }"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(w));
}
__device__ __forceinline__ void scl4(uint32_t w, uint32_t* o) {  // bytes (g,g8) j0, (g,g8) j1 -> {g,g},{g8,g8},{g,g},{g8,g8}
  uint32_t lo, hi;
  asm("{ .reg .b16 l,h; mov.b32 {l,h}, %2; cvt.rn.f16x2.e4m3x2 %0, l; cvt.rn.f16x2.e4m3x2 %1, h; }" : "=r"(lo), "=r"(hi) : "r"(w));
  asm("{ .reg .b16 a,b; mov.b32 {a,b}, %4; mov.b32 %0, {a,a}; mov.b32 %1, {b,b}; mov.b32 {a,b}, %5; mov.b32 %2, {a,a}; mov.b32 %3, {b,b}; }"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(lo), "r"(hi));
}
__device__ __forceinline__ void fp4_unit(const uint4& cw, const uint2& sw, const uint4& b01, const uint4& b23, float* acc) {
  uint32_t cc[4] = {cw.x, cw.y, cw.z, cw.w};
  uint32_t ss[2] = {sw.x, sw.y};
  uint32_t bb[8] = {b01.x, b01.y, b01.z, b01.w, b23.x, b23.y, b23.z, b23.w};
#pragma unroll
  for (int jj = 0; jj < 2; jj++) {
    uint32_t sc[4];
    scl4(ss[jj], sc);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      int j = jj * 2 + h;
      uint32_t a[4];
      cvt4(cc[j], a);
      uint32_t sg = sc[2 * h], sg8 = sc[2 * h + 1];
      mma_f16(acc, hmul2(a[0], sg), hmul2(a[1], sg8), hmul2(a[2], sg), hmul2(a[3], sg8), bb[2 * j], bb[2 * j + 1]);
    }
  }
}
// MXFP8 unit (64 k = 2 k32 steps): codes [s][lane][16B] (1024 B) + scales [g][s][hi] (32 B).
// B frags e4m3 [unit][lane][s][2 u32] (16 B per lane); act block scales fp32 [kblock32][MAXB]
#define UB8 1056
__device__ __forceinline__ void fp8_unit(const uint4& c0, const uint4& c1, uint32_t sw, const uint4& b, const float* xs, int u, float* acc) {
  const int lane = threadIdx.x & 31, t = lane & 3;
#pragma unroll
  for (int s = 0; s < 2; s++) {
    const uint4& cc = s ? c1 : c0;
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_e4m3(d, cc.x, cc.y, cc.z, cc.w, s ? b.z : b.x, s ? b.w : b.y);
    float fg = ue8m0((sw >> (16 * s)) & 0xff), fg8 = ue8m0((sw >> (16 * s + 8)) & 0xff);
    float2 fa = *reinterpret_cast<const float2*>(xs + (u * 2 + s) * MAXB + 2 * t);
    acc[0] = __fmaf_rn(d[0], __fmul_rn(fg, fa.x), acc[0]);
    acc[1] = __fmaf_rn(d[1], __fmul_rn(fg, fa.y), acc[1]);
    acc[2] = __fmaf_rn(d[2], __fmul_rn(fg8, fa.x), acc[2]);
    acc[3] = __fmaf_rn(d[3], __fmul_rn(fg8, fa.y), acc[3]);
  }
}
template <bool CPT>
__device__ __forceinline__ void run_q(const uint8_t* wstrip, const uint8_t* xf, const float* xs, int u0, int u1, float* acc, int B, int mx,
                                      uint8_t* ring = nullptr) {
  const int lane = threadIdx.x & 31, g = lane >> 2;
  const bool act = CPT ? (g < B) : true;   // 8-column layout: unused columns compute garbage nobody reads
#ifdef FP4_CPA
  // Software pipeline through shared memory: the next two units are copied global -> shared asynchronously
  // (cp.async, no registers) while the current two are computed. `ring`: this warp's 4 unit slots (2304 B) in the
  // attention's K/V staging area, which is free during the GEMV phases.
  if (!mx && ring && !CPT) {
    int u = u0, cur = 0, ncur = min(2, u1 - u0);
    const unsigned rs = (unsigned)__cvta_generic_to_shared(ring);
#define CPA_ISSUE(slot2, uu, n)                                                                                   \
    for (int q_ = 0; q_ < (n); q_++) {                                                                            \
      const unsigned d_ = rs + ((slot2) * 2 + q_) * UB4;                                                          \
      const uint8_t* s_ = wstrip + (long)((uu) + q_) * UB4;                                                       \
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d_ + lane * 16), "l"(s_ + lane * 16) : "memory");   \
      if (lane < 4) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d_ + 512 + lane * 16), "l"(s_ + 512 + lane * 16) : "memory"); \
    }                                                                                                             \
    asm volatile("cp.async.commit_group;" ::: "memory");
    if (ncur > 0) { CPA_ISSUE(0, u, ncur) }
    while (ncur > 0) {
      const int nn = min(2, u1 - (u + ncur));
      if (nn > 0) { CPA_ISSUE(cur ^ 1, u + ncur, nn) asm volatile("cp.async.wait_group 1;" ::: "memory"); }
      else asm volatile("cp.async.wait_group 0;" ::: "memory");
      for (int q = 0; q < ncur; q++) {
        const uint8_t* slot = ring + (cur * 2 + q) * UB4;
        const uint4 cw = ld4(slot + lane * 16);
        const uint2 sw = *reinterpret_cast<const uint2*>(slot + 512 + g * 8);
        const uint4 b0 = ld4(xf + ((u + q) * LSZ + LSL) * 32), b1 = ld4(xf + ((u + q) * LSZ + LSL) * 32 + 16);
        fp4_unit(cw, sw, b0, b1, acc);
      }
      u += ncur; ncur = nn; cur ^= 1;
    }
#undef CPA_ISSUE
    return;
  }
#endif
  if (!mx) {
    int u = u0;
#ifndef FP4_UNROLL
#define FP4_UNROLL 5   // B=8 fixed-token step: 4: 5161 us, 5: 5072, 6: 5109 (with FP8_UNROLL 4; B=1 +19 us; e0085)
#endif
#ifdef FP4_PIPE
    // two register sets of 2 units: the next pair's loads are in flight while this pair is computed
    if (u + 2 <= u1) {
      uint4 ca0, ca1, cb0, cb1; uint2 sa0, sa1, sb0, sb1;
#define LDU(cw_, sw_, uu) { cw_ = ldg_nc(wstrip + (long)(uu) * UB4 + lane * 16); sw_ = ldg_nc2(wstrip + (long)(uu) * UB4 + 512 + g * 8); }
#define CPU(cw_, sw_, uu) { uint4 b0 = ld4(xf + ((uu) * LSZ + LSL) * 32), b1 = ld4(xf + ((uu) * LSZ + LSL) * 32 + 16); fp4_unit(cw_, sw_, b0, b1, acc); }
      LDU(ca0, sa0, u) LDU(ca1, sa1, u + 1)
      for (;;) {
        const bool m1 = u + 4 <= u1;
        if (m1) { LDU(cb0, sb0, u + 2) LDU(cb1, sb1, u + 3) }
        CPU(ca0, sa0, u) CPU(ca1, sa1, u + 1)
        u += 2;
        if (!m1) break;
        const bool m2 = u + 4 <= u1;
        if (m2) { LDU(ca0, sa0, u + 2) LDU(ca1, sa1, u + 3) }
        CPU(cb0, sb0, u) CPU(cb1, sb1, u + 1)
        u += 2;
        if (!m2) break;
      }
#undef LDU
#undef CPU
    }
#else
    for (; u + FP4_UNROLL <= u1; u += FP4_UNROLL) {
      uint4 cw[FP4_UNROLL]; uint2 sw[FP4_UNROLL];
#pragma unroll
      for (int q = 0; q < FP4_UNROLL; q++) {
        cw[q] = ldg_nc(wstrip + (long)(u + q) * UB4 + lane * 16);
        sw[q] = ldg_nc2(wstrip + (long)(u + q) * UB4 + 512 + g * 8);
      }
#pragma unroll
      for (int q = 0; q < FP4_UNROLL; q++) {
        uint4 b0 = act ? ld4(xf + ((u + q) * LSZ + LSL) * 32) : make_uint4(0, 0, 0, 0);
        uint4 b1 = act ? ld4(xf + ((u + q) * LSZ + LSL) * 32 + 16) : make_uint4(0, 0, 0, 0);
        fp4_unit(cw[q], sw[q], b0, b1, acc);
      }
    }
#if FP4_UNROLL > 2
    for (; u + 2 <= u1; u += 2) {
      uint4 cw[2]; uint2 sw[2];
#pragma unroll
      for (int q = 0; q < 2; q++) {
        cw[q] = ldg_nc(wstrip + (long)(u + q) * UB4 + lane * 16);
        sw[q] = ldg_nc2(wstrip + (long)(u + q) * UB4 + 512 + g * 8);
      }
#pragma unroll
      for (int q = 0; q < 2; q++) {
        uint4 b0 = act ? ld4(xf + ((u + q) * LSZ + LSL) * 32) : make_uint4(0, 0, 0, 0);
        uint4 b1 = act ? ld4(xf + ((u + q) * LSZ + LSL) * 32 + 16) : make_uint4(0, 0, 0, 0);
        fp4_unit(cw[q], sw[q], b0, b1, acc);
      }
    }
#endif
#endif
    for (; u < u1; u++) {
      uint4 cw = ldg_nc(wstrip + (long)u * UB4 + lane * 16);
      uint2 sw = ldg_nc2(wstrip + (long)u * UB4 + 512 + g * 8);
      uint4 b0 = act ? ld4(xf + (u * LSZ + LSL) * 32) : make_uint4(0, 0, 0, 0);
      uint4 b1 = act ? ld4(xf + (u * LSZ + LSL) * 32 + 16) : make_uint4(0, 0, 0, 0);
      fp4_unit(cw, sw, b0, b1, acc);
    }
  } else {
    int u = u0;
#ifndef FP8_UNROLL
#define FP8_UNROLL 4
#endif
#if FP8_UNROLL > 1
    for (; u + FP8_UNROLL <= u1; u += FP8_UNROLL) {
      uint4 c0[FP8_UNROLL], c1[FP8_UNROLL]; uint32_t sw[FP8_UNROLL];
#pragma unroll
      for (int q = 0; q < FP8_UNROLL; q++) {
        c0[q] = ldg_nc(wstrip + (long)(u + q) * UB8 + lane * 16);
        c1[q] = ldg_nc(wstrip + (long)(u + q) * UB8 + 512 + lane * 16);
        sw[q] = ldg_nc1(wstrip + (long)(u + q) * UB8 + 1024 + g * 4);
      }
#pragma unroll
      for (int q = 0; q < FP8_UNROLL; q++) {
        uint4 b = act ? ld4(xf + ((u + q) * LSZ + LSL) * 16) : make_uint4(0, 0, 0, 0);
        fp8_unit(c0[q], c1[q], sw[q], b, xs, u + q, acc);
      }
    }
#endif
    for (; u < u1; u++) {
      uint4 c0 = ldg_nc(wstrip + (long)u * UB8 + lane * 16);
      uint4 c1 = ldg_nc(wstrip + (long)u * UB8 + 512 + lane * 16);
      uint32_t sw = ldg_nc1(wstrip + (long)u * UB8 + 1024 + g * 4);
      uint4 b = act ? ld4(xf + (u * LSZ + LSL) * 16) : make_uint4(0, 0, 0, 0);
      fp8_unit(c0, c1, sw, b, xs, u, acc);
    }
  }
}

// ------------------------------------------------------------------------------------------------ tcgen05 primitives
// The NVFP4 expert GEMVs through the 5th-generation tensor core (tools/bench/tc5.cu): a block-scaled MMA
// (kind::mxf4nvf4, scale_vec::4X, UE4M3 scales) takes 128 rows x 64 k of E2M1 codes and their block scales straight
// from shared memory, where cp.async.bulk puts them: no per-byte instructions (the m16n8k16 formulation needs 44-46
// instructions per 576 B and is bound by the SM's instruction rate at ~4.3 TB/s per card; this path streams 6.1-6.5).
#ifndef NO_TC13
#define TC13 1
#else
#define TC13 0
#endif
#ifdef TC2_ON     // w2 on the same path: written, not yet verified
#define TC2 1
#else
#define TC2 0
#endif
#define TC_NC 4          // streams per block: producer thread (warp 4 + 2c), consumer thread (warp 5 + 2c)
#define TC_S 2           // ring slots per stream
#define TC_G 2           // units per ring slot
#define TC_UB 4608       // unit: 128 rows x 64 k: 4096 B of codes (K-major, no swizzle) + 512 B scale-factor atom
#define TC_MAXSUB 40
#ifndef TC_MINB
#define TC_MINB 4        // batches below run the m16n8k16 path on the fragment layout (fixed costs: B=1 +4% with tcgen05, e0162)
#endif
#ifndef TC_COLS
#define TC_COLS 512
#endif
// progress of block 0 on every card, in host-mapped memory: where a launch that does not end is waiting
#ifndef TC_NOPROG
#define TC_PROG(k, v) do { if (c.prog && blockIdx.x == 0 && threadIdx.x == 0) c.prog[c.h.card * 16 + (k)] = (int)(v); } while (0)
#else
#define TC_PROG(k, v)
#endif
#define TC_IDESC4 ((1u << 7) | (1u << 10) | (1u << 17) | (1u << 27))   // E2M1 x E2M1, N = 8, UE4M3 scales, M = 128, K = 64
struct TcSub { const uint8_t* src; unsigned short n, ks0; int item; int task; int pad; };
__device__ __forceinline__ uint32_t sh(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ bool mbar_try(uint32_t a, uint32_t par) {
  uint32_t ok; asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(a), "r"(par) : "memory"); return ok; }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t par) { while (!mbar_try(a, par)) {} }
__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void tc_bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory"); }
// weights: first in the L2's eviction order (TC_L2 0: default policy, 1: evict_first)
#ifndef TC_L2
#define TC_L2 1
#endif
__device__ __forceinline__ uint64_t tc_policy() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void tc_bulk_w(uint32_t dst, const void* src, uint32_t n, uint32_t bar, uint64_t pol) {
#if TC_L2
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(n), "r"(bar), "l"(pol) : "memory");
#else
  tc_bulk(dst, src, n, bar);
#endif
}
__device__ __forceinline__ void tc_alloc(uint32_t smem_addr, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_addr), "r"(ncols) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_dealloc(uint32_t taddr, uint32_t ncols) { asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory"); }
__device__ __forceinline__ void tc_cp(uint32_t taddr, uint64_t desc) { asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(desc) : "memory"); }
__device__ __forceinline__ void tc_mma4(uint32_t td, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc, uint32_t tsfa, uint32_t tsfb) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; }"
               :: "r"(td), "l"(da), "l"(db), "r"(idesc), "r"(acc), "r"(tsfa), "r"(tsfb) : "memory"); }
__device__ __forceinline__ void tc_commit(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_ld8(uint32_t taddr, float* d) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr) : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int j = 0; j < 8; j++) d[j] = __uint_as_float(r[j]);
}
__device__ __forceinline__ void tc_st1(uint32_t taddr, uint32_t v) { asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(taddr), "r"(v) : "memory"); }
// eight K steps' scale words at a pitch of 4 columns in one store (the other columns get zeros)
__device__ __forceinline__ void tc_st32(uint32_t taddr, const uint4& a, const uint4& b) {
  const uint32_t z = 0;
  asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%9,%9,%9, %2,%9,%9,%9, %3,%9,%9,%9, %4,%9,%9,%9, %5,%9,%9,%9, %6,%9,%9,%9, %7,%9,%9,%9, %8,%9,%9,%9};"
               :: "r"(taddr), "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w), "r"(z) : "memory");
}
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ uint64_t tc_sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {   // shared memory descriptor: K-major, no swizzle
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) | (1ull << 46);
}

// ------------------------------------------------------------------------------------------------ ops
struct Smem {
  float red[NWARP * 40 + 64];
  float bestv[NWARP * MAXB];
  int besti[NWARP * MAXB];
  float att[6 * CHUNK];
  float qv[6 * HD];
  float relb[6 * CHUNK];
  float misc[128];
  float opart[SM_OPART];    // GEMV block partials (NWARP x 32 x 4), routing scores (MAXB x 384), attention V halves
  int ids[MAXB * TOPK];
  float wts[MAXB * TOPK];
  int uni[MAXU];
  int nuni;
  float minv[10];
  float mixp[30];
  float tt[40];
  float tdot[40];
  int uix[MAXB * TOPK];
  int upre[12];
  float wtu[MAXU * MAXB];   // routing weight of (union expert, token), 0 if not routed
  unsigned rmask[MAXU];
  float fiv[10];
  float piv;
  // per-token state of token blockIdx.x (< B) for the whole step: finished blocks 0..7, partial (8), attention out
  // (team layout: a block holds 768 elements of its token. Shared memory is taken from the SM's L1: with 222 KB of
  //  shared memory the FP4 GEMV microbenchmark ran 15% slower than with 150 KB, e0072)
  __align__(16) bf16 blk[9][SM_TOK];
  __align__(16) bf16 ats[SM_TOK];
  // tensor-core GEMV: B operand (w13: 72 K steps of 256 B; w2: 8 slots of 9 K steps), mbarriers (full[8], empty[8],
  // done[8], ready[8], B copy), the block's sub-items and their column scales, the tensor memory base
  __align__(16) uint8_t tcB[18432];
  __align__(8) unsigned long long tcbar[34];
  unsigned tcph[34];                     // phases every mbarrier has completed before the current GEMV (they are initialised once per launch)
  TcSub tctab[TC_MAXSUB];
  float tcs[TC_MAXSUB * 8];
  unsigned tct;
#ifndef SM_STAGE
#define SM_STAGE 73728    // the attention's K/V rows of one chunk (2 x 128 rows x 272 B); GEMV fragments are read through L1
#endif
  union {
    struct {  // members of the old B == 1 redundant-block kernel (not launched any more): stubs so that it compiles
      __align__(16) bf16 fx[3][8];
      __align__(16) bf16 fx2[8];
      __align__(16) uint8_t f13[16];
      __align__(16) uint8_t f2[16];
      __align__(16) bf16 fh[8];
      float sc13[8];
      float sc2[8];
    };
    __align__(16) uint8_t stage[SM_STAGE];
  };
};
// std mode: copy a GEMV's B fragments (global, L2-resident) into shared memory once per block
__device__ __forceinline__ void stage_frags(Smem& sm, const void* src, int bytes, const void* src2 = nullptr, int bytes2 = 0) {
  const uint4* s4 = reinterpret_cast<const uint4*>(src);
  uint4* d4 = reinterpret_cast<uint4*>(sm.stage);
  for (int k = threadIdx.x; k < bytes / 16; k += NTH) d4[k] = __ldcg(s4 + k);
  if (src2) {
    const uint4* t4 = reinterpret_cast<const uint4*>(src2);
    for (int k = threadIdx.x; k < bytes2 / 16; k += NTH) d4[bytes / 16 + k] = __ldcg(t4 + k);
  }
  __syncthreads();
}

template <bool CPT>
struct OpQKV {
  int layer; const bf16* xf0; const bf16* xs0; int tm = 0;   // xs0: staged copy of group 0 (std mode) or null   // three frags (xq, xk, xv), each 72 units
  unsigned tag = 0;   // tag of the strip hand-off slots (flat GEMV)
  int smap = 0;       // 0: all strips; 1: all but the 96 gate strips (rows 1536..3071); 2: the gate strips only
  __device__ __forceinline__ long ms(long s) const { return smap == 1 ? (s < 96 ? s : s + 96) : (smap == 2 ? s + 96 : s); }
  __device__ void run(const Ctx& c, long s, int u0, int u1, float* acc) {
    const LayerW& L = c.L[layer];
    s = ms(s);
    int grp = 0;
    if (L.glob) grp = s < 205 ? 0 : (s < 222 ? 1 : 2);
    const uint32_t* xf = reinterpret_cast<const uint32_t*>((grp == 0 && xs0) ? xs0 : xf0 + grp * (72 * LSZ * 16));
#ifdef W12
    run_w12<CPT>(L.qkv12, s, xf, u0, u1, acc, c.h.B, 72);
#else
    run_bf16<CPT>(L.wqkv + s * 72L * 2048, xf, u0, u1, acc, c.h.B);
#endif
  }
  __device__ void epi(const Ctx& c, long s, const float* a) {
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    s = ms(s);
    int r0 = (int)s * 16 + g;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int row = r0 + (i >> 1) * 8, col = 2 * t + (i & 1);
      if (col >= c.h.B) continue;
      if (smap == 2) put_l2(&c.gll[col * 1536 + (row - 1536)], sigm(rbf(a[i])), c.h.tag0 + layer + 1);   // read tag-checked by the combining block
      else c.qkvo[col * 3840 + row] = tobf(a[i]);
    }
  }
};

template <bool CPT>
struct OpOproj {
  int layer; const bf16* xf; int tm = 25;
  unsigned tag = 0;   // tag of the strip hand-off slots (flat GEMV)
  __device__ void run(const Ctx& c, long s, int u0, int u1, float* acc) {
#ifdef W12
    run_w12<CPT>(c.L[layer].wo12, s, reinterpret_cast<const uint32_t*>(xf), u0, u1, acc, c.h.B, 24);
    return;
#endif
    run_bf16<CPT>(c.L[layer].wo + s * 24L * 2048, reinterpret_cast<const uint32_t*>(xf), u0, u1, acc, c.h.B);
  }
  __device__ void epi(const Ctx& c, long s, const float* a) {
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int row = (int)s * 16 + g + (i >> 1) * 8, col = 2 * t + (i & 1);
      if (col < c.h.B) for (int d = 0; d < NCARD; d++) ll_put(&c.comm[d]->lla[layer & 1][c.h.card][col][row], a[i], c.h.tag0 + layer + 1);
    }
  }
};

template <bool CPT>
struct OpRouter {
  int layer; const bf16* xf; float* rt; int tm = 0;
  unsigned tag = 0;   // tag of the strip hand-off slots (flat GEMV)
  __device__ void run(const Ctx& c, long s, int u0, int u1, float* acc) {
#ifdef W12
    run_w12<CPT>(c.L[layer].wr12, s, reinterpret_cast<const uint32_t*>(xf), u0, u1, acc, c.h.B, 72);
#else
    run_bf16<CPT>(c.L[layer].wr + s * 72L * 2048, reinterpret_cast<const uint32_t*>(xf), u0, u1, acc, c.h.B);
#endif
  }
  __device__ void epi(const Ctx& c, long s, const float* a) {
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int row = (int)s * 16 + g + (i >> 1) * 8, col = 2 * t + (i & 1);
      (void)row; (void)col; rt[lane * 4 + i] = a[i];   // sigmoid + publish by four warps after the GEMV (step_body)
    }
  }
};

__device__ __forceinline__ float act_fn(float g, float u) {
  float a = (4.0f * tanhf(g / 4.0f)) * sigm(g);
  return a * (25.0f * tanhf(u / 25.0f));
}

template <bool CPT>
struct OpW13 {
  int layer; int nu; const Smem* sm; const uint8_t* xf; const float* xs; const float* outer; int tm = 40;
  unsigned tag = 0;   // tag of the strip hand-off slots (flat GEMV)
  __device__ void run(const Ctx& c, long s, int u0, int u1, float* acc) {
    const LayerW& L = c.L[layer];
    int ui = (int)(s / 72), ls = (int)(s % 72);
    int e = sm->uni[ui];
    long ub = L.mx ? UB8 : UB4;
    const uint8_t* w = L.w13 + ((long)e * 72 + ls) * 72 * ub;
    run_q<CPT>(w, xf, xs, u0, u1, acc, c.h.B, L.mx, const_cast<uint8_t*>(sm->stage) + (threadIdx.x >> 5) * 2304);
  }
  __device__ void epi(const Ctx& c, long s, const float* a) {
    const LayerW& L = c.L[layer];
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int ui = (int)(s / 72), ls = (int)(s % 72);
    int e = sm->uni[ui];
    // rows g (gate if g even... ) : local row r = ls*16 + g (+8). interleaved: even row gate, odd row up.
    // lane holds rows g and g+8 for cols 2t, 2t+1. Pair (gate,up) = rows (2p, 2p+1) -> lanes g=2p and g=2p+1.
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int col = 2 * t + (i & 1);
      float sc = L.mx ? 1.0f : outer[col] * L.f13[e];
      v[i] = rbf(a[i] * sc);
    }
    // partner lane (g^1) holds the other member of the pair
    float o[4];
#pragma unroll
    for (int i = 0; i < 4; i++) o[i] = __shfl_xor_sync(0xffffffff, v[i], 4);
    {
      // interleaved rows: even = gate, odd = up. The gate lane (g even) does the pair of rows (g, g+1), the up lane
      // (g odd) the pair (g-1+8, g+8): two activation values per lane instead of four on half the lanes.
      // Each value of a routed (expert, token) pair goes out as a (value, tag) word: the w2 input quantization
      // reads them tag-checked, so this epilogue needs no fence, no counter and no atomic maximum (those cost
      // ~4 us per layer at B=8; without the fences the old count-then-read protocol was not reproducible).
      const bool gl = (g & 1) == 0;
      const int j = (ls * 16 + (gl ? g : g + 7)) >> 1;   // local intermediate index
      // The epilogue only hands over the (gate, up) pre-activations: the activation function (two tanh, a sigmoid)
      // is evaluated by the w2 quantization block, one value per thread, instead of two values per lane here
      // (~130 instructions each on the path of the warp that finishes the strip).
      uint4* actw = reinterpret_cast<uint4*>(c.act);
      const unsigned rm = sm->rmask[ui];
      const unsigned tg = c.h.tag0 + layer + 1;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int col = 2 * t + h;
        if (col >= c.h.B || !((rm >> col) & 1u)) continue;
        const float gate = gl ? v[h] : o[2 + h], up = gl ? o[h] : v[2 + h];
        st_l2(&actw[(ui * MAXB + col) * ILOC + j], make_uint4(__float_as_uint(gate), tg, __float_as_uint(up), tg));
      }
    }
  }
};

template <bool CPT>
struct OpW2 {
  int layer; int nu; const Smem* sm; const uint8_t* xf0; const float* sc0; int tm = 28;   // frags [ui][9 units], scales [ui][18][MAXB]
  unsigned tag = 0;   // tag of the strip hand-off slots (flat GEMV)
  __device__ void run(const Ctx& c, long s, int u0, int u1, float* acc) {
    // units: (ui, k-unit) with 9 units per expert
    const LayerW& L = c.L[layer];
    int lane = threadIdx.x & 31, t = lane & 3;
    long ub = L.mx ? UB8 : UB4;
    int u = u0;
#ifdef W2_PIECE_PF
    if (u1 > u0) {   // lane q: the q-th expert segment of this warp's range (one bulk L2 prefetch each)
      const int ui0 = u0 / 9, ui1 = (u1 - 1) / 9, ui = ui0 + lane;
      if (ui <= ui1) {
        const int k0 = (ui == ui0) ? u0 % 9 : 0, k1 = (ui == ui1) ? (u1 - 1) % 9 + 1 : 9;
        pf_l2(L.w2 + ((long)sm->uni[ui] * 288 + s) * 9 * ub + (long)k0 * ub, (unsigned)((k1 - k0) * ub));
      }
    }
#endif
    while (u < u1) {
      int ui = u / 9, k0 = u % 9, k1 = min(9, u1 - ui * 9);
      int e = sm->uni[ui];
      const uint8_t* w = L.w2 + ((long)e * 288 + s) * 9 * ub;
      float a[4] = {0.f, 0.f, 0.f, 0.f};
      const uint8_t* xf = xf0 + (long)ui * 9 * LSZ * (L.mx ? 16 : 32);
      const float* xs = sc0 + (long)ui * 18 * MAXB;
      run_q<CPT>(w, xf, xs, k0, k1, a, c.h.B, L.mx, const_cast<uint8_t*>(sm->stage) + (threadIdx.x >> 5) * 2304);
      if (L.mx) {
#pragma unroll
        for (int i = 0; i < 4; i++) acc[i] += a[i];
      } else {
        float f = L.f2[e];
        float s0 = sc0[ui * 18 * MAXB + 2 * t] * f, s1 = sc0[ui * 18 * MAXB + 2 * t + 1] * f;   // outer at [ui][0][col]
        // explicit FMAs: the compiler may contract the four columns' multiply-adds differently, which made a
        // sequence's result depend on its batch slot (1 ulp, s0004 http_tokens)
        acc[0] = __fmaf_rn(a[0], s0, acc[0]); acc[1] = __fmaf_rn(a[1], s1, acc[1]);
        acc[2] = __fmaf_rn(a[2], s0, acc[2]); acc[3] = __fmaf_rn(a[3], s1, acc[3]);
      }
      u = ui * 9 + k1;
    }
  }
  __device__ void epi(const Ctx& c, long s, const float* a) {
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int row = (int)s * 16 + g + (i >> 1) * 8, col = 2 * t + (i & 1);
      if (col < c.h.B) for (int d = 0; d < NCARD; d++) ll_put(&c.comm[d]->llm[layer & 1][c.h.card][col][row], a[i], c.h.tag0 + layer + 1);
    }
  }
};

template <bool CPT>
struct OpUnembed {
  float* bestv; int* besti; const bf16* xf; int tm = 0; int layer = 0;   // per-warp running best per col (smem)
  unsigned tag = 0;   // tag of the strip hand-off slots (flat GEMV)
  __device__ void run(const Ctx& c, long s, int u0, int u1, float* acc) {
    // The storage of the unembedding is a build choice (track word W12U -> -DW12U, engine/build.sh): with both
    // decoders compiled into this GEMV and chosen at load, the BF16 path was 8 us per step slower at B=1 (e0108).
#if defined(W12) || defined(W12U)
    run_w12<CPT>(c.un12, s, reinterpret_cast<const uint32_t*>(xf), u0, u1, acc, c.h.B, 72);
#else
    run_bf16<CPT>(c.unembed + s * 72L * 2048, reinterpret_cast<const uint32_t*>(xf), u0, u1, acc, c.h.B);
#endif
  }
  __device__ void epi(const Ctx& c, long s, const float* a) {
    int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, w = threadIdx.x >> 5;
    float bv[2] = {-INFINITY, -INFINITY};
    int bi[2] = {1 << 30, 1 << 30};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int row = c.un_row0 + (int)s * 16 + g + (i >> 1) * 8, col = 2 * t + (i & 1);
      if (col < c.h.B && row < VOCAB_REAL) {
        float z = 30.0f * tanhf(a[i] / 30.0f);
        if (c.h.want_logits) c.h.logits[(long)col * VOCAB_REAL + row] = z;
        int k = i & 1;
        if (z > bv[k] || (z == bv[k] && row < bi[k])) { bv[k] = z; bi[k] = row; }
      }
    }
#pragma unroll
    for (int k = 0; k < 2; k++) {
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv[k], o);
        int oi = __shfl_xor_sync(0xffffffff, bi[k], o);
        if (ov > bv[k] || (ov == bv[k] && oi < bi[k])) { bv[k] = ov; bi[k] = oi; }
      }
      int col = 2 * t + k;
      if (g == 0 && col < c.h.B) {
        float cv = bestv[w * MAXB + col];
        int ci = besti[w * MAXB + col];
        if (bv[k] > cv || (bv[k] == cv && bi[k] < ci)) { bestv[w * MAXB + col] = bv[k]; besti[w * MAXB + col] = bi[k]; }
      }
    }
  }
};

// ------------------------------------------------------------------------------------------------ phases


template <bool CPT> __device__ __forceinline__ void put16(bf16* f, int b, int k, float v) { f[fi16<CPT>(b, k)] = tobf(v); }

// ------------------------------------------------------------------------------------------------ per-token vector code
// thread t < 576 owns the 8 consecutive elements [8t, 8t+8) of a 4608-vector: one 16-byte load per bf16 vector.
#define NV8 (H / 8)
#define W8 (NV8 / 32)          // 18 active warps
#define ACT8 (threadIdx.x < NV8)
__device__ __forceinline__ void ld8(const bf16* p, float* v) {
  uint4 u = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(h[i]); v[2 * i] = f.x; v[2 * i + 1] = f.y; }
}
__device__ __forceinline__ void ld8g(const bf16* p, float* v) {   // global, L2 (data written in this launch)
  uint4 u = __ldcg(reinterpret_cast<const uint4*>(p));
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) { float2 f = __bfloat1622float2(h[i]); v[2 * i] = f.x; v[2 * i + 1] = f.y; }
}
__device__ __forceinline__ void st8(bf16* p, const float* v) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) h[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}
// block-wide sums of N values held by the 576 active threads; result broadcast through sm.red
template <int N>
__device__ __forceinline__ void bsum8(float* v, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (w < W8) {
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int o = 16; o; o >>= 1) v[i] += __shfl_xor_sync(0xffffffff, v[i], o);
      if (lane == 0) red[w * N + i] = v[i];
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; i++) {
    float t = 0.f;
#pragma unroll
    for (int j = 0; j < W8; j++) t += red[j * N + i];
    v[i] = t;
  }
  __syncthreads();
}
__device__ __forceinline__ float bmax8(float m, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (w < W8) {
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) red[w] = m;
  }
  __syncthreads();
  float r = red[0];
#pragma unroll
  for (int j = 1; j < W8; j++) r = fmaxf(r, red[j]);
  __syncthreads();
  return r;
}
// rms_norm (fp32 in) with a bf16 weight -> bf16-rounded values
// in place: u (fp32) -> rms_norm(u) * w rounded to bf16
__device__ __forceinline__ void rmsnorm8(float* u, const bf16* w, float* red) {
  float s[1] = {0.f};
  if (ACT8) {
#pragma unroll
    for (int i = 0; i < 8; i++) s[0] += u[i] * u[i];
  }
  bsum8<1>(s, red);
  if (!ACT8) return;
  const float r = rsqrtf(s[0] / (float)H + 1e-6f);
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(w + threadIdx.x * 8);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 f = __bfloat1622float2(w2[i]);
    u[2 * i] = rbf((u[2 * i] * r) * f.x);
    u[2 * i + 1] = rbf((u[2 * i + 1] * r) * f.y);
  }
}
// 8 bf16 values at k0..k0+7 of token b into a 16-bit fragment: pairs (k, k+1) are one 32-bit word
template <bool CPT>
__device__ __forceinline__ void put8(bf16* f, int b, int k0, const float* y) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int k = k0 + 2 * i;
    *reinterpret_cast<__nv_bfloat162*>(f + fi16<CPT>(b, k)) = __floats2bfloat162_rn(y[2 * i], y[2 * i + 1]);
  }
}
// NVFP4 input quantization of one token (four over six); a block of 16 = threads (2m, 2m+1). Writes f16 frags.
__device__ __forceinline__ float e2m1_rne(float mag) {
  float q;
  if (mag <= 0.25f) q = 0.f; else if (mag < 0.75f) q = 0.5f; else if (mag <= 1.25f) q = 1.0f; else if (mag < 1.75f) q = 1.5f;
  else if (mag <= 2.5f) q = 2.0f; else if (mag < 3.5f) q = 3.0f; else if (mag <= 5.0f) q = 4.0f; else q = 6.0f;
  return q;
}
// core: per-thread max m, the token's amax, element offset k0; `on`: this thread holds 8 valid elements
template <bool CPT>
__device__ __forceinline__ float quant_nvfp4_core(const float* x, int b, __half* frag, float m, float amax, int k0, bool on) {
  const float FMAX = 3.4028234663852886e38f;
  const float encoder = fminf(1536.0f / amax, FMAX), decoder = 1.0f / encoder, outer = amax * (1.0f / 1536.0f);
  if (on) {
    float bm = m;
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, 1));
    float sc[2], iv[2], er[2];
#pragma unroll
    for (int cand = 0; cand < 2; cand++) {
      float inv_max = cand == 0 ? (1.0f / 6.0f) : 0.25f;
      sc[cand] = float(__nv_fp8_e4m3(fminf(bm * (encoder * inv_max), 448.0f)));
      iv[cand] = fminf(1.0f / (sc[cand] * decoder), FMAX);
      float err = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float val = x[i] * iv[cand];
        float q = (val != val) ? 0.f : e2m1_rne(fabsf(val));
        err += fabsf(((signbit(val) ? -q : q) * sc[cand]) * outer - x[i]);
      }
      er[cand] = err + __shfl_xor_sync(0xffffffff, err, 1);
    }
    const int pick = (er[1] < er[0]) ? 1 : 0;   // a strictly better /4 wins
    const float scale = sc[pick], inverse = iv[pick];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float v2[2];
#pragma unroll
      for (int h = 0; h < 2; h++) {
        float val = x[2 * i + h] * inverse;
        float q = (val != val) ? 0.f : e2m1_rne(fabsf(val));
        v2[h] = (signbit(val) ? -q : q) * scale;
      }
      *reinterpret_cast<__half2*>(frag + fi16<CPT>(b, k0 + 2 * i)) = __floats2half2_rn(v2[0], v2[1]);
    }
  }
  return outer;
}
template <bool CPT>
__device__ float quant_nvfp4_8(const float* x, int b, __half* frag, float* red) {
  float m = 0.f;
  if (ACT8) {
#pragma unroll
    for (int i = 0; i < 8; i++) m = fmaxf(m, fabsf(x[i]));
  }
  const float amax = bmax8(m, red);
  return quant_nvfp4_core<CPT>(x, b, frag, m, amax, threadIdx.x * 8, ACT8);
}
// MXFP8 input quantization: a block of 32 = threads 4m..4m+3
template <bool CPT>
__device__ __forceinline__ void quant_mxfp8_8(const float* x, int b, uint8_t* frag, float* scales, int k0, bool on) {
  if (!on) return;
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) m = fmaxf(m, fabsf(x[i]));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
  float raw = m * (1.0f / 448.0f);
  unsigned bits = __float_as_uint(raw);
  int ex = (int)(((bits + 0x007FFFFFu) & 0x7F800000u) >> 23);
  if (raw <= 5.877471754111438e-39f) ex = 0;
  if (ex > 254) ex = 254;
  float inverse = (ex == 0) ? 1.7014118346046923e38f : 1.0f / ue8m0((unsigned)ex);
  uint32_t w[2] = {0, 0};
#pragma unroll
  for (int i = 0; i < 8; i++) {
    float v = fminf(fmaxf(x[i] * inverse, -448.f), 448.f);
    w[i >> 2] |= (uint32_t)__nv_fp8_e4m3(v).__x << (8 * (i & 3));
  }
  *reinterpret_cast<uint32_t*>(frag + fi8<CPT>(b, k0)) = w[0];
  *reinterpret_cast<uint32_t*>(frag + fi8<CPT>(b, k0 + 4)) = w[1];
  if ((threadIdx.x & 3) == 0) scales[(k0 >> 5) * MAXB + b] = ue8m0((unsigned)ex);
}
// all-reduce sum of the cards' partials (LL mailbox of one layer parity), bf16-rounded; waits for the tags
__device__ __forceinline__ void raw8(const uint2 (*mb)[MAXB][H], int b, unsigned tag, float* raw, int k0) {
#pragma unroll
  for (int h = 0; h < 4; h++) {
    float2 v0 = ll_get2(&mb[0][b][k0 + 2 * h], tag), v1 = ll_get2(&mb[1][b][k0 + 2 * h], tag);
    float2 v2 = ll_get2(&mb[2][b][k0 + 2 * h], tag), v3 = ll_get2(&mb[3][b][k0 + 2 * h], tag);
    raw[2 * h] = rbf(((v0.x + v1.x) + v2.x) + v3.x);
    raw[2 * h + 1] = rbf(((v0.y + v1.y) + v2.y) + v3.y);
  }
}
// short conv: hist [3][H] (oldest first), taps w4 [H][4]; writes the shifted history (writer only)
__device__ __forceinline__ void conv8(const bf16* __restrict__ hist, const float* raw, const bf16* w4,
                                      bf16* __restrict__ hw, bool writer, float* out, int k0) {
  float h0[8], h1[8], h2[8];
  ld8g(hist + k0, h0); ld8g(hist + H + k0, h1); ld8g(hist + 2 * H + k0, h2);
#pragma unroll
  for (int q = 0; q < 4; q++) {
    float wt[8];
    ld8(w4 + (k0 * 4) + q * 8, wt);
    out[2 * q] = sconv4f(h0[2 * q], h1[2 * q], h2[2 * q], raw[2 * q], wt);
    out[2 * q + 1] = sconv4f(h0[2 * q + 1], h1[2 * q + 1], h2[2 * q + 1], raw[2 * q + 1], wt + 4);
  }
  if (writer) { st8(hw + k0, h1); st8(hw + H + k0, h2); st8(hw + 2 * H + k0, raw); }
}
// residual mix: sources s < nb: finished blocks, s == nb: the partial (+ fp32 addend), s == nb+1: attention out
__device__ __forceinline__ const bf16* msrc(Smem& sm, int nb, int s) { return s < nb ? sm.blk[s] : (s == nb ? sm.blk[8] : sm.ats); }
template <int NQ>
__device__ __forceinline__ void mix_part8(Smem& sm, int nb, int s0, int s1, const float* addend, const float* q) {
  if (!ACT8) return;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, k0 = threadIdx.x * 8;
  const float qs = 0.014731391274719739f;  // 4608 ** -0.5 rounded to fp32
  constexpr int G = (NQ == 1) ? 4 : 2;     // sources per batch: G * (NQ + 1) = 8 independent shuffle chains
  float qv[NQ][8];
#pragma unroll
  for (int qi = 0; qi < NQ; qi++) {
    float4 a = *reinterpret_cast<const float4*>(q + qi * H + k0), b = *reinterpret_cast<const float4*>(q + qi * H + k0 + 4);
    qv[qi][0] = a.x * qs; qv[qi][1] = a.y * qs; qv[qi][2] = a.z * qs; qv[qi][3] = a.w * qs;
    qv[qi][4] = b.x * qs; qv[qi][5] = b.y * qs; qv[qi][6] = b.z * qs; qv[qi][7] = b.w * qs;
  }
  for (int sb = s0; sb < s1; sb += G) {
    float a[G][1 + NQ];
#pragma unroll
    for (int g = 0; g < G; g++) {
#pragma unroll
      for (int k = 0; k <= NQ; k++) a[g][k] = 0.f;
      const int s = sb + g;
      if (s < s1) {
        uint4 u = *reinterpret_cast<const uint4*>(msrc(sm, nb, s) + k0);
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
        const bool add = (s == nb) && addend;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float2 f = __bfloat1622float2(h[i]);
          if (add) { f.x += addend[2 * i]; f.y += addend[2 * i + 1]; }
          a[g][0] += f.x * f.x;
          a[g][0] += f.y * f.y;
#pragma unroll
          for (int qi = 0; qi < NQ; qi++) { a[g][1 + qi] += f.x * qv[qi][2 * i]; a[g][1 + qi] += f.y * qv[qi][2 * i + 1]; }
        }
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1)
#pragma unroll
      for (int g = 0; g < G; g++)
#pragma unroll
        for (int k = 0; k <= NQ; k++) a[g][k] += __shfl_xor_sync(0xffffffff, a[g][k], o);
    if (lane == 0) {
#pragma unroll
      for (int g = 0; g < G; g++)
        if (sb + g < s1)
#pragma unroll
          for (int k = 0; k <= NQ; k++) sm.red[w * 40 + (sb + g) * 4 + k] = a[g][k];
    }
  }
}
template <int NQ>
__device__ __forceinline__ void mix_finish8(Smem& sm, int n) {
  float* red = sm.red;
  __syncthreads();
  // warp 0, lane s < n: the source's sums over the 18 active warps, 1/rms, and the softmax over sources (shuffles)
  if (threadIdx.x < 32) {
    const int s = threadIdx.x;
    const bool on = s < n;
    float f[1 + NQ];
#pragma unroll
    for (int k = 0; k <= NQ; k++) {
      float t = 0.f;
      if (on) {
#pragma unroll
        for (int j = 0; j < W8; j++) t += red[j * 40 + s * 4 + k];
      }
      f[k] = t;
    }
    const float iv = on ? rsqrtf(f[0] / (float)H + 1e-6f) : 0.f;
    if (on) sm.minv[s] = iv;
#pragma unroll
    for (int qi = 0; qi < NQ; qi++) {
      float lg = on ? f[1 + qi] * iv : -INFINITY;
      float m = lg;
#pragma unroll
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
      float p = on ? expf(lg - m) : 0.f;
      // the sum in source order (as the serial loop did): prefix through shuffles from lane 0 upwards
      float z = 0.f;
      for (int q = 0; q < n; q++) z += __shfl_sync(0xffffffff, p, q);
      if (on) sm.mixp[qi * 10 + s] = p / z;
    }
  }
  __syncthreads();
}
__device__ __forceinline__ void mix_out8(Smem& sm, int nb, int n, const float* addend, int qi, float* out) {
  if (!ACT8) return;
  const int k0 = threadIdx.x * 8;
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = 0.f;
  for (int s = 0; s < n; s++) {
    uint4 u = *reinterpret_cast<const uint4*>(msrc(sm, nb, s) + k0);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
    const float p = sm.mixp[qi * 10 + s], iv = sm.minv[s];
    const bool add = (s == nb) && addend;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float2 f = __bfloat1622float2(h[i]);
      if (add) { f.x += addend[2 * i]; f.y += addend[2 * i + 1]; }
      out[2 * i] += p * (f.x * iv);
      out[2 * i + 1] += p * (f.y * iv);
    }
  }
}

// phase: (mlp all-reduce of layer i-1, sconv, finalize) + attention input of layer i. One block per token.
template <bool RED>
__device__ void phase_attn_in(const Ctx& c, int i, int b, Smem& sm, bool dowait, unsigned ev) {
  const LayerW& L = c.L[i];
  const bool writer = !RED || blockIdx.x == 0;
  bf16* xfd = RED ? &sm.fx[0][0] : c.xfrag;
  const int xstride = RED ? H : 72 * 32 * 16;
  float* red = sm.red;
  bf16* part = sm.blk[8];
  const int k0 = threadIdx.x * 8;
  if (i == 0) {
    float u[8];
    if (ACT8) ld8g(c.embed + (long)c.h.tokens[b] * H + k0, u);
    rmsnorm8(u, c.embed_norm, red);
    if (ACT8) st8(part + k0, u);
    rmsnorm8(u, L.norm[0], red);
    if (ACT8) put8<RED>(xfd, b, k0, u);
    return;
  }
  const int lp = i - 1;
  // ---- before the cross-card wait: the finished blocks' statistics
  const int nb_before = (lp == 0) ? 0 : ((lp - 1) / 6 + 1);
  const bool boundary = (lp % 6) == 0;
  const int nb = nb_before + (boundary ? 1 : 0);
  if (boundary) {
    if (ACT8) *reinterpret_cast<uint4*>(sm.blk[nb_before] + k0) = *reinterpret_cast<const uint4*>(part + k0);
    __syncthreads();
  }
  if (L.glob) mix_part8<3>(sm, nb, 0, nb, nullptr, L.aq);
  else mix_part8<1>(sm, nb, 0, nb, nullptr, L.aq);
  (void)dowait; (void)ev;   // the mlp values are polled by tag in raw8
  TM(i, 1);
  if (ACT8) {
    float rw[8], mlp[8], at[8], pv[8];
    raw8(c.comm[c.h.card]->llm[lp & 1], b, c.h.tag0 + lp + 1, rw, k0);
    conv8(c.hist_mlp + ((((long)c.h.parity * NL + lp) * MAXB + b) * 3) * H, rw, c.L[lp].mlp_sc,
          c.hist_mlp + ((((long)(c.h.parity ^ 1) * NL + lp) * MAXB + b) * 3) * H, writer, mlp, k0);
    ld8(sm.ats + k0, at);
    ld8(part + k0, pv);
    if (c.dbg && writer) st8(c.dbg + ((long)(lp * 2 + 1) * MAXB + b) * H + k0, mlp);
#pragma unroll
    for (int e = 0; e < 8; e++) pv[e] = boundary ? (at[e] + mlp[e]) : ((pv[e] + at[e]) + mlp[e]);
    st8(part + k0, pv);
  }
  TM(i, 16);
  __syncthreads();
  TM(i, 17);
  const int nq = L.glob ? 3 : 1;
  if (L.glob) { mix_part8<3>(sm, nb, nb, nb + 1, nullptr, L.aq); mix_finish8<3>(sm, nb + 1); }
  else { mix_part8<1>(sm, nb, nb, nb + 1, nullptr, L.aq); mix_finish8<1>(sm, nb + 1); }
  TM(i, 18);
  for (int k = 0; k < nq; k++) {
    float o[8];
    mix_out8(sm, nb, nb + 1, nullptr, k, o);
    rmsnorm8(o, L.norm[k], red);
    if (ACT8) put8<RED>(xfd + k * xstride, b, k0, o);
  }
}

// attention: block per (token b, local kv head kh, chunk ch)
__device__ PHASE_NI void phase_attention(const Ctx& c, int i, int b, int kh, int ch, int nch, Smem& sm, bool combine, int cs = CHUNK, bool gs = false) {
  const LayerW& L = c.L[i];
#define TMA(id) do { if (b == 0 && kh == 0 && ch == 0) TM(i, id); \
    } while (0)
  TMA(32);
  const int t = c.h.pos;
  const int lo = L.glob ? 0 : max(0, t + 1 - WIN);
  const int p0 = lo + ch * cs, p1 = min(t + 1, p0 + cs);   // cs: positions per chunk block (128, or 64 / 32 when blocks are free)
  const bf16* qkv = c.qkvo + b * 3840;
  // row offsets in the local qkv matrix
  const int OQ = 0, OG = 1536, OR = 3072, OS = 3264, OK = 3280, ODK = 3536, OV = L.glob ? 3552 : 3536, ODV = 3808;
  float tau = 1.0f;
  if (L.glob) tau = 1.0f + 0.1f * logf(fmaxf((float)(t + 1) / 128000.0f, 1.0f));
  bf16* kcache = c.kc + (((long)(i * MAXB + b) * NKVL + kh) * c.maxctx) * HD;
  bf16* vcache = c.vc + (((long)(i * MAXB + b) * NKVL + kh) * c.maxctx) * HD;
  int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  float* red = sm.red;
  // q and r of the 6 heads (scale, l2 clip, tau) as BF16 B fragments of the score MMAs: sm.qv as bf16 [8][128],
  // sm.misc + 16 as bf16 [8][16]; heads 6 and 7 are zero columns. Warp w < 6 does head w.
  bf16* qb = reinterpret_cast<bf16*>(sm.qv);
  bf16* rb = reinterpret_cast<bf16*>(sm.misc + 16);
  if (w < 6) {
    int hl = kh * 6 + w;
    float v[4], ss = 0.f;
    float rv = 0.f;
    if (lane < 16) rv = bf(qkv[OR + hl * 16 + lane]) * bf(L.rscale[hl * 16 + lane]);
    for (int k = 0; k < 4; k++) {
      int d = lane + 32 * k;
      v[k] = bf(qkv[OQ + hl * HD + d]) * bf(L.qscale[hl * HD + d]);
      ss += v[k] * v[k];
    }
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    float f = fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
    for (int k = 0; k < 4; k++) {
      float qq = rbf(v[k] * f);
      if (L.glob) qq = rbf(qq * tau);
      qb[w * HD + lane + 32 * k] = tobf(qq);
    }
    float rs = rv * rv;
    for (int o = 16; o; o >>= 1) rs += __shfl_xor_sync(0xffffffff, rs, o);
    float rf = fminf(8.0f / (sqrtf(rs) + 1e-6f), 1.0f);
    float rr = rbf(rv * rf);
    if (L.glob) rr = rbf(rr * tau);
    if (lane < 16) rb[w * 16 + lane] = tobf(rr);
  } else if (w >= 14 && w < 16) {
    for (int k = 0; k < 4; k++) qb[(w - 8) * HD + lane + 32 * k] = tobf(0.f);
    if (lane < 16) rb[(w - 8) * 16 + lane] = tobf(0.f);
  }
  // K and V rows of this chunk into shared memory (the B-fragment area is free during attention); row t later
#ifndef ATT_NOSTAGE   // K/V rows in shared memory: read from L1 instead, the V loop and the score MMAs took 5 us longer (e0075/e0076)
#define KST (HD + 8)   // padded row stride (272 B): a warp reading 32 rows at one offset hits distinct banks
  bf16* ks = reinterpret_cast<bf16*>(sm.stage);
  bf16* vs = ks + CHUNK * KST;
  {
    const int np = p1 - p0;
    for (int k = tid; k < np * 16; k += NTH) {   // 16 x 16 B per row, asynchronous
      int r = k >> 4, q = k & 15;
      if (p0 + r == t) continue;
      unsigned dk = (unsigned)__cvta_generic_to_shared(ks + r * KST + q * 8), dv = (unsigned)__cvta_generic_to_shared(vs + r * KST + q * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dk), "l"(kcache + (long)(p0 + r) * HD + q * 8) : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dv), "l"(vcache + (long)(p0 + r) * HD + q * 8) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
#else
  // K and V rows of the chunk are read where they are (prefetched to L2 during the attention-input phase, then L1):
  // staging them cost 70 KB of shared memory, i.e. of the SM's L1 (see Smem)
#define KST HD
  const bf16* ks = kcache + (long)p0 * HD;
  const bf16* vs = vcache + (long)p0 * HD;
  {   // pull the chunk's rows into L1 now, one 128-byte line per thread (the loops below would take the L2 misses
      // one after the other: attention 11.2 -> 16.5 us without this, e0075)
    const int nl = (p1 - p0) * 2;
    for (int k = tid; k < 2 * nl; k += NTH) {
      const char* q = (k < nl) ? (const char*)ks + (long)k * 128 : (const char*)vs + (long)(k - nl) * 128;
      unsigned dummy;
      asm volatile("ld.global.u32 %0, [%1];" : "=r"(dummy) : "l"(q) : "memory");
    }
  }
#endif
  // new k, v at position t (by the chunk that holds t): warps 6-9 k, warps 10-13 v, one channel per lane;
  // the per-head norm over 128 channels through shared memory and a named barrier of these 8 warps
  if (p1 == t + 1 && w >= 6 && w < 14) {
    const int kv = (w - 6) >> 2, q4 = (w - 6) & 3;
    const int d = q4 * 32 + lane, ch_ = kh * HD + d;
    const int base = (kv == 0) ? OK : OV;
    const bf16* sc = (kv == 0) ? L.ksc : L.vsc;
    bf16* hist = (kv == 0 ? c.hist_k : c.hist_v) + ((long)(i * MAXB + b) * 3) * 256;
    const float raw = bf(qkv[base + ch_]);
    const float h0 = bf(hist[ch_]), h1 = bf(hist[256 + ch_]), h2 = bf(hist[512 + ch_]);
    float y;
    if (!L.glob) {
      y = sconv4(h0, h1, h2, raw, sc + ch_ * 4);
    } else {
      const int dbase = (kv == 0) ? ODK : ODV;
      float acc = 0.f;
      const float xs[4] = {raw, h2, h1, h0};
#pragma unroll
      for (int lag = 0; lag < 4; lag++) {
        float tap = bf(sc[lag * 256 + ch_]) + bf(qkv[dbase + kh * 4 + lag]);
        acc = acc + xs[lag] * tap;
      }
      y = rbf((acc + raw) * 0.3333333333333333f);
    }
    hist[ch_] = tobf(h1); hist[256 + ch_] = tobf(h2); hist[512 + ch_] = tobf(raw);
    float ss = y * y;
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
    if (lane == 0) sm.misc[112 + kv * 4 + q4] = ss;
    asm volatile("bar.sync 1, 256;" ::: "memory");
    float tot = sm.misc[112 + kv * 4 + 0] + sm.misc[112 + kv * 4 + 1] + sm.misc[112 + kv * 4 + 2] + sm.misc[112 + kv * 4 + 3];
    float yo = (kv == 0) ? rbf(y * fminf(22.627416997969522f / (sqrtf(tot) + 1e-6f), 1.0f)) : rbf(y * rsqrtf(tot / 128.0f + 1e-6f));
    (kv == 0 ? kcache : vcache)[(long)t * HD + d] = tobf(yo);
#ifndef ATT_NOSTAGE
    (kv == 0 ? ks : vs)[(t - p0) * KST + d] = tobf(yo);
#endif
  }
  TMA(43);
#ifndef ATT_NOSTAGE
  asm volatile("cp.async.wait_all;" ::: "memory");
#endif
  TMA(44);
  __syncthreads();
  TMA(33);
  TMA(34);
  const int np = p1 - p0;
  // scores: warp w < 8 takes the 16 positions p0 + 16 w ..: K rows x q as 8 BF16 MMAs (FP32 accumulation of
  // BF16 x BF16 products, as the engines' QK^T), the relative bias as one more MMA over the 16 rel dimensions
#ifdef ATT_SCALAR
  {
    const int sj = tid >> 7, sp = tid & 127;
    if (sj < 6 && sp < np) {
      float dot = 0.f;
      for (int dd = 0; dd < HD; dd++) dot += bf(qb[sj * HD + dd]) * bf(ks[sp * KST + dd]);
      float pr = 0.f;
      const int dist = t - (p0 + sp);
      if (dist < RELEXT) for (int dd = 0; dd < DREL; dd++) pr += bf(rb[sj * 16 + dd]) * bf(L.relp[(long)dist * DREL + dd]);
      sm.att[sj * CHUNK + sp] = dot * 0.08838834764831845f + rbf(pr);
    }
  }
  if (false) {
#else
  if (w < 8 && w * 16 < np) {
#endif
    const int g = lane >> 2, tq = lane & 3;
    const bf16* k0 = ks + min(w * 16 + g, np - 1) * KST + 2 * tq;       // (rows >= np: not used; stay inside the cache)
    const bf16* k1 = ks + min(w * 16 + g + 8, np - 1) * KST + 2 * tq;
    const bf16* qh = qb + g * HD + 2 * tq;
    float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int m = 0; m < 8; m++) {
      const uint32_t a0 = *reinterpret_cast<const uint32_t*>(k0 + 16 * m), a1 = *reinterpret_cast<const uint32_t*>(k1 + 16 * m);
      const uint32_t a2 = *reinterpret_cast<const uint32_t*>(k0 + 16 * m + 8), a3 = *reinterpret_cast<const uint32_t*>(k1 + 16 * m + 8);
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(qh + 16 * m), b1 = *reinterpret_cast<const uint32_t*>(qh + 16 * m + 8);
      mma_bf16(d, a0, a1, a2, a3, b0, b1);
    }
    float e[4] = {0.f, 0.f, 0.f, 0.f};
    {
      const int dist0 = t - (p0 + w * 16 + g), dist1 = dist0 - 8;
      const bool v0 = dist0 >= 0 && dist0 < RELEXT, v1 = dist1 >= 0 && dist1 < RELEXT;
      const bf16* r0p = L.relp + (long)(v0 ? dist0 : 0) * DREL + 2 * tq;
      const bf16* r1p = L.relp + (long)(v1 ? dist1 : 0) * DREL + 2 * tq;
      const uint32_t a0 = v0 ? *reinterpret_cast<const uint32_t*>(r0p) : 0u, a1 = v1 ? *reinterpret_cast<const uint32_t*>(r1p) : 0u;
      const uint32_t a2 = v0 ? *reinterpret_cast<const uint32_t*>(r0p + 8) : 0u, a3 = v1 ? *reinterpret_cast<const uint32_t*>(r1p + 8) : 0u;
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(rb + g * 16 + 2 * tq), b1 = *reinterpret_cast<const uint32_t*>(rb + g * 16 + 2 * tq + 8);
      mma_bf16(e, a0, a1, a2, a3, b0, b1);
    }
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const int row = w * 16 + g + (q >> 1) * 8, head = 2 * tq + (q & 1);
      if (head < 6 && row < np) sm.att[head * CHUNK + row] = d[q] * 0.08838834764831845f + rbf(e[q]);
    }
  }
  __syncthreads();
#ifdef ATT_CHECK
  if (c.tprof) {
    const int sj = tid >> 7, sp = tid & 127;
    if (sj < 6 && sp < np) {
      float dot = 0.f;
      for (int dd = 0; dd < HD; dd++) dot += bf(qb[sj * HD + dd]) * bf(ks[sp * KST + dd]);
      float pr = 0.f;
      const int dist = t - (p0 + sp);
      if (dist < RELEXT) for (int dd = 0; dd < DREL; dd++) pr += bf(rb[sj * 16 + dd]) * bf(L.relp[(long)dist * DREL + dd]);
      const float ref = dot * 0.08838834764831845f + rbf(pr), got = sm.att[sj * CHUNK + sp];
      const float df = fabsf(ref - got);
      unsigned long long* dbgp = (unsigned long long*)&c.tprof[49 * 64 + 1184 + 150 * 16];
      if (!(df <= 0.05f)) {
        atomicAdd(dbgp, 1ull);
        if (atomicAdd(dbgp + 1, 1ull) < 6ull) {   // first few: record
          unsigned long long k = atomicAdd(dbgp + 2, 1ull);
          if (k < 6) { dbgp[4 + k * 4] = ((unsigned long long)i << 32) | (unsigned)(sj * 1000 + sp); dbgp[5 + k * 4] = __float_as_uint(ref); dbgp[6 + k * 4] = __float_as_uint(got); dbgp[7 + k * 4] = ((unsigned long long)np << 32) | (unsigned)(t - p0); }
        }
      }
      atomicAdd(dbgp + 3, 1ull);
    }
  }
  __syncthreads();
#endif
  TMA(35);
  // softmax pieces of the chunk: one thread per (head, position); max and sum of a head over its 4 warps
  {
    const int sj = tid >> 7, sp = tid & 127;
    const bool on = sj < 6 && sp < np;
    const float x = on ? sm.att[sj * CHUNK + sp] : -INFINITY;
    float m = x;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) red[w] = m;
    __syncthreads();
    const int w4 = w & ~3;
    m = fmaxf(fmaxf(red[w4], red[w4 + 1]), fmaxf(red[w4 + 2], red[w4 + 3]));
    const float ex = on ? expf(x - m) : 0.f;
    if (on) sm.att[sj * CHUNK + sp] = ex;
    float l = ex;
#pragma unroll
    for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffff, l, o);
    if (lane == 0) sm.bestv[w] = l;   // (scratch: the unembedding re-initialises it)
    __syncthreads();
    if (sj < 6 && sp == 0) { sm.misc[sj * 2] = m; sm.misc[sj * 2 + 1] = ((sm.bestv[w4] + sm.bestv[w4 + 1]) + sm.bestv[w4 + 2]) + sm.bestv[w4 + 3]; }
  }
  TMA(36);
  // V: one thread per (head, dimension pair), all positions of the chunk; writes the chunk partial directly
  const bool owner = combine && ch == nch - 1;   // this block combines the chunks of its (token, KV head)
  float* chunkp = c.achunk + ((long)((b * NKVL + kh) * MAXCHUNK + ch) * 6) * (2 + HD);
  // (all threads: (head, dimension pair, half of the positions); the two halves summed through shared memory)
  {
    const int half = tid / (6 * (HD / 2)), r6 = tid % (6 * (HD / 2));
    const int j = r6 / (HD / 2), d = (r6 % (HD / 2)) * 2;
    const int pm = (np + 1) / 2, pa = half ? pm : 0, pb = half ? np : pm;
    float o0 = 0.f, o1 = 0.f;
    const float* aj = sm.att + j * CHUNK;
    if (half < 2) {
#pragma unroll 4
      for (int p = pa; p < pb; p++) {
        float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(vs + p * KST + d));
        const float e = aj[p];
        o0 += e * v.x; o1 += e * v.y;
      }
      if (half == 1) *reinterpret_cast<float2*>(&sm.opart[0] + r6 * 2) = make_float2(o0, o1);
    }
    __syncthreads();
    if (half == 0) {
      float2 hv = *reinterpret_cast<const float2*>(&sm.opart[0] + r6 * 2);
      const float2 fin = make_float2(o0 + hv.x, o1 + hv.y);
      if (owner) *reinterpret_cast<float2*>(&sm.opart[0] + r6 * 2) = fin;   // the combining block keeps its own chunk: [head][dim]
      else st_l2(reinterpret_cast<float2*>(chunkp + j * (2 + HD) + 2 + d), fin);
    }
  }
  if (tid < 6 && !owner) { st_l2(&chunkp[tid * (2 + HD)], sm.misc[tid * 2]); st_l2(&chunkp[tid * (2 + HD) + 1], sm.misc[tid * 2 + 1]); }
  __syncthreads();
  TMA(36);
  TMA(37);
  if (!combine) return;   // (old B=1 kernel: every block combines the chunks itself in front of o_proj)
  // The block of the last chunk (the one with the new position, the longest path) combines. The others publish a
  // tag after a fence; no counter, and the combining block needs no fence of its own (2 x ~0.5 us off its path).
  const unsigned ctg = c.h.tag0 + i + 1;
  uint2* ctags = c.ctag + (b * NKVL + kh) * MAXCHUNK;
  if (!owner) {
    if (tid == 0) { FENCE_A(); put_l2(&ctags[ch], 0.f, ctg); }
    return;
  }
  if (nch > 1) {
    if (tid < nch - 1) { uint2 q = ld_gpu_v2(&ctags[tid]); while (q.y != ctg) q = ld_gpu_v2(&ctags[tid]); }
    __syncthreads();
  }
  TMA(38);
  const float* base = c.achunk + ((long)((b * NKVL + kh) * MAXCHUNK) * 6) * (2 + HD);
  // per head: chunk scale factors exp(m_q - M) into shared memory (sm.att is free now), then 1/L and lse
  if (w < 6) {
    const int j = w;
    float mq = (lane < nch - 1) ? __ldcg(base + (lane * 6 + j) * (2 + HD)) : (lane == nch - 1 ? sm.misc[j * 2] : -INFINITY);
    float lq = (lane < nch - 1) ? __ldcg(base + (lane * 6 + j) * (2 + HD) + 1) : (lane == nch - 1 ? sm.misc[j * 2 + 1] : 0.f);
    float M = mq;
    for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o));
    float f = (lane < nch) ? expf(mq - M) : 0.f;
    // L = sum over chunks in chunk order (lane order)
    float Ls = 0.f;
    for (int q = 0; q < nch; q++) Ls += __shfl_sync(0xffffffff, lq * f, q);
    if (lane < nch) sm.att[j * CHUNK + lane] = f;
    if (lane == 0) { sm.misc[j * 2] = sigm((M + logf(Ls)) - bf(qkv[OS + kh * 6 + j])); sm.misc[j * 2 + 1] = Ls; }   // sink gate, once per head
  }
  __syncthreads();
  for (int idx = tid; idx < 6 * HD; idx += NTH) {
    int j = idx / HD, d = idx % HD;
    float O = 0.f;
    for (int q = 0; q < nch - 1; q++) O += __ldcg(base + (q * 6 + j) * (2 + HD) + 2 + d) * sm.att[j * CHUNK + q];
    O += sm.opart[j * HD + d] * sm.att[j * CHUNK + nch - 1];   // this block's own chunk
    const float Ls = sm.misc[j * 2 + 1];
    float out = O / Ls;
    int hl = kh * 6 + j;
    if (gs) {   // sigmoid(gate) from the blocks that ran the gate GEMV beside this phase
      const uint2* gp = &c.gll[b * 1536 + hl * HD + d];
      const unsigned gtag = c.h.tag0 + i + 1;
      uint2 gq = ld_gpu_v2(gp);
      while (gq.y != gtag) gq = ld_gpu_v2(gp);
      out = out * __uint_as_float(gq.x);
    } else {
      float gate = bf(qkv[OG + hl * HD + d]);
      out = out * sigm(gate);
    }
    out = out * sm.misc[j * 2];
    put16<false>(c.ofrag, b, hl * HD + d, out);
  }
  TMA(39);
#undef TMA
}

// phase: attention all-reduce + sconv + mlp input mix + norm + quantization. One block per token.
template <bool RED>
__device__ void phase_post_attn(const Ctx& c, int i, int b, Smem& sm, bool dowait, unsigned ev) {
  const LayerW& L = c.L[i];
  const bool writer = !RED || blockIdx.x == 0;
  float* red = sm.red;
  const int k0 = threadIdx.x * 8;
  // ---- before the wait
  const int nb = (i == 0) ? 0 : ((i - 1) / 6 + 1);
  const bool boundary = (i % 6) == 0;
  mix_part8<1>(sm, nb, 0, boundary ? nb + 1 : nb, nullptr, L.mq);   // finished blocks (and the old partial at a boundary)
  (void)dowait; (void)ev;
  TM(i, 6);
  float at[8];
  if (ACT8) {
    float rw[8];
    raw8(c.comm[c.h.card]->lla[i & 1], b, c.h.tag0 + i + 1, rw, k0);
    conv8(c.hist_attn + ((((long)c.h.parity * NL + i) * MAXB + b) * 3) * H, rw, L.attn_sc,
          c.hist_attn + ((((long)(c.h.parity ^ 1) * NL + i) * MAXB + b) * 3) * H, writer, at, k0);
    st8(sm.ats + k0, at);
    if (c.dbg && writer) st8(c.dbg + ((long)(i * 2) * MAXB + b) * H + k0, at);
  }
  TM(i, 20);
  __syncthreads();
  float out[8];
  if (boundary) {
    mix_part8<1>(sm, nb, nb + 1, nb + 2, nullptr, L.mq);
    mix_finish8<1>(sm, nb + 2);
    mix_out8(sm, nb, nb + 2, nullptr, 0, out);
  } else {
    mix_part8<1>(sm, nb, nb, nb + 1, at, L.mq);
    mix_finish8<1>(sm, nb + 1);
    mix_out8(sm, nb, nb + 1, at, 0, out);
  }
  TM(i, 21);
  rmsnorm8(out, L.mlp_norm, red);
  float* x = out;
  TM(i, 22);
  if (ACT8) put8<RED>(RED ? sm.fx2 : c.x2frag, b, k0, x);
  uint8_t* f13 = RED ? sm.f13 : c.w13frag;
  float* sc13 = RED ? sm.sc13 : c.w13sc;
  if (!L.mx) {
    float outer = quant_nvfp4_8<RED>(x, b, reinterpret_cast<__half*>(f13), red);
    if (threadIdx.x == 0) sc13[b] = outer;
  } else {
    quant_mxfp8_8<RED>(x, b, f13, sc13, threadIdx.x * 8, ACT8);
  }
}

// routing, computed redundantly by every block: warp b < B does token b's top-8; thread 0 the sorted union.
// The w13 input of the layer for the tensor core, issued beside the routing phase (warps 20-23 idle there): the
// block scales of every token into tensor memory (lane n of every 32-lane partition: token n; 4 columns per K step)
// and the codes into shared memory. In the GEMV this took 1.5-3 us before the first MMA (e0156).
__device__ __noinline__ void tc_prep13(const Ctx& c, Smem& sm) {
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  if (w < 12) return;   // warps 12-23: three groups of four warps, three stores of 32 columns each
  const uint32_t la = (uint32_t)((w & 3) * 32) << 16, t_sfb = sm.tct + 96;
  if (tid == 640) { const uint32_t bb = sh(sm.tcbar) + 256; mbar_expect(bb, 18432); tc_bulk(sh(sm.tcB), c.w13frag, 18432, bb); }
  const uint4* sf = reinterpret_cast<const uint4*>(c.w13frag + 40960) + (lane & 7) * 18;
  const uint4 zz = make_uint4(0, 0, 0, 0);
  const int g3 = ((w - 12) >> 2) * 3;
  for (int k8 = g3; k8 < g3 + 3; k8++) tc_st32(t_sfb + 32 * k8 + la, lane < 8 ? sf[k8 * 2] : zz, lane < 8 ? sf[k8 * 2 + 1] : zz);
  tc_wait_st(); tc_fence_before();
  if (c.prog && blockIdx.x == 0 && tid == 640) c.prog[c.h.card * 16 + 7] += 1;
}
template <bool W2> __device__ __noinline__ void gemv_tc(const Ctx& c, Smem& sm, int layer, int nu);
__device__ __noinline__ void tc_begin(const Ctx& c, Smem& sm);
__device__ __noinline__ void tc_end(const Ctx& c, Smem& sm);
template <bool TC>
__device__ PHASE_NI void phase_route(const Ctx& c, int i, Smem& sm) {
#ifndef TC_BODY_CALLS
  // tensor memory and mbarriers of the launch: set up in the first layer's routing phase, released in the last
  // one's (the host passes first | last << 8 in pad1); as calls in the kernel body they cost its loops registers
  if (TC && i == (int)(c.h.pad1 & 255u)) tc_begin(c, sm);
#endif
  if (TC) TC_PROG(6, i + 1);
  if (TC && TC13 && !c.L[i].mx) tc_prep13(c, sm);   // (called from here: a call in the kernel body cost its GEMV loops registers, e0159)
  const LayerW& L = c.L[i];
  const int lane = threadIdx.x & 31, b = threadIdx.x >> 5;
  const int B = c.h.B;
  float* sel = &sm.opart[0];   // [B][384] selection scores (attention scratch, free here)
  const unsigned tag = c.h.tag0 + i + 1;
  {   // sigmoid(router logit) of every (token, expert) from the router blocks: up to 4 words per thread
    const int k0 = threadIdx.x;
    const int nv = (B * NE - k0 + NTH - 1) / NTH;   // values of this thread (B / 2)
    uint2 r0 = make_uint2(0u, tag), r1 = r0, r2 = r0, r3 = r0;
    do {
      if (nv > 0) r0 = ld_l2_v2(&c.rll[k0]);
      if (nv > 1) r1 = ld_l2_v2(&c.rll[k0 + NTH]);
      if (nv > 2) r2 = ld_l2_v2(&c.rll[k0 + 2 * NTH]);
      if (nv > 3) r3 = ld_l2_v2(&c.rll[k0 + 3 * NTH]);
    } while (r0.y != tag || r1.y != tag || r2.y != tag || r3.y != tag);
    if (nv > 0) sel[k0] = __uint_as_float(r0.x) + L.rbias[k0 % NE];
    if (nv > 1) sel[k0 + NTH] = __uint_as_float(r1.x) + L.rbias[(k0 + NTH) % NE];
    if (nv > 2) sel[k0 + 2 * NTH] = __uint_as_float(r2.x) + L.rbias[(k0 + 2 * NTH) % NE];
    if (nv > 3) sel[k0 + 3 * NTH] = __uint_as_float(r3.x) + L.rbias[(k0 + 3 * NTH) % NE];
  }
  __syncthreads();
  if (b < B) {
    float* sb = sel + b * NE;
    int cid = 0;
#ifdef TOPK_NEW   // not faster than the rescan (3.98 vs 3.65 us at B=1, e0054): the phase is not bound by the rounds
    // Lane holds experts lane + 32 k (k < 12) and keeps its best two; a round is a warp max, a ballot and a shuffle
    // (~26 instructions; rescanning 12 values every round was ~70). A lane that wins a second time rescans.
    float b1 = -INFINITY, b2 = -INFINITY; int k1 = 0, k2 = 0;
    bool have2 = true;
#pragma unroll
    for (int k = 0; k < 12; k++) {
      const float x = sb[lane + 32 * k];
      if (x > b1) { b2 = b1; k2 = k1; b1 = x; k1 = k; } else if (x > b2) { b2 = x; k2 = k; }
    }
    for (int r = 0; r < TOPK; r++) {
      float m = b1;
#pragma unroll
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
      const unsigned bal = __ballot_sync(0xffffffff, b1 == m);
      int wl = __ffs(bal) - 1;
      if (bal & (bal - 1)) {   // equal values in several lanes: the lowest expert index wins
        int key = (b1 == m) ? (k1 * 32 + lane) : 0x7fffffff;
#pragma unroll
        for (int o = 16; o; o >>= 1) key = min(key, __shfl_xor_sync(0xffffffff, key, o));
        wl = key & 31;
      }
      const int be = __shfl_sync(0xffffffff, k1, wl) * 32 + wl;
      if (lane == r) cid = be;
      bool need = false;
      if (lane == wl) {
        sb[be] = -INFINITY;
        if (have2) { b1 = b2; k1 = k2; have2 = false; } else need = true;
      }
      if (__any_sync(0xffffffff, need)) {
        if (need) {
          b1 = -INFINITY; b2 = -INFINITY; k1 = 0; k2 = 0;
#pragma unroll
          for (int k = 0; k < 12; k++) {
            const float x = sb[lane + 32 * k];
            if (x > b1) { b2 = b1; k2 = k1; b1 = x; k1 = k; } else if (x > b2) { b2 = x; k2 = k; }
          }
          have2 = true;
        }
      }
    }
#else
    for (int r = 0; r < TOPK; r++) {
      float bv = -INFINITY; int bi = 1 << 30;
#pragma unroll
      for (int k = 0; k < 12; k++) {
        int e = lane + 32 * k;
        float v = sb[e];
        if (v > bv) { bv = v; bi = e; }
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, o); int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if (lane == r) cid = bi;
      if (lane == 0) sb[bi] = -INFINITY;
      __syncwarp();
    }
#endif
    // lanes 0..7 hold the chosen experts in rank order
    float sg = 0.f;
    if (lane < TOPK) sg = __uint_as_float(ld_l2_v2(&c.rll[b * NE + cid]).x);
    float z = sg;
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) z += __shfl_xor_sync(0xffffffff, z, o);
    if (lane < TOPK) {
      sm.ids[b * TOPK + lane] = cid;
      sm.wts[b * TOPK + lane] = sg / z * 8.0f;
      if (c.dbg_ids && blockIdx.x == 0) c.dbg_ids[(i * MAXB + b) * TOPK + lane] = cid;
    }
  }
  __syncthreads();
  TM(i, 24);
  // sorted union: a 384-bit mask in shared memory (sm.red is free here), then warp 0 compacts it
  unsigned* msk = reinterpret_cast<unsigned*>(sm.red);
  if (threadIdx.x < 12) msk[threadIdx.x] = 0;
  __syncthreads();
  if (threadIdx.x < c.h.B * TOPK) { int e = sm.ids[threadIdx.x]; atomicOr(&msk[e >> 5], 1u << (e & 31)); }
  __syncthreads();
  if (threadIdx.x < 32) {
    unsigned m = (lane < 12) ? msk[lane] : 0u;
    int cnt = __popc(m), pre = cnt;
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffff, pre, o); if (lane >= o) pre += v; }
    int n = pre - cnt;   // exclusive prefix
    if (lane < 12) sm.upre[lane] = n;
    while (m) { int bit = __ffs(m) - 1; m &= m - 1; sm.uni[n++] = lane * 32 + bit; }
    if (lane == 31) sm.nuni = pre;
  }
  if (threadIdx.x < MAXU) sm.rmask[threadIdx.x] = 0u;
  if (threadIdx.x < MAXU * MAXB) sm.wtu[threadIdx.x] = 0.f;
  __syncthreads();
  // position of every routed (token, rank) pair's expert in the sorted union, and the tokens routed to each union expert
  if (threadIdx.x < c.h.B * TOPK) {
    const int e = sm.ids[threadIdx.x];
    const int ui = sm.upre[e >> 5] + __popc(msk[e >> 5] & ((1u << (e & 31)) - 1u));
    sm.uix[threadIdx.x] = ui;
    sm.wtu[ui * MAXB + threadIdx.x / TOPK] = sm.wts[threadIdx.x];
    atomicOr(&sm.rmask[ui], 1u << (threadIdx.x / TOPK));
  }
  __syncthreads();
  // the NVFP4 w13 GEMV of the tensor-core kernel runs from here: a call in the kernel body costs the fragment-path
  // loops of the MXFP8 layers registers (w13 +3.7, w2 +3.1 us per layer at B=8, s0015)
  if (TC && TC13 && !c.L[i].mx) gemv_tc<false>(c, sm, i, sm.nuni);
#ifndef TC_BODY_CALLS
  if (TC && i == (int)((c.h.pad1 >> 8) & 255u)) { __syncthreads(); tc_end(c, sm); }
#endif
}

// quantize act (per (ui, token)) for w2 with the global amax. Block per (ui, token) pair.
__device__ void phase_quant2(const Ctx& c, int i, int ui, int b, Smem& sm) {
  const LayerW& L = c.L[i];
  // 576 values: thread t < 576? NTH=512 -> 2 passes; use per-thread 2 elements: k = tid, tid+512 (<576)
  if (threadIdx.x == 0) {   // wait for every card's tag of (expert ui, token b): this card's act is complete then
    float am = 0.f;
    for (int d = 0; d < NCARD; d++) am = fmaxf(am, ll_get1(&c.comm[c.h.card]->llx[i][d][ui][b], c.h.tag0 + i + 1));
    sm.misc[0] = am;
    __threadfence();
  }
  __syncthreads();
  if (threadIdx.x >= ILOC) return;   // 18 whole warps
  float x[1];
  const int n = 1;
  x[0] = bf(__ldcg(&c.act[(ui * MAXB + b) * ILOC + threadIdx.x]));
  uint8_t* frag = c.w2frag + (long)ui * 9 * 32 * (L.mx ? 16 : 32);
  if (!L.mx) {
    const float amax = sm.misc[0];
    // the block-of-16 shuffles need full warps: elements k >= 576 are padding in a separate warp range
    // k = tid + 512: tids 0..63 valid, rest padding -> whole warps 0,1 valid; padding warps compute garbage not stored
    __half* f = reinterpret_cast<__half*>(frag);
    // inline quantization (amax given); store only k < 576
    const float FMAX = 3.4028234663852886e38f;
    float outer = amax * (1.0f / 1536.0f);
    if (amax == 0.f) {
      for (int e = 0; e < n; e++) { int k = threadIdx.x + e * NTH; if (k < ILOC) f[fidx16(b, k)] = __float2half_rn(0.f); }
    } else {
      float encoder = fminf(1536.0f / amax, FMAX), decoder = 1.0f / encoder;
      for (int e = 0; e < n; e++) {
        int k = threadIdx.x + e * NTH;
        float xv = x[e];
        float bm = fabsf(xv);
        for (int o = 8; o; o >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, o));
        float best_val = 0.f, best_err = 0.f;
        for (int cand = 0; cand < 2; cand++) {
          float inv_max = cand == 0 ? (1.0f / 6.0f) : 0.25f;
          float sraw = fminf(bm * (encoder * inv_max), 448.0f);
          float scale = float(__nv_fp8_e4m3(sraw));
          float inverse = fminf(1.0f / (scale * decoder), FMAX);
          float val = xv * inverse, mag = fabsf(val), q;
          if (mag <= 0.25f) q = 0.f; else if (mag < 0.75f) q = 0.5f; else if (mag <= 1.25f) q = 1.0f; else if (mag < 1.75f) q = 1.5f;
          else if (mag <= 2.5f) q = 2.0f; else if (mag < 3.5f) q = 3.0f; else if (mag <= 5.0f) q = 4.0f; else q = 6.0f;
          if (val != val) q = 0.f;
          float vals = (signbit(val) ? -q : q) * scale;
          float err = fabsf(vals * outer - xv);
          for (int o = 8; o; o >>= 1) err += __shfl_xor_sync(0xffffffff, err, o);
          if (cand == 0) { best_val = vals; best_err = err; } else if (err < best_err) best_val = vals;
        }
        if (k < ILOC) f[fidx16(b, k)] = __float2half_rn(best_val);
      }
    }
    if (threadIdx.x == 0) c.w2sc[ui * 18 * MAXB + b] = outer;
  } else {
    float* sc = c.w2sc + (long)ui * 18 * MAXB;
    for (int e = 0; e < n; e++) {
      int k = threadIdx.x + e * NTH;
      float xv = x[e];
      float m = fabsf(xv);
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
      float raw = m * (1.0f / 448.0f);
      unsigned bits = __float_as_uint(raw);
      int ex = (int)(((bits + 0x007FFFFFu) & 0x7F800000u) >> 23);
      if (raw <= 5.877471754111438e-39f) ex = 0;
      if (ex > 254) ex = 254;
      float inverse = (ex == 0) ? 1.7014118346046923e38f : 1.0f / ue8m0((unsigned)ex);
      float v = fminf(fmaxf(xv * inverse, -448.f), 448.f);
      if (k < ILOC) {
        frag[fidx8(b, k)] = __nv_fp8_e4m3(v).__x;
        if ((threadIdx.x & 31) == 0) sc[(k >> 5) * MAXB + b] = ue8m0((unsigned)ex);
      }
    }
  }
}

// B == 1: every block quantizes the w2 inputs of all union experts into its shared memory (compact frags).
// Flat index over nu * 576 values: blocks of 16 (nvfp4) / 32 (mxfp8) never straddle experts.
__device__ void phase_quant2_red(const Ctx& c, int i, int nu, Smem& sm) {
  const LayerW& L = c.L[i];
  const int total = nu * ILOC;
  const float FMAX = 3.4028234663852886e38f;
  if (threadIdx.x < nu) {   // one poller per expert: every card's max for this expert (and this card's act is complete)
    float am = 0.f;
    for (int d = 0; d < NCARD; d++) am = fmaxf(am, ll_get1(&c.comm[c.h.card]->llx[i][d][threadIdx.x][0], c.h.tag0 + i + 1));
    sm.misc[threadIdx.x] = am;
    __threadfence();
  }
  __syncthreads();
  for (int base = 0; base < total; base += NTH) {
    int idx = base + threadIdx.x;
    if (base + (threadIdx.x & ~31) >= total) break;   // whole warps beyond the end
    int ui = idx / ILOC, kk = idx % ILOC;
    const float amax = sm.misc[ui];
    float xv = bf(__ldcg(&c.act[(ui * MAXB) * ILOC + kk]));
    if (!L.mx) {
      float outer = amax * (1.0f / 1536.0f);
      float best_val = 0.f;
      float bm = fabsf(xv);
      for (int o = 8; o; o >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, o));
      if (amax != 0.f) {
        float encoder = fminf(1536.0f / amax, FMAX), decoder = 1.0f / encoder;
        float best_err = 0.f;
        for (int cand = 0; cand < 2; cand++) {
          float inv_max = cand == 0 ? (1.0f / 6.0f) : 0.25f;
          float sraw = fminf(bm * (encoder * inv_max), 448.0f);
          float scale = float(__nv_fp8_e4m3(sraw));
          float inverse = fminf(1.0f / (scale * decoder), FMAX);
          float val = xv * inverse, mag = fabsf(val), q;
          if (mag <= 0.25f) q = 0.f; else if (mag < 0.75f) q = 0.5f; else if (mag <= 1.25f) q = 1.0f; else if (mag < 1.75f) q = 1.5f;
          else if (mag <= 2.5f) q = 2.0f; else if (mag < 3.5f) q = 3.0f; else if (mag <= 5.0f) q = 4.0f; else q = 6.0f;
          if (val != val) q = 0.f;
          float vals = (signbit(val) ? -q : q) * scale;
          float err = fabsf(vals * outer - xv);
          for (int o = 8; o; o >>= 1) err += __shfl_xor_sync(0xffffffff, err, o);
          if (cand == 0) { best_val = vals; best_err = err; } else if (err < best_err) best_val = vals;
        }
      }
      __half* f = reinterpret_cast<__half*>(sm.f2 + ui * 9 * 4 * 32);
      f[fidx16c(kk)] = __float2half_rn(best_val);
      if (kk == 0) sm.sc2[ui * 18 * MAXB] = outer;
    } else {
      float m = fabsf(xv);
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
      float raw = m * (1.0f / 448.0f);
      unsigned bits = __float_as_uint(raw);
      int ex = (int)(((bits + 0x007FFFFFu) & 0x7F800000u) >> 23);
      if (raw <= 5.877471754111438e-39f) ex = 0;
      if (ex > 254) ex = 254;
      float inverse = (ex == 0) ? 1.7014118346046923e38f : 1.0f / ue8m0((unsigned)ex);
      float v = fminf(fmaxf(xv * inverse, -448.f), 448.f);
      (sm.f2 + ui * 9 * 4 * 16)[fidx8c(kk)] = __nv_fp8_e4m3(v).__x;
      if ((kk & 31) == 0) sm.sc2[ui * 18 * MAXB + (kk >> 5) * MAXB] = ue8m0((unsigned)ex);
    }
  }
  __syncthreads();
}

// final: mlp all-reduce of layer 47 + finalize + output mix + norm -> hfrag
template <bool RED>
__device__ void phase_final(const Ctx& c, int b, Smem& sm) {
  const bool writer = !RED || blockIdx.x == 0;
  float* red = sm.red;
  bf16* part = sm.blk[8];
  const int lp = NL - 1, k0 = threadIdx.x * 8;
  if (ACT8) {
    float rw[8], mlp[8], at[8], pv[8];
    raw8(c.comm[c.h.card]->llm[lp & 1], b, c.h.tag0 + lp + 1, rw, k0);
    conv8(c.hist_mlp + ((((long)c.h.parity * NL + lp) * MAXB + b) * 3) * H, rw, c.L[lp].mlp_sc, c.hist_mlp + ((((long)(c.h.parity ^ 1) * NL + lp) * MAXB + b) * 3) * H, writer, mlp, k0);
    ld8(sm.ats + k0, at);
    ld8(part + k0, pv);
    if (c.dbg && writer) st8(c.dbg + ((long)(lp * 2 + 1) * MAXB + b) * H + k0, mlp);
#pragma unroll
    for (int e = 0; e < 8; e++) pv[e] = (pv[e] + at[e]) + mlp[e];   // layer 47 is not a boundary
    st8(part + k0, pv);
  }
  __syncthreads();
  mix_part8<1>(sm, 8, 0, 9, nullptr, c.oq);
  mix_finish8<1>(sm, 9);
  float out[8];
  mix_out8(sm, 8, 9, nullptr, 0, out);
  rmsnorm8(out, c.fnorm, red);
  if (ACT8) put8<RED>(RED ? sm.fh : c.hfrag, b, k0, out);
}

// ------------------------------------------------------------------------------------------------ team vector phases
// What costs time in the per-token vector phases (all-reduce, short conv, residual mix, norms, input quantization) is
// the instruction path of a thread (measured: ~5 ns per instruction of a lone warp, ~9 ns with 24 warps on the SM), not
// the SM's capacity: 8 elements per thread (576 threads of one block) are ~1800 instructions = 10-20 us per phase.
// So a token's 4608 elements go one per thread to TB = 6 whole blocks ("team"): block b * TB + tj owns elements
// [768 tj, 768 tj + 768). Reductions over the vector (sum of squares, query dots, amax) are done per warp (shuffles),
// per block (shared memory) and then exchanged between the 6 blocks as (value, tag) words in global memory; every
// block combines the 6 partials in block order, so all blocks (and all cards) hold bitwise the same totals.
#define TB 6
#define TNT 768            // team threads per block (the first 768 of the block), 24 warps
#define TNW (TNT / 32)
static_assert(TB * TNT == H && NTH >= TNT, "team layout: 6 blocks of 768 threads");
// barrier of the team threads only (the other threads of a larger block wait at the grid barrier)
#if NTH == TNT
#define TSYNC() __syncthreads()
#else
#define TSYNC() asm volatile("bar.sync 2, 768;" ::: "memory")
#endif
__device__ __forceinline__ uint2* trow(const Ctx& c, int rid, int b, int tj) { return c.tred + (long)((rid * MAXB + b) * TB + tj) * 32; }
// hardware conversions (tools/ptx/t5.cu, t6.cu: equal to the model's E2M1 rounding / to __nv_fp8_e4m3 on 2^22 / 2^24 values)
__device__ __forceinline__ float rt_e2m1(float v) {    // nearest E2M1 value (ties to the even code, saturating), signed
  uint32_t r;
  asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(r) : "f"(v));
  return __half2float(__ushort_as_half((unsigned short)r));
}
__device__ __forceinline__ unsigned enc_e4m3(float v) {
  unsigned short r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(r) : "f"(v));
  return r & 0xffu;
}
__device__ __forceinline__ float rt_e4m3(float v) {
  uint32_t r;
  asm("{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %1, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "f"(v));
  return __half2float(__ushort_as_half((unsigned short)r));
}
__device__ __forceinline__ float bfbits(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
// N (<= 4) per-thread values -> totals over the token's vector (sum, or max of non-negative values), in every thread
template <int N, bool MX>
__device__ __forceinline__ void tred(const Ctx& c, Smem& sm, int rid, int b, int tj, unsigned tag, float* v) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < N; i++) {
#pragma unroll
    for (int o = 16; o; o >>= 1) { const float t = __shfl_xor_sync(0xffffffff, v[i], o); v[i] = MX ? fmaxf(v[i], t) : v[i] + t; }
  }
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < N; i++) sm.red[w * 4 + i] = v[i];
  }
  TSYNC();
  if (w == 0) {
    float mine = 0.f;
#pragma unroll
    for (int i = 0; i < N; i++) {
      float x = (lane < TNW) ? sm.red[lane * 4 + i] : 0.f;
#pragma unroll
      for (int o = 16; o; o >>= 1) { const float t = __shfl_xor_sync(0xffffffff, x, o); x = MX ? fmaxf(x, t) : x + t; }
      if (lane == i) mine = x;
    }
    if (lane < N) put_gpu(trow(c, rid, b, tj) + lane, mine, tag);
    const int ci = lane >> 3, cj = lane & 7;   // lane = value * 8 + team block
    float r = 0.f;
    if (ci < N && cj < TB) {
      const uint2* p = trow(c, rid, b, cj) + ci;
      uint2 q = ld_gpu_v2(p);
      while (q.y != tag) q = ld_gpu_v2(p);
      r = __uint_as_float(q.x);
    }
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) { const float t = __shfl_xor_sync(0xffffffff, r, o); r = MX ? fmaxf(r, t) : r + t; }
    if (cj == 0 && ci < N) sm.tt[ci] = r;
  }
  TSYNC();
#pragma unroll
  for (int i = 0; i < N; i++) v[i] = sm.tt[i];
}
// dots of mix sources [s0, s1) with NQ queries over the token's vector: warp p = (s - s0) * NQ + qi takes one pair
// (at most 24 pairs). Totals to sm.tdot[s * 4 + qi]; visible to the other warps after the next __syncthreads.
template <int NQ>
__device__ __forceinline__ void tdots(const Ctx& c, Smem& sm, int rid, int b, int tj, unsigned tag, int nb, int s0, int s1, const float* q) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int V = (s1 - s0) * NQ;
  const float qs = 0.014731391274719739f;  // 4608 ** -0.5 rounded to fp32
  if (w >= V) return;
  const int s = s0 + w / NQ, qi = w % NQ;
  const bf16* src = msrc(sm, nb, s);
  const float* qp = q + qi * H + tj * TNT;
  float a = 0.f;
#pragma unroll 4
  for (int e = lane; e < TNT; e += 32) a += bf(src[e]) * (qp[e] * qs);
#pragma unroll
  for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
  if (lane == 0) put_gpu(trow(c, rid, b, tj) + w, a, tag);
  float r = 0.f;
  if (lane < TB) {
    const uint2* p = trow(c, rid, b, lane) + w;
    uint2 qq = ld_gpu_v2(p);
    while (qq.y != tag) qq = ld_gpu_v2(p);
    r = __uint_as_float(qq.x);
  }
#pragma unroll
  for (int o = 1; o < 8; o <<= 1) r += __shfl_xor_sync(0xffffffff, r, o);
  if (lane == 0) sm.tdot[s * 4 + qi] = r;
}
// softmax over the n mix sources (warp 0, lane = source). Source np is the one whose statistics v = (sum of squares,
// dots) were just reduced; finished blocks s < nb use their cached 1/rms (sm.fiv), the old partial at a boundary sm.piv.
template <int NQ>
__device__ __forceinline__ void tmixfinish(Smem& sm, int n, int nb, int np, const float* v, bool cache_piv) {
  if (threadIdx.x < 32) {
    const int s = threadIdx.x;
    const bool on = s < n;
    float iv = 0.f;
    if (on) iv = (s == np) ? rsqrtf(v[0] / (float)H + 1e-6f) : (s < nb ? sm.fiv[s] : sm.piv);
    if (on) sm.minv[s] = iv;
    if (cache_piv && s == np) sm.piv = iv;
#pragma unroll
    for (int qi = 0; qi < NQ; qi++) {
      float lg = on ? ((s == np) ? v[1 + qi] : sm.tdot[s * 4 + qi]) * iv : -INFINITY;
      float m = lg;
#pragma unroll
      for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
      const float pr = on ? expf(lg - m) : 0.f;
      float z = 0.f;
      for (int q = 0; q < n; q++) z += __shfl_sync(0xffffffff, pr, q);
      if (on) sm.mixp[qi * 10 + s] = pr / z;
    }
  }
  TSYNC();
}
// all-reduce of the four cards' partials of element k (bf16-rounded); spins until the four tags are current
__device__ __forceinline__ float t_raw(const uint2 (*mb)[MAXB][H], int b, unsigned tag, int k) {
  uint2 r0 = ld_relaxed_sys_v2(&mb[0][b][k]), r1 = ld_relaxed_sys_v2(&mb[1][b][k]);
  uint2 r2 = ld_relaxed_sys_v2(&mb[2][b][k]), r3 = ld_relaxed_sys_v2(&mb[3][b][k]);
  while (r0.y != tag || r1.y != tag || r2.y != tag || r3.y != tag) {
    __nanosleep(LL_BACKOFF);
    r0 = ld_relaxed_sys_v2(&mb[0][b][k]); r1 = ld_relaxed_sys_v2(&mb[1][b][k]);
    r2 = ld_relaxed_sys_v2(&mb[2][b][k]); r3 = ld_relaxed_sys_v2(&mb[3][b][k]);
  }
  return rbf(((__uint_as_float(r0.x) + __uint_as_float(r1.x)) + __uint_as_float(r2.x)) + __uint_as_float(r3.x));
}
// short conv of element k: hist [3][H] (oldest first), taps w4 [H][4]; writes the shifted history to hw
__device__ __forceinline__ float t_conv(const bf16* __restrict__ hist, float raw, const bf16* w4, bf16* __restrict__ hw, int k) {
  const unsigned short* hb = reinterpret_cast<const unsigned short*>(hist);
  const unsigned short b0 = __ldcg(hb + k), b1 = __ldcg(hb + H + k), b2 = __ldcg(hb + 2 * H + k);
  const uint2 u = *reinterpret_cast<const uint2*>(w4 + k * 4);
  const __nv_bfloat162* t2 = reinterpret_cast<const __nv_bfloat162*>(&u);
  const float2 ta = __bfloat1622float2(t2[0]), tb = __bfloat1622float2(t2[1]);
  const float wt[4] = {ta.x, ta.y, tb.x, tb.y};
  const float out = sconv4f(bfbits(b0), bfbits(b1), bfbits(b2), raw, wt);
  unsigned short* hwb = reinterpret_cast<unsigned short*>(hw);
  hwb[k] = b1; hwb[H + k] = b2; hw[2 * H + k] = tobf(raw);
  return out;
}
// NVFP4 input quantization of one element (four over six; a block of 16 = 16 consecutive lanes). Returns code x scale.
__device__ __forceinline__ float rt_e2m1c(float v, unsigned* code) {   // as rt_e2m1, and the 4-bit code
  uint32_t r, cb;
  asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %2; cvt.rn.f16x2.e2m1x2 %0, t; cvt.u32.u8 %1, t; }" : "=r"(r), "=r"(cb) : "f"(v));
  *code = cb & 15u;
  return __half2float(__ushort_as_half((unsigned short)r));
}
__device__ __forceinline__ float t_quant_nvfp4(float x, float amax, float* outer_out, unsigned* nib_out = nullptr, unsigned* scb_out = nullptr) {
  const float FMAX = 3.4028234663852886e38f;
  const float encoder = fminf(1536.0f / amax, FMAX), decoder = 1.0f / encoder, outer = amax * (1.0f / 1536.0f);
  float bm = fabsf(x);
#pragma unroll
  for (int o = 8; o; o >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffff, bm, o));
  float vals[2], er[2], scs[2];
  unsigned nibs[2];
#pragma unroll
  for (int cand = 0; cand < 2; cand++) {
    const float inv_max = cand == 0 ? (1.0f / 6.0f) : 0.25f;
    const float sc = rt_e4m3(fminf(bm * (encoder * inv_max), 448.0f));
    const float iv = fminf(1.0f / (sc * decoder), FMAX);
    float val = x * iv;
    if (val != val) val = 0.f;
    scs[cand] = sc;
    vals[cand] = rt_e2m1c(val, &nibs[cand]) * sc;
    float err = fabsf(vals[cand] * outer - x);
#pragma unroll
    for (int o = 8; o; o >>= 1) err += __shfl_xor_sync(0xffffffff, err, o);
    er[cand] = err;
  }
  *outer_out = outer;
  const bool c1 = er[1] < er[0];   // a strictly better /4 wins
  if (nib_out) { *nib_out = c1 ? nibs[1] : nibs[0]; *scb_out = enc_e4m3(c1 ? scs[1] : scs[0]); }
  return c1 ? vals[1] : vals[0];
}
// MXFP8 input quantization of one element (a block of 32 = the warp): code byte and the block's UE8M0 scale
__device__ __forceinline__ unsigned t_quant_mxfp8(float x, float* scale_out) {
  float m = fabsf(x);
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
  const float raw = m * (1.0f / 448.0f);
  const unsigned bits = __float_as_uint(raw);
  int ex = (int)(((bits + 0x007FFFFFu) & 0x7F800000u) >> 23);
  if (raw <= 5.877471754111438e-39f) ex = 0;
  if (ex > 254) ex = 254;
  const float inverse = (ex == 0) ? 1.7014118346046923e38f : 1.0f / ue8m0((unsigned)ex);
  *scale_out = ue8m0((unsigned)ex);
  return enc_e4m3(fminf(fmaxf(x * inverse, -448.f), 448.f));
}
#define TMT(id) do { if (c.tprof && i == 20 && threadIdx.x == 0) c.tprof[49 * 64 + 1184 + blockIdx.x * 16 + (id)] = gtime(); } while (0)
// mix of the sources with the fresh statistics v of source nb (value f of this thread), NQ rms norms, fragment writes
template <int NQ>
__device__ __forceinline__ void team_mix_norm(const Ctx& c, Smem& sm, int rid, int b, int tj, unsigned tag, int nb, float f,
                                              const float* v, const bf16* const* wn, bf16* frag, int fstride) {
  const int tid = threadIdx.x, k = tj * TNT + tid;
  tmixfinish<NQ>(sm, nb + 1, nb, nb, v, true);
  float o[NQ];
#pragma unroll
  for (int qi = 0; qi < NQ; qi++) o[qi] = 0.f;
  for (int s = 0; s < nb; s++) {
    const float fs = bf(sm.blk[s][tid]) * sm.minv[s];
#pragma unroll
    for (int qi = 0; qi < NQ; qi++) o[qi] += sm.mixp[qi * 10 + s] * fs;
  }
  {
    const float fs = f * sm.minv[nb];
#pragma unroll
    for (int qi = 0; qi < NQ; qi++) o[qi] += sm.mixp[qi * 10 + nb] * fs;
  }
  float ss[NQ];
#pragma unroll
  for (int qi = 0; qi < NQ; qi++) ss[qi] = o[qi] * o[qi];
  tred<NQ, false>(c, sm, rid, b, tj, tag, ss);
#pragma unroll
  for (int qi = 0; qi < NQ; qi++) {
    const float r = rsqrtf(ss[qi] / (float)H + 1e-6f);
    frag[qi * fstride + fidx16(b, k)] = tobf((o[qi] * r) * bf(wn[qi][k]));
  }
}
// attention input of layer i for token b; every thread of team block tj
__device__ void team_attn_in(const Ctx& c, int i, int b, int tj, Smem& sm) {
  const LayerW& L = c.L[i];
  const int tid = threadIdx.x, k = tj * TNT + tid;
  const unsigned tag = c.h.tag0 + i + 1;
  const float qs = 0.014731391274719739f;
  bf16* part = sm.blk[8];
  if (i == 0) {
    const unsigned mode = c.h.pad1 >> 16;
    const long tok = c.h.tokens[b];
    float y, ss[1];
    if (mode == 2) {   // track B: the normed embedding and the layer-0 projection rows of this token id from the tables
      y = bf(c.embed[tok * H + k]);
      if (tj == 0 && tid < 288) {
        const uint4 v = __ldcg(reinterpret_cast<const uint4*>(c.l0tab + tok * 2304) + tid);
        reinterpret_cast<uint4*>(c.qkvo + b * 3840)[tid < 192 ? tid : tid + 192] = v;   // rows 0..1535 and 3072..3839
      }
    } else {
      const float u = bf(c.embed[tok * H + k]);
      ss[0] = u * u;
      tred<1, false>(c, sm, 0, b, tj, tag, ss);
      y = rbf((u * rsqrtf(ss[0] / (float)H + 1e-6f)) * bf(c.embed_norm[k]));
      if (mode == 1) c.l0emb[tok * H + k] = tobf(y);
    }
    part[tid] = tobf(y);
    ss[0] = y * y;
    tred<1, false>(c, sm, 1, b, tj, tag, ss);
    const float r = rsqrtf(ss[0] / (float)H + 1e-6f);
    if (tid == 0) sm.piv = r;
    c.xfrag[fidx16(b, k)] = tobf((y * r) * bf(L.norm[0][k]));
    return;
  }
  const int lp = i - 1;
  const int nb_before = (lp == 0) ? 0 : ((lp - 1) / 6 + 1);
  const bool boundary = (lp % 6) == 0;
  const int nb = nb_before + (boundary ? 1 : 0);
  if (boundary) {   // the partial becomes a finished block (its 1/rms is the one computed when it was last mixed)
    sm.blk[nb_before][tid] = part[tid];
    if (tid == 0) sm.fiv[nb_before] = sm.piv;
    TSYNC();
  }
  // before the cross-card wait: the finished blocks' dots with this layer's queries
  if (L.glob) tdots<3>(c, sm, 0, b, tj, tag, nb, 0, nb, L.aq);
  else tdots<1>(c, sm, 0, b, tj, tag, nb, 0, nb, L.aq);
  TM(i, 1);
  const float rw = t_raw(c.comm[c.h.card]->llm[lp & 1], b, c.h.tag0 + lp + 1, k);
  const float mlp = t_conv(c.hist_mlp + ((((long)c.h.parity * NL + lp) * MAXB + b) * 3) * H, rw, c.L[lp].mlp_sc,
                           c.hist_mlp + ((((long)(c.h.parity ^ 1) * NL + lp) * MAXB + b) * 3) * H, k);
  if (c.dbg) c.dbg[((long)(lp * 2 + 1) * MAXB + b) * H + k] = tobf(mlp);
  const float at = bf(sm.ats[tid]), pv = bf(part[tid]);
  const float f = rbf(boundary ? (at + mlp) : ((pv + at) + mlp));
  part[tid] = tobf(f);
  TM(i, 16);
  TM(i, 17);
  if (L.glob) {
    float v[4] = {f * f, f * (L.aq[k] * qs), f * (L.aq[H + k] * qs), f * (L.aq[2 * H + k] * qs)};
    tred<4, false>(c, sm, 1, b, tj, tag, v);
    TM(i, 18);
    const bf16* wn[3] = {L.norm[0], L.norm[1], L.norm[2]};
    team_mix_norm<3>(c, sm, 2, b, tj, tag, nb, f, v, wn, c.xfrag, 72 * 32 * 16);
  } else {
    float v[2] = {f * f, f * (L.aq[k] * qs)};
    tred<2, false>(c, sm, 1, b, tj, tag, v);
    TM(i, 18);
    const bf16* wn[1] = {L.norm[0]};
    team_mix_norm<1>(c, sm, 2, b, tj, tag, nb, f, v, wn, c.xfrag, 0);
  }
}
// post-attention of layer i for token b: attention all-reduce + sconv, mlp input mix, norm, input quantization
// w13 input of one element for the tensor-core path:
// E2M1 codes K-major [k/64][(k%64)/32][token][16 B] (two codes per byte), block scales as UE4M3 bytes
// [token][k/64][4] at +40960. Returns the token's outer scale.
// (inlined: it is only called in the tensor-core kernel, whose team phase has registers to spare; out of line it
//  cost that kernel 12 us per step at B=8, replay 5346 -> 5334 us, e0215)
__device__ __forceinline__ float tc_quant13(const Ctx& c, int b, int k, float x, float amax) {
  const int lane = threadIdx.x & 31;
  float outer; unsigned nib, scb;
  t_quant_nvfp4(x, amax, &outer, &nib, &scb);
  const unsigned nn = __shfl_xor_sync(0xffffffff, nib, 1);
  if (!(lane & 1)) c.w13frag[(k >> 6) * 256 + ((k >> 5) & 1) * 128 + b * 16 + (lane >> 1)] = (uint8_t)(nib | (nn << 4));
  if (!(lane & 15)) c.w13frag[40960 + (b * 72 + (k >> 6)) * 4 + ((k >> 4) & 3)] = (uint8_t)scb;
  return outer;
}
template <bool TC>
__device__ void team_post(const Ctx& c, int i, int b, int tj, Smem& sm) {
  const LayerW& L = c.L[i];
  const int tid = threadIdx.x, lane = tid & 31, k = tj * TNT + tid;
  const unsigned tag = c.h.tag0 + i + 1;
  const float qs = 0.014731391274719739f;
  const bf16* part = sm.blk[8];
  const int nb = (i == 0) ? 0 : ((i - 1) / 6 + 1);
  const bool boundary = (i % 6) == 0;
  TMT(0);
  tdots<1>(c, sm, 3, b, tj, tag, nb, 0, boundary ? nb + 1 : nb, L.mq);   // finished blocks (and the old partial at a boundary)
  TMT(1);
  TM(i, 6);
  const float rw = t_raw(c.comm[c.h.card]->lla[i & 1], b, tag, k);
  TMT(2);
  const float at = t_conv(c.hist_attn + ((((long)c.h.parity * NL + i) * MAXB + b) * 3) * H, rw, L.attn_sc,
                          c.hist_attn + ((((long)(c.h.parity ^ 1) * NL + i) * MAXB + b) * 3) * H, k);
  sm.ats[tid] = tobf(at);
  if (c.dbg) c.dbg[((long)(i * 2) * MAXB + b) * H + k] = tobf(at);
  TMT(3);
  TM(i, 20);
  const float pvf = bf(part[tid]);
  const float f = boundary ? at : (pvf + at);     // the source whose statistics are new: attention out, or partial + attention out
  float v[2] = {f * f, f * (L.mq[k] * qs)};
  tred<2, false>(c, sm, 4, b, tj, tag, v);
  TMT(4);
  const int np = boundary ? nb + 1 : nb;
  tmixfinish<1>(sm, np + 1, nb, np, v, false);
  float o = 0.f;
  for (int s = 0; s < nb; s++) o += sm.mixp[s] * (bf(sm.blk[s][tid]) * sm.minv[s]);
  if (boundary) o += sm.mixp[nb] * (pvf * sm.minv[nb]);
  o += sm.mixp[np] * (f * sm.minv[np]);
  TMT(5);
  TM(i, 21);
  float ss[1] = {o * o};
  tred<1, false>(c, sm, 5, b, tj, tag, ss);
  const float x = rbf((o * rsqrtf(ss[0] / (float)H + 1e-6f)) * bf(L.mlp_norm[k]));
  TMT(6);
  TM(i, 22);
  c.x2frag[fidx16(b, k)] = tobf(x);
  if (!L.mx) {
    float am[1] = {fabsf(x)};
    tred<1, true>(c, sm, 6, b, tj, tag, am);
    float outer;
#if TC13
    if (TC) outer = tc_quant13(c, b, k, x, am[0]);
    else
#endif
    {
    const float qv = t_quant_nvfp4(x, am[0], &outer);
    reinterpret_cast<__half*>(c.w13frag)[fidx16(b, k)] = __float2half_rn(qv);
    }
    if (tj == 0 && tid == 0) c.w13sc[b] = outer;
  } else {
    float sc;
    const unsigned code = t_quant_mxfp8(x, &sc);
    c.w13frag[fidx8(b, k)] = (uint8_t)code;
    if (lane == 0) c.w13sc[(k >> 5) * MAXB + b] = sc;
  }
  TMT(7);
}
// final: mlp all-reduce of layer 47 + finalize + output mix + norm -> hfrag
__device__ void team_final(const Ctx& c, int b, int tj, Smem& sm) {
  const int tid = threadIdx.x, k = tj * TNT + tid;
  const unsigned tag = c.h.tag0 + NL;   // rids 7-9 only
  const float qs = 0.014731391274719739f;
  bf16* part = sm.blk[8];
  const int lp = NL - 1;
  tdots<1>(c, sm, 7, b, tj, tag, 8, 0, 8, c.oq);
  const float rw = t_raw(c.comm[c.h.card]->llm[lp & 1], b, c.h.tag0 + lp + 1, k);
  const float mlp = t_conv(c.hist_mlp + ((((long)c.h.parity * NL + lp) * MAXB + b) * 3) * H, rw, c.L[lp].mlp_sc,
                           c.hist_mlp + ((((long)(c.h.parity ^ 1) * NL + lp) * MAXB + b) * 3) * H, k);
  if (c.dbg) c.dbg[((long)(lp * 2 + 1) * MAXB + b) * H + k] = tobf(mlp);
  const float f = rbf((bf(part[tid]) + bf(sm.ats[tid])) + mlp);   // layer 47 is not a boundary
  part[tid] = tobf(f);
  float v[2] = {f * f, f * (c.oq[k] * qs)};
  tred<2, false>(c, sm, 8, b, tj, tag, v);
  const bf16* wn[1] = {c.fnorm};
  team_mix_norm<1>(c, sm, 9, b, tj, tag, 8, f, v, wn, c.hfrag, 0);
}

// w2 input of one routed (union expert ui, token b) pair: one element per thread, the global amax from the four cards
template <bool TC>
__device__ PHASE_NI void phase_quant2_pair(const Ctx& c, int i, int ui, int b, Smem& sm) {
  const LayerW& L = c.L[i];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const unsigned tag = c.h.tag0 + i + 1;
  float x = 0.f;
  if (tid < ILOC) {   // this card's 576 activation values of the pair, as the w13 epilogues publish them
    const uint4* p = reinterpret_cast<const uint4*>(c.act) + (ui * MAXB + b) * ILOC + tid;
    // (relaxed loads: with ld.global.cg the tag matched but the values were wrong, e0064-e0067)
    uint4 q = ld_gpu_v4(p);
    while (q.y != tag || q.w != tag) q = ld_gpu_v4(p);
    x = rbf(act_fn(__uint_as_float(q.x), __uint_as_float(q.z)) * sm.wtu[ui * MAXB + b]);
  }
  if (!L.mx) {
    // per-token amax over the expert's whole intermediate vector: this card's maximum to every card, then the four
    float m = fabsf(x);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) sm.red[w] = m;
    __syncthreads();
    if (w == 0) {
      float lm = (lane < NWARP) ? sm.red[lane] : 0.f;
#pragma unroll
      for (int o = 16; o; o >>= 1) lm = fmaxf(lm, __shfl_xor_sync(0xffffffff, lm, o));
      float am = 0.f;
      if (lane < NCARD) {
        ll_put(&c.comm[lane]->llx[i][c.h.card][ui][b], lm, tag);
        am = ll_get1(&c.comm[c.h.card]->llx[i][lane][ui][b], tag);
      }
      am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 1));
      am = fmaxf(am, __shfl_xor_sync(0xffffffff, am, 2));
      if (lane == 0) sm.misc[0] = am;
    }
    __syncthreads();
    if (tid >= ILOC) return;   // 18 whole warps
    float outer;
#if TC2
    if (TC) {
    // tensor-core operand of union expert ui: codes [k/64][(k%64)/32][token][16 B], scale bytes [token][k/64][4] at +4096
    unsigned nib, scb;
    t_quant_nvfp4(x, sm.misc[0], &outer, &nib, &scb);
    const unsigned nn = __shfl_xor_sync(0xffffffff, nib, 1);
    uint8_t* tb = c.w2frag + (long)ui * 9216;
    if (!(lane & 1)) tb[(tid >> 6) * 256 + ((tid >> 5) & 1) * 128 + b * 16 + (lane >> 1)] = (uint8_t)(nib | (nn << 4));
    if (!(lane & 15)) tb[4096 + (b * 9 + (tid >> 6)) * 4 + ((tid >> 4) & 3)] = (uint8_t)scb;
    } else
#endif
    {
    const float qv = t_quant_nvfp4(x, sm.misc[0], &outer);
    reinterpret_cast<__half*>(c.w2frag + (long)ui * 9 * 32 * 32)[fidx16(b, tid)] = __float2half_rn(qv);
    }
    if (tid == 0) c.w2sc[ui * 18 * MAXB + b] = outer;
  } else {
    if (tid >= ILOC) return;
    float sc;
    const unsigned code = t_quant_mxfp8(x, &sc);
    (c.w2frag + (long)ui * 9 * 32 * 16)[fidx8(b, tid)] = (uint8_t)code;
    if (lane == 0) c.w2sc[(ui * 18 + (tid >> 5)) * MAXB + b] = sc;
  }
}
// zero the fragment columns and scales of the tokens that are not routed to union expert ui (a stale fragment of
// the other format could hold Inf)
template <bool TC>
__device__ PHASE_NI void phase_quant2_zero(const Ctx& c, int i, int ui, Smem& sm) {
  const LayerW& L = c.L[i];
  const int tid = threadIdx.x;
  const unsigned m = sm.rmask[ui];
  for (int b = 0; b < c.h.B; b++) {
    if ((m >> b) & 1u) continue;
    if (tid < 18) c.w2sc[(ui * 18 + tid) * MAXB + b] = 0.f;
    if (tid < ILOC) {
      if (L.mx) (c.w2frag + (long)ui * 9 * 32 * 16)[fidx8(b, tid)] = 0;
#if TC2
      else if (TC) { if (tid < 36) (c.w2frag + (long)ui * 9216 + 4096)[b * 36 + tid] = 0; }   // zero block scales: the column contributes 0
#endif
      else reinterpret_cast<__half*>(c.w2frag + (long)ui * 9 * 32 * 32)[fidx16(b, tid)] = __float2half_rn(0.f);
    }
  }
}
// L2 prefetch of this layer's K and V rows (this card's 2 heads, every token, the positions attention will read):
// thread it < B * 4 takes (token, head, k|v); every block its share of the rows
__device__ __forceinline__ void prefetch_kv(const Ctx& c, int i, int part, int nparts, int pw = 0) {
  const int t = c.h.pos;
  const int lo = c.L[i].glob ? 0 : max(0, t + 1 - WIN);
  const int rows = t - lo;   // position t is written this step
  const int it = (int)threadIdx.x - pw * 32;
  if (rows <= 0 || it < 0 || it >= c.h.B * NKVL * 2) return;
  const int bh = it >> 1;
  const long off = (((long)(i * MAXB + bh / NKVL) * NKVL + bh % NKVL) * c.maxctx + lo) * HD;
  const unsigned bytes = (unsigned)rows * HD * 2;
  const bf16* base = ((it & 1) ? c.vc : c.kc) + off;
  if (bytes / nparts <= 32768u) prefetch_items(base, bytes, part, nparts);
}
// B=1: combine the attention chunks of all 12 local heads (every block redundantly) into the compact o_proj
// fragment in shared memory (sm.fx[0]; the q/k/v fragments are dead after the qkv GEMV)
__device__ void phase_ocombine_red(const Ctx& c, int i, int nch, Smem& sm) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const bf16* qkv = c.qkvo;   // token 0
  const int OG = 1536, OS = 3264;
  // per head (warp w < 12): chunk factors exp(m_q - M), the sum L and lse
  if (w < NQL) {
    const int kh = w / 6, j = w % 6;
    const float* base = c.achunk + ((long)(kh * MAXCHUNK) * 6) * (2 + HD);
    float mq = (lane < nch) ? __ldcg(base + (lane * 6 + j) * (2 + HD)) : -INFINITY;
    float lq = (lane < nch) ? __ldcg(base + (lane * 6 + j) * (2 + HD) + 1) : 0.f;
    float M = mq;
    for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffff, M, o));
    float f = (lane < nch) ? expf(mq - M) : 0.f;
    float Ls = 0.f;
    for (int q = 0; q < nch; q++) Ls += __shfl_sync(0xffffffff, lq * f, q);
    if (lane < nch) sm.att[w * 32 + lane] = f;      // 12 x 32 floats (sm.att holds 6 x 128)
    if (lane == 0) { sm.misc[w * 2] = M + logf(Ls); sm.misc[w * 2 + 1] = Ls; }
  }
  __syncthreads();
  for (int idx = tid; idx < NQL * HD; idx += NTH) {
    const int hl = idx / HD, d = idx % HD, kh = hl / 6, j = hl % 6;
    const float* base = c.achunk + ((long)(kh * MAXCHUNK) * 6) * (2 + HD);
    float O = 0.f;
    for (int q = 0; q < nch; q++) O += __ldcg(base + (q * 6 + j) * (2 + HD) + 2 + d) * sm.att[hl * 32 + q];
    float out = O / sm.misc[hl * 2 + 1];
    out = out * sigm(bf(qkv[OG + hl * HD + d]));
    out = out * sigm(sm.misc[hl * 2] - bf(qkv[OS + hl]));
    put16<true>(&sm.fx[0][0], 0, hl * HD + d, out);
  }
  __syncthreads();
}
// L2 prefetch of a layer's small weights (scales, taps, norms, queries, router bias), one region per block
__device__ __forceinline__ void prefetch_small(const Ctx& c, int i, int part, int pw = 0) {
  const LayerW& L = c.L[i];
  const void* p = nullptr; long n = 0;
  switch (part) {
    case 0: p = L.qscale; n = NQL * HD * 2; break;
    case 1: p = L.rscale; n = NQL * DREL * 2; break;
    case 2: p = L.relp; n = DREL * RELEXT * 2; break;
    case 3: p = L.ksc; n = 256 * 4 * 2; break;
    case 4: p = L.vsc; n = 256 * 4 * 2; break;
    case 5: p = L.attn_sc; n = H * 4 * 2; break;
    case 6: p = L.mlp_sc; n = H * 4 * 2; break;
    case 7: p = L.mq; n = H * 4; break;
    case 8: p = L.mlp_norm; n = H * 2; break;
    case 9: p = L.rbias; n = NE * 4; break;
    case 10: p = L.f13; n = NE * 4; break;
    case 11: p = L.f2; n = NE * 4; break;
    case 12: if (i + 1 < NL) { p = c.L[i + 1].aq; n = (c.L[i + 1].glob ? 3 : 1) * H * 4; } break;
    case 13: if (i + 1 < NL) { p = c.L[i + 1].norm[0]; n = H * 2; } break;
    case 14: if (i + 1 < NL && c.L[i + 1].glob) { p = c.L[i + 1].norm[1]; n = H * 2; } break;
    case 15: if (i + 1 < NL && c.L[i + 1].glob) { p = c.L[i + 1].norm[2]; n = H * 2; } break;
    default: break;
  }
  if (p && n >= 16) prefetch_l2(p, n, 0, 1, pw);
}
// ------------------------------------------------------------------------------------------------ the step kernel
// ------------------------------------------------------------------------------------------------ tcgen05 expert GEMV
// Row tasks of 128 rows (w13: (union expert, 64 gate + 64 up rows), 72 units; w2: 128 output rows over all union
// experts, 9 units per expert). The grid's units are split evenly over the blocks; a block cuts its range into
// sub-items (<= SUBL units of one expert) and deals them to its TC_NC streams. A stream's producer thread copies
// units from global memory into its ring (cp.async.bulk), its consumer thread issues one scale-factor copy and one
// MMA per unit into one of its two accumulators. Warps 0-3 read the accumulators in sub-item order (D x column
// scale summed in registers), and at the end of a row task either run the epilogue or, if the task began in an
// earlier block, hand the partial to that block as tagged words (it adds the partials in block order).
#ifdef TC_DEBUG   // a wait that does not end leaves a code in host-mapped memory and goes on (tools/t_tcdbg.py)
#define TC_FAIL(code, det) do { int* dp_ = c.h.gen_out; if (dp_ && dp_[64 + c.h.card * 8] == 0) { \
    dp_[64 + c.h.card * 8 + 1] = (det); dp_[64 + c.h.card * 8] = (code) | (blockIdx.x << 8) | (layer << 16) | ((unsigned)W2 << 24); } } while (0)
#define TC_WAIT(addr, par, code, det) do { unsigned n_ = 0; while (!mbar_try(addr, par)) { if (++n_ > (1u << 21)) { TC_FAIL(code, det); break; } } } while (0)
#define TC_MARK(k, v) do { if (c.h.gen_out && blockIdx.x == 0 && threadIdx.x == 0) c.h.gen_out[128 + c.h.card * 8 + (k)] = (v); } while (0)
#define TC_T(k) do { if (c.h.gen_out && layer == 20 && c.h.card == 0 && (blockIdx.x == 0 || blockIdx.x == 77)) \
    c.h.gen_out[256 + (int)W2 * 64 + (blockIdx.x ? 32 : 0) + (k)] = (int)(gtime() - tc_t0); } while (0)
#else
#define TC_T(k)
#define TC_WAIT(addr, par, code, det) mbar_wait(addr, par)
#define TC_MARK(k, v)
#endif
// one stream of a block's tensor-core GEMV: producer (even warps from 4) or consumer (odd warps) of stream cs.
// The whole warp runs the loop and lane 0 issues.
template <bool W2>
__device__ __noinline__ void tc_stream(const Ctx& c, Smem& sm, int layer, int nsub) {
  const int w = threadIdx.x >> 5, cs = (w - 4) >> 1;
  const bool lead = (threadIdx.x & 31) == 0;
  const uint32_t bar = sh(sm.tcbar);
  const uint32_t b_full = bar, b_empty = bar + 64, b_done = bar + 128, b_ready = bar + 192, b_b = bar + 256;
  const uint32_t t = sm.tct, t_sfa = t + 64, t_sfb = t + 96;
    const uint32_t full = b_full + 16 * cs, empty = b_empty + 16 * cs;
    const uint32_t ring = sh(sm.stage) + cs * (TC_S * TC_G * TC_UB);
    unsigned st = 0;
    const unsigned pf[2] = {sm.tcph[cs * 2], sm.tcph[cs * 2 + 1]}, pe[2] = {sm.tcph[8 + cs * 2], sm.tcph[8 + cs * 2 + 1]};
    if (!(w & 1)) {                                           // producer
      const uint64_t pol = tc_policy();
      for (int q = cs; q < nsub; q += TC_NC) {
        const int seq = q >> 2, par = seq & 1;
        const TcSub e = sm.tctab[q];
        const uint8_t* src = e.src;
        int n = e.n;
        bool first = W2;
        if (W2 && seq >= 2) TC_WAIT(b_ready + 8 * (cs * 2 + par), (sm.tcph[24 + cs * 2 + par] + (seq >> 1)) & 1, 1, q);   // the B slot's previous MMAs are done
        while (n > 0) {
          const int g = min(TC_G, n);
          const unsigned slot = st & 1, r = st >> 1;
          if (r) TC_WAIT(empty + 8 * slot, (pe[slot] + r - 1) & 1, 2, q | (st << 8));
          if (lead) {
            mbar_expect(full + 8 * slot, g * TC_UB + (first ? 2304 : 0));
            tc_bulk_w(ring + slot * (TC_G * TC_UB), src, g * TC_UB, full + 8 * slot, pol);
            if (first) tc_bulk(sh(sm.tcB) + (cs * 2 + par) * 2304, c.w2frag + (long)e.item * 9216, 2304, full + 8 * slot);
          }
          first = false;
          src += g * TC_UB; n -= g; st++;
        }
      }
      return;
    }
    if (lead) tc_fence_after();                               // consumer
    if (!W2) TC_WAIT(b_b, sm.tcph[32] & 1, 3, 0);
    const uint64_t hiA = tc_sdesc(0, 128, 256), hiS = tc_sdesc(0, 0, 128);
    const uint32_t tsa = t_sfa + 4 * cs;
    const unsigned pr[2] = {sm.tcph[24 + cs * 2], sm.tcph[24 + cs * 2 + 1]};
    for (int q = cs; q < nsub; q += TC_NC) {
      const int seq = q >> 2, par = seq & 1, a = cs * 2 + par;
      TC_WAIT(b_ready + 8 * a, (pr[par] + (seq >> 1)) & 1, 4, q);             // accumulator read out, B scale factors in tensor memory
      if (lead) tc_fence_after();
      const TcSub e = sm.tctab[q];
      int n = e.n;
      const uint32_t td = t + 8 * a;
      uint64_t dB = hiA | (uint64_t)(((sh(sm.tcB) + (W2 ? a * 2304 : 0)) >> 4) + e.ks0 * 16);
      uint32_t tsfb = t_sfb + 4 * ((W2 ? a * 9 : 0) + e.ks0);
      uint32_t accf = 0;
      while (n > 0) {
        const int g = min(TC_G, n);
        const unsigned slot = st & 1, r = st >> 1;
        TC_WAIT(full + 8 * slot, (pf[slot] + r) & 1, 5, q | (st << 8));
        uint32_t ad = (ring + slot * (TC_G * TC_UB)) >> 4;
        for (int j = 0; j < g; j++) {
          if (lead) {
            tc_cp(tsa, hiS | (uint64_t)(ad + 256));
            tc_mma4(td, hiA | (uint64_t)ad, dB, TC_IDESC4, accf, tsa, tsfb);
          }
          accf = 1; ad += TC_UB >> 4; dB += 16; tsfb += 4;
        }
        if (lead) tc_commit(empty + 8 * slot);
#ifdef TC_DEBUG
        if (lead && cs == 0 && st == 0 && c.h.gen_out && layer == 20 && c.h.card == 0 && blockIdx.x == 0) c.h.gen_out[256 + (int)W2 * 64 + 8] = (int)(gtime() & 0x7fffffff);
#endif
        n -= g; st++;
      }
      if (lead) tc_commit(b_done + 8 * a);
    }
    // every commit has arrived before the next GEMV; the phases this stream's ring barriers went through
    if (st >= 1) { const unsigned l = st - 1; TC_WAIT(empty + 8 * (l & 1), (pe[l & 1] + (l >> 1)) & 1, 6, st); }
    if (st >= 2) { const unsigned l = st - 2; TC_WAIT(empty + 8 * (l & 1), (pe[l & 1] + (l >> 1)) & 1, 7, st); }
#ifdef TC_DEBUG
    if (lead && cs == 0 && c.h.gen_out && layer == 20 && c.h.card == 0 && blockIdx.x == 0) { c.h.gen_out[256 + (int)W2 * 64 + 9] = (int)(gtime() & 0x7fffffff); c.h.gen_out[256 + (int)W2 * 64 + 10] = (int)st; }
#endif
    if (lead) {
      sm.tcph[cs * 2] = pf[0] + ((st + 1) >> 1); sm.tcph[cs * 2 + 1] = pf[1] + (st >> 1);
      sm.tcph[8 + cs * 2] = pe[0] + ((st + 1) >> 1); sm.tcph[8 + cs * 2 + 1] = pe[1] + (st >> 1);
    }
    return;
}
// tensor memory and the mbarriers of a launch (out of line: the kernel's frame stays at its register cap)
__device__ __noinline__ void tc_begin(const Ctx& c, Smem& sm) {
  if (threadIdx.x < 32) tc_alloc(sh(&sm.tct), TC_COLS);     // tensor memory columns of this block
  if (threadIdx.x < 33) {
    // mbarriers: once per launch (initialising them again per GEMV lost arrivals); sm.tcph counts their phases
    mbar_init(sh(sm.tcbar) + 8 * threadIdx.x, 1); sm.tcph[threadIdx.x] = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  TC_PROG(0, c.h.tag0);
}
__device__ __noinline__ void tc_end(const Ctx& c, Smem& sm) { if (threadIdx.x < 32) tc_dealloc(sm.tct, TC_COLS); TC_PROG(5, c.h.tag0); }
template <bool W2>
__device__ __noinline__ void gemv_tc(const Ctx& c, Smem& sm, int layer, int nu) {
  const LayerW& L = c.L[layer];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const unsigned tag = c.h.tag0 + layer + 1;
  const unsigned NB = gridDim.x;
  const unsigned T = W2 ? 9u * (unsigned)nu : 72u, I = W2 ? 9u : 72u;   // units per row task, per item
  const unsigned U = (W2 ? 36u : 9u * (unsigned)nu) * T;
  const unsigned u0 = U * blockIdx.x / NB, u1 = U * (blockIdx.x + 1) / NB;
  // sub-items of <= 12 units of one expert, dealt to the streams in turn. Replay at B=8 (tools/t_replay.py, e0207):
  // 9 units 5399 us per step, 12: 5385, 24: 5424, 36: 5512 (fewer read-outs against equal work per stream)
  const unsigned SUBL = W2 ? 9u : 12u;
  const unsigned q0 = u0 / SUBL;
  const int nsub = (int)((u1 + SUBL - 1) / SUBL - q0);
  const uint32_t bar = sh(sm.tcbar);
  const uint32_t b_full = bar, b_empty = bar + 64, b_done = bar + 128, b_ready = bar + 192, b_b = bar + 256;
  TC_MARK(0, (layer + 1) | ((int)W2 << 8) | (nu << 16));
  TC_PROG(1, (layer + 1) | ((int)W2 << 8) | (nu << 16));
#ifdef TC_DEBUG
  const unsigned long long tc_t0 = gtime();
  if (tid == 0 && c.h.gen_out && layer == 20 && c.h.card == 0 && blockIdx.x == 0) c.h.gen_out[256 + (int)W2 * 64 + 11] = (int)(tc_t0 & 0x7fffffff);
#endif
  if (tid < nsub) {
    const unsigned a = max(u0, (q0 + tid) * SUBL), b = min(u1, (q0 + tid + 1) * SUBL);
    const unsigned task = a / T, r = a - task * T, it = r / I, ks0 = r - it * I;
    const int ui = W2 ? (int)it : (int)(task / 9u), rg = W2 ? (int)task : (int)(task - (task / 9u) * 9u);
    const int e = sm.uni[ui];
    TcSub su;
    su.src = (W2 ? L.w2t + ((long)(e * 36 + rg) * 9 + ks0) * TC_UB : L.w13t + ((long)(e * 9 + rg) * 72 + ks0) * TC_UB);
    su.n = (unsigned short)(b - a); su.ks0 = (unsigned short)ks0; su.item = ui; su.task = (int)task; su.pad = 0;
    sm.tctab[tid] = su;
    const float f = W2 ? L.f2[e] : L.f13[e];
#pragma unroll
    for (int col = 0; col < MAXB; col++) sm.tcs[tid * 8 + col] = (W2 ? c.w2sc[ui * 18 * MAXB + col] : c.w13sc[col]) * f;
  }
  __syncthreads();
  if (tid == 0) TC_T(0);
  const uint32_t t = sm.tct, t_sfa = t + 64, t_sfb = t + 96;   // D: 8 accumulators x 8 columns; SFA: 4 x 4; SFB: 72 x 4
  if (w >= 4) { if (((w - 4) >> 1) < min(TC_NC, nsub)) tc_stream<W2>(c, sm, layer, nsub); return; }
  // warps 0-3: scale factors of B into tensor memory (lane n: the four block scales of token n per K step), read-out
  const uint32_t la = (uint32_t)(w * 32) << 16;
  if (W2) {   // (w13: tc_prep13 did this beside the routing phase)
    for (int q = 0; q < min(nsub, 8); q++) {
      const uint32_t* sf = reinterpret_cast<const uint32_t*>(c.w2frag + (long)sm.tctab[q].item * 9216 + 4096) + (lane & 7) * 9;
      const int a = (q & 3) * 2 + (q >> 2);
      const uint4 zz = make_uint4(0, 0, 0, 0);
      tc_st32(t_sfb + 4 * (a * 9) + la, lane < 8 ? make_uint4(sf[0], sf[1], sf[2], sf[3]) : zz, lane < 8 ? make_uint4(sf[4], sf[5], sf[6], sf[7]) : zz);
      tc_st1(t_sfb + 4 * (a * 9 + 8) + la, lane < 8 ? sf[8] : 0u);
    }
    tc_wait_st(); tc_fence_before();
    asm volatile("bar.sync 3, 128;" ::: "memory");
  }
  TC_MARK(3, layer + 1);
  if (tid < 8) mbar_arrive(b_ready + 8 * tid);
  if (tid == 0) TC_T(1);
  TC_MARK(4, (layer + 1) | (nsub << 8));
  float P[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  uint4* const hand = reinterpret_cast<uint4*>(c.part) + (W2 ? 152 * 512 : 0);   // per block: 128 rows x 8 (value, tag) words
  for (int q = 0; q < nsub; q++) {
    const int cs = q & 3, seq = q >> 2, par = seq & 1, a = cs * 2 + par;
    TC_WAIT(b_done + 8 * a, (sm.tcph[16 + a] + (seq >> 1)) & 1, 8, q | (nsub << 8));
    tc_fence_after();
    float d[8];
    if (tid == 0 && q == 0) TC_T(2);
    if (tid == 0 && q == nsub - 1) TC_T(3);
    tc_ld8(t + 8 * a + la, d);
#pragma unroll
    for (int j = 0; j < 8; j++) P[j] = __fmaf_rn(d[j], sm.tcs[q * 8 + j], P[j]);
    if (q + 8 < nsub) {                                       // the slot's next sub-item may start
      if (W2) {
        const uint32_t* sf = reinterpret_cast<const uint32_t*>(c.w2frag + (long)sm.tctab[q + 8].item * 9216 + 4096) + (lane & 7) * 9;
        const uint4 zz = make_uint4(0, 0, 0, 0);
        tc_st32(t_sfb + 4 * (a * 9) + la, lane < 8 ? make_uint4(sf[0], sf[1], sf[2], sf[3]) : zz, lane < 8 ? make_uint4(sf[4], sf[5], sf[6], sf[7]) : zz);
        tc_st1(t_sfb + 4 * (a * 9 + 8) + la, lane < 8 ? sf[8] : 0u);
        tc_wait_st();
      }
      tc_fence_before();
      asm volatile("bar.sync 3, 128;" ::: "memory");
      if (tid == 0) mbar_arrive(b_ready + 8 * a);
    }
    const TcSub e = sm.tctab[q];
    if (q == nsub - 1 || sm.tctab[q + 1].task != e.task) {    // the block's last sub-item of this row task
      const unsigned tb = (unsigned)e.task * T, te = tb + T;
      if (tb < u0) {                                          // begun in an earlier block: hand the partial over
        uint4* slot = hand + (blockIdx.x * 128 + tid) * 4;
#pragma unroll
        for (int j = 0; j < 4; j++) st_l2(slot + j, make_uint4(__float_as_uint(P[2 * j]), tag, __float_as_uint(P[2 * j + 1]), tag));
      } else {
        if (te > u1) {                                        // continues in later blocks: add their partials in block order
          const unsigned ul = te - 1;
          unsigned bl = ul * NB / U;
          while (bl + 1 < NB && U * (bl + 1) / NB <= ul) bl++;
          for (unsigned bb = blockIdx.x + 1; bb <= bl; bb++) {
            const uint4* src = hand + (bb * 128 + tid) * 4;
#pragma unroll
            for (int j = 0; j < 4; j++) {
              uint4 pp = ld_gpu_v4(src + j);
#ifdef TC_DEBUG
              { unsigned n_ = 0; while (pp.y != tag || pp.w != tag) { pp = ld_gpu_v4(src + j); if (++n_ > (1u << 19)) { TC_FAIL(9, bb | (bl << 8) | (e.task << 16)); break; } } }
#else
              while (pp.y != tag || pp.w != tag) pp = ld_gpu_v4(src + j);
#endif
              P[2 * j] += __uint_as_float(pp.x); P[2 * j + 1] += __uint_as_float(pp.z);
            }
          }
        }
        if (W2) {
          const int row = e.task * 128 + tid;
          for (int col = 0; col < c.h.B; col++)
            for (int dd = 0; dd < NCARD; dd++) ll_put(&c.comm[dd]->llm[layer & 1][c.h.card][col][row], P[col], tag);
        } else {
          // rows 0-63 of a w13 row group: gate of intermediates rg * 64 + 0..63; rows 64-127: up. Each half of the
          // (gate, tag, up, tag) word of a routed (expert, token) pair is one 8-byte store.
          const int ui = e.item, rg = e.task - ui * 9;
          const unsigned rm = sm.rmask[ui];
          uint2* aw = reinterpret_cast<uint2*>(c.act) + ((long)(ui * MAXB) * ILOC + rg * 64 + (tid & 63)) * 2 + (tid >> 6);
          for (int col = 0; col < c.h.B; col++)
            if ((rm >> col) & 1u) put_l2(aw + (long)col * ILOC * 2, rbf(P[col]), tag);
        }
      }
#pragma unroll
      for (int j = 0; j < 8; j++) P[j] = 0.f;
    }
    TC_MARK(2, q + 1);
    if ((q & 7) == 7) TC_PROG(3, ((q + 1) << 8) | nsub);
  }
  if (tid == 0) { TC_T(4); if (c.h.gen_out && layer == 20 && c.h.card == 0 && (blockIdx.x == 0 || blockIdx.x == 77)) c.h.gen_out[256 + (int)W2 * 64 + (blockIdx.x ? 32 : 0) + 5] = nsub | (nu << 8); }
  // (all four warps are out of the loop before the phase counts move: the loop reads them)
  asm volatile("bar.sync 3, 128;" ::: "memory");
  if (tid < 8) {   // slot tid = (stream, parity): its sub-items; "ready" was signalled once at the start and once per re-use
    const int cs = tid >> 1, nseq = cs < nsub ? (nsub - cs + 3) >> 2 : 0, na = (nseq + 1 - (tid & 1)) >> 1;
    sm.tcph[16 + tid] += na; sm.tcph[24 + tid] += max(1, na);
  }
  if (!W2 && tid == 8) sm.tcph[32] += 1;
  TC_MARK(1, (layer + 1) | ((int)W2 << 8));
  TC_PROG(2, (layer + 1) | ((int)W2 << 8) | (nsub << 16));
}

template <bool RED, bool TC>
__device__ __forceinline__ void step_body(const Ctx& c, Smem& sm, int layer_lo, int layer_hi) {
#ifdef TC_BODY_CALLS
  if (TC) tc_begin(c, sm);
#endif
  if (RED) {
    for (int k = threadIdx.x; k < 144 * MAXB; k += NTH) sm.sc13[k] = 0.f;
    for (int k = threadIdx.x; k < 8 * 18 * MAXB; k += NTH) sm.sc2[k] = 0.f;
    __syncthreads();
  }
  unsigned nbar = 0;
  const unsigned barbase = *((volatile unsigned*)&c.bar[1]);
  const int B = c.h.B;
  const int nblk = gridDim.x;
  unsigned ev = 0;  // comm events of this launch (per counter kind)
  unsigned ev_attn = 0, ev_mlp = 0, ev_amax = 0;
  for (int i = layer_lo; i < layer_hi; i++) {
    const LayerW& L = c.L[i];
    // A: attention input
    TM(i, 0);
    if (RED) {
      prefetch_l2(L.wqkv, (long)L.qkv_strips * 72 * 2048, blockIdx.x, nblk);
      prefetch_kv(c, i, blockIdx.x, nblk);
      if (blockIdx.x >= nblk - 16) prefetch_small(c, i, nblk - 1 - blockIdx.x);
      phase_attn_in<true>(c, i, 0, sm, i > 0, ev_mlp);
      __syncthreads();   // the fragments of every warp are read by the next GEMV
    } else {
#ifdef NO_TEAM
      if (blockIdx.x < B) phase_attn_in<false>(c, i, blockIdx.x, sm, i > 0, ev_mlp);
      else {
        prefetch_l2(L.wqkv, (long)L.qkv_strips * 72 * 2048, blockIdx.x - B, nblk - B); prefetch_kv(c, i, blockIdx.x - B, nblk - B);
        if (blockIdx.x >= nblk - 16) prefetch_small(c, i, nblk - 1 - blockIdx.x);
      }
#else
      if (blockIdx.x < B * TB) { if (threadIdx.x < TNT) team_attn_in(c, i, blockIdx.x / TB, blockIdx.x % TB, sm); }
      else {
        prefetch_l2(L.wqkv + (long)L.qkv_skip * 72 * 2048, (long)(L.qkv_strips - L.qkv_skip) * 72 * 2048, blockIdx.x - B * TB, nblk - B * TB); prefetch_kv(c, i, blockIdx.x - B * TB, nblk - B * TB);
        if (blockIdx.x >= nblk - 16) prefetch_small(c, i, nblk - 1 - blockIdx.x);
      }
#endif
      grid_bar(c, nbar, barbase);
    }
    TM(i, 2);
    // attention geometry of this layer (needed already by the qkv GEMV)
    const int apos = c.h.pos, alo = L.glob ? 0 : max(0, apos + 1 - WIN);
    // Positions per chunk block (cs) and gate split (gs), decided together.
    // cs: the V sum is a per-thread loop over the chunk's positions, so smaller chunks are faster as long as every
    //     (token, KV head, chunk) still gets its own block and the combine fits (MAXCHUNK).
    // gs: the output-gate projection (96 of ~240 strips of the qkv GEMV) is only needed when the chunks are
    //     combined; with enough blocks free during the attention phase they compute it there (fixed-token step at
    //     B=1: 3374 -> 3287 us, e0102). The smallest chunk that still leaves GS_MIN_FREE blocks is taken.
#ifndef GS_MIN_FREE
#define GS_MIN_FREE 24   // B=8 at context 960: 5073 us with 48 (no split there), 4993 with 24 (4 gate strips per free block); e0106
#endif
    int cs = CHUNK;
    bool gs = false;
    if (!RED) {
      const int n = apos + 1 - alo;
      bool found = false;
#ifndef NO_GATE_SPLIT
      for (int k = 32; k <= CHUNK && !found; k <<= 1) {
        const int nc = (n + k - 1) / k;
        if (nc <= MAXCHUNK && nblk - B * NKVL * nc >= GS_MIN_FREE) { cs = k; gs = true; found = true; }
      }
#endif
#ifndef NO_SMALL_CHUNK
      for (int k = 32; k < CHUNK && !found; k <<= 1) {
        const int nc = (n + k - 1) / k;
        if (nc <= MAXCHUNK && B * NKVL * nc <= nblk) { cs = k; found = true; }
      }
#endif
    }
    const int nch = (apos + 1 - alo + cs - 1) / cs;
    const int nitems = B * NKVL * nch;
    // B: qkv
#ifdef STAGE_QKV   // staging costs ~1.2 us and a barrier; a block reads this fragment only once or twice (e0045)
    if (!RED) stage_frags(sm, c.xfrag, 73728);
    { OpQKV<RED> op{i, RED ? &sm.fx[0][0] : c.xfrag, RED ? nullptr : reinterpret_cast<const bf16*>(sm.stage)};
#else
    { OpQKV<RED> op{i, RED ? &sm.fx[0][0] : c.xfrag, nullptr};   // a block reads the fragment once or twice: straight from L2
#endif
      op.smap = gs ? 1 : 0;
      gemv_sel(c, op, gs ? L.qkv_strips - 96 - L.qkv_skip : L.qkv_strips, 72, 1, &sm.opart[0]); }
    __syncthreads();
    if (c.tprof && threadIdx.x == 0 && i == 20) {
      unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      c.tprof[49 * 64 + 0 * 152 + blockIdx.x] = gtime();
      c.tprof[49 * 64 + 3 * 152 + blockIdx.x] = smid;
    }
    grid_bar(c, nbar, barbase);
    TM(i, 3);
    // C: attention
    {
      if (blockIdx.x >= nitems && nitems < nblk) {
        prefetch_l2(L.wo, 288L * 24 * 2048, blockIdx.x - nitems, nblk - nitems);
        if (gs) { OpQKV<RED> opg{i, c.xfrag, nullptr}; opg.smap = 2; gemv_blk(c, opg, 96, 72, &sm.opart[0], 4, nitems, nblk - nitems); }
      }
      for (int it = blockIdx.x; it < nitems; it += nblk) {
        int b = it / (NKVL * nch), r = it % (NKVL * nch);
        phase_attention(c, i, b, r / nch, r % nch, nch, sm, !RED, cs, gs);
        __syncthreads();
      }
      if (c.tprof && threadIdx.x == 0 && i == 20) c.tprof[49 * 64 + 4 * 152 + blockIdx.x] = gtime();
    }
    grid_bar(c, nbar, barbase);
    TM(i, 4);
    // D: o_proj -> all cards
    if (RED) {
      const int t = c.h.pos, lo = L.glob ? 0 : max(0, t + 1 - WIN);
      phase_ocombine_red(c, i, (t + 1 - lo + CHUNK - 1) / CHUNK, sm);
    }
    { OpOproj<RED> op{i, RED ? &sm.fx[0][0] : c.ofrag}; gemv_sel(c, op, 288, 24, 2, &sm.opart[0]); }
    ev_attn++;
    TM(i, 5);
    // E: post-attention (token blocks)
    if (RED) {
      prefetch_l2(L.wr, 24L * 72 * 2048, blockIdx.x, nblk);
      phase_post_attn<true>(c, i, 0, sm, true, ev_attn);
      __syncthreads();
    } else {
#ifdef NO_TEAM
      if (blockIdx.x < B) phase_post_attn<false>(c, i, blockIdx.x, sm, true, ev_attn);
      else prefetch_l2(L.wr, 24L * 72 * 2048, blockIdx.x - B, nblk - B);
#else
      if (blockIdx.x < B * TB) { if (threadIdx.x < TNT) team_post<TC>(c, i, blockIdx.x / TB, blockIdx.x % TB, sm); }
      else prefetch_l2(L.wr, 24L * 72 * 2048, blockIdx.x - B * TB, nblk - B * TB);
#endif
      grid_bar(c, nbar, barbase);
    }
    TM(i, 7);
    // F: router
#ifdef STAGE_RT
    if (!RED) stage_frags(sm, c.x2frag, 73728);
    { OpRouter<RED> op{i, RED ? sm.fx2 : reinterpret_cast<const bf16*>(sm.stage), sm.relb}; gemv_sel(c, op, 24, 72, 4, &sm.opart[0]); }
#else
    { OpRouter<RED> op{i, RED ? sm.fx2 : c.x2frag, sm.relb}; gemv_sel(c, op, 24, 72, 4, &sm.opart[0]); }
#endif
    // the 24 router blocks publish sigmoid(logit) of their 16 experts x B tokens: warp q < 4 takes fragment value q
    // (one sigmoid per thread here instead of B/2 per thread in every block of the grid)
    if (blockIdx.x < 24 && threadIdx.x < 128) {
      const int q = threadIdx.x >> 5, ln = threadIdx.x & 31;
      const int row = blockIdx.x * 16 + (ln >> 2) + (q >> 1) * 8, col = 2 * (ln & 3) + (q & 1);
      if (col < B) put_l2(&c.rll[col * NE + row], sigm(sm.relb[ln * 4 + q]), c.h.tag0 + i + 1);
    }
    TM(i, 8);
    // G: routing (every block)
#ifdef ICTEST
    phase_route<TC>(c, i, sm);   // a second, i-cache-warm run is timed 31 -> 9
    TM(i, 31);
#endif
    phase_route<TC>(c, i, sm);
    TM(i, 9);
    int nu = sm.nuni;
    if (c.tprof && blockIdx.x == 0 && threadIdx.x == 0) c.tprof[i * 64 + 50] = nu;
    // H: w13 (+ act, amax to all cards)
#ifndef W13_STAGE_MINB
#define W13_STAGE_MINB 99   // never: with the shared memory at ~47 KB the fragment sits in L1 (e0074)
#endif
    const bool st13 = !RED && B >= W13_STAGE_MINB;   // few tokens: a warp reads only ~11 units of the fragment
    if (st13) {
      if (L.mx) stage_frags(sm, c.w13frag, 36864, c.w13sc, 144 * MAXB * 4);
      else stage_frags(sm, c.w13frag, 73728);
    }
    if (!(TC && TC13 && !L.mx))   // (tensor-core kernel, NVFP4 layer: done at the end of phase_route)
    { OpW13<RED> op{i, nu, &sm, RED ? sm.f13 : (st13 ? sm.stage : c.w13frag),
                    RED ? sm.sc13 : (L.mx ? (st13 ? reinterpret_cast<const float*>(sm.stage + 36864) : c.w13sc) : c.w13sc), RED ? sm.sc13 : c.w13sc};
      op.tag = c.h.tag0 + i + 1;
#ifndef W13_BLK_MAXNU
#define W13_BLK_MAXNU 16   // B=1: w13 12.1 -> 8.3 us per layer (3640 -> 3431 us per step), B=2 3927 -> 3811; from ~25 experts on the flat schedule wins (e0097/e0098)
#endif
      if (!RED && nu <= W13_BLK_MAXNU) gemv_blk(c, op, nu * 72, 72, &sm.opart[0], 4);   // few experts: whole strips per block
      else gemv_sel(c, op, (long)nu * 72, 72, 8, &sm.opart[0]); }
    __syncthreads();
    if (c.tprof && threadIdx.x == 0 && i == 20) c.tprof[49 * 64 + 1 * 152 + blockIdx.x] = gtime();
    {
      // every block: its share of the union experts' w2 slabs (thread u: expert u; one instruction each)
#ifndef NO_W2PF
      const unsigned per = 288u * 9u * (L.mx ? UB8 : UB4);
      if (threadIdx.x < nu) prefetch_items(L.w2 + (long)sm.uni[threadIdx.x] * per, per, blockIdx.x, nblk);
#endif
    }
    ev_amax++;
    TM(i, 10);
    TM(i, 11);
    // I: quantize act for w2
    if (RED) {
      phase_quant2_red(c, i, nu, sm);
    } else {
      // blocks 0 .. 8B-1: one routed (token, rank) pair each; the next nu blocks: zero-fill of one union expert
      for (int it = blockIdx.x; it < B * TOPK + nu; it += nblk) {
        if (it < B * TOPK) phase_quant2_pair<TC>(c, i, sm.uix[it], it / TOPK, sm);
        else phase_quant2_zero<TC>(c, i, it - B * TOPK, sm);
        __syncthreads();
      }
      grid_bar(c, nbar, barbase);
    }
    TM(i, 12);
    // J: w2 -> all cards
    if (TC && TC2 && !L.mx) gemv_tc<true>(c, sm, i, nu);
    else
    { OpW2<RED> op{i, nu, &sm, RED ? sm.f2 : c.w2frag, RED ? sm.sc2 : c.w2sc}; gemv_sel(c, op, 288, nu * 9, 16, &sm.opart[0]); }
    __syncthreads();
    if (c.tprof && threadIdx.x == 0 && i == 20) c.tprof[49 * 64 + 2 * 152 + blockIdx.x] = gtime();
    ev_mlp++;
    TM(i, 13);
  }
  if (layer_hi == NL) {
    TM(NL, 0);
    if (RED) {
      TM(NL, 1); phase_final<true>(c, 0, sm);
      __syncthreads();
    } else {
#ifdef NO_TEAM
      if (blockIdx.x < B) { TM(NL, 1); phase_final<false>(c, blockIdx.x, sm); }
#else
      TM(NL, 1);
      if (blockIdx.x < B * TB) { if (threadIdx.x < TNT) team_final(c, blockIdx.x / TB, blockIdx.x % TB, sm); }
#endif
      grid_bar(c, nbar, barbase);
    }
    TM(NL, 2);
    for (int k = threadIdx.x; k < NWARP * MAXB; k += NTH) { sm.bestv[k] = -INFINITY; sm.besti[k] = 1 << 30; }
    __syncthreads();
#ifdef UN_STAGE
    if (!RED) stage_frags(sm, c.hfrag, 73728);
    { OpUnembed<RED> op{sm.bestv, sm.besti, RED ? sm.fh : reinterpret_cast<const bf16*>(sm.stage)};
#else
    { OpUnembed<RED> op{sm.bestv, sm.besti, RED ? sm.fh : c.hfrag};
#endif
      op.tag = (c.h.tag0 + NL + 1) | 0x80000000u; gemv_sel(c, op, c.un_strips, 72, 32, &sm.opart[0]); }
    __syncthreads();
    // block best per col -> candidates on every card
    if (threadIdx.x < B) {
      int col = threadIdx.x;
      float bv = -INFINITY; int bi = 1 << 30;
      for (int w = 0; w < NWARP; w++) {
        float v = sm.bestv[w * MAXB + col]; int ii = sm.besti[w * MAXB + col];
        if (v > bv || (v == bv && ii < bi)) { bv = v; bi = ii; }
      }
      for (int d = 0; d < NCARD; d++) { c.comm[d]->cand_v[c.h.card][col][blockIdx.x] = bv; c.comm[d]->cand_i[c.h.card][col][blockIdx.x] = bi; }
    }
    xsignal(c, 3, nbar, barbase);
    TM(NL, 3);
    if (blockIdx.x == 0) {
      xwait(c, 3, 1);
      TM(NL, 4);
      if (threadIdx.x < B) {
        int col = threadIdx.x;
        float bv = -INFINITY; int bi = 1 << 30;
        for (int d = 0; d < NCARD; d++)
          for (int k = 0; k < nblk; k++) {
            float v = c.comm[c.h.card]->cand_v[d][col][k]; int ii = c.comm[c.h.card]->cand_i[d][col][k];
            if (v > bv || (v == bv && ii < bi)) { bv = v; bi = ii; }
          }
        c.next_tok[col] = bi;
        if (c.h.gen_out) {
          if (c.h.gen_n) c.h.gen_out[c.h.gen_idx * MAXB + col] = bi + 1;          // host-mapped stream: 0 = not yet
          else c.h.gen_out[c.h.gen_idx * MAXB + col] = bi;
        }
      }
    }
  }
  grid_bar(c, nbar, barbase);
  if (blockIdx.x == 0 && threadIdx.x == 0) c.bar[1] = barbase + nbar * gridDim.x;
#ifdef TC_BODY_CALLS
  if (TC) tc_end(c, sm);
#endif
}

// the kernel parameters (Ctx, ~14 KB) are copied into shared memory once: every later access is a shared load,
// immune to the L1 invalidations of the barriers' acquire fences
#define CTXB ((sizeof(Ctx) + 127) / 128 * 128)
#define SMEM_TOTAL (CTXB + sizeof(Smem))
__device__ __forceinline__ const Ctx& load_ctx(const Ctx* cp, const Hdr& hdr, unsigned char* smraw) {
  static_assert(sizeof(Ctx) % 8 == 0 && sizeof(Hdr) % 8 == 0, "Ctx size");
  const uint2* src = reinterpret_cast<const uint2*>(cp);
  uint2* dst = reinterpret_cast<uint2*>(smraw);
  const int nh = sizeof(Hdr) / 8;
  for (int k = threadIdx.x; k < (int)(sizeof(Ctx) / 8); k += NTH)
    dst[k] = (k < nh) ? reinterpret_cast<const uint2*>(&hdr)[k] : src[k];   // the per-step header comes by value
  __syncthreads();
  return *reinterpret_cast<const Ctx*>(smraw);
}
extern "C" __global__ void __launch_bounds__(NTH, 1) ink_step_std(const Ctx* __restrict__ cp, const Hdr hdr, int lo, int hi) {
  extern __shared__ __align__(16) unsigned char smraw[];
  const Ctx& c = load_ctx(cp, hdr, smraw);
  step_body<false, false>(c, *reinterpret_cast<Smem*>(smraw + CTXB), lo, hi);
}
// the same step with the NVFP4 expert GEMVs on the tensor-core path (launched for batches >= TC_MINB: the fixed
// costs of that path lose at one or two sequences, and compiled into one kernel its branches cost the other path
// registers: B=1 +1.4%, e0165-e0169)
extern "C" __global__ void __launch_bounds__(NTH, 1) ink_step_tc(const Ctx* __restrict__ cp, const Hdr hdr, int lo, int hi) {
  extern __shared__ __align__(16) unsigned char smraw[];
  const Ctx& c = load_ctx(cp, hdr, smraw);
  step_body<false, (TC13 || TC2) != 0>(c, *reinterpret_cast<Smem*>(smraw + CTXB), lo, hi);
}
extern "C" __global__ void __launch_bounds__(NTH, 1) ink_step_red(const Ctx* __restrict__ cp, const Hdr hdr, int lo, int hi) {
  extern __shared__ __align__(16) unsigned char smraw[];
  const Ctx& c = load_ctx(cp, hdr, smraw);
  step_body<true, false>(c, *reinterpret_cast<Smem*>(smraw + CTXB), lo, hi);
}

// ------------------------------------------------------------------------------------------------ host API
extern "C" int ink_smem_bytes() { return (int)sizeof(Smem); }
extern "C" int ink_tc_mode() { return TC13 | (TC2 << 1); }
extern "C" int ink_tc_minb() { return (TC13 || TC2) ? TC_MINB : 1 << 30; }   // batches from this size on are launched as ink_step_tc   // which NVFP4 expert matrices the loader lays out for tcgen05
extern "C" int ink_ctx_bytes() { return (int)sizeof(Ctx); }
extern "C" int ink_un_w12() {   // 1: this build reads the unembedding in the 12-bit code (the loader must agree)
#if defined(W12) || defined(W12U)
  return 1;
#else
  return 0;
#endif
}
// positions a sequence may have: a global layer's attention is combined from at most MAXCHUNK chunk records
extern "C" int ink_max_positions() { return MAXCHUNK * CHUNK; }
// Layer-0 token tables (track B) are bit-identical to the computation they replace only if the tabulated rows are
// summed in one order whatever the batch and the position: the team phases, and the gate split always taken in a
// local layer (then the main qkv GEMV is one strip per block, its 72 units over 24 warps). 1 if that holds.
extern "C" int ink_l0t_ok(int nblk) {
#if defined(NO_TEAM) || defined(NO_GATE_SPLIT) || defined(W12)
  return 0;
#else
  return ((GEMV_BLK & 1) && nblk - MAXB * NKVL * ((WIN + CHUNK - 1) / CHUNK) >= GS_MIN_FREE && (WIN + CHUNK - 1) / CHUNK <= MAXCHUNK
          && 237 - 96 <= nblk) ? 1 : 0;
#endif
}
extern "C" int ink_comm_bytes() { return (int)sizeof(Comm); }
extern "C" int ink_layer_bytes() { return (int)sizeof(LayerW); }

extern "C" int ink_setup(int dev) {
  cudaSetDevice(dev);
  cudaError_t e = cudaFuncSetAttribute(ink_step_std, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_TOTAL);
  if (e) return (int)e;
  e = cudaFuncSetAttribute(ink_step_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_TOTAL);
  if (e) return (int)e;
  e = cudaFuncSetAttribute(ink_step_red, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_TOTAL);
  return (int)e;
}

extern "C" int ink_enable_peer(int n) {
  for (int a = 0; a < n; a++) {
    cudaSetDevice(a);
    for (int b = 0; b < n; b++) if (a != b) {
      cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) return (int)e;
      cudaGetLastError();
    }
  }
  return 0;
}
extern "C" void* ink_host_alloc(size_t n) {
  void* p = nullptr;
  if (cudaHostAlloc(&p, n, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) return nullptr;
  memset(p, 0, n);
  return p;
}
extern "C" int ink_h2d(void* dst, const void* src, size_t n, void* stream) {
  return (int)cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}
extern "C" int ink_launch(int dev, const void* ctx, const void* hdr, int nblk, void* stream, int lo, int hi, int red) {
  cudaSetDevice(dev);
  const Hdr h = *reinterpret_cast<const Hdr*>(hdr);
  if (red == 2) ink_step_tc<<<nblk, NTH, SMEM_TOTAL, (cudaStream_t)stream>>>((const Ctx*)ctx, h, lo, hi);
  else if (red) ink_step_red<<<nblk, NTH, SMEM_TOTAL, (cudaStream_t)stream>>>((const Ctx*)ctx, h, lo, hi);
  else ink_step_std<<<nblk, NTH, SMEM_TOTAL, (cudaStream_t)stream>>>((const Ctx*)ctx, h, lo, hi);
  return (int)cudaGetLastError();
}
