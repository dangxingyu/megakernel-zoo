// Inkling-2-Small pass engine for one GB200 node (4 x B200).
//
// A pass carries T rows (a row = one position of one sequence: prompt chunks and decoding sequences together)
// through the 48 layers. Attention is split over the cards by heads (12 q heads / 2 KV heads per card, K/V pages on
// every card for its heads); every expert lives whole on one card (96 per card; the scheduler swaps experts between
// cards as the traffic changes). Everything else of a layer (the residual state, its mixes and norms, o_proj, the
// router and the routing, the MoE-input quantization) is done once per row, on the row's home card (the cards own
// consecutive quarters of a pass; passes of up to REPL_T rows are replicated instead). Per layer the cards meet four
// times through peer memory, each time after a flag written by the producing card: the attention input of the other
// cards' rows (k_xin), the attention outputs of a card's own rows (k_gather_o), the routing and quantized MoE input
// (k_xroute), the expert outputs of a card's own rows (k_combine). Nothing waits for the host inside a pass, and the
// scheduler that composes the passes is a kernel too (k_sched, engine/sched.h).
// Every row's arithmetic is independent of the other rows of the pass, of how a sequence is cut into passes and of
// which card is its home: GEMM columns are independent, attention sums page-aligned segments in order, expert
// outputs are summed in expert order, every reduction has one thread layout for all pass sizes.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <algorithm>

typedef __nv_bfloat16 bf16;

#define H 4608
#define NL 48
#define NE 384
#define NEC 96            // experts per card
#define TOPK 8
#define HD 128
#define NQL 12
#define NKVL 2
#define DREL 16
#define RELEXT 1024
#define WIN 1024
#define NCARD 4
#define PAGE 512
#define MAXPG 68          // pages per sequence (34816 positions)
#define VOCAB_REAL 200058
#define HS 9728           // short-convolution state per (layer, lag): k 256, v 256, attn 4608, mlp 4608
#define NTB 256           // threads of the per-row kernels
#define XS8W 384          // bytes of a row of block scales (288) with its routing appended (PCtx.xs8)
#define EPT (H / NTB)     // 18 elements per thread
#define COLT 128          // GEMM column tile
#define QG 1536
#define QR 3072
#define QS 3264
#define QK 3280
#define UB4 576
#define UB8 1056
#define APW 132           // attention record of a q vector and page: max, sum, (2 unused), 128 values
#define NSIG (NL * 4 + 24)       // exchange flags (XS_*), then the three of the serving loop
#define XS_ATT(i) ((i) * 2)            // attention outputs of layer i
#define XS_EXP(i) ((i) * 2 + 1)        // expert outputs
#define XS_JOIN (NL * 2)               // unembedding candidates
#define XS_FIN (NL * 2 + 1)            // end of the layers: final hidden rows, last short-convolution rows
#define XS_ROUTE(i) (NL * 2 + 8 + (i)) // routing and MoE input
#define XS_XIN(i) (NL * 3 + 8 + (i))   // attention input
#define SIG_SCHED (NSIG - 1)      // card 0: tag of the pass it has composed
#define SIG_TAKEN (NSIG - 2)      // every card: tag of the pass whose tables it has taken
#define SIG_SWAP (NSIG - 3)       // every card: its copies of the experts to be swapped are made
#define SWB 48                    // blocks per swap
#define SWTMP (36L * 144 * 4608 * 3 / 2)   // bytes of the largest expert (w13 + w2, mxfp8)

struct PLayer {
  // weights in the tensor core's operand layout: units of 128 rows x one K step (bf16: 16 k, 4096 B; nvfp4: 64 k,
  // 4096 B of codes + 512 B scale atom; mxfp8: 32 k, 4096 B + 512 B atom), [row group][K step][unit]
  const uint8_t* wqkv;      // [30 or 32 row groups][288]: q|gate|r|sink (26), k (+taps) (2 or 3), v (+taps) (2 or 3)
  const uint8_t* wo;        // [36][384] (full o_proj, replicated)
  const uint8_t* wr;        // router [3][288]
  const float* rbias;       // [384]
  const uint8_t* w13;       // [96][36][72 or 144]
  const uint8_t* w2;        // [96][36][36 or 72]
  const float* f13;         // [384] weight outer factor amax / denominator (nvfp4), by expert
  const float* f2;
  const bf16* norm[3];
  const bf16* mlp_norm;
  const bf16* ksc;          // local: [256][4] (channel, tap); global: [4][256] (lag, channel)
  const bf16* vsc;
  const bf16* attn_sc;      // [4608][4]
  const bf16* mlp_sc;
  const bf16* qscale;       // [12][128]
  const bf16* rscale;       // [12][16]
  const uint8_t* relp;      // relative-bias projection [1024][16] as tensor-core units [8 row groups][1 K step][4096 B]
  const float* aq;          // [nq][4608], pre-scaled by H^-0.5 (layer 0: null)
  const float* mq;          // [4608], pre-scaled
  uint8_t* kpg;             // K / V pages of this layer [npages][2 kv][131072 B], tensor-core operand layouts (kaddr, vaddr)
  uint8_t* vpg;
  int glob, mx, qkv_strips, pad;
};

#include "sched.h"      // PHdr (what a pass is) and the scheduler that composes passes on the device

// A pass row has a home card. The residual state and what is computed from it per row (the mixes and norms, o_proj,
// the router and the routing, the MoE input) exist only there; attention (q/k/v by heads, K/V pages) and the experts
// stay spread over the four cards, and what they need of a row (attention input, routing, quantized MoE input) is
// fetched from its home. The cards own consecutive quarters of the pass's rows.
// In a pass of up to REPL_T rows every card owns every row, as if replicated: that work is small then, and the
// fetches (two remote round trips each, on every layer) are not.
#define REPL_T 64
__host__ __device__ inline int own_lo(int T, int card) { return T <= REPL_T ? 0 : (T * card + 3) >> 2; }
__host__ __device__ inline int own_hi(int T, int card) { return (T <= REPL_T || card >= NCARD - 1) ? T : (T * (card + 1) + 3) >> 2; }
__host__ __device__ inline int row_home(int T, int r) {       // passes of more than REPL_T rows
  int h = 0;
  for (int q = 1; q < NCARD; q++) if (r >= ((T * q + 3) >> 2)) h = q;
  return h;
}
enum { PX_XIN = 0, PX_IDS, PX_WTS, PX_XOUTER, PX_CNT, PX_XQ, PX_XS8, PX_ATTN_RAW, PX_HFIN, NPX };
struct PCtx {
  PHdr h;
  PLayer L[NL];
  int card, npages, tmax, pmax;
  float eps; int umax; int pad0, pad1;
  const bf16* embed; const bf16* embed_norm; const float* oq; const bf16* fnorm;
  const uint8_t* unembed; int un_strips; int un_row0;
  int* tab;                 // per-pass tables (offsets in the header)
  bf16* blocks;             // [9][tmax][H]: finished blocks and the partial block (index = blocks finished)
  bf16* xin;                // [3][tmax][H] attention inputs, tile layout (64 rows per tile, see xta)
  bf16* qkvo;               // [tmax][3824]
  bf16* qn;                 // [tmax][12][128]
  bf16* relb;               // [tmax][12][1024]
  uint8_t* rt;              // [tmax*12/128 tiles][4096 B]: clipped r of (row, head), tile layout (input of the bias GEMM)
  float* apart;             // [umax][128][APW]
  bf16* attn_out;           // [tmax][1536]
  bf16* xo;                 // [tmax][6144] tile layout
  bf16* attn_raw;           // [tmax][H]
  bf16* attn_s;             // [tmax][H]
  bf16* mlp_in;             // [tmax][H]
  bf16* mlp_inx;            // [tmax][H] tile layout (router input)
  float* rlog;              // [tmax][384]
  int* ids; float* wts;     // [tmax][8]
  int* pidx;                // [tmax][8] pair index on the expert's card
  int* cnt;                 // [NL][384]
  int* emap;                // [NL][384]: where an expert's weights are: card | index on that card << 8 (the host moves
                            // experts between cards to even out the cards' load; the same table on every card)
  uint8_t* xq;              // [tmax][H]: e2m1 codes (2304 B, low nibble first) or e4m3 codes (4608 B) of the MoE input
  float* xouter;            // [tmax] (nvfp4)
  uint8_t* xs8;             // [tmax][XS8W]: 288 block scales, e4m3 per 16 (nvfp4) or ue8m0 exponent per 32 (mxfp8); then the
                            // row's ids (32 B), weights (32 B) and outer scale (4 B) once more, so that another card
                            // fetches all of a row's routing in one contiguous piece (five pieces: +20 us per layer)
  int* ptok; int* pslot;    // [4][pmax]
  int* items;               // [4][pmax][4]: expert, first pair, pairs
  int* ginfo;               // [4][4]: items, pairs
  bf16* h13;                // [pmax][H]
  uint8_t* xt13;            // [items][72 or 144 K steps][2560]: w13 inputs of an item's pairs, operand layout + scale atom
  float* aouter;            // [pmax]
  uint8_t* xt2;             // [items][36 or 72][2560]: w2 inputs
  int* pitem;               // [pmax] item of a pair (this card)
  bf16* pairout;            // [pmax][H]
  bf16* mlp_raw;            // [tmax][H]
  bf16* hist;               // [nslot][NL][3][HS]
  bf16* snap;               // [npages][NL][3][HS]
  bf16* hfin;               // [tmax][H] tile layout
  unsigned long long* cand; // [tmax]
  unsigned* sig;            // [NSIG]
  unsigned* err;            // [4]
  int* cntg;                // [NL][384]: pairs per expert of the pass, all cards' rows (cnt: this card's rows)
  unsigned* gate;           // [NSIG]: block 0 of an exchange kernel has seen the other cards (xwait)
  bf16* xnat;               // [3][tmax][H]: the attention inputs of this card's rows, row by row (what the others fetch)
  float* kscr;              // [XS_U][128 rows][64 columns]: first halves of the dense GEMMs' K ranges (small passes)
  unsigned* kflag;          // [XS_U]: pass and layer whose first half is in kscr
  const bf16* attn_out_p[NCARD];
  const bf16* pairout_p[NCARD];
  unsigned* sig_p[NCARD];
  const bf16* mlp_raw_p[NCARD];
  const uint8_t* peer[NCARD][NPX];      // the cards' xin, ids, wts, xouter, cnt, xq, xs8, attn_raw, hfin
  // device-side serving loop (card 0 composes the passes; the others copy its tables)
  Sched* sched; SShared* shr; const int* arena;
  const unsigned long long* cand_p[NCARD];
  PCtx* ctx_p[NCARD];
  const int* tab_p[NCARD];
  const uint8_t* w13_p[NL][NCARD];      // the cards' expert weights (experts are swapped between cards)
  const uint8_t* w2_p[NL][NCARD];
  uint8_t* swtmp;                       // [S_SWMAX][SWTMP]: an incoming expert, before it replaces the outgoing one
  const int* harena;                    // the host's page-locked copy of the arriving prompts (same offsets as the arena)
};

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 tobf(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
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
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) { uint32_t r; asm("mul.rn.f16x2 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b)); return r; }
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
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) { unsigned v; asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) { asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }

// Tile layout of the dense GEMM inputs (the tensor core's B operand): 256 rows per tile, per K step of 16 a unit of
// 8192 B: row n, element k at (n/8)*256 + ((k/8)&1)*128 + (n%8)*16 + (k%8)*2. Byte offset of element k of `row`:
__device__ __forceinline__ long xta(int row, int k, int KS) {
  const int n = row & 255;
  return ((long)(row >> 8) * KS + (k >> 4)) * 8192 + (n >> 3) * 256 + ((k >> 3) & 1) * 128 + (n & 7) * 16 + (k & 7) * 2;
}
// (unused) B-fragment order of a row of K values: element k at ((t * (K/16) + s) * 4 + e), t = (k%8)/2, s = k/16,
// e = (k%2) + 2*((k%16)/8): lane (col, t) of an m16n8k16 reads its 8 bytes of step s at one address.
__device__ __forceinline__ int xfi(int k, int K) { return (((k & 7) >> 1) * (K >> 4) + (k >> 4)) * 4 + (k & 1) + 2 * ((k >> 3) & 1); }

// block-wide sums of n per-thread values (every thread gets the totals); red: shared [nwarp][n]
__device__ __forceinline__ void block_sum_n(float* v, int n, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int i = 0; i < n; i++) {
    float x = v[i];
#pragma unroll
    for (int off = 16; off; off >>= 1) x += __shfl_xor_sync(0xffffffff, x, off);
    if (lane == 0) red[w * n + i] = x;
  }
  __syncthreads();
  for (int i = 0; i < n; i++) {
    float s = 0.f;
    for (int k = 0; k < nw; k++) s += red[k * n + i];
    v[i] = s;
  }
  __syncthreads();
}
__device__ __forceinline__ float block_max1(float x, float* red) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
  for (int off = 16; off; off >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, off));
  if (lane == 0) red[w] = x;
  __syncthreads();
  float s = 0.f;
  for (int k = 0; k < nw; k++) s = fmaxf(s, red[k]);
  __syncthreads();
  return s;
}

// the producers of exchange `idx` on this card have finished (this kernel runs after them on the card's stream):
// say so, then wait until the other three cards have said the same for this pass.
__device__ __forceinline__ void xwait(const PCtx* c, int idx) {
  // Block 0 of the kernel talks to the other cards (one thread per card: a remote read is ~2 us); the kernel's other
  // blocks wait for block 0 on a word of this card's memory.
  const unsigned tag = (unsigned)c->h.tag;
  const int me = c->card, p = threadIdx.x;
  if (blockIdx.x == 0) {
    if (p == 0) st_release_sys(&c->sig_p[me][idx], tag);
    if (p < NCARD && p != me) {
      unsigned long long spins = 0;
      while (ld_acquire_sys(&c->sig_p[p][idx]) != tag) {
        if (++spins > (1ull << 31)) { c->err[0] = 1 + idx; break; }
      }
    }
    __syncthreads();
    if (p == 0) st_release_sys(&c->gate[idx], tag);
  } else if (p == 0) {
    unsigned long long spins = 0;
    while (ld_acquire_sys(&c->gate[idx]) != tag) {
      if (++spins > (1ull << 33)) { c->err[1] = 1 + idx; break; }
    }
  }
  __syncthreads();
}

#define TAB(o) (c->tab + c->h.o)

// ------------------------------------------------------------------------------------------------ embedding
// Per-row vector kernels: one block per row, 288 threads, 16 consecutive elements per thread (two 16-byte loads
// or stores per bf16 vector). Sums over the 4608 elements: per thread in element order, then the warp's shuffle
// tree, then the nine warps in order: the same for every pass size.
#define RT 288
#define RE 16
__device__ __forceinline__ void ld16(const bf16* p, float* v) {
  const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
  const bf16* x = reinterpret_cast<const bf16*>(&a);
  const bf16* y = reinterpret_cast<const bf16*>(&b);
#pragma unroll
  for (int e = 0; e < 8; e++) { v[e] = bf(x[e]); v[8 + e] = bf(y[e]); }
}
__device__ __forceinline__ uint4 pk8(const float* v) {
  uint4 o;
  bf16* b = reinterpret_cast<bf16*>(&o);
#pragma unroll
  for (int e = 0; e < 8; e++) b[e] = tobf(v[e]);
  return o;
}
__device__ __forceinline__ void st16(bf16* p, const float* v) {
  *reinterpret_cast<uint4*>(p) = pk8(v);
  *reinterpret_cast<uint4*>(p + 8) = pk8(v + 8);
}
// the same 16 values into the tile layout of a dense GEMM input (two 16-byte pieces)
__device__ __forceinline__ void st16t(uint8_t* base, int row, int k0, int KS, const float* v) {
  *reinterpret_cast<uint4*>(base + xta(row, k0, KS)) = pk8(v);
  *reinterpret_cast<uint4*>(base + xta(row, k0 + 8, KS)) = pk8(v + 8);
}
__global__ void __launch_bounds__(RT) k_embed(const PCtx* c) {
  __shared__ float red[9 * 4];
  const int tid = threadIdx.x, k0 = tid * RE;
  if (blockIdx.x == 0) for (int i = tid; i < NL * NE; i += RT) c->cnt[i] = 0;
  const int r = own_lo(c->h.T, c->card) + blockIdx.x;
  if (r >= own_hi(c->h.T, c->card)) return;
  const int tok = c->h.tok_dev ? c->h.tokens[r] : TAB(o_tok)[r];
  float u[RE], g[RE], ss = 0.f;
  ld16(c->embed + (long)tok * H + k0, u);
  ld16(c->embed_norm + k0, g);
#pragma unroll
  for (int e = 0; e < RE; e++) ss += u[e] * u[e];
  block_sum_n(&ss, 1, red);
  const float inv = rsqrtf(ss / (float)H + c->eps);
#pragma unroll
  for (int e = 0; e < RE; e++) u[e] = u[e] * inv * g[e];
  st16(c->blocks + (long)r * H + k0, u);
}

// ------------------------------------------------------------------------------------------------ attention residual mix
__device__ __forceinline__ void ld16u(const uint4* p, float* v) {      // 16 bf16 held in two registers' worth of words
  const uint32_t w[8] = {p[0].x, p[0].y, p[0].z, p[0].w, p[1].x, p[1].y, p[1].z, p[1].w};
#pragma unroll
  for (int e = 0; e < 8; e++) { v[2 * e] = __uint_as_float(w[e] << 16); v[2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u); }
}
// sources: n rows of bf16 [H] (+ an addend on the last one, summed in fp32); NQ queries (pre-scaled); out [NQ][RE]
template <int NQ>
__device__ __forceinline__ void mix_row(const bf16* const* src, int n, const bf16* add, const float* qv, float (*out)[RE], float* red) {
  const int k0 = threadIdx.x * RE;
  float acc[10 * (1 + NQ)];
  float qq[NQ][RE];
#pragma unroll
  for (int q = 0; q < NQ; q++)
#pragma unroll
    for (int e = 0; e < RE; e += 4) {
      const float4 t = *reinterpret_cast<const float4*>(qv + (long)q * H + k0 + e);
      qq[q][e] = t.x; qq[q][e + 1] = t.y; qq[q][e + 2] = t.z; qq[q][e + 3] = t.w;
    }
  // A row kernel is bound by the loads it has in flight: the loads of four sources are issued before the first is used.
  uint4 ad[2];
  if (add) { const uint4* g = reinterpret_cast<const uint4*>(add + k0); ad[0] = g[0]; ad[1] = g[1]; }
  for (int s0 = 0; s0 < n; s0 += 4) {
    uint4 t[8];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint4* g = reinterpret_cast<const uint4*>(src[min(s0 + j, n - 1)] + k0);
      t[2 * j] = g[0]; t[2 * j + 1] = g[1];
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int s = s0 + j;
      if (s >= n) break;
      float v[RE], ss = 0.f, d[NQ];
      ld16u(t + 2 * j, v);
      if (add && s == n - 1) {
        float a[RE];
        ld16u(ad, a);
#pragma unroll
        for (int e = 0; e < RE; e++) v[e] += a[e];
      }
#pragma unroll
      for (int q = 0; q < NQ; q++) d[q] = 0.f;
#pragma unroll
      for (int e = 0; e < RE; e++) {
        ss += v[e] * v[e];
#pragma unroll
        for (int q = 0; q < NQ; q++) d[q] += v[e] * qq[q][e];
      }
      acc[s * (1 + NQ)] = ss;
#pragma unroll
      for (int q = 0; q < NQ; q++) acc[s * (1 + NQ) + 1 + q] = d[q];
    }
  }
  // The row's sums (per warp: the shuffle tree; then the nine warps in order) and the softmax over the sources are the
  // same scalars for every thread of the block: warp 0 makes them once (lane = value, then lane = query), the block
  // reads them from shared memory. red: [nw][n(1+NQ)] partials, then the totals, 1/rms per source, weights per query.
  const int nv = n * (1 + NQ), lane = threadIdx.x & 31, wp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  float* tot = red + nw * nv;
  float* sinv = tot + nv;
  float* sp = sinv + 10;
  for (int j = 0; j < nv; j++) {
    float x = acc[j];
#pragma unroll
    for (int off = 16; off; off >>= 1) x += __shfl_xor_sync(0xffffffff, x, off);
    if (lane == 0) red[wp * nv + j] = x;
  }
  __syncthreads();
  if (wp == 0) {
    for (int j = lane; j < nv; j += 32) {
      float t = 0.f;
      for (int k = 0; k < nw; k++) t += red[k * nv + j];
      tot[j] = t;
    }
    __syncwarp();
    if (lane < NQ) {
      const int q = lane;
      float iv[10], pq[10];
      for (int s = 0; s < n; s++) iv[s] = rsqrtf(tot[s * (1 + NQ)] / (float)H + 1e-6f);
      float m = -3.0e38f, sum = 0.f;
      for (int s = 0; s < n; s++) { pq[s] = tot[s * (1 + NQ) + 1 + q] * iv[s]; m = fmaxf(m, pq[s]); }
      for (int s = 0; s < n; s++) { pq[s] = expf(pq[s] - m); sum += pq[s]; }
      for (int s = 0; s < n; s++) { sp[q * 10 + s] = pq[s] / sum; if (q == 0) sinv[s] = iv[s]; }
    }
  }
  __syncthreads();
  float inv[10], p[NQ][10];
  for (int s = 0; s < n; s++) inv[s] = sinv[s];
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    for (int s = 0; s < n; s++) p[q][s] = sp[q * 10 + s];
#pragma unroll
    for (int e = 0; e < RE; e++) out[q][e] = 0.f;
  }
  // second read of the sources (kept in registers they spill: measured 62 -> 97 us per layer at 2048 rows; staged in
  // shared memory, two blocks per SM instead of seven: 73.0 -> 75.7 ms per pass), again four sources' loads at a time
  for (int s0 = 0; s0 < n; s0 += 4) {
    uint4 t[8];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint4* g = reinterpret_cast<const uint4*>(src[min(s0 + j, n - 1)] + k0);
      t[2 * j] = g[0]; t[2 * j + 1] = g[1];
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int s = s0 + j;
      if (s >= n) break;
      float v[RE];
      ld16u(t + 2 * j, v);
      if (add && s == n - 1) {
        float a[RE];
        ld16u(ad, a);
#pragma unroll
        for (int e = 0; e < RE; e++) v[e] += a[e];
      }
#pragma unroll
      for (int e = 0; e < RE; e++) {
        const float x = v[e] * inv[s];
#pragma unroll
        for (int q = 0; q < NQ; q++) out[q][e] += p[q][s] * x;
      }
    }
  }
}
// RMS norm with gain of NQ per-thread vectors (in place)
template <int NQ>
__device__ __forceinline__ void rms_rows(float (*u)[RE], const bf16* const* w, float eps, float* red) {
  const int k0 = threadIdx.x * RE;
  float ss[NQ];
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    ss[q] = 0.f;
#pragma unroll
    for (int e = 0; e < RE; e++) ss[q] += u[q][e] * u[q][e];
  }
  block_sum_n(ss, NQ, red);
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    const float inv = rsqrtf(ss[q] / (float)H + eps);
    float g[RE];
    ld16(w[q] + k0, g);
#pragma unroll
    for (int e = 0; e < RE; e++) u[q][e] = u[q][e] * inv * g[e];
  }
}

// attention input of layer i: mix of the finished blocks and the partial block, RMS norm -> xin (tile layout)
template <int NQ>
__device__ __forceinline__ void mix_attn_row(const PCtx* c, int i, int r, float* red) {
  const int k0 = threadIdx.x * RE;
  const PLayer& L = c->L[i];
  const int nb = (i + 5) / 6;             // finished blocks; the partial block is blocks[nb]
  const long tm = c->tmax;
  float u[NQ][RE];
  if (i == 0) ld16(c->blocks + (long)r * H + k0, u[0]);
  else {
    const bf16* src[10];
    for (int s = 0; s <= nb; s++) src[s] = c->blocks + ((long)s * tm + r) * H;
    mix_row<NQ>(src, nb + 1, nullptr, L.aq, u, red);
  }
  const bf16* w[3] = {L.norm[0], L.norm[1], L.norm[2]};
  rms_rows<NQ>(u, w, c->eps, red);
#pragma unroll
  for (int q = 0; q < NQ; q++) {
    st16t(reinterpret_cast<uint8_t*>(c->xin + (long)q * tm * H), r, k0, H / 16, u[q]);
    st16(c->xnat + ((long)q * tm + r) * H + k0, u[q]);
  }
}
template <int NQ>
__global__ void __launch_bounds__(RT) k_mix_attn(const PCtx* c, int i) {
  const int r = own_lo(c->h.T, c->card) + blockIdx.x;
  if (r >= own_hi(c->h.T, c->card)) return;       // this card's rows only
  __shared__ float red[9 * 40 + 40 + 10 + 30];
  mix_attn_row<NQ>(c, i, r, red);
}

// the attention inputs of the rows whose home is another card: read row by row (contiguous remote reads: the tile
// layout's scattered 16-byte pieces came at 80 GB/s), written into this card's tile layout
template <int NQ>
__global__ void __launch_bounds__(576) k_xin(const PCtx* c, int i) {
  const int T = c->h.T, me = c->card;
  if (((int)blockIdx.x >= T && blockIdx.x > 0) || T <= REPL_T) return;
  xwait(c, XS_XIN(i));
  const long tm = c->tmax;
  // a block takes rows blockIdx.x, + gridDim.x, ...: the grid is a few blocks per SM, not one block per row of the
  // largest pass (the device loop's grids are fixed: 3584 blocks that find no row cost a small pass ~2 us)
  for (int r = blockIdx.x; r < T; r += gridDim.x) {
    if (r >= own_lo(T, me) && r < own_hi(T, me)) continue;
    const int hm = row_home(T, r);
#pragma unroll
    for (int q = 0; q < NQ; q++) {
      const uint4* src = reinterpret_cast<const uint4*>(c->peer[hm][PX_XIN] + ((long)q * tm + r) * H * 2);
      uint8_t* dst = reinterpret_cast<uint8_t*>(c->xin) + (long)q * tm * H * 2;
      for (int a = threadIdx.x; a < H / 8; a += blockDim.x) *reinterpret_cast<uint4*>(dst + xta(r, a * 8, H / 16)) = src[a];
    }
  }
}
// the routing and the quantized MoE input of the rows whose home is another card; the pass's pairs per expert
template <bool MX>
__global__ void __launch_bounds__(320) k_xroute(const PCtx* c, int i) {
  const int T = c->h.T, r = blockIdx.x, me = c->card, tid = threadIdx.x;
  if (r >= T && r > 0) return;
  if (T <= REPL_T) {                      // every card has routed every row
    if (r == 0) for (int e = tid; e < NE; e += blockDim.x) c->cntg[i * NE + e] = c->cnt[i * NE + e];
    return;
  }
  xwait(c, XS_ROUTE(i));
  if (r == 0)
    for (int e = tid; e < NE; e += blockDim.x) {
      int n[NCARD];
#pragma unroll
      for (int d = 0; d < NCARD; d++) n[d] = reinterpret_cast<const int*>(c->peer[d][PX_CNT])[i * NE + e];
      c->cntg[i * NE + e] = (n[0] + n[1]) + (n[2] + n[3]);
    }
  if (r >= T || (r >= own_lo(T, me) && r < own_hi(T, me))) return;
  const int hm = row_home(T, r);
  const uint4* xq = reinterpret_cast<const uint4*>(c->peer[hm][PX_XQ] + (long)r * H);
  uint4* dq = reinterpret_cast<uint4*>(c->xq + (long)r * H);
  for (int a = tid; a < (MX ? H / 16 : H / 32); a += blockDim.x) dq[a] = xq[a];
  // scales, ids, weights, outer scale: 23 consecutive 16-byte pieces of the home card's scale row, read by threads
  // that have no piece of the codes (296 ..: a thread's remote loads come one after the other)
  const int pt = tid - 296;
  if (pt >= 0 && pt < 23) {
    const uint4 v = reinterpret_cast<const uint4*>(c->peer[hm][PX_XS8] + (long)r * XS8W)[pt];
    if (pt < 18) reinterpret_cast<uint4*>(c->xs8 + (long)r * XS8W)[pt] = v;
    else if (pt < 20) reinterpret_cast<uint4*>(c->ids + r * TOPK)[pt - 18] = v;
    else if (pt < 22) reinterpret_cast<uint4*>(c->wts + r * TOPK)[pt - 20] = v;
    else if (!MX) c->xouter[r] = __uint_as_float(v.x);
  }
}
// rows of the o_proj and expert outputs that a short-convolution state of this pass needs (the last three rows of
// every sequence and of every page that ends in it) and that another card made; and the final hidden rows
// (576 threads: a thread's remote loads come one after the other at ~2 us each; with 256 threads a row was six in
// series, now two)
__global__ void __launch_bounds__(576) k_xhist(const PCtx* c) {
  const int T = c->h.T, me = c->card;
  const int nj = c->h.nseq + c->h.nsnap, j = blockIdx.x / 3;
  if (j >= nj || T <= REPL_T) return;
  const int* job = j < c->h.nseq ? TAB(o_seq) + j * 4 : TAB(o_snap) + (j - c->h.nseq) * 4;
  const int o = job[3] - 3 + (int)(blockIdx.x % 3);
  if (o < 0) return;
  const int r = job[2] + o;
  const int lo = max(own_lo(T, me) - 3, 0);
  if (r >= lo && r < own_hi(T, me)) return;       // made here (its own rows and the three before them)
  const int hm = row_home(T, r);
  const uint4* sa = reinterpret_cast<const uint4*>(c->peer[hm][PX_ATTN_RAW] + (long)r * H * 2);
  const uint4* sm = reinterpret_cast<const uint4*>(c->mlp_raw_p[hm] + (long)r * H);
  uint4* da = reinterpret_cast<uint4*>(c->attn_raw + (long)r * H);
  uint4* dm = reinterpret_cast<uint4*>(c->mlp_raw + (long)r * H);
  for (int a = threadIdx.x; a < H / 8; a += blockDim.x) { da[a] = sa[a]; dm[a] = sm[a]; }
}
__global__ void __launch_bounds__(576) k_xfin(const PCtx* c) {
  const int li = blockIdx.x, me = c->card, T = c->h.T;
  if (T <= REPL_T) return;
  xwait(c, XS_FIN);
  if (li >= c->h.nlog || !c->h.mode || c->h.hi != NL) return;
  const int hm = row_home(T, TAB(o_log)[li]);
  if (hm == me) return;
  const uint8_t* src = c->peer[hm][PX_HFIN];
  uint8_t* dst = reinterpret_cast<uint8_t*>(c->hfin);
  for (int a = threadIdx.x; a < H / 8; a += blockDim.x) {
    const long o = xta(li, a * 8, H / 16);
    *reinterpret_cast<uint4*>(dst + o) = *reinterpret_cast<const uint4*>(src + o);
  }
}

// ------------------------------------------------------------------------------------------------ tensor-core GEMM
// Every GEMM of the model through tcgen05: D[128 rows x N columns] += A[128 x K step] B[K step x N] with the operands
// in shared memory, where the copy engine puts them (cp.async.bulk + mbarriers), and the accumulators in tensor memory.
//   bf16  (kind::f16):                    K step 16; attention projections, o_proj, router, unembedding
//   nvfp4 (kind::mxf4nvf4, block scales): K step 64; the checkpoint's E2M1 codes and E4M3 scales as stored, times the
//                                         model-defined quantized activations (codes + block scales)
//   mxfp8 (kind::mxf8f6f4, block scales): K step 32; E4M3 codes with UE8M0 scales
// A block takes (a tile of columns, a few row groups: see tc_rg) and runs the K steps through a ring of stages in
// shared memory; a stage is G consecutive K steps: the B units and, per row group, G A units, each one bulk copy.
// One thread (warp 4) issues the copies of a stage as soon as its slot is free, one thread per row group the scale
// copies and MMAs (warps 5-7 and TC_CW4), and sixteen warps (0-3, 8-19) read D from tensor memory and write the
// results: tensor memory gives a warp the rows of its index mod 4, so four groups of four warps share the columns.
// Measured: a chain's time has a part per bulk copy (hence several K steps per stage), small MMAs take ~0.1 us each
// whatever their size, full ones ~0.07-0.15 us per 128 x 128 unit, and the read-out by four warps alone was a third
// of a GEMM kernel's time.
#define TC_RING 221184
// K steps per stage when a block has several row groups (one bulk copy per row group and stage, one for the B units):
// dense, experts. A chain's time has a part per bulk copy: measured on a 2048-row pass (dense steps, expert steps per
// stage): (1,1) 85.1 ms, (2,1) 80.3, (2,2) 75.3, (4,2) 73.6, (4,4) 73.9; a 1-row pass 15.1 -> 13.4 ms with (4,2).
#define TC_GB 4
#ifndef TC_RWG
#define TC_RWG 4
#endif
#define TC_CW4 (4 + 4 * TC_RWG)       // warp of a fourth row group's operations
#define TC_NT ((TC_CW4 + 1) * 32)    // warps: readers 0-3 and 8.., the copies 4, the tensor-core operations 5-7 and TC_CW4
#define TC_GF 2
#define TC_MAXS 8
#define TC_UB 4608
#define TC_BUB 4608
#define TC_SMEM (TC_RING + 1024)
__device__ __forceinline__ uint32_t sh(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t a, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt) : "memory"); }
__device__ __forceinline__ bool mbar_try(uint32_t a, uint32_t par) {
  uint32_t ok; asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(a), "r"(par) : "memory"); return ok; }
__device__ __forceinline__ void mbar_wait(uint32_t a, uint32_t par) { while (!mbar_try(a, par)) {} }
__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(a), "r"(bytes) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t a) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory"); }
__device__ __forceinline__ void tc_bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory"); }
__device__ __forceinline__ void tc_alloc(uint32_t smem_addr, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_addr), "r"(ncols) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void tc_dealloc(uint32_t taddr, uint32_t ncols) { asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory"); }
__device__ __forceinline__ void tc_cp(uint32_t taddr, uint64_t desc) { asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(desc) : "memory"); }
__device__ __forceinline__ void tc_mma_fp4(uint32_t td, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc, uint32_t tsfa, uint32_t tsfb) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; }"
               :: "r"(td), "l"(da), "l"(db), "r"(idesc), "r"(acc), "r"(tsfa), "r"(tsfb) : "memory"); }
__device__ __forceinline__ void tc_mma_fp8(uint32_t td, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc, uint32_t tsfa, uint32_t tsfb) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %3, [%5], [%6], p; }"
               :: "r"(td), "l"(da), "l"(db), "r"(idesc), "r"(acc), "r"(tsfa), "r"(tsfb) : "memory"); }
__device__ __forceinline__ void tc_mma_bf16(uint32_t td, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }"
               :: "r"(td), "l"(da), "l"(db), "r"(idesc), "r"(acc) : "memory"); }
__device__ __forceinline__ void tc_commit(uint32_t bar) { asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tc_ld8(uint32_t taddr, float* d) {
  uint32_t r[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]) : "r"(taddr) : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int j = 0; j < 8; j++) d[j] = __uint_as_float(r[j]);
}
__device__ __forceinline__ uint64_t tc_sdesc(uint32_t addr, uint32_t lbo, uint32_t sbo) {   // shared memory descriptor: K-major, no swizzle
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) | (1ull << 46);
}
enum { TK_FP4 = 0, TK_FP8 = 1, TK_BF16 = 2 };
enum { G_QKV = 0, G_OPROJ = 1, G_ROUTER = 2, G_UNEMBED = 3, G_W13 = 4, G_W2 = 5, G_RELB = 6 };
#define QKV_RGA 26
// chunks of RG row groups; a chunk of the q/k/v matrix never mixes rows that take different inputs
__host__ __device__ inline int tc_nchunks(int which, int glob, int nrg, int RG) {
  if (which == G_QKV) { const int nb = glob ? 3 : 2; return (QKV_RGA + RG - 1) / RG + 2 * ((nb + RG - 1) / RG); }
  return (nrg + RG - 1) / RG;
}
__device__ __forceinline__ void tc_ld16(uint32_t taddr, float* d) {
  uint32_t r[16];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                 "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]) : "r"(taddr) : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
  for (int j = 0; j < 16; j++) d[j] = __uint_as_float(r[j]);
}
__device__ __forceinline__ void tc_st16(uint32_t taddr, const float* d) {       // 16 values of this thread's row back into tensor memory
  uint32_t r[16];
#pragma unroll
  for (int j = 0; j < 16; j++) r[j] = __float_as_uint(d[j]);
  asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%16], {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15};"
               :: "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]),
                  "r"(r[8]), "r"(r[9]), "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(taddr) : "memory");
}
__device__ __forceinline__ void tc_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
// row groups per block of a GEMM: experts `rg`; the dense ones (q/k/v, o_proj, router, unembedding) one row group,
// and their K range is summed as two halves: D = (K steps of the first half) + (K steps of the second half), for every
// pass size. In a pass of more than XS_T rows one block runs both halves side by side (two accumulators of up to 256
// columns: the 512 columns of tensor memory). In a pass of up to XS_T rows the halves are two blocks on two SMs: the
// block of the first half leaves its accumulators in a scratch buffer, the block of the second half adds its own and
// writes the results. A tensor core takes its SM's operations one after the other (~0.1 us each), so a small pass's
// chain of 288-384 operations only gets shorter on another SM: 1-row pass 13.2 -> ~11 ms.
// The bias table's eight row groups go to two blocks.
#define XS_T 64
#define XS_U 994                        // scratch units (128 rows x 64 columns of fp32): q/k/v 128, o_proj 72, router 8, unembedding 786
__host__ __device__ inline int tc_rg(int which, int rg, int rgd) {
  return which <= G_UNEMBED ? 1 : ((which == G_W13 || which == G_W2) ? rg : (rg == 1 ? 1 : 4));
}
// small passes (one row group per block): the dense GEMMs run 64-row MMAs, two blocks per row group, so that twice
// as many SMs pull the weights (a 128-row unit is two 64-row units one after the other)
// (not the unembedding: its 393 row groups are more than the SMs, it is bound by operations, not by a chain's length)
__host__ __device__ inline int tc_half(int which, int rg) { return rg == 1 && which < G_UNEMBED; }
__host__ __device__ inline int tc_units(int which, int glob, int nrg, int rg, int rgd) {
  return tc_nchunks(which, glob, nrg, tc_rg(which, rg, rgd)) * (tc_half(which, rg) ? 2 : 1);
}
template <int KIND, int WHICH>
__global__ void __launch_bounds__(TC_NT) k_tc(const PCtx* c, int i) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + TC_RING);
  const uint32_t b_full = sh(bars), b_empty = sh(bars + TC_MAXS), b_done = sh(bars + 2 * TC_MAXS);
  uint32_t* tptr = reinterpret_cast<uint32_t*>(bars + 2 * TC_MAXS + 1);
  float* colf = reinterpret_cast<float*>(bars + 2 * TC_MAXS + 2);
  int* colp = reinterpret_cast<int*>(colf);
  const PLayer& L = c->L[WHICH == G_UNEMBED ? 0 : i];
  const bool expert = WHICH == G_W13 || WHICH == G_W2;
  const int me = c->card;
  // row groups per block. Experts of a small pass: one per block (36 blocks per expert: the shortest chains) while that
  // is one round of blocks; with more experts on the card than that, three per block (an 8-row pass: ~12 experts per
  // card, 450 units in three rounds -> 150 in one)
  const int RG = ((WHICH == G_W13 || WHICH == G_W2) && c->h.rg == 1) ? (c->ginfo[me * 4] * 36 <= (int)gridDim.x ? 1 : 3)
                                                                     : tc_rg(WHICH, c->h.rg, c->h.rgd);
  const bool HM = KIND == TK_BF16 && tc_half(WHICH, c->h.rg);
  const long tm = c->tmax, pm = c->pmax;
  // dense GEMMs: tiles of 256 columns (their B unit is 8192 B), at most two row groups per block: 16 KB of operands per
  // K step for the work that four row groups x 128 columns did with 20 KB
  const bool wide = KIND == TK_BF16 && WHICH != G_RELB;
  const int CW = wide ? 256 : COLT, pitch = wide ? 256 : COLT;
  const int UB = KIND == TK_BF16 ? 4096 : 4608, BUB = wide ? 8192 : (KIND == TK_BF16 ? 4096 : 4608);
  int NRG, KS, T = c->h.T;
  const uint8_t* W; const uint8_t* Xt;
  if (WHICH == G_QKV) { NRG = L.glob ? 32 : 30; KS = 288; W = L.wqkv; Xt = reinterpret_cast<const uint8_t*>(c->xin); }
  else if (WHICH == G_OPROJ) { NRG = 36; KS = 384; W = L.wo; Xt = reinterpret_cast<const uint8_t*>(c->xo); }
  else if (WHICH == G_ROUTER) { NRG = 3; KS = 288; W = L.wr; Xt = reinterpret_cast<const uint8_t*>(c->mlp_inx); }
  else if (WHICH == G_UNEMBED) { NRG = c->un_strips; KS = 288; W = c->unembed; Xt = reinterpret_cast<const uint8_t*>(c->hfin); T = c->h.nlog; }
  else if (WHICH == G_RELB) { NRG = 8; KS = 1; W = L.relp; Xt = c->rt; T = c->h.T * NQL; }
  else if (WHICH == G_W13) { NRG = 36; KS = KIND == TK_FP4 ? 72 : 144; W = L.w13; Xt = c->xt13; }
  else { NRG = 36; KS = KIND == TK_FP4 ? 36 : 72; W = L.w2; Xt = c->xt2; }
  // stage geometry: G K steps; B slots at 0 (bsl bytes each), then the A slots (asl bytes each): row group r, step g
  // at aoff + (r * G + g) * asl. Small passes (at most 16 columns per tile, one row group per block) use short
  // slots, so that the ring holds 8 stages: the chain is then bound by what is in flight, not by slot count.
  const bool small = c->h.rg == 1 && WHICH != G_RELB;
  const int G = KS == 1 ? 1 : (RG == 1 ? (KIND == TK_BF16 ? 4 : 2) : (KIND == TK_BF16 ? TC_GB : TC_GF));
  const uint32_t bsl = small ? (KIND == TK_BF16 ? 512 : 1024) : BUB, sfo = small ? 512 : 4096;   // B slot, offset of its scale atom
  const uint32_t asl = HM ? 2048 : UB;
  const bool XS = wide && c->h.T <= XS_T && WHICH != G_UNEMBED;       // the K range as two halves in two blocks (this block: one of them)
  const bool SEG = wide && !XS;                 // ... or as two halves in this block, each with its own B and A slots and accumulator
  const uint32_t aoff = (SEG ? 2 : 1) * G * bsl, stg = SEG ? 2 * G * (bsl + asl) : G * (bsl + RG * asl);
  const uint32_t S = min((uint32_t)TC_MAXS, (uint32_t)TC_RING / stg);
  const int nstage = (wide ? KS / 2 : KS) / G;
  // o_proj and the router run on a row's home card: this card's rows (o_proj: and the three before them) are a window
  // [wlo, whi) of the pass's columns
  const bool WND = WHICH == G_OPROJ || WHICH == G_ROUTER;
  const int wlo = WND ? max(own_lo(T, me) - (WHICH == G_OPROJ ? 3 : 0), 0) : 0, whi = WND ? own_hi(T, me) : T;
  const int t0 = WND ? wlo / CW : 0;
  const int ntiles = expert ? c->ginfo[me * 4] : (whi > wlo ? (whi - 1) / CW - t0 + 1 : 0);
  const int nch = tc_nchunks(WHICH, WHICH == G_QKV ? L.glob : 0, NRG, RG) * (HM ? 2 : 1);
  const long nun = (long)ntiles * nch;         // units; with the halves in two blocks: units 0 .. nun-1 are the first halves
  const long total = XS ? 2 * nun : nun;
  if (blockIdx.x >= total || (WHICH == G_UNEMBED && !c->h.mode)) return;       // nothing for this block (fixed-grid launches)
  // one consumer thread per row group (warps 5 .. 5 + RG - 1): the scale copies and the MMA of a row group are a
  // serial path of ~0.1 us per operation, so the row groups of a stage are issued side by side. A stage's slot is
  // free, and the block-unit done, when every consumer has said so: those barriers count RG arrivals.
  const int RGc = SEG ? 2 : RG;
  if (tid < TC_MAXS) mbar_init(sh(bars + tid), 1);
  else if (tid < 2 * TC_MAXS + 1) mbar_init(sh(bars + tid), RGc);
  __syncthreads();
  if (w == 0) tc_alloc(sh(tptr), 512);
  __syncthreads();
  fence_async();
  const uint32_t t = *tptr;
  uint32_t slot = 0, par = (w == 4) ? 1 : 0, dpar = 0;      // ring position of the producer / consumer; parity of `done`
  for (long bu = blockIdx.x; bu < total; bu += gridDim.x) {
    const int xseg = XS && bu >= nun;      // this unit is the second half of the K range
    const long bu2 = xseg ? bu - nun : bu;
    const int tile = (expert ? 0 : t0) + (int)(bu2 / nch), half = HM ? (int)(bu2 % nch) & 1 : 0, ch = (int)(bu2 % nch) >> (HM ? 1 : 0);
    const long koff = xseg ? KS / 2 : 0;   // first K step of this unit's chain
    int rg0, nrg, sel = 0;
    if (WHICH == G_QKV) {
      const int nA = (QKV_RGA + RG - 1) / RG, nb = L.glob ? 3 : 2, nB = (nb + RG - 1) / RG;
      if (ch < nA) { rg0 = ch * RG; nrg = min(RG, QKV_RGA - rg0); }
      else if (ch < nA + nB) { const int o = (ch - nA) * RG; rg0 = QKV_RGA + o; nrg = min(RG, nb - o); sel = L.glob ? 1 : 0; }
      else { const int o = (ch - nA - nB) * RG; rg0 = QKV_RGA + nb + o; nrg = min(RG, nb - o); sel = L.glob ? 2 : 0; }
    } else { rg0 = ch * RG; nrg = min(RG, NRG - rg0); }
    int ncols, e = 0, p0 = 0;
    const int g0 = WND ? max(wlo - (t0 + (int)(bu2 / nch)) * CW, 0) >> 3 : 0;      // first column group of the window in this tile
    if (expert) {
      const int* it = c->items + ((long)me * pm + tile) * 4;
      e = it[0]; p0 = it[1]; ncols = it[2];
      const int eg = it[3];
      if (KIND == TK_FP4)
        for (int k = tid; k < COLT; k += blockDim.x) {
          const int pair = p0 + min(k, ncols - 1);
          colf[k] = (WHICH == G_W2 ? c->aouter[pair] : c->xouter[c->ptok[me * pm + pair]]) * (WHICH == G_W2 ? L.f2 : L.f13)[eg];
        }
    } else ncols = min(CW, whi - tile * CW) - g0 * 8;
    if (WHICH == G_RELB)                  // position of each column's row: the bias of distance d goes to slot (position - d) mod 1024
      for (int k = tid; k < COLT; k += blockDim.x) colp[k] = TAB(o_pos)[min(tile * COLT + k, T - 1) / NQL];
    const int ncg = (ncols + 7) >> 3;
    const uint8_t* Wb = W + ((long)(expert ? e * NRG : 0) + rg0) * KS * UB;
    const uint8_t* Xb = Xt + (WHICH == G_QKV ? (long)sel * tm * H * 2 : 0) + (long)tile * KS * BUB;
    __syncthreads();
    if (w == 4 && lane == 0) {            // producer: the next stage's operands, as soon as its slot is free
      const uint32_t bcp = ncg * 256;       // bytes of a B unit that hold columns
      const bool whole = !small;            // large passes: the stage's B units in one copy, whatever the tile's columns
      const uint32_t bbytes = whole ? G * BUB : G * (bcp + (KIND == TK_BF16 ? 0 : 512));
      for (int st = 0; st < nstage; st++) {
        mbar_wait(b_empty + 8 * slot, par);
        const uint32_t dst = sh(sm) + slot * stg, bf_ = b_full + 8 * slot;
        if (SEG) {                          // the stage's K steps of both halves of the K range (one row group)
          mbar_expect(bf_, 2 * (bbytes + (HM ? G * 2048 : G * UB)));
          for (int sg = 0; sg < 2; sg++) {
            const long k0 = (long)sg * (KS / 2) + (long)st * G;
            const uint8_t* xs = Xb + k0 * BUB;
            if (whole) tc_bulk(dst + sg * G * bsl, xs, G * BUB, bf_);
            else for (int g = 0; g < G; g++) tc_bulk(dst + (sg * G + g) * bsl, xs + g * BUB + g0 * 256, bcp, bf_);
            if (HM) for (int g = 0; g < G; g++) tc_bulk(dst + aoff + (sg * G + g) * asl, Wb + (k0 + g) * UB + half * 2048, 2048, bf_);
            else tc_bulk(dst + aoff + sg * G * asl, Wb + k0 * UB, G * UB, bf_);
          }
        } else {
          mbar_expect(bf_, bbytes + (HM ? G * 2048 : nrg * G * UB));
          const long k0 = koff + (long)st * G;
          const uint8_t* xs = Xb + k0 * BUB;
          if (whole) tc_bulk(dst, xs, G * BUB, bf_);
          else
            for (int g = 0; g < G; g++) {
              tc_bulk(dst + g * bsl, xs + g * BUB + g0 * 256, bcp, bf_);
              if (KIND != TK_BF16) tc_bulk(dst + g * bsl + sfo, xs + g * BUB + 4096, 512, bf_);
            }
          if (HM) for (int g = 0; g < G; g++) tc_bulk(dst + aoff + g * asl, Wb + (k0 + g) * UB + half * 2048, 2048, bf_);
          else for (int r = 0; r < nrg; r++) tc_bulk(dst + aoff + r * G * asl, Wb + ((long)r * KS + k0) * UB, G * UB, bf_);
        }
        if (++slot == S) { slot = 0; par ^= 1; }
      }
    } else if (((w >= 5 && w < 8) || w == TC_CW4) && (w == TC_CW4 ? 3 : w - 5) < RGc && lane == 0) {     // consumer of a row group (or of a half of the K range): scale copies and MMAs
      const int r = w == TC_CW4 ? 3 : w - 5;
      const int ncon = SEG ? 2 : nrg;
      const uint32_t t_sfa = t + 384 + 4 * r, t_sfb = t + 396 + 4 * r;
      tc_fence_after();
      const uint32_t idesc = KIND == TK_FP4 ? ((1u << 7) | (1u << 10) | ((uint32_t)ncg << 17) | (8u << 24))
                           : KIND == TK_FP8 ? ((1u << 23) | ((uint32_t)ncg << 17) | (8u << 24))
                           : ((1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)ncg << 17) | ((HM ? 4u : 8u) << 24));
      const uint64_t hi = tc_sdesc(0, 128, 256), his = tc_sdesc(0, 0, 128);
      uint32_t accf = 0;
      for (int st = 0; st < nstage; st++) {
        mbar_wait(b_full + 8 * slot, par);
        if (r < ncon) {
          const uint32_t base = sh(sm) + slot * stg;
          for (int g = 0; g < G; g++) {
            const uint32_t bb = (base + ((SEG ? r * G : 0) + g) * bsl + (small ? 0 : g0 * 256)) >> 4;
            const uint64_t db = hi | bb;
            const uint32_t a = (base + aoff + (r * G + g) * asl) >> 4;
            if (KIND == TK_FP4) { tc_cp(t_sfb, his | (bb + (sfo >> 4))); tc_cp(t_sfa, his | (a + 256)); tc_mma_fp4(t + r * pitch, hi | a, db, idesc, accf, t_sfa, t_sfb); }
            else if (KIND == TK_FP8) { tc_cp(t_sfb, his | (bb + (sfo >> 4))); tc_cp(t_sfa, his | (a + 256)); tc_mma_fp8(t + r * pitch, hi | a, db, idesc, accf, t_sfa, t_sfb); }
            else tc_mma_bf16(t + r * pitch, hi | a, db, idesc, accf);
            accf = 1;
          }
          tc_commit(b_empty + 8 * slot);
        } else mbar_arrive(b_empty + 8 * slot);
        if (++slot == S) { slot = 0; par ^= 1; }
      }
      if (r < ncon) tc_commit(b_done); else mbar_arrive(b_done);
    }
    if (w < 4 || (w >= 8 && w < 4 + 4 * TC_RWG)) {    // readers: D of this block-unit
      // TC_RWG groups of four warps (0-3, 8-11, 12-15, ...) read the same 128 rows (tensor memory gives a warp the rows
      // of its index mod 4); group g takes the chunks of 16 columns with index g mod TC_RWG
      const int lw = w & 3, rh = w < 4 ? 0 : (w - 4) >> 2;
      mbar_wait(b_done, dpar);
      tc_fence_after();
      // halves in two blocks: where the first half's accumulators are (this unit's place in the scratch), and the word
      // that says they are there for this pass and layer
      const long xu = (WHICH == G_QKV ? 0 : WHICH == G_OPROJ ? 128 : WHICH == G_ROUTER ? 200 : 208) + bu2;
      const unsigned xtag = ((unsigned)c->h.tag << 6) | (unsigned)(i + 1);
      if (xseg) { unsigned long long spins = 0; while (ld_acquire_sys(&c->kflag[xu]) != xtag) { if (++spins > (1ull << 31)) { c->err[2] = 1 + WHICH; break; } } }
      // a 64-row MMA keeps its rows in lanes 0-15 of the four warps
      const int rin = HM ? lw * 16 + lane + half * 64 : lw * 32 + lane;
      for (int r = 0; r < nrg; r++) {
        const int rg = rg0 + r;
        for (int cq = rh; cq * 2 < ncg; cq += TC_RWG) {
          float d[16];
          tc_ld16(t + r * pitch + cq * 16 + ((uint32_t)(lw * 32) << 16), d);
          if (SEG) {                        // the two halves of the K range, in this block
            float d2[16];
            tc_ld16(t + pitch + cq * 16 + ((uint32_t)(lw * 32) << 16), d2);
#pragma unroll
            for (int j = 0; j < 16; j++) d[j] = __fadd_rn(d[j], d2[j]);
          } else if (XS) {                  // ... in two blocks: the first leaves its half, the second adds its own
            float4* xp = reinterpret_cast<float4*>(c->kscr + ((xu * 128 + lw * 32 + lane) * 64 + cq * 16));
            if (!xseg) {
#pragma unroll
              for (int j = 0; j < 4; j++) xp[j] = make_float4(d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3]);
              continue;
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
              const float4 x = xp[j];
              d[4 * j] = __fadd_rn(x.x, d[4 * j]); d[4 * j + 1] = __fadd_rn(x.y, d[4 * j + 1]); d[4 * j + 2] = __fadd_rn(x.z, d[4 * j + 2]); d[4 * j + 3] = __fadd_rn(x.w, d[4 * j + 3]);
            }
          }
#pragma unroll
          for (int j = 0; j < 16; j++) {
            const int col = cq * 16 + j;
            const bool dead = HM && lane >= 16;       // a 64-row MMA has no row in this lane
            if (col >= ncols || (dead && WHICH != G_UNEMBED) || (WND && tile * CW + g0 * 8 + col < wlo)) continue;
            float v = d[j];
            if (WHICH == G_W13 || WHICH == G_W2) {
              if (KIND == TK_FP4) v *= colf[col];
              (WHICH == G_W2 ? c->pairout : c->h13)[(long)(p0 + col) * H + rg * 128 + rin] = tobf(v);
            } else {
              const int tok = tile * CW + g0 * 8 + col;
              if (WHICH == G_QKV) {
                const int nb = L.glob ? 3 : 2;
                int n, lim;
                if (rg < QKV_RGA) { n = rg * 128 + rin; lim = QK; }
                else if (rg < QKV_RGA + nb) { n = (rg - QKV_RGA) * 128 + rin; lim = L.glob ? 272 : 256; n += QK; lim += QK; }
                else { n = (rg - QKV_RGA - nb) * 128 + rin; lim = L.glob ? 272 : 256; n += QK + lim; lim = QK + 2 * lim; }
                if (n < lim) c->qkvo[(long)tok * 3824 + n] = tobf(v);
              } else if (WHICH == G_OPROJ) c->attn_raw[(long)tok * H + rg * 128 + rin] = tobf(v);
              else if (WHICH == G_RELB) c->relb[(long)tok * RELEXT + ((colp[col] - (rg * 128 + rin)) & (RELEXT - 1))] = tobf(v);   // slot = key position mod 1024
              else if (WHICH == G_ROUTER) c->rlog[(long)tok * NE + rg * 128 + rin] = v;
              else {
                const int vid = c->un_row0 + rg * 128 + rin;
                const bool real = !dead && vid < VOCAB_REAL && rg * 128 + rin < 201024 / NCARD;
                const float z = 30.0f * tanhf(v / 30.0f);
                if (c->h.mode == 1) { if (real) c->h.logits[(long)tok * VOCAB_REAL + vid] = z; }
                else {
                  // candidate of this column: the warp's best (soft-capped logit, then the lower token id), one atomic
                  unsigned bits = __float_as_uint(z);
                  bits = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
                  unsigned long long key = real ? (((unsigned long long)bits << 32) | (0xFFFFFFFFu - (unsigned)vid)) : 0ull;
#pragma unroll
                  for (int off = 16; off; off >>= 1) { const unsigned long long o2 = __shfl_xor_sync(0xffffffff, key, off); key = o2 > key ? o2 : key; }
                  if (lane == 0) atomicMax(&c->cand[tok], key);
                }
              }
            }
          }
        }
      }
      tc_fence_before();
      dpar ^= 1;
    }
    __syncthreads();
    // the first half of a K range whose second half is another block's: its accumulators are in the scratch now
    if (XS && !xseg && tid == 0)
      st_release_sys(&c->kflag[(WHICH == G_QKV ? 0 : WHICH == G_OPROJ ? 128 : WHICH == G_ROUTER ? 200 : 208) + bu2], ((unsigned)c->h.tag << 6) | (unsigned)(i + 1));
  }
  if (w == 0) tc_dealloc(t, 512);
}

// ------------------------------------------------------------------------------------------------ K/V, q, relative bias
// K page: the B operand of the scores, per KV head and half page: 8 K steps (16 features) x 256 keys
__device__ __forceinline__ long kaddr(int pos, int kv, int f) {   // byte offset of K[pos][kv][f] inside its page
  const int pp = pos & (PAGE - 1), n = pp & 255;
  return (long)kv * 131072 + (pp >> 8) * 65536 + (f >> 4) * 8192 + (n >> 3) * 256 + ((f >> 3) & 1) * 128 + (n & 7) * 16 + (f & 7) * 2;
}
// V page: the B operand of the value product, per KV head: 32 K steps (16 keys) x 128 features
__device__ __forceinline__ long vaddr(int pos, int kv, int f) {
  const int pp = pos & (PAGE - 1), k = pp & 15;
  return (long)kv * 131072 + (pp >> 4) * 4096 + (f >> 3) * 256 + (k >> 3) * 128 + (f & 7) * 16 + (k & 7) * 2;
}
// raw short-convolution input of row r at lag j (0 = the row itself): the pass's own rows, then the slot's state
__device__ __forceinline__ float raw_at(const bf16* rows, long stride, int r, int j, int back, const bf16* hist, int ch, int hch) {
  if (j <= back) return bf(rows[(long)(r - j) * stride + ch]);
  return bf(hist[(long)(3 - (j - back)) * HS + hch]);
}

__global__ void __launch_bounds__(NTB) k_prep(const PCtx* c, int i) {
  if (blockIdx.x >= c->h.T) return;       // fixed-grid launches: no such row in this pass
  __shared__ float red[8 * 8];
  const int r = blockIdx.x, tid = threadIdx.x, w = tid >> 5;
  const PLayer& L = c->L[i];
  const int slot = TAB(o_slot)[r], pos = TAB(o_pos)[r], back = TAB(o_back)[r];
  const bf16* q = c->qkvo + (long)r * 3824;
  const bf16* hist = c->hist + ((long)slot * NL + i) * 3 * HS;
  const int V0 = QK + (L.glob ? 272 : 256);
  const int page = TAB(o_pt)[slot * MAXPG + (pos >> 9)];
  float tau = 1.0f;
  if (L.glob) tau = 1.0f + 0.1f * logf(fmaxf(((float)pos + 1.0f) / 128000.0f, 1.0f));
  // K and V: thread = channel (2 heads x 128)
  float y[2], ssk[2];
#pragma unroll
  for (int kvsel = 0; kvsel < 2; kvsel++) {
    const int base = kvsel ? V0 : QK, hoff = kvsel ? 256 : 0, head = tid >> 7;
    const bf16* sc = kvsel ? L.vsc : L.ksc;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; j++) x[j] = raw_at(c->qkvo + base, 3824, r, j, back, hist, tid, hoff + tid);
    float v;
    if (L.glob) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) acc = acc + x[j] * (bf(sc[j * 256 + tid]) + bf(q[base + 256 + head * 4 + j]));
      v = (acc + x[0]) * 0.3333333333333333f;
    } else {
      float acc = 0.f;
#pragma unroll
      for (int tap = 0; tap < 4; tap++) acc = acc + x[3 - tap] * bf(sc[tid * 4 + tap]);
      v = (acc + x[0]) * 0.4472135954999579f;
    }
    y[kvsel] = bf(tobf(v));
    float ss = y[kvsel] * y[kvsel];
#pragma unroll
    for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
    ssk[kvsel] = ss;
  }
  if ((tid & 31) == 0) { red[w] = ssk[0]; red[8 + w] = ssk[1]; }
  // q: thread = (head pair hp, feature) for six head pairs
  float qv[NQL / 2], qs[NQL / 2];
#pragma unroll
  for (int hp = 0; hp < NQL / 2; hp++) {
    const int head = hp * 2 + (tid >> 7), f = tid & 127;
    qv[hp] = bf(q[head * HD + f]) * bf(L.qscale[head * HD + f]);
    float ss = qv[hp] * qv[hp];
#pragma unroll
    for (int off = 16; off; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
    qs[hp] = ss;
  }
  if ((tid & 31) == 0) {
#pragma unroll
    for (int hp = 0; hp < NQL / 2; hp++) red[16 + hp * 8 + w] = qs[hp];
  }
  __syncthreads();
  {
    const int hh = tid >> 7, head = hh, f = tid & 127;
    const float sk = (red[hh * 4] + red[hh * 4 + 1]) + (red[hh * 4 + 2] + red[hh * 4 + 3]);
    const float sv = (red[8 + hh * 4] + red[8 + hh * 4 + 1]) + (red[8 + hh * 4 + 2] + red[8 + hh * 4 + 3]);
    const float fac = fminf(22.627416997969522f / (sqrtf(sk) + 1e-6f), 1.0f);
    *reinterpret_cast<bf16*>(L.kpg + (long)page * 262144 + kaddr(pos, head, f)) = tobf(y[0] * fac);
    *reinterpret_cast<bf16*>(L.vpg + (long)page * 262144 + vaddr(pos, head, f)) = tobf(y[1] * rsqrtf(sv / 128.0f + c->eps));
#pragma unroll
    for (int hp = 0; hp < NQL / 2; hp++) {
      const float* rq = red + 16 + hp * 8 + hh * 4;
      const float ss = (rq[0] + rq[1]) + (rq[2] + rq[3]);
      bf16 o = tobf(qv[hp] * fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f));
      if (L.glob) o = tobf(bf(o) * tau);
      c->qn[((long)r * NQL + hp * 2 + hh) * HD + f] = o;
    }
  }
  // r: 12 heads x 16, clipped, as column (row * 12 + head) of the bias GEMM's input
  if (tid < NQL * DREL) {
    const int head = tid >> 4, k = tid & 15, col = r * NQL + head, n = col & 127;
    const float v = bf(q[QR + tid]) * bf(L.rscale[tid]);
    float ss = v * v;
#pragma unroll
    for (int off = 8; off; off >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, off);
    bf16 o = tobf(v * fminf(8.0f / (sqrtf(ss) + 1e-6f), 1.0f));
    if (L.glob) o = tobf(bf(o) * tau);
    *reinterpret_cast<bf16*>(c->rt + (long)(col >> 7) * 4096 + (n >> 3) * 256 + (k >> 3) * 128 + (n & 7) * 16 + (k & 7) * 2) = o;
  }
}

// ------------------------------------------------------------------------------------------------ attention
// A unit = (group of up to 21 consecutive rows of one sequence, KV head, one page of keys). The group's q vectors (rows
// x the six q heads of the KV head: up to 126) are the rows of a tensor-core tile, one block per unit:
//   per half page (256 keys):  S[q][key] = Q K^T        8 MMAs; the K page is stored as the B operand
//                              the q's thread reads its 256 scores from tensor memory twice: scale, relative bias, mask,
//                              maximum; then exp(s - max) rounded to bf16, written as the A operand of
//                              O[q][128] = P V          16 MMAs; the V page is stored as the B operand
//   the two halves' (max, sum, O) are combined into the unit's record; k_attn_fold sums a row's pages.
// Every q's result depends only on its own keys in this fixed order: independent of the grouping and of the pass.
// group record: row0, n, slot, pos0, pgL0, npgL, ubaseL, ubaseG  (global: pages 0 .. npgG-1 with npgG = (pos0+n-1)/512+1)
#define GRPW 8
#define AT_Q 0
#define AT_K 32768
#define AT_V (AT_K + 65536)
#define AT_P (AT_V + 65536)
#define AT_BARS (AT_P + 65536)
#define AT_PM (AT_BARS + 128)
#define AT_SMEM (AT_PM + 2048)
#define AT_SW 16         // softmax warps: warp w reads the q vectors (w & 3) * 32 .. and the columns (w >> 2) * 64 ..
// 16 scores of a q (columns cq*16 ..): scaled, plus the bias where the key is within its extent, masked to -1e30
__device__ __forceinline__ void at_scores(uint32_t taddr, const uint4* bp, int cq, int cb, int lo, int hi, float* v) {
  float d[16];
  tc_ld16(taddr + cq * 16, d);
  const uint4 b0 = bp[cq * 2], b1 = bp[cq * 2 + 1];
  const uint32_t bw[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
  for (int e = 0; e < 16; e++) {
    const int col = cq * 16 + e;
    const float bias = __uint_as_float((e & 1) ? (bw[e >> 1] & 0xffff0000u) : (bw[e >> 1] << 16));
    float sc = __fmul_rn(d[e], 0.08838834764831845f);
    sc = col >= cb ? __fadd_rn(sc, bias) : sc;
    v[e] = (col >= lo && col <= hi) ? sc : -1.0e30f;
  }
}
__global__ void __launch_bounds__((AT_SW + 2) * 32) k_attn(const PCtx* c, int i) {
  extern __shared__ __align__(1024) uint8_t sm[];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const PLayer& L = c->L[i];
  const int total = L.glob ? c->h.unitsG : c->h.unitsL;
  if (blockIdx.x >= total) return;        // nothing for this block (fixed-grid launches)
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + AT_BARS);
  // barriers: Q written, K / V half page copied, scores done, maxima exchanged, P written, value product done
  const uint32_t b_q = sh(bars), b_k = sh(bars + 1), b_v = sh(bars + 2), b_s = sh(bars + 3), b_p = sh(bars + 4), b_o = sh(bars + 5), b_m = sh(bars + 6);
  uint32_t* tptr = reinterpret_cast<uint32_t*>(bars + 7);
  float* pm = reinterpret_cast<float*>(sm + AT_PM);       // [4 column quarters][128 q]: a half page's partial maxima
  float* ql = reinterpret_cast<float*>(sm + AT_Q);        // [2 halves][4 quarters][128 q]: partial sums (after the scores)
  if (tid == 0) { mbar_init(b_q, AT_SW); mbar_init(b_k, 1); mbar_init(b_v, 1); mbar_init(b_s, 1); mbar_init(b_p, AT_SW); mbar_init(b_o, 1); mbar_init(b_m, AT_SW); }
  __syncthreads();
  if (w == 0) tc_alloc(sh(tptr), 512);
  __syncthreads();
  fence_async();
  const uint32_t t = *tptr;              // tensor memory: scores at columns 0-255, the halves' outputs at 256 and 384
  const int* grp = TAB(o_grp);
  const int ngrp = c->h.ngrp;
  uint32_t uph = 0, tph = 0, mph = 0;    // units / half pages / exchanges this thread has been through (barrier phases)
  for (int u = blockIdx.x; u < total; u += gridDim.x) {
    int lo = 0, hi = ngrp - 1;            // group of this unit: the last group whose base is <= u
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (grp[mid * GRPW + (L.glob ? 7 : 6)] <= u) lo = mid; else hi = mid - 1;
    }
    const int* G = grp + lo * GRPW;
    const int row0 = G[0], n = G[1], slot = G[2], pos0 = G[3];
    const int pg0 = L.glob ? 0 : G[4];
    const int rel = u - G[L.glob ? 7 : 6];
    const int kv = rel & 1, pi = pg0 + (rel >> 1);
    const int page = TAB(o_pt)[slot * MAXPG + pi];
    const uint8_t* kp = L.kpg + (long)page * 262144 + (long)kv * 131072;
    const uint8_t* vp = L.vpg + (long)page * 262144 + (long)kv * 131072;
    const int plast = pos0 + n - 1;
    bool need[2];                         // a half page with a key some row of the group attends to
#pragma unroll
    for (int tl = 0; tl < 2; tl++) {
      const int j0 = pi * PAGE + tl * 256;
      need[tl] = j0 <= plast && (L.glob || j0 + 255 >= pos0 - (WIN - 1));
    }
    if (w < AT_SW) {                      // thread = (q vector: row of the group x q head of the KV head, column quarter)
      const int lw = w & 3, cg = w >> 2;
      const int qi = lw * 32 + lane, nq = n * 6;
      const bool act = qi < nq, wact = lw * 32 < nq;
      const int ri = act ? qi / 6 : 0, hq = act ? qi % 6 : 0;
      const int r = row0 + ri, p = pos0 + ri, head = kv * 6 + hq;
      const uint32_t tl32 = t + ((uint32_t)(lw * 32) << 16);
      {                                   // the group's q vectors as the A operand of the scores
        const uint4* qsrc = reinterpret_cast<const uint4*>(c->qn + ((long)r * NQL + head) * HD);
        uint8_t* qd = sm + AT_Q + (qi >> 3) * 256 + (qi & 7) * 16;
#pragma unroll
        for (int a = cg * 4; a < cg * 4 + 4; a++) *reinterpret_cast<uint4*>(qd + (a >> 1) * 4096 + (a & 1) * 128) = act ? qsrc[a] : make_uint4(0u, 0u, 0u, 0u);
      }
      fence_async();
      if (lane == 0) mbar_arrive(b_q);
      float mt[2] = {-1.0e30f, -1.0e30f}, lp[2] = {0.f, 0.f};
      bool first = true;
#pragma unroll
      for (int tl = 0; tl < 2; tl++) {
        if (!need[tl]) continue;
        const int j0 = pi * PAGE + tl * 256;
        const int hi_c = act ? p - j0 : -1;                    // last column this q may see
        const int cb = p - (RELEXT - 1) - j0;                  // first column within the bias extent (= the local window)
        const int lo_c = L.glob ? 0 : max(cb, 0);
        const uint4* bp = reinterpret_cast<const uint4*>(c->relb + ((long)r * NQL + head) * RELEXT + (j0 & (RELEXT - 1)));
        // a half page beyond the bias extent of every row of the group (global layers): every key is seen, none has
        // a bias: the same values without the bias load, the selects and the mask (the scaling is monotone: the
        // maximum of the scaled scores is the scaled maximum)
        const bool plain = L.glob && j0 + 255 < pos0 - (RELEXT - 1);
        mbar_wait(b_s, tph & 1);
        float m = -1.0e30f, l = 0.f;
        if (wact) {
          tc_fence_after();
          if (plain) {
            for (int cq = cg * 4; cq < cg * 4 + 4; cq++) {
              float d[16];
              tc_ld16(tl32 + cq * 16, d);
#pragma unroll
              for (int e = 0; e < 16; e++) m = fmaxf(m, d[e]);
            }
            m = act ? __fmul_rn(m, 0.08838834764831845f) : -1.0e30f;
          } else {
            // the scaled, biased, masked scores go back into tensor memory: the second pass reads them as they are
            for (int cq = cg * 4; cq < cg * 4 + 4; cq++) {
              float v[16];
              at_scores(tl32, bp, cq, cb, lo_c, hi_c, v);
#pragma unroll
              for (int e = 0; e < 16; e++) m = fmaxf(m, v[e]);
              tc_st16(tl32 + cq * 16, v);
            }
            tc_wait_st();
          }
          pm[cg * 128 + qi] = m;
        }
        if (lane == 0) mbar_arrive(b_m);
        mbar_wait(b_m, mph & 1); mph++;
        if (!first) mbar_wait(b_o, (tph - 1) & 1);             // the previous half's value product has read P
        if (wact) {
          m = fmaxf(fmaxf(pm[qi], pm[128 + qi]), fmaxf(pm[256 + qi], pm[384 + qi]));
          // a masked score is -1e30: exp(-1e30 - m) is exactly 0 for a finite m; a q without any key in this half page
          // (m = -1e30) gets its zeros from an infinite reference
          const float mref = m > -1.0e29f ? m : __int_as_float(0x7f800000);
          uint8_t* pd = sm + AT_P + (qi >> 3) * 256 + (qi & 7) * 16;
          for (int cq = cg * 4; cq < cg * 4 + 4; cq++) {
            float v[16];
            if (plain) {
              tc_ld16(tl32 + cq * 16, v);
#pragma unroll
              for (int e = 0; e < 16; e++) v[e] = act ? __fmul_rn(v[e], 0.08838834764831845f) : -1.0e30f;
            } else tc_ld16(tl32 + cq * 16, v);
            uint32_t pk[8];
            float ls = 0.f;
#pragma unroll
            for (int e = 0; e < 8; e++) {
              const float pa = expf(v[2 * e] - mref);
              const float pb = expf(v[2 * e + 1] - mref);
              const __nv_bfloat162 x = __floats2bfloat162_rn(pa, pb);
              ls = __fadd_rn(__fadd_rn(ls, __bfloat162float(x.x)), __bfloat162float(x.y));
              pk[e] = *reinterpret_cast<const uint32_t*>(&x);
            }
            *reinterpret_cast<uint4*>(pd + cq * 4096) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
            *reinterpret_cast<uint4*>(pd + cq * 4096 + 128) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
            l = __fadd_rn(l, ls);
          }
          tc_fence_before();
        }
        fence_async();
        if (lane == 0) mbar_arrive(b_p);
        mt[tl] = m; lp[tl] = l; first = false; tph++;
      }
      // the unit's record: the two halves relative to their common maximum
      mbar_wait(b_o, (tph - 1) & 1);
      if (wact) { ql[cg * 128 + qi] = lp[0]; ql[(4 + cg) * 128 + qi] = lp[1]; }
      if (lane == 0) mbar_arrive(b_m);
      mbar_wait(b_m, mph & 1); mph++;
      if (wact) {
        tc_fence_after();
        const float m = fmaxf(mt[0], mt[1]);
        const float f0 = expf(mt[0] - m), f1 = expf(mt[1] - m);
        float* rec = c->apart + ((long)u * 128 + qi) * APW;
        for (int cq = cg * 2; cq < cg * 2 + 2; cq++) {
          float a[16], b[16], o[16];
          if (need[0]) tc_ld16(tl32 + 256 + cq * 16, a);
          if (need[1]) tc_ld16(tl32 + 384 + cq * 16, b);
#pragma unroll
          for (int e = 0; e < 16; e++) o[e] = __fadd_rn(need[0] ? __fmul_rn(a[e], f0) : 0.f, need[1] ? __fmul_rn(b[e], f1) : 0.f);
          if (act) {
#pragma unroll
            for (int e = 0; e < 4; e++) *reinterpret_cast<float4*>(rec + 4 + cq * 16 + e * 4) = make_float4(o[e * 4], o[e * 4 + 1], o[e * 4 + 2], o[e * 4 + 3]);
          }
        }
        if (act && cg == 0) {
          const float l0 = __fadd_rn(__fadd_rn(__fadd_rn(ql[qi], ql[128 + qi]), ql[256 + qi]), ql[384 + qi]);
          const float l1 = __fadd_rn(__fadd_rn(__fadd_rn(ql[512 + qi], ql[640 + qi]), ql[768 + qi]), ql[896 + qi]);
          *reinterpret_cast<float2*>(rec) = make_float2(m, __fadd_rn(__fmul_rn(l0, f0), __fmul_rn(l1, f1)));
        }
        tc_fence_before();
      }
    } else if (w == AT_SW && lane == 0) {     // the copies: a half's K and V as soon as the previous half has used the buffer
      bool first = true;
#pragma unroll
      for (int tl = 0; tl < 2; tl++) {
        if (!need[tl]) continue;
        if (!first) mbar_wait(b_s, (tph - 1) & 1);
        mbar_expect(b_k, 65536);
        tc_bulk(sh(sm) + AT_K, kp + tl * 65536, 65536, b_k);
        if (!first) mbar_wait(b_o, (tph - 1) & 1);
        mbar_expect(b_v, 65536);
        tc_bulk(sh(sm) + AT_V, vp + tl * 65536, 65536, b_v);
        first = false; tph++;
      }
    } else if (w == AT_SW + 1 && lane == 0) { // the tensor-core operations
      tc_fence_after();
      const uint64_t dsc = tc_sdesc(0, 128, 256);
      const uint32_t id_s = (1u << 4) | (1u << 7) | (1u << 10) | (32u << 17) | (8u << 24);     // 128 x 256, bf16 x bf16 -> f32
      const uint32_t id_o = (1u << 4) | (1u << 7) | (1u << 10) | (16u << 17) | (8u << 24);     // 128 x 128
      const uint32_t base = sh(sm);
      mbar_wait(b_q, uph & 1);
#pragma unroll
      for (int tl = 0; tl < 2; tl++) {
        if (!need[tl]) continue;
        mbar_wait(b_k, tph & 1);
        for (int ks = 0; ks < 8; ks++) tc_mma_bf16(t, dsc | ((base + AT_Q + ks * 4096) >> 4), dsc | ((base + AT_K + ks * 8192) >> 4), id_s, ks);
        tc_commit(b_s);
        mbar_wait(b_p, tph & 1);
        mbar_wait(b_v, tph & 1);
        for (int ks = 0; ks < 16; ks++) tc_mma_bf16(t + 256 + tl * 128, dsc | ((base + AT_P + ks * 4096) >> 4), dsc | ((base + AT_V + ks * 4096) >> 4), id_o, ks);
        tc_commit(b_o);
        tph++;
      }
    }
    __syncthreads();
    uph++;
  }
  if (w == 0) tc_dealloc(t, 512);
}

// sum of a row's pages (records relative to their own maxima), the sink and the output gate
// thread = (head, 16 features): 16-byte loads of the records, a page loop per thread (not a head x page loop)
__global__ void __launch_bounds__(96) k_attn_fold(const PCtx* c, int i) {
  if (blockIdx.x >= c->h.T) return;       // fixed-grid launches: no such row in this pass
  __shared__ float fac[NQL][MAXPG], sm_l[NQL], sm_m[NQL];
  const int r = blockIdx.x, tid = threadIdx.x, head = tid >> 3, f0 = (tid & 7) * 16;
  const PLayer& L = c->L[i];
  const int* grp = TAB(o_grp);
  int lo = 0, hi = c->h.ngrp - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (grp[mid * GRPW] <= r) lo = mid; else hi = mid - 1;
  }
  const int* G = grp + lo * GRPW;
  const int gn = G[1];
  const int rr = r - G[0], npg = L.glob ? ((G[3] + gn - 1) >> 9) + 1 : G[5], ubase = G[L.glob ? 7 : 6];
  const long sstep = 2L * 128 * APW;      // a page: two units (KV heads) of 128 records
  const bf16* q = c->qkvo + (long)r * 3824;
  const float* rec = c->apart + ((long)(ubase + head / 6) * 128 + rr * 6 + head % 6) * APW;
  if ((tid & 7) == 0) {                 // one thread per head: its maximum over the pages, the pages' factors, the sum
    float m = -1.0e30f;
    for (int s = 0; s < npg; s++) m = fmaxf(m, rec[s * sstep]);
    float l = 0.f;
    for (int s = 0; s < npg; s++) { const float f = expf(rec[s * sstep] - m); fac[head][s] = f; l += rec[s * sstep + 1] * f; }
    sm_l[head] = l; sm_m[head] = m;
  }
  __syncthreads();
  float a[16];
#pragma unroll
  for (int e = 0; e < 16; e++) a[e] = 0.f;
  for (int s = 0; s < npg; s++) {
    const float f = fac[head][s];
    const float4* p4 = reinterpret_cast<const float4*>(rec + s * sstep + 4 + f0);
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const float4 v = p4[e];
      a[4 * e] += v.x * f; a[4 * e + 1] += v.y * f; a[4 * e + 2] += v.z * f; a[4 * e + 3] += v.w * f;
    }
  }
  const float l = sm_l[head], lse = sm_m[head] + logf(l);
  const float gs = sigm(lse - bf(q[QS + head]));
  float g[16], o[16];
  ld16(q + QG + head * HD + f0, g);
#pragma unroll
  for (int e = 0; e < 16; e++) {
    float x = a[e] / l;
    x = x * sigm(g[e]);
    o[e] = x * gs;
  }
  st16(c->attn_out + (long)r * (NQL * HD) + head * HD + f0, o);
}

// attention outputs of the four cards -> o_proj input, for this card's rows and the three before them (the short
// convolution over the o_proj output of a row reaches three rows back)
__global__ void __launch_bounds__(768) k_gather_o(const PCtx* c, int i, int cls) {
  // cls: -1 any pass; 0 / 1: this launch serves passes of up to 64 rows / of more (fixed-grid launches issue both)
  const int T = c->h.T;
  if (cls >= 0 && (T <= 64) != (cls == 0)) return;
  const int r = max(own_lo(T, c->card) - 3, 0) + blockIdx.x, hi = own_hi(T, c->card), tid = threadIdx.x;
  if (r >= hi && blockIdx.x > 0) return;
  xwait(c, XS_ATT(i));
  if (r >= hi) return;
  uint8_t* dst = reinterpret_cast<uint8_t*>(c->xo);
  for (int q = tid; q < 768; q += blockDim.x)
    *reinterpret_cast<uint4*>(dst + xta(r, q * 8, 6144 / 16)) = reinterpret_cast<const uint4*>(c->attn_out_p[q / 192] + (long)r * 1536)[q % 192];
}

// ------------------------------------------------------------------------------------------------ after attention
// 16 raw short-convolution inputs of row r at lag j (0 = the row itself): the pass's own rows, then the slot's state
__device__ __forceinline__ void raw16(const bf16* rows, long stride, int r, int j, int back, const bf16* hist, int ch, int hch, float* v) {
  ld16(j <= back ? rows + (long)(r - j) * stride + ch : hist + (long)(3 - (j - back)) * HS + hch, v);
}
// static short convolution of 16 channels from k0: taps [channel][4] (tap 3 = the current row), identity residual
__device__ __forceinline__ void sconv16(const bf16* rows, int r, int back, const bf16* hist, int hoff, const bf16* taps, int k0, float* y) {
  float x[4][RE], w[4][RE];
#pragma unroll
  for (int j = 0; j < 4; j++) raw16(rows, H, r, j, back, hist, k0, hoff + k0, x[j]);
#pragma unroll
  for (int q = 0; q < 4; q++) ld16(taps + (long)k0 * 4 + q * 16, w[q]);     // w[q][4*m + tap] = channel k0 + 4q + m
#pragma unroll
  for (int e = 0; e < RE; e++) {
    float acc = 0.f;
#pragma unroll
    for (int tap = 0; tap < 4; tap++) acc = acc + x[3 - tap][e] * w[e >> 2][(e & 3) * 4 + tap];
    y[e] = bf(tobf((acc + x[0][e]) * 0.4472135954999579f));
  }
}
__global__ void __launch_bounds__(RT) k_post_attn(const PCtx* c, int i) {
  const int r = own_lo(c->h.T, c->card) + blockIdx.x, k0 = threadIdx.x * RE;
  if (r >= own_hi(c->h.T, c->card)) return;       // this card's rows only
  __shared__ float red[9 * 20 + 20 + 10 + 10];
  const PLayer& L = c->L[i];
  const long tm = c->tmax;
  const int slot = TAB(o_slot)[r], back = TAB(o_back)[r];
  const bf16* hist = c->hist + ((long)slot * NL + i) * 3 * HS;
  bf16* as = c->attn_s + (long)r * H;
  {
    float y[RE];
    sconv16(c->attn_raw, r, back, hist, 512, L.attn_sc, k0, y);
    st16(as + k0, y);
  }
  const int nb = (i + 5) / 6;
  const bool boundary = (i % 6) == 0;
  const bf16* src[10];
  int n = 0;
  for (int s = 0; s < nb; s++) src[n++] = c->blocks + ((long)s * tm + r) * H;
  float u[1][RE];
  if (boundary) {
    src[n++] = c->blocks + ((long)nb * tm + r) * H;
    src[n++] = as;
    mix_row<1>(src, n, nullptr, L.mq, u, red);
  } else {
    src[n++] = c->blocks + ((long)nb * tm + r) * H;
    mix_row<1>(src, n, as, L.mq, u, red);
  }
  const bf16* w[1] = {L.mlp_norm};
  rms_rows<1>(u, w, c->eps, red);
  st16(c->mlp_in + (long)r * H + k0, u[0]);
  st16t(reinterpret_cast<uint8_t*>(c->mlp_inx), r, k0, H / 16, u[0]);
}

// ------------------------------------------------------------------------------------------------ quantizers
// E2M1 code (index 0..7 of 0, .5, 1, 1.5, 2, 3, 4, 6) of a magnitude, round to nearest, ties to the even code
__device__ __forceinline__ int e2m1_idx(float mag) {
  int q;
  if (mag <= 0.25f) q = 0; else if (mag < 0.75f) q = 1; else if (mag <= 1.25f) q = 2; else if (mag < 1.75f) q = 3;
  else if (mag <= 2.5f) q = 4; else if (mag < 3.5f) q = 5; else if (mag <= 5.0f) q = 6; else q = 7;
  return q;
}
__device__ __forceinline__ float e2m1_val(int q) { return q < 4 ? 0.5f * q : (q == 4 ? 2.0f : (q == 5 ? 3.0f : (q == 6 ? 4.0f : 6.0f))); }
// NVFP4 ("four over six"): this thread's block of 16 (x), the row's amax -> 8 bytes of codes (low nibble = even
// element) and the block's E4M3 scale byte
__device__ __forceinline__ void quant_nvfp4_blk(const float* x, float amax, uint2& codes, uint8_t& sbyte) {
  const float FMAX = 3.4028234663852886e38f;
  const float encoder = fminf(1536.0f / amax, FMAX), decoder = 1.0f / encoder, outer = amax * (1.0f / 1536.0f);
  float bm = 0.f;
#pragma unroll
  for (int e = 0; e < 16; e++) bm = fmaxf(bm, fabsf(x[e]));
  float sc[2], iv[2], er[2]; uint8_t sb[2];
#pragma unroll
  for (int cand = 0; cand < 2; cand++) {
    const float inv_max = cand == 0 ? (1.0f / 6.0f) : 0.25f;
    const __nv_fp8_e4m3 s8 = __nv_fp8_e4m3(fminf(bm * (encoder * inv_max), 448.0f));
    sb[cand] = s8.__x;
    sc[cand] = float(s8);
    iv[cand] = fminf(1.0f / (sc[cand] * decoder), FMAX);
    float err = 0.f;
#pragma unroll
    for (int e = 0; e < 16; e++) {
      const float val = x[e] * iv[cand];
      const float q = (val != val) ? 0.f : e2m1_val(e2m1_idx(fabsf(val)));
      err += fabsf(((signbit(val) ? -q : q) * sc[cand]) * outer - x[e]);
    }
    er[cand] = err;
  }
  const int pick = (er[1] < er[0]) ? 1 : 0;
  const float inverse = iv[pick];
  uint32_t wlo = 0, whi = 0;
#pragma unroll
  for (int e = 0; e < 16; e++) {
    const float val = x[e] * inverse;
    const uint32_t nib = (val != val) ? 0u : ((uint32_t)e2m1_idx(fabsf(val)) | (signbit(val) ? 8u : 0u));
    if (e < 8) wlo |= nib << (4 * e); else whi |= nib << (4 * (e - 8));
  }
  codes = make_uint2(wlo, whi);
  sbyte = sb[pick];
}
// MXFP8: this thread's block of 32 -> 32 E4M3 codes and the block's UE8M0 exponent
__device__ __forceinline__ void quant_mxfp8_blk(const float* x, uint4& lo, uint4& hi, uint8_t& sbyte) {
  float m = 0.f;
#pragma unroll
  for (int e = 0; e < 32; e++) m = fmaxf(m, fabsf(x[e]));
  const float raw = m * (1.0f / 448.0f);
  const unsigned bits = __float_as_uint(raw);
  int ex = (int)(((bits + 0x007FFFFFu) & 0x7F800000u) >> 23);
  if (raw <= 5.877471754111438e-39f) ex = 0;
  if (ex > 254) ex = 254;
  const float inverse = (ex == 0) ? 1.7014118346046923e38f : 1.0f / ue8m0((unsigned)ex);
  uint32_t wd[8];
#pragma unroll
  for (int q = 0; q < 8; q++) {
    uint32_t w0 = 0;
#pragma unroll
    for (int b = 0; b < 4; b++) w0 |= (uint32_t)__nv_fp8_e4m3(fminf(fmaxf(x[4 * q + b] * inverse, -448.f), 448.f)).__x << (8 * b);
    wd[q] = w0;
  }
  lo = make_uint4(wd[0], wd[1], wd[2], wd[3]); hi = make_uint4(wd[4], wd[5], wd[6], wd[7]);
  sbyte = (uint8_t)ex;
}

// ------------------------------------------------------------------------------------------------ routing + input quantization
__global__ void __launch_bounds__(384) k_route(const PCtx* c, int i) {
  const int r = own_lo(c->h.T, c->card) + blockIdx.x, tid = threadIdx.x;
  if (r >= own_hi(c->h.T, c->card)) return;       // this card's rows only
  __shared__ float cs[TOPK], red[12];
  __shared__ int flag[NE];
  __shared__ unsigned long long wmax[12];
  const PLayer& L = c->L[i];
  const float logit = c->rlog[(long)r * NE + tid];
  const float sg = sigm(logit);
  // key: the selection score in an order-preserving code, then the lower expert index wins a tie
  unsigned bits = __float_as_uint(sg + L.rbias[tid]);
  bits = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  unsigned long long key = ((unsigned long long)bits << 32) | (unsigned)(NE - tid);
  bool chosen = false;
  for (int round = 0; round < TOPK; round++) {
    unsigned long long m = key;
#pragma unroll
    for (int off = 16; off; off >>= 1) { const unsigned long long o = __shfl_xor_sync(0xffffffff, m, off); m = o > m ? o : m; }
    if ((tid & 31) == 0) wmax[tid >> 5] = m;
    __syncthreads();
    m = wmax[0];
#pragma unroll
    for (int k = 1; k < 12; k++) m = wmax[k] > m ? wmax[k] : m;
    __syncthreads();
    if (key == m) { chosen = true; key = 0ull; }
  }
  flag[tid] = chosen;
  __syncthreads();
  if (flag[tid]) {
    int slot = 0;
    for (int e = 0; e < tid; e++) slot += flag[e];
    c->ids[r * TOPK + slot] = tid;
    reinterpret_cast<int*>(c->xs8 + (long)r * XS8W + 288)[slot] = tid;
    cs[slot] = sg;
    atomicAdd(&c->cnt[i * NE + tid], 1);
  }
  __syncthreads();
  if (tid < TOPK) {
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; k++) sum += cs[k];
    const float wv = cs[tid] / sum * 8.0f;
    c->wts[r * TOPK + tid] = wv;
    reinterpret_cast<float*>(c->xs8 + (long)r * XS8W + 320)[tid] = wv;
  }
  // quantization of the MoE input (codes in natural order; k_packx lays them out per item)
  const bf16* x = c->mlp_in + (long)r * H;
  if (!L.mx) {
    float v[16], m = 0.f;
    if (tid < 288) {
#pragma unroll
      for (int e = 0; e < 16; e++) { v[e] = bf(x[tid * 16 + e]); m = fmaxf(m, fabsf(v[e])); }
    }
    const float amax = block_max1(m, red);
    if (tid < 288) {
      uint2 cd; uint8_t sb;
      quant_nvfp4_blk(v, amax, cd, sb);
      *reinterpret_cast<uint2*>(c->xq + (long)r * H + tid * 8) = cd;
      c->xs8[(long)r * XS8W + tid] = sb;
    }
    if (tid == 0) { c->xouter[r] = amax * (1.0f / 1536.0f); *reinterpret_cast<float*>(c->xs8 + (long)r * XS8W + 352) = amax * (1.0f / 1536.0f); }
  } else if (tid < 144) {
    float v[32];
#pragma unroll
    for (int e = 0; e < 32; e++) v[e] = bf(x[tid * 32 + e]);
    uint4 lo, hi; uint8_t sb;
    quant_mxfp8_blk(v, lo, hi, sb);
    *reinterpret_cast<uint4*>(c->xq + (long)r * H + tid * 32) = lo;
    *reinterpret_cast<uint4*>(c->xq + (long)r * H + tid * 32 + 16) = hi;
    c->xs8[(long)r * XS8W + tid] = sb;
  }
}

// pair lists per card (expert-major, rows ascending) and the GEMM items; block = expert
__global__ void __launch_bounds__(NTB) k_assign(const PCtx* c, int i) {
  if (!c->h.T) return;
  __shared__ int cj[NTB];
  __shared__ int sm_poff, sm_ioff;
  const int e = blockIdx.x, tid = threadIdx.x;
  const int* em = c->emap + i * NE;
  const int cc = em[e] & 255;           // the card that holds this expert
  const int T = c->h.T;
  const int* cnt = c->cntg + i * NE;
  {   // a card's pairs and items: its experts in ascending order; pairs, items and "a later expert exists" before e
    int v[3] = {0, 0, 0};
    for (int o = tid; o < NE; o += NTB)
      if ((em[o] & 255) == cc) {
        if (o < e) { v[0] += cnt[o]; v[1] += (cnt[o] + COLT - 1) / COLT; }
        else if (o > e) v[2] = 1;
      }
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int off = 16; off; off >>= 1) v[k] += __shfl_xor_sync(0xffffffff, v[k], off);
      if ((tid & 31) == 0) cj[(tid >> 5) * 3 + k] = v[k];
    }
  }
  __syncthreads();
  if (tid == 0) {
    int poff = 0, ioff = 0, later = 0;
#pragma unroll
    for (int w8 = 0; w8 < NTB / 32; w8++) { poff += cj[w8 * 3]; ioff += cj[w8 * 3 + 1]; later += cj[w8 * 3 + 2]; }
    sm_poff = poff; sm_ioff = ioff;
    const int n = cnt[e];
    for (int ti = 0; ti * COLT < n; ti++) {
      int* it = c->items + ((long)cc * c->pmax + ioff + ti) * 4;
      it[0] = em[e] >> 8; it[1] = poff + ti * COLT; it[2] = min(COLT, n - ti * COLT); it[3] = e;
    }
    if (!later) { c->ginfo[cc * 4] = ioff + (n + COLT - 1) / COLT; c->ginfo[cc * 4 + 1] = poff + n; }
  }
  __syncthreads();
  if (cnt[e] == 0) return;              // no row chose this expert (two thirds of them, with the trace's prompts)
  // A block's time is the latency of one thread's loads: a thread's rows are read four at a time (16-byte loads of
  // the eight ids of a row, all issued before the first is compared) instead of id by id.
  const int per = (T + NTB - 1) / NTB, r0 = tid * per, r1 = min(T, r0 + per);
  const uint4* idv = reinterpret_cast<const uint4*>(c->ids);
  int mine = 0;
  for (int rb = r0; rb < r1; rb += 4) {
    uint4 v[8];
#pragma unroll
    for (int j = 0; j < 4; j++) { const int r = min(rb + j, r1 - 1); v[2 * j] = idv[r * 2]; v[2 * j + 1] = idv[r * 2 + 1]; }
#pragma unroll
    for (int j = 0; j < 4; j++)
      if (rb + j < r1) {
        const uint4 a = v[2 * j], b = v[2 * j + 1];
        mine += (a.x == e) + (a.y == e) + (a.z == e) + (a.w == e) + (b.x == e) + (b.y == e) + (b.z == e) + (b.w == e);
      }
  }
  cj[tid] = mine;
  __syncthreads();
  if (!mine) return;
  int pos = sm_poff;
  for (int j = 0; j < tid; j++) pos += cj[j];
  for (int r = r0; r < r1; r++) {
    const uint4 a = idv[r * 2], b = idv[r * 2 + 1];
    const unsigned w8[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int k = 0; k < TOPK; k++)
      if ((int)w8[k] == e) {
        c->ptok[(long)cc * c->pmax + pos] = r; c->pslot[(long)cc * c->pmax + pos] = k; c->pidx[r * TOPK + k] = pos;
        if (cc == c->card) c->pitem[pos] = sm_ioff + (pos - sm_poff) / COLT;
        pos++;
      }
  }
}

// ------------------------------------------------------------------------------------------------ expert inputs
// the quantized MoE input of a pair's row, laid out as column (pair - first pair of its item) of the item's B units:
// per K step two 16-byte pieces at (n/8)*256 + h*128 + (n%8)*16 and the scales at 4096 + (n%32)*16 + (n/32)*4
template <bool MX>
__global__ void __launch_bounds__(160) k_packx(const PCtx* c, int i) {
  const int tid = threadIdx.x, me = c->card;
  const int np = c->ginfo[me * 4 + 1];
  const long pm = c->pmax;
  const int KS = MX ? 144 : 72;
  for (int pair = blockIdx.x; pair < np; pair += gridDim.x) {
    const int row = c->ptok[me * pm + pair], item = c->pitem[pair];
    const int n = pair - c->items[((long)me * pm + item) * 4 + 1];
    const uint8_t* src = c->xq + (long)row * H;
    const uint8_t* ssrc = c->xs8 + (long)row * XS8W;
    uint8_t* dst = c->xt13 + (long)item * KS * TC_BUB + (n >> 3) * 256 + (n & 7) * 16;
    uint8_t* sdst = c->xt13 + (long)item * KS * TC_BUB + 4096 + (n & 31) * 16 + (n >> 5) * 4;
    for (int ks = tid; ks < KS; ks += blockDim.x) {
      *reinterpret_cast<uint4*>(dst + (long)ks * TC_BUB) = *reinterpret_cast<const uint4*>(src + ks * 32);
      *reinterpret_cast<uint4*>(dst + (long)ks * TC_BUB + 128) = *reinterpret_cast<const uint4*>(src + ks * 32 + 16);
      *reinterpret_cast<uint32_t*>(sdst + (long)ks * TC_BUB) = MX ? (uint32_t)ssrc[ks] : *reinterpret_cast<const uint32_t*>(ssrc + ks * 4);
    }
  }
}

// activation of a pair, times its routing weight, rounded to bf16, quantized for w2 into the item's B units
template <bool MX>
__global__ void __launch_bounds__(160) k_act(const PCtx* c, int i) {
  __shared__ float red[8];
  const int tid = threadIdx.x, me = c->card;
  const int np = c->ginfo[me * 4 + 1];
  const long pm = c->pmax;
  const int KS = MX ? 72 : 36;
  for (int pair = blockIdx.x; pair < np; pair += gridDim.x) {
    const int row = c->ptok[me * pm + pair], k = c->pslot[me * pm + pair], item = c->pitem[pair];
    const int n = pair - c->items[((long)me * pm + item) * 4 + 1];
    const float wt = c->wts[row * TOPK + k];
    const bf16* h = c->h13 + (long)pair * H;
    uint8_t* dst = c->xt2 + (long)item * KS * TC_BUB + (n >> 3) * 256 + (n & 7) * 16;
    uint8_t* sdst = c->xt2 + (long)item * KS * TC_BUB + 4096 + (n & 31) * 16 + (n >> 5) * 4;
    if (!MX) {
      float v[16], m = 0.f;
      if (tid < 144) {
#pragma unroll
        for (int e = 0; e < 16; e++) {
          const uint32_t gu = *reinterpret_cast<const uint32_t*>(h + (tid * 16 + e) * 2);
          const float gg = bf(*reinterpret_cast<const bf16*>(&gu)), uu = bf(reinterpret_cast<const bf16*>(&gu)[1]);
          const float a = (4.0f * tanhf(gg / 4.0f)) * sigm(gg);
          v[e] = bf(tobf((a * (25.0f * tanhf(uu / 25.0f))) * wt));
          m = fmaxf(m, fabsf(v[e]));
        }
      }
      const float amax = block_max1(m, red);
      if (tid < 144) {       // block tid = k16 block (tid & 3) of K step (tid >> 2)
        uint2 cd; uint8_t sb;
        quant_nvfp4_blk(v, amax, cd, sb);
        const int ks = tid >> 2, j = tid & 3;
        *reinterpret_cast<uint2*>(dst + (long)ks * TC_BUB + (j >> 1) * 128 + (j & 1) * 8) = cd;
        sdst[(long)ks * TC_BUB + j] = sb;
      }
      if (tid == 0) c->aouter[pair] = amax * (1.0f / 1536.0f);
    } else if (tid < 72) {
      float v[32];
#pragma unroll
      for (int e = 0; e < 32; e++) {
        const uint32_t gu = *reinterpret_cast<const uint32_t*>(h + (tid * 32 + e) * 2);
        const float gg = bf(*reinterpret_cast<const bf16*>(&gu)), uu = bf(reinterpret_cast<const bf16*>(&gu)[1]);
        const float a = (4.0f * tanhf(gg / 4.0f)) * sigm(gg);
        v[e] = bf(tobf((a * (25.0f * tanhf(uu / 25.0f))) * wt));
      }
      uint4 lo, hi; uint8_t sb;
      quant_mxfp8_blk(v, lo, hi, sb);
      *reinterpret_cast<uint4*>(dst + (long)tid * TC_BUB) = lo;
      *reinterpret_cast<uint4*>(dst + (long)tid * TC_BUB + 128) = hi;
      *reinterpret_cast<uint32_t*>(sdst + (long)tid * TC_BUB) = (uint32_t)sb;
    }
    __syncthreads();
  }
}

// ------------------------------------------------------------------------------------------------ after the experts
// sum of a row's eight expert outputs in expert order, on the row's home card (pair outputs live on the experts' cards),
// for this card's rows and the three before them (the short convolution over the MoE output reaches three rows back)
__global__ void __launch_bounds__(576) k_combine(const PCtx* c, int i) {
  const int T = c->h.T;
  const int r = max(own_lo(T, c->card) - 3, 0) + blockIdx.x, hi = own_hi(T, c->card), tid = threadIdx.x;
  if (r >= hi && blockIdx.x > 0) return;
  xwait(c, XS_EXP(i));
  if (r >= hi) return;
  const uint4* po[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; k++) po[k] = reinterpret_cast<const uint4*>(c->pairout_p[c->emap[i * NE + c->ids[r * TOPK + k]] & 255] + (long)c->pidx[r * TOPK + k] * H);
  uint4* mr = reinterpret_cast<uint4*>(c->mlp_raw + (long)r * H);
  for (int q = tid; q < H / 8; q += blockDim.x) {
    float s[8];
#pragma unroll
    for (int e = 0; e < 8; e++) s[e] = 0.f;
#pragma unroll
    for (int j = 0; j < TOPK; j++) {
      const uint4 v = po[j][q];
      const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
      for (int e = 0; e < 8; e++) s[e] += bf(b[e]);
    }
    mr[q] = pk8(s);
  }
}
__device__ __forceinline__ void post_mlp_row(const PCtx* c, int i, int r) {
  const int k0 = threadIdx.x * RE;
  const PLayer& L = c->L[i];
  const long tm = c->tmax;
  const int slot = TAB(o_slot)[r], back = TAB(o_back)[r];
  const bf16* hist = c->hist + ((long)slot * NL + i) * 3 * HS;
  const int nb = (i + 5) / 6;
  const bool boundary = (i % 6) == 0;
  float mlp[RE], a[RE], pt[RE], o[RE];
  sconv16(c->mlp_raw, r, back, hist, 5120, L.mlp_sc, k0, mlp);
  ld16(c->attn_s + (long)r * H + k0, a);
  if (!boundary) ld16(c->blocks + ((long)nb * tm + r) * H + k0, pt);
#pragma unroll
  for (int e = 0; e < RE; e++) o[e] = boundary ? a[e] + mlp[e] : pt[e] + a[e] + mlp[e];
  st16(c->blocks + ((long)(boundary ? nb + 1 : nb) * tm + r) * H + k0, o);
}
__global__ void __launch_bounds__(RT) k_post_mlp(const PCtx* c, int i) {
  const int r = own_lo(c->h.T, c->card) + blockIdx.x;
  if (r >= own_hi(c->h.T, c->card)) return;       // this card's rows only
  post_mlp_row(c, i, r);
}
// The end of layer i and the attention input of layer i + 1 in one kernel: both are work of a row's own block (a
// thread reads back only the 16 elements it has just written), and a kernel boundary costs a small pass ~1.4 us.
template <int NQ>
__global__ void __launch_bounds__(RT) k_post_mix(const PCtx* c, int i) {
  const int r = own_lo(c->h.T, c->card) + blockIdx.x;
  if (r >= own_hi(c->h.T, c->card)) return;       // this card's rows only
  __shared__ float red[9 * 40 + 40 + 10 + 30];
  post_mlp_row(c, i, r);
  mix_attn_row<NQ>(c, i + 1, r, red);
}


// short-convolution state: a job = (dst kind: 0 slot state, 1 page snapshot; dst index; slot; row0; n)
__global__ void __launch_bounds__(NTB) k_hist(const PCtx* c, int i, int snaps) {
  if (blockIdx.x >= (snaps ? c->h.nsnap : c->h.nseq)) return;
  const int tid = threadIdx.x;
  const PLayer& L = c->L[i];
  const int* job = (snaps ? TAB(o_snap) : TAB(o_seq)) + blockIdx.x * 4;
  const int dsti = job[0], slot = job[1], row0 = job[2], n = job[3];
  const int V0 = QK + (L.glob ? 272 : 256);
  bf16* dst = (snaps ? c->snap + ((long)dsti * NL + i) * 3 * HS : c->hist + ((long)dsti * NL + i) * 3 * HS);
  const bf16* old = c->hist + ((long)slot * NL + i) * 3 * HS;
  for (int q = tid; q < HS / 8; q += NTB) {
    const int ch = q * 8;
    const bf16* rows; long stride; int off;
    if (ch < 256) { rows = c->qkvo; stride = 3824; off = QK + ch; }
    else if (ch < 512) { rows = c->qkvo; stride = 3824; off = V0 + ch - 256; }
    else if (ch < 5120) { rows = c->attn_raw; stride = H; off = ch - 512; }
    else { rows = c->mlp_raw; stride = H; off = ch - 5120; }
#pragma unroll
    for (int idx = 0; idx < 3; idx++) {
      const int o = n - 3 + idx;
      const uint4 v = *reinterpret_cast<const uint4*>(o >= 0 ? rows + (long)(row0 + o) * stride + off : old + (long)(idx + n) * HS + ch);
      *reinterpret_cast<uint4*>(dst + (long)idx * HS + ch) = v;
    }
  }
}

// state of sequences that start in this pass: zero, or the snapshot of the last shared page. job = (slot, page or -1)
__global__ void __launch_bounds__(NTB) k_prologue(const PCtx* c) {
  if (blockIdx.x >= c->h.nprol * NL) return;
  const int* job = TAB(o_prol) + (blockIdx.x / NL) * 2;
  const int layer = blockIdx.x % NL, slot = job[0], page = job[1];
  bf16* dst = c->hist + ((long)slot * NL + layer) * 3 * HS;
  const bf16* src = c->snap + ((long)max(page, 0) * NL + layer) * 3 * HS;
  for (int k = threadIdx.x; k < 3 * HS; k += NTB) dst[k] = page >= 0 ? src[k] : tobf(0.f);
}

// final mix + norm of the rows that need logits -> hfin (B-fragment order)
__global__ void __launch_bounds__(RT) k_final(const PCtx* c) {
  if (blockIdx.x >= c->h.nlog || !c->h.mode) return;
  __shared__ float red[9 * 20 + 20 + 10 + 10];
  const int li = blockIdx.x, r = TAB(o_log)[li], k0 = threadIdx.x * RE;
  const long tm = c->tmax;
  if (c->h.mode == 2 && threadIdx.x == 0) c->cand[li] = 0ull;
  if (r < own_lo(c->h.T, c->card) || r >= own_hi(c->h.T, c->card)) return;       // the others fetch the row (k_xfin)
  const bf16* src[10];
  for (int s = 0; s < 9; s++) src[s] = c->blocks + ((long)s * tm + r) * H;
  float u[1][RE];
  mix_row<1>(src, 9, nullptr, c->oq, u, red);
  const bf16* w[1] = {c->fnorm};
  rms_rows<1>(u, w, c->eps, red);
  st16t(reinterpret_cast<uint8_t*>(c->hfin), li, k0, H / 16, u[0]);
}
__global__ void k_join(const PCtx* c) { if (!c->h.nlog || !c->h.mode) return; xwait(c, XS_JOIN); }

// ------------------------------------------------------------------------------------------------ serving loop on the device
// k_sched (card 0) is the scheduler: it reads the tokens of the pass that just ran, takes arrivals from the request
// ring, keeps the slot table, the page free list and the prefix cache, and writes the next pass (header and tables).
// k_wait (cards 1-3) takes that pass over. The host only enqueues the fixed pass program again and again, copies
// prompts into the arena and reads the token ring: no decision, and nothing here waits for it.
__device__ __forceinline__ double dev_now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return (double)t * 1e-9;
}
__global__ void k_sched_init(PCtx* c, int npages, int tmax, int umax, int ncache, int* freep, int* page_ref, SNode* cache, const double* cfg, float* eload, int* emaster) {
  if (threadIdx.x || blockIdx.x) return;
  s_init(c->sched, npages, tmax, umax, ncache, freep, page_ref, cache);
  Sched* S = c->sched;
  if (cfg[10] > 0) {       // expert placement follows the traffic
    S->eload = eload; S->emap = emaster; S->reb_every = (int)cfg[10];
    for (int i = 0; i < NL * NE; i++) { eload[i] = 0.f; emaster[i] = c->emap[i]; }
  }
  S->base = cfg[0]; S->dec = cfg[1]; S->row = cfg[2]; S->ctx = cfg[3]; S->target_decode_ms = cfg[4]; S->target_idle_ms = cfg[5];
  S->ttft_slo = cfg[6]; S->gap_limit = cfg[7]; S->min_pass_ms = cfg[8]; S->host_ms = cfg[9];
  if (cfg[11] > 0 && cfg[11] < tmax) S->soft_rows = (int)cfg[11];
  S->lazy_rows = 1;        // k_sched writes the rows' tables with all its threads
  for (int i = 0; i < S_NSLOT * S_MAXPG; i++) c->tab[(4 + S_GRPW) * tmax + 4 * S_MAXE + 4 * (S_MAXE + 64) + S_MAXE + 2 * S_MAXE + i] = 0;
}
#define S_CPMAX 64       // prompts copied in per pass boundary
__global__ void __launch_bounds__(1024) k_sched(PCtx* c) {
  __shared__ unsigned long long best[S_MAXE];
  __shared__ int cp_off[S_CPMAX], cp_len[S_CPMAX], cp_n;
  __shared__ unsigned cp_head;
  __shared__ int prop[S_NL * S_PROPW], obs, reb, ne_s;
  const int tid = threadIdx.x;
  Sched* S = c->sched;
  SShared* sh = c->shr;
  // the other cards have taken the pass composed last (so its tables may be overwritten)
  if (tid == 0) {
    const unsigned tag = (unsigned)S->pass_no;
    for (int p = 1; p < NCARD; p++) {
      unsigned long long spins = 0;
      while (ld_acquire_sys(&c->sig_p[p][SIG_TAKEN]) != tag) { if (++spins > (1ull << 31)) { c->err[1] = 100 + p; break; } }
    }
  }
  // candidates of the pass that just ran: the best of the four cards per row that wanted a token
  const int nlog = S->ne ? c->h.nlog : 0;
  for (int li = tid; li < nlog; li += blockDim.x) {
    unsigned long long b = 0;
#pragma unroll
    for (int d = 0; d < NCARD; d++) { const unsigned long long v = c->cand_p[d][li]; b = v > b ? v : b; }
    best[li] = b;
  }
  __syncthreads();
  if (tid == 0) {
    // an idle engine: wait a little for a request before composing an empty pass
    bool work = false;
    for (int i = 0; i < S_NSLOT; i++) work |= S->slot[i].state != 0;
    if (!work && S->ne == 0) {
      for (int spin = 0; spin < 200000; spin++)
        if (sh->ring_head != sh->ring_tail || sh->cmd_drop != sh->ack_drop || sh->stop) break;
    }
    // the requests written since the last pass: their prompts go from the host's page-locked memory into the arena
    // here (all threads of this block), so that submit() makes no call into the device at all
    unsigned hd = sh->ring_head;
    if (hd - S->copied > S_CPMAX) hd = S->copied + S_CPMAX;
    cp_n = (int)(hd - S->copied); cp_head = hd;
    for (int k = 0; k < cp_n; k++) {
      const SReq& e = sh->ring[(S->copied + k) & (S_RING - 1)];
      cp_off[k] = e.off; cp_len[k] = e.plen;
    }
    obs = s_obs_due(S, c->cntg) ? 1 : 0; reb = (obs && s_reb_due(S)) ? 1 : 0;
  }
  __syncthreads();
  for (int k = 0; k < cp_n; k++) {
    const uint4* src = reinterpret_cast<const uint4*>(c->harena + cp_off[k]);
    uint4* dst = reinterpret_cast<uint4*>(const_cast<int*>(c->arena) + cp_off[k]);
    const int n16 = (cp_len[k] + 3) >> 2;
    for (int q = tid; q < n16; q += blockDim.x) dst[q] = src[q];
  }
  // What one thread took 1-3 ms for on every prompt pass is done by the block: the running average of the pairs per
  // expert and layer (18,432 values), each layer's swaps when a rebalance is due (48 threads), and below the per-row
  // tables of the pass. The decisions stay with thread 0 (s_step).
  if (obs) for (int i = tid; i < S_NL * S_NE; i += blockDim.x) S->eload[i] = 0.75f * S->eload[i] + 0.25f * (float)c->cntg[i];
  __syncthreads();
  if (reb && tid < S_NL) prop[tid * S_PROPW] = s_reb_layer(S->eload + tid * S_NE, S->emap + tid * S_NE, prop + tid * S_PROPW + 1);
  __syncthreads();
  if (tid == 0) {
    S->copied = cp_head; S->head_limit = cp_head; S->has_limit = 1;
    const unsigned long long* cp[4] = {best, best, best, best};
    s_step(S, sh, c->arena, cp, c->cntg, c->tab, &c->h, dev_now(), obs ? prop : 0);
    ne_s = S->ne;
  }
  __syncthreads();
  {   // rows: slot, position, input token, rows back (thread = rows tid, tid + 1024, ... of the pass)
    int* t_slot = c->tab; int* t_pos = t_slot + S->tmax; int* t_tok = t_pos + S->tmax; int* t_back = t_tok + S->tmax;
    for (int e = 0; e < ne_s; e++) {
      const SEntry en = S->ent[e];
      const SSlot& q = S->slot[en.slot];
      for (int x = (((tid - en.r0) % (int)blockDim.x) + (int)blockDim.x) % (int)blockDim.x; x < en.n; x += blockDim.x) {
        const int r = en.r0 + x;
        t_slot[r] = en.slot; t_pos[r] = en.p0 + x; t_back[r] = x < 3 ? x : 3;
        t_tok[r] = en.kind ? c->arena[q.off + en.p0 + x] : q.last_tok;
      }
    }
  }
  __syncthreads();
  if (tid) return;
  st_release_sys(&c->sig_p[0][SIG_SCHED], (unsigned)c->h.tag);
  st_release_sys(&c->sig_p[0][SIG_TAKEN], (unsigned)c->h.tag);
}
__global__ void __launch_bounds__(1024) k_wait(PCtx* c) {
  __shared__ PHdr hs;
  const int tid = threadIdx.x, me = c->card;
  if (tid == 0) {
    const unsigned last = ld_acquire_sys(&c->sig_p[me][SIG_TAKEN]);
    unsigned long long spins = 0;
    while (ld_acquire_sys(&c->sig_p[0][SIG_SCHED]) == last) { if (++spins > (1ull << 33)) { c->err[1] = 200 + me; break; } }
    hs = c->ctx_p[0]->h;
  }
  __syncthreads();
  // the tables of the pass: rows, groups, jobs, page table (16 bytes per load)
  const int* src = c->tab_p[0];
  const int tmax = hs.pad2;
  const int secs[11][2] = {{hs.o_slot, hs.T}, {hs.o_pos, hs.T}, {hs.o_tok, hs.T}, {hs.o_back, hs.T}, {hs.o_grp, hs.ngrp * S_GRPW}, {hs.o_seq, hs.nseq * 4},
                           {hs.o_snap, hs.nsnap * 4}, {hs.o_log, hs.nlog}, {hs.o_prol, hs.nprol * 2}, {hs.o_pt, S_NSLOT * S_MAXPG}, {hs.o_swap, hs.nswap * 7}};
  (void)tmax;
#pragma unroll 1
  for (int k = 0; k < 11; k++) {
    const int o = secs[k][0] & ~3, n4 = (secs[k][1] + (secs[k][0] & 3) + 3) >> 2;
    for (int q = tid; q < n4; q += blockDim.x)
      *reinterpret_cast<uint4*>(c->tab + o + q * 4) = *reinterpret_cast<const uint4*>(src + o + q * 4);
  }
  __syncthreads();
  if (tid == 0) {
    c->h = hs;
    st_release_sys(&c->sig_p[me][SIG_TAKEN], (unsigned)hs.tag);
  }
}

// Expert swaps decided by the scheduler: each of the two cards of a swap copies the other's expert into its temp
// (k_swap1), and when every card has its copies, puts it in the place of the expert it gives away (k_swap2); all
// cards update the placement table.
__global__ void __launch_bounds__(256) k_swap1(const PCtx* c) {
  const int sidx = blockIdx.x / SWB, part = blockIdx.x % SWB;
  if (sidx >= c->h.nswap) return;
  const int* sw = TAB(o_swap) + 7 * sidx;
  const int i = sw[0], ca = sw[3], ia = sw[4], cb = sw[5], ib = sw[6], me = c->card;
  if (me != ca && me != cb) return;
  const int pc = me == ca ? cb : ca, pi = me == ca ? ib : ia;       // the other card and the expert's index there
  const long e13 = 36L * (i >= 40 ? 144 : 72) * 4608, e2 = e13 / 2, n13 = e13 / 16, n16 = (e13 + e2) / 16;
  const uint4* s13 = reinterpret_cast<const uint4*>(c->w13_p[i][pc] + pi * e13);
  const uint4* s2 = reinterpret_cast<const uint4*>(c->w2_p[i][pc] + pi * e2);
  uint4* dst = reinterpret_cast<uint4*>(c->swtmp + sidx * SWTMP);
  for (long q = n16 * part / SWB + threadIdx.x; q < n16 * (part + 1) / SWB; q += blockDim.x) dst[q] = q < n13 ? s13[q] : s2[q - n13];
}
__global__ void __launch_bounds__(256) k_swap2(const PCtx* c) {
  const int sidx = blockIdx.x / SWB, part = blockIdx.x % SWB;
  if (sidx >= c->h.nswap) return;
  xwait(c, SIG_SWAP);
  const int* sw = TAB(o_swap) + 7 * sidx;
  const int i = sw[0], a = sw[1], b = sw[2], ca = sw[3], ia = sw[4], cb = sw[5], ib = sw[6], me = c->card;
  if (part == 0 && threadIdx.x == 0) { int* em = c->emap + i * NE; em[a] = cb | (ib << 8); em[b] = ca | (ia << 8); }
  if (me != ca && me != cb) return;
  const int mi = me == ca ? ia : ib;                                 // the place on this card
  const long e13 = 36L * (i >= 40 ? 144 : 72) * 4608, e2 = e13 / 2, n13 = e13 / 16, n16 = (e13 + e2) / 16;
  uint4* d13 = reinterpret_cast<uint4*>(const_cast<uint8_t*>(c->L[i].w13) + mi * e13);
  uint4* d2 = reinterpret_cast<uint4*>(const_cast<uint8_t*>(c->L[i].w2) + mi * e2);
  const uint4* src = reinterpret_cast<const uint4*>(c->swtmp + sidx * SWTMP);
  for (long q = n16 * part / SWB + threadIdx.x; q < n16 * (part + 1) / SWB; q += blockDim.x) { if (q < n13) d13[q] = src[q]; else d2[q - n13] = src[q]; }
}

// ------------------------------------------------------------------------------------------------ host API
extern "C" int pe_ctx_bytes() { return (int)sizeof(PCtx); }
extern "C" int pe_layer_bytes() { return (int)sizeof(PLayer); }
extern "C" int pe_hdr_bytes() { return (int)sizeof(PHdr); }

extern "C" int pe_enable_peer(int n) {
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
extern "C" int pe_setup(int dev) {
  cudaSetDevice(dev);
  int e = 0;
#define SA(K) e |= (int)cudaFuncSetAttribute(K, cudaFuncAttributeMaxDynamicSharedMemorySize, TC_SMEM)
  SA((k_tc<TK_BF16, G_QKV>)); SA((k_tc<TK_BF16, G_OPROJ>)); SA((k_tc<TK_BF16, G_ROUTER>)); SA((k_tc<TK_BF16, G_UNEMBED>)); SA((k_tc<TK_BF16, G_RELB>));
  SA((k_tc<TK_FP4, G_W13>)); SA((k_tc<TK_FP4, G_W2>)); SA((k_tc<TK_FP8, G_W13>)); SA((k_tc<TK_FP8, G_W2>));
#undef SA
  e |= (int)cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM);
  return e;
}
extern "C" void* pe_host_alloc(size_t n) {
  void* p = nullptr;
  if (cudaHostAlloc(&p, n, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) return nullptr;
  memset(p, 0, n);
  return p;
}

struct PassArgs { int dev; void* ctx; const void* hdr; const void* tab; long tab_bytes; void* tab_dev; void* stream; int nsm; int flags; };   // flags: 1 fixed grids, 2 tables are written on the device

static int launch_pass(const PassArgs& a) {
  cudaSetDevice(a.dev);
  cudaStream_t s = (cudaStream_t)a.stream;
  const PCtx* c = (const PCtx*)a.ctx;
  // The pass as a fixed program (flags & 1): grids for the largest pass, every kernel leaves when the pass has no
  // work for a block. Then nothing here depends on what the pass contains: it can be enqueued before the device
  // has decided it.
  const bool fixed = a.flags & 1;
  PHdr h = *(const PHdr*)a.hdr;
  if (!(a.flags & 2)) {
    cudaMemcpyAsync(a.tab_dev, a.tab, a.tab_bytes, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(a.ctx, a.hdr, sizeof(PHdr), cudaMemcpyHostToDevice, s);
  }
  const int nsm = a.nsm;
  if (fixed) { h.T = h.pad2; h.nprol = S_NSLOT; h.nseq = S_NSLOT; h.nsnap = S_NSLOT + 64; h.nlog = S_NSLOT; h.lo = 0; h.hi = NL; h.mode = 2; h.unitsL = h.unitsG = 1 << 30; h.rg = 3; h.rgd = 3; }
  const int T = h.T, RG = h.rg, RGD = h.rgd;
  const int ntt = (T + 255) / 256;          // column tiles of the dense GEMMs
  auto gd = [&](long items, int mx) { long g = items; if (g > mx) g = mx; if (g < 1) g = 1; return (int)g; };
  // dense GEMM grids: units of the pass, at most one block per SM
  const int xs2 = T <= XS_T ? 2 : 1;         // small passes: the two halves of a dense GEMM's K range are two units
  auto gt = [&](int which, int glob, int nrg) { return fixed ? nsm : gd((long)ntt * tc_units(which, glob, nrg, RG, RGD) * xs2, nsm); };
  const bool smc = T <= 64;
  if (a.flags & 2) {
    if (a.dev == 0) k_sched<<<1, 1024, 0, s>>>((PCtx*)a.ctx); else k_wait<<<1, 1024, 0, s>>>((PCtx*)a.ctx);
    k_swap1<<<S_SWMAX * SWB, 256, 0, s>>>(c);
    k_swap2<<<S_SWMAX * SWB, 256, 0, s>>>(c);
  }
  if (h.nprol) k_prologue<<<h.nprol * NL, NTB, 0, s>>>(c);
  // rows of this card (a quarter of the pass), with the three before them; the windows of o_proj and the router
  const int me = a.dev;
  const int nown = T <= REPL_T ? (T > 0 ? T : 1) : (T + 3) / 4 + 1, nhalo = nown + 3, nall = T > 0 ? T : 1;
  auto wt = [&](int halo) {     // column tiles of this card's window
    if (fixed) return 0;
    const int wl = own_lo(T, me) - halo, wlo = wl > 0 ? wl : 0, whi = own_hi(T, me);
    return whi > wlo ? (whi - 1) / 256 - wlo / 256 + 1 : 1;
  };
  // the attention-input fetch: blocks of 576 threads that loop over rows. (The routing fetch keeps one block per row:
  // with the same loop it was 12-25 us per layer slower at 3584 rows, whatever the grid.)
  const int nxi = gd(nall, 3 * nsm), nxr = nall;
  const int nhj = (h.nseq + h.nsnap) * 3;       // (fixed grids: the header holds the largest counts)
  if (h.lo == 0) k_embed<<<nown, RT, 0, s>>>(c);
  for (int i = h.lo; i < h.hi; i++) {
    const bool glob = (i % 6) == 5, mx = i >= 40;
    // (the attention mix of every layer but the first ran with the end of the layer before it: k_post_mix)
    if (glob) { if (i == h.lo) k_mix_attn<3><<<nown, RT, 0, s>>>(c, i); k_xin<3><<<nxi, 576, 0, s>>>(c, i); }
    else { if (i == h.lo) k_mix_attn<1><<<nown, RT, 0, s>>>(c, i); k_xin<1><<<nxi, 576, 0, s>>>(c, i); }
    if (i > h.lo) {             // the previous layer's short-convolution state: every card is past that layer now
      if (nhj) k_xhist<<<nhj, 576, 0, s>>>(c);
      if (h.nsnap) k_hist<<<h.nsnap, NTB, 0, s>>>(c, i - 1, 1);
      k_hist<<<h.nseq, NTB, 0, s>>>(c, i - 1, 0);
    }
    k_tc<TK_BF16, G_QKV><<<gt(G_QKV, glob, 0), TC_NT, TC_SMEM, s>>>(c, i);
    k_prep<<<T, NTB, 0, s>>>(c, i);
    k_tc<TK_BF16, G_RELB><<<fixed ? nsm : gd((long)((T * NQL + COLT - 1) / COLT) * tc_nchunks(G_RELB, 0, 8, tc_rg(G_RELB, RG, RGD)), nsm), TC_NT, TC_SMEM, s>>>(c, i);
    k_attn<<<fixed ? nsm : gd(glob ? h.unitsG : h.unitsL, nsm), (AT_SW + 2) * 32, AT_SMEM, s>>>(c, i);
    k_attn_fold<<<T, 96, 0, s>>>(c, i);
    // (768 threads for every pass size: one remote load per thread and row. With 256 threads a thread did three in
    // series, and the fixed program needed a second launch for small passes.)
    k_gather_o<<<nhalo, 768, 0, s>>>(c, i, -1);
    k_tc<TK_BF16, G_OPROJ><<<fixed ? nsm : gd((long)wt(3) * tc_units(G_OPROJ, 0, 36, RG, RGD) * xs2, nsm), TC_NT, TC_SMEM, s>>>(c, i);
    k_post_attn<<<nown, RT, 0, s>>>(c, i);
    k_tc<TK_BF16, G_ROUTER><<<fixed ? nsm : gd((long)wt(0) * tc_units(G_ROUTER, 0, 3, RG, RGD) * xs2, nsm), TC_NT, TC_SMEM, s>>>(c, i);
    k_route<<<nown, 384, 0, s>>>(c, i);
    if (mx) k_xroute<true><<<nxr, 320, 0, s>>>(c, i); else k_xroute<false><<<nxr, 320, 0, s>>>(c, i);
    k_assign<<<NE, NTB, 0, s>>>(c, i);
    const int ge = fixed ? nsm : gd((long)(T * 8 / COLT + NEC + 1) * tc_nchunks(G_W13, 0, 36, RG), nsm);
    const int ga = fixed ? 8 * nsm : gd((long)T * 8, 8 * nsm);
    if (mx) {
      k_packx<true><<<ga, 160, 0, s>>>(c, i);
      k_tc<TK_FP8, G_W13><<<ge, TC_NT, TC_SMEM, s>>>(c, i);
      k_act<true><<<ga, 160, 0, s>>>(c, i);
      k_tc<TK_FP8, G_W2><<<ge, TC_NT, TC_SMEM, s>>>(c, i);
    } else {
      k_packx<false><<<ga, 160, 0, s>>>(c, i);
      k_tc<TK_FP4, G_W13><<<ge, TC_NT, TC_SMEM, s>>>(c, i);
      k_act<false><<<ga, 160, 0, s>>>(c, i);
      k_tc<TK_FP4, G_W2><<<ge, TC_NT, TC_SMEM, s>>>(c, i);
    }
    k_combine<<<nhalo, 576, 0, s>>>(c, i);
    if (i + 1 >= h.hi) k_post_mlp<<<nown, RT, 0, s>>>(c, i);
    else if ((i + 1) % 6 == 5) k_post_mix<3><<<nown, RT, 0, s>>>(c, i);
    else k_post_mix<1><<<nown, RT, 0, s>>>(c, i);
  }
  const bool logits = h.hi == NL && h.nlog && h.mode;
  if (logits) k_final<<<h.nlog, RT, 0, s>>>(c);
  k_xfin<<<h.nlog > 0 ? h.nlog : 1, 576, 0, s>>>(c);       // every card is past the last layer; the final hidden rows
  if (h.hi > h.lo) {
    if (nhj) k_xhist<<<nhj, 576, 0, s>>>(c);
    if (h.nsnap) k_hist<<<h.nsnap, NTB, 0, s>>>(c, h.hi - 1, 1);
    k_hist<<<h.nseq, NTB, 0, s>>>(c, h.hi - 1, 0);
  }
  if (logits) {
    k_tc<TK_BF16, G_UNEMBED><<<fixed ? nsm : gd((long)((h.nlog + 255) / 256) * tc_units(G_UNEMBED, 0, 393, RG, RGD), nsm), TC_NT, TC_SMEM, s>>>(c, 0);
    k_join<<<1, 32, 0, s>>>(c);
  }
  return (int)cudaGetLastError();
}

// four launcher threads, one per card
struct Launcher {
  std::thread th[NCARD];
  std::mutex mu;
  std::condition_variable cv, done_cv;
  PassArgs args[NCARD];
  int gen[NCARD] = {0, 0, 0, 0}, want = 0, pending = 0, rc[NCARD] = {0, 0, 0, 0};
  bool started = false;
  void run(int c) {
    int seen = 0;
    for (;;) {
      PassArgs a;
      {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return gen[c] != seen; });
        seen = gen[c];
        a = args[c];
      }
      int r = launch_pass(a);
      {
        std::lock_guard<std::mutex> lk(mu);
        rc[c] = r;
        pending--;
      }
      done_cv.notify_all();
    }
  }
};
static Launcher& g_l = *new Launcher;   // never destroyed: its threads outlive main

extern "C" int pe_pass(const PassArgs* a, int n, int threads) {
  if (!threads) {
    int r = 0;
    for (int k = 0; k < n; k++) r |= launch_pass(a[k]);
    return r;
  }
  if (!g_l.started) {
    g_l.started = true;
    for (int c = 0; c < NCARD; c++) { g_l.th[c] = std::thread([c] { g_l.run(c); }); g_l.th[c].detach(); }
  }
  {
    std::lock_guard<std::mutex> lk(g_l.mu);
    for (int k = 0; k < n; k++) { g_l.args[k] = a[k]; g_l.gen[k]++; }
    g_l.pending = n;
  }
  g_l.cv.notify_all();
  std::unique_lock<std::mutex> lk(g_l.mu);
  g_l.done_cv.wait(lk, [&] { return g_l.pending == 0; });
  int r = 0;
  for (int k = 0; k < n; k++) r |= g_l.rc[k];
  return r;
}
// ---- device-side loop: set-up and the pump (a thread per card that keeps two pass programs enqueued)
extern "C" int pe_sched_bytes() { return (int)sizeof(Sched); }
extern "C" int pe_shared_bytes() { return (int)sizeof(SShared); }
extern "C" int pe_node_bytes() { return (int)sizeof(SNode); }
extern "C" int pe_nslot() { return S_NSLOT; }
extern "C" long pe_cache_bytes(int ncache, int npages) { return s_cache_alloc_nodes(ncache, npages) * (long)sizeof(SNode); }
extern "C" int pe_tab_ints(int tmax) { return s_tab_ints(tmax); }
extern "C" int pe_sched_init(void* ctx, int npages, int tmax, int umax, int ncache, void* freep, void* page_ref, void* cache, const double* cfg, void* cfg_dev, void* eload, void* emaster) {
  cudaSetDevice(0);
  cudaMemcpy(cfg_dev, cfg, 12 * sizeof(double), cudaMemcpyHostToDevice);
  k_sched_init<<<1, 32>>>((PCtx*)ctx, npages, tmax, umax, ncache, (int*)freep, (int*)page_ref, (SNode*)cache, (const double*)cfg_dev, (float*)eload, (int*)emaster);
  cudaDeviceSynchronize();
  return (int)cudaGetLastError();
}
struct Pump {
  std::thread th[NCARD];
  PassArgs args[NCARD];
  std::atomic<int> running{0};
  std::atomic<long> launched[NCARD], final{-1};
  int rc[NCARD] = {0, 0, 0, 0};
};
static Pump& g_p = *new Pump;
extern "C" void* pe_stream_create(int dev) {
  cudaSetDevice(dev);
  cudaStream_t s;
  if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return nullptr;
  return (void*)s;
}
extern "C" int pe_copy_in(int dev, void* stream, void* dst, const void* src, size_t n) {      // a prompt into the arena
  cudaSetDevice(dev);
  cudaError_t e = cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, (cudaStream_t)stream);
  if (e) return (int)e;
  return (int)cudaStreamSynchronize((cudaStream_t)stream);
}
extern "C" int pe_pump_start(const PassArgs* a, volatile unsigned* stop_word) {
  if (g_p.running.load()) return 0;
  g_p.final = -1; g_p.running = NCARD;
  for (int c = 0; c < NCARD; c++) {
    g_p.args[c] = a[c]; g_p.launched[c] = 0;
    g_p.th[c] = std::thread([c, stop_word] {
      cudaSetDevice(g_p.args[c].dev);
      cudaStream_t s = (cudaStream_t)g_p.args[c].stream;
      cudaEvent_t ev[2];
      cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming); cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming);
      long n = 0;
      for (;;) {
        if (*stop_word && g_p.final.load() < 0) {      // every card ends after the same number of programs
          long mx = 0;
          for (int k = 0; k < NCARD; k++) mx = std::max(mx, g_p.launched[k].load());
          long expect = -1;
          g_p.final.compare_exchange_strong(expect, mx + 1);
        }
        const long fin = g_p.final.load();
        if (fin >= 0 && n >= fin) break;
        // the pass before the last has ended: at most two enqueued. Polled, not cudaEventSynchronize: a blocking wait
        // keeps other calls on this card (the copy of an arriving prompt) out for up to a whole pass
        if (n >= 2) while (cudaEventQuery(ev[n & 1]) == cudaErrorNotReady) std::this_thread::sleep_for(std::chrono::microseconds(50));
        g_p.rc[c] |= launch_pass(g_p.args[c]);
        cudaEventRecord(ev[n & 1], s);
        n++; g_p.launched[c] = n;
      }
      cudaStreamSynchronize(s);
      g_p.running--;
    });
    g_p.th[c].detach();
  }
  return 0;
}
extern "C" int pe_pump_stop() {
  for (int k = 0; k < 30000 && g_p.running.load(); k++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  int r = 0;
  for (int c = 0; c < NCARD; c++) r |= g_p.rc[c];
  return g_p.running.load() ? -1 : r;
}
extern "C" long pe_pump_launched(int c) { return g_p.launched[c].load(); }
extern "C" int pe_sync(int dev, void* stream) {
  cudaSetDevice(dev);
  return (int)cudaStreamSynchronize((cudaStream_t)stream);
}
