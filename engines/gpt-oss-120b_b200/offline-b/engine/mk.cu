// gpt-oss-120b decode megakernel for one B200 (sm_100a): one persistent cooperative kernel per step / generate call /
// served cohort. 152 blocks (one per SM) x 512 threads.
//
// Every matrix-vector product is a chain of mma.m16n8k16 (bf16 x bf16 -> f32); the 8 columns of the B operand are
// the sequences of the batch. Weights are re-laid at load time (a permutation of the stored bytes, no value change):
//  - BF16 matrices: 512-byte tiles [rowtile][kstep][32 lanes][16 B] in A-fragment order.
//  - MXFP4 experts: per (rowtile, 64-column chunk) 512 B of codes [32 lanes][16 B] (4 mma A fragments of 4 bytes,
//    each byte = 2 e2m1 codes = one bf16x2 register after cvt.rn.bf16x2.e2m1x2) and 32 B of E8M0 scales [8 g][4 B].
//    The scale is applied in float32 to the partial sum of each 32-column group (exact: power of two).
// Phases per layer (v1: grid barrier between phases):
//  A  norm1 + qkv (+bias, bf16, RoPE) -> q buffer, K/V cache
//  B  attention, one block per (sequence, kv head) -> attn
//  C  o_proj (+bias) + residual -> xm
//  D  norm2 (every block) ; router on 8 blocks -> logits
//  E  top-4 (every block) ; gate_up of the union of chosen experts (+bias, swiglu) -> act
//  F  down for owned rows x all chosen experts, weighted sum, residual -> x
// Head: final norm, lm_head -> logits / argmax.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "mkprof.cuh"

typedef unsigned u32;
typedef unsigned short u16;
typedef unsigned long long u64;
typedef __nv_bfloat16 bf16;

#define HID 2880
#define NL 36
#define NQH 64
#define NKV 8
#define HD 64
#define QKVR 5120
#define NE 128
#define VOCAB 201088
#define NB 152
#ifndef NT
#define NT 256
#endif
#define NW (NT / 32)
#define MAXB 8
#ifndef RQ
#define RQ 4        // the router's K is split in RQ quarters over 8 * RQ blocks; E sums the RQ partials
#endif
#ifndef FLAT_MAXB
#define FLAT_MAXB 2   // batches up to this use the flat instance (norm_to_smem sizes its registers by it)
#endif
#define XSH (HID + 8)      // smem row stride (bf16) of a hidden-size vector
#define XSA (4096 + 8)     // smem row stride of the attention output
#define KS_H 180           // ksteps (16) of K=2880
#define KS_A 256           // ksteps of K=4096
#define KC 45              // 64-chunks of K=2880
#define RT_QKV 320
#define RT_O 180
#define RT_GU 360
#define RT_DN 180
#define RT_HEAD 12568
// track B head sketch (greedy paths of the B <= 2 instance): layout of the red area during the head
#ifndef HSK
#define HSK 0
#endif
#ifndef ATMA_FLAT
#define ATMA_FLAT 1   // the attention-output bulk copy in the B <= 2 instance too (with FTMA_FLAT: B=1 -22 us)
#endif
#ifndef FTMA_FLAT
#define FTMA_FLAT 1   // the act-row bulk copies in the B <= 2 instance too
#endif
#ifndef BCOV
#define BCOV 0        // 1: no grid barrier between attention and o_proj (per-kv-head counters, per-task copies): B=1 +65, B=8 +68 us
#endif
#ifndef OALIGN
#define OALIGN 1      // o_proj segment bounds on whole ring stages (63, 63, 63, 67 ksteps): B=8 -39 us
#endif
#ifndef CSPLIT
#define CSPLIT 0      // 1: two-rowtile o_proj blocks share half their second rowtile with a partner (same bits; +33 us)
#endif
#ifndef FHALF
#ifndef FCOMB
#define FCOMB 0       // B=8 +76 us. B >= 3: the block finishing a rowtile's last down unit does its combine (no combine step)
#endif
#if FCOMB && FHALF
#error FCOMB needs FHALF 0
#endif
#define FHALF 0       // 1: B >= 3 down units as two K-half tasks, same order in B <= 2 (bit-identical; B=8 +163 us)
#endif
#define task_tail(t) (((t) >> 29) & 1)
#ifndef ETAIL
#define ETAIL (!HSK)  // (track B after s0042: 0 is -5.8 us at B=8; track A: 0 is +23) B >= 3 gate_up: the n mod 8 remainder units as one-stage segments (per-warp balance, same bits)
#endif
#ifndef FTAIL
#define FTAIL 0       // B >= 3 down phase: the n mod 8 remainder units as one-stage segments (same bits; s0027: -5 us, after s0034: +10 us on the HTTP-like bench)
#endif
#ifndef EPF
#define EPF 0         // 1: B <= 2: the down phase's first stages requested before gate_up's epilogue (B=1 +25 us)
#endif
#ifndef NORM_TMA_A
#define NORM_TMA_A 0
#endif
#ifndef NORM_TMA_D
#define NORM_TMA_D 0
#endif
#ifndef NORM_TMA
#define NORM_TMA 0   // 1: B >= 3 A / D norms stage their input rows by bulk copies: B=8 +160 us (queued behind the prefetched weight stages)
#endif
#ifndef OSPLIT
#define OSPLIT 0      // 1: one o_proj rowtile per block, the 28 extra rowtiles in K quarters, summed at D's start into each
                      // block's norm input: C -0.9, D +0.7, A +1.0 us/layer (allocation); first form +1000 us at B=8
#endif
#ifndef HSK_L2PF
#define HSK_L2PF 0    // items of each warp's first sketch task pulled into L2 at the head's prefetch (0: off)
#endif
#ifndef PREP_UNROLL
#define PREP_UNROLL 1   // unroll factor of the wide sketch prep's per-element loops
#endif
#ifndef PREP_RESTRICT
#define PREP_RESTRICT 0
#endif
constexpr int kPrepUnroll = PREP_UNROLL;   // (pragma arguments are not macro-expanded)
#ifndef HSK_TAILREG
#define HSK_TAILREG 0 // 1: B <= 2 sketch tail records by plain loads, 10 whole stages per task (B=1 +10 us)
#endif
#define RS_SLOTS ((NW * RS * RI * RING_ITEM) / (RI * 512))   // 9-item bf16 stages that fit in the pooled ring (25)
#ifndef STEPARG
#define STEPARG 1     // the per-step token pick as a warp-parallel argmax over the per-block candidates
#endif
#ifndef HSK_RSPRE
#define HSK_RSPRE 0   // candidate rescoring (FASTRS): a stage's fragments loaded before its mma chain
#endif
#ifndef HSK_RSROLL
#define HSK_RSROLL 0  // candidate rescoring (FASTRS): the stage's items in a rolled loop over pairs (same order)
#endif
#ifndef HSK_L2PFW
#define HSK_L2PFW 0   // wide sketch: each warp's first task pulled into L2 at the head start, during the prep
#endif
#ifndef HSK_WARM
#define HSK_WARM 0    // B <= 2 sketch: the rescoring code warmed by a dummy 2-stage pass while waiting at the barrier
#endif
#ifndef PFDLY
#define PFDLY 0       // ns that blocks without head candidates wait before the next step's cross-barrier prefetch
#endif
#ifndef HSK_RQ
#define HSK_RQ 0      // wide sketch: candidate rowtiles through a global queue, tile j rescored by block j mod NB
#endif
#define HQ_CAP 12800  // HSK_RQ queue ring (>= the 12568 head rowtiles: one step never wraps onto itself)
#ifndef NORM_PIPE1
#define NORM_PIPE1 1  // norm_warp (B >= 3): each norm weight word loaded one iteration ahead of its use (track B
                      // build: B=8 -23.5 us; track A: +25 alone, -2.7 together with HTAIL)
#endif
#if NORM_PIPE1 && DQ
#error "DQ needs NORM_PIPE1 0 (the pipelined store loop stores every column)"
#endif
#ifndef DQ
#define DQ 0          // B >= 3 router blocks: norm2 of their K quarter before the router, the rest after arriving
#endif
#ifndef ATT_BLKW
#define ATT_BLKW 2   // chunks per warp allowed in ATT_BLK's B >= 3 form (partials in xs: up to 30 chunks; 2: s0027 +15 us, after s0035 -11)
#endif
#ifndef ATT_BLKW_F
#define ATT_BLKW_F 1 // the same for the B <= 2 instance (> 1: partials in xs, which needs V_ASYNC_FLAT 0)
#endif
#ifndef ATT_BLK
#define ATT_BLK 1    // units with at most NW chunks: one block per (sequence, kv head), partials merged in smem
#endif
#ifndef HTS
#define HTS 2         // HTAIL: K segments per tail rowtile (KS_H / HTS ksteps each: whole ring stages for 2, 4, 5, 10)
#endif
#ifndef HTAIL
#define HTAIL (!HSK)  // track A exact head: each block's n mod 8 last rowtiles as 4 K segments spread over the warps
#endif
#if HTAIL && HEAD_OL
#error "HTAIL: HEAD_OL's out-of-line consume does not know the segment tasks"
#endif
#if HTAIL && HSK
#error "HTAIL changes the exact head's sums: track B's candidate rescoring would no longer match step()"
#endif
#ifndef QXO
#define QXO (HSK ? 3 : 0) // (track B: B=1 -4.5, B=8 +3.3 us) QX with the V(pos) sum as an out-of-line call (1: both instances, 2: B >= 3 only, 3: B <= 2 only)
#endif
#ifndef EBPF
#define EBPF 0        // B <= 2: E's start pulls this block's gate_up / down bias rows into L2
#endif
#ifndef SPF_ROPE
#define SPF_ROPE 0    // SPF also pulls the RoPE cos / sin row of the position into L2
#endif
#ifndef SPF
#define SPF HSK       // phase A pulls the layer's bias slices, router bias and sinks into L2 (track B: B=1 -5.7 us; track A: B=8 +13)
#endif
#ifndef KVPF_F
#define KVPF_F HSK    // KVPF in the B <= 2 instance (track A: +13 us at B=1 there; KVPF itself: the B >= 3 instance)
#endif
#ifndef KVPF_PLACE
#define KVPF_PLACE 0  // 1: KVPF issued with the previous phase's cross-barrier prefetch of the layer's qkv weights
#endif
#ifndef KVPF_OL
#define KVPF_OL 0     // KVPF as an out-of-line call
#endif
#ifndef KVPF
#define KVPF 1        // (track A after s0048: -33 us at B=8) phase A pulls the K/V chunks of this block's attention items into L2 (track B: B=1 -14, B=8 -21 us; track A: +17)
#endif
#ifndef HSK_XG
#define HSK_XG 0      // (s0034: 1 was -15 us; after s0044: 0 is -24 us at B=8) wide sketch: normed head input published by blocks 0..B-1, copied (not re-normed) by candidate blocks
#endif
#ifndef HSK_WIDE_OL
#define HSK_WIDE_OL 1 // the wide sketch called out of line (B=8 serve -16 us; a plain out-of-line call costs ~11, inline ~33)
#endif
#ifndef HEAD_OL
#define HEAD_OL 0     // study: the wide instance's exact head consume as an out-of-line call
#endif
#ifndef FASTRS_FMAX
#define FASTRS_FMAX 3   // B <= 2: candidate tiles rescored at once
#endif
#ifndef FASTRS_WMAX
#define FASTRS_WMAX 8   // wide: candidate tiles rescored at once (pooled slots shared out, >= 3 each)
#endif
#ifndef HSK_FASTRS_W
#define HSK_FASTRS_W 1  // the pooled-ring parallel rescoring in the wide sketch too (B=8: +5 us at s0031, -10 after s0045)
#endif
#ifndef HSK_FASTRS
#define HSK_FASTRS 1  // B <= 2 sketch: a candidate rowtile is rescored with all 20 stages in flight in the idle ring
#endif
#ifndef HSK_WIDE
#define HSK_WIDE 1    // the sketch in the B >= 3 instance too, out of line (HSK_WIDE_OL); inlined it cost the layers
                      // more than the head gained after ATMA/FTMA
#endif
#define HSK_ITEMS 94          // 90 int8 chunks + 3 error-norm tiles + 1 row-constant record
#define HSK_XST 2896          // bytes per digit row (8 rows: hi digits of columns 0..3, lo digits of columns 0..3)
#define HSK_XNF (8 * HSK_XST) // 3 k16 chunks x 32 lanes x 8 B: column block norms as B fragments
#define HSK_ST (HSK_XNF + 768)        // floats [0..7] 2^-k, [8..15] |r| term, [16..23] |x| term, [24..31] global bound;
                                      // ints [32..39] best-lower-bound keys, [40] count, [41] overflow, [42] ncand, [43..45] mask
#define HSK_LIST (HSK_ST + 256)       // candidates (row | column << 24, upper bound)
#define HSK_CAP 128
#define HSK_CAPW (HSK_XNF / 8)       // wide: the list takes the (unused) digit area of red
#define HSK_CAND (HSK_LIST + HSK_CAP * 8)   // candidate rowtiles of the block

#ifndef NSEG_QKV
#define NSEG_QKV 10
#endif
#ifndef NSEG_O
#define NSEG_O 4      // o_proj K split (re-tuned at ~9.5 agent-h: 16 -> 4: B=1 -30, B=8 -86 us; 2 and 8 worse)
#endif
#ifndef NSEG_GU
#define NSEG_GU 5
#endif
#ifndef NSEG_DN
#define NSEG_DN 5
#endif
#ifndef RTASK
#define RTASK 5       // router tasks per router block (K pieces of its quarter: 45 ksteps = 5 full stages; NW=8: +5 us)
#endif
#ifndef DA
#define DA 8
#endif
#ifndef DB
#define DB 4
#endif
#ifndef ROUTER_BF16
#define ROUTER_BF16 1      // router logits rounded to bf16 before top-4 (as the BF16 path and the general engines do)
#endif

struct LayerP {
  const uint4* wqkv; const float* bqkv;
  const uint4* wo; const float* bo;
  const uint4* wr; const float* br;
  const uint4* gu_c; const u32* gu_s; const float* gu_b;
  const uint4* dn_c; const u32* dn_s; const float* dn_b;
  const float* ln1; const float* ln2; const float* sinks;
  const u16* ln1b; const u16* ln2b;        // the same norm weights as BF16 (exact: they are stored BF16)
  bf16* kc; bf16* vc;
  const uint4* wp;                         // expert predictor: router rows x (ln2 / ln1) as BF16 tiles (only steers prefetch)
  int sliding;
};

struct Params {
  LayerP L[NL];
  const bf16* embed; const uint4* head; const float* lnf; const u16* lnfb;
  const float* cosT; const float* sinT;     // [maxctx][32] (bf16 values as float)
  bf16 *x, *xm, *q, *attn, *act;           // act: [32 slots][MAXB][HID]
  float* rlog;                              // [MAXB][128]
  float* ypart;                             // [32 slots][MAXB][HID] down outputs (+bias, unweighted), flat path (B <= 2)
  u32* ecnt;                                // [32] gate_up rowtiles finished per expert slot (grows; reset at exit)
  float* apart;                             // [64 units][32 chunks][8 heads][66] attention partials (acc, m, l)
  u32* acnt;                                // [64] chunks finished per (sequence, kv head); reset by the merging warp
  float* amv; int* ami;                     // [NB][MAXB]
  u32* bar;                                 // [0] barrier counter, [1] exit counter
  int maxctx;
  unsigned long long* prof; unsigned prof_cap;   // mkprof dump (only with -DMKPROF)
  const unsigned char* skel;                // -DSKEL: every weight load reads a 512 KB window here (results wrong)
  float* plog;                              // [RQ][MAXB][128] predictor partials (XPF)
  const uint4* hsk;                         // HSK: per head rowtile 94 records of 512 B: 90 int8 A chunks (k32), 3 BF16
                                            // tiles of the error-block norms (k = 45 blocks + 3 zero), row constants
  float* hsl;                               // HSK: [NB][MAXB] each block's best lower bound; then (HSK_XG) bf16 [MAXB][HID] normed head input, fragment order
  int* hdbg;                                // HSK: [0] candidate rowtiles (sum), [1] overflowing blocks, [2] heads
  float* opart;                             // OSPLIT: [112 quarter pieces][16 rows][MAXB] o_proj partial sums
  u32* adone;                               // BCOV: [8] attention outputs finished per kv head (grows; reset at exit)
  float* cpart;                             // CSPLIT: [28 shared rowtiles][2 segments][128] partner partials (mma layout)
  u32* ccnt;                                // CSPLIT: [28] partner segments finished (grows; reset at exit)
  float* ypart1;                            // FHALF: [32 slots][MAXB][HID] second K half of the wide down units
  u32* fcnt;                                // FCOMB: [180] down units finished per rowtile (the last one resets it)
};

struct RunArgs {
  const long long* tok; int tok_stride;     // mode 0/1: tok[b]; mode 2: tok[b*tok_stride + t] (prompt ids)
  float* logits;                            // mode 0: [B][VOCAB]
  long long* out_tok;                       // mode 1: [B][nsteps]
  int* host_tok; volatile int* host_prog;   // mode 2: host-mapped [B][nsteps] and progress
  int B, pos0, nprompt, nsteps, mode;
};

// ------------------------------------------------------------------------------------------------ primitives
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// int8 x int8 -> s32 (accumulator bits kept in float registers)
__device__ __forceinline__ void mma_s8(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("{ .reg .s32 d0, d1, d2, d3;\n"
      "mov.b32 d0, %0; mov.b32 d1, %1; mov.b32 d2, %2; mov.b32 d3, %3;\n"
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {d0,d1,d2,d3},{%4,%5,%6,%7},{%8,%9},{d0,d1,d2,d3};\n"
      "mov.b32 %0, d0; mov.b32 %1, d1; mov.b32 %2, d2; mov.b32 %3, d3; }"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16816z(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}
#ifdef SKEL
__device__ const unsigned char* g_skel;
#define WP(p) ((const void*)(g_skel + ((uintptr_t)(p) & 0x7FFF0ull)))
#else
#define WP(p) (p)
#endif
__device__ __forceinline__ uint4 ldw(const void* p) {
  p = WP(p);
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ u32 ldw32(const void* p) {
  p = WP(p);
  u32 r; asm volatile("ld.global.nc.L1::no_allocate.u32 %0,[%1];" : "=r"(r) : "l"(p)); return r;
}
// data written by other blocks during this launch: bypass L1
__device__ __forceinline__ u32 ldcg32(const void* p) { u32 r; asm volatile("ld.global.cg.u32 %0,[%1];" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ float ldcgf(const float* p) { float r; asm volatile("ld.global.cg.f32 %0,[%1];" : "=f"(r) : "l"(p)); return r; }
__device__ __forceinline__ int ldcgi(const int* p) { int r; asm volatile("ld.global.cg.s32 %0,[%1];" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ u16 ldcg16(const void* p) { u16 r; asm volatile("ld.global.cg.u16 %0,[%1];" : "=h"(r) : "l"(p)); return r; }
__device__ __forceinline__ uint4 ldcg128(const void* p) {
  uint4 r; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r;
}
__device__ __forceinline__ void sig_add_rel(u32* c, u32 v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(c), "r"(v) : "memory"); }
__device__ __forceinline__ u32 ld_acquire(const u32* p) { u32 r; asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ void cvt4(u32 x, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4; cvt.rn.bf16x2.e2m1x2 %0, b0; cvt.rn.bf16x2.e2m1x2 %1, b1;"
      " cvt.rn.bf16x2.e2m1x2 %2, b2; cvt.rn.bf16x2.e2m1x2 %3, b3; }"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(x));
}
__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float u16f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ u16 f2u16(float v) { bf16 h = __float2bfloat16_rn(v); return *reinterpret_cast<u16*>(&h); }
__device__ __forceinline__ float scale_f(u32 s, int k) { return __uint_as_float(((s >> (8 * k)) & 0xffu) << 23); }

#ifdef SIG_OLD
#define SIG_FENCE() __threadfence()
__device__ __forceinline__ void sig_add(u32* c, u32 v) { atomicAdd(c, v); }
#else
#define SIG_FENCE() do { } while (0)
__device__ __forceinline__ void sig_add(u32* c, u32 v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(c), "r"(v) : "memory"); }
#endif
#define MARK(tag) do { if (threadIdx.x == 0) MKPROF_MARK(prof, pn, tag); } while (0)
__device__ __forceinline__ void grid_bar(u32* ctr, u32& target) {
#ifdef NOBAR
  __syncthreads(); return;
#endif
  __syncthreads();
  target += NB;
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(ctr, 1u);
    while (ld_acquire(ctr) < target) { }
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------ GEMV tasks
// C fragment of one task = 16 rows x 8 columns; lane (g,t): c0,c1 = row g cols 2t,2t+1; c2,c3 = row g+8.
// one 64-column chunk of an MXFP4 rowtile: codes q (16 B of this lane), scale word s (4 B), B fragments b[8].
// cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 converts two e2m1 codes and multiplies each by 2^(S-127) of its own scale
// byte: exact for every code and every scale in 100..150 (bench/cvt3.cu, exhaustive; the checkpoint's are 111..138),
// so the A operand is the dequantised BF16 weight and the mma accumulates as the BF16-weight data flow does.
#ifndef FP4_POSTSCALE
__device__ __forceinline__ void cvt4s(u32 x, u32 sa, u32 sb, u32& r0, u32& r1, u32& r2, u32& r3) {
  asm("{ .reg .b8 b0,b1,b2,b3; .reg .b16 ha, hb; mov.b32 {b0,b1,b2,b3}, %4; cvt.u16.u32 ha, %5; cvt.u16.u32 hb, %6;"
      " cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %0, b0, ha; cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %1, b1, hb;"
      " cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %2, b2, ha; cvt.rn.scaled::n2::ue8m0.bf16x2.e2m1x2 %3, b3, hb; }"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(x), "r"(sa), "r"(sb));
}
__device__ __forceinline__ void chunk_fp4(float* c, uint4 q, u32 s, const u32* b) {
  const u32 sg0 = __byte_perm(s, 0, 0x0000), sg1 = __byte_perm(s, 0, 0x1111);
  const u32 sh0 = __byte_perm(s, 0, 0x2222), sh1 = __byte_perm(s, 0, 0x3333);
  u32 a0, a1, a2, a3;
#ifdef FP4_SPLIT2
  // two independent 2-mma chains per item (k 0..31 into c, k 32..63 into t), summed at the item's end
  float t[4];
  u32 e0, e1, e2, e3;
  cvt4s(q.x, sg0, sh0, a0, a1, a2, a3); cvt4s(q.z, sg1, sh1, e0, e1, e2, e3);
  mma16816(c, a0, a1, a2, a3, b[0], b[1]); mma16816z(t, e0, e1, e2, e3, b[4], b[5]);
  cvt4s(q.y, sg0, sh0, a0, a1, a2, a3); cvt4s(q.w, sg1, sh1, e0, e1, e2, e3);
  mma16816(c, a0, a1, a2, a3, b[2], b[3]); mma16816(t, e0, e1, e2, e3, b[6], b[7]);
  c[0] += t[0]; c[1] += t[1]; c[2] += t[2]; c[3] += t[3];
#else
  cvt4s(q.x, sg0, sh0, a0, a1, a2, a3); mma16816(c, a0, a1, a2, a3, b[0], b[1]);
  cvt4s(q.y, sg0, sh0, a0, a1, a2, a3); mma16816(c, a0, a1, a2, a3, b[2], b[3]);
  cvt4s(q.z, sg1, sh1, a0, a1, a2, a3); mma16816(c, a0, a1, a2, a3, b[4], b[5]);
  cvt4s(q.w, sg1, sh1, a0, a1, a2, a3); mma16816(c, a0, a1, a2, a3, b[6], b[7]);
#endif
}
#else
__device__ __forceinline__ void chunk_fp4(float* c, uint4 q, u32 s, const u32* b) {
  float t0[4], t1[4];
  u32 a0, a1, a2, a3, e0, e1, e2, e3;
  cvt4(q.x, a0, a1, a2, a3); cvt4(q.z, e0, e1, e2, e3);
  mma16816z(t0, a0, a1, a2, a3, b[0], b[1]);
  mma16816z(t1, e0, e1, e2, e3, b[4], b[5]);
  cvt4(q.y, a0, a1, a2, a3); cvt4(q.w, e0, e1, e2, e3);
  mma16816(t0, a0, a1, a2, a3, b[2], b[3]);
  mma16816(t1, e0, e1, e2, e3, b[6], b[7]);
  const float s0 = scale_f(s, 0), s2 = scale_f(s, 2), s1 = scale_f(s, 1), s3 = scale_f(s, 3);
  c[0] = fmaf(t0[0], s0, c[0]); c[1] = fmaf(t0[1], s0, c[1]);
  c[2] = fmaf(t0[2], s2, c[2]); c[3] = fmaf(t0[3], s2, c[3]);
  c[0] = fmaf(t1[0], s1, c[0]); c[1] = fmaf(t1[1], s1, c[1]);
  c[2] = fmaf(t1[2], s3, c[2]); c[3] = fmaf(t1[3], s3, c[3]);
}
#endif
__device__ __forceinline__ void store_task(float* red, int slot, const float* c) {
  const int lane = threadIdx.x & 31;
  *reinterpret_cast<float4*>(red + slot * 128 + lane * 4) = make_float4(c[0], c[1], c[2], c[3]);
}

// ------------------------------------------------------------------------------------------------ TMA task runner
// Every warp streams its weights through its own ring of RS stages in shared memory, filled by bulk copies
// (cp.async.bulk, completion on an mbarrier per stage) that lane 0 issues: bytes in flight cost no registers.
// A warp runs tasks warp, warp+NW, ... < ntask (at most 32 per warp and call). A task = n consecutive items of one
// rowtile, contiguous in memory (bf16: 512 B tile = one mma; fp4: 512 B codes + 32 B scales = four mmas). A stage
// holds up to RI items of one task. B operands come from shared memory: row (rows8 >> 8g & 255) of the task's x
// base, live when bit g of bm is set. fin(slot, c) consumes a task's 16x8 result fragment.
#ifndef RS
#define RS 3
#endif
#ifndef BPRED
#define BPRED 0      // 1: B loads of absent columns predicated off (harness: slower, 39 -> 35 GB/s/SM)
#endif
#ifndef RI
#define RI 9
#endif
#define RING_ITEM 544
#define RING_WARP (RS * RI * RING_ITEM)
struct TD {
  const uint4* w;      // codes / tiles of item 0 (32 uint4 per item)
  const u32* s;        // fp4: scales of item 0 (8 words per item)
  const u16* x;        // B base for item 0 (smem): element (row, k) at x + row * xst + k
  int n;               // items
  int slot;            // where the result goes
  u32 bm;              // bit g set: B column g is live
  u64 rows8;           // byte g: smem row of column g
  int k0;              // first kstep (bf16) / chunk (fp4) of the task inside the B rows
};
struct TT { const uint4* w; const u32* s; const u16* x; u64 rows8; int n; int slot; u32 bm; int k0; };
#ifndef TTN
#define TTN 16       // task-table rows per warp (most tasks a warp gets in one call: head 11, E wide 8)
#endif
#define SM_TT_BYTES (NW * TTN * (int)sizeof(TT))
#define ROWS_ID 0x0706050403020100ull

__device__ __forceinline__ u32 smaddr(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, u32 n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 b, u32 ph) {
  asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W%=; }" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void tma_load(u32 dst, const void* src, u32 bytes, u32 bar) {
#ifdef SKEL
  src = WP(src);
#endif
#ifdef TMA_NOHINT
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
#else
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(0x12F0000000000000ull) : "memory");
#endif
}

__device__ __forceinline__ void l2_prefetch(const void* src, u32 bytes) {
#ifdef SKEL
  src = WP(src);
#endif
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(bytes) : "memory");
}
struct Ring { u32 data; u32 bars; u32 pos; const unsigned char* gdata; };   // smem addresses of this warp's stages and mbarriers; stages consumed
// producer state of a warp's task list (kept in shared memory across a grid barrier)
struct PState { int nt, pt, pi, kind; u32 issued; int pad[3]; };

// lane 0 issues one stage: the next (up to RI) items of the current task
template <int KIND>
__device__ __forceinline__ void issue_stage(const TT* tt, PState& ps, const Ring& rg, int lane) {
  constexpr int IB = 512, SB = (KIND == 1) ? 32 : 0;
  const u32 sidx = ps.issued % RS;
  const u32 bar = rg.bars + sidx * 8, dst = rg.data + sidx * (RI * RING_ITEM);
  const TT& q = tt[ps.pt];
  const int m = min(RI, q.n - ps.pi);
  if (lane == 0) {
    mbar_expect(bar, m * (IB + SB));
    if (KIND == 1) tma_load(dst, q.w + (size_t)ps.pi * 34, m * 544, bar);   // records: 512 B codes + 32 B scales
    else tma_load(dst, q.w + (size_t)ps.pi * 32, m * IB, bar);
  }
  ps.pi += m;
  if (ps.pi == q.n) { ps.pt++; ps.pi = 0; }
  ps.issued++;
}

// build the warp's task table and issue the first RS stages (the ring must be drained: rg.pos == issued)
#ifndef PF_STAGES
#define PF_STAGES 2
#endif
#ifndef PF_STAGES_W
#if HSK
#define PF_STAGES_W PF_STAGES   // the B >= 3 instance's prefetch depth (the kernel's call sites pass it); track B: 3 is +6 us
#else
#define PF_STAGES_W 3           // track A build: B=8 -19 us on the HTTP-like bench (B=1 instance unaffected)
#endif
#endif
template <int KIND, int PFS = PF_STAGES, class Desc>
__device__ __forceinline__ void prefetch_tasks(int ntask, TT* table, const Ring& rg, PState& ps, Desc desc) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nt = (ntask > warp) ? (ntask - warp + NW - 1) / NW : 0;

  TT* tt = table + warp * TTN;
  if (lane < nt) {
    TD d; d.s = nullptr; d.rows8 = ROWS_ID; d.k0 = 0; desc(warp + NW * lane, d);
    tt[lane] = TT{d.w, d.s, d.x, d.rows8, d.n, d.slot, d.bm, d.k0};
  }
  __syncwarp();
#ifndef L2PF
#define L2PF 0     // bit 0: fp4 (expert) task lists, bit 1: bf16 task lists of at most 64 items per task (not the head)
#endif
  // the whole task list into L2 at once (the ring then reads at L2 speed); the first RS stages are skipped
  if (lane < nt && (((L2PF & 1) && KIND == 1) || ((L2PF & 2) && KIND == 0 && tt[0].n <= 64))) {
    const TT& q = tt[lane];
    const int skip = (lane == 0) ? min(q.n, RS * RI) : 0;
    if (q.n > skip) {
      if (KIND == 1) l2_prefetch(q.w + (size_t)skip * 34, (q.n - skip) * 544);
      else l2_prefetch(q.w + (size_t)skip * 32, (q.n - skip) * 512);
    }
  }
  PState q; q.nt = nt; q.pt = 0; q.pi = 0; q.kind = KIND; q.issued = rg.pos;
  for (int k = 0; k < PFS && q.pt < nt; k++) issue_stage<KIND>(tt, q, rg, lane);
  if (lane == 0) ps = q;
  __syncwarp();
}

__device__ __forceinline__ u32 lds32(u32 a) { u32 r; asm("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(a)); return r; }
__device__ __forceinline__ uint4 lds128(u32 a) {
  uint4 r; asm("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a)); return r;
}
// one item of a stage: A from the ring stage at st, B from shared memory at xb (byte address of this lane's row)
template <int KIND>
__device__ __forceinline__ void item(float* c, u32 st, u32 xb, int kb, int i, int lane, int g, bool cb) {
  constexpr int IB = 512, SB = 32;
#ifdef NOCONSUME
  if (KIND <= 1) { c[0] += 1.f; return; }    // study switch: the runner without per-item work (results wrong)
#endif
  const uint4 q = lds128(st + i * (KIND == 1 ? 544 : IB) + lane * 16);
  if (KIND == 2) {
    // int8 head sketch chunk (k32): B = 8 int8 digit columns, lane's 8 bytes at xb + chunk * 32
    u32 b0, b1;
    asm("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(xb + (kb + i) * 32));
    mma_s8(c, q.x, q.y, q.z, q.w, b0, b1);
  } else if (KIND == 0) {
    const int k = kb + i;
    u32 b0, b1;
    asm("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(xb + (k >> 2) * 128 + (k & 3) * 8));
#ifdef KEEPSEL
    b0 = cb ? b0 : 0u; b1 = cb ? b1 : 0u;
#endif
    mma16816(c, q.x, q.y, q.z, q.w, b0, b1);
  } else {
    const u32 sw = lds32(st + i * 544 + 512 + g * 4);
#if BPRED
    // B columns of absent sequences are not loaded (their registers keep stale values: they only feed output
    // columns nobody reads), so shared-memory bandwidth goes to the live columns only
    uint4 x0, x1;
    asm("{ .reg .pred p; setp.ne.b32 p, %8, 0;\n"
                 "@p ld.shared.v4.u32 {%0,%1,%2,%3}, [%9];\n"
                 "@p ld.shared.v4.u32 {%4,%5,%6,%7}, [%10]; }"
                 : "+r"(x0.x), "+r"(x0.y), "+r"(x0.z), "+r"(x0.w), "+r"(x1.x), "+r"(x1.y), "+r"(x1.z), "+r"(x1.w)
                 : "r"((u32)cb), "r"(xb + (kb + i) * 128), "r"(xb + (kb + i) * 128 + 16));
#else
    const uint4 x0 = lds128(xb + (kb + i) * 128), x1 = lds128(xb + (kb + i) * 128 + 16);
#endif
    u32 b[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#ifdef KEEPSEL
#pragma unroll
    for (int j = 0; j < 8; j++) b[j] = cb ? b[j] : 0u;
#endif
#ifdef NOCVT
    { float t0[4]; mma16816z(t0, q.x, q.y, q.z, q.w, b[0], b[1]); mma16816(t0, q.y, q.x, q.z, q.w, b[2], b[3]);
      mma16816(t0, q.z, q.y, q.x, q.w, b[4], b[5]); mma16816(t0, q.w, q.y, q.z, q.x, b[6], b[7]);
      c[0] = fmaf(t0[0], scale_f(sw, 0), c[0]); c[1] += t0[1]; c[2] += t0[2]; c[3] += t0[3]; }
#else
    chunk_fp4(c, q, sw, b);
#endif
  }
}

// issue more stages of the prefetched list until RS are in flight (call right after a barrier)
template <int KIND>
__device__ __forceinline__ void topup_tasks(TT* table, const Ring& rg, PState& psm) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  PState ps = psm;
  const TT* tt = table + warp * TTN;
  bool any = false;
  while (ps.pt < ps.nt && (int)(ps.issued - rg.pos) < RS) { issue_stage<KIND>(tt, ps, rg, lane); any = true; }
  if (any) { __syncwarp(); if (lane == 0) psm = ps; __syncwarp(); }
}

// consume the prefetched task list (more stages are issued as stages are freed)
struct NoPre { __device__ __forceinline__ void operator()(int) const {} };
template <int KIND, class Fin, class Pre = NoPre>
__device__ __forceinline__ void consume_tasks(TT* table, int xst, Ring& rg, PState& psm, Fin fin, Pre pre = Pre()) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  PState ps = psm;
  const int nt = ps.nt;
  if (nt == 0) return;
  const TT* tt = table + warp * TTN;
  while (ps.pt < nt && (int)(ps.issued - rg.pos) < RS) issue_stage<KIND>(tt, ps, rg, lane);   // top up the ring
  constexpr int IB = 512, SB = (KIND == 1) ? 32 : 0;
  constexpr int XST = (KIND == 0) ? 16 : 64;
  int ct = 0, ci = 0;
  TT cq = tt[0];
  u32 cx; bool cb;   // shared-space byte address of this lane's B row
  float bpre[4] = {0.f, 0.f, 0.f, 0.f};   // KIND 1: the task's bias rows g, g+1, g+8, g+9 (TT.s), loaded at its start
#if HSK_TAILREG
  uint4 hta[3], htr[2];                   // KIND 2: the task's error-norm A fragments and row constants
#endif
  auto setx = [&]() {
    int row = (int)((cq.rows8 >> (8 * g)) & 0xff);
    cb = (cq.bm >> g) & 1;
    cx = smaddr(cq.x + row * xst) + t * (KIND >= 2 ? 8 : 32);
    if (KIND == 1) {
      const float* bb = reinterpret_cast<const float*>(cq.s);
      bpre[0] = bb[g]; bpre[1] = bb[g + 1]; bpre[2] = bb[g + 8]; bpre[3] = bb[g + 9];
    }
#if HSK_TAILREG
    if (KIND == 2) {
      // the tile's 4 extra records (error-norm tiles, row constants) by plain loads at the task's start; the ring
      // streams only its 90 int8 chunks (10 whole stages)
      const uint4* tb = cq.w + 90 * 32;
#pragma unroll
      for (int kc = 0; kc < 3; kc++) hta[kc] = __ldcg(tb + kc * 32 + lane);
      htr[0] = __ldcg(tb + 3 * 32 + g * 2); htr[1] = __ldcg(tb + 3 * 32 + (g + 8) * 2);
    }
#endif
  };
  setx();
  pre(cq.slot);
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  float ua[4] = {0.f, 0.f, 0.f, 0.f};     // segmented tasks (slot bit 30): sum of the per-stage partials, in order
  float hx[16];                           // KIND 2/3: error-norm products (4), row constants (2 x 4), KIND 3: lo sums (4)
  float c3[4] = {0.f, 0.f, 0.f, 0.f};     // KIND 3: the lo-digit accumulators
  while (ct < nt) {
    const u32 sidx = rg.pos % RS;
    const int n = min(RI, cq.n - ci);
    mbar_wait(rg.bars + sidx * 8, (rg.pos / RS) & 1);
    const u32 st = rg.data + sidx * (RI * RING_ITEM);
    const int kb = cq.k0 + ci;
    if constexpr (KIND == 2 || KIND == 3) {
      if (n == RI) {
        if constexpr (KIND == 2) {
#pragma unroll
          for (int i = 0; i < RI; i++) item<KIND>(c, st, cx, kb, i, lane, g, cb);
        } else {
#pragma unroll
          for (int i = 0; i < RI; i++) {
            const uint4 q = lds128(st + i * 512 + lane * 16);
            u32 h0, h1, l0, l1;
            asm("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(h0), "=r"(h1) : "r"(cx + (kb + i) * 32));
            asm("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(l0), "=r"(l1) : "r"(cx + 8 * HSK_XST + (kb + i) * 32));
            mma_s8(c, q.x, q.y, q.z, q.w, h0, h1);
            mma_s8(c3, q.x, q.y, q.z, q.w, l0, l1);
          }
        }
      } else {
        // the task's tail: 3 BF16 tiles of error-block norms x the column block norms (B fragments at TT.s), then
        // the constants of rows g and g + 8
        const u32 xn = smaddr(cq.s) + lane * 8;
        hx[0] = hx[1] = hx[2] = hx[3] = 0.f;
#pragma unroll
        for (int kc = 0; kc < 3; kc++) {
          const uint4 a = lds128(st + kc * 512 + lane * 16);
          u32 b0, b1;
          asm("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(xn + kc * 256));
          mma16816(hx, a.x, a.y, a.z, a.w, b0, b1);
        }
        const uint4 r0 = lds128(st + 3 * 512 + g * 32), r1 = lds128(st + 3 * 512 + (g + 8) * 32);
        hx[4] = __uint_as_float(r0.x); hx[5] = __uint_as_float(r0.y); hx[6] = __uint_as_float(r0.z); hx[7] = __uint_as_float(r0.w);
        hx[8] = __uint_as_float(r1.x); hx[9] = __uint_as_float(r1.y); hx[10] = __uint_as_float(r1.z); hx[11] = __uint_as_float(r1.w);
      }
    } else {
#ifndef ACC1
    if (KIND == 1) {
      // fp4: two accumulators (even / odd chunks of the stage) so that two mma chains are in flight; added at the
      // stage end (the same in both kernel instances, so the bits stay batch-independent)
#ifdef ACC3
      float c2[4] = {0.f, 0.f, 0.f, 0.f}, c3[4] = {0.f, 0.f, 0.f, 0.f};
      if (n == RI) {
#pragma unroll
        for (int i = 0; i < RI; i++) item<KIND>((i % 3 == 0) ? c : ((i % 3 == 1) ? c2 : c3), st, cx, kb, i, lane, g, cb);
      } else {
#pragma unroll 1
        for (int i = 0; i < n; i++) item<KIND>((i % 3 == 0) ? c : ((i % 3 == 1) ? c2 : c3), st, cx, kb, i, lane, g, cb);
      }
#pragma unroll
      for (int j = 0; j < 4; j++) c[j] = (c[j] + c2[j]) + c3[j];
#else
      float c2[4] = {0.f, 0.f, 0.f, 0.f};
      if (n == RI) {
#pragma unroll
        for (int i = 0; i < RI; i++) item<KIND>((i & 1) ? c2 : c, st, cx, kb, i, lane, g, cb);
      } else {
#pragma unroll 1
        for (int i = 0; i < n; i += 2) {
          item<KIND>(c, st, cx, kb, i, lane, g, cb);
          if (i + 1 < n) item<KIND>(c2, st, cx, kb, i + 1, lane, g, cb);
        }
      }
#pragma unroll
      for (int j = 0; j < 4; j++) c[j] += c2[j];
#endif
    } else
#endif
#ifndef NO_ACC2_BF16
    if (KIND == 0) {
      float c2[4] = {0.f, 0.f, 0.f, 0.f};
      if (n == RI) {
#pragma unroll
        for (int i = 0; i < RI; i++) item<KIND>((i & 1) ? c2 : c, st, cx, kb, i, lane, g, cb);
      } else {
#pragma unroll 1
        for (int i = 0; i < n; i += 2) {
          item<KIND>(c, st, cx, kb, i, lane, g, cb);
          if (i + 1 < n) item<KIND>(c2, st, cx, kb, i + 1, lane, g, cb);
        }
      }
#pragma unroll
      for (int j = 0; j < 4; j++) c[j] += c2[j];
    } else
#endif
    if (n == RI) {
#pragma unroll
      for (int i = 0; i < RI; i++) item<KIND>(c, st, cx, kb, i, lane, g, cb);
    } else {
#pragma unroll 1
      for (int i = 0; i < n; i++) item<KIND>(c, st, cx, kb, i, lane, g, cb);
    }
    }
    __syncwarp();
    rg.pos++;
    if (ps.pt < nt) {
      if (lane == 0) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      issue_stage<KIND>(tt, ps, rg, lane);
    }
    ci += n;
    if (cq.slot >> 30) {
#pragma unroll
      for (int j = 0; j < 4; j++) { ua[j] += c[j]; c[j] = 0.f; }
    }
    if (ci == cq.n) {
      if (cq.slot >> 30) {
        if constexpr (KIND == 1) fin(cq.slot & 0xffff, ua, bpre); else if constexpr (KIND >= 2) fin(cq.slot & 0xffff, ua, hx); else fin(cq.slot & 0xffff, ua);
#pragma unroll
        for (int j = 0; j < 4; j++) ua[j] = 0.f;
      } else {
        if constexpr (KIND == 1) fin(cq.slot, c, bpre);
        else if constexpr (KIND == 2) {
#if HSK_TAILREG
          const u32 xn = smaddr(cq.s) + lane * 8;
          hx[0] = hx[1] = hx[2] = hx[3] = 0.f;
#pragma unroll
          for (int kc = 0; kc < 3; kc++) {
            u32 b0, b1;
            asm("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(xn + kc * 256));
            mma16816(hx, hta[kc].x, hta[kc].y, hta[kc].z, hta[kc].w, b0, b1);
          }
          hx[4] = __uint_as_float(htr[0].x); hx[5] = __uint_as_float(htr[0].y); hx[6] = __uint_as_float(htr[0].z); hx[7] = __uint_as_float(htr[0].w);
          hx[8] = __uint_as_float(htr[1].x); hx[9] = __uint_as_float(htr[1].y); hx[10] = __uint_as_float(htr[1].z); hx[11] = __uint_as_float(htr[1].w);
#endif
          fin(cq.slot, c, hx);
        }
        else if constexpr (KIND == 3) { hx[12] = c3[0]; hx[13] = c3[1]; hx[14] = c3[2]; hx[15] = c3[3]; fin(cq.slot, c, hx); c3[0] = c3[1] = c3[2] = c3[3] = 0.f; }
        else fin(cq.slot, c);
      }
      c[0] = c[1] = c[2] = c[3] = 0.f;
      ci = 0;
      if (++ct < nt) { cq = tt[ct]; setx(); pre(cq.slot); }
    }
  }
  if (lane == 0) psm = ps;
  __syncwarp();
}

template <int KIND, class Desc, class Fin>
__device__ __forceinline__ void run_tasks(int ntask, TT* table, int xst, Ring& rg, PState& ps, Desc desc, Fin fin) {
  prefetch_tasks<KIND>(ntask, table, rg, ps, desc);
  consume_tasks<KIND>(table, xst, rg, ps, fin);
}

// element (row i, column b) of a task's 16 x 8 result in red slot
__device__ __forceinline__ float red_at(const float* red, int slot, int i, int b) {
  return red[slot * 128 + ((i & 7) * 4 + (b >> 1)) * 4 + (i >> 3) * 2 + (b & 1)];
}

// B operands in shared memory are kept in fragment order: inside each 64-column chunk, lane t's eight words
// (j = 0..3: columns 16j+2t, +1 and 16j+2t+8, +9) are contiguous, so a chunk is two 16-byte loads per lane.
// element offset (bf16 units) of column k inside a row
__device__ __forceinline__ int xfo(int k) {
  const int c = k >> 6, r = k & 63, j = r >> 4, rr = r & 15;
  return c * 64 + (((rr & 7) >> 1) * 8 + j * 2 + (rr >> 3)) * 2 + (rr & 1);
}
// write 4 consecutive columns k..k+3 (k % 4 == 0) given as two bf16x2 words
__device__ __forceinline__ void xfo_store4(u16* row, int k, u32 w01, u32 w23) {
  *reinterpret_cast<u32*>(row + xfo(k)) = w01;
  *reinterpret_cast<u32*>(row + xfo(k + 2)) = w23;
}
// write 8 consecutive columns k..k+7 (k % 8 == 0): words land at t = 0..3 for the same (j, half)
__device__ __forceinline__ void xfo_store8(u16* row, int k, uint4 v) {
  *reinterpret_cast<u32*>(row + xfo(k)) = v.x; *reinterpret_cast<u32*>(row + xfo(k + 2)) = v.y;
  *reinterpret_cast<u32*>(row + xfo(k + 4)) = v.z; *reinterpret_cast<u32*>(row + xfo(k + 6)) = v.w;
}

// ------------------------------------------------------------------------------------------------ norms
// xs[b][k] = bf16(w[k] * (x[b][k] * rsqrt(mean(x^2) + eps))), x read with ld.cg (bf16 [B][HID] at src, or embed rows)
__device__ __forceinline__ uint2 ldrow(const u16* r, int k, bool sm) {
  uint2 v;
  if (sm) asm("ld.shared.v2.u32 {%0,%1},[%2];" : "=r"(v.x), "=r"(v.y) : "r"((u32)__cvta_generic_to_shared(r + k)));
  else v = __ldcg(reinterpret_cast<const uint2*>(r + k));
  return v;
}
#define SPLIT_K0 (NB * 16)    // OSPLIT: the o_proj rows of the split rowtiles (2432 .. 2879)
__device__ void norm_to_smem(const bf16* const* rows, int B, const u16* wsm, u16* xs, float* scratch, bool sm = false, const u16* patch = nullptr) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // sum of squares: warp w handles column b = w (w < B); lanes stride over 2880 in 8-byte pieces
  if (warp < B) {
    const u16* r = reinterpret_cast<const u16*>(rows[warp]);
    float s = 0.f;
    // loads in two batches (12 + 11 per lane) so that they are in flight together; same order of the sums
#pragma unroll
    for (int k0 = lane * 4; k0 < HID; k0 += 128 * 12) {
      uint2 v[12];
#pragma unroll
      for (int u = 0; u < 12; u++) if (k0 + 128 * u < HID) {
        const int k = k0 + 128 * u;
        v[u] = (patch && k >= SPLIT_K0) ? *reinterpret_cast<const uint2*>(patch + warp * (HID - SPLIT_K0) + k - SPLIT_K0) : ldrow(r, k, sm);
      }
#pragma unroll
      for (int u = 0; u < 12; u++) if (k0 + 128 * u < HID) {
        float a = __uint_as_float(v[u].x << 16), b = __uint_as_float(v[u].x & 0xffff0000u);
        float c = __uint_as_float(v[u].y << 16), d = __uint_as_float(v[u].y & 0xffff0000u);
        s += a * a + b * b + c * c + d * d;
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) scratch[warp] = rsqrtf(s / (float)HID + 1e-5f);
  }
  __syncthreads();
#ifdef NORM_BATCH   // spills in the flat instance (+140 us at B=1): not used
  // all loads of a thread (at most 6 items for B <= 2) before its stores
  constexpr int NJ = (FLAT_MAXB * (HID / 4) + NT - 1) / NT;
  uint2 vv[NJ], ww[NJ];
#pragma unroll
  for (int j = 0; j < NJ; j++) {
    const int i = tid + NT * j;
    if (i < B * (HID / 4)) {
      const int b = i / (HID / 4), k = (i % (HID / 4)) * 4;
      vv[j] = ldrow(reinterpret_cast<const u16*>(rows[b]), k, sm);
      ww[j] = *reinterpret_cast<const uint2*>(wsm + k);
    }
  }
#pragma unroll
  for (int j = 0; j < NJ; j++) {
    const int i = tid + NT * j;
    if (i >= B * (HID / 4)) break;
    const int b = i / (HID / 4), k = (i % (HID / 4)) * 4;
    const uint2 v = vv[j], wb = ww[j];
    const float ri = scratch[b];
#else
#pragma unroll 4
  for (int i = tid; i < B * (HID / 4); i += NT) {
    int b = i / (HID / 4), k = (i % (HID / 4)) * 4;
    const u16* r = reinterpret_cast<const u16*>(rows[b]);
    uint2 v = (patch && k >= SPLIT_K0) ? *reinterpret_cast<const uint2*>(patch + b * (HID - SPLIT_K0) + k - SPLIT_K0) : ldrow(r, k, sm);
    float ri = scratch[b];
    const uint2 wb = *reinterpret_cast<const uint2*>(wsm + k);
#endif
    const float4 wv = make_float4(__uint_as_float(wb.x << 16), __uint_as_float(wb.x & 0xffff0000u),
                                  __uint_as_float(wb.y << 16), __uint_as_float(wb.y & 0xffff0000u));
    u16 o0 = f2u16(wv.x * (__uint_as_float(v.x << 16) * ri));
    u16 o1 = f2u16(wv.y * (__uint_as_float(v.x & 0xffff0000u) * ri));
    u16 o2 = f2u16(wv.z * (__uint_as_float(v.y << 16) * ri));
    u16 o3 = f2u16(wv.w * (__uint_as_float(v.y & 0xffff0000u) * ri));
    xfo_store4(xs + b * XSH, k, o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
  }
  __syncthreads();
}

// warp b does the whole norm of column b from registers: one trip to L2 instead of two (same arithmetic and order)
#ifndef NORM_C
#define NORM_C 0   // 1: rolled-loop form: same bits, B=8 +270 us/step (per-iteration smem latency)
#endif
// compact form of norm_warp (same sums in the same order, same rounding): rows staged into xs with batched loads,
// then normed in place by rolled loops (one-shot code: small instruction footprint)
__device__ void norm_warp_c(const bf16* const* rows, int B, const u16* wsm, u16* xs) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int i0 = 0; i0 < B * (HID / 8); i0 += 8 * NT) {
    uint4 v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) { const int i = i0 + tid + NT * j; if (i < B * (HID / 8)) v[j] = __ldcg(reinterpret_cast<const uint4*>(rows[i / (HID / 8)]) + i % (HID / 8)); }
#pragma unroll
    for (int j = 0; j < 8; j++) { const int i = i0 + tid + NT * j; if (i < B * (HID / 8)) xfo_store8(xs + (i / (HID / 8)) * XSH, (i % (HID / 8)) * 8, v[j]); }
  }
  __syncthreads();
  if (warp < B) {
    u16* r = xs + warp * XSH;
    float s = 0.f;
#pragma unroll 1
    for (int k = lane * 4; k < HID; k += 128) {
      const u32 x01 = *reinterpret_cast<const u32*>(r + xfo(k)), x23 = *reinterpret_cast<const u32*>(r + xfo(k + 2));
      float a = __uint_as_float(x01 << 16), b = __uint_as_float(x01 & 0xffff0000u);
      float c = __uint_as_float(x23 << 16), d = __uint_as_float(x23 & 0xffff0000u);
      s += a * a + b * b + c * c + d * d;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float ri = rsqrtf(s / (float)HID + 1e-5f);
#pragma unroll 1
    for (int k = lane * 4; k < HID; k += 128) {
      const u32 x01 = *reinterpret_cast<const u32*>(r + xfo(k)), x23 = *reinterpret_cast<const u32*>(r + xfo(k + 2));
      const uint2 wb = *reinterpret_cast<const uint2*>(wsm + k);
      u16 o0 = f2u16(__uint_as_float(wb.x << 16) * (__uint_as_float(x01 << 16) * ri));
      u16 o1 = f2u16(__uint_as_float(wb.x & 0xffff0000u) * (__uint_as_float(x01 & 0xffff0000u) * ri));
      u16 o2 = f2u16(__uint_as_float(wb.y << 16) * (__uint_as_float(x23 << 16) * ri));
      u16 o3 = f2u16(__uint_as_float(wb.y & 0xffff0000u) * (__uint_as_float(x23 & 0xffff0000u) * ri));
      xfo_store4(r, k, o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
    }
  }
  __syncthreads();
}
// klo, khi, inv: only the columns k with (klo <= k < khi) != inv are stored (the sums always cover the whole row)
__device__ void norm_warp(const bf16* const* rows, int B, const u16* wsm, u16* xs, const u16* patch = nullptr, int klo = 0, int khi = HID, bool inv = false) {
#if NORM_C
  norm_warp_c(rows, B, wsm, xs); return;
#endif
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr int NIT = (HID / 4 + 31) / 32;   // 23
  if (warp < B) {
    const u16* r = reinterpret_cast<const u16*>(rows[warp]);
    uint2 v[NIT], wb[NIT];
#pragma unroll
    for (int it = 0; it < NIT; it++) {
      const int k = lane * 4 + 128 * it;
      if (k < HID) v[it] = (patch && k >= SPLIT_K0) ? *reinterpret_cast<const uint2*>(patch + warp * (HID - SPLIT_K0) + k - SPLIT_K0)
                                                    : __ldcg(reinterpret_cast<const uint2*>(r + k));
    }
#ifdef NORM_WREG   // weight words in registers first: faster alone (2.13 -> 1.74 us at B=8) but +25 us/step at B=8 in the kernel
    // the weight words before any store to xs (generic smem stores could alias them: loads after a store serialize)
#pragma unroll
    for (int it = 0; it < NIT; it++) { const int k = lane * 4 + 128 * it; if (k < HID) wb[it] = *reinterpret_cast<const uint2*>(wsm + k); }
#endif
    float s = 0.f;
#pragma unroll
    for (int it = 0; it < NIT; it++) {
      const int k = lane * 4 + 128 * it;
      if (k < HID) {
        float a = __uint_as_float(v[it].x << 16), b = __uint_as_float(v[it].x & 0xffff0000u);
        float c = __uint_as_float(v[it].y << 16), d = __uint_as_float(v[it].y & 0xffff0000u);
        s += a * a + b * b + c * c + d * d;
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float ri = rsqrtf(s / (float)HID + 1e-5f);
#if !NORM_PIPE1   // 1: weight words one iteration ahead (warm 2.14 -> 1.75 us at B=8; in the kernel: 0 at ~6 h, -23.5 us at B=8 after s0036)
#pragma unroll
    for (int it = 0; it < NIT; it++) {
      const int k = lane * 4 + 128 * it;
      if (k < HID && ((k >= klo && k < khi) != inv)) {
        wb[it] = *reinterpret_cast<const uint2*>(wsm + k);
        u16 o0 = f2u16(__uint_as_float(wb[it].x << 16) * (__uint_as_float(v[it].x << 16) * ri));
        u16 o1 = f2u16(__uint_as_float(wb[it].x & 0xffff0000u) * (__uint_as_float(v[it].x & 0xffff0000u) * ri));
        u16 o2 = f2u16(__uint_as_float(wb[it].y << 16) * (__uint_as_float(v[it].y << 16) * ri));
        u16 o3 = f2u16(__uint_as_float(wb[it].y & 0xffff0000u) * (__uint_as_float(v[it].y & 0xffff0000u) * ri));
        xfo_store4(xs + warp * XSH, k, o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
      }
    }
#else
    // the weight words one iteration ahead: each load is issued before the previous iteration's stores (generic
    // smem stores could alias it, so it cannot be hoisted further by the compiler)
    uint2 wn = *reinterpret_cast<const uint2*>(wsm + lane * 4);
#pragma unroll
    for (int it = 0; it < NIT; it++) {
      const int k = lane * 4 + 128 * it;
      const uint2 w = wn;
      if (it + 1 < NIT && k + 128 < HID) wn = *reinterpret_cast<const uint2*>(wsm + k + 128);
      if (k < HID) {
        u16 o0 = f2u16(__uint_as_float(w.x << 16) * (__uint_as_float(v[it].x << 16) * ri));
        u16 o1 = f2u16(__uint_as_float(w.x & 0xffff0000u) * (__uint_as_float(v[it].x & 0xffff0000u) * ri));
        u16 o2 = f2u16(__uint_as_float(w.y << 16) * (__uint_as_float(v[it].y << 16) * ri));
        u16 o3 = f2u16(__uint_as_float(w.y & 0xffff0000u) * (__uint_as_float(v[it].y & 0xffff0000u) * ri));
        xfo_store4(xs + warp * XSH, k, o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
      }
    }
#endif
  }
  __syncthreads();
}

// norm_warp with the B input rows staged into xs by bulk copies first (natural order, row stride XSH), read back from
// shared memory, then normed in place into fragment order: the same sums in the same order (same bits)
__device__ void norm_warp_tma(const bf16* const* rows, int B, const u16* wsm, u16* xs, u32 bar, u32& bpos) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr int NIT = (HID / 4 + 31) / 32;   // 23
  if (tid == 0) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    mbar_expect(bar, B * HID * 2);
    for (int b = 0; b < B; b++) tma_load(smaddr(xs + b * XSH), rows[b], HID * 2, bar);
  }
  mbar_wait(bar, bpos & 1); bpos++;
  uint2 v[NIT];
  float ri = 0.f;
  if (warp < B) {
    const u32 r = smaddr(xs + warp * XSH);
#pragma unroll
    for (int it = 0; it < NIT; it++) {
      const int k = lane * 4 + 128 * it;
      if (k < HID) asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v[it].x), "=r"(v[it].y) : "r"(r + k * 2));
    }
    float s = 0.f;
#pragma unroll
    for (int it = 0; it < NIT; it++) {
      const int k = lane * 4 + 128 * it;
      if (k < HID) {
        float a = __uint_as_float(v[it].x << 16), b = __uint_as_float(v[it].x & 0xffff0000u);
        float c = __uint_as_float(v[it].y << 16), d = __uint_as_float(v[it].y & 0xffff0000u);
        s += a * a + b * b + c * c + d * d;
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    ri = rsqrtf(s / (float)HID + 1e-5f);
  }
  __syncwarp();   // the warp's own row is read entirely before it is overwritten (rows are per warp)
  if (warp < B) {
#pragma unroll
    for (int it = 0; it < NIT; it++) {
      const int k = lane * 4 + 128 * it;
      if (k < HID) {
        const uint2 wb = *reinterpret_cast<const uint2*>(wsm + k);
        u16 o0 = f2u16(__uint_as_float(wb.x << 16) * (__uint_as_float(v[it].x << 16) * ri));
        u16 o1 = f2u16(__uint_as_float(wb.x & 0xffff0000u) * (__uint_as_float(v[it].x & 0xffff0000u) * ri));
        u16 o2 = f2u16(__uint_as_float(wb.y << 16) * (__uint_as_float(v[it].y << 16) * ri));
        u16 o3 = f2u16(__uint_as_float(wb.y & 0xffff0000u) * (__uint_as_float(v[it].y & 0xffff0000u) * ri));
        xfo_store4(xs + warp * XSH, k, o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
      }
    }
  }
  __syncthreads();
}

// K/V cache of a (sequence, kv head) in mma fragment order: K per 8 keys [32 lanes][8 u32] (lane (g, t): key g,
// dims 16ks+2t,+1 and 16ks+2t+8,+9 for ks = 0..3), V per 16 keys [32 lanes][16 u32] (lane (g, t): dim 8nb+g, keys
// 2t,2t+1 and 2t+8,2t+9 for nb = 0..7); element offsets in bf16 units
__device__ __forceinline__ int kfo(int pos, int d) {
  const int ks = d >> 4, r = d & 15;
  return (((pos >> 3) * 32 + (pos & 7) * 4 + ((r & 7) >> 1)) * 8 + ks * 2 + (r >> 3)) * 2 + (r & 1);
}
__device__ __forceinline__ int vfo(int pos, int d) {
  const int kk = pos & 15;
  return (((pos >> 4) * 32 + (d & 7) * 4 + ((kk & 7) >> 1)) * 16 + (d >> 3) * 2 + (kk >> 3)) * 2 + (kk & 1);
}

// ------------------------------------------------------------------------------------------------ smem layout
// flat down path (B <= 2): the layer output from the down partials, for all rows, into shared memory (row-major
// bf16 [B][HID] at dst); the block also stores its share (every NB-th group of 4 columns) to x.
// [0, 65664)          xs: bf16 [8][4104] (hidden vectors use stride 2888)
// [65664, +40960)     red: float [80][128]
// attention reuses: sc float [8][2048] at 0 (64 KB), ps bf16 [8][2048+16] at 65664 (32.3 KB), ared aliases sc
// small scalars at SM_MISC
#define SM_XS 0
#ifndef XS_BYTES
#define XS_BYTES 65664
#endif
#define SM_RED XS_BYTES
#ifndef RED_SLOTS
#define RED_SLOTS 50
#endif
#define SM_MISC (SM_RED + RED_SLOTS * 512)
#define SM_TT (SM_MISC + 4096)
#define SM_BARS (SM_TT + SM_TT_BYTES)
#define SM_NWBAR (SM_BARS + NW * RS * 8)
#define SM_PS (SM_BARS + NW * RS * 8 + 64)
#define SM_RING (SM_PS + NW * 32)
#define SM_NWB (SM_RING + NW * RING_WARP)
#define SM_TOTAL (SM_NWB + HID * 2)

struct Misc {
  float rinv[MAXB];
  int U;                 // number of chosen experts (union over the batch)
  int sexp[32];          // slot -> expert id (ascending)
  float rw[MAXB][32];    // routing weight of (sequence, slot), 0 if not routed
  u32 smask[32];         // bit b set if sequence b routed to slot
  int tok[MAXB];
  int fr_s0, fr_s1, fr_np, fr_u0, fr_u1, fr_cur, fr_rs;
  int nrs[MAXB]; int rsl[MAXB][4];     // routed slots of each column, ascending
  int eep[32];          // per slot: how many layers have used it in this launch (target of ecnt = 360 x eep)
  u64 srows[32];
  float tmpf[128];
  int tmpi[128];
  int cs_role, cs_k, cs_x;   // CSPLIT: 1 owner of shared rowtile cs_x / 2 its partner (pair index cs_k), 0 neither
  u32 hqb;                   // HSK_RQ: queue entries taken by earlier steps of this launch
  u64 rsbar[RS_SLOTS];       // HSK_FASTRS: one mbarrier per stage slot of the pooled ring
  u32 rsuse[RS_SLOTS];       // HSK_FASTRS: completed uses of each slot's mbarrier in this launch (parity)
};

// XPF: one warp of each of the last XPF_NB blocks takes the predicted top-4 of every sequence (predictor partials
// from phase A), forms their union and pulls its share of those experts' weights into L2 (gate_up, and down for
// B <= XPF_DNB), at most XPF_CAP_MB in all. Changes no arithmetic: only where E/F's bulk copies find their bytes.
#ifndef XPF
#define XPF 0
#endif
#ifndef XPF_NB
#define XPF_NB 16
#endif
#ifndef XPF_DNB
#define XPF_DNB 2
#endif
#ifndef XPF_CAP_MB
#define XPF_CAP_MB 96
#endif
#ifndef XPF_K
#define XPF_K 4
#endif
#define XPF_PIECE 32768
__device__ __noinline__ void xpf_issue(const Params& p, const LayerP& Lp, int B, int j) {
  const int lane = threadIdx.x & 31;
  bool sel[4] = {false, false, false, false};
  float lg[MAXB][4];
#pragma unroll
  for (int b = 0; b < MAXB; b++)
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float vv = 0.f;
      if (b < B) {
#pragma unroll
        for (int q = 0; q < RQ; q++) vv += ldcgf(p.plog + (q * MAXB + b) * NE + lane + 32 * k);
      }
      lg[b][k] = vv;
    }
#pragma unroll
  for (int b = 0; b < MAXB; b++) {
    if (b >= B) break;
    u32 key[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const int ix = lane + 32 * k;
      const float vv = lg[b][k] + Lp.br[ix];
      const u32 hb = __float_as_uint(vv) >> 16;
      const u32 ord = (hb & 0x8000u) ? (~hb & 0xffffu) : (hb | 0x8000u);
      key[k] = (ord << 16) | (0xffffu - (u32)ix);
    }
#pragma unroll
    for (int r = 0; r < XPF_K; r++) {
      const u32 lm = max(max(key[0], key[1]), max(key[2], key[3]));
      const u32 best = __reduce_max_sync(0xffffffffu, lm);
#pragma unroll
      for (int k = 0; k < 4; k++) if (key[k] == best) { key[k] = 0; sel[k] = true; }
    }
  }
  u32 bal[4];
#pragma unroll
  for (int k = 0; k < 4; k++) bal[k] = __ballot_sync(0xffffffffu, sel[k]);   // bit l of bal[k]: expert l + 32k
  const int n = __popc(bal[0]) + __popc(bal[1]) + __popc(bal[2]) + __popc(bal[3]);
  const unsigned long long GU = (unsigned long long)RT_GU * KC * 544, DN = (unsigned long long)RT_DN * KC * 544;
  const bool dn = B <= XPF_DNB;
  const unsigned long long per = GU + (dn ? DN : 0ull);
  unsigned long long tot = (unsigned long long)n * per;
  const unsigned long long cap = (unsigned long long)XPF_CAP_MB << 20;
  if (tot > cap) tot = cap;
  const unsigned long long r0 = (tot * j / XPF_NB) & ~15ull, r1 = (j == XPF_NB - 1) ? tot : ((tot * (j + 1) / XPF_NB) & ~15ull);
#if XPF == 2
  // per-lane L2 line prefetches (no bulk-copy unit): lane l takes lines l, l + 32, ... of the block's range
  for (unsigned long long off = r0 + (unsigned long long)lane * 128; off < r1; off += 32ull * 128) {
    const int idx = (int)(off / per);
    const unsigned long long w = off % per;
    int e = 0, c = idx;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const int pc = __popc(bal[k]);
      if (c >= 0 && c < pc) { u32 m = bal[k]; for (int t = 0; t < c; t++) m &= m - 1; e = 32 * k + __ffs(m) - 1; }
      c -= pc;
    }
    const unsigned char* a = (w < GU) ? reinterpret_cast<const unsigned char*>(Lp.gu_c) + (unsigned long long)e * GU + w
                                      : reinterpret_cast<const unsigned char*>(Lp.dn_c) + (unsigned long long)e * DN + (w - GU);
    asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
  }
#else
  for (unsigned long long off = r0 + (unsigned long long)lane * XPF_PIECE; off < r1; off += 32ull * XPF_PIECE) {
    const int idx = (int)(off / per);
    const unsigned long long w = off % per;
    // idx-th set bit of bal (ascending expert order)
    int e = 0, c = idx;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      const int pc = __popc(bal[k]);
      if (c >= 0 && c < pc) { u32 m = bal[k]; for (int t = 0; t < c; t++) m &= m - 1; e = 32 * k + __ffs(m) - 1; }
      c -= pc;
    }
    const unsigned char* base; unsigned long long segend;
    if (w < GU) { base = reinterpret_cast<const unsigned char*>(Lp.gu_c) + (unsigned long long)e * GU + w; segend = GU; }
    else { base = reinterpret_cast<const unsigned char*>(Lp.dn_c) + (unsigned long long)e * DN + (w - GU); segend = per; }
    unsigned long long sz = XPF_PIECE;
    if (off + sz > r1) sz = r1 - off;
    if (w + sz > segend) sz = segend - w;
    if (sz >= 16) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(base), "r"((u32)(sz & ~15ull)) : "memory");
  }
#endif
}

#ifdef PREP_T
__device__ unsigned long long g_prep_t[64][6];
__device__ unsigned g_prep_n;
__device__ __forceinline__ unsigned long long mkprof_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
extern "C" void mk_prep_t(unsigned long long* out) { cudaMemcpyFromSymbol(out, g_prep_t, sizeof(g_prep_t)); }
#endif
struct HskOut { u32 pos, target; float bv, bv2; int bi, bi2; u32 pn; };
// study: the exact head's consume out of line (the wide instance), to measure what a call costs that instance
__device__ __noinline__ HskOut head_exact_ol(TT* ttab, Ring rg, PState* pst, int B, float* logits, int mode, u32 target) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  float bv = -INFINITY; int bi = 0x7fffffff;
  float bv2 = -INFINITY; int bi2 = 0x7fffffff;
  consume_tasks<0>(ttab, XSH, rg, pst[warp], [&](int rt, const float* c) {
    int row0 = rt * 16 + g, row1 = row0 + 8;
    int b0 = 2 * t, b1 = 2 * t + 1;
    if (mode == 0) {
      if (b0 < B) { logits[(size_t)b0 * VOCAB + row0] = c[0]; logits[(size_t)b0 * VOCAB + row1] = c[2]; }
      if (b1 < B) { logits[(size_t)b1 * VOCAB + row0] = c[1]; logits[(size_t)b1 * VOCAB + row1] = c[3]; }
    } else {
      if (c[0] > bv || (c[0] == bv && row0 < bi)) { bv = c[0]; bi = row0; }
      if (c[2] > bv || (c[2] == bv && row1 < bi)) { bv = c[2]; bi = row1; }
      if (c[1] > bv2 || (c[1] == bv2 && row0 < bi2)) { bv2 = c[1]; bi2 = row0; }
      if (c[3] > bv2 || (c[3] == bv2 && row1 < bi2)) { bv2 = c[3]; bi2 = row1; }
    }
  });
  return HskOut{rg.pos, target, bv, bv2, bi, bi2, 0u};
}
#if HSK
// wide head sketch prep (compact rolled loops: this code runs once per step, from a cold instruction cache).
// The B rows of p.x are staged in shared memory (stg, natural order); warp b then normes row b exactly as norm_warp
// does (same per-lane order of the squares, same shuffle tree, same rounding), and writes its int8 digits (hi row b,
// lo row 8 + b of dg), its 64-block norms (B fragments) and its statistics.
// (PREP_RESTRICT: the staged rows, digit rows, squares and statistics as non-aliasing pointers, so that the loads of
// unrolled iterations can be issued before the previous iterations' stores)
#if PREP_RESTRICT
#define PRR __restrict__
#else
#define PRR
#endif
__device__ __noinline__ void hsk_prep_w(const bf16* PRR p_x, const u16* PRR nwb, unsigned char* PRR dg, unsigned char* PRR hb8, float* PRR hst, int B, u16* PRR stg, float* PRR sqs0) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  constexpr int LO = 8;
#ifdef PREP_T
  unsigned long long T[6]; T[0] = mkprof_gt();
#endif
  for (int i0 = 0; i0 < B * (HID / 8); i0 += 8 * NT) {
    uint4 v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) { const int i = i0 + tid + NT * j; if (i < B * (HID / 8)) v[j] = __ldcg(reinterpret_cast<const uint4*>(p_x) + i); }
#pragma unroll
    for (int j = 0; j < 8; j++) { const int i = i0 + tid + NT * j; if (i < B * (HID / 8)) reinterpret_cast<uint4*>(stg)[i] = v[j]; }
  }
  __syncthreads();
#ifdef PREP_T
  T[1] = mkprof_gt();
#endif
  if (warp < B) {
    const int b = warp;
    u16* const r = stg + b * HID;
    float s = 0.f;
#pragma unroll kPrepUnroll
    for (int k = lane * 4; k < HID; k += 128) {
      const uint2 v = *reinterpret_cast<const uint2*>(r + k);
      float a = __uint_as_float(v.x << 16), bb = __uint_as_float(v.x & 0xffff0000u);
      float c = __uint_as_float(v.y << 16), d = __uint_as_float(v.y & 0xffff0000u);
      s += a * a + bb * bb + c * c + d * d;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float ri = rsqrtf(s / (float)HID + 1e-5f);
#ifdef PREP_T
    T[2] = mkprof_gt();
#endif
    // normed values (as norm_warp: bf16(w * (x * ri))) back into stg, and their maximum
    float m = 0.f;
#pragma unroll kPrepUnroll
    for (int k = lane * 4; k < HID; k += 128) {
      const uint2 v = *reinterpret_cast<const uint2*>(r + k);
      const uint2 wb = *reinterpret_cast<const uint2*>(nwb + k);
      const u16 o0 = f2u16(__uint_as_float(wb.x << 16) * (__uint_as_float(v.x << 16) * ri));
      const u16 o1 = f2u16(__uint_as_float(wb.x & 0xffff0000u) * (__uint_as_float(v.x & 0xffff0000u) * ri));
      const u16 o2 = f2u16(__uint_as_float(wb.y << 16) * (__uint_as_float(v.y << 16) * ri));
      const u16 o3 = f2u16(__uint_as_float(wb.y & 0xffff0000u) * (__uint_as_float(v.y & 0xffff0000u) * ri));
      m = fmaxf(m, fmaxf(fmaxf(fabsf(u16f(o0)), fabsf(u16f(o1))), fmaxf(fabsf(u16f(o2)), fabsf(u16f(o3)))));
      *reinterpret_cast<uint2*>(r + k) = make_uint2(o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
#ifdef PREP_T
    T[3] = mkprof_gt();
#endif
    int e = 0;
    if (m > 0.f) frexpf(m, &e);                 // m < 2^e
    const int kx = 13 - e;
    const float sc = ldexpf(1.f, kx), isc = ldexpf(1.f, -kx);
    float rr = 0.f;
    float* const sqs = sqs0 + warp * 32 * 23;   // per (it, lane) squares
#pragma unroll kPrepUnroll
    for (int it = 0; it < 23; it++) {
      const int k = lane * 4 + 128 * it;
      float sq = 0.f;
      if (k < HID) {
        const uint2 v = *reinterpret_cast<const uint2*>(r + k);
        const float o4[4] = {__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                             __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u)};
        u32 hw = 0, lw = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          // X = rint(v 2^k) without conversion instructions: |v 2^k| < 2^13, so v 2^k + 1.5 2^23 rounds to an
          // integer (nearest even) and its low bits are X; hi = round-half-up(X / 128), lo = X - 128 hi in [-64, 63]
          const float y = fmaf(o4[j], sc, 12582912.f);
          const int Xi = __float_as_int(y) - 0x4B400000;
          const float X = y - 12582912.f;
          const int hi = (Xi + 64) >> 7, lo = Xi - (hi << 7);
          hw |= ((u32)hi & 0xffu) << (8 * j); lw |= ((u32)lo & 0xffu) << (8 * j);
          const float rj = fmaf(-X, isc, o4[j]);
          rr = fmaf(rj, rj, rr); sq = fmaf(o4[j], o4[j], sq);
        }
        const int c = k >> 5, kk = k & 31, off = c * 32 + ((kk & 15) >> 2) * 8 + (kk >> 4) * 4;
        *reinterpret_cast<u32*>(dg + b * HSK_XST + off) = hw;
        *reinterpret_cast<u32*>(dg + (LO + b) * HSK_XST + off) = lw;
      }
      sqs[it * 32 + lane] = sq;
    }
    __syncwarp();
#ifdef PREP_T
    T[4] = mkprof_gt();
#endif
    // 64-block j = elements of lanes 16 (j & 1) .. 16 (j & 1) + 15 at it = j / 2
    float xx = 0.f;
#pragma unroll kPrepUnroll
    for (int blk = lane; blk < 45; blk += 32) {
      const float* q = sqs + (blk >> 1) * 32 + 16 * (blk & 1);
      float sq = 0.f;
#pragma unroll
      for (int l = 0; l < 16; l++) sq += q[l];
      xx += sq;
      // block norm rounded up to BF16 (sqrt.approx and the fp32 sums are within the 2^-12 margin)
      float sr; asm("sqrt.approx.f32 %0, %1;" : "=f"(sr) : "f"(sq));
      u32 u = __float_as_uint(sr * (1.f + 1.f / 4096.f));
      if (u & 0xffffu) u = (u + 0x10000u) & 0xffff0000u;
      const int kc = blk >> 4, kb = blk & 15, tt = (kb & 7) >> 1, half = kb >> 3, pos = kb & 1;
      reinterpret_cast<u16*>(hb8 + HSK_XNF)[((kc * 32 + b * 4 + tt) * 2 + half) * 2 + pos] = (u16)(u >> 16);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) { rr += __shfl_xor_sync(0xffffffffu, rr, o); xx += __shfl_xor_sync(0xffffffffu, xx, o); }
    if (lane == 0) {
      hst[b] = isc;
      hst[8 + b] = sqrtf(rr) * (1.f + 1.f / 1024.f) * (1.f + 1.f / 256.f);
      hst[16 + b] = sqrtf(xx) * (1.f + 1.f / 1024.f) * (1.f / 2048.f);
    }
  }
#ifdef PREP_T
  if (blockIdx.x == 0 && tid == 0) { T[5] = mkprof_gt(); for (int k = 0; k < 6; k++) g_prep_t[g_prep_n % 64][k] = T[k]; g_prep_n++; }
#endif
}

#ifdef MKPROF
#define HSK_PA , MkProf prof, u32 pn
#define HSK_PV , prof, pn
#else
#define HSK_PA , int pn
#define HSK_PV , 0
#endif
// certified int8 sketch of the head, then the exact head on the candidate rowtiles only (greedy paths).
// x = BF16 head input, per column b: X = rint(x 2^k) (|X| <= 2^13), X = 128 hi + lo, r = x - 2^-k X.
// approx_i = s_i 2^-k (q_i . X); |w_i . x - approx_i| <= sum_blk |e_i,blk| |x_blk| + |s_i q_i| |r| and the exact
// head's own fp32 error is below 2^-11 |w_i| |x|; rows whose upper bound is below the best lower bound cannot be the
// argmax. The candidates are recomputed with the exact head's tasks (same bits). Kept out of line (ring and barrier
// state by value). W = false (B <= 2): digits of the normed head input xs in red (hi rows 0..3, lo rows 4..7, one
// B tile); W = true (B <= 8): the norm of p.x is done here (norm_warp's arithmetic), digits in the xs area (hi rows
// 0..7, lo rows 8..15, two B tiles), and blocks with candidates rebuild xs with norm_warp before the rescoring.
// inlined in the B >= 3 instance (no spills, -25 us at B=8), called out of line in the B <= 2 one (-9 us at B=1)
#if HSK_WARM
// HSK_WARM: the pooled candidate rescoring (FASTRS) as one out-of-line function, so that a short dummy call before
// the post-sketch barrier wait brings exactly these instructions into the SM's instruction cache (the head's code
// runs once per step: cold). nst stages per tile (KS_H / RI for real; fewer for the warm-up, results unused).
struct RsBest { float bv, bv2; int bi, bi2; };
__device__ __noinline__ RsBest rescore_tiles(const uint4* p_head, const int* hcand, int nc, int nact, int nst, Misc* ms,
                                            u32 ring0, const u16* xs, RsBest r) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  if (warp >= nact) return r;
  const int S = min(KS_H / RI, RS_SLOTS / nact), sb = warp * S, ni = min(S, nst);
  const u32 xb = smaddr(xs + g * XSH) + t * 32;
  for (int ci = warp; ci < nc; ci += nact) {
    const int rt = hcand[ci];
    __syncwarp();
    auto issue = [&](int st) {
      const int sl = sb + st % S;
      const u32 bb = smaddr(&ms->rsbar[sl]);
      mbar_expect(bb, RI * 512);
      tma_load(ring0 + sl * RI * 512, p_head + ((size_t)rt * KS_H + st * RI) * 32, RI * 512, bb);
    };
    if (lane == 0) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      for (int st = 0; st < ni; st++) issue(st);
    }
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int st = 0; st < nst; st++) {
      const int sl = sb + st % S;
      mbar_wait(smaddr(&ms->rsbar[sl]), ms->rsuse[sl] & 1);
      const u32 stg = ring0 + sl * RI * 512;
      float c2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < RI; i++) item<0>((i & 1) ? c2 : c, stg, xb, st * RI, i, lane, g, true);
#pragma unroll
      for (int j = 0; j < 4; j++) c[j] += c2[j];
      __syncwarp();
      if (lane == 0) {
        ms->rsuse[sl]++;
        if (st + S < nst) { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); issue(st + S); }
      }
      __syncwarp();
    }
    if (nst == KS_H / RI) {
      const int row0 = rt * 16 + g, row1 = row0 + 8;
      if (c[0] > r.bv || (c[0] == r.bv && row0 < r.bi)) { r.bv = c[0]; r.bi = row0; }
      if (c[2] > r.bv || (c[2] == r.bv && row1 < r.bi)) { r.bv = c[2]; r.bi = row1; }
      if (c[1] > r.bv2 || (c[1] == r.bv2 && row0 < r.bi2)) { r.bv2 = c[1]; r.bi2 = row0; }
      if (c[3] > r.bv2 || (c[3] == r.bv2 && row1 < r.bi2)) { r.bv2 = c[3]; r.bi2 = row1; }
    }
  }
  return r;
}
#endif
template <bool W>
__device__ __forceinline__ HskOut hsk_head(const uint4* p_head, const uint4* p_hsk, float* p_hsl, int* p_hdbg, const bf16* p_x, const u16* nwb,
                                       Misc* ms, u16* xs, float* red, TT* ttab, Ring rg, PState* pst,
                                       int B, u32* bar, u32 bar_target, int r0, int r1 HSK_PA) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3, bid = blockIdx.x;
  float bv = -INFINITY; int bi = 0x7fffffff;
  float bv2 = -INFINITY; int bi2 = 0x7fffffff;
  unsigned char* hb8 = reinterpret_cast<unsigned char*>(red);
  unsigned char* dg = W ? reinterpret_cast<unsigned char*>(xs) : hb8;   // digit rows
  constexpr int LO = W ? 8 : 4;                                          // row of the lo digits of column 0
  float* hst = reinterpret_cast<float*>(hb8 + HSK_ST);
  int* hsi = reinterpret_cast<int*>(hb8 + HSK_ST);
  int2* hlist = reinterpret_cast<int2*>(W ? hb8 : hb8 + HSK_LIST);
  constexpr int CAP = W ? HSK_CAPW : HSK_CAP;
  int* hcand = reinterpret_cast<int*>(hb8 + HSK_CAND);
  for (int i = tid; i < 768 / 4; i += NT) reinterpret_cast<u32*>(hb8 + HSK_XNF)[i] = 0u;
  if (tid < 8) hsi[32 + tid] = (int)0x80800000;   // key of -inf
  if (tid == 0) { for (int i = 40; i < 49; i++) hsi[i] = 0; }
  __syncthreads();
  MARK(32);
  if constexpr (W) {
#if HSK_L2PFW
    // the first sketch task of every warp into L2 while the prep runs (HBM is idle then; ~58 MB in all)
    if (lane == 0 && warp < r1 - r0) l2_prefetch(p_hsk + (size_t)(r0 + warp) * HSK_ITEMS * 32, HSK_ITEMS * 512);
#endif
    // stage area: the ring (drained: the wide sketch's stream starts after the prep)
    {
      u16* const stg = reinterpret_cast<u16*>(const_cast<unsigned char*>(rg.gdata) - warp * RING_WARP);
      hsk_prep_w(p_x, nwb, dg, hb8, hst, B, stg, reinterpret_cast<float*>(stg + B * HID));
    }
    __syncthreads();
#if HSK_XG
    // block b < B publishes normed row b (norm_warp's bits) in fragment order for the candidate blocks' rescoring
    if (bid < B) {
      const u16* stg = reinterpret_cast<const u16*>(rg.gdata - warp * RING_WARP) + bid * HID;
      u16* dst = reinterpret_cast<u16*>(p_hsl + NB * MAXB) + bid * HID;
      for (int k = tid * 4; k < HID; k += NT * 4) {
        const uint2 v = *reinterpret_cast<const uint2*>(stg + k);
        xfo_store4(dst, k, v.x, v.y);
      }
    }
#endif
  } else {
    // column maxima (all warps), then per 64-block: digits, |r|^2, |x|^2 and the block norm (warp w: blocks w, w+8..)
    {
      float m0 = 0.f, m1 = 0.f;
      for (int k = tid; k < HID; k += NT) {
        m0 = fmaxf(m0, fabsf(u16f(xs[xfo(k)])));
        if (B > 1) m1 = fmaxf(m1, fabsf(u16f(xs[XSH + xfo(k)])));
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) { m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o)); }
      if (lane == 0) { ms->tmpf[warp] = m0; ms->tmpf[8 + warp] = m1; }
    }
    __syncthreads();
    for (int b = 0; b < B; b++) {
      float m = 0.f;
#pragma unroll
      for (int w = 0; w < NW; w++) m = fmaxf(m, ms->tmpf[8 * b + w]);
      int e = 0;
      if (m > 0.f) frexpf(m, &e);                 // m < 2^e
      const int kx = 13 - e;
      const float sc = ldexpf(1.f, kx), isc = ldexpf(1.f, -kx);
      const u16* xr = xs + b * XSH;
      float rr = 0.f, xx = 0.f;
      for (int blk = warp; blk < 45; blk += NW) {
        float sq = 0.f;
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int k = blk * 64 + h * 32 + lane;
          const float v = u16f(xr[xfo(k)]);
          const float y = fmaf(v, sc, 12582912.f);          // as in hsk_prep_w: X = rint(v 2^k) by the magic add
          const int Xi = __float_as_int(y) - 0x4B400000;
          const float X = y - 12582912.f;
          const int hi = (Xi + 64) >> 7, lo = Xi - (hi << 7);
          const int c = k >> 5, kk = k & 31, off = c * 32 + ((kk & 15) >> 2) * 8 + (kk >> 4) * 4 + (kk & 3);
          dg[b * HSK_XST + off] = (unsigned char)hi;
          dg[(LO + b) * HSK_XST + off] = (unsigned char)lo;
          const float r = fmaf(-X, isc, v);
          rr = fmaf(r, r, rr); sq = fmaf(v, v, sq);
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        xx += sq;
        if (lane == 0) {
          // block norm rounded up to BF16 (the fp32 sum of 64 squares is within 2^-18 relative)
          u32 u = __float_as_uint(sqrtf(sq) * (1.f + 1.f / 4096.f));
          if (u & 0xffffu) u = (u + 0x10000u) & 0xffff0000u;
          const int kc = blk >> 4, kb = blk & 15, tt = (kb & 7) >> 1, half = kb >> 3, pos = kb & 1;
          reinterpret_cast<u16*>(hb8 + HSK_XNF)[((kc * 32 + b * 4 + tt) * 2 + half) * 2 + pos] = (u16)(u >> 16);
        }
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) rr += __shfl_xor_sync(0xffffffffu, rr, o);
      if (lane == 0) { ms->tmpf[16 + 16 * b + warp] = rr; ms->tmpf[24 + 16 * b + warp] = xx; }
      if (tid == 0) hst[b] = isc;
    }
    __syncthreads();
    if (tid < B) {
      float rr = 0.f, xx = 0.f;
      for (int w = 0; w < NW; w++) { rr += ms->tmpf[16 + 16 * tid + w]; xx += ms->tmpf[24 + 16 * tid + w]; }
      hst[8 + tid] = sqrtf(rr) * (1.f + 1.f / 1024.f) * (1.f + 1.f / 256.f);
      hst[16 + tid] = sqrtf(xx) * (1.f + 1.f / 1024.f) * (1.f / 2048.f);
    }
    __syncthreads();
  }
  MARK(29);
  if constexpr (W) {
    prefetch_tasks<3>(r1 - r0, ttab, rg, pst[warp], [&](int task, TD& d) {
      d.w = p_hsk + (size_t)(r0 + task) * HSK_ITEMS * 32; d.x = xs;
      d.s = reinterpret_cast<const u32*>(hb8 + HSK_XNF);
      d.n = HSK_ITEMS; d.slot = r0 + task; d.bm = 0xffu;
    });
  }
  consume_tasks<W ? 3 : 2>(ttab, HSK_XST / 2, rg, pst[warp], [&](int rt, const float* c, const float* hx) {
    // lane (g, t): rows g, g + 8 x columns 2t, 2t + 1. W: hi sums in c, lo sums in hx[12..15];
    // !W: hi sums of column b in column b, lo sums in column 4 + b (lane t + 2)
    int lo[4];
#pragma unroll
    for (int j = 0; j < 4; j++) lo[j] = W ? __float_as_int(hx[12 + j]) : __shfl_xor_sync(0xffffffffu, __float_as_int(c[j]), 2);
    if (W || t < 1) {
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int q = 0; q < 2; q++) {
          const int b = 2 * t + q, j = 2 * h + q;
          if (b < B) {
            const float D = fmaf((float)__float_as_int(c[j]), 128.f, (float)lo[j]);
            const float sx = hx[4 + 4 * h], sqn = hx[5 + 4 * h], wn = hx[6 + 4 * h];
            const float ap = sx * hst[b] * D;
            const float bnd = (hx[j] * (1.f + 1.f / 256.f) + sqn * hst[8 + b] + wn * hst[16 + b]) * (1.f + 1.f / 1024.f);
            const float lb = ap - bnd, ub = ap + bnd;
            const int lk = __float_as_int(lb) >= 0 ? __float_as_int(lb) : (__float_as_int(lb) ^ 0x7fffffff);
            atomicMax(&hsi[32 + b], lk);
            const int cur = hsi[32 + b];
            const float cv = __int_as_float(cur >= 0 ? cur : (cur ^ 0x7fffffff));
            if (ub >= cv) {
              const int ix = atomicAdd(&hsi[40], 1);
              if (ix < CAP) hlist[ix] = make_int2((rt * 16 + g + 8 * h) | (b << 24), __float_as_int(ub));
              else hsi[41] = 1;
            }
          }
        }
    }
  });
        __syncthreads();
        MARK(30);
        if (tid < B) { const int k = hsi[32 + tid]; p_hsl[bid * MAXB + tid] = __int_as_float(k >= 0 ? k : (k ^ 0x7fffffff)); }
#ifndef HSK_PF
#define HSK_PF 0     // local candidate rowtiles pulled into L2 before the barrier, at most (2: +3.5 us at B=1 since HSK_FASTRS)
#endif
        if (HSK_PF > 0) {
          // the final candidates are a subset of these; their exact rescoring then streams from L2
          u32* hlm = reinterpret_cast<u32*>(hsi + 46);
          const int nl = min(hsi[40], CAP);
          for (int i = tid; i < nl; i += NT) {
            const int2 en = hlist[i];
            const int b = en.x >> 24, k = hsi[32 + b];
            if (__int_as_float(en.y) >= __int_as_float(k >= 0 ? k : (k ^ 0x7fffffff))) {
              const int r = ((en.x & 0xffffff) >> 4) - r0;
              atomicOr(&hlm[r >> 5], 1u << (r & 31));
            }
          }
          __syncthreads();
          if (tid == 0) {
            int np = 0;
            for (int w = 0; w < 3 && np < HSK_PF; w++)
              for (u32 m = hlm[w]; m && np < HSK_PF; m &= m - 1, np++)
                l2_prefetch(p_head + (size_t)(r0 + 32 * w + __ffs(m) - 1) * KS_H * 32, KS_H * 512);
          }
        }
        MARK(MKPROF_WAIT);
#if HSK_WARM
        if constexpr (!W && HSK_FASTRS) {
          // arrive, warm the rescoring code with 2 stages of this block's first head tile, then wait
          __syncthreads();
          bar_target += NB;
          if (tid == 0) { __threadfence(); atomicAdd(bar, 1u); }
          if (warp == 0) {
            if (lane == 0) hcand[0] = r0;
            __syncwarp();
            rescore_tiles(p_head, hcand, 1, 1, 2, ms, rg.data, xs, RsBest{0.f, 0.f, 0, 0});
          }
          if (tid == 0) { while (ld_acquire(bar) < bar_target) { } }
          __syncthreads();
        } else
#endif
        grid_bar(bar, bar_target);
        MARK(31);
        u32* hmask = reinterpret_cast<u32*>(hsi + 43);   // [13..15]: candidate rowtile bits of this block
        if (warp < B) {
          // warp b: the global best lower bound of column b
          const int b = warp;
          float L = -INFINITY;
#pragma unroll
          for (int j = 0; j < (NB + 31) / 32; j++) { const int k = lane + 32 * j; if (k < NB) L = fmaxf(L, ldcgf(p_hsl + k * MAXB + b)); }
#pragma unroll
          for (int o = 16; o; o >>= 1) L = fmaxf(L, __shfl_xor_sync(0xffffffffu, L, o));
          if (lane == 0) hst[24 + b] = L;
        }
        __syncthreads();
        {
          const int nl = min(hsi[40], CAP);
          if (hsi[41]) { if (tid < r1 - r0) atomicOr(&hmask[tid >> 5], 1u << (tid & 31)); }
          else for (int i = tid; i < nl; i += NT) {
            const int2 en = hlist[i];
            const int b = en.x >> 24, row = en.x & 0xffffff;
            if (__int_as_float(en.y) >= hst[24 + b]) { const int r = (row >> 4) - r0; atomicOr(&hmask[r >> 5], 1u << (r & 31)); }
          }
        }
        __syncthreads();
        if (tid == 0) {
          int nc = 0;
          for (int w = 0; w < 3; w++) for (u32 m = hmask[w]; m; m &= m - 1) hcand[nc++] = r0 + 32 * w + __ffs(m) - 1;
          hsi[42] = nc;
          if (nc) atomicAdd(p_hdbg, nc);
          if (hsi[41]) atomicAdd(p_hdbg + 1, 1);
          if (bid == 0) atomicAdd(p_hdbg + 2, 1);
        }
        __syncthreads();
        int nc = hsi[42];
#if HSK_RQ
        if constexpr (W) {
          // the block's candidate rowtiles into a global queue; after one more grid barrier block j rescores this
          // step's entries j, j + NB, ... (the per-block (value, row) results are reduced order-free: ties go to the
          // lowest row), so that tiles that cluster on one block's rows are spread over SMs
          u32* hq = reinterpret_cast<u32*>(p_hsl + NB * MAXB + MAXB * HID / 2);   // [0] entries so far, [1..] tiles
          if (tid < nc) { const u32 j = atomicAdd(hq, 1u); hq[1 + j % HQ_CAP] = (u32)hcand[tid]; }
          MARK(MKPROF_WAIT);
          grid_bar(bar, bar_target);
          MARK(40);
          if (tid == 0) {
            const u32 base = ms->hqb, tot = ld_acquire(hq);
            int n2 = 0;
            for (u32 j = base + bid; j < tot; j += NB) hcand[n2++] = (int)__ldcg(hq + 1 + j % HQ_CAP);
            hsi[42] = n2; ms->hqb = tot;
          }
          __syncthreads();
          nc = hsi[42];
        }
#endif
        MARK(38);
        if (W && nc > 0) {
#if HSK_XG == 2
          // the published rows by one bulk copy each (pooled-ring mbarrier slot RS_SLOTS - 1: idle until FASTRS)
          if (tid == 0) {
            const u32 bb = smaddr(&ms->rsbar[RS_SLOTS - 1]);
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbar_expect(bb, B * HID * 2);
            for (int b = 0; b < B; b++) tma_load(smaddr(xs + b * XSH), reinterpret_cast<const u16*>(p_hsl + NB * MAXB) + b * HID, HID * 2, bb);
          }
          mbar_wait(smaddr(&ms->rsbar[RS_SLOTS - 1]), ms->rsuse[RS_SLOTS - 1] & 1);
          __syncthreads();
          if (tid == 0) ms->rsuse[RS_SLOTS - 1]++;
          __syncthreads();
#elif HSK_XG
          // the normed rows published by blocks 0..B-1 before the barrier (fragment order): a copy, no norm
          const uint4* src = reinterpret_cast<const uint4*>(p_hsl + NB * MAXB);
          for (int i0 = 0; i0 < B * (HID / 8); i0 += 4 * NT) {
            uint4 v[4];
#pragma unroll
            for (int j = 0; j < 4; j++) { const int i = i0 + tid + NT * j; if (i < B * (HID / 8)) v[j] = __ldcg(src + i); }
#pragma unroll
            for (int j = 0; j < 4; j++) { const int i = i0 + tid + NT * j; if (i < B * (HID / 8)) *reinterpret_cast<uint4*>(xs + (i / (HID / 8)) * XSH + (i % (HID / 8)) * 8) = v[j]; }
          }
          __syncthreads();
#else
          const bf16* rows[MAXB];
          for (int b = 0; b < MAXB; b++) rows[b] = p_x + b * HID;
          norm_warp(rows, B, nwb, xs);
#endif
        }
        MARK(39);
        auto rs_fin = [&](int rt, const float* c) {
          const int row0 = rt * 16 + g, row1 = row0 + 8;
          if (c[0] > bv || (c[0] == bv && row0 < bi)) { bv = c[0]; bi = row0; }
          if (c[2] > bv || (c[2] == bv && row1 < bi)) { bv = c[2]; bi = row1; }
          if (c[1] > bv2 || (c[1] == bv2 && row0 < bi2)) { bv2 = c[1]; bi2 = row0; }
          if (c[3] > bv2 || (c[3] == bv2 && row1 < bi2)) { bv2 = c[3]; bi2 = row1; }
        };
        MARK(36);
#if HSK_WARM
        if constexpr (!W && HSK_FASTRS) {
          const RsBest r = rescore_tiles(p_head, hcand, nc, min(nc, 3), KS_H / RI, ms, rg.data - warp * RING_WARP, xs, RsBest{bv, bv2, bi, bi2});
          bv = r.bv; bv2 = r.bv2; bi = r.bi; bi2 = r.bi2;
          MARK(37);
        } else
#endif
        if ((W ? (HSK_FASTRS_W || HSK_RQ) : HSK_FASTRS)) {
          // warp 0 rescores the candidate rowtiles one after the other, each with all its 20 stages in flight at once
          // in the (now idle) ring memory of all warps; the per-stage arithmetic is consume_tasks<0>'s (same bits)
          // up to RS_SLOTS / 8 candidate tiles at once, one warp each, the pooled ring's stage slots shared out (a
          // warp with S slots keeps S of its tile's 20 stages in flight and refills a slot once it is consumed)
          const int nact = min(nc, W ? FASTRS_WMAX : FASTRS_FMAX);
          if (warp < nact) {
            const int S = min(KS_H / RI, RS_SLOTS / nact), sb = warp * S;
            const u32 ring0 = rg.data - warp * RING_WARP;
            const u32 xb = smaddr(xs + g * XSH) + t * 32;
            for (int ci = warp; ci < nc; ci += nact) {
              const int rt = hcand[ci];
              __syncwarp();
              auto issue = [&](int st) {
                const int sl = sb + st % S;
                const u32 bb = smaddr(&ms->rsbar[sl]);
                mbar_expect(bb, RI * 512);
                tma_load(ring0 + sl * RI * 512, p_head + ((size_t)rt * KS_H + st * RI) * 32, RI * 512, bb);
              };
#ifdef RSTWICE
              for (int pass = 0; pass < 2; pass++) {
              if (pass == 1 && ci == 0) MARK(41);
#endif
              if (lane == 0) {
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                for (int st = 0; st < S; st++) issue(st);
              }
              float c[4] = {0.f, 0.f, 0.f, 0.f};
              for (int st = 0; st < KS_H / RI; st++) {
                const int sl = sb + st % S;
                mbar_wait(smaddr(&ms->rsbar[sl]), ms->rsuse[sl] & 1);
                const u32 stg = ring0 + sl * RI * 512;
                float c2[4] = {0.f, 0.f, 0.f, 0.f};
#if HSK_RSPRE
                // the stage's A and B fragments first, then item<0>'s mma sequence (same operands, same order)
                uint4 qa[RI]; u32 bb0[RI], bb1[RI];
#pragma unroll
                for (int i = 0; i < RI; i++) {
                  const int k = st * RI + i;
                  qa[i] = lds128(stg + i * 512 + lane * 16);
                  asm("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(bb0[i]), "=r"(bb1[i]) : "r"(xb + (k >> 2) * 128 + (k & 3) * 8));
                }
#pragma unroll
                for (int i = 0; i < RI; i++) mma16816((i & 1) ? c2 : c, qa[i].x, qa[i].y, qa[i].z, qa[i].w, bb0[i], bb1[i]);
#elif HSK_RSROLL
                // rolled over item pairs (a small loop body: this code runs once per step, from a cold instruction
                // cache); the same accumulation order: items 0, 2, .., 8 into c, 1, 3, .., 7 into c2
#pragma unroll 1
                for (int i = 0; i < RI - 1; i += 2) {
                  item<0>(c, stg, xb, st * RI, i, lane, g, true);
                  item<0>(c2, stg, xb, st * RI, i + 1, lane, g, true);
                }
                item<0>(c, stg, xb, st * RI, RI - 1, lane, g, true);
#else
#pragma unroll
                for (int i = 0; i < RI; i++) item<0>((i & 1) ? c2 : c, stg, xb, st * RI, i, lane, g, true);
#endif
#pragma unroll
                for (int j = 0; j < 4; j++) c[j] += c2[j];
                __syncwarp();
                if (lane == 0) {
                  ms->rsuse[sl]++;
                  if (st + S < KS_H / RI) { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); issue(st + S); }
                }
                __syncwarp();
              }
#ifdef RSTWICE
              if (pass == 1) rs_fin(rt, c);
              }
#else
              rs_fin(rt, c);
#endif
            }
          }
          MARK(37);
        } else
        run_tasks<0>(nc, ttab, XSH, rg, pst[warp], [&](int task, TD& d) {
          const int rt = hcand[task];
          d.w = p_head + (size_t)rt * KS_H * 32; d.x = xs; d.n = KS_H; d.slot = rt; d.bm = (1u << B) - 1;
        }, rs_fin);
  return HskOut{rg.pos, bar_target, bv, bv2, bi, bi2, (u32)pn};
}
__device__ __noinline__ HskOut hsk_head_wide(const uint4* p_head, const uint4* p_hsk, float* p_hsl, int* p_hdbg, const bf16* p_x,
                                            const u16* nwb, Misc* ms, u16* xs, float* red, TT* ttab, Ring rg, PState* pst,
                                            int B, u32* bar, u32 bar_target, int r0, int r1 HSK_PA) {
  return hsk_head<true>(p_head, p_hsk, p_hsl, p_hdbg, p_x, nwb, ms, xs, red, ttab, rg, pst, B, bar, bar_target, r0, r1 HSK_PV);
}
__device__ __noinline__ HskOut hsk_head_flat(const uint4* p_head, const uint4* p_hsk, float* p_hsl, int* p_hdbg, const bf16* p_x,
                                            const u16* nwb, Misc* ms, u16* xs, float* red, TT* ttab, Ring rg, PState* pst,
                                            int B, u32* bar, u32 bar_target, int r0, int r1 HSK_PA) {
  return hsk_head<false>(p_head, p_hsk, p_hsl, p_hdbg, p_x, nwb, ms, xs, red, ttab, rg, pst, B, bar, bar_target, r0, r1 HSK_PV);
}
#endif

static_assert(sizeof(Misc) <= 4096, "Misc must fit its 4 KB of shared memory");
__device__ void combine_flat(const Params& p, const Misc* ms, int B, u16* dst) {
  const int tid = threadIdx.x;
#ifdef COMB_BATCH   // loads first, 3 items per batch: spills or (noinline) +80 us at B=1; not used
  // 3 items per thread per batch, all loads issued before the stores (B = 1: one batch)
  for (int i0 = 0; i0 < B * (HID / 4); i0 += 3 * NT) {
    uint4 y[3][4]; uint2 xv[3];
#pragma unroll
    for (int j = 0; j < 3; j++) {
      const int i = i0 + tid + NT * j;
      if (i < B * (HID / 4)) {
        const int b = i / (HID / 4), k = (i % (HID / 4)) * 4, nr = ms->nrs[b];
#pragma unroll
        for (int q = 0; q < 4; q++)
          y[j][q] = (q < nr) ? __ldcg(reinterpret_cast<const uint4*>(p.ypart + ((size_t)ms->rsl[b][q] * MAXB + b) * HID + k)) : make_uint4(0, 0, 0, 0);
        xv[j] = __ldcg(reinterpret_cast<const uint2*>(p.xm + b * HID + k));
      }
    }
#pragma unroll
    for (int j = 0; j < 3; j++) {
      const int i = i0 + tid + NT * j;
      if (i < B * (HID / 4)) {
        const int b = i / (HID / 4), k = (i % (HID / 4)) * 4, nr = ms->nrs[b];
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 4; q++) if (q < nr) {
          const float wv = ms->rw[b][ms->rsl[b][q]];
          acc[0] = fmaf(wv, __uint_as_float(y[j][q].x), acc[0]); acc[1] = fmaf(wv, __uint_as_float(y[j][q].y), acc[1]);
          acc[2] = fmaf(wv, __uint_as_float(y[j][q].z), acc[2]); acc[3] = fmaf(wv, __uint_as_float(y[j][q].w), acc[3]);
        }
        u16 o0 = f2u16(__uint_as_float(xv[j].x << 16) + bfr(acc[0]));
        u16 o1 = f2u16(__uint_as_float(xv[j].x & 0xffff0000u) + bfr(acc[1]));
        u16 o2 = f2u16(__uint_as_float(xv[j].y << 16) + bfr(acc[2]));
        u16 o3 = f2u16(__uint_as_float(xv[j].y & 0xffff0000u) + bfr(acc[3]));
        uint2 o = make_uint2(o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
        *reinterpret_cast<uint2*>(dst + b * HID + k) = o;
        if ((k >> 2) % NB == (int)blockIdx.x) *reinterpret_cast<uint2*>(p.x + b * HID + k) = o;
      }
    }
  }
#else
  for (int i = tid; i < B * (HID / 4); i += NT) {
    const int b = i / (HID / 4), k = (i % (HID / 4)) * 4;
    const int nr = ms->nrs[b];
    uint4 y[4];
#pragma unroll
    for (int q = 0; q < 4; q++)
      y[q] = (q < nr) ? __ldcg(reinterpret_cast<const uint4*>(p.ypart + ((size_t)ms->rsl[b][q] * MAXB + b) * HID + k)) : make_uint4(0, 0, 0, 0);
    const uint2 xv = __ldcg(reinterpret_cast<const uint2*>(p.xm + b * HID + k));
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int q = 0; q < 4; q++) if (q < nr) {
      const float wv = ms->rw[b][ms->rsl[b][q]];
      acc[0] = fmaf(wv, __uint_as_float(y[q].x), acc[0]); acc[1] = fmaf(wv, __uint_as_float(y[q].y), acc[1]);
      acc[2] = fmaf(wv, __uint_as_float(y[q].z), acc[2]); acc[3] = fmaf(wv, __uint_as_float(y[q].w), acc[3]);
    }
    u16 o0 = f2u16(__uint_as_float(xv.x << 16) + bfr(acc[0]));
    u16 o1 = f2u16(__uint_as_float(xv.x & 0xffff0000u) + bfr(acc[1]));
    u16 o2 = f2u16(__uint_as_float(xv.y << 16) + bfr(acc[2]));
    u16 o3 = f2u16(__uint_as_float(xv.y & 0xffff0000u) + bfr(acc[3]));
    uint2 o = make_uint2(o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
    *reinterpret_cast<uint2*>(dst + b * HID + k) = o;
    if ((k >> 2) % NB == (int)blockIdx.x) *reinterpret_cast<uint2*>(p.x + b * HID + k) = o;
  }
#endif
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------ the kernel
// two instances: WIDE = false for batches of 1-2 (flat down phase, sums in the next phase), true for 3-8 (combine
// step); each gets its own register allocation and none of the other's code
#if KVPF_OL
// KVPF as an out-of-line call (the same prefetches; called by threads 0 .. 2 NW - 1 of every block)
__device__ __noinline__ void kvpf_issue(const Params& p, const LayerP& Lp, int pos, int B, int bid, int tid) {
  const int lo = Lp.sliding ? max(0, pos - 127) : 0;
  const int c0 = lo >> 6, nch = (pos >> 6) - c0 + 1;
  if (ATT_BLK && nch <= NW * (B >= 3 ? ATT_BLKW : 1) && B * NKV <= NB) {
    if (bid < B * NKV && tid < 2) {
      const uint4* base = reinterpret_cast<const uint4*>(tid ? Lp.vc : Lp.kc) + ((size_t)bid * p.maxctx * HD) / 8 + (size_t)c0 * 512;
      l2_prefetch(base, nch * 8192);
    }
  } else {
    const int item = bid + NB * (tid >> 1);
    if (item < B * NKV * nch) {
      const int unit = item / nch, ch = c0 + item % nch;
      const uint4* base = reinterpret_cast<const uint4*>((tid & 1) ? Lp.vc : Lp.kc) + ((size_t)unit * p.maxctx * HD) / 8 + (size_t)ch * 512;
      l2_prefetch(base, 8192);
    }
  }
}
#endif
#if QXO
// QXO: V(pos) of one (sequence, kv head >= 4) from the 10 segment partials of its 4 rowtiles (the qkv epilogue's
// arithmetic, same bits); called by the attention warp of the unit's last chunk only
__device__ __noinline__ void qx_vpos(const float* ypart1, const float* bqkv, bf16* vc, int maxctx, int b, int kvh, int pos) {
  const int lane = threadIdx.x & 31, jj = lane >> 3, i = lane & 7, x = (kvh - 4) * 4 + jj, grt = 304 + x;
  const float* xb = ypart1 + (size_t)x * NSEG_QKV * 128;
  const int o1 = ((i & 7) * 4 + (b >> 1)) * 4 + (b & 1), o2 = o1 + 2;
  float v1 = 0.f, v2 = 0.f;
  for (int sg = 0; sg < NSEG_QKV; sg++) { v1 += __ldcg(xb + sg * 128 + o1); v2 += __ldcg(xb + sg * 128 + o2); }
  v1 = bfr(v1 + bqkv[grt * 16 + i]);
  v2 = bfr(v2 + bqkv[grt * 16 + i + 8]);
  u16* vp = reinterpret_cast<u16*>(vc) + ((size_t)b * NKV + kvh) * maxctx * HD;
  vp[vfo(pos, 16 * jj + i)] = f2u16(v1);
  vp[vfo(pos, 16 * jj + i + 8)] = f2u16(v2);
  __syncwarp();
}
#endif
template <bool WIDE>
__global__ void __launch_bounds__(NT, 1) mk_kernel(const __grid_constant__ Params p, const __grid_constant__ RunArgs a) {
  constexpr int PFK = WIDE ? PF_STAGES_W : PF_STAGES;   // prefetch depth of this instance
#if QXO
  constexpr bool QXI = (QXO == 1) || (QXO == 2 && WIDE) || (QXO == 3 && !WIDE);   // QXO in this instance
#endif
  extern __shared__ __align__(16) unsigned char smem[];
  u16* xs = reinterpret_cast<u16*>(smem + SM_XS);
  float* red = reinterpret_cast<float*>(smem + SM_RED);
  Misc* ms = reinterpret_cast<Misc*>(smem + SM_MISC);
  TT* ttab = reinterpret_cast<TT*>(smem + SM_TT);
  PState* pst = reinterpret_cast<PState*>(smem + SM_PS);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  Ring rg;
  rg.data = smaddr(smem + SM_RING) + warp * RING_WARP;
  rg.gdata = smem + SM_RING + warp * RING_WARP;
  rg.bars = smaddr(smem + SM_BARS) + warp * RS * 8;
  rg.pos = 0;
  if (lane == 0) for (int k = 0; k < RS; k++) mbar_init(rg.bars + k * 8, 1);
  const u32 nwbar = smaddr(smem + SM_NWBAR), nwdst = smaddr(smem + SM_NWB);
  const u16* nwb = reinterpret_cast<const u16*>(smem + SM_NWB);
  u32 nwpos = 0;   // norm-weight loads waited for
  if (tid == 0) mbar_init(nwbar, 1);
  const u32 abar = nwbar + 8;      // ATMA: the attention output's bulk copies
  u32 apos = 0;
  if (tid == 0) mbar_init(abar, 1);
  if (tid < 32) ms->eep[tid] = 0;
  if (tid == 0) ms->hqb = 0;
  if (tid == 0) { for (int k = 0; k < RS_SLOTS; k++) { mbar_init(smaddr(&ms->rsbar[k]), 1); ms->rsuse[k] = 0; } }
  if (tid == 0) {
    // CSPLIT roles: the k-th block with two o_proj rowtiles (owner) shares the K segments 2, 3 of its second
    // rowtile with the k-th block with one rowtile (partner)
    ms->cs_role = 0; ms->cs_k = 0; ms->cs_x = 0;
    if (CSPLIT) {
      int ntwo = 0, none = 0;
      for (int j = 0; j < NB; j++) {
        const int r0j = j * RT_O / NB, Rj = (j + 1) * RT_O / NB - r0j;
        if (Rj == 2) { if (j == (int)blockIdx.x) { ms->cs_role = 1; ms->cs_k = ntwo; ms->cs_x = r0j + 1; } ntwo++; }
        else { if (j == (int)blockIdx.x && none < RT_O - NB) { ms->cs_role = 2; ms->cs_k = none; } none++; }
      }
      if (ms->cs_role == 2) {    // the owner's shared rowtile
        int k = 0;
        for (int j = 0; j < NB; j++) {
          const int r0j = j * RT_O / NB, Rj = (j + 1) * RT_O / NB - r0j;
          if (Rj == 2) { if (k == ms->cs_k) ms->cs_x = r0j + 1; k++; }
        }
      }
    }
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  const int bid = blockIdx.x;
  const int B = a.B;
  u32 bar_target = 0;
  u32 lep = 0;          // layers run in this launch (BCOV: target of the per-kv-head attention counters)
  MkProf prof{p.prof, p.prof_cap};
  MKPROF_DECL(pn);
#ifdef SKEL
  if (tid == 0 && bid == 0) g_skel = p.skel;
  // (every block reads g_skel after the first barrier; set again by each block: same value)
  g_skel = p.skel;
#endif
  u32* bar = p.bar;
  // P = prompt length (mode 0/1: 1); total steps = P + nsteps - 1; step st reads token st of the prompt while
  // st < P, otherwise the argmax of the previous step (= generated token st - P); the head runs for st >= P - 1.
  const int P = (a.mode == 2) ? a.nprompt : 1;
#ifndef FLAT_MAXB
#define FLAT_MAXB 2
#endif
  constexpr bool flat = !WIDE;   // down phase: even (slot, rowtile) units, partials summed by the next phase
  const int total = P + a.nsteps - 1;

  // ---- the next norm's weights (BF16, 5760 B) into shared memory by one bulk copy; wait_nw() before use
  auto issue_nw = [&](const u16* src) {
    if (tid == 0) { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); mbar_expect(nwbar, HID * 2); tma_load(nwdst, src, HID * 2, nwbar); }
  };
  auto wait_nw = [&]() { mbar_wait(nwbar, nwpos & 1); nwpos++; };
  // ---- prefetch helpers: build a warp's task table and issue its first ring stages (weights only, no data needed)
  auto pf_qkv = [&](int l, int kp) {
    const LayerP& L = p.L[l];
#if QXO
    const int r0 = QXI ? 2 * bid : bid * RT_QKV / NB, R = QXI ? 2 : (bid + 1) * RT_QKV / NB - r0;
    const int nx = (QXI && bid < 40) ? 4 : 0;
#else
    const int r0 = bid * RT_QKV / NB, R = (bid + 1) * RT_QKV / NB - r0;
#endif
#if KVPF && KVPF_PLACE == 1
    // KVPF one phase earlier: with the cross-barrier prefetch of layer l's qkv weights (position kp)
    if (tid < 2 * NW) {
      const int lo = L.sliding ? max(0, kp - 127) : 0;
      const int c0 = lo >> 6, nch = (kp >> 6) - c0 + 1;
      if (ATT_BLK && nch <= NW * (WIDE ? ATT_BLKW : ATT_BLKW_F) && B * NKV <= NB) {
        if (bid < B * NKV && tid < 2) {
          const uint4* base = reinterpret_cast<const uint4*>(tid ? L.vc : L.kc) + ((size_t)bid * p.maxctx * HD) / 8 + (size_t)c0 * 512;
          l2_prefetch(base, nch * 8192);
        }
      } else {
        const int item = bid + NB * (tid >> 1);
        if (item < B * NKV * nch) {
          const int unit = item / nch, ch = c0 + item % nch;
          const uint4* base = reinterpret_cast<const uint4*>((tid & 1) ? L.vc : L.kc) + ((size_t)unit * p.maxctx * HD) / 8 + (size_t)ch * 512;
          l2_prefetch(base, 8192);
        }
      }
    }
#else
    (void)kp;
#endif
#if !XPF && QXO
    if constexpr (QXI) {
    prefetch_tasks<0, PFK>(R * NSEG_QKV + nx, ttab, rg, pst[warp], [&](int task, TD& d) {
      int r = task / NSEG_QKV, s = task % NSEG_QKV;
      int rt = r0 + r;
      if (task >= R * NSEG_QKV) { const int gi = bid * 4 + task - R * NSEG_QKV; rt = 304 + gi / NSEG_QKV; s = gi % NSEG_QKV; }
      const int k0 = s * KS_H / NSEG_QKV, k1 = (s + 1) * KS_H / NSEG_QKV;
      d.w = L.wqkv + ((size_t)rt * KS_H + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
    });
    } else {
    prefetch_tasks<0, PFK>(R * NSEG_QKV, ttab, rg, pst[warp], [&](int task, TD& d) {
      int r = task / NSEG_QKV, s = task % NSEG_QKV, k0 = s * KS_H / NSEG_QKV, k1 = (s + 1) * KS_H / NSEG_QKV;
      d.w = L.wqkv + ((size_t)(r0 + r) * KS_H + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
    });
    }
#elif !XPF
    prefetch_tasks<0, PFK>(R * NSEG_QKV, ttab, rg, pst[warp], [&](int task, TD& d) {
      int r = task / NSEG_QKV, s = task % NSEG_QKV, k0 = s * KS_H / NSEG_QKV, k1 = (s + 1) * KS_H / NSEG_QKV;
      d.w = L.wqkv + ((size_t)(r0 + r) * KS_H + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
    });
#else
    const int nq = R * NSEG_QKV;
    // XPF: the router blocks also compute the expert predictor (its rowtile and K quarter, as in phase D)
    const bool xp = bid >= NB - 8 * RQ;
    prefetch_tasks<0, PFK>(nq + (xp ? NW : 0), ttab, rg, pst[warp], [&](int task, TD& d) {
      if (task < nq) {
        int r = task / NSEG_QKV, s = task % NSEG_QKV, k0 = s * KS_H / NSEG_QKV, k1 = (s + 1) * KS_H / NSEG_QKV;
        d.w = L.wqkv + ((size_t)(r0 + r) * KS_H + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
      } else {
        const int rr = bid - (NB - 8 * RQ), rt = rr & 7, q = rr >> 3, w = task - nq;
        const int kq0 = q * KS_H / RQ, kq1 = (q + 1) * KS_H / RQ;
        int k0 = kq0 + w * (kq1 - kq0) / NW, k1 = kq0 + (w + 1) * (kq1 - kq0) / NW;
        d.w = L.wp + ((size_t)rt * KS_H + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
      }
    });
#endif
  };
  auto pf_o = [&](int l) {
    const LayerP& L = p.L[l];
#if OSPLIT
    // every block one full rowtile (bid); the RT_O - NB extra rowtiles in K quarters on blocks 0 .. 4 (RT_O - NB) - 1
    const int nq = (bid < 4 * (RT_O - NB)) ? NSEG_O / 4 : 0;
    prefetch_tasks<0, PFK>(NSEG_O + nq, ttab, rg, pst[warp], [&](int task, TD& d) {
      const int rt = (task < NSEG_O) ? bid : NB + bid / 4;
      const int s = (task < NSEG_O) ? task : (bid % 4) * (NSEG_O / 4) + task - NSEG_O;
      const int k0 = s * KS_A / NSEG_O, k1 = (s + 1) * KS_A / NSEG_O;
      d.w = L.wo + ((size_t)rt * KS_A + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
    });
    return;
#endif
    const int r0 = bid * RT_O / NB, R = (bid + 1) * RT_O / NB - r0;
#if CSPLIT
    if (ms->cs_role) {
      // owner: own rowtile r0 (tasks 0..3) + segments 0, 1 of its shared rowtile (tasks 4, 5);
      // partner: segments 2, 3 of the owner's shared rowtile first (tasks 0, 1), then its own rowtile (tasks 2..5)
      const int role = ms->cs_role, x = ms->cs_x;
      prefetch_tasks<0, PFK>(NSEG_O + 2, ttab, rg, pst[warp], [&](int task, TD& d) {
        int rt, s;
        if (role == 1) { rt = (task < NSEG_O) ? r0 : x; s = (task < NSEG_O) ? task : task - NSEG_O; }
        else { rt = (task < 2) ? x : r0; s = (task < 2) ? 2 + task : task - 2; }
        const int sl = (KS_A / NSEG_O) / RI * RI;
        const int k0 = s * sl, k1 = (s == NSEG_O - 1) ? KS_A : (s + 1) * sl;
        d.w = L.wo + ((size_t)rt * KS_A + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
      });
      return;
    }
#endif
    prefetch_tasks<0, PFK>(R * NSEG_O, ttab, rg, pst[warp], [&](int task, TD& d) {
      int r = task / NSEG_O, s = task % NSEG_O, k0 = s * KS_A / NSEG_O, k1 = (s + 1) * KS_A / NSEG_O;
#if OALIGN
      // segment bounds on whole ring stages (RI ksteps); the remainder goes to the last segment
      const int sl = (KS_A / NSEG_O) / RI * RI;
      k0 = s * sl; k1 = (s == NSEG_O - 1) ? KS_A : (s + 1) * sl;
#endif
      d.w = L.wo + ((size_t)(r0 + r) * KS_A + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
    });
  };
  auto pf_router = [&](int l) {
    const LayerP& L = p.L[l];
    const int rr = bid - (NB - 8 * RQ), rt = rr & 7, q = rr >> 3;
    const int kq0 = q * KS_H / RQ, kq1 = (q + 1) * KS_H / RQ;
    prefetch_tasks<0, PFK>(RTASK, ttab, rg, pst[warp], [&](int task, TD& d) {
      int k0 = kq0 + task * (kq1 - kq0) / RTASK, k1 = kq0 + (task + 1) * (kq1 - kq0) / RTASK;
      d.w = L.wr + ((size_t)rt * KS_H + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
    });
  };
  auto pf_head = [&]() {
    const int r0 = (int)((long)bid * RT_HEAD / NB), r1 = (int)((long)(bid + 1) * RT_HEAD / NB);
#if HSK
    if ((flat || HSK_WIDE) && a.mode != 0) {
      // B <= 2: the sketch stream starts here; wide: after the prep (which stages its rows in the ring)
      if (flat) prefetch_tasks<2, PFK>(r1 - r0, ttab, rg, pst[warp], [&](int task, TD& d) {
        d.w = p.hsk + (size_t)(r0 + task) * HSK_ITEMS * 32; d.x = reinterpret_cast<const u16*>(red);
        d.s = reinterpret_cast<const u32*>(reinterpret_cast<const unsigned char*>(red) + HSK_XNF);
        d.n = HSK_TAILREG ? 90 : HSK_ITEMS; d.slot = r0 + task; d.bm = 0xffu;
      });
#if HSK_L2PF > 0
      // the rest of each warp's first sketch task into L2 while the head input is prepared (HBM is idle then)
      if (flat && lane == 0 && warp < r1 - r0)
        l2_prefetch(p.hsk + ((size_t)(r0 + warp) * HSK_ITEMS + PF_STAGES * RI) * 32, (HSK_L2PF) * 512);
#endif
      return;
    }
#endif
#if HTAIL
    {
      // exact head (track A): the block's n mod 8 last rowtiles as 4 whole-stage K segments each, spread over the warps
      const int n = r1 - r0, nw8 = n / NW * NW;
      prefetch_tasks<0, PFK>(nw8 + (n - nw8) * HTS, ttab, rg, pst[warp], [&](int task, TD& d) {
        if (task < nw8) { d.w = p.head + (size_t)(r0 + task) * KS_H * 32; d.n = KS_H; d.slot = r0 + task; }
        else {
          const int q = task - nw8, rt = r0 + nw8 + q / HTS, sg = q % HTS;
          d.w = p.head + ((size_t)rt * KS_H + sg * (KS_H / HTS)) * 32; d.k0 = sg * (KS_H / HTS); d.n = KS_H / HTS; d.slot = q | (1 << 29);
        }
        d.x = xs; d.bm = (1u << B) - 1;
      });
    }
#else
    prefetch_tasks<0, PFK>(r1 - r0, ttab, rg, pst[warp], [&](int task, TD& d) {
      d.w = p.head + (size_t)(r0 + task) * KS_H * 32; d.x = xs; d.n = KS_H; d.slot = r0 + task; d.bm = (1u << B) - 1;
    });
#endif
  };
  pf_qkv(0, a.pos0);
  issue_nw(p.L[0].ln1b);

  for (int st = 0; st < total; st++) {
    const int pos = a.pos0 + st;
#if STEPARG
    // the previous step's token: warp b reduces the NB per-block candidates of sequence b (5 loads per lane at once,
    // then a shuffle argmax with the lowest index on ties: the result of the serial scan)
    if (st >= P && warp < B) {
      const int b = warp;
      float bv = -INFINITY; int bi = 0x7fffffff;
#pragma unroll
      for (int j = 0; j < (NB + 31) / 32; j++) {
        const int k = lane + 32 * j;
        if (k < NB) {
          const float v = ldcgf(p.amv + k * MAXB + b); const int ix = ldcgi(p.ami + k * MAXB + b);
          if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
        }
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      if (lane == 0) {
        const long long tk = (bi >= 0 && bi < VOCAB) ? bi : 0;
        if (bid == 0) {
          if (a.mode == 1) a.out_tok[(size_t)b * a.nsteps + (st - P)] = tk;
          else a.host_tok[b * a.nsteps + (st - P)] = (int)tk;
        }
        ms->tok[b] = (int)tk;
      }
    }
    if (st < P && tid < B) {
      long long tk = a.tok[(size_t)tid * a.tok_stride + st];
      if (tk < 0 || tk >= VOCAB) {   // an id outside the vocabulary: flagged (the host refuses the next call), read as 0
        if (bid == 0) a.host_prog[1] = 1;
        tk = 0;
      }
      ms->tok[tid] = (int)tk;
    }
#else
    if (tid < B) {
      long long tk;
      if (st < P) {
        tk = a.tok[(size_t)tid * a.tok_stride + st];
        if (tk < 0 || tk >= VOCAB) {   // an id outside the vocabulary: flagged (the host refuses the next call), read as 0
          if (bid == 0) a.host_prog[1] = 1;
          tk = 0;
        }
      }
      else {
        float bv = -INFINITY; int bi = 0x7fffffff;
        for (int k = 0; k < NB; k++) {
          float v = ldcgf(p.amv + k * MAXB + tid); int ix = ldcgi(p.ami + k * MAXB + tid);
          if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
        }
        tk = (bi >= 0 && bi < VOCAB) ? bi : 0;
        if (bid == 0) {
          if (a.mode == 1) a.out_tok[(size_t)tid * a.nsteps + (st - P)] = tk;
          else a.host_tok[tid * a.nsteps + (st - P)] = (int)tk;
        }
      }
      ms->tok[tid] = (int)tk;
    }
#endif
    __syncthreads();
    MARK(MKPROF_STEP);
    if (a.mode == 2 && bid == 0 && tid == 0 && st > P - 1 + 0 && st >= P) { __threadfence_system(); *a.host_prog = st - P + 1; }

    const bf16* rows_in[MAXB];
    for (int layer = 0; layer < NL; layer++) {
      lep++;
      const LayerP& Lp = p.L[layer];
      const int dr0 = bid * RT_DN / NB, dR = (bid + 1) * RT_DN / NB - dr0;   // down rowtiles of this block (1 or 2)
      // rounds of the down phase: slots [s0, s1) whose live (slot, column) activation rows fit in xs (11 rows)
      auto fround = [&](int s0) {
        const int U = ms->U;
        const u32 bmask = (1u << B) - 1;
        if (tid == 0) {
          int s1 = s0, np = 0;
          while (s1 < U && np + __popc(ms->smask[s1] & bmask) <= 11 && (s1 + 1 - s0) * dR * NSEG_DN <= RED_SLOTS) { np += __popc(ms->smask[s1] & bmask); s1++; }
          ms->fr_s0 = s0; ms->fr_s1 = s1; ms->fr_np = np;
          int base = 0;
          for (int k = s0; k < s1; k++) {
            u32 m = ms->smask[k] & bmask;
            u64 r8 = 0;
            for (int b = 0; b < 8; b++) if (m >> b & 1) r8 |= (u64)(base++) << (8 * b);
            ms->srows[k - s0] = r8;
          }
        }
      };
      // flat path: units u in [u0, u1) of U * 180 (slot = u / 180, rowtile = u % 180); staged rows: the routed
      // (slot, column) pairs of the slots [fs0, fs1] this block touches
      // the gate_up rowtiles of slots [s0, s1) are all written (replaces the E -> F grid barrier)
      auto wait_slots = [&](int s0, int s1) {
#ifndef EF_BARRIER
        if (tid < s1 - s0) {
          const u32 target = (u32)RT_GU * (u32)ms->eep[s0 + tid];
          while (ld_acquire(p.ecnt + s0 + tid) < target) { }
        }
        __syncthreads();
#endif
      };
      // stage the routed activation rows of slots [s0, s1) (np rows, slot-major, sequence order) into dst in fragment
      // order; the loads of a thread are issued in groups of 8 before their stores (latency of L2, not serial)
      auto load_act_rows = [&](int s0, int s1, int np, u16* dst) {
        if (tid < np) {
          const u32 bmask = (1u << B) - 1;
          int q = tid, sl = s0, b = 0;
          for (; sl < s1; sl++) { const int c = __popc(ms->smask[sl] & bmask); if (q < c) break; q -= c; }
          const u32 m = ms->smask[sl] & bmask;
          for (b = 0; b < 8; b++) if (m >> b & 1) { if (q == 0) break; q--; }
          ms->tmpi[tid] = sl * MAXB + b;
        }
        __syncthreads();
        const int tot = np * (HID / 8);
        for (int i0 = 0; i0 < tot; i0 += 8 * NT) {
          uint4 v[8];
#pragma unroll
          for (int j = 0; j < 8; j++) {
            const int it = i0 + tid + NT * j;
            if (it < tot) v[j] = ldcg128(p.act + (size_t)ms->tmpi[it / (HID / 8)] * HID + (it % (HID / 8)) * 8);
          }
#pragma unroll
          for (int j = 0; j < 8; j++) {
            const int it = i0 + tid + NT * j;
            if (it < tot) xfo_store8(dst + (it / (HID / 8)) * XSH, (it % (HID / 8)) * 8, v[j]);
          }
        }
      };
      auto fround_flat = [&]() {
        const int N = ms->U * RT_DN;
        if (tid == 0) {
          const int u0 = (int)((long)bid * N / NB), u1 = (int)((long)(bid + 1) * N / NB);
          const u32 bmask = (1u << B) - 1;
          int s0 = u1 > u0 ? u0 / RT_DN : 0, s1 = u1 > u0 ? (u1 - 1) / RT_DN + 1 : 0;
          ms->fr_s0 = s0; ms->fr_s1 = s1; ms->fr_u0 = u0; ms->fr_u1 = u1;
          int base = 0;
          for (int k = s0; k < s1; k++) {
            u32 m = ms->smask[k] & bmask;
            u64 r8 = 0;
            for (int b = 0; b < 8; b++) if (m >> b & 1) r8 |= (u64)(base++) << (8 * b);
            ms->srows[k - s0] = r8;
          }
          ms->fr_np = base;
        }
      };
      auto pf_down_flat = [&]() {
        const int u0 = ms->fr_u0, s0 = ms->fr_s0;
        prefetch_tasks<1, PFK>((ms->fr_u1 - u0) * NSEG_DN, ttab, rg, pst[warp], [&](int task, TD& d) {
          int uu = u0 + task / NSEG_DN, sg = task % NSEG_DN;
          int slot = uu / RT_DN, rt = uu % RT_DN, e = ms->sexp[slot];
          int k0 = sg * KC / NSEG_DN, k1 = (sg + 1) * KC / NSEG_DN;
          d.w = Lp.dn_c + (((size_t)e * RT_DN + rt) * KC + k0) * 34;
          d.s = reinterpret_cast<const u32*>(Lp.dn_b + (size_t)e * HID + rt * 16);
          d.x = xs; d.k0 = k0; d.rows8 = ms->srows[slot - s0];
          d.n = k1 - k0; d.slot = task; d.bm = ms->smask[slot] & ((1u << B) - 1);
        });
      };
      // wide path (B >= 3): the same even (slot, rowtile) units as the flat path, done one slot at a time (its routed
      // activation rows staged in xs), one warp per unit with the 5 segments summed in order (bits of the flat path)
      // a round = slots [s, fr_cur) of this block whose routed activation rows fit in xs (11 rows)
      auto fround_wide = [&](int s) {
        if (tid == 0) {
          const u32 bmask = (1u << B) - 1;
          int e = s, base = 0;
          while (e < ms->fr_s1 && base + __popc(ms->smask[e] & bmask) <= 11) {
            const u32 m = ms->smask[e] & bmask;
            u64 r8 = 0;
            for (int b = 0; b < 8; b++) if (m >> b & 1) r8 |= (u64)(base++) << (8 * b);
            ms->srows[e - s] = r8;
            e++;
          }
          ms->fr_np = base; ms->fr_rs = s; ms->fr_cur = e;
        }
      };
      auto pf_down_wide = [&]() {
        const int s = ms->fr_rs, e2 = ms->fr_cur;
        const int ua = max(ms->fr_u0, s * RT_DN), ub = min(ms->fr_u1, e2 * RT_DN);
#if FTAIL
        // whole units (5 stages, one warp each) for the first 8 floor(n / 8) units; the n mod 8 remaining units as 5
        // one-stage segments each (summed in shared memory in the same order), so that no warp runs a 4th unit
        {
          const int n = ub - ua, nw8 = n / NW * NW, nt = nw8 + (n - nw8) * NSEG_DN;
          prefetch_tasks<1, PFK>(nt, ttab, rg, pst[warp], [&](int task, TD& d) {
            int uu, k0, nn, slot;
            if (task < nw8) { uu = ua + task; k0 = 0; nn = KC; slot = task | (1 << 30); }
            else { const int q = task - nw8; uu = ua + nw8 + q / NSEG_DN; k0 = (q % NSEG_DN) * (KC / NSEG_DN); nn = KC / NSEG_DN; slot = q | (1 << 29); }
            const int sl = uu / RT_DN, rt = uu % RT_DN, e = ms->sexp[sl];
            d.w = Lp.dn_c + (((size_t)e * RT_DN + rt) * KC + k0) * 34;
            d.s = reinterpret_cast<const u32*>(Lp.dn_b + (size_t)e * HID + rt * 16);
            d.x = xs; d.k0 = k0; d.rows8 = ms->srows[sl - s];
            d.n = nn; d.slot = slot; d.bm = ms->smask[sl] & ((1u << B) - 1);
          });
        }
#elif FHALF
        // each unit as two tasks: K chunks 0..26 (3 stages) and 27..44 (2 stages); halves alternate per warp
        prefetch_tasks<1, PFK>(2 * (ub - ua), ttab, rg, pst[warp], [&](int task, TD& d) {
          const int hf = (task & 1) ^ ((task >> 3) & 1);
          int uu = ua + (task >> 1), sl = uu / RT_DN, rt = uu % RT_DN, e = ms->sexp[sl];
          const int k0 = hf ? 27 : 0;
          d.w = Lp.dn_c + (((size_t)e * RT_DN + rt) * KC + k0) * 34;
          d.s = reinterpret_cast<const u32*>(Lp.dn_b + (size_t)e * HID + rt * 16);
          d.x = xs; d.k0 = k0; d.rows8 = ms->srows[sl - s];
          d.n = hf ? KC - 27 : 27; d.slot = task | (1 << 30); d.bm = ms->smask[sl] & ((1u << B) - 1);
        });
#else
        prefetch_tasks<1, PFK>(ub - ua, ttab, rg, pst[warp], [&](int task, TD& d) {
          int uu = ua + task, sl = uu / RT_DN, rt = uu % RT_DN, e = ms->sexp[sl];
          d.w = Lp.dn_c + (((size_t)e * RT_DN + rt) * KC) * 34;
          d.s = reinterpret_cast<const u32*>(Lp.dn_b + (size_t)e * HID + rt * 16);
          d.x = xs; d.k0 = 0; d.rows8 = ms->srows[sl - s];
          d.n = KC; d.slot = task | (1 << 30); d.bm = ms->smask[sl] & ((1u << B) - 1);
        });
#endif
      };
      auto pf_down = [&](int s0, int s1) {
        prefetch_tasks<1, PFK>((s1 - s0) * dR * NSEG_DN, ttab, rg, pst[warp], [&](int task, TD& d) {
          int uu = task / NSEG_DN, sg = task % NSEG_DN;
          int slot = s0 + uu / dR, r = uu % dR, e = ms->sexp[slot];
          int k0 = sg * KC / NSEG_DN, k1 = (sg + 1) * KC / NSEG_DN;
          d.w = Lp.dn_c + (((size_t)e * RT_DN + dr0 + r) * KC + k0) * 34;
          d.s = reinterpret_cast<const u32*>(Lp.dn_b + (size_t)e * HID + (dr0 + r) * 16);
          d.x = xs; d.k0 = k0; d.rows8 = ms->srows[uu / dR];
          d.n = k1 - k0; d.slot = task; d.bm = ms->smask[slot] & ((1u << B) - 1);
        });
      };
#ifndef PF_LATE_F
#define PF_LATE_F 0
#endif
      auto pf_next = [&]() {
        if (layer < NL - 1) { pf_qkv(layer + 1, pos); issue_nw(p.L[layer + 1].ln1b); }
        else if (st >= P - 1) { pf_head(); issue_nw(p.lnfb); }
        else if (st + 1 < total) { pf_qkv(0, pos + 1); issue_nw(p.L[0].ln1b); }
      };
      // ---------------- phase A: norm1 + qkv
      MARK(1);
#if KVPF && KVPF_PLACE == 1
#elif KVPF && KVPF_OL
      if (tid < 2 * NW) kvpf_issue(p, Lp, pos, B, bid, tid);
#elif KVPF || KVPF_F
      if constexpr (WIDE ? (KVPF != 0) : (KVPF_F != 0)) {
        // this layer's K/V chunks that this block's attention items will read, pulled into L2 now (the cache does not
        // stay in L2 between steps); the chunk of the current position is completed in L2 by this phase's writes
        const int lo = Lp.sliding ? max(0, pos - 127) : 0;
        const int c0 = lo >> 6, nch = (pos >> 6) - c0 + 1;
        if (ATT_BLK && nch <= NW * (WIDE ? ATT_BLKW : ATT_BLKW_F) && B * NKV <= NB) {
          if (bid < B * NKV && tid < 2) {
            const uint4* base = reinterpret_cast<const uint4*>(tid ? Lp.vc : Lp.kc) + ((size_t)bid * p.maxctx * HD) / 8 + (size_t)c0 * 512;
            l2_prefetch(base, nch * 8192);
          }
        } else if (tid < 2 * NW) {
          const int item = bid + NB * (tid >> 1);
          if (item < B * NKV * nch) {
            const int unit = item / nch, ch = c0 + item % nch;
            const uint4* base = reinterpret_cast<const uint4*>((tid & 1) ? Lp.vc : Lp.kc) + ((size_t)unit * p.maxctx * HD) / 8 + (size_t)ch * 512;
            l2_prefetch(base, 8192);
          }
        }
      }
#endif
#if SPF
      // this layer's small parameters read right after later barriers: this block's qkv / o_proj bias slices, the
      // router bias and the attention sinks (they do not stay in L2 between steps either)
      if (tid >= 32 && tid < 36) {
        const int j = tid - 32;
        if (j == 0) { const int q0 = bid * RT_QKV / NB, q1 = (bid + 1) * RT_QKV / NB; l2_prefetch(Lp.bqkv + q0 * 16, (q1 - q0) * 64); }
        else if (j == 1) { const int o0 = bid * RT_O / NB, o1 = (bid + 1) * RT_O / NB; if (o1 > o0) l2_prefetch(Lp.bo + o0 * 16, (o1 - o0) * 64); }
        else if (j == 2) l2_prefetch(Lp.br, NE * 4);
        else l2_prefetch(Lp.sinks, NQH * 4);
      }
#if SPF_ROPE
      // the RoPE row of this position (a new line every step, read by A's epilogue)
      if (tid == 36) l2_prefetch(p.cosT + pos * 32, 128);
      else if (tid == 37) l2_prefetch(p.sinT + pos * 32, 128);
#endif
#endif
#ifdef TOPUP
      topup_tasks<0>(ttab, rg, pst[warp]);
#endif
      if (flat && layer > 0) {
        u16* stg = reinterpret_cast<u16*>(red);
        combine_flat(p, ms, B, stg);
        MARK(16);
        for (int b = 0; b < MAXB; b++) rows_in[b] = reinterpret_cast<const bf16*>(stg + b * HID);
        wait_nw(); norm_to_smem(rows_in, B, nwb, xs, ms->rinv, true);
      } else {
        for (int b = 0; b < MAXB; b++)
          rows_in[b] = (layer == 0) ? p.embed + (size_t)(b < B ? ms->tok[b] : 0) * HID : p.x + b * HID;
        wait_nw();
        MARK(28);
        if (!WIDE) norm_to_smem(rows_in, B, nwb, xs, ms->rinv);
        else if (NORM_TMA || NORM_TMA_A) norm_warp_tma(rows_in, B, nwb, xs, abar, apos);
        else norm_warp(rows_in, B, nwb, xs);
      }
      MARK(11);
      {
#if QXO
        const int r0 = QXI ? 2 * bid : bid * RT_QKV / NB, R = QXI ? 2 : (bid + 1) * RT_QKV / NB - r0;
#else
        const int r0 = bid * RT_QKV / NB, r1 = (bid + 1) * RT_QKV / NB, R = r1 - r0;
#endif
        consume_tasks<0>(ttab, XSH, rg, pst[warp], [&](int slot, const float* c) { store_task(red, slot, c); });
#ifndef PF_LATE
        pf_o(layer);
#endif
        __syncthreads();
#if QXO
        if (QXI && bid < 40 && tid < 128)
          reinterpret_cast<float4*>(p.ypart1 + (size_t)(bid * 4 + (tid >> 5)) * 128)[tid & 31] =
              reinterpret_cast<const float4*>(red + (R * NSEG_QKV + (tid >> 5)) * 128)[tid & 31];
#endif
        MARK(17);
        // epilogue: item = (r, i<8, b)
        for (int it = tid; it < R * 8 * B; it += NT) {
          int r = it / (8 * B), i = (it / B) % 8, b = it % B;
          float v1 = 0.f, v2 = 0.f;
          for (int s = 0; s < NSEG_QKV; s++) { v1 += red_at(red, r * NSEG_QKV + s, i, b); v2 += red_at(red, r * NSEG_QKV + s, i + 8, b); }
          int grt = r0 + r;
          v1 = bfr(v1 + Lp.bqkv[grt * 16 + i]);
          v2 = bfr(v2 + Lp.bqkv[grt * 16 + i + 8]);
          if (grt < 288) {     // q or k: rows i, i+8 = dims d, d+32 of one head
            int j = grt & 3, d = 8 * j + i;
            float cs = p.cosT[pos * 32 + d], sn = p.sinT[pos * 32 + d];
            float o1 = bfr(bfr(v1 * cs) - bfr(v2 * sn));
            float o2 = bfr(bfr(v2 * cs) + bfr(v1 * sn));
            if (grt < 256) {
              int h = grt >> 2;
              p.q[(size_t)b * 4096 + h * 64 + d] = __float2bfloat16_rn(o1);
              p.q[(size_t)b * 4096 + h * 64 + d + 32] = __float2bfloat16_rn(o2);
            } else {
              int kvh = (grt - 256) >> 2;
              u16* kp = reinterpret_cast<u16*>(Lp.kc) + ((size_t)b * NKV + kvh) * p.maxctx * HD;
              kp[kfo(pos, d)] = f2u16(o1); kp[kfo(pos, d + 32)] = f2u16(o2);
            }
          } else {
            int kvh = (grt - 288) >> 2, jj = (grt - 288) & 3;
            int d1 = 16 * jj + i, d2 = d1 + 8;
            u16* vp = reinterpret_cast<u16*>(Lp.vc) + ((size_t)b * NKV + kvh) * p.maxctx * HD;
            vp[vfo(pos, d1)] = f2u16(v1);
            vp[vfo(pos, d2)] = f2u16(v2);
          }
        }
#if XPF
        if (bid >= NB - 8 * RQ && tid < 16 * B) {
          const int rr = bid - (NB - 8 * RQ), rt = rr & 7, q = rr >> 3, base = R * NSEG_QKV;
          const int i = tid / B, b = tid % B;
          float v = 0.f;
          for (int w = 0; w < NW; w++) v += red_at(red, base + w, i, b);
          p.plog[(q * MAXB + b) * NE + rt * 16 + i] = v;
        }
#endif
      }
#ifdef PF_LATE
      pf_o(layer);
#endif
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
      MARK(2);
      // ---------------- phase B: attention. Work item = (sequence, kv head, 64-key chunk aligned to positions
      // 64c..64c+63), one warp each, spread over all blocks. A warp computes the chunk's scores for the 8 query heads
      // (mma: q rows x 8 keys), the chunk max m, l = sum exp(s - m) and acc = sum bf16(exp(s - m)) v (fp32), as
      // flash decoding does; the last warp of a (sequence, head) to finish merges the chunks with the sink:
      // out = sum_c e^(m_c - M) acc_c / (sum_c e^(m_c - M) l_c + e^(sink - M)), M = max(sink, m_c).
#ifndef FTMA
#define FTMA 1       // B >= 3: activation rows in fragment order, staged by the down phase with one bulk copy per row
#endif
#ifndef ATMA
#define ATMA 1       // attention output written in fragment order; phase C stages it with one bulk copy per row
#endif
#ifndef ATT_CL2
#define ATT_CL2 0    // launch in clusters of 2; attention units of 9..16 chunks on a block pair with a DSMEM merge
#endif
#ifndef V_ASYNC_FLAT
#define V_ASYNC_FLAT 0   // (~16 agent-h: 0 is -6 us at B=1 again) the B <= 2 instance loads V by cp.async with K (re-measured at ~9.7 agent-h: B=1 -10)
#endif
#if ATT_BLKW_F > 1 && V_ASYNC_FLAT
#error "ATT_BLKW_F > 1 puts the attention partials in xs, where V_ASYNC_FLAT stages V"
#endif
#ifndef V_ASYNC
#define V_ASYNC 0    // 1: V loads of an attention item as cp.async into shared memory with the K loads (B=1 +41, B=8 +55 us)
#endif
#ifndef ATT_HYB
#define ATT_HYB 0    // the one-block-per-(sequence, kv head) path is gone: nonzero values would skip attention
#endif
#if ATT_HYB
#error "ATT_HYB: the non-split attention path no longer exists (any nonzero value skips attention: wrong results)"
#endif
#if XPF
      if (warp == NW - 1 && bid >= NB - XPF_NB) xpf_issue(p, Lp, B, bid - (NB - XPF_NB));
#endif
      const bool split_att = !((ATT_HYB >= 1 && Lp.sliding) || (ATT_HYB >= 2 && pos < 128));
      if (split_att) {
        const int lo = Lp.sliding ? max(0, pos - 127) : 0;
        const int c0 = lo >> 6, c1 = pos >> 6, nch = c1 - c0 + 1;
        // merge of the chunk partials of one (sequence, kv head) by one warp (global partials of the split form, or
        // the shared-memory partials of ATT_BLK: the same arithmetic and order, so the same bits)
        auto att_merge = [&](const float* base, bool shm, int b, int kvh, float sink, const float* base1 = nullptr) {
              // lane (g, t): head g, dims 16t .. 16t+15. Chunk statistics: lane t holds chunks c = t (mod 4), all
              // loaded at once; the acc partials in batches of 4 chunks (16 loads in flight per lane)
              float mc[8], lc[8];
#pragma unroll
              for (int q = 0; q < 8; q++) {
                const int c = t + 4 * q;
                if (c < nch) { const float2 ml = shm ? *reinterpret_cast<const float2*>(((base1 && c >= NW) ? base1 + (size_t)(c - NW) * 8 * 68 : base + (size_t)c * 8 * 68) + 64) : __ldcg(reinterpret_cast<const float2*>(((base1 && c >= NW) ? base1 + (size_t)(c - NW) * 8 * 68 : base + (size_t)c * 8 * 68) + 64)); mc[q] = ml.x; lc[q] = ml.y; }
                else { mc[q] = -INFINITY; lc[q] = 0.f; }
              }
              float M = sink;
#pragma unroll
              for (int q = 0; q < 8; q++) M = fmaxf(M, mc[q]);
              M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 1));
              M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, 2));
              float wq[8], Lp_ = 0.f;
#pragma unroll
              for (int q = 0; q < 8; q++) { wq[q] = (t + 4 * q < nch) ? expf(mc[q] - M) : 0.f; Lp_ += wq[q] * lc[q]; }
              Lp_ += __shfl_xor_sync(0xffffffffu, Lp_, 1);
              Lp_ += __shfl_xor_sync(0xffffffffu, Lp_, 2);
              const float L = Lp_ + expf(sink - M);
              float o[16];
#pragma unroll
              for (int d = 0; d < 16; d++) o[d] = 0.f;
#pragma unroll
              for (int q = 0; q < 8; q++) {
                if (4 * q >= nch) break;
                float4 v[4][4];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                  const int c = 4 * q + u;
                  const float* pc = ((base1 && c >= NW) ? base1 + (size_t)(c - NW) * 8 * 68 : base + (size_t)c * 8 * 68) + 16 * t;
#pragma unroll
                  for (int d = 0; d < 4; d++) v[u][d] = (c < nch) ? (shm ? *reinterpret_cast<const float4*>(pc + 4 * d) : __ldcg(reinterpret_cast<const float4*>(pc + 4 * d))) : make_float4(0.f, 0.f, 0.f, 0.f);
                }
#pragma unroll
                for (int u = 0; u < 4; u++) {
                  // weight of chunk 4q + u lives in lane (g, u), slot q
                  const float w = __shfl_sync(0xffffffffu, wq[q], (lane & ~3) | u);
#pragma unroll
                  for (int d = 0; d < 4; d++) {
                    o[4 * d] += w * v[u][d].x; o[4 * d + 1] += w * v[u][d].y; o[4 * d + 2] += w * v[u][d].z; o[4 * d + 3] += w * v[u][d].w;
                  }
                }
              }
              const float iL = 1.f / L;
              if constexpr (WIDE ? ATMA : ATMA_FLAT) {
                u16* ox = reinterpret_cast<u16*>(p.attn) + (size_t)b * 4096;
#pragma unroll
                for (int d = 0; d < 16; d += 2) {
                  const int e = (kvh * 8 + g) * 64 + 16 * t + d;
                  *reinterpret_cast<u32*>(ox + xfo(e)) = (u32)f2u16(o[d] * iL) | ((u32)f2u16(o[d + 1] * iL) << 16);
                }
              } else {
                bf16* out = p.attn + (size_t)b * 4096 + (kvh * 8 + g) * 64 + 16 * t;
#pragma unroll
                for (int d = 0; d < 16; d++) out[d] = __float2bfloat16_rn(o[d] * iL);
              }
              if (BCOV) { __syncwarp(); if (lane == 0) sig_add_rel(p.adone + kvh, 1u); }
        };
        // ATT_BLK: when a unit has at most NW chunks, one block per (sequence, kv head), one chunk per warp, partials
        // in shared memory (red), merged after a block barrier (no global partials, no acq_rel hand-off)
        // (the B >= 3 instance: up to ATT_BLKW chunks per warp, partials in xs, which is free during attention there)
        constexpr int bmax = NW * (WIDE ? ATT_BLKW : ATT_BLKW_F);
        const bool blk = ATT_BLK && nch <= bmax && B * NKV <= NB;
        // ATT_CL2: units of NW+1 .. 2 NW chunks on the block pair (2u, 2u+1) of a cluster, one chunk per warp; the
        // partner's partials are read through distributed shared memory by rank 0's warp 0 (one merge, same bits)
        const bool cl2 = ATT_CL2 && !blk && nch <= 2 * NW && 2 * B * NKV <= NB;
        const int nitem = blk ? ((bid < B * NKV && warp < nch) ? (nch - warp + NW - 1) / NW : 0)
                        : cl2 ? ((bid < 2 * B * NKV && (bid & 1) * NW + warp < nch) ? 1 : 0) : B * NKV * nch;
        // partials of ATT_BLK: red (B <= 2, and B >= 3 with V_ASYNC: V then takes xs), else xs
        float* const apb = (WIDE ? !V_ASYNC : (ATT_BLKW_F > 1)) ? reinterpret_cast<float*>(xs) : red;
#if V_ASYNC && ATT_BLKW > 1
#error "V_ASYNC: the ATT_BLK partials go to red, which holds NW chunks only"
#endif
        for (int item = (blk || cl2) ? 0 : bid + NB * warp; item < nitem; item += (blk || cl2) ? 1 : NB * NW) {
          const int unit = blk ? bid : cl2 ? bid >> 1 : item / nch;
          const int ch = blk ? c0 + warp + item * NW : cl2 ? c0 + (bid & 1) * NW + warp : c0 + item % nch;
          const int b = unit >> 3, kvh = unit & 7;
#if QXO
          if (QXI && kvh >= 4 && ch == c1) qx_vpos(p.ypart1, Lp.bqkv, Lp.vc, p.maxctx, b, kvh, pos);
#endif
          MARK(20);
          // q fragments of the 8 heads (A operand rows g = head, rows g+8 zero)
          const u16* qg = reinterpret_cast<const u16*>(p.q) + (size_t)b * 4096 + (kvh * 8 + g) * 64 + 2 * t;
          u32 qa[4][2];
#pragma unroll
          for (int ks = 0; ks < 4; ks++) { qa[ks][0] = ldcg32(qg + 16 * ks); qa[ks][1] = ldcg32(qg + 16 * ks + 8); }
          // K (8 groups of 8 keys) and V (4 blocks of 16 keys) of the chunk in fragment order: all loads issued now
          const uint4* kb0 = reinterpret_cast<const uint4*>(Lp.kc + ((size_t)b * NKV + kvh) * p.maxctx * HD) + (size_t)ch * 8 * 64 + lane * 2;
          const uint4* vb0 = reinterpret_cast<const uint4*>(Lp.vc + ((size_t)b * NKV + kvh) * p.maxctx * HD) + (size_t)ch * 4 * 128 + lane * 4;
          float sc[8][2];
          u32 kf[8][8];
#pragma unroll
          for (int gr = 0; gr < 8; gr++) {
            const uint4 k0v = __ldcg(kb0 + gr * 64), k1v = __ldcg(kb0 + gr * 64 + 1);
            kf[gr][0] = k0v.x; kf[gr][1] = k0v.y; kf[gr][2] = k0v.z; kf[gr][3] = k0v.w;
            kf[gr][4] = k1v.x; kf[gr][5] = k1v.y; kf[gr][6] = k1v.z; kf[gr][7] = k1v.w;
          }
          uint4 vq[4][4];
          const u32 vsm = smaddr(xs) + warp * 8192 + lane * 16;
          if constexpr (WIDE ? V_ASYNC : V_ASYNC_FLAT) {
            // V of the chunk straight into this warp's 8 KB of xs (free during attention) while the scores run
#pragma unroll
            for (int kb = 0; kb < 4; kb++)
#pragma unroll
              for (int j = 0; j < 4; j++)
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(vsm + (kb * 4 + j) * 512), "l"(vb0 + kb * 128 + j) : "memory");
            asm volatile("cp.async.commit_group;" ::: "memory");
          }
#ifdef V_EARLY
#pragma unroll
          for (int kb = 0; kb < 4; kb++)
#pragma unroll
            for (int j = 0; j < 4; j++) vq[kb][j] = __ldcg(vb0 + kb * 128 + j);
#endif
          float m = -INFINITY;
#pragma unroll
          for (int gr = 0; gr < 8; gr++) {
            float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 4; ks++) mma16816(c, qa[ks][0], 0u, qa[ks][1], 0u, kf[gr][2 * ks], kf[gr][2 * ks + 1]);
            const int k0 = ch * 64 + 8 * gr + 2 * t;
            sc[gr][0] = (k0 >= lo && k0 <= pos) ? c[0] * 0.125f : -INFINITY;
            sc[gr][1] = (k0 + 1 >= lo && k0 + 1 <= pos) ? c[1] * 0.125f : -INFINITY;
            m = fmaxf(m, fmaxf(sc[gr][0], sc[gr][1]));
          }
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
          MARK(21);
          // e = exp(s - m); P (bf16) for the V mix comes straight from the score layout
          float l = 0.f;
          u32 pa[8];
#pragma unroll
          for (int gr = 0; gr < 8; gr++) {
            const float e0 = expf(sc[gr][0] - m), e1 = expf(sc[gr][1] - m);
            l += e0 + e1;
            pa[gr] = (u32)f2u16(e0) | ((u32)f2u16(e1) << 16);
          }
          l += __shfl_xor_sync(0xffffffffu, l, 1);
          l += __shfl_xor_sync(0xffffffffu, l, 2);
          if constexpr (WIDE ? V_ASYNC : V_ASYNC_FLAT) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll
            for (int kb = 0; kb < 4; kb++)
#pragma unroll
              for (int j = 0; j < 4; j++) vq[kb][j] = lds128(vsm + (kb * 4 + j) * 512);
          } else {
#ifndef V_EARLY
#pragma unroll
            for (int kb = 0; kb < 4; kb++)
#pragma unroll
              for (int j = 0; j < 4; j++) vq[kb][j] = __ldcg(vb0 + kb * 128 + j);
#endif
          }
          // V mix: 4 key blocks of 16 (groups 2kb, 2kb+1), 8 dim groups of 8
          float acc[8][4];
#pragma unroll
          for (int nb = 0; nb < 8; nb++) { acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f; }
#pragma unroll
          for (int kb = 0; kb < 4; kb++) {
            const u32 vf[16] = {vq[kb][0].x, vq[kb][0].y, vq[kb][0].z, vq[kb][0].w, vq[kb][1].x, vq[kb][1].y, vq[kb][1].z, vq[kb][1].w,
                                vq[kb][2].x, vq[kb][2].y, vq[kb][2].z, vq[kb][2].w, vq[kb][3].x, vq[kb][3].y, vq[kb][3].z, vq[kb][3].w};
#pragma unroll
            for (int nb = 0; nb < 8; nb++) mma16816(acc[nb], pa[2 * kb], 0u, pa[2 * kb + 1], 0u, vf[2 * nb], vf[2 * nb + 1]);
          }
          const float sink = Lp.sinks[kvh * 8 + g];
          if (nch == 1) {
            // single chunk: finish here
            const float M = fmaxf(m, sink);
            const float sc_ = expf(m - M), L = l * sc_ + expf(sink - M);
            const float f = sc_ / L;
            if constexpr (WIDE ? ATMA : ATMA_FLAT) {
              // in fragment order (xfo) so that phase C takes each row with one bulk copy
              u16* ox = reinterpret_cast<u16*>(p.attn) + (size_t)b * 4096;
#pragma unroll
              for (int nb = 0; nb < 8; nb++) {
                const int e = (kvh * 8 + g) * 64 + 8 * nb + 2 * t;
                *reinterpret_cast<u32*>(ox + xfo(e)) = (u32)f2u16(acc[nb][0] * f) | ((u32)f2u16(acc[nb][1] * f) << 16);
              }
            } else {
              bf16* out = p.attn + (size_t)b * 4096 + (kvh * 8 + g) * 64 + 2 * t;
#pragma unroll
              for (int nb = 0; nb < 8; nb++) {
                out[8 * nb] = __float2bfloat16_rn(acc[nb][0] * f);
                out[8 * nb + 1] = __float2bfloat16_rn(acc[nb][1] * f);
              }
            }
            if (BCOV) { __syncwarp(); if (lane == 0) sig_add_rel(p.adone + kvh, 1u); }
          } else {
            // partial: [unit][chunk slot][8 heads][64 + 2]
            float* pp = blk ? apb + ((ch - c0) * 8 + g) * 68 : cl2 ? apb + ((ch - c0 - (bid & 1) * NW) * 8 + g) * 68 : p.apart + (((size_t)unit * 32 + (ch - c0)) * 8 + g) * 68;
#pragma unroll
            for (int nb = 0; nb < 8; nb++) { pp[8 * nb + 2 * t] = acc[nb][0]; pp[8 * nb + 2 * t + 1] = acc[nb][1]; }
            if (t == 0) { pp[64] = m; pp[65] = l; }
            if (blk || cl2) continue;
            u32 last = 0;
#ifdef ATT_FENCE
            __threadfence();
            __syncwarp();
            MARK(22);
            if (lane == 0) last = (atomicAdd(p.acnt + unit, 1u) == (u32)(nch - 1));
#else
            // the warp's partial stores, then one acquire-release add by lane 0 (orders them before the count and,
            // for the last warp, the other warps' partials before its reads)
            __syncwarp();
            MARK(22);
            if (lane == 0) {
              u32 old;
              asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(p.acnt + unit), "r"(1u) : "memory");
              last = (old == (u32)(nch - 1));
            }
#endif
            last = __shfl_sync(0xffffffffu, last, 0);
            MARK(23);
            if (last) {
#ifdef ATT_FENCE
              __threadfence();
#endif
              if (lane == 0) p.acnt[unit] = 0;
              att_merge(p.apart + ((size_t)unit * 32 * 8 + g) * 68, false, b, kvh, sink);
              MARK(24);
            }
          }
        }
        if (blk && bid < B * NKV) {
          __syncthreads();
          if (nch > 1 && warp == 0) att_merge(apb + g * 68, true, bid >> 3, bid & 7, Lp.sinks[(bid & 7) * 8 + g]);
        }
#if ATT_CL2
        if (cl2 && bid < 2 * B * NKV) {
          cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
          cl.sync();
          if ((bid & 1) == 0 && warp == 0) {
            const int u = bid >> 1;
            const float* pb = cl.map_shared_rank(apb, 1);
            att_merge(apb + g * 68, true, u >> 3, u & 7, Lp.sinks[(u & 7) * 8 + g], pb + g * 68);
          }
          cl.sync();   // the partner keeps its partials until they are read
        }
#endif
      }
      if (!BCOV) { MARK(MKPROF_WAIT); grid_bar(bar, bar_target); }
      MARK(3);
      // ---------------- phase C: o_proj + residual
      {
#ifdef TOPUP
        topup_tasks<0>(ttab, rg, pst[warp]);
#endif
        if constexpr (!BCOV) {
        if constexpr (WIDE ? ATMA : ATMA_FLAT) {
        // the attention output rows (already in fragment order) by bulk copies; the generic writes of other blocks are
        // ordered before this async-proxy read by the grid barrier and the proxy fence
        if (tid == 0) {
          asm volatile("fence.proxy.async.global;" ::: "memory");
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbar_expect(abar, B * 8192);
          for (int b = 0; b < B; b++) tma_load(smaddr(xs + b * XSA), p.attn + (size_t)b * 4096, 8192, abar);
        }
        mbar_wait(abar, apos & 1); apos++;
        } else {
#ifndef CCOPY_OLD
        // 16-byte loads, 8 in flight per thread
        for (int i0 = 0; i0 < B * 512; i0 += 8 * NT) {
          uint4 v[8];
#pragma unroll
          for (int j = 0; j < 8; j++) { const int i = i0 + tid + NT * j; if (i < B * 512) v[j] = __ldcg(reinterpret_cast<const uint4*>(p.attn) + i); }
#pragma unroll
          for (int j = 0; j < 8; j++) { const int i = i0 + tid + NT * j; if (i < B * 512) { const int b = i >> 9, k = (i & 511) * 8; xfo_store8(xs + b * XSA, k, v[j]); } }
        }
#else
        for (int i = tid; i < B * 1024; i += NT) {
          int b = i >> 10, k = (i & 1023) * 4;
          uint2 v; asm volatile("ld.global.cg.v2.u32 {%0,%1},[%2];" : "=r"(v.x), "=r"(v.y) : "l"(p.attn + (size_t)b * 4096 + k));
          xfo_store4(xs + b * XSA, k, v.x, v.y);
        }
#endif
        }
        }
        __syncthreads();
        MARK(18);
#if OSPLIT
        const int r0 = bid, R = 1;
#else
        const int r0 = bid * RT_O / NB, r1 = (bid + 1) * RT_O / NB, R = r1 - r0;
#endif
#if BCOV && !(ATMA && ATMA_FLAT)
#error "BCOV copies the attention output in fragment order: needs ATMA and ATMA_FLAT"
#endif
#if BCOV
        // no barrier after attention: before each task, its warp waits for the kv head its K segment covers and
        // copies those 256 columns (fragment order, every sequence) from the attention output into xs
        consume_tasks<0>(ttab, XSA, rg, pst[warp], [&](int slot, const float* c) { store_task(red, slot, c); }, [&](int slot) {
          const int sg = slot % NSEG_O, col = sg * (KS_A / NSEG_O) * 16, kvh = col / 512;
          if (lane == 0) { const u32 tgt = (u32)B * lep; while (ld_acquire(p.adone + kvh) < tgt) { } }
          __syncwarp();
          for (int b = 0; b < B; b++)
            for (int c0 = lane * 8; c0 < (KS_A / NSEG_O) * 16; c0 += 256) {
              const uint4 v = __ldcg(reinterpret_cast<const uint4*>(p.attn + (size_t)b * 4096 + col + c0));
              *reinterpret_cast<uint4*>(xs + b * XSA + col + c0) = v;
            }
          __syncwarp();
        });
#else
#if CSPLIT && NSEG_O != 4
#error "CSPLIT assumes NSEG_O 4"
#endif
#if CSPLIT
        const int cs_role = ms->cs_role, cs_k = ms->cs_k;
        consume_tasks<0>(ttab, XSA, rg, pst[warp], [&](int slot, const float* c) {
          if (cs_role == 2 && slot < 2) {
            // partner: the shared segment's 16 x 8 partial to global, then one release add
            *reinterpret_cast<float4*>(p.cpart + ((size_t)cs_k * 2 + slot) * 128 + lane * 4) = make_float4(c[0], c[1], c[2], c[3]);
            __syncwarp();
            if (lane == 0) sig_add_rel(p.ccnt + cs_k, 1u);
          } else store_task(red, slot, c);
        });
#else
        consume_tasks<0>(ttab, XSA, rg, pst[warp], [&](int slot, const float* c) { store_task(red, slot, c); });
#endif
#endif
#ifndef PF_LATE
        // the next phase's weights are issued before this phase's epilogue (the ring is drained here)
        if (bid >= NB - 8 * RQ) pf_router(layer);
        issue_nw(Lp.ln2b);
#endif
        __syncthreads();
        MARK(19);
#if CSPLIT
        // owner: wait for the partner's two segments of the shared rowtile (normally long done)
        if (ms->cs_role == 1) {
          if (tid == 0) { const u32 tgt = 2u * lep; while (ld_acquire(p.ccnt + ms->cs_k) < tgt) { } }
          __syncthreads();
        }
        const int csr = ms->cs_role, csb = (csr == 2) ? 2 : 0;   // partner: own rowtile in slots 2..5
        const int RR = (csr == 1) ? 1 : R;
#else
        const int csr = 0, csb = 0, RR = R;
#endif
        for (int it = tid; it < (RR + (csr == 1)) * 16 * B; it += NT) {
          int r = it / (16 * B), i = (it / B) % 16, b = it % B;
          float v = 0.f;
          if (r < RR) { for (int s = 0; s < NSEG_O; s++) v += red_at(red, csb + r * NSEG_O + s, i, b); }
          else {
            // the shared rowtile: segments 0, 1 here (slots 4, 5), 2, 3 from the partner, in the usual order
            v += red_at(red, NSEG_O, i, b); v += red_at(red, NSEG_O + 1, i, b);
#if CSPLIT
            const float* cp = p.cpart + (size_t)ms->cs_k * 256 + ((i & 7) * 4 + (b >> 1)) * 4 + (i >> 3) * 2 + (b & 1);
            v += ldcgf(cp); v += ldcgf(cp + 128);
#endif
          }
          int row = (r < RR) ? (r0 + r) * 16 + i : ms->cs_x * 16 + i;
          float o = bfr(v + Lp.bo[row]);
          const bf16* xin = (layer == 0) ? p.embed + (size_t)ms->tok[b] * HID : p.x + b * HID;
          float xv = u16f(ldcg16(xin + row));
          p.xm[b * HID + row] = __float2bfloat16_rn(xv + o);
        }
#if OSPLIT
        if (bid < 4 * (RT_O - NB) && tid < 16 * B) {
          const int i = tid / B, b = tid % B;
          float v = 0.f;
          for (int s = 0; s < NSEG_O / 4; s++) v += red_at(red, NSEG_O + s, i, b);
          p.opart[((size_t)bid * MAXB + b) * 16 + i] = v;
        }
#endif
      }
#ifdef PF_LATE
      if (bid >= NB - 8 * RQ) pf_router(layer);
      issue_nw(Lp.ln2b);
#endif
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
      MARK(4);
      // ---------------- phase D: norm2 (all blocks), router (blocks NB-8 .. NB-1)
#if OSPLIT
      // the split o_proj rows (2432..2879): ((q0 + q1) + q2) + q3 of the quarter partials + bias, bf16, + residual;
      // every block builds them in shared memory for its own norm (patch, [b][448]); blocks 0..27 also write their
      // rowtile into xm (one writer per row; read again only after later barriers)
      u16* xpatch = reinterpret_cast<u16*>(red);
      for (int it = tid; it < B * (HID - SPLIT_K0) / 4; it += NT) {
        const int b = it / ((HID - SPLIT_K0) / 4), kk = (it % ((HID - SPLIT_K0) / 4)) * 4, e = kk >> 4, i = kk & 15;
        float4 q[4];
#pragma unroll
        for (int j = 0; j < 4; j++) q[j] = __ldcg(reinterpret_cast<const float4*>(p.opart + ((size_t)(4 * e + j) * MAXB + b) * 16 + i));
        const float4 bo = *reinterpret_cast<const float4*>(Lp.bo + SPLIT_K0 + kk);
        const bf16* xin = (layer == 0) ? p.embed + (size_t)ms->tok[b] * HID : p.x + b * HID;
        const uint2 xv = __ldcg(reinterpret_cast<const uint2*>(xin + SPLIT_K0 + kk));
        const float o0 = bfr(((q[0].x + q[1].x) + q[2].x) + q[3].x + bo.x), o1 = bfr(((q[0].y + q[1].y) + q[2].y) + q[3].y + bo.y);
        const float o2 = bfr(((q[0].z + q[1].z) + q[2].z) + q[3].z + bo.z), o3 = bfr(((q[0].w + q[1].w) + q[2].w) + q[3].w + bo.w);
        const u32 w01 = (u32)f2u16(__uint_as_float(xv.x << 16) + o0) | ((u32)f2u16(__uint_as_float(xv.x & 0xffff0000u) + o1) << 16);
        const u32 w23 = (u32)f2u16(__uint_as_float(xv.y << 16) + o2) | ((u32)f2u16(__uint_as_float(xv.y & 0xffff0000u) + o3) << 16);
        *reinterpret_cast<uint2*>(xpatch + b * (HID - SPLIT_K0) + kk) = make_uint2(w01, w23);
        if (e == bid) *reinterpret_cast<uint2*>(p.xm + b * HID + SPLIT_K0 + kk) = make_uint2(w01, w23);
      }
      __syncthreads();
#else
      const u16* xpatch = nullptr;
#endif
      for (int b = 0; b < MAXB; b++) rows_in[b] = p.xm + b * HID;
      wait_nw();
#if DQ
      // router blocks (B >= 3): only their K quarter normed before the router; the rest after their barrier arrival
      const bool dq = WIDE && bid >= NB - 8 * RQ;
      const int dq0 = dq ? ((bid - (NB - 8 * RQ)) >> 3) * (HID / RQ) : 0, dq1 = dq ? dq0 + HID / RQ : HID;
#endif
      if (!WIDE) norm_to_smem(rows_in, B, nwb, xs, ms->rinv, false, xpatch);
      else if ((NORM_TMA || NORM_TMA_D) && !OSPLIT) norm_warp_tma(rows_in, B, nwb, xs, abar, apos);
#if DQ
      else norm_warp(rows_in, B, nwb, xs, xpatch, dq0, dq1, false);
#else
      else norm_warp(rows_in, B, nwb, xs, xpatch);
#endif
      if (bid >= NB - 8 * RQ) {
        const int rr = bid - (NB - 8 * RQ), rt = rr & 7, q = rr >> 3;
        consume_tasks<0>(ttab, XSH, rg, pst[warp], [&](int slot, const float* c) { store_task(red, slot, c); });
        __syncthreads();
        if (tid < 16 * B) {
          int i = tid / B, b = tid % B;
          float v = 0.f;
          for (int w = 0; w < RTASK; w++) v += red_at(red, w, i, b);
          p.rlog[(q * MAXB + b) * NE + rt * 16 + i] = v;
        }
      }
      MARK(MKPROF_WAIT);
#if DQ
      if (WIDE && dq) {
        // arrive, the other three quarters of the norm while the barrier fills, then wait
        __syncthreads();
        bar_target += NB;
        if (tid == 0) { __threadfence(); atomicAdd(bar, 1u); }
        norm_warp(rows_in, B, nwb, xs, xpatch, dq0, dq1, true);
        if (tid == 0) { while (ld_acquire(bar) < bar_target) { } }
        __syncthreads();
      } else
#endif
      grid_bar(bar, bar_target);
      MARK(5);
      // ---------------- phase E: top-4 + union (every block), gate_up
      {
        __shared__ int sel[MAXB][4];
        __shared__ float selw[MAXB][4];
        __syncthreads();
        MARK(15);
        if (warp < B) {
          const int b = warp;
          float v[4]; int ix[4];
#pragma unroll
          for (int k = 0; k < 4; k++) {
            ix[k] = lane + 32 * k;
            float pq[RQ];
#pragma unroll
            for (int q = 0; q < RQ; q++) pq[q] = ldcgf(p.rlog + (q * MAXB + b) * NE + ix[k]);
            float vv = 0.f;
#pragma unroll
            for (int q = 0; q < RQ; q++) vv += pq[q];
            vv += Lp.br[ix[k]];
#if ROUTER_BF16
            vv = bfr(vv);
#endif
            v[k] = vv;
          }
          // top 4 by one warp-wide integer max per round: key = order-preserving 16-bit image of the bf16
          // logit, then 0xFFFF - expert (equal logits: the lower expert wins, as torch.topk on equal values here)
          u32 key[4];
#pragma unroll
          for (int k = 0; k < 4; k++) {
#if ROUTER_BF16
            const u32 hb = __float_as_uint(v[k]) >> 16;          // exact: v is bf16-rounded
#else
            const u32 hb = __float_as_uint(v[k]) >> 16;          // (fp32 logits: ordering by the top 16 bits only)
#endif
            const u32 ord = (hb & 0x8000u) ? (~hb & 0xffffu) : (hb | 0x8000u);
            key[k] = (ord << 16) | (0xffffu - (u32)ix[k]);
          }
          u32 tk[4];
#pragma unroll
          for (int r = 0; r < 4; r++) {
            const u32 lm = max(max(key[0], key[1]), max(key[2], key[3]));
            const u32 best = __reduce_max_sync(0xffffffffu, lm);
            tk[r] = best;
#pragma unroll
            for (int k = 0; k < 4; k++) if (key[k] == best) key[k] = 0;
          }
          if (lane < 4) {
            // lane r: expert and logit of rank r; softmax over the 4 (fp32)
            const u32 kr = tk[lane];
            const int e = 0xffff - (int)(kr & 0xffffu);
            const u32 ord = kr >> 16, hb = (ord & 0x8000u) ? (ord & 0x7fffu) : (~ord & 0xffffu);
            const float lv = __uint_as_float(hb << 16);
            const float l0 = __shfl_sync(0x0000000fu, lv, 0);
            const float ex = expf(lv - l0);
            float sum = ex;
            sum += __shfl_xor_sync(0x0000000fu, sum, 1);
            sum += __shfl_xor_sync(0x0000000fu, sum, 2);
            sel[b][lane] = e; selw[b][lane] = ex / sum;
          }
        }
        __syncthreads();
        MARK(14);
        // union of the chosen experts in ascending expert order, by warp 0: lane i < 4B holds (b = i / 4, r = i % 4)
        if (warp == 0) {
          const bool on = lane < 4 * B;
          const int bb = lane >> 2;
          const int e = on ? sel[bb][lane & 3] : 1000;
          const float wv = on ? selw[bb][lane & 3] : 0.f;
          u32 w4[4];
#pragma unroll
          for (int q = 0; q < 4; q++) w4[q] = __reduce_or_sync(0xffffffffu, (on && (e >> 5) == q) ? (1u << (e & 31)) : 0u);
          int slot = 0;
          if (on) {
            const int q = e >> 5;
#pragma unroll
            for (int qq = 0; qq < 4; qq++) if (qq < q) slot += __popc(w4[qq]);
            slot += __popc(w4[q] & ((1u << (e & 31)) - 1));
          }
          const u32 same = __match_any_sync(0xffffffffu, e);       // lanes holding the same expert
          u32 sm = 0;
          for (u32 m2 = same; m2; m2 &= m2 - 1) sm |= 1u << ((__ffs(m2) - 1) >> 2);
          // rank of this expert's slot among the column's 4 (all distinct)
          int rk = 0;
#pragma unroll
          for (int j = 0; j < 4; j++) { const int os = __shfl_sync(0xffffffffu, slot, (lane & ~3) + j); rk += os < slot; }
          if (on) {
            if ((int)(__ffs(same) - 1) == lane) { ms->sexp[slot] = e; ms->smask[slot] = sm; }
            ms->rw[bb][slot] = wv;
            ms->rsl[bb][rk] = slot;
            if ((lane & 3) == 0) ms->nrs[bb] = 4;
          }
          if (lane == 0) ms->U = __popc(w4[0]) + __popc(w4[1]) + __popc(w4[2]) + __popc(w4[3]);
        }
        __syncthreads();
        const int U = ms->U;
        if (tid < U) ms->eep[tid]++;
        MARK(12);
        const int N = U * RT_GU;
        const int u0 = (int)((long)bid * N / NB), u1 = (int)((long)(bid + 1) * N / NB);
#if EBPF
        if constexpr (!WIDE) {
          // the gate_up and down bias rows of this block's units into L2 now (their epilogues read them from global)
          if (tid < 8) {
            const int j = tid & 3;
            if (tid < 4) {
              const int s = u0 / RT_GU + j;
              if (u1 > u0 && s <= (u1 - 1) / RT_GU) {
                const int ra = max(u0, s * RT_GU) - s * RT_GU, rb = min(u1, (s + 1) * RT_GU) - s * RT_GU;
                l2_prefetch(Lp.gu_b + (size_t)ms->sexp[s] * 5760 + ra * 16, (rb - ra) * 64);
              }
            } else {
              const int NF = U * RT_DN, f0 = (int)((long)bid * NF / NB), f1 = (int)((long)(bid + 1) * NF / NB);
              const int s = f0 / RT_DN + j;
              if (f1 > f0 && s <= (f1 - 1) / RT_DN) {
                const int ra = max(f0, s * RT_DN) - s * RT_DN, rb = min(f1, (s + 1) * RT_DN) - s * RT_DN;
                l2_prefetch(Lp.dn_b + (size_t)ms->sexp[s] * HID + ra * 16, (rb - ra) * 64);
              }
            }
          }
        }
#endif
        if (B >= 3) {
          // one warp per unit (all 45 chunks; the 5 segments of 9 chunks are summed in order as in the split
          // form, so the bits are those of B <= 2), epilogue by the warp: no reduction rounds, the ring never drains
          if constexpr (ETAIL && WIDE) {
          // ETAIL: the n mod 8 remainder units as 5 one-stage segments each (summed below in the same order)
          const int en = u1 - u0, enw8 = ETAIL ? en / NW * NW : en;
          run_tasks<1>(enw8 + (en - enw8) * NSEG_GU, ttab, XSH, rg, pst[warp], [&](int task, TD& d) {
            int uu, k0, nn, sl0;
            if (task < enw8) { uu = u0 + task; k0 = 0; nn = KC; sl0 = task | (1 << 30); }
            else { const int q = task - enw8; uu = u0 + enw8 + q / NSEG_GU; k0 = (q % NSEG_GU) * (KC / NSEG_GU); nn = KC / NSEG_GU; sl0 = q | (1 << 29); }
            int slot = uu / RT_GU, rt = uu % RT_GU, e = ms->sexp[slot];
            d.w = Lp.gu_c + (((size_t)e * RT_GU + rt) * KC + k0) * 34;
            d.s = reinterpret_cast<const u32*>(Lp.gu_b + (size_t)e * 5760 + rt * 16);
            d.x = xs; d.k0 = k0; d.n = nn; d.slot = sl0; d.bm = (1u << B) - 1;
          }, [&](int task, const float* c, const float* bp) {
            if (ETAIL && task_tail(task)) { store_task(red, task & 0xffff, c); return; }
            const int uu = u0 + task, slot = uu / RT_GU, rt = uu % RT_GU, e = ms->sexp[slot];
            // lane (g,t): rows g, g+8 x columns 2t, 2t+1; gate = even rows, up = odd rows (partner lane g+1 = lane+4)
            float o[4];
#pragma unroll
            for (int j = 0; j < 4; j++) o[j] = __shfl_xor_sync(0xffffffffu, c[j], 4);
            if (!(g & 1)) {
#pragma unroll
              for (int h = 0; h < 2; h++) {          // h = 0: rows g / g+1, h = 1: rows g+8 / g+9
                const int i = (g >> 1) + 4 * h;        // pair index in the rowtile
                const float bg = bp[2 * h], bu = bp[2 * h + 1];   // rows 2i, 2i+1 = g (+8), g+1 (+8)
#pragma unroll
                for (int q = 0; q < 2; q++) {
                  const int b = 2 * t + q;
                  if (b < B && (ms->smask[slot] >> b & 1)) {
                    float gt = c[2 * h + q] + bg, up = o[2 * h + q] + bu;
                    gt = fminf(gt, 7.0f);
                    up = fmaxf(fminf(up, 7.0f), -7.0f);
                    float glu = gt * (1.0f / (1.0f + expf(-1.702f * gt)));
                    // B >= 3 (wide instance): rows in fragment order for the down phase's bulk copies (FTMA)
                    p.act[((size_t)slot * MAXB + b) * HID + ((FTMA && WIDE) ? xfo(rt * 8 + i) : rt * 8 + i)] = __float2bfloat16_rn((up + 1.0f) * glu);
                  }
                }
              }
            }
          });
          if (ETAIL && en > enw8) {
            __syncthreads();
            // the tail units: gate / up = 5 segments in order + bias (the bits of a whole unit), then the activation
            for (int it = tid; it < (en - enw8) * 8 * B; it += NT) {
              const int ul = it / (8 * B), i = (it / B) % 8, b = it % B;
              const int uu = u0 + enw8 + ul, slot = uu / RT_GU, rt = uu % RT_GU, e = ms->sexp[slot];
              if (!(ms->smask[slot] >> b & 1)) continue;
              float gt = 0.f, up = 0.f;
              for (int sg = 0; sg < NSEG_GU; sg++) { gt += red_at(red, ul * NSEG_GU + sg, 2 * i, b); up += red_at(red, ul * NSEG_GU + sg, 2 * i + 1, b); }
              gt += Lp.gu_b[(size_t)e * 5760 + rt * 16 + 2 * i];
              up += Lp.gu_b[(size_t)e * 5760 + rt * 16 + 2 * i + 1];
              gt = fminf(gt, 7.0f);
              up = fmaxf(fminf(up, 7.0f), -7.0f);
              const float glu = gt * (1.0f / (1.0f + expf(-1.702f * gt)));
              p.act[((size_t)slot * MAXB + b) * HID + ((FTMA && WIDE) ? xfo(rt * 8 + i) : rt * 8 + i)] = __float2bfloat16_rn((up + 1.0f) * glu);
            }
          }
          } else {
          run_tasks<1>(u1 - u0, ttab, XSH, rg, pst[warp], [&](int task, TD& d) {
            int uu = u0 + task, slot = uu / RT_GU, rt = uu % RT_GU, e = ms->sexp[slot];
            d.w = Lp.gu_c + (((size_t)e * RT_GU + rt) * KC) * 34;
            d.s = reinterpret_cast<const u32*>(Lp.gu_b + (size_t)e * 5760 + rt * 16);
            d.x = xs; d.k0 = 0; d.n = KC; d.slot = task | (1 << 30); d.bm = (1u << B) - 1;
          }, [&](int task, const float* c, const float* bp) {
            const int uu = u0 + task, slot = uu / RT_GU, rt = uu % RT_GU, e = ms->sexp[slot];
            // lane (g,t): rows g, g+8 x columns 2t, 2t+1; gate = even rows, up = odd rows (partner lane g+1 = lane+4)
            float o[4];
#pragma unroll
            for (int j = 0; j < 4; j++) o[j] = __shfl_xor_sync(0xffffffffu, c[j], 4);
            if (!(g & 1)) {
#pragma unroll
              for (int h = 0; h < 2; h++) {          // h = 0: rows g / g+1, h = 1: rows g+8 / g+9
                const int i = (g >> 1) + 4 * h;        // pair index in the rowtile
                const float bg = bp[2 * h], bu = bp[2 * h + 1];   // rows 2i, 2i+1 = g (+8), g+1 (+8)
#pragma unroll
                for (int q = 0; q < 2; q++) {
                  const int b = 2 * t + q;
                  if (b < B && (ms->smask[slot] >> b & 1)) {
                    float gt = c[2 * h + q] + bg, up = o[2 * h + q] + bu;
                    gt = fminf(gt, 7.0f);
                    up = fmaxf(fminf(up, 7.0f), -7.0f);
                    float glu = gt * (1.0f / (1.0f + expf(-1.702f * gt)));
                    // B >= 3 (wide instance): rows in fragment order for the down phase's bulk copies (FTMA)
                    p.act[((size_t)slot * MAXB + b) * HID + ((FTMA && WIDE) ? xfo(rt * 8 + i) : rt * 8 + i)] = __float2bfloat16_rn((up + 1.0f) * glu);
                  }
                }
              }
            }
          });
          }
          MARK(13);
#ifndef EF_BARRIER
          // one signal per block and slot once all of the block's units are written (release add after the block
          // barrier: the block's act stores are ordered before the count)
          SIG_FENCE();
          __syncthreads();
          if (tid == 0) {
            for (int sl = u0 / RT_GU; u1 > u0 && sl <= (u1 - 1) / RT_GU; sl++) {
              const int a0 = max(u0, sl * RT_GU), a1 = min(u1, (sl + 1) * RT_GU);
              sig_add(p.ecnt + sl, (u32)(a1 - a0));
            }
          }
#endif
        } else
        for (int ub = u0; ub < u1; ub += RED_SLOTS / NSEG_GU) {
          const int nu = min(RED_SLOTS / NSEG_GU, u1 - ub);
          const int ntask = nu * NSEG_GU;
          run_tasks<1>(ntask, ttab, XSH, rg, pst[warp], [&](int task, TD& d) {
            int uu = ub + task / NSEG_GU, s = task % NSEG_GU;
            int slot = uu / RT_GU, rt = uu % RT_GU, e = ms->sexp[slot];
            int k0 = s * KC / NSEG_GU, k1 = (s + 1) * KC / NSEG_GU;
            d.w = Lp.gu_c + (((size_t)e * RT_GU + rt) * KC + k0) * 34;
            d.s = reinterpret_cast<const u32*>(Lp.gu_b + (size_t)e * 5760 + rt * 16);
            d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
          }, [&](int slot, const float* c, const float*) { store_task(red, slot, c); });
          MARK(13);
          __syncthreads();
          if constexpr (EPF && !WIDE) {
            // last round: the down phase's row table and first weight stages now (they depend on the routing only),
            // so that their fetch overlaps this epilogue
            if (ub + nu >= u1) { fround_flat(); __syncthreads(); pf_down_flat(); }
          }
          // epilogue: item = (unit, pair i<8, b): gate = row 2i, up = row 2i+1
          for (int it = tid; it < nu * 8 * B; it += NT) {
            int ul = it / (8 * B), i = (it / B) % 8, b = it % B;
            int uu = ub + ul, slot = uu / RT_GU, rt = uu % RT_GU, e = ms->sexp[slot];
            if (!(ms->smask[slot] >> b & 1)) continue;
            float gt = 0.f, up = 0.f;
            for (int s = 0; s < NSEG_GU; s++) { gt += red_at(red, ul * NSEG_GU + s, 2 * i, b); up += red_at(red, ul * NSEG_GU + s, 2 * i + 1, b); }
            gt += Lp.gu_b[(size_t)e * 5760 + rt * 16 + 2 * i];
            up += Lp.gu_b[(size_t)e * 5760 + rt * 16 + 2 * i + 1];
            gt = fminf(gt, 7.0f);
            up = fmaxf(fminf(up, 7.0f), -7.0f);
            float glu = gt * (1.0f / (1.0f + expf(-1.702f * gt)));
            float o = (up + 1.0f) * glu;
            p.act[((size_t)slot * MAXB + b) * HID + ((FTMA_FLAT && !WIDE) ? xfo(rt * 8 + i) : rt * 8 + i)] = __float2bfloat16_rn(o);
          }
#ifndef EF_BARRIER
          SIG_FENCE();
          __syncthreads();
          if (tid == 0) {
            for (int sl = ub / RT_GU; sl <= (ub + nu - 1) / RT_GU; sl++) {
              const int a0 = max(ub, sl * RT_GU), a1 = min(ub + nu, (sl + 1) * RT_GU);
              sig_add(p.ecnt + sl, (u32)(a1 - a0));
            }
          }
#endif
          __syncthreads();
        }
        // the down phase's first round: row table and prefetch of its weights
        if (flat) {
          if (!(EPF && !WIDE) || u1 <= u0) {
            fround_flat();
            __syncthreads();
            pf_down_flat();
          }
        } else {
          fround_flat();
          __syncthreads();
          if (ms->fr_s1 > ms->fr_s0) { fround_wide(ms->fr_s0); __syncthreads(); pf_down_wide(); }
        }
      }
#ifdef EF_BARRIER
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
#endif
      MARK(6);
      // ---------------- phase F: down for owned rowtiles x all slots, weighted sum, residual
      if (flat) {
        const int s0 = ms->fr_s0, s1 = ms->fr_s1, np = ms->fr_np, u0 = ms->fr_u0, u1 = ms->fr_u1;
        u16* xa = xs;
        wait_slots(s0, s1);
        if constexpr (FTMA_FLAT && !WIDE) {
          if (tid < np) {
            const u32 bmask = (1u << B) - 1;
            int q = tid, sl = s0, b = 0;
            for (; sl < s1; sl++) { const int c = __popc(ms->smask[sl] & bmask); if (q < c) break; q -= c; }
            const u32 m = ms->smask[sl] & bmask;
            for (b = 0; b < 8; b++) if (m >> b & 1) { if (q == 0) break; q--; }
            ms->tmpi[tid] = sl * MAXB + b;
          }
          __syncthreads();
          if (tid == 0) {
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbar_expect(abar, np * HID * 2);
            for (int pr = 0; pr < np; pr++) tma_load(smaddr(xa + pr * XSH), p.act + (size_t)ms->tmpi[pr] * HID, HID * 2, abar);
          }
          mbar_wait(abar, apos & 1); apos++;
        } else load_act_rows(s0, s1, np, xa);
        __syncthreads();
        consume_tasks<1>(ttab, XSH, rg, pst[warp], [&](int slot, const float* c, const float*) { store_task(red, slot, c); });

        __syncthreads();
        for (int it = tid; it < (u1 - u0) * 16 * B; it += NT) {
          int ul = it / (16 * B), i = (it / B) % 16, b = it % B;
          int uu = u0 + ul, slot = uu / RT_DN, rt = uu % RT_DN, e = ms->sexp[slot];
          if (!(ms->smask[slot] >> b & 1)) continue;
          float y = 0.f;
#if FHALF
          // ((s0 + s1) + s2 + bias) + (s3 + s4): the order of the B >= 3 instance's two K halves (same bits)
          for (int sg = 0; sg < 3; sg++) y += red_at(red, ul * NSEG_DN + sg, i, b);
          y += Lp.dn_b[(size_t)e * HID + rt * 16 + i];
          float y1 = 0.f;
          for (int sg = 3; sg < NSEG_DN; sg++) y1 += red_at(red, ul * NSEG_DN + sg, i, b);
          y += y1;
#else
          for (int sg = 0; sg < NSEG_DN; sg++) y += red_at(red, ul * NSEG_DN + sg, i, b);
          y += Lp.dn_b[(size_t)e * HID + rt * 16 + i];
#endif
          p.ypart[((size_t)slot * MAXB + b) * HID + rt * 16 + i] = y;
        }
      } else {
        // wide path: rounds of slots whose activation rows fit in xs
        bool first = true;
#if FCOMB
        if (tid == 0) reinterpret_cast<int*>(red)[(RED_SLOTS - 1) * 128] = 0;   // the round's completed-rowtile list
#endif
        while (ms->fr_s1 > ms->fr_s0) {
          if (!first) { if (ms->fr_cur >= ms->fr_s1) break; fround_wide(ms->fr_cur); __syncthreads(); }
          const int rs = ms->fr_rs, re = ms->fr_cur, np = ms->fr_np;
          const u32 bmask = (1u << B) - 1;
          wait_slots(rs, re);
          MARK(25);
          if constexpr (FTMA && WIDE) {
            // the routed activation rows (written in fragment order by gate_up) by one bulk copy each
            if (tid < np) {
              int q = tid, sl = rs, b = 0;
              for (; sl < re; sl++) { const int c = __popc(ms->smask[sl] & bmask); if (q < c) break; q -= c; }
              const u32 m = ms->smask[sl] & bmask;
              for (b = 0; b < 8; b++) if (m >> b & 1) { if (q == 0) break; q--; }
              ms->tmpi[tid] = sl * MAXB + b;
            }
            __syncthreads();
            if (tid == 0) {
              asm volatile("fence.proxy.async.global;" ::: "memory");
              asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
              mbar_expect(abar, np * HID * 2);
              for (int pr = 0; pr < np; pr++) tma_load(smaddr(xs + pr * XSH), p.act + (size_t)ms->tmpi[pr] * HID, HID * 2, abar);
            }
            mbar_wait(abar, apos & 1); apos++;
          } else load_act_rows(rs, re, np, xs);
          __syncthreads();
          MARK(26);
          if (!first) pf_down_wide();
          first = false;
          const int ua = max(ms->fr_u0, rs * RT_DN);
          consume_tasks<1>(ttab, XSH, rg, pst[warp], [&](int task, const float* c, const float* bp) {
#if FTAIL
            if (task_tail(task)) { store_task(red, task & 0xffff, c); return; }
#endif
#if FHALF
            const int hf = (task & 1) ^ ((task >> 3) & 1);
            const int uu = ua + (task >> 1), sl = uu / RT_DN, rt = uu % RT_DN;
            float* yd = hf ? p.ypart1 : p.ypart;
#else
            const int hf = 0;
            const int uu = ua + task, sl = uu / RT_DN, rt = uu % RT_DN;
            float* yd = p.ypart;
#endif
            const u32 m = ms->smask[sl] & bmask;
            // lane (g,t): rows g, g+8 of the rowtile, columns 2t, 2t+1
#pragma unroll
            for (int h = 0; h < 2; h++) {
              const int row = rt * 16 + g + 8 * h;
              const float bias = bp[2 * h];
#pragma unroll
              for (int q = 0; q < 2; q++) {
                const int b = 2 * t + q;
                if (b < B && (m >> b & 1)) yd[((size_t)sl * MAXB + b) * HID + row] = hf ? c[2 * h + q] : c[2 * h + q] + bias;
              }
            }
          });
          MARK(27);
          __syncthreads();
#if FTAIL
          {
            // the tail units: 5 segments in order, + bias (the bits of a whole unit and of the B <= 2 epilogue)
            const int n = min(ms->fr_u1, re * RT_DN) - ua, nw8 = n / NW * NW;
            for (int it = tid; it < (n - nw8) * 16 * B; it += NT) {
              const int ul = it / (16 * B), i = (it / B) % 16, b = it % B;
              const int uu = ua + nw8 + ul, sl = uu / RT_DN, rt = uu % RT_DN, e = ms->sexp[sl];
              if (!(ms->smask[sl] >> b & 1)) continue;
              float y = 0.f;
              for (int sg = 0; sg < NSEG_DN; sg++) y += red_at(red, ul * NSEG_DN + sg, i, b);
              y += Lp.dn_b[(size_t)e * HID + rt * 16 + i];
              p.ypart[((size_t)sl * MAXB + b) * HID + rt * 16 + i] = y;
            }
            __syncthreads();
          }
#endif
#if FCOMB
          if constexpr (WIDE) {
            // count this round's units per rowtile; the block that completes a rowtile (all U slots) writes its 16
            // columns of x = bf16(xm + bf16(sum over the routed slots of w * y)) (the combine step's arithmetic)
            int* fl = reinterpret_cast<int*>(red) + (RED_SLOTS - 1) * 128;
            const int ue = min(ms->fr_u1, re * RT_DN);
            if (tid < ue - ua) {
              const int rt = (ua + tid) % RT_DN;
              u32 old;
              asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p.fcnt + rt) : "memory");
              if (old == (u32)ms->U - 1) { p.fcnt[rt] = 0; fl[1 + atomicAdd(fl, 1)] = rt; }
            }
            __syncthreads();
            const int nl4 = fl[0] * 4;
            for (int it = tid; it < nl4 * B; it += NT) {
              const int b = it / nl4, r = it % nl4, k = fl[1 + (r >> 2)] * 16 + (r & 3) * 4;
              const int nr = ms->nrs[b];
              uint4 y[4];
#pragma unroll
              for (int q = 0; q < 4; q++)
                y[q] = (q < nr) ? __ldcg(reinterpret_cast<const uint4*>(p.ypart + ((size_t)ms->rsl[b][q] * MAXB + b) * HID + k)) : make_uint4(0, 0, 0, 0);
              const uint2 xv = __ldcg(reinterpret_cast<const uint2*>(p.xm + b * HID + k));
              float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
              for (int q = 0; q < 4; q++) if (q < nr) {
                const float wv = ms->rw[b][ms->rsl[b][q]];
                acc[0] = fmaf(wv, __uint_as_float(y[q].x), acc[0]); acc[1] = fmaf(wv, __uint_as_float(y[q].y), acc[1]);
                acc[2] = fmaf(wv, __uint_as_float(y[q].z), acc[2]); acc[3] = fmaf(wv, __uint_as_float(y[q].w), acc[3]);
              }
              u16 o0 = f2u16(__uint_as_float(xv.x << 16) + bfr(acc[0]));
              u16 o1 = f2u16(__uint_as_float(xv.x & 0xffff0000u) + bfr(acc[1]));
              u16 o2 = f2u16(__uint_as_float(xv.y << 16) + bfr(acc[2]));
              u16 o3 = f2u16(__uint_as_float(xv.y & 0xffff0000u) + bfr(acc[3]));
              *reinterpret_cast<uint2*>(p.x + b * HID + k) = make_uint2(o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
            }
            __syncthreads();
            if (tid == 0) fl[0] = 0;
          }
#endif
        }
      }
      pf_next();
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
      if (!FCOMB && !flat) {
        // wide path: the layer output x = bf16(xm + bf16(sum over the routed slots of w * y)), rows owned by blocks
        MARK(8);
        const int g0 = bid * (HID / 4) / NB, g1 = (bid + 1) * (HID / 4) / NB;
        for (int it = tid; it < (g1 - g0) * B; it += NT) {
          const int b = it / (g1 - g0), k = (g0 + it % (g1 - g0)) * 4;
          const int nr = ms->nrs[b];
          uint4 y[4];
#pragma unroll
          for (int q = 0; q < 4; q++)
            y[q] = (q < nr) ? __ldcg(reinterpret_cast<const uint4*>(p.ypart + ((size_t)ms->rsl[b][q] * MAXB + b) * HID + k)) : make_uint4(0, 0, 0, 0);
#if FHALF
          uint4 y1[4];
#pragma unroll
          for (int q = 0; q < 4; q++)
            y1[q] = (q < nr) ? __ldcg(reinterpret_cast<const uint4*>(p.ypart1 + ((size_t)ms->rsl[b][q] * MAXB + b) * HID + k)) : make_uint4(0, 0, 0, 0);
#pragma unroll
          for (int q = 0; q < 4; q++) {
            y[q].x = __float_as_uint(__uint_as_float(y[q].x) + __uint_as_float(y1[q].x));
            y[q].y = __float_as_uint(__uint_as_float(y[q].y) + __uint_as_float(y1[q].y));
            y[q].z = __float_as_uint(__uint_as_float(y[q].z) + __uint_as_float(y1[q].z));
            y[q].w = __float_as_uint(__uint_as_float(y[q].w) + __uint_as_float(y1[q].w));
          }
#endif
          const uint2 xv = __ldcg(reinterpret_cast<const uint2*>(p.xm + b * HID + k));
          float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int q = 0; q < 4; q++) if (q < nr) {
            const float wv = ms->rw[b][ms->rsl[b][q]];
            acc[0] = fmaf(wv, __uint_as_float(y[q].x), acc[0]); acc[1] = fmaf(wv, __uint_as_float(y[q].y), acc[1]);
            acc[2] = fmaf(wv, __uint_as_float(y[q].z), acc[2]); acc[3] = fmaf(wv, __uint_as_float(y[q].w), acc[3]);
          }
          u16 o0 = f2u16(__uint_as_float(xv.x << 16) + bfr(acc[0]));
          u16 o1 = f2u16(__uint_as_float(xv.x & 0xffff0000u) + bfr(acc[1]));
          u16 o2 = f2u16(__uint_as_float(xv.y << 16) + bfr(acc[2]));
          u16 o3 = f2u16(__uint_as_float(xv.y & 0xffff0000u) + bfr(acc[3]));
          *reinterpret_cast<uint2*>(p.x + b * HID + k) = make_uint2(o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
        }
        MARK(MKPROF_WAIT);
        grid_bar(bar, bar_target);
      }
    }  // layers
    // ---------------- head
    const bool need_head = st >= P - 1;
    if (need_head) {
      MARK(7);
      if (flat) {
        u16* stg = reinterpret_cast<u16*>(red);
        combine_flat(p, ms, B, stg);
        for (int b = 0; b < MAXB; b++) rows_in[b] = reinterpret_cast<const bf16*>(stg + b * HID);
        wait_nw(); norm_to_smem(rows_in, B, nwb, xs, ms->rinv, true);
      } else {
        for (int b = 0; b < MAXB; b++) rows_in[b] = p.x + b * HID;
        wait_nw();
        if (!(HSK && HSK_WIDE && a.mode != 0)) { if (!WIDE) norm_to_smem(rows_in, B, nwb, xs, ms->rinv); else norm_warp(rows_in, B, nwb, xs); }
      }
      const int r0 = (int)((long)bid * RT_HEAD / NB), r1 = (int)((long)(bid + 1) * RT_HEAD / NB);
      float bv = -INFINITY; int bi = 0x7fffffff;   // this lane's best for column (2t or 2t+1) — tracked per lane below
      float bv2 = -INFINITY; int bi2 = 0x7fffffff;
      (void)r1;
      auto head_fin = [&](int rt, const float* c) {
        int row0 = rt * 16 + g, row1 = row0 + 8;
        int b0 = 2 * t, b1 = 2 * t + 1;
        if (a.mode == 0) {
          if (b0 < B) { a.logits[(size_t)b0 * VOCAB + row0] = c[0]; a.logits[(size_t)b0 * VOCAB + row1] = c[2]; }
          if (b1 < B) { a.logits[(size_t)b1 * VOCAB + row0] = c[1]; a.logits[(size_t)b1 * VOCAB + row1] = c[3]; }
        } else {
          if (c[0] > bv || (c[0] == bv && row0 < bi)) { bv = c[0]; bi = row0; }
          if (c[2] > bv || (c[2] == bv && row1 < bi)) { bv = c[2]; bi = row1; }
          if (c[1] > bv2 || (c[1] == bv2 && row0 < bi2)) { bv2 = c[1]; bi2 = row0; }
          if (c[3] > bv2 || (c[3] == bv2 && row1 < bi2)) { bv2 = c[3]; bi2 = row1; }
        }
      };
#if HSK
      if ((flat || HSK_WIDE) && a.mode != 0) {
        HskOut o;
        if constexpr (flat) o = hsk_head_flat(p.head, p.hsk, p.hsl, p.hdbg, p.x, nwb, ms, xs, red, ttab, rg, pst, B, bar, bar_target, r0, r1 HSK_PV);
        else if constexpr (HSK_WIDE_OL) o = hsk_head_wide(p.head, p.hsk, p.hsl, p.hdbg, p.x, nwb, ms, xs, red, ttab, rg, pst, B, bar, bar_target, r0, r1 HSK_PV);
        else o = hsk_head<true>(p.head, p.hsk, p.hsl, p.hdbg, p.x, nwb, ms, xs, red, ttab, rg, pst, B, bar, bar_target, r0, r1 HSK_PV);
        rg.pos = o.pos; bar_target = o.target; bv = o.bv; bi = o.bi; bv2 = o.bv2; bi2 = o.bi2;
#ifdef MKPROF
        pn = o.pn;
#endif
      } else
#endif
#if HEAD_OL
      if constexpr (WIDE) {
        const HskOut o = head_exact_ol(ttab, rg, pst, B, a.logits, a.mode, bar_target);
        rg.pos = o.pos; bv = o.bv; bi = o.bi; bv2 = o.bv2; bi2 = o.bi2;
      } else
#endif
#if HTAIL
      consume_tasks<0>(ttab, XSH, rg, pst[warp], [&](int sl, const float* c) {
        if (task_tail(sl)) store_task(red, sl & 0xffff, c); else head_fin(sl, c);
      });
      {
        // the tail rowtiles: warp w sums tail rowtile w's 4 segments in order and finishes it like a whole rowtile
        __syncthreads();
        const int n = r1 - r0, nw8 = n / NW * NW;
        if (warp < n - nw8) {
          float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int sg = 0; sg < HTS; sg++) {
            const float4 v = *reinterpret_cast<const float4*>(red + (warp * HTS + sg) * 128 + lane * 4);
            c[0] += v.x; c[1] += v.y; c[2] += v.z; c[3] += v.w;
          }
          head_fin(r0 + nw8 + warp, c);
        }
      }
#else
      consume_tasks<0>(ttab, XSH, rg, pst[warp], head_fin);
#endif
      if (a.mode != 0) {
        // reduce over g (lanes with the same t), then over warps
        #pragma unroll
        for (int o = 4; o < 32; o <<= 1) {
          float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
          if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
          ov = __shfl_xor_sync(0xffffffffu, bv2, o); oi = __shfl_xor_sync(0xffffffffu, bi2, o);
          if (ov > bv2 || (ov == bv2 && oi < bi2)) { bv2 = ov; bi2 = oi; }
        }
        if (g == 0) {
          ms->tmpf[warp * 8 + 2 * t] = bv; ms->tmpi[warp * 8 + 2 * t] = bi;
          ms->tmpf[warp * 8 + 2 * t + 1] = bv2; ms->tmpi[warp * 8 + 2 * t + 1] = bi2;
        }
        __syncthreads();
        if (tid < B) {
          float v = -INFINITY; int ix = 0x7fffffff;
          for (int w = 0; w < NW; w++) {
            float ov = ms->tmpf[w * 8 + tid]; int oi = ms->tmpi[w * 8 + tid];
            if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
          }
          p.amv[bid * MAXB + tid] = v; p.ami[bid * MAXB + tid] = ix;
        }
      }
#if HSK && PFDLY
      // blocks without head candidates hold back the next step's first weight stages for a while, so that their
      // HBM traffic does not slow down the candidate blocks' rescoring (the step barrier waits for those anyway)
      if (a.mode != 0 && st + 1 < total && reinterpret_cast<const int*>(reinterpret_cast<const unsigned char*>(red) + HSK_ST)[42] == 0)
        __nanosleep(PFDLY);
#endif
      if (st + 1 < total) { pf_qkv(0, pos + 1); issue_nw(p.L[0].ln1b); }
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
    }
  }  // steps
  // the last token chosen: reduce once more and record
  if (a.mode != 0) {
    if (tid < B) {
      float bv = -INFINITY; int bi = 0x7fffffff;
      for (int k = 0; k < NB; k++) {
        float v = ldcgf(p.amv + k * MAXB + tid); int ix = ldcgi(p.ami + k * MAXB + tid);
        if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
      }
      if (bid == 0) {
        if (a.mode == 1) a.out_tok[tid * a.nsteps + a.nsteps - 1] = bi;
        else a.host_tok[tid * a.nsteps + a.nsteps - 1] = bi;
      }
    }
    __syncthreads();
    if (a.mode == 2 && bid == 0 && tid == 0) { __threadfence_system(); *a.host_prog = a.nsteps; }
  }
  // reset the barrier counter for the next launch: the last block out
  if (tid == 0) {
    __threadfence();
    u32 v = atomicAdd(bar + 1, 1u);
    if (v == NB - 1) { bar[0] = 0; bar[1] = 0; for (int k = 0; k < 32; k++) p.ecnt[k] = 0; for (int k = 0; k < 8; k++) p.adone[k] = 0; for (int k = 0; k < 32; k++) p.ccnt[k] = 0;
#if HSK && HSK_RQ
      reinterpret_cast<u32*>(p.hsl + NB * MAXB + MAXB * HID / 2)[0] = 0;
#endif
      __threadfence(); }
  }
}

// ------------------------------------------------------------------------------------------------ host side
struct Engine {
  Params p;
  int* host_tok; int* host_prog; int* dev_host_tok; int* dev_host_prog;
  int ntok_cap;
  cudaStream_t stream;
};

extern "C" void* mk_create(const void* params) {
  Engine* e = new Engine;
  memcpy(&e->p, params, sizeof(Params));
  if (cudaFuncSetAttribute(mk_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL) != cudaSuccess) return nullptr;
  if (cudaFuncSetAttribute(mk_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL) != cudaSuccess) return nullptr;
  e->ntok_cap = MAXB * 8192;
  cudaHostAlloc(&e->host_tok, e->ntok_cap * sizeof(int), cudaHostAllocMapped);
  cudaHostAlloc(&e->host_prog, 64, cudaHostAllocMapped);
  cudaHostGetDevicePointer(&e->dev_host_tok, e->host_tok, 0);
  cudaHostGetDevicePointer(&e->dev_host_prog, e->host_prog, 0);
  memset((void*)e->host_prog, 0, 64);
  e->stream = 0;
  return e;
}
extern "C" int mk_params_size() { return (int)sizeof(Params); }
extern "C" int mk_hsk() { return HSK; }
extern "C" int mk_layer_size() { return (int)sizeof(LayerP); }

extern "C" int mk_run(void* h, const void* tok, int tok_stride, void* logits, void* out_tok, int B, int pos0, int nprompt, int nsteps, int mode, void* stream) {
  Engine* e = (Engine*)h;
  RunArgs a;
  a.tok = (const long long*)tok; a.tok_stride = tok_stride; a.logits = (float*)logits; a.out_tok = (long long*)out_tok;
  a.host_tok = e->dev_host_tok; a.host_prog = e->dev_host_prog; a.B = B; a.pos0 = pos0; a.nprompt = nprompt; a.nsteps = nsteps; a.mode = mode;
  if (mode == 2) { *(volatile int*)e->host_prog = 0; if (B * nsteps > e->ntok_cap) return -3; }
  void* args[] = {(void*)&e->p, (void*)&a};
  void* fn = (B <= FLAT_MAXB) ? (void*)mk_kernel<false> : (void*)mk_kernel<true>;
#if ATT_CL2
  // cooperative launch in clusters of 2 blocks (ATT_CL2: attention units of 9..16 chunks on a block pair)
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NB); cfg.blockDim = dim3(NT); cfg.dynamicSmemBytes = SM_TOTAL; cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeCooperative; at[0].val.cooperative = 1;
  at[1].id = cudaLaunchAttributeClusterDimension; at[1].val.clusterDim.x = 2; at[1].val.clusterDim.y = 1; at[1].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 2;
  cudaError_t err = cudaLaunchKernelExC(&cfg, fn, args);
#else
  cudaError_t err = cudaLaunchCooperativeKernel(fn, dim3(NB), dim3(NT), args, SM_TOTAL, (cudaStream_t)stream);
#endif
  if (err != cudaSuccess) { fprintf(stderr, "mk_run: %s\n", cudaGetErrorString(err)); return -1; }
  return 0;
}
extern "C" int mk_flag(void* h) { return ((volatile int*)((Engine*)h)->host_prog)[1]; }
extern "C" void mk_clear_flag(void* h) { ((volatile int*)((Engine*)h)->host_prog)[1] = 0; }
extern "C" int mk_poll(void* h, int* buf, int B, int nsteps) {
  Engine* e = (Engine*)h;
  int done = *(volatile int*)e->host_prog;
  if (done > 0) memcpy(buf, e->host_tok, sizeof(int) * B * nsteps);
  return done;
}

// study helper: how many clusters of C blocks of the B >= 3 instance can be co-resident (cooperative + cluster launch)
extern "C" int mk_cluster_check(int C) {
  cudaFuncSetAttribute(mk_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL);
  cudaFuncSetAttribute(mk_kernel<true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NB); cfg.blockDim = dim3(NT); cfg.dynamicSmemBytes = SM_TOTAL;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  int n = -1;
  cudaError_t e = cudaOccupancyMaxActiveClusters(&n, (void*)mk_kernel<true>, &cfg);
  return e == cudaSuccess ? n : -(int)e;
}
