// gpt-oss-20b decode megakernel for one H100 (sm_90a): one persistent cooperative kernel per step / generate call /
// served cohort. 132 blocks (one per SM). Derived from the s12-oss120-b200-a1 engine (s0005), ported to sm_90.
//
// Every matrix-vector product is a chain of mma.m16n8k16 (bf16 x bf16 -> f32); the 8 columns of the B operand are
// the sequences of the batch. Weights are re-laid at load time (a permutation of the stored bytes, no value change):
//  - BF16 matrices: 512-byte tiles [rowtile][kstep][32 lanes][16 B] in A-fragment order.
//  - MXFP4 experts: per (rowtile, 64-column chunk) 512 B of codes [32 lanes][16 B] (4 mma A fragments of 4 bytes,
//    each byte = 2 e2m1 codes = one bf16x2 register) and 32 B of E8M0 scales [8 g][4 B].
//    sm_90 decode: code c -> bf16 bits (sign<<15 | mag<<6) = e2m1(c) * 2^-126 exactly (subnormal for e=0; the
//    tensor core takes subnormal bf16, checked in bench/subn.cu). The B operand (MoE inputs) is stored * 2^63
//    (exact), and the partial sum of each 32-column group is multiplied by 2^(S-64) in float32 (exact).
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
#ifndef SCS
#define SCS 2048   // attention score row stride (floats): positions per sequence the scratch holds
#endif
#define NL 24
#define NQH 64
#define NKV 8
#define HD 64
#define QKVR 5120
#define NE 32
#define RB (NE / 16)       // router rowtiles
#define RKB 45             // router K slices (4 ksteps each), one per block 0..44; partials summed by every block
#define VOCAB 201088
#define NB 132
#ifndef NT
#define NT 384
#endif
#define TTN 16             // task-table entries per warp
#define NW (NT / 32)
#define MAXB 8
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
#define SKS 90             // sketch: k32-steps per tile
#ifndef SK5
#define SK5 0              // track B sketch: 5-bit codes with per-(row, k32-step) u8 scales (api.pack_sketch5)
#endif
#ifndef SKQP
#define SKQP 5             // SK5: k32-step pairs per load batch (divides 45)
#endif
#if SK5
#define SKT ((SKS / 2) * 672 + 256)   // per k32-step pair: 5 code words + 1 scale word per lane (168 words)
#define SKHDR ((SKS / 2) * 672)
#else
#define SKT (SKS * 384 + 256)   // 6-bit codes: per k32-step 3 words x 32 lanes
#define SKHDR (SKS * 384)
#endif
#ifndef SKQ
#define SKQ 15
#endif
static_assert(90 % SKQ == 0, "SKQ must divide the 90 k32-steps of a sketch tile");
//             // sketch loads in flight per lane (e0182: 15 -10 us vs 9)
#define DSTR (HID + 16)    // digit row stride (bytes): conflict-free B fragments
#ifndef NSEG_QKV
#define NSEG_QKV 4    // 45 k-steps = 5 full ring stages per segment
#endif
#ifndef NSEG_O
#define NSEG_O 6      // 42-43 k-steps = 5 ring stages per segment
#endif
#define NSEG_GU 5
#define NSEG_DN 5
#ifndef DSPLIT
#define DSPLIT 1
#endif
//                    down units: K split in halves (chunks [0,23) and [23,45)), summed in the combine
#ifndef DA
#define DA 8
#endif
#ifndef DB
#define DB 4
#endif
#ifndef RFOLD
#define RFOLD 0           // track B: router weights folded with ln2, router partials in phase C (no phase D)
#endif
#ifndef NSS
#define NSS 0             // track B (needs RFOLD): norms take the sum of squares from partials of the producing phase
#endif
#define NORM_SS(r, B, w, xs, sc, ssp, n, st) norm_to_smem(r, B, w, xs, nullptr, false, sc, ssp, n, st)
#ifndef KVPF
#define KVPF 1            // L2 prefetch of the attention window's K/V at the start of phase A
#endif
#ifndef KVPF_MAXB
#define KVPF_MAXB 4
#endif
#ifndef EPL
#define EPL 1             // qkv / o_proj epilogue inputs loaded between the stream and the block barrier
#endif
#ifndef EPLA
#define EPLA EPL
#endif
#ifndef EPLC
#define EPLC 0            // (o_proj part: B=1 -6, B=8 +10, e0251)
#endif
#ifndef MULDEC
#define MULDEC 1          // e2m1 decode by masked multiply (13 instead of ~19 instructions per word, same bits)
#endif
#ifndef NSS4
#define NSS4 1            // NSS norms: each row over 4 warps
#endif
#ifndef RFPF
#define RFPF 1            // L1 prefetch of the folded router rows at phase C start
#endif
#ifndef DBPF
#define DBPF 1            // L2 prefetch of the down combine's bias rows at gate_up start
#endif
#ifndef SVPF
#define SVPF 1            // L2 prefetch of the next layer's small vectors during the down phase
#endif
#ifndef NSS4W
#define NSS4W 1           // NSS4 norm: weights loaded with the row
#endif
#ifndef FUSE
#define FUSE 1            // MXFP4 items stored as 512 B codes + 32 B scales (one bulk copy per ring stage)
#endif
#ifndef SKEF
#define SKEF 1            // sketch pass-1 loads with the L2 evict_first policy
#endif
#ifndef PFOE
#define PFOE 1            // o_proj ring prefetch issued right after each warp's qkv stream
#endif
#ifndef GURR
#define GURR 1            // gate_up units round-robin over blocks
#endif
#ifndef BFIX
#define BFIX 0
#endif
#ifndef DORD
#define DORD 0            // down units half-major inside a slot (h, rt) instead of (rt, h)
#endif
#ifndef P2TMA
#define P2TMA 0           // head recompute: candidate tile by one bulk copy
#endif
#ifndef P2CH
#define P2CH 1            // P2TMA: chunks (mbarriers) per tile; 8 P2CH + 4 <= 64 (the pad after the ring barriers)
#endif
static_assert(8 * P2CH + 4 <= 64 && 180 % P2CH == 0, "P2CH");
#ifndef DPK
#define DPK 0             // B = 1 build: hi / lo x digits in columns 0 / 1 of one sketch mma
#endif
#if DPK && BFIX != 1
#error DPK needs BFIX=1
#endif
#ifndef SKPF0
#define SKPF0 0           // track B: L2 prefetch of the first sketch load batches at the end of the last layer
#endif
#ifndef SKDB
#define SKDB 0            // SK5 + DPK: double-buffered sketch load batches
#endif
#if SKDB && !DPK
#error SKDB needs DPK
#endif
#ifndef TOPR
#define TOPR 0            // routing top-4 by rank counting (same experts, same order)
#endif
#ifndef RFPRE
#define RFPRE 0
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
  bf16* kc; bf16* vc;
  int sliding;
};

struct Params {
  LayerP L[NL];
  const bf16* embed; const uint4* head; const float* lnf;
  const float* cosT; const float* sinT;     // [maxctx][32] (bf16 values as float)
  bf16 *x, *xm, *q, *attn, *act;           // act: [32 slots][MAXB][HID]
  float* rlog;                              // [MAXB][NE]
  float* ypart;                             // [32 slots][MAXB][HID] down outputs (+bias, unweighted), flat path (B <= 2)
  float* amv; int* ami;                     // [NB][MAXB]
  u32* bar;                                 // [0] barrier counter, [1] exit counter
  int maxctx;
  unsigned long long* prof; unsigned prof_cap;   // mkprof dump (only with -DMKPROF)
  const unsigned char* skel;                // -DSKEL: every weight load reads a 512 KB window here (results wrong)
  u32* dcnt;                                // [RT_DN] down units finished per rowtile (the last one combines; reset by it)
  const unsigned char* sk;                  // track B: certified int8 head sketch [RT_HEAD][SKT] (nullptr: exact head)
  float* gth;                               // [NB][MAXB] best sketch lower bound per block and column
  int* clist;                               // [2][RT_HEAD] candidate tiles of the sketch head (per head parity)
};

struct RunArgs {
  const long long* tok; int tok_stride;     // mode 0/1: tok[b]; mode 2: tok[b*tok_stride + t] (prompt ids)
  float* logits;                            // mode 0: [B][VOCAB]
  long long* out_tok;                       // mode 1: [B][nsteps]
  int* host_tok; volatile int* host_prog;   // mode 2: host-mapped [B][nsteps] and progress
  volatile int* err;                        // host-mapped: set when an input token id is out of range
  int B, pos0, nprompt, nsteps, mode;
  unsigned hstep0;                          // greedy heads run before this launch (parity of the sketch candidate list)
};

// ------------------------------------------------------------------------------------------------ primitives
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma8(int* c, const uint4& a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
// 4 six-bit codes (bits 0..23) -> 4 int8 bytes holding 4 q (two's complement wraps exactly)
// SK5: register r (0..7) of a k32-step pair -> 4 int8 bytes 8q: low 4 bits from nibble words w[r/2] (low / high
// nibbles), bit 4 from word w[4] bit 8j + r
__device__ __forceinline__ u32 dec5(const u32* w, int r) {
  const u32 lo = (r & 1) ? ((w[r >> 1] >> 1) & 0x78787878u) : ((w[r >> 1] << 3) & 0x78787878u);
  return lo | ((w[4] << (7 - r)) & 0x80808080u);
}
__device__ __forceinline__ u32 up6(u32 x) { return ((x << 2) & 0xFCu) | ((x << 4) & 0xFC00u) | ((x << 6) & 0xFC0000u) | ((x << 8) & 0xFC000000u); }
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
// the same with an L2 cache policy (evict_first for streams that are read once)
__device__ __forceinline__ u32 ldw32h(const void* p, unsigned long long pol) {
  u32 r; asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0,[%1], %2;" : "=r"(r) : "l"(p), "l"(pol)); return r;
}
// data written by other blocks during this launch: bypass L1
__device__ __forceinline__ u32 ldcg32(const void* p) { u32 r; asm volatile("ld.global.cg.u32 %0,[%1];" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ float ldcgf(const float* p) { float r; asm volatile("ld.global.cg.f32 %0,[%1];" : "=f"(r) : "l"(p)); return r; }
__device__ __forceinline__ int ldcgi(const int* p) { int r; asm volatile("ld.global.cg.s32 %0,[%1];" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ u16 ldcg16(const void* p) { u16 r; asm volatile("ld.global.cg.u16 %0,[%1];" : "=h"(r) : "l"(p)); return r; }
__device__ __forceinline__ uint4 ldcg128(const void* p) {
  uint4 r; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r;
}
__device__ __forceinline__ u32 ld_acquire(const u32* p) { u32 r; asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r; }
// One word of codes = 4 bf16x2 registers. Layout (set by api.pack_fp4, a permutation of the nibbles inside each
// 32-bit word): register r takes nibble 3-r (low half) and nibble 7-r (high half). w << 4r moves them to nibbles
// 3 and 7, where the sign is already bit 15 / 31 and the magnitude moves to bits 6..8 / 22..24 with one shift:
// bf16 bits = sign<<15 | mag<<6 = e2m1(code) * 2^-126 exactly (subnormal for e = 0).
__device__ __forceinline__ u32 dec1(u32 v) { return (v & 0x80008000u) | ((v >> 6) & 0x01C001C0u); }
__device__ __forceinline__ void cvt4(u32 x, u32& r0, u32& r1, u32& r2, u32& r3) {
#if MULDEC
  // same bits as dec1(x << 4r) (bench/dec.cu): a masked nibble pair times (2^12 + 2^6) or (2^8 + 2^2) lands its sign at
  // bit 15 and its three magnitude bits at 8..6 of each half; the copies are disjoint (no carries); one mask clears the rest
  const u32 M = 0x81C081C0u, y = x >> 8;
  r3 = ((x & 0x000F000Fu) * 0x1040u) & M;
  r2 = ((x & 0x00F000F0u) * 0x0104u) & M;
  r1 = ((y & 0x000F000Fu) * 0x1040u) & M;
  r0 = ((y & 0x00F000F0u) * 0x0104u) & M;
#else
  r0 = dec1(x); r1 = dec1(x << 4); r2 = dec1(x << 8); r3 = dec1(x << 12);
#endif
}
// K/V cache layouts in mma fragment order (16-byte loads in the attention):
// K row (64 dims): lane t's 8 words (ks = 0..3: dims 16ks+2t,+1 then 16ks+2t+8,+9) contiguous at t * 32 B.
__device__ __forceinline__ int kperm(int d) {
  const int ks = d >> 4, r = d & 15, j = r >> 3, t = (r & 7) >> 1, e = d & 1;
  return (t * 8 + 2 * ks + j) * 2 + e;
}
// V block (64 dims x 16 keys): lane (g, t)'s 16 words (nb = 0..7: dim 8nb+g, keys 2t,+1 then 2t+8,+9) at lane * 64 B.
__device__ __forceinline__ int vperm(int d, int kk) {
  const int nb = d >> 3, g = d & 7, j = kk >> 3, t = (kk & 7) >> 1, e = kk & 1;
  return ((g * 4 + t) * 16 + nb * 2 + j) * 2 + e;
}
__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float u16f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ u16 f2u16(float v) { bf16 h = __float2bfloat16_rn(v); return *reinterpret_cast<u16*>(&h); }
// 2^(S-64) for byte k of the scale word (S = E8M0 code, 111..138 in this checkpoint): undoes the 2^-126 of the decode
// and the 2^63 of the B operand. sp = s + 0x3F3F3F3F (bytes stay below 256: no carries).
__device__ __forceinline__ float scale_f(u32 s, int k) { return __uint_as_float((((s >> (8 * k)) & 0xffu) + 63u) << 23); }
__device__ __forceinline__ float scale_p0(u32 sp) { return __uint_as_float((sp << 23) & 0x7F800000u); }
__device__ __forceinline__ float scale_p1(u32 sp) { return __uint_as_float((sp << 15) & 0x7F800000u); }
__device__ __forceinline__ float scale_p2(u32 sp) { return __uint_as_float((sp << 7) & 0x7F800000u); }
__device__ __forceinline__ float scale_p3(u32 sp) { return __uint_as_float((sp >> 1) & 0x7F800000u); }

#ifdef DBG
#define MARK(tag) do { __syncthreads(); if ((threadIdx.x & 31) == 0) { const PState& q_ = pst[threadIdx.x >> 5]; const TT* t_ = ttab + (threadIdx.x >> 5) * 32; \
  for (int i_ = 0; i_ < q_.nt; i_++) if (!__isShared(t_[i_].x)) { printf("TT bad at mark %d layer %d block %d warp %d entry %d/%d\n", tag, dbg_layer, blockIdx.x, threadIdx.x >> 5, i_, q_.nt); break; } } } while (0)
#else
#define MARK(tag) do { if (threadIdx.x == 0) MKPROF_MARK(prof, pn, tag); } while (0)
#endif
#ifndef GBATOM
#define GBATOM 1
#endif
__device__ __forceinline__ void grid_bar(u32* ctr, u32& target) {
  __syncthreads();
  target += NB;
  if (threadIdx.x == 0) {
#ifdef FENCEBAR
    __threadfence();
    atomicAdd(ctr, 1u);
#else
#if GBATOM
    // the returned-value form (bench/gbar.cu V3: 0.97 vs 1.05 us); the value itself is not needed
    u32 old;
    asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(ctr) : "memory");
    if (old + 1 < target)   // the last arriver (acq_rel) has seen every block's release: no polling
#else
    // release add (cumulative over the block's stores, ordered by the bar.sync above), acquire polls
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(ctr) : "memory");
#endif
#endif
    while (ld_acquire(ctr) < target) { }
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------ GEMV tasks
// C fragment of one task = 16 rows x 8 columns; lane (g,t): c0,c1 = row g cols 2t,2t+1; c2,c3 = row g+8.
// one 64-column chunk of an MXFP4 rowtile: codes q (16 B of this lane), scale word s (4 B), B fragments b[8]
__device__ __forceinline__ void chunk_fp4(float* c, uint4 q, u32 s, const u32* b) {
  float t0[4], t1[4];
  u32 a0, a1, a2, a3, e0, e1, e2, e3;
  cvt4(q.x, a0, a1, a2, a3); cvt4(q.z, e0, e1, e2, e3);
  mma16816z(t0, a0, a1, a2, a3, b[0], b[1]);
  mma16816z(t1, e0, e1, e2, e3, b[4], b[5]);
  cvt4(q.y, a0, a1, a2, a3); cvt4(q.w, e0, e1, e2, e3);
  mma16816(t0, a0, a1, a2, a3, b[2], b[3]);
  mma16816(t1, e0, e1, e2, e3, b[6], b[7]);
  const u32 sp = s + 0x3F3F3F3Fu;
  const float s0 = scale_p0(sp), s2 = scale_p2(sp), s1 = scale_p1(sp), s3 = scale_p3(sp);
  c[0] = fmaf(t0[0], s0, c[0]); c[1] = fmaf(t0[1], s0, c[1]);
  c[2] = fmaf(t0[2], s2, c[2]); c[3] = fmaf(t0[3], s2, c[3]);
  c[0] = fmaf(t1[0], s1, c[0]); c[1] = fmaf(t1[1], s1, c[1]);
  c[2] = fmaf(t1[2], s3, c[2]); c[3] = fmaf(t1[3], s3, c[3]);
}

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
#define RS 2
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
// Read a task descriptor field by field through a volatile pointer. With a plain struct copy (cq = tt[ct]) ptxas
// -O1..3 produced code whose consumer held the NEXT task's descriptor when the ring had 3+ stages (bench/ring.cu:
// -G / -Xptxas -O0 pass, -O1/-O3 fail; a volatile re-read fixes it). This was the 'RS >= 3 corruption'.
__device__ __forceinline__ TT ld_tt(const TT* p) {
  const volatile TT* v = p;
  TT r; r.w = v->w; r.s = v->s; r.x = v->x; r.rows8 = v->rows8; r.n = v->n; r.slot = v->slot; r.bm = v->bm; r.k0 = v->k0;
  return r;
}
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
  // L2 evict_first policy made by createpolicy (a hard-coded policy word is architecture-specific)
#ifdef NOHINT
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
#else
  u64 pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
#endif
}

#ifdef DBG3
__device__ void* g_dbg;
#endif
// bulk prefetch of [p, p + bytes) into L2 (no shared memory, no completion): issued by one thread
__device__ __forceinline__ void l2pf(const void* p, u32 bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}
struct Ring { u32 data; u32 bars; u32 pos; const unsigned char* gdata; };   // smem addresses of this warp's stages and mbarriers; stages consumed
// producer state of a warp's task list (kept in shared memory across a grid barrier)
struct PState { int nt, pt, pi, kind; u32 issued; int pad[3]; };
__device__ __forceinline__ PState ld_ps(const PState* p) {
  const volatile PState* v = p; PState r; r.nt = v->nt; r.pt = v->pt; r.pi = v->pi; r.kind = v->kind; r.issued = v->issued; return r;
}
__device__ __forceinline__ void st_ps(PState* p, const PState& q) {
  volatile PState* v = p; v->nt = q.nt; v->pt = q.pt; v->pi = q.pi; v->kind = q.kind; v->issued = q.issued;
}

// lane 0 issues one stage: the next (up to RI) items of the current task
template <int KIND>
__device__ __forceinline__ void issue_stage(const TT* tt, PState& ps, const Ring& rg, int lane) {
  constexpr int IB = 512, SB = (KIND == 1) ? 32 : 0;
  const u32 sidx = ps.issued % RS;
  const u32 bar = rg.bars + sidx * 8, dst = rg.data + sidx * (RI * RING_ITEM);
  const TT& q = tt[ps.pt];
  const int m = min(RI, q.n - ps.pi);
#ifdef DBG2
  if (lane == 0 && (dst < rg.data || dst + m * (IB + SB) > rg.data + RS * RI * RING_ITEM || m <= 0 || m > RI || ps.pt >= ps.nt))
    printf("BAD issue block %d warp %d dst %u data %u m %d pt %d nt %d n %d pi %d\n", blockIdx.x, threadIdx.x >> 5, dst, rg.data, m, ps.pt, ps.nt, q.n, ps.pi);
#endif
#ifdef DBG3
  if (lane == 0 && (dst < rg.data || dst + m * (IB + SB) > rg.data + RS * RI * RING_ITEM || m <= 0 || m > RI || ps.pt >= ps.nt)) {
    int* d = (int*)g_dbg; if (atomicAdd(d, 1) == 0) { d[1] = blockIdx.x; d[2] = threadIdx.x >> 5; d[3] = dst; d[4] = rg.data; d[5] = m; d[6] = ps.pt; d[7] = ps.nt; d[8] = q.n; d[9] = ps.pi; } }
#endif
  if (lane == 0) {
    mbar_expect(bar, m * (IB + SB));
#if FUSE
    tma_load(dst, q.w + (size_t)ps.pi * ((IB + SB) / 16), m * (IB + SB), bar);   // one copy per stage (fused items)
#else
    tma_load(dst, q.w + (size_t)ps.pi * 32, m * IB, bar);
    if (KIND == 1) tma_load(dst + RI * IB, q.s + ps.pi * 8, m * SB, bar);
#endif
  }
  ps.pi += m;
  if (ps.pi == q.n) { ps.pt++; ps.pi = 0; }
  ps.issued++;
}

// build the warp's task table and issue the first RS stages (the ring must be drained: rg.pos == issued)
template <int KIND, class Desc>
__device__ __forceinline__ void prefetch_tasks(int ntask, TT* table, const Ring& rg, PState& ps, Desc desc) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nt = (ntask > warp) ? (ntask - warp + NW - 1) / NW : 0;
  TT* tt = table + warp * TTN;
  if (nt > TTN) asm volatile("trap;");
  if (lane < nt) {
    TD d; d.s = nullptr; d.rows8 = ROWS_ID; d.k0 = 0; desc(warp + NW * lane, d);
    tt[lane] = TT{d.w, d.s, d.x, d.rows8, d.n, d.slot, d.bm, d.k0};
  }
  __syncwarp();
  PState q; q.nt = nt; q.pt = 0; q.pi = 0; q.kind = KIND; q.issued = rg.pos;
  for (int k = 0; k < RS && q.pt < nt; k++) issue_stage<KIND>(tt, q, rg, lane);
  if (lane == 0) ps = q;
  __syncwarp();
}

#ifdef VOLLD
#define LDSASM asm volatile
#define LDSCLOB : "memory"
#else
#define LDSASM asm
#define LDSCLOB
#endif
__device__ __forceinline__ u32 lds32(u32 a) { u32 r; LDSASM("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(a) LDSCLOB); return r; }
__device__ __forceinline__ uint4 lds128(u32 a) {
  uint4 r; LDSASM("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a) LDSCLOB); return r;
}
// one item of a stage: A from the ring stage at st, B from shared memory at xb (byte address of this lane's row)
template <int KIND>
__device__ __forceinline__ void item(float* c, u32 st, u32 xb, int kb, int i, int lane, int g, bool cb) {
  constexpr int IB = 512, SB = 32;
#if FUSE
  const uint4 q = lds128(st + i * (KIND == 1 ? IB + SB : IB) + lane * 16);   // KIND 1 items: 512 B codes then 32 B scales
#else
  const uint4 q = lds128(st + i * IB + lane * 16);
#endif
  if (KIND == 0) {
    const int k = kb + i;
    u32 b0, b1;
    LDSASM("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(xb + (k >> 2) * 128 + (k & 3) * 8) LDSCLOB);
#ifdef SEL
    b0 = cb ? b0 : 0u; b1 = cb ? b1 : 0u;
#endif
    mma16816(c, q.x, q.y, q.z, q.w, b0, b1);
  } else {
#if FUSE
    const u32 sw = lds32(st + i * (IB + SB) + IB + g * 4);
#else
    const u32 sw = lds32(st + RI * IB + i * SB + g * 4);
#endif
    const uint4 x0 = lds128(xb + (kb + i) * 128), x1 = lds128(xb + (kb + i) * 128 + 16);
    u32 b[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#ifdef SEL
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

// consume the prefetched task list (more stages are issued as stages are freed)
#ifndef DBGT_GU
#define DBGT_GU 0
#endif
#ifdef DBGT
// diagnostics (tests/trace_down.py): per block 64 stamps of warp 0 in the traced consume call (down, layer 5, last step)
__device__ unsigned long long* g_tr;
__device__ int g_trk[132];
#define TRS(k_) do { if (KIND == 1 && threadIdx.x == 0 && g_trk[blockIdx.x] && trn < 63) g_tr[blockIdx.x * 64 + 1 + trn++] = (clock64() << 8) | (k_); } while (0)
#else
#define TRS(k_) do { } while (0)
#endif
template <int KIND, class Fin>
__device__ __forceinline__ void consume_tasks(TT* table, int xst, Ring& rg, PState& psm, Fin fin) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  PState ps = psm;
  const int nt = ps.nt;
  if (nt == 0) return;
  const TT* tt = table + warp * TTN;
  constexpr int IB = 512, SB = (KIND == 1) ? 32 : 0;
  constexpr int XST = (KIND == 0) ? 16 : 64;
#ifdef DBG
  if (lane == 0) for (int i = 0; i < nt; i++) if (!__isShared(tt[i].x)) { printf("consume start: block %d warp %d entry %d/%d bad (tt at %u) issued %u pos %u\n", blockIdx.x, warp, i, nt, smaddr(tt + i), ps.issued, rg.pos); break; }
#endif
  // ct is volatile: with 3+ ring stages ptxas -O1..3 otherwise produced a consumer holding the NEXT task's
  // descriptor (bench/ring.cu reproduces it; -Xptxas -O0 or a volatile ct fix it)
#if RS >= 3 || VOLCT
  volatile int ct = 0;
#else
  int ct = 0;   // RS = 2 never showed the bug; the volatile copy cost a local-memory load per stage
#endif
  int ci = 0;
  TT cq = tt[0];
  u32 cx; bool cb;   // shared-space byte address of this lane's B row
  auto setx = [&]() {
    int row = (int)((cq.rows8 >> (8 * g)) & 0xff);
    cb = (cq.bm >> g) & 1;
    cx = smaddr(cq.x + row * xst) + t * 32;
#ifdef DBG
    if (cx >= 232448u && lane == 0) printf("bad x: block %d warp %d ct %d nt %d x %p row %d n %d slot %d w %p k0 %d\n", blockIdx.x, warp, ct, nt, cq.x, row, cq.n, cq.slot, cq.w, cq.k0);
#endif
  };
  setx();
  float c[4] = {0.f, 0.f, 0.f, 0.f};
#ifdef DBGT
  int trn = 0;
#endif
  TRS(0);
  while (ct < nt) {
    const u32 sidx = rg.pos % RS;
    const int n = min(RI, cq.n - ci);
    mbar_wait(rg.bars + sidx * 8, (rg.pos / RS) & 1);
    TRS(1);
    const u32 st = rg.data + sidx * (RI * RING_ITEM);
    const int kb = cq.k0 + ci;
    if (n == RI) {
#pragma unroll
      for (int i = 0; i < RI; i++) item<KIND>(c, st, cx, kb, i, lane, g, cb);
    } else {
#pragma unroll 1
      for (int i = 0; i < n; i++) item<KIND>(c, st, cx, kb, i, lane, g, cb);
    }
    __syncwarp();
    TRS(2);
    rg.pos++;
    if (ps.pt < nt) {
      if (lane == 0) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      issue_stage<KIND>(tt, ps, rg, lane);
    }
    ci += n;
    if (ci == cq.n) {
      TRS(3);
      fin(cq.slot, c);
      TRS(4);
      c[0] = c[1] = c[2] = c[3] = 0.f;
      ci = 0;
      if (++ct < nt) { cq = tt[ct]; setx(); }
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
  if (sm) v = *reinterpret_cast<const uint2*>(r + k);
  else asm volatile("ld.global.cg.v2.u32 {%0,%1},[%2];" : "=r"(v.x), "=r"(v.y) : "l"(r + k));
  return v;
}
// xs[b][k] = bf16(w[k] * (x[b][k] * rsqrt(mean(x^2) + eps)) * sc), x read with ld.cg (bf16 [B][HID] rows). One warp
// per sequence loads its whole row at once (23 x 8 B per lane), sums the squares and writes the normed row.
// sc: power-of-two factor applied before the bf16 rounding (2^63 for the MoE input: exact)
#define NCH4 ((HID / 4 + 31) / 32)   // 4-value chunks per lane
#define NH4 12                         // chunks per half-row pass
__device__ __forceinline__ uint2 ldcg64(const u16* p) { uint2 v; asm volatile("ld.global.cg.v2.u32 {%0,%1},[%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v; }
#ifndef NORM_INLINE
__device__ __noinline__
#else
__device__ __forceinline__
#endif
void norm_to_smem(const bf16* const* rows, int B, const float* w, u16* xs, float* scratch, bool sm = false, float sc = 1.f,
                  const float* ssp = nullptr, int nss = 0, int sstride = 0) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  (void)scratch; (void)sm;
#if NSS4
  if (ssp && B <= 2) {   // track B (B <= 2: clear of the routing warps NW-B..): rows split over 4 warps each (chunks q, q+4, ...): no cross-warp reduction is needed with
               // the sum of squares from partials (every warp sums them in the same order: same bits)
    for (int task = warp; task < 4 * B; task += NW) {
      const int bb = task >> 2, qq = task & 3;
      const u16* r = reinterpret_cast<const u16*>(rows[bb]);
      uint2 v[6];
#pragma unroll
      for (int u = 0; u < 6; u++) {
        const int jj = qq + 4 * u, k = (lane + 32 * jj) * 4;
        v[u] = (jj < NCH4 && k < HID) ? ldcg64(r + k) : make_uint2(0u, 0u);
      }
#if NSS4W
      float4 wq[6];   // the norm weights in flight with the row (24 registers)
#pragma unroll
      for (int u = 0; u < 6; u++) {
        const int jj = qq + 4 * u, k = (lane + 32 * jj) * 4;
        wq[u] = (jj < NCH4 && k < HID) ? *reinterpret_cast<const float4*>(w + k) : make_float4(0.f, 0.f, 0.f, 0.f);
      }
#endif
      const float* q = ssp + (size_t)bb * sstride;
      float pq[6], s = 0.f;
#pragma unroll
      for (int k = 0; k < 6; k++) pq[k] = lane + 32 * k < nss ? ldcgf(q + lane + 32 * k) : 0.f;
#pragma unroll
      for (int k = 0; k < 6; k++) s += pq[k];
#pragma unroll
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      const float ri = rsqrtf(s / (float)HID + 1e-5f);
#pragma unroll
      for (int u = 0; u < 6; u++) {
        const int jj = qq + 4 * u, k = (lane + 32 * jj) * 4;
        if (jj < NCH4 && k < HID) {
#if NSS4W
          const float4 wv = wq[u];
#else
          float4 wv = *reinterpret_cast<const float4*>(w + k);
#endif
          u16 o0 = f2u16(wv.x * (__uint_as_float(v[u].x << 16) * ri) * sc);
          u16 o1 = f2u16(wv.y * (__uint_as_float(v[u].x & 0xffff0000u) * ri) * sc);
          u16 o2 = f2u16(wv.z * (__uint_as_float(v[u].y << 16) * ri) * sc);
          u16 o3 = f2u16(wv.w * (__uint_as_float(v[u].y & 0xffff0000u) * ri) * sc);
          xfo_store4(xs + bb * XSH, k, o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
        }
      }
    }
    __syncthreads();
    return;
  }
#endif
  if (warp < B) {
    const u16* r = reinterpret_cast<const u16*>(rows[warp]);
    float s = 0.f;
    if (ssp) {   // track B: the row's sum of squares from partials written where the row was produced (nss <= 192);
                 // the whole row and the partials in flight at once, one pass
      uint2 v[NCH4];
#pragma unroll
      for (int jj = 0; jj < NCH4; jj++) {
        const int k = (lane + 32 * jj) * 4;
        v[jj] = k < HID ? ldcg64(r + k) : make_uint2(0u, 0u);
      }
      const float* q = ssp + (size_t)warp * sstride;
      float pq[6];
#pragma unroll
      for (int k = 0; k < 6; k++) pq[k] = lane + 32 * k < nss ? ldcgf(q + lane + 32 * k) : 0.f;
#pragma unroll
      for (int k = 0; k < 6; k++) s += pq[k];
#pragma unroll
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      const float ri = rsqrtf(s / (float)HID + 1e-5f);
#pragma unroll
      for (int jj = 0; jj < NCH4; jj++) {
        const int k = (lane + 32 * jj) * 4;
        if (k < HID) {
          float4 wv = *reinterpret_cast<const float4*>(w + k);
          u16 o0 = f2u16(wv.x * (__uint_as_float(v[jj].x << 16) * ri) * sc);
          u16 o1 = f2u16(wv.y * (__uint_as_float(v[jj].x & 0xffff0000u) * ri) * sc);
          u16 o2 = f2u16(wv.z * (__uint_as_float(v[jj].y << 16) * ri) * sc);
          u16 o3 = f2u16(wv.w * (__uint_as_float(v[jj].y & 0xffff0000u) * ri) * sc);
          xfo_store4(xs + warp * XSH, k, o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
        }
      }
    } else {
    // sum of squares in two passes of 12 chunks (all loads of a pass in flight at once); lane order j = 0..22
#pragma unroll
    for (int h = 0; h < 2; h++) {
      uint2 v[NH4];
#pragma unroll
      for (int jj = 0; jj < NH4; jj++) {
        const int k = (lane + 32 * (h * NH4 + jj)) * 4;
        v[jj] = (h * NH4 + jj < NCH4 && k < HID) ? ldcg64(r + k) : make_uint2(0u, 0u);
      }
#pragma unroll
      for (int jj = 0; jj < NH4; jj++) {
        float a = __uint_as_float(v[jj].x << 16), b = __uint_as_float(v[jj].x & 0xffff0000u);
        float c = __uint_as_float(v[jj].y << 16), d = __uint_as_float(v[jj].y & 0xffff0000u);
        s += a * a + b * b + c * c + d * d;
      }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float ri = rsqrtf(s / (float)HID + 1e-5f);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      uint2 v[NH4];
#pragma unroll
      for (int jj = 0; jj < NH4; jj++) {
        const int k = (lane + 32 * (h * NH4 + jj)) * 4;
        v[jj] = (h * NH4 + jj < NCH4 && k < HID) ? ldcg64(r + k) : make_uint2(0u, 0u);
      }
#pragma unroll
      for (int jj = 0; jj < NH4; jj++) {
        const int k = (lane + 32 * (h * NH4 + jj)) * 4;
        if (h * NH4 + jj < NCH4 && k < HID) {
          float4 wv = *reinterpret_cast<const float4*>(w + k);
          u16 o0 = f2u16(wv.x * (__uint_as_float(v[jj].x << 16) * ri) * sc);
          u16 o1 = f2u16(wv.y * (__uint_as_float(v[jj].x & 0xffff0000u) * ri) * sc);
          u16 o2 = f2u16(wv.z * (__uint_as_float(v[jj].y << 16) * ri) * sc);
          u16 o3 = f2u16(wv.w * (__uint_as_float(v[jj].y & 0xffff0000u) * ri) * sc);
          xfo_store4(xs + warp * XSH, k, o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
        }
      }
    }
    }
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------ smem layout
// flat down path (B <= 2): the layer output from the down partials, for all rows, into shared memory (row-major
// bf16 [B][HID] at dst); the block also stores its share (every NB-th group of 4 columns) to x.
// [0, 65664)          xs: bf16 [8][4104] (hidden vectors use stride 2888)
// [65664, +40960)     red: float [80][128]
// attention reuses: sc float [8][2048] at 0 (64 KB), ps bf16 [8][2048+16] at 65664 (32.3 KB), ared aliases sc
// small scalars at SM_MISC
#define SM_XS 0
#ifndef XS_ROWS
#define XS_ROWS 8
#endif
#define SM_RED (XS_ROWS * 8208)
#ifndef RED_SLOTS
#define RED_SLOTS 60
#endif
#define SM_MISC (SM_RED + RED_SLOTS * 512)
#ifdef TTLAST
#define SM_BARS (SM_MISC + 4096)
#define SM_PS (SM_BARS + NW * RS * 8 + 64)
#define SM_RING (SM_PS + NW * 32)
#define SM_TT (SM_RING + NW * RING_WARP)
#define SM_TOTAL (SM_TT + SM_TT_BYTES)
#else
#define SM_TT (SM_MISC + 4096)
#define SM_BARS (SM_TT + SM_TT_BYTES)
#define SM_PS (SM_BARS + NW * RS * 8 + 64)
#define SM_RING (SM_PS + NW * 32)
#define SM_TOTAL (SM_RING + NW * RING_WARP)
#endif

struct Misc {
  float rinv[MAXB];
  int U;                 // number of chosen experts (union over the batch)
  int sexp[32];          // slot -> expert id (ascending)
  float rw[MAXB][32];    // routing weight of (sequence, slot), 0 if not routed
  int bsl[MAXB][4];      // the 4 slots of sequence b, ascending (= ascending expert id)
  unsigned char prs[16]; // down staging: (slot - fr_s0) << 3 | b of staged row pr
  float bw[MAXB][4];     // their routing weights
  u32 smask[32];         // bit b set if sequence b routed to slot
  int tok[MAXB];
  int fr_s0, fr_s1, fr_np, fr_u0, fr_u1;
  u64 srows[32];
  float tmpf[128];
  float rlg[MAXB * NE];   // router logits (bf16-rounded)
  int tmpi[128];
};
static_assert(sizeof(Misc) <= 4096, "Misc must fit its 4 KB");

__device__ void combine_flat(const Params& p, const Misc* ms, int B, u16* dst) {
  const int tid = threadIdx.x;
  const int U = ms->U;
  for (int i = tid; i < B * (HID / 4); i += NT) {
    const int b = i / (HID / 4), k = (i % (HID / 4)) * 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int sl = 0; sl < U; sl++) {
      if (!(ms->smask[sl] >> b & 1)) continue;
      const float wv = ms->rw[b][sl];
      uint4 y = ldcg128(p.ypart + ((size_t)sl * MAXB + b) * HID + k);
      acc[0] = fmaf(wv, __uint_as_float(y.x), acc[0]); acc[1] = fmaf(wv, __uint_as_float(y.y), acc[1]);
      acc[2] = fmaf(wv, __uint_as_float(y.z), acc[2]); acc[3] = fmaf(wv, __uint_as_float(y.w), acc[3]);
    }
    uint2 xv; asm volatile("ld.global.cg.v2.u32 {%0,%1},[%2];" : "=r"(xv.x), "=r"(xv.y) : "l"(p.xm + b * HID + k));
    u16 o0 = f2u16(__uint_as_float(xv.x << 16) + bfr(acc[0]));
    u16 o1 = f2u16(__uint_as_float(xv.x & 0xffff0000u) + bfr(acc[1]));
    u16 o2 = f2u16(__uint_as_float(xv.y << 16) + bfr(acc[2]));
    u16 o3 = f2u16(__uint_as_float(xv.y & 0xffff0000u) + bfr(acc[3]));
    uint2 o = make_uint2(o0 | ((u32)o1 << 16), o2 | ((u32)o3 << 16));
    *reinterpret_cast<uint2*>(dst + b * HID + k) = o;
    if ((k >> 2) % NB == (int)blockIdx.x) *reinterpret_cast<uint2*>(p.x + b * HID + k) = o;
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------ token choice
// argmax over the blocks' (value, index) candidates of sequence b by one warp: all loads in flight, then a shuffle
// reduction (largest value, lowest index on ties: the same choice as the sequential scan)
__device__ __forceinline__ int argmax_blocks(const Params& p, int b, int lane) {
  float bv = -INFINITY; int bi = 0x7fffffff;
  float v[(NB + 31) / 32]; int ix[(NB + 31) / 32];
#pragma unroll
  for (int j = 0; j < (NB + 31) / 32; j++) {
    const int k = lane + 32 * j;
    v[j] = k < NB ? ldcgf(p.amv + k * MAXB + b) : -INFINITY; ix[j] = k < NB ? ldcgi(p.ami + k * MAXB + b) : 0x7fffffff;
  }
#pragma unroll
  for (int j = 0; j < (NB + 31) / 32; j++) if (v[j] > bv || (v[j] == bv && ix[j] < bi)) { bv = v[j]; bi = ix[j]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  return bi;
}

// ------------------------------------------------------------------------------------------------ attention
// Block (sequence b, K/V head): flash-decoding inside the block. Warp w walks the 16-key blocks w, w + NW, ... of
// the window [lo, pos] with an online softmax in registers: scores (8 query heads x 16 keys, fp32, two mma), running
// max m, e = exp(s - m), P = bf16(e) straight into the A operand of the P V mma, acc rescaled by exp(m_old - m),
// l = sum e. One combine over the warps in warp order with the sink:
// out = sum_w acc_w e^(m_w - M) / (sum_w l_w e^(m_w - M) + e^(sink - M)), M = max(sink, m_w). The arithmetic does not
// depend on the batch (same bits alone and in a batch). Out of line: its own register allocation.
__device__ __noinline__ void attention_flash(const Params& p, const LayerP& Lp, unsigned char* smem, int pos) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int b = blockIdx.x >> 3, kvh = blockIdx.x & 7;
  const int lo = Lp.sliding ? max(0, pos - 127) : 0;
  u16* qs = reinterpret_cast<u16*>(smem + SM_RED);              // [8][72]
  float* wacc = reinterpret_cast<float*>(smem);                 // [NW][8][64]
  float* wml = reinterpret_cast<float*>(smem + NW * 8 * 64 * 4);  // [NW][8] m, then [NW][8] l
  if (tid < 8 * 32) {
    const int h = tid >> 5, d = (tid & 31) * 2;
    *reinterpret_cast<u32*>(qs + h * 72 + d) = ldcg32(p.q + (size_t)b * 4096 + (kvh * 8 + h) * 64 + d);
  }
  __syncthreads();
  u32 qa[4][2];
#pragma unroll
  for (int ks = 0; ks < 4; ks++) {
    qa[ks][0] = *reinterpret_cast<const u32*>(qs + g * 72 + 16 * ks + 2 * t);
    qa[ks][1] = *reinterpret_cast<const u32*>(qs + g * 72 + 16 * ks + 2 * t + 8);
  }
  const bf16* kbase = Lp.kc + ((size_t)b * NKV + kvh) * p.maxctx * HD;
  const bf16* vbase = Lp.vc + ((size_t)b * NKV + kvh) * (p.maxctx / 16) * HD * 16;
  float acc[8][4];
#pragma unroll
  for (int nb = 0; nb < 8; nb++) { acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f; }
  float m = -INFINITY, l = 0.f;
  const int bl0 = lo >> 4, bl1 = pos >> 4;
  for (int blk = bl0 + warp; blk <= bl1; blk += NW) {
    const int k0 = blk * 16;
    // K rows k0 + g and k0 + 8 + g (lane t's 32 B of each), V block (lane's 64 B)
    const uint4* kq0 = reinterpret_cast<const uint4*>(kbase + (size_t)(k0 + g) * HD) + 2 * t;
    const uint4* kq1 = reinterpret_cast<const uint4*>(kbase + (size_t)(k0 + 8 + g) * HD) + 2 * t;
    const uint4 x0 = ldcg128(kq0), x1 = ldcg128(kq0 + 1), y0 = ldcg128(kq1), y1 = ldcg128(kq1 + 1);
    const uint4* vq = reinterpret_cast<const uint4*>(vbase + (size_t)blk * HD * 16) + lane * 4;
    uint4 v4[4];
#pragma unroll
    for (int u = 0; u < 4; u++) v4[u] = ldcg128(vq + u);
    const u32 ka[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const u32 kb[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
    float c1[4] = {0.f, 0.f, 0.f, 0.f}, c2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      mma16816(c1, qa[ks][0], 0u, qa[ks][1], 0u, ka[2 * ks], ka[2 * ks + 1]);
      mma16816(c2, qa[ks][0], 0u, qa[ks][1], 0u, kb[2 * ks], kb[2 * ks + 1]);
    }
    // head g: keys k0+2t, +1 (c1[0], c1[1]) and k0+8+2t, +1 (c2[0], c2[1])
    const int ka0 = k0 + 2 * t, kb0 = k0 + 8 + 2 * t;
    const float s0 = (ka0 >= lo && ka0 <= pos) ? c1[0] * 0.125f : -INFINITY;
    const float s1 = (ka0 + 1 >= lo && ka0 + 1 <= pos) ? c1[1] * 0.125f : -INFINITY;
    const float s2 = (kb0 >= lo && kb0 <= pos) ? c2[0] * 0.125f : -INFINITY;
    const float s3 = (kb0 + 1 >= lo && kb0 + 1 <= pos) ? c2[1] * 0.125f : -INFINITY;
    float mb = fmaxf(fmaxf(s0, s1), fmaxf(s2, s3));
    mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
    mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 2));
    const float mn = fmaxf(m, mb);   // finite: every block holds a key of the window
    const float al = expf(m - mn);
    const float e0 = expf(s0 - mn), e1 = expf(s1 - mn), e2 = expf(s2 - mn), e3 = expf(s3 - mn);
    l = l * al + ((e0 + e1) + (e2 + e3));
    m = mn;
#pragma unroll
    for (int nb = 0; nb < 8; nb++) { acc[nb][0] *= al; acc[nb][1] *= al; }
    const u32 a0 = (u32)f2u16(e0) | ((u32)f2u16(e1) << 16), a2 = (u32)f2u16(e2) | ((u32)f2u16(e3) << 16);
    const u32* vw = reinterpret_cast<const u32*>(v4);
#pragma unroll
    for (int nb = 0; nb < 8; nb++) mma16816(acc[nb], a0, 0u, a2, 0u, vw[2 * nb], vw[2 * nb + 1]);
  }
  // l of head g over the warp's keys: the 4 lanes of the row group
  l += __shfl_xor_sync(0xffffffffu, l, 1);
  l += __shfl_xor_sync(0xffffffffu, l, 2);
#pragma unroll
  for (int nb = 0; nb < 8; nb++) {
    wacc[(warp * 8 + g) * 64 + 8 * nb + 2 * t] = acc[nb][0];
    wacc[(warp * 8 + g) * 64 + 8 * nb + 2 * t + 1] = acc[nb][1];
  }
  if (t == 0) { wml[warp * 8 + g] = m; wml[NW * 8 + warp * 8 + g] = l; }
  __syncthreads();
  for (int it = tid; it < 512; it += NT) {
    const int h = it >> 6, d = it & 63;
    const float sink = Lp.sinks[kvh * 8 + h];
    float M = sink;
#pragma unroll
    for (int w = 0; w < NW; w++) M = fmaxf(M, wml[w * 8 + h]);
    float L = expf(sink - M), O = 0.f;
#pragma unroll
    for (int w = 0; w < NW; w++) {
      const float f = expf(wml[w * 8 + h] - M);   // warps without keys: m = -inf -> 0
      L = fmaf(wml[NW * 8 + w * 8 + h], f, L);
      O = fmaf(wacc[(w * 8 + h) * 64 + d], f, O);
    }
    p.attn[(size_t)b * 4096 + (kvh * 8 + h) * 64 + d] = __float2bfloat16_rn(O / L);
  }
}

// ------------------------------------------------------------------------------------------------ the kernel
extern "C" __global__ void __launch_bounds__(NT, 1) mk_kernel(const __grid_constant__ Params p, const __grid_constant__ RunArgs a) {
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
#if P2TMA
  // head recompute tile copies: one mbarrier and its completed-phase count in the pad after the ring barriers
  if (tid == 0) { for (int c = 0; c < P2CH; c++) mbar_init(smaddr(smem + SM_BARS + NW * RS * 8) + 8 * c, 1); *reinterpret_cast<u32*>(smem + SM_BARS + NW * RS * 8 + 8 * P2CH) = 0; }
#endif
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  const int bid = blockIdx.x;
#if BFIX
  const int B = BFIX;   // build specialised for one batch size (the host must launch it with a.B == BFIX)
#else
  const int B = a.B;
#endif
  u32 bar_target = 0;
  MkProf prof{p.prof, p.prof_cap};
  MKPROF_DECL(pn);
#ifdef SKEL
  if (tid == 0 && bid == 0) g_skel = p.skel;
  // (every block reads g_skel after the first barrier; set again by each block: same value)
  g_skel = p.skel;
#endif
  u32* bar = p.bar;
#ifdef DBG3
  g_dbg = (void*)p.skel;
#endif
  // P = prompt length (mode 0/1: 1); total steps = P + nsteps - 1; step st reads token st of the prompt while
  // st < P, otherwise the argmax of the previous step (= generated token st - P); the head runs for st >= P - 1.
  const int P = (a.mode == 2) ? a.nprompt : 1;
  const int total = P + a.nsteps - 1;
  const bool sketch = a.mode != 0 && p.sk != nullptr;   // track B greedy head
  u32 hstep = a.hstep0;                                  // heads run so far (parity of the candidate list)

  // ---- prefetch helpers: build a warp's task table and issue its first ring stages (weights only, no data needed)
  auto pf_qkv = [&](int l) {
    const LayerP& L = p.L[l];
    const int r0 = bid * RT_QKV / NB, R = (bid + 1) * RT_QKV / NB - r0;
    prefetch_tasks<0>(R * NSEG_QKV, ttab, rg, pst[warp], [&](int task, TD& d) {
      int r = task / NSEG_QKV, s = task % NSEG_QKV, k0 = s * KS_H / NSEG_QKV, k1 = (s + 1) * KS_H / NSEG_QKV;
      d.w = L.wqkv + ((size_t)(r0 + r) * KS_H + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
    });
  };
  auto pf_o = [&](int l) {
    const LayerP& L = p.L[l];
    const int r0 = bid * RT_O / NB, R = (bid + 1) * RT_O / NB - r0;
    prefetch_tasks<0>(R * NSEG_O, ttab, rg, pst[warp], [&](int task, TD& d) {
      int r = task / NSEG_O, s = task % NSEG_O, k0 = s * KS_A / NSEG_O, k1 = (s + 1) * KS_A / NSEG_O;
      d.w = L.wo + ((size_t)(r0 + r) * KS_A + k0) * 32; d.x = xs; d.k0 = k0; d.n = k1 - k0; d.slot = task; d.bm = (1u << B) - 1;
    });
  };
  auto pf_router = [&](int l) {   // blocks < RKB: ksteps [4 bid, 4 bid + 4) of both router rowtiles
    const LayerP& L = p.L[l];
    prefetch_tasks<0>(RB, ttab, rg, pst[warp], [&](int task, TD& d) {
      d.w = L.wr + ((size_t)task * KS_H + 4 * bid) * 32; d.x = xs; d.k0 = 4 * bid; d.n = 4; d.slot = task; d.bm = (1u << B) - 1;
    });
  };

  auto pf_head = [&]() {
    const int r0 = (int)((long)bid * RT_HEAD / NB), r1 = (int)((long)(bid + 1) * RT_HEAD / NB);
    prefetch_tasks<0>(r1 - r0, ttab, rg, pst[warp], [&](int task, TD& d) {
      d.w = p.head + (size_t)(r0 + task) * KS_H * 32; d.x = xs; d.n = KS_H; d.slot = r0 + task; d.bm = (1u << B) - 1;
    });
  };
  pf_qkv(0);

#ifdef DBG
  int dbg_layer = -1;
#endif
  for (int st = 0; st < total; st++) {
    const int pos = a.pos0 + st;
    if (st >= P && warp < B) {   // the previous step's greedy token: one warp per sequence
      const int bi = argmax_blocks(p, warp, lane);
      const long long tk = (bi >= 0 && bi < VOCAB) ? bi : 0;
      if (lane == 0) {
        if (bid == 0) {
          if (a.mode == 1) a.out_tok[(size_t)warp * a.nsteps + (st - P)] = tk;
          else a.host_tok[warp * a.nsteps + (st - P)] = (int)tk;
        }
        ms->tok[warp] = (int)tk;
      }
    } else if (st < P && tid < B) {
      long long tk = a.tok[(size_t)tid * a.tok_stride + st];
      if (tk < 0 || tk >= VOCAB) { tk = 0; if (bid == 0) *a.err = 1; }   // never read past the embedding
      ms->tok[tid] = (int)tk;
    }
    __syncthreads();
    MARK(MKPROF_STEP);
    if (a.mode == 2 && bid == 0 && tid == 0 && st > P - 1 + 0 && st >= P) { __threadfence_system(); *a.host_prog = st - P + 1; }

    const bf16* rows_in[MAXB];
    for (int layer = 0; layer < NL; layer++) {
#ifdef DBG
      dbg_layer = layer;
#endif
      const LayerP& Lp = p.L[layer];
      // flat path: units u in [u0, u1) of U * 180 (slot = u / 180, rowtile = u % 180); staged rows: the routed
      // (slot, column) pairs of the slots [fs0, fs1] this block touches
      auto fround_flat = [&]() {
        const int N = ms->U * RT_DN * DSPLIT;
        if (tid == 0) {
          const int u0 = (int)((long)bid * N / NB), u1 = (int)((long)(bid + 1) * N / NB);
          const u32 bmask = (1u << B) - 1;
          int s0 = u1 > u0 ? u0 / (RT_DN * DSPLIT) : 0, s1 = u1 > u0 ? (u1 - 1) / (RT_DN * DSPLIT) + 1 : 0;
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
      // down: one task per unit (slot, rowtile), whole K; units slot-major, contiguous per block
      auto pf_down = [&]() {
        const int u0 = ms->fr_u0, s0 = ms->fr_s0;
        prefetch_tasks<1>(ms->fr_u1 - u0, ttab, rg, pst[warp], [&](int task, TD& d) {
          int uu = u0 + task, slot = uu / (RT_DN * DSPLIT), rt = DORD ? (uu % RT_DN) : (uu / DSPLIT) % RT_DN, h = DORD ? (uu / RT_DN) % DSPLIT : uu % DSPLIT, e = ms->sexp[slot];
          const int k0 = h * KC / DSPLIT, k1 = (h + 1) * KC / DSPLIT;
          d.w = Lp.dn_c + (((size_t)e * RT_DN + rt) * KC + k0) * (FUSE ? 34 : 32);
          d.s = Lp.dn_s + (((size_t)e * RT_DN + rt) * KC + k0) * 8;
          d.x = xs; d.k0 = k0; d.rows8 = ms->srows[slot - s0];
          d.n = k1 - k0; d.slot = uu; d.bm = ms->smask[slot] & ((1u << B) - 1);
        });
      };
      // ---------------- phase A: norm1 + qkv
      MARK(1);
#if KVPF
      // attention blocks: pull this layer's K/V of the window (positions before pos) into L2 for phase B
      if (B <= KVPF_MAXB && bid < B * NKV && tid == 0 && pos > 0) {   // (at B = 8 the extra traffic cost +20 us, e0240)
        const int ab = bid >> 3, akv = bid & 7;
        const int lo = Lp.sliding ? max(0, pos - 127) : 0;
        const bf16* kb = Lp.kc + (((size_t)ab * NKV + akv) * p.maxctx + lo) * HD;
        l2pf(kb, (pos - lo) * HD * 2);
        const bf16* vb = Lp.vc + (((size_t)ab * NKV + akv) * (p.maxctx / 16) + (lo >> 4)) * HD * 16;
        l2pf(vb, ((pos >> 4) - (lo >> 4) + 1) * HD * 16 * 2);
      }
#endif
      for (int b = 0; b < MAXB; b++)
        rows_in[b] = (layer == 0) ? p.embed + (size_t)(b < B ? ms->tok[b] : 0) * HID : p.x + b * HID;
#if NSS
      if (layer > 0) NORM_SS(rows_in, B, Lp.ln1, xs, 1.f, p.rlog + (size_t)MAXB * NB * (NE + 1), RT_DN, RT_DN);
      else
#endif
      norm_to_smem(rows_in, B, Lp.ln1, xs, ms->rinv);
      MARK(11);
      {
        const int r0 = bid * RT_QKV / NB, r1 = (bid + 1) * RT_QKV / NB, R = r1 - r0;
        consume_tasks<0>(ttab, XSH, rg, pst[warp], [&](int slot, const float* c) { store_task(red, slot, c); });
#if PFOE
        pf_o(layer);   // this warp's ring is drained: the o_proj stages go out before the epilogue
#endif
#if EPLA
        // the epilogue's bias and rotary inputs are loaded before the block barrier (one item per thread)
        static_assert(3 * 8 * MAXB <= NT, "one qkv epilogue item per thread");
        float eb1 = 0.f, eb2 = 0.f, ecs = 0.f, esn = 0.f;
        if (tid < R * 8 * B) {
          const int r = tid / (8 * B), i = (tid / B) % 8, grt = r0 + r;
          eb1 = Lp.bqkv[grt * 16 + i]; eb2 = Lp.bqkv[grt * 16 + i + 8];
          if (grt < 288) { const int d = 8 * (grt & 3) + i; ecs = p.cosT[pos * 32 + d]; esn = p.sinT[pos * 32 + d]; }
        }
#endif
        __syncthreads();
        MARK(26);
        // epilogue: item = (r, i<8, b)
        for (int it = tid; it < R * 8 * B; it += NT) {
          int r = it / (8 * B), i = (it / B) % 8, b = it % B;
          float v1 = 0.f, v2 = 0.f;
          for (int s = 0; s < NSEG_QKV; s++) { v1 += red_at(red, r * NSEG_QKV + s, i, b); v2 += red_at(red, r * NSEG_QKV + s, i + 8, b); }
          int grt = r0 + r;
#if EPLA
          v1 = bfr(v1 + eb1);
          v2 = bfr(v2 + eb2);
#else
          v1 = bfr(v1 + Lp.bqkv[grt * 16 + i]);
          v2 = bfr(v2 + Lp.bqkv[grt * 16 + i + 8]);
#endif
          if (grt < 288) {     // q or k: rows i, i+8 = dims d, d+32 of one head
            int j = grt & 3, d = 8 * j + i;
#if EPLA
            float cs = ecs, sn = esn;
#else
            float cs = p.cosT[pos * 32 + d], sn = p.sinT[pos * 32 + d];
#endif
            float o1 = bfr(bfr(v1 * cs) - bfr(v2 * sn));
            float o2 = bfr(bfr(v2 * cs) + bfr(v1 * sn));
            if (grt < 256) {
              int h = grt >> 2;
              p.q[(size_t)b * 4096 + h * 64 + d] = __float2bfloat16_rn(o1);
              p.q[(size_t)b * 4096 + h * 64 + d + 32] = __float2bfloat16_rn(o2);
            } else {
              int kvh = (grt - 256) >> 2;
              bf16* kp = Lp.kc + (((size_t)b * NKV + kvh) * p.maxctx + pos) * HD;
              kp[kperm(d)] = __float2bfloat16_rn(o1); kp[kperm(d + 32)] = __float2bfloat16_rn(o2);
            }
          } else {
            int kvh = (grt - 288) >> 2, jj = (grt - 288) & 3;
            int d1 = 16 * jj + i, d2 = d1 + 8;
            bf16* vp = Lp.vc + (((size_t)b * NKV + kvh) * (p.maxctx / 16) + pos / 16) * HD * 16;
            vp[vperm(d1, pos & 15)] = __float2bfloat16_rn(v1);
            vp[vperm(d2, pos & 15)] = __float2bfloat16_rn(v2);
          }
        }
      }
#if !PFOE
      pf_o(layer);
#endif
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
      MARK(2);
      // ---------------- phase B: attention, block = (b, kvh)
#ifdef PFO
      // blocks without attention work: pull this block's o_proj rows into L2 while HBM is idle
      if (bid >= B * NKV && tid == 0) {
        const int r0 = bid * RT_O / NB, r1 = (bid + 1) * RT_O / NB;
        for (int r = r0; r < r1; r++) l2pf(Lp.wo + (size_t)r * KS_A * 32, KS_A * 512);
      }
#endif
#ifdef OLDATT
      if (bid < B * NKV) {
        const int b = bid >> 3, kvh = bid & 7;
        const int lo = Lp.sliding ? max(0, pos - 127) : 0;
        const int n = pos - lo + 1;
        float* sc = reinterpret_cast<float*>(smem);                     // [8][2048]
        u16* qs = reinterpret_cast<u16*>(smem + SM_RED);                // [8][72] (red is free during attention)
        float* ilh = reinterpret_cast<float*>(smem + SM_RED + 8 * 72 * 2);   // [8] 1 / denominator per head
        if (tid < 8 * 32) {
          int h = tid >> 5, d = (tid & 31) * 2;
          u32 v = ldcg32(p.q + (size_t)b * 4096 + (kvh * 8 + h) * 64 + d);
          *reinterpret_cast<u32*>(qs + h * 72 + d) = v;
        }
        __syncthreads();
        u32 qa[4][2];
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
          qa[ks][0] = *reinterpret_cast<const u32*>(qs + g * 72 + 16 * ks + 2 * t);
          qa[ks][1] = *reinterpret_cast<const u32*>(qs + g * 72 + 16 * ks + 2 * t + 8);
        }
        const bf16* kbase = Lp.kc + ((size_t)b * NKV + kvh) * p.maxctx * HD;
        const int ngrp = (n + 7) >> 3;
        for (int gr = warp; gr < ngrp; gr += NW) {
          int key = lo + 8 * gr + g;
          bool valid = key <= pos;
          u32 kb[8];
          { const uint4* kq = reinterpret_cast<const uint4*>(kbase + (size_t)(valid ? key : lo) * HD) + 2 * t;
            uint4 x0 = ldcg128(kq), x1 = ldcg128(kq + 1); kb[0] = x0.x; kb[1] = x0.y; kb[2] = x0.z; kb[3] = x0.w; kb[4] = x1.x; kb[5] = x1.y; kb[6] = x1.z; kb[7] = x1.w; }
          float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int ks = 0; ks < 4; ks++) mma16816(c, qa[ks][0], 0u, qa[ks][1], 0u, kb[2 * ks], kb[2 * ks + 1]);
          int j0 = 8 * gr + 2 * t;
          sc[g * SCS + j0] = (lo + j0 <= pos) ? c[0] * 0.125f : -INFINITY;
          sc[g * SCS + j0 + 1] = (lo + j0 + 1 <= pos) ? c[1] * 0.125f : -INFINITY;
        }
        __syncthreads();
        // softmax per head (warps 0..7), sink joins the denominator
        if (warp < 8) {
          const int h = warp;
          float sink = Lp.sinks[kvh * 8 + h];
          float m = sink;
          for (int j = lane; j < n; j += 32) m = fmaxf(m, sc[h * SCS + j]);
#pragma unroll
          for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
          float l = 0.f;
          for (int j = lane; j < n; j += 32) { float e = expf(sc[h * SCS + j] - m); sc[h * SCS + j] = e; l += e; }
#pragma unroll
          for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
          l += expf(sink - m);
          if (lane == 0) ilh[h] = 1.f / l;
        }
        __syncthreads();
        // P V: 16-key blocks aligned at multiples of 16 (absolute); warps round-robin; acc[8 dim groups][4]
        {
          const int base = lo & ~15;
          const int nblk = ((pos - base) >> 4) + 1;
          const bf16* vbase = Lp.vc + ((size_t)b * NKV + kvh) * (p.maxctx / 16) * HD * 16;
          float acc[8][4];
#pragma unroll
          for (int nb = 0; nb < 8; nb++) { acc[nb][0] = acc[nb][1] = acc[nb][2] = acc[nb][3] = 0.f; }
          for (int kb = warp; kb < nblk; kb += NW) {
            int blk = (base >> 4) + kb;
            // P = bf16(e * (1/l)) of keys base+16kb+2t(+1) and +8(+9); 0 outside [lo, pos]
            const float il = ilh[g];
            auto pk = [&](int key) -> u32 {
              int j = key - lo;
              float v0 = (j >= 0 && key <= pos) ? sc[g * SCS + j] * il : 0.f;
              float v1 = (j + 1 >= 0 && key + 1 <= pos) ? sc[g * SCS + j + 1] * il : 0.f;
              return (u32)f2u16(v0) | ((u32)f2u16(v1) << 16);
            };
            u32 a0 = pk(base + 16 * kb + 2 * t), a2 = pk(base + 16 * kb + 2 * t + 8);
            const u16* vb = reinterpret_cast<const u16*>(vbase + (size_t)blk * HD * 16);
#pragma unroll
            { const uint4* vq = reinterpret_cast<const uint4*>(vb) + lane * 4; uint4 v4[4];
#pragma unroll
              for (int u = 0; u < 4; u++) v4[u] = ldcg128(vq + u);
              const u32* vw = reinterpret_cast<const u32*>(v4);
#pragma unroll
              for (int nb = 0; nb < 8; nb++) mma16816(acc[nb], a0, 0u, a2, 0u, vw[2 * nb], vw[2 * nb + 1]); }
          }
          __syncthreads();   // sc no longer needed: reuse as ared [16 warps][8 heads][64]
          float* ared = sc;
#pragma unroll
          for (int nb = 0; nb < 8; nb++) {
            ared[(warp * 8 + g) * 64 + 8 * nb + 2 * t] = acc[nb][0];
            ared[(warp * 8 + g) * 64 + 8 * nb + 2 * t + 1] = acc[nb][1];
          }
          __syncthreads();
          for (int it = tid; it < 512; it += NT) {
            int h = it >> 6, d = it & 63;
            float s = 0.f;
            for (int w = 0; w < NW; w++) s += ared[(w * 8 + h) * 64 + d];
            p.attn[(size_t)b * 4096 + (kvh * 8 + h) * 64 + d] = __float2bfloat16_rn(s);
          }
        }
      }
#else
      if (bid < B * NKV) attention_flash(p, Lp, smem, pos);
#endif
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
      MARK(3);
      // ---------------- phase C: o_proj + residual
      {
        for (int i = tid; i < B * 1024; i += NT) {
          int b = i >> 10, k = (i & 1023) * 4;
          uint2 v; asm volatile("ld.global.cg.v2.u32 {%0,%1},[%2];" : "=r"(v.x), "=r"(v.y) : "l"(p.attn + (size_t)b * 4096 + k));
          xfo_store4(xs + b * XSA, k, v.x, v.y);
        }
#if RFOLD && RFPF
        // this block's folded router rows (R*16 lines of 128 B) into L1 for the epilogue: no registers held
        if (warp == NW - 1) {
          const int r0 = bid * RT_O / NB, R = (bid + 1) * RT_O / NB - r0;
          if (lane < R * 16) asm volatile("prefetch.global.L1 [%0];" :: "l"(reinterpret_cast<const float*>(Lp.wr) + (size_t)(r0 * 16 + lane) * 32));
        }
#endif
        __syncthreads();
        MARK(24);
        const int r0 = bid * RT_O / NB, r1 = (bid + 1) * RT_O / NB, R = r1 - r0;
        consume_tasks<0>(ttab, XSA, rg, pst[warp], [&](int slot, const float* c) { store_task(red, slot, c); });
#if EPLC
        static_assert(2 * 16 * MAXB <= NT, "one o_proj epilogue item per thread");
        float ebo = 0.f, exv = 0.f;
        if (tid < R * 16 * B) {
          const int r = tid / (16 * B), i = (tid / B) % 16, b = tid % B, row = (r0 + r) * 16 + i;
          ebo = Lp.bo[row];
          const bf16* xin = (layer == 0) ? p.embed + (size_t)ms->tok[b] * HID : p.x + b * HID;
          exv = u16f(ldcg16(xin + row));
        }
#endif
        __syncthreads();
        MARK(25);
#if RFOLD || NSS
        __shared__ float xo[MAXB][32];
#endif
#if RFOLD
        // track B: router weights folded with ln2 (fp32 products, exact): rf[row][e] = ln2[row] * Wr[e][row]
        float rfv[32];
        if (RFPRE && warp < B) {
          const float* rf = reinterpret_cast<const float*>(Lp.wr) + (size_t)r0 * 16 * 32 + lane;
#pragma unroll
          for (int j = 0; j < 32; j++) rfv[j] = j < R * 16 ? ldcgf(rf + j * 32) : 0.f;
        }
#endif
        for (int it = tid; it < R * 16 * B; it += NT) {
          int r = it / (16 * B), i = (it / B) % 16, b = it % B;
          float v = 0.f;
          for (int s = 0; s < NSEG_O; s++) v += red_at(red, r * NSEG_O + s, i, b);
          int row = (r0 + r) * 16 + i;
#if EPLC
          float o = bfr(v + ebo);
          const float xv = exv;
#else
          float o = bfr(v + Lp.bo[row]);
          const bf16* xin = (layer == 0) ? p.embed + (size_t)ms->tok[b] * HID : p.x + b * HID;
          float xv = u16f(ldcg16(xin + row));
#endif
          p.xm[b * HID + row] = __float2bfloat16_rn(xv + o);
#if RFOLD || NSS
          xo[b][r * 16 + i] = bfr(xv + o);
#endif
        }
#if NSS && !RFOLD
        __syncthreads();
        if (warp < B) {   // sum of squares of this block's rows of xm (sequence = warp) for the single-pass norm2
          const int b = warp;
          float q = lane < R * 16 ? xo[b][lane] * xo[b][lane] : 0.f;
#pragma unroll
          for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
          if (lane == 0) p.rlog[(size_t)MAXB * NB * NE + b * NB + bid] = q;
        }
#endif
#if RFOLD
        __syncthreads();
        if (warp < B) {   // partial router sums and sum of squares of this block's rows of xm (sequence = warp)
          const int b = warp;
          float S = 0.f;
          if (!RFPRE) {
            const float* rf = reinterpret_cast<const float*>(Lp.wr) + (size_t)r0 * 16 * 32 + lane;
#pragma unroll
            for (int j = 0; j < 32; j++) rfv[j] = j < R * 16 ? ldcgf(rf + j * 32) : 0.f;
          }
#pragma unroll
          for (int j = 0; j < 32; j++) if (j < R * 16) S = fmaf(rfv[j], xo[b][j], S);
          float q = lane < R * 16 ? xo[b][lane] * xo[b][lane] : 0.f;
#pragma unroll
          for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
          p.rlog[((size_t)b * NB + bid) * NE + lane] = S;
          if (lane == 0) p.rlog[(size_t)MAXB * NB * NE + b * NB + bid] = q;
        }
#endif
      }
#if !RFOLD
      if (bid < RKB) pf_router(layer);
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
      MARK(4);
      // ---------------- phase D: norm2 (all blocks), router (blocks NB-8 .. NB-1)
      for (int b = 0; b < MAXB; b++) rows_in[b] = p.xm + b * HID;
#if NSS
      NORM_SS(rows_in, B, Lp.ln2, xs, 0x1p63f, p.rlog + (size_t)MAXB * NB * NE, NB, NB);   // MoE input * 2^63
#else
      norm_to_smem(rows_in, B, Lp.ln2, xs, ms->rinv, false, 0x1p63f);   // MoE input * 2^63 (router undoes it)
#endif
      if (bid < RKB) {   // partial router logits of this block's K slice -> rlog[bid][b][e]
        consume_tasks<0>(ttab, XSH, rg, pst[warp], [&](int rt, const float* c) {
          float* rp = p.rlog + (size_t)bid * MAXB * NE + rt * 16;
#pragma unroll
          for (int j = 0; j < 2; j++) {
            const int b = 2 * t + j;
            if (b < B) { rp[b * NE + g] = c[j]; rp[b * NE + g + 8] = c[2 + j]; }
          }
        });
      }
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
      MARK(5);
#else
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
      MARK(5);
#endif
      // ---------------- phase E: top-4 + union (every block), gate_up
      {
        __shared__ int sel[MAXB][4];
        __shared__ float selw[MAXB][4];
        __shared__ u32 selm[MAXB];
#if RFOLD
        // warp NW-1-b: sequence b's router logits from the NB partial sums of phase C (ln2 folded into the router
        // weights): logit = rinv * sum + bias; meanwhile warps < B run norm2
        if (warp >= NW - B) {
          const int b = NW - 1 - warp;
          const float4* P4 = reinterpret_cast<const float4*>(p.rlog + (size_t)b * NB * NE);
          float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
          static_assert(NB * NE / 4 == 33 * 32, "33 float4 per lane");
#pragma unroll
          for (int h = 0; h < 3; h++) {
            float4 v[11];
#pragma unroll
            for (int j = 0; j < 11; j++) { const uint4 u = ldcg128(P4 + 32 * (11 * h + j) + lane);
              v[j] = make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
#pragma unroll
            for (int j = 0; j < 11; j++) { acc.x += v[j].x; acc.y += v[j].y; acc.z += v[j].z; acc.w += v[j].w; }
          }
          float qs = 0.f;
          const float* SS = p.rlog + (size_t)MAXB * NB * NE + b * NB;
#pragma unroll
          for (int k = 0; k < 5; k++) qs += (lane + 32 * k < NB) ? ldcgf(SS + lane + 32 * k) : 0.f;
#pragma unroll
          for (int o = 8; o < 32; o <<= 1) {
            acc.x += __shfl_xor_sync(0xffffffffu, acc.x, o); acc.y += __shfl_xor_sync(0xffffffffu, acc.y, o);
            acc.z += __shfl_xor_sync(0xffffffffu, acc.z, o); acc.w += __shfl_xor_sync(0xffffffffu, acc.w, o);
          }
#pragma unroll
          for (int o = 16; o; o >>= 1) qs += __shfl_xor_sync(0xffffffffu, qs, o);
          // lane l < 8 holds experts 4l..4l+3; lane e takes component e & 3 of lane e >> 2
          const float cx = __shfl_sync(0xffffffffu, acc.x, lane >> 2), cy = __shfl_sync(0xffffffffu, acc.y, lane >> 2);
          const float cz = __shfl_sync(0xffffffffu, acc.z, lane >> 2), cw = __shfl_sync(0xffffffffu, acc.w, lane >> 2);
          const int c4 = lane & 3;
          const float S = c4 == 0 ? cx : c4 == 1 ? cy : c4 == 2 ? cz : cw;
          const float ri = rsqrtf(qs / (float)HID + 1e-5f);
          float v0 = S * ri + Lp.br[lane];
#else
        // warp b: router logits of sequence b in registers (lane = expert; the RKB partials summed in slice order,
        // * 2^-63, + bias, bf16), top-4 by shuffles, softmax of the four
        if (warp < B) {
          const int b = warp;
          static_assert(NE == 32, "top-4: one expert per lane");
          float pv[RKB];
#pragma unroll
          for (int j = 0; j < RKB; j++) pv[j] = ldcgf(p.rlog + ((size_t)j * MAXB + b) * NE + lane);
          float v0 = 0.f;
#pragma unroll
          for (int j = 0; j < RKB; j++) v0 += pv[j];
          v0 = v0 * 0x1p-63f + Lp.br[lane];
#endif
#if ROUTER_BF16
          v0 = bfr(v0);
#endif
          float tv[4]; int ti[4];
#if TOPR
          // rank of each lane's logit (larger values first, lower index first among equals) from 31 independent
          // shuffles; the lanes of rank 0..3 are the same top-4 in the same order as four argmax rounds
          int rk = 0;
#pragma unroll
          for (int o = 1; o < 32; o++) {
            const int j = (lane + o) & 31;
            const float ov = __shfl_sync(0xffffffffu, v0, j);
            rk += (ov > v0 || (ov == v0 && j < lane)) ? 1 : 0;
          }
#pragma unroll
          for (int r = 0; r < 4; r++) {
            ti[r] = __ffs(__ballot_sync(0xffffffffu, rk == r)) - 1;
            tv[r] = __shfl_sync(0xffffffffu, v0, ti[r]);
          }
#else
#pragma unroll
          for (int r = 0; r < 4; r++) {
            float bv = v0; int bi = lane;
#pragma unroll
            for (int o = 16; o; o >>= 1) {
              float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
              if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            tv[r] = bv; ti[r] = bi;
            if (lane == bi) v0 = -INFINITY;
          }
#endif
          if (lane == 0) {
            float e[4], s = 0.f;
#pragma unroll
            for (int r = 0; r < 4; r++) { e[r] = expf(tv[r] - tv[0]); s += e[r]; }
            u32 m = 0;
#pragma unroll
            for (int r = 0; r < 4; r++) { sel[b][r] = ti[r]; selw[b][r] = e[r] / s; m |= 1u << ti[r]; }
            selm[b] = m;
          }
        }
#if RFOLD
        for (int b = 0; b < MAXB; b++) rows_in[b] = p.xm + b * HID;
#if NSS
        NORM_SS(rows_in, B, Lp.ln2, xs, 0x1p63f, p.rlog + (size_t)MAXB * NB * NE, NB, NB);
#else
        norm_to_smem(rows_in, B, Lp.ln2, xs, ms->rinv, false, 0x1p63f);   // MoE input * 2^63 (ends with a block sync)
#endif
#else
        __syncthreads();
#endif
        MARK(15);
        // union in ascending expert order (lane = expert): warp 0 slots and routing weights, warp 1 the per-sequence
        // slot lists, warp 2 the down phase's unit range and staged rows; each derives the union from selm itself
        if (warp < 3) {
          u32 m = 0;   // sequences that chose expert `lane`
          for (int b = 0; b < B; b++) m |= ((selm[b] >> lane) & 1u) << b;
          const u32 all = __ballot_sync(0xffffffffu, m != 0);
          if (warp == 0) {
            if (m) {
              const int slot = __popc(all & ((1u << lane) - 1));
              ms->sexp[slot] = lane; ms->smask[slot] = m;
              for (int b = 0; b < MAXB; b++) {
                float wv = 0.f;
                if (m >> b & 1) {
#pragma unroll
                  for (int r = 0; r < 4; r++) if (sel[b][r] == lane) wv = selw[b][r];
                }
                ms->rw[b][slot] = wv;
              }
            }
            if (lane == 0) ms->U = __popc(all);
          } else if (warp == 1) {
            if (lane < B) {   // sequence `lane`: its 4 slots in ascending order and their weights
              const int b = lane;
              u32 em = selm[b];
              int k = 0;
              while (em) {
                const int e = __ffs(em) - 1; em &= em - 1;
                float wv = 0.f;
#pragma unroll
                for (int r = 0; r < 4; r++) if (sel[b][r] == e) wv = selw[b][r];
                ms->bsl[b][k] = __popc(all & ((1u << e) - 1)); ms->bw[b][k] = wv; k++;
              }
            }
          } else {
            // slot k's sequence mask = m of the lane holding the k-th set bit of `all`
            const int N = __popc(all) * RT_DN * DSPLIT;
            const int u0 = (int)((long)bid * N / NB), u1 = (int)((long)(bid + 1) * N / NB);
            const int s0 = u1 > u0 ? u0 / (RT_DN * DSPLIT) : 0, s1 = u1 > u0 ? (u1 - 1) / (RT_DN * DSPLIT) + 1 : 0;
            const int myslot = m ? __popc(all & ((1u << lane) - 1)) : -1;
            const u32 bmask = (1u << B) - 1;
            if (lane == 0) { ms->fr_s0 = s0; ms->fr_s1 = s1; ms->fr_u0 = u0; ms->fr_u1 = u1; }
            int base = 0;
            for (int k = s0; k < s1; k++) {
              const u32 mk = __shfl_sync(0xffffffffu, m, __fns(all, 0, k + 1)) & bmask;
              (void)myslot;
              if (lane == 0) {
                u64 r8 = 0;
                for (int b = 0; b < 8; b++) if (mk >> b & 1) { ms->prs[base] = (unsigned char)(((k - s0) << 3) | b); r8 |= (u64)(base++) << (8 * b); }
                ms->srows[k - s0] = r8;
              }
              base = __shfl_sync(0xffffffffu, base, 0);
            }
            if (lane == 0) ms->fr_np = base;
          }
        }
        __syncthreads();
        MARK(14);
        const int U = ms->U;
#if DBPF
        // the down combine's expert bias rows (16 floats per slot) for the rowtiles of this block's down units: into L2
        // now, while gate_up runs (the combining block is whichever arrives last)
        if (warp == NW - 1 && B <= 2) {   // (B = 8: +8 us, e0286)
          const int u0 = ms->fr_u0, u1 = ms->fr_u1;
          for (int x = lane; x < (u1 - u0) * U; x += 32) {
            const int uu = u0 + x / U, sl = x % U, rt = uu % RT_DN;
            asm volatile("prefetch.global.L2 [%0];" :: "l"(Lp.dn_b + (size_t)ms->sexp[sl] * HID + rt * 16));
          }
        }
#endif
        MARK(12);
#if defined(DBGT) && DBGT_GU
        if (tid == 0) { g_tr = reinterpret_cast<unsigned long long*>(const_cast<unsigned char*>(p.skel)); g_trk[bid] = (layer == 5 && st == total - 1) ? 1 : 0;
                        if (g_trk[bid]) g_tr[bid * 64] = clock64(); }
#endif
        {
          // gate_up: units (slot, rowtile) of U * 360, contiguous per block, one warp task per unit (whole K);
          // gate (even rows) and up (odd rows) of a pair sit in lanes g, g+1: swiglu in the warp, no block sync
          const int N = U * RT_GU;
#if GURR
          // units dealt round-robin (block b: b, b + NB, ...): every block reads rows of all experts and regions
          const int nmine = N > bid ? (N - bid + NB - 1) / NB : 0;
          run_tasks<1>(nmine, ttab, XSH, rg, pst[warp], [&](int task, TD& d) {
            int uu = bid + NB * task, slot = uu / RT_GU, rt = uu % RT_GU, e = ms->sexp[slot];
#else
          const int u0 = (int)((long)bid * N / NB), u1 = (int)((long)(bid + 1) * N / NB);
          run_tasks<1>(u1 - u0, ttab, XSH, rg, pst[warp], [&](int task, TD& d) {
            int uu = u0 + task, slot = uu / RT_GU, rt = uu % RT_GU, e = ms->sexp[slot];
#endif
            d.w = Lp.gu_c + ((size_t)e * RT_GU + rt) * KC * (FUSE ? 34 : 32);
            d.s = Lp.gu_s + ((size_t)e * RT_GU + rt) * KC * 8;
            d.x = xs; d.k0 = 0; d.n = KC; d.slot = uu; d.bm = (1u << B) - 1;
          }, [&](int uu, const float* c) {
            const int slot = uu / RT_GU, rt = uu % RT_GU, e = ms->sexp[slot];
            float up[4];
#pragma unroll
            for (int k = 0; k < 4; k++) up[k] = __shfl_down_sync(0xffffffffu, c[k], 4);
            if (!(g & 1)) {
              const u32 m = ms->smask[slot];
              const float* bb = Lp.gu_b + (size_t)e * 5760 + rt * 16;
#pragma unroll
              for (int h = 0; h < 2; h++) {           // rows g (h = 0) and g + 8 (h = 1): pair (g + 8h) / 2
                const float gbias = bb[g + 8 * h], ubias = bb[g + 8 * h + 1];
#pragma unroll
                for (int j = 0; j < 2; j++) {
                  const int b = 2 * t + j;
                  if (b < B && (m >> b & 1)) {
                    float gt = c[2 * h + j] + gbias, u = up[2 * h + j] + ubias;
                    gt = fminf(gt, 7.0f);
                    u = fmaxf(fminf(u, 7.0f), -7.0f);
                    float glu = gt * (1.0f / (1.0f + expf(-1.702f * gt)));
                    float o = (u + 1.0f) * glu;
                    p.act[((size_t)slot * MAXB + b) * HID + rt * 8 + (g + 8 * h) / 2] = __float2bfloat16_rn(o * 0x1p63f);
                  }
                }
              }
            }
          });
        }
        MARK(18);
#if defined(DBGT) && DBGT_GU
        if (tid == 0) g_trk[bid] = 0;
#endif
        pf_down();   // the down phase's first stages (its unit table was made with the routing)
      }
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
      MARK(6);
#if SVPF
      // the next layer's small vectors (norm weights, biases, sinks; ~55 KB) into L2 while the down phase streams:
      // their first reads come after barriers (every block reads them at once)
      if (tid == 0 && bid >= NB - 6) {
        const int nl = layer + 1 < NL ? layer + 1 : 0;
        const LayerP& Ln = p.L[nl];
        const int which = NB - 1 - bid;
        if (which == 0) l2pf(Ln.ln1, HID * 4);
        else if (which == 1) l2pf(Ln.bqkv, 5120 * 4);
        else if (which == 2) { l2pf(Ln.bo, HID * 4); l2pf(Ln.sinks, 64 * 4); }
        else if (which == 3) { l2pf(Ln.ln2, HID * 4); l2pf(Ln.br, NE * 4); }
        else if (which == 4 && layer + 1 == NL) l2pf(p.lnf, HID * 4);
      }
#endif
      // ---------------- phase F: down units (slot, rowtile); y = W x + bias to ypart; the last unit of a rowtile
      // (counter) sums its rows over the slots in slot order, adds the residual and writes x
      {
        const int s0 = ms->fr_s0, s1 = ms->fr_s1, np = ms->fr_np;
        const int U = ms->U;
        u16* xa = xs;   // up to 16 rows of XSH: runs on into red (unused here)
        for (int it = tid; it < np * (HID / 8); it += NT) {
          const int pr = it / (HID / 8), k = (it % (HID / 8)) * 8;
          const int sl = s0 + (ms->prs[pr] >> 3), b = ms->prs[pr] & 7;
          uint4 v = ldcg128(p.act + ((size_t)sl * MAXB + b) * HID + k);
          xfo_store8(xa + pr * XSH, k, v);
        }
        __syncthreads();
        MARK(16);
#if defined(DBGT) && !DBGT_GU
        if (tid == 0) { g_tr = reinterpret_cast<unsigned long long*>(const_cast<unsigned char*>(p.skel)); g_trk[bid] = (layer == 5 && st == total - 1) ? 1 : 0;
                        if (g_trk[bid]) g_tr[bid * 64] = clock64(); }
#endif
        consume_tasks<1>(ttab, XSH, rg, pst[warp], [&](int uu, const float* c) {
          const int slot = uu / (RT_DN * DSPLIT), rt = DORD ? (uu % RT_DN) : (uu / DSPLIT) % RT_DN, h = DORD ? (uu / RT_DN) % DSPLIT : uu % DSPLIT;
          const u32 m = ms->smask[slot] & ((1u << B) - 1);
          float* yp = p.ypart + ((size_t)(slot * DSPLIT + h) * MAXB) * HID + rt * 16;   // [slot][half][b][row]
#pragma unroll
          for (int j = 0; j < 2; j++) {
            const int b = 2 * t + j;
            if (m >> b & 1) { yp[b * HID + g] = c[j]; yp[b * HID + g + 8] = c[2 + j]; }
          }
#ifndef FENCECNT
          // one acq_rel atomic by lane 0 after the warp barrier: release is cumulative over the warp's stores
          __syncwarp();
          int last = 0;
          if (lane == 0) {
            u32 old;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p.dcnt + rt) : "memory");
            last = (old == (u32)(U * DSPLIT - 1));
          }
          last = __shfl_sync(0xffffffffu, last, 0);
          if (last) {
#else
          __threadfence();
          __syncwarp();
          int last = 0;
          if (lane == 0) last = (atomicAdd(p.dcnt + rt, 1u) == (u32)(U * DSPLIT - 1));
          last = __shfl_sync(0xffffffffu, last, 0);
          if (last) {
            __threadfence();
#endif
#if NSS
            // track B: rows in lane order (b = it / 16), the 16 rows' sum of squares per sequence for the next norm
            for (int it0 = 0; it0 < 16 * B; it0 += 32) {
              const int it = it0 + lane;
              const bool ok = it < 16 * B;
              const int i = it & 15, b = ok ? it >> 4 : 0, row = rt * 16 + i;
#else
            for (int it = lane; it < 16 * B; it += 32) {
              const int i = it / B, b = it % B, row = rt * 16 + i;
#endif
              float y0[4], y1[4], bb[4];
#pragma unroll
              for (int r = 0; r < 4; r++) {     // all loads in flight, then the sum in slot order
                const int sl = ms->bsl[b][r];
                const float* yq = p.ypart + ((size_t)sl * DSPLIT * MAXB + b) * HID + row;
                y0[r] = ldcgf(yq); y1[r] = (DSPLIT == 2) ? ldcgf(yq + MAXB * HID) : 0.f;
                bb[r] = Lp.dn_b[(size_t)ms->sexp[sl] * HID + row];
              }
              float acc = 0.f;
#pragma unroll
              for (int r = 0; r < 4; r++) acc = fmaf(ms->bw[b][r], (DSPLIT == 2 ? y0[r] + y1[r] : y0[r]) + bb[r], acc);
              const float xv = u16f(ldcg16(p.xm + b * HID + row));
#if NSS
              const float xo1 = bfr(xv + bfr(acc));
              if (ok) p.x[b * HID + row] = __float2bfloat16_rn(xo1);
              float q = ok ? xo1 * xo1 : 0.f;
#pragma unroll
              for (int o = 1; o < 16; o <<= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
              if (ok && i == 0) p.rlog[(size_t)MAXB * NB * (NE + 1) + b * RT_DN + rt] = q;
#else
              p.x[b * HID + row] = __float2bfloat16_rn(xv + bfr(acc));
#endif
            }
            if (lane == 0) p.dcnt[rt] = 0;
          }
        });
      }
#ifdef DBGT
      if (tid == 0) g_trk[bid] = 0;
#endif
      if (layer < NL - 1) pf_qkv(layer + 1);
      else if (st >= P - 1) {
        if (!sketch) pf_head();
#if SKPF0
        // track B: the first SKPF0 load batches of each warp's first sketch tile into L2 (no registers)
        else if (lane == 0) {
          const int r0h = (int)((long)bid * RT_HEAD / NB), r1h = (int)((long)(bid + 1) * RT_HEAD / NB);
          if (r0h + warp < r1h) l2pf(p.sk + (size_t)(r0h + warp) * SKT, SKPF0 * (SK5 ? SKQP * 672 : SKQ * 384));
        }
#endif
      }
      else if (st + 1 < total) pf_qkv(0);
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
    }  // layers
    // ---------------- head
    const bool need_head = st >= P - 1;
    if (need_head) {
      MARK(7);
      for (int b = 0; b < MAXB; b++) rows_in[b] = p.x + b * HID;
#if NSS
      NORM_SS(rows_in, B, p.lnf, xs, 1.f, p.rlog + (size_t)MAXB * NB * (NE + 1), RT_DN, RT_DN);
#else
      norm_to_smem(rows_in, B, p.lnf, xs, ms->rinv);
#endif
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
      if (!sketch) {
        consume_tasks<0>(ttab, XSH, rg, pst[warp], head_fin);
      } else {
        // ---- track B: certified int8 sketch (pass 1), one exchange of the best lower bound, exact recompute of the
        // tiles whose upper bound reaches it (pass 2) with the arithmetic of the exact head (same bits as track A).
        unsigned char* skm = smem + SM_RING;                    // the ring holds no stages during the head
        signed char* dh = reinterpret_cast<signed char*>(skm);  // [8][DSTR] hi digits, then lo digits
        signed char* dl = dh + 8 * DSTR;
        float* sxn = reinterpret_cast<float*>(skm + 16 * DSTR); // t[8], X[8] >= ||x||, R[8] >= ||r||
        if (warp < B) {
          const u16* xr = xs + warp * XSH;
          float amax = 0.f, ss = 0.f;
          for (int k = lane; k < HID; k += 32) { const float v = u16f(xr[xfo(k)]); amax = fmaxf(amax, fabsf(v)); ss = fmaf(v, v, ss); }
#pragma unroll
          for (int o = 16; o; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); ss += __shfl_xor_sync(0xffffffffu, ss, o); }
          const float tq = fmaxf(amax, 1e-30f) * (1.0f / 16256.0f), it = 1.0f / tq;
          float rr = 0.f;
          for (int k = lane; k < HID; k += 32) {
            const float v = u16f(xr[xfo(k)]);
            const int P = max(-16256, min(16256, __float2int_rn(v * it)));
            const int lo = ((P + 64) & 127) - 64, hi = (P - lo) >> 7;   // P = 128 hi + lo, both int8
            const float r = v - tq * (float)P;
            rr = fmaf(r, r, rr);
#if DPK
            dh[k] = (signed char)hi; dh[DSTR + k] = (signed char)lo;   // B = 1: hi digits in column 0, lo in column 1
#else
            dh[warp * DSTR + k] = (signed char)hi; dl[warp * DSTR + k] = (signed char)lo;
#endif
          }
#pragma unroll
          for (int o = 16; o; o >>= 1) rr += __shfl_xor_sync(0xffffffffu, rr, o);
          if (lane == 0) {
            const float xn = sqrtf(ss) * 1.001f + 1e-30f;
            sxn[warp] = tq; sxn[8 + warp] = xn; sxn[16 + warp] = sqrtf(rr) * 1.001f + xn * 0x1p-20f;
          }
        }
        __syncthreads();
        const int r0 = (int)((long)bid * RT_HEAD / NB), r1 = (int)((long)(bid + 1) * RT_HEAD / NB);
        const int ca = 2 * t, cb = 2 * t + 1;
        const float ta = ca < B ? sxn[ca] : 0.f, tb = cb < B ? sxn[cb] : 0.f;
        const float Xa = ca < B ? sxn[8 + ca] : 0.f, Xb = cb < B ? sxn[8 + cb] : 0.f;
        const float Ra = ca < B ? sxn[16 + ca] : 0.f, Rb = cb < B ? sxn[16 + cb] : 0.f;
        float2* UT = reinterpret_cast<float2*>(red);              // [NW][8 tiles][32 lanes] upper bounds (columns ca, cb)
        float lwa = -INFINITY, lwb = -INFINITY;
        const u32* dh32 = reinterpret_cast<const u32*>(dh + g * DSTR) + t;
        const u32* dl32 = reinterpret_cast<const u32*>(dl + g * DSTR) + t;
        int n = 0;
#if SKEF
        unsigned long long skpol;
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(skpol));
#endif
        for (int i = r0 + warp; i < r1; i += NW, n++) {
          const u32* tp = reinterpret_cast<const u32*>(p.sk + (size_t)i * SKT) + lane;
#if SK5
          // 5-bit codes, group (k32-step) scales m: f = sum over steps of m * (128 h + l), h / l the integer dots of
          // 8q with the hi / lo digits (exact in int32); the float sum's error is in the header's en (2^-15 qn)
          float f[4] = {0.f, 0.f, 0.f, 0.f};
          const u32* sp = reinterpret_cast<const u32*>(p.sk + (size_t)i * SKT) + 160 + g;
#if SKDB
          // double-buffered batches of SKQP pairs: batch k+1's loads are in flight during batch k's math (same order)
          u32 bufA[SKQP][6], bufB[SKQP][6];
          auto ldb = [&](u32 (&w)[SKQP][6], int pq) {
#pragma unroll
            for (int u = 0; u < SKQP; u++) {
#pragma unroll
              for (int v = 0; v < 5; v++) w[u][v] = ldw32h(tp + (pq + u) * 168 + v * 32, skpol);
              w[u][5] = ldw32h(sp + (pq + u) * 168, skpol);
            }
          };
          auto cmp = [&](const u32 (&w)[SKQP][6], int pq) {
#pragma unroll
            for (int u = 0; u < SKQP; u++) {
#pragma unroll
              for (int mm = 0; mm < 2; mm++) {
                const int ks = 2 * (pq + u) + mm;
                uint4 av;
                av.x = dec5(w[u], 4 * mm + 0); av.y = dec5(w[u], 4 * mm + 1);
                av.z = dec5(w[u], 4 * mm + 2); av.w = dec5(w[u], 4 * mm + 3);
                const float m0 = (float)((w[u][5] >> (16 * mm)) & 0xFFu), m1 = (float)((w[u][5] >> (16 * mm + 8)) & 0xFFu);
                int h[4] = {0, 0, 0, 0};
                mma8(h, av, dh32[ks * 8], dh32[ks * 8 + 4]);
                f[0] = fmaf(m0, (float)(h[0] * 128 + h[1]), f[0]);
                f[2] = fmaf(m1, (float)(h[2] * 128 + h[3]), f[2]);
              }
            }
          };
          constexpr int NBT = (SKS / 2) / SKQP;
          ldb(bufA, 0);
#pragma unroll 1
          for (int k = 0; k + 1 < NBT; k += 2) {
            ldb(bufB, (k + 1) * SKQP);
            cmp(bufA, k * SKQP);
            if (k + 2 < NBT) ldb(bufA, (k + 2) * SKQP);
            cmp(bufB, (k + 1) * SKQP);
          }
          if (NBT % 2) cmp(bufA, (NBT - 1) * SKQP);
#else
#pragma unroll 1
          for (int pq = 0; pq < SKS / 2; pq += SKQP) {
            u32 wq[SKQP][5], ws[SKQP];
#pragma unroll
            for (int u = 0; u < SKQP; u++) {
#pragma unroll
              for (int v = 0; v < 5; v++) wq[u][v] = ldw32h(tp + (pq + u) * 168 + v * 32, skpol);
              ws[u] = ldw32h(sp + (pq + u) * 168, skpol);
            }
#pragma unroll
            for (int u = 0; u < SKQP; u++) {
#pragma unroll
              for (int mm = 0; mm < 2; mm++) {
                const int ks = 2 * (pq + u) + mm;
                uint4 av;
                av.x = dec5(wq[u], 4 * mm + 0); av.y = dec5(wq[u], 4 * mm + 1);
                av.z = dec5(wq[u], 4 * mm + 2); av.w = dec5(wq[u], 4 * mm + 3);
                const float m0 = (float)((ws[u] >> (16 * mm)) & 0xFFu), m1 = (float)((ws[u] >> (16 * mm + 8)) & 0xFFu);
#if DPK
                // one mma: column 0 = hi digits, column 1 = lo digits (same integers as two mmas)
                int h[4] = {0, 0, 0, 0};
                mma8(h, av, dh32[ks * 8], dh32[ks * 8 + 4]);
                f[0] = fmaf(m0, (float)(h[0] * 128 + h[1]), f[0]);
                f[2] = fmaf(m1, (float)(h[2] * 128 + h[3]), f[2]);
#else
                int h[4] = {0, 0, 0, 0}, l[4] = {0, 0, 0, 0};
                mma8(h, av, dh32[ks * 8], dh32[ks * 8 + 4]);
                mma8(l, av, dl32[ks * 8], dl32[ks * 8 + 4]);
                f[0] = fmaf(m0, (float)(h[0] * 128 + l[0]), f[0]);
                f[1] = fmaf(m0, (float)(h[1] * 128 + l[1]), f[1]);
                f[2] = fmaf(m1, (float)(h[2] * 128 + l[2]), f[2]);
                f[3] = fmaf(m1, (float)(h[3] * 128 + l[3]), f[3]);
#endif
              }
            }
          }
#endif
#else
          int h[4] = {0, 0, 0, 0}, l[4] = {0, 0, 0, 0};
#pragma unroll 1
          for (int m = 0; m < SKS; m += SKQ) {
            u32 w0[SKQ], w1[SKQ], w2[SKQ];
#pragma unroll
#if SKEF
            for (int u = 0; u < SKQ; u++) { w0[u] = ldw32h(tp + (m + u) * 96, skpol); w1[u] = ldw32h(tp + (m + u) * 96 + 32, skpol); w2[u] = ldw32h(tp + (m + u) * 96 + 64, skpol); }
#else
            for (int u = 0; u < SKQ; u++) { w0[u] = ldw32(tp + (m + u) * 96); w1[u] = ldw32(tp + (m + u) * 96 + 32); w2[u] = ldw32(tp + (m + u) * 96 + 64); }
#endif
#pragma unroll
            for (int u = 0; u < SKQ; u++) {
              const int mm = m + u;
              uint4 av;
              av.x = up6(w0[u]); av.y = up6(w1[u]); av.z = up6(w2[u]);
              av.w = up6(prmt(prmt(w0[u], w1[u], 0x0073), w2[u], 0x0710));
              mma8(h, av, dh32[mm * 8], dh32[mm * 8 + 4]);
#if !DPK
              mma8(l, av, dl32[mm * 8], dl32[mm * 8 + 4]);
#endif
            }
          }
#endif
          const float* hd = reinterpret_cast<const float*>(p.sk + (size_t)i * SKT + SKHDR) + g * 8;
          const float sa = hd[0], sb = hd[1], ea = hd[2], eb = hd[3], qa = hd[4], qb = hd[5];
          // c0: (row g, column ca), c1: (row g, cb), c2: (row g + 8, ca), c3: (row g + 8, cb)
          // the int8 operands hold 4 q: products are 4x; the 0.25 is exact
#if SK5
          // the int8 operands hold 8 q: the 0.125 is exact
          const float s00 = sa * (ta * (f[0] * 0.125f)), s01 = sa * (tb * (f[1] * 0.125f));
          const float s10 = sb * (ta * (f[2] * 0.125f)), s11 = sb * (tb * (f[3] * 0.125f));
#elif DPK
          const float s00 = sa * (ta * ((float)((long long)h[0] * 128 + h[1]) * 0.25f)), s01 = 0.f;
          const float s10 = sb * (ta * ((float)((long long)h[2] * 128 + h[3]) * 0.25f)), s11 = 0.f;
          (void)l;
#else
          const float s00 = sa * (ta * ((float)((long long)h[0] * 128 + l[0]) * 0.25f)), s01 = sa * (tb * ((float)((long long)h[1] * 128 + l[1]) * 0.25f));
          const float s10 = sb * (ta * ((float)((long long)h[2] * 128 + l[2]) * 0.25f)), s11 = sb * (tb * ((float)((long long)h[3] * 128 + l[3]) * 0.25f));
#endif
          const float e00 = (ea * Xa + qa * Ra) * 1.0001f + fabsf(s00) * 0x1p-20f, e01 = (ea * Xb + qa * Rb) * 1.0001f + fabsf(s01) * 0x1p-20f;
          const float e10 = (eb * Xa + qb * Ra) * 1.0001f + fabsf(s10) * 0x1p-20f, e11 = (eb * Xb + qb * Rb) * 1.0001f + fabsf(s11) * 0x1p-20f;
          lwa = fmaxf(lwa, fmaxf(s00 - e00, s10 - e10)); lwb = fmaxf(lwb, fmaxf(s01 - e01, s11 - e11));
          UT[(warp * 8 + n) * 32 + lane] = make_float2(ca < B ? fmaxf(s00 + e00, s10 + e10) : -INFINITY, cb < B ? fmaxf(s01 + e01, s11 + e11) : -INFINITY);
        }
        // best lower bound per column: warp (over g), block, then all blocks (one grid barrier)
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) { lwa = fmaxf(lwa, __shfl_xor_sync(0xffffffffu, lwa, o)); lwb = fmaxf(lwb, __shfl_xor_sync(0xffffffffu, lwb, o)); }
        if (g == 0) { ms->tmpf[warp * 8 + ca] = lwa; ms->tmpf[warp * 8 + cb] = lwb; }
        __syncthreads();
        if (tid < B) {
          float v = -INFINITY;
          for (int w = 0; w < NW; w++) v = fmaxf(v, ms->tmpf[w * 8 + tid]);
          p.gth[bid * MAXB + tid] = v;
        }
        MARK(MKPROF_WAIT);
        grid_bar(bar, bar_target);
        MARK(19);
        if (warp < B) {
          float v = -INFINITY;
          for (int j = lane; j < NB; j += 32) v = fmaxf(v, ldcgf(p.gth + j * MAXB + warp));
#pragma unroll
          for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
          if (lane == 0) ms->rlg[warp] = v;
        }
        __syncthreads();
        const float Ta = ca < B ? ms->rlg[ca] : INFINITY, Tb = cb < B ? ms->rlg[cb] : INFINITY;
        u32 mk = 0;
        n = 0;
        for (int i = r0 + warp; i < r1; i += NW, n++) {
          const float2 ub = UT[(warp * 8 + n) * 32 + lane];
          mk |= (__any_sync(0xffffffffu, (ub.x >= Ta) | (ub.y >= Tb)) ? 1u : 0u) << n;
        }
        if (lane == 0) ms->tmpi[warp] = (int)mk;
        __syncthreads();
#ifdef SKSTAT
        if (tid == 0) { int nc = 0; for (int w = 0; w < NW; w++) nc += __popc((u32)ms->tmpi[w]); atomicAdd(p.bar + 8, (u32)nc); atomicMax(p.bar + 9, (u32)nc); atomicAdd(p.bar + 10, 1u); }
#endif
        // candidate tiles go to a global list (counter per head parity), then every block recomputes list[bid + k NB]
        u32* const ccnt = p.bar + 12 + (hstep & 1);
        int* const clist = p.clist + (hstep & 1) * RT_HEAD;
        if (tid == 0) {
          int nc = 0;
          for (int w = 0; w < NW; w++) nc += __popc((u32)ms->tmpi[w]);
          if (nc) {
            int base = (int)atomicAdd(ccnt, (u32)nc);
            for (int w = 0; w < NW; w++) {
              u32 m = (u32)ms->tmpi[w];
              while (m) { const int nn = __ffs(m) - 1; m &= m - 1; clist[base++] = r0 + w + nn * NW; }
            }
          }
          if (bid == 0) p.bar[12 + ((hstep + 1) & 1)] = 0;   // the other parity's counter (last read two heads ago)
        }
        MARK(MKPROF_WAIT);
        grid_bar(bar, bar_target);
        MARK(20);
        const int ncand = (int)ldcg32(ccnt);
        uint4* FR = reinterpret_cast<uint4*>(smem + SM_RING);   // [180 ksteps][32 lanes] one exact tile (digits are dead)
#if P2TMA
        u32* const p2n = reinterpret_cast<u32*>(smem + SM_BARS + NW * RS * 8 + 8 * P2CH);
        const u32 p2bar = smaddr(smem + SM_BARS + NW * RS * 8), ph0 = *p2n;
        int nmy = 0;
#endif
        for (int jj = bid; jj < ncand; jj += NB) {
          const int i = ldcgi(clist + jj);
          const uint4* hp = p.head + (size_t)i * KS_H * 32;
#if P2TMA
          // the whole 92 KB tile by one bulk copy (no registers, all bytes in flight)
          if (tid == 0) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#if P2CH > 1
            // in P2CH chunks, one mbarrier each: the chain starts on chunk 0 while the rest arrives
            for (int c = 0; c < P2CH; c++) {
              mbar_expect(p2bar + 8 * c, KS_H * 32 * 16 / P2CH);
              tma_load(smaddr(FR) + c * (KS_H * 32 * 16 / P2CH), hp + c * (KS_H * 32 / P2CH), KS_H * 32 * 16 / P2CH, p2bar + 8 * c);
            }
#else
            mbar_expect(p2bar, KS_H * 32 * 16);
            tma_load(smaddr(FR), hp, KS_H * 32 * 16, p2bar);
#endif
          }
#if P2CH == 1
          mbar_wait(p2bar, (ph0 + nmy) & 1);
#endif
          const u32 p2ph = (ph0 + nmy) & 1;
          nmy++;
#else
          for (int k = tid; k < KS_H * 32; k += NT) FR[k] = ldw(hp + k);
          __syncthreads();
#endif
          MARK(27);
          if (warp == 0) {   // the exact head's chain: 180 k-steps in order from zero
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            const u32 xb = smaddr(xs + g * XSH) + t * 32;
#if P2TMA && P2CH > 1
            for (int ch = 0; ch < P2CH; ch++) {
              mbar_wait(p2bar + 8 * ch, p2ph);
#pragma unroll 4
              for (int k = ch * (KS_H / P2CH); k < (ch + 1) * (KS_H / P2CH); k++) {
                const uint4 q = FR[k * 32 + lane];
                u32 b0, b1;
                asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(xb + (k >> 2) * 128 + (k & 3) * 8));
                mma16816(c, q.x, q.y, q.z, q.w, b0, b1);
              }
            }
#else
#pragma unroll 4
            for (int k = 0; k < KS_H; k++) {
              const uint4 q = FR[k * 32 + lane];
              u32 b0, b1;
              asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(xb + (k >> 2) * 128 + (k & 3) * 8));
              mma16816(c, q.x, q.y, q.z, q.w, b0, b1);
            }
#endif
            head_fin(i, c);
          }
          __syncthreads();
          MARK(28);
        }
#if P2TMA
        if (tid == 0) *p2n = ph0 + nmy;
#endif
      }
      MARK(29);
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
      if (st + 1 < total) pf_qkv(0);
      hstep++;
      MARK(MKPROF_WAIT);
      grid_bar(bar, bar_target);
    }
  }  // steps
  // the last token chosen: reduce once more and record
  if (a.mode != 0) {
    if (warp < B && bid == 0) {
      const int bi = argmax_blocks(p, warp, lane);
      if (lane == 0) {
        if (a.mode == 1) a.out_tok[warp * a.nsteps + a.nsteps - 1] = bi;
        else a.host_tok[warp * a.nsteps + a.nsteps - 1] = bi;
      }
    }
    __syncthreads();
    if (a.mode == 2 && bid == 0 && tid == 0) { __threadfence_system(); *a.host_prog = a.nsteps; }
  }
  // reset the barrier counter for the next launch: the last block out
  if (tid == 0) {
    __threadfence();
    u32 v = atomicAdd(bar + 1, 1u);
    if (v == NB - 1) { bar[0] = 0; bar[1] = 0; __threadfence(); }
  }
}

// ------------------------------------------------------------------------------------------------ host side
struct Engine {
  Params p;
  int* host_tok; int* host_prog; int* dev_host_tok; int* dev_host_prog; int* host_err; int* dev_host_err;
  int ntok_cap;
  cudaStream_t stream;
};

extern "C" void* mk_create(const void* params) {
  Engine* e = new Engine;
  memcpy(&e->p, params, sizeof(Params));
  if (cudaFuncSetAttribute(mk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL) != cudaSuccess) return nullptr;
  e->ntok_cap = MAXB * 8192;
  cudaHostAlloc(&e->host_tok, e->ntok_cap * sizeof(int), cudaHostAllocMapped);
  cudaHostAlloc(&e->host_prog, 64, cudaHostAllocMapped);
  cudaHostGetDevicePointer(&e->dev_host_tok, e->host_tok, 0);
  cudaHostGetDevicePointer(&e->dev_host_prog, e->host_prog, 0);
  cudaHostAlloc(&e->host_err, 64, cudaHostAllocMapped); *e->host_err = 0;
  cudaHostGetDevicePointer(&e->dev_host_err, e->host_err, 0);
  e->stream = 0;
  return e;
}
extern "C" int mk_params_size() { return (int)sizeof(Params); }
extern "C" int mk_rfold() { return RFOLD; }
extern "C" int mk_fuse() { return FUSE; }
extern "C" int mk_take_err(void* h) { Engine* e = (Engine*)h; int v = *(volatile int*)e->host_err; *(volatile int*)e->host_err = 0; return v; }
extern "C" int mk_layer_size() { return (int)sizeof(LayerP); }

extern "C" int mk_run(void* h, const void* tok, int tok_stride, void* logits, void* out_tok, int B, int pos0, int nprompt, int nsteps, int mode, void* stream, unsigned hstep0) {
  Engine* e = (Engine*)h;
  RunArgs a;
  a.tok = (const long long*)tok; a.tok_stride = tok_stride; a.logits = (float*)logits; a.out_tok = (long long*)out_tok;
  a.host_tok = e->dev_host_tok; a.host_prog = e->dev_host_prog; a.err = e->dev_host_err; a.B = B; a.pos0 = pos0; a.nprompt = nprompt; a.nsteps = nsteps; a.mode = mode; a.hstep0 = hstep0;
  if (mode == 2) { *(volatile int*)e->host_prog = 0; if (B * nsteps > e->ntok_cap) return -3; }
  void* args[] = {(void*)&e->p, (void*)&a};
  cudaError_t err = cudaLaunchCooperativeKernel((void*)mk_kernel, dim3(NB), dim3(NT), args, SM_TOTAL, (cudaStream_t)stream);
  if (err != cudaSuccess) { fprintf(stderr, "mk_run: %s\n", cudaGetErrorString(err)); return -1; }
  return 0;
}
extern "C" int mk_poll(void* h, int* buf, int B, int nsteps) {
  Engine* e = (Engine*)h;
  int done = *(volatile int*)e->host_prog;
  if (done > 0) memcpy(buf, e->host_tok, sizeof(int) * B * nsteps);
  return done;
}
