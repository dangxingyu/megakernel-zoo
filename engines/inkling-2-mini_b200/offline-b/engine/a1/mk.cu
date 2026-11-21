// Inkling-2-Mini decode megakernel for one B200 (sm_100a). One persistent cooperative kernel per call
// (152 blocks x 256 threads, one block per SM); a call runs one or more decode steps of B <= 8 sequences.
//
// Track A: weights as stored (BF16; NVFP4 experts = E2M1 codes x E4M3 scale per 16 x per-expert fp32 factor;
// MXFP8 experts = E4M3 x UE8M0 per 32), re-laid at load time (a permutation of the stored bytes). Expert inputs are
// NOT quantized: the weights are decoded exactly to BF16 in registers (code x block scale is exact in BF16) and
// multiplied with the BF16 activations on the tensor cores (fp32 accumulation); the NVFP4 per-expert factor is
// applied in fp32 to the sum. Tensors that the specification materialises in BF16 are BF16 here as well.
//
// Every matrix-vector product is a chain of mma.m16n8k16 (bf16 x bf16 -> f32); the 8 columns of the B operand are the
// sequences. Weights stream through per-warp rings in shared memory filled by bulk copies (cp.async.bulk + mbarrier).
//
// Per layer application (grid barrier between phases):
//   [M]  start of loop passes 2, 3: the 48 messages of the pass (attention over depth of the previous pass)
//   [A0] loop passes 2,3: attention-input mixes; rank-64 gate inputs z of the q/k/v message gates
//   [A1] loop passes 2,3: gated inputs x' = (1 - tanh(F z)) x + tanh(G z) m, a slice per block
//   A    q/k/v input norms (every block, into smem) + one GEMV for q|r|gate|sink|k|(dk)|v|(dv)
//   B    attention: (sequence, kv head, 64-key chunk) items; K/V conv + clip/norm of the new position; partials
//        merged by the last-arriving chunk; output gate, sink
//   C    o_proj + attention short conv -> attn_s
//   [D0, D1] loop passes 2,3: the MLP-input message gate as A0/A1
//   D    MLP-input mix + norm (every block, smem) ; router GEMV on blocks 0..7
//   E    top-8 (every block), union of experts, w13 of the union (exact NVFP4/MXFP8 decode), activation
//   F    w2 per owned rowtile over all chosen experts, weighted sum, MLP short conv, new partial sum
// Head: final mix + norm, unembedding GEMV, 30 tanh(z / 30) -> logits (or the device argmax).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include "mkprof.cuh"

typedef unsigned u32;
typedef unsigned short u16;
typedef unsigned long long u64;
typedef __nv_bfloat16 bf16;

#define H 2304
#define NL 24
#define NAPP 48
#define NQH 24
#define NKV 8
#define HD 128
#define QW 3072
#define KVW 1024
#define DREL 16
#define RELEXT 1024
#define WIN 1024
#define NE 128
#define TOPK 8
#define IM 1152
#define VOCAB 200058
#define RT_HEAD 12504
#define NB 152
#ifndef NT
#define NT 256
#endif
#define NW (NT / 32)
#define MAXB 8
#define MAXU 64
#define KS_H 144
#define KS_A 192
#define XSH (H + 8)
#define XSA (QW + 8)
#define XSI (IM + 8)
#define CH 64
#define MAXCH 64
#define NVEC 9
#define V_H0 0
#define V_P0 1
#define V_P1 2
#define V_P2 5
#define V_P3 8
#define SRC_AS 16
#define SRC_SUM 32
// qkv rowtiles: q 0..191, r 192..215, gate 216..407, sink 408..409 (24 rows + 8 zero), k 410..473, then
// local: v 474..537 (538) ; global: dk 474..475, v 476..539, dv 540..541 (542)
#define RT_K 410
#define OFF_R (192 * 16)
#define OFF_G (216 * 16)
#define OFF_S (408 * 16)
#define OFF_K (410 * 16)
#define OFF_DK (474 * 16)
#define QKVW (542 * 16)
#define ACC_STRIDE 136

struct LayerP {
  const uint4* wqkv; const uint4* wo; const uint4* wr; const float* rbias;
  const uint4* w13c; const unsigned char* w13s; const uint4* w2c; const unsigned char* w2s;
  const float* f13; const float* f2;
  long long mx, glob, loop;
  const bf16* nq; const bf16* nk; const bf16* nv; const bf16* nm;
  const bf16* ksc; const bf16* vsc; const bf16* asc; const bf16* msc;
  const bf16* qscale; const bf16* rscale; const bf16* relp;
  const float* aq; const float* mq;
  const bf16* cD; const bf16* cF; const bf16* cG;
  long long nrt;
  const bf16* relpT; long long pad1;
};

struct Params {
  LayerP L[NL];
  const bf16* embed; const bf16* embed_norm; const float* oq; const bf16* fnorm; const uint4* unembed; const float* lq;
  bf16* vecs; bf16* qkvo; bf16* attn_out; bf16* attn_s; bf16* act; float* rlog; float* zb; float* xg; float* msg;
  float* msgir; float* apart; u32* acnt; bf16* kc; bf16* vc; bf16* rk; bf16* rv; bf16* ro; bf16* rm; u32* bar;
  float* amv; int* ami;
  long long maxctx;
  unsigned long long* prof; long long prof_cap; const unsigned char* skel;
  float* zm; bf16* xn; bf16* xnd; float* xm; float* rrx; int* rdbg; bf16* yb;
  // track B (SKH, appended): sketch codes, per-row scale / error / norm, state words, eps of the softcap (device float)
  const uint4* skq; const float* sks; const float* ske; const float* skn; u32* skl; const float* skeps;
};

struct RunArgs {
  const long long* tok;   // [B][nprompt] input tokens of the first nprompt steps (prompt; mode 0: nprompt = 1)
  float* logits;          // mode 0: [B][VOCAB] of the last step
  long long* out_tok;     // mode 1: [B][ngen] greedy tokens
  int* host_tok;          // mode 2: host-mapped [B][ngen]
  volatile int* host_prog;   // mode 2: number of tokens written for every sequence
  int B, pos0, nsteps, mode;
  int stop_app;           // debug: stop after this many applications (NAPP = full)
  int nprompt;            // steps st >= nprompt feed the argmax of the previous step; ngen = nsteps - nprompt + 1
};

// ------------------------------------------------------------------------------------------------ primitives
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 ldcg128(const void* p) {
  uint4 r; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r;
}
__device__ __forceinline__ uint2 ldcg64(const void* p) {
  uint2 r; asm volatile("ld.global.cg.v2.u32 {%0,%1},[%2];" : "=r"(r.x), "=r"(r.y) : "l"(p)); return r;
}
__device__ __forceinline__ u32 ldcg32(const void* p) { u32 r; asm volatile("ld.global.cg.u32 %0,[%1];" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ float ldcgf(const float* p) { float r; asm volatile("ld.global.cg.f32 %0,[%1];" : "=f"(r) : "l"(p)); return r; }
__device__ __forceinline__ int ldcgi(const int* p) { int r; asm volatile("ld.global.cg.s32 %0,[%1];" : "=r"(r) : "l"(p)); return r; }
__device__ __forceinline__ u16 ldcg16(const void* p) { u16 r; asm volatile("ld.global.cg.u16 %0,[%1];" : "=h"(r) : "l"(p)); return r; }
#ifdef ACQ_POLL
__device__ __forceinline__ u32 ld_acquire(const u32* p) { u32 r; asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(r) : "l"(p)); return r; }
#else
// relaxed poll: an acquire load invalidates the whole L1 (CCTL.IVALL) at every iteration, which evicts the stack;
// all data written by other blocks is read with ld.global.cg (L2) or bulk copies, so no L1 invalidation is needed
__device__ __forceinline__ u32 ld_acquire(const u32* p) { u32 r; asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];" : "=r"(r) : "l"(p) : "memory"); return r; }
#endif
__device__ __forceinline__ void cvt4(u32 x, u32& r0, u32& r1, u32& r2, u32& r3) {
#ifdef FAKE_CVT
  r0 = x; r1 = x ^ 0x11111111u; r2 = x ^ 0x22222222u; r3 = x ^ 0x44444444u; return;
#endif
  asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4; cvt.rn.bf16x2.e2m1x2 %0, b0; cvt.rn.bf16x2.e2m1x2 %1, b1;"
      " cvt.rn.bf16x2.e2m1x2 %2, b2; cvt.rn.bf16x2.e2m1x2 %3, b3; }"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(x));
}
// two E4M3 (low byte -> low half) to f16x2, exact
__device__ __forceinline__ u32 e4m3x2_f16x2(u32 v16) {
  u32 r; asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(r) : "r"(v16)); return r;
}
// f16x2 holding E4M3 values -> bf16x2 bits of value * 2^-112 (normal, exact)
__device__ __forceinline__ u32 f16e4_to_bf(u32 h) { return ((h >> 3) & 0x0FFF0FFFu) | (h & 0x80008000u); }
__device__ __forceinline__ u32 hmul2u(u32 a, u32 b) {
  u32 r; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s)); return r; }
__device__ __forceinline__ u32 imad_u32(u32 a, u32 b, u32 c) { u32 r; asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
#ifndef SCIM
#define SCIM 8   // merge run: NVFP4 block-scale pairs by one E4M3x2->f16x2 conversion + HADD2.F32 (FMA pipe); bitwise equal. (4: IMAD scales, s0006)
#endif
#ifndef FP8D
#define FP8D 1   // merge run: MXFP8 weights to bf16 bits straight from the bytes (ALU halved); exhaustive check tests/fp8d
#endif
__device__ u32 g_mul20v = 1048576u;
__device__ u32 g_mul8v = 256u;   // SCIM 4: read once per consume_tasks call
#if SCIM == 2
__shared__ u32 s_mul20;   // 1 << 20, set at kernel start (a value ptxas cannot fold into a shift)
#define g_mul20 (s_mul20)
#endif
__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float u16f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ u16 f2u16(float v) { bf16 h = __float2bfloat16_rn(v); return *reinterpret_cast<u16*>(&h); }
__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float ldgbf(const bf16* p) { return __bfloat162float(__ldg(p)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// watchdog: a wait that lasts more than ~2 s traps (the kernel fails with an error instead of hanging the card)
__device__ __forceinline__ void watchdog(long long t0, int where) {
  if (clock64() - t0 > 4000000000ll) { printf("mk watchdog: block %d thread %d stuck at %d\n", blockIdx.x, threadIdx.x, where); __trap(); }
}
#define grid_bar(c, t) do { MARK(MKPROF_WAIT); grid_bar_((c), (t)); } while (0)
__device__ __forceinline__ void grid_bar_(u32* ctr, u32& target) {
  __syncthreads();
  target += NB;
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(ctr, 1u);
    if (ld_acquire(ctr) < target) {
      const long long t0 = clock64();
      while (ld_acquire(ctr) < target) {
#if BAR_SLEEP > 0
        __nanosleep(BAR_SLEEP);
#endif
        watchdog(t0, 2);
      }
    }
  }
  __syncthreads();
}

// block-wide sum, same order in every thread (all threads must call)
__device__ __forceinline__ float block_sum(float v, float* scr) {
  v = warp_sum(v);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) scr[threadIdx.x >> 5] = v;
  __syncthreads();
  float r = 0.f;
#pragma unroll
  for (int w = 0; w < NW; w++) r += scr[w];
  return r;
}

// ------------------------------------------------------------------------------------------------ TMA task runner
// (after the gpt-oss-120b engine of run s12) Every warp streams its weights through its own ring of RS stages in shared
// memory, filled by bulk copies issued by lane 0. A task = n consecutive items of one rowtile:
//   KIND 0  bf16: item = one kstep (16 columns) = 512 B tile in A-fragment order
//   KIND 1  nvfp4: item = 64 columns: 512 B codes [32 lanes][4 ksteps][4 regs] + 64 B E4M3 scales [8 g][4 ksteps][2]
//   KIND 2  mxfp8: item = 32 columns: 512 B codes [32 lanes][2 ksteps][4 regs][2] + 16 B UE8M0 scales [8 g][2]
#ifndef RKQ
#define RKQ 0   // 0: per kernel (3 at B=1, 2 otherwise)
#endif
#ifndef RS
#define RS 2   // RI 12 x RS 2: +27% decode rate vs 6 x 3 (per-stage overhead), B=8 -390 us
#endif
#ifndef TPW
#define TPW 16
#endif
#ifndef RI
#define RI 9   // 9 x 2 (one engine per process): B=8 -84, B=1 -18 vs 12 x 2 (18-, 36-, 72-item tasks without partial stages)
#endif
#define IBYTES 512
#define ITEM 576
#define RING_WARP (RS * RI * ITEM)
struct TD { const uint4* w; const unsigned char* s; const u16* x; int n; int slot; u32 bm; u64 rows8; int k0; };
struct TT { const uint4* w; const unsigned char* s; const u16* x; u64 rows8; int n; int slot; u32 bm; int k0; };
#define ROWS_ID 0x0706050403020100ull

__device__ __forceinline__ u32 smaddr(const void* p) { return (u32)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(u32 b, u32 n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n)); }
__device__ __forceinline__ void mbar_expect(u32 b, u32 bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(u32 b, u32 ph) {
  u32 ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
  if (ok) return;
  const long long t0 = clock64();
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(ok) : "r"(b), "r"(ph) : "memory");
    if (!ok) watchdog(t0, 1);
  } while (!ok);
}
#ifdef SKEL
__device__ const unsigned char* g_skel;
#endif
__device__ __forceinline__ void tma_load(u32 dst, const void* src, u32 bytes, u32 bar) {
#ifdef SKEL
  src = (const void*)(g_skel + ((uintptr_t)src & 0x7FFF0ull));
#endif
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(0x12F0000000000000ull) : "memory");
}
__device__ __forceinline__ void tma_load_plain(u32 dst, const void* src, u32 bytes, u32 bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void l2pf(const void* src, u32 bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src), "r"(bytes) : "memory");
}
struct Ring { u32 data; u32 bars; u32 pos; };
struct PState { int nt, pt, pi, kind; u32 issued; int pad[3]; };

template <int KIND> struct KindInfo;
template <> struct KindInfo<0> { static constexpr int SB = 0; };
template <> struct KindInfo<1> { static constexpr int SB = 64; };
template <> struct KindInfo<2> { static constexpr int SB = 16; };

template <int KIND>
__device__ __forceinline__ void issue_stage(const TT* tt, PState& ps, const Ring& rg, int lane) {
  constexpr int SB = KindInfo<KIND>::SB;
  const u32 sidx = ps.issued % RS;
  const u32 bar = rg.bars + sidx * 8, dst = rg.data + sidx * (RI * ITEM);
  const TT& q = tt[ps.pt];
  const int m = min(RI, q.n - ps.pi);
  if (lane == 0) {
    mbar_expect(bar, m * (IBYTES + SB));
    tma_load(dst, q.w + (size_t)ps.pi * 32, m * IBYTES, bar);
    if (SB) tma_load(dst + RI * IBYTES, q.s + (size_t)ps.pi * SB, m * SB, bar);
  }
  ps.pi += m;
  if (ps.pi == q.n) { ps.pt++; ps.pi = 0; }
  ps.issued++;
}

template <int KIND, class Desc>
__device__ __forceinline__ void prefetch_tasks(int ntask, TT* table, const Ring& rg, PState& ps, Desc desc) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int nt = (ntask > warp) ? (ntask - warp + NW - 1) / NW : 0;
  TT* tt = table + warp * TPW;
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

__device__ __forceinline__ u32 lds32(u32 a) { u32 r; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(a)); return r; }
__device__ __forceinline__ u32 lds16(u32 a) { u16 r; asm volatile("ld.shared.u16 %0, [%1];" : "=h"(r) : "r"(a)); return r; }
__device__ __forceinline__ uint2 lds64(u32 a) { uint2 r; asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "r"(a)); return r; }
__device__ __forceinline__ uint4 lds128(u32 a) {
  uint4 r; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(a)); return r;
}

#define BF_2P112 0x77807780u   // bf16x2 (2^112, 2^112)

// accumulators: NACC independent C fragments per warp (breaks the MMA dependency chain); summed in a fixed order at
// the end of a task: ((c0 + c1) + (c2 + c3))
#define NACC 4
template <int KIND>
__device__ __forceinline__ void item(float (*c)[4], u32 st, u32 xb, int kb, int i, int lane, int g, u32 mul20 = 1048576u, u32 mul8 = 256u, u32 mul4 = 16u) {
  const uint4 q = lds128(st + i * IBYTES + lane * 16);
  if (KIND == 0) {
    const int k = kb + i;
    const uint2 bb = lds64(xb + (k >> 2) * 128 + (k & 3) * 8);
    mma16816(c[0], q.x, q.y, q.z, q.w, bb.x, bb.y);
  } else if (KIND == 1) {
    const uint2 sc = lds64(st + RI * IBYTES + i * 64 + g * 8);
    const uint4 x0 = lds128(xb + (kb + i) * 128), x1 = lds128(xb + (kb + i) * 128 + 16);
    const u32 b[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const u32 cw[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const u32 sw4 = (j < 2) ? sc.x : sc.y, bsel = (j & 1) * 2;
      u32 a0, a1, a2, a3;
      cvt4(cw[j], a0, a1, a2, a3);
#ifdef SCALE_BF16
      // E4M3 scale (positive, normal: checked at load) -> bf16 bits (b << 4) + (120 << 7), broadcast to both halves
      const u32 sg = prmt(sw4, 0, 0x4040 + bsel * 0x0101) * 16u + 0x3C003C00u;
      const u32 sh = prmt(sw4, 0, 0x4141 + bsel * 0x0101) * 16u + 0x3C003C00u;
      a0 = hmul2u(a0, sg); a1 = hmul2u(a1, sh); a2 = hmul2u(a2, sg); a3 = hmul2u(a3, sh);
      mma16816(c[j], a0, a1, a2, a3, b[2 * j], b[2 * j + 1]);
#else
      // codes x x on the tensor core, then the E4M3 row scale in fp32: bits (b << 20) + (120 << 23) (positive normal)
#if SCIM == 1
      // merge run (SCIM): the scale's (b << 20) + (120 << 23) as an integer multiply-add (FMA pipe), not LEA (ALU pipe)
      const float sg = __uint_as_float(imad_u32(prmt(sw4, 0, 0x4440 + bsel), 1048576u, 0x3C000000u));
      const float sh = __uint_as_float(imad_u32(prmt(sw4, 0, 0x4441 + bsel), 1048576u, 0x3C000000u));
#elif SCIM == 2
      const float sg = __uint_as_float(imad_u32(prmt(sw4, 0, 0x4440 + bsel), g_mul20, 0x3C000000u));
      const float sh = __uint_as_float(imad_u32(prmt(sw4, 0, 0x4441 + bsel), g_mul20, 0x3C000000u));
#elif SCIM == 4
      const float sg = __uint_as_float(imad_u32(prmt(sw4, 0, 0x4440 + bsel), mul20, 0x3C000000u));
      const float sh = __uint_as_float(imad_u32(prmt(sw4, 0, 0x4441 + bsel), mul20, 0x3C000000u));
#elif SCIM == 7
      // both scales of this j (rows g, g+8 = bytes bsel, bsel + 1) by one exact E4M3x2 -> f16x2 conversion, then
      // f16 -> f32 (HADD2.F32 on the FMA pipe): the same values as (b << 20) + (120 << 23)
      const u32 hf = e4m3x2_f16x2(bsel ? (sw4 >> 16) : (sw4 & 0xffffu));
      const float sg = __half2float(__ushort_as_half((unsigned short)(hf & 0xffffu)));
      const float sh = __half2float(__ushort_as_half((unsigned short)(hf >> 16)));
#elif SCIM == 8
      u32 hf;
      if (bsel) asm("{ .reg .b16 l,h; mov.b32 {l,h}, %1; cvt.rn.f16x2.e4m3x2 %0, h; }" : "=r"(hf) : "r"(sw4));
      else asm("{ .reg .b16 l,h; mov.b32 {l,h}, %1; cvt.rn.f16x2.e4m3x2 %0, l; }" : "=r"(hf) : "r"(sw4));
      const float sg = __half2float(__ushort_as_half((unsigned short)(hf & 0xffffu)));
      const float sh = __half2float(__ushort_as_half((unsigned short)(hf >> 16)));
#elif SCIM == 5 || SCIM == 6
      // byte k of sw4 on the FMA pipe: shift it to the top byte by a multiply (k < 3), take it by a multiply-high by
      // 256 (opaque multipliers: ptxas would turn powers of two into ALU shifts)
      auto byte_f = [&](int k) -> u32 {
        const u32 y = k == 3 ? sw4 : k == 2 ? sw4 * mul8 : sw4 * (mul8 * mul8);
        return __umulhi(y, mul8);
      };
      const int kg = bsel, kh = bsel + 1;
      const u32 bg = (SCIM == 6 && kg != 0) ? byte_f(kg) : prmt(sw4, 0, 0x4440 + bsel);
      const u32 bh = (kh == 3 || SCIM == 6) ? byte_f(kh) : prmt(sw4, 0, 0x4441 + bsel);
      const float sg = __uint_as_float(imad_u32(bg, mul20, 0x3C000000u));
      const float sh = __uint_as_float(imad_u32(bh, mul20, 0x3C000000u));
#elif SCIM == 3
      // byte placed at bits 16..23 by the permute, then x 16 + (120 << 23) by a multiply-add
      const float sg = __uint_as_float(imad_u32(prmt(sw4, 0, 0x4044 + (bsel << 8)), 16u, 0x3C000000u));
      const float sh = __uint_as_float(imad_u32(prmt(sw4, 0, 0x4144 + (bsel << 8)), 16u, 0x3C000000u));
#else
      const float sg = __uint_as_float(prmt(sw4, 0, 0x4440 + bsel) * 1048576u + 0x3C000000u);
      const float sh = __uint_as_float(prmt(sw4, 0, 0x4441 + bsel) * 1048576u + 0x3C000000u);
#endif
      float tq[4];
      asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};"
          : "=f"(tq[0]), "=f"(tq[1]), "=f"(tq[2]), "=f"(tq[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b[2 * j]), "r"(b[2 * j + 1]), "f"(0.f));
      c[j][0] = fmaf(tq[0], sg, c[j][0]); c[j][1] = fmaf(tq[1], sg, c[j][1]);
      c[j][2] = fmaf(tq[2], sh, c[j][2]); c[j][3] = fmaf(tq[3], sh, c[j][3]);
#endif
    }
  } else {
    const u32 sw = lds16(st + RI * IBYTES + i * 16 + g * 2);
#if FP8D
    // merge run (FP8D): weights decoded straight from the E4M3 bytes to bf16 bits (byte << 4, the sign moved from
    // bit 11 to 15 by adding (byte & 0x80) x 240): the bf16 value is the E4M3 value x 2^-120 for normal and
    // subnormal codes alike (bf16 subnormals); the scale is 2^(e-7). 2 ALU + 2 FMA-pipe instructions per pair
    const u32 S = (prmt(sw, 0, 0x4140) << 7) + 0x3C003C00u;    // bf16 2^(e-7): lo row g, hi row g+8
    auto dec2 = [&](u32 xx) { return imad_u32(xx & 0x00800080u, 240u, imad_u32(xx, mul4, 0u)); };
#else
    const u32 S = (prmt(sw, 0, 0x4140) << 7) + 0x38003800u;    // bf16 2^(e-15): lo row g, hi row g+8
#endif
    const u32 sg = prmt(S, 0, 0x1010), sh = prmt(S, 0, 0x3232);
    const int k = kb + i;
    const uint4 x = lds128(xb + (k >> 1) * 128 + (k & 1) * 16);
    const u32 b[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const u32 wa = j ? q.z : q.x, wb = j ? q.w : q.y;
#if FP8D
      const u32 a0 = hmul2u(dec2(prmt(wa, 0, 0x4140)), sg);
      const u32 a1 = hmul2u(dec2(prmt(wa, 0, 0x4342)), sh);
      const u32 a2 = hmul2u(dec2(prmt(wb, 0, 0x4140)), sg);
      const u32 a3 = hmul2u(dec2(prmt(wb, 0, 0x4342)), sh);
      mma16816(c[j], a0, a1, a2, a3, b[2 * j], b[2 * j + 1]);
#else
      const u32 a0 = hmul2u(f16e4_to_bf(e4m3x2_f16x2(wa & 0xffffu)), sg);
      const u32 a1 = hmul2u(f16e4_to_bf(e4m3x2_f16x2(wa >> 16)), sh);
      const u32 a2 = hmul2u(f16e4_to_bf(e4m3x2_f16x2(wb & 0xffffu)), sg);
      const u32 a3 = hmul2u(f16e4_to_bf(e4m3x2_f16x2(wb >> 16)), sh);
      mma16816(c[j], a0, a1, a2, a3, b[2 * j], b[2 * j + 1]);
#endif
    }
  }
}

// merge run (CTNI): the decode loop out of line, so that its code does not follow the register pressure of the rest
// of the kernel (the same loop compiled 540 vs 583 instructions per 40 HMMA with an unrelated change elsewhere)
#ifndef CTNI
#define CTNI 0
#endif
#if CTNI
#define CT_INL __noinline__
#else
#define CT_INL __forceinline__
#endif
template <int KIND, class Fin>
__device__ CT_INL void consume_tasks(TT* table, int xst, Ring& rg, PState& psm, Fin fin) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  PState ps = psm;
  const int nt = ps.nt;
  if (nt == 0) return;
  const TT* tt = table + warp * TPW;
  int ct = 0, ci = 0;
  TT cq = tt[0];
  u32 cx;
  // B column g reads smem row (rows8 >> 8g) & 255; columns that are not live point at a valid row (results unused)
  auto setx = [&]() {
    const int row = (int)((cq.rows8 >> (8 * g)) & 0xff);
    cx = smaddr(cq.x + row * xst) + t * 32;
  };
  setx();
  float c[NACC][4];
#pragma unroll
  for (int a = 0; a < NACC; a++) c[a][0] = c[a][1] = c[a][2] = c[a][3] = 0.f;
  u32 mul20 = 1048576u, mul8 = 256u;
#if SCIM >= 4
  asm volatile("mov.u32 %0, %1;" : "=r"(mul20) : "r"(g_mul20v));   // opaque: ptxas must keep a multiply
#endif
#if SCIM >= 5
  asm volatile("mov.u32 %0, %1;" : "=r"(mul8) : "r"(g_mul8v));
#endif
  const u32 mul4 = mul20 >> 16;   // 16, opaque as mul20 (FP8D)
  while (ct < nt) {
    const u32 sidx = rg.pos % RS;
    const int n = min(RI, cq.n - ci);
    mbar_wait(rg.bars + sidx * 8, (rg.pos / RS) & 1);
    const u32 st = rg.data + sidx * (RI * ITEM);
    const int kb = cq.k0 + ci;
    if (n == RI) {
#pragma unroll
      for (int i = 0; i < RI; i++) item<KIND>(c, st, cx, kb, i, lane, g, mul20, mul8, mul4);
    } else {
#pragma unroll 1
      for (int i = 0; i < n; i++) item<KIND>(c, st, cx, kb, i, lane, g, mul20, mul8, mul4);
    }
    __syncwarp();
    rg.pos++;
    if (ps.pt < nt) {
#ifdef RFENCE   // not needed: the slot's shared loads have returned (registers in use) before __syncwarp
      if (lane == 0) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
      issue_stage<KIND>(tt, ps, rg, lane);
    }
    ci += n;
    if (ci == cq.n) {
      float r[4];
#pragma unroll
      for (int q = 0; q < 4; q++) { r[q] = (c[0][q] + c[1][q]) + (c[2][q] + c[3][q]); }
      fin(cq.slot, r);
#pragma unroll
      for (int a = 0; a < NACC; a++) c[a][0] = c[a][1] = c[a][2] = c[a][3] = 0.f;
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

__device__ __forceinline__ void store_task(float* red, int slot, const float* c) {
  const int lane = threadIdx.x & 31;
  *reinterpret_cast<float4*>(red + slot * 128 + lane * 4) = make_float4(c[0], c[1], c[2], c[3]);
}
__device__ __forceinline__ float red_at(const float* red, int slot, int i, int b) {
  return red[slot * 128 + ((i & 7) * 4 + (b >> 1)) * 4 + (i >> 3) * 2 + (b & 1)];
}

// B operand rows in shared memory, fragment order inside every 64-column chunk
__device__ __forceinline__ int xfo(int k) {
  const int c = k >> 6, r = k & 63, j = r >> 4, rr = r & 15;
  return c * 64 + (((rr & 7) >> 1) * 8 + j * 2 + (rr >> 3)) * 2 + (rr & 1);
}

// ------------------------------------------------------------------------------------------------ shared memory
#ifndef XS_BYTES
#define XS_BYTES 74240
#endif
#define SM_XS 0
#define SM_RED XS_BYTES
#ifndef RED_SLOTS
#define RED_SLOTS 32   // 48 -> 32: B=8 -6 (lone-process), B=1 equal
#endif
#define SM_MISC (SM_RED + RED_SLOTS * 512)
#define MISC_BYTES 12288
#define SM_TT (SM_MISC + MISC_BYTES)
#define SM_BARS (SM_TT + NW * TPW * (int)sizeof(TT))
#define SM_PS (SM_BARS + 512)   // ring barriers [NW][RS], then the deep-w13 barriers [NW][3] at + 128
#define SM_RING (SM_PS + 256)
#define SM_XF SM_XS
#define SM_TOTAL (SM_RING + NW * RING_WARP)

struct Misc {
  int tok[MAXB];
  float scr[32];
  float irms[MAXB][8];
  float dots[MAXB][3][8];
  float coef[MAXB][4][8];
  float rr[MAXB][4];
  int U;
  int fr_s0, fr_s1, fr_np;
  int last;
  int pad3[3];
  int sexp[MAXU];
  int pcnt[MAXU + 1];   // exclusive prefix of the routed (slot, token) pairs
  long long ats[12];
  u32 smask[MAXU];
  float rw[MAXB][MAXU];
  float f13[MAXU], f2[MAXU];
  u64 sbar; u32 sphase; u32 pad4;
  u64 sbar2; u32 sphase2; u32 pad5;   // attention V tiles
  int sel[MAXB][TOPK];
  float selw[MAXB][TOPK];
  u64 srows[MAXU];
  float sl[MAXB][4][16];
  float tmpf[64];
  int tmpi[64];
  int zp[4][2];
  int nzp;
  float wred[NW][20];
  float tot[20];
  float coefq[3][8];
  float rs[4];
  u32 stgt[MAXU];   // per-slot targets of the w13 completion counters (B = 8 path)
  u32 dpc[NW][3];   // uses of the deep-w13 barriers (parity)
  // mixnorm argument arrays (block-uniform; in shared memory: as stack arrays they live in local memory = L2 and cost
  // a dependent round trip before every load of the sources)
  int ablk[6], aasrc[6], amsrc[6], amsg[2], afs[5], aq012[3], aqsel3[3], azero[1];
  const float* aaqs[3]; const float* amq[1]; const float* aqp[1]; const float* aoq[1];
  const bf16* anws[3]; const bf16* anw3[3]; const bf16* anw1[1]; const bf16* agw[3];
  u16* arows3[3]; u16* agr[3]; u16* arow1[1];
  const float* asrc3[3]; const float* asrc1[1];
  int sel8[MAXB][TOPK];   // the routed slots of each sequence in slot order (B = 8 w2 combine)
};
static_assert(sizeof(Misc) <= MISC_BYTES, "misc");
static_assert(SM_TOTAL <= 232448, "smem");

// attention scratch of the tensor-core attention (in the red region, free during phase B)
#ifndef CH2
#define CH2 256
#endif
#define NKT (CH2 / 16)   // key tiles per chunk
#define KPW (NKT / 8)    // key tiles per warp
static_assert(CH2 % 128 == 0 && CH2 <= 256, "CH2");
struct AttS2 {
  u16 qb[3][HD];
  u16 rb[8][DREL];
  float sc[3][CH2];
  u16 ph[3][CH2], pl[3][CH2];
  float acc[3][HD];
  float ml[3][2];
  float red[16];
  float sink[3];
  float alpha[4];
  int kvoff[2][HD];
  float mlb[MAXCH * 3][2];
  float mw[MAXCH][3];
};
static_assert(36 % RI == 0 || RI == 12 || true, "deep w13 stages");
static_assert(sizeof(AttS2) <= RED_SLOTS * 512, "AttS2");
// attention scratch (in the xs region, free during phase B)
struct AttS {
  float mlb[MAXCH * 3][2];
  float mw[MAXCH][3];
  float q[3][HD];
  float r[3][DREL];
  float kn[HD];
  float vn[HD];
  float sc[3][CH];
  float ml[3][2];
  float acc[4][3][HD];
  float red[16];
  float sink[3];
};

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ int app_layer(int a) { return a < 6 ? a : (a < 42 ? 6 + (a - 6) % 12 : a - 24); }
__device__ __forceinline__ int app_pass(int a) { return (a >= 6 && a < 42) ? (a - 6) / 12 : 0; }
__device__ __forceinline__ int cache_slot(int i, int pas) { return (pas == 0) ? i : NL + (pas - 1) * 12 + (i - 6); }

// eight consecutive elements of a mix source (fp32): code < 16: vector; SRC_AS: attn_s; SRC_SUM + v: vector v + attn_s
__device__ __forceinline__ void unpack8(uint4 v, float* o) {
  o[0] = __uint_as_float(v.x << 16); o[1] = __uint_as_float(v.x & 0xffff0000u);
  o[2] = __uint_as_float(v.y << 16); o[3] = __uint_as_float(v.y & 0xffff0000u);
  o[4] = __uint_as_float(v.z << 16); o[5] = __uint_as_float(v.z & 0xffff0000u);
  o[6] = __uint_as_float(v.w << 16); o[7] = __uint_as_float(v.w & 0xffff0000u);
}
__device__ __forceinline__ void src_load8(const Params& p, int code, int b, int k, float* v) {
  if (code < SRC_AS) {
    unpack8(ldcg128(p.vecs + ((size_t)b * NVEC + code) * H + k), v);
  } else if (code == SRC_AS) {
    unpack8(ldcg128(p.attn_s + (size_t)b * H + k), v);
  } else {
    float w[8];
    unpack8(ldcg128(p.vecs + ((size_t)b * NVEC + (code - SRC_SUM)) * H + k), v);
    unpack8(ldcg128(p.attn_s + (size_t)b * H + k), w);
#pragma unroll
    for (int j = 0; j < 8; j++) v[j] += w[j];
  }
}

#define MAXSRC 5
#ifdef ATSA
#define XTS(k) do { if (threadIdx.x == 0 && ap == ATSA) { ms->ats[k] = clock64(); if ((k) == 0 || (k) == 2 || (k) == 3) { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); ms->ats[(k) == 0 ? 8 : (k) == 2 ? 9 : 11] = (long long)t_; } } } while (0)
#else
#define XTS(k) do {} while (0)
#endif
#ifndef ATSGG
#define ATSGG 0
#endif
#ifdef ATSG
#define ZTS(k) do { if (threadIdx.x == 0 && ap == ATSG && g0 == ATSGG) ms->ats[k] = clock64(); } while (0)
#else
#define ZTS(k) do {} while (0)
#endif
#ifdef ATSE
#define YTS(k) do { if (threadIdx.x == 0 && ap == ATSE) ms->ats[k] = clock64(); } while (0)
#else
#define YTS(k) do {} while (0)
#endif
#ifdef ATTTS
#define ATS(k) do { if (threadIdx.x == 0) ms->ats[k] = clock64(); } while (0)
#else
#define ATS(k) do {} while (0)
#endif
// ring prefetch placement: 1 = the next GEMV's first stages are issued before the barrier that precedes it
#ifndef MERGE_IN_C
#define MERGE_IN_C 0
#endif
#ifndef BAR_SLEEP
#define BAR_SLEEP 0
#endif
#ifndef NO_L2BIG
#define NO_L2BIG 1
#endif
#ifndef PFE_QKV
#define PFE_QKV 1
#endif
#ifndef SCNT
#define SCNT 1
#endif
#ifndef FUNIT
#define FUNIT 1
#endif
#ifndef PFO_ALL
#define PFO_ALL 1
#endif
#ifndef PUB_REL
#define PUB_REL 0
#endif
#ifndef PMAXC
#define PMAXC 0
#endif
#ifndef ATT_ACQREL
#define ATT_ACQREL 1
#endif
#ifndef SCNT_REL
#define SCNT_REL 0
#endif
#ifndef W13D
#define W13D 1
#endif
#ifndef BSPEC
#define BSPEC 1
#endif
#ifndef MIXT
#define MIXT 1
#endif
#ifndef CH2B1
#define CH2B1 128
#endif
#ifndef ONER1
#define ONER1 1
#endif
#ifndef NOFU1
#define NOFU1 1
#endif
#ifndef DEADB1
#define DEADB1 1
#endif
#ifndef NOFR8
#define NOFR8 1
#endif
#ifndef RDBG
#define RDBG 0   // record the chosen experts per application (tests/t_atse.py union statistics)
#endif
#ifndef MIXHI
#define MIXHI 1   // mixer blocks NB - B .. NB - 1 (no o_proj / combine rows): B=8 -19 us lone-process
#endif
#define MIXB(x) (MIXHI ? (x) - (NB - B) : (x))   // mixer index of block x
#ifndef APF
#define APF 0
#endif
#ifndef GROT
#define GROT 1   // gate phase 1: F/G chunk order rotated by row (bank spread): B=8 -29, B=1 -7 lone-process
#endif
#ifndef PCEIL
#define PCEIL 0
#endif
#ifndef W2L2
#define W2L2 0   // measured: -430 us B=8, -100 B=1 without it (the prefetch delayed the w13 start and was evicted)
#endif
#ifndef PFE_O
#define PFE_O 0
#endif
#ifndef PFE_R
#define PFE_R 1   // router stages issued at the end of C (lone-process A/B: B=1 -2, B=8 -2..-6)
#endif
#ifndef PFE_DN
#define PFE_DN 0
#endif
#ifndef PADN
#define PADN 0
#endif
#ifndef GZ0P0
#define GZ0P0 GZ0
#endif
#ifndef GZ0P1
#define GZ0P1 GZ0
#endif
#ifndef GZK
#define GZK 1   // keeps the mixer path compiled (never taken): without it the decode loops compile 8-21% longer
#endif
#ifndef GZV
#define GZV 0   // GZ0 code variants (code-generation search)
#endif
#ifndef GZ0
#define GZ0 1   // merge run: B > 1 gate phase 0 without mixer blocks (each block mixes one sequence, own z rows); bitwise equal
#endif
#ifndef MSGW
#define MSGW 1   // merge run: warp-per-message M phase at B > 1 (pass-2 messages rel 1e-7 vs the old path)
#endif
#ifndef MSGV
#define MSGV 4   // bit 2: no first-row D preload (with it the decode loops compile longer)
#endif
#ifndef MSGK
#define MSGK 1   // as GZK for the old M path
#endif
#ifndef MSGW_INL
#define MSGW_INL __forceinline__
#endif
// ablation switches (timing studies only; results are wrong with any of them set)
#ifndef SKE1
#define SKE1 0
#endif
#ifndef SKE2
#define SKE2 0
#endif
#ifndef SKA1
#define SKA1 0
#endif
#ifndef SKA2
#define SKA2 0
#endif
#ifndef SKG1
#define SKG1 0
#endif
#ifndef SKG2
#define SKG2 0
#endif
#ifndef SKG3
#define SKG3 0
#endif
#ifndef SKA
#define SKA 0
#endif
#ifndef SKB
#define SKB 0
#endif
#ifndef SKC
#define SKC 0
#endif
#ifndef SKD
#define SKD 0
#endif
#ifndef SKE
#define SKE 0
#endif
#ifndef SKF
#define SKF 0
#endif
#ifndef SKG
#define SKG 0
#endif
#ifndef SKIPH   // debug: skip the head (was SKH in s15-mini-b200-a1; the name SKH is the sketch head now)
#define SKIPH 0
#endif
#ifndef SKH
#define SKH 0
#endif
#ifdef MKPROF
// profiler state in shared memory (a mark must not cost a memory round trip)
__shared__ unsigned long long* s_prof_buf;
__shared__ unsigned s_prof_cap, s_pn;
#define FMARK(tag) do { if (threadIdx.x == 0) { MkProf pr_{s_prof_buf, s_prof_cap}; unsigned n_ = s_pn; MKPROF_MARK(pr_, n_, tag); s_pn = n_; } } while (0)
#else
#ifdef MXTS
#define FMARK(tag) do { if (threadIdx.x == 0 && (tag) != 31) ms->ats[(tag) == 29 ? 7 : (tag) == 30 ? 8 : (tag) - 23] = clock64(); } while (0)
#else
#define FMARK(tag) do {} while (0)
#endif
#endif
__device__ __forceinline__ const bf16* src_ptr(const Params& p, int code, int b) {
  if (code == SRC_AS) return p.attn_s + (size_t)b * H;
  return p.vecs + ((size_t)b * NVEC + (code >= SRC_SUM ? code - SRC_SUM : code)) * H;
}
__device__ __forceinline__ void store_row8(u16* row, int k, const float* v, const uint4 w) {
  float wf[8]; unpack8(w, wf);
  u32 o[4];
#pragma unroll
  for (int j = 0; j < 4; j++) o[j] = (u32)f2u16(wf[2 * j] * v[2 * j]) | ((u32)f2u16(wf[2 * j + 1] * v[2 * j + 1]) << 16);
  *reinterpret_cast<u32*>(row + xfo(k)) = o[0]; *reinterpret_cast<u32*>(row + xfo(k + 2)) = o[1];
  *reinterpret_cast<u32*>(row + xfo(k + 4)) = o[2]; *reinterpret_cast<u32*>(row + xfo(k + 6)) = o[3];
}
// thread chunks of 8 elements: tid and tid + NT (the second only for tid < 32)
#define NCHK(tid) (((tid) < H / 8 - NT) ? 2 : 1)

// Block attention residual + RMSNorm of one sequence, all loads in one round: sources codes[0..nsrc) (only the last
// may be SRC_SUM + v), nq queries; ns streams, stream s mixes with query qsel[s]: its fp32 mix -> XF + s * H (if XF),
// its 1/rms -> ms->rs[s]; with nw[s] the normalised bf16 row w * (x / rms) -> xrow[s] (fragment order).
template <int MQ, int MS>
__device__ __noinline__ void mixnorm_t(const Params& p, int b, const int* codes, int nsrc, const float* const* qs, int nq,
                        int ns, const int* qsel, const bf16* const* nw, u16* const* xrow, float* XF, Misc* ms) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  FMARK(29);
  const int nck = NCHK(tid);
  uint4 sv[MAXSRC][2], sx[2];
  const bool sum_last = codes[nsrc - 1] >= SRC_SUM;
#pragma unroll
  for (int n = 0; n < MAXSRC; n++)
    if (n < nsrc) {
      const bf16* sp = src_ptr(p, codes[n], b);
#pragma unroll
      for (int k = 0; k < 2; k++) if (k < nck) sv[n][k] = ldcg128(sp + (tid + k * NT) * 8);
    }
#pragma unroll
  for (int k = 0; k < 2; k++) if (sum_last && k < nck) sx[k] = ldcg128(p.attn_s + (size_t)b * H + (tid + k * NT) * 8);
  float part[MAXSRC][1 + MQ];
#pragma unroll
  for (int n = 0; n < MAXSRC; n++)
#pragma unroll
    for (int j = 0; j <= MQ; j++) part[n][j] = 0.f;
#pragma unroll
  for (int k = 0; k < 2; k++) {
    if (k >= nck) break;
    const int e0 = (tid + k * NT) * 8;
    float qv[MQ][8];
#pragma unroll
    for (int q = 0; q < MQ; q++)
      if (q < nq) {
        const float4 a0 = __ldg(reinterpret_cast<const float4*>(qs[q] + e0)), a1 = __ldg(reinterpret_cast<const float4*>(qs[q] + e0 + 4));
        qv[q][0] = a0.x; qv[q][1] = a0.y; qv[q][2] = a0.z; qv[q][3] = a0.w; qv[q][4] = a1.x; qv[q][5] = a1.y; qv[q][6] = a1.z; qv[q][7] = a1.w;
      }
#pragma unroll
    for (int n = 0; n < MAXSRC; n++)
      if (n < nsrc) {
        float v[8]; unpack8(sv[n][k], v);
        if (sum_last && n == nsrc - 1) { float w[8]; unpack8(sx[k], w);
#pragma unroll
          for (int j = 0; j < 8; j++) v[j] += w[j]; }
#pragma unroll
        for (int j = 0; j < 8; j++) part[n][0] = fmaf(v[j], v[j], part[n][0]);
#pragma unroll
        for (int q = 0; q < MQ; q++)
          if (q < nq) {
#pragma unroll
            for (int j = 0; j < 8; j++) part[n][1 + q] = fmaf(qv[q][j], v[j], part[n][1 + q]);
          }
      }
  }
#pragma unroll
  for (int n = 0; n < MAXSRC; n++)
    if (n < nsrc) {
#pragma unroll
      for (int j = 0; j <= MQ; j++)
        if (j <= nq) { const float v = warp_sum(part[n][j]); if (lane == 0) ms->wred[warp][n * 4 + j] = v; }
    }
  FMARK(30);
  __syncthreads();
  FMARK(31);
  if (tid < nsrc * 4) { float t = 0.f;
#pragma unroll
    for (int w = 0; w < NW; w++) t += ms->wred[w][tid];
    ms->tot[tid] = t; }
  __syncthreads();
  if (tid < nq) {
    float lg[MAXSRC], ir[MAXSRC], m = -INFINITY;   // unrolled with predicates: registers, not local memory
#pragma unroll
    for (int n = 0; n < MAXSRC; n++) if (n < nsrc) { ir[n] = rsqrtf(ms->tot[n * 4] * (1.0f / H) + 1e-6f); lg[n] = ms->tot[n * 4 + 1 + tid] * ir[n] * (1.0f / 48.0f); m = fmaxf(m, lg[n]); }
    float sum = 0.f;
#pragma unroll
    for (int n = 0; n < MAXSRC; n++) if (n < nsrc) { lg[n] = expf(lg[n] - m); sum += lg[n]; }
    const float inv = 1.0f / sum;
#pragma unroll
    for (int n = 0; n < MAXSRC; n++) if (n < nsrc) ms->coefq[tid][n] = lg[n] * inv * ir[n];
  }
  __syncthreads();
  FMARK(32);
  // the streams' values (recomputed from the loaded sources): pass 1 sums of squares, pass 2 outputs
  auto xval = [&](int q, int k, float* x) {
#pragma unroll
    for (int j = 0; j < 8; j++) x[j] = 0.f;
#pragma unroll
    for (int n = 0; n < MAXSRC; n++)
      if (n < nsrc) {
        float v[8]; unpack8(sv[n][k], v);
        if (sum_last && n == nsrc - 1) { float w[8]; unpack8(sx[k], w);
#pragma unroll
          for (int j = 0; j < 8; j++) v[j] += w[j]; }
        const float cf = ms->coefq[q][n];
#pragma unroll
        for (int j = 0; j < 8; j++) x[j] = fmaf(cf, v[j], x[j]);
      }
  };
  float ssx[MS];
#pragma unroll
  for (int s = 0; s < MS; s++) ssx[s] = 0.f;
#pragma unroll
  for (int s = 0; s < MS; s++)
    if (s < ns) {
#pragma unroll
      for (int k = 0; k < 2; k++) if (k < nck) { float x[8]; xval(qsel[s], k, x);
#pragma unroll
        for (int j = 0; j < 8; j++) ssx[s] = fmaf(x[j], x[j], ssx[s]); }
      const float v = warp_sum(ssx[s]);
      if (lane == 0) ms->wred[warp][s] = v;
    }
  __syncthreads();
  if (tid < ns) { float t = 0.f;
#pragma unroll
    for (int w = 0; w < NW; w++) t += ms->wred[w][tid];
    ms->rs[tid] = rsqrtf(t * (1.0f / H) + 1e-6f); }
  __syncthreads();
  FMARK(33);
#pragma unroll
  for (int s = 0; s < MS; s++)
    if (s < ns) {
      const float r = ms->rs[s];
#pragma unroll
      for (int k = 0; k < 2; k++) if (k < nck) {
        const int e0 = (tid + k * NT) * 8;
        float x[8]; xval(qsel[s], k, x);
        if (XF) {
          *reinterpret_cast<float4*>(XF + s * H + e0) = make_float4(x[0], x[1], x[2], x[3]);
          *reinterpret_cast<float4*>(XF + s * H + e0 + 4) = make_float4(x[4], x[5], x[6], x[7]);
        }
        if (nw && nw[s]) {
#pragma unroll
          for (int j = 0; j < 8; j++) x[j] *= r;
          store_row8(xrow[s], e0, x, __ldg(reinterpret_cast<const uint4*>(nw[s] + e0)));
        }
      }
    }
  __syncthreads();
  FMARK(34);
}

// B > 1 (merge run, MSGW): phase M. (1) one warp per (sequence, layer-gate) message: depth attention over the two
// previous-pass sources, its 1/rms, the unit row bf16(m / rms) into yb (free here); (2) after a grid barrier the zm rows
// with every D row loaded once for all sequences (unit rows of the block's <= 2 layer-gates staged in smem).
// Out of line and with few live registers: inlined, it made the B = 8 kernel spill (16 -> 744 B stack, +400 us).
__device__ MSGW_INL void msg_w(const Params& p, int B, int pas, float* XF, u32* bar, u32& bar_target) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, bid = blockIdx.x;
  const int ms0 = V_P1 + pas - 1, ms1 = V_P2 + pas - 1;
  for (int pr = bid * NW + warp; pr < ((MSGV & 1) ? 0 : B * 48); pr += NB * NW) {
    const int b = pr % B, lg = pr / B;
    const bf16* s0 = src_ptr(p, ms0, b);
    const bf16* s1 = src_ptr(p, ms1, b);
    const float* qq = p.lq + (size_t)lg * H;
    float ss0 = 0.f, ss1 = 0.f, d0 = 0.f, d1 = 0.f;
#pragma unroll 1
    for (int c3 = 0; c3 < 9; c3 += 3) {
      uint4 v0[3], v1[3];
#pragma unroll
      for (int c = 0; c < 3; c++) { v0[c] = ldcg128(s0 + (lane + 32 * (c3 + c)) * 8); v1[c] = ldcg128(s1 + (lane + 32 * (c3 + c)) * 8); }
#pragma unroll
      for (int c = 0; c < 3; c++) {
        const int e0 = (lane + 32 * (c3 + c)) * 8;
        const float4 qa = __ldg(reinterpret_cast<const float4*>(qq + e0)), qb = __ldg(reinterpret_cast<const float4*>(qq + e0 + 4));
        const float qv[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
        float a[8], bb[8]; unpack8(v0[c], a); unpack8(v1[c], bb);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          ss0 = fmaf(a[j], a[j], ss0); ss1 = fmaf(bb[j], bb[j], ss1);
          d0 = fmaf(qv[j], a[j], d0); d1 = fmaf(qv[j], bb[j], d1);
        }
      }
    }
    ss0 = warp_sum(ss0); ss1 = warp_sum(ss1); d0 = warp_sum(d0); d1 = warp_sum(d1);
    const float ir0 = rsqrtf(ss0 * (1.0f / H) + 1e-6f), ir1 = rsqrtf(ss1 * (1.0f / H) + 1e-6f);
    const float l0 = d0 * ir0 * (1.0f / 48.0f), l1 = d1 * ir1 * (1.0f / 48.0f), mx = fmaxf(l0, l1);
    const float e0_ = expf(l0 - mx), e1_ = expf(l1 - mx), inv = 1.0f / (e0_ + e1_);
    const float c0 = e0_ * inv * ir0, c1 = e1_ * inv * ir1;
    float ssx = 0.f;
#pragma unroll 1
    for (int c3 = 0; c3 < 9; c3 += 3) {
      uint4 v0[3], v1[3];
#pragma unroll
      for (int c = 0; c < 3; c++) { v0[c] = ldcg128(s0 + (lane + 32 * (c3 + c)) * 8); v1[c] = ldcg128(s1 + (lane + 32 * (c3 + c)) * 8); }
#pragma unroll
      for (int c = 0; c < 3; c++) {
        float a[8], bb[8]; unpack8(v0[c], a); unpack8(v1[c], bb);
#pragma unroll
        for (int j = 0; j < 8; j++) { const float x = fmaf(c1, bb[j], fmaf(c0, a[j], 0.f)); ssx = fmaf(x, x, ssx); }
      }
    }
    const float r = rsqrtf(warp_sum(ssx) * (1.0f / H) + 1e-6f);
    float* mrow = p.msg + ((size_t)lg * MAXB + b) * H;
    u16* urow = reinterpret_cast<u16*>(p.yb) + ((size_t)lg * MAXB + b) * H;
#pragma unroll 1
    for (int c3 = 0; c3 < 9; c3 += 3) {
      uint4 v0[3], v1[3];
#pragma unroll
      for (int c = 0; c < 3; c++) { v0[c] = ldcg128(s0 + (lane + 32 * (c3 + c)) * 8); v1[c] = ldcg128(s1 + (lane + 32 * (c3 + c)) * 8); }
#pragma unroll
      for (int c = 0; c < 3; c++) {
        const int e0 = (lane + 32 * (c3 + c)) * 8;
        float a[8], bb[8], x[8]; unpack8(v0[c], a); unpack8(v1[c], bb);
#pragma unroll
        for (int j = 0; j < 8; j++) x[j] = fmaf(c1, bb[j], fmaf(c0, a[j], 0.f));
        *reinterpret_cast<float4*>(mrow + e0) = make_float4(x[0], x[1], x[2], x[3]);
        *reinterpret_cast<float4*>(mrow + e0 + 4) = make_float4(x[4], x[5], x[6], x[7]);
        uint4 u;
        u.x = (u32)f2u16(x[0] * r) | ((u32)f2u16(x[1] * r) << 16); u.y = (u32)f2u16(x[2] * r) | ((u32)f2u16(x[3] * r) << 16);
        u.z = (u32)f2u16(x[4] * r) | ((u32)f2u16(x[5] * r) << 16); u.w = (u32)f2u16(x[6] * r) | ((u32)f2u16(x[7] * r) << 16);
        *reinterpret_cast<uint4*>(urow + e0) = u;
      }
    }
    if (lane == 0) p.msgir[lg * MAXB + b] = r;
  }
  grid_bar_(bar, bar_target);
          {
            const int r0 = bid * 3072 / NB, r1 = (MSGV & 2) ? r0 : (bid + 1) * 3072 / NB;
            const int lga = r0 / 64, nlg = (r1 - 1) / 64 - lga + 1;   // <= 2
            u16* us = reinterpret_cast<u16*>(XF);                       // [nlg][B][H] bf16 (<= 73728 B)
            uint4 dv[9];   // the warp's first row, loaded before the staging; later rows (<= 3 per warp) in the loop
            auto drow = [&](int rr) {
              const int lg = rr / 64, j = rr % 64;
              return p.L[6 + lg / 4].cD + (size_t)(lg % 4) * 64 * 2 * H + H + (size_t)j * 2 * H;
            };
            if (!(MSGV & 4) && r0 + warp < r1) {
              const bf16* D = drow(r0 + warp);
#pragma unroll
              for (int c = 0; c < 9; c++) dv[c] = __ldg(reinterpret_cast<const uint4*>(D + (lane + 32 * c) * 8));
            }
            const int nst = nlg * B * (H / 8);
#if MSGV & 16
#pragma unroll 1
#endif
            for (int i = tid; i < nst; i += NT) {
              const int row = i / (H / 8), k8 = i % (H / 8), lgo = row / B, b = row % B;
              reinterpret_cast<uint4*>(us)[i] = ldcg128(reinterpret_cast<const u16*>(p.yb) + ((size_t)(lga + lgo) * MAXB + b) * H + k8 * 8);
            }
            __syncthreads();
            for (int rr = r0 + warp; rr < r1; rr += NW) {
              if ((MSGV & 4) || rr != r0 + warp) {
                const bf16* D = drow(rr);
#pragma unroll
                for (int c = 0; c < 9; c++) dv[c] = __ldg(reinterpret_cast<const uint4*>(D + (lane + 32 * c) * 8));
              }
              const int lg = rr / 64, j = rr % 64;
#if MSGV & 8
#pragma unroll 1
#endif
              for (int b = 0; b < B; b++) {
                const u16* ur = us + ((size_t)(lg - lga) * B + b) * H;
                float acc = 0.f;
#pragma unroll
                for (int c = 0; c < 9; c++) {
                  float w8[8], x8[8];
                  unpack8(dv[c], w8);
                  unpack8(*reinterpret_cast<const uint4*>(ur + (lane + 32 * c) * 8), x8);
#pragma unroll
                  for (int e = 0; e < 8; e++) acc = fmaf(w8[e], x8[e], acc);
                }
                acc = warp_sum(acc);
                if (lane == 0) p.zm[((size_t)b * 48 + lg) * 64 + j] = acc;
              }
            }
            __syncthreads();
          }
}

// GZ0 z rows of one sequence (out of line variant, GZV bit 2)
__device__ __noinline__ void gz_zrows(const Params& p, const bf16* cD, const float* XF, const float* rs, int b, int l, int g0, int zq0, int zq1, bool sbg) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int r = zq0 + warp; r < zq1; r += NW) {
    const int gg = g0 + r / 64, j = r % 64, sg = sbg ? r / 64 : 0;
    const bf16* drow = cD + ((size_t)gg * 64 + j) * 2 * H;
    const float* xp = XF + sg * H;
    uint4 dv[9];
#pragma unroll
    for (int q = 0; q < 9; q++) dv[q] = __ldg(reinterpret_cast<const uint4*>(drow + (lane + 32 * q) * 8));
    const float zm_ = ldcgf(p.zm + ((size_t)b * 48 + l * 4 + gg) * 64 + j);
    const float rx = rs[sg];
    float acc = 0.f;
#pragma unroll
    for (int q = 0; q < 9; q++) {
      const int e0 = (lane + 32 * q) * 8;
      float w8[8]; unpack8(dv[q], w8);
      const float4 xa = *reinterpret_cast<const float4*>(xp + e0), xb = *reinterpret_cast<const float4*>(xp + e0 + 4);
      const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
      for (int e = 0; e < 8; e++) acc = fmaf(w8[e], bfr(xv[e] * rx), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) p.zb[(b * 4 + gg) * 64 + j] = bfr(acc + zm_);
  }
}

// 1 query / 1 stream (local layers, MLP inputs, messages, head) or up to 3 / 3 (global layers): a third of the code
// for the common case (41% of mixnorm's stall samples were instruction-cache misses)
// the small version only in the B > 1 kernels (in the B = 1 kernel the second call site cost +55 us in allocation)
__device__ __forceinline__ void mixnorm_d(int Bk, const Params& p, int b, const int* codes, int nsrc, const float* const* qs, int nq,
                        int ns, const int* qsel, const bf16* const* nw, u16* const* xrow, float* XF, Misc* ms) {
#if MIXT
  if (Bk != 1 && nq == 1 && ns == 1) mixnorm_t<1, 1>(p, b, codes, nsrc, qs, nq, ns, qsel, nw, xrow, XF, ms);
  else
#endif
  mixnorm_t<3, 3>(p, b, codes, nsrc, qs, nq, ns, qsel, nw, xrow, XF, ms);
}
#define mixnorm(...) mixnorm_d(B, __VA_ARGS__)

// The same for B >= 2 sequences at once: warp w owns sequence w (no block barriers); three passes over the sources
// (dots and sums of squares; the streams' sums of squares; outputs), each lane 9 chunks of 8 elements
__device__ __noinline__ void mixnorm_w(const Params& p, int B, const int* codes, int nsrc, const float* const* qs, int nq,
                                       int ns, const int* qsel, const bf16* const* nw, u16* const* xrow0, int xrow_bstride, Misc* ms) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int b = warp;
  if (b < B) {
    const bool sum_last = codes[nsrc - 1] >= SRC_SUM;
    const bf16* sp[MAXSRC];
#pragma unroll
    for (int n = 0; n < MAXSRC; n++) sp[n] = n < nsrc ? src_ptr(p, codes[n], b) : nullptr;
    const bf16* asp = p.attn_s + (size_t)b * H;
    auto ld = [&](int n, int e0, float* v) {
      unpack8(ldcg128(sp[n] + e0), v);
      if (sum_last && n == nsrc - 1) { float w[8]; unpack8(ldcg128(asp + e0), w);
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] += w[j]; }
    };
    float part[MAXSRC][4];
#pragma unroll
    for (int n = 0; n < MAXSRC; n++) part[n][0] = part[n][1] = part[n][2] = part[n][3] = 0.f;
#pragma unroll 3
    for (int c = lane; c < H / 8; c += 32) {
      const int e0 = c * 8;
#pragma unroll
      for (int n = 0; n < MAXSRC; n++)
        if (n < nsrc) {
          float v[8]; ld(n, e0, v);
#pragma unroll
          for (int j = 0; j < 8; j++) part[n][0] = fmaf(v[j], v[j], part[n][0]);
#pragma unroll
          for (int q = 0; q < 3; q++)
            if (q < nq) {
              const float4 a0 = __ldg(reinterpret_cast<const float4*>(qs[q] + e0)), a1 = __ldg(reinterpret_cast<const float4*>(qs[q] + e0 + 4));
              part[n][1 + q] = fmaf(a0.x, v[0], fmaf(a0.y, v[1], fmaf(a0.z, v[2], fmaf(a0.w, v[3], fmaf(a1.x, v[4], fmaf(a1.y, v[5], fmaf(a1.z, v[6], fmaf(a1.w, v[7], part[n][1 + q]))))))));
            }
        }
    }
    float cf[3][MAXSRC];
#pragma unroll
    for (int n = 0; n < MAXSRC; n++)
#pragma unroll
      for (int j = 0; j < 4; j++) part[n][j] = warp_sum(part[n][j]);
#pragma unroll
    for (int q = 0; q < 3; q++)
      if (q < nq) {
        float lg[MAXSRC], ir[MAXSRC], m = -INFINITY;
#pragma unroll
        for (int n = 0; n < MAXSRC; n++) if (n < nsrc) { ir[n] = rsqrtf(part[n][0] * (1.0f / H) + 1e-6f); lg[n] = part[n][1 + q] * ir[n] * (1.0f / 48.0f); m = fmaxf(m, lg[n]); }
        float sum = 0.f;
#pragma unroll
        for (int n = 0; n < MAXSRC; n++) if (n < nsrc) { lg[n] = expf(lg[n] - m); sum += lg[n]; }
        const float inv = 1.0f / sum;
#pragma unroll
        for (int n = 0; n < MAXSRC; n++) cf[q][n] = n < nsrc ? lg[n] * inv * ir[n] : 0.f;
      }
    auto xv = [&](int q, int e0, float* x) {
#pragma unroll
      for (int j = 0; j < 8; j++) x[j] = 0.f;
#pragma unroll
      for (int n = 0; n < MAXSRC; n++)
        if (n < nsrc) { float v[8]; ld(n, e0, v);
#pragma unroll
          for (int j = 0; j < 8; j++) x[j] = fmaf(cf[q][n], v[j], x[j]); }
    };
    float ssx[3] = {0.f, 0.f, 0.f};
#pragma unroll 3
    for (int c = lane; c < H / 8; c += 32) {
#pragma unroll
      for (int s = 0; s < 3; s++)
        if (s < ns) { float x[8]; xv(qsel[s], c * 8, x);
#pragma unroll
          for (int j = 0; j < 8; j++) ssx[s] = fmaf(x[j], x[j], ssx[s]); }
    }
    float r[3];
#pragma unroll
    for (int s = 0; s < 3; s++) r[s] = rsqrtf(warp_sum(ssx[s]) * (1.0f / H) + 1e-6f);
#pragma unroll 3
    for (int c = lane; c < H / 8; c += 32) {
#pragma unroll
      for (int s = 0; s < 3; s++)
        if (s < ns) { float x[8]; xv(qsel[s], c * 8, x);
#pragma unroll
          for (int j = 0; j < 8; j++) x[j] *= r[s];
          store_row8(xrow0[s] + b * xrow_bstride, c * 8, x, __ldg(reinterpret_cast<const uint4*>(nw[s] + c * 8))); }
    }
  }
  __syncthreads();
}

// warp-per-sequence form of norm_f32_rows for B >= 2: src[s] + b * sstride, rows xrow0[s] + b * rstride
__device__ __noinline__ void norm_f32_rows_w(const float* const* src, int sstride, int B, int ns, const bf16* const* nw, u16* const* xrow0, int rstride) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, b = warp;
  if (b < B) {
    for (int s = 0; s < ns; s++) {
      const float* sp = src[s] + (size_t)b * sstride;
      float ss = 0.f;
#pragma unroll 3
      for (int c = lane; c < H / 8; c += 32) {
        float4 x0, x1;
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(x0.x), "=f"(x0.y), "=f"(x0.z), "=f"(x0.w) : "l"(sp + c * 8));
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(x1.x), "=f"(x1.y), "=f"(x1.z), "=f"(x1.w) : "l"(sp + c * 8 + 4));
        ss = fmaf(x0.x, x0.x, fmaf(x0.y, x0.y, fmaf(x0.z, x0.z, fmaf(x0.w, x0.w, fmaf(x1.x, x1.x, fmaf(x1.y, x1.y, fmaf(x1.z, x1.z, fmaf(x1.w, x1.w, ss))))))));
      }
      const float r = rsqrtf(warp_sum(ss) * (1.0f / H) + 1e-6f);
#pragma unroll 3
      for (int c = lane; c < H / 8; c += 32) {
        float4 x0, x1;
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(x0.x), "=f"(x0.y), "=f"(x0.z), "=f"(x0.w) : "l"(sp + c * 8));
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(x1.x), "=f"(x1.y), "=f"(x1.z), "=f"(x1.w) : "l"(sp + c * 8 + 4));
        float v[8] = {x0.x * r, x0.y * r, x0.z * r, x0.w * r, x1.x * r, x1.y * r, x1.z * r, x1.w * r};
        store_row8(xrow0[s] + b * rstride, c * 8, v, __ldg(reinterpret_cast<const uint4*>(nw[s] + c * 8)));
      }
    }
  }
  __syncthreads();
}

// RMSNorm with weight of fp32 vectors in global memory (one round of loads) into bf16 rows
__device__ __noinline__ void norm_f32_rows(const float* const* src, int ns, const bf16* const* nw, u16* const* xrow, Misc* ms) {
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nck = NCHK(tid);
  float x[3][2][8];
#pragma unroll
  for (int s = 0; s < 3; s++)
    if (s < ns) {
#pragma unroll
      for (int k = 0; k < 2; k++) if (k < nck) {
        const float* sp = src[s] + (tid + k * NT) * 8;
        float4 a, c;
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(a.x), "=f"(a.y), "=f"(a.z), "=f"(a.w) : "l"(sp));
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w) : "l"(sp + 4));
        x[s][k][0] = a.x; x[s][k][1] = a.y; x[s][k][2] = a.z; x[s][k][3] = a.w; x[s][k][4] = c.x; x[s][k][5] = c.y; x[s][k][6] = c.z; x[s][k][7] = c.w;
      }
    }
#pragma unroll
  for (int s = 0; s < 3; s++)
    if (s < ns) {
      float ss = 0.f;
#pragma unroll
      for (int k = 0; k < 2; k++) if (k < nck) {
#pragma unroll
        for (int j = 0; j < 8; j++) ss = fmaf(x[s][k][j], x[s][k][j], ss);
      }
      ss = warp_sum(ss);
      if (lane == 0) ms->wred[warp][s] = ss;
    }
  __syncthreads();
  if (tid < ns) { float t = 0.f;
#pragma unroll
    for (int w = 0; w < NW; w++) t += ms->wred[w][tid];
    ms->rs[tid] = rsqrtf(t * (1.0f / H) + 1e-6f); }
  __syncthreads();
#pragma unroll
  for (int s = 0; s < 3; s++)
    if (s < ns) {
      const float r = ms->rs[s];
#pragma unroll
      for (int k = 0; k < 2; k++) if (k < nck) {
        const int e0 = (tid + k * NT) * 8;
        float v[8];
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = x[s][k][j] * r;
        store_row8(xrow[s], e0, v, __ldg(reinterpret_cast<const uint4*>(nw[s] + e0)));
      }
    }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------ the kernel
// BSPEC: instantiated for B = 1, B = 8 (the batch size a compile-time constant: the other sizes' paths fold away)
// and a generic version (BC = 0: B from the arguments)
// ------------------------------------------------------------------------------------------------ SKH (track B)
// Certified int8 sketch of the unembedding for the greedy head (modes 1 / 2: generate() and the serving loop; mode 0,
// step() with logits, keeps the exact head). Port of the a2 engine's SKH (from s15-mini-b200-b2). Part 1: per row r an
// int8 code s_r q_r ~ w_r, x carried as sigma (d1 + d2 / 254) with int8 digits (sigma a power of two), s8 MMA through
// the weight rings -> an interval [lo_r, up_r] containing the fp32 value a_r the exact head computes; per rowtile the
// max of up, grid-wide per sequence L = max_r lo_r. A grid barrier. Part 2: with eps >= |a1cap(a) - 30 tanh(a / 30)|
// over every finite float a (measured exhaustively at load, mk_cap_eps) a rowtile is skipped only if
// 30 tanh(u / 30) + 2 eps + 1e-6 < 30 tanh(L / 30) (in double): then every row of it has a1cap(a_r) < a1cap(a_rL),
// strictly below the row that gave L, so it cannot be the argmax (the head's tie rule never sees it). The other
// rowtiles are recomputed with the head's own item chain (same operands, same order, same sum) -> the same values.
// State: p.skl (u32 [64]): [8 par + b] L keys of head parity par (reset by block 0 after the head's last barrier,
// reused two heads later); [32] exact rowtiles, [33] sketch heads, [35] max exact rowtiles in a head, [36] heads
// without a usable L (all exact), [40 + par] per-head count, [48 + min(n, 15)] histogram of exact rowtiles per head.
__device__ __forceinline__ float a1cap(float x) { return 30.0f * tanhf(x * (1.0f / 30.0f)); }   // as the exact head
#if SKH
#ifndef SK_STAGED
#define SK_STAGED 2   // candidates per block up to which they are staged (else one per warp through the rings)
#endif
#define SK_DST 2320                      // digit row stride (bytes): 580 words = 4 mod 32, conflict-free B fragments
#define SK_DIG_OFF (MAXB * XSH * 2)      // digit rows [MAXB][2][SK_DST] in xs, past the 8 x rows
static_assert(SK_DIG_OFF + 2 * MAXB * SK_DST <= XS_BYTES, "SKH digits must fit in xs");
#ifndef SK_FP
#define SK_FP 6.0e-4f   // |a_r - z_r| of the fp32 MMA chain <= SK_FP n_r |x| (2304 terms x 2^-23 = 2.75e-4, x2 margin)
#endif
struct SkS {           // in the red region (unused during the head)
  float u[96][MAXB];
  float sig[MAXB], rs[MAXB], xn[MAXB];
  double T[MAXB];
  float wr[NW][2 * MAXB];
  int cand[96]; int ncand;
};
static_assert(sizeof(SkS) <= RED_SLOTS * 512, "SkS must fit in red");
__device__ __forceinline__ void mma_s8(int* d, uint4 a, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 sk_fsort(float z) { u32 u = __float_as_uint(z); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
__device__ __forceinline__ float sk_funsort(u32 u) { return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u); }
__device__ __forceinline__ float sk_nmax(float a, float b) { return !(b <= a) ? b : a; }   // NaN propagates
// the sketch's s8 items through the warp's ring (tasks issued by prefetch_tasks<0>: 72 items of 512 B per rowtile)
template <class Fin>
__device__ __forceinline__ void consume_sk(TT* table, Ring& rg, PState& psm, u32 dig, Fin fin) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  PState ps = psm;
  const int nt = ps.nt;
  if (nt == 0) return;
  const TT* tt = table + warp * TPW;
  int ct = 0, ci = 0;
  TT cq = tt[0];
  const u32 d1 = dig + (2 * g) * SK_DST + 4 * t, d2 = d1 + SK_DST;   // columns g >= B read unused rows (in bounds)
  int a1[4] = {0, 0, 0, 0}, a2[4] = {0, 0, 0, 0};
  while (ct < nt) {
    const u32 sidx = rg.pos % RS;
    const int n = min(RI, cq.n - ci);
    mbar_wait(rg.bars + sidx * 8, (rg.pos / RS) & 1);
    const u32 st = rg.data + sidx * (RI * ITEM);
    const int kb = cq.k0 + ci;
    auto itm = [&](int i) {
      const uint4 q = lds128(st + i * IBYTES + lane * 16);
      const int k0 = (kb + i) * 32;
      mma_s8(a1, q, lds32(d1 + k0), lds32(d1 + k0 + 16));
      mma_s8(a2, q, lds32(d2 + k0), lds32(d2 + k0 + 16));
    };
    if (n == RI) {
#pragma unroll
      for (int i = 0; i < RI; i++) itm(i);
    } else {
#pragma unroll 1
      for (int i = 0; i < n; i++) itm(i);
    }
    __syncwarp();
    rg.pos++;
    if (ps.pt < nt) issue_stage<0>(tt, ps, rg, lane);
    ci += n;
    if (ci == cq.n) {
      fin(cq.slot, a1, a2);
#pragma unroll
      for (int q = 0; q < 4; q++) { a1[q] = 0; a2[q] = 0; }
      ci = 0;
      if (++ct < nt) cq = tt[ct];
    }
  }
  if (lane == 0) psm = ps;
  __syncwarp();
}
#endif

#if SKH
#ifdef MKPROF
#define SKMARK(t) FMARK(t)
#else
#define SKMARK(t) do {} while (0)
#endif
// the sketch head (modes 1, 2), out of line: the hot kernel keeps its register allocation. Returns (ring position,
// barrier target).
template <int BC>
__device__ __noinline__ uint2 sk_head(const Params& p, unsigned char* smem, const int st, const int B, Ring rg, u32 bar_target) {
  u16* xs = reinterpret_cast<u16*>(smem + SM_XS);
  float* red = reinterpret_cast<float*>(smem + SM_RED);
  Misc* ms = reinterpret_cast<Misc*>(smem + SM_MISC);
  TT* ttab = reinterpret_cast<TT*>(smem + SM_TT);
  PState* pst = reinterpret_cast<PState*>(smem + SM_PS);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, bid = blockIdx.x;
  const u32 bmask = (1u << B) - 1;
  u32* bar = p.bar;
    SkS& sk = *reinterpret_cast<SkS*>(red);
    unsigned char* dig = smem + SM_XS + SK_DIG_OFF;
    const int par = st & 1;
    const int g = lane >> 2, t = lane & 3;
    const int r0 = (int)((long)bid * RT_HEAD / NB), r1 = (int)((long)(bid + 1) * RT_HEAD / NB);
    SKMARK(22);
    {   // digits, in xs's own (fragment) order: the sketch's columns are stored in that order (eng.py), the dot
        // products are the same sums. Per sequence max |x| and |x|^2, then sigma, then 2 int8 digits per element.
      float mx[MAXB], ss[MAXB];
#pragma unroll
      for (int b = 0; b < MAXB; b++) {
        mx[b] = 0.f; ss[b] = 0.f;
        if (b < B)
          for (int j = tid; j < H / 8; j += NT) {
            const uint4 v = *reinterpret_cast<const uint4*>(xs + b * XSH + j * 8);
            const u32 w4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int q = 0; q < 4; q++) {
              const float lo = __uint_as_float(w4[q] << 16), hi = __uint_as_float(w4[q] & 0xffff0000u);
              mx[b] = fmaxf(mx[b], fmaxf(fabsf(lo), fabsf(hi))); ss[b] = fmaf(lo, lo, ss[b]); ss[b] = fmaf(hi, hi, ss[b]);
            }
          }
      }
#pragma unroll
      for (int b = 0; b < MAXB; b++) { mx[b] = warp_max(mx[b]); ss[b] = warp_sum(ss[b]); }
      if (lane == 0) {
#pragma unroll
        for (int b = 0; b < MAXB; b++) { sk.wr[warp][b] = ss[b]; sk.wr[warp][MAXB + b] = mx[b]; }
      }
      __syncthreads();
      SKMARK(26);
      if (tid < B) {
        float s2 = 0.f, m = 0.f;
        for (int w = 0; w < NW; w++) { s2 += sk.wr[w][tid]; m = fmaxf(m, sk.wr[w][MAXB + tid]); }
        // sigma = the smallest power of two with 127 sigma >= max|x| (exact comparisons): |x / sigma| <= 127, no
        // digit clamps, y = x * (1 / sigma) exact. Else sigma = 1: tiny x stays within the bound; huge x overflows
        // |x| -> infinite error terms -> no lower bound -> every rowtile exact.
        float sig = 1.0f, rs = 1.0f;
        if (m > 1e-30f && m < 1e30f) {
          sig = __uint_as_float(__float_as_uint(m * (1.0f / 127.0f)) & 0x7f800000u);
          while (127.0f * sig < m) sig *= 2.0f;
          while (127.0f * (0.5f * sig) >= m) sig *= 0.5f;
          rs = __uint_as_float((254u - (__float_as_uint(sig) >> 23)) << 23);
        }
        sk.sig[tid] = sig; sk.rs[tid] = rs; sk.xn[tid] = __fsqrt_ru(s2) * 1.00001f + 1e-30f;
      }
      __syncthreads();
      SKMARK(27);
      // one flat loop over (sequence, 8-element chunk): independent iterations. rint by the 1.5 * 2^23 shift (exact for
      // |y| < 2^22, round to nearest even = rintf); |y| <= 127 and |y - d1| <= 1/2 make the old clamps no-ops, so the
      // digits are the same: d1 = rint(y), d2 = rint((y - d1) * 254)
#pragma unroll 3
      for (int idx = tid; idx < B * (H / 8); idx += NT) {
        const int b = idx / (H / 8), j = idx - b * (H / 8);
        const float rs = sk.rs[b];
        const uint4 v = *reinterpret_cast<const uint4*>(xs + b * XSH + j * 8);
        const u32 w4[4] = {v.x, v.y, v.z, v.w};
        int i1[8], i2[8];
#pragma unroll
        for (int q = 0; q < 8; q++) {
          const float y = __fmul_rn(__uint_as_float((q & 1) ? (w4[q >> 1] & 0xffff0000u) : (w4[q >> 1] << 16)), rs);
          const float m1 = __fadd_rn(y, 12582912.0f);
          const float q1 = __fsub_rn(m1, 12582912.0f);
          const float m2 = __fadd_rn(__fmul_rn(__fsub_rn(y, q1), 254.0f), 12582912.0f);
          i1[q] = __float_as_int(m1) - 0x4B400000; i2[q] = __float_as_int(m2) - 0x4B400000;
        }
        const u32 a0 = __byte_perm(__byte_perm(i1[0], i1[1], 0x0040), __byte_perm(i1[2], i1[3], 0x0040), 0x5410);
        const u32 a1 = __byte_perm(__byte_perm(i1[4], i1[5], 0x0040), __byte_perm(i1[6], i1[7], 0x0040), 0x5410);
        const u32 b0 = __byte_perm(__byte_perm(i2[0], i2[1], 0x0040), __byte_perm(i2[2], i2[3], 0x0040), 0x5410);
        const u32 b1 = __byte_perm(__byte_perm(i2[4], i2[5], 0x0040), __byte_perm(i2[6], i2[7], 0x0040), 0x5410);
        *reinterpret_cast<uint2*>(dig + (2 * b) * SK_DST + j * 8) = make_uint2(a0, a1);
        *reinterpret_cast<uint2*>(dig + (2 * b + 1) * SK_DST + j * 8) = make_uint2(b0, b1);
      }
      __syncthreads();
    }
    SKMARK(23);
    // part 1: sketch GEMV of this block's rowtiles through the rings, bounds
    prefetch_tasks<0>(r1 - r0, ttab, rg, pst[warp], [&](int task, TD& d) {
      d.w = p.skq + (size_t)(r0 + task) * 72 * 32; d.x = xs; d.n = 72; d.slot = r0 + task; d.bm = bmask;
    });
    float lm0 = -INFINITY, lm1 = -INFINITY;
    consume_sk(ttab, rg, pst[warp], smaddr(dig), [&](int rt, const int* A1, const int* A2) {
      float up0 = -INFINITY, up1 = -INFINITY;
#pragma unroll
      for (int h = 0; h < 2; h++) {
        const int row = rt * 16 + g + 8 * h;
        const float sr = __ldg(p.sks + row), er = __ldg(p.ske + row), nr = __ldg(p.skn + row);
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
          const int tok = 2 * t + cc;
          if (tok >= B || row >= VOCAB) continue;
          // z_r = s_r q_r.x^ + (w_r - s_r q_r).x^ + w_r.(x - x^): |z_r - av| <= e_r |x^| + n_r |x - x^|;
          // |x - x^| <= 48 sigma (1 / 508 + 1e-5); |a_r - z_r| <= SK_FP n_r |x|; 1.01 and 1e-6: fp32 rounding of E, av
          const float sig = sk.sig[tok], xn = sk.xn[tok];
          const float dx = 48.0f * sig * (1.0f / 508.0f + 1e-5f);
          const float S = (float)A1[2 * h + cc] + (float)A2[2 * h + cc] * (1.0f / 254.0f);
          const float av = sr * sig * S;
          const float E = (er * (xn + dx) + nr * dx + SK_FP * nr * xn + 1e-6f * fabsf(av) + 1e-6f) * 1.01f;
          if (cc == 0) { up0 = sk_nmax(up0, av + E); lm0 = fmaxf(lm0, av - E); }
          else { up1 = sk_nmax(up1, av + E); lm1 = fmaxf(lm1, av - E); }
        }
      }
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) {
        up0 = sk_nmax(up0, __shfl_xor_sync(0xffffffffu, up0, o)); up1 = sk_nmax(up1, __shfl_xor_sync(0xffffffffu, up1, o));
      }
      if (g == 0) { sk.u[rt - r0][2 * t] = up0; sk.u[rt - r0][2 * t + 1] = up1; }
    });
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) { lm0 = fmaxf(lm0, __shfl_xor_sync(0xffffffffu, lm0, o)); lm1 = fmaxf(lm1, __shfl_xor_sync(0xffffffffu, lm1, o)); }
    if (g == 0 && 2 * t < B && lm0 > -INFINITY) atomicMax(p.skl + par * 8 + 2 * t, sk_fsort(lm0));
    if (g == 0 && 2 * t + 1 < B && lm1 > -INFINITY) atomicMax(p.skl + par * 8 + 2 * t + 1, sk_fsort(lm1));
    SKMARK(24);
    grid_bar_(bar, bar_target);
    SKMARK(25);
    // part 2: candidate rowtiles of this block
    if (tid < B) {
      __threadfence();
      const double L = (double)sk_funsort(ldcg32(p.skl + par * 8 + tid));   // never set: NaN
      const double cl = 30.0 * tanh(L / 30.0) - 2.0 * (double)__ldg(p.skeps) - 1e-6;
      sk.T[tid] = (cl > -30.0) ? 30.0 * atanh(cl / 30.0) : -INFINITY;     // (NaN: no rowtile skipped)
      if (!(cl > -30.0) && bid == 0) atomicAdd(p.skl + 36, 1u);
    }
    if (tid == 0) sk.ncand = 0;
    __syncthreads();
    if (tid < r1 - r0) {
      bool cand = false;
      for (int b = 0; b < B; b++) if (!((double)sk.u[tid][b] < sk.T[b])) cand = true;   // (NaN u: exact)
      if (cand) { const int q = atomicAdd(&sk.ncand, 1); sk.cand[q] = r0 + tid; }
    }
    __syncthreads();
    const int ncand = sk.ncand;
    if (tid == 0 && ncand) { atomicAdd(p.skl + 32, (u32)ncand); atomicAdd(p.skl + 40 + par, (u32)ncand); }
    // exact recompute: each candidate rowtile staged in the (idle) ring memory by the whole block, then the head's
    // item chain (item<0>, items 0..143 into accumulator 0, the same final sum) and the head's argmax by warp 0
    float hb0 = -INFINITY, hb1 = -INFINITY; int hi0 = 0x7fffffff, hi1 = 0x7fffffff;
    auto amax_fin = [&](int rt, const float* c) {   // = the head's argmax (mode != 0), verbatim
      const int row0 = rt * 16 + g, row1 = row0 + 8;
      const float v0 = 30.0f * tanhf(c[0] * (1.0f / 30.0f)), v1 = 30.0f * tanhf(c[1] * (1.0f / 30.0f));
      const float v2 = 30.0f * tanhf(c[2] * (1.0f / 30.0f)), v3 = 30.0f * tanhf(c[3] * (1.0f / 30.0f));
      if (row0 < VOCAB) { if (v0 > hb0 || (v0 == hb0 && row0 < hi0)) { hb0 = v0; hi0 = row0; } if (v1 > hb1 || (v1 == hb1 && row0 < hi1)) { hb1 = v1; hi1 = row0; } }
      if (row1 < VOCAB) { if (v2 > hb0 || (v2 == hb0 && row1 < hi0)) { hb0 = v2; hi0 = row1; } if (v3 > hb1 || (v3 == hb1 && row1 < hi1)) { hb1 = v3; hi1 = row1; } }
    };
    if (ncand > SK_STAGED) {
      // many candidates in this block: one per warp through its ring, the head's own task runner and chain
      prefetch_tasks<0>(ncand, ttab, rg, pst[warp], [&](int task, TD& d) {
        const int rt = sk.cand[task];
        d.w = p.unembed + (size_t)rt * KS_H * 32; d.x = xs; d.n = KS_H; d.slot = rt; d.bm = bmask;
      });
      consume_tasks<0>(ttab, XSH, rg, pst[warp], amax_fin);
    } else {
      // few: each staged in the (idle) ring memory by the whole block in one round of loads, then the head's item
      // chain (item<0>, items 0..143 into accumulator 0, the same final sum) by warp 0
      unsigned char* stg = smem + SM_RING;
      for (int q = 0; q < ncand; q++) {
        const int rt = sk.cand[q];
        const uint4* src = p.unembed + (size_t)rt * KS_H * 32;
        for (int k = tid; k < KS_H * 32; k += NT) reinterpret_cast<uint4*>(stg)[k] = ldcg128(src + k);
        __syncthreads();
        if (warp == 0) {
          float c[NACC][4];
#pragma unroll
          for (int aa = 0; aa < NACC; aa++) c[aa][0] = c[aa][1] = c[aa][2] = c[aa][3] = 0.f;
          const u32 cx = smaddr(xs + g * XSH) + t * 32, sb = smaddr(stg);
#pragma unroll 4
          for (int i = 0; i < KS_H; i++) item<0>(c, sb, cx, 0, i, lane, g);
          float r[4];
#pragma unroll
          for (int qq = 0; qq < 4; qq++) { r[qq] = (c[0][qq] + c[1][qq]) + (c[2][qq] + c[3][qq]); }
          amax_fin(rt, r);
        }
        __syncthreads();
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // generic accesses before the rings' next bulk copies
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, hb0, o); int oi = __shfl_xor_sync(0xffffffffu, hi0, o);
      if (ov > hb0 || (ov == hb0 && oi < hi0)) { hb0 = ov; hi0 = oi; }
      ov = __shfl_xor_sync(0xffffffffu, hb1, o); oi = __shfl_xor_sync(0xffffffffu, hi1, o);
      if (ov > hb1 || (ov == hb1 && oi < hi1)) { hb1 = ov; hi1 = oi; }
    }
    if (lane < 4) {
      ms->tmpf[warp * 8 + 2 * lane] = hb0; ms->tmpi[warp * 8 + 2 * lane] = hi0;
      ms->tmpf[warp * 8 + 2 * lane + 1] = hb1; ms->tmpi[warp * 8 + 2 * lane + 1] = hi1;
    }
    __syncthreads();
    if (tid < B) {
      float v = -INFINITY; int ix = 0x7fffffff;
      for (int w = 0; w < NW; w++) {
        const float ov = ms->tmpf[w * 8 + tid]; const int oi = ms->tmpi[w * 8 + tid];
        if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
      }
      p.amv[bid * MAXB + tid] = v; p.ami[bid * MAXB + tid] = ix;
    }
    grid_bar_(bar, bar_target);
    if (bid == 0) {
      if (tid < MAXB) p.skl[par * 8 + tid] = 0u;   // next used two heads later, behind this step's barriers
      if (tid == 32) {
        const u32 n = atomicExch(p.skl + 40 + par, 0u);
        atomicAdd(p.skl + 33, 1u); atomicMax(p.skl + 35, n); atomicAdd(p.skl + 48 + min(n, 15u), 1u);
      }
    }
  return make_uint2(rg.pos, bar_target);
}
#endif

template <int BC>
__global__ void __launch_bounds__(NT, 1) mk_kernel_t(const __grid_constant__ Params p, const __grid_constant__ RunArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  u16* xs = reinterpret_cast<u16*>(smem + SM_XS);
  float* red = reinterpret_cast<float*>(smem + SM_RED);
  float* XF = reinterpret_cast<float*>(smem + SM_XF);       // fp32 [3][H] mix outputs
  Misc* ms = reinterpret_cast<Misc*>(smem + SM_MISC);
  TT* ttab = reinterpret_cast<TT*>(smem + SM_TT);
  PState* pst = reinterpret_cast<PState*>(smem + SM_PS);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  Ring rg;
  rg.data = smaddr(smem + SM_RING) + warp * RING_WARP;
  rg.bars = smaddr(smem + SM_BARS) + warp * RS * 8;
  rg.pos = 0;
  if (lane == 0) for (int k = 0; k < RS; k++) mbar_init(rg.bars + k * 8, 1);
  if (lane == 0) for (int k = 0; k < 3; k++) { mbar_init(smaddr(smem + SM_BARS) + 128 + (warp * 3 + k) * 8, 1); ms->dpc[warp][k] = 0; }
#if SCIM == 2
  if (tid == 0) s_mul20 = 1048576u;
#endif
  if (tid == 0) { mbar_init(smaddr(&ms->sbar), 1); ms->sphase = 0; mbar_init(smaddr(&ms->sbar2), 1); ms->sphase2 = 0; }
  if (tid < MAXU) ms->stgt[tid] = 0;
  if (tid < 8) { ms->sexp[tid] = tid; ms->smask[tid] = 1; ms->f13[tid] = ms->f2[tid] = 1.f; for (int b = 0; b < MAXB; b++) ms->rw[b][tid] = 1.f; }
  if (tid == 0) ms->U = 8;
  if (tid <= MAXU) ms->pcnt[tid] = tid < 8 ? tid : 8;
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();
  const int bid = blockIdx.x;
  const int B = BC > 0 ? BC : a.B;
  constexpr int RKQB = RKQ > 0 ? RKQ : (BC == 1 ? 3 : 2);   // router K parts (8 RKQB router blocks): B=1 -38, B=8 -16 (lone-process)
  const u32 bmask = (1u << B) - 1;
  u32 bar_target = 0;
  u32* bar = p.bar;
  MkProf prof{p.prof, (unsigned)p.prof_cap};
  MKPROF_DECL(pn);
#ifdef MKPROF
#define MARK(tag) FMARK(tag)
  if (tid == 0) { s_prof_buf = prof.buf; s_prof_cap = prof.cap; s_pn = 0; }
  __syncthreads();
#else
#define MARK(tag) do {} while (0)
#endif
#ifdef SKEL
  g_skel = p.skel;
#endif

  // ---- prefetch helpers (weights only)
  auto qkv_range = [&](const LayerP& L, int& r0, int& r1) {
    r0 = bid * (int)L.nrt / NB; r1 = (bid + 1) * (int)L.nrt / NB;
  };
  auto stream_of = [&](const LayerP& L, int rt) { return rt < RT_K ? 0 : (rt < (L.glob ? 476 : 474) ? 1 : 2); };
  auto pf_qkv = [&](int i) {
    const LayerP& L = p.L[i];
    int r0, r1; qkv_range(L, r0, r1);
    const int s0 = stream_of(L, r0);
    prefetch_tasks<0>((r1 - r0) * 8, ttab, rg, pst[warp], [&](int task, TD& d) {
      const int r = task >> 3, sg = task & 7;
      const int grp = stream_of(L, r0 + r) - s0;
      d.w = L.wqkv + ((size_t)(r0 + r) * KS_H + sg * 18) * 32; d.x = xs + grp * B * XSH; d.k0 = sg * 18; d.n = 18;
      d.slot = task; d.bm = bmask;
    });
  };
  auto pf_o = [&](int i) {
    const LayerP& L = p.L[i];
    const int R = bid < 144 ? 1 : 0;
    prefetch_tasks<0>(R * 8, ttab, rg, pst[warp], [&](int task, TD& d) {
      d.w = L.wo + ((size_t)bid * KS_A + task * 24) * 32; d.x = xs; d.k0 = task * 24; d.n = 24; d.slot = task; d.bm = bmask;
    });
  };
  auto pf_router = [&](int i) {
    const LayerP& L = p.L[i];
    constexpr int KN = 18 / RKQB;
    const int rt = bid % 8, kq = bid / 8;
    prefetch_tasks<0>(bid < 8 * RKQB ? 8 : 0, ttab, rg, pst[warp], [&](int task, TD& d) {
      const int k0 = kq * (KS_H / RKQB) + task * KN;
      d.w = L.wr + ((size_t)rt * KS_H + k0) * 32; d.x = xs; d.k0 = k0; d.n = KN; d.slot = task; d.bm = bmask;
    });
  };
  auto pf_head = [&]() {
    const int r0 = (int)((long)bid * RT_HEAD / NB), r1 = (int)((long)(bid + 1) * RT_HEAD / NB);
    prefetch_tasks<0>(r1 - r0, ttab, rg, pst[warp], [&](int task, TD& d) {
      d.w = p.unembed + (size_t)(r0 + task) * KS_H * 32; d.x = xs; d.n = KS_H; d.slot = r0 + task; d.bm = bmask;
    });
  };
  // down-projection rounds: slots [s0, s1) whose (slot, token) activation rows fit in xs (32 rows)
  // down-projection rounds: slots [s0, s1) whose (slot, token) activation rows fit in xs (32 rows); pcnt (prefix of
  // the pairs per slot) is computed once after the union; every slot's row map is built by its own thread
  auto fround = [&](int s0) {
    if (tid == 0) {
      const int U = ms->U;
      int s1 = s0;
      while (s1 < U && ms->pcnt[s1 + 1] - ms->pcnt[s0] <= 32 && (s1 + 1 - s0) * 2 <= RED_SLOTS) s1++;
      ms->fr_s0 = s0; ms->fr_s1 = s1; ms->fr_np = ms->pcnt[s1] - ms->pcnt[s0];
    }
    __syncthreads();
    const int a0 = ms->fr_s0, a1 = ms->fr_s1;
    if (tid < a1 - a0) {
      const int k = a0 + tid;
      const u32 m = ms->smask[k];
      int base = ms->pcnt[k] - ms->pcnt[a0];
      u64 r8 = 0;
#pragma unroll
      for (int b = 0; b < 8; b++) if (m >> b & 1) r8 |= (u64)(base++) << (8 * b);
      ms->srows[tid] = r8;
    }
  };
  auto pf_down = [&](int i, int s0, int s1) {
    const LayerP& L = p.L[i];
    const int ntask = bid < 144 ? (s1 - s0) * 2 : 0;
    if (L.mx) {
      prefetch_tasks<2>(ntask, ttab, rg, pst[warp], [&](int task, TD& d) {
        const int sl = s0 + (task >> 1), sg = task & 1, e = ms->sexp[sl];
        const size_t it = ((size_t)e * 144 + bid) * 36 + sg * 18;
        d.w = L.w2c + it * 32; d.s = L.w2s + it * 16; d.x = xs; d.k0 = sg * 18; d.n = 18; d.slot = task;
        d.rows8 = ms->srows[sl - s0]; d.bm = ms->smask[sl];
      });
    } else {
      prefetch_tasks<1>(ntask, ttab, rg, pst[warp], [&](int task, TD& d) {
        const int sl = s0 + (task >> 1), sg = task & 1, e = ms->sexp[sl];
        const size_t it = ((size_t)e * 144 + bid) * 18 + sg * 9;
        d.w = L.w2c + it * 32; d.s = L.w2s + it * 64; d.x = xs; d.k0 = sg * 9; d.n = 9; d.slot = task;
        d.rows8 = ms->srows[sl - s0]; d.bm = ms->smask[sl];
      });
    }
  };

  __syncthreads();

  // mixer blocks (B >= 2): block b < B computes sequence b's mix and publishes the normalised rows to global memory,
  // the other blocks wait for its flag (an event counter per sequence, reset at the end of the launch) and bulk-copy
  u32* flg = p.bar + 64;   // mixer flag of sequence b at flg[32 b] (one 128-byte line each)
  u32 mev = 0;
  // row k = (stream k / B, sequence k % B): src = sbase + b sbs + (k / B) sss -> xs + k XSH (no pointer arrays: stack
  // arrays live in local memory = an L2 round trip); lane k of warp 0 issues row k (nrow <= 24)
  auto mix_wait_rows = [&](u32 ev, int nrow, const bf16* sbase, int sbs, int sss) {
    if (warp == 0) {   // lane b waits for the flag of sequence b (all flags polled at once, not one round trip each)
      if (lane < B && ld_acquire(flg + 32 * lane) < ev) { const long long t0 = clock64(); while (ld_acquire(flg + 32 * lane) < ev) watchdog(t0, 4); }
      __syncwarp();
      if (lane == 0) { asm volatile("fence.proxy.async.global;" ::: "memory"); mbar_expect(smaddr(&ms->sbar), nrow * H * 2); }
      __syncwarp();
      if (lane < nrow) {
        const int b = lane % B, sk = lane / B;
        tma_load_plain(smaddr(xs + lane * XSH), sbase + (size_t)b * sbs + (size_t)sk * sss, H * 2, smaddr(&ms->sbar));
      }
    }
  };
  auto mix_wait_rows_done = [&]() {
    mbar_wait(smaddr(&ms->sbar), ms->sphase & 1);
    __syncthreads();
    if (tid == 0) ms->sphase++;
  };
  auto mix_publish = [&]() {   // after the mixer block wrote its rows
#if PUB_REL
    __syncthreads();   // the block's row stores, then one release increment (cumulative over the barrier)
    if (tid == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(flg + 32 * MIXB(bid)) : "memory");
#else
    __threadfence();
    __syncthreads();
    if (tid == 0) atomicAdd(flg + 32 * MIXB(bid), 1u);
#endif
  };
  for (int st = 0; st < a.nsteps; st++) {
    const int pos = a.pos0 + st;
    MARK(MKPROF_STEP);
    const int P = a.nprompt, ngen = a.nsteps - P + 1;
    // input token of this step: the prompt, or the argmax of the last step's logits (partials of all blocks)
    if (warp < B) {
      const int b = warp;
      int tk;
      if (st < P) { const long long t = a.tok[(size_t)b * P + st]; tk = (t >= 0 && t < 201024) ? (int)t : 0; }
      else {
        float bv = -INFINITY; int bi = 0x7fffffff;
        for (int k = lane; k < NB; k += 32) {
          const float v = ldcgf(p.amv + k * MAXB + b); const int ix = ldcgi(p.ami + k * MAXB + b);
          if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
          if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        tk = (bi >= 0 && bi < VOCAB) ? bi : 0;
        if (bid == 0 && lane == 0) {
          if (a.mode == 1) a.out_tok[(size_t)b * ngen + (st - P)] = tk;
          else if (a.mode == 2) a.host_tok[b * ngen + (st - P)] = tk;
        }
      }
      if (lane == 0) ms->tok[b] = tk;
    }
    __syncthreads();
    if (a.mode == 2 && st >= P && bid == 0 && tid == 0) { __threadfence_system(); *a.host_prog = st - P + 1; }
    const int nstop = (st == a.nsteps - 1) ? a.stop_app : NAPP;
    if (PFE_QKV) pf_qkv(0);
    for (int ap = 0; ap < nstop; ap++) {
      const int i = app_layer(ap), pas = app_pass(ap), slot = cache_slot(i, pas);
      const LayerP& L = p.L[i];
      const bool loopg = L.loop && pas > 0;
      const bool bnd = (i % 6) == 0;
      const int seg = i / 6;
      const int pp = (seg == 3) ? 2 : pas;            // pass whose P1/P2 the layer sees
      int* blk = ms->ablk; int nblk = 0;
      blk[nblk++] = V_H0;
      if (seg >= 1) blk[nblk++] = V_P0;
      if (seg >= 2) blk[nblk++] = V_P1 + pp;
      if (seg >= 3) blk[nblk++] = V_P2 + pp;
      const int cur = seg == 0 ? V_P0 : seg == 1 ? V_P1 + pas : seg == 2 ? V_P2 + pas : V_P3;
      // attention-input sources
      int* asrc = ms->aasrc; int nas = 0;
      for (int k = 0; k < nblk; k++) asrc[nas++] = blk[k];
      if (!bnd) asrc[nas++] = cur;
      const int naq = L.glob ? 3 : 1;
      const float** aqs = ms->aaqs; aqs[0] = L.aq; aqs[1] = L.aq + H; aqs[2] = L.aq + 2 * H;
      const bf16** nws = ms->anws; nws[0] = L.nq; nws[1] = L.nk; nws[2] = L.nv;
      ms->aq012[0] = 0; ms->aq012[1] = 1; ms->aq012[2] = 2; ms->azero[0] = 0;

      // ================= [M] messages of the pass and the message halves of the gate inputs
      // msg[l*4+g][b] = attention over depth of the previous pass (blocks after the pre-loop ones + final partial);
      // zm[b][l*4+g][j] = D[l][g][j][H:] . bf16(unit(msg)) (the x half is added in A0 / D0)
      if (L.loop && pas > 0 && i == 6) {
        MARK(1);
        if (!SKG && !SKG3 && MSGW && B > 1 && !(MSGK && p.prof_cap == -77777)) {   // MSGK: old path stays compiled
          msg_w(p, B, pas, XF, bar, bar_target);
        } else if (!SKG && !SKG3) {
        int* msrc = ms->amsg; msrc[0] = V_P1 + pas - 1; msrc[1] = V_P2 + pas - 1;
        const int T = B * 48 * 64;
        const int t0 = bid * T / NB, t1 = (bid + 1) * T / NB;
        u16* um = reinterpret_cast<u16*>(XF + H);
        for (int bl = t0 / 64; bl * 64 < t1; bl++) {
          const int b = bl / 48, lg = bl % 48, ll = lg / 4, gg = lg % 4;
          ms->aqp[0] = p.lq + (size_t)lg * H;
          mixnorm(p, b, msrc, 2, ms->aqp, 1, 1, ms->azero, nullptr, nullptr, XF, ms);
          const float r = ms->rs[0];
          const bool owner = t0 <= bl * 64;
          for (int k = tid; k < H; k += NT) {
            const float m = XF[k];
            um[k] = f2u16(m * r);
            if (owner) p.msg[((size_t)lg * MAXB + b) * H + k] = m;
          }
          if (owner && tid == 0) p.msgir[lg * MAXB + b] = r;
          __syncthreads();
          const int ja = max(t0, bl * 64) - bl * 64, jb = min(t1, bl * 64 + 64) - bl * 64;
          const bf16* D = p.L[6 + ll].cD + (size_t)gg * 64 * 2 * H + H;
          for (int j = ja + warp; j < jb; j += NW) {
            float acc = 0.f;
#pragma unroll 3
            for (int c = lane; c < H / 8; c += 32) {
              float w8[8], x8[8];
              unpack8(__ldg(reinterpret_cast<const uint4*>(D + (size_t)j * 2 * H + c * 8)), w8);
              unpack8(*reinterpret_cast<const uint4*>(um + c * 8), x8);
#pragma unroll
              for (int q = 0; q < 8; q++) acc = fmaf(w8[q], x8[q], acc);
            }
            acc = warp_sum(acc);
            if (lane == 0) p.zm[((size_t)b * 48 + lg) * 64 + j] = acc;
          }
          __syncthreads();
        }
        }
        grid_bar(bar, bar_target);
      }

      // ================= [A0], [A1] loop message gates for q, k, v ; [D0], [D1] for the MLP input
      // gate g of loop layer l: x' = (1 - tanh(F z)) x + tanh(G z) m, z = bf16(D [unit(x), unit(m)]) -> xg[b][g]
      auto gate_phases = [&](int g0, int ng, const int* srcs, int nsrc, const float* const* qs, int nq, bool sidx_by_gate) {
        // ---- phase 0: mixes, this block's z rows, this block's slices of x
        MARK(g0 == 0 ? 2 : 8);
        ZTS(0);
        if (SKG1) { grid_bar(bar, bar_target); }
        else {
        const int l = i - 6;
        const int nz = ng * 64;
        const int zr0 = bid * nz / NB, zr1 = (bid + 1) * nz / NB;   // at most 2 rows
        const int j0 = bid * H / NB, j1 = (bid + 1) * H / NB;
        const int nsm = sidx_by_gate ? ng : 1;
        const int* qsel = ms->aq012;
        // D rows (x half) of this block: issued before the mixes
        const int nck = NCHK(tid);
        uint4 dw[2][2];
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int k = 0; k < 2; k++)
            if (zr0 + r < zr1 && k < nck) {
              const int gg = g0 + (zr0 + r) / 64, j = (zr0 + r) % 64;
              dw[r][k] = __ldg(reinterpret_cast<const uint4*>(L.cD + ((size_t)gg * 64 + j) * 2 * H + (tid + k * NT) * 8));
            }
        if (GZ0P0 && B > 1 && !(GZK && p.prof_cap == -77777)) {   // GZK: the old path stays compiled (never taken)
          // merge run (GZ0): every block mixes one sequence b = bid % B itself (NB / B blocks per sequence), takes that
          // sequence's z rows from its own copy in smem (the same arithmetic as the published-row path: bitwise equal)
          // and writes its share of the columns of xm; the slices for phase 1 are loaded there (after the barrier).
          // No mixer -> flag hand-off and no xm round trip before the z rows.
          const int b = bid % B, kb = bid / B, nkb = (NB - b + B - 1) / B;
          const int zq0 = kb * nz / nkb, zq1 = (kb + 1) * nz / nkb;   // this block's z rows of sequence b
          if (!(GZV & 16)) mixnorm(p, b, srcs, nsrc, qs, nq, nsm, qsel, nullptr, nullptr, XF, ms);
          if (!(GZV & 8)) {
            const int c0 = kb * H / nkb, c1 = (kb + 1) * H / nkb;
            for (int k = tid; k < nsm * (c1 - c0); k += NT) {
              const int sg = k / (c1 - c0), col = c0 + k % (c1 - c0);
              p.xm[((size_t)b * 4 + g0 + sg) * H + col] = XF[sg * H + col];
            }
            if (kb == 0 && tid < nsm) p.rrx[b * 4 + g0 + tid] = ms->rs[tid];
          }
#if GZV & 4
          gz_zrows(p, L.cD, XF, ms->rs, b, l, g0, zq0, zq1, sidx_by_gate);
#else
#if GZV & 1
#pragma unroll 1
#endif
          for (int r = zq0 + warp; r < zq1; r += NW) {
            const int gg = g0 + r / 64, j = r % 64, sg = sidx_by_gate ? r / 64 : 0;
            const bf16* drow = L.cD + ((size_t)gg * 64 + j) * 2 * H;
            const float* xp = XF + sg * H;
            const float zm_ = ldcgf(p.zm + ((size_t)b * 48 + l * 4 + gg) * 64 + j);
            const float rx = ms->rs[sg];
            float acc = 0.f;
#if GZV & 2
#pragma unroll 1
            for (int q3 = 0; q3 < 9; q3 += 3) {
              uint4 dv[3];
#pragma unroll
              for (int q = 0; q < 3; q++) dv[q] = __ldg(reinterpret_cast<const uint4*>(drow + (lane + 32 * (q3 + q)) * 8));
#pragma unroll
              for (int q = 0; q < 3; q++) {
                const int e0 = (lane + 32 * (q3 + q)) * 8;
                float w8[8]; unpack8(dv[q], w8);
#else
            uint4 dv[9];
#pragma unroll
            for (int q = 0; q < 9; q++) dv[q] = __ldg(reinterpret_cast<const uint4*>(drow + (lane + 32 * q) * 8));
            {
#pragma unroll
              for (int q = 0; q < 9; q++) {
                const int e0 = (lane + 32 * q) * 8;
                float w8[8]; unpack8(dv[q], w8);
#endif
                const float4 xa = *reinterpret_cast<const float4*>(xp + e0), xb = *reinterpret_cast<const float4*>(xp + e0 + 4);
                const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
                for (int e = 0; e < 8; e++) acc = fmaf(w8[e], bfr(xv[e] * rx), acc);
              }
            }
            acc = warp_sum(acc);
            if (lane == 0) p.zb[(b * 4 + gg) * 64 + j] = bfr(acc + zm_);
          }
#endif
          __syncthreads();
        } else if (B > 1 && (!GZ0P0 || GZK)) {
          // mixer block b publishes sequence b's raw mix (fp32) and 1/rms; every block then takes its z rows for all
          // sequences in one pass and one block reduction, and its slices from the published mixes
          ++mev;
          if (MIXB(bid) >= 0 && MIXB(bid) < B) {
            mixnorm(p, MIXB(bid), srcs, nsrc, qs, nq, nsm, qsel, nullptr, nullptr, XF, ms);
            for (int k = tid; k < nsm * H; k += NT) p.xm[((size_t)MIXB(bid) * 4 + g0 + k / H) * H + k % H] = XF[k];
            if (tid < nsm) p.rrx[MIXB(bid) * 4 + g0 + tid] = ms->rs[tid];
            ZTS(1);
            mix_publish();
            ZTS(2);
          }
          float zmv[MAXB];
          for (int b = 0; b < B; b++) zmv[b] = (tid < zr1 - zr0) ? ldcgf(p.zm + ((size_t)b * 48 + l * 4 + g0 + (zr0 + tid) / 64) * 64 + (zr0 + tid) % 64) : 0.f;
          if (warp == 0 && lane < B && ld_acquire(flg + 32 * lane) < mev) { const long long t0 = clock64(); while (ld_acquire(flg + 32 * lane) < mev) watchdog(t0, 5); }
          __syncthreads();
          ZTS(3);
          // z rows: one warp per (row, sequence) pair, every lane's 9 chunks loaded in one batch
          {
            const int nr = zr1 - zr0;
            for (int pr_ = warp; pr_ < nr * B; pr_ += NW) {
              const int r = zr0 + pr_ / B, b = pr_ % B;
              const int gg = g0 + r / 64, j = r % 64, sg = sidx_by_gate ? r / 64 : 0;
              const bf16* drow = L.cD + ((size_t)gg * 64 + j) * 2 * H;
              const float* xp = p.xm + ((size_t)b * 4 + g0 + sg) * H;
              uint4 dv[9]; float4 xa[9], xb[9];
#pragma unroll
              for (int q = 0; q < 9; q++) {
                const int e0 = (lane + 32 * q) * 8;
                dv[q] = __ldg(reinterpret_cast<const uint4*>(drow + e0));
                xa[q] = __ldcg(reinterpret_cast<const float4*>(xp + e0));
                xb[q] = __ldcg(reinterpret_cast<const float4*>(xp + e0 + 4));
              }
              const float rx = ldcgf(p.rrx + b * 4 + g0 + sg);
              const float zm_ = ldcgf(p.zm + ((size_t)b * 48 + l * 4 + gg) * 64 + j);
              float acc = 0.f;
#pragma unroll
              for (int q = 0; q < 9; q++) {
                float w8[8]; unpack8(dv[q], w8);
                const float xv[8] = {xa[q].x, xa[q].y, xa[q].z, xa[q].w, xb[q].x, xb[q].y, xb[q].z, xb[q].w};
#pragma unroll
                for (int e = 0; e < 8; e++) acc = fmaf(w8[e], bfr(xv[e] * rx), acc);
              }
              acc = warp_sum(acc);
              if (lane == 0) p.zb[(b * 4 + gg) * 64 + j] = bfr(acc + zm_);
            }
          }
          for (int it = tid; it < B * ng * (j1 - j0); it += NT) {
            const int b = it / (ng * (j1 - j0)), gg = (it / (j1 - j0)) % ng, k = j0 + it % (j1 - j0);
            ms->sl[b][gg][k - j0] = ldcgf(p.xm + ((size_t)b * 4 + g0 + (sidx_by_gate ? gg : 0)) * H + k);
          }
          (void)zmv;
          __syncthreads();
          ZTS(4);
        } else
        for (int b = 0; b < B; b++) {
          float zmv = 0.f;
          if (tid < zr1 - zr0) zmv = ldcgf(p.zm + ((size_t)b * 48 + l * 4 + g0 + (zr0 + tid) / 64) * 64 + (zr0 + tid) % 64);
          mixnorm(p, b, srcs, nsrc, qs, nq, nsm, qsel, nullptr, nullptr, XF, ms);
          float pr[2] = {0.f, 0.f};
#pragma unroll
          for (int r = 0; r < 2; r++)
            if (zr0 + r < zr1) {
              const int sg = sidx_by_gate ? (zr0 + r) / 64 : 0;
              const float rx = ms->rs[sg];
#pragma unroll
              for (int k = 0; k < 2; k++) if (k < nck) {
                const int e0 = (tid + k * NT) * 8;
                float w8[8]; unpack8(dw[r][k], w8);
#pragma unroll
                for (int q = 0; q < 8; q++) pr[r] = fmaf(w8[q], bfr(XF[sg * H + e0 + q] * rx), pr[r]);
              }
              const float v = warp_sum(pr[r]);
              if (lane == 0) ms->wred[warp][r] = v;
            }
          for (int it = tid; it < ng * (j1 - j0); it += NT) {
            const int gg = it / (j1 - j0), k = j0 + it % (j1 - j0);
            ms->sl[b][gg][k - j0] = XF[(sidx_by_gate ? gg : 0) * H + k];
          }
          __syncthreads();
          if (tid < zr1 - zr0) {
            const int r = zr0 + tid, gg = g0 + r / 64, j = r % 64;
            float t = 0.f;
#pragma unroll
            for (int w = 0; w < NW; w++) t += ms->wred[w][tid];
            p.zb[(b * 4 + gg) * 64 + j] = bfr(t + zmv);
          }
          __syncthreads();
        }
        // F / G rows of this block's slice into smem (red) for phase 1: [ng][slice][64] each
        bf16* Fs = reinterpret_cast<bf16*>(red);
        bf16* Gs = Fs + 3 * 16 * 64;
        if (tid == 0) {
          const u32 sb = smaddr(&ms->sbar);
          mbar_expect(sb, 2 * ng * (j1 - j0) * 128);
          for (int gg = 0; gg < ng; gg++) {
            tma_load_plain(smaddr(Fs + gg * 16 * 64), L.cF + ((size_t)(g0 + gg) * H + j0) * 64, (j1 - j0) * 128, sb);
            tma_load_plain(smaddr(Gs + gg * 16 * 64), L.cG + ((size_t)(g0 + gg) * H + j0) * 64, (j1 - j0) * 128, sb);
          }
        }
        grid_bar(bar, bar_target);
        }
        // ---- phase 1: gated slices
        MARK(g0 == 0 ? 3 : 9);
        ZTS(5);
        if (SKG2) { if (!SKG1) { mbar_wait(smaddr(&ms->sbar), ms->sphase & 1); __syncthreads(); if (tid == 0) ms->sphase++; } grid_bar(bar, bar_target); return; }
        const int l = i - 6;
        const int j0 = bid * H / NB, j1 = (bid + 1) * H / NB;
        bf16* Fs = reinterpret_cast<bf16*>(red);
        bf16* Gs = Fs + 3 * 16 * 64;
        const int nsl = B * ng * (j1 - j0);
        float mreg[2], xreg[2] = {0.f, 0.f};
#pragma unroll
        for (int q = 0; q < 2; q++) {
          const int it = tid + q * NT;
          if (it < nsl) {
            const int b = it / (ng * (j1 - j0)), gg = (it / (j1 - j0)) % ng, k = j0 + it % (j1 - j0);
            mreg[q] = ldcgf(p.msg + ((size_t)(l * 4 + g0 + gg) * MAXB + b) * H + k);
            if (GZ0P1 && B > 1) xreg[q] = ldcgf(p.xm + ((size_t)b * 4 + g0 + (sidx_by_gate ? gg : 0)) * H + k);
          }
        }
        if (B > 2) {   // all loads first (6 per thread at B = 8), then the stores
          float zv[6];
#pragma unroll
          for (int q = 0; q < 6; q++) {
            const int it = tid + q * NT;
            zv[q] = 0.f;
            if (it < B * ng * 64) { const int b = it / (ng * 64), gg = (it / 64) % ng, j = it % 64; zv[q] = ldcgf(p.zb + (b * 4 + g0 + gg) * 64 + j); }
          }
#pragma unroll
          for (int q = 0; q < 6; q++) {
            const int it = tid + q * NT;
            if (it < B * ng * 64) { const int b = it / (ng * 64), gg = (it / 64) % ng, j = it % 64; XF[(b * 3 + gg) * 64 + j] = zv[q]; }
          }
        } else
        for (int it = tid; it < B * ng * 64; it += NT) {
          const int b = it / (ng * 64), gg = (it / 64) % ng, j = it % 64;
          XF[(b * 3 + gg) * 64 + j] = ldcgf(p.zb + (b * 4 + g0 + gg) * 64 + j);
        }
        if (!SKG1) mbar_wait(smaddr(&ms->sbar), ms->sphase & 1);
        __syncthreads();
        if (tid == 0 && !SKG1) ms->sphase++;
#pragma unroll
        for (int q = 0; q < 2; q++) {
          const int it = tid + q * NT;
          if (it >= nsl) break;
          const int b = it / (ng * (j1 - j0)), gg = (it / (j1 - j0)) % ng, k = j0 + it % (j1 - j0), gl = g0 + gg;
          const bf16* fr = Fs + (gg * 16 + (k - j0)) * 64;
          const bf16* gr = Gs + (gg * 16 + (k - j0)) * 64;
          const float m = mreg[q];
          float fa = 0.f, ga = 0.f;
          const int rot = GROT ? ((k - j0) & 7) : 0;
#pragma unroll
          for (int c_ = 0; c_ < 8; c_++) {
            const int j = ((c_ + rot) & 7) * 8;
            float f8[8], g8[8];
            unpack8(*reinterpret_cast<const uint4*>(fr + j), f8);
            unpack8(*reinterpret_cast<const uint4*>(gr + j), g8);
#pragma unroll
            for (int q = 0; q < 8; q++) { const float zz = XF[(b * 3 + gg) * 64 + j + q]; fa = fmaf(f8[q], zz, fa); ga = fmaf(g8[q], zz, ga); }
          }
          const float x = (GZ0P1 && B > 1) ? xreg[q] : ms->sl[b][gg][k - j0];
          p.xg[((size_t)b * 4 + gl) * H + k] = (1.0f - tanhf(fa)) * x + tanhf(ga) * m;
        }
        ZTS(6);
#ifdef ATSG
        if (tid == 0 && ap == ATSG && g0 == ATSGG) { long long* o = reinterpret_cast<long long*>(p.prof) + bid * 12; for (int k = 0; k < 12; k++) o[k] = ms->ats[k] - ms->ats[0]; }
#endif
        grid_bar(bar, bar_target);
      };

      // ================= phase A: attention inputs + qkv GEMV
      if (loopg) {
        if (!SKG) gate_phases(0, 3, asrc, nas, aqs, naq, L.glob != 0);
        else { grid_bar(bar, bar_target); grid_bar(bar, bar_target); }
      }
      if (!SKA) {
        MARK(4);
        if (APF && warp == 6 && lane == 0 && bid >= 3 && bid < 6) {
          if (bid == 3) { l2pf(L.ksc, 8192); l2pf(L.vsc, 8192); }
          if (bid == 4) { l2pf(L.qscale, 6144); l2pf(L.rscale, 768); }
          if (bid == 5) l2pf(L.relpT, RELEXT * DREL * 2);
        }
        int r0, r1; qkv_range(L, r0, r1);
        const int s0 = stream_of(L, r0), s1 = stream_of(L, r1 - 1);
#ifdef PROBE
        if (tid == 0) { FMARK(36); u32 v = ldcg32(p.vecs + (size_t)(bid * 64) % (H * 9)); if (v == 0x12345678u) p.bar[40] = v; FMARK(37);
          v = ldcg32(p.vecs + (size_t)(bid * 64 + 8) % (H * 9)); if (v == 0x12345678u) p.bar[40] = v; FMARK(38); }
#endif
        u16** rows3 = ms->arows3; const bf16** nw3 = ms->anw3; int* qsel3 = ms->aqsel3;
        const int nsA = s1 - s0 + 1;
        XTS(0);
        if (SKA1) {
        } else if (loopg && B > 1) {
          const float** src3 = ms->asrc3;
          for (int s = s0; s <= s1; s++) { rows3[s - s0] = xs + (s - s0) * B * XSH; nw3[s - s0] = nws[s]; src3[s - s0] = p.xg + (size_t)s * H; }
          norm_f32_rows_w(src3, 4 * H, B, nsA, nw3, rows3, XSH);
        } else if (i != 0 && !loopg && B > 1) {
          ++mev;
          if (MIXB(bid) >= 0 && MIXB(bid) < B) {
            u16** gr = ms->agr; const bf16** gw = ms->agw;
            for (int s = 0; s < naq; s++) { gr[s] = reinterpret_cast<u16*>(p.xn + ((size_t)MIXB(bid) * 3 + s) * H); gw[s] = nws[s]; }
            mixnorm(p, MIXB(bid), asrc, nas, aqs, naq, naq, ms->aq012, gw, gr, nullptr, ms);
            XTS(1);
            mix_publish();
            XTS(2);
          }
          mix_wait_rows(mev, (s1 - s0 + 1) * B, p.xn + (size_t)(L.glob ? s0 : 0) * H, 3 * H, L.glob ? H : 0);
          XTS(3);
          mix_wait_rows_done();
          XTS(4);
        } else for (int b = 0; b < B; b++) {
          if (b > 0) __syncthreads();   // rows3 / nw3 / qsel3 are shared: the previous sequence's stores are done
          for (int s = s0; s <= s1; s++) { rows3[s - s0] = xs + ((s - s0) * B + b) * XSH; nw3[s - s0] = nws[s]; qsel3[s - s0] = L.glob ? s : 0; }
          if (i == 0) {
            // h0 = bf16(embed_norm * rmsnorm(embed[tok])) (written by block 0), then the layer-0 norm
            const bf16* er = p.embed + (size_t)ms->tok[b] * H;
            const int nck = NCHK(tid);
            float e[2][8];
            for (int k = 0; k < nck; k++) unpack8(__ldg(reinterpret_cast<const uint4*>(er + (tid + k * NT) * 8)), e[k]);
            float ss = 0.f;
            for (int k = 0; k < nck; k++)
#pragma unroll
              for (int j = 0; j < 8; j++) ss = fmaf(e[k][j], e[k][j], ss);
            ss = block_sum(ss, ms->scr);
            const float r = rsqrtf(ss * (1.0f / H) + 1e-6f);
            for (int k = 0; k < nck; k++) {
              const int e0 = (tid + k * NT) * 8;
              float w8[8]; unpack8(__ldg(reinterpret_cast<const uint4*>(p.embed_norm + e0)), w8);
#pragma unroll
              for (int j = 0; j < 8; j++) {
                e[k][j] = bfr(w8[j] * (e[k][j] * r));
                if (bid == 0) p.vecs[((size_t)b * NVEC + V_H0) * H + e0 + j] = __float2bfloat16_rn(e[k][j]);
              }
            }
            __syncthreads();
            float ss2 = 0.f;
            for (int k = 0; k < nck; k++)
#pragma unroll
              for (int j = 0; j < 8; j++) ss2 = fmaf(e[k][j], e[k][j], ss2);
            ss2 = block_sum(ss2, ms->scr);
            const float r2 = rsqrtf(ss2 * (1.0f / H) + 1e-6f);
            for (int s = 0; s < nsA; s++)
              for (int k = 0; k < nck; k++) {
                const int e0 = (tid + k * NT) * 8;
                float v[8];
#pragma unroll
                for (int j = 0; j < 8; j++) v[j] = e[k][j] * r2;
                store_row8(rows3[s], e0, v, __ldg(reinterpret_cast<const uint4*>(nw3[s] + e0)));
              }
          } else if (loopg) {
            const float** src3 = ms->asrc3;
            for (int s = s0; s <= s1; s++) src3[s - s0] = p.xg + ((size_t)b * 4 + s) * H;
            norm_f32_rows(src3, nsA, nw3, rows3, ms);
          } else {
#ifdef TWICE
            mixnorm(p, b, asrc, nas, aqs, naq, nsA, qsel3, nw3, rows3, nullptr, ms);
            FMARK(35);
#endif
            mixnorm(p, b, asrc, nas, aqs, naq, nsA, qsel3, nw3, rows3, nullptr, ms);
          }
        }
        XTS(5);
        if (!SKA2 && !PFE_QKV) pf_qkv(i);
        __syncthreads();
        MARK(5);
        if (!SKA2) consume_tasks<0>(ttab, XSH, rg, pst[warp], [&](int sl, const float* c) { store_task(red, sl, c); });
        XTS(7);
        __syncthreads();
        const int R = r1 - r0;
        for (int it = tid; it < R * 16 * B; it += NT) {
          const int r = it / (16 * B), ii = (it / B) % 16, b = it % B;
          float v = 0.f;
#pragma unroll
          for (int s = 0; s < 8; s++) v += red_at(red, r * 8 + s, ii, b);
          p.qkvo[(size_t)b * QKVW + (r0 + r) * 16 + ii] = __float2bfloat16_rn(v);
        }
      }
      // attention items of this block: itm = bid, bid + NB, ...: (sequence, kv head, 128-key chunk). The K/V cache is
      // stored in mma fragment order, 16-key tiles of 4 KB: K tile = A operand [8 ksteps (16 dims)][32 lanes][4 regs]
      // (rows = keys), V tile = A operand of V^T [8 dim tiles][32 lanes][4 regs] (rows = dims, columns = keys).
      // The first item's cached tiles and bias-table rows are fetched by bulk copies before the barrier.
      const int kstart = L.glob ? 0 : max(0, pos - (WIN - 1));
      // items = (sequence, kv head, piece): the window's keys of each (sequence, kv head) split into npc pieces, npc =
      // min(128-key chunks, 152 / (8 B)): one round of the grid; a piece = sub-chunks of up to CH2 keys (K tiles in xs,
      // V tiles in the idle weight rings, separate barriers: the next sub-chunk's K loads during this one's softmax/PV)
      // sub-chunk size: 128 keys in the B = 1 kernel (shorter code, no long pieces there: B=1 -28 us), CH2 otherwise
      constexpr int CHB = (BC == 1 && CH2B1 == 128) ? 128 : CH2, NKTB = CHB / 16, KPWB = NKTB / 8;
      const int nc128 = pos / 128 - kstart / 128 + 1;
      // pieces per (sequence, kv head), whole 128-key chunks each: one grid round, at most PMAXC chunks per piece
      // where possible (a piece's sub-chunks cost less than a merge of partials)
      const int nch = min(min(nc128, max(1, NB / (B * NKV))), PMAXC > 0 ? (nc128 + PMAXC - 1) / PMAXC : 64);
      const int nitems = B * NKV * nch;
      unsigned char* abase = smem + SM_XS;
      bf16* KS = reinterpret_cast<bf16*>(abase);                // NKT tiles
      // V tiles: next to K in xs when every piece is one 128-key chunk (B = 1, 2): the rings stay free for o_proj's
      // stream, issued at the end of the attention phase; else in the idle rings
      const bool vin_xs = (nch == nc128);
      bf16* VS = reinterpret_cast<bf16*>(vin_xs ? abase + 32768 : smem + SM_RING);
      bf16* RPs = reinterpret_cast<bf16*>(abase + 65536);   // bias rows of the sub-chunk's distances [CH2][16] (past K and, for CH2 = 128, V)
      AttS2* at = reinterpret_cast<AttS2*>(red);
      auto piece = [&](int itm, int& b, int& kvh, int& plo, int& phi) {
        const int pr = itm / nch, q = itm % nch, cb = kstart / 128;
        b = pr / NKV; kvh = pr % NKV;
        const int qa = PCEIL ? (q * nc128 + nch - 1) / nch : q * nc128 / nch, qb = PCEIL ? ((q + 1) * nc128 + nch - 1) / nch : (q + 1) * nc128 / nch;
        plo = max(kstart, (cb + qa) * 128); phi = min(pos, (cb + qb) * 128 - 1);
      };
      auto nsub_of = [&](int plo, int phi) { return ((phi >> 4) - (plo >> 4)) / NKTB + 1; };
      auto sub_range = [&](int plo, int phi, int sc, int& kbase, int& klo, int& khi) {
        kbase = ((plo >> 4) + sc * NKTB) * 16; klo = max(plo, kbase); khi = min(phi, kbase + CHB - 1);
      };
      // one thread: mode 0 = K tiles + bias rows (sbar), 1 = V tiles (sbar2), 2 = both on sbar (one-sub-chunk pieces)
      auto fetch_kv = [&](int itm, int sc, int mode) {
        int b, kvh, plo, phi, kbase, klo, khi;
        piece(itm, b, kvh, plo, phi); sub_range(plo, phi, sc, kbase, klo, khi);
        const int tlo = klo >> 4, thi = khi >> 4, ntile = thi - tlo + 1, ct = tlo - kbase / 16;
        const size_t base = (((size_t)slot * MAXB + b) * NKV + kvh) * p.maxctx * HD;
        asm volatile("fence.proxy.async.global;" ::: "memory");
        if (mode != 1) {
          const int d0 = pos - khi, d1 = min(pos - klo, RELEXT - 1), nd = d1 >= d0 ? d1 - d0 + 1 : 0;
          const u32 sb = smaddr(&ms->sbar);
          mbar_expect(sb, ntile * 4096 * (mode == 2 ? 2 : 1) + nd * DREL * 2);
          tma_load_plain(smaddr(KS + ct * 2048), p.kc + base + (size_t)tlo * 2048, ntile * 4096, sb);
          if (mode == 2) tma_load_plain(smaddr(VS + ct * 2048), p.vc + base + (size_t)tlo * 2048, ntile * 4096, sb);
          if (nd) tma_load_plain(smaddr(RPs), L.relpT + (size_t)d0 * DREL, nd * DREL * 2, sb);
        } else {
          const u32 sb = smaddr(&ms->sbar2);
          mbar_expect(sb, ntile * 4096);
          tma_load_plain(smaddr(VS + ct * 2048), p.vc + base + (size_t)tlo * 2048, ntile * 4096, sb);
        }
      };
      auto fetch_first = [&](int itm) {
        int b, kvh, plo, phi; piece(itm, b, kvh, plo, phi);
        if (nsub_of(plo, phi) == 1) fetch_kv(itm, 0, 2); else { fetch_kv(itm, 0, 0); fetch_kv(itm, 0, 1); }
      };
      if (!SKB && tid == 0 && bid < nitems) fetch_first(bid);
      XTS(6);
#ifdef ATSA
      if (tid == 0 && ap == ATSA) { long long* o = reinterpret_cast<long long*>(p.prof) + bid * 12; for (int k = 0; k < 8; k++) o[k] = ms->ats[k] - ms->ats[0]; o[8] = ms->ats[8]; o[9] = ms->ats[9]; o[11] = ms->ats[11]; }
#endif
      grid_bar(bar, bar_target);

      // ================= phase B: attention
      auto l2_next = [&]() {
#ifdef NO_L2NEXT
        return;
#endif
        if (warp == 7 && lane == 0 && !NO_L2BIG) {
          if (bid < 144) for (int k = 0; k < 4; k++) l2pf(L.wo + ((size_t)bid * KS_A + k * 48) * 32, 48 * 512);
          if (bid < 8) for (int k = 0; k < 4; k++) l2pf(L.wr + ((size_t)bid * KS_H + k * 36) * 32, 36 * 512);
          if (ap + 1 < NAPP) {
            const LayerP& Ln = p.L[app_layer(ap + 1)];
            int r0, r1; qkv_range(Ln, r0, r1);
            for (int r = r0; r < r1; r++) for (int k = 0; k < 4; k++) l2pf(Ln.wqkv + ((size_t)r * KS_H + k * 36) * 32, 36 * 512);
          }
        }
        // predicted experts of this application (the same layer's previous pass for loop passes 2-3, else this
        // application in the previous token): their w13 rows of rowtile bid into L2 (the ring then fills from L2)
#ifdef EXPF
        if (warp == 5 && bid < 144 && lane < TOPK * B) {
          const int pa = (L.loop && pas > 0) ? ap - 12 : ap;
          const int e = p.rdbg[(pa * MAXB + lane / TOPK) * TOPK + lane % TOPK];
          if (e >= 0 && e < NE) {
            if (L.mx) { const size_t it = ((size_t)e * 144 + bid) * 72; l2pf(L.w13c + it * 32, 72 * 512); l2pf(L.w13s + it * 16, 72 * 16); }
            else { const size_t it = ((size_t)e * 144 + bid) * 36; l2pf(L.w13c + it * 32, 36 * 512); l2pf(L.w13s + it * 64, 36 * 64); }
          }
        }
#endif
        // small parameters: this layer's routing and expert factors, the next application's everything
        if (warp == 6 && lane == 0) {
          const LayerP& Ln = p.L[app_layer(ap + 1 < NAPP ? ap + 1 : ap)];
          const int nx = app_pass(ap + 1 < NAPP ? ap + 1 : ap) > 0 && Ln.loop;
          switch (bid) {
            case 0: l2pf(L.rbias, 512); if (!L.mx) { l2pf(L.f13, 512); l2pf(L.f2, 512); } break;
            case 1: l2pf(Ln.nq, H * 2); l2pf(Ln.nk, H * 2); l2pf(Ln.nv, H * 2); l2pf(Ln.nm, H * 2); break;
            case 2: l2pf(Ln.aq, (Ln.glob ? 3 : 1) * H * 4); l2pf(Ln.mq, H * 4); break;
            case 3: l2pf(Ln.ksc, 8192); l2pf(Ln.vsc, 8192); l2pf(Ln.qscale, 6144); l2pf(Ln.rscale, 768); break;
            case 4: l2pf(Ln.asc, H * 8); l2pf(Ln.msc, H * 8); break;
            case 5: l2pf(Ln.relpT, 32768); break;
            case 6: if (ap + 1 < NAPP) { const LayerP& L2 = p.L[app_layer(ap + 1)]; l2pf(L2.rbias, 512); if (!L2.mx) { l2pf(L2.f13, 512); l2pf(L2.f2, 512); } } break;
            default:
              if (nx && bid >= 8 && bid < 8 + 72) {
                // controllers of the next loop layer: D [4][64][2H], F, G [4][H][64]: 72 pieces of 64 KB each
                const int k = bid - 8;
                if (k < 36) l2pf(Ln.cD + (size_t)k * 32768, 65536);
                else if (k < 54) l2pf(Ln.cF + (size_t)(k - 36) * 32768, 65536);
                else l2pf(Ln.cG + (size_t)(k - 54) * 32768, 65536);
              }
          }
        }
      };
      if (!SKB) {
        MARK(6);
        if (bid >= nitems) l2_next();
        const int OFF_V = L.glob ? 476 * 16 : 474 * 16;
        const float tau = 1.0f + 0.1f * logf(fmaxf((float)(pos + 1) / 8192.0f, 1.0f));
        const int g = lane >> 2, t4 = lane & 3;
        int nmine = 0;
        for (int itm = bid; itm < nitems; itm += NB, nmine++) {
          int b, kvh, plo, phi;
          piece(itm, b, kvh, plo, phi);
          const bf16* qk = p.qkvo + (size_t)b * QKVW;
          const bool hascur = (phi == pos);
          const int nsub = nsub_of(plo, phi);
          const int kblast = ((plo >> 4) + (nsub - 1) * NKTB) * 16;   // the last sub-chunk's base (holds pos if hascur)
          ATS(0);
          if (nmine > 0 && tid == 0) fetch_first(itm);
          if (nmine == 0) l2_next();
          // ---- loads of this step's projections
          float qv[4] = {0.f, 0.f, 0.f, 0.f}, rv = 0.f;
          if (tid < 64) {
            const int j = tid >> 4, e = tid & 15;
            if (j < 3) rv = u16f(ldcg16(qk + OFF_R + (kvh * 3 + j) * DREL + e)) * ldgbf(L.rscale + (kvh * 3 + j) * DREL + e);
          } else if (tid < 160) {
            const int j = (tid - 64) >> 5, h = kvh * 3 + j;
            const uint2 u = ldcg64(qk + h * HD + lane * 4);
            qv[0] = __uint_as_float(u.x << 16); qv[1] = __uint_as_float(u.x & 0xffff0000u);
            qv[2] = __uint_as_float(u.y << 16); qv[3] = __uint_as_float(u.y & 0xffff0000u);
#pragma unroll
            for (int q = 0; q < 4; q++) qv[q] *= ldgbf(L.qscale + h * HD + lane * 4 + q);
          } else if (tid < 163) {
            at->sink[tid - 160] = u16f(ldcg16(qk + OFF_S + kvh * 3 + tid - 160));
          }
          const int isv = tid >> 7, dch = tid & 127, ch = kvh * HD + dch;
          float xt = 0.f, xh0 = 0.f, xh1 = 0.f, xh2 = 0.f, w0 = 0.f, w1 = 0.f, w2 = 0.f, w3 = 0.f;
          bf16* rbp = (isv ? p.rv : p.rk) + ((size_t)slot * MAXB + b) * 4 * KVW + ch;
          if (hascur) {
            xt = u16f(ldcg16(qk + (isv ? OFF_V : OFF_K) + ch));
            xh0 = u16f(ldcg16(rbp + ((pos - 1) & 3) * KVW));
            xh1 = u16f(ldcg16(rbp + ((pos - 2) & 3) * KVW));
            xh2 = u16f(ldcg16(rbp + ((pos - 3) & 3) * KVW));
            if (!L.glob) {
              const bf16* w = (isv ? L.vsc : L.ksc) + ch * 4;   // taps: 3 = current
              w0 = ldgbf(w); w1 = ldgbf(w + 1); w2 = ldgbf(w + 2); w3 = ldgbf(w + 3);
            } else {
              const bf16* sw = isv ? L.vsc : L.ksc;              // static_w [4 lags][1024]; dynamic taps follow v / k rows
              const int doff = isv ? (OFF_V + 64 * 16) : OFF_DK;
              w0 = ldgbf(sw + ch) + u16f(ldcg16(qk + doff + kvh * 4 + 0));
              w1 = ldgbf(sw + KVW + ch) + u16f(ldcg16(qk + doff + kvh * 4 + 1));
              w2 = ldgbf(sw + 2 * KVW + ch) + u16f(ldcg16(qk + doff + kvh * 4 + 2));
              w3 = ldgbf(sw + 3 * KVW + ch) + u16f(ldcg16(qk + doff + kvh * 4 + 3));
            }
          }
          ATS(1);
          // ---- q clip (warps 2..4) -> bf16 smem; r clip (warps 0, 1)
          if (tid < 64) {
            float ss = rv * rv;
#pragma unroll
            for (int o = 8; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o, 16);
            const int j = tid >> 4;
            if (j < 3) {
              float o = bfr(rv * fminf(8.0f / (sqrtf(ss) + 1e-6f), 1.0f));
              if (L.glob) o = bfr(o * tau);
              at->rb[j][tid & 15] = f2u16(o);
            }
          } else if (tid < 160) {
            float ss = qv[0] * qv[0];
            ss = fmaf(qv[1], qv[1], ss); ss = fmaf(qv[2], qv[2], ss); ss = fmaf(qv[3], qv[3], ss);
            ss = warp_sum(ss);
            const float f = fminf(22.627416997969522f / (sqrtf(ss) + 1e-6f), 1.0f);
            const int j = (tid - 64) >> 5;
            u16 o16[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
              float o = bfr(qv[q] * f);
              if (L.glob) o = bfr(o * tau);
              o16[q] = f2u16(o);
            }
            *reinterpret_cast<uint2*>(&at->qb[j][lane * 4]) = make_uint2((u32)o16[0] | ((u32)o16[1] << 16), (u32)o16[2] | ((u32)o16[3] << 16));
          }
          __syncthreads();
          // ---- the new position's K / V: short conv, then L2 clip (K) / RMS norm (V); into the cache tile layout
          float kvfin = 0.f;
          if (hascur) {
            float o;
            if (!L.glob) {
              float acc = w0 * xh2;
              acc = fmaf(w1, xh1, acc); acc = fmaf(w2, xh0, acc); acc = fmaf(w3, xt, acc);
              o = bfr((acc + xt) / 2.23606797749979f);
            } else {
              float acc = xt * w0;
              acc = fmaf(xh0, w1, acc); acc = fmaf(xh1, w2, acc); acc = fmaf(xh2, w3, acc);
              o = bfr((acc + xt) / 3.0f);
            }
            const float ss = warp_sum(o * o);
            if (lane == 0) at->red[warp] = ss;
            __syncthreads();
            const float tot = (at->red[isv * 4 + 0] + at->red[isv * 4 + 1]) + (at->red[isv * 4 + 2] + at->red[isv * 4 + 3]);
            kvfin = isv ? bfr(o * rsqrtf(tot * (1.0f / HD) + 1e-6f)) : bfr(o * fminf(22.627416997969522f / (sqrtf(tot) + 1e-6f), 1.0f));
            const int T = pos >> 4, r = pos & 15;
            int off;
            if (!isv) {   // K tile: row = key r, column = dim
              const int gg = r & 7, hr = r >> 3, j = dch >> 4, dd = dch & 15, hc = dd >> 3, tq = (dd & 7) >> 1, pr = dd & 1;
              off = ((j * 32 + gg * 4 + tq) * 4 + hc * 2 + hr) * 2 + pr;
            } else {      // V^T tile: row = dim, column = key r
              const int hc = r >> 3, tq = (r & 7) >> 1, pr = r & 1, dt = dch >> 4, dd = dch & 15, gg = dd & 7, hr = dd >> 3;
              off = ((dt * 32 + gg * 4 + tq) * 4 + hc * 2 + hr) * 2 + pr;
            }
            bf16* cp = (isv ? p.vc : p.kc) + (((size_t)slot * MAXB + b) * NKV + kvh) * p.maxctx * HD + (size_t)T * 2048 + off;
            *cp = __float2bfloat16_rn(kvfin);
            at->kvoff[isv][dch] = (T - kblast / 16) * 2048 + off;
            rbp[(pos & 3) * KVW] = __float2bfloat16_rn(xt);
          }
          ATS(2);
          if (tid < 3) { at->ml[tid][0] = -INFINITY; at->ml[tid][1] = 0.f; }
          float cv[4] = {0.f, 0.f, 0.f, 0.f};   // P V of warp w = dim tile w, accumulated over the sub-chunks
          for (int sc = 0; sc < nsub; sc++) {
            int kbase, klo, khi;
            sub_range(plo, phi, sc, kbase, klo, khi);
            const int d0 = pos - khi;
            const bool cursub = hascur && sc == nsub - 1;
            mbar_wait(smaddr(&ms->sbar), ms->sphase & 1);
            __syncthreads();
            if (tid == 0) ms->sphase++;
            if (cursub && (!isv || nsub == 1)) (isv ? VS : KS)[at->kvoff[isv][dch]] = __float2bfloat16_rn(kvfin);   // the current key into the tiles
            __syncthreads();
            ATS(3);
            // ---- scores on the tensor cores: warp w = key tile w of the chunk; C[key][head] = K . q
  #pragma unroll
            for (int s_ = 0; s_ < KPWB; s_++) {
              const int kt = warp + 8 * s_, key0 = kbase + kt * 16;
              if (key0 + 15 >= klo && key0 <= khi) {
                float cs[4] = {0.f, 0.f, 0.f, 0.f};
                const u32* qw = reinterpret_cast<const u32*>(at->qb[g < 3 ? g : 0]);
  #pragma unroll
                for (int j = 0; j < 8; j++) {
                  const uint4 a = *reinterpret_cast<const uint4*>(KS + kt * 2048 + (j * 32 + lane) * 8);
                  const u32 b0 = g < 3 ? qw[j * 8 + t4] : 0u, b1 = g < 3 ? qw[j * 8 + 4 + t4] : 0u;
                  mma16816(cs, a.x, a.y, a.z, a.w, b0, b1);
                }
                float cb[4] = {0.f, 0.f, 0.f, 0.f};
                {   // relative bias: A = bias-table rows of the tile's keys (smem, row = distance - d0), B = r
                  const int ka = key0 + g, kb = ka + 8, da = pos - ka, db = pos - kb;
                  const u32* rp = reinterpret_cast<const u32*>(RPs);
                  u32 a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
                  if (ka >= klo && ka <= khi && da < RELEXT) { a0 = rp[(da - d0) * 8 + t4]; a2 = rp[(da - d0) * 8 + 4 + t4]; }
                  if (kb >= klo && kb <= khi && db < RELEXT) { a1 = rp[(db - d0) * 8 + t4]; a3 = rp[(db - d0) * 8 + 4 + t4]; }
                  const u32* rw = reinterpret_cast<const u32*>(at->rb[g]);
                  mma16816(cb, a0, a1, a2, a3, rw[t4], rw[4 + t4]);
                }
                // lane: keys key0 + g, key0 + g + 8; heads 2 t4, 2 t4 + 1
  #pragma unroll
                for (int e = 0; e < 4; e++) {
                  const int key = key0 + g + (e >> 1) * 8, h = 2 * t4 + (e & 1);
                  if (h < 3) {
                    float sc = -INFINITY;
                    if (key >= klo && key <= khi) sc = fmaf(cs[e], 0.08838834764831845f, cb[e]);
                    at->sc[h][key - kbase] = sc;
                  }
                }
              } else if (lane < 16) {
  #pragma unroll
                for (int h = 0; h < 3; h++) at->sc[h][kt * 16 + lane] = -INFINITY;
              }
            }

            __syncthreads();
            if (tid == 0 && sc + 1 < nsub) fetch_kv(itm, sc + 1, 0);   // K buffer free
            ATS(4);
            // ---- online softmax per head (warps 0..2): running max / sum over the piece's sub-chunks
            if (warp < 3) {
              float v[CHB / 32], mc = -INFINITY;
#pragma unroll
              for (int q = 0; q < CHB / 32; q++) { v[q] = at->sc[warp][lane + 32 * q]; mc = fmaxf(mc, v[q]); }
              mc = warp_max(mc);
              const float mo = at->ml[warp][0], mn = fmaxf(mo, mc);
              const float al = (mn == -INFINITY) ? 1.f : expf(mo - mn);
              float l = 0.f;
#pragma unroll
              for (int q = 0; q < CHB / 32; q++) {
                const float e = (v[q] == -INFINITY) ? 0.f : expf(v[q] - mn);
                const u16 hi = f2u16(e);
                at->ph[warp][lane + 32 * q] = hi; at->pl[warp][lane + 32 * q] = f2u16(e - u16f(hi));
                l += e;
              }
              l = warp_sum(l);
              __syncwarp();
              if (lane == 0) { at->alpha[warp] = al; at->ml[warp][0] = mn; at->ml[warp][1] = at->ml[warp][1] * al + l; }
            }
            if (nsub > 1) {
              mbar_wait(smaddr(&ms->sbar2), ms->sphase2 & 1);
              __syncthreads();
              if (tid == 0) ms->sphase2++;
              if (cursub && isv) VS[at->kvoff[1][dch]] = __float2bfloat16_rn(kvfin);
            }
            __syncthreads();
            ATS(5);
            // ---- P V on the tensor cores: warp w = dim tile w; C[dim][head] = V^T . P^T, P as bf16 hi + lo
            {
#pragma unroll
              for (int e = 0; e < 4; e++) { const int h = 2 * t4 + (e & 1); if (h < 3) cv[e] *= at->alpha[h]; }
              const int dt = warp;
#pragma unroll
              for (int kt = 0; kt < CHB / 16; kt++) {
                const int key0 = kbase + kt * 16;
                if (key0 + 15 < klo || key0 > khi) continue;
                const uint4 a = *reinterpret_cast<const uint4*>(VS + kt * 2048 + (dt * 32 + lane) * 8);
                u32 bh0 = 0u, bh1 = 0u, bl0 = 0u, bl1 = 0u;
                if (g < 3) {
                  const u32* hp = reinterpret_cast<const u32*>(at->ph[g] + kt * 16);
                  const u32* lp = reinterpret_cast<const u32*>(at->pl[g] + kt * 16);
                  bh0 = hp[t4]; bh1 = hp[4 + t4]; bl0 = lp[t4]; bl1 = lp[4 + t4];
                }
                mma16816(cv, a.x, a.y, a.z, a.w, bh0, bh1);
                mma16816(cv, a.x, a.y, a.z, a.w, bl0, bl1);
              }
            }
            __syncthreads();
            if (tid == 0 && sc + 1 < nsub) fetch_kv(itm, sc + 1, 1);   // V buffer free
          }
          {   // lane: dims 16 w + g, + 8; heads 2 t4, 2 t4 + 1
#pragma unroll
            for (int e = 0; e < 4; e++) {
              const int d = warp * 16 + g + (e >> 1) * 8, h = 2 * t4 + (e & 1);
              if (h < 3) at->acc[h][d] = cv[e];
            }
          }
          __syncthreads();
          ATS(6);
          if (nch == 1 && !(B <= 2 && MERGE_IN_C)) {
            for (int it = tid; it < 3 * HD; it += NT) {
              const int h = it / HD, d = it % HD, hh = kvh * 3 + h;
              const float denom = at->ml[h][1] + expf(at->sink[h] - at->ml[h][0]);
              const float gt = sigm(u16f(ldcg16(qk + OFF_G + hh * HD + d)));
              p.attn_out[(size_t)b * QW + xfo(hh * HD + d)] = __float2bfloat16_rn(at->acc[h][d] / denom * gt);
            }
            __syncthreads();
            continue;
          }
          float* pa = p.apart + (((size_t)(b * NKV + kvh) * MAXCH + itm % nch) * 3) * ACC_STRIDE;
          for (int it = tid; it < 3 * HD; it += NT) {
            const int h = it / HD, d = it % HD;
            pa[h * ACC_STRIDE + 8 + d] = at->acc[h][d];
          }
          if (tid < 3) { pa[tid * ACC_STRIDE + 0] = at->ml[tid][0]; pa[tid * ACC_STRIDE + 1] = at->ml[tid][1]; }
          __syncthreads();
          if (B <= 2 && MERGE_IN_C) continue;   // merged by every o_proj block
          if (tid == 0) {
#if ATT_ACQREL
            // one acq_rel increment: releases this block's partial (bar.sync before it), acquires the others' for the
            // last arriver
            u32 old;
            asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(p.acnt + b * NKV + kvh) : "memory");
            const int last = old == (u32)(nch - 1);
            if (last) p.acnt[b * NKV + kvh] = 0;
#else
            __threadfence();
            const u32 old = atomicAdd(p.acnt + b * NKV + kvh, 1u);
            const int last = old == (u32)(nch - 1);
            if (last) { __threadfence(); p.acnt[b * NKV + kvh] = 0; }
#endif
            ms->last = last;
          }
          __syncthreads();
          ATS(7);
          if (!ms->last) continue;
          // ---- the last chunk to finish merges all of them (in chunk order); gate and sink
          {
            float* mbuf = reinterpret_cast<float*>(abase);       // [16 chunks][3][HD] (the tiles are no longer needed)
            const float* pb = p.apart + ((size_t)(b * NKV + kvh) * MAXCH * 3) * ACC_STRIDE;
            float gt[2];
            for (int k = 0; k < 2; k++) {
              const int it = tid + k * NT;
              gt[k] = it < 3 * HD ? sigm(u16f(ldcg16(qk + OFF_G + (kvh * 3 + it / HD) * HD + it % HD))) : 0.f;
            }
            for (int it = tid; it < nch * 3; it += NT) { at->mlb[it][0] = ldcgf(pb + (size_t)it * ACC_STRIDE); at->mlb[it][1] = ldcgf(pb + (size_t)it * ACC_STRIDE + 1); }
            __syncthreads();
            if (warp < 3) {
              float m = -INFINITY;
              for (int cc = lane; cc < nch; cc += 32) m = fmaxf(m, at->mlb[cc * 3 + warp][0]);
              m = warp_max(m);
              float l = 0.f;
              for (int cc = lane; cc < nch; cc += 32) { const float w = expf(at->mlb[cc * 3 + warp][0] - m); at->mw[cc][warp] = w; l = fmaf(at->mlb[cc * 3 + warp][1], w, l); }
              l = warp_sum(l);
              if (lane == 0) { at->ml[warp][0] = m; at->ml[warp][1] = l; }
            }
            float A[2] = {0.f, 0.f};
            for (int cb = 0; cb < nch; cb += 16) {
              const int nb = min(16, nch - cb);
              __syncthreads();
              for (int it = tid; it < nb * 3 * (HD / 4); it += NT) {
                const int cc = it / (3 * (HD / 4)), h = (it / (HD / 4)) % 3, d4 = it % (HD / 4);
                float4 w; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];" : "=f"(w.x), "=f"(w.y), "=f"(w.z), "=f"(w.w) : "l"(pb + ((size_t)(cb + cc) * 3 + h) * ACC_STRIDE + 8 + d4 * 4));
                *reinterpret_cast<float4*>(mbuf + ((size_t)cc * 3 + h) * HD + d4 * 4) = w;
              }
              __syncthreads();
              for (int k = 0; k < 2; k++) {
                const int it = tid + k * NT;
                if (it < 3 * HD) {
                  const int h = it / HD, d = it % HD;
                  for (int cc = 0; cc < nb; cc++) A[k] = fmaf(at->mw[cb + cc][h], mbuf[((size_t)cc * 3 + h) * HD + d], A[k]);
                }
              }
            }
            for (int k = 0; k < 2; k++) {
              const int it = tid + k * NT;
              if (it < 3 * HD) {
                const int h = it / HD, d = it % HD, hh = kvh * 3 + h;
                const float denom = at->ml[h][1] + expf(at->sink[h] - at->ml[h][0]);
                p.attn_out[(size_t)b * QW + xfo(hh * HD + d)] = __float2bfloat16_rn(A[k] / denom * gt[k]);
              }
            }
            __syncthreads();
          }
        }
          ATS(9);
#ifdef ATTTS
          if (tid == 0 && ap == 20) { long long* o = reinterpret_cast<long long*>(p.prof) + bid * 12; for (int k = 0; k < 10; k++) o[k] = ms->ats[k] - ms->ats[0]; o[10] = ms->ats[0]; }
#endif
        if (PFE_O || vin_xs || PFO_ALL) pf_o(i);   // the rings are free once the last item is done
      }
      grid_bar(bar, bar_target);

      // ================= phase C: o_proj + attention short conv
      if (!SKC) {
        MARK(7);
        if (bid < 144) {
          if (B <= 2 && MERGE_IN_C) {
            // B <= 2: every o_proj block merges the attention partials of all heads itself (no atomics, no last-arriver
            // merge on the attention phase's critical path), into its xs rows
            if (!PFE_O) pf_o(i);
            float* mlc = red;                          // [B][24][nch][2]
            float* wts = red + B * 24 * nch * 2;       // [B][24][nch]
            float* dn = wts + B * 24 * nch;            // [B][24]
            for (int it = tid; it < B * 24 * nch; it += NT) {
              const int b = it / (24 * nch), hh = (it / nch) % 24, cc = it % nch;
              const float* q = p.apart + (((size_t)(b * NKV + hh / 3) * MAXCH + cc) * 3 + hh % 3) * ACC_STRIDE;
              mlc[2 * it] = ldcgf(q); mlc[2 * it + 1] = ldcgf(q + 1);
            }
            __syncthreads();
            if (tid < B * 24) {
              const int b = tid / 24, hh = tid % 24;
              const float* mc = mlc + (size_t)tid * nch * 2;
              float M = -INFINITY;
              for (int cc = 0; cc < nch; cc++) M = fmaxf(M, mc[2 * cc]);
              float Ls = 0.f;
              for (int cc = 0; cc < nch; cc++) { const float w = expf(mc[2 * cc] - M); wts[tid * nch + cc] = w; Ls = fmaf(mc[2 * cc + 1], w, Ls); }
              const float snk = u16f(ldcg16(p.qkvo + (size_t)b * QKVW + OFF_S + hh));
              dn[tid] = 1.0f / (Ls + expf(snk - M));
            }
            __syncthreads();
            for (int it = tid; it < B * QW; it += NT) {
              const int b = it / QW, k = it % QW, hh = k / HD, d = k % HD;
              const float* q = p.apart + (((size_t)(b * NKV + hh / 3) * MAXCH) * 3 + hh % 3) * ACC_STRIDE + 8 + d;
              const float* w = wts + (b * 24 + hh) * nch;
              float A = 0.f;
              int cc = 0;
              for (; cc + 3 < nch; cc += 4) {
                const float a0 = ldcgf(q + (size_t)(cc + 0) * 3 * ACC_STRIDE), a1 = ldcgf(q + (size_t)(cc + 1) * 3 * ACC_STRIDE);
                const float a2 = ldcgf(q + (size_t)(cc + 2) * 3 * ACC_STRIDE), a3 = ldcgf(q + (size_t)(cc + 3) * 3 * ACC_STRIDE);
                A = fmaf(w[cc], a0, A); A = fmaf(w[cc + 1], a1, A); A = fmaf(w[cc + 2], a2, A); A = fmaf(w[cc + 3], a3, A);
              }
              for (; cc < nch; cc++) A = fmaf(w[cc], ldcgf(q + (size_t)cc * 3 * ACC_STRIDE), A);
              const float gt = sigm(u16f(ldcg16(p.qkvo + (size_t)b * QKVW + OFF_G + k)));
              xs[b * XSA + xfo(k)] = f2u16(A * dn[b * 24 + hh] * gt);
            }
            __syncthreads();
          } else {
          if (tid == 0) {
            asm volatile("fence.proxy.async.global;" ::: "memory");
            const u32 sb = smaddr(&ms->sbar);
            mbar_expect(sb, B * QW * 2);
            for (int b = 0; b < B; b++) tma_load_plain(smaddr(xs + b * XSA), p.attn_out + (size_t)b * QW, QW * 2, sb);
          }
          if (!(PFE_O || vin_xs || PFO_ALL)) pf_o(i);
          mbar_wait(smaddr(&ms->sbar), ms->sphase & 1);
          __syncthreads();
          if (tid == 0) ms->sphase++;
          }
          // the short conv's taps and history do not depend on the GEMV: loaded before it (thread (ii, b) = tid < 16 B)
          float cw0 = 0.f, cw1 = 0.f, cw2 = 0.f, cw3 = 0.f, ch0 = 0.f, ch1 = 0.f, ch2 = 0.f;
          if (tid < 16 * B) {
            const int ii = tid / B, b = tid % B, row = bid * 16 + ii;
            const bf16* w = L.asc + row * 4;
            const bf16* rb = p.ro + ((size_t)slot * MAXB + b) * 4 * H;
            cw0 = ldgbf(w + 0); cw1 = ldgbf(w + 1); cw2 = ldgbf(w + 2); cw3 = ldgbf(w + 3);
            ch0 = u16f(ldcg16(rb + ((pos - 3) & 3) * H + row)); ch1 = u16f(ldcg16(rb + ((pos - 2) & 3) * H + row)); ch2 = u16f(ldcg16(rb + ((pos - 1) & 3) * H + row));
          }
          consume_tasks<0>(ttab, XSA, rg, pst[warp], [&](int sl, const float* c) { store_task(red, sl, c); });
          __syncthreads();
          if (tid < 16 * B) {
            const int ii = tid / B, b = tid % B, row = bid * 16 + ii;
            float v = 0.f;
#pragma unroll
            for (int s = 0; s < 8; s++) v += red_at(red, s, ii, b);
            const float o = bfr(v);
            bf16* rb = p.ro + ((size_t)slot * MAXB + b) * 4 * H;
            float acc = cw0 * ch0;
            acc = fmaf(cw1, ch1, acc);
            acc = fmaf(cw2, ch2, acc);
            acc = fmaf(cw3, o, acc);
            rb[(pos & 3) * H + row] = __float2bfloat16_rn(o);
            p.attn_s[(size_t)b * H + row] = __float2bfloat16_rn((acc + o) / 2.23606797749979f);
          }
        }
        if (PFE_R && bid < 8 * RKQB) pf_router(i);
        // mixer blocks (no o_proj rows): a throw-away mix warms the instruction cache for the MLP-input mix
      }
      grid_bar(bar, bar_target);

      // ================= phase D: MLP input + router
      int* msrc = ms->amsrc; int nms = 0;
      for (int k = 0; k < nblk; k++) msrc[nms++] = blk[k];
      if (bnd) msrc[nms++] = SRC_AS; else msrc[nms++] = SRC_SUM + cur;
      {
        const float** mqs = ms->amq; mqs[0] = L.mq;
        MARK(10);
        if (loopg) { if (!SKG) gate_phases(3, 1, msrc, nms, mqs, 1, false); else { grid_bar(bar, bar_target); grid_bar(bar, bar_target); } }
        if (!SKD) {
        if (B > 1) {
          u16** row1 = ms->arow1; row1[0] = xs;
          const bf16** nw1 = ms->anw1; nw1[0] = L.nm;
          if (loopg) { const float** src1 = ms->asrc1; src1[0] = p.xg + 3 * H; norm_f32_rows_w(src1, 4 * H, B, 1, nw1, row1, XSH); }
          else {
            ++mev;
            if (MIXB(bid) >= 0 && MIXB(bid) < B) {
              u16** gr = ms->agr; gr[0] = reinterpret_cast<u16*>(p.xnd + (size_t)MIXB(bid) * H);
              const bf16** nwd = ms->anw1; nwd[0] = L.nm;
              mixnorm(p, MIXB(bid), msrc, nms, mqs, 1, 1, ms->azero, nwd, gr, nullptr, ms);
              mix_publish();
            }
            mix_wait_rows(mev, B, p.xnd, H, 0);
            mix_wait_rows_done();
          }
        } else for (int b = 0; b < B; b++) {
          u16** row1 = ms->arow1; row1[0] = xs + b * XSH;
          const bf16** nw1 = ms->anw1; nw1[0] = L.nm;
          if (loopg) {
            const float** src1 = ms->asrc1; src1[0] = p.xg + ((size_t)b * 4 + 3) * H;
            norm_f32_rows(src1, 1, nw1, row1, ms);
          } else {
            u16** rw = ms->arow1; rw[0] = xs + b * XSH;
            const bf16** nwd = ms->anw1; nwd[0] = L.nm;
            mixnorm(p, b, msrc, nms, mqs, 1, 1, ms->azero, nwd, rw, nullptr, ms);
          }
        }
        if (bid < 8 * RKQB && !PFE_R) pf_router(i);
        MARK(11);
        if (bid < 8 * RKQB) {
          consume_tasks<0>(ttab, XSH, rg, pst[warp], [&](int sl, const float* c) { store_task(red, sl, c); });
          __syncthreads();
          if (tid < 16 * B) {
            const int ii = tid / B, b = tid % B;
            float v = 0.f;
#pragma unroll
            for (int s = 0; s < 8; s++) v += red_at(red, s, ii, b);
            p.rlog[((bid / 8) * MAXB + b) * NE + (bid % 8) * 16 + ii] = v;
          }
        }
        }
      }
      grid_bar(bar, bar_target);

      // ================= phase E: top-8, union, w13 + activation
      if (!SKE) {
        MARK(12);
        YTS(0);
#ifdef PROBE
        if (tid == 0) { FMARK(39); u32 v = ldcg32(p.vecs + (size_t)(bid * 64) % (H * 9)); if (v == 0x12345678u) p.bar[40] = v; FMARK(40);
          v = ldcg32(p.vecs + (size_t)(bid * 64 + 8) % (H * 9)); if (v == 0x12345678u) p.bar[40] = v; FMARK(41); }
#endif
        // router selection, one warp per sequence: radix select of the 8th largest key (sigmoid + bias as an ordered
        // integer) over 32 ballots; ties go to the lower expert index; experts lane + 32 j in lane j
        {
          __shared__ u32 selb[MAXB][4];
          __shared__ u32 wcnt[4];
          const float rb_l[4] = {__ldg(L.rbias + lane), __ldg(L.rbias + lane + 32), __ldg(L.rbias + lane + 64), __ldg(L.rbias + lane + 96)};
          float fe13 = 1.f, fe2 = 1.f;
          if (!L.mx && tid < NE) { fe13 = __ldg(L.f13 + tid); fe2 = __ldg(L.f2 + tid); }
          float* sgv = red + MAXB * NE;                         // [MAXB][128] sigmoid(logit)
          if (warp < B) {
            // top-8 of sequence b = warp by redux.sync: 8 rounds of (warp max of the remaining keys, lowest expert
            // index among the maxima); the weights' sum in rank order (as the rank method)
            const int b = warp;
            u32 key[4]; float sg[4]; u32 sel[4] = {0u, 0u, 0u, 0u};
#pragma unroll
            for (int j = 0; j < 4; j++) {
              float lgt = ldcgf(p.rlog + b * NE + lane + 32 * j);
#pragma unroll
              for (int q = 1; q < RKQB; q++) lgt += ldcgf(p.rlog + (q * MAXB + b) * NE + lane + 32 * j);
              sg[j] = sigm(lgt);
              const u32 u = __float_as_uint(sg[j] + rb_l[j]);
              key[j] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // >= 1 for every float but one NaN pattern
              sgv[b * NE + lane + 32 * j] = sg[j];
            }
            float t = 0.f;
#pragma unroll
            for (int r = 0; r < TOPK; r++) {
              u32 lm = 0u;
#pragma unroll
              for (int j = 0; j < 4; j++) lm = max(lm, sel[j] ? 0u : key[j]);
              const u32 M = __reduce_max_sync(0xffffffffu, lm);
              u32 idx = 255u;
#pragma unroll
              for (int j = 3; j >= 0; j--) if (!sel[j] && key[j] == M) idx = (u32)(lane + 32 * j);
              const u32 E = __reduce_min_sync(0xffffffffu, idx);
              const int jj = (int)(E >> 5), ow = (int)(E & 31);
#pragma unroll
              for (int j = 0; j < 4; j++) if (j == jj && lane == ow) sel[j] = 1u;
              const float sv_ = jj == 0 ? sg[0] : jj == 1 ? sg[1] : jj == 2 ? sg[2] : sg[3];
              t += __shfl_sync(0xffffffffu, sv_, ow);
            }
#pragma unroll
            for (int j = 0; j < 4; j++) { const u32 bl = __ballot_sync(0xffffffffu, sel[j] != 0u); if (lane == 0) selb[b][j] = bl; }
            YTS(1);
            if (lane == 0) ms->tot[b] = 8.0f / (t + 1e-20f);
          }
          __syncthreads();
          u32 m = 0, bal = 0;
          if (tid < NE) {
            for (int b = 0; b < B; b++) m |= ((selb[b][tid >> 5] >> (tid & 31)) & 1u) << b;
            bal = __ballot_sync(0xffffffffu, m != 0);
            if (lane == 0) wcnt[warp] = __popc(bal);
          }
          __syncthreads();
          if (tid < NE && m) {
            int sl = __popc(bal & ((1u << lane) - 1));
            for (int w = 0; w < warp; w++) sl += wcnt[w];
            const int e = tid;
            ms->sexp[sl] = e; ms->smask[sl] = m;
            ms->f13[sl] = fe13; ms->f2[sl] = fe2;
            for (int b = 0; b < B; b++) ms->rw[b][sl] = (m >> b & 1) ? sgv[b * NE + e] * ms->tot[b] : 0.f;
          }
          YTS(2);
          if (tid == 0) ms->U = wcnt[0] + wcnt[1] + wcnt[2] + wcnt[3];
          __syncthreads();
          // prefix of the routed pairs per slot (warp 0: two slots per lane)
          if (warp == 0) {
            const int U = ms->U;
            const int c0_ = (2 * lane < U) ? __popc(ms->smask[2 * lane]) : 0, c1_ = (2 * lane + 1 < U) ? __popc(ms->smask[2 * lane + 1]) : 0;
            int incl = c0_ + c1_;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { const int v = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += v; }
            const int excl = incl - c0_ - c1_;
            if (2 * lane <= U) ms->pcnt[2 * lane] = excl;
            if (2 * lane + 1 <= U) ms->pcnt[2 * lane + 1] = excl + c0_;
            if (lane == 31) ms->pcnt[64] = incl;
          }
          __syncthreads();
        }
        const int U = ms->U;
        if (RDBG && bid == 0 && tid < TOPK * B) {   // the chosen experts of every application [ap][b][8] (diagnostics only)
          const int b = tid / TOPK, r = tid % TOPK;
          int cnt = 0, e = -1;
          for (int sl = 0; sl < U && e < 0; sl++) if (ms->smask[sl] >> b & 1) { if (cnt == r) e = ms->sexp[sl]; cnt++; }
          p.rdbg[(ap * MAXB + b) * TOPK + r] = e;
        }
        if (W2L2 && bid < 144 && tid < U) {   // w2 rows of this block's rowtile into L2 during w13
          const int e = ms->sexp[tid];
          if (L.mx) { const size_t it = ((size_t)e * 144 + bid) * 36; l2pf(L.w2c + it * 32, 36 * 512); l2pf(L.w2s + it * 16, 36 * 16); }
          else { const size_t it = ((size_t)e * 144 + bid) * 18; l2pf(L.w2c + it * 32, 18 * 512); l2pf(L.w2s + it * 64, 18 * 64); }
        }
#if PADN > 0
        // merge run: code-placement study (never taken: prof_cap is never this value); shifts the code that follows
        if (p.prof_cap == -77777) {
#pragma unroll
          for (int k = 0; k < PADN; k++) p.bar[200 + k] = (u32)(k * 2654435761u) ^ (u32)tid;
        }
#endif
        MARK(13);
        YTS(3);
        const int N = U * 144;
        const int u0 = (int)((long)bid * N / NB), u1 = SKE2 ? u0 : (int)((long)(bid + 1) * N / NB);
        // one task per (expert slot, rowtile) unit over the whole K: no rounds, no reduction buffer; the activation is
        // computed in the task's completion (gate row 2 ii in lane g = 2 (ii % 4), up row 2 ii + 1 in lane g + 1)
        auto fin13 = [&](int uu, const float* r) {
          const int sl = uu / 144, rt = uu % 144, g = lane >> 2, t = lane & 3;
          const float f = ms->f13[sl];
          const u32 m = ms->smask[sl];
          float o[4];
#pragma unroll
          for (int q = 0; q < 4; q++) o[q] = __shfl_xor_sync(0xffffffffu, r[q], 4);
          if (!(g & 1)) {
#pragma unroll
            for (int h = 0; h < 2; h++)
#pragma unroll
              for (int cc = 0; cc < 2; cc++) {
                const int b = 2 * t + cc;
                if (b < B && (m >> b & 1)) {
                  const float gt = bfr(r[h * 2 + cc] * f), up = bfr(o[h * 2 + cc] * f);
                  const float a = (4.0f * tanhf(gt * 0.25f)) * sigm(gt) * (25.0f * tanhf(up * 0.04f));
                  p.act[((size_t)sl * MAXB + b) * IM + xfo(rt * 8 + h * 4 + (g >> 1))] = __float2bfloat16_rn(a);
                }
              }
          }
        };
        // B = 1 (at most one unit per warp): the unit's 12-item stages all in flight at once (3 buffers: the warp's two
        // ring stages, idle here, and one in xs past the x row) on dedicated barriers, instead of the 2-stage ring
        auto deep13 = [&](auto kind_tag) {
          constexpr int KIND = decltype(kind_tag)::value;
          constexpr int SB = KindInfo<KIND>::SB, IT = KIND == 2 ? 72 : 36, NSG = IT / RI;   // stages of RI items
          if (warp >= u1 - u0) return;
          const int uu = u0 + warp, sl = uu / 144, rt = uu % 144, e = ms->sexp[sl];
          const size_t it = ((size_t)e * 144 + rt) * IT;
          const uint4* w = L.w13c + it * 32;
          const unsigned char* sc = L.w13s + it * SB;
          const u32 bar0 = smaddr(smem + SM_BARS) + 128 + warp * 24;
          auto buf = [&](int k) -> u32 { return k < 2 ? rg.data + k * (RI * ITEM) : smaddr(xs) + 8192 + warp * (RI * ITEM); };
          auto issue = [&](int k) {
            const int bk = k % 3;
            mbar_expect(bar0 + bk * 8, RI * (IBYTES + SB));
            tma_load(buf(bk), w + (size_t)k * RI * 32, RI * IBYTES, bar0 + bk * 8);
            tma_load(buf(bk) + RI * IBYTES, sc + (size_t)k * RI * SB, RI * SB, bar0 + bk * 8);
          };
          if (lane == 0) { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); for (int k = 0; k < 3 && k < NSG; k++) issue(k); }
          const int g = lane >> 2, t = lane & 3;
          const u32 cx = smaddr(xs + g * XSH) + t * 32;
          float c[NACC][4];
#pragma unroll
          for (int a_ = 0; a_ < NACC; a_++) c[a_][0] = c[a_][1] = c[a_][2] = c[a_][3] = 0.f;
#pragma unroll 1
          for (int k = 0; k < NSG; k++) {
            const int bk = k % 3;
            mbar_wait(bar0 + bk * 8, ms->dpc[warp][bk] & 1);
            const u32 st = buf(bk);
#pragma unroll
            for (int i = 0; i < RI; i++) item<KIND>(c, st, cx, k * RI, i, lane, g);
            __syncwarp();
            if (lane == 0) { ms->dpc[warp][bk]++; if (k + 3 < NSG) issue(k + 3); }
            __syncwarp();
          }
          float r[4];
#pragma unroll
          for (int q = 0; q < 4; q++) r[q] = (c[0][q] + c[1][q]) + (c[2][q] + c[3][q]);
          fin13(uu, r);
        };
        if (W13D && ((BC == 1 && DEADB1) || (B == 1 && u1 - u0 <= NW))) {   // B = 1: at most 8 units per block (1152 / 152)
          if (L.mx) deep13(std::integral_constant<int, 2>());
          else deep13(std::integral_constant<int, 1>());
        } else if (L.mx) {
          run_tasks<2>(u1 - u0, ttab, XSH, rg, pst[warp], [&](int task, TD& d) {
            const int uu = u0 + task, sl = uu / 144, rt = uu % 144, e = ms->sexp[sl];
            const size_t it = ((size_t)e * 144 + rt) * 72;
            d.w = L.w13c + it * 32; d.s = L.w13s + it * 16; d.x = xs; d.k0 = 0; d.n = 72; d.slot = uu; d.bm = bmask;
          }, fin13);
        } else {
          run_tasks<1>(u1 - u0, ttab, XSH, rg, pst[warp], [&](int task, TD& d) {
            const int uu = u0 + task, sl = uu / 144, rt = uu % 144, e = ms->sexp[sl];
            const size_t it = ((size_t)e * 144 + rt) * 36;
            d.w = L.w13c + it * 32; d.s = L.w13s + it * 64; d.x = xs; d.k0 = 0; d.n = 36; d.slot = uu; d.bm = bmask;
          }, fin13);
        }
        YTS(4);
        __syncthreads();
        MARK(17);
        YTS(5);
        if (!(BC == 8 && NOFR8)) { fround(0);   // B = 8 never takes the round-based w2 path
        __syncthreads(); }
        if (PFE_DN) pf_down(i, ms->fr_s0, ms->fr_s1);
#ifdef ATSE
        if (tid == 0 && ap == ATSE) { long long* o = reinterpret_cast<long long*>(p.prof) + bid * 12; for (int k = 0; k < 6; k++) o[k] = ms->ats[k] - ms->ats[0]; }
#endif
      }
      // B = 8 path: no grid barrier into F; each block counts its finished w13 units per slot, F waits per slot
      // B = 8: always 64 routed pairs (> 32); B = 1: 8 pairs: decided at compile time in those kernels
      const bool funit = !SKE && !SKF && FUNIT && SCNT && (BC == 8 || (!(BC == 1 && NOFU1) && (ms->pcnt[ms->U] > 32 || 2 * ms->U > RED_SLOTS)));
      if (funit) {
        const int U = ms->U, N = U * 144;
        const int u0 = (int)((long)bid * N / NB), u1 = (int)((long)(bid + 1) * N / NB);
        const int sa = u0 / 144, sb = (u1 - 1) / 144;
#if SCNT_REL
        __syncthreads();   // the block's activations, then release increments (cumulative over the barrier)
        if (tid <= sb - sa) {
          const u32 v = (u32)(min(u1, (sa + tid + 1) * 144) - max(u0, (sa + tid) * 144));
          asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p.bar + 320 + sa + tid), "r"(v) : "memory");
        }
#else
        __threadfence();
        __syncthreads();
        if (tid <= sb - sa) atomicAdd(p.bar + 320 + sa + tid, (u32)(min(u1, (sa + tid + 1) * 144) - max(u0, (sa + tid) * 144)));
#endif
        if (tid < U) ms->stgt[tid] += 144;
      } else grid_bar(bar, bar_target);

      // ================= phase F: w2, weighted sum, MLP conv, new partial
      if (!SKF) {
        MARK(14);
        const int U = ms->U;
        float* oacc = reinterpret_cast<float*>(smem + SM_MISC) + 0;   // placeholder (not used)
        (void)oacc;
        float acc_r = 0.f;   // thread (ii, b) = tid < 16 * B accumulates its row
        const int NPT = ms->pcnt[U];
        if (FUNIT && (BC == 8 || (!(BC == 1 && NOFU1) && (NPT > 32 || 2 * U > RED_SLOTS)))) {
          // many (slot, token) pairs (B = 8): w2 split like w13, contiguous (slot, rowtile) units per block (at most two
          // slots: <= 16 activation rows in xs), y = bf16(sum * f2) of every routed pair -> yb; after a grid barrier
          // rowtile block bid sums each token's experts in slot order
          const int N = U * 144;
          const int u0 = (int)((long)bid * N / NB), u1 = (int)((long)(bid + 1) * N / NB);
          const int sa = u0 / 144, sb = u1 > u0 ? (u1 - 1) / 144 : sa;
          const int npl = u1 > u0 ? ms->pcnt[sb + 1] - ms->pcnt[sa] : 0;
          if (tid < B) { int n = 0; for (int sl = 0; sl < U && n < TOPK; sl++) if (ms->smask[sl] >> tid & 1) ms->sel8[tid][n++] = sl; }
          if (tid <= sb - sa && u1 > u0) {
            const int sl = sa + tid;
            const u32 m = ms->smask[sl];
            int base = ms->pcnt[sl] - ms->pcnt[sa];
            u64 r8 = 0;
#pragma unroll
            for (int b = 0; b < 8; b++) r8 |= (u64)((m >> b & 1) ? base++ : 0) << (8 * b);
            ms->srows[tid] = r8;
          }
          __syncthreads();
          if (L.mx) {
            prefetch_tasks<2>(u1 - u0, ttab, rg, pst[warp], [&](int task, TD& d) {
              const int uu = u0 + task, sl = uu / 144, rt = uu % 144, e = ms->sexp[sl];
              const size_t it = ((size_t)e * 144 + rt) * 36;
              d.w = L.w2c + it * 32; d.s = L.w2s + it * 16; d.x = xs; d.k0 = 0; d.n = 36; d.slot = uu;
              d.rows8 = ms->srows[sl - sa]; d.bm = ms->smask[sl];
            });
          } else {
            prefetch_tasks<1>(u1 - u0, ttab, rg, pst[warp], [&](int task, TD& d) {
              const int uu = u0 + task, sl = uu / 144, rt = uu % 144, e = ms->sexp[sl];
              const size_t it = ((size_t)e * 144 + rt) * 18;
              d.w = L.w2c + it * 32; d.s = L.w2s + it * 64; d.x = xs; d.k0 = 0; d.n = 18; d.slot = uu;
              d.rows8 = ms->srows[sl - sa]; d.bm = ms->smask[sl];
            });
          }
          if (SCNT && warp == 0 && npl > 0 && lane <= sb - sa) {   // the w13 units of this block's slots are done
            const u32* c = p.bar + 320 + sa + lane;
            const u32 tg = ms->stgt[sa + lane];
            if (ld_acquire(c) < tg) { const long long t0 = clock64(); while (ld_acquire(c) < tg) watchdog(t0, 6); }
          }
          if (warp == 0 && npl > 0) {
            if (lane == 0) { asm volatile("fence.proxy.async.global;" ::: "memory"); mbar_expect(smaddr(&ms->sbar), npl * IM * 2); }
            __syncwarp();
            if (lane < npl) {
              int sl = sa;
              while (ms->pcnt[sl + 1] - ms->pcnt[sa] <= lane) sl++;
              u32 m = ms->smask[sl];
              for (int q = lane - (ms->pcnt[sl] - ms->pcnt[sa]); q > 0; q--) m &= m - 1;
              const int b = __ffs(m) - 1;
              tma_load_plain(smaddr(xs + lane * XSI), p.act + ((size_t)sl * MAXB + b) * IM, IM * 2, smaddr(&ms->sbar));
            }
          }
          auto finy = [&](int uu, const float* r) {
            const int sl = uu / 144, rt = uu % 144, g = lane >> 2, t = lane & 3;
            const u32 m = ms->smask[sl];
            const float f = ms->f2[sl];
#pragma unroll
            for (int cc = 0; cc < 2; cc++) {
              const int b = 2 * t + cc;
              if (b < B && (m >> b & 1)) {
                bf16* yp = p.yb + ((size_t)sl * MAXB + b) * H + rt * 16 + g;
                yp[0] = __float2bfloat16_rn(r[cc] * f);
                yp[8] = __float2bfloat16_rn(r[2 + cc] * f);
              }
            }
          };
          if (npl > 0) { mbar_wait(smaddr(&ms->sbar), ms->sphase & 1); }
          __syncthreads();
          if (tid == 0 && npl > 0) ms->sphase++;
          MARK(18);
          if (L.mx) consume_tasks<2>(ttab, XSI, rg, pst[warp], finy);
          else consume_tasks<1>(ttab, XSI, rg, pst[warp], finy);
          grid_bar(bar, bar_target);
          MARK(19);
          if (bid < 144 && tid < 16 * B) {
            const int ii = tid / B, b = tid % B, row = bid * 16 + ii;
            const int* sls = ms->sel8[b];
            float yv[TOPK];
#pragma unroll
            for (int k = 0; k < TOPK; k++) yv[k] = u16f(ldcg16(p.yb + ((size_t)sls[k] * MAXB + b) * H + row));
#pragma unroll
            for (int k = 0; k < TOPK; k++) acc_r = fmaf(ms->rw[b][sls[k]], yv[k], acc_r);
          }
        } else if (bid < 144) {
          bool first = true;
          while (true) {
            const int s0 = ms->fr_s0, s1 = ms->fr_s1, np = ms->fr_np;
            if (s0 >= U) break;
            // bulk copies of the (slot, token) activation rows (fragment order) into xs: pair k by lane k of warp 0
            if (warp == 0) {
              if (lane == 0) { asm volatile("fence.proxy.async.global;" ::: "memory"); mbar_expect(smaddr(&ms->sbar), np * IM * 2); }
              __syncwarp();
              if (lane < np) {
                int sl = s0;
                while (ms->pcnt[sl + 1] - ms->pcnt[s0] <= lane) sl++;
                const int rk = lane - (ms->pcnt[sl] - ms->pcnt[s0]);
                u32 m = ms->smask[sl];
                for (int q = 0; q < rk; q++) m &= m - 1;
                const int b = __ffs(m) - 1;
                tma_load_plain(smaddr(xs + lane * XSI), p.act + ((size_t)sl * MAXB + b) * IM, IM * 2, smaddr(&ms->sbar));
              }
            }
            if (first && !PFE_DN) pf_down(i, s0, s1);
            first = false;
            mbar_wait(smaddr(&ms->sbar), ms->sphase & 1);
            __syncthreads();
            if (tid == 0) ms->sphase++;
            __syncthreads();
            MARK(18);
            if (L.mx) consume_tasks<2>(ttab, XSI, rg, pst[warp], [&](int sl, const float* c) { store_task(red, sl, c); });
            else consume_tasks<1>(ttab, XSI, rg, pst[warp], [&](int sl, const float* c) { store_task(red, sl, c); });
            __syncthreads();
            MARK(19);
            // plan the next round and start its weight stream before this round's accumulation (the ring keeps going)
            if (!(BC == 1 && ONER1) && s1 < U) { fround(s1); __syncthreads(); pf_down(i, ms->fr_s0, ms->fr_s1); }   // B = 1: one round (8 pairs)
            else if (tid == 0) ms->fr_s0 = U;
            if (tid < 16 * B) {
              const int ii = tid / B, b = tid % B;
              for (int sl = s0; sl < s1; sl++) {
                if (!(ms->smask[sl] >> b & 1)) continue;
                const int e = ms->sexp[sl];
                const float f = ms->f2[sl];
                const int ul = sl - s0;
                const float y = bfr((red_at(red, ul * 2, ii, b) + red_at(red, ul * 2 + 1, ii, b)) * f);
                acc_r = fmaf(ms->rw[b][sl], y, acc_r);
              }
            }
            if ((BC == 1 && ONER1) || s1 >= U) break;
            __syncthreads();   // red and xs free for the next round
          }
        }
        if (bid < 144) {
          MARK(20);
          if (tid < 16 * B) {
            const int ii = tid / B, b = tid % B, row = bid * 16 + ii;
            const float o = bfr(acc_r);
            const bf16* w = L.msc + row * 4;
            bf16* rb = p.rm + ((size_t)slot * MAXB + b) * 4 * H;
            float acc = ldgbf(w + 0) * u16f(ldcg16(rb + ((pos - 3) & 3) * H + row));
            acc = fmaf(ldgbf(w + 1), u16f(ldcg16(rb + ((pos - 2) & 3) * H + row)), acc);
            acc = fmaf(ldgbf(w + 2), u16f(ldcg16(rb + ((pos - 1) & 3) * H + row)), acc);
            acc = fmaf(ldgbf(w + 3), o, acc);
            rb[(pos & 3) * H + row] = __float2bfloat16_rn(o);
            const float msv = bfr((acc + o) / 2.23606797749979f);
            const float as = u16f(ldcg16(p.attn_s + (size_t)b * H + row));
            bf16* cp = p.vecs + ((size_t)b * NVEC + cur) * H + row;
            const float res = bnd ? as : (u16f(ldcg16(cp)) + as);
            *cp = __float2bfloat16_rn(res + msv);
          }
        }
        MARK(21);
        if (PFE_QKV && ap + 1 < NAPP) pf_qkv(app_layer(ap + 1));
      }
      grid_bar(bar, bar_target);
    }  // applications
    if (nstop < NAPP) break;

    // ================= head (only where the next token is needed)
    if (st >= P - 1 && !SKIPH) {
      MARK(15);
      int* fsrc = ms->afs; fsrc[0] = V_H0; fsrc[1] = V_P0; fsrc[2] = V_P1 + 2; fsrc[3] = V_P2 + 2; fsrc[4] = V_P3;
      const float** oqs = ms->aoq; oqs[0] = p.oq; ms->azero[0] = 0;
      if (B > 1) {
        u16* row1[1] = {xs};
        const bf16* nw1[1] = {p.fnorm};
        const int q0 = 0;
        ++mev;
        if (MIXB(bid) >= 0 && MIXB(bid) < B) {
          u16** gr = ms->agr; gr[0] = reinterpret_cast<u16*>(p.xnd + (size_t)MIXB(bid) * H);
          const bf16** nwh = ms->anw1; nwh[0] = p.fnorm;
          mixnorm(p, MIXB(bid), fsrc, 5, oqs, 1, 1, ms->azero, nwh, gr, nullptr, ms);
          mix_publish();
        }
        mix_wait_rows(mev, B, p.xnd, H, 0);
        mix_wait_rows_done();
      } else {
        u16* row1[1] = {xs};
        const bf16* nw1[1] = {p.fnorm};
        const int q0 = 0;
        (void)q0; (void)row1; (void)nw1;
        u16** rw = ms->arow1; rw[0] = xs;
        const bf16** nwh = ms->anw1; nwh[0] = p.fnorm;
        mixnorm(p, 0, fsrc, 5, oqs, 1, 1, ms->azero, nwh, rw, nullptr, ms);
      }
#if SKH
      if (a.mode != 0) {   // ---- track B: certified int8 sketch, then the exact head on the candidate rowtiles only (out of line)
        const uint2 r = sk_head<BC>(p, smem, st, B, rg, bar_target);
        rg.pos = r.x; bar_target = r.y;
        continue;
      }
#endif
      pf_head();
      float hb0 = -INFINITY, hb1 = -INFINITY; int hi0 = 0x7fffffff, hi1 = 0x7fffffff;
      consume_tasks<0>(ttab, XSH, rg, pst[warp], [&](int rt, const float* c) {
        const int g = lane >> 2, t = lane & 3;
        const int row0 = rt * 16 + g, row1 = row0 + 8;
        const int b0 = 2 * t, b1 = 2 * t + 1;
        if (a.mode == 0) {
          if (b0 < B && row0 < VOCAB) a.logits[(size_t)b0 * VOCAB + row0] = 30.0f * tanhf(c[0] * (1.0f / 30.0f));
          if (b1 < B && row0 < VOCAB) a.logits[(size_t)b1 * VOCAB + row0] = 30.0f * tanhf(c[1] * (1.0f / 30.0f));
          if (b0 < B && row1 < VOCAB) a.logits[(size_t)b0 * VOCAB + row1] = 30.0f * tanhf(c[2] * (1.0f / 30.0f));
          if (b1 < B && row1 < VOCAB) a.logits[(size_t)b1 * VOCAB + row1] = 30.0f * tanhf(c[3] * (1.0f / 30.0f));
        } else {
          // argmax of the soft-capped logits (the values step() returns), lowest index on ties, as torch.argmax
          const float v0 = 30.0f * tanhf(c[0] * (1.0f / 30.0f)), v1 = 30.0f * tanhf(c[1] * (1.0f / 30.0f));
          const float v2 = 30.0f * tanhf(c[2] * (1.0f / 30.0f)), v3 = 30.0f * tanhf(c[3] * (1.0f / 30.0f));
          if (row0 < VOCAB) { if (v0 > hb0 || (v0 == hb0 && row0 < hi0)) { hb0 = v0; hi0 = row0; } if (v1 > hb1 || (v1 == hb1 && row0 < hi1)) { hb1 = v1; hi1 = row0; } }
          if (row1 < VOCAB) { if (v2 > hb0 || (v2 == hb0 && row1 < hi0)) { hb0 = v2; hi0 = row1; } if (v3 > hb1 || (v3 == hb1 && row1 < hi1)) { hb1 = v3; hi1 = row1; } }
        }
      });
      if (a.mode != 0) {
        // lanes with the same t hold columns 2t, 2t+1: reduce over g, then over warps
#pragma unroll
        for (int o = 4; o < 32; o <<= 1) {
          float ov = __shfl_xor_sync(0xffffffffu, hb0, o); int oi = __shfl_xor_sync(0xffffffffu, hi0, o);
          if (ov > hb0 || (ov == hb0 && oi < hi0)) { hb0 = ov; hi0 = oi; }
          ov = __shfl_xor_sync(0xffffffffu, hb1, o); oi = __shfl_xor_sync(0xffffffffu, hi1, o);
          if (ov > hb1 || (ov == hb1 && oi < hi1)) { hb1 = ov; hi1 = oi; }
        }
        if (lane < 4) {
          ms->tmpf[warp * 8 + 2 * lane] = hb0; ms->tmpi[warp * 8 + 2 * lane] = hi0;
          ms->tmpf[warp * 8 + 2 * lane + 1] = hb1; ms->tmpi[warp * 8 + 2 * lane + 1] = hi1;
        }
        __syncthreads();
        if (tid < B) {
          float v = -INFINITY; int ix = 0x7fffffff;
          for (int w = 0; w < NW; w++) {
            const float ov = ms->tmpf[w * 8 + tid]; const int oi = ms->tmpi[w * 8 + tid];
            if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
          }
          p.amv[bid * MAXB + tid] = v; p.ami[bid * MAXB + tid] = ix;
        }
      }
      grid_bar(bar, bar_target);
    }
  }  // steps
  // the last step's token
  if (a.mode != 0 && warp < B) {
    const int b = warp, P = a.nprompt, ngen = a.nsteps - P + 1;
    float bv = -INFINITY; int bi = 0x7fffffff;
    for (int k = lane; k < NB; k += 32) {
      const float v = ldcgf(p.amv + k * MAXB + b); const int ix = ldcgi(p.ami + k * MAXB + b);
      if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o); const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    const int tk = (bi >= 0 && bi < VOCAB) ? bi : 0;
    if (bid == 0 && lane == 0) {
      if (a.mode == 1) a.out_tok[(size_t)b * ngen + ngen - 1] = tk;
      else a.host_tok[b * ngen + ngen - 1] = tk;
    }
  }
  if (a.mode == 2 && bid == 0) { __syncthreads(); if (tid == 0) { __threadfence_system(); *a.host_prog = a.nsteps - a.nprompt + 1; } }
  // reset the barrier counter for the next launch: the last block out
  if (tid == 0) {
    __threadfence();
    const u32 v = atomicAdd(bar + 1, 1u);
    if (v == NB - 1) { bar[0] = 0; bar[1] = 0; for (int b = 0; b < MAXB; b++) bar[64 + 32 * b] = 0; for (int k = 0; k < MAXU; k++) bar[320 + k] = 0; __threadfence(); }
  }
}

// ------------------------------------------------------------------------------------------------ host side
struct Engine { Params p; int* host_tok; int* host_prog; size_t cap; };

extern "C" void* mk_create(const void* params) {
  Engine* e = new Engine;
  memcpy(&e->p, params, sizeof(Params));
  if (cudaFuncSetAttribute(mk_kernel_t<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL) != cudaSuccess) return nullptr;
#if BSPEC
  if (cudaFuncSetAttribute(mk_kernel_t<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL) != cudaSuccess) return nullptr;
  if (cudaFuncSetAttribute(mk_kernel_t<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL) != cudaSuccess) return nullptr;
#endif
  e->cap = (size_t)MAXB * (size_t)e->p.maxctx;
  if (cudaHostAlloc(&e->host_tok, e->cap * sizeof(int), cudaHostAllocMapped) != cudaSuccess) return nullptr;
  if (cudaHostAlloc(&e->host_prog, 64, cudaHostAllocMapped) != cudaSuccess) return nullptr;
  return e;
}
extern "C" void* mk_host_tok(void* h) { Engine* e = (Engine*)h; void* d; cudaHostGetDevicePointer(&d, e->host_tok, 0); return d; }
extern "C" void* mk_host_prog(void* h) { Engine* e = (Engine*)h; void* d; cudaHostGetDevicePointer(&d, e->host_prog, 0); return d; }
extern "C" void mk_prog_reset(void* h) { Engine* e = (Engine*)h; *(volatile int*)e->host_prog = 0; __atomic_thread_fence(__ATOMIC_SEQ_CST); }
// progress (tokens written per sequence) and a copy of the first `upto` tokens of every sequence (row stride ngen)
extern "C" int mk_poll(void* h, int* buf, int B, int ngen) {
  Engine* e = (Engine*)h;
  const int done = *(volatile int*)e->host_prog;
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  if (done > 0) for (int b = 0; b < B; b++) memcpy(buf + (size_t)b * ngen, e->host_tok + (size_t)b * ngen, sizeof(int) * done);
  return done;
}
extern "C" int mk_skh() { return SKH; }
#if SKH
// every finite float32 bit pattern (all 2^32 minus Inf / NaN): the error of the exact head's softcap a1cap
__global__ void mk_capcheck(unsigned long long* out) {
  double em = 0.0;
  for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < (1ull << 32); i += (unsigned long long)gridDim.x * blockDim.x) {
    const u32 bits = (u32)i;
    if (((bits >> 23) & 0xFFu) == 0xFFu) continue;
    const float x = __uint_as_float(bits);
    const double e = fabs((double)a1cap(x) - 30.0 * tanh((double)x / 30.0));
    em = e > em ? e : em;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) em = fmax(em, __shfl_xor_sync(0xffffffffu, em, o));
  if ((threadIdx.x & 31) == 0) atomicMax(out, (unsigned long long)__double_as_longlong(em));
}
#endif
// SKH: max over every finite float a of |a1cap(a) - 30 tanh(a / 30)| (double reference), exhaustive, on the device
extern "C" double mk_cap_eps() {
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
extern "C" int mk_params_size() { return (int)sizeof(Params); }
// placement tuning (this run): replace the parameter block (buffer pointers) of an engine between launches
extern "C" void mk_set_params(void* h, const void* params) { cudaDeviceSynchronize(); memcpy(&((Engine*)h)->p, params, sizeof(Params)); }
extern "C" int mk_layer_size() { return (int)sizeof(LayerP); }
extern "C" int mk_misc_size() { return (int)sizeof(Misc); }

extern "C" int mk_run(void* h, const void* tok, void* logits, void* out_tok, int B, int pos0, int nsteps, int mode, int stop_app,
                      int nprompt, void* host_tok, void* host_prog, void* stream) {
  Engine* e = (Engine*)h;
  RunArgs a;
  memset(&a, 0, sizeof(a));
  a.tok = (const long long*)tok; a.logits = (float*)logits; a.out_tok = (long long*)out_tok;
  a.B = B; a.pos0 = pos0; a.nsteps = nsteps; a.mode = mode; a.stop_app = stop_app; a.nprompt = nprompt;
  a.host_tok = (int*)host_tok; a.host_prog = (volatile int*)host_prog;
  void* args[] = {(void*)&e->p, (void*)&a};
  void* kf = (void*)mk_kernel_t<0>;
#if BSPEC
  if (B == 1) kf = (void*)mk_kernel_t<1>;
  else if (B == 8) kf = (void*)mk_kernel_t<8>;
#endif
  cudaError_t err = cudaLaunchCooperativeKernel(kf, dim3(NB), dim3(NT), args, SM_TOTAL, (cudaStream_t)stream);
  if (err != cudaSuccess) { fprintf(stderr, "mk_run: %s\n", cudaGetErrorString(err)); return -1; }
  return 0;
}

// ------------------------------------------------------------------------------------------------ test kernel
// y[r][b] for rowtiles 0..nrt-1 of a tiled matrix (KIND 0/1/2), K columns, B operand x [8][K] bf16 (plain order);
// one block, one task per rowtile (all items), results [nrt*16][8] fp32
template <int KIND>
__global__ void __launch_bounds__(NT, 1) gemv_test(const uint4* w, const unsigned char* s, const bf16* x, int K, int nrt, float* y) {
  extern __shared__ __align__(128) unsigned char smem[];
  u16* xs = reinterpret_cast<u16*>(smem + SM_XS);
  TT* ttab = reinterpret_cast<TT*>(smem + SM_TT);
  PState* pst = reinterpret_cast<PState*>(smem + SM_PS);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  Ring rg;
  rg.data = smaddr(smem + SM_RING) + warp * RING_WARP;
  rg.bars = smaddr(smem + SM_BARS) + warp * RS * 8;
  rg.pos = 0;
  if (lane == 0) for (int k = 0; k < RS; k++) mbar_init(rg.bars + k * 8, 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const int xst = K + 8;
  for (int it = tid; it < 8 * K; it += NT) {
    const int b = it / K, k = it % K;
    xs[b * xst + xfo(k)] = *reinterpret_cast<const u16*>(x + it);
  }
  __syncthreads();
  const int nit = KIND == 0 ? K / 16 : KIND == 1 ? K / 64 : K / 32;
  const int SB = KindInfo<KIND>::SB;
#ifdef GTS
  unsigned long long g0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  const long long c0 = clock64();
#endif
  const size_t t0 = (size_t)blockIdx.x * nrt;   // grid version: block b streams rowtiles [b nrt, (b + 1) nrt)
  w += t0 * nit * 32; s += t0 * nit * SB; y += t0 * 16 * 8;
  run_tasks<KIND>(nrt, ttab, xst, rg, pst[warp], [&](int task, TD& d) {
    d.w = w + (size_t)task * nit * 32; d.s = s + (size_t)task * nit * SB; d.x = xs; d.n = nit; d.slot = task; d.bm = 0xff;
  }, [&](int rt, const float* c) {
    const int g = lane >> 2, t = lane & 3;
    y[(rt * 16 + g) * 8 + 2 * t] = c[0]; y[(rt * 16 + g) * 8 + 2 * t + 1] = c[1];
    y[(rt * 16 + g + 8) * 8 + 2 * t] = c[2]; y[(rt * 16 + g + 8) * 8 + 2 * t + 1] = c[3];
#ifdef GTS
    if (lane == 0 && rt == (int)t0 + warp) { const long long c1 = clock64(); unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
      if (blockIdx.x % 50 == 0) printf("blk %d warp %d first task done %.2f us (gt %.2f)\n", blockIdx.x, warp, (c1 - c0) / 1965.0, (g1 - g0) / 1000.0); }
#endif
  });
#ifdef GTS
  __syncthreads();
  if (tid == 0) { const long long c1 = clock64(); unsigned long long g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    printf("blk %d all done %.2f us start gt %llu end gt %llu\n", blockIdx.x, (c1 - c0) / 1965.0, g0 % 100000000ull, g1 % 100000000ull); }
#endif
}
extern "C" int mk_gemv_test(int kind, const void* w, const void* s, const void* x, int K, int nrt, void* y) {
  void* args[] = {(void*)&w, (void*)&s, (void*)&x, (void*)&K, (void*)&nrt, (void*)&y};
  const void* f = kind == 0 ? (const void*)gemv_test<0> : kind == 1 ? (const void*)gemv_test<1> : (const void*)gemv_test<2>;
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL);
  cudaError_t err = cudaLaunchKernel(f, dim3(1), dim3(NT), args, SM_TOTAL, 0);
  if (err != cudaSuccess) { fprintf(stderr, "gemv_test: %s\n", cudaGetErrorString(err)); return -1; }
  return 0;
}

extern "C" int mk_gemv_grid(int kind, const void* w, const void* s, const void* x, int K, int nrt, int nblk, void* y) {
  void* args[] = {(void*)&w, (void*)&s, (void*)&x, (void*)&K, (void*)&nrt, (void*)&y};
  const void* f = kind == 0 ? (const void*)gemv_test<0> : kind == 1 ? (const void*)gemv_test<1> : (const void*)gemv_test<2>;
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL);
  return cudaLaunchKernel(f, dim3(nblk), dim3(NT), args, SM_TOTAL, 0) == cudaSuccess ? 0 : -1;
}

// ------------------------------------------------------------------------------------------------ mix microbenchmark
__global__ void __launch_bounds__(NT, 1) mix_test(const __grid_constant__ Params p, int reps, int nsrc, long long* cyc) {
  extern __shared__ __align__(128) unsigned char smem[];
  u16* xs = reinterpret_cast<u16*>(smem + SM_XS);
  Misc* ms = reinterpret_cast<Misc*>(smem + SM_MISC);
  const int codes[5] = {V_H0, V_P0, V_P1, V_P2, V_P3};
  const float* qs[1] = {p.L[3].mq};
  const bf16* nw[1] = {p.L[3].nm};
  u16* rows[1] = {xs};
  const int q0 = 0;
  for (int r = 0; r < reps; r++) {
    __syncthreads();
    const long long t0 = clock64();
    const int B = 1;
    mixnorm(p, 0, codes, nsrc, qs, 1, 1, &q0, nw, rows, nullptr, ms);
    const long long t1 = clock64();
    if (threadIdx.x == 0) cyc[blockIdx.x * reps + r] = t1 - t0;
  }
}
extern "C" int mk_mix_test(void* h, int nblk, int reps, int nsrc, void* cyc) {
  Engine* e = (Engine*)h;
  cudaFuncSetAttribute(mix_test, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL);
  void* args[] = {(void*)&e->p, (void*)&reps, (void*)&nsrc, (void*)&cyc};
  cudaError_t err = cudaLaunchKernel((const void*)mix_test, dim3(nblk), dim3(NT), args, SM_TOTAL, 0);
  return err == cudaSuccess ? 0 : -1;
}
