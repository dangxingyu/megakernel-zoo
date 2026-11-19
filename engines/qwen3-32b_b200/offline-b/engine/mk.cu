// Decode megakernel for Qwen3-32B (BF16) on one B200 (GB200 card, 152 SMs).
// One persistent cooperative kernel runs whole decode steps: an op list (embed, per layer: qkv GEMM, attention,
// o GEMM + residual, gate|up GEMM + SiLU*up, down GEMM + residual; lm_head + argmax) with a grid barrier after
// every op. Track A data flow: BF16 weights as stored (re-laid out in MMA fragment order), BF16 activations
// wherever the Hugging Face BF16 model has BF16 tensors, float32 inside operators (RMSNorm statistics, GEMM
// accumulators, attention scores / softmax).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#define NTHR 512
#define NWARP 16
#define HID 5120
#define HD 128
#define QKV_M 10240
#define QDIM 8192
#define FF 25600
#define VOCAB 151936
#define MAXB 8
#define MAXSEG 6
#ifndef W12NQ
#define W12NQ 3   // track B: quads per batch in the norm-prep GEMMs (inputs from shared memory just in time)
#endif
#ifndef APRE
#define APRE 1   // attention prep constants requested before the q/k/v wait
#endif
#ifndef CLNORM
#define CLNORM 1   // track B, 7-8 sequences: the norm prep split over the cluster pair (DSMEM)
#endif
#ifndef DCOMB
#define DCOMB 1   // the global attention combine split over all blocks of a unit
#endif
#ifndef NORM_PACKED
#define NORM_PACKED 1
#endif
#ifndef W12XS1
#define W12XS1 1   // track B, B=1: o and down with their input column in shared memory
#endif
#ifndef W12PAR
#define W12PAR 1   // track B: a quad's escapes patched by their lanes in parallel
#endif
#ifndef W12JIT
#define W12JIT 1
#endif
#ifndef PREB
#define PREB 1
#endif
#ifndef PREN
#define PREN 1   // track A: also the norm-prep GEMMs (1-2 sequences)
#endif
#ifndef PREB_KV
#define PREB_KV 2   // instances 0..PREB_KV use the split barrier and preloads
#endif
#define RMS_EPS 1e-6f
#define NLAYERS 64

enum { OP_GEMM = 1, OP_ATTN = 2 };
enum { EP_QKV = 1, EP_RESID = 2, EP_GATEUP = 3, EP_LOGITS = 4 };
enum { IN_NORM = 1, IN_ATTN = 2, IN_ACT = 3, IN_NORMF = 4 };   // IN_NORMF: RMSNorm applied per k-step in the stream

// op words (int64): 0 type, 1 ep, 2 NT, 3 KS, 4 W, 5 norm weight / q_norm, 6 k_norm, 7 in_mode, 8 layer,
// 9 next GEMM's W (prefetch), 10 its units, 11 bytes per unit; 12 W12 storage; 13 row factors (W12),
// 14 escape list starts per warp (W12), 15 escape entries (W12); 16 words per op
#define OPW 16

struct Params {
  const long long* ops; int nops;
  int B, nsteps, flags, nblk;
  int pos0[MAXB];
  int nforced[MAXB];
  const long long* forced; int fstride;     // forced tokens [b][fstride] (prompt / given)
  const u16* embed;                         // [VOCAB][HID]
  const u16* final_norm;                    // [HID]
  u16* x;                                   // residual stream bf16 [MAXB][HID]
  u16* qkv;                                 // [MAXB][QKV_M]
  u16* attn;                                // fragment layout [QDIM/16][8][16]
  u16* act;                                 // fragment layout [FF/16][8][16]
  float* logits;                            // [MAXB][VOCAB]
  float4* gpart;                            // cut-tile partials [tiles][2][32]
  u32* tcnt;                                // [tiles]
  u64* gkey;                                // [nblk][MAXB] argmax keys
  long long* tok_out; int tok_ld;           // [b][tok_ld] tokens chosen at each step
  int* host_tok; int* host_prog;            // host-mapped: tokens [b][tok_ld], steps done (or null)
  u32* bar; u32 bar_base;
  const u16* rope;                          // [max_ctx][2][128] bf16: cos, sin
  int max_ctx;
  u16* kc; u16* vc;                         // [layer][b][kvh][max_ctx][128] in fragment layouts
  int pf_bytes;                             // L2 prefetch of the next GEMM's first bytes at each op end
  int pf_idle_bytes;                        // ... by blocks without attention work during attention
  u32* dcnt;                                // dataflow counters (zeroed before each launch): [0] attention units done
  int pf_warp15;                            // (unused)
  int host_tag;                             // cohort tag of the host-mapped tokens (12 bits)
  int big_min_b;                            // the large-batch kernel instance from this many sequences on
  int clus;                                 // launched with clusters of 2 blocks (attention combine through DSMEM at S = 2)
  int att_flag;                             // 1: the o projection waits on dcnt[0] instead of a grid barrier
  u16* xf;                                  // residual stream also in B-fragment layout [320][8][16] (IN_NORMF)
  int normf_min_b;                          // IN_NORMF from this many sequences on
  float* ssq;                               // [MAXB][320] per 16-row tile sum of x^2 of the residual stream
  float* apart; float* aml; u32* acnt;       // attention partials [unit][part][8][128], [unit][part][8][2], counters [unit]
  int dbg;                                  // debug ablations (wrong results): 1 no K loads, 2 no score mma
  int att_smax;                             // most blocks per attention unit
  int pf_att;                               // busy attention blocks: o-projection bytes prefetched after the score loads
  int pf_first;                             // ... of which this many right after the block's stream, the rest paced in the barrier
  int pf_piece;                             // bytes per barrier poll
  u64* prof;                                // debug: [step][op] global timer after each barrier (block 0)
  u64* prof2;                               // debug: [block][op][4] phase stamps of the last step
};
__device__ __forceinline__ u64 gtimer() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define MKP(k) do { if (P.prof2 && threadIdx.x == 0) P.prof2[((size_t)blockIdx.x * 800 + opi) * 4 + (k)] = gtimer(); } while (0)
#define MKQ(k) do { if (P.prof2 && threadIdx.x == 0) P.prof2[((size_t)blockIdx.x * 800 + 400 + opi) * 4 + (k)] = gtimer(); } while (0)

#define F_LOGITS 1
#define F_ARGMAX 2

__device__ __forceinline__ float bf2f(u16 b) { return __uint_as_float(((u32)b) << 16); }
__device__ __forceinline__ u16 f2bf(float f) { u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
__device__ __forceinline__ u32 f2bf2(float lo, float hi) { u32 r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r; }
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ u32 f2bf2(float lo, float hi);
// RMSNorm of a bf16 pair as the model does it: bf16(w * bf16(x * r))
// (the product of two bf16 values is exact in float32, so bf16(w * t) is the correctly rounded bf16 product: mul.rn.bf16x2)
__device__ __forceinline__ u32 rmsn2(u32 xw, u32 ww, float r) {
  const u32 t = f2bf2(__uint_as_float(xw << 16) * r, __uint_as_float(xw & 0xffff0000u) * r);
  u32 y; asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(y) : "r"(ww), "r"(t));
  return y;
}
__device__ __forceinline__ u32 pack2(u16 lo, u16 hi) { return (u32)lo | ((u32)hi << 16); }

__device__ __forceinline__ uint4 ldw(const uint4* p) { uint4 v; asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ uint2 ldcg2(const uint2* p) { uint2 v; asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v; }
__device__ __forceinline__ uint4 ldcg4(const uint4* p) { uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ float4 ldcgf4(const float4* p) { float4 v; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ u16 ldcg_u16(const u16* p) { unsigned short v; asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(v) : "l"(p)); return v; }
__device__ __forceinline__ float ld_relaxed(const float* p) { float v; asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ float ldcgf(const float* p) { float v; asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v; }
__device__ __forceinline__ float4 ld_rlx4(const float4* p) { float4 v; asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rlx4(float4* p, float4 v) { asm volatile("st.relaxed.gpu.global.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory"); }
#define SENT_BITS 0x7fbadbadu   // a NaN payload FMA/MMA never produce (their NaNs are canonical)
__device__ __forceinline__ bool is_sent(float x) { return __float_as_uint(x) == SENT_BITS; }
__device__ __forceinline__ float4 sent4() { const float f = __uint_as_float(SENT_BITS); return make_float4(f, f, f, f); }
__device__ __forceinline__ u64 ldcg_u64(const u64* p) { u64 v; asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p)); return v; }
__device__ __forceinline__ void prefetch_l2(const void* p, u32 bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mma(float* c, const uint4& a, const uint2& b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}
__device__ __forceinline__ void mma4(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ int tid_v() { int v; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(v)); return v; }

// slot of k (0..15) inside a k-step's 16 values of one column in B-fragment order: lane t holds k = 2t, 2t+1, 2t+8, 2t+9
__device__ __forceinline__ int fslot(int kk) { return ((kk & 7) >> 1) * 4 + (kk & 1) + ((kk >> 3) << 1); }
__device__ __forceinline__ size_t fidx(int n, int k) { return ((size_t)(k >> 4) * 8 + n) * 16 + fslot(k & 15); }

__device__ __forceinline__ u32 fkey(float f) { const u32 b = __float_as_uint(f); return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }

__device__ __forceinline__ void red_release(u32* p, u32 v) { asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ u32 atom_acqrel(u32* p, u32 v) { u32 r; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
// ------------------------------------------------------------------------------------------------ grid barrier
__device__ __forceinline__ void prefetch_l2(const void* p, u32 bytes);
__device__ __forceinline__ void grid_barrier(const Params& P, u32& nbar, const char* pfp = nullptr, long long pfn = 0) {
  __syncthreads();
  nbar++;
  if (threadIdx.x == 0) {
    const u32 target = P.bar_base + nbar * (u32)P.nblk;
    red_release(P.bar, 1u);
    u32 v;
    while (true) {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(P.bar) : "memory");
      if ((int)(v - target) >= 0) break;
      if (pfn > 0) {   // one piece of the next op's weights per poll
        const u32 n = (u32)min(pfn, (long long)P.pf_piece);
        prefetch_l2(pfp, n); pfp += n; pfn -= n;
      } else {
        __nanosleep(20);
      }
    }
  }
  if (P.prof && blockIdx.x == 0 && threadIdx.x == 0) {
    u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.prof[nbar] = t;
  }
  __syncthreads();
}
// the barrier in two halves: arrive at the end of an op, wait inside the next GEMM after its first weight loads
// (the prefetch range for the wait is kept by thread 0 in shared memory: no state carried in registers across ops)
__device__ __forceinline__ void bar_arrive(const Params& P, u32& nbar, unsigned char* pfs, const char* pfp, long long pfn) {
  __syncthreads();
  nbar++;
  if (threadIdx.x == 0) { red_release(P.bar, 1u); *(const char**)(pfs) = pfp; *(long long*)(pfs + 8) = pfn; }
}
__device__ __forceinline__ bool split_before(const Params& P, int o) {   // a split barrier precedes op o
  if (o <= 0 || o >= P.nops) return false;
  const long long* pr = P.ops + (size_t)(o - 1) * OPW;
  if ((int)pr[0] != OP_GEMM || ((P.att_flag & 2) && pr[1] == EP_QKV)) return false;
  return (int)P.ops[(size_t)o * OPW] == OP_GEMM;
}
__device__ __forceinline__ void bar_wait(const Params& P, u32 nbar, unsigned char* pfs) {
  if (threadIdx.x == 0) {
    const char* pfp = *(const char**)(pfs);
    long long pfn = *(const long long*)(pfs + 8);
    const u32 target = P.bar_base + nbar * (u32)P.nblk;
    u32 v;
    while (true) {
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(P.bar) : "memory");
      if ((int)(v - target) >= 0) break;
      if (pfn > 0) {
        const u32 n = (u32)min(pfn, (long long)P.pf_piece);
        prefetch_l2(pfp, n); pfp += n; pfn -= n;
      } else {
        __nanosleep(20);
      }
    }
    if (P.prof && blockIdx.x == 0) {
      u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); P.prof[nbar] = t;
    }
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------ shared memory
// GEMM: [xs: 320 k-steps x 8 columns x 16 bf16 = 80 KB][red: 32 warps x MAXSEG x 32 lanes x float4 = 64 KB][misc]
// attention: [S: 8 heads x (max_ctx + 8) floats][Q frag 8 KB][O partial 2 x 8 x 128 floats][misc]
#define XROW 5136   // bf16 per column of the input (5120 + 16 pad: conflict-free reads and writes)
#define XS_BYTES (8 * XROW * 2)
#define RED_OFF XS_BYTES
#define RED_BYTES (NWARP * MAXSEG * 32 * 16)
#define MISC_OFF (RED_OFF + RED_BYTES)
#define MISC_BYTES 2048
#define PFS_OFF (MISC_OFF + 1104)   // split barrier: thread 0's prefetch range for the wait
// misc: [0, 768) norm partials and 1/rms; [768, 832) argmax keys; [832, 1088) warp tile bounds; [1088, 1104) token
#define SCL_OFF (MISC_OFF + MISC_BYTES)   // W12: row factors 2^(lo - 128) of the block's tiles
#define PATCH_OFF (SCL_OFF + 64 * 16 * 4 + 256)   // W12: per-warp buffer of one quad's decoded operands (escape patches)
#ifdef W12_ONLY
#define SMEM_GEMM (PATCH_OFF + NWARP * 4 * 32 * 16)
#else
#define SMEM_GEMM PATCH_OFF   // (track A: no patch buffer; 8 KB more shared memory costs it +165 us at B=8, e0168)
#endif

// ------------------------------------------------------------------------------------------------ GEMM
// this block's share of a GEMM's weights (U units of ub bytes): start and length in bytes (at most `bytes`)
__device__ __forceinline__ const char* pf_plan(const Params& P, const void* W, long long U, long long ub, long long bytes, long long& nb) {
  nb = 0;
  if (!W || bytes <= 0) return nullptr;
  const long long b0 = U * blockIdx.x / P.nblk, b1 = U * (blockIdx.x + 1) / P.nblk;
  nb = min((b1 - b0) * ub, bytes);
  return (const char*)W + b0 * ub;
}
__device__ __forceinline__ void prefetch_range(const Params& P, const void* W, long long U, long long ub, int bytes) {
  if (threadIdx.x != 0) return;
  long long nb;
  const char* p = pf_plan(P, W, U, ub, bytes, nb);
  for (long long o = 0; o < nb; o += 65536) prefetch_l2(p + o, (u32)min((long long)65536, nb - o));
}
__device__ __forceinline__ void prefetch_next_t(const Params& P, const long long* op, int bytes) {   // (any one thread)
  long long nb;
  const char* p = pf_plan(P, (const void*)op[9], op[10], op[11], bytes, nb);
  for (long long o = 0; o < nb; o += 65536) prefetch_l2(p + o, (u32)min((long long)65536, nb - o));
}
__device__ __forceinline__ void prefetch_next(const Params& P, const long long* op, int bytes) {
  prefetch_range(P, (const void*)op[9], op[10], op[11], bytes);
}

// W12 decode of one k-step's operand registers from its three words (see engine/w12.py)
__device__ __forceinline__ u32 lop_and_or(u32 a, u32 m, u32 c) { u32 r; asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(a), "r"(m), "r"(c)); return r; }   // (a & m) | c
__device__ __forceinline__ uint4 w12_dec3(u32 P, u32 Q, u32 R) {
  uint4 a;
  a.x = lop_and_or(P, 0x87FF87FFu, 0x40004000u);
  a.y = lop_and_or(Q, 0x87FF87FFu, 0x40004000u);
  a.z = lop_and_or(R, 0x87FF87FFu, 0x40004000u);
  const u32 t = lop_and_or(R >> 3, 0x0F000F00u, lop_and_or(Q >> 7, 0x00F000F0u, (P >> 11) & 0x000F000Fu));
  a.w = lop_and_or(t << 4, 0x80008000u, lop_and_or(t, 0x07FF07FFu, 0x40004000u));
  return a;
}
__device__ __forceinline__ u32 pick4(const uint4& v, int j) { return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w; }
__device__ __forceinline__ void setreg(uint4& a, int r, int h, u32 v) {
  u32 x = r == 0 ? a.x : r == 1 ? a.y : r == 2 ? a.z : a.w;
  x = h ? ((x & 0xffffu) | (v << 16)) : ((x & 0xffff0000u) | v);
  if (r == 0) a.x = x; else if (r == 1) a.y = x; else if (r == 2) a.z = x; else a.w = x;
}
// escape (unit, packed = bits << 16 | lane << 11 | j << 9 | r << 7 | h << 6) into the batch's registers (NQ quads from ubase)
template <int NQ>
__device__ __forceinline__ void w12_patch(uint4 (&a)[NQ * 4], int2 en, int ubase, int lane) {
  const u32 pk = (u32)en.y;
  const int k = (en.x - ubase) * 4 + ((pk >> 9) & 3), r = (pk >> 7) & 3, h = (pk >> 6) & 1;
  if (lane == (int)((pk >> 11) & 31)) {
#pragma unroll
    for (int s = 0; s < NQ * 4; s++) if (s == k) setreg(a[s], r, h, pk >> 16);
  }
}
__device__ __forceinline__ int2 ldcg_i2(const int2* p) { int2 v; asm volatile("ld.global.cg.v2.s32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v; }

__device__ __forceinline__ void epilogue(const Params& P, int ep, int T, const float* c, u64* bkey, u64& wa, u64& wb) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n0 = 2 * t, n1 = 2 * t + 1;
  const int B = P.B;
  if (ep == EP_QKV) {
    const int r0 = T * 16 + g, r1 = r0 + 8;
    if (n0 < B) { P.qkv[n0 * QKV_M + r0] = f2bf(c[0]); P.qkv[n0 * QKV_M + r1] = f2bf(c[2]); }
    if (n1 < B) { P.qkv[n1 * QKV_M + r0] = f2bf(c[1]); P.qkv[n1 * QKV_M + r1] = f2bf(c[3]); }
    if (P.att_flag & 2) {   // tile done: count it for its kv group (q tiles 64 per group, then k and v 8 per group)
      __syncwarp();
      if ((threadIdx.x & 31) == 0) red_release(&P.dcnt[1 + (T < 512 ? T >> 6 : ((T - 512) & 63) >> 3)], 1u);
    }
  } else if (ep == EP_RESID) {   // x = bf16(x + bf16(sum)); per tile and column the sum of x^2 (next RMSNorm)
    const int r0 = T * 16 + g, r1 = r0 + 8;
    float q0 = 0.f, q1 = 0.f;
    if (n0 < B) {
      u16* x = P.x + n0 * HID;
      const u16 y0 = f2bf(bf2f(ldcg_u16(x + r0)) + rbf(c[0])), y1 = f2bf(bf2f(ldcg_u16(x + r1)) + rbf(c[2]));
      x[r0] = y0; x[r1] = y1;
      P.xf[fidx(n0, r0)] = y0; P.xf[fidx(n0, r1)] = y1;
      q0 = fmaf(bf2f(y0), bf2f(y0), bf2f(y1) * bf2f(y1));
    }
    if (n1 < B) {
      u16* x = P.x + n1 * HID;
      const u16 y0 = f2bf(bf2f(ldcg_u16(x + r0)) + rbf(c[1])), y1 = f2bf(bf2f(ldcg_u16(x + r1)) + rbf(c[3]));
      x[r0] = y0; x[r1] = y1;
      P.xf[fidx(n1, r0)] = y0; P.xf[fidx(n1, r1)] = y1;
      q1 = fmaf(bf2f(y0), bf2f(y0), bf2f(y1) * bf2f(y1));
    }
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) { q0 += __shfl_xor_sync(0xffffffffu, q0, o); q1 += __shfl_xor_sync(0xffffffffu, q1, o); }
    if (g == 0) {
      if (n0 < B) P.ssq[n0 * 320 + T] = q0;
      if (n1 < B) P.ssq[n1 * 320 + T] = q1;
    }
  } else if (ep == EP_GATEUP) {   // tile T: rows g = gate row T*8+g, rows g+8 = up row T*8+g
    const int i = T * 8 + g;
    if (n0 < B) {
      const float gt = rbf(c[0]); const float s = rbf(gt / (1.0f + expf(-gt)));
      P.act[fidx(n0, i)] = f2bf(s * rbf(c[2]));
    }
    if (n1 < B) {
      const float gt = rbf(c[1]); const float s = rbf(gt / (1.0f + expf(-gt)));
      P.act[fidx(n1, i)] = f2bf(s * rbf(c[3]));
    }
  } else {   // EP_LOGITS
    const int r0 = T * 16 + g, r1 = r0 + 8;
    const float v0 = rbf(c[0]), v1 = rbf(c[1]), v2 = rbf(c[2]), v3 = rbf(c[3]);
    if (P.flags & F_LOGITS) {
      if (n0 < B) { P.logits[(size_t)n0 * VOCAB + r0] = v0; P.logits[(size_t)n0 * VOCAB + r1] = v2; }
      if (n1 < B) { P.logits[(size_t)n1 * VOCAB + r0] = v1; P.logits[(size_t)n1 * VOCAB + r1] = v3; }
    }
    // best per column: larger key wins; ties -> lower row (key low word = ~row)
    u64 k0 = ((u64)fkey(v0) << 32) | (u32)~r0, k2 = ((u64)fkey(v2) << 32) | (u32)~r1;
    u64 k1 = ((u64)fkey(v1) << 32) | (u32)~r0, k3 = ((u64)fkey(v3) << 32) | (u32)~r1;
    u64 a = k0 > k2 ? k0 : k2, b = k1 > k3 ? k1 : k3;
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      const u64 ao = __shfl_xor_sync(0xffffffffu, a, o), bo = __shfl_xor_sync(0xffffffffu, b, o);
      a = ao > a ? ao : a; b = bo > b ? bo : b;
    }
    if (g == 0) {
      wa = a > wa ? a : wa; wb = b > wb ? b : wb;   // the warp's best so far (columns 2t, 2t+1 in lanes t = 0..3)
    }
  }
}

// RMSNorm of the residual stream into the block's B-fragment input. 1/rms per column from the producers' per-tile
// sums of squares (warps 0..B-1); the 640 x B (item of 8 values, column) tasks spread evenly over the threads
// (task t: item t % 640, column t / 640); one round trip for x, weights and partials.
// CL (clusters of 2): each block of the pair normalizes half of the items and writes them into both blocks' buffers
// (DSMEM): half the residual read from L2 per block
template <int NB, bool CL = false>
__device__ void norm_input_t(const Params& P, const u16* w, unsigned char* sm) {
  constexpr int NI = HID / 8;                       // 640 items per column
  constexpr int NIB = CL ? NI / 2 : NI;             // items of this block
  constexpr int NT = (NIB * NB + NTHR - 1) / NTHR;  // tasks per thread
  u16* xs = (u16*)sm;
  int ib = 0;   // first item of this block
  u16* xso = xs;   // CL: the partner's buffer
  if constexpr (CL) {
    cg::cluster_group cl = cg::this_cluster();
    ib = (int)cl.block_rank() * NIB;
    xso = cl.map_shared_rank(xs, (int)cl.block_rank() ^ 1);
  }
  float* rr = (float*)(sm + MISC_OFF);              // [8] 1/rms
  const int tid = tid_v(), wid = tid >> 5, lane = tid & 31;
  const int B = P.B;
  float v[10];
  if (wid < B) {
#pragma unroll
    for (int i = 0; i < 10; i++) v[i] = ldcgf(&P.ssq[wid * 320 + i * 32 + lane]);
  }
  uint4 xr[NT], wv[NT];
#pragma unroll
  for (int k = 0; k < NT; k++) {
    const int t = min(tid + k * NTHR, NIB * NB - 1), item = ib + t % NIB, col = t / NIB;
    wv[k] = __ldg((const uint4*)w + item);
    xr[k] = ldcg4((const uint4*)(P.x + (col < B ? col : 0) * HID) + item);
  }
  if (wid < B) {
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < 10; i++) q += v[i];
#pragma unroll
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if (lane == 0) rr[wid] = rsqrtf(q * (1.0f / HID) + RMS_EPS);
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < NT; k++) {
    const int t = tid + k * NTHR;
    if (t - lane < NIB * NB) {   // warp-uniform: the partner exchange needs the whole warp
      const int tc = min(t, NIB * NB - 1), item = ib + tc % NIB, col = tc / NIB;
      const float r = rr[col];
      const u32 x[4] = {xr[k].x, xr[k].y, xr[k].z, xr[k].w};
      const u32 ww[4] = {wv[k].x, wv[k].y, wv[k].z, wv[k].w};
      u32 o[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
#if NORM_PACKED   // the same bits: bf16(x * r) packed, then w * that in one bf16x2 multiply (exact product, one rounding)
        o[j] = rmsn2(x[j], ww[j], r);
#else
        const float a0 = rbf(bf2f((u16)(x[j] & 0xffff)) * r), a1 = rbf(bf2f((u16)(x[j] >> 16)) * r);
        o[j] = f2bf2(bf2f((u16)(ww[j] & 0xffff)) * a0, bf2f((u16)(ww[j] >> 16)) * a1);
#endif
      }
      const int k0 = item * 8, ks = k0 >> 4, hh = (k0 >> 3) & 1;
      // u32 slot 2j + hh of the k-step holds pair j of half hh: exchange with the partner thread (other half, the
      // neighbouring item of the same column) so that each writes 4 consecutive slots
      const u32 s0 = hh ? o[0] : o[2], s1 = hh ? o[1] : o[3];
      const u32 r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
      const uint4 q = hh ? make_uint4(r0, o[2], r1, o[3]) : make_uint4(o[0], r0, o[1], r1);
      if (t < NIB * B) {
        *(uint4*)(xs + (size_t)col * XROW + ks * 16 + hh * 8) = q;
        if (CL) *(uint4*)(xso + (size_t)col * XROW + ks * 16 + hh * 8) = q;
      }
    }
  }
  if constexpr (CL) cg::this_cluster().sync();   // both halves in both buffers
  else __syncthreads();
}
__device__ __forceinline__ void norm_input(const Params& P, const u16* w, unsigned char* sm) {
  if (P.B == 1) norm_input_t<1>(P, w, sm);
  else if (P.B <= 2) norm_input_t<2>(P, w, sm);
  else if (P.B <= 4) norm_input_t<4>(P, w, sm);
  else norm_input_t<8>(P, w, sm);
}

// one GEMM op: this block's contiguous range of (tile, k-step) units, split over its warps
template <int IN, bool W12, bool PRE = false, bool XS1 = false, bool DC = false, bool CLN = false>
__device__ void gemm_op(const Params& P, const long long* op, unsigned char* sm, int opi, int step,
                        u32 nbar = 0, bool pend = false) {
  MKP(0);
  const int ep = (int)op[1];
  const int NT = (int)op[2], KS = (int)op[3];
  const int KU = W12 ? KS / 4 : KS;   // units per tile: k-steps, or quads of 4 k-steps (W12)
  const uint4* W = (const uint4*)op[4];
  const int tid = tid_v(), wid = tid >> 5, lane = tid & 31;
  // the work is split in quads of 4 k-steps for both storages (the same summation order: track B bit-identical to A)
  const long long UQ = (long long)NT * (KS / 4);
  const int q0 = (int)(UQ * blockIdx.x / P.nblk), q1 = (int)(UQ * (blockIdx.x + 1) / P.nblk);
  const int qw0 = q0 + (int)((long long)(q1 - q0) * wid / NWARP), qw1 = q0 + (int)((long long)(q1 - q0) * (wid + 1) / NWARP);
  const int US = W12 ? 1 : 4;   // units per quad
  const int b0 = q0 * US, b1 = q1 * US, w0 = qw0 * US, w1 = qw1 * US;
  u64* bkey = (u64*)(sm + MISC_OFF + 768);
  // BF16: the warp's first 8 k-steps of weights (or its first segment's, clamped as the tail does) are requested
  // before the wait for the previous op (grid barrier or attention units): their latency overlaps the wait
  uint4 pre[8];
  if constexpr (PRE && !W12 && (IN == IN_ATTN || IN == IN_ACT || (PREN && IN == IN_NORM))) {   // (not across the norm prep: registers; unconditional, clamped)
    const int cnt0 = max(min((w0 / KS + 1) * KS, w1) - w0, 1), ulast = NT * KS - 1;
#pragma unroll
    for (int j = 0; j < 8; j++) pre[j] = ldw(W + (size_t)min(w0 + min(j, cnt0 - 1), ulast) * 32 + lane);
  }
  // W12: NQ quads per batch; JIT: inputs read from shared memory right before their MMA (norm-prep GEMMs; XS1: the
  // single input column of o / down at B=1 staged into shared memory), which frees registers for a third quad
  constexpr bool JIT = W12 && ((IN == IN_NORM && W12JIT) || XS1);
  constexpr int NQ = JIT ? W12NQ : 2;
  uint4 preL[3 * NQ];   // W12: the first batch's quads of words (the same clamping as ld_batch)
  if constexpr (PRE && W12 && (IN == IN_ATTN || IN == IN_ACT)) {
#pragma unroll
    for (int qd = 0; qd < NQ; qd++) {
      const int qq = min(min(w0 + qd, w1 - 1), (int)UQ - 1);
#pragma unroll
      for (int jj = 0; jj < 3; jj++) preL[qd * 3 + jj] = ldw(W + ((size_t)max(qq, 0) * 3 + jj) * 32 + lane);
    }
  }
  if (pend) bar_wait(P, nbar, sm + PFS_OFF);
  if (W12) {   // row factors of this block's tiles (used after the barrier that ends the stream)
    const int t0 = b0 / KU, nt = (b1 - 1) / KU - t0 + 1;
    const float* fs = (const float*)op[13] + t0 * 16;
    for (int i = tid; i < nt * 16; i += NTHR) ((float*)(sm + SCL_OFF))[i] = __ldg(fs + i);
  }
  if constexpr (IN == IN_NORM && CLN) norm_input_t<8, true>(P, (const u16*)op[5], sm);   // (7-8 sequences, clusters of 2)
  else if (IN == IN_NORM) norm_input(P, (const u16*)op[5], sm);
  if (IN == IN_NORMF) {   // 1/rms per column from the per-tile sums of squares; norm weights in fragment order
    float* rr = (float*)(sm + MISC_OFF);
    if (wid < P.B) {
      float v[10];
#pragma unroll
      for (int i = 0; i < 10; i++) v[i] = ldcgf(&P.ssq[wid * 320 + i * 32 + lane]);
      float q = 0.f;
#pragma unroll
      for (int i = 0; i < 10; i++) q += v[i];
#pragma unroll
      for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
      if (lane == 0) rr[wid] = rsqrtf(q * (1.0f / HID) + RMS_EPS);
    }
    for (int i = tid; i < HID / 8; i += NTHR) {   // 8 weights k0 = 8 i: k-step i / 2, half i & 1
      const uint4 v = __ldg((const uint4*)op[5] + i);
      u32* d = (u32*)sm + (i >> 1) * 8 + (i & 1);
      d[0] = v.x; d[2] = v.y; d[4] = v.z; d[6] = v.w;
    }
    __syncthreads();
  }
  if (IN == IN_ATTN && (P.att_flag & 1)) {   // no grid barrier after attention: wait for every unit of this layer
    if (tid == 0) {

      const u32 target = (u32)(8 * P.B * (DC ? min(P.nblk / (8 * P.B), P.att_smax) : 1)) * (u32)(step * NLAYERS + (int)op[8] + 1);   // (DCOMB: S signals per unit)
      u32 v;
      while (true) {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(P.dcnt) : "memory");
        if ((int)(v - target) >= 0) break;
        __nanosleep(32);
      }
    }
    __syncthreads();
  }
  if constexpr (XS1) {   // B=1: column 0 of the input ([k-step][16] bf16) into shared memory
    const uint4* src = (const uint4*)(IN == IN_ATTN ? P.attn : P.act);
    for (int i = tid; i < KS * 2; i += NTHR) ((uint4*)sm)[i] = ldcg4(src + (size_t)(i >> 1) * 16 + (i & 1));
    __syncthreads();
  }
  MKP(1);
  const uint2* xg = (const uint2*)(IN == IN_ATTN ? P.attn : IN == IN_NORMF ? P.xf : P.act);
  const uint2* wfr = (const uint2*)sm + (lane & 3);                        // IN_NORMF: norm weight pairs of k-step kk at [kk * 4]
  const float rcol = IN == IN_NORMF ? ((const float*)(sm + MISC_OFF))[min(lane >> 2, MAXB - 1)] : 0.f;
  const uint2* xsm = (const uint2*)sm;
  const int xoff = (lane >> 2) * (XROW / 4) + (lane & 3);
  const bool xact = (lane >> 2) < P.B;
  float4* red = (float4*)(sm + RED_OFF) + wid * MAXSEG * 32;
  // stream: one segment per tile the warp's range touches
  if (lane == 0) {   // the warp's first and last tile (empty range: none)
    int* wft = (int*)(sm + MISC_OFF + 832);
    wft[2 * wid] = w1 > w0 ? w0 / KU : 1 << 30;
    wft[2 * wid + 1] = w1 > w0 ? (w1 - 1) / KU : -1;
  }
  int seg = 0;
  if (W12) {
    // the warp's escape list through a window of 32 entries, one per lane (one coalesced load per 32 escapes)
    const int2* ent = (const int2*)op[15];
    int ebase = __ldg((const int*)op[14] + blockIdx.x * NWARP + wid), eidx = 0;
    int2 ewin = ldcg_i2(ent + ebase + lane);
    int2 en = make_int2(__shfl_sync(0xffffffffu, ewin.x, 0), __shfl_sync(0xffffffffu, ewin.y, 0));
#define W12_NEXT() do { if (++eidx == 32) { ebase += 32; eidx = 0; ewin = ldcg_i2(ent + ebase + lane); } \
      en = make_int2(__shfl_sync(0xffffffffu, ewin.x, eidx), __shfl_sync(0xffffffffu, ewin.y, eidx)); } while (0)
    // flat loop over the warp's quads, two per batch; the next batch's weights and inputs are requested before the
    // current batch is decoded (k-step by k-step, right before its MMA); a tile boundary flushes the accumulator
    if (w1 > w0) {
      int T = w0 / KU, tend = (T + 1) * KU, tbeg = T * KU;
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      // NQ quads per batch; JIT: the inputs read from shared memory right before their MMA (norm-prep GEMMs), which
      // frees the next batch's input registers for a third quad of weights in flight
      uint4 Lc[3 * NQ]; uint2 xc[4 * NQ];
      auto ld_batch = [&](int q, uint4 (&L)[3 * NQ], uint2 (&xb)[4 * NQ], int tb, bool wl = true) {
#pragma unroll
        for (int qd = 0; qd < NQ; qd++) {
          const int qq = min(q + qd, w1 - 1);
#pragma unroll
          if (wl) for (int jj = 0; jj < 3; jj++) L[qd * 3 + jj] = ldw(W + ((size_t)qq * 3 + jj) * 32 + lane);
          if (JIT) continue;
          int rel = qq - tb; if (rel >= KU) rel -= KU;   // (the second quad may be in the next tile)
#pragma unroll
          for (int j = 0; j < 4; j++) {
            const int kk = rel * 4 + j;
            if (IN == IN_NORM) xb[qd * 4 + j] = xsm[xoff + kk * 4];
            else xb[qd * 4 + j] = xact ? ldcg2(xg + (size_t)kk * 32 + lane) : make_uint2(0u, 0u);
          }
        }
      };
      if constexpr (PRE && (IN == IN_ATTN || IN == IN_ACT)) {   // weights preloaded before the wait: inputs only
#pragma unroll
        for (int jj = 0; jj < 3 * NQ; jj++) Lc[jj] = preL[jj];
        ld_batch(w0, Lc, xc, tbeg, false);
      } else {
        ld_batch(w0, Lc, xc, tbeg);
      }
#define NORMX(xv, kk) (IN == IN_NORMF ? make_uint2(rmsn2((xv).x, wfr[(kk) * 4].x, rcol), rmsn2((xv).y, wfr[(kk) * 4].y, rcol)) : (xv))
#define W12W(qd, i) pick4(Lc[(qd) * 3 + ((i) >> 2)], (i) & 3)   // word i (0..11) of quad qd of the current batch
#define XIN(qd, j) (XS1 ? (lane < 4 ? ((const uint2*)sm)[((Q - tbeg) * 4 + (j)) * 4 + lane] : make_uint2(0u, 0u)) : JIT ? xsm[xoff + ((Q - tbeg) * 4 + (j)) * 4] : xc[(qd) * 4 + (j)])
      for (int q = w0; q < w1; q += NQ) {
        uint4 Ln[3 * NQ]; uint2 xn[4 * NQ];
        {
          const int qn = q + NQ;
          const int tbn = (qn >= tend) ? tend : tbeg;   // tile start of the next batch's first quad
          ld_batch(min(qn, w1 - 1), Ln, xn, tbn);
        }
#pragma unroll
        for (int qd = 0; qd < NQ; qd++) {
          const int Q = q + qd;
          if (Q < w1) {
            if (Q == tend) {   // tile boundary
              red[seg * 32 + lane] = make_float4(c[0], c[1], c[2], c[3]);
              seg++; c[0] = c[1] = c[2] = c[3] = 0.f; tbeg = tend; tend += KU;
            }
            if (en.x <= Q) {   // escapes in this quad: decode its 4 k-steps, patch, multiply
              uint4 a[4];
#pragma unroll
              for (int j = 0; j < 4; j++) a[j] = w12_dec3(W12W(qd, 3 * j), W12W(qd, 3 * j + 1), W12W(qd, 3 * j + 2));
              {   // the quad's operands through the warp's shared-memory buffer: one 16-bit store per escape
                uint4* pb = (uint4*)(sm + PATCH_OFF) + wid * 128;   // [4 k-steps][32 lanes]
#pragma unroll
                for (int j = 0; j < 4; j++) pb[j * 32 + lane] = a[j];
                __syncwarp();
#if W12PAR
                while (en.x <= Q) {   // the quad's escapes in the window (lanes eidx.. up to the first later entry) at once
                  const unsigned later = __ballot_sync(0xffffffffu, lane >= eidx && ewin.x > Q);
                  const int e1 = later ? __ffs(later) - 1 : 32;
                  if (lane >= eidx && lane < e1) {
                    const u32 pk = (u32)ewin.y;
                    const int el = (pk >> 11) & 31, j = (pk >> 9) & 3, r = (pk >> 7) & 3, h = (pk >> 6) & 1;
                    *((u16*)(pb + j * 32 + el) + r * 2 + h) = (u16)(pk >> 16);
                  }
                  eidx = e1;
                  if (eidx == 32) { ebase += 32; eidx = 0; ewin = ldcg_i2(ent + ebase + lane); }
                  en = make_int2(__shfl_sync(0xffffffffu, ewin.x, eidx), __shfl_sync(0xffffffffu, ewin.y, eidx));
                }
#else
                while (en.x <= Q) {
                  const u32 pk = (u32)en.y;
                  if (lane == 0) {   // (one lane writes: the entries are warp-uniform)
                    const int el = (pk >> 11) & 31, j = (pk >> 9) & 3, r = (pk >> 7) & 3, h = (pk >> 6) & 1;
                    *((u16*)(pb + j * 32 + el) + r * 2 + h) = (u16)(pk >> 16);
                  }
                  W12_NEXT();
                }
#endif
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 4; j++) a[j] = pb[j * 32 + lane];
              }
#pragma unroll
              for (int j = 0; j < 4; j++) mma(c, a[j], NORMX(XIN(qd, j), (Q - tbeg) * 4 + j));
            } else {
#pragma unroll
              for (int j = 0; j < 4; j++) {
                const uint4 a = w12_dec3(W12W(qd, 3 * j), W12W(qd, 3 * j + 1), W12W(qd, 3 * j + 2));
                mma(c, a, NORMX(XIN(qd, j), (Q - tbeg) * 4 + j));
              }
            }
          }
        }
#pragma unroll
        for (int jj = 0; jj < 3 * NQ; jj++) Lc[jj] = Ln[jj];
#pragma unroll
        for (int j = 0; j < 4 * NQ; j++) xc[j] = xn[j];
      }
      red[seg * 32 + lane] = make_float4(c[0], c[1], c[2], c[3]);
      seg++;
    }
  } else {
  // PRE: the first batch from the preloaded weights; its sums start the first segment's accumulator
  float c0[4] = {0.f, 0.f, 0.f, 0.f};
  int ub = w0;
  if (PRE && (IN == IN_ATTN || IN == IN_ACT || (PREN && IN == IN_NORM)) && w1 > w0) {
    const int T = w0 / KS, kk0 = w0 - T * KS;
    const int n = min(min((T + 1) * KS, w1) - w0, 8);
    uint2 xb[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int kk = kk0 + min(j, n - 1);
      if (IN == IN_NORM) xb[j] = xsm[xoff + kk * 4];
      else xb[j] = xact ? ldcg2(xg + (size_t)kk * 32 + lane) : make_uint2(0u, 0u);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) if (j < n) mma(c0, pre[j], xb[j]);
    ub = w0 + n;
    if (ub == (T + 1) * KS || ub == w1) {   // the peeled batch ended its segment
      red[seg * 32 + lane] = make_float4(c0[0], c0[1], c0[2], c0[3]);
      seg++; c0[0] = c0[1] = c0[2] = c0[3] = 0.f;
    }
  }
  for (int u = ub; u < w1;) {
    const int T = u / KS;
    const int ue = min((T + 1) * KS, w1);
    const int cnt = ue - u;
    const uint4* wp = W + (size_t)u * 32 + lane;
    const int kk0 = u - T * KS;
    float c[4] = {c0[0], c0[1], c0[2], c0[3]};
    c0[0] = c0[1] = c0[2] = c0[3] = 0.f;
    int i = 0;
    for (; i + 8 <= cnt; i += 8) {
      uint4 a[8]; uint2 xb[8];
#pragma unroll
      for (int j = 0; j < 8; j++) a[j] = ldw(wp + (size_t)(i + j) * 32);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int kk = kk0 + i + j;
        if (IN == IN_NORM) xb[j] = xsm[xoff + kk * 4];
        else xb[j] = xact ? ldcg2(xg + (size_t)kk * 32 + lane) : make_uint2(0u, 0u);
      }
      if (IN == IN_NORMF) {
#pragma unroll
        for (int j = 0; j < 8; j++) { const uint2 wv = wfr[(kk0 + i + j) * 4]; xb[j] = make_uint2(rmsn2(xb[j].x, wv.x, rcol), rmsn2(xb[j].y, wv.y, rcol)); }
      }
#pragma unroll
      for (int j = 0; j < 8; j++) mma(c, a[j], xb[j]);
    }
    if (i < cnt) {   // 1-7 left: every load issued (clamped address), the extra ones not used
      const int n = cnt - i;
      uint4 a[7]; uint2 xb[7];
#pragma unroll
      for (int j = 0; j < 7; j++) a[j] = ldw(wp + (size_t)(i + min(j, n - 1)) * 32);
#pragma unroll
      for (int j = 0; j < 7; j++) {
        const int kk = kk0 + i + min(j, n - 1);
        if (IN == IN_NORM) xb[j] = xsm[xoff + kk * 4];
        else xb[j] = xact ? ldcg2(xg + (size_t)kk * 32 + lane) : make_uint2(0u, 0u);
      }
      if (IN == IN_NORMF) {
#pragma unroll
        for (int j = 0; j < 7; j++) { const uint2 wv = wfr[(kk0 + i + min(j, n - 1)) * 4]; xb[j] = make_uint2(rmsn2(xb[j].x, wv.x, rcol), rmsn2(xb[j].y, wv.y, rcol)); }
      }
#pragma unroll
      for (int j = 0; j < 7; j++) if (j < n) mma(c, a[j], xb[j]);
    }
    red[seg * 32 + lane] = make_float4(c[0], c[1], c[2], c[3]);
    seg++;
    u = ue;
  }
  }
  __syncthreads();
  MKP(2);
#ifdef W12_ONLY   // track B: from the last warp (it has the fewest tiles to finish): -40 us at B=1 and B=8 (e0157)
  if (tid == NTHR - 32) prefetch_next_t(P, op, P.pf_first);
#else             // track A: from thread 0 (the same code from the last warp shifts the large-batch kernel's register
  prefetch_next(P, op, P.pf_first);   // allocation: B=8 +344 us, e0160)
#endif
  // block reduction: tile T gets the warps whose ranges overlap it, in warp order
  const int tb0 = b0 / KU, tb1 = (b1 - 1) / KU;
  u64 wa = 0ull, wb = 0ull;   // EP_LOGITS: the warp's best (value key, ~row) for columns 2t, 2t+1
  const float4* redall = (const float4*)(sm + RED_OFF);
  const int* wft = (const int*)(sm + MISC_OFF + 832);
  for (int T = tb0 + wid; T <= tb1; T += NWARP) {
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    for (int w = 0; w < NWARP; w++) {
      const int f = wft[2 * w], l = wft[2 * w + 1];
      if (T < f || T > l) continue;
      const float4 v = redall[(w * MAXSEG + (T - f)) * 32 + lane];
      s[0] += v.x; s[1] += v.y; s[2] += v.z; s[3] += v.w;
    }
    const bool full = ((T * KU >= b0) && ((T + 1) * KU <= b1)) || (P.dbg & 4);
    float f0 = 1.f, f1 = 1.f;
    if (W12) { const float* fsm = (const float*)(sm + SCL_OFF) + (T - tb0) * 16; f0 = fsm[lane >> 2]; f1 = fsm[(lane >> 2) + 8]; }
    if (full) {
      if (W12) { s[0] *= f0; s[1] *= f0; s[2] *= f1; s[3] *= f1; }
      epilogue(P, ep, T, s, bkey, wa, wb);
    } else if (T * KU < b0) {
      // the tile began in the previous block: wait for its partial (sentinel hand-over), reset the slot, finish
      float4* slot = &P.gpart[(size_t)T * 32 + lane];
      float4 o;
      do { o = ld_rlx4(slot); } while (is_sent(o.x) || is_sent(o.y) || is_sent(o.z) || is_sent(o.w));
      st_rlx4(slot, sent4());
      float r[4] = {o.x + s[0], o.y + s[1], o.z + s[2], o.w + s[3]};   // p0 + p1, as before
      if (W12) { r[0] *= f0; r[1] *= f0; r[2] *= f1; r[3] *= f1; }
      epilogue(P, ep, T, r, bkey, wa, wb);
    } else {   // the tile continues in the next block: hand the partial over
      st_rlx4(&P.gpart[(size_t)T * 32 + lane], make_float4(s[0], s[1], s[2], s[3]));
    }
  }
  u64* wkey = (u64*)(sm + RED_OFF);   // [NWARP][MAXB] (the partial-sum area is free now)
  if (ep == EP_LOGITS) {
    __syncthreads();   // (every warp is done with the partials)
    if (lane < 4) { wkey[wid * MAXB + 2 * lane] = wa; wkey[wid * MAXB + 2 * lane + 1] = wb; }
    __syncthreads();
    if (tid < P.B) {   // the block's best per column over its warps (max with lowest-row ties: order-independent)
      u64 best = 0ull;
      for (int w = 0; w < NWARP; w++) { const u64 k = wkey[w * MAXB + tid]; best = k > best ? k : best; }
      P.gkey[blockIdx.x * MAXB + tid] = best;
    }
  }
}

// ------------------------------------------------------------------------------------------------ attention
// the combine of a unit's parts: one part at a time, in part order (a prefix's state is well defined, so a block can
// fold its own parts and hand the state on)
__device__ __forceinline__ void att_fold(float& m, float& L, float& acc, float mp, float lp, float op) {
  const float mn = fmaxf(m, mp);
  const float sc = __expf(m - mn), w = __expf(mp - mn);   // sc = 0 on the first part (m = -inf)
  L = fmaf(w, lp, L * sc);
  acc = fmaf(w, op, acc * sc);
  m = mn;
}
// Positions are cut into parts of 64 (a function of the position count only: the arithmetic of a sequence does not
// depend on the batch). Unit (sequence b, kv head h) owns S = nblk / (8 B) consecutive blocks; its parts are spread
// over them in contiguous runs. Per part: scores on tensor cores (Q as A with the 8 heads of the group as rows 0-7,
// K as B with 8 positions as columns), softmax statistics of the part (max, sum in float32), P (bf16, relative to
// the part's max) x V. Partials (m, l, O[8][128]) go to global memory; the last block of the unit to arrive combines
// them in part order and writes the bf16 output. K cache per unit: 8-position blocks of [lane][8 k-steps][2 u32]
// (B fragments); V cache: 16-position tiles of [16 dim blocks][lane][2 u32].
#define APART 64
#ifndef JBR_K2
#define JBR_K2 3
#endif
#ifndef KVPF
#define KVPF 1
#endif
#define ATT_BASE(mc) (8 * ((mc) / 2 + 72) * 4 + 2304 + 10 * 128 * 4 + 64 * 16 * 4 + 16)
#define ATT_CO_OFF(mc) ((ATT_BASE(mc) + 15) / 16 * 16)
#define ATT_CLUS_END(mc) (ATT_CO_OFF(mc) + ((mc) / APART / 2 + 1) * 8 * 128 * 4 + 1040 * 4)
template <int KV>   // kernel instance: 0 = 1-2 sequences, 1 = 3-6, 2 = 7-8 (cluster combine; no global combine compiled in)
__device__ void attn_op(const Params& P, const long long* op, int step, unsigned char* sm, int opi) {
  MKP(0);
  const int tid = tid_v(), wid = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int layer = (int)op[8];
  const int S = min(P.nblk / (8 * P.B), P.att_smax);
  const int u = blockIdx.x / S, j = blockIdx.x - u * S;
  if (u >= P.B * 8) { prefetch_next(P, op, P.pf_idle_bytes); return; }
  const int b = u >> 3, h = u & 7;
  const int pos = P.pos0[b] + step;
  const int n = pos + 1;
  const int np = (n + APART - 1) / APART;            // parts of the unit
  const int nact = min(S, np);                       // blocks of the unit with work
  if (j >= nact) {   // (DCOMB: every block of a unit signals once per layer; o waits for 8 B S signals)
    prefetch_next(P, op, P.pf_idle_bytes);
    if (DCOMB && KV != 2 && (P.att_flag & 1) && tid == 0) red_release(&P.dcnt[0], 1u);
    return;
  }
  const int pa = j * np / nact, pb = (j + 1) * np / nact;   // this block's parts
  const int q0 = pa * APART, q1 = min(pb * APART, n);       // its positions
  const bool last = pb == np;                        // holds the new position
  const int mc = P.max_ctx;
  const int SST = mc / 2 + 72;   // score row stride: a block holds at most ceil(np / nact) parts <= mc / 2 + 64 positions (np > 1), 64 (np = 1)
  float* Sm = (float*)sm;                                  // [8][SST] scores of positions q0.. (index p - q0)
  u16* Qs = (u16*)(sm + 8 * SST * 4);                      // [8 heads][136] bf16 after norm + rope (padded rows)
  float* tmp = (float*)(sm + 8 * SST * 4 + 2304);          // [10][128] pre-rope values
  float* ML = tmp + 10 * 128;                              // [parts of this block][8][2] (m, l)
#ifndef CLUS_ATT
#define CLUS_ATT 1
#endif
  const bool clus = KV == 2 && CLUS_ATT && P.clus && S == 2 && nact == 2;   // B = 7-8: the unit's two blocks form a cluster
  float* CO = (float*)(sm + ATT_CO_OFF(mc));                // cluster path: own parts' partial outputs [part][8][128]
  float* ST = CO + (size_t)(mc / APART / 2 + 1) * 8 * 128;   // cluster path: block 0's state [1024 acc][8 m][8 L]
  const size_t unit = ((size_t)layer * MAXB + b) * 8 + h;
  u16* Kc = P.kc + unit * (size_t)mc * HD;
  u16* Vc = P.vc + unit * (size_t)mc * HD;
  const u16* qkv = P.qkv + b * QKV_M;
  // 1-2 sequences: K/V of the cached positions into L2 ahead of the score and P x V rounds while the q/k/v tiles
  // finish (-60 us/step at B=1; at 3-8 sequences equal or worse, also K alone or after the wait: e0194-e0202)
  if (KVPF && KV == 0 && P.B <= 2 && tid == 32 && q0 < pos) {   // (compiled out of the 3-8 instances: allocation)
    const int e = min(q1, pos);
    const int k0 = q0 >> 3, k1 = (e + 7) >> 3, v0 = q0 >> 4, v1 = (e + 15) >> 4;
    for (int x = k0; x < k1; x += 32) prefetch_l2(Kc + (size_t)x * 1024, (u32)min(32, k1 - x) * 2048u);
    for (int x = v0; x < v1; x += 16) prefetch_l2(Vc + (size_t)x * 2048, (u32)min(16, v1 - x) * 4096u);
  }
  // (the prep's constant inputs requested before the wait: norm weights, cos, sin)
  const int pw = threadIdx.x >> 5, pl = threadIdx.x & 31;
  const bool pact = pw < 8 || (pb == np && pw < 10);
  const u16* pnw = (const u16*)(pw < 8 ? op[5] : op[6]);
  const u16* prope = P.rope + (size_t)pos * 256;
  uint2 nwv0 = make_uint2(0u, 0u), cs0 = make_uint2(0u, 0u), sn0 = make_uint2(0u, 0u);
  if (APRE && pact) { nwv0 = __ldg((const uint2*)(pnw + pl * 4)); cs0 = __ldg((const uint2*)(prope + pl * 4)); sn0 = __ldg((const uint2*)(prope + 128 + pl * 4)); }
  if (P.att_flag & 2) {   // no grid barrier after q/k/v: wait for the 80 tiles of this kv group
    if (tid == 0) {
      const u32 target = 80u * (u32)(step * NLAYERS + layer + 1);
      u32 v;
      while (true) {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(P.dcnt + 1 + h) : "memory");
        if ((int)(v - target) >= 0) break;
        __nanosleep(32);
      }
    }
    __syncthreads();
  }
  const u16* rope = P.rope + (size_t)pos * 256;
  const int maxp = mc / APART;
  float* Op = P.apart + ((size_t)u * maxp) * 8 * 128;      // [part][8][128]
  float* Mg = P.aml + ((size_t)u * maxp) * 16;             // [part][8][2]
  // 1. q (8 heads, warps 0-7), k (warp 8, last block only): RMSNorm over 128 dims, RoPE; v (warp 9, last block)
  if (wid < 8 || (last && wid < 10)) {
    const u16* src = wid < 8 ? qkv + (h * 8 + wid) * HD : (wid == 8 ? qkv + QDIM + h * HD : qkv + QDIM + 1024 + h * HD);
    float v[4];
    const uint2 raw = ldcg2((const uint2*)(src + lane * 4));
    const u16* nw = (const u16*)(wid < 8 ? op[5] : op[6]);
    const uint2 nwv = APRE ? nwv0 : __ldg((const uint2*)(nw + lane * 4));   // norm weights, cos, sin
    const uint2 cs = APRE ? cs0 : __ldg((const uint2*)(rope + lane * 4));
    const uint2 sn = APRE ? sn0 : __ldg((const uint2*)(rope + 128 + lane * 4));
    v[0] = bf2f(raw.x & 0xffff); v[1] = bf2f(raw.x >> 16); v[2] = bf2f(raw.y & 0xffff); v[3] = bf2f(raw.y >> 16);
    if (wid < 9) {
      float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
      for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      const float r = rsqrtf(ss * (1.0f / HD) + RMS_EPS);
      const float nwf[4] = {bf2f(nwv.x & 0xffff), bf2f(nwv.x >> 16), bf2f(nwv.y & 0xffff), bf2f(nwv.y >> 16)};
#pragma unroll
      for (int jj = 0; jj < 4; jj++) v[jj] = rbf(nwf[jj] * rbf(v[jj] * r));
    }
    const float cf[4] = {bf2f(cs.x & 0xffff), bf2f(cs.x >> 16), bf2f(cs.y & 0xffff), bf2f(cs.y >> 16)};
    const float sf[4] = {bf2f(sn.x & 0xffff), bf2f(sn.x >> 16), bf2f(sn.y & 0xffff), bf2f(sn.y >> 16)};
#pragma unroll
    for (int jj = 0; jj < 4; jj++) tmp[wid * 128 + lane * 4 + jj] = v[jj];
    __syncwarp();
    if (wid < 9) {
      float o[4];
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        const int d = lane * 4 + jj;
        const float rh = d < 64 ? -tmp[wid * 128 + d + 64] : tmp[wid * 128 + d - 64];
        o[jj] = rbf(rbf(v[jj] * cf[jj]) + rbf(rh * sf[jj]));
      }
      if (wid < 8) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) Qs[wid * 136 + lane * 4 + jj] = f2bf(o[jj]);
      } else {
        const int blk = pos >> 3, cn = pos & 7;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int d = lane * 4 + jj, ks = d >> 4, kk = d & 15;
          const int tt = (kk & 7) >> 1, reg = kk >> 3, half = kk & 1;
          Kc[(((size_t)blk * 4 + (ks >> 1)) * 32 + cn * 4 + tt) * 8 + (((ks & 1) * 2 + reg) * 2 + half)] = f2bf(o[jj]);
        }
      }
    } else {
      const int tile = pos >> 4, pp = pos & 15;
      const int tt = (pp & 7) >> 1, reg = pp >> 3, half = pp & 1;
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        const int d = lane * 4 + jj, db = d >> 3, gg = d & 7;
        Vc[((((size_t)tile * 16 + db) * 32 + gg * 4 + tt) * 2 + reg) * 2 + half] = f2bf(v[jj]);
      }
    }
  }
  __syncthreads();
  MKP(1);
  // 2. scores of positions q0..q1-1: warp w takes 8-position blocks q0/8 + w, + 32, ... (two per round)
  {
    u32 qa[8][2];
#pragma unroll
    for (int ks = 0; ks < 8; ks++) {
      qa[ks][0] = *(const u32*)(Qs + g * 136 + ks * 16 + 2 * t);
      qa[ks][1] = *(const u32*)(Qs + g * 136 + ks * 16 + 8 + 2 * t);
    }
    const float scale = 0.08838834764831845f;   // 128^-0.5
    const int jb0 = q0 >> 3, jb1 = (q1 + 7) >> 3;
    constexpr int JBR = KV == 2 ? JBR_K2 : 2;   // 8-position blocks per warp and round (their K loads in flight together)
    for (int jb = jb0 + wid; jb < jb1; jb += JBR * NWARP) {
      uint4 kv[JBR][4];
#pragma unroll
      for (int r = 0; r < JBR; r++) {
        const int jc = min(jb + r * NWARP, jb1 - 1);   // (a repeat of the last block when past the end)
        const uint4* kp = (const uint4*)Kc + (size_t)jc * 128 + lane;   // [block][k-step pair][lane]: 512 B per load
#pragma unroll
        for (int i = 0; i < 4; i++) kv[r][i] = ldcg4(kp + i * 32);
      }
#pragma unroll
      for (int r = 0; r < JBR; r++) {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 4; i++) {
          mma4(c, qa[2 * i][0], 0u, qa[2 * i][1], 0u, kv[r][i].x, kv[r][i].y);
          mma4(c, qa[2 * i + 1][0], 0u, qa[2 * i + 1][1], 0u, kv[r][i].z, kv[r][i].w);
        }
        const int jc = jb + r * NWARP;
        if (jc < jb1) {
          const int p0 = jc * 8 + 2 * t;
          Sm[g * SST + p0 - q0] = p0 < n ? c[0] * scale : -INFINITY;
          Sm[g * SST + p0 + 1 - q0] = p0 + 1 < n ? c[1] * scale : -INFINITY;
        }
      }
    }
  }
  __syncthreads();
  MKP(2);
  prefetch_next(P, op, P.pf_att);
  const int nbp = pb - pa;
#ifndef SMX2
#define SMX2 1
#endif
#if SMX2
  // 3. per part, one warp: lanes 4h..4h+3 hold head h's 64 scores (16 each, float4 chunks q, q + 4, ...): max, exp
  //    (relative to the part's max), sum over the 4 lanes; positions up to the part's 16-multiple get 0
  for (int pi = wid; pi < nbp; pi += NWARP) {
    const int hh = lane >> 2, qq = lane & 3;
    const int ps0 = (pa + pi) * APART, ps1 = min(ps0 + APART, n), ps16 = min(ps0 + APART, (n + 15) & ~15);
    float* Sh = Sm + hh * SST + ps0 - q0;
    float4 v[4];
#pragma unroll
    for (int i = 0; i < 4; i++) v[i] = *(const float4*)(Sh + (qq + 4 * i) * 4);
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int p = ps0 + (qq + 4 * i) * 4;
      if (p < ps1) m = fmaxf(m, v[i].x);
      if (p + 1 < ps1) m = fmaxf(m, v[i].y);
      if (p + 2 < ps1) m = fmaxf(m, v[i].z);
      if (p + 3 < ps1) m = fmaxf(m, v[i].w);
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    float l = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int p = ps0 + (qq + 4 * i) * 4;
      v[i].x = p < n ? __expf(v[i].x - m) : 0.f;
      v[i].y = p + 1 < n ? __expf(v[i].y - m) : 0.f;
      v[i].z = p + 2 < n ? __expf(v[i].z - m) : 0.f;
      v[i].w = p + 3 < n ? __expf(v[i].w - m) : 0.f;
      l += v[i].x; l += v[i].y; l += v[i].z; l += v[i].w;
      if (p < ps16) *(float4*)(Sh + (qq + 4 * i) * 4) = v[i];
    }
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (qq == 0) { ML[(pi * 8 + hh) * 2] = m; ML[(pi * 8 + hh) * 2 + 1] = l; }
  }
#else
  // 3. per (part, head): max, exp (relative to the part's max), sum; positions up to the part's 16-multiple get 0
  for (int task = wid; task < nbp * 8; task += NWARP) {
    const int pi = task >> 3, hh = task & 7;
    const int ps0 = (pa + pi) * APART, ps1 = min(ps0 + APART, n), ps16 = min(ps0 + APART, (n + 15) & ~15);
    float* Sh = Sm + hh * SST - q0;
    float m = -INFINITY;
    for (int p = ps0 + lane; p < ps1; p += 32) m = fmaxf(m, Sh[p]);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.f;
    for (int p = ps0 + lane; p < ps16; p += 32) {
      const float e = p < n ? __expf(Sh[p] - m) : 0.f;
      Sh[p] = e; l += e;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
    if (lane == 0) { ML[(pi * 8 + hh) * 2] = m; ML[(pi * 8 + hh) * 2 + 1] = l; }
  }
#endif
  __syncthreads();
  // 4. P x V per (part, dim block), two tasks per warp and round: partial outputs to global memory
  //    (one part in all: normalised here and written as the output, the same arithmetic as the combine)
  for (int task0 = wid; task0 < nbp * 16; task0 += 2 * NWARP) {
    uint2 vb[2][4];
    int nt[2], i0[2];
#pragma unroll
    for (int k = 0; k < 2; k++) {   // two tasks: (part, dim block) task0 and task0 + 32 (a repeat when past the end)
      const int task = min(task0 + k * NWARP, nbp * 16 - 1);
      const int pi = task >> 4, db = task & 15;
      const int ps0 = (pa + pi) * APART;
      nt[k] = (min(ps0 + APART, n) - ps0 + 15) >> 4;
      i0[k] = ps0 >> 4;
#pragma unroll
      for (int i = 0; i < 4; i++) vb[k][i] = ldcg2((const uint2*)(Vc + (((size_t)(i0[k] + min(i, nt[k] - 1)) * 16 + db) * 32 + lane) * 4));
    }
#pragma unroll
    for (int k = 0; k < 2; k++) {
      const int task = task0 + k * NWARP;
      if (task >= nbp * 16) break;
      const int pi = task >> 4, db = task & 15;
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        if (i < nt[k]) {
          const float2 p0 = *(const float2*)(Sm + g * SST + (i0[k] + i) * 16 + 2 * t - q0);
          const float2 p1 = *(const float2*)(Sm + g * SST + (i0[k] + i) * 16 + 8 + 2 * t - q0);
          mma4(c, f2bf2(p0.x, p0.y), 0u, f2bf2(p1.x, p1.y), 0u, vb[k][i].x, vb[k][i].y);
        }
      }
      if (np == 1) {
        const float L = ML[g * 2 + 1];
        P.attn[fidx(b, (h * 8 + g) * HD + db * 8 + 2 * t)] = f2bf(fmaf(1.0f, c[0], 0.f) / L);
        P.attn[fidx(b, (h * 8 + g) * HD + db * 8 + 2 * t + 1)] = f2bf(fmaf(1.0f, c[1], 0.f) / L);
      } else if (clus) {   // own parts' partials stay in shared memory
        *(float2*)(CO + ((size_t)pi * 8 + g) * 128 + db * 8 + 2 * t) = make_float2(c[0], c[1]);
      } else {
        float* o = Op + ((size_t)(pa + pi) * 8 + g) * 128 + db * 8 + 2 * t;
        *(float2*)o = make_float2(c[0], c[1]);
      }
    }
  }
  if (!clus && tid < nbp * 16) Mg[(pa + (tid >> 4)) * 16 + (tid & 15)] = ML[tid];
  if (np == 1) {   // output written by this block: the unit is done
    __syncthreads();
    if (tid == 0) red_release(&P.dcnt[0], 1u);
    return;
  }
  __syncthreads();
  MKP(3);
  // The combine folds the parts in order in two halves, [0, np/2) and [np/2, np) (the two blocks of a unit at 7-8
  // sequences), then merges the second half's state into the first's (the same arithmetic on every path)
  if (clus) {   // two blocks per unit in one cluster: both fold their own parts; block 0 hands its state to block 1 (DSMEM)
    cg::cluster_group cl = cg::this_cluster();
    float* st1 = cl.map_shared_rank(ST, 1);   // the state area of block 1 ([1024] acc, [8] m, [8] L)
    float fm[1024 / NTHR], fl[1024 / NTHR], fa[1024 / NTHR];
#pragma unroll
    for (int r = 0; r < 1024 / NTHR; r++) {
      const int it = tid + r * NTHR, hh = it >> 7, d = it & 127;
      float m = -INFINITY, L = 0.f, acc = 0.f;
      for (int pi = 0; pi < nbp; pi++) att_fold(m, L, acc, ML[(pi * 8 + hh) * 2], ML[(pi * 8 + hh) * 2 + 1], CO[((size_t)pi * 8 + hh) * 128 + d]);
      fm[r] = m; fl[r] = L; fa[r] = acc;
      if (j == 0) {
        st1[it] = acc;
        if (d == 0) { st1[1024 + hh] = m; st1[1032 + hh] = L; }
      }
    }
    MKQ(0);
    cl.sync();
    MKQ(1);
    if (j == 0) return;
#pragma unroll
    for (int r = 0; r < 1024 / NTHR; r++) {
      const int it = tid + r * NTHR, hh = it >> 7, d = it & 127;
      float m = ST[1024 + hh], L = ST[1032 + hh], acc = ST[it];
      att_fold(m, L, acc, fm[r], fl[r], fa[r]);
      P.attn[fidx(b, (h * 8 + hh) * HD + d)] = f2bf(acc / L);
    }
    __syncthreads();
    MKQ(2);
    if (tid == 0) red_release(&P.dcnt[0], 1u);   // the unit is done
    return;
  }
  if (KV == 2) return;   // (unreachable: instance 2 runs only where every multi-part unit takes the cluster path)
  __syncthreads();
#if DCOMB
  {   // (instances without the cluster combine; the 7-8 instance keeps one signal per unit: its code unchanged)
  // 5. every block of the unit folds a slice of the 1024 (head, dim) outputs over all parts, in part order, once all
  //    partials are written: the last to arrive (arrival atomic) releases the unit's flag with this layer's epoch
  {
    u32* uflag = P.dcnt + 16 + u;   // (zeroed before each launch)
    const u32 epoch = (u32)(step * NLAYERS + layer + 1);
    if (tid == 0) {
      const u32 old = atom_acqrel(&P.acnt[u], 1u);
      if ((int)old == nact - 1) {
        P.acnt[u] = 0;
        asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(uflag), "r"(epoch) : "memory");
      } else {
        u32 v;
        while (true) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(uflag) : "memory");
          if (v == epoch) break;
          __nanosleep(20);
        }
      }
    }
    __syncthreads();
    const int i0 = j * 1024 / nact, i1 = (j + 1) * 1024 / nact;
    const int it = i0 + tid;
    if (it < i1) {
      const int hh = it >> 7, d = it & 127;
      float m = -INFINITY, L = 0.f, acc = 0.f, m1 = -INFINITY, L1 = 0.f, acc1 = 0.f;
      const int hs = np / 2;   // the second half's first part: its state is folded separately, then merged
      for (int p0 = 0; p0 < np; p0 += 8) {
        float ov[8], mv[8], lv[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int p = min(p0 + i, np - 1);
          ov[i] = ld_relaxed(&Op[((size_t)p * 8 + hh) * 128 + d]);
          mv[i] = ld_relaxed(&Mg[p * 16 + hh * 2]);
          lv[i] = ld_relaxed(&Mg[p * 16 + hh * 2 + 1]);
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
          if (p0 + i < hs) att_fold(m, L, acc, mv[i], lv[i], ov[i]);
          else if (p0 + i < np) att_fold(m1, L1, acc1, mv[i], lv[i], ov[i]);
        }
      }
      att_fold(m, L, acc, m1, L1, acc1);
      P.attn[fidx(b, (h * 8 + hh) * HD + d)] = f2bf(acc / L);
    }
    __syncthreads();
  }
  if (tid == 0) red_release(&P.dcnt[0], 1u);
  return;
  }
#endif
  // 5. the unit's last block to arrive combines the parts in order
  int* flag = (int*)(sm + 8 * SST * 4 + 2304 + 10 * 128 * 4 + 64 * 16 * 4);
  if (tid == 0) {
    const u32 old = atom_acqrel(&P.acnt[u], 1u);
    const int lastb = (int)old == nact - 1;
    if (lastb) P.acnt[u] = 0;
    *flag = lastb;
  }
  __syncthreads();
  if (!*flag) return;
  // every thread (head, dim) folds the parts in order, 8 parts' loads at a time (running max)
  for (int it = tid; it < 1024; it += NTHR) {
    const int hh = it >> 7, d = it & 127;
    float m = -INFINITY, L = 0.f, acc = 0.f, m1 = -INFINITY, L1 = 0.f, acc1 = 0.f;
    const int hs = np / 2;   // the second half's first part: its state is folded separately, then merged
    for (int p0 = 0; p0 < np; p0 += 8) {
      float ov[8], mv[8], lv[8];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int p = min(p0 + i, np - 1);
        ov[i] = ld_relaxed(&Op[((size_t)p * 8 + hh) * 128 + d]);
        mv[i] = ld_relaxed(&Mg[p * 16 + hh * 2]);
        lv[i] = ld_relaxed(&Mg[p * 16 + hh * 2 + 1]);
      }
#pragma unroll
      for (int i = 0; i < 8; i++) {
        if (p0 + i < hs) att_fold(m, L, acc, mv[i], lv[i], ov[i]);
        else if (p0 + i < np) att_fold(m1, L1, acc1, mv[i], lv[i], ov[i]);
      }
    }
    att_fold(m, L, acc, m1, L1, acc1);
    P.attn[fidx(b, (h * 8 + hh) * HD + d)] = f2bf(acc / L);
  }
  __syncthreads();
  if (tid == 0) red_release(&P.dcnt[0], 1u);   // the unit is done
}

// ------------------------------------------------------------------------------------------------ embed / argmax
__device__ __forceinline__ int resolve(const Params& P, int b) {   // argmax over the blocks' best keys (one warp)
  const int lane = threadIdx.x & 31;
  u64 best = 0;
  for (int i = lane; i < P.nblk; i += 32) { const u64 k = ldcg_u64(&P.gkey[i * MAXB + b]); best = k > best ? k : best; }
#pragma unroll
  for (int o = 16; o; o >>= 1) { const u64 k = __shfl_xor_sync(0xffffffffu, best, o); best = k > best ? k : best; }
  return (int)(~(u32)(best & 0xffffffffu));
}

__device__ void embed_op(const Params& P, int s, unsigned char* sm) {
  int* tok = (int*)(sm + MISC_OFF + 1088);
  const int tid = tid_v(), wid = tid >> 5, lane = tid & 31;
  const bool have_prev = (s > 0) && (P.flags & F_ARGMAX);
  if (blockIdx.x == 0 && have_prev) {
    if (wid < P.B) {
      const int tk = resolve(P, wid);
      if (lane == 0) {
        P.tok_out[wid * P.tok_ld + s - 1] = tk;
        if (P.host_tok) P.host_tok[wid * P.tok_ld + s - 1] = (P.host_tag << 18) | tk;   // tagged with the cohort (ids < 2^18)
      }
    }
    __syncthreads();
    if (tid == 0 && P.host_prog) { __threadfence_system(); *(volatile int*)P.host_prog = s; }
  }
  if (s >= P.nsteps) return;
  const int b = blockIdx.x;
  if (b >= P.B) return;
  if (wid == 0) {
    int tk;
    if (s < P.nforced[b]) tk = (int)P.forced[b * P.fstride + s];
    else tk = resolve(P, b);
    if (lane == 0) tok[0] = tk;
  }
  __syncthreads();
  const uint4* src = (const uint4*)(P.embed + (size_t)tok[0] * HID);
  uint4* dst = (uint4*)(P.x + b * HID);
  for (int i = tid; i < HID / 8; i += NTHR) {   // 8 values per item; pairs of items make a 16-row tile's sum of squares
    const uint4 v = __ldg(src + i);
    dst[i] = v;
    {   // the same 8 values into the fragment layout: k0 = 8 i is k-step i / 2, half i & 1 (pair j -> slot 4 j + 2 half)
      u32* xfw = (u32*)(P.xf + ((size_t)(i >> 1) * 8 + b) * 16);
      const int hh = i & 1;
      xfw[0 + hh] = v.x; xfw[2 + hh] = v.y; xfw[4 + hh] = v.z; xfw[6 + hh] = v.w;
    }
    const u32 w[4] = {v.x, v.y, v.z, v.w};
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) { const float a = bf2f(w[j] & 0xffff), c = bf2f(w[j] >> 16); q = fmaf(a, a, q); q = fmaf(c, c, q); }
    q += __shfl_xor_sync(0xffffffffu, q, 1);
    if ((i & 1) == 0) P.ssq[b * 320 + (i >> 1)] = q;
  }
}

// ------------------------------------------------------------------------------------------------ kernel
// Three instances (a register allocation each): 0 for 1-2 sequences (no per-k-step RMSNorm, no cluster combine),
// 1 for 3-6 (per-k-step RMSNorm, global attention combine), 2 for 7-8 (launched with clusters of 2: DSMEM combine)
template <int KV>
__global__ void __launch_bounds__(NTHR, 1) mk_kernel_t(Params P) {
  extern __shared__ __align__(16) unsigned char sm[];
  u32 nbar = 0;
  for (int s = 0; s <= P.nsteps; s++) {
    embed_op(P, s, sm);
    if (s == P.nsteps) break;
    { const long long* lo = P.ops + (size_t)(P.nops - 1) * OPW; prefetch_range(P, (const void*)lo[9], lo[10], lo[11], P.pf_bytes); }   // the first GEMM (layer 0 qkv)
    grid_barrier(P, nbar);
    for (int o = 0; o < P.nops; o++) {
      const long long* op = P.ops + (size_t)o * OPW;
      const int type = (int)op[0];
      if (type == OP_GEMM) {
        const int in = (int)op[7];
#ifdef W12_ONLY   // track B build: every GEMM in 12-bit storage (engine/TRACK = B)
        if (in == IN_NORM) gemm_op<IN_NORM, true, PREB && KV <= PREB_KV, false, false, CLNORM && KV == 2>(P, op, sm, o, s, nbar, PREB && KV <= PREB_KV && split_before(P, o));   // (track B: no per-k-step norm, its decode leaves no issue slots)
        else if (W12XS1 && KV == 0 && P.B == 1 && in == IN_ATTN) gemm_op<IN_ATTN, true, PREB && KV <= PREB_KV, true, DCOMB && KV != 2>(P, op, sm, o, s, nbar, PREB && KV <= PREB_KV && split_before(P, o));
        else if (W12XS1 && KV == 0 && P.B == 1 && in == IN_ACT) gemm_op<IN_ACT, true, PREB && KV <= PREB_KV, true>(P, op, sm, o, s, nbar, PREB && KV <= PREB_KV && split_before(P, o));
        else if (in == IN_ATTN) gemm_op<IN_ATTN, true, PREB && KV <= PREB_KV, false, DCOMB && KV != 2>(P, op, sm, o, s, nbar, PREB && KV <= PREB_KV && split_before(P, o));
        else gemm_op<IN_ACT, true, PREB && KV <= PREB_KV>(P, op, sm, o, s, nbar, PREB && KV <= PREB_KV && split_before(P, o));
#else             // track A build: BF16 as the checkpoint stores it
        if (KV > 0 && in == IN_NORM && P.B >= P.normf_min_b) gemm_op<(KV > 0 ? IN_NORMF : IN_NORM), false, PREB && KV == 0>(P, op, sm, o, s, nbar, PREB && KV <= PREB_KV && split_before(P, o));
        else if (in == IN_NORM) gemm_op<IN_NORM, false, PREB && KV == 0, false, false, CLNORM && KV == 2>(P, op, sm, o, s, nbar, PREB && KV <= PREB_KV && split_before(P, o));
        else if (in == IN_ATTN) gemm_op<IN_ATTN, false, PREB && KV <= PREB_KV, false, DCOMB && KV != 2>(P, op, sm, o, s, nbar, PREB && KV <= PREB_KV && split_before(P, o));
        else gemm_op<IN_ACT, false, PREB && KV <= PREB_KV>(P, op, sm, o, s, nbar, PREB && KV <= PREB_KV && split_before(P, o));
#endif
        { const int opi = o; __syncthreads(); MKP(3); }
        if ((P.att_flag & 2) && op[1] == EP_QKV) continue;   // attention waits on per-group tile counters
        long long nb;
        const char* p = pf_plan(P, (const void*)op[9], op[10], op[11], P.pf_bytes, nb);
        if (PREB && KV <= PREB_KV && split_before(P, o + 1)) {   // the next GEMM waits
          // (1-2 sequences instance only: in the 3-8 instances the same code slowed every stream by ~8%, e0228)
          bar_arrive(P, nbar, sm + PFS_OFF, p + P.pf_first, nb - P.pf_first);
        } else {
          grid_barrier(P, nbar, p + P.pf_first, nb - P.pf_first);
        }
      } else {
        attn_op<KV>(P, op, s, sm, o);
        if (!(P.att_flag & 1)) grid_barrier(P, nbar);
      }
    }
  }
}

extern "C" {
int mk_smem_bytes(int max_ctx) {
  const int att = ATT_CLUS_END(max_ctx);   // attention scratch incl. the cluster path's partials and state
  if (att > MISC_OFF) return 1 << 30;      // it would run into the misc area: max_context too large
  return SMEM_GEMM;
}
int mk_launch(const Params* p, void* stream) {
  static int configured = -1;
  const int smem = mk_smem_bytes(p->max_ctx);
  if (configured != smem) {
    cudaError_t e = cudaFuncSetAttribute(mk_kernel_t<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(mk_kernel_t<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(mk_kernel_t<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return (int)e;
    configured = smem;
  }
  const bool big = p->B >= p->big_min_b;
  const bool clus2 = big && p->clus && p->B >= 7 && p->nblk / (8 * p->B) == 2;   // instance 2
  Params P = *p;
  void* args[] = {&P};
  cudaError_t e;
  if (clus2) {   // clusters of 2 (DSMEM attention combine at 2 blocks per unit); cooperative: all blocks co-resident
    static bool clus_attr = false;
    if (!clus_attr) { cudaFuncSetAttribute(mk_kernel_t<2>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); clus_attr = true; }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(p->nblk); cfg.blockDim = dim3(NTHR); cfg.dynamicSmemBytes = smem; cfg.stream = (cudaStream_t)stream;
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeCooperative; at[1].val.cooperative = 1;
    cfg.attrs = at; cfg.numAttrs = 2;
    e = cudaLaunchKernelExC(&cfg, (const void*)mk_kernel_t<2>, args);
  } else {
    e = cudaLaunchCooperativeKernel(big ? (void*)mk_kernel_t<1> : (void*)mk_kernel_t<0>, dim3(p->nblk), dim3(NTHR), args, smem, (cudaStream_t)stream);
  }
  return (int)e;
}
int mk_params_size() { return (int)sizeof(Params); }
int mk_nwarp() { return NWARP; }
#ifdef W12_ONLY
int mk_track() { return 'B'; }
#else
int mk_track() { return 'A'; }
#endif
int mk_host_alloc(size_t bytes, void** host, void** dev) {
  cudaError_t e = cudaHostAlloc(host, bytes, cudaHostAllocMapped);
  if (e != cudaSuccess) return (int)e;
  return (int)cudaHostGetDevicePointer(dev, *host, 0);
}
}
