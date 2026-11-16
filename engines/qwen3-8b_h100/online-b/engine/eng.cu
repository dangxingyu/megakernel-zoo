// Online engine for Qwen3-8B (bf16) on one H100: a "pass" runs T token columns (decode rows and prefill chunks
// together) once over the weights. Every column's arithmetic is independent of the other columns of the pass,
// of where a prefill is cut into chunks and of whether its prefix K/V came from a cache.
//
// Canonical arithmetic (what makes batching / chunking / prefix reuse invisible):
//  * matvec: y[row] = one f32 accumulator per (row, column), 16 input dims per tensor-core instruction
//    (wgmma m64nNk16, bf16 x bf16 -> f32), dims ascending. Inputs are bf16.
//  * attention of a query at position t over keys 0..t. Positions are cut (at absolute positions) into
//    segments of 128 and super-segments of 128 << SS_LOG.
//    q is stored as bf16(q * log2 e / sqrt(128)), so a score s_j = q . k_j (MMA over the 8 dim-ksteps, f32)
//    is in units of log2. State of a query inside a super-segment: (M, Z, O), starting at (-inf, 0, 0).
//    Per segment, in ascending order: m = max of s_j over the valid positions, R = max(M, m),
//    a = 2^(M - R) (exactly 1 when M == R, 0 for M = -inf), O *= a, Z *= a, w_j = bf16(2^(s_j - R)) (0 for
//    j > t), O += sum_j w_j V_j and Z += sum_j w_j (tensor-core instructions over 16 positions each,
//    positions ascending, accumulating into O and Z; Z against a V dim group of ones), M = R.
//    The super-segments of a query are folded in ascending order from (-inf, 0, 0): M' = max(M, m),
//    a = 2^(M - M'), b = 2^(m - M') (exactly 1 / 0 as above), Z' = Z a + z b, O' = O a + o b.
//    Output = bf16(O * (1 / Z)).
//    So a (query, super-segment) part can be computed anywhere, in any pass, next to any other work.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>

#define H 4096
#define NL 36
#define QKV_ROWS 6144
#define INNER 12288
#define VOCAB 151936
#define QKV_T 384
#define O_T 256
#define GU_T 1536
#define DN_T 256
#define KS_H 256
#define KS_I 768
#define OFF_QKV 0ull
#define OFF_O (OFF_QKV + (size_t)QKV_T * KS_H * 32)
#define OFF_GU (OFF_O + (size_t)O_T * KS_H * 32)
#define OFF_DN (OFF_GU + (size_t)GU_T * KS_H * 32)
#define LAYER_U4 (OFF_DN + (size_t)DN_T * KS_I * 32)

#define NSLOT 64
#define MAXPG 68                         // pages per sequence: 34816 / 512
#define PG_TILES 32                      // a page = 512 positions = 32 position tiles
#define PG_U4 (PG_TILES * 8 * 32)        // uint4 per (page, layer, kv head) for K; the same again for V
#define PAGE_BYTES ((size_t)NL * 8 * 2 * PG_U4 * 16)
#define SEG 128                          // attention segment (positions)

#include "dsched.h"
// The plan of a pass (ints): written by the device scheduler (ds_step in dsched.h) or, for the offline contract and
// the tests, by the host. Layout in dsched.h.
#define TCAP DS_TCAP                     // columns per pass at most
#define SELCAP DS_SELCAP                 // columns per pass that need logits / a token
#define MAXU DS_MAXU                     // attention units per pass (a unit: up to 16 consecutive positions of one sequence)
#define MAXBT DS_MAXBT                   // attention tasks per pass
#ifndef SS_LOG
#define SS_LOG 5
#endif
#define SS_SEGS (1 << SS_LOG)            // segments per super-segment
#define SS_SHIFT (7 + SS_LOG)            // log2 of the positions of a super-segment
#define PARTCAP (SS_LOG >= 5 ? 2048 : SS_LOG == 4 ? 4096 : 8192)   // (unit, super-segment) parts kept in the scratch per pass
#define PL_SLOT DS_PL_SLOT               // [TCAP] slot | (last column of its run ? LASTBIT : 0) | (column - first column of its attention unit) << 17
#define PL_POS DS_PL_POS                 // [TCAP] position of the column in its sequence
#define PL_TOK DS_PL_TOK                 // [TCAP] -1: the slot's last generated token; else the token id (host plan) or its index in the prompt arena (device plan)
#define PL_UNIT DS_PL_UNIT               // [MAXU][4] attention units: first column, columns (<= 16), first part in the scratch (-1: one super-segment, no fold), super-segments
#define PL_BT DS_PL_BT                   // [MAXBT] attention tasks, those of two units first: first unit | super-segment << 16 | kv head << 22
#define PL_FOLD DS_PL_FOLD               // [MAXU] units whose super-segments have to be folded
#define PL_SEL DS_PL_SEL                 // [SELCAP][2]: column, slot (-1: none)
#define PL_PT DS_PL_PT                   // [NSLOT][MAXPG] page ids
#define PL_CU DS_PL_CU                   // [TCAP] attention unit of the column
#define PL_HDR DS_PL_HDR                 // [16] 0 columns, 1 tasks of two units, 2 tasks of one unit, 3 units to fold, 4 selected columns, 5 pages freed,
                                         //      6 tasks of one unit that are decode rows (one position): listed first among the single ones
#define PL_INTS DS_PL_INTS
// a kernel argument >= 0 is the value; -1 - i means: element i of the plan header (the host does not know it)
#define HV(arg, plan) ((arg) >= 0 ? (arg) : (plan)[PL_HDR + (-(arg) - 1)])
// position offset of a pass: the argument, or (negative) the cell the generate() loop on the device advances
#define PV(arg, D) ((arg) >= 0 ? (arg) : __ldcg((D).cell))
#define LASTBIT 0x10000

__device__ __forceinline__ unsigned short f2bf(float f) {
  unsigned u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}
__device__ __forceinline__ float bf2f(unsigned v) { return __uint_as_float(v << 16); }
#ifndef TRACK_A
#define TRACK_A 0   // see the comment above SC2
#endif
#ifndef TA_NOR
#define TA_NOR 0     // timing experiments only: track A without the roundings
#endif
__device__ __forceinline__ float ta_r(float x) { return TRACK_A && !TA_NOR ? bf2f(f2bf(x)) : x; }   // an operator's output on track A
// Layout of every GEMM operand in global memory, weights [M rows][K] and activations [columns][K] alike
// ("rows" below are M rows or token columns): [k / 64][row / 8][(k % 64) / 8][row % 8][k % 8] in bf16 elements.
// One (K chunk of 64, 8 rows) is 1 KB; inside it the 8 rows x 8 K-elements of each K-octet are 128 contiguous
// bytes. A K chunk of consecutive row groups is therefore one contiguous byte range that the copy engine moves
// into shared memory as it is, and there it is the canonical K-major operand of wgmma (no swizzle: leading
// offset 128 B between the two K-octets of an instruction, stride 1 KB between groups of 8 rows).
__device__ __forceinline__ size_t xidx(int k, int row, int groups) {
  return ((((size_t)(k >> 6) * groups + (row >> 3)) * 8 + ((k >> 3) & 7)) * 8 + (row & 7)) * 8 + (k & 7);
}
#define NG_X (TCAP / 8)      // column groups of the activation buffers
#define NG_LM (SELCAP / 8)   // ... of the lm_head input
__device__ __forceinline__ float warp_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}
#define NINF (-__int_as_float(0x7f800000))
// TRACK_A = 1: the general engines' number formats and data flow. Every operator's output is rounded to BF16 (the
// residual stream after each add, q/k/v, gate, up, SiLU, their product, the logits), RMSNorm is x * rsqrt(mean x^2)
// in f32, rounded to BF16, times the weight, rounded, BEFORE the matvec, RoPE works on BF16 values with BF16
// cos / sin, q is stored unscaled and the scores are scaled by 1 / sqrt(128) in f32. What stays: the softmax is
// taken in segments against a running maximum, and its weights are BF16 when they multiply V (tensor-core operands).
// TRACK_A = 0 (default): f32 between operators, the RMSNorm scalar commuted behind the matvec, q stored pre-scaled.
#define SC2 ((float)(0.08838834764831845 * 1.4426950408889634))   // log2(e) / sqrt(128)

struct Dev {
  const unsigned short* W; const unsigned short* embed; const float* norm_w; const float* qk_norm_w; const float* rope;
  uint4* kv;                 // page pool: [page][layer][kv head][K | V][PG_U4]
  const int* plan;           // device copy of the pass plan
  float* x;                  // [TCAP][H] residual stream
  unsigned short* xfrag;     // [TCAP][H] bf16(w * x) in the operand layout: input of the next norm + matvec
  float* ssq;                // [TCAP][256]
  float* rn;                 // [TCAP]
  float* qkv;                // [TCAP][6144]
  unsigned short* qfrag;     // [MAXU][8 kv heads][16 KB] q after norm, RoPE and score scale, bf16 (see qidx)
  unsigned short* afrag;     // [TCAP][H] attention output, operand layout
  unsigned short* mfrag;     // [TCAP][INNER] silu(gate) * up, operand layout
  unsigned short* lmfrag;    // [SELCAP][H] final-norm input of the selected columns, operand layout (NG_LM groups)
  float* lm_rn;              // [SELCAP]
  float* bm_val; int* bm_idx;// [VOCAB / 16][SELCAP] best logit and row per (row block, warp) of the lm_head: VOCAB / 16 of them with 64-row blocks
  int* last_tok;             // [NSLOT] last generated token of a slot
  int* sel_tok;              // [SELCAP] tokens of this pass
  int* cell;                 // [4] generate() loop on the device: position offset of the pass, step, end step
  float4* att_part;          // [PARTCAP][8 kv heads][17][128]: fold state of a (unit, kv head, super-segment)
  int* ctr;                  // [3 NL] task counters of the three attention kernels of a layer (zeroed by k_embed)
};

// K, V and q are stored as operands of the warpgroup MMA (8 x 8 bf16 per 128 bytes, see wg_desc):
//   K of a page (512 positions) of one (layer, kv head): [pos / 8][dim / 8][pos % 8][dim % 8]; a segment of 128
//     positions is 32 KB contiguous = the B operand (positions x dims) of the score product;
//   V: [pos / 128][dim / 8][(pos % 128) / 8][dim % 8][pos % 8]; a segment is 32 KB contiguous = the B operand
//     (dims x positions) of the value product;
//   q of a unit (16 positions x the 4 heads of a kv head = 64 rows, row = 4 * position + head): [row / 8][dim / 8][row % 8][dim % 8].
__device__ __forceinline__ size_t kidx(int pp, int d) { return ((((size_t)(pp >> 3) * 16 + (d >> 3)) * 8 + (pp & 7)) * 8 + (d & 7)); }
__device__ __forceinline__ size_t vidx(int pp, int d) { return (((((size_t)(pp >> 7) * 16 + (d >> 3)) * 16 + ((pp >> 3) & 15)) * 8 + (d & 7)) * 8 + (pp & 7)); }
__device__ __forceinline__ size_t qidx(int row, int d) { return ((((size_t)(row >> 3) * 16 + (d >> 3)) * 8 + (row & 7)) * 8 + (d & 7)); }
__device__ __forceinline__ unsigned short* page_k(const Dev& D, int slot, int pos, int layer, int g) {
  const int page = D.plan[PL_PT + slot * MAXPG + (pos >> 9)];
  return (unsigned short*)(D.kv + ((((size_t)page * NL + layer) * 8 + g) * 2) * PG_U4);
}

// One of the 48 vectors of 128 of a column after the q/k/v matvec (32 q heads, 8 k heads, 8 v heads), by one warp
// (lane l holds dims 4 l .. 4 l + 3 in x). q, k: RMSNorm over the head, RoPE at the column's position.
// q -> qfrag; k, v -> the sequence's KV pages.
__device__ __forceinline__ void prep_vec(const Dev& D, int layer, int padd, int col, int vec, float4 x, int lane) {
  const int sl = D.plan[PL_SLOT + col], slot = sl & 0xffff, pos = D.plan[PL_POS + col] + padd;
  float xe[4] = {x.x, x.y, x.z, x.w};
  unsigned short ob[4];
  if (vec < 40) {
    const float4 wn = ((const float4*)(D.qk_norm_w + (layer * 2 + (vec < 32 ? 0 : 1)) * 128))[lane];
    const float4 cs = ((const float4*)(D.rope + (size_t)pos * 128))[lane & 15];
    const float4 sn = ((const float4*)(D.rope + (size_t)pos * 128 + 64))[lane & 15];
    const float ss = warp_sum(__fadd_rn(__fmaf_rn(x.x, x.x, __fmul_rn(x.y, x.y)), __fmaf_rn(x.z, x.z, __fmul_rn(x.w, x.w))));
    const float r = rsqrtf(__fmaf_rn(ss, 1.0f / 128, 1e-6f));
    const float we[4] = {wn.x, wn.y, wn.z, wn.w}, ce[4] = {cs.x, cs.y, cs.z, cs.w}, se[4] = {sn.x, sn.y, sn.z, sn.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float yv = ta_r(__fmul_rn(we[i], ta_r(__fmul_rn(xe[i], r))));
      const float other = __shfl_xor_sync(0xffffffffu, yv, 16);
      float rot;
      if (TRACK_A) {
        const float c = ta_r(ce[i]), sn = ta_r(se[i]);
        rot = lane < 16 ? __fsub_rn(ta_r(__fmul_rn(yv, c)), ta_r(__fmul_rn(other, sn))) : __fadd_rn(ta_r(__fmul_rn(yv, c)), ta_r(__fmul_rn(other, sn)));
      } else rot = lane < 16 ? __fmaf_rn(yv, ce[i], -__fmul_rn(other, se[i])) : __fmaf_rn(yv, ce[i], __fmul_rn(other, se[i]));
      ob[i] = f2bf(vec < 32 && !TRACK_A ? __fmul_rn(rot, SC2) : rot);   // track B: q carries the score scale log2(e) / sqrt(128)
    }
  } else {
#pragma unroll
    for (int i = 0; i < 4; i++) ob[i] = f2bf(xe[i]);
  }
  if (vec < 32) {
    const int g = vec >> 2, h = vec & 3, iq = (sl >> 17) & 15;
    unsigned short* Q = D.qfrag + ((size_t)D.plan[PL_CU + col] * 8 + g) * 8192;
    // the 4 dims of a lane are 8 contiguous bytes in the operand layout: one store instead of four
    *(unsigned long long*)(Q + qidx(4 * iq + h, 4 * lane)) = (unsigned long long)ob[0] | ((unsigned long long)ob[1] << 16) | ((unsigned long long)ob[2] << 32) | ((unsigned long long)ob[3] << 48);
  } else if (vec < 40) {
    unsigned short* Kc = page_k(D, slot, pos, layer, vec - 32);
    *(unsigned long long*)(Kc + kidx(pos & 511, 4 * lane)) = (unsigned long long)ob[0] | ((unsigned long long)ob[1] << 16) | ((unsigned long long)ob[2] << 32) | ((unsigned long long)ob[3] << 48);
  } else {
    unsigned short* Vc = page_k(D, slot, pos, layer, vec - 40) + (size_t)PG_U4 * 8;
    // The last column of a run also clears the rest of its group of 16 positions: they enter the value product
    // of that group with weight 0 and must be finite. A run of one column (a decode row) that does not start a
    // group needs no clearing: the run that wrote the position before it cleared the same group to its end.
    // (The clearing is 2-byte stores to memory that is not in the cache: 0.9 us per position; a decode step
    // spent 3 to 17 us per layer here depending on the position.)
    const int p0 = pos & 511;
    const bool single = (sl & LASTBIT) && (col == 0 || (D.plan[PL_SLOT + col - 1] & LASTBIT));
    const int pend = ((sl & LASTBIT) && !(single && (p0 & 15))) ? (p0 | 15) : p0;
    if (single && pend != p0) {
      // a decode row at the start of a group: the 16 positions of a dim are two rows of 8, one 16-byte store each
#pragma unroll
      for (int i = 0; i < 4; i++) {
        *(uint4*)(Vc + vidx(p0, 4 * lane + i)) = make_uint4(ob[i], 0u, 0u, 0u);
        *(uint4*)(Vc + vidx(p0 + 8, 4 * lane + i)) = make_uint4(0u, 0u, 0u, 0u);
      }
    } else {
      for (int pp = p0; pp <= pend; pp++) {
#pragma unroll
        for (int i = 0; i < 4; i++) Vc[vidx(pp, 4 * lane + i)] = pp == p0 ? ob[i] : (unsigned short)0;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// GEMM: Y[M x T] = W[M x K] X[K x T] on the Hopper path.
//   * wgmma: one instruction of a warpgroup (4 warps) multiplies 64 rows x NB columns x 16 K, both operands read
//     from shared memory through descriptors, accumulators in registers (NB / 2 per thread);
//   * cp.async.bulk: the copy engine moves a contiguous byte range global -> shared memory; no warp loads
//     anything, so the copy overlaps the multiply completely;
//   * mbarriers: a stage (K chunk of 64) is "full" when its bytes have arrived, "empty" when both consumer
//     warpgroups are done with it.
// A block = 2 consumer warpgroups + producer warps (one thread each). It computes units of 64 NT rows x 2 NB columns
// (warpgroup w: columns w * NB ..) with stages of (A: 64 NT rows x 64 K, B: 2 NB columns x 64 K) in shared memory;
// block b takes units b, b + grid, ... in row-major order. The shape (NB, NT) is chosen per matrix and number of
// columns (gemm_shape() below).
// Why this path (all measured here, T = 1024, 24576 x 4096): my m16n8k16 kernels with warp loads into shared
// memory reached 255 / 334 / 292 / 301 TFLOPS in four designs. tests/copy_rate.cu: an SM copies global -> shared
// memory at 10 KB/us per loading warp up to ~55 KB/us; tests/roles_rate.cu, tests/l1_stage.cu: while it does,
// the multiplies stall (same warps, other warps, or L1 as the buffer: always copy time + multiply time).
// The vendor GEMM runs at 790-820 TFLOPS on these shapes; its launch (132 blocks x 384 threads, 168 registers,
// 213 KB shared memory, "64x4" in the kernel name) pointed to wgmma + the copy engine. This kernel: ~750 TFLOPS
// at 1024+ columns and 2.7-3.0 TB/s of weights at T <= 64.
// What bounds it (tests/wg_scale.cu, tests/kv_stride.cu, notes/journal.md): the tensor cores of an SM are
// saturated by two warpgroups at ~965 TFLOPS over the GPU; the kernel is below that because a block holds
// 128 x 256 accumulators (its registers), so the weights are read from HBM once per 256 columns (4 x 15 GB per
// 1024-column pass = 20 ms at 3 TB/s), and wider blocks would have to copy more activations per stage than the
// copy engine of an SM carries (64-127 KB/us).
// Instruction forms after CUTLASS (cute/arch/mma_sm90_gmma.hpp, copy_sm90_tma.hpp, cutlass/arch/barrier.h).
// ---------------------------------------------------------------------------------------------------------
#include "wg_gen.h"
#ifndef WG_NS1
#define WG_NS1 12                            // stages of the 64-row variant (weights straight from HBM: deeper look-ahead)
#endif
#ifndef WG_NSQ
#define WG_NSQ 10                            // stages of the 8 x 2 shape (q/k/v with preparation, 48 blocks): 8: +10-25 us per step, 12: +80 us
#endif
#define WG_NS(NB, NT) ((NT) != 1 ? ((NB) <= 8 ? WG_NSQ : 4) : (NB) > 64 ? 4 : (NB) <= 16 ? WG_NS1 : 8)   // stages in shared memory
#ifndef WG_PIPE
#define WG_PIPE 1
#endif
#define WG_KC 64                               // K per stage: 4 wgmma instructions per 64-row tile
#define WG_A_TILE (64 * WG_KC * 2)              // one 64-row tile of a stage: 8 KB
// Producer threads: one per warp behind the two consumer warpgroups, the stages dealt to them in turn. One thread
// issuing every copy request was the limit of the small shapes (a request costs ~40 ns of its time, a stage ~0.2
// us in all): with 4, the GEMMs of a decode step take 5.09 ms instead of 5.30, of a 256-column pass 7.10 instead
// of 7.34 (2 producers: 5.12 / 7.20). The blocks of 128 rows and more (4 stages) keep one: with 2 or 4 they are
// 0-3% slower.
#ifndef EPI_XNC
#define EPI_XNC 0          // 1: the residual through the read-only path (no change measured)
#endif
#ifndef EPI_XPF
#define EPI_XPF 0          // residual add: its values into L2 this many stages before the unit end (4 / 8 / 16: no change)
#endif
#ifndef EPI_XINIT
#define EPI_XINIT 0        // 1: (track B) the residual add's accumulators start at the residual: other bits; prompt passes of
#endif                     // > 512 columns 2-3.5% faster, but 64 x 1 / 32 x 1 get serialized (C7515): 65-256 columns +5%
#ifndef EPI_RNU
#define EPI_RNU 1
#endif
#ifndef EPI_XST_NB
#define EPI_XST_NB 64      // ... for blocks of at least this many columns per warpgroup
#endif
#ifndef EPI_XST
#define EPI_XST 1          // residual add: the last unit's residual values staged in shared memory by cp.async
#endif
#ifndef EPI_ABL
#define EPI_ABL 0          // timing experiments only: 1 no 1/rms loads in the epilogues, 2 no residual loads, 8 no GU / ADD epilogue (NOT a measure: unused products let the compiler drop the MMAs)
#endif
#ifndef WG_NEXT_OFF
#define WG_NEXT_OFF 0      // first stage of the next kernel prefetched into L2, and how many (a multiple of 4)
#endif
#ifndef WG_NEXT_N
#define WG_NEXT_N 8
#endif
#ifndef WG_NEXT
#define WG_NEXT 1
#endif
#ifndef WG_NPROD
#define WG_NPROD 4          // for 64-row blocks
#endif
#ifndef WG_NPROD2
#define WG_NPROD2 1         // for the others
#endif
#define WG_THREADS (256 + 32 * WG_NPROD)
#define WG_SMEM(NB, NT) (WG_NS(NB, NT) * ((NT) * WG_A_TILE + 2 * (NB) * WG_KC * 2) + 16 * WG_NS(NB, NT) + 256 + ((NB) == 8 && (NT) == 2 ? 8192 : 2048))   // stages, barriers, 64 x 1/rms, EP_QKVP: 16 columns x 128 f32
__device__ __forceinline__ uint32_t sm32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
// Programmatic dependent launch (PDL): consecutive kernels of a graph are joined by programmatic edges. Every
// kernel calls pdl_wait() before it touches memory that an earlier kernel writes or reads (it returns when the
// kernel before has completed and its writes are visible; transitively all before it). A kernel's successor is
// launched when the kernel completes, except after a GEMM: its producers let the successor launch once they have
// asked for every copy (PDL_TRIG 3), and a per-shape GEMM (k_gemm1) requests the weights of its first PDL_EARLY
// stages before it waits (they do not depend on the kernel before), so the next matrix starts streaming while the
// last stages of this one are multiplied. Decode step, same run (tests/ab_defs.sh): no PDL 5.68 / 5.84 ms;
// programmatic edges only 5.65 / 5.80; + producers' trigger, early weights after the barrier set-up: 5.57 / 5.71;
// with the barriers set up before the wait (so that the early copies really go out before it): 3 or 6 stages
// 5.46 / 5.59 (0: 5.53, 12: 5.51). Every kernel triggering at its start: up to 0.2 ms slower (B=8).
// tests/stream_pdl.cu: the same on bare weight streams, 4730 -> 4583 us.
#ifndef PDL
#define PDL 1
#endif
#ifndef PDL_TRIG
#define PDL_TRIG 4        // 0: at completion only; 2: every kernel at its start; 3: the GEMM producers after their last copy request; 4: 3 and the other kernels at their start (3 -> 4: -20 us per step)
#endif
#ifndef PDL_EARLY
#define PDL_EARLY 3
#endif
__device__ __forceinline__ void pdl_wait(const bool gemm = false) {
#if PDL
  if (PDL_TRIG == 2 || (PDL_TRIG == 4 && !gemm)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t count) { asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" :: "r"(count), "r"(sm32(b)) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(sm32(b)) : "memory"); }
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;" :: "r"(bytes), "r"(sm32(b)) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
  asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra DONE;\nbra LAB_WAIT;\nDONE:\n}" :: "r"(sm32(b)), "r"(phase) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* smem_dst, const void* gsrc, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" :: "r"(sm32(smem_dst)), "l"(gsrc), "r"(bytes), "r"(sm32(b)) : "memory");
}
__device__ __forceinline__ uint64_t wg_desc(const void* smem_ptr) {
  return (uint64_t)((sm32(smem_ptr) >> 4) & 0x3fff) | ((uint64_t)8 << 16) | ((uint64_t)64 << 32);
}
template <int NB> struct WG;
template <> struct WG<8> { static __device__ __forceinline__ void mma(float (&d)[4], uint64_t da, uint64_t db) { WGMMA_N8(d, da, db); }
                          static __device__ __forceinline__ void touch(float (&d)[4]) { WG_TOUCH_N8(d); } };
template <> struct WG<16> { static __device__ __forceinline__ void mma(float (&d)[8], uint64_t da, uint64_t db) { WGMMA_N16(d, da, db); }
                          static __device__ __forceinline__ void touch(float (&d)[8]) { WG_TOUCH_N16(d); } };
template <> struct WG<32> { static __device__ __forceinline__ void mma(float (&d)[16], uint64_t da, uint64_t db) { WGMMA_N32(d, da, db); }
                          static __device__ __forceinline__ void touch(float (&d)[16]) { WG_TOUCH_N32(d); } };
template <> struct WG<64> { static __device__ __forceinline__ void mma(float (&d)[32], uint64_t da, uint64_t db) { WGMMA_N64(d, da, db); }
                          static __device__ __forceinline__ void touch(float (&d)[32]) { WG_TOUCH_N64(d); } };
template <> struct WG<128> { static __device__ __forceinline__ void mma(float (&d)[64], uint64_t da, uint64_t db) { WGMMA_N128(d, da, db); }
                          static __device__ __forceinline__ void touch(float (&d)[64]) { WG_TOUCH_N128(d); } };

enum { EP_QKV = 0, EP_ADD = 1, EP_GU = 2, EP_LM = 3, EP_LMARG = 4, EP_RAW = 5, EP_QKVP = 6 };   // QKVP: q/k/v with the vectors' preparation in the epilogue

struct GemmArgs {
  const unsigned short* W;   // weights [M][K] in the operand layout (see xidx)
  const unsigned short* X;   // activations [columns][K] in the operand layout with NG column groups
  int M, K, T, NG;
  const float* rn;        // [T] 1/rms of the input norm (QKV, GU, LM)
  float* out;             // QKV: [T][6144]; LM: logits [T][VOCAB]; RAW: [T][M]
  float* x;               // ADD: residual stream [T][4096]
  const float* aux;       // ADD: weights of the next RMSNorm [4096]
  unsigned short* frag;   // ADD: next norm's input (K = 4096); GU: silu(gate)*up (K = 12288); operand layout, NG_X groups
  float* ssq;             // ADD: [T][256] per-16-row sums of x^2
  float* bm_val; int* bm_idx;   // LMARG: [M / 32][T] best logit and its row per (row block, warp)
  const int* plan;        // T < 0: the number of columns is element -T - 1 of the plan header
  int shape;              // block shape (gemm_shape()), or -1: chosen on the device from the number of columns
  const unsigned short* nextW; int nextM;   // the matrix of the next GEMM kernel (prefetch of its first stages), or null
  const float* ssq_in;    // QKV, GU: [T][256] sums of x^2 of the input norm: up to 64 columns the kernel computes 1/rms itself
  Dev dev; int layer, pos_add, prep;   // QKV: up to 16 columns (prep != 0) the kernel prepares q, K and V itself (EP_QKVP)
  int ta_in;              // track A (TA_PB): QKV / GU normalize their input themselves (x: residual, aux: norm weights)
};

// Block shapes. A block computes 64 NT rows x 2 NB columns per unit (NB columns per consumer warpgroup); the units of
// a matrix (row blocks x column blocks) are dealt to the blocks of the grid, one block per SM. The arithmetic of
// an output element does not depend on the shape (tests/gemm_shapes.py compares the bits), so the shape is free:
//  - up to 64 columns the kernel is bound by streaming the weights: 64-row blocks, so that 64 SMs stream the
//    4096-row matrices (an SM takes 40-60 KB/us from the copy engine), and the smallest wgmma that covers T;
//  - beyond, by how full the SMs are: a 4096-row matrix at 256 columns has 32 units of 128 x 256 for 132 SMs. A stage
//    (64 of K) of a block takes about 0.19 us + 0.072 us per 64 x 64 of the block's area, so the time of a matrix
//    is about (units per SM, rounded up) x stage time x K / 64: gemm_shape() takes the shape with the least.
//    Measured over all shapes per matrix and T (tests/gemm_shapes.py): the GEMMs of a pass of 256 columns
//    11.1 -> 7.2 ms, 512: 13.2 -> 10.8, 768: 18.4 -> 16.5, 1024: 21.2 -> 19.8.
#define N_SHAPES 8
#ifndef SHAPE8_T
#define SHAPE8_T 16                           // up to this many columns: 8 columns per warpgroup
#endif
__host__ __device__ __forceinline__ void shape_dims(int s, int& nb, int& nt, float& stage_us) {
  switch (s) {
    case 0: nb = 16; nt = 1; stage_us = 0.22f; break;
    case 1: nb = 32; nt = 1; stage_us = 0.25f; break;
    case 2: nb = 64; nt = 1; stage_us = 0.31f; break;
    case 3: nb = 64; nt = 2; stage_us = 0.49f; break;
    case 4: nb = 64; nt = 3; stage_us = 0.60f; break;
    case 5: nb = 64; nt = 4; stage_us = 0.77f; break;
    case 7: nb = 8; nt = 1; stage_us = 0.20f; break;
    default: nb = 128; nt = 2; stage_us = 0.76f; break;
  }
}
__host__ __device__ __forceinline__ int gemm_shape(int M, int T, int nsm) {
  if (T <= SHAPE8_T) return 7;
  if (T <= 32) return 0;
  if (T <= 64) return 1;
  int best = 6; float best_c = 1e30f;
  for (int s = 1; s <= 6; s++) {
    int nb, nt; float us;
    shape_dims(s, nb, nt, us);
    if (M % (64 * nt)) continue;
    const int units = (M / (64 * nt)) * ((T + 2 * nb - 1) / (2 * nb));
    const float c = (float)((units + nsm - 1) / nsm) * us;
    if (c < best_c) { best_c = c; best = s; }
  }
  return best;
}

// Track A, offline steps of up to 8 columns (PB): the input of QKV / GU is normalized once per block, for the whole
// K (4096), into a resident region of shared memory (warpgroup 0's 8 columns; warpgroup 1's columns are beyond T and
// use the same region), and the stages carry only weights: no k_rnorm kernel (73 per step, ~3.8 us each).
#ifndef TA_PB
#define TA_PB 1            // track A, offline steps up to TA_PB_T columns: QKV / GU normalize their input (no k_rnorm)
#endif
#ifndef TA_PB_T
#define TA_PB_T 1          // (8 columns: B=8 5.84 -> 6.06 ms, the 16 vectors per thread of the normalization in a row)
#endif
#define WG_NS_PB(NT) ((NT) == 2 ? 9 : WG_NS1)
#define WG_PB_BYTES (8 * 2 * 4096)
#define WG_SMEM_PB(NT) (WG_NS_PB(NT) * ((NT) * WG_A_TILE) + WG_PB_BYTES + 16 * WG_NS_PB(NT) + 256 + ((NT) == 2 ? 8192 : 2048))
template <int EP, int NB, int NT, int PB = 0>
__device__ __forceinline__ void gemm_body(const GemmArgs& a, const int T, unsigned char* smem, const int early = 0) {
  enum { A_STAGE = NT * WG_A_TILE, B_STAGE = PB ? 0 : 2 * NB * WG_KC * 2, STAGE = A_STAGE + B_STAGE, WG_STAGES = PB ? WG_NS_PB(NT) : WG_NS(NB, NT),
         PB_BYTES = PB ? WG_PB_BYTES : 0 };
  static_assert(!PB || (TRACK_A && NB == 8), "PB: track A, 8-column shapes");
  unsigned char* const Bp = smem + WG_STAGES * STAGE;   // PB: [64 chunks][8 x 8][8 columns][8] bf16
  uint64_t* full = (uint64_t*)(smem + WG_STAGES * STAGE + PB_BYTES);
  uint64_t* empty = full + WG_STAGES;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wg = warp >> 2;
  // Up to 64 columns the 1/rms of the input norm is computed here instead of in a kernel of its own (two kernels
  // per layer less, ~1.4 us each): a consumer warp per column, the same sum as k_rnorm, before the block starts.
  float* rn_s = (float*)(empty + WG_STAGES);
  const bool rn_here = (!TRACK_A || PB) && ((EP == EP_QKV || EP == EP_GU) && NT == 1 || EP == EP_QKVP) && NB <= 32 && a.ssq_in != nullptr;
  if (PB && (!rn_here || a.K != 4096 || T > 8)) __trap();
  float* xs = rn_s + 64;           // EP_QKVP: [16 columns][128] the block's vector of every column
  // the 1/rms of a unit's columns, read at the unit's start (two buffers, by unit): the epilogue would otherwise
  // wait for one round trip per column group (16 in a row for 128 columns per warpgroup: ~8% of a prompt pass)
  float* rn_u = rn_s + 64;         // [2][256] (not with EP_QKVP: xs)
  constexpr bool RN_U = !TRACK_A && EPI_RNU && (EP == EP_QKV || EP == EP_GU || EP == EP_LM || EP == EP_LMARG) && NB > 32;
  constexpr bool XINIT = !TRACK_A && EPI_XINIT && EP == EP_ADD;
  // A block without a unit leaves at once (block 0 has one and waits: the kernel cannot end before the one before)
  if ((int)blockIdx.x >= ((T + 2 * NB - 1) / (2 * NB)) * (a.M / (64 * NT)) && T > 0) return;
  // The barriers first, then the wait for the kernel before: with `early` the producers ask for the first weight
  // stages while that kernel still runs (the consumers wait, compute the 1/rms and meet at a barrier of their own).
  if (tid == 0) {
    for (int s = 0; s < WG_STAGES; s++) { mbar_init(full + s, 1); mbar_init(empty + s, 2); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (!early) pdl_wait(true);
  __syncthreads();
  if (wg < 2) {
    if (early) pdl_wait(true);
    if (rn_here) {
      for (int col = warp; col < T; col += 8) {
        const float4* q = (const float4*)(a.ssq_in + (size_t)col * 256) + lane * 2;
        float4 u = __ldcg(q), v = __ldcg(q + 1);
        float sm = ((u.x + u.y) + (u.z + u.w)) + ((v.x + v.y) + (v.z + v.w));
        sm = warp_sum(sm);
        if (lane == 0) rn_s[col] = rsqrtf(__fmaf_rn(sm, 1.0f / H, 1e-6f));
      }
    }
    if (early) asm volatile("bar.sync 1, 256;" ::: "memory");   // rn_s complete for both consumer warpgroups
    if constexpr (PB) {
      // the normalized input bf16(w * bf16(x * 1/rms)) of the T columns, as k_rnorm writes it, in the stage layout
      const float4* xq = (const float4*)a.x; const float4* wq = (const float4*)a.aux;
      for (int v = tid; v < T * 512; v += 256) {
        const int col = v >> 9, vk = v & 511;
        const float rn = rn_s[col];
        const float4 x0 = __ldcg(xq + (size_t)col * (H / 4) + 2 * vk), x1 = __ldcg(xq + (size_t)col * (H / 4) + 2 * vk + 1);
        const float4 w0 = wq[2 * vk], w1 = wq[2 * vk + 1];
#define PB_E(xv, wv) ((unsigned)f2bf((wv) * ta_r((xv) * rn)))
        *(uint4*)(Bp + (size_t)(((vk >> 3) * 8 + (vk & 7)) * 8 + col) * 16) =
            make_uint4(PB_E(x0.x, w0.x) | (PB_E(x0.y, w0.y) << 16), PB_E(x0.z, w0.z) | (PB_E(x0.w, w0.w) << 16),
                       PB_E(x1.x, w1.x) | (PB_E(x1.y, w1.y) << 16), PB_E(x1.z, w1.z) | (PB_E(x1.w, w1.w) << 16));
#undef PB_E
      }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      asm volatile("bar.sync 1, 256;" ::: "memory");
    }
  }
  const int n_cb = (T + 2 * NB - 1) / (2 * NB), n_rb = a.M / (64 * NT), n_units = n_rb * n_cb, nchunk = a.K / WG_KC;
  const int mg = a.M / 8;
  unsigned gc = 0;                               // chunks so far: the producer and the consumers count alike
  if (wg == 2) {
    // ---- producers: a thread asks the copy engine for A and B of its chunks, up to WG_STAGES ahead in all
    if (lane != 0) return;
    enum { NPROD = (NT == 1 || NB <= 8) ? WG_NPROD : WG_NPROD2 };     // K / 64 is a multiple of it: producer p has chunks p, p + NPROD, ..
    const int prod = warp - 8;
    if (prod >= NPROD) return;
    const int ne = early < WG_STAGES ? early : WG_STAGES;
    if (early) {
      if (PDL_TRIG == 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
      if ((int)blockIdx.x < n_units) {
        const int rb = (int)blockIdx.x / n_cb;
        for (int c = prod; c < nchunk && c < ne; c += NPROD) {
          mbar_expect(full + c, STAGE);
          bulk_g2s(smem + c * STAGE, a.W + ((size_t)c * mg + rb * 8 * NT) * 512, A_STAGE, full + c);
        }
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    for (int u = blockIdx.x; u < n_units; u += gridDim.x, gc += nchunk) {
      const int rb = u / n_cb, cb = u - rb * n_cb;
      for (int c = prod; c < nchunk; c += NPROD) {
        const unsigned g2 = gc + c;
        const int s = g2 % WG_STAGES; const unsigned r = g2 / WG_STAGES;
        if (r >= 1) mbar_wait(empty + s, (r - 1) & 1);
        if (!(g2 < (unsigned)ne)) {
          mbar_expect(full + s, STAGE);
          bulk_g2s(smem + s * STAGE, a.W + ((size_t)c * mg + rb * 8 * NT) * 512, A_STAGE, full + s);
        }
        if (!PB) bulk_g2s(smem + s * STAGE + A_STAGE, a.X + ((size_t)c * a.NG + cb * (NB / 4)) * 512, B_STAGE, full + s);
      }
    }
    if (PDL_TRIG >= 3) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    return;
  }
  // ---- consumers: warpgroup wg computes rows 0..127 x columns wg * NB .. + NB - 1 of the block
  int ui = 0;
  for (int u = blockIdx.x; u < n_units; u += gridDim.x, ui++) {
    const int rb = u / n_cb, cb = u - rb * n_cb;
    if (RN_U && tid < 2 * NB) { const int n = cb * 2 * NB + tid; rn_u[(ui & 1) * 256 + tid] = n < T ? __ldcg(a.rn + n) : 0.f; }
    float acc[NT][NB / 2];
    if constexpr (XINIT) {
      // the residual stream is where the products are summed into (track B): its values are requested at the unit's
      // start, all at once, instead of in the epilogue one column group after another
      const int g = lane >> 2, t = lane & 3, t2 = warp & 3;
#pragma unroll
      for (int i = 0; i < NT; i++)
#pragma unroll
        for (int j = 0; j < NB / 8; j++) {
          const int n0 = cb * 2 * NB + wg * NB + 8 * j + 2 * t, row = (rb * (4 * NT) + 4 * i + t2) * 16 + g;
          const float* p0 = a.x + (size_t)n0 * H + row;
          acc[i][4 * j + 0] = n0 < T ? __ldcg(p0) : 0.f;
          acc[i][4 * j + 2] = n0 < T ? __ldcg(p0 + 8) : 0.f;
          acc[i][4 * j + 1] = n0 + 1 < T ? __ldcg(p0 + H) : 0.f;
          acc[i][4 * j + 3] = n0 + 1 < T ? __ldcg(p0 + H + 8) : 0.f;
        }
#pragma unroll
      for (int i = 0; i < NT; i++) WG<NB>::touch(acc[i]);    // all defined here, before the first MMA
    } else {
#pragma unroll
      for (int i = 0; i < NT; i++)
#pragma unroll
        for (int j = 0; j < NB / 2; j++) acc[i][j] = 0.f;
    }
    for (int c = 0; c < nchunk; c++, gc++) {
      const int s = gc % WG_STAGES; const unsigned r = gc / WG_STAGES;
      if (EP == EP_ADD && EPI_XPF > 0 && NB >= 32 && c == (nchunk > EPI_XPF ? nchunk - EPI_XPF : 0)) {
        // the residual values of the unit into L2 a few stages before the epilogue needs them (they come from HBM
        // otherwise, one round trip per column group): 64 NT rows x 4 B per column, 2 NB columns
        constexpr int LPC = NT * 64 * 4 / 128;        // 128-byte lines per column
        for (int q = tid; q < 2 * NB * LPC; q += 256) {
          const int n = cb * 2 * NB + q / LPC;
          if (n < T) asm volatile("prefetch.global.L2 [%0];" :: "l"(a.x + (size_t)n * H + rb * 64 * NT + (q % LPC) * 32));
        }
      }
      mbar_wait(full + s, r & 1);
      const unsigned char* As = smem + s * STAGE;
      const unsigned char* Bs = PB ? Bp + c * (NB * WG_KC * 2) : As + A_STAGE + wg * (NB * WG_KC * 2);
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int j = 0; j < WG_KC / 16; j++) {
        const uint64_t db = wg_desc(Bs + j * 256);
#pragma unroll
        for (int i = 0; i < NT; i++) WG<NB>::mma(acc[i], wg_desc(As + i * WG_A_TILE + j * 256), db);
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
#if WG_PIPE
      // the instructions of this stage stay in flight while the next stage is issued; the stage before is released
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      if (c > 0 && (tid & 127) == 0) mbar_arrive(empty + (gc - 1) % WG_STAGES);
#else
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      if ((tid & 127) == 0) mbar_arrive(empty + s);
#endif
    }
#if WG_PIPE
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if ((tid & 127) == 0) mbar_arrive(empty + (gc - 1) % WG_STAGES);
#endif
    if (RN_U) asm volatile("bar.sync 1, 256;" ::: "memory");   // rn_u of this unit written by all consumer threads
    // ADD, the block's last unit, 64+ columns per warpgroup: the residual values of the whole unit come into the (now
    // idle) stage buffers by cp.async, all requested at once, one wait: in the column-group loop below each group's
    // loads waited for a round trip behind the previous group's stores (2.1 ms of a 1024-column pass)
    constexpr bool XST = EP == EP_ADD && !XINIT && EPI_XST && NB >= EPI_XST_NB;
    static_assert(!XST || NB * NT * 512 <= WG_STAGES * STAGE, "XST: residual staging fits in the stage buffers");
    const bool xst = XST && u + (int)gridDim.x >= n_units;
    float* const xsg = (float*)smem;          // [NB / 8][NT][4][256 threads]
    if (XST && xst) {
      asm volatile("bar.sync 1, 256;" ::: "memory");   // both warpgroups are done with the stages
      const int g = lane >> 2, t = lane & 3, t2 = warp & 3;
#pragma unroll
      for (int j = 0; j < NB / 8; j++) {
        const int n0 = cb * 2 * NB + wg * NB + 8 * j + 2 * t, n1 = n0 + 1;
#pragma unroll
        for (int i = 0; i < NT; i++) {
          const int row = (rb * (4 * NT) + 4 * i + t2) * 16 + g;
          float* d = xsg + ((j * NT + i) * 4) * 256 + tid;
          if (n0 < T) {
            const float* p = a.x + (size_t)n0 * H + row;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sm32(d)), "l"(p) : "memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sm32(d + 256)), "l"(p + 8) : "memory");
          }
          if (n1 < T) {
            const float* p = a.x + (size_t)n1 * H + row;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sm32(d + 512)), "l"(p) : "memory");
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(sm32(d + 768)), "l"(p + 8) : "memory");
          }
        }
      }
      asm volatile("cp.async.wait_all;" ::: "memory");
    }
    if (TRACK_A) {
      // the epilogue's rounding must not be scheduled before the wait above (the assembler would serialize the wgmma)
#pragma unroll
      for (int i = 0; i < NT; i++) WG<NB>::touch(acc[i]);
    }
    // Up to 64 columns (64-row blocks, bound by streaming): the last unit of the block is computed; ask L2 for the
    // first 4 stages this block will copy in the next GEMM kernel (block b starts with row block b), so that its
    // start does not wait for HBM. One prefetch per consumer thread: 4 stages x 64 lines. Step 6.09 -> 6.02 ms
    // (2 stages 6.04, 8 stages 6.02).
    if (WG_NEXT && (NT == 1 || EP == EP_QKVP) && NB <= 32 && T <= 64 && a.nextW && u + gridDim.x >= n_units && (int)blockIdx.x < a.nextM / 64) {
      const int line = tid & 63;
#pragma unroll
      for (int k = 0; k < WG_NEXT_N / 4; k++) {
        const int c = WG_NEXT_OFF + (tid >> 6) + 4 * k;
        const char* pw = (const char*)(a.nextW + ((size_t)c * (a.nextM / 8) + blockIdx.x * 8) * 512);
        asm volatile("prefetch.global.L2 [%0];" :: "l"(pw + line * 128));
      }
    }
    // ---- epilogue. Register 4 j + e of thread (warp t2 of the group, lane (g, t)) of the 64-row tile i holds
    // column group j of the warpgroup, e: 0 (row g, col 2t), 1 (row g, col 2t+1), 2 (row g+8, col 2t), 3 (row g+8, col 2t+1)
    // with rows relative to the 16-row tile 4 i + t2 of the block.
    {
      const int g = lane >> 2, t = lane & 3, t2 = warp & 3;
#pragma unroll
      for (int j = 0; j < NB / 8; j++) {
        const int n0 = cb * 2 * NB + wg * NB + 8 * j + 2 * t, n1 = n0 + 1;
        const bool v0 = n0 < T, v1 = n1 < T;
        float r0 = 0.f, r1 = 0.f;
        float bv0 = NINF, bv1 = NINF; int bi0 = 0x7fffffff, bi1 = 0x7fffffff;
        if (EP == EP_QKV || EP == EP_QKVP || EP == EP_GU || EP == EP_LM || EP == EP_LMARG) {
          if (TRACK_A) r0 = r1 = 1.0f;     // the input was normalized (k_rnorm)
          else {
            if (EPI_ABL & 1) { r0 = r1 = 1.0f; }
            else if (RN_U) { r0 = rn_u[(ui & 1) * 256 + (n0 - cb * 2 * NB)]; r1 = rn_u[(ui & 1) * 256 + (n1 - cb * 2 * NB)]; }
            else {
            if (v0) r0 = rn_here ? rn_s[n0] : __ldcg(a.rn + n0);
            if (v1) r1 = rn_here ? rn_s[n1] : __ldcg(a.rn + n1);
            }
          }
        }
        // ADD: the residual values of this column pair are requested together, before anything is stored
        // (a load between stores waits for its turn: the epilogue was 4.4 of 23.7 ms of GEMM time per 1024-column pass)
        float xin[NT][4];
        if (EP == EP_ADD && !XINIT) {
#pragma unroll
          for (int i = 0; i < NT; i++) {
            const int row = (rb * (4 * NT) + 4 * i + t2) * 16 + g;
            if (EPI_ABL & 2) { xin[i][0] = xin[i][1] = xin[i][2] = xin[i][3] = 0.f; continue; }
            if (XST && xst) {
              const float* d = xsg + ((j * NT + i) * 4) * 256 + tid;
              if (v0) { xin[i][0] = d[0]; xin[i][1] = d[256]; }
              if (v1) { xin[i][2] = d[512]; xin[i][3] = d[768]; }
              continue;
            }
            // (EPI_XNC: through the read-only path: every value is read once, by the thread that writes it back; the
            // compiler may then request the values of later column groups before the stores of earlier ones)
            if (v0) { const float* p = a.x + (size_t)n0 * H + row; xin[i][0] = EPI_XNC ? __ldg(p) : __ldcg(p); xin[i][1] = EPI_XNC ? __ldg(p + 8) : __ldcg(p + 8); }
            if (v1) { const float* p = a.x + (size_t)n1 * H + row; xin[i][2] = EPI_XNC ? __ldg(p) : __ldcg(p); xin[i][3] = EPI_XNC ? __ldg(p + 8) : __ldcg(p + 8); }
          }
        }
#pragma unroll
        for (int i = 0; i < NT; i++) {
          const int tile = rb * (4 * NT) + 4 * i + t2;
          const int row = tile * 16 + g;
          const float* cc = &acc[i][4 * j];
          if (EP == EP_QKVP) {
            // the block has one head (128 rows) of every column: into shared memory, prepared below
            const int d = (4 * i + t2) * 16 + g;
            if (v0) { xs[n0 * 128 + d] = ta_r(cc[0] * r0); xs[n0 * 128 + d + 8] = ta_r(cc[2] * r0); }
            if (v1) { xs[n1 * 128 + d] = ta_r(cc[1] * r1); xs[n1 * 128 + d + 8] = ta_r(cc[3] * r1); }
          } else if ((EPI_ABL & 4) && EP == EP_RAW) {
            // timing only: no stores
          } else if (EP == EP_QKV || EP == EP_LM || EP == EP_RAW) {
            const size_t stride = EP == EP_QKV ? QKV_ROWS : (EP == EP_LM ? VOCAB : (size_t)a.M);
            const float q0 = EP == EP_RAW ? 1.f : r0, q1 = EP == EP_RAW ? 1.f : r1;
            const bool rd = EP != EP_RAW;      // track A: q/k/v and the logits are BF16 values
            if (v0) { a.out[(size_t)n0 * stride + row] = rd ? ta_r(cc[0] * q0) : cc[0] * q0; a.out[(size_t)n0 * stride + row + 8] = rd ? ta_r(cc[2] * q0) : cc[2] * q0; }
            if (v1) { a.out[(size_t)n1 * stride + row] = rd ? ta_r(cc[1] * q1) : cc[1] * q1; a.out[(size_t)n1 * stride + row + 8] = rd ? ta_r(cc[3] * q1) : cc[3] * q1; }
          } else if ((EPI_ABL & 8) && (EP == EP_GU || EP == EP_ADD)) {
            // timing only: no epilogue work
          } else if (EP == EP_GU) {
            const int jr = tile * 8 + g;   // rows 0..7 of the tile are gate[jr], rows 8..15 are up[jr]
            if (v0) { float gt = ta_r(cc[0] * r0); float m = ta_r(gt / (1.0f + __expf(-gt))) * ta_r(cc[2] * r0); a.frag[xidx(jr, n0, NG_X)] = f2bf(m); }
            if (v1) { float gt = ta_r(cc[1] * r1); float m = ta_r(gt / (1.0f + __expf(-gt))) * ta_r(cc[3] * r1); a.frag[xidx(jr, n1, NG_X)] = f2bf(m); }
          } else if (EP == EP_ADD) {
            const float w0 = a.aux[row], w1 = a.aux[row + 8];
            float s0 = 0.f, s1 = 0.f;
            if (v0) {
              float* p = a.x + (size_t)n0 * H + row;
              float xa = XINIT ? cc[0] : ta_r(xin[i][0] + ta_r(cc[0])), xb = XINIT ? cc[2] : ta_r(xin[i][1] + ta_r(cc[2]));
              p[0] = xa; p[8] = xb;
              if (!TRACK_A) { a.frag[xidx(row, n0, NG_X)] = f2bf(w0 * xa); a.frag[xidx(row + 8, n0, NG_X)] = f2bf(w1 * xb); }
              s0 = __fmaf_rn(xa, xa, __fmul_rn(xb, xb));
            }
            if (v1) {
              float* p = a.x + (size_t)n1 * H + row;
              float xa = XINIT ? cc[1] : ta_r(xin[i][2] + ta_r(cc[1])), xb = XINIT ? cc[3] : ta_r(xin[i][3] + ta_r(cc[3]));
              p[0] = xa; p[8] = xb;
              if (!TRACK_A) { a.frag[xidx(row, n1, NG_X)] = f2bf(w0 * xa); a.frag[xidx(row + 8, n1, NG_X)] = f2bf(w1 * xb); }
              s1 = __fmaf_rn(xa, xa, __fmul_rn(xb, xb));
            }
            s0 += __shfl_xor_sync(0xffffffffu, s0, 4); s1 += __shfl_xor_sync(0xffffffffu, s1, 4);
            s0 += __shfl_xor_sync(0xffffffffu, s0, 8); s1 += __shfl_xor_sync(0xffffffffu, s1, 8);
            s0 += __shfl_xor_sync(0xffffffffu, s0, 16); s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
            if (g == 0) {
              if (v0) a.ssq[(size_t)n0 * 256 + tile] = s0;
              if (v1) a.ssq[(size_t)n1 * 256 + tile] = s1;
            }
          } else if (EP == EP_LMARG) {
            // the logits of EP_LM, reduced to the best row per column; rows ascend, strict > keeps the lowest
            float v;
            v = ta_r(cc[0] * r0); if (v > bv0) { bv0 = v; bi0 = row; }
            v = ta_r(cc[2] * r0); if (v > bv0) { bv0 = v; bi0 = row + 8; }
            v = ta_r(cc[1] * r1); if (v > bv1) { bv1 = v; bi1 = row; }
            v = ta_r(cc[3] * r1); if (v > bv1) { bv1 = v; bi1 = row + 8; }
          }
        }
        if (EP == EP_LMARG) {
#pragma unroll
          for (int o = 16; o >= 4; o >>= 1) {
            float uv = __shfl_xor_sync(0xffffffffu, bv0, o); int ix = __shfl_xor_sync(0xffffffffu, bi0, o);
            if (uv > bv0 || (uv == bv0 && ix < bi0)) { bv0 = uv; bi0 = ix; }
            uv = __shfl_xor_sync(0xffffffffu, bv1, o); ix = __shfl_xor_sync(0xffffffffu, bi1, o);
            if (uv > bv1 || (uv == bv1 && ix < bi1)) { bv1 = uv; bi1 = ix; }
          }
          if (g == 0) {
            const size_t base = (size_t)(rb * 4 + t2) * T;
            if (v0) { a.bm_val[base + n0] = bv0; a.bm_idx[base + n0] = bi0; }
            if (v1) { a.bm_val[base + n1] = bv1; a.bm_idx[base + n1] = bi1; }
          }
        }
      }
    }
    if (EP == EP_QKVP) {
      // Up to 16 columns a unit is one head (128 rows) of every column, and both warpgroups have stored their parts:
      // a warp per column prepares the vector exactly as k_prep would (36 kernels less per decode step, and the
      // q/k/v values never go through global memory).
      __syncthreads();
      const int padd = PV(a.pos_add, a.dev);
      for (int col = warp; col < T; col += 8) prep_vec(a.dev, a.layer, padd, col, rb, ((const float4*)(xs + col * 128))[lane], lane);
    }
  }
}

template <int EP>
__global__ void __launch_bounds__(WG_THREADS, 1) k_gemm(GemmArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  // the number of columns comes from the plan before the wait: the plan is older than the kernel before
  // (k_sched, or the host); gemm_body waits (every block, before it ends: a kernel that ended without waiting
  // would let its successor run alongside its predecessor)
  const int T = HV(a.T, a.plan);
  if (T == 0) { pdl_wait(true); return; }
  constexpr int E = PDL_EARLY;
  switch (a.shape >= 0 ? a.shape : gemm_shape(a.M, T, gridDim.x)) {
    case 0: gemm_body<EP, 16, 1>(a, T, smem, E); break;
    case 7:
      if (EP == EP_QKV && a.prep) gemm_body<EP == EP_QKV ? EP_QKVP : EP, 8, EP == EP_QKV ? 2 : 1>(a, T, smem, E);
      else gemm_body<EP, 8, 1>(a, T, smem, E);
      break;
    case 1: gemm_body<EP, 32, 1>(a, T, smem, E); break;
    case 2: gemm_body<EP, 64, 1>(a, T, smem, E); break;
    case 3: gemm_body<EP, 64, 2>(a, T, smem, E); break;
    case 4: if constexpr (EP != EP_LM && EP != EP_LMARG) gemm_body<EP, 64, 3>(a, T, smem, E); else pdl_wait(true); break;   // the vocabulary is not
    case 5: if constexpr (EP != EP_LM && EP != EP_LMARG) gemm_body<EP, 64, 4>(a, T, smem, E); else pdl_wait(true); break;   // a multiple of 192
    default: gemm_body<EP, 128, 2>(a, T, smem, E); break;
  }
}
// the same with the shape fixed at compile time (the host launches these when it knows the number of columns)
template <int EP, int NB, int NT, int PB = 0>
__global__ void __launch_bounds__(WG_THREADS, 1) k_gemm1(GemmArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  gemm_body<EP, NB, NT, PB>(a, HV(a.T, a.plan), smem, PDL_EARLY);   // (T < 0: the serving loop's light pass; the plan is older than the kernel before)
}
#define WG_SMEM_MAX2 (WG_SMEM(64, 1) > WG_SMEM(16, 1) ? WG_SMEM(64, 1) : WG_SMEM(16, 1))
#define WG_SMEM_MAX (WG_SMEM_MAX2 > WG_SMEM(8, 2) ? WG_SMEM_MAX2 : WG_SMEM(8, 2))
static_assert(WG_SMEM(64, 4) <= WG_SMEM_MAX && WG_SMEM(128, 2) <= WG_SMEM_MAX && WG_SMEM(64, 3) <= WG_SMEM_MAX && WG_SMEM(32, 1) <= WG_SMEM_MAX && WG_SMEM(8, 1) <= WG_SMEM_MAX && WG_SMEM(8, 2) <= WG_SMEM_MAX && WG_SMEM_MAX <= 232448, "smem");

// ---------------------------------------------------------------------------------------------------------
// small per-column kernels
// ---------------------------------------------------------------------------------------------------------

// embedding: x = embed[token]; fragments of bf16(w0 * x) and per-tile sums of x^2 for the first norm
struct EmbedArgs { Dev d; int T; int pos_add; const long long* tok64; const int* arena; int cell_pos; };   // arena: PL_TOK holds indices into it; cell_pos >= 0: step(): the position cell (as k_setcell)
__global__ void __launch_bounds__(256) k_embed(EmbedArgs a) {
  pdl_wait();
  const Dev& D = a.d;
  if (blockIdx.x == 0 && threadIdx.x < NL * 3) D.ctr[threadIdx.x] = 0;
  if (a.cell_pos >= 0 && blockIdx.x == 0 && threadIdx.x == 0) { D.cell[0] = a.cell_pos; D.cell[1] = 0; D.cell[2] = 0; }
  const int T = HV(a.T, D.plan);
  for (int col = blockIdx.x; col < T; col += gridDim.x) {
    long long tok;
    if (a.tok64) tok = a.tok64[col];
    else {
      tok = D.plan[PL_TOK + col];
      if (tok < 0) tok = __ldcg(D.last_tok + (D.plan[PL_SLOT + col] & 0xffff));
      else if (a.arena) tok = a.arena[tok];
    }
    const uint2* e = (const uint2*)(D.embed + (size_t)tok * H);
    // all loads first (a store between them makes each wait for the one before: 4 round trips, 6.9 us per step)
    uint2 ev[H / 1024]; float4 wq[H / 1024];
#pragma unroll
    for (int r = 0; r < H / 1024; r++) { ev[r] = e[threadIdx.x + 256 * r]; wq[r] = ((const float4*)D.norm_w)[threadIdx.x + 256 * r]; }
#pragma unroll
    for (int r = 0; r < H / 1024; r++) {
      const int i = threadIdx.x + 256 * r;
      const uint2 v = ev[r];
      float4 f = make_float4(bf2f(v.x & 0xffffu), bf2f(v.x >> 16), bf2f(v.y & 0xffffu), bf2f(v.y >> 16));
      ((float4*)(D.x + (size_t)col * H))[i] = f;
      const float4 wv = wq[r];
      const int k = i * 4;
      D.xfrag[xidx(k + 0, col, NG_X)] = f2bf(wv.x * f.x);
      D.xfrag[xidx(k + 1, col, NG_X)] = f2bf(wv.y * f.y);
      D.xfrag[xidx(k + 2, col, NG_X)] = f2bf(wv.z * f.z);
      D.xfrag[xidx(k + 3, col, NG_X)] = f2bf(wv.w * f.w);
      float p = __fadd_rn(__fmaf_rn(f.x, f.x, __fmul_rn(f.y, f.y)), __fmaf_rn(f.z, f.z, __fmul_rn(f.w, f.w)));
      p += __shfl_xor_sync(0xffffffffu, p, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2);
      if ((i & 3) == 0) D.ssq[(size_t)col * 256 + (i >> 2)] = p;
    }
  }
}

// 1/rms per column from the 256 per-tile partial sums (fixed order). One warp per column.
struct RnArgs { const float* ssq; float* rn; int T; const int* plan; int skip_small; const float* x; const float* w; unsigned short* frag; };   // x, w, frag: track A   // skip_small: nothing to do up to 64 columns (the GEMM does it)
#ifndef TA_RN_LINES
#define TA_RN_LINES 1      // track A, more than 64 columns: whole-line stores (see k_rnorm)
#endif
__global__ void __launch_bounds__(256) k_rnorm(RnArgs a) {
  pdl_wait();
  const int lane = threadIdx.x & 31;
  const int T = HV(a.T, a.plan);
  if (!TRACK_A && a.skip_small && T <= 64) return;
  if (TRACK_A) {
    // track A: the matvec's input is the normalized vector itself, bf16(w * bf16(x * 1/rms)), and the matvec has
    // nothing left to scale. 8 warps per column, each an eighth of the vector (two groups of 8 elements per
    // thread, a 16-byte store each) and all of them the same 1/rms. Every load of an item is requested before
    // anything is computed (three round trips to memory in a row made this kernel 100 us at 2040 columns).
    const float4* wq = (const float4*)a.w;
#define RN_E(x, w) ((unsigned)f2bf((w) * ta_r((x) * rn)))
#define RN_P(xa, wa) (RN_E(xa.x, wa.x) | (RN_E(xa.y, wa.y) << 16)), (RN_E(xa.z, wa.z) | (RN_E(xa.w, wa.w) << 16))
    if (TA_RN_LINES && T > 64) {
      // More than 64 columns: a warp per (group of 8 columns, eighth of the vector). Lane (c, r) = (lane & 7,
      // lane >> 3) handles column 8g + c, so one store instruction fills four whole 128-byte lines of the operand
      // layout (one column per lane wrote 16 bytes into 32 lines shared with 7 other blocks: 2 ms per 1024-column
      // pass). Each of the 8 columns' 1/rms is k_rnorm's sum (same lanes, same order): bits unchanged.
      const int c = lane & 7, r = lane >> 3, ng = (T + 7) >> 3;
      for (int it = blockIdx.x * 8 + (threadIdx.x >> 5); it < ng * 8; it += gridDim.x * 8) {
        const int g = it >> 3, sl = it & 7, col = 8 * g + c;
        float4 uu[8], vv[8];
#pragma unroll
        for (int cc = 0; cc < 8; cc++) {
          const int cl = 8 * g + cc < T ? 8 * g + cc : T - 1;
          const float4* q = (const float4*)(a.ssq + (size_t)cl * 256) + lane * 2;
          uu[cc] = __ldcg(q); vv[cc] = __ldcg(q + 1);
        }
        float rn = 0.0f;
#pragma unroll
        for (int cc = 0; cc < 8; cc++) {
          float s = ((uu[cc].x + uu[cc].y) + (uu[cc].z + uu[cc].w)) + ((vv[cc].x + vv[cc].y) + (vv[cc].z + vv[cc].w));
          s = warp_sum(s);
          const float rc = rsqrtf(__fmaf_rn(s, 1.0f / H, 1e-6f));
          if (cc == c) rn = rc;
        }
        const float4* xp = (const float4*)(a.x + (size_t)(col < T ? col : T - 1) * H);
#pragma unroll
        for (int b = 0; b < 2; b++) {
          float4 xa[8], xb[8];
#pragma unroll
          for (int u = 0; u < 8; u++) { const int kc = sl * 64 + r + 4 * (8 * b + u); xa[u] = __ldcg(xp + 2 * kc); xb[u] = __ldcg(xp + 2 * kc + 1); }
#pragma unroll
          for (int u = 0; u < 8; u++) {
            const int kc = sl * 64 + r + 4 * (8 * b + u);
            const float4 w0 = wq[2 * kc], w1 = wq[2 * kc + 1];
            if (col < T) *(uint4*)(a.frag + xidx(8 * kc, col, NG_X)) = make_uint4(RN_P(xa[u], w0), RN_P(xb[u], w1));
          }
        }
      }
      return;
    }
    for (int it = blockIdx.x * 8 + (threadIdx.x >> 5); it < T * 8; it += gridDim.x * 8) {
      const int col = it >> 3, o0 = (it & 7) * (H / 64) + lane, o1 = o0 + 32;
      const float4* q = (const float4*)(a.ssq + (size_t)col * 256) + lane * 2;
      const float4* xp = (const float4*)(a.x + (size_t)col * H);
      const float4 u = __ldcg(q), v = __ldcg(q + 1);
      const float4 x0 = __ldcg(xp + 2 * o0), x1 = __ldcg(xp + 2 * o0 + 1), x2 = __ldcg(xp + 2 * o1), x3 = __ldcg(xp + 2 * o1 + 1);
      const float4 w0 = wq[2 * o0], w1 = wq[2 * o0 + 1], w2 = wq[2 * o1], w3 = wq[2 * o1 + 1];
      float s = ((u.x + u.y) + (u.z + u.w)) + ((v.x + v.y) + (v.z + v.w));
      s = warp_sum(s);
      const float rn = rsqrtf(__fmaf_rn(s, 1.0f / H, 1e-6f));
#ifdef TA_RN_LIN_ABL   // timing ablation only (wrong layout): contiguous stores
      *(uint4*)(a.frag + (size_t)col * H + 8 * o0) = make_uint4(RN_P(x0, w0), RN_P(x1, w1));
      *(uint4*)(a.frag + (size_t)col * H + 8 * o1) = make_uint4(RN_P(x2, w2), RN_P(x3, w3));
#else
      *(uint4*)(a.frag + xidx(8 * o0, col, NG_X)) = make_uint4(RN_P(x0, w0), RN_P(x1, w1));
      *(uint4*)(a.frag + xidx(8 * o1, col, NG_X)) = make_uint4(RN_P(x2, w2), RN_P(x3, w3));
#endif
    }
#undef RN_P
#undef RN_E
    return;
  }
  for (int col = blockIdx.x * 8 + (threadIdx.x >> 5); col < T; col += gridDim.x * 8) {   // a warp per column
    const float4* q = (const float4*)(a.ssq + (size_t)col * 256) + lane * 2;
    float4 u = __ldcg(q), v = __ldcg(q + 1);
    float s = ((u.x + u.y) + (u.z + u.w)) + ((v.x + v.y) + (v.z + v.w));
    s = warp_sum(s);
    if (lane == 0) a.rn[col] = rsqrtf(__fmaf_rn(s, 1.0f / H, 1e-6f));
  }
}

// After the q/k/v matvec: per column 48 vectors of 128 (32 q heads, 8 k heads, 8 v heads), one warp each.
// q, k: RMSNorm over the head, RoPE at the column's position. q -> qfrag; k, v -> the sequence's KV pages.
#ifndef PREP_BLOCKS
#define PREP_BLOCKS 2      // blocks of 16 warps per SM (4 or 8: no difference)
#endif
struct PrepArgs { Dev d; int T; int layer; int pos_add; int skip16; };   // skip16: nothing to do up to 16 columns (the q/k/v GEMM does it)
__global__ void __launch_bounds__(512) k_prep(PrepArgs a) {
  pdl_wait();
  const Dev& D = a.d;
  const int lane = threadIdx.x & 31;
  const int T = HV(a.T, D.plan);
  if (a.skip16 && T <= 16) return;
  const int total = T * 48, padd = PV(a.pos_add, D);
  for (int wt = blockIdx.x * 16 + (threadIdx.x >> 5); wt < total; wt += gridDim.x * 16) {
    const int col = wt / 48, vec = wt - col * 48;
    const int row = vec < 32 ? vec * 128 : (vec < 40 ? 4096 + (vec - 32) * 128 : 5120 + (vec - 40) * 128);
    const float4 x = __ldcg((const float4*)(D.qkv + (size_t)col * QKV_ROWS + row) + lane);
    prep_vec(D, a.layer, padd, col, vec, x, lane);
  }
}

// ---------------------------------------------------------------------------------------------------------
// attention (see the canonical arithmetic at the top), on the warpgroup MMA.
// A task = (1 or 2 units of up to 16 consecutive positions of one sequence, kv head, super-segment), claimed by a
// block from a counter. Warpgroup w of the block computes unit w: 64 query rows (16 positions x 4 heads).
// Per segment of 128 keys:
//   scores [64 rows x 128 positions] = 8 instructions m64n128k16 (A = q rows x dims from shared memory, B = K);
//   thread (warp t2, lane (g8, t)) holds rows g8, g8 + 8 of the warp's 16 rows and positions 8 v2 + 2t, 2t + 1:
//   register 4 v2 + 2 v1 + v0. Max, 2^(s - R), bf16 pack: per thread, in registers;
//   the packed weights of 16 positions ARE the register A operand of the value product
//   O [64 rows x 128 dims] += W x V (B = V from shared memory), and Z against a group of ones.
// K and V of a segment (32 KB each, contiguous) come into shared memory by the copy engine, one segment ahead.
// Decode rows are units of one position: the same kernel, one warpgroup active.
// A unit with more than one super-segment leaves its parts in a scratch and k_attn_fold folds them in order; with
// one super-segment the fold is the identity and the kernel writes the output itself.
// History: v1 scores through shared memory 6.0 ns per (query, key) pair; v3 block kernel + two-level fold 3.0;
// v4 state in registers 2.35; v5 copy engine for K/V 1.63 (all m16n8k16; the instruction count per segment
// bounded them); v6 this one.
// ---------------------------------------------------------------------------------------------------------
#define A_THREADS 256
__device__ __forceinline__ float ex2(float x) { float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ unsigned cvt_bf(float x) { unsigned short r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x)); return r; }
__device__ __forceinline__ void fold_factors(float M, float m, float& Mn, float& fa, float& fb) {
  Mn = fmaxf(M, m);
  fa = (M == Mn) ? 1.0f : (M == NINF ? 0.0f : ex2(__fsub_rn(M, Mn)));
  fb = (m == Mn) ? 1.0f : (m == NINF ? 0.0f : ex2(__fsub_rn(m, Mn)));
}
__device__ __forceinline__ unsigned pack2(float a, float b) { return cvt_bf(a) | (cvt_bf(b) << 16); }
__device__ __forceinline__ uint64_t wg_desc2(const void* smem_ptr, unsigned stride16) {   // operand with `stride16` x 16 bytes per 8 rows
  return (uint64_t)((sm32(smem_ptr) >> 4) & 0x3fff) | ((uint64_t)8 << 16) | ((uint64_t)stride16 << 32);
}

// maxima of the valid scores of the two rows of this thread, reduced over the 4 lanes that share the rows
template <bool FULL>
__device__ __forceinline__ void seg_max(const float (&sc)[64], int limA, int limB, int t, float& mA, float& mB) {
  mA = NINF; mB = NINF;
#pragma unroll
  for (int r = 0; r < 64; r++) {
    const int pos = 8 * (r >> 2) + 2 * t + (r & 1);
    if (r & 2) { if (FULL || pos <= limB) mB = fmaxf(mB, sc[r]); }
    else { if (FULL || pos <= limA) mA = fmaxf(mA, sc[r]); }
  }
  mA = fmaxf(mA, __shfl_xor_sync(0xffffffffu, mA, 1)); mB = fmaxf(mB, __shfl_xor_sync(0xffffffffu, mB, 1));
  mA = fmaxf(mA, __shfl_xor_sync(0xffffffffu, mA, 2)); mB = fmaxf(mB, __shfl_xor_sync(0xffffffffu, mB, 2));
}
// the weights of positions 16 jv .. 16 jv + 15 as the register A operand: (row A, 2t), (row B, 2t), (row A, 2t + 8), (row B, 2t + 8), two positions each
template <bool FULL>
__device__ __forceinline__ uint4 seg_weights(const float (&sc)[64], int jv, int limA, int limB, int t, float RA, float RB) {
  unsigned w[8];
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const int r = 8 * jv + e;              // 4 v2 + 2 v1 + v0 with v2 = 2 jv + (e >> 2)
    const int pos = 8 * (r >> 2) + 2 * t + (r & 1);
    const unsigned b = cvt_bf(ex2(__fsub_rn(sc[r], (r & 2) ? RB : RA)));
    w[e] = (FULL || pos <= ((r & 2) ? limB : limA)) ? b : 0u;
  }
  return make_uint4(w[0] | (w[1] << 16), w[2] | (w[3] << 16), w[4] | (w[5] << 16), w[6] | (w[7] << 16));
}
// output of this thread's two rows (positions pA, pB of the unit, head h): bf16(O / Z) into the o projection's input
__device__ __forceinline__ void att_out(const Dev& D, const float* O, float ZA, float ZB, int col0, int nq, int g, int t2, int lane) {
  const int g8 = lane >> 2, t = lane & 3, pA = 4 * t2 + (g8 >> 2), pB = pA + 2, kb = (4 * g + (g8 & 3)) * 128;
  const float rzA = __fdiv_rn(1.0f, ZA), rzB = __fdiv_rn(1.0f, ZB);
#pragma unroll
  for (int v2 = 0; v2 < 16; v2++) {
    if (pA < nq) *(unsigned*)(D.afrag + xidx(kb + 8 * v2 + 2 * t, col0 + pA, NG_X)) = pack2(__fmul_rn(O[4 * v2], rzA), __fmul_rn(O[4 * v2 + 1], rzA));
    if (pB < nq) *(unsigned*)(D.afrag + xidx(kb + 8 * v2 + 2 * t, col0 + pB, NG_X)) = pack2(__fmul_rn(O[4 * v2 + 2], rzB), __fmul_rn(O[4 * v2 + 3], rzB));
  }
}

// part: 0 tasks task0 .. + n_tasks; 1 the single-unit tasks that are decode rows (the first hdr[6] of them);
// 2 the other single-unit tasks
struct AttArgs { Dev d; int task0; int n_tasks; int layer; int pos_add; int* ctr; int part; };
// Pipeline of a task over its segments s (per warpgroup; "group" = a committed batch of async instructions):
//   A  scores(s+1) -> scB                       needs K(s+1) in shared memory
//   B  wait: values(s-1) done; block barrier; thread 0 asks the copy engine for V(s+1) and K(s+2)
//   C  scalar work of segment s on scA: maxima, rescale of O/Z if a maximum moved, weights of the 8 x 16 positions
//   D  values(s): O += W V, Z += W 1            needs V(s) in shared memory
//   E  wait: scores(s+1) done; scA = scB
// so the tensor cores always have a batch while the threads do the scalar work, and the copies have a whole
// iteration. The warpgroup MMAs are issued without any condition around them (ptxas serializes them otherwise,
// "C7519"): a warpgroup without valid positions in a segment runs with all weights 0 (an exact identity), and
// the groups of 16 positions past the end of the sequence take V from the last valid group (finite) with
// weights 0. Hence two instances: NWG = 2 for pairs of units, NWG = 1 for single units (decode rows).
// Ablation of the unpipelined version (1024 columns at context 16k, attention 27.5 ms): tensor instructions 14.4,
// scalar 6.8, copies and barriers ~6: they ran one after the other.
#ifndef ATT_ABL
#define ATT_ABL 0     // timing experiments only: 1 no K/V copies after the first, 2 no scalar work, 4 no tensor instructions
#endif
#ifndef ATT_ROT
#define ATT_ROT 1
#endif
#ifndef ATT_ASYNC
#define ATT_ASYNC 0        // 1: warpgroup 0 leaves the scores of s + 1 in flight during its scalar work of s (same bits; measured: no change)
#endif
#ifndef ATT_DEC
#define ATT_DEC 1
#endif
#define AT_NBUF(NWG) ((NWG) == 1 ? 3 : 2)     // segments of K and of V in shared memory (decode rows look 2 ahead)
#define AT_K 0                                // [AT_NBUF][32 KB] K segments
#define AT_V(NWG) (AT_NBUF(NWG) * 32768)      // [AT_NBUF][32 KB V segment | 2 KB: a 17th dim group of ones, which gives Z]
#define AT_VBUF (32768 + 2048)
#define AT_Q(NWG) (AT_V(NWG) + AT_NBUF(NWG) * AT_VBUF)   // q of the units, 16 KB each
#define AT_BAR(NWG) (AT_Q(NWG) + (NWG) * 16384)   // uint64 [7]: K0, K1, V0, V1, q, K2, V2 arrived
#define AT_KB(i) ((i) < 2 ? (i) : 5)          // barrier of K buffer i, of V buffer i
#define AT_VB(i) ((i) < 2 ? 2 + (i) : 6)
#define AT_W(NWG) (AT_BAR(NWG) + 128)         // decode rows: the weights of a segment as a B operand, 2 KB
#define AT_SCR(NWG) (AT_W(NWG) + 2048)        // decode rows: float [32] maxima per (warp, head), [4] Z per head
#define AT_SMEM(NWG) (AT_SCR(NWG) + 256)
// The layout with NWG = 1 is the decode rows' kernel (k_attn_dec, 3 warpgroups); k_attn uses the layout of 2 for
// pairs of units and for single units alike.
#define AD_THREADS 384
static_assert(AT_SMEM(1) <= 232448 && AT_SMEM(2) <= 232448, "shared memory of the attention kernel");

// ---- A unit of ONE position (a decode row): the same numbers with the operand roles swapped.
// The general kernel computes 64 query rows x 128 keys per instruction and a decode row has 4 real rows (its 4
// heads of the kv head). Here
//   scores [64 keys x 8 heads]  = A (K: keys x dims) x B (q: dims x heads)          2 key tiles x 8 instructions m64n8k16
//   O      [64 dims x 8 heads] += A (V: dims x keys) x B (weights: keys x heads)    (2 dim tiles + the ones: Z) x 8
// from the same bytes of K, V and q (an operand is [major / 8][k / 8][major % 8][k % 8] on either side). The
// tensor cores give the same bits for both arrangements (tests/att_roles.cu: 0 of 1024 scores, 0 of 1032 values
// differ): each element is the same 16 exact products, summed over the same k, accumulated in the same order.
// An instruction costs a warpgroup ~40 ns whatever its width, and a block alone on its K/V gets a segment from
// HBM in 0.5 us (tests/kv_stride.cu), so the tiles are dealt to THREE warpgroups that run side by side: 8
// instructions each for the scores (warpgroups 0, 1: a key tile each) and for the values (0, 1: a dim tile
// each; 2: the ones). (All on one warpgroup, 40 instructions per segment: no faster than the general kernel.)
// Thread (warpgroup wg, warp t2, lane (g8, t)) holds (row g8, col 2t), (g8, 2t + 1), (g8 + 8, 2t), (g8 + 8, 2t + 1) of
// the warp's 16 rows of its tile; columns 0..3 are the heads (t < 2), 4..7 are whatever follows in the q buffer
// (ignored). The maximum of a head is over keys, i.e. over 8 warps: through shared memory, one block barrier.
// K and V are requested three segments deep.
__device__ __forceinline__ void att_dec(const Dev& D, int layer, unsigned char* smem, uint64_t* bar, unsigned* ph, int slot, int g, int s0, int s1,
                                        int pos, int col0, int pbase, int ss, int unit) {
  const int tid = threadIdx.x, lane = tid & 31, wg = tid >> 7, t2 = (tid >> 5) & 3, g8 = lane >> 2, t = lane & 3;
  const bool live = t < 2;
#define AD_KPTR(sx) (page_k(D, slot, (sx) * SEG, layer, g) + (size_t)((sx) & 3) * 16384)
  const unsigned char* qs = smem + AT_Q(1);
  unsigned short* wb = (unsigned short*)(smem + AT_W(1));
  float* scr = (float*)(smem + AT_SCR(1));       // [8 warps][4 heads] maxima, then [4] Z
  float O[4] = {0.f, 0.f, 0.f, 0.f};             // warpgroup 0: dims 0..63, 1: dims 64..127, 2: row 0 = Z
  float M0 = NINF, M1 = NINF;                    // running maxima of heads 2t and 2t + 1
  if (tid == 0) {     // the row's 4 heads of q (2 KB), K and V of the first three segments
    mbar_expect(bar + 4, 2048);
    bulk_g2s(smem + AT_Q(1), D.qfrag + ((size_t)unit * 8 + g) * 8192, 2048, bar + 4);
#pragma unroll
    for (int e = 0; e < 3; e++) {
      if (s0 + e < s1) {
        const unsigned short* Kn = AD_KPTR(s0 + e);
        mbar_expect(bar + AT_KB(e), 32768);
        bulk_g2s(smem + AT_K + e * 32768, Kn, 32768, bar + AT_KB(e));
        mbar_expect(bar + AT_VB(e), 32768);
        bulk_g2s(smem + AT_V(1) + e * AT_VBUF, Kn + (size_t)PG_U4 * 8, 32768, bar + AT_VB(e));
      }
    }
  }
  mbar_wait(bar + 4, ph[4] & 1); ph[4]++;
  int b = 0;                                     // buffer of K(s) and of V(s): (s - s0) % 3
  for (int s = s0; s < s1; s++, b = b == 2 ? 0 : b + 1) {
    mbar_wait(bar + AT_KB(b), ph[AT_KB(b)] & 1); ph[AT_KB(b)]++;
    const unsigned char* Ks = smem + AT_K + b * 32768 + (wg & 1) * 16384;    // the warpgroup's 64 keys
    const int lim = pos - s * SEG;               // keys 0 .. lim of the segment exist for this row
    float sc[4] = {0.f, 0.f, 0.f, 0.f};
    float m0 = NINF, m1 = NINF;
    if (wg < 2) {
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int j = 0; j < 8; j++) WGMMA_N8_P(sc, wg_desc2(Ks + j * 256, 128), wg_desc2(qs + j * 256, 128), j ? 1 : 0);
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      if (TRACK_A) { sc[0] = __fmul_rn(sc[0], SC2); sc[1] = __fmul_rn(sc[1], SC2); sc[2] = __fmul_rn(sc[2], SC2); sc[3] = __fmul_rn(sc[3], SC2); }
      // maxima over the valid keys: this thread's, then the 8 lanes of the warp with the same columns
#pragma unroll
      for (int e = 0; e < 4; e++) {
        const int kk = 64 * wg + 16 * t2 + g8 + 8 * (e >> 1);
        if (kk <= lim) { if (e & 1) m1 = fmaxf(m1, sc[e]); else m0 = fmaxf(m0, sc[e]); }
      }
#pragma unroll
      for (int o = 4; o <= 16; o <<= 1) { m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o)); }
      if (g8 == 0 && live) { scr[(wg * 4 + t2) * 4 + 2 * t] = m0; scr[(wg * 4 + t2) * 4 + 2 * t + 1] = m1; }
    }
    __syncthreads();
    // every thread is past its waits for K(s) and V(s - 1): their buffers take K(s + 3) and V(s + 2)
    if (tid == 0) {
      if (s > s0 && s + 2 < s1) {
        const int bv = b == 0 ? 2 : b - 1;
        const unsigned short* Kp = AD_KPTR(s + 2);
        mbar_expect(bar + AT_VB(bv), 32768);
        bulk_g2s(smem + AT_V(1) + bv * AT_VBUF, Kp + (size_t)PG_U4 * 8, 32768, bar + AT_VB(bv));
      }
      if (s + 3 < s1) {
        const unsigned short* Kp = AD_KPTR(s + 3);
        mbar_expect(bar + AT_KB(b), 32768);
        bulk_g2s(smem + AT_K + b * 32768, Kp, 32768, bar + AT_KB(b));
      }
    }
    if (live) {
      m0 = NINF; m1 = NINF;
#pragma unroll
      for (int w = 0; w < 8; w++) { m0 = fmaxf(m0, scr[w * 4 + 2 * t]); m1 = fmaxf(m1, scr[w * 4 + 2 * t + 1]); }
      const float R0 = fmaxf(M0, m0), R1 = fmaxf(M1, m1);
      const float fa0 = (M0 == R0) ? 1.0f : (M0 == NINF ? 0.0f : ex2(__fsub_rn(M0, R0)));
      const float fa1 = (M1 == R1) ? 1.0f : (M1 == NINF ? 0.0f : ex2(__fsub_rn(M1, R1)));
      if (M0 != R0 || M1 != R1) {
        O[0] = __fmul_rn(O[0], fa0); O[1] = __fmul_rn(O[1], fa1); O[2] = __fmul_rn(O[2], fa0); O[3] = __fmul_rn(O[3], fa1);
      }
      M0 = R0; M1 = R1;
      // the weights of the thread's keys and heads into the B operand: [step of 16 keys][key / 8][head][key % 8]
      if (wg < 2) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int kk = 64 * wg + 16 * t2 + g8 + 8 * (e >> 1), h = 2 * t + (e & 1);
          const unsigned w = cvt_bf(ex2(__fsub_rn(sc[e], (e & 1) ? R1 : R0)));
          wb[(kk >> 4) * 128 + (((kk & 15) >> 3) * 8 + h) * 8 + (kk & 7)] = (unsigned short)(kk <= lim ? w : 0u);
        }
      }
    }
    __syncthreads();
    // values: O += V x W over the 8 groups of 16 keys; the groups past the end of the sequence take V of the last
    // one that exists (finite) with weights 0, as in the general kernel
    const int gnpt = min(8, (lim >> 4) + 1);
    mbar_wait(bar + AT_VB(b), ph[AT_VB(b)] & 1); ph[AT_VB(b)]++;
    const unsigned char* Vs = smem + AT_V(1) + b * AT_VBUF + wg * 16384;     // the warpgroup's 64 dims (2: the ones)
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int jv = 0; jv < 8; jv++) WGMMA_N8(O, wg_desc2(Vs + min(jv, gnpt - 1) * 256, 128), wg_desc2((const unsigned char*)wb + jv * 256, 128));
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  }
  // Z of head h is row 0 of the ones' tile: thread (warpgroup 2, warp 0, g8 0, t = h / 2)
  if (wg == 2 && t2 == 0 && g8 == 0 && live) { scr[32 + 2 * t] = O[0]; scr[32 + 2 * t + 1] = O[1]; }
  __syncthreads();
  if (!live || wg == 2) return;
  if (pbase < 0) {
    const float rz0 = __fdiv_rn(1.0f, scr[32 + 2 * t]), rz1 = __fdiv_rn(1.0f, scr[32 + 2 * t + 1]);
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int dim = 64 * wg + 16 * t2 + g8 + 8 * (e >> 1), h = 2 * t + (e & 1);
      D.afrag[xidx((4 * g + h) * 128 + dim, col0, NG_X)] = (unsigned short)cvt_bf(__fmul_rn(O[e], (e & 1) ? rz1 : rz0));
    }
  } else {
    // the part of this super-segment, where k_attn_fold expects it: thread 4 h + (dim % 8) / 2 of the general
    // kernel holds (row h, dims 8 i + 2 t', + 1) in components x, y of entry i, and (M, -, Z, -) in entry 16
    float* P = (float*)(D.att_part + ((size_t)(pbase + ss) * 8 + g) * (17 * 128));
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const int dim = 64 * wg + 16 * t2 + g8 + 8 * (e >> 1), h = 2 * t + (e & 1);
      P[((dim >> 3) * 128 + h * 4 + ((dim & 7) >> 1)) * 4 + (dim & 1)] = O[e];
    }
    if (wg == 0 && t2 == 0 && g8 < 4) {     // 8 threads: (head 2t or 2t + 1, lane g8 of the general kernel's row)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        const int h = 2 * t + e;
        P[(16 * 128 + h * 4 + g8) * 4 + 0] = e ? M1 : M0;
        P[(16 * 128 + h * 4 + g8) * 4 + 2] = scr[32 + h];
      }
    }
  }
#undef AD_KPTR
}

template <int NWG>
__global__ void __launch_bounds__(NWG * 128, 1) k_attn(AttArgs a) {
  pdl_wait();
  extern __shared__ __align__(128) unsigned char smem[];
  const Dev& D = a.d;
  const int tid = threadIdx.x, lane = tid & 31, wg = tid >> 7, t2 = (tid >> 5) & 3, g8 = lane >> 2, t = lane & 3;
  int n_tasks = HV(a.n_tasks, D.plan), task0 = HV(a.task0, D.plan);
  if (a.part == 2) { const int nd = D.plan[PL_HDR + 6]; task0 += nd; n_tasks -= nd; }   // the decode rows are k_attn_dec's
  if (n_tasks <= 0) return;
  const int padd = PV(a.pos_add, D);
  uint64_t* bar = (uint64_t*)(smem + AT_BAR(2));
  int* ctl = (int*)(bar + 7);
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < 7; i++) mbar_init(bar + i, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < 1024; i += NWG * 128) {   // bf16 1.0
    ((unsigned short*)(smem + AT_V(2) + 32768))[i] = 0x3f80; ((unsigned short*)(smem + AT_V(2) + AT_VBUF + 32768))[i] = 0x3f80;
  }
  unsigned ph[7] = {0, 0, 0, 0, 0, 0, 0};                       // completions of each barrier so far (every thread counts alike)
#define phk0 ph[0]
#define phk1 ph[1]
#define phv0 ph[2]
#define phv1 ph[3]
#define phq ph[4]
  const unsigned char* qs = smem + AT_Q(2) + (NWG == 1 ? 0 : wg) * 16384;
  for (;;) {
    if (tid == 0) ctl[0] = atomicAdd(a.ctr, 1);
    __syncthreads();
    const int task = ctl[0];
    if (task >= n_tasks) break;
    const int bt = D.plan[PL_BT + task0 + task];
    const int u0 = bt & 0xfff, ss = (bt >> 16) & 0x3f, g = bt >> 22;
    // the group: positions up to gtmax of one sequence
    const int ul = u0 + NWG - 1;
    const int gcol = D.plan[PL_UNIT + 4 * ul];
    const int slot = D.plan[PL_SLOT + gcol] & 0xffff;
    const int gtmax = D.plan[PL_POS + gcol] + padd + D.plan[PL_UNIT + 4 * ul + 1] - 1;
    const int s0 = ss * SS_SEGS, s1 = min((gtmax >> 7) + 1, s0 + SS_SEGS);
    // this warpgroup's unit
    const int unit = u0 + (NWG == 1 ? 0 : wg);
    const int col0 = D.plan[PL_UNIT + 4 * unit], nq = D.plan[PL_UNIT + 4 * unit + 1], pbase = D.plan[PL_UNIT + 4 * unit + 2];
    const int pos0 = D.plan[PL_POS + col0] + padd;
    const int pA = 4 * t2 + (g8 >> 2), pB = pA + 2;
    const int tA = pA < nq ? pos0 + pA : -1, tB = pB < nq ? pos0 + pB : -1;
    if (tid == 0) {
      mbar_expect(bar + 4, NWG * 16384);
#pragma unroll
      for (int w = 0; w < NWG; w++) bulk_g2s(smem + AT_Q(2) + w * 16384, D.qfrag + ((size_t)(u0 + w) * 8 + g) * 8192, 16384, bar + 4);
      // K(s0), K(s0 + 1), V(s0)
#pragma unroll
      for (int e = 0; e < 2; e++) {
        if (s0 + e < s1) {
          const unsigned short* Kn = page_k(D, slot, (s0 + e) * SEG, a.layer, g) + (size_t)((s0 + e) & 3) * 16384;
          mbar_expect(bar + e, 32768);
          bulk_g2s(smem + AT_K + e * 32768, Kn, 32768, bar + e);
          if (e == 0) { mbar_expect(bar + 2, 32768); bulk_g2s(smem + AT_V(2), Kn + (size_t)PG_U4 * 8, 32768, bar + 2); }
        }
      }
    }
    float O[68], scA[64], scB[64];          // O[64 .. 67]: Z of row A (twice) and of row B (twice)
#pragma unroll
    for (int i = 0; i < 64; i++) { O[i] = 0.f; scA[i] = 0.f; scB[i] = 0.f; }
    O[64] = 0.f; O[65] = 0.f; O[66] = 0.f; O[67] = 0.f;
    float MA = NINF, MB = NINF;
    mbar_wait(bar + 4, phq & 1); phq++;
    mbar_wait(bar, phk0 & 1); phk0++;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int j = 0; j < 8; j++) WGMMA_N128_P(scA, wg_desc2(qs + j * 256, 128), wg_desc2(smem + AT_K + j * 256, 128), j ? 1 : 0);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if (TRACK_A) {      // q is stored unscaled: the scores get log2(e) / sqrt(128) in f32
#pragma unroll
      for (int i = 0; i < 64; i++) scA[i] = __fmul_rn(scA[i], SC2);
    }
    // one segment: CUR holds its scores, NXT receives those of the next one (two register sets, used in turn)
    /* A: the scores of the next segment */
#define ATT_A(NXT, s, W) do { \
      if (!(ATT_ABL & 1)) { if (b) { mbar_wait(bar, phk0 & 1); phk0++; } else { mbar_wait(bar + 1, phk1 & 1); phk1++; } } \
      const unsigned char* Kn = smem + AT_K + (b ^ 1) * 32768; \
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); \
      _Pragma("unroll") \
      for (int j = 0; j < 8; j++) { if (!(ATT_ABL & 4)) WGMMA_N128_P(NXT, wg_desc2(qs + j * 256, 128), wg_desc2(Kn + j * 256, 128), j ? 1 : 0); } \
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); \
      if (W) { \
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); \
        if (TRACK_A) { _Pragma("unroll") for (int i = 0; i < 64; i++) NXT[i] = __fmul_rn(NXT[i], SC2); } \
      } \
    } while (0)
#if ATT_ROT
    // The two warpgroups of a block take the three phases of a segment in different orders: warpgroup 0 scores of
    // s + 1, scalar work of s, values of s; warpgroup 1 scalar, values, scores. The tensor cores of an SM serve two
    // warpgroups at 72 ns per instruction each and one alone at 43 ns, and they idle while both do scalar work:
    // this way one warpgroup's instructions run while the other is in its scalar part.
#define ATT_STEP(CUR, NXT, s) do { \
      const int b = ((s) - s0) & 1; \
      const bool more = (s) + 1 < s1; \
      /* ---- B: everything of the previous segment is done; the copies of V(s+1) and K(s+2) may start */ \
      __syncthreads(); \
      if (tid == 0 && !(ATT_ABL & 1)) { \
        if ((s) + 1 < s1) { \
          const unsigned short* Kp = page_k(D, slot, ((s) + 1) * SEG, a.layer, g) + (size_t)(((s) + 1) & 3) * 16384; \
          mbar_expect(bar + 2 + (b ^ 1), 32768); \
          bulk_g2s(smem + AT_V(2) + (b ^ 1) * AT_VBUF, Kp + (size_t)PG_U4 * 8, 32768, bar + 2 + (b ^ 1)); \
        } \
        if ((s) + 2 < s1) { \
          const unsigned short* Kp = page_k(D, slot, ((s) + 2) * SEG, a.layer, g) + (size_t)(((s) + 2) & 3) * 16384; \
          mbar_expect(bar + b, 32768); \
          bulk_g2s(smem + AT_K + b * 32768, Kp, 32768, bar + b); \
        } \
      } \
      if (more && !(NWG == 2 && wg == 1)) ATT_A(NXT, s, !ATT_ASYNC); /* ATT_ASYNC: in flight during the scalar work */ \
      /* ---- C: scalar work of segment s */ \
      const int limA = tA - (s) * SEG, limB = tB - (s) * SEG; \
      const int gnpt = min(8, ((gtmax - (s) * SEG) >> 4) + 1);          /* groups of 16 positions that exist in the sequence */ \
      float mA, mB; \
      const bool full = pos0 >= (s) * SEG + SEG - 1; \
      if (ATT_ABL & 2) { mA = CUR[0]; mB = CUR[2]; } \
      else if (full) seg_max<true>(CUR, 0, 0, t, mA, mB); else seg_max<false>(CUR, limA, limB, t, mA, mB); \
      const float RA = fmaxf(MA, mA), RB = fmaxf(MB, mB); \
      const float faA = (MA == RA) ? 1.0f : (MA == NINF ? 0.0f : ex2(__fsub_rn(MA, RA))); \
      const float faB = (MB == RB) ? 1.0f : (MB == NINF ? 0.0f : ex2(__fsub_rn(MB, RB))); \
      /* a factor of exactly 1 changes nothing: skip the multiplies when no row of the warp has a new maximum */ \
      if (__any_sync(0xffffffffu, MA != RA || MB != RB)) { \
        _Pragma("unroll") \
        for (int i = 0; i < 68; i++) O[i] = __fmul_rn(O[i], (i & 2) ? faB : faA); \
      } \
      MA = RA; MB = RB; \
      uint4 pa[8]; \
      if (ATT_ABL & 2) { \
        _Pragma("unroll") \
        for (int jv = 0; jv < 8; jv++) pa[jv] = make_uint4(__float_as_uint(CUR[8 * jv]), __float_as_uint(CUR[8 * jv + 2]), 0x3f803f80u, 0x3f803f80u); \
      } else if (full) { \
        _Pragma("unroll") \
        for (int jv = 0; jv < 8; jv++) pa[jv] = seg_weights<true>(CUR, jv, 0, 0, t, RA, RB); \
      } else { \
        _Pragma("unroll") \
        for (int jv = 0; jv < 8; jv++) pa[jv] = seg_weights<false>(CUR, jv, limA, limB, t, RA, RB); \
      } \
      /* ---- D: values: O += W V over the 16 dim groups, and against the group of ones behind them: Z += W 1 */ \
      if (!(ATT_ABL & 1)) { if (b) { mbar_wait(bar + 3, phv1 & 1); phv1++; } else { mbar_wait(bar + 2, phv0 & 1); phv0++; } } \
      const unsigned char* Vs = smem + AT_V(2) + b * AT_VBUF; \
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); \
      _Pragma("unroll") \
      for (int jv = 0; jv < 8; jv++) { if (!(ATT_ABL & 4)) WGMMA_RS_N136(O, pa[jv], wg_desc2(Vs + min(jv, gnpt - 1) * 256, 128)); } \
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); \
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); \
      if (ATT_ASYNC && TRACK_A && more && !(NWG == 2 && wg == 1)) { _Pragma("unroll") for (int i = 0; i < 64; i++) NXT[i] = __fmul_rn(NXT[i], SC2); } \
      if (more && NWG == 2 && wg == 1) ATT_A(NXT, s, 1); \
    } while (0)
#else
#error "the lockstep order of the two warpgroups is in git history (before ATT_ROT)"
#endif
    for (int s = s0; s < s1; s += 2) {
      ATT_STEP(scA, scB, s);
      if (s + 1 < s1) ATT_STEP(scB, scA, s + 1);
    }
#undef ATT_STEP
#undef ATT_A
#undef phk0
#undef phk1
#undef phv0
#undef phv1
#undef phq
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if (pbase < 0) att_out(D, O, O[64], O[66], col0, nq, g, t2, lane);
    else {
      float4* P = D.att_part + ((size_t)(pbase + ss) * 8 + g) * (17 * 128) + (tid & 127);
#pragma unroll
      for (int i = 0; i < 16; i++) P[i * 128] = make_float4(O[4 * i], O[4 * i + 1], O[4 * i + 2], O[4 * i + 3]);
      P[16 * 128] = make_float4(MA, MB, O[64], O[66]);
    }
  }
}

// ---- pairs of units, ping-pong (ATT_PP; FlashAttention-3's schedule). Per warpgroup and segment s:
//   G(s): values of s (O += P(s) V(s)) and scores of s + 1 (Q K(s+1)), issued together; wait for them
//   S(s+1): scalar work on the scores of s + 1 (maxima, rescale of O, the weights P(s+1))
// The two warpgroups take turns to issue their G (named barriers 1 and 2), so that one's S runs while the
// other's instructions occupy the tensor cores. No block barrier per segment: a K or V buffer is refilled by
// whichever warpgroup releases it second (a counter in shared memory). The scores of a segment past the last are
// computed on whatever the buffer holds and dropped (no condition around a warpgroup MMA: C7519).
// Same instructions on the same operands per row as k_attn<2>: the same bits.
#ifndef ATT_PP
#define ATT_PP 0
#endif
#define PP_THREADS 256
#ifndef PP_QREG
#define PP_QREG 1
#endif
__global__ void __launch_bounds__(PP_THREADS, 1) k_attn_pp(AttArgs a) {
  pdl_wait();
  extern __shared__ __align__(128) unsigned char smem[];
  const Dev& D = a.d;
  const int tid = threadIdx.x, lane = tid & 31, wg = tid >> 7, t2 = (tid >> 5) & 3, g8 = lane >> 2, t = lane & 3;
  const int n_tasks = HV(a.n_tasks, D.plan), task0 = HV(a.task0, D.plan);
  if (n_tasks <= 0) return;
  const int padd = PV(a.pos_add, D);
  uint64_t* bar = (uint64_t*)(smem + AT_BAR(2));                // 0, 1 K full; 2, 3 V full; 4 q full
  int* ctl = (int*)(bar + 7);                                   // [2] the task (two cells used in turn)
  int* rel = (int*)(bar + 8);                                   // [4] releases of K0, K1, V0, V1 so far
  if (tid == 0) {
    for (int i = 0; i < 5; i++) mbar_init(bar + i, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 4) rel[tid] = 0;
  for (int i = tid; i < 1024; i += PP_THREADS) {   // bf16 1.0
    ((unsigned short*)(smem + AT_V(2) + 32768))[i] = 0x3f80; ((unsigned short*)(smem + AT_V(2) + AT_VBUF + 32768))[i] = 0x3f80;
  }
  unsigned ph[5] = {0, 0, 0, 0, 0};      // completions of the full barriers so far
  const unsigned char* qs = smem + AT_Q(2) + wg * 16384;
  const bool elect = (tid & 127) == 0;
  for (int it = 0;; it++) {
    if (tid == 0) ctl[it & 1] = atomicAdd(a.ctr, 1);
    __syncthreads();
    const int task = ctl[it & 1];
    if (task >= n_tasks) break;
    const int bt = D.plan[PL_BT + task0 + task];
    const int u0 = bt & 0xfff, ss = (bt >> 16) & 0x3f, g = bt >> 22;
    const int ul = u0 + 1;
    const int gcol = D.plan[PL_UNIT + 4 * ul];
    const int slot = D.plan[PL_SLOT + gcol] & 0xffff;
    const int gtmax = D.plan[PL_POS + gcol] + padd + D.plan[PL_UNIT + 4 * ul + 1] - 1;
    const int s0 = ss * SS_SEGS, s1 = min((gtmax >> 7) + 1, s0 + SS_SEGS);
#define PP_KPTR(sx) (page_k(D, slot, (sx) * SEG, a.layer, g) + (size_t)((sx) & 3) * 16384)
#define PP_LOADK(sx, bb) do { mbar_expect(bar + (bb), 32768); bulk_g2s(smem + AT_K + (bb) * 32768, PP_KPTR(sx), 32768, bar + (bb)); } while (0)
#define PP_LOADV(sx, bb) do { mbar_expect(bar + 2 + (bb), 32768); bulk_g2s(smem + AT_V(2) + (bb) * AT_VBUF, PP_KPTR(sx) + (size_t)PG_U4 * 8, 32768, bar + 2 + (bb)); } while (0)
    // released by this warpgroup; refilled with segment sx if the other one has released it too
#define PP_REL(ci, sx, isv, bb) do { if (elect) { __threadfence_block(); \
      if (atomicAdd(rel + (ci), 1) & 1) { __threadfence_block(); if (isv) PP_LOADV(sx, bb); else PP_LOADK(sx, bb); } } } while (0)
    if (tid == 0) {
      mbar_expect(bar + 4, 2 * 16384);
      for (int w = 0; w < 2; w++) bulk_g2s(smem + AT_Q(2) + w * 16384, D.qfrag + ((size_t)(u0 + w) * 8 + g) * 8192, 16384, bar + 4);
      for (int e = 0; e < 2 && s0 + e < s1; e++) { PP_LOADK(s0 + e, e); PP_LOADV(s0 + e, e); }
    }
    const int unit = u0 + wg;
    const int col0 = D.plan[PL_UNIT + 4 * unit], nq = D.plan[PL_UNIT + 4 * unit + 1], pbase = D.plan[PL_UNIT + 4 * unit + 2];
    const int pos0 = D.plan[PL_POS + col0] + padd;
    const int pA = 4 * t2 + (g8 >> 2), pB = pA + 2;
    const int tA = pA < nq ? pos0 + pA : -1, tB = pB < nq ? pos0 + pB : -1;
    float O[68], sc[64];
#pragma unroll
    for (int i = 0; i < 64; i++) { O[i] = 0.f; sc[i] = 0.f; }
    O[64] = 0.f; O[65] = 0.f; O[66] = 0.f; O[67] = 0.f;
    float MA = NINF, MB = NINF;
    uint4 pa[8];
    // scores of s0 (both warpgroups at once: once per task)
    mbar_wait(bar + 4, ph[4] & 1); ph[4]++;
#if PP_QREG
    // q of this warpgroup's 64 rows as register operands (the scores' A side): the tensor cores then read only K
    // from shared memory (16 KB less per segment and warpgroup)
    uint4 qa[8];
    {
      const unsigned short* q16 = (const unsigned short*)qs;
      const int r0 = 16 * t2 + g8, r1 = r0 + 8;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int k0 = 16 * j + 2 * t;
        qa[j] = make_uint4(*(const unsigned*)(q16 + qidx(r0, k0)), *(const unsigned*)(q16 + qidx(r1, k0)),
                           *(const unsigned*)(q16 + qidx(r0, k0 + 8)), *(const unsigned*)(q16 + qidx(r1, k0 + 8)));
      }
    }
#define PP_SCORES(KB) do { _Pragma("unroll") for (int j = 0; j < 8; j++) WGMMA_RS_N128_P(sc, qa[j], wg_desc2((KB) + j * 256, 128), j ? 1 : 0); } while (0)
#else
#define PP_SCORES(KB) do { _Pragma("unroll") for (int j = 0; j < 8; j++) WGMMA_N128_P(sc, wg_desc2(qs + j * 256, 128), wg_desc2((KB) + j * 256, 128), j ? 1 : 0); } while (0)
#endif
    mbar_wait(bar, ph[0] & 1); ph[0]++;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    PP_SCORES(smem + AT_K);
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if (TRACK_A) {
#pragma unroll
      for (int i = 0; i < 64; i++) sc[i] = __fmul_rn(sc[i], SC2);
    }
    if (s0 + 2 < s1) PP_REL(0, s0 + 2, false, 0);                // K(s0) -> K(s0 + 2)
    if (wg == 1) asm volatile("bar.arrive 1, 256;" ::: "memory");   // warpgroup 0 issues first
    /* S(sx): the scalar work of segment sx on the scores in sc: maxima, rescale of O, the weights pa */
#define PP_S(sx) do { \
      const int limA = tA - (sx) * SEG, limB = tB - (sx) * SEG; \
      float mA, mB; \
      const bool full = pos0 >= (sx) * SEG + SEG - 1; \
      if (full) seg_max<true>(sc, 0, 0, t, mA, mB); else seg_max<false>(sc, limA, limB, t, mA, mB); \
      const float RA = fmaxf(MA, mA), RB = fmaxf(MB, mB); \
      const float faA = (MA == RA) ? 1.0f : (MA == NINF ? 0.0f : ex2(__fsub_rn(MA, RA))); \
      const float faB = (MB == RB) ? 1.0f : (MB == NINF ? 0.0f : ex2(__fsub_rn(MB, RB))); \
      if (__any_sync(0xffffffffu, MA != RA || MB != RB)) { \
        _Pragma("unroll") \
        for (int i = 0; i < 68; i++) O[i] = __fmul_rn(O[i], (i & 2) ? faB : faA); \
      } \
      MA = RA; MB = RB; \
      if (full) { \
        _Pragma("unroll") \
        for (int jv = 0; jv < 8; jv++) pa[jv] = seg_weights<true>(sc, jv, 0, 0, t, RA, RB); \
      } else { \
        _Pragma("unroll") \
        for (int jv = 0; jv < 8; jv++) pa[jv] = seg_weights<false>(sc, jv, limA, limB, t, RA, RB); \
      } \
    } while (0)
    PP_S(s0);
    for (int s = s0; s < s1; s++) {
      const int b = (s - s0) & 1;
      const bool more = s + 1 < s1;
      // ---- G(s): values of s, scores of s + 1, in turn with the other warpgroup
      mbar_wait(bar + 2 + b, ph[2 + b] & 1); ph[2 + b]++;                       // V(s)
      if (more) { mbar_wait(bar + (b ^ 1), ph[b ^ 1] & 1); ph[b ^ 1]++; }       // K(s + 1)
      if (wg == 0) asm volatile("bar.sync 1, 256;" ::: "memory"); else asm volatile("bar.sync 2, 256;" ::: "memory");
      const int gnpt = min(8, ((gtmax - s * SEG) >> 4) + 1);
      const unsigned char* Vs = smem + AT_V(2) + b * AT_VBUF;
      const unsigned char* Kn = smem + AT_K + (b ^ 1) * 32768;
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int jv = 0; jv < 8; jv++) WGMMA_RS_N136(O, pa[jv], wg_desc2(Vs + min(jv, gnpt - 1) * 256, 128));
      PP_SCORES(Kn);
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      // the other warpgroup's turn (warpgroup 1 does not hand over after its last G: the counts stay even)
      if (wg == 0) asm volatile("bar.arrive 2, 256;" ::: "memory"); else if (more) asm volatile("bar.arrive 1, 256;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      WG_TOUCH_N128(sc);
      if (s + 2 < s1) PP_REL(2 + b, s + 2, true, b);                            // V(s) -> V(s + 2)
      if (s + 3 < s1) PP_REL(b ^ 1, s + 3, false, b ^ 1);                       // K(s + 1) -> K(s + 3)
      if (more) {
        if (TRACK_A) {
#pragma unroll
          for (int i = 0; i < 64; i++) sc[i] = __fmul_rn(sc[i], SC2);
        }
        PP_S(s + 1);
      }
    }
#undef PP_S
#undef PP_SCORES
#undef PP_REL
#undef PP_LOADK
#undef PP_LOADV
#undef PP_KPTR
    if (pbase < 0) att_out(D, O, O[64], O[66], col0, nq, g, t2, lane);
    else {
      float4* P = D.att_part + ((size_t)(pbase + ss) * 8 + g) * (17 * 128) + (tid & 127);
#pragma unroll
      for (int i = 0; i < 16; i++) P[i * 128] = make_float4(O[4 * i], O[4 * i + 1], O[4 * i + 2], O[4 * i + 3]);
      P[16 * 128] = make_float4(MA, MB, O[64], O[66]);
    }
  }
}

// ---- the decode rows' kernel: a task = (row, kv head, super-segment), three warpgroups (att_dec)
__global__ void __launch_bounds__(AD_THREADS, 1) k_attn_dec(AttArgs a) {
  extern __shared__ __align__(128) unsigned char smem[];
  const Dev& D = a.d;
  const int tid = threadIdx.x;
  const int n_tasks = D.plan[PL_HDR + 6], task0 = HV(a.task0, D.plan);
  // set-up and the plan (older than the kernel before) under the kernel before; q, K, V after the wait. A block
  // that ends early waits too (else its successor could run alongside its predecessor)
  if (n_tasks <= 0) { if (blockIdx.x == 0) pdl_wait(); return; }
  const int padd = PV(a.pos_add, D);
  uint64_t* bar = (uint64_t*)(smem + AT_BAR(1));
  int* ctl = (int*)(bar + 7);
  // With a block per task (step(), generate(): the grid is the number of tasks) there is no counter: a round
  // trip to it before the first task and one after the last, ~1 us each per layer. The plan of the first task
  // is read before the set-up, so that its chain of dependent loads runs under it.
  const bool stat = (int)gridDim.x >= n_tasks;
  int task = blockIdx.x;
  // (the device loop's grid is the number of SMs.) A block without a task leaves at once: block 0 has one and
  // waits, so the kernel still cannot end before the one before it; the idle blocks do not hold their SMs.
  if (stat && task >= n_tasks) return;
  int bt = stat ? D.plan[PL_BT + task0 + task] : 0;
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < 7; i++) mbar_init(bar + i, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < 1024; i += AD_THREADS) {
#pragma unroll
    for (int e = 0; e < 3; e++) ((unsigned short*)(smem + AT_V(1) + e * AT_VBUF + 32768))[i] = 0x3f80;   // bf16 1.0
    ((unsigned short*)(smem + AT_W(1)))[i] = 0;                    // the weights of columns 4..7 stay 0
  }
  pdl_wait();
  if (!stat && tid == 0) ctl[0] = atomicAdd(a.ctr, 1);
  __syncthreads();
  unsigned ph[7] = {0, 0, 0, 0, 0, 0, 0};       // completions of each barrier so far (every thread counts alike)
  for (;;) {
    if (!stat) {
      task = ctl[0];
      if (task >= n_tasks) break;
      bt = D.plan[PL_BT + task0 + task];
    }
    const int unit = bt & 0xfff, ss = (bt >> 16) & 0x3f, g = bt >> 22;
    const int col0 = D.plan[PL_UNIT + 4 * unit], pbase = D.plan[PL_UNIT + 4 * unit + 2];
    const int slot = D.plan[PL_SLOT + col0] & 0xffff, pos = D.plan[PL_POS + col0] + padd;
    const int s0 = ss * SS_SEGS, s1 = min((pos >> 7) + 1, s0 + SS_SEGS);
    att_dec(D, a.layer, smem, bar, ph, slot, g, s0, s1, pos, col0, pbase, ss, unit);
    if (stat) break;
    __syncthreads();               // everybody has read ctl[0]
    if (tid == 0) ctl[0] = atomicAdd(a.ctr, 1);
    __syncthreads();
  }
}

// ---- fold of the super-segments of a unit, in order: 128 threads per (unit, kv head)
struct FoldArgs { Dev d; int n_fold; int pos_add; };
__global__ void __launch_bounds__(A_THREADS, 1) k_attn_fold(FoldArgs a) {
  pdl_wait();
  const Dev& D = a.d;
  const int tid = threadIdx.x, lane = tid & 31, wg = tid >> 7, t2 = (tid >> 5) & 3;
  const int total = HV(a.n_fold, D.plan) * 8, padd = PV(a.pos_add, D);
  for (int task = blockIdx.x * 2 + wg; task < total; task += gridDim.x * 2) {
    const int unit = D.plan[PL_FOLD + (task >> 3)], g = task & 7;
    const int col0 = D.plan[PL_UNIT + 4 * unit], nq = D.plan[PL_UNIT + 4 * unit + 1], pbase = D.plan[PL_UNIT + 4 * unit + 2];
    const int nss = ((D.plan[PL_POS + col0] + padd + nq - 1) >> SS_SHIFT) + 1;
    // a thread holds positions pA and pA + 2 of the unit: with neither in the unit (all but 16 threads for a decode
    // row) it has nothing to fold, and its part entries were not written in this pass (they would come from HBM)
    if (4 * t2 + ((lane >> 2) >> 2) >= nq) continue;
    float O[64];
#pragma unroll
    for (int i = 0; i < 64; i++) O[i] = 0.f;
    float MA = NINF, MB = NINF, ZA = 0.f, ZB = 0.f;
    for (int ss = 0; ss < nss; ss++) {
      const float4* P = D.att_part + ((size_t)(pbase + ss) * 8 + g) * (17 * 128) + (tid & 127);
      float4 v[17];
#pragma unroll
      for (int i = 0; i < 17; i++) v[i] = __ldcg(P + i * 128);
      float faA, fbA, faB, fbB, Mn;
      fold_factors(MA, v[16].x, Mn, faA, fbA); MA = Mn; ZA = __fmaf_rn(ZA, faA, __fmul_rn(v[16].z, fbA));
      fold_factors(MB, v[16].y, Mn, faB, fbB); MB = Mn; ZB = __fmaf_rn(ZB, faB, __fmul_rn(v[16].w, fbB));
#pragma unroll
      for (int i = 0; i < 16; i++) {
        O[4 * i + 0] = __fmaf_rn(O[4 * i + 0], faA, __fmul_rn(v[i].x, fbA));
        O[4 * i + 1] = __fmaf_rn(O[4 * i + 1], faA, __fmul_rn(v[i].y, fbA));
        O[4 * i + 2] = __fmaf_rn(O[4 * i + 2], faB, __fmul_rn(v[i].z, fbB));
        O[4 * i + 3] = __fmaf_rn(O[4 * i + 3], faB, __fmul_rn(v[i].w, fbB));
      }
    }
    att_out(D, O, ZA, ZB, col0, nq, g, t2, lane);
  }
}

// ---------------------------------------------------------------------------------------------------------
// lm_head helpers: gather the selected columns' final-norm input; reduce the per-warp maxima to a token
// ---------------------------------------------------------------------------------------------------------
struct SelArgs { Dev d; int n_sel; long long* out64; int out_stride; int out_step; };
__global__ void __launch_bounds__(256) k_gather(SelArgs a) {
  pdl_wait();
  const Dev& D = a.d;
  const int n_sel = HV(a.n_sel, D.plan);
  for (int j = blockIdx.x; j < n_sel; j += gridDim.x) {
    const int col = D.plan[PL_SEL + 2 * j];
    for (int i = threadIdx.x; i < H / 8; i += 256)   // 16 bytes = 8 consecutive K elements of the column
      ((uint4*)D.lmfrag)[xidx(8 * i, j, NG_LM) >> 3] = __ldcg((const uint4*)D.xfrag + (xidx(8 * i, col, NG_X) >> 3));
    if (TRACK_A) { if (threadIdx.x == 0) D.lm_rn[j] = __ldcg(D.rn + col); }
    else if (threadIdx.x < 32) {   // the final norm's 1/rms of the column, as k_rnorm computes it (that kernel is skipped)
      const float4* q = (const float4*)(D.ssq + (size_t)col * 256) + threadIdx.x * 2;
      const float4 u = __ldcg(q), v = __ldcg(q + 1);
      float sm = ((u.x + u.y) + (u.z + u.w)) + ((v.x + v.y) + (v.z + v.w));
      sm = warp_sum(sm);
      if (threadIdx.x == 0) D.lm_rn[j] = rsqrtf(__fmaf_rn(sm, 1.0f / H, 1e-6f));
    }
  }
}
__global__ void __launch_bounds__(256) k_argmax(SelArgs a) {
  pdl_wait();
  const Dev& D = a.d;
  __shared__ float sv[8]; __shared__ int si[8];
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  // the LM head's blocks have 64 rows up to 64 columns and 128 rows beyond (gemm_shape(): the vocabulary is not a multiple of 192)
  const int n_sel = HV(a.n_sel, D.plan), ncand = n_sel <= 64 ? VOCAB / 16 : VOCAB / 32;
  for (int j = blockIdx.x; j < n_sel; j += gridDim.x) {
    float bv = NINF; int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < ncand; i += 256) {
      float v = __ldcg(D.bm_val + (size_t)i * n_sel + j); int ix = __ldcg(D.bm_idx + (size_t)i * n_sel + j);
      if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float v = __shfl_xor_sync(0xffffffffu, bv, o); int ix = __shfl_xor_sync(0xffffffffu, bi, o);
      if (v > bv || (v == bv && ix < bi)) { bv = v; bi = ix; }
    }
    if (lane == 0) { sv[w] = bv; si[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i = 1; i < 8; i++) if (sv[i] > bv || (sv[i] == bv && si[i] < bi)) { bv = sv[i]; bi = si[i]; }
      D.sel_tok[j] = bi;
      const int slot = D.plan[PL_SEL + 2 * j + 1];
      if (slot >= 0) D.last_tok[slot] = bi;
      if (a.out64) a.out64[(size_t)j * a.out_stride + (a.out_step >= 0 ? a.out_step : __ldcg(D.cell + 1))] = bi;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------------------------
// The scheduler kernel: one block at every pass boundary. All threads hash the pages of the prompts that arrived
// since the last pass; then one thread runs ds_step (dsched.h): requests leave and enter slots, pages are taken,
// adopted from the prefix cache, registered, evicted; the plan of the pass and the result header are written.
// The host does not see any of it: it copied prompts into the arena and will copy the result block out.
// ---------------------------------------------------------------------------------------------------------
#define NRES 256                         // result blocks kept on the device (the host has to read them before they wrap)
struct DsCtl { int tail; float now; float speed; int harvested; };   // written by the host (other stream): requests so far, its clock, cost-model correction, result blocks read
struct DsStatus { int done; int pad[3]; };                     // written by the device: steps finished
#define NBUCKET 32
struct SchedArgs { DsState* S; const DsReqHost* rh; int* plan; DsResult* res; const int* arena; const DsCtl* ctl; int cmd;
                   cudaGraphConditionalHandle hsw; int sw;   // sw: set the branch of the pass (see launch_step)
                   int nrun; int run_hi[NBUCKET]; };          // the buckets of columns with one set of GEMM shapes
__global__ void __launch_bounds__(256) k_sched(SchedArgs a) {
  pdl_wait();
  const int seen = a.S->seen, tail = __ldcg(&a.ctl->tail);   // the host writes it while kernels run: never from L1
  for (int q = seen; q < tail; q++) {
    const int ri = q % DS_RING;
    const int np = a.rh[ri].n_in / DS_PAGE;
    for (int k = threadIdx.x; k < np; k += blockDim.x) a.S->req[ri].chain[k] = ds_hash_page(a.arena + a.rh[ri].prompt_off + k * DS_PAGE);
  }
  __syncthreads();
  for (int q = seen + threadIdx.x; q < tail; q += blockDim.x) ds_chain(a.S->req[q % DS_RING].chain, a.rh[q % DS_RING].n_in / DS_PAGE);
  __syncthreads();
  if (threadIdx.x == 0) {
    const float sp = __ldcg(&a.ctl->speed);
    if (sp > 0.f) a.S->P.speed = sp;
    ds_step(*a.S, a.rh, a.plan, a.res[a.S->cnt[DS_C_STEPS] % NRES], (double)__ldcg(&a.ctl->now), tail, a.cmd);
    if (a.sw) {
      const int* h = a.plan + PL_HDR;
      const bool dec = h[0] > 0 && h[0] <= 64 && h[1] == 0 && h[2] == h[6];   // every attention task a unit of one position
      int v = 0;
      if (dec) v = h[0] <= SHAPE8_T ? 1 : h[0] <= 32 ? 2 : 3;
      else if (h[0] > 0) for (int i = 0; i < a.nrun; i++) if (h[0] <= a.run_hi[i]) { v = 4 + i; break; }
      cudaGraphSetConditional(a.hsw, v);
    }
  }
}
// End of a pass: its tokens go into its result block, the step is published, and the loop on the device continues
// while a request is waiting, running or arriving.
struct CommitArgs { const DsState* S; DsResult* res; const int* tok; DsStatus* status; const DsCtl* ctl; cudaGraphConditionalHandle handle; int in_graph; };
__global__ void k_commit(CommitArgs a) {
  pdl_wait();
  const int step = (int)a.S->cnt[DS_C_STEPS] - 1;
  DsResult& R = a.res[step % NRES];
  for (int j = threadIdx.x; j < R.n_sel; j += blockDim.x) R.tok[j] = __ldcg(a.tok + j);
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    a.status->done = step + 1;
    // the loop goes on while there is work, and stops (the host restarts it) if the host has not read the result
    // blocks for so long that they would be overwritten
    const bool work = R.n_running > 0 || R.n_queued > 0 || __ldcg(&a.ctl->tail) > a.S->seen;
    const bool room = step + 1 - __ldcg(&a.ctl->harvested) < NRES - 8;
    if (a.in_graph) cudaGraphSetConditional(a.handle, work && room ? 1 : 0);
  }
}
// tests: NaN into every page the scheduler returned to the free list in this step
__global__ void k_poison_freed(const DsState* S, uint4* kv) {
  pdl_wait();
  const int n = S->n_freed;
  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    unsigned short* p = (unsigned short*)((char*)kv + PAGE_BYTES * (size_t)S->freed[i]);
    for (size_t j = threadIdx.x; j < PAGE_BYTES / 2; j += blockDim.x) p[j] = 0x7fc0;
  }
}

// generate() on the device: the first input tokens become the slots' "last generated token"; after every pass the
// position and the step advance and the loop goes on until the last step
__global__ void k_settok(int* last_tok, const long long* tok64, int B) {
  pdl_wait();
  if (threadIdx.x < B) last_tok[threadIdx.x] = (int)tok64[threadIdx.x];
}
// the cells of a generate() / step() launch; the values travel with the launch (the host may queue many launches
// before the first one runs)
__global__ void k_setcell(int* cell, int pos, int step, int end) {
  pdl_wait();
  if (threadIdx.x == 0) { cell[0] = pos; cell[1] = step; cell[2] = end; }
}
__global__ void k_gen_commit(int* cell, cudaGraphConditionalHandle handle) {
  pdl_wait();
  if (threadIdx.x == 0) {
    cell[0] += 1;
    const int step = cell[1] + 1;
    cell[1] = step;
    cudaGraphSetConditional(handle, step < cell[2] ? 1 : 0);
  }
}

// test helper: fill a page with NaN (bf16 0x7fc0)
__global__ void k_fill16(unsigned short* p, size_t n, unsigned short v) {
  pdl_wait();
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = v;
}

// ---------------------------------------------------------------------------------------------------------
// host API
// ---------------------------------------------------------------------------------------------------------
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)

#ifndef ARENA_TOK
#define ARENA_TOK (16 << 20)             // prompt arena, tokens
#endif
struct Eng {
  Dev D;
  int nsm, n_pages;
  int* plan_dev;
  long long passes, launches, plan_uploads, tok_reads;
  int force_shape;
  // device scheduler
  DsState* S; DsReqHost* rh; DsResult* res; int* arena; DsCtl* ctl; DsStatus* status; int* tok_cur;     // device
  DsReqHost* rh_host; DsResult* res_host; int* arena_host; DsCtl* ctl_host; DsStatus* status_host;      // pinned host mirrors
  cudaStream_t sA, sB;                   // sA: the serving loop; sB: the host's copies (prompts in, results out)
  cudaGraph_t graph; cudaGraphExec_t exec; cudaGraphConditionalHandle hwhile, hsw;
  long long tail, arena_pos, harvested, graph_launches, host_launched, sched_only;
  int poison, use_graph;
  // generate(): the loop over the steps as a graph (rebuilt when batch or plan counts change)
  cudaGraph_t ggraph; cudaGraphExec_t gexec; cudaGraphConditionalHandle ghandle; int gkey[4]; long long* gen_buf; long long gen_launches;
  // step(): one pass as a graph; the token pointer of its embedding node and the logits pointer of its LM-head node
  // are set per call
  cudaGraph_t sgraph; cudaGraphExec_t sexec; int skey[4]; long long step_launches;
  cudaGraphNode_t s_embed, s_lm; cudaKernelNodeParams s_embed_kp, s_lm_kp; EmbedArgs s_ea; GemmArgs s_lmg;
  cudaGraphNode_t c_embed, c_lm; cudaKernelNodeParams c_embed_kp, c_lm_kp; EmbedArgs c_ea; GemmArgs c_lmg;   // of the graph built last
  // launching: directly, or as nodes of the graph being built
  cudaGraph_t gb; cudaGraphNode_t glast; int ghave; int gcond; cudaStream_t cur;
  int bk_lo, bk_hi;          // while a bucket branch of the serving loop is built: its range of columns (0: none)
  cudaKernelNodeParams last_kp; GemmArgs last_gemm;     // of the node added last / the GEMM launched last
};
// one kernel of a pass: launched on the current stream, or appended to the graph under construction
template <class A>
static void emit(Eng* E, void (*fn)(A), int grid, int block, size_t smem, A args) {
  if (!E->gb) { fn<<<grid, block, smem, E->cur>>>(args); return; }
  cudaKernelNodeParams kp = {};
  void* argv[1] = {&args};
  kp.func = (void*)fn; kp.gridDim = dim3(grid); kp.blockDim = dim3(block); kp.sharedMemBytes = (unsigned)smem; kp.kernelParams = argv;
  cudaGraphNode_t n;
  if (PDL && E->ghave && !E->gcond) {     // a programmatic edge from the kernel before (see pdl_wait)
    CK(cudaGraphAddKernelNode(&n, E->gb, nullptr, 0, &kp));
    cudaGraphEdgeData ed = {};
    ed.from_port = cudaGraphKernelNodePortProgrammatic; ed.to_port = 0; ed.type = cudaGraphDependencyTypeProgrammatic;
    CK(cudaGraphAddDependencies(E->gb, &E->glast, &n, &ed, 1));
  } else CK(cudaGraphAddKernelNode(&n, E->gb, E->ghave ? &E->glast : nullptr, E->ghave ? 1 : 0, &kp));
  E->glast = n; E->ghave = 1; E->gcond = 0; E->last_kp = kp;
}

static int g_split = 2, g_prepq = 1, g_light = 1;
template <int EP, int NB, int NT>
static void emit1(Eng* E, int grid, const GemmArgs& g) {
  static bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(k_gemm1<EP, NB, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, WG_SMEM(NB, NT))); attr = true; }
  emit<GemmArgs>(E, k_gemm1<EP, NB, NT>, grid, WG_THREADS, WG_SMEM(NB, NT), g);
}
template <int EP, int NT>
static void emit1pb(Eng* E, int grid, const GemmArgs& g) {
  static bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(k_gemm1<EP, 8, NT, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, WG_SMEM_PB(NT))); attr = true; }
  emit<GemmArgs>(E, k_gemm1<EP, 8, NT, 1>, grid, WG_THREADS, WG_SMEM_PB(NT), g);
}
// One GEMM. Launched by the host for a known number of columns: the shape is chosen here and the grid is no larger
// than the number of units. In the serving loop on the device the kernel reads the number of columns from the plan
// and chooses the shape itself.
// a per-shape GEMM kernel for `shape`, grid blocks
template <int EP>
static void emit_shape(Eng* E, int shape, int grid, const GemmArgs& g) {
  if (shape == 2) emit1<EP, 64, 1>(E, grid, g);
  else if (shape == 3) emit1<EP, 64, 2>(E, grid, g);
  else if (shape == 4) { if constexpr (EP != EP_LM && EP != EP_LMARG) emit1<EP, 64, 3>(E, grid, g); }
  else if (shape == 5) { if constexpr (EP != EP_LM && EP != EP_LMARG) emit1<EP, 64, 4>(E, grid, g); }
  else if (shape == 6) emit1<EP, 128, 2>(E, grid, g);
  else if (shape == 0) emit1<EP, 16, 1>(E, grid, g);
  else if (shape == 1) emit1<EP, 32, 1>(E, grid, g);
  else emit1<EP, 8, 1>(E, grid, g);
}
template <int EP>
static void launch_gemm(Eng* E, GemmArgs g, bool dev = false, int light = 0) {
  int grid = E->nsm;
  g.shape = -1;
  if (dev && E->bk_hi > 0 && EP != EP_LM && EP != EP_LMARG) {
    // a bucket branch: every number of columns in [bk_lo, bk_hi] gets the same shape (gemm_shape), so the
    // per-shape kernel (columns from the plan); the grid covers the most units of the range
    const int sh = gemm_shape(g.M, E->bk_lo, E->nsm);
    int nb, nt; float us;
    shape_dims(sh, nb, nt, us);
    const int units = (g.M / (64 * nt)) * ((E->bk_hi + 2 * nb - 1) / (2 * nb));
    g.shape = sh;
    if (sh == 7 && EP == EP_QKV && g.prep) emit1<EP_QKVP, 8, 2>(E, QKV_ROWS / 128 < grid ? QKV_ROWS / 128 : grid, g);
    else emit_shape<EP>(E, sh, units < grid ? units : grid, g);
    E->last_gemm = g;
    E->launches++;
    return;
  }
  if (light) {   // the serving loop's light passes (columns from the plan, g.T < 0): 1 up to 16 columns, 2 up to 32, 3 up to 64
    g.shape = light == 1 ? 7 : light == 2 ? 0 : 1;
    const int gr = g.M / 64 < grid ? g.M / 64 : grid;
    if (light == 1 && EP == EP_QKV && g.prep) emit1<EP_QKVP, 8, 2>(E, QKV_ROWS / 128 < grid ? QKV_ROWS / 128 : grid, g);
    else if (light == 1) emit1<EP, 8, 1>(E, gr, g);
    else if (light == 2) emit1<EP, 16, 1>(E, gr, g);
    else emit1<EP, 32, 1>(E, gr, g);
    E->last_gemm = g;
    E->launches++;
    return;
  }
  if (TRACK_A && TA_PB && !dev && g.ta_in && (EP == EP_QKV || EP == EP_GU) && g.T <= TA_PB_T && g.T <= 8) {   // the norm inside (PB)
    g.shape = 7;
    if constexpr (TRACK_A && EP == EP_QKV) { if (!g.prep) __builtin_trap(); emit1pb<EP_QKVP, 2>(E, QKV_ROWS / 128 < grid ? QKV_ROWS / 128 : grid, g); }
    else if constexpr (TRACK_A && EP == EP_GU) emit1pb<EP_GU, 1>(E, g.M / 64 < grid ? g.M / 64 : grid, g);
    E->last_gemm = g;
    E->launches++;
    return;
  }
  if (!dev && EP == EP_QKV && g.prep && g.T <= 16) {   // one head per block, prepared in the epilogue
    g.shape = 7;
    emit1<EP_QKVP, 8, 2>(E, QKV_ROWS / 128 < grid ? QKV_ROWS / 128 : grid, g);
    E->last_gemm = g;
    E->launches++;
    return;
  }
  if (!dev) {
    int nb, nt; float us;
    g.shape = gemm_shape(g.M, g.T, E->nsm);
    shape_dims(g.shape, nb, nt, us);
    const int n_units = (g.M / (64 * nt)) * ((g.T + 2 * nb - 1) / (2 * nb));
    if (n_units < grid) grid = n_units;
  }
  // up to 64 columns (step(), generate()) the kernel with just that shape: the general one costs 0.8 us more per
  // launch (5.81 against 5.69 ms for the GEMMs of a decode step)
  if (!dev && g_split >= 2 && g.shape >= 2 && g.shape <= 6) emit_shape<EP>(E, g.shape, grid, g);   // per-shape kernels for the large shapes (2 - 6% faster)
  else if (!dev && g_split && g.shape == 0) emit1<EP, 16, 1>(E, grid, g);
  else if (!dev && g_split && g.shape == 7) emit1<EP, 8, 1>(E, grid, g);
  else if (!dev && g_split && g.shape == 1) emit1<EP, 32, 1>(E, grid, g);
  else emit<GemmArgs>(E, k_gemm<EP>, grid, WG_THREADS, WG_SMEM_MAX, g);
  E->last_gemm = g;
  E->launches++;
}
template <int EP>
static void gemm_attr() {
  CK(cudaFuncSetAttribute(k_gemm<EP>, cudaFuncAttributeMaxDynamicSharedMemorySize, WG_SMEM_MAX));
}

extern "C" {

void* eng_create(int n_pages, const void* W, const void* embed, const void* norm_w, const void* qk_norm_w, const void* rope) {
  if (getenv("MK_GEMM_SPLIT")) g_split = atoi(getenv("MK_GEMM_SPLIT"));
  if (getenv("MK_PREPQ")) g_prepq = atoi(getenv("MK_PREPQ"));
  if (getenv("MK_LIGHT")) g_light = atoi(getenv("MK_LIGHT"));
  Eng* E = new Eng();
  memset(E, 0, sizeof(Eng));
  int dev = 0; CK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  E->nsm = prop.multiProcessorCount;
  E->n_pages = n_pages;
  E->force_shape = getenv("MK_GEMM_NB") ? atoi(getenv("MK_GEMM_NB")) : -1;
  Dev& D = E->D;
  D.W = (const unsigned short*)W; D.embed = (const unsigned short*)embed; D.norm_w = (const float*)norm_w;
  D.qk_norm_w = (const float*)qk_norm_w; D.rope = (const float*)rope;
  CK(cudaMalloc(&D.kv, PAGE_BYTES * (size_t)n_pages));
  CK(cudaMalloc(&E->plan_dev, sizeof(int) * PL_INTS)); D.plan = E->plan_dev;
  CK(cudaMalloc(&D.x, sizeof(float) * (size_t)TCAP * H));
  CK(cudaMalloc(&D.xfrag, (size_t)TCAP * H * 2));
  CK(cudaMalloc(&D.ssq, sizeof(float) * (size_t)TCAP * 256));
  CK(cudaMalloc(&D.rn, sizeof(float) * TCAP));
  CK(cudaMalloc(&D.qkv, sizeof(float) * (size_t)TCAP * QKV_ROWS));
  CK(cudaMalloc(&D.qfrag, (size_t)MAXU * 8 * 16384));
  CK(cudaMalloc(&D.afrag, (size_t)TCAP * H * 2));
  CK(cudaMalloc(&D.mfrag, (size_t)TCAP * INNER * 2));
  CK(cudaMalloc(&D.lmfrag, (size_t)SELCAP * H * 2));
  CK(cudaMalloc(&D.lm_rn, sizeof(float) * SELCAP));
  CK(cudaMalloc(&D.bm_val, sizeof(float) * (size_t)(VOCAB / 16) * SELCAP));
  CK(cudaMalloc(&D.bm_idx, sizeof(int) * (size_t)(VOCAB / 16) * SELCAP));
  CK(cudaMalloc(&D.last_tok, sizeof(int) * NSLOT));
  CK(cudaMalloc(&D.sel_tok, sizeof(int) * SELCAP));
  CK(cudaMalloc(&D.att_part, sizeof(float4) * (size_t)PARTCAP * 8 * 17 * 128));
  CK(cudaMalloc(&D.cell, sizeof(int) * 4));
  CK(cudaMemset(D.cell, 0, sizeof(int) * 4));
  CK(cudaMalloc(&D.ctr, sizeof(int) * NL * 3));
  CK(cudaMemset(D.ctr, 0, sizeof(int) * NL * 3));
  CK(cudaMemset(D.kv, 0, PAGE_BYTES * (size_t)n_pages));
  CK(cudaMemset(D.xfrag, 0, (size_t)TCAP * H * 2));
  CK(cudaMemset(D.qfrag, 0, (size_t)MAXU * 8 * 16384));
  CK(cudaMemset(D.afrag, 0, (size_t)TCAP * H * 2));
  CK(cudaMemset(D.mfrag, 0, (size_t)TCAP * INNER * 2));
  CK(cudaMemset(D.lmfrag, 0, (size_t)SELCAP * H * 2));
  CK(cudaMemset(D.last_tok, 0, sizeof(int) * NSLOT));
  CK(cudaMemset(E->plan_dev, 0, sizeof(int) * PL_INTS));
  gemm_attr<EP_QKV>(); gemm_attr<EP_ADD>(); gemm_attr<EP_GU>(); gemm_attr<EP_LM>(); gemm_attr<EP_LMARG>(); gemm_attr<EP_RAW>();
  CK(cudaFuncSetAttribute(k_attn<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM(2)));
  CK(cudaFuncSetAttribute(k_attn_dec, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM(1)));
  CK(cudaFuncSetAttribute(k_attn<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM(2)));
  CK(cudaFuncSetAttribute(k_attn_pp, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM(2)));
  return E;
}

int eng_plan_ints() { return PL_INTS; }
int eng_tcap() { return TCAP; }
// plan layout for the host: 0 PL_UNIT, 1 (unused), 2 PL_BT, 3 PL_FOLD, 4 PL_SEL, 5 PL_PT, 6 MAXU, 7 (unused), 8 MAXBT, 9 PARTCAP, 10 PL_CU, 11 SS_SHIFT
int eng_const(int i) { static const int c[13] = {PL_UNIT, 0, PL_BT, PL_FOLD, PL_SEL, PL_PT, MAXU, 0, MAXBT, PARTCAP, PL_CU, SS_SHIFT, TRACK_A}; return c[i]; }
// bytes of device memory the engine allocates besides the KV pages
size_t eng_fixed_bytes() { return sizeof(float4) * (size_t)PARTCAP * 8 * 17 * 128 + (size_t)TCAP * (H * 4 + H * 2 * 2 + INNER * 2 + QKV_ROWS * 4 + 256 * 4) + (size_t)MAXU * 8 * 16384 + (size_t)(VOCAB / 16) * SELCAP * 8; }
int eng_selcap() { return SELCAP; }
size_t eng_page_bytes() { return PAGE_BYTES; }

// copy the plan (host ints, layout PL_*) to the device. `n` leading ints are copied.
void eng_set_plan(void* h, const int* plan, int off, int n) {
  Eng* E = (Eng*)h;
  CK(cudaMemcpy(E->plan_dev + off, plan + off, sizeof(int) * (size_t)n, cudaMemcpyHostToDevice));
  E->plan_uploads++;
}

// One pass over the weights for the T columns of the current plan.
//   pos_add: added to every column's position (step()/generate() reuse one plan)
//   tok64:   device int64 [T] input tokens, or null (tokens from the plan / the slots' last tokens)
//   logits:  device float [n_sel][VOCAB] or null (then only the argmax: sel_tok, last_tok, out64)
// The kernels of one pass. dev = false: the host knows the counts (offline contract, tests). dev = true: the plan
// was written by the scheduler kernel just before; the kernels read their counts from its header and are launched
// with full grids (blocks without work return at once).
static void launch_pass(Eng* E, bool dev, int T, int n_bt2, int n_bt1, int n_fold, int n_sel, int pos_add, const void* tok64, void* logits,
                        void* out64, int out_stride, int out_step, int* sel_tok, bool all_dec = false, int light = 0) {
  // light (dev only): a pass of at most 16 / 32 / 64 columns (1 / 2 / 3) whose attention tasks are all units of one
  // position (the serving loop's decode passes; k_sched picks the branch): per-shape GEMM kernels, no 1/rms kernel
  // (track B), no prompt attention kernels; up to 16 columns q/k/v prepared in the GEMM's epilogue (no k_prep)
  Dev D = E->D;   // all_dec: every column is a decode row (step(), generate())
  if (sel_tok) D.sel_tok = sel_tok;
  const int nsm = E->nsm;
  const int cT = dev ? -1 : T;
  EmbedArgs ea{D, cT, pos_add, (const long long*)tok64, dev ? E->arena : nullptr, -1};
  emit<EmbedArgs>(E, k_embed, light ? (light == 1 ? 16 : 64) : dev ? 4 * nsm : (T < 4 * nsm ? T : 4 * nsm), 256, 0, ea);
  if (E->gb) { E->c_embed = E->glast; E->c_embed_kp = E->last_kp; E->c_ea = ea; }
  RnArgs ra{D.ssq, D.rn, cT, D.plan, 0, D.x, D.norm_w + (size_t)2 * NL * H, D.xfrag}, ra_l{D.ssq, D.rn, cT, D.plan, 1, D.x, D.norm_w, D.xfrag};   // ra_l: the norms inside the layers
#ifndef TA_RN_BLOCKS
#define TA_RN_BLOCKS 1     // track A: blocks of k_rnorm per SM in the light passes
#endif
#ifndef TA_RN_BIG
#define TA_RN_BIG 1        // ... and in the others (4: 0.3 ms per 1024-column pass slower)
#endif
  const int ta_hi = dev ? (E->bk_hi > 0 && E->bk_hi < nsm * TA_RN_BIG ? (E->bk_hi > nsm ? E->bk_hi : nsm) : nsm * TA_RN_BIG) : (T < nsm * TA_RN_BIG ? T : nsm * TA_RN_BIG);
  const int rn_grid = TRACK_A ? (light ? nsm * TA_RN_BLOCKS : ta_hi) : dev ? nsm : ((T + 7) / 8 < nsm ? (T + 7) / 8 : nsm);
  const int ta_pb = TRACK_A && TA_PB && !dev && T <= TA_PB_T && TA_PB_T <= 8;   // QKV / GU normalize their own input (gemm_body PB)
  for (int l = 0; l < NL; l++) {
    const unsigned short* WL = D.W + (size_t)l * LAYER_U4 * 8;
    ra_l.w = D.norm_w + (size_t)(2 * l) * H;
    if (TRACK_A ? !(ta_pb && l > 0) : (dev && !light && !(E->bk_hi > 0 && E->bk_hi <= 64)) || T > 64) emit<RnArgs>(E, k_rnorm, rn_grid, 256, 0, ra_l);
    GemmArgs g{};
    g.W = WL + OFF_QKV * 8; g.X = D.xfrag; g.NG = NG_X; g.M = QKV_ROWS; g.K = H; g.T = cT; g.plan = D.plan; g.rn = D.rn; g.out = D.qkv;
    g.ssq_in = D.ssq;
    if (ta_pb && l > 0) { g.ta_in = 1; g.x = D.x; g.aux = D.norm_w + (size_t)(2 * l) * H; }
    g.nextW = WL + OFF_O * 8; g.nextM = H;
    g.dev = D; g.layer = l; g.pos_add = pos_add; g.prep = g_prepq;
    launch_gemm<EP_QKV>(E, g, dev, light);
    PrepArgs pa{D, cT, l, pos_add, g_prepq};
    const int pgrid = dev ? PREP_BLOCKS * nsm : ((T * 48 + 15) / 16 < PREP_BLOCKS * nsm ? (T * 48 + 15) / 16 : PREP_BLOCKS * nsm);
    if ((dev && light != 1 && !(E->bk_hi > 0 && E->bk_hi <= 16)) || T > 16 || !g_prepq) emit<PrepArgs>(E, k_prep, pgrid, 512, 0, pa);
    if ((dev && !light) || n_bt2 > 0) {   // tasks of two units: tasks 0 .. n_bt2 - 1 of the plan
      AttArgs aa{D, 0, dev ? -2 : n_bt2, l, pos_add, D.ctr + 3 * l, 0};
      if (ATT_PP) emit<AttArgs>(E, k_attn_pp, dev ? nsm : (n_bt2 < nsm ? n_bt2 : nsm), PP_THREADS, AT_SMEM(2), aa);
      else emit<AttArgs>(E, k_attn<2>, dev ? nsm : (n_bt2 < nsm ? n_bt2 : nsm), 256, AT_SMEM(2), aa);
    }
    // tasks of one unit, after them: first the decode rows (their own kernel), then the odd ends of prompt runs
    if (ATT_DEC && (dev || n_bt1 > 0)) {
      AttArgs aa{D, dev ? -2 : n_bt2, 0, l, pos_add, D.ctr + 3 * l + 1, 1};
      emit<AttArgs>(E, k_attn_dec, dev ? nsm : (n_bt1 < nsm ? n_bt1 : nsm), AD_THREADS, AT_SMEM(1), aa);
    }
    if ((dev && !light) || (n_bt1 > 0 && !(all_dec && ATT_DEC))) {
      AttArgs aa{D, dev ? -2 : n_bt2, dev ? -3 : n_bt1, l, pos_add, D.ctr + 3 * l + 2, ATT_DEC ? 2 : 0};
      emit<AttArgs>(E, k_attn<1>, dev ? nsm : (n_bt1 < nsm ? n_bt1 : nsm), 128, AT_SMEM(2), aa);
    }
    if (dev || n_fold > 0) {
      FoldArgs fa{D, dev ? -4 : n_fold, pos_add};
      emit<FoldArgs>(E, k_attn_fold, dev ? nsm : (n_fold * 4 < nsm ? n_fold * 4 : nsm), A_THREADS, 0, fa);
    }
    g = GemmArgs{};
    g.W = WL + OFF_O * 8; g.X = D.afrag; g.NG = NG_X; g.M = H; g.K = H; g.T = cT; g.plan = D.plan;
    g.x = D.x; g.aux = D.norm_w + (size_t)(2 * l + 1) * H; g.frag = D.xfrag; g.ssq = D.ssq;
    g.nextW = WL + OFF_GU * 8; g.nextM = 2 * INNER;
    launch_gemm<EP_ADD>(E, g, dev, light);
    ra_l.w = D.norm_w + (size_t)(2 * l + 1) * H;
    if (TRACK_A ? !ta_pb : (dev && !light && !(E->bk_hi > 0 && E->bk_hi <= 64)) || T > 64) emit<RnArgs>(E, k_rnorm, rn_grid, 256, 0, ra_l);
    g = GemmArgs{};
    g.W = WL + OFF_GU * 8; g.X = D.xfrag; g.NG = NG_X; g.M = 2 * INNER; g.K = H; g.T = cT; g.plan = D.plan; g.rn = D.rn; g.frag = D.mfrag;
    g.ssq_in = D.ssq;
    if (ta_pb) { g.ta_in = 1; g.x = D.x; g.aux = D.norm_w + (size_t)(2 * l + 1) * H; }
    g.nextW = WL + OFF_DN * 8; g.nextM = H;
    launch_gemm<EP_GU>(E, g, dev, light);
    g = GemmArgs{};
    g.W = WL + OFF_DN * 8; g.X = D.mfrag; g.NG = NG_X; g.M = H; g.K = INNER; g.T = cT; g.plan = D.plan;
    g.x = D.x; g.aux = D.norm_w + (size_t)(2 * l + 2) * H; g.frag = D.xfrag; g.ssq = D.ssq;
    if (l + 1 < NL) { g.nextW = D.W + (size_t)(l + 1) * LAYER_U4 * 8 + OFF_QKV * 8; g.nextM = QKV_ROWS; }
    else { g.nextW = D.W + (size_t)NL * LAYER_U4 * 8; g.nextM = VOCAB; }
    launch_gemm<EP_ADD>(E, g, dev, light);
    E->launches += 5;
  }
  if (dev || n_sel > 0) {
    if (TRACK_A) emit<RnArgs>(E, k_rnorm, rn_grid, 256, 0, ra);   // (track B: k_gather computes the 1/rms of its columns)
    SelArgs sa{D, dev ? -5 : n_sel, (long long*)out64, out_stride, out_step};
    emit<SelArgs>(E, k_gather, dev ? 64 : n_sel, 256, 0, sa);
    GemmArgs g{};
    g.W = D.W + (size_t)NL * LAYER_U4 * 8; g.X = D.lmfrag; g.NG = NG_LM; g.M = VOCAB; g.K = H; g.T = dev ? -5 : n_sel; g.plan = D.plan; g.rn = D.lm_rn;
    if (logits) {
      g.out = (float*)logits; launch_gemm<EP_LM>(E, g, dev);
      if (E->gb) { E->c_lm = E->glast; E->c_lm_kp = E->last_kp; E->c_lmg = E->last_gemm; }
    }
    else {
      g.bm_val = D.bm_val; g.bm_idx = D.bm_idx;
      launch_gemm<EP_LMARG>(E, g, dev, light);
      emit<SelArgs>(E, k_argmax, dev ? 64 : n_sel, 256, 0, sa);
    }
    E->launches += 3;
  }
  E->launches += 1;
  E->passes++;
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) fprintf(stderr, "launch_pass: launch error: %s\n", cudaGetErrorString(e));
}

// One pass over the weights for the T columns of the current (host-written) plan.
//   pos_add: added to every column's position (step()/generate() reuse one plan)
//   tok64:   device int64 [T] input tokens, or null (tokens from the plan / the slots' last tokens)
//   logits:  device float [n_sel][VOCAB] or null (then only the argmax: sel_tok, last_tok, out64)
void eng_pass(void* h, int T, int n_bt2, int n_bt1, int n_fold, int n_sel, int pos_add, const void* tok64, void* logits, void* out64, int out_stride, int out_step) {
  Eng* E = (Eng*)h;
  E->gb = nullptr; E->cur = 0;
  launch_pass(E, false, T, n_bt2, n_bt1, n_fold, n_sel, pos_add, tok64, logits, out64, out_stride, out_step, nullptr);
}

// step(): one pass for the B rows at position `pos` as ONE graph launch (190 kernel nodes) instead of 190 launches.
// The graph is built per (B, plan counts); per call the host writes the position cell and points the embedding
// node at the caller's tokens and the LM-head node at the caller's logits.
void eng_step(void* h, int B, int n_bt2, int n_bt1, int n_fold, int pos, const void* tok64, void* logits) {
  Eng* E = (Eng*)h;
  Dev& D = E->D;
  const int key[4] = {B, n_bt2, n_bt1, n_fold};
  if (!E->sexec || memcmp(key, E->skey, sizeof(key))) {
    if (E->sexec) { CK(cudaDeviceSynchronize()); CK(cudaGraphExecDestroy(E->sexec)); CK(cudaGraphDestroy(E->sgraph)); }
    CK(cudaGraphCreate(&E->sgraph, 0));
    E->gb = E->sgraph; E->ghave = 0;
    launch_pass(E, false, B, n_bt2, n_bt1, n_fold, B, -1, tok64, logits, nullptr, 0, 0, nullptr, true);
    E->gb = nullptr;
    E->s_embed = E->c_embed; E->s_embed_kp = E->c_embed_kp; E->s_ea = E->c_ea;
    E->s_lm = E->c_lm; E->s_lm_kp = E->c_lm_kp; E->s_lmg = E->c_lmg;
    CK(cudaGraphInstantiate(&E->sexec, E->sgraph, 0));
    memcpy(E->skey, key, sizeof(key));
  } else {
    void* argv[1];
    cudaKernelNodeParams kp = E->s_lm_kp;
    E->s_lmg.out = (float*)logits; argv[0] = &E->s_lmg; kp.kernelParams = argv;
    CK(cudaGraphExecKernelNodeSetParams(E->sexec, E->s_lm, &kp));
    E->passes++;
  }
  {   // the tokens and the position travel in the embedding node's arguments (no kernel of their own before the graph)
    void* argv[1];
    cudaKernelNodeParams kp = E->s_embed_kp;
    E->s_ea.tok64 = (const long long*)tok64; E->s_ea.cell_pos = pos; argv[0] = &E->s_ea; kp.kernelParams = argv;
    CK(cudaGraphExecKernelNodeSetParams(E->sexec, E->s_embed, &kp));
  }
  CK(cudaGraphLaunch(E->sexec, 0));
  E->step_launches++;
}

// generate(): `nrun` greedy steps from step `step0` of `n_total` at position `pos`, with the current (host-written)
// plan of B decode rows, as ONE graph launch: WHILE (step < end) { pass; position and step advance }. Argmax, token
// feedback and the loop are on the device; the host launches once per stretch of steps with the same plan (the
// plan changes when the rows enter a new super-segment of 4096 positions) and does not wait.
#define GEN_STRIDE 34816
void eng_generate(void* h, int B, int n_bt2, int n_bt1, int n_fold, int pos, const void* tok64, void* out64, int n_total, int step0, int nrun) {
  Eng* E = (Eng*)h;
  Dev& D = E->D;
  if (!E->gen_buf) CK(cudaMalloc(&E->gen_buf, sizeof(long long) * (size_t)NSLOT * GEN_STRIDE));
  const int key[4] = {B, n_bt2, n_bt1, n_fold};
  if (!E->gexec || memcmp(key, E->gkey, sizeof(key))) {
    if (E->gexec) { CK(cudaDeviceSynchronize()); CK(cudaGraphExecDestroy(E->gexec)); CK(cudaGraphDestroy(E->ggraph)); }
    CK(cudaGraphCreate(&E->ggraph, 0));
    CK(cudaGraphConditionalHandleCreate(&E->ghandle, E->ggraph, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams wp = {};
    wp.type = cudaGraphNodeTypeConditional;
    wp.conditional.handle = E->ghandle; wp.conditional.type = cudaGraphCondTypeWhile; wp.conditional.size = 1;
    cudaGraphNode_t wnode;
    CK(cudaGraphAddNode(&wnode, E->ggraph, nullptr, nullptr, 0, &wp));
    E->gb = wp.conditional.phGraph_out[0]; E->ghave = 0;
    launch_pass(E, false, B, n_bt2, n_bt1, n_fold, B, -1, nullptr, nullptr, E->gen_buf, GEN_STRIDE, -1, nullptr, true);
    cudaKernelNodeParams kp = {};
    void* argv[2] = {&D.cell, &E->ghandle};
    kp.func = (void*)k_gen_commit; kp.gridDim = dim3(1); kp.blockDim = dim3(32); kp.kernelParams = argv;
    cudaGraphNode_t n; CK(cudaGraphAddKernelNode(&n, E->gb, &E->glast, 1, &kp));
    E->gb = nullptr;
    CK(cudaGraphInstantiate(&E->gexec, E->ggraph, 0));
    memcpy(E->gkey, key, sizeof(key));
  }
  k_setcell<<<1, 32>>>(D.cell, pos, step0, step0 + nrun);
  if (step0 == 0) k_settok<<<1, 64>>>(D.last_tok, (const long long*)tok64, B);
  CK(cudaGraphLaunch(E->gexec, 0));
  if (step0 + nrun == n_total)
    for (int b = 0; b < B; b++)
      CK(cudaMemcpyAsync((long long*)out64 + (size_t)b * n_total, E->gen_buf + (size_t)b * GEN_STRIDE, sizeof(long long) * (size_t)n_total, cudaMemcpyDeviceToDevice, 0));
  E->gen_launches++;
  E->passes += nrun;
}

// ---------------------------------------------------------------------------------------------------------
// online path with the scheduler on the device
// ---------------------------------------------------------------------------------------------------------
int eng_ds_param_bytes() { return (int)sizeof(DsParams); }
int eng_ds_result_bytes() { return (int)sizeof(DsResult); }
int eng_ds_ring() { return DS_RING; }
// the step and the pass it plans (and, tests, the poison of freed pages), ending with the commit of the results
static void launch_step(Eng* E, int cmd, bool with_pass, bool in_graph) {
  const bool sw = in_graph && with_pass && E->gb && g_light && g_prepq && ATT_DEC;
  if (sw) CK(cudaGraphConditionalHandleCreate(&E->hsw, E->gb, 0, cudaGraphCondAssignDefault));
  SchedArgs sa{E->S, E->rh, E->plan_dev, E->res, E->arena, E->ctl, cmd, sw ? E->hsw : cudaGraphConditionalHandle{}, sw ? 1 : 0, 0, {}};
  // buckets: runs of numbers of columns for which the four GEMMs of a layer keep their shapes (gemm_shape)
  int run_lo[NBUCKET];
  if (sw) {
    const int Ms[4] = {QKV_ROWS, H, 2 * INNER, H};
    int prev[4] = {-1, -1, -1, -1};
    for (int T = 1; T <= TCAP; T++) {
      int cur[4];
      for (int k = 0; k < 4; k++) cur[k] = gemm_shape(Ms[k], T, E->nsm);
      if (memcmp(cur, prev, sizeof(cur))) {
        if (sa.nrun == NBUCKET) { fprintf(stderr, "launch_step: more than %d GEMM shape buckets\n", NBUCKET); abort(); }
        run_lo[sa.nrun++] = T; memcpy(prev, cur, sizeof(cur));
      }
      sa.run_hi[sa.nrun - 1] = T;
    }
  }
  emit<SchedArgs>(E, k_sched, 1, 256, 0, sa);
  if (E->poison) {
    if (E->gb) {
      cudaKernelNodeParams kp = {};
      void* argv[2] = {&E->S, &E->D.kv};
      kp.func = (void*)k_poison_freed; kp.gridDim = dim3(64); kp.blockDim = dim3(256); kp.kernelParams = argv;
      cudaGraphNode_t n; CK(cudaGraphAddKernelNode(&n, E->gb, &E->glast, 1, &kp)); E->glast = n;
    } else k_poison_freed<<<64, 256, 0, E->cur>>>(E->S, E->D.kv);
  }
  if (with_pass && sw) {
    // SWITCH (k_sched's choice) { 0: the general pass; 1, 2, 3: the light passes }: a decode pass costs ~0.8 ms less
    // without the kernels that have nothing to do in it (tests/online_dec.py)
    cudaGraphNodeParams sp = {};
    sp.type = cudaGraphNodeTypeConditional;
    sp.conditional.handle = E->hsw; sp.conditional.type = cudaGraphCondTypeSwitch; sp.conditional.size = 4 + sa.nrun;
    cudaGraphNode_t sn;
    CK(cudaGraphAddNode(&sn, E->gb, &E->glast, nullptr, 1, &sp));
    cudaGraph_t outer = E->gb;
    E->gb = sp.conditional.phGraph_out[0]; E->ghave = 0;
    launch_pass(E, true, 0, 0, 0, 0, 0, 0, nullptr, nullptr, nullptr, 0, 0, E->tok_cur);
    for (int b = 1; b < 4; b++) {
      E->gb = sp.conditional.phGraph_out[b]; E->ghave = 0;
      launch_pass(E, true, 0, 0, 0, 0, 0, 0, nullptr, nullptr, nullptr, 0, 0, E->tok_cur, false, b);
    }
    for (int i = 0; i < sa.nrun; i++) {   // bucket branches: the general pass with per-shape GEMM kernels
      E->gb = sp.conditional.phGraph_out[4 + i]; E->ghave = 0;
      E->bk_lo = run_lo[i]; E->bk_hi = sa.run_hi[i];
      launch_pass(E, true, 0, 0, 0, 0, 0, 0, nullptr, nullptr, nullptr, 0, 0, E->tok_cur);
      E->bk_lo = E->bk_hi = 0;
    }
    E->gb = outer; E->glast = sn; E->ghave = 1; E->gcond = 1;
  } else if (with_pass) launch_pass(E, true, 0, 0, 0, 0, 0, 0, nullptr, nullptr, nullptr, 0, 0, E->tok_cur);
  CommitArgs ca{E->S, E->res, E->tok_cur, E->status, E->ctl, E->hwhile, in_graph ? 1 : 0};
  emit<CommitArgs>(E, k_commit, 1, 256, 0, ca);
}
void eng_ds_init(void* h, const DsParams* P, int poison, int use_graph) {
  Eng* E = (Eng*)h;
  if (!E->S) {
    CK(cudaMalloc(&E->S, sizeof(DsState)));
    CK(cudaMalloc(&E->rh, sizeof(DsReqHost) * DS_RING));
    CK(cudaMalloc(&E->res, sizeof(DsResult) * NRES));
    CK(cudaMalloc(&E->arena, sizeof(int) * (size_t)ARENA_TOK));
    CK(cudaMalloc(&E->ctl, sizeof(DsCtl)));
    CK(cudaMalloc(&E->status, sizeof(DsStatus)));
    CK(cudaMalloc(&E->tok_cur, sizeof(int) * SELCAP));
    CK(cudaHostAlloc(&E->rh_host, sizeof(DsReqHost) * DS_RING, cudaHostAllocDefault));
    CK(cudaHostAlloc(&E->res_host, sizeof(DsResult) * NRES, cudaHostAllocDefault));
    CK(cudaHostAlloc(&E->arena_host, sizeof(int) * (size_t)ARENA_TOK, cudaHostAllocDefault));
    CK(cudaHostAlloc(&E->ctl_host, sizeof(DsCtl), cudaHostAllocDefault));
    CK(cudaHostAlloc(&E->status_host, sizeof(DsStatus), cudaHostAllocDefault));
    CK(cudaStreamCreateWithFlags(&E->sA, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&E->sB, cudaStreamNonBlocking));
  }
  CK(cudaDeviceSynchronize());
  DsState* S0 = (DsState*)calloc(1, sizeof(DsState));
  ds_init(*S0, *P);
  CK(cudaMemcpy(E->S, S0, sizeof(DsState), cudaMemcpyHostToDevice));
  free(S0);
  CK(cudaMemset(E->res, 0, sizeof(DsResult) * NRES));
  CK(cudaMemset(E->ctl, 0, sizeof(DsCtl)));
  CK(cudaMemset(E->status, 0, sizeof(DsStatus)));
  memset(E->ctl_host, 0, sizeof(DsCtl));
  E->tail = 0; E->arena_pos = 0; E->harvested = 0; E->poison = poison; E->use_graph = use_graph;
  if (use_graph) {
    // The serving loop as one graph: WHILE (a request is waiting, running or arriving) { step; pass; commit }.
    // Once launched it runs on the device without the host. Conditional kernels are not IF nodes (5.7 us each,
    // tests/graph_while.cu) but plain kernels that return at once when they have no work (0.9 us).
    if (E->exec) { CK(cudaGraphExecDestroy(E->exec)); CK(cudaGraphDestroy(E->graph)); E->exec = nullptr; }
    CK(cudaGraphCreate(&E->graph, 0));
    CK(cudaGraphConditionalHandleCreate(&E->hwhile, E->graph, 1, cudaGraphCondAssignDefault));
    cudaGraphNodeParams wp = {};
    wp.type = cudaGraphNodeTypeConditional;
    wp.conditional.handle = E->hwhile; wp.conditional.type = cudaGraphCondTypeWhile; wp.conditional.size = 1;
    cudaGraphNode_t wnode;
    CK(cudaGraphAddNode(&wnode, E->graph, nullptr, nullptr, 0, &wp));
    E->gb = wp.conditional.phGraph_out[0]; E->ghave = 0;
    launch_step(E, 0, true, true);
    E->gb = nullptr;
    CK(cudaGraphInstantiate(&E->exec, E->graph, 0));
  }
}
// A request arrives: the prompt goes into the arena, its record into the ring, then the ring count is published
// (three asynchronous copies from pinned memory on the copy stream; the serving loop is not touched).
// `oldest`: ring count of the oldest request the host still waits for (its arena space and everything after it is
// in use). Returns the request's ring count, or -1 if ring or arena are full (the host retries later).
long long eng_ds_submit(void* h, const int* prompt, int n_in, int n_out, double t_arrive, long long oldest, long long oldest_off) {
  Eng* E = (Eng*)h;
  if (E->tail - oldest >= DS_RING - 1) return -1;
  // the live prompts occupy the cyclic range [oldest_off, arena_pos) of the arena
  long long off = E->arena_pos;
  if (oldest == E->tail) off = 0;                               // nothing alive
  else if (off >= oldest_off) {                                 // not wrapped: free are [off, end) and [0, oldest_off)
    if (off + n_in > ARENA_TOK) { if (n_in >= oldest_off) return -1; off = 0; }
  } else if (off + n_in >= oldest_off) return -1;               // wrapped: free is [off, oldest_off)
  memcpy(E->arena_host + off, prompt, sizeof(int) * (size_t)n_in);
  CK(cudaMemcpyAsync(E->arena + off, E->arena_host + off, sizeof(int) * (size_t)n_in, cudaMemcpyHostToDevice, E->sB));
  const int ri = (int)(E->tail % DS_RING);
  E->rh_host[ri].n_in = n_in; E->rh_host[ri].n_out = n_out; E->rh_host[ri].prompt_off = (int)off; E->rh_host[ri].t_arrive = (float)t_arrive;
  CK(cudaMemcpyAsync(E->rh + ri, E->rh_host + ri, sizeof(DsReqHost), cudaMemcpyHostToDevice, E->sB));
  E->arena_pos = off + n_in;
  E->tail++;
  CK(cudaStreamSynchronize(E->sB));   // the copies are there (waits for the copy stream only) before the count says so
  E->ctl_host->tail = (int)E->tail;
  CK(cudaMemcpyAsync(E->ctl, E->ctl_host, sizeof(DsCtl), cudaMemcpyHostToDevice, E->sB));
  return E->tail - 1;
}
long long eng_ds_arena_off(void* h, long long count) { return ((Eng*)h)->rh_host[count % DS_RING].prompt_off; }
// The host's clock and the cost-model correction, for the next step the device takes.
void eng_ds_tick(void* h, double now, double speed) {
  Eng* E = (Eng*)h;
  CK(cudaStreamSynchronize(E->sB));
  E->ctl_host->now = (float)now; E->ctl_host->speed = (float)speed; E->ctl_host->harvested = (int)E->harvested;
  CK(cudaMemcpyAsync(E->ctl, E->ctl_host, sizeof(DsCtl), cudaMemcpyHostToDevice, E->sB));
}
// Result blocks of the steps finished since the last call (at most `maxn`), copied on the copy stream.
int eng_ds_harvest(void* h, DsResult* out, int maxn) {
  Eng* E = (Eng*)h;
  CK(cudaMemcpyAsync(E->status_host, E->status, sizeof(DsStatus), cudaMemcpyDeviceToHost, E->sB));
  CK(cudaStreamSynchronize(E->sB));
  const long long done = E->status_host->done;
  if (done - E->harvested > NRES) { fprintf(stderr, "eng_ds_harvest: %lld result blocks were overwritten before the host read them\n", done - E->harvested - NRES); abort(); }
  int n = 0;
  while (E->harvested < done && n < maxn) {
    const int slot = (int)(E->harvested % NRES);
    CK(cudaMemcpyAsync(E->res_host + slot, E->res + slot, sizeof(DsResult), cudaMemcpyDeviceToHost, E->sB));
    CK(cudaStreamSynchronize(E->sB));
    out[n++] = E->res_host[slot];
    E->harvested++;
  }
  return n;
}
// 1 if the serving loop is on the device now.
int eng_ds_running(void* h) {
  Eng* E = (Eng*)h;
  const cudaError_t e = cudaStreamQuery(E->sA);
  if (e == cudaErrorNotReady) return 1;
  CK(e);
  return 0;
}
// Start the serving loop (graph), or, without the graph, launch one step and its pass.
void eng_ds_run(void* h) {
  Eng* E = (Eng*)h;
  if (E->use_graph) { CK(cudaGraphLaunch(E->exec, E->sA)); E->graph_launches++; }
  else { E->gb = nullptr; E->cur = E->sA; launch_step(E, 0, true, false); E->cur = 0; E->host_launched++; }
}
// Off the serving path (waits for the loop to end): one scheduler step without a pass. cmd 2: bookkeeping only
// (requests whose last token is out leave their slots); cmd 1: also forget every retained prefix.
void eng_ds_step_only(void* h, int cmd) {   // the caller has made sure the loop is not running
  Eng* E = (Eng*)h;
  CK(cudaStreamSynchronize(E->sA));
  E->gb = nullptr; E->cur = E->sA;
  launch_step(E, cmd, false, false);
  E->cur = 0;
  CK(cudaStreamSynchronize(E->sA));
  E->sched_only++;
}
void eng_ds_counters(void* h, long long* out) {   // statistics: read while the loop may be running
  Eng* E = (Eng*)h;
  CK(cudaMemcpyAsync(out, (char*)E->S + offsetof(DsState, cnt), sizeof(long long) * DS_NCNT, cudaMemcpyDeviceToHost, E->sB));
  CK(cudaStreamSynchronize(E->sB));
  out[DS_NCNT - 1] = E->graph_launches; out[DS_NCNT - 2] = E->host_launched;
}

// tokens of the last pass (blocks until the pass is done: one host sync)
void eng_read_tokens(void* h, int* out, int n_sel) {
  Eng* E = (Eng*)h;
  CK(cudaMemcpy(out, E->D.sel_tok, sizeof(int) * (size_t)n_sel, cudaMemcpyDeviceToHost));
  E->tok_reads++;
}

void eng_set_last_tok(void* h, int slot, int tok) {
  Eng* E = (Eng*)h;
  CK(cudaMemcpy(E->D.last_tok + slot, &tok, sizeof(int), cudaMemcpyHostToDevice));
}

// tests: poison a page's K and V with NaN
void eng_poison_page(void* h, int page) {
  Eng* E = (Eng*)h;
  k_fill16<<<256, 256>>>((unsigned short*)((char*)E->D.kv + PAGE_BYTES * (size_t)page), PAGE_BYTES / 2, (unsigned short)0x7fc0);
}

// tests / bench: a bare matvec through the GEMM kernel. W: [M][K], X: [columns][K] with ng column groups, both in
// the operand layout (xidx); out [T][M] f32
void eng_gemm_raw(void* h, const void* W, const void* X, void* out, int M, int K, int T, int ng) {
  Eng* E = (Eng*)h;
  GemmArgs g{};
  g.W = (const unsigned short*)W; g.X = (const unsigned short*)X; g.NG = ng; g.M = M; g.K = K; g.T = T; g.out = (float*)out;
  launch_gemm<EP_RAW>(E, g);
}

// bench (tests/gemm_shapes.py): the same matvec with a given block shape; returns the shape used
int eng_gemm_shape(void* h, const void* W, const void* X, void* out, int M, int K, int T, int ng, int shape) {
  Eng* E = (Eng*)h;
  GemmArgs g{};
  g.W = (const unsigned short*)W; g.X = (const unsigned short*)X; g.NG = ng; g.M = M; g.K = K; g.T = T; g.out = (float*)out;
  int nb, nt; float us;
  if (shape < 0) shape = gemm_shape(M, T, E->nsm);
  shape_dims(shape, nb, nt, us);
  if (M % (64 * nt)) return -1;
  g.shape = shape;
  const int n_units = (M / (64 * nt)) * ((T + 2 * nb - 1) / (2 * nb)), grid = n_units < E->nsm ? n_units : E->nsm;
  if (g_split >= 3) emit_shape<EP_RAW>(E, shape, grid, g);   // (tests: the per-shape kernel)
  else emit<GemmArgs>(E, k_gemm<EP_RAW>, grid, WG_THREADS, WG_SMEM_MAX, g);
  return shape;
}

long long eng_stat(void* h, int which) {
  Eng* E = (Eng*)h;
  return which == 0 ? E->launches : which == 1 ? E->passes : which == 2 ? E->nsm : which == 3 ? E->plan_uploads : which == 4 ? E->tok_reads : which == 5 ? E->gen_launches : 0;
}

void eng_sync() { CK(cudaDeviceSynchronize()); }

void eng_destroy(void* h) {
  Eng* E = (Eng*)h;
  cudaDeviceSynchronize();
  Dev& D = E->D;
  cudaFree(D.kv); cudaFree(E->plan_dev); cudaFree(D.x); cudaFree(D.xfrag); cudaFree(D.ssq); cudaFree(D.rn); cudaFree(D.qkv);
  cudaFree(D.qfrag); cudaFree(D.afrag); cudaFree(D.mfrag); cudaFree(D.lmfrag); cudaFree(D.lm_rn); cudaFree(D.bm_val);
  cudaFree(D.bm_idx); cudaFree(D.last_tok); cudaFree(D.sel_tok); cudaFree(D.att_part); cudaFree(D.ctr);
  delete E;
}

}  // extern "C"
