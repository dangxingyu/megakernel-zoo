// North-Mini-Code-1.0 decode megakernel for one H100.
//
// One persistent kernel per step() (or per generate() call): 132 blocks x 16 warps, one block per SM.
// Every block walks the same static program; nothing is claimed dynamically. The unit of work is a quantum:
// 8 k-steps (4 KB) of one 16-row tile.
// Per layer:
//   prologue  every block reads the residual and computes the RMSNorm into its own shared memory
//   A         q|k|v and router rows: every warp of blocks 1..131 reads 20 k-steps (router rows first); block 0
//             takes the top 8 per sequence as soon as the router rows are in and publishes the union of experts
//   B         blocks 1..4B: attention of one (sequence, K/V head) each; the others: gate|up of the chosen experts,
//             each block's tiles dealt as quanta to its 16 warps and summed in shared memory
//   C         block r < 128 owns residual rows 16r..16r+15: o_proj and the down rows of every chosen expert as
//             quanta, summed in shared memory and added to the residual
// Blocks meet only through monotonic counters (router rows done, A done, route published, B done, C done).
// Every matrix product is a tensor-core m16n8k16 whose A operand is read straight from memory in fragment
// order and whose B operand holds the batch in its 8 columns: batch 1 and batch 8 run the same instructions, and
// every sum has a fixed order that does not depend on the batch.
//
// Model facts used (CohereLabs/North-Mini-Code-1.0): parallel block (attention and MLP read the same normed
// input), 8 of 128 experts per token, 4 K/V heads x 8 query heads (the 8 query heads of a K/V head are the 8
// columns of an mma), tied head.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstddef>

#define HID 2048
#define NLAYER 49
#define TOPK 8
#define MAXB 8
#define MAXU 64
#define VOCAB 262144
#define QKV_ROWS 5120
#define PART_ROWS 5248  // q 4096, k 512, v 512, router 128
#define FULL_TILES 262   // stage A: tiles 0..261 are done whole by one block, tiles 262..327 (the last 8: router) in two halves
#define SHARED_ROW0 (FULL_TILES * 16)
#define TB 12           // gate|up tiles per shared-memory batch
#define QB 192          // quanta per shared-memory batch
#define HMAX 3072
#define SCALE 0.08838834764831845f
#define CTX 2048        // largest context this build's shared-memory layout holds
#define NCH 8           // attention: position tiles are dealt to 8 chunks (tile P belongs to chunk P % 8)
#define NATT 32         // blocks reserved for attention in stage B
#define PREC 1040       // floats of one chunk's partial result: out[8][128], sum[8], max[8]

// Weight storage. A tile is 16 rows x K k-steps (16 columns each), stored per k-step in mma fragment order.
//   track A (default): the checkpoint's BF16 bits, 512 B per k-step.
//   track B (-DW12):   12 bits per weight, 384 B per k-step, plus a header of EXB bytes per tile:
//     W[i][j] = 2^(r_i + c_j) * V[i][j], V with a 4-bit exponent code 1..15 (0 = zero), sign and 7 mantissa bits.
//     r_i (one byte per row, header) is undone after the product; c_j is moved into the x operand (only the down
//     matrices use it: one shift per expert neuron, kept in the header of the neuron's gate|up tile). The x
//     operands are stored times 2^XS so that the 4-bit code can sit in the bf16 exponent field as it is: the
//     tensor core sees V * (x 2^XS) and every product is the true product times a power of two.
//     Weights outside the 15-value window (1.3e-4 of them) are 0 in the tile and listed in the header with their
//     true value; their products are added after the tile's sum.
typedef const unsigned char* WP;
#ifdef W12
#define KSB 384
#define XS 96
#define EXB(K_) ((K_) == 48 ? 128 : 256)
#else
#define KSB 512
#define XS 0
#define EXB(K_) 0
#endif
#define QBY (8 * KSB)                                   // bytes of a quantum (8 k-steps)
#define TBY(K_) ((size_t)(K_) * KSB + EXB(K_))          // bytes of a tile
#define XSF __uint_as_float((uint32_t)(127 + XS) << 23) // 2^XS

struct LayerW {
  WP qkv;               // 320 row tiles, K = 128 k-steps
  WP router;            // 8 row tiles, K = 128 (unused in layer 0)
  WP o;                 // 128 row tiles, K = 256
  WP expert;            // expert e at expert + e * exp_stride: gate|up interleaved (gu_tiles x 128), then down (128 x down_K)
  const uint16_t* normw;
  uint64_t exp_stride;  // in bytes
  int gu_tiles, down_K, rope, dense;
};

struct Ctl {  // monotonic counters and sequence words; every wait is "word >= target"
  uint32_t tok_seq; uint32_t pad0[15];
  uint32_t route_seq; uint32_t pad1[15];
  uint32_t cntA; uint32_t pad2[15];
  uint32_t cntB; uint32_t pad3[15];
  uint32_t cntC; uint32_t pad4[15];
  uint32_t cntH; uint32_t pad5[15];
  uint32_t cntR; uint32_t pad6[15];
  uint32_t prog; uint32_t pad7[15];  // serving: tokens produced so far by the running launch
  uint32_t tst[32][8];  // per attention task and block: stamp (layer count + 1) once its partial results are stored
  unsigned long long um[4]; unsigned long long pad8[4];
  uint32_t adone[32];   // per attention task: stamp (layer count + 1) once its output rows are in M->attn  // (layer stamp << 32) | 32 experts of the batch's union: gate|up starts from these
};

struct Dyn {
  int U;
  int slot_expert[MAXU];
  int bslot[MAXB][TOPK];
  float bw[MAXB][TOPK];
  int bmask[MAXU];  // which sequences use the expert of a slot
};

struct Model {
  LayerW L[NLAYER];
  WP head;
  const uint16_t* embed;
  const uint16_t* normf;
  const float* rope;  // [ctx][64][2]
  uint4* kc; uint4* vc;
  uint16_t* x; uint16_t* qkvo; float* qkvp; uint2* rl; uint16_t* attn; float* attp; uint16_t* h;
  float* logits; float* candv; int* candi; int64_t* curtok;
  const uint32_t* zeros;
  Ctl* ctl; Dyn* dyn;
  const uint16_t* l0qkv;  // optional [VOCAB][5120]: q|k|v of layer 0 per token (before RoPE), see notes/checkpoint.md section 6
  const uint16_t* l0mlp;  // optional [VOCAB][HID]: MLP output of layer 0 per token
  uint16_t* l0m;          // [MAXB][HID] the MLP rows of this step's tokens
  WP sk;                  // optional: int8 sketch of the head for the greedy path (certified, see the head stage)
  uint16_t* dbg_x;     // optional [NLAYER+1][MAXB][HID]
  const int* force;    // optional [NLAYER][MAXB][TOPK] forced expert ids (debug)
  long long* prof;     // optional [32] cycle counters of the leader's SM (debug)
  unsigned long long* trace;   // optional [132][49][12] %globaltimer at stage boundaries of one step (debug, -DTRACE)
  const int16_t* role;        // optional [256]: the role (block number of the schedule) of the block running on SM %smid
  int max_ctx, maxK;
};

struct Launch {
  const Model* M;
  const int64_t* tok_in;
  float* logits_out;
  int64_t* tok_out;
  int B, pos, nsteps, gen, tok_stride;
  int P, kmajor;        // gen: the first P inputs of a sequence come from tok_in[b * tok_stride + s] (a prompt; P >= 1);
                        // kmajor: tok_out is [k][B] instead of [B][k]
  uint32_t g0, s0, h0;  // layers, steps and generate-steps completed before this launch
};

#define MMA(c0, c1, c2, c3, a, b0, b1)                                                                          \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)                                                         \
               : "r"((a).x), "r"((a).y), "r"((a).z), "r"((a).w), "r"(b0), "r"(b1))

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(((uint32_t)v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
// two floats rounded to bf16 (nearest even) in one instruction: the same bits as f2bf for every finite value
__device__ __forceinline__ uint32_t f2bf2(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ float bflo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bfhi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t vload(const uint32_t* p) { return *(const volatile uint32_t*)p; }
// Store discipline (all measured on this card, see notes/journal.md):
//  * an SM's L1 keeps lines of global memory; an atomic of another SM does not invalidate them, a __threadfence
//    of any thread of the SM does (one per layer by thread 0 is enough to make an all-atomic build deterministic);
//  * a plain store needs a later __threadfence of its thread to be visible in time;
//  * a __threadfence costs ~1 us on a quiet queue and several under the weight stream.
// So: results that are handed over at the end of a stage (quiet queue) use plain stores + fence, which also
// refreshes the block's L1 once per stage; attention, which hands over under the stream, uses xstore.
// xstore: a store that has reached L2 when the thread moves on to its next barrier (the exchange returns a value): used
// for data another SM reads after a signal. A plain store would need a __threadfence, which under the weight
// stream costs several microseconds (bench/hop_bench.cu: store + fence + load 7-9 us, an atomic 0.3-0.7 us).
// The caller must consume the returned old values (XDONE) before its barrier: only then has it waited for them.
#ifdef XS_PLAIN
__device__ __forceinline__ uint32_t xstore(uint32_t* p, uint32_t v) { *p = v; __threadfence(); return 0; }
__device__ __forceinline__ uint32_t xstoref(float* p, float v) { *p = v; __threadfence(); return 0; }
#else
__device__ __forceinline__ uint32_t xstore(uint32_t* p, uint32_t v) { return atomicExch(p, v); }
__device__ __forceinline__ uint32_t xstoref(float* p, float v) { return __float_as_uint(atomicExch(p, v)); }
#endif
// (a volatile store: it must not be optimized away, it is what makes the thread wait for its exchanges)
#define XDONE(acc_) { *(volatile uint32_t*)&sm.W[128 + (threadIdx.x & 31)] = (acc_); }  // W[128..159]: nothing else lives there
// bf16 results of a 16-row tile live two rows per 32-bit word: the lane of the even row stores its own value and
// the value of the lane 4 above (the odd row). Returns the old word (0 for odd lanes).
__device__ __forceinline__ uint32_t xstore_rowpair(uint16_t* p_even_row, uint16_t mine, bool doit) {
  const uint32_t other = __shfl_xor_sync(0xffffffffu, (uint32_t)mine, 4);
  if (doit && !((threadIdx.x >> 2) & 1)) return xstore((uint32_t*)p_even_row, (uint32_t)mine | (other << 16));
  return 0;
}

// a quantum: 8 k-steps (4 KB) of a 16-row tile, read with QU loads in flight per thread. Under a saturated HBM
// queue the latency of EVERY memory operation of every SM (a poll, an atomic, a store) is about
// (bytes in flight) / (3.1 TB/s) (bench/l2_bench2.cu: 2-3 us with 16 loads in flight per thread, a __threadfence
// 10 us), so the queue is kept shallow: 16 warps per SM x QU loads x 512 B still saturates the card.
#ifndef QU
#define QU 2
#endif
// gate|up results and the residual rows go out as exchanges instead of stores + fence (-9 us at B=1, -20 at long
// contexts, e0357/e0358); -DFENCED_HC restores the fences
#ifndef EARLY_B
#define EARLY_B 0   // (batch-1 kernel) the stage-B barrier's poll issued with the last o_proj quantum
#endif
#ifndef EARLY_U
#define EARLY_U 1
#endif
#ifndef FENCED_HC
#define ATOMIC_H
#define ATOMIC_C
#endif
#ifndef ATT_STAMP
#define ATT_STAMP 0
#endif
#ifndef WNA2
#define WNA2 0   // the same for the sketch head's rows and the attention K/V tiles
#endif
__device__ __forceinline__ uint4 ldna4(const void* p) {
  uint4 v;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
#define LDS4(p_) (WNA2 ? ldna4(p_) : *(const uint4*)(p_))
// WNA: weight loads do not allocate in L1 (the x operands of stage C need it: -90 us at B=1, -273 at B=8, e0340);
// ZSM: the zero operand of unused batch columns from shared memory instead of global memory (same purpose)
#ifndef WNA
#define WNA 1
#endif
#ifndef ZSM
#define ZSM 1
#endif
#ifndef FENCE
#define FENCE() __threadfence()
#endif

// L2 as a staging buffer (experiment, -DPF=mask): while a block waits at a barrier it asks L2 for the weights it
// will read after the barrier. prefetch.global.L2 returns no value, so the warp does not wait for it; HBM keeps
// delivering during the skeleton work and the next stage's loads are L2 hits.
//   bit 0: end of stage C  -> the block's stage A share of the next layer (or its first head tile)
//   bit 1: end of stage B  -> the block's down tiles of the chosen experts
//   bit 2: end of stage A  -> the block's o_proj tile (attention blocks: none, they are busy)
//   bit 3: end of stage C  -> attention blocks: the K/V cache tiles of their task in the next layer
#ifndef PF
#define PF 0
#endif
#ifndef PFS
#define PFS 64   // bytes one prefetch brings in (bench/pf_bench.cu)
#endif
#define PFL2(p_) asm volatile("prefetch.global.L2 [%0];" :: "l"(p_))
__device__ __forceinline__ void pf_range(WP p, int bytes) {
  for (int off = (threadIdx.x & 31) * PFS; off < bytes; off += 32 * PFS) PFL2(p + off);
}
#define MMA4(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1)                                                           \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)                                                         \
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
// Certified sketch of the head (greedy path only; the technique is the certified LM head of arXiv 2611.02501, the
// int8 form is ours). Row i of the head is also stored as int8: w_i ~ s_i q_i (s_i = max|w_i| / 127), in the
// fragment order of the integer tensor-core instruction m16n8k32: 512 weights per 512-byte load, no widening.
// A sketch tile is 16 rows: 64 k-steps of 32 columns, then 256 bytes: per row pair (g, g + 8) the floats
// {s, s', en, en', qn, qn', 0, 0} with en >= ||w - s q|| + 2^-10 ||w|| and qn >= ||s q||.
// The input goes in as two int8 digits, x = t (128 hi + lo) + r, so that the integer sums are exact and
//   | logit_i - s_i t (128 <q_i, hi> + <q_i, lo>) |  <=  en_i ||x|| + qn_i ||r||      (Cauchy-Schwarz).
// (2^-10 ||w|| ||x|| covers the float32 rounding of the exact path: 2176 additions at 4 ulp each are 5.2e-4.)
#define SKT (64 * 512 + 256)
#ifndef SKQ
#define SKQ 4   // sketch loads in flight per thread
#endif
#define MMA8(c0, c1, c2, c3, a, b0, b1)                                                                         \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
               : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)                                                         \
               : "r"((a).x), "r"((a).y), "r"((a).z), "r"((a).w), "r"(b0), "r"(b1))
// NK k-steps starting at byte pointer p (a k-step boundary of a tile), times the matching k-steps of x
// (DP: k-step pairs in flight per thread on track B. 2 while the card streams; more for a lone tile on a quiet card,
//  where only the number of round trips counts. The order of the tensor-core additions is the same for every DP.)
// register q of a thread from its two low bytes (already at bytes 0 and 2) and its high word (see engine/api.py hi_pack)
#define W12_ROT(q_) (((q_) & 3) == 0 ? 0 : ((q_) & 3) == 1 ? 4 : ((q_) & 3) == 2 ? 8 : 28)
#define W12_REG(low_, hw_, q_) ((low_) | (__funnelshift_l((hw_), (hw_), W12_ROT(q_)) & 0x87008700u))
// Stage A staging (-DSTAGE_A=1, track B): at the end of stage C every lane copies, with cp.async into shared memory,
// the bytes it will read first in the next layer's stage A (the half tile and its quantum of the first whole tile:
// 6 k-step pairs, the same layout as in global memory), so that HBM delivers them while the block waits at the
// barrier and normalises. Each lane reads only what it copied: cp.async.wait_all of the lane is enough.
#ifndef STAGE_A
#define STAGE_A 0
#endif
#if STAGE_A >= 2 && NPOLL > 1
#error the pollers' words share the mbarrier area
#endif
#define STG_OFF 24576     // bytes into the slot area: below it stage A keeps its 48 quantum sums
#define STG_WARP 4608     // per warp: 2 pairs of the half tile, then 4 pairs of the first whole tile
__device__ __forceinline__ void cpa16(void* sdst, const void* gsrc) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(sdst)), "l"(gsrc));
}
__device__ __forceinline__ void cpa8(void* sdst, const void* gsrc) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"((uint32_t)__cvta_generic_to_shared(sdst)), "l"(gsrc));
}
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_all;" ::: "memory"); }
// (STAGE_A >= 2) the warp's staging as two bulk copies issued by lane 0, completion counted on the warp's mbarrier
__device__ __forceinline__ void stage_bulk(unsigned char* s, WP g0, int n0, WP g1, int n1, uint64_t* mb) {
  if ((threadIdx.x & 31) == 0) {
    const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // (the slot area was last written by ordinary stores)
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(m), "r"(n0 + n1) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"((uint32_t)__cvta_generic_to_shared(s)), "l"(g0), "r"(n0), "r"(m) : "memory");
    if (n1) asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"((uint32_t)__cvta_generic_to_shared(s + n0)), "l"(g1), "r"(n1), "r"(m) : "memory");
  }
}
__device__ __forceinline__ void stage_bulk_wait(uint64_t* mb, uint32_t& ph) {
  const uint32_t m = (uint32_t)__cvta_generic_to_shared(mb);
  uint32_t done = 0;
  while (!done) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }" : "=r"(done) : "r"(m), "r"(ph) : "memory");
  ph ^= 1;
}
// npairs k-step pairs starting at g (global) into s (shared), this lane's part only
__device__ __forceinline__ void stage_pairs(unsigned char* s, WP g, int npairs) {
  const int lane = threadIdx.x & 31;
  for (int kp = 0; kp < npairs; kp++) {
    cpa16(s + kp * 768 + lane * 16, g + (size_t)kp * 768 + lane * 16);
    cpa8(s + kp * 768 + 512 + lane * 8, g + (size_t)kp * 768 + 512 + lane * 8);
  }
}
template <int NK, int DP = 2>
__device__ __forceinline__ void kmma_acc(WP __restrict__ p, const uint32_t* __restrict__ xb, float& c0, float& c1, float& c2, float& c3) {
#ifdef ABL_NOSTREAM
  c0 += 1e-3f; return;
#endif
  const int lane = threadIdx.x & 31;
#ifndef W12
  const uint4* __restrict__ q = (const uint4*)p + lane;
#pragma unroll
  for (int k = 0; k < NK; k += QU) {
    uint4 w[QU];
#pragma unroll
    for (int u = 0; u < QU; u++) w[u] = WNA ? ldna4(q + (k + u) * 32) : q[(k + u) * 32];
#pragma unroll
    for (int u = 0; u < QU; u++) {
      uint32_t b0 = xb[(k + u) * 8], b1 = xb[(k + u) * 8 + 4];
      MMA(c0, c1, c2, c3, w[u], b0, b1);
    }
  }
#else
  // two k-steps of a thread are 16 low bytes [c0 | mantissa] and two 32-bit words with the sign and code bits 1..3
  // of the 16 weights, laid out so that a register of two bf16 is one byte permutation, one rotation and one
  // three-input logic operation (W12_REG; engine/api.py hi_pack). The earlier layout (a nibble per weight) needed
  // about 12 instructions per register and made the stream instruction-bound (e0225: -63 us at B=1, -340 at B=8).
#pragma unroll
  for (int kp = 0; kp < NK / 2; kp += DP) {
    uint4 lo[DP]; uint2 hi[DP];
#pragma unroll
    for (int u = 0; u < DP; u++) {
#if WNA
      // (weights are read once: they should not take L1 lines from the x operands of stage C)
      asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lo[u].x), "=r"(lo[u].y), "=r"(lo[u].z), "=r"(lo[u].w) : "l"(p + (size_t)(kp + u) * 768 + lane * 16));
      asm volatile("ld.global.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(hi[u].x), "=r"(hi[u].y) : "l"(p + (size_t)(kp + u) * 768 + 512 + lane * 8));
#else
      lo[u] = *(const uint4*)(p + (size_t)(kp + u) * 768 + lane * 16);
      hi[u] = *(const uint2*)(p + (size_t)(kp + u) * 768 + 512 + lane * 8);
#endif
    }
#pragma unroll
    for (int u = 0; u < DP; u++) {
      uint32_t r[8];
      const uint32_t l4[4] = {lo[u].x, lo[u].y, lo[u].z, lo[u].w};
      const uint32_t h2[2] = {hi[u].x, hi[u].y};
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const uint32_t lw = l4[q >> 1];
        const uint32_t lowpair = (q & 1) ? __byte_perm(lw, 0, 0x4342) : __byte_perm(lw, 0, 0x4140);   // [b, 0, b', 0]
        r[q] = W12_REG(lowpair, h2[q >> 2], q);
      }
      uint32_t b0 = xb[(2 * (kp + u)) * 8], b1 = xb[(2 * (kp + u)) * 8 + 4];
      MMA4(c0, c1, c2, c3, r[0], r[1], r[2], r[3], b0, b1);
      b0 = xb[(2 * (kp + u) + 1) * 8]; b1 = xb[(2 * (kp + u) + 1) * 8 + 4];
      MMA4(c0, c1, c2, c3, r[4], r[5], r[6], r[7], b0, b1);
    }
  }
#endif
}
#ifdef W12
// pair kp (two k-steps) of a tile widened to bf16 fragments: the very bits kmma_acc gives the tensor core
__device__ __forceinline__ void w12_pair(WP __restrict__ p, int kp, uint32_t* r) {
  const int lane = threadIdx.x & 31;
  const uint4 lo = *(const uint4*)(p + (size_t)kp * 768 + lane * 16);
  const uint2 hi = *(const uint2*)(p + (size_t)kp * 768 + 512 + lane * 8);
  const uint32_t l4[4] = {lo.x, lo.y, lo.z, lo.w};
  const uint32_t h2[2] = {hi.x, hi.y};
#pragma unroll
  for (int q = 0; q < 8; q++) {
    const uint32_t lw = l4[q >> 1];
    const uint32_t lowpair = (q & 1) ? __byte_perm(lw, 0, 0x4342) : __byte_perm(lw, 0, 0x4140);
    r[q] = W12_REG(lowpair, h2[q >> 2], q);
  }
}
#endif
// one quantum from zero
#ifndef QDP
#define QDP 2
#endif
__device__ __forceinline__ float4 qrun(WP __restrict__ p, const uint32_t* __restrict__ xb) {
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  kmma_acc<8, QDP>(p, xb, c0, c1, c2, c3);
  return make_float4(c0, c1, c2, c3);
}
// half a quantum (4 k-steps) from zero, its loads in one group
__device__ __forceinline__ float4 hrun(WP __restrict__ p, const uint32_t* __restrict__ xb) {
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#ifdef ABL_NOSTREAM
  return make_float4(1e-3f, 0, 0, 0);
#endif
#ifndef W12
  const uint4* __restrict__ q = (const uint4*)p + (threadIdx.x & 31);
  uint4 w[4];
#pragma unroll
  for (int u = 0; u < 4; u++) w[u] = WNA ? ldna4(q + u * 32) : q[u * 32];
#pragma unroll
  for (int u = 0; u < 4; u++) { uint32_t b0 = xb[u * 8], b1 = xb[u * 8 + 4]; MMA(c0, c1, c2, c3, w[u], b0, b1); }
#else
  kmma_acc<4>(p, xb, c0, c1, c2, c3);
#endif
  return make_float4(c0, c1, c2, c3);
}
__device__ __forceinline__ float4 f4add(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

struct Smem {
  uint16_t* xn;   // [MAXB][HID] normed input of the current layer (bf16)
  uint32_t* zero; // 8 KB of zeros: the B operand of batch columns that are not in use
  float4* slot;   // [QB][32] one mma result per quantum and lane
  float4* Cst;    // [4 + MAXU][32] totals per unit of stage C
  float* S;       // [8][ctx] attention scores, then exp        (attention overlays slot and Cst)
  uint16_t* Pm;   // [8][ctx + 128] probabilities, bf16
  uint16_t* Qs;   // [8][128] rotated queries
  float* O;       // [2][8][128]
  float* R;       // [512]
  float* Mx;      // [8]
  uint32_t* W;    // [128] misc words
  WP* eb;         // [MAXU] expert base per slot
  int* bm;        // [MAXU] sequences per slot
  int* bslot;     // [MAXB][TOPK]
  float* bw;      // [MAXB][TOPK]
  WP* tp;         // [TB] tile pointers of a gate|up batch
  float* Part;    // [NCH][PREC] attention partials
  long long* prof; // [64] cycle counters of this block (debug), flushed to global memory when the launch ends
  uint32_t* EX;   // [16 warps][64] exception entries of the tile a warp is finishing (track B)
  uint16_t* XV;   // [16 warps][64][8] x values of those entries (stage C)
  int2* gt;       // [1 + MAXU] this block's gate|up share (tiles, first tile) for a union of U experts (0: the dense layer)
  uint64_t* mb;   // [16] one mbarrier per warp (bulk copies of stage A staging)
};

// ---------------------------------------------------------------------------------------------------------
// attention for one (sequence, K/V head): the whole block works on it
// ---------------------------------------------------------------------------------------------------------
#ifndef NPOLL
#define NPOLL 1         // warps that poll a counter at a barrier, their first polls staggered by POLL_STAGGER cycles
#endif
#ifndef POLL_STAGGER
#define POLL_STAGGER 400
#endif
#if NPOLL == 1
#define WAIT_GE(ptr_, target_)                                                                              \
  {                                                                                                         \
    if (tid == 0) { const uint32_t tg__ = (uint32_t)(target_); while ((int)(vload(ptr_) - tg__) < 0) {} }   \
    __syncthreads();                                                                                        \
  }
#else
// Several pollers: under the weight stream one poll is a round trip of ~1 us, and a single poller sees the final
// value on average half a round trip after it has landed. Lane 0 of warps 0..NPOLL-1 poll, staggered; the first to
// see the target leaves the value in a shared-memory word of that counter (one word per Ctl field: the targets of
// different counters take the same values), which the others check between their polls.
__device__ __forceinline__ void wait_ge(const uint32_t* p, uint32_t tg, volatile uint32_t* fl) {
  const int tid = threadIdx.x, warp = tid >> 5;
  if ((tid & 31) == 0 && warp < NPOLL && (int)(*fl - tg) < 0) {
    const long long t0 = clock64();
    while (clock64() - t0 < (long long)warp * POLL_STAGGER) { if ((int)(*fl - tg) >= 0) break; }
    while ((int)(*fl - tg) < 0) {
      const uint32_t v = vload(p);
      if ((int)(v - tg) >= 0) { *fl = v; break; }
    }
  }
  __syncthreads();
}
#define WAIT_GE(ptr_, target_) wait_ge((const uint32_t*)(ptr_), (uint32_t)(target_), (volatile uint32_t*)sm.mb + (((const char*)(ptr_) - (const char*)C) >> 6))
#endif
// (the stores a signal covers are fenced where they are made: a fence after the next stage's loads have been
// issued would wait for those loads)
#define SIGNAL(ptr_)                                                                                        \
  {                                                                                                         \
    __syncthreads();                                                                                        \
    if (tid == 0) atomicAdd(ptr_, 1u);                                                                      \
  }

// Track B: the part of a tile's header one lane needs: its two rows' exponents, the neuron shift, where the
// escapes of its row group start and end (they are sorted by row % 8) and two entries of the escape list.
// Loaded by the warp that will finish the tile while the tile's quanta are still being read.
// Header: bytes 0..15 row exponents, 16..23 neuron shifts, 24..31 end of each row group's entries, 32.. entries
// [value:16 | k-step:8 | row:4 | col:4].
struct Hdr { uint32_t r0, r1, st, en, ea, eb; int cd; };
__device__ __forceinline__ Hdr hdr_load(WP hdr) {
  Hdr h;
#ifdef W12
  const int lane = threadIdx.x & 31, g8 = lane >> 2;
  h.r0 = hdr[g8]; h.r1 = hdr[g8 + 8]; h.cd = (int)(signed char)hdr[16 + g8];
  const uint32_t prev = hdr[24 + (g8 ? g8 - 1 : 0)];
  h.en = hdr[24 + g8]; h.st = g8 ? prev : 0;
  h.ea = *(const uint32_t*)(hdr + 32 + 4 * lane); h.eb = *(const uint32_t*)(hdr + 32 + 4 * (lane + 32));
#else
  h.r0 = h.r1 = h.st = h.en = h.ea = h.eb = 0; h.cd = 0;
#endif
  return h;
}
// Track B, stage C: the x values the escapes of a tile multiply (x lives in global memory there), fetched early
// into shared memory: lane i fetches for entries i and i + 32, all batch columns. xbase + b * xstride is column b.
__device__ __forceinline__ void xv_gather(const Hdr& h, const uint16_t* xbase, size_t xstride, int B, const Smem& sm) {
#ifdef W12
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int ia = ((h.ea >> 8) & 255) * 16 + (h.ea & 15), ib = ((h.eb >> 8) & 255) * 16 + (h.eb & 15);
  uint16_t va[MAXB], vb[MAXB];
#pragma unroll
  for (int b = 0; b < MAXB; b++) { const int bb = b < B ? b : 0; va[b] = xbase[bb * xstride + ia]; vb[b] = xbase[bb * xstride + ib]; }
  uint16_t* XV = sm.XV + warp * 512;
#pragma unroll
  for (int b = 0; b < MAXB; b++) { XV[lane * 8 + b] = va[b]; XV[(lane + 32) * 8 + b] = vb[b]; }
#endif
}
// Track B: finish the sum of a tile (this lane's rows g8 and g8 + 8, batch columns 2 t4 and 2 t4 + 1): undo the row
// exponents and add the products of the escaped weights of its row group. x0 / x1: the two batch columns of x as
// stored (times 2^XS), or null (column not in use); if xv, the values come from xv_gather instead. Only entries
// with k-step in [k_lo, k_hi) are added (a tile cut in halves has two owners).
__device__ __forceinline__ void tile_fix(float4& c, const Hdr& h, const uint16_t* x0, const uint16_t* x1, bool xv, int k_lo, int k_hi, const Smem& sm) {
#ifdef W12
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, t4 = lane & 3;
  uint32_t* EX = sm.EX + warp * 64;
  EX[lane] = h.ea; EX[lane + 32] = h.eb;
  const float s0 = __uint_as_float((127u + h.r0 - XS) << 23), s1 = __uint_as_float((127u + h.r1 - XS) << 23);
  c.x *= s0; c.y *= s0; c.z *= s1; c.w *= s1;
  const float un = __uint_as_float((uint32_t)(127 - XS) << 23);
  const uint16_t* XV = sm.XV + warp * 512;
  for (uint32_t i = h.st; i < h.en; i++) {
    const uint32_t e = EX[i];
    const int row = (e >> 4) & 15, k = (e >> 8) & 255, idx = k * 16 + (e & 15);
    const float w = (k >= k_lo && k < k_hi) ? bf2f((uint16_t)(e >> 16)) * un : 0.0f;
    const float a0 = x0 ? w * bf2f(xv ? XV[i * 8 + 2 * t4] : x0[idx]) : 0.0f, a1 = x1 ? w * bf2f(xv ? XV[i * 8 + 2 * t4 + 1] : x1[idx]) : 0.0f;
    if (row < 8) { c.x += a0; c.y += a1; } else { c.z += a0; c.w += a1; }
  }
#endif
}

#ifndef L1FLUSH
#define L1FLUSH 0
#endif
#ifndef ATT_EXP
#define ATT_EXP expf
#endif
#define PROFA(k_) { if (M->prof && blockIdx.x == 1 && threadIdx.x == 0) { long long t__ = clock64(); sm.prof[k_] += t__ - tprof; tprof = t__; } }
// Attention for one (sequence, K/V head), spread over nblk blocks.
// Under the weight stream an SM receives only its share of the queue (about 24 GB/s), so the K and V bytes of a
// long context must not sit on one SM. The position tiles are dealt to 8 chunks (tile P -> chunk P % 8); a chunk
// is complete in itself: scores, its own max m_c, weights w = exp(s - m_c) as bf16, their float32 sum z_c and the
// weighted sum o_c over V. The chunks are merged at the end: out = sum_c e^(m_c - M) o_c / sum_c e^(m_c - M) z_c,
// which is the softmax-weighted sum in exact arithmetic. A block runs the chunks c = cidx, cidx + nblk, ...;
// block 0 of the task merges. The arithmetic per chunk and the merge order do not depend on nblk, so a sequence
// gives the same bits alone (8 blocks per task) and in a batch of 8 (1 block per task).
// The new token's own K and V are used from shared memory, not read back from the cache.
__device__ __forceinline__ void attention_task(const Model* M, const Smem& sm, int l, int task, int cidx, int nblk, int pos, uint32_t G, bool nopart, long long& tprof, bool stamp) {
  PROFA(32);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
  const int b = task >> 2, g = task & 3;
  const LayerW& LW = M->L[l];
  Ctl* C = M->ctl;
  const int maxK = M->maxK;
  uint4* kc = M->kc + ((((size_t)l * MAXB + b) * 4 + g) * (size_t)maxK * 256);
  uint4* vc = M->vc + ((((size_t)l * MAXB + b) * 4 + g) * (size_t)maxK * 256);
  const int nk = (pos + 16) >> 4;          // position tiles in use, the new token's included
  const int cstar = (pos >> 4) & (NCH - 1);  // the chunk of the new token
  float* const SelfP = sm.O + 2048;      // [512]
  float* const Kn = SelfP + 512;         // [128]
  float* const Vn = Kn + 128;            // [128]
  float* const Wc = Vn + 128;            // [8 chunks][8 heads]
  float* const Part = sm.Part;
  // phase 0: q, k, v of the new token; RoPE; append K and V (one block of the task writes the cache)
  {
    const int hq = tid >> 6, iq = tid & 63;
    const uint32_t qraw = *(const uint32_t*)(M->qkvo + (size_t)b * QKV_ROWS + (8 * g + hq) * 128 + 2 * iq);
    const float* cs = M->rope + ((size_t)pos * 64 + iq) * 2;
    const float co = cs[0], si = cs[1];
    // threads < 128 also take a k pair (tid < 64) or a v pair; the other threads read the same addresses again
    const int te = tid & 127;
    const int rowe = (te < 64 ? 4096 + g * 128 + 2 * te : 4608 + g * 128 + 2 * (te - 64));
    const float* q0 = M->qkvp + (size_t)b * QKV_ROWS + rowe;
    const float a0 = q0[0], a1 = q0[1], b0 = q0[(size_t)MAXB * QKV_ROWS], b1 = q0[(size_t)MAXB * QKV_ROWS + 1];
    const uint32_t er = *(const uint32_t*)(M->qkvo + (size_t)b * QKV_ROWS + rowe);
    // rows of the tiles that two blocks share come as two partial sums
    const bool shared = rowe >= SHARED_ROW0 && !nopart;
    float e0 = shared ? rbf(a0 + b0) : bf2f(er & 0xffff), e1 = shared ? rbf(a1 + b1) : bf2f(er >> 16);
    float v0 = bf2f(qraw & 0xffff), v1 = bf2f(qraw >> 16);
    if (LW.rope) {
      const float r0 = __fadd_rn(__fmul_rn(v0, co), __fmul_rn(-v1, si));
      const float r1 = __fadd_rn(__fmul_rn(v1, co), __fmul_rn(v0, si));
      v0 = r0; v1 = r1;
    }
    *(uint32_t*)(sm.Qs + hq * 128 + 2 * iq) = (uint32_t)f2bf(v0) | ((uint32_t)f2bf(v1) << 16);
    const bool writer = cidx == cstar % nblk;
    if (tid < 64) {
      if (LW.rope) {
        const float r0 = __fadd_rn(__fmul_rn(e0, co), __fmul_rn(-e1, si));
        const float r1 = __fadd_rn(__fmul_rn(e1, co), __fmul_rn(e0, si));
        e0 = r0; e1 = r1;
      }
      // K cache: A operand with rows = positions, k = dims
      const int d = 2 * tid, p16 = pos & 15;
      const size_t off = ((((size_t)(pos >> 4) * 8 + (d >> 4)) * 32 + ((p16 & 7) * 4 + ((d & 7) >> 1))) * 16) + (((d & 15) >> 3) * 2 + (p16 >> 3)) * 4;
      const uint16_t u0 = f2bf(e0), u1 = f2bf(e1);
      if (writer) *(uint32_t*)((char*)kc + off) = (uint32_t)u0 | ((uint32_t)u1 << 16);
      Kn[d] = bf2f(u0); Kn[d + 1] = bf2f(u1);
    } else if (tid < 128) {
      // V cache: A operand with rows = dims, k = positions (the two dims of this pair live in different lanes)
      const int p16 = pos & 15, d = 2 * (tid - 64);
      const size_t off = ((((size_t)(d >> 4) * maxK + (pos >> 4)) * 32 + ((d & 7) * 4 + ((p16 & 7) >> 1))) * 16) + ((p16 >> 3) * 2 + ((d & 15) >> 3)) * 4 + (p16 & 1) * 2;
      const uint16_t u0 = f2bf(e0), u1 = f2bf(e1);
      if (writer) { *(uint16_t*)((char*)vc + off) = u0; *(uint16_t*)((char*)vc + off + 64) = u1; }  // dim d + 1: the next lane group
      Vn[d] = bf2f(u0); Vn[d + 1] = bf2f(u1);
    }
  }
  __syncthreads();
  // score of the new token against itself: 64 partial sums per head, summed by threads 0..7
  {
    const int hd = tid & 7, part = tid >> 3;
    const uint32_t qq = *(const uint32_t*)(sm.Qs + hd * 128 + 2 * part);
    SelfP[tid] = bf2f(qq & 0xffff) * Kn[2 * part] + bf2f(qq >> 16) * Kn[2 * part + 1];
  }
  __syncthreads();
  if (tid < 8) { float ss = 0; for (int i = 0; i < 64; i++) ss += SelfP[tid + 8 * i]; sm.Mx[16 + tid] = ss * SCALE; }
  PROFA(33);
  for (int c = cidx; c < NCH; c += nblk) {
    const int ntc = nk > c ? (nk - c + NCH - 1) / NCH : 0;   // tiles of this chunk: c, c + 8, ...  (at most 16)
    const bool self = c == cstar;
    // scores of the 8 query heads against the chunk's positions: one K tile per warp
    if (warp < ntc) {
      const uint32_t* qp = (const uint32_t*)(sm.Qs + g8 * 128) + t4;
      const uint4* p0 = kc + (size_t)(c + NCH * warp) * 256 + lane;
      uint4 a[8];
#pragma unroll
      for (int k = 0; k < 8; k++) a[k] = LDS4(p0 + k * 32);
      float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
      for (int k = 0; k < 8; k++) { const uint32_t b0 = qp[k * 8], b1 = qp[k * 8 + 4]; MMA(c0, c1, c2, c3, a[k], b0, b1); }
      const int q0 = warp * 16 + g8, h0 = 2 * t4;
      sm.S[h0 * 256 + q0] = c0 * SCALE;
      sm.S[(h0 + 1) * 256 + q0] = c1 * SCALE;
      sm.S[h0 * 256 + q0 + 8] = c2 * SCALE;
      sm.S[(h0 + 1) * 256 + q0 + 8] = c3 * SCALE;
    }
    PROFA(40);
    __syncthreads();
    PROFA(41);
    // the chunk's max per head (positions before the new token; its own score joins in its chunk)
    const int hd = tid & 7, slot = tid >> 3;
    float* S = sm.S + hd * 256;
    uint32_t valid = 0;  // bit m: local position slot + 64 m is a cached position before the new token
#pragma unroll
    for (int m4 = 0; m4 < 4; m4++) {
      const int q = slot + 64 * m4;
      const int gp = (c + NCH * (q >> 4)) * 16 + (q & 15);
      valid |= (uint32_t)((q < ntc * 16) & (gp < pos)) << m4;
    }
    // (a head's 64 threads are 4 lanes in each of the 16 warps: lanes hd, hd + 8, hd + 16, hd + 24)
    {
      float m = -INFINITY;
#pragma unroll
      for (int m4 = 0; m4 < 4; m4++) m = fmaxf(m, ((valid >> m4) & 1) ? S[slot + 64 * m4] : -INFINITY);
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 8));
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 16));
      if (lane < 8) sm.R[warp * 8 + lane] = m;
    }
    __syncthreads();
    PROFA(42);
    if (tid < 8) {
      float mm = self ? sm.Mx[16 + tid] : -INFINITY;
#pragma unroll
      for (int i = 0; i < 16; i++) mm = fmaxf(mm, sm.R[tid + 8 * i]);
      sm.Mx[tid] = mm;
    }
    __syncthreads();
    PROFA(43);
    {
      const float m = sm.Mx[hd];
      float z = 0;
      uint16_t* Pm = sm.Pm + hd * 272;
#pragma unroll
      for (int m4 = 0; m4 < 4; m4++) {
        const float e = ((valid >> m4) & 1) ? ATT_EXP(S[slot + 64 * m4] - m) : 0.0f;
        z += e;
        Pm[slot + 64 * m4] = f2bf(e);
      }
      z += __shfl_xor_sync(0xffffffffu, z, 8);
      z += __shfl_xor_sync(0xffffffffu, z, 16);
      if (lane < 8) sm.R[warp * 8 + lane] = z;
    }
    __syncthreads();
    PROFA(44);
    // o_c[head][dim] = sum over the chunk's positions of w V: two warps per 16 dims, 8 position tiles each
    {
      const int rt = warp & 7, half = warp >> 3;
      const uint4* p = vc + (size_t)rt * maxK * 32 + lane;
      const uint32_t* pb = (const uint32_t*)(sm.Pm + g8 * 272) + t4 + half * 64;
      uint4 a[8];
#pragma unroll
      for (int u = 0; u < 8; u++) { const int i = half * 8 + u; a[u] = LDS4(p + (size_t)(c + NCH * (i < ntc ? i : 0)) * 32); }
      float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
      for (int u = 0; u < 8; u++) { const uint32_t b0 = pb[u * 8], b1 = pb[u * 8 + 4]; MMA(c0, c1, c2, c3, a[u], b0, b1); }
      float* O = sm.O + half * 1024;
      const int d0 = rt * 16 + g8, h0 = 2 * t4;
      O[h0 * 128 + d0] = c0; O[(h0 + 1) * 128 + d0] = c1; O[h0 * 128 + d0 + 8] = c2; O[(h0 + 1) * 128 + d0 + 8] = c3;
      if (tid < 8) {
        const float es = self ? ATT_EXP(sm.Mx[16 + tid] - sm.Mx[tid]) : 0.0f;
        float zz = es;
#pragma unroll
        for (int i = 0; i < 16; i++) zz += sm.R[tid + 8 * i];
        Part[c * PREC + 1024 + tid] = zz;
        Part[c * PREC + 1032 + tid] = sm.Mx[tid];
        sm.Mx[24 + tid] = rbf(es);
      }
    }
    PROFA(45);
    __syncthreads();
    PROFA(46);
    for (int i = tid; i < 1024; i += 512)
      Part[c * PREC + i] = sm.O[i] + sm.O[1024 + i] + (self ? sm.Mx[24 + (i >> 7)] * Vn[i & 127] : 0.0f);
    __syncthreads();
  }
  PROFA(34);
  if (nblk > 1) {
#if ATT_STAMP
    // Every partial value goes out with the layer's stamp in the same 8-byte exchange; the merging block polls the
    // values themselves: no wait for the exchanges' returns, no separate stamp, one round trip less (e0342).
    unsigned long long* gp = (unsigned long long*)M->attp + (size_t)task * NCH * PREC;
    const unsigned long long st = (unsigned long long)(G + 1) << 32;
    for (int c = cidx; c < NCH; c += nblk) {
      if (c % nblk == 0 && cidx == 0) continue;   // (the merging block's own chunks stay in shared memory)
      atomicExch(gp + c * PREC + tid, st | __float_as_uint(Part[c * PREC + tid]));
      atomicExch(gp + c * PREC + 512 + tid, st | __float_as_uint(Part[c * PREC + 512 + tid]));
      if (tid < PREC - 1024) atomicExch(gp + c * PREC + 1024 + tid, st | __float_as_uint(Part[c * PREC + 1024 + tid]));
    }
    if (cidx != 0) return;
    for (;;) {
      float v[NCH][2], w[NCH];
      bool ok = true;
#pragma unroll
      for (int c = 0; c < NCH; c++) {
        if (c % nblk == 0) continue;
        const unsigned long long a0 = *(const volatile unsigned long long*)(gp + c * PREC + tid);
        const unsigned long long a1 = *(const volatile unsigned long long*)(gp + c * PREC + 512 + tid);
        const unsigned long long a2 = *(const volatile unsigned long long*)(gp + c * PREC + 1024 + (tid & 15));
        ok = ok & ((uint32_t)(a0 >> 32) == G + 1) & ((uint32_t)(a1 >> 32) == G + 1) & ((uint32_t)(a2 >> 32) == G + 1);
        v[c][0] = __uint_as_float((uint32_t)a0); v[c][1] = __uint_as_float((uint32_t)a1); w[c] = __uint_as_float((uint32_t)a2);
      }
      if (__syncthreads_and(ok)) {
#pragma unroll
        for (int c = 0; c < NCH; c++) if (c % nblk != 0) { Part[c * PREC + tid] = v[c][0]; Part[c * PREC + 512 + tid] = v[c][1]; if (tid < 16) Part[c * PREC + 1024 + tid] = w[c]; }
        break;
      }
    }
    __syncthreads();
#else
    // (all loads of a group in straight-line code: a branch or a dependent store between two loads makes them two
    // round trips)
    float* gp = M->attp + (size_t)task * NCH * PREC;
    uint32_t xa = 0;
    for (int c = cidx; c < NCH; c += nblk) {
      const float v0 = Part[c * PREC + tid], v1 = Part[c * PREC + 512 + tid];
      const uint32_t r0 = xstoref(gp + c * PREC + tid, v0);
      const uint32_t r1 = xstoref(gp + c * PREC + 512 + tid, v1);
      xa ^= r0 ^ r1;
      if (tid < PREC - 1024) xa ^= xstoref(gp + c * PREC + 1024 + tid, Part[c * PREC + 1024 + tid]);
    }
    XDONE(xa);
    __syncthreads();
    if (tid == 0) atomicExch(&C->tst[task][cidx], G + 1);
    if (cidx != 0) return;
    if (tid == 0) {
      for (;;) {
        bool ok = true;
#pragma unroll
        for (int c = 0; c < NCH; c++) ok = ok & (c >= nblk || vload(&C->tst[task][c]) == G + 1);
        if (ok) break;
      }
    }
    __syncthreads();
    {
      float v[NCH][2], w[NCH];
#pragma unroll
      for (int c = 0; c < NCH; c++) { v[c][0] = gp[c * PREC + tid]; v[c][1] = gp[c * PREC + 512 + tid]; w[c] = gp[c * PREC + 1024 + (tid & 15)]; }
#pragma unroll
      for (int c = 0; c < NCH; c++) if (c % nblk != 0) { Part[c * PREC + tid] = v[c][0]; Part[c * PREC + 512 + tid] = v[c][1]; if (tid < 16) Part[c * PREC + 1024 + tid] = w[c]; }
    }
    __syncthreads();
  #endif
  }
  PROFA(35);
  // merge the chunks
  if (tid < 64) {
    const int h = tid & 7, c = tid >> 3;
    float mx = -INFINITY;
#pragma unroll
    for (int cc = 0; cc < NCH; cc++) mx = fmaxf(mx, Part[cc * PREC + 1032 + h]);
    Wc[tid] = ATT_EXP(Part[c * PREC + 1032 + h] - mx);
  }
  __syncthreads();
  if (tid < 8) {
    float z = 0;
#pragma unroll
    for (int c = 0; c < NCH; c++) z += Wc[c * 8 + tid] * Part[c * PREC + 1024 + tid];
    sm.Mx[8 + tid] = z;
  }
  __syncthreads();
  uint32_t xfin = 0;
  for (int i = tid; i < 1024; i += 512) {
    const int h = i >> 7;
    float acc = 0;
#pragma unroll
    for (int c = 0; c < NCH; c++) acc += Wc[c * 8 + h] * Part[c * PREC + i];
    // two bf16 per 32-bit word: the even lane stores its own value and its odd neighbour's
    const uint32_t mine = f2bf(rbf(acc / sm.Mx[8 + h]) * XSF);
    const uint32_t other = __shfl_xor_sync(0xffffffffu, mine, 1);
    if (!(lane & 1)) xfin ^= xstore((uint32_t*)(M->attn + (size_t)b * 4096 + g * 1024 + i), mine | (other << 16));
  }
  XDONE(xfin);
  if (stamp) { __syncthreads(); if (tid == 0) atomicExch(&C->adone[task], G + 1); }   // (only read by o_proj in stage B)
  PROFA(36);
}

// RMSNorm of the residual into shared memory: float32 mean of squares, x * rsqrt rounded to bf16, times the weight
struct NoHook { __device__ void operator()() const {} };
template <class F = NoHook>
__device__ __forceinline__ void prologue_norm(const Model* M, const Smem& sm, uint2 nw, int B, const F& after_loads = F()) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint2 xr[MAXB];
#pragma unroll
  for (int b = 0; b < MAXB; b++) if (b < B) xr[b] = ((const uint2*)(M->x + (size_t)b * HID))[tid];
  after_loads();   // (issued after the loads of x: in the queue they come after them)
  for (int b = 0; b < B; b++) {
    float v0 = bf2f(xr[b].x & 0xffff), v1 = bf2f(xr[b].x >> 16), v2 = bf2f(xr[b].y & 0xffff), v3 = bf2f(xr[b].y >> 16);
    float ss = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if (lane == 0) sm.R[warp * 8 + b] = ss;
  }
  __syncthreads();
  // (the weight carries the 2^XS of the stored x operand: rounding commutes with a power of two)
  const float w0 = bf2f(nw.x & 0xffff) * XSF, w1 = bf2f(nw.x >> 16) * XSF, w2 = bf2f(nw.y & 0xffff) * XSF, w3 = bf2f(nw.y >> 16) * XSF;
#ifndef NORM_OLD
  // batch > 1: the totals once per column (thread b) instead of once per thread: 16 shared-memory reads per column
  // and thread were 2.4 us of a layer at batch 8. Same partial sums in the same order: batch-invariant.
  if (B > 1) {
    if (tid < B) {
      float tot = 0;
#pragma unroll
      for (int i = 0; i < 16; i++) tot += sm.R[i * 8 + tid];
      sm.R[128 + tid] = rsqrtf(tot * (1.0f / HID) + 1e-6f);
    }
    __syncthreads();
  }
#endif
  for (int b = 0; b < B; b++) {
    float rs;
#ifndef NORM_OLD
    if (B > 1) rs = sm.R[128 + b];
    else
#endif
    {
      float tot = 0;
#pragma unroll
      for (int i = 0; i < 16; i++) tot += sm.R[i * 8 + b];
      rs = rsqrtf(tot * (1.0f / HID) + 1e-6f);
    }
#ifndef NORM_SLOW
    // (x * rs rounded to bf16, times the weight, rounded again: two cvt.rn.bf16x2 per pair of elements)
    const uint32_t r01 = f2bf2(bflo(xr[b].x) * rs, bfhi(xr[b].x) * rs), r23 = f2bf2(bflo(xr[b].y) * rs, bfhi(xr[b].y) * rs);
    uint2 o;
    o.x = f2bf2(w0 * bflo(r01), w1 * bfhi(r01));
    o.y = f2bf2(w2 * bflo(r23), w3 * bfhi(r23));
#else
    float v0 = bf2f(xr[b].x & 0xffff), v1 = bf2f(xr[b].x >> 16), v2 = bf2f(xr[b].y & 0xffff), v3 = bf2f(xr[b].y >> 16);
    uint2 o;
    o.x = (uint32_t)f2bf(w0 * rbf(v0 * rs)) | ((uint32_t)f2bf(w1 * rbf(v1 * rs)) << 16);
    o.y = (uint32_t)f2bf(w2 * rbf(v2 * rs)) | ((uint32_t)f2bf(w3 * rbf(v3 * rs)) << 16);
#endif
    ((uint2*)(sm.xn + b * HID))[tid] = o;
  }
  __syncthreads();
}

// leader block: router logits (two partial sums per logit from the router blocks), top 8 per sequence, sigmoid weights, union of the
// chosen experts over the batch
#ifndef TOPK_RANK
#define TOPK_RANK 0
#endif
#ifndef LEADER_ORDER
#define LEADER_ORDER 0
#endif
#ifndef LEADER_FIRST
#define LEADER_FIRST 1
#endif
#define PROFR(k_) { if (M->prof && tid == 0) { long long t__ = clock64(); sm.prof[k_] += t__ - tprof; tprof = t__; } }
// leader block: router logits (two stamped partial sums per logit, straight from the router blocks), top 8 per
// sequence, sigmoid weights, union of the chosen experts over the batch
__device__ __forceinline__ void leader_route(const Model* M, const Smem& sm, int l, int B, uint32_t stamp, long long& tprof) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  Ctl* C = M->ctl;
  uint32_t* masks = sm.W + 8;          // [MAXB][4] the chosen experts of a sequence, 32 per word
  uint32_t* um = sm.W + 8 + 32 + 64;   // [4] union of the batch
  float v[4] = {0, 0, 0, 0};           // this lane's router logits: experts lane, lane + 32, lane + 64, lane + 96
  uint32_t mw[4] = {0, 0, 0, 0};       // the sequence's mask words (the same in every lane of the warp)
  uint32_t mem = 0;                    // bit j: expert lane + 32 j is chosen
#if TOPK_RANK
  int* const KEY = (int*)sm.slot + 8192;   // [MAXB][128] keys (32 KB into the slot area: free while the leader routes)
#endif
  if (warp < B) {
    const int b = warp;
    const uint2* r0 = M->rl + b * 128 + lane;
    // a partial sum is valid once it carries this layer's stamp (the router blocks write value and stamp in one store)
    unsigned long long p[8];
    for (;;) {
#pragma unroll
      for (int j = 0; j < 4; j++) {
        p[j] = *(const volatile unsigned long long*)(r0 + 32 * j);
        p[4 + j] = *(const volatile unsigned long long*)(r0 + MAXB * 128 + 32 * j);
      }
      bool ok = true;
#pragma unroll
      for (int j = 0; j < 8; j++) ok = ok && (uint32_t)(p[j] >> 32) == stamp;
      if (ok) break;
    }
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = rbf(__uint_as_float((uint32_t)p[j]) + __uint_as_float((uint32_t)p[4 + j]));
    PROFR(28);
    // 8 rounds of a warp-wide max over sortable integer keys: (bf16 logit as an ordered 16-bit integer) << 7 |
    // (127 - expert), so that the larger logit wins and, at a tie, the lower expert index. No data-dependent branch,
    // and nothing but the membership bit is kept: the order of the eight does not matter further on.
    if (!M->force) {
      int key[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int u = (int)__float_as_uint(v[j]);
        const int k = u ^ ((u >> 31) & 0x7fffffff);
        key[j] = ((k >> 16) << 7) | (127 - (lane + 32 * j));
      }
#if TOPK_RANK
#pragma unroll
      for (int j = 0; j < 4; j++) KEY[b * 128 + lane + 32 * j] = key[j];
#else
#pragma unroll 1
      for (int r = 0; r < TOPK; r++) {
        const int best = __reduce_max_sync(0xffffffffu, max(max(key[0], key[1]), max(key[2], key[3])));
#pragma unroll
        for (int j = 0; j < 4; j++) { const bool hit = key[j] == best; mem |= (hit ? 1u : 0u) << j; key[j] = hit ? (int)0x80000000 : key[j]; }
      }
#endif
    } else {
      for (int r = 0; r < TOPK; r++) {
        const int e = M->force[((size_t)l * MAXB + b) * TOPK + r];
#pragma unroll
        for (int j = 0; j < 4; j++) mem |= (e == lane + 32 * j ? 1u : 0u) << j;
      }
    }
    if (!TOPK_RANK || M->force) {
#pragma unroll
      for (int j = 0; j < 4; j++) mw[j] = __ballot_sync(0xffffffffu, (mem >> j) & 1u);
      if (lane < 4) masks[b * 4 + lane] = mw[lane];
    }
  }
  __syncthreads();
#if TOPK_RANK
  if (!M->force) {
    // the rank of every key among the 128 of its sequence, one thread per key (keys are unique: the expert index is
    // in them): chosen = fewer than 8 larger keys. The same choice as 8 rounds of a warp max, in one pass.
    for (int t = tid; t < 128 * B; t += 512) {
      const int* K = KEY + (t >> 7) * 128;
      const int k = K[t & 127];
      int cnt = 0;
#pragma unroll 8
      for (int i = 0; i < 128; i += 4) { const int4 q = *(const int4*)(K + i); cnt += (q.x > k) + (q.y > k) + (q.z > k) + (q.w > k); }
      const uint32_t m = __ballot_sync(0xffffffffu, cnt < TOPK);
      if (lane == 0) masks[(t >> 7) * 4 + ((t & 127) >> 5)] = m;
    }
    __syncthreads();
    if (warp < B) {
#pragma unroll
      for (int j = 0; j < 4; j++) { mw[j] = masks[warp * 4 + j]; mem |= ((mw[j] >> lane) & 1u) << j; }
    }
  }
#endif
  // The union goes out at once, stamped: every block starts gate|up from it (the slot of an expert is its rank in
  // the union, which each block works out itself). Slots per sequence, weights and sequence masks are only needed
  // in stage C and follow in leader_details.
  if (tid < 4) {
    uint32_t u = 0;
    for (int b = 0; b < B; b++) u |= masks[b * 4 + tid];
    um[tid] = u;
    XDONE((uint32_t)atomicExch(&C->um[tid], ((unsigned long long)stamp << 32) | u));
  }
  // (the logits stay in shared memory for leader_details: Cst is free until stage C)
  if (warp < B) { float* RL = (float*)sm.Cst + warp * 128 + lane;
#pragma unroll
    for (int j = 0; j < 4; j++) RL[32 * j] = v[j]; }
  __syncthreads();
  PROFR(29);
}

// leader block: slots per sequence, weights, sequence masks of the union (needed in stage C)
__device__ __forceinline__ void leader_details(const Model* M, const Smem& sm, int l, int B, long long& tprof) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  Dyn& D = M->dyn[l];
  const uint32_t* masks = sm.W + 8;
  const uint32_t* um = sm.W + 8 + 32 + 64;
  if (tid < 128) {
    const int e = tid, wi = e >> 5;
    if ((um[wi] >> (e & 31)) & 1) {
      int slot = __popc(um[wi] & ((1u << (e & 31)) - 1));
      for (int j = 0; j < wi; j++) slot += __popc(um[j]);
      int bm = 0;
      for (int b = 0; b < B; b++) bm |= ((masks[b * 4 + wi] >> (e & 31)) & 1) << b;
      D.slot_expert[slot] = e;
      D.bmask[slot] = bm;
    }
  }
  if (warp < B) {
    const int b = warp;
    const uint32_t below = (1u << lane) - 1;
    const float* RL = (const float*)sm.Cst + b * 128 + lane;
    int slot = 0, rank = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint32_t mwj = masks[b * 4 + j];
      if ((mwj >> lane) & 1u) {
        const int sl = slot + __popc(um[j] & below), rk = rank + __popc(mwj & below);
        D.bslot[b][rk] = sl;
        D.bw[b][rk] = rbf(1.0f / (1.0f + expf(-RL[32 * j])));
      }
      slot += __popc(um[j]); rank += __popc(mwj);
    }
  }
  PROFR(30);
  if (tid == 0) D.U = __popc(um[0]) + __popc(um[1]) + __popc(um[2]) + __popc(um[3]);
  FENCE();
}

// greedy choice from per-block candidates (leader block)
__device__ __forceinline__ void leader_argmax(const Model* M, const Launch& A, int step_done, int NB) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (warp < A.B) {
    float best = -INFINITY; int bi = 0x7fffffff;
    for (int j = lane; j < NB; j += 32) {
      float v = M->candv[j * MAXB + warp]; int ix = M->candi[j * MAXB + warp];
      if (v > best || (v == best && ix < bi)) { best = v; bi = ix; }
    }
    for (int o = 16; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, best, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (lane == 0) {
      M->curtok[warp] = bi;
      unsigned long long* o = (unsigned long long*)A.tok_out + (A.kmajor ? (size_t)step_done * A.B + warp : (size_t)warp * A.nsteps + step_done);
      *(volatile unsigned long long*)&M->candv[0] = atomicExch(o, (unsigned long long)bi);  // (wait for it: the progress word follows)
    }
  }
  __syncthreads();
  if (tid == 0) atomicExch(&M->ctl->prog, (uint32_t)(step_done + 1));
}

#ifndef PROFBLK
#define PROFBLK 0   // the block whose timeline goes to M->prof[0..31] (debug)
#endif
// (debug) cycles every block spends in the three barrier waits of a layer: M->prof[64 + 4 j + k]
#define PROFW0() { if (M->prof && tid == 0) twait = clock64(); }
#define PROFW(k_) { if (M->prof && tid == 0) sm.prof[56 + k_] += clock64() - twait; }
#ifdef TRACE
// (debug) %globaltimer at stage boundaries of step A.nsteps - 1 (or the last generated step), kept in registers of
// thread 0 for the current layer and stored at the end of the layer
#define TR(k_) { if (tid == 0 && M->trace && step == trace_step) { unsigned long long t__; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t__)); trv[k_] = t__; } }
#define TRFLUSH() { if (tid == 0 && M->trace && step == trace_step) { for (int k__ = 0; k__ < 12; k__++) M->trace[((size_t)j * NLAYER + l) * 12 + k__] = trv[k__]; } }
#else
#define TR(k_) {}
#define TRFLUSH() {}
#endif
#define PROF(k_) { if (M->prof && blockIdx.x == PROFBLK && tid == 0) { long long t__ = clock64(); sm.prof[k_] += t__ - tprof; tprof = t__; } }

// Gate|up shares of a layer with n tiles. The blocks do not start stage B together: the leader routes first, an
// attention block has its attention task first. Each class of blocks gets a delay d (in 1/8 of the time one block
// needs for one tile), and the shares are such that all classes finish together: a block of class c takes T - d_c
// tiles, T from sum_c N_c (T - d_c) = n; a class with d_c >= T takes none. Ranges of consecutive tiles, in the
// order ordinary blocks (33..131), leader, attention blocks: this block's count and first tile.
// The delays are measured (tools/test_engine.py --prof prints every class's wait at the stage B barrier).
__device__ __forceinline__ void gu_share(int j, int B, int n, int nblk, int dL, int dAN, int dAM, int dI, int dO, int& ct, int& off) {
  const int nbusy = min(32, 4 * B * nblk);     // attention blocks with a task: 1 .. nbusy; merging ones: (j - 1) % nblk == 0
  const int nmerge = nbusy / nblk;
  // classes: 0 ordinary with a residual row tile (33..127), 1 idle attention, 2 leader, 3 attention, 4 attention +
  // merge, 5 ordinary without a row tile (128..131). dO: the o_proj tile that blocks 0..127 run in stage B (OPB)
  const int N[6] = {95, 32 - nbusy, 1, nbusy - nmerge, nmerge, 4};
  const int d[6] = {dO, dI + dO, dL + dO, dAN + dO, dAM + dO, 0};
  bool on[6] = {true, N[1] > 0, true, N[3] > 0, N[4] > 0, true};
  int T8 = 0;
  for (int it = 0; it < 6; it++) {
    int sn = 0, sd = 0;
    for (int c = 0; c < 6; c++) if (on[c]) { sn += N[c]; sd += N[c] * d[c]; }
    T8 = (8 * n + sd) / sn;
    int worst = -1;
    for (int c = 0; c < 5; c++) if (on[c] && d[c] >= T8 && (worst < 0 || d[c] > d[worst])) worst = c;
    if (worst < 0) break;
    on[worst] = false;
  }
  int c[6], tot = 0;
  for (int k = 0; k < 6; k++) { c[k] = on[k] ? max(0, (T8 - d[k]) >> 3) : 0; tot += N[k] * c[k]; }
  const int r = n - tot, q = r / 132, rm = r % 132;   // what rounding left over: one more tile for the first rm blocks
#define GS_CLS(jj_) ((jj_) == 0 ? 2 : (jj_) >= 128 ? 5 : (jj_) > 32 ? 0 : (jj_) > nbusy ? 1 : (((jj_) - 1) % nblk == 0 ? 4 : 3))
#define GS_RANK(jj_) ((jj_) == 0 ? 99 : (jj_) > 32 ? (jj_) - 33 : 99 + (jj_))
#define GS_CT(jj_) (c[GS_CLS(jj_)] + q + (GS_RANK(jj_) < rm ? 1 : 0))
  ct = GS_CT(j);
  const int rk = GS_RANK(j), r99 = min(rk, 99);
  off = min(r99, 95) * (c[0] + q) + max(0, r99 - 95) * (c[5] + q) + min(r99, rm);
  if (rk > 99) { off += GS_CT(0); for (int jj = 1; jj < j; jj++) off += GS_CT(jj); }
}
// Delays in 1/8 tile, fitted to the waits at the stage B barrier at contexts 64-320 and 800-1056, batch 1 and 8
// (e0206, e0207). Leader: its routing chain. Attention block with cpb chunks of a task and ntc K tiles per chunk:
// 26 + cpb (14 + 0.6 ntc), + 0.7 ntc when the task is shared by several blocks, + GS_AX for the block that merges.
#ifndef GS_DL
#define GS_DL 16
#endif
#ifndef GS_A0
#define GS_A0 26
#endif
#ifndef GS_A1
#define GS_A1 112   // (1/64 tile per chunk)
#endif
#ifndef GS_A2
#define GS_A2 5     // (1/64 tile per chunk and K tile)
#endif
#ifndef GS_A3
#define GS_A3 6     // (1/64 tile per K tile, shared tasks)
#endif
#ifndef GS_AX
#define GS_AX 20
#endif
#ifndef ATTB1
#define ATTB1 8
#endif
#ifndef OPB
#define OPB 1       // o_proj in stage B, right after the block's gate|up share (needs only the attention output)
#endif
#ifndef GS_DO
#define GS_DO 20    // its delay: the o_proj tile is 256 k-steps = 2 gate|up tiles
#endif
#ifndef XO_B1
#define XO_B1 0   // early residual rows at batch 1 also in the o_proj-in-B instantiation
#endif
#ifndef GS_DO1
#define GS_DO1 GS_DO   // (batch 1)
#endif
#ifndef GS_DI
#define GS_DI 4
#endif

// The schedule's block number. The card's SMs do not stream equally fast (on this card 24 SM ids get 34% more of the
// HBM queue, always the same ones: bench/smid_bench.cu); the engine measures that when it loads and maps the roles
// on the routing chain to fast SMs (engine/api.py). Without a table: the block index.
__device__ __forceinline__ int my_role(const Model* M) {
  if (!M->role) return blockIdx.x;
  uint32_t s; asm("mov.u32 %0, %%smid;" : "=r"(s));
  return M->role[s & 255];
}
// (inlined: as a real call it costs 480 us per step, e0181: 448 bytes of stack are saved and restored around it)
#ifdef HEAD_NOINLINE
#define HEAD_ATTR __noinline__
#else
#define HEAD_ATTR __forceinline__
#endif
__device__ HEAD_ATTR void head_stage(const Model* M, const Smem& sm, const Launch& A, int step, long long& tprof) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
  Ctl* C = M->ctl;
  const int b0i = 2 * t4, b1i = 2 * t4 + 1;
  const int B = A.B, NB = gridDim.x, j = my_role(M), NW = 16 * NB;
  const bool act = g8 < B;
  const int gw = warp * NB + j;
  const uint32_t* xn_b = act ? (const uint32_t*)(sm.xn + g8 * HID) + t4 : sm.zero;
  const uint16_t* const xs0 = b0i < B ? sm.xn + b0i * HID : nullptr;
  const uint16_t* const xs1 = b1i < B ? sm.xn + b1i * HID : nullptr;
  {
      const uint2 nw = ((const uint2*)M->normf)[tid];
      WAIT_GE(&C->cntC, NB * (A.g0 + (step + 1) * NLAYER));
      if (M->dbg_x && j == 0) for (int b = 0; b < B; b++) for (int i = tid; i < HID; i += 512) M->dbg_x[((size_t)NLAYER * MAXB + b) * HID + i] = M->x[(size_t)b * HID + i];
      prologue_norm(M, sm, nw, B);
      PROF(8);
      float bv0 = -INFINITY, bv1 = -INFINITY; int bi0 = 0x7fffffff, bi1 = 0x7fffffff;
      const WP hp = M->head;
      float* const lg = A.logits_out;
      // ---- greedy path with a sketch: pass 1 scores every row from the int8 sketch, with a certified error bound;
      // the best lower bound of the whole vocabulary (one exchange between the blocks) is the threshold; pass 2
      // computes exactly, with the code of the full head, only the tiles that hold a row whose upper bound reaches it.
      const bool sketch = A.gen && M->sk != nullptr;
      float T0 = -INFINITY, T1 = -INFINITY;
      float2* const UT = (float2*)((char*)sm.slot + 32768);   // [16 warps][8 tiles][32 lanes] upper bounds of a tile's rows, per batch column pair
      if (sketch) {
        uint32_t* const dig = (uint32_t*)sm.slot;              // int8 digits: hi [8][2048] then lo [8][2048]
        const float unx = __uint_as_float((uint32_t)(127 - XS) << 23);
        for (int b = 0; b < B; b++) {
          const uint2 xv = ((const uint2*)(sm.xn + b * HID))[tid];
          const float v0 = bf2f(xv.x & 0xffff) * unx, v1 = bf2f(xv.x >> 16) * unx, v2 = bf2f(xv.y & 0xffff) * unx, v3 = bf2f(xv.y >> 16) * unx;
          float ss = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3, mx = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
          for (int o = 16; o > 0; o >>= 1) { ss += __shfl_xor_sync(0xffffffffu, ss, o); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
          if (lane == 0) { sm.R[warp * 8 + b] = ss; sm.R[128 + warp * 8 + b] = mx; }
        }
        __syncthreads();
        for (int b = 0; b < B; b++) {
          float amax = 0;
#pragma unroll
          for (int i = 0; i < 16; i++) amax = fmaxf(amax, sm.R[128 + i * 8 + b]);
          const float t = fmaxf(amax, 1e-30f) * (1.0f / 16256.0f), it = 1.0f / t;
          const uint2 xv = ((const uint2*)(sm.xn + b * HID))[tid];
          const float v[4] = {bf2f(xv.x & 0xffff) * unx, bf2f(xv.x >> 16) * unx, bf2f(xv.y & 0xffff) * unx, bf2f(xv.y >> 16) * unx};
          uint32_t hw = 0, lw = 0; float rr = 0;
#pragma unroll
          for (int k = 0; k < 4; k++) {
            const int P = max(-16256, min(16256, __float2int_rn(v[k] * it)));
            const int lo = ((P + 64) & 127) - 64, hi = (P - lo) >> 7;      // P = 128 hi + lo, both int8
            const float r = v[k] - t * (float)P;
            rr += r * r;
            hw |= (uint32_t)(hi & 0xff) << (8 * k); lw |= (uint32_t)(lo & 0xff) << (8 * k);
          }
          for (int o = 16; o > 0; o >>= 1) rr += __shfl_xor_sync(0xffffffffu, rr, o);
          if (lane == 0) sm.R[256 + warp * 8 + b] = rr;
          dig[b * 512 + tid] = hw; dig[4096 + b * 512 + tid] = lw;
        }
        __syncthreads();
        if (tid < B) {
          float tot = 0, rt = 0, amax = 0;
#pragma unroll
          for (int i = 0; i < 16; i++) { tot += sm.R[i * 8 + tid]; rt += sm.R[256 + i * 8 + tid]; amax = fmaxf(amax, sm.R[128 + i * 8 + tid]); }
          sm.W[64 + tid] = __float_as_uint(fmaxf(amax, 1e-30f) * (1.0f / 16256.0f));           // t
          sm.W[72 + tid] = __float_as_uint(sqrtf(tot) * 1.001f);                                 // >= ||x||
          sm.W[80 + tid] = __float_as_uint((sqrtf(rt) + amax * 1.2e-5f) * 1.01f);                // >= ||r|| (with the rounding of r itself)
        }
        __syncthreads();
        // B fragment of m16n8k32 (bench/mma8_probe.cu): thread (g, t) holds k = 8 t + 0..7 of column g in b0, b1
        const uint32_t* dh = act ? dig + g8 * 512 + 2 * t4 : sm.zero;
        const uint32_t* dl = act ? dig + 4096 + g8 * 512 + 2 * t4 : sm.zero;
        const float t0 = b0i < B ? __uint_as_float(sm.W[64 + b0i]) : 0.f, t1 = b1i < B ? __uint_as_float(sm.W[64 + b1i]) : 0.f;
        const float X0 = b0i < B ? __uint_as_float(sm.W[72 + b0i]) : 0.f, X1 = b1i < B ? __uint_as_float(sm.W[72 + b1i]) : 0.f;
        const float R0 = b0i < B ? __uint_as_float(sm.W[80 + b0i]) : 0.f, R1 = b1i < B ? __uint_as_float(sm.W[80 + b1i]) : 0.f;
        float lw0 = -INFINITY, lw1 = -INFINITY;
        int n = 0;
        for (int i = gw; i < VOCAB / 16; i += NW, n++) {
          const WP p = M->sk + (size_t)i * SKT;
          const uint4 ha = *(const uint4*)(p + 64 * 512 + g8 * 32);
          const uint2 hb = *(const uint2*)(p + 64 * 512 + g8 * 32 + 16);
          int h0 = 0, h1 = 0, h2 = 0, h3 = 0, l0 = 0, l1 = 0, l2 = 0, l3 = 0;
#pragma unroll 1
          for (int m = 0; m < 64; m += SKQ) {
            uint4 a[SKQ];
#pragma unroll
            for (int u = 0; u < SKQ; u++) a[u] = LDS4(p + (size_t)(m + u) * 512 + lane * 16);
#pragma unroll
            for (int u = 0; u < SKQ; u++) {
              MMA8(h0, h1, h2, h3, a[u], dh[(m + u) * 8], dh[(m + u) * 8 + 1]);
              MMA8(l0, l1, l2, l3, a[u], dl[(m + u) * 8], dl[(m + u) * 8 + 1]);
            }
          }
          const float sa = __uint_as_float(ha.x), sb = __uint_as_float(ha.y), ea = __uint_as_float(ha.z), eb = __uint_as_float(ha.w);
          const float qa = __uint_as_float(hb.x), qb = __uint_as_float(hb.y);
          // c0: (row g8, column b0i), c1: (row g8, b1i), c2: (row g8 + 8, b0i), c3: (row g8 + 8, b1i)
          const float s00 = sa * (t0 * ((float)h0 * 128.0f + (float)l0)), s01 = sa * (t1 * ((float)h1 * 128.0f + (float)l1));
          const float s10 = sb * (t0 * ((float)h2 * 128.0f + (float)l2)), s11 = sb * (t1 * ((float)h3 * 128.0f + (float)l3));
          const float e00 = ea * X0 + qa * R0, e01 = ea * X1 + qa * R1, e10 = eb * X0 + qb * R0, e11 = eb * X1 + qb * R1;
          lw0 = fmaxf(lw0, fmaxf(s00 - e00, s10 - e10)); lw1 = fmaxf(lw1, fmaxf(s01 - e01, s11 - e11));
          UT[(warp * 8 + n) * 32 + lane] = make_float2(b0i < B ? fmaxf(s00 + e00, s10 + e10) : -INFINITY, b1i < B ? fmaxf(s01 + e01, s11 + e11) : -INFINITY);
        }
        // the best lower bound of every batch column: block, then all blocks (stored with an exchange: no fence)
        float* LV = (float*)sm.slot + 16384;
        LV[b0i * 128 + warp * 8 + g8] = lw0; LV[b1i * 128 + warp * 8 + g8] = lw1;
        __syncthreads();
        float* const gth = M->candv + MAXB * 256;
        if (warp < B) {
          float best = -INFINITY;
#pragma unroll
          for (int q = 0; q < 4; q++) best = fmaxf(best, LV[warp * 128 + lane + 32 * q]);
          for (int o = 16; o > 0; o >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
          if (lane == 0) XDONE(xstoref(gth + j * MAXB + warp, best));
        }
        SIGNAL(&C->cntR);
        WAIT_GE(&C->cntR, NB * (A.h0 + (step - (A.P - 1)) + 1));
        if (warp < B) {
          float best = -INFINITY;
          for (int jj = lane; jj < NB; jj += 32) best = fmaxf(best, *(const volatile float*)(gth + jj * MAXB + warp));
          for (int o = 16; o > 0; o >>= 1) best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, o));
          if (lane == 0) sm.W[88 + warp] = __float_as_uint(best);
        }
        __syncthreads();
        T0 = b0i < B ? __uint_as_float(sm.W[88 + b0i]) : INFINITY; T1 = b1i < B ? __uint_as_float(sm.W[88 + b1i]) : INFINITY;
      }
      // what follows the tensor-core sum of a head tile (the full head and the survivors of the sketch alike)
      auto head_fin = [&](int i, const Hdr& hd, float c0, float c1, float c2, float c3) {
#ifdef W12
        { float4 c = make_float4(c0, c1, c2, c3); tile_fix(c, hd, xs0, xs1, false, 0, 128, sm); c0 = c.x; c1 = c.y; c2 = c.z; c3 = c.w; }
#endif
        const int row = i * 16 + g8;
        if (A.gen) {
          if (c0 > bv0) { bv0 = c0; bi0 = row; }
          if (c2 > bv0) { bv0 = c2; bi0 = row + 8; }
          if (c1 > bv1) { bv1 = c1; bi1 = row; }
          if (c3 > bv1) { bv1 = c3; bi1 = row + 8; }
        } else {
          if (b0i < B) { lg[(size_t)b0i * VOCAB + row] = c0; lg[(size_t)b0i * VOCAB + row + 8] = c2; }
          if (b1i < B) { lg[(size_t)b1i * VOCAB + row] = c1; lg[(size_t)b1i * VOCAB + row + 8] = c3; }
        }
      };
      if (!sketch) {
        for (int i = gw; i < VOCAB / 16; i += NW) {
          const WP p = hp + (size_t)i * TBY(128);
          const Hdr hd = hdr_load(p + 128 * KSB);
          float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
          kmma_acc<128>(p, xn_b, c0, c1, c2, c3);
          head_fin(i, hd, c0, c1, c2, c3);
        }
      } else {
        // pass 2. One warp alone needs 35 us for a tile (it issues an instruction only every few cycles, and widening
        // the 12-bit weights is most of the work), so the whole block prepares a surviving tile: every warp reads and
        // widens 1/16 of it into shared memory, then the owner runs the 128 tensor-core steps in the usual order on
        // those fragments. Survivors are found per warp (bit tn of its mask), then listed per tn over the warps.
        uint32_t mk = 0;
        { int tn = 0; for (int i = gw; i < VOCAB / 16; i += NW, tn++) { const float2 ub = UT[(warp * 8 + tn) * 32 + lane]; mk |= (__any_sync(0xffffffffu, (ub.x >= T0) | (ub.y >= T1)) ? 1u : 0u) << tn; } }
        if (lane == 0) sm.W[96 + warp] = mk;
        __syncthreads();
        if (tid < 8) { uint32_t m = 0; for (int w = 0; w < 16; w++) m |= ((sm.W[96 + w] >> tid) & 1u) << w; sm.W[112 + tid] = m; }
        __syncthreads();
        uint4* const FR = (uint4*)sm.slot;   // [128 k-steps][32 lanes] (the digits and bounds of pass 1 are no longer needed)
        for (int tn = 0; tn < 8; tn++) {
          uint32_t m = sm.W[112 + tn];
          while (m) {
            const int w = __ffs(m) - 1; m &= m - 1;
            const int i = w * NB + j + tn * NW;
            const WP p = hp + (size_t)i * TBY(128);
            const long long tsk = M->prof ? clock64() : 0;
            const Hdr hd = hdr_load(p + 128 * KSB);
#ifdef W12
#pragma unroll
            for (int u = 0; u < 4; u++) {
              uint32_t r[8];
              w12_pair(p, warp * 4 + u, r);
              FR[(2 * (warp * 4 + u)) * 32 + lane] = make_uint4(r[0], r[1], r[2], r[3]);
              FR[(2 * (warp * 4 + u) + 1) * 32 + lane] = make_uint4(r[4], r[5], r[6], r[7]);
            }
#else
#pragma unroll
            for (int u = 0; u < 8; u++) FR[(warp * 8 + u) * 32 + lane] = ((const uint4*)p + lane)[(size_t)(warp * 8 + u) * 32];
#endif
            __syncthreads();
            if (warp == w) {
              float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll 4
              for (int k = 0; k < 128; k++) { const uint4 a = FR[k * 32 + lane]; const uint32_t b0 = xn_b[k * 8], b1 = xn_b[k * 8 + 4]; MMA(c0, c1, c2, c3, a, b0, b1); }
              head_fin(i, hd, c0, c1, c2, c3);
              if (M->prof && lane == 0) { atomicAdd(&C->cntR + 1, 1u); atomicAdd(&C->cntR + 2, (uint32_t)(clock64() - tsk)); }   // (debug: exact tiles, cycles)
            }
            __syncthreads();   // (the fragments are reused by the next survivor)
          }
        }
      }
      if (A.gen) {
        // candidates of this block: [8 batch columns][128 = 16 warps x 8 row groups]
        float* CV = (float*)sm.slot; int* CI = (int*)sm.slot + 1024;
        CV[b0i * 128 + warp * 8 + g8] = bv0; CI[b0i * 128 + warp * 8 + g8] = bi0;
        CV[b1i * 128 + warp * 8 + g8] = bv1; CI[b1i * 128 + warp * 8 + g8] = bi1;
        __syncthreads();
        if (warp < B) {
          float best = -INFINITY; int bi = 0x7fffffff;
#pragma unroll
          for (int q = 0; q < 4; q++) {
            float v = CV[warp * 128 + lane + 32 * q]; int ix = CI[warp * 128 + lane + 32 * q];
            if (v > best || (v == best && ix < bi)) { best = v; bi = ix; }
          }
          for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
          }
          if (lane == 0) { M->candv[j * MAXB + warp] = best; M->candi[j * MAXB + warp] = bi; FENCE(); }
        }
        SIGNAL(&C->cntH);
      }
      PROF(9);
  }
}

template <int OPBT, int EUT>
__global__ void __launch_bounds__(512, 1) mk_kernel(Launch A) {
  extern __shared__ __align__(16) unsigned char smraw[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, t4 = lane & 3;
  // The model's table of pointers lives in shared memory for the whole launch: a field read from global memory
  // would be one round trip through the L2 queue each time (0.3 us on a quiet card, more under a stream).
  for (int i = tid; i < (int)((sizeof(Model) + 7) / 8); i += 512) ((uint64_t*)smraw)[i] = ((const uint64_t*)A.M)[i];
  __syncthreads();
  const Model* M = (const Model*)smraw;
  Ctl* C = M->ctl;
  const int b0i = 2 * t4, b1i = 2 * t4 + 1;
  const int B = A.B, NB = gridDim.x, j = my_role(M), NW = 16 * NB;
  const bool act = g8 < B;
  // shared-memory layout: every offset is a compile-time constant, so that none of these pointers holds a register
  // (the kernel must stay below 128 registers for 512 threads per SM; a spill goes to local memory)
  enum : size_t {
    OFF_XN = (sizeof(Model) + 63) & ~(size_t)63,
    OFF_ZERO = OFF_XN + MAXB * HID * 2,
    OFF_SCR = OFF_ZERO + 8192,
    OFF_CST = OFF_SCR + QB * 32 * 16,
    OFF_R = OFF_CST + (4 + MAXU) * 32 * 16,
    OFF_MX = OFF_R + 512 * 4,
    OFF_W = OFF_MX + 32 * 4,
    OFF_EB = OFF_W + 160 * 4,
    OFF_BM = OFF_EB + MAXU * 8,
    OFF_BSLOT = OFF_BM + MAXU * 4,
    OFF_BW = OFF_BSLOT + MAXB * TOPK * 4,
    OFF_TP = OFF_BW + MAXB * TOPK * 4,
    OFF_PM = OFF_SCR + 8 * 256 * 4,       // attention arrays overlay the quantum slots (an attention block runs no quanta in stage B)
    OFF_QS = OFF_PM + 8 * 272 * 2,
    OFF_O = OFF_QS + 8 * 128 * 2,         // O [2][8][128], then SelfP [512], Kn [128], Vn [128], Wc [64]
    OFF_PART = OFF_O + (2048 + 512 + 128 + 128 + 64) * 4,
    OFF_PROF = OFF_TP + TB * 8 + 64,
    OFF_EX = OFF_PROF + 64 * 8,
    OFF_XV = OFF_EX + 16 * 64 * 4,
    OFF_GT = OFF_XV + 16 * 64 * 8 * 2,
    OFF_MB = (OFF_GT + (1 + MAXU) * 8 + 15) & ~(size_t)15,
  };
  const Smem sm = {(uint16_t*)(smraw + OFF_XN), (uint32_t*)(smraw + OFF_ZERO), (float4*)(smraw + OFF_SCR), (float4*)(smraw + OFF_CST),
                   (float*)(smraw + OFF_SCR), (uint16_t*)(smraw + OFF_PM), (uint16_t*)(smraw + OFF_QS), (float*)(smraw + OFF_O),
                   (float*)(smraw + OFF_R), (float*)(smraw + OFF_MX), (uint32_t*)(smraw + OFF_W), (WP*)(smraw + OFF_EB),
                   (int*)(smraw + OFF_BM), (int*)(smraw + OFF_BSLOT), (float*)(smraw + OFF_BW), (WP*)(smraw + OFF_TP),
                   (float*)(smraw + OFF_PART), (long long*)(smraw + OFF_PROF), (uint32_t*)(smraw + OFF_EX), (uint16_t*)(smraw + OFF_XV), (int2*)(smraw + OFF_GT), (uint64_t*)(smraw + OFF_MB)};
  if (tid < 64) sm.prof[tid] = 0;
  if (tid < 32) ((uint32_t*)sm.mb)[tid] = 0;   // (the pollers' words, one per Ctl field)
#if STAGE_A >= 2
  if (lane == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(sm.mb + warp)));
  __syncthreads();
  uint32_t stg_ph = 0;   // parity of this warp's mbarrier
#endif
  for (int i = tid; i < 2048; i += 512) sm.zero[i] = 0;
  // attention blocks: 32 (taking 64 at batch 8 was slower: 7827 against 7365 us; the gate|up stream loses more)
  const int natt = NATT;
  const bool att_block = j >= 1 && j <= natt;
  const int att_nblk = B == 1 ? ATTB1 : max(1, 8 / B);   // (ATTB1: blocks per attention task at batch 1, a divisor of 8)  // blocks per attention task
  // o_proj in stage B (OPB): 1 always, 2 only at batch > 1 (at batch 1 it was 12-45 us slower, e0264). Two
  // instantiations of the kernel: a run-time switch made both batch sizes slower (e0265: code layout).
  constexpr bool opb = OPBT != 0;
#ifndef LROUTE
#define LROUTE 0
#endif
  // batch 1 (the instantiation without o_proj in stage B): every block routes itself from the router rows
  constexpr bool lroute = LROUTE && !opb;
  const int gw = warp * NB + j;                                // this warp's index in the grid-wide round robin
  const uint32_t* xn_b = act ? (const uint32_t*)(sm.xn + g8 * HID) + t4 : sm.zero;  // B operand of the normed input
  const int xn_step = act ? 64 : 0;                            // u32 per quantum (8 k-steps x 16 values) along it
  // stage A work of this block. Whole tiles (tile id < 320: q|k|v row tile; 320 + r: router row tile r): blocks
  // 1..131 take tiles j-1 and 131+j-1. Tiles 262..327 are cut in two halves of 64 k-steps; block j takes half j
  // (4 k-steps per warp). Every warp of blocks 1..131 thus reads 20 k-steps; the leader only 4, then it routes.
  const int htile = FULL_TILES + (j >> 1), hhalf = j & 1;
  const bool hrouter = htile >= 320;
  long long tprof = clock64();
  long long twait = 0;
#ifdef TRACE
  unsigned long long trv[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif
  const int trace_step = (A.gen ? A.P - 1 + A.nsteps : A.nsteps) - 1;
#define A_TILE(LW_, t_) ((t_) < 320 ? (LW_).qkv + (size_t)(t_) * TBY(128) : (LW_).router + (size_t)((t_) - 320) * TBY(128))
  const uint16_t* const xs0 = b0i < B ? sm.xn + b0i * HID : nullptr;  // the normed input of this lane's two batch columns (exceptions)
  const uint16_t* const xs1 = b1i < B ? sm.xn + b1i * HID : nullptr;
  const int total_steps = A.gen ? A.P - 1 + A.nsteps : A.nsteps;  // gen: P - 1 prompt-only steps, then one new token per step
  for (int step = 0; step < total_steps; step++) {
    const bool want_head = !A.gen || step >= A.P - 1;
    const int pos = A.pos + step;
    const uint32_t S = A.s0 + step;
    if (j == 0) {
      if (A.gen && step >= A.P) { WAIT_GE(&C->cntH, NB * (A.h0 + step - (A.P - 1))); leader_argmax(M, A, step - A.P, NB); __syncthreads(); }
      else if (step > 0) { WAIT_GE(&C->cntC, NB * (A.g0 + step * NLAYER)); }  // every row of the residual is final before it is replaced
      for (int b = 0; b < B; b++) {
        int64_t tk = (!A.gen || step < A.P) ? A.tok_in[(size_t)b * A.tok_stride + step] : M->curtok[b];
        ((uint2*)(M->x + (size_t)b * HID))[tid] = ((const uint2*)(M->embed + (size_t)tk * HID))[tid];
        if (M->l0qkv) {
          // Layer 0 reads only the token (its input is the embedding and attention and MLP both read the normed
          // input): q|k|v before RoPE and the MLP output come from tables made when the engine loads.
          const uint2* tq = (const uint2*)(M->l0qkv + (size_t)tk * QKV_ROWS);
          const uint2 q0 = tq[tid], q1 = tq[512 + tid], q2 = tq[1024 + (tid & 255)], m0 = ((const uint2*)(M->l0mlp + (size_t)tk * HID))[tid];
          uint2* dq = (uint2*)(M->qkvo + (size_t)b * QKV_ROWS);
          dq[tid] = q0; dq[512 + tid] = q1;
          if (tid < 256) dq[1024 + tid] = q2;
          ((uint2*)(M->l0m + (size_t)b * HID))[tid] = m0;
        }
      }
      FENCE();
      __syncthreads();
      if (tid == 0) atomicExch(&C->tok_seq, S + 1);
    }
    WAIT_GE(&C->tok_seq, S + 1);
    // this step's gate|up shares, for every size of the union (the attention delay depends on the context length)
    if (tid <= MAXU) {
      const int ntc = (((pos + 16) >> 4) + NCH - 1) / NCH;            // K tiles per chunk
      const int dAN = GS_A0 + ((NCH / att_nblk) * (GS_A1 + GS_A2 * ntc) + (att_nblk > 1 ? GS_A3 * ntc : 0)) / 8;
      int ct, off;
      gu_share(j, B, tid == 0 ? M->L[0].gu_tiles : tid * M->L[1].gu_tiles, att_nblk, GS_DL, dAN, dAN + (att_nblk > 1 ? GS_AX : 0), GS_DI, opb ? (B == 1 ? GS_DO1 : GS_DO) : 0, ct, off);
      sm.gt[tid] = make_int2(ct, off);
    }
    __syncthreads();
    if (M->l0qkv && tid == 0) __threadfence();   // (flush this SM's L1: layer 0 has no stage A whose fence would do it)
    PROF(0);
    for (int l = 0; l < NLAYER; l++) {
      const LayerW& LW = M->L[l];
      const uint32_t G = A.g0 + step * NLAYER + l;
#if EARLY_U
      unsigned long long upre[4] = {0, 0, 0, 0};
#endif
      uint32_t preB = 0;
      const bool tab0 = LW.dense && M->l0qkv != nullptr;   // layer 0 from the token tables: no norm, no q|k|v, no MLP
      const uint2 nw = ((const uint2*)LW.normw)[tid];
      PROFW0();
      if (l > 0) WAIT_GE(&C->cntC, NB * G);
      PROFW(2);
#if L1FLUSH == 1
      __threadfence();
#elif L1FLUSH == 2
      if (tid == 0) __threadfence();
#elif L1FLUSH == 3
      if (lane == 0) __threadfence();
#endif
      PROF(1);
      TR(0);
      if (M->dbg_x && j == 0) for (int b = 0; b < B; b++) for (int i = tid; i < HID; i += 512) M->dbg_x[((size_t)l * MAXB + b) * HID + i] = M->x[(size_t)b * HID + i];
#if STAGE_A == 3 && defined(W12)
      if (!tab0 && l > 0) {
        unsigned char* st = (unsigned char*)sm.slot + STG_OFF + warp * STG_WARP;
        prologue_norm(M, sm, nw, B, [&]() { stage_bulk(st, A_TILE(LW, htile) + (size_t)(hhalf * 64 + warp * 4) * KSB, 1536, LW.qkv + (size_t)(max(j, 1) - 1) * TBY(128) + warp * QBY, j > 0 ? 3072 : 0, sm.mb + warp); });
      } else
#endif
      if (!tab0) prologue_norm(M, sm, nw, B);
      PROF(2);
      TR(1);
      // ---- A: q|k|v and router rows.
      // LEADER_FIRST 1: the leader reads its q|k|v half tile first and signals, then routes: attention (which waits
      // for every block's stage A) no longer waits for the routing chain, and the router rows take 5 us to arrive
      // anyway. 0: the leader routes first (as up to s0010).
#if LEADER_FIRST == 0
      if (j == 0 && !LW.dense && !lroute) {
        leader_route(M, sm, l, B, G + 1, tprof);
#if LEADER_ORDER == 0
        leader_details(M, sm, l, B, tprof);
#endif
        PROF(26);
      }
      if (j == 0) { __syncthreads(); if (tid == 0) atomicExch(&C->route_seq, G + 1); PROF(27); }
#endif
      if (!tab0) {
        const bool hdo = !(hrouter && LW.dense);
        // (the warps that will finish a tile fetch its header now: warp 0 the half tile, warps 1 and 2 the whole ones)
        const Hdr hd = hdr_load(warp == 0 ? A_TILE(LW, hdo ? htile : 0) + 128 * KSB : LW.qkv + (size_t)(warp == 1 ? max(j - 1, 0) : NB - 1 + max(j - 1, 0)) * TBY(128) + 128 * KSB);
#if STAGE_A && defined(W12)
        // (staged at the end of the previous layer's stage C: every layer after the first of a step)
        const bool stg = l > 0 && (STAGE_A != 4 || hrouter);
        WP const stA = stg ? (WP)sm.slot + STG_OFF + warp * STG_WARP : A_TILE(LW, htile) + (size_t)(hhalf * 64 + warp * 4) * KSB;
        WP const stB = (stg && STAGE_A != 4) ? stA + 1536 : LW.qkv + (size_t)(max(j, 1) - 1) * TBY(128) + warp * QBY;
#if STAGE_A == 1 || STAGE_A == 4
        if (stg) cpa_wait();
#else
        if (stg) stage_bulk_wait(sm.mb + warp, stg_ph);
#endif
#else
        WP const stA = A_TILE(LW, htile) + (size_t)(hhalf * 64 + warp * 4) * KSB;
        WP const stB = LW.qkv + (size_t)(max(j, 1) - 1) * TBY(128) + warp * QBY;
#endif
        if (hdo) sm.slot[warp * 32 + lane] = hrun(stA, xn_b + (hhalf * 64 + warp * 4) * (xn_step >> 3));
        if (hdo && hrouter) {
          // router rows first: the leader is polling for them. Value and stamp go out in one 8-byte store.
          __syncthreads();
          if (warp == 0) {
            float4 c = sm.slot[lane];
#pragma unroll
            for (int w = 1; w < 16; w++) c = f4add(c, sm.slot[w * 32 + lane]);
            tile_fix(c, hd, xs0, xs1, false, hhalf * 64, hhalf * 64 + 64, sm);
            uint2* o = M->rl + (size_t)hhalf * MAXB * 128 + (htile - 320) * 16 + g8;
            unsigned long long* o8 = (unsigned long long*)o;
            const unsigned long long st = (unsigned long long)(G + 1) << 32;
            if (b0i < B) { atomicExch(o8 + b0i * 128, st | __float_as_uint(c.x)); atomicExch(o8 + b0i * 128 + 8, st | __float_as_uint(c.z)); }
            if (b1i < B) { atomicExch(o8 + b1i * 128, st | __float_as_uint(c.y)); atomicExch(o8 + b1i * 128 + 8, st | __float_as_uint(c.w)); }
          }
          TR(11);
        }
        if (j > 0) {
          sm.slot[(16 + warp) * 32 + lane] = qrun(stB, xn_b + warp * xn_step);
          sm.slot[(32 + warp) * 32 + lane] = qrun(LW.qkv + (size_t)(NB - 1 + j - 1) * TBY(128) + warp * QBY, xn_b + warp * xn_step);
#if EARLY_U
          // (the union's load goes out now and is looked at after stage A: often it is already published)
          if (EUT && tid == 0 && !LW.dense) {   // (EUT: the batch-1 instantiation; at batch 8 +64..+215 us, e0361/e0362)
#pragma unroll
            for (int q = 0; q < 4; q++) upre[q] = *(const volatile unsigned long long*)&C->um[q];
          }
#endif
        }
        __syncthreads();
        if (warp == 0 && hdo && !hrouter) {
          float4 c = sm.slot[lane];
#pragma unroll
          for (int w = 1; w < 16; w++) c = f4add(c, sm.slot[w * 32 + lane]);
          tile_fix(c, hd, xs0, xs1, false, hhalf * 64, hhalf * 64 + 64, sm);
          float* o = M->qkvp + (size_t)hhalf * MAXB * QKV_ROWS + htile * 16 + g8;
#ifndef ATOMIC_Q
          if (j != 0 || LEADER_FIRST == 0) {
            if (b0i < B) { o[b0i * QKV_ROWS] = c.x; o[b0i * QKV_ROWS + 8] = c.z; }
            if (b1i < B) { o[b1i * QKV_ROWS] = c.y; o[b1i * QKV_ROWS + 8] = c.w; }
            FENCE();
          } else
#endif
          {
            // (exchanges, no fence: under the stream a fence costs 7 us; the leader's fence comes with the routing)
            uint32_t xa = 0;
            if (b0i < B) { xa ^= xstoref(o + b0i * QKV_ROWS, c.x); xa ^= xstoref(o + b0i * QKV_ROWS + 8, c.z); }
            if (b1i < B) { xa ^= xstoref(o + b1i * QKV_ROWS, c.y); xa ^= xstoref(o + b1i * QKV_ROWS + 8, c.w); }
            XDONE(xa);
          }
        }
        if (j > 0 && (warp == 1 || warp == 2)) {
          float4 c = sm.slot[(warp * 16) * 32 + lane];
#pragma unroll
          for (int kq = 1; kq < 16; kq++) c = f4add(c, sm.slot[(warp * 16 + kq) * 32 + lane]);
          tile_fix(c, hd, xs0, xs1, false, 0, 128, sm);
          uint16_t* o = M->qkvo + (warp == 1 ? j - 1 : NB - 1 + j - 1) * 16 + g8;
          uint32_t xa = 0;
#ifndef ATOMIC_Q
          if (b0i < B) { o[b0i * QKV_ROWS] = f2bf(c.x); o[b0i * QKV_ROWS + 8] = f2bf(c.z); }
          if (b1i < B) { o[b1i * QKV_ROWS] = f2bf(c.y); o[b1i * QKV_ROWS + 8] = f2bf(c.w); }
          __threadfence();
#else
          xa ^= xstore_rowpair(o + b0i * QKV_ROWS, f2bf(c.x), b0i < B);
          xa ^= xstore_rowpair(o + b0i * QKV_ROWS + 8, f2bf(c.z), b0i < B);
          xa ^= xstore_rowpair(o + b1i * QKV_ROWS, f2bf(c.y), b1i < B);
          xa ^= xstore_rowpair(o + b1i * QKV_ROWS + 8, f2bf(c.w), b1i < B);
#endif
          XDONE(xa);
        }
      }
      SIGNAL(&C->cntA);
#if LEADER_FIRST == 1
      if (j == 0 && !LW.dense && !lroute) {
        leader_route(M, sm, l, B, G + 1, tprof);
#if LEADER_ORDER == 0
        leader_details(M, sm, l, B, tprof);
#endif
        PROF(26);
      }
      if (j == 0) { __syncthreads(); if (tid == 0) atomicExch(&C->route_seq, G + 1); PROF(27); }
#endif
#if LEADER_ORDER == 1
      if (j == 0 && !LW.dense && !lroute) { leader_details(M, sm, l, B, tprof); __syncthreads(); }
#endif
      if ((PF & 4) && !att_block && j < 128) pf_range(LW.o + (size_t)j * TBY(256), (int)TBY(256));
      PROF(3);
      TR(2);
      if (M->prof && j == PROFBLK && tid == 0) sm.prof[22] += (int)(vload(&C->cntA) - NB * G);
      // ---- B: attention on blocks 1..4B, gate|up of the chosen experts on the others
      if (att_block) {
        WAIT_GE(&C->cntA, NB * (G + 1));
#ifndef ABL_NOATT
        if ((j - 1) / att_nblk < 4 * B) attention_task(M, sm, l, (j - 1) / att_nblk, (j - 1) % att_nblk, att_nblk, pos, G, tab0, tprof, opb);
#endif
        __syncthreads();
      }
      if (LW.dense) {
        WAIT_GE(&C->route_seq, G + 1);
        PROF(4);
        if (tid < MAXU) { sm.eb[tid] = LW.expert; sm.bm[tid] = 0xff; }
        if (tid == 2 * MAXU) { const int2 g = sm.gt[0]; sm.W[0] = 1; sm.W[5] = tab0 ? 0 : g.x; sm.W[6] = g.y; }
        __syncthreads();
      } else {
        // the union of the chosen experts, as soon as the leader has it (four stamped words, one poll)
        PROFW0();
        if (lroute) {
          // batch 1: the router rows straight from the router blocks (two stamped partial sums per logit), top 8 here
          if (warp == 0) {
            const uint2* r0 = M->rl + lane;
            unsigned long long p[8];
            for (;;) {
#pragma unroll
              for (int q = 0; q < 4; q++) { p[q] = *(const volatile unsigned long long*)(r0 + 32 * q); p[4 + q] = *(const volatile unsigned long long*)(r0 + MAXB * 128 + 32 * q); }
              bool ok = true;
#pragma unroll
              for (int q = 0; q < 8; q++) ok = ok && (uint32_t)(p[q] >> 32) == G + 1;
              if (__all_sync(0xffffffffu, ok)) break;
            }
            float v[4];
            uint32_t mem = 0;
#pragma unroll
            for (int q = 0; q < 4; q++) v[q] = rbf(__uint_as_float((uint32_t)p[q]) + __uint_as_float((uint32_t)p[4 + q]));
            if (!M->force) {
              int key[4];
#pragma unroll
              for (int q = 0; q < 4; q++) { const int u = (int)__float_as_uint(v[q]); const int k = u ^ ((u >> 31) & 0x7fffffff); key[q] = ((k >> 16) << 7) | (127 - (lane + 32 * q)); }
#pragma unroll 1
              for (int r = 0; r < TOPK; r++) {
                const int best = __reduce_max_sync(0xffffffffu, max(max(key[0], key[1]), max(key[2], key[3])));
#pragma unroll
                for (int q = 0; q < 4; q++) { const bool hit = key[q] == best; mem |= (hit ? 1u : 0u) << q; key[q] = hit ? (int)0x80000000 : key[q]; }
              }
            } else {
              for (int r = 0; r < TOPK; r++) {
                const int e = M->force[(size_t)l * MAXB * TOPK + r];
#pragma unroll
                for (int q = 0; q < 4; q++) mem |= (e == lane + 32 * q ? 1u : 0u) << q;
              }
            }
            uint32_t mw[4];
#pragma unroll
            for (int q = 0; q < 4; q++) mw[q] = __ballot_sync(0xffffffffu, (mem >> q) & 1u);
            if (lane < 4) sm.W[1 + lane] = mw[lane];
            // slots are the experts in index order, so a chosen expert's slot is its rank; its weight is sigmoid(logit)
            int rank = 0;
#pragma unroll
            for (int q = 0; q < 4; q++) {
              if ((mem >> q) & 1u) { const int rk = rank + __popc(mw[q] & ((1u << lane) - 1)); sm.bw[rk] = rbf(1.0f / (1.0f + expf(-v[q]))); sm.bslot[rk] = rk; }
              rank += __popc(mw[q]);
            }
          }
        } else if (tid == 0) {
          unsigned long long u[4];
#if EARLY_U
#pragma unroll
          for (int q = 0; q < 4; q++) u[q] = upre[q];
          if (!EUT || !((uint32_t)(u[0] >> 32) == G + 1 && (uint32_t)(u[1] >> 32) == G + 1 && (uint32_t)(u[2] >> 32) == G + 1 && (uint32_t)(u[3] >> 32) == G + 1))
#endif
          for (;;) {
#pragma unroll
            for (int q = 0; q < 4; q++) u[q] = *(const volatile unsigned long long*)&C->um[q];
            if ((uint32_t)(u[0] >> 32) == G + 1 && (uint32_t)(u[1] >> 32) == G + 1 && (uint32_t)(u[2] >> 32) == G + 1 && (uint32_t)(u[3] >> 32) == G + 1) break;
          }
#pragma unroll
          for (int q = 0; q < 4; q++) sm.W[1 + q] = (uint32_t)u[q];
        }
        __syncthreads();
        PROFW(0);
        PROF(4);
        if (tid < 128) {
          const int e = tid, wi = e >> 5;
          const uint32_t w = sm.W[1 + wi];
          if ((w >> (e & 31)) & 1u) {
            int slot = __popc(w & ((1u << (e & 31)) - 1));
            for (int q = 0; q < wi; q++) slot += __popc(sm.W[1 + q]);
            sm.eb[slot] = LW.expert + (size_t)e * LW.exp_stride;
          }
        } else if (tid == 2 * MAXU) {
          const int Uv = __popc(sm.W[1]) + __popc(sm.W[2]) + __popc(sm.W[3]) + __popc(sm.W[4]);
          const int2 g = sm.gt[Uv];
          sm.W[0] = Uv; sm.W[5] = g.x; sm.W[6] = g.y;
        }
        __syncthreads();
      }
      const int U = (int)sm.W[0];
      PROF(10);
      TR(3);
      {
        const int gu = LW.gu_tiles;
        const int ct = (int)sm.W[5];
        const int goff = (int)sm.W[6];
#define GU_TILE(t_) (goff + (t_))
        for (int t0 = 0; t0 < ct; t0 += TB) {
          const int nt = min(TB, ct - t0);
          if (tid < nt) { const int i = GU_TILE(t0 + tid); sm.tp[tid] = sm.eb[i / gu] + (size_t)(i % gu) * TBY(128); }
          __syncthreads();
          // quantum q of the batch: tile q >> 4, k-quantum q & 15; this warp takes k-quantum `warp` of every tile
          const Hdr hd = hdr_load(sm.tp[warp < nt ? warp : 0] + 128 * KSB);  // warp w finishes tile w of the batch
          for (int k = 0; k < nt; k++) sm.slot[(k * 16 + warp) * 32 + lane] = qrun(sm.tp[k] + warp * QBY, xn_b + warp * xn_step);
          PROF(11);
          __syncthreads();
          PROF(12);
          if (warp < nt) {
            // this warp finishes tile t0 + warp: sum its 16 quanta in order, SwiGLU, store h
            float4 c = sm.slot[(warp * 16) * 32 + lane];
#pragma unroll
            for (int kq = 1; kq < 16; kq++) c = f4add(c, sm.slot[(warp * 16 + kq) * 32 + lane]);
            const int i = GU_TILE(t0 + warp), u = i / gu, m = i % gu;
            uint16_t* h = M->h + (size_t)u * MAXB * HMAX + m * 8 + g8;
            tile_fix(c, hd, xs0, xs1, false, 0, 128, sm);
            const float g0 = rbf(c.x), g1 = rbf(c.y);
            // h goes out times 2^(XS + the neuron's down-column shift) (track B; 1 on track A)
#ifdef W12
            const float hs = __uint_as_float((uint32_t)(127 + XS + hd.cd) << 23);
#else
            const float hs = 1.0f;
#endif
            const uint16_t h0 = f2bf(rbf(rbf(g0 / (1.0f + expf(-g0))) * rbf(c.z)) * hs);
            uint16_t h1 = 0;
            if (B > 1) h1 = f2bf(rbf(rbf(g1 / (1.0f + expf(-g1))) * rbf(c.w)) * hs);   // (the odd batch columns exist from batch 2 on)
#ifndef ATOMIC_H
            if (b0i < B) h[(size_t)b0i * HMAX] = h0;
            if (b1i < B) h[(size_t)b1i * HMAX] = h1;
            __threadfence();
#else
            uint32_t xa = xstore_rowpair(h + (size_t)b0i * HMAX, h0, b0i < B);
            xa ^= xstore_rowpair(h + (size_t)b1i * HMAX, h1, b1i < B);
            XDONE(xa);
#endif
          }
          __syncthreads();
        }
      }
      PROF(13);
      TR(4);
      SIGNAL(&C->cntB);
      if (opb && j < 128) {
        // o_proj of this block's row tile: it needs only the attention output (per-task stamps), not gate|up
#ifndef ABL_NOATT
        if (warp == 0) {
          for (;;) {
            const bool ok = lane >= 4 * B || vload(&C->adone[lane]) == G + 1;
            if (__all_sync(0xffffffffu, ok)) break;
          }
        }
        __syncthreads();
#endif
        const WP op = LW.o + (size_t)j * TBY(256);
        const uint32_t* at_b = act ? (const uint32_t*)(M->attn + g8 * 4096) + t4 : (ZSM ? (const uint32_t*)sm.zero : M->zeros);
#ifdef W12
        const Hdr hd = hdr_load(op + 256 * KSB);
        if (warp == 0 && B == 1) xv_gather(hd, M->attn, 4096, B, sm);
#endif
        sm.slot[warp * 32 + lane] = qrun(op + warp * QBY, at_b + warp * xn_step);
        sm.slot[(16 + warp) * 32 + lane] = qrun(op + (16 + warp) * QBY, at_b + (16 + warp) * xn_step);
        if (EARLY_B && EUT && tid == 0) preB = vload(&C->cntB);   // (looked at after o_proj: the barrier is usually complete by then)
        __syncthreads();
        if (warp == 0) {
          float4 c = sm.slot[lane];
          for (int k = 1; k < 32; k++) c = f4add(c, sm.slot[k * 32 + lane]);
#ifdef W12
          tile_fix(c, hd, b0i < B ? M->attn + b0i * 4096 : nullptr, b1i < B ? M->attn + b1i * 4096 : nullptr, B == 1, 0, 256, sm);
#endif
          sm.Cst[lane] = c;
        }
        __syncthreads();
      }
      if ((PF & 2) && j < 128 && !tab0) {
        const size_t dn_off = (size_t)LW.gu_tiles * TBY(128) + (size_t)j * TBY(LW.down_K);
        for (int u = 0; u < U; u++) pf_range(sm.eb[u] + dn_off, (int)TBY(LW.down_K));
        if (!(PF & 4) || att_block) pf_range(LW.o + (size_t)j * TBY(256), (int)TBY(256));
      }
      PROF(5);
      if (M->prof && j == PROFBLK && tid == 0) sm.prof[20] += (int)(vload(&C->cntB) - NB * G);
      // ---- C: one residual row tile per block: o_proj (32 quanta) + down of every chosen expert, combined here
      TR(6);
      PROFW0();
      if (EARLY_B && EUT && opb && j < 128) {
        if (tid == 0 && (int)(preB - (uint32_t)(NB * (G + 1))) < 0) { while ((int)(vload(&C->cntB) - (uint32_t)(NB * (G + 1))) < 0) {} }
        __syncthreads();
      } else
      WAIT_GE(&C->cntB, NB * (G + 1));
      TR(5);
      PROFW(1);
#ifndef LOCAL_DETAILS
#define LOCAL_DETAILS 1
#endif
      // the rest of the routing (the leader finished it before its own stage B signal, which this wait covers).
      // LOCAL_DETAILS, batch 1: the sequence's experts are the union, so slot masks and slots per rank are known here;
      // only the weights are loaded, and they are first needed in the residual add (the load is not waited for now).
      float bw_late = 0.f;
      if (!LW.dense) {
        const Dyn& D = M->dyn[l];
        if (lroute) {
          if (tid < MAXU) sm.bm[tid] = 1;           // (slots and weights were set when the block routed)
        } else if (LOCAL_DETAILS && B == 1) {
          if (tid < MAXU) sm.bm[tid] = 1;
          else if (tid < MAXU + TOPK) { sm.bslot[tid - MAXU] = tid - MAXU; bw_late = *(const volatile float*)&D.bw[0][tid - MAXU]; }
        } else {
          if (tid < MAXU) sm.bm[tid] = *(const volatile int*)&D.bmask[tid];
          else if (tid < 2 * MAXU) { sm.bslot[tid - MAXU] = *(const volatile int*)&D.bslot[0][tid - MAXU]; sm.bw[tid - MAXU] = *(const volatile float*)&D.bw[0][tid - MAXU]; }
        }
        __syncthreads();
      }
      PROF(6);
      TR(7);
      if (j < 128) {
#ifndef XO_EARLY
#define XO_EARLY 1   // the residual's old rows are loaded at the start of stage C (this block stored them a layer ago)
#endif
        uint16_t xo_pre = 0;
        if (XO_EARLY && (!opb || (XO_B1 && B == 1)) && warp < 4) {   // (batch 1 instantiation: at batch 8 it was 54 us slower, e0298)
          const int b = 2 * t4 + (warp & 1), row = j * 16 + g8 + (warp >> 1) * 8;
          if (b < B) xo_pre = M->x[(size_t)b * HID + row];
        }
        const int nun = tab0 ? 0 : U;                     // down units
        const WP op = LW.o + (size_t)j * TBY(256);
        const size_t dn_off = (size_t)LW.gu_tiles * TBY(128) + (size_t)j * TBY(LW.down_K);
        const uint32_t* at_b = act ? (const uint32_t*)(M->attn + g8 * 4096) + t4 : (ZSM ? (const uint32_t*)sm.zero : M->zeros);
        const int dq = LW.dense ? 24 : 6;                 // quanta per down unit
        // batches of whole units, at most QB quanta: batch 0 = o_proj + the first units
        int u0 = 0;
        for (int bt = 0; u0 < nun || bt == 0; bt++) {
          const int qo = (bt == 0 && !opb) ? 32 : 0;      // o_proj quanta in this batch
          const int ho = qo ? 1 : 0;                      // (the o_proj total is unit -1 of the batch)
          const int un = min(nun - u0, (QB - qo) / dq), nq = qo + un * dq;
          // (warp w finishes units w, w + 16, ...: the header of the first one is fetched with the quanta)
          const bool e_isO = ho && warp == 0;
          const int e_u = u0 + warp - ho;
          const bool e_ok = e_isO || (warp < ho + un);
          const Hdr hd = hdr_load(e_isO || !e_ok ? op + 256 * KSB : sm.eb[e_u] + dn_off + (size_t)LW.down_K * KSB);
          // (batch 1 only: with more columns the gather is 16 loads per lane and unit, and reading x where it is needed is cheaper)
          const bool use_xv = B == 1;
#ifndef XV_POS
#define XV_POS 1   // where the escape x values are gathered: 0 before the quanta, 1 after the first, 2 after all
#endif
#define XV_DO() { if (use_xv) { if (e_isO || !e_ok) xv_gather(hd, M->attn, 4096, B, sm); else xv_gather(hd, M->h + (size_t)e_u * MAXB * HMAX, HMAX, B, sm); } }
          if (XV_POS == 0) XV_DO();
          for (int q = warp; q < nq; q += 16) {
            WP p1; const uint32_t* x1;
            if (q < qo) { p1 = op + q * QBY; x1 = at_b + q * xn_step; }
            else {
              const int qq = q - qo;
              const int u = u0 + (LW.dense ? 0 : qq / 6), kq = LW.dense ? qq : qq % 6;
              p1 = sm.eb[u] + dn_off + kq * QBY;
              x1 = (act && ((sm.bm[u] >> g8) & 1)) ? (const uint32_t*)(M->h + ((size_t)u * MAXB + g8) * HMAX) + t4 + kq * 64 : (ZSM ? (const uint32_t*)sm.zero : M->zeros);
            }
            sm.slot[q * 32 + lane] = qrun(p1, x1);
            if (XV_POS == 1 && q == warp) XV_DO();
          }
          if (XV_POS == 1 && warp >= nq) XV_DO();
          if (XV_POS == 2) XV_DO();
          PROF(14);
          __syncthreads();
          PROF(15);
          // unit totals, in quantum order: Cst[0] holds o_proj, Cst[4 + u] the down of slot u
          for (int w = warp; w < ho + un; w += 16) {
            const bool isO = ho && w == 0;
            const int uu = w - ho;
            const int q0 = isO ? 0 : qo + uu * dq, cnt = isO ? 32 : dq;
            float4 c = sm.slot[q0 * 32 + lane];
            for (int k = 1; k < cnt; k++) c = f4add(c, sm.slot[(q0 + k) * 32 + lane]);
#ifdef W12
            if (isO) tile_fix(c, hd, b0i < B ? M->attn + b0i * 4096 : nullptr, b1i < B ? M->attn + b1i * 4096 : nullptr, use_xv, 0, 256, sm);
            else {
              const int u = u0 + uu;
              const uint16_t* hx0 = (b0i < B && ((sm.bm[u] >> b0i) & 1)) ? M->h + ((size_t)u * MAXB + b0i) * HMAX : nullptr;
              const uint16_t* hx1 = (b1i < B && ((sm.bm[u] >> b1i) & 1)) ? M->h + ((size_t)u * MAXB + b1i) * HMAX : nullptr;
              if (w == warp) tile_fix(c, hd, hx0, hx1, use_xv, 0, LW.down_K, sm);
              else { const Hdr hl = hdr_load(sm.eb[u] + dn_off + (size_t)LW.down_K * KSB); tile_fix(c, hl, hx0, hx1, false, 0, LW.down_K, sm); }
            }
#endif
            sm.Cst[(isO ? 0 : 4 + u0 + uu) * 32 + lane] = c;
          }
          __syncthreads();
          u0 += un;
        }
        if (!lroute && LOCAL_DETAILS && B == 1 && !LW.dense) {
          if (tid >= MAXU && tid < MAXU + TOPK) sm.bw[tid - MAXU] = bw_late;
          __syncthreads();
        }
        TR(8);
        // the residual add: x = (x + attn) + mlp, every sum rounded to bf16 as the model's dtype does. A lane holds 4
        // elements of the row tile (2 rows x 2 batch columns); warp q takes element q, so the four go in parallel.
        if (warp < 4) {
          const int q = warp, b = 2 * t4 + (q & 1), row = j * 16 + g8 + (q >> 1) * 8;
#ifndef ATOMIC_C
          if (b < B) {
            const uint16_t xo = (XO_EARLY && (!opb || (XO_B1 && B == 1))) ? xo_pre : M->x[(size_t)b * HID + row];
            const float* cs = (const float*)(sm.Cst + lane) + q;  // unit un at cs[un * 128]
            const float av = rbf(cs[0]);
            float mlp;
            if (LW.dense) mlp = tab0 ? bf2f(M->l0m[(size_t)b * HID + row]) : rbf(cs[4 * 128]);
            else {
              mlp = 0;
#pragma unroll
              for (int jj = 0; jj < TOPK; jj++) mlp = rbf(mlp + rbf(rbf(cs[(4 + sm.bslot[b * TOPK + jj]) * 128]) * sm.bw[b * TOPK + jj]));
            }
            M->x[(size_t)b * HID + row] = f2bf(rbf(bf2f(xo) + av) + mlp);
            __threadfence();
          }
#else
          // (no fence: the new value goes out as an exchange of the row pair's 32-bit word; every lane computes, the
          // lanes of batch columns not in use store nothing)
          const int bb = b < B ? b : 0;
          const uint16_t xo = *(const volatile uint16_t*)&M->x[(size_t)bb * HID + row];
          const float* cs = (const float*)(sm.Cst + lane) + q;
          const float av = rbf(cs[0]);
          float mlp;
          if (LW.dense) mlp = tab0 ? bf2f(*(const volatile uint16_t*)&M->l0m[(size_t)bb * HID + row]) : rbf(cs[4 * 128]);
          else {
            mlp = 0;
#pragma unroll
            for (int jj = 0; jj < TOPK; jj++) mlp = rbf(mlp + rbf(rbf(cs[(4 + sm.bslot[bb * TOPK + jj]) * 128]) * sm.bw[bb * TOPK + jj]));
          }
          const uint32_t xa = xstore_rowpair(M->x + (size_t)bb * HID + row, f2bf(rbf(bf2f(xo) + av) + mlp), b < B);
          XDONE(xa);
#endif
        }
      }
      TR(9);
      SIGNAL(&C->cntC);
      TR(10);
#if STAGE_A == 4 && defined(W12)
      // the router blocks' half tile of the next layer (it is the first link of the routing chain; 393 KB in all)
      if (l + 1 < NLAYER && hrouter) {
        stage_pairs((unsigned char*)sm.slot + STG_OFF + warp * STG_WARP, A_TILE(M->L[l + 1], htile) + (size_t)(hhalf * 64 + warp * 4) * KSB, 2);
        asm volatile("cp.async.commit_group;");
      }
#endif
#if (STAGE_A == 1 || STAGE_A == 2) && defined(W12)
      if (l + 1 < NLAYER) {
        const LayerW& LN = M->L[l + 1];
        unsigned char* st = (unsigned char*)sm.slot + STG_OFF + warp * STG_WARP;
#if STAGE_A == 1
        stage_pairs(st, A_TILE(LN, htile) + (size_t)(hhalf * 64 + warp * 4) * KSB, 2);
        if (j > 0) stage_pairs(st + 1536, LN.qkv + (size_t)(j - 1) * TBY(128) + warp * QBY, 4);
        asm volatile("cp.async.commit_group;");
#else
        stage_bulk(st, A_TILE(LN, htile) + (size_t)(hhalf * 64 + warp * 4) * KSB, 1536, LN.qkv + (size_t)(max(j, 1) - 1) * TBY(128) + warp * QBY, j > 0 ? 3072 : 0, sm.mb + warp);
#endif
      }
#endif
      if ((PF & 8) && att_block && l + 1 < NLAYER && (j - 1) / att_nblk < 4 * B) {
        // the K and V position tiles this block will read in the next layer's attention (tile w of a chunk: warp w)
        const int task = (j - 1) / att_nblk, nk = (pos + 16) >> 4, maxK = M->maxK;
        const size_t kvo = (((size_t)(l + 1) * MAXB + (task >> 2)) * 4 + (task & 3)) * (size_t)maxK * 256;
        const unsigned char* kc = (const unsigned char*)(M->kc + kvo);
        const unsigned char* vc = (const unsigned char*)(M->vc + kvo);
        for (int c = (j - 1) % att_nblk; c < NCH; c += att_nblk) {
          const int tile = c + NCH * warp;
          if (tile < nk) {
            pf_range(kc + (size_t)tile * 4096, 4096);
            for (int o = (lane & 3) * PFS; o < 512; o += 4 * PFS) PFL2(vc + ((size_t)(lane >> 2) * maxK + tile) * 512 + o);
          }
        }
      }
      if (PF & 1) {
        if (l + 1 < NLAYER) {
          const LayerW& LN = M->L[l + 1];
          pf_range(A_TILE(LN, htile) + (size_t)(hhalf * 64 + warp * 4) * KSB, 4 * KSB);
          if (j > 0) {
            pf_range(LN.qkv + (size_t)(j - 1) * TBY(128) + warp * QBY, QBY);
            pf_range(LN.qkv + (size_t)(NB - 1 + j - 1) * TBY(128) + warp * QBY, QBY);
          }
        } else if (want_head) {
          if (A.gen && M->sk) pf_range(M->sk + (size_t)gw * SKT, SKT); else pf_range(M->head + (size_t)gw * TBY(128), (int)TBY(128));
        }
      }
      PROF(7);
      TRFLUSH();
      if (M->prof && j == PROFBLK && tid == 0) sm.prof[21] += (int)(vload(&C->cntC) - NB * G);
    }
    // ---- head: final norm, then the vocabulary
    if (want_head) head_stage(M, sm, A, step, tprof);
  }
  if (A.gen && j == 0) { WAIT_GE(&C->cntH, NB * (A.h0 + A.nsteps)); leader_argmax(M, A, A.nsteps - 1, NB); __syncthreads(); }
  if (M->prof && (j == PROFBLK || j == 1) && tid < 64 && (j == PROFBLK) == (tid < 32)) M->prof[tid] += sm.prof[tid];
  if (M->prof && tid < 3) M->prof[64 + 4 * j + tid] += sm.prof[56 + tid];
}

// ---------------------------------------------------------------------------------------------------------
// host side
static Model* g_model_dev = nullptr;
static uint32_t g_g0 = 0, g_s0 = 0, g_h0 = 0;
static int g_smem = 0;
static int g_blocks = 132;

extern "C" {

int mk_sizeof_model() { return (int)sizeof(Model); }
int mk_sizeof_ctl() { return (int)sizeof(Ctl); }
int mk_sizeof_dyn() { return (int)sizeof(Dyn); }
int mk_sizeof_layer() { return (int)sizeof(LayerW); }
int mk_is_w12() {
#ifdef W12
  return 1;
#else
  return 0;
#endif
}

// model_host: a filled Model struct (device pointers inside); model_dev: device memory to hold it
int mk_init(const void* model_host, void* model_dev, int max_ctx) {
  cudaError_t e = cudaMemcpy(model_dev, model_host, sizeof(Model), cudaMemcpyHostToDevice);
  if (e != cudaSuccess) { fprintf(stderr, "mk_init memcpy: %s\n", cudaGetErrorString(e)); return 1; }
  g_model_dev = (Model*)model_dev;
  if (max_ctx > 2048) { fprintf(stderr, "mk_init: contexts above 2048 need a larger attention scratch\n"); return 3; }
  if (max_ctx > CTX) return 3;
  g_smem = (int)((sizeof(Model) + 63) & ~(size_t)63) + MAXB * HID * 2 + 8192 + QB * 32 * 16 + (4 + MAXU) * 32 * 16 + 512 * 4 + 32 * 4 + 160 * 4 + MAXU * 12 + MAXB * TOPK * 8 + TB * 8 + 64 + 64 * 8 + 16 * 64 * 4 + 16 * 64 * 8 * 2 + (1 + MAXU) * 8 + 16 + 16 * 8 + 16;
  // (more than half an SM's shared memory: one block per SM)
  e = cudaFuncSetAttribute(mk_kernel<0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(mk_kernel<1, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(mk_kernel<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem);
  if (e != cudaSuccess) { fprintf(stderr, "mk_init smem %d: %s\n", g_smem, cudaGetErrorString(e)); return 2; }
  int dev = 0; cudaGetDevice(&dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  g_blocks = prop.multiProcessorCount;
  if (g_blocks != 132) { fprintf(stderr, "mk_init: this build is laid out for the 132 SMs of an H100, found %d\n", g_blocks); return 4; }
  g_g0 = g_s0 = g_h0 = 0;
  return 0;
}

static cudaStream_t g_side = nullptr;  // a second stream: reads the progress of a running launch

static int launch(const int64_t* tok_in, int tok_stride, float* logits_out, int64_t* tok_out, int B, int pos, int nsteps, int gen, int P, int kmajor) {
  Launch A;
  A.M = g_model_dev; A.tok_in = tok_in; A.logits_out = logits_out; A.tok_out = tok_out;
  A.B = B; A.pos = pos; A.nsteps = nsteps; A.gen = gen; A.tok_stride = tok_stride; A.P = P; A.kmajor = kmajor;
  A.g0 = g_g0; A.s0 = g_s0; A.h0 = g_h0;
  // (instantiations: o_proj in stage B or not; the early union load at batch 1)
  if (OPB == 1 || (OPB == 2 && B > 1)) { if (EARLY_U && B == 1) mk_kernel<1, 1><<<g_blocks, 512, g_smem, 0>>>(A); else mk_kernel<1, 0><<<g_blocks, 512, g_smem, 0>>>(A); }
  else mk_kernel<0, 0><<<g_blocks, 512, g_smem, 0>>>(A);
  const int total = gen ? P - 1 + nsteps : nsteps;
  g_g0 += (uint32_t)total * NLAYER; g_s0 += total; if (gen) g_h0 += nsteps;
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "mk launch: %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}

int mk_step(const int64_t* tok_in, int tok_stride, float* logits_out, int B, int pos) { return launch(tok_in, tok_stride, logits_out, nullptr, B, pos, 1, 0, 1, 0); }
int mk_generate(const int64_t* tok_in, int tok_stride, int64_t* tok_out, int B, int pos, int nsteps) { return launch(tok_in, tok_stride, nullptr, tok_out, B, pos, nsteps, 1, 1, 0); }

// Serving: one launch decodes a whole cohort: prompts [B][P] (row stride tok_stride), then nsteps greedy tokens per
// sequence into tok_out[k][B]. Returns at once; mk_serve_poll() says how many tokens exist, mk_serve_fetch() copies them.
int mk_serve_start(const int64_t* prompt, int tok_stride, int64_t* tok_out, int B, int P, int nsteps) {
  if (!g_side) cudaStreamCreateWithFlags(&g_side, cudaStreamNonBlocking);
  Ctl* ctl = nullptr;
  cudaMemcpy(&ctl, (char*)g_model_dev + offsetof(Model, ctl), sizeof(ctl), cudaMemcpyDeviceToHost);  // (idle: nothing is running)
  cudaMemset(&ctl->prog, 0, 4);
  return launch(prompt, tok_stride, nullptr, tok_out, B, 0, nsteps, 1, P, 1);
}
static Ctl* g_ctl_cache = nullptr;
int mk_serve_poll() {
  if (!g_ctl_cache) cudaMemcpyAsync(&g_ctl_cache, (char*)g_model_dev + offsetof(Model, ctl), sizeof(g_ctl_cache), cudaMemcpyDeviceToHost, g_side), cudaStreamSynchronize(g_side);
  uint32_t prog = 0;
  cudaMemcpyAsync(&prog, &g_ctl_cache->prog, 4, cudaMemcpyDeviceToHost, g_side);
  cudaStreamSynchronize(g_side);
  return (int)prog;
}
// Calibration: every block streams the same bytes (2 loads in flight per thread, like the engine) and reports its
// %smid and its duration in cycles. out: [2][132] (smid, cycles).
__global__ void __launch_bounds__(512, 1) mk_calib_kernel(const uint4* W, int tpw, long long* out) {
  extern __shared__ unsigned char big[];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gw = blockIdx.x * 16 + warp;
  if (threadIdx.x == 0) big[0] = 1;
  __syncthreads();
  const long long t0 = clock64();
  uint32_t acc = 0;
  for (int t = 0; t < tpw; t++) {
    const uint4* p = W + ((size_t)t * 2112 + gw) * 4096 + lane;
    for (int k = 0; k < 128; k += 2) { const uint4 a = p[k * 32], b = p[(k + 1) * 32]; acc += a.x ^ b.w; }
  }
  __syncthreads();
  uint32_t s; asm("mov.u32 %0, %%smid;" : "=r"(s));
  if (threadIdx.x == 0) { out[blockIdx.x] = s; out[132 + blockIdx.x] = clock64() - t0 + (acc == 0x12345 ? 1 : 0); }
}
int mk_calibrate(const void* W, int tpw, long long* out_dev) {
  cudaFuncSetAttribute(mk_calib_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, g_smem);
  mk_calib_kernel<<<g_blocks, 512, g_smem>>>((const uint4*)W, tpw, out_dev);
  return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
}
int mk_serve_fetch(const int64_t* tok_out, int64_t* host, int B, int k0, int k1) {
  cudaMemcpyAsync(host, tok_out + (size_t)k0 * B, (size_t)(k1 - k0) * B * 8, cudaMemcpyDeviceToHost, g_side);
  return cudaStreamSynchronize(g_side) == cudaSuccess ? 0 : 1;
}

}  // extern "C"
