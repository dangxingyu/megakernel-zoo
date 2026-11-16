// Megakernel for Qwen3.8-27B text decode on one B200.
//
// One kernel launch = one pass of the whole model (one decode step, or P positions per sequence at once).
// The grid is one 1024-thread block per SM; the model is a "program" of ops; all blocks walk the program
// together and meet at a spin barrier after every op.
//
// Every big matrix is stored in tensor-core fragment order (m16n8k16, weights = A operand): a warp reads 512
// contiguous bytes per 16 rows x 16 columns and feeds them to MMA without any shuffling. The up to 8 MMA columns
// are the sequences of the batch (or positions of one sequence). Each warp gets an equal contiguous share of a
// matrix (split along K too); a 16-row tile cut by warp boundaries is put together in a fixed order, so results
// do not depend on timing, on the batch size or on the column a sequence is in.
//
// What the card taught (notes/findings.md): (1) L1 is not coherent between SMs inside one grid: data another
// block wrote in this launch is read with ld.global.cg, or from a buffer nobody read before in this launch;
// (2) a spin-wait must use a volatile load; (3) a thread's stack is real memory: nothing big is passed by
// reference to a function, and a function with spills is entered only by warps that have work;
// (4) 64 registers per thread: the hot loop is not unrolled.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

typedef unsigned long long u64;
typedef uint16_t u16;
typedef uint32_t u32;

#define HID 5120
#define VOCAB 248320
#define NCM 32            // max columns (buffer strides)
#define MROW 16480        // row stride of the mixer input buffer
#define RING 64           // conv history ring (positions per pass + 3 must fit)
#define PMAX 32           // max positions per pass

enum { OP_EMBED = 1, OP_GEMM = 2, OP_LMIX = 3, OP_ATT_PREP = 4, OP_ATT_SCORE = 5, OP_ATT_SOFTMAX = 6, OP_ATT_VSUM = 7,
       OP_ATT_OUT = 8, OP_ARGMAX = 9, OP_FINISH = 10, OP_MTP_IN = 11, OP_ACCEPT = 12, OP_CHAIN_FIN = 13, OP_ATT_DIST = 15 };
enum { EP_STORE = 1, EP_RESID = 2, EP_GATEUP = 3, EP_LOGITS = 4, EP_SET = 5 };
enum { F_SPEC = 1, F_LOGITS = 2, F_FEEDBACK = 4, F_CHAIN = 1024, F_DEFER = 16384 };
// F_DEFER (one-token passes, no speculation): the recurrent state is written to memory only on every
// Ctx.defer_n-th step of a sequence. In between a step saves its rank-1 update (decay, k, delta: 1 KB per head
// instead of 64 KB) and the following steps replay the saved updates on the stored state with the same lm_upd in
// the same order before their own: the state in the registers is bit for bit the state of a step-by-step run.
// Why: 24.6 MB of dirty state per layer at B=8 cost ~10 us per layer when L2 writes them back during the next
// weight streams (findings S1). nacc[NCM + b] = updates of sequence b not yet in its stored state.
#ifndef NGRAM
#define NGRAM 4           // a draft may be copied from the context after a match of this many tokens (6: one more ramp
                          // pass on looping text, hour 7.4; 2: loops -2.4 % but prose +2 %, B=8 +2.5-4 %, hour 12.1)
#endif

struct Op {   // 20 words
  long long type, ep, NT, KQ, use_r, M, i6, i7;
  void *W, *xf, *out, *gnext, *p4, *p5, *p6, *p7, *p8, *p9, *p10, *p11;
};

struct Ctx {
  int B, P, NC, NG;            // sequences, positions per pass, columns (B*P), column groups of 8
  int nwarps, nblocks, max_ctx, flags;
  int pos_ub, NGq, tok_stride, na_max;
  int pos_off, src_P, src_NG, src_sel;   // attention position offset; where MTP_IN finds the hidden state
  int next_P, defer_n, pad1, tok_tag;    // positions per sequence of the next verify pass (layout of tokbuf); F_DEFER window;
                                         // tag put into the upper half of the tokens written to tok_out (host-mapped output)
  u64 bar_base;
  u64* bar;
  int* pos;                    // [B] tokens already consumed per sequence
  int* nacc;                   // [B] pending state updates to commit (speculative mode)
  const long long* tok_in;     // input token of column c at tok_in[c * tok_stride]
  u16* h;                      // [NCM][HID] residual stream, bf16
  u16* mbuf;                   // [NCM][MROW] mixer inputs, bf16
  float* ssq;                  // [320][NCM] partial sums of squares of h
  float4* part;                // partial accumulators of tiles cut by a block boundary
  int* cnt;                    // per-tile counters for those
  u16* logits;                 // [NC][VOCAB] bf16, when F_LOGITS
  float* amax_val;             // [15520][NCM]
  int* amax_idx;
  int* tok_res;                // [NCM] argmax per column
  long long* tok_out;          // [NC] int64, may be null
  long long* tokbuf;           // [NCM] device-side next input tokens
  u16* qf;                     // attention: query fragments
  float* sc;                   // attention: scores
  u16* pf;                     // attention: probability fragments
  float4* vpart;               // attention: partial V sums
  const float* rope_cos;       // [max_ctx][32]
  const float* rope_sin;
  int* limit;                  // [B] a sequence stops advancing at this position (speculative mode)
  int* mtp_res;                // [NCM] argmax of the draft head per column
  int* cnt_out;                // [B] tokens accepted in this pass, may be null
  const long long* tok_next;   // prefill: the token after each column's token
  int* hist;                   // [B][max_ctx] the tokens of each sequence by position (speculative mode)
  int* lkf;                    // [B] 1 when the drafts of the next pass were copied from the context
  const long long* pft;        // [nops][nblocks][2]: (address, bytes) of this block's share of the next weight matrix
  unsigned long long* ts;      // debug: block 0 records the global timer after every barrier (null: off)
  unsigned long long* wts;     // debug: [nwarps] global timer when each warp's stream of op wts_op ends (null: off)
  int wts_op, pad3;
  const int* cst;               // [B+1] first column of each sequence in this pass (null: b * P)
  const int* nst;               // [B+1] the same for the drafts of the next pass in tokbuf_next (null: b * next_P)
  const int* sst;               // [B+1] the same for the pass MTP_IN takes its hidden states from (null: b * src_P)
  long long* tokbuf_next;       // where the verify pass's accept writes the next pass's drafts (null: tokbuf)
  unsigned long long* bts;      // debug: [nblocks][4] times in op wts_op: post-stream sync, finalize done, barrier arrival, release
};

// Shared memory: [0, 512) bytes the launch context and the current op, [512, 768) r per column, [768, 960) the
// warps' quad boundaries (QTAB_OFF), per-sequence state buffer and replay counts, the warps' finalize roles,
// [1280, ...) the work
// area (recurrent-state slots, or the partial sums of cut tiles). Its size is chosen per launch: what is not
// shared memory is L1 cache, and the stream is 6-12 % faster with 80 KB of shared memory than with 200 KB.
#define CTX_WORDS ((sizeof(Ctx) + 7) / 8)   // Ctx is 376 bytes: the op's last 3 words (p9-p11) share bytes 512-535
                                            // with the RMSNorm scalars (SMR_OFF); only GEMMs, which do not use them,
                                            // write r there - do not grow Ctx without moving SMR_OFF
#define SMR_OFF 128       // floats
#define RCNT_OFF (SMR_OFF + 48)   // int in shared memory: warps that have published r in this op (few columns)
#ifndef HC_TILES
#define HC_TILES 6                // residual rows preloaded for at most this many tiles per block (api.py: the same)
#endif
#define QTAB_OFF 192      // ints: [33] the first quad of each warp of the block in the current GEMM, and the block's end
#define SPAR_OFF 228      // ints: [8] per sequence the state buffer index (nacc[2 NCM + b]), fixed inside a launch
#define SNA_OFF 236       // ints: [8] per sequence the positions to replay (nacc[NCM + b]), fixed inside a launch
#define PFGO_OFF 244      // ints: [0] the op whose barrier increment thread 0 has issued (the block's prefetch waits for
                          // it), [1] thread 0's polls so far, [2] the op whose barrier this block has seen released
#define ROLE_OFF 256      // ints: [33] per warp the tile it finalizes after the block sync (+1, 0: none); [32] the block's outgoing tile
#define MAIN_OFF 292      // the main loop's program pointer (2 words) and op count: reloaded from here after every op
                          // instead of being kept across the op's call (a local-memory reload after a weight stream)
#define WORK_OFF 320      // floats
// Thread and block index read afresh (volatile): values derived from them were kept across the main loop's op
// calls and reloaded from local memory after each (findings C14)
__device__ __forceinline__ int tid_v() { int v; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(v)); return v; }
__device__ __forceinline__ int cta_v() { int v; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(v)); return v; }
__device__ __forceinline__ const Ctx& CX() { extern __shared__ float sm[]; return *(const Ctx*)sm; }
__device__ __forceinline__ const Op& OP() { extern __shared__ float sm[]; return *(const Op*)((const long long*)sm + CTX_WORDS); }

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ float bf2f(u16 b) { return __uint_as_float(((u32)b) << 16); }
__device__ __forceinline__ u16 f2bf(float f) {
  u32 u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (u16)(u >> 16);
}
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x * sigmoidf_(x); }

// Loads of anything another block may have written during this launch (bench/coh.cu: a default load can return
// the value this SM read before another SM overwrote it; ld.global.cg returns the new one).
__device__ __forceinline__ u16 ldv(const u16* p) { u32 v; asm volatile("ld.global.cg.u16 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return (u16)v; }
__device__ __forceinline__ int ldv(const int* p) { int v; asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ float ldv(const float* p) { float v; asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint2 ldv(const uint2* p) { uint2 v; asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint4 ldv(const uint4* p) { uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ u64 ldv(const u64* p) { u64 v; asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ float4 ldv(const float4* p) { float4 v; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory"); return v; }
// weights: read once, kept out of L1 so that the activations stay there
__device__ __forceinline__ uint2 ldx(const uint2* p) { uint2 v; asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p)); return v; }
// Cache operator of the K and V cache loads of full attention (each tile read once per step, appended by another SM in
// the same launch: never through L1): 0 cg, 1 cs
#ifndef KVL_K
#define KVL_K 0
#endif
__device__ __forceinline__ uint4 ld_kv(const uint4* p) {
#if KVL_K == 0
  return ldv(p);
#else
  uint4 v; asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v;
#endif
}
// Cache operator of the weight stream's loads: 0 cg (L2, not L1: the inherited engine), 1 ca, 2 cs ("cache streaming,
// likely to be accessed once"), 3 lu, 4 cv. Same card, B=1 / B=8 us per step: cg 8878 / 9785, cs 8658 / 9452,
// lu 8662 / 9464, cv 8877 / 9706, ca 8876 / 9710; identical logits (findings S2).
#ifndef LDW_K
#define LDW_K 2
#endif
#if LDW_K == 0
#define LDW_OP "ld.global.cg.v4.u32"
#elif LDW_K == 1
#define LDW_OP "ld.global.ca.v4.u32"
#elif LDW_K == 2
#define LDW_OP "ld.global.cs.v4.u32"
#elif LDW_K == 3
#define LDW_OP "ld.global.lu.v4.u32"
#else
#define LDW_OP "ld.global.cv.v4.u32"
#endif
__device__ __forceinline__ uint4 ldw(const uint4* p) { uint4 v; asm volatile(LDW_OP " {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }

// W12 (track B): 12 bits per weight, [sign][exponent - lo + 1][7 mantissa bits]; lo is the tensor's exponent
// window (15 consecutive exponents). Two codes in the low 24 bits of x -> two bf16 in one register:
// bf16 = code + C + sign * 0x7800 with C = (lo - 1) << 7 (the add moves the sign from bit 11 to bit 15).
// Weights outside the window (and exact zeros) carry code 0 and are patched from a list (see gemm_stream).
__device__ __forceinline__ u32 dec2(u32 x, u32 C2) {
  const u32 t = (x & 0x00FFF000u) * 16u + (x & 0x00000FFFu);   // two codes in the two halves
  return (t & 0x08000800u) * 15u + t + C2;                     // sign bit 11 -> bit 15, exponent base added
}
// 96 bits = the 8 weights of one thread and k-step -> the four MMA operand registers
__device__ __forceinline__ uint4 dec8(u32 u0, u32 u1, u32 u2, u32 C2) {
  return make_uint4(dec2(u0, C2), dec2((u0 >> 24) | (u1 << 8), C2), dec2((u1 >> 16) | (u2 << 16), C2), dec2(u2 >> 8, C2));
}
__device__ __forceinline__ void w12_patch(uint4& r, u32 reg, u32 mk, u32 vv) {
  if (reg < 2) { if (reg == 0) r.x = (r.x & mk) | vv; else r.y = (r.y & mk) | vv; }
  else { if (reg == 2) r.z = (r.z & mk) | vv; else r.w = (r.w & mk) | vv; }
}
// The B operand of the GEMM streams: lane = column * 4 + t; lanes of columns >= NC (at B=1: 28 of 32) take zeros
// instead of loading (-DX_ALL: every column, as the inherited engine). Used by gemm_stream_bf only. The real columns' results are unchanged bit for bit.
__device__ __forceinline__ uint2 xld(const uint2* p, bool act) {
#ifdef X_ALL   // every column loaded (as the inherited engine); default since hour 3.4: real columns only (-0.65 % at B=1)
  return *p;
#else
  uint2 v = make_uint2(0u, 0u);
  if (act) v = *p;
  return v;
#endif
}
__device__ __forceinline__ void mma(float* c, const uint4& a, const uint2& b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}
// index (in u16 units) of element (column c, input k) in an x-fragment buffer (B operand of m16n8k16)
__device__ __forceinline__ size_t xf_idx(int c, int k, int NG) {
  int s = k >> 4, r = k & 15;
  int t = (r & 7) >> 1, half = r & 1, reg = r >> 3;
  int lane = (c & 7) * 4 + t;
  return ((size_t)(s * NG + (c >> 3)) * 32 + lane) * 4 + reg * 2 + half;
}
// GEMM input buffers at one column (NC == 1, -DX_COMPACT, W12 builds): only column 0's B-fragment values, 32 bytes
// per k-step instead of 256 (lane t of column 0 holds k = 2t, 2t+1, 2t+8, 2t+9). The streams' lanes of the other 7
// columns read column 0's values too (no predicate); those output columns are never written out (epilogues: cb < NC).
// The activations' L1 footprint at B = 1 shrinks 8x (a K = 5120 input: 10 KB instead of 80 KB).
__device__ __forceinline__ size_t xc_idx(int k) {
#ifdef X_C2   // lane t's values of two consecutive k-steps contiguous: one 16-byte load for two k-steps
  const int r = k & 15;
  return ((size_t)((k >> 5) * 4 + ((r & 7) >> 1))) * 8 + ((k >> 4) & 1) * 4 + (r >> 3) * 2 + (r & 1);
#else
  int s = k >> 4, r = k & 15;
  return ((size_t)(s * 4 + ((r & 7) >> 1))) * 4 + (r >> 3) * 2 + (r & 1);
#endif
}
__device__ __forceinline__ bool x_compact() {
#ifdef X_COMPACT
  return CX().NC == 1;
#else
  return false;
#endif
}
__device__ __forceinline__ size_t xg_idx(int c, int k, int NG) { return x_compact() ? xc_idx(k) : xf_idx(c, k, NG); }
__device__ __forceinline__ size_t xg_idx(int c, int k, int NG, bool cmp) { return cmp ? xc_idx(k) : xf_idx(c, k, NG); }
// index (u16 units) of element (row r of 16, col kk of 16) inside one 512-byte A-operand tile
__device__ __forceinline__ int af_idx(int r, int kk) {
  int lane = (r & 7) * 4 + ((kk & 7) >> 1);
  return lane * 8 + ((r >> 3) + 2 * (kk >> 3)) * 2 + (kk & 1);
}
// Column layout of a pass: sequence b owns columns [c_start, c_start + c_len). Uniform (b * P) unless the host
// gave per-sequence lengths (cst), which it does at B > 1 to give more positions to sequences whose drafts were copied.
__device__ __forceinline__ void dbg_stamp(int k) {   // debug timer of the tail of op wts_op (thread 0 of each block)
#ifdef NO_STAMPS
  return;
#endif
  extern __shared__ float sm[];
  const Ctx& cx = *(const Ctx*)sm;
  if (cx.bts && threadIdx.x == 0 && *(const volatile int*)(sm + RCNT_OFF + 1) == cx.wts_op) {
#ifdef HO_STAMPS
    unsigned long long t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); cx.bts[blockIdx.x * 12 + k] = t;
#else
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); cx.bts[blockIdx.x * 12 + k] = t;
#endif
  }
}
__device__ __forceinline__ void dbg_stamp31(int k) {   // the same, from lane 0 of warp 31 (slots 4-11)
#ifdef NO_STAMPS
  return;
#endif
  extern __shared__ float sm[];
  const Ctx& cx = *(const Ctx*)sm;
#ifdef HO_STAMPS
  if (false) {
#else
  if (cx.bts && threadIdx.x == 992 && *(const volatile int*)(sm + RCNT_OFF + 1) == cx.wts_op) {
#endif
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); cx.bts[blockIdx.x * 12 + k] = t;
  }
}
__device__ __forceinline__ void dbg_stampw(int k) {   // the same, from lane 0 of the calling warp (debug of the hand-over)
#ifdef NO_STAMPS
  return;
#endif
  extern __shared__ float sm[];
  const Ctx& cx = *(const Ctx*)sm;
  if (cx.bts && (threadIdx.x & 31) == 0 && *(const volatile int*)(sm + RCNT_OFF + 1) == cx.wts_op) {
    unsigned long long t; asm volatile("mov.u64 %0, %%clock64;" : "=l"(t)); cx.bts[blockIdx.x * 12 + k] = t;
  }
}
// Sequence positions do not change inside a launch (only its last op, accept or finish, moves them): read once at
// the start into shared memory instead of a global round trip at the start of every recurrent and attention op.
__device__ __forceinline__ int spos(int b) {
  extern __shared__ float sm[];
  return b < 8 ? ((volatile const int*)(sm + RCNT_OFF + 6))[b] : ldv(CX().pos + b);
}
// The same for the recurrent state's buffer index and the replay count (written only by the accept op, the last
// op of a pass): they were a dependent global load at the start of every recurrent unit
__device__ __forceinline__ int spar(int b) {
  extern __shared__ float sm[];
  return b < 8 ? ((volatile const int*)(sm + SPAR_OFF))[b] : ldv(CX().nacc + 2 * NCM + b);
}
__device__ __forceinline__ int sna(int b) {
  extern __shared__ float sm[];
  return b < 8 ? ((volatile const int*)(sm + SNA_OFF))[b] : ldv(CX().nacc + NCM + b);
}
// The partition of a GEMM, computed instead of loaded: warp w (of nw) streams quads [qw(w), qw(w + 1)) of the
// matrix's UQ = NT * KQ quads. The tables in global memory were first-touch L2 loads on the critical path at the
// start of the stream and in the finalize chain (~1 us each under load).
// floor(a / b) for 0 <= a < 2^53 in double precision: exact here (a quotient that is not an integer is at least 1 / b
// away from one, far more than the rounding error), and much cheaper than a 64-bit integer division.
// floor(a / b) for a >= 0, b > 0, quotient < 2^24: a float estimate (no slow-path call, unlike a double division)
// corrected exactly with integer arithmetic
__device__ __forceinline__ int fdiv(long long a, long long b) {
  int q = (int)__fdividef((float)a, (float)b);
  long long r = a - (long long)q * b;
  while (r < 0) { --q; r += b; }
  while (r >= b) { ++q; r -= b; }
  return q;
}
__device__ __forceinline__ int qw(long long UQ, int w, int nw) { return fdiv(UQ * w, nw); }
__device__ __forceinline__ int c_start(const Ctx& cx, int b) { return cx.cst ? cx.cst[b] : b * cx.P; }
__device__ __forceinline__ int c_len(const Ctx& cx, int b) { return cx.cst ? cx.cst[b + 1] - cx.cst[b] : cx.P; }
__device__ __forceinline__ int c_seq(const Ctx& cx, int c) {
  if (!cx.cst) return c / cx.P;
  int b = 0;
  while (cx.cst[b + 1] <= c) ++b;
  return b;
}
__device__ __forceinline__ int n_start(const Ctx& cx, int b) { return cx.nst ? cx.nst[b] : b * cx.next_P; }
__device__ __forceinline__ int n_len(const Ctx& cx, int b) { return cx.nst ? cx.nst[b + 1] - cx.nst[b] : cx.next_P; }
__device__ __forceinline__ int s_start(const Ctx& cx, int b) { return cx.sst ? cx.sst[b] : b * cx.src_P; }
// Running argmax as one 64-bit integer: a larger value wins, an equal value with a smaller index wins (the tie rule
// of every other argmax here). Exact and independent of the order in which tiles arrive.
__device__ __forceinline__ u64 amax_key(float v, int i) {
  const u32 b = __float_as_uint(v);
  const u32 k = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  return ((u64)k << 32) | (u64)(0xFFFFFFFFu - (u32)i);
}
__device__ __forceinline__ float warp_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}
// Grid-wide barrier; it also brings the next op into shared memory. The wait reads the counter with a volatile
// load: a spin on ld.global(.cg/.cv) never sees it change (bench/bar.cu). 0.8 us with 152 blocks.
// Release/acquire at GPU scope (PTX memory model): the arrival is a release add (cumulative over the block's stores,
// which __syncthreads orders before it), the wait an acquire load, so stores before the barrier are visible after it.
__device__ __forceinline__ void red_release(u64* p, u64 v) { asm volatile("red.release.gpu.global.add.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory"); }
__device__ __forceinline__ u64 ld_acquire(const u64* p) { u64 v; asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
#ifdef SENT_DBG
__device__ unsigned long long sent_dbg[4];
__device__ unsigned long long sent_ep[1 << 16];
__device__ unsigned long long sent_log[2][1 << 17];
extern "C" int mk_sent_dbg(unsigned long long* out) { return (int)cudaMemcpyFromSymbol(out, sent_dbg, 32); }
extern "C" int mk_sent_log(unsigned long long* out) { return (int)cudaMemcpyFromSymbol(out, sent_log, sizeof(sent_log)); }
#endif
// The hand-over words are read and written with morally strong (relaxed, gpu-scope) accesses: a weak ld.cg poll
// was seen returning an old copy of the slot (the previous op's data) - on this two-die card weak loads are not
// coherent with another SM's later stores without an acquire.
__device__ __forceinline__ float4 ld_rlx4(const float4* p) { float4 v; asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ float ld_rlx(const float* p) { float v; asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rlx(float* p, float v) { asm volatile("st.relaxed.gpu.global.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }
__device__ __forceinline__ void st_rlx4(float4* p, float4 v) { asm volatile("st.relaxed.gpu.global.v4.f32 [%0], {%1,%2,%3,%4};" :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory"); }
#define SENT_BITS 0x7fbadbadu   // a NaN payload FMA/MMA never produce (their NaNs are canonical)
__device__ __forceinline__ bool is_sent(float x) { return __float_as_uint(x) == SENT_BITS; }
__device__ __forceinline__ float4 sent4() { const float f = __uint_as_float(SENT_BITS); return make_float4(f, f, f, f); }
__device__ __forceinline__ int atom_add_acqrel(int* p, int v) { int r; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
__device__ __forceinline__ void barrier(const long long* prog, int o, int nops) {
  extern __shared__ float sm[];
  __syncthreads();
  if (threadIdx.x == 32) {
    // While this block waits for the others (and while the small ops that follow run), DRAM would idle: ask L2 to
    // fetch the start of this block's share of the next weight matrix (the host chose the size per op).
    const Ctx& cx = CX();
    if (cx.pft) {
      const long long* e = cx.pft + ((size_t)o * cx.nblocks + blockIdx.x) * 2;
      const unsigned long long addr = (unsigned long long)e[0];
      long long left = e[1];
#ifndef PF_WITH_RED   // the requests go out only when thread 0's counter increment is out. Issued alongside it (as
                      // inherited), the 160-414 KB of prefetch lines could sit in front of the increment in the SM's
                      // memory queue and hold the release for every block: B=1 -27 us per step, B=8 -10 (findings S12)
      if (left > 0) while (*(volatile int*)(sm + PFGO_OFF) != o + 1) {}
#endif
      for (unsigned long long a = addr; left > 0; a += 65536, left -= 65536) {
        const unsigned n = left < 65536 ? (unsigned)left : 65536u;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a), "r"(n) : "memory");
#ifndef PF_NOSTEP   // one 64 KB piece per poll of thread 0 and none after the release: the block's polls no longer
                   // queue behind all of its prefetch lines, and a block that arrives late stops asking (findings S12)
        const int c0 = *(volatile int*)(sm + PFGO_OFF + 1);
        while (*(volatile int*)(sm + PFGO_OFF + 1) == c0 && *(volatile int*)(sm + PFGO_OFF + 2) != o + 1) {}
        if (*(volatile int*)(sm + PFGO_OFF + 2) == o + 1) break;
#endif
      }
    }
  }
  if (threadIdx.x == 0) {
    const Ctx& cx = CX();
    u64* bar = cx.bar;
    const u64 target = cx.bar_base + (u64)(o + 1) * (u64)cx.nblocks;
    dbg_stamp(2);
#ifdef NO_FENCE
    atomicAdd(bar, 1ull);
    while (*(volatile u64*)bar < target) {}
#else
    red_release(bar, 1ull);
#ifndef PF_WITH_RED
    *(volatile int*)(sm + PFGO_OFF) = o + 1;
#endif
    while (ld_acquire(bar) < target) {
#ifndef PF_NOSTEP
      ++*(volatile int*)(sm + PFGO_OFF + 1);
#endif
#ifdef BAR_SLEEP_NS
      { unsigned ns = BAR_SLEEP_NS; asm volatile("nanosleep.u32 %0;" :: "r"(ns)); }
#endif
    }
#ifndef PF_NOSTEP
    *(volatile int*)(sm + PFGO_OFF + 2) = o + 1;
#endif
    dbg_stamp(3);
#ifndef NO_STAMPS
    if (cx.ts && blockIdx.x == 0) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); cx.ts[o] = t; }
#endif
    *(volatile int*)(sm + RCNT_OFF) = 0;
    *(volatile int*)(sm + RCNT_OFF + 1) = o + 1;
    *(volatile int*)(sm + RCNT_OFF + 5) = 0;
#endif
  } else if (threadIdx.x <= 20 && o + 1 < nops) {
    ((long long*)sm + CTX_WORDS)[threadIdx.x - 1] = prog[(size_t)(o + 1) * 20 + threadIdx.x - 1];
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------ GEMM
#ifndef R_ASYNC_MAX
#define R_ASYNC_MAX 4   // RMSNorm scalar without a block barrier up to this many columns (8: no change at B=8; 32: +1.8 ms per 32-column pass)
#endif
// Tile sums live in shared memory (tried in a per-block global scratch area to leave L1 larger: no gain in the
// stream, slower epilogues; see journal).
// What happens to the 16 x 8 results of a tile (per column group), read from shared-memory slot `sidx`.
// RMSNorm scalar with more than R_ASYNC_MAX columns: every warp has left its partial sums of squares in shared
// memory (RPART) and counted itself; the first epilogue that needs r makes one warp finish the 32-way reduction in
// the order of warp_sum's butterfly (the same bits as every other path), the others wait for it.
#define RPART_FLOATS(NG) (32 * 2 * (NG) * 128 + HC_TILES * 16 * NCM / 2)   // after the tile-sum slots and residual rows
__device__ __forceinline__ void ensure_r_big(int NC, int NG) {
  extern __shared__ float sm[];
  volatile int* fl = (volatile int*)(sm + RCNT_OFF + 5);
  if (*fl == 2) { __threadfence_block(); return; }
  while (*(volatile int*)(sm + RCNT_OFF) < 32) {}
  __threadfence_block();
  const int lane = threadIdx.x & 31;
  int won = 0;
  if (lane == 0) won = atomicCAS((int*)fl, 0, 1) == 0;
  won = __shfl_sync(0xffffffffu, won, 0);
  if (won) {
    if (lane < NC) {
      const volatile float* part = sm + WORK_OFF + RPART_FLOATS(NG);
      float v[32];
      #pragma unroll
      for (int j = 0; j < 32; ++j) v[j] = part[j * 32 + lane];
      #pragma unroll
      for (int h = 16; h >= 1; h >>= 1)
        #pragma unroll
        for (int j = 0; j < h; ++j) v[j] = v[j] + v[j + h];
      sm[SMR_OFF + lane] = rsqrtf(v[0] * (1.0f / HID) + 1e-6f);
    }
    __syncwarp();
    __threadfence_block();
    if (lane == 0) *fl = 2;
    __syncwarp();
  } else {
    while (*fl != 2) {}
    __threadfence_block();
  }
}
template <int NGT, int EPT, bool POST>
__device__ __noinline__ void epilogue_k(int T, int sidx) {
  const bool post = POST;
  extern __shared__ float sm[];
  const Ctx& cx = CX(); const Op& op = OP();
  const float* smr = sm + SMR_OFF;
  const float4* ps = (const float4*)(sm + WORK_OFF);
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int NC = cx.NC, ep = EPT;   // one instance per kind: each gets its own registers (spills cost ~1 us per
                                    // column group when L1 is small)
  const bool use_r = op.use_r != 0;
  if (use_r) {
    if (NC <= R_ASYNC_MAX) {
      while (*(volatile int*)(sm + RCNT_OFF) < NC) {}
      __threadfence_block();
    } else ensure_r_big(NC, NGT);
  }
  // residual epilogue: every load (gains, residual rows of all column groups) before the first store, so that
  // they are one round trip and not one per column group
  float g0 = 0.f, g1 = 0.f;
  u32 ho[NGT][2];   // packed: (row0 | row1 << 16) of columns cb, cb + 1
  const int* hci = (const int*)(sm + RCNT_OFF + 2);   // first tile and count of the block's preloaded residual rows
  const bool inhc = POST && T >= hci[0] && T < hci[0] + hci[1];   // after the block barrier: the copy loaded while streaming
  if (ep == EP_RESID || ep == EP_SET) {
    const int row0 = T * 16 + g, row1 = row0 + 8;
    const float* gn = (const float*)op.gnext;
    g0 = gn[row0]; g1 = gn[row1];
    const int* hci = (const int*)(sm + RCNT_OFF + 2);   // first tile and count of the block's preloaded residual rows
    const int t0 = hci[0], nt = hci[1];
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) {
      const int cb = cg * 8 + 2 * t;
      const u16* hp0 = cx.h + (size_t)cb * HID;
      const u16* hp1 = hp0 + HID;
      u32 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      if (ep == EP_RESID && !POST) {   // after the block barrier the rows are read from shared memory in the loop
        (void)t0; (void)nt;
        {
          if (cb < NC) { a0 = ldv(hp0 + row0); a1 = ldv(hp0 + row1); }
          if (cb + 1 < NC) { a2 = ldv(hp1 + row0); a3 = ldv(hp1 + row1); }
        }
      }
      ho[cg][0] = a0 | (a1 << 16); ho[cg][1] = a2 | (a3 << 16);
    }
  }
  if (post) dbg_stamp31(8);
  #pragma unroll
  for (int cg = 0; cg < NGT; ++cg) {
    const int cb = cg * 8 + 2 * t;
    float r0 = 1.0f, r1 = 1.0f;
    if (use_r) { r0 = ((volatile const float*)smr)[cb]; r1 = ((volatile const float*)smr)[cb + 1]; }
    const float4 cc = ps[sidx + cg * 32];
    const float v00 = cc.x * r0, v01 = cc.y * r1, v10 = cc.z * r0, v11 = cc.w * r1;
    if (ep == EP_STORE) {
      u16* out = (u16*)op.out;
      const int row0 = T * 16 + g, row1 = row0 + 8;
      if (cb < NC) { out[(size_t)cb * MROW + row0] = f2bf(v00); out[(size_t)cb * MROW + row1] = f2bf(v10); }
      if (cb + 1 < NC) { out[(size_t)(cb + 1) * MROW + row0] = f2bf(v01); out[(size_t)(cb + 1) * MROW + row1] = f2bf(v11); }
    } else if (ep == EP_RESID || ep == EP_SET) {
      const int row0 = T * 16 + g, row1 = row0 + 8;
      u16* xn = (u16*)op.out;   // the next matrix's x fragments: bf16(h (1+w)), r is applied after that GEMM
      u16* hp0 = cx.h + (size_t)cb * HID;
      u16* hp1 = hp0 + HID;
      u16 o00, o10, o01, o11;
      if (!POST) { o00 = (u16)(ho[cg][0] & 0xffffu); o10 = (u16)(ho[cg][0] >> 16); o01 = (u16)(ho[cg][1] & 0xffffu); o11 = (u16)(ho[cg][1] >> 16); }
      else {
        o00 = 0; o10 = 0; o01 = 0; o11 = 0;
        if (ep == EP_RESID) {
          if (inhc) {
            const u16* hc = (const u16*)(sm + WORK_OFF + 32 * 2 * NGT * 128) + (size_t)(T - hci[0]) * 16;
            if (cb < NC) { o00 = hc[cb * (HC_TILES * 16) + g]; o10 = hc[cb * (HC_TILES * 16) + g + 8]; }
            if (cb + 1 < NC) { o01 = hc[(cb + 1) * (HC_TILES * 16) + g]; o11 = hc[(cb + 1) * (HC_TILES * 16) + g + 8]; }
          } else {
            if (cb < NC) { o00 = ldv(hp0 + row0); o10 = ldv(hp0 + row1); }
            if (cb + 1 < NC) { o01 = ldv(hp1 + row0); o11 = ldv(hp1 + row1); }
          }
        }
      }
      float s0 = 0.f, s1 = 0.f;
      if (cb < NC) {
        const float a = rbf(bf2f(o00) + rbf(v00)), b = rbf(bf2f(o10) + rbf(v10));
        hp0[row0] = f2bf(a); hp0[row1] = f2bf(b);
        xn[xg_idx(cb, row0, NGT)] = f2bf(a * g0);
        xn[xg_idx(cb, row1, NGT)] = f2bf(b * g1);
        s0 = a * a + b * b;
      }
      if (cb + 1 < NC) {
        const float a = rbf(bf2f(o01) + rbf(v01)), b = rbf(bf2f(o11) + rbf(v11));
        hp1[row0] = f2bf(a); hp1[row1] = f2bf(b);
        xn[xg_idx(cb + 1, row0, NGT)] = f2bf(a * g0);
        xn[xg_idx(cb + 1, row1, NGT)] = f2bf(b * g1);
        s1 = a * a + b * b;
      }
      if (post && cg == 0) dbg_stamp31(9);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 4);  s1 += __shfl_xor_sync(0xffffffffu, s1, 4);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 8);  s1 += __shfl_xor_sync(0xffffffffu, s1, 8);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 16); s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
      if (g == 0) {
        if (cb < NC) cx.ssq[(size_t)T * NCM + cb] = s0;
        if (cb + 1 < NC) cx.ssq[(size_t)T * NCM + cb + 1] = s1;
      }
      if (post && cg == 0) dbg_stamp31(10);
    } else if (ep == EP_GATEUP) {
      u16* out = (u16*)op.out;
      const int n = T * 8 + g;
      if (cb < NC) out[xg_idx(cb, n, NGT)] = f2bf(rbf(siluf_(rbf(v00))) * rbf(v10));
      if (cb + 1 < NC) out[xg_idx(cb + 1, n, NGT)] = f2bf(rbf(siluf_(rbf(v01))) * rbf(v11));
    } else {  // EP_LOGITS
      const int row0 = T * 16 + g, row1 = row0 + 8;
      const u16 b00 = f2bf(v00), b01 = f2bf(v01), b10 = f2bf(v10), b11 = f2bf(v11);
      if ((cx.flags & F_LOGITS) && op.i6) {
        if (cb < NC) { cx.logits[(size_t)cb * VOCAB + row0] = b00; cx.logits[(size_t)cb * VOCAB + row1] = b10; }
        if (cb + 1 < NC) { cx.logits[(size_t)(cb + 1) * VOCAB + row0] = b01; cx.logits[(size_t)(cb + 1) * VOCAB + row1] = b11; }
      }
      float m0 = bf2f(b00), m1 = bf2f(b01); int i0 = row0, i1 = row0;
      { float x = bf2f(b10); if (x > m0) { m0 = x; i0 = row1; } }
      { float x = bf2f(b11); if (x > m1) { m1 = x; i1 = row1; } }
      #pragma unroll
      for (int sh = 4; sh <= 16; sh <<= 1) {
        float om0 = __shfl_xor_sync(0xffffffffu, m0, sh), om1 = __shfl_xor_sync(0xffffffffu, m1, sh);
        int oi0 = __shfl_xor_sync(0xffffffffu, i0, sh), oi1 = __shfl_xor_sync(0xffffffffu, i1, sh);
        if (om0 > m0 || (om0 == m0 && oi0 < i0)) { m0 = om0; i0 = oi0; }
        if (om1 > m1 || (om1 == m1 && oi1 < i1)) { m1 = om1; i1 = oi1; }
      }
      if (g == 0) {
        if (op.out) {   // fold into the column's running argmax: (order-preserving key of the value, ~index); exact, any order
          u64* am = (u64*)op.out;
          if (cb < NC) atomicMax(am + cb, amax_key(m0, i0));
          if (cb + 1 < NC) atomicMax(am + cb + 1, amax_key(m1, i1));
        } else {
          if (cb < NC) { cx.amax_val[(size_t)T * NCM + cb] = m0; cx.amax_idx[(size_t)T * NCM + cb] = i0; }
          if (cb + 1 < NC) { cx.amax_val[(size_t)T * NCM + cb + 1] = m1; cx.amax_idx[(size_t)T * NCM + cb + 1] = i1; }
        }
      }
    }
  }
}

template <int NGT, bool POST>
__device__ __forceinline__ void epilogue_p(int T, int sidx) {
  switch ((int)OP().ep) {
    case EP_STORE: epilogue_k<NGT, EP_STORE, POST>(T, sidx); break;
    case EP_RESID: epilogue_k<NGT, EP_RESID, POST>(T, sidx); break;
    case EP_SET: epilogue_k<NGT, EP_SET, POST>(T, sidx); break;
    case EP_GATEUP: epilogue_k<NGT, EP_GATEUP, POST>(T, sidx); break;
    default: epilogue_k<NGT, EP_LOGITS, POST>(T, sidx); break;
  }
}
template <int NGT>
__device__ __forceinline__ void epilogue(int T, int sidx, bool post) {
  if (post) epilogue_p<NGT, true>(T, sidx); else epilogue_p<NGT, false>(T, sidx);
}

// A tile cut by warp boundaries, put together by the warp of this block that holds its last piece: the pieces
// inside the block through shared memory in warp order; if the tile is also cut by a block boundary, one partial
// per block through global memory, summed in block order by the block that arrives last (a counter per tile).
template <int NGT>
__device__ __noinline__ void finalize(int T, int myslot, int wib) {   // wib: the warp whose slot holds the last piece
  extern __shared__ float sm[];
  const Ctx& cx = CX(); const Op& op = OP();
  const int lane = threadIdx.x & 31, bl = blockIdx.x;
  const int KQ = (int)op.KQ, nw = cx.nwarps;
  const long long UQ = (long long)op.NT * KQ;
  float4* ps = (float4*)(sm + WORK_OFF);
  const int own = (wib * 2) * NGT * 32 + lane;
  dbg_stamp31(4);
#ifdef HO_STAMPS
  if (myslot == 0) dbg_stampw(9);
#endif
  const int* qt = (const int*)(sm + QTAB_OFF);   // the block's quad boundaries (qw), from the stream prologues
  // the first warp of this block with a piece of T: the last warp starting at or before quad T KQ (0 if none)
  const unsigned xm = __ballot_sync(0xffffffffu, qt[lane] <= T * KQ);
  const int x = xm ? 31 - __clz(xm) : 0;
  #pragma unroll
  for (int cg = 0; cg < NGT; ++cg) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    for (int i = x; i < wib; ++i) {
      const float4 v = ps[((i * 2 + 1) * NGT + cg) * 32 + lane];
      s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
    }
    const float4 v = ps[own + (myslot * NGT + cg) * 32];
    ps[own + (myslot * NGT + cg) * 32] = make_float4(s0 + v.x, s1 + v.y, s2 + v.z, s3 + v.w);
  }
#ifdef HO_STAMPS
  if (myslot == 0) dbg_stampw(10);
#endif
  const bool in_first = T * KQ >= qt[0], in_last = (T + 1) * KQ <= qt[32];
  if (in_first && in_last) { epilogue<NGT>(T, own + myslot * NGT * 32, true); return; }
  // first and last block with a piece of T (the blocks of quads T KQ and (T + 1) KQ - 1)
  // when a tile is not longer than the shortest block share (floor(32 UQ / nw) >= KQ), it spans at most two blocks
  const bool two = (long long)(KQ + 1) * nw <= 32 * UQ;
  const int bf = in_first ? bl : (two ? bl - 1 : fdiv(((long long)T * KQ + 1) * nw - 1, UQ) >> 5);
  const int bla = in_last ? bl : (two ? bl + 1 : fdiv(((long long)(T + 1) * KQ) * nw - 1, UQ) >> 5);
#ifndef NO_SENTINEL
  if (bla - bf == 1) {
    // A tile cut once, handed over with sentinels: every partial slot holds a NaN pattern arithmetic never makes
    // (put back by the reader after use, ordered before the slot's next writer by the grid barrier); the first
    // block just stores its partial, the last block polls the slot until the data is there (one trip through the
    // memory queue instead of store + release + atomic + load), adds in block order and puts the sentinel back.
    float4* slot = (float4*)cx.part + ((size_t)(T + bf) * NGT) * 32 + lane;
#ifdef HO_STAMPS
    dbg_stampw(bl == bf ? 7 : 4);
#endif
    if (bl == bf) {
#ifdef SENT_DBG
      { extern __shared__ float sm_[]; if (lane == 0) { sent_ep[(T + bf) & 65535] = (unsigned long long)cx.bar_base + *(volatile int*)(sm_ + RCNT_OFF + 1) + 1; } __threadfence(); }
#endif
      #pragma unroll
      for (int cg = 0; cg < NGT; ++cg) st_rlx4(slot + cg * 32, ps[own + (myslot * NGT + cg) * 32]);
#ifdef SENT_DBG
      if (lane == 0) { const unsigned long long i = atomicAdd(sent_dbg + 0, 1ull); if (i < (1u << 17)) { extern __shared__ float sm_[]; sent_log[0][i] = ((unsigned long long)*(volatile int*)(sm_ + RCNT_OFF + 1) << 40) | ((unsigned long long)T << 20) | (unsigned)(T + bf); } }
#endif
      return;
    }
#ifdef SENT_DBG
    if (lane == 0) { const unsigned long long i = atomicAdd(sent_dbg + 1, 1ull); if (myslot != 0) atomicAdd(sent_dbg + 3, 1ull);
      if (i < (1u << 17)) { extern __shared__ float sm_[]; sent_log[1][i] = ((unsigned long long)*(volatile int*)(sm_ + RCNT_OFF + 1) << 40) | ((unsigned long long)T << 20) | (unsigned)(T + bf); } }
#endif
    float4 v0[NGT];
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) {
      float4 v;
      do { v = ld_rlx4(slot + cg * 32); } while (is_sent(v.x) || is_sent(v.y) || is_sent(v.z) || is_sent(v.w));
      v0[cg] = v;
    }
#ifdef HO_STAMPS
    dbg_stampw(5);
#endif
#ifdef SENT_DBG
    { extern __shared__ float sm_[]; __threadfence();
      if (lane == 0) { const unsigned long long e = *(volatile unsigned long long*)(sent_ep + ((T + bf) & 65535));
        if (e != (unsigned long long)cx.bar_base + *(volatile int*)(sm_ + RCNT_OFF + 1) + 1) atomicAdd(sent_dbg + 2, 1ull); } }
#endif
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) {   // the same additions in the same order as the counter path
      const float4 m = ps[own + (myslot * NGT + cg) * 32];
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
      s0 += v0[cg].x; s1 += v0[cg].y; s2 += v0[cg].z; s3 += v0[cg].w;
      s0 += m.x; s1 += m.y; s2 += m.z; s3 += m.w;
      ps[own + (myslot * NGT + cg) * 32] = make_float4(s0, s1, s2, s3);
      st_rlx4(slot + cg * 32, sent4());
    }
    epilogue<NGT>(T, own + myslot * NGT * 32, true);
#ifdef HO_STAMPS
    dbg_stampw(6);
#endif
    return;
  }
#endif
  #pragma unroll
  for (int cg = 0; cg < NGT; ++cg) cx.part[((size_t)(T + bl) * NGT + cg) * 32 + lane] = ps[own + (myslot * NGT + cg) * 32];
  __syncwarp();   // the warp's partial stores are ordered before lane 0's release
  int old = 0;
  if (lane == 0) old = atom_add_acqrel(cx.cnt + T, 1);
  old = __shfl_sync(0xffffffffu, old, 0);
  __syncwarp();   // ... and lane 0's acquire before the warp's loads of the other blocks' partials
  dbg_stamp31(5);
  if (old + 1 != bla - bf + 1) return;
  if (bla - bf <= 1) {   // the usual case, a tile cut once: every partial of every column group requested at once
    float4 v0[NGT], v1[NGT];
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) {
      v0[cg] = ldv(cx.part + ((size_t)(T + bf) * NGT + cg) * 32 + lane);
      if (bla > bf) v1[cg] = ldv(cx.part + ((size_t)(T + bla) * NGT + cg) * 32 + lane);
    }
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) {   // the same additions in the same order as the general loop below
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
      s0 += v0[cg].x; s1 += v0[cg].y; s2 += v0[cg].z; s3 += v0[cg].w;
      if (bla > bf) { s0 += v1[cg].x; s1 += v1[cg].y; s2 += v1[cg].z; s3 += v1[cg].w; }
      ps[own + (myslot * NGT + cg) * 32] = make_float4(s0, s1, s2, s3);
    }
  } else
  #pragma unroll
  for (int cg = 0; cg < NGT; ++cg) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    for (int b = bf; b <= bla; b += 4) {   // 4 loads in flight, summed in block order
      float4 v[4];
      #pragma unroll
      for (int u = 0; u < 4; ++u)
        if (b + u <= bla) v[u] = ldv(cx.part + ((size_t)(T + b + u) * NGT + cg) * 32 + lane);
      #pragma unroll
      for (int u = 0; u < 4; ++u)
        if (b + u <= bla) { s0 += v[u].x; s1 += v[u].y; s2 += v[u].z; s3 += v[u].w; }
    }
    ps[own + (myslot * NGT + cg) * 32] = make_float4(s0, s1, s2, s3);
  }
#ifndef NO_SENTINEL
  for (int b = bf; b <= bla; ++b) {   // the slots go back to the sentinel for their next (sentinel) use
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) cx.part[((size_t)(T + b) * NGT + cg) * 32 + lane] = sent4();
  }
#endif
  if (lane == 0) cx.cnt[T] = 0;
  dbg_stamp31(6);
  epilogue<NGT>(T, own + myslot * NGT * 32, true);
  dbg_stamp31(7);
}

// The BF16 stream with the fewest live registers: the loop carries two pointers, a down-counter, the quads left in
// the current tile, the accumulators, one quad of weights and one k-step of activations; everything else is
// recomputed (from the op in shared memory) at the end of a tile. With more live state the 32-column loop spilled
// 8 registers per iteration (SASS) and streamed at 44 us where bench/ng.cu does 28.
// After the block sync: a warp whose share holds the end of a tile it did not start puts that tile together; the
// block's outgoing cut tile (warp 31's last piece, slot 1) is finalized by warp 30 or 29 when that warp has no tile
// of its own to finish, so that it does not wait behind warp 31's own epilogue.
template <int NGT>
__device__ __forceinline__ void finalize_all(int q0, int q1, int KQ) {
  const int wib = threadIdx.x >> 5;
  const int Ta = q0 / KQ;
#ifdef HO_STAMPS
  if (q0 > Ta * KQ && (Ta + 1) * KQ <= q1) dbg_stampw(8);
#endif
  if (q0 > Ta * KQ && (Ta + 1) * KQ <= q1) finalize<NGT>(Ta, 0, wib);
  if (wib >= 29) {
    const Ctx& cx = CX(); const Op& op = OP();
    const long long UQ = (long long)op.NT * KQ;
    const int wb = blockIdx.x * 32;
    extern __shared__ float sm[];
    const int* qt = (const int*)(sm + QTAB_OFF);
    const int e31 = qt[32], Tb = (e31 - 1) / KQ;
    if (e31 < (Tb + 1) * KQ) {   // the block's last tile continues in the next block
      auto busy = [&](int j) { const int a = qt[j], b = qt[j + 1], t = a / KQ;
                               return a > t * KQ && (t + 1) * KQ <= b; };
      const int ex = !busy(30) ? 30 : (!busy(29) ? 29 : 31);
      if (wib == ex) finalize<NGT>(Tb, 1, 31);
    }
  }
}

template <int NGT>
__device__ __noinline__ void tile_done(int qend) {   // a whole tile ended at quad qend: its sums are in slot 1
  const Ctx& cx = CX(); const Op& op = OP();
#ifndef NO_STAMPS   // debug flags 8 / 16 / 24 (timing tools) compiled out with the stamps
  if (cx.flags & 16) return;
#endif
  const int KQ = (int)op.KQ;
  epilogue<NGT>(qend / KQ - 1, ((threadIdx.x >> 5) * 2 + 1) * NGT * 32 + (threadIdx.x & 31), false);
}
template <int NGT, int XSC = 32>   // XSC = 4: the compact one-column activation layout (X_COMPACT, NC = 1)
__device__ __noinline__ void gemm_stream_bf() {
  constexpr int XS = XSC;
  extern __shared__ float sm[];
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  int left, cnt, qe;
  const uint4* wp;
  const uint2* xp;
  {
    const Op& op = OP();
    const int w = blockIdx.x * 32 + wib, KQ = (int)op.KQ;
    const long long UQ = (long long)op.NT * KQ;
    const int qq = qw(UQ, w + (lane & 1), CX().nwarps);   // both bounds in one pass of the division (lanes 0, 1)
    const int q0 = __shfl_sync(0xffffffffu, qq, 0);
    qe = __shfl_sync(0xffffffffu, qq, 1);
    if (lane == 0) { int* qt = (int*)(sm + QTAB_OFF); qt[wib] = q0; if (wib == 31) qt[32] = qe; }   // read after the block sync
    cnt = qe - q0;
    left = KQ - (q0 % KQ);
    wp = (const uint4*)op.W + (size_t)q0 * 128 + lane;
    xp = (const uint2*)op.xf + (XS == 4 ? (lane & 3) : lane) + (KQ - left) * 4 * NGT * XS;
#ifndef NO_STAMPS   // debug flags 8 / 16 / 24 (timing tools) compiled out with the stamps
    if (CX().flags & 8) cnt = 0;   // debug: prologue and barrier only
#endif
  }
  bool first = left != (int)OP().KQ;   // the share starts inside a tile: its first piece goes to slot 0
  const bool xact = XS == 4 || NGT > 1 || (lane >> 2) < CX().NC;
  float a_[NGT][4];
  #pragma unroll
  for (int cg = 0; cg < NGT; ++cg) { a_[cg][0] = 0.f; a_[cg][1] = 0.f; a_[cg][2] = 0.f; a_[cg][3] = 0.f; }
  #pragma unroll 1
  for (; cnt > 0; --cnt) {
    const uint4 a0 = ldw(wp), a1 = ldw(wp + 32), a2 = ldw(wp + 64), a3 = ldw(wp + 96);
    if (NGT <= 2) {
      #pragma unroll
      for (int cg = 0; cg < NGT; ++cg) {
        const uint2 b0 = xld(xp + ((0 * NGT + cg) * XS), xact), b1 = xld(xp + ((1 * NGT + cg) * XS), xact), b2 = xld(xp + ((2 * NGT + cg) * XS), xact), b3 = xld(xp + ((3 * NGT + cg) * XS), xact);
        mma(a_[cg], a0, b0); mma(a_[cg], a1, b1); mma(a_[cg], a2, b2); mma(a_[cg], a3, b3);
      }
    } else {
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint4 a = ks == 0 ? a0 : ks == 1 ? a1 : ks == 2 ? a2 : a3;
        uint2 b[NGT];
        #pragma unroll
        for (int cg = 0; cg < NGT; ++cg) {
#ifndef X4_L1
          b[cg] = ldx(xp + (ks * NGT + cg) * 32);   // 32 columns: from L2 (L1 hit rate ~8 %): -0.6 % per pass
#else
          b[cg] = xld(xp + ((ks * NGT + cg) * 32), xact);
#endif
        }
        #pragma unroll
        for (int cg = 0; cg < NGT; ++cg) mma(a_[cg], a, b[cg]);
      }
    }
    wp += 128;
    xp += 4 * NGT * XS;
    if (--left == 0) {   // end of a tile
      float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + (first ? 0 : 1)) * NGT * 32 + lane;
      #pragma unroll
      for (int cg = 0; cg < NGT; ++cg) {
        ps[cg * 32] = make_float4(a_[cg][0], a_[cg][1], a_[cg][2], a_[cg][3]);
        a_[cg][0] = 0.f; a_[cg][1] = 0.f; a_[cg][2] = 0.f; a_[cg][3] = 0.f;
      }
      if (!first) tile_done<NGT>(qe - cnt + 1);
      first = false;
      const int KQ = (int)OP().KQ;
      left = KQ;
      xp -= KQ * 4 * NGT * XS;
    }
  }
#ifdef NO_STAMPS
  if (false) {
#else
  if (CX().wts && lane == 0) {
#endif
    const unsigned* oi = (const unsigned*)(sm + RCNT_OFF + 1);
    if ((int)*oi == CX().wts_op) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); CX().wts[blockIdx.x * 32 + wib] = t; }
  }
  if (left != (int)OP().KQ) {   // the share ends inside a tile: the last piece goes to slot 1
    float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + 1) * NGT * 32 + lane;
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) ps[cg * 32] = make_float4(a_[cg][0], a_[cg][1], a_[cg][2], a_[cg][3]);
  }
  // this warp's finalize role, decided at its own stream end (while other warps still stream) instead of by all
  // 32 warps at once after the block sync: the tile whose last piece it holds (finalize_all's first test), and for
  // warp 31 the block's outgoing cut tile
  __syncwarp();
  {
    const int KQ = (int)OP().KQ, a = ((volatile const int*)(sm + QTAB_OFF))[wib], t = a / KQ;
    int* rt = (int*)(sm + ROLE_OFF);
    if (lane == 0) rt[wib] = (a > t * KQ && (t + 1) * KQ <= qe) ? t + 1 : 0;
    if (wib == 31 && lane == 0) { const int Tb = (qe - 1) / KQ; rt[32] = qe < (Tb + 1) * KQ ? Tb + 1 : 0; }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");   // the residual rows (EP_RESID) copied while streaming
  __syncthreads();
  dbg_stamp(0);
#ifndef NO_STAMPS   // debug flags 8 / 16 / 24 (timing tools) compiled out with the stamps
  if (CX().flags & 24) return;   // debug: 8 = prologue and barrier only, 16 = stream only
#endif
  {
    const int* rt = (const int*)(sm + ROLE_OFF);
    const int ra = rt[wib];
#ifdef HO_STAMPS
    if (ra) dbg_stampw(8);
#endif
    if (ra) finalize<NGT>(ra - 1, 0, wib);
    if (wib >= 29) {
      const int rb = rt[32];
      if (rb) {   // the outgoing tile: warp 30 or 29 when it has no tile of its own to finish, else warp 31
        const int ex = !rt[30] ? 30 : (!rt[29] ? 29 : 31);
        if (wib == ex) finalize<NGT>(rb - 1, 1, 31);
      }
    }
  }
  if (wib == 0 || wib == 31) { __syncwarp(); }
  if (threadIdx.x == 0 || threadIdx.x == 992) dbg_stamp(1);   // a lower bound on the block's finalize end (warps 0, 31)
}

// W12 with the out-of-window weights as an exact sparse correction (track B). The hot loop is the BF16 loop with 3
// loads instead of 4 and the integer decode: no per-quad test, no patch state. Each output row has its own exponent
// window (api.w12_window_rows; decode base C_row, two per lane: rows g and g + 8 of the tile). An out-of-window weight
// carries code 0, which decodes to v0 = 2^(lo_row - 1); at the end of each piece of a tile the warp adds w x - v0 x
// for every such weight of the piece to its accumulators: two exact products per column, so the result equals
// sum_k W[r][k] x[k] in exact arithmetic; only the rounding order of the fp32 sum differs from the BF16 path.
// Per-warp lists: p6 = header per warp (start | min(n + 1, 32) << 20), p7 = entries (position, bf16 bits), p8 = C_row.
#ifdef W12_PRMT
// Byte layout (api.W12_FMT "prmt"): per lane and k-step u0 = [sign|mantissa] bytes of weights 0-3, u1 = of weights 4-7,
// u2 = 8 exponent codes (weight k at bit 4 (k >> 1) + 16 (k & 1)). One byte permute with sign replication puts sign
// (bit 15) and mantissa (bits 0-6) of two weights in place; the exponent codes are shifted into bits 7-10 and the
// row's base added: 5 instructions per register instead of ~6-7 for the packed codes.
__device__ __forceinline__ u32 prmt_(u32 a, u32 sel) { u32 r; asm("prmt.b32 %0, %1, 0, %2;" : "=r"(r) : "r"(a), "r"(sel)); return r; }
__device__ __forceinline__ uint4 dec8r(u32 u0, u32 u1, u32 u2, u32 Ca, u32 Cb) {   // Ca: row g, Cb: row g + 8
  const u32 M = 0x807F807Fu, EM = 0x07800780u;
  return make_uint4((prmt_(u0, 0x9180u) & M) + ((u2 << 7) & EM) + Ca, (prmt_(u0, 0xB3A2u) & M) + ((u2 << 3) & EM) + Cb,
                    (prmt_(u1, 0x9180u) & M) + ((u2 >> 1) & EM) + Ca, (prmt_(u1, 0xB3A2u) & M) + ((u2 >> 5) & EM) + Cb);
}
#else
__device__ __forceinline__ uint4 dec8r(u32 u0, u32 u1, u32 u2, u32 Ca, u32 Cb) {   // Ca: row g, Cb: row g + 8
  return make_uint4(dec2(u0, Ca), dec2((u0 >> 24) | (u1 << 8), Cb), dec2((u1 >> 16) | (u2 << 16), Ca), dec2(u2 >> 8, Cb));
}
#endif
// Corrections of the entries of this piece (quads of the share < pb; kq_done: quads of the tile streamed so far).
// Lane-parallel: each lane takes one entry and loads its x values (all columns < NC at once); then the entries are
// folded into the owners' accumulators in list order (shuffles only), so the order is fixed: deterministic and the
// same for a column alone or in a batch.
__device__ __forceinline__ void w12_esc(float* a, uint2& eL, int& ek, const uint2* ep, int pb, int kq_done, u32 Ca, u32 Cb) {
#ifdef W12_NOESC
  return;   // timing experiment only: wrong results
#endif
  const Op& op = OP();
  const int lane = threadIdx.x & 31, t = lane & 3, NC = CX().NC;
  const u16* xb = (const u16*)op.xf;
  const float v0a = bf2f((u16)(Ca & 0xFFFFu)), v0b = bf2f((u16)(Cb & 0xFFFFu));
  #pragma unroll 1
  while (true) {
    const int base = ek & 31;
    const u32 epos = eL.x;
    // this piece's entries: the run of lanes from base on that are neither a sentinel nor past the piece (after a
    // sentinel the window may hold the next warp's list)
    const unsigned ge = 0xffffffffu << base;
    const unsigned ok = __ballot_sync(0xffffffffu, epos != 0xFFFFFFFFu && (int)(epos >> 10) < pb) & ge;
    const unsigned stop = ~ok & ge;
    const unsigned m = stop ? ok & ((1u << (__ffs(stop) - 1)) - 1u) : ok;
    if (!m) break;
    const bool mine = (m >> lane) & 1u;
    int row = 0;
    u32 X0 = 0, X1 = 0, X2 = 0, X3 = 0;   // bf16 pairs: x[k][2i], x[k][2i + 1]
    if (mine) {
      const int ls = (epos >> 5) & 31, ks = (epos >> 3) & 3, slot = epos & 7, reg = slot >> 1;
      row = (ls >> 2) + 8 * (reg & 1);
      const int k = (kq_done - (pb - (int)(epos >> 10))) * 64 + ks * 16 + 2 * (ls & 3) + (slot & 1) + 8 * (reg >> 1);
      const u16* xk = xb + ((size_t)((k >> 4) * 32 + ((k & 7) >> 1))) * 4 + ((k >> 3) & 1) * 2 + (k & 1);   // column c at xk[16 c]
      X0 = (u32)xk[0] | (NC > 1 ? (u32)xk[16] << 16 : 0u);
      if (NC > 2) { X1 = (u32)xk[32] | ((u32)xk[48] << 16); X2 = (u32)xk[64] | ((u32)xk[80] << 16); X3 = (u32)xk[96] | ((u32)xk[112] << 16); }
    }
    unsigned mm = m;
    while (mm) {
      const int sl = __ffs(mm) - 1;
      mm &= mm - 1;
      const int rs = __shfl_sync(0xffffffffu, row, sl);
      const u32 ws = __shfl_sync(0xffffffffu, eL.y, sl);
      u32 xs = __shfl_sync(0xffffffffu, X0, sl);
      if (NC > 2) {
        const u32 y1 = __shfl_sync(0xffffffffu, X1, sl), y2 = __shfl_sync(0xffffffffu, X2, sl), y3 = __shfl_sync(0xffffffffu, X3, sl);
        xs = t == 0 ? xs : t == 1 ? y1 : t == 2 ? y2 : y3;
      }
      if ((lane >> 2) == (rs & 7)) {
        const float w = bf2f((u16)ws), x0 = __uint_as_float(xs << 16), x1 = __uint_as_float(xs & 0xFFFF0000u);
        if (rs < 8) {   // constant indices only: a dynamic index would put the accumulators in local memory
          a[0] = __fmaf_rn(-v0a, x0, __fmaf_rn(w, x0, a[0]));
          a[1] = __fmaf_rn(-v0a, x1, __fmaf_rn(w, x1, a[1]));
        } else {
          a[2] = __fmaf_rn(-v0b, x0, __fmaf_rn(w, x0, a[2]));
          a[3] = __fmaf_rn(-v0b, x1, __fmaf_rn(w, x1, a[3]));
        }
      }
    }
    ek += __popc(m);
    if (ek & 31) break;                 // the piece's entries ended inside the window
    eL = ep[ek + lane];                 // the window was used up (lists longer than 32): the next 32 entries
  }
}
template <int NGT>
__device__ __noinline__ void gemm_stream_w12c() {
  static_assert(NGT == 1, "W12c: one column group");
  extern __shared__ float sm[];
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  int left, cnt, qe, hdr, T;
  const uint4* wp;
  const uint2* xp;
  {
    const Op& op = OP();
    const int w = blockIdx.x * 32 + wib, KQ = (int)op.KQ;
    const long long UQ = (long long)op.NT * KQ;
    hdr = ((const int*)op.p6)[w];                         // the warp's escape list; read after the first quad's loads
    const int qq = qw(UQ, w + (lane & 1), CX().nwarps);
    const int q0 = __shfl_sync(0xffffffffu, qq, 0);
    qe = __shfl_sync(0xffffffffu, qq, 1);
    if (lane == 0) { int* qt = (int*)(sm + QTAB_OFF); qt[wib] = q0; if (wib == 31) qt[32] = qe; }
    cnt = qe - q0;
    T = q0 / KQ;
    left = KQ - (q0 - T * KQ);
    wp = (const uint4*)op.W + (size_t)q0 * 96 + lane;
    xp = (const uint2*)op.xf + lane + (KQ - left) * 4 * 32;
#ifndef NO_STAMPS
    if (CX().flags & 8) cnt = 0;
#endif
  }
  const u32* ct = (const u32*)OP().p8 + (lane >> 2);
#ifdef W12_DBG_CONSTC   // timing experiment only: wrong results
  u32 Ca = 0x36803680u, Cb = Ca, Na = Ca, Nb = Ca;
#else
  u32 Ca = ct[T * 16], Cb = ct[T * 16 + 8];                // this tile's rows g, g + 8
  u32 Na = ct[T * 16 + 16], Nb = ct[T * 16 + 24];          // the next tile's (one past the last tile: the table is padded)
#endif
  const uint2* ep = (const uint2*)OP().p7 + (hdr & 0xFFFFF);
  const int cnt0 = cnt;
  uint2 eL = make_uint2(0xFFFFFFFFu, 0u);
  int ek = 0;
  bool first = left != (int)OP().KQ;
  float a_[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll 1
  while (cnt > 0) {
#ifdef W12_PAIR   // two quads per iteration: spills at 64 registers, slower (hour 1.6)
    if (left >= 2 && cnt >= 2) {   // two quads per iteration: 96 bytes per thread in flight
      const uint4 p0 = ldw(wp), p1 = ldw(wp + 32), p2 = ldw(wp + 64), s0 = ldw(wp + 96), s1 = ldw(wp + 128), s2 = ldw(wp + 160);
      if (cnt == cnt0 && lane < (hdr >> 20)) eL = ep[lane];          // after the first quads' loads are in flight
      uint4 r = dec8r(p0.x, p0.y, p0.z, Ca, Cb); mma(a_, r, xp[0]);
      r = dec8r(p0.w, p1.x, p1.y, Ca, Cb); mma(a_, r, xp[32]);
      r = dec8r(p1.z, p1.w, p2.x, Ca, Cb); mma(a_, r, xp[64]);
      r = dec8r(p2.y, p2.z, p2.w, Ca, Cb); mma(a_, r, xp[96]);
      r = dec8r(s0.x, s0.y, s0.z, Ca, Cb); mma(a_, r, xp[128]);
      r = dec8r(s0.w, s1.x, s1.y, Ca, Cb); mma(a_, r, xp[160]);
      r = dec8r(s1.z, s1.w, s2.x, Ca, Cb); mma(a_, r, xp[192]);
      r = dec8r(s2.y, s2.z, s2.w, Ca, Cb); mma(a_, r, xp[224]);
      wp += 192; xp += 8 * 32; cnt -= 2; left -= 2;
    } else
#endif
    {
      const uint4 p0 = ldw(wp), p1 = ldw(wp + 32), p2 = ldw(wp + 64);   // 48 bytes: 32 weights of this thread
#ifndef W12_DBG_NOEL
      if (cnt == cnt0 && lane < (hdr >> 20)) eL = ep[lane];            // after the first quad's loads are in flight
#endif
      uint4 r = dec8r(p0.x, p0.y, p0.z, Ca, Cb); mma(a_, r, xp[0]);
      r = dec8r(p0.w, p1.x, p1.y, Ca, Cb); mma(a_, r, xp[32]);
      r = dec8r(p1.z, p1.w, p2.x, Ca, Cb); mma(a_, r, xp[64]);
      r = dec8r(p2.y, p2.z, p2.w, Ca, Cb); mma(a_, r, xp[96]);
      wp += 96; xp += 4 * 32; cnt -= 1; left -= 1;
    }
    if (left == 0) {   // end of a tile (cnt: quads left after it)
      const int KQ = (int)OP().KQ;
      w12_esc(a_, eL, ek, ep, cnt0 - cnt, KQ, Ca, Cb);
      float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + (first ? 0 : 1)) * 32 + lane;
      ps[0] = make_float4(a_[0], a_[1], a_[2], a_[3]);
      a_[0] = 0.f; a_[1] = 0.f; a_[2] = 0.f; a_[3] = 0.f;
      if (!first) tile_done<NGT>(qe - cnt);
      first = false;
      left = KQ;
      xp -= KQ * 4 * 32;
      ++T;
      Ca = Na; Cb = Nb;
      Na = ct[T * 16 + 16]; Nb = ct[T * 16 + 24];   // used only by shares longer than a tile (lm_head); the table is padded
    }
  }
  if (left != (int)OP().KQ) {   // the share ends inside a tile: the last piece goes to slot 1
    w12_esc(a_, eL, ek, ep, cnt0, (int)OP().KQ - left, Ca, Cb);
    float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + 1) * 32 + lane;
    ps[0] = make_float4(a_[0], a_[1], a_[2], a_[3]);
  }
  __syncwarp();
  {
    const int KQ = (int)OP().KQ, a = ((volatile const int*)(sm + QTAB_OFF))[wib], t = a / KQ;
    int* rt = (int*)(sm + ROLE_OFF);
    if (lane == 0) rt[wib] = (a > t * KQ && (t + 1) * KQ <= qe) ? t + 1 : 0;
    if (wib == 31 && lane == 0) { const int Tb = (qe - 1) / KQ; rt[32] = qe < (Tb + 1) * KQ ? Tb + 1 : 0; }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  dbg_stamp(0);
#ifndef NO_STAMPS
  if (CX().flags & 24) return;
#endif
  {
    const int* rt = (const int*)(sm + ROLE_OFF);
    const int ra = rt[wib];
    if (ra) finalize<NGT>(ra - 1, 0, wib);
    if (wib >= 29) {
      const int rb = rt[32];
      if (rb) {
        const int ex = !rt[30] ? 30 : (!rt[29] ? 29 : 31);
        if (wib == ex) finalize<NGT>(rb - 1, 1, 31);
      }
    }
  }
  if (wib == 0 || wib == 31) { __syncwarp(); }
  if (threadIdx.x == 0 || threadIdx.x == 992) dbg_stamp(1);
}

// The same W12 stream with lean loop state (so that two quads per iteration fit in 64 registers without spills):
// what is needed only at tile ends lives in shared memory, per warp 72 words at W12S_OFF: the next tile's decode
// bases per lane [0, 64) (copied there with cp.async, no register held), the list header [64], the list position
// [65], the share's first quad and end [66], [67].
#define W12S_OFF (WORK_OFF + RPART_FLOATS(1) + 1024)   // floats; api.py adds 72 * 32 * 4 bytes of shared memory for W12
__device__ __forceinline__ void cp_async4(u32* dst, const u32* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void w12_esc_s(float* a, uint2& eL, u32* ws, int pb, int kq_done, u32 Ca, u32 Cb) {
  int ek = (int)ws[65];
  const uint2* ep = (const uint2*)OP().p7 + (ws[64] & 0xFFFFFu);
  w12_esc(a, eL, ek, ep, pb, kq_done, Ca, Cb);
  __syncwarp();
  if ((threadIdx.x & 31) == 0) ws[65] = (u32)ek;
}
template <int NGT>
__device__ __noinline__ void gemm_stream_w12d() {
  static_assert(NGT == 1, "W12d: one column group");
  extern __shared__ float sm[];
  const int wib = threadIdx.x >> 5;
  u32* ws = (u32*)(sm + W12S_OFF) + wib * 72;
  int left, cnt, hdr;
  const uint4* wp;
  const uint2* xp;
  u32 Ca, Cb;
  {
    const int lane = threadIdx.x & 31;
    const Op& op = OP();
    const int w = blockIdx.x * 32 + wib, KQ = (int)op.KQ;
    const long long UQ = (long long)op.NT * KQ;
    hdr = ((const int*)op.p6)[w];                         // the warp's escape list; read after the first quad's loads
    const int qq = qw(UQ, w + (lane & 1), CX().nwarps);
    const int q0 = __shfl_sync(0xffffffffu, qq, 0);
    const int qe = __shfl_sync(0xffffffffu, qq, 1);
    if (lane == 0) { int* qt = (int*)(sm + QTAB_OFF); qt[wib] = q0; if (wib == 31) qt[32] = qe; ws[65] = 0; ws[66] = q0; ws[67] = qe; }
    cnt = qe - q0;
    const int T = q0 / KQ;
    left = KQ - (q0 - T * KQ);
    wp = (const uint4*)op.W + (size_t)q0 * 96 + lane;
    xp = (const uint2*)op.xf + lane + (KQ - left) * 4 * 32;
    const u32* ct = (const u32*)op.p8 + T * 16 + (lane >> 2);
    Ca = ct[0]; Cb = ct[8];                               // this tile's rows g, g + 8
    cp_async4(ws + lane, ct + 16); cp_async4(ws + 32 + lane, ct + 24);   // the next tile's (the table is padded)
    asm volatile("cp.async.commit_group;" ::: "memory");
#ifndef NO_STAMPS
    if (CX().flags & 8) cnt = 0;
#endif
  }
  uint2 eL = make_uint2(0xFFFFFFFFu, 0u);
  bool fresh = true;
  float a_[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll 1
  while (cnt > 0) {
#ifndef W12D_SINGLE
    if (left >= 2 && cnt >= 2) {   // two quads per iteration: 96 bytes per thread in flight
      const uint4 p0 = ldw(wp), p1 = ldw(wp + 32), p2 = ldw(wp + 64), s0 = ldw(wp + 96), s1 = ldw(wp + 128), s2 = ldw(wp + 160);
      if (fresh) {   // after the first quads' loads are in flight
        fresh = false;
        const int lane = threadIdx.x & 31;
        if (lane < (hdr >> 20)) eL = ((const uint2*)OP().p7 + (hdr & 0xFFFFF))[lane];
        if (lane == 0) ws[64] = (u32)hdr;
      }
      uint4 r = dec8r(p0.x, p0.y, p0.z, Ca, Cb); mma(a_, r, xp[0]);
      r = dec8r(p0.w, p1.x, p1.y, Ca, Cb); mma(a_, r, xp[32]);
      r = dec8r(p1.z, p1.w, p2.x, Ca, Cb); mma(a_, r, xp[64]);
      r = dec8r(p2.y, p2.z, p2.w, Ca, Cb); mma(a_, r, xp[96]);
      r = dec8r(s0.x, s0.y, s0.z, Ca, Cb); mma(a_, r, xp[128]);
      r = dec8r(s0.w, s1.x, s1.y, Ca, Cb); mma(a_, r, xp[160]);
      r = dec8r(s1.z, s1.w, s2.x, Ca, Cb); mma(a_, r, xp[192]);
      r = dec8r(s2.y, s2.z, s2.w, Ca, Cb); mma(a_, r, xp[224]);
      wp += 192; xp += 8 * 32; cnt -= 2; left -= 2;
    } else
#endif
    {
      const uint4 p0 = ldw(wp), p1 = ldw(wp + 32), p2 = ldw(wp + 64);
      if (fresh) {
        fresh = false;
        const int lane = threadIdx.x & 31;
        if (lane < (hdr >> 20)) eL = ((const uint2*)OP().p7 + (hdr & 0xFFFFF))[lane];
        if (lane == 0) ws[64] = (u32)hdr;
      }
      uint4 r = dec8r(p0.x, p0.y, p0.z, Ca, Cb); mma(a_, r, xp[0]);
      r = dec8r(p0.w, p1.x, p1.y, Ca, Cb); mma(a_, r, xp[32]);
      r = dec8r(p1.z, p1.w, p2.x, Ca, Cb); mma(a_, r, xp[64]);
      r = dec8r(p2.y, p2.z, p2.w, Ca, Cb); mma(a_, r, xp[96]);
      wp += 96; xp += 4 * 32; cnt -= 1; left -= 1;
    }
    if (left == 0) {   // end of a tile (cnt: quads left after it)
      __syncwarp();
      const int KQ = (int)OP().KQ, q0 = (int)ws[66], qe = (int)ws[67], Te = (qe - cnt - 1) / KQ;
      const bool first = q0 > Te * KQ;   // the share started inside this tile: its piece goes to slot 0
      w12_esc_s(a_, eL, ws, qe - q0 - cnt, KQ, Ca, Cb);
      const int lane = threadIdx.x & 31;
      float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + (first ? 0 : 1)) * 32 + lane;
      ps[0] = make_float4(a_[0], a_[1], a_[2], a_[3]);
      a_[0] = 0.f; a_[1] = 0.f; a_[2] = 0.f; a_[3] = 0.f;
      if (!first) tile_done<NGT>(qe - cnt);
      left = KQ;
      xp -= KQ * 4 * 32;
      asm volatile("cp.async.wait_all;" ::: "memory");
      Ca = ws[lane]; Cb = ws[32 + lane];
      __syncwarp();
      const u32* ct = (const u32*)OP().p8 + (Te + 2) * 16 + (lane >> 2);   // the tile after the next (lm_head shares)
      cp_async4(ws + lane, ct); cp_async4(ws + 32 + lane, ct + 8);
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
  }
  if (left != (int)OP().KQ) {   // the share ends inside a tile: the last piece goes to slot 1
    __syncwarp();
    const int lane = threadIdx.x & 31;
    w12_esc_s(a_, eL, ws, (int)(ws[67] - ws[66]), (int)OP().KQ - left, Ca, Cb);
    float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + 1) * 32 + lane;
    ps[0] = make_float4(a_[0], a_[1], a_[2], a_[3]);
  }
  __syncwarp();
  {
    const int lane = threadIdx.x & 31;
    const int KQ = (int)OP().KQ, a = (int)ws[66], t = a / KQ, qe = (int)ws[67];
    int* rt = (int*)(sm + ROLE_OFF);
    if (lane == 0) rt[wib] = (a > t * KQ && (t + 1) * KQ <= qe) ? t + 1 : 0;
    if (wib == 31 && lane == 0) { const int Tb = (qe - 1) / KQ; rt[32] = qe < (Tb + 1) * KQ ? Tb + 1 : 0; }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  dbg_stamp(0);
#ifndef NO_STAMPS
  if (CX().flags & 24) return;
#endif
  {
    const int* rt = (const int*)(sm + ROLE_OFF);
    const int ra = rt[wib];
    if (ra) finalize<NGT>(ra - 1, 0, wib);
    if (wib >= 29) {
      const int rb = rt[32];
      if (rb) {
        const int ex = !rt[30] ? 30 : (!rt[29] ? 29 : 31);
        if (wib == ex) finalize<NGT>(rb - 1, 1, 31);
      }
    }
  }
  if (wib == 0 || wib == 31) { __syncwarp(); }
  if (threadIdx.x == 0 || threadIdx.x == 992) dbg_stamp(1);
}

// Exact W12 (bit-identical to BF16 storage): every out-of-window weight is patched into the MMA operand register
// before the multiply, so the tensor cores see exactly the checkpoint's bf16 bits. The lists sit at a fixed stride per
// warp (p7 + w * CAP, CAP = (u32)op.i7: no header load before the first quad), entries in stream order: position =
// [quad in the share][k-step (2)][lane (5)][slot (3)]. One shuffle and compare per quad (or pair of quads) finds the
// quads that hold an escape (~15 % of quads in in_proj); only those take the per-k-step patch path.
// The list window: 32 entries in a ring over the lanes (entry i in lane i & 31); when the position ek crosses a
// multiple of 16, the half just used up is refilled with entries ek + 16 .. ek + 31, so a refill is never waited for
// (lists of CAP = 16 never need one: their sentinel is in the first half).
__device__ __forceinline__ void w12_patch1(uint4& r, u32 slot, u32 v) {
  const u32 sh = (slot & 1u) << 4, mk = ~(0xFFFFu << sh), vv = v << sh;
  const u32 reg = slot >> 1;
  if (reg == 0) r.x = (r.x & mk) | vv;
  if (reg == 1) r.y = (r.y & mk) | vv;
  if (reg == 2) r.z = (r.z & mk) | vv;
  if (reg == 3) r.w = (r.w & mk) | vv;
}
// one quad (its 48 bytes loaded) for share-relative quad q: plain when the next escape is elsewhere, else k-step by
// k-step with the escapes patched in. One column group (decode passes).
__device__ __forceinline__ void w12e_quad1(float* a_, const uint4& p0, const uint4& p1, const uint4& p2, const uint2* xq, int q,
                                           u32 Ca, u32 Cb, uint2& eL, int& ek, const uint2* ep, bool xact = true, int xs = 32) {
  const u32 nq = __shfl_sync(0xffffffffu, eL.x, ek & 31) >> 10;
  uint2 b0, b1, b2, b3;
#ifdef X_C2
  if (xs == 4) {
    const uint4 xa = ((const uint4*)xq)[0], xb = ((const uint4*)xq)[4];
    b0 = make_uint2(xa.x, xa.y); b1 = make_uint2(xa.z, xa.w); b2 = make_uint2(xb.x, xb.y); b3 = make_uint2(xb.z, xb.w);
  } else
#endif
  { b0 = xld(xq + 0, xact); b1 = xld(xq + xs, xact); b2 = xld(xq + 2 * xs, xact); b3 = xld(xq + 3 * xs, xact); }
  if (nq != (u32)q) {
    uint4 r = dec8r(p0.x, p0.y, p0.z, Ca, Cb); mma(a_, r, b0);
    r = dec8r(p0.w, p1.x, p1.y, Ca, Cb); mma(a_, r, b1);
    r = dec8r(p1.z, p1.w, p2.x, Ca, Cb); mma(a_, r, b2);
    r = dec8r(p2.y, p2.z, p2.w, Ca, Cb); mma(a_, r, b3);
  } else {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const u32 u0 = ks == 0 ? p0.x : ks == 1 ? p0.w : ks == 2 ? p1.z : p2.y;
      const u32 u1 = ks == 0 ? p0.y : ks == 1 ? p1.x : ks == 2 ? p1.w : p2.z;
      const u32 u2 = ks == 0 ? p0.z : ks == 1 ? p1.y : ks == 2 ? p2.x : p2.w;
      uint4 r = dec8r(u0, u1, u2, Ca, Cb);
      #pragma unroll 1
      while (true) {
        const u32 e = __shfl_sync(0xffffffffu, eL.x, ek & 31);
        if ((e >> 8) != (((u32)q << 2) | (u32)ks)) break;   // not this quad and k-step (sentinel: never)
        const u32 v = __shfl_sync(0xffffffffu, eL.y, ek & 31);
        if (((e >> 3) & 31u) == (u32)lane) w12_patch1(r, e & 7u, v);
        if ((++ek & 15) == 0 && ((ek >> 4) & 1) != (lane >> 4)) eL = ep[ek + 16 + (lane & 15)];   // refill the used half
      }
      mma(a_, r, ks == 0 ? b0 : ks == 1 ? b1 : ks == 2 ? b2 : b3);
    }
  }
}
// the same for NGT column groups (prefill passes: 2 or 4), k-step by k-step always
template <int NGT>
__device__ __forceinline__ void w12e_quadN(float (*a_)[4], const uint4& p0, const uint4& p1, const uint4& p2, const uint2* xq, int q,
                                           u32 Ca, u32 Cb, uint2& eL, int& ek, const uint2* ep) {
  const int lane = threadIdx.x & 31;
  #pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const u32 u0 = ks == 0 ? p0.x : ks == 1 ? p0.w : ks == 2 ? p1.z : p2.y;
    const u32 u1 = ks == 0 ? p0.y : ks == 1 ? p1.x : ks == 2 ? p1.w : p2.z;
    const u32 u2 = ks == 0 ? p0.z : ks == 1 ? p1.y : ks == 2 ? p2.x : p2.w;
    uint4 r = dec8r(u0, u1, u2, Ca, Cb);
    #pragma unroll 1
    while (true) {
      const u32 e = __shfl_sync(0xffffffffu, eL.x, ek & 31);
      if ((e >> 8) != (((u32)q << 2) | (u32)ks)) break;
      const u32 v = __shfl_sync(0xffffffffu, eL.y, ek & 31);
      if (((e >> 3) & 31u) == (u32)lane) w12_patch1(r, e & 7u, v);
      if ((++ek & 15) == 0 && ((ek >> 4) & 1) != (lane >> 4)) eL = ep[ek + 16 + (lane & 15)];
    }
    uint2 bb[NGT];
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) bb[cg] = xq[(ks * NGT + cg) * 32];
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) mma(a_[cg], r, bb[cg]);
  }
}
#define W12S_F(NG) (WORK_OFF + RPART_FLOATS(NG) + 1024)   // floats: the per-warp W12 area for NG column groups
template <int NGT, int XSC = 32>   // XSC: activation stride per k-step in uint2 (4: compact layout at NC = 1)
__device__ __noinline__ void gemm_stream_w12e() {
  extern __shared__ float sm[];
  const int wib = threadIdx.x >> 5;
  u32* ws = (u32*)(sm + W12S_F(NGT)) + wib * 72;
  int tend, cnt, ek = 0;   // cnt: quads left in the share; tend: cnt at the end of the current tile
  constexpr int xs = XSC;
  const uint4* wp;
  const uint2* xp;
  const uint2* ep;
  u32 Ca, Cb;
  uint2 eL;
  {
    const int lane = threadIdx.x & 31;
    const Op& op = OP();
    const int w = blockIdx.x * 32 + wib, KQ = (int)op.KQ;
    const long long UQ = (long long)op.NT * KQ;
    const u32 CAP = (u32)op.i7;
    ep = (const uint2*)op.p7 + (size_t)w * CAP;
    eL = (lane < 16 || CAP > 16) ? ep[lane] : make_uint2(0u, 0u);   // quad field 0: end of list
    const int qq = qw(UQ, w + (lane & 1), CX().nwarps);
    const int q0 = __shfl_sync(0xffffffffu, qq, 0);
    const int qe = __shfl_sync(0xffffffffu, qq, 1);
    if (lane == 0) { int* qt = (int*)(sm + QTAB_OFF); qt[wib] = q0; if (wib == 31) qt[32] = qe; ws[66] = q0; ws[67] = qe; }
    cnt = qe - q0;
    const int T = q0 / KQ;
    const int left = KQ - (q0 - T * KQ);
    tend = cnt - left;
    wp = (const uint4*)op.W + (size_t)q0 * 96 + lane;
#ifdef X_C2
    xp = (const uint2*)op.xf + (xs == 4 ? 2 * (lane & 3) : lane) + (KQ - left) * 4 * NGT * xs;
#else
    xp = (const uint2*)op.xf + (xs == 4 ? (lane & 3) : lane) + (KQ - left) * 4 * NGT * xs;
#endif
    const u32* ct = (const u32*)op.p8 + T * 16 + (lane >> 2);
    Ca = ct[0]; Cb = ct[8];
    cp_async4(ws + lane, ct + 16); cp_async4(ws + 32 + lane, ct + 24);
    asm volatile("cp.async.commit_group;" ::: "memory");
#ifndef NO_STAMPS
    if (CX().flags & 8) cnt = 0;
#endif
  }
  // (activation loads for real columns only, as in gemm_stream_bf: bit-identical but +1.4 % per W12 pass; hour 2.6)
#ifdef W12_XACT   // activation loads for real columns only (B = 1: 8x fewer)
  const bool xact = NGT > 1 || ((threadIdx.x & 31) >> 2) < CX().NC;
#else
  const bool xact = true;
#endif
  float a_[NGT][4];
  #pragma unroll
  for (int cg = 0; cg < NGT; ++cg) { a_[cg][0] = 0.f; a_[cg][1] = 0.f; a_[cg][2] = 0.f; a_[cg][3] = 0.f; }
  #pragma unroll 1
  while (cnt > 0) {
#ifndef W12E_SINGLE
    if (NGT == 1 && cnt - tend >= 2 && cnt >= 2) {   // two quads per iteration (loads issued before the escape test)
      const uint4 p0 = ldw(wp), p1 = ldw(wp + 32), p2 = ldw(wp + 64), s0 = ldw(wp + 96), s1 = ldw(wp + 128), s2 = ldw(wp + 160);
      const u32 nq = __shfl_sync(0xffffffffu, eL.x, ek & 31) >> 10;
      if (nq + 2u <= (u32)cnt) {   // the next escape is beyond both quads (entries are in decreasing quad order)
#ifdef X_C2
       if (xs == 4) {
        const uint4* x4 = (const uint4*)xp;
        const uint4 xa = x4[0], xb = x4[4], xc = x4[8], xd = x4[12];
        uint4 r = dec8r(p0.x, p0.y, p0.z, Ca, Cb); mma(a_[0], r, make_uint2(xa.x, xa.y));
        r = dec8r(p0.w, p1.x, p1.y, Ca, Cb); mma(a_[0], r, make_uint2(xa.z, xa.w));
        r = dec8r(p1.z, p1.w, p2.x, Ca, Cb); mma(a_[0], r, make_uint2(xb.x, xb.y));
        r = dec8r(p2.y, p2.z, p2.w, Ca, Cb); mma(a_[0], r, make_uint2(xb.z, xb.w));
        r = dec8r(s0.x, s0.y, s0.z, Ca, Cb); mma(a_[0], r, make_uint2(xc.x, xc.y));
        r = dec8r(s0.w, s1.x, s1.y, Ca, Cb); mma(a_[0], r, make_uint2(xc.z, xc.w));
        r = dec8r(s1.z, s1.w, s2.x, Ca, Cb); mma(a_[0], r, make_uint2(xd.x, xd.y));
        r = dec8r(s2.y, s2.z, s2.w, Ca, Cb); mma(a_[0], r, make_uint2(xd.z, xd.w));
       } else
#endif
       {
        uint4 r = dec8r(p0.x, p0.y, p0.z, Ca, Cb); mma(a_[0], r, xld(xp + 0, xact));
        r = dec8r(p0.w, p1.x, p1.y, Ca, Cb); mma(a_[0], r, xld(xp + xs, xact));
        r = dec8r(p1.z, p1.w, p2.x, Ca, Cb); mma(a_[0], r, xld(xp + 2 * xs, xact));
        r = dec8r(p2.y, p2.z, p2.w, Ca, Cb); mma(a_[0], r, xld(xp + 3 * xs, xact));
        r = dec8r(s0.x, s0.y, s0.z, Ca, Cb); mma(a_[0], r, xld(xp + 4 * xs, xact));
        r = dec8r(s0.w, s1.x, s1.y, Ca, Cb); mma(a_[0], r, xld(xp + 5 * xs, xact));
        r = dec8r(s1.z, s1.w, s2.x, Ca, Cb); mma(a_[0], r, xld(xp + 6 * xs, xact));
        r = dec8r(s2.y, s2.z, s2.w, Ca, Cb); mma(a_[0], r, xld(xp + 7 * xs, xact));
       }
      } else {
        w12e_quad1(a_[0], p0, p1, p2, xp, cnt, Ca, Cb, eL, ek, ep, xact, xs);
        w12e_quad1(a_[0], s0, s1, s2, xp + 4 * xs, cnt - 1, Ca, Cb, eL, ek, ep, xact, xs);
      }
      wp += 192; xp += 8 * xs; cnt -= 2;
    } else
#endif
    {
      const uint4 p0 = ldw(wp), p1 = ldw(wp + 32), p2 = ldw(wp + 64);
      if (NGT == 1) w12e_quad1(a_[0], p0, p1, p2, xp, cnt, Ca, Cb, eL, ek, ep, xact, xs);
      else w12e_quadN<NGT>(a_, p0, p1, p2, xp, cnt, Ca, Cb, eL, ek, ep);
      wp += 96; xp += 4 * NGT * xs; cnt -= 1;
    }
    if (cnt == tend) {   // end of a tile (cnt: quads left after it)
      __syncwarp();
      const int KQ = (int)OP().KQ, q0 = (int)ws[66], qe = (int)ws[67], Te = (qe - cnt - 1) / KQ;
      const bool first = q0 > Te * KQ;
      const int lane = threadIdx.x & 31;
      float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + (first ? 0 : 1)) * NGT * 32 + lane;
      #pragma unroll
      for (int cg = 0; cg < NGT; ++cg) {
        ps[cg * 32] = make_float4(a_[cg][0], a_[cg][1], a_[cg][2], a_[cg][3]);
        a_[cg][0] = 0.f; a_[cg][1] = 0.f; a_[cg][2] = 0.f; a_[cg][3] = 0.f;
      }
      if (!first) tile_done<NGT>(qe - cnt);
      tend = cnt - KQ;
      xp -= KQ * 4 * NGT * xs;
      asm volatile("cp.async.wait_all;" ::: "memory");
      Ca = ws[lane]; Cb = ws[32 + lane];
      __syncwarp();
      const u32* ct = (const u32*)OP().p8 + (Te + 2) * 16 + (lane >> 2);
      cp_async4(ws + lane, ct); cp_async4(ws + 32 + lane, ct + 8);
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
  }
  if (tend != -(int)OP().KQ) {   // the share ends inside a tile: the last piece goes to slot 1
    const int lane = threadIdx.x & 31;
    float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + 1) * NGT * 32 + lane;
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) ps[cg * 32] = make_float4(a_[cg][0], a_[cg][1], a_[cg][2], a_[cg][3]);
  }
  __syncwarp();
  {
    const int lane = threadIdx.x & 31;
    const int KQ = (int)OP().KQ, a = (int)ws[66], t = a / KQ, qe = (int)ws[67];
    int* rt = (int*)(sm + ROLE_OFF);
    if (lane == 0) rt[wib] = (a > t * KQ && (t + 1) * KQ <= qe) ? t + 1 : 0;
    if (wib == 31 && lane == 0) { const int Tb = (qe - 1) / KQ; rt[32] = qe < (Tb + 1) * KQ ? Tb + 1 : 0; }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  dbg_stamp(0);
#ifndef NO_STAMPS
  if (CX().flags & 24) return;
#endif
  {
    const int* rt = (const int*)(sm + ROLE_OFF);
    const int ra = rt[wib];
    if (ra) finalize<NGT>(ra - 1, 0, wib);
    if (wib >= 29) {
      const int rb = rt[32];
      if (rb) {
        const int ex = !rt[30] ? 30 : (!rt[29] ? 29 : 31);
        if (wib == ex) finalize<NGT>(rb - 1, 1, 31);
      }
    }
  }
  if (wib == 0 || wib == 31) { __syncwarp(); }
  if (threadIdx.x == 0 || threadIdx.x == 992) dbg_stamp(1);
}

// The W12 stream with the same minimal loop state. Fast path: decode one k-step (8 weights of this thread) at a
// time; a quad that holds out-of-window weights (1.9e-4 of the weights, ~2 per warp share) takes a slow path that
// decodes its four k-steps, patches them from the warp's list (walked with shuffles) and then multiplies.
template <int NGT>
__device__ __noinline__ void gemm_stream_w12() {
  extern __shared__ float sm[];
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  int left, cnt, qe, qrel = 0;
  const uint4* wp;
  const uint2* xp;
  const uint2* ep;
  u32 C2;
  {
    const Op& op = OP();
    const int w = blockIdx.x * 32 + wib, KQ = (int)op.KQ;
    const long long UQ = (long long)op.NT * KQ;
    const int q0 = qw(UQ, w, CX().nwarps);
    qe = qw(UQ, w + 1, CX().nwarps);
    if (lane == 0) { int* qt = (int*)(sm + QTAB_OFF); qt[wib] = q0; if (wib == 31) qt[32] = qe; }   // read after the block sync
    cnt = qe - q0;
    left = KQ - (q0 % KQ);
    wp = (const uint4*)op.W + (size_t)q0 * 96 + lane;
    xp = (const uint2*)op.xf + lane + (KQ - left) * 4 * NGT * 32;
    ep = (const uint2*)op.p7 + ((const int*)op.p6)[w];
    C2 = (u32)op.i7;
    if (CX().flags & 8) cnt = 0;
  }
  uint2 eL = ep[lane];
  int ek = 0;
  u32 epos = __shfl_sync(0xffffffffu, eL.x, 0), eval_ = __shfl_sync(0xffffffffu, eL.y, 0);
  bool first = left != (int)OP().KQ;
  float a_[NGT][4];
  #pragma unroll
  for (int cg = 0; cg < NGT; ++cg) { a_[cg][0] = 0.f; a_[cg][1] = 0.f; a_[cg][2] = 0.f; a_[cg][3] = 0.f; }
  #pragma unroll 1
  for (; cnt > 0; --cnt, ++qrel) {
    const uint4 p0 = ldw(wp), p1 = ldw(wp + 32), p2 = ldw(wp + 64);   // 48 bytes: 32 weights of this thread
    if ((epos >> 10) != (u32)qrel) {   // fast path: no out-of-window weight in this quad (the same in every lane)
      #pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const u32 u0 = ks == 0 ? p0.x : ks == 1 ? p0.w : ks == 2 ? p1.z : p2.y;
        const u32 u1 = ks == 0 ? p0.y : ks == 1 ? p1.x : ks == 2 ? p1.w : p2.z;
        const u32 u2 = ks == 0 ? p0.z : ks == 1 ? p1.y : ks == 2 ? p2.x : p2.w;
        const uint4 r = dec8(u0, u1, u2, C2);
        #pragma unroll
        for (int cg = 0; cg < NGT; ++cg) mma(a_[cg], r, xp[(ks * NGT + cg) * 32]);
      }
    } else {
      uint4 r0 = dec8(p0.x, p0.y, p0.z, C2), r1 = dec8(p0.w, p1.x, p1.y, C2);
      uint4 r2 = dec8(p1.z, p1.w, p2.x, C2), r3 = dec8(p2.y, p2.z, p2.w, C2);
      do {
        if (((epos >> 5) & 31u) == (u32)lane) {
          const u32 sh = (epos & 1u) << 4;
          const u32 mk = ~(0xFFFFu << sh), vv = eval_ << sh, reg = (epos >> 1) & 3u, ks = (epos >> 3) & 3u;
          if (ks < 2) { if (ks == 0) w12_patch(r0, reg, mk, vv); else w12_patch(r1, reg, mk, vv); }
          else { if (ks == 2) w12_patch(r2, reg, mk, vv); else w12_patch(r3, reg, mk, vv); }
        }
        if (++ek == 32) { ep += 32; ek = 0; eL = ep[lane]; }
        epos = __shfl_sync(0xffffffffu, eL.x, ek); eval_ = __shfl_sync(0xffffffffu, eL.y, ek);
      } while ((epos >> 10) == (u32)qrel);
      #pragma unroll
      for (int cg = 0; cg < NGT; ++cg) {
        mma(a_[cg], r0, xp[(0 * NGT + cg) * 32]); mma(a_[cg], r1, xp[(1 * NGT + cg) * 32]);
        mma(a_[cg], r2, xp[(2 * NGT + cg) * 32]); mma(a_[cg], r3, xp[(3 * NGT + cg) * 32]);
      }
    }
    wp += 96;
    xp += 4 * NGT * 32;
    if (--left == 0) {
      float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + (first ? 0 : 1)) * NGT * 32 + lane;
      #pragma unroll
      for (int cg = 0; cg < NGT; ++cg) {
        ps[cg * 32] = make_float4(a_[cg][0], a_[cg][1], a_[cg][2], a_[cg][3]);
        a_[cg][0] = 0.f; a_[cg][1] = 0.f; a_[cg][2] = 0.f; a_[cg][3] = 0.f;
      }
      if (!first) tile_done<NGT>(qe - cnt + 1);
      first = false;
      const int KQ = (int)OP().KQ;
      left = KQ;
      xp -= KQ * 4 * NGT * 32;
    }
  }
  if (left != (int)OP().KQ) {
    float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + 1) * NGT * 32 + lane;
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) ps[cg * 32] = make_float4(a_[cg][0], a_[cg][1], a_[cg][2], a_[cg][3]);
  }
  asm volatile("cp.async.wait_all;" ::: "memory");   // residual rows copied while streaming
  __syncthreads();
  const Ctx& cx = CX(); const Op& op = OP();
  if (cx.flags & 24) return;
  const int w = blockIdx.x * 32 + wib, KQ = (int)op.KQ;
  const long long UQ = (long long)op.NT * KQ;
  const int* qt = (const int*)(sm + QTAB_OFF);
  const int q0 = qt[wib], q1 = qt[wib + 1];   // = qw(UQ, w), qw(UQ, w + 1), written by the warps before streaming
  finalize_all<NGT>(q0, q1, KQ);
}

template <int NGT, bool W12>
__device__ __noinline__ void gemm_stream_v1() {   // the s0007 loop (kept for W12: the minimal-state W12 loop measured slower)
  extern __shared__ float sm[];
  const Ctx& cx = CX(); const Op& op = OP();
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  const int w = blockIdx.x * 32 + wib;
  const int* tab = (const int*)op.p4;
  const int KQ = (int)op.KQ;
  const int q0 = tab[w], q1 = tab[w + 1];
  if (lane == 0) { int* qt = (int*)(sm + QTAB_OFF); qt[wib] = q0; if (wib == 31) qt[32] = q1; }
  const int flags = cx.flags;
  float4* ps = (float4*)(sm + WORK_OFF);
  const int own = (wib * 2) * NGT * 32 + lane;   // slot 0: first cut piece; slot 1: scratch for whole tiles, then the last cut piece
  if (!(flags & 8)) {
    int T = q0 / KQ;
    int left = KQ - (q0 - T * KQ);
    bool whole = (left == KQ);
    const uint4* wp = (const uint4*)op.W + (size_t)q0 * (W12 ? 96 : 128) + lane;
    const uint2* xb = (const uint2*)op.xf + lane;
    const uint2* xp = xb + (KQ - left) * 4 * NGT * 32;
    // W12: the next out-of-window weight of this warp's share, [quad in share][lane][k-step][slot] and its bf16 bits
    const u32 C2 = (u32)op.i7;
    // The warp's list is loaded once, one entry per lane (all lists end with an all-ones sentinel), and walked
    // with shuffles: a dependent global load per patch cost ~3 us under full load (bench/lat.cu).
    const uint2* ep = W12 ? (const uint2*)op.p7 + ((const int*)op.p6)[w] : (const uint2*)0;
    uint2 eL = make_uint2(0xFFFFFFFFu, 0);
    int ek = 0;
    u32 epos = 0xFFFFFFFFu, eval_ = 0;
    if (W12 && !(flags & 2048)) {   // debug 2048: no patches
      eL = ep[lane];
      epos = __shfl_sync(0xffffffffu, eL.x, 0); eval_ = __shfl_sync(0xffffffffu, eL.y, 0);
    }
    float a_[NGT][4];
    #pragma unroll
    for (int cg = 0; cg < NGT; ++cg) { a_[cg][0] = 0.f; a_[cg][1] = 0.f; a_[cg][2] = 0.f; a_[cg][3] = 0.f; }
    #pragma unroll 1
    for (int q = q0; q < q1; ++q) {
      if (!W12) {
        const uint4 a0 = ldw(wp), a1 = ldw(wp + 32), a2 = ldw(wp + 64), a3 = ldw(wp + 96);
        if (NGT <= 2) {
          #pragma unroll
          for (int cg = 0; cg < NGT; ++cg) {
            const uint2 b0 = xp[(0 * NGT + cg) * 32], b1 = xp[(1 * NGT + cg) * 32], b2 = xp[(2 * NGT + cg) * 32], b3 = xp[(3 * NGT + cg) * 32];
            mma(a_[cg], a0, b0); mma(a_[cg], a1, b1); mma(a_[cg], a2, b2); mma(a_[cg], a3, b3);
          }
        } else {   // 32 columns: k-step by k-step, so that at most one k-step of activations is live (fewer spills)
          #pragma unroll
          for (int ks = 0; ks < 4; ++ks) {
            const uint4 a = ks == 0 ? a0 : ks == 1 ? a1 : ks == 2 ? a2 : a3;
            uint2 b[NGT];
            #pragma unroll
            for (int cg = 0; cg < NGT; ++cg) b[cg] = xp[(ks * NGT + cg) * 32];
            #pragma unroll
            for (int cg = 0; cg < NGT; ++cg) mma(a_[cg], a, b[cg]);
          }
        }
        wp += 128;
      } else {
        const uint4 p0 = ldw(wp), p1 = ldw(wp + 32), p2 = ldw(wp + 64);   // 48 bytes: 32 weights of this thread
        uint2 xb_[NGT <= 2 ? 4 * NGT : 1];
        if (NGT <= 2) {
          #pragma unroll
          for (int i = 0; i < 4 * NGT; ++i) xb_[i] = xp[i * 32];
        }
        uint4 r0 = dec8(p0.x, p0.y, p0.z, C2), r1 = dec8(p0.w, p1.x, p1.y, C2);
        uint4 r2 = dec8(p1.z, p1.w, p2.x, C2), r3 = dec8(p2.y, p2.z, p2.w, C2);
        if ((epos >> 10) == (u32)(q - q0)) {            // rare: this quad has out-of-window weights (same test in every lane)
          do {
            if (((epos >> 5) & 31u) == (u32)lane) {
              const u32 sh = (epos & 1u) << 4;
              const u32 mk = ~(0xFFFFu << sh), vv = eval_ << sh, reg = (epos >> 1) & 3u, ks = (epos >> 3) & 3u;
              if (ks < 2) { if (ks == 0) w12_patch(r0, reg, mk, vv); else w12_patch(r1, reg, mk, vv); }
              else { if (ks == 2) w12_patch(r2, reg, mk, vv); else w12_patch(r3, reg, mk, vv); }
            }
            if (++ek == 32) { ep += 32; ek = 0; eL = ep[lane]; }   // a list longer than 32 (rare): the next 32
            epos = __shfl_sync(0xffffffffu, eL.x, ek); eval_ = __shfl_sync(0xffffffffu, eL.y, ek);
          } while ((epos >> 10) == (u32)(q - q0));
        }
        #pragma unroll
        for (int cg = 0; cg < NGT; ++cg) {
          if (NGT <= 2) {
            mma(a_[cg], r0, xb_[0 * NGT + cg]); mma(a_[cg], r1, xb_[1 * NGT + cg]);
            mma(a_[cg], r2, xb_[2 * NGT + cg]); mma(a_[cg], r3, xb_[3 * NGT + cg]);
          } else {
            mma(a_[cg], r0, xp[(0 * NGT + cg) * 32]); mma(a_[cg], r1, xp[(1 * NGT + cg) * 32]);
            mma(a_[cg], r2, xp[(2 * NGT + cg) * 32]); mma(a_[cg], r3, xp[(3 * NGT + cg) * 32]);
          }
        }
        wp += 96;
      }
      xp += 4 * NGT * 32;
      if (--left == 0) {   // end of a tile
        const int sl = whole ? 1 : 0;
        #pragma unroll
        for (int cg = 0; cg < NGT; ++cg) {
          ps[own + (sl * NGT + cg) * 32] = make_float4(a_[cg][0], a_[cg][1], a_[cg][2], a_[cg][3]);
          a_[cg][0] = 0.f; a_[cg][1] = 0.f; a_[cg][2] = 0.f; a_[cg][3] = 0.f;
        }
        if (whole && !(flags & 16)) epilogue<NGT>(T, own + 1 * NGT * 32, false);
        ++T; left = KQ; whole = true; xp = xb;
      }
    }
    if (left != KQ) {     // the share ends inside a tile
      #pragma unroll
      for (int cg = 0; cg < NGT; ++cg) ps[own + (1 * NGT + cg) * 32] = make_float4(a_[cg][0], a_[cg][1], a_[cg][2], a_[cg][3]);
    }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");   // residual rows copied while streaming
  __syncthreads();
  if (flags & 24) return;   // debug: 8 = prologue and barrier only, 16 = stream only
  finalize_all<NGT>(q0, q1, KQ);
}

#ifdef W12_RING   // STALE experiment (not built): written for the 8-column layout; api.py no longer sizes its memory. -0.2 % at best, see journal
// Exact W12 at one real column (B = 1 decode), weights staged through shared memory: each lane copies its own 48
// bytes of quad q + 1 with cp.async (no registers held) while it decodes quad q, so two quads per warp are in flight
// at all times (bench/w12b.cu mode 12: -6 % per stream in isolation). Activations: only column 0 is loaded (lanes
// 0-3), so the activations' L1 footprint is 1/8 and the 96 KB ring costs less L1. Ring after the W12 area.
#define W12R_F (W12S_F(1) + 72 * 32)   // floats
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void w12r_quad(float* a_, const uint4& p0, const uint4& p1, const uint4& p2, const uint2* xq, int q,
                                          u32 Ca, u32 Cb, uint2& eL, int& ek, const uint2* ep, bool xact) {
  const uint2 b0 = xld(xq, xact), b1 = xld(xq + 32, xact), b2 = xld(xq + 64, xact), b3 = xld(xq + 96, xact);
  const u32 nq = __shfl_sync(0xffffffffu, eL.x, ek & 31) >> 10;
  const int lane = threadIdx.x & 31;
  #pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const u32 u0 = ks == 0 ? p0.x : ks == 1 ? p0.w : ks == 2 ? p1.z : p2.y;
    const u32 u1 = ks == 0 ? p0.y : ks == 1 ? p1.x : ks == 2 ? p1.w : p2.z;
    const u32 u2 = ks == 0 ? p0.z : ks == 1 ? p1.y : ks == 2 ? p2.x : p2.w;
    uint4 r = dec8r(u0, u1, u2, Ca, Cb);
    if (nq == (u32)q) {
      #pragma unroll 1
      while (true) {
        const u32 e = __shfl_sync(0xffffffffu, eL.x, ek & 31);
        if ((e >> 8) != (((u32)q << 2) | (u32)ks)) break;
        const u32 v = __shfl_sync(0xffffffffu, eL.y, ek & 31);
        if (((e >> 3) & 31u) == (u32)lane) w12_patch1(r, e & 7u, v);
        if ((++ek & 15) == 0 && ((ek >> 4) & 1) != (lane >> 4)) eL = ep[ek + 16 + (lane & 15)];
      }
    }
    mma(a_, r, ks == 0 ? b0 : ks == 1 ? b1 : ks == 2 ? b2 : b3);
  }
}
__device__ __noinline__ void gemm_stream_w12r() {
  constexpr int NGT = 1;
  extern __shared__ float sm[];
  const int wib = threadIdx.x >> 5;
  u32* ws = (u32*)(sm + W12S_F(1)) + wib * 72;
  uint4* ring = (uint4*)(sm + W12R_F) + threadIdx.x;   // stage st, plane j at ring[(st * 3 + j) * 1024]
  int left, cnt, qrel = 0, ek = 0;
  const uint4* wp;
  const uint2* xp;
  const uint2* ep;
  u32 Ca, Cb;
  uint2 eL;
  {
    const int lane = threadIdx.x & 31;
    const Op& op = OP();
    const int w = blockIdx.x * 32 + wib, KQ = (int)op.KQ;
    const long long UQ = (long long)op.NT * KQ;
    const u32 CAP = (u32)op.i7;
    ep = (const uint2*)op.p7 + (size_t)w * CAP;
    const int qq = qw(UQ, w + (lane & 1), CX().nwarps);
    const int q0 = __shfl_sync(0xffffffffu, qq, 0);
    const int qe = __shfl_sync(0xffffffffu, qq, 1);
    wp = (const uint4*)op.W + (size_t)q0 * 96 + lane;
    cnt = qe - q0;
#ifndef NO_STAMPS
    if (CX().flags & 8) cnt = 0;
#endif
    if (cnt > 0) { cp_async16(ring, wp); cp_async16(ring + 1024, wp + 32); cp_async16(ring + 2048, wp + 64); }   // quad 0
    asm volatile("cp.async.commit_group;" ::: "memory");
    eL = (lane < 16 || CAP > 16) ? ep[lane] : make_uint2(0xFFFFFFFFu, 0u);
    if (lane == 0) { int* qt = (int*)(sm + QTAB_OFF); qt[wib] = q0; if (wib == 31) qt[32] = qe; ws[66] = q0; ws[67] = qe; }
    const int T = q0 / KQ;
    left = KQ - (q0 - T * KQ);
    xp = (const uint2*)op.xf + lane + (KQ - left) * 4 * 32;
    const u32* ct = (const u32*)op.p8 + T * 16 + (lane >> 2);
    Ca = ct[0]; Cb = ct[8];
    cp_async4(ws + lane, ct + 16); cp_async4(ws + 32 + lane, ct + 24);
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  const bool xact = ((threadIdx.x & 31) >> 2) < CX().NC;
  float a_[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll 1
  while (cnt > 0) {
    const int st = qrel & 1;
    wp += 96;
    if (cnt > 1) { uint4* nx = ring + (st ^ 1) * 3072; cp_async16(nx, wp); cp_async16(nx + 1024, wp + 32); cp_async16(nx + 2048, wp + 64); }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 1;" ::: "memory");   // quad q (and everything older) has landed
    const uint4* cu = ring + st * 3072;
    const uint4 p0 = cu[0], p1 = cu[1024], p2 = cu[2048];
    w12r_quad(a_, p0, p1, p2, xp, qrel, Ca, Cb, eL, ek, ep, xact);
    xp += 4 * 32; cnt -= 1; left -= 1; qrel += 1;
    if (left == 0) {   // end of a tile (cnt: quads left after it)
      __syncwarp();
      const int KQ = (int)OP().KQ, q0 = (int)ws[66], qe = (int)ws[67], Te = (qe - cnt - 1) / KQ;
      const bool first = q0 > Te * KQ;
      const int lane = threadIdx.x & 31;
      float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + (first ? 0 : 1)) * 32 + lane;
      ps[0] = make_float4(a_[0], a_[1], a_[2], a_[3]);
      a_[0] = 0.f; a_[1] = 0.f; a_[2] = 0.f; a_[3] = 0.f;
      if (!first) tile_done<NGT>(qe - cnt);
      left = KQ;
      xp -= KQ * 4 * 32;
      asm volatile("cp.async.wait_all;" ::: "memory");   // the next bases (and the next quad: one stall per tile end)
      Ca = ws[lane]; Cb = ws[32 + lane];
      __syncwarp();
      const u32* ct = (const u32*)OP().p8 + (Te + 2) * 16 + (lane >> 2);
      cp_async4(ws + lane, ct); cp_async4(ws + 32 + lane, ct + 8);
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
  }
  if (left != (int)OP().KQ) {
    const int lane = threadIdx.x & 31;
    float4* ps = (float4*)(sm + WORK_OFF) + (wib * 2 + 1) * 32 + lane;
    ps[0] = make_float4(a_[0], a_[1], a_[2], a_[3]);
  }
  __syncwarp();
  {
    const int lane = threadIdx.x & 31;
    const int KQ = (int)OP().KQ, a = (int)ws[66], t = a / KQ, qe = (int)ws[67];
    int* rt = (int*)(sm + ROLE_OFF);
    if (lane == 0) rt[wib] = (a > t * KQ && (t + 1) * KQ <= qe) ? t + 1 : 0;
    if (wib == 31 && lane == 0) { const int Tb = (qe - 1) / KQ; rt[32] = qe < (Tb + 1) * KQ ? Tb + 1 : 0; }
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  dbg_stamp(0);
#ifndef NO_STAMPS
  if (CX().flags & 24) return;
#endif
  {
    const int* rt = (const int*)(sm + ROLE_OFF);
    const int ra = rt[wib];
    if (ra) finalize<NGT>(ra - 1, 0, wib);
    if (wib >= 29) {
      const int rb = rt[32];
      if (rb) {
        const int ex = !rt[30] ? 30 : (!rt[29] ? 29 : 31);
        if (wib == ex) finalize<NGT>(rb - 1, 1, 31);
      }
    }
  }
  if (wib == 0 || wib == 31) { __syncwarp(); }
  if (threadIdx.x == 0 || threadIdx.x == 992) dbg_stamp(1);
}
#endif
// The r partials of 5-8 columns by warps 0-3 (see gemm_phase): thread (warp w', lane l) sums column c = l & 7 for the
// two partial-sum rows w = 8 w' + 2 (l >> 3) + {0, 1}: tiles w, w + 32, ... in that order, as each warp did for itself
// in_proj (op.p5 = the conv history ring of the recurrent op that follows, set by the host): L2 is asked for sequence
// b's three history rows that op's input phase reads (20 KB each, written 1-3 steps ago: cold; adjacent in the ring
// unless it wraps: one request of 60 KB), while this GEMM streams. Called by one thread of block b < B.
// (24 requests of one row each from one warp instead: +85 us per B=8 step - the warp became the block's straggler.
// The same call in the other kernel: B=1 +27 us, B=2 +9: only the B >= 7 kernel has it.)
__device__ __forceinline__ void ring_prefetch() {
  const int b = blockIdx.x, pos = spos(b);
  if (pos >= 3) {
    const int r0 = (pos - 3) & (RING - 1);
    const char* base = (const char*)OP().p5 + (size_t)b * RING * 20480;
    if (r0 <= RING - 3) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(base + (size_t)r0 * 20480), "r"(61440u) : "memory");
    else for (int j = 0; j < 3; ++j) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(base + (size_t)((r0 + j) & (RING - 1)) * 20480), "r"(20480u) : "memory");
  }
}
__device__ __noinline__ void r_partials4() {
  extern __shared__ float sm[];
  const Ctx& cx = CX();
  const int wi = threadIdx.x >> 5, ln = threadIdx.x & 31, NC = cx.NC;
#if !defined(NO_RING_PF) && !defined(W12_ONLY)   // BF16 engine: -11 to -20 us per B=8 step; 12-bit engine: +25 us (s0015 vs s0013)
  if (wi == 2 && ln == 0 && OP().p5 != nullptr && (int)blockIdx.x < cx.B) ring_prefetch();
#endif
  float* part = sm + WORK_OFF + RPART_FLOATS(cx.NG);   // [32 warps][32 columns]
  const int c = ln & 7, w0 = wi * 8 + (ln >> 3) * 2;
  float v[2][10];
  #pragma unroll
  for (int k = 0; k < 2; ++k)
    #pragma unroll
    for (int i = 0; i < 10; ++i) v[k][i] = c < NC ? ldv(cx.ssq + (size_t)(w0 + k + 32 * i) * NCM + c) : 0.f;
  #pragma unroll
  for (int k = 0; k < 2; ++k) {
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < 10; ++i) s += v[k][i];
    part[(w0 + k) * 32 + c] = s;
  }
  __syncwarp();
  __threadfence_block();
  if (ln == 0) atomicAdd((int*)(sm + RCNT_OFF), 8);
#ifndef R_LAZY
  // warp 0 also finishes r now (the reduction every epilogue would otherwise race for after the stream): the same code
  if (wi == 0) ensure_r_big(NC, cx.NG);
#endif
}
template <bool TRI>   // the kernel entry point (mk_body): the B >= 7 one-token kernel has its own form of the r partials
__device__ __forceinline__ void gemm_phase() {
  extern __shared__ float sm[];
  const Ctx& cx = CX(); const Op& op = OP();
  if (op.use_r) {
    // r[c] = rsqrt(mean(h^2) + eps) from the per-tile partial sums of squares, the same in every block and for every
    // batch size: warp w sums tiles w, w + 32, ... (in that order) for all columns at once (lane = column, coalesced),
    // then warp c adds the 32 warp sums of column c with the butterfly of warp_sum.
    const int wi = threadIdx.x >> 5, ln = threadIdx.x & 31, NC = cx.NC;
    if (NC <= R_ASYNC_MAX) {   // few columns: one warp per column, lane j sums tiles j, j + 32, ... (the same sums, the same order)
      // r is needed only by the epilogues: the other warps start streaming at once and an epilogue waits for the
      // count of warps that have published their column's r (a block barrier here cost ~1.5 us per op).
      if (wi < NC) {
        float v[10];
        #pragma unroll
        for (int i = 0; i < 10; ++i) v[i] = ldv(cx.ssq + (size_t)(ln + 32 * i) * NCM + wi);
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 10; ++i) s += v[i];
        s = warp_sum(s);
        if (ln == 0) {
          sm[SMR_OFF + wi] = rsqrtf(s * (1.0f / HID) + 1e-6f);
          __threadfence_block();
          atomicAdd((int*)(sm + RCNT_OFF), 1);
        }
        __syncwarp();
      }
    } else if (TRI && NC <= 8) {
      // 5-8 columns in the B >= 7 kernel: the same partial sums (tiles w, w + 32, ... of column c, added in that order,
      // for w = 0..31), but all of them by warps 0-3, two (w, c) pairs per thread: those warps' shares were prefetched
      // and they end their streams 4-6 us early, while with every warp loading its own 10 partials first all 32
      // streams of the 113 normalised GEMMs of a step started a round trip late (R_ASYNC_MAX = 8, which delays
      // warps 4-7 instead, measured +115 us per B=8 step).
      if (wi < 4) r_partials4();   // its own function: inlined here it took the main loop from 36 to 116 bytes of spill
    } else {   // warp w: tiles w, w + 32, ... of all columns (lane = column, coalesced); no block barrier here
    float* part = sm + WORK_OFF + RPART_FLOATS(cx.NG);   // [32 warps][32 columns]
    float v[10];
    #pragma unroll
    for (int i = 0; i < 10; ++i) v[i] = ln < NC ? ldv(cx.ssq + (size_t)(wi + 32 * i) * NCM + ln) : 0.f;
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < 10; ++i) s += v[i];
    part[wi * 32 + ln] = s;
    __syncwarp();
    __threadfence_block();
    if (ln == 0) atomicAdd((int*)(sm + RCNT_OFF), 1);
    }
  }
  const int NG = cx.NG;
  if (op.ep == EP_RESID && (int)threadIdx.x < cx.NC * 2 * HC_TILES) {   // only the copying threads (and thread 0)
    // The residual rows of the tiles this block may finish, loaded now (their latency hides behind the stream)
    // instead of after the cross-block partial sums, where they were a second dependent round trip.
    const int KQ = (int)op.KQ;
    const long long UQ = (long long)op.NT * KQ;
    const int t0 = qw(UQ, blockIdx.x * 32, cx.nwarps) / KQ, t1 = (qw(UQ, blockIdx.x * 32 + 32, cx.nwarps) - 1) / KQ, nt = t1 - t0 + 1;
    int* hci = (int*)(sm + RCNT_OFF + 2);
    if (nt <= HC_TILES) {
      // column-major copy [column][row of the block's tiles] with cp.async (16 bytes per thread, no registers): a
      // thread does not wait for its load before it starts streaming; the copies are waited for at the block sync
      u16* hc = (u16*)(sm + WORK_OFF + 32 * 2 * NG * 128);
      const int NC = cx.NC, per = nt * 2, k = threadIdx.x;
      if (k < NC * per) {
        const int c = k / per, q = k - c * per;
        const u16* src = cx.h + (size_t)c * HID + t0 * 16 + q * 8;
        const unsigned dst = (unsigned)__cvta_generic_to_shared(hc + c * (HC_TILES * 16) + q * 8);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
      if (threadIdx.x == 0) { hci[0] = t0; hci[1] = nt; }
    } else if (threadIdx.x == 0) { hci[0] = 0; hci[1] = 0; }
  }
  if (op.i7 >> 32) {   // the matrix is stored in 12 bits per weight: exact W12 (escapes patched into the operands)
    // (the s0002 sparse-correction loops gemm_stream_w12c / w12d and the s4 loops gemm_stream_v1 / w12 are kept in
    // the source but no longer called: their call sites alone cost 0.7 % per decode pass, hour 3.1)
#ifdef W12_RING   // B = 1 decode passes (one real column) through the cp.async ring; api.py sizes its shared memory
    if (CX().NC == 1) gemm_stream_w12r();
    else
#endif
#ifdef X_COMPACT
    if (CX().NC == 1) gemm_stream_w12e<1, 4>();
    else
#endif
    if (NG == 1) gemm_stream_w12e<1>();
    else if (NG == 2) gemm_stream_w12e<2>();
    else gemm_stream_w12e<4>();
  } else {
#if !defined(W12_ONLY) || defined(W12_MIXED)   // the W12 build stores every streamed matrix in 12 bits unless W12_MIXED
                   // (out_proj / o_proj in BF16); without these call sites the pass was 0.7 % faster at s0004 (now ~0.1 %)
#ifdef X_COMPACT
    if (CX().NC == 1) gemm_stream_bf<1, 4>(); else
#endif
    if (NG == 1) gemm_stream_bf<1>();
    else if (NG == 2) gemm_stream_bf<2>();
    else gemm_stream_bf<4>();
#endif
  }
}

// ------------------------------------------------------------------------------------------------ embedding
__device__ __noinline__ void embed_work() {
  const Ctx& cx = CX(); const Op& op = OP();
  const int lane = threadIdx.x & 31;
  const int w = blockIdx.x * 32 + (threadIdx.x >> 5);
  const u16* emb = (const u16*)op.W;
  const float* gn = (const float*)op.gnext;
  u16* xn = (u16*)op.out;
  const int nunits = cx.NC * 320;
  for (int u = w; u < nunits; u += cx.nwarps) {
    const int c = u / 320, tile = u % 320;
    float sq = 0.f;
    if (lane < 16) {
      const long long tok = cx.tok_in[(size_t)c * cx.tok_stride];
      const int row = tile * 16 + lane;
      const u16 e = emb[(size_t)tok * HID + row];
      const float f = bf2f(e);
      cx.h[(size_t)c * HID + row] = e;
      xn[xg_idx(c, row, cx.NG)] = f2bf(f * gn[row]);
      sq = f * f;
    }
    sq = warp_sum(sq);
    if (lane == 0) cx.ssq[(size_t)tile * NCM + c] = sq;
  }
}

// ------------------------------------------------------------------------------------------------ linear attention
// Cache operators of the recurrent state's loads and stores (each read once and written once per step, 64 KB per head):
// STL_K 0 cg, 1 cs, 2 lu; STS_K 0 default (wb), 1 cs, 2 wt, 3 cg
#ifndef STL_K
#define STL_K 0
#endif
#ifndef STS_K
#define STS_K 0
#endif
__device__ __forceinline__ float ld_state(const float* p) {
#if STL_K == 0
  return ldv(p);
#elif STL_K == 1
  float v; asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory"); return v;
#else
  float v; asm volatile("ld.global.lu.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory"); return v;
#endif
}
__device__ __forceinline__ void st_state(float* p, float v) {
#if STS_K == 0
  *p = v;
#elif STS_K == 1
  asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
#elif STS_K == 2
  asm volatile("st.global.wt.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
#else
  asm volatile("st.global.cg.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
#endif
}
__device__ __forceinline__ float lm_upd(float s, float d, float k, float dl) { return __fmaf_rn(s, d, __fmul_rn(k, dl)); }

// Gated delta rule, one (sequence, value head) per block. Thread (v, g) = (tid >> 3, tid & 7) keeps rows 16g..16g+15
// of column v of the 128 x 128 float32 state in registers; a warp holds 4 columns, so both reductions over k of a
// position (S^T k and S^T q) are 3 shuffles and the loop over the positions of a pass has no block barrier. The state
// is stored in this thread order ([warp][row i][lane]: coalesced). The update is rank 1 (S = d S + k (x) delta): a
// speculative pass leaves the stored state alone and saves (d, k, delta) per position; the next pass replays the
// accepted ones with the same lm_upd in the same order, so the committed state is bit-identical to stepping.
#define LKS 16                                   // a 16-row group of k or q; its 4 float4 chunks are XOR-swizzled by group
__device__ __forceinline__ int lsw(int g, int i) { return g * 16 + (((i >> 2) ^ ((g >> 1) & 3)) << 2) + (i & 3); }   // conflict-free LDS.128
// Shared-memory layout (floats from WORK_OFF) for at most PM positions; PM is a template parameter (8 or PMAX) so
// that the offsets are compile-time constants and a single step does not pay for the space of 32 positions
// (run-time offsets measured +0.13 ms per step; 84 KB of shared memory +0.15 ms).
#define LQ 0                                     // q [PM][8 * 16]
#define LK (LQ + PM * 8 * LKS)                   // k [PM][8 * 16]
#define LV (LK + PM * 8 * LKS)                   // v [PM][128]
#define LZ (LV + PM * 128)                       // z [PM][128]
#define LO (LZ + PM * 128)                       // core output [PM][128]
#define LS (LO + PM * 128)                       // (beta, decay) [PM][2]
#define LR (LS + 2 * PM)                         // reductions [PM][8] (at least 32)
#define LM_FLOATS(PM) (5 * (PM) * 128 + 10 * (PM) + 32)
#define LQ8 0                                    // the same layout for PM = 8 (lmix_dual)
#define LK8 (8 * 8 * LKS)
#define LV8 (2 * 8 * 8 * LKS)
#define LZ8 (LV8 + 8 * 128)
#define LO8 (LZ8 + 8 * 128)
#define LS8 (LO8 + 8 * 128)
#define LR8 (LS8 + 2 * 8)
__device__ __forceinline__ float grp8_sum(float x) {   // butterfly over the 8 lanes of a column: the same value in all 8
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  x += __shfl_xor_sync(0xffffffffu, x, 4);
  return x;
}
#ifdef LMIX_PROF
__device__ unsigned long long lmprof[8];
__device__ __forceinline__ void lmp(int i, unsigned long long& t) {
  if (blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long n; asm volatile("mov.u64 %0, %%clock64;" : "=l"(n)); if (i > 0) atomicAdd(lmprof + i, n - t); t = n; }   // cycles
}
extern "C" int mk_lmprof(unsigned long long* out) { return (int)cudaMemcpyFromSymbol(out, lmprof, 64); }
#define LMP(i) lmp(i, lmt)
#else
#define LMP(i)
#endif
// The state's whole life inside one pass: load, replay of the previous pass's accepted updates, the recurrence over
// this pass's positions, store. Its own function so that nothing else is live next to the 16 state registers (with
// the input phase in the same function the compiler spilled inside the recurrence: ~1 us per position).
template <int PM, int PART>   // PART -1: the whole head in this block; 0 / 1: columns 0-63 / 64-127 (compile time:
                               // with the half as a run-time value the recurrence spilled 19 registers per position)
__device__ __noinline__ void lm_core(int unit, int P, int pos0, int na, int flags, long long sc, long long so) {
  // sc: offset (floats) of the committed state buffer of this sequence, so: of the other one, where a speculative
  // pass leaves its final state (taken over without a replay when the next pass learns that all were accepted)
  extern __shared__ float sm[];
  float* L = sm + WORK_OFF;
  // part >= 0: this block holds columns 64 part .. 64 part + 63 of the head (threads 0-511 stand for 512 part + tid);
  // the arithmetic of a column does not depend on the split
  constexpr bool SPLIT = PART >= 0;
  if (SPLIT && threadIdx.x >= 512) {   // no columns here: meet the block barrier below and leave (conditionals
    asm volatile("bar.sync 0;" ::: "memory");   // around the state's life made the compiler spill ~700 bytes)
    return;
  }
  const int tid = SPLIT ? threadIdx.x + 512 * PART : threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int g = tid & 7, v = tid >> 3;
  const bool spec = (flags & F_SPEC) != 0;
  // deferred mode: this step only saves its update (delta through the v area, as a speculative pass does)
  const bool keep = (flags & F_DEFER) && P == 1 && na + 1 < CX().defer_n;
  float s[16];
  {
    const float* Sg = (const float*)OP().p4 + sc + ((size_t)unit << 14) + (size_t)wid * 512 + lane;
    #pragma unroll
    for (int i = 0; i < 16; ++i) s[i] = (pos0 > ((flags & F_DEFER) ? na : 0) && !(flags & 32)) ? ld_state(Sg + i * 32) : 0.f;
  }
  if (na > 0) {   // replay: k staged in the core-output area, delta and decay read here, 8 positions per batch
    const float* pd = (const float*)OP().p11 + (size_t)unit * PMAX * 128 + v;
    const float* pdec = (const float*)OP().W + (size_t)unit * PMAX;
    for (int j0 = 0; j0 < na; j0 += 8) {
      float dv[8], dj[8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) { dv[e] = j0 + e < na ? ldv(pd + (j0 + e) * 128) : 0.f; dj[e] = j0 + e < na ? ldv(pdec + j0 + e) : 0.f; }
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        if (j0 + e >= na) break;
        const float4* kk4 = (const float4*)(L + LO + (j0 + e) * 8 * LKS);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const float4 k4 = kk4[g * 4 + (c ^ ((g >> 1) & 3))];
          s[4 * c] = lm_upd(s[4 * c], dj[e], k4.x, dv[e]); s[4 * c + 1] = lm_upd(s[4 * c + 1], dj[e], k4.y, dv[e]);
          s[4 * c + 2] = lm_upd(s[4 * c + 2], dj[e], k4.z, dv[e]); s[4 * c + 3] = lm_upd(s[4 * c + 3], dj[e], k4.w, dv[e]);
        }
      }
    }
    if (!(flags & F_DEFER)) {   // deferred mode: the stored state stays as it is until the commit step
      float* Sg = (float*)OP().p4 + sc + ((size_t)unit << 14) + (size_t)wid * 512 + lane;
      #pragma unroll
      for (int i = 0; i < 16; ++i) st_state(Sg + i * 32, s[i]);
    }
  }
  asm volatile("bar.sync 0;" ::: "memory");   // the staged k are consumed before the recurrence writes its outputs there
  if (flags & 512) return;   // debug 512: inputs and replay only
  const int sw = (g >> 1) & 3;
  for (int p = 0; p < P; ++p) {
    const float4* kp = (const float4*)(L + LK + p * 8 * LKS) + g * 4;
    const float4* qp = (const float4*)(L + LQ + p * 8 * LKS) + g * 4;
    const float beta = L[LS + p * 2], dec = L[LS + p * 2 + 1];
    float kr[16];
    #pragma unroll
    for (int c = 0; c < 4; ++c) { const float4 t = kp[c ^ sw]; kr[4 * c] = t.x; kr[4 * c + 1] = t.y; kr[4 * c + 2] = t.z; kr[4 * c + 3] = t.w; }
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc = __fmaf_rn(s[i], kr[i], acc);
    acc = grp8_sum(acc);
    const float d = (L[LV + p * 128 + v] - dec * acc) * beta;
    if (spec || keep) {   // delta for the next pass's replay: over this position's (consumed) v input, written out after the
      __syncwarp();   // loop in one coalesced pass (per-position global stores cost ~1.8 us per layer at 32 positions)
      if (g == 0) L[LV + p * 128 + v] = d;
    }
    float acc2 = 0.f;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float4 q4 = qp[c ^ sw];
      s[4 * c] = lm_upd(s[4 * c], dec, kr[4 * c], d);         acc2 = __fmaf_rn(s[4 * c], q4.x, acc2);
      s[4 * c + 1] = lm_upd(s[4 * c + 1], dec, kr[4 * c + 1], d); acc2 = __fmaf_rn(s[4 * c + 1], q4.y, acc2);
      s[4 * c + 2] = lm_upd(s[4 * c + 2], dec, kr[4 * c + 2], d); acc2 = __fmaf_rn(s[4 * c + 2], q4.z, acc2);
      s[4 * c + 3] = lm_upd(s[4 * c + 3], dec, kr[4 * c + 3], d); acc2 = __fmaf_rn(s[4 * c + 3], q4.w, acc2);
    }
    acc2 = grp8_sum(acc2);
    if (g == 0) L[LO + p * 128 + v] = rbf(acc2);
  }
  if (!(flags & 32) && !keep) {   // committed (single steps) or the speculative final state, in the other buffer
    float* Sg = (float*)OP().p4 + (spec ? so : sc) + ((size_t)unit << 14) + (size_t)wid * 512 + lane;
    #pragma unroll
    for (int i = 0; i < 16; ++i) st_state(Sg + i * 32, s[i]);
  }
}

__device__ __forceinline__ void st_release(u64* p, u64 v) { asm volatile("st.release.gpu.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory"); }
// Inputs of the recurrent op, 4 adjacent channels per thread (8-byte loads): 8 runs of positions x 128 threads,
// a warp = one role (q, k, v, z) of one run, so the L2 norm of q and k is finished inside the warp: its sum of
// squares reproduces warp_sum's butterfly over each 32-channel group (channel c ^ 16, ^ 8, ^ 4 are lanes ^ 4, ^ 2,
// ^ 1; c ^ 2 and c ^ 1 are in the thread) and then (r0 + r1) + (r2 + r3), as before: the same bits.
template <int PM>
__device__ __noinline__ void lm_inputs(int u, int b, int hv, int hk, int P, int c0, int pos0, int na, int part) {
  extern __shared__ float sm[];
  const Ctx& cx = CX(); const Op& op = OP();
  float* L = sm + WORK_OFF;
  const int tid = threadIdx.x, lane = tid & 31;
  const bool spec = (cx.flags & F_SPEC) != 0;
  float* pk = (float*)op.p10 + (size_t)u * PMAX * 128;
  #pragma unroll
  for (int i = 0; i < PM / 8; ++i) {   // na <= PM: all loads in flight together
    const int idx = tid + 1024 * i;
    if (idx < na * 128) { const int j = idx >> 7, k = idx & 127; L[LO + j * 8 * LKS + lsw(k >> 4, k & 15)] = ldv(pk + idx); }
  }
  if (na > 0) __syncthreads();   // the previous pass's k are read before this pass's k overwrite them (pk): a race
                                 // without this barrier (B=8 decodes diverged at random tokens)
  constexpr int H = PM / 8;   // positions per run
  const int run = tid >> 7, role = (tid >> 5) & 3, j = lane * 4;   // channels j .. j + 3 of the role
  const int pr = (P + 7) >> 3, p_lo = run * pr, p_hi = min(P, p_lo + pr);
  const u16* mb = cx.mbuf + (size_t)c0 * MROW;
  if (p_lo >= p_hi) return;   // whole warps: the run's positions are uniform in the warp
  if (role < 3) {
    const int ch = role == 0 ? hk * 128 + j : (role == 1 ? 2048 + hk * 128 + j : 4096 + hv * 128 + j);
    const bool wr = (role == 2 || hv % 3 == 0) && part <= 0;   // both blocks of a split head compute the inputs
    u16* ring = (u16*)op.p5 + (size_t)b * RING * 10240 + ch;
    uint2 xr[H + 3];
    #pragma unroll
    for (int i = 0; i < H + 3; ++i) {
      const int pp = p_lo - 3 + i;
      uint2 xv = make_uint2(0, 0);
      if (pp < p_hi) {
        if (pp >= 0) xv = ldv((const uint2*)(mb + (size_t)pp * MROW + ch));
        else if (pos0 + pp >= 0) xv = ldv((const uint2*)(ring + (size_t)((pos0 + pp) & (RING - 1)) * 10240));
      }
      xr[i] = xv;
    }
    float4 cw[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) cw[c] = *((const float4*)op.p6 + ch + c);
    const int jx = lsw(j >> 4, j & 15);
    #pragma unroll
    for (int i = 0; i < H; ++i) {
      const int p = p_lo + i;
      if (p >= p_hi) break;
      if (wr) *(uint2*)(ring + (size_t)((pos0 + p) & (RING - 1)) * 10240) = xr[i + 3];
      float val[4];
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        auto el = [&](const uint2& x) { const u32 w = c < 2 ? x.x : x.y; return (u16)((c & 1) ? (w >> 16) : (w & 0xffffu)); };
        float y = cw[c].w * bf2f(el(xr[i + 3]));
        y += cw[c].z * bf2f(el(xr[i + 2]));
        y += cw[c].y * bf2f(el(xr[i + 1]));
        y += cw[c].x * bf2f(el(xr[i]));
        val[c] = rbf(siluf_(rbf(y)));
      }
      if (role == 2) { *(float4*)(L + LV + p * 128 + j) = make_float4(val[0], val[1], val[2], val[3]); continue; }
      float q[4];
      #pragma unroll
      for (int c = 0; c < 4; ++c) q[c] = val[c] * val[c];
      #pragma unroll
      for (int c = 0; c < 4; ++c) q[c] += __shfl_xor_sync(0xffffffffu, q[c], 4);
      #pragma unroll
      for (int c = 0; c < 4; ++c) q[c] += __shfl_xor_sync(0xffffffffu, q[c], 2);
      #pragma unroll
      for (int c = 0; c < 4; ++c) q[c] += __shfl_xor_sync(0xffffffffu, q[c], 1);
      q[0] += q[2]; q[1] += q[3];
      float r = q[0] + q[1];                       // the warp_sum of this 32-channel group
      r += __shfl_xor_sync(0xffffffffu, r, 8);     // r0 + r1, r2 + r3
      r += __shfl_xor_sync(0xffffffffu, r, 16);    // (r0 + r1) + (r2 + r3)
      const float inv = rsqrtf(r + 1e-6f);
      float nv[4];
      #pragma unroll
      for (int c = 0; c < 4; ++c) nv[c] = role == 0 ? val[c] * inv * 0.08838834764831845f : val[c] * inv;
      *(float4*)(L + (role == 0 ? LQ : LK) + p * 8 * LKS + jx) = make_float4(nv[0], nv[1], nv[2], nv[3]);
      if (spec && role == 1 && part <= 0) *(float4*)(pk + p * 128 + j) = make_float4(nv[0], nv[1], nv[2], nv[3]);
    }
  } else {
    uint2 zr[H];
    #pragma unroll
    for (int i = 0; i < H; ++i) zr[i] = p_lo + i < p_hi ? ldv((const uint2*)(mb + (size_t)(p_lo + i) * MROW + 10240 + hv * 128 + j)) : make_uint2(0, 0);
    #pragma unroll
    for (int i = 0; i < H; ++i)
      if (p_lo + i < p_hi)
        *(float4*)(L + LZ + (p_lo + i) * 128 + j) = make_float4(bf2f((u16)(zr[i].x & 0xffffu)), bf2f((u16)(zr[i].x >> 16)), bf2f((u16)(zr[i].y & 0xffffu)), bf2f((u16)(zr[i].y >> 16)));
    if (lane < H && p_lo + lane < p_hi) {   // decay and beta: one position per thread
      const float* negA = (const float*)op.p7; const float* dtb = (const float*)op.p8;
      const u16* m = mb + (size_t)(p_lo + lane) * MROW;
      const u16 bb = ldv(m + 16384 + hv), ab = ldv(m + 16432 + hv);
      const float aa = bf2f(ab) + dtb[hv];
      const float sp = aa > 20.f ? aa : log1pf(__expf(aa));
      L[LS + (p_lo + lane) * 2] = rbf(sigmoidf_(bf2f(bb)));
      L[LS + (p_lo + lane) * 2 + 1] = __expf(negA[hv] * sp);
    }
  }
}


// One position per sequence, no speculation, more units than blocks (B >= 4): each half of the block (512 threads)
// runs a unit of its own, so 384 units take 2 rounds instead of 3. Every value is computed exactly as in lmix_work
// at one position (its input phase already used threads 0-511 only); in the recurrence a thread holds the state
// columns v and v + 64 of the unsplit layout one after the other: the same arithmetic per column.
__device__ __forceinline__ void hbar(int h) { asm volatile("bar.sync %0, 512;" :: "r"(1 + h) : "memory"); }
// na: saved updates to replay first (F_DEFER; staged by the input phase in the unused positions 1.. of the k, v and
// (beta, decay) areas); pdn: where this step's delta goes when it only saves its update (null: the state is stored)
#define PEND_K(u) ((float*)OP().p10 + (size_t)(u) * PMAX * 128)
#define PEND_D(u) ((float*)OP().p11 + (size_t)(u) * PMAX * 128)
#define PEND_C(u) ((float*)OP().W + (size_t)(u) * PMAX)
// per half, written by thread 0 of the half at the start of a unit (read from shared memory at each use, so that
// nothing of it is live next to the state registers): [0] saved updates to replay, [1] this step saves its update
// instead of storing the state, [2] the stored state exists (position > saved updates), [4..5] where this step's
// delta goes when it is saved
#define DSTASH(L) ((volatile int*)((L) + LR8 + 16))
template <bool DEFER>
__device__ __forceinline__ void lm_dual_col(float* L, const float* Sg, float* Sgw, int g, int v, int pos0, int flags) {
  float s[16];
  const bool ld = DEFER ? DSTASH(L)[2] != 0 : pos0 > 0;
  #pragma unroll
  for (int i = 0; i < 16; ++i) s[i] = (ld && !(flags & 32)) ? ld_state(Sg + i * 32) : 0.f;
  const int sw = (g >> 1) & 3;
  for (int e = 1; DEFER && e <= DSTASH(L)[0]; ++e) {   // the same lm_upd in the same order as the steps that saved them
    const float4* ke = (const float4*)(L + LK8 + e * 8 * LKS) + g * 4;
    const float dj = L[LS8 + e * 2 + 1], dv = L[LV8 + e * 128 + v];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float4 k4 = ke[c ^ sw];
      s[4 * c] = lm_upd(s[4 * c], dj, k4.x, dv); s[4 * c + 1] = lm_upd(s[4 * c + 1], dj, k4.y, dv);
      s[4 * c + 2] = lm_upd(s[4 * c + 2], dj, k4.z, dv); s[4 * c + 3] = lm_upd(s[4 * c + 3], dj, k4.w, dv);
    }
  }
  const float4* kp = (const float4*)(L + LK8) + g * 4;
  const float4* qp = (const float4*)(L + LQ8) + g * 4;
  const float beta = L[LS8], dec = L[LS8 + 1];
  float kr[16];
  #pragma unroll
  for (int c = 0; c < 4; ++c) { const float4 t = kp[c ^ sw]; kr[4 * c] = t.x; kr[4 * c + 1] = t.y; kr[4 * c + 2] = t.z; kr[4 * c + 3] = t.w; }
  float acc = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) acc = __fmaf_rn(s[i], kr[i], acc);
  acc = grp8_sum(acc);
  const float d = (L[LV8 + v] - dec * acc) * beta;
  float acc2 = 0.f;
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float4 q4 = qp[c ^ sw];
    s[4 * c] = lm_upd(s[4 * c], dec, kr[4 * c], d);         acc2 = __fmaf_rn(s[4 * c], q4.x, acc2);
    s[4 * c + 1] = lm_upd(s[4 * c + 1], dec, kr[4 * c + 1], d); acc2 = __fmaf_rn(s[4 * c + 1], q4.y, acc2);
    s[4 * c + 2] = lm_upd(s[4 * c + 2], dec, kr[4 * c + 2], d); acc2 = __fmaf_rn(s[4 * c + 2], q4.z, acc2);
    s[4 * c + 3] = lm_upd(s[4 * c + 3], dec, kr[4 * c + 3], d); acc2 = __fmaf_rn(s[4 * c + 3], q4.w, acc2);
  }
  acc2 = grp8_sum(acc2);
  if (g == 0) L[LO8 + v] = rbf(acc2);
  if (DEFER && DSTASH(L)[1]) { if (g == 0) (*(float* volatile*)(DSTASH(L) + 4))[v] = d; }
  else if (!(flags & 32)) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) st_state(Sgw + i * 32, s[i]);
  }
}
// DEFER (F_DEFER launches): a second instance, so that the plain one keeps its registers (with the saved-update
// pointers and counts live in one function ptxas spilled 68 / 120 bytes instead of 32 / 44: +110 us per B=8 step).
// The saved-update state is re-read from shared memory at each use instead of being kept across the phases.
template <bool DEFER>
__device__ __noinline__ void lmix_dual() {
  const bool xcmp = x_compact();   // hoisted: the producers write the compact layout at NC = 1
  extern __shared__ float sm[];
  const Ctx& cx = CX(); const Op& op = OP();
  const int h = threadIdx.x >> 9, lt = threadIdx.x & 511, lane = lt & 31, wl = lt >> 5;
  float* L = sm + WORK_OFF + h * LM_FLOATS(8);
  const int NG = cx.NG, flags = cx.flags, nunits = cx.B * 48;
#ifdef LMIX_PROF
  unsigned long long lmt = 0;
#endif
  LMP(0);
  // the states of this half's units: L2 is asked to fetch them (thread 32 of the half)
  auto pf_states = [&]() {
    for (int r = 0;; ++r) {
      const int u = (r * cx.nblocks + blockIdx.x) * 2 + h;
      if (u >= nunits) break;
      if (spos(u / 48) <= 0) continue;
      const long long sc = spar(u / 48) ? op.M : 0;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], 65536;" :: "l"((const float*)op.p4 + sc + ((size_t)u << 14)) : "memory");
    }
  };
#ifdef LMPF_EARLY   // as inherited: before any input load (B=8 +80 us per step: findings S3)
  if (lt == 32) pf_states();
#endif
  for (int r = 0;; ++r) {
    const int u = (r * cx.nblocks + blockIdx.x) * 2 + h;
    if (u >= nunits) break;
    const int b = u / 48, hv = u % 48, hk = hv / 3;
    const int c0 = c_start(cx, b), pos0 = spos(b);
    const long long sc = spar(b) ? op.M : 0;
    if (DEFER && lt == 0) {
      const int na = sna(b);
      DSTASH(L)[0] = na; DSTASH(L)[1] = na + 1 < CX().defer_n; DSTASH(L)[2] = pos0 > na;
      *(float* volatile*)(DSTASH(L) + 4) = PEND_D(u) + na * 128;
    }
    LMP(1);
    // inputs (lmix_work at one position)
    {
      const float* negA = (const float*)op.p7; const float* dtb = (const float*)op.p8;
      const int role = (lt >> 7) & 3, j = lt & 127;
      const int ch = role == 0 ? hk * 128 + j : (role == 1 ? 2048 + hk * 128 + j : 4096 + hv * 128 + j);
      const bool wr = role == 2 || hv % 3 == 0;
      u16* ring = (u16*)op.p5 + (size_t)b * RING * 10240 + ch;
      const u16* mb = cx.mbuf + (size_t)c0 * MROW;
      const int jx = lsw(j >> 4, j & 15);
      // F_DEFER: thread (role, j) stages element j of saved update `role` (at most 4: defer_n <= 5) in the unused
      // positions 1.. of the k, v and decay areas. Issued right after the thread's input loads (all in flight
      // together) and stored at once, so that nothing of it is live during the convolution.
      auto stage = [&]() {
        if (!DEFER) return;
        const int na = sna(b);
        if (role < na) {
          const float kv = ldv(PEND_K(u) + role * 128 + j), dv = ldv(PEND_D(u) + role * 128 + j);
          L[LK8 + (role + 1) * 8 * LKS + jx] = kv; L[LV8 + (role + 1) * 128 + j] = dv;
        }
        if (lt < na) L[LS8 + (lt + 1) * 2 + 1] = ldv(PEND_C(u) + lt);
      };
      if (role < 3) {
        const float4 cw = *((const float4*)op.p6 + ch);
        u16 xr[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int pp = i - 3;
          u16 xv = 0;
          if (pp >= 0) xv = ldv(mb + ch);
          else if (pos0 + pp >= 0) xv = ldv(ring + (size_t)((pos0 + pp) & (RING - 1)) * 10240);
          xr[i] = xv;
        }
        stage();
#ifndef LMPF_EARLY   // after the first unit's input loads: those (conv history, saved updates: DRAM) do not queue behind 25 MB of state
        if (lt == 32 && r == 0) pf_states();
#endif
        if (wr) ring[(size_t)(pos0 & (RING - 1)) * 10240] = xr[3];
        float y = cw.w * bf2f(xr[3]);
        y += cw.z * bf2f(xr[2]);
        y += cw.y * bf2f(xr[1]);
        y += cw.x * bf2f(xr[0]);
        const float val = rbf(siluf_(rbf(y)));
        if (role == 0) L[LQ8 + jx] = val;
        else if (role == 1) L[LK8 + jx] = val;
        else L[LV8 + j] = val;
        const float ssum = warp_sum(val * val);
        if (lane == 0 && role < 2) L[LR8 + role * 4 + (wl & 3)] = ssum;
      } else {
        const u16 zv = ldv(mb + 10240 + hv * 128 + j);
        stage();
        L[LZ8 + j] = bf2f(zv);
        if (j == 0) {
          const u16 bb = ldv(mb + 16384 + hv), ab = ldv(mb + 16432 + hv);
          const float aa = bf2f(ab) + dtb[hv];
          const float sp = aa > 20.f ? aa : log1pf(__expf(aa));
          L[LS8] = rbf(sigmoidf_(bf2f(bb)));
          const float dcy = __expf(negA[hv] * sp);
          L[LS8 + 1] = dcy;
          if (DEFER) { const int na = sna(b); if (na + 1 < CX().defer_n) PEND_C(u)[na] = dcy; }
        }
      }
      hbar(h);
      if (role < 2) {
        const float* r2 = L + LR8 + role * 4;
        const float inv = rsqrtf((r2[0] + r2[1]) + (r2[2] + r2[3]) + 1e-6f);
        float* x = L + (role == 0 ? LQ8 : LK8) + jx;
        const float nv = role == 0 ? *x * inv * 0.08838834764831845f : *x * inv;
        *x = nv;
        if (DEFER && role == 1 && DSTASH(L)[1]) PEND_K(u)[DSTASH(L)[0] * 128 + j] = nv;
      }
      hbar(h);
    }
    LMP(2);
    // recurrence: columns v (threads' first) and v + 64 of the unsplit layout
    {
      const int g = lt & 7;
      const float* S0 = (const float*)op.p4 + sc + ((size_t)u << 14);
      float* W0 = (float*)op.p4 + sc + ((size_t)u << 14);
      lm_dual_col<DEFER>(L, S0 + wl * 512 + lane, W0 + wl * 512 + lane, g, lt >> 3, pos0, flags);
      lm_dual_col<DEFER>(L, S0 + (wl + 16) * 512 + lane, W0 + (wl + 16) * 512 + lane, g, (lt >> 3) + 64, pos0, flags);
    }
    hbar(h);
    LMP(3);
    // gated RMSNorm (lmix_work at one position: threads 0-127)
    {
      const float* nwt = (const float*)op.p9;
      u16* xn = (u16*)op.xf;
      const int vv = lt & 127;
      const bool on = lt < 128;
      const float ob = on ? L[LO8 + vv] : 0.f;
      const float so = warp_sum(ob * ob);
      if (lane == 0) L[LR8 + wl] = so;
      hbar(h);
      if (on) {
        const float* r2 = L + LR8;
        const float rs = rsqrtf(((r2[0] + r2[1]) + (r2[2] + r2[3])) * (1.0f / 128.0f) + 1e-6f);
        const float n = rbf(ob * rs);
        const float mm = rbf(nwt[vv] * n);
        xn[xg_idx(c0, hv * 128 + vv, NG, xcmp)] = f2bf(mm * siluf_(L[LZ8 + vv]));
      }
      hbar(h);
    }
    LMP(4);
  }
}

// ---- lmix_tri: the B >= 4 one-token recurrent op in ONE pass per block (no speculation, no deferred commits).
// lmix_dual runs its units in rounds (2 per block and round; 384 units at B=8: a second round on blocks 0-39), and
// every round pays its own input phase and four half-block barriers (~3 us of latency per round, findings S5).
// Here a block has up to three units: one per half (512 threads, state columns v and v + 64 one after the other, as
// lmix_dual) and a third one shared by both halves: half h takes its columns 64 h .. 64 h + 63, one column pass each
// (the arithmetic of a column does not depend on who holds it). All units' inputs are loaded and convolved in one
// phase (half 0's threads do the third unit's as a second task, loads of both tasks in flight together), so no
// thread makes more than three column passes and the phase latencies are paid once.
// Slots: unit of half h = positions 0 of that half's PM = 8 area; the shared unit = positions 1 of half 0's area.
template <int PP>
__device__ __forceinline__ void lm_tri_col(float* L, float* Sg, int g, int v, bool ld, int flags) {
  float s[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) s[i] = (ld && !(flags & 32)) ? ld_state(Sg + i * 32) : 0.f;
  const int sw = (g >> 1) & 3;
  const float4* kp = (const float4*)(L + LK8 + PP * 8 * LKS) + g * 4;
  const float4* qp = (const float4*)(L + LQ8 + PP * 8 * LKS) + g * 4;
  const float beta = L[LS8 + 2 * PP], dec = L[LS8 + 2 * PP + 1];
  float kr[16];
  #pragma unroll
  for (int c = 0; c < 4; ++c) { const float4 t = kp[c ^ sw]; kr[4 * c] = t.x; kr[4 * c + 1] = t.y; kr[4 * c + 2] = t.z; kr[4 * c + 3] = t.w; }
  float acc = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) acc = __fmaf_rn(s[i], kr[i], acc);
  acc = grp8_sum(acc);
  const float d = (L[LV8 + PP * 128 + v] - dec * acc) * beta;
  float acc2 = 0.f;
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float4 q4 = qp[c ^ sw];
    s[4 * c] = lm_upd(s[4 * c], dec, kr[4 * c], d);         acc2 = __fmaf_rn(s[4 * c], q4.x, acc2);
    s[4 * c + 1] = lm_upd(s[4 * c + 1], dec, kr[4 * c + 1], d); acc2 = __fmaf_rn(s[4 * c + 1], q4.y, acc2);
    s[4 * c + 2] = lm_upd(s[4 * c + 2], dec, kr[4 * c + 2], d); acc2 = __fmaf_rn(s[4 * c + 2], q4.z, acc2);
    s[4 * c + 3] = lm_upd(s[4 * c + 3], dec, kr[4 * c + 3], d); acc2 = __fmaf_rn(s[4 * c + 3], q4.w, acc2);
  }
  acc2 = grp8_sum(acc2);
  if (g == 0) L[LO8 + PP * 128 + v] = rbf(acc2);
  if (!(flags & 32)) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) st_state(Sg + i * 32, s[i]);
  }
}
// The recurrence of one state column on registers s[16] that the caller loaded (and stores): lm_tri_col without its
// memory accesses and with k read from shared memory at both uses instead of held in 16 registers, so that the caller
// can keep the NEXT column's 16 loads in flight during this column's arithmetic (the same operations in the same order).
template <int PP>
__device__ __forceinline__ void lm_tri_core(float* L, float* s, int g, int v) {
  const int sw = (g >> 1) & 3;
  const float4* kp = (const float4*)(L + LK8 + PP * 8 * LKS) + g * 4;
  const float4* qp = (const float4*)(L + LQ8 + PP * 8 * LKS) + g * 4;
  const float beta = L[LS8 + 2 * PP], dec = L[LS8 + 2 * PP + 1];
  float acc = 0.f;
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float4 k4 = kp[c ^ sw];
    acc = __fmaf_rn(s[4 * c], k4.x, acc); acc = __fmaf_rn(s[4 * c + 1], k4.y, acc);
    acc = __fmaf_rn(s[4 * c + 2], k4.z, acc); acc = __fmaf_rn(s[4 * c + 3], k4.w, acc);
  }
  acc = grp8_sum(acc);
  const float d = (L[LV8 + PP * 128 + v] - dec * acc) * beta;
  float acc2 = 0.f;
  #pragma unroll
  for (int c = 0; c < 4; ++c) {
    const float4 k4 = kp[c ^ sw];
    const float4 q4 = qp[c ^ sw];
    s[4 * c] = lm_upd(s[4 * c], dec, k4.x, d);         acc2 = __fmaf_rn(s[4 * c], q4.x, acc2);
    s[4 * c + 1] = lm_upd(s[4 * c + 1], dec, k4.y, d); acc2 = __fmaf_rn(s[4 * c + 1], q4.y, acc2);
    s[4 * c + 2] = lm_upd(s[4 * c + 2], dec, k4.z, d); acc2 = __fmaf_rn(s[4 * c + 2], q4.z, acc2);
    s[4 * c + 3] = lm_upd(s[4 * c + 3], dec, k4.w, d); acc2 = __fmaf_rn(s[4 * c + 3], q4.w, acc2);
  }
  acc2 = grp8_sum(acc2);
  if (g == 0) L[LO8 + PP * 128 + v] = rbf(acc2);
}
__device__ __noinline__ void lmix_tri() {
  const bool xcmp = x_compact();
  extern __shared__ float sm[];
  const Ctx& cx = CX(); const Op& op = OP();
  const int h = threadIdx.x >> 9, lt = threadIdx.x & 511, lane = lt & 31, wl = lt >> 5;
  float* L0 = sm + WORK_OFF;
  float* L = L0 + h * LM_FLOATS(8);
  const int NG = cx.NG, flags = cx.flags, nunits = cx.B * 48;
  const int ua = blockIdx.x * 2 + h, uc = 2 * cx.nblocks + blockIdx.x;
  if (ua >= nunits) return;                   // the block has no unit (both halves or none: nunits is even)
  const bool hasC = uc < nunits, tc = hasC && h == 0;
  const int role = (lt >> 7) & 3, j = lt & 127, jx = lsw(j >> 4, j & 15);
  // ---- inputs of the own unit (and, half 0, of the shared unit): conv + SiLU of q, k, v; z; decay and beta
  struct In { u16 xr[4]; float4 cw; u16 zv, bb, ab; };
  auto load = [&](int u, In& in) {
    const int b = u / 48, hv = u % 48, hk = hv / 3, pos0 = spos(b);
    const int ch = role == 0 ? hk * 128 + j : (role == 1 ? 2048 + hk * 128 + j : 4096 + hv * 128 + j);
    const u16* mb = cx.mbuf + (size_t)c_start(cx, b) * MROW;
    if (role < 3) {
      const u16* ring = (const u16*)op.p5 + (size_t)b * RING * 10240 + ch;
      in.cw = *((const float4*)op.p6 + ch);
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int pp = i - 3;
        u16 xv = 0;
        if (pp >= 0) xv = ldv(mb + ch);
        else if (pos0 + pp >= 0) xv = ldv(ring + (size_t)((pos0 + pp) & (RING - 1)) * 10240);
        in.xr[i] = xv;
      }
    } else {
      in.zv = ldv(mb + 10240 + hv * 128 + j);
      if (j == 0) { in.bb = ldv(mb + 16384 + hv); in.ab = ldv(mb + 16432 + hv); }
    }
  };
  auto calc = [&](int u, const In& in, float* A, int P) {   // A: the slot's area, P: its position
    const int b = u / 48, hv = u % 48, hk = hv / 3, pos0 = spos(b);
    if (role < 3) {
      const int ch = role == 0 ? hk * 128 + j : (role == 1 ? 2048 + hk * 128 + j : 4096 + hv * 128 + j);
      const bool wr = role == 2 || hv % 3 == 0;
      u16* ring = (u16*)op.p5 + (size_t)b * RING * 10240 + ch;
      if (wr) ring[(size_t)(pos0 & (RING - 1)) * 10240] = in.xr[3];
      float y = in.cw.w * bf2f(in.xr[3]);
      y += in.cw.z * bf2f(in.xr[2]);
      y += in.cw.y * bf2f(in.xr[1]);
      y += in.cw.x * bf2f(in.xr[0]);
      const float val = rbf(siluf_(rbf(y)));
      if (role == 0) A[LQ8 + P * 8 * LKS + jx] = val;
      else if (role == 1) A[LK8 + P * 8 * LKS + jx] = val;
      else A[LV8 + P * 128 + j] = val;
      const float ssum = warp_sum(val * val);
      if (lane == 0 && role < 2) A[LR8 + P * 8 + role * 4 + (wl & 3)] = ssum;
    } else {
      A[LZ8 + P * 128 + j] = bf2f(in.zv);
      if (j == 0) {
        const float* negA = (const float*)op.p7; const float* dtb = (const float*)op.p8;
        const float aa = bf2f(in.ab) + dtb[hv];
        const float sp = aa > 20.f ? aa : log1pf(__expf(aa));
        A[LS8 + 2 * P] = rbf(sigmoidf_(bf2f(in.bb)));
        A[LS8 + 2 * P + 1] = __expf(negA[hv] * sp);
      }
    }
  };
  auto pf_state = [&](int u) {
    if (spos(u / 48) <= 0) return;
    const long long sc = spar(u / 48) ? op.M : 0;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], 65536;" :: "l"((const float*)op.p4 + sc + ((size_t)u << 14)) : "memory");
  };
  {
    In ia, ic;
    load(ua, ia);
    if (tc) load(uc, ic);
    if (lt == 32) { pf_state(ua); if (tc) pf_state(uc); }   // after the input loads (findings S3); asked for at the
                                                            // barrier before the op instead: +18 to +28 us per B=8 step
    calc(ua, ia, L, 0);
    if (tc) calc(uc, ic, L0, 1);
  }
  __syncthreads();
  if (role < 2) {   // L2-normalise q and k
    {
      const float* r2 = L + LR8 + role * 4;
      const float inv = rsqrtf((r2[0] + r2[1]) + (r2[2] + r2[3]) + 1e-6f);
      float* x = L + (role == 0 ? LQ8 : LK8) + jx;
      *x = role == 0 ? *x * inv * 0.08838834764831845f : *x * inv;
    }
    if (tc) {
      const float* r2 = L0 + LR8 + 8 + role * 4;
      const float inv = rsqrtf((r2[0] + r2[1]) + (r2[2] + r2[3]) + 1e-6f);
      float* x = L0 + (role == 0 ? LQ8 : LK8) + 8 * LKS + jx;
      *x = role == 0 ? *x * inv * 0.08838834764831845f : *x * inv;
    }
  }
  __syncthreads();
  // ---- recurrence: the own unit's columns v and v + 64, then (blocks with a third unit) one column of the shared unit
  {
    const int g = lt & 7;
#ifdef LM_TRI_SEQ   // one column after the other, each with its own load -> arithmetic -> store (as lmix_dual)
    {
      const int b = ua / 48;
      float* S0 = (float*)op.p4 + (spar(b) ? op.M : 0) + ((size_t)ua << 14);
      const bool ld = spos(b) > 0;
      lm_tri_col<0>(L, S0 + wl * 512 + lane, g, lt >> 3, ld, flags);
      lm_tri_col<0>(L, S0 + (wl + 16) * 512 + lane, g, (lt >> 3) + 64, ld, flags);
    }
    if (hasC) {
      const int b = uc / 48;
      float* S0 = (float*)op.p4 + (spar(b) ? op.M : 0) + ((size_t)uc << 14);
      lm_tri_col<1>(L0, S0 + (wl + 16 * h) * 512 + lane, g, (lt >> 3) + 64 * h, spos(b) > 0, flags);
    }
#else
    // the next column's 16 state loads are in flight while the current column is computed (each column pass was a
    // load round trip of ~1.5 us followed by its arithmetic: findings S5, S8)
    {
      const int ba = ua / 48, bc = uc / 48;
      float* Sa = (float*)op.p4 + (spar(ba) ? op.M : 0) + ((size_t)ua << 14);
      float* S1 = Sa + wl * 512 + lane;
      float* S2 = Sa + (wl + 16) * 512 + lane;
      float* S3 = hasC ? (float*)op.p4 + (spar(bc) ? op.M : 0) + ((size_t)uc << 14) + (wl + 16 * h) * 512 + lane : S1;
      const bool lda = spos(ba) > 0 && !(flags & 32), ldc = hasC && spos(bc) > 0 && !(flags & 32);
      float sa[16], sb[16];
      #pragma unroll
      for (int i = 0; i < 16; ++i) sa[i] = lda ? ld_state(S1 + i * 32) : 0.f;
      #pragma unroll
      for (int i = 0; i < 16; ++i) sb[i] = lda ? ld_state(S2 + i * 32) : 0.f;
      lm_tri_core<0>(L, sa, g, lt >> 3);
      if (!(flags & 32)) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) st_state(S1 + i * 32, sa[i]);
      }
      #pragma unroll
      for (int i = 0; i < 16; ++i) sa[i] = ldc ? ld_state(S3 + i * 32) : 0.f;   // the shared unit's column
      lm_tri_core<0>(L, sb, g, (lt >> 3) + 64);
      if (!(flags & 32)) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) st_state(S2 + i * 32, sb[i]);
      }
      if (hasC) {
        lm_tri_core<1>(L0, sa, g, (lt >> 3) + 64 * h);
        if (!(flags & 32)) {
          #pragma unroll
          for (int i = 0; i < 16; ++i) st_state(S3 + i * 32, sa[i]);
        }
      }
    }
#endif
  }
  __syncthreads();
  // ---- gated RMSNorm and the out_proj input: own unit on threads 0-127 of the half, shared unit on 128-255 of half 0
  {
    const float* nwt = (const float*)op.p9;
    u16* xn = (u16*)op.xf;
    const int vv = lt & 127;
    const bool on = lt < 128, onc = tc && lt >= 128 && lt < 256;
    float ob = 0.f;
    if (on) ob = L[LO8 + vv];
    else if (onc) ob = L0[LO8 + 128 + vv];
    const float so = warp_sum(ob * ob);
    if (lane == 0) {
      if (on) L[LR8 + wl] = so;
      else if (onc) L0[LR8 + 8 + (wl - 4)] = so;
    }
    __syncthreads();
    if (on || onc) {
      const float* r2 = on ? L + LR8 : L0 + LR8 + 8;
      const float rs = rsqrtf(((r2[0] + r2[1]) + (r2[2] + r2[3])) * (1.0f / 128.0f) + 1e-6f);
      const float n = rbf(ob * rs);
      const float mm = rbf(nwt[vv] * n);
      const int u = on ? ua : uc;
      const float z = on ? L[LZ8 + vv] : L0[LZ8 + 128 + vv];
      xn[xg_idx(c_start(cx, u / 48), (u % 48) * 128 + vv, NG, xcmp)] = f2bf(mm * siluf_(z));
    }
  }
}

template <int PM, bool split, bool VIN>   // VIN: inputs 4 channels per thread (lm_inputs)
__device__ __noinline__ void lmix_work() {
  const bool xcmp = x_compact();   // hoisted: the producers write the compact layout at NC = 1
  extern __shared__ float sm[];
  const Ctx& cx = CX(); const Op& op = OP();
  float* L = sm + WORK_OFF;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  const int NG = cx.NG;
  const bool spec = (cx.flags & F_SPEC) != 0;
  // split (one sequence, long passes): two blocks per head, 64 state columns each, so the recurrence (issue-bound,
  // ~160 instructions per position per warp) runs on 96 SMs instead of 48; every block of a unit is resident
  const int nunits = cx.B * 48 * (split ? 2 : 1);
#ifdef LMIX_PROF
  unsigned long long lmt = 0;
#endif
  LMP(0);
  // per-sequence layouts carry a table of the units of each block (longest first, balanced by the host)
  const int* ut = (!split && cx.cst) ? cx.cst + 32 : nullptr;
  uint4 utv = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
  if (ut) utv = ldv((const uint4*)(ut + blockIdx.x * 4));   // the block's 4 entries in one load
  auto unit_k = [&](int k) -> int {
    if (ut) return k == 0 ? (int)utv.x : k == 1 ? (int)utv.y : k == 2 ? (int)utv.z : k == 3 ? (int)utv.w : -1;
    const int uu = blockIdx.x + k * cx.nblocks;
    return uu < nunits ? uu : -1;
  };
  // the states of all this block's units (read after their input phases): L2 is asked to fetch them (thread 32)
  auto pf_states = [&]() {
    for (int k = 0;; ++k) {
      const int uu = unit_k(k);
      if (uu < 0) break;
      const int u = split ? uu >> 1 : uu, part = split ? uu & 1 : -1;
      if (spos(u / 48) <= 0) continue;
      const long long sc = spar(u / 48) ? op.M : 0;
      const char* sa = (const char*)((const float*)op.p4 + sc + ((size_t)u << 14)) + (part > 0 ? 32768 : 0);
      if (part < 0) asm volatile("cp.async.bulk.prefetch.L2.global [%0], 65536;" :: "l"(sa) : "memory");
      else asm volatile("cp.async.bulk.prefetch.L2.global [%0], 32768;" :: "l"(sa) : "memory");
    }
  };
#ifdef LMPF2_LATE
  if (tid == 32 && VIN) pf_states();
#else
  if (tid == 32) pf_states();
#endif
  for (int k = 0;; ++k) {
    const int uu = unit_k(k);
    if (uu < 0) break;
    const int u = split ? uu >> 1 : uu, part = split ? uu & 1 : -1;
    const int b = u / 48, hv = u % 48, hk = hv / 3;
    const int P = c_len(cx, b), c0 = c_start(cx, b);   // this sequence's positions and first column
    const int pos0 = spos(b);
    const bool defer = (cx.flags & F_DEFER) != 0;
    const int na = (spec || defer) ? sna(b) : 0;   // positions to replay (0 when the whole last pass was accepted)
    const bool keep = defer && P == 1 && na + 1 < cx.defer_n;   // this step saves its update in slot na instead of storing the state
    const int spar = ::spar(b);
    const long long sc = spar ? op.M : 0, so = spar ? 0 : op.M;
    LMP(1);
    // 1. conv + SiLU of q, k, v for every position, z, decay and beta (lm_inputs); the k of the previous pass's
    //    accepted positions are staged (core-output area) for the replay
#ifndef LM_INPUTS_OLD
    if (VIN) {   // 4 channels per thread: -4 % per 32-position pass; with 1-3 positions too few threads (+1 %)
      lm_inputs<PM>(u, b, hv, hk, P, c0, pos0, na, part);
      __syncthreads();
    } else
#endif
    {
      float* pk = (float*)op.p10 + (size_t)u * PMAX * 128;
      #pragma unroll
      for (int i = 0; i < PM / 8; ++i) {   // na <= PM: all loads in flight together
        const int idx = tid + 1024 * i;
        if (idx < na * 128) { const int j = idx >> 7, k = idx & 127; L[LO + j * 8 * LKS + lsw(k >> 4, k & 15)] = ldv(pk + idx); }
      }
      const float* negA = (const float*)op.p7; const float* dtb = (const float*)op.p8;
      const int half = tid >> 9, role = (tid >> 7) & 3, j = tid & 127;
      const int ch = role == 0 ? hk * 128 + j : (role == 1 ? 2048 + hk * 128 + j : 4096 + hv * 128 + j);
      const bool wr = (role == 2 || hv % 3 == 0) && part <= 0;   // both blocks of a split head compute the inputs
      u16* ring = (u16*)op.p5 + (size_t)b * RING * 10240 + ch;
      const u16* mb = cx.mbuf + (size_t)c0 * MROW;
      const int jx = lsw(j >> 4, j & 15);
      // each half of the block takes a contiguous run of positions; every input of the run is requested at once
      constexpr int H = PM / 2;
      const int ph = (P + 1) >> 1, p_lo = half * ph, p_hi = min(P, p_lo + ph);
      if (role < 3) {
        const float4 cw = *((const float4*)op.p6 + ch);
        u16 xr[H + 3];
        #pragma unroll
        for (int i = 0; i < H + 3; ++i) {
          const int pp = p_lo - 3 + i;
          u16 xv = 0;
          if (pp < p_hi) {
            if (pp >= 0) xv = ldv(mb + (size_t)pp * MROW + ch);
            else if (pos0 + pp >= 0) xv = ldv(ring + (size_t)((pos0 + pp) & (RING - 1)) * 10240);
          }
          xr[i] = xv;
        }
#ifdef LMPF2_LATE   // after the first unit's input loads (findings S3)
        if (tid == 32 && k == 0) pf_states();
#endif
        #pragma unroll
        for (int i = 0; i < H; ++i) {
          const int p = p_lo + i;
          if (p >= p_hi) break;
          if (wr) ring[(size_t)((pos0 + p) & (RING - 1)) * 10240] = xr[i + 3];
          float y = cw.w * bf2f(xr[i + 3]);
          y += cw.z * bf2f(xr[i + 2]);
          y += cw.y * bf2f(xr[i + 1]);
          y += cw.x * bf2f(xr[i]);
          const float val = rbf(siluf_(rbf(y)));
          if (role == 0) L[LQ + p * 8 * LKS + jx] = val;
          else if (role == 1) L[LK + p * 8 * LKS + jx] = val;
          else L[LV + p * 128 + j] = val;
          const float ssum = warp_sum(val * val);
          if (lane == 0 && role < 2) L[LR + p * 8 + role * 4 + ((tid >> 5) & 3)] = ssum;
        }
      } else {
        u16 zr[H];
        #pragma unroll
        for (int i = 0; i < H; ++i) zr[i] = p_lo + i < p_hi ? ldv(mb + (size_t)(p_lo + i) * MROW + 10240 + hv * 128 + j) : (u16)0;
        #pragma unroll
        for (int i = 0; i < H; ++i) if (p_lo + i < p_hi) L[LZ + (p_lo + i) * 128 + j] = bf2f(zr[i]);
        if (j < H && p_lo + j < p_hi) {   // decay and beta: one position per thread
          const u16* m = mb + (size_t)(p_lo + j) * MROW;
          const u16 bb = ldv(m + 16384 + hv), ab = ldv(m + 16432 + hv);
          const float aa = bf2f(ab) + dtb[hv];
          const float sp = aa > 20.f ? aa : log1pf(__expf(aa));
          L[LS + (p_lo + j) * 2] = rbf(sigmoidf_(bf2f(bb)));
          L[LS + (p_lo + j) * 2 + 1] = __expf(negA[hv] * sp);
        }
      }
      __syncthreads();
      if (role < 2) {   // L2-normalise q and k
        for (int p = p_lo; p < p_hi; ++p) {
          const float* r2 = L + LR + p * 8 + role * 4;
          const float inv = rsqrtf((r2[0] + r2[1]) + (r2[2] + r2[3]) + 1e-6f);
          float* x = L + (role == 0 ? LQ : LK) + p * 8 * LKS + jx;
          const float nv = role == 0 ? *x * inv * 0.08838834764831845f : *x * inv;
          *x = nv;
          if (spec && role == 1 && part <= 0) pk[p * 128 + j] = nv;
          if (keep && role == 1 && part <= 0) pk[na * 128 + j] = nv;
        }
      }
      __syncthreads();
    }
    LMP(2);
    // 2. replay, recurrence (no block barrier per position), state store
    if (!split) lm_core<PM, -1>(u, P, pos0, na, cx.flags, sc, so);
    else if (part == 0) lm_core<PM, 0>(u, P, pos0, na, cx.flags, sc, so);
    else lm_core<PM, 1>(u, P, pos0, na, cx.flags, sc, so);
    __syncthreads();
    if (spec) {   // the deltas (left over v in the core) and decays of every position, for the next pass's replay
      float* pd = (float*)OP().p11 + (size_t)u * PMAX * 128;
      const int cl0 = split ? 64 * part : 0, nc = split ? 64 : 128;
      for (int i = tid; i < P * nc; i += 1024) { const int p = i / nc, v = cl0 + (i - p * nc); pd[p * 128 + v] = L[LV + p * 128 + v]; }
      if (part <= 0 && tid < P) ((float*)OP().W + (size_t)u * PMAX)[tid] = L[LS + tid * 2 + 1];
    }
    if (keep) {   // deferred mode: this step's delta and decay go to slot na (k went there in the input phase)
      float* pd = (float*)OP().p11 + (size_t)u * PMAX * 128 + na * 128;
      const int cl0 = split ? 64 * part : 0, nc = split ? 64 : 128;
      if (tid < nc) pd[cl0 + tid] = L[LV + cl0 + tid];
      if (part <= 0 && tid == 0) ((float*)OP().W + (size_t)u * PMAX)[na] = L[LS + 1];
    }
    LMP(3);
    if (cx.flags & 512) continue;
    // 3. gated RMSNorm per position (8 positions per round) and the out_proj input
    const float* nwt = (const float*)OP().p9;
    u16* xn = (u16*)OP().xf;
    if (part >= 0) {   // the norm's sum of squares spans both blocks: the 32-column warp sums of the partner block
      // arrive through sentinel slots polled with relaxed gpu-scope loads (one memory trip; findings C11); a block
      // keeps its own sums in shared memory, so a slot is read only by the partner, which restores the sentinel
      float* xs = (float*)OP().out + (size_t)hv * PMAX * 4;   // [position][group k = column / 32]
      float* own = L + LR;                                    // [position][own group 0 / 1]
      for (int p0 = 0; p0 < P; p0 += 16) {
        const int p = p0 + (tid >> 6), vv = 64 * part + (tid & 63);
        const float ob = p < P ? L[LO + p * 128 + vv] : 0.f;
        const float so = warp_sum(ob * ob);
        if (lane == 0 && p < P) { st_rlx(xs + p * 4 + (vv >> 5), so); own[p * 2 + ((vv >> 5) & 1)] = so; }
      }
      __syncthreads();
      for (int p0 = 0; p0 < P; p0 += 16) {
        const int p = p0 + (tid >> 6), vv = 64 * part + (tid & 63);
        if (p < P) {
          float a[4], q0, q1;
          const float* ps2 = xs + p * 4 + 2 * (1 - part);
          do { q0 = ld_rlx(ps2); } while (__float_as_uint(q0) == SENT_BITS);
          do { q1 = ld_rlx(ps2 + 1); } while (__float_as_uint(q1) == SENT_BITS);
          a[2 * part] = own[p * 2]; a[2 * part + 1] = own[p * 2 + 1];
          a[2 * (1 - part)] = q0; a[2 * (1 - part) + 1] = q1;
          const float rs = rsqrtf(((a[0] + a[1]) + (a[2] + a[3])) * (1.0f / 128.0f) + 1e-6f);
          const float n = rbf(L[LO + p * 128 + vv] * rs);
          const float mm = rbf(nwt[vv] * n);
          xn[xg_idx(c0 + p, hv * 128 + vv, NG, xcmp)] = f2bf(mm * siluf_(L[LZ + p * 128 + vv]));
        }
      }
      __syncthreads();   // every read of the partner's slots is done: restore their sentinels
      if (tid < 2 * P) st_rlx(xs + (tid >> 1) * 4 + 2 * (1 - part) + (tid & 1), __uint_as_float(SENT_BITS));
      LMP(4);
      continue;
    }
    for (int p0 = 0; p0 < P; p0 += 8) {
      const int p = p0 + (tid >> 7), vv = tid & 127;
      const float ob = p < P ? L[LO + p * 128 + vv] : 0.f;
      const float so = warp_sum(ob * ob);
      if (lane == 0) L[LR + wid] = so;
      __syncthreads();
      if (p < P) {
        const float* r2 = L + LR + (tid >> 7) * 4;
        const float rs = rsqrtf(((r2[0] + r2[1]) + (r2[2] + r2[3])) * (1.0f / 128.0f) + 1e-6f);
        const float n = rbf(ob * rs);
        const float mm = rbf(nwt[vv] * n);
        xn[xg_idx(c0 + p, hv * 128 + vv, NG, xcmp)] = f2bf(mm * siluf_(L[LZ + p * 128 + vv]));
      }
      __syncthreads();
    }
    LMP(4);
  }
}

// ------------------------------------------------------------------------------------------------ full attention
// K cache: per (b, kv head): [key tile][16 ksteps][512 B] in A-operand order (rows = keys, cols = dims)
// V cache: per (b, kv head): [dim tile (16)][key group][512 B] in A-operand order (rows = dims, cols = keys)
// Attention units go round-robin over blocks first (unit u: warp u / nblocks of block u % nblocks): with
// consecutive warps a B=1 step put its 152 key tiles on 5 SMs, whose L2 bandwidth bounded the op
__device__ __forceinline__ int att_w() {
#ifdef ATT_LINEAR
  return blockIdx.x * 32 + (threadIdx.x >> 5);
#else
  return (tid_v() >> 5) * CX().nblocks + cta_v();
#endif
}
__device__ __noinline__ void att_prep_work() {
  const Ctx& cx = CX(); const Op& op = OP();
  const int lane = threadIdx.x & 31;
  const int w = att_w();
  const int nunits = cx.NC * 32, P = cx.P, NGq = cx.NGq, mc = cx.max_ctx;
  const float* gq = (const float*)op.p6; const float* gk = (const float*)op.p7;
  for (int u = w; u < nunits; u += cx.nwarps) {
    const int c = u >> 5, j = u & 31;
    const int b = c_seq(cx, c), p = c - c_start(cx, b);
    const int Tpos = spos(b) + cx.pos_off + p;
    const u16* m = cx.mbuf + (size_t)c * MROW;
    const int kind = j < 24 ? 0 : (j < 28 ? 1 : 2);
    const int hd = j < 24 ? j : (j - 24) & 3;
    const u16* src = m + (kind == 0 ? hd * 512 : (kind == 1 ? 12288 + hd * 256 : 13312 + hd * 256));
    u16 raw[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) raw[e] = ldv(src + lane + 32 * e);
    float x[8];
    float ss = 0.f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) { x[e] = bf2f(raw[e]); ss += x[e] * x[e]; }
    ss = warp_sum(ss);
    if (kind < 2) {
      const float rs = rsqrtf(ss * (1.0f / 256.0f) + 1e-6f);
      const float* gg = kind == 0 ? gq : gk;
      #pragma unroll
      for (int e = 0; e < 8; ++e) x[e] = rbf(x[e] * rs * gg[lane + 32 * e]);
      const float cs = cx.rope_cos[(size_t)Tpos * 32 + lane], sn = cx.rope_sin[(size_t)Tpos * 32 + lane];
      const float x0 = rbf(rbf(x[0] * cs) + rbf(-x[1] * sn));
      const float x1 = rbf(rbf(x[1] * cs) + rbf(x[0] * sn));
      x[0] = x0; x[1] = x1;
    }
    if (kind == 0) {
      const int kh = hd / 6, qq = p * 6 + hd % 6;
      u16* qf = cx.qf + (size_t)(b * 4 + kh) * (16 * NGq * 128);
      #pragma unroll
      for (int e = 0; e < 8; ++e) qf[xf_idx(qq, lane + 32 * e, NGq)] = f2bf(x[e]);
    } else if (kind == 1) {
      u16* Kc = (u16*)op.p4 + (size_t)(b * 4 + hd) * mc * 256;
      const int kt = Tpos >> 4, r = Tpos & 15;
      #pragma unroll
      for (int e = 0; e < 8; ++e) { const int i = lane + 32 * e; Kc[((size_t)kt * 16 + (i >> 4)) * 256 + af_idx(r, i & 15)] = f2bf(x[e]); }
    } else {
      u16* Vc = (u16*)op.p5 + (size_t)(b * 4 + hd) * mc * 256;
      const int kg = Tpos >> 4, kk = Tpos & 15;
      #pragma unroll
      for (int e = 0; e < 8; ++e) { const int i = lane + 32 * e; Vc[((size_t)(i >> 4) * (mc >> 4) + kg) * 256 + af_idx(i & 15, kk)] = f2bf(x[e]); }
    }
  }
}

__device__ __noinline__ void att_score_work() {
  const Ctx& cx = CX(); const Op& op = OP();
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int w = att_w();
  const int P = cx.P, NGq = cx.NGq, mc = cx.max_ctx;
  const int nkt = (cx.pos_ub + P + 15) >> 4;
  const int nunits = cx.B * 4 * nkt * NGq;   // (sequence, kv head, key tile, column group): one mma chain each
  for (int u = w; u < nunits; u += cx.nwarps) {
    const int cg = u % NGq, ub = u / NGq;
    const int bk = ub / nkt, kt = ub % nkt, b = bk >> 2, kh = bk & 3;
    const int pos0 = spos(b) + cx.pos_off, Pb = c_len(cx, b);
    if (kt * 16 > pos0 + Pb - 1) continue;
    const uint4* Kc = (const uint4*)op.p4 + ((size_t)bk * mc * 256 + (size_t)kt * 16 * 256) / 8 + lane;
    const uint2* qf = (const uint2*)cx.qf + (size_t)bk * (16 * NGq * 32) + lane;
    {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int s0 = 0; s0 < 16; s0 += 4) {
        uint4 a[4]; uint2 bb[4];
        #pragma unroll
        for (int s = 0; s < 4; ++s) { a[s] = ld_kv(Kc + (s0 + s) * 32); bb[s] = ldv(qf + ((s0 + s) * NGq + cg) * 32); }
        #pragma unroll
        for (int s = 0; s < 4; ++s) mma(c, a[s], bb[s]);
      }
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int qq = cg * 8 + 2 * t + i;
        if (qq < 6 * Pb) {
          const int p = qq / 6, hh = qq % 6;
          float* sc = cx.sc + ((size_t)(b * 24 + kh * 6 + hh) * P + p) * mc + kt * 16 + g;
          sc[0] = c[i] * 0.0625f;
          sc[8] = c[2 + i] * 0.0625f;
        }
      }
    }
  }
}

__device__ __noinline__ void att_softmax_work() {
  const Ctx& cx = CX();
  const int lane = threadIdx.x & 31;
  const int w = blockIdx.x * 32 + (threadIdx.x >> 5);
  const int P = cx.P, NGq = cx.NGq, mc = cx.max_ctx;
  const int nunits = cx.B * 24 * P;
  for (int u = w; u < nunits; u += cx.nwarps) {
    const int p = u % P, bh = u / P, b = bh / 24, h = bh % 24, kh = h / 6, hh = h % 6;
    const int Pb = c_len(cx, b);
    if (p >= Pb) continue;
    const int pos0 = spos(b) + cx.pos_off;
    const int n = pos0 + p + 1, nend = ((pos0 + Pb + 15) >> 4) << 4;
    const float* sc = cx.sc + (size_t)u * mc;
    u16* pf = cx.pf + (size_t)(b * 4 + kh) * ((mc >> 4) * NGq * 128);
    const int qq = p * 6 + hh;
    // pass 1: max, 8 loads in flight
    float mx = -3.0e38f;
    for (int j = lane; j < n; j += 256) {
      float v[8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) v[e] = (j + 32 * e < n) ? ldv(sc + j + 32 * e) : -3.0e38f;
      #pragma unroll
      for (int e = 0; e < 8; ++e) mx = fmaxf(mx, v[e]);
    }
    #pragma unroll
    for (int sh = 16; sh >= 1; sh >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, sh));
    // pass 2: sum of exp (per lane in key order, then the fixed lane tree)
    float sum = 0.f;
    for (int j = lane; j < n; j += 256) {
      float v[8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) v[e] = (j + 32 * e < n) ? ldv(sc + j + 32 * e) : -3.0e38f;
      #pragma unroll
      for (int e = 0; e < 8; ++e) if (j + 32 * e < n) sum += __expf(v[e] - mx);
    }
    sum = warp_sum(sum);
    const float inv = 1.0f / sum;
    // pass 3: probabilities in bf16, zeros for the keys this position must not see
    for (int j = lane; j < nend; j += 256) {
      float v[8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) v[e] = (j + 32 * e < n) ? ldv(sc + j + 32 * e) : -3.0e38f;
      #pragma unroll
      for (int e = 0; e < 8; ++e)
        if (j + 32 * e < nend) pf[xf_idx(qq, j + 32 * e, NGq)] = (j + 32 * e < n) ? f2bf(__expf(v[e] - mx) * inv) : (u16)0;
    }
  }
}

// Four warps per row (8 rows per block): warp q of a row takes the 32-key chunks q, q + 4, ... (one batch of loads
// per pass for rows up to ~1000 keys instead of three dependent batches); max over the four partial maxima, sum as
// (s0 + s1) + (s2 + s3) of per-warp lane trees of per-lane sums in key order - the same structure at every batch
// size and position count.
__device__ __noinline__ void att_softmax4_work() {
  extern __shared__ float sm[];
  const Ctx& cx = CX();
  const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5, rl = wb >> 2, q = wb & 3;
  const int P = cx.P, NGq = cx.NGq, mc = cx.max_ctx;
  const int nunits = cx.B * 24 * P;
  float* part = sm + WORK_OFF;   // [8 rows][max 4, sum 4]
  for (int base = blockIdx.x * 8; base < nunits; base += cx.nblocks * 8) {
    const int u = base + rl;
    bool ok = u < nunits;
    int p = 0, b = 0, kh = 0, hh = 0, Pb = 0, n = 0, nend = 0;
    if (ok) {
      p = u % P; const int bh = u / P; b = bh / 24; const int h = bh % 24; kh = h / 6; hh = h % 6;
      Pb = c_len(cx, b);
      ok = p < Pb;
      const int pos0 = spos(b) + cx.pos_off;
      n = pos0 + p + 1; nend = ((pos0 + Pb + 15) >> 4) << 4;
    }
    const float* sc = cx.sc + (size_t)(ok ? u : 0) * mc;
    // rows of at most 1024 keys (one batch of 8 loads per lane): the scores stay in registers for all three passes
    const bool one = nend <= 1024;
    float vr[8];
    if (ok && one) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) { const int j = (q + 4 * e) * 32 + lane; vr[e] = j < n ? ldv(sc + j) : -3.0e38f; }
    }
    float mx = -3.0e38f;
    if (ok && one) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) mx = fmaxf(mx, vr[e]);
    } else if (ok) {
      for (int c0 = q; c0 * 32 < n; c0 += 32) {
        float v[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) { const int j = (c0 + 4 * e) * 32 + lane; v[e] = j < n ? ldv(sc + j) : -3.0e38f; }
        #pragma unroll
        for (int e = 0; e < 8; ++e) mx = fmaxf(mx, v[e]);
      }
    }
    #pragma unroll
    for (int sh = 16; sh >= 1; sh >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, sh));
    if (lane == 0) part[rl * 8 + q] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(part[rl * 8], part[rl * 8 + 1]), fmaxf(part[rl * 8 + 2], part[rl * 8 + 3]));
    float sum = 0.f;
    if (ok && one) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) if ((q + 4 * e) * 32 + lane < n) sum += __expf(vr[e] - mx);
    } else if (ok) {
      for (int c0 = q; c0 * 32 < n; c0 += 32) {
        float v[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) { const int j = (c0 + 4 * e) * 32 + lane; v[e] = j < n ? ldv(sc + j) : -3.0e38f; }
        #pragma unroll
        for (int e = 0; e < 8; ++e) if ((c0 + 4 * e) * 32 + lane < n) sum += __expf(v[e] - mx);
      }
    }
    sum = warp_sum(sum);
    if (lane == 0) part[rl * 8 + 4 + q] = sum;
    __syncthreads();
    const float inv = 1.0f / ((part[rl * 8 + 4] + part[rl * 8 + 5]) + (part[rl * 8 + 6] + part[rl * 8 + 7]));
    if (ok && one) {
      u16* pf = cx.pf + (size_t)(b * 4 + kh) * ((mc >> 4) * NGq * 128);
      const int qq = p * 6 + hh;
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int j = (q + 4 * e) * 32 + lane;
        if (j < nend) pf[xf_idx(qq, j, NGq)] = j < n ? f2bf(__expf(vr[e] - mx) * inv) : (u16)0;
      }
    } else if (ok) {
      u16* pf = cx.pf + (size_t)(b * 4 + kh) * ((mc >> 4) * NGq * 128);
      const int qq = p * 6 + hh;
      for (int c0 = q; c0 * 32 < nend; c0 += 32) {
        float v[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) { const int j = (c0 + 4 * e) * 32 + lane; v[e] = j < n ? ldv(sc + j) : -3.0e38f; }
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int j = (c0 + 4 * e) * 32 + lane;
          if (j < nend) pf[xf_idx(qq, j, NGq)] = j < n ? f2bf(__expf(v[e] - mx) * inv) : (u16)0;
        }
      }
    }
    __syncthreads();   // the partials of this round are read before the next round writes them
  }
}

// One output unit (sequence, kv head, dim tile, column group): sum of the value-sum chunks in chunk order, gate.
__device__ __noinline__ void att_out_unit(int uc, int nchb) {
  const bool xcmp = x_compact();
  const Ctx& cx = CX(); const Op& op = OP();
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int NGq = cx.NGq, mc = cx.max_ctx;
  u16* xn = (u16*)op.xf;
  const int cg = uc % NGq, u = uc / NGq;
  const int dt = u & 15, bk = u >> 4, b = bk >> 2, kh = bk & 3;
  const int Pb = c_len(cx, b), cb0 = c_start(cx, b);
  u16 gt[2][2];   // the gates first: their load does not wait behind the partial sums'
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int qq = cg * 8 + 2 * t + i;
    gt[i][0] = 0; gt[i][1] = 0;
    if (qq < 6 * Pb) {
      const int p = qq / 6, h = kh * 6 + qq % 6, col = cb0 + p;
      const u16* m = cx.mbuf + (size_t)col * MROW + h * 512 + 256;
      gt[i][0] = ldv(m + dt * 16 + g); gt[i][1] = ldv(m + dt * 16 + g + 8);
    }
  }
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  for (int ch = 0; ch < nchb; ch += 4) {
    float4 v[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e)
      if (ch + e < nchb) v[e] = ldv(cx.vpart + (((size_t)u * (mc >> 8) + ch + e) * NGq + cg) * 32 + lane);
    #pragma unroll
    for (int e = 0; e < 4; ++e)
      if (ch + e < nchb) { c[0] += v[e].x; c[1] += v[e].y; c[2] += v[e].z; c[3] += v[e].w; }
  }
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int qq = cg * 8 + 2 * t + i;
    if (qq < 6 * Pb) {
      const int p = qq / 6, h = kh * 6 + qq % 6, col = cb0 + p;
      const u16* m = cx.mbuf + (size_t)col * MROW + h * 512 + 256;
      const int d0 = dt * 16 + g, d1 = d0 + 8;
      (void)m;
      xn[xg_idx(col, h * 256 + d0, cx.NG, xcmp)] = f2bf(rbf(c[i]) * rbf(sigmoidf_(bf2f(gt[i][0]))));
      xn[xg_idx(col, h * 256 + d1, cx.NG, xcmp)] = f2bf(rbf(c[2 + i]) * rbf(sigmoidf_(bf2f(gt[i][1]))));
    }
  }
}

__device__ __noinline__ void att_vsum_work() {
  const Ctx& cx = CX(); const Op& op = OP();
  const int lane = threadIdx.x & 31;
  const int w = att_w();
  const int P = cx.P, NGq = cx.NGq, mc = cx.max_ctx;
  const int nch = (cx.pos_ub + P + 255) >> 8;
  const int nunits = cx.B * 4 * 16 * nch * NGq;   // (sequence, kv head, dim tile, key chunk, column group)
  for (int u = w; u < nunits; u += cx.nwarps) {
    const int cg = u % NGq, ub = u / NGq;
    const int ch = ub % nch, r = ub / nch, dt = r & 15, bk = r >> 4, b = bk >> 2;
    const int pos0 = spos(b) + cx.pos_off, Pb = c_len(cx, b);
    if (ch * 256 > pos0 + Pb - 1) continue;
    const int nkg = (pos0 + Pb + 15) >> 4;
    const uint4* Vc = (const uint4*)op.p5 + ((size_t)bk * mc * 256 + ((size_t)dt * (mc >> 4) + ch * 16) * 256) / 8 + lane;
    const uint2* pf = (const uint2*)cx.pf + (size_t)bk * ((mc >> 4) * NGq * 32) + (size_t)ch * 16 * NGq * 32 + lane;
    {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      for (int s0 = 0; s0 < 16 && ch * 16 + s0 < nkg; s0 += 4) {
        uint4 a[4]; uint2 bb[4];
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
          const bool on = ch * 16 + s0 + s < nkg;
          a[s] = on ? ld_kv(Vc + (s0 + s) * 32) : make_uint4(0, 0, 0, 0);
          bb[s] = on ? ldv(pf + ((s0 + s) * NGq + cg) * 32) : make_uint2(0, 0);
        }
        #pragma unroll
        for (int s = 0; s < 4; ++s) mma(c, a[s], bb[s]);
      }
      cx.vpart[(((size_t)r * (mc >> 8) + ch) * NGq + cg) * 32 + lane] = make_float4(c[0], c[1], c[2], c[3]);
      if (op.i6) {   // the old ATT_OUT, done by the last chunk to arrive (a counter per output unit): no extra op
        const int nchb = (pos0 + Pb + 255) >> 8, ci = r * NGq + cg;
        __syncwarp();
        int old = 0;
        if (lane == 0) old = atom_add_acqrel(cx.cnt + ci, 1);
        old = __shfl_sync(0xffffffffu, old, 0);
        __syncwarp();
        if (old + 1 == nchb) {
          if (lane == 0) cx.cnt[ci] = 0;
          att_out_unit(r * NGq + cg, nchb);
        }
      }
    }
  }
}

__device__ __noinline__ void att_out_work() {
  const bool xcmp = x_compact();   // hoisted: the producers write the compact layout at NC = 1
  const Ctx& cx = CX(); const Op& op = OP();
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int w = att_w();
  const int P = cx.P, NGq = cx.NGq, mc = cx.max_ctx;
  const int nunits = cx.B * 4 * 16 * NGq;   // (sequence, kv head, dim tile, column group)
  u16* xn = (u16*)op.xf;
  for (int uc = w; uc < nunits; uc += cx.nwarps) {
    const int cg = uc % NGq, u = uc / NGq;
    const int dt = u & 15, bk = u >> 4, b = bk >> 2, kh = bk & 3;
    const int pos0 = spos(b) + cx.pos_off, Pb = c_len(cx, b), cb0 = c_start(cx, b);
    const int nchb = (pos0 + Pb + 255) >> 8;
    {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      for (int ch = 0; ch < nchb; ch += 4) {
        float4 v[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e)
          if (ch + e < nchb) v[e] = ldv(cx.vpart + (((size_t)u * (mc >> 8) + ch + e) * NGq + cg) * 32 + lane);
        #pragma unroll
        for (int e = 0; e < 4; ++e)
          if (ch + e < nchb) { c[0] += v[e].x; c[1] += v[e].y; c[2] += v[e].z; c[3] += v[e].w; }
      }
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int qq = cg * 8 + 2 * t + i;
        if (qq < 6 * Pb) {
          const int p = qq / 6, h = kh * 6 + qq % 6, col = cb0 + p;
          const u16* m = cx.mbuf + (size_t)col * MROW + h * 512 + 256;
          const int d0 = dt * 16 + g, d1 = d0 + 8;
          const u16 g0 = ldv(m + d0), g1 = ldv(m + d1);
          xn[xg_idx(col, h * 256 + d0, cx.NG, xcmp)] = f2bf(rbf(c[i]) * rbf(sigmoidf_(bf2f(g0))));
          xn[xg_idx(col, h * 256 + d1, cx.NG, xcmp)] = f2bf(rbf(c[2 + i]) * rbf(sigmoidf_(bf2f(g1))));
        }
      }
    }
  }
}

// ---- OP_ATT_DIST: a full-attention layer of a one-token pass in one op, one block per 256-key chunk of each
// (sequence, kv head). A block per (sequence, kv head) would put the unit's whole K and V strips (0.6-1.1 MB) through one SM; here block
// (u, ch) = u * S + ch (S = blocks / units) streams one chunk: 16 key tiles of K and the 16 x 16 V tiles of those
// keys (131 KB each, whatever the context). Per block: the 6 query heads' fragments (shared memory); the last chunk's
// block appends K and V; scores of its tiles (kept locally and stored to the global score rows); the unit's first
// block (hub) waits for all chunks' scores, runs the softmax reductions of att_softmax4_work over the global rows and
// publishes (max, 1 / sum) per head; every block then makes the probability fragments of its own keys, the value
// sums of its chunk, and hands them to the hub, which adds them in chunk order and writes the gated output.
// Hand-overs between blocks (scores to the hub, (max, 1 / sum) back, value sums to the hub): slots that hold a sentinel
// between ops, written and polled with relaxed gpu-scope stores and loads, restored by their reader (as the GEMMs'
// cut tiles). With acquire/release counters instead the op took 20 us: each strong atomic costs ~3 us (findings S7).
// The arithmetic is the four ops' (same MMA chains, same softmax partition and reductions): bit-identical.
#define DQF 0            // q fragments [16 k-steps][32 lanes][8 B]
#define DPART 1024       // softmax partials [8 rows][max 4, sum 4]
#define DSC 1088         // this chunk's scores [6 heads][256]
#define DPF 2624         // this chunk's probability fragments [16 k-steps][32 lanes][8 B]
#define DMI 3648         // (max[6], 1 / sum[6]) of the unit's rows
#define ATT_DIST_FLOATS 3664
// relaxed gpu-scope accesses: a store and a later load of the SAME location are ordered (findings C11 of s4, S7)
__device__ __forceinline__ uint4 ld_rlx_u4(const uint4* p) { uint4 v; asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rlx_u16(u16* p, u16 v) { const u32 w = v; asm volatile("st.relaxed.gpu.global.u16 [%0], %1;" :: "l"(p), "r"(w) : "memory"); }
__device__ __forceinline__ float poll_rlx(const float* p) { float v; do { v = ld_rlx(p); } while (is_sent(v)); return v; }
__device__ __noinline__ void att_dist_work() {
  extern __shared__ float sm[];
  const bool xcmp = x_compact();
  const Ctx& cx = CX(); const Op& op = OP();
  const int lane = threadIdx.x & 31, wb = threadIdx.x >> 5;
  const int mc = cx.max_ctx;
  const int S = cx.nblocks / (4 * cx.B);
  const int u = blockIdx.x / S, ch = blockIdx.x % S, b = u >> 2, kh = u & 3;
  const int col = c_start(cx, b);
  const int Tpos = spos(b) + cx.pos_off;                 // the new key's position
  const int n = Tpos + 1, nkg = (Tpos + 16) >> 4, nend = nkg << 4, nchb = (Tpos + 256) >> 8;
  if (ch >= nchb) return;
#ifdef LMIX_PROF
  unsigned long long lmt = 0;
#endif
  LMP(0);
  const bool hub = ch == 0, last = ch == nchb - 1;
  float* A = sm + WORK_OFF;
  u16* QF = (u16*)(A + DQF);
  float* part = A + DPART;
  float* LSC = A + DSC;
  u16* PF = (u16*)(A + DPF);
  float* MI = A + DMI;
  const u16* m = cx.mbuf + (size_t)col * MROW;
  u16* Kc = (u16*)op.p4 + (size_t)(b * 4 + kh) * mc * 256;
  u16* Vc = (u16*)op.p5 + (size_t)(b * 4 + kh) * mc * 256;
  // hand-over slots (global, sentinels between ops): (max, 1 / sum) per reader chunk, score rows, value sums per chunk
  float* mi2 = (float*)op.p8 + (size_t)u * 10 * 12;
  float* sc2 = (float*)op.p9 + (size_t)u * 6 * mc;
  float4* vp2 = (float4*)op.p10 + (size_t)u * 16 * 10 * 32;
  const int kt_new = Tpos >> 4;                         // the key tile / key group the new key and value go to
  const int nkc = min(16, nkg - ch * 16);                // key tiles (= key groups) of this chunk
  // ---- 0. this chunk's K tiles and V tiles: L2 fetches them now
#ifndef ATT_NOPF
  if (lane == 0 && wb >= 8 && wb < 26) {
    if (wb < 10) {
      const long long off = (long long)(wb - 8) * 65536, kb = (long long)nkc * 8192;
      if (off < kb) {
        const unsigned nb = kb - off < 65536 ? (unsigned)(kb - off) : 65536u;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"((const char*)Kc + (size_t)ch * 16 * 8192 + off), "r"(nb) : "memory");
      }
    } else {
      const unsigned nb = (unsigned)nkc * 512u;
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"((const char*)Vc + ((size_t)(wb - 10) * (mc >> 4) + ch * 16) * 512), "r"(nb) : "memory");
    }
  }
#endif
  // ---- 1. prep (att_prep_work): warps 0-5 the query heads; in the last chunk's block warp 6 the key, warp 7 the value
  if (wb < 6 || (last && wb < 8)) {
    const float* gq = (const float*)op.p6; const float* gk = (const float*)op.p7;
    const int kind = wb < 6 ? 0 : (wb == 6 ? 1 : 2);
    const u16* src = m + (kind == 0 ? (kh * 6 + wb) * 512 : (kind == 1 ? 12288 + kh * 256 : 13312 + kh * 256));
    u16 raw[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) raw[e] = ldv(src + lane + 32 * e);
    float x[8];
    float ss = 0.f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) { x[e] = bf2f(raw[e]); ss += x[e] * x[e]; }
    ss = warp_sum(ss);
    if (kind < 2) {
      const float rs = rsqrtf(ss * (1.0f / 256.0f) + 1e-6f);
      const float* gg = kind == 0 ? gq : gk;
      #pragma unroll
      for (int e = 0; e < 8; ++e) x[e] = rbf(x[e] * rs * gg[lane + 32 * e]);
      const float cs = cx.rope_cos[(size_t)Tpos * 32 + lane], sn = cx.rope_sin[(size_t)Tpos * 32 + lane];
      const float x0 = rbf(rbf(x[0] * cs) + rbf(-x[1] * sn));
      const float x1 = rbf(rbf(x[1] * cs) + rbf(x[0] * sn));
      x[0] = x0; x[1] = x1;
    }
    if (kind == 0) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) QF[xf_idx(wb, lane + 32 * e, 1)] = f2bf(x[e]);
    } else if (kind == 1) {
      const int kt = Tpos >> 4, r = Tpos & 15;
      #pragma unroll
      for (int e = 0; e < 8; ++e) { const int i = lane + 32 * e; st_rlx_u16(Kc + ((size_t)kt * 16 + (i >> 4)) * 256 + af_idx(r, i & 15), f2bf(x[e])); }
    } else {
      const int kg = Tpos >> 4, kk = Tpos & 15;
      #pragma unroll
      for (int e = 0; e < 8; ++e) { const int i = lane + 32 * e; st_rlx_u16(Vc + ((size_t)(i >> 4) * (mc >> 4) + kg) * 256 + af_idx(i & 15, kk), f2bf(x[e])); }
    }
  }
  __syncthreads();
  LMP(1);
  // ---- 2. scores of this chunk's key tiles (att_score_work): warp = tile; kept in shared memory and stored to the rows
  if (wb < nkc) {
    const int g = lane >> 2, t = lane & 3, kt = ch * 16 + wb;
    const uint2* qf = (const uint2*)QF + lane;
    const uint4* Kt = (const uint4*)Kc + ((size_t)kt * 16 * 256) / 8 + lane;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int s0 = 0; s0 < 16; s0 += 8) {
      uint4 a[8];
      #pragma unroll
      for (int s = 0; s < 8; ++s) a[s] = kt == kt_new ? ld_rlx_u4(Kt + (s0 + s) * 32) : ld_kv(Kt + (s0 + s) * 32);   // the tile just appended to
      #pragma unroll
      for (int s = 0; s < 8; ++s) mma(c, a[s], qf[(s0 + s) * 32]);
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int qq = 2 * t + i;
      if (qq < 6) {
        const float v0 = c[i] * 0.0625f, v1 = c[2 + i] * 0.0625f;
        LSC[qq * 256 + wb * 16 + g] = v0; LSC[qq * 256 + wb * 16 + g + 8] = v1;
        float* sc = sc2 + (size_t)qq * mc + kt * 16 + g;   // to the hub: the valid keys only (it reads exactly those)
        if (kt * 16 + g < n) st_rlx(sc, v0);
        if (kt * 16 + g + 8 < n) st_rlx(sc + 8, v1);
      }
    }
  }
  __syncthreads();
  LMP(2);
  // ---- 3. the hub: softmax reductions over the whole rows (att_softmax4_work), (max, 1 / sum) published
  if (hub) {
    LMP(3);
    const int rl = wb >> 2, q = wb & 3;
    const bool ok = rl < 6;
    float* sc = sc2 + (size_t)(ok ? rl : 0) * mc;
    const bool one = nend <= 1024;
    float vr[8];
    if (ok && one) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) { const int j = (q + 4 * e) * 32 + lane; vr[e] = j < n ? ld_rlx(sc + j) : -3.0e38f; }
      #pragma unroll
      for (int e = 0; e < 8; ++e) { const int j = (q + 4 * e) * 32 + lane; if (j < n) { while (is_sent(vr[e])) vr[e] = ld_rlx(sc + j); st_rlx(sc + j, __uint_as_float(SENT_BITS)); } }
    }
    float mx = -3.0e38f;
    if (ok && one) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) mx = fmaxf(mx, vr[e]);
    } else if (ok) {
      for (int c0 = q; c0 * 32 < n; c0 += 32) {
        float v[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) { const int j = (c0 + 4 * e) * 32 + lane; v[e] = j < n ? poll_rlx(sc + j) : -3.0e38f; }
        #pragma unroll
        for (int e = 0; e < 8; ++e) mx = fmaxf(mx, v[e]);
      }
    }
    #pragma unroll
    for (int sh = 16; sh >= 1; sh >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, sh));
    if (lane == 0) part[rl * 8 + q] = mx;
    __syncthreads();
    mx = fmaxf(fmaxf(part[rl * 8], part[rl * 8 + 1]), fmaxf(part[rl * 8 + 2], part[rl * 8 + 3]));
    float sum = 0.f;
    if (ok && one) {
      #pragma unroll
      for (int e = 0; e < 8; ++e) if ((q + 4 * e) * 32 + lane < n) sum += __expf(vr[e] - mx);
    } else if (ok) {
      for (int c0 = q; c0 * 32 < n; c0 += 32) {
        float v[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) { const int j = (c0 + 4 * e) * 32 + lane; v[e] = j < n ? ld_rlx(sc + j) : -3.0e38f; }
        #pragma unroll
        for (int e = 0; e < 8; ++e) { const int j = (c0 + 4 * e) * 32 + lane; if (j < n) { sum += __expf(v[e] - mx); st_rlx(sc + j, __uint_as_float(SENT_BITS)); } }
      }
    }
    sum = warp_sum(sum);
    if (lane == 0) part[rl * 8 + 4 + q] = sum;
    __syncthreads();
    const float inv = 1.0f / ((part[rl * 8 + 4] + part[rl * 8 + 5]) + (part[rl * 8 + 6] + part[rl * 8 + 7]));
    if (ok && q == 0 && lane == 0) {
      MI[rl] = mx; MI[6 + rl] = inv;
      for (int c1 = 1; c1 < nchb; ++c1) { st_rlx(mi2 + c1 * 12 + rl, mx); st_rlx(mi2 + c1 * 12 + 6 + rl, inv); }
    }
    __syncthreads();
  } else {
    if (threadIdx.x < 12) { float* sl = mi2 + ch * 12 + threadIdx.x; MI[threadIdx.x] = poll_rlx(sl); st_rlx(sl, __uint_as_float(SENT_BITS)); }
    __syncthreads();
  }
  LMP(4);
  // ---- 4. probability fragments of this chunk's keys: the expression of att_softmax4_work, element by element
  {
    const int rl = wb >> 2, q = wb & 3;
    if (rl < 6) {
      const float mx = MI[rl], inv = MI[6 + rl];
      #pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int jl = (q + 4 * e) * 32 + lane, j = ch * 256 + jl;
        if (j < nend) PF[xf_idx(rl, jl, 1)] = j < n ? f2bf(__expf(LSC[rl * 256 + jl] - mx) * inv) : (u16)0;
      }
    }
  }
  __syncthreads();
  LMP(5);
  // ---- 5. value sums of this chunk (att_vsum_work): warp = dim tile; 6. the hub adds the chunks in order (att_out_unit)
  {
    const int g = lane >> 2, t = lane & 3, dt = wb & 15;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    u16 gt[2][2];
    if (wb < 16) {
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int qq = 2 * t + i;
        gt[i][0] = 0; gt[i][1] = 0;
        if (qq < 6 && hub) {
          const u16* gm = m + (kh * 6 + qq) * 512 + 256;
          gt[i][0] = ldv(gm + dt * 16 + g); gt[i][1] = ldv(gm + dt * 16 + g + 8);
        }
      }
      const uint4* Vt = (const uint4*)Vc + (((size_t)dt * (mc >> 4) + ch * 16) * 256) / 8 + lane;
      const uint2* pf = (const uint2*)PF + lane;
      #pragma unroll
      for (int s0 = 0; s0 < 16; s0 += 8) {
        uint4 a[8];
        #pragma unroll
        for (int s = 0; s < 8; ++s) a[s] = ch * 16 + s0 + s < nkg ? (ch * 16 + s0 + s == kt_new ? ld_rlx_u4(Vt + (s0 + s) * 32) : ld_kv(Vt + (s0 + s) * 32)) : make_uint4(0, 0, 0, 0);
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
          if (ch * 16 + s0 + (s & ~3) < nkg) {   // att_vsum_work: groups of 4 k-steps while the group's first exists
            const bool on = ch * 16 + s0 + s < nkg;
            mma(c, a[s], on ? pf[(s0 + s) * 32] : make_uint2(0, 0));
          }
        }
      }
      if (!hub) st_rlx4(vp2 + ((size_t)dt * 10 + ch) * 32 + lane, make_float4(c[0], c[1], c[2], c[3]));
    }
    if (!hub) return;
    LMP(6);
    if (wb < 16) {
      u16* xn = (u16*)op.xf;
      float tot[4] = {0.f, 0.f, 0.f, 0.f};
      tot[0] += c[0]; tot[1] += c[1]; tot[2] += c[2]; tot[3] += c[3];
      for (int c1 = 1; c1 < nchb; c1 += 4) {
        float4 v[4];
        #pragma unroll
        for (int e = 0; e < 4; ++e)
          if (c1 + e < nchb) {
            float4* sl = vp2 + ((size_t)dt * 10 + c1 + e) * 32 + lane;
            do { v[e] = ld_rlx4(sl); } while (is_sent(v[e].x) || is_sent(v[e].y) || is_sent(v[e].z) || is_sent(v[e].w));
            st_rlx4(sl, sent4());
          }
        #pragma unroll
        for (int e = 0; e < 4; ++e)
          if (c1 + e < nchb) { tot[0] += v[e].x; tot[1] += v[e].y; tot[2] += v[e].z; tot[3] += v[e].w; }
      }
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int qq = 2 * t + i;
        if (qq < 6) {
          const int h = kh * 6 + qq, d0 = dt * 16 + g, d1 = d0 + 8;
          xn[xg_idx(col, h * 256 + d0, cx.NG, xcmp)] = f2bf(rbf(tot[i]) * rbf(sigmoidf_(bf2f(gt[i][0]))));
          xn[xg_idx(col, h * 256 + d1, cx.NG, xcmp)] = f2bf(rbf(tot[2 + i]) * rbf(sigmoidf_(bf2f(gt[i][1]))));
        }
      }
    }
    LMP(7);
  }
}

// ------------------------------------------------------------------------------------------------ argmax, finish
__device__ __noinline__ void argmax_work() {
  extern __shared__ float sm[];
  const Ctx& cx = CX(); const Op& op = OP();
  const int col = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wib = tid >> 5;
  const int NT = (int)op.NT;
  int* dst = (int*)op.out;
  float* smv = sm + WORK_OFF;
  int* smi = (int*)(sm + WORK_OFF + 64);
  float m = -3.0e38f; int idx = 0x7fffffff;
  #pragma unroll
  for (int e0 = 0; e0 < 16; e0 += 8) {
    float v[8]; int ix[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int T = tid + 1024 * (e0 + e);
      v[e] = -3.0e38f; ix[e] = 0x7fffffff;
      if (T < NT) { v[e] = ldv(cx.amax_val + (size_t)T * NCM + col); ix[e] = ldv(cx.amax_idx + (size_t)T * NCM + col); }
    }
    #pragma unroll
    for (int e = 0; e < 8; ++e) if (v[e] > m || (v[e] == m && ix[e] < idx)) { m = v[e]; idx = ix[e]; }
  }
  #pragma unroll
  for (int sh = 16; sh >= 1; sh >>= 1) {
    const float om = __shfl_xor_sync(0xffffffffu, m, sh); const int oi = __shfl_xor_sync(0xffffffffu, idx, sh);
    if (om > m || (om == m && oi < idx)) { m = om; idx = oi; }
  }
  if (lane == 0) { smv[wib] = m; smi[wib] = idx; }
  __syncthreads();
  if (wib == 0) {
    m = smv[lane]; idx = smi[lane];
    #pragma unroll
    for (int sh = 16; sh >= 1; sh >>= 1) {
      const float om = __shfl_xor_sync(0xffffffffu, m, sh); const int oi = __shfl_xor_sync(0xffffffffu, idx, sh);
      if (om > m || (om == m && oi < idx)) { m = om; idx = oi; }
    }
    if (lane == 0) dst[col] = idx;
  }
  __syncthreads();
}

__device__ __noinline__ void finish_work() {
  const Ctx& cx = CX(); const Op& op = OP();
  const int b = blockIdx.x, P = c_len(cx, b), c0 = c_start(cx, b);
  if (op.i6) {  // the head ran: publish the chosen tokens
    for (int p = 0; p < P; ++p) {
      const long long tk = ldv(cx.tok_res + c0 + p);
      if (cx.tok_out) cx.tok_out[c0 + p] = tk | ((long long)cx.tok_tag << 32);
      if ((cx.flags & F_FEEDBACK) && p == P - 1) cx.tokbuf[b] = tk;
    }
  }
  cx.pos[b] = ldv(cx.pos + b) + P;
  if (cx.flags & F_DEFER) {   // updates of this sequence not yet in its stored state (the recurrent ops' `keep`)
    const int na = ldv(cx.nacc + NCM + b);
    cx.nacc[NCM + b] = (P == 1 && na + 1 < cx.defer_n) ? na + 1 : 0;
  }
}

// ------------------------------------------------------------------------------------------------ MTP drafter
// Input of mtp.fc for column c: cat[rms(embed(token), pre_fc_norm_embedding), rms(hidden, pre_fc_norm_hidden)],
// hidden = the post-norm output of the pass that ran before: r * bf16(h (1+w)), from that pass's last x buffer
// (op.p5) and the partial sums of squares.
__device__ __noinline__ void mtp_in_work() {
  extern __shared__ float sm[];
  const Ctx& cx = CX(); const Op& op = OP();
  const int c = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wib = tid >> 5;
  const int b = c_seq(cx, c), p = c - c_start(cx, b);
  float* w0 = sm + WORK_OFF;
  int na = ldv(cx.nacc + b); if (na < 1) na = 1;
  const int cs = cx.src_sel ? s_start(cx, b) + (na - 1) : s_start(cx, b) + p;
  long long tok;
  if (op.i6 == 0) tok = ldv(cx.tok_res + c);
  else if (op.i6 == 1) tok = cx.tokbuf[cx.nst ? cx.nst[b] + (int)op.i7 : b * (int)op.M + (int)op.i7];
  else tok = cx.tok_next[(size_t)c * cx.tok_stride];
  const u16* emb = (const u16*)op.W + (size_t)tok * HID;
  const float* ge = (const float*)op.gnext; const float* gh = (const float*)op.p4;
  const u16* src = (const u16*)op.p5;
  u16* out = (u16*)op.out;
  // r of the source column, summed exactly as the GEMM prologue does
  if (wib == 0) {
    float v[10];
    #pragma unroll
    for (int i = 0; i < 10; ++i) v[i] = ldv(cx.ssq + (size_t)(lane + 32 * i) * NCM + cs);
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < 10; ++i) s += v[i];
    s = warp_sum(s);
    if (lane == 0) w0[0] = rsqrtf(s * (1.0f / HID) + 1e-6f);
  }
  u16 xr[5]; float e[5], hn[5];
  #pragma unroll
  for (int i = 0; i < 5; ++i) xr[i] = ldv(src + xf_idx(cs, tid + 1024 * i, cx.src_NG));
  __syncthreads();
  const float r = w0[0];
  float se = 0.f, sh = 0.f;
  #pragma unroll
  for (int i = 0; i < 5; ++i) {
    e[i] = bf2f(emb[tid + 1024 * i]); hn[i] = rbf(r * bf2f(xr[i]));
    se += e[i] * e[i]; sh += hn[i] * hn[i];
  }
  se = warp_sum(se); sh = warp_sum(sh);
  if (lane == 0) { w0[64 + wib] = se; w0[128 + wib] = sh; }
  __syncthreads();
  if (wib == 0) {
    const float a = warp_sum(w0[64 + lane]), bq = warp_sum(w0[128 + lane]);
    if (lane == 0) { w0[1] = rsqrtf(a * (1.0f / HID) + 1e-6f); w0[2] = rsqrtf(bq * (1.0f / HID) + 1e-6f); }
  }
  __syncthreads();
  const float re = w0[1], rh = w0[2];
  #pragma unroll
  for (int i = 0; i < 5; ++i) {
    const int j = tid + 1024 * i;
    out[xf_idx(c, j, cx.NG)] = f2bf(e[i] * re * ge[j]);
    out[xf_idx(c, HID + j, cx.NG)] = f2bf(hn[i] * rh * gh[j]);
  }
  __syncthreads();
}

// End of a verify pass (one warp per sequence): how many of the drafted tokens the model confirmed, what comes
// out, what goes in next. Column 0 held the true token x, columns 1..P-1 the drafts; y[p] is the model's choice
// after column p. Drafts for the next pass: if the last NGRAM tokens occurred before in the sequence and what
// followed them agrees with the MTP head's first draft, the continuation is copied from the context (and the
// chained MTP launches skip themselves); otherwise draft 1 is the MTP head's and the chain makes the rest.
__device__ __noinline__ void accept_work() {
  const Ctx& cx = CX();
  const int b = blockIdx.x, P = c_len(cx, b), c0 = c_start(cx, b), NP = n_len(cx, b), n0 = n_start(cx, b), lane = threadIdx.x & 31;
  long long* tbn = cx.tokbuf_next ? cx.tokbuf_next : cx.tokbuf;   // the next pass's drafts
  int* hist = cx.hist + (size_t)b * cx.max_ctx;
  int pos = 0, a = 0, d1 = 0;
  int y[PMAX];
  #pragma unroll
  for (int p = 0; p < PMAX; ++p) y[p] = 0;
  if (lane == 0) {
    pos = ldv(cx.pos + b);
    const int lim = ldv(cx.limit + b);
    for (int p = 0; p < P; ++p) y[p] = ldv(cx.tok_res + c0 + p);
    a = 1;
    while (a < P && cx.tokbuf[c0 + a] == (long long)y[a - 1]) ++a;
    if (a > lim - pos) a = lim - pos;
    if (a < 0) a = 0;
    if (cx.tok_out) for (int p = 0; p < P; ++p) cx.tok_out[c0 + p] = y[p];
    if (cx.cnt_out) cx.cnt_out[b] = a;
    cx.nacc[b] = a;
    // the recurrent layers left this pass's final state in the other buffer: when every position was accepted it
    // becomes the committed state (no replay); otherwise the accepted positions are replayed on the committed one
    const bool all = a == P;
    cx.nacc[NCM + b] = all ? 0 : a;
    if (all) cx.nacc[2 * NCM + b] ^= 1;
    cx.pos[b] = pos + a;
    for (int p = 0; p < a; ++p) hist[pos + 1 + p] = y[p];
    if (a > 0) {
      // op.i6: a pass without the MTP head (the host's choice while a copied continuation keeps being accepted):
      // no MTP draft, a context copy is taken without the agreement check
      d1 = OP().i6 ? -1 : ldv(cx.mtp_res + c0 + a - 1);
      tbn[n0] = y[a - 1];
      if (NP > 1) tbn[n0 + 1] = d1 >= 0 ? d1 : y[a - 1];
    }
  }
  pos = __shfl_sync(0xffffffffu, pos, 0); a = __shfl_sync(0xffffffffu, a, 0); d1 = __shfl_sync(0xffffffffu, d1, 0);
  #pragma unroll
  for (int p = 0; p < PMAX; ++p) y[p] = __shfl_sync(0xffffffffu, y[p], 0);
  int found = -1;
  const int cur = pos + a;                      // position of the new current token
  if (a > 0 && cur >= NGRAM) {   // search even when the next pass has no room for copied drafts: the flag tells the
                                 // host this sequence is copyable, so it can get more positions next time
    // tokens at positions <= pos come from the history buffer (older launches), newer ones from y
    #define TOKAT(i) ((i) <= pos ? ldv(hist + (i)) : y[(i) - pos - 1])
    int key[NGRAM];
    #pragma unroll
    for (int t = 0; t < NGRAM; ++t) key[t] = TOKAT(cur - t);
    for (int j0 = cur - 1; j0 >= NGRAM - 1 && found < 0; j0 -= 32) {   // 32 candidate end positions per round, newest first
      const int j = j0 - lane;
      int m = -1;
      if (j >= NGRAM - 1 && TOKAT(j) == key[0]) {
        bool ok = true;
        #pragma unroll
        for (int t = 1; t < NGRAM; ++t) ok = ok && (TOKAT(j - t) == key[t]);
        if (ok && (d1 < 0 || TOKAT(j + 1) == d1)) m = j;
      }
      #pragma unroll
      for (int sh = 16; sh >= 1; sh >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, sh));
      found = m;
    }
    if (lane == 0 && found >= 0 && NP > 2) {
      int d[PMAX];
      for (int i = 1; i < NP; ++i) {
        const int src = found + i;
        d[i] = src <= cur ? TOKAT(src) : d[src - cur];
        tbn[n0 + i] = d[i];
      }
    }
    #undef TOKAT
  }
  if (lane == 0) cx.lkf[b] = found >= 0 ? 1 : 0;
}

// ------------------------------------------------------------------------------------------------ the kernel
// Only warps (or blocks) that have work enter a phase function: a function's spills are executed by every thread
// that enters it (measured: 3 us per op for an empty pass through a function with a large frame).
// Two entry points share one body (mk_launch picks). TRI = true: one-token passes of B >= 4 sequences without
// speculation (more recurrent units than blocks), whose recurrent op is lmix_tri; TRI = false: every other pass, with
// the call sites of the inherited kernel. Two kernels because code generation is per entry point and fragile here:
// lmix_tri in the common kernel made B=1 (which never calls it) 22 us slower, and a kernel pruned to the one-column-
// group GEMM instances got 72 / 52 bytes of spill in gemm_stream_bf<1> (0 in the full kernel): every GEMM +33 %.
// So each kernel keeps a call graph that was measured: the inherited one, and the inherited one with lmix_tri in
// the place of lmix_dual<false>.
template <bool TRI>
__device__ __forceinline__ void mk_body(const long long* prog, int nops, const Ctx& cxp) {
  extern __shared__ float sm[];
  if (cxp.flags & F_CHAIN) {   // a draft-chain launch has nothing to do when every sequence copied its drafts
    bool all = true;            // ... or (per-sequence lengths) needs no draft as deep as this launch's
    const int k = cxp.pos_off + 2;
    for (int b = 0; b < cxp.B; ++b) all = all && (cxp.lkf[b] != 0 || (cxp.nst && cxp.nst[b + 1] - cxp.nst[b] <= k));
    if (all) {   // the host has already counted this launch's barriers
      if (threadIdx.x == 0) atomicAdd(cxp.bar, (u64)nops);
      return;
    }
  }
  if (threadIdx.x == 0) { *(Ctx*)sm = cxp; *(volatile int*)(sm + RCNT_OFF) = 0; *(volatile int*)(sm + RCNT_OFF + 1) = 0; *(volatile int*)(sm + RCNT_OFF + 5) = 0;
                          *(const long long* volatile*)(sm + MAIN_OFF) = prog; *(volatile int*)(sm + MAIN_OFF + 2) = nops; }
  else if (threadIdx.x >= 32 && threadIdx.x < 40 && (int)threadIdx.x - 32 < cxp.B) ((volatile int*)(sm + RCNT_OFF + 6))[threadIdx.x - 32] = ldv(cxp.pos + threadIdx.x - 32);
  else if (threadIdx.x >= 64 && threadIdx.x < 72 && (int)threadIdx.x - 64 < cxp.B) ((volatile int*)(sm + SPAR_OFF))[threadIdx.x - 64] = ldv(cxp.nacc + 2 * NCM + threadIdx.x - 64);
  else if (threadIdx.x >= 96 && threadIdx.x < 104 && (int)threadIdx.x - 96 < cxp.B) ((volatile int*)(sm + SNA_OFF))[threadIdx.x - 96] = ldv(cxp.nacc + NCM + threadIdx.x - 96);
  else if (threadIdx.x <= 20) ((long long*)sm + CTX_WORDS)[threadIdx.x - 1] = prog[threadIdx.x - 1];
  __syncthreads();
  for (;;) {
#ifdef MAIN_REGS
    const int o = *(volatile const int*)(sm + RCNT_OFF + 1);
    if (o >= nops) break;
#else
    if (*(volatile const int*)(sm + RCNT_OFF + 1) >= *(volatile const int*)(sm + MAIN_OFF + 2)) break;
#endif
    const int w = cta_v() * 32 + (tid_v() >> 5);
    const Ctx& cx = CX(); const Op& op = OP();
    const int B = cx.B, P = cx.P, NC = cx.NC;
    switch (op.type) {
      case OP_EMBED: if (w < NC * 320) embed_work(); break;
      case OP_GEMM: gemm_phase<TRI>(); break;
      case OP_LMIX: {
        if (P == 1 && !(cx.flags & F_SPEC) && B * 48 > cx.nblocks && !(cx.flags & 8192)) {
          if (cx.flags & F_DEFER) lmix_dual<true>();
          else if (TRI) lmix_tri();
          else lmix_dual<false>();
          break;
        }
        // split: 2 blocks per head for one sequence with > 8 positions, and (flag-free) for a single step of one
        // sequence (48 latency-bound units on 152 blocks)
        const bool split1 = P == 1 && !(cx.flags & 2048 << 5) && B * 96 <= cx.nblocks;   // debug 65536: off
        const bool split = (P > 8 || split1) && B * 96 <= cx.nblocks && !(cx.flags & 4096);   // flag 4096: A/B without the split
        // with a unit table (per-sequence layouts) any block may hold units: every block enters
        if ((int)blockIdx.x < B * 48 * (split ? 2 : 1) || (!split && cx.cst)) {
          if (split && P == 1 && cx.na_max <= 8) lmix_work<8, true, false>();
          else if (P <= 8 && cx.na_max <= 8) { if (P >= 4) lmix_work<8, false, true>(); else lmix_work<8, false, false>(); }
          else if (split) lmix_work<PMAX, true, true>();
          else lmix_work<PMAX, false, true>();
        }
      } break;
      case OP_ATT_PREP: if (att_w() < NC * 32) att_prep_work(); break;
      case OP_ATT_SCORE: if (att_w() < B * 4 * ((cx.pos_ub + P + 15) >> 4) * cx.NGq) att_score_work(); break;
      case OP_ATT_SOFTMAX:
#ifndef SOFTMAX1
        if ((int)blockIdx.x * 8 < B * 24 * P) att_softmax4_work();   // whole blocks: block barriers inside
#else
        if (w < B * 24 * P) att_softmax_work();
#endif
        break;
      case OP_ATT_VSUM: if (att_w() < B * 64 * ((cx.pos_ub + P + 255) >> 8) * cx.NGq) att_vsum_work(); break;
      case OP_ATT_OUT: if (att_w() < B * 64 * cx.NGq) att_out_work(); break;
      case OP_ATT_DIST: {  // one block per (sequence, kv head, 256-key chunk)
        const int st = cx.nblocks / (4 * B);
        if ((int)blockIdx.x / st < 4 * B && (int)blockIdx.x % st < ((cx.pos_ub + 256) >> 8)) att_dist_work();
      } break;
      case OP_ARGMAX:
        if (op.p4) {   // folded during the head GEMM: unpack, and clear for the next pass
          if (blockIdx.x == 0 && (int)threadIdx.x < NC) {
            u64* am = (u64*)op.p4;
            const u64 k = ldv((const u64*)am + threadIdx.x);
            ((int*)op.out)[threadIdx.x] = (int)(0xFFFFFFFFu - (u32)k);
            am[threadIdx.x] = 0ull;
          }
        } else if ((int)blockIdx.x < NC) argmax_work();
        break;
      case OP_FINISH: if ((int)blockIdx.x < B && threadIdx.x == 0) finish_work(); break;
      case OP_MTP_IN: if ((int)blockIdx.x < NC) mtp_in_work(); break;
      case OP_ACCEPT: if ((int)blockIdx.x < B && threadIdx.x < 32) accept_work(); break;
      case OP_CHAIN_FIN:
        if ((int)blockIdx.x < B && threadIdx.x == 0 && !ldv(cx.lkf + blockIdx.x)) {
          const int bb = blockIdx.x;
          if (!cx.nst) cx.tokbuf[bb * (int)op.M + op.i6] = ldv(cx.mtp_res + bb);
          else if ((int)op.i6 < n_len(cx, bb)) cx.tokbuf[cx.nst[bb] + op.i6] = ldv(cx.mtp_res + bb);
        }
        break;
    }
#ifdef MAIN_REGS
    barrier(prog, o, nops);
#else
    barrier(*(const long long* volatile*)(sm + MAIN_OFF), *(volatile const int*)(sm + RCNT_OFF + 1), *(volatile const int*)(sm + MAIN_OFF + 2));
#endif
  }
}

__global__ void __launch_bounds__(1024, 1) mk_kernel(const long long* prog, int nops, Ctx cxp) { mk_body<false>(prog, nops, cxp); }
__global__ void __launch_bounds__(1024, 1) mk_kernel_b(const long long* prog, int nops, Ctx cxp) { mk_body<true>(prog, nops, cxp); }

// ------------------------------------------------------------------------------------------------ host side
extern "C" {
int mk_init(int* sm_count, int* max_smem) {
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(sm_count, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaError_t e = cudaFuncSetAttribute(mk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, *max_smem);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(mk_kernel_b, cudaFuncAttributeMaxDynamicSharedMemorySize, *max_smem);
  if (e != cudaSuccess) { fprintf(stderr, "mk_init: %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
int mk_launch(const long long* prog, int nops, const Ctx* cx, int smem_bytes) {
  // mk_body<true> for the one-token passes with more than two recurrent units per block (B >= 7 on 152 blocks): their
  // recurrent op is lmix_tri. With at most two units per block (B = 4..6) lmix_dual's single round is the same work
  // with half-block barriers: B=4 8848 us per step against 8870 through lmix_tri.
#ifndef LM_TRI_OFF
  if (cx->P == 1 && !(cx->flags & (F_SPEC | F_DEFER)) && cx->B * 48 > 2 * cx->nblocks && !(cx->flags & 8192))
    mk_kernel_b<<<cx->nblocks, 1024, smem_bytes, 0>>>(prog, nops, *cx);
  else
#endif
  mk_kernel<<<cx->nblocks, 1024, smem_bytes, 0>>>(prog, nops, *cx);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "mk_launch: %s\n", cudaGetErrorString(e)); return 1; }
  return 0;
}
// Page-locked host memory the device can write (zero copy): the served decode's chosen tokens go there straight from
// the kernel's last op, so no device-to-host copy sits between two passes.
int mk_host_alloc(size_t bytes, void** host, void** dev) {
  if (cudaHostAlloc(host, bytes, cudaHostAllocMapped) != cudaSuccess) return 1;
  return cudaHostGetDevicePointer(dev, *host, 0) != cudaSuccess;
}
int mk_host_free(void* host) { return cudaFreeHost(host) != cudaSuccess; }
int mk_sizes(int* op_bytes, int* ctx_bytes) { *op_bytes = sizeof(Op); *ctx_bytes = sizeof(Ctx); return 0; }
}
