// Flint megakernel (v6): one persistent kernel per decode step, per generate() call or per served cohort, on one B200.
//
// 152 blocks, one per SM. Blocks 0..127 ("main") stream the weights: every matrix-vector product is a chain of
// tensor-core mma.m16n8k16 (bf16 x bf16 -> f32) over weights re-laid at load time into 512-byte "tiles" in the
// register order of the mma A operand (one 16-byte load per thread feeds one mma); the 8 columns of the B operand
// are the sequences of the batch. Blocks 128..151 ("aux") run attention (aux a: pairs a and a+24 of
// (sequence, K/V head)) and the router (aux 8..23, one 16-row group each).
//
// What the card dictates (measured, notes/journal.md): a warp has one global load outstanding and blocks 0.15-0.3 us
// on it; a load costs by the cache lines it touches; shared memory moves only 60-90 GB/s; a register spill reload
// can cost a microsecond. So: blocks of 512 threads (16 warps x 128 registers); weights go DRAM -> registers -> mma,
// and the tiles of the next phase (and the K/V tiles of the next layer) are loaded into registers while a block
// waits for the others; every activation buffer is laid out so that a consumer warp reads contiguous bytes; one
// warp per block does the epilogue stores, fences (stores are not visible to other SMs otherwise) and bumps a
// counter; three counters (main phases, attention, router) replace a global barrier.
//
// Track A (default): the number formats of the Hugging Face BF16 path, every rounding point reproduced.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#define HID 2048
#define NLAYER 49
#define NKV 4
#define NEXP 256
// Scheduling changes after s0020, each with a switch back to the old behaviour (all bit-identical; see notes/journal.md):
//   ROUTE_RANK=0      top 6 of the router by six rounds of warp maxima (batch 1-2) instead of ranks        (s0021)
//   E_LEADERS_FIXED   the first warp of each half always writes its half's hidden units                    (s0022)
//   D_EPI_EARLY       warp 0 writes the shared expert's hidden units before the routing                    (s0023)
//   KV_STORE_EARLY    the attention stores this position's K and V into the cache before the scores        (s0023)
//   KV_BATCH=0        K/V cache tiles beyond the staged ones one by one (batch 1-2; batch > 2 is always so)    (s0025)
//#define E_LEADERS_FIXED
//#define D_EPI_EARLY
//#define KV_STORE_EARLY
#ifndef KV_VEARLY
#define KV_VEARLY 1      // (with KV_BATCH) the first batch of V levels is requested right after the scores, not at the mix
#endif
#ifndef MATTN_CPRE
#ifndef MATTN_CPRE
#define MATTN_CPRE 0     // experiment: pairs on main blocks (batch 7-8) request the block's output-projection tiles inside the attention: slower (3991 -> 4071 us)
#endif
#endif
#ifndef KV_BATCH_MAIN
#ifndef KV_BATCH_MAIN
#define KV_BATCH_MAIN 1  // batched K/V loads for the pairs on main blocks (batch 7-8); see KV_BATCH_AUXBIG
#endif
#endif
#ifndef KV_BATCH_AUXBIG
#ifndef KV_BATCH_AUXBIG
#define KV_BATCH_AUXBIG 1  // batched K/V loads on the attention blocks of the batch > 2 kernel (with KV_ASM: -30 to -55 us per step at batch 8 with KV_BATCH_MAIN; as plain C loads: nothing, the compiler moved them to their uses)
#endif
#endif
#ifndef KV_BATCH
#define KV_BATCH 1       // batch 1-2: K/V cache tiles beyond the staged ones are requested 8 at a time (0: one by one)
#endif
#define TOPK 6
#ifndef SOFTMAX_DIV
#define SOFTMAX_DIV 1            // softmax weights: 1 = reciprocal + one exact correction step (divr), 0 = the compiler's division
#endif
#ifndef SOFTMAX_GUARD
#define SOFTMAX_GUARD 0          // 1: exponentials below 2^-100 go through the compiler's division (see attention(), stage 5)
#endif
#ifndef MATTN_EPOLL
#define MATTN_EPOLL 1            // batch 7-8, main blocks with a pair: the counter of phase A is polled by an idle warp during the leaders' stores
#endif
#ifndef NORM_WS
#define NORM_WS 1                // batch > 2: the norms' per-sequence scales once per warp (0: by every thread)
#endif
#ifndef LDH_AB
#define LDH_AB 16                // a sentinel read looks at the abort word on every 16th attempt (1: on every attempt, in front of the reload)
#endif
// Every read of a word that another block writes (ldx, and the retries of ldh) is a strong load at gpu scope
// (ld.relaxed.gpu), not a weak load with a cache hint (ld.global.cg, LDX_RELAXED=0). The writers use atomic exchanges
// and no stale read was ever seen here (tools/flake.py), but another run of this project on the same card found that
// a weak load in a spin can miss another SM's later store (its notes, C11); the strong form costs nothing here
// (same register allocation, same timings, same bits).
#ifndef LDX_RELAXED
#define LDX_RELAXED 1
#endif
#if LDX_RELAXED
#define LDX_OP "ld.relaxed.gpu.global"
#else
#define LDX_OP "ld.global.cg"
#endif
#ifndef HOSTC
#define HOSTC 1                  // batch 7-8: the output projection of a main block with an attention pair is computed by another main block (block 8 + o), which does the owner's rows first and its own second
#endif
// ---- Switches of the extension (2026-11-15). The ones marked "experiment" are off and measured slower or no better;
// notes/journal.md has each measurement. They are kept so that the measurements can be repeated.
// WHATIF (timing studies only: the results are wrong; batch > 2). Value = sum of: 1 forced routing (36 units, 48
// picks, the same every layer and step: needed by the others, whose garbage would change the routing); 2 (with
// F_ONE=0) the down phase copies fragments in its first round only; 4 the norms load the first pair of sequences
// only; 8 the attention sees no cached keys (4096 / 8192 / 16384: only its scores / exponentials and weights / V
// mix); 16 no routing computation (512 / 1024 / 2048: no union / no candidate extraction / no merge); 32 the copy of
// the attention output loads the first sequence only; 64 no fragment copy in the down phase; 128 no copy of the
// attention output; 256 no sentinels at the start of the expert phase.
#ifndef WHATIF
#define WHATIF 0
#endif
#ifndef HS_EARLY
#define HS_EARLY 0               // experiment (off: same bits, +7 us; six times the exchanges): batch > 2, the expert phase's sentinels issued during the wait for the attention, for every slot
#endif
#ifndef E_CTR
#define E_CTR 0                  // experiment (off, NOT STRICT: 4 mismatches in 398 batch-8 repetitions of tools/flake.py; -17 us): batch > 2, the expert phase's hidden units handed over with a counter that every warp bumps after its own writes, not with sentinels
#endif
#ifndef HST_PLAIN
#define HST_PLAIN 0              // experiment (off: +25 us): hidden units written with plain stores instead of exchanges
#endif
#ifndef HS_PLAIN
#define HS_PLAIN 0               // timing probe (not safe as it is: no confirmation): the expert phase's sentinels as plain stores, -25 us
#endif
#ifndef E_SETTLE_LATE
#define E_SETTLE_LATE 0          // experiment (off: +3 us): the wait for the expert phase's sentinel exchanges behind its first slot
#endif
#ifndef FO_V
#define FO_V 1                   // the F_ONE copy's four addresses: 0 kept in a pointer array (spills 52/88 bytes), 1 recomputed where used (24/24, -40 us per step at batch 8), 2 = 1 without the joint re-request (+30)
#endif
#ifndef F_ONE
#define F_ONE 1                  // batch > 2, down phase: one copy of all fragments before the first round (compact slots), then tiles only
#endif
#ifndef NORM_B
#define NORM_B 0                 // experiment (off: 1 = four in flight +33 us, 2 = two in flight +7 us): batch > 2, the norm's loads of the pairs of sequences issued together
#endif
#ifndef ES_WM
#define ES_WM 1                  // share of the expert phase of a main block with an attention pair ...
#define ES_WO 1                  // ... against a main block without one (experiment: 11:12, 7:8, 5:6, 3:4 change nothing)
#endif
#ifndef FPRE_EARLY
#define FPRE_EARLY 0             // experiment (off: +26 us): batch > 2, the down phase's first tiles requested before the expert phase's hidden units are written
#endif
#ifndef APRE_EARLY
#define APRE_EARLY 0             // experiment (off: nothing): the next layer's phase-A tiles requested before the sentinel exchanges of the layer end
#endif
#ifndef XPLAIN
#define XPLAIN 0                 // experiment (off: -4 to -10 us, not taken: no order against the next sentinel): residual-stream rows written with a plain store instead of an exchange
#endif
#ifndef HSTG
#define HSTG 0                   // experiment (off: with the batched K/V loads it is 16-45 us slower than without): the owners stage their pair's first K/V tiles in registers after their phase A
#endif
#ifndef K2B
#define K2B 0                    // experiment (off: same bits, +10 us, spill line 160/52/60): batch 7-8, pairs on main blocks: two batches of K tiles in flight in the scores
#endif
#ifndef RSTG
#define RSTG 0                   // experiment (off: same bits, -2 us = nothing; the pairs on main blocks are the last ones anyway): batch > 2, router blocks with an attention pair stage key groups 16..27 in shared memory, too
#endif
#ifndef FCB_W
#define FCB_W 1                  // the batched copy of the attention output requests every pending word again in each attempt (-12.6 us per step at batch 8) instead of waiting for one alone
#endif
#ifndef FCB
#define FCB 4                    // batch > 2: the copy of the attention output loads this many sequences per trip to memory (0: one by one; 2: +18 us, 8: +40 us, 4: -42 us per step at batch 8)
#endif
#ifndef MATTN_TAIL
#define MATTN_TAIL 0             // experiment (off: +46 us; the two exchanges in front of the attention delay it by 1.9 us): batch 7-8, main blocks with a pair: sentinels of the output projection before the attention, attention counter asked inside it
#endif
#ifndef F_PIPE
#define F_PIPE 0                 // experiment (off: +69 us per step at batch 8, journal): batch > 2, down phase: the next round's fragments and all 16 tiles requested together
#endif
#ifndef AFIRST
#define AFIRST 0                 // experiment (off: +310 us per step at batch 8, journal): batch 7-8, the main blocks with an attention pair do no phase A (hosted elsewhere) and run the pair first
#endif
#ifndef GATHER_BIG
#define GATHER_BIG 1             // generate/serve, batch > 2: the gather loop in 16 strides on 16 threads per sequence
#endif
#ifndef GATHER_PAR
#define GATHER_PAR 1             // generate/serve: block 0 gathers the 152 blocks' best tokens with all its threads (0: one thread per sequence)
#endif
#ifndef RDEFER
#define RDEFER 1                 // batch > 2: router tiles preloaded on the router blocks, their K/V staging after the router duty
#endif
#ifndef ROUTE_RANK_BIG
#define ROUTE_RANK_BIG 1         // batch > 2: the 12 candidates of a sequence merged by ranks (bit-identical)
#endif
#ifndef ROUTE_RANK
#define ROUTE_RANK 1             // batch 1-2: top 6 by ranks instead of rounds of warp maxima (bit-identical choice and weights)
#endif
#define VOCAB 150144
#define NMAIN 128
#define NAUX 24
#define NROUT 16
#define NBLK 152
#define MAXB 8
#define MAXU 48
#define RMS_EPS 1e-6f
#define CM 0          // counter of finished main phases (x NMAIN; the expert gate/up phase counts KE per block: its two leaders)
#define KE 2
#define NMF 4           // BF16 limbs of the fused router matrix M (32 bits)
#define KEBIG 16      // batch > 2 instance: all 16 warps write the hidden units at the end of the phase, each bumps once
#define KA 5          // ... and phase A counts 5: its five leader warps
#define CX 32         // counter of finished attention phases (x number of attention blocks)
#define CR 64         // counter of finished router phases (x NROUT)
#define CZ 80         // track B, early routing: router blocks that have read the layer input (x NROUT); the mains
                      // wait for it before the output projection overwrites the stream in place
#define HSLOT_SHARED 48
// Track B precision mode (lossless in exact arithmetic): the residual stream is never rounded to BF16 (float32
// instead), intermediate sums that feed only float32 code are not rounded, and the router sees its input as
// hi + lo BF16 parts (float32-accurate logits for the discrete top-6 choice). Operands of the tensor-core products
// (normed stream, q/k/v, attention weights and output, SwiGLU hidden) stay BF16.
#ifdef TRACK_B
#define TB 1
#else
#define TB 0
#endif
#if TB && defined(EARLY_ROUTE)
#define ER 1          // early routing (track B only): see the router blocks
#else
#define ER 0
#endif
__device__ __forceinline__ float rb(float f);
__device__ __forceinline__ float rbA(float f);
#define HCHUNKS (50 * 32)

struct LayerP {
    const uint4 *qkv, *o, *dgu, *ddown, *router, *sgu, *sdown, *egu, *edown;
    const uint4 *rp, *mf;        // track B, early routing: R' = R diag(w2) as 2 BF16 limbs [2][16 rg][128 chunks] tiles; M = R' W_o as NMF limbs [NMF][16 rg][256 chunks]
    uint4 *kc, *vc;
    const float *ln1, *ln2, *qn, *kn, *sinks;
    int sliding, moe;
};

struct Params {
    LayerP L[NLAYER];
    const u16* embed;
    const uint4* head;
    const float* fnorm;
    const float *cosT, *sinT;
    int maxctx, kgmax;
    // scratch (allocated by mk_create)
    // Activation buffers. Each is laid out so that a consumer warp reads 512 contiguous bytes (a warp load costs by
    // the cache lines it touches).
    float4* xm;          // the stream after the attention (same layout as xc): written by phase C, read by phase D and the router
    float4* xc;          // residual stream [4 column pairs t][128 row groups][8 g]: track A as BF16 bits (uint2: {row g: col 2t | col 2t+1 << 16, row g+8: ...}), track B as float4 (row g | g+8) x (col 2t | 2t+1)
    float4* qraw;        // raw q/k/v sums per sequence: [MAXB][320 row groups][8 g] = {half 0: row g, row g+8; half 1: row g, row g+8}
    uint2* attnf;        // attention output as B fragments, per sequence: [MAXB][256 chunks][4 lanes]
    float* rl;           // [MAXB][NEXP] router logits
    uint2* hbuf;         // SwiGLU hidden as B fragments, per sequence: [MAXB][50 slots x 32 chunks][4 lanes]; slots 48,49 = shared expert; dense MLP: 384 chunks from slot 0
    float* argv;         // [NMAIN][MAXB]
    int* argi;           // [NMAIN][MAXB]
    long long* curtok;   // [MAXB]
    int* gen;            // served cohorts: [MAXB][maxctx] generated tokens; progress in ctr[96]
    unsigned* ctr;       // counters (CM, CX, CR), [96] = generated steps finished
    float* dbg;          // [NLAYER+1][MAXB][HID] or null
    unsigned* prof;      // profiling build: [3][PROF_MAX][2] (id, tick) samples
    unsigned* clk;       // profiling build: free-running counter written by the clock block
    unsigned* bt;        // profiling build: [16 events][NBLK] clock value of every main block at the phase boundaries of layer 8
    int fence;           // (unused) Every warp that stores results for other blocks calls __threadfence before the counter moves:
                         // without it about 8% of 600-token generations differed between runs (a store can land after the atomic).
};

#ifdef MK_PROF
#define PROF_MAX 4096
#ifdef MK_BTONLY
#define PROF_BLK(b) ((b) == 0 ? 0 : -1)
#define PROF_ID_OK(id) ((id) == 30)
#else
#ifndef PROF_MAIN
#define PROF_MAIN 0              // the main block that is sampled
#endif
#define PROF_BLK(b) ((b) == PROF_MAIN ? 0 : (b) == NMAIN ? 1 : (b) == NMAIN + 8 ? 2 : -1)
#define PROF_ID_OK(id) ((id) < 31 || PROF_FINE)
#endif
// Sampled: only the layers with l % 6 == 2 are recorded (8 sliding MoE layers), so the events (a global load of the
// clock each) do not slow the whole step; ids of 31 and above only with -DMK_PROF_FINE.
#ifdef MK_PROF_FINE
#define PROF_FINE 1
#else
#define PROF_FINE 0
#endif
__device__ __forceinline__ unsigned prof_clk(const unsigned* c) { unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(c)); return v; }
#define PROF(id) do { int pb_ = PROF_BLK(blk); if ((profl || (id) == 30) && PROF_ID_OK(id) && tid == 0 && pb_ >= 0 && pi < PROF_MAX) { unsigned* q_ = P->prof + (pb_ * PROF_MAX + pi) * 2; q_[0] = (id); q_[1] = prof_clk(P->clk); pi++; } } while (0)
#define BT(ev) do { if (l == 8 && tid == 0) P->bt[(ev) * NBLK + blk] = prof_clk(P->clk); } while (0)
#define PROFX(id) do { int pb_ = PROF_BLK(pblk); if (ppi && PROF_ID_OK(id) && threadIdx.x == 0 && pb_ >= 0 && *ppi < PROF_MAX) { unsigned* q_ = P->prof + (pb_ * PROF_MAX + *ppi) * 2; q_[0] = (id); q_[1] = prof_clk(P->clk); (*ppi)++; } } while (0)
#define BT_FLUSH() do {} while (0)
#define CLK_END() do {} while (0)
#define CLK_A(e) do {} while (0)
#elif defined(MK_CLK)
// MK_CLK (a study build of the normal kernel: MK_EXTRA=-DMK_CLK, tools/bt_clk.py): the phase boundaries of layer
// MK_CLK_L of every main block, read from the cycle counter %clock64. No memory operation in the path: the stamps go
// to shared memory (16 words of the float area that nothing else uses) and are written out when the layer is over (the profiling build's markers are loads, and a load
// waits for what its warp issued before it). Slots 14 / 15: the step's first A start and the end of its head.
#ifndef MK_CLK_L
#define MK_CLK_L 8
#endif
__device__ __forceinline__ unsigned clk_lo() { unsigned long long v; asm volatile("mov.u64 %0, %%clock64;" : "=l"(v)); return (unsigned)v; }
// Attention blocks (columns 128.. of the same table): slot 0 = every block has begun A, 1..8 = the attention's stages
// (marker id - 60; 5 is the weights), 9 = back from the attention; kept in 16 more free words, written out at slot 9.
#define PROF(id) do {} while (0)
#define CLK_A(e) do { if (tid == 0 && l == MK_CLK_L) { unsigned* a_ = (unsigned*)(sb + SM_SM + 648 * 4); a_[e] = clk_lo(); \
    if ((e) == 9) for (int e_ = 0; e_ < 10; e_++) P->bt[e_ * NBLK + blk] = a_[e_]; } } while (0)
#define PROFX(id) do { if (ppi && threadIdx.x == 0 && (id) >= 61 && (id) <= 68) ((unsigned*)ppi)[(id) - 60] = clk_lo(); } while (0)
#define BT(ev) do { if (tid == 0 && (l == MK_CLK_L || ((ev) == 0 && l == 0))) ((unsigned*)(sb + SM_SM + 632 * 4))[l == MK_CLK_L ? (ev) : 14] = clk_lo(); } while (0)
#define BT_FLUSH() do { if (tid == 0 && l == MK_CLK_L) { for (int e_ = 0; e_ < 14; e_++) P->bt[e_ * NBLK + blk] = ((const unsigned*)(sb + SM_SM + 632 * 4))[e_]; } } while (0)
#define CLK_END() do { if (tid == 0) { P->bt[14 * NBLK + blk] = ((const unsigned*)(sb + SM_SM + 632 * 4))[14]; P->bt[15 * NBLK + blk] = clk_lo(); } } while (0)
#else
#define PROF(id) do {} while (0)
#define PROFX(id) do {} while (0)
#define BT(ev) do {} while (0)
#define BT_FLUSH() do {} while (0)
#define CLK_END() do {} while (0)
#define CLK_A(e) do {} while (0)
#endif

// Which hand-overs fence before the counter moves (experiment mask; the default is all of them):
// 1 attention output, 2 phase A (q/k/v sums, 5 leader warps), 4 phase C, 8 expert gate/up (2 leaders), 16 down phase,
// 32 router, 64 dense layers, 128 head.
#ifndef FENCE_MASK
#ifdef PLAIN_STORE
#define FENCE_MASK 255
#else
#define FENCE_MASK 0
#endif
#endif
#define FENCE(bit) do { if (FENCE_MASK & (bit)) __threadfence(); } while (0)

// ------------------------------------------------------------------------------------------------ small helpers
__device__ __forceinline__ float rb(float f) {       // round to nearest-even BF16, result as float
    u32 u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xFFFF0000u);
}
__device__ __forceinline__ u32 fbits(float f) {      // BF16 bits of f rounded to nearest even
    u32 u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return u >> 16;
}
__device__ __forceinline__ float b2f(u16 b) { return __uint_as_float(((u32)b) << 16); }
// a rounding point of the Hugging Face BF16 path that track B does not need
__device__ __forceinline__ float rbA(float f) { return TB ? f : rb(f); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ void mma(float4& c, const uint4& a, const uint2& b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y));
}
__device__ __forceinline__ float4 f4add(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float cget(const float4& c, int i) { return i == 0 ? c.x : i == 1 ? c.y : i == 2 ? c.z : c.w; }

// u16 index of element (k within chunk, column n) inside a 256-byte B-fragment chunk
__device__ __forceinline__ int bidx(int kk, int n) { return (((n * 4 + ((kk & 7) >> 1)) * 2 + (kk >> 3)) * 2) + (kk & 1); }
// u16 index of element (row m, col k) inside a 512-byte A tile
__device__ __forceinline__ int aidx(int m, int k) { return ((((m & 7) * 4 + ((k & 7) >> 1)) * 4 + (m >> 3) + 2 * (k >> 3)) * 2) + (k & 1); }
// u16 index of element (key 0..15, dim 0..15) inside a 512-byte K-cache tile (two B operands of 8 keys side by side)
__device__ __forceinline__ int kidx(int key, int dd) { return ((((key & 7) * 4 + ((dd & 7) >> 1)) * 4 + (key >> 3) * 2 + (dd >> 3)) * 2) + (dd & 1); }

// Loads of data another block wrote in this launch bypass this SM's L1 (ld.global.cg): L1 caches are not coherent
// across SMs, a default load can return the line as it was a phase ago (tools/bench8.cu: 2.8% stale reads without a
// fence, none with these loads). With them the writers need no __threadfence.
__device__ __forceinline__ float4 ldx(const float4* p) { float4 r; asm volatile("" LDX_OP ".v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p)); return r; }
__device__ __forceinline__ uint4 ldx(const uint4* p) { uint4 r; asm volatile("" LDX_OP ".v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r; }
__device__ __forceinline__ uint2 ldx(const uint2* p) { uint2 r; asm volatile("" LDX_OP ".v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p)); return r; }
__device__ __forceinline__ float ldx(const float* p) { float r; asm volatile("" LDX_OP ".f32 %0, [%1];" : "=f"(r) : "l"(p)); return r; }
__device__ __forceinline__ int ldx(const int* p) { int r; asm volatile("" LDX_OP ".s32 %0, [%1];" : "=r"(r) : "l"(p)); return r; }
// Results that other blocks read are written with atomic exchanges, and every warp that writes bumps the counter
// itself afterwards: the atomics of one warp reach L2 in order, so a reader that sees the counter sees the data.
// Plain stores do not have this property, not even from the same warp and not with st.global.cg: about 8% of
// 600-token generations differed between runs until the writers fenced (-DPLAIN_STORE: plain stores + __threadfence
// before every bump, 11% slower) or switched to atomics (1.5% slower). tools/flake_gen1.py is the test.
#ifndef PLAIN_STORE
#define ATOMIC_STORE        // the default
#endif
#ifdef ATOMIC_STORE
__device__ __forceinline__ u32 put64(void* p, u32 lo, u32 hi) { return (u32)atomicExch((unsigned long long*)p, ((unsigned long long)hi << 32) | lo); }
__device__ __forceinline__ u32 put32(void* p, u32 v) { return atomicExch((unsigned int*)p, v); }
#else
__device__ __forceinline__ u32 put64(void* p, u32 lo, u32 hi) { *(uint2*)p = make_uint2(lo, hi); return 0u; }
__device__ __forceinline__ u32 put32(void* p, u32 v) { *(u32*)p = v; return 0u; }
#endif
// SETTLE(dep), by every lane of a writing warp before the warp bumps a counter: dep is the OR of what the warp's
// exchanges returned, so the ballot cannot run before every one of them has come back from L2. (An exchange whose
// result nobody reads does not hold the warp up, and the bump that follows can overtake it: tools/flake.py.)
#ifdef NO_SETTLE
#define SETTLE(dep) do { (void)(dep); } while (0)
#else
#define SETTLE(dep) do { if (__ballot_sync(0xffffffffu, (dep) == 0x7fc0deadu) == 0xffffffffu) __threadfence(); } while (0)
#endif
__device__ __forceinline__ long long ldx(const long long* p) { long long r; asm volatile("" LDX_OP ".s64 %0, [%1];" : "=l"(r) : "l"(p)); return r; }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Wait until a counter reaches its target. Watchdog: after about a second of spinning the launch is declared dead
// (ctr[100] = 1) and every wait in every block returns at once, so a bug can never leave a kernel spinning forever.
#define CABORT 100
// WAIT_TID: the thread that asks. A load waits for what its warp issued before it, exchanges and counter bumps
// included (about 2 us each; measured: a main block's two sentinel exchanges in front of its attention delayed the
// attention by 1.9 us), and warp 0 is the leader that issues them at the end of most phases. It was thread 0;
// thread 192 (warp 6: never a leader of phase A, of the attention's stores or of a fold): batch 1 1403 -> 1389 us,
// batch 8 3690 -> 3682 (threads 256 and 480: the same).
#ifndef WAIT_TID
#define WAIT_TID 192
#endif
__device__ __forceinline__ void wait_ctr(const Params* P, int which, unsigned target) {
    if (threadIdx.x == WAIT_TID) {
        volatile unsigned* c = P->ctr;
        unsigned spins = 0;
        while ((int)(c[which] - target) < 0) {
            if ((spins++ & 1023u) == 1023u && (c[CABORT] || spins > (4u << 20))) { c[CABORT] = 1u; break; }
        }
    }
    __syncthreads();
}
// the same for the calling thread alone, without the barrier (the caller's next barrier passes the fact on to the block)
__device__ __forceinline__ void poll_ctr(const Params* P, int which, unsigned target) {
    volatile unsigned* c = P->ctr;
    unsigned spins = 0;
    while ((int)(c[which] - target) < 0) {
        if ((spins++ & 1023u) == 1023u && (c[CABORT] || spins > (4u << 20))) { c[CABORT] = 1u; break; }
    }
}
// two counters, one barrier
__device__ __forceinline__ void wait_ctr2(const Params* P, int w0, unsigned t0, int w1, unsigned t1) {
    if (threadIdx.x == 0) {
        volatile unsigned* c = P->ctr;
        unsigned spins = 0;
        while ((int)(c[w0] - t0) < 0 || (int)(c[w1] - t1) < 0) {
            if ((spins++ & 1023u) == 1023u && (c[CABORT] || spins > (4u << 20))) { c[CABORT] = 1u; break; }
        }
    }
    __syncthreads();
}

// ---- Strict hand-over without a wait at the end of a phase: the sentinel.
// A bump can overtake the exchanges of its own warp (SETTLE above), and waiting for them costs about 0.3 us on the
// path every other block waits on. So the writer of the residual stream, of q/k/v and of the attention output writes
// SENT into its words at the START of its phase, when every reader of their old content is known to be done, and
// reads the exchanges' results back before it computes (no time lost: the phase's first loads are in flight). At the
// end it exchanges the data in and bumps at once. A reader that sees the bump finds in every word either the data or
// the sentinel, never the old content, and loads a word again while it holds the sentinel.
// SENT is a NaN with a payload in every format used (two float32, or four BF16): no result of the model's arithmetic.
#define SENT 0x7fc17fc1u
// The bump of such a hand-over only has to tell the readers that the sentinels are in place, so it is issued right
// after they are read back, at the START of the phase; the readers then wait on the data words themselves (ldh): one
// trip through the memory queue (the exchange) instead of three (exchange and bump, the reader's poll, its load).
#ifdef LATE_BUMP
#define EB 0
#else
#define EB 1
#endif
__device__ __forceinline__ bool is_sent(const uint4& r) { return r.x == SENT || r.y == SENT || r.z == SENT || r.w == SENT; }
__device__ __forceinline__ bool is_sent(const float4& r) { return is_sent(make_uint4(__float_as_uint(r.x), __float_as_uint(r.y), __float_as_uint(r.z), __float_as_uint(r.w))); }
__device__ __forceinline__ bool is_sent(float r) { return __float_as_uint(r) == SENT; }
template <class V> __device__ __noinline__ V ldh(const V* p, const Params* P) {      // rare: off the hot path
    V r = ldx(p);
    unsigned spins = 0;
#if LDH_AB > 1
    // The abort word is looked at on every LDH_AB-th attempt only. It used to be loaded in front of every reload, and
    // the reload waited for it: one more trip through the memory per attempt, in every hand-over of every layer (where
    // the word happened to lie moved the step by 17 us at batch 1: tools/padsweep.sh, journal).
    while (is_sent(r)) {
        if ((++spins & (LDH_AB - 1)) == 0) {
            if (((volatile unsigned*)P->ctr)[CABORT]) break;
            if (spins > (1u << 22)) { ((volatile unsigned*)P->ctr)[CABORT] = 1u; break; }
        }
        r = ldx(p);
    }
    return r;
}
#else
    while (is_sent(r)) {
        unsigned ab = ((volatile unsigned*)P->ctr)[CABORT];        // (also keeps the load inside the loop)
#ifdef LDH_BACKOFF
        // pause between two attempts (loads of one cached word: no traffic in the memory queue)
        for (int d = 0; d < LDH_BACKOFF; d++) ab |= ((volatile unsigned*)P->ctr)[CABORT];
#endif
        if (ab) break;
        if (++spins > (1u << 22)) { ((volatile unsigned*)P->ctr)[CABORT] = 1u; break; }
        r = ldx(p + ab);
    }
    return r;
}
#endif

// ---- shared-memory B fragments: [chunk][Ls lanes] of 8 bytes; Ls = 4 * (B rounded up to a power of two): only the
// lanes of the sequences in use exist.
// B operand of an mma from the fragment buffer in shared memory: chunk c, this lane (lanes without a column: zero).
// XGET_BF: without a conditional load. Written as "lane < Ls ? xf[...] : 0" the compiler left a branch around the load
// in some builds and predicated the load in others, depending on code elsewhere in the kernel: 23 instructions per
// tile instead of 5 in the phases it hit (tools/sass_seg.py; any edit could move the batch > 2 step by 10-25%).
// Now every lane loads: a lane without a column reads the zero word in front of the buffer, with stride 0.
// (Bit 0: the batch 1-2 kernel, bit 1: the batch > 2 kernel. The batch 1-2 kernel had the predicated form and got
//  47 us slower at batch 1 with this one, 1403 -> 1451: it keeps the conditional.)
#ifndef XGET_BF
#define XGET_BF 2
#endif
__device__ __forceinline__ uint2 xget_c(const uint2* xf, int Ls, int chunk, int lane) { return lane < Ls ? xf[chunk * Ls + lane] : make_uint2(0u, 0u); }
__device__ __forceinline__ uint2 xget(const uint2* xf, int Ls, int chunk, int lane) {
#if XGET_BF
    if (!((XGET_BF >> (Ls >= 16 ? 1 : 0)) & 1)) return xget_c(xf, Ls, chunk, lane);
    const bool ok = lane < Ls;
    const uint2* b = ok ? xf + lane : xf - 1;
    const int st = ok ? Ls : 0;
    return b[chunk * st];
#else
    return xget_c(xf, Ls, chunk, lane);
#endif
}
// u16 index inside a per-sequence global fragment buffer [chunk][4 lanes]
__device__ __forceinline__ int gidx(int chunk, int kk) { return ((chunk * 4 + ((kk & 7) >> 1)) * 2 + (kk >> 3)) * 2 + (kk & 1); }

// TLOAD_ASM: the tile loads as volatile asm: the compiler keeps them where the source has them, relative to the other
// memory operations of the warp (a warp's memory operations complete in the order issued; as plain C loads their
// place between the source position and the first use was the compiler's choice, different from build to build).
// Bit 0: the batch 1-2 kernel's tiles, bit 1: the batch > 2 kernel's (the caller passes which).
#ifndef TLOAD_ASM
#define TLOAD_ASM 0              // experiment (off: +4 to +10 us at both batch sizes)
#endif
#ifndef KV_ASM
#define KV_ASM 1
#endif
#ifndef EXS
#define EXS 0                    // experiment (off: same bits, batch 8 +14 us, batch 1 +30): the attention's scores of a thread's 2nd and 3rd key group in shared memory instead of on the stack
#endif
#ifndef UNI2
#define UNI2 1                   // batch > 2, the union of the routing with two barriers fewer: its byte table lives in the partial-sum area (free between phases D and E), is zeroed in stage 1 and written by stage 2 (-10 us per step at batch 8). 2: also the prefix counts without a barrier, by each warp's own ballots over the warps before it (measured the same as 1: not taken)
#endif
#ifndef RSORT
#define RSORT 2                  // batch > 2, candidate stage of the routing: every thread sorts its four scores once (stable). 1: a round is then the warp maximum of the threads' first scores and a shift in the winner (-2.5 us per step at batch 8: the rounds are bound by their shuffles, not by the instructions around them). 2: no rounds, two rank counts through shared memory (-14 us)
#endif
#ifndef RKT
#define RKT 1                    // batch 1-2, the merge of the routing ranks at most 27 of the 48 candidates, one per lane (see there): -16 us per token at batch 1, same bits
#endif
#ifndef UNI4
#define UNI4 1                   // batch > 2, the union's last pass (unit tables) on all 16 warps: threads tid and tid + 256 share an expert, the first writes sequences 0..3, the second 4..7 (-6 us per step at batch 8, same bits)
#endif
#ifndef UNI5
#define UNI5 1                   // batch > 2, the merge writes the union's table bytes itself (rank and expert are in the lane); the sum and the weights of a sequence are computed by warp 8 + b during the ballots (-3 us per step at batch 8, same bits)
#endif
#define UNI5X (UNI5 && UNI4 && UNI2 == 1 && ROUTE_RANK_BIG)
#ifndef RTS
#define RTS 1
#endif
#ifndef VPAD
#define VPAD 0                   // experiment (off: same bits, batch 8 +10 us, batch 1 +64 by another allocation): V mix in whole batches of 8 tiles, zero weights past the context (no test per tile)
#endif
#ifndef KV_DB
#define KV_DB 0                  // experiment (off: batch 8 -5 us, batch 1 +38): V mix with two batches of tiles in flight (see the attention's stage 6)
#endif
#ifndef KV_STG_ASM
#define KV_STG_ASM 0             // experiment (off: nothing): the staged K/V tiles (attn_stage) as volatile asm loads, too
#endif
// one tile, as volatile asm (the K/V batches of the attention: as plain C loads the compiler was free to move them
// down to their uses, one trip to memory each again)
__device__ __forceinline__ uint4 lda(const uint4* p) {
#if KV_ASM
    uint4 r; asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r;
#else
    return *p;
#endif
}
template <int N> __device__ __forceinline__ void tload(uint4* T, const uint4* p) {
#if TLOAD_ASM
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(T[i].x), "=r"(T[i].y), "=r"(T[i].z), "=r"(T[i].w) : "l"(p + i * 32));
#else
#pragma unroll
    for (int i = 0; i < N; i++) T[i] = p[i * 32];
#endif
}

// norm weights of the next RMSNorm into shared memory: one 16-byte load per thread
__device__ __forceinline__ void nw_load(float* nw, const float* w) {
    const int tid = threadIdx.x;
    ((float4*)nw)[tid] = ((const float4*)w)[tid];
}

#if ER
// Early routing (track B): limb p (0, 1, 2) of the float32 residual stream as B fragments (hi, the BF16 of what hi lost,
// and of what that lost: 24 bits, the whole float32). Same thread/row layout as norm_frag.
__device__ __forceinline__ void x_limb(const Params* P, int B, int Ls, uint2* xf, int p) {
    const int tid = threadIdx.x;
    const int rg = tid >> 2, sub = tid & 3, g0 = 2 * sub;
    const int npair = (B + 1) >> 1;
    for (int t2 = 0; t2 < npair; t2++) {
        const float4* q = P->xc + t2 * 1024 + rg * 8 + g0;
        float4 xa = ldx(q), xb = ldx(q + 1);
        if (is_sent(xa)) xa = ldh(q, P);
        if (is_sent(xb)) xb = ldh(q + 1, P);
        for (int j = 0; j < 2; j++) {
            const int b = 2 * t2 + j;
            if (b >= B) break;
            float n0 = j ? xa.y : xa.x, n1 = j ? xb.y : xb.x, n8 = j ? xa.w : xa.z, n9 = j ? xb.w : xb.z;
            for (int k = 0; k < p; k++) { n0 -= b2f((u16)fbits(n0)); n1 -= b2f((u16)fbits(n1)); n8 -= b2f((u16)fbits(n8)); n9 -= b2f((u16)fbits(n9)); }
            xf[rg * Ls + b * 4 + sub] = make_uint2(fbits(n0) | (fbits(n1) << 16), fbits(n8) | (fbits(n9) << 16));
        }
    }
    __syncthreads();
}
#endif

// ---- RMSNorm of the residual stream into B fragments. Thread i: row group i/4, rows 2s, 2s+1 and +8 (s = i%4): two
// 16-byte loads per pair of sequences, and its 8 bytes of output are exactly one lane of the B operand.
// emb != null: layer 0, the stream is the embedding rows (block 0 stores it).
// WS: the scale of a sequence (16 partial sums from shared memory, rsqrt) is computed once per warp, by lane b, and
// read by shuffle (batch > 2: every thread computed all 8 of them, 264 instructions where 45 do).
template <int WS>
__device__ __forceinline__ void norm_frag(const Params* P, const u16* emb, const long long* toks, int tstride, int tcol,
                                          const float* nw, int B, int Ls, uint2* xf, float* sm, bool store_x, float* dbg, float4* xo, int blk, uint2* xlo = nullptr, uint4* nsc = nullptr, bool mid = false) {
    const float4* X = mid ? P->xm : P->xc;
    // nsc: scratch in shared memory. The pairs of sequences after the first are kept there between the two passes instead
    // of being loaded again (at batch 8 the reload was 3 of 7 load round trips, one after the other).
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int rg = tid >> 2, sub = tid & 3, g0 = 2 * sub, r0 = rg * 16 + g0;
    const int npair = (B + 1) >> 1;
    float4 ka = make_float4(0.f, 0.f, 0.f, 0.f), kb = ka;
    uint4 wr0 = make_uint4(0u, 0u, 0u, 0u);      // (WHATIF)
    // NORM_B (batch > 2, track A): the loads of the (up to four) pairs of sequences are issued together and kept in
    // registers, instead of load, sentinel test, sums, next load: one trip to memory behind the other. A word that holds
    // the sentinel is waited for alone; the later ones that held it are then requested again together.
    uint4 nb0 = wr0, nb1 = wr0, nb2 = wr0, nb3 = wr0;
    const bool nbatch = NORM_B == 1 && WS && !TB && !emb && !(WHATIF & 4);
    const bool nbatch2 = NORM_B == 2 && WS && !TB && !emb && !(WHATIF & 4);      // (two loads in flight: pairs 0-1, then 2-3)
    if (nbatch) {
        const uint4* xp = (const uint4*)((const uint2*)X + rg * 8 + g0);
        nb0 = ldx(xp);
        if (npair > 1) nb1 = ldx(xp + 512);
        if (npair > 2) nb2 = ldx(xp + 1024);
        if (npair > 3) nb3 = ldx(xp + 1536);
        if (is_sent(nb0)) {
            nb0 = ldh(xp, P);
            if (npair > 1 && is_sent(nb1)) nb1 = ldx(xp + 512);
            if (npair > 2 && is_sent(nb2)) nb2 = ldx(xp + 1024);
            if (npair > 3 && is_sent(nb3)) nb3 = ldx(xp + 1536);
        }
        if (npair > 1 && is_sent(nb1)) {
            nb1 = ldh(xp + 512, P);
            if (npair > 2 && is_sent(nb2)) nb2 = ldx(xp + 1024);
            if (npair > 3 && is_sent(nb3)) nb3 = ldx(xp + 1536);
        }
        if (npair > 2 && is_sent(nb2)) {
            nb2 = ldh(xp + 1024, P);
            if (npair > 3 && is_sent(nb3)) nb3 = ldx(xp + 1536);
        }
        if (npair > 3 && is_sent(nb3)) nb3 = ldh(xp + 1536, P);
    }
    for (int t2 = 0; t2 < npair; t2++) {
        float4 xa, xb;      // xa: rows g0 and g0+8, xb: rows g0+1 and g0+9; (x, z) sequence 2*t2, (y, w) sequence 2*t2+1
        if (emb) {
            const u16* e0 = emb + (size_t)((unsigned long long)ldx(toks + (size_t)(2 * t2) * tstride + tcol) % VOCAB) * HID;
            xa = make_float4(b2f(e0[r0]), 0.f, b2f(e0[r0 + 8]), 0.f);
            xb = make_float4(b2f(e0[r0 + 1]), 0.f, b2f(e0[r0 + 9]), 0.f);
            if (2 * t2 + 1 < B) {
                const u16* e1 = emb + (size_t)((unsigned long long)ldx(toks + (size_t)(2 * t2 + 1) * tstride + tcol) % VOCAB) * HID;
                xa.y = b2f(e1[r0]); xa.w = b2f(e1[r0 + 8]); xb.y = b2f(e1[r0 + 1]); xb.w = b2f(e1[r0 + 9]);
            }
            if (rg == blk) { xo[g0 * 4 + t2] = xa; xo[(g0 + 1) * 4 + t2] = xb; }   // own 16 rows stay in shared memory
        } else {
#if TB
            const float4* p = X + t2 * 1024 + rg * 8 + g0;
            xa = ldx(p); xb = ldx(p + 1);
            if (is_sent(xa)) xa = ldh(p, P);
            if (is_sent(xb)) xb = ldh(p + 1, P);
            if (nsc && t2 > 0) { ((float4*)nsc)[((t2 - 1) * 2) * 512 + tid] = xa; ((float4*)nsc)[((t2 - 1) * 2 + 1) * 512 + tid] = xb; }
#else
            uint4 r;
            if (nbatch) r = t2 == 0 ? nb0 : t2 == 1 ? nb1 : t2 == 2 ? nb2 : nb3;
            else if (nbatch2) {
                const uint4* xp = (const uint4*)((const uint2*)X + t2 * 1024 + rg * 8 + g0);
                if (!(t2 & 1)) {
                    nb0 = ldx(xp);
                    if (t2 + 1 < npair) nb1 = ldx(xp + 512);
                    if (is_sent(nb0)) { nb0 = ldh(xp, P); if (t2 + 1 < npair && is_sent(nb1)) nb1 = ldx(xp + 512); }
                    r = nb0;
                } else {
                    if (is_sent(nb1)) nb1 = ldh(xp, P);
                    r = nb1;
                }
            }
            else if ((WHATIF & 4) && t2 > 0) r = wr0;
            else {
            r = ldx((const uint4*)((const uint2*)X + t2 * 1024 + rg * 8 + g0));     // rows g0, g0+8 | g0+1, g0+9, two sequences each
            if (is_sent(r)) r = ldh((const uint4*)((const uint2*)X + t2 * 1024 + rg * 8 + g0), P);
            wr0 = r;
            }
            if (nsc && t2 > 0) nsc[(t2 - 1) * 512 + tid] = r;
            xa = make_float4(b2f((u16)r.x), b2f((u16)(r.x >> 16)), b2f((u16)r.y), b2f((u16)(r.y >> 16)));
            xb = make_float4(b2f((u16)r.z), b2f((u16)(r.z >> 16)), b2f((u16)r.w), b2f((u16)(r.w >> 16)));
#endif
        }
        if (t2 == 0) { ka = xa; kb = xb; }
        const float s0 = warp_sum(xa.x * xa.x + xa.z * xa.z + xb.x * xb.x + xb.z * xb.z);
        const float s1 = warp_sum(xa.y * xa.y + xa.w * xa.w + xb.y * xb.y + xb.w * xb.w);
        if (lane == 0) { sm[(2 * t2) * 16 + warp] = s0; sm[(2 * t2 + 1) * 16 + warp] = s1; }
        if (dbg) {
            dbg[(2 * t2) * HID + r0] = xa.x; dbg[(2 * t2) * HID + r0 + 8] = xa.z; dbg[(2 * t2) * HID + r0 + 1] = xb.x; dbg[(2 * t2) * HID + r0 + 9] = xb.z;
            if (2 * t2 + 1 < B) { dbg[(2 * t2 + 1) * HID + r0] = xa.y; dbg[(2 * t2 + 1) * HID + r0 + 8] = xa.w; dbg[(2 * t2 + 1) * HID + r0 + 1] = xb.y; dbg[(2 * t2 + 1) * HID + r0 + 9] = xb.w; }
        }
    }
    __syncthreads();
    const float2 wlo = *(const float2*)(nw + r0), whi = *(const float2*)(nw + r0 + 8);
    float scl = 0.f;
    if (WS) {
        const int bq = lane < B ? lane : 0;
        float s = 0.f;
        for (int w = 0; w < 16; w++) s += sm[bq * 16 + w];
        scl = rsqrtf(s * (1.0f / HID) + RMS_EPS);
    }
    for (int t2 = 0; t2 < npair; t2++) {
        float4 xa = ka, xb = kb;
        if (t2 > 0) {
            if (emb) {
                const u16* e0 = emb + (size_t)((unsigned long long)ldx(toks + (size_t)(2 * t2) * tstride + tcol) % VOCAB) * HID;
                xa = make_float4(b2f(e0[r0]), 0.f, b2f(e0[r0 + 8]), 0.f);
                xb = make_float4(b2f(e0[r0 + 1]), 0.f, b2f(e0[r0 + 9]), 0.f);
                if (2 * t2 + 1 < B) {
                    const u16* e1 = emb + (size_t)((unsigned long long)ldx(toks + (size_t)(2 * t2 + 1) * tstride + tcol) % VOCAB) * HID;
                    xa.y = b2f(e1[r0]); xa.w = b2f(e1[r0 + 8]); xb.y = b2f(e1[r0 + 1]); xb.w = b2f(e1[r0 + 9]);
                }
            } else {
#if TB
                const float4* p = X + t2 * 1024 + rg * 8 + g0;
                if (nsc) { xa = ((const float4*)nsc)[((t2 - 1) * 2) * 512 + tid]; xb = ((const float4*)nsc)[((t2 - 1) * 2 + 1) * 512 + tid]; }
                else { xa = ldx(p); xb = ldx(p + 1); }
#else
                const uint4 r = nsc ? nsc[(t2 - 1) * 512 + tid] : ldx((const uint4*)((const uint2*)X + t2 * 1024 + rg * 8 + g0));
                xa = make_float4(b2f((u16)r.x), b2f((u16)(r.x >> 16)), b2f((u16)r.y), b2f((u16)(r.y >> 16)));
                xb = make_float4(b2f((u16)r.z), b2f((u16)(r.z >> 16)), b2f((u16)r.w), b2f((u16)(r.w >> 16)));
#endif
            }
        }
        for (int j = 0; j < 2; j++) {
            const int b = 2 * t2 + j;
            if (b >= B) break;
            float sc;
            if (WS) sc = __shfl_sync(0xffffffffu, scl, b);
            else {
                float s = 0.f;
                for (int w = 0; w < 16; w++) s += sm[b * 16 + w];
                sc = rsqrtf(s * (1.0f / HID) + RMS_EPS);
            }
            const float n0 = wlo.x * ((j ? xa.y : xa.x) * sc), n1 = wlo.y * ((j ? xb.y : xb.x) * sc);   // rows g0, g0+1
            const float n8 = whi.x * ((j ? xa.w : xa.z) * sc), n9 = whi.y * ((j ? xb.w : xb.z) * sc);   // rows g0+8, g0+9
            const u32 h0 = fbits(n0), h1 = fbits(n1), h8 = fbits(n8), h9 = fbits(n9);
            xf[rg * Ls + b * 4 + sub] = make_uint2(h0 | (h1 << 16), h8 | (h9 << 16));
            if (xlo)    // what the BF16 operand lost, as a second BF16 operand
                xlo[rg * Ls + b * 4 + sub] = make_uint2(fbits(n0 - b2f((u16)h0)) | (fbits(n1 - b2f((u16)h1)) << 16), fbits(n8 - b2f((u16)h8)) | (fbits(n9 - b2f((u16)h9)) << 16));
        }
    }
    __syncthreads();
}

// Copy the B fragments of the sequences in use from a per-sequence global buffer into shared memory: nchunks (<= 256)
// chunks, src(b, chunk) = pointer to the 4 lanes of that chunk. One contiguous 16-byte load per thread and sequence.
template <class Src>
__device__ __forceinline__ void frag_copy(uint2* xf, int Ls, int nchunks, int B, Src src, const Params* Pchk = nullptr) {
    // Pchk: the source is written with the sentinel protocol
    const int tid = threadIdx.x, chunk = tid >> 1, q = tid & 1;
    if (chunk >= nchunks) return;
    for (int b = 0; b < B; b++) {
        uint4 r = ldx((const uint4*)(src(b, chunk) + 2 * q));
        if (Pchk && is_sent(r)) r = ldh((const uint4*)(src(b, chunk) + 2 * q), Pchk);
        *(uint4*)(xf + chunk * Ls + b * 4 + 2 * q) = r;
    }
}

// The same with the loads of N sequences issued together. frag_copy loads, tests for the sentinel and stores one
// sequence after the other: at batch 8 that is 8 trips to memory, one behind the other, per thread (the test stands
// between a load and the next), and they are all still to come when the last attention output has arrived (timing
// study WHATIF bit 5: 100 us per step at batch 8). Here N loads are issued together and kept in registers; a word
// that still holds the sentinel is waited for alone (one load per attempt, as before), and when it has arrived the
// later ones that held the sentinel are requested again together.
template <int N, class Src>
__device__ __forceinline__ void frag_copy_n(uint2* xf, int Ls, int nchunks, int B, Src src, const Params* P) {
    const int tid = threadIdx.x, chunk = tid >> 1, q = tid & 1;
    if (chunk >= nchunks) return;
    for (int b0 = 0; b0 < B; b0 += N) {
        uint4 r[N];
#pragma unroll
        for (int k = 0; k < N; k++) r[k] = b0 + k < B ? ldx((const uint4*)(src(b0 + k, chunk) + 2 * q)) : make_uint4(0u, 0u, 0u, 0u);
#if FCB_W
        // (FCB_W: every word of the batch that holds the sentinel is requested again in each attempt, so that no trip
        //  is left when the last one arrives; more loads while the attention blocks work)
        for (unsigned spins = 0;;) {
            bool any = false;
#pragma unroll
            for (int k = 0; k < N; k++) any |= b0 + k < B && is_sent(r[k]);
            if (!any) break;
            if ((++spins & (LDH_AB - 1)) == 0) {
                if (((volatile unsigned*)P->ctr)[CABORT]) break;
                if (spins > (1u << 22)) { ((volatile unsigned*)P->ctr)[CABORT] = 1u; break; }
            }
#pragma unroll
            for (int k = 0; k < N; k++) if (b0 + k < B && is_sent(r[k])) r[k] = ldx((const uint4*)(src(b0 + k, chunk) + 2 * q));
        }
#else
#pragma unroll
        for (int k = 0; k < N; k++) {
            if (b0 + k < B && is_sent(r[k])) {
                r[k] = ldh((const uint4*)(src(b0 + k, chunk) + 2 * q), P);
#pragma unroll
                for (int j = k + 1; j < N; j++) if (b0 + j < B && is_sent(r[j])) r[j] = ldx((const uint4*)(src(b0 + j, chunk) + 2 * q));
            }
        }
#endif
#pragma unroll
        for (int k = 0; k < N; k++) if (b0 + k < B) *(uint4*)(xf + chunk * Ls + (b0 + k) * 4 + 2 * q) = r[k];
    }
}

// Write one hidden unit of a SwiGLU (gate row g of the row group, up row g+8) into the per-sequence B fragments;
// chunk0: first chunk of the unit in hbuf.
__device__ __forceinline__ u32 h_store(const Params* P, const float4& c, int lr, int lane, int B, int chunk0, const float* member = nullptr) {
    u32 dep = 0u;
    // called by a whole warp. Units g (even) and g+1 of the row group share one 32-bit word of the fragment buffer:
    // the even lane gets its partner's half (lane ^ 4) and writes the word.
    const int g = lane >> 2, t = lane & 3;
    const int j = 8 * lr + g;
    for (int i = 0; i < 2; i++) {
        const int b = 2 * t + i;
        const bool on = b < B && (!member || member[b] != 0.f);
        const float gate = rbA(cget(c, i)), up = rbA(cget(c, 2 + i));
        const u32 hv = fbits(rbA(silu(gate)) * up) & 0xFFFFu;
        const u32 pv = __shfl_xor_sync(0xffffffffu, hv, 4);
#if HST_PLAIN
        // (experiment: the value as a plain store; the word holds this block's confirmed sentinel and its readers wait while they see it)
        if (on && !(g & 1)) *((u32*)(P->hbuf + (size_t)b * HCHUNKS * 4) + (gidx(chunk0 + (j >> 4), j & 15) >> 1)) = hv | (pv << 16);
#else
        if (on && !(g & 1)) dep |= put32((u32*)(P->hbuf + (size_t)b * HCHUNKS * 4) + (gidx(chunk0 + (j >> 4), j & 15) >> 1), hv | (pv << 16));
#endif
    }
    return dep;
}

// The sentinel into the q/k/v words of a main block's five leader warps, and into the attention-output words of an
// attention pair's eight leader warps.
__device__ __forceinline__ u32 q_sentinel(const Params* P, int blk, int B) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int hg = blk * 5 + warp, rg = hg >> 1, hh = hg & 1;
    u32 dep = 0u;
    for (int j = 0; j < 2; j++) {
        const int b = 2 * t + j;
        if (b < B) dep |= put64((float2*)(P->qraw + (size_t)b * 2560 + rg * 8 + g) + hh, SENT, SENT);
    }
    return dep;
}
// the same for one half row group hg, by the whole warp that hosts it (AFIRST)
__device__ __forceinline__ u32 q_sent_hg(const Params* P, int hg, int B) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int rg = hg >> 1, hh = hg & 1;
    u32 dep = 0u;
    for (int j = 0; j < 2; j++) {
        const int b = 2 * t + j;
        if (b < B) dep |= put64((float2*)(P->qraw + (size_t)b * 2560 + rg * 8 + g) + hh, SENT, SENT);
    }
    return dep;
}
__device__ __forceinline__ u32 q_chk_hg(const Params* P, int hg, int B, unsigned vo) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int rg = hg >> 1, hh = hg & 1;
    u32 bad = 0u;
    for (int j = 0; j < 2; j++) {
        const int b = 2 * t + j;
        if (b < B) bad |= ldx((const uint2*)((const float2*)(P->qraw + (size_t)b * 2560 + rg * 8 + g) + hh) + vo).x ^ SENT;
    }
    return bad;
}
__device__ __forceinline__ u32 attn_sentinel(const Params* P, int b, int kvg) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    u32 dep = 0u;
    if (!(g & 1)) {
        uint2* out = P->attnf + (size_t)b * 256 * 4;
        dep = put64(out + ((kvg * 8 + 2 * t) * 8 + warp) * 4 + (g >> 1), SENT, SENT);
        dep |= put64(out + ((kvg * 8 + 2 * t + 1) * 8 + warp) * 4 + (g >> 1), SENT, SENT);
    }
    return dep;
}

// Read-back checks of the sentinels above (nonzero: a word does not hold the sentinel yet). vo is 0; it comes from a
// volatile load so that a retry really loads again. The exchanges are issued a phase ahead and their results are not
// kept (a result carried in a register across a phase costs spills in the main blocks); the read-back load is issued
// together with the first loads of the phase that will write the words.
#define CONFIRM(bad, EXPR) do { unsigned s_ = 0; while (bad) { const unsigned vo = ((volatile unsigned*)P->ctr)[CABORT]; \
    if (vo || ++s_ > (1u << 22)) { ((volatile unsigned*)P->ctr)[CABORT] = 1u; break; } bad = (EXPR); } } while (0)
__device__ __forceinline__ u32 q_chk(const Params* P, int blk, int B, unsigned vo) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int hg = blk * 5 + warp, rg = hg >> 1, hh = hg & 1;
    u32 bad = 0u;
    for (int j = 0; j < 2; j++) {
        const int b = 2 * t + j;
        if (b < B) bad |= ldx((const uint2*)((const float2*)(P->qraw + (size_t)b * 2560 + rg * 8 + g) + hh) + vo).x ^ SENT;
    }
    return bad;
}
__device__ __forceinline__ u32 attn_chk(const Params* P, int b, int kvg, unsigned vo) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    u32 bad = 0u;
    if (!(g & 1)) {
        const uint2* out = P->attnf + (size_t)b * 256 * 4 + vo;
        bad = ldx(out + ((kvg * 8 + 2 * t) * 8 + warp) * 4 + (g >> 1)).x ^ SENT;
        bad |= ldx(out + ((kvg * 8 + 2 * t + 1) * 8 + warp) * 4 + (g >> 1)).x ^ SENT;
    }
    return bad;
}
__device__ __forceinline__ u32 x_chk(const Params* P, int blk, int lane, int B, bool mid, unsigned vo) {
    const int g = lane >> 2, t = lane & 3;
    const float4* X = (mid ? P->xm : P->xc) + vo;
    u32 bad = 0u;
    if (2 * t < B) {
#if TB
        const float4 v = ldx(X + t * 1024 + blk * 8 + g);
        bad = (__float_as_uint(v.x) ^ SENT) | (__float_as_uint(v.z) ^ SENT);
#else
        bad = ldx((const uint2*)X + t * 1024 + blk * 8 + g).x ^ SENT;
#endif
    }
    return bad;
}
__device__ __forceinline__ u32 hs_chk(const Params* P, int lr, int lane, int B, int chunk0, unsigned vo) {
    u32 bad = 0u;
    const int g = lane >> 2, t = lane & 3;
    const int j = 8 * lr + g;
    for (int i = 0; i < 2; i++) {
        const int b = 2 * t + i;
        if (b < B && !(g & 1)) bad |= (u32)ldx((const int*)(P->hbuf + (size_t)b * HCHUNKS * 4) + (gidx(chunk0 + (j >> 4), j & 15) >> 1) + vo) ^ SENT;
    }
    return bad;
}

// The sentinel into the words h_store will write (same arguments).
__device__ __forceinline__ u32 h_sent(const Params* P, int lr, int lane, int B, int chunk0, const float* member = nullptr) {
    u32 dep = 0u;
    const int g = lane >> 2, t = lane & 3;
    const int j = 8 * lr + g;
    for (int i = 0; i < 2; i++) {
        const int b = 2 * t + i;
        const bool on = b < B && (!member || member[b] != 0.f);
        if (on && !(g & 1)) dep |= put32((u32*)(P->hbuf + (size_t)b * HCHUNKS * 4) + (gidx(chunk0 + (j >> 4), j & 15) >> 1), SENT);
    }
    return dep;
}

// The same sentinel written with plain stores (HS_PLAIN: timing probe / experiment, see the journal).
__device__ __forceinline__ void h_sent_p(const Params* P, int lr, int lane, int B, int chunk0, const float* member) {
    const int g = lane >> 2, t = lane & 3;
    const int j = 8 * lr + g;
    for (int i = 0; i < 2; i++) {
        const int b = 2 * t + i;
        const bool on = b < B && member[b] != 0.f;
        if (on && !(g & 1)) *((u32*)(P->hbuf + (size_t)b * HCHUNKS * 4) + (gidx(chunk0 + (j >> 4), j & 15) >> 1)) = SENT;
    }
}

// The sentinel into this block's 16 rows of the residual stream (warp 0, at the start of a phase that will update them).
__device__ __forceinline__ u32 x_sentinel(const Params* P, int blk, int lane, int B, bool mid) {
    const int g = lane >> 2, t = lane & 3;
    float4* X = mid ? P->xm : P->xc;
    u32 dep = 0u;
    if (2 * t < B) {
#if TB
        dep = put64(X + t * 1024 + blk * 8 + g, SENT, SENT);
        dep |= put64((float2*)(X + t * 1024 + blk * 8 + g) + 1, SENT, SENT);
#else
        dep = put64((uint2*)X + t * 1024 + blk * 8 + g, SENT, SENT);
#endif
    }
    return dep;
}

// Leader epilogue of a phase that produces the residual stream: x[rows of this block] = bf16(x + delta). The block
// is the only writer of its 16 rows, so their current value is kept in shared memory (xo) and never loaded.
__device__ __forceinline__ u32 x_update(const Params* P, int blk, int lane, int B, const float4& d, float4* xo, bool mid) {
    const int g = lane >> 2, t = lane & 3;
    float4* X = mid ? P->xm : P->xc;
    u32 dep = 0u;
    if (2 * t < B) {
        const float4 xq = xo[lane];
        const float4 xn = make_float4(rbA(xq.x + d.x), rbA(xq.y + d.y), rbA(xq.z + d.z), rbA(xq.w + d.w));
        xo[lane] = xn;
#if TB
        dep = put64(X + t * 1024 + blk * 8 + g, __float_as_uint(xn.x), __float_as_uint(xn.y));
        dep |= put64((float2*)(X + t * 1024 + blk * 8 + g) + 1, __float_as_uint(xn.z), __float_as_uint(xn.w));
#else
#if XPLAIN
        // (a plain store: the words hold this block's confirmed sentinel, and their readers wait while they see it)
        *((uint2*)X + t * 1024 + blk * 8 + g) = make_uint2((__float_as_uint(xn.x) >> 16) | (__float_as_uint(xn.y) & 0xFFFF0000u), (__float_as_uint(xn.z) >> 16) | (__float_as_uint(xn.w) & 0xFFFF0000u));
#else
        dep = put64((uint2*)X + t * 1024 + blk * 8 + g, (__float_as_uint(xn.x) >> 16) | (__float_as_uint(xn.y) & 0xFFFF0000u), (__float_as_uint(xn.z) >> 16) | (__float_as_uint(xn.w) & 0xFFFF0000u));
#endif
#endif
    }
    return dep;
}

// ------------------------------------------------------------------------------------------------ attention
// The attention's float expressions with their multiply-adds spelled out. The function is inlined twice (attention
// blocks; main blocks with a pair at batch 7-8) and the compiler contracted a*b + c*d differently in the two copies:
// softmax weights one rounding step apart, so a sequence was not bitwise the same alone and in a batch (s0017-s0019,
// track B, sequences 6-7; tools/bcheck.py). Mode 1: the left product is fused; 2: the right one; 0: no fusion;
// -1: plain C. Mode 1 everywhere is what the copy on the attention blocks did: tools/digest1.py is unchanged by it.
#ifndef FM_SS
#define FM_SS 1
#endif
#ifndef FM_NE
#define FM_NE 1
#endif
#ifndef FM_ROT
#define FM_ROT 1
#endif
#ifndef FM_DOT
#define FM_DOT 1
#endif
#ifndef FM_OUT
#define FM_OUT 1
#endif
__device__ __forceinline__ float pm2(float a, float b, float c, float d, int mode) {      // a*b + c*d
    return mode == 1 ? __fmaf_rn(a, b, __fmul_rn(c, d)) : mode == 2 ? __fmaf_rn(c, d, __fmul_rn(a, b)) : mode == 0 ? __fadd_rn(__fmul_rn(a, b), __fmul_rn(c, d)) : a * b + c * d;
}
// a / S from r = 1 / S (one real division per thread): product, exact remainder by FMA, one correction. With r the
// correctly rounded reciprocal this is the correctly rounded quotient (Markstein's division step) as long as the
// remainder is exact, which it is for a >= 2^-102 (S < 2^12); tools/bench19.cu: 0 mismatches against the compiler's
// division in 6e10 quotients (a down to 2^-100, all mantissas), 38.5 -> 12.2 ns per quotient and warp.
__device__ __forceinline__ float divr(float a, float S, float r) {
    const float q = __fmul_rn(a, r);
    return __fmaf_rn(__fmaf_rn(-q, S, a), r, q);
}
__device__ __forceinline__ float pma(float a, float b, float c, int mode) {                // a*b + c
    return mode > 0 ? __fmaf_rn(a, b, c) : mode == 0 ? __fadd_rn(__fmul_rn(a, b), c) : a * b + c;
}
#define NK2 12
#define NV2 0                     // V levels staged in shared memory (0: the reservation must stay small, see SM_TOTAL)
struct AttnSm {              // shared-memory views of an attention block
    uint2* qA;               // [8 dim chunks][32 lanes] q as mma A operand (rows = heads): registers 0 and 2 (rows 8..15 are zero)
    uint2* wb;               // [key groups][32 lanes] attention weights as mma B operand
    float* cst;              // [0..127] q_norm w, [128..255] k_norm w, [256..263] sinks, [272..399] cos, [400..527] sin
    float* ks;               // [0..127] k of this position (float), [128..255] v of this position, [256..263] its attention weight, [272..399] rotary exchange of k
    float2* rq;              // [512] rotary exchange of q
    uint4* part;             // [16 warps][32] partial sums of the V mix
    uint4* k2;               // [NK2 key groups][8][32] K tiles of key groups 16 .. 16+NK2-1 (contexts over 256)
    float4* exs;             // (EXS) [2][512 threads] the scores / exponentials of a thread's second and third key group
    uint4* v2;               // [NV2 levels][16 warps][32] V tiles: level j-8 of warp (dc, parity) = key group parity + 2j, j = 8 .. 8+NV2-1
    float* sm;
};

__device__ __forceinline__ void attn_consts(const Params* P, const LayerP& L, int kvg, int pos, const AttnSm& S) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (warp == 11) ((float4*)S.cst)[lane] = ((const float4*)L.qn)[lane];
    if (warp == 12) ((float4*)S.cst)[32 + lane] = ((const float4*)L.kn)[lane];
    if (warp == 13 && lane < 2) ((float4*)S.cst)[64 + lane] = ((const float4*)L.sinks)[kvg * 2 + lane];
    if (warp == 14) ((float4*)S.cst)[68 + lane] = ((const float4*)(P->cosT + (size_t)pos * 128))[lane];
    if (warp == 15) ((float4*)S.cst)[100 + lane] = ((const float4*)(P->sinT + (size_t)pos * 128))[lane];
}

// Constants (shared memory) and the cached K and V tiles of this warp's first key group (registers T[0..7], T[8..15])
// of the layer this block serves next. Runs while the block is idle.
__device__ __forceinline__ void attn_stage(const Params* P, const LayerP& L, int b, int kvg, int pos, const AttnSm& S, uint4* T, bool smstage) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t cbase = ((size_t)(b * NKV + kvg) * P->kgmax) * 8 * 32;
    const int lo = L.sliding ? (pos - 512 > 0 ? pos - 512 : 0) : 0;
    const int kg0 = lo >> 4, nkg = (WHATIF & 8) ? 0 : pos > lo ? ((pos - 1) >> 4) - kg0 + 1 : 0;
    attn_consts(P, L, kvg, pos, S);
    if (warp < nkg) {
        const uint4* kt0 = L.kc + cbase + (size_t)(kg0 + warp) * 8 * 32 + lane;
#pragma unroll
        for (int i = 0; i < 8; i++) T[i] = KV_STG_ASM ? lda(kt0 + i * 32) : kt0[i * 32];
    }
    {   // V: this warp mixes dim chunk warp%8 over the key groups of parity warp/8: the first 8 of them in registers
        const int dc = warp & 7, cls = warp >> 3;
        const uint4* vt = L.vc + cbase + (size_t)((kg0 + cls) * 8 + dc) * 32 + lane;
#pragma unroll
        const bool vall = VPAD && kg0 + 16 <= P->kgmax;      // (VPAD: every level, the ones past the context get a zero weight)
#pragma unroll
        for (int j = 0; j < 8; j++) if (vall || cls + 2 * j < nkg) T[8 + j] = KV_STG_ASM ? lda(vt + (size_t)j * 16 * 32) : vt[(size_t)j * 16 * 32];
    }
    // contexts over 256: more key groups go to shared memory (slower than registers, but loaded while the block is
    // idle instead of on the attention's critical path): K of key groups 16..35, V of the next 8 key groups per warp
    if (!smstage) return;
    for (int i = 16 + warp; i < nkg && i < 16 + NK2; i += 16) {
        const uint4* kt = L.kc + cbase + (size_t)(kg0 + i) * 8 * 32 + lane;
        for (int dc = 0; dc < 8; dc++) S.k2[((i - 16) * 8 + dc) * 32 + lane] = kt[dc * 32];
    }
    {
        const int dc = warp & 7, cls = warp >> 3;
        for (int j = 8; j < 8 + NV2 && cls + 2 * j < nkg; j++)
            S.v2[((j - 8) * 16 + warp) * 32 + lane] = L.vc[cbase + (size_t)((kg0 + cls + 2 * j) * 8 + dc) * 32 + lane];
    }
}

// One (sequence b, K/V head kvg) pair = 8 query heads. The key of the current position is handled from registers.
// staged: attn_stage ran for this layer and position (T holds the K and V tiles of key group kg0 + warp).
__device__ __forceinline__ void attention(const Params* P, const LayerP& L, int b, int kvg, int pos, const AttnSm& S, uint4* T, bool staged, bool smstaged, int pblk, int* ppi, bool has_pre, bool bkv, const uint4* ctiles = nullptr, unsigned cxt = 0u, bool k2b = false) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    float* sm = S.sm;
    const size_t cbase = ((size_t)(b * NKV + kvg) * P->kgmax) * 8 * 32;
    uint4* kc = L.kc + cbase;
    uint4* vc = L.vc + cbase;
    const int lo = L.sliding ? (pos - 512 > 0 ? pos - 512 : 0) : 0;
    const int kg0 = lo >> 4, nkg = (WHATIF & 8) ? 0 : pos > lo ? ((pos - 1) >> 4) - kg0 + 1 : 0;
    const float scaling = 0.08838834764831845f;
    if (!staged) attn_consts(P, L, kvg, pos, S);   // slow path: constants from global memory now
    // The words of the output this warp will write hold the sentinel before anything else happens: issued ahead (at
    // the end of the previous layer, has_pre) or now; read back before the first barrier.
    u32 sbad = 0u;
    if (warp < 8) {
        if (!has_pre) attn_sentinel(P, b, kvg);
        sbad = attn_chk(P, b, kvg, 0u);
    }
    // ---- stage 1: raw projections, contiguous 16-byte loads. Every thread: q (head i/64, row group (i/8)%8, g = i%8:
    //      dims rgi*16+g and +8); threads 0..63 also k, 64..127 also v (same dims)
    const int hq = tid >> 6, rgi = (tid >> 3) & 7, gq = tid & 7;
    const int d0 = rgi * 16 + gq, d1 = d0 + 8;
    float v0, v1, k0 = 0.f, k1 = 0.f;
    {
        const float4* qr = P->qraw + (size_t)b * 2560;
        float4 p = ldx(qr + kvg * 512 + tid);
        float4 pk = make_float4(0.f, 0.f, 0.f, 0.f);
        if (tid < 128) pk = ldx(qr + (tid < 64 ? (256 + kvg * 8) * 8 : (288 + kvg * 8) * 8 - 64) + tid);
        // A pair on a main block (batch 7-8): the block's tiles of the output projection are requested now, behind this
        // warp's q/k/v loads (in front of them they held the attention up; after the attention every other block waited
        // 3 us for them). The staging registers are free: this path loads its K/V tiles where it uses them.
        if (ctiles && MATTN_CPRE == 1) tload<16>(T, ctiles);
        if (is_sent(p)) p = ldh(qr + kvg * 512 + tid, P);
        v0 = rbA(p.x + p.z); v1 = rbA(p.y + p.w);
        const float ss = warp_sum(pm2(v0, v0, v1, v1, FM_SS));
        if (lane == 0) sm[warp] = ss;
        if (tid < 128) {
            if (is_sent(pk)) pk = ldh(qr + (tid < 64 ? (256 + kvg * 8) * 8 : (288 + kvg * 8) * 8 - 64) + tid, P);
            k0 = pk.x + pk.z; k1 = pk.y + pk.w;
            if (tid >= 64 || !TB) { k0 = rb(k0); k1 = rb(k1); }     // v is stored as BF16; k is normed first
            if (tid < 64) {
                const float sk = warp_sum(pm2(k0, k0, k1, k1, FM_SS));
                if (lane == 0) sm[16 + warp] = sk;
            }
        }
    }
    if (warp < 8) { CONFIRM(sbad, attn_chk(P, b, kvg, vo)); if (EB && lane == 0) atomicAdd(P->ctr + CX, 1u); }
    PROFX(61);
    __syncthreads();
    // ---- stage 2: norms, rotary (partner dim d^64 is 32 threads away: exchanged through shared memory), fragments, cache stores
    {
        const float sc = rsqrtf(pma(sm[2 * hq] + sm[2 * hq + 1], 1.0f / 128, RMS_EPS, FM_NE));
        v0 = S.cst[d0] * (v0 * sc); v1 = S.cst[d1] * (v1 * sc);
        if (!(TB && L.sliding)) { v0 = rb(v0); v1 = rb(v1); }      // track B: rounded once, after the rotation
        if (tid < 64) {
            const float sck = rsqrtf(pma(sm[16] + sm[17], 1.0f / 128, RMS_EPS, FM_NE));
            k0 = S.cst[128 + d0] * (k0 * sck); k1 = S.cst[128 + d1] * (k1 * sck);
            if (!(TB && L.sliding)) { k0 = rb(k0); k1 = rb(k1); }
        }
        if (L.sliding) {
            S.rq[tid] = make_float2(v0, v1);
            if (tid < 64) ((float2*)(S.ks + 272))[tid] = make_float2(k0, k1);
            __syncthreads();
            const float c0 = S.cst[272 + d0], s0 = S.cst[400 + d0], c1 = S.cst[272 + d1], s1 = S.cst[400 + d1];
            const float2 pq = S.rq[tid ^ 32];
            if (TB && FM_ROT >= 0) {
                v0 = rb(pm2(v0, c0, d0 < 64 ? -pq.x : pq.x, s0, FM_ROT));
                v1 = rb(pm2(v1, c1, d1 < 64 ? -pq.y : pq.y, s1, FM_ROT));
            } else {
                v0 = rb(rbA(v0 * c0) + rbA((d0 < 64 ? -pq.x : pq.x) * s0));
                v1 = rb(rbA(v1 * c1) + rbA((d1 < 64 ? -pq.y : pq.y) * s1));
            }
            if (tid < 64) {
                const float2 pk = ((const float2*)(S.ks + 272))[tid ^ 32];
                if (TB && FM_ROT >= 0) {
                    k0 = rb(pm2(k0, c0, d0 < 64 ? -pk.x : pk.x, s0, FM_ROT));
                    k1 = rb(pm2(k1, c1, d1 < 64 ? -pk.y : pk.y, s1, FM_ROT));
                } else {
                    k0 = rb(rbA(k0 * c0) + rbA((d0 < 64 ? -pk.x : pk.x) * s0));
                    k1 = rb(rbA(k1 * c1) + rbA((d1 < 64 ? -pk.y : pk.y) * s1));
                }
            }
        }
        // element (row = head hq, col = dim gq | gq+8) of tile rgi: lane hq*4 + gq/2, register 0 | 2, half gq&1
        u16* q16 = (u16*)(S.qA + rgi * 32 + hq * 4 + (gq >> 1));
        q16[gq & 1] = (u16)fbits(v0);
        q16[2 + (gq & 1)] = (u16)fbits(v1);
        // (this position's K and V go to shared memory here and to the cache at the very end, behind the output: a
        //  store into the cache waits for its line like a load, and nobody reads these entries before the next step)
        if (tid < 64) { S.ks[d0] = k0; S.ks[d1] = k1; }
        else if (tid < 128) { S.ks[128 + d0] = k0; S.ks[128 + d1] = k1; }
#ifdef KV_STORE_EARLY
        const int kg = pos >> 4, tm = pos & 15;
        if (tid < 64) {
            u16* k16 = (u16*)(kc + (size_t)(kg * 8 + rgi) * 32);
            k16[kidx(tm, gq)] = (u16)fbits(k0);
            k16[kidx(tm, gq + 8)] = (u16)fbits(k1);
        } else if (tid < 128) {
            u16* v16 = (u16*)(vc + (size_t)(kg * 8 + rgi) * 32);
            v16[aidx(gq, tm)] = (u16)fbits(k0);
            v16[aidx(gq + 8, tm)] = (u16)fbits(k1);
        }
#endif
    }
    PROFX(62);
    __syncthreads();
    PROFX(63);
    const int nkgS = (WHATIF & 4096) ? 0 : nkg, nkgE = (WHATIF & 8192) ? 0 : nkg, nkgM = (WHATIF & 16384) ? 0 : nkg;      // (timing studies: the stages' key groups)
    // VPAD: the V mix without a test per tile. Its batches are always 8 tiles; a key group past the context gets the
    // weight zero (one zero word per warp in stage 5) and its tile is whatever the cache holds there (zeros or an
    // earlier sequence's values: finite), so it adds exactly nothing. Not at the very end of the cache or of the
    // weight buffer (128 key groups), where the padded batch would read past them.
    const bool vpad = VPAD && bkv && !(WHATIF & 16384) && kg0 + nkg + 16 <= P->kgmax && nkg + 16 <= 128;
    // ---- stage 3: score of the current key (partial per warp), scores of the cached keys, per-warp maxima.
    //      Warp w owns key groups kg0 + w (scores e0), + 16, + 32, ... (contexts over 256: ex).
    //      Thread (g, t): head g, keys 2t, 2t+1, 2t+8, 2t+9 of each group.
    float e0[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float ex[7][4];
    uint4 Tx = make_uint4(0u, 0u, 0u, 0u);      // (KV_DB) the first V tile of the second batch, requested with the first batch
    {
        const float ps = warp_sum(pm2(v0, S.ks[d0], v1, S.ks[d1], FM_DOT));
        if (lane == 0) sm[64 + warp] = ps;
        float mx = -INFINITY;
        int j = 0;
        if (k2b) {
            // K2B (a pair on a main block: nothing is staged, and the staging registers are all free): two batches of
            // K tiles are in flight, one in each half of the registers, each requested when the batch two rounds earlier
            // has been used; with one batch at a time every round after the first waited a trip to memory, and these
            // pairs are the ones every block waits for.
            auto fin = [&](const float4& c0, const float4& c1, int i, int jj) {
                const int base = (kg0 + i) * 16 + 2 * t;
                const float q0 = (base >= lo && base < pos) ? c0.x * scaling : -INFINITY;
                const float q1 = (base + 1 >= lo && base + 1 < pos) ? c0.y * scaling : -INFINITY;
                const float q2 = (base + 8 >= lo && base + 8 < pos) ? c1.x * scaling : -INFINITY;
                const float q3 = (base + 9 >= lo && base + 9 < pos) ? c1.y * scaling : -INFINITY;
                if (jj == 0) { e0[0] = q0; e0[1] = q1; e0[2] = q2; e0[3] = q3; }
                else { ex[jj - 1][0] = q0; ex[jj - 1][1] = q1; ex[jj - 1][2] = q2; ex[jj - 1][3] = q3; }
                mx = fmaxf(fmaxf(mx, fmaxf(q0, q1)), fmaxf(q2, q3));
            };
            if (warp < nkgS) {
#pragma unroll
                for (int dc = 0; dc < 8; dc++) T[dc] = lda(kc + (size_t)(kg0 + warp) * 8 * 32 + lane + dc * 32);
            }
            if (warp + 16 < nkgS) {
#pragma unroll
                for (int dc = 0; dc < 8; dc++) T[8 + dc] = lda(kc + (size_t)(kg0 + warp + 16) * 8 * 32 + lane + dc * 32);
            }
            for (int i = warp; i < nkgS; i += 32, j += 2) {
                {
                    float4 c0 = make_float4(0.f, 0.f, 0.f, 0.f), c1 = c0;
#pragma unroll
                    for (int dc = 0; dc < 8; dc++) {
                        const uint2 q2 = S.qA[dc * 32 + lane];
                        const uint4 a = make_uint4(q2.x, 0u, q2.y, 0u);
                        mma(c0, a, make_uint2(T[dc].x, T[dc].y));
                        mma(c1, a, make_uint2(T[dc].z, T[dc].w));
                    }
                    if (i + 32 < nkgS) {
#pragma unroll
                        for (int dc = 0; dc < 8; dc++) T[dc] = lda(kc + (size_t)(kg0 + i + 32) * 8 * 32 + lane + dc * 32);
                    }
                    fin(c0, c1, i, j);
                }
                if (i + 16 < nkgS) {
                    float4 c0 = make_float4(0.f, 0.f, 0.f, 0.f), c1 = c0;
#pragma unroll
                    for (int dc = 0; dc < 8; dc++) {
                        const uint2 q2 = S.qA[dc * 32 + lane];
                        const uint4 a = make_uint4(q2.x, 0u, q2.y, 0u);
                        mma(c0, a, make_uint2(T[8 + dc].x, T[8 + dc].y));
                        mma(c1, a, make_uint2(T[8 + dc].z, T[8 + dc].w));
                    }
                    if (i + 48 < nkgS) {
#pragma unroll
                        for (int dc = 0; dc < 8; dc++) T[8 + dc] = lda(kc + (size_t)(kg0 + i + 48) * 8 * 32 + lane + dc * 32);
                    }
                    fin(c0, c1, i + 16, j + 1);
                }
            }
        } else
        for (int i = warp; i < nkgS; i += 16, j++) {
            float4 c0 = make_float4(0.f, 0.f, 0.f, 0.f), c1 = c0;
            if (j == 0 && staged) {
#pragma unroll
                for (int dc = 0; dc < 8; dc++) {
                    const uint2 q2 = S.qA[dc * 32 + lane];
                    const uint4 a = make_uint4(q2.x, 0u, q2.y, 0u);
                    mma(c0, a, make_uint2(T[dc].x, T[dc].y));
                    mma(c1, a, make_uint2(T[dc].z, T[dc].w));
                }
            } else {
                const uint4* kt = (smstaged && i < 16 + NK2 ? S.k2 + (size_t)(i - 16) * 8 * 32 : kc + (size_t)(kg0 + i) * 8 * 32) + lane;
                if (!bkv) {
                for (int dc = 0; dc < 8; dc++) {
                    const uint4 k = kt[dc * 32];
                    const uint2 q2 = S.qA[dc * 32 + lane];
                    const uint4 a = make_uint4(q2.x, 0u, q2.y, 0u);
                    mma(c0, a, make_uint2(k.x, k.y));
                    mma(c1, a, make_uint2(k.z, k.w));
                }
                } else {
                // The 8 tiles of a key group are requested together: one trip to memory per group, not eight. They go
                // into the staging registers of the first key group, which is done (more registers would spill).
#pragma unroll
                for (int dc = 0; dc < 8; dc++) T[dc] = (smstaged && i < 16 + NK2) ? kt[dc * 32] : lda(kt + dc * 32);      // (the staged groups are in shared memory)
#pragma unroll
                for (int dc = 0; dc < 8; dc++) {
                    const uint2 q2 = S.qA[dc * 32 + lane];
                    const uint4 a = make_uint4(q2.x, 0u, q2.y, 0u);
                    mma(c0, a, make_uint2(T[dc].x, T[dc].y));
                    mma(c1, a, make_uint2(T[dc].z, T[dc].w));
                }
                }
            }
            const int base = (kg0 + i) * 16 + 2 * t;
            const float q0 = (base >= lo && base < pos) ? c0.x * scaling : -INFINITY;
            const float q1 = (base + 1 >= lo && base + 1 < pos) ? c0.y * scaling : -INFINITY;
            const float q2 = (base + 8 >= lo && base + 8 < pos) ? c1.x * scaling : -INFINITY;
            const float q3 = (base + 9 >= lo && base + 9 < pos) ? c1.y * scaling : -INFINITY;
            if (j == 0) { e0[0] = q0; e0[1] = q1; e0[2] = q2; e0[3] = q3; }
            else if (EXS && j - 1 < 2) S.exs[(j - 1) * 512 + tid] = make_float4(q0, q1, q2, q3);
            else { ex[j - 1][0] = q0; ex[j - 1][1] = q1; ex[j - 1][2] = q2; ex[j - 1][3] = q3; }
            mx = fmaxf(fmaxf(mx, fmaxf(q0, q1)), fmaxf(q2, q3));
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        if (t == 0) sm[128 + warp * 8 + g] = mx;
#if KV_VEARLY
        if (bkv) {      // the first batch of V levels beyond the staged ones: requested now (the K registers are free), used in the mix
            const int dc = warp & 7, i = (warp >> 3) + (staged ? 16 : 0);
#pragma unroll
            for (int jv = 0; jv < 8; jv++) if (vpad ? i < nkgM : i + 2 * jv < nkgM) T[jv] = lda(vc + (size_t)((kg0 + i + 2 * jv) * 8 + dc) * 32 + lane);
            // KV_DB: the batch after this one is on its way as well. With staged tiles in the V half of the registers
            // only one more register quad is free: it takes the first tile of that batch, which is all of it at a full
            // sliding window (33 key groups: 8 staged + 8 + 1 for one parity); without staging the whole batch fits.
            if (KV_DB) {
                if (staged) { if (i + 16 < nkgM) Tx = lda(vc + (size_t)((kg0 + i + 16) * 8 + dc) * 32 + lane); }
                else {
#pragma unroll
                    for (int jv = 0; jv < 8; jv++) if (i + 16 + 2 * jv < nkgM) T[8 + jv] = lda(vc + (size_t)((kg0 + i + 16 + 2 * jv) * 8 + dc) * 32 + lane);
                }
            }
        }
#endif
    }
    PROFX(64);
    __syncthreads();
    // ---- stage 4: softmax with the sink (per head g): maximum, exponentials, partial sums
    const float sink = S.cst[256 + g];
    const float self = (sm[64 + 2 * g] + sm[64 + 2 * g + 1]) * scaling;
    float M = fmaxf(sink, self);
    {
        float m = -INFINITY;
        for (int w = t; w < 16; w += 4) m = fmaxf(m, sm[128 + w * 8 + g]);
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        M = fmaxf(M, m);
        float s = 0.f;
        for (int q = 0; q < 4; q++) { e0[q] = __expf(e0[q] - M); s += e0[q]; }
        int j = 1;
        for (int i = warp + 16; i < nkgE; i += 16, j++)
            if (EXS && j - 1 < 2) {
                // EXS: the second and third key group of a thread (all a sliding layer has) live in shared memory. As
                // a local array they were on the stack, and a stack access is a memory operation of the warp: it
                // waited behind the V tiles requested at the end of the scores.
                float4 e4 = S.exs[(j - 1) * 512 + tid];
                e4.x = __expf(e4.x - M); s += e4.x; e4.y = __expf(e4.y - M); s += e4.y; e4.z = __expf(e4.z - M); s += e4.z; e4.w = __expf(e4.w - M); s += e4.w;
                S.exs[(j - 1) * 512 + tid] = e4;
            } else
            for (int q = 0; q < 4; q++) { ex[j - 1][q] = __expf(ex[j - 1][q] - M); s += ex[j - 1][q]; }
        s += __shfl_xor_sync(0xffffffffu, s, 1);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        if (t == 0) sm[272 + warp * 8 + g] = s;
    }
    PROFX(65);
    __syncthreads();
    // ---- stage 5: weights (BF16) in the register layout of the B operand of the V mix, to shared memory
    {
        float Ssum = 0.f;
        for (int w = t; w < 16; w += 4) Ssum += sm[272 + w * 8 + g];
        Ssum += __shfl_xor_sync(0xffffffffu, Ssum, 1);
        Ssum += __shfl_xor_sync(0xffffffffu, Ssum, 2);
        Ssum += __expf(sink - M) + __expf(self - M);
        if (warp == 0 && t == 0) S.ks[256 + g] = rb(__expf(self - M) / Ssum);
        int j = 0;
#if SOFTMAX_DIV
        const float rS = 1.0f / Ssum;
#endif
        for (int i = warp; i < nkgE; i += 16, j++) {
            float a0, a1, a2, a3;
            if (j == 0) { a0 = e0[0]; a1 = e0[1]; a2 = e0[2]; a3 = e0[3]; }
            else if (EXS && j - 1 < 2) { const float4 e4 = S.exs[(j - 1) * 512 + tid]; a0 = e4.x; a1 = e4.y; a2 = e4.z; a3 = e4.w; }
            else { a0 = ex[j - 1][0]; a1 = ex[j - 1][1]; a2 = ex[j - 1][2]; a3 = ex[j - 1][3]; }
#if SOFTMAX_DIV
            // The weights e / sum. The compiler's division is 38 ns per warp and there are 16 of them per key group
            // and warp (4224 quotients per pair at a full window): reciprocal once, then the exact quotient in three
            // operations (divr): the same float as the division for every exponential of 2^-102 and more. Below that
            // (a key more than 70 under the largest score) the remainder may not be representable and the weight, itself
            // below 1e-30, can be one unit in its last place off. SOFTMAX_GUARD=1 sends such threads to the real
            // division (the same bits for every input); it is off because the extra branch made the batch > 2 kernel of
            // track A spill (+50 us per step at batch 8), in both forms tried (decided on the inputs, on the results).
            float w0, w1, w2, w3;
#if SOFTMAX_GUARD
            if (min(min(__float_as_uint(a0) - 1u, __float_as_uint(a1) - 1u), min(__float_as_uint(a2) - 1u, __float_as_uint(a3) - 1u)) < 0x0d800000u - 1u) {
                w0 = a0 / Ssum; w1 = a1 / Ssum; w2 = a2 / Ssum; w3 = a3 / Ssum;
            } else
#endif
            { w0 = divr(a0, Ssum, rS); w1 = divr(a1, Ssum, rS); w2 = divr(a2, Ssum, rS); w3 = divr(a3, Ssum, rS); }
#else
            const float w0 = a0 / Ssum, w1 = a1 / Ssum, w2 = a2 / Ssum, w3 = a3 / Ssum;
#endif
            S.wb[i * 32 + lane] = make_uint2(fbits(w0) | (fbits(w1) << 16), fbits(w2) | (fbits(w3) << 16));
        }
        if (vpad) S.wb[(nkg + ((warp - nkg) & 15)) * 32 + lane] = make_uint2(0u, 0u);
    }
    PROFX(65);
    __syncthreads();
    // ---- stage 6: V mix. Warp = (dim chunk dc, parity cls): sums [dim 16][head 8] over the key groups of its parity
    {
        const int dc = warp & 7, cls = warp >> 3;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        if (!bkv) {
#pragma unroll
        for (int j = 0; j < 8; j++) if (staged && cls + 2 * j < nkgM) mma(acc, T[8 + j], S.wb[(cls + 2 * j) * 32 + lane]);
        int i = cls;
        if (staged) i = cls + 16;
        if (smstaged) {
            for (int j = 8; j < 8 + NV2 && cls + 2 * j < nkgM; j++) mma(acc, S.v2[((j - 8) * 16 + warp) * 32 + lane], S.wb[(cls + 2 * j) * 32 + lane]);
            i = cls + 2 * (8 + NV2);
        }
        for (; i < nkgM; i += 2) {
            const uint4 a = vc[(size_t)((kg0 + i) * 8 + dc) * 32 + lane];
            mma(acc, a, S.wb[i * 32 + lane]);
        }
        } else {
        // Levels beyond the staged ones are requested 8 at a time (it was one load per level inside the dependent
        // chain of the mix: a trip to memory each), the first batch before the staged levels are mixed. Same order
        // of the sum: ascending key groups of this warp's parity.
        // The batches use the K half of the staging registers (the scores are done). (Two batches in flight, the
        // second in the V half, and a look-ahead for the K tiles were slower at every context: 1507 -> 1547 us.)
        int i = staged ? cls + 16 : cls;
        if (vpad && KV_VEARLY && !KV_DB) {
            if (staged) {
#pragma unroll
                for (int j = 0; j < 8; j++) mma(acc, T[8 + j], S.wb[(cls + 2 * j) * 32 + lane]);
            }
            while (i < nkgM) {
#pragma unroll
                for (int j = 0; j < 8; j++) mma(acc, T[j], S.wb[(i + 2 * j) * 32 + lane]);
                i += 16;
                if (i < nkgM) {
#pragma unroll
                    for (int j = 0; j < 8; j++) T[j] = lda(vc + (size_t)((kg0 + i + 2 * j) * 8 + dc) * 32 + lane);
                }
            }
        } else {
#if !KV_VEARLY
#pragma unroll
        for (int j = 0; j < 8; j++) if (i + 2 * j < nkgM) T[j] = lda(vc + (size_t)((kg0 + i + 2 * j) * 8 + dc) * 32 + lane);
#endif
#pragma unroll
        for (int j = 0; j < 8; j++) if (staged && cls + 2 * j < nkgM) mma(acc, T[8 + j], S.wb[(cls + 2 * j) * 32 + lane]);
        if (KV_DB && KV_VEARLY) {
            // Two batches alternate between the two halves of the staging registers: while one is mixed, the one after
            // the next is requested into the half that has just been used (a batch was requested when the one before it
            // had been mixed: a trip to memory per batch on the path). The same order of the sum.
            if (staged) {       // the V half is free now: the rest of the second batch (its first tile came early)
                if (i + 16 < nkgM) T[8] = Tx;
#pragma unroll
                for (int j = 1; j < 8; j++) if (i + 16 + 2 * j < nkgM) T[8 + j] = lda(vc + (size_t)((kg0 + i + 16 + 2 * j) * 8 + dc) * 32 + lane);
            }
            while (i < nkgM) {
#pragma unroll
                for (int j = 0; j < 8; j++) if (i + 2 * j < nkgM) mma(acc, T[j], S.wb[(i + 2 * j) * 32 + lane]);
                i += 16;
#pragma unroll
                for (int j = 0; j < 8; j++) if (i + 16 + 2 * j < nkgM) T[j] = lda(vc + (size_t)((kg0 + i + 16 + 2 * j) * 8 + dc) * 32 + lane);
                if (i >= nkgM) break;
#pragma unroll
                for (int j = 0; j < 8; j++) if (i + 2 * j < nkgM) mma(acc, T[8 + j], S.wb[(i + 2 * j) * 32 + lane]);
                i += 16;
#pragma unroll
                for (int j = 0; j < 8; j++) if (i + 16 + 2 * j < nkgM) T[8 + j] = lda(vc + (size_t)((kg0 + i + 16 + 2 * j) * 8 + dc) * 32 + lane);
            }
        } else {
        while (i < nkgM) {
#pragma unroll
            for (int j = 0; j < 8; j++) if (i + 2 * j < nkgM) mma(acc, T[j], S.wb[(i + 2 * j) * 32 + lane]);
            i += 16;
#pragma unroll
            for (int j = 0; j < 8; j++) if (i + 2 * j < nkgM) T[j] = lda(vc + (size_t)((kg0 + i + 2 * j) * 8 + dc) * 32 + lane);
        }
        }
        }
        }
        S.part[warp * 32 + lane] = *(const uint4*)&acc;
        // (MATTN_CPRE == 2: the block's output-projection tiles are requested here, behind the last K/V loads: they
        //  arrive while the leaders store the pair's output)
        if (ctiles && MATTN_CPRE == 2) tload<16>(T, ctiles);
    }
    PROFX(66);
    __syncthreads();
    PROFX(67);
    // ---- stage 7: leaders (warp dc < 8): the two parities, the current key's term, and one 8-byte lane per head and
    //      dim pair into the per-sequence fragment buffer; each leader bumps the attention counter itself (the bump
    //      follows the stores of the same warp)
    if (warp < 8) {
        const int dc = warp;
        const float4 c = f4add(*(const float4*)&S.part[dc * 32 + lane], *(const float4*)&S.part[(8 + dc) * 32 + lane]);
        // c: (dim g, head 2t), (dim g, head 2t+1), (dim g+8, head 2t), (dim g+8, head 2t+1)
        const float vlo = S.ks[128 + dc * 16 + g], vhi = S.ks[128 + dc * 16 + g + 8];
        const float wa = S.ks[256 + 2 * t], wbb = S.ks[256 + 2 * t + 1];
        const u32 b00 = fbits(pma(wa, vlo, c.x, FM_OUT)), b01 = fbits(pma(wbb, vlo, c.y, FM_OUT)), b10 = fbits(pma(wa, vhi, c.z, FM_OUT)), b11 = fbits(pma(wbb, vhi, c.w, FM_OUT));
        // partner lane g^1 (lane ^ 4) holds the odd dim of the pair
        const u32 p00 = __shfl_xor_sync(0xffffffffu, b00, 4), p01 = __shfl_xor_sync(0xffffffffu, b01, 4);
        const u32 p10 = __shfl_xor_sync(0xffffffffu, b10, 4), p11 = __shfl_xor_sync(0xffffffffu, b11, 4);
        if (!(g & 1)) {
            uint2* out = P->attnf + (size_t)b * 256 * 4;
            put64(out + ((kvg * 8 + 2 * t) * 8 + dc) * 4 + (g >> 1), b00 | (p00 << 16), b10 | (p10 << 16));
            put64(out + ((kvg * 8 + 2 * t + 1) * 8 + dc) * 4 + (g >> 1), b01 | (p01 << 16), b11 | (p11 << 16));
        }
        if (!EB && lane == 0) atomicAdd(P->ctr + CX, 1u);
    }
    // (a pair on a main block: "has every pair begun?", which the block's output projection asks next, is asked here
    //  by an idle warp, under the leaders' stores)
    else if (cxt && tid == 8 * 32) poll_ctr(P, CX, cxt);
#ifndef KV_STORE_EARLY
    {
        const int kg = pos >> 4, tm = pos & 15;
        if (tid < 64) {
            u16* k16 = (u16*)(kc + (size_t)(kg * 8 + rgi) * 32);
            k16[kidx(tm, gq)] = (u16)fbits(S.ks[d0]);
            k16[kidx(tm, gq + 8)] = (u16)fbits(S.ks[d1]);
        } else if (tid < 128) {
            u16* v16 = (u16*)(vc + (size_t)(kg * 8 + rgi) * 32);
            v16[aidx(gq, tm)] = (u16)fbits(S.ks[128 + d0]);
            v16[aidx(gq + 8, tm)] = (u16)fbits(S.ks[128 + d1]);
        }
    }
#endif
    PROFX(68);
    __syncthreads();
}

// ------------------------------------------------------------------------------------------------ router top-k
// In every main block (the result is the same everywhere, no hand-over).
// Stage 1, candidates. Batch 1-2: a sequence on 8 warps, one expert per thread; each warp extracts its 6 largest
// sigmoid scores (6 rounds of warp max). Larger batches: ALL sequences at once, a sequence on 2 warps, 4 experts per
// thread (expert = 128 * (warp & 1) + 4 * lane + slot); each warp extracts its 6 largest. Ties go to the lower expert id.
// Stage 2: one warp per sequence holds the candidates (48 or 12) and extracts the 6 best the same way.
// Stage 3, the union over the batch in ascending expert id ("units"): a byte table [expert][sequence] = slot + 1,
// a ballot per warp of the experts in use, prefix counts.
// un[0] = number of units, un[1 + u] = expert of unit u; uwl[u*8 + b] = routing weight (0: not a member).
// Shared memory: sm[b*48 + c], smi[b*48 + c]: candidates; sm[400 + b*8 + r], smi[400 + b*8 + r]: the top 6;
// smi[464 + w]: experts in use in warp w's 32; the byte table reuses smi[0..255] (experts 0..127) and sm[0..255]
// (batch 1-2), or is the first 2 KB of the partial-sum area (batch > 2, UNI2: zeroed in stage 1, written by stage 2).
__device__ __forceinline__ void route_local(const Params* P, const float* rl, int B, float* sm, int* smi, int* un, float* uwl, int pblk, int* ppi, const bool big, const bool zmode = false) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // zmode (track B, early routing): rl holds z = R' x_mid, the router logits before the division by the RMS of x_mid
    // (the top 6 do not depend on it). The logits are z * sc with sc from the partial sums the post-attention norm of
    // this block left in sm[b * 16 + w], added in the same order as there.
    if (zmode) {
        if (tid < B) { float s = 0.f; for (int w = 0; w < 16; w++) s += sm[tid * 16 + w]; sm[480 + tid] = rsqrtf(s * (1.0f / HID) + RMS_EPS); }
        __syncthreads();
    }
    if (!big) {
        const int b = warp >> 3;
        if (b < B) {
            float z = ldx(rl + b * NEXP + (tid & 255));
            if (is_sent(z)) z = ldh(rl + b * NEXP + (tid & 255), P);
            float v = 1.0f / (1.0f + __expf(-(z * (zmode ? sm[480 + b] : 1.0f))));
#if ROUTE_RANK
            // Each thread ranks its score among the 32 of its warp (ties: the lower lane first, as six rounds of
            // "largest, lowest lane" would): 8 loads from shared memory instead of 6 x 7 dependent warp operations.
            float* vb = sm + 4096;                       // (the partial-sum area: free between phases D and E)
            vb[tid] = v;
            __syncwarp();
            {
                const float4* row = (const float4*)(vb + (tid & ~31));
                int rank = 0;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float4 q = row[j];
                    rank += (q.x > v || (q.x == v && 4 * j < lane)) + (q.y > v || (q.y == v && 4 * j + 1 < lane)) + (q.z > v || (q.z == v && 4 * j + 2 < lane)) + (q.w > v || (q.w == v && 4 * j + 3 < lane));
                }
                if (rank < TOPK) { sm[b * 48 + (warp & 7) * 6 + rank] = v; smi[b * 48 + (warp & 7) * 6 + rank] = (warp & 7) * 32 + lane; }
            }
#else
            for (int r = 0; r < TOPK; r++) {
                float m = v;
                for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                const int win = __ffs(__ballot_sync(0xffffffffu, v == m)) - 1;
                if (lane == 0) { sm[b * 48 + (warp & 7) * 6 + r] = m; smi[b * 48 + (warp & 7) * 6 + r] = (warp & 7) * 32 + win; }
                if (lane == win) v = -INFINITY;
            }
#endif
        }
    } else if (WHATIF & 1024) {
    } else {
        const int b = warp >> 1, hf = warp & 1;
        if (UNI2) { ((unsigned*)(sm + 4096))[tid] = 0u; if (tid < MAXU * MAXB) uwl[tid] = 0.f; }
        if (b < B) {
            float4 x = ldx((const float4*)(rl + b * NEXP + hf * 128) + lane);
            if (is_sent(x)) x = ldh((const float4*)(rl + b * NEXP + hf * 128) + lane, P);
            if (zmode) { const float sc = sm[480 + b]; x.x *= sc; x.y *= sc; x.z *= sc; x.w *= sc; }
            float v0 = 1.0f / (1.0f + __expf(-x.x)), v1 = 1.0f / (1.0f + __expf(-x.y)), v2 = 1.0f / (1.0f + __expf(-x.z)), v3 = 1.0f / (1.0f + __expf(-x.w));
#if RSORT
            // The thread's four scores in descending order, equal ones in ascending slot (adjacent exchanges on
            // "strictly larger" only: stable), the slots packed two bits each. A round: the warp maximum of the first
            // scores, the lowest lane holding it stores its score and expert and shifts. The same order as below:
            // largest first, the lower expert on a tie.
            unsigned k0 = 0u, k1 = 1u, k2 = 2u, k3 = 3u;
#define RCE(a, b) { const bool sw_ = v##b > v##a; const float tv_ = sw_ ? v##b : v##a; v##b = sw_ ? v##a : v##b; v##a = tv_; const unsigned tk_ = sw_ ? k##b : k##a; k##b = sw_ ? k##a : k##b; k##a = tk_; }
            RCE(0, 1) RCE(1, 2) RCE(2, 3) RCE(0, 1) RCE(1, 2) RCE(0, 1)
#undef RCE
#if RSORT == 2
            // No rounds (each is five dependent shuffles and a ballot). The rank R of the thread's best score among the
            // 32 of its warp (through shared memory; ties: the lower lane first). The thread's j-th score has at
            // least R + j scores in front of it, so only the first 6 - R scores of the six threads with R < 6 can be
            // among the six best of the warp: 4 + 4 + 4 + 3 + 2 + 1 = 18 candidates, all 18 slots written every time.
            // Lanes 0..17 then rank one candidate each among the 18 (largest first, the lower expert on a tie) and the
            // six best go where the rounds put them, in the same order.
            {
                float* vb = sm + 4096 + 512;                 // (the partial-sum area behind the union's byte table: free between phases D and E)
                vb[tid] = v0;
                __syncwarp();
                const float4* row = (const float4*)(vb + (tid & ~31));
                int R = 0;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float4 q = row[j];
                    R += (q.x > v0 || (q.x == v0 && 4 * j < lane)) + (q.y > v0 || (q.y == v0 && 4 * j + 1 < lane)) + (q.z > v0 || (q.z == v0 && 4 * j + 2 < lane)) + (q.w > v0 || (q.w == v0 && 4 * j + 3 < lane));
                }
                float* cv = sm + 4096 + 1024 + warp * 32; int* ci = (int*)(sm + 4096 + 1536) + warp * 32;
                if (R < 6) {
                    const int o = (int)((0x110F0C080400ull >> (8 * R)) & 255ull), idb = hf * 128 + lane * 4;
                    cv[o] = v0; ci[o] = idb + (int)k0;
                    if (R < 5) { cv[o + 1] = v1; ci[o + 1] = idb + (int)k1; }
                    if (R < 4) { cv[o + 2] = v2; ci[o + 2] = idb + (int)k2; }
                    if (R < 3) { cv[o + 3] = v3; ci[o + 3] = idb + (int)k3; }
                }
                __syncwarp();
                const float va = cv[lane]; const int ia = ci[lane];          // (lanes 18..31: not candidates, not stored)
                int ra = 0;
#pragma unroll
                for (int j = 0; j < 5; j++) {
                    const float4 q = ((const float4*)cv)[j]; const int4 qi = ((const int4*)ci)[j];
                    if (4 * j < 18) ra += (q.x > va || (q.x == va && qi.x < ia));
                    if (4 * j + 1 < 18) ra += (q.y > va || (q.y == va && qi.y < ia));
                    if (4 * j + 2 < 18) ra += (q.z > va || (q.z == va && qi.z < ia));
                    if (4 * j + 3 < 18) ra += (q.w > va || (q.w == va && qi.w < ia));
                }
                if (lane < 18 && ra < TOPK) { sm[b * 48 + hf * 6 + ra] = va; smi[b * 48 + hf * 6 + ra] = ia; }
            }
#else
            unsigned pk = k0 | (k1 << 2) | (k2 << 4) | (k3 << 6);
            for (int r = 0; r < TOPK; r++) {
                float m = v0;
                for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                const int win = __ffs(__ballot_sync(0xffffffffu, v0 == m)) - 1;
                if (lane == win) {
                    sm[b * 48 + hf * 6 + r] = v0; smi[b * 48 + hf * 6 + r] = hf * 128 + lane * 4 + (int)(pk & 3u);
                    v0 = v1; v1 = v2; v2 = v3; v3 = -INFINITY; pk >>= 2;
                }
            }
#endif
#else
            for (int r = 0; r < TOPK; r++) {
                float lm = v0; int ls = 0;                       // this thread's largest, the lowest slot on a tie
                if (v1 > lm) { lm = v1; ls = 1; }
                if (v2 > lm) { lm = v2; ls = 2; }
                if (v3 > lm) { lm = v3; ls = 3; }
                float m = lm;
                for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
                const int win = __ffs(__ballot_sync(0xffffffffu, lm == m)) - 1;
                const int wslot = __shfl_sync(0xffffffffu, ls, win);
                if (lane == 0) { sm[b * 48 + hf * 6 + r] = m; smi[b * 48 + hf * 6 + r] = hf * 128 + win * 4 + wslot; }
                if (lane == win) {
                    v0 = ls == 0 ? -INFINITY : v0; v1 = ls == 1 ? -INFINITY : v1; v2 = ls == 2 ? -INFINITY : v2; v3 = ls == 3 ? -INFINITY : v3;
                }
            }
#endif
        }
    }
    PROFX(71);
    __syncthreads();
    PROFX(72);
    if (warp < B && !(WHATIF & 2048)) {
        // this warp: sequence b = warp. lane holds candidates lane (a) and, with 48 candidates, lane + 32 (c, lanes 0..15)
        const int b = warp;
        const bool wide = !big;
        float va = (wide || lane < 12) ? sm[b * 48 + lane] : -INFINITY, vc = (wide && lane < 16) ? sm[b * 48 + 32 + lane] : -INFINITY;
        const int ia = (wide || lane < 12) ? smi[b * 48 + lane] : 0, ic = (wide && lane < 16) ? smi[b * 48 + 32 + lane] : 0;
        float sum = 0.f, ts[TOPK]; int ti[TOPK];
#if ROUTE_RANK && RKT
        if (wide) {
            // The merge below ranks two candidates per lane among all 48 on this one warp (96 comparisons per lane:
            // 1.25 us of a 28 us layer at batch 1, by the cycle counter). Most of the 48 cannot be among the six best.
            // Its order is: larger score first; on a tie the lower key, key = 2 * slot for slots 0..31 and
            // 2 * (slot - 32) + 1 for slots 32..47 (slot = 6 * warp + rank in the warp). With R = the rank of a warp's
            // best candidate among the 8 best ones (same order), the candidate (warp, r) has at least R + r candidates
            // in front of it: the best ones of the R better warps and its own r predecessors (keys ascend with r in a
            // warp, except in warp 5, whose slots 30..35 straddle the two key ranges). So only r < 6 - R can be among
            // the six best: 6 + 5 + 4 + 3 + 2 + 1 = 21 candidates on lanes 0..20; warp 5's six are all kept, on lanes
            // 21..26 (and left out of the 21). One candidate per lane, 28 comparisons, the same order and ties.
            const float* cb = sm + b * 48; const int* cib = smi + b * 48;
            int* inv = smi + 464 + b * 8;                                          // warp of each rank (free until the union)
            float* cc = sm + 4096 + b * 64; int* ck = (int*)(sm + 4096 + b * 64 + 32);     // (stage 1's score area: free after its barrier)
            {
                const int w = lane & 7;
                const float mw = cb[w * 6];
                const int kw = (int)((0x15093C3024180C00ull >> (8 * w)) & 255ull);       // key of slot 6 * w
                int Rw = 0;
#pragma unroll
                for (int j = 0; j < 8; j++) { const float mj = cb[j * 6]; const int kj = (int)((0x15093C3024180C00ull >> (8 * j)) & 255ull); Rw += (mj > mw || (mj == mw && kj < kw)); }
                if (lane < 8) inv[Rw] = lane;
            }
            __syncwarp();
            const int Rl = (lane >= 6) + (lane >= 11) + (lane >= 15) + (lane >= 18) + (lane >= 20);
            const int st = (int)((0x14120F0B0600ull >> (8 * Rl)) & 255ull);
            int cw = 5, cr = lane - 21; bool have = lane < 27;
            if (lane < 21) { cw = inv[Rl]; cr = lane - st; have = cw != 5; }
            const int slot = cw * 6 + cr;
            const float xa = have ? cb[slot] : -INFINITY; const int xi = cib[slot];
            const int xk = slot < 32 ? 2 * slot : 2 * (slot - 32) + 1;
            cc[lane] = xa; ck[lane] = xk;
            __syncwarp();
            int xr = 0;
#pragma unroll
            for (int j = 0; j < 7; j++) {
                const float4 q = ((const float4*)cc)[j]; const int4 qk = ((const int4*)ck)[j];
                xr += (q.x > xa || (q.x == xa && qk.x < xk)) + (q.y > xa || (q.y == xa && qk.y < xk)) + (q.z > xa || (q.z == xa && qk.z < xk)) + (q.w > xa || (q.w == xa && qk.w < xk));
            }
            if (have && xr < TOPK) { sm[416 + b * 8 + xr] = xa; smi[416 + b * 8 + xr] = xi; }
            __syncwarp();
            for (int r = 0; r < TOPK; r++) { ts[r] = sm[416 + b * 8 + r]; ti[r] = smi[416 + b * 8 + r]; sum += ts[r]; }
        } else
#elif ROUTE_RANK
        if (wide) {
            // rank of this lane's candidates among the 48 (ties: lower lane first, a before c: the order of six rounds
            // of "largest, lowest lane"); the six best go to shared memory in rank order and are summed in that order
            const float4* cv = (const float4*)(sm + b * 48);
            int ra = 0, rc = 0;
#pragma unroll
            for (int j = 0; j < 12; j++) {
                const float4 q = cv[j];
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int d = 4 * j + k, key = (d < 32 ? d : d - 32) * 2 + (d >= 32);
                    const float vd = cget(q, k);
                    ra += (vd > va || (vd == va && key < 2 * lane));
                    rc += (vd > vc || (vd == vc && key < 2 * lane + 1));
                }
            }
            if (ra < TOPK) { sm[416 + b * 8 + ra] = va; smi[416 + b * 8 + ra] = ia; }
            if (lane < 16 && rc < TOPK) { sm[416 + b * 8 + rc] = vc; smi[416 + b * 8 + rc] = ic; }
            __syncwarp();
            for (int r = 0; r < TOPK; r++) { ts[r] = sm[416 + b * 8 + r]; ti[r] = smi[416 + b * 8 + r]; sum += ts[r]; }
        } else
#endif
#if ROUTE_RANK_BIG
        if (!wide) {
            // batch > 2: 12 candidates (lanes 0..11), ranked the same way (ties: the lower lane first); the six best go
            // to the free part of the sequence's candidate area in rank order
            const float4* cv = (const float4*)(sm + b * 48);
            int ra = 0;
#pragma unroll
            for (int j = 0; j < 3; j++) {
                const float4 q = cv[j];
#pragma unroll
                for (int k = 0; k < 4; k++) { const float vd = cget(q, k); ra += (vd > va || (vd == va && 4 * j + k < lane)); }
            }
            if (lane < 12 && ra < TOPK) {
                sm[b * 48 + 16 + ra] = va; smi[b * 48 + 16 + ra] = ia;
                if (UNI5X) ((unsigned char*)(sm + 4096))[(ia & 255) * 8 + b] = (unsigned char)(ra + 1);
            }
            if (!UNI5X) {
                __syncwarp();
                for (int r = 0; r < TOPK; r++) { ts[r] = sm[b * 48 + 16 + r]; ti[r] = smi[b * 48 + 16 + r]; sum += ts[r]; }
            }
        } else
#endif
        for (int r = 0; r < TOPK; r++) {
            float m = fmaxf(va, vc);
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            const int win = __ffs(__ballot_sync(0xffffffffu, va == m || vc == m)) - 1;      // lowest lane holding the maximum
            const int id = __shfl_sync(0xffffffffu, va == m ? ia : ic, win);
            if (lane == win) { if (va == m) va = -INFINITY; else vc = -INFINITY; }
            ts[r] = m; ti[r] = id; sum += m;
        }
        if (!(UNI5X && big) && lane < TOPK) {
            // (RTS, batch > 2: this lane's candidate read from shared memory, where the rank merge left it, instead of
            //  from a local array indexed by the lane, which put the array on the stack)
            const float w = (RTS && ROUTE_RANK_BIG && !wide ? sm[b * 48 + 16 + lane] : ts[lane]) / sum;
            const int id = RTS && ROUTE_RANK_BIG && !wide ? smi[b * 48 + 16 + lane] : ti[lane];
            if (B == 1) {     // the units are the 6 experts in ascending id
                int u = 0;
                for (int j = 0; j < TOPK; j++) u += ti[j] < id;
                un[1 + u] = id & 255; uwl[u * MAXB] = w;
                if (lane == 0) un[0] = TOPK;
            } else {
                sm[400 + b * 8 + lane] = w; smi[400 + b * 8 + lane] = id;
                if (UNI2 && big) ((unsigned char*)(sm + 4096))[(id & 255) * 8 + b] = (unsigned char)(lane + 1);
            }
        }
    }
    PROFX(73);
    __syncthreads();
    if (B == 1 || (WHATIF & 512)) return;
    // ---- union. (The earlier version kept "first occurrence" flags at smi[480..543], past smi's 512 ints and into
    // un[0..31] for batches over 4: a warp could read another warp's freshly written unit as a flag, a rare race.)
    const bool uni2 = UNI2 && big, uni3 = UNI2 == 2 && big;
    if (UNI4 && uni2 && !uni3) {
        // The last pass was 8 warps with 8 sequences per thread (0.55 us of a layer at batch 8, by the cycle counter).
        // All 16 warps: thread tid and thread tid + 256 look at expert tid & 255; both count its unit the same way,
        // the first writes the unit and sequences 0..3, the second sequences 4..7.
        const int tq = tid & 255, wq = warp & 7, bh = tid >> 8;
        const unsigned* tw = (const unsigned*)(sm + 4096) + tq * 2;
        const unsigned lo4 = tw[0], hi4 = tw[1];
        const unsigned bal4 = __ballot_sync(0xffffffffu, (lo4 | hi4) != 0u);
        if (tid < 256 && lane == 0) smi[464 + warp] = __popc(bal4);
        if (UNI5X && warp >= 8 && warp - 8 < B) {
            // (UNI5) the six scores of sequence warp - 8 in rank order: their sum in that order, the weights, the ids
            const int b5 = warp - 8;
            float s5 = 0.f;
            for (int r = 0; r < TOPK; r++) s5 += sm[b5 * 48 + 16 + r];
            if (lane < TOPK) { sm[400 + b5 * 8 + lane] = sm[b5 * 48 + 16 + lane] / s5; smi[400 + b5 * 8 + lane] = smi[b5 * 48 + 16 + lane]; }
        }
        __syncthreads();
        int u = __popc(bal4 & ((1u << lane) - 1u));
#ifndef UNI6
#define UNI6 1                   // the union's last pass, batch > 2: the counts of the warps before this one by two loads and seven selects instead of a loop of up to seven loads (-4 us per step at batch 8, same bits)
#endif
#if UNI6
        {   // (the counts of the warps before this one without a loop: two loads, seven selects)
            const int4 c0 = ((const int4*)(smi + 464))[0], c1 = ((const int4*)(smi + 464))[1];
            u += (wq > 0 ? c0.x : 0) + (wq > 1 ? c0.y : 0) + (wq > 2 ? c0.z : 0) + (wq > 3 ? c0.w : 0) + (wq > 4 ? c1.x : 0) + (wq > 5 ? c1.y : 0) + (wq > 6 ? c1.z : 0);
        }
#else
        for (int w = 0; w < wq; w++) u += smi[464 + w];
#endif
        if ((lo4 | hi4) != 0u && u < MAXU) {
            if (bh == 0) un[1 + u] = tq;
            const unsigned wd = bh ? hi4 : lo4;
            for (int k = 0; k < 4; k++) {
                const int b = bh * 4 + k;
                if (b < B) {
                    const unsigned sl = (wd >> (8 * k)) & 255u;
                    if (sl) uwl[u * MAXB + b] = sm[400 + b * 8 + sl - 1];
                    if (F_ONE) {
                        ((unsigned char*)(sm + 512))[u * MAXB + b] = (unsigned char)sl;
                        if (sl) ((unsigned char*)(sm + 608))[b * 8 + sl - 1] = (unsigned char)u;
                    }
                }
            }
        }
        if (tid == 0) { int n = 0; for (int w = 0; w < 8; w++) n += smi[464 + w]; un[0] = n; }
        __syncthreads();
        return;
    }
    unsigned char* tab = uni2 ? (unsigned char*)(sm + 4096) + tid * 8 : tid < 128 ? (unsigned char*)smi + tid * 8 : (unsigned char*)sm + (tid - 128) * 8;     // this thread's expert (tid < 256)
    if (!uni2) {
        if (tid < 256) { smi[tid] = 0; ((int*)sm)[tid] = 0; }
        if (tid < MAXU * MAXB) uwl[tid] = 0.f;
        __syncthreads();
        if (tid < 8 * B && (tid & 7) < TOPK) {
            const int id = smi[400 + tid] & 255;
            unsigned char* te = id < 128 ? (unsigned char*)smi + id * 8 : (unsigned char*)sm + (id - 128) * 8;
            te[tid >> 3] = (unsigned char)((tid & 7) + 1);
        }
        __syncthreads();
    }
    unsigned lo = 0u, hi = 0u, bal = 0u; int ubase = 0;
    if (tid < 256) {
        if (uni3) {
            // every warp counts the experts in use of the warps before it by itself (one ballot each): no barrier
            const unsigned* t0 = (const unsigned*)(sm + 4096) + lane * 2;
            for (int w = 0; w <= warp; w++) {
                const unsigned l_ = t0[w * 64], h_ = t0[w * 64 + 1];
                const unsigned bl = __ballot_sync(0xffffffffu, (l_ | h_) != 0u);
                if (w < warp) ubase += __popc(bl); else { lo = l_; hi = h_; bal = bl; }
            }
        } else {
            lo = ((const unsigned*)tab)[0]; hi = ((const unsigned*)tab)[1];
            bal = __ballot_sync(0xffffffffu, (lo | hi) != 0u);
            if (lane == 0) smi[464 + warp] = __popc(bal);
        }
    }
    if (!uni3) __syncthreads();
    if (tid < 256) {
        int u = __popc(bal & ((1u << lane) - 1u)) + ubase;
        if (!uni3) for (int w = 0; w < warp; w++) u += smi[464 + w];
        if ((lo | hi) != 0u && u < MAXU) {
            un[1 + u] = tid;
            for (int b = 0; b < B; b++) {
                const unsigned sl = ((b < 4 ? lo : hi) >> (8 * (b & 3))) & 255u;
                if (sl) uwl[u * MAXB + b] = sm[400 + b * 8 + sl - 1];
                if (big && F_ONE) {     // (down phase, F_ONE) rank + 1 of this unit in sequence b's six, and the unit of each rank
                    ((unsigned char*)(sm + 512))[u * MAXB + b] = (unsigned char)sl;
                    if (sl) ((unsigned char*)(sm + 608))[b * 8 + sl - 1] = (unsigned char)u;
                }
            }
        }
        if (uni3) { if (tid == 224) un[0] = ubase + __popc(bal); }
        else if (tid == 0) { int n = 0; for (int w = 0; w < 8; w++) n += smi[464 + w]; un[0] = n; }
    }
    __syncthreads();
}

#ifdef MK_PROF
#define NHB_ 128        // (= NHB, defined with the kernel: the profiling build streams the head on the main blocks only)
#else
#define NHB_ 152
#endif
// generate/serve: block 0 collects the blocks' best tokens (see the call). Not inlined: inlined, its few registers made
// every phase of the kernel slower (step 1499 -> 1617 us), although the step path never runs it.
// AFIRST: a half row group of the q/k/v projection that this block computes for a block with an attention pair
// (tiles 64*oh .. 64*oh+63 of block ob). The owner's warp v would have taken tiles 20v .. 20v+19 of its block, even
// and odd ones in two chains: chain (v, parity) is the calling warp's (8..15), its sum goes to shared memory.
// Not inlined: the kernel has no register to spare, and inlined this code made every phase of every block slower.
#ifndef HOST_INL
#define HOST_INL __noinline__
#endif
__device__ HOST_INL void host_a(const uint4* qkv, const uint2* xf, int Ls, int hostE, float4* hred) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ob = hostE / 5, oh = hostE - ob * 5;
    const int v = (64 * oh) / 20 + ((warp - 8) >> 1), par = warp & 1;
    const int tlo = 20 * v > 64 * oh ? 20 * v : 64 * oh, thi = 20 * v + 19 < 64 * oh + 63 ? 20 * v + 19 : 64 * oh + 63;
    const int i0 = tlo + ((((tlo - 20 * v) & 1) != par) ? 1 : 0);
    const uint4* wo = qkv + (size_t)(ob * 320) * 32 + lane;
    uint4 H[10];
#pragma unroll
    for (int k = 0; k < 10; k++) if (i0 + 2 * k <= thi) H[k] = wo[(size_t)(i0 + 2 * k) * 32];
    float4 cc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 0; k < 10; k++) if (i0 + 2 * k <= thi) mma(cc, H[k], xget(xf, Ls, (ob * 320 + i0 + 2 * k) & 127, lane));
    hred[(warp - 8) * 32 + lane] = cc;
}

__device__ __noinline__ void gather_best(const Params* P, int B, float* sm, int* smi) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int b = tid & 7;
    float v = -INFINITY; int ix = 0x7fffffff;
    if (b < B) {
        float vv[3]; int ii[3];
#pragma unroll
        for (int k = 0; k < 3; k++) { const int idx = tid + 512 * k; if (idx < NHB_ * MAXB) { vv[k] = ldx(P->argv + idx); ii[k] = ldx(P->argi + idx); } }
#pragma unroll
        for (int k = 0; k < 3; k++) { const int idx = tid + 512 * k; if (idx < NHB_ * MAXB && (vv[k] > v || (vv[k] == v && ii[k] < ix))) { v = vv[k]; ix = ii[k]; } }
    }
    for (int o = 8; o <= 16; o <<= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
        if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
    }
    if (lane < 8) { sm[warp * 8 + lane] = v; smi[warp * 8 + lane] = ix; }
    __syncthreads();
}

// ------------------------------------------------------------------------------------------------ the kernel
// shared memory map (bytes)
#define SM_SM 0                  // 4K: 1024 floats (the last 128: this block's own rows of the residual stream)
#define SM_SMI (4 << 10)         // 2K: 512 ints   | attention: rotary exchange of q (4K, with SM_UN)
#define SM_UN (6 << 10)          // 2K: units
#define SM_NW (8 << 10)          // 8K: norm weights of the next norm
#define SM_RED (16 << 10)        // 16K: 32 partial slots
#define SM_QA (32 << 10)         // 4K: q as A operand
#define SM_CST (36 << 10)        // 4K: attention constants and the current key/value
#define SM_XF (40 << 10)         // 64K: B fragments (256 chunks x Ls lanes x 8 bytes)
#define SM_PART (120 << 10)      // 8K: attention partial sums
#define SM_WB (88 << 10)         // 32K: attention weights as B operand (128 key groups); [SM_XF, SM_XF + 48K) is the K stage of attention blocks.
                                 // Router blocks: lo fragments (track B) at SM_XF + 32K (they overlap SM_WB in space, not in time)
#define SM_V2 (88 << 10)         // (unused: NV2 = 0)
#define SM_TOTAL (128 << 10)     // above 160 KB the L1 that is left makes every global load slower (tools/bench12.cu): +8% above 128 KB, 1.5x above 160 KB, 3x above 192 KB
#define NTHR 512
#ifndef NACC
#define NACC 4                    // accumulators per mma chain (1, 2 or 4): the summation order of every matrix-vector product
#endif
#ifdef MK_PROF
#define NHB NMAIN                 // blocks that stream the head (profiling build: main blocks only)
#else
#define NHB NBLK                  // all 152 blocks stream the head
#endif
#define NPA 16                   // tiles of phase A loaded ahead (of 20 per warp)

// mode 0: one step, logits out. mode 1: nsteps greedy steps into gen_out [B][nsteps] (int64).
// mode 2: served cohort: npre prompt steps without the head, then nsteps greedy steps into P->gen [B][maxctx] (int32), progress in ctr[96].
// tok_in[b * tstride + step] for the forced steps (0 .. npre) ; later inputs are the chosen tokens.
// Two instances: BIG = 0 for batches of 1 and 2, BIG = 1 for larger ones. Each is compiled without the other's code
// paths (attention pairs on main blocks, the all-sequences router top-6, member-only fragment copies ...): the kernel
// uses all 128 registers per thread, and a path that is never taken still costs spills in the paths that are.
// BODY_WRAP: the kernel's body as an inlined function that the kernel calls, instead of the kernel itself. The code is
// the same; what changes is the register allocation the compiler arrives at, and that is worth 5% of the batch-1 step:
// with the wrapper the batch 1-2 instance has 36/16 bytes of spills on track B and 68/84 on track A, without it the
// other way round, and the instance with 36/16 is 50-70 us per token faster on either track (journal, findings 13i).
// So each track takes the form in which its batch 1-2 instance draws the better allocation.
#ifndef BODY_WRAP
#define BODY_WRAP TB
#endif
#if BODY_WRAP
template <int BIG>
__device__ __forceinline__ void mk_body(uint4* smem, const Params* Pp, const long long* tok_in, int tstride, float* logits, long long* gen_out, int B, int pos0, int npre, int nsteps, int mode,
          unsigned tm0, unsigned tx0, unsigned tr0) {
#else
template <int BIG>
__global__ void __launch_bounds__(NTHR, 1)
mk_kernel(const Params* Pp, const long long* tok_in, int tstride, float* logits, long long* gen_out, int B, int pos0, int npre, int nsteps, int mode,
          unsigned tm0, unsigned tx0, unsigned tr0) {
    extern __shared__ uint4 smem[];
#endif
    const Params* P = Pp;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int blk = blockIdx.x;
    const bool is_main = blk < NMAIN;
    const int aux = blk - NMAIN;
    const bool is_router = aux >= NAUX - NROUT;
    const int rj = aux - (NAUX - NROUT);                      // router row group of this block
    const int nattn_blocks = 4 * B < NAUX ? 4 * B : NAUX;     // attention blocks
#ifdef MK_PROF
    const int npairs = 4 * B - (4 * B > 7 ? 1 : 0);     // the clock block's pair is skipped (timing only)
#else
    const int npairs = 4 * B;                                 // (sequence, K/V head) pairs: 8 leader bumps each
#endif
    const bool has_attn = !is_main && aux < nattn_blocks;
    const bool big = BIG != 0;                                 // B > 2 (the host picks the instance)
    // AFIRST (batch 7-8): the main blocks that carry an attention pair (blocks 0..3 or 0..7) do no phase A. Their
    // registers hold the pair's K/V tiles from the end of the previous layer, and they run the attention as soon as
    // q/k/v arrive, like an attention block (they used to start it after their own share of A, with nothing staged,
    // and every other block waited 6-9 us per layer for their output projection). Their 5 half row groups of the
    // q/k/v projection are hosted, one each, by blocks 8, 11, 14, ...: the host's warps 8..15 compute the 64 tiles in
    // the chains the owner's warps would have used (same sums in the same order: same bits), warp 5 is the leader.
    // (No value of its own lives through the kernel for this: afirst is mattn, the host index is recomputed where used.)
    const bool mattn = big && is_main && blk + NAUX < 4 * B;
#ifndef AF_TEST
#define AF_TEST 0
#endif
#ifndef AF_T
#define AF_T 0                   // (tests: 1 = the pair blocks do not stage K/V ahead, 2 = they request their old phase-A tiles instead)
#endif
#define afirst (AFIRST && !AF_TEST && mattn)
    // HOSTC (batch 7-8). A main block with an attention pair (blocks 0..3 or 0..7) came out of its pair's attention when
    // every other main block had long had its output-projection tiles and most of the attention output, and then needed
    // 4.5 us for its own output projection (tiles, copy of the attention output, mma, residual): every block waited for
    // its 16 rows before the next norm. Now block HOSTC_B + o computes the output projection of block o after its own: the
    // same 16 warps x 16 tiles with the same fragments (the same sums: same bits), the owner's rows of the layer input
    // read from the stream, the result written into the owner's rows of the post-attention stream once the owner's
    // sentinel is there. The owner reads its rows back (it keeps its rows in shared memory) and does nothing else in C.
#define hostc_own (HOSTC && mattn)
    // The expert phase's row groups by block: block b takes [ESPLIT(b), ESPLIT(b + 1)) of the G row groups of the
    // step's units. ES_WM : ES_WO is the share of a main block with an attention pair against the others' (batch 7-8;
    // these blocks reach the phase 1.5 to 3 us after the others and every block's down phase waits for their hidden
    // units). Equal weights give b * G / 128.
#define ES_NM (big && B > 6 ? 4 * B - NAUX : 0)
#define ESPLIT(b, G) ((((b) < ES_NM ? (b) * ES_WM : ES_NM * ES_WM + ((b) - ES_NM) * ES_WO) * (G)) / (ES_NM * ES_WM + (NMAIN - ES_NM) * ES_WO))
#ifndef HOSTC_B
#define HOSTC_B 24               // the first host block (the hosts are HOSTC_B .. HOSTC_B + 7; any 8 main blocks without a pair would do). Scored workload, us per step: 8: 3512.5, 16: 3509, 24: 3502.6, 32: 3508, 40: 3509, 56: 3510, 88: 3516, 104: 3508, 120: 3533
#endif
#define HOSTC_O() ((HOSTC && big && is_main && B > 6 && blk >= HOSTC_B && blk < HOSTC_B + 4 * B - NAUX) ? blk - HOSTC_B : -1)
#ifndef AF_NEVER
#define AF_NEVER 0               // (test: the hosted code is compiled in but never runs)
#endif
#define HOST_E() ((AFIRST && big && is_main && B > 6 + 100 * AF_NEVER && blk >= 8 && (blk - 8) % 3 == 0 && (blk - 8) / 3 < 5 * (4 * B - NAUX)) ? (blk - 8) / 3 : -1)          // more pairs than aux blocks (batch 7, 8): main block m takes pair NAUX + m while it waits
    const bool rpre = is_router && !big && !ER;        // (track B: the early router loads its own tiles)
    const bool late_stage = ER && is_router && aux < nattn_blocks;       // track B router blocks with an attention pair: staged at the layer's start, after pass 1                     // router tiles loaded ahead into registers (no attention duty on these blocks then)
    const bool stg = !is_router || big;                      // attention K/V tiles in registers
    // Router blocks that also have an attention pair (batch > 2): after their attention they first request the router
    // tiles into the staging registers and stage the next layer's K/V only after the router duty. (Staging first left
    // the router tiles to be loaded when needed: a trip through the busy memory queue on the path every main block waits on.)
    const bool rdefer = RDEFER && big && is_router && has_attn && !late_stage;
    // ... and further key groups in shared memory. Router blocks need that memory for their fragments; batch > 2
    // (RSTG): their staging is put off until the router duty is done (rdefer), so the fragment buffer is free
    // from then until their next attention is over, and they stage there like the other attention blocks.
    const bool stgsm = !is_router || (RSTG && rdefer);
    const int Ls = big ? (B <= 4 ? 16 : 32) : (B <= 1 ? 4 : 8);   // lanes per chunk of the shared-memory B fragments
    char* sb = (char*)smem;
    uint4* nscr = (big && is_main) ? (uint4*)(sb + SM_XF + (32 << 10)) : nullptr;      // norm scratch (main blocks, batch > 2): 48 KB at most
    float* sm = (float*)(sb + SM_SM);
    int* smi = (int*)(sb + SM_SMI);
    int* un = (int*)(sb + SM_UN);
    float* uwl = (float*)(un + 64);
    float* nw = (float*)(sb + SM_NW);
    float4* red = (float4*)(sb + SM_RED);
    uint2* xf = (uint2*)(sb + SM_XF);
    float4* xo = (float4*)(sm + 896);            // this block's own 16 rows of the residual stream
    AttnSm AS;
    AS.qA = (uint2*)(sb + SM_QA); AS.wb = (uint2*)(sb + SM_WB);
    AS.cst = (float*)(sb + SM_CST); AS.ks = AS.cst + 528; AS.rq = (float2*)(sb + SM_SMI);
    AS.part = (uint4*)(sb + SM_PART); AS.k2 = (uint4*)(sb + SM_XF); AS.v2 = (uint4*)(sb + SM_V2); AS.sm = sm;
    AS.exs = (float4*)(sb + SM_RED);      // (the partial-sum slots: not in use while a block runs an attention pair)
#ifdef MK_PROF
    int pi = 0;
    bool profl = false;
#define PPI (profl ? &pi : nullptr)
    if (aux == 7) {
        if (tid == 0) {
            unsigned n = 0;
            volatile unsigned* stop = P->clk + 1;
            volatile unsigned* c = P->clk;
            while (true) { n++; *c = n; if ((n & 255u) == 0 && *stop) break; }
        }
        return;
    }
#endif
#ifdef MK_CLK
#define PPI (l == MK_CLK_L ? (int*)(sb + SM_SM + 648 * 4) : nullptr)      // (where the attention stamps its stages)
#elif !defined(MK_PROF)
#define PPI nullptr
#endif
    // zero what is never written: fragment lanes of unused sequences, rows 8..15 of the q operand
    for (int i = tid; i < (64 << 10) / 4; i += NTHR) ((u32*)xf)[i] = 0u;
    for (int i = tid; i < 1024; i += NTHR) ((u32*)AS.qA)[i] = 0u;
    static_assert(SM_XF - SM_CST == 4096, "the zero word of xget is the last 8 bytes of the constants area");
    if (tid == 0) xf[-1] = make_uint2(0u, 0u);      // (constants and current key/value end at byte 3712 of that area)
    if (TB && is_router) for (int i = tid; i < (32 << 10) / 4; i += NTHR) ((u32*)(sb + SM_XF + (32 << 10)))[i] = 0u;
    __syncthreads();

    unsigned tm = tm0, tx = tx0, tr = tr0;
    u32 hdep = 0u;
    const float4 zero4 = make_float4(0.f, 0.f, 0.f, 0.f);
    const int total = npre + nsteps;
    uint4 T[NPA];       // weight tiles loaded ahead into registers (main: next phase; router: its 8 tiles; attention: K and V tiles)

    for (int step = 0; step < total; step++) {
        if (tid == 0) smi[500] = (int)*(volatile unsigned*)(P->ctr + CABORT);
        __syncthreads();
        if (smi[500]) break;                      // a wait gave up somewhere: leave
        const int pos = pos0 + step;
        const bool forced = step <= npre;
        const long long* toks = forced ? tok_in : P->curtok;
        const int ts = forced ? tstride : 1, tcol = forced ? step : 0;
        const bool do_head = step >= npre;
        if (is_main) {
            if (afirst) { if (AF_T != 1) attn_stage(P, P->L[0], (blk + NAUX) >> 2, (blk + NAUX) & 3, pos, AS, T, true); if (AF_T == 2) tload<NPA>(T, P->L[0].qkv + (size_t)(blk * 320 + 20 * warp) * 32 + lane); }
            else {
                tload<NPA>(T, P->L[0].qkv + (size_t)(blk * 320 + 20 * warp) * 32 + lane);
                nw_load(nw, P->L[0].ln1);
            }
            if (step > 0) wait_ctr(P, CM, tm); else __syncthreads();
        } else {
            if (has_attn && stg && !late_stage && step == 0) attn_stage(P, P->L[0], aux >> 2, aux & 3, pos, AS, T, stgsm);
            if (is_router && step == 0) { if (rpre) tload<8>(T, P->L[2].router + (size_t)(rj * 128 + 8 * warp) * 32 + lane); nw_load(nw, P->L[2].ln2); }
            __syncthreads();
        }

        for (int l = 0; l < NLAYER; l++) {
            const LayerP& L = P->L[l];
#ifdef MK_PROF
            profl = (l % 6) == 2;
            PROF(99);
#endif
#ifdef WD_TEST
            if (step == 2 && l == 5 && blk == 3 && pos == 2) wait_ctr(P, CM, tm + 100000000u);       // (test of the watchdog path: a wait that cannot be met)
#endif
            if (is_main) {
                // ============ A: input norm, q/k/v projections. A block owns 320 consecutive tiles (5 half row groups), 20 per warp.
                PROF(1); BT(0);
                if (afirst) {
                    // (no phase A here, see AFIRST) this block's own 16 rows of the stream at the first layer
                    if (l == 0 && (tid >> 2) == blk) {
                        const int g0 = 2 * (tid & 3), r0 = blk * 16 + g0;
                        for (int t2 = 0; t2 < ((B + 1) >> 1); t2++) {
                            const u16* e0 = P->embed + (size_t)((unsigned long long)ldx(toks + (size_t)(2 * t2) * ts + tcol) % VOCAB) * HID;
                            float4 xa = make_float4(b2f(e0[r0]), 0.f, b2f(e0[r0 + 8]), 0.f), xb = make_float4(b2f(e0[r0 + 1]), 0.f, b2f(e0[r0 + 9]), 0.f);
                            if (2 * t2 + 1 < B) {
                                const u16* e1 = P->embed + (size_t)((unsigned long long)ldx(toks + (size_t)(2 * t2 + 1) * ts + tcol) % VOCAB) * HID;
                                xa.y = b2f(e1[r0]); xa.w = b2f(e1[r0 + 8]); xb.y = b2f(e1[r0 + 1]); xb.w = b2f(e1[r0 + 9]);
                            }
                            xo[g0 * 4 + t2] = xa; xo[(g0 + 1) * 4 + t2] = xb;
                        }
                    }
                    tm += KA * NMAIN;
                    PROF(3); BT(1);
                    nw_load(nw, L.ln2);
                    wait_ctr(P, CM, tm);      // every other block has begun A: q/k/v hold the sentinel or this layer's values
                    // (so every block is through the previous layer: the sentinels of this block's output projection can go out now)
                    if (warp == 0) {
                        x_sentinel(P, blk, lane, B, true);
                        if (L.moe) h_sent(P, blk, lane, B, HSLOT_SHARED * 32);
                    }
                    attention(P, L, (blk + NAUX) >> 2, (blk + NAUX) & 3, pos, AS, T, AF_T == 0, AF_T == 0, blk, PPI, l > 0, false, nullptr, tx + 8 * npairs);
                    if (P->fence) __threadfence();
                    __syncthreads();
                    tload<16>(T, L.o + (size_t)(blk * 256 + 16 * warp) * 32 + lane);      // preload C
                } else {
                // The five leaders' q/k/v words hold the sentinel: issued at the end of the previous layer (first layer of
                // a step: now), confirmed here. The signal follows the norm: it also says "this block has read x".
                const int hostE = HOST_E();
                {
                    u32 qbad = 0u;
                    if (warp < 5) {
                        if (l == 0) q_sentinel(P, blk, B);
                        qbad = q_chk(P, blk, B, 0u);
                    } else if (!AF_TEST && hostE >= 0 && warp == 5) {
                        if (l == 0) q_sent_hg(P, hostE, B);
                        qbad = q_chk_hg(P, hostE, B, 0u);
                    }
                    const int dblk = (AFIRST && big && B > 6) ? 8 : 0;      // (the block that keeps the debug copy of the layer input)
                    norm_frag<big && NORM_WS>(P, l == 0 ? P->embed : nullptr, toks, ts, tcol, nw, B, Ls, xf, sm, blk == dblk, (P->dbg && blk == dblk) ? P->dbg + (size_t)l * MAXB * HID : nullptr, xo, blk, nullptr, nscr);
                    if (warp < 5) { CONFIRM(qbad, q_chk(P, blk, B, vo)); if (EB && lane == 0) atomicAdd(P->ctr + CM, 1u); }
                    else if (!AF_TEST && hostE >= 0 && warp == 5) { CONFIRM(qbad, q_chk_hg(P, hostE, B, vo)); if (EB && lane == 0) atomicAdd(P->ctr + CM, 1u); }
                }
                PROF(2);
                {
                    const int g0 = 20 * warp;
                    int nA = 64 - (g0 & 63); if (nA > 20) nA = 20;
                    float4 cA = zero4, cB = zero4, cA1 = zero4, cB1 = zero4;      // two accumulators per group: shorter dependent mma chains
                    const uint4* wt = L.qkv + (size_t)(blk * 320 + g0) * 32 + lane;
#pragma unroll
                    for (int i = 0; i < 20; i++) {
                        const uint4 a = i < NPA ? T[i < NPA ? i : 0] : wt[i * 32];
                        const uint2 bb = xget(xf, Ls, (blk * 320 + g0 + i) & 127, lane);
                        if (i < nA) { if (i & (NACC > 1)) mma(cA1, a, bb); else mma(cA, a, bb); }
                        else { if (i & (NACC > 1)) mma(cB1, a, bb); else mma(cB, a, bb); }
                    }
                    red[(warp * 2) * 32 + lane] = f4add(cA, cA1);
                    red[(warp * 2 + 1) * 32 + lane] = f4add(cB, cB1);
                }
                if (hostE >= 0 && warp >= 8) host_a(L.qkv, xf, Ls, hostE, (float4*)(sb + SM_PART));
                PROF(31);
                __syncthreads();
                PROF(32);
                // HSTG (with HOSTC): an owner needs no output-projection tiles, so its staging registers are free from here
                // on: the K/V tiles of its pair's first key groups go there now (the pair used to load them where it used
                // them, and finished 2.6 us after the pairs on the attention blocks). The leaders first store q/k/v.
                if (HSTG && hostc_own && warp >= 5) attn_stage(P, L, (blk + NAUX) >> 2, (blk + NAUX) & 3, pos, AS, T, false);
                if (warp < 5) {       // five leaders, one half row group each
                    const int h = warp;
                    const int w_lo = (64 * h) / 20, w_hi = (64 * h + 63) / 20;
                    float4 c = zero4;
                    for (int w = w_lo; w <= w_hi; w++) c = f4add(c, red[(w * 2 + (((20 * w) >> 6) == h ? 0 : 1)) * 32 + lane]);
                    // half hh of row group rg of the stacked [q; k; v] matrix: one 8-byte store per lane and sequence
                    const int hg = blk * 5 + h, rg = hg >> 1, hh = hg & 1;
                    for (int j = 0; j < 2; j++) {
                        const int b = 2 * t + j;
                        if (b < B) put64((float2*)(P->qraw + (size_t)b * 2560 + rg * 8 + g) + hh, __float_as_uint(j ? c.y : c.x), __float_as_uint(j ? c.w : c.z));
                    }
                    if (!EB && lane == 0) atomicAdd(P->ctr + CM, 1u);
                }
                else if (hostE >= 0 && warp == 5) {      // leader of the hosted half row group
                    const int oh = hostE % 5, nv = (64 * oh + 63) / 20 - (64 * oh) / 20 + 1;
                    const float4* hred = (const float4*)(sb + SM_PART);
                    float4 c = zero4;
                    for (int k = 0; k < nv; k++) c = f4add(c, f4add(hred[(2 * k) * 32 + lane], hred[(2 * k + 1) * 32 + lane]));
                    const int rg = hostE >> 1, hh = hostE & 1;
                    for (int j = 0; j < 2; j++) {
                        const int b = 2 * t + j;
                        if (b < B) put64((float2*)(P->qraw + (size_t)b * 2560 + rg * 8 + g) + hh, __float_as_uint(j ? c.y : c.x), __float_as_uint(j ? c.w : c.z));
                    }
                    if (!EB && !AF_TEST && lane == 0) atomicAdd(P->ctr + CM, 1u);
                }
                // A main block with an attention pair (batch 7-8): "has every block started A?" (then q/k/v hold the sentinel
                // or this layer's values) is asked now by a warp that is idle while the leaders reduce and store, not after
                // them by thread 0: one trip to memory less in front of the attention every other block waits for.
                else if (MATTN_EPOLL && !afirst && mattn && tid == 5 * 32) poll_ctr(P, CM, tm + KA * NMAIN);
                if (HSTG && hostc_own && warp < 5) attn_stage(P, L, (blk + NAUX) >> 2, (blk + NAUX) & 3, pos, AS, T, false);
                PROF(33);
                __syncthreads();      // the leaders' writes and counters go out before the preload traffic
                PROF(34);
                tm += KA * NMAIN;
                PROF(3); BT(1);
                nw_load(nw, L.ln2);
                if (!afirst && mattn) {
                    if (!MATTN_EPOLL) wait_ctr(P, CM, tm);
                    // MATTN_TAIL: what stood between the pair's last store and this block's output projection (its
                    // sentinels, the question "has every pair begun?") moves in front of and inside the attention.
                    // (Every block has begun A, so every block is through the previous layer: the sentinels may go out.)
                    // (Measured: these two exchanges in front of the attention delay it by 1.9 us: the warp's next loads wait
                    //  for them. With HOSTC they are issued after the attention, as before.)
                    if (MATTN_TAIL && !HOSTC && warp == 0) {
                        x_sentinel(P, blk, lane, B, true);
                        if (L.moe) h_sent(P, blk, lane, B, HSLOT_SHARED * 32);
                    }
                    attention(P, L, (blk + NAUX) >> 2, (blk + NAUX) & 3, pos, AS, T, HSTG && HOSTC, false, blk, PPI, l > 0, KV_BATCH_MAIN != 0,
                              MATTN_CPRE && !HOSTC ? L.o + (size_t)(blk * 256 + 16 * warp) * 32 + lane : nullptr, MATTN_TAIL && !HOSTC ? tx + 8 * npairs : 0u, K2B && KV_BATCH_MAIN && !(HSTG && HOSTC));
                    if (P->fence) __threadfence();
                    __syncthreads();
                }
                // preload C (HOSTC: a host starts with its owner's tiles, so that the owner's rows are not late; its own follow)
                if (!(mattn && (MATTN_CPRE || HOSTC))) { const int ho = HOSTC_O(); tload<16>(T, L.o + (size_t)((ho >= 0 ? ho : blk) * 256 + 16 * warp) * 32 + lane); }
                }
                PROF(35);
                // ============ (B: attention on the aux blocks) ============ C: output projection + residual
                // Sentinel into this block's rows of the post-attention stream and into the shared expert's hidden units
                // (their readers, the previous layer's phase D and down phase, are done: this block has read the whole
                // layer input, which every block's down phase had to write): issued before the wait for the attention,
                // confirmed after it.
                if (!afirst && !(MATTN_TAIL && !HOSTC && mattn) && warp == 0) {
                    const u32 xdep = x_sentinel(P, blk, lane, B, true);
                    if (L.moe) h_sent(P, blk, lane, B, HSLOT_SHARED * 32);
                    // HOSTC, an owner: its rows are written by its host, which waits for this sentinel; a read-back would
                    // race with that write (it could find the value and wait for the sentinel for ever), so the owner
                    // waits for the exchange itself to return instead: then the sentinel is, or has been, in the words.
                    if (hostc_own) SETTLE(xdep);
                }
                tx += 8 * npairs;
                // (a block with a pair: asked inside the attention, or, HOSTC, it reads no attention output at all)
                if (!afirst && !((MATTN_TAIL || HOSTC) && mattn)) wait_ctr(P, CX, tx);
#if ER
                if (L.moe) wait_ctr(P, CZ, tr + NROUT);      // (the early router has read the layer input, too)
#endif
                PROF(4); BT(2);
                {
                    u32 mbad = 0u;
                    // HS_EARLY (batch > 2): the sentinels of the expert phase's hidden units are issued here, for every
                    // unit slot and sequence, whatever the routing will be, by the main blocks that are waiting for the
                    // attention anyway (not by the blocks with a pair). They stood between the routing and the first
                    // expert tiles (timing study WHATIF bit 8: 81 us per step). Every block is through the previous
                    // layer here (the pairs began after every block had begun A), so the previous readers are done;
                    // the read-back is confirmed before this block's C signal, and nobody writes a hidden unit before
                    // every block has signalled C. A word of a (unit, sequence) that is not a member keeps the sentinel:
                    // nobody reads it. 48 slots x 64 row groups, 24 to 26 per block, one or two per warp.
                    const int hs_nm = big && B > 6 ? 4 * B - NAUX : 0, hs_per = (48 * 64 + NMAIN - hs_nm - 1) / (NMAIN - hs_nm);
                    const int hq0 = (blk - hs_nm) * hs_per + warp, hq1 = hq0 + 16;
                    const bool hs_do = HS_EARLY && big && L.moe && !mattn;
                    const bool hs1 = hs_do && hq0 < 48 * 64, hs2 = hs_do && warp + 16 < hs_per && hq1 < 48 * 64;
                    if (hs1) h_sent(P, hq0 & 63, lane, B, (hq0 >> 6) * 32);
                    if (hs2) h_sent(P, hq1 & 63, lane, B, (hq1 >> 6) * 32);
#ifndef C_TEST
#define C_TEST 0                 // (timing tests on the pair blocks, wrong results: 1 no copy, 2 no sentinel check, 3 neither)
#endif
                    if (warp == 0 && !((C_TEST & 2) && mattn)) { if (!hostc_own) mbad = x_chk(P, blk, lane, B, true, 0u); if (L.moe) mbad |= hs_chk(P, blk, lane, B, HSLOT_SHARED * 32, 0u); }
                    const int hown = HOSTC_O();
                    float4* xh = (float4*)(sb + SM_PART);      // (host) the owner's 16 rows of the layer input, laid out like xo
                    if (hown >= 0 && warp == 1 && 2 * t < B) {
                        // The owner's rows of the layer input. They are valid: this block's norm has read every row, and
                        // the owner puts its next sentinel there only after it has read what this block writes below.
                        float4 v;
                        if (l == 0) {
                            const int r0 = hown * 16 + g;
                            const u16* e0 = P->embed + (size_t)((unsigned long long)ldx(toks + (size_t)(2 * t) * ts + tcol) % VOCAB) * HID;
                            v = make_float4(b2f(e0[r0]), 0.f, b2f(e0[r0 + 8]), 0.f);
                            if (2 * t + 1 < B) {
                                const u16* e1 = P->embed + (size_t)((unsigned long long)ldx(toks + (size_t)(2 * t + 1) * ts + tcol) % VOCAB) * HID;
                                v.y = b2f(e1[r0]); v.w = b2f(e1[r0 + 8]);
                            }
                        } else {
                            const uint2 w = ldx((const uint2*)P->xc + t * 1024 + hown * 8 + g);
                            v = make_float4(b2f((u16)w.x), b2f((u16)(w.x >> 16)), b2f((u16)w.y), b2f((u16)(w.y >> 16)));
                        }
                        xh[lane] = v;
                    }
                    if (((C_TEST & 1) && mattn) || hostc_own || (WHATIF & 128)) {}
                    else if (big && FCB) frag_copy_n<FCB ? FCB : 1>(xf, Ls, 256, B, [&](int b, int chunk) { return P->attnf + ((size_t)b * 256 + chunk) * 4; }, P);
                    else frag_copy(xf, Ls, 256, (WHATIF & 32) ? 1 : B, [&](int b, int chunk) { return P->attnf + ((size_t)b * 256 + chunk) * 4; }, P);
                    PROF(76);
                    if (warp == 0 && !((C_TEST & 2) && mattn)) CONFIRM(mbad, (hostc_own ? 0u : x_chk(P, blk, lane, B, true, vo)) | (L.moe ? hs_chk(P, blk, lane, B, HSLOT_SHARED * 32, vo) : 0u));
                    if (hs1 || hs2) {
                        u32 hbad = (hs1 ? hs_chk(P, hq0 & 63, lane, B, (hq0 >> 6) * 32, 0u) : 0u) | (hs2 ? hs_chk(P, hq1 & 63, lane, B, (hq1 >> 6) * 32, 0u) : 0u);
                        CONFIRM(hbad, (hs1 ? hs_chk(P, hq0 & 63, lane, B, (hq0 >> 6) * 32, vo) : 0u) | (hs2 ? hs_chk(P, hq1 & 63, lane, B, (hq1 >> 6) * 32, vo) : 0u));
                    }
                    PROF(77);
                }
                __syncthreads();
                PROF(78);
                if (EB && tid == 0) atomicAdd(P->ctr + CM, 1u);      // (after the barrier: it also says "this whole block has read the attention output")
                const int hown2 = HOSTC_O();
                if (!hostc_own) {
                    float4 c4[4] = {zero4, zero4, zero4, zero4};
#pragma unroll
                    for (int i = 0; i < 16; i++) mma(c4[i & (NACC - 1)], T[i], xget(xf, Ls, 16 * warp + i, lane));
                    red[((hown2 >= 0 ? 16 : 0) + warp) * 32 + lane] = f4add(f4add(c4[0], c4[1]), f4add(c4[2], c4[3]));      // (a host: this was its owner's pass)
                }
                if (hown2 >= 0) tload<16>(T, L.o + (size_t)(blk * 256 + 16 * warp) * 32 + lane);      // a host's own tiles: on their way while the owner's rows are folded and written
                PROF(79);
                __syncthreads();
                PROF(80);
                if (hown2 >= 0) {
                    // HOSTC: the owner's rows (its 16 warps x 16 tiles were this block's first pass, with the same fragments)
                    if (warp == 1 && 2 * t < B) {
                        float4 c = red[16 * 32 + lane];
                        for (int w = 1; w < 16; w++) c = f4add(c, red[(16 + w) * 32 + lane]);
                        const float4 xq = ((const float4*)(sb + SM_PART))[lane];
                        const float4 xn = make_float4(rbA(xq.x + rbA(c.x)), rbA(xq.y + rbA(c.y)), rbA(xq.z + rbA(c.z)), rbA(xq.w + rbA(c.w)));
                        uint2* xp = (uint2*)P->xm + t * 1024 + hown2 * 8 + g;
                        // the owner's sentinel has to be in the words before the value goes there (an exchange that
                        // arrived after this store would wipe the value out)
                        unsigned s_ = 0;
                        while (ldx((const uint2*)xp).x != SENT) {
                            if ((++s_ & (LDH_AB - 1)) == 0) {
                                if (((volatile unsigned*)P->ctr)[CABORT]) break;
                                if (s_ > (1u << 22)) { ((volatile unsigned*)P->ctr)[CABORT] = 1u; break; }
                            }
                        }
                        put64(xp, (__float_as_uint(xn.x) >> 16) | (__float_as_uint(xn.y) & 0xFFFF0000u), (__float_as_uint(xn.z) >> 16) | (__float_as_uint(xn.w) & 0xFFFF0000u));
                    }
                    {   // the host's own pass
                        float4 c4[4] = {zero4, zero4, zero4, zero4};
#pragma unroll
                        for (int i = 0; i < 16; i++) mma(c4[i & (NACC - 1)], T[i], xget(xf, Ls, 16 * warp + i, lane));
                        red[warp * 32 + lane] = f4add(f4add(c4[0], c4[1]), f4add(c4[2], c4[3]));
                    }
                    __syncthreads();
                }
                if (warp == 0) {
                  if (hostc_own) {
                    // this block's rows of the post-attention stream, written by block blk + HOSTC_B over this block's sentinel
                    if (2 * t < B) {
                        const uint2* xp = (const uint2*)P->xm + t * 1024 + blk * 8 + g;
                        uint2 w = ldx(xp);
                        unsigned s_ = 0;
                        while (w.x == SENT || w.y == SENT) {
                            if ((++s_ & (LDH_AB - 1)) == 0) {
                                if (((volatile unsigned*)P->ctr)[CABORT]) break;
                                if (s_ > (1u << 22)) { ((volatile unsigned*)P->ctr)[CABORT] = 1u; break; }
                            }
                            w = ldx(xp);
                        }
                        xo[lane] = make_float4(b2f((u16)w.x), b2f((u16)(w.x >> 16)), b2f((u16)w.y), b2f((u16)(w.y >> 16)));
                    }
                  } else {
                    float4 c = red[lane];
                    for (int w = 1; w < 16; w++) c = f4add(c, red[w * 32 + lane]);
                    x_update(P, blk, lane, B, make_float4(rbA(c.x), rbA(c.y), rbA(c.z), rbA(c.w)), xo, true);
                  }
                    if (!EB && tid == 0) atomicAdd(P->ctr + CM, 1u);
                    // Sentinel into this block's rows of the layer-output stream (every block has read the layer input:
                    // the attention started after all of them had signalled A); read back in phase D.
                    if (L.moe) x_sentinel(P, blk, lane, B, false);
                }
                // (no barrier here: the other warps request the next phase's tiles while warp 0 stores, fences and signals;
                //  the block meets again at the barrier inside the next wait)
                tm += NMAIN;
                PROF(5); BT(3);
                if (L.moe) {
                    tload<8>(T, L.sgu + (size_t)(blk * 128 + 8 * warp) * 32 + lane);   // preload D
                    // ============ D: post-attention norm; shared expert gate/up (router logits on aux)
                    wait_ctr(P, CM, tm);
                    PROF(6); BT(4);
                    {
                        u32 abad = 0u;
                        if (warp == 0) abad = x_chk(P, blk, lane, B, false, 0u);
                        norm_frag<big && NORM_WS>(P, nullptr, nullptr, 1, 0, nw, B, Ls, xf, sm, false, nullptr, xo, -1, nullptr, nscr, true);
                        if (warp == 0) CONFIRM(abad, x_chk(P, blk, lane, B, false, vo));
                    }
                    PROF(51);
                    {
                        float4 c4[4] = {zero4, zero4, zero4, zero4};
#pragma unroll
                        for (int i = 0; i < 8; i++) mma(c4[i & (NACC - 1)], T[i], xget(xf, Ls, 8 * warp + i, lane));
                        red[(16 + warp) * 32 + lane] = f4add(f4add(c4[0], c4[1]), f4add(c4[2], c4[3]));        // slots 16..31: the expert phase below uses 0..15
                    }
                    PROF(52);
                    __syncthreads();
                    // The shared expert's hidden units are folded and written by one warp; nobody reads them before the
                    // down phase. It was warp 0 right here, and the routing started that much later. Now: batch 1, warp
                    // 15 after the wait below, while warps 0-7 rank the router's logits; batch > 2, warp 0 after it has
                    // requested its first expert tiles (every warp is busy in that routing).
#ifndef D_EPI_EARLY
                    const bool depi_late = !big && B == 1, depi_e = big;
#else
                    const bool depi_late = false, depi_e = false;
#endif
                    if (!depi_late && !depi_e && warp == 0) {
                        float4 c = red[16 * 32 + lane];
                        for (int w = 1; w < 16; w++) c = f4add(c, red[(16 + w) * 32 + lane]);
                        h_store(P, c, blk, lane, B, HSLOT_SHARED * 32);
                    }
                    tr += NROUT;
                    PROF(7);
                    // ============ E (same main phase: no hand-over between D and E): routing in every main block as soon as
                    // the router blocks are done, then expert gate/up: 64 row groups per unit, 3 teams of 5 warps
                    wait_ctr(P, CR, tr);
                    PROF(8); BT(5);
                    if (depi_late && warp == 15) {
                        float4 c = red[16 * 32 + lane];
                        for (int w = 1; w < 16; w++) c = f4add(c, red[(16 + w) * 32 + lane]);
                        h_store(P, c, blk, lane, B, HSLOT_SHARED * 32);
                    }
                    if (!(WHATIF & 16)) route_local(P, P->rl, B, sm, smi, un, uwl, blk, PPI, big, ER != 0);
                    if (WHATIF & 1) {
                        __syncthreads();
                        if (tid < 36) {
                            const int rr = tid / 6, kk = tid % 6;
                            un[1 + tid] = rr * 40 + kk * 3;
                            for (int b = 0; b < MAXB; b++) {
                                uwl[tid * MAXB + b] = (b < B && (b % 6) == kk) ? (1.0f / 6) : 0.f;
                                ((unsigned char*)(sm + 512))[tid * MAXB + b] = (unsigned char)((b < B && (b % 6) == kk) ? rr + 1 : 0);
                            }
                        }
                        if (tid < 48) ((unsigned char*)(sm + 608))[(tid / 6) * 8 + tid % 6] = (unsigned char)((tid % 6) * 6 + (tid / 6) % 6);
                        if (tid == 0) un[0] = 36;
                        __syncthreads();
                    }
#ifdef ROUTE_DUMP
                    // (study build: the six experts of every sequence go to rows 2040..2045 of the debug copy of the layer input)
                    if (P->dbg && blk == 1 && tid < 64 && (tid >> 3) < B && (tid & 7) < 6) P->dbg[(size_t)l * MAXB * HID + (size_t)(tid >> 3) * HID + 2040 + (tid & 7)] = (float)(smi[400 + tid] & 255);
#endif
                    PROF(9); BT(6);
                    const int U = un[0] < 6 * B ? (un[0] > 0 ? un[0] : 1) : 6 * B;
                    {
                        // A row group (16 rows x 128 tiles) = 8 pieces of 16 tiles; a slot = 2 row groups = one 16-tile batch
                        // on each of the 16 warps (16 loads in flight per warp: the memory's full rate). The next slot's
                        // tiles are requested before this slot's partial sums are folded.
                        const int G = U * 64;
                        const int e0 = ESPLIT(blk, G), e1 = ESPLIT(blk + 1, G);
                        const int half = warp >> 3, wi = warp & 7;
                        {
                            const int rg = e0 + half;
                            if (rg < e1) tload<16>(T, L.egu + ((size_t)((un[1 + (rg >> 6)] & 255) * 64 + (rg & 63)) * 128 + wi * 16) * 32 + lane);
                        }
                        // the hidden-unit words this warp will write: sentinel now (the previous layer's down phase is
                        // done everywhere), read back and signalled while the first tiles are on their way
                        u32 hdep2 = 0u;
                        {
                            // E_CTR (batch > 2): no sentinels for the expert phase's hidden units. Their readers (the down
                            // phase's copy, in every block) wait for a counter that every warp bumps after its own
                            // writes instead: the atomics of one warp reach the L2 in order, so a reader that sees the
                            // counter sees the data. The sentinel exchanges stood between the routing and the first
                            // slot's tiles (timing study WHATIF bit 8: 81 us per step).
                            if ((WHATIF & 256) || (big && (E_CTR || HS_EARLY))) {}
                            else if (big && HS_PLAIN) { for (int rg = e0 + warp; rg < e1; rg += 16) h_sent_p(P, rg & 63, lane, B, (rg >> 6) * 32, uwl + (rg >> 6) * MAXB); }
                            else if (big) { for (int rg = e0 + warp; rg < e1; rg += 16) hdep2 |= h_sent(P, rg & 63, lane, B, (rg >> 6) * 32, uwl + (rg >> 6) * MAXB); }
                            else if (wi == 0) { for (int rg = e0 + half; rg < e1; rg += 2) hdep2 |= h_sent(P, rg & 63, lane, B, (rg >> 6) * 32, uwl + (rg >> 6) * MAXB); }
                        }
                        if (depi_e && warp == 0) {      // (phase D's epilogue, see there; its partial sums are in slots 16..31, used again from the second slot on)
                            float4 c = red[16 * 32 + lane];
                            for (int w = 1; w < 16; w++) c = f4add(c, red[(16 + w) * 32 + lane]);
                            h_store(P, c, blk, lane, B, HSLOT_SHARED * 32);
                        }
                        if (!(big && (E_SETTLE_LATE || E_CTR)) && (big || wi == 0)) { SETTLE(hdep2); if (EB && lane == 0) atomicAdd(P->ctr + CM, 1u); }
                        PROF(53);
                        for (int sl = 0; e0 + 2 * sl < e1; sl++) {
                            const int rg = e0 + 2 * sl + half;
                            const bool act = rg < e1;
                            const int u = rg >> 6, lr = rg & 63;
                            float4* redE = red + (sl & 1) * 16 * 32;
                            float4 c = zero4;
                            if (act) {
                                float4 c4[4] = {zero4, zero4, zero4, zero4};
#pragma unroll
                                for (int i = 0; i < 16; i++) mma(c4[i & (NACC - 1)], T[i], xget(xf, Ls, wi * 16 + i, lane));
                                c = f4add(f4add(c4[0], c4[1]), f4add(c4[2], c4[3]));
                            }
                            redE[warp * 32 + lane] = c;
                            PROF(54);
                            {
                                const int rn = rg + 2;
                                if (rn < e1) tload<16>(T, L.egu + ((size_t)((un[1 + (rn >> 6)] & 255) * 64 + (rn & 63)) * 128 + wi * 16) * 32 + lane);
                            }
                            PROF(55);
                            // (E_SETTLE_LATE: the wait for the sentinel exchanges and the signal come behind the first
                            //  slot's mma and the request for the second slot's tiles, not in front of them)
                            if (big && E_SETTLE_LATE && sl == 0) { SETTLE(hdep2); if (EB && lane == 0) atomicAdd(P->ctr + CM, 1u); }
                            __syncthreads();
                            PROF(56);
                            // Which half's sums does this warp fold and write? Normally the first warp of each half its own.
                            // Batch 1-2, when the next slot has work for the first half only (3 row groups per block at
                            // batch 1): warps 8 and 9, idle in that slot, take the two halves, so that no warp of the
                            // first half does an epilogue before its next 16 tiles.
                            int hE = -1;
#ifndef E_LEADERS_FIXED
                            const bool tail = !big && e0 + 2 * (sl + 1) < e1 && e0 + 2 * (sl + 1) + 1 >= e1;
#else
                            const bool tail = false;
#endif
                            if (!tail) { if (act && wi == 0) hE = half; }
                            else if (warp == 8) hE = 1;
                            else if (warp == 9) hE = 0;
                            if (hE >= 0) {
                                const int rgE = e0 + 2 * sl + hE;
                                float4 cq = redE[(hE * 8) * 32 + lane];
                                for (int w = 1; w < 8; w++) cq = f4add(cq, redE[(hE * 8 + w) * 32 + lane]);
                                // Larger batches: the folded sums wait in shared memory and are written out after the last
                                // slot by all 16 warps (the atomic writes of two leader warps stalled every slot by about 1.4 us).
                                if (big) ((float4*)(sb + (104 << 10)))[(rgE - e0) * 32 + lane] = cq;
                                else h_store(P, cq, rgE & 63, lane, B, (rgE >> 6) * 32, uwl + (rgE >> 6) * MAXB);
                            }
                            PROF(57);
                        }
                    }
                    const int wj = warp & 1;
                    if (big) {
                        __syncthreads();         // every folded sum is in shared memory
                        // FPRE_EARLY: the down phase's first tiles are requested before the hidden units are written:
                        // a warp's loads return behind the exchanges it issued before them (about 2 us)
                        if (FPRE_EARLY) {
                            const int it = warp >> 1;
                            if (it < U) tload<16>(T, L.edown + ((size_t)((un[1 + it] & 255) * 128 + blk) * 32 + 16 * wj) * 32 + lane);
                            else if (it < U + 2) tload<16>(T, L.sdown + ((size_t)(blk * 64 + (it - U) * 32) + 16 * wj) * 32 + lane);
                        }
                        const int G = U * 64, e0 = ESPLIT(blk, G), e1 = ESPLIT(blk + 1, G);
                        for (int rg = e0 + warp; rg < e1; rg += 16) {
                            const float4 cq = ((const float4*)(sb + (104 << 10)))[(rg - e0) * 32 + lane];
                            h_store(P, cq, rg & 63, lane, B, (rg >> 6) * 32, uwl + (rg >> 6) * MAXB);
                        }
                        if ((!EB || E_CTR) && lane == 0) atomicAdd(P->ctr + CM, 1u);      // every warp, after its own writes
                        tm += KEBIG * NMAIN;
                    } else {
                        if ((warp & 7) == 0) {       // the two leaders (warp 0 also for its shared-expert stores of phase D):
                            if (!EB && lane == 0) atomicAdd(P->ctr + CM, 1u);
                        }
                        tm += KE * NMAIN;
                    }
                    PROF(10); BT(7);
                    // preload F: the first 8 items (item = warp/2: units in ascending order, then the two halves of the shared expert)
                    if (!(big && FPRE_EARLY)) {
                        const int it = warp >> 1;
                        if (it < U) tload<16>(T, L.edown + ((size_t)((un[1 + it] & 255) * 128 + blk) * 32 + 16 * wj) * 32 + lane);
                        else if (it < U + 2) tload<16>(T, L.sdown + ((size_t)(blk * 64 + (it - U) * 32) + 16 * wj) * 32 + lane);
                    }
                    PROF(48);
                    // ============ F: down projections (items of 32 chunks, 2 warps each), routing weights, shared expert, residual
                    wait_ctr(P, CM, tm);
                    PROF(11); BT(8);
                    {
                        const int nitems = U + 2;
                        float fin[4] = {0.f, 0.f, 0.f, 0.f}, sh[4] = {0.f, 0.f, 0.f, 0.f};
                        for (int it0 = 0; it0 < nitems; it0 += 8) {
                            const int it = it0 + (warp >> 1);
                            // (first round) every reader of the post-attention stream is done: sentinel into this block's
                            // rows, read back after this round's copy
                            if (it0 == 0 && EB && tid == 0) atomicAdd(P->ctr + CM, 1u);      // "the sentinels of this block's rows are in place" (phases C/D)
                            if (!big) {
                                frag_copy(xf, Ls, 256, B, [&](int b, int chunk) {
                                    const int itc = it0 + (chunk >> 5);
                                    const int hs = itc < U ? itc : HSLOT_SHARED + (itc - U < 2 ? itc - U : 0);
                                    return P->hbuf + ((size_t)b * HCHUNKS + hs * 32 + (chunk & 31)) * 4;
                                }, P);
                            } else if (F_ONE) {
                                // F_ONE: every fragment the phase needs is copied once, before the first round, into a
                                // compact layout: slot b * 8 + r holds sequence b's hidden units of its r-th expert
                                // (r = 6, 7: the two halves of the shared expert), 32 chunks x 4 lanes. 64 slots = the
                                // 64 KB of the fragment buffer. The rounds then only stream tiles. (Copying per round put
                                // strong loads between the tile requests of every round: 8 tiles, the copy, 8 tiles;
                                // timing study WHATIF bit 1: 210 us per step.)
                                // Thread: sequence tid / 64, chunk (tid / 2) % 32, half tid % 2; eight loads, four in flight.
                                if (it0 == 0 && !(WHATIF & 64)) {
                                    const int cb = tid >> 6, cc = (tid >> 1) & 31, cq = tid & 1;
                                    if (cb < B) {
                                        const unsigned char* bu = (const unsigned char*)(sm + 608) + cb * 8;
                                        for (int h4 = 0; h4 < 8; h4 += 4) {
                                            uint4 r[4];
#if FO_V == 0
                                            const uint4* ap[4];
#pragma unroll
                                            for (int k = 0; k < 4; k++) {
                                                const int rr = h4 + k, hs = rr < 6 ? (int)bu[rr] : HSLOT_SHARED + rr - 6;
                                                ap[k] = (const uint4*)(P->hbuf + ((size_t)cb * HCHUNKS + hs * 32 + cc) * 4 + 2 * cq);
                                                r[k] = ldx(ap[k]);
                                            }
#pragma unroll
                                            for (int k = 0; k < 4; k++) {
                                                if (is_sent(r[k])) {
                                                    r[k] = ldh(ap[k], P);
#pragma unroll
                                                    for (int j = k + 1; j < 4; j++) if (is_sent(r[j])) r[j] = ldx(ap[j]);
                                                }
                                            }
#else
#define FO_AP(k) ((const uint4*)(P->hbuf + ((size_t)cb * HCHUNKS + ((h4 + (k)) < 6 ? (int)bu[h4 + (k)] : HSLOT_SHARED + h4 + (k) - 6) * 32 + cc) * 4 + 2 * cq))
#pragma unroll
                                            for (int k = 0; k < 4; k++) r[k] = ldx(FO_AP(k));
#if FO_V == 3
                                            for (unsigned spins = 0;;) {       // (as FCB_W: all pending words of the batch per attempt)
                                                if (!(is_sent(r[0]) || is_sent(r[1]) || is_sent(r[2]) || is_sent(r[3]))) break;
                                                if ((++spins & (LDH_AB - 1)) == 0) {
                                                    if (((volatile unsigned*)P->ctr)[CABORT]) break;
                                                    if (spins > (1u << 22)) { ((volatile unsigned*)P->ctr)[CABORT] = 1u; break; }
                                                }
#pragma unroll
                                                for (int k = 0; k < 4; k++) if (is_sent(r[k])) r[k] = ldx(FO_AP(k));
                                            }
#endif
#pragma unroll
                                            for (int k = 0; k < 4 && FO_V != 3; k++) {
                                                if (is_sent(r[k])) {
                                                    r[k] = ldh(FO_AP(k), P);
#if FO_V == 1
#pragma unroll
                                                    for (int j = k + 1; j < 4; j++) if (is_sent(r[j])) r[j] = ldx(FO_AP(j));
#endif
                                                }
                                            }
#endif
#pragma unroll
                                            for (int k = 0; k < 4; k++) *(uint4*)(xf + ((cb * 8 + h4 + k) * 32 + cc) * 4 + 2 * cq) = r[k];
                                        }
                                    }
                                }
                            } else if ((WHATIF & 2) && it0 > 0) {
                                if (it < nitems) {
                                    const uint4* dt = it < U ? L.edown + ((size_t)((un[1 + it] & 255) * 128 + blk) * 32 + 16 * wj) * 32 + lane
                                                             : L.sdown + ((size_t)(blk * 64 + (it - U) * 32) + 16 * wj) * 32 + lane;
                                    tload<8>(T + 8, dt + 8 * 32);
                                }
                            } else if (!F_PIPE || it0 == 0) {
                                // larger batches: a unit has few member sequences; copy only their fragments (the others'
                                // columns keep old values and are multiplied by a zero weight)
                                const int chunk = tid >> 1, q = tid & 1, itc = it0 + (chunk >> 5);
                                if (itc < U) {
                                    for (int b = 0; b < B; b++)
                                        if (uwl[itc * MAXB + b] != 0.f)
                                            { uint4 r = ldx((const uint4*)(P->hbuf + ((size_t)b * HCHUNKS + itc * 32 + (chunk & 31)) * 4 + 2 * q)); if (is_sent(r)) r = ldh((const uint4*)(P->hbuf + ((size_t)b * HCHUNKS + itc * 32 + (chunk & 31)) * 4 + 2 * q), P); *(uint4*)(xf + chunk * Ls + b * 4 + 2 * q) = r; }
                                }
                                if (U >= it0 && U < it0 + 8) {
                                    // the shared expert's two items are in this round: all sequences, spread over all threads
                                    const int idx = tid & 127, ch64 = idx >> 1, qq = idx & 1, slot0 = U - it0;      // 64 chunks, slots slot0, slot0+1
                                    if (slot0 * 32 + ch64 < 256)
                                        for (int b = tid >> 7; b < B; b += 4)
                                            { uint4 r = ldx((const uint4*)(P->hbuf + ((size_t)b * HCHUNKS + HSLOT_SHARED * 32 + ch64) * 4 + 2 * qq)); if (is_sent(r)) r = ldh((const uint4*)(P->hbuf + ((size_t)b * HCHUNKS + HSLOT_SHARED * 32 + ch64) * 4 + 2 * qq), P); *(uint4*)(xf + (slot0 * 32 + ch64) * Ls + b * 4 + 2 * qq) = r; }
                                } else if (U + 1 >= it0 && U + 1 < it0 + 8) {
                                    // only the second half of the shared expert falls into this round (slot 0)
                                    const int idx = tid & 63, ch32 = idx >> 1, qq = idx & 1;
                                    for (int b = tid >> 6; b < B; b += 8)
                                        { uint4 r = ldx((const uint4*)(P->hbuf + ((size_t)b * HCHUNKS + (HSLOT_SHARED + 1) * 32 + ch32) * 4 + 2 * qq)); if (is_sent(r)) r = ldh((const uint4*)(P->hbuf + ((size_t)b * HCHUNKS + (HSLOT_SHARED + 1) * 32 + ch32) * 4 + 2 * qq), P); *(uint4*)(xf + ch32 * Ls + b * 4 + 2 * qq) = r; }
                                }
                                // Only now the second half of this round's tiles (round 0: all 16 were loaded ahead). A warp's
                                // loads return in the order issued: with all 16 tile requests ahead of it, the copy above had
                                // to wait for all of them; now it waits for 8, and the other 8 arrive during the first 8 mma.
                                if (!F_PIPE && it0 > 0 && it < nitems) {
                                    const uint4* dt = it < U ? L.edown + ((size_t)((un[1 + it] & 255) * 128 + blk) * 32 + 16 * wj) * 32 + lane
                                                             : L.sdown + ((size_t)(blk * 64 + (it - U) * 32) + 16 * wj) * 32 + lane;
                                    tload<8>(T + 8, dt + 8 * 32);
                                }
                            }
                            PROF(41);
                            if (!(big && (F_PIPE || F_ONE)) || it0 == 0) __syncthreads();
                            PROF(42);
                            float4 c = zero4;
                            if (it < nitems) {
                                float4 c4[4] = {zero4, zero4, zero4, zero4};
                                if (big && F_ONE) {
                                    // this lane's column is sequence lane / 4: its slot if it is a member of the item (the
                                    // shared expert: every sequence), else the zero word with stride 0
                                    const int fb = lane >> 2;
                                    int r1 = 0;
                                    if (fb < B) r1 = it < U ? (int)((const unsigned char*)(sm + 512))[it * MAXB + fb] : 7 + (it - U);
                                    const uint2* bp = r1 ? xf + (((fb * 8 + r1 - 1) * 32 + 16 * wj) * 4 + (lane & 3)) : xf - 1;
                                    const int bst = r1 ? 4 : 0;
#pragma unroll
                                    for (int i = 0; i < 16; i++) mma(c4[i & (NACC - 1)], T[i], bp[i * bst]);
                                } else {
#pragma unroll
                                for (int i = 0; i < 16; i++) mma(c4[i & (NACC - 1)], T[i], xget(xf, Ls, (warp >> 1) * 32 + 16 * wj + i, lane));
                                }
                                c = f4add(f4add(c4[0], c4[1]), f4add(c4[2], c4[3]));
                            }
                            red[warp * 32 + lane] = c;
                            // F_PIPE (batch > 2): one trip to memory per round. The next round's fragments are requested
                            // first and its 16 tiles right behind them, as soon as this round's mma are done; the fragments
                            // (few bytes, first in the warp's queue) are back by the barrier and go to shared memory while
                            // the tiles are still on their way. Before, 8 tiles were requested, then (after two barriers)
                            // the fragments behind them, then the other 8 tiles: two trips per round, and the down phase
                            // streamed at half the rate of the expert phase (3.6 TB/s).
                            // A thread holds at most four fragment words: two member sequences of its unit's chunk and
                            // two of the shared expert's; a unit with more members finishes after the barrier (rare).
                            uint4 cv0 = make_uint4(0u, 0u, 0u, 0u), cv1 = cv0, cv2 = cv0, cv3 = cv0;
                            int m0 = -1, m1 = -1;
                            const bool fpn = big && F_PIPE && it0 + 8 < nitems;
                            const int it0n = it0 + 8, fchunk = tid >> 1, fq = tid & 1, fitc = it0n + (fchunk >> 5);
                            const int fsl = U - it0n;       // slot of the shared expert's first half in the next round (0..7: in it; -1: only its second half is)
                            if (fpn) {
                                if (fitc < U) {
                                    for (int b = 0; b < B; b++) if (uwl[fitc * MAXB + b] != 0.f) { if (m0 < 0) m0 = b; else if (m1 < 0) m1 = b; }
                                    if (m0 >= 0) cv0 = ldx((const uint4*)(P->hbuf + ((size_t)m0 * HCHUNKS + fitc * 32 + (fchunk & 31)) * 4 + 2 * fq));
                                    if (m1 >= 0) cv1 = ldx((const uint4*)(P->hbuf + ((size_t)m1 * HCHUNKS + fitc * 32 + (fchunk & 31)) * 4 + 2 * fq));
                                }
                                if (fsl >= 0 && fsl < 8) {
                                    const int idx = tid & 127, ch64 = idx >> 1, qq = idx & 1, b0 = tid >> 7;
                                    if (fsl * 32 + ch64 < 256) {
                                        if (b0 < B) cv2 = ldx((const uint4*)(P->hbuf + ((size_t)b0 * HCHUNKS + HSLOT_SHARED * 32 + ch64) * 4 + 2 * qq));
                                        if (b0 + 4 < B) cv3 = ldx((const uint4*)(P->hbuf + ((size_t)(b0 + 4) * HCHUNKS + HSLOT_SHARED * 32 + ch64) * 4 + 2 * qq));
                                    }
                                } else if (fsl == -1) {
                                    const int idx = tid & 63, ch32 = idx >> 1, qq = idx & 1, b0 = tid >> 6;
                                    if (b0 < B) cv2 = ldx((const uint4*)(P->hbuf + ((size_t)b0 * HCHUNKS + (HSLOT_SHARED + 1) * 32 + ch32) * 4 + 2 * qq));
                                }
                            }
                            {   // the next round's tiles go into the registers now: they arrive while the leaders work
                                const int itn = it + 8;
                                if (itn < nitems) {
                                    const uint4* dt = itn < U ? L.edown + ((size_t)((un[1 + itn] & 255) * 128 + blk) * 32 + 16 * wj) * 32 + lane
                                                              : L.sdown + ((size_t)(blk * 64 + (itn - U) * 32) + 16 * wj) * 32 + lane;
                                    if (big && !F_PIPE && !F_ONE) tload<8>(T, dt); else tload<16>(T, dt);      // (the old per-round copy, larger batches: the first half only, see above)
                                }
                            }
                            PROF(43);
                            __syncthreads();
                            PROF(44);
                            if (fpn) {      // every warp is through this round's mma: the fragment buffer is free
                                if (fitc < U) {
                                    if (m0 >= 0) {
                                        if (is_sent(cv0)) cv0 = ldh((const uint4*)(P->hbuf + ((size_t)m0 * HCHUNKS + fitc * 32 + (fchunk & 31)) * 4 + 2 * fq), P);
                                        *(uint4*)(xf + fchunk * Ls + m0 * 4 + 2 * fq) = cv0;
                                    }
                                    if (m1 >= 0) {
                                        if (is_sent(cv1)) cv1 = ldh((const uint4*)(P->hbuf + ((size_t)m1 * HCHUNKS + fitc * 32 + (fchunk & 31)) * 4 + 2 * fq), P);
                                        *(uint4*)(xf + fchunk * Ls + m1 * 4 + 2 * fq) = cv1;
                                        for (int b = m1 + 1; b < B; b++)
                                            if (uwl[fitc * MAXB + b] != 0.f)
                                                { uint4 r = ldx((const uint4*)(P->hbuf + ((size_t)b * HCHUNKS + fitc * 32 + (fchunk & 31)) * 4 + 2 * fq)); if (is_sent(r)) r = ldh((const uint4*)(P->hbuf + ((size_t)b * HCHUNKS + fitc * 32 + (fchunk & 31)) * 4 + 2 * fq), P); *(uint4*)(xf + fchunk * Ls + b * 4 + 2 * fq) = r; }
                                    }
                                }
                                if (fsl >= 0 && fsl < 8) {
                                    const int idx = tid & 127, ch64 = idx >> 1, qq = idx & 1, b0 = tid >> 7;
                                    if (fsl * 32 + ch64 < 256) {
                                        if (b0 < B) {
                                            if (is_sent(cv2)) cv2 = ldh((const uint4*)(P->hbuf + ((size_t)b0 * HCHUNKS + HSLOT_SHARED * 32 + ch64) * 4 + 2 * qq), P);
                                            *(uint4*)(xf + (fsl * 32 + ch64) * Ls + b0 * 4 + 2 * qq) = cv2;
                                        }
                                        if (b0 + 4 < B) {
                                            if (is_sent(cv3)) cv3 = ldh((const uint4*)(P->hbuf + ((size_t)(b0 + 4) * HCHUNKS + HSLOT_SHARED * 32 + ch64) * 4 + 2 * qq), P);
                                            *(uint4*)(xf + (fsl * 32 + ch64) * Ls + (b0 + 4) * 4 + 2 * qq) = cv3;
                                        }
                                    }
                                } else if (fsl == -1) {
                                    const int idx = tid & 63, ch32 = idx >> 1, qq = idx & 1, b0 = tid >> 6;
                                    if (b0 < B) {
                                        if (is_sent(cv2)) cv2 = ldh((const uint4*)(P->hbuf + ((size_t)b0 * HCHUNKS + (HSLOT_SHARED + 1) * 32 + ch32) * 4 + 2 * qq), P);
                                        *(uint4*)(xf + ch32 * Ls + b0 * 4 + 2 * qq) = cv2;
                                    }
                                }
                            }
                            // team leaders: the item's sum; an expert's output is rounded to BF16 and scaled by its routing weight
                            if (wj == 0 && it < nitems) {
                                float4 cq = f4add(red[warp * 32 + lane], red[(warp + 1) * 32 + lane]);
                                if (it < U) {
                                    const float wa = 2 * t < B ? uwl[it * MAXB + 2 * t] : 0.f, wb = 2 * t + 1 < B ? uwl[it * MAXB + 2 * t + 1] : 0.f;
                                    // (a column whose sequence is not a member of the unit holds whatever was in the fragment buffer,
                                    //  possibly bit patterns that are not numbers: select zero, do not multiply by zero)
                                    cq = make_float4(wa != 0.f ? rbA(cq.x) * wa : 0.f, wb != 0.f ? rbA(cq.y) * wb : 0.f, wa != 0.f ? rbA(cq.z) * wa : 0.f, wb != 0.f ? rbA(cq.w) * wb : 0.f);
                                }
                                red[(16 + (warp >> 1)) * 32 + lane] = cq;
                            }
                            __syncthreads();
                            if (warp == 0) {
                                for (int q = 0; q < 8 && it0 + q < nitems; q++) {
                                    const float4 cq = red[(16 + q) * 32 + lane];
                                    if (it0 + q < U) { fin[0] += cq.x; fin[1] += cq.y; fin[2] += cq.z; fin[3] += cq.w; }
                                    else { sh[0] += cq.x; sh[1] += cq.y; sh[2] += cq.z; sh[3] += cq.w; }
                                }
                            }
                            // (F_PIPE: no barrier here. Warp 0 has read the items' sums before it reaches the next barrier, behind
                            //  its own mma; the fragments of the next round were stored before the last barrier.)
                            if (!(big && (F_PIPE || F_ONE)) && it0 + 8 < nitems) __syncthreads();
                        }
                        PROF(45);
                        if (warp == 0) {
                            x_update(P, blk, lane, B, make_float4(rbA(rbA(fin[0]) + rbA(sh[0])), rbA(rbA(fin[1]) + rbA(sh[1])), rbA(rbA(fin[2]) + rbA(sh[2])), rbA(rbA(fin[3]) + rbA(sh[3]))), xo, false);
                            PROF(46);
                            PROF(47);
                            if (!EB && tid == 0) atomicAdd(P->ctr + CM, 1u);
                        }
                        tm += NMAIN;
                    }
                    PROF(12); BT(9); BT_FLUSH();
                } else {
                    // ============ dense MLP (layers 0 and 1). D': gate/up, 768 consecutive tiles (6 row groups) per block, 48 per warp
                    wait_ctr(P, CM, tm);
                    norm_frag<big && NORM_WS>(P, nullptr, nullptr, 1, 0, nw, B, Ls, xf, sm, false, nullptr, xo, -1, nullptr, nscr, true);
                    {
                        const int g0 = 48 * warp;
                        int nA = 128 - (g0 & 127); if (nA > 48) nA = 48;
                        float4 cA = zero4, cB = zero4;
                        const uint4* wt = L.dgu + (size_t)(blk * 768 + g0) * 32 + lane;
                        for (int i = 0; i < 48; i++) {
                            const uint4 a = wt[(size_t)i * 32];
                            const uint2 bb = xget(xf, Ls, (g0 + i) & 127, lane);
                            if (i < nA) mma(cA, a, bb); else mma(cB, a, bb);
                        }
                        red[(warp * 2) * 32 + lane] = cA;
                        red[(warp * 2 + 1) * 32 + lane] = cB;
                    }
                    __syncthreads();
                    if (warp == 0) {
                        for (int h = 0; h < 6; h++) {
                            const int w_lo = (128 * h) / 48, w_hi = (128 * h + 127) / 48;
                            float4 c = zero4;
                            for (int w = w_lo; w <= w_hi; w++) c = f4add(c, red[(w * 2 + (((48 * w) >> 7) == h ? 0 : 1)) * 32 + lane]);
                            { const u32 dep = h_store(P, c, blk * 6 + h, lane, B, 0); SETTLE(dep); }
                        }
                        FENCE(64);
                        if (tid == 0) atomicAdd(P->ctr + CM, 1u);
                    }
                    tm += NMAIN;
                    // E': down projection (K = 6144: 384 tiles per block, 24 per warp) + residual
                    wait_ctr(P, CM, tm);
                    {
                        float4 c = zero4;
                        const uint4* wt = L.ddown + (size_t)(blk * 384 + 24 * warp) * 32 + lane;
                        const uint2* hb = P->hbuf + ((size_t)(lane >> 2) * HCHUNKS + 24 * warp) * 4 + (lane & 3);
                        for (int i = 0; i < 24; i++) {
                            const uint4 a = wt[(size_t)i * 32];
                            uint2 bb = make_uint2(0u, 0u);
                            if (lane < 4 * B) bb = ldx(hb + i * 4);
                            mma(c, a, bb);
                        }
                        red[warp * 32 + lane] = c;
                    }
                    __syncthreads();
                    if (warp == 0) {
                        float4 c = red[lane];
                        for (int w = 1; w < 16; w++) c = f4add(c, red[w * 32 + lane]);
                        { const u32 dep = x_update(P, blk, lane, B, make_float4(rbA(c.x), rbA(c.y), rbA(c.z), rbA(c.w)), xo, false); SETTLE(dep); }
                        FENCE(64);
                        if (tid == 0) atomicAdd(P->ctr + CM, 1u);
                    }
                    __syncthreads();
                    tm += NMAIN;
                }
#if APRE_EARLY
                // (APRE_EARLY: the tiles first. A warp's loads return behind the exchanges it issued before them.)
                // preload A of the next layer (or the final norm weights), then wait for every main block
                if (l + 1 < NLAYER) {
                    if (afirst) { if (AF_T != 1) attn_stage(P, P->L[l + 1], (blk + NAUX) >> 2, (blk + NAUX) & 3, pos, AS, T, true); if (AF_T == 2) tload<NPA>(T, P->L[l + 1].qkv + (size_t)(blk * 320 + 20 * warp) * 32 + lane); }
                    else {
                        tload<NPA>(T, P->L[l + 1].qkv + (size_t)(blk * 320 + 20 * warp) * 32 + lane);
                        nw_load(nw, P->L[l + 1].ln1);
                    }
                } else nw_load(nw, P->fnorm);
                // Sentinels for the next layer, confirmed after the wait below: q/k/v (this layer's attention has read them:
                // this block has its output) and, for a main block with an attention pair, the pair's output words
                // (every block has copied this layer's attention output: they all signalled C).
                if (l + 1 < NLAYER) {
                    if (!afirst && warp < 5) q_sentinel(P, blk, B);
                    if (!AF_TEST && warp == 5) { const int hostE = HOST_E(); if (hostE >= 0) q_sent_hg(P, hostE, B); }
                    if (mattn && warp < 8) attn_sentinel(P, (blk + NAUX) >> 2, (blk + NAUX) & 3);
                }
#else
                // Sentinels for the next layer, confirmed after the wait below: q/k/v (this layer's attention has read them:
                // this block has its output) and, for a main block with an attention pair, the pair's output words
                // (every block has copied this layer's attention output: they all signalled C).
                if (l + 1 < NLAYER) {
                    if (!afirst && warp < 5) q_sentinel(P, blk, B);
                    if (!AF_TEST && warp == 5) { const int hostE = HOST_E(); if (hostE >= 0) q_sent_hg(P, hostE, B); }
                    if (mattn && warp < 8) attn_sentinel(P, (blk + NAUX) >> 2, (blk + NAUX) & 3);
                }
                // preload A of the next layer (or the final norm weights), then wait for every main block
                if (l + 1 < NLAYER) {
                    if (afirst) { if (AF_T != 1) attn_stage(P, P->L[l + 1], (blk + NAUX) >> 2, (blk + NAUX) & 3, pos, AS, T, true); if (AF_T == 2) tload<NPA>(T, P->L[l + 1].qkv + (size_t)(blk * 320 + 20 * warp) * 32 + lane); }
                    else {
                        tload<NPA>(T, P->L[l + 1].qkv + (size_t)(blk * 320 + 20 * warp) * 32 + lane);
                        nw_load(nw, P->L[l + 1].ln1);
                    }
                } else nw_load(nw, P->fnorm);
#endif
                PROF(49);
                if (l + 1 < NLAYER || do_head) wait_ctr(P, CM, tm);
            } else {
                // ---------------- aux blocks
                const unsigned tmA = tm + KA * NMAIN, tmC = tmA + NMAIN;
#if ER
                // ---- Early routing (track B). The router ranks sigmoid(R norm2(x_mid)), x_mid = x + W_o a; the norm divides
                // by one positive number per token and the sigmoid is monotonic, so the top 6 are those of
                //     z = R' x_mid = R' x + (R' W_o) a,      R' = R diag(w2),   M = R' W_o  (built at load time).
                // Pass 1, at the layer's start: z0 = R' x (x as 3 BF16 limbs, R' as 2: both exact).
                // Pass 2, as soon as the attention is done: z = z0 + M a (a is BF16; M as NMF limbs) -> rl.
                // The main blocks do not wait for C, the norm and a router pass any more; they scale z by the norm's
                // 1/rms when they need the routing weights (route_local, zmode).
                tx += 8 * npairs;
                if (is_router && L.moe) {
                    wait_ctr(P, CM, tm);                 // the layer's input is final
                    tload<8>(T, L.rp + (size_t)(rj * 128 + 8 * warp) * 32 + lane);
                    tload<8>(T + 8, L.rp + (size_t)((16 + rj) * 128 + 8 * warp) * 32 + lane);
                    float4 cz = zero4;
                    for (int p = 0; p < 3; p++) {
                        x_limb(P, B, Ls, xf, p);
                        if (p == 2 && tid == 0) atomicAdd(P->ctr + CZ, 1u);      // every thread's last read of x has returned (barrier in x_limb)
#pragma unroll
                        for (int i = 0; i < 8; i++) mma(cz, T[i], xget(xf, Ls, 8 * warp + i, lane));
#pragma unroll
                        for (int i = 0; i < 8; i++) mma(cz, T[8 + i], xget(xf, Ls, 8 * warp + i, lane));
                        __syncthreads();
                    }
                    red[warp * 32 + lane] = cz;
                    __syncthreads();
                    if (warp == 0) {
                        float4 c = red[lane];
                        for (int w = 1; w < 16; w++) c = f4add(c, red[w * 32 + lane]);
                        ((float4*)(sm + 512))[lane] = c;      // z0 of this block's 16 experts, until pass 2
                    }
                    __syncthreads();
                }
                if (late_stage) { attn_stage(P, L, aux >> 2, aux & 3, pos, AS, T, stgsm); __syncthreads(); }
#endif
#if !ER
                if (is_router && L.moe) {
                    // The logits' words: sentinel as soon as every main block has read the previous layer's logits (they
                    // all have begun its down phase), then the signal; the main blocks wait on the logits themselves.
                    wait_ctr(P, CM, tm);
                    if (warp == 0) {
                        u32 dep = 0u;
                        for (int i = 0; i < 4; i++) {
                            const int b = 2 * t + (i & 1), e = rj * 16 + g + (i >> 1) * 8;
                            if (b < B) dep |= put32(P->rl + b * NEXP + e, SENT);
                        }
                        SETTLE(dep);
                        if (EB && tid == 0) atomicAdd(P->ctr + CR, 1u);
                    }
                }
#endif
                if (has_attn) {
                    wait_ctr(P, CM, tmA);
                    PROF(21); CLK_A(0);
                    attention(P, L, aux >> 2, aux & 3, pos, AS, T, stg, stg && stgsm, blk, PPI, l > 0 && !ER, KV_BATCH && (!big || KV_BATCH_AUXBIG));
                    PROF(22); CLK_A(9);
                    // (the K/V stores of this position are read again only by this block: no fence)
                    if (rdefer && L.moe) tload<8>(T, L.router + (size_t)(rj * 128 + 8 * warp) * 32 + lane);
                    else if (stg && !late_stage) {
                        if (l + 1 < NLAYER) attn_stage(P, P->L[l + 1], aux >> 2, aux & 3, pos, AS, T, stgsm);
                        else if (step + 1 < total && !(do_head && NHB == NBLK)) attn_stage(P, P->L[0], aux >> 2, aux & 3, pos + 1, AS, T, stgsm);
                    }
                    __syncthreads();
                    PROF(23);
#if !ER
                    if (l + 1 < NLAYER) {      // the next layer's output words: sentinel once every main block has copied this layer's
                        wait_ctr(P, CM, tmC);
                        if (warp < 8) attn_sentinel(P, aux >> 2, aux & 3);
                    }
#endif
                }
#if ER
                if (is_router && L.moe) {
                    tload<16>(T, L.mf + (size_t)(rj * 256 + 16 * warp) * 32 + lane);      // first limb of M, while the attention runs
                    wait_ctr(P, CX, tx);
                    PROF(24);
                    frag_copy(xf, Ls, 256, B, [&](int b, int chunk) { return P->attnf + ((size_t)b * 256 + chunk) * 4; }, P);
                    __syncthreads();
                    {
                        float4 c = zero4;
                        for (int q = 0; q < NMF; q++) {
                            if (q > 0) tload<16>(T, L.mf + (size_t)((q * 16 + rj) * 256 + 16 * warp) * 32 + lane);
#pragma unroll
                            for (int i = 0; i < 16; i++) mma(c, T[i], xget(xf, Ls, 16 * warp + i, lane));
                        }
                        red[warp * 32 + lane] = c;
                    }
                    __syncthreads();
                    if (warp == 0) {
                        float4 c = red[lane];
                        for (int w = 1; w < 16; w++) c = f4add(c, red[w * 32 + lane]);
                        c = f4add(c, ((const float4*)(sm + 512))[lane]);
                        u32 dep = 0u;
                        for (int i = 0; i < 4; i++) {
                            const int b = 2 * t + (i & 1), e = rj * 16 + g + (i >> 1) * 8;
                            if (b < B) dep |= put32(P->rl + b * NEXP + e, __float_as_uint(cget(c, i)));
                        }
                        SETTLE(dep);
                        FENCE(32);
                        if (tid == 0) atomicAdd(P->ctr + CR, 1u);
                    }
                    __syncthreads();
                    PROF(25);
                }
#else
                if (is_router && L.moe) {
                    wait_ctr(P, CM, tmC);
                    PROF(24);
                    uint2* xlo = TB ? (uint2*)(sb + SM_XF + (32 << 10)) : nullptr;       // 128 chunks of lo fragments (track B)
                    norm_frag<big && NORM_WS>(P, nullptr, nullptr, 1, 0, nw, B, Ls, xf, sm, false, nullptr, xo, -1, xlo, nullptr, true);
                    {
                        float4 c = zero4;
                        uint4 R8[8];
                        if (rpre || rdefer) {
#pragma unroll
                            for (int i = 0; i < 8; i++) R8[i] = T[i];
                        } else tload<8>(R8, L.router + (size_t)(rj * 128 + 8 * warp) * 32 + lane);
#pragma unroll
                        for (int i = 0; i < 8; i++) mma(c, R8[i], xget(xf, Ls, 8 * warp + i, lane));
                        if (TB) {
                            float4 cl = zero4;
#pragma unroll
                            for (int i = 0; i < 8; i++) mma(cl, R8[i], xget_c(xlo, Ls, 8 * warp + i, lane));
                            c = f4add(c, cl);
                        }
                        red[warp * 32 + lane] = c;
                    }
                    __syncthreads();
                    if (warp == 0) {
                        float4 c = red[lane];
                        for (int w = 1; w < 16; w++) c = f4add(c, red[w * 32 + lane]);
                        u32 dep = 0u;
                        for (int i = 0; i < 4; i++) {
                            const int b = 2 * t + (i & 1), e = rj * 16 + g + (i >> 1) * 8;
                            if (b < B) dep |= put32(P->rl + b * NEXP + e, __float_as_uint(cget(c, i)));
                        }
                        if (!EB && tid == 0) atomicAdd(P->ctr + CR, 1u);
                    }
                    __syncthreads();
                    PROF(25);
                    // preload the next router duty
                    int ln = l + 1;
                    if (ln >= NLAYER) ln = (step + 1 < total && !(do_head && NHB == NBLK)) ? 2 : -1;
                    if (ln >= 0) { if (rpre) tload<8>(T, P->L[ln].router + (size_t)(rj * 128 + 8 * warp) * 32 + lane); nw_load(nw, P->L[ln].ln2); }
                    if (rdefer) {      // (the staging that was put off)
                        if (l + 1 < NLAYER) attn_stage(P, P->L[l + 1], aux >> 2, aux & 3, pos, AS, T, stgsm);
                        else if (step + 1 < total && !(do_head && NHB == NBLK)) attn_stage(P, P->L[0], aux >> 2, aux & 3, pos + 1, AS, T, stgsm);
                    }
                    __syncthreads();
                }
#endif
                tm += (L.moe ? KA + 2 + (big ? KEBIG : KE) : KA + 3) * NMAIN;
            }
        }
        // ============ final norm + head (9384 row groups; one warp per row group, 128 tiles each), on every block
        if (!do_head) continue;
        if (!is_main) {
            if (NHB != NBLK) { tm += NHB; if (mode != 0) tm += NHB; continue; }
            nw_load(nw, P->fnorm);
            wait_ctr(P, CM, tm);
        }
#ifdef MK_PROF
        profl = true;
        PROF(99);
#endif
        PROF(13);
        norm_frag<big && NORM_WS>(P, nullptr, nullptr, 1, 0, nw, B, Ls, xf, sm, false, (P->dbg && blk == 0) ? P->dbg + (size_t)NLAYER * MAXB * HID : nullptr, xo, -1);
        {
            const int G = VOCAB / 16;
            const int e0 = (int)((long long)blk * G / NHB), e1 = (int)((long long)(blk + 1) * G / NHB);
            float bv[2] = {-INFINITY, -INFINITY};
            int bi[2] = {0, 0};
            for (int rg = e0 + warp; rg < e1; rg += 16) {
                const uint4* wt = P->head + (size_t)rg * 128 * 32 + lane;
                float4 c4[4] = {zero4, zero4, zero4, zero4};
                for (int ch = 0; ch < 128; ch += 16) {
                    tload<16>(T, wt + (size_t)ch * 32);        // 16 tiles per load round trip (the staging registers are free here)
#pragma unroll
                    for (int i = 0; i < 16; i++) mma(c4[i & (NACC - 1)], T[i], xget(xf, Ls, ch + i, lane));
                }
                const float4 c = f4add(f4add(c4[0], c4[1]), f4add(c4[2], c4[3]));
                for (int i = 0; i < 4; i++) {
                    const int b = 2 * t + (i & 1), row = rg * 16 + g + (i >> 1) * 8;
                    if (b < B) {
                        const float val = cget(c, i);
                        if (mode == 0) logits[(size_t)b * VOCAB + row] = val;
                        else if (val > bv[i & 1]) { bv[i & 1] = val; bi[i & 1] = row; }
                    }
                }
            }
            if (mode != 0) {
                for (int i = 0; i < 2; i++) {
                    for (int o = 4; o <= 16; o <<= 1) {
                        const float ov = __shfl_xor_sync(0xffffffffu, bv[i], o);
                        const int oi = __shfl_xor_sync(0xffffffffu, bi[i], o);
                        if (ov > bv[i] || (ov == bv[i] && oi < bi[i])) { bv[i] = ov; bi[i] = oi; }
                    }
                    if (g == 0) { sm[warp * 8 + 2 * t + i] = bv[i]; smi[warp * 8 + 2 * t + i] = bi[i]; }
                }
                __syncthreads();
                if (tid < B) {
                    float v = sm[tid]; int ix = smi[tid];
                    for (int w = 1; w < 16; w++) {
                        const float ov = sm[w * 8 + tid]; const int oi = smi[w * 8 + tid];
                        if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
                    }
                    hdep = put32(P->argv + blk * MAXB + tid, __float_as_uint(v)); hdep |= put32(P->argi + blk * MAXB + tid, (u32)ix);
                }
            }
        }
        __syncthreads();
        if (warp == 0) {
            SETTLE(hdep); hdep = 0u;
            FENCE(128);
            if (tid == 0) atomicAdd(P->ctr + CM, 1u);
        }
        tm += NHB;
        PROF(14);
        if (mode != 0) {
            wait_ctr(P, CM, tm);
            // Block 0 gathers the blocks' best tokens. Batch 1-2 kernel: every thread takes up to 3 of the 152 x 8
            // (value, index) pairs of sequence tid % 8, all loads in one trip; the lanes of a warp with the same sequence,
            // then the warps, are combined (largest value, lowest index on a tie: any order gives the same token). It was
            // one thread per sequence loading 302 words one after the other: 38-50 us per token. Batch > 2 kernel: the old
            // loop (there the call costs more than it saves: the register picture of that instance).
            const bool gpar = GATHER_PAR && !big, gstr = GATHER_BIG && big;
            if (gpar && blk == 0) gather_best(P, B, sm, smi);
            if (gstr && blk == 0) {
                // batch > 2: the old loop cut into 16 strides (thread = sequence tid % 8, stride tid / 8: 10 trips, not 151)
                if (tid < 128 && (tid & 7) < B) {
                    const int b = tid & 7, p = tid >> 3;
                    float v = ldx(P->argv + p * MAXB + b); int ix = ldx(P->argi + p * MAXB + b);
                    for (int w = p + 16; w < NHB; w += 16) {
                        const float ov = ldx(P->argv + w * MAXB + b); const int oi = ldx(P->argi + w * MAXB + b);
                        if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
                    }
                    sm[p * 8 + b] = v; smi[p * 8 + b] = ix;
                }
                __syncthreads();
            }
            if (blk == 0 && tid < B) {
                float v; int ix;
                if (gpar || gstr) {
                    v = sm[tid]; ix = smi[tid];
                    for (int w = 1; w < 16; w++) {
                        const float ov = sm[w * 8 + tid]; const int oi = smi[w * 8 + tid];
                        if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
                    }
                } else {
                    v = ldx(P->argv + tid); ix = ldx(P->argi + tid);
                    for (int w = 1; w < NHB; w++) {
                        const float ov = ldx(P->argv + w * MAXB + tid); const int oi = ldx(P->argi + w * MAXB + tid);
                        if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
                    }
                }
                P->curtok[tid] = ix;
                if (mode == 1) gen_out[(size_t)tid * nsteps + (step - npre)] = ix;
                else P->gen[(size_t)tid * P->maxctx + (step - npre)] = ix;
            }
            __syncthreads();
            if (warp == 0) {
                if (blk == 0) {
                    __threadfence();
                    if (tid == 0 && mode == 2) { P->ctr[96] = (unsigned)(step - npre + 1); __threadfence(); }
                }
                if (tid == 0) atomicAdd(P->ctr + CM, 1u);
            }
            tm += NHB;
        }
        CLK_END();
        // aux blocks: what was deferred past the head (it needed their shared memory and registers)
        if (!is_main && step + 1 < total) {
            if (has_attn && stg && !late_stage) attn_stage(P, P->L[0], aux >> 2, aux & 3, pos + 1, AS, T, stgsm);
            if (is_router) { if (rpre) tload<8>(T, P->L[2].router + (size_t)(rj * 128 + 8 * warp) * 32 + lane); nw_load(nw, P->L[2].ln2); }
            __syncthreads();
        }
    }
#ifdef MK_PROF
    PROF(30);
    if (blk == 0 && tid == 0) P->clk[1] = 1u;
#endif
}
#if BODY_WRAP
template <int BIG>
__global__ void __launch_bounds__(NTHR, 1)
mk_kernel(const Params* Pp, const long long* tok_in, int tstride, float* logits, long long* gen_out, int B, int pos0, int npre, int nsteps, int mode,
          unsigned tm0, unsigned tx0, unsigned tr0) {
    extern __shared__ uint4 smem[];
    mk_body<BIG>(smem, Pp, tok_in, tstride, logits, gen_out, B, pos0, npre, nsteps, mode, tm0, tx0, tr0);
}
#endif

// ------------------------------------------------------------------------------------------------ host side
// The two kernel instances (batch 1-2, batch > 2) have a counter array each, and each array can be moved between 16
// pages (mk_set_ctr): where the counters lie changes the step (findings 13g), and the engine picks the pages by
// timing itself when it starts (api.py). host.ctr is the array of the instance launched last.
// The same for three small buffers every block reads (router logits, post-attention stream, attention output): at
// batch 1 each is one or two pages and has its own two levels; they have NBPAGE pages of slack and the batch 1-2
// instance may start them on a later page (the batch > 2 instance uses them from the start: there they span many pages).
#define NCPAGE 16
#define NBPAGE 16
struct Handle {
    Params host;
    Params* devv[2];
    unsigned* cslab[2];
    unsigned* ctrv[2];
    char* alt[3];         // the batch 1-2 instance's own copies of rl, xm, attnf, each with NBPAGE pages of slack
    int boff[3];          // and the page each of them starts on
    unsigned tmv[2], txv[2], trv[2];
    int cur;
    cudaStream_t poll_stream;
    int* poll_host;       // pinned: [0..7] counters 96..103 (progress at [0], the watchdog flag at [4]), then tokens
    int serve_B, serve_n, smtot;
};

static int mk_upload(Handle* h) {      // the device copies of the parameters: the same but for the counter array
    for (int c = 0; c < 2; c++) {
        Params t = h->host;
        t.ctr = h->ctrv[c];
        if (c == 0) {
            t.rl = (float*)(h->alt[0] + (size_t)h->boff[0] * 4096);
            t.xm = (float4*)(h->alt[1] + (size_t)h->boff[1] * 4096);
            t.attnf = (uint2*)(h->alt[2] + (size_t)h->boff[2] * 4096);
        }
        if (cudaMemcpy(h->devv[c], &t, sizeof(Params), cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    }
    return 0;
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "mk: CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

template <class T> static int dalloc(T*& p, size_t n) {
    void* q = nullptr;
    if (cudaMalloc(&q, n * sizeof(T)) != cudaSuccess) return 0;
    if (cudaMemset(q, 0, n * sizeof(T)) != cudaSuccess) return 0;
    p = (T*)q;
    return 1;
}

extern "C" {

// slots: per layer 20 entries [qkv, o, dgu, ddown, router, sgu, sdown, egu, edown, kc, vc, ln1, ln2, qn, kn, sinks, sliding, moe, -, -],
// then [embed, head, fnorm, cos, sin, maxctx, debug]
void* mk_create(const long long* s) {
    Handle* h = new Handle();
    memset(h, 0, sizeof(Handle));
    Params& P = h->host;
    for (int l = 0; l < NLAYER; l++) {
        const long long* q = s + l * 20;
        LayerP& L = P.L[l];
        L.qkv = (const uint4*)q[0]; L.o = (const uint4*)q[1]; L.dgu = (const uint4*)q[2]; L.ddown = (const uint4*)q[3];
        L.router = (const uint4*)q[4]; L.sgu = (const uint4*)q[5]; L.sdown = (const uint4*)q[6]; L.egu = (const uint4*)q[7]; L.edown = (const uint4*)q[8];
        L.kc = (uint4*)q[9]; L.vc = (uint4*)q[10];
        L.rp = (const uint4*)q[18]; L.mf = (const uint4*)q[19];
        L.ln1 = (const float*)q[11]; L.ln2 = (const float*)q[12]; L.qn = (const float*)q[13]; L.kn = (const float*)q[14]; L.sinks = (const float*)q[15];
        L.sliding = (int)q[16]; L.moe = (int)q[17];
    }
    const long long* q = s + NLAYER * 20;
    P.embed = (const u16*)q[0]; P.head = (const uint4*)q[1]; P.fnorm = (const float*)q[2]; P.cosT = (const float*)q[3]; P.sinT = (const float*)q[4];
    P.maxctx = (int)q[5]; P.kgmax = (P.maxctx + 15) / 16;
    int ok = dalloc(P.xc, 4 * 1024) && dalloc(P.xm, 4 * 1024) && dalloc(P.qraw, MAXB * 320 * 8) && dalloc(P.attnf, MAXB * 256 * 4) && dalloc(P.rl, MAXB * NEXP) &&
             dalloc(P.hbuf, (size_t)MAXB * HCHUNKS * 4) && dalloc(P.argv, NBLK * MAXB) && dalloc(P.argi, NBLK * MAXB) && dalloc(P.curtok, MAXB) &&
             dalloc(P.gen, (size_t)MAXB * P.maxctx) && dalloc(h->cslab[0], NCPAGE * 1024 + 128) && dalloc(h->cslab[1], NCPAGE * 1024 + 128);
    // (after everything above, whose addresses the batch > 2 instance's placement of s0049/s0050 was found with)
    if (ok) ok = dalloc(h->alt[0], MAXB * NEXP * 4 + NBPAGE * 4096) && dalloc(h->alt[1], 4 * 1024 * 16 + NBPAGE * 4096) && dalloc(h->alt[2], MAXB * 256 * 4 * 8 + NBPAGE * 4096);
    h->ctrv[0] = h->cslab[0]; h->ctrv[1] = h->cslab[1]; h->cur = 0;
    P.ctr = h->ctrv[0];
    if (ok && q[6]) ok = dalloc(P.dbg, (size_t)(NLAYER + 1) * MAXB * HID);
#ifdef MK_PROF
    if (ok) ok = dalloc(P.prof, 3 * PROF_MAX * 2) && dalloc(P.clk, 16) && dalloc(P.bt, 16 * NBLK);
#elif defined(MK_CLK)
    if (ok) ok = dalloc(P.bt, 16 * NBLK);
#endif
    if (!ok) { fprintf(stderr, "mk: allocation failed\n"); return nullptr; }
    for (int c = 0; c < 2; c++) {
        void* d = nullptr;
        if (cudaMalloc(&d, sizeof(Params)) != cudaSuccess) return nullptr;
        h->devv[c] = (Params*)d;
    }
    if (mk_upload(h)) return nullptr;
    h->smtot = SM_TOTAL;
    if (cudaFuncSetAttribute(mk_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, h->smtot) != cudaSuccess ||
        cudaFuncSetAttribute(mk_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, h->smtot) != cudaSuccess) {
        fprintf(stderr, "mk: cannot set %d bytes of shared memory\n", h->smtot);
        return nullptr;
    }
    if (cudaStreamCreateWithFlags(&h->poll_stream, cudaStreamNonBlocking) != cudaSuccess) return nullptr;
    if (cudaMallocHost((void**)&h->poll_host, sizeof(int) * (8 + (size_t)MAXB * P.maxctx)) != cudaSuccess) return nullptr;
    return h;
}

// mode 0: one step, logits written. mode 1: nsteps greedy steps, tokens to gen_out [B][nsteps].
// mode 2: npre forced prompt steps (tok [B][npre+1] with row stride tstride), then nsteps greedy steps into the device buffer.
int mk_run(void* hv, const long long* tok, int tstride, float* logits, long long* gen_out, int B, int pos0, int npre, int nsteps, int mode) {
    Handle* h = (Handle*)hv;
#ifdef MK_PROF
    cudaMemset(h->host.clk, 0, 64); cudaMemset(h->host.prof, 0, 3 * PROF_MAX * 2 * 4);
#endif
    const int c = B > 2;
    h->cur = c; h->host.ctr = h->ctrv[c];
    if (mode == 2) { cudaMemsetAsync(h->host.ctr + 96, 0, 4, 0); h->serve_B = B; h->serve_n = nsteps; }
    // (the watchdog flag ctr[CABORT] is not cleared here: once a wait has given up, every later launch returns at once
    //  until the host has seen it: mk_check, mk_poll)
    if (B <= 2) mk_kernel<0><<<NBLK, NTHR, h->smtot>>>(h->devv[0], tok, tstride, logits, gen_out, B, pos0, npre, nsteps, mode, h->tmv[0], h->txv[0], h->trv[0]);
    else mk_kernel<1><<<NBLK, NTHR, h->smtot>>>(h->devv[1], tok, tstride, logits, gen_out, B, pos0, npre, nsteps, mode, h->tmv[1], h->txv[1], h->trv[1]);
#ifdef MK_PROF
    const int npairs = 4 * B - (4 * B > 7 ? 1 : 0);
#else
    const int npairs = 4 * B;
#endif
    const unsigned layers_m = 47 * (KA + 2 + (B > 2 ? KEBIG : KE)) + 2 * (KA + 3);      // counter bumps per block and step
    h->tmv[c] += (unsigned)NMAIN * (unsigned)(npre + nsteps) * layers_m + (unsigned)NHB * (unsigned)nsteps * (mode ? 2 : 1);
    h->txv[c] += 8u * (unsigned)npairs * (unsigned)(npre + nsteps) * NLAYER;
    h->trv[c] += (unsigned)NROUT * (unsigned)(npre + nsteps) * 47;
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "mk: launch failed: %s\n", cudaGetErrorString(e)); return 1; }
    return 0;
}

// Served cohort: copy progress and tokens while the kernel runs. out: [B][n] int32. Returns the number of finished steps.
int mk_poll(void* hv, int* out) {
    Handle* h = (Handle*)hv;
    const Params& P = h->host;
    if (cudaMemcpyAsync(h->poll_host, P.ctr + 96, 20, cudaMemcpyDeviceToHost, h->poll_stream) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(h->poll_stream) != cudaSuccess) return -1;
    if (h->poll_host[CABORT - 96]) return -2;        // a wait inside the kernel gave up: the cohort's tokens are not valid
    int done = h->poll_host[0];
    if (done > h->serve_n) done = h->serve_n;
    if (done > 0) {
        if (cudaMemcpyAsync(h->poll_host + 8, P.gen, sizeof(int) * (size_t)h->serve_B * P.maxctx, cudaMemcpyDeviceToHost, h->poll_stream) != cudaSuccess) return -1;
        if (cudaStreamSynchronize(h->poll_stream) != cudaSuccess) return -1;
        for (int b = 0; b < h->serve_B; b++) memcpy(out + (size_t)b * h->serve_n, h->poll_host + 8 + (size_t)b * P.maxctx, sizeof(int) * done);
    }
    return done;
}

int mk_smem(void* hv) { return ((Handle*)hv)->smtot; }
int mk_set_fence(void* hv, int v) { Handle* h = (Handle*)hv; h->host.fence = v; return mk_upload(h); }
// Move the counter array of kernel instance c (0: batch 1-2, 1: batch > 2) to page `page` of its slab. Only between
// launches (the device is idle: the caller has synchronized).
int mk_set_off(void* hv, int c, int which, int page);
int mk_set_ctr(void* hv, int c, int page) {
    return mk_set_off(hv, c, 0, page);
}
// which: 0 = the counter array (NCPAGE pages), 1 = router logits, 2 = post-attention stream, 3 = attention output
// (NBPAGE pages, batch 1-2 instance only; their contents are transient within a step, a launch does not rely on what a buffer held before)
int mk_set_off(void* hv, int c, int which, int page) {
    Handle* h = (Handle*)hv;
    if (c < 0 || c > 1 || which < 0 || which > 3 || page < 0 || page >= (which ? NBPAGE : NCPAGE) || (which && c)) return 1;
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    if (which) { h->boff[which - 1] = page; return mk_upload(h); }
    h->ctrv[c] = h->cslab[c] + (size_t)page * 1024;
    if (cudaMemset(h->ctrv[c], 0, 128 * sizeof(unsigned)) != cudaSuccess) return 1;
    h->tmv[c] = h->txv[c] = h->trv[c] = 0u;
    if (h->cur == c) h->host.ctr = h->ctrv[c];
    return mk_upload(h);
}

// Did a wait inside the kernel give up (its watchdog) since the last call? Waits for the device; clears the flag.
int mk_check(void* hv) {
    Handle* h = (Handle*)hv;
    unsigned v = 0;
    if (cudaMemcpy(&v, h->host.ctr + CABORT, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    if (v) {
        // The launch that gave up left the counters wherever its blocks stopped: the next launch starts from what
        // they hold now (targets are relative), with the flag cleared.
        unsigned c[128];
        if (cudaMemcpy(c, h->host.ctr, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
        h->tmv[h->cur] = c[CM]; h->txv[h->cur] = c[CX]; h->trv[h->cur] = c[CR];
        cudaMemset(h->host.ctr + CABORT, 0, 4);
    }
    return v ? 1 : 0;
}
int mk_read_dbg(void* hv, float* out) {
    Handle* h = (Handle*)hv;
    if (!h->host.dbg) return 1;
    CK(cudaMemcpy(out, h->host.dbg, (size_t)(NLAYER + 1) * MAXB * HID * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

int mk_read_bt(void* hv, unsigned* out) {
    Handle* h = (Handle*)hv;
    if (!h->host.bt) return -1;
    return cudaMemcpy(out, h->host.bt, sizeof(unsigned) * 16 * NBLK, cudaMemcpyDeviceToHost) != cudaSuccess;
}
int mk_read_prof(void* hv, unsigned* out) {
#ifdef MK_PROF
    Handle* h = (Handle*)hv;
    CK(cudaMemcpy(out, h->host.prof, 3 * PROF_MAX * 2 * 4, cudaMemcpyDeviceToHost));
    return 0;
#else
    return 1;
#endif
}

}  // extern "C"
