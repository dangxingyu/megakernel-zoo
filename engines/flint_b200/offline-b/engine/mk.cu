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
#define MATTN_CPRE 0     // experiment: pairs on main blocks (batch 7-8) request the block's output-projection tiles inside the attention: slower (3991 -> 4071 us)
#endif
#ifndef KV_BATCH_MAIN
#define KV_BATCH_MAIN 0  // experiment: batched K/V loads for the pairs on main blocks (batch 7-8): nothing (journal)
#endif
#ifndef KV_BATCH_AUXBIG
#define KV_BATCH_AUXBIG 0  // experiment: batched K/V loads on the attention blocks of the batch > 2 kernel: nothing
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
#define PROF_BLK(b) ((b) == 0 ? 0 : (b) == NMAIN ? 1 : (b) == NMAIN + 8 ? 2 : -1)
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
#else
#define PROF(id) do {} while (0)
#define PROFX(id) do {} while (0)
#define BT(ev) do {} while (0)
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
__device__ __forceinline__ float4 ldx(const float4* p) { float4 r; asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p)); return r; }
__device__ __forceinline__ uint4 ldx(const uint4* p) { uint4 r; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r; }
__device__ __forceinline__ uint2 ldx(const uint2* p) { uint2 r; asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p)); return r; }
__device__ __forceinline__ float ldx(const float* p) { float r; asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(r) : "l"(p)); return r; }
__device__ __forceinline__ int ldx(const int* p) { int r; asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(r) : "l"(p)); return r; }
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
__device__ __forceinline__ long long ldx(const long long* p) { long long r; asm volatile("ld.global.cg.s64 %0, [%1];" : "=l"(r) : "l"(p)); return r; }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Wait until a counter reaches its target. Watchdog: after about a second of spinning the launch is declared dead
// (ctr[100] = 1) and every wait in every block returns at once, so a bug can never leave a kernel spinning forever.
#define CABORT 100
__device__ __forceinline__ void wait_ctr(const Params* P, int which, unsigned target) {
    if (threadIdx.x == 0) {
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
__device__ __forceinline__ uint2 xget(const uint2* xf, int Ls, int chunk, int lane) { return lane < Ls ? xf[chunk * Ls + lane] : make_uint2(0u, 0u); }
// u16 index inside a per-sequence global fragment buffer [chunk][4 lanes]
__device__ __forceinline__ int gidx(int chunk, int kk) { return ((chunk * 4 + ((kk & 7) >> 1)) * 2 + (kk >> 3)) * 2 + (kk & 1); }

template <int N> __device__ __forceinline__ void tload(uint4* T, const uint4* p) {
#pragma unroll
    for (int i = 0; i < N; i++) T[i] = p[i * 32];
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
            uint4 r = ldx((const uint4*)((const uint2*)X + t2 * 1024 + rg * 8 + g0));     // rows g0, g0+8 | g0+1, g0+9, two sequences each
            if (is_sent(r)) r = ldh((const uint4*)((const uint2*)X + t2 * 1024 + rg * 8 + g0), P);
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
        if (on && !(g & 1)) dep |= put32((u32*)(P->hbuf + (size_t)b * HCHUNKS * 4) + (gidx(chunk0 + (j >> 4), j & 15) >> 1), hv | (pv << 16));
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
        dep = put64((uint2*)X + t * 1024 + blk * 8 + g, (__float_as_uint(xn.x) >> 16) | (__float_as_uint(xn.y) & 0xFFFF0000u), (__float_as_uint(xn.z) >> 16) | (__float_as_uint(xn.w) & 0xFFFF0000u));
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
    const int kg0 = lo >> 4, nkg = pos > lo ? ((pos - 1) >> 4) - kg0 + 1 : 0;
    attn_consts(P, L, kvg, pos, S);
    if (warp < nkg) tload<8>(T, L.kc + cbase + (size_t)(kg0 + warp) * 8 * 32 + lane);
    {   // V: this warp mixes dim chunk warp%8 over the key groups of parity warp/8: the first 8 of them in registers
        const int dc = warp & 7, cls = warp >> 3;
        const uint4* vt = L.vc + cbase + (size_t)((kg0 + cls) * 8 + dc) * 32 + lane;
#pragma unroll
        for (int j = 0; j < 8; j++) if (cls + 2 * j < nkg) T[8 + j] = vt[(size_t)j * 16 * 32];
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
__device__ __forceinline__ void attention(const Params* P, const LayerP& L, int b, int kvg, int pos, const AttnSm& S, uint4* T, bool staged, bool smstaged, int pblk, int* ppi, bool has_pre, bool bkv, const uint4* ctiles = nullptr) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    float* sm = S.sm;
    const size_t cbase = ((size_t)(b * NKV + kvg) * P->kgmax) * 8 * 32;
    uint4* kc = L.kc + cbase;
    uint4* vc = L.vc + cbase;
    const int lo = L.sliding ? (pos - 512 > 0 ? pos - 512 : 0) : 0;
    const int kg0 = lo >> 4, nkg = pos > lo ? ((pos - 1) >> 4) - kg0 + 1 : 0;
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
        if (ctiles) tload<16>(T, ctiles);
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
    // ---- stage 3: score of the current key (partial per warp), scores of the cached keys, per-warp maxima.
    //      Warp w owns key groups kg0 + w (scores e0), + 16, + 32, ... (contexts over 256: ex).
    //      Thread (g, t): head g, keys 2t, 2t+1, 2t+8, 2t+9 of each group.
    float e0[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float ex[7][4];
    {
        const float ps = warp_sum(pm2(v0, S.ks[d0], v1, S.ks[d1], FM_DOT));
        if (lane == 0) sm[64 + warp] = ps;
        float mx = -INFINITY;
        int j = 0;
        for (int i = warp; i < nkg; i += 16, j++) {
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
                for (int dc = 0; dc < 8; dc++) T[dc] = kt[dc * 32];
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
            for (int jv = 0; jv < 8; jv++) if (i + 2 * jv < nkg) T[jv] = vc[(size_t)((kg0 + i + 2 * jv) * 8 + dc) * 32 + lane];
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
        for (int i = warp + 16; i < nkg; i += 16, j++)
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
        for (int i = warp; i < nkg; i += 16, j++) {
            const float a0 = j == 0 ? e0[0] : ex[j - 1][0], a1 = j == 0 ? e0[1] : ex[j - 1][1];
            const float a2 = j == 0 ? e0[2] : ex[j - 1][2], a3 = j == 0 ? e0[3] : ex[j - 1][3];
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
    }
    PROFX(65);
    __syncthreads();
    // ---- stage 6: V mix. Warp = (dim chunk dc, parity cls): sums [dim 16][head 8] over the key groups of its parity
    {
        const int dc = warp & 7, cls = warp >> 3;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        if (!bkv) {
#pragma unroll
        for (int j = 0; j < 8; j++) if (staged && cls + 2 * j < nkg) mma(acc, T[8 + j], S.wb[(cls + 2 * j) * 32 + lane]);
        int i = cls;
        if (staged) i = cls + 16;
        if (smstaged) {
            for (int j = 8; j < 8 + NV2 && cls + 2 * j < nkg; j++) mma(acc, S.v2[((j - 8) * 16 + warp) * 32 + lane], S.wb[(cls + 2 * j) * 32 + lane]);
            i = cls + 2 * (8 + NV2);
        }
        for (; i < nkg; i += 2) {
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
#if !KV_VEARLY
#pragma unroll
        for (int j = 0; j < 8; j++) if (i + 2 * j < nkg) T[j] = vc[(size_t)((kg0 + i + 2 * j) * 8 + dc) * 32 + lane];
#endif
#pragma unroll
        for (int j = 0; j < 8; j++) if (staged && cls + 2 * j < nkg) mma(acc, T[8 + j], S.wb[(cls + 2 * j) * 32 + lane]);
        while (i < nkg) {
#pragma unroll
            for (int j = 0; j < 8; j++) if (i + 2 * j < nkg) mma(acc, T[j], S.wb[(i + 2 * j) * 32 + lane]);
            i += 16;
#pragma unroll
            for (int j = 0; j < 8; j++) if (i + 2 * j < nkg) T[j] = vc[(size_t)((kg0 + i + 2 * j) * 8 + dc) * 32 + lane];
        }
        }
        S.part[warp * 32 + lane] = *(const uint4*)&acc;
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
// smi[464 + w]: experts in use in warp w's 32; the byte table reuses smi[0..255] (experts 0..127) and sm[0..255].
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
    } else {
        const int b = warp >> 1, hf = warp & 1;
        if (b < B) {
            float4 x = ldx((const float4*)(rl + b * NEXP + hf * 128) + lane);
            if (is_sent(x)) x = ldh((const float4*)(rl + b * NEXP + hf * 128) + lane, P);
            if (zmode) { const float sc = sm[480 + b]; x.x *= sc; x.y *= sc; x.z *= sc; x.w *= sc; }
            float v0 = 1.0f / (1.0f + __expf(-x.x)), v1 = 1.0f / (1.0f + __expf(-x.y)), v2 = 1.0f / (1.0f + __expf(-x.z)), v3 = 1.0f / (1.0f + __expf(-x.w));
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
        }
    }
    PROFX(71);
    __syncthreads();
    PROFX(72);
    if (warp < B) {
        // this warp: sequence b = warp. lane holds candidates lane (a) and, with 48 candidates, lane + 32 (c, lanes 0..15)
        const int b = warp;
        const bool wide = !big;
        float va = (wide || lane < 12) ? sm[b * 48 + lane] : -INFINITY, vc = (wide && lane < 16) ? sm[b * 48 + 32 + lane] : -INFINITY;
        const int ia = (wide || lane < 12) ? smi[b * 48 + lane] : 0, ic = (wide && lane < 16) ? smi[b * 48 + 32 + lane] : 0;
        float sum = 0.f, ts[TOPK]; int ti[TOPK];
#if ROUTE_RANK
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
            if (lane < 12 && ra < TOPK) { sm[b * 48 + 16 + ra] = va; smi[b * 48 + 16 + ra] = ia; }
            __syncwarp();
            for (int r = 0; r < TOPK; r++) { ts[r] = sm[b * 48 + 16 + r]; ti[r] = smi[b * 48 + 16 + r]; sum += ts[r]; }
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
        if (lane < TOPK) {
            const float w = ts[lane] / sum;
            const int id = ti[lane];
            if (B == 1) {     // the units are the 6 experts in ascending id
                int u = 0;
                for (int j = 0; j < TOPK; j++) u += ti[j] < id;
                un[1 + u] = id & 255; uwl[u * MAXB] = w;
                if (lane == 0) un[0] = TOPK;
            } else { sm[400 + b * 8 + lane] = w; smi[400 + b * 8 + lane] = id; }
        }
    }
    PROFX(73);
    __syncthreads();
    if (B == 1) return;
    // ---- union. (The earlier version kept "first occurrence" flags at smi[480..543], past smi's 512 ints and into
    // un[0..31] for batches over 4: a warp could read another warp's freshly written unit as a flag, a rare race.)
    unsigned char* tab = tid < 128 ? (unsigned char*)smi + tid * 8 : (unsigned char*)sm + (tid - 128) * 8;     // this thread's expert (tid < 256)
    if (tid < 256) { smi[tid] = 0; ((int*)sm)[tid] = 0; }
    if (tid < MAXU * MAXB) uwl[tid] = 0.f;
    __syncthreads();
    if (tid < 8 * B && (tid & 7) < TOPK) {
        const int id = smi[400 + tid] & 255;
        unsigned char* te = id < 128 ? (unsigned char*)smi + id * 8 : (unsigned char*)sm + (id - 128) * 8;
        te[tid >> 3] = (unsigned char)((tid & 7) + 1);
    }
    __syncthreads();
    unsigned lo = 0u, hi = 0u, bal = 0u;
    if (tid < 256) {
        lo = ((const unsigned*)tab)[0]; hi = ((const unsigned*)tab)[1];
        bal = __ballot_sync(0xffffffffu, (lo | hi) != 0u);
        if (lane == 0) smi[464 + warp] = __popc(bal);
    }
    __syncthreads();
    if (tid < 256) {
        int u = __popc(bal & ((1u << lane) - 1u));
        for (int w = 0; w < warp; w++) u += smi[464 + w];
        if ((lo | hi) != 0u && u < MAXU) {
            un[1 + u] = tid;
            for (int b = 0; b < B; b++) {
                const unsigned sl = ((b < 4 ? lo : hi) >> (8 * (b & 3))) & 255u;
                if (sl) uwl[u * MAXB + b] = sm[400 + b * 8 + sl - 1];
            }
        }
        if (tid == 0) { int n = 0; for (int w = 0; w < 8; w++) n += smi[464 + w]; un[0] = n; }
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
    const bool mattn = big && is_main && blk + NAUX < 4 * B;          // more pairs than aux blocks (batch 7, 8): main block m takes pair NAUX + m while it waits
    const bool rpre = is_router && !big && !ER;        // (track B: the early router loads its own tiles)
    const bool late_stage = ER && is_router && aux < nattn_blocks;       // track B router blocks with an attention pair: staged at the layer's start, after pass 1                     // router tiles loaded ahead into registers (no attention duty on these blocks then)
    const bool stg = !is_router || big;                      // attention K/V tiles in registers
    // Router blocks that also have an attention pair (batch > 2): after their attention they first request the router
    // tiles into the staging registers and stage the next layer's K/V only after the router duty. (Staging first left
    // the router tiles to be loaded when needed: a trip through the busy memory queue on the path every main block waits on.)
    const bool rdefer = RDEFER && big && is_router && has_attn && !late_stage;
    const bool stgsm = !is_router;                             // ... and further key groups in shared memory (router blocks need it for fragments)
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
#ifndef MK_PROF
#define PPI nullptr
#endif
    // zero what is never written: fragment lanes of unused sequences, rows 8..15 of the q operand
    for (int i = tid; i < (64 << 10) / 4; i += NTHR) ((u32*)xf)[i] = 0u;
    for (int i = tid; i < 1024; i += NTHR) ((u32*)AS.qA)[i] = 0u;
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
            tload<NPA>(T, P->L[0].qkv + (size_t)(blk * 320 + 20 * warp) * 32 + lane);
            nw_load(nw, P->L[0].ln1);
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
                // The five leaders' q/k/v words hold the sentinel: issued at the end of the previous layer (first layer of
                // a step: now), confirmed here. The signal follows the norm: it also says "this block has read x".
                {
                    u32 qbad = 0u;
                    if (warp < 5) {
                        if (l == 0) q_sentinel(P, blk, B);
                        qbad = q_chk(P, blk, B, 0u);
                    }
                    norm_frag<big && NORM_WS>(P, l == 0 ? P->embed : nullptr, toks, ts, tcol, nw, B, Ls, xf, sm, blk == 0, (P->dbg && blk == 0) ? P->dbg + (size_t)l * MAXB * HID : nullptr, xo, blk, nullptr, nscr);
                    if (warp < 5) { CONFIRM(qbad, q_chk(P, blk, B, vo)); if (EB && lane == 0) atomicAdd(P->ctr + CM, 1u); }
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
                PROF(31);
                __syncthreads();
                PROF(32);
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
                // A main block with an attention pair (batch 7-8): "has every block started A?" (then q/k/v hold the sentinel
                // or this layer's values) is asked now by a warp that is idle while the leaders reduce and store, not after
                // them by thread 0: one trip to memory less in front of the attention every other block waits for.
                else if (MATTN_EPOLL && mattn && tid == 5 * 32) poll_ctr(P, CM, tm + KA * NMAIN);
                PROF(33);
                __syncthreads();      // the leaders' writes and counters go out before the preload traffic
                PROF(34);
                tm += KA * NMAIN;
                PROF(3); BT(1);
                nw_load(nw, L.ln2);
                if (mattn) {
                    if (!MATTN_EPOLL) wait_ctr(P, CM, tm);
                    attention(P, L, (blk + NAUX) >> 2, (blk + NAUX) & 3, pos, AS, T, false, false, blk, PPI, l > 0, KV_BATCH_MAIN != 0,
                              MATTN_CPRE ? L.o + (size_t)(blk * 256 + 16 * warp) * 32 + lane : nullptr);
                    if (P->fence) __threadfence();
                    __syncthreads();
                }
                if (!(mattn && MATTN_CPRE)) tload<16>(T, L.o + (size_t)(blk * 256 + 16 * warp) * 32 + lane);      // preload C
                PROF(35);
                // ============ (B: attention on the aux blocks) ============ C: output projection + residual
                // Sentinel into this block's rows of the post-attention stream and into the shared expert's hidden units
                // (their readers, the previous layer's phase D and down phase, are done: this block has read the whole
                // layer input, which every block's down phase had to write): issued before the wait for the attention,
                // confirmed after it.
                if (warp == 0) {
                    x_sentinel(P, blk, lane, B, true);
                    if (L.moe) h_sent(P, blk, lane, B, HSLOT_SHARED * 32);
                }
                tx += 8 * npairs;
                wait_ctr(P, CX, tx);
#if ER
                if (L.moe) wait_ctr(P, CZ, tr + NROUT);      // (the early router has read the layer input, too)
#endif
                PROF(4); BT(2);
                {
                    u32 mbad = 0u;
                    if (warp == 0) { mbad = x_chk(P, blk, lane, B, true, 0u); if (L.moe) mbad |= hs_chk(P, blk, lane, B, HSLOT_SHARED * 32, 0u); }
                    frag_copy(xf, Ls, 256, B, [&](int b, int chunk) { return P->attnf + ((size_t)b * 256 + chunk) * 4; }, P);
                    if (warp == 0) CONFIRM(mbad, x_chk(P, blk, lane, B, true, vo) | (L.moe ? hs_chk(P, blk, lane, B, HSLOT_SHARED * 32, vo) : 0u));
                }
                __syncthreads();
                if (EB && tid == 0) atomicAdd(P->ctr + CM, 1u);      // (after the barrier: it also says "this whole block has read the attention output")
                {
                    float4 c4[4] = {zero4, zero4, zero4, zero4};
#pragma unroll
                    for (int i = 0; i < 16; i++) mma(c4[i & (NACC - 1)], T[i], xget(xf, Ls, 16 * warp + i, lane));
                    red[warp * 32 + lane] = f4add(f4add(c4[0], c4[1]), f4add(c4[2], c4[3]));
                }
                __syncthreads();
                if (warp == 0) {
                    float4 c = red[lane];
                    for (int w = 1; w < 16; w++) c = f4add(c, red[w * 32 + lane]);
                    x_update(P, blk, lane, B, make_float4(rbA(c.x), rbA(c.y), rbA(c.z), rbA(c.w)), xo, true);
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
                    route_local(P, P->rl, B, sm, smi, un, uwl, blk, PPI, big, ER != 0);
                    PROF(9); BT(6);
                    const int U = un[0] < 6 * B ? (un[0] > 0 ? un[0] : 1) : 6 * B;
                    {
                        // A row group (16 rows x 128 tiles) = 8 pieces of 16 tiles; a slot = 2 row groups = one 16-tile batch
                        // on each of the 16 warps (16 loads in flight per warp: the memory's full rate). The next slot's
                        // tiles are requested before this slot's partial sums are folded.
                        const int G = U * 64;
                        const int e0 = blk * G / NMAIN, e1 = (blk + 1) * G / NMAIN;
                        const int half = warp >> 3, wi = warp & 7;
                        {
                            const int rg = e0 + half;
                            if (rg < e1) tload<16>(T, L.egu + ((size_t)((un[1 + (rg >> 6)] & 255) * 64 + (rg & 63)) * 128 + wi * 16) * 32 + lane);
                        }
                        // the hidden-unit words this warp will write: sentinel now (the previous layer's down phase is
                        // done everywhere), read back and signalled while the first tiles are on their way
                        u32 hdep2 = 0u;
                        {
                            if (big) { for (int rg = e0 + warp; rg < e1; rg += 16) hdep2 |= h_sent(P, rg & 63, lane, B, (rg >> 6) * 32, uwl + (rg >> 6) * MAXB); }
                            else if (wi == 0) { for (int rg = e0 + half; rg < e1; rg += 2) hdep2 |= h_sent(P, rg & 63, lane, B, (rg >> 6) * 32, uwl + (rg >> 6) * MAXB); }
                        }
                        if (depi_e && warp == 0) {      // (phase D's epilogue, see there; its partial sums are in slots 16..31, used again from the second slot on)
                            float4 c = red[16 * 32 + lane];
                            for (int w = 1; w < 16; w++) c = f4add(c, red[(16 + w) * 32 + lane]);
                            h_store(P, c, blk, lane, B, HSLOT_SHARED * 32);
                        }
                        if (big || wi == 0) { SETTLE(hdep2); if (EB && lane == 0) atomicAdd(P->ctr + CM, 1u); }
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
                    if (big) {
                        __syncthreads();         // every folded sum is in shared memory
                        const int G = U * 64, e0 = blk * G / NMAIN, e1 = (blk + 1) * G / NMAIN;
                        for (int rg = e0 + warp; rg < e1; rg += 16) {
                            const float4 cq = ((const float4*)(sb + (104 << 10)))[(rg - e0) * 32 + lane];
                            h_store(P, cq, rg & 63, lane, B, (rg >> 6) * 32, uwl + (rg >> 6) * MAXB);
                        }
                        if (!EB && lane == 0) atomicAdd(P->ctr + CM, 1u);      // every warp, after its own writes
                        tm += KEBIG * NMAIN;
                    } else {
                        if ((warp & 7) == 0) {       // the two leaders (warp 0 also for its shared-expert stores of phase D):
                            if (!EB && lane == 0) atomicAdd(P->ctr + CM, 1u);
                        }
                        tm += KE * NMAIN;
                    }
                    PROF(10); BT(7);
                    // preload F: the first 8 items (item = warp/2: units in ascending order, then the two halves of the shared expert)
                    const int wj = warp & 1;
                    {
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
                            } else {
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
                                if (it0 > 0 && it < nitems) {
                                    const uint4* dt = it < U ? L.edown + ((size_t)((un[1 + it] & 255) * 128 + blk) * 32 + 16 * wj) * 32 + lane
                                                             : L.sdown + ((size_t)(blk * 64 + (it - U) * 32) + 16 * wj) * 32 + lane;
                                    tload<8>(T + 8, dt + 8 * 32);
                                }
                            }
                            PROF(41);
                            __syncthreads();
                            PROF(42);
                            float4 c = zero4;
                            if (it < nitems) {
                                float4 c4[4] = {zero4, zero4, zero4, zero4};
#pragma unroll
                                for (int i = 0; i < 16; i++) mma(c4[i & (NACC - 1)], T[i], xget(xf, Ls, (warp >> 1) * 32 + 16 * wj + i, lane));
                                c = f4add(f4add(c4[0], c4[1]), f4add(c4[2], c4[3]));
                            }
                            red[warp * 32 + lane] = c;
                            {   // the next round's tiles go into the registers now: they arrive while the leaders work
                                const int itn = it + 8;
                                if (itn < nitems) {
                                    const uint4* dt = itn < U ? L.edown + ((size_t)((un[1 + itn] & 255) * 128 + blk) * 32 + 16 * wj) * 32 + lane
                                                              : L.sdown + ((size_t)(blk * 64 + (itn - U) * 32) + 16 * wj) * 32 + lane;
                                    if (big) tload<8>(T, dt); else tload<16>(T, dt);      // larger batches: the first half only (see above)
                                }
                            }
                            PROF(43);
                            __syncthreads();
                            PROF(44);
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
                            if (it0 + 8 < nitems) __syncthreads();
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
                    PROF(12); BT(9);
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
                // Sentinels for the next layer, confirmed after the wait below: q/k/v (this layer's attention has read them:
                // this block has its output) and, for a main block with an attention pair, the pair's output words
                // (every block has copied this layer's attention output: they all signalled C).
                if (l + 1 < NLAYER) {
                    if (warp < 5) q_sentinel(P, blk, B);
                    if (mattn && warp < 8) attn_sentinel(P, (blk + NAUX) >> 2, (blk + NAUX) & 3);
                }
                // preload A of the next layer (or the final norm weights), then wait for every main block
                if (l + 1 < NLAYER) {
                    tload<NPA>(T, P->L[l + 1].qkv + (size_t)(blk * 320 + 20 * warp) * 32 + lane);
                    nw_load(nw, P->L[l + 1].ln1);
                } else nw_load(nw, P->fnorm);
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
                    PROF(21);
                    attention(P, L, aux >> 2, aux & 3, pos, AS, T, stg, stg && stgsm, blk, PPI, l > 0 && !ER, KV_BATCH && (!big || KV_BATCH_AUXBIG));
                    PROF(22);
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
                            for (int i = 0; i < 8; i++) mma(cl, R8[i], xget(xlo, Ls, 8 * warp + i, lane));
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
