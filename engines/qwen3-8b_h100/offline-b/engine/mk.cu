// Decode megakernel for Qwen3-8B (bf16 weights) on one H100.
// One CUDA source: device code for every operator of a decode step + a small host API (ctypes).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Two builds (engine/TRACK, read by build.sh):
//   track B (default): model-specific, lossless: weights as a 12-bit code of the checkpoint's BF16 values (3 loads
//     per 4 ksteps) with exact power-of-two scales; residual stream and all accumulators float32; RMSNorm fused
//     into the matvecs (the input is bf16(w * x), the outputs are multiplied by 1/rms).
//   track A (-DTRACK_A): aligned with the general engines. Weights: BF16 as stored, re-laid out only (MMA
//     A-fragment order, 4 loads per 4 ksteps). Activations: BF16 wherever the Hugging Face BF16 model has a BF16
//     tensor (every matvec output, residual stream, SiLU, products, q/k norm and RoPE steps, attention
//     probabilities, logits), float32 only inside an operator (RMSNorm statistics, matvec accumulators, and the
//     attention operator from the q . k accumulator through the softmax: scores are float32, as in the fused
//     attention kernels of torch SDPA / flash-attention that the general engines and the HF default path run;
//     see SCORE_RB).
//     Data flow: RMSNorm is applied before the matvec, y = bf16(w * bf16(x / rms)), as in the model.
#ifdef TRACK_A
#define LPG 4
#define NOTAB          // no per-tile table: no row scales, no exceptions
#define NO_ATT_TT
#else
#define LPG 3
#endif
#define GU4 (LPG * 32)   // uint4 per group of 4 ksteps (one load = 32 lanes x 16 B)
#define H 4096
#define NL 36
#define QKV_ROWS 6144
#define INNER 12288
#define VOCAB 151936
// Warps per block. 65536 registers per SM: 32 warps -> 64 registers per thread, 24 -> 80 (the build passes
// -maxrregcount accordingly), 16 -> 128. Measured (generate us/step B=1 / B=8), each with its best pipeline depth:
// 32 warps, 2 loads: 5377 / 5857; 24 warps, 3 loads: 5340 / 5838; 16 warps, 8 loads: 5350 / 5878.
#ifndef NWARP
#define NWARP 24
#endif
#define NTHREAD (NWARP * 32)
#define MAXB 8
#ifndef LM_NEXT
#define LM_NEXT 1      // step(): prefetch layer 0's first q/k/v tile at the end of lm_head for the next launch
#endif
#ifndef WCB
#define WCB 3               // wide-layout input copy: loads per thread requested before the stores (4 spills)
#endif
#ifndef ECB
#define ECB 2               // input copy: loads requested back to back per thread before the stores
#endif
#ifndef NORM_LOAD
#define NORM_LOAD 1      // track A, 3+ sequences: the loaders normalise the input fragments as they use them (1) instead of a norm phase of its own (0)
#endif
#ifndef NORM_NB
#define NORM_NB 1        // ... and the loaders wait only for each other (named barrier behind their first loads), not for the coordinator
#endif
#ifndef NORM_SX
#define NORM_SX 1        // ... once per block: the first tile's normalised fragments are kept in shared memory for the others
#endif
#ifndef XS_ALL
#define XS_ALL 0         // the same for every K=4096 phase
#endif
#ifndef NORM_WOFF
#define NORM_WOFF HF_OFF // track A, NORM_LOAD: the norm weights (8 KB, wide fragment order) in shared memory (free in wide mode)
#endif
#ifndef NORM_WARP0
#define NORM_WARP0 (NWARP / 2)   // first of the 8 warps that compute 1/rms (one per sequence) in a norm phase's entry
#endif
#ifndef STASH_ALL
#define STASH_ALL 1   // every K=4096 phase starts from a stashed tile (0: only the o projection, as before s0029)
#endif
#ifndef WPF_REST
#define WPF_REST 1       // wait-time prefetch after a partial stash: the rest of the first tile (0: the second tile)
#endif
#ifndef CAP_H
#define CAP_H 4       // end-of-phase stash: at most this many loads per warp of a K=4096 tile (up to 4 sequences)
#endif
#ifndef CAP_HW
#define CAP_HW 4      // ... with 5 to 8 sequences
#endif
#ifndef CAP_D
#define CAP_D 3          // ... of a K=12288 tile (room for 3)
#endif
#ifndef ADAPT
#define ADAPT true       // stop stashing when the grid barrier is complete
#endif
#ifndef SG
#ifdef TRACK_A
#define SG 4            // (track A, staged engine: B=1 -7, B=8 -22 vs 2, e0130; track B: 4 is +20 at B=1)
#else
#define SG 2            // end-of-phase stash: loads per warp in flight at a time
#endif
#endif
#ifndef FENCE_BAR
#define FENCE_BAR 1    // loaders start their end-of-phase stash only after the coordinator's last epilogue is fenced
#endif
// Split attention (long contexts, 1-2 sequences). One block per (sequence, kv head) reads 512 B of K and V per
// position at ~60 GB/s: at 9216 positions that is 76 us per layer with 8 blocks while 124 SMs wait. From LONG_NT
// position tiles on (1136 positions, beyond the prompt-64 workload) the work is spread over all SMs in two
// sub-phases with a grid barrier between them (one more per layer):
// (1) scores: NB/(8B) blocks per unit, each a chunk of the positions, results to global memory;
// (2) softmax + V: 8 blocks per unit, one per dim tile of 16, all 24 warps on it.
// With 8 sequences the 64 one-block units already run at 92% of the HBM rate (2.86 TB/s at 9000 positions); the
// same split there (2 blocks per unit, 4 dim tiles each, from 4096 positions) measured 8728 us per step against
// 8173, so it is not used: SPLIT_ON is false for 3+ sequences.
#define SPLIT_ON(B, pos) ((B) <= 2 && (((pos) + 16) >> 4) >= LONG_NT)
#define SPLIT_KB(B, nb) ((nb) / (8 * (B)))                              // score blocks per (sequence, kv head)
#define SPLIT_VB(B, nb) (SPLIT_KB(B, nb) < 8 ? SPLIT_KB(B, nb) : 8)     // softmax + V blocks per (sequence, kv head)
#ifndef SM_UNROLL
#define SM_UNROLL 1       // 4: softmax loops unrolled (same order of operations; measured: no effect at 9000 positions)
#endif
#if SM_UNROLL > 1
#define SMU_ _Pragma("unroll 4")
#else
#define SMU_
#endif
#ifndef VSB
#define VSB 0         // one-block attention below LONG_NT tiles: V loads in flight per warp (0 / 1: one at a time, as inherited)
#endif
#ifndef VBATCH
#define VBATCH 6      // one-block attention, long contexts (3+ sequences): V loads in flight per warp (4 -> 6: -70 us per step at 8192 positions, 8 sequences)
#endif
#ifndef PREP_FENCE
#define PREP_FENCE 0   // 0: no fence behind the K/V append (the block reads the row itself, after a block barrier, which
                       // makes the block's global stores visible to its threads; other blocks read it only behind the
                       // grid barrier, before which the block fences: end of attention_body / att_k_chunk)
#endif
#ifndef XSLICE
#define XSLICE 1        // 1-2 sequences: each loader warp fetches the slice of the input fragments it multiplies (no block-wide input copy and no block barrier at the phase start; track A: phases without a norm)
#endif
#ifndef COORD_RMS
#define COORD_RMS 0     // 1: 1/rms of a norm phase computed by the coordinator warp instead of loader warps (measured: +25..35 us at B=1-2, the coordinator is late for its first tile; with the entry barrier also +80 at B=8)
#endif
#ifndef ENTRY_NB
#define ENTRY_NB 1      // track B, 3+ sequences: no block barrier at the phase start (nothing is copied)
#endif
#ifndef NB_MAT_W
#define NB_MAT_W 128     // the same with 3+ sequences (track A too: -27 us at B=8)
#endif
#ifndef NB_MAT
#ifdef TRACK_A
#define NB_MAT 0         // (track A, measured with 128: B=1 +20, B=2 +25, B=8 -27 us per step)
#else
#define NB_MAT 128       // q/k/v, gate+up, down: tiles on the first 128 blocks only (384 / 1536 / 256 tiles: 3 / 12 / 2 each, no block with one more); 0: all blocks
#endif
#endif
#ifndef QKV_NOWAIT
#define QKV_NOWAIT 1     // one-block attention: only the attention blocks wait at the end of q/k/v; the others count in and go on
#endif
#ifdef TRACK_A
#define QKV_NW(B, pos) (QKV_NOWAIT && !SPLIT_ON(B, pos) && (B) >= WIDE_MIN_B)   // (track A, 1-2 sequences: measured +30 us)
#else
#define QKV_NW(B, pos) (QKV_NOWAIT && !SPLIT_ON(B, pos))
#endif
#ifndef KV_IL
#define KV_IL 0
#endif
#ifndef KV_PAD
#define KV_PAD 0       // position tiles of padding between the K/V regions of two (layer, sequence, head) units
#endif
#ifndef VBATCH_S
#define VBATCH_S 4    // the same in the split path's V blocks (1-2 sequences: 6 and 8 measured slower)
#endif
#ifndef ONFLY
#define ONFLY 1          // one-block attention at long context: bf16 probabilities built by the V pass (0: by a pass of the softmax)
#endif
#ifndef VPF
#define VPF 0            // long contexts: 4 KB chunks of V each warp prefetches while the softmax runs (measured at 8192 positions: 2 chunks +40 us at B=1, +50 at B=8)
#endif
#ifndef KBATCH
#define KBATCH 1
#endif
#ifndef TOUCH_VS
#define TOUCH_VS 4       // split attention: rounds of 8 loads per idle warp that pull V into L2 during the score copy and softmax (0: off)
#endif
#ifndef TOUCH_VS_AFTER
#define TOUCH_VS_AFTER 0   // 1: after the block's o stash instead of before it
#endif
#ifndef PREP_PF
#define PREP_PF 1
#endif
#ifndef SPLIT_PF
#define SPLIT_PF 0        // 1: split attention prefetches the K chunk during the prep and the V slice during the softmax (measured +54 us at 9000 positions)
#endif
#ifndef LAST_NO_STASH
#define LAST_NO_STASH 0    // n > 0: the last n blocks to arrive at a grid barrier skip the end-of-phase stash (measured: no gain, 1 / 8 / 32: +2..+15 us)
#endif
#ifndef EARLY_COUNT
#define EARLY_COUNT 1      // the grid barrier's increment right behind the fence, before the block's own barrier and control code
#endif
#ifndef ATT_WAIT
#ifdef TRACK_A
#define ATT_WAIT 16    // (track A: as before)
#else
#define ATT_WAIT 8         // idle blocks: polls (at most) for the attention blocks' score pass before stashing; 0 = stash at once  (track B re-tune on the staged engine: 8 / 7, e0205, e0208)
#endif
#endif
#ifndef ATT_WAIT_MAX_B
#define ATT_WAIT_MAX_B 2    // ... with up to this many sequences (measured: B=1 -17 us per step, B=8 +14)
#endif
#ifndef O2PF
#define O2PF 0   // first tile from the stash: L2 prefetch of the rest of the second tile (6 before the L2 staging; with staging 0 is -26 us at B=1, 4 rounds, B=8 +-0)
#endif
#ifndef O_ATT2
#define O_ATT2 0           // o tile table: the attention blocks take two (cold) tiles whenever the other blocks would be left with more than O_ATT2 / 10 tiles each; measured us/step B=1..5: threshold 2.5: 4100 4206 4283 4287 4304, always: 4088 4155 4246 4248 4265
#endif
#ifndef TOUCH_KL
#define TOUCH_KL 0          // attention at long context: K position tiles (x sequences) the idle blocks pull into L2 at the start. Measured with 8 sequences, us per token: prompt 8192 + 256 tokens 8010 -> 7954 with 128, but + 1024 tokens 8177 -> 8202: off
#endif
#ifndef ATT_POLITE
#ifdef TRACK_A
#define ATT_POLITE 5    // (track A: as before)
#else
#define ATT_POLITE 7        // attention, 1-2 sequences: the idle blocks request their first loads of the o stash one at a time (less in flight while the attention blocks read K and V: measured -55 us per step; 2..6 alike, 9 worse)  (track B re-tune on the staged engine: 8 / 7, e0205, e0208)
#endif
#endif
#ifndef ATT_STASH_CAP
#define ATT_STASH_CAP 1000
#endif
#ifndef TOUCH
#define TOUCH 1
#endif
#ifndef ATT_PF
#define ATT_PF 0          // attention blocks prefetch their own K/V history: 1 = behind the prep's loads, 2 = first (both measured slower: +10 / +20 us), 0 = off
#endif
#ifndef KV_PF
#define KV_PF 0          // 1: K/V history prefetched during the q/k/v phase (measured +55 us: it does not stay in L2 that long); 0: pulled by the idle blocks at the start of attention
#endif
#ifndef LONG_NT
#define LONG_NT 72      // position tiles (16 positions) from which attention sums V in 24 sub-segments (and may be split over blocks)
#endif
// Idle blocks' K/V touch at the start of attention. 0: unit by unit, K then V of each (a late unit's K is asked for
// after the V of the early ones). 1: K of all units first, units interleaved, TOUCH_FL chunks in flight per warp.
// Reads only: the logits cannot change. Measured on track A, tests/time_ctx.py 64..1088 positions, us per step:
// B=1 three rounds 5337.2 / 5337.1 / 5338.9 -> 5329.1 / 5326.2 / 5330.1 with FL 1 (FL 2: 5329.6, FL 3: 5332.9, one
// round); B=8 5578.1 -> 5581.8 / 5574.6 / 5575.4 (FL 1 / 2 / 3, inside the spread): on for 1-2 sequences, track A
// (not measured on track B: its build keeps the inherited order).
#ifndef TOUCH_ORD
#ifdef TRACK_A
#define TOUCH_ORD 1
#else
#define TOUCH_ORD 0
#endif
#endif
#ifndef TOUCH_ORD_MAX_B
#define TOUCH_ORD_MAX_B 2 // ... with up to this many sequences
#endif
#ifndef TOUCH_FL
#define TOUCH_FL 1
#endif
#ifndef TOUCH_TILES
#define TOUCH_TILES 256   // attention: position tiles (8 KB of K and V each) the idle blocks pull into L2, for all sequences together / 8
#endif
#ifndef NPF_DN
#define NPF_DN 3      // end of gate+up: loads per warp of the down projection's first tile asked from L2 behind the stashed group
#endif
#ifndef NPF2
#define NPF2 0        // after stashing a tile: loads per warp of the block's second tile asked from L2
#endif
// end-of-phase prefetch of the next phase's first loads: NPF loads (512 B) per warp
#ifndef NPF
#define NPF 2
#endif
#define NLOAD (NWARP - 1)  // warps 0..NLOAD-1 stream weights
#define CW (NWARP - 1)     // coordinator warp: sums the partial accumulators, runs the epilogue, drives the barrier

// tiles x ksteps of each matrix (tile = 16 output rows, kstep = 16 input dims). Stream: 12 bits per weight;
// a group = 4 ksteps = 3 loads of 512 bytes per warp (96 uint4), a tile = ks/4 groups.
#define QKV_T 384
#define O_T 256
#define GU_T 1536
#define DN_T 256
#define LM_T 9496
#define KS_H 256
#define KS_I 768
// offsets inside a layer, in uint4 (16 byte) units
#define OFF_QKV 0ull
#define TU4(ks) ((size_t)(ks) * 8 * LPG)   // uint4 per tile
#define OFF_O (OFF_QKV + (size_t)QKV_T * TU4(KS_H))
#define OFF_GU (OFF_O + (size_t)O_T * TU4(KS_H))
#define OFF_DN (OFF_GU + (size_t)GU_T * TU4(KS_H))
#define LAYER_U4 (OFF_DN + (size_t)DN_T * TU4(KS_I))
#define LAYER_T (QKV_T + O_T + GU_T + DN_T)   // tiles per layer (index into the per-tile table)
#define TT 16                                 // uint4 per tile in the per-tile table
#define INLINE_CHUNKS 3
#ifndef TTPF
#define TTPF 1                                // per-tile table: entries prefetched this many tiles ahead
#endif
// first group of loader warp w when the kg groups of a tile are split over the loader warps
#define G0(w, kg) ((w) * ((kg) / NLOAD) + min((int)(w), (int)((kg) % NLOAD)))

// Shared memory of a block: [misc][partial sums][sx]. sx holds the bf16 input of the current matvec as MMA B
// fragments, [kstep][lane = 4*sequence + t][4 bf16]:
//   narrow (up to 4 sequences): 16 lanes = 128 B per kstep (lanes 16..31 of the MMA read their neighbour's
//                               data; their columns are not used);  wide (5..8 sequences): 32 lanes = 256 B.
//   K = 12288 (down projection): at SX_OFF;   K = 4096: at HF_OFF, behind the scratch region.
// Scratch region [SX_OFF, SX_OFF + 98304): image of the block's first o tile (the stash) / attention scratch
// (which may run on into the K = 4096 fragments: they are dead during attention).
// Size matters: with 200704 B or more of shared memory the SM is left with so little L1 that the decoded
// stream runs 4% slower (tests/bw12.cu: 3124 -> 2997 GB/s). Narrow needs 157.8 KB, wide 223.2 KB.
#define MISC_OFF 0                        // Params copy, tile table, small scratch
#define MISC_BYTES 2048
#define RED_OFF MISC_BYTES
#define RED_HALF (NWARP * 32)             // float4 per buffer (two buffers: tile parity)
#define RED_BYTES (2 * RED_HALF * 16)
#define SX_OFF (RED_OFF + RED_BYTES)
#define SCRATCH_BYTES (KS_H / 4 * LPG * 512) // one K=4096 tile
#define HF_OFF (SX_OFF + SCRATCH_BYTES)
#define DSTASH_BYTES (NLOAD * CAP_D * 512) // first CAP_D loads of every loader warp of a K=12288 tile
#define SX_NARROW (SCRATCH_BYTES + DSTASH_BYTES)
#define TB_OFF (MISC_OFF + 296)           // int[8]: this block's [t0,t1) for qkv, down, gate+up, lm
#define TBO_OFF (TB_OFF + 48)             // int[2]: this block's [t0,t1) for o (depends on the batch size)
#define RN_OFF (MISC_OFF + 352)           // float[8]: 1/rms of the current norm, per sequence
#define SCR_OFF (MISC_OFF + 384)          // float[132]
#define CTL_OFF (MISC_OFF + 1120)         // Ctl: layer, barrier index, generate step
#define SCL_OFF (MISC_OFF + 1152)         // float2[32]: coordinator lanes' row scales of the current tile
#define ARG_OFF (MISC_OFF + 1408)         // float4[32]: coordinator lanes' running argmax (generate)
#define GO_OFF (MISC_OFF + 1920)          // int: the grid barrier of this phase is complete (stop stashing)
#define CNT_OFF (MISC_OFF + 1936)         // int[NWARP]: loads of the next phase's first tile each loader warp has stashed
#define SMEM_MAX (SX_OFF + SX_NARROW)
#define ATT_OFF_BYTES SX_OFF              // attention scratch: 5120 + 32 B per (padded) position
#define STASH_OFF SX_OFF                  // image of the block's first tile of a K=4096 matrix (64 groups x 1536 B)
#define DSTASH_OFF HF_OFF                 // narrow: first group per warp of the block's first K=12288 tile (the K=4096 fragments are dead then)
#define MAX_CTX_SMEM 12288                // attention scratch: 5120 + 16 B per position (see att_bytes)
#define ATT_BYTES(ctx) (5120 + 16 * (((ctx) + 15) / 16 * 16) + 256)

#define BARRIERS_PER_LAYER 5
#define BARRIERS_PER_STEP (1 + NL * BARRIERS_PER_LAYER)
#define BARRIERS_PER_GEN_STEP (2 + NL * BARRIERS_PER_LAYER)

struct Params {
  const uint4* W;             // packed weights: 36 layers (qkv, o, gateup, down), then lm_head
  const uint4* ttab;          // per tile (same order), 16 uint4: {first overflow chunk, chunks, -, -}, {16 row shifts as int8}, 3 inline chunks
  const uint4* exc;           // overflow exception chunks: 8 records of 8 bytes (see build_tables in api.py)
  const unsigned short* embed;// [VOCAB][H] bf16
  const float* norm_w;        // [2*NL+1][H]: input norm l, post-attention norm l, ..., final norm
  const float* qk_norm_w;     // [NL][2][128]
  const float* rope;          // [max_ctx][128]: cos[64], sin[64]
  float* xres;                // [MAXB][H] residual stream, float32
  float* qkv;                 // [MAXB][QKV_ROWS]
  unsigned short* afrag;      // [MAXB][KS_H][4][4] bf16: attention output as MMA B-fragments
  unsigned short* mfrag;      // [MAXB][KS_I][4][4] bf16: silu(gate)*up as MMA B-fragments
  unsigned short* xfrag;      // [MAXB][KS_H][4][4] bf16: w_norm * x (not yet scaled by 1/rms) for the next norm+matvec (track A: the norm's output)
  unsigned short* xraw;       // track A: bf16 residual stream as fragments (the input of the next RMSNorm)
  const unsigned short* wfrag;// track A: norm weights in fragment order, [2*NL+1][2 layouts: compact, wide][H]
  float* ssq;                 // [MAXB][256]: per 16-row tile sum of x^2 (partial sums of the next RMSNorm)
  uint4* kc;                  // K cache, MMA A-fragment layout: [NL][MAXB][8][ctx_tiles][8][32]
  uint4* vc;                  // V cache (transposed), [NL][MAXB][8][ctx_tiles][8][32]
  int* att_done;              // [8*MAXB]: stamp (position, layer) of the last score pass of each attention block (a hint, see att_phase)
  float* asc;                 // split attention: scores [8*MAXB units][4 heads][ctx_pad], written by the K blocks
  unsigned* sync;             // barrier counter
#ifdef TIMING
  unsigned* tstat;            // [blocks][200 phases][8] cycle counts
#endif
  unsigned* wstat;            // calibration: [9 phase kinds][blocks] barrier poll iterations (how long a block waited)
  const int* tiles;           // [blocks][8]: tile ranges per block
  const int* tiles_o;         // [2 * MAXB][blocks][2]: tile ranges of the o projection per batch size (rows MAXB..: split attention)
  const long long* tokens;    // [B]
  float* logits;              // [B][VOCAB]
  long long* gen_out;         // generate(): [B][n_steps] tokens
  float* bm_val;              // generate(): [MAXB][blocks] best logit seen by each block
  int* bm_idx;                // generate(): [MAXB][blocks] its row
  int n_steps;
  int l0build;                // 1: the launch computes layer 0's q/k/v only (building l0tab)
  const float* l0tab;         // track B: [VOCAB][QKV_ROWS] layer 0's q/k/v output per token id (nullptr: computed)
  unsigned* trace;            // TRACE: [blocks][8 phases][8 points]
  int nwork;                  // number of working blocks (TRACE: one less than the grid)
  int stop_layer;             // debug: run only the first stop_layer layers (NL in production)
  int calib;                  // calibration: record barrier waits
  unsigned gen_tag;           // online path: 0 = generate(); else the cohort tag, stored in the upper half of every token
                              // written to gen_out (host-mapped memory), followed by a system-scope fence
  size_t kv_stride;           // uint4 per (layer, seq, kv head)
  unsigned sync_base;
  int sx_bytes;               // size of the sx area of this launch
  int B, pos;
};

#define TSEL_O T_O
enum { T_QKV = 0, T_DN = 1, T_GU = 2, T_LM = 3, T_O = 6 };   // index into the block's tile table (T_O: per batch size)

// TRACE build: the last block is a software clock (a counter in global memory), the coordinator thread of every
// working block timestamps points of the phases of one layer. Diagnosis only; never in a submission.
#ifdef TRACE
#ifndef TRACE_LAYER
#define TRACE_LAYER 17
#endif
#define TR_PARAM , int tr
#define TR_ARG(x) , (x)
#define TP(i) do { if (tr >= 0 && TIDX == CW * 32) P.trace[(blockIdx.x * 8 + tr) * 8 + (i)] = *(volatile unsigned*)(P.trace + 64 * 256); } while (0)
#define NWORK (P.nwork)
#define TRACE_EXTRA 1
#else
#define TRACE_EXTRA 0
#define TR_PARAM
#define TR_ARG(x)
#define TP(i) do { } while (0)
#define NWORK fresh_nblocks()   // (gridDim.x read through asm: hoisted to the kernel start it was spilled)
#endif

// Control state of the block lives in shared memory, not in registers: the kernel's top level only makes
// calls, and a value kept across a call is spilled to local memory by every one of the 1024 threads at every
// phase boundary (measured: +85 us/step). The coordinator thread updates it before a block barrier.
// K/V cache addressing. A unit = (layer, sequence, kv head); a position tile of a unit = 4 KB (16 positions x
// 128 dims, MMA fragment order; V: 8 dim tiles of 512 B). KV_IL = 0: a unit's tiles are contiguous
// ([layer][unit][tile]); KV_IL = 1: tile-major ([layer][tile][unit]): the units, which stream through their
// positions at the same pace, read neighbouring 4 KB blocks instead of addresses one unit-size apart.
#define KVT (KV_IL ? (size_t)MAXB * 8 * 256 : (size_t)256)          // uint4 from one position tile of a unit to the next
#define KVU(P, layer, u) (KV_IL ? ((size_t)(layer) * ((P).kv_stride / 256) * MAXB * 8 + (u)) * 256 : ((size_t)(layer) * MAXB * 8 + (u)) * (P).kv_stride)
struct Ctl { int l; unsigned bi; int s; int lm; };   // layer, barrier index, generate step, lm phase (PH_LM / PH_LMARG)

// TIMING build (diagnosis only): the coordinator thread of every block records cycle counts (clock64(), a per-SM
// counter, ~1830 ticks/us) of the parts of each phase: tstat[block][barrier index][8]. See tests/timing.py.
#ifdef TIMING
#define ATS(i) do { if (TIDX == CW * 32) { const unsigned b_ = ((const Ctl*)(smem + CTL_OFF))->bi; if (b_ < 200u) ((const Params*)(smem + MISC_OFF))->tstat[((size_t)blockIdx.x * 200 + b_) * 8 + 4 + (i)] = (unsigned)clock64(); } } while (0)
#define ATSP(i) do { if (TIDX == CW * 32) { const unsigned b_ = ((const Ctl*)(smem + CTL_OFF))->bi - 1u;   /* split path, behind the first barrier: the phase's own slot */ if (b_ < 200u) ((const Params*)(smem + MISC_OFF))->tstat[((size_t)blockIdx.x * 200 + b_) * 8 + 4 + (i)] = (unsigned)clock64(); } } while (0)
#define TS_DECL long long ts_[8]; int tsn_ = 0; const unsigned tsb_ = ((const Ctl*)(smem + CTL_OFF))->bi
#define TS() do { if (tsn_ < 8) ts_[tsn_++] = clock64(); } while (0)
#define TS_FLUSH(kind) do { if ((TIDX & 31) == 0 && tsb_ < 200u) { unsigned* q_ = P.tstat + ((size_t)blockIdx.x * 200 + tsb_) * 8; q_[0] = (unsigned)(kind); q_[1] = (unsigned)ts_[0]; \
    for (int i_ = 1; i_ < 7; i_++) if (i_ < tsn_) q_[i_ + 1] = (unsigned)(ts_[i_] - ts_[i_ - 1]); } } while (0)
#else
#define ATS(i) do { } while (0)
#define ATSP(i) do { } while (0)
#define TS_DECL
#define TS() do { } while (0)
#define TS_FLUSH(kind) do { } while (0)
#endif

#define MMA(c, a, b) asm volatile( \
  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
  : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) \
  : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y))

// The thread index as a value the compiler cannot merge across the kernel: with plain TIDX it computes
// lane / warp once at the top, keeps them for the whole kernel and, being short of registers in the tile loops,
// parks them in local memory and reloads them at every use.
__device__ __forceinline__ int fresh_tid() {
  int t;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
  return t;
}
#define TIDX fresh_tid()
__device__ __forceinline__ int fresh_bid() {   // the block index, likewise
  int t;
  asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(t));
  return t;
}
#define BIDX fresh_bid()
__device__ __forceinline__ int fresh_nblocks() {   // the grid size, likewise
  int t;
  asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(t));
  return t;
}

#ifdef TRACK_A
#define RB(x) bf2f(f2bf(x))   // a BF16 tensor in the reference model: round the value (it stays in a float register)
#else
#define RB(x) (x)
#endif
// Attention scores s = (q . k) / sqrt(128) from the float32 MMA accumulator c (`scale` is in scope at the use).
//   SCORE_RB 2: bf16(bf16(c) * scale)  - the "eager" formulation of the model code (a BF16 matmul result, then a
//               BF16 multiply): two roundings of a value of magnitude up to ~30 (1 BF16 ulp = 0.125 at 16..32)
//   SCORE_RB 1: bf16(c * scale)        - one rounding
//   SCORE_RB 0: c * scale in float32   - the fused attention kernels (torch SDPA / flash-attention, what the
//               general engines and the Hugging Face default path run): scores never leave the accumulator format
// Measured (tests/long_err.py, 8 x 129 positions 128..1152, relative L2 of the logits against float32; the HF BF16
// model itself: median 0.0119 with its default fused attention, 0.0144 with attn_implementation="eager"):
// SCORE_RB 2: 0.0145 (the residual error follows HF eager layer by layer, 1.17-1.28x HF default from layer 1 on),
// 1: 0.0134, 0: 0.0122 (follows HF default layer by layer, 0.99-1.02x). Track B never rounded the scores.
#ifndef SCORE_RB
#define SCORE_RB 0
#endif
#ifndef SCORE_SCALE
#define SCORE_SCALE 0.08838834764831845f   // 1 / sqrt(128). (Tests only: a value 1-2 float32 ulp away gives an independent draw of every BF16 rounding downstream, tests/long_err.py)
#endif
#if SCORE_RB == 2
#define SCORE_(c) RB(RB(c) * scale)
#elif SCORE_RB == 1
#define SCORE_(c) RB((c) * scale)
#else
#define SCORE_(c) ((c) * scale)
#endif
__device__ __forceinline__ unsigned short f2bf(float f);
__device__ __forceinline__ float bf2f(unsigned v);
#define MMA4(c, a0, a1, a2, a3, b0, b1) asm volatile( \
  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
  : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) \
  : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// Weights are stored with 12 bits each (lossless, see pack_rows in api.py): code = sign << 11 | n << 7 | mantissa,
// bf16 exponent field = EXP0 + n. EXP0 = 0x70, so a bf16 is  sign << 15 | 0b0111 << 11 | low 11 bits of the code:
// the code passes through with a mask and an OR.
#define EXP0 112u
#define NEG_V0 (0x8000u | (EXP0 << 7))
// Exception chunk (see build_tables in api.py): one MMA for 8 weights that the 12-bit code cannot hold. k-slot e
// (and e+8) of the MMA is exception e: A = its exact value in its row (resp. minus the stream's stand-in),
// B = x[col] of this lane's sequence, gathered from the fragments in shared memory. rr = records 2t and 2t+1
// (t = lane & 3), g = lane >> 2, sx16 = shared-memory fragments of the matrix + 16 * g.
// record word 1: row | (t*4 + slot) << 4 | kstep << 8. x[col] of sequence n: narrow from shared memory (sx16 = the
// sequence's fragments, 64 bf16 per kstep), wide from global memory (fg = the matrix's input fragments).
__device__ __forceinline__ unsigned exc_x(unsigned v, const unsigned short* sx16, const unsigned short* fg, int n, bool wide, bool live, unsigned sxs = 0u) {
  const unsigned ks = v >> 8, off = (v >> 4) & 15u;
#if XSLICE
  // from the second tile of a phase on, every loader warp has stored its slice in shared memory (layout
  // [kstep][sequence][16]: sxs = slice base + 16 * sequence): no L2 round trip before the tile's block barrier
  if (!wide && sxs) { unsigned short r = 0; if (live) asm volatile("ld.shared.u16 %0, [%1];" : "=h"(r) : "r"(sxs + (((ks << 6) + off) << 1))); return r; }
  // 1-2 sequences: the input fragments are in shared memory only slice by slice (each loader warp stores its
  // own, behind its first loads), so read the compact global copy ([n][kstep][16 bf16]); sx16 = fg + n * ksteps * 16
  if (!wide) { unsigned r = 0; if (live) r = __ldcg(sx16 + ((size_t)ks << 4) + off); return r; }
#else
  if (!wide) return sx16[(ks << 6) + off];
#endif
  unsigned r = 0;
  if (live) r = __ldcg(fg + ((((size_t)(ks >> 2) * 2 + ((ks & 3u) >> 1)) * 32 + 4 * n + (off >> 2)) << 3) + (ks & 1u) * 4 + (off & 3u));
  return r;
}
__device__ __forceinline__ void exc_mma(float* cex, const uint4 rr, const unsigned short* sx16, const unsigned short* fg, unsigned g, bool wide, bool live, unsigned sxs = 0u) {
  const unsigned b = exc_x(rr.y, sx16, fg, g, wide, live, sxs) | (exc_x(rr.w, sx16, fg, g, wide, live, sxs) << 16);
  const unsigned ro0 = rr.y & 15u, ro1 = rr.w & 15u;
  const unsigned a0 = (ro0 == g ? (rr.x & 0xffffu) : 0u) | (ro1 == g ? (rr.z << 16) : 0u);
  const unsigned a1 = (ro0 == g + 8 ? (rr.x & 0xffffu) : 0u) | (ro1 == g + 8 ? (rr.z << 16) : 0u);
  const unsigned a2 = (ro0 == g ? (rr.x >> 16) : 0u) | (ro1 == g ? (rr.z & 0xffff0000u) : 0u);
  const unsigned a3 = (ro0 == g + 8 ? (rr.x >> 16) : 0u) | (ro1 == g + 8 ? (rr.z & 0xffff0000u) : 0u);
  MMA4(cex, a0, a1, a2, a3, b, b);
}
// one group: 3 loads (12 words) of this lane -> 4 ksteps; xb = this lane's B fragment of the group's first kstep
#ifdef TRACK_A
// one group: 4 loads of this lane = the A registers of 4 MMAs (BF16 weights as stored)
__device__ __forceinline__ void group_mma(float* c, const uint4* A, const uint2* xb, const int S) {
#pragma unroll
  for (int m = 0; m < 4; m++) {
    const uint2 bb = xb[m * S];
    MMA4(c, A[m].x, A[m].y, A[m].z, A[m].w, bb.x, bb.y);
  }
}
// the same with the B fragments of the group in registers (two loads from global memory: ksteps 0,1 and 2,3)
__device__ __forceinline__ void group_mma_g(float* c, const uint4* A, const uint4 X0, const uint4 X1) {
  const unsigned xd[8] = {X0.x, X0.y, X0.z, X0.w, X1.x, X1.y, X1.z, X1.w};
#pragma unroll
  for (int m = 0; m < 4; m++) MMA4(c, A[m].x, A[m].y, A[m].z, A[m].w, xd[2 * m], xd[2 * m + 1]);
}
#else
__device__ __forceinline__ void group_mma(float* c, const uint4* A, const uint2* xb, const int S) {
  const uint4 A0 = A[0], A1 = A[1], A2 = A[2];
  const unsigned wd[12] = {A0.x, A0.y, A0.z, A0.w, A1.x, A1.y, A1.z, A1.w, A2.x, A2.y, A2.z, A2.w};
  const unsigned KK = 0x38003800u, MM = 0x87ff87ffu;
#pragma unroll
  for (int m = 0; m < 4; m++) {
    // 3 words -> the 4 A registers of one MMA: words 0..2 are registers 0..2 with 4 bits of register 3 in place of
    // the constant exponent bits (11 instructions per MMA; the first, denser layout needed 31, and an SM then
    // decoded a tile in 2.3 us instead of 1.6: tests/bw12.cu)
    const unsigned w0 = wd[3 * m], w1 = wd[3 * m + 1], w2 = wd[3 * m + 2];
    const unsigned a0 = (w0 & MM) | KK, a1 = (w1 & MM) | KK, a2 = (w2 & MM) | KK;
    const unsigned a3 = ((w0 >> 11) & 0x000f000fu) | ((w1 >> 7) & 0x00f000f0u) | ((w2 >> 3) & 0x07000700u) | ((w2 << 1) & 0x80008000u) | KK;
    const uint2 bb = xb[m * S];
    MMA4(c, a0, a1, a2, a3, bb.x, bb.y);
  }
}

// the same with the B fragments of the group in registers (two loads from global memory: ksteps 0,1 and 2,3)
__device__ __forceinline__ void group_mma_g(float* c, const uint4* A, const uint4 X0, const uint4 X1) {
  const uint4 A0 = A[0], A1 = A[1], A2 = A[2];
  const unsigned wd[12] = {A0.x, A0.y, A0.z, A0.w, A1.x, A1.y, A1.z, A1.w, A2.x, A2.y, A2.z, A2.w};
  const unsigned xd[8] = {X0.x, X0.y, X0.z, X0.w, X1.x, X1.y, X1.z, X1.w};
  const unsigned KK = 0x38003800u, MM = 0x87ff87ffu;
#pragma unroll
  for (int m = 0; m < 4; m++) {
    const unsigned w0 = wd[3 * m], w1 = wd[3 * m + 1], w2 = wd[3 * m + 2];
    const unsigned a0 = (w0 & MM) | KK, a1 = (w1 & MM) | KK, a2 = (w2 & MM) | KK;
    const unsigned a3 = ((w0 >> 11) & 0x000f000fu) | ((w1 >> 7) & 0x00f000f0u) | ((w2 >> 3) & 0x07000700u) | ((w2 << 1) & 0x80008000u) | KK;
    MMA4(c, a0, a1, a2, a3, xd[2 * m], xd[2 * m + 1]);
  }
}

#endif

__device__ __forceinline__ unsigned short f2bf(float f) {
  unsigned u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}
__device__ __forceinline__ float bf2f(unsigned v) { return __uint_as_float(v << 16); }

// index (in bf16 units) of element k of batch column n inside a B-fragment buffer of `stride` entries per kstep
// smem layout: [kstep][lane = 4n + t][slot]; global compact layout: [n][kstep][t][slot]
__device__ __forceinline__ int frag_slot(int k) { return (k & 1) + 2 * ((k >> 3) & 1); }
__device__ __forceinline__ int frag_t(int k) { return (k & 7) >> 1; }
__device__ __forceinline__ int sx_index(int k, int n) { return (((k >> 4) * 32 + 4 * n + frag_t(k)) << 2) + frag_slot(k); }
__device__ __forceinline__ size_t gfrag_index_c(int k, int n, int ksteps) { return ((((size_t)n * ksteps + (k >> 4)) * 4 + frag_t(k)) << 2) + frag_slot(k); }
// With WIDE_MIN_B or more sequences the fragments (64 - 196 KB) are not copied into shared memory at all: the
// loader warps read them from global memory (L2) group by group next to the weights (tests/bw12.cu mode 2: as
// fast as from shared memory), which saves the input copy of every phase (5 us per layer with 8 sequences) and
// keeps the block's shared memory under the L1 threshold. With few sequences the compact per-sequence layout is
// copied (8 - 24 KB). Same values either way.
#ifndef WIDE_MIN_B
#define WIDE_MIN_B 3
#endif
// wide: [group of 4 ksteps][j = 0, 1][lane = 4n + t][2 ksteps x 4 bf16]: a loader warp reads the B fragments of a
// group with two loads of 512 contiguous bytes, as it reads the weights
__device__ __forceinline__ size_t gfrag_index_w(int k, int n) {
  const int ks = k >> 4;
  return (((((size_t)(ks >> 2) * 2 + ((ks & 3) >> 1)) * 32 + 4 * n + frag_t(k)) << 3) + (ks & 1) * 4) + frag_slot(k);
}
__device__ __forceinline__ size_t gfrag_index(int k, int n, int ksteps, int B) {
  return B >= WIDE_MIN_B ? gfrag_index_w(k, n) : gfrag_index_c(k, n, ksteps);
}

__device__ __forceinline__ float warp_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return v;
}

// ---------------------------------------------------------------------------------------------------------
// entries: put the bf16 input of a matvec into shared memory as MMA B-fragments
// ---------------------------------------------------------------------------------------------------------

// x = embedding row (bf16 -> f32); also publishes the input of the first RMSNorm+matvec:
// fragments of bf16(w * x) and per-tile partial sums of x^2 (the 1/rms factor is applied to matvec OUTPUTS)
// gs < 0: step(): token = tokens[b]. gs >= 0: generate() step gs: token = tokens[b] for gs == 0, otherwise the
// argmax of the previous step's logits (reduced over the per-block maxima, ties -> lowest row, like torch.argmax),
// which is also stored as output token gs-1. embed == 0: only the argmax (after the last step).
__device__ __forceinline__ void op_embed(unsigned char* smem, int gs, int embed) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const int b = BIDX, tid = TIDX, lane = tid & 31;
  if (b < P.B) {
    long long tok;
    if (gs <= 0) {
      tok = P.tokens[b];
      // a token id outside the vocabulary would read past the embedding (and the layer-0 table): use id 0 instead
      // and, for step(), raise the host-mapped error word (P.gen_out, see mk_step) that api.step() checks
      if (tok < 0 || tok >= VOCAB) {
        if (gs < 0 && P.gen_out && tid == 0) { *(volatile long long*)P.gen_out = 1; __threadfence_system(); }
        tok = 0;
      }
    } else {
      int* stok = (int*)(smem + SCR_OFF);
      if (tid < 32) {
        const int nb = NWORK;
        float bv = -__int_as_float(0x7f800000); int bidx = 0x7fffffff;
        for (int i = lane; i < nb; i += 32) {
          float v = __ldcg(P.bm_val + b * nb + i); int ix = __ldcg(P.bm_idx + b * nb + i);
          if (v > bv || (v == bv && ix < bidx)) { bv = v; bidx = ix; }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          float v = __shfl_xor_sync(0xffffffffu, bv, o); int ix = __shfl_xor_sync(0xffffffffu, bidx, o);
          if (v > bv || (v == bv && ix < bidx)) { bv = v; bidx = ix; }
        }
        if (lane == 0) {
          stok[0] = bidx;
          const unsigned tg = P.gen_tag;
          P.gen_out[(size_t)b * P.n_steps + gs - 1] = (long long)(((unsigned long long)tg << 32) | (unsigned)bidx);
          if (tg) __threadfence_system(); else __threadfence();
        }
      }
      __syncthreads();
      tok = stok[0];
    }
    if (embed && P.l0tab && !P.l0build) {
      // track B, layer-0 table: layer 0's input is the embedding row, so its q/k/v output (before the q/k norms and
      // RoPE) depends on the token alone; the row was computed by this kernel's own q/k/v phase at load time
      // (Engine._build_l0), bit for bit what the phase would compute now. The residual stream gets the embedding.
      const float4* src = (const float4*)(P.l0tab + (size_t)tok * QKV_ROWS);
      float4* dst = (float4*)(P.qkv + (size_t)b * QKV_ROWS);
      for (int i = tid; i < QKV_ROWS / 4; i += NTHREAD) dst[i] = __ldcs(src + i);
      const uint2* e = (const uint2*)(P.embed + (size_t)tok * H);
      for (int i = tid; i < H / 4; i += NTHREAD) {
        const uint2 v = e[i];
        ((float4*)(P.xres + (size_t)b * H))[i] = make_float4(bf2f(v.x & 0xffffu), bf2f(v.x >> 16), bf2f(v.y & 0xffffu), bf2f(v.y >> 16));
      }
      __threadfence();
    } else if (embed) {
      const uint2* e = (const uint2*)(P.embed + (size_t)tok * H);
      for (int i = tid; i < H / 4; i += NTHREAD) {      // 4 elements per (virtual) thread i; 4 consecutive i = one 16-row tile
        uint2 v = e[i];
        float4 f = make_float4(bf2f(v.x & 0xffffu), bf2f(v.x >> 16), bf2f(v.y & 0xffffu), bf2f(v.y >> 16));
        ((float4*)(P.xres + (size_t)b * H))[i] = f;
        float4 wv = ((const float4*)P.norm_w)[i];
        const int k = i * 4;
#ifdef TRACK_A
        P.xraw[gfrag_index(k + 0, b, KS_H, P.B)] = (unsigned short)(v.x & 0xffffu);
        P.xraw[gfrag_index(k + 1, b, KS_H, P.B)] = (unsigned short)(v.x >> 16);
        P.xraw[gfrag_index(k + 2, b, KS_H, P.B)] = (unsigned short)(v.y & 0xffffu);
        P.xraw[gfrag_index(k + 3, b, KS_H, P.B)] = (unsigned short)(v.y >> 16);
#else
        P.xfrag[gfrag_index(k + 0, b, KS_H, P.B)] = f2bf(wv.x * f.x);
        P.xfrag[gfrag_index(k + 1, b, KS_H, P.B)] = f2bf(wv.y * f.y);
        P.xfrag[gfrag_index(k + 2, b, KS_H, P.B)] = f2bf(wv.z * f.z);
        P.xfrag[gfrag_index(k + 3, b, KS_H, P.B)] = f2bf(wv.w * f.w);
#endif
        float p = (f.x * f.x + f.y * f.y) + (f.z * f.z + f.w * f.w);
        p += __shfl_xor_sync(0xffffffffu, p, 1);
        p += __shfl_xor_sync(0xffffffffu, p, 2);
        if ((i & 3) == 0) P.ssq[b * 256 + (i >> 2)] = p;
      }
      __threadfence();
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------------------------------------
// One generic phase = (input copy, matvec, epilogue), parameterized at run time, so that the kernel contains ONE
// copy of the hot code for all five matvec phases (code that is not re-executed for a while has to be
// fetched again: the SM's instruction cache is small).
// ---------------------------------------------------------------------------------------------------------
enum { EP_QKV = 0, EP_ADD = 1, EP_GU = 2, EP_LM = 3, EP_LMARG = 4 };
// phase ids (0..4 are also the trace slots)
enum { PH_QKV = 0, PH_ATT = 1, PH_O = 2, PH_GU = 3, PH_DN = 4, PH_LM = 5, PH_LMARG = 6, PH_EMBED = 7, PH_DONE = 8, PH_NORM = 9, PH_SK2 = 10 };

// The current phase as a descriptor in shared memory ("instruction"): the coordinator thread of the block writes
// it before the grid barrier that precedes the phase; every thread then reads the few fields it needs. Deriving
// these values in every thread costs ~100 instructions x 1024 threads per phase, which is real time at 181
// phases per step.
// L2 staging in the attention window (1-2 sequences): HBM idles while the attention blocks work and the idle blocks
// wait at the barrier (stash done). Whole later gate+up tiles of the block are asked into L2 with one bulk prefetch
// per 16 KB piece; when the gate+up phase reaches them they are L2 hits, served next to the HBM stream of the other
// blocks (microbenchmark tests/vl/l2stage2.cu: 151 MB stream with 25 MB staged 55.0 -> 48.3 us). Reads only.
#ifndef STG_GU
#ifdef TRACK_A
#define STG_GU 2          // gate+up tiles per idle block staged into L2 during attention (0: off). Measured (B=1,
                          // 64..1088, 3 rounds): track A 5330 -> 5284 us/step with 1 tile (128 KB); 12 pieces (1.5 tiles) a further -22 (e0131)
#else
#define STG_GU 2          // track B: 4120 -> 4033 with 9 pieces of 16 KB (1.5 tiles of 98 KB); 1 tile 4051, 2: 4039
#endif
#endif
#ifndef STG_O
#define STG_O 0           // 1: also the rest of the block's second o tile
#endif
#ifndef STG_POL
#define STG_POL 0         // 1: L2::evict_last hint on the staging (measured ~25 us slower than none), 0: none
#endif
#ifndef STG_MAX_B
#define STG_MAX_B 2       // with up to this many sequences
#endif
#ifndef STG_TAIL
// end of o / gate+up / down phases (1-2 sequences): 16 KB pieces of the next phase's middle tile staged into L2 by
// one thread per piece, behind the loaders' stash, while the grid barrier runs. Measured (B=1, 64..1088, 3 rounds):
// track A 5287 -> 5163 us with 4 (2: 5215, 6: 5162); track B 4027 -> 3950 with 3 (4: 3965); B=8 not used
#ifdef TRACK_A
#define STG_TAIL 4
#else
#define STG_TAIL 3
#endif
#endif
#ifndef STG_TAIL_MAX_B
#define STG_TAIL_MAX_B 8  // tail staging with up to this many sequences
#endif
#ifndef STG_TAIL_W
#define STG_TAIL_W 3      // ... pieces with 3+ sequences (B=8, 64..1088, 2 rounds: track B 4419 -> 4320 with 3, 4: 4350; track A 5578 -> 5481)
#endif
#ifndef STG_GU_A
#define STG_GU_A 0        // > 0: attention-window staging targets gate+up tiles STG_GU_A + j * STG_GU_S (0: spread over the block's tiles)
#endif
#ifndef STG_GU_S
#define STG_GU_S 1
#endif
#ifndef STG_TAIL_D
#define STG_TAIL_D STG_TAIL   // pieces at the end of down (next: q/k/v of the next layer, or the head), 1-2 sequences
#endif
#ifndef STG_TAIL_GU_TI
#ifdef TRACK_A
#define STG_TAIL_GU_TI 0  // (track A: 2 is +8.5 at B=1, 1 is +2.5 / B=8 -34; e0196)
#else
#define STG_TAIL_GU_TI 2  // > 0: the tail staging after o targets gate+up tile STG_TAIL_GU_TI of the block (0: the middle one). 2: B=1 -20 (6 rounds), B=8 -8 (e0193)
#endif
#endif
#ifndef STG_TAIL_FIRST
#define STG_TAIL_FIRST 0  // bit per transition (as STG_TAIL_MASK): stage the next phase's FIRST tile (from piece STG_TAIL_FOFF) instead of its middle one
#endif
#ifndef STG_TAIL_FOFF
#define STG_TAIL_FOFF 0
#endif
#ifndef STG_TAIL_MASK
#define STG_TAIL_MASK 7   // bit 0: after o (gate+up), 1: after gate+up (down), 2: after down (next q/k/v)
#endif
#ifndef EXC_SMEM
#define EXC_SMEM 1        // 1-2 sequences: exception x values from the loaders' shared-memory slices from the 2nd tile on (B=1 -40, B=2 -58 us, 4 rounds, e0151; bit-identical)
#endif
#ifndef LM_NEXT_O
#define LM_NEXT_O 0       // 1: with the layer-0 table, the end-of-head prefetch targets layer 0's o tiles (not q/k/v)
#endif
#ifndef STG_EARLY
#define STG_EARLY 0       // 1: attention-window staging requested before the o stash instead of after it
#endif
#ifndef GU_SHIFT
#define GU_SHIFT 0       // gate+up tiles moved from each attention block (0..7) to blocks NB_MAT.. (1-2 sequences)
#endif
#ifndef STG_DN
#define STG_DN 0
#endif
#ifndef STG_DN_TI
#define STG_DN_TI 0
#endif
#ifndef STG_DN_OFF
#define STG_DN_OFF 0
#endif
#ifndef STG_GUA
#define STG_GUA 0         // 1: idle block 8B + j also stages the gate+up pieces of attention block j
#endif
#ifndef STG_OA
#define STG_OA 0          // 1: idle blocks stage the attention blocks' o tiles into L2 at the start of attention
#endif
#ifndef STG_NP
#ifdef TRACK_A
#define STG_NP 12         // > 0: only the first STG_NP 16 KB pieces of the STG_GU tiles
#else
#define STG_NP 9
#endif
#endif
__device__ __forceinline__ void l2_stage(const uint4* p, unsigned bytes) {
#if STG_POL
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" :: "l"(p), "r"(bytes), "l"(pol) : "memory");
#else
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p), "r"(bytes) : "memory");
#endif
}
struct Desc {
  const uint4* W;               // weights of the matrix
  const unsigned short* frag;   // input fragments (global memory)
  const float* aux;             // ADD epilogue: weights of the next RMSNorm
  int t0, t1;                   // this block's tiles
  int ks, ep;                   // ksteps of the matrix (256 / 768); epilogue type
  int norm;                     // input is a norm: outputs are scaled by 1/rms
  int k, l;                     // phase id, layer
  const uint4* nextW;           // first tile of this block in the NEXT matvec phase (warp 0, lane 0); prefetch target
  const uint4* tt;              // per-tile table of the matrix (TT uint4 per tile)
  const uint4* nextT;           // table entry of this block's first tile in the NEXT matvec phase (prefetch target)
  int nb, ne;                   // NEXT matrix: groups per loader warp and number of warps with one more (nb 0: no prefetch)
  const uint4* nextP;           // where the coordinator's wait-time prefetch for the next phase starts (warp 0)
  const unsigned short* wfrag;  // track A, norm phases: the norm's weights in the order of the fragments
  int kafter;                   // track A: phase that follows the PH_NORM phase
  int pre;                      // this phase: 1 = the block's first tile is in the stash, 2 = the first group of every warp
  int nst;                      // next phase: what the loaders stash at the end of this one (0: only a prefetch, 1, 2 as above)
  const uint4* stgW;            // STG_TAIL: tile of the next phase staged into L2 at the end of this one (nullptr: none)
};
#define DESC_OFF (MISC_OFF + 912)
static_assert(DESC_OFF + sizeof(Desc) <= CTL_OFF && sizeof(Params) <= 296 && TBO_OFF + 8 <= RN_OFF && CTL_OFF + sizeof(Ctl) <= SCL_OFF && SCL_OFF + 256 <= ARG_OFF && ARG_OFF + 512 <= GO_OFF && CNT_OFF + 4 * NWARP <= MISC_OFF + MISC_BYTES && SX_OFF + SX_NARROW < 199680 && CAP_D <= LPG && DSTASH_BYTES >= KS_H * 16 * 8 + 128 && SX_OFF + ATT_BYTES(MAX_CTX_SMEM) <= 232448 && ATT_BYTES(8000) <= SX_NARROW, "shared-memory layout");
__device__ __forceinline__ void set_desc(unsigned char* smem, int k, int l) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const int* tb = (const int*)(smem + TB_OFF);
  Desc d;
  const uint4* WL = P.W + (size_t)(k >= PH_LM ? NL : l) * LAYER_U4;
  int tsel = T_QKV;
  const uint4* TL = P.ttab + (size_t)(k >= PH_LM ? NL : l) * LAYER_T * TT;
  d.W = WL; d.tt = TL; d.frag = P.xfrag; d.aux = nullptr; d.ks = KS_H; d.ep = EP_QKV; d.norm = 1; d.k = k; d.l = l;
  if (k == PH_O) { d.W = WL + OFF_O; d.tt = TL + TT * QKV_T; d.frag = P.afrag; d.aux = P.norm_w + (size_t)(2 * l + 1) * H; tsel = T_O; d.ep = EP_ADD; d.norm = 0; }
  else if (k == PH_GU) { d.W = WL + OFF_GU; d.tt = TL + TT * (QKV_T + O_T); tsel = T_GU; d.ep = EP_GU; }
  else if (k == PH_DN) { d.W = WL + OFF_DN; d.tt = TL + TT * (QKV_T + O_T + GU_T); d.frag = P.mfrag; d.aux = P.norm_w + (size_t)(2 * l + 2) * H; tsel = T_DN; d.ks = KS_I; d.ep = EP_ADD; d.norm = 0; }
  else if (k == PH_LM) { tsel = T_LM; d.ep = EP_LM; }
  else if (k == PH_LMARG) { tsel = T_LM; d.ep = EP_LMARG; }
  d.t0 = tb[2 * tsel]; d.t1 = tb[2 * tsel + 1];
  // the phase after this one (for the prefetch at the end of this phase)
  d.nextW = nullptr; d.nextT = nullptr;
  int nextks = 0;
  if (k == PH_QKV) { d.nextT = TL + (size_t)(QKV_T + ((const int*)(smem + TBO_OFF))[0]) * TT; }   // o follows attention (no weight prefetch here)
  else if (k == PH_O) { d.nextW = WL + OFF_GU + (size_t)tb[2 * T_GU] * TU4(KS_H); d.nextT = TL + (size_t)(QKV_T + O_T + tb[2 * T_GU]) * TT; nextks = KS_H; }
  else if (k == PH_GU) { d.nextW = WL + OFF_DN + (size_t)tb[2 * T_DN] * TU4(KS_I); d.nextT = TL + (size_t)(QKV_T + O_T + GU_T + tb[2 * T_DN]) * TT; nextks = KS_I; }
  else if (k == PH_DN) {
    if (l + 1 < P.stop_layer) { d.nextW = WL + LAYER_U4 + OFF_QKV + (size_t)tb[2 * T_QKV] * TU4(KS_H); d.nextT = TL + (size_t)(LAYER_T + tb[2 * T_QKV]) * TT; }
    else { d.nextW = P.W + (size_t)NL * LAYER_U4 + (size_t)tb[2 * T_LM] * TU4(KS_H); d.nextT = P.ttab + ((size_t)NL * LAYER_T + tb[2 * T_LM]) * TT; }
    nextks = KS_H;
  } else if (k == PH_LMARG || (k == PH_LM && LM_NEXT)) {
    if (LM_NEXT_O && P.l0tab && !P.l0build) {   // layer-0 table: the next step's first weights are layer 0's o tiles
      const int to0 = ((const int*)(smem + TBO_OFF))[0];
      d.nextW = P.W + OFF_O + (size_t)to0 * TU4(KS_H); d.nextT = P.ttab + (size_t)(QKV_T + to0) * TT;
    } else { d.nextW = P.W + OFF_QKV + (size_t)tb[2 * T_QKV] * TU4(KS_H); d.nextT = P.ttab + (size_t)tb[2 * T_QKV] * TT; }
    nextks = KS_H;
  }
  d.nb = (nextks / 4) / NLOAD; d.ne = (nextks / 4) % NLOAD;
  d.stgW = nullptr;
#if STG_TAIL > 0
  if (d.nextW && P.B <= STG_TAIL_MAX_B) {
    const int tsn = k == PH_O ? T_GU : k == PH_GU ? T_DN : k == PH_DN ? (l + 1 < P.stop_layer ? T_QKV : T_LM) : -1;
    if (tsn >= 0 && (STG_TAIL_MASK >> (k == PH_O ? 0 : k == PH_GU ? 1 : 2) & 1)) {
      const int kb = k == PH_O ? 0 : k == PH_GU ? 1 : 2;
      const int ng = tb[2 * tsn + 1] - tb[2 * tsn], ti = (STG_TAIL_FIRST >> kb & 1) ? 0 : (kb == 0 && STG_TAIL_GU_TI > 0 ? (STG_TAIL_GU_TI < ng ? STG_TAIL_GU_TI : ng - 1) : ng / 2);
      if ((ti > 0 || (STG_TAIL_FIRST >> kb & 1)) && ti < ng) d.stgW = d.nextW + (size_t)ti * TU4(nextks) + ((STG_TAIL_FIRST >> kb & 1) ? (size_t)STG_TAIL_FOFF * 1024 : 0);
    }
  }
#endif
  // Stash: the first tile of a K=4096 phase (the first group per warp of a K=12288 tile) is
  // streamed into shared memory by the loaders at the end of the phase before it (o: during attention; layer 0:
  // during the embedding), so HBM does not idle while the last epilogue and the grid barrier run.
  const bool narrow = P.B < WIDE_MIN_B;
  d.pre = STASH_ALL ? (k == PH_DN ? 2 : 1) : (k == PH_O ? 1 : 0);
  d.nst = !STASH_ALL || k >= PH_LM || k == PH_QKV ? 0 : (k == PH_GU ? 2 : 1);
  d.nextP = d.nextW ? d.nextW + (d.nst == 1 ? ((narrow ? CAP_H : CAP_HW) >= 3 * LPG || !WPF_REST ? TU4(KS_H) : (size_t)(narrow ? CAP_H : CAP_HW) * 32) : d.nst == 2 ? (size_t)(CAP_D + NPF_DN) * 32 : (size_t)NPF * 32) : nullptr;
  d.wfrag = nullptr; d.kafter = k;
#ifdef TRACK_A
  if (d.norm) {
    // RMSNorm before the matvec. 1-2 sequences: the input copy normalises (it reads the raw fragments);
    // 3+: a phase of its own writes the normalised fragments to global memory, where the loaders read them.
    const int id = k >= PH_LM ? 2 * NL : (k == PH_GU ? 2 * l + 1 : 2 * l);
    d.wfrag = P.wfrag + ((size_t)id * 2 + (narrow ? 0 : 1)) * H;
    if (narrow || NORM_LOAD) d.frag = P.xraw; else d.k = PH_NORM;
  }
#endif
  *(Desc*)(smem + DESC_OFF) = d;
}
#ifndef TRACK_A
// k_gensk, the last layer's down phase: what its tail fetches for the next phase (set_desc: the 12-bit head's
// first tile into the stash, its middle tile into L2, the coordinator's wait-time prefetch) is never read by the
// sketch. Instead: no stash, no prefetch, and the tail staging asks L2 for the start of the block's first int8
// sketch tile (STG_TAIL pieces of 16 KB of its 64 KB), which pass 1 reads first.
#ifndef SK_DN_FIX
#define SK_DN_FIX 1   // k_gensk: the last down phase's tail fetches the sketch's first tile instead of the 12-bit head's (sk_dn_desc)
#endif
struct SkArgs;
__device__ __forceinline__ const uint4* sk_w8_of(const Params& P);
__device__ __forceinline__ void sk_dn_desc(unsigned char* smem) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const int* tb = (const int*)(smem + TB_OFF);
  Desc& d = *(Desc*)(smem + DESC_OFF);
  d.nst = 0; d.nb = 0; d.ne = 0; d.nextW = nullptr; d.nextP = nullptr; d.nextT = nullptr;
  d.stgW = nullptr;
#if STG_TAIL > 0
  if (P.B <= STG_TAIL_MAX_B && tb[2 * T_LM] < tb[2 * T_LM + 1]) d.stgW = sk_w8_of(P) + (size_t)tb[2 * T_LM] * 128 * 32;
#endif
}
#endif
#ifdef TRACE
#define TR_OF(D) int tr = ((D).l == TRACE_LAYER && (D).k < PH_LM) ? (D).k : -100
#else
#define TR_OF(D)
#endif

__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

// epilogue of one 16-row tile, run by the coordinator warp. Lane (g, t) holds rows g and g+8 of batch columns
// 2t and 2t+1: c.x (row g, n0), c.y (row g, n1), c.z (row g+8, n0), c.w (row g+8, n1).
// r0, r1: 1/rms of the input norm for the two columns (QKV, GU, LM). ADD: w0, w1 = weight of the NEXT RMSNorm for
// the two rows, xin = residual rows; the residual add publishes that norm's input (bf16(w*x) and sum of x^2).
// what the residual add publishes for the next RMSNorm + matvec: track B the fused input bf16(w * x), track A x itself
#ifdef TRACK_A
#define NORM_IN(idx, wv, x) P.xraw[idx] = f2bf(x)
#else
#define NORM_IN(idx, wv, x) P.xfrag[idx] = f2bf((wv) * (x))
#endif
__device__ __forceinline__ void epilogue(const Params& P, int ep, int tile, int lane, float4 c, float r0, float r1, float w0, float w1, float4 xin) {
  const int g = lane >> 2, t = lane & 3, B = P.B;
  const int n0 = 2 * t, n1 = 2 * t + 1;
  const int row = tile * 16 + g;
  if (ep == EP_ADD) {
    float s0 = 0.f, s1 = 0.f;
    if (n0 < B) {
      float* p = P.xres + (size_t)n0 * H + row;
      float a = RB(xin.x + RB(c.x)), b = RB(xin.y + RB(c.z));
      p[0] = a; p[8] = b;
      NORM_IN(gfrag_index(row, n0, KS_H, B), w0, a);
      NORM_IN(gfrag_index(row + 8, n0, KS_H, B), w1, b);
      s0 = a * a + b * b;
    }
    if (n1 < B) {
      float* p = P.xres + (size_t)n1 * H + row;
      float a = RB(xin.z + RB(c.y)), b = RB(xin.w + RB(c.w));
      p[0] = a; p[8] = b;
      NORM_IN(gfrag_index(row, n1, KS_H, B), w0, a);
      NORM_IN(gfrag_index(row + 8, n1, KS_H, B), w1, b);
      s1 = a * a + b * b;
    }
    // sum of x^2 over the 16 rows of the tile: lanes with the same t hold the 8 row pairs
    s0 += __shfl_xor_sync(0xffffffffu, s0, 4); s1 += __shfl_xor_sync(0xffffffffu, s1, 4);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 8); s1 += __shfl_xor_sync(0xffffffffu, s1, 8);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 16); s1 += __shfl_xor_sync(0xffffffffu, s1, 16);
    if (g == 0) {
      if (n0 < B) P.ssq[n0 * 256 + tile] = s0;
      if (n1 < B) P.ssq[n1 * 256 + tile] = s1;
    }
  } else if (ep == EP_GU) {
    const int j = tile * 8 + g;     // rows 0..7 of the tile are gate[j], rows 8..15 are up[j]
    if (n0 < B) { float gt = RB(c.x * r0); float m = RB(gt / (1.0f + __expf(-gt))) * RB(c.z * r0); P.mfrag[gfrag_index(j, n0, KS_I, B)] = f2bf(m); }
    if (n1 < B) { float gt = RB(c.y * r1); float m = RB(gt / (1.0f + __expf(-gt))) * RB(c.w * r1); P.mfrag[gfrag_index(j, n1, KS_I, B)] = f2bf(m); }
  } else {
    // QKV: [B][6144] ; LM: logits [B][VOCAB]
    float* out = ep == EP_QKV ? P.qkv : P.logits;
    const size_t stride = ep == EP_QKV ? QKV_ROWS : VOCAB;
#ifdef NOLMSTORE
    if (ep == EP_QKV || tile == 0) {
#endif
    if (n0 < B) { out[(size_t)n0 * stride + row] = RB(c.x * r0); out[(size_t)n0 * stride + row + 8] = RB(c.z * r0); }
    if (n1 < B) { out[(size_t)n1 * stride + row] = RB(c.y * r1); out[(size_t)n1 * stride + row + 8] = RB(c.w * r1); }
#ifdef NOLMSTORE
    }
#endif
  }
}

// grid-wide barrier: every block adds one to a counter, then waits until all have (132 blocks: 0.54 us when all
// arrive together; a per-block flag variant measured slower). Run by the coordinator thread only; every
// operator ends with its global stores fenced and a block barrier.
__device__ __forceinline__ void grid_barrier(unsigned char* smem, int kind TR_PARAM) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  Ctl& C = *(Ctl*)(smem + CTL_OFF);
  const unsigned idx = C.bi;
  atomicAdd(P.sync, 1u);
  TP(4);
  unsigned target = P.sync_base + (idx + 1u) * NWORK;
  unsigned polls = 0;
  while ((int)(*(volatile unsigned*)P.sync - target) < 0) polls++;
  TP(5);
  if (P.calib) P.wstat[kind * NWORK + BIDX] += polls;   // how long this block waited for the others
  C.bi = idx + 1u;
}

// 1/rms of sequence n from the per-tile partial sums (fixed order, the same wherever it is computed). Whole warp.
__device__ __forceinline__ float rms_from(const float4 a, const float4 b) {   // whole warp: lane's 8 partials
  float v = ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w));
  v = warp_sum(v);
  return 1.0f / sqrtf(v * (1.0f / H) + 1e-6f);
}
__device__ __forceinline__ float rms_inv(const Params& P, int n, int lane) {
  const float4* q = (const float4*)(P.ssq + n * 256) + lane * 2;
  return rms_from(__ldcg(q), __ldcg(q + 1));
}
#ifdef TRACK_A
// RMSNorm of 8 elements: y = bf16(w * bf16(x * r)), x and w as bf16 fragments
__device__ __forceinline__ unsigned pack_bf2(float lo, float hi) {   // = f2bf(lo) | f2bf(hi) << 16 (round to nearest even)
  unsigned d; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d;
}
__device__ __forceinline__ unsigned norm2(unsigned x, unsigned w, float r) {
  const unsigned t = pack_bf2(__uint_as_float(x << 16) * r, __uint_as_float(x & 0xffff0000u) * r);
  return pack_bf2(__uint_as_float(w << 16) * __uint_as_float(t << 16), __uint_as_float(w & 0xffff0000u) * __uint_as_float(t & 0xffff0000u));
}
__device__ __forceinline__ uint4 norm8(uint4 x, uint4 w, float r) {
  return make_uint4(norm2(x.x, w.x, r), norm2(x.y, w.y, r), norm2(x.z, w.z, r), norm2(x.w, w.w, r));
}
#endif

// input copy: sx <- fragments written to global memory by the previous phase ([n][kstep][4 lanes][4 bf16]).
// norm phases: the fragments are bf16(w * x); also compute 1/rms per sequence from the per-tile partial sums
// (fixed order, the same in every block); the matvec epilogue multiplies its outputs by it.
__device__ __forceinline__ void entry_copy(unsigned char* smem) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const Desc& D = *(const Desc*)(smem + DESC_OFF);
  const int ks = D.ks, per = ks * 2, tot = per * P.B;
  uint4* sx4 = (uint4*)(smem + (ks == KS_I ? SX_OFF : HF_OFF));
  const uint4* g = (const uint4*)D.frag;
  // Flat index over (sequence, entry) so that the loads are spread over all threads of the block. Every
  // instruction here is executed by up to 1024 threads at each of the 181 phases, so: threads without work leave
  // at once, a single load takes the short path, and several loads of one thread are requested ECB at a time
  // before the first store (a load->store chain makes every load a full round trip of its own).
  const int i0 = TIDX;
#ifdef TRACK_A
  if (D.norm) {
    if (P.B < WIDE_MIN_B) {
      // RMSNorm while copying. 1/rms by one warp per sequence (every warp computing it for itself made the copy
      // 4 us slower: 132 x 24 warps asking for the same 8 lines), then a block barrier. The copy's own loads are
      // requested before that barrier: they travel while the 1/rms is computed.
      const uint4* wf = (const uint4*)D.wfrag;
      unsigned long long xa = 0, xb = 0, wa = 0, wb = 0;
      const int n0 = i0 >> 9, r0 = i0 - n0 * per;
      if (i0 < tot) {
        // (loaded as 64-bit words: as uint4 structs the compiler kept them in local memory - an L2 round trip each)
        xa = __ldcg((const unsigned long long*)(g + i0)); xb = __ldcg((const unsigned long long*)(g + i0) + 1);
        wa = __ldcg((const unsigned long long*)(wf + r0)); wb = __ldcg((const unsigned long long*)(wf + r0) + 1);
      }
      {
        const int n = (i0 >> 5) - NORM_WARP0, lane = i0 & 31;
        if (n >= 0 && n < P.B) {
          const float r = rms_inv(P, n, lane);
          if (lane == 0) ((float*)(smem + RN_OFF))[n] = r;
        }
      }
      __syncthreads();
      const float* rr = (const float*)(smem + RN_OFF);
      for (int j = i0; j < tot; j += NTHREAD) {
        const int n = j >> 9, r = j - n * per;
        if (j != i0) {
          xa = __ldcg((const unsigned long long*)(g + j)); xb = __ldcg((const unsigned long long*)(g + j) + 1);
          wa = __ldcg((const unsigned long long*)(wf + r)); wb = __ldcg((const unsigned long long*)(wf + r) + 1);
        }
        const float rn_ = rr[n];
        const unsigned y0 = norm2((unsigned)xa, (unsigned)wa, rn_), y1 = norm2((unsigned)(xa >> 32), (unsigned)(wa >> 32), rn_);
        const unsigned y2 = norm2((unsigned)xb, (unsigned)wb, rn_), y3 = norm2((unsigned)(xb >> 32), (unsigned)(wb >> 32), rn_);
        sx4[(r >> 1) * 8 + 2 * n + (r & 1)] = make_uint4(y0, y1, y2, y3);
      }
    } else if (NORM_LOAD) {
      // 3+ sequences: 1/rms per sequence (one warp each) and the norm weights (wide fragment order, 8 KB) into
      // shared memory; the loaders normalise the fragments they read (tile_loop_g)
      if (i0 < KS_H * 2) ((uint4*)(smem + NORM_WOFF))[i0] = __ldcg((const uint4*)D.wfrag + i0);
      if (i0 + NTHREAD < KS_H * 2) ((uint4*)(smem + NORM_WOFF))[i0 + NTHREAD] = __ldcg((const uint4*)D.wfrag + i0 + NTHREAD);
      const int n = (i0 >> 5) - NORM_WARP0, lane = i0 & 31;
      if (n >= 0 && n < P.B) {
        const float r = rms_inv(P, n, lane);
        if (lane == 0) ((float*)(smem + RN_OFF))[n] = r;
      }
#if NORM_NB
      return;   // no block barrier: the loaders meet at a barrier of their own behind their first loads (tile_loop_g)
#endif
    }
    __syncthreads();
    return;
  }
#endif
  if (P.B < WIDE_MIN_B && !XSLICE) {
   if (i0 < tot) {
    if (i0 + NTHREAD >= tot) {
      const int n = ks == KS_H ? (i0 >> 9) : (i0 / (2 * KS_I)), r = i0 - n * per;
      sx4[(r >> 1) * 8 + 2 * n + (r & 1)] = __ldcg(g + i0);
    } else {
      for (int i = i0; i < tot; i += ECB * NTHREAD) {
        uint4 v[ECB];
#pragma unroll
        for (int u = 0; u < ECB; u++) { v[u] = make_uint4(0, 0, 0, 0); if (i + u * NTHREAD < tot) v[u] = __ldcg(g + i + u * NTHREAD); }
#pragma unroll
        for (int u = 0; u < ECB; u++) {
          const int j = i + u * NTHREAD;
          if (j < tot) {
            const int n = ks == KS_H ? (j >> 9) : (j / (2 * KS_I)), r = j - n * per;
            sx4[(r >> 1) * 8 + 2 * n + (r & 1)] = v[u];
          }
        }
      }
    }
   }
  }
  if (D.norm) {
#if COORD_RMS
    // 1/rms per sequence by the coordinator warp, the only one that uses it (epilogue, behind the first tile's
    // barrier): the loader warps start streaming at once. Loads of 4 sequences at a time, then the same
    // arithmetic as rms_inv.
    if ((TIDX >> 5) == CW) {
      const int lane = TIDX & 31;
      for (int n0 = 0; n0 < P.B; n0 += 4) {
        float4 a[4], b[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
          a[k] = make_float4(0.f, 0.f, 0.f, 0.f); b[k] = a[k];
          if (n0 + k < P.B) { const float4* q = (const float4*)(P.ssq + (n0 + k) * 256) + lane * 2; a[k] = __ldcg(q); b[k] = __ldcg(q + 1); }
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
          const float r = rms_from(a[k], b[k]);
          if (lane == 0 && n0 + k < P.B) ((float*)(smem + RN_OFF))[n0 + k] = r;
        }
      }
    }
#else
    // warps NORM_WARP0.. (no copy loads with few sequences): one warp per sequence
    const int n = (TIDX >> 5) - NORM_WARP0, lane = TIDX & 31;
    if (n >= 0 && n < P.B) {
      const float r = rms_inv(P, n, lane);
      if (lane == 0) ((float*)(smem + RN_OFF))[n] = r;
    }
#endif
  }
  // No barrier where nothing was copied (XSLICE with 1-2 sequences; 3+ sequences read the input from L2): the
  // coordinator reads 1/rms only behind the first tile's barrier.
  if (P.B < WIDE_MIN_B ? !XSLICE : !ENTRY_NB) __syncthreads();
}

// Matvec, block team. The block owns tiles [t0, t1). Loader warp w streams ksteps [s0, s0 + BASE (+1 for the first
// EXTRA warps)) of every tile and leaves its partial accumulator in shared memory; one block barrier per tile.
// Loaders and the coordinator run DIFFERENT functions: in one function the registers of the coordinator's
// epilogue push the loaders' weights into local memory, and with 229 KB of shared memory the SM has almost
// no L1 left, so every spill is an L2 round trip queued behind the warp's own weight loads (+900..2000 us/step).
// Weight loads carry the cache operator "last use" (ld.global.lu): a weight line is read exactly once per step,
// and with the default operator the stream evicts what the phases share through L2 (input fragments, residual,
// K/V). Measured, generate us/step B=1 / B=8: default (ca) 5649 / 6184, cg 5618 / 6128, cs 5637 / 6100, lu 5602 / 6047.
#ifdef SKEL   // diagnosis only (wrong results): no weight stream, the skeleton of the step
#define WLOAD(ptr) make_uint4(0u, 0u, 0u, 0u)
#endif
#ifndef WLOAD
#define WLOAD(ptr) __ldlu(ptr)
#endif
// chunks prefetched per poll while a block waits at the grid barrier after a matvec phase (0: off), and how many
// ksteps deep per loader warp at most
#ifndef WPF
#define WPF 8
#endif
#ifndef WPF_SMALL_B
#define WPF_SMALL_B 2
#endif
#ifndef WPF_DEPTH
#define WPF_DEPTH 4
#endif
#ifndef NPF_ATT
#define NPF_ATT 3   // attention blocks, end of attention: loads (512 B per warp) of their o tile
#endif
#ifndef NPF_ST
#define NPF_ST 4    // stash blocks, during attention: ksteps of their second o tile
#endif
#ifndef NPF_GU
#define NPF_GU 0     // stash blocks, during attention: loads of their first gate+up tile asked from L2 (measured: 2 / 4 / 9 loads: +15 / +38 / +49 us per step at B=1)
#endif
#ifndef NPF_STW
#define NPF_STW 9   // ... with 6 or more sequences (measured with 8: 4314 -> 4303 us per step; with 3-5: +6..+14; with 1: +26)
#endif
#ifndef NPF_EMB
#define NPF_EMB 4   // embedding phase: ksteps of the first q/k/v tile
#endif
// `prefetch.global.L2` asks L2 for a line without producing a value, so the warp never waits for it (the
// disassembler shows it as a NOP, but it is effective: 1 / 2 / 3 / 4 / 8 ksteps at the end of a phase give
// 5478 / 5449 / 5455 / 5476 / 5612 us against 5520 without; a load into a dead register is removed by the compiler)
#ifndef PREFETCH
#define PREFETCH(ptr) asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr))
#endif
#ifndef GB
#define GB (LPG == 3 ? 3 : 2)     // groups (3 loads each) requested back to back before the first MMA of a batch
#endif
template <int KS, int S, int LITE = 0>   // LITE (k_gensk only): register-lighter variant of the first tile, same arithmetic; S: lanes per kstep of the fragments in shared memory (16 narrow, 32 wide)
__device__ __forceinline__ void tile_loop(unsigned char* smem, const uint4* W, int t, int t1, bool pre, bool pre1) {
  enum { KG = KS / 4, BASE = KG / NLOAD, EXTRA = KG % NLOAD, NGW = BASE + 1, TU = KG * GU4 };
  const int lane = TIDX & 31, w = TIDX >> 5;
  const int g0 = w * BASE + min(w, (int)EXTRA);
  const bool extra = w < EXTRA;                          // this warp has NGW groups, the others BASE
  const uint2* xb = (const uint2*)(smem + (KS == KS_I ? SX_OFF : HF_OFF)) + g0 * (4 * S) + lane;
  float4* redw = (float4*)(smem + RED_OFF) + w * 32 + lane;
  const uint4* p = W + ((size_t)t * KG + g0) * GU4 + lane;
#if XSLICE
  // The warp's slice of the input fragments (its ksteps, all sequences: compact global layout [n][kstep][16 bf16]
  // -> shared [kstep][n][16 bf16]) is requested first and stored behind the first weight loads.
  enum { XPL = (NGW * 4 * 2 * 2 + 31) / 32 };          // uint4 per lane (at most 2 sequences)
#ifdef TRACK_A
  const bool xs_on = !((const Desc*)(smem + DESC_OFF))->norm;   // (norm phases: the block-wide normalising copy)
#else
  const bool xs_on = true;
#endif
  const int xper = (extra ? NGW : BASE) * 8, xtot = xs_on ? xper * ((const Params*)(smem + MISC_OFF))->B : 0;
  uint4 xr[XPL];
  {
    const uint4* gx = (const uint4*)((const Desc*)(smem + DESC_OFF))->frag;
#pragma unroll
    for (int u = 0; u < XPL; u++) {
      const int i = lane + 32 * u, n = i >= xper ? 1 : 0, r = i - n * xper;
      xr[u] = make_uint4(0, 0, 0, 0);
      if (i < xtot) xr[u] = __ldcg(gx + (size_t)n * (KS * 2) + g0 * 8 + r);
    }
  }
#define XS_STORE_() do { uint4* sxw_ = (uint4*)(smem + (KS == KS_I ? SX_OFF : HF_OFF)); \
    _Pragma("unroll") for (int u = 0; u < XPL; u++) { const int i = lane + 32 * u, n = i >= xper ? 1 : 0, r = i - n * xper; \
      if (i < xtot) sxw_[(g0 * 4 + (r >> 1)) * 8 + 2 * n + (r & 1)] = xr[u]; } \
    __syncwarp(); } while (0)
#else
#define XS_STORE_() do { } while (0)
#endif
  // Software pipeline across the block barrier: the first group (3 loads) of the NEXT tile is requested before
  // the barrier, so the warp's load queue is not empty while the block synchronizes.
  uint4 nx[LPG];
  if (pre && t < t1) {   // (a block may own no tile of a phase: NB_MAT)
    // First tile: the warp's first `cnt` loads are in the stash (same layout as a tile), the rest is requested
    // now, and with it the first group of the second tile.
    const int cnt = ((const int*)(smem + CNT_OFF))[w];
    const uint4* q = (const uint4*)(smem + STASH_OFF) + g0 * GU4 + lane;
#ifdef TRACK_A
    // (12 loads of 16 B and the next tile's 4 do not fit in the registers: GB groups at a time)
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < NGW; j += GB) {
      uint4 a[GB * LPG];
#pragma unroll
      for (int u = 0; u < GB * LPG; u++) {
        const int gi = j + u / LPG, idx = j * LPG + u;
        if (gi > BASE) continue;
        a[u] = make_uint4(0, 0, 0, 0);
        if (gi < BASE || extra) { if (idx < cnt) a[u] = q[idx * 32]; else a[u] = WLOAD(p + idx * 32); }
      }
      if (j == 0) XS_STORE_();
#pragma unroll
      for (int g = 0; g < GB; g++) {
        const int gi = j + g;
        if (gi > BASE) continue;
        if (gi < BASE || extra) group_mma(c, &a[g * LPG], xb + gi * (4 * S), S);
      }
    }
    if (t + 1 < t1) {
#pragma unroll
      for (int u = 0; u < LPG; u++) nx[u] = WLOAD(p + TU + u * 32);
    }
#else
    uint4 a[NGW * LPG];
#pragma unroll
    for (int u = 0; u < BASE * LPG; u++) { if (u < cnt) a[u] = q[u * 32]; else a[u] = WLOAD(p + u * 32); }
    if (extra) {
#pragma unroll
      for (int u = BASE * LPG; u < NGW * LPG; u++) {
        if (LITE & 2) { if (u >= cnt) PREFETCH(p + u * 32); }   // (LITE 2: loaded behind the base groups' MMAs)
        else { if (u < cnt) a[u] = q[u * 32]; else a[u] = WLOAD(p + u * 32); }
      }
    }
    if (t + 1 < t1 && !(LITE & 4)) {
#pragma unroll
      for (int u = 0; u < LPG; u++) nx[u] = WLOAD(p + TU + u * 32);
#if O2PF > 0
      // ... and L2 is asked for the rest of the second tile: it arrives while the stashed tile is multiplied
#pragma unroll
      for (int u = LPG; u < LPG + O2PF; u++) if (!(LITE & 1) && (u < BASE * LPG || extra)) PREFETCH(p + TU + u * 32);
#endif
    }
    XS_STORE_();
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int g = 0; g < BASE; g++) group_mma(c, &a[g * LPG], xb + g * (4 * S), S);
    if ((LITE & 4) && t + 1 < t1) {   // (LITE 4: the next tile's first group requested behind the MMAs)
#pragma unroll
      for (int u = 0; u < LPG; u++) nx[u] = WLOAD(p + TU + u * 32);
    }
    if (extra) {
      if (LITE & 2) {
#pragma unroll
        for (int u = BASE * LPG; u < NGW * LPG; u++) { if (u < cnt) a[u] = q[u * 32]; else a[u] = WLOAD(p + u * 32); }
      }
      group_mma(c, &a[BASE * LPG], xb + BASE * (4 * S), S);
    }
#endif
    redw[(t & 1) * RED_HALF] = make_float4(c[0], c[1], c[2], c[3]);
    t++; p += TU;
    __syncthreads();
  } else if (pre1) {
    // the warp's first `cnt` loads (at most one group) of the first tile are in shared memory
    const int cnt = ((const int*)(smem + CNT_OFF))[w];
    const uint4* q = (const uint4*)(smem + DSTASH_OFF) + w * (CAP_D * 32) + lane;
#pragma unroll
    for (int u = 0; u < LPG; u++) { if (u < cnt) nx[u] = q[u * 32]; else nx[u] = WLOAD(p + u * 32); }
    XS_STORE_();
  } else if (t < t1) {
#pragma unroll
    for (int u = 0; u < LPG; u++) nx[u] = WLOAD(p + u * 32);
    XS_STORE_();
  }
#undef XS_STORE_
  for (; t < t1; t++, p += TU) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < NGW; j += GB) {
      uint4 a[GB * LPG];
#pragma unroll
      for (int u = 0; u < GB * LPG; u++) {
        const int gi = j + u / LPG;
        if (gi > BASE) continue;
        a[u] = make_uint4(0, 0, 0, 0);          // (an array assigned under a condition would live in local memory)
        if (gi == 0) a[u] = nx[u];
        else if (gi < BASE || extra) a[u] = WLOAD(p + (j * LPG + u) * 32);
      }
#pragma unroll
      for (int g = 0; g < GB; g++) {
        const int gi = j + g;
        if (gi > BASE) continue;
        if (gi < BASE || extra) group_mma(c, &a[g * LPG], xb + gi * (4 * S), S);
      }
    }
    redw[(t & 1) * RED_HALF] = make_float4(c[0], c[1], c[2], c[3]);
    if (t + 1 < t1) {
      const uint4* pn = p + TU;
#pragma unroll
      for (int u = 0; u < LPG; u++) nx[u] = WLOAD(pn + u * 32);
    }
    __syncthreads();
  }
}

// Stream this loader warp's share of a tile (tile = the first load of warp 0, lane 0; KG groups per tile) into
// shared memory at st0, SG loads at a time, at most `cap` loads, and only until the coordinator reports that the
// grid barrier is complete (`adaptive`): what is in flight on this SM delays everything else the SM asks for,
// including the coordinator's look at the barrier counter. The number of loads stashed is left in CNT[warp].
__device__ __forceinline__ void stash_loads(unsigned char* smem, const uint4* tile, int KG, int stash_off, bool per_warp, int cap, bool adaptive, int polite = 0) {
  const int tid = TIDX, lane = tid & 31, w = tid >> 5;
  const int base = KG / NLOAD, ex = KG % NLOAD;
  const int g0 = w * base + min(w, ex);
  const int my = min(cap, LPG * (base + (w < ex ? 1 : 0)));
  const uint4* p = tile + g0 * GU4 + lane;
  uint4* st = (uint4*)(smem + stash_off) + (per_warp ? w * (CAP_D * 32) : g0 * GU4) + lane;
  const volatile int* go = (const volatile int*)(smem + GO_OFF);
  int u = 0;
  for (; u < polite && u < my; u++) {      // the first loads one at a time (half the traffic in flight)
    const uint4 a = WLOAD(p + u * 32);
    st[u * 32] = a;
  }
  for (; u + SG <= my; u += SG) {
    if (adaptive && *go) break;
    uint4 a[SG];
#pragma unroll
    for (int i = 0; i < SG; i++) a[i] = WLOAD(p + (u + i) * 32);
#pragma unroll
    for (int i = 0; i < SG; i++) st[(u + i) * 32] = a[i];
  }
  for (; u < my; u++) {
    if (adaptive && *go) break;
    const uint4 a = WLOAD(p + u * 32);
    st[u * 32] = a;
  }
  if (lane == 0) ((int*)(smem + CNT_OFF))[w] = u;
}
// The loader loop for 5 to 8 sequences: B fragments from global memory (two loads per group next to the three
// weight loads), GBW groups requested back to back. The first `cnt` loads of the warp's first tile are in shared
// memory (stash; K=12288: at most the first group).
#ifndef GBW
#define GBW 2
#endif
template <int KS>
__device__ __forceinline__ void tile_loop_g(unsigned char* smem, const uint4* W, const uint4* X, int t, int t1, bool pre, bool nrm = false) {
  enum { KG = KS / 4, BASE = KG / NLOAD, EXTRA = KG % NLOAD, NGW = BASE + 1, TU = KG * GU4 };
  const int lane = TIDX & 31, w = TIDX >> 5;
  const int g0 = w * BASE + min(w, (int)EXTRA);
  const bool extra = w < EXTRA;
  const bool xl = lane < 4 * ((const Params*)(smem + MISC_OFF))->B;      // lanes of unused sequences load nothing
  const uint4* xg = X + g0 * 64 + lane;
  float4* redw = (float4*)(smem + RED_OFF) + w * 32 + lane;
  const uint4* p = W + ((size_t)t * KG + g0) * GU4 + lane;
  const uint4* q = (const uint4*)(smem + (KS == KS_I ? DSTASH_OFF : STASH_OFF)) + (KS == KS_I ? w * (CAP_D * 32) : g0 * GU4) + lane;
  int cnt = pre ? ((const int*)(smem + CNT_OFF))[w] : 0;
  // K=4096: the input fragments of the warp's groups (track A norm phases: normalised), as read in the first
  // tile, are kept in the warp's own part of the stash image (group gi: the first 1 KB of its slot, already
  // consumed) for the block's other tiles, instead of being read from L2 again (and normalised again)
  uint4* xs = (uint4*)(smem + STASH_OFF) + g0 * GU4 + lane;
  const int tf = t;
#if defined(TRACK_A) && NORM_LOAD
  const bool keep = KS == KS_H && (XS_ALL || (NORM_SX && nrm));
#else
  const bool keep = KS == KS_H && XS_ALL;
#endif
  uint4 nx[LPG];
  if (t < t1) {
#pragma unroll
    for (int u = 0; u < LPG; u++) { if (u < cnt) nx[u] = q[u * 32]; else nx[u] = WLOAD(p + u * 32); }
  }
#if defined(TRACK_A) && NORM_LOAD && NORM_NB
  if (nrm) asm volatile("bar.sync 1, %0;" :: "r"(NLOAD * 32) : "memory");   // norm weights and 1/rms are in shared memory
#endif
  for (; t < t1; t++, p += TU) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < NGW; j += GBW) {
      uint4 a[GBW * LPG], xa[GBW * 2];
#pragma unroll
      for (int g = 0; g < GBW; g++) {
        const int gi = j + g;
        if (gi > BASE) continue;
#pragma unroll
        for (int u = 0; u < LPG; u++) a[g * LPG + u] = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int i = 0; i < 2; i++) xa[g * 2 + i] = make_uint4(0, 0, 0, 0);
        if (gi < BASE || extra) {
#pragma unroll
          for (int i = 0; i < 2; i++) {
            if (xl) xa[g * 2 + i] = keep && t != tf ? xs[gi * GU4 + i * 32] : __ldcg(xg + gi * 64 + i * 32);
          }
#pragma unroll
          for (int u = 0; u < LPG; u++) {
            const int idx = gi * LPG + u;
            if (gi == 0) a[g * LPG + u] = nx[u];
            else if (KS != KS_I && idx < cnt) a[g * LPG + u] = q[idx * 32];
            else a[g * LPG + u] = WLOAD(p + idx * 32);
          }
        }
      }
#pragma unroll
      for (int g = 0; g < GBW; g++) {
        const int gi = j + g;
        if (gi > BASE) continue;
        if (gi < BASE || extra) {
          if (xl && (nrm || keep) && t == tf) {
#if defined(TRACK_A) && NORM_LOAD
            if (nrm) {   // y = bf16(w * bf16(x / rms)), as the norm phase / the narrow input copy compute it
              const uint4* wv = (const uint4*)(smem + NORM_WOFF) + (g0 + gi) * 8 + (lane & 3);
              const float r = ((const float*)(smem + RN_OFF))[lane >> 2];
              xa[g * 2] = norm8(xa[g * 2], wv[0], r); xa[g * 2 + 1] = norm8(xa[g * 2 + 1], wv[4], r);
            }
#endif
            // kept for the block's further tiles in this warp's part of the stash image (already consumed)
            if (keep) { xs[gi * GU4] = xa[g * 2]; xs[gi * GU4 + 32] = xa[g * 2 + 1]; }
          }
          group_mma_g(c, &a[g * LPG], xa[g * 2], xa[g * 2 + 1]);
        }
      }
    }
    redw[(t & 1) * RED_HALF] = make_float4(c[0], c[1], c[2], c[3]);
    cnt = 0;
    if (t + 1 < t1) {
      const uint4* pn = p + TU;
#pragma unroll
      for (int u = 0; u < LPG; u++) nx[u] = WLOAD(pn + u * 32);
    }
    __syncthreads();
  }
}

// End of a loader warp's phase. HBM would idle from here until the next phase has copied its input (last
// epilogue, fence, grid barrier, input copy: ~2 us, and the first tile then starts cold). So the loaders stream
// the block's first tile of the next phase into shared memory now; it needs no input. Where there is no room
// (K=12288 tiles: one group per warp, narrow layout) or no next phase in this launch, L2 is asked for the first
// loads instead (`prefetch` does not block the warp).
__device__ __forceinline__ void loader_tail(unsigned char* smem) {
  // The descriptor is read before the barrier below: the coordinator replaces it with the next phase's after that
  // barrier. The barrier itself keeps the loaders' loads out of the SM's memory pipeline while the coordinator
  // stores and fences the last epilogue, which is the critical path of the whole grid (stashing right away made
  // that epilogue 1.8 us slower).
  const volatile Desc& D = *(const volatile Desc*)(smem + DESC_OFF);
  const uint4* const nextW = D.nextW;
  const int nst = D.nst, nb = D.nb, ne = D.ne;
#if STG_TAIL > 0
  const uint4* const stgW = D.stgW;
  const int stgk = D.k;
#endif
#if FENCE_BAR
  __syncthreads();
#endif
  if (nb) {
    if (nst == 1) stash_loads(smem, nextW, KS_H / 4, STASH_OFF, false, ((const Params*)(smem + MISC_OFF))->B >= WIDE_MIN_B ? CAP_HW : CAP_H, ADAPT);
    else if (nst == 2) stash_loads(smem, nextW, KS_I / 4, DSTASH_OFF, true, CAP_D, ADAPT);
    else {
      const int tid = fresh_tid(), lane = tid & 31, w = tid >> 5;
      const uint4* p = nextW + (w * nb + min(w, ne)) * GU4 + lane;   // this warp's first group there
#pragma unroll
      for (int u = 0; u < NPF; u++) PREFETCH(p + u * 32);
    }
  }
#if STG_TAIL > 0
  if (stgW) { const int tid = fresh_tid(); if (tid < (((const Params*)(smem + MISC_OFF))->B <= 2 ? (stgk == PH_DN ? STG_TAIL_D : STG_TAIL) : STG_TAIL_W)) l2_stage(stgW + tid * 1024, 16384); }
#endif
  __syncthreads();
}
// a loader warp's phase: input copy, tile loop, then wait for the coordinator (last epilogue + grid barrier)
template <int LITE = 0>
__device__ __forceinline__ void loader_phase(unsigned char* smem) {
  entry_copy(smem);
  const Desc& D = *(const Desc*)(smem + DESC_OFF);
  // the start of the block's first tile is in the stash
  const bool pre = D.pre == 1;   // (o: the blocks that ran attention have stashed nothing)
  if (((const Params*)(smem + MISC_OFF))->B >= WIDE_MIN_B) {
#if defined(TRACK_A) && NORM_LOAD
    if (D.ks == KS_H) tile_loop_g<KS_H>(smem, D.W, (const uint4*)D.frag, D.t0, D.t1, pre, D.norm);
#else
    if (D.ks == KS_H) tile_loop_g<KS_H>(smem, D.W, (const uint4*)D.frag, D.t0, D.t1, pre);
#endif
    else tile_loop_g<KS_I>(smem, D.W, (const uint4*)D.frag, D.t0, D.t1, D.pre == 2);
  } else {
    if (D.ks == KS_H) tile_loop<KS_H, 16, LITE>(smem, D.W, D.t0, D.t1, pre, false);
    else tile_loop<KS_I, 16, LITE>(smem, D.W, D.t0, D.t1, false, D.pre == 2);
  }
  loader_tail(smem);
}

// The coordinator warp's phase: input copy; per tile: request the epilogue inputs (they arrive while the
// loaders stream), meet the loaders at the block barrier, sum the 31 partials in a fixed order (tree), run the
// epilogue; then advance the control state, write the next phase's descriptor and run the grid barrier.
template <bool SK = false>
__device__ __forceinline__ void coord_phase(unsigned char* smem) {
  TS_DECL;
  TS();                                   // 0: phase start (block barrier of the previous phase released)
  entry_copy(smem);
  TS();                                   // 1: input copy done
  Params& P = *(Params*)(smem + MISC_OFF);
#ifdef DESC_COPY
  const Desc D = *(const Desc*)(smem + DESC_OFF);
#else
  const volatile Desc& D = *(const volatile Desc*)(smem + DESC_OFF);   // read where used: a register copy gets spilled
#endif
  TR_OF(D);
  TP(0);
  const int ep = D.ep;
#if PREP_PF
  if (D.k == PH_QKV) {
    // The block that will append this step's K and V rows asks L2 for the position tile they go to (4 KB each)
    // and for what else its prep reads once per step (q/k norm weights, RoPE row): the appended rows are stores
    // into lines last touched a step ago, and the fence behind them waited for those lines to come from HBM.
    const int b = BIDX, lane = TIDX & 31;
    int u = -1;
    if (SPLIT_ON(P.B, P.pos)) { const int kb = SPLIT_KB(P.B, NWORK); if (b < 8 * P.B * kb && b % kb == kb - 1) u = b / kb; }
    else if (b < 8 * P.B) u = b;
    if (u >= 0) {
      const size_t base = KVU(P, D.l, u) + (size_t)(P.pos >> 4) * KVT + lane * 8;
      PREFETCH(P.kc + base); PREFETCH(P.vc + base);
      if (lane < 8) PREFETCH(P.qk_norm_w + (size_t)D.l * 256 + lane * 32);
      else if (lane < 12) PREFETCH(P.rope + (size_t)P.pos * 128 + (lane - 8) * 32);
    }
#if KV_PF
    // The other blocks ask L2 for the layer's K/V history now, 4 KB per instruction, while the q/k/v weights
    // stream: attention then starts on L2 hits instead of an HBM round trip (as long as all of it fits in L2).
    const int nt = (P.pos + 16) >> 4;
    if (b >= 8 * P.B && !SPLIT_ON(P.B, P.pos) && nt * P.B <= TOUCH_TILES) {
      const int total = 16 * P.B * nt, nidle = NWORK - 8 * P.B;
      for (int c = b - 8 * P.B; c < total; c += nidle) {
        const int uu = c / (2 * nt), i = c - uu * 2 * nt;
        PREFETCH((i < nt ? P.kc : P.vc) + KVU(P, D.l, uu) + (size_t)(i < nt ? i : i - nt) * KVT + lane * 8);
      }
    }
#endif
  }
#endif
#ifdef TRACK_A
  {
    // The weights of the next RMSNorm (8 KB, used once per step, so not in L2): ask for them a phase ahead; the
    // normalising input copy of the next phase would otherwise start with an HBM round trip in every block.
    const int k = D.k, l = D.l;
    if (k == PH_O || k == PH_DN || k >= PH_LM) {       // (lm_head: layer 0's first norm of the next step)
      const int id = k == PH_O ? 2 * l + 1 : k == PH_DN ? 2 * l + 2 : 0;
      const uint4* wf = (const uint4*)(P.wfrag + ((size_t)id * 2 + (P.B < WIDE_MIN_B ? 0 : 1)) * H) + (TIDX & 31) * 8;
      PREFETCH(wf); PREFETCH(wf + 256);
    }
  }
#endif
  {
    const int lane = TIDX & 31;
    // EP_LMARG: running argmax of this lane's rows, kept in shared memory (in registers it is spilled on every tile):
    // {best value n0, best value n1, row n0 (int bits), row n1}
    if (ep == EP_LMARG) ((float4*)(smem + ARG_OFF))[lane] = make_float4(-__int_as_float(0x7f800000), -__int_as_float(0x7f800000), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
  }
  for (int t = D.t0; t < D.t1; t++) {
    const int lane = TIDX & 31;                 // per tile: a value kept over the loop would be spilled
    const float4* red = (const float4*)(smem + RED_OFF) + lane;
    float4* const am = (float4*)(smem + ARG_OFF) + lane;
    // per-tile table entry (two cache lines at a known address): row shifts and the tile's first exception chunks.
    // The entry of the next tile (or, at the first tile, of this block's first tile in the next phase) is
    // requested from L2 now: under the weight flood a cold load takes several microseconds.
    const uint4* te = D.tt + (size_t)t * TT;
#ifdef NOTAB
    const uint4 te0 = make_uint4(0, 0, 0, 0), te1 = te0;
    uint4 ri[INLINE_CHUNKS];
#pragma unroll
    for (int c = 0; c < INLINE_CHUNKS; c++) ri[c] = te0;
#else
    const uint4 te0 = __ldcg(te), te1 = __ldcg(te + 1);
    uint4 ri[INLINE_CHUNKS];
#pragma unroll
    for (int c = 0; c < INLINE_CHUNKS; c++) ri[c] = __ldcg(te + 2 + c * 4 + (lane & 3));
#ifndef NOTTPF
    if (t + 1 < D.t1) PREFETCH(te + TT * TTPF + (lane >> 4) * 8);
#else
    if (t + 1 < D.t1) { }
#endif
#ifndef NEXTT_FIRST
    else if (D.nextT) PREFETCH(D.nextT + (lane >> 4) * 8);        // last tile: the next phase's first entry
#else
    if (t == D.t0 && D.nextT) PREFETCH(D.nextT + (lane >> 4) * 8);
#endif
#endif
    float w0 = 0.f, w1 = 0.f;
    float4 xin = make_float4(0.f, 0.f, 0.f, 0.f);
    if (ep == EP_ADD) {
      const int row = t * 16 + (lane >> 2), n0 = 2 * (lane & 3), n1 = n0 + 1;
      w0 = (*(D.aux + row)); w1 = (*(D.aux + row + 8));
      if (n0 < P.B) { const float* q = P.xres + (size_t)n0 * H + row; xin.x = __ldcg(q); xin.y = __ldcg(q + 8); }
      if (n1 < P.B) { const float* q = P.xres + (size_t)n1 * H + row; xin.z = __ldcg(q); xin.w = __ldcg(q + 8); }
    }
    // 2^-shift of this lane's two rows (the stream holds row * 2^shift): to shared memory at once, so that the
    // table entry's second half does not occupy registers during the exception MMAs
    {
      const unsigned sw0 = (lane >> 4) ? te1.y : te1.x, sw1 = (lane >> 4) ? te1.w : te1.z;
      const int sbit = ((lane >> 2) & 3) * 8;
      ((float2*)(smem + SCL_OFF))[lane] = make_float2(__uint_as_float((unsigned)(127 - (int)(signed char)(sw0 >> sbit)) << 23),
                                                      __uint_as_float((unsigned)(127 - (int)(signed char)(sw1 >> sbit)) << 23));
    }
    // Exceptions: weights outside the 16 binades of the code are stored as code 0 (= +2^(EXP0-127)) in the
    // stream; one extra MMA per 8 of them adds value * x and subtracts the stand-in.
    float cex[4] = {0.f, 0.f, 0.f, 0.f};
    {
#if XSLICE
      const unsigned short* sx16 = D.frag + (size_t)(lane >> 2) * D.ks * 16;
#else
      const unsigned short* sx16 = (const unsigned short*)(smem + (D.ks == KS_I ? SX_OFF : HF_OFF)) + 16 * (lane >> 2);
#endif
      const bool wide = P.B >= WIDE_MIN_B, live = (lane >> 2) < P.B;
      const unsigned short* fg = D.frag;
      const unsigned g = lane >> 2;
#if XSLICE && EXC_SMEM
      // (a 32-bit shared-memory address: a 64-bit pointer here made k_step / k_generate spill)
      // only in k_gensk (the greedy/online kernel): in k_step / k_generate this path made the coordinator spill (+69 us)
      const unsigned sxs = (SK || EXC_SMEM > 1) && t > D.t0 ? (unsigned)__cvta_generic_to_shared(smem + (D.ks == KS_I ? SX_OFF : HF_OFF)) + 32u * g : 0u;
#else
      const unsigned sxs = 0u;
#endif
#ifdef NOEXC
      const int n = 0;
#elif defined(NOEXC_LM)
      const int n = ep >= EP_LM ? 0 : (int)te0.y;
#elif defined(NOEXC_LAYERS)
      const int n = ep >= EP_LM ? (int)te0.y : 0;
#else
      const int n = (int)te0.y;
#endif
#pragma unroll
      for (int c = 0; c < INLINE_CHUNKS; c++) if (c < n) exc_mma(cex, ri[c], sx16, fg, g, wide, live, sxs);
      if (n > INLINE_CHUNKS) {
        const uint4* ex = P.exc + (size_t)te0.x * 4 + (lane & 3);
        PREFETCH(P.exc + (size_t)(te0.x + n - INLINE_CHUNKS) * 4 + (lane >> 3) * 8);   // the next tile's overflow chunks follow
        for (int i = 0; i < n - INLINE_CHUNKS; i += 2) {
          const uint4 ra = __ldcg(ex + i * 4), rb = __ldcg(ex + i * 4 + 4);
          exc_mma(cex, ra, sx16, fg, g, wide, live, sxs);
          if (i + 1 < n - INLINE_CHUNKS) exc_mma(cex, rb, sx16, fg, g, wide, live, sxs);
        }
      }
    }
    // the exception partial waits in shared memory (the coordinator's own slot of the partial-sum buffer)
    ((float4*)(smem + RED_OFF) + (t & 1) * RED_HALF + NLOAD * 32 + lane)[0] = make_float4(cex[0], cex[1], cex[2], cex[3]);
    __syncthreads();
    if (t == D.t0) TS();                  // 2: first tile done
    if (t == D.t1 - 1) TS();              // 3: last tile's barrier passed
    if (t == D.t0) TP(1);
    if (t == D.t1 - 1) TP(2);
    const float4* r = red + (t & 1) * RED_HALF;
#define R_(i) r[(i) * 32]
#define S8_(o) add4(add4(add4(R_(o), R_(o + 1)), add4(R_(o + 2), R_(o + 3))), add4(add4(R_(o + 4), R_(o + 5)), add4(R_(o + 6), R_(o + 7))))
#if NLOAD == 31
    float4 q0 = S8_(0), q1 = S8_(8), q2 = S8_(16);
    float4 q3 = add4(add4(add4(R_(24), R_(25)), add4(R_(26), R_(27))), add4(add4(R_(28), R_(29)), R_(30)));
    float4 c = add4(add4(q0, q1), add4(q2, q3));
#elif NLOAD == 15
    float4 q0 = S8_(0);
    float4 q1 = add4(add4(add4(R_(8), R_(9)), add4(R_(10), R_(11))), add4(add4(R_(12), R_(13)), R_(14)));
    float4 c = add4(q0, q1);
#elif NLOAD == 23
    float4 q0 = S8_(0), q1 = S8_(8);
    float4 q2 = add4(add4(add4(R_(16), R_(17)), add4(R_(18), R_(19))), add4(add4(R_(20), R_(21)), R_(22)));
    float4 c = add4(add4(q0, q1), q2);
#else
#error "reduction tree for this NLOAD"
#endif
    {
      const float4 cx = R_(NLOAD);
      const float2 sc = ((const float2*)(smem + SCL_OFF))[lane];
      c.x = (c.x + cx.x) * sc.x; c.y = (c.y + cx.y) * sc.x; c.z = (c.z + cx.z) * sc.y; c.w = (c.w + cx.w) * sc.y;
    }
#undef S8_
#undef R_
    const float* rn = (const float*)(smem + RN_OFF);
#ifdef TRACK_A
    const float r0 = 1.0f, r1 = 1.0f;             // the norm was applied to the input
#else
    const float r0 = rn[2 * (lane & 3)], r1 = rn[2 * (lane & 3) + 1];
#endif
    if (ep == EP_LMARG) {
      // same values as the logits of step(): c * r; rows ascend, so strict > keeps the lowest row among equals
      const int row = t * 16 + (lane >> 2);
      const float4 a4 = *am;
      float bv0 = a4.x, bv1 = a4.y;
      int bi0 = __float_as_int(a4.z), bi1 = __float_as_int(a4.w);
      float v;
      v = RB(c.x * r0); if (v > bv0) { bv0 = v; bi0 = row; }
      v = RB(c.z * r0); if (v > bv0) { bv0 = v; bi0 = row + 8; }
      v = RB(c.y * r1); if (v > bv1) { bv1 = v; bi1 = row; }
      v = RB(c.w * r1); if (v > bv1) { bv1 = v; bi1 = row + 8; }
      *am = make_float4(bv0, bv1, __int_as_float(bi0), __int_as_float(bi1));
      if (t == D.t1 - 1) {
#pragma unroll
        for (int o = 16; o >= 4; o >>= 1) {
          float u = __shfl_xor_sync(0xffffffffu, bv0, o); int ix = __shfl_xor_sync(0xffffffffu, bi0, o);
          if (u > bv0 || (u == bv0 && ix < bi0)) { bv0 = u; bi0 = ix; }
          u = __shfl_xor_sync(0xffffffffu, bv1, o); ix = __shfl_xor_sync(0xffffffffu, bi1, o);
          if (u > bv1 || (u == bv1 && ix < bi1)) { bv1 = u; bi1 = ix; }
        }
        if (lane < 4) {
          const int n0 = 2 * lane, n1 = 2 * lane + 1, nb = NWORK, b = BIDX;
          if (n0 < P.B) { P.bm_val[n0 * nb + b] = bv0; P.bm_idx[n0 * nb + b] = bi0; }
          if (n1 < P.B) { P.bm_val[n1 * nb + b] = bv1; P.bm_idx[n1 * nb + b] = bi1; }
        }
      }
    } else {
      epilogue(P, ep, t, lane, c, r0, r1, w0, w1, xin);
    }
    if (t == D.t1 - 1) { __threadfence(); TP(3); TS(); }   // 4: last epilogue stored and fenced
  }
  const int lane = TIDX & 31;
#if EARLY_COUNT
  // The block's epilogues are fenced: count it in at once. The grid waits for the last block's increment, and
  // everything below (block barrier, next descriptor) concerns only this block.
  // If this block is the last one of the grid to arrive, its loaders stash nothing (GO is set at once): what they
  // would have in flight is what the whole grid would then wait for.
  int go0 = 0;
  if (lane == 0 && ((const volatile Desc*)(smem + DESC_OFF))->k != PH_LM) {
    const unsigned old = atomicAdd(P.sync, 1u);
#if LAST_NO_STASH
    go0 = (int)(P.sync_base + (((const Ctl*)(smem + CTL_OFF))->bi + 1u) * NWORK - old) <= LAST_NO_STASH;
#endif
  }
  if (lane == 0) *(volatile int*)(smem + GO_OFF) = go0;
#else
  if (lane == 0) *(volatile int*)(smem + GO_OFF) = 0;
#endif
#if FENCE_BAR
  __syncthreads();   // releases the loaders into their end-of-phase stash (see loader_tail); the descriptor may change now
#endif
  // control: next phase
  // re-read from shared memory (kept in registers over the tile loop these are spilled), before the descriptor changes
  const int kk = ((const volatile Desc*)(smem + DESC_OFF))->k;
  const uint4* const nW = ((const volatile Desc*)(smem + DESC_OFF))->nextP;
  const int nb = ((const volatile Desc*)(smem + DESC_OFF))->nb, ne = ((const volatile Desc*)(smem + DESC_OFF))->ne;
  if (lane == 0) {
    Ctl& C = *(Ctl*)(smem + CTL_OFF);
    const int k = kk, l = ((const volatile Desc*)(smem + DESC_OFF))->l;
    if (k == PH_QKV) { if (P.l0build) ((Desc*)(smem + DESC_OFF))->k = PH_DONE; else set_desc(smem, PH_O, l); }   // (l0build: layer 0's q/k/v only)
    else if (k == PH_O) set_desc(smem, PH_GU, l);
    else if (k == PH_GU) {
      set_desc(smem, PH_DN, l);
#ifndef TRACK_A
      if (SK && SK_DN_FIX && l + 1 == P.stop_layer && P.logits) sk_dn_desc(smem);   // k_gensk: the head is read through the sketch
#endif
    }
    else if (k == PH_DN) { const bool more = l + 1 < P.stop_layer; set_desc(smem, more ? PH_QKV : C.lm, more ? l + 1 : 0); }
    else if (k == PH_LMARG) { C.s = C.s + 1; P.pos = P.pos + 1; ((Desc*)(smem + DESC_OFF))->k = PH_EMBED; }   // generate(): next step
    else ((Desc*)(smem + DESC_OFF))->k = PH_DONE;                                                           // step(): logits are out
  }
  // Grid barrier run by the whole coordinator warp: while this block waits for the others, every poll also
  // prefetches WPF more 512-byte chunks of the block's first tile of the next phase, one kstep deeper for all
  // 31 loader warps in turn (the loaders prefetched ksteps 0 and 1). A block that waits longer prefetches more;
  // the block everybody waits for prefetches nothing.
  if (kk != PH_LM) {
    Ctl& C = *(Ctl*)(smem + CTL_OFF);
    const unsigned idx = C.bi;
#if !EARLY_COUNT
    if (lane == 0) atomicAdd(P.sync, 1u);
#endif
    TS();                                 // 5: counter incremented
    const unsigned target = P.sync_base + (idx + 1u) * NWORK;
    unsigned polls = 0;
    int j = 0;
#if QKV_NOWAIT
    // after q/k/v (one-block attention): only the attention blocks need the whole grid's outputs; the others have
    // counted themselves in and go on to their o stash at once (they meet everybody at the end of attention)
    const bool nowait = kk == PH_QKV && BIDX >= 8 * P.B && QKV_NW(P.B, P.pos);
#else
    const bool nowait = false;
#endif
    while (!nowait && (int)(*(volatile unsigned*)P.sync - target) < 0) {
      polls++;
      if (nb && j < NLOAD * WPF_DEPTH) {
#pragma unroll
        for (int q = 0; q < WPF; q++, j++) {
          const int d = j / NLOAD, w = j - d * NLOAD;
          PREFETCH(nW + (w * nb + min(w, ne)) * GU4 + d * 32 + lane);
        }
        if (P.B <= WPF_SMALL_B) {   // few sequences: the barrier chain is short, twice the intensity pays (measured)
#pragma unroll
          for (int q = 0; q < WPF; q++, j++) {
            const int d = j / NLOAD, w = j - d * NLOAD;
            PREFETCH(nW + (w * nb + min(w, ne)) * GU4 + d * 32 + lane);
          }
        }
      }
    }
    TS();                                 // 6: all blocks arrived
    TS_FLUSH(kk);
    if (lane == 0) {
      *(volatile int*)(smem + GO_OFF) = 1;   // the loaders stop stashing
      if (P.calib) P.wstat[kk * NWORK + BIDX] += polls;
      C.bi = idx + 1u;
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------------------------------------
// Certified int8 sketch of lm_head (greedy paths only: generate() and the online path; k_gensk). step() keeps
// the exact head. The chosen token is the exact head's argmax (proof: notes in api.py, build_sketch / poll docs):
//  pass 1 (sk1_phase): every row i of the block's lm tiles: zh_i = s_i 2^-kx (128 A1 + A0), A1/A0 = int8 weights
//    q_i . the two int8 digits of x quantized to x^ = q 2^-kx (exact int32 MMAs), and the bound
//    |z_i - zh_i| <= c1_i ||x|| + c2_i ||x - x^|| (+ slack), z_i = the exact head's float32 value of the row
//    before the 1/rms factor. Per tile and sequence: the largest upper bound (tmax, shared memory); per block and
//    sequence: the largest lower bound (global). One grid barrier.
//  pass 2 (sk2_phase): L = max lower bound per sequence, thr = L (1 - 2^-20) (L > 0; otherwise every tile).
//    Tiles whose tmax is not below thr for some sequence are recomputed with the exact head's own code
//    (tile_loop / tile_loop_g and a copy of the coordinator's LMARG epilogue): same operands, same order of
//    operations, so the same float32 logits; argmax over them, lowest row on ties, published as the exact phase.
//    A row of a tile not recomputed has z < thr <= z_j (1 - 2^-20) for the row j with lower bound L, so its
//    logit fl(z r) is strictly below row j's: it can be neither the maximum nor tie with it.
// ---------------------------------------------------------------------------------------------------------
#ifndef SK_RT
#define SK_RT 0     // measurement build: the variants below chosen at run time (SkArgs::pad, mk_sketch_flags) in ONE engine instance
#endif
#ifndef SK_PF
#define SK_PF 0     // pass 1's end: at most this many likely candidate tiles per block asked into L2 (0: off)
#endif
#ifndef SK_PREP
#define SK_PREP 2   // sketch x preparation: 0 two passes of 4-byte loads, 1 max pass with 16-byte loads, 2 both passes with 16-byte loads
#endif
#ifndef SK_LITE
#define SK_LITE 2   // k_gensk: the first tile's extra group loaded behind the base groups' MMAs (prefetched to L2 first): without it the hot loop spilled (+70..150 us at B=1)
#endif
struct SkArgs { const uint4* w8; const float4* rc; float* bl; int force; int pad; unsigned* stat; int* lrun; };   // lrun [2 step parities][8]: running max lower bound (SK_PF)   // via Params.logits (unused by generate)
#ifndef TRACK_A
__device__ __forceinline__ const uint4* sk_w8_of(const Params& P) { return ((const SkArgs*)P.logits)->w8; }
#endif
#ifndef TRACK_A
#define SK_XF_OFF SX_OFF                    // int8 B fragments of x: [hi | lo][128 chunks][32 lanes][8 B] (64 KB)
#define SK_RED_OFF (SX_OFF + 65536)         // pass-1 partials: [2 parity][NLOAD][32 lanes][2 uint4] (47 KB)
#define SK_TMX_OFF (SX_OFF + 131072)        // float [SK_TMAX tiles][8]: largest upper bound per tile and sequence
#define SK_TMAX 80
#define SK_CST_OFF (SX_OFF + 112640)        // pass-1 coordinator staging (SK_COORD_LOOP): [32 lanes][12 floats]
#ifndef SK_LD
#define SK_LD(ptr) WLOAD(ptr)
#endif
#ifndef SK_LOADMODE
#define SK_LOADMODE 1   // pass-1 loader: 0 next tile's loads into registers behind the MMAs (spills the hot loops), 1 loads at the tile's start + L2 prefetch of the next
#endif
#ifndef SK_COORD_LOOP
#define SK_COORD_LOOP 0
#endif
#define SK_CH 128                           // k-chunks of 32 per tile
#define SK_CB (SK_CH / NLOAD)               // chunks per loader warp (+1 for the first SK_CE warps)
#define SK_CE (SK_CH % NLOAD)
#define SK_CW (SK_CB + 1)
static_assert(SK_RED_OFF + 2 * NLOAD * 32 * 32 <= SK_TMX_OFF && SK_TMX_OFF + SK_TMAX * 8 * 4 <= SX_OFF + SX_NARROW && SK_RED_OFF >= SX_OFF + 65536, "sketch smem");
// per-sequence statistics of pass 1 (SCR area): [0,32) max |x| bits, [32,64) kx, [64,128) sum x^2, [128,192) sum r^2
#define SK_ST_MAX(n) (((unsigned*)(smem + SCR_OFF))[n])
#define SK_ST_KX(n) (((int*)(smem + SCR_OFF + 32))[n])
#define SK_ST_XX(n) (((double*)(smem + SCR_OFF + 64))[n])
#define SK_ST_RR(n) (((double*)(smem + SCR_OFF + 128))[n])
#define MMA8(c, a, b0, b1) asm volatile( \
  "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
  : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) \
  : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1))

#ifdef SK_NOINLINE
#define SK_FN __device__ __noinline__   // (measured: much slower, +200..350 us per step)
#else
#define SK_FN __device__ __forceinline__
#endif
#ifndef SKDBG
#define SKDBG 0   // compile experiments: 1 no pass-2 tile loops, 2 no pass-1 loader prefetch, 4 no pass-1 coordinator math, 8 no prep
#endif
SK_FN void sk1_phase(unsigned char* smem) {
  Params& P = *(Params*)(smem + MISC_OFF);
  const SkArgs* SA = (const SkArgs*)P.logits;
  const int tid = TIDX, lane = tid & 31, w = tid >> 5;
  const int* tb = (const int*)(smem + TB_OFF);
  const int t0 = tb[2 * T_LM], t1 = tb[2 * T_LM + 1];
  const int cb = w * SK_CB + min(w, (int)SK_CE), cn = SK_CB + (w < SK_CE ? 1 : 0);
  const uint4* w8 = SA->w8;
  // loaders: the first tile's loads go out before the preparation of x
  uint4 nx[SK_CW];
#pragma unroll
  for (int u = 0; u < SK_CW; u++) nx[u] = make_uint4(0, 0, 0, 0);
  if (w < NLOAD && t0 < t1) {
    const uint4* p = w8 + ((size_t)t0 * SK_CH + cb) * 32 + lane;
#pragma unroll
    for (int u = 0; u < SK_CW; u++) if (u < cn) nx[u] = SK_LD(p + u * 32);
  }
  // x (the head's bf16 input fragments) per sequence: max |x| -> kx; digits; sums of squares
  const int B = P.B, tot = 1024 * B;
  if (tid < 8) { SK_ST_MAX(tid) = 0u; SK_ST_XX(tid) = 0.0; SK_ST_RR(tid) = 0.0; }
  __syncthreads();
  const unsigned* xw = (const unsigned*)P.xfrag;
#if SK_RT
  if (SA->pad & 1) {
#endif
#if SK_PREP >= 1 || SK_RT
  // max |x|: the fragment buffer read as it lies, 8 values per 16-byte load; sequence of load i: compact layout
  // [n][kstep][16] -> i >> 9; wide [kstep pair][lane 4n + t][8] -> (i >> 2) & 7 (slots of n >= B are stale: skipped)
  {
    const uint4* xq = (const uint4*)P.xfrag;
    const bool wide = B >= WIDE_MIN_B;
    const int totq = (SKDBG & 8) ? 0 : (wide ? 4096 : 512 * B);
    for (int i = tid; i < totq; i += NTHREAD) {
      const uint4 v = __ldcg(xq + i);
      const int n = wide ? (i >> 2) & 7 : i >> 9;
      unsigned m = max(max(max(v.x & 0x7fffu, (v.x >> 16) & 0x7fffu), max(v.y & 0x7fffu, (v.y >> 16) & 0x7fffu)),
                       max(max(v.z & 0x7fffu, (v.z >> 16) & 0x7fffu), max(v.w & 0x7fffu, (v.w >> 16) & 0x7fffu)));
      if (n >= B) m = 0u;
      m = max(m, __shfl_xor_sync(0xffffffffu, m, 1));
      m = max(m, __shfl_xor_sync(0xffffffffu, m, 2));    // lanes 4j..4j+3: one sequence in both layouts
      if ((lane & 3) == 0 && n < B) atomicMax(&SK_ST_MAX(n), m);
    }
  }
#endif
#if SK_RT
  } else {
#endif
#if SK_PREP < 1 || SK_RT
  for (int i = tid; i < ((SKDBG & 8) ? 0 : tot); i += NTHREAD) {      // a warp's 32 items belong to one sequence
    const int n = i >> 10, k0 = (i & 1023) * 4;
    const unsigned v0 = __ldcg(xw + (gfrag_index(k0, n, KS_H, B) >> 1)), v1 = __ldcg(xw + (gfrag_index(k0 + 2, n, KS_H, B) >> 1));
    unsigned m = max(max(v0 & 0x7fffu, (v0 >> 16) & 0x7fffu), max(v1 & 0x7fffu, (v1 >> 16) & 0x7fffu));   // |bf16| bits order like the values
    m = __reduce_max_sync(0xffffffffu, m);
    if (lane == 0) atomicMax(&SK_ST_MAX(n), m);
  }
#endif
#if SK_RT
  }
#endif
  __syncthreads();
  if (tid < 8) {
    const float M = __uint_as_float(SK_ST_MAX(tid) << 16);
    int E = 0;
    frexpf(M, &E);                                 // M < 2^E
    SK_ST_KX(tid) = M > 0.f && M <= 3.0e38f && E >= -87 ? 13 - E : 0;   // |x| 2^kx < 2^13: digits in [-64, 64]; kx <= 100
                                                                         // (2^-kx a normal float; else kx = 0: x^ = 0, the residual bound carries x)
  }
  __syncthreads();
#if SK_RT
  if (SA->pad & 1) {
#endif
#if SK_PREP >= 2 || SK_RT
  // digits: 16-byte loads as above, float arithmetic (all exact: x 2^kx, rint, the split, the residual, see below).
  // Element j of load i: compact: t = 2h + (j >> 2), slot j & 3 (h = i & 1) -> natural k within the kstep
  // 2t + (s & 1) + 8 (s >> 1); a load fills digit words t8 = h (j = 0,1,4,5) and t8 = h + 2 (j = 2,3,6,7).
  // wide: j < 4 kstep 2p, j >= 4 kstep 2p + 1, t = i & 3: the thread holds bytes 0,1 (t even) or 2,3 (t odd) of words
  // t8 = t >> 1 (slots 0,1) and (t >> 1) + 2 (slots 2,3); the pair (t, t ^ 1) completes them by a shuffle.
  {
    unsigned* xf = (unsigned*)(smem + SK_XF_OFF);
    const uint4* xq = (const uint4*)P.xfrag;
    const bool wide = B >= WIDE_MIN_B;
    const int totq = (SKDBG & 8) ? 0 : (wide ? 4096 : 512 * B);
    for (int i = tid; i < totq; i += NTHREAD) {
      const uint4 v = __ldcg(xq + i);
      const int n = wide ? (i >> 2) & 7 : i >> 9;
      const int kx = SK_ST_KX(n < B ? n : 0);
      const float sc = __uint_as_float((unsigned)(127 + kx) << 23), isc = __uint_as_float((unsigned)(127 - kx) << 23);
      const unsigned vv[4] = {v.x, v.y, v.z, v.w};
      unsigned hb[8], lb[8];
      double sxx = 0.0, srr = 0.0;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float x = bf2f((j & 1) ? vv[j >> 1] >> 16 : vv[j >> 1] & 0xffffu);
        const float q = fminf(fmaxf(rintf(x * sc), -8256.f), 8256.f);   // |x| 2^kx < 2^13 (non-finite x: bounds NaN anyway)
        const float d1 = rintf(q * 0.0078125f), d0 = q - 128.f * d1;    // exact: |d1|, |d0| <= 64
        const float r = x - q * isc;                                     // exact (r = x when q = 0, else |r| <= 2^-kx / 2)
        sxx += (double)x * x; srr += (double)r * r;
        hb[j] = (unsigned)(int)d1 & 0xffu; lb[j] = (unsigned)(int)d0 & 0xffu;
      }
      if (!wide) {
        const int ks = (i >> 1) & 255, h = i & 1, c = ks >> 1, kh = ks & 1;
        const int w0 = (c * 32 + n * 4 + h) * 2 + kh, w2 = w0 + 4;     // t8 = h, h + 2
        xf[w0] = hb[0] | hb[1] << 8 | hb[4] << 16 | hb[5] << 24; xf[SK_CH * 64 + w0] = lb[0] | lb[1] << 8 | lb[4] << 16 | lb[5] << 24;
        xf[w2] = hb[2] | hb[3] << 8 | hb[6] << 16 | hb[7] << 24; xf[SK_CH * 64 + w2] = lb[2] | lb[3] << 8 | lb[6] << 16 | lb[7] << 24;
      } else {
        const int t = i & 3, p = i >> 5;
        // my halves: [kstep q][slot pair g2] -> 16 bits
        unsigned H0 = (hb[0] | hb[1] << 8) | (hb[2] | hb[3] << 8) << 16, H1 = (hb[4] | hb[5] << 8) | (hb[6] | hb[7] << 8) << 16;
        unsigned L0 = (lb[0] | lb[1] << 8) | (lb[2] | lb[3] << 8) << 16, L1 = (lb[4] | lb[5] << 8) | (lb[6] | lb[7] << 8) << 16;
        const unsigned pH0 = __shfl_xor_sync(0xffffffffu, H0, 1), pH1 = __shfl_xor_sync(0xffffffffu, H1, 1);
        const unsigned pL0 = __shfl_xor_sync(0xffffffffu, L0, 1), pL1 = __shfl_xor_sync(0xffffffffu, L1, 1);
        const bool ev = (t & 1) == 0;
        // the even thread writes kstep 2p (q = 0), the odd one kstep 2p + 1 (q = 1); word = even's half | odd's half << 16
        const unsigned mh = ev ? H0 : H1, ph = ev ? pH0 : pH1, ml = ev ? L0 : L1, pl = ev ? pL0 : pL1;   // (q of the writer)
        // even writer, q = 0: even's halves are mine (H0), odd's are the partner's q = 0 halves (pH0)
        // odd writer, q = 1: even's halves are the partner's q = 1 halves (pH1), odd's are mine (H1)
        const unsigned eh = ev ? mh : ph, oh = ev ? ph : mh, el = ev ? ml : pl, ol = ev ? pl : ml;
        const int q = ev ? 0 : 1, t8 = t >> 1;
        if (n < B) {
          const int wa = (p * 32 + n * 4 + t8) * 2 + q, wb = wa + 4;
          xf[wa] = (eh & 0xffffu) | (oh << 16); xf[SK_CH * 64 + wa] = (el & 0xffffu) | (ol << 16);
          xf[wb] = (eh >> 16) | (oh & 0xffff0000u); xf[SK_CH * 64 + wb] = (el >> 16) | (ol & 0xffff0000u);
        }
      }
      if (n >= B) { sxx = 0.0; srr = 0.0; }
      sxx += __shfl_xor_sync(0xffffffffu, sxx, 1); srr += __shfl_xor_sync(0xffffffffu, srr, 1);
      sxx += __shfl_xor_sync(0xffffffffu, sxx, 2); srr += __shfl_xor_sync(0xffffffffu, srr, 2);
      if ((lane & 3) == 0 && n < B) { atomicAdd(&SK_ST_XX(n), sxx); atomicAdd(&SK_ST_RR(n), srr); }
    }
  }
#endif
#if SK_RT
  } else {
#endif
#if SK_PREP < 2 || SK_RT
  unsigned* xf = (unsigned*)(smem + SK_XF_OFF);
  for (int i = tid; i < ((SKDBG & 8) ? 0 : tot); i += NTHREAD) {
    const int n = i >> 10, jj = i & 1023, k0 = jj * 4;
    const unsigned v0 = __ldcg(xw + (gfrag_index(k0, n, KS_H, B) >> 1)), v1 = __ldcg(xw + (gfrag_index(k0 + 2, n, KS_H, B) >> 1));
    const float xv[4] = {bf2f(v0 & 0xffffu), bf2f(v0 >> 16), bf2f(v1 & 0xffffu), bf2f(v1 >> 16)};
    const int kx = SK_ST_KX(n);
    const double sc = ldexp(1.0, kx), isc = ldexp(1.0, -kx);
    unsigned hi = 0, lo = 0;
    double sxx = 0.0, srr = 0.0;
#pragma unroll
    for (int e = 0; e < 4; e++) {
      const double x = (double)xv[e];
      double q = rint(x * sc);
      q = fmin(fmax(q, -8256.0), 8256.0);          // (only a non-finite x gets here out of range: its bounds are NaN)
      const double d1 = rint(q * (1.0 / 128.0)), d0 = q - 128.0 * d1;
      const double r = x - q * isc;
      sxx += x * x; srr += r * r;
      hi |= ((unsigned)(int)d1 & 0xffu) << (8 * e);
      lo |= ((unsigned)(int)d0 & 0xffu) << (8 * e);
    }
    const int c = jj >> 3, kh = (jj >> 2) & 1, t8 = jj & 3;
    const int wi = (c * 32 + n * 4 + t8) * 2 + kh;
    xf[wi] = hi; xf[SK_CH * 32 * 2 + wi] = lo;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { sxx += __shfl_xor_sync(0xffffffffu, sxx, o); srr += __shfl_xor_sync(0xffffffffu, srr, o); }
    if (lane == 0) { atomicAdd(&SK_ST_XX(n), sxx); atomicAdd(&SK_ST_RR(n), srr); }
  }
#endif
#if SK_RT
  }
#endif
  __syncthreads();
  if (w < NLOAD) {
    // pass 1, loader warp: chunks [cb, cb + cn) of every tile, int8 MMAs (hi and lo digits), partials to shared
    // memory. One register buffer: the next tile's loads go out right behind this tile's MMAs (3 KB per warp in
    // flight is enough for the SM's share of HBM; a second buffer pushed the hot loops' registers to local memory)
    const uint2* xh = (const uint2*)(smem + SK_XF_OFF) + lane;
    const uint2* xl = xh + SK_CH * 32;
    uint4* red = (uint4*)(smem + SK_RED_OFF) + (w * 32 + lane) * 2;
    int par = 0;
    for (int t = t0; t < t1; t++, par ^= 1) {
#if SK_LOADMODE == 1
      // this tile's loads now (the first tile's went out before the preparation); the next tile is asked from L2
      // with prefetches, which hold no registers (a register buffer for it spilled the hot loops)
      if (t > t0) {
        const uint4* p = w8 + ((size_t)t * SK_CH + cb) * 32 + lane;
#pragma unroll
        for (int u = 0; u < SK_CW; u++) if (u < cn) nx[u] = SK_LD(p + u * 32);
      }
      if (t + 1 < t1) {
        const uint4* p = w8 + ((size_t)(t + 1) * SK_CH + cb) * 32 + lane;
#pragma unroll
        for (int u = 0; u < SK_CW; u++) if (u < cn) PREFETCH(p + u * 32);
      }
#endif
      int ch[4] = {0, 0, 0, 0}, cl[4] = {0, 0, 0, 0};
#pragma unroll
      for (int u = 0; u < SK_CW; u++) {
        if (u < cn) {
          const uint2 bh = xh[(cb + u) * 32], bl = xl[(cb + u) * 32];
          MMA8(ch, nx[u], bh.x, bh.y);
          MMA8(cl, nx[u], bl.x, bl.y);
        }
      }
      if (SK_LOADMODE == 0 && t + 1 < t1 && !(SKDBG & 2)) {
        const uint4* p = w8 + ((size_t)(t + 1) * SK_CH + cb) * 32 + lane;
#pragma unroll
        for (int u = 0; u < SK_CW; u++) if (u < cn) nx[u] = SK_LD(p + u * 32);
      }
      red[par * (NLOAD * 64)] = make_uint4(ch[0], ch[1], ch[2], ch[3]);
      red[par * (NLOAD * 64) + 1] = make_uint4(cl[0], cl[1], cl[2], cl[3]);
      __syncthreads();
    }
  } else {
    // pass 1, coordinator warp: bounds per (row, sequence), float32 with directed rounding:
    //   A = 128 A1 + A0 (int64, exact), zh = fl(fl(A) s_i) 2^-kx: two roundings, |zh - A s_i 2^-kx| <= 2^-22.9 |zh|
    //   (+ subnormal loss < 2^-126), covered by the 2^-20 |zh| + 2^-120 slack;
    //   bd = c1 nx + c2 nr + 2^-20 |zh| + 2^-120 rounded up (nx >= ||x||, nr >= ||x - x^|| rounded up);
    //   zu = zh + bd rounded up, zl = zh - bd rounded down.
    const int g = lane >> 2, n0 = 2 * (lane & 3), n1 = n0 + 1;
    float kx0 = 0.f, kx1 = 0.f, nx0 = 0.f, nx1 = 0.f, nr0 = 0.f, nr1 = 0.f;
    if (n0 < B) { kx0 = __uint_as_float((unsigned)(127 - SK_ST_KX(n0)) << 23); nx0 = __double2float_ru(sqrt(SK_ST_XX(n0)) * (1.0 + 0x1p-30)); nr0 = __double2float_ru(sqrt(SK_ST_RR(n0)) * (1.0 + 0x1p-30)); }
    if (n1 < B) { kx1 = __uint_as_float((unsigned)(127 - SK_ST_KX(n1)) << 23); nx1 = __double2float_ru(sqrt(SK_ST_XX(n1)) * (1.0 + 0x1p-30)); nr1 = __double2float_ru(sqrt(SK_ST_RR(n1)) * (1.0 + 0x1p-30)); }
    float lb0 = -INFINITY, lb1 = -INFINITY;
    float* tmx = (float*)(smem + SK_TMX_OFF);
    const float4* rc = SA->rc;
    int par = 0;
    for (int t = t0; t < t1; t++, par ^= 1) {
      const float4 q0 = __ldcg(rc + t * 16 + g), q1 = __ldcg(rc + t * 16 + g + 8);   // {s, c1, c2, -} of rows g, g + 8
      __syncthreads();
      const uint4* red = (const uint4*)(smem + SK_RED_OFF) + par * (NLOAD * 64) + lane * 2;
      int h0 = 0, h1 = 0, h2 = 0, h3 = 0, l0 = 0, l1 = 0, l2 = 0, l3 = 0;
#pragma unroll 4
      for (int j = 0; j < NLOAD; j++) {
        const uint4 hh = red[j * 64], ll = red[j * 64 + 1];
        h0 += (int)hh.x; h1 += (int)hh.y; h2 += (int)hh.z; h3 += (int)hh.w;
        l0 += (int)ll.x; l1 += (int)ll.y; l2 += (int)ll.z; l3 += (int)ll.w;
      }
      if (SKDBG & 4) { if (g == 0) tmx[(t - t0) * 8 + n0] = h0 + l0 + h1 + l1 + h2 + l2 + h3 + l3; continue; }
      float m0, m1;
#if SK_COORD_LOOP
      // (variant: one entry at a time from shared memory, fewer live registers)
      {
        float* st = (float*)(smem + SK_CST_OFF) + lane * 12;
        ((float4*)st)[0] = q0; ((float4*)st)[1] = q1;
        st[8] = __ll2float_rn(128ll * h0 + l0); st[9] = __ll2float_rn(128ll * h1 + l1); st[10] = __ll2float_rn(128ll * h2 + l2); st[11] = __ll2float_rn(128ll * h3 + l3);
        m0 = -INFINITY; m1 = -INFINITY;
#pragma unroll 1
        for (int e = 0; e < 4; e++) {
          const volatile float* sv = st;
          const int qo = (e >> 1) * 4, c1st = (e & 1) == 0;
          const float zh = __fmul_rn(__fmul_rn(sv[8 + e], sv[qo]), c1st ? kx0 : kx1);
          const float bd = __fmaf_ru(fabsf(zh), 0x1p-20f, __fadd_ru(__fadd_ru(__fmul_ru(sv[qo + 1], c1st ? nx0 : nx1), __fmul_ru(sv[qo + 2], c1st ? nr0 : nr1)), 0x1p-120f));
          float zu = __fadd_ru(zh, bd), zl = __fsub_rd(zh, bd);
          if (!(zu == zu)) zu = INFINITY;
          if (!(zl == zl)) zl = -INFINITY;
          if (c1st) { m0 = fmaxf(m0, zu); lb0 = fmaxf(lb0, zl); } else { m1 = fmaxf(m1, zu); lb1 = fmaxf(lb1, zl); }
        }
      }
#else
#define SK_ENT(hh, ll, q, kx, nxv, nrv, zu, zl) { \
        const float zh = __fmul_rn(__fmul_rn(__ll2float_rn(128ll * (hh) + (ll)), (q).x), kx); \
        const float bd = __fmaf_ru(fabsf(zh), 0x1p-20f, __fadd_ru(__fadd_ru(__fmul_ru((q).y, nxv), __fmul_ru((q).z, nrv)), 0x1p-120f)); \
        zu = __fadd_ru(zh, bd); zl = __fsub_rd(zh, bd); \
        if (!(zu == zu)) zu = INFINITY; if (!(zl == zl)) zl = -INFINITY; }
      {
        float u0, d0, u1, d1, u2, d2, u3, d3;
        SK_ENT(h0, l0, q0, kx0, nx0, nr0, u0, d0)   // (row g, n0)
        SK_ENT(h1, l1, q0, kx1, nx1, nr1, u1, d1)   // (row g, n1)
        SK_ENT(h2, l2, q1, kx0, nx0, nr0, u2, d2)   // (row g + 8, n0)
        SK_ENT(h3, l3, q1, kx1, nx1, nr1, u3, d3)   // (row g + 8, n1)
        m0 = fmaxf(u0, u2); m1 = fmaxf(u1, u3);
        lb0 = fmaxf(lb0, fmaxf(d0, d2)); lb1 = fmaxf(lb1, fmaxf(d1, d3));
      }
#undef SK_ENT
#endif
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) { m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o)); }
      const int tl = t - t0;
      if (g == 0 && tl < SK_TMAX) { tmx[tl * 8 + n0] = m0; tmx[tl * 8 + n1] = m1; }
    }
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) { lb0 = fmaxf(lb0, __shfl_xor_sync(0xffffffffu, lb0, o)); lb1 = fmaxf(lb1, __shfl_xor_sync(0xffffffffu, lb1, o)); }
    if (g == 0) {
      const int nb = NWORK, b = BIDX;
      if (n0 < B) SA->bl[n0 * nb + b] = lb0;
      if (n1 < B) SA->bl[n1 * nb + b] = lb1;
    }
#if SK_PF > 0 || SK_RT
    if (SK_RT ? (SA->pad >> 4) > 0 : true) {
      const int pfmax = SK_RT ? (SA->pad >> 4) & 15 : SK_PF;
      // Likely candidates of pass 2 asked into L2 now (their 12-bit tiles; HBM is nearly idle from here to the end of
      // pass 2): tiles whose largest upper bound reaches the running maximum of the lower bounds published so far
      // by all blocks (a lower bound of the final L, so the true candidates of this block are among them). Reads only.
      int* lr = SA->lrun + (((const Ctl*)(smem + CTL_OFF))->s & 1) * 8;
      float L0 = -INFINITY, L1 = -INFINITY;
      if (g == 0) {
        if (n0 < B && lb0 > 0.f) L0 = __int_as_float(max(atomicMax(lr + n0, __float_as_int(lb0)), __float_as_int(lb0)));
        if (n1 < B && lb1 > 0.f) L1 = __int_as_float(max(atomicMax(lr + n1, __float_as_int(lb1)), __float_as_int(lb1)));
      }
      float thr[8];
#pragma unroll
      for (int n = 0; n < 8; n++) {
        const float v = __shfl_sync(0xffffffffu, (n & 1) ? L1 : L0, n >> 1);
        thr[n] = v > 0.f ? __fmul_rd(v, 1.0f - 0x1p-20f) : INFINITY;   // (no positive bound yet: nothing staged)
      }
      const float* tmx = (const float*)(smem + SK_TMX_OFF);
      const int nt = min(t1 - t0, (int)SK_TMAX);
      int done = 0;
      for (int j0 = 0; j0 < nt && done < pfmax; j0 += 32) {
        const int tl = j0 + lane;
        bool c = false;
        if (tl < nt) {
#pragma unroll
          for (int n = 0; n < 8; n++) if (n < B && !(tmx[tl * 8 + n] < thr[n])) c = true;
        }
        unsigned bal = __ballot_sync(0xffffffffu, c);
        while (bal && done < pfmax) {
          const int j = __ffs(bal) - 1;
          bal &= bal - 1;
          enum { PIECES = (int)(TU4(KS_H) / 1024) };
          if (lane < PIECES) l2_stage(P.W + (size_t)NL * LAYER_U4 + (size_t)(t0 + j0 + j) * TU4(KS_H) + lane * 1024, 16384);
          done++;
        }
      }
    }
#endif
    __threadfence();
    __syncwarp();
    if (lane == 0) {
      set_desc(smem, PH_LMARG, 0);
      ((Desc*)(smem + DESC_OFF))->k = PH_SK2;
      grid_barrier(smem, PH_LMARG TR_ARG(-100));
    }
  }
  __syncthreads();
}

SK_FN void sk2_phase(unsigned char* smem) {
  Params& P = *(Params*)(smem + MISC_OFF);
  const SkArgs* SA = (const SkArgs*)P.logits;
  const Desc& D = *(const Desc*)(smem + DESC_OFF);
  const int tid = TIDX, lane = tid & 31, w = tid >> 5;
  const int B = P.B;
  int* cand = (int*)(smem + SCR_OFF);              // [0] count, [1..] candidate tiles in ascending order
  if (w < B) {                                     // 1/rms per sequence, as entry_copy computes it
    const float r = rms_inv(P, w, lane);
    if (lane == 0) ((float*)(smem + RN_OFF))[w] = r;
  }
  if (w == CW) {
    float thr[8];
    const int nb = NWORK;
#pragma unroll
    for (int n = 0; n < 8; n++) {
      float L = -INFINITY;
      if (n < B) {
        for (int i = lane; i < nb; i += 32) L = fmaxf(L, __ldcg(SA->bl + n * nb + i));
        L = warp_max(L);
      }
      thr[n] = (L > 1e-30f && L < 3.0e38f && !SA->force) ? __fmul_rd(L, 1.0f - 0x1p-20f) : -INFINITY;   // (1 - 2^-20 is exact in float32)
    }
    const float* tmx = (const float*)(smem + SK_TMX_OFF);
    const int nt = D.t1 - D.t0;
    int cnt = 0;
    for (int j0 = 0; j0 < nt; j0 += 32) {
      const int tl = j0 + lane;
      bool c = false;
      if (tl < nt) {
        if (tl >= SK_TMAX) c = true;
#pragma unroll
        for (int n = 0; n < 8; n++) if (n < B && tl < SK_TMAX && !(tmx[tl * 8 + n] < thr[n])) c = true;
      }
      const unsigned bal = __ballot_sync(0xffffffffu, c);
      if (c) cand[1 + cnt + __popc(bal & ((1u << lane) - 1u))] = D.t0 + tl;
      cnt += __popc(bal);
    }
    if (lane == 0) { cand[0] = cnt; if (SA->stat) atomicAdd(SA->stat, (unsigned)cnt); }
    ((float4*)(smem + ARG_OFF))[lane] = make_float4(-__int_as_float(0x7f800000), -__int_as_float(0x7f800000), __int_as_float((SKDBG & 16) ? 0 : 0x7fffffff), __int_as_float((SKDBG & 16) ? 0 : 0x7fffffff));
  }
  __syncthreads();
  const int ncand = (SKDBG & 16) ? 0 : (SK_RT && (SA->pad & 0x400)) ? min(cand[0], 1) : cand[0];
  for (int ci = 0; ci < ncand; ci++) {
    const int t = ((const volatile int*)cand)[1 + ci];
    if (w < NLOAD) {
      if (!(SKDBG & 1) && !(SK_RT && (SA->pad & 0x200))) {
      if (B >= WIDE_MIN_B) tile_loop_g<KS_H>(smem, D.W, (const uint4*)D.frag, t, t + 1, false);
      else tile_loop<KS_H, 16>(smem, D.W, t, t + 1, false, false);
      } else __syncthreads();
    } else {
      // the coordinator's EP_LMARG tile, as in coord_phase
      const float4* red = (const float4*)(smem + RED_OFF) + lane;
      float4* const am = (float4*)(smem + ARG_OFF) + lane;
      const uint4* te = D.tt + (size_t)t * TT;
      const uint4 te0 = __ldcg(te), te1 = __ldcg(te + 1);
      uint4 ri[INLINE_CHUNKS];
#pragma unroll
      for (int c = 0; c < INLINE_CHUNKS; c++) ri[c] = __ldcg(te + 2 + c * 4 + (lane & 3));
      {
        const unsigned sw0 = (lane >> 4) ? te1.y : te1.x, sw1 = (lane >> 4) ? te1.w : te1.z;
        const int sbit = ((lane >> 2) & 3) * 8;
        ((float2*)(smem + SCL_OFF))[lane] = make_float2(__uint_as_float((unsigned)(127 - (int)(signed char)(sw0 >> sbit)) << 23),
                                                        __uint_as_float((unsigned)(127 - (int)(signed char)(sw1 >> sbit)) << 23));
      }
      float cex[4] = {0.f, 0.f, 0.f, 0.f};
      {
#if XSLICE
        const unsigned short* sx16 = D.frag + (size_t)(lane >> 2) * D.ks * 16;
#else
        const unsigned short* sx16 = (const unsigned short*)(smem + (D.ks == KS_I ? SX_OFF : HF_OFF)) + 16 * (lane >> 2);
#endif
        const bool wide = B >= WIDE_MIN_B, live = (lane >> 2) < B;
        const unsigned short* fg = D.frag;
        const unsigned g = lane >> 2;
        const int n = (SK_RT && (SA->pad & 0x100)) ? 0 : (int)te0.y;
#pragma unroll
        for (int c = 0; c < INLINE_CHUNKS; c++) if (c < n) exc_mma(cex, ri[c], sx16, fg, g, wide, live);
        if (n > INLINE_CHUNKS) {
          const uint4* ex = P.exc + (size_t)te0.x * 4 + (lane & 3);
          for (int i = 0; i < n - INLINE_CHUNKS; i += 2) {
            const uint4 ra = __ldcg(ex + i * 4), rb = __ldcg(ex + i * 4 + 4);
            exc_mma(cex, ra, sx16, fg, g, wide, live);
            if (i + 1 < n - INLINE_CHUNKS) exc_mma(cex, rb, sx16, fg, g, wide, live);
          }
        }
      }
      ((float4*)(smem + RED_OFF) + (t & 1) * RED_HALF + NLOAD * 32 + lane)[0] = make_float4(cex[0], cex[1], cex[2], cex[3]);
      __syncthreads();
      const float4* r = red + (t & 1) * RED_HALF;
#define R_(i) r[(i) * 32]
#define S8_(o) add4(add4(add4(R_(o), R_(o + 1)), add4(R_(o + 2), R_(o + 3))), add4(add4(R_(o + 4), R_(o + 5)), add4(R_(o + 6), R_(o + 7))))
#if NLOAD == 31
      float4 q0 = S8_(0), q1 = S8_(8), q2 = S8_(16);
      float4 q3 = add4(add4(add4(R_(24), R_(25)), add4(R_(26), R_(27))), add4(add4(R_(28), R_(29)), R_(30)));
      float4 c = add4(add4(q0, q1), add4(q2, q3));
#elif NLOAD == 15
      float4 q0 = S8_(0);
      float4 q1 = add4(add4(add4(R_(8), R_(9)), add4(R_(10), R_(11))), add4(add4(R_(12), R_(13)), R_(14)));
      float4 c = add4(q0, q1);
#elif NLOAD == 23
      float4 q0 = S8_(0), q1 = S8_(8);
      float4 q2 = add4(add4(add4(R_(16), R_(17)), add4(R_(18), R_(19))), add4(add4(R_(20), R_(21)), R_(22)));
      float4 c = add4(add4(q0, q1), q2);
#else
#error "reduction tree for this NLOAD"
#endif
      {
        const float4 cx = R_(NLOAD);
        const float2 sc = ((const float2*)(smem + SCL_OFF))[lane];
        c.x = (c.x + cx.x) * sc.x; c.y = (c.y + cx.y) * sc.x; c.z = (c.z + cx.z) * sc.y; c.w = (c.w + cx.w) * sc.y;
      }
#undef S8_
#undef R_
      const float* rn = (const float*)(smem + RN_OFF);
      const float r0 = rn[2 * (lane & 3)], r1 = rn[2 * (lane & 3) + 1];
      const int row = t * 16 + (lane >> 2);
      const float4 a4 = *am;
      float bv0 = a4.x, bv1 = a4.y;
      int bi0 = __float_as_int(a4.z), bi1 = __float_as_int(a4.w);
      float v;
      v = RB(c.x * r0); if (v > bv0) { bv0 = v; bi0 = row; }
      v = RB(c.z * r0); if (v > bv0) { bv0 = v; bi0 = row + 8; }
      v = RB(c.y * r1); if (v > bv1) { bv1 = v; bi1 = row; }
      v = RB(c.w * r1); if (v > bv1) { bv1 = v; bi1 = row + 8; }
      *am = make_float4(bv0, bv1, __int_as_float(bi0), __int_as_float(bi1));
    }
    __syncthreads();                               // the next candidate may have the same parity
  }
  if (w == CW) {
    const float4 a4 = ((const float4*)(smem + ARG_OFF))[lane];
    float bv0 = a4.x, bv1 = a4.y;
    int bi0 = __float_as_int(a4.z), bi1 = __float_as_int(a4.w);
#pragma unroll
    for (int o = 16; o >= 4; o >>= 1) {
      float u = __shfl_xor_sync(0xffffffffu, bv0, o); int ix = __shfl_xor_sync(0xffffffffu, bi0, o);
      if (u > bv0 || (u == bv0 && ix < bi0)) { bv0 = u; bi0 = ix; }
      u = __shfl_xor_sync(0xffffffffu, bv1, o); ix = __shfl_xor_sync(0xffffffffu, bi1, o);
      if (u > bv1 || (u == bv1 && ix < bi1)) { bv1 = u; bi1 = ix; }
    }
    if (lane < 4) {
      const int n0 = 2 * lane, n1 = 2 * lane + 1, nb = NWORK, b = BIDX;
      if (n0 < B) { P.bm_val[n0 * nb + b] = bv0; P.bm_idx[n0 * nb + b] = bi0; }
      if (n1 < B) { P.bm_val[n1 * nb + b] = bv1; P.bm_idx[n1 * nb + b] = bi1; }
    }
    __threadfence();
    __syncwarp();
    if (lane == 0) {
      Ctl& C = *(Ctl*)(smem + CTL_OFF);
#if SK_PF > 0 || SK_RT
      if (BIDX == 0) { int* lr = SA->lrun + ((C.s + 1) & 1) * 8; for (int n = 0; n < 8; n++) lr[n] = 0; __threadfence(); }   // the next step's slot
#endif
      C.s = C.s + 1; P.pos = P.pos + 1; ((Desc*)(smem + DESC_OFF))->k = PH_EMBED;   // next step
      grid_barrier(smem, PH_LMARG TR_ARG(-100));
    }
  }
  __syncthreads();
}

#endif  // !TRACK_A (sketch)

// MMA B operand of one kstep for the V sum from the exponentials of the softmax: positions 16s + 2t, +1 (q[0]) and
// 16s + 8 + 2t, +1 (q[4]) of the lane's head, p = bf16(e * (1 / Z)). One definition for every code path.
__device__ __forceinline__ uint2 pfrag_e(const float2* q, float rz) {
  const float2 lo = q[0], hi = q[4];
  return make_uint2((unsigned)f2bf(lo.x * rz) | ((unsigned)f2bf(lo.y * rz) << 16), (unsigned)f2bf(hi.x * rz) | ((unsigned)f2bf(hi.y * rz) << 16));
}

// the same operand from probabilities already converted in place (bf16 in the low half of each slot)
__device__ __forceinline__ uint2 pfrag_u(const uint2* q) {
  const uint2 lo = q[0], hi = q[4];
  return make_uint2(lo.x | (lo.y << 16), hi.x | (hi.y << 16));
}

// ---------------------------------------------------------------------------------------------------------
// attention: block b < 8B handles (sequence n = b/8, kv head g = b%8): q/k norms, RoPE, append K/V,
// scores (MMA over the K cache), softmax, weighted V sum (MMA over the V cache)
// ---------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void attention_body(unsigned char* smem, int layer TR_PARAM) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const int b = BIDX, B = P.B;
  const int n = b >> 3, g = b & 7;
  const int tid = TIDX, lane = tid & 31, w = tid >> 5;
#ifdef WHATIF_T   // diagnosis only (wrong results): attention over at most WHATIF_T positions
  const int pos = P.pos, T = min(pos + 1, WHATIF_T), nt = (T + 15) >> 4, NTP = nt * 16;
#else
  const int pos = P.pos, T = pos + 1, nt = (T + 15) >> 4, NTP = nt * 16;
#endif
  float* y = (float*)(smem + ATT_OFF_BYTES);                 // 3072 B: byte flags of step 4 (1024 B used)
  unsigned short* qfrag16 = (unsigned short*)(y + 768);      // [8][32][4]
  const uint2* qfrag = (const uint2*)qfrag16;
  float* sc = (float*)(qfrag16 + 8 * 32 * 4);                // [4][NTP]: scores, then exp, then bf16 probabilities in place
  float* scr = (float*)(smem + SCR_OFF);
  float4* red = (float4*)(smem + RED_OFF);
  uint4* Kc = P.kc + KVU(P, layer, n * 8 + g);
  uint4* Vc = P.vc + KVU(P, layer, n * 8 + g);

  // The block's own K/V history: every warp asks L2 for its position tiles (4 KB of K and of V per instruction)
  // before the prep computes, so that the score and V loads find them there.
#define ATT_PREFETCH_() do { if (nt * B <= TOUCH_TILES) for (int tile_ = w; tile_ < nt; tile_ += NWARP) { \
    PREFETCH(Kc + (size_t)tile_ * KVT + lane * 8); PREFETCH(Vc + (size_t)tile_ * KVT + lane * 8); } } while (0)
#if ATT_PF == 2
  ATT_PREFETCH_();
#endif
  // ---- step 1: q (4 heads), k, v of the new token. One warp per vector (warp = vec: 0..3 q heads, 4 k, 5 v),
  // lane holds elements 4*lane .. 4*lane+3: RMSNorm is a warp reduction and the RoPE partner (d ^ 64) is
  // lane ^ 16, so this stage needs no block barrier and no shared-memory round trip.
  if (w < 6) {
    const int vec = w;
    const int row = vec < 4 ? (4 * g + vec) * 128 : (vec == 4 ? 4096 + g * 128 : 5120 + g * 128);
    const float4 x = __ldcg((const float4*)(P.qkv + (size_t)n * QKV_ROWS + row) + lane);
    float xe[4] = {x.x, x.y, x.z, x.w};
    unsigned short ob[4];
    if (vec < 5) {
      const float4 wn = ((const float4*)(P.qk_norm_w + (layer * 2 + (vec < 4 ? 0 : 1)) * 128))[lane];
      const float4 cs = ((const float4*)(P.rope + (size_t)pos * 128))[lane & 15];
      const float4 sn = ((const float4*)(P.rope + (size_t)pos * 128 + 64))[lane & 15];
#if ATT_PF == 1
      ATT_PREFETCH_();
#endif
      const float ss = warp_sum((x.x * x.x + x.y * x.y) + (x.z * x.z + x.w * x.w));
      const float r = 1.0f / sqrtf(ss * (1.0f / 128) + 1e-6f);
      const float we[4] = {wn.x, wn.y, wn.z, wn.w}, ce[4] = {cs.x, cs.y, cs.z, cs.w}, se[4] = {sn.x, sn.y, sn.z, sn.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const float yv = RB(we[i] * RB(xe[i] * r));
        const float other = __shfl_xor_sync(0xffffffffu, yv, 16);
        ob[i] = f2bf(lane < 16 ? RB(yv * ce[i]) - RB(other * se[i]) : RB(yv * ce[i]) + RB(other * se[i]));
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) ob[i] = f2bf(xe[i]);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int d = 4 * lane + i;
      if (vec < 4) {
        qfrag16[(((d >> 4) * 32 + 4 * vec + frag_t(d)) << 2) + frag_slot(d)] = ob[i];
      } else if (vec == 4) {
        const int m = pos & 15, kcol = d & 15;
        const int ln = (m & 7) * 4 + ((kcol & 7) >> 1), el = (kcol >> 3) * 4 + (m >> 3) * 2 + (kcol & 1);
        ((unsigned short*)Kc)[(size_t)(pos >> 4) * KVT * 8 + ((((d >> 4) * 32 + ln) << 3) + el)] = ob[i];
      } else {
        const int kcol = pos & 15, m = d & 15;
        const int ln = (m & 7) * 4 + ((kcol & 7) >> 1), el = (kcol >> 3) * 4 + (m >> 3) * 2 + (kcol & 1);
        ((unsigned short*)Vc)[(size_t)(pos >> 4) * KVT * 8 + ((((d >> 4) * 32 + ln) << 3) + el)] = ob[i];
      }
    }
#if PREP_FENCE
    __threadfence();
#endif
  }
#if ATT_PF == 1
  else ATT_PREFETCH_();
  if (w == 5) ATT_PREFETCH_();          // (the v warp has no loads of its own before)
#endif
  __syncthreads();
  TP(1);
  ATS(0);

  // ---- step 2: scores[h][j] = q_h . K_j / sqrt(128), and their maximum per head on the way
  float m0 = -__int_as_float(0x7f800000), m1 = m0;       // heads 2t and 2t+1 of this lane (t = lane & 3 < 2)
  for (int tile = w; tile < nt; tile += NWARP) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const uint4* p = Kc + (size_t)tile * KVT + lane;
#if KBATCH
    uint4 a[8];                         // the 8 loads of a position tile back to back (one at a time: 8 round trips)
#pragma unroll
    for (int s = 0; s < 8; s++) a[s] = __ldcg(p + s * 32);
#pragma unroll
    for (int s = 0; s < 8; s++) {
      uint2 bq = make_uint2(0, 0);
      if (lane < 16) bq = qfrag[s * 32 + lane];
      MMA(c, a[s], bq);
    }
#else
#pragma unroll
    for (int s = 0; s < 8; s++) {
      uint4 a = __ldcg(p + s * 32);
      uint2 bq = make_uint2(0, 0);
      if (lane < 16) bq = qfrag[s * 32 + lane];
      MMA(c, a, bq);
    }
#endif
    int t = lane & 3, j = tile * 16 + (lane >> 2);
    if (t < 2) {
      const float scale = SCORE_SCALE;
      const float ninf = -__int_as_float(0x7f800000);
      const float s0 = j < T ? SCORE_(c[0]) : ninf, s1 = j < T ? SCORE_(c[1]) : ninf;
      const float s2 = j + 8 < T ? SCORE_(c[2]) : ninf, s3 = j + 8 < T ? SCORE_(c[3]) : ninf;
      sc[(2 * t) * NTP + j] = s0;
      sc[(2 * t + 1) * NTP + j] = s1;
      sc[(2 * t) * NTP + j + 8] = s2;
      sc[(2 * t + 1) * NTP + j + 8] = s3;
      m0 = fmaxf(m0, fmaxf(s0, s2)); m1 = fmaxf(m1, fmaxf(s1, s3));
    }
  }
  {
    // the warp's maximum per head: lanes with the same t hold the same two heads
#pragma unroll
    for (int o = 4; o <= 16; o <<= 1) { m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o)); }
    if (lane < 2) { scr[(2 * lane) * NWARP + w] = m0; scr[(2 * lane + 1) * NWARP + w] = m1; }
  }
  __syncthreads();
#if ATT_WAIT
  if (tid == 0 && B <= ATT_WAIT_MAX_B) P.att_done[b] = (pos << 6) | (layer + 1);   // "my K loads are done": the idle blocks start their stash
#endif
  TP(2);
  ATS(1);

#if VPF > 0
  if (nt >= LONG_NT) {
    // long contexts: HBM would idle during the softmax. Each warp asks L2 for the first VPF x 8 ksteps (512 B each)
    // of the V range it will read in step 4 (dim tile w & 7, third w >> 3).
    const int s0 = (w >> 3) * nt / 3;
#pragma unroll
    for (int k = 0; k < VPF; k++) PREFETCH(Vc + (size_t)(s0 + 8 * k + (lane >> 2)) * KVT + (w & 7) * 32 + (lane & 3) * 8);
  }
#endif
  // ---- step 3: softmax per head; warp w: head w&3, slice w>>2 of 6 (the maximum comes from step 2)
  {
    enum { NSL = NWARP / 4 };          // slices of the position range per head
    const int h = w & 3, r = w >> 2;
    const float* s = sc + h * NTP;
    float mx = scr[h * NWARP];
#pragma unroll
    for (int i = 1; i < NWARP; i++) mx = fmaxf(mx, scr[h * NWARP + i]);
    float sum = 0.f;
SMU_
    for (int j = r + NSL * lane; j < NTP; j += NSL * 32) { float e = __expf(s[j] - mx); sc[h * NTP + j] = e; sum += e; }
    sum = warp_sum(sum);
    if (lane == 0) scr[100 + w] = sum;
    __syncthreads();
    float Z = scr[100 + h];
    for (int i = 1; i < NSL; i++) Z += scr[100 + h + 4 * i];
    if (w == 0 && nt >= LONG_NT) {   // sub-segment starts of step 4 as byte flags per kstep (one warp: zero, then set)
      unsigned char* fl = (unsigned char*)(smem + ATT_OFF_BYTES);
      ((uint4*)fl)[lane] = make_uint4(0, 0, 0, 0);
      if (nt > 512) ((uint4*)fl)[32 + lane] = make_uint4(0, 0, 0, 0);
      if (lane < 24) fl[lane * nt / 24] = 1;
    }
    if (nt < LONG_NT) {   // short contexts: as MMA B fragments behind the scores (one load per MMA in step 4)
      unsigned short* pfrag16 = (unsigned short*)(sc + 4 * NTP);   // [nt][32][4]
SMU_
      for (int j = r + NSL * lane; j < NTP; j += NSL * 32) {
        float p = s[j] / Z;
        pfrag16[(((j >> 4) * 32 + 4 * h + frag_t(j)) << 2) + frag_slot(j)] = f2bf(p);
      }
    } else {
      // long contexts: the exponentials stay where they are (16 B of shared memory per position); step 4 turns
      // them into bf16 probabilities as it builds its MMA operands: p = bf16(e * (1 / Z))
      if (r == 0 && lane == 0) scr[96 + h] = 1.0f / Z;
#if !ONFLY
      // ... or, the same values, in a pass of its own (bf16 in the low half of the slot)
      const float rz1 = 1.0f / Z;
      for (int j = r + NSL * lane; j < NTP; j += NSL * 32) ((unsigned*)sc)[h * NTP + j] = f2bf(s[j] * rz1);
#endif
    }
    __syncthreads();
    TP(6);
    ATS(2);
  }

  // ---- step 4: out[h][d] = sum_j p[h][j] V[j][d]; warp w: dim tile w&7, third w>>3 of the kstep range.
  // Fixed summation order, whatever the number of warps on a dim tile (the long-context path puts all 24 on one).
  // From LONG_NT position tiles on, the kstep range is cut into 24 sub-segments, each an MMA chain from zero; a
  // third is the sequential sum (((0+s0)+s1)+...)+s7 of its 8 sub-segments, the result (T0+T1)+T2. Below, a third
  // is a single chain.
  {
    enum { NSEG = NWARP / 8, VB = VBATCH };  // thirds per dim tile; V loads requested back to back
    const int dt = w & 7, r = w >> 3;
    const float2* pu = (const float2*)(sc + (lane >> 2) * NTP) + (lane & 3);   // lane = 4*head + t: e[head][16s + 2t (+8)], 2 positions per load
    const float rz = lane < 16 ? scr[96 + (lane >> 2)] : 0.f;                   // 1 / Z of the lane's head   // lane = 4*head + t: p[head][16s + 2t (+8)], 2 positions per load
    float4 tot = make_float4(0.f, 0.f, 0.f, 0.f);
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const unsigned char* fl = (const unsigned char*)(smem + ATT_OFF_BYTES);   // fl[s] != 0: a sub-segment starts at kstep s
    const int send = (r + 1) * nt / NSEG;
    // one kstep: (new sub-segment: add the finished chain to the third's running sum), B operand from the
    // in-place probabilities, MMA
#if ONFLY
#define VOPND_(ss) pfrag_e(pu + (ss) * 8, rz)
#else
#define VOPND_(ss) pfrag_u((const uint2*)pu + (ss) * 8)
#endif
#define VSTEP_(ss, aa) do { \
      if (fl[(ss)]) { tot = add4(tot, make_float4(c[0], c[1], c[2], c[3])); c[0] = 0.f; c[1] = 0.f; c[2] = 0.f; c[3] = 0.f; } \
      uint2 bp = make_uint2(0, 0); \
      if (lane < 16) bp = VOPND_(ss); \
      MMA(c, aa, bp); } while (0)
    int s = r * nt / NSEG;
    if (nt < LONG_NT) {
      // up to 1136 positions: a third is one chain (no sub-segments), one load at a time: every instruction counts
      const uint2* pfrag = (const uint2*)(sc + 4 * NTP);
#if VSB > 1
      // VSB V loads in flight per warp when the third is long enough (same MMA chain, same order: same bits)
      for (; s + VSB <= send; s += VSB) {
        uint4 a[VSB];
#pragma unroll
        for (int u = 0; u < VSB; u++) a[u] = __ldcg(Vc + (size_t)(s + u) * KVT + dt * 32 + lane);
#pragma unroll
        for (int u = 0; u < VSB; u++) {
          uint2 bp = make_uint2(0, 0);
          if (lane < 16) bp = pfrag[(s + u) * 32 + lane];
          MMA(c, a[u], bp);
        }
      }
#endif
      for (; s < send; s++) {
        const uint4 a = __ldcg(Vc + (size_t)s * KVT + dt * 32 + lane);
        uint2 bp = make_uint2(0, 0);
        if (lane < 16) bp = pfrag[s * 32 + lane];
        MMA(c, a, bp);
      }
    }
    for (; s + VB <= send; s += VB) {      // long ranges: VB loads in flight
      uint4 a[VB];
#pragma unroll
      for (int u = 0; u < VB; u++) a[u] = __ldcg(Vc + (size_t)(s + u) * KVT + dt * 32 + lane);
#pragma unroll
      for (int u = 0; u < VB; u++) VSTEP_(s + u, a[u]);
    }
    for (; s < send; s++) {
      const uint4 a = __ldcg(Vc + (size_t)s * KVT + dt * 32 + lane);
      VSTEP_(s, a);
    }
#undef VSTEP_
    tot = add4(tot, make_float4(c[0], c[1], c[2], c[3]));   // (empty sub-segments would add +0)
    c[0] = tot.x; c[1] = tot.y; c[2] = tot.z; c[3] = tot.w;
    red[w * 32 + lane] = make_float4(c[0], c[1], c[2], c[3]);
    __syncthreads();
    ATS(3);
    if (w < 8) {
      float4 v = red[w * 32 + lane];
      for (int i = 1; i < NSEG; i++) { float4 u = red[(w + 8 * i) * 32 + lane]; v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }
      int t = lane & 3, m = lane >> 2;
      if (t < 2) {
        int base0 = (4 * g + 2 * t) * 128 + dt * 16 + m, base1 = (4 * g + 2 * t + 1) * 128 + dt * 16 + m;
        P.afrag[gfrag_index(base0, n, KS_H, P.B)] = f2bf(v.x);
        P.afrag[gfrag_index(base1, n, KS_H, P.B)] = f2bf(v.y);
        P.afrag[gfrag_index(base0 + 8, n, KS_H, P.B)] = f2bf(v.z);
        P.afrag[gfrag_index(base1 + 8, n, KS_H, P.B)] = f2bf(v.w);
      }
      __threadfence();
    }
    __syncthreads();
  }
}


// ---------------------------------------------------------------------------------------------------------
// split attention, sub-phase 1: block b = unit * kb + chunk: q fragments of the unit, scores of its chunk of
// position tiles -> P.asc. The block with the last chunk also appends the new K and V rows.
// ---------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void att_k_chunk(unsigned char* smem, int layer) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const int kb = SPLIT_KB(P.B, NWORK);
  const int u = BIDX / kb, ck = BIDX - u * kb;
  const int n = u >> 3, g = u & 7;
  const int tid = TIDX, lane = tid & 31, w = tid >> 5;
  const int pos = P.pos, T = pos + 1, nt = (T + 15) >> 4;
  unsigned short* qfrag16 = (unsigned short*)(smem + ATT_OFF_BYTES + 3072);   // [8][32][4]
  const uint2* qfrag = (const uint2*)qfrag16;
  uint4* Kc = P.kc + KVU(P, layer, n * 8 + g);
  uint4* Vc = P.vc + KVU(P, layer, n * 8 + g);
  const size_t ctx_pad = P.kv_stride / 16;
  float* S = P.asc + (size_t)u * 4 * ctx_pad;
  const bool last = ck == kb - 1;

  // The chunk's K tiles: asked from L2 (4 KB per instruction) behind the prep's own loads, so that HBM works
  // while the prep computes and the score loads find them in L2.
#define KCHUNK_PREFETCH_() do { for (int tile_ = ck * nt / kb + w; tile_ < (ck + 1) * nt / kb; tile_ += NWARP) PREFETCH(Kc + (size_t)tile_ * KVT + lane * 8); } while (0)
  // q (4 heads) in every block of the unit; k and v of the new token in the block of the last chunk (see attention_body)
  if (w < 4 || (last && w < 6)) {
    const int vec = w;
    const int row = vec < 4 ? (4 * g + vec) * 128 : (vec == 4 ? 4096 + g * 128 : 5120 + g * 128);
    const float4 x = __ldcg((const float4*)(P.qkv + (size_t)n * QKV_ROWS + row) + lane);
    float xe[4] = {x.x, x.y, x.z, x.w};
    unsigned short ob[4];
    if (vec < 5) {
      const float4 wn = ((const float4*)(P.qk_norm_w + (layer * 2 + (vec < 4 ? 0 : 1)) * 128))[lane];
      const float4 cs = ((const float4*)(P.rope + (size_t)pos * 128))[lane & 15];
      const float4 sn = ((const float4*)(P.rope + (size_t)pos * 128 + 64))[lane & 15];
#if SPLIT_PF
      KCHUNK_PREFETCH_();
#endif
      const float ss = warp_sum((x.x * x.x + x.y * x.y) + (x.z * x.z + x.w * x.w));
      const float r = 1.0f / sqrtf(ss * (1.0f / 128) + 1e-6f);
      const float we[4] = {wn.x, wn.y, wn.z, wn.w}, ce[4] = {cs.x, cs.y, cs.z, cs.w}, se[4] = {sn.x, sn.y, sn.z, sn.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const float yv = RB(we[i] * RB(xe[i] * r));
        const float other = __shfl_xor_sync(0xffffffffu, yv, 16);
        ob[i] = f2bf(lane < 16 ? RB(yv * ce[i]) - RB(other * se[i]) : RB(yv * ce[i]) + RB(other * se[i]));
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) ob[i] = f2bf(xe[i]);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int d = 4 * lane + i;
      if (vec < 4) {
        qfrag16[(((d >> 4) * 32 + 4 * vec + frag_t(d)) << 2) + frag_slot(d)] = ob[i];
      } else if (vec == 4) {
        const int m = pos & 15, kcol = d & 15;
        const int ln = (m & 7) * 4 + ((kcol & 7) >> 1), el = (kcol >> 3) * 4 + (m >> 3) * 2 + (kcol & 1);
        ((unsigned short*)Kc)[(size_t)(pos >> 4) * KVT * 8 + ((((d >> 4) * 32 + ln) << 3) + el)] = ob[i];
      } else {
        const int kcol = pos & 15, m = d & 15;
        const int ln = (m & 7) * 4 + ((kcol & 7) >> 1), el = (kcol >> 3) * 4 + (m >> 3) * 2 + (kcol & 1);
        ((unsigned short*)Vc)[(size_t)(pos >> 4) * KVT * 8 + ((((d >> 4) * 32 + ln) << 3) + el)] = ob[i];
      }
    }
#if PREP_FENCE
    __threadfence();
#endif
  }
#if SPLIT_PF
  else KCHUNK_PREFETCH_();
  if (w == 5 && last) KCHUNK_PREFETCH_();
#endif
  __syncthreads();
  ATS(0);

  // scores[h][j] = q_h . K_j / sqrt(128) for the position tiles of this chunk (same MMA chain per tile as attention_body)
  const int c1 = (ck + 1) * nt / kb;
  for (int tile = ck * nt / kb + w; tile < c1; tile += NWARP) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const uint4* p = Kc + (size_t)tile * KVT + lane;
    uint4 a[8];
#pragma unroll
    for (int s = 0; s < 8; s++) a[s] = __ldcg(p + s * 32);
#pragma unroll
    for (int s = 0; s < 8; s++) {
      uint2 bq = make_uint2(0, 0);
      if (lane < 16) bq = qfrag[s * 32 + lane];
      MMA(c, a[s], bq);
    }
    const int t = lane & 3, j = tile * 16 + (lane >> 2);
    if (t < 2) {
      const float scale = SCORE_SCALE;
      const float ninf = -__int_as_float(0x7f800000);
      S[(2 * t) * ctx_pad + j] = j < T ? SCORE_(c[0]) : ninf;
      S[(2 * t + 1) * ctx_pad + j] = j < T ? SCORE_(c[1]) : ninf;
      S[(2 * t) * ctx_pad + j + 8] = j + 8 < T ? SCORE_(c[2]) : ninf;
      S[(2 * t + 1) * ctx_pad + j + 8] = j + 8 < T ? SCORE_(c[3]) : ninf;
    }
  }
  __threadfence();
  __syncthreads();   // every warp's scores are out before the coordinator thread enters the grid barrier
  ATS(1);
}

// ---------------------------------------------------------------------------------------------------------
// split attention, sub-phase 2: block b = unit * 8 + dim tile: scores of the unit from P.asc, softmax (the code
// of attention_body), then out[h][16 dims] = sum_j p[h][j] V[j][dims] with one warp per sub-segment and the
// partial sums combined in the fixed order of attention_body.
// ---------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void att_v_tile(unsigned char* smem, int layer) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const int vb = SPLIT_VB(P.B, NWORK);
  const int u = BIDX / vb, rv = BIDX - u * vb;      // this block: dim tiles [rv * 8 / vb, (rv + 1) * 8 / vb) of unit u
  const int n = u >> 3, g = u & 7;
  const int tid = TIDX, lane = tid & 31, w = tid >> 5;
  const int pos = P.pos, T = pos + 1, nt = (T + 15) >> 4, NTP = nt * 16;
  float* sc = (float*)(smem + ATT_OFF_BYTES + 5120);         // [4][NTP]
  float* scr = (float*)(smem + SCR_OFF);
  float4* red = (float4*)(smem + RED_OFF);
  const uint4* Vc = P.vc + KVU(P, layer, n * 8 + g);
  const size_t ctx_pad = P.kv_stride / 16;
  // Copy the unit's scores from global memory and take the maximum per head on the way (exact, whatever the
  // order). All loads of a thread are requested before the first store (one at a time, the copy was 8 round
  // trips: ~3 us at 9216 positions).
  {
    float mh[4] = {-__int_as_float(0x7f800000), -__int_as_float(0x7f800000), -__int_as_float(0x7f800000), -__int_as_float(0x7f800000)};
    const int q = NTP / 4;                                    // uint4 per head (at most 4 per thread)
#pragma unroll
    for (int hp = 0; hp < 4; hp += 2) {
      uint4 v[8];
#pragma unroll
      for (int a = 0; a < 8; a++) {
        const int h = hp + (a >> 2), i = tid + (a & 3) * NTHREAD;
        v[a] = make_uint4(0, 0, 0, 0);
        if (i < q) v[a] = __ldcg((const uint4*)(P.asc + ((size_t)u * 4 + h) * ctx_pad) + i);
      }
#pragma unroll
      for (int a = 0; a < 8; a++) {
        const int h = hp + (a >> 2), i = tid + (a & 3) * NTHREAD;
        if (i < q) {
          ((uint4*)(sc + h * NTP))[i] = v[a];
          mh[h] = fmaxf(mh[h], fmaxf(fmaxf(__uint_as_float(v[a].x), __uint_as_float(v[a].y)), fmaxf(__uint_as_float(v[a].z), __uint_as_float(v[a].w))));
        }
      }
    }
#pragma unroll
    for (int h = 0; h < 4; h++) {
      const float m = warp_max(mh[h]);
      if (lane == 0) scr[h * NWARP + w] = m;
    }
  }
  __syncthreads();
  ATSP(2);
#if VPF > 0
  {   // the start of this warp's V range (sub-segment w of dim tile rv), as in attention_body
    const int s0 = w * nt / 24;
#pragma unroll
    for (int k = 0; k < VPF; k++) PREFETCH(Vc + ((size_t)(s0 + 8 * k + (lane >> 2)) * 8 + rv) * 32 + (lane & 3) * 8);
  }
#endif
  // softmax per head; warp w: head w&3, slice w>>2 (as step 3 of attention_body for long contexts; the maximum
  // comes from the copy)
  {
    enum { NSL = NWARP / 4 };
    const int h = w & 3, r = w >> 2;
    const float* s = sc + h * NTP;
    float mx = scr[h * NWARP];
#pragma unroll
    for (int i = 1; i < NWARP; i++) mx = fmaxf(mx, scr[h * NWARP + i]);
    float sum = 0.f;
SMU_
    for (int j = r + NSL * lane; j < NTP; j += NSL * 32) { float e = __expf(s[j] - mx); sc[h * NTP + j] = e; sum += e; }
    sum = warp_sum(sum);
    if (lane == 0) scr[100 + w] = sum;
    __syncthreads();
    float Z = scr[100 + h];
    for (int i = 1; i < NSL; i++) Z += scr[100 + h + 4 * i];
    if (r == 0 && lane == 0) scr[96 + h] = 1.0f / Z;   // the V pass builds p = bf16(e * (1 / Z)) (see pfrag_e)
    __syncthreads();
  }
  ATSP(3);
  // V: warp w = sub-segment w (one MMA chain from zero)
  {
    const int dt = rv;                                        // (vb == 8: one dim tile per block)
    enum { VB = VBATCH_S };
    const float2* pu = (const float2*)(sc + (lane >> 2) * NTP) + (lane & 3);   // lane = 4*head + t: e[head][16s + 2t (+8)], 2 positions per load
    const float rz = lane < 16 ? scr[96 + (lane >> 2)] : 0.f;                   // 1 / Z of the lane's head
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const int send = (w + 1) * nt / 24;
    int s = w * nt / 24;
#define VSTEP_(ss, aa) do { \
      uint2 bp = make_uint2(0, 0); \
      if (lane < 16) bp = pfrag_e(pu + (ss) * 8, rz); \
      MMA(c, aa, bp); } while (0)
    for (; s + VB <= send; s += VB) {
      uint4 a[VB];
#pragma unroll
      for (int i = 0; i < VB; i++) a[i] = __ldcg(Vc + (size_t)(s + i) * KVT + dt * 32 + lane);
#pragma unroll
      for (int i = 0; i < VB; i++) VSTEP_(s + i, a[i]);
    }
    for (; s < send; s++) {
      const uint4 a = __ldcg(Vc + (size_t)s * KVT + dt * 32 + lane);
      VSTEP_(s, a);
    }
#undef VSTEP_
    red[w * 32 + lane] = make_float4(c[0], c[1], c[2], c[3]);
    __syncthreads();
    if (w == 0) {
      // (T0 + T1) + T2, T_k = (((0 + s_8k) + s_8k+1) + ...) + s_8k+7
      float4 tk[3];
#pragma unroll
      for (int k = 0; k < 3; k++) {
        float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int i = 0; i < 8; i++) t = add4(t, red[(8 * k + i) * 32 + lane]);
        tk[k] = t;
      }
      const float4 v = add4(add4(tk[0], tk[1]), tk[2]);
      const int t = lane & 3, m = lane >> 2;
      if (t < 2) {
        const int base0 = (4 * g + 2 * t) * 128 + dt * 16 + m, base1 = (4 * g + 2 * t + 1) * 128 + dt * 16 + m;
        P.afrag[gfrag_index(base0, n, KS_H, P.B)] = f2bf(v.x);
        P.afrag[gfrag_index(base1, n, KS_H, P.B)] = f2bf(v.y);
        P.afrag[gfrag_index(base0 + 8, n, KS_H, P.B)] = f2bf(v.z);
        P.afrag[gfrag_index(base1 + 8, n, KS_H, P.B)] = f2bf(v.w);
      }
      __threadfence();
    }
    __syncthreads();
  }
}


#ifdef TRACK_A
// Track A, 3+ sequences: RMSNorm as a phase of its own (the loaders read the matvec input from global memory, so
// the normalised fragments must be there). Every block computes 1/rms per sequence (one warp each) and
// normalises its slice of the 4096 fragment entries (16 B each: 8 elements), then a grid barrier.
__device__ __forceinline__ void norm_phase(unsigned char* smem) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const volatile Desc& D = *(const volatile Desc*)(smem + DESC_OFF);
  const int tid = TIDX, lane = tid & 31, w = tid >> 5;
  {
    const int n = w - NORM_WARP0;
    if (n >= 0 && n < P.B) {
      const float r = rms_inv(P, n, lane);
      if (lane == 0) ((float*)(smem + RN_OFF))[n] = r;
    }
  }
  __syncthreads();
  if (w == 0) {
    const int nb = NWORK, b = BIDX;
    const int i = (int)((long long)b * (H / 8 * 8) / nb) + lane, i1 = (int)((long long)(b + 1) * (H / 8 * 8) / nb);
    if (i < i1) {
      const int fl = i & 31, n = fl >> 2, t = fl & 3;        // wide layout: entry = (group * 2 + j) * 32 + 4 * sequence + t
      if (n < P.B) {
        const uint4 x = __ldcg((const uint4*)P.xraw + i), wv = __ldcg((const uint4*)D.wfrag + (i >> 5) * 4 + t);
        ((uint4*)P.xfrag)[i] = norm8(x, wv, ((const float*)(smem + RN_OFF))[n]);
      }
    }
    __threadfence();
  }
  __syncthreads();
  if (tid == CW * 32) {
    ((Desc*)(smem + DESC_OFF))->k = D.kafter;
    grid_barrier(smem, PH_EMBED TR_ARG(-1));
  }
  __syncthreads();
}
#define NORM_BARRIERS(B, layers) ((B) >= WIDE_MIN_B && !NORM_LOAD ? 2 * (layers) + 1 : 0)
#else
#define NORM_BARRIERS(B, layers) 0
#endif

__device__ __forceinline__ void kernel_init(unsigned char* smem, const Params& Pk, int lm) {
  uint4* sx4 = (uint4*)(smem + SX_OFF);
  for (int i = TIDX; i < Pk.sx_bytes / 16; i += NTHREAD) sx4[i] = make_uint4(0, 0, 0, 0);
  const unsigned* src = (const unsigned*)&Pk;
  if (TIDX < sizeof(Params) / 4) ((unsigned*)(smem + MISC_OFF))[TIDX] = src[TIDX];
  if (TIDX < 8) ((int*)(smem + TB_OFF))[TIDX] = Pk.tiles[(Pk.B >= WIDE_MIN_B ? Pk.nwork * 8 : 0) + BIDX * 8 + TIDX];
  if (TIDX < 4) ((int*)(smem + CTL_OFF))[TIDX] = 0;
  if (TIDX < 2) ((int*)(smem + TBO_OFF))[TIDX] = Pk.tiles_o[((Pk.B - 1) * Pk.nwork + BIDX) * 2 + TIDX];
  __syncthreads();
  if (TIDX == 0) { ((Ctl*)(smem + CTL_OFF))->lm = lm; ((Desc*)(smem + DESC_OFF))->k = PH_EMBED; }
  __syncthreads();
}

// Split path, sub-phase 2: the V blocks copy the scores and run the softmax for ~7 us before their first V
// load, and HBM would carry only the o stash meanwhile. The idle blocks pull the units' V history into L2
// (8 loads of 512 B per warp at a time, at most TOUCH_VS rounds: with 2 sequences there are too few idle
// blocks), so that the V pass runs on L2 hits.
__device__ __forceinline__ void touch_v_split(unsigned char* smem, int layer, int natt) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const int lane = TIDX & 31, w = TIDX >> 5;
  const int per_unit = 8 * ((P.pos + 16) >> 4), total = 8 * P.B * per_unit;
  const int nidle = (NWORK - natt) * NLOAD;
  if (total > nidle * 8 * TOUCH_VS) return;
  unsigned acc = 0;
  for (int c = (BIDX - natt) * NLOAD + w; c < total; c += 8 * nidle) {
    uint4 v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int cc = c + k * nidle;
      v[k] = make_uint4(0, 0, 0, 0);
      if (cc < total) { const int u = cc / per_unit, i = cc - u * per_unit; v[k] = __ldcg(P.vc + KVU(P, layer, u) + (size_t)(i >> 3) * KVT + (i & 7) * 32 + lane); }
    }
#pragma unroll
    for (int k = 0; k < 8; k++) acc ^= v[k].x;
  }
  if (acc == 0x9e3779b9u) ((unsigned*)(smem + SCR_OFF))[lane] = acc;   // keeps the loads alive
}

// block barrier + (coordinator thread) grid barrier, for the phases that are not matvecs
__device__ __forceinline__ void barrier(unsigned char* smem TR_PARAM) {
  if (TIDX == CW * 32) grid_barrier(smem, PH_ATT TR_ARG(tr));
  __syncthreads();
}

// Attention phase. While the 8B attention blocks work, every other SM would idle, so its loader warps stream
// their share of the block's first o tile into shared memory (the 131 KB behind the K=4096 input fragments,
// free outside the down projection); those bytes cross HBM during attention and the o matvec takes its first
// tile from there. Attention blocks own a single o tile (tile table), the others share the rest.
#define STG_GU_BLOCK_() do { \
      if (w == 0 && P.B <= STG_MAX_B) { \
        enum { PIECES = (int)(TU4(KS_H) / 1024) }; \
        const int* tb = (const int*)(smem + TB_OFF); \
        const int ng = tb[2 * T_GU + 1] - tb[2 * T_GU]; \
        if (STG_O && D.t0 + 1 < D.t1 && lane < PIECES) l2_stage(D.W + (size_t)(D.t0 + 1) * TU4(KS_H) + lane * 1024, 16384); \
        if (STG_GU > 0 && ng > 1) { \
          for (int i = lane; i < (STG_NP > 0 ? STG_NP : STG_GU * PIECES); i += 32) { \
            const int j = i / PIECES, pc = i - j * PIECES; \
            const int ti = STG_GU_A > 0 ? STG_GU_A + j * STG_GU_S : (j + 1) * ng / (STG_GU + 1); \
            if (ti > 0 && ti < ng) l2_stage(D.nextW + (size_t)ti * TU4(KS_H) + pc * 1024, 16384); \
          } \
        } \
      } \
  } while (0)
__device__ __forceinline__ void att_phase(unsigned char* smem) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const Desc& D = *(const Desc*)(smem + DESC_OFF);   // the o projection of this layer
#ifdef TRACE
  int tr = (D.l == TRACE_LAYER) ? PH_ATT : -100;
#endif
  TP(0);
  TS_DECL;
  TS();                                   // 0: start
  const bool split = SPLIT_ON(P.B, P.pos);
  const int natt = split ? 8 * P.B * SPLIT_VB(P.B, NWORK) : 8 * P.B;   // blocks that end the phase with a cold o tile
  if (split) {
    if (BIDX < 8 * P.B * SPLIT_KB(P.B, NWORK)) att_k_chunk(smem, D.l);
    else __syncthreads();
    barrier(smem TR_ARG(tr));             // all scores are in global memory
  }
  if (BIDX < natt) {
    if (split) att_v_tile(smem, D.l);
    else attention_body(smem, D.l TR_ARG(tr));
    {   // this block's o tile is cold: nothing stashed (this reset was inside the NPF_ATT block before: with
        // NPF_ATT 0 the o tile loop read a stale count and stale stash entries - found by bitcheck, e0139)
      const int lane = TIDX & 31, w = TIDX >> 5;
      if (w < NLOAD) {
#if NPF_ATT > 0
        // ... and ask L2 for its first ksteps now
        const uint4* p = D.W + (size_t)D.t0 * TU4(KS_H) + G0(w, KS_H / 4) * GU4 + lane;
#pragma unroll
        for (int u = 0; u < NPF_ATT; u++) if (u < (KS_H / 4 / NLOAD) * LPG || w < (KS_H / 4) % NLOAD) PREFETCH(p + u * 32);
#endif
        if (lane == 0) ((int*)(smem + CNT_OFF))[w] = 0;
      }
    }
  } else {
    const int lane = TIDX & 31, w = TIDX >> 5;
    enum { KG = KS_H / 4, BASE = KG / NLOAD, EXTRA = KG % NLOAD };
    if (w < NLOAD) {
      // First pull this layer's K/V history into L2 for the attention blocks: their loads would otherwise
      // be HBM misses queued behind the weight flood below. 512-byte chunks, spread over all idle loader warps.
#if STG_OA
      // the attention blocks end with a cold o tile, the last work of the phase: idle block 8B + j asks L2 for the o
      // tile(s) of attention block j right away (warp 1, one 16 KB piece per lane)
      if (!split && w == 1 && P.B <= STG_MAX_B && BIDX < 16 * P.B) {
        const int j = BIDX - 8 * P.B, row = P.B - 1;
        const int a0 = P.tiles_o[(row * P.nwork + j) * 2], a1 = P.tiles_o[(row * P.nwork + j) * 2 + 1];
        enum { PIECES = (int)(TU4(KS_H) / 1024) };
        if (lane < (a1 - a0) * PIECES) l2_stage(D.W + (size_t)a0 * TU4(KS_H) + (size_t)lane * 1024, 16384);
      }
#endif
      if (!split && !KV_PF && TOUCH) {
        // (only while all of it fits in L2, TOUCH_TILES position tiles x 8 sequences: beyond that it would be read twice;
        // measured with 8 sequences at 9000 positions: 8456 us per step with a partial touch, 8298 without)
        // Long contexts (TOUCH_KL > 0): only the K tiles the attention blocks read first, TOUCH_KL / B per unit: HBM
        // would idle while they prepare q.
        int kt = (P.pos + 16) >> 4, vt = kt;
        if (kt * P.B > TOUCH_TILES) { kt = TOUCH_KL / P.B; vt = 0; }
        const int per_unit = 8 * (kt + vt), total = 8 * P.B * per_unit;
        const int nidle = (NWORK - 8 * P.B) * NLOAD;
        unsigned acc = 0;
#if TOUCH_ORD
        if (P.B <= TOUCH_ORD_MAX_B) {
          // K of every unit first (the score pass reads it first; unit by unit, a late unit's K came after the
          // V of the early ones), the units interleaved, and TOUCH_FL chunks in flight per warp instead of one.
          const int nu = 8 * P.B, nk = nu * 8 * kt;
          for (int c = (BIDX - 8 * P.B) * NLOAD + w; c < total; c += TOUCH_FL * nidle) {
            unsigned v[TOUCH_FL];
#pragma unroll
            for (int f = 0; f < TOUCH_FL; f++) {
              const int cc = c + f * nidle;
              v[f] = 0;
              if (cc < total) {
                const bool isk = cc < nk;
                const int j = isk ? cc : cc - nk;
                const int u = j % nu, ci = j / nu;
                v[f] = __ldcg((isk ? P.kc : P.vc) + KVU(P, D.l, u) + (size_t)(ci >> 3) * KVT + (ci & 7) * 32 + lane).x;
              }
            }
#pragma unroll
            for (int f = 0; f < TOUCH_FL; f++) acc ^= v[f];
          }
        } else
#endif
        for (int c = (BIDX - 8 * P.B) * NLOAD + w; c < total; c += nidle) {
          const int u = c / per_unit, i = c - u * per_unit;
          const uint4* base = (i < 8 * kt ? P.kc : P.vc) + KVU(P, D.l, u);
          const int ci = i < 8 * kt ? i : i - 8 * kt;
          acc ^= __ldcg(base + (size_t)(ci >> 3) * KVT + (ci & 7) * 32 + lane).x;
        }
        if (acc == 0x9e3779b9u) ((unsigned*)(smem + SCR_OFF))[lane] = acc;   // keeps the loads alive
      }
#if TOUCH_VS > 0 && !TOUCH_VS_AFTER
      if (split) touch_v_split(smem, D.l, natt);
#endif
#if ATT_WAIT
      // The stash floods HBM, and while it does the attention blocks' K loads (L2 hits) take 1.8 us longer. So the
      // stash starts when every attention block has stamped its score pass as done (a hint only: a stale or late
      // stamp changes the timing, not the result; the number of polls is bounded).
      if (!split && P.B <= ATT_WAIT_MAX_B) {
        const int stamp = (P.pos << 6) | (D.l + 1), na = 8 * P.B;
        for (int it = 0; it < ATT_WAIT; it++) {
          int ok = 1;
          if (lane < na) ok = __ldcg(P.att_done + lane) == stamp;
          if (na > 32 && lane + 32 < na) ok &= __ldcg(P.att_done + lane + 32) == stamp;
          if (__all_sync(0xffffffffu, ok)) break;
        }
      }
#endif
#if STG_GU > 0 || STG_O
      if (STG_EARLY) STG_GU_BLOCK_();   // (variant: staging requested before the o stash)
#endif
      stash_loads(smem, D.W + (size_t)D.t0 * TU4(KS_H), KS_H / 4, STASH_OFF, false, ATT_STASH_CAP, false, P.B <= ATT_WAIT_MAX_B ? ATT_POLITE : 0);
#if TOUCH_VS > 0 && TOUCH_VS_AFTER
      if (split) touch_v_split(smem, D.l, natt);
#endif
      const uint4* p = D.W + (size_t)D.t0 * TU4(KS_H) + (w * BASE + min(w, (int)EXTRA)) * GU4 + lane;
#if NPF_ST > 0
      if (D.t0 + 1 < D.t1) {   // and the start of this block's second o tile into L2 (6+ sequences: all of it)
        const int npf = P.B >= 6 ? NPF_STW : NPF_ST;
#pragma unroll
        for (int u = 0; u < (NPF_STW > NPF_ST ? NPF_STW : NPF_ST); u++) if (u < npf && (u < BASE * LPG || w < EXTRA)) PREFETCH(p + TU4(KS_H) + u * 32);
      }
#endif
#if NPF_GU > 0
      // ... and of its first gate+up tile: those weights do not depend on attention either, and HBM is not
      // saturated during this phase
      if (D.nb) {
        const uint4* pg = D.nextW + (w * D.nb + min(w, D.ne)) * GU4 + lane;
#pragma unroll
        for (int u = 0; u < NPF_GU; u++) PREFETCH(pg + u * 32);
      }
#endif
#if STG_GU > 0 || STG_O
      if (!STG_EARLY) STG_GU_BLOCK_();
#if STG_DN > 0
      // variant: stage pieces of the block's down tile STG_DN_TI instead of (or besides) gate+up
      if (w == 3 && P.B <= STG_MAX_B) {
        const int* tb = (const int*)(smem + TB_OFF);
        if (tb[2 * T_DN + 1] - tb[2 * T_DN] > STG_DN_TI) {
          const uint4* bd = D.W - OFF_O + OFF_DN + ((size_t)tb[2 * T_DN] + STG_DN_TI) * TU4(KS_I);
          for (int i = lane; i < STG_DN; i += 32) l2_stage(bd + (size_t)(STG_DN_OFF + i) * 1024, 16384);
        }
      }
#endif
#if STG_GUA
      // ... and the same pieces of the gate+up tiles of attention block j = BIDX - 8B (busy now, it cannot stage its
      // own; without this the attention blocks were the last ones of every gate+up phase: tests/timing_blocks.py)
      if (STG_GU > 0 && w == 2 && P.B <= STG_MAX_B && BIDX < 16 * P.B) {
        enum { PIECES = (int)(TU4(KS_H) / 1024) };
        const int* tb = (const int*)(smem + TB_OFF);
        const int j = BIDX - 8 * P.B;
        const int* tj = P.tiles + (P.B >= WIDE_MIN_B ? P.nwork * 8 : 0) + j * 8;
        const int t0j = tj[2 * T_GU], ngj = tj[2 * T_GU + 1] - t0j;
        const uint4* bj = D.nextW + ((long long)t0j - tb[2 * T_GU]) * (long long)TU4(KS_H);
        if (ngj > 1) {
          for (int i = lane; i < (STG_NP > 0 ? STG_NP : STG_GU * PIECES); i += 32) {
            const int jj = i / PIECES, pc = i - jj * PIECES;
            const int ti = (jj + 1) * ngj / (STG_GU + 1);
            if (ti > 0 && ti < ngj) l2_stage(bj + (size_t)ti * TU4(KS_H) + pc * 1024, 16384);
          }
        }
      }
#endif
#endif
    }
  }
#ifndef NO_ATT_TT
  if ((TIDX >> 5) == CW) PREFETCH(D.tt + (size_t)D.t0 * TT + ((TIDX & 31) >> 4) * 8);   // table entry of the first o tile
#endif
  TP(3);
#if QKV_NOWAIT
  // (blocks that did not wait at the end of q/k/v: count in for this barrier only once that one is complete,
  // otherwise this increment could complete it for the blocks still waiting there)
  if (QKV_NW(P.B, P.pos) && BIDX >= 8 * P.B && TIDX == CW * 32) {
    const unsigned prev = P.sync_base + ((const Ctl*)(smem + CTL_OFF))->bi * (unsigned)NWORK;
    while ((int)(*(volatile unsigned*)P.sync - prev) < 0) { }
  }
#endif
#ifdef TIMING
  TS();                                   // 1: attention / stash done (this warp)
  if (TIDX == CW * 32) grid_barrier(smem, PH_ATT TR_ARG(tr));
  TS();                                   // 2: all blocks arrived (coordinator thread)
  if ((TIDX >> 5) == CW) TS_FLUSH(PH_ATT);
  __syncthreads();
#else
  barrier(smem TR_ARG(tr));
#endif
}

// embedding of the step's input token; generate(): token = argmax of the previous step (see op_embed), and
// after the last step only the argmax
__device__ __forceinline__ void embed_phase(unsigned char* smem) {
  const Params& P = *(const Params*)(smem + MISC_OFF);
  const Ctl& C = *(const Ctl*)(smem + CTL_OFF);
#ifdef TRACE
  int tr = -100;
#endif
  const bool gen = C.lm == PH_LMARG;
  const int gs = gen ? C.s : -1;
  const bool embed = gen ? (gs < P.n_steps) : true;
  op_embed(smem, gs, embed);
  const bool l0t = P.l0tab && !P.l0build;
  if (embed && !l0t && (TIDX >> 5) < NLOAD) {
    const uint4* tile = P.W + OFF_QKV + (size_t)((const int*)(smem + TB_OFF))[2 * T_QKV] * TU4(KS_H);
#if STASH_ALL
    stash_loads(smem, tile, KS_H / 4, STASH_OFF, false, 1000, false);   // layer 0's first q/k/v tile of this block
#else
    const int lane = TIDX & 31, w = TIDX >> 5;
#pragma unroll
    for (int u = 0; u < NPF_EMB; u++) PREFETCH(tile + G0(w, KS_H / 4) * GU4 + lane + u * 32);
#endif
  }
  if (TIDX == CW * 32) {
    if (embed) {   // the o tile ranges depend on which blocks end the attention phase with a cold tile
      const int row = SPLIT_ON(P.B, P.pos) ? MAXB + P.B - 1 : P.B - 1;
      int* tbo = (int*)(smem + TBO_OFF);
      tbo[0] = P.tiles_o[(row * P.nwork + BIDX) * 2]; tbo[1] = P.tiles_o[(row * P.nwork + BIDX) * 2 + 1];
    }
    if (embed) { set_desc(smem, l0t ? PH_O : PH_QKV, 0); grid_barrier(smem, PH_EMBED TR_ARG(tr)); }
    else ((Desc*)(smem + DESC_OFF))->k = PH_DONE;
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------------------------------------
// The kernel: an interpreter of phase descriptors. step(): one decode step, logits out (lm = PH_LM).
// generate(): n_steps greedy steps; argmax, token feedback and the loop are on the device (lm = PH_LMARG).
// The loop keeps nothing in registers from one phase to the next and makes no calls: a value kept across a
// call (and the return address of a register-hungry callee) is spilled to local memory by all 1024 threads.
// ---------------------------------------------------------------------------------------------------------
template <bool SK>   // SK: greedy head through the certified int8 sketch (k_gensk)
__device__ __forceinline__ void run(unsigned char* smem, const Params& Pk, int lm) {
  kernel_init(smem, Pk, lm);
  for (;;) {
    const int k = ((const volatile Desc*)(smem + DESC_OFF))->k;
    if (k == PH_DONE) break;
    if (k == PH_EMBED) { embed_phase(smem); continue; }
#ifndef TRACK_A
    if (SK && k == PH_LMARG && ((const Params*)(smem + MISC_OFF))->logits) { sk1_phase(smem); continue; }   // (null: diagnosis, exact head in this kernel)
    if (SK && k == PH_SK2) { sk2_phase(smem); continue; }
#endif
#ifdef TRACK_A
    if (k == PH_NORM) { norm_phase(smem); continue; }
#endif
    if (k == PH_O) att_phase(smem);
    if ((TIDX >> 5) < NLOAD) loader_phase<SK ? SK_LITE : 0>(smem); else coord_phase<SK>(smem);
  }
}

__global__ void k_step(Params Pk) {
  extern __shared__ __align__(16) unsigned char smem[];
#ifdef TRACE
  if (blockIdx.x == Pk.nwork) {   // the clock
    if (threadIdx.x == 0) {
      const unsigned stop = Pk.sync_base + (unsigned)(BARRIERS_PER_STEP + 1) * Pk.nwork;
      unsigned c = 0;
      for (;;) {
        c++;
        *(volatile unsigned*)(Pk.trace + 64 * 256) = c;
        if ((c & 255u) == 0 && (int)(*(volatile unsigned*)Pk.sync - stop) >= 0) break;
      }
    }
    return;
  }
#endif
  run<false>(smem, Pk, PH_LM);
#ifdef TRACE
  if (threadIdx.x == 0) atomicAdd(((const Params*)(smem + MISC_OFF))->sync, 1u);   // lets the clock block stop
#endif
}

__global__ void k_generate(Params Pk) {
  extern __shared__ __align__(16) unsigned char smem[];
  run<false>(smem, Pk, PH_LMARG);
}

__global__ void k_gensk(Params Pk) {
  extern __shared__ __align__(16) unsigned char smem[];
  run<true>(smem, Pk, PH_LMARG);
}

// ---------------------------------------------------------------------------------------------------------
// host API
// ---------------------------------------------------------------------------------------------------------
struct Engine {
  Params P;
  int nsm, max_batch, max_ctx;
  int sx_bytes;      // sx area of a block's shared memory (grows with max_ctx beyond 8000 positions)
  long long launches, steps, errors;
  int sk_mode;       // generate()/online: 0 exact head, 1 int8 sketch + exact rescoring, 2 the same with every tile rescored (test)
  SkArgs* sk_dev;
  int* sk_lrun;      // SkArgs::lrun (host copy of the pointer)    // device copy of the sketch's arguments (passed to k_gensk as Params.logits)
};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)

extern "C" {

void* mk_create(int max_batch, int max_ctx, const void* W, const void* embed, const void* norm_w, const void* qk_norm_w, const void* rope, const void* ttab, const void* exc, const void* wfrag) {
  if (max_batch > MAXB || max_ctx > MAX_CTX_SMEM) { fprintf(stderr, "mk_create: unsupported max_batch %d / max_ctx %d\n", max_batch, max_ctx); return nullptr; }
  Engine* E = new Engine();
  memset(E, 0, sizeof(Engine));
  int dev = 0; CK(cudaGetDevice(&dev));
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  E->nsm = prop.multiProcessorCount;
  E->max_batch = max_batch; E->max_ctx = max_ctx;
  E->sx_bytes = ATT_BYTES(max_ctx) > SX_NARROW ? (ATT_BYTES(max_ctx) + 255) / 256 * 256 : SX_NARROW;
  Params& P = E->P;
  P.W = (const uint4*)W; P.embed = (const unsigned short*)embed; P.norm_w = (const float*)norm_w;
  P.qk_norm_w = (const float*)qk_norm_w; P.rope = (const float*)rope;
  P.ttab = (const uint4*)ttab; P.exc = (const uint4*)exc; P.wfrag = (const unsigned short*)wfrag;
  size_t ctx_tiles = (max_ctx + 15) / 16 + KV_PAD;
  P.kv_stride = ctx_tiles * 256;
  size_t kv_bytes = (size_t)NL * MAXB * 8 * P.kv_stride * 16;
  CK(cudaMalloc(&P.xres, sizeof(float) * MAXB * H));
  CK(cudaMalloc(&P.qkv, sizeof(float) * MAXB * QKV_ROWS));
  CK(cudaMalloc(&P.afrag, 2 * MAXB * H));
  CK(cudaMalloc(&P.mfrag, 2 * MAXB * INNER));
  CK(cudaMalloc(&P.xfrag, 2 * MAXB * H));
  CK(cudaMalloc(&P.xraw, 2 * MAXB * H)); CK(cudaMemset(P.xraw, 0, 2 * MAXB * H));
  CK(cudaMalloc(&P.ssq, sizeof(float) * MAXB * 256));
  CK(cudaMemset(P.xfrag, 0, 2 * MAXB * H));
  CK(cudaMemset(P.ssq, 0, sizeof(float) * MAXB * 256));
  CK(cudaMalloc(&P.kc, kv_bytes));
  CK(cudaMalloc(&P.vc, kv_bytes));
  CK(cudaMalloc(&P.att_done, sizeof(int) * 8 * MAXB)); CK(cudaMemset(P.att_done, 0, sizeof(int) * 8 * MAXB));
  CK(cudaMalloc(&P.asc, sizeof(float) * 8 * MAXB * 4 * ctx_tiles * 16)); CK(cudaMemset(P.asc, 0, sizeof(float) * 8 * MAXB * 4 * ctx_tiles * 16));
  CK(cudaMalloc(&P.sync, 64));
  CK(cudaMalloc(&P.wstat, 4 * 9 * 256)); CK(cudaMemset(P.wstat, 0, 4 * 9 * 256));
#ifdef TIMING
  CK(cudaMalloc(&P.tstat, 4 * 256 * 200 * 8)); CK(cudaMemset(P.tstat, 0, 4 * 256 * 200 * 8));
#endif
  CK(cudaMalloc(&P.bm_val, sizeof(float) * MAXB * E->nsm));
  CK(cudaMalloc(&P.bm_idx, sizeof(int) * MAXB * E->nsm));
  {
#ifdef TRACE
    int nb = E->nsm - 1;
#else
    int nb = E->nsm;
#endif
    P.nwork = nb;
    P.stop_layer = NL;
    CK(cudaMalloc(&P.trace, sizeof(unsigned) * (64 * 256 + 16))); CK(cudaMemset(P.trace, 0, sizeof(unsigned) * (64 * 256 + 16)));
    int* tb = new int[2 * nb * 8];
    const int nt[4] = {QKV_T, O_T, GU_T, LM_T};
    // NB_MAT > 0: q/k/v, gate+up and down tiles only on the first NB_MAT blocks (128 divides all three tile
    // counts: no block has one tile more than another); the others get empty ranges. Two tables: 1-2 sequences
    // (NB_MAT) and 3+ (NB_MAT_W).
    for (int w = 0; w < 2; w++) for (int b = 0; b < nb; b++) for (int k = 0; k < 4; k++) {
      const int nm = w ? NB_MAT_W : NB_MAT;
      const int m = nm > 0 && nm < nb && k != 3 ? nm : nb;
      const int bb = b < m ? b : m;
      int* e = tb + w * nb * 8 + b * 8 + 2 * k;
      e[0] = (int)((long long)bb * nt[k] / m); e[1] = (int)((long long)(b < m ? b + 1 : m) * nt[k] / m);
      if (b < m && e[1] <= e[0]) { fprintf(stderr, "mk_create: block without tiles\n"); abort(); }
    }
#if GU_SHIFT > 0
    // 1-2 sequences (narrow table): the attention blocks 0..7 start gate+up last (cold o tile, short stash) and were
    // the last blocks of the phase; GU_SHIFT of their tiles each go to the blocks that hold no gate+up tiles (NB_MAT)
    if (NB_MAT > 0 && NB_MAT + 4 <= nb && nb - NB_MAT >= 1 && (8 * GU_SHIFT) % (nb - NB_MAT) == 0) {
      const int extra = nb - NB_MAT, per = 12 - GU_SHIFT, give = 8 * GU_SHIFT / extra;
      int t = 0;
      for (int b = 0; b < nb; b++) {
        const int c = b < 8 ? per : b < NB_MAT ? GU_T / NB_MAT : give;
        int* e = tb + b * 8 + 2 * T_GU;
        e[0] = t; e[1] = t + c; t += c;
      }
      if (t != GU_T) { fprintf(stderr, "mk_create: GU_SHIFT table\n"); abort(); }
    }
#endif
    int* d; CK(cudaMalloc(&d, sizeof(int) * 2 * nb * 8)); CK(cudaMemcpy(d, tb, sizeof(int) * 2 * nb * 8, cudaMemcpyHostToDevice));
    P.tiles = d; delete[] tb;
    // o projection: blocks 0..8B-1 are busy with attention while the others stream their first o tile into shared
    // memory, so the attention blocks own one o tile each and the other blocks share the rest evenly
    int* to = new int[2 * MAXB * nb * 2];
    for (int R = 1; R <= 2 * MAXB; R++) {
      const int B = R;                              // row R-1; rows MAXB.. are split attention with R-MAXB sequences
      int A = R <= MAXB ? 8 * B : 8 * (R - MAXB) * SPLIT_VB(R - MAXB, nb);
      if (A > nb - 1) A = nb - 1;
      const int N = nb - A;
      int cnt[1024];
      // attention blocks: one cold tile each - or two, when the other blocks would otherwise be left with more
      // than O_ATT2 / 10 tiles each (8 sequences: 2.8; they then run two cold tiles in a row while 64 SMs wait)
      const int a = (O_T - A) * 10 > O_ATT2 * N && 2 * A < O_T ? 2 : 1;
      int rest = O_T - a * A;
      for (int b = 0; b < A; b++) cnt[b] = a;
      for (int b = A; b < nb; b++) cnt[b] = (int)((long long)(b - A + 1) * rest / N) - (int)((long long)(b - A) * rest / N);
      if (rest > 3 * N)   // (nearly) every block runs attention: no point in favouring the few others
        for (int b = 0; b < nb; b++) cnt[b] = (int)((long long)(b + 1) * O_T / nb) - (int)((long long)b * O_T / nb);
      int t = 0;
      for (int b = 0; b < nb; b++) {
        if (cnt[b] < 1) { fprintf(stderr, "mk_create: block without o tiles\n"); abort(); }
        to[((B - 1) * nb + b) * 2] = t; t += cnt[b]; to[((B - 1) * nb + b) * 2 + 1] = t;
      }
      if (t != O_T) { fprintf(stderr, "mk_create: o tile table broken\n"); abort(); }
    }
    CK(cudaMalloc(&d, sizeof(int) * 2 * MAXB * nb * 2)); CK(cudaMemcpy(d, to, sizeof(int) * 2 * MAXB * nb * 2, cudaMemcpyHostToDevice));
    P.tiles_o = d; delete[] to;
  }
  CK(cudaMemset(P.xres, 0, sizeof(float) * MAXB * H));
  CK(cudaMemset(P.qkv, 0, sizeof(float) * MAXB * QKV_ROWS));
  CK(cudaMemset(P.afrag, 0, 2 * MAXB * H));
  CK(cudaMemset(P.mfrag, 0, 2 * MAXB * INNER));
  CK(cudaMemset(P.kc, 0, kv_bytes));
  CK(cudaMemset(P.vc, 0, kv_bytes));   // V must stay finite: masked positions are multiplied by p = 0
  CK(cudaMemset(P.sync, 0, 64));
  static int sx_max = 0;                          // the attribute belongs to the kernel, not to the engine
  if (E->sx_bytes > sx_max) sx_max = E->sx_bytes;
  CK(cudaFuncSetAttribute(k_step, cudaFuncAttributeMaxDynamicSharedMemorySize, SX_OFF + sx_max));
  CK(cudaFuncSetAttribute(k_generate, cudaFuncAttributeMaxDynamicSharedMemorySize, SX_OFF + sx_max));
  CK(cudaFuncSetAttribute(k_gensk, cudaFuncAttributeMaxDynamicSharedMemorySize, SX_OFF + sx_max));
  CK(cudaMalloc(&E->sk_dev, sizeof(SkArgs))); CK(cudaMemset(E->sk_dev, 0, sizeof(SkArgs)));
  return E;
}

void mk_step(void* h, const void* tokens, void* logits, int B, int pos, int mode, void* err) {
  Engine* E = (Engine*)h;
  Params P = E->P;
  P.tokens = (const long long*)tokens; P.logits = (float*)logits; P.B = B; P.pos = pos;
  P.gen_out = (long long*)err;   // step(): host-mapped error word (token id outside the vocabulary)
  P.sx_bytes = E->sx_bytes;
  k_step<<<E->nsm, NTHREAD, SX_OFF + E->sx_bytes>>>(P);
  E->P.sync_base += (unsigned)(1 + E->P.stop_layer * (BARRIERS_PER_LAYER + (SPLIT_ON(B, pos) ? 1 : 0)) + NORM_BARRIERS(B, E->P.stop_layer) + TRACE_EXTRA - (E->P.l0tab ? 1 : 0)) * E->P.nwork;
  E->launches += 1;
  E->steps += 1;
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess && E->errors++ < 5) fprintf(stderr, "mk_step: launch error: %s\n", cudaGetErrorString(e));
}

// online path: out = device address of host-mapped memory, tag != 0 goes into the upper 32 bits of every token
void mk_generate_tag(void* h, const void* tokens, void* out, int B, int pos, int n_steps, unsigned tag) {
  Engine* E = (Engine*)h;
  Params P = E->P;
  P.tokens = (const long long*)tokens; P.gen_out = (long long*)out; P.B = B; P.pos = pos; P.n_steps = n_steps;
  P.gen_tag = tag;
  P.sx_bytes = E->sx_bytes;
#ifndef TRACK_A
  const int sk = E->sk_mode && E->sk_mode != 3 ? 1 : 0;
#else
  const int sk = 0;
#endif
#ifndef TRACK_A
  if (sk && (SK_PF > 0 || SK_RT) && E->sk_lrun) CK(cudaMemsetAsync(E->sk_lrun, 0, 64));   // step 0's slot (stream-ordered)
#endif
  if (E->sk_mode) { P.logits = sk ? (float*)E->sk_dev : nullptr; k_gensk<<<E->nsm, NTHREAD, SX_OFF + E->sx_bytes>>>(P); }   // mode 3: diagnosis (exact head in k_gensk)
  else k_generate<<<E->nsm, NTHREAD, SX_OFF + E->sx_bytes>>>(P);
  for (int i = 0; i < n_steps; i++) E->P.sync_base += (unsigned)(BARRIERS_PER_GEN_STEP + sk + (SPLIT_ON(B, pos + i) ? NL : 0) + NORM_BARRIERS(B, NL) - (E->P.l0tab ? 1 : 0)) * E->P.nwork;
  E->launches += 1;
  E->steps += n_steps;
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess && E->errors++ < 5) fprintf(stderr, "mk_generate: launch error: %s\n", cudaGetErrorString(e));
}

// greedy head of generate() / the online path: mode 0 exact, 1 sketch (w8: int8 tiles, rc: float4 {s, c1, c2, 0} per
// row; see build_sketch in api.py), 2 sketch with every tile rescored (test), 3 exact head inside k_gensk (diagnosis).
// Returns 0, or -1 (track A: no sketch).
int mk_sketch(void* h, const void* w8, const void* rc, int mode) {
  Engine* E = (Engine*)h;
#ifdef TRACK_A
  if (mode) return -1;
#endif
  CK(cudaDeviceSynchronize());
  SkArgs a; memset(&a, 0, sizeof(a));
  if (!E->sk_dev) return -2;
  CK(cudaMemcpy(&a, E->sk_dev, sizeof(a), cudaMemcpyDeviceToHost));
  if (!a.bl) CK(cudaMalloc(&a.bl, sizeof(float) * MAXB * E->nsm));
  a.w8 = (const uint4*)w8; a.rc = (const float4*)rc; a.force = mode == 2;
  if (!a.stat) { CK(cudaMalloc(&a.stat, 64)); CK(cudaMemset(a.stat, 0, 64)); }
  if (!a.lrun) { CK(cudaMalloc(&a.lrun, 64)); CK(cudaMemset(a.lrun, 0, 64)); }
  CK(cudaMemcpy(E->sk_dev, &a, sizeof(a), cudaMemcpyHostToDevice));
  E->sk_lrun = a.lrun;
  E->sk_mode = (w8 && rc) || mode == 3 ? mode : 0;
  return 0;
}

// measurement builds (SK_RT): variant flags (bit 0: 16-byte x preparation, bits 4..7: candidate tiles staged per block)
void mk_sketch_flags(void* h, int flags) {
  Engine* E = (Engine*)h;
  CK(cudaDeviceSynchronize());
  SkArgs a;
  CK(cudaMemcpy(&a, E->sk_dev, sizeof(a), cudaMemcpyDeviceToHost));
  a.pad = flags;
  CK(cudaMemcpy(E->sk_dev, &a, sizeof(a), cudaMemcpyHostToDevice));
}

// candidate tiles rescored since the last call (sum over steps and blocks); resets the counter
long long mk_sketch_count(void* h) {
  Engine* E = (Engine*)h;
  SkArgs a; unsigned v = 0;
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&a, E->sk_dev, sizeof(a), cudaMemcpyDeviceToHost));
  if (!a.stat) return -1;
  CK(cudaMemcpy(&v, a.stat, 4, cudaMemcpyDeviceToHost)); CK(cudaMemset(a.stat, 0, 4));
  return (long long)v;
}

void mk_generate(void* h, const void* tokens, void* out, int B, int pos, int n_steps) {
  mk_generate_tag(h, tokens, out, B, pos, n_steps, 0u);
}

// page-locked host memory mapped into the device's address space (tokens of the online path)
int mk_host_alloc(size_t bytes, void** hptr, void** dptr) {
  void* hp = nullptr;
  if (cudaHostAlloc(&hp, bytes, cudaHostAllocMapped) != cudaSuccess) { cudaGetLastError(); return -1; }
  if (cudaHostGetDevicePointer(dptr, hp, 0) != cudaSuccess) { cudaGetLastError(); cudaFreeHost(hp); return -2; }
  *hptr = hp;
  return 0;
}
void mk_host_free(void* hptr) { if (hptr) cudaFreeHost(hptr); }

// TRACE: copy out [blocks][8][8] timestamps and the final clock value
void mk_trace(void* h, unsigned* out, unsigned* clk) {
  Engine* E = (Engine*)h;
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out, E->P.trace, sizeof(unsigned) * 64 * 256, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(clk, E->P.trace + 64 * 256, sizeof(unsigned), cudaMemcpyDeviceToHost));
}

// ---- calibration of the tile tables (engine start, off the per-token path) ----
// record barrier waits (poll iterations per block and phase kind) while on
void mk_calib(void* h, int on) { ((Engine*)h)->P.calib = on; }
#ifdef TIMING
// copy out tstat[blocks][200][8]
void mk_get_tstat(void* h, unsigned* out) {
  Engine* E = (Engine*)h;
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out, E->P.tstat, sizeof(unsigned) * (size_t)E->P.nwork * 200 * 8, cudaMemcpyDeviceToHost));
}
#endif
// read and clear the wait statistics: out[9][blocks]
void mk_get_wstat(void* h, unsigned* out) {
  Engine* E = (Engine*)h;
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out, E->P.wstat, sizeof(unsigned) * 9 * E->P.nwork, cudaMemcpyDeviceToHost));
  CK(cudaMemset(E->P.wstat, 0, sizeof(unsigned) * 9 * E->P.nwork));
}
// set the number of tiles of every block for one matrix kind (0 qkv, 1 down, 2 gate+up, 3 lm_head); contiguous
// ranges in block order. Any table gives bitwise the same results: a tile's arithmetic does not depend on the block.
int mk_set_tiles(void* h, int kind, const int* counts) {
  Engine* E = (Engine*)h;
  const int nt[4] = {QKV_T, DN_T, GU_T, LM_T};
  const int nb = E->P.nwork;
  int tot = 0;
  for (int b = 0; b < nb; b++) { if (counts[b] < 1) return -1; tot += counts[b]; }
  if (kind < 0 || kind > 3 || tot != nt[kind]) return -2;
  CK(cudaDeviceSynchronize());
  int t = 0;
  for (int b = 0; b < nb; b++) {
    int pair[2] = {t, t + counts[b]};
    CK(cudaMemcpy((void*)(E->P.tiles + b * 8 + 2 * kind), pair, sizeof(pair), cudaMemcpyHostToDevice));
    t += counts[b];
  }
  return 0;
}

// debug: run only the first n layers from now on (NL = normal); read the residual stream [MAXB][H]
void mk_set_stop_layer(void* h, int n) { ((Engine*)h)->P.stop_layer = n; }
void mk_read_x(void* h, float* out) {
  Engine* E = (Engine*)h;
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out, E->P.xres, sizeof(float) * MAXB * H, cudaMemcpyDeviceToHost));
}

long long mk_stat(void* h, int which) {
  Engine* E = (Engine*)h;
  return which == 0 ? E->launches : which == 1 ? E->steps : which == 2 ? E->nsm : which == 3 ? E->errors :
         which == 4 ? (long long)(size_t)E->P.kc : which == 5 ? (long long)(size_t)E->P.vc : which == 6 ? (long long)(size_t)E->P.W : 0;   // 4-6: buffer addresses (diagnosis)
}

int mk_track() {
#ifdef TRACK_A
  return 1;
#else
  return 0;
#endif
}

// track B, layer-0 table. mk_l0_rows: one launch computing layer 0's q/k/v rows of B tokens (embedding + q/k/v phase
// of the normal kernel, then stop), copied into rows of `out` ([VOCAB][QKV_ROWS] float32). mk_set_l0tab: use it.
void mk_l0_rows(void* h, const void* tokens, void* out, int B) {
  Engine* E = (Engine*)h;
  Params P = E->P;
  P.tokens = (const long long*)tokens; P.logits = nullptr; P.B = B; P.pos = 0; P.l0build = 1; P.l0tab = nullptr;
  P.sx_bytes = E->sx_bytes;
  k_step<<<E->nsm, NTHREAD, SX_OFF + E->sx_bytes>>>(P);
  E->P.sync_base += 2u * E->P.nwork;      // embedding, q/k/v
  CK(cudaMemcpy2DAsync(out, sizeof(float) * QKV_ROWS, P.qkv, sizeof(float) * QKV_ROWS, sizeof(float) * QKV_ROWS, B, cudaMemcpyDeviceToDevice));
}
void mk_set_l0tab(void* h, const void* tab) { ((Engine*)h)->P.l0tab = (const float*)tab; }
int mk_check() { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) fprintf(stderr, "mk_check: %s\n", cudaGetErrorString(e)); return (int)e; }

void mk_destroy(void* h) {
  Engine* E = (Engine*)h;
  cudaDeviceSynchronize();
  Params& P = E->P;
  cudaFree(P.xres); cudaFree(P.qkv); cudaFree(P.afrag); cudaFree(P.mfrag); cudaFree(P.xfrag); cudaFree(P.xraw); cudaFree(P.ssq); cudaFree(P.kc); cudaFree(P.vc); cudaFree(P.asc); cudaFree(P.att_done); cudaFree(P.sync); cudaFree(P.wstat); cudaFree(P.bm_val); cudaFree(P.bm_idx);
  if (E->sk_dev) { SkArgs a; if (cudaMemcpy(&a, E->sk_dev, sizeof(a), cudaMemcpyDeviceToHost) == cudaSuccess && a.bl) cudaFree(a.bl); if (a.stat) cudaFree(a.stat); if (a.lrun) cudaFree(a.lrun); cudaFree(E->sk_dev); } cudaFree((void*)P.tiles); cudaFree((void*)P.tiles_o);
  delete E;
}

}  // extern "C"
