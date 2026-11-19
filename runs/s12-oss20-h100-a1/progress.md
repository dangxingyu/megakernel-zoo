# s12-oss20-h100-a1: gpt-oss-20b on H100

Score: `http_step_us_b1` (lower is better). Budget 32.0 agent-hours. 50 submissions so far, 49 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 1109
- 1.10x the fastest general engine: 3046
- vllm (general engine): 3351
- sglang (general engine): 3467

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Port of the gpt-oss-120b B200 megakernel to 20b/H100 (sm_90 e2m1 decode)

- status: **valid**, http_step_us_b1 = 2314 (B=8: 5745), at 0.63 agent-hours, category structure
- hypothesis: track A (MXFP4 experts as stored incl. E8M0 scales, BF16 elsewhere, fp32 attention scores). First own engine: one cooperative persistent kernel, 6 grid barriers per layer, TMA weight rings; e2m1 decoded by PRMT/shift/LOP3 as code*2^-126 with 2^63-scaled B operand and 2^(S-64) group scale (all exact). Own tests: error vs fp32 ref equal to HF bf16 (median 0.067 both), batch-invariant bitwise. Expect HTTP B=1 ~2345 us (serve in process 2333), B=8 ~5900 (serve 5889).
- evaluator: valid: 2314 us/step at B=1 and 5745 at B=8 over HTTP (online (submit/poll)); in process 2256 us/step at B=1 (step, MBU 0.492)

### #2 s0002: MoE as whole-unit warp streams + last-arriver down combine + cheaper e2m1 decode

- status: **valid**, http_step_us_b1 = 2275 (B=8: 4284), at 0.98 agent-hours, parent s0001, category gemm
- hypothesis: track A (MXFP4 codes as stored, nibbles permuted inside each 32-bit word = layout only; BF16 elsewhere). gate_up/down units (expert,rowtile) are whole-K warp tasks with swiglu / partial-y epilogues in the warp (no rounds, no block reductions); the last unit of a down rowtile combines over experts and writes x (no next-layer combine); 12 warps; decode 11 ops per 8 codes (Marlin-style nibble order), no dead-column selects. Same-card in process: serve B=1 2333->2275, B=8 5889->4327. Expect HTTP B=1 ~2255 us (s0001 2314), B=8 ~4200 (s0001 5745).
- evaluator: valid: 2275 us/step at B=1 and 4284 at B=8 over HTTP (online (submit/poll)); in process 2182 us/step at B=1 (step, MBU 0.508)

### #3 s0003: Faster norms (one warp per row, loads in flight) and down combine (4 slots/sequence)

- status: **valid**, http_step_us_b1 = 2275 (B=8: 3853), at 1.73 agent-hours, parent s0002, category sync
- hypothesis: track A (formats as s0002). RMSNorm: one warp per sequence loads its row in two passes of 12 x 8 B per lane (was a dependent loop), noinline. Down combine: each sequence's 4 slots precomputed, all ypart/bias loads issued before the ordered sum (was a loop over all U slots with conditional loads). Same-card A/B vs s0002 (e0054, e0078): B=1 -3 us, B=8 -440 us. Expect HTTP B=1 ~2270 (s0002 2275), B=8 ~3850 (4284).
- evaluator: valid: 2275 us/step at B=1 and 3853 at B=8 over HTTP (online (submit/poll)); in process 2179 us/step at B=1 (step, MBU 0.509)

### #4 s0004: Router on 45 blocks: K-slice partials summed by every block, top-4 in smem

- status: **valid**, http_step_us_b1 = 2254 (B=8: 3850), at 2.03 agent-hours, parent s0003, category scheduling
- hypothesis: track A (formats as s0003; engine/TRACK=A). The router (32x2880) ran on 2 blocks streaming 92 KB each while 130 waited; now 45 blocks each compute a 4-kstep slice of both router rowtiles in phase D and every block sums the 45 partials in fixed order before top-4. Same-card A/B (e0081): B=1 -12 us, B=8 -52 us. Expect HTTP B=1 ~2263 (s0003 2275), B=8 ~3800 (3853).
- evaluator: valid: 2254 us/step at B=1 and 3850 at B=8 over HTTP (online (submit/poll)); in process 2157 us/step at B=1 (step, MBU 0.514)

### #5 s0005: Track B: certified int8 head sketch on the greedy path, exact recompute of candidates

- status: **valid**, http_step_us_b1 = 2061 (B=8: 3671), at 2.12 agent-hours, parent s0004, category memory
- hypothesis: track B (engine/TRACK=B): s0004 + a certified int8 sketch of lm_head (582 MB extra, int8 rows with per-row scale, m16n8k32 integer sums, Cauchy-Schwarz bound incl. 2^-10||w|| for the exact path's fp32 rounding); tiles whose upper bound reaches the best lower bound are recomputed with the exact head's MMA chain, spread over all blocks. step() logits use the exact head; generated/served tokens are those of track A (soak: 54/54 sequences x 256-512 tokens identical to exact step greedy). Same-card in process: serve B=1 2204 -> 1999, B=8 3832 -> 3656. Expect HTTP B=1 ~2050 (s0004 2254), B=8 ~3680 (3850).
- evaluator: valid: 2061 us/step at B=1 and 3671 at B=8 over HTTP (online (submit/poll)); in process 1966 us/step at B=1 (generate, MBU 0.514)

### #6 s0006: K/V cache in MMA fragment order: attention reads with 16-byte loads

- status: **valid**, http_step_us_b1 = 2037 (B=8: 3641), at 2.74 agent-hours, parent s0005, category attention
- hypothesis: track B (engine/TRACK=B, formats as s0005). The attention read K/V with 4-byte loads touching 8 rows per instruction (half-used sectors, ~2000 load instructions per SM at 1000 keys); K rows and V blocks are now stored permuted into fragment order so each lane reads 2 (K) / 4 (V) x 16 B contiguous. Same arithmetic. Same-card A/B (e0127, gen 1024): B=1 -36 us, B=8 -42. Expect HTTP B=1 ~2025 (s0005 2061), B=8 ~3630 (3671).
- evaluator: valid: 2037 us/step at B=1 and 3641 at B=8 over HTTP (online (submit/poll)); in process 1957 us/step at B=1 (generate, MBU 0.516)

### #7 s0007: Track A: K/V cache in MMA fragment order (16-byte loads in attention)

- status: **valid**, http_step_us_b1 = 2237 (B=8: 3819), at 3.18 agent-hours, parent s0006, category attention
- hypothesis: track A (engine/TRACK=A; MXFP4 experts as stored, nibble-permuted words; BF16 elsewhere): s0004 + the token-id guard + K/V rows/blocks stored in fragment order so the attention reads them with 16-byte loads (same arithmetic; track B s0006 measured -24 us HTTP for the same change). Same-card teacher-forced/gen: B=1 -36 us, B=8 -42. Expect HTTP B=1 ~2220 (s0004 2254), B=8 ~3810 (3850).
- evaluator: valid: 2237 us/step at B=1 and 3819 at B=8 over HTTP (online (submit/poll)); in process 2153 us/step at B=1 (step, MBU 0.515)

### #8 s0008: Release atomics in barrier/counters, warp-parallel greedy token choice

- status: **valid**, http_step_us_b1 = 2026 (B=8: 3599), at 3.35 agent-hours, parent s0007, category sync
- hypothesis: track B (engine/TRACK=B, formats as s0006). Grid barrier: one red.release.gpu add per block instead of __threadfence + atomicAdd; down-rowtile counter: one atom.acq_rel by lane 0 after __syncwarp instead of a fence in every lane; the next token is chosen by one warp per sequence with all 132 block candidates loaded at once (was one thread scanning them). Same logits, same tokens. Same-card serve: B=1 2206 -> 2163, B=8 3827 -> 3789. Expect HTTP B=1 ~1995 (s0006 2037), B=8 ~3600 (3641).
- evaluator: valid: 2026 us/step at B=1 and 3599 at B=8 over HTTP (online (submit/poll)); in process 1937 us/step at B=1 (generate, MBU 0.520)

### #9 s0009: Routing block in registers: per-warp logits+top-4, warp-0 union/slots (2 syncs, was 5)

- status: **valid**, http_step_us_b1 = 1992 (B=8: 3457), at 3.39 agent-hours, parent s0008, category sync
- hypothesis: track B (engine/TRACK=B, formats as s0008). The router-sum / top-4 / union / slot-list section ran as 5 dependent block-wide steps through shared memory; now warp b sums its 32 router logits in registers (lane = expert, same summation order), takes top-4 by shuffles, and warp 0 builds union, slots, weights and per-sequence slot lists. Same bits. Same-card teacher forced (e0159): B=1 2199 -> 2110, B=8 3637 -> 3475. Expect HTTP B=1 ~1940 (s0008 2026), B=8 ~3450 (3599).
- evaluator: valid: 1992 us/step at B=1 and 3457 at B=8 over HTTP (online (submit/poll)); in process 1890 us/step at B=1 (generate, MBU 0.538)

### #10 s0010: Attention as in-block flash decoding: online softmax per warp, 2 syncs (was 5)

- status: **valid**, http_step_us_b1 = 1935 (B=8: 3411), at 3.54 agent-hours, parent s0009, category attention
- hypothesis: track B (engine/TRACK=B, formats as s0009). Attention block (b, kv head): each warp walks its 16-key blocks with an online softmax in registers (fp32 scores, P = bf16(exp(s - running max)) straight from the score registers into the P V mma, acc rescaled), then one combine over warps with the sink; same arithmetic for every batch size (bit-identical alone vs batched). P is now relative to the running max instead of the final normalisation (the flash-decoding data flow of the general engines). Same-card teacher forced ctx 64-1064: B=1 -27 us, B=8 -28. Expect HTTP B=1 ~1970 (s0009 1992), B=8 ~3430 (3457); numeric gates unchanged within noise.
- evaluator: valid: 1935 us/step at B=1 and 3411 at B=8 over HTTP (online (submit/poll)); in process 1876 us/step at B=1 (generate, MBU 0.542)

### #11 s0011: Track A catch-up: routing in registers, flash attention, release atomics, sync trims

- status: **valid**, http_step_us_b1 = 2072 (B=8: 3509), at 3.74 agent-hours, parent s0010, category structure
- hypothesis: track A (engine/TRACK=A; MXFP4 experts as stored with nibble-permuted words, BF16 elsewhere, exact head). s0007 + the changes measured on track B since: release-atomic barrier/counter, warp-parallel token choice, routing block in registers (2 syncs), in-block flash-decoding attention (P relative to the running max, the general engines' data flow), down tables in the routing warps. Track B moved 2037 -> 1935 over these (s0006 -> s0010) plus -12 us since. Expect HTTP B=1 ~2125 (s0007 2237), B=8 ~3590 (3819).
- evaluator: valid: 2072 us/step at B=1 and 3509 at B=8 over HTTP (online (submit/poll)); in process 2016 us/step at B=1 (step, MBU 0.550)

### #12 s0012: Routing tail over 3 warps, 5-stage qkv/o segments, sketch loads in flight 15

- status: **valid**, http_step_us_b1 = 1896 (B=8: 3345), at 3.88 agent-hours, parent s0011, category cleanup
- hypothesis: track B (engine/TRACK=B, formats as s0010). Three small changes measured separately on one card: routing tail split over warps 0-2 (B=1 -12 us, B=8 -17, teacher forced), qkv/o_proj K segments of exactly 5 ring stages (NSEG 4/6: -6), sketch pass-1 loads in flight 9 -> 15 (-10). Expect HTTP B=1 ~1910 (s0010 1935), B=8 ~3390 (3411).
- evaluator: valid: 1896 us/step at B=1 and 3345 at B=8 over HTTP (online (submit/poll)); in process 1840 us/step at B=1 (generate, MBU 0.554)

### #13 s0013: Track B: 6-bit certified head sketch (434 MB instead of 579)

- status: **valid**, http_step_us_b1 = 1863 (B=8: 3333), at 4.31 agent-hours, parent s0012, category memory
- hypothesis: track B (engine/TRACK=B). Head sketch rows now 6-bit (q in [-31,31], 4 codes per 3 bytes, unpacked in registers to int8 4q for the integer mma; bounds recomputed for the coarser rows); pass 1 reads 434 MB instead of 579, pass 2 recomputes ~45 candidate tiles per step at B=1 (was ~1-4) with the exact head. Tokens unchanged (soak 27/27). Same-card gen: B=1 -20 us, B=8 -12. Expect HTTP B=1 ~1876 (s0012 1896), B=8 ~3375.
- evaluator: valid: 1863 us/step at B=1 and 3333 at B=8 over HTTP (online (submit/poll)); in process 1817 us/step at B=1 (generate, MBU 0.552)

### #14 s0014: Track A: stage-aligned qkv/o segments, routing tail over 3 warps

- status: **valid**, http_step_us_b1 = 2078 (B=8: 3501), at 4.44 agent-hours, parent s0013, category cleanup
- hypothesis: track A (engine/TRACK=A; checkpoint formats, exact head). s0011 + the two small changes measured since: qkv/o_proj K segments of exactly 5 ring stages (NSEG 4/6, -6 us B=1) and the routing tail split over warps 0-2 (-12 us B=1, -17 B=8). Expect HTTP B=1 ~2055 (s0011 2072), B=8 ~3490 (3509).
- evaluator: valid: 2078 us/step at B=1 and 3501 at B=8 over HTTP (online (submit/poll)); in process 2024 us/step at B=1 (generate, MBU 0.548)

### #15 s0015: Track B: router weights folded with ln2; router partials in o_proj epilogue, no phase D

- status: **valid**, http_step_us_b1 = 1813 (B=8: 3328), at 4.94 agent-hours, parent s0013, category sync
- hypothesis: track B; formats differ from the checkpoint: 6-bit certified head sketch (as s0013) + router weights stored as fp32 products ln2_k*Wr_ek (exact). logit = rinv*sum + bias is the HF router in exact arithmetic; each block sums its o_proj rows' router partials in phase C, removing phase D's grid barrier and router stream and overlapping routing with norm2. Expect B=1 1863 -> ~1815 (same-card serve -49), B=8 ~unchanged (~3330).
- evaluator: valid: 1813 us/step at B=1 and 3328 at B=8 over HTTP (online (submit/poll)); in process 1777 us/step at B=1 (generate, MBU 0.563)

### #16 s0016: Track B: L2 prefetch of each attention block's K/V window at the start of phase A

- status: **valid**, http_step_us_b1 = 1802 (B=8: 3343), at 5.44 agent-hours, parent s0015, category attention
- hypothesis: track B; formats as s0015 (6-bit certified head sketch, ln2-folded fp32 router). Attention loads were one dependent HBM round trip per 16-key block per warp; the attention blocks now bulk-prefetch their K/V window into L2 at phase-A start (B<=4), so phase B reads hit L2. Expect B=1 1813 -> ~1800 (same-card serve -13), B=8 ~+10 (code layout).
- evaluator: valid: 1802 us/step at B=1 and 3343 at B=8 over HTTP (online (submit/poll)); in process 1778 us/step at B=1 (generate, MBU 0.564)

### #17 s0017: Track B: norms take sums of squares from partials of the producing phase (one pass)

- status: **valid**, http_step_us_b1 = 1777 (B=8: 3353), at 5.57 agent-hours, parent s0016, category sync
- hypothesis: track B; formats as s0016. The down combine and the o_proj epilogue write per-rowtile sums of squares of the rows they store; norm1, norm2 and the final norm read the row once instead of twice (same function in exact arithmetic, summation order differs). Expect B=1 1802 -> ~1780 (same-card serve -21); B=8 teacher-forced neutral, HTTP B=8 may move with the trajectory.
- evaluator: valid: 1777 us/step at B=1 and 3353 at B=8 over HTTP (online (submit/poll)); in process 1758 us/step at B=1 (generate, MBU 0.568)

### #18 s0018: qkv epilogue bias and RoPE inputs loaded between the stream and the block barrier

- status: **valid**, http_step_us_b1 = 1766 (B=8: 3354), at 5.89 agent-hours, parent s0017, category scheduling
- hypothesis: track B; formats as s0017 (6-bit certified head sketch, ln2-folded fp32 router, partial sums of squares). Each thread loads its qkv epilogue item's biases and cos/sin after its warp's stream and before the block barrier, hiding one L2 round trip per layer behind the barrier wait. Expect B=1 1777 -> ~1768 (same-card gen -9.5), B=8 unchanged.
- evaluator: valid: 1766 us/step at B=1 and 3354 at B=8 over HTTP (online (submit/poll)); in process 1749 us/step at B=1 (generate, MBU 0.571)

### #19 s0019: Track A catch-up: K/V window L2 prefetch, qkv epilogue inputs before the block barrier

- status: **valid**, http_step_us_b1 = 2063 (B=8: 3520), at 6.25 agent-hours, parent s0014, category scheduling
- hypothesis: track A; checkpoint formats only (MXFP4 experts with E8M0 scales, BF16 elsewhere, exact BF16 head). Brings the shared non-numeric changes since s0014 to track A: L2 prefetch of the attention K/V window (B<=4) and qkv epilogue loads before the block barrier. Expect B=1 2078 -> ~2055, B=8 ~3510.
- evaluator: valid: 2063 us/step at B=1 and 3520 at B=8 over HTTP (online (submit/poll)); in process 2033 us/step at B=1 (generate, MBU 0.543)

### #20 s0020: E2M1 decode by masked integer multiply (same bits, ~30% fewer decode instructions)

- status: **valid**, http_step_us_b1 = 1752 (B=8: 3331), at 6.59 agent-hours, parent s0018, category gemm
- hypothesis: track B; formats as s0018 (6-bit certified head sketch, ln2-folded fp32 router, partial sums of squares). The MoE item loop is issue-bound (bench/dec.cu: ~98 instructions/item, 75 decode); decoding a masked nibble pair with one multiply by (2^12+2^6) or (2^8+2^2) gives the same bf16 bits with 13 instead of ~19 instructions per word. Expect B=1 1766 -> ~1758, B=8 3354 -> ~3325.
- evaluator: valid: 1752 us/step at B=1 and 3331 at B=8 over HTTP (online (submit/poll)); in process 1737 us/step at B=1 (generate, MBU 0.575)

### #21 s0021: Track B: single-pass norms split over 4 warps per row at B<=2

- status: **valid**, http_step_us_b1 = 1734 (B=8: 3322), at 6.97 agent-hours, parent s0020, category sync
- hypothesis: track B; formats as s0020. With sums of squares from partials no cross-warp reduction is left, so at B<=2 each norm row is split over 4 warps (same bits); fewer live registers per warp and 4x parallel compute shorten norm1 and norm2 (~3.2 us each per layer). Expect B=1 1752 -> ~1728 (same-card gen -24), B=8 unchanged.
- evaluator: valid: 1734 us/step at B=1 and 3322 at B=8 over HTTP (online (submit/poll)); in process 1714 us/step at B=1 (generate, MBU 0.584)

### #22 s0022: Prefetch of cold epilogue inputs: folded router rows (L1) and down biases (L2)

- status: **valid**, http_step_us_b1 = 1721 (B=8: 3331), at 7.12 agent-hours, parent s0021, category memory
- hypothesis: track B; formats as s0021. Two epilogue inputs were HBM-cold loads issued after a barrier: the o_proj epilogue's folded router rows (4 KB/block) and the down combine's expert biases. They are now prefetched without registers (prefetch.global.L1 at phase-C start; prefetch.global.L2 at gate_up start, B<=2). Expect B=1 1734 -> ~1720 (same-card gen -14), B=8 ~-5.
- evaluator: valid: 1721 us/step at B=1 and 3331 at B=8 over HTTP (online (submit/poll)); in process 1702 us/step at B=1 (generate, MBU 0.588)

### #23 s0023: L2 prefetch of the next layer's norm weights, biases and sinks during the down phase

- status: **valid**, http_step_us_b1 = 1713 (B=8: 3315), at 7.21 agent-hours, parent s0022, category memory
- hypothesis: track B; formats as s0022. The next layer's small vectors (ln1, qkv/o biases, sinks, ln2, router bias; ~55 KB) were first read from HBM right after barriers by every block; 5 blocks now bulk-prefetch them into L2 at the start of the down phase. Expect B=1 1721 -> ~1709 (same-card gen -12.5), B=8 ~-10.
- evaluator: valid: 1713 us/step at B=1 and 3315 at B=8 over HTTP (online (submit/poll)); in process 1690 us/step at B=1 (generate, MBU 0.589)

### #24 s0024: Track A catch-up: masked-multiply decode, L2 prefetch of small vectors and down biases

- status: **valid**, http_step_us_b1 = 2035 (B=8: 3480), at 7.39 agent-hours, parent s0019, category gemm
- hypothesis: track A; checkpoint formats only (MXFP4 experts with E8M0 scales, BF16 elsewhere, exact BF16 head). Brings shared changes since s0019: e2m1 decode by masked multiply (same bits), L2 prefetch of the next layer's small vectors, L2 prefetch of the down combine's biases (B<=2). Expect B=1 2063 -> ~2035, B=8 3520 -> ~3490.
- evaluator: valid: 2035 us/step at B=1 and 3480 at B=8 over HTTP (online (submit/poll)); in process 1987 us/step at B=1 (generate, MBU 0.556)

### #25 s0025: Track A: single-pass norms with sums of squares accumulated where rows are produced

- status: **valid**, http_step_us_b1 = 1952 (B=8: 3487), at 7.54 agent-hours, parent s0024, category sync
- hypothesis: track A; checkpoint formats only (MXFP4 experts with E8M0 scales, BF16 elsewhere, exact BF16 head); same fp32 sum of squares, accumulated by the phase that writes each row (o_proj epilogue, down combine) instead of a second pass over the row; rows split over 4 warps at B<=2. Expect B=1 2035 -> ~1955 (same-card gen -83), B=8 ~3490.
- evaluator: valid: 1952 us/step at B=1 and 3487 at B=8 over HTTP (online (submit/poll)); in process 1926 us/step at B=1 (generate, MBU 0.575)

### #26 s0026: Consume loop counter in a register at RS=2 (no local-memory round trip per stage)

- status: **valid**, http_step_us_b1 = 1704 (B=8: 3308), at 7.93 agent-hours, parent s0023, category cleanup
- hypothesis: track B; formats as s0023. 'volatile int ct' (a workaround for a ptxas bug seen only with 3+ ring stages) kept the task counter in local memory: every ring stage of every GEMV did a dependent local load. A plain counter at RS=2 (results bit-identical, checks clean). Expect B=1 1713 -> ~1706 (same-card gen -7), B=8 ~-8.
- evaluator: valid: 1704 us/step at B=1 and 3308 at B=8 over HTTP (online (submit/poll)); in process 1686 us/step at B=1 (generate, MBU 0.591)

### #27 s0027: 4-warp norm loads its norm weights together with the row

- status: **valid**, http_step_us_b1 = 1699 (B=8: 3305), at 8.23 agent-hours, parent s0026, category sync
- hypothesis: track B; formats as s0026. In the 4-warp single-pass norm each lane now loads its 6 norm-weight float4 together with the row (24 registers, affordable with 6 chunks per warp), removing a dependent L2 round trip. Expect B=1 1704 -> ~1700 (same-card gen -4.5), B=8 ~-5.
- evaluator: valid: 1699 us/step at B=1 and 3305 at B=8 over HTTP (online (submit/poll)); in process 1679 us/step at B=1 (generate, MBU 0.594)

### #28 s0028: MXFP4 items stored codes+scales contiguous: one bulk copy per ring stage

- status: **valid**, http_step_us_b1 = 1692 (B=8: 3283), at 8.53 agent-hours, parent s0027, category memory
- hypothesis: track B; formats as s0027, plus a layout-only change: each 64-column MXFP4 item's 512 B of codes and 32 B of E8M0 scales stored contiguously (pure byte permutation of the checkpoint's bytes), so every MoE ring stage is one bulk copy instead of two. Expect B=1 1699 -> ~1690 (same-card gen -9.6), B=8 3305 -> ~3285.
- evaluator: valid: 1692 us/step at B=1 and 3283 at B=8 over HTTP (online (submit/poll)); in process 1672 us/step at B=1 (generate, MBU 0.596)

### #29 s0029: Track A catch-up: fused MXFP4 items, register loop counter, 4-warp norm weight loads

- status: **valid**, http_step_us_b1 = 1924 (B=8: 3444), at 8.59 agent-hours, parent s0025, category memory
- hypothesis: track A; checkpoint formats only (MXFP4 experts with E8M0 scales, BF16 elsewhere, exact BF16 head); layout and kernel organisation only. Brings shared changes since s0025: codes+scales stored contiguously per item (one bulk copy per stage), the consume loop counter in a register, norm weights loaded with the row. Expect B=1 1952 -> ~1925, B=8 3487 -> ~3450.
- evaluator: valid: 1924 us/step at B=1 and 3444 at B=8 over HTTP (online (submit/poll)); in process 1898 us/step at B=1 (generate, MBU 0.584)

### #30 s0030: Earlier o_proj ring prefetch; sketch loads with L2 evict_first

- status: **valid**, http_step_us_b1 = 1680 (B=8: 3274), at 8.85 agent-hours, parent s0028, category scheduling
- hypothesis: track B; formats as s0028. Two scheduling changes: each warp issues its o_proj ring prefetch as soon as its own qkv stream is done (before the epilogue), and the 434 MB sketch pass reads with the L2 evict_first policy so it no longer flushes the small hot data. Expect B=1 1692 -> ~1680 (same-card gen -9.4 and -3.5), B=8 ~-15.
- evaluator: valid: 1680 us/step at B=1 and 3274 at B=8 over HTTP (online (submit/poll)); in process 1658 us/step at B=1 (generate, MBU 0.600)

### #31 s0031: gate_up units dealt round-robin over blocks (B=8 balance)

- status: **valid**, http_step_us_b1 = 1680 (B=8: 3250), at 10.18 agent-hours, parent s0030, category scheduling
- hypothesis: track B; formats as s0030. gate_up units were contiguous ranges per block (at B=8 ~40 units of one or two experts); dealing them round-robin spreads each block's reads over all experts and regions. Unit results are independent, so logits are unchanged. Expect B=1 unchanged (~1680), B=8 3274 -> ~3245 (same-card tf -19, gen -31).
- evaluator: valid: 1680 us/step at B=1 and 3250 at B=8 over HTTP (online (submit/poll)); in process 1658 us/step at B=1 (generate, MBU 0.599)

### #32 s0032: Grid barrier: acq_rel atomic add; the last arriving block skips polling

- status: **valid**, http_step_us_b1 = 1678 (B=8: 3255), at 10.45 agent-hours, parent s0031, category sync
- hypothesis: track B; formats as s0031. The grid barrier's arrival becomes atom.add.acq_rel with the returned value (bench: 0.97 vs 1.05 us for the release red), and the last arriver, having acquired every block's release, does not poll. ~123 barriers per step. Expect B=1 1680 -> ~1676 (same-card gen -4.2), B=8 ~+7.
- evaluator: valid: 1678 us/step at B=1 and 3255 at B=8 over HTTP (online (submit/poll)); in process 1658 us/step at B=1 (generate, MBU 0.600)

### #33 s0033: Track A catch-up: acq_rel grid barrier, round-robin gate_up units

- status: **valid**, http_step_us_b1 = 1896 (B=8: 3393), at 10.50 agent-hours, parent s0029, category sync
- hypothesis: track A; checkpoint formats only (MXFP4 experts with E8M0 scales, BF16 elsewhere, exact BF16 head); scheduling only. Brings the shared changes since s0029: grid barrier arrival as atom.add.acq_rel with no polling by the last arriver, gate_up units dealt round-robin over blocks. Expect B=1 1924 -> ~1920, B=8 3444 -> ~3425.
- evaluator: valid: 1896 us/step at B=1 and 3393 at B=8 over HTTP (online (submit/poll)); in process 1872 us/step at B=1 (generate, MBU 0.592)

### #34 s0034: B=1-specialised kernel build per cohort; down units split in K halves (track B)

- status: **valid**, http_step_us_b1 = 1668 (B=8: 3406), at 11.45 agent-hours, parent s0032, category gemm
- hypothesis: track B; formats as s0032. Two builds of the same kernel: batch size 1 runs on a build with B a compile-time constant, other sizes on the general one; both split down units into K halves (DSPLIT=2), so every sequence's arithmetic is identical at every batch size (bitwise checks pass). Trade-off chosen for the score: B=1 1678 -> ~1668 (same-card serve -10.5), B=8 3255 -> ~3355 (+100).
- evaluator: valid: 1668 us/step at B=1 and 3406 at B=8 over HTTP (online (submit/poll)); in process 1647 us/step at B=1 (generate, MBU 0.603)

### #35 s0035: B=1 build: contiguous gate_up unit ranges (round-robin only in the general build)

- status: **valid**, http_step_us_b1 = 1653 (B=8: 3405), at 11.65 agent-hours, parent s0034, category scheduling
- hypothesis: track B; formats as s0034. The B=1-specialised build goes back to contiguous gate_up unit ranges per block (round-robin helped only at B=8); unit assignment does not change any arithmetic. Expect B=1 1668 -> ~1652 (same-card serve -15), B=8 unchanged (~3406).
- evaluator: valid: 1653 us/step at B=1 and 3405 at B=8 over HTTP (online (submit/poll)); in process 1631 us/step at B=1 (generate, MBU 0.607)

### #36 s0036: Track A: batch-size-1 specialised kernel build per cohort, contiguous gate_up ranges there

- status: **valid**, http_step_us_b1 = 1877 (B=8: 3394), at 12.04 agent-hours, parent s0033, category scheduling
- hypothesis: track A; checkpoint formats only (MXFP4 experts with E8M0 scales, BF16 elsewhere, exact BF16 head); kernel organisation only. Batch size 1 runs on a build with B a compile-time constant and contiguous gate_up unit ranges; other sizes on the general build; identical arithmetic per sequence. Expect B=1 1896 -> ~1873 (same-card serve -23), B=8 ~3393.
- evaluator: valid: 1877 us/step at B=1 and 3394 at B=8 over HTTP (online (submit/poll)); in process 1850 us/step at B=1 (generate, MBU 0.597)

### #37 s0037: B=1 build: sketch loads 10 k-steps per batch, no down-bias prefetch

- status: **valid**, http_step_us_b1 = 1651 (B=8: 3408), at 12.18 agent-hours, parent s0035, category scheduling
- hypothesis: track B; formats as s0035. In the B=1-specialised build: sketch pass-1 batches of 10 k-steps and no L2 prefetch of the down biases (both only change load scheduling; tokens and logits identical, bitwise checks pass). Expect B=1 1653 -> ~1648 (same-card gen -5), B=8 unchanged.
- evaluator: valid: 1651 us/step at B=1 and 3408 at B=8 over HTTP (online (submit/poll)); in process 1629 us/step at B=1 (generate, MBU 0.608)

### #38 s0038: B=1 build: down units half-major (both K halves of a rowtile in different blocks' ranges)

- status: **valid**, http_step_us_b1 = 1647 (B=8: 3405), at 13.01 agent-hours, parent s0037, category scheduling
- hypothesis: track B; formats as s0037. In the B=1-specialised build the down units of a slot are ordered by K half first, then rowtile (assignment only; arithmetic identical, bitwise checks pass). Expect B=1 1651 -> ~1648 (same-card serve -3.2), B=8 unchanged.
- evaluator: valid: 1647 us/step at B=1 and 3405 at B=8 over HTTP (online (submit/poll)); in process 1627 us/step at B=1 (generate, MBU 0.610)

### #39 s0039: Head recompute: candidate tile loaded by one 92 KB bulk copy (TMA) instead of a load loop

- status: **valid**, http_step_us_b1 = 1639 (B=8: 3396), at 13.43 agent-hours, parent s0038, category memory
- hypothesis: track B; formats as s0038. The exact recompute of each candidate head tile now loads its 92 KB by one cp.async.bulk on its own mbarrier (no registers, all bytes in flight) instead of a per-thread load->store loop (6-9 us per tile in the profile); same bits. Expect B=1 1646.6 -> ~1638 (same-card serve -8.9), B=8 3405 -> ~3398.
- evaluator: valid: 1639 us/step at B=1 and 3396 at B=8 over HTTP (online (submit/poll)); in process 1618 us/step at B=1 (generate, MBU 0.609)

### #40 s0040: Head recompute copy in 6 chunks so the exact chain overlaps it (general build, B=8)

- status: **valid**, http_step_us_b1 = 1639 (B=8: 3386), at 13.68 agent-hours, parent s0039, category memory
- hypothesis: track B; formats as s0039. In the general track-B build (B>1 cohorts) each candidate tile's 92 KB bulk copy is split in 6 chunks with one mbarrier each and the exact 180-step chain waits per chunk (same order, same bits); the B=1 build is unchanged. Expect B=8 3396 -> ~3388 (same-card serve -9), B=1 unchanged (~1639).
- evaluator: valid: 1639 us/step at B=1 and 3386 at B=8 over HTTP (online (submit/poll)); in process 1618 us/step at B=1 (generate, MBU 0.609)

### #41 s0041: Track A catch-up: B=1 build with half-major down units and no down-bias prefetch

- status: **valid**, http_step_us_b1 = 1878 (B=8: 3394), at 13.76 agent-hours, parent s0036, category scheduling
- hypothesis: track A; checkpoint formats (MXFP4 experts with E8M0 scales, BF16 elsewhere). The track-A B=1 build now carries DORD (down units half-major) and DBPF=0 from s0037/s0038 (assignment and prefetch only; bitwise checks pass). Expect B=1 1876.5 -> ~1870, B=8 ~3394 unchanged.
- evaluator: valid: 1878 us/step at B=1 and 3394 at B=8 over HTTP (online (submit/poll)); in process 1854 us/step at B=1 (generate, MBU 0.597)

### #42 s0042: Head sketch: 5-bit codes with per-32-column u8 group scales (381 MB instead of 434)

- status: **valid**, http_step_us_b1 = 1634 (B=8: 3405), at 14.33 agent-hours, parent s0040, category memory
- hypothesis: track B; checkpoint formats plus a derived certified head sketch (now 5-bit codes with u8 per-(row, 32-column) scales on a float32 row base instead of 6-bit row-scaled; exact greedy argmax via the certified bound and exact recompute, verified bound ratio <= 0.1 on real activations); B=1 build packs hi/lo x digits into one integer mma (same integers). Pass 1 reads 12% fewer bytes but decodes more. Expect B=1 1639 -> ~1633 (same-card gen -6.5), B=8 3386 -> ~3376.
- evaluator: valid: 1634 us/step at B=1 and 3405 at B=8 over HTTP (online (submit/poll)); in process 1628 us/step at B=1 (generate, MBU 0.607)

### #43 s0043: Keep the 6-bit head sketch for B>1 cohorts; 5-bit group-scaled sketch only at B=1

- status: **valid**, http_step_us_b1 = 1634 (B=8: 3396), at 14.43 agent-hours, parent s0042, category memory
- hypothesis: track B; checkpoint formats plus two derived certified head sketches (6-bit row-scaled for the general build, 5-bit group-scaled for the B=1 build; both exact via certified bounds and exact recompute). The 5-bit sketch's heavier candidate tail cost B=8 ~20 us on the evaluator's workloads (s0042); B=8 cohorts go back to the 6-bit one. Expect B=8 3405 -> ~3386, B=1 unchanged ~1634.
- evaluator: valid: 1634 us/step at B=1 and 3396 at B=8 over HTTP (online (submit/poll)); in process 1628 us/step at B=1 (generate, MBU 0.607)

### #44 s0044: B=1 build: L2 prefetch of each warp's first sketch batch at the end of the last layer

- status: **valid**, http_step_us_b1 = 1634 (B=8: 3392), at 14.74 agent-hours, parent s0043, category memory
- hypothesis: track B; formats as s0043. In the B=1 build each warp's first sketch load batch (3.4 KB) is bulk-prefetched into L2 right before the last layer's final barrier, so pass 1 does not start cold after the norm and quantization (prefetch only, same bits). Expect B=1 1634 -> ~1632 (same-card gen -2), B=8 unchanged.
- evaluator: valid: 1634 us/step at B=1 and 3392 at B=8 over HTTP (online (submit/poll)); in process 1626 us/step at B=1 (generate, MBU 0.608)

### #45 s0045: B=1 build: double-buffered sketch loads in head pass 1 (next batch in flight)

- status: **valid**, http_step_us_b1 = 1629 (B=8: 3396), at 15.32 agent-hours, parent s0044, category memory
- hypothesis: track B; formats as s0044. In the B=1 build, head pass 1 loads its sketch batches (3 pairs of k32-steps) double-buffered so the next batch's loads are in flight during the current batch's decode and integer mma (same order, same bits; 168 registers, no new spills). Expect B=1 1634 -> ~1630 (same-card gen -3.8), B=8 unchanged.
- evaluator: valid: 1629 us/step at B=1 and 3396 at B=8 over HTTP (online (submit/poll)); in process 1624 us/step at B=1 (generate, MBU 0.609)

### #46 s0046: B=1 build: router top-4 by rank counting (independent shuffles, same selection)

- status: **valid**, http_step_us_b1 = 1625 (B=8: 3396), at 15.93 agent-hours, parent s0045, category sync
- hypothesis: track B; formats as s0045. In the B=1 build the router's top-4 is found by ranking every lane's logit with 31 independent shuffles instead of four dependent 5-step shuffle argmax rounds (same experts, same order, same tie rule; bitwise checks pass). Expect B=1 1628.5 -> ~1625 (same-card gen -3.8), B=8 unchanged.
- evaluator: valid: 1625 us/step at B=1 and 3396 at B=8 over HTTP (online (submit/poll)); in process 1621 us/step at B=1 (generate, MBU 0.610)

### #47 s0047: Track A: router top-4 by rank counting in the batch-size-1 build

- status: **valid**, http_step_us_b1 = 1871 (B=8: 3394), at 16.72 agent-hours, parent s0041, category sync
- hypothesis: track A; checkpoint formats (MXFP4 experts with E8M0 scales, BF16 elsewhere). The track-A B=1 build finds the router's top-4 by ranking with 31 independent shuffles instead of four dependent argmax rounds (same experts, order and tie rule; bitwise checks pass). Expect B=1 1876.5 -> ~1873, B=8 ~3394 unchanged.
- evaluator: valid: 1871 us/step at B=1 and 3394 at B=8 over HTTP (online (submit/poll)); in process 1844 us/step at B=1 (generate, MBU 0.600)

### #48 s0048: B=1 build: router bias loaded at the start of the routing warp

- status: **build_failed**, no score, at 16.99 agent-hours, parent s0046, category sync
- hypothesis: track B; formats as s0046. In the B=1 build the routing warp loads its router bias before the 132 partial-sum loads instead of after the reductions (same values, one dependent load off phase E's long pole; bitwise checks pass). Expect B=1 1625 -> ~1622 (same-card gen -3.2), B=8 unchanged.
- evaluator: engine/build.sh failed (rc 1)

### #49 s0049: B=1 build: router bias loaded at the start of the routing warp (resubmit, fixed build)

- status: **valid**, http_step_us_b1 = 1617 (B=8: 3399), at 17.13 agent-hours, parent s0046, category sync
- hypothesis: track B; formats as s0046. In the B=1 build the routing warp loads its router bias before the 132 partial-sum loads instead of after the reductions (same values; bitwise checks pass). s0048 was the same idea with a duplicated patch that did not compile. Expect B=1 1625 -> ~1622 (same-card gen -3.2), B=8 unchanged.
- evaluator: valid: 1617 us/step at B=1 and 3399 at B=8 over HTTP (online (submit/poll)); in process 1609 us/step at B=1 (generate, MBU 0.614)

### #50 s0050: B=1 build: gate_up bias line prefetched into L1 when each unit's task is built

- status: **valid**, http_step_us_b1 = 1613 (B=8: 3388), at 17.25 agent-hours, parent s0049, category memory
- hypothesis: track B; formats as s0049. In the B=1 build each gate_up unit's 64-B bias line is prefetched into L1 when its task descriptor is built, so the epilogue after the whole-K stream no longer waits on a cold dependent load (prefetch only; same bits). Expect B=1 1617 -> ~1613 (same-card gen -4.1), B=8 unchanged.
- evaluator: valid: 1613 us/step at B=1 and 3388 at B=8 over HTTP (online (submit/poll)); in process 1606 us/step at B=1 (generate, MBU 0.616)

