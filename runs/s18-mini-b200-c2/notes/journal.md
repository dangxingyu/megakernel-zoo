# Journal (run s18-mini-b200-c2, direction: batch-8 speculation)

## s0001 (track A, quick) — step() for B>=2 on the new wide pass (engine/wk.cu)
Expected: all gates pass (batch p99 ~0.15 vs 0.254), B=8 step slower (~10 ms). Measured: valid; numeric batched median
0.0257 (megakernel 0.0267), p99 0.094; invariance bitwise; generate 8/8; b8 step 10157 us in process.
Concluded: a batch-invariant B>=2 verifier with its own arithmetic passes every gate; B=1 stays the megakernel.

## s0002 (track C) — B>=2 cohorts by draft-and-verify (8 FusedDraft clones, one wide verify pass per cycle)
Expected: b8 6302 -> ~3500, natural_b8 6508 -> ~5700, b1 unchanged. Measured: b1 1523, natural_b1 1015, b8 8761 (!),
natural_b8 5364; tokens/event b8 2.66, natural_b8 4.64; http_tokens 8/8, hidden pass.
Concluded: the prediction for b8 was wrong by 2.5x: on the evaluator prompts acceptance at B=8 is 2.66 tokens/cycle
(my word-list test prompts gave 7.6: they loop), and a cycle with ~40-50 rows costs ~23 ms (wide pass 10.7 ms at 8
rows, 25 ms at 64). The wide pass's fixed cost must come down before speculation pays on low-acceptance text.

## s0003 (track C) — PDL, attention staging, GEMM prefetch depth 4
Expected: natural_b8 5364 -> ~4500, b8 8761 -> ~7400. Measured: natural_b8 4476, b8 7181, b1 1523. Concluded: held.
In process: wide pass 10.7 -> 9.2 ms (8 rows), 25.5 -> 20.7 ms (64 rows); natural serving 5733 -> 4782 us/token.

## s0004 (track C) — attention items of several chunks, K/V double-buffered (bit-identical, A/B 0 rows differ)
Expected: -4% (in-process A/B: -3.5% at 8 rows, -4.4% at 64). Measured over HTTP: natural_b8 4464 (-0.3%), b8 7186 (0).
Concluded: the same-job A/B used a constant token (cheap routing, no realistic MoE spread); over HTTP the attention
part of the cycle is smaller than that A/B suggested. Off-submission findings (all in process, natural B=8 serving):
- cycle = verify 14.9 ms (93%) + 8 drafts 1.06 ms on side streams (1.68 serially);
- per-sequence row caps 3/4/5 are worse than 8 (6306/5767/5829 vs 5524 us/token); draft threshold 0.2/0.4/0.6 within 3%;
- MoE: weight-ring depth 4/6/8 KB per warp and two independent mma chains per unit: no change (8949-9032 us at 8 rows).
- fusing the MoE sum and conv kernels (18 blocks): no gain, left switched off (g_fuse2 = 0).

## s0005 (track C) — routing tables in parallel, 24-warp register-pipelined MoE, GEMM L2 prefetch (bit-identical)
Expected: natural_b8 4464 -> ~4000, b8 7186 -> ~6400. Measured: natural_b8 3966, b8 6437, b1 1522. Held.
Finding behind it: in-kernel marks showed the routing tables (built in every block at the start of k_w13) took 42 us
per application at 64 rows (serial per-expert scans over 512 entries, a one-warp chunk builder); masks + scans -> 15 us.
A diagnostic trap: "loads only" / "compute only" MoE builds produce garbage, the router then sees NaN and routes every
row to experts 0..7: those builds move far fewer bytes, so their times say nothing (my first reading was wrong).
128 unrolled warp ballots measured 13 us where a per-expert loop over the rows takes 1 us (kept the loop).
Doubling every FP4 conversion: +0.6 ms (8 rows) / +1.8 ms (64 rows) per pass -> conversions are ~1/6 of the MoE.
A race caught by the A/B (4772 rows differ): GEMM lane partials aliased onto the B staging buffer without a barrier.

## s0006 (track C) — 64-row GEMM chunks, row-job weights staged before the PDL wait, loads hoisted (bit-identical)
Expected: natural_b8 3966 -> ~3800, b8 6437 -> ~6150. Measured: natural_b8 3811, b8 6353 (-1.3%, half the expected).
Off-submission: splitting the 16-lane GEMMs' lanes over 4 blocks with global partials (dgemm3, bit-identical) was
slower (o_proj 20 -> 32 us at 64 rows): B staging was not their bound; left off (DG3_ON 0).
MoE: B fragments by ldmatrix + running weight pointer: -2.5% pass at 64 rows. Instruction mix of the NVFP4 unit
(SASS, 3 units of 2 n-groups): F2FP 51, HMUL2 44, IMAD 44, HMMA 24, LDSM 11 of 257. Per-SM F2FP count x 4068 units
(w13, 64 rows) at 16 conversions / clock would be ~80 us, the measured item time: conversions are a likely bound.

## s0007 (track C) — MoE input groups double-buffered, items round-robin over the block range (bit-identical)
Expected: natural_b8 3811 -> ~3600, b8 6353 -> ~5950. Measured: natural_b8 3641, b8 6048, b1 1521. b8 is now below the
inherited 6302 (natural_b8 -44% vs 6508). Off-submission (all bit-identical): attention chunks-per-item 1 item per
(sequence, head) 57 vs 42 us (worse); relp for the whole item + bias fused into the score epilogue: no change; q/r
preparation moved into k_kv: -0.3%. Per later chunk of an attention item: relp+issue 1.3, bias 0.8, K/V wait 0.3,
scores 1.3, softmax 1.0, PV+records 1.0 us at 8 rows: each step is barrier + latency, not work. The 16-B weight
L2 prefetch in dense GEMMs: no measurable effect (on / off / after B within noise).

## s0008 (track C) — INCORRECT (http_tokens: engine crashed with an illegal address)
Cause 1: dense-GEMM kernels launched with shared memory sized from the host's row bound (64 in the B=8 cycle -> the
64-row variant, 66.5 KB) while the kernel picks its variant from the device-side row count: 9..32 rows took the 16 / 32
row variants with full-K staging (74 / 148 KB) -> shared-memory overflow. My serving tests had high acceptance (~64
rows per cycle) and every direct test launched with the exact row count, so neither covered it; the evaluator's
32-token prompts did. Cause 2 (found while fixing): the CUDA-graph capture ran a warm-up pass on a fake row (sequence 0,
position 0), overwriting that sequence's cached K/V mid-request (7/8 streams in the first request). Also: a 0-row
pass would have run the GEMM epilogue at negative row indices.
Fix: k-stage sizes per variant so none exceeds the bound's (36/16/16/8 units for 4-lane GEMMs), launch size = max over
the variants up to the bound; capture without executing; no epilogue for empty passes. New tests: tests/w/wbound.py
(ragged rows under the fixed bound 64 vs exact launches: 0 of 1586 rows differ), wserve with 32-token prompts and
1120 tokens, staggered arrivals: all 8/8.
Fused phase groups (cooperative kernels with grid barriers, bit-identical): slower in every combination (fuse off).

## s0009 (track C) — s0008 fixed: valid. natural_b8 3577 (expected ~3550), b8 5948 (expected ~5900), b1 1521.

## gate F/G epilogue moved into the norm phase
First version not bit-identical (4800 rows differ): the old epilogue had compiled to fma(1 - f, x, g * m) (SASS: FADD,
FMUL, FFMA), the new code was contracted differently. With the operation sequence written as intrinsics: 0 rows differ.
Rule: when moving arithmetic between kernels, check the old SASS and write its sequence explicitly.
s0010 (track C): valid. Expected natural_b8 3577 -> ~3510, b8 5948 -> ~5850. Measured natural_b8 3573, b8 5930, b1 1522:
-0.1% / -0.3%, a fifth of the prediction (the gate-F/G phases are ~3% of a pass; their epilogue was not the cost).
Top-8 routing with two rows per warp (bit-identical): neutral (R=48 13638 vs 13599 us); kept.

## Per-launch timeline (tests/w/wtl.py, -DWTPROF build, ctx 600): where the pass goes
Gaps between a kernel's end and its successor's wait release: 1.0-1.9 us per launch, ~1.0 ms per pass of 675 launches
at every row count. Active time (wait release -> last warp exits) per pass:
  R=8 (7.7 ms): w13 1526 (32/app), attn 1260 (26), w2 894 (19), oproj 333, qkv 540, gates ~1100, router 183, kv 200.
  R=48 (14.2 ms): w13 3935 (82), w2 2024 (42), attn 1643 (34), oproj 691, qkv 1067, gate D 930, router 577.
=> the MoE is 42% of a 48-row pass, attention 12%, launch gaps 7%.

## MoE / GEMM / draft measurements (after s0010, all on card 1)
- MoE (1-application passes, NAPP=1): a loads-only build (MOE_DIAG=2) takes ~96% of the full w13 time at 8-64 rows;
  without FP4 conversions (MOE_DIAG=1) it is not faster. Per block (WDPROF marks): routing 5.4 / 10.8 / 16 us at 8 / 48 /
  64 rows (the top-8 loop alone 1.3 -> 4.3 us from 4 to 24 busy warps: redux throughput), then streaming. ncu k_w13
  (64 rows): issue 55% of cycles, stalls per instruction: wait 2.3, barrier 2.0, not-selected 1.5, math-throttle 1.4,
  long-scoreboard 1.1 of 10.8; ALU pipe 46%, FMA-heavy 35%. No single bound to remove.
- Dense GEMMs at 48 rows: qkv 22 us, o_proj 14, gate D 20, router 12 (3.8 ms per pass vs 1.5 at 8 rows). Double-buffered
  B stages (DGDB, bit-identical): no change. ncu k_qkv: barrier 5.6, no-instruction 5.1, short/long scoreboard 3.3 / 3.2
  stalls per instruction: latency of a short dependent chain, not bytes.
- Drafts (8 sequences per cycle): one draft alone ~110 us, 8 side by side 1.06 ms. All 8 in one launch per kernel
  (dk_run_multi, block -> sequence b % 8; bit-identical drafts: same cycles / events): 1.01 ms. The 8-sequence base
  logits GEMV took 568 us of it (each sequence's blocks decode the FP8 unembedding again and read it from L2 again).
  k_base8 (every FP8 fragment decoded once for all 8 sequences, X staged per k half; staging with 8 loads in flight per
  thread): base 568 -> ~200 us, draft 0.854 ms per cycle, tokens/event unchanged. Tried and slower: E4M3 X rows with
  m16n8k32 E4M3 MMA (1.17 ms), next-unit weight prefetch in k_gemv (1.17), a shared-weight kernel for the layer GEMVs
  (k_gemvm: 1.04; 1 block per SM, staging and compute serialized).

## s0011 (track C) — 8 drafts in one graph, shared-decode base logits (verifier bit-identical)
Expected natural_b8 3573 -> ~3520, b8 5930 -> ~5830. Measured b8 5740 (-3.2%, more than expected: the evaluator's
word-list prompts have small passes, so the draft is a larger share), natural_b8 3603 (+0.8%: inside the repeat spread,
decode 1971 / 2221 / 2361 tok/s min / median / max vs 1973 / 2239 / 2333 before; tokens per event identical), b1 1526.
Top-8 with one reduction + ballot per round (bit-identical): top-8 phase unchanged (4.6 us at 48 rows) -> left off (TOP8B 0).

## Draft length threshold for B >= 2 cohorts (MK_CUM_TH8, B = 1 keeps MK_CUM_TH 0.4)
wserve 512 tokens, us/token/stream: random-token prompts (2.6-2.8 tokens/event, like the evaluator's b8 2.66):
theta 0.2 6095, 0.4 5769, 0.5 5663-5705, 0.6 5617-5649, 0.7 5611-5646, 0.8 5703; natural: 0.2 4547, 0.4 4435,
0.5 4376-4386, 0.6 4370-4403, 0.7 4347-4358, 0.8 4452. -> 0.6 (s0012).
Attention relp rows prefetched one chunk ahead / before the PDL wait (bit-identical): 8 rows 26.3 -> 25.0 us per
application, 48 rows 34.1 -> 36.4 (the restructured kernel alone +1.1 us at 48 rows) -> reverted.
s0012 (track C, theta 0.6 for B >= 2): valid. Expected b8 5740 -> ~5600, natural_b8 -> ~3520. Measured b8 5616 (-2.2%),
natural_b8 3613 (decode 1941 / 2214 / 2377 tok/s min / median / max: the natural median moves +-1% between equal
engines), tokens/event 2.58 / 4.52 (from 2.66 / 4.64), b1 1525.
Fewer launches, bit-identical but slower (same-job A/B vs the committed library, ctx 300): MoE output conv computed
by the next application's row jobs (FW2C) +2% at 64 rows; MoE output sum by the last-arriving expert item per
(row, strip) in the w2 kernel (FW2S: fence + atomics per item) +4%. Reverted.

## Top-8 once per 8-row group (FRT, bit-identical)
Router GEMM as (strip, 8-row group) blocks (dgemm2 8-row variant, the same per-row arithmetic); the last of a group's
8 blocks computes the group's top-8 (one row per warp) into global memory; every k_w13 block loads it instead of
computing all rows. Same-job A/B vs the committed library: 64 rows 14300 -> 13673 us (-4.4%), 8 rows -0.5%;
wtime1 48 rows 13598 -> 13229; wserve natural 4370 -> 4294, random 5617 -> 5569 us/token/stream. All tests OK.
s0013 (track C, FRT): valid. Expected b8 5616 -> ~5520, natural_b8 -> ~3540. Measured b8 5528, natural_b8 3501 (best), b1 1525.

## Gate-D GEMM in 8-row groups (GDG8, bit-identical)
k_gate_dg ran on 12 (and 4) blocks, one strip each over all rows (20 us at 48 rows vs 6 at 8): now (task, 8-row group)
blocks with the 8-row dgemm2 variant. Same-job A/B: 64 rows 13654 -> 12873 us (-5.7%), 8 rows -1%; wtime1 48 rows
13229 -> 12514; wserve natural 4294 -> 4133, random 5569 -> 5394 us/token/stream. All tests OK.
Gate F/G staging with float4 loads (GFG4): no change (left off).
s0014 (track C, GDG8): valid. Expected b8 5528 -> ~5360, natural_b8 3501 -> ~3380. Measured b8 5335, natural_b8 3345, b1 1524.

## Gate F/G in 8-row groups (GFG8, > 8 rows) + FRT only above 8 rows (bit-identical)
The timeline after s0014: router 7.4 us at 8 rows (3.8 before FRT: the group's top-8 tail), gate F/G 12.4 us at 48
rows (3.6 at 8). Gate F/G as (warp-task group, 8-row group) blocks for > 8 rows; FRT for > 8 rows only (the kernel
chooses by the device row count). Same-job A/B vs s0014: 8 rows 6085 -> 5964 (-2.0%), 64 rows 12854 -> 12661 (-1.5%);
wtime1 48 rows 12514 -> 12250; wserve natural 4133 -> 4069-4096, random 5394 -> 5340-5372. Tests OK.
Draft with FP8 layer weights too (MK_DK_FP8=all): random 5411 vs 5372, natural 4108 vs 4096 -> unchanged (3).
