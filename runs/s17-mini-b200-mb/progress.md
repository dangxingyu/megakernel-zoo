# s17-mini-b200-mb: Inkling-2-Mini on GB200

Score: `http_step_us_b1` (lower is better). Budget 5.0 agent-hours. 21 submissions so far, 21 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 692
- 1.10x the fastest general engine: 3370
- SGLang (PR #51926 image) (general engine): 4329
- SGLang, GB300 recipe flags at TP1 (general engine): 4557
- vLLM (inkling-2-mini image) (general engine): 3707

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Track A base: s15-mini-b200-a2 s0025 engine unchanged (merge base)

- status: **valid**, http_step_us_b1 = 2900 (B=8: 5860), at 0.09 agent-hours, category structure
- hypothesis: track A (checkpoint formats: BF16, NVFP4 codes+E4M3 scales+amax factor, MXFP8 codes+UE8M0; expert inputs BF16 unquantized; --use_fast_math in vector code as a2). Re-evaluation of a2 s0025 source in this run as the merge base: expect 2755 +-30 us at B=1 and 5825 +-60 at B=8 over HTTP (a2 s0025 measured 2755 / 5825).
- evaluator: valid: 2900 us/step at B=1 and 5860 at B=8 over HTTP (online (submit/poll)); in process 2856 us/step at B=1 (generate, MBU 0.242)

### #2 s0002: Re-evaluate b2 s0047 unchanged (card/pod calibration vs s0001)

- status: **valid**, http_step_us_b1 = 2949 (B=8: 6511), at 0.17 agent-hours, parent s0001, category other
- hypothesis: track B (MSGZ, SKH, L0T, GLIN as b2 s0047; formats as the checkpoint). Calibration: b2 s0047 measured 2879 / 6509 in its run; s0001 (a2 s0025) came out 145 us slower at B=1 than in a2's run. If this card is slower, expect ~3000 / ~6550; if s0001 was placement noise, ~2880 / 6510.
- evaluator: valid: 2949 us/step at B=1 and 6511 at B=8 over HTTP (online (submit/poll)); in process 2910 us/step at B=1 (generate, MBU 0.227)

### #3 s0003: ATT3: B>=3 attention by 3 independent 8-warp teams per block, pipelined K/V tiles

- status: **valid**, http_step_us_b1 = 3190 (B=8: 5701), at 0.74 agent-hours, parent s0002, category attention
- hypothesis: track A (checkpoint formats unchanged; also K1 = batch-1 kernel compiled separately, -8/-16 us on card 1). At B=8 attention is 10.8 us/application at ctx 200 and 31.8 at ctx 1000 (3.4 waves of 128-key block items, each a sync chain). ATT3: one 64-key record per 8-warp team (3 teams per block, named barriers, swizzled unpadded tiles in the same smem, next record's K/V tile loaded during softmax/PV), block-level merge. Bit-identical to s0001's arithmetic (bitcheck B=1/B=8, 1152 positions). Card 1: -298 us at ctx ~960, -64 at ctx 64-320. Expect B=8 HTTP 5860 -> ~5680 (-180), B=1 unchanged ~2900.
- evaluator: valid: 3190 us/step at B=1 and 5701 at B=8 over HTTP (online (submit/poll)); in process 3164 us/step at B=1 (generate, MBU 0.218)

### #4 s0004: ATT3 in the single kernel (K1 dropped): team attention for B>=3

- status: **valid**, http_step_us_b1 = 2911 (B=8: 5701), at 0.96 agent-hours, parent s0003, category attention
- hypothesis: track A (checkpoint formats unchanged). s0003 = K1 + ATT3; K1 measured +384 us at B=1 once the test harness really used it (2752 vs 3136 on card 1), which explains s0003's 3190. This is s0001's kernel + ATT3 (bit-identical; card 1: B=1 +14 us noise, B=8 -81 at ctx 64-320, -298 at ctx ~960). Expect B=1 ~2900 (s0001), B=8 ~5700 (s0003: 5701).
- evaluator: valid: 2911 us/step at B=1 and 5701 at B=8 over HTTP (online (submit/poll)); in process 2870 us/step at B=1 (generate, MBU 0.241)

### #5 s0005: Dual engine: a2 lineage at B=1, a1 s0025 kernel at B>=2 (track A)

- status: **valid**, http_step_us_b1 = 2888 (B=8: 5464), at 2.14 agent-hours, parent s0004, category structure
- hypothesis: track A (checkpoint formats; two engines, each its own layout of the same weights). Structure from s17-mini-b200-ma s0002: B=1 served by our a2-lineage kernel (s0004's without ATT3), B>=2 by s15-mini-b200-a1 s0025's kernel (9x2 ring stages). Card 1, B=8 same job: a1 5919 vs a2+ATT3 6355 at ctx ~960, 5927 vs 6003 at ~560. Expect B=8 HTTP 5701 -> ~5450 (ma s0002 measured 5450), B=1 ~2900 (unchanged kernel). Batch gate: B=1 and B=8 now different arithmetic (not bitwise; within tolerance as ma s0002).
- evaluator: valid: 2888 us/step at B=1 and 5464 at B=8 over HTTP (online (submit/poll)); in process 2849 us/step at B=1 (generate, MBU 0.242)

### #6 s0006: SKH: certified int8 head sketch + exact rescoring on the greedy path (B=1 engine)

- status: **valid**, http_step_us_b1 = 2867 (B=8: 5480), at 2.21 agent-hours, parent s0005, category memory
- hypothesis: track B (formats as the checkpoint; adds an int8 copy of the unembedding used only to bound logits, the argmax is recomputed from the BF16 rows). Port of s15-mini-b200-b2's SKH onto the a2 lineage (subagent): the head reads the 0.46 GB sketch, then ~1-2 exact strips per token; certificate redone for fast-math tanh (eps 2.4e-4 measured over all float32 inputs at load). Greedy tokens identical to track A (4 texts x 1024 steps, B=1..8, contexts to 2048), logits path unchanged (bitcheck). Card 1: -49 us at B=1. Expect B=1 2888 -> ~2845, B=8 unchanged ~5465 (a1 engine).
- evaluator: valid: 2867 us/step at B=1 and 5480 at B=8 over HTTP (online (submit/poll)); in process 2811 us/step at B=1 (generate, MBU 0.241)

### #7 s0007: B>=2 engine: a1 kernel with ma s0004's gate phase 0 / message phase (track A part)

- status: **valid**, http_step_us_b1 = 2862 (B=8: 5364), at 2.41 agent-hours, parent s0006, category structure
- hypothesis: track B (SKH on the B=1 engine as s0006; the B>=2 engine is track A). The a1 kernel taken from s17-mini-b200-ma s0004 (every block mixes its own sequence in gate phase 0, one warp per message in phase M; ma measured 5450 -> 5355 at B=8). Card 1, same workload: B=8 generate 5727 -> 5612 (-115). Expect B=8 HTTP 5480 -> ~5380, B=1 unchanged ~2870.
- evaluator: valid: 2862 us/step at B=1 and 5364 at B=8 over HTTP (online (submit/poll)); in process 2801 us/step at B=1 (generate, MBU 0.238)

### #8 s0008: Track A of the current tree: dual engine with ma s0004 a1 kernel, no SKH

- status: **valid**, http_step_us_b1 = 2892 (B=8: 5349), at 2.48 agent-hours, parent s0007, category structure
- hypothesis: track A (checkpoint formats; two engines, each its own layout). Same tree as s0007 with the track-A switch (no SKH on the B=1 engine). Expect B=1 ~2885 (s0005: 2888), B=8 ~5365 (s0007: 5364): track B (SKH) minus track A ~ -20 us at B=1, 0 at B=8.
- evaluator: valid: 2892 us/step at B=1 and 5349 at B=8 over HTTP (online (submit/poll)); in process 2853 us/step at B=1 (generate, MBU 0.242)

### #9 s0009: Start-up placement tuning of the B=1 engine's polled buffers on the evaluating card

- status: **valid**, http_step_us_b1 = 2838 (B=8: 5352), at 2.60 agent-hours, parent s0008, category memory
- hypothesis: track B (SKH as s0007; placement is layout only). Card 0 of this pod runs the a2 kernel ~150 us slower at B=1 than card 1 (2801 vs ~2690 generate for the same build); card 1 shows no placement spread (9 trials 2698-2709). If card 0's deficit is placement-dependent, the 9 start-up trials (b2 s0010 method, B=1 objective) recover part of it: B=1 2862 -> ~2750; if not, unchanged ~2862 +-15. B=8 unchanged ~5360. stats() records the 9 trial times for card 0.
- evaluator: valid: 2838 us/step at B=1 and 5352 at B=8 over HTTP (online (submit/poll)); in process 2783 us/step at B=1 (generate, MBU 0.244)

### #10 s0010: Final-source check: s0009 configuration on the final tree

- status: **valid**, http_step_us_b1 = 2838 (B=8: 5347), at 2.84 agent-hours, parent s0009, category other
- hypothesis: track B (SKH on the B=1 engine + start-up placement tuning; B>=2 engine track A). Same source as s0009 apart from notes, the off-by-default TUNE8 code in a1/eng.py (not active) and tests. Expect 2838 +-30 / 5352 +-30 (repeatability).
- evaluator: valid: 2838 us/step at B=1 and 5347 at B=8 over HTTP (online (submit/poll)); in process 2804 us/step at B=1 (generate, MBU 0.242)

### #11 s0011: Start-up placement tuning of the B>=2 engine's polled buffers (TUNE8)

- status: **valid**, http_step_us_b1 = 2837 (B=8: 5377), at 2.91 agent-hours, parent s0010, category memory
- hypothesis: track B (as s0010; placement is layout only, tokens identical). On card 1 the a1 engine's 9 placements spread only 13 us; card 0 showed a 34 us spread for the B=1 engine. Expect B=8 5347 -> ~5330 if card 0 has a similar spread for a1, else unchanged +-25; B=1 unchanged ~2838. stats b8_placement_us records card 0's trials.
- evaluator: valid: 2837 us/step at B=1 and 5377 at B=8 over HTTP (online (submit/poll)); in process 2796 us/step at B=1 (generate, MBU 0.242)

### #12 s0012: SKH in the B>=2 engine too: certified int8 head sketch in the a1 kernel's greedy path

- status: **valid**, http_step_us_b1 = 2839 (B=8: 5335), at 3.01 agent-hours, parent s0011, category memory
- hypothesis: track B (int8 copy of the unembedding used only to bound logits; argmax recomputed from BF16 rowtiles). Subagent port of the sketch head into the a1 kernel (bitcheck exact path B=2/8 identical; greedy tokens identical B=2..8 to ctx 2048 and serving). Card 1: -16 us at B=8 (agent, 3x3 instances), 0 on my workload: expect B=8 5347 -> ~5335 +-25, B=1 unchanged ~2838.
- evaluator: valid: 2839 us/step at B=1 and 5335 at B=8 over HTTP (online (submit/poll)); in process 2801 us/step at B=1 (generate, MBU 0.242)

### #13 s0013: B=1 engine compiled for batch 1 only with the inactive mkprof mark sites (KB1+MKPROF)

- status: **valid**, http_step_us_b1 = 2764 (B=8: 5366), at 3.14 agent-hours, parent s0012, category launch
- hypothesis: track B (as s0012; the B=1 kernel is the same source with the batch a compile-time 1 and the mkprof mark sites compiled in but inactive: no profile buffer). Idea from s17-mini-b200-ma s0003 (KB1 alone is a +375 us code-generation cliff, with the mark sites -20). Needed a fix: SKH's bound pass had an unguarded profile write (crashed MKPROF builds). Card 1, 3 instances each: B=1 2717 -> 2676 (-40), greedy tokens identical (4 texts x 1024). Expect B=1 2839 -> ~2805 (layout effect, may transfer partly), B=8 unchanged ~5335.
- evaluator: valid: 2764 us/step at B=1 and 5366 at B=8 over HTTP (online (submit/poll)); in process 2713 us/step at B=1 (generate, MBU 0.250)

### #14 s0014: Track A of the final tree: KB1+MKPROF B=1 engine without SKH, a1 at B>=2

- status: **valid**, http_step_us_b1 = 2799 (B=8: 5369), at 3.21 agent-hours, parent s0013, category structure
- hypothesis: track A (checkpoint formats; layout-only start-up placement and compile options). The s0013 tree with the track-A switch: no head sketch in either engine. Expect B=1 ~2795 (s0013 + SKH's ~30 us), B=8 ~5350 (s0008: 5349).
- evaluator: valid: 2799 us/step at B=1 and 5369 at B=8 over HTTP (online (submit/poll)); in process 2773 us/step at B=1 (generate, MBU 0.249)

### #15 s0015: B>=2 engine: ma s0006/s0007 expert decode (NVFP4 scales on FMA pipe, MXFP8 to bf16 bits)

- status: **valid**, http_step_us_b1 = 2767 (B=8: 5265), at 3.33 agent-hours, parent s0014, category gemm
- hypothesis: track B (as s0013; the a1-side change is track A: bitwise-equal decode reorganisation, ma's exhaustive E4M3/UE8M0 check). Taken from s17-mini-b200-ma s0006 + s0007 (their B=8 5355 -> 5231), patched onto our a1 copy (SKH:8, mk_set_params). Card 1: B=8 5726 -> 5592 (-134, 2 processes each), tokens identical at B=8 and B=1. Expect B=8 5366 -> ~5240, B=1 unchanged ~2765.
- evaluator: valid: 2767 us/step at B=1 and 5265 at B=8 over HTTP (online (submit/poll)); in process 2719 us/step at B=1 (generate, MBU 0.249)

### #16 s0016: B>=2 engine: faster sketch-digit stage in the a1 head (same digits)

- status: **valid**, http_step_us_b1 = 2767 (B=8: 5240), at 3.98 agent-hours, parent s0015, category memory
- hypothesis: track B (as s0015). The a1 head's SKH digit stage (int8 digits of the final hidden state) as one flat loop with exact add/subtract rounding and byte packing: the same digit values (|y| <= 127 so the clamps never acted), tokens identical. Profile: digit writes 7.9 -> 3.3 us per head; card 1, 4 process pairs: -8..-64 us (mean -38, mostly placement noise). Expect B=8 5265 -> ~5255 (-5..-35), B=1 unchanged ~2767.
- evaluator: valid: 2767 us/step at B=1 and 5240 at B=8 over HTTP (online (submit/poll)); in process 2720 us/step at B=1 (generate, MBU 0.249)

### #17 s0017: Track A of the final tree (s0016 without the head sketches)

- status: **valid**, http_step_us_b1 = 2800 (B=8: 5221), at 4.05 agent-hours, parent s0016, category structure
- hypothesis: track A (checkpoint formats; layout-only compile options and start-up placement). The s0016 tree with the track-A switch: no SKH in either engine (the a1 digit-stage change is SKH code, unused here). Expect B=1 ~2800 (s0014: 2799), B=8 ~5255 (s0015/s0016 minus SKH:8's ~-10).
- evaluator: valid: 2800 us/step at B=1 and 5221 at B=8 over HTTP (online (submit/poll)); in process 2773 us/step at B=1 (generate, MBU 0.249)

### #18 s0018: B>=2 engine: ma s0009 NVFP4 scale-pair decode; sketch head only in the B=1 engine

- status: **valid**, http_step_us_b1 = 2757 (B=8: 5167), at 4.17 agent-hours, parent s0017, category gemm
- hypothesis: track B (SKH in the B=1 engine; the B>=2 engine is track A). a1 kernel + s17-mini-b200-ma s0009 (both block scales of a k16 block by one exact cvt.rn.f16x2.e4m3x2, bitwise equal per ma); SKH:8 dropped (s0016 5240 vs its track A s0017 5221; card 1 +10). Card 1, 2 processes each: B=8 5584 -> 5442 (-142), tokens identical. Expect B=8 ~5100 (s0017 5221 - ~120), B=1 unchanged ~2767.
- evaluator: valid: 2757 us/step at B=1 and 5167 at B=8 over HTTP (online (submit/poll)); in process 2714 us/step at B=1 (generate, MBU 0.250)

### #19 s0019: Track A of the final tree (s0018 without the head sketch)

- status: **valid**, http_step_us_b1 = 2801 (B=8: 5170), at 4.24 agent-hours, parent s0018, category structure
- hypothesis: track A (checkpoint formats; layout-only compile options and start-up placement). The s0018 tree with the track-A switch (no SKH in the B=1 engine; the B>=2 engine identical to s0018's). Expect B=1 ~2795 (SKH ~-35), B=8 ~5167 (same engine as s0018, +-30).
- evaluator: valid: 2801 us/step at B=1 and 5170 at B=8 over HTTP (online (submit/poll)); in process 2773 us/step at B=1 (generate, MBU 0.249)

### #20 s0020: B>=2 engine: ma s0010's a1 settings PMAXC 2 (attention pieces of <=2 chunks) and SCNT_REL 1

- status: **valid**, http_step_us_b1 = 2770 (B=8: 5128), at 4.34 agent-hours, parent s0019, category attention
- hypothesis: track B (SKH at B=1 as s0018; the B>=2 change is track A: the attention key range split into pieces of at most 2 128-key chunks changes only the merge's rounding, not the function; greedy tokens at B=8 differ at near-ties on our texts). From s17-mini-b200-ma s0010 (5141 -> 5119). Card 1: B=8 5443 -> 5394 (-49, 2 processes each). Expect B=8 5167 -> ~5135, B=1 unchanged ~2760.
- evaluator: valid: 2770 us/step at B=1 and 5128 at B=8 over HTTP (online (submit/poll)); in process 2724 us/step at B=1 (generate, MBU 0.249)

### #21 s0021: Track A of the final tree (s0020 without the head sketch)

- status: **valid**, http_step_us_b1 = 2806 (B=8: 5155), at 4.41 agent-hours, parent s0020, category structure
- hypothesis: track A (checkpoint formats; layout-only compile options, start-up placement, attention piece size). The s0020 tree with the track-A switch. Expect B=1 ~2800 (s0019: 2801), B=8 ~5130 (same B>=2 engine as s0020).
- evaluator: valid: 2806 us/step at B=1 and 5155 at B=8 over HTTP (online (submit/poll)); in process 2772 us/step at B=1 (generate, MBU 0.249)

