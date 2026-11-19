# What is specific to gpt-oss-20b on an H100 (each item with its measurement)

## Card
- Read stream (bench/bw.cu, 132 blocks, 16-B ld.global.nc.L1::no_allocate, 8 in flight/thread, cold, e0027):
  1160 MB: 3068 GB/s (1024 thr), 3052 (512), 2694 (256). 106 MB (one layer's bytes): 2.78 TB/s incl. launch
  ramp; 53 MB 2.43; 16 MB 1.48; 4 MB 0.65 TB/s. -> floor for 3.72 GB at ~3.05 TB/s: ~1220 us.
- mma.sync bf16 takes subnormal A operands; HMUL2.BF16 keeps subnormals (bench/subn.cu, e0003).

## Where the B=1 step goes (measured, e0137-e0149)
- Skeleton: weights served from a 512 KB L2-resident window (SKEL build) B=1 1707 us vs 2175 real (gen 256);
  + no e2m1 decode 1570. Phase spans in SKEL: gate_up 13.8, down 13.5, qkv 9.1, o 7.3 us/layer.
- Nsight Compute, one generate step (e0143/e0144): schedulers issue on 18% of cycles; ALU pipe 15% of peak,
  math-pipe throttle 0.29 cycles/instr; stalls per issued instruction: CTA barrier 7.4, long scoreboard 4.7,
  wait 1.1. ~18% of stall samples at mbarrier waits (TMA data), ~35% at BAR.SYNC sites. Not pipe-bound.
- Standalone gate_up phase (bench/ringbw.cu, 1440 whole-K units, cold L2 via a 120 MB memset): stream only
  21.1 us (1.67 TB/s incl. launch), + decode/MMA 22.8 us, + MMA without decode 21.5; ring depth RS 2-8, stage
  size RI 6-18, warps 4-16 per SM: all 22.5-25 us; direct ld.global.nc register loads 20-21 us. A cold 35 MB
  burst costs ~15-20 us on this card whatever the load mechanism: the MoE phases at B=1 pay the HBM ramp
  because expert weights cannot be requested before routing. Decode is ~7% of the phase.

## Verification of the current engine (e0203, e0204, soaks)
- tools/kit/bitcheck.py, 600 positions, B 1 and 8: track A vs track B step() logits bit-identical (1.0, max |d| 0);
  batch invariance bit-identical in both builds.
- tests/bcheck.py: for B = 2..8 every row equals that sequence run alone, bit for bit; contexts up to 2036 at
  B=1 and B=8: finite logits, generate() == step()-greedy over 60 tokens.
- Track B greedy path (certified 6-bit head sketch): tests/soak_greedy.py 27/27 sequences x 400 tokens identical
  to exact step-greedy; earlier int8 sketch 54/54 x 256-512 tokens.
- Sketch bound (engine/api.py pack_sketch): |logit_exact - sketch| <= en_i ||x|| + qn_i ||r|| with
  en_i >= ||w_i - s_i q_i|| + 2^-10 ||w_i|| (float64, rounded up to float32) and qn_i >= ||s_i q_i||; x = t (128 hi + lo)
  + r with integer digits so the integer mma sums are exact; ||x||, ||r|| computed in float32 with x1.001 and
  + 2^-20 ||x|| margins; the sketch value's own float32 evaluation gets |s| 2^-20. The 2^-10 ||w|| ||x|| term covers
  the exact path's float32 accumulation (2880 products: <= 2880 * 2^-23 = 3.4e-4 relative to sum |w x| <= ||w|| ||x||,
  also under truncating adds). Candidate tiles are recomputed with the exact head's own mma chain (same bits as
  step()), so the chosen token equals the exact argmax with the same lowest-index tie rule.
- Experts census (tests/zero_census.py): zero rows 0, zero 16x64 items 0, zero 32-groups 0 (gate_up and down): nothing exactly skippable.

## Final hidden state and the head (tests/sketch_study2.py, e0235; real text, 8 x 64 positions, fp32 reference)
- Energy of the final normed hidden state is concentrated: one dim holds 20.1% of mean |x|^2, top 16 dims 28.9%,
  top 64 36.0%, top 256 49.9%. Allows: certified head sketches that keep a few columns exact have much smaller
  bounds (int6 mean candidate rows 130 -> 59 with 16 exact dims; int5 7476 -> 1937). Does not allow: anything
  uncertified (the sketch must still bound the rest for every input).
## Router fold (s0015): ln2_k * Wr_ek is exact in fp32 for every layer (asserted at load against float64).
- L2 prefetch of weights on this card loses whenever it is large (7 attempts, last: predicted experts' first gate_up
  stages during the HBM-idle attention window, top-4/5/6: +29/+40/+60 us, e0342): bulk L2 prefetch of the down experts' 17.6 MB
  during gate_up's 35 MB stream: +112 us/step at B=1 (e0256); next-layer qkv during attention: +145 (e0142); small
  prefetches pay (K/V window, <= 2 MB/layer: -13 us, s0016). Likely L2 capacity/eviction (50 MB L2, two halves)
  and contention with the TMA streams on the critical path. Allows: prefetch of small, soon-used data. Rules out:
  merging the MoE bursts through L2.
- MXFP4 storage roundtrip (tests/roundtrip.py, e0328): the engine's expert layout (pack_fp4 nibble/item
  permutation + fuse_items: 512 B codes then 32 B scales per 64-column item) inverted on all 48 expert tensors
  (10,152,345,600 bytes of blocks + scales) reproduces the checkpoint bytes exactly. Allows: the layout in both
  tracks (pure byte permutation).
- Verification of s0030 / track A at s0029 code (e0326): bcheck B=2..8 bitwise vs alone, contexts to 2036 (both
  tracks); soak 27/27 x 400 tokens (track B); limits; HF bf16 comparison (track A B=8 vs HF median 0.063).
- Output head census (tests/head_census.py, e0345): no all-zero rows, no row with < 16 nonzeros, 201083 distinct
  row hashes of 201088; row norms 0.0149 .. 0.426 (median 0.288). Cauchy-Schwarz skipping (tests/norm_skip.py,
  e0346, real text): ||w_i|| ||x|| < top logit for only 0.5% of rows (||x|| median 227, top logit median 16).
  Rules out: skipping head rows without reading them.

## Head sketch with group scales (tests/tier_study2.py e0428, tests/sk5_check.py e0429/e0441; real text)
- Measured: candidate tiles (16 rows) per position under the engine's bound model, 2048 teacher-forced positions:
  int6 row scale (435 MB) mean 163 / p99 1042; int6 with one scale per row and 32 columns (452 MB) 15.7 / 59;
  int5 g32 (380 MB) 201 / 1465; int5 g16 (398 MB) 85 / 556; int4 g32 (308 MB) 7624 (useless); int5 row 4891.
  Two tiers by tile are impossible: under int5/int4/int3 row scales every tile is a candidate at some position.
- SK5 (int5 codes, u8 group multipliers on a float32 row base; api.pack_sketch5): codes and scales equal an
  independent quantization; |exact - sketch| / bound <= 0.099 over 1024 positions x 201088 rows; ||w - w^|| / ||w||
  median 0.047. Served-like prompts (tests/skstat2.py e0447): int6 47 vs SK5 49 candidate tiles per step; SK5's
  tail is heavier (extra recompute rounds per 1024 steps ~120 vs ~206). Random-token prompts inflate candidates
  enormously (up to ~67 tiles per block per step): A/B timings on random prompts overstate tail costs.
- Load-only bandwidth of the pass-1 pattern (bench/sk5bw.cu e0448): int6 160.6 us, SK5 146 us (16-B loads: worse).
- Allows: a 12% smaller certified head at B=1 (in the engine -5 us with DPK, -4 more with double-buffered loads).
  Does not allow: B=8 gains (more candidates); anything uncertified.
