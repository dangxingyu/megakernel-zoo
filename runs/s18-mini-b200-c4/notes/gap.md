# Gap accounting, track C at batch 1 (run s18-mini-b200-c4)

Cycle = verify pass V(K) + draft D. Engine s0002 (merged verify kernel). Card 1, ctx 600, natural text (tv3.py), us.
- V(K) K=1..8: 3336 3755 3940 4309 4851 5272 5558 5698 (random tokens: 3326 3726 3795 4017 4479 4824 5044 5115).
  a2 step (track A) at the same ctx: ~2790. V(1) - step = 550; V(2) - step = 965.
- D (nsys, eval prompt 1): span between verify passes median 407 us (DSpark cycle): setup 38, 3 layers ~170,
  base unembedding (7 rows, FP8 copy 0.46 GB) 107, Markov 7 steps 74 (early exit). n-gram cycles ~120 (c1).
- Cycle mix (cycstat.py, 8 prompts x 1024): eval 2.80 tok/cycle: K=2 32% (1.33 tok), K=3 21% (1.80), K=4 14% (2.19),
  K=5 7%, K=6 4%, K=7 2%, K=8 21% (6.51). natural 4.69: K=2 20%, K=3 9%, ..., K=8 55% (6.93).
  -> eval: half the cycles are K<=3 and cost more per token (K=2: ~3120 us/token) than a plain step (2790).

## Per-application critical path (crit_app.py, MKPROF build, ctx 600 natural; us; plain / gated)
| | step | V1 | V2 | V4 | V8 |
|---|---|---|---|---|---|
| sum | 43.7 / 65.7 | 59.8 / 79.9 | 67.8 / 88.2 | 80.2 / 97.1 | 105.2 / 129.6 |
| attn_in | 4.2 | 6.0 | 8.3 | 6.2 | 6.7 |
| attn | 8.4 | 13.4 | 12.4 | 13.7 | 15.3 |
| oproj | 3.5 | 5.0 | 5.8 | 6.1 | 7.0 |
| mlp_in | 3.6 | 3.6 | 6.4 | 5.2 | 7.7 |
| w13 | 6.4 | 7.8 | 13.0 | 18.1 | 26.1 |
| w2 | 5.6 | 9.8 | 10.0 | 14.5 | 26.3 |

## MoE at K rows (the largest K>=4 loss)
- Union (ustat.py, natural, per application): K=4 18.3 entries / 18.3 distinct experts; K=8 30-34 entries / 26-32
  distinct (random tokens: 21). Split entries (<= 4 tokens) cost ~8% extra reads at K=8.
- K=8: ~135 MB expert bytes per application -> floor ~19-22 us at 7 TB/s vs w13 + w2 ~52 us.
- w13 rounds (8 items x 3 warps x 12 units, 166 KB/SM): ~5 us each, 4 rounds at K=8. Timing-only builds:
  no weight loads -> rounds 3.5-5.5 us (decode/MMA compute ~58 SASS instr per 576-B unit); loads only -> ~5 us.
  Removing loads saves 790 us at K=8, removing compute 1050 us: load and compute serialize per round.
- w2 (one warp per item, 2 rounds at K=8): items median 15.9, max 24.5 us; loads-only 7.1 (at the per-SM floor),
  compute-only 12.1.
- Upper bound of serving rounds >= 1 of w13 from smem (DIAG_MOE=3): K2 -73, K3 -156, K4 -256, K5 -315, K8 -394 us.
- Dead ends measured here: bulk L2 prefetch of the block's expert items after routing (+200..+430 us), L1 prefetch of
  the next round (+40..+150), allocating loads (+900 at K=8), W2_BPC 3/6, W2V=0 (K1 -180, else neutral).

## Update at ~4.4 agent-h (engine s0009: 1310.5 us/token b1, natural 830.7; specialized kernels per K + kflags)
V(K) on natural text at ctx 600 (tv3, all eight specialized kernels launched): K1..8 ~2890 3424 3574 3813 4196 4482
4615 4680 us (nat2) / 2907 3451 3690 3949 4313 4595 4828 4936 (nat0). Step (a2, track A) ~2790.
-> V(1) = step + ~100, V(2) = 1.23 x step, V(4) = 1.37 x, V(8) = 1.68 x.
Per application, specialized K=8 kernel (MKPROF build), plain: attn_in 6.4, qkv 7.8, attn 17.0, oproj 6.8,
mlp_in 7.6, router 1.6, routing 3.7, w13 25.6, w2 14.8, w2b 1.6 = 92.9 (step 43.7). K=2 plain 63.2.
Largest named losses at K=8 (vs step): w13 +19 us/app (union of ~34 entries, rounds of 8 items, decode-issue bound
per round), attention +8.6 (the record item holding the pass's positions runs its k/v convolutions with per-channel
polls: 11 us vs 8.5 for other records; every fix that adds code costs more in register allocation), w2 +9,
mixes (attn_in / mlp_in / oproj) +2..4 each.
Per cycle outside the verify: DSpark draft ~410 us (layers 3 x 59, base unembedding 115, Markov ~50); n-gram cycles
~41 us; seven empty verify launches ~25-30 us (a SWITCH node removes them, no gain measured over HTTP).
