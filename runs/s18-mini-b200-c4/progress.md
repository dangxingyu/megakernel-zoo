# s18-mini-b200-c4: Inkling-2-Mini on GB200

Score: `http_step_us_b1` (lower is better). Budget 12.0 agent-hours. 13 submissions so far, 13 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 692
- 1.10x the fastest general engine with speculative decoding: 1404
- vLLM (inkling-2-mini image) + DSpark (general engine, track C): 1544
- vLLM (PR #69028 source build) + DSpark (general engine, track C): 1552
- SGLang (PR #51926 image, GB300-style settings) + DSpark (general engine, track C): 2347
- SGLang (PR #51926 image) + DSpark (general engine, track C): 2828
- vLLM (inkling-2-mini image) without speculative decoding (track A line, for scale): 3694
- SGLang (PR #51926 image) without speculative decoding (track A line, for scale): 4389
- vLLM (PR #69028 source build) without speculative decoding (track A line, for scale): 3720

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: Track C baseline: s17 step/B>=2 kernels + c1 s0022 verify kernel and fused DSpark draft for B=1

- status: **valid**, http_step_us_b1 = 1522 (B=8: 5075), at 0.20 agent-hours, category structure
- hypothesis: track C (verifier = track-A s17 s0014 step, checkpoint formats unchanged). B=1 requests are served by c1 s0022's draft-and-verify engine (its verify kernel is bit-identical to the s17 a2 step: 0 of 600 positions differ); step()/generate()/B>=2 are s17 s0014. Expect b1 ~1530 us/token, natural_b1 ~1020 (as c1 s0022), b8 ~5090 (a1 kernel, no speculation), all gates pass.
- evaluator: valid: 1522 us/step at B=1 and 5075 at B=8 over HTTP (online (submit/poll)); in process 2721 us/step at B=1 (generate, MBU 0.254)

### #2 s0002: Verify kernel: c1's verify path merged onto the a2 s0025 + s17 kernel (3-way merge)

- status: **valid**, http_step_us_b1 = 1457 (B=8: 5075), at 0.32 agent-hours, parent s0001, category structure
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats). The verify kernel now carries a2 s0016->s0025's latency work (loads before polls, union split <= 4 tokens, deeper w13 loads); bit-identical to the s17 step (xcheck 1200 positions). Verify passes -3..-6% (K=8 5415 -> 5115 us at ctx 600); in-process serve -4.5%. Expect b1 1522 -> ~1455, natural_b1 1016 -> ~975, b8 unchanged ~5075.
- evaluator: valid: 1457 us/step at B=1 and 5075 at B=8 over HTTP (online (submit/poll)); in process 2714 us/step at B=1 (generate, MBU 0.254)

### #3 s0003: Verify kernel: w2 staged expert inputs padded against 8-way smem bank conflicts

- status: **valid**, http_step_us_b1 = 1404 (B=8: 5075), at 1.13 agent-hours, parent s0002, category memory
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats). The w2a input rows in shared memory had a stride of 576 words, so the 8 B-operand columns of a 4-token expert hit one bank (8-way conflict; item time 14 us at 1 token vs 22 at 4). Padding the stride by 8 halves is layout only (xcheck: bit-identical). Verify K=8 -8%, K=4 -5% on natural text; in-process serve natural -5.9%, eval -3.2%. Expect natural_b1 954 -> ~900, b1 1457 -> ~1410, b8 unchanged.
- evaluator: valid: 1404 us/step at B=1 and 5075 at B=8 over HTTP (online (submit/poll)); in process 2714 us/step at B=1 (generate, MBU 0.254)

### #4 s0004: Verify kernel: w2 items on 3 warps (the step's path) instead of one warp per item

- status: **valid**, http_step_us_b1 = 1381 (B=8: 5075), at 1.30 agent-hours, parent s0003, category scheduling
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats). With the w2 bank conflicts gone (s0003), the step kernel's w2a organisation (8 items per round, 3 warps x 6 units, one batch) beats c1's one-warp-per-item chains at every K (tv3: -200 us K=1, -90 K=2, -30..-110 K=3..8); same sums in the same order (xcheck bit-identical). In-process serve -1.7% on both workloads. Expect b1 1403 -> ~1380, natural_b1 890 -> ~875.
- evaluator: valid: 1381 us/step at B=1 and 5075 at B=8 over HTTP (online (submit/poll)); in process 2714 us/step at B=1 (generate, MBU 0.254)

### #5 s0005: Draft layers from FP8 copies; verify kernel MXFP8 pieces 3 deep, router polls batched

- status: **valid**, http_step_us_b1 = 1374 (B=8: 5078), at 2.09 agent-hours, parent s0004, category scheduling
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats; the draft head's layer weights get FP8 per-row-scaled copies for the draft only). Two small measured items: the draft's GEMVs from FP8 copies (in-process natural -0.4%, eval -0.8%, tokens per cycle unchanged) and verify-kernel load depths retuned after s0003/s0004 (MXFP8 3 units, router logits polled together: -25..-45 us per pass at every K, bit-identical by xcheck). In-process serve natural 963 -> 953, eval 1593 -> 1571. Expect b1 1381 -> ~1360, natural_b1 877 -> ~868.
- evaluator: valid: 1374 us/step at B=1 and 5078 at B=8 over HTTP (online (submit/poll)); in process 2719 us/step at B=1 (generate, MBU 0.254)

### #6 s0006: Verify kernel: w2 groups carry two items per round (12 units in flight per warp)

- status: **valid**, http_step_us_b1 = 1353 (B=8: 5072), at 2.19 agent-hours, parent s0005, category scheduling
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats). NVFP4 w2a in verify passes: each 3-warp group takes two items per round, every warp loading the same 6-unit chain of both in one batch (166 KB in flight per SM instead of 83 KB), halving the rounds at K>=4; each item keeps piece_fp4's MMA order and block_strips' reduction (xcheck bit-identical). tv3: K2 -57, K4 -55, K8 -100..-110 us. In-process eval 1571 -> 1544. Expect b1 1374 -> ~1355, natural_b1 872 -> ~855.
- evaluator: valid: 1353 us/step at B=1 and 5072 at B=8 over HTTP (online (submit/poll)); in process 2718 us/step at B=1 (generate, MBU 0.254)

### #7 s0007: Draft: DSpark kernels in a conditional graph node, not launched on n-gram cycles

- status: **valid**, http_step_us_b1 = 1350 (B=8: 5072), at 2.27 agent-hours, parent s0006, category launch
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats; draft unchanged in function). On cycles whose drafts come from the n-gram rule, ~40 DSpark kernels used to launch and return at once (~90 us of launch chain, nsys). Under capture they now form the body of a CUDA graph IF node whose condition k_ngram clears (cudaGraphSetConditional). Same tokens and cycles in-process; natural 937 -> 930 us/token, eval unchanged. Expect natural_b1 855 -> ~849, b1 ~1352.
- evaluator: valid: 1350 us/step at B=1 and 5072 at B=8 over HTTP (online (submit/poll)); in process 2711 us/step at B=1 (generate, MBU 0.255)

### #8 s0008: Verify kernels specialized per row count (8 libraries; 7 return at entry)

- status: **valid**, http_step_us_b1 = 1318 (B=8: 5074), at 3.11 agent-hours, parent s0007, category structure
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats). The verify kernel compiled with its row count K as a compile-time constant (one library per K, built in parallel) spills 376 B instead of 600 B and is 50-200 us faster per pass (fixed-K builds: K=2 -200, K=8 -120). The draft decides K on the device, so each cycle launches all eight cooperative kernels; seven return at entry (~5 us each). Same arithmetic (xcheck: bit-identical at mixed K). In-process serve natural 930 -> 910, eval 1544 -> 1509 (-2.2%). Expect b1 1350 -> ~1320, natural_b1 849 -> ~830.
- evaluator: valid: 1318 us/step at B=1 and 5074 at B=8 over HTTP (online (submit/poll)); in process 2712 us/step at B=1 (generate, MBU 0.255)

### #9 s0009: Specialized verify kernels: per-row-count compile switches (load depths)

- status: **valid**, http_step_us_b1 = 1311 (B=8: 5075), at 3.41 agent-hours, parent s0008, category scheduling
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats). Each row count's verify library is now compiled with the switches that measured best for it (BF_UNR=2 for K>=2, W13X2 for K=1/5/6, OP_UNR_V=2 for K=3): sweeps of 14 switches per K (abk.sh), the chosen set re-measured against the plain set in two rounds: -19..-70 us per pass, bit-identical (xcheck). Expect b1 1318 -> ~1308, natural_b1 834 -> ~828.
- evaluator: valid: 1311 us/step at B=1 and 5075 at B=8 over HTTP (online (submit/poll)); in process 2715 us/step at B=1 (generate, MBU 0.254)

### #10 s0010: Cycle graph runs the verify pass through a SWITCH node (no empty launches)

- status: **valid**, http_step_us_b1 = 1312 (B=8: 5076), at 3.50 agent-hours, parent s0009, category launch
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats). The cycle's verify pass is captured into the draft's cycle graph: a one-thread kernel sets a CUDA graph SWITCH node to vpos[1]-1 and only the specialized kernel for that row count launches (previously all eight, seven returning at once). Specialized kernels take their tag range from a device counter (graph parameters are fixed). Same tokens and cycles in-process (6 prompts, xcheck); in-process natural 909 -> 906, eval 1507 -> 1500. Expect b1 1310 -> ~1304, natural_b1 831 -> ~828.
- evaluator: valid: 1312 us/step at B=1 and 5076 at B=8 over HTTP (online (submit/poll)); in process 2708 us/step at B=1 (generate, MBU 0.255)

### #11 s0011: Final source re-evaluated (s0009 engine; off-by-default experiment switches)

- status: **valid**, http_step_us_b1 = 1317 (B=8: 5076), at 4.98 agent-hours, parent s0009, category cleanup
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats). The committed source: s0009's engine (8 specialized verify kernels, per-K switches, FP8 draft layers, DSpark behind a graph IF node) with the later experiments compiled out or off by default (SWITCH-node verify, certified IMMA argmax, W13X2/X2_SHAREB, G_ALLLOAD). Expect b1 ~1311, natural_b1 ~831, b8 ~5075 (reproducing s0009 within noise).
- evaluator: valid: 1317 us/step at B=1 and 5076 at B=8 over HTTP (online (submit/poll)); in process 2711 us/step at B=1 (generate, MBU 0.255)

### #12 s0012: 16-row verify passes for 15-token n-gram copies (run c3's library + this run's w2 padding)

- status: **valid**, http_step_us_b1 = 1260 (B=8: 5081), at 5.46 agent-hours, parent s0011, category structure
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats). Adds run s18-mini-b200-c3's 16-row verify library (n-gram copies of 15 tokens in a copying regime, its draft-side rules) with this run's w2 bank-conflict fix (V(16) 9.5 -> 7.9-8.4 ms on real text), this run's verify-cost table in the draft-length rule and its DSpark IF node; 32-slot conv rings in every library. Bit-identical (xcheck: 0 of 1000 positions over mixed 1..16-row passes), served streams identical. In-process (natural prompts 1-3 as the evaluator, two rounds): natural median 836 -> 814, eval mean 1507 -> 1496. Expect natural_b1 831 -> ~810, b1 1310 -> ~1302.
- evaluator: valid: 1260 us/step at B=1 and 5081 at B=8 over HTTP (online (submit/poll)); in process 2677 us/step at B=1 (generate, MBU 0.258)

### #13 s0013: 16-row library: the step's 3-warp w2 items and 12-unit w13 batches

- status: **valid**, http_step_us_b1 = 1253 (B=8: 5094), at 5.81 agent-hours, parent s0012, category scheduling
- hypothesis: track C (verifier unchanged = s17 s0014 step; checkpoint formats). Two of this run's measured 8-row kernel changes ported into the 16-row library: w2a on the step's 3-warp items (instead of one warp per item) and NVFP4 w13 batches of 12 units: V(16) on real text 8374 / 7804 -> 8175 / 7637 us, bit-identical (xcheck, mixed 1..8 and 16-row passes). In-process natural median 814 -> 801, eval mean 1496 -> 1487. Expect natural_b1 810 -> ~797, b1 1260 -> ~1253.
- evaluator: valid: 1253 us/step at B=1 and 5094 at B=8 over HTTP (online (submit/poll)); in process 2674 us/step at B=1 (generate, MBU 0.258)

