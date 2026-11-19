# Journal of run s11-flint-b200-b1 (track B on the inherited s0077 engine)

## 2026-11-16, hours 0-4: accounting, probes, head sketch

Expected / measured / concluded, in order.

1. Baseline (e0001): gen_long B=1 1467 us, B=8 3452 (in process, scored workload). MK_CLK (e0002): B=1 layer 24.8 us
   at pos 200, 27.4 at pos 900; streamed phases A 3.2, D 2.0, E 5.0, F 3.2; attention chain ~10 us.
2. Weight stream off (WSKEL probes, e0003): weights from a 512 KB L2 window: B=1 1465 -> 1199 (layers) / 1409 (head)
   / 1142 (both). Concluded: ~320 us of the B=1 step depend on weight bytes; the rest is skeleton. Track B formats can
   act on at most that.
3. W12 cost model (mkprobe.py, e0012-e0017, teacher-forced so routing stays realistic; B=8 with forced routing):
   3/4 bytes for every batch -80 us at B=1; expert batches only -35 (B=1) / -200 (B=8); a 15-op decode at the expert
   mmas +130..157 (B=1) and +230..420 (B=8, with the 3/4 bytes). Expected (playbook) a loss at the mma; measured a
   larger one. Concluded: no 12-bit code for the layers in this kernel (the expert phases are 1.5 batches per warp at
   B=1: nothing to hide a decode behind). (One probe run was void: my regex turned mma into a recursive mmaw.)
4. Census (e0006, all 39.8e9 weights): 12-bit escapes per row window 6.26e-5, per 16-row group 1.12e-4, per matrix
   1.29e-4; zero weights none. (Recorded; W12 not pursued, see 3.)
5. Head sketch study (head_study.py, e0011): int8 per row + Cauchy-Schwarz bounds over 32 column blocks: candidate row
   groups per step p50 1, mean 101 (real text, teacher forced). int7 and below: thousands. Chose int8.
6. bench_mma.cu (e0026): mma.sync bf16 m16n8k16 and s8 m16n8k32 both 20 cycles latency, 1 per 8 cycles per SMSP.
   bench_hsk.cu (e0034): exact head loop 84.9 us (7.24 TB/s), int8 pass-1 loop 45.6 us (6.74 TB/s).
7. Head sketch in the engine (e0023-e0037): tokens identical from the first version; speed was the problem (register
   allocation of the main kernels, see notes/headsketch.md). Final: same binary sketch on/off 1466 vs 1479 (B=1),
   vs track A build 1474; B=8 no change.
8. s0001 (track A, current source, inherited arithmetic): 1487.7 / 3426 (HTTP). Reference for this pod.
9. s0002 (track B, int8 head sketch on the greedy paths): expected 1488 -> ~1475 (-10..-15), B=8 unchanged. Measured
   1472.5 (-15.2) / 3438 (+11, noise); in process generate 1375 -> 1361, w1_like (host argmax path, exact head) 1524 ->
   1521 (unchanged, as it should). Confirmed. Bitcheck: step() logits bit-identical to track A.
10. WHATIF bits on the batch-1 kernel (e0041) are not usable as timing studies: WHATIF=8 (no cached keys) gave 1485
   vs 1470, 4096 gave 1611, both with other register allocations; they were written for batch > 2. The position
   dependence comes from tf_time instead: B=1 1378 us at positions 64-320, 1536 at 700-956 (0.25 us per position).
11. Allocation sweep of the track-B build by compiler flags (e0043/e0044): -Xptxas -O2, --allow-expensive-optimizations,
   --extra-device-vectorization, -lineinfo, LDH_AB=8, WAIT_TID=256 all give the same allocation (176/16/8) and
   1463-1467 us; -O1 1815. Nothing to take.
12. HALFK probe (e0042, results wrong): the attention sees half of its key groups: B=1 1471 -> 1376 (-95 us); a
   quarter: 1360 (-111). The context-dependent attention is the largest loss I can name at batch 1.
13. KVPF (e0045/e0046): the attention blocks bulk-prefetch the non-staged part of their K/V window into L2 when the
   layer begins: B=1 1480 -> 1542 (+62), the wait alone 1481. Expected -30..-50 (HBM trips become L2 hits);
   measured a loss. Not understood (the prefetch traffic is ~1 MB per layer); dropped.
14. Attention split over two SMs at batch 1 (ASPLIT, AH2; source kept as tools/mk_ah2_asplit.cu.bak):
   - prototype, owner merges after an exchange with the helper (aux 4..7): B=1 1484 -> 1520 (+36). By position: the
     slope fell 0.26 -> 0.15 us per position as intended, but +118 us at positions 64-320: the exchange is two dependent
     L2 trips (~1.6 us per layer) whether the helper has work or not (ASPLIT_T=2: +85 us with an idle helper).
   - bound without any exchange (ASPLIT_T=3, results wrong): 1412 (-72). So a split pays only if the CONSUMERS merge.
   - batch invariance needs the same two-half arithmetic at every batch size (per-half max and sum, sums per key
     group in a fixed tree, V-mix chains per half, one merge function): AH2 on one SM. Its numbers: same allocation,
     B=1 +64 us with the second half always empty, +250..+272 with the real split (B=8 +115); the attention stages
     at position 900: weights 1.3 -> 2.4 us, V mix 1.1 -> 1.6, outputs 0.4 -> 0.7, and main-block phases slower too.
     SASS: +376 instructions, LDL/STL unchanged. Not understood within the time I gave it; dropped.
   Conclusion: the context-dependent attention (~125 us per token) stays the largest named loss at batch 1; what
   would remove it is a consumer-side merge with batch-invariant arithmetic that costs < 1 us per layer.
15. AH2 bisection (e0062/e0063, B=1 gen_long, base 1484, AH2 1759): without the V-mix accumulator split 1622; with
   all weights from the first half's sum 1739; with one maximum 1787; without the second half's sum loop 1668; all
   four off 1556. SASS (e0067): the split adds 16 HMMA in branches, each wrapped by the compiler in
   BSSY/BSYNC/WARPSYNC (+15-17 each): mma.sync inside a data-dependent branch is not free. Closed.
16. Head sketch, row groups assigned by a bijection (rg = (blk + 152 j) 4999 mod 9384) to spread frequent tokens over
   blocks (e0064): B=8 pass 2 7.8 -> 6.9 us on block 0, but pass 1 49 -> 55 us at B=1 and another allocation. Off.
17. Layer-0 tables, probed before building (e0069, L0P: no phase A at layer 0, the attention reads stale q/k/v at once,
   results wrong): B=1 1466 -> 1470 (nothing). At layer 0 the attention blocks are busy with the K/V staging they defer
   past the head; phase A was not on the critical path. Not built (it would have cost ~4 h).
18. KQT (e0070-e0073): the scores of a key group as K x q^T (K tile = A operand after a register permutation, q = B
   operand): 8 mma per group instead of 16 (q as A had 8 zero rows). bench_mmaT.cu: 0 of 1.25e9 chained scores differ
   between the two forms. Logits bit-identical at B=1 and B=8 to 1100 positions; bcheck B=2..8 bitwise equal over 600
   steps. In process B=1 1481 -> 1476, B=8 3517 -> 3508. s0003 (track A): expected 1488 -> ~1483, measured 1476.7
   (-11), B=8 3426 -> 3417 (-9, as expected). Gate numbers identical to s0001.
19. Track B on top of KQT (e0076-e0083): the sketch still saves 14 us inside its binary (on 1475 / off 1489), but the
   track-B binary's batch-1 kernel draws an allocation whose attention path is 0.2-0.3 us per layer slower than track
   A's (MK_CLK: q/k/v norm 1.0 vs 0.8 us): B 1472-1475 vs A 1475. Call-site variants do not move the draw.
   Next: separate kernel instances in the track-B build (step(): track A's kernels unchanged; generate/serve: a kernel
   with the sketch and no exact head). First draw of that greedy instance: 192/28/36, 1522 us (bad); sweeping.
20. Separate greedy kernel instances for track B (sketch only, no exact head): their draws were worse (1486-1525 us
   over 8 variants). Exact head as an out-of-line function in both tracks (HEADFN): A 1546 / B 1523 (BODY_WRAP: 1495
   / 1496). Compiler-barrier draws at 5 places in attention/routing for the B build only: all 1474-1476, same
   allocation. Track B with the previous scores code (KQT=0, the same bits): 1463 vs track A 1471 on one card, i.e. the
   s0002 configuration. KQT and the sketch do not add up in one build: the allocation of the batch-1 kernel moves the
   step by about as much as either change.
21. Allocation search for track A by compiler barriers after 13 phase markers (KDRAW, e0094-e0096): every build has
   the same spill lines and 1470-1478 us (card-to-card offsets of ~5 us). Barriers do not move this allocation.
22. Correction: tools/gen_cmp.py's served-cohort loop stopped once every request had produced ANY tokens, so the
   "served tokens identical" lines of e0023/e0039/e0075 compared short prefixes only (the generate() comparisons were
   complete). Fixed (waits for finished, asserts 1024 tokens, compares served with generate() of the same prompts).
   Re-run on the s0003 source (e0104): served == generate() 8/8 in both builds; track B == track A on all groups
   (B=1 6 x 1024, B=8 3 x 8 x 1024, served 8 x 1024). The evaluator's http_tokens gate had checked s0002's full
   streams (8/8 identical).
23. s0004 (track B on the s0003 source; sketch; KQT off in the B build): expected 1476.7 -> ~1467, measured 1471.4
   (-5.3; inside one evaluation's spread), B=8 3437 (expected 3420-3430). New best by 1 us over s0002.
24. Attention split, second design (AH3 + ASPLIT3; source kept as tools/mk_ah3_asplit3.cu.bak). Arithmetic made
   independent of the block a half runs on: split point a multiple of 16 key groups (group i on warp i % 16 in either
   numbering), so the per-warp sums and the cross-warp order of the old code serve each half; V mix by (dim chunk,
   half), one branch-free chain per half; one merge function (explicit intrinsics). Verified: batch 1 split over two
   SMs (helpers aux 4..7, main blocks merge while copying) gives logits bit-identical to the one-SM two-half build.
   Speed (gen_long B=1): base 1476; one-SM two halves 1561 (per-half sums through per-group smem first cost 0.95 us
   per layer, fixed by the multiple-of-16 split); split, first version 1673 (allocation 192/44/44), with all five
   words re-requested together 1602, with sentinels placed a layer ahead and no extra barrier 1578. By position:
   the split takes the slope from 0.33 to 0.10 us per position but keeps ~2 us per layer of fixed cost (the merge
   copy, twice the CX signals, 4x the atomics of the output), more than it saves at positions 64-1088.
   Polling variants (one word with pauses of 32/128 ns, all words) changed nothing. Closed: what would make it pay is
   a merge that costs nothing on the critical path, which this structure (every main block waits for the slowest of
   eight blocks and merges) does not have.
25. Head sketch: the next row group's first batch issued before the epilogue (HSK_PIPE, e0120): 1236 B of spills in
   the sketch and 750-byte frames in the kernels (the callee's stack counts for the caller): 1630 us. Dropped.
26. Head sketch at every batch size (tools/hsk_bsizes.py, e0121): greedy tokens at B=1..8, a context to ~1900 and
   vocabulary-tail inputs identical to the exact head of the same binary.
27. s0005 (track B, sketch and 16-mma scores in the batch 1-2 instance only; batch > 2 instance = track A's code):
   expected B=1 unchanged, B=8 3437 -> ~3417; measured 1471.8 / 3428.
28. Served path in process (tools/serve_long.py, e0126): 1467 us at B=1 against 1466 for generate(): the ~7 us between
   the HTTP figure and w1_like_generate is outside the kernel (front end, client).
29. Expert prefetch (EPF; source tools/mk_epf.cu.bak). Upper bound first (mkprobe WSKEL=64: expert tiles from an
   L2-resident window, e0128): B=1 1475 -> 1366 (-109). Built: router blocks predict the layer's logits from the
   layer input (their router tiles are in registers; post-attention norm weights), tagged words; main blocks rank
   the top K while they wait for the attention and bulk-prefetch those experts' tiles to L2 (each block 1/128).
   Measured (e0129-e0131): prediction and ranking alone +8 us (same allocation); with prefetches K=8 gate/up/down
   1639, K=8 gate/up 1642, K=12 1684, K=6 1621, K=4 1581, K=2 1559 (allocation 192/48/88 in all of them). About
   +0.35 us per layer for every 4 MB prefetched: the prefetch traffic costs more than the L2 hits save, as for the
   K/V prefetch (13). Closed.
30. Upper-bound probes for the record (e0132, e0133): the shared expert's down projection removed from the down
   phase: -14 us (not worth a phase of its own); no routing computation and no wait for the router logits (fixed
   experts): -37 us with a worse allocation (routing is ~1 us per MoE layer on the critical path, the rest of the 2.3
   us the timeline shows overlaps).
31. KQT with the transposition through shared memory instead of 8 shuffles (KQT_SMEM, e0136): track B 1517, track A
   1481 (vs 1475 with shuffles). Dropped; the per-track choice (A: KQT, B: 16-mma form) stays.
32. Robustness of the final track-B engine (e0134, e0135): tools/flake.py 0 mismatches in 199 repetitions of four
   workloads (step B=1/B=8, generate B=8 300 steps, generate B=1 600 steps); flake_b 0 in 59 at B=7/5/3 and 14 long
   B=8 runs; long_ctx: generate (sketch) == step-by-step greedy (exact head) to position 2040 at B=1 and B=2,
   over-long requests refused; api_stress all 40 checks ok (e0127); submit() refuses token ids outside [0, 150144).
33. s0006 (final source: s0005's kernel, submit() input check): 1472.0 / 3456. Batch 8 inside its noise (the batch-8
   kernel is the same code as s0003-s0005's, measured 3417-3437); this draw missed the 1.10x line (3451) by 5 us.
34. Batch > 2 kernel gathering the blocks' best tokens in one trip (gather_best, GATHER_ALL, e0137): B=8 3507 -> 3541
   (allocation 144 -> 160 bytes of stack), as the previous run had found. Dropped.
35. More K groups staged in shared memory (NK2 12 -> 16, 17; the extra groups in the attention-weight area, free
   until stage 5) to take the last global K loads of a full sliding window off the scores stage (e0138/e0139): B=1
   1475 -> 1478 / 1477, B=8 unchanged; teacher-forced positions 700-956: 1542 -> 1536. Nothing; not kept.
36. W12 for the experts at batch 8, decode cost curve (mkprobe, forced routing WHATIF=1, tf B=8, e0140/e0141):
   base 3344 / 3588 (positions 64-320 / 700-956); 3/4 bytes 3182 / 3442; 3/4 bytes + 4-op decode 3439 / 3816;
   + 8-op 3580 / 3964; + 15-op 3623 / 3914; a 4-op decode at full bytes 3477 / 3806. Any integer work on the mma
   operand in the expert loops costs 130-220 us per step at batch 8 (lock-step slots: the decode adds to every
   slot). W12 is dead at both batch sizes in this kernel.
37. Batch-8 routing parts (forced routing, tf B=8, e0142): no routing computation -108/-130 us; no candidate stage
   -111/-123; no union -43/-45; no merge +37/+71 (another allocation). The candidate stage includes the wait for
   the router logits, so compute and wait are not separable this way.
38. Router sigmoid with the IEEE-rounded reciprocal (__frcp_rn, bit-identical to 1.0f/d for every d >= 1: checked
   exhaustively, tools/rcp_check.cu): no change at either batch size (e0145). The divisions are not on the path.
39. L2 prefetch from an idle warp (QPF_W = 12, cp.async.bulk.prefetch.L2): QPF = the next layer's q/k/v slice of
   this block during routing; QPF2 = the shared expert's tiles during the attention wait (MoE layers, batch 1-2
   kernel). Calibrated A/B (AB_CAL=1; with MK_NOCAL the jobs disagreed with each other): B=1 1464-1467 -> 1434-1438
   on two cards, B=8 unchanged. The earlier warp-15 issuer collided with phase-C work (+31 us). Combinations against
   QPF+QPF2 (e0162-e0164): + QPF3 (dense layers' gate/up/down) 1435.6 -> 1456.4; QPF2+QPF4 (q/k/v from warp 13 in
   the attention wait) 1435.6 -> 1456.6; + QPF4 1433.4 -> 1455.0. QPF+QPF2 kept as the candidate.
40. EPF again, prefetches issued by an idle warp (warp 14) after the main blocks' rank (e0165-e0171, against
   QPF+QPF2 ~1433): K=4/6/8 (gate/up, with or without down) all ~1703 (+270); K=2 gate/up 1599; rank only, no
   prefetch 1599 (+166); rank only with a 1-iteration poll (no wait for the prediction) 1586 (+152); no prediction
   published (every poll runs to its 4096-iteration timeout) 39500. On this source the prediction + ranking alone
   cost more than the -109 us ceiling (journal 29 measured +8 for it on the s0003 source). Closed for good.
41. s0007 (track A, QPF+QPF2 on): HTTP 1463.0 (s0003 1476.7: -13.7), B=8 3422; in process (w1_like_generate) 1467.2
   -> 1437.9 (-29, as measured). The served path in process gets the full gain too (serve_long, e0183: 1466 -> 1433),
   so the rest of the HTTP difference is outside the kernel or the evaluation's spread. Bit-identical to s0006's
   track A (bitcheck 1152 positions, B=1 and 8; digests unchanged); flake 0/199, flake_b 0.
42. Track B on the s0007 source loses its lead: the sketch still saves ~14 us inside its binary (MK_HSK_OFF A/B,
   e0181/e0182), but the HSK=1 binary's other code draws a slower allocation: B 1437-1461 over variants (QPF on/off,
   KQT_SMALL, HSK_PF; e0174-e0179, e0184) against track A 1430-1433 (QPF_W 11-14 all the same, e0185). No track-B
   submission on this source. QPF2 in the batch > 2 kernel: B=8 +8 (e0187), not taken.
43. K/V prefetch by idle warps, two more designs, both losses: main blocks' warp 13 sends the next layer's unstaged
   K/V during the routing (QPF5, e0188/e0189): B=1 +20, B=8 0; the attention block itself, idle after staging the
   next layer (QPF6, kv_pf, e0190/e0191): B=1 +80, B=8 +25. tools/bench_l2pf.cu: a bulk L2 prefetch issued on one SM
   helps reads from every other SM equally (8.3k vs 12.5k cycles per 64 KB, readers in both halves of the SM ids):
   no die locality problem.
44. Expert repetition (tools/route_rep.py, ROUTE_DUMP now records un[] at B=1): of a layer's 6 experts, on average
   0.9 (greedy) - 1.1 (teacher-forced) were also chosen for the previous token. Prefetching the previous token's 6
   experts (QPF7, warp 13 in the attention wait): +75 us with or without the down tiles (e0195/e0196). Mechanism
   consistent with 40-44: bulk traffic during the attention window delays the attention's own K/V loads (the critical
   path); data that is not used there costs ~1.5 us per layer per 24 MB.
45. Where else to put prefetch traffic (e0199-e0205): the next layer's output projection and/or shared expert at the
   end of the layer, before the wait for every block (QPF8 1/2/3): +50 each (early blocks' traffic lands in the
   late blocks' down phase); the output projection during the routing (QPF9) next to QPF, or instead of QPF, or QPF
   off: all the same (1439-1441). QPF alone is worth ~0-4 us; QPF2 carries the gain.
46. Dormant code is not free: the working source with all of 43-45 compiled out (one new Params field, unused
   functions) measured 1441 against s0007's source 1434 (tools/absrc.sh, e0206). Restored mk.cu to s0007; the
   experiments are kept in tools/mk_qpfx.cu.bak. Next: a padding field in Params (PPAD) as a layout sweep.
47. Layout sweep: a padding field in Params (PPAD 1..8, 16 words; e0207-e0209): 1431-1434 for all. Not the lever.
48. E phase (routed experts' gate/up, batch 1: three row groups per block, the third one a round trip after the first
   two). (a) The block's own third row group sent to L2 at the start of E by one warp (EPF_E, warp 15 or 7,
   e0210/e0211): +12. (b) The second half's warps, idle in the second slot, copy it into shared memory with cp.async
   together with the first slot's request; the second slot reads shared memory (ESTG, e0213/e0214): bit-identical
   (digests, bcheck 2..8) and +36. (c) Probe that skips the second slot's load (values wrong): +47, an artifact of
   wrong values (the hand-over sentinels are NaN patterns). Whatever the mechanism, more of a block's traffic in
   flight at once loses on this card in every form tried (40-48).
49. METHOD ERROR, found 2026-11-17 (agent clock ~6.3 h): /tmp is shared by the jobs of all cards in the pod, and
   tools/ab2.sh, absrc.sh, abserve.sh kept their two libraries at fixed names (/tmp/ab_A.so, /tmp/ab_B.so). A/B jobs
   that overlapped in time with different variants therefore timed whichever libraries were built last. Fixed (per-card
   names, $MK_CARD). Affected (results of only one variant per group, or mixed): e0162-e0164 (QPF3 / QPF2+QPF4 /
   +QPF4: all "1455"), e0165-e0169 (EPF variants), e0176/e0177, e0179/e0180, e0187-e0189, e0199-e0201 (QPF8 1/2/3: all
   "+50"), e0203-e0205 (QPF9), e0210/e0211, e0220/e0221 (ES_TB), e0224-e0229 (QPF_AT; one "default = 1418" reading).
   Not affected: single A/B jobs (s0007's QPF+QPF2 decision: same A/B on two cards; confirmed by the evaluator),
   sweep.sh / chk.sh / abenv.sh / vfy.sh (per-card engine/build), ESTG (e0213), QPF6 (e0190/e0191: same batch-1 code
   in both B builds), QPF7 (e0195/e0196: both QPF7), the track-B sweeps e0184/e0185/e0215/e0216. Re-measuring the
   affected groups that matter.
50. Cycle-counter timelines (tools/clkcmp.sh, MK_CLK, B=1, position 700, e0218/e0219/e0222): with QPF issued after the
   router's signal, blocks 112-127 finish the routing ~1.7 us after the median and their E ~1.1 us late; the down
   phase waits for them. Not caused by their own prefetch (QPF_SKIP=112: still late; QPF_ROT 16/64: the same blocks):
   other blocks' prefetch traffic in the routing window delays those 16 blocks' reads of the router logits. Uneven E
   shares for them (ES_TB=112/104, weights 2:3; clean re-run pending): +3..+5 in the full step.
51. QPF issued earlier, clean A/Bs (e0234-e0238): at the end of phase C (QPF_AT=2, batch 1-2 kernel) B=1 -12; after
   phase D's mma (QPF_AT=1) -2; QPF_AT=2 with the shared expert (QPF2) moved from the attention wait to the start of
   phase A (QPF2_AT=1; A reads L2 only, so HBM is idle) -16..-17 (1433-1435 -> 1416-1417), B=8 unchanged (3493-3497).
   The same move for the output projection (QPFO, 16 MB at the start of A): +91. Dense layers' MLP (QPF3) on top: +20
   (layers 0 and 1 only are dense). Defaults now QPF_AT=2, QPF2_AT=1.
52. s0008 (track A; QPF at the end of C, QPF2 at the start of A): HTTP 1452.0 (s0007 1463.0, -11), B=8 3426;
   w1_like_generate 1427.3 (s0007 1437.9, -10.6). New best. HTTP - in-process gap: s0001-s0006 7-13 us, s0007 and
   s0008 25 us. Not the poll cadence (in process, served path through submit/poll with 0 / 1 / 5 ms sleeps,
   e0247/e0248: no-prefetch build 1466 -> current 1420, the full gain at every cadence); the harness's engine loop
   is the same as tools/serve_long.py (poll, 0.2 ms sleep when empty) and the client's decode window is first chunk
   to last chunk, as in serve_long. The client's median token gap is 1.410-1.421 ms in every submission whatever the
   kernel speed (tails make the mean). One difference seen: since s0007 the B=1 counter-page calibration is bimodal
   (pages at ~1310 vs ~1319 us in s0008's evaluation; flat in s0006), so the page each process picks matters more
   than before. Unresolved.
53. After s0008 (timeline e0250, B=1 position 700): the routing stragglers are gone (route done max 18.9 vs median
   18.6); the layer ends at 27.0 (27.4-28 before). Tried on top (clean A/Bs): the next layer's output projection to L2
   in the routing window (QPF9=1) +27, at the end of C (QPF9=2) +39; uneven E shares (ES_TB=112) +11; QPF2 at the
   start of A in the batch > 2 kernel (QPF2_BIG) B=8 +20..+38. The current position's v (written by blocks 115-127,
   the slowest in phase A) read by the attention at the end of the V mix instead of with q/k (VDEF, tools/vdef.patch):
   bit-identical (digests), but +10 (load at the end) / +75 (load at the start of the V mix, held across it).
   Attention stages at position 700 (aux blocks): norms + rotary 0.9-1.2, scores 2.2, weights 1.35, V mix 1.05,
   outputs 0.4 us.
54. HTTP gap hypotheses tested and rejected: GPU memory fragmented before the engine loads (tools/frag_long.py: 25 GB
   held in mixed chunks, e0262/e0263): no change in either build (1418 / 1418; 1465 / 1465). BODY_WRAP=1 (kernel body
   as an inlined function; an allocation knob of the previous run): A 1508, B 1550 (e0256/e0257).
55. Scores stage (attention blocks), what its key groups beyond the staged ones cost (probes, results wrong;
   e0265-e0267): skipping their loads and their mma chains (SPROBE=1) 1419 -> 1374 (-45); skipping only the loads,
   keeping the mma on stale registers (SPROBE=2) -15. So the per-group dependent chain (8 mma into one accumulator,
   8 shuffles, masking) costs about as much as the trip to memory. Per-line L2 prefetch of those groups by the idle
   attention block (prefetch.global.L2, KVL2, e0264): 0.
   Bit-identical rearrangements (same chain per group, digests unchanged): the next group from memory requested before
   a group in shared memory is computed (SCO, e0268/e0269): B=1 +13, B=8 0; a warp's first two groups (registers +
   shared memory) as two interleaved chains (ILP2, e0270/e0271): B=1 +35. The batch-1 kernel's allocation decides
   again; sweeping combinations of these switches next (all bit-identical).
56. Bit-identical attention switches swept together (SCO, ILP2, VDEF 1/2, QPF_W 13/14; e0272-e0274): every combination
   1424-1455 against the base build's 1419; ILP2 +25..35, SCO +15..25, VDEF +20..30 in every company. They are slower,
   not unlucky. HTTP gap, one more hypothesis rejected: an NVML utilization poller on all cards every 50 ms (what the
   client runs during the measurement) changes nothing in either build (tools/nvml_long.py, e0275/e0276).
57. Batch 8, clean: QPF at the end of C in the batch > 2 kernel (QPF_AT_BIG) +40 (e0277); no QPF at all in it
   (QPF_BIG=0) -5, noise (e0278). Batch 1: the transposition after K x q^T skipped (TRPROBE, values wrong) +7: not a
   cost worth removing (e0279). Local memory of the batch-1 kernel (tools/spill_lines.sh: nvdisasm -g on a -lineinfo
   cubin, e0280): 46 LDL/STL; most on the attention's per-thread score array (warps with more than three key groups:
   long full layers) and two in the routing (the lane's winner read back from a local array). Reading it from shared
   memory where the rank merge left it (WRTS, bit-identical, digests unchanged): +80, because the new allocation put
   local accesses into the main blocks' phases instead (e0281-e0284). s0008's batch-1 allocation is a good draw that
   almost any edit loses. Next: sweep the existing bit-identical switches on this source (e0285-e0287).
58. Existing bit-identical switches swept on the s0008 source (B=1, e0285-e0287; base 1420-1431 across the three
   cards): KV_VEARLY=0 +12, KV_BATCH=0 +77, LDH_AB 4 / 64: 0 / -3, APRE_EARLY=1 0, GATHER_PAR=0 +90, ROUTE_RANK=0
   +75, RKT=0 +20, EXS=1 +5, VPAD=1 +55, KV_DB=1 +25, KV_STG_ASM=1 0, TLOAD_ASM=1 +6, NPA=12 0, D_EPI_EARLY +60,
   E_LEADERS_FIXED +23. The source is at a local optimum of its switches.
59. Track B, last sweep on the s0008 source (e0288-e0290): base 1433-1440; the sketch forced inline (HSK_INL) 1518;
   KQT in the B build with QPF_W 13/14, LDH_AB=64, NPA=12: 1490-1495; without KQT: LDH_AB=64 / NPA=12 / KV_STG_ASM /
   APRE_EARLY 1432-1442. Track A on the same card class 1419-1425. Track B is closed on this source.
60. The output projection's tiles requested after the "pairs begun" wait instead of before it (CLATE, e0291): +19.
   The early request stays. Calibration data of the evaluations (engine_stats.placement_us): s0006's 16 counter pages
   were flat (1360-1362 us per step at the calibration's positions); since s0007 they are bimodal (s0008: 1310 vs
   1319), the other buffers flat. A fresh process separates the two classes cleanly (+-1 us inside a class), so a
   wrong pick in the HTTP engine is unlikely to explain a systematic +17; the harness builds the engine exactly as the
   tests here do (max_batch 8, max_context 2048, no environment changes).
61. Batch-8 kernel switches on the s0008 source (B=8 gen_long, e0292-e0294; base 3489-3508 by card): FO_V 0/3 +15/+50,
   FCB_W=0 +30, K2B=1 noise, RSTG=1 +8, RSORT=1 +25, UNI2=2 +10, KQT_BIG=0 +5, HOSTC_B 32/16 noise, LDH_AB 64/4
   0/+10, HS_EARLY=1 +45, E_SETTLE_LATE=1 -10 (checking), QPF_W=13 +3.
62. THE HTTP GAP, found (e0296-e0307). A two-process copy of the server's shape written here (tools/sock_long.py:
   engine child running the online loop, JSON events over a Unix socket, the parent timestamps every token) measures
   the s0008 build at 1452.5 us/token, the evaluator's HTTP figure, and the no-prefetch build at 1478.5. The same
   loop in one process (tools/rows_long.py, gaps_long.py) gives the same 1452 / 1478; my earlier serve_long and
   gen_long gave 1420 / 1466. The difference: those created a CUDA tensor (the prompt ids) before the Engine. The
   step depends on whether anything was allocated on the device before the engine's own allocations
   (tools/gaps_long.py with a PyTorch pad of N bytes first, B=1): none 1450-1455; 8 B, 512 B, 4 KB, 64 KB:
   1417-1421; 256 KB, 512 KB, 1 MB: 1446-1452; 4 MB 1432. The token gaps show it: median 1.40 ms in both, but in the
   bad layout many gaps of ~1.7 ms. The most likely carrier: the engine's small tensors (each layer's norm weights,
   read by every block in every layer) and where they fall in PyTorch's small-block pool. The evaluation's engine
   process imports torch and builds the Engine first, so every HTTP measurement since s0007 ran in the bad layout
   (s0006 and before: the no-prefetch builds lose 12 us there, the prefetch builds 32).
   Fix: Engine.__init__ allocates 512 B first (MK_PAD, default 512; 0 = off). Two-process copy: 1449 -> 1422
   (e0306); generate() after a prompt tensor unchanged 1427 / 1427 (e0307); served batch 8 3482-3489 -> 3490-3495
   (noise level). Layout only: every result bit-identical.
63. s0009 (track A, s0008 + the 512-byte allocation first): HTTP 1422.2 (s0008 1452.0, -29.8; expected ~1425),
   B=8 3427; w1_like_generate 1421.4: the HTTP - in-process gap is 0.8 us now (25 before). New best.
64. What carries the layout effect (gaps_long, served B=1): not the parameter block (its page, 16 candidates: flat in
   either layout, e0315/e0316), and not the start pages of xc / qraw / hbuf (5-8 us each at most, e0312-e0314; their
   extra allocations put the step back into the bad layout). Copying every layer's norm / sink vectors into one
   contiguous buffer after loading (MK_NWOFF, any offset 0-64 KB) gives the good layout without the pad (1420-1422,
   e0311). Checking whether that holds under every allocation history (e0317/e0318).
65. Bisection of the layout effect (e0317-e0320, gaps_long B=1): the relocated norm vectors are good without the pad
   (1423) and bad with it (1449), bad after a prompt tensor; a 512-byte allocation inserted after loading, before
   the rotary tables, before mk_create, or at three points inside mk_create (before its buffers, before the
   calibrated copies, before the parameter blocks) never flips the state; only an allocation before the weights are
   loaded does. Reading: the per-layer norm / sink vectors (read by every block in every layer) decide it by WHERE
   they lie at a coarse scale (the offset inside the relocation buffer, 0-64 KB, did not matter). Testing coarse
   offsets (0.5-8 MB) next, for a start-up calibration of their place.
66. The carrier, by bisection (e0323-e0325): not the norm vectors (moving them after the engine is built, 9 candidate
   places, times flat in either state: 1334 / 1320). A 2 MB allocation right before the kernel library's own small
   buffers (xc, xm, qraw, attnf, rl, hbuf, counters, the calibrated copies, the parameter blocks; ~1.5 MB) turns the
   bad state good (1420) and the good one bad (1450); 4 MB or 64 KB change nothing. So the state is the parity of a
   2 MB-granular address bit of those buffers, which the existing page calibration (16 x 4 KB) never crosses.
   Fix (CARVE): the buffers are carved from one 2 MB-aligned slab the engine owns, at offset 0 and at 2 MB (two
   engines, exactly opposite parity); both are timed (24 greedy steps, best of 3) and the faster is kept.
67. CARVE tested (e0326-e0329; tools/carve.patch): with 2 / 4 / 8 carving candidates chosen by the start-up timing
   (24 greedy steps from position 0), every allocation history lands at 1420-1447 instead of 1420 / 1450, so the
   worst case is gone; but the short timing does not predict the long run (it picks layouts that run 1429-1447 over
   positions 64-1088; the bad layout shows as intermittent ~300 us gaps, not a uniform cost), and in the evaluation's
   own condition (nothing allocated before the engine, plus the 512-byte pad) the uncarved s0009 layout measures
   1419 against 1429-1431 carved. Not taken; s0009's pad stays. A calibration with a long timing per candidate
   (hundreds of steps) could make the choice robust; not done.
68. Token gaps of the served path, read correctly (e0330, tools/gaps_long.py dumps): the arrival times are quantized
   by the poll loop (~0.3 ms per iteration), so gaps cluster at ~1.1 / 1.4 / 1.7 ms and the median stays ~1.40 ms
   whatever the kernel does: this is why the client's TPOT p50 was 1.41 ms in every submission. Means by position
   (good layout / bad layout): tokens 0-256 1322 / 1350, 256-512 1396 / 1430, 512-768 1463 / 1498, 768-1023
   1495 / 1530. The bad layout is a uniform ~30-35 us per step at every position, not an intermittent stall; the step
   grows by ~0.17 us per position of context in both.
69. Track B under the layout effect (served path, e0331-e0333): pads 8 B .. 1 MB 1434-1436, no pad / 2 MB / 4 MB
   1450-1452; track A 1419-1420 on the same card (1426-1433 on card 3). The track-B deficit on this source (~14 us)
   is real, not a layout draw. Track B stays closed.
70. CARVE with the start-up timing at position 512 (e0334/e0335): the four carvings separate cleanly by 2 MB parity
   (1468 vs 1481 us per step at that position), but the best carving serves at 1428-1429 in the histories where the
   uncarved s0009 build serves at 1419 (packing every small buffer into one 2 MB window costs ~9 us of its own).
   Closed; tools/carve.patch kept. The good / bad state also depends on the card: on card 3 the uncarved build
   serves at 1426-1433 for every pad, never 1450.
71. s0009 source, last verification part (e0336): watchdog path ok (injected hang raised from poll(), generate ==
   step-by-step afterwards, reset ok); served long run 1420 / 3493 (B=1 / B=8, in process, prompt tensor first);
   against the float32 reference (tools/test_engine.py --check --long): single 0.074 median / 0.876 max (HF BF16 0.082 /
   1.113), long contexts 0.118 / 1.451 (HF BF16 0.119 / 1.287), top-1 on long contexts 99.2% (HF 99.2%).
72. XPLACE (tools/xplace.patch; e0337/e0338): xc, qraw, hbuf of the batch-1 instance in movable copies (2 MB + 16
   pages of slack), each timed at position 512 over pages {0..3, 512..515} and the fastest kept: in the good history
   1416.5 vs 1417.8 (-1); in the no-pad history 1432-1438 (not 1418); card 3 1422-1432 vs 1427. These three are not
   the parity carrier. Next: a 2 MB allocation before each buffer of mk_create's list in turn (MK_BPAD, e0339/e0340).
73. 2 MB allocation before buffer i of mk_create's list (xc 0, xm 1, qraw 2, attnf 3, rl 4, hbuf 5, argv 6, argi 7,
   lbv 8, rlp 9, curtok 10, gen 11, counter slabs 12-13, then the calibrated copies and the parameter blocks; e0339/
   e0340, card 1, s0009 build): none 1417-1419; before 0: 1450 (the full bad state); before any of 2..14: 1429-1437
   (half). So at least two groups contribute ~15 us each: one among xc / xm and one at or after the end of the list
   (the batch-1 calibrated copies of rl / xm / attnf, the parameter blocks), and XPLACE showed that moving xc alone
   does not fix the bad state: relative placement of hot buffers, not one buffer's address. No single buffer to
   calibrate; s0009's pad keeps the good state where it was measured. Closed.
74. The prefetches in the scored process's layout (tools/gab.sh: two builds alternated through tools/gaps_long.py,
   e0341/e0342): without QPF2 at the start of A 1504 against 1419 (-85 for the shared expert's prefetch, not the -30
   measured in the other layout); QPF back at the routing (QPF_AT=0) 1450 against 1425 (+25). Prefetch verdicts
   depend on the layout too: re-testing the closest losers in this layout (QPFO, QPF9=2, QPF3; e0343-e0345).
75. In the scored layout (gab.sh, e0343-e0345): QPFO (output projection at the start of A) +102, QPF9=2 (at the end of
   C) +39, QPF3 (dense MLP) +13: the earlier verdicts stand.
76. HPF (in the last layer, the end-of-C window sends the first 2 / 4 row groups of every head block's range to L2;
   e0346/e0347, scored layout): +168 / +168 us per step. Far beyond any traffic cost (once per step): some
   pathological interaction; not pursued.
77. s0010 = s0009's source re-evaluated: HTTP 1418.5 (s0009 1422.2), B=8 see result; the 512-byte allocation lands the
   evaluation's engine process in the good layout again. Best so far 1418.5 (two evaluations of the same engine:
   1418.5 / 1422.2).
78. In the scored layout: QPF_W=13 and LDH_AB=64 neutral (e0348/e0349). s0009/s0010's source is the engine/ state.
79. Scored layout, two more earlier losers (e0350/e0351): per-line L2 prefetch of the unstaged K/V by the idle
   attention block (KVL2) -1..-3, inside the noise; the block's own third E row group to L2 (EPF_E) +22. Nothing to
   submit.
80. KQT2 (tools/kqt2.patch): the 8 K x q^T mma of a key group as two interleaved chains of 4 added at the end (half the
   dependent latency; new float summation order, track A, both kernel instances alike): batch invariance kept (bcheck
   B=2..8 bitwise equal to B=1; digests change as they must), served B=1 1427-1436 against 1417-1427 (e0352/e0353).
   Not taken.
81. New pod (epoch 2 replaced): the pad still gives the good batch-1 layout on cards 1-3 (1418 / 1421-1424 /
   1428-1432 against 1451-1457 without; e0354-e0356). Batch 8, served, rows 0-7 of tools/ids.npy: 3491 / 3509 /
   3505 with the pad, 3486 / 3536 / 3531 without: batch 8 has a layout dependence of its own (~25-50 us, card
   dependent). Bisecting it (e0357/e0358).
82. Batch 8 layout: a 2 MB shift before any library buffer moves batch 8 by +-5 (cards 2, 3; e0357/e0358); pad sizes
   8 B .. 64 KB: 3491-3518 with no consistent order (e0359/e0360). Nothing to take at batch 8.
83. NACC (accumulators per mma chain, both instances; a different float summation order, track A) in the scored
   layout (tools/gabn.sh, e0361-e0363): NACC=4 B=1 +77, B=8 -50 (3490-3513 -> 3443-3449); NACC=1 B=1 +22. The batch-8
   gain is in its expert phases (most of its time); testing 4 accumulators there only (NACC_E) and in the exact head
   (NACC_H), keeping both instances alike (bitwise batch invariance).
84. NACC by phase (scored layout, both instances alike): expert phases (E and F) at 4: B=1 +15, B=8 -25 (e0364/e0365);
   gate/up only (E): B=1 +1.5, B=8 -13; down only (F): B=1 +7, B=8 -15 (e0367/e0368); exact head at 4: B=1 0
   (e0366). Mapping C, D and the head at B=8 next (e0369-e0371).
85. NACC map completed (e0369-e0371): head at 4 B=8 0; output projection (C) at 4 B=1 0, B=8 noise (3485-3502 vs
   3493); shared expert (D) at 4 B=1 +60; gate/up only (E) at 4, three more rounds B=1 +2..3. The only batch-8 gain
   (-13 with E at 4) costs the headline a few us: not taken (tools/nacc_phase.patch). Using different NACC in the
   two instances would end bitwise batch invariance (the gate only needs tolerance, but the engine keeps bitwise).
86. Late result of e0371: shared expert (D) at 4 accumulators, batch 8: 3502-3518 -> 3442-3462 (-50..-70, most of the
   global NACC=4 gain at batch 8), against +60 at batch 1. Same trade as 85: not taken while batch 1 is the headline;
   for a run that weighs batch 8 (or the 1.10x claim there), NACC_D=4 in both instances is the lever to look at,
   ideally with a batch-1 allocation that does not pay for it.
87. NACC_D=4 (shared expert phase D, both instances) with the batch-1 instance computing the same four chains two at a
   time (DSEQ=1: the same sums, bcheck B=2..8 bitwise equal to B=1; e0372-e0374): B=8 3503-3511 -> 3443-3449 (-60),
   B=1 1419 -> 1500-1513 (+83; the batch-1 frame grows to 192 B with 52/84 B of spills). Trying other orderings of the
   same arithmetic for the batch-1 instance (e0375-e0377).
88. The other orderings for the batch-1 instance (e0375-e0377): one chain at a time with running pair sums +83, four
   accumulators issued 0 4 1 5 2 6 3 7 +64, the plain interleaved form +58. Every form of D at 4 accumulators costs the
   batch-1 step 60-80 us, so it is not the register allocation of one form. Closed (tools/naccd.patch); batch 8's
   -60 stays unclaimed.
89. Control (e0378/e0379): phase D's two chains of four computed one after the other instead of interleaved (DCTL;
   the same bits, digests unchanged): 1418-1422 against 1418-1420, no change. So the +60..+83 of NACC_D=4 at batch 1
   comes from the changed VALUES (another greedy continuation of the benchmark prompt), not from the code: the batch-1
   step depends on the tokens it generates, and every numerics-changing experiment timed on prompt row 0 alone
   (KQT2, NACC variants) may have measured a different continuation. Timing both builds on all 8 prompt rows
   (tools/rowsab.sh, e0380/e0381).
90. Correction to 89: across all 8 prompt rows NACC_D=4 is +64 on every row (e0380/e0381: 1418-1428 -> 1483-1494), so
   not the token stream; and phase D's sums scaled by 1 + 2^-22 (DPERT, other values, same code) change nothing
   (e0382/e0383). So the values do not matter: every 4-chain form of D draws a slow batch-1 code generation, the
   2-chain reordering (DCTL) happened to keep the fast one. The numerics-changing experiments timed on row 0 stand.
91. NACC_D=4 draw sweep (e0384-e0386): 4 batch-1 orderings x QPF_W 11/13/14 = 12 builds: every one puts the batch-1
   kernel at a 192-byte frame and 1482-1506 us (base 1418-1430 per card, 176-byte frame). Whatever form the batch-1
   instance computes D in, 4 accumulators there cost it 60-85 us; batch 8's -60 is not available without it. Closed.
92. ptxas --register-usage-level (e0388-e0390, B=1, gen_long): 0 / 2 / 3: 1624-1640 (192-byte frame); 4: = default;
   6 / 7 / 8 / 10: +4..+5; allow-expensive-optimizations: 0. Nothing on the base source. Trying it on the NACC_D=4
   source, whose batch-1 cost was its frame (91).
93. NACC_D=4 with ptxas register-usage levels (e0391-e0393): levels 6 / 8 / 10 keep the batch-1 frame at 176 B and
   bring the batch-1 cost down from +65 to +8..+12 (1427-1432 against 1418-1421 per card; the level alone costs ~+5
   on the base, 92); level 4 keeps the 192-byte frame (+67); other orderings at 7 / 9: +46. Still a loss on the
   headline for batch 8's -60. Closed. (For a run where batch 8 weighs: NACC_D=4 + -Xptxas -regUsageLevel=8.)
94. ptxas register-usage level at batch 8 (e0394-e0396, base 3489-3506 per card): levels 6 / 8 -3..-4, 7 / 10
   -7..-9, 9 0, expensive optimizations 0. Too small to justify compiling the two instances separately. Closed.
95. Track B with ptxas register-usage levels (gen_long, e0397-e0399): with KQT in the batch-1 instance 1468-1494 at
   every level (default 1494); without KQT (the B default) 1434 -> 1427-1428 at levels 6 / 8 / 10 (card 2). Same-card
   comparison against track A in the scored layout running (tools/abtrack.sh, e0400/e0401).
96. Same card, scored layout (e0400/e0401): track A default 1417-1419 vs track B at register-usage level 8 1429
   (+11); card 2: A 1421-1429 vs B at level 6 1428-1436 (+6). Track B stays behind track A on this source. Closed.
97. Earlier losers under register-usage levels (e0402-e0404): KQT2 at levels 8 / 6: 1422-1427 / 1421-1423 against
   1418-1421 (+2..+6); global NACC=4 at levels 6 / 8 / 10: B=1 1437-1438 (+17), B=8 3445-3452 against 3500-3509
   (-50). Nothing improves the batch-1 headline. Closed.
98. SDPRE (tools/sdpre.patch): the shared expert's down projection (F's last two items) computed by warps 8-11 in E's
   second slot at batch 1 (idle there: three row groups per block), from tiles requested after their first slot and
   the shared hidden units copied into free shared memory; F processes only the routed experts and adds the stored
   sums in the same order. Bit-identical (digests, bcheck B=2..8; e0406), but B=1 1418 -> 1642 (+225), B=2 +165, and
   +263 at register-usage level 8 (e0405, e0407, e0408): warps 8-11 wait in E's slot for the slowest block's shared
   hidden units (written during the routing, at batch 1 by warp 15) and hold E's barrier. Closed.
99. Batch-8 trade-off variant (NACC_D=4 interleaved in both instances, ptxas register-usage level 8), scored layout
   (e0410-e0412): B=1 1418-1422 -> 1428-1441 (+11), B=8 3508-3515 -> 3451-3459 (-57); bcheck B=2..8 bitwise equal to
   B=1 (new digests: another summation order in phase D), batch-1 frame 176 B. Running the full checks (vfy 0/1/3,
   e0413-e0415) before submitting it as a labelled trade-off (not a headline candidate): it moves the batch-8 margin
   under the 1.10x line (3451) from 0.7% to ~2%.
100. s0011 (the batch-8 trade-off of 99): valid, HTTP B=1 1430.2 (expected ~1430), B=8 3443.4 (expected ~3370; s0010
   3427.2). The evaluator's batch-8 workloads did not get the gain my benchmark showed: generate 3417 -> 3400,
   w1_like 3542 -> 3539, step 3424 -> 3416, HTTP +16, against -57 on tools/ids.npy rows 0-7 (gaps_long GL_B=8). Those
   prompts route differently from the evaluator's (natural-text prompts, a different stream per sequence), so batch-8
   decisions need a workload closer to the client's before they can be trusted. Not kept: engine/ is s0010 again.
101. A batch-8 benchmark that matches the client (tools/client_like.py: prompts built like the client's steady
   workload, our own seeds; gaps_long GL_CLIENT=seed): the NACC_D=4 variant is +17 at B=8 there (3460-3468 ->
   3478-3483, e0417), as s0011 was over HTTP (+16), and +9 at B=1 (e0418). ids.npy rows had shown -57. Re-checking the
   batch-8-only switches with it.
102. Batch-8-only switches on the client-like benchmark (GL_B=8 GL_CLIENT=100, e0419-e0421): E_SETTLE_LATE +10, RSTG
   +20, KQT_BIG=0 +3..+18, HOSTC_B=16 +4, HS_EARLY +40, K2B (attention pairs on main blocks: two batches of K tiles in
   flight) -11 in both rounds (3450-3458 vs 3461-3472). Checking K2B at B=1, its bits, and on other prompts.
103. K2B, checked (e0422-e0424): B=1 unchanged (1417-1426 vs 1418-1424, client-like prompts); B=8 on other prompts
   (seed 300) -5..-20 (about -11); bit-identical (digests, bcheck B=2..8). Made the default (K2B 1); full checks running.
104. s0012 (s0010 + K2B): HTTP B=1 1419.0 (s0010 1418.5), B=8 3425.5 (s0010 3427.2); evaluator in process B=8:
   generate 3421 (3417), w1_like 3547 (3542). The -11 of the client-like benchmark did not appear: neutral within the
   evaluation's spread. Bit-identical and harmless, kept (engine/ = s0012). The client-like benchmark predicted the
   direction of s0011 but not this small effect: batch-8 effects under ~15 us are below what one evaluation resolves.
105. Batch-8 timeline on client-like text (tools/mk_client_ids.py -> teacher-forced bt_clk with MK_IDS, e0428):
   blocks 112-121 finish the routing 2.7 us late (QPF's traffic in the routing window, as at batch 1 before s0008),
   and the hosts of the main-block pairs (24-31) finish C ~2 us late; layer ~65 us (E 22, F 14). Without QPF in the
   batch-8 kernel (QPF_BIG=0, e0429-e0431) the routing stragglers go, but the layer ends at the same time (65.5) and
   the client-like A/B is +2 (seed 100) / -8..-14 (seed 300): ~-5, below an evaluation's resolution. Not taken.
106. Batch-8 C stragglers (hosts 24-31, ~2 us late in C, so D starts late for every block): not the hosts' extra
   output projection but their owners, the 8 main blocks running attention pairs, whose attention ends at ~18 us
   against ~10 us for the aux blocks' pairs (e0428). That is the batch-8 long pole: 32 pairs on 24 aux + 8 main
   blocks, already tuned by the previous run (HOSTC_B, AFIRST, HSTG, K2B...). Closing the batch-8 line here.
107. Why ESTG (48b) lost, on the cycle counter (e0432, B=1, position 700): E itself gets slower, route done -> E done
   5.0 -> 5.7 us (median block), and the layer ends 0.7-1 us later. Requesting all of a block's E bytes at once (the
   third row group by cp.async into shared memory with the first two) is slower than the two staggered rounds; the
   second round is not an idle round trip. The family "fetch E's second round early" (EPF_E, ESTG) is closed.
108. Random combinations of bit-identical switches (24 builds: QPF_W 11-14 x LDH_AB 8-64 x NPA 12/16 x KV_STG_ASM x
   APRE_EARLY; gen_long B=1, e0433-e0435; bases 1417-1422 / 1421 / 1429 on cards 1 / 2 / 3): every combination at or
   above its card's base (best 1418, the default's neighbourhood). No luckier batch-1 allocation in this space.
109. Coarser head sketch for track B? (tools/hsk4_study.py, real head inputs, ids.npy + client-like text, e0436): with
   the certified per-block Cauchy-Schwarz bound, candidate row groups per sequence-step (of 9384): int8 (today) mean
   258 / p50 105 / p99 2439; int6 7635 / p50 9376; int4 per row 9384 (all); int4 with per-block scales 9377. The
   bound is too loose below 8 bits: no byte saving is left in a coarser sketch. Track B stays where it is.
110. Persisting L2 for the head (tools/persist.patch: launches on a stream with an access-policy window over the first
   N MB of lm_head, hitRatio 1; e0437/e0438): B=1 16 MB +3, 32 MB +19, 64 MB +50; B=8 32 MB +2. Pinned L2 takes the
   room the q/k/v and shared-expert prefetches need. Closed.
111. EVF: the expert phases' tile loads (gate/up and down, both instances) with an L2 evict-first cache policy
   (createpolicy + ld.global.L2::cache_hint, L1::no_allocate). Bit-identical (digests unchanged, e0441). First A/B
   (e0439/e0440): B=1 1418-1423 -> 1420-1425 (~+1), B=8 client-like 3465-3475 -> 3404-3426 (-50..-60). Confirming
   on other prompt sets and trying the same hint on the exact head's tiles (EVFH) (e0442-e0444).
112. EVF confirmed (e0442-e0444): B=1 0..+1 (three rounds, noise); B=8 seed 300 3481-3491 -> 3419-3421 (-65), seed 700
   3514-3528 -> 3460-3464 (-60). The exact head's tiles with the same hint (EVFH) on top: B=1 -1..-2, B=8 -5 (noise).
   Both made the default (one mechanism: the evict-first hint on weights streamed once per step); checks running.
113. s0013 (s0012 + evict-first L2 hint on the expert and head tiles): HTTP B=1 1417.4 (new best; s0010 1418.5),
   B=8 3356 (s0012 3425.5: -69; expected ~3360). The batch-8 margin under the 1.10x line (3451) goes from 0.7% to 2.8%.
   The client-like benchmark predicted it; the random-token rows would have shown it too, but they also showed
   gains (s0011) that did not exist.
114. The hint on other streams (scored layout, B=1 and client-like B=8, e0448/e0449): the output projection's tiles
   evict-first (EVFO) B=1 +56, B=8 0; the router's tiles evict-first (EVFR) B=1 +1..+3, B=8 -1 (noise). K/V loads
   evict-last (KVEL) running (e0452/e0453).
115. K/V loads with an L2 evict-last policy (KVEL, bit-identical; e0452/e0453): B=1 1429 -> 1565 (+135), B=8 3409 ->
   3600 (+190). Pinning the K/V cache crowds the weights' prefetches out of L2. Of the cache-policy family only the
   evict-first hint on the streamed experts and head (s0013) pays (tools/hints.patch keeps EVFO, EVFR, KVEL).
116. Earlier prefetch losers on the s0013 engine (the evict-first hint keeps L2 free of expert tiles; e0454-e0456):
   output projection at the start of A (QPFO) +82, at the end of C (QPF9=2) +27, the shared expert at the start of A
   in the batch-8 kernel (QPF2_BIG) +10..+27. Still losses: their cost was not L2 room. Closed.
117. More hints (tools/hints2.patch; e0458-e0460): K/V loads evict-first (KVEF) B=1 +70, B=8 +160 (both K/V hints lose:
   the default policy is right for the cache); the dense layers' MLP tiles evict-first (DEVF) B=1 0..+3, B=8 +5 / -13
   (noise). Closed.
118. Forms of the evict-first hint (e0461/e0462): without L1::no_allocate (EVFV=1) B=8 3402-3409 -> 3377-3388 (-22),
   B=1 1422-1425 -> 1415-1418 (-6); evict-first on half the lines (fraction 0.5, EVFV=2) B=8 +90, B=1 +3.
   Confirming EVFV=1 (e0463-e0465).
119. EVFV=1 confirmed (e0463-e0465): B=8 seed 300 3416-3418 -> 3390-3393 (-24), seed 700 3464-3474 -> 3454-3466
   (-5..-11); B=1 (card 3) 1429-1438 -> 1420-1425 (-6..-14). Made the default; checks running.
120. The evict-first hint on the last two weight streams (tools/hints3.patch; scored layout; e0469-e0471): phase A's
   preloaded q/k/v tiles (EVFQ) B=1 1415-1417 -> 1415-1417 (+0..+1, one 1428 outlier); the shared expert's gate/up
   preload and down tiles (EVFS) B=1 1413-1416 -> 1528-1553 (+120: the shared expert's tiles are prefetched into L2
   at the start of A and must stay there until D/F); both together B=8 (client-like, seed 300) 3407-3416 -> 3408-3417
   (0). Closed: of the weight streams only the routed experts and the head take the hint.
121. s0014 (s0013 + EVFV=1: the evict-first loads without L1::no_allocate): HTTP B=1 1411.8 (new best; s0013 1417.4,
   expected ~1411), B=8 3373 (s0013 3356; expected ~3340: +17, inside one evaluation's batch-8 spread, journal 104).
122. The last layer's idle q/k/v-prefetch window (end of C; there is no next layer) for the head: the first half row
   groups of each block's head slice (HPF=4/8/16, own code at the QPF site; e0472-e0474) B=1 +12/+21/+19, size-blind:
   the batch-1 frame goes 176 -> 192 B (tools/res.sh; also with cheaper address math, and folded into the QPF
   statement, HPF2). HPF3: no new kernel code, a layer entry past the last whose q/k/v pointer is the head, so the
   existing statement fetches the head's first 20 MB (the slices of blocks 0-4) in the last layer: frame 176, B=1
   (e0481) 1410-1412.5 -> 1405-1411 (~-3; expected -2.6 = 20 MB of 615 MB of the ~79 us head). Confirming (e0482-e0484).
123. Layer 0's q/k/v tiles past the register preload (NPA..19 per warp) prefetched into L2 during the gather wait (QPF0;
   e0475/e0476): B=1 1415.6-1421.2 -> 1415.0-1426.2 (noise). Closed.
124. HPF3 confirmed (e0482-e0484): B=1 card 1 1415-1428 -> 1411-1422 (medians 1416 -> 1412), card 2 1413-1414.5 ->
   1407.7-1408.8 (-5, every round); B=8 (client-like, seed 300) 3409-3422 -> 3408-3413 (0..-3; the batch-8 kernel does not
   issue it). Made the default; checks running (e0485-e0489).
125. s0015 (s0014 + HPF3): HTTP B=1 1407.8 (new best; s0014 1411.8, expected ~1407), B=8 3368 (s0014 3373).
126. HPF3 in the batch-8 kernel too (HPF3B: its q/k/v prefetch statement, before the routing, also runs for the layer
   entry past the last; frame unchanged; e0491/e0492): B=8 client-like seed 300 3417-3429 -> 3415-3423 (-2), seed 700
   3459-3467 -> 3457-3458 (-3..-9). Small and consistent; kept for the next submission after a batch-1 check.
127. HPF3B at batch 1 (e0496): 1410-1425 -> 1410-1417 (no harm). Made the default (rides with the next submission).
   The dense layers' MLP prefetch in part (QPF3 with QPF3N = 4 / 8 of the 18 chunks of 32 KB per block; e0495):
   B=1 0 / +1..+2. Closed with QPF3 (journal 75).
128. Track B re-measured on the s0015 source (tools/gabt.sh: a track per side, scored layout; e0498-e0504): B as built
   (HPF3 prefetches head bytes B does not read) +4..+6 vs A; B with HPF3=0 -1; HPF3 aimed at the sketch's first
   20 MB instead (HPF3Q) +5..+10; the sketch's tiles evict-first (HSKE: batch-1 frame 176 -> 400 B, also with the
   policy hoisted) +12..+15. The track-B build used the 16-mma scores form for its allocation (KQT_SMALL=0, ~+11 for
   A); with KQT_SMALL=1 its batch-1 frame is now 176/16/8 too: measuring (e0508-e0510).
129. Track B with KQT_SMALL=1 (e0508-e0510) vs track A (s0015): with HPF3=0 card 1 1410-1419 -> 1405-1407 (-6), card 2
   1407-1409 -> 1402-1403 (-6); with HPF3 (card 3) 1410-1414 -> 1397-1406 (-4..-12). B against B with and without
   HPF3 next (e0511-e0513).
130. B against B, HPF3 on vs off (e0511-e0513): 0 / -0.5 / -1.5 (HPF3 stays on in both tracks: the batch-8 kernel
   reads the exact head in either). Track B candidate = s0015 + HPF3B + KQT_SMALL=1 in both tracks + engine/TRACK=B:
   ~-6 vs A at B=1. Checks (bitcheck A vs B, generate/served/batch-size tokens A vs B, bcheck, flake, vfy) e0514-e0520.
   (HPF3B now also requires HPF3: without the extra layer entry it would read past the layer array.)
131. s0016 (track B: s0015's source as TRACK=B with KQT_SMALL=1; HPF3B left off as a track-A change not yet
   submitted): HTTP B=1 1400.4 (new best; track A s0015 1407.8; expected ~1402), B=8 3362 (s0015 3368).
   Track A stays buildable: engine/TRACK=A (or MK_TRACK=A) builds s0015's arithmetic with the exact head.
132. Head sketch tuning on s0016 (track B, B against B): HSK_PF (first bytes of each warp's first row group to L2)
   32768 instead of 16384: 0 / -0.5 (e0522, e0524); 0: +1 (e0523): noise, stays 16384. Each warp sending its next
   row group to L2 as it starts one (HSK_PFN; frame unchanged): pass 1 49.7 -> 73.9 us per step (HSK_STATS, block 0,
   e0528), like every in-stream prefetch before (journal 107). Stage times now: prep 5.0, pass 1 49.7 (317 MB:
   6.4 TB/s), publish 3.1, exchange 1.3, pass 2 1.5 = 62.9 us vs the exact head's 77 us.
133. The sketch's pass 1 runs at 6.4 TB/s against the exact head's 8.0 (615 MB in 77 us, evict-first tiles). Its
   tiles as ld.global.cs (evict-first without a policy register; HSKE=3, frame 176 kept): pass 1 49.7 -> 51.4 us
   (HSK_STATS, e0531): the policy is not what the exact head has over it. Without HSK_PF: prep 5.0 -> 3.7, pass 1
   +2.3. Study code kept in tools/hsk_tune.patch (HSK_PFN, HSKE 1-3, HPF3Q); the sketch is left as in s0016.
134. Allocation screen of the track-B build (tools/sweep.sh B, gen_long B=1, e0532-e0534: QPF_W 11/13/14, LDH_AB
   8/32/64, NPA 12/14, KV_STG_ASM, APRE_EARLY, HSK_PF 8192/24576, three pairs): every build within +-2 of its card's
   base (frame 176 throughout). No luckier allocation.
135. Every block gathers the chosen tokens itself (ALLG, batch-1 kernel: no second hand-over after the head, each
   block reads its own copy of the token; block 0 still writes curtok/gen and the progress word; the host's counter
   targets one bump less per step). Tokens identical (gen_cmp, hsk_bsizes: e0539; the first try hit the watchdog
   until the host's counter arithmetic knew). B=1 +12 / +10 (e0540/e0541). The layer-0 preload that the second
   hand-over hid is now exposed; ALLG=2 adds a bulk prefetch of layer 0's q/k/v before the heads-done wait
   (e0543-e0545).
136. ALLG=2 (e0543-e0545): +4..+6 vs base (ALLG=1 -> 2: -6, the exposed preload was half the loss). ALLG=3: the aux
   blocks need no token and skip the wait and the gather, going straight to their staging (e0547-e0549): tokens
   identical, B=1 +3..+5. The second hand-over after the head is not on the step's critical path: closed
   (tools/allg.patch).
137. The sketch's prep (5 us per step for a few hundred instructions; HSK_STATS marks, e0551/e0553): stage 1 takes
   2.7-3.2 us on the first pass and 1.3 us when repeated at once (HSK_REP=3): ~1.9 us of the first pass is fetching
   the code (it runs once per step), and the digits stage is about as slow. No clean lever (no way to prefetch
   instructions); a smaller prep would gain ~1-2 us at most, since pass 1 is bandwidth-bound behind it.
138. Layer 2 (the first MoE layer of a step) in the generate path (new tools/bt_gen.py: MK_CLK stamps of the last
   step of one generate() launch; e0560): F done 31.96 us vs 25.21 in layer 20: D + router 3.85 vs 1.79 and the
   routing 4.4 vs 2.25 on every block, and blocks 112-127 finish E ~3 us late (also in step() mode: tools/bt_clk.py
   with MK_CLK_BLK, e0555-e0558; both tracks, positions 300-900). Evict-first dense MLP tiles (DEVF as ld.global.cs)
   and sketch tiles (HSKE=3) change nothing (e0559/e0561): the MoE code is not evicted from L2; layer 2 is where the
   per-SM instruction cache, filled by the head, gather and dense-layer code, misses on the MoE code again. ~6.5 us
   per token with no clean lever found (CUDA has no instruction prefetch).
139. Layer 2's late blocks per block (tools/bt_gen.py MK_CLK_EDUR=1, generate path, e0564): the expert phase takes
   exactly 8.2 us (pos 700) / 7.7 (pos 400) on every one of blocks 112-127 and 4.8-5.4 on the others; the routing
   takes the same time within bands of blocks (0-71, 72-111, 112-127). Uniform per 16-block group, independent of
   the token: a property of where those blocks run (one GPC, possibly shared with aux blocks running other code), not
   of the experts. Costs ~3 us per token through the E -> F barrier; layer 20 shows nothing like it. Left as a lead:
   a per-GPC placement of roles (which block ids are aux) might remove it.
140. Why the sketch's pass 1 streams at 6.4 TB/s and the exact head at 8.0: the SASS of pass 1 has generic LD.E.128
   (not LDG; the pointer comes in through the out-of-line function) interleaved with the IMMAs, ~5-6 tiles in flight
   per warp instead of 16. Forcing 16 up front: volatile ld.global asm (HSKE=4) or __ldg (HSKE=5), frames unchanged,
   pass 1 52.1 us (HSK_STATS) and B=1 0 / +1..+2 in the scored layout (e0571-e0574); the cache-hint asm forms make
   head_sketch spill (216 B; HSKE=1). Contiguous ranges per block like the exact head (HSK_CONTIG=1, ng fixed;
   tokens identical): pass 1 50.4, pass 2 12 us on block 0 (the likely tokens are low ids, in block 0's range: the
   interleaving spreads them). Loads in flight are not what holds pass 1 back; left as it is.
141. The dense layers (0 and 1) take ~51 us each against ~26.7 for a warm MoE layer (tools/bt_gen.py: layer 2 starts
   102.7 us after the step's first A start, layer 3 at 137.9, layer 4 at 164.6; e0577): their MLP (75 MB) streams at
   ~2 TB/s. Its loops load tile by tile (48 gate/up, 24 down per warp; no preload before the barriers, unlike every
   MoE phase). DMLP (same mma order, bit-identical by construction): 1 = gate/up 16 tiles preloaded before the
   barrier + batches of 16; 2 = down 16 tiles preloaded before its barrier; scored layout B=1 (e0580-e0582): DMLP=1
   -9..-10, DMLP=2 -9, both (3) +17 (the batch-8 kernel spills; batch-1 frame 176/24/24). DMLP=4 = the preload only
   (the other 32 gate/up tiles as before); 6 = 4 + 2: frames 176/16/8. Measuring (e0584-e0586).
142. DMLP in the track-B build (e0584-e0586): DMLP=6 -17..-20 vs base (card 1), -11 vs DMLP=2 (card 2); DMLP=4 alone
   -15 (card 3). Timeline with DMLP=6 (e0589): layer 2 starts 84.3 us after the step's first A start (102.7 before):
   the dense layers -18 us. In the track-A build DMLP=6 (e0587/e0588): B=1 +8 (frame unchanged: the A build's
   allocation draw), B=8 -14. Track A variants DMLP=4/2/1 measuring (e0590-e0592).
143. Track-A variants of DMLP (e0590-e0592): 4 +14, 2 +11, 1 -2..-3 (the A build's allocation; the dense layers
   themselves get faster in every form). DMLP=6 made the default for both tracks; checks for both (track A
   e0593-e0597; track B e0598-e0604: bitcheck A vs B 1.0 at B=1/8, tokens identical, digests unchanged, flake 0).
144. s0017 (track A, DMLP=6; submitted first as the kernel change behind s0018): HTTP B=1 1417.6 (expected ~1415: a
   loss in this build, as measured), B=8 3379 (expected ~3355). s0018 (track B, same source) submitted.
145. s0018 (track B, s0017's source): HTTP B=1 1379.6 (new best; s0016 1400.4; expected ~1383), B=8 3367.
   DMLP bit 8 (one bulk L2 prefetch per warp of its tiles past the preloaded 16, issued with the preload; DMLP=14,
   frame 176/20/16; e0605/e0606): +32. Another in-stream prefetch that slows the stream. Closed (code kept, off).
146. More on the dense layers (track B build, tools/bt_gen.py with stamps added to the dense path: BT 4 = MLP start,
   5 = gate/up done (after the block barrier), 8 = warp 0's six writes done, 6 = bump, 7 = down barrier passed,
   9 = end; e0614-e0622). Layer 1 with DMLP=6: gate/up 7.8 us (6.4 TB/s), then warp 0's epilogue 8 us (six row
   groups folded and written one after the other: 4.3 us for the writes alone), down 4.5 us. Variants: one settle for
   the six (64): no change; the down preload after the bump (128): epilogue 6.5, layer end 34.05 -> 32.55; + plain
   stores and one fence (256): 31.48; the six row groups on six warps in parallel (512, frame 176 -> 192): 30.31.
   Per-lane L2 prefetch of the later gate/up tiles (16; DMLP=22) +2..+3 and the batched form with the down preload
   (7) +34: closed. Gate/up tiles 16..31 to L2 at the start of A (DMLPA) -1..-2 (e0612/e0613). A/Bs e0623-e0625.
147. End-to-end (track B build, scored layout, e0623-e0625): DMLP=134 (down preload after the bump) -2; 390 (+ plain
   stores, one fence) -3..-5; 646 (six warps in parallel, frame 192/40/48) +70; 902 (the same with plain stores) has
   the same frame: dropped. DMLP=390 made the default; checks for both tracks (the plain stores' visibility: flake)
   e0627-e0638; track A 6 vs 390 e0639.
148. s0019 (track A, DMLP=390): HTTP B=1 1401.8 (expected ~1407; the best track-A figure of the run: s0015 1407.8),
   B=8 3367. s0020 (track B, same source) submitted after it.
149. The dense epilogue's write loop, stamped per row group (BT 10-12, MK_CLK_X=9 in tools/bt_gen.py; e0640/e0642):
   ~1 us per row group, the first and the later ones alike (so not instruction fetch): h0 folded +0.7, h0 stored
   +0.5, h2 stored +2.1, loop end +3.2. Reading the buffer pointer once and plain non-volatile stores (DMLP 1024:
   P->hbuf was re-read after every volatile store) changes nothing. Cause not found; left at s0019's form.
150. s0020 (track B, s0019's source): HTTP B=1 1374.4 (new best; s0018 1379.6; expected ~1376), B=8 3350.
151. Where the step goes now (tools/bt_gen.py "starts", MK_CLK build, which runs ~100 us slower than the real one;
   e0646): layers 3-48 ~28 us each and even (no other outlier like the dense layers); layer 2 starts at 82 us and
   takes ~33; after layer 48 the head and the step boundary take ~73 us (the sketch ~63). The next layer's q/k/v
   prefetched in the dense layers' epilogue window (DMLP 2048; HBM idle there now): layer 2 starts 82.9 vs 82.0: no
   gain, dropped. The dense down phase's last 8 tiles to L2 at its preload (4 KB per warp, DMLP 4096): A/B e0648/e0649.
152. DMLP 4096 (e0648/e0649): +32 / +33 (frame 176/20/16). Study code of journal 149-152 (DMLP 1024, 2048, 4096, the
   per-row-group stamps) kept in tools/dense_study.patch; engine/ stays s0020's source.
153. The dense epilogue's ~1 us per row group, chased further (study builds from tools/dense_study.patch; e0640-e0660):
   with no stores at all (DMLP 8192, timing only) still ~1 us per row group; not unrolled (#pragma unroll 1, DUNR)
   the same, so not instruction fetch; the SASS of the marked loop (pmevent -> PMTRIG) is shared loads (LDS.128) and
   arithmetic only. Unexplained (the stamps may mislead here); the stored values' pointer is re-read from the
   parameter block before every store in the default form (LDC + LDG.E.64 +0x1e38) but hoisting it changed nothing.
154. Allocation screen on s0020's source (tools/sweep.sh B, e0651-e0653): every bit-identical switch within +-2 of its
   card's base, KQT_SMALL=0 +31. No luckier allocation.
155. Two more dense-layer forms (track B build, e0661-e0663): the down phase's last 8 tiles requested right after the
   barrier into their own registers (DMLP 16384; batch-8 frame 144/40/40, batch-1 176/24/24): +15; the gate/up loop
   over the 32 tiles not preloaded with #pragma unroll 16 (DMLP 32768, frames unchanged): -3..-5. Confirming on two
   more cards and in track A (e0664-e0666).
156. DMLP=33158 checks for both tracks passed (e0667-e0678). s0021 (track A): HTTP B=1 1398.2 (expected ~1399; the
   best track-A figure), B=8 3361. s0022 (track B, same source): HTTP B=1 1371.9 (new best; expected ~1371), B=8 3353.
   The dense down loop fully unrolled (DMLP 65536; e0680/e0681): 0..-1, noise.
157. The dense epilogue on two warps (three row groups each, constant ranges, plain stores; DMLP 131072; frame
   176/16/8 unchanged; e0684-e0686): timeline epilogue 5 -> 2.7 us, dense layer end 31.5 -> 29.2, but end to end
   +38 / +37: the batch-1 kernel's allocation (not visible in the frame size) again. Closed.
158. The dense MLP as an out-of-line function (DNI; to shield the caller's allocation from changes inside it):
   dense_mlp spills 392 B, frames 480/496 cumulative, B=1 +115 (e0687-e0690). Closed.
159. The sketch's per-row-group error-norm tiles and scales as volatile loads at the row group's start (HSKH; were they
   sunk to their use, a round trip per row group?): pass 1 49.55 -> 49.17 us (HSK_STATS, e0691): no. Closed.
160. The head and the step boundary, stamped (study stamps HSTAMP 10-13 and per-block pass-2 time + candidate count;
   tools/bt_gen.py MK_CLK_X=10 / MK_CLK_P2=1; e0692-e0694): after layer 48 the median block finishes its head at
   +89.5 us but one block at +97.9 and every block waits for it. That block is the one with a candidate row group:
   most steps have one candidate in one block (the winner), some up to 12 blocks with one each. Pass 2 recomputes a
   candidate on two warps (the exact head's even/odd chains), 64 tiles each in four rounds of 16: ~9 us per candidate.
   HSP2: one candidate at a time, all 16 warps load its 128 tiles into shared memory in one round (56-120 KB: free in
   pass 2 at batch <= 2), then the two warps run the same chains from there (same order: bit-identical by
   construction); frames unchanged: pass 2 per candidate 9.0 -> 3.6 us (e0695). A/B and tokens e0696-e0698.
161. HSP2 (e0696-e0698): B=1 -5..-6 on two cards, tokens identical to track A. Track-B checks (e0699-e0705): bitcheck
   1.0, tokens identical, digests, flake 0. s0023 (track B; only the sketch's code changed, so no track-A twin):
   HTTP B=1 1368.1 (new best; expected ~1366), B=8 3352.
162. The likely candidate's exact rows (the row group with this block's largest upper bound) bulk-prefetched by warp
   1 during the exchange (HSPF): pass 2 3.6 -> 3.0-3.5 us, end to end +7..+10 (e0706-e0708). Closed.
163. Block 0's token gather: the bump right after the fence that publishes the tokens, the host's progress word and
   its fence after the bump (GORD; e0709-e0713): -1..-2 / -1..-2 / 0 on three cards. Inside one evaluation's
   noise; not submitted (it reorders the host's progress handshake for ~1 us).
164. The boundary on s0023's source (HSTAMP study stamps, e0714/e0715): after layer 48 (F done ~+26.6) heads end at
   median +90.2, the candidate's block at +93.8 (pass 2 now ~3.6 us: +3.5 instead of +8); the heads-done wait passes
   at +94.2; block 0's gather ends at +98.3 vs the others' +95: block 0's ~4 us gather (strong loads of 152 x 8
   pairs, reduce, curtok/gen, two fences in serve mode, bump) is on the path into the next step (GORD took ~1 of it).
   Not done: an atomicMax of a packed (ordered value, ~index) key per sequence instead of the gather (same token,
   one load for block 0; needs a new buffer, its reset before the bump and at launch for the watchdog path): ~1-2 us.
165. GAMX (journal 164's atomicMax key; tools/gamx.patch: key helper, atomicMax at both publish sites, block 0 reads
   and resets with one atomicExch, reset at launch; e0716-e0719): tokens identical (gen_cmp, hsk_bsizes), but the
   batch-1 kernel's spills 16/8 -> 40/40 and B=1 +29 / +28. Closed.
166. Code-generation options on s0023's track-B build (tools/sweep.sh, e0720/e0721): -Xptxas -O2, -Xptxas
   --allow-expensive-optimizations=true (with -O3 too), -Xcicc -O2: all within +-2; -Xptxas -O1 (frame 208) +285,
   -Xcicc -O1 +3800. The default build is not beaten by any option tried.
167. ncu on the persistent kernel works through mk exec (tools/ncu_step.py: a short step() workload; `ncu --section
   SourceCounters --kernel-name regex:mk_kernel --launch-skip 40 --launch-count 1`, then `ncu --import R --page source
   --csv --print-source sass`: per-SASS warp-stall samples with the reasons as columns; e0722/e0723). With a pmevent
   marker (PMTRIG) in the dense epilogue's loop (executed 1536 = 6 x 2 layers x 128 blocks): only single-digit stall
   samples per instruction there, ~150 in all; the loop is not where the time goes, the ~1 us per row group of the
   stamps (journal 149, 153) is suspect. A sampled-stall view of the whole step is the better tool for what is left.
168. The whole step() kernel by stall samples (same report, e0724; 95752 samples): the top sites are warps parked at
   barriers while one warp polls (stall_barrier on BSSY/BRA/LDG/LDC lines: ~45% of the top 45) and sentinel polls
   (ISETP ... 0x7fc17fc1, stall_long_sb); the streaming loops (LDG.E.128, HMMA) are a few percent each. The step is
   hand-over latency, as the timelines say; no single hot instruction to remove.
169. Stall samples per source line (build with -lineinfo; `ncu --import R --page source --csv --print-source
   cuda,sass`, aggregated by tools/ncu_lines.py; e0725-e0729): sentinel checks (is_sent) 6.6%, the norm's stream
   read 5.7%, ... and nw_load (the next norm's 8 KB of weights, global -> shared, every thread waits for its load
   at the store) 4.1% + 0.6%: at the layer's end it sits right before the barrier into the next layer, so the last
   block to arrive pays its latency. NWE: the next layer's norm weights copied with cp.async right after phase D's
   norm (its last reader; after the barrier that follows D's mma), cp.async.wait_all where the load was; MoE layers,
   main blocks (frames unchanged). A/B and tokens e0731-e0733.
170. NWE (e0730-e0733): tokens identical, frames unchanged, B=1 +13 / +14. The same weights only prefetched to L2 at
   the start of A by an idle warp (NWP, 2 x 8 KB; e0734-e0736): +3..+5. Closed: the norm weights' load stays where
   it is.
171. Layer 0's dense MLP slice (gate/up 384 KB + down 192 KB per main block) bulk-prefetched to L2 by the idle warp
   right after the block's head (DPF0; HBM idles at the step boundary for ~8 us; e0737-e0739): +0..+2, closed.
   Layer 1's A and C phases match layer 20's (e0740; layer 0's stamps are not valid with MK_CLK_L=0: slot 14).
172. The sketch's tiles as 16 volatile ld.global.cs loads up front (HSKE=6: evict-first and all in flight, the two
   properties the exact head's loads have; e0741): pass 1 50.1 -> 51.2 us. Closed with journal 140.
173. Block 0's publish of the chosen tokens (GORD; e0742-e0745): 2 = the tokens by atomic exchange, settled, then the
   bump (the pattern every other cross-block result uses), the generated tokens and the host's progress word fenced
   after the bump: B=1 -1.5..-2.5 / -1..-4 on two cards, tokens identical. Made the default (both tracks); checks
   e0746-e0757, track A A/B e0758.
174. GORD=2 checks for both tracks passed (e0746-e0757); track A A/B 0 (e0758). s0024 (track A): HTTP B=1 1399.5
   (expected ~1398), B=8 3355. s0025 (track B, same source): HTTP B=1 1360.9 (new best; expected ~1366: -7 against
   s0023, more than the A/B's -2), B=8 3332.
175. The dense epilogue's writes as atomics with one settle and no fence (DMLP 64 instead of 256; 32966 vs 33158;
   e0759-e0761): +4..+6; the plain stores + one fence stay. Boundary on s0025 (HSTAMP study, e0763): heads end at
   median +90.0 / candidate block +93.5, heads-done wait passed +94.2, block 0 ends its step (after the post-bump
   fences now) at +98.5. The pod was replaced during this (epoch 4): the saved reference tokens in its /tmp were
   regenerated (tools/bcmp_tracks.sh, e0762: track B identical to track A).
176. Random combinations of the bit-identical switches on s0025's track-B build (18 builds, e0764-e0766): all within
   +-2 of their card's base (card 3 ~10 us slower throughout). No luckier allocation.
177. Pass 2's candidate tiles copied global -> shared with cp.async instead of through registers (HSP2C; frames
   unchanged; e0767-e0770): tokens identical, B=1 +2..+4. Closed.
178. The older load-form and gather switches on s0025's track-B build (e0771): TLOAD_ASM=1 +6, KV_ASM=0 (frame 192)
   +31, GATHER_PAR=0 (one thread per sequence gathers) +46. The defaults stand.
179. Head durations per block on each block's own clock (study stamps; e0772): main blocks 62 us from head entry to
   done, the aux blocks 74 us (they enter ~12 us earlier, during the last layer's E/F, and stream slower beside
   them), both groups then wait ~4.3-4.8 us; the candidate's block waits 0.4: it is still the last (pass 2).
180. Speculative pass 2 (HSPS, tools/hsps.patch): after its bump, each block recomputes its row group with the largest
   upper bound while the exchange runs; pass 2 reuses it when it is a candidate (same values; tokens identical,
   e0776). B=1 +38 / +38 (e0774/e0775; batch-1 spills 16/8 -> 24/24, and 152 x 64 KB of exact rows read while
   blocks are still streaming pass 1, like HSPF's +7..+10). Closed.
181. HPF3Q again on s0025 (the last layer's idle slot fetches the sketch's first 20 MB = each block's first ~4 row
   groups, instead of the exact head's; host-side pointer only; e0777/e0778 and reversed e0780/e0781): -1..-3, 0..-2,
   then mixed / -1..-2: ~-1, inside one evaluation's noise. Not submitted.
182. MK_PAD on s0025 (track B, scored layout, e0779): 512 / 1024 / 4096 equal (1371-1378 on card 3), 0 +10 (the layout
   effect is now ~10 us instead of ~30). The 512-byte pad stays.
183. Allocation-lottery probe: a compiler scheduling barrier (asm volatile("" ::: "memory"), no instruction) at one
   phase marker at a time (LOTT = 1..99 over the PROF ids; 41 sites measured, e0782-e0784): every one within +-1.5 of
   its card's base. The phase boundaries are already scheduling barriers; this does not reach the allocation.
184. Final check of the committed engine (s0025, TRACK=B) on the current pod (epoch 4) through the two-process served
   path (tools/sock_long.py, e0785): 1359.6 / 1358.7 / 1360.2 us per token, as the evaluation's 1360.9.
185. Longer determinism runs of the final engine (s0025, TRACK=B; e0786-e0788): flake.py 500 on two cards: step B=1,
   step B=8, generate B=8 (300 steps), generate B=1 (600 steps): 0 mismatches in 499 repetitions each, on both;
   flake_b.py 120 (batches 3, 5, 7 step and generate, batch 8 prompt 64 + 700 steps): 0 mismatches in 119 (29).
   The plain stores + fence of the dense epilogue (s0019) and the token publish by exchange (s0024) hold up.
186. Stall samples per line in the generate() path (tools/ncu_gen.py: step() to 300, then one generate(32) launch,
   -lineinfo, track B; e0789): sentinel checks 9.0%, the norms' reads of the stream 5.9%, phase C's x_chk (warp 0
   waiting for the other blocks' rows) 5.7%, nw_load 4.0%, the head's final norm 3.9% (mostly aux blocks parked in
   it while the last layer's E/F runs), barriers 3.5%; the sketch's IMMA loop 1.4%. Hand-over waits, as before.
187. Speculative pass 2 gated (HSPS=2, tools/hsps2.patch): a block speculates only if its best upper bound reaches the
   previous step's L (kept in its unused lbv slot 7 at batch <= 6), so only the likely winners' blocks read exact rows
   during the exchange. Tokens identical (e0793); B=1 +20..+24 (e0791/e0792; HSPS ungated +38): the contention part
   went, the rest is the batch-1 kernel's allocation (spills 16/8 -> 24/24, as with HSPS). Closed.
188. Speculative pass 2 out of line (hsk_spec, __noinline__; tools/hsps3.patch): the kernel's frame back to 176/16/8,
   tokens identical (e0798), B=1 +4..+6 (e0796/e0797; in-function +20..+24). Coverage counted (HSK_STATS, scored
   workload, e0799/e0800): ~9.6 candidate row groups per step over several blocks, often more than one per block;
   gated speculation (3 blocks/step) reuses 2.5 of them (hit rate 84%), margins 1.0 / 2.0 reuse 3.0 / 3.5, and even
   every block speculating reuses only 4.4: the blocks with an unspeculated candidate still set the tail. Closed.
189. Pass 2, the block's later candidates (2nd on) bulk-prefetched to L2 at its start (HSP2P; HBM idles after the
   exchange; frame 176/16/8; e0801-e0803): 0..+2. Closed. Left for whoever continues: blocks often hold 2-3 of the
   ~9.6 candidates per step (journal 188) and HSP2 takes them one at a time (~3.6 us each); overlapping one
   candidate's load with the previous one's chains needs a second 64 KB staging buffer that pass 2 does not have.
190. Pass 2 in pairs (HSP2=2, tools/hsp2pairs.patch): two candidates at once, 32 KB of staging each per half
   (tiles 0-63, then 64-127), warps 0-3 run the two pairs of chains in the same tile order. Tokens identical
   (e0807); batch-1 frame 176 -> 192: B=1 +22..+25 (e0805/e0806); out of line (hsk_pass2) the frame stays 192 and
   B=1 +30 (e0808/e0809). Closed.
191. MK_PAD further (scored layout, e0810/e0811): 512, 64 KB, 512 KB, 1 MB all equal (the good state); 2 MB + 512,
   3 MB, 6 MB +12..+18 (the slow state of the 2 MB-granular placement, journal 62-73). No pad beats 512 B.
192. HPF3B (the batch-8 kernel's q/k/v slot also fetching the head in the last layer; B=8 -3 on s0015's source) on
   s0025 (e0812-e0814): B=8 client-like seeds 300 / 700: -1..+0 / -1..+0 (noise), B=1 +1. No longer worth a
   submission; stays off.
193. Pass 1's four 16-tile chunks per row group as a loop that is not unrolled (HSKU; e0815): pass 1 49.7 -> 54.3 us
   (HSK_STATS). The unrolled form's interleaving is the better one. Closed.
194. The sketch's first-row-group prefetches all issued by warp 15 instead of by each warp (HSKPF2; e0816): prep
   6.43 -> 6.57, pass 1 49.74 -> 49.69 (HSK_STATS): the preparation does not stall on the prefetch issue. Closed.
195. Where the sketch's 6.4 us of preparation goes (timing-only study builds that skip work, PREPX; e0817): without
   stage 1 (per-block sums, warp 0 alone at batch 1) 5.4, without the digits 4.8, without both 2.4 (barriers, norm
   fragments, scales, the prefetch issue). Stage 1 on 8 warps would save ~0.8 us (a new summation order inside the
   certified bound: still valid, tokens unchanged) — below one evaluation's noise; not done.
196. Stage 1 of the preparation on 8 warps at batch 1 (PREP8: 8 of each lane's 64 squares per warp, combined in order
   by warp 0; frame unchanged; e0818): prep 6.50 -> 6.60 us, candidates unchanged (9.60 per step). The two extra
   barriers take what the parallel sums save. Closed.
197. head_sketch inlined (HSK_INL=__forceinline__) on s0025's source (e0819/e0820): batch-1 frame 192/40/40, B=1 +45.
   Out of line stays right.
198. NACC=4 (four accumulator chains in the matvecs; would change the summation order, so a re-validation of both
   tracks) does not build on this source (e0821); NACC=2 stays the only form.
199. gather_best inlined (e0822/e0823): B=1 +125. Out of line stays right (as journal 197 for the sketch).
200. The head's final norm as an out-of-line call (FNNI; e0824/e0825): frame 448 (cumulative), B=1 +47. Moving
   once-per-step code out of the kernel does not free the hot paths' allocation either. Closed.
201. More code-generation options (e0826): --extra-device-vectorization, -Xcicc -O3, -Xptxas
   --disable-optimizer-constants: within +-2; an undocumented ptxas switch: frame 208, +160. Defaults stand.
202. The L1/shared split (cudaFuncAttributePreferredSharedMemoryCarveout, host side; e0827/e0828): 0 / 25 / 50 / 56 %
   equal to the default (the smallest carveout that fits the 128 KB is already used); 58 / 70 % +20 (a larger
   carveout, less L1), 100 % +245. The default stands; worth knowing: the step is sensitive to L1 size.
203. Lead from 202 (not done): one step up in shared carveout (less L1) costs +20, so the reverse may pay: a batch-1
   kernel with <= 100 KB of shared memory would get the 100 KB carveout and ~32 KB more L1. What holds it at 128 KB:
   the aux blocks' K/V staging for long contexts (from SM_XF upward), the attention partials at SM_PART (120 KB),
   and in track B the sketch's digits (88-120 KB), norms/bounds (120 KB+) and pass 2's staging (56-120 KB).
204. The sketch's tiles with L1::no_allocate (non-volatile asm, no policy register; HSKE=7; frame unchanged;
   e0829/e0830): pass 1 49.7 -> 50.9 us, B=1 +3. Allocating them in L1 does not hurt; closed.
205. lda (the K/V loads issued through it) with L1::no_allocate (KVNA; e0831/e0832): +0..+1, noise. Closed.
206. The staged K/V loads (KV_STG_ASM=1) with and without L1::no_allocate (e0833): -1..+1, noise. The K/V traffic's
   L1 allocation does not matter; closed.
207. The final source as track A through the served path on the current pod (e0834): 1397.9 / 1395.1 / 1396.2 us per
   token (s0024's evaluation 1399.5); as track B 1358.7-1360.2 (journal 184). engine/ = s0025's submitted source
   exactly (mk.cu, api.py, build.sh, TRACK compared with the submission's archive).
208. DMLP bit 16384 (the dense down phase's last 8 tiles right after the barrier, own registers) on s0025 (e0835):
   +20 (on s0020's source +15). Stays off.
209. Where the next layer's q/k/v prefetch is issued, on s0025 (e0836): QPF_AT=1 (after D's mma) +17, 0 (before the
   routing) +19 against 2 (end of C). The s0008 placement still holds.
210. Re-checking the prefetch switches on s0025 (e0837): QPF2_AT=0 / QPF2=0 within noise, EVF=0 +8, QPF=0 -7. QPF
   only gates the batch-8 q/k/v prefetch statement, which sits in the MoE path but carried a "!L.moe" clause: the
   batch-1 kernel could not prove it dead (L.moe is a runtime field) and kept the branch. QPFNM=1 drops the clause:
   batch-1 SASS identical to the QPF=0 build (same hash), the batch-8 kernel unchanged. Scored layout (e0838/e0839):
   -6..-9; digests, bcheck unchanged (e0840, e0842), flake 0/199 both tracks (e0843/e0844).
211. s0026 (track A): HTTP B=1 1393.1 (best track-A figure; expected ~1392), B=8 3339. s0027 (track B, same source):
   HTTP B=1 1352.6 (new best; expected ~1353), B=8 3333. A dead branch was worth 8 us of allocation.
