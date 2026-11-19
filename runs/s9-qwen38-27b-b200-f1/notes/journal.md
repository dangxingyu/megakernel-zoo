# Journal of run s9-qwen38-27b-b200-f1 (inherited: s5's s0016 track A / s0017 track B; their notes: *_s5.md, *_s4.md)

Goal: track A >= 1.10x SGLang at both batch sizes: <= 8720 us at B=1 and <= 9330 us at B=8 over HTTP (inherited:
8886 / 9843). All timings below are same-card alternating A/Bs by CUDA events (tools/abn.sh + tools/tv.py: 3 x 48
one-token passes, context ~600) unless they say "HTTP"; tv.py also hashes the logits of 6 steps at B=1, 3, 8, and every
variant below has the hash of the inherited kernel (bit-identical) unless stated.

## hour 0-0.3: baseline and where B=8 loses
- Inherited engine on this machine: events 8875-8895 us at B=1, 9770-9810 at B=8. s0001 (inherited + stats() label):
  VALID 8885 / 9836 over HTTP. Gates: numeric max 0.14 (limit 0.38), numeric_long max 0.457 (limit 0.592).
- Per-op times (tools/b8_split.py, stamped build, us per op, B=1 -> B=8): gate_up in recurrent layers 54.7 -> 62.1,
  gate_up in attention layers 53.4 -> 55.0, recurrent op 8.8 -> 17.2, in_proj 26.6 -> 27.7, down 28.6 -> 28.9,
  out_proj 13.0 -> 13.9. So B=8 loses ~+400 us in the recurrent op and ~+355 us in the gate_up that follows it,
  and almost nothing anywhere else.
- In isolation (tools/prof_kinds.py: 40 gate_up ops in one launch) gate_up costs the same at B=1 and B=8 (53.6 /
  54.6 us): the B=8 loss of gate_up is not the 8 columns, it is what ran before it.
- Timing-only flags (wrong results): recurrent state not stored (flag 128): gate_up:L 62.1 -> 54.9, recurrent op
  17.2 -> 14.3; not loaded (256): gate_up unchanged, recurrent op 16.0. By batch: gate_up:L excess over gate_up:A
  B=4 +3.1 us (12.3 MB of state per layer), B=6 +5.3 (18.4 MB), B=8 +7.1 (24.6 MB): 0.29 us per MB written.
  => findings S1: the dirty state is written back to DRAM when the next big stream (gate_up, 356 MB) pushes it out
  of L2, and that costs twice what reading the same bytes costs. ~485 us per B=8 step (5 %).

## hour 0.3-0.6: deferred state commits (F_DEFER)
- Design: the stored state is rewritten only on every N-th one-token step; the steps between save their rank-1
  update (decay, k[128], delta[128]: 1 KB per head instead of 64 KB) and the following steps replay the saved
  updates with the same lm_upd in the same order before their own (the machinery of the speculative passes). Every
  operator of every token is evaluated; the state in the registers is bit for bit that of a step-by-step run.
  Counter per sequence in nacc[NCM + b], advanced by OP_FINISH; a multi-position pass replays, commits and clears.
- tools/defer_check.py: logits of 40 steps bitwise equal to N = 1 for N = 2, 3, 4, 5 at B = 1, 2, 3, 4, 5, 8, also with
  a 4-position pass in the middle (replay + commit path).
- First version: B=8 9920 (N=1) -> 9692 (N=4), but N=1 itself was +110 us over the inherited kernel: ptxas spills
  of lmix_dual 32/44 -> 68/120 bytes (saved-update pointers and counts live across the phases).
  Second: two instances lmix_dual<DEFER>; the plain one is back at the inherited time (9805 vs 9808), the deferred
  one 52/80 bytes: N=3 9680, N=4 9674, N=5 9690.

## hour 0.6-1.1: what a step's "cold" memory costs; the cache operator of the weight loads
- Negative (cg regime, findings S4): per-warp early prefetch of the next GEMM (+390 us at B=1), next-op descriptor
  preload (+125), prefetch of the gains / output lines of the next epilogue (+20 / +30), X_COMPACT for BF16 (-16 /
  +95). The per-warp map (tools/warp_map.py) showed the prefetched warps 0-3 ending 4-6 us early in every GEMM, but
  moving the prefetch only moves traffic in front of the slower warps.
- Weight loads with another cache operator (LDW_K): cs 8658 / 9452 us (B=1 / B=8) against cg 8878 / 9785; lu the
  same as cs; ca, cv no gain. Bit-identical. findings S2.
- s0002 (track A, ld.global.cs for the weight streams): predicted ~8675 / ~9480; VALID 8701 / 9453 over HTTP
  (s0001: 8885 / 9836): -2.1 % / -3.9 %. B=1 is under the 1.10x line (8720) by 19 us; B=8 needs 9330.
- State prefetch of the B >= 4 recurrent op issued after the first unit's input loads (LMPF_LATE, findings S3):
  B=8 9454 -> 9376 (cs regime); with deferred commits (N = 4) 9301. Deferred commits alone in the cs regime: 9454 ->
  9384 (N = 4), 9373 (N = 3): half of what they gave in the cg regime (9779 -> 9634).
- Cache operators of the recurrent state (cs regime, B=8, N = 1): stores cs 9419 (+50), wt 9368, cg 9368 (no change);
  loads cs 9368 (no change). Not taken.

## hour 1.1-1.6: the B=8 recurrent op in one pass (lmix_tri); kernel entry points
- cs-regime profile at B=8 (b8_split, stamped): gate_up after a recurrent op 55.0 us = gate_up after attention 55.0:
  the write-back penalty of S1 is gone from gate_up (dirty state no longer competes with weight lines in L2), which
  is where most of the cs gain at B=8 came from. Deferred commits therefore give only -70 us now (9454 -> 9384), by
  making every GEMM a little faster, and the recurrent op itself (16.2 us per layer, 777 us per step) is the largest
  B=8-only cost. Phase timer (block 0): inputs 3.4, recurrence 9.2, norm 1.3 us per layer over two rounds.
- K/V cache loads with cs: B=8 -10 us. Not taken: the K/V tiles are appended by another SM in the same launch and I
  have not measured whether cs loads bypass L1 as cg does (findings C2 of s4 tested cg, cv, volatile only).
- Near-prefetch sweep in the cs regime (bytes of each block's share of the next GEMM asked from L2 at the barrier):
  0: +615 / +650 us (B=1 / B=8), 64 KB +225 / +280, 96 KB +90 / +150, 128 KB (default) 0, 160 KB -12 / -44,
  192 KB +16 / -30, 256 KB +105 / +70, 512 KB +300 / +280. Far prefetch (before small ops): 64 KB -25 / -22, 128 KB
  -20 / +5, 256 KB +28 / +38, 512 KB +82 / +115, 1 MB +90 / +85.
- lmix_tri (findings S5): one pass per block for one-token passes of 7-8 sequences. Bit-identical at B=1..8 (hash of
  20 steps each, against the same source with -DLM_TRI_OFF). B=8 9376 -> 9244; B=7 9346 -> 9227; B=4 +22 us (no third
  unit there: B=4..6 stay on lmix_dual).
- Code generation, again: lmix_tri in the common kernel made B=1 22 us slower (8662 -> 8684) though B=1 never calls
  it. A kernel pruned to its own code paths (only NG = 1 GEMM instances) was 33 % slower in every GEMM: ptxas
  reported 72 / 52 bytes of spill in gemm_stream_bf<1> there (0 in the full kernel), 284 / 328 in finalize<1> (160 /
  144). Final form: two entry points with one body; mk_kernel keeps the inherited call sites (B=1: 8651), mk_kernel_b
  has lmix_tri in the place of lmix_dual<false> (B=8: 9244).
- Checks of the final source: logits vs HF BF16 reference rel-L2 median 0.0101 / max 0.0296, top-1 99.4 % (as the
  inherited engine); single == batched bitwise; generate == step greedy; submit/poll == step greedy at B=1 and 8;
  10 x 96 steps repeated bitwise at B=1 and B=8 (tools/stress.py).

## hour 1.4-2.0: s0003; prefetch in the cs regime; the served path
- s0003 (track A, lmix_tri): predicted ~8695 / ~9250; VALID 8694 / 9222 over HTTP. Both under the 1.10x lines of the
  same-card rematch (8720 / 9330). The evaluator's own speedup block divides the trial figures (9523 / 10231):
  1.095x / 1.109x, so its 10 % milestone still wants B=1 <= 8657.
- Prefetch 4 x 4 sweep (near 128-176 KB x far 0-96 KB, one card): best near 160 KB + far 64 KB: 8664 -> 8622 /
  9221 -> 9140. s0004 submitted with these two constants.
- Negative: the barrier's 64 KB prefetch pieces issued by two or three warps at once instead of one after the other
  by thread 32 (-DPF_PAR): +5 us at both B. Removed. Negative: state prefetch after the first unit's input loads in
  the B=1 recurrent path too (-DLMPF2_LATE): B=1 +15 us; left off.
- Track B check of the same source (tools/w12_full.sh, TRACK "A W12"): 257 GEMMs bitwise equal to BF16 storage at B=8
  and B=1, storage bit for bit (0 of 25,621,954,560 weights differ), logits bitwise, submit/poll == step greedy;
  served ~7885 us at B=1 and ~8371 at B=8 in process.
- Served path: between two passes the inherited engine enqueued a device-to-host copy of the chosen tokens and an
  event. With the tokens written by the kernel's last op straight into page-locked host memory mapped for the device
  (cudaHostAllocMapped; a slot counts when its upper half carries the cohort's tag): served us per step 8639-8652 ->
  8621-8622 at B=1, 9142-9160 -> 9128-9153 at B=8 (tools/online_time.py, 512 tokens), tokens equal to step-by-step
  greedy. Kept for its own submission.
- Cumulative far prefetch (MK_PF_CUM: the barriers before the small ops that precede a GEMM ask for consecutive
  64 KB pieces of the block's share, then the near piece): 8621 -> 8605 / 9168 -> 9150; 96 KB pieces -9 / -8,
  128 KB -8 / -3.

## hour 1.7-2.1: s0004, s0005; idle blocks; the fused attention op
- s0004 (track A, near 160 KB / far 64 KB): predicted ~8652 / ~9145; VALID 8641 / 9145 over HTTP: 1.110x / 1.122x
  the same-card SGLang rematch (9594 / 10265), 1.102x / 1.119x the trial figures the evaluator divides by: its
  milestone h0_step (10 % over every general engine at both batch sizes) passed.
- s0005 (track A, host-mapped tokens) submitted: expected ~8628 / ~9130.
- Blocks without a unit in the B=1 recurrent op (56 of 152) asking L2 for their whole share of out_proj at the barrier
  before it (MK_PF_IDLE): the last 56: 8606 -> 8588; only the last 32: 8574. Per-batch-size prefetch tables (built
  on first use) to make this a policy; the count is being swept.
- W12 engine prefetch sweep (cs loads): flat; 144 / 64 KB 7840 / 8350 against the s5 default 128 / 96 KB 7838 / 8372.
- Fused attention (OP_ATT_ALL, att_all_work): one op per attention layer, one block per (sequence, kv head), written
  as a copy of the four ops' arithmetic; check tool tools/att_check.py (checksum of every step's logits to position
  1150 against the separate ops, same engine).

## hour 2.1-2.4: s0005, s0006 (track A), s0007 (track B); fused attention, first form
- s0005 (track A, host-mapped tokens): predicted ~8628 / ~9130; VALID 8628 / 9134.
- s0006 (track A, prefetch schedule: consecutive far pieces, idle-block boost at B <= 3; kernel = s0005): predicted
  ~8582 / ~9125; VALID 8587 / 9114: 1.117x / 1.126x the same-card SGLang rematch (9594 / 10265).
- s0007 (track B = the s0006 source built with TRACK "A W12") submitted after tools/w12_full.sh on that source: 257
  GEMMs bitwise at B=8 and B=1, 25,621,954,560 weights bit for bit, logits bitwise against BF16 storage, submit/poll ==
  greedy; in-process serving 7822 / 8316 us per step.
- Fused attention (one op, one block per (sequence, kv head); bench/variants/mk_attall.cu + api_attall.py, run with
  bench/variants/run_attall.sh): bit-identical to the four ops over 1150 steps at B=1, 3, 8 (tools/att_check.py), but
  the op takes 25.7 us at B=1 (context 600), 30.0 at B=8, 33.5 at context 1000, against 23.3 for the four ops
  together; without its L2 prefetch of the K and V strips 34.2; the fence after the K/V append costs nothing.
  Reading: a block that loads 4 x 512 B per warp and batch gets ~64 KB per round trip of ~1 us (the same ~55-64 GB/s
  per SM as the recurrent op's state loads), and the unit makes ~20 such round trips for 0.6 MB of K and V.
  Second form (8 loads in flight per batch, two warps per dim tile in the value sum) being measured.

## hour 2.4-3.3: track B on the board; the grid barrier's release; attention in one op (three forms)
- s0007 (track B, the s0006 source with TRACK "A W12"): predicted ~7835 / ~8330; VALID 7847 / 8360 over HTTP: 1.223x /
  1.228x the same-card SGLang rematch (s5's s0017: 8117 / 8864 in the operator's rematch).
- Grid barrier with a relaxed add (findings S7): -157 / -196 us per step, identical logits in 12 steps - and unsafe:
  bench/rel.cu finds 864 stale reads in 1.25e11 when the memory system is idle (0 with the release add). Dropped.
- Attention as one op per layer, measured per op in a stamped build (us per attention layer, B=1 / B=8), the four
  separate ops being 24.3 / 22.3 at context 100, 23.7 / 24.7 at 600, 24.9 / 27.8 at 1000:
  1. one block per (sequence, kv head), 4 loads in flight: 25.7 / 30.0 at 600, 33.5 at 1000 (B=1);
  2. the same with 8 loads in flight and two warps per dim tile: 19.2 / 24.0 at 600, 21.0 / 27.6 at 1000;
  3. one block per 256-key chunk, hand-overs by acquire/release counters: 19.8 / 19.4, 22.3 / 25.5, 22.7 / 29.3;
     phase timer of the hub block: the three atomics, two spins and the fence are most of it (~3 us each);
  4. the same with sentinel slots (relaxed gpu-scope stores and polls on one location) and relaxed K/V appends:
     16.5 / 15.6, 19.0 / 22.1, 19.7 / 26.4. Moving its L2 prefetch of the chunk's K and V behind the prep loads
     made it slower again (19.7 / 18.3, 22.1 / 22.9, 22.6 / 25.2): kept at the op's start.
  All forms bit-identical to the four ops over 1150 teacher-forced steps at B=1, 3, 8 (tools/att_check.py).
  Form 4 ported onto the current engine as a candidate (bench/variants/mk_attdist.cu, api_attdist.py) for a same-card A/B.

## hour 3.3-3.6: the one-op attention layer in the engine; s0008 (track A), s0009 (track B)
- Candidate on the current engine (bench/variants/run_attdist.sh, same card, events, context ~620): engine 8585 / 9130;
  candidate with the four ops 8585 / 9121 (the new code's presence costs nothing); one op 8560 / 9091. Prefetch piece
  at the barrier before the op (MK_PF_FAR_ATT): 64 KB 8566 / 9094, 128 KB 8557 / 9097, 192 KB 8566 / 9112, 256 KB
  8571 / 9122, the whole rest of the share 8520 / 9100: whole share at B <= 3, 64 KB otherwise.
- Checks of the integrated source: one op == four ops over 1150 steps at B=1..8; hash of 20 steps at B=1..8 equal to
  the four-op run; 4 x 640 steps repeated bitwise at B=1 and B=8; test_engine; served 1024 tokens == greedy.
  Events: 8587 -> 8534 (B=1), 8814 -> 8759 (B=2), 9113 -> 9088 (B=8).
- s0008 (track A, one-op attention layer): predicted ~8535 / ~9090; VALID 8492 / 9077 over HTTP: 1.130x / 1.131x the
  same-card SGLang rematch. (B=1 better than predicted: the served workload spends 19 % of its steps below position
  256, where the op has one chunk and no hand-over.)
- s0009 (track B = the s0008 source with TRACK "A W12") submitted after tools/w12_full.sh + att_check on that source
  (257 GEMMs bitwise at both batch sizes, storage bit for bit, logits bitwise, one op == four ops over 1150 steps at
  B=1, 3, 8): in-process serving 7700 / 8229.

## hour 3.6-4.1: parameter re-sweeps; slack warps (B=8)
- s0009 (track B, the s0008 source): predicted ~7715 / ~8245; VALID 7706 / 8253 over HTTP: 1.245x / 1.244x SGLang.
- Negative: the attention op's score warps loading their first 8 K steps during the query prep: +68 / +64 us (448
  bytes of spill in att_dist_work: 8 operand registers live across a barrier). Reverted.
- Re-sweeps on the s0008 engine, all flat: near/far prefetch (144-176 KB x 48-80 KB: within 5 us, except far 80 KB
  -9 us at B=8), idle-block count 24-48 (within 8 us), HC_TILES 4, barrier back-off 0-100 ns (within 3 us).
- R_ASYNC_MAX = 8: B=5 -30 us, B=8 +115 us -> findings S9: r partials by warps 0-3 only, then r finished eagerly by
  warp 0, in the B >= 7 kernel: B=8 9088 -> 9037 -> 9004, B=7 9099 -> 9010, B=1 unchanged; logits hash identical.
- Negative: gains of the residual epilogue preloaded into shared memory (-8 / -3 us): dropped.

## hour 4.1-4.6: s0010 / s0011; the recurrent op at B=8 once more
- s0010 (track A, r partials by the prefetched warps at B=7-8): predicted ~8492 / ~8990; VALID 8494 / 8982 over HTTP:
  1.130x / 1.143x the same-card SGLang rematch.
- s0011 (track B, the s0010 source): predicted ~7706 / ~8165; VALID 7709 / 8184: 1.245x / 1.254x.
- Phase timer of lmix_tri (bench/variants/mk_triprof.cu, block 0 with three units, us per layer): inputs 3.27,
  normalise 0.22, own unit's two columns 4.78, shared unit's column 2.19, norm + output 0.73 (sum 11.2; the op 13.3).
- Pipelined state loads: the second column's 16 loads issued with the first's, the shared unit's column loaded while
  the second is computed; k read from shared memory at both uses instead of 16 registers (lm_tri_core). Bit-identical
  (hash at B=1, 5, 7, 8). B=8 8994 -> 8965 us per step, B=7 9006 -> 8960; spill of lmix_tri 48 -> 40 bytes.
- Negative: the conv-history rows of the next recurrent op prefetched during in_proj by a slack warp (24 bulk
  prefetches of 20 KB): B=8 +85 us, B=7 +60. Reverted.

## hour 4.6-5.0: s0012 / s0013; conv-history prefetch
- s0012 (track A, pipelined state loads in lmix_tri): predicted ~8494 / ~8952; VALID 8492 / 8956: 1.130x / 1.146x.
- s0013 (track B, the s0012 source): predicted ~7709 / ~8155; VALID 7708 / 8166: 1.245x / 1.257x.
- Conv-history rows of the next recurrent op prefetched during in_proj, second form: one request of 60 KB per sequence
  (the three rows are adjacent in the ring), issued by one thread of block b: B=8 8970 -> 8949, B=7 8960 -> 8928,
  bit-identical. The first form's +85 us was its 24 requests from one warp. The same for the conv weights (160 KB by
  three blocks): 8959, worse than without: they are not cold.

## hour 5.0-5.4: s0014 / s0015; the attention op's K chunk one barrier earlier
- s0014 (track A, conv-history prefetch during in_proj + 80 KB far piece at B >= 4): predicted ~8492 / ~8940; VALID
  8492 / 8948 (s0012: 8492 / 8956): at the evaluator's repeatability.
- s0015 (track B, the s0014 source): predicted ~7708 / ~8155; VALID 7708 / 8193: B=8 27 us WORSE than s0013 (8166).
  I had checked the conv-history prefetch on the 12-bit build for bit-identity but not timed it there. Timed afterwards
  (same card, 3 alternations): off 8156-8168, on 8187-8189 at B=8; B=7 +13. It is compiled out of the W12 build now
  (the best track-B numbers so far are s0013's, from the s0012 source).
- Clean phase timer of the one-op attention layer (bench/variants/mk_attprof.cu: the recurrent ops' stamps off), hub
  block, us per layer at B=1, context 100 / 600 / 1000: prep 1.9 / 1.8 / 1.9, scores + store 4.5 / 4.5 / 4.5, softmax +
  publish 3.4 / 3.8 / 4.1, probability fragments 0.3-0.4, value sums 1.4 / 2.0 / 2.1, gather + output 0.8 / 2.6 / 2.3;
  B=8 at 600: 4.4, 7.0, 3.2, 0.45, 2.6, 3.0.
- The blocks of the attention op asking for their K chunk (131 KB) at the barrier before the op, in the place of a
  weights piece (host table only): B=8 8937 -> 8880 us per step, B=1 8530 -> 8526. On for B >= 4.

## hour 5.4-5.7: s0016 / s0017
- s0016 (track A, the attention blocks' K chunk requested at the barrier before the op, B >= 4): predicted ~8492 /
  ~8900; VALID 8494 / 8905: 1.130x / 1.153x the same-card SGLang rematch.
- s0017 (track B, the s0016 source; conv-history prefetch compiled out of the 12-bit build): predicted ~7708 / ~8115;
  VALID 7710 / 8131: 1.244x / 1.262x. Verified before: tools/w12_full.sh (257 GEMMs bitwise at both batch sizes,
  storage bit for bit, logits bitwise), att_check at B=8 over 1150 steps, served 1024 tokens == greedy.

## hour 5.7-6.0: last tweaks of the attention op
- Host-table sweeps at B=1 (far piece of the blocks that work in the recurrent op 0 / 32 / 128 KB; near piece after a
  small op 128 / 224 / 288 KB): all within 3 us of the defaults except 128 KB far (+14). Unchanged.
- The hub reads its own chunk's scores from shared memory (no slot stores, polls and restores for chunk 0; with one
  chunk, below position 256, no hand-over at all): B=1 8538 -> 8525 us, B=8 8892 -> 8884, bit-identical (hash;
  att_check 1150 steps).
- The chunk's V strips requested after the score loads are out (by warps without a tile) instead of with the K request
  at the op's start: B=8 8885 -> 8868, B=1 unchanged.
- Negative: no K request inside the op: B=1 +61 us (B=8 +5, where the table asks for it one barrier earlier).
- Decision: the two tweaks above are NOT submitted. On the 12-bit build the same source change is +10 to +16 us at
  B=1 whichever tweak is on (old 7702-7715, new 7716-7725; B=8 -23), and a re-check of the BF16 build on another card
  gave B=1 +7 us where the first card gave -12 (B=8 -14 / -24). Not a reproducible gain for the headline number.
  engine/ is the s0016 source again (byte-identical to the submission; TRACK "A W12" builds s0017); the variant is
  bench/variants/mk_hubown_vlate.cu.
- Negative: cs loads for the K / V tiles of the one-op attention layer (-DKVL_K=1; the appended tile is read with
  relaxed loads anyway): B=1 +5 us, B=8 +15. Left on cg.
- The machine was replaced once (epoch 2) during this measurement; the job re-ran by itself.

## hour 6.0-6.4: final per-op profile; the barrier's prefetch behind its increment
- Final-state profiles (tools/b8_split.py, tools/warp_map.py, stamped build, context ~600), us per op at B=1 / B=8:
  gate_up 52.0-52.9 / 53.3-53.7, down 27.8-28.1 / 28.2-28.3, in_proj 26.1 / 26.4, qkv 25.3 / 23.6, out_proj 12.4-13.0 /
  12.9-13.3, recurrent op 8.7 / 12.6, attention op 17.1 / 21.1, lm_head 352 / 363. Per GEMM at B=1 the time from the
  bulk of the warps ending their stream to the op's end is 5.7 us (in_proj: 20.3 -> 26.1), 6.5 (out_proj), 9.1
  (gate_up), 7.0 (down): slowest warp of a block +1.3 to +2.1, slowest block +0.7 to +2.8, hand-over and epilogue
  +2.2, last arrival to release +1.8 to +2.1.
- findings S12: prefetch requests after the increment: B=1 -25 to -31 us, B=8 -9 to -12, both builds. Verification
  and a last submission pair follow.

## hour 6.4-6.8: s0018 / s0019; the barrier's prefetch in steps
- s0018 (track A, prefetch requests after the barrier increment): predicted ~8467 / ~8895; VALID 8466 / 8899: 1.133x /
  1.154x. s0019 (track B, same source; timed on the 12-bit build first): predicted ~7694 / ~8120; VALID 7689 / 8116:
  1.248x / 1.265x.
- Negative: the last block to arrive skips its prefetch (thread 0 polls once before giving the go): +36 / +57 us (the
  extra poll delays every block's prefetch).
- One 64 KB piece per poll of thread 0, none after the release (bench/variants/mk_pfstep.cu): B=1 8505 -> 8463 us,
  B=8 8875 -> 8874 on one card; B=1 -19 to -35, B=2 -47 to -60, B=8 -3 to -27 on another; 12-bit engine 7687 -> 7662 /
  8102 -> 8090. Bit-identical. Verification and a last pair follow.
- s0020 (track A, one prefetch piece per poll, none after the release): predicted ~8434 / ~8890; VALID 8455 / 8880:
  1.135x / 1.156x. B=1 gained 11 us over HTTP where the same-card events (context ~620) said 36: the served workload's
  mean over contexts 64-1088 is not the context-620 figure for this change (not investigated).
- Re-sweeps with the stepped prefetch (events, same card): near 224 KB -4 / -3 us (B=1 / B=8), 288 KB +8 / +25, 384 KB
  +12 / +50; far 128 KB -10 / -10, 256 KB +7 / +25; all 56 idle blocks boosted at B=1: -4. Not taken (far 128 KB would
  be worth ~10 us; below what another evaluation pair can resolve).
- s0021 (track B, the s0020 source; timed on the 12-bit build first): predicted ~7680 / ~8105; VALID 7665 / 8102:
  1.252x / 1.267x the same-card SGLang rematch.

## End of run
- Track A: s0020, 8455 us per token at B=1 and 8880 us per step at B=8 over HTTP (inherited: 8885 / 9836 on this machine):
  -4.8 % / -9.7 %; 1.135x / 1.156x SGLang 0.5.23 (target 1.10x at both).
- Track B: s0021, 7665 / 8102 (inherited s0017 of s5: 8117 / 8864 in the operator's rematch): 1.252x / 1.267x.
- engine/ = the s0020 source (TRACK "A"; "A W12" builds s0021). 21 submissions, all valid, all bit-identical in their
  logits to the inherited engine.
- After the last pair, measured and left as notes (engine/ stays the s0020 source): prefetch pieces of 32 KB under the
  stepping: +125 / +145 us (B=1 / B=8); 128 KB pieces: +18 / +17; 64 KB stays. Far piece 128 KB (BF16 engine): -10 /
  -11 us in three A/Bs on one card - and +8 (B=1), +38 (B=2), +37 (B=5), +58 (B=8) on another card of the machine; 12-bit
  engine: +2 / +2. So it is not a gain, and prefetch constants at the 10-us level are card-specific: the ones in the
  engine were each confirmed by an evaluation on the evaluator's card (near 160 KB, far 64 / 80 KB, the idle-block and
  K-chunk entries).
- Check on the evaluator's card (card 0, events, s0020 engine 8458 / 8865 us at B=1 / B=8): near 128 KB +46 / +56, near
  224 KB -2 / -8, far 128 KB -11 / -7 (as on card 2; card 3 had +8 / +58), the s0018 kernel +31 / +1 (so the stepped
  prefetch is -31 us at B=1 by events on this card, where the HTTP figure moved by 11). The engine's constants hold on
  the card that scores; far 128 KB stays a ~10 us candidate for the BF16 engine only, not submitted.
- More on card 0 (s0020 engine 8458 / 8865): far 128 KB -10 / -8 again; the hub-own + late-V attention tweaks rebased
  onto s0020 (bench/variants/mk_hv20.cu) +10 / -9; no prefetch piece before thread 0's first poll (mk_pffirst.cu)
  +65 / +100; barrier back-off 0 or 100 ns: within 4 us.
- Far piece swept on card 0 (events, B=1 / B=8): 64/80 KB 8462 / 8858, 96 KB 8453 / 8851, 112 KB 8451 / 8851, 128 KB
  8448 / 8850, 160 KB 8454 / 8859. BF16 engine set to 128 KB (12-bit engine stays at 64 KB). Check on card 0: hash at
  B=1..8 unchanged, served 1024 tokens == greedy (8446 / 8865 us per step). s0022 submitted: expected ~8447 / ~8873.
- s0022 (track A, far piece 128 KB for the BF16 engine): predicted ~8447 / ~8873; VALID 8445 / 8878: 1.136x / 1.156x.
- s0023 (12-bit build of the s0022 source; behaviour of s0021): VALID 7668 / 8114 against s0021's 7665 / 8102: the
  evaluator's repeatability on an unchanged build is 2 us at B=1 and 12 us at B=8.
