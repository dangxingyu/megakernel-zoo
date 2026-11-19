# Run s11 (Qwen3.8-27B, one B200, track B on the s9 s0022 source): summary

Inherited: track B 7668 / 8114 us per token (HTTP, B=1 / B=8), track A 8445 / 8878.
Best on the board and final engine: s0029 7170 / 7652 (-6.5 % / -5.7 %), in process 7154. Before it: s0028 7175 / 7681,
s0027 7221 / 7698;
the earlier best, s0020
7238 / 7679, was measured at an earlier, lower evaluation level (F17: s0023 = s0020's configuration re-evaluated read
7261 / 7711). Every submission's logits are bit-identical to track A (bitcheck, 1152 positions, B=1 and B=8); the head
sketch picks exactly the exact head's greedy token.

## Submissions (all valid, all track B)
| sid | change | B=1 | B=8 |
|---|---|---|---|
| s0001 | W12 rows stored scaled by exact powers of two (one decode base; factor in the epilogue) | 7557 | 8003 |
| s0002 | layer 0's in_proj output from a per-token table (8.2 GB, made by the kernel's own in_proj) | 7542 | 7969 |
| s0003 | greedy passes: certified int8 sketch of lm_head + exact rescoring of candidate tiles | 7525 | 7950 |
| s0004 | the sketch head in its own kernel (main-kernel code generation untouched) | 7495 | 7936 |
| s0005 | 12-bit stream: next escape's quad kept in a register (no shuffle per 2 quads) | 7337 | 7763 |
| s0006 | epilogue per-row constants copied to shared memory at op start | 7327 | 7763 |
| s0007 | no BF16 twins of out_proj / o_proj; W12 at every batch size | 7304 | 7741 |
| s0008 | sketch tile counts per block from measured block speed (no gain beyond noise) | 7313 | 7738 |
| s0009 | 7-bit sketch (q + 64 in 7 of 8 bytes, exact digit-sum correction): 1.113 GB instead of 1.272 | 7291 | 7712 |
| s0010 | barrier L2 prefetch of the next GEMM's first piece 144 -> 176 KB | 7273 | 7683 |
| s0011 | head: argmax op folded into the finish op (one barrier less) | 7275 | 7697 |
| s0012 | head: exact rescoring chain relayed in 40 turns of 8 k-steps | 7266 | 7682 |
| s0013 | head: power-of-two x scale (digits without divisions) | 7263 | 7675 |
| s0014 | B<=3: attention barrier asks for 256 KB of o_proj + the K chunk | 7261 | 7674 |
| s0015 | B<=3: 20 K chunks per attention unit at that barrier | 7258 | 7688 |
| s0016 | B<=3 prefetch table retune: no K past chunk 5, whole o_proj share, 40 idle blocks | 7249 | 7682 |
| s0017 | head sketch: 93.9 % of tiles in 6 bits (rarely candidates on own text), the rest 7 (0.96 GB) | 7244 | 7684 |
| s0018 | B>1 passes keep an all-7-bit sketch copy through a plain kernel instance | 7239 | 7677 |
| s0019 | load-time tile calibration back on for the mixed sketch (reverted: in-process +31 us) | 7245 | 7680 |
| s0020 | B<=3 near prefetch per matrix: gate_up 160 KB, down 192 KB | 7238 | 7679 |
| s0021 | confirmation of the s0020 configuration with later experiments compiled out | 7242 | 7679 |
| s0022 | B=1: the rescoring's last block finishes the pass (SK_LAST; reverted, see F17) | 7259 | 7708 |
| s0023 | s0020 configuration again (evaluation level ~+20 / +30 us by then, F17) | 7261 | 7711 |
| s0024 | B=1 sketch: the 6 % hot tiles in 8 bits instead of 7 (candidates mean 33 -> 11) | 7252 | 7705 |
| s0025 | the mixed 6/8-bit sketch for every batch size | 7255 | 7707 |
| s0026 | head sketch prologue reads x as aligned u32 pairs instead of u16 scalars | 7250 | 7692 |
| s0027 | recurrent op re-reads B / spec flag from shared memory (also re-draws the hot loop, F24) | 7221 | 7698 |
| s0028 | the same source with ptxas --register-usage-level=10 (another register allocation) | 7175 | 7681 |
| s0029 | + ptxas -disable-optimizer-consts (same source) | 7170 | 7652 |

## What mattered
1. Removing per-tile work from the 12-bit stream (s0001 -106 us, s0005 -141..-178): the stream's cost was in its
   loads of decode bases and its escape test, not in its arithmetic.
2. The head on greedy passes (s0003, s0004, s0009, s0017 and the head-kernel items): 274 us of exact W12 head ->
   ~175 us at B=1 (7-bit sketch, then 6 bits for the 94 % of tiles that are rarely candidates and 8 bits for the rest).
   The sketch is certified (notes/head_sketch.md): bounds from integer tensor-core sums, candidates rescored with
   the exact head's own MMA chain, so the served tokens are the exact greedy tokens.
3. Prefetch sizes re-tuned for this build and for the scored workload's long contexts (s0010, s0014-s0016): the
   attention op is latency-bound and slows down when ~30 MB of weight prefetch runs beside it (F15); measured
   with nsys main-kernel medians at contexts to 1088 (NSTEP=1024), single card per job.
4. Late head-kernel gains (s0024-s0026): frequent-candidate tiles in 8 bits (candidates 33 -> 11 per pass), one mixed
   sketch for every B, and the prologue's x read as 4-byte pairs (head -3 / -9.5 us at B=1 / 8; F18: the prologue
   was load-instruction bound).
5. s0027 (-29 us over HTTP): the recurrent op re-reads two context values instead of spilling them; the same edit also
   re-scheduled the 12-bit hot loop (tools/hotloop_hash.sh), so the gain is measured but its mechanism is not
   established (F21, F24).

## What did not work (details in notes/journal.md, findings F8-F25)
- Extension (hours 19.7-21.2 of the agent clock): bit-identical escape regrouping inside cut-structure classes is worth
  -14.5 / -18.5 us at most and only -5 us without restructuring the norm's per-tile sums (F19; the old -80 us estimate
  was wrong in sign); code placement fillers 0..+5.6 us (F20); instruction-cache stalls sit at op entries, but warming
  them, keeping the loop's values in registers or moving the GEMM start-up out of line all cost +27..+150 us (F21);
  build-option draws within +-5 us, -rdc +1266 us (F22).
- After s0027: the same re-read idea in the attention op (+12..+42 us) and after the recurrence call (+61); thread-block
  clusters for the split recurrent unit's norm exchange (exact; +41 with a cluster barrier, net zero with remote
  stores, F25); the recurrent state prefetched one barrier earlier (+1.5).
- Why main-kernel work stopped paying (F22): the kernel is one ptxas unit, so any source edit re-schedules every
  subroutine, the 12-bit hot loop included (MAIN_REGS: 1108 of its 2342 SASS lines change). The kept build is the
  survivor of many draws; re-draws have measured +27, +44, +150 us.
- Late head experiments, all exact and measured: SK_DSR (warp-reduced atomics), SK_XU (unrolled loads, +6.5),
  SK_XS on SK_XV (+3.5), SK_ST31, SK_Z4 (-0.15), SK_W0X 12 / 24 (-0.3 / +10.5): none kept.
- Any code change in the 12-bit hot loop: +219 to +1640 us even when bit-identical and spill-free (F10).
- Load balancing of the sketch (static by measured speed: 0; dynamic units: +33..+45 us), in-sketch rescoring
  (+15), own-block rescoring (+4), L2 prefetch in the sketch (+38..+67), L2 persistence windows (+3..+336).
- Rounding-changing ideas were measured but not submitted: they re-draw the long-context gate's tail (F12;
  ~40 % fail chance per change from an own fp32 reference). The largest of them, a row permutation for escape
  locality, was first estimated at -80 us; measured later (F19) it is worth at most -14.5 / -18.5 us (the op waits for
  its slowest warp, which a plain sort makes worse), and its bit-identical form inside cut-structure classes gets the
  same, so it needs no gate risk but does need a norm restructure. The MLP-index permutation clusters nothing.
- Every 12-bit GEMM still carries ~5.5 us over its bytes (gap.md, end of run): ~1.3 ms per pass.

## How to build, check and measure
- engine/build.sh (engine/TRACK "A W12" = track B, "A" = track A). Flags of the track-B build are listed there.
- Before a track-B submission: tools/sub_check.sh (sketch exactness, bitcheck vs track A, served == offline).
- Same-job A/B of builds: tools/var_gen.sh (generate on real text); small changes: tools/head_nsys.sh (head
  kernel), tools/main_nsys.sh / tools/env_nsys.sh (main kernel; NSTEP=1024 for the scored contexts). Process-level
  shifts of up to ~20 us exist; repeat runs and compare medians.
- Own fp32 reference for gate questions: tools/own_ref64.py (+ tools/reroll_stats.py).

## Final state and checks
- engine/ = s0027: s0020's configuration + 8-bit hot tiles in a mixed 6/8-bit head sketch for every batch size + u32 x loads in
  the sketch prologue (head -3 / -9.5 us at B=1 / 8) + the recurrent op's context values re-read (LM_VOLC); experiment
  code behind flags that are off. At the evaluation's later level (F17) s0023 (= s0020 config) read 7261 / 7711 and
  s0024 / s0025 7252-7255 / 7705-7707, s0026 7250 / 7692, s0027 7221 / 7698.
- s0027 (final): sketch 0 / 21504 and 0 / 18432 (two prompt sets), served == greedy at B = 1, 3, 8, 20 x 128-step
  determinism runs bitwise identical at B=1 and 8.
- Extended (hour ~21.5): the sketch path matches the exact greedy token at 92160 positions (four prompt sets, 768 steps,
  B = 1, 2, 4, 8; 939 exact ties); re-verified on the replaced pod (bitcheck, served == offline, build_check).
- From-scratch build of both tracks; bitcheck track B vs track A bit-identical (1152 positions, B=1 and 8); the
  sketch path matches the exact greedy token at 21504 positions (B = 1, 2, 4, 6, 8); submit / poll == greedy over
  1024 tokens at B = 1, 3, 8.

## For a next run (what is open, with the numbers that decide it)
- Rounding-changing work (deferred escape corrections, GEMM rebalancing, attention re-chunking) re-draws the
  long-context gate (own-reference estimate ~40 % fail per change, F12); worth it only with a plan for the gate. The
  escape-locality permutation is not in this list any more: F19 measured it at -14.5 us at most.
- Edits inside an op's own function are mild draws (F24: one -29 us win, several +4..+61 losses); time every one with
  3 nsys rounds at B=1, B=8 and NSTEP=1024. Stamped builds cannot resolve such effects (gap.md, end).
- The 12-bit hot loop: any code change measured +200..+1600 us here (F10); a new loop would have to be designed and
  tuned from scratch against that sensitivity.
- Head: ~175.5 us at B=1 (B=8 ~187) against ~133 us of bytes; the remaining overhead is block imbalance (~14: warp
  median 139 vs last block 154), rescoring (~17) and the prologue (4.9; B=8 10.8). Fixes tried: SK_DYN, SK_SPEC,
  SK_OWN, SCAN1-3, SK_LAST, PF2, PFW, W0X, DSR, XU, XS, ST31, Z4. What I would try next there: the B=8 prologue with
  one 16-byte (or TMA) copy of x into shared memory per block plus digits from shared memory (needs a larger hk_smem
  and a free 80 KB region; B=8 only), and rescoring overlapped with the sketch's tail (blocks whose tiles are done).
- notes/next_run.md is the one-page hand-off. What I would try first in a next run: make the 12-bit stream's code independent of the rest of the main kernel
  (its own kernel instance whose source never changes, or a frozen cubin, checked by a SASS hash in the build; not
  -rdc, F22), so that the remaining fixed costs (op entries, barriers, the recurrent op's cold start, F21) can be
  worked on without re-drawing the hot loop. The escape-locality permutation is worth far less than estimated (F19).
- Measure small changes with tools/main_nsys.sh / env_nsys.sh (NSTEP=1024) on one card per job; tools/sub_check.sh
  and tools/build_check.sh before every submission.
