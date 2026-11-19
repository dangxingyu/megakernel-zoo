# Findings of run s11 (track B on the s9 s0022 source). Earlier runs' findings: notes/prev/

Each entry: what was measured, on what, the number, what it allows and what it does not.

### F1. Absolute timings drift between jobs; only alternating same-job comparisons hold (tools/gen_ab.sh, v_time.sh)
- The same s0005 source measured 7483 us (as "previous") against 7393 ("current", build.sh) in one job, and 7337 /
  7337 in a later job on the same card; another job read 7393 for the current build where an earlier and a later
  job read 7336. A whole job's level can sit ~50-90 us off. Every comparison here alternates builds inside one job.
- Side finding while chasing it: `-Xptxas -v` (used by engine/build.sh) changes the generated SASS (md5 aef763899042
  vs 4fd754bff1a3, deterministic), but not the timing (generate 7336 / 7336, tv.py 7406 / 7408). The A/B tools now
  build with -v anyway, exactly as build.sh.

### F2. Exact row scaling removes the per-tile decode-base loads of the 12-bit format (s0001)
- W[r] stored as W[r] 2^(107 - lo_r) (exact: powers of two, all weights normal, no zeros in the matrices), one decode
  base, the row sum multiplied by 2^(lo_r - 107) in the epilogue. Codes and escape counts unchanged.
- Ablation first: constant bases (wrong results) -144 us per B=1 pass. Built: -106 / -108 us (B=1 / B=8), HTTP
  7668 -> 7557. Verified on all 25,621,954,560 stored weights (bits and values) and bitcheck (1152 positions).
- Trap: torch.pow(2.0, -9) on CUDA float64 is not exact (0.0019531249999999998); use torch.ldexp for scale tables.

### F3. Layer 0's in_proj depends on the token only (s0002)
- 248,320 x 16,480 bf16 table (8.2 GB) made by the kernel's own in_proj op at load; all rows equal to one-column passes.
- -15 us / -27 us (B=1 / B=8) for removing a 24 us op: the embed op's 32 KB table copy costs part of it.
- Does not extend further: layer 0's convolution mixes 4 tokens; every later input depends on the context.

### F4. A certified int8 sketch of lm_head selects the greedy token exactly (s0003, s0004)
- Bound and proof: notes/head_sketch.md. Candidates on real text (tools/head_census.py, engine's exact rule):
  2.3 tiles of 16 rows on average (p99 16, max 67) out of 15,520. int7 per 128-group 3.3 (p99 31); int6 per 64-group
  13.2 (p99 212, max 2965); int5/int4 useless (Cauchy-Schwarz bound too wide).
- Head cost on greedy passes 274 us (exact W12) -> ~216 us (sketch ~197 incl. imbalance, rescore ~15). Exact on
  9216/9216 teacher-forced positions incl. 105 bf16 ties, and with every tile forced to be a candidate.
- Compiled into the main kernel the two new op cases cost +28 us per pass (code generation; F1 may have been part
  of it); in their own kernel (launched after the main one, gap 0.26 us) the main kernel is unchanged.

### F5. The escape test, not the escape patches, was most of the 12-bit loop's escape cost (s0005)
- On the RS build: no escape code -280 us B=1; lists empty but loaded -106; lists L2-hot -20. Keeping the next
  escape's quad in a register (no shuffle per 2-quad iteration): -141 / -178, HTTP 7495 -> 7337.
- Remaining escape cost (patches of ~7 % of quads): ~114 us B=1 (lists emptied on the NQREG build).

### F6. Where a B=1 pass's warps wait (ncu SourceCounters, one pass)
- ~55 % of stall samples in the 12-bit loop waiting for weights, ~15 % in the grid barrier (10.5 % parked after
  the barrier's block sync, 4.3 % thread 32's prefetch loop), 4.5 % at the block sync after the stream (stragglers).
- Per GEMM (tools/tail_b.py): stream end -> barrier arrival ~2 us (finalize + epilogue), slowest block +1.2 us,
  last arrival -> release seen ~1.1 us. out_proj at B=1: 8.2 us of stream, 4.3 us after it.

### F7. Things this checkpoint does not offer
- Exact zeros in GEMM inputs: 0.05-0.08 % of elements, no all-zero 16-wide k-step (tools/zero_census.py).
- Per-column power-of-two shifts on top of per-row windows: escapes -6 % only (tools/w12v2_census.py): the escapes
  are the Gaussian tail of tiny weights below a 16-binade window.

### F8. The sketch's per-block speed is systematic, but balancing it does not show over HTTP (s0008)
- Per-block sketch stream end (24 passes): spread max - median 11.5 us per pass; per-block lag correlated 0.93
  between the first and second half of the passes. A load-time calibration (tiles proportional to measured speed,
  94-106 tiles per block) gained -5 / -4 us in process, 7313 / 7738 over HTTP (s0007 7304 / 7741): noise.
- Within a block warp 0 ends ~40 us before the others (115 vs 154 us median): the barrier's 144 KB L2 prefetch
  covers ~63 % of its share.

### F9. A 7-bit per-row sketch certifies the greedy token at 1.113 GB (s0009)
- Candidates (head_census, 3072 real positions, the engine's bound): int8 2.3 tiles mean (p99 16), int7 6.9 (p99 78,
  max 821); int6 per row 264 (p99 4667), per 1024 53, per 256 26 (p99 417): int6 is not usable at any group size
  that saves bytes.
- 7 of every 8 bytes, unsigned (q + 64) x signed digits on the integer tensor cores, exact correction by the digit sums.
  -18..-21 us B=1 in process, s0009 7291 / 7712 over HTTP (s0008 7313 / 7738).

### F10. The 12-bit hot loop does not tolerate structural changes (escape slow path)
- No escape code at all (wrong results): -178 / -180 us (B=1 / B=8): the escape mechanism's whole cost.
- Two bit-identical rewrites of the one-escape case cost +375 us (a second copy of the 8-k-step body) and +1640 us
  (a per-k-step uniform test in the plain body), with no or 8 B of spills: code size / placement of the loop matters
  more than its instruction count. Any change there has to keep the loop's code as it is.

### F11. L2 persistence does not pay on this pass
- A persisting window (host launch attribute) on 32 / 64 MB of the sketch: +3 / +93 us B=1, +12 / +336 B=8: the
  carve-out takes L2 from the main kernel's weight prefetch and activations.

### F12. Why every change here stays bit-identical: the long-context gate's tail re-rolls (own fp32 reference)
- tools/own_ref.py: own prompts (seeds 70000+, not the evaluator's data), 8 x 1152 tokens teacher forced (B=8), every
  8th position against a float32 Hugging Face run. Median rel-L2: HF BF16 0.0098, engine 0.0099; p99 0.595 / 0.603;
  max 0.773 / 0.720: the engine is as close as the HF BF16 path; the large values sit in a few sequences.
- A pure rounding change (PERTURB_REV: reversed in-block order of cut-tile pieces; identical in exact arithmetic):
  per-sequence maxima move 0.356 -> 0.140, 0.231 -> 0.265, 0.720 -> 0.771 (the rest ~0.05 both).
  The evaluator's numeric_long max (s0008 0.457, limit 0.592; s5's rounding change gave 0.713) is such a tail value:
  any change of rounding is a fresh draw with a real chance (~1/3) of failing. So: row permutations for escape
  locality (est. -80 us, F10 census) and deferred escape corrections are not taken.
  (Correction, F19: the -80 us estimate was wrong; the permutation is worth at most -14.5 / -18.5 us and has a
  bit-identical form.)

### F13. Measuring changes of a few microseconds (tools/head_nsys.sh, tools/main_nsys.sh, tools/env_nsys.sh)
- generate() timings of separate processes scatter by +-3 us, sometimes in two levels ~25 us apart; cards differ by
  ~16 us in absolute level. The nsys median of the head kernel (324 launches) or of the main kernel (387 launches)
  repeats within ~0.6 us: head-kernel changes of 1-3 us (SK_FOLD -2, SK_KPT=8 -3.4, SK_P2X -1.2) were only visible
  this way; over HTTP they are inside the noise (s0011-s0013).
- With it, every prefetch knob of the engine is at its optimum within ~0.5 us (journal, hour ~9.8).

### F14. The head kernel is register- and bandwidth-bound, not latency-bound
- Anything that adds requests or live registers to the sketch stream loses: L2 prefetch of the next units +38..+67 us,
  dynamic units +33..+45 (per-unit atomics and fences), the next entry's x in registers +2..+5 (spills 96 -> 196 B).
  The 7-bit stream runs at ~7.2 TB/s aggregate; the remaining head overhead (~36 us over its 152 us of bytes) is
  spread over prologue (4.4), stream imbalance (~14), rescoring (~15) and barriers.
- (F12, continued) Conditional re-roll rates from the own 64-sequence reference (tools/reroll_stats.py; PERTURB_REV and
  REV2 against the current engine): a position whose error is 0.1-0.2 / 0.2-0.3 / 0.3-0.4 / 0.4-0.5 / 0.5-0.6 lands
  above 0.592 after a re-roll with probability 1.3 / 6.4 / 6.7 / 13.6 / 23.4 %; re-rolled / current error has median
  0.92, p90 1.35, p99 4.0. For the evaluator's long check (our max 0.457, p99 0.098) that is a pass chance of roughly
  60 % per rounding change. The cheapest row-permutation variant (gate_up rows + down columns per layer, weights
  only, no kernel code) would gain ~38 us: expected value ~25 us at a ~40 % risk of an invalid result. Not taken.

### F15. The attention op is sensitive to prefetch traffic around it (s0014, s0015)
- At the scored contexts (to 1088), the barrier before the one-op attention layer was asking L2 for the whole o_proj
  share of ~132 idle blocks (~33 MB) while the attention op's dependent round trips ran. Fewer o_proj bytes there
  (256 KB per block, and 60 blocks asking for 128 KB K chunks instead) made the attention op ~0.35 us faster per
  layer and the pass -3 / -7 us (nsys, three cards); asking for too little of o_proj (16 KB per block, or no block)
  costs more in o_proj (+14..+18). Measured at short contexts (<= 364) the effect is ~-1.5 us, which is why the
  earlier sweeps (contexts <= 364) did not show it.

### F16. Most of the vocabulary never comes near the top logit: a mixed 6 / 7-bit sketch (s0017, s0018)
- tools/mixed_census.py: tiles that are candidates under a 6-bit sketch at <= 80 of 6144 own greedy positions: 93.9 %
  (weight-only selections - row norms, max |w| - are far worse: p99 candidates 400-600). 5 bits: no tile qualifies.
- The sketch's warps cost about the same per unit for 6 or 7 bits (balance by tile count, not bytes); the extra
  candidates (median 3, mean ~60 with rare passes of thousands) cost the rescoring's scan and chains part of the
  byte gain: head 188 -> ~180 us at B=1. At B=8 the candidates of 8 columns cost more than the bytes save, so B>1
  passes keep an all-7-bit copy (a separately compiled kernel instance: the mixed code path slows 7-bit tiles too).
- A load-time calibration of tiles per block helped in three nsys processes (-2.6 us) but the scored process measured
  +31 us in process (s0019): per-process load-time measurements are a risk for the one scored process.

### F17. The evaluation's own level drifts by tens of microseconds
- The same configuration was evaluated three times: 7238 / 7679 (s0020), 7242 / 7679 (s0021), 7261 / 7711 (s0023,
  ~1.4 agent-hours later; pod unchanged). Differences of a few microseconds between submissions are not resolvable
  on the board; only same-job in-process A/B (nsys medians, findings F13) resolves them.

### F18. The head sketch's prologue was load-instruction bound, not latency or atomic bound (s0026)
- Every one of the 152 head blocks reads the whole x (B=8: 80 KB) twice from L2 in the B-fragment layout. SK_PROF
  phases at B=8: x loaded + reduced 6.1-6.7 us, digits 13.1-13.6 us. Reading the (k, k+1) pairs as aligned 4-byte
  words instead of 2-byte scalars (same values; both layouts keep the pair adjacent): B=8 prologue -2.5 us, head
  kernel -9.5 us at B=8 and -3 us at B=1 (3 nsys rounds), s0026 7250 / 7692 over HTTP (s0025 7255 / 7707).
- The other prologue changes measured nothing or worse: warp-reduced digit-sum atomics (SK_DSR: equal), loads
  unrolled to issue together (SK_XU: B=8 prologue 13.3 -> 20 us), shared-memory staging of x (SK_XS: +3.5 / +4 us
  with 4-byte loads), column stats on an idle warp (SK_ST31: equal).
- Allows: narrowing the instruction count of redundant per-block reads pays even where the bytes are L2 hits. Does not
  show: that the main kernel's x reads would gain the same (its code generation is locked, F10).

### F19. Bit-identical escape regrouping: real but small, and mostly blocked by the norm's per-tile sums (hour ~20.5)
- A row's GEMM result depends only on the cut structure of its tile under the static partition (piece offsets and
  block boundaries; tools/rowperm.py (copy to engine/ with tools/rowperm_api.patch applied to engine/api.py)). Tiles with equal structure form classes (in_proj 154 classes of 2-44 tiles,
  qkv 25 of 8-112, down / out_proj / o_proj 23 of 8-40); rows may move inside a class without any rounding change.
- tools/cls_perm_census.py (all 192 non-gate_up matrices, exact escape masks): sorting rows by escape count cuts escape
  quads 7.61 % -> 5.87 % (global sort 3.02 %) but RAISES the busiest warp's slow quads 5.8 -> 10.5 (the op waits for
  its slowest warp): the old "-80 us" estimate (journal hour ~6) was wrong in sign. Best fit inside classes (balance
  the per-piece union) gives 6.85 % and per-op max 1118 -> 632 summed over the ops.
- Timing with rows regrouped and outputs not mapped back (MK_ROWPERM=fit, wrong values, real work; 3 nsys rounds):
  main kernel B=1 7036 -> 7021 (-14.5 us), B=8 7464 -> 7445 (-18.5); in_proj + qkv alone (EP_STORE) -5 us at B=1.
- Blocked: down / out_proj / o_proj end in the residual epilogue, which stores the sum of squares of each tile's 16
  rows for the next RMSNorm (cx.ssq[T], summed in tile order): moving rows between tiles regroups that sum (a
  rounding change). Only in_proj / qkv (-5 us) are free, and they need an output-row map in the main kernel's
  epilogue (code generation risk +11..+59 us, F10). Not built. gate_up moves only (gate_j, up_j) pairs (F12 notes:
  total clustering ~nil); not pursued.

### F20. Code placement of the main kernel is a small effect, and the current placement is among the best (hour ~21)
- A never-called subroutine of N one-instruction fillers (-DPADN=N, called under nops < 0) shifts the code after it
  by 16 N bytes. Main-kernel nsys medians, 2 rounds, each card against its own base (cards differ by ~50 us in level):
  N = 0: +0.5; 1: +4.4; 2: +5.6; 3: +5.0; 4: -0.5; 5: -0.7; 6: +0.3; 7: -2 (rounds +1.4 / -5.0, card 3 noisier);
  16: +5.4 us. Placement moves the pass by up to ~5 us; no shift is better than the current code by more than noise.
- So F10's +200..+1600 us for bit-identical rewrites of the 12-bit loop are not placement: they are register
  allocation / scheduling. Allows: a code change elsewhere in the kernel can cost ~5 us through placement alone,
  independent of what it does. Closed.

### F21. Instruction-cache stalls are at op entries; the main loop's code generation blocks every fix (hour ~22)
- ncu warp-state sampling of one B=1 main-kernel pass (tools/ncu_ws.sh, tools/ncu_fn.py): "no instruction" stalls are
  1.9 % of all warp samples, concentrated in the small ops (recurrent op lmix_work 18.8 % of its samples, lm_core 16.8,
  finalize 16.5, r_partials4 12.7, epilogue_k 11.4; the 12-bit stream 1.3 %). The top lines are function entries (op
  dispatch after the barrier, the stream's entry, lmix_work's first line). LMIX_PROF with two extra stamps: ~1.0 us of
  the recurrent op's 7.7 us (per layer, B=1) is spent on a stretch of arithmetic and shared-memory reads right after
  its entry: cold code.
- Warming that stretch at the barrier before the op (warp 2 calls lmix_work in a warm mode, IC_WARM): main kernel
  +149 us at B=1 and +156 at B=8, where the warm path never runs: code generation of the main loop, not a measurement
  of warming. Not testable in this form.
- The main loop (mk_body with the barrier and gemm_phase inlined) spills 48 B. Variants: MAIN_REGS 56 / 60 B, +44 / +91
  us (B=1 / B=8); IC_WARM 76 / 84 B, +149 / +156; GEMM start-up moved into a noinline function (GP_NOINL) 16 / 12 B but
  +27.5 (B=1, 3 rounds), +18 (NSTEP=1024), +9..+41 (B=8). Spill bytes do not predict the time (the call's register
  save / restore and the streams' later start are not counted as spills). Every change to the main loop so far has
  cost tens to hundreds of microseconds; none gained.

### F22. Why the main kernel is so fragile: ptxas re-schedules the whole kernel, hot loop included (hour ~22.5)
- tools/sass_diff.sh + tools/sass_fn_diff.py (nvdisasm of the s0026 flags vs the same + one flag, per subroutine,
  addresses and labels normalised): with -DMAIN_REGS, which changes only the top-level loop's source, every subroutine
  of mk_kernel compiles differently, including untouched ones (finish_work, finalize, epilogue_k: same counts, other
  instruction choices and order), and the 12-bit hot loop gemm_stream_w12e<1,4> changes in 1108 of its 2342 SASS
  lines (2 instructions fewer). The kernel is one ptxas compilation unit: any edit anywhere re-draws the hot loop's
  schedule. This is the mechanism behind F10 (+219..+1640 us for bit-identical rewrites), F20 (placement fillers 0..+5.6)
  and F21 (MAIN_REGS +44, IC_WARM +150, GP_NOINL +27.5). The current build is the survivor of many such draws (every
  kept change measured better), which is why almost every change now measures worse.
- Caveat: the neutral filler build (PADN=4, 5 top-level lines changed, -0.5 us) also re-emits ~470 of the hot loop's
  2342 lines with the same instruction count; changed-line counts (with or without register renaming,
  tools/sass_loop_cmp.py) do not predict the time. Only timing does.
- Allows: a next run should first make the hot loop's code independent of the rest, then tune the rest. Without that,
  main-kernel work is a lottery with mostly losing tickets. Separate compilation is NOT that fix: the whole file built
  with -rdc=true (-maxrregcount=64, needed because r_partials4 wants 68 registers under the ABI) makes the 12-bit loop
  spill 148 B (0 in the whole-program build, where ptxas allocates registers across calls) and the main kernel
  7036 -> 8303 us, the head 175.6 -> 213.3. Candidates: the stream as its own kernel instance whose source never
  changes, or a frozen cubin for it, each checked by a SASS hash in the build.

### F23. The head sketch is at its byte floor: a provably tighter group-wise bound does not open 5 bits (hour ~21.8)
- The sketch's error term uses Cauchy-Schwarz with each row's exact error norm (||e_i|| ||x||). A fixed partition of
  the channels into groups gives |e_i . x| <= sum_g ||e_i,g|| ||x_g|| <= ||e_i|| ||x|| (never looser).
- tools/block_census.py (4608 own greedy positions; channels ranked by mean x^2 on half A, groups 16 / 48 / 192 / rest,
  counted on half B): the head input's energy is only moderately concentrated (top 16 / 64 / 256 channels: 13.6 /
  22.4 / 32.4 % of ||x||^2). Candidate tiles per position, engine formats (6-bit set + 8-bit hot): plain mean 9.4,
  p99 151 -> grouped 4.5, p99 57. With 5-bit cold tiles: plain median 1554, grouped median 194, mean 1033: far too many.
- So the cold tiles cannot go below 6 bits with any bound of this kind, and halving the candidates at 6 bits is worth
  at most the rescoring's share of them (~1-2 us). Not built.

### F24. One callee edit gained 29 us; similar ones lost: the ops' functions are still a draw, only a milder one
- A never-taken block inside lmix_work: top-level SASS identical, hot loop 28 of 2342 lines changed, B=1 -0.4 us,
  B=8 +3 (callee edits are much milder than top-level edits, F22).
- LM_VOLC (lmix_work re-reads B and the spec flag from shared memory instead of keeping them live; ptxas had spilled
  them at entry and reloaded them before the unit was chosen): spill loads 104 -> 52 B; main kernel -31 us at B=1,
  -38 at the scored contexts, +10 at B=8; s0027 7221 / 7698 over HTTP (s0026 7250 / 7692).
- The same idea elsewhere lost: the attention op's pointers re-read per use (ATT_VOLC, 172 -> 144 B) +42 us; only its
  entry prefetch addresses (ATT_PFV) +12; lmix_work's values after the lm_core call (LM_VOLC2, 52 -> 12 B) +61. The
  attribution of LM_VOLC's gain to the two entry reloads is plausible (it removed exactly the ~1 us stretch of F21)
  but not proven: spill counts do not predict time even inside a callee. Every such edit has to be timed (3 rounds,
  B=1 and B=8, NSTEP=1024) before it is kept.
- Tally after a bounded search with a fixed acceptance rule (epilogue_k, finalize, lm_core, lmix_work variants): one
  win, fifteen losers (+1..+61 us at B=1). The distribution of such draws is mostly worse than the kept build.

### F25. Thread-block clusters for the split recurrent unit: feasible, exact, and net zero (hour ~23)
- Persistent launch with 2-block clusters works (76 resident clusters = all 152 blocks; bench/cluster_probe.cu).
- Main kernel, B=1: cluster launch alone +6 us (block placement); norm sums exchanged over DSMEM after a cluster
  barrier +41 in total (the barrier's release waits for the 32 KB state write-back just before); a relaxed remote
  store into the partner's shared memory with a local sentinel poll +6 / +0 at the scored contexts: the exchange
  becomes ~6 us cheaper per pass, which only pays for the launch. The global sentinel exchange stays.

### F26. The serving path costs nothing: HTTP equals in-process generate() on a job card (hour ~24)
- The harness's own HTTP stage run on card 1 (tools/http_local.sh: $MK_HARNESS/eval/http_stage.py without the token
  check, B=1, 1 warm-up + 2 repeats; server and client pinned as in the evaluation): 7193.8 us per token, tpot p50
  7.195 ms, p99 7.45 ms, TTFT 88 ms. generate() on the same workload in process, same card (tools/w1_gen.py: prompt 64,
  1024 greedy tokens): 7193.7-7195.7 us. B=8 the same way: HTTP 7697.3 us per step (min / max 7696.6 / 7698.1),
  generate() 7690.6-7697.6 (tools/w8_gen.py).
- The evaluation's own gap for s0027 (HTTP 7221 vs in-process 7202 on card 0) is therefore not the engine's submit /
  poll path, the server or the client: it is the card or the stage (the HTTP stage runs last, after the timed and traced
  stages). Nothing to gain on the engine side there.

### F27. Exact column power-of-two scaling of the 12-bit matrices: bit-identical, ~-3 us (hour ~24.5)
- Scaling input channel k of a matrix by 2^c_k and its input by 2^-c_k changes no product (power-of-two scaling commutes
  with every rounding; the checkpoint has no subnormal weights: tools/colscale_census.py). The input scale folds into
  host tables without kernel code: the input / post-attention norm gains, the up rows (so the SwiGLU output carries
  2^-c for down's columns), the gated norm's weight (out_proj, one c per head dimension).
- Census (7 layers, c in [-6, 6] by alternating optimisation): escapes -6 %, escape quads 7.41 % -> 6.97 %: the escapes are
  sparse outliers, not column-structured. Timing with all matrices shifted and inputs not compensated (wrong values):
  main kernel B=1 -10.6 us, B=8 -6.5.
- The exact version (tools/colscale_api.patch, MK_COLSCALE=1; layer 0's in_proj, o_proj and the head unshifted):
  bitcheck bit-identical to track A at B=1 and 8, sketch exact, served == offline; engine escapes 1922136 -> 1851153
  (-3.7 %, head included); main kernel B=1 -4.4 / +3.3 / -5.8 us (3 rounds), B=8 -1.8; load +18 s. Not kept: below
  anything the evaluation resolves and mixed in sign.
- (Correction, hour ~24.5, tools/hotloop_hash.sh) LM_VOLC also changed the 12-bit hot loop's code: without it the loop
  is 2342 instructions (sha1 9582e515...), with it 2345 (3437373a...). s0027's -29 us is therefore not attributable to
  the removed reloads alone; part or all of it may be a favourable re-schedule of the hot loop. The gain is real
  (in process and over HTTP); its mechanism is not established. A head-kernel-only edit (SK_DSR) leaves the hash
  unchanged; MAIN_REGS changes it (2346, b4716919...).
- Draw distribution around s0027 (tools/draw_n.sh: a never-taken block of N statements in lmix_work, N = 1, 2, 3, 4, 6,
  8, 12, 16, 24; 1 round each against s0027 on the same card): all nine slower, +4.8..+33 us at B=1. Their hot loops
  have 5 distinct hashes, none equal to s0027's (2346 instructions vs 2345), and draws with the same hot-loop hash
  (N = 1, 6, 8) differ by 11.5 / 23.5 / 33 us: the rest of the kernel's code matters too, so the hash flags a re-draw
  but does not predict its cost. s0027 is better than every sampled neighbour.

### F28. ptxas --register-usage-level is a strong, semantics-preserving knob for this kernel (s0028)
- Same source (s0027), only the ptxas option (main kernel nsys medians against the default on the same card):
  level 0 / 3 +160 us, 5 = default, 7 / 8 / 10 -42 at B=1 (3 rounds), -38..-51 at the scored contexts, B=8 -21..-23 for
  10 (7: -23 / +8). Bitcheck identical, sketch exact, determinism runs identical. s0028 (level 10): 7175 / 7681 over HTTP
  (s0027 7221 / 7698), in process 7161.
- It interacts with source changes: level 10 without LM_VOLC is +55 against level 10 with it, so both are kept.
  Combined with other options it gets worse (NVVM -O2 +18, allow-expensive-optimizations +15).
- On top of level 10, -disable-optimizer-consts: -7 us at B=1 in process; s0029 7170 / 7652 over HTTP (-5 / -29).
- Allows: a cheap, exact sweep that a next run should repeat after every source change (levels 5..10, B=1, NSTEP=1024
  and B=8), together with the hot-loop hash (tools/hotloop_hash.sh).
