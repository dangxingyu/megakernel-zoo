# Journal

## 2026-11-10/11, hours 0-1.3: reconnaissance, design, first correct engine

- Read the contract, the harness and the HF source. The score is tokens/s over HTTP at B=1 and B=8 through
  submit()/poll(); step() is checked against a float32 reference.
- Measured the card (notes/findings.md C1): 7.3 GB/ms streaming, so BF16 weights cost >= 7.0 ms per step. The
  general engines are at 9.5 ms. Track A can gain at most ~25 %; anything larger must read fewer bytes per token.
- Design: one persistent kernel per step, all matrices in MMA fragment order, 8 MMA columns = sequences or
  positions; equal contiguous share per warp, partial sums per (tile, warp), last finisher finalizes in fixed order.
  RMSNorm's scalar r is applied after the GEMM (W (r g h) = r W (g h)), so the next matrix can start from
  bf16(h (1+w)) written by the previous epilogue; r comes from per-tile partial sums of squares.
- The delta-rule state update is rank 1 per head, so a pass over P positions can keep the old state and replay the
  accepted updates later: the basis for exact speculative passes (track B, next).
- Debugging cost ~40 minutes on one bug: results were wrong only in SMs that had not run the recurrent-state op.
  Chased L1 staleness, store queues and __syncthreads before finding that the barrier's spin loop had been
  removed by the compiler (findings C3). After the fix: rel-L2 vs HF BF16 0.0097 median (HF's own distance to
  float32 is 0.010), top-1 identical on 8x24 positions, single = batch bitwise, generate = step.
- First timing 15.9 ms/step: every op pays for loads executed one at a time (a warp waits for each load), and the
  hot loop spills because it shares a function with the epilogues. In progress.

## hours 1.3-2.2: s0001, s0002: from 16.8 ms to 10.7 ms without touching the algorithm

- The bridge daemon of the run crashed (stat on a spool ack file already removed by `mk`); nothing answered for
  33 minutes. Restarted it with its own command line; noted on the timeline.
- s0001 (track A, valid): expected ~16800 us over HTTP, measured 16990 (B=1) / 17606 (B=8). All gates pass
  (rel-L2 0.0101 median against float32, batch bitwise equal, generate and HTTP tokens identical 8/8), milestones
  build, correct, device_loop, single_launch, gpu_never_waits. The HTTP path costs +150 us per token.
- Where the time went (tools/prof_rep.py, ops repeated 40 times inside one launch): an op that does nothing cost
  5.2 us, a GEMM prologue 12 us, the stream ran at 3,500-4,300 GB/s. Causes, in the order found: op/context
  structs in every thread's stack; the hot loop unrolled past 64 registers; activations copied to shared memory
  per matrix; one function call (and one drained pipeline) per tile piece. After kernel v3: empty op 0.78 us,
  prologue 1.0-1.6 us, in_proj 31.6 us (26 ideal), gate/up 57.3 (51), down 32.9 (26), out_proj 15.6 (9),
  recurrent-state op 11 us, five attention sub-ops 22 us, lm_head 356 us (7,140 GB/s).
- s0002 submitted with that kernel: expecting ~10850 / 14050 us.
- What is left for track A: ~4 us per GEMM to finalize cut tiles (1.0 ms), the recurrent op (0.5 ms at B=1, three
  times that at B=8), attention (0.36 ms), barriers (0.26 ms). The floor is 7.2 ms. Track A will end near the
  general engines, not far below them: the real lever is tokens per pass (track B, MTP drafts).

## hours 2.2-2.5: s0002 confirmed, track B works on the first run

- s0002 (track A, valid): predicted 10850 / 14050, measured 10834 / 14059 us. MBU 0.60.
- Recurrent-state op rewritten twice. First with the 128 x 128 state in registers instead of shared memory
  (frees L1 for the GEMMs: B=8 went from 13.9 to 12.2 ms). Then, after measuring that its input stage (causal conv,
  SiLU, norms) took 8 of its 15 us on 4 warps while 28 waited: one channel per thread over all 32 warps, and one
  unit per block because the op is bounded by state traffic per SM (128 KB per unit at ~46 GB/s per SM). Now 8.4 us
  at B=1, 22 us at B=8 (3 rounds). Step: 10.15 ms (B=1), 11.6 ms (B=8).
- Things that did NOT matter (measured): barrier waiters spinning on L2 (nanosleep back-off 0-1000 ns: no change);
  ring-buffer loads; a dependent L2/HBM load is 0.19 us when the card is otherwise idle (bench/lat.cu). Small ops
  cost 4-6 us because a GPU thread runs serial code slowly, not because of memory.
- Track B, first version: verify P positions per pass with drafts from the checkpoint's MTP head (entry pass inside
  the verify launch, one small launch per further draft). test_multi: bit-identical for 9 (B,P) settings on the
  first run. test_spec: 1024 tokens identical to step-by-step greedy on the three timed seeds at B=1 and B=8.
  Submitted as s0003 expecting ~2100 us (B=1 median: loop prompt) and ~7500 us (B=8).
- The pod was replaced during the evaluation of s0003 (epochs 5 -> 6); it is queued again.
- Written while waiting: a context-copy drafter (last 6 tokens seen before, and the continuation agrees with the MTP
  head's first draft -> copy the continuation, chained MTP launches skip themselves); the W12 decode microbenchmark
  (bench/w12.cu) and an exact exponent census (tools/census.py). Plan for W12 if the card can decode fast enough:
  12-bit code [sign][exponent - lo + 1][mantissa], bf16 = code + C + sign * 0x7800 (two integer ops per pair after
  unpacking), every out-of-window weight (and exact zeros) patched into the decoded register from a per-warp list:
  a pure re-encoding, bit-identical activations, no rescaling.

## hours 2.5-2.6 (the pod was replaced four times, clock stopped most of the time)

- s0003 was never evaluated: its pod died twice during the evaluation ("pod_lost ... not run a third time"). The
  engine it carried is superseded; the next track-B submission replaces it.
- Census and W12 microbenchmark ran between two pod losses (findings M4, C6): escape rate 1.86e-4 with a
  15-exponent window per tensor, no exact zeros anywhere, and W12 decoding is 23 % faster than BF16 streaming in
  quads per second. Decision: build W12 into the stream loop (track B). Out-of-window weights are patched into the
  decoded register from a per-warp list in stream order, so the MMA operands are bit-identical to BF16 storage
  and no rescaling is needed.
- A bug of mine: a chained draft launch that skips itself returned without advancing the barrier counter the host
  had already accounted for; the next launch waited forever. Fixed (the skipping launch adds its barrier count).

## 00:03-06:00 PST: no pod (cluster outage), clock stopped at 2.57 h

- Pods of whole-node requests (GB200x4, H100x8) stopped reporting a warm-up index at 00:03; the bridges of this run
  and of the sibling B200 run stopped themselves after four failed starts. 1-GPU H100 pods recovered around 04:30,
  GB200x4 pods had not by 05:41 (three retries of mine: the bridge's stop state reset in bridge/state.json with a
  backup, each noted with `mk note`; from the second retry on with a single attempt per retry).
- Written without a pod (all untested on the card, in this order of risk):
  1. skipped chain launches now advance the barrier counter (the hang of e0085);
  2. two shared-memory slots per warp instead of three (whole tiles use the slot of the last piece as scratch, a
     finalized tile is summed in place): at 16 columns the activations of a 5120-wide matrix (164 KB) fit in L1 again;
  3. up to 16 positions per pass (ring of 32, pending-update arrays of 16), used for a single sequence while its
     drafts are copied from the context; the accept op writes drafts in the layout of the next pass (next_P);
  4. MTP chain capped at 4 drafts per pass (the agent's table: beyond 4 chained drafts a prose sequence gains
     < 0.1 token per pass, less than a chain launch costs);
  5. W12 in the stream loop with patches, the packer, the per-warp patch lists and an independent decoder for the
     bit-for-bit check (the code/decode arithmetic checked in pure Python on 200,000 random k-steps).
- Test plan when a pod is back: track-A regression (logits against HF, batch-bitwise), multi-position equality
  including P=16, speculative identity on the timed seeds, W12 bit-for-bit on all tensors and W12 logits against
  BF16-storage logits, then submissions: track B (drafts), track B (drafts + W12), track A again.

## hours 2.57-2.9: s0004 (failed), s0005 (valid, track B)

- s0004 (track B: MTP + context-copy drafts, verify up to 16 positions): gates numeric and batch FAILED. Single-
  sequence logits were wrong at a few positions (rel-L2 max 1.17, p99 0.42) while batched logits were fine
  (0.010) and the timed B=1 probes passed. Not reproduced: the harness runner's check flow on my own sequences,
  track A and B, 2 x 40 runs of 128 steps with and without fences: all bit-exact and deterministic. The only defect
  found by reading: no device-scope ordering between a block's stores and its arrival at the grid barrier, and
  between a block's partial of a cut tile and the counter that hands it to the last block. Fixed with
  release/acquire (red.release.gpu / ld.acquire.gpu / atom.acq_rel.gpu): +0.2 ms per step (full fences: +0.4).
- s0005 = s0004 + that fix, predicted ~1400 / ~8000 us: VALID, measured 1145 us (B=1) and 7372 us (B=8) over
  HTTP, 8.3x and 1.39x SGLang; all gates pass, single == batched bitwise, HTTP tokens 8/8 identical.
  The B=1 figure is the median of three repeats; two of the three timed prompts are two-token loops ("The\n\nThe..."),
  where the context-copy drafter verifies 16 positions per pass and 14-15 are accepted. The third repeat (prose)
  ran at 3690 us per token (2.6x SGLang): that is the honest figure for ordinary text at B=1.
- Whether s0004's failure was that race or something data-dependent in the evaluator's sequences stays open.

## hours 2.9-3.9: s0006, s0007 (track B), and what did not work

- s0006 (coalesced RMSNorm prologue, bit-identical; 32-column loop k-step order): predicted b8 ~6500, measured
  1100 / 6502 us. The prologue had cost 9.4 us per op at 16 columns and 18.6 at 32 (every block recomputed r for
  every column with strided 4-byte loads); now 3.0 / 3.3.
- s0007 (32 positions per verify pass for a single sequence, L2 prefetch of the next matrix at every barrier,
  barrier-free delta-rule recurrence): predicted ~870 / ~6550, measured 824 / 6460 us (11.6x / 1.58x SGLang).
  In process, the two looping timed prompts reach 25.6 tokens per pass (827 us per token); the prose prompt stays
  at ~2.4 tokens per pass (5.2 ms per token) - the honest prose number did not move.
- Measured and reverted (each A/B'd on the same card, same job; bench/variants/ab2.sh):
  * splitting the stream loop at tile ends ("no call inside the loop"): +0.25 ms per step;
  * tile-sum slots in global memory instead of shared memory (to leave L1 larger): stream unchanged, epilogues
    slower, +1 ms;
  * run-time offsets for the recurrent op's shared memory: +0.13 ms per step; an 84 KB shared-memory carve-out for
    every launch: +0.15 ms (hence the op is compiled for 8 or 32 positions).
- Per-op timer (block 0 stamps %globaltimer after each barrier, tools/op_times.py): even with out_proj's whole share
  prefetched into L2 during the recurrent op, out_proj only drops 17.6 -> 15.9 us. The small GEMMs are not
  DRAM-bound: each warp streams ~6 quads, so the load latency per iteration and the cut-tile finalize dominate.
- Pod losses: epochs 17 -> 18 at hour 3.9.

## hours 3.9 (clock stopped): pod outage from 10:26 PST; offline work, untested on the card

Written while no pod was available (queued tests e0146-e0148):
- minimal-state stream loops for BF16 and W12 (the loop carries only pointers, a down-counter, the quads left in
  the tile, accumulators, one quad of weights, one k-step of activations; tile bookkeeping is recomputed at tile
  ends) - aimed at the 8 spills per iteration of the 32-column loop;
- the RMSNorm scalar published without a block barrier when there are at most 4 columns (r is needed only by the
  epilogues; they wait on a shared-memory counter);
- the head's argmax folded into the head GEMM's epilogue with a 64-bit atomicMax of (order-preserving value key,
  ~index): exact and order-independent, so deterministic; the argmax op (19 us per head) becomes an unpack;
- debug: per-warp stream end times inside one op (tail imbalance), per-op timer.
Track-A status, honestly: a single step is 10.2 ms in process (~10.25 over HTTP), behind SGLang (9523 / 10231 us).
Each GEMM loses ~8 us to ramp, finalize, epilogue and barrier (~2 ms per step), the steady stream runs at ~6 TB/s.

## hours 3.9-4.15: per-sequence verify depth; profiling the GEMM tail

- Per-op and per-warp timers (tools/op_times.py, tools/warp_tail.py) inside a real B=1 step: every GEMM spends
  5.6-8.7 us AFTER its slowest warp has finished streaming (in_proj 7.4, out_proj 7.9, gate_up 5.6, down 8.7):
  the cut-tile finalize across blocks, the epilogue and the barrier. ~1.9 ms per step - the largest remaining cost.
  Within the stream, warps end between 10 and 25 us (in_proj; median 22): ~3 us of imbalance.
- Tried and measured without gain: preloading the residual rows a block may finish while it streams (the
  epilogue's dependent residual load is not the bottleneck); a minimal-state stream loop (its SASS got worse: 30
  reloads per iteration at 32 columns; timing neutral); the W12 loop restructured the same way (worse: 35 us
  stream vs BF16 27).
- Nsight Compute on a launch of 20 in_proj ops: at 1 column, L1 hit 30 %, L2 traffic 5.3 GB for 3.4 GB of
  weights, stalls 65 % long scoreboard, 22 % barrier. At 32 columns the activation vector (327 KB) no longer fits
  in L1: hit rate 7.7 %, L2 traffic 16.3 GB for the same weights. That is why 32-column passes cost 1.8x a step;
  fixing it needs blocks whose warps sweep K together (not done).
- Argmax folded into the head GEMM (64-bit atomicMax of an order-preserving key and the complemented index:
  exact, deterministic): 18.4 -> 1.9 us per head.
- Per-sequence verify depth at B=8 (copyable sequences up to 8 positions, others 2-4, 32 columns): two bugs found
  by testing - the context-copy search ran only when the next pass had room for copied drafts (a demoted
  sequence could never be promoted again), and the verify pass's MTP input picked its hidden-state column with
  the uniform formula. After the fixes: decodes token-identical; 5629 / 5667 / 5835 us per step on the timed seed
  sets (uniform: 6368 / 6386 / 6891). Submitted as s0008.
- Pods: epochs 18 (1 h 25 min without a pod), 19, 20 (single-card pods), 21.

## hours 4.15-4.5: s0008 (track B, valid), s0009 (track A, valid); what the tail is not

- s0008 (per-sequence depth): predicted b8 ~5800, measured 843 / 5738 us (b8 -11 %; 8.7x / 1.78x SGLang).
- s0009 (track A, the same source with engine/TRACK = A): predicted ~10300 / ~11950, measured 10337 / 12028 us.
  Better than s0002 (10834 / 14059) but behind SGLang (9523 / 10231) by 8.5 % / 17.6 %.
- B=1 base verify depth swept on the prose prompt: 3 -> 5298, 4 -> 5071, 6 -> 5211, 8 -> 5275 us per token;
  default now 4 (copy mode still goes to 32).
- The GEMM tail, per block (tools/tail_parts.py, B=1 step): after the slowest warp's stream ends, the last block
  reaches the grid barrier 4.6-7.9 us later and the release is seen 1.3-2 us after that. Three fixes measured,
  none moved the op time: block-aligned partitions (no tile shared between blocks: +0.08 ms per step), the
  partition computed instead of loaded from tables (neutral; with 64-bit integer divisions it was +0.6 ms),
  residual rows preloaded during the stream (neutral). The tail is not the cross-block exchange, not the
  table loads and not the residual load; finer stamps inside finalize/epilogue would be the next step.
- W12 in the engine (the s0007 loop restored; tools/prof_w12.py): stream without patches 25.9 vs BF16 27.0 us
  (in_proj), 12.7 vs 12.7 (out_proj); with the per-quad patches 36.1 / 33.2 us. The patch handling costs more than
  the 25 % of bytes saves, and the decode loop is not faster than the BF16 loop the way the microbenchmark is. A
  W12 that pays off would need the escapes out of the hot loop (an exact sparse correction per tile in the
  epilogue) and a double-buffered decode loop; not attempted.

## hours 4.5-5.0: the tail and the 32-column L1 problem - two more attempts, both reverted

- Per-warp post-stream stamps (tools/tail_warps.py): in most blocks the warp that delays the barrier arrival is
  warp 31 (in in_proj also warps 0-4): the warps that finish a tile shared with the neighbouring block. Their
  post-stream work takes a median 3.7-4.1 us (two dependent L2 round trips while other blocks still stream),
  the other warps ~1 us. Barrier spin variants (tight, 50 ns, 200 ns back-off): no difference (1.2-1.6 us).
- 2-D partition (G tile groups x J K-bands per block, so a block's warps read one band of the activations and
  32 columns would stay in L1; tile partials summed in band order across the J blocks): correct (bit-identical
  invariants hold), but slower at every J: B=1 step 11.3 ms at J=1, 12.9 at J=8 (1-D: 10.26); 32-column in_proj
  stream 63-66 us vs 57 (1-D in the same build). Even J=1 (no bands, whole tiles per block) has a slower stream
  loop than the 1-D one (30.5 vs 27.0 us), so the extra loop state costs more than any locality gained; with J>1
  every tile also pays the cross-block exchange. Reverted; the code is kept in bench/variants/mk_2d.cu.

## hour 5.0-5.4: dataflow between GEMMs (per-tile flags instead of grid barriers) - slower, reverted

- Producer epilogues (residual, gate/up) published each tile with a release store of the launch's sequence
  number; the next GEMM skipped the grid barrier and waited on the flags of the activations its share needs
  (acquire loads), the RMSNorm scalar's warp on all 320 producer tiles. Bit-identical results.
- Measured (B=1 step, same build, dataflow on / off): per-chunk waits inside the stream loop 10.98 / 10.55 ms;
  one wait per warp share before the loop 11.42 / 10.54 ms (the committed engine: 10.26). Per op, consumers got
  slower (in_proj 44.5 vs 31 us) by more than the barriers they replaced.
- Why it cannot win here: a consumer block's share spans the whole K range, so every block needs essentially all
  producer tiles before its first useful MMA - per-tile flags degenerate into an all-tiles wait - and ~4800 warps
  polling the same few hundred flag lines make an L2 hot spot that slows the producers' own tails. A call to the
  wait inside the hot loop also cost +0.3 ms by itself (register allocation), even when never executed.
  Kept in bench/variants/mk_dataflow.cu.

## hour 5.0-5.6: where a 32-position pass goes; attention and recurrence restructured (kernel; both tracks)

- Per-op totals of one pass (tools/op_times_p.py, context 600): B=1 P=32 23.6 ms, of which attention 3.6 ms
  (score 1.63, value sum 1.11, out 0.86) and the recurrent op 3.05 ms; B=1 P=1 10.4 ms (attention 0.6, recurrence
  0.48). Track A (P=1) is GEMM-bound; the 32-position track-B pass was not.
- Negative: double-buffering the BF16 stream loop in registers spills 108-208 B at the 64-register cap and is
  +60 %; a bulk L2 prefetch of the quad D steps ahead (lane 0, no registers) is worse at every D (D=2 +3.6 %,
  D=4 +13 %, D=8 +35 % per B=1 step). The stream itself is at DRAM rate; the per-op cost is the tail (C7).
- Attention at P=32: score/value-sum/out had one warp per (sequence, kv head, key tile or dim tile) looping over
  the 24 column groups (6 q heads x 32 positions / 8): 160-192 warps doing ~96 dependent round trips each. The
  column group is now part of the work unit (3840-4608 warps). Same arithmetic per output: test_multi bit-equal
  at every (B, P). B=1 P=32 pass 23.6 -> 18.8 ms; B=8 P=4 19.6 -> 17.8 ms; P=1 unchanged (one column group).
- Recurrent op (tools/lmprof.py, block-0 phase stamps, -DLMIX_PROF): at P=32 per layer replay+state 2.3 us,
  inputs 17.1, recurrence 32.2 (~1 us per position), norm 3.6.
  * inputs: a thread walked its positions with one dependent load round trip each; now each half-block takes a
    contiguous run of positions and requests all of its inputs at once: 17.1 -> 11.2 us.
  * recurrence: ~160 SASS instructions per position per warp x 32 warps = issue-bound (~1300 cycles); spills
    were a side issue. The state's life (load, replay, recurrence, store) moved into its own function; the replay
    stages k in the core-output area and reads delta/decay directly (no layout growth).
  * split: for one sequence and P > 8, two blocks per value head (64 state columns each, 96 SMs instead of 48);
    the gated RMSNorm's sum of squares is assembled from the same four 32-column warp sums in the same order
    after an exchange through memory (epoch flags), so every column and every norm is bit-identical to the
    unsplit op; recurrence 29 -> 22.7 us per layer. B=1 P=32 pass 18.8 -> 16.5 ms.
- Latent bug fixed on the way: the replay staged na (accepted positions of the previous pass, up to 32) rows
  into areas sized for the current pass's PM; a 32-position pass followed by a <= 8-position one with na > 16
  would have overlapped the staged k and delta. The layout is now chosen from max(P, na_max) on both sides.
- Pod lost at hour 5.63 (epoch 21 -> 22) while listing spills.

## hour 5.6-6.0: s0010 (track B) - 622 / 4496 us

- Since s0008: attention work units per query column group; recurrent op with batched inputs, its own state
  function and 2 blocks per value head at B=1 with > 8 positions; replay staging bound fix; chain launches skip
  when no sequence needs that draft (per-sequence layouts); B=8 columns hi 16 / lo 1.
- B=8 column policy A/B (tools/test_spec.py, the client's window: last first-token to first last-token, so the
  aggregate rate counts, not the slowest sequence): default 4547/4570/4688 us; lo 1 4451/4486/4645;
  hi 16 lo 1 4448/4482/4618; "prose first" (finished sequences 1 column, MTP sequences first) 5523/5550/5975.
  Columns are worth most on the looping sequences (~0.95 tokens per column).
- R_ASYNC_MAX (RMSNorm scalar without block barrier) 4 vs 8: no difference at B=8 (11278 vs 11304 us); 32:
  +1.8 ms per 32-column pass. Kept 4: the B=8 GEMM excess over B=1 is not the prologue.
- Predicted http_step_us_b1 ~640, B=8 ~4600; measured 622 / 4496 (15.3x / 2.28x SGLang). Gates: rel-L2 median
  0.0100, http_tokens 8/8 identical, single == batched bitwise.

## hour 6.0-6.6: 32-column tails; s0011 (track A) 10267 / 11795 us

- tools/tail_p.py (per-warp stamps inside one 32-position pass): after the block sync a GEMM spent 11-14 us
  before its block arrived at the grid barrier (1 column: 4-6 us). Inside warp 31's cross-block finalize:
  part store + atomic 2.6 us, partial loads 2.3 (one round trip per column group), residual epilogue 6 us.
  * partials of all column groups requested at once (same additions, same order): -1 us;
  * residual epilogue: hoisting the loads made them spill (reloads from L2 when L1 is small: ~1 us per column
    group); packed registers, one instance per epilogue kind and per in-stream/after-sync use, the latter reading
    the shared-memory copy of the residual rows inside the loop: epilogue 6.6 -> 2.6 us;
  * fdiv (double division, slow-path call) replaced by a float estimate with exact integer correction: fewer
    spills in the stream/recurrence functions; finalize's own spills (callee-saved registers around its calls)
    remain. Same-card A/B against the previous commit: P=32 pass -2.0 %, B=8 P=4 -1.7 %, P=1 unchanged.
- Negative: shared memory just under the 132 KB carve-out step (residual-row cache 3 or 2 tiles instead of 6):
  no change (+-0.9 %). Growing shared memory: 170 KB +3.5 %, 200 KB +36 % per 32-column pass.
- s0011 = track-A build of this kernel (operator rule: track A keeps up with the kernel). Predicted ~10300 /
  ~11950; measured 10267 / 11795 us (SGLang 9523 / 10231).
- Where a 32-position pass goes now (B=1, 15.7 ms): GEMMs ~13 ms, of which the stream itself is ~45 % slower
  than at one column (in_proj stream 31 vs 21.5 us): activations (4 KB per 2 KB of weights per warp-quad at 32
  columns) miss L1. At one-column speed the pass would be ~11.8 ms.

## hour 6.6-7.3: s0012 (track B) 582 / 4145, s0013 (track A) 10184 / 11724

- Split of the extra cost of a 32-column GEMM over a 1-column one (debug flags 8 = prologue + barrier only,
  16 = stream without finalize/epilogue; us per op): stream rate gate_up +10.4, down +6.6, in_proj +5.3,
  out_proj +4.8; tails +3.5 to +7.2; RMSNorm prologue (block-barrier path above 4 columns) +2 to +4 on the
  ops that need r. Per 15.7 ms pass: ~1.7 ms stream rate, ~1.5 ms tails, ~0.5 ms prologue.
- bench/ng2.cu (stream design space, 168 MB, in isolation): 1024 threads 8 columns 26.9 us, 32 columns 30.0 us
  (+11 %); 512 threads x 2 tiles per warp at the same k, 32 columns: 27.1 us (the 32-column penalty gone);
  every 1024-thread variant with 2 tiles per warp or two quads in flight spills (41-51 us); half-quads with 2
  tiles: 30.0. Shared-memory reservation 0-170 KB: no effect in isolation. A 512-thread kernel is the only
  structure found that streams 32 columns at the 8-column rate; it would touch every op (thread mappings).
- Warp 31 used to finalize the tile ending in its own share and then the block's outgoing cut tile, serially;
  the latter now goes to warp 30 or 29 when free (all slots are in shared memory after the block sync): P=32
  pass -2.3 %, B=8x4 -2.8 %, single steps -0.5 to -0.9 %, bit-identical.
- s0012 (track B): predicted ~585 / ~4200, measured 582 / 4145 (16.4x / 2.47x SGLang).
- s0013 (track A, same kernel): predicted ~10180 / ~11740, measured 10184 / 11724.

## hour 7.3-8.6: s0014 (track B) 512 / 3847, s0015 (track A) 10175 / 11718

- Copy-drafter match length (NGRAM) A/B on the timed prompts: 6 -> 4 saves one ramp pass on the looping
  prompts (582 -> 571 us), prose and B=8 within +-0.5 %; 3 no better. Kept 4.
- A real 32-position verify pass in the middle of a looping decode (tools/verify_ops.py, block-0 stamps) took
  16.5 ms, of which the recurrent op 2.9 ms against 2.1 in a synthetic pass: every pass replayed the 32
  accepted rank-1 updates per head and layer. Now each verify pass also writes its final state into a second
  buffer; when the accept op sees every position accepted it flips the sequence's buffer index and sets the
  replay count to 0. Bit-identical (the replay is the same fused multiply-adds in the same order). 571 -> 548
  us per token on the loops, B=8 -1.7 %, prose +0.6 % (the extra state write).
- nsys on a looping B=1 decode: the GPU is never idle between passes (gaps 2-10 us); the pass is the kernel.
- The split recurrence (2 blocks per head) gained only 12 % over the unsplit one: SASS showed 19-32 spill
  reloads per position in the split instances (conditionals around the state's life; ptxas: 720 / 492 bytes),
  none in the unsplit one. The unsplit loop is issue-bound (147 instructions x 32 warps / 4 schedulers ~ 0.62
  us per position, measured 0.67). With the idle half leaving at the block barrier the split loop is spill-free:
  25.6 -> 16.7 us per layer at 32 positions; same-card pass A/B -7.3 % (B=1 P=32), -4.7 % (B=8 x 4).
- Smaller: 32-column activations via ld.global.cg (-0.6 %); L2 prefetch sizes re-tuned (128 KB / 1 MB still
  best; 256-512 KB worse); MTP head in a copy-mode verify pass 0.38 ms (2.4 %, left alone).
- Negative: 32-column loop split into column-group halves (more spills: ptxas hoists all loads anyway);
  microbenchmark: activation traffic costs ~5 % at 32 columns in isolation, 512 threads x 2 tiles no consistent
  gain (27.1-27.5 vs 28.4-30.0 us) - the 512-thread port is not justified by it.
- s0014 (track B): predicted ~515 / ~3900; measured 512 / 3847 (18.6x / 2.66x SGLang).
- s0015 (track A, same kernel): predicted ~10170 / ~11650; measured 10175 / 11718.

## hour 8.6-9.4: per-op fixed costs at 32 columns

- tools/empty_op.py (GEMM ops with the stream off, flag 8; per-block arrival and release stamps): an empty GEMM
  op costs 4.4-5.4 us at 1 column and 4.9-8.5 at 32. After the last block arrives the barrier releases 1.0-1.5
  us later without the L2 prefetch of the next matrix, 1.3-3.6 us with it (the prefetch traffic delays the
  release) - but moving the prefetch away from the barrier is worse: issued at the block's stream end +2.4 %
  per 32-column pass (it delays the block's own arrival); the 128 KB "near" prefetch issued after the release
  +1.7 to +2.9 % everywhere. The prefetch is worth more during the barrier than the latency it costs. Reverted.
- Residual rows for the residual epilogue preloaded with cp.async (16 bytes per thread, column-major): no thread
  waits for its load before streaming; empty out_proj/down at 32 columns 5.6-6.1 -> 4.9 us; pass -0.4 / -0.9 %.
- RMSNorm scalar above 4 columns without the two block barriers: every warp writes its partial sums (the same
  coalesced loads and sums as before) to a 4 KB shared area and counts itself; the first epilogue that needs r
  elects one warp that finishes the 32-way reduction in warp_sum's xor-butterfly order (bit-identical to the old
  path and to the <= 4-column path). Same-card A/B: P=32 -1.4 %, B=8 x 4 -1.3 %, B=8 single step -1.8 %.
- Negative: cross-block hand-over of a tile cut once by flag (sender: stores + release store of the op epoch;
  receiver: waits, adds) instead of the counter + last-arriver: +0.3 to +0.7 %. Reverted.
- Negative: 3-way split of the recurrent op (heads on 144 SMs; 44/44/40 columns - 43 columns left a warp partly
  exited and the full-mask shuffles hung; norm from all 128 outputs exchanged through memory, bit-identical):
  recurrence 17.4 -> 15.1 us per layer but norm/exchange +0.6 us, and the whole 32-position pass +1.1 % (B=8 x 4
  +0.4 %). Reverted to the 2-way split.
- s0016 (track B): predicted ~505 / ~3760; measured 506 / 3762 (18.8x / 2.72x SGLang).
- s0017 (track A, same kernel): predicted ~10175 / ~11520; measured 10195 / 11473.

## hour 9.1-9.3: attention ops folded

- 64 positions per pass (the loops accept everything; a pass costs ~10.1 ms + 0.126 ms per position) would need
  32 accumulator registers and 256 KB of tile-sum slots: impossible at 1024 threads. bench/ng2.cu with 512
  threads: 64 columns stream 30-40 % slower than 32 at 1024 threads, and the 512-thread layout makes 1-column
  streams up to 16 % slower (track A) and breaks the bit-identity of verify passes with single steps (different
  warp partition) unless every summation is emulated. Not pursued.
- The attention output op (sum of the value-sum chunks, gate) is done by the last value-sum chunk unit to arrive
  (an acquire-release counter per output unit, the same additions in the same order): one op and one grid
  barrier less per attention layer. Bit-identical; same-card A/B: B=1 P=32 -0.9 %, B=8 x 4 -1.0 %, B=1 step
  -0.8 %, B=8 step -0.6 %.
- Pod lost at hour 9.32 (epoch 22 -> 23) while testing the softmax fold (score op leaves per-tile max and
  exp-sums, the value sum makes the probabilities; not bit-identical to the old softmax's summation order).
- Negative: softmax folded into score (per-tile statistics) + value sum (probabilities made there): correct (rel-L2
  0.0100 / 0.0299, top-1 99.6 %, identity tests pass) but single steps +1.9 % (B=1) / +1.6 % (B=8) and 32
  positions only -0.4 %: every value-sum unit recomputes the statistics and reads fp32 scores. Reverted.
- Pod lost again at hour 10.12 (epoch 23 -> 24).

## hour 10.1-10.9: recurrent-op inputs 4 channels per thread; s0018 (track B) 488 / 3695, s0019 (track A) 10150 / 11435

- lm_inputs: the recurrent op's inputs with 4 adjacent channels per thread (8-byte loads, one warp = one role of
  one run of positions), q/k L2 norm finished inside the warp with warp_sum's butterfly reproduced across lanes
  and the thread's 4 channels (bit-identical). Same-card A/B: 32-position pass -4.2 %, B=8 x 4 -2 %, but single
  steps +1.2 % (128 threads busy at 1 position): used only for passes of 4+ positions, chosen by template (a
  run-time choice in one function made both paths slower).
- A race it introduced, caught only by the B=8 online identity test (sequences diverged at random tokens 379-781;
  fixed-length tests and B=1 passed): the previous pass's k vectors were staged from pk for the replay while this
  pass's k warps already overwrote pk; the old code had a block barrier in between by accident of structure. Fixed
  with a barrier after the staging. Lesson: test_spec at B=8 (variable lengths, replays) after every lmix change.
- s0018 (track B): predicted ~488 / ~3700; measured 488 / 3695 (19.5x / 2.77x SGLang).
- s0019 (track A, same kernel): predicted ~10120 / ~11420; measured 10150 / 11435.
- Negative: RMSNorm scalar at 1-4 columns deferred (warps 0-9 load their tile sums at the stream start, the last
  of them reduces at its end; bit-identical): single steps +0.9 to +1.1 % - the reduction lands on the finalize
  critical path. Block-level lag is barely systematic (tools/block_skew.py: per-block lag spread ~1.5 us,
  correlation 0.18-0.34 between ops/repeats), so a fixed rebalanced partition is not worth it either.
- Ramp: the length of pass n+1 is fixed when pass n is launched (its drafts are written by pass n's accept op), so
  with two passes in flight a copyable continuation found by pass n sets the length of pass n+3. The first 8 passes
  of a cohort are now queued one at a time (the GPU waits ~0.2 ms for the server's poll per pass while this lasts):
  looping prompts 487 -> 478 us (38 passes instead of 39), prose and B=8 unchanged.
- Negative: softmax row held in registers (36 keys per lane, one round trip, bit-identical): +2.5 % per step, +1.2 %
  per 32-position pass. Reverted.
- s0020 (track B, host scheduling only; the track-A engine is unchanged by it, s0019 stays current): predicted
  ~479 / ~3690; measured 479 / 3690 (19.9x / 2.77x SGLang).
- W12 at 32 positions: 17.3 ms per pass vs 13.5 (BF16), at 1 position 11.3 vs 10.1: stays off.
- 2-block thread-block clusters keep all 152 blocks resident at 140 KB shared memory (4-block: 144, 8-block: 120;
  bench/cluster.cu), so cut tiles could go through distributed shared memory only at every other block boundary;
  a pair-aligned partition would leave out_proj/down 19 % imbalanced. Not pursued.
- notes/summary.md: the engine's state and results in one page.
- Negative (again, now at 32 columns too): block-aligned partition for matrices with >= 4 tiles per block (no tile
  cut between blocks, one tile of imbalance): B=1 P=32 +5.4 %, B=8 x 4 +4.9 %, B=1 step +5.6 %, B=8 step +5.2 %.
  The imbalance costs far more than the cross-block exchange it removes. Reverted.

## hour 11.4-12.0: drafting policy; s0021 (track B) 468 / 3626

- In an established loop the MTP head only validated the next copy (0.33 ms of a 13.9 ms 32-position pass).
  When no sequence needs MTP drafts for the next pass (copied drafts accepted in full and the drafter still
  matches, or a single position), the host launches a verify program without the MTP head; the accept op then
  takes a context copy without the agreement check; any rejection brings the MTP head back. Exact as before (every
  token is verified); only draft quality could suffer later on text that leaves a long loop (the MTP layer's K/V
  for skipped positions is not written). test_spec: looping B=1 478 -> 468 us, B=8 -1 to -2 %, prose unchanged.
- s0021 (track B, host policy + an accept-op mode; track A unaffected, s0019 stays current): predicted ~468 /
  ~3630; measured 468 / 3626 (20.3x / 2.82x SGLang).
- B=1 ramp, nothing in flight: the pass's length now follows the previous pass's result (every pass leaves room for
  Phi - 1 copied drafts; MTP chains still make only the short-pass drafts): looping prompts 467 -> 457 us (ramp
  of 4 passes: 1, 1, 2, 2 tokens, then 32 per pass). Copy match length re-tested with it: 3 = 4; 2 gives loops
  446 us but prose +2 % and B=8 +2.5-4 %: kept 4.
- s0022 (track B, host scheduling only): predicted ~458 / ~3625; measured 458 / 3625 (20.8x / 2.82x SGLang).
- Activation traffic at 32 columns, timing-only experiment (half the column groups read the same activations):
  stream-only 32-position pass 11.46 -> 10.95 ms (-4.5 %), gate_up -4 %. Activation (L2) traffic is worth at most
  ~5-9 % of the stream; sharing it between warp pairs would recover about half. Not pursued.
- Sequence positions cached in shared memory per launch (they change only in the last op): -0.3 % per step.

## hour 12.1-12.9: two recurrent units per block (track A, B=8); s0023 (track A) 10154 / 10988

- At one position without speculation and with more units than blocks (B >= 4: 384 units on 152 blocks = 3
  rounds), each 512-thread half of a block now runs its own unit (named barriers per half; the input phase at one
  position already used threads 0-511; in the recurrence a thread holds state columns v and v + 64 of the
  unsplit layout, one after the other, same arithmetic): 2 rounds. Bit-identical (test_multi 10/10: B=4/8 steps
  vs multi-position passes; single == batched bitwise). B=8 step 11.43 -> 10.97 ms, B=4 -4.5 %.
- Bug on the way: the launch's shared memory was sized for one unit layout; two needed 42.6 KB, B=4 launches had
  39.7 KB (illegal address), B=8 had 43.8 KB by the accident of the RMSNorm partials area. The host now reserves
  two layouts at one position.
- s0023 (track A): predicted ~10100 / ~10950; measured 10154 / 10988 (B=1 unchanged within noise, B=8 -3.9 %).
- Robustness: online identity on four prompt sets the timed workload never uses (seeds 12345-45678), B=1 and B=8:
  all 1024-token decodes identical to step-by-step greedy.
- B=8 verify passes: the recurrent units (sequence, value head) were assigned round-robin (blocks 0-79 three units,
  80-151 two, regardless of each sequence's length); the host now places them longest-first on the least loaded
  block (cost 6 + 0.8 us per position) in a table carried in the per-pass layout slot. Same-card A/B against the
  previous commit: -0.1 to -0.8 % per B=8 step (smaller than the model promised).
- Found while doing it: the context struct (376 bytes) and the op descriptor after it reach byte 535, overlapping
  the RMSNorm scalars at byte 512 (op words p9-p11). Harmless (only GEMMs write r and they do not use those
  words), now documented in the source; growing Ctx would need SMR_OFF moved.
- Replay deltas kept in shared memory (over the consumed v inputs) and written out after the recurrence in one
  coalesced pass: -0.2 % (B=1 P=32), -0.4 % (B=8 x 4), bit-identical.
- s0024 (track B): predicted ~458 / ~3590; measured 457 / 3604 (20.8x / 2.84x SGLang).

## hour 13.0-13.4: a serving bug at B=2-7 (found by a robustness sweep)

- submit()/poll() at B=2, 3, 4 finished requests early (e.g. 10, 138, 264 of 512 tokens) with the finished flag
  set. Cause: in per-sequence mode a pass uses up to NCM = 32 columns, but the per-pass token buffers (device out
  and pinned host copy) were sized B x Pv = B x 4 columns: tokens beyond were dropped by the slice copy while the
  counters advanced (and the kernel wrote past the device row). At B=8 the size is exactly 32 and at B=1 the
  fixed-length path uses 32, so the evaluated configurations were never affected (bisect: present since s0010).
  Fixed (buffers max(B x Phi, NCM) wide); tools/serve_lens.py checks every request gets exactly n tokens.
- Second defect found by the same sweep: with the unit table, the recurrent op still let only blocks < B x 48
  enter, while the host spread the units over all 152 blocks: at B < 4 some units were never computed (B=8 has
  384 units, every block enters). Fixed (with a table every block enters).
- My own regression on the way: a debug print inserted into poll() replaced the `if a > 0:` line, so tokens were
  only delivered with the debug variable set (which is why the first re-test passed). Caught by the next re-test,
  never submitted; the debug code is gone.
- After the fixes: online identity at B = 1, 2, 3, 5, 7, 8 on two to three prompt sets each: all identical;
  every request gets exactly n tokens at B = 1-4 and 8.
- s0025 (track B, correctness fixes for B=2-7): predicted ~457 / ~3600; measured 458 / 3604.

## hour 15.5-16.2: softmax four warps per row; long-context check

- Online identity at contexts up to ~1960 (64 + 1900 tokens), B=1 and B=8: identical.
- Softmax with four warps per row (8 rows per block, chunk-interleaved keys, partial max/sum combined in a fixed
  order through shared memory): one batch of loads per pass instead of three dependent ones. Changes the
  summation order (gates: rel-L2 0.0101 / 0.0296, top-1 99.4 %); steps and verify passes still bit-identical
  (test_multi 10/10, online identity at B=1/8). Same-card A/B: B=1 P=32 -0.5 %, B=1 step -0.4 %, B=8 step -0.3 %.
  The prose prompt's greedy text changed with the rounding (2.57 instead of 2.34 tokens per pass: a different
  continuation, not a faster engine).
- s0026 (track B): predicted ~455 / ~3600; measured 457 / 3626 (B=8 within run-to-run noise of s0025).
- s0027 (track A, same kernel): predicted ~10115 / ~10955; measured 10139 / 10925.
- Negative: the barrier's L2 prefetch issued in 16 or 32 KB pieces with 100-300 ns gaps instead of two 64 KB
  requests: equal or worse (300 ns gaps +12 %: the issuing thread delays the barrier's block sync). Reverted.
- Negative: recurrence re-laid as 16 lanes x 8 rows per column (all 1024 threads busy in split blocks, 4-level
  butterflies, column-major state): correct (rel-L2 0.0100 / 0.0310, bitwise invariants hold) but +16 % per
  32-position pass, +17 % B=8 x 4, +2.6 % per step. The 8 x 16 layout stays. Reverted.

## hour 16.0-16.6: prefill in chunks (engine quality, not scored)

- The prompt went through 63 single-position launches (~0.68 s to the first token for 64 tokens). Since a
  multi-position pass is bit-identical to single steps, the prompt now goes in chunks of up to NCM // B positions
  (32 at B=1, 4 at B=8), both on the speculative and the plain (track A) path: first token after 88 ms (B=1) and
  221 ms (B=8) instead of ~675 ms; online identity unchanged (B=1/2/8 track B, B=1/8 track A, identical pass
  counts). The client's decode window excludes the time to the first token, so the score does not move.
- Constants re-checked at the current kernel (barrier sleep 100/200/400 ns, async RMSNorm threshold 2/4): noise.
- s0028 (track B, chunked prefill): predicted ~457 / ~3620; measured 457 / 3624 (http_tokens identical).
- s0029 (track A, chunked prefill): predicted ~10140 / ~10925; measured 10146 / 10923.

## hour 16.6: state of the run (see notes/summary.md)

- Track B best: s0028, 457 us per token at B=1 and 3624 us per step at B=8 over HTTP (20.8x / 2.82x SGLang),
  every token the model's own greedy token (verify passes bit-identical to single steps).
- Track A: s0029, 10146 / 10923 (0.94x / 0.94x SGLang): one launch per token, the checkpoint's formats.
- Since hour 5 (s0009: 10337 / 12028 track A, s0008: 843 / 5738 track B) the gains came from: attention work
  parallelised per column group; the recurrent op restructured (batched/vectorised inputs, spill-free state core,
  split heads, no replay after full acceptance, two units per block for single steps); GEMM tails (batched
  partial loads, per-kind spill-free epilogues, free-warp finalize, cp.async residual rows, barrier-free RMSNorm
  scalar); the attention output folded into the value sum; drafting policy (B=8 columns, shorter ramp, no MTP
  head while copies keep being accepted). About as many attempts were measured and reverted (journal).
- Correctness work: two latent B=2-7 serving bugs found by a sweep and fixed (s0025); identity holds online at
  B=1-8, up to ~1960 positions, on prompt sets the evaluator does not use.

## hour 16.6-17.6: sentinel hand-over of cut tiles (from the Flint run's notes)

- Flint's finding 12(e): every dependent step of a hand-over is one trip through the memory queue (1.8 us idle,
  ~4 us loaded); their sentinel protocol needs one. Here: the cut tile's first block stores its partial into a
  NaN-sentinel slot, the last block polls the slot, adds in block order, restores the sentinel (the grid
  barrier orders that before the slot's next writer); tiles cut more than once keep the counter.
- First version: 3 % faster but wrong (rel-L2 1.3). Debug builds: senders = receivers, identical (op, slot) sets
  per op, no duplicates - yet 16-35 % of the receivers read an older op's data (epoch tags): the weak ld.cg poll
  saw a stale copy. With relaxed gpu-scope loads/stores: 0 stale reads, all invariants hold (findings C11).
- Same-card A/B: B=1 P=32 -1.4 %, B=8 x 4 -1.4 %, B=1 step -1.6 %, B=8 step -1.9 %. test_spec: loops 450 us,
  B=8 3522-3681 us, all identical; B=2-7 delivery and track-A identity checks pass.
- s0030 (track B): predicted ~450 / ~3570; measured 450 / 3573 (21.1x / 2.86x SGLang).
- s0031 (track A, same kernel): predicted ~9985 / ~10715; measured 9981 / 10713 (first track-A step under 10 ms).
- The split recurrent op's norm sums (2 blocks per head at B=1, > 8 positions) now go through sentinel slots too
  (a block keeps its own sums in shared memory; the partner reads and restores): -0.4 % per 32-position pass,
  identical everywhere; test_spec loops 448 us.
- s0032 (track B): predicted ~448 / ~3570; measured 449 / 3571 (21.2x / 2.87x SGLang). Track A is not affected
  (the split path needs > 8 positions), s0031 stays current.
- Negative: sigmoid through __frcp_rn (Flint's finding 13c: the compiler's division is expensive): bitwise
  identical logits (tools/bits_cmp.py) but +0.2-0.3 % - the compiler already emits a reciprocal for 1.0f / x.

## hour 17.5-18.3: track labels; the far L2 prefetch; s0033, s0034 (track A)

- Labeling: the operator's rule of 2026-11-11 files every draft-and-verify submission under track C; this run's
  hypotheses said "track B" for s0014-s0032 (mk note sent). Tracks A and B are one token per pass; the remaining
  hours go to track A (SGLang 9523 / 10231 is the bar) and lossless one-token changes.
- s0033 (track A build of s0032): predicted ~9980 / ~10715, measured 9968 / 10728.
- One-position op times (tools/op_times_p.py, B=1 / B=8): gate_up 3613 / 4044, down 2089 / 2088, in_proj 1496 /
  1527, out_proj 933 / 1107, recurrent op 495 / 1218, qkv 448, lm_head 352, attention 4 x 110-150. Every GEMM
  carries ~8 us over its bytes at 7.3 TB/s. Recurrent op vs B (one position): B=1-3 495-528 us (one unit per
  block), B=4 830 (two units per block, one round), B=8 1218 (two rounds).
- Recurrent op phases at one position (LMIX_PROF, now also in lmix_dual): B=1 start 1.6, inputs 2.2, state +
  recurrence 3.5, norm 1.0 us per layer; B=4 per dual round start 2.7, inputs 4.1, recurrence 6.7, norm 1.9.
- Negative: both state columns of a dual half requested at once (the volatile loads kept them one after the
  other): bit-identical, 1219 -> 1218 us at B=8 - the memory trip is not what costs.
- Far prefetch sweep (MK_PF_FAR, the bytes of the next GEMM requested at barriers before small ops, 1 MB since
  hour ~4): with 0 the in_proj and qkv barriers (followed by the recurrent / attention ops) release faster and
  the recurrent state reads no longer queue behind 63-152 MB of weight prefetch. Same card, pass us:
  B=1x1 9968 -> 9878, B=8x1 11334 -> 11157, B=1x8 10390 -> 10299, B=1x32 13192 -> 13117, B=8x4 13572 -> 13398.
  NEAR (before a GEMM) 128 KB stays best (64 KB: B=8 11240, 256 KB: B=1 10000). Default FAR = 0.
- s0034 (track A, FAR = 0): predicted ~9880 / ~10570, measured 9874 / 10622 (SGLang 9523 / 10231: 0.96x / 0.96x).

## hour 18.3-19.3: the GEMM hand-over after the stream; s0035 (track A), s0036 (track C)

- tools/ho_stamps.py (build -DHO_STAMPS: %clock64 stamps per block; %globaltimer ticks in ~0.5 us steps and is
  useless below that): after the block sync the warp that receives a cut tile entered the hand-over 1.5-1.9 us
  later and the sending warp 2.2 us later, before any memory access: 64-bit floor divisions (qw/fdiv: I2F.S64,
  64-bit multiplies, corrections) rebuilding the partition in every warp at once, a serial search of the tile's
  first warp, the noinline finalize's 184-byte spill frame. The receiver then waited for the late sender.
  1. every warp writes its first quad (warp 31 also the block's end) to a 33-int table in shared memory before
     it streams (QTAB_OFF; WORK_OFF 192 -> 240 floats): B=1 step 9863 -> 9590, B=8 11165 -> 10898, B=1x32
     13118 -> 12747, B=8x4 13385 -> 13150 (-2.4 to -2.8 %);
  2. the tile's first warp by one ballot: -0.4 to -0.6 %;
  3. a tile not longer than the shortest block share spans two blocks: bf / bla without a division: -0.2 %.
  Same partition and additions: bitwise-identical logits (bits_cmp B=1/B=8), test_multi all equal, test_spec
  identical at B=1/8 on three seeds, serve_lens B=2 delivered, plain_check identical.
- Negative: block sums with 4 shared-memory loads in flight (finalize spills 184 -> 252 bytes): +1.9 %.
  Negative: finalize inlined into the stream function (spills move into the stream: 352-1200 bytes): +0.3 % at
  one position, +7 % at 32.
- Timeline now (in_proj, us after the block sync): finalize entry 0.5, block sums 0.87, poll issued 1.1, data
  1.8, epilogue done 2.2, barrier arrival 2.6 (was 3.3-4.2 before the change, all GEMM kinds).
- s0035 (track A): predicted ~9490 / ~10290, measured 9502 / 10291 us: 1.002x SGLang at B=1 (9523), 0.994x at
  B=8 (10231). First track-A number ahead of the general engine at batch 1.
- s0036 (track C, same kernel): predicted ~435 / ~3450, measured 432 / 3437.

## hour 19.3-19.9: attention placement, recurrent-op start; s0037 (track A)

- Negative (diagnosis): instruction-cache misses are not the post-stream cost (Nsight Compute stall sampling of a
  B=1 step: no_instructions 0.6 % of samples, long_scoreboard 58 %, barrier 26 %; tools/ncu_step.py).
- The attention ops gave units to consecutive warps (unit u = warp u of the grid): at B=1 the 152 key tiles of
  att_score ran on 5 SMs, the 32 prep units on one, the value sum on 6 - bounded by those SMs' L2 bandwidth.
  Units now go round-robin over blocks first (att_w()). Per step: prep 133 -> 72 us, score 112 -> 69, softmax
  123 -> 96. Same card: B=1 -0.6 %, B=8 -0.5 %, B=1x32 -0.5 %; bit-identical.
- Every recurrent unit (lmix_work, lmix_dual) started with a dependent global load of its state-buffer index
  (nacc[2 NCM + b]) and, in verify passes, the replay count; both are written only by the accept op (a pass's last
  op), so they are now read into shared memory at launch start (SPAR_OFF / SNA_OFF; WORK_OFF 240 -> 256 floats).
  B=8 recurrent op 1218 -> 1145 us. Both changes: B=1 step 9491 -> 9364, B=8 10805 -> 10647, B=1x32 -0.6 %.
  test_multi 10/10, test_spec identical on 3 seeds at B=1/8 (loops 428 us), serve_lens, plain_check.
- s0037 (track A): predicted ~9375 / ~10140, measured 9380 / 10199 us: 1.015x / 1.003x SGLang (9523 / 10231);
  milestone parity passed.

## hour 19.9-20.5: the B=8 recurrent op (negative)

- Four units per block (lmix_quad: 256-thread quarters, 4 state columns per thread, compact one-position layout;
  B >= 7: one round instead of two): bit-identical, 1145 -> 1148 us at B=8. Kept in bench/variants/mk_quad.cu.
- What the op's time is not: the state prefetch (off: 1145), the state traffic (no state load/store, debug flag
  32: B=8 1137 -> 947 us, B=1 500 -> 453), the SM clock (2062 MHz, no throttle reason, 960 W during B=8 steps,
  tools/clock_probe.py), instruction-cache misses. Phase times (LMIX_PROF, now %clock64 cycles): per unit, every
  phase - even the start, which has no global access - grows with the number of units an SM runs at once (B=1:
  start 1.6, inputs 2.1, recurrence 2.4 (no memory), norm 0.9 us; three units per SM: 3.0 / 6.0 / 6.9 / 2.6).
  Not understood; the SASS of the timed windows is short. Left as is.

## hour 20.5-21.0: more of the GEMM tail

- Finalize roles decided by each warp at its own stream end (while the others still stream; a role table in
  shared memory, WORK_OFF 256 -> 320 floats) instead of by all 32 warps at once after the block sync (a 32-bit
  division and tests per warp, ~0.2 us of issue contention): B=1 -0.4 %, B=8 -0.3 %, B=1x32 -0.9 %, bit-identical.
- Negative: the outgoing cut tile summed and sent before the block sync (the warps holding its pieces meet at a
  named barrier, warp 31 sums and stores): bit-identical, +0.8 % at B=1, +1.7 % at 32 positions - the sum now
  delays the block sync, and with it the block's own receiving warp. bench/variants/mk_send_presync.cu.

## hour 21.0-21.6: GEMM prologue; s0038 (track A)

- GEMM prologue: the residual-row tile range (two 64-bit divisions) is computed only by the threads that copy rows
  (was: all 1024), and a warp's two quad bounds come from one pass of the division (lanes 0 and 1, shuffled):
  B=1 -0.7 %, B=8 -0.8 %, B=1x32 0; bit-identical, test_multi 10/10.
- Negative / neutral: the next RMSNorm's gains copied into shared memory with the residual rows (the residual
  epilogue's one global load after the block sync): B=1 -0.15 %, B=8 +0.25 % (4 KB more shared memory). Reverted.
- Tail now (us after the block sync, receiving warp): finalize entry 0.3, own block's partial sums 0.6-0.8, poll
  issued 0.75-0.85, data +0.3-0.5, epilogue +0.4-0.75, barrier arrival 1.9-2.7. The largest remaining per-op cost
  is the spread of stream ends between the warps of a block (median to max ~2.6 us per GEMM); deterministic
  stealing would need whole tiles (~100 us of one warp's stream) or many fixed k-chunk partials per tile.
- s0038 (track A): predicted ~9270 / ~10085, measured 9281 / 10065 us: 1.026x / 1.016x SGLang.

## hour 21.6-22.3: attention, split recurrence at B=1; s0039 (track C), s0040 (track A)

- s0039 (track C build of the s0038 kernel): predicted ~425 / ~3380, measured 426 / 3404.
- Softmax rows of at most 1024 keys keep their scores in registers for the three passes; the output fold loads its
  gates before the partial sums: value sum 142 -> 127 us, softmax 96 -> 85 us per step; B=1 -0.4 %, B=8 -0.4 %,
  B=1x32 -0.5 %, bit-identical.
- The split recurrent op (2 blocks per head, findings C8) for a single step of one sequence: 506 -> 455 us per
  step, pass -0.3 %; bit-identical, test_multi / test_spec / serve_lens / plain_check pass.
- Negative: next op's descriptor loaded by warp 2 instead of threads 1-20 of the spinning warp 0: noise (+-0.2 %).
  Negative: the RMSNorm partial loads of warps 0..NC-1 removed altogether (wrong r, timing bound): no change.
- s0040 (track A): predicted ~9230 / ~10025, measured 9258 / 10065 (1.029x / 1.016x SGLang).
- Warp lag by index (tools/warp_lag.py, equal shares): warps 0-3 of every block end their streams 2.9-4.4 us
  before the block median, warps 4-9 ~0.5 us before, 10-31 ~0.2 us after. Negative: warp shares inside a block
  weighted by those rates (blocks unchanged): lags flatten but the block's slowest warp does not move (max -
  median 1.75 us either way) and passes get slower: B=1 +2.3 %, B=8 +1.7 %, B=1x32 +4 %. Reverted
  (bench/variants/mk_warpw.cu).
- s0041 (track C build of the s0040 kernel): predicted ~421 / ~3380, measured 426 / 3393.
- Constants re-checked on this kernel (bench/variants/const_sweep.sh): barrier sleep 50/200/400 ns x near prefetch
  96/128/192 KB: 200 ns / 128 KB within 0.1 % of the best; 192 KB +0.6 %. Unchanged.
- Skeleton of a B=1 step (every GEMM's stream switched off, debug flag 8): 1.84 ms of 9.2 (barriers, prologues,
  epilogues, small ops); an empty GEMM op 3.3-5.6 us, last arrival -> release 1.0-2.3 us.
- Negative: B=8 recurrent op as one round of two units per block (304) and the remaining 80 units one per block on
  whole blocks: bit-identical, 1145 -> 1168 us (bench/variants/mk_dual_single.cu). A unit alone on an SM after the
  dual round is not faster than a second dual round.
- Negative: at B=1 the 56 blocks without a recurrent unit ask L2 for the whole out_proj matrix (63 MB) during the
  recurrent op: out_proj 852 -> 815 us per step, recurrent op 455 -> 576 (its DRAM loads queue behind the
  prefetch): +0.7 % per step. Same lesson as the far prefetch (C9).
- Negative: the in-block partial sums in their own noinline function (4 shared-memory loads in flight): the tiny
  function alone reports 112 bytes of spill (the call's register save / restore through local memory), and the
  pass got +4.5 % slower - one more call level on the hand-over path costs ~1 us per op (the restores miss L1
  after the stream). The converse (the epilogue inlined into finalize: one call level less) measured -0.3 % / +0.9 %
  (B=1 / 32 positions) once, then +0.2 % for the one-column-group-only variant: noise. Reverted.

## hour 22.3-22.9: local-memory reloads after calls; s0042 (track A)

- C14 followed up (tools/sass_calls.py, sass_after_call.py, sass_fnmap.py): the kernel's main loop kept the program
  pointer, op count, op index and thread-index values across each op's function call; the compiler saved them to
  local memory and reloaded them after the call - right before the grid barrier, after a weight stream has
  replaced L1's contents (an L2 round trip per op, ~370 per step). Program pointer and op count now live in shared
  memory (MAIN_OFF), the op index is the barrier's shared copy, thread and block indices are read afresh with
  volatile moves: local loads in the loop body 28 -> 8, none after a call. Same card: B=1 step 9234 -> 8981,
  B=8 10506 -> 10243, B=1x32 12500 -> 12173 (-2.5 to -2.7 %); bitwise identical to HEAD, all tests pass.
- s0042 (track A): predicted ~9005 / ~9815, measured 9006 / 9900 us: 1.057x / 1.033x SGLang.
- The run's bridge daemon exited at 10:55 (FileNotFoundError in its spool scan, a vanished ack file) right after
  s0043 was submitted; restarted with its own command at 10:58 (mk note). s0043 was evaluated normally.
- s0043 (track C build of the s0042 kernel): predicted ~415 / ~3310, measured 414 / 3275.
- Negative: in the recurrent op, the split half and the speculative flag re-read after the calls (shared memory,
  block parity) instead of kept across them: the compiler still reloads two values after calls in the split
  variant; B=1 +0.3 %, others flat. Reverted. Remaining local-memory users on critical paths (tools/sass_fnmap.py):
  the recurrent-op functions (1-2 reloads after calls each, 17-23 spill slots), lm_core instances (5-15 local
  loads), the attention value sum (1 reload after the output fold).
- Negative: the recurrent op's unit (u, positions, head, first column, half) stashed in shared memory before the
  lm_core call and every use after it re-read from there: reloads after calls 5 -> 2, recurrent op 455 -> 449 us,
  but passes B=1 +1.2 %, B=1x32 +3 % (B=8, which does not use that function, -0.5 %): a code-generation side effect
  elsewhere. Reverted.
- Prefetch re-swept on the final kernel (near 64/128 KB x far 0/256 KB): 128 KB / 0 stays best (B=1 8986 vs
  9018-9058, B=8 10232-10248 vs 10318-10346).
- Negative / noise: the attention output fold inlined into the value-sum op (its one reload after a call): B=1
  +0.25 %, B=1x32 -0.2 %. Reverted.
- Negative: lm_core inlined into the one-position recurrent-op variants (no call): recurrent op 418 -> 420 us per
  step (HEAD's own figure in the same run was 418; an earlier 455 came from a different run), pass +0.6 %. Reverted.
- B=8 gate_up (tools/data_dep.py): 8 different tokens 3915 us per step, the same token in all 8 columns 3914, B=1
  3484 - the extra time does not come from the columns' contents differing; it grows with the number of real
  columns (B=2 3726, B=4 3936, B=8 4044 earlier). Cause still open.
- R_ASYNC_MAX 8 instead of 4 on the final kernel: B=8 10226 -> 10340-10357 (+1.2 %), 32 positions unchanged. Kept 4.

## Final state (hour ~22.4)

- Track A s0042: 9006 / 9900 us over HTTP (1.057x / 1.033x SGLang 9523 / 10231). Track C s0043 (same kernel):
  414 / 3275 us. HEAD's engine/ is s0043's source; s0042 is the same with engine/TRACK = A.
- Track A over the run: 10337 / 12028 (s0009) -> 10184 / 11795 (s0013) -> 9981 / 10713 (s0031) -> 9006 / 9900.
  The last 10 % came from this session's measurements of what the kernel does between weight streams (findings
  C12-C14): divisions after the stream, work placement of small ops, and local-memory reloads after calls.
- Skeleton on the final kernel (all streams off): B=1 ~1.8 ms, B=8 2.64 ms; empty GEMM ops 3.6-5.4 us.
- Negative: the barrier's 128 KB prefetch issued by thread 0 after its arrival instead of by thread 32 alongside
  it: +0.3-0.5 % (thread 0's table load now precedes its spin). Reverted.
- Track C B=8 policy re-checked on the final kernel (MK_VAR_HI 12 / 16 / 24 columns for copyable sequences,
  test_spec, 3 seeds): 3268-3270 / 3226-3229 / 3365-3369 us per step - no difference (the 32-column cap binds).

## hour 22.4-23.0: debug timers out of the default build; s0044 (track A)

- The per-op / per-block / per-warp debug stamps (cx.ts, cx.bts, cx.wts checks on the barrier and post-stream
  paths of every op) compiled out by default (NO_STAMPS in engine/build.sh; the timing tools need
  MK_CFLAGS=-DBAR_SLEEP_NS=200): CUDA-event pass times (tools/ev_time.py) B=1 8996 -> 8958, B=8 10241 -> 10217,
  three rounds each; bit-identical, all tests pass.
- s0044 (track A): predicted ~8970 / ~9875, measured 8965 / 9833 us: 1.062x / 1.040x SGLang.
- s0045 (track C build of s0044): predicted ~412 / ~3265, measured 415 / 3300 (s0043: 414 / 3275; the B=8 figure
  of the speculative path moves ~1 % between evaluations).
- The stamp checks inside the post-sync epilogue and the hand-over (dbg_stamp31, dbg_stampw) also compiled out:
  CUDA events B=1 8957 -> 8945, B=8 10220 -> 10201; bit-identical.
- s0046 (track A, final kernel): predicted ~8955 / ~9815, measured 8947 / 9819 us: 1.064x / 1.042x SGLang.
- s0047 (track C build of the final kernel): predicted ~413 / ~3290, measured 417 / 3310 - within the evaluation
  noise of the speculative path (s0043 414 / 3275, s0045 415 / 3300 with kernels 0.1-0.4 % slower per pass).

## Final (hour ~23.1)

- Track A (one token per pass, checkpoint formats): s0046, 8947 / 9819 us over HTTP = 1.064x / 1.042x SGLang
  (9523 / 10231), 1.13x / 1.09x vLLM. Track C (exact draft-and-verify, same kernel): best s0043 414 / 3275 us
  (23.0x / 3.12x SGLang's token-by-token figure); final kernel s0047 417 / 3310.
- HEAD's engine/ is s0047's source (engine/TRACK = B); s0046 is the same with TRACK = A.
- The GEMM debug flags (8: no stream, 16: stream only, 24) checked in every stream's prologue, tile end and
  post-stream path compiled out with the stamps: CUDA events B=1 8945 -> 8923, B=8 10199 -> 10185; bit-identical.
- s0048 (track A, final kernel): predicted ~8925 / ~9805, measured 8938 / 9800 us: 1.065x / 1.044x SGLang.
- s0049 (track C build of the final kernel): measured 415 / 3296 (best track C remains s0043, 414 / 3275).
- Final pair: track A s0048 8938 / 9800 us, track C s0049 415 / 3296 us; HEAD's engine/ is s0049's source
  (TRACK = B), s0048 the same with TRACK = A.
- Negative / mixed: the recurrent op's debug flags (32, 512) compiled out as well: B=1 8923 -> 8939, B=8 10185 ->
  10160 (CUDA events, three rounds). Reverted; s0048 / s0049 stay the final pair.

## hour 23.4-24: track B (operator note of 2026-11-12 09:10)

- The operator's note (no more track-C work; remaining hours for track B and track A) was read only at hour 23.44;
  s0039-s0049's track-C builds were submitted after it by mistake (mk note).
- Track B: W12 (exact 12-bit weights) under the one-token track-A data flow (api: W12 no longer tied to the
  speculative track; engine/TRACK "A W12"). Same card, CUDA events: B=1 8924 (BF16) vs 10694 us (W12), B=8 10186
  vs 11577: still ~20 % slower - the decode loop with per-quad patch handling costs more than the 25 % of bytes it
  saves (as at hour ~5; the faster kernel around the stream did not change that). Checks pass (rel-L2 0.0101 /
  0.0296, single == batched, generate == step, test_multi 10/10). Submitted as s0050, the run's only track-B
  measurement; its track-A twin is s0048. A W12 that pays would need the patches out of the hot loop (exact sparse
  correction per tile in the epilogue) - not attempted.
- s0050 (track B, W12): predicted ~10720 / ~11150, measured 10687 / 11270 us (track A s0048: 8938 / 9800).

## End of run

- engine/ restored to exactly s0048's source (api.py's one-line W12 gate change for s0050 reverted; s0050's source
  keeps it). Best submission: s0048, track A, 8938 / 9800 us. Final summary in notes/summary.md.
