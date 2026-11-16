# Findings: the card, the checkpoint, the workload

Each entry: what was measured, on what, the number, what it allows and what it does not.

## Card (NVIDIA GB200, sm_100, 152 SMs, 129 MB L2, 64 K registers per SM)

### C1. Streaming bandwidth is ~7,300 GB/s, not 8,000 (bench/bw.cu, 24 GB buffer, 152 blocks x 1024 threads)
- plain 16-byte loads, one stream per warp: 7,270 GB/s; the same feeding `mma.m16n8k16` (bf16, fp32 accumulate):
  6,930 GB/s with activations read through L1, 7,120-7,185 GB/s with activations in shared memory.
- 32 warps per SM are needed (16 warps: 4,780 GB/s); 4 loads in flight per warp are enough (1-2: 4,830 GB/s).
- Inside one persistent kernel, short bursts into new pages run at the full rate (bench/burst.cu: 60 MB in 9.9 us,
  159 MB in 26 us, 349 MB in 49 us, 996 MB in 136 us = 7,340 GB/s, barrier included). No TLB or ramp penalty per
  matrix: the per-matrix overhead of a step is in what the kernel does around the stream, not in the memory system.
- Allows: a floor of 51.25 GB / 7.3 GB/ms = 7.0 ms per step for any engine that reads the BF16 weights once per step
  (the board's 6,406 us line assumes 8,000 GB/s). Does not allow: reaching the board's bound in BF16.

### C2. L1 caches of different SMs are NOT coherent inside one grid (bench/coh.cu)
- Every block reads X, block 0 overwrites X, every block reads X again: with a default `ld.global`, 100 % of the
  re-reads of a 16 KB array (and ~16.8 KB worth of a 256 KB array) return the OLD value in all 151 other blocks.
  `ld.global.cg`, `ld.global.cv` and `ld.volatile.global` return the new value in 100 % of cases. The store kind
  (default or `st.global.cg`) makes no difference.
- Allows: a persistent megakernel is correct only if everything written by one block and read by another during the
  same launch is read with `ld.global.cg` (full-rate in the hot loop: 6,900-7,050 GB/s).

### C3. A spin-wait must use a volatile load or an atomic (bench/bar.cu)
- A grid barrier whose wait loop reads the counter with `ld.global`, `ld.global.cg` or `ld.global.cv` (inline asm,
  even with a memory clobber) never sees the counter change: the kernel hangs. With an empty loop body the loop is
  removed altogether and blocks run through the barrier (this silently "worked" in a first microbenchmark and
  gave wrong activations in the engine: blocks that had no recurrent state to update started the next matrix
  before the others had finished). `ld.volatile.global` (C++ `volatile`) or `atomicAdd(p, 0)` work: 0.82 / 0.88 us
  per barrier with 152 blocks.

### C4. 64 registers per thread, and a thread's stack is real memory (tools/prof_rep.py, SASS of the hot loop)
- One 1024-thread block per SM leaves 65536 / 1024 = 64 registers per thread. The compiler unrolled the 4-k-step
  hot loop further, ran out of registers and reloaded spills from local memory inside the loop; with
  `#pragma unroll 1` the loop is 4 x LD.128, 4 x LDS.64, 4 x HMMA and a branch.
- Structs passed by reference to a non-inlined device function live in every thread's local memory. With 155,648
  threads an empty op cost 5.2 us (the 160-byte op descriptor copied to each stack and read back), against 0.8 us
  for the barrier alone. A function with a large spill frame costs every thread that enters it, working or not
  (3 us per op). Fix: context and current op in shared memory (one copy per block), phase functions entered only
  by warps that have work.
- Allows / does not allow: a megakernel on this card must treat local memory as a scarce shared resource; "let
  the compiler handle it" cost a factor of 1.6 in step time here (16.8 ms -> 10.7 ms with nothing else changed).

### C5. Shared memory competes with L1 (bench/burst.cu)
- The same 349 MB burst takes 49.3 us with 80 KB of dynamic shared memory and 55.6 us with 202 KB (159 MB: 26.1 vs
  27.7 us). The engine therefore sizes shared memory per launch (70 KB at batch 1) and reads activations through
  L1 from buffers that are written once per launch, instead of copying them to shared memory per matrix (7 us).

### C7. Where a GEMM op's time goes in this engine (tools/op_times.py, tools/warp_tail.py, Nsight Compute)
- Per-op timer (block 0 stamps %globaltimer after every grid barrier) inside a real B=1 step: in_proj 31 us,
  out_proj 16, gate_up 57, down 33.5, lm_head 362 (2.54 GB), recurrent op 9.3, attention 5 x 4-7.
- Per-warp stream end times inside one op: every warp has finished streaming 5.6-8.7 us before the op ends
  (in_proj: slowest warp at 25.1 us of 32.5; out_proj 9.0 of 16.9; gate_up 51.5 of 57.1; down 25.1 of 33.8).
  That post-stream phase (cut-tile finalize across blocks, epilogue, grid barrier) is ~1.9 ms of a 10.1 ms step.
  The spread of stream end times between warps of one op is 3-5 us (in_proj median 22.0, max 25.1).
- Nsight Compute on a launch of 20 in_proj ops: 1 column: DRAM 5.3 TB/s, L1 hit 30 %, L2 traffic 5.3 GB for
  3.4 GB of weights, warps stalled 65 % on memory (long scoreboard) and 22 % at barriers. 32 columns: the
  activation vector (5120 x 32 x 2 B = 327 KB) no longer fits in L1: L1 hit 7.7 %, L2 traffic 16.3 GB for the
  same weights, 3.4 TB/s - why a 32-column pass costs 1.8x a step.
- L2 prefetch of the next matrix at each barrier (cp.async.bulk.prefetch.L2): -0.2 ms per step. Even out_proj with
  its whole block share prefetched into L2 only drops 17.6 -> 15.9 us: the small GEMMs are not DRAM-bound.
- Allows: a ~25 % faster track-A step if the post-stream phase and the 32-column L1 problem were solved (blocks
  whose warps sweep K together and finish their tiles inside the block). Not done in this run.

### C8. Register spills, not arithmetic, set the speed of several phases (ptxas -v, SASS loop listings)
- 64 registers per thread (1024 threads): a spill reload in a loop goes to L1, which is small and thrashed when
  32 columns are streamed (137 KB shared memory), so each reload can cost an L2 round trip.
- Measured cases: the split recurrence loop had 19-32 spill reloads per position because the block half was a
  run-time value and the state's life sat inside conditionals (ptxas 720 / 492 bytes); with the half as a
  template parameter and the idle threads leaving at the block barrier: none, 25.6 -> 16.7 us per layer at 32
  positions. The residual epilogue at 32 columns: one reload per column group, ~1 us each (6.6 -> 2.6 us after
  one instance per epilogue kind and use). A double division (floor) is a subroutine call that makes its caller
  spill; a float estimate with an exact integer correction does not.
- The unsplit recurrence is issue-bound, not latency-bound: 147 SASS instructions per position x 32 warps / 4
  schedulers ~ 0.62 us per position, measured 0.67.
- Allows: reading ptxas -v and the SASS of every hot loop after each change (tools/spills.py, tools/sass_loops.py).

### C9. Barrier, prefetch and fixed costs per op (tools/empty_op.py, tools/tail_p.py)
- A GEMM op with its stream switched off costs 4.4-5.4 us at 1 column, 4.9-8.5 at 32 (prologue + barrier).
  After the last block arrives, the grid barrier releases 1.0-1.5 us later without the L2 prefetch of the next
  matrix and 1.3-3.6 us with it: the prefetch traffic delays the release. Moving the prefetch away from the
  barrier is still worse (+1.7 to +2.9 %): DRAM work during the wait is worth more than the latency.
- At 32 columns the extra cost of a GEMM over 1 column was (per op) stream +5 to +10 us, tail +3.5 to +7, prologue
  +1 to +4. Shortened by: partials of all column groups loaded at once, spill-free epilogues, the block's
  outgoing cut tile finalized by a free warp, residual rows copied with cp.async, the RMSNorm scalar without
  block barriers above 4 columns. Not shortened by: flag hand-over instead of counter + last arriver, a larger
  or earlier prefetch, shared memory below the 132 KB carve-out step, L2-only loads of weights ahead.
- In isolation (bench/ng2.cu) 32 columns cost only +6-11 % over 8 (activation traffic ~5 %); 512 threads with 2
  tiles per warp did not consistently remove that; 64 columns at 512 threads cost +30-40 % over 32 at 1024.

### M7. A fully accepted pass needs no replay of the recurrent state (tools/verify_ops.py, tools/test_spec.py)
- A verify pass leaves the state untouched and saves (decay, k, delta) per position; the next pass replays the
  accepted ones. On looping text every position is accepted, so every pass replayed 32 rank-1 updates per head and
  layer (2.9 ms of a 16.5 ms pass went to the recurrent op). Each verify pass now also writes its final state to
  a second buffer; when all positions were accepted the accept op flips the sequence's buffer index instead.
  Bit-identical (the replay was the same fused multiply-adds in the same order); 571 -> 548 us per token on the
  looping prompts, +0.6 % on prose (the extra state write).

## Workload (continued)

### W2. Policies measured on the timed prompts (tools/test_spec.py, the client's window)
- The B=8 figure counts tokens while every stream decodes (until the first stream ends), so aggregate tokens per
  pass matter, not the slowest sequence: columns to the looping sequences (up to 16, prose 1) 4448-4618 us vs
  4547-4688 (8 / 2); "prose first" 5523-5975.
- Copy-drafter match length 4 instead of 6: one ramp pass less on the looping prompts (582 -> 571 us); prose and
  B=8 within 0.5 %. A looping B=1 decode: 7 ramp passes of 4 positions (16 tokens), then 31 passes of 32.

### W3. Drafting policy around the ramp and established loops (tools/test_spec.py, tools/pass_trace.py)
- A pass's length is fixed when the previous pass is launched (that pass's accept op writes the drafts). Queuing
  the first 8 passes of a cohort one at a time, and (B=1) choosing each of them at launch from the previous
  pass's result, with room for 31 copied drafts in every pass: the looping prompts' ramp went from 7 short passes
  (16 tokens) to 4 (6 tokens): 487 -> 457 us per token over HTTP-equivalent timing, prose unchanged.
- While no sequence needs MTP drafts for the next pass (copied drafts accepted in full, or a single position at
  B=8), verify passes run without the MTP head (0.33 ms of a 13.9 ms pass) and copies are taken without the MTP
  agreement check: 478 -> 468 us (loops), B=8 -1 to -2 %, every token still verified. Cost: the MTP layer's K/V
  is not written for those positions (draft quality only, after leaving a long loop).
- A 2-token copy match would give loops 446 us but prose +2 % and B=8 +2.5-4 %: kept 4 tokens.

### C10. What a pass costs per number of positions (tools/op_times_p.py, B=1, context 600, final kernel)
- 1 position 10.11 ms, 2: 10.22, 4: 10.37, 8: 10.55, 12: 11.21, 16: 11.28, 24: 13.10, 32: 13.43 ms (single steps
  of B sequences: B=2 10.28, B=4 10.34, B=8 10.91). The steps follow the column groups of 8 (1, 2, 4 groups of
  the MMA): 8 positions cost 4 % more than one, 32 positions 33 % more, so a verify pass pays for itself from
  ~1.05 accepted tokens at 8 positions and ~1.33 at 32. With the 32-column cap a looping B=1 sequence decodes at
  13.4 ms / 32 tokens = 0.42 ms per token (HTTP 457 us including the ramp); the cap is the design's
  (accumulator registers, tile-sum slots, activation traffic all scale with the columns).

### C11. A weak load does not see another SM's later store - not even ld.global.cg (tools/sent_dbg3.py)
- Sentinel hand-over of cut tiles (idea from the Flint run): the first block stores its partial into a slot that
  holds a NaN sentinel, the last block polls the slot until the data is there, then puts the sentinel back. With
  the poll as a plain ld.global.cg (a weak load) 16-35 % of the receivers read the PREVIOUS op's data (epoch-tagged
  debug build: senders = receivers, identical (op, slot) sets, no duplicate slots - and still stale), giving
  garbage logits. With ld.relaxed.gpu / st.relaxed.gpu (morally strong) for the poll, the data and the reset:
  0 stale reads, all invariants hold. Everywhere else the engine reads other SMs' data only after an acquire
  (grid barrier, acq_rel counter), which is why its weak loads are safe.
- Rule: a spin on data written by another SM in the same launch must use relaxed/acquire loads at gpu scope; a
  cache hint (.cg) is not a coherence guarantee on this two-die card.
- Gain once correct: one trip through the memory queue per cut tile instead of three (store, release + atomic,
  load): -1.4 % per 32-position pass, -1.6 % per B=1 step, -1.9 % per B=8 step, bit-identical.

### C12. After a GEMM's stream, integer work and timers, not memory, held the hand-over (tools/ho_stamps.py)
- Per-block %clock64 stamps (build -DHO_STAMPS; %globaltimer ticks in ~0.5 us steps and is useless below that):
  the warp that receives a cut tile entered the hand-over 1.5-1.9 us after its block sync and the sender 2.2 us
  after, before any memory access: every warp rebuilt the partition with 64-bit floor divisions (I2F.S64, 64-bit
  multiplies) at the same moment, searched the tile's first warp serially, and the receiver then waited for the
  late sender. Quad bounds published in shared memory before the stream, one ballot for the first warp, roles
  decided at each warp's own stream end: block sync -> barrier arrival 3.3-4.2 us -> 1.9-2.7 us; B=1 step -4.3 %,
  B=8 -3.8 %, bit-identical.
- Not the cause: instruction-cache misses (Nsight Compute: no_instructions 0.6 % of stall samples), local-memory
  spills on the sentinel path (none in its SASS), the RMSNorm partial loads of the first warps (removing them:
  no change).
- Allows: on this card a few hundred integer instructions per warp, issued by all 32 warps at once, cost
  ~0.2-0.3 us; anything the critical path can precompute while the stream runs should be.

### C13. Work placement: consecutive warps put small ops on few SMs (att_w(), tools/op_times_p.py)
- Units of the attention ops were given to warps 0, 1, 2, ... of the grid, i.e. to the first blocks: at B=1 the
  152 key tiles of the score op ran on 5 SMs and the 32 prep units on one, bounded by those SMs' L2 bandwidth.
  Round-robin over blocks first: prep 133 -> 72 us, score 112 -> 69 us per step (-0.6 %).
- The same for dependent loads at op start: the recurrent units' state-buffer index (a global load before
  anything else, constant within a launch) read once per launch into shared memory: B=8 recurrent op 1218 ->
  1145 us.

### C14. A device-function call costs a local-memory round trip on this kernel (ptxas -v, same-card A/B)
- With 64 registers per thread a __noinline__ call saves and restores registers through local memory (ptxas lists
  it as spill: 112 bytes for a 20-line helper). After a weight stream the L1 holds other lines, so the restores
  miss: one more call level on the GEMM hand-over path made every pass +4.5 % slower. Inlining instead moves the
  pressure into the caller (finalize inlined into the stream: 352-1200 bytes of spill in the stream function).
- The same mechanism in the kernel's main loop: the program pointer, op count, op index and thread indices were
  kept across every op's call in local memory and reloaded after it, right before the grid barrier (SASS: 28 local
  loads in the loop body). Re-read from shared memory and special registers instead: -2.5 to -2.7 % per pass at
  every shape, bit-identical (s0042: 9258 -> 9006 us at B=1, 10065 -> 9900 at B=8 over HTTP).
- Allows: keep the post-stream path flat, with as few call levels as the 64-register budget permits; measure every
  new helper on the critical path with ptxas -v and an A/B, never assume a call is free; after any change, list the
  local loads that follow a CALL in the SASS (tools/sass_fnmap.py, tools/sass_ldl_after_call.py).

## Checkpoint

### M1. The MTP head predicts the main model's own greedy token (tools/mtp/RESULTS.md, measured by a subagent)
- Forward (determined empirically, each choice by accuracy): x = fc(cat[rms(embed(tok[s+1])), rms(h[s])]) with h the
  main model's output after the final norm, one full-attention decoder layer with its own K/V history, mtp.norm,
  shared lm_head; all MTP norms centred (1 + w); chained drafts feed the post-norm MTP hidden back.
- Draft 1 equals the main model's greedy token at 84.0 % of 7,154 natural-text positions, 82.5 % on the
  evaluator-style workload. Wrong variants: concat order swapped 0.2 %, no attention history 65 %.
- Simulated exact speculative decoding on 64 evaluator-style prompts x 1024 greedy tokens: 1.86 / 2.97 / 3.88 tokens
  per verify pass with 1 / 3 / 7 chained drafts; on non-looping continuations 1.75 / 2.40 / 2.63.
- Allows: several tokens per pass of the full model, exactly the greedy tokens, if a verify pass over P positions
  costs about one step (it does when the step is memory-bound: the extra positions are extra MMA columns).

### M4. Exponent census: 15 consecutive exponents per tensor hold all but 1.86e-4 of the weights (tools/census.py)
- Exact counts over all 26,046,627,840 weights of the 2-D matrices the engine streams (text decoder without the
  embedding table, lm_head, and the MTP head): per tensor, the best window of 15 consecutive BF16 exponent fields
  misses 4,837,918 weights (1.857e-4; per kind 1.5e-4 for up_proj to 4.1e-4 for mtp.fc). There is no exact zero in
  any matrix. The exponent field spans up to 39 values inside one tensor (86..127 over the model).
- Entropy: 2.56-2.65 bits for the exponent field, 10.48-10.57 bits for the whole 16-bit value (per kind).
- Allows: the known 12-bit format at 12.012 bits per weight (12 + 1.857e-4 x 64-bit patch entries), the figure the
  SpecPilot paper reports for other models. Does not allow much more: a fixed 11-bit code would have to escape
  several percent of the weights, and the entropy bound (10.5 bits) is only reachable with variable-length codes.

### C6. The card decodes 12-bit weights faster than it reads 16-bit ones (bench/w12.cu)
- Same stream loop, 3 x 16-byte loads per thread and quad instead of 4, each pair of weights rebuilt with integer
  ops (bf16 = code + C + sign * 0x7800 after spreading two 12-bit codes into two 16-bit halves): 4.00-4.55 M quads
  per ms against 3.26-3.68 M for BF16 (+23 %), i.e. 8,200-9,300 GB/s BF16-equivalent. The 160-odd integer
  instructions per quad are not the bottleneck; bytes still are.

### M2. A pass over P positions costs about one step and is bit-identical to P steps (tools/test_multi.py, tools/test_spec.py)
- Architecture facts used: (a) every weight is read once per pass whatever the number of columns, and the MMA
  instruction takes 8 columns for the price of one; (b) the 48 linear-attention layers keep a 128 x 128 float32
  state per head whose update is rank 1, S <- d S + k (x) delta: a pass can leave the committed state untouched,
  save (d, k, delta) per position and replay the accepted ones later with the same fused multiply-add in the same
  order; (c) the 16 full-attention layers only append K/V, so a rejected position is forgotten by resetting the
  length; (d) the 4-tap causal convolution needs the last 3 inputs: a ring of 16 positions per channel.
- Measured: logits of a multi-position pass equal the logits of single steps bit for bit for (B, P) = (1,2) (1,3)
  (1,8) (2,4) (2,8) (3,6) (4,8) (8,2) (8,4), 48 positions each. 1024-token decodes of the evaluator's three timed
  prompt sets through draft-and-verify are identical to step-by-step greedy, token for token, at B=1 and B=8.
- Measured cost and yield (in process, before later tuning): B=1, 8 positions per pass: 7.7 tokens per pass on the
  two looping prompts (2,158 us per token), 3.0 on the prose prompt (5,401 us); B=8, 4 positions: 2.2-2.4 tokens
  per pass and sequence (8,240-8,640 us per step); B=8, 2 positions: 1.7 (8,520-8,630 us).
- Allows: tokens per second several times above the "read every weight once per token" bound without changing a
  single emitted token. Does not allow: more than P tokens per pass; at B=8 the activations of 32 columns no longer
  fit in L1 (328 KB per matrix) and the pass itself slows down, which is why P=4 gains little there.

### M3. HF's own multi-token forward is not argmax-identical to its single steps (tools/mtp/RESULTS.md)
- Chunked vs recurrent forms of the delta rule and different summation orders: the greedy token differs at 0.44 %
  of generated positions (40 of 64 sequences) in HF BF16. A verify pass that used "the multi-token forward" as a
  general engine would have to would not reproduce step-by-step greedy; this engine's pass computes each position
  with the single-step arithmetic, which is what makes the speculation exact rather than "accurate enough".

### M5. Draft-and-verify on this checkpoint and workload: where the tokens per pass come from (tools/test_spec.py)
- Drafts come from the checkpoint's MTP head (at most 4 per pass, chained) or are copied from the context when the
  last 6 tokens occurred before and the copied continuation agrees with the MTP head's first draft. Verification is
  exact (logits of a P-position pass are bit-identical to P single steps for P up to 32; 1024-token decodes are
  token-identical to step-by-step greedy on the evaluator's timed prompts at B=1 and B=8).
- Measured on the evaluator's three timed B=1 prompts, 1024 tokens: 25.6 tokens per pass on the two looping ones
  (32 positions per pass), 2.45 on the prose one (8 positions per pass, MTP drafts). At B=8 (4 positions per
  sequence per pass): 2.2-2.4 tokens per sequence per pass.
- Pass cost (in process): 1 position 10.2 ms; 8 positions 11.1 ms; 16 positions 14.1 ms; 32 positions 18.5 ms.
- Allows: on repetitive greedy output a single sequence decodes at ~0.8 ms per token, 12x a general engine, with
  identical tokens. Does not allow: much on prose at B=1 (2.4 tokens per pass x 10.2 / 11.1 ms) - the MTP head's
  acceptance, not the engine, limits it there.

### M6. Per-sequence verify depth in a batch (tools/test_spec.py, per-sequence statistics)
- At B=8 with 4 positions per sequence per pass, the evaluator's timed prompt sets contain 3-4 sequences that
  loop (3.95 of 4 positions accepted, drafts copied from the context in 98 % of passes) and 4-5 prose sequences
  (2.2-2.8 accepted with MTP drafts). Giving copyable sequences up to 8 positions and the others 2-4 within the
  same 32 columns: loops 5.7-7.3, prose 2.0-2.9 tokens per pass; 5629 / 5667 / 5835 us per step instead of
  6368 / 6386 / 6891 (in process), token-identical decodes.

## Workload

### W1. About 40 % of the evaluator's greedy continuations are a two-token loop (tools/mtp/RESULTS.md)
- 26 of 64 prompts of `prompts.make(tok, 64, seed)` continue as "The\n\nThe\n\n..." from the first token, 2 more loop
  with period 30/31. Exact BF16 top-1 ties occur at 0.99 % of generated positions (HF BF16).
- Allows: nothing by itself (no output may be cached or recognised); it explains why drafts are accepted more
  often on this workload than on prose, and why tie-breaking must be identical in every path of the engine.
