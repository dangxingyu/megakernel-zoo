# Findings of run s9 (the earlier runs' findings: findings_s5.md, findings_s4.md)

Each entry: what was measured, on what, the number, what it allows and what it does not.

### S1. Dirty recurrent state costs ~2x its bytes when L2 writes it back (tools/b8_split.py, stamped build, same card)
- Per-op times inside real one-token steps, inherited kernel: gate_up after a recurrent op 54.7 us at B=1, 57.5 at
  B=4, 60.3 at B=6, 62.1 at B=8; gate_up after attention 53.4-55.0 at every B. With the recurrent state not stored
  (debug flag 128, wrong results) gate_up at B=8 is 54.9 us and the recurrent op 14.3 instead of 17.2; with the state
  not loaded (flag 256) nothing changes in gate_up. The same 40 gate_up ops alone in a launch: 53.6 (B=1) / 54.6 (B=8).
- So: the 24.6 MB of float32 state a B=8 step writes per recurrent layer sit dirty in L2 until the next large weight
  stream (gate_up, 356 MB) evicts them; their write-back takes ~7.1 us of that stream (0.29 us per MB = 3.4 GB/s,
  against 7.3 GB/ms for reads) plus ~2.9 us inside the recurrent op: ~485 us of a 9.8 ms B=8 step.
- Allows: any way of writing less state per step is worth ~0.4 us per MB not written. Does not allow: explaining
  the B=8 cost of gate_up by its 8 activation columns (the earlier runs' open question): that cost is zero.

### S2. The cache operator of the weight loads decides 2.5-3.4 % of a step: ld.global.cs (tools/abn.sh, same card, bit-identical)
- The GEMM streams read every weight with ld.global.cg ("cache at global level": L2, not L1) since the first run. One
  token changed, same-card alternating A/B (events, us per one-token pass, B=1 / B=8), logits hash identical in all:
  cg 8878 / 9785 | cs ("cache streaming, likely to be accessed once") 8658 / 9452 | lu ("last use") 8662 / 9464 |
  cv 8877 / 9706 | ca (default) 8876 / 9710 (first round 8975 / 9833).
- Reading: with cg, 51 GB of weights per step pass through the 129 MB L2 as ordinary lines and push out everything
  else a step touches once (activation buffers written by one op and read by the next, descriptors, norm gains,
  conv history, recurrent state); with cs the weight lines are the ones that leave first. Not measured directly (no
  L2 counters without Nsight Compute); what was measured is the step time.
- Allows: -220 us at B=1 and -330 us at B=8 on track A, and the same for the 12-bit stream of track B (same ldw).
  Earlier negative results about L2 (prefetch sizes, far prefetch, cold loads) were measured in the cg regime: they
  are hypotheses again.

### S3. Small DRAM loads issued behind a prefetch burst wait for the burst (lmix_dual, -DLMPF_LATE; same card)
- The B >= 4 recurrent op asked L2 for all its units' states (64 KB each, 24.6 MB per layer at B=8) before the first
  unit's input loads (conv history rows, and the saved updates of F_DEFER) were issued. Issuing the state prefetch
  after the first unit's input loads instead: B=8 9778 -> 9697 us per step (stores every step), 9632 -> 9530 with
  deferred commits (cg regime). Phase timer before the change (block 0, per layer): input phases 4.4 us without and
  7.0 us with saved updates to load.
- Allows: order DRAM requests by need: latency-critical small loads first, bulk prefetch after.

### S4. Negative results of this run (each a same-card alternating A/B against the same source without the change)
- Per-warp early prefetch (each finished warp asks L2 for the start of shares of the next GEMM, 4-16 KB per share,
  instead of the block's first 128 KB at the barrier): B=1 +390 / +540 / +770 us for 4 / 8 / 16 KB. Prefetch
  requests issued while the block's slower warps still stream delay those warps (bench/variants/mk_pfw.cu).
- Next op's descriptor preloaded one op ahead (no global load in the barrier): +125 us at B=1. Prefetch of the norm
  gains the next epilogue reads: +20 us; of the x lines it writes: +30 us (bench/variants/mk_cold.cu). All in the cg regime.
- Compact one-column activation layout (-DX_COMPACT) for the BF16 engine: B=1 -16 us, B=8 +95 us (the extra
  instantiations): not taken.

### S5. The B=8 recurrent op was bound by per-round latency, not by work (lmix_tri; tools/lmprof.py, same-card A/B)
- lmix_dual at B=8 (384 units on 152 blocks): two rounds of two units per block (the second on blocks 0-39 only);
  block 0 per layer: inputs 3.4 us, recurrence 9.2, norm 1.3. Each round has its own input phase (cold conv-history
  loads) and four half-block barriers. Earlier attempts to balance the units (a third round of single units on whole
  blocks; four quarter-units per block) were neutral: they kept one latency chain per round or per unit.
- lmix_tri: all units' inputs in one phase (a thread issues the loads of two units together), the own unit's two
  state columns, then one column of a third unit shared by both halves of the block (so no thread makes more than
  three column passes), one norm phase. Same per-column arithmetic: bit-identical. B=8 9376 -> 9244 us per step
  (-2.75 us per recurrent layer), B=7 -119 us.
- Allows: when an op is a chain of short latency-bound phases, merge the phases of its units before trying to
  balance or speed up the units. Still there: 13.4 us per layer at B=8 against 8.8 at B=1.

### S6. Prefetch schedule around small ops, and blocks that idle (host tables only; same card, bit-identical)
- With cs weight loads the barrier-time prefetch is worth 6-7 % of a step (none: 9279 / 10019 us at B=1 / B=8 against
  8664 / 9221 with 128 KB) and wants to be larger than with cg loads: near 160 KB + far 64 KB: 8622 / 9140.
- Consecutive pieces at the barriers before small ops (instead of the same first bytes): 8621 -> 8605 / 9168 -> 9150.
- At B=1 the recurrent op uses 96 of 152 blocks. The last 32 idle blocks asking for their whole out_proj share (414 KB)
  at the barrier before it: 8612 -> 8579; all 56: 8588; 16: 8591. More prefetch traffic at that moment delays the
  recurrent op's own DRAM loads (S3), so it is a trade, not "use every idle block".
- Not a gain: issuing the 64 KB pieces from two or three warps at once (+5 us); per-warp early prefetch (S4).

### S7. The grid barrier's release costs ~0.45 us per op and cannot be dropped (bench/rel.cu; tools/abn.sh)
- The barrier adds to its counter with red.release.gpu and polls with ld.acquire.gpu. With the add relaxed
  (red.relaxed.gpu), poll unchanged: B=1 8564 -> 8407 us per step, B=8 9147 -> 8951 (-0.4 / -0.5 us per barrier, 390
  barriers per step), logits of 12 steps identical; with only the poll relaxed: no change. The cost is the release:
  the add waits until the SM's earlier stores are globally visible.
- Is that wait needed? bench/rel.cu: 152 blocks, each stores an epoch into its 4096 words from all warps,
  __syncthreads, thread 0 adds, every block polls and at once reads another block's words with ld.global.cg.
  200,000 iterations (1.25e11 reads): release add: 0 stale reads; relaxed add with a 64 KB or 256 KB load burst in
  flight per block: 0; relaxed add with no other traffic: 864 stale reads; negative control (add before the stores):
  7.4e6-3.2e8. So a relaxed atomic can overtake the same SM's earlier weak stores, rarely, and exactly when the
  memory system is idle: a short logits check does not see it.
- Does not allow: a cheaper grid barrier by relaxing its atomics. Allows: hand-overs on ONE location (a slot that
  holds a sentinel, written and polled with relaxed gpu-scope stores and loads, s4 C11) stay the cheap exact form;
  anything "data here, flag there" needs release / acquire, ~3 us per hand-over in a small op.

### S8. One op per attention layer pays only when no block streams much and no hand-over needs a strong atomic
- The four attention ops are overhead: 22-28 us per layer at B=1 and B=8, nearly independent of the context (24.3 us
  at context 100, 24.9 at 1000 at B=1). Four forms of a single op were built, each bit-identical to the four ops over
  1150 steps (tools/att_check.py); per-op times in journal.md. What decided them:
  - one block per (sequence, kv head) streams 0.6-1.1 MB of K and V through one SM: ~64 KB per load round trip of
    ~1 us with 4 loads in flight per warp; 8 in flight and two warps per dim tile helped (25.7 -> 19.2 us at B=1,
    context 600) but B=8 gained nothing;
  - one block per 256-key chunk (131 KB of K and of V per block whatever the context) with acquire/release counters
    for its three hand-overs was no faster: each strong atomic or fence costs ~3 us in a small op (phase timer);
  - with sentinel slots (one location, relaxed gpu-scope store and poll, restored by the reader) and the K/V append
    read back through relaxed loads: 16.5-19.7 us at B=1, 15.6-26.4 at B=8.
- In the engine (same card, events, context ~620): B=1 8585 -> 8566 us per step, 8520 with the blocks' whole o_proj
  share prefetched at the barrier before the op (at B=1 only 4-20 of 152 blocks work in it); B=8 9126 -> 9094.
- Allows: -0.8 % at B=1, -0.35 % at B=8. The op is still 3-4x its arithmetic; what remains are its ~9 dependent
  memory round trips (prep loads, K, score slots, (max, 1/sum) slots, V, value-sum slots, output) at ~1-2 us each.

### S9. Work at the start of a GEMM belongs to the warps whose shares were prefetched (B=8; same card, bit-identical)
- Every GEMM leaves warps 0-3 of a block 4-6 us of slack (their shares are the block's prefetched 160 KB; tools/
  warp_map.py), while the block's time is set by the other 28. At 5-8 columns the RMSNorm scalar's partial sums were
  loaded by every warp before its stream (10 global loads per lane), in the 113 normalised GEMMs of a step.
- All partials by warps 0-3 instead (two (row, column) pairs per thread, the same sums in the same order):
  B=8 9088 -> 9048 us per step inlined into the main loop (which went from 36 to 116 bytes of spill), 9037 as its own
  function (36 bytes again); B=7 9099 -> 9037. With warp 0 also finishing r at once (the 32-way reduction the first
  epilogue after the stream otherwise makes while the others spin): 9004 / 9010.
- The inverse experiment: R_ASYNC_MAX = 8 (warps 0-7 compute r, one column each, the others start at once) is +115 us
  at B=8: warps 4-7 have no slack. At B=5 the same switch is -30 us (warps 0-4).
- Not a gain: the next norm's gains copied to shared memory during the stream for the residual epilogue (-8 / -3 us,
  with +6 us for the code's presence): they are L2 hits since the cs change.
- Allows: -0.9 % at B=8 from moving ~1 us of per-op prologue off 28 warps. Applied in the B >= 7 kernel only (the
  other kernel's code generation is B=1's).

### S10. The B=8 recurrent op after lmix_tri: where 13 us per layer go and what moved (bench/variants/mk_triprof.cu)
- Phase timer, block 0 (three units), us per layer: inputs 3.27, normalise 0.22, own unit's two state columns 4.78,
  shared unit's column 2.19, norm + output 0.73. A column pass is a 16-load round trip (~1.5 us), ~0.6 us of
  arithmetic and 16 stores; the state of a layer (24.6 MB at B=8) needs 3.4 us of DRAM whatever is done.
- Next column's loads in flight during the current column's arithmetic (k read from shared memory twice instead of
  held in 16 registers): -29 us per step at B=8, -46 at B=7.
- The three conv-history rows per sequence (cold: written 1-3 steps ago) asked from L2 during the preceding in_proj:
  as 24 one-row requests from one warp +85 us (that warp became its block's straggler); as one 60 KB request per
  sequence from one thread of block b: -21 us at B=8, -32 at B=7 (-13 after integration). Conv weights the same way:
  no gain. The own units' states asked for one barrier earlier (host table): +18 to +28 us.
- The same history prefetch in the kernel of the other batch sizes: B=1 +27 us, B=2 +9: not there.

### S11. Ask at the barrier before a small op for what the op reads first (host table; same card, bit-identical)
- Phase timer of the one-op attention layer (stamps of the recurrent ops off): the scores phase is its largest
  (4.5 us of 15 at B=1, 7.0 of 21 at B=8): the K tiles were requested by the op itself at its start.
- Each attention block's entry of the barrier-time prefetch table (the barrier after the qkv GEMM) changed from a
  weights piece to its 131 KB K chunk: B=8 8937 -> 8880 us per step (BF16 engine), 8168 -> 8113 (12-bit engine);
  B=5 -54 us; B=1 8530 -> 8526 (no difference: at one sequence the K cache stays in L2 between steps).
- The same idea for the recurrent state (the block's two own units, 128 KB, at the barrier before the op): +18 to
  +28 us at B=8: there the request displaces the conv-history and input loads that the op needs first (S3).
- Allows: per-(op, block) prefetch entries are free to tune (no kernel code, no code-generation risk); what to ask for
  is whatever the next op's first dependent load is, if nothing smaller must come before it.

### S12. A block's prefetch requests must not be issued alongside its barrier increment (same card, bit-identical)
- At every grid barrier thread 32 of a block issued the block's prefetch requests (160 KB before a GEMM, up to 414 KB
  at small batches) while thread 0 issued the counter increment (red.release). Final per-op profile at B=1: qkv had
  become 2.2 us longer after the barrier following it got the whole-share request (S8): the op's time runs to the
  barrier's release, and the increment of the last block was queuing behind that block's own prefetch lines.
- Thread 32 now waits (a flag in shared memory) until thread 0's increment is out, then issues the requests:
  B=1 8529 -> 8504 us per step, B=2 -31, B=8 8886 -> 8877 on one card; -27 to -31 / -9 to -12 on another; 12-bit
  engine 7702 -> 7686 / 8114 -> 8103.
- The earlier runs had measured "the prefetch traffic delays the release" (s4 C9) and moved the whole prefetch after
  the release or onto thread 0 (both worse); ordering it behind the increment only had not been tried.
- Second step: thread 32 issues one 64 KB piece per poll of thread 0 (a counter in shared memory) and stops when the
  block has seen the release: the block's own polls no longer queue behind all of its prefetch lines, and a block
  that arrives late does not go on asking after everybody may run. B=1 8505 -> 8463 us per step, B=8 unchanged, on one
  card; B=1 -19 to -35, B=2 -47 to -60, B=8 -3 to -27 on another; 12-bit engine 7687 -> 7662 / 8102 -> 8090.
- Not a gain: the last block to arrive skipping its prefetch altogether (it needs a poll before the go: +36 / +57 us).
