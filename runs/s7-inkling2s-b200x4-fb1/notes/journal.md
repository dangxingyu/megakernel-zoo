# Journal

## 0.0-0.5 h: bring-up

- Pod facts: 152 SMs/card, 227 KB smem/block, 126 MB L2, 955 GB host RAM (the 393 GB of weights stay in the page cache:
  the starter loads in ~2 min, my loader 48 layers in ~50 s with one thread per card).
- sm_100a has no warp-level block-scaled MMA (`mma ... kind::mxf4nvf4.block_scale` is sm_120a only); only tcgen05.
  `cvt.rn.f16x2.e2m1x2` and `cvt.rn.f16x2.e4m3x2` exist but need `-gencode arch=compute_100a,code=sm_100a`
  (plain `-arch=sm_100a` emitted compute_100 PTX and ptxas rejected the e2m1 cvt).
- F2FP.E2M1.UNPACK_B reads a byte selector (.B1/.B2/.B3) for free when the PTX unpacks with `mov.b32 {b0,b1,b2,b3}`;
  with `(w >> 8) & 0xff` ptxas emits a LOP/SHF per register.
- GEMV microbenchmark (tools/bench/gemv2.cu, one card, 300 MB, m16n8k16 MMA, fragment-ordered weights):
  BF16 6.2 TB/s (1024 threads, 2-unit unroll), NVFP4 (cvt + HMUL2 by E4M3 scale) 4.4 TB/s with byte extraction,
  5.9 TB/s with free byte selects and H0_H0-swizzled scale multiplies. 512 threads/block: 5.8 / 4.5 TB/s.
- First engine (6 layers, B=2, 3 steps teacher-forced) vs the starter's fork path: routes identical at step 0 in all
  6 layers; attention outputs rel 1e-3..2e-2, MLP outputs rel 1e-2..8e-2 growing with depth; the four cards' replicated
  state bitwise identical.

## 0.5 h (pod outage, clock stopped): structure changes written before any timing

- Routing (top-8 + union) computed redundantly by every block from the router logits: -2 grid barriers per layer.
- Per-token step state (8 finished attention-residual blocks, the partial block, the attention output) lives in the
  shared memory of the block that owns the token: these vectors exist only within one step (only K/V and the conv
  histories persist), so they never need global memory.
- B=1 kernel instance (`ink_step_red`): every block computes the per-token vector phases itself (mlp/attn all-reduce
  sums, convs, residual mixes, norms, NVFP4/MXFP8 input quantization, w2 input quantization) into compact
  one-column MMA fragments in its own shared memory: the GEMVs read their B operand from shared memory and three
  grid barriers per layer disappear (after attn_in, after post_attn, after quant2). Conv histories double-buffered by
  step parity (one writer, many readers). B>=2 keeps one block per token + barriers (`ink_step_std`).
- L2 prefetch (cp.async.bulk.prefetch.L2) of the next GEMV's weights by blocks idle in latency phases.

## 0.5-1.7 h: first full-model runs (pod epochs 3-4, long outages in between)

- Correct from the first full run: argmax identical to the fork reference at 12/12 steps x 2 sequences, the 4 cards'
  replicated state bitwise identical, online submit/poll and generate() == step-by-step greedy at B=1/2/8.
- Own float64 check (tools/ref_dump.py --mode high, 2 sequences x 12 positions, tools/t_gate.py): engine vs float64
  centred rel-L2 median 0.050, max 0.10, top-1 100%; the fork path on the same positions 0.054 / 0.11. Limits 0.179 / 0.707.
- Speed far below the byte count: 10.7 ms/step at B=1 first (222 us/layer). Causes found with in-kernel globaltimer
  stamps (MK_TPROF):
  * remote atomicMax for the w13 amax: ~18K NVLink atomics per card per layer onto 8 addresses -> warp max + local
    atomic + one remote store per (expert, card) by the expert's last strip: w13 47 -> 23 us.
  * register spills (stack frame 320 B/thread x 1024 threads = 327 KB per SM vs ~34 KB of L1 next to 222 KB of
    shared memory: every spill access went to L2) and pointer arrays with dynamic indexing in the residual mixes and
    the top-8: mixes 16.7 -> 4.0 us, top-8 13.5 -> 3.4 us after keeping per-source scalars in shared memory.
  * B=1 mode race: GEMVs read shared-memory fragments written by other warps without a __syncthreads (the grid
    barrier hid it at B>=2) -> runs differed from step 0; fixed, B=1 and B=8 now bitwise reproducible.
- s0001 deadlocked in the evaluator's time stage: the per-step header was copied from a 64-slot pinned ring and the
  host ran >64 launches ahead (256 steps without a sync) -> headers overwritten before their async copy -> cards
  waited for counter values that never came. My tests never queued >64 steps. Fixed (header as kernel argument);
  tools/rehearse.py now reproduces the evaluator's unsynchronised loops and the HTTP path before any submission.

## s0002 (4.6 h): first valid submission, track A
- Expected ~6700 / ~10400 us (B=1 / B=8 over HTTP); measured 6564 / 10330. Gates: median 0.078 (limit 0.179), max 0.31
  (0.707); long 0.059 / 0.46, top-1 98.5% (limit 97.6%); http tokens 8/8 identical; generate 8/8.
- What moved it from s0001's 8.3 ms: (1) the per-step header passed by value (deadlock fix); (2) cross-card signal:
  one system fence then relaxed reductions instead of four release reductions per block (o_proj+signal 19 -> 8 us,
  w2+signal 22 -> 10 us; a release reduction to peer memory costs about a system fence each, ~2 us measured); (3) B
  fragments staged in shared memory in the B>=2 kernel (w13 at B=8 67 -> 47 us); (4) history loads and finished-block
  mix statistics before the cross-card waits.
- Primitive latencies (tools/bench/prim.cu, e0029): grid barrier 152 blocks 1.25 us; dependent L2 load 0.15 us warm,
  0.21-0.42 us after streaming 8 GB; fence.sc.sys after remote writes 2.05 us; fence.gpu 0.53 us.
- Instruction cache is not the problem (routing phase 4.6 us cold vs 4.0 us warm, -DICTEST).

## 5.4 h: s0003 submitted (queued by a pod loss); findings since s0002 (all with in-kernel globaltimer stamps)
- Strided GEMV warp mapping (logical warp = warp x 152 + block) made the same ~6-10 blocks 2x slower in every
  expert GEMV (per-block end times correlate 0.94-0.99 across layers and runs; the slow blocks' indices form
  arithmetic progressions, their SM ids look random -> a mapping effect: a block's 32 streams sit 729 KB apart in
  lockstep). Block-contiguous ranges when all warps are busy: B=8 w13 per-block ends balanced (31.7-35.1 us).
- Block-per-strip GEMV (32 warps split one strip's K, shared-memory reduction, no global hand-off) wins for qkv,
  o_proj and w2 (w2 13.7 -> 9.6 us, qkv 12 -> 10.5) but loses badly for w13 (21.7 -> 29.5; B=8 56 -> 127) and for
  the unembedding (80 -> 110): GEMV_BLK=19.
- 768 threads per block (80 registers, spills halved) instead of 1024 (64): B=1 5.95 -> 5.40 ms, B=8 unchanged.
- Deeper FP4 unroll (2/4/8 units in flight per warp): no difference -> per-warp memory parallelism is not the limit.
- Instruction cache: routing phase 4.6 us cold / 4.0 us warm -> not the main cost.
- Written after s0003 (not yet measured): LL exchange (every cross-card value travels with its tag in one 8-byte
  store, consumers poll their own elements; no system fence, grid barrier or counter for the 3 exchanges per layer);
  L2 prefetch of K/V rows and of the small per-layer weights during the attention-input phase.
- s0003 (5.4 h, track A): 5324 / 9310 us (pred 5350 / 9450). 768 threads, contiguous GEMV, blk GEMV for qkv/o_proj/w2, V8 vector code, attention rewrite.
- s0004 (6.1 h): incorrect, http_tokens 0.155 prefix, 0/8 identical. Cause found (tools/t_evalorder.py, t_perm.py,
  t_perm2.py): results depended on the batch slot of a sequence: w2's inputs bitwise equal in both orders, outputs
  1 ulp apart -> the per-column scaling acc += a * s was contracted to FMA differently for even and odd columns. The
  HTTP threads deliver the 8 requests in another order than the evaluator's rows. Fixed with explicit __fmaf_rn;
  slot-order invariance now tested (t_perm: logits bitwise equal under a permutation).
- s0005 (6.6 h, track A): 4778 / 8845 us (pred ~4800 / ~9150). Slot-invariant; LL exchange; router blk; attention fixes.
- W12 (track B) build: verified bit for bit on all 5,984,649,216 dense values (432,825 escapes) and exact tokens, but 47.6 ms/step (10x slower): the decode/escape path inside the engine is pathological (not investigated yet; dense GEMVs are latency-bound, so W12's upside is small here).

# Run s7-fb1 (track B on the inherited engine)

## 0.0-0.3 h: baseline and census
- s0001 (inherited engine unmodified): **incorrect**, http_tokens 1/8 identical (mean prefix 0.39). The inherited s0003 still has
  the slot-order dependence s6 found after s0004 (FMA contraction per MMA column in the w2 / MXFP8 scaling); it passed in s6
  only because the HTTP threads happened to deliver the requests in row order. Speed as in s6: 5317 / 9297 us.
- s0002 (track A, + explicit __fmaf_rn in the two scaling sites; tools/t_perm.py logits bitwise equal under a row permutation,
  t_evalorder online 8/8): **valid, 5317 us (B=1) / 9340 us (B=8)**. This is the baseline every later number is compared with.
  In process: step 5308 / 9514, w1-like 5415 / 9600.
- Census of all weights (findings G1, exec e0002): nothing to skip and nothing shared in this checkpoint; the only lossless
  storage gain with a fixed-width code is the dense BF16 -> 12 bits (F1). Decision: track B = (1) W12 for the dense matrices,
  (2) load-time tables for what depends on the token id only (layer 0 up to its attention), each measured against the same
  kernel without it; kernel improvements go first as track A (the s6 s0005 kernel: LL exchange etc.).
- s0003 (0.4 h, track A): the s6 s0005 kernel on this node: **4819 / 8895 us** (pred ~4800 / ~8950): -9.4 % / -4.8 % vs s0002.
  In process: step 4809 / 9060, w1-like 4936 / 9202. Per-phase means at B=1, context 576 (exec e0004, us per layer):
  attn_in 15.1, qkv 9.1, attention 10.6, o_proj 6.2, exchange 2.1, post-attention 16.1 (loads 5.2, mix 6.2, norm 1.1, quant 3.6),
  router 2.2, top-8 4.7, w13 16.8, quant2 8.9, w2 5.5; final 97 (unembedding 77). 45 of 97 us per layer are per-token vector
  phases on one block (not weights), 40 us the five GEMVs (their bytes need ~14 us at 6 TB/s).
- A random-token step at B=8 (7.6 ms) is much faster than a greedy step (9.0 ms): the union of experts differs. All later
  timings use the score's workload (tools/t_prof.py: 64 prompt tokens, 1024 greedy tokens).
- 0.5-1.0 h, findings with tools/t_prof.py (score's workload, phase stamps):
  * W12 measured (findings G2): only the unembedding gains.
  * `cp.async.bulk.prefetch.L2` issued serially by one thread costs ~0.4 us each: the w2 prefetch loop over the union was 20 us
    per layer at B=8 (51 experts); at B=1 its 4 us hide in the wait for the other cards' w13 (and removing it there made qkv
    and w2 slower: 99.3 -> 103.7 us per layer).
  * The step function's spills are global: adding 6 registers of loads to quant2 slowed qkv by 1.8 us (stack frame 216 -> 256 B,
    768 threads x frame >> L1). Phases as own functions: B=1 -2.4 %, B=8 +4 % (w13); GEMV calls as own functions: +4-5 %.
    tools/ptxas_stats.sh prints the frame / spill bytes per function: check after every change.
- s0004 (1.1 h, track A): parallel mailbox polls, parallel w2 prefetch at B>=2, B=1 token phases as functions:
  **4794 / 8486 us** (pred 4795 / 8470).
- B>=2 w2 with compact one-column fragments per routed (token, expert) pair, staged in shared memory (the dense 8-column
  fragments were 1 KB of activations per 576 B of weights, read through L2): logits bitwise equal to the dense build
  (tools/cmp2.sh), w2 30.9 -> 23.8 us (NVFP4 layers), quant2 one block per pair (64 items instead of nu x B): in process
  8510 -> 8069 us at B=8.
- s0005 (1.5 h, track A): compact per-pair w2 fragments at B>=2: **4797 / 8031 us** (pred 4794 / 8040).
- GEMV schedule "ctg" (block b owns whole strips, its warps split the block's units, cut strips summed through shared-memory
  slots in warp order: no global slot, fence or atomic) for w13 and the unembedding; NVFP4 units in flight per warp 2 at B=1, 4
  at B>=2 (FP8: 2 in flight is slower, 8163 vs 7983): in process B=1 4796 -> 4684, B=8 7975 -> 7653 (w13 16.8 -> 14.4 us at B=1,
  51.0 -> 42.7 at B=8). Results differ from the previous build in the last bit (the pieces of a strip are other ranges);
  determinism and slot invariance hold (t_det, t_perm).
- s0007 (2.6 h, track A, loader only): **4688 / 7606**. s0008 (2.7 h, **track B**: unembedding in the exact 12-bit code, same binary,
  storage chosen at load): **4679 / 7596**: -9.2 us (-0.20 %) at B=1, -10 us (-0.13 %, inside the noise) at B=8, as predicted
  from the op's own stamp (77 -> 67 us). First clean track-B number of this run.
  Two traps on the way: (1) with the W12 loop inlined, the rest of the B=1 kernel got 45 us slower (register allocation);
  (2) with one binary, the layer loop was still 25 us slower with the 12-bit unembedding: the runtime buffers were allocated
  after the unembedding and landed elsewhere once the packer had run. Buffers are now allocated before the unembedding
  (B=1: "first" 4715, "late" 4691, arenas 4673-4746: placement alone moves B=1 by 1.5 %; B=8 by 0.2 %).
- 2.8-3.4 h: where the token phases' time goes (tools/t_xcard.py, t_arrive.py, t_rc.py, stamps inside the poll+conv):
  * the data of the other cards is there within ~1 us of the slowest producer block's end (t_arrive): the "wait" was not a wait.
  * cross-card stores: halving their number (16-byte stores of 2 elements) changes nothing; one store instruction per
    destination card per strip instead of two: -0.9 % at both batch sizes (the instruction costs the warp, not the lane).
  * a bulk L2 prefetch instruction holds its warp ~0.4 us: QKV (7 lanes) + 4 serial K/V prefetches were 5 us in front of the
    B=1 attention-input phase; issued by warps 21-23 (idle in token phases): B=1 -1.4 %.
  * re-reading all stale mailbox words per round instead of one after the other: B=8 -3.6 %, B=1 +2 % (152 blocks poll there).
  * a cache of the finished blocks' norms and query dots (bit-identical) and a single polling block that publishes the conv
    output: both slower (4.6 KB more shared memory slowed every phase by 6 %; the single reducer's own poll+conv is the slow part).
  * inside poll+conv (B=1, layer 20, warp 0): 8 polls + 4 sums with the data present 0.8-1.0 us, taps + conv 1.0-1.3 us, in
    total 6-7 us for ~230 instructions per thread on 18 warps: the SM runs these phases at ~0.6 warp-instructions per cycle,
    and a single warp's dependent chain (top-8: 8 rounds of shuffle reductions) at ~16 cycles per instruction.
    The per-token phases (46 of 92 us per layer at B=1, 54 of 152 at B=8) are bound by instruction count on one SM, not by memory.
- s0009 (3.6 h, track A): idle-warp prefetch, one cross-card store per destination, batched re-polls at B>=2: **4568 / 7263**.
- s0010 (4.3 h, track A): hardware E2M1 conversion in the NVFP4 input quantizers, w2 inputs one 16-block per thread:
  **4246 / 7117** (pred 4245 / 7120), logits bitwise equal to s0009's build. B=1 is at 1.109x SGLang (line: 4283).
  quant2 7.7 -> 2.9 us per layer at B=1; the compare chain was evaluated 3 times per value.
- Not kept: block-contiguous schedule for w2 (B=8 -1.5 % but B=1 +4 %; strips per block are 1 or 2 either way), for qkv / o_proj /
  router (slower); 1024 threads per block for the B>=2 kernel (illegal address: something assumes 768; not pursued).
- s0011 (5.0 h, track A): top-8 with scores in registers / tree / ballot, routing per token block and conv helpers on idle
  blocks at B>=2: **4239 / 7035** (pred 4240 / 7040). Conv helpers at B=1 (idle warps 18-20 of 6 blocks) were slower (+1.6 %):
  the helper warps share their SM with the block's own token phase. Unrolled fragment staging: B=8 +10 % (24 more live
  registers in the B>=2 step function: frame 288 -> 536 B). 1024 / 896 / 832 threads per block for the B>=2 kernel: slower
  (only the MXFP8 w13 gains: 69 -> 57 us, it is bound by loads in flight per warp), FP8 two units in flight: slower (spills).
- Layer-0 token tables (track B, findings G4): first version with a branch in the layer loop made the same binary 0.85 %
  slower with the switch off (4241 -> 4277); now the row copy sits in layer 0's attention-input phase and the loader sets
  layer 0's qkv_strips to 0: switch off 4236 / 7095, tables 4222 / 7088, tables + 12-bit unembedding 4212 / 7076.
- s0012 (5.1 h, A, reference binary) 4235 / 7044; s0013 (B, layer-0 tables) 4221.5 / 7037; s0014 (B, both) 4195 / 7004, but the
  pod was replaced before s0014 (epoch 2; the new node runs the same binary ~0.45 % faster: A in process 4217 / 7060 against
  4236 / 7095). Same-pod pairs since then: s0015 (A) 4214.8 / 6899 and s0016 (B both) **4188.4 / 6884**: track B = -26 us
  (-0.63 %) at B=1 and -15 us (-0.22 %) at B=8, as in process (-24 / -19).
- s0015 (6.0 h, track A): MXFP8 loads pipelined in the B>=2 kernel (next unit's loads before the current unit's MMAs): B=8
  7060 -> 6938 in process, 6899 over HTTP. Two units ahead: +0.6 %. The same for NVFP4 (as a second loop): frame 240 -> 352-504 B,
  +20-33 %: not kept. In the B=1 kernel the FP8 pipeline costs 0.4 %.
- Conv helpers called only by the helper threads + taps loaded up front: lost the helpers' gain (7083 -> 7150): reverted.
- s0017 (7.2 h, track A): attention scores by m16n8k16 BF16 MMAs and the value sum with 8 dimensions per thread, B=1 kernel
  only: **4174 / 6901**. Checked against the scalar path on layer-0 attention outputs (a function of the tokens alone):
  rel L2 1e-5 median, 5e-4 max, 42 % of the steps bitwise equal (tools/t_attn0.py); the logits of the two B=1 builds differ
  by 0.07 median after 48 layers, like any two valid paths. In the B>=2 kernel the same code costs 3 % inlined and 1 % as
  an own function.
- s0018 (7.5 h, **track B** on the s0017 kernel): **4151 / 6882**: -23 us (-0.56 %) / -19 us (-0.28 %) against s0017.
- s0019 (8.0 h, track A): NVFP4 group size of the B>=2 streams 6 (w13) / 5 (w2): **4176 / 6803** (pred 4174 / 6805).
  The load-ahead form (as for MXFP8) makes the NVFP4 loop spill (frame 240 -> 360 B) and the whole step 15 % slower, also
  when it replaces the grouped loop instead of preceding it.
- s0020 (8.6 h, track A): **4074 / 6540: parity at both batch sizes** (1.156x / 1.010x SGLang). A sweep of the scheduling
  choices after the bottlenecks had moved (execs e0073-e0077, in process, B=8 unless said):
  * K/V prefetch at B>=2: serial on warp 22 of every non-token block (32 bulk prefetches, ~13 us) in front of the grid
    barrier: 6639; one thread per range: 6473; none: 6511. At B=1: serial 4106, at once 4084, none 4072.
  * w2 prefetch at B>=2: at once 6840, none 6757. At B=1: serial 4142, at once 4099, none 4154.
  * conv helper blocks per token: 2: 6907, 4: 6840, 8: 6808.  NVFP4 group at B=1: 2: 4176, 3: 4160, 4: 4186.
  * reader-side fences (quant2, attention combine) off: B=1 4160 -> 4141, B=8 6675 -> 6637.
  * tensor-core scores at B>=2 inline: 6842 -> 6785; the 8-dimension value sum there: 7260; as an own function: 7019.
  * poll back-off 8 / 16 / 32 ns: no difference at B=8.
  The prefetch choices of run s6 and of my first hours were measured when other things were slower; each had to be re-measured.
- s0021 (9.1 h, **track B** on the s0020 source): **4051 / 6513** (pred 4050 / 6522): -22 us (-0.55 %) at B=1, -27 us (-0.41 %)
  at B=8 against s0020; the gate statistics of the two are equal in every digit (bit-identical logits on the hidden data).
- After s0020: per-prefetch switches (B>=2 without the qkv weight prefetch 6473 -> 6455, the others help; at B=1 all four
  help), w2 on the block-contiguous schedule at B>=2 (6851), 16 helpers per token, MINU 2 / 8: no gain; the variant source
  is in scratch/ink_after_s0020_sweeps.cu (B=1 +0.2 %, B=8 -0.3 % against s0020: not submitted).
- Final state: engine/ = s0020 source, engine/TRACK = "B W12U L0T" (s0021); "A" gives s0020.
- s0022 (9.3 h, track A): block-per-strip GEMVs run a block's two strips at once (12 warps each): **4014 / 6425**
  (pred 4015 / ~6475). On the greedy workload the B=8 figure had first looked like -7 % (6473 -> 6022): the new rounding took
  another greedy text with 44 instead of 50 distinct experts per layer; on a fixed token stream it is -1.0 % (findings G7,
  tools/t_replay.py). A define placed behind its first use had left the change off in the default build: caught because
  the rehearsal's timings had not moved.
- s0023 (9.5 h, **track B** on the s0022 source): **3991 / 6406** (pred 3991 / 6405): -22.7 us (-0.57 %) / -19 us (-0.30 %).
  Final state: engine/ = s0022 source, engine/TRACK = "B W12U L0T" (s0023); "A" gives s0022.
- After s0023 (in process, bit-identical variants): NVFP4 group sizes, helper count, FP4 group at B=1, prefetch modes
  re-swept under the paired-strip build: the defaults stay (6025 / 4013). The own card's copy of the GEMV results with a
  gpu-scope store instead of a system-scope one: no gain (scratch/ink_own_gpu_variant.cu).
- 10.1 h: a third track-B candidate, pre-scaled attention-residual queries (bit-identical): slower in both kernels (findings
  G8); not kept. Final track-B table in process on the final kernel (e0090): A 4013 / 6023; 12-bit unembedding -11 / -9 us;
  layer-0 tables -11 / -11; both -23 / -21.
- 10.0-10.3 h, on the final kernel, all bit-identical and none kept: arenas / buffers-first layouts (late0 stays best: B=1
  4013 against 4032-4046), conv-helper call restricted to the helper threads (6022 -> 6043), w2 prefetch at B=1 issued from
  an idle warp (w13 -0.8 us, quant2 +0.9: the time only moves between the two phases).
- 10.4 h: attention chunks of 64 positions in the B=1 kernel (twice as many items on otherwise idle blocks): B=1 4015 -> 3995 us
  (-0.5 %; attention 9.2 -> 8.7 us, o_proj +0.7 because the combine reads twice as many chunk partials), layer-0 attention
  outputs equal to rounding. Not submitted: the combine handles 32 chunks, i.e. 2048 positions; choosing the chunk size at
  run time costs most of the gain (4009), and the fallback I tried, the B>=2 kernel for a single sequence beyond 2048
  positions, hangs: since s0005 the B>=2 kernel builds its (token, expert) pair map only for B > 1, so it cannot run B = 1
  (reachable only with MK_NO_RED=1 or such a fallback; the evaluator's B=1 always takes the B=1 kernel). Left as it is and
  written down here; variant in scratch/ink_chunk64_variant.cu. The hanging test held the pod until its timeout.
- s0024 (11.1 h, track A, pod epoch 4 after a third pod loss): attention items of 64 positions in the B=1 kernel, done
  properly: the chunk size is a template parameter of the step function, `ink_step_red` uses 64, a twin `ink_step_red128`
  takes over for a single sequence beyond 2048 positions (tested across the switch to 2170 positions): **3996.5 / 6459**
  (pred 3995 / 6425; the B>=2 kernel is bit-identical to s0022's: the B=8 difference is the node).
- s0025 (11.2 h, **track B** on the s0024 source): **3974.7 / 6436** (pred 3974 / 6440): -21.8 us (-0.55 %) / -23 us (-0.36 %).
  Fifth same-pod pair; the track-B difference has been 22-26 us at B=1 every time.
- Final state: engine/ = s0024 source, engine/TRACK = "B W12U L0T" (= s0025); "A" gives s0024.
  Open: the B>=2 kernel cannot run B=1 (summary, "Known limitation"); B=8 is 6.7 % short of the 1.10x line.
- s0026 (11.5 h, track B, final): s0025 with MK_NO_RED ignored in api.py (same kernel binary): **3975.02 / 6436**, against
  3974.72 / 6436.0 for s0025 on the same pod: the evaluator repeats to 0.3 us at B=1 with one binary on one pod.
  engine/ = s0026 source, engine/TRACK = "B W12U L0T"; "A" gives the track-A engine of s0024.
- 11.4 h, on the final kernel (pod epoch 4, bit-identical variants, B=1): K/V prefetch at once 4008 / serial 4010 against
  none 3998; NVFP4 group 2 instead of 3: 3997. The defaults stay; nothing further submitted.
- s0027 (11.6 h, track A): the B>=2 kernel builds its pair map for every batch size (phase_route gets the kernel kind
  instead of testing B > 1); MK_NO_RED works again (B=1 through the B>=2 kernel: 4390 us per token, tokens consistent).
  Logits bitwise equal to s0026's build on the normal paths: **3996.7 / 6455** (s0024: 3996.5 / 6459).
- s0028 (11.7 h, **track B**, final): **3974.0 / 6433**: -22.7 us (-0.57 %) / -22 us against s0027. Sixth same-pod pair.
- Final state: engine/ = s0028 source, engine/TRACK = "B W12U L0T"; "A" gives s0027. No known defect left open;
  B=8 remains 6.7 % short of the 1.10x line (6004).
- 11.8 h, reference for later comparisons (final kernel, pod epoch 4, tools/t_replay.py: its own greedy stream of 64 + 1024
  tokens saved and teacher-forced through step(), 41.4 distinct experts per layer at B=8): track A 4010 / 6077 us per step
  at B=1 / B=8, track B 3990 / 6056 (-20 / -21 us). The stream is not kept (pod files are lost); the tool regenerates it.

## Extension (operator, 2026-11-15 22:37): 3 more agent-hours, 11.72 -> 15.00

Task, in this order: (1) the sibling run fa1's final source (its s0020) unchanged as this run's track-A baseline;
(2) my two track-B changes on that kernel, verified bit for bit against the same source at track A; (3) a loud failure
beyond the context limit (the audit, `audits/ink-final/REPORT.md`, open doubt 1). Then: what else the checkpoint offers.

- 11.75 h, **s0029 (track A)**: `engine/` = the unpacked `s0020.tgz` of run fa1, `diff -r` empty, TRACK = A.
  **3308 / 5394 us** over HTTP (in its own run 3307 / 5378), in process 3252. My s0028 engine is kept in
  `scratch/engine_s0028/`, the sibling's source in `scratch/fa20/engine/`.
- 11.8-12.0 h, the port. The sibling's kernel differs from mine where the tables touch it:
  - Its token phases are "team" phases (6 blocks of 768 threads per token, one element per thread, sums through
    `tred`); layer 0's attention input is `team_attn_in`. The 1/rms of the normed embedding (`sm.piv`) is needed later
    by the block mix, so the second reduction of layer 0 stays; the first one (embedding norm) is replaced by the table.
  - Its qkv GEMV is split: the 96 output-gate strips are computed beside the attention by the blocks that hold no
    attention item ("gate split", always taken in a local layer with 152 blocks and B <= 8), 4 strips per block when
    few blocks are free. **The gate rows' FP32 summation order therefore depends on batch and position** (B=8: from
    position ~96 on a block holds 2-4 gate strips and splits its 24 warps over them). A gate table could not be
    bit-identical to track A, so the gate is not tabulated here (it was in s0028). The other rows (q, r, sink, k, v:
    141 strips, one per block, 72 units over 24 warps) are summed in one order whatever B and position: those are
    tabulated ([vocab][2304] BF16 per card, 0.93 GB).
  - Mode word of a launch: bits 16+ of the header word that already carries first | last layer (no new kernel
    parameter bytes: the sibling found 8 more bytes there as STACK:104 instead of 96). `LayerW.pad` became `qkv_skip`
    (layer 0 with tables: 192 strips, 96 skipped, so that only the gate strips are prefetched and the main GEMV has
    none); `Ctx` gained `l0tab`, `l0emb`. `ink_l0t_ok(nblk)` states the condition for bit-identity next to the macros
    it depends on; the loader refuses the tables otherwise.
  - Unembedding: both storages compiled, chosen at load (`c.un12.data`), loaded after the runtime buffers (as in s0007).
  - ptxas: stack frames unchanged (std 96, tc 176 bytes), spill stores +4 bytes in both kernels.
- 12.0 h, verification (exec e0105, `tools/x_verify.sh`, `tools/x_l0t.py`), all on real weights:
  - logits (sha1 of the FP32 logits of every step; 64 random prompt tokens then greedy through step()): the unchanged
    s0020 source, this source at TRACK=A, this source at `B W12U L0T` with `MK_W12_VERIFY=1`: **512/512 at B=1,
    1152/1152 at B=8 (to position 1151), 256/256 at B=2, 256/256 at B=4, 160/160 at B=5 equal** in all three;
    generate() (B=1, B=8) and the online path (8 requests, lengths 51-72) return the same tokens.
  - 12-bit code: round trip asserted on all 926,318,592 values at load (55,788 escapes).
  - tables: layer 0 alone in mode 1 for every one of the 201,024 ids, all four cards, against the table as installed
    (B=8, position 0): B=8 at positions 1023 and 2047, B=1 at 0 and 700 (plain kernel), B=3 at 300 (plain), B=5 at 100
    and B=4 at 1500 (tensor-core kernel, 5 / 4 live columns): **0 of 1,852,637,184 projection values and 0 of
    3,705,274,368 normed-embedding values differ** in each of the 7 comparisons; no non-finite value.
- 12.1 h, context guard (`tools/x_guard.py`): `Engine.ctx_limit = min(max_context, ink_max_positions() = 2176)`.
  `step()`, `generate()` (before the first launch), `submit()` (prompt + n - 1 > limit, empty prompt, n < 1) raise
  ValueError; `_launch` itself refuses any position outside [0, limit) (RuntimeError), so no path reaches the kernel
  beyond it. Tested with max_context 96 at B=8: 96 steps fine, the 97th refused, generate(97) refused with nothing
  launched, a request of 64 + 34 refused, 64 + 33 (last position 95) served and equal to generate().
- 12.1 h, rehearsal of the evaluator's loops with the track-B engine (e0106): B=1 step 3276 / generate 3255 / w1-like
  3421 / online 3300 us; B=8 4642 (context 64-320) / 5305 / 5598 / 5469; online = greedy 100 %.
- 12.25 h, same-pod A/B in process (e0107, generate 1024 tokens after a 64-token prompt, min of 3, two rounds):

  | build | B=1 | B=8 |
  |---|---|---|
  | s0020 source (= s0029) | 3304.3, 3304.4 | 5452.9, 5452.1 |
  | this source, TRACK=A | 3312.6, 3311.6 | 5448.5, 5448.3 |
  | B W12U | 3305.4, 3306.3 | 5443.1, 5441.1 |
  | B L0T | 3305.5, 3305.2 | 5443.3, 5440.8 |
  | B W12U L0T | 3302.6, 3299.7 | 5434.2, 5434.1 |

  The two changes give -11 us at B=1 and -14 at B=8 against the same binary at TRACK=A; that binary is 8 us slower
  than s0020's at B=1 (and 4 faster at B=8). Net against s0020: -3 us (0.1 %) / -18 us (0.33 %).
- 12.3 h, **s0030 (track B)**, hypothesis written at 12.3 h: this source, TRACK = `B W12U L0T`, with the context guard.
  **3301 / 5376 us** over HTTP (pred 3305 / 5376) against s0029's 3308 / 5394: **-7.7 us (0.23 %) at B=1, -18 us
  (0.33 %) at B=8**. In process 3254 (s0029: 3252).
- 12.4 h, where the 8 us of the new binary at TRACK=A come from (e0108, B=1, one change taken back at a time, two
  rounds): s0020 3303.8 / 3305.4; this source 3311.9 / 3312.5; unembedding loaded before the buffers as in s0020
  3311.9 / 3312.8 (the order is irrelevant here); **only the BF16 decoder compiled into the unembedding GEMV 3304.2 /
  3304.0**; table code compiled out 3312.1 / 3311.8; everything taken back 3303.9 / 3303.3. So the port's layer-0
  code is free and the whole cost is the load-time choice between two decoders inside one GEMV. In my own kernel the
  same construction cost nothing (s0007). Consequence: the storage becomes a build choice (`-DW12U` from the track
  word, `ink_un_w12()` checked by the loader).
- 12.4 h, guard at the kernel's own limit (e0108, max_context 4096, B=1): 2176 steps to position 2175 run (logits
  finite; the first time positions 1152-2175 of this kernel are run at all), step 2177 and generate/submit beyond
  refused.
- 12.6 h, the build-time variant (e0109, `tools/x_verify.sh`; four libraries from one source: A, B W12U, B L0T, B W12U L0T).
  Bitwise: the unchanged s0020 source, this source as A and as `B W12U L0T` (`MK_W12_VERIFY=1`) agree on all 2,336
  steps again (512 B=1, 1152 B=8, 256 B=2, 256 B=4, 160 B=5), generate() and online tokens equal. Tables for all
  201,024 ids: reference (B=8, position 0) against B=1 at 0, B=3 at 300, B=8 at 1023 with the A binary and against
  B=1 at 700, B=5 at 100, B=4 at 1500, B=8 at 2047 with the `B W12U` binary: 0 differing values in each; **sha1 of
  the tables equal between the two binaries** (7c02497a...). Timing in process (generate 1024, min of 3, two rounds):

  | build | B=1 | B=8 |
  |---|---|---|
  | s0020 source | 3304.6, 3304.0 | 5453.8, 5454.6 |
  | this source, A | 3304.0, 3303.8 | 5460.0, 5456.2 |
  | B W12U | 3298.2, 3298.1 | 5442.0, 5445.6 |
  | B L0T | 3299.8, 3298.9 | 5448.9, 5451.0 |
  | B W12U L0T | 3288.7, 3289.7 | 5436.3, 5437.4 |

  Track A build = s0020's speed at B=1 (B=8 +2..6 us, at the edge of the rounds' spread). Track B: **-15 us (0.45 %)
  at B=1, -17 us (0.32 %) at B=8** against the s0020 source; the two items are 6 + 5 us alone at B=1 and 15 together.
- 12.7 h, rehearsal of the default build path (e0110, TRACK file `B W12U L0T` -> `-DW12U`): B=1 step 3265 / generate
  3243 / w1-like 3407 / online 3290; B=8 4644 / 5302 / 5590 / 5467; online = greedy 100 %.
- 12.72 h, **s0031 (track B, final)**, prediction written before: HTTP about 3293 / 5376. Result: **3291 / 5376 us**
  (in process 3243): against s0029 **-17.4 us (0.53 %) at B=1 and -18 us (0.33 %) at B=8**; against s0030 -9.8 us at
  B=1 (the 8 us of the double decoder), equal at B=8.
- 12.75 h, s0032 (track A) queued: the same source with TRACK = A (guard + dormant table code on the s0020 kernel).
- 12.95 h, **s0032 (track A)**: the source of s0031 with TRACK = A. **3307.9 / 5395 us** (pred 3308 / 5396); s0029
  (unchanged s0020 source): 3308.4 / 5394. The context guard and the dormant table code cost nothing; the evaluator
  repeats to 0.5 us at B=1 and 1 us at B=8 on this pod. Track B over HTTP from one source: s0031 against s0032
  **-17.0 us (0.51 %) / -19 us (0.35 %)**.
- 13.0 h, what else: the 12-bit code for one more dense matrix at a time on this kernel (e0111, `tools/x_w12x.sh`,
  source in `scratch/w12x/`; each variant's logits bitwise equal to the s0020 source on 336 steps, round trip of every
  packed matrix asserted; in process, two rounds, us per step at B=1 / B=8):

  | variant | B=1 | B=8 |
  |---|---|---|
  | B W12U (12-bit unembedding only) | 3297.2, 3298.5 | 5441.5, 5443.6 |
  | + router (W12R) | 3373.1, 3373.0 (+75) | 5532.3, 5534.9 (+91) |
  | + o_proj (W12O) | 3322.9, 3322.6 (+25) | 5458.0, 5459.1 (+16) |
  | + qkv (W12Q) | 3288.1, 3288.3 (**-9.6**) | 5463.9, 5465.2 (+22) |

  Router and o_proj lose at both batch sizes, as on my kernel (G2). qkv looked like a gain at B=1 and a loss at B=8;
  I took the two kernels' difference for the cause. **Wrong, see the next three entries.**
- 13.05 h, qkv in both forms, the plain kernel (batch 1-3) reading the 12-bit one (e0112): bitwise equal on 2,496
  steps (now with B=3), tables equal for all ids through the 12-bit decoder too. B=1 3279.9 (-9.3 against s0031's
  configuration), **B=3 3924 (+30)**, B=8 5431 (-5.5, although the tensor-core kernel's code did not change).
- 13.25 h, a third kernel instantiation so that only a single sequence reads the 12-bit qkv (e0113): bitwise equal
  again, but **B=1 3300.6 (+11 against s0031's configuration)**, B=3 3885.8 (-7.5), B=8 5444.2 (+7). The same kernel
  code for one sequence as in e0112, the same allocations, 20 us apart: the comparison across builds is not measuring
  the qkv storage.
- 13.45 h, the same library with a switch (MK_NO_ONE: one sequence through the BF16 kernel) and phase stamps (e0114,
  `tools/x_same.sh`, `tools/x_stamp.py`): 12-bit qkv 3299.6 / 3300.0 against BF16 3291.8 / 3292.3: **+8 us; its GEMV
  7.16 instead of 6.28 us per layer.** Not kept; the variant is in `scratch/w12q_one/`. `engine/` restored to the
  s0031 source (checked against its src.tgz with diff -r). Findings G11 (the build-to-build effect of about +-10 us
  that produced the false gain) and G12 (stamps: unembedding 69.7 -> 64.8 us, layer 0 attention input + main qkv GEMV
  13.1 -> 6.4 us at B=1; the same at B=8).
- 13.6 h, stamps of s0030's source against s0031's (e0115, `tools/x_dual.sh`, B=1, two rounds): unembedding GEMV 70.36 /
  70.37 us (two decoders compiled, BF16 chosen) against 69.59 / 69.55 (one); with the 12-bit code 64.77 / 64.66
  against 64.81 / 64.78. The 48 layers: 3160.8 / 3166.6 against 3153.1 / 3153.6 at track A. So what I wrote in the
  hypothesis of s0031 ("this alone made the BF16 path 8 us slower") is right as an observation of the build and
  wrong as a mechanism: the unembedding itself differs by 0.8 us, the rest is the layers' code in that build.
  Findings G9 corrected, G12 extended: by stamps and by the one-binary A/B the two items are worth 11-14 us per step.
- Final state: `engine/` = s0031's source (TRACK `B W12U L0T`; `A` gives s0032). Not kept: `scratch/w12q_one/`
  (12-bit qkv for one sequence), `scratch/w12x/` (per-matrix 12-bit variants), `scratch/var/` (the take-back builds),
  `scratch/s0030/` (the two-decoder source), `scratch/fa20/` (the sibling's unchanged source).
- 13.75 h, **s0033 (track B, final)**: s0031 with one host-side addition, `submit()` refuses token ids outside
  [0, 201024) (they would index past the embedding and the tables; the audit lists it under input validation).
  `ink.cu` byte-identical to s0031's. Guard test extended (ids 201024 and -1 refused, 201023 served). **3290.0 /
  5378 us** (pred = s0031: 3290.9 / 5376): the track-B build repeats to 1 us at B=1 and 2 us at B=8.
  `engine/` = s0033's source (checked with diff -r against its src.tgz).
- 13.85 h, why builds differ (e0117, `tools/x_sass.sh`): disassembly of the kernels. Between my source built as A and
  as `B W12U` (only the unembedding decoder differs) 20,404 of 21,064 instruction lines of the plain kernel differ
  with immediates masked, 5,430 with registers masked too, 2,608 by opcode: the compiler redoes the register
  assignment of the whole step function. Findings G11.
- 13.95 h, the remedy tried (e0118, `scratch/ni/`, `tools/x_ni.sh`): the unembedding GEMV out of line, one function
  per decoder, chosen at load: **one binary for both tracks again**. In process, two rounds, us per step:

  | TRACK (one library) | B=1 | B=8 |
  |---|---|---|
  | s0020 source (other library) | 3304.8, 3303.1 | 5455.1, 5454.5 |
  | A | 3304.0, 3303.7 | 5516.1, 5514.3 |
  | B W12U | 3299.2, 3299.5 (-4.6) | 5509.9, 5508.7 (-5.9) |
  | B L0T | 3297.9, 3298.4 (-5.7) | 5506.7, 5507.8 (-8.0) |
  | B W12U L0T | 3293.3, 3292.1 (**-11.2**) | 5501.2, 5499.3 (**-15.0**) |

  The cleanest A/B of the extension: the two items add up and agree with the stamps (this library: unembedding
  71.1 -> 65.4 us at B=1, 72.0 -> 65.7 at B=8; layer 0 13.1 -> 6.5 / 12.9 -> 6.9). But this library's tensor-core
  kernel is 60 us (1.1 %) slower at B=8 than every other build of the same layer code (5515 against 5455): the
  register lottery of G11 again, this time against the build. Not submitted; s0033 stays.
- Final state (14.0 h): `engine/` = s0033's source, TRACK `B W12U L0T` (`A` builds s0032's kernel). Scratch: `ni/`
  (one-binary variant), `w12q_one/`, `w12x/`, `var/`, `s0030/`, `fa20/` (the sibling's unchanged source),
  `engine_s0028/` (my own kernel's final source).
- 14.1 h, one binary for both tracks without the slow kernels (e0119, `scratch/mix/`, `tools/x_mix.sh`): entry
  functions are compiled separately, so each kernel gets the form of the unembedding GEMV that left its layers fast:
  out of line per decoder in the plain kernel, both decoders inline in the tensor-core kernel. `build.sh` no longer
  reads the track word W12U. One library, two rounds, in process: A 3303.9 / 3304.8 and 5448.5 / 5452.3; `B W12U
  L0T` 3292.5 / 3292.2 and 5439.1 / 5438.5: **-11.9 us at B=1, -11.6 us at B=8**. Bitwise: the unchanged s0020
  source, this library as A and as B equal on 2,496 steps (B = 1, 2, 3, 4, 5, 8); tables for all ids equal in 4
  further configurations, sha1 7c02497a... as in every build before; rehearsal: online = greedy 100 % at B=1 and B=8.
- 14.15 h, **s0034 (track A)** and **s0035 (track B)** submitted from that one source and binary (TRACK only).
  Predictions written before: s0034 about 3308 / 5392; s0035 about 12 us below s0034 at both batch sizes.
- 14.2 h, **s0034 (track A, one binary for both tracks): 3307.1 / 5394.5 us** (pred 3308 / 5392; s0029 3308.4 / 5393.7,
  s0032 3307.9 / 5395.4: three track-A builds within 1.3 us at B=1 and 1.7 us at B=8).
- 14.25 h, **s0035 (track B, the same binary): 3294.7 / 5382.7 us** (pred about 12 us below s0034 at both: 3296 /
  5380). **Against s0034: -12.3 us (0.37 %) at B=1, -11.8 us (0.22 %) at B=8**; the evaluator's in-process generate
  figures 3261.7 -> 3249.8 and 5410.5 -> 5395.7. This is the track-B figure of the extension: the same kernel binary
  with and without the two changes, measured by the evaluator, and it equals the stamps (-11.5 / -11.7) and the
  in-process A/B of that library (-11.9 / -11.6).
- The fastest valid submission stays s0033 (3290.0 / 5377.5): its two-build form happens to run 4.7 us (B=1) and
  5 us (B=8) faster than the one-binary form of s0035, the build effect of G11 in its favour. **Final state:
  `engine/` = s0033's source** (diff -r against its src.tgz empty), TRACK `B W12U L0T`; `A` builds s0032's kernel.
  The one-binary source of s0034 / s0035 is `scratch/mix/engine/` (diff -r against s0035's src.tgz empty).
- 14.35 h, **s0036 (track B, final source)**: s0033 with the round trip of the 12-bit unembedding asserted at every
  load instead of only under MK_W12_VERIFY (926,318,592 values, about a second; `ink.cu` and `build.sh` byte-identical
  to s0033's). **3291.2 / 5378.3 us** (pred = s0033). The same build three times over HTTP: s0031 3290.9 / 5375.7,
  s0033 3290.0 / 5377.5, s0036 3291.2 / 5378.3. The assertion is not vacuous (e0121: a flipped bit at four places
  of a packed matrix is detected each time).
- **Final state (14.4 h): `engine/` = s0036's source** (diff -r against its src.tgz empty), TRACK `B W12U L0T`;
  TRACK `A` builds s0032's kernel. Lowest figure on the board: s0033, 3290.0, the same kernel 1.2 us apart.
  Submissions of the extension: s0029 A (sibling's source unchanged), s0030 B, s0031 B, s0032 A, s0033 B, s0034 A and
  s0035 B (one binary), s0036 B: 8 of 8 valid.
- 14.45 h, last check of the final engine (e0122, `tools/x_final.sh`), the batch sizes and positions the earlier
  runs left out: the unchanged s0020 source, `engine/` built as A and as `B W12U L0T`: logits of every step equal at
  **B=1 for 1300 steps** (across the 1024-position window of the local layers), **B=8 for 2048 steps** (to the last
  position of a 2048 context), **B=6 and B=7 for 200 steps each**. With the earlier runs every batch size 1-8 is
  covered: 6,244 steps compared bit for bit between the sibling's source and both tracks of mine.
- 14.47 h, per-phase profile of the final engine (e0123, `tools/x_phase.py`; findings G13): B=1 66.8 us per layer
  (attention input 10.7, qkv 6.3, attention 10.8, o_proj 3.0, post-attention 10.6, router 3.0, routing 3.8, w13 8.0,
  quant2 5.4, w2 5.0), unembedding 64.9; B=8 112 us per layer, MXFP8 layers 134. Layer 0's post-attention showed 41
  us in this mode; checked in e0124 (`tools/x_skew.py`): it is the launch skew of synchronised single steps (the four
  kernels start 30 us apart), in `generate()` they start within 2.4 us and the phase takes 10.6 us. Not a loss of
  the scored paths.
- 14.6 h, the last checkpoint-specific item, measured in scratch only (e0125, `scratch/piv/`, `tools/x_piv.sh`): layer
  0's second reduction (1/rms of the normed embedding) from a per-token table as well, switched at load in one
  library (MK_L0PIV). Logits bit-identical to the s0020 source on 464 steps. Stamps: layer 0's attention input 5.47
  -> 4.36 us at B=1, 5.14 -> 4.04 at B=8 (-1.1 us); end to end 3295.2 / 3294.6 -> 3294.3 / 3294.6 at B=1 and 5550.2 /
  5550.4 -> 5550.3 / 5551.7 at B=8: nothing measurable. Not kept. (That library runs B=8 at 5550 us, 110 us slower
  than the others: ten more lines in layer 0's branch of the attention-input phase, and the tensor-core kernel's
  layers came out 2 % slower. The largest instance of G11 seen.)
- End of the extension (14.65 h). `engine/` = s0036's source; git clean; notes: this journal, findings G9-G13,
  summary ("Extension" section first), plan.md ("State at the end of the extension").
- 14.7 h, operator's time check (23 minutes left): no new experiments. Summary given its closing block (best submission, what worked, what did not, next). `engine/` = s0036's source, submitted and valid.
