# Decode megakernel playbook: what 12 runs measured

Facts distilled from the ledgers, notes and audits of 12 autoresearch runs. A number from one model or card is a
hypothesis for another, not a law. Unless stated, numbers are the evaluator's headline (us per token over HTTP at
B=1, lower is better) or the run's own in-process measurement where the notes say so; "−x%" is against the previous
best of that run.

Run ids used below:

| id | run | model, card | track(s) | agent-h |
|---|---|---|---|---|
| s2 | s2-q8b-a1 | Qwen3-8B dense, 1x H100 | A + B | 39.4 |
| s10 | s10-q8b-h100-f1 | Qwen3-8B, accuracy fix of s2 | A + B | 2.3 |
| s3 | s3-q8b-online-a1 | Qwen3-8B serving (slo_share) | A + B | 39.4 |
| s4n | s4-north-h100-a1 | North-Mini-Code MoE (128 experts, top-8), 1x H100 | A + B | 22.6 |
| s4f | s4-flint-b200-a2 | Flint MoE (256 experts, top-6), 1x B200 | A (+ "B") | 35.9 |
| s4q | s4-qwen38-27b-b200-a1 | Qwen3.8-27B hybrid (48 gated-delta + 16 attention layers), 1x B200 | A, C | 23.6 |
| s5 | s5-qwen38-27b-b200-b1 | same, track B | B | 5.6 |
| s9 | s9-qwen38-27b-b200-f1 | same, A + B | A + B | 7.2 |
| s6 | s6-inkling2s-b200x4-a1 | Inkling-2-Small 594B MoE, NVFP4/MXFP8 experts, 4x B200 | A | 13.6 |
| s7a | s7-inkling2s-b200x4-fa1 | same | A | 11.5 |
| s7b | s7-inkling2s-b200x4-fb1 | same | B | 14.4 |
| s8 | s8-inkling2s-b200x4-on1 | Inkling serving (slo_share) | A | 30.5 |

Track A = checkpoint formats and the general engines' data flow; track B = anything lossless on top; "track C"
(s4q only) = exact draft-and-verify, ended by the operator.

---------------------------------------------------------------------------------------------------------------

## 1. Where the time goes

Speed of light (SOL) = evaluator bytes per step / nominal bandwidth (H100 3.35 TB/s, B200 8 TB/s per card). Every
run measured a lower practical streaming rate first; that rate is the more useful floor.

| model / card | bytes per B=1 step | SOL nominal | practical floor (measured rate) | final engine (HTTP B=1) | baseline |
|---|---|---|---|---|---|
| Qwen3-8B / H100 | 15.17 GB | 4527 us | 4850 us at 3.12 TB/s (s2 bw.cu: 3.0-3.1 TB/s = 89-92% of nominal) | A 5365 (s10 s0004), B 4136 (s10 s0002); in process A 5230 (s2 s0053), B 4020 (s2 s0063) | SGLang 6689, vLLM 6789, MPK 7252; SGLang+torch.compile 6031-6066 |
| North MoE / H100 | 6.63 GB (8 of 128 experts) | 1979 us | ~2140 at 3.1 TB/s; B: 12-bit stream ~1600 | A 2533 (s0022), B 2146 (s0021) | vLLM 5049, Cohere megakernel 3616 |
| Flint MoE / B200 | 5.05 GB (6 of 256 experts) | 631 us | ~690 at 7.3 TB/s | A 1479 (s0077; in process 1364-1382); B=8 3394 | vLLM 1971 / 3796 |
| Qwen3.8-27B / B200 | 51.4 GB | 6427 us | 7.0 ms at 7.3 TB/s (s4q C1: 7,270 GB/s plain, 6,930-7,185 with MMA) | A 8445 (s9 s0022), B 7665 (s9 s0021) | SGLang 9523-9594, vLLM 10110 |
| Inkling / 4x B200 | 19.33 GB (all cards) | 604 us | per card 6.2 TB/s BF16, 5.9 TB/s NVFP4 GEMV (s7a F3) | A 3307 (s7a s0020), B 3290 (s7b s0033) | SGLang 4711 / 6604 |

What the remaining gap was attributed to:

- **Qwen3-8B (s2, track B, B=1)**: per layer ~104 us against 92.8 us of bytes at 3.12 TB/s. Attention + o projection
  ~13.4 us for 8.1 us worth of o bytes (a chain of latencies on few SMs); ~4.5 us in the three matvec transitions;
  grid-barrier latency ~2 us per phase x 5 phases per layer. The three matvec phases that follow a matvec ran at
  94-97% of the HBM bound transitions included (s2 journal, timing build). Over HTTP the headline adds ~100 us of
  attention growth from position 64 to 1088 (one SM per unit reads L2 at 165 GB/s) and ~46 us of host round trip per
  token (s10). Track A (BF16, 15.1 GB) sat 11% above its 4850 us floor.
- **North (s4n)**: stream time and skeleton time add. Skeleton build (weight streaming switched off, results wrong):
  826 us of 2588 at B=1 (16.5 us per layer), 1479 of 6603 at B=8; with neither streaming nor attention: 826 vs 1170.
  Handoffs were ~18 of 46 us per layer at B=1 on track B. Skeleton trims took it to 747 us.
- **Flint (s4f)**: B=1 in process ~27 us per layer for 107 MB = 15 us of memory, plus 70 us of head. The extra per
  layer: wait of the 128 weight SMs for attention ~4 us, router and routing 2.3 us, six hand-overs ~1 us each. B=8:
  ~70 us per layer, of which 40 us in the two streamed phases (within 5 us of memory time) and 30 us in front of them
  (six hand-overs at 2-2.5 us, attention 4-8 us growing with context, routing 3.7 us, norms). The step grows ~200 us
  from position 64 to 1088 at B=1.
- **Qwen3.8-27B (s9)**: 8.47 ms in process = ~7.0 ms weight stream + ~3.3 us fixed cost per GEMM op x 257 ops
  (~0.85 ms: stream-end spread inside a block 1.3-2.1 us, slowest block 0.7-2.8, hand-over + epilogue 2.2, last
  arrival to release 1.8-2.1) + recurrent op 8.7 us x 48 + attention ~17 us x 16 + lm_head 350 us. With every weight
  stream switched off a B=1 step still took ~1.8 ms, B=8 2.6 ms (s4q). Track B: W12 loop streams 6.3-6.6 TB/s of real
  bytes vs 7.3 for BF16, and exact escape handling costs ~245-300 us per pass (s5 K4, K6).
- **Inkling (s7b, B=1, card 0)**: per layer 66.8 us: attention input 10.7, qkv 6.3, attention 10.8, o_proj 3.0,
  post-attention 10.6, router 3.0, routing 3.8, w13 8.0, quant2 5.4, w2 5.0; unembedding 65 us per step. "No per-layer
  dense GEMV is bound by its bytes" (MBU 0.186). It is a latency problem of ~10 dependent phases (s6 lesson 1).
  B=8: ~112 us per layer (NVFP4 layers 106-112, MXFP8 layers 134 with a 50 us w13).

Cross-model reading: on dense models with large weights (Qwen3-8B, Qwen3.8-27B) the engines got within 10-20% of
the measured streaming floor and the residual is per-phase fixed cost. On MoE and multi-card models (North, Flint,
Inkling) the active bytes per layer are small and the step is dominated by the per-layer skeleton (hand-overs,
routing, norms, attention), 2-5x the byte time.

---------------------------------------------------------------------------------------------------------------

## 2. Changes that paid off, by mechanism

### 2.1 Single persistent kernel and step structure
- One launch per step instead of one per operator: s2 s0003 −12.2% (7366 -> 6465; launch gaps had cost ~0.9 ms).
  Own CUDA operators replacing HF first: s2 s0002 −65% (21051 -> 7366).
- Phase interpreter with no calls: s2 s0010 −0.7% (5811-5896 -> 5771). Mechanism: shared memory took 229 of 256 KB
  so L1 was nearly gone; every spill or saved return address became an L2 round trip queued behind weight loads.
  A generic phase function spilled (7906 us); separate loader/coordinator functions 6726 -> 6048 after a register
  diet; no calls at all 5781.
- Removing per-thread stack traffic: s4q s0002 −36.2% (16990 -> 10834). A 160-byte op descriptor copied to 155,648
  thread stacks made an empty op cost 5.2 us vs 0.8 us for the barrier alone (s4q C4).
- Reloads of loop state after op calls: s4q s0042 −2.7% (9258 -> 9006); the main loop kept program pointer, op
  count and thread indices in local memory and reloaded them (28 local loads) right before each barrier. Re-read
  from shared memory and special registers instead (s4q C14). Each extra noinline helper on the hand-over path: +4.5%.
- Thread/block index through inline asm (`mov.u32 %0, %tid.x`): stops the compiler from keeping lane/warp alive and
  spilling them (s2 s0030: one slot, 59 LDL). Arrays assigned under a condition live in local memory; initialise
  unconditionally (s2).
- Kernel body wrapped in an inlined function, giving a different register allocation: s4f s0055 −2.8% (1557 -> 1473,
  B=1 kernel only). Two kernel instances (batch 1-2 and 3-8): B=1 serve 1753 -> 1728, bit-identical (s4f 13).
- Out-of-line phases at a register cap: s7a s0013 −0.1%; the opposite direction (un-inlining attention and routing)
  was +33% on s4n (calls spilled 416 bytes). Whether a call pays depended on the register budget per thread.
- Fused RMSNorm (scale by 1/rms after the matvec, bf16(w*x) published by the previous epilogue): s2 s0006 −3.1%.
  Track-A variant (norm applied by the loaders, no norm phase, 73 grid barriers fewer at B>=3): s2 s0049, B=8 −3%.
- Fewer phases per layer: attention as one op with a block per 256-key chunk and sentinel hand-overs: s9 s0008 −1.1%
  (only the fourth form tried paid). o_proj inside stage B: s4n s0018 −2.9% (after the L1 change; before it, +12-45 us).
- Transferred: the "no calls / no local memory on the per-phase path" rule paid on all three dense/hybrid engines
  (s2, s4q, s4f) and its violation was the first large cost found on Inkling (s6: 320 B/thread frame, mixes 16.7 ->
  4.0 us, top-8 13.5 -> 3.4 us after keeping scalars in shared memory).

### 2.2 Grid barriers and synchronisation
- Barrier form: atomic counter + poll stayed. Flag or stamp barriers were slower in every form (s2: three variants,
  +50 us; attention -> o via stamps +35-45 us; s4n several polling warps +45-80 us; s4q dataflow flags no gain).
- `__threadfence` cost: 47 us per step in s2 (0.9%), kept because without it data/counter order is not guaranteed.
  Removing the fence behind the K/V append (the block itself is the only reader before the grid barrier;
  `bar.sync` has membar.cta semantics): s2 s0052 −0.3% B=1, s0053 −54 us on track A.
- Grid-barrier increment directly behind the fence: s2 s0035 −0.2%. Barrier back-off 200 -> 25 ns: s5 s0017 −0.1%.
  Barrier's own L2 prefetch issued after the counter increment, one 64 KB piece per poll, none after release:
  s9 s0018/s0020 −0.5% B=1, −0.3% B=8.
- Waiting only where there is a dependency: attention starts on a counter of the qkv strips it reads (140 of 237):
  s6 s0010 −4.3% (−174 us). Only the attention blocks wait at the q/k/v barrier (s2 s0059, neutral to −0.4%). No
  block barrier at phase entry with 3+ sequences: s2 s0056 −0.1% B=1, −0.9% B=8.
- Sentinel hand-over (writer exchanges a NaN sentinel in at phase start, data in at the end, reader polls the data
  word; one trip through the memory queue instead of three): s4f 12d; s4q s0030 −1.4%, −1.6% per B=1 step. Requires
  strong loads (`ld.relaxed.gpu`) on the poll: with `ld.global.cg` 16-35% stale reads on B200 (s4q C11).
- Tagged-data exchange (value + tag in one 8-byte relaxed store, consumer polls its own element; no fences or
  counters): s6 s0005 −10.2% (with the slot-order FMA fix), s7a s0004 −2.3%.
- Shallow memory queue: "the L2/HBM queue is one FIFO for everybody: latency of anything = bytes in flight /
  bandwidth" (s4n bench: a resident load 3 us, atomic 2.3 us, store+fence 10-13 us under a full stream vs
  0.2-0.45 us quiet). 16 warps x 2 loads in flight still saturated H100 with ~0.7 us round trips (s4n v4). On s2 the
  streaming loop kept deep queues (GB 3 -> 2 or 1: +63-68 us) but limited what was in flight at phase ends.
- Polling thread choice: hand-over counters polled from warp 6 instead of leader warp 0 (a warp's load waits for
  the exchanges it issued before): s4f s0061 −17 us B=1.
- Counter page placement chosen by a start-up timing (fastest of 16 pages): s4f s0049-s0054, track A 3914 -> 3870 at
  B=8, 1527 -> 1516 at B=1 ("the address of a 4 KB page every block polls is worth 1% of the step").

### 2.3 Weight layout and loads
- Weights pre-permuted into MMA fragment order (all runs); each warp reads 512 contiguous bytes per MMA (s2).
- Shared-memory reservation under the L1 threshold: transferred to every card measured.
  s2 s0027 −2.2% (223 -> 154 KB; 12-bit stream 3124 GB/s up to 199,680 B smem, 2997 GB/s above);
  s4f s0007/s0008 −6.3% / −11.4% (224 -> 128 KB; track A B=8 6081 -> 4796; 16 tile loads 0.76 us at <=128 KB vs
  2.27 us above 192 KB); s7a s0005 −6.3% (222 -> 121 KB; B=8 5533 -> 5154); s4q C5 (349 MB burst 49.3 vs 55.6 us).
- Cache operator on weight loads: s2 s0015 `ld.lu` −1.0% (ca 5649, cg 5618, cs 5637, lu 5602 in the A/B); raw
  microbenchmarks showed no difference, the gain is less L2 pollution of shared data. s9 s0002 `ld.global.cs`
  −2.1% B=1 / −3.9% B=8 (also removed an unexplained B=8 penalty). s4n s0017 `L1::no_allocate` weight loads + zero
  operand from shared memory −4.8% ("the largest single step after the 12-bit format"; re-read x stayed in 28 KB L1).
  Bulk copies of weights need `L2::cache_hint` evict_first or they evict the step's working data (+1-3 us per later
  phase, s7a F9).
- Prefetch depth: the optimum was small and close to use. s2 s0016 2 loads + one k-step of the next tile before the
  tile barrier −1.2% (3-8 loads spilled); end-of-phase `prefetch.global.L2` depth 2 −1.1% (depth 1/2/3/4/8: 5478 /
  5449 / 5455 / 5476 / 5612); s2 s0029 adaptive stash of the next phase's first tile into shared memory, stopped by a
  "barrier complete" flag, cap 4 loads per warp: −1.7%. "prefetch.global.L2 ahead of time helped exactly twice, both
  times for a few loads a few hundred ns ahead" (s2). On H100 `prefetch.global.L2` did not stage data at all (s4n
  pf_bench); on B200 `cp.async.bulk.prefetch.L2` of the next matrix at each barrier gave −0.2 ms per step (s4q C7),
  and the B=1 near/far piece tables re-tuned for `.cs` loads gave −1.2% / −0.3% (s9 s0004, s0006).
- Warps per block: s2 s0022 24 warps (80 regs) −0.8% vs 32; 16 warps gave the same speed and fitted an 8-load
  pipeline. s6 s0003 768-thread blocks + contiguous GEMV ranges −18.9%. B200 needed 32 warps per SM for full stream
  rate (16 warps: 4,780 GB/s) (s4q C1); s4f ran 512 threads x 128 regs on main blocks.
- Equal tiles per block: q/k/v, gate+up, down on 128 of 132 SMs so 128 divides the tile counts: s2 s0057 −0.4% (track
  A worse at 1-2 sequences, so chosen per width).
- B>=3 activation fragments read from L2 by the loaders instead of a block copy: s2 s0030, B=8 4690 -> 4379 (−6.6%).
  Compact one-column activation layout at B=1 (8x smaller L1 footprint): s5 s0008 −1.7%.
- Avoiding integer work on the critical path: GEMM hand-over without 64-bit divisions, bounds published in shared
  memory, first warp by ballot, roles at each warp's stream end: s4q s0035 −3.8%, s0038 −1.1% (s4q C12). Softmax
  weights by one reciprocal + exact FMA correction instead of IEEE divisions: s4f s0041 −2.9% (38.5 vs 12.2 ns per warp).
  fast-math: s2 s0021 −0.3%.

### 2.4 Attention
- Split over position chunks for one sequence: s4n s0006 −7.5% (8 self-contained chunks merged at the end, same
  arithmetic for every batch size, flat in context); s2 split path for B<=2 from 1136 positions (6434 -> 5066 us at
  9000 positions; not used at short contexts, where its extra grid barrier costs more). s7a s0008 32/64-position
  items while blocks are free −1.1%; s7b s0024 64-position items at B=1 (in sibling kernel).
- Idle SMs pull the layer's K/V history into L2 before attention: s2 s0012 −1.1%; K of all units before any V
  (s10 s0004 −9 us). K/V cache tiles requested 8 per trip at long context: s4f s0025 −1.3%; first V batch right after
  the scores: s4f s0027 −1.0%.
- Overlap the o projection with attention: idle SMs stash their first o tile into shared memory (s2 s0008 −1.3%);
  attention blocks take two o tiles, second prefetched (s2 s0039 −0.4%); the idle blocks' stash flood delayed the
  attention blocks' K loads by 1.8 us, so stash starts after the score pass (s2 s0037 −1.2%).
- Work beside attention: output-gate projection computed by idle blocks (s7a s0009 −2.7%); batch 7-8 output
  projection of 8 pair blocks hosted by other blocks (s4f s0068, B=8 3499 -> 3460).
- Placement: units round-robin over blocks instead of consecutive warps (score op had run on 5 SMs): s4q −0.6%.
- Tensor cores for scores: s7a s0004 (with tag hand-offs) −2.3%; s7b s0017 −0.3%. GQA 8 = n of m16n8k16 makes the
  softmax output the B operand of the V mix without stores (s4f 5).
- Long-context traps: compiler moved plain K/V loads down to their uses; pinning batches with volatile asm gave
  s4f s0062 −20 us B=1, −45 us B=8.

### 2.5 MoE routing and expert loading
- Router top-k: warp max over sortable integer keys 1.7 us vs rank count with `a > b || (a==b && i<j)` 9.3 us
  (s4n); `__reduce_max_sync` top-8 (s4n s0009 −1.0% with other trims); "ranks instead of six rounds of warp maxima"
  (s4f s0021). Cycle-counter stamps inside Flint's top-6 found 3.5 us of a 70 us layer at B=8; six rewrites with the
  same order and ties: B=8 3426 -> 3394, B=1 1493 -> 1478 (s4f s0072-s0077).
- Leader routes after signalling stage A; union of chosen experts published first; gate|up shares per block from a
  model of start delays: s4n s0011 −2.5% (stage-B wait 4.2 -> 2.8 us per layer).
- Routing-dependent placement: whole w13 strips per block (4 at a time) at B=1: s7a s0007 −5.6% (12.1 -> 8.3 us per
  layer); flat schedule wins again above ~25 distinct experts (s7a F8). One fragment copy per down phase instead of
  per round: s4f s0064, B=8 3735 -> 3577 (−4.2%).
- NVFP4 input quantization with the hardware E2M1 conversion instead of a compare chain: s6 s0006 −4.1%, s7b s0010
  −7.1% (bit-identical to software rounding).
- tcgen05 block-scaled MMA (kind::mxf4nvf4) for NVFP4 w13 at B>=4, second weight layout (+46 GB per card):
  s7a s0016-s0020, B=8 5484 -> 5378 (−1.9%); layouts and descriptors in s7a F9.
- Expert layout does not matter (6 random 4 MB units 6.42 vs 6.41 TB/s contiguous, s4f 13a).
- B=8 unions: 27.8 experts per layer of 64 possible on North (s4n checkpoint 5), 38.4 of 48 on Flint, 31-55 on
  Inkling depending on text (s7a F7). Random token ids route differently from real text: profile on real text (s6).

### 2.6 Multi-card communication (Inkling, 4x B200)
- Split: attention TP4 by heads, experts TP4 inside every expert (576 of 2304 intermediate per card): all cards read
  1/4 of every chosen expert, no token all-to-all at B=1; router, mixes, convolutions replicated (s6).
- B=1 "red" mode: every block computes per-token vector phases redundantly in shared memory, only grid barriers
  around attention (s6). Later replaced by team vector phases (one element per thread on 6 blocks per token):
  s7a s0002 −13.4% (post-attention 15.5 -> 7.8 us).
- Cross-card partials as bf16 pairs with tags instead of fp32: s6 s0013 −8.5% (half the polled bytes; L2 bandwidth,
  not HBM, was the shared resource: ~64 MB of L2 traffic per card per layer from 152 blocks re-reading).
- Remote atomicMax for the w13 amax (~18K NVLink atomics per card per layer) replaced by warp max + local atomic +
  one remote store: w13 47 -> 23 us (s6). Release reduction to peer memory costs ~a system fence (~2 us); one
  system fence then relaxed reductions: o_proj+signal 19 -> 8 us (s6). fence.sc.sys after remote writes 2.05 us,
  fence.gpu 0.53 us, grid barrier of 152 blocks 1.25 us (s6 prim.cu).
- Loads all in flight before a spin (a spin loop blocks every load behind it): s6 s0008 −5.1%.
- Not paid: two-level all-reduce (+150 us, s6).

### 2.7 LM head
- The head streams at the bandwidth bound in every engine (s2 429 us at 2.9 TB/s; s4n 351 us for 1.07 GB).
- Gains came from bytes: 12-bit head (s4n head 351 -> 283 us; s5 lm_head 350 -> 281 us; s7b unembedding 77 -> 67 us)
  and the certified int8 sketch (s4n head 305 -> 192 us, section 3).
- Prefetch of layer 0 for the next launch from the head phase: s2 −6 us. Head in 16-tile batches: s4f s0015 (neutral).

### 2.8 Output / argmax
- Device greedy loop (argmax + token feedback in the kernel): s2 s0007 −1.1%.
- Greedy token gathered from all blocks in one trip instead of a chain of 302 loads: s4f s0037 −2.4% (1650 -> 1606);
  16 strides at B 3-8 (s0039).
- Host-side cost of not doing this: torch argmax 15-18 us + launch and host read 22-23 us per token (s10
  host_trip.py); 80-170 us per step on the host-driven path (s4f step_cost.py).
- NaN never wins the device argmax (`v > bv`), unlike torch.argmax (q8b audit item 10).

### 2.9 Host loop, generate, HTTP path
- Device loop behind submit/poll (one launch per serving cohort, tokens polled from a second stream): part of
  s4n s0006 −7.5%. Tokens written by the kernel into host-mapped memory: s9 s0005 −0.15%.
- The score switched from in-process step to HTTP mid-run on s4n and s4f: the same kernel measured +18% (s4n
  s0001 3497 -> 3603) and +12% (s4f 2679 -> 3003). The HTTP workload runs positions 64-1088, so long-context
  attention matters there: s4n s0007 missed its B=8 prediction by 390 us because it had predicted from contexts 64-320.
- Four cards launched by four Python calls start 30 us apart; in the device loop within 2.4 us (s7b G13).
- Serving engines (s3, s8) are a different design (passes of up to 2048-3584 rows, device-side scheduler in one CUDA
  graph WHILE loop / scheduler kernel, paged K/V, prefix cache). Largest serving gains: wgmma/bulk-copy GEMM (s3
  s0005 0.831 -> 0.905), home card per row (s8 s0012 0.963 -> 0.979), 128 sequences in flight (s8 s0037 0.836 ->
  0.990 at 6.5 req/s). Steady decode of the s8 serving engine was 11.0 ms vs 3.3 ms for the megakernel it replaced.

---------------------------------------------------------------------------------------------------------------

## 3. Track B: model-specific changes

### 3.1 Lossless 12-bit weight code (W12)
Census results (all exact counts over all weights):

| model | escapes outside the window | bits per weight | 11-bit alternative |
|---|---|---|---|
| Qwen3-8B | 825,104 of 7.57e9 (1.09e-4), per-row 16-binade window; 0.6-1.2e-4 per tensor (s2) | 12 + table | not measured |
| North | 3,710,152 of 30.48e9 (1.217e-4), max 56 per 16x2048 tile (s4n) | 12.07 | 7-value window escapes 2.4-2.9% |
| Flint | 6.3e-5 (s4f) | 12.002 | 1.3% escapes |
| Qwen3.8-27B | 15-window 1.48e-4; 16-window 7.5e-5 (1,922,136 of 25.6e9) (s5 K4) | 12.015 / 12.01 | "several percent" |
| Inkling dense BF16 | 7.3e-5 per-row 16-window (s7a F1); unembedding 55,788 of 926M | 12.0009 | — |

No checkpoint had zeros, duplicates or dead structure to exploit (exponent entropy 2.56-2.65 bits everywhere;
Inkling NVFP4 codes use 3.97 of 4 bits, MXFP8 6.53 of 8: no smaller fixed-width code for experts, s7b G1).

Format variants and decode cost:
- **s2 (H100)**: code = sign<<11 | (exp−109)<<7 | mantissa; row r stores W[r]·2^shift[r] (exact); escapes stored as
  code 0 (= +2^-18) and corrected by one extra MMA per 8 escapes (k-slot e holds the true value, slot e+8 holds
  −2^-18). Per-tile 256 B table at a computable address with 3 inline chunks. First decode ~31 integer ops per MMA
  (~35 with overhead); microbench 3120 GB/s packed vs 3134 GB/s 16-bit. Result s0025 −15.3% (5336 -> 4522; 15.1 ->
  11.4 GB). It took: inline exception chunks + prefetch of the next tile's entry (first tile 4.3 us instead of 0.7
  otherwise), zero spills (4547 -> 4511). Exceptions cost ~80 us B=1 / ~120 us B=8 (-DNOTAB 4434).
  Then: exact per-column power-of-two scales folded into the RMSNorm weight / up-row scale, fitted by 3 alternating
  row/column rounds, worst tile 553 -> 44 escapes (s0026 −1.3%); "mask-and-OR" layout with window base 0x70 so
  bf16 = sign<<15 | 0b0111<<11 | code[10:0], `a_r = (w_r & 0x87ff87ff) | 0x38003800`, 11 ops per MMA instead of 31,
  SM compute limit 2.3 -> 1.59 us per tile (s0028 −2.2%, logits bitwise those of s0027).
- **s4n (H100)**: W = 2^(r_i + c_j)·V; the 4-bit exponent code sits directly in the bf16 exponent field (values near
  2^-120) and x is stored ×2^96, so no "add base" step; one shift per expert neuron for 1,210 near-dead neurons
  (without it, up to 5,936 escapes in one tile). Escapes kept exactly in the tile header. s0007 −10.4% (2933 -> 2628).
  First version 2741 (a loop with a load and branch per escape); escapes sorted by row group + early x gather -> 2590.
  Decode had capped the stream (15 GB/s per SM, ~12 instructions per register); hi bits re-laid out with rotations
  (0, 4, 8, 28) so a register is PRMT + SHF + LOP3: s0012 −1.9%.
- **s5 (B200)**: [sign][exp − lo_row (4 bits)][7 mantissa], stored as [sign|mantissa] byte planes + an exponent-nibble
  plane, decoded with a byte permute (5-6 ops per register); one window per output row (per tensor, a few
  residual-stream rows held thousands of escapes, layer-0 down_proj one row 12,359, serializing the slowest warp;
  max escapes per warp in_proj 820 -> 20). Escapes patched into the MMA operand registers from per-warp fixed-stride
  lists, so every GEMM output and logit is bit-identical to BF16 storage. With every escape patched, code 0 needs no
  escape meaning and a 16-exponent window halves escapes (s0009 −1.7%). Path 8940 -> 8095 (−9.4%) over 13
  submissions in 5.6 h; s9 s0007 −8.6% on the later kernel (8587 -> 7847). Byte saving 25%, stream gain 10-14% on
  long shares, ~0-7% on short ones; out_proj +11% at B=1 (6 quads per warp, ~1 escape per warp).
- **s7b (Inkling)**: 12-bit unembedding only paid: GEMV 77 -> 67 us, −9.2 us (−0.20%). 12-bit for router (+75/+91 us),
  o_proj (+25/+16 us) and qkv (+8 us at B=1 in a same-library A/B; an apparent −10 us had been a build difference).
- **s4q (B200, first attempt)**: W12 on the track-A engine was +19.6% (s0050: 10687 vs 8938); "per-quad patches inside
  the decode loop cost more than the bytes saved". The same model gained −9.4% in s5 with per-row windows, patches
  after the quad's loads, list refills a half-window ahead and no unused stream call sites (+0.7% per pass if present).
- **s4f (Flint)**: W12 benchmarked (+17-27% stream rate in isolation) but not built: decode 0.8 us per 16 tiles and
  warp; with 0.85 us of other work between request and use the W12 batch would take 2.7 vs 2.2 us; estimated ≤5-10%
  in gate/up and a loss in down.

Decode-speed facts: H100 decoded as fast as HBM delivered once the decode was ~11-12 ops per MMA (s2, s4n). B200
microbenchmarks showed +23-27% values/s for W12 (s4q C6, s7a F2), but inside the engine at 64 registers per thread
the W12 loop held 48 B per thread in flight vs 64 for BF16 and was limited by bytes in flight, not decode (s5 K2):
a decode-free 12-bit loop was only 2% faster; +54 integer instructions per quad cost +20%.

Bit-identity tricks and failures:
- Sparse fp32 correction of escapes after the sum (s5 s0002): exact in exact arithmetic, but changed the rounding of
  1-3 outputs per op; the evaluator's numeric_long max rose 0.457 -> 0.713 (limit 0.592): invalid. Patching values
  into operand registers before the MMA made it bit-identical (s5 s0004 valid).
- s4n escapes summed after the tile and 1,334 tiny weights underflowing: audit found logits bitwise equal at 0 of
  32,768 positions vs the same engine without W12, greedy tokens identical for 8 of 64 prompts ("same results"
  refuted). s2's exception MMAs likewise change the f32 order of those terms (track B was never claimed
  bit-identical to A there).
- Exponent shift tricks are exact only while scaled values stay normal fp32 (fast-math flush-to-zero; q8b audit doubt 4).
- Verification used by runs that passed audits: decode every packed weight back with the kernel's arithmetic (s5
  `MK_W12_VERIFY`, s4n verify_w12.py on all 30.5e9 weights), round trip asserted at every load (s7b s0036), per-GEMM
  bitwise check vs BF16 storage at B=1 and B=8, logits hash over 20-48 steps.

### 3.2 Layer-0 per-token tables
- Layer 0's input is the embedding row, so its norm and projections depend on the token id only.
- s4n (parallel block: attention and MLP both read norm(x)): q|k|v [262144x5120] + dense-MLP output [262144x2048],
  3.76 GB, built in 3 s; in process B=8 −61 us, B=1 +13 us (within 1% spread); HTTP s0008 −1.1% (gain at long context,
  where layer 0's attention had waited behind the MLP's 38 MB). Audit: the tables were built with another float32
  summation order; they differ from the kernel's own layer 0 for 224,203 of 262,144 token ids.
- s7b: normed embedding + q, r, sink, k, v rows built by the engine's own kernel at load (3.9 s), verified for all
  201,024 ids on all 4 cards: −13.3 us (−0.31%) on its own kernel. On the sibling kernel the output-gate rows had to
  be left out because track A summed them in a batch- and position-dependent order. Rest of layer 0 (1/rms of the
  normed embedding from a table): −1.1 us in its phase, nothing end to end.
- Does not extend to layer 1 (its input contains layer 0's attention output).

### 3.3 Certified 8-bit LM-head sketch (s4n only)
- Tied head 262144x2048 quantised per row to int8, s_i = max|w_i|/127; input as two exact int8 digits so integer
  sums are exact; per-logit bound en_i·||x|| + qn_i·||r|| with en_i >= ||w_i − s_i q_i|| + 2^-10 ||w_i|| (float32
  rounding of the exact path), norms in float64 rounded up at load. Pass 1 reads 541 MB instead of 805 MB with the
  integer tensor core (m16n8k32 s8, fragment layout found by probing); one exchange gives the best lower bound;
  pass 2 recomputes tiles whose upper bound reaches it with the exact head's arithmetic: 1.2-1.5 tiles of 16 rows
  (of 16,384) per sequence and step. Exact tile on one warp took 35 us (widening was the limit); 16 warps widening
  into shared memory: 5.7 us. Head 305 -> 192 us; s0010 −3.8% (2572 -> 2475). Argmax identical in 4,608 / 4,608 own
  tests and 208,896 audit positions.
- Only on the greedy path (generate / serving); `step()` logits use the exact head. A 4-bit sketch would have ~20x
  wider bounds; low rank was excluded by a flat spectrum (top 256 directions hold 30% of the energy).
- Audit doubt: the 2^-10 rounding allowance is an argument, not a proof (mma accumulation rounding is undocumented);
  under a pessimistic model it holds with ~2x margin.

### 3.4 Float32 intermediates
- s2/s10 track B: float32 residual stream, unrounded float32 activations, fused RMSNorm. s10: track B 4136 vs track A
  5365 with three differences (12-bit, f32 intermediates, fused norm); error median 0.0074 vs 0.0157 for A
  (numeric_long). Audit: "track B" is not "track A + lossless weights"; it changes data flow too (operator's call).
- s4f "track B": float32 residual + hi+lo BF16 router operands, built for routing experiments that did not pay; ~6%
  cost in bytes and instructions at B=8 (3950 vs 3877), faster at B=1 only through a better register allocation
  (1475 vs 1520). Audit: 0 of 14,336 positions equal to its track-A build, 10.8% different top-1; hi+lo carries
  ~16 significant bits, not 24. Not submitted as track B afterwards.

### 3.5 Per-operator format choice
- s5 s0014: BF16 twins of out_proj/o_proj used by one-column passes (W12 slower there): B=1 −0.55%, B=8 +0.3%;
  memory 116 GB vs 64 GB for track A (audit). "Which storage pays depends on the op and the batch" (s5 K9).
- s7b: only the unembedding in W12; build-time decoder choice vs load-time made 8-10 us difference at B=1 (s0030 vs
  s0031) through code generation, not through the GEMV (0.8 us).

### 3.6 Other track-B / model-specific items tried
- Exact draft-and-verify ("track C", s4q): MTP head + n-gram copy drafts, up to 32 positions per verify pass (a
  32-position pass cost 1.33x a step, 8 positions 1.04x, s4q C10); 414 us per token at B=1. Audit: a loop artefact
  (4 of 11 evaluation prompts fall into a two-token loop); on natural text 3.7 ms (B=1) vs 8.87 ms for track A
  (2.4x) and 5.5 vs 9.8 ms at B=8. The operator ended track C (note of 2026-11-12 09:10), read by the agent 14 h late.
- s4n studied copy drafts (8-token drafts accepted 6.2 tokens on average; 8-position pass 2.49x a step; estimated
  ~860 us per token) but did not build it.
- Speculative gate|up for the previous token's experts (4.62 of 8 repeat, s4n): bit-identical, slower (2927-2964 vs
  2880; bookkeeping cost more than the 4 us of router latency).
- Early routing via M = R'·W_o (s4f 14): numerically right (9216/9216 sets) but slower (1635 vs 1577); predictor
  running unused 1500 -> 1899 us. Prediction from R'x alone: recall 82.6% at top-6, 97.1% at top-16; not built into the
  engine.

### 3.7 Track-B ideas mentioned but never tried
- Certified int8 head sketch on Qwen3-8B (head 1.245 GB, ~8% of bytes), Qwen3.8-27B (lm_head 2.54 GB, 281-350 us),
  Flint (head ~70 us of ~1380), Inkling (unembedding 65 us): built only on North.
- Layer-0 token tables on Qwen3-8B (layer 0 q/k/v read the normed embedding), Qwen3.8-27B and Flint: built only on
  North and Inkling.
- Computing each distinct head row once (North: 7,129 duplicate rows, 2.7% of the head, 0.44% of a B=1 step).
- s2: column scales for the o projection (needs the V rows of qkv scaled, tied over 4 heads; ~3% of its exceptions).
- Bit-identical North track B: "tables made by the kernel itself and escapes summed inside their k-step" (audit).
- s5: a stream structure with registers to spare (fewer, fatter warps or producer/consumer warps) to absorb escape
  patches (escape-free loop is 3.7% faster); measuring every op in both formats at each batch size.
- s7b: 12-bit unembedding loop with two units in flight like the BF16 loop.
- s2 correctness insurance: second down-projection pass for the last layer with the low part m − bf16(m) (+34 us).

---------------------------------------------------------------------------------------------------------------

## 4. Dead ends (hours spent, nothing gained)

- **Load imbalance as a target** (s2, ~5 h): dynamic tile claiming (built fully, correct, +21 / −25 us), stride-
  permuted claims (60-75 us worse), fractional tiles of 1/31 (correct, no gain), tile counts from measured per-SM
  rates (+100 us). Reason: HBM is the shared limit; when some SMs finish early the others get their bandwidth;
  barrier waits measured slack, not waste. Per-SM speed differences were not static between engine instances.
- **Run-ahead and dependency-started phases** (s2): correct, bitwise equal, 104.05 vs 103.68 us per layer; run-ahead
  108.5. The coordinator's epilogue queued behind its own block's new loads (SM memory responses are in order).
  Register preload of the next phase across the barrier: +90 us (and +400 when spilled). s4n "issue next stage's
  loads before the barrier" (v3): 3370 vs 3050.
- **L2 staging with prefetch** (s4n §10): prefetch.global.L2 does not stage on H100 (17.3 MB read 4.8-5.8 us after
  prefetch vs 6.6-7.0 cold); cp.async/bulk staging +40 / +70 us because the FIFO queue delays the skeleton's own
  round trips by as much as it saves. s2: K/V prefetched during q/k/v +55 us ("L2 does not keep it through 12 us of
  weight stream"). s4f: speculative expert prefetch measured not built (L2 keeps ~64 MB, loses it to 48 MB of
  other traffic). B200 `prefetch.global.L1/L2` "behave like blocking loads" (s4f 11, 15).
- **Shared-memory staging of tiles or K/V**: s2 K/V image copy (5951 vs 5872); s4f smem loads/stores 60-90 GB/s per
  SM, "staging through smem is a net loss"; s5 2-stage cp.async ring −6% per stream in isolation, +3.5-4% in the
  engine (96 KB of smem take L1 from activations).
- **Fence-free or counter-only hand-overs**: s4n all-atomicExch (3040-3207 vs 2920); s4f atomicExch + bump not strict
  (1 of 357), E_CTR counter −17 us and not strict (4 mismatches in 398); s9 relaxed atomic add in the barrier
  −157/−196 us and wrong (864 stale reads in 1.25e11, bench/rel.cu).
- **More parallelism under a register cap**: s6 deeper unrolls / two strips at once / fused phases all lost to spills
  at 80 registers; s4f register-operand down phase (156-180 B spills, 4120 vs 3854); s9 kernel pruned to its own GEMM
  instances +33% (spills); s4f split attention pair for B=7-8 (correct, slower in all three forms).
- **Register-allocation lottery**: s4f batch-1 kernel 1481 -> 2005 us with study code compiled in; s7b builds of one
  layer code differ up to 10 us at B=1 and 60-110 us at B=8; s4n 96/112/144 B of stack for equal code moved the
  step by up to 40 us. Many "negatives" in s4f and s7b were allocation draws, not ideas.
- **Track-B work that did not pay**: s4q W12 twice (+20%); s7b W12 for router/o_proj/qkv; s4n 11-bit (2.6% escapes);
  s7b W12 qkv "gain" that was a build difference (one wrong conclusion acted on).
- **Track C on s4q**: ~11 submissions (s0039-s0049) after the operator had ended it.
- **Inkling multi-card**: two-level all-reduce (+150 us), L2 bulk prefetch per warp range, cp.async rings for w2,
  tcgen05 for w2 (correct, not faster: row groups split over ~4 blocks), tcgen05 MXFP8 w13 (8% per expert, cancelled by
  18 KB more smem) (s6, s7a).
- **Serving scheduler knob sweeps** (s3): "never 181 under the tbt line; the share sits on a knife edge".
- **Runtime ablation branches inside the kernel** (s2 day 1): the branches alone slowed the step by 420 us and made a
  correct optimisation look 500 us slower.

---------------------------------------------------------------------------------------------------------------

## 5. Correctness traps

| trap | where | how it showed | how caught / fixed |
|---|---|---|---|
| L1 not coherent across SMs | s4q C2 (100% stale re-reads with default ld), s4f 12a (2.8%) | wrong activations | `ld.global.cg` / volatile for data written in the same launch; s4n: one fence per layer per block flushes stale L1 |
| Spin loop removed or never sees the counter | s4q C3 | barrier hang, or blocks running through the barrier ("silently worked" in a microbenchmark) | `ld.volatile.global` or `atomicAdd(p,0)` |
| Weak load on a sentinel poll | s4q C11 | 16-35% receivers read the previous op's data on B200 | `ld.relaxed.gpu` / `st.relaxed.gpu`; epoch-tagged debug build |
| Plain store not ordered before a later atomic | s4f 12b, s4n, s7a F6 | ~8% of 600-token generations differed between processes (s4f s0005-s0008: the 1685 headline was a racy kernel); B=8 irreproducible after 30-70 steps | flake.py repetitions (200+ before hand-over changes), `__threadfence` in storing warps, sentinel or tagged words |
| Missing block barrier before a grid barrier | s2 s0031 split attention | logits off 1-2%, inside tolerance | bitwise comparison between batch sizes, not the tolerance |
| B=1 smem fragments read without `__syncthreads` | s6 | runs differed from step 0 (grid barrier hid it at B>=2) | bitwise B=1 / B=8 reproducibility |
| Race behind timing at B>=2 | s6 s0010 | combine read rows of the second qkv wave; all gates passed | found by code review, fixed s0011 |
| mbarrier phase-count race | s7a s0014 | deadlock in the evaluator's timing stage, only with 1-3 distinct experts (shared prompt tokens) | soak with --distinct 1/2/3 for 700-5000 steps; one barrier |
| Host ran > 64 launches ahead of a 64-slot header ring | s6 s0001 | deadlock in timing stage; own tests never queued > 64 steps | header by value; rehearse.py reproduces the evaluator's unsynchronised loops and HTTP path |
| Batch-slot dependence | s7b s0001 (inherited s6 engine invalid: http_tokens) | a sequence's logits depended on its slot | explicit FMAs in expert scaling (slot-order invariant) |
| Different FMA contraction in two compiled copies | s4f s0017-s0019 | sequences 6-7 differed by a rounding step alone vs batched | bcheck.py checks every batch size and sequence (gate checks one); intrinsics |
| Batch-dependent MoE results | ink audit | max logit diff 6.48 alone vs batched: a 1-ulp difference from summation order (expert union changes warp split) amplified by near-tie expert flips; 37% of (layer, token) expert sets differ | legitimate per audit; user-visible (two requests sharing a cohort produced different greedy tokens) |
| Near-tie routing makes top-1 gate a lottery | s4f 3 | margin 6th-7th score median 0.011; top-1 moved ±1.5 points between equally exact orders; s0003 failed at 91.6% (limit 91.9%) | forced-routing distance (s4n) separates arithmetic from routing |
| BF16 rounding of attention scores | s10 | inherited track A failed numeric_long (0.0192 / 1.183 vs limits) | HF eager vs SDPA layer-by-layer comparison; `SCORE_RB 0` (f32 scores), no speed cost; audit: median fixed (0.99x HF), largest error re-drawn (failed max rule on 1 of 3 audit sets) |
| Largest-error gate is a draw | s2, s10 | any change of arithmetic order re-draws the tail; 14 jitter draws 0.46-1.85x HF's largest | keep speed changes bitwise identical (cmp_exact.py) or re-measure with jitter builds |
| Sparse fp32 escape correction | s5 s0002 | numeric_long 0.713 > 0.592 | bit-identical patching |
| "Lossless" claims not bit-identical | s4n audit (W12 after-sum escapes, L0 tables), s4f audit (f32 stream), s10/q8b audit | 0 of 32,768 / 0 of 14,336 equal positions | the evaluator never compares track B with track A; only the audits did |
| Long-context and B=2-7 paths never run by the evaluator | q8b, ink, flint audits | split attention from 1136, B=1 128-position twin kernel from 2048, B=2-7 kernels unchecked | audits ran 1136-1399 positions and B=2..8 bitwise; s2 test_ctx to 9216 |
| No context guard | ink audit, flint audit, q8on audit | past max_context K/V written into the next slice; >2176 positions overwrite chunk records; served request over limit returns 0 tokens, "finished"; online positions past max_context read past the RoPE table | ValueError guards (s7b s0030, s4f s0059) |
| No input validation | q8b, ink, q8on audits | token id >= vocab reads past the embedding or faults; flint reduced ids modulo vocab silently; B > max_batch writes past K/V | submit() range checks (s7b s0033, s8 s0025) |
| Hash-only prefix matching | q8on audit | a constructed 64-bit collision gave one prompt another's answer | content verification of cache hits (s8 s0036 "cache matches verified") |
| Served path checked leniently | north, flint, q27 audits | the http_tokens gate: 8 prompts, mean prefix >= 0.75, half identical; timed tokens never checked | audits compared served tokens with generate() |
| Persistent kernel without cooperative launch | q8b, flint audits | spinning grid barrier hangs if any SM is taken | documented limit |
| Build/env switches that skip work | all audits | MK_DEFS/-DNOEXC, ABL_NOSTREAM, MK_W12_DROP_ESC etc. inactive in scoring | `stats()` labels (s4q mislabeled track C as "B") |
| Stale prebuilt library | s2 | build.sh silently kept the old .so when a cleaned build failed; an A/B script compiled the current source whatever SRC said (an hour of "drift") | tests `rm` the library first |

---------------------------------------------------------------------------------------------------------------

## 6. Process lessons

### 6.1 What slowed the agents
- **Evaluation turnaround**: median submit-to-result 1.9 min (s2), 2.1 (s4f), 4-4.5 (Inkling), 5-6 (s4n, s4q, s5, s9),
  8.4 (s8 serving); eval seconds median 106-330 for decode, 492 for serving. Worst cases 17 min (s2), 129 min (s4q),
  192-265 min (s6, s7a) when a pod was lost during evaluation.
- **Pod loss**: pod epochs per run: s2 9, s4n 11, s4f 24, s4q 26 (replaced four times in hours 2.5-2.6), s5 7. s6 had no
  pod access from 14.0 h to the end (its B=8 work, −4% in process, never reached the evaluator; s0014 had failed to
  build through an agent error just before). Engines that time themselves at start-up (s4f counter placement, ~13 s)
  and own HF references had to be rebuilt after each replacement. Nsight Compute runs coincided with pod losses on s5.
- **Noise floor**: same kernel re-evaluated: s2 ~20-30 us between pods (s0059 4047 vs s0061 4026); s4f ~6 us B=1,
  ~30 us B=8; s7b 1.2 / 2.6 us on one pod; nodes differ ~0.5%; s4n ±10 us run-to-run (role placement). Separate builds
  of the same code differ more than repeat runs (s7b: up to 10 us B=1, 60-110 us B=8; s4f: code compiled in but never
  run moves B=8 by 0.6%). Late-run gains of 0.1-0.4% (s2 s0052-s0063, s4f s0072-s0077) were below one evaluation's spread.
- **Profiling a megakernel**: nsys shows one kernel ("the pass is the kernel", s4q); Nsight Compute gave stall shares
  (long_scoreboard 58%, barrier 26%) and L1/L2 traffic for a launch of 20 ops, not phase times. Profiling builds
  perturb what they measure: s2 TIMING build ran ~150 us slower with spills ("trust differences, not totals"); s2's
  software trace clock (a counter in global memory) had a traffic-dependent tick rate; %globaltimer ticks in ~0.5 us
  steps (s4q C12); s4f's profiling build used loads as markers and "misled where the question is who waits for whom";
  two extra stamps moved s7a's kernel to STACK:104 (+1.4% at B=8).
- **Workload mismatch**: s4n and s4f predicted from contexts 64-320 while the HTTP headline runs 64-1088 (HOSTC on s4f
  had the opposite sign there). s3: a 40 s quick online evaluation passed and hid a liveness failure of the full
  replay. MoE builds that change rounding generate different text and see different expert unions (s7a s0015 predicted
  −7% from generated text, measured 0).
- **Other**: s4q read the operator's note ending track C 14 h late; s7b spent 11 h on its own kernel and then gained
  −16.7% in one step by adopting the sibling run's kernel (s0029); s4n lost mk.cu to a patch script that opened the file
  for writing before an assertion (now temp file + rename); s9 timed a change on the A build only and submitted a +27 us
  B build (s0015).

### 6.2 How phases were instrumented inside the megakernel
- `%clock64` stamps by one thread per block written to shared memory and flushed at layer end, no memory operation in
  the measured path (s4f bt_clk.py, s4q ho_stamps.py): found the routing top-6 cost (s4f) and the integer hand-over
  work (s4q C12). Per-SM cycle counter 1830 ticks/us on H100 (s2 clk.cu).
- Ablation / "what-if" builds of the normal kernel (results wrong on purpose): s2 phase-repeat builds (-DREP_x=2: run a
  phase twice per layer, delta/36 = in-place cost; showed o costs 9 us even with weights in L2); s4n skeleton build
  (no weight streaming: 826 of 2588 us); s4q streams off (1.8 ms); s4f -DWHATIF bits; s5 escape lists emptied (300 us).
  "Changes measured on the skeleton build transfer to the full build" (s4n §9).
- Microbenchmarks before engine changes: s2 bw.cu / bw12.cu (12-bit decode keeps up before building it); s4n hop_bench,
  pf_bench (killed the L2-staging plan before engine work); s4f bench1-24; s4q bench/burst, coh, bar; s7a tc5.cu.
- SASS / ptxas checks after every change: spill lines, LDL after CALL (s2 sass_spills.py, s4q sass_ldl_after_call.py,
  s4f sass_seg.py).

### 6.3 Working methods of the fastest-converging runs
- Convergence (hours to reach within 10% / 5% / 2% of the run's final best): s2 12.1 / 13.4 / 23.0 of 39.4; s4n 15.4 /
  15.4 / 18.6 of 22.6; s4f 13.7 / 15.6 / 21.3 of 35.9; s5 1.1 / 2.0 / 2.8 of 5.6; s9 2.4 / 2.4 / 3.4 of 7.2; s7a 3.8 / 3.8 /
  4.2 of 11.5. Runs that inherited an engine converged in hours; from scratch it took 12-16 h; the last 2% took
  another 5-10 h.
- Submission cadence: median 0.25-0.34 agent-hours between submissions on s2, s4f, s4q, s5, s9, s7b; 0.72-0.88 h on
  s4n, s3, s6.
- Prediction before each result: s2 wrote "Expected / Measured / Concluded" for every entry and its predictions held
  to a few us (e.g. s0025 expected 4520, measured 4522); s9's predictions were within 12 us at B=1 except three;
  s4n gave a range per submission. Wrong predictions were followed by a measurement of why (s2 s0009: generate path
  not A/B-timed -> tests/time_step.py now times both).
- Same-card interleaved A/B of whole steps against the exact previous build, several rounds: s2 tests/ab.sh
  ("whole-step A/B only"; reproduced the evaluator to 0.1%), s4n tools/ab.sh after s0015, s9 tools/abn.sh with a logits
  hash at B=1..8, s4f abg.sh on the scored workload (gen_long.py: prompt 64 + 1024 tokens, 35 s per build), s7b switches
  inside one library ("never compare separately built variants"), s7a t_replay.py (teacher-forced replay of one
  generated text through both kernels, predicted s0016-s0020 to within 0.2%).
- Bitwise identity as the default acceptance test for speed changes: s10 cmp_exact.py, s4f digests + flake.py 200-399
  repetitions, s9 logits hash before every submission, s4n logit_hash.py ("every change after s0012 left outputs bit for
  bit unchanged"). With identity kept, the numeric gates did not move and no tail re-draw risk was taken.
- Full test suite per track before submitting (s2 run_all.sh: vs HF f32, B=1..8 bitwise, 700 positions, cross-split
  generate, soak 11k steps, split-vs-one-block bitwise at 2400; s4f vfy.sh in 4 parts on 4 cards under time caps).
- Reading other runs' progress notes paid: s4q took the sentinel hand-over from s4f (−1.4%), s4f took the strong-load
  rule back from s4q; s2 at 37.6 h checked s4n's shallow-queue finding against its own engine (deep queue was better
  there: +63 us with GB 2). s7b's largest gain was the sibling's kernel.
