# s2-q8b-a1: Qwen3-8B on H100

Score: `step_us_b1` (lower is better). Budget 40.0 agent-hours. 66 submissions so far, 66 valid.

Reference lines of this run:

- bandwidth bound (100% MBU): 4520
- NEW 2026-11-10 target: 1.10x the fastest general engine: 5515
- NEW 2026-11-10 SGLang 0.5.18 + torch.compile (fastest general engine): 6066
- SGLang 0.5.23: 6689
- vLLM 0.32.0: 6789
- MPK (megakernel compiler) @ 97de2592: 7252
- ForgeMegakernel, as reported in its paper: 5601

The source of submission `sNNNN` is `src/sNNNN.tgz`; the run's own notes are in `notes/`.

## Submissions

### #1 s0001: starter: Hugging Face reference

- status: **valid**, step_us_b1 = 2.105e+04, at 0.06 agent-hours, category structure
- hypothesis: Unchanged starter (HF model behind the contract): origin of the progress plot; expect valid, step_us_b1 well above the general engines (~20-30 ms), many launches and syncs per step.
- evaluator: valid: 21051 us/step at B=1 (MBU 0.215), 20836 us/step at B=8

### #2 s0002: own CUDA operators: MMA matvec block team + attention, one launch per operator

- status: **valid**, step_us_b1 = 7366, at 0.37 agent-hours, parent s0001, category structure
- hypothesis: Replace HF with my own kernels (tensor-core bf16 matvec with f32 accumulate, weights re-laid out as MMA fragments, f32 residual stream), one launch per operator (182 launches/step, no host sync). Expect valid with error below HF bf16, bitwise batch equality, step_us_b1 about 7000-7500 (launch gaps between 182 kernels cost about 1 ms).
- evaluator: valid: 7366 us/step at B=1 (MBU 0.615), 8062 us/step at B=8

### #3 s0003: whole decode step in one kernel launch, grid barriers between phases

- status: **valid**, step_us_b1 = 6465, at 0.40 agent-hours, parent s0002, category launch
- hypothesis: Same operators inside one kernel per step; the 181 phase boundaries become in-kernel barriers (atomic counter + polling) instead of kernel launches. Expect launches/step = 1, step_us_b1 about 6450 (-12%), B=8 about 7300.
- evaluator: valid: 6465 us/step at B=1 (MBU 0.700), 7353 us/step at B=8

### #4 s0004: down-projection entry: copy only live batch columns, no zero fill

- status: **valid**, step_us_b1 = 6385, at 0.96 agent-hours, parent s0003, category memory
- hypothesis: The down entry rewrote all 196 KB of shared-memory fragments every layer (zeros for unused batch columns, 12 stores per thread); now only live columns are copied and attention blocks wipe their own scratch. Bench showed the entry at 4.2 us/layer at B=1; expect step_us_b1 -80 to -120 us, B=8 unchanged. No sync point added or removed.
- evaluator: valid: 6385 us/step at B=1 (MBU 0.709), 7340 us/step at B=8

### #5 s0005: lean phases: tile tables, fixed-trip unrolled loaders, coordinator warp, no divisions

- status: **valid**, step_us_b1 = 6139, at 1.07 agent-hours, parent s0004, category gemm
- hypothesis: Per phase every warp ran a 64-bit division call, I2F/RCP divisions and a remainder-unrolled loop; now tile ranges come from a host table, loaders run 8 (or 24) unrolled ksteps (+1 predicated), one warp only reduces (tree sum) and drives the barrier (3 block syncs per boundary instead of 5). Each phase is a separate noinline function so codegen of one cannot disturb another. Expect step_us_b1 about 6135 (-250), B=8 about 7320 (-20). Logits change at the bf16-rounding noise level (different but fixed summation order).
- evaluator: valid: 6139 us/step at B=1 (MBU 0.737), 7320 us/step at B=8

### #6 s0006: fuse RMSNorm into the previous epilogue: publish bf16(w*x), scale matvec outputs by 1/rms

- status: **valid**, step_us_b1 = 5951, at 1.17 agent-hours, parent s0005, category structure
- hypothesis: The per-block RMSNorm entry (every SM re-reading the f32 residual, 12 us/phase at B=8) is removed: the residual-add epilogue publishes bf16(w*x) fragments and per-tile sums of x^2; the next phase copies fragments (8 KB per sequence) and multiplies its OUTPUTS by 1/rms (the matvec is linear). No barrier added. Expect step_us_b1 about 5950 (-190) and B=8 about 6420 (-900): both under the 1.10x targets.
- evaluator: valid: 5951 us/step at B=1 (MBU 0.761), 6422 us/step at B=8

### #7 s0007: generate(): greedy loop on the device (argmax + token feedback in the kernel, one launch)

- status: **valid**, step_us_b1 = 5886, at 1.24 agent-hours, parent s0006, category launch
- hypothesis: Adds generate(): n_steps decode steps in one kernel launch; lm_head epilogue keeps a running argmax per block instead of storing logits, the next step's embedding block reduces the 132 block maxima (ties to lowest row) and feeds the token back. One extra barrier per step (182), no launch gap, no logits store. step() path unchanged (expect 5951/6422); expect generate_step_us about 5900 at B=1 and 6350 at B=8, gate generate passing with identical tokens.
- evaluator: valid: 5947 us/step at B=1 (MBU 0.761), 6418 us/step at B=8

### #8 s0008: SMs idle during attention stream their first O tile into shared memory

- status: **valid**, step_us_b1 = 5810, at 2.37 agent-hours, parent s0007, category scheduling
- hypothesis: During attention only 8B SMs work. The others now load their share of the block's first O tile (131 KB) into the unused part of shared memory, so those bytes cross HBM during attention and the O matvec starts from shared memory; attention blocks own one O tile, the others share the rest. No sync point added. Local A/B: step 5948 -> 5872 at B=1 (-76), 6422 -> 6414 at B=8; expect generate_step_us about 5810 / 6330. (A register preload variant was +400 us: the compiler spilled it to local memory.)
- evaluator: valid: 5873 us/step at B=1 (MBU 0.771), 6413 us/step at B=8

### #9 s0009: coordinator requests residual rows and norm weights before the tile barrier

- status: **valid**, step_us_b1 = 5859, at 2.69 agent-hours, parent s0008, category sync
- hypothesis: The residual-add epilogue (O and down) read its inputs (residual rows via L2, next norm weights) after the last tile's block barrier: an L2 round trip on the phase's critical path, twice per layer. The coordinator warp now issues those loads before the barrier, while the loaders are still streaming (loads do not block a warp). No sync point added. Local A/B: step 5873 -> 5858 at B=1, 6413 -> 6314 at B=8; expect generate about 5795 / 6235.
- evaluator: valid: 5859 us/step at B=1 (MBU 0.773), 6315 us/step at B=8

### #10 s0010: kernel = call-free interpreter of phase descriptors; loader and coordinator code split

- status: **valid**, step_us_b1 = 5771, at 3.29 agent-hours, parent s0009, category structure
- hypothesis: Findings: with 229 KB of shared memory the SM has almost no L1, so every spill to local memory (and the return address of a call into a register-hungry function) is an L2 round trip made by all 1024 threads; that explained the unexplained +-80 us between equivalent builds (s0009 generate regression). Now: one loop interprets a phase descriptor kept in shared memory (written by the coordinator thread), no calls, loaders and coordinator run separate code so the epilogue's registers cannot push weights out of registers; one copy of the matvec for all phases; step and generate share the code. Local: step 5781 / generate 5774 at B=1, 6280 / 6258 at B=8; expect headline about 5775 and B=8 about 6260.
- evaluator: valid: 5779 us/step at B=1 (MBU 0.783), 6258 us/step at B=8

### #11 s0011: 1/rms of a norm phase computed by warps that have no input-copy loads

- status: **valid**, step_us_b1 = 5756, at 3.43 agent-hours, parent s0010, category sync
- hypothesis: In a norm phase's entry, warp n did its fragment copy load and then two more sequential loads for the per-tile sums of x^2 of sequence n (three L2 round trips in a row on the critical path). Warps 16..23 now do the sums (they have no copy loads below 5 sequences). Local A/B: -17 us at B=1 (5783 -> 5766 step, 5775 -> 5758 generate), B=8 unchanged; expect headline about 5755.
- evaluator: valid: 5764 us/step at B=1 (MBU 0.785), 6258 us/step at B=8

### #12 s0012: SMs idle during attention first pull the layer's K/V history into L2

- status: **valid**, step_us_b1 = 5694, at 3.53 agent-hours, parent s0011, category attention
- hypothesis: Trace: the 8B attention blocks need ~10.7 us because their K/V loads are HBM misses queued behind the O-weight flood of the other SMs (each of 8+4 sequential loads per warp takes 0.5-0.7 us). The idle SMs' loader warps now first touch the K/V history of the layer (512-byte chunks, at most a few per warp, non-blocking), so the attention blocks hit L2. Local A/B: -60 us at B=1 and B=8 (step 5766 -> 5706, 6278 -> 6219); expect headline about 5697, B=8 generate about 6195.
- evaluator: valid: 5702 us/step at B=1 (MBU 0.794), 6228 us/step at B=8

### #13 s0013: attention q/k/v prep: one warp per vector, norm and RoPE by warp shuffles, no block barrier

- status: **valid**, step_us_b1 = 5659, at 4.01 agent-hours, parent s0012, category attention
- hypothesis: The q/k/v stage of attention used two block barriers and shared-memory round trips (norm sum across 4 warps, RoPE partner 64 elements away) plus 4 sequential loads per thread. Now one warp handles one vector with 4 consecutive elements per lane: RMSNorm is a warp reduction, the RoPE partner is lane^16, loads are float4. Summation order of the 128-element norm changes (fixed, batch-independent), so logits move at the bf16-noise level. Local A/B: step 5706 -> 5671, generate 5698 -> 5665 at B=1, B=8 unchanged; expect headline about 5660.
- evaluator: valid: 5666 us/step at B=1 (MBU 0.799), 6243 us/step at B=8

### #14 s0014: input copy: idle threads leave at once, loads requested in pairs before the stores

- status: **valid**, step_us_b1 = 5651, at 4.45 agent-hours, parent s0013, category memory
- hypothesis: The input copy ran a load->store chain per thread (each load a full L2 round trip: 12 in a row for the down projection at B=8) and every thread paid the loop overhead even without work. Now threads without work exit immediately, single loads take a short path, multiple loads are requested two at a time before the stores (4 at a time measured WORSE: +60 us, more instructions per thread per phase). Local A/B: generate 5665 -> 5653 at B=1, 6195 -> 6160 at B=8.
- evaluator: valid: 5658 us/step at B=1 (MBU 0.800), 6190 us/step at B=8

### #15 s0015: weight loads use the last-use cache operator (ld.lu)

- status: **valid**, step_us_b1 = 5595, at 4.68 agent-hours, parent s0014, category memory
- hypothesis: The weight stream (15 GB per step) passes through L2 with the default cache operator and evicts the small data the phases share through L2 (input fragments, residual rows, K/V history), so entries and epilogues miss. Marking weight loads last-use lets L2 drop those lines first. Same instructions otherwise; results bitwise unchanged. Local A/B (generate): 5649 -> 5602 at B=1, 6184 -> 6047 at B=8 (cg: 5618 / 6128, cs: 5637 / 6100).
- evaluator: valid: 5600 us/step at B=1 (MBU 0.808), 6061 us/step at B=8

### #16 s0016: loaders request 2 loads + the extra kstep of the next tile before the tile barrier

- status: **valid**, step_us_b1 = 5528, at 6.41 agent-hours, parent s0015, category gemm
- hypothesis: At each tile's block barrier every loader warp's load queue ran empty. Now a warp requests the first two ksteps and its extra kstep of the next tile right after writing its partial sum (12 registers carried across the barrier; loads do not block a warp). More than that makes the compiler spill elsewhere and loses. No sync point added, results bitwise unchanged. Local A/B (generate): 5600 -> 5524 at B=1, 6060 -> 5958 at B=8; expect headline about 5520.
- evaluator: valid: 5531 us/step at B=1 (MBU 0.818), 5975 us/step at B=8

### #17 s0017: O phase: request the second tile's first loads while the stash tile is computed

- status: **valid**, step_us_b1 = 5515, at 6.60 agent-hours, parent s0016, category gemm
- hypothesis: Blocks that take their first O tile from shared memory (the stash) reached that tile's block barrier with an empty load queue. They now request 2 loads + the extra kstep of their next O tile before that barrier, like every other tile since s0016. No sync point added, results bitwise unchanged. Local A/B (generate): 5525 -> 5509.5 at B=1, 5960 -> 5948 at B=8; expect headline about 5510 (new target 5515).
- evaluator: valid: 5518 us/step at B=1 (MBU 0.820), 5960 us/step at B=8

### #18 s0018: at the end of a phase, prefetch the next phase's first 2 ksteps per warp into L2

- status: **valid**, step_us_b1 = 5455, at 7.62 agent-hours, parent s0017, category memory
- hypothesis: HBM idles from a block's last load of a phase until the next phase has copied its input (reduce, epilogue, fence, barrier, copy). Loader warps now issue prefetch.global.L2 (no result, so nothing waits) for their first 2 ksteps of the next matvec phase right after their last tile; the next phase's first loads then hit L2. Deeper prefetch loses (the coordinator's own loads queue behind it). The coordinator publishes the next phase's start address in the phase descriptor. Results bitwise unchanged. Local A/B (generate): 5520 -> 5449 at B=1, 5962 -> 5913 at B=8; expect headline about 5455.
- evaluator: valid: 5459 us/step at B=1 (MBU 0.829), 5937 us/step at B=8

### #19 s0019: L2 prefetch in the attention and embedding windows (second O tile, attention blocks' O tile, first QKV tile)

- status: **valid**, step_us_b1 = 5422, at 7.90 agent-hours, parent s0018, category memory
- hypothesis: Same non-blocking prefetch as s0018 in the other windows where SMs have spare time: during attention the stash blocks also prefetch 4 ksteps of their second O tile; attention blocks prefetch 8 ksteps of their (cold) O tile when their attention is done; during the embedding phase every block prefetches 4 ksteps of its first q/k/v tile. Results bitwise unchanged. Local A/B (generate): 5449 -> 5409 at B=1 (from the second O tile), 5913 -> 5880 at B=8 (from the attention blocks' O tile); expect headline about 5413.
- evaluator: valid: 5424 us/step at B=1 (MBU 0.835), 5914 us/step at B=8

### #20 s0020: while a block waits at the grid barrier, its coordinator prefetches more of the next phase

- status: **valid**, step_us_b1 = 5398, at 8.28 agent-hours, parent s0019, category memory
- hypothesis: A block that has finished a phase polls the barrier counter until the slowest block arrives (about 2 us on average). Its coordinator warp now also issues L2 prefetches for the block's first tile of the next phase at every poll, one kstep deeper for all 31 loader warps in turn (8 chunks per poll, 16 with 1-2 sequences), up to 4 more ksteps. The block everybody waits for prefetches nothing, so the critical chain is unchanged. Results bitwise unchanged. Local A/B (generate): 5411 -> 5392 at B=1, 5891 -> 5878 at B=8; expect headline about 5400.
- evaluator: valid: 5401 us/step at B=1 (MBU 0.838), 5882 us/step at B=8

### #21 s0021: compile with fast-math: approximate division and rsqrt in the scalar epilogue code

- status: **valid**, step_us_b1 = 5382, at 8.58 agent-hours, parent s0020, category cleanup
- hypothesis: nvcc -use_fast_math replaces IEEE division and 1/sqrt in the float32 scalar code (norm scale, silu, softmax normalisation) by reciprocal / rsqrt instructions (error ~1-2 ulp of float32, far below the bf16 rounding of every matvec input) and drops their slow-path subroutine calls; fewer instructions on the coordinator and attention paths. The matvecs (tensor cores) are untouched. Logits change at the bf16-noise level; batch equality stays bitwise (checked for B=1..8). Local A/B (generate): 5396 -> 5378 at B=1, 5881 -> 5860 at B=8; expect headline about 5385 and unchanged error statistics.
- evaluator: valid: 5386 us/step at B=1 (MBU 0.840), 5862 us/step at B=8

### #22 s0022: 24 warps per block (80 registers per thread) with a 3-load tile pipeline

- status: **valid**, step_us_b1 = 5340, at 9.05 agent-hours, parent s0021, category structure
- hypothesis: With 32 warps a thread has 64 registers, and carrying more than 2 loads + the extra kstep across the tile barrier made the compiler spill. The block size is now a parameter: 24 warps (23 loaders + coordinator) leave 80 registers per thread, a 3-load pipeline fits without spills, a tile's partials are 23 instead of 31, and everything all threads do at a phase boundary shrinks by a quarter. 16 warps (128 registers, 8-load pipeline) measured in between. Summation order of the partials changes (fixed, batch-independent): logits move at the bf16-noise level. Local A/B (generate): 5377 -> 5340 at B=1, 5857 -> 5840 at B=8; expect headline about 5346.
- evaluator: valid: 5344 us/step at B=1 (MBU 0.847), 5850 us/step at B=8

### #23 s0023: source cleanup; lm_head prefetches layer 0 for the next launch

- status: **valid**, step_us_b1 = 5336, at 9.56 agent-hours, parent s0022, category cleanup
- hypothesis: Settled experiment switches removed from the kernel source (identical logits; A/B says +5 us from code layout), and at the end of lm_head in the step path the loaders prefetch layer 0's first q/k/v tile for the next launch (-6 us on the step path). Expect headline within 10 us of s0022 (about 5345), nothing else changes.
- evaluator: valid: 5341 us/step at B=1 (MBU 0.848), 5851 us/step at B=8

### #24 s0024: 5+ sequences: input fragments stored in the shared-memory layout, input copy becomes a block copy

- status: **valid**, step_us_b1 = 5347, at 9.87 agent-hours, parent s0023, category memory
- hypothesis: At B=8 the input copy costs about 13 us per layer: 64-196 KB per SM, and ~12 instructions of index arithmetic per 16 bytes on every thread to map the compact per-sequence layout to MMA fragments. With 5 or more sequences the producers now write the fragments to global memory in the shared-memory layout itself, so the copy is a plain block copy (3 loads per thread requested before the stores). Fewer sequences keep the compact layout (less to copy). Values and results are bitwise unchanged. Local A/B (generate): B=8 5846 -> 5683 (-160); B=1 5346 -> 5353 (+7, code layout only: the new path is not executed at B=1). Expect headline about 5345 and B=8 about 5690.
- evaluator: valid: 5351 us/step at B=1 (MBU 0.846), 5685 us/step at B=8

### #25 s0025: weights stored losslessly in 12 bits, decoded in registers before the MMA

- status: **valid**, step_us_b1 = 4522, at 11.10 agent-hours, parent s0024, category memory
- hypothesis: The step is bounded by HBM bytes, so read fewer: each bf16 weight is stored as sign + 4-bit exponent code + 7-bit mantissa (rows pre-scaled by exact powers of two; the ~1e-4 of elements outside the 16-binade window are patched exactly by extra MMAs in the coordinator), 3 loads decode to 4 MMAs. Bytes read per step 15.1 -> 11.4 GB at the same ~3.0 TB/s; no sync point added or removed (181 grid barriers). Expect step_us_b1 5340 -> ~4520 (-15%), b8 5674 -> ~4830, numerics unchanged (same weights), and MBU as defined (16-bit bytes) to rise to ~100%.
- evaluator: valid: 4527 us/step at B=1 (MBU 1.000), 4841 us/step at B=8

### #26 s0026: exact per-column power-of-two weight scales: no exception-heavy tiles

- status: **valid**, step_us_b1 = 4464, at 11.83 agent-hours, parent s0025, category memory
- hypothesis: s0025's exception path costs ~67 us, 52 of them in layers 1-3 whose MLP tiles have up to 550 out-of-window weights (coordinator runs dozens of dependent loads + MMAs per tile). Scaling column j of a matrix by 2^cs[j] and its input by 2^-cs[j] (folded on the host into the RMSNorm weights / the up-row scales, exact, no kernel change) cuts those tiles to <=44 exceptions (825k -> 550k total). Expect step_us_b1 4522 -> ~4460, b8 4833 -> ~4775; no sync point added or removed.
- evaluator: valid: 4464 us/step at B=1 (MBU 1.014), 4781 us/step at B=8

### #27 s0027: up to 4 sequences: narrow input fragments, block shared memory 223 -> 154 KB

- status: **valid**, step_us_b1 = 4366, at 12.07 agent-hours, parent s0026, category memory
- hypothesis: With 196 KB or more of shared memory per block the SM keeps almost no L1 and the 12-bit decode stream runs 4% slower (tests/bw12.cu: 3124 vs 2997 GB/s, threshold at 200704 B). For batch <= 4 the input fragments need only 16 of 32 lanes per kstep; with the sx area last in the layout the launch asks for 157.8 KB. Same arithmetic, bitwise the same logits. Expect step_us_b1 4463 -> ~4360 (-2.3%); b8 unchanged (wide layout still 223 KB, ~4780-4810). No sync point added or removed.
- evaluator: valid: 4369 us/step at B=1 (MBU 1.036), 4808 us/step at B=8

### #28 s0028: 12-bit stream laid out for a mask-and-OR decode (11 instead of 31 ops per MMA)

- status: **valid**, step_us_b1 = 4270, at 12.33 agent-hours, parent s0027, category gemm
- hypothesis: Decode is the SM's compute limit: a tile takes 2.3 us of pure compute, which bounds the o projection after attention and every catch-up after a barrier. With the exponent window at 0x70 (rows shifted by 2^3, exact) a bf16 is sign<<15 | 0111<<11 | code[10:0], so three stream words are three MMA registers after one mask+OR and the fourth is gathered from the 24 spare bits: 11 ops per MMA, 1.6 us per tile (tests/bw12.cu). Same weights, same arithmetic. Expect step_us_b1 4366 -> ~4270 (-2.2%), b8 4808 -> ~4725. No sync point added or removed.
- evaluator: valid: 4273 us/step at B=1 (MBU 1.059), 4722 us/step at B=8

### #29 s0029: loaders stash the start of the next phase's first tile while the grid barrier runs

- status: **valid**, step_us_b1 = 4198, at 13.42 agent-hours, parent s0028, category scheduling
- hypothesis: At a matvec-to-matvec transition HBM idles ~2 us (last epilogue, fence, grid barrier, input copy) and the first tile then starts cold. After a new block barrier behind the coordinator's fence, each loader warp streams up to 4 loads (6 with 5-8 sequences; 3 for the K=12288 matrix) of its first tile of the next phase into shared memory, 2 at a time, and stops when its coordinator reports the grid barrier complete (an SM's memory responses arrive in order, so a deeper stash delays the barrier poll: a full-tile stash made every block wait 3.3 us). Adds one block barrier per matvec phase, no grid sync. Also orders the loaders' read of the phase descriptor before the coordinator rewrites it. Expect step_us_b1 4270 -> ~4200 (-1.7%), b8 4714 -> ~4685.
- evaluator: valid: 4202 us/step at B=1 (MBU 1.077), 4690 us/step at B=8

### #30 s0030: 3+ sequences: loaders read the input fragments from L2, no input copy

- status: **valid**, step_us_b1 = 4184, at 14.19 agent-hours, parent s0029, category memory
- hypothesis: With 5-8 sequences the block copies 64-196 KB of input fragments into shared memory at every matvec phase (1.1-2.5 us x 4 per layer) and its 223 KB of shared memory leave the SM without L1 (stream 4% slower). Instead each loader warp reads the B operands of a group with two 512 B loads from global memory (they sit in L2) next to its three weight loads; shared memory is 158 KB for every batch size. tests/bw12.cu mode 2 says this costs 1% of stream rate. Used from 3 sequences up (measured crossover). Same arithmetic, bitwise the same logits. Expect b8 4680 -> ~4380 (-6%), b1 4198 -> ~4180 (only codegen: opaque thread index, no spills). No sync point added or removed.
- evaluator: valid: 4187 us/step at B=1 (MBU 1.081), 4379 us/step at B=8

### #31 s0031: attention to 9216+ positions: 16 B/position scratch, split over all SMs for 1-2 sequences

- status: **valid**, step_us_b1 = 4186, at 18.71 agent-hours, parent s0030, category attention
- hypothesis: track B. Formats that differ from the checkpoint: weights as a lossless 12-bit code of the BF16 values with exact power-of-two row/column scales; residual stream and accumulators float32 (K/V cache BF16 as the general engines). Change: (1) attention scratch 16 B per position (probabilities converted to bf16 in place), so the context limit goes from 3904 to 12288; V loads 4 at a time beyond 1136 positions. (2) With 1-2 sequences beyond 1136 positions the 8B attention blocks are SM-bound (76 us per layer at 9216), so attention is split: scores by 16 blocks per (sequence, kv head) in position chunks, then softmax + V by one block per dim tile; fixed summation order, bitwise equal to the one-block path. Adds one grid barrier per layer only in that mode (217 per step). Expect the evaluator's numbers unchanged (b1 ~4185, b8 ~4370); at 9000 positions B=1 6434 -> ~5070 us per step, B=8 ~8300.
- evaluator: valid: 4192 us/step at B=1 (MBU 1.080), 4378 us/step at B=8

### #32 s0032: track A: BF16 weights as stored, BF16 activations, RMSNorm before the matvec

- status: **valid**, step_us_b1 = 5386, at 19.20 agent-hours, parent s0031, category structure
- hypothesis: track A. Formats that differ from the checkpoint's: none (weights BF16 as stored, only permuted into the kernel's read order; activations rounded to BF16 wherever the Hugging Face BF16 model has a BF16 tensor; K/V cache BF16; float32 only inside RMSNorm, softmax and the matvec accumulators, as there). Data flow as the model: RMSNorm is applied to the matvec input (in the input copy with 1-2 sequences, as a phase of its own with a grid barrier with 3+: 254 grid barriers per step instead of 181), nothing fused or precomputed. Same kernel organisation as track B (s0031) otherwise: 4 loads per 4 ksteps instead of 3, no exception path. Expect step_us_b1 ~5380 and b8 ~5630 (bytes bound at 3.12 TB/s: 4850), numeric error near HF BF16's own (median ~1.2x, limit 2x), bitwise batch invariance.
- evaluator: valid: 5388 us/step at B=1 (MBU 0.840), 5639 us/step at B=8

### #33 s0033: attention: prefetch the K/V append lines; idle blocks stash after the K loads (track B)

- status: **valid**, step_us_b1 = 4161, at 20.36 agent-hours, parent s0031, category attention
- hypothesis: track B. Formats that differ from the checkpoint: weights as a lossless 12-bit code of the BF16 values with exact power-of-two row/column scales; residual stream and accumulators float32; RMSNorm fused into the matvec. Change (scheduling inside the attention phase, no arithmetic touched): (1) the block that appends the new K/V rows prefetches the position tile during the q/k/v phase (the fence behind stores into lines last touched a step ago waited ~1 us for HBM); (2) with 1-2 sequences the idle blocks start their o-tile stash only when all attention blocks have stamped their score pass done (a hint array, bounded polling): the stash flood delayed the attention blocks' K loads by 1.8 us; (3) K loads of a position tile 8 at a time. No grid sync added. Expect step_us_b1 4186 -> ~4160 (-0.6%), b8 unchanged (~4370).
- evaluator: valid: 4168 us/step at B=1 (MBU 1.086), 4377 us/step at B=8

### #34 s0034: track A with the attention scheduling of s0033

- status: **valid**, step_us_b1 = 5343, at 20.39 agent-hours, parent s0032, category attention
- hypothesis: track A. Formats that differ from the checkpoint's: none (BF16 weights as stored, BF16 activations wherever the Hugging Face BF16 model has a BF16 tensor, K/V cache BF16; RMSNorm applied before the matvec). Same source as s0033 built with TRACK=A: the append-line prefetch and the stash start signalled by the attention blocks (1-2 sequences). No arithmetic touched, logits bitwise those of s0032. Expect step_us_b1 5386 -> ~5340 (-0.9%), b8 ~5620.
- evaluator: valid: 5346 us/step at B=1 (MBU 0.847), 5640 us/step at B=8

### #35 s0035: grid-barrier increment directly behind the fence (track B)

- status: **valid**, step_us_b1 = 4154, at 21.50 agent-hours, parent s0033, category sync
- hypothesis: track B. Formats that differ from the checkpoint: weights as a lossless 12-bit code of the BF16 values with exact power-of-two row/column scales; residual stream and accumulators float32; RMSNorm fused into the matvec. Change: at the end of a matvec phase the coordinator increments the grid barrier's counter right after the fence of its last epilogue, before the block barrier that releases its loaders and before it writes the next phase's descriptor (both concern only its own block); the grid waits for the last block's increment, which now comes ~0.2 us earlier. Same number of sync points. Expect step_us_b1 4161 -> ~4150, b8 4366 -> ~4340 (-0.6%).
- evaluator: valid: 4160 us/step at B=1 (MBU 1.088), 4354 us/step at B=8

### #36 s0036: track A with the early grid-barrier increment of s0035

- status: **valid**, step_us_b1 = 5332, at 21.55 agent-hours, parent s0034, category sync
- hypothesis: track A. Formats that differ from the checkpoint's: none (BF16 weights as stored, BF16 activations wherever the Hugging Face BF16 model has a BF16 tensor, K/V cache BF16; RMSNorm applied before the matvec). Same source as s0035 built with TRACK=A: the grid-barrier increment right behind the fence; in the normalising input copy the loads are requested before the 1/rms barrier. No arithmetic touched. Expect step_us_b1 5343 -> ~5330, b8 ~5615.
- evaluator: valid: 5335 us/step at B=1 (MBU 0.849), 5628 us/step at B=8

### #37 s0037: attention: the idle blocks' stash starts one load at a time (track B)

- status: **valid**, step_us_b1 = 4105, at 22.42 agent-hours, parent s0035, category attention
- hypothesis: track B. Formats that differ from the checkpoint: weights as a lossless 12-bit code of the BF16 values with exact power-of-two row/column scales; residual stream and accumulators float32; RMSNorm fused into the matvec. Change: with 1-2 sequences the 124 blocks that stash their first o tile during attention request their first 5 loads one at a time instead of two at a time. While HBM is saturated by that stash, the attention blocks' K and V loads (L2 hits) take ~1.8 us longer; half the traffic in flight at the start leaves them room. Scheduling only, logits bitwise unchanged, no sync point added. Expect step_us_b1 4154 -> ~4100 (-1.3%), b8 unchanged (~4340).
- evaluator: valid: 4112 us/step at B=1 (MBU 1.101), 4355 us/step at B=8

### #38 s0038: track A with the gentle attention-phase stash of s0037

- status: **valid**, step_us_b1 = 5299, at 22.49 agent-hours, parent s0036, category attention
- hypothesis: track A. Formats that differ from the checkpoint's: none (BF16 weights as stored, BF16 activations wherever the Hugging Face BF16 model has a BF16 tensor, K/V cache BF16; RMSNorm applied before the matvec). Same source as s0037 built with TRACK=A: with 1-2 sequences the idle blocks' o stash requests its first 5 loads one at a time, so that the attention blocks' K/V loads are not delayed by a saturated HBM. Scheduling only. Expect step_us_b1 5332 -> ~5290 (-0.8%), b8 ~5610.
- evaluator: valid: 5301 us/step at B=1 (MBU 0.854), 5635 us/step at B=8

### #39 s0039: o projection: attention blocks take two tiles; second tile prefetched at phase start (track B)

- status: **valid**, step_us_b1 = 4088, at 23.05 agent-hours, parent s0037, category scheduling
- hypothesis: track B. Formats that differ from the checkpoint: weights as a lossless 12-bit code of the BF16 values with exact power-of-two row/column scales; residual stream and accumulators float32; RMSNorm fused into the matvec. Change: (1) o tile table: the blocks that ran attention take two tiles instead of one, so the others are left with at most two (one of them stashed) instead of up to 2.8: with 8 sequences the 68 other blocks ran two cold tiles in a row while 64 SMs waited (o phase 10.7 us for 8.1 us of bytes). (2) At the start of the o phase a block asks L2 for the rest of its second tile while it multiplies the stashed one. Table and scheduling only, logits bitwise unchanged. Expect b8 4343 -> ~4300 (-1%), b1 4105 -> ~4090.
- evaluator: valid: 4095 us/step at B=1 (MBU 1.106), 4322 us/step at B=8

### #40 s0040: track A with the o tile table of s0039 (attention blocks take two tiles)

- status: **valid**, step_us_b1 = 5298, at 23.09 agent-hours, parent s0038, category scheduling
- hypothesis: track A. Formats that differ from the checkpoint's: none (BF16 weights as stored, BF16 activations wherever the Hugging Face BF16 model has a BF16 tensor, K/V cache BF16; RMSNorm applied before the matvec). Same source as s0039 built with TRACK=A: the blocks that ran attention take two o tiles instead of one, so that no block runs two cold o tiles in a row while others wait. (The L2 prefetch of the second tile at the start of the o phase exists only in the 12-bit path.) Table only, logits bitwise unchanged. Expect b8 5615 -> ~5580 (-0.6%), b1 ~5293.
- evaluator: valid: 5301 us/step at B=1 (MBU 0.854), 5597 us/step at B=8

### #41 s0041: 6+ sequences: stash blocks prefetch all of their second o tile (track B)

- status: **valid**, step_us_b1 = 4093, at 23.68 agent-hours, parent s0039, category attention
- hypothesis: track B. Formats that differ from the checkpoint: weights as a lossless 12-bit code of the BF16 values with exact power-of-two row/column scales; residual stream and accumulators float32; RMSNorm fused into the matvec. Change: with 6 or more sequences the blocks that stash their first o tile during attention ask L2 for all 9 loads per warp of their second tile instead of 4 (with fewer sequences 4 stays: the extra traffic delays the attention blocks more than it helps). Prefetch depth only. Expect b8 generate 4309 -> ~4285 (-0.5%), b1 unchanged (~4090).
- evaluator: valid: 4099 us/step at B=1 (MBU 1.104), 4307 us/step at B=8

### #42 s0042: track A with the second-tile prefetch rule of s0041

- status: **valid**, step_us_b1 = 5304, at 23.71 agent-hours, parent s0040, category attention
- hypothesis: track A. Formats that differ from the checkpoint's: none (BF16 weights as stored, BF16 activations wherever the Hugging Face BF16 model has a BF16 tensor, K/V cache BF16; RMSNorm applied before the matvec). Same source as s0041 built with TRACK=A (6+ sequences: the stash blocks prefetch all of their second o tile). Expect b8 ~5575, b1 ~5300, logits bitwise unchanged.
- evaluator: valid: 5306 us/step at B=1 (MBU 0.853), 5593 us/step at B=8

### #43 s0043: softmax in two passes less at long context; split path copies scores in one round trip (track B)

- status: **valid**, step_us_b1 = 4085, at 25.67 agent-hours, parent s0041, category attention
- hypothesis: track B. Formats that differ from the checkpoint: weights as a lossless 12-bit code of the BF16 values with exact power-of-two row/column scales; residual stream and accumulators float32; RMSNorm fused into the matvec. Change (attention only): the maximum per head is taken during the score pass (exact), and from 1136 positions on the bf16 conversion of the probabilities is done by the V pass while it waits for memory (p = bf16(e * (1/Z)), one definition for both attention paths) instead of a pass of its own; the split path's score copy requests all loads before the first store. HBM idles during the softmax, so this is pure gain at long context: expect the prompt-8192 workload at B=1 4995 -> ~4850 us per token and at B=8 8107 -> ~7930; evaluator numbers (context 64-320) unchanged: b1 ~4090, b8 ~4300.
- evaluator: valid: 4097 us/step at B=1 (MBU 1.105), 4306 us/step at B=8

### #44 s0044: track A with the long-context softmax changes of s0043

- status: **valid**, step_us_b1 = 5286, at 25.70 agent-hours, parent s0042, category attention
- hypothesis: track A. Formats that differ from the checkpoint's: none (BF16 weights as stored, BF16 activations wherever the Hugging Face BF16 model has a BF16 tensor, K/V cache BF16; RMSNorm applied before the matvec). Same source as s0043 built with TRACK=A: maximum per head taken during the score pass, bf16 probabilities built by the V pass from 1136 positions on, split-path score copy in one round trip. Expect evaluator numbers unchanged (b1 ~5290, b8 ~5570) and the prompt-8192 workload at B=1 6154 -> ~6020 us per token.
- evaluator: valid: 5290 us/step at B=1 (MBU 0.856), 5584 us/step at B=8

### #45 s0045: long context, 3+ sequences: idle blocks pull the first K tiles into L2 (track B)

- status: **valid**, step_us_b1 = 4095, at 27.26 agent-hours, parent s0043, category attention
- hypothesis: track B. Formats that differ from the checkpoint: weights as a lossless 12-bit code of the BF16 values with exact power-of-two row/column scales; residual stream and accumulators float32; RMSNorm fused into the matvec. Change: when the K/V history no longer fits in L2 (context x sequences beyond 4096 positions) the idle blocks used to touch nothing at the start of attention; now they pull the first 128/B position tiles of K of every (sequence, kv head) into L2, which the attention blocks read first, so HBM works while those prepare q. Scheduling only. Not visible to the evaluator (context 64-320): expect b1 ~4085, b8 ~4290 unchanged; prompt 8192 + 256 tokens at B=8 measured 8010 -> 7954 us per token (-0.7%).
- evaluator: valid: 4103 us/step at B=1 (MBU 1.103), 4301 us/step at B=8

### #46 s0046: track A with the long-context K touch of s0045

- status: **valid**, step_us_b1 = 5288, at 27.29 agent-hours, parent s0044, category attention
- hypothesis: track A. Formats that differ from the checkpoint's: none (BF16 weights as stored, BF16 activations wherever the Hugging Face BF16 model has a BF16 tensor, K/V cache BF16; RMSNorm applied before the matvec). Same source as s0045 built with TRACK=A: at long context with 3+ sequences the idle blocks pull the first K tiles into L2 at the start of attention. Not visible to the evaluator: expect b1 ~5280, b8 ~5565 unchanged; B=8 at 8192 positions about -0.6%.
- evaluator: valid: 5292 us/step at B=1 (MBU 0.855), 5580 us/step at B=8

### #47 s0047: long context, 1 sequence: idle blocks pull V into L2 during score copy + softmax

- status: **valid**, step_us_b1 = 4086, at 29.11 agent-hours, parent s0043, category attention
- hypothesis: track B; formats that differ from the checkpoint: 12-bit lossless weight code with exact power-of-two row/column scales, float32 residual stream. In the split attention path (1-2 sequences, 1152+ positions) the 64 V blocks copy scores and run the softmax for ~7 us while HBM carries only the o stash; the 68 idle blocks now load the units' V history (8 loads per warp at a time) so the V pass runs on L2 hits. Results are bitwise unchanged. Expect the evaluator numbers (context 64-320, one-block path) unchanged within the build spread (4085-4095 B=1, ~4290-4310 B=8); prompt-8192 workload at B=1 4881 -> ~4810 us per token (-1.5%), mid contexts 1152-1500 -55 us. No sync point added or removed.
- evaluator: valid: 4091 us/step at B=1 (MBU 1.107), 4307 us/step at B=8

### #48 s0048: track A with the split-path V touch of s0047

- status: **valid**, step_us_b1 = 5292, at 29.15 agent-hours, parent s0044, category attention
- hypothesis: track A; formats that differ from the checkpoint: none (BF16 weights as stored, BF16 activations and KV cache, RMSNorm computed per step before each matvec). Same change as s0047: in the split attention path (1-2 sequences, 1152+ positions) the idle blocks load the V history into L2 while the V blocks copy scores and run the softmax. Bitwise unchanged results. Expect the evaluator numbers unchanged (5286-5288 B=1, ~5565 B=8); prompt-8192 workload at B=1 6061 -> ~5970 us per token. No sync point added or removed.
- evaluator: valid: 5294 us/step at B=1 (MBU 0.855), 5583 us/step at B=8

### #49 s0049: track A, 3+ sequences: loaders apply RMSNorm to the input; no norm phase

- status: **valid**, step_us_b1 = 5278, at 30.12 agent-hours, parent s0048, category sync
- hypothesis: track A; formats that differ from the checkpoint: none (BF16 weights as stored, BF16 activations and KV cache, RMSNorm computed per step before each matvec). With 3+ sequences the RMSNorm before q/k/v, gate/up and lm_head ran as a grid phase of its own (73 extra grid barriers per step). Now each block computes 1/rms per sequence and copies the norm weights to shared memory at the phase start, and the loader warps compute bf16(w * bf16(x / rms)) on the fragments they read from L2, right before the MMA (same operators, same order, same values: bitwise equal to s0048). Expect B=8 5570 -> ~5450 us (-2%), B=1 unchanged (~5290). Removes 73 grid barriers per step at B>=3, no host sync change.
- evaluator: valid: 5282 us/step at B=1 (MBU 0.857), 5460 us/step at B=8

### #50 s0050: track A: normalised input kept in shared memory after a block's first tile

- status: **valid**, step_us_b1 = 5284, at 30.39 agent-hours, parent s0049, category memory
- hypothesis: track A; formats that differ from the checkpoint: none (BF16 weights as stored, BF16 activations and KV cache, RMSNorm computed per step before each matvec). Since s0049 the loaders normalise the input fragments of every tile again (12 gate/up tiles and ~72 lm_head tiles per block with 8 sequences). Now a warp normalises its groups' fragments once, in the block's first tile, and keeps them in its own (already consumed) part of the stash image for the other tiles; same values, bitwise equal. Expect B=8 5448 -> ~5395 us (-1%), B=1 unchanged.
- evaluator: valid: 5288 us/step at B=1 (MBU 0.856), 5403 us/step at B=8

### #51 s0051: long context, 3+ sequences: 6 V loads in flight per warp instead of 4

- status: **valid**, step_us_b1 = 4094, at 30.91 agent-hours, parent s0047, category attention
- hypothesis: track B; formats that differ from the checkpoint: 12-bit lossless weight code with exact power-of-two row/column scales, float32 residual stream. With 8 sequences at 8192+ positions the 64 attention blocks read V at ~83% of the HBM rate with 4 loads of 512 B in flight per warp; 6 put 50% more in flight. Only the one-block path from 1136 positions with 3+ sequences changes (bitwise same results). Expect the evaluator unchanged (~4085 / ~4290); prompt-8192 workload at B=8 ~8140 -> ~8100 us per token (-0.5..-1%).
- evaluator: valid: 4101 us/step at B=1 (MBU 1.104), 4303 us/step at B=8

### #52 s0052: attention: no fence behind the K/V append

- status: **valid**, step_us_b1 = 4083, at 31.38 agent-hours, parent s0051, category sync
- hypothesis: track B; formats that differ from the checkpoint: 12-bit lossless weight code with exact power-of-two row/column scales, float32 residual stream. The attention block that appends this step's K/V row fenced it before its block barrier, but only the block itself reads the row in this phase (the block barrier makes the block's global stores visible to its threads), and other blocks read it only behind the grid barrier, before which the block fences anyway. Dropping the fence shortens the attention prep by one store round trip. Results bitwise unchanged; same-machine A/B -10 us at B=1 and B=8. Expect ~4075 / ~4280 us (-0.3%).
- evaluator: valid: 4088 us/step at B=1 (MBU 1.107), 4296 us/step at B=8

### #53 s0053: track A with the fence removal of s0052 and the V batching of s0051

- status: **valid**, step_us_b1 = 5230, at 31.41 agent-hours, parent s0050, category sync
- hypothesis: track A; formats that differ from the checkpoint: none (BF16 weights as stored, BF16 activations and KV cache, RMSNorm computed per step before each matvec). Same two changes as s0051 (6 V loads in flight in the one-block path at 1136+ positions with 3+ sequences: long context only) and s0052 (no fence behind the K/V append). Results bitwise unchanged. Expect B=1 5284 -> ~5270, B=8 ~5390 (the suite measured 5230 / 5385 at B=1 / B=8 on this machine, so maybe more).
- evaluator: valid: 5235 us/step at B=1 (MBU 0.865), 5429 us/step at B=8

### #54 s0054: 1-2 sequences: loader warps fetch their own input slice; no block copy/barrier

- status: **valid**, step_us_b1 = 4061, at 33.49 agent-hours, parent s0052, category sync
- hypothesis: track B; formats that differ from the checkpoint: 12-bit lossless weight code with exact power-of-two row/column scales, float32 residual stream. With 1-2 sequences every phase started with a block-wide copy of the input fragments into shared memory and a block barrier (down projection: 0.77 us, others ~0.17 us per phase at B=1). Now each loader warp requests only the slice it multiplies, issues its first weight loads, then stores the slice (no block barrier); the coordinator's exception MMAs read the compact global copy. Bitwise equal; same-machine A/B B=1 4082 -> 4062, B=2 4157 -> 4121, B=8 unchanged. Expect ~4065 / ~4280 us.
- evaluator: valid: 4066 us/step at B=1 (MBU 1.113), 4310 us/step at B=8

### #55 s0055: track A with the per-warp input slices of s0054 (phases without a norm)

- status: **valid**, step_us_b1 = 5233, at 33.62 agent-hours, parent s0053, category sync
- hypothesis: track A; formats that differ from the checkpoint: none (BF16 weights as stored, BF16 activations and KV cache, RMSNorm computed per step before each matvec). With 1-2 sequences the o and down projections no longer start with a block-wide input copy and block barrier: each loader warp fetches the slice it multiplies (norm phases keep the block-wide normalising copy). Bitwise equal to s0053; same-machine A/B B=1 5240 -> 5229, B=2 5344 -> 5317. Expect ~5220 / ~5410 us.
- evaluator: valid: 5237 us/step at B=1 (MBU 0.864), 5416 us/step at B=8

### #56 s0056: 3+ sequences: no block barrier at the start of each phase

- status: **valid**, step_us_b1 = 4057, at 33.93 agent-hours, parent s0054, category sync
- hypothesis: track B; formats that differ from the checkpoint: 12-bit lossless weight code with exact power-of-two row/column scales, float32 residual stream. With 3+ sequences the loaders read their input from L2, so the block barrier that ended every phase's entry only waited for the 1/rms of norm phases (loader warps 12-19), which the coordinator reads behind the first tile's barrier anyway. Removing it: same-machine A/B B=8 4286/4300 -> 4247/4249 us per step, B=1-2 unchanged, bitwise equal. Expect B=8 ~4255 (-1%), B=1 ~4060.
- evaluator: valid: 4062 us/step at B=1 (MBU 1.115), 4257 us/step at B=8

### #57 s0057: q/k/v, gate+up, down on 128 of the 132 SMs: every block gets the same tile count

- status: **valid**, step_us_b1 = 4043, at 34.43 agent-hours, parent s0056, category scheduling
- hypothesis: track B; formats that differ from the checkpoint: 12-bit lossless weight code with exact power-of-two row/column scales, float32 residual stream. 384 / 1536 / 256 tiles over 132 blocks leave some blocks with one tile more than others (gate+up: 84 blocks x 12, 48 x 11), so the tail of each phase runs on fewer SMs than needed to saturate HBM. On 128 blocks every block has exactly 3 / 12 / 2 tiles and all finish together; the 4 other SMs idle in these phases. Bitwise equal (a tile's arithmetic does not depend on its block). Same-machine A/B: B=1 4059 -> 4045, B=2 4122 -> 4101, B=8 4248 -> 4222. Expect ~4045 / ~4225.
- evaluator: valid: 4048 us/step at B=1 (MBU 1.118), 4224 us/step at B=8

### #58 s0058: track A: no entry barrier with 3+ sequences, matvec tiles on 128 SMs with 3+

- status: **valid**, step_us_b1 = 5235, at 34.58 agent-hours, parent s0055, category scheduling
- hypothesis: track A; formats that differ from the checkpoint: none (BF16 weights as stored, BF16 activations and KV cache, RMSNorm computed per step before each matvec). Two scheduling changes of track B (s0056, s0057) applied where they measured better in track A: with 3+ sequences the o / down phases start without a block barrier, and q/k/v, gate+up, down run on 128 blocks (even tile counts); with 1-2 sequences all 132 blocks as before (128 measured +20 us there). Bitwise equal. Same-machine: B=1 5230 -> 5230, B=8 5425 -> 5402. Expect ~5230 / ~5390 us.
- evaluator: valid: 5241 us/step at B=1 (MBU 0.864), 5413 us/step at B=8

### #59 s0059: only the attention blocks wait at the q/k/v barrier; the others go on

- status: **valid**, step_us_b1 = 4047, at 35.68 agent-hours, parent s0057, category sync
- hypothesis: track B; formats that differ from the checkpoint: 12-bit lossless weight code with exact power-of-two row/column scales, float32 residual stream. In the one-block attention path the blocks without attention need nothing q/k/v produces before the end of attention, so they count in at the q/k/v barrier without waiting and start their o stash at once; before counting in for the attention barrier they check that the q/k/v barrier is complete (else their increment could complete it early for the attention blocks). Bitwise equal; same-machine A/B B=1 -12..-19, B=8 -11..-27 us. Expect ~4030 / ~4200 us.
- evaluator: valid: 4052 us/step at B=1 (MBU 1.117), 4208 us/step at B=8

### #60 s0060: track A: only attention blocks wait at the q/k/v barrier (3+ sequences)

- status: **valid**, step_us_b1 = 5234, at 35.87 agent-hours, parent s0058, category sync
- hypothesis: track A; formats that differ from the checkpoint: none (BF16 weights as stored, BF16 activations and KV cache, RMSNorm computed per step before each matvec). Same change as s0059, used in track A only with 3+ sequences (it measured +30 us with one): the blocks without attention count in at the q/k/v barrier and go on to their o stash; they check that barrier is complete before counting in at the end of attention. Bitwise equal; same-machine B=1 5230 -> 5232, B=8 5406 -> 5374. Expect ~5233 / ~5370 us.
- evaluator: valid: 5239 us/step at B=1 (MBU 0.864), 5390 us/step at B=8

### #61 s0061: re-evaluation of the final track B source (s0059 unchanged)

- status: **valid**, step_us_b1 = 4026, at 37.85 agent-hours, parent s0059, category other
- hypothesis: track B; formats that differ from the checkpoint: 12-bit lossless weight code with exact power-of-two row/column scales, float32 residual stream. No change: the source of s0059 (journal commits only since). A second evaluator sample of the final engine, to show the run-to-run spread of the headline. Expect 4035-4055 us at B=1 and 4190-4210 at B=8.
- evaluator: valid: 4030 us/step at B=1 (MBU 1.123), 4205 us/step at B=8

### #62 s0062: re-evaluation of the final track A source (s0060 unchanged)

- status: **valid**, step_us_b1 = 5237, at 37.88 agent-hours, parent s0060, category other
- hypothesis: track A; formats that differ from the checkpoint: none (BF16 weights as stored, BF16 activations and KV cache, RMSNorm computed per step before each matvec). No change: the source of s0060. A second evaluator sample of the final track-A engine on the current machine. Expect 5220-5245 us at B=1 and 5365-5390 at B=8.
- evaluator: valid: 5243 us/step at B=1 (MBU 0.863), 5385 us/step at B=8

### #63 s0063: attention blocks: L2 prefetch of 3 (not 6) loads per warp of their cold o tile

- status: **valid**, step_us_b1 = 4020, at 38.22 agent-hours, parent s0061, category memory
- hypothesis: track B; formats that differ from the checkpoint: 12-bit lossless weight code with exact power-of-two row/column scales, float32 residual stream. After attention each attention block asks L2 for the start of its first (cold) o tile; with fewer requests in flight its own first loads and the barrier polls queue behind less. Prefetch only, results bitwise unchanged. Same-machine 5 rounds: B=1 4048 -> 4043 (mean), B=8 4204 -> 4195. Expect -5..-10 us at both batch sizes, i.e. within the spread between machines (~20 us).
- evaluator: valid: 4025 us/step at B=1 (MBU 1.125), 4192 us/step at B=8

### #64 s0064: track A with the o-tile prefetch of s0063 (3 loads per warp)

- status: **valid**, step_us_b1 = 5235, at 38.36 agent-hours, parent s0062, category memory
- hypothesis: track A; formats that differ from the checkpoint: none (BF16 weights as stored, BF16 activations and KV cache, RMSNorm computed per step before each matvec). Same change as s0063: the attention blocks ask L2 for 3 instead of 6 loads per warp of their cold o tile. Prefetch only, bitwise unchanged. Same-machine 4 rounds: B=1 5233 -> 5232, B=8 5387 -> 5372. Expect ~5235 / ~5360 us.
- evaluator: valid: 5242 us/step at B=1 (MBU 0.864), 5383 us/step at B=8

### #65 s0065: final track B source (s0063 kernel; host-side diagnostic export only)

- status: **valid**, step_us_b1 = 4047, at 39.33 agent-hours, parent s0063, category other
- hypothesis: track B; formats that differ from the checkpoint: 12-bit lossless weight code with exact power-of-two row/column scales, float32 residual stream. The committed final source: s0063's kernel unchanged, plus mk_stat 4-6 returning buffer addresses (host code, not on the step path). Expect s0063's numbers within the machine spread: ~4020-4045 us at B=1, ~4180-4200 at B=8.
- evaluator: valid: 4051 us/step at B=1 (MBU 1.117), 4186 us/step at B=8

### #66 s0066: final track A source (s0064 kernel; host-side diagnostic export only)

- status: **valid**, step_us_b1 = 5236, at 39.36 agent-hours, parent s0064, category other
- hypothesis: track A; formats that differ from the checkpoint: none (BF16 weights as stored, BF16 activations and KV cache, RMSNorm computed per step before each matvec). The committed final source built with engine/TRACK=A: s0064's kernel unchanged plus a host-side diagnostic export. Expect s0064's numbers within the machine spread: ~5230-5245 / ~5365-5380 us.
- evaluator: valid: 5242 us/step at B=1 (MBU 0.864), 5381 us/step at B=8

