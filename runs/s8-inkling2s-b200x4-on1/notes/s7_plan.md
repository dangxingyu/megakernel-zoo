# Engine as built (run s7-inkling2s-b200x4-fa1, track A) and what is still open

engine/ is the source of s0020: 3307 us per token at B=1 and 5378 us per step at B=8 over HTTP (1.424x / 1.228x
SGLang's 4711 / 6604), track A, measured on the last pod of the run; on the previous pod the same B=1 kernel gave
3296-3300 us (s0013, the best headline figure, and s0016-s0018). Against s0013 the engine adds the tensor-core path
for the NVFP4 w13 GEMV at batch >= 4: in-process B=8 step 5522 -> 5381 us (-2.6%), +46 GB per card.
`MK_CFLAGS=-DNO_TC13` builds the engine of s0013 (one layout, one kernel).

## Structure
- One persistent kernel per card per step (152 blocks x 768 threads), the four cards started by the host and
  synchronised only through peer memory. Two kernels: `ink_step_std` (the engine of s0013, batches 1-3) and
  `ink_step_tc` (the same step with the NVFP4 w13 GEMV on the tensor-core path, batches >= 4). `ink_step_red` (B=1
  redundant-block kernel of run s6) is still compiled but never launched.
- Split over the cards as in s6: attention TP4 by heads (12 q heads / 2 KV heads per card), every expert TP4 by its
  intermediate dimension (576 of 2304 per card), router / residual mixes / short convs / norms replicated.
- Per layer, in order (us per layer in process at B=1 / B=8 with ~38-42 distinct experts, fixed tokens):
  1. attention input (10.5 / 12): "team" phase, see below; ends with a grid barrier.
  2. q/r/sink/k/v GEMV (6.3 / 6): BF16 MMAs, B fragment straight from global memory. The output-gate rows (40% of
     the old qkv GEMV) run beside the attention phase in idle blocks when at least 24 blocks are free.
  3. attention (10 / 10): one block per (token, KV head, chunk of 32 / 64 / 128 positions): K/V rows by cp.async
     into shared memory, scores and relative bias as BF16 MMAs, softmax one (head, position) per thread, V sum
     scalar FP32; the block of the last chunk combines the chunks (tags from the others); grid barrier.
  4. o_proj GEMV (3.0 / 3.2) -> (value, tag) words to all four cards.
  5. post-attention (7.8 / 8.6 incl. the wait for the other cards): team phase; grid barrier.
  6. router GEMV (3.0 / 3.2): 24 blocks; sigmoids published as tagged words.
  7. routing (3.7 / 4.2): every block computes top-8, union, routed-token masks itself (no barrier).
  8. w13 GEMV: fragment path (m16n8k16 MMAs after cvt + HMUL2; <= 16 experts: whole strips per block, 4 at a time;
     more: flat schedule over all warps with tagged strip hand-off): 7.8 us at B=1, 0.69 us per distinct expert at
     B=8 (NVFP4), 1.07 (MXFP8). Tensor-core path (NVFP4, batch >= 4): 0.59 us per distinct expert.
     The epilogue stores (gate, up) words of the routed (expert, token) pairs.
  9. w2 input quantization (4.3 / 6.7): one routed pair per block (activation function, this card's max, cross-card
     max for NVFP4, four-over-six / MXFP8 encode with hardware conversions); other blocks zero the unrouted columns;
     grid barrier.
  10. w2 GEMV (4.8 / 0.39 us per distinct expert NVFP4, 0.57 MXFP8): whole strips per block -> tagged words to all cards.
- Team phases (attention input, post-attention, final): a token's 4608 elements one per thread on 6 blocks; every
  reduction over the vector = warp shuffles, shared memory, then the 6 blocks' partials as tagged words in global
  memory, combined in block order (all blocks and all cards get bitwise the same totals).
- Unembedding: flat BF16 GEMV over all warps, ~70 us per step (463 MB per card).

## tcgen05 (findings.md F9, journal 5.8-11 h)
- `gemv_tc`: units of 128 rows x 64 k (4096 B of E2M1 codes + 512 B of UE4M3 block scales, a second layout of the
  same checkpoint bytes, +46 GB per card) streamed by cp.async.bulk (L2 evict_first) into a ring in shared memory,
  one scale copy + one block-scaled MMA (kind::mxf4nvf4) per unit, four streams per block (2 ring slots of 2 units
  each: the attention's 73.7 KB staging area), sub-items of <= 12 units read out by warps 0-3, row groups that span
  blocks finished through tagged partials. The w13 input goes out of the post-attention phase
  as E2M1 codes + E4M3 scale bytes; its scales are stored into tensor memory beside the routing phase by three groups
  of four idle warps (s0019: -0.76% of the B=8 step against one group).
- State: s0016-s0020 valid. s0014 deadlocked in the timing stage: a race of the read-out warps on the barrier phase counts
  with few distinct experts, reproduced with tools/t_soak.py --distinct 1|2|3 and fixed in s0015 by one barrier.
  s0015 gained nothing because the calls to the tensor-core functions sat in the kernel body and slowed the MXFP8
  layers' loops; s0016 makes them from inside the routing phase function.
- Break-even (replay, fragment / tensor-core kernel, us per step): B=2 3593 / 3650, B=3 3923 / 3900, B=4 4377 / 4327,
  B=6 4932 / 4866, B=8 5467 / 5393. TC_MINB = 4.
- Measured and not adopted (source in prev/exp_tcmx/, journal 9.0-9.5 h): w2 on the same path (every row group is
  split over ~4 blocks and recombined: not faster), the MXFP8 w13 (`kind::mxf8f6f4`: correct, 8% per expert, cancelled
  by 18 KB more shared memory), one 16-byte store per warp for the codes (nothing).

## Rules that turned out to matter (details and measurements in findings.md)
- A phase costs (instructions on the longest thread path) x 5-10 ns: spread elements over threads, not threads over SMs.
- Shared memory is L1: 222 KB -> 121 KB of shared memory was worth 6-7% of the step; 141 KB (with the tensor-core
  operand area) costs nothing measurable, 18 KB more cost the tensor-core kernel 1.3%: the knee is right there.
- The kernel sits at the register cap (REG:80 at 768 threads): anything that adds live registers in a hot loop shows
  up as STACK > 96 in `cuobjdump -res-usage` and slows every phase; a call to an out-of-line function from the kernel
  body does the same to the loops around it. New paths go into their own out-of-line functions, called from inside
  an existing out-of-line phase, or into their own kernel.
- A fence is ~0.5 us; cross-block data without fences needs a tag in every word and relaxed loads on the reader.
- A/B: t_perf / rehearse on fixed tokens when the arithmetic is bitwise the same. Two builds with different rounding
  generate different text and see different expert unions: compare them with tools/t_replay.py (one generated text
  replayed teacher-forced through both), never on the rehearsal's generated-text figures (s0015).

## Checks used before every submission
tools/t_gate.py (float64 reference, 2 sequences x 12 positions; MK_TC_MINB=1 forces the tensor-core kernel),
tools/t_gate8.py (the reference rows inside a batch of 8 with a large union), tools/t_perm.py and tools/t_dup.py (slot
independence, bitwise), tools/t_det.py with DET_T=120..1100 (two runs bitwise equal at B=1, 5, 8), tools/t_same.py
against the previous build when the arithmetic is unchanged, tools/t_tc_layout.py (second layout against the
checkpoint, all layers), tools/t_soak.py with --distinct 1, 2, 3 and without (>= 20 000 steps each: rare deadlocks,
few and many distinct experts) and tools/t_timeseq.py (the evaluator's timing-stage call pattern), tools/t_replay.py
(the A/B of two kernels on one generated text), tools/rehearse.py --online (the evaluator's loops and the HTTP path).

## Open (not done, with the measurement that motivates each)
- Tensor-core path at small batches: fixed costs (set-up 0.9 us, first data ~2.5 us after entry because the rings
  of all SMs fill at the memory rate, hand-off ~2 us) make it 9.5 us against 7.7 at B=1. Whole row groups per pair
  of blocks would remove most hand-offs.
- Grid barrier: 1268 ns, of which the release-add is ~460 ns more than a relaxed add (tools/bench/prim.cu). Five
  barriers per layer. A relaxed barrier needs every store that another block reads after it to bypass L1 or carry
  a tag: an audit of all stores, not attempted (risk of rare stale reads).
- B=8: the fixed phases (attention input, attention, post-attention, routing: ~45 us per layer) keep most SMs idle
  while the expert GEMVs keep all of them busy. Running half of the batch half a layer ahead of the other would
  overlap them; it needs two interleaved phase machines in the kernel.
- Track B (not this run): the exact 12-bit BF16 code of finding F1 would shrink qkv/o_proj/router/unembedding bytes
  by 25%; the unembedding alone is 70 us per step at ~6.4 TB/s.
