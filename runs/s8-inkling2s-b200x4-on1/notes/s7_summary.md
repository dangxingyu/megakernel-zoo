# Summary of run s7-inkling2s-b200x4-fa1 (track A, 12 agent-hours, 20 submissions: 19 valid, 1 failed)

## Best submission
- engine/ = **s0020** (submitted, valid; engine/ink.cu and engine/api.py are byte-identical to its src.tgz):
  **3307 us per token at B=1, 5378 us per step at B=8** over HTTP = 1.424x / 1.228x SGLang (4711 / 6604 us).
  Goal (>= 1.10x at both batch sizes: 4283 / 6004 us) met with margin. Milestones: build, correct, device_loop,
  gpu_never_waits, parity, h0_step.
- Best headline (B=1) figure on the board: **s0013, 3295.6 us** (5484 at B=8). s0020 runs the same B=1 kernel; its
  3307 was measured on the last pod, where that kernel is ~0.3% slower in process (3257 against 3250 us). At B=8
  s0020 is the best of the run (in-process step 5381 us against 5522 for s0013).
- `MK_CFLAGS=-DNO_TC13` builds the single-layout, single-kernel engine of s0013 (99.9 GB per card instead of 145.8).

## What worked (HTTP figures B=1 / B=8)
- s0001 4807 / 8861 (inherited s6 source).
- Team vector phases, one element per thread on 6 blocks per token: 4163 / 7125 (s0002).
- GEMV scheduling (strips per block, flat schedule with tagged hand-off): 3984 / 6537 (s0003).
- Attention scores as BF16 MMAs, tag-checked hand-offs without fences: 3891 / 6335 (s0004).
- Shared memory 222 -> 121 KB (it is taken from the L1): 3646 / 5870 (s0005).
- Work placed in idle blocks (output-gate rows beside attention, small attention chunks, whole w13 strips per
  block at B=1), counter-free hand-offs: 3299 / 5558 (s0007-s0012).
- Attention / routing / quantization phases compiled out of line (register cap): 3296 / 5484 (s0013).
- NVFP4 w13 at batch >= 4 through tcgen05 block-scaled MMAs (kind::mxf4nvf4) from a second layout of the same
  bytes, verified bit for bit on all 40 layers; its calls made from inside the routing phase function; scale-factor
  stores over three idle warp groups; emit inlined: B=8 5484 -> 5378 (s0016-s0020), B=1 kernel untouched.
- Method: tools/t_replay.py (one generated text replayed teacher-forced through both kernels) predicted s0016,
  s0017, s0019 and s0020 to within 0.2% of the step; s0018 gave about half of what it predicted.

## What did not work
- s0014 failed: deadlock in the timing stage. A race of the read-out warps on the mbarrier phase counts that only
  shows with few distinct experts per layer (prompts sharing tokens). Reproduced with tools/t_soak.py --distinct
  1|2|3 in 700-5000 steps, fixed by one barrier (60 000+ steps clean).
- s0015: valid, no gain. Predicted -5% at B=8 from the rehearsal's generated-text figures; builds with different
  rounding generate different text and see different expert unions. Also the tensor-core calls sat in the kernel
  body and slowed the MXFP8 layers' loops 10-15%.
- tcgen05 for w2: correct, not faster (every row group is split over ~4 blocks and recombined; 3.6 us of scale
  stores per phase). tcgen05 for the MXFP8 w13 (kind::mxf8f6f4): correct, only 8% per expert, cancelled by 18 KB
  more shared memory. Source kept in prev/exp_tcmx/.
- Dead ends measured and dropped: software-pipelined loads, 896/1024-thread blocks, GEMV schedulers out of line,
  relaxed grid barrier, K/V unstaged attention, ballot top-k, offloaded dots, 16-byte code stores, four prep groups,
  sub-items of 24/36 units, operand area inside the staging ring (details in journal.md).

## What I would try next
1. B=8: overlap the fixed phases (attention input, attention, post-attention, routing: ~55 of ~110 us per layer,
   most SMs idle) with the expert GEMVs by running half the batch half a layer ahead: two interleaved phase
   machines. This is the only item left with more than a few percent.
2. Tensor-core w13 fixed costs (~4 us per phase: set-up 0.9, ring fill, hand-off ~2): whole row groups per pair of
   blocks to remove most hand-offs; then lower TC_MINB (it already pays at B=3: 3923 -> 3900 us).
3. Grid barrier (1.27 us, five per layer): a relaxed barrier after an audit of every store read across it.
4. Track B (not this run): the exact 12-bit BF16 code (finding F1) for qkv / o_proj / router / unembedding bytes.

Details: notes/plan.md (engine as built, checks, open items), notes/journal.md (every submission, predicted
against measured), notes/findings.md (F1-F9).
