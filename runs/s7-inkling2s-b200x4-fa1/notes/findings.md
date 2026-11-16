# Findings about the checkpoint and the node (each with the measurement behind it)

## F1. BF16 dense weights fit an exact 12-bit code (track B)  [measured, all weights]
tools/census_bf16.py, exec e0020: every value of the 337 dense BF16 matrices of the decoder (per layer wq, gate, wk,
wv, wo, wr, router; and the unembedding): 5,710,970,880 values.
- zeros: 1; subnormals: 0; exponent entropy 2.57 bits.
- values outside the best 16-exponent window chosen per row: 419,063 (7.3e-5); per 16-row strip: 596,073 (1.0e-4);
  per tensor: 668,233 (1.2e-4).
- As packed by engine/w12.py for my layout (row-sliced per card, plus the per-card zero pad rows which get window
  base 0 so that their zeros are exact): 432,825 escapes in 5,984,649,216 values, 12.0009 bits per value.
Allows: 25% fewer bytes for the 11.5 GB of dense weights read every step (~2.9 GB per step at B=1, ~720 MB per card).
Does not allow: anything below 12 bits with a fixed-width code (entropy 2.57 bits of exponent + 1 sign + 7 mantissa).

## F2. 12-bit decode speed on one B200  [measured, tools/bench/w12.cu, e0023]
GEMV streaming of 150 M values: BF16 3.13 G values/s (6.26 TB/s), 12-bit 3.99 G values/s (5.98 TB/s of 12-bit bytes):
+27% values per second. Decode per bf16x2 register: PRMT with sign-replicate (sign to bit 15/31) + LOP3 mask +
SHF/LOP3 nibble extract + IMAD (offset*128 + base) + LOP3 = ~5 instructions.

## F3. Engine-visible properties of the node
- sm_100a: no warp-level block-scaled FP4 MMA; cvt.rn.f16x2.e2m1x2 with free byte selects; NVFP4 GEMV via
  cvt + HMUL2 (E4M3 block scale, exact in f16) + mma.m16n8k16.f16 streams at 5.9 TB/s (BF16 6.2 TB/s) per card.
- 1024-thread blocks get 64 registers; with 222 KB of shared memory per SM the L1 has ~34 KB: any register spill or
  dynamically indexed local array costs an L2 round trip per access (seen as 5-10x slow phases).

# Findings of run s7-inkling2s-b200x4-fa1 (each with the measurement behind it)

## F4. What bounds a phase of the persistent kernel  [measured, tools/bench/red.cu, cold.cu, in-kernel timers]
- A lone warp executes ~3.5-5 ns per instruction; with 24 warps busy an SM delivers 2.2-2.6 G warp-instructions/s.
  The per-token vector phases were ~1800 instructions per thread (8 elements per thread): 10-20 us regardless of
  how many SMs worked. One element per thread on 6 blocks: post-attention phase 15.5 -> 7.8 us (B=1), 21.9 -> 8.6 (B=8).
- The NVFP4 GEMV unit (64 k x 16 rows, 576 B) is 16 E2M1 conversions + 16 HMUL2 + 4 E4M3 conversions + 4 HMMA + loads
  = 44-46 instructions (SASS). In isolation it streams at 6.6 TB/s marginal (gemv2.cu, 768 threads, 300 MB vs 35 MB),
  in the engine at ~3.7-4.5 TB/s.
- A fence (__threadfence) costs ~0.5 us; a relaxed load 0.1-0.3 us and a thread's relaxed loads run one after the
  other; an L2 hit of an L1-bypassing load 37 ns, a cold DRAM load 180-240 ns for any span up to 64 GB.
- Allows: counting instructions on the longest thread path to predict a phase. Does not allow: speeding up a phase
  by adding SMs without shortening the per-thread path.

## F5. Shared memory is taken from L1  [measured, gemv2.cu e0072; engine e0073-e0078]
- FP4 GEMV microbenchmark, 300 MB, 768 threads: 54.7 us with 73 KB of dynamic shared memory, 54.9 us with 150 KB,
  62.7 us with 222 KB (+15%).
- Engine, 222 KB -> 121 KB (token state 768 instead of 4608 elements per block, no GEMV fragment staging), results
  bitwise identical: fixed-token step 3868 -> 3613 us at B=1, 5533 -> 5154 us at B=8 (w2 23.3 -> 18.2 us per layer,
  qkv 7.9 -> 7.0, router 3.8 -> 3.0, unembedding 80 -> 72).
- 47 KB (attention K/V rows read from L2/L1 instead of staged): unembedding 70 us, but attention 11.2 -> 16.5 us
  (plain global loads in the V loop are much slower than shared memory even after an L1 preload): staging kept.
- Allows: treating every KB of shared memory as a KB of L1 lost. Does not allow: moving latency-critical gathers
  (the attention's V sum) from shared memory to L1.

## F6. Cross-block hand-offs without fences  [measured, tools/t_det.py, e0061-e0068]
- Removing the fences between "store data" and "bump counter" made B=8 irreproducible after 30-70 steps (the
  consumer saw the count before the data); B=1 and 40-step runs did not show it.
- Reliable without a fence: (value, tag) in one 8-byte store (st.relaxed or st.global.L1::no_allocate) and a
  relaxed load per word, repeated until the tag is current. 120 steps x 2 runs bitwise equal at B=1 and B=8.
- Not reliable: ld.global.cg for the tag check (current tag, wrong value); a struct load through __ldcg; plain
  L1-bypassing loads faulted (illegal address) in the routing phase for a reason not found.

## F7. Routing statistics that set the B=8 load  [measured, in-kernel counter, e0025/e0026/e0053]
- Distinct experts per layer for a batch of 8: 39.6 (random token ids), 31.6 (continuations of random-token
  prefixes late in a 1000-token context), 50-55 (free-running generation from 8 different 64-token prefixes).
- The expert phases cost ~0.8 us (w13) + ~0.5 us (w2) per distinct expert and layer at B=8.
- Allows: predicting the B=8 step from the union size. Does not allow: comparing two builds on generated text when
  they differ in summation order (the tokens, and so the unions, differ).

## F8. Where work is placed matters more than how fast it streams  [measured, e0097-e0114, s0007-s0011]
- Few items, many warps: at B=1 the flat GEMV schedule cut each of the 576 w13 strips into ~6 pieces on different
  blocks; whole strips per block (4 at a time, reduction in shared memory) took w13 from 12.1 to 8.3 us per layer.
  From ~25 distinct experts per layer on, the flat schedule wins again (B=8: 5086 -> 5146 us with whole strips).
- Work that is needed late can run in blocks that idle during a few-block phase: the output-gate projection (96 of
  ~240 strips of the q/gate/r/sink/k/v GEMV) beside the attention phase, handed over as tagged words: fixed-token
  step 3374 -> 3287 us at B=1; with 24 free blocks (4 strips each) it still pays at B=8 (context 960: 5073 -> 4993).
  Every other GEMV output of a layer is needed by the very next phase.
- The order of a warp's work can replace a protocol: a strip split over two warps is the LAST piece of the first
  warp and the FIRST piece of the second, so the first warp can finish the strip without a counter (B=8 4944 -> 4893).
- The attention's V sum is a per-thread loop over the chunk's positions: 32 or 64 positions per block instead of
  128 while blocks are free: -1% at B=1.
- Allows: looking for idle blocks and late consumers before optimising a loop. Does not allow: assuming that a
  profile build measures the plain build (two extra timer stamps moved the kernel to STACK:104, +1.4% at B=8).

## F9. NVFP4 through tcgen05: the checkpoint's format is the tensor core's native operand  [measured, tools/bench/tc5.cu e0131-e0135, engine e0143-e0172]
- `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X` (sm_100a) computes D[128 x N] (+)= (A x sA)(B x sB)
  with A = 128 rows x 64 k of packed E2M1 codes, sA = one UE4M3 scale per (row, 16 k), B = 64 k x N columns of E2M1
  codes with their UE4M3 block scales: exactly a checkpoint NVFP4 weight block times a model-defined quantized
  activation block. One MMA against the exact sum on random codes: 1024 of 1024 values equal with scales in a
  moderate range (see the journal for the full-range run). The largest block-scale byte in all NVFP4 w13 weights
  is 0x7e (tools/t_tc_layout.py), so reading the scale as unsigned E4M3 changes nothing.
- Layouts that work without a tensor map (plain 1-D `cp.async.bulk`): operands K-major, no swizzle: row m (of 128),
  element k (of 64) of A at byte (m/8)*256 + (k/32)*128 + (m%8)*16 + (k%32)/2, low nibble = even k; descriptor =
  (addr >> 4) | (128 >> 4) << 16 | (256 >> 4) << 32 | 1 << 46. Scale-factor atom (512 B, copied with
  `tcgen05.cp ... 32x128b.warpx4`): row m, block j at (m%32)*16 + (m/32)*4 + j. Scales of B: the same atom over the
  column index; with N = 8 only word 0 of lanes 0-7 is read, and it can be written with `tcgen05.st` (32 columns
  per instruction, pitch 4: a scale address that is not a multiple of 4 columns faults with "misaligned address").
  Instruction descriptor for E2M1 x E2M1, UE4M3 scales, M = 128, N = 8, K = 64: (1<<7)|(1<<10)|(1<<17)|(1<<27).
- Rates on one B200 (151 MB cold per card, the B=8 w13 pattern): one issuing thread per block 4.8 TB/s, four
  streams per block with a 73.7 KB ring 6.5 TB/s; the m16n8k16 formulation: ~4.3 TB/s. In the engine the stream part
  of a phase runs at 6.3-7 TB/s; a phase has ~0.9 us of set-up and ~2 us of hand-off.
- Rules learned: (1) the issuing thread is a serial path: no divisions, descriptors by addition; (2) mbarriers are
  initialised once per launch (re-initialising them lost arrivals even with the init fence); (3) bulk copies of
  weights need `L2::cache_hint` with an evict_first policy, otherwise they evict the step's working data from the L2
  (+1 to +3 us in every later phase at B=8); (4) tensor memory costs nothing while allocated; (5) below ~30 distinct
  experts the fixed costs lose against the fragment path, so both layouts are kept (+46 GB per card for w13).
- In the evaluation (s0016): B=8 in-process step 5522 -> 5433 us (-1.6%), HTTP 5484 -> 5413; B=1 runs the plain kernel.
  The gain needs the tensor-core functions to be called from inside an existing out-of-line phase (as calls in the
  kernel body they cost the MXFP8 layers' loops 10-15%, s0015) and the kernel's shared memory to stay at 155 KB.
- MXFP8: `kind::mxf8f6f4.block_scale.scale_vec::1X` with UE8M0 scales, K = 32, same operand geometry (16 codes per
  16 B), scale id j = byte j of the atom's word: verified (tc5.cu stage 1b, engine gates, layout on all 8 layers;
  scale exponent bytes of the checkpoint 111..118). Only 8% faster per expert than the fragment path there.
- Not a gain for w2: its row groups sum over all union experts, so they must be split over blocks and recombined.
- A/B method that this finding forced: a build that changes rounding generates different text and so sees different
  expert unions; the rehearsal's generated-text figures showed -7% where the evaluation showed 0 (s0015).
  tools/t_replay.py replays one generated text through both kernels: that figure predicted s0016 to 0.3%.
- The one deadlock (s0014, timing stage) is explained: a race of the read-out warps on the mbarrier phase counts that
  shows only with few distinct experts per layer (prompts that share tokens): reproduced on the s0014 source by
  tools/t_soak.py --distinct 1|2|3 within 700-5000 steps, gone with the one-line barrier of s0015 (60 000 steps).
- Does not allow: a block-scaled MMA at warp level on this architecture (sm_120a only), or N < 8.
