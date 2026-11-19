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

# Findings of run s8-inkling2s-b200x4-on1 (serving; each with the measurement behind it)

## F10. tcgen05 as a GEMM for every format of this checkpoint  [measured: tools/t_pe_layer.py against the reference, e0023 ff.]
- Operands K-major without swizzle, units of 128 rows x one K step: bf16 `kind::f16` K = 16 (4096 B), NVFP4
  `kind::mxf4nvf4.block_scale.scale_vec::4X` K = 64 (4096 B + 512 B scale atom), MXFP8 `kind::mxf8f6f4...scale_vec::1X`
  K = 32 (4096 B + atom, scale in byte 0). N = 8 .. 128 columns in steps of 8; M = 64 also works (its rows sit in
  lanes 0-15 of the four warps; `tcgen05.ld` must be executed by the whole warp). The B operand has the same atom
  layout over the column index, so the quantized activations of up to 128 (row, expert) pairs are one 4608-byte unit
  per K step. All three matched the reference on the first build; the NVFP4 results are bitwise those of the
  m16n8k16 formulation (the sums are exact in FP32).
- A block's chain (ring of stages in shared memory fed by cp.async.bulk, one thread issuing scale copies and MMAs):
  ~0.11 us per tensor-core operation and at most ~46-50 GB/s into one SM. So a GEMM costs max(operations x 0.11 us,
  bytes / 50 GB/s) per block-unit and as many rounds as block-units need on 148 SMs. Consequences measured:
  128-column tiles x 3 row groups sharing the B unit (139.5 -> 123.7 ms per 2048 rows), one issuing thread per row
  group (expert GEMMs -15%), 4 row groups for the dense kinds on large passes so that q/k/v and o_proj fit one round
  (96.5 -> 87.2 ms), and no gain from more or smaller blocks on a 1-row pass (384 serial K steps for o_proj).
- Allows: one kernel for all GEMMs of the model, prompt passes at ~50 us per row. Does not allow: short chains for
  small passes without splitting the K range (a different summation, which every pass size would have to share).

## F11. One arithmetic for every pass size  [measured: tools/t_pe_ref.py, evaluator gates]
- Rows are bitwise independent of batching, chunking and scheduling when: GEMM columns are independent (true for
  tcgen05 and m16n8k16), attention folds page-aligned segments in order with masked keys contributing exact zeros,
  expert outputs are summed in expert order, and every block reduction uses one thread layout for all pass sizes.
  Measured: stepwise == chunked == chunks of 7 / 64 / 501 across page boundaries; schedule gate 12/12 and
  online_vs_step 8/8 + 8/8 in every submission; batch gate bitwise equal.
- What breaks it (measured): another thread count per row for small passes (reductions add in another order: max abs
  diff 10 in the residual state after 6 layers).
- The inherited megakernel could not serve under this constraint: two paths with different rounding diverge within
  ~8 generated tokens on the evaluator's prompts (free-run agreement with the reference path 12.6% of the length).

## F12. Where a prompt pass's time goes on this node  [measured: nsys per kernel, 2048 rows at position 4096, s0007 kernels]
- Per card and pass (103 ms): attention 20.5 ms (m16n8k16 MMAs are ~15 instruction slots each on this card, about
  100 TFLOP/s per card for the legacy MMA; softmax ~30 instructions per (query, key)), dense GEMMs 21.5, expert GEMMs
  19.7, exchanges between cards 16 (expert outputs 11.8, attention outputs 4.2), row kernels 23.
- Cost model fitted on a replay's passes: 17.6 ms + 0.0347 ms per prompt row + 0.84e-6 ms per row and position
  (+0.03 per decode row); decode-only passes 19.3 ms + 0.28 per row. 18.5k prompt rows/s in prompt passes.
- Expert load: with the trace's prompts (words from a pool of ~100 tokens) a layer uses ~120-150 of 384 experts, the
  top 16 take 78% of the pairs, one expert up to 2010 of 2048 rows; pairs per card differ 1.48x (median over layers,
  worst 2.5x) with a fixed placement. The router bias does not predict popularity (rank correlation 0.06). Swapping
  experts between cards from a running count evens the tiles per card to 1.07 (pass -4%).
- Peer memory from kernels: 560-725 GB/s for one pair and direction, ~250-470 GB/s per card when all four read; a
  thread's remote loads are serial at ~2 us. Summing a row's eight expert outputs on the card that holds most of
  them and fetching the result elsewhere moves a third of the bytes of every card summing every row.

## F13. The trace and what decides the share  [measured: tools/trace400.jsonl, tools/sim.py, replays]
- The replayed 120 s: bursts every 10 s of 11-22 simultaneous requests with 14k-209k new prompt tokens each (after
  30% prefix reuse); outputs median 34 tokens. A request misses only on the first token (5 s); the misses are the
  15k-32k-token prompts of the 136k-209k-token bursts.
- Scheduling that turned a 15k rows/s engine from 0.085 to 0.89: passes sized by time; prompts ordered by (can still
  make 5 s, shortest remaining first); a pass as long as the decoding request with the least credit allows (the
  service level is a mean gap); a prompt waits for a block another request is computing; blocks cached after
  admission are picked up at page boundaries.
- The simulator (the engine's scheduler code on a virtual clock with the fitted pass model) predicted every replay to
  within one request; each further 10% of prompt throughput is worth about 2 requests of 189.

## F14. A scheduler that runs on the device  [measured: s0008, s0009; tools/sched_sim.cpp]
- The scheduler is 450 lines of allocation-free C++ (engine/sched.h) executed by one GPU thread per pass (hashing a
  32k-token prompt at admission ~0.7 ms, composing a 2048-row pass ~0.1 ms). The same file compiled for the host
  replays the trace on a virtual clock and reproduces the Python scheduler's result and counters exactly: the port
  was checked before it ever ran on a card.
- Host-mapped memory for the request ring and the token ring, a device arena for prompts (reading 128 KB of prompt
  token by token from host-mapped memory would cost ~30 ms per request: the host copies prompts in bulk instead).
- The pass as a fixed kernel program (largest grids, early exits) costs 3% on a 1-row pass and nothing on large ones;
  the host enqueues it blindly two passes ahead. 0 host syncs per pass; the sealed evaluation gave 0.926 with the
  loop on the device against 0.926 for the same kernels with the Python scheduler.

## F15. What limits a tcgen05 block on this card, by measurement  [e0104-e0163, s0010-s0017]
- The read-out, not the chain, was the largest avoidable part of every GEMM kernel: after the chain four warps read
  the accumulators (`tcgen05.ld`) and write up to 512 two-byte results per thread. Tensor memory gives a warp the 32
  rows of its warp index mod 4 (reading with warps 9-12 gave wrong rows), so more groups of four warps can read the
  same rows, each group other columns. Reader warps 4 / 8 / 12 / 16: 2560-row pass 68.7 / 61.9 / 60.3 / 59.6 ms
  (q/k/v 135 -> 85 us per layer, expert GEMMs 185 + 128 -> 153 + 93, bias GEMM 79 -> 57).
- The same for the softmax of the attention kernel: 4 warps reading scores 37 us per half page, 16 warps 18 us; 28
  warps were slower again (61.1 against 59.0 ms per pass). About 16 working warps saturate an SM (roughly 3 warp
  instructions per ns); row kernels with 9 warps x 7 blocks per SM are at the same rate.
- A chain's time has a part per bulk copy: K steps per stage (dense, experts) (1,1) 85.1 ms, (2,1) 80.3, (2,2) 75.3,
  (4,2) 73.6, (4,4) 73.9 per 2048-row pass (before the read-out change).
- What did not move a chain: 256-column tiles with two row groups instead of 128 x 4 (20% fewer operand bytes: q/k/v
  113 -> 113 us), one copy of the B block scales per stage instead of one per row group (9 -> 7 operations per K
  step), scale columns alternating between two or four sets (no dependency of a copy on the previous MMA). A chain
  of small MMAs is ~0.1 us per K step for bf16 (K = 16) and ~0.24 us for NVFP4 (K = 64, two scale copies), whatever
  the column count; full tiles run at ~0.07-0.1 us per 128 x 128 x 16 bf16 MMA and ~0.15 us per 128 x 128 x 64 NVFP4
  MMA per SM.
- Allows: sizing units so that one round of blocks covers a pass (tensor memory holds 128 x 512 cells per SM: 148 SMs
  cover 2105 rows of o_proj, 2526 of q/k/v). Does not allow: shorter chains for small passes without another
  summation order (a 1-row pass is 13 ms, of which ~5 ms are the three dense chains of 288-384 serial operations).

## F16. Work that every card repeated, and what a fetch over the links costs  [s0012; e0125-e0143, e0150]
- With attention split by heads and experts split by card, everything else of a layer (residual mixes and norms,
  o_proj, router GEMM, routing, MoE-input quantization, short convolutions) was computed on all four cards for every
  row: ~20 ms of a 73 ms pass. A home card per row (quarters of the pass) removes it: 72.7 -> 59.5 ms per 2048 rows.
- What the others need is fetched by kernels reading peer memory: attention input (9216 B per row and input), routing
  and quantized MoE input (2.7-5 KB per row), and the home card gathers attention outputs (9.2 KB per row) and its
  rows' eight expert outputs (55 KB per row). Measured by launching a fetch twice: expert sum +4.0 ms, attention
  input +3.0, gather +1.0 per 2560-row pass (9.5 of 59 ms with the routing fetch).
- A kernel's reads of another card run at ~300 GB/s per card when row-contiguous and at 80 GB/s when scattered in
  16-byte pieces (the GEMM tile layout): the home card keeps a row-contiguous copy of the attention input for that.
  Polling three remote flags from every block of an exchange kernel cost ~15 us per kernel: one block polls (one
  thread per card), the others wait on a local word.
- Two remote round trips per fetch and four fetches per layer are too much for small passes (1-row pass 13.3 -> 14.6
  ms): passes of up to 64 rows stay replicated. Bitwise the same in both modes.
- The window code for "only this card's columns" slowed GEMM kinds that have no window (bias GEMM 61 -> 102 us) until
  it was guarded by a compile-time constant: check every template instance after touching shared kernel code.

## F17. The host's calls wait behind the device's queue  [s0015; e0151-e0154]
- With passes enqueued two ahead, any call the host makes into a card executes after them: a 128 KB prompt copy took
  up to 62 ms (also from page-locked memory, also with the launch threads polling instead of blocking), a read of
  eight error words up to 316 ms. In the evaluation's numbers: submit() max 50-78 ms, driver lateness 72-100 ms.
- Host and device now meet only in page-locked host memory the cards can read: the request ring, the prompts (the
  scheduler kernel copies them into its arena with its 1024 threads at the pass boundary where it admits them), the
  token ring and the pass counter (the watchdog). submit() <= 0.6 ms, poll() <= 1 ms, driver lateness 12 ms (the
  first submit starts the launch threads).

## F18. What decided the last requests  [s0013, s0017, s0019; tools/sim.py, tools/sched_sim.cpp]
- After s0012 every miss was a 26k-32k-token prompt at the end of one of the two largest bursts (up to 209k new
  prompt tokens at once): shortest-first is the order that maximises the count inside 5 s, so only prompt throughput
  moved them: 31.7k rows/s -> 0.984, 40.6k rows/s -> 0.995 (one request at 5.34 s).
- Per-row cost falls with the pass size (23.5 us at 2560 rows, 22.1 at 3584) but long passes are long gaps for the
  decoding requests. Rule: 2560-row passes, 3584 rows only while some waiting prompt (shortest first, at the normal
  size) is projected late; the gap credit of the decoding requests bounds every pass as before. All 189 inside
  (longest first token 4.88 s), time between tokens p99 64 -> 80 ms.
- The simulators (the Python scheduler and the device scheduler's C++ on a virtual clock) agree with each other on
  every variant and were pessimistic by 0.3-0.5 s on the last request: they use a linear pass-time model.

## F19. The attention kernel is bound by softmax instructions  [timestamps in one block, e0194; s0021]
- %globaltimer written by one softmax thread and by the operations thread of an attention block (tiles of 128 keys in
  that build): first softmax pass 2-5 us, second 4-8 us per tile, read-out of a half page 5-8 us. In the adopted kernel
  (256-key tiles, 18 us each) the softmax is ~15 us: 126 q x 256 keys x ~33 instructions = 33k warp instructions at ~3
  per ns. The tensor-core and copy time is ~3 us per tile.
- Hence: a pipeline that overlaps tensor-core work with the softmax loses (attention 201 -> 308 us per layer, the extra
  read-outs and barriers cost more than the 3 us hidden); 28 softmax warps instead of 16 lose; what helps is fewer
  instructions per (q, key) without more live registers: masked scores stored back into tensor memory (`tcgen05.st`)
  for the second pass (-8 instructions), zeros through exp(-1e30 - m) (-2), far keys of global layers without bias
  and mask (s0014). Keeping the scores in registers instead (64 per thread) was slower: spills.
- tcgen05.st / tcgen05.wait::st work as documented with the 32x32b shape: a thread writes 16 cells of its row.
- A tensor core serves its SM's operations one after the other (~0.1 us for a small MMA): two K halves of a dense
  GEMM accumulated side by side in one block gave 13.0 -> 12.4 ms on a 1-row pass, not the half chain time; six row
  groups of small expert tiles in one block were slower than three. Shorter chains need other SMs.

## F20. A deadline rule should not aim at the deadline  [s0019-s0022]
- The pressure rule (largest passes while some prompt is projected late at the normal pass size) with 0.05 s of slack
  put the last request of the largest burst at 4.87-4.94 s in every run, whatever the kernels' speed: it returned to
  normal passes as soon as the projection held. With 0.3 s of slack: 4.62 s, nothing else moved.
- Smaller normal passes under the rule do not lower the time between tokens (simulator): the median request's gaps are
  decode-only passes (13.9 ms + 0.17 per row) and the passes of the burst it arrived in.

## F21. Small passes: what a decode pass is made of, and the K range on two SMs  [s0016, s0023; e0205-e0211]
- A 1-row pass is 48 layers x ~230 us of kernels that are each bound by latency, not throughput: three dense chains
  (q/k/v 288, o_proj 384, router 288 MMAs in series at ~0.1 us), two expert chains (72 and 36 K steps at ~0.24 us),
  attention (two tiles in series), a dozen row kernels of 4-12 us each.
- Two chains in one block do not run side by side (F19), two blocks on two SMs do: with every dense GEMM defined as
  the FP32 sum of two half chains, small passes put the halves on different SMs (hand-over of the first half's
  accumulators through a scratch buffer and a flag word; a two-term sum, so the same bits as both halves in one
  block): q/k/v 32 -> 24 us, o_proj 39 -> 27, router 30 -> 21 per layer; 1-row pass 13.2 -> 11.3 ms. Larger passes run
  both halves in one block: +0.8 ms per 2560 rows (one row group per block, two B streams).
- Expert GEMMs of small passes: one row group per block while that is one round of blocks, three when the card has
  more experts than that (8 rows: 17.3 -> 15.4 ms).
- A decode-only pass of the replay: 12.0 ms + 0.16 per row (s0023); the megakernel this run started from did 3.3 ms
  per token at batch 1 with two kernels that round differently; one arithmetic for every pass shape costs that factor.

## F22. What the replay cannot show: limits, crowds, slow hosts  [s0024, s0025; tools/t_stress.py, e0212-e0217]
- The replay's prompts end at 32,768 tokens and its page store never fills; the contract allows 34,816 positions and
  any page count. A crowd test (150 requests in three waves, 20 .. 34,516 prompt tokens, families of shared
  prefixes, 150-700 pages) found in its first run what 24 valid submissions had not: a 64-entry table of block keys
  per slot, one entry short for prompts of 65-67 full blocks, overwrote the next slot's state; six requests stood
  forever while the loop composed 45,000 empty passes and the watchdog (which watched the pass counter) stayed
  quiet. A fixed-size table in a device-side scheduler needs a test at its last index, not at the workload's.
- After the fix the same test passes with 700 and 150 pages (1582 evictions, 472 recomputations; sampled requests
  bitwise equal to the same request alone), at every length limit (1; 511 / 512 / 513; 3583 / 3584 / 3585; 33279 /
  33280 / 33281; 34815 + 1; 34304 + 512), with 1500 requests at once (the request ring holds 1024: the rest wait on
  the host) and with drop_cache during the run.
- What the host guards now (none of it costs a device call): request sizes and token ids at submit(); the prompt
  memory as a ring whose places are released by the request's first token; the request ring; the ring of token
  records (the device composes no rows while the host is more than 114k records behind); a watchdog on rows.
- Waits between the cards in the replay's prompt passes (in-kernel timestamps around the flag polls): ~0.3 ms per
  pass and exchange is latency (6 us per layer); the expert exchange adds ~1.1 ms because in every layer some card
  has more tiles than the others, the attention exchange 0-0.9 ms (cards 1 and 2 finish their attention later than
  cards 0 and 3 on identical work). Together under 5% of a 58 ms pass.
- `__expf` for `expf` in the softmax: no change in pass time. A second launch program for small passes: the fixed
  grids cost only 0.4 ms of an 11.6 ms pass. Expert tiles of 256 columns, a hot expert's weights on every card:
  estimated at no gain and ~1% (journal, 14.6 - 15.0 h).

## F23. A scheduler on the device is a kernel too: profile it  [s0026, s0027; e0234-e0248, tools/nsys_sched.py]
- `k_sched` had one deciding thread and 1023 that copied prompts and left. Its time was never looked at apart from
  the pass: 65 us on a decode-only pass, but 1.9 ms on every prompt pass and 3.4 ms on every sixth (a GPU thread
  runs dependent global loads at 40-200 ns each): the running average over 18,432 (layer, expert) counts, the swap
  search (48 layers x ~2700 steps), the per-row tables (up to 3584 rows x 4 words), and one walk over the 16,384
  nodes of the cache table per evicted page once the store is full (~1 s per large burst in a 400 s replay).
- All of it is either data-parallel (average, row tables: all threads; swap proposals: a pure function per layer, one
  thread each, merged in layer order by the deciding thread) or avoidable (eviction: per stripe of the table its
  least recently used evictable leaf, kept as a lower bound and repaired when found stale; the same victim as the
  scan). Prompt passes: 2.1 ms -> 0.13 ms of scheduler. 120 s replay: TTFT p99 4.28 -> 4.15 s, time between tokens
  p99 76.5 -> 74.8 ms. 400 s replay (store full after 110 s): 642 -> 643 of 643 in time, TTFT p99 4.07 -> 3.43 s,
  time between tokens p50 43.5 -> 36.3 ms.
- What found it: a replay longer than the evaluation's (the store only fills at its end) and the simulator, which
  runs the same scheduler code with a pass-time model: where the device is slower than the simulator, the time is
  outside the pass.
- The same reading found a cost that only hours of uptime show: an open-addressing table that marks removals and
  never frees a slot has no empty slot left after ~200k evictions (3.3 h of this trace on the virtual clock), and
  every unsuccessful lookup then walks the whole table. Removal now frees the mark when the next slot is empty.

## F24. What a small pass pays for its kernels, and what it does not  [s0028, s0029; e0228-e0264]
- A kernel that finds nothing to do costs ~1.4 us; a grid of the largest pass's size whose blocks exit at once costs
  ~0.5 us per 1000 blocks more (a 1-row pass with the device loop's fixed grids: 11.63 ms, with exact grids 11.25).
  Removing one launch per layer by fusing two per-row kernels with no barrier between them (end of layer i +
  attention mix of layer i + 1), and capping the two fetch kernels' grids with a row loop: 11.47 -> 11.26 ms.
- The same row loop in kernels that reduce over the block (prep, fold) is slower at every pass size: the barrier
  between two rows of a looping block costs more than 3584 empty blocks.
- Choices that shorten a chain are wrong for a GEMM bound by operations: the unembedding (393 row groups, 148 SMs)
  as 1572 quarter units cost 207 us per pass, as 393 whole units ~90 (s0028).
- A pass's rows are not equal: rows of the replay cost 0.16 ms each in a decode pass, rows with unrelated tokens
  0.24 - 0.36 ms: each brings ~8 experts of its own, and an expert tile costs 17.9 MB of weights and 3888
  tensor-core operations whatever its column count. Two expert row groups per block (between "one" and "three")
  changed nothing; neither did the fast exponential in the softmax.
- The device loop adds nothing to a pass: token gaps through submit / poll equal the pass launched from the host
  (11.71 / 11.74 ms at 1 row, 14.64 / 14.57 at 8).

## F25. Reads of another card's memory from a kernel: what makes them slow  [s0031-s0033; e0268-e0298]
- Pieces: a row's routing came from five arrays of its home card (codes, scales, ids, weights, outer scale): 67 us
  per layer at 3584 rows for 2.7 KB per row, against 74 us for the 9.2 KB of the attention input. Without the four
  small pieces 46 us; with them appended to the scale row (one 368-byte piece) 53 us.
- Series: a thread's remote loads complete one after the other (~2 us). The convolution-state fetch with 256
  threads per 18 KB row (six loads in series) took 11 us per layer for a few dozen rows; with 576 threads it is two.
  The gather of attention outputs with 768 threads per row (one load each, three rounds of blocks) instead of 256
  (three in series, one round): -0.4 ms per large pass (s0034).
- Loops: the attention-input fetch as a few blocks per SM that loop over rows is 2 us per layer faster than one
  block per row; the routing fetch with the same loop 12-25 us slower (not understood). Measure each kernel in
  both builds' profiles.
- Not the limit: the request size. Bulk copies (`cp.async.bulk`) work on peer memory, bitwise the same, and the
  expert sum (55 KB per row from three cards) is no faster with them than with 16-byte loads: 35 MB per card and
  layer at the ~420 GB/s the links give when all four cards read.

## F26. Capacity at 8 req/s: the slot table and the admission order, not the kernels  [s0036-s0038; extension]
- The trace at 8 req/s nominal: 7.1 requests/s over 300 s, 58.9k prompt tokens/s offered (37.2k rows/s after cache
  hits), 1289 output tokens/s (mean 182 per request, median 31). One token per sequence and pass, passes with
  prompts ~51-60 ms: ~65-75 sequences decode at any moment. The scheduler had 64 slots and admitted in arrival
  order: with the slots full, every request waited behind the head of the ring. Operator, 300 s over HTTP: 33.6%
  within the service levels, TTFT p50 7.6 s.
- A simulator that runs the scheduler's own code needs only a pass-time model to answer "what if": fitted on one
  measured result (s0036 at 8 req/s, 120 s: 0.836; model 11.6 ms + 16.0 us per prompt row + 0.07 ms per decode row,
  decode-only 11.3 + 0.18 per row), it reproduced the operator's 300 s figure (0.317 for 0.336; TTFT p50 8.1 / 7.6 s)
  and predicted the next three results to the third digit: 128 slots at 120 s 0.991 (measured 0.990), at 300 s
  0.952 (0.947), with priority admission at 300 s 0.982 (0.982).
- 128 slots: 0.336 -> 0.947 at 300 s (in process). 96 slots 0.94; 192 as 128: then the page store binds (~120
  sequences x ~18 pages of 2149).
- Admission by priority (requests that can still answer in time first, cheapest first; the others in arrival order
  while 24 slots and an eighth of the store stay free): 0.947 -> 0.982 at 300 s, TTFT p50 0.44 -> 0.19 s, p90 4.2 ->
  1.8 s; the requests given up on wait longer (p99 9.6 -> 12.1 s). At 10 req/s (simulated) 0.17 -> 0.95. The constants
  (cost of an output token 0-32 rows, 8-48 slots kept free) do not matter.
- Sensitivity at 8 req/s, 300 s (simulated; 38 late of 2120 now): prompt-row cost -5% 23, -10% 21; pass base -2 ms
  26; decode-row cost halved 25; 3000 pages instead of 2156: 23. Kernel speed now buys the tail, not the share.
- Limits: past ~12 req/s the work left when the arrivals end takes more than 90 s (the driver's drain): no order of
  service changes that.

## F27. After the slots: pages, long prompts, and what a decode row costs next to prompts  [s0039-s0042; extension]
- The page store was sized from an estimate made before the pass buffers existed; with the engine serving, 14.7 GB
  per card were unused. Sized after all other allocations (driver's free memory less 0.6 GB of later allocations and
  2.7 GB): 2439 pages instead of 2149, 7.7 GB per card free (the driver reports ~4.8 GB less free at sizing time
  than is free later). 8 req/s: 772 -> 775 of 780 at 120 s, 0.982 -> 0.987 at 300 s; more pages only pay again from
  ~3000 (simulated). A page is 28 MB per card: K and V of 48 layers (25.2) and the convolution snapshots (2.8).
- A decode row in a pass that carries prompts costs 0.11 ms at position 8192 (profile with and without 64 decoding
  sequences): all of it in the attention kernel, 16 us per (row, KV head, page) unit, a 128-row tensor-core tile with
  6 rows in use. The other kernels do not notice the row (its expert pairs join tiles that exist). In a decode-only
  pass the same row costs 0.17 ms up to 64 rows (its ~8 experts are tiles of their own: 3888 operations each) and
  0.04-0.09 ms beyond (64 rows 22.2 ms, 96 26.7, 128 27.9).
- Long prompts: a lone 32,000-token prompt answers after 752 ms (808 before s0039): 9 passes x 11.6 ms of base and
  20 us per row. At position 24,000 the global layers' attention is 10.4 ms of a 70 ms pass: 12.2k units per card
  and layer at ~16 us; a third of a unit is the copy of its K and V half pages into shared memory, repeated for
  every group of 21 rows that reads the page (122 times per page and pass). Keeping a page's operands for several
  groups needs both halves resident (262 KB, more than a block's shared memory) or a record per half page: not built.
- Not the softmax's time, measured: the maximum pass (reference 0 instead: -1.6%), the exponential, the width of
  the tensor-memory reads (64 cells per read: +8%).
- The pressure rule's projection is optimistic under decode load (its rate counts prompt rows only, and the next
  burst is not in it): with a margin of 2 s instead of 0.3 s while 32 or more sequences decode, 775 -> 778 of 780 at
  120 s and 2093 -> 2101 of 2120 at 300 s (TTFT p99 9.7 -> 4.9 s) for +0.8 ms of median gap; 2 req/s unchanged (it
  rarely has 32 decoding). A load-aware rate in the same test instead: no gain (s0042). The last two of 780 need
  prompt rows 20% cheaper (simulated: -5% and -10% leave both late).
- The median time between tokens under load is the load: 37k prompt rows/s through passes of 11.6 ms + ~7 ms of
  decode rows + 16 us per row need ~1690 rows and 46 ms per pass with perfectly even arrivals; the engine has 59.6 ms
  on the bursty trace. Smaller passes while many sequences decode reach 56 - 57 ms, lose a request at the neighbouring
  settings and 20 - 30 more at 10 req/s (simulated, s0042 scheduler). Only the pass cost moves this figure.
- Late requests at 8 req/s are all 32,768-token prompts at the end of their burst; a shorter normal pass while many
  sequences decode buys median time between tokens (59 -> 51-54 ms) for one request of 780: not taken while the
  share is the score.

## F28. The order of service and the third digit of the score  [s0043; extension]
- Shortest-first among the prompts still in time is not the order that loses the fewest deadlines when bursts
  overlap: a later burst's short prompts overtake an earlier burst's long one. Promoting one prompt per step - the
  earliest deadline among those projected to miss in shortest-first order at the largest pass size, if it and every
  prompt it overtakes keep 0.3 s to spare by the same projection - saves one request of 780 at 8 req/s on the pod
  (A/B on one pod epoch: 777 -> 778; 7 promotions in 120 s; none at 2 req/s), and in the simulator 20 -> 17 of 2120
  over 300 s, 6 -> 3 of 1598 at 6 req/s, 103 -> 89 of 2608 at 10 req/s. TTFT p90 at 8 req/s 1.60 -> 1.66 s.
- The same source scores 777 or 778 of 780 on two pods: one 32,768-token prompt (burst at 2.5 s) answers after 4.95
  s in the simulator, under 5 s on pod epoch 6 and after 5.01 - 5.02 s on epoch 7. Margins (2 s from 20 decoding
  sequences), earlier promotion and an honest arrival stamp move it by 0.01 - 0.09 s: its burst needs 4.9 s of the
  largest passes. A result at this level is one request wide; compare builds on one pod and name the request.
- The simulator (scheduler's code, fitted pass model) predicted s0037 - s0042 to the request and s0043 to within
  this one request: at the limit it is 0.07 s optimistic against pod epoch 7.
