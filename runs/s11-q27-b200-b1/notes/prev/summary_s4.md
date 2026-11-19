# Qwen3.8-27B decode engine on one B200: final summary (hour 24 of 24)

## Best submission

**s0048, track A: 8938 us per token at B=1 and 9800 us per step at B=8 over HTTP** (the evaluator's client),
1.065x / 1.044x SGLang 0.5.23 (9523 / 10231) and 1.13x / 1.09x vLLM 0.32.0 (10110 / 10705). The 1.10x target
(8657 / 9301) is not reached: 3.2 % / 5.1 % short. `engine/` is byte-identical to s0048's source
(engine/TRACK = A).

| | B=1 | B=8 | vs SGLang |
|---|---|---|---|
| track A, s0048 (one token per pass, checkpoint formats) | 8938 | 9800 | 1.065x / 1.044x |
| track B, s0050 (exact 12-bit weights, one token per pass) | 10687 | 11270 | 0.89x / 0.91x |
| track C, s0043 (exact draft-and-verify; not comparable to token-by-token numbers) | 414 | 3275 | - |
| SGLang 0.5.23 | 9523 | 10231 | |
| vLLM 0.32.0 | 10110 | 10705 | |

- Track A: the checkpoint's BF16 weights re-laid into MMA fragment order, BF16 activations and K/V, float32 where
  the HF path uses float32 (recurrent state, norms, softmax); one token per sequence per pass, no speculation.
  Checks of the final kernel: logits vs the float32 reference rel-L2 median 0.0101 / max 0.0296, top-1 99.4 %;
  single == batched bitwise; generate == step-by-step greedy at B=1 and B=8; serving at B=2/3/5/7.
- Track B: the only measurement is s0050, the same engine with the weights as an exact 12-bit code. It is 20 %
  slower. Its source is in submissions/s0050 (it needs the one-line api change that allows W12 with TRACK
  "A W12").
- Track C: exact draft-and-verify from the MTP head and context copies, every emitted token the model's own greedy
  token. Mislabeled "track B" in the hypotheses of s0014-s0032. Builds s0039-s0049 were submitted after the
  operator's note of 2026-11-12 09:10 ended track-C work; that note was read too late (mk note sent).

Track A over the run, at B=1 / B=8: 10337 / 12028 (s0009) -> 10184 / 11795 (s0013) -> 9981 / 10713 (s0031) -> 9502 /
10291 (s0035) -> 9380 / 10199 (s0037) -> 9006 / 9900 (s0042) -> 8938 / 9800 (s0048).

## The engine (engine/mk.cu, engine/api.py)

- One persistent kernel per model pass: 152 blocks x 1024 threads run a program of ops (GEMM, recurrent op,
  attention ops, head, argmax) with a grid barrier after every op. The barrier right before a GEMM asks L2 for the
  first 128 KB of each block's share of that GEMM.
- GEMM: m16n8k16 BF16 fragments, up to 32 columns. Each warp streams an equal contiguous share of 2 KB quads.
  Tiles cut between warps are summed in shared memory. Tiles cut once between blocks are handed over through
  sentinel slots polled with relaxed gpu-scope loads. The summation order is fixed, so results are deterministic
  and batch-invariant. RMSNorm is folded into the epilogue, and the residual epilogue writes the next matrix's
  input.
- Recurrent op (48 gated delta-rule layers): the state stays in registers.
  - B=1: each head runs on 2 blocks.
  - B >= 4: two units per block.
- Attention (16 layers): prep, MMA scores, softmax (four warps per row, scores in registers), MMA value sum with
  the gate folded in. Units go round-robin over blocks.

## What worked (track A, all bit-identical)

1. **Reloads after op calls (-2.5 to -2.7 %).** The main loop kept the program pointer, op count and thread
   indices in local memory across each op's call. They were reloaded right before every barrier, after a weight
   stream had evicted L1. Now they are re-read from shared memory and special registers (findings C14).
2. **GEMM hand-over after the stream (-2.4 to -2.8 %, then -0.6 % and -0.7 % more).** All 32 warps rebuilt the
   partition with 64-bit divisions after the stream. Now:
   - quad bounds are published in shared memory before streaming;
   - the first warp of a tile is found with one ballot;
   - finalize roles are decided at each warp's own stream end;
   - the prologue is leaner (findings C12).
3. **Far L2 prefetch off (-0.9 % B=1, -1.5 % B=8).** The 1 MB prefetch before small ops delayed barrier releases
   and the recurrent-state reads.
4. **Attention units round-robin over blocks (-0.6 %).** They used to run on 1-6 SMs. The recurrent op's buffer
   index and replay count are also cached per launch (B=8 -0.8 %) (findings C13).
5. **Smaller wins (-0.3 % to -0.6 % each):**
   - softmax scores held in registers, and gates loaded early in the attention output fold;
   - the recurrent op split over 2 blocks per head at B=1;
   - the cut-tile sentinel hand-over (from the Flint run);
   - debug timers and debug flags compiled out of the default build.

## What did not work (numbers in journal.md)

- **Kernel shape and data flow:** 512 threads and 64 columns; 2-D partitions; dataflow flags between GEMMs;
  double-buffered or L2-prefetched stream loops; block-aligned partitions.
- **Weight format:** 12-bit weights, twice. The per-quad patches inside the decode loop cost more than the bytes
  saved.
- **Prefetching:** larger, earlier or later L2 prefetches; prefetch issued after the barrier arrival; idle blocks
  prefetching out_proj (it slowed the recurrent op).
- **GEMM tail:**
  - weighted warp shares inside a block (+2 to +4 %);
  - the outgoing tile sent before the block sync;
  - batched block sums (spills);
  - any extra noinline helper on the hand-over path (+4.5 %: a call costs a local-memory round trip after a
    stream);
  - finalize inlined into the stream.
- **Recurrent op:** four units per block; a second B=8 round on whole blocks; lm_core inlined; the unit re-read
  from shared memory after its calls. All neutral or worse, mostly through code-generation side effects.
- **Small re-tunings:** next-norm gains in shared memory; the next op's descriptor loaded by another warp; barrier
  sleep and R_ASYNC_MAX. All noise or worse.

## What I would try next

1. **The skeleton.** With every weight stream switched off a B=1 step still takes ~1.8 ms (B=8: 2.6 ms). Per GEMM
   op that is about:
   - a stream-end spread inside a block of ~1.8 us;
   - hand-over plus epilogue of ~2 us;
   - a barrier release of ~1 us.
   The structural fix is to let the next GEMM start on blocks that are done, without a grid barrier between GEMMs.
2. **Flat recurrent-op functions.** One function per variant with no inner calls removes the remaining
   local-memory save/restore around calls. Check the SASS with tools/sass_fnmap.py and tools/sass_ldl_after_call.py.
3. **A W12 that pays (track B).** Keep the hot loop free of patches and apply an exact sparse correction per tile
   in the epilogue. The 12-bit decode loop alone was +23 % faster than BF16 in isolation (findings C6).
4. **The B=8 extras, both unexplained.**
   - The recurrent op (~1.15 ms) scales with the units an SM runs, even without state traffic.
   - gate_up's slowest warp ends ~9 us later than at B=1. This grows with the number of real columns and is the
     same with identical columns.
5. **Fold attention prep into the score op:** one op and barrier less per attention layer, ~0.5 %.

Tools for the measurements: tools/op_times_p.py, tail_p.py, ho_stamps.py and lmprof.py (build with
MK_CFLAGS=-DBAR_SLEEP_NS=200; add -DHO_STAMPS or -DLMIX_PROF for those two). tools/ev_time.py works with the
default build. bench/variants/ab_bits.sh runs a same-card A/B with a bitwise check.
