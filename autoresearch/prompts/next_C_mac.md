You are starting a long autonomous run: <hours> hours, this Mac (<card>: Apple M5 Max, 40-core GPU, 48 GB of unified
memory, Metal 4), measured by the run's evaluator. It continues an earlier run on the same task.

## Goal

Make `<model>` decode one sequence faster on this Mac, with exact speculative decoding that drafts with the
checkpoint's own DSpark head (`mtp.safetensors`, `mtp_config` in `config.json`). The general engines measured here:
<baseline>. The bar is the fastest, oMLX on its own 4-bit requantization of the weights with DSpark speculation
(`omlx-oq4e-ds`). The evaluator serves the engine over HTTP: a 64-token prompt, 1,024 greedy new tokens, one
sequence; the rate is counted in tokens (a verification step that commits five tokens counts five). It measures the
synthetic prompts (the headline, `http_step_us_b1`) and natural text (prose, Python source, word problems;
`natural_b1`).

## Where the earlier run left off

`engine/` is the earlier run's final engine (its submission s0014): 8982 us per token synthetic, 5333 natural
(1.53x / 1.56x oMLX + DSpark), one-token step about 13970 us. Its records are in `notes/run1/`: `summary.md` (read it
first: the engine's design, what moved the number, the caveats), `journal.md` (measurements: bandwidths, where a
pass's time goes), the experiment diffs, and `ledger.jsonl` / `notes.jsonl` (every submission and the run's notes).
Its test scripts are in `tests/`. Three of its findings set the terms of this run:

1. **The passes are near the memory roofline.** A one-row pass reads about 3.4 GB of dense BF16 weights and about
   2-3 GB of experts at 450-560 GB/s. Kernel work alone has little left to give; the next large factors are in how
   many tokens each pass commits and how many bytes each pass reads.
2. **The headline is noisy.** The three repeats of each workload are different prompts; a continuation that falls
   into a loop decodes 2-3x faster (copy proposals), and which prompts loop is decided by last-bit numerics. One
   source measured 10070, 8658 and 9070 us. Treat a difference under about 5% as noise until it reproduces, repeat a
   result before you build on it, and keep engine-internal measures (pass time per K, tokens per verify on fixed
   continuations, with and without copy proposals) next to the board's number.
3. **Summation order is a lottery ticket.** A build that changes any per-row summation order changes the
   continuations and the numeric gate's p99 (0.064 to 0.121 over builds, against a limit of 0.1207; s0010 failed
   it). Speed that keeps every sum's order (as s0012 and s0014 did) is a clean A/B; prefer it.

## This run: algorithms, not only kernels

Look for changes to the algorithm that commit more tokens per pass or read fewer bytes per token, while every
returned token stays exactly the engine's own one-token greedy token. Candidates, to measure and rank yourself (none
is known to win here):

- **Lossless re-encoding of the weights.** The dense BF16 matrices are the largest read of every pass. Their
  exponents carry far fewer than 8 bits of information; a lossless code (for example a short exponent code with an
  escape for rare exponents, or an entropy code as in DFloat11 / ZipNN) cut the bytes read while the kernel decodes
  every weight to the identical BF16 bits. This is allowed: the weights the model computes with are bit-for-bit the
  stored ones (check it), so it is not a requantization. The same question for the NVFP4 / MXFP8 codes and scales.
  Whether decoding costs less than the bytes it saves on this GPU is the experiment.
- **Token-tree verification.** Verify several candidate continuations in one pass (the DSpark head's top
  alternatives at uncertain positions, copy proposals, both), each row attending to its own ancestors; the short
  convolutions and the attention residuals then follow the tree path, not the row order. Dense weights are read once
  for all rows; the experts' union grows with the rows, which is the cost to weigh.
- **A draft policy from a cost model.** Measure the pass time for each K (and tree shape), estimate each draft row's
  acceptance from the draft's probabilities, and choose the rows that maximize expected tokens per millisecond.
- **Fewer host round trips.** The earlier engine syncs with the host twice per cycle. Accepting on the GPU and
  drafting the next block before the host has seen the result (on the assumption that all drafts are accepted,
  discarded otherwise) can hide that latency.
- **Better proposals than n-gram copies**, for example a suffix structure over the prompt and the output
  (SuffixDecoding), combined with the DSpark drafts.

One measurement from outside the run that bears on bit-identity: on this GPU (MLX 0.32.4), `mx.matmul` in float32
with more than one row is not float32-accurate (relative error about 7.7e-4 against float64, as TF32), while one row
(GEMV) and MLX's CPU are (about 1e-7); `mx.gather_qmm` was accurate in a 16-row test. A K-row pass built on such a
product would not have the one-row step's bits.

Run the cheap measurements that decide between these before building any of them, and write down why you chose.
Report the DSpark-only control (copy proposals off) for your final engine, as the earlier run did.

Load the `megakernel` skill now: it has the contract on this device, how the evaluator scores and what its gates
check, how to log, and how to work. Your tool is `<mk>` (`mk submit`, `mk board`, `mk note`); put its directory on
your PATH.

## Your track: C

Track C is exact draft-and-verify. Every token the engine returns must be the token its own one-token path (`step()` +
argmax) chooses at that point: speculation changes the speed, never the text. The evaluator checks this exactly, not
on average: the engine's `generate()` and the tokens its server streams for the reference prompts, one request at a
time, must equal its step-by-step greedy tokens on every sequence, full length (gates `generate` and
`http_tokens_b1`). A verify pass whose per-row arithmetic differs from `step()`'s can flip a near-tie and fail the
gate. Every submission of this run is track C (`engine/TRACK` says C).

## What must stay the same

1. **The model's function.** For every input the target returns what the checkpoint defines, in its stored formats:
   NVFP4 experts (E2M1 codes, E4M3 scales per 16, per-expert amax and outer-scale denominator), the MXFP8 experts of four
   layers (E4M3 codes, UE8M0 scales per 32), everything else BF16. No requantization, pruning, distillation or
   approximation of weights or activations, no part of the model skipped. A lossless re-encoding that the kernels
   decode to the identical stored values is allowed. The draft only proposes (it may use lossy copies, as now); the
   evaluator's gates decide.
2. **The interface and the workload.** The contract in `engine/api.py` (on this device: NumPy int64 tokens in, an
   mx.array or NumPy array of logits out; `submit` / `poll` for serving, where speculation lives) and what the
   evaluator measures. Batch 1 is the objective; batch 8 is not measured.
3. **The setting.** This Mac, the checkpoint's stored weights as the source, every result reproducible from the
   submitted source. No general engine or library model code in the timed path: not mlx-lm, mlx-vlm, oMLX, llama.cpp /
   ggml, transformers or any other engine's model, layer or generation code (you may read them, and run them in your
   own tests as a reference). MLX core is allowed: arrays, `mx.fast.*` (including `mx.fast.metal_kernel` with your own
   Metal source), `mx.compile`, streams. So is your own Metal compiled at run time (there is no offline `metal`
   compiler on this machine: Command Line Tools only, no Xcode), and C, C++ or Objective-C++ you compile with `clang`
   in `engine/build.sh` (Metal through `MTLDevice newLibraryWithSource:`). A re-encoded copy of the weights is made by
   the engine from the checkpoint (at load or in `build.sh`, cached on disk if you like), never by hand.
4. **The evaluator is not yours.** Do not read, edit or work around the evaluator or its reference data.

One job at a time on the GPU: CPU and GPU share one memory system, so a test of yours running during an evaluation
spoils its timing, and so does anything heavy left running on the Mac.

Nobody is watching live and nobody will answer questions: decide, write down why, and continue until the time on the board
is used up. Keep `notes/summary.md` current as the earlier run did: final source, what moved the number, controls,
caveats.
