You are starting a long autonomous run: <hours> hours, this Mac (<card>: Apple M5 Max, 40-core GPU, 48 GB of unified
memory, Metal 4), measured by the run's evaluator.

## Goal

Make `<model>` decode one sequence faster on this Mac than the fastest general engine here, with exact speculative
decoding that drafts with the checkpoint's own DSpark head (`mtp.safetensors`, `mtp_config` in `config.json`). The target
is at least 1.10x the fastest general engine at batch 1 (<baseline>). Two of these lines matter most. The practical bar is
the fastest, oMLX on its own 4-bit requantization of the weights with DSpark speculation (`omlx-oq4e-ds`): beat it, while
your engine stays lossless on the checkpoint's stored formats (it reads more bytes per token than a 4-bit copy does, so
the speed has to come from the engine and from speculation). The like-for-like line is mlx-lm on the stored formats, one
token per pass (`mlxlm-native`): the starter is that code. The evaluator serves the engine over HTTP: a 64-token
prompt, 1,024 greedy new tokens, one sequence; the rate is counted in tokens (every streamed chunk weighted by the tokens
it carries, so a verification step that commits five tokens counts five). It measures two workloads: the synthetic prompts
(the headline, `http_step_us_b1`) and natural text (prose, Python source, word problems; `natural_b1`). Both are compared
with the general engines; beat them on both.

`engine/` is the engine you start from: mlx-lm's model code behind the contract, one token per step, no speculation.
Load the `megakernel` skill now: it has the contract on this device, how the evaluator scores and what its gates check,
how to log, and how to work. Your tool is `<mk>` (`mk submit`, `mk board`, `mk note`); put its directory on your PATH.

## Your track: C

Track C is exact draft-and-verify. The engine may propose several tokens with the DSpark head and check them with one
pass of the target model over those positions, but every token it returns must be the token its own one-token path
(`step()` + argmax) chooses at that point: speculation changes the speed, never the text. The evaluator checks this
exactly, not on average: the engine's `generate()` and the tokens its server streams for the reference prompts, one
request at a time, must equal its step-by-step greedy tokens on every sequence, full length (gates `generate` and
`http_tokens_b1`). A verify pass whose per-row arithmetic differs from `step()`'s can flip a near-tie and fail the gate.
Every submission of this run is track C (`engine/TRACK` says C).

The draft head is part of the checkpoint: three decoder layers over a block of 8 rows (the last committed token and seven
mask rows), conditioned on features of the target (the hidden state entering layer applications 5, 23 and 41 and the
final norm's output), and a low-rank Markov recurrence that turns the block into seven drafts. The B200 track-C engine
in this repository implements it (`engines/inkling-2-mini_b200/offline-c/engine/c/draft.py`, PyTorch on CUDA; read it
as a description, its kernels do not run here). Where the time goes on this Mac is yours to find and measure.

## What must stay the same

1. **The model's function.** For every input the target returns what the checkpoint defines, in its stored formats:
   NVFP4 experts (E2M1 codes, E4M3 scales per 16, per-expert amax and outer-scale denominator), the MXFP8 experts of four
   layers (E4M3 codes, UE8M0 scales per 32), everything else BF16. No requantization, pruning, distillation or
   approximation of weights or activations, no part of the model skipped. The draft only proposes; the evaluator's
   gates decide.
2. **The interface and the workload.** The contract in `engine/api.py` (on this device: NumPy int64 tokens in, an
   mx.array or NumPy array of logits out; `submit` / `poll` for serving, where speculation lives) and what the
   evaluator measures. Batch 1 is the objective; batch 8 is not measured.
3. **The setting.** This Mac, the checkpoint's stored weights as the source, every result reproducible from the
   submitted source. No general engine or library model code in the timed path: not mlx-lm, mlx-vlm, oMLX, llama.cpp /
   ggml, transformers or any other engine's model, layer or generation code (you may read them, and run them in your
   own tests as a reference). MLX core is allowed: arrays, `mx.fast.*` (including `mx.fast.metal_kernel` with your own
   Metal source), `mx.compile`, streams. So is your own Metal compiled at run time (there is no offline `metal`
   compiler on this machine: Command Line Tools only, no Xcode), and C, C++ or Objective-C++ you compile with `clang`
   in `engine/build.sh` (Metal through `MTLDevice newLibraryWithSource:`).
4. **The evaluator is not yours.** Do not read, edit or work around the evaluator or its reference data.

One job at a time on the GPU: CPU and GPU share one memory system, so a test of yours running during an evaluation
spoils its timing, and so does anything heavy left running on the Mac.

The records of earlier runs are in the repository's `runs/` folder (every submission's source, hypothesis and result,
and each run's notes), CUDA engines all of them. What worked on another model or another GPU is a hypothesis for this
one, to be measured here.

Nobody is watching live and nobody will answer questions: decide, write down why, and continue until the time on the board
is used up.
