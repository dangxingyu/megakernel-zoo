---
name: megakernel
description: How to work in this run on Apple silicon: build a model-specific decode engine (Metal / MLX) for one model on this Mac, measure it through `mk`, and log every step so the progress can be plotted. Read before writing code, before every `mk submit`, and whenever a result surprises you.
---

# Building a model-specific decode engine on a Mac

## What you are making

An engine that decodes the run's model on this Mac (`mk board` names both) faster than the general engines that run
here (llama.cpp, oMLX, mlx-lm: `mk board` shows their numbers), at batch 1. The thesis being tested: a general inference
engine is the wrong shape for decoding; an engine written for one model, with the GPU never waiting for the host, is
faster. On this run the engine is also speculative (track C): it drafts with the checkpoint's own DSpark head and
verifies exactly. Your engine and its measured numbers are the product.

Target: at least **1.10x** the fastest general engine at batch 1. Matching it is not a result.

## The contract on this device

`engine/api.py` in your workspace defines `Engine` (the starter there is mlx-lm's model behind it). The harness runs with
`MK_DEVICE=metal` (the default on macOS; `eval/_device.py` in the harness is the reference for the types):

```python
Engine(model_dir: str, max_batch: int, max_context: int)    # max_batch is 1 in this run
reset()                       # forget every sequence
step(tokens)                  # NumPy int64 [B]: next input token of each sequence (B fixed between resets);
                              # returns next-token logits [B, 200058] as an mx.array (may be lazy: the harness calls
                              # mx.async_eval on it, and mx.eval + mx.synchronize before it stops a clock) or NumPy;
                              # float32, bfloat16 or float16
close(), stats()              # optional; stats() is a dict of your own counters, stored with the result
generate(tokens, n_steps)     # optional: greedy decode inside the engine; returns integers [B, n_steps] (mx or NumPy)
submit(rid, prompt_ids, n)    # optional, the online contract (eval/serve_engine.py): Python lists of token ids in,
poll() -> [(rid, [tokens], finished), ...]   # every new token out; a verify step may hand over several tokens at once
```

The HTTP server drives the engine through `submit` / `poll` when it has both (that is where draft-and-verify runs),
otherwise through `reset` / `step` and one argmax per token. `engine/build.sh` runs before every evaluation (cwd = the
workspace). Everything else under `engine/` is yours: Python, Metal source, C / C++ / Objective-C++ compiled with
`clang`. The specification of what to compute is the checkpoint (`config.json`, `model-00001-of-00001.safetensors`,
`mtp.safetensors`); mlx-lm's `inkling2` model (in `.venv-mac`) and the B200 engines under
`engines/inkling-2-mini_b200/` are readable descriptions of it.

Not allowed (a submission that does any of these is void whatever its numbers): mlx-lm, mlx-vlm, oMLX, llama.cpp /
ggml, transformers or any other engine's model, layer or generation code in the timed path (step, generate, submit,
poll); caching or hard-coding outputs; a path that skips work when it recognizes the evaluator's inputs; reading the
evaluator's reference data. MLX core (`mx.array`, `mx.fast.*`, `mx.fast.metal_kernel`, `mx.compile`, streams) and
your own Metal kernels are fine; so is `mx.load` / safetensors for reading the weights.

## Your tools

You work on the Mac itself. The Python with MLX is `.venv-mac/bin/python` in the repository (`$MK_PY`; add packages with
`VIRTUAL_ENV=.venv-mac uv pip install ...`), the model is at `$MK_MODEL_DIR`.

- **Metal without Xcode.** Only the Command Line Tools are installed: no offline `metal` / `metallib` compiler and no
  Xcode GUI. Compile Metal source at run time: `mx.fast.metal_kernel(name, input_names, output_names, source, header)`
  (MLX wraps your body in a kernel with the given buffers; `threadgroup` memory, SIMD-group functions, `atomic_*` all
  work), or your own Objective-C++ (`clang++ -fobjc-arc -framework Metal -framework Foundation`) that calls
  `[MTLDevice newLibraryWithSource:options:error:]` and owns its command queue. Mixing the two means two queues; order
  them explicitly (MLX streams, `MTLSharedEvent`) and measure the cost.
- **Timing.** `time.perf_counter()` around N steps after `mx.synchronize()`, as the evaluator does. For GPU time per
  command buffer, `MTLCommandBuffer.GPUStartTime` / `GPUEndTime` from your Objective-C++ path. For time inside a
  command buffer, `MTLCounterSampleBuffer` with the timestamp counter set: Apple GPUs sample at stage boundaries (start
  and end of a compute encoder; check `supportsCounterSampling:`), not at every dispatch, so put the dispatches you want
  to separate in their own encoders while you measure. Keep instrumentation out of the submitted timed path, or behind
  an environment variable that is off by default.
- **Traces.** Metal System Trace (`xcrun xctrace record --template 'Metal System Trace'`) needs full Xcode; check
  `xcrun xctrace version` before relying on it (on this Mac it was not installed when the run was prepared).
  `mx.metal.start_capture(path)` (with `MTL_CAPTURE_ENABLED=1`) writes a `.gputrace` you can only open in Xcode. Without
  them: microbenchmarks of single kernels against the bandwidth bound, and GPU timestamps as above.
- **Bandwidth.** `scripts/mac_membw.py` measures the memory read rate; `eval/modelinfo.py` has the bytes per decode step.
  MBU = bytes a step must read / step time / the measured rate (`modelinfo.CARDS["m5max"]`: 565 GB/s read; a hand-written
  uint4 read kernel reaches about 530).
- `mk submit --title ... --hypothesis ... [--category ...]` sends `engine/` as it is now to the evaluator and returns
  the result. This is the only thing that counts. The evaluator runs on a copy of your source; `engine/build.sh` must
  be able to build everything from scratch. `--quick` runs a shorter evaluation that is logged but never ranked.
- `mk board` shows every submission, the best so far, the reference lines and the time left.
- `mk note "..."` puts a line on the run's timeline.

One job at a time on the GPU: CPU and GPU share one memory system; a test of yours during an evaluation spoils its
timing. Long runs heat the chip: compare numbers taken under the same conditions (`pmset -g therm`).

## What the evaluator does with a submission (on this Mac)

1. **Build** (`engine/build.sh`).
2. **Check** against a float64 run of the model on 8 fixed sequences, each **one at a time** (batch 1): every
   position's logits to position 127 (gate `numeric`), every 8th position to 1151 (gate `numeric_long`), the KL of the
   next-token distribution (`kl`, `kl_long`), top-1 agreement with float64 (`tokens`), all finite, the first sequences
   decoded twice (`batch`: here a repeatability check). Limits: 2x the error of the BF16 reference implementation
   against float64 (median and 99th percentile), 4x its KL, top-1 within one point. The slot-order invariance gate
   needs a batch of 8 and is skipped here. Then the free-running greedy continuation from each prompt (step + argmax),
   and `generate()` when the engine has it: **track C requires every sequence identical** (gate `generate`).
3. **Time**: batch 1, 64 warm-up steps, then 256 timed steps, three times; median microseconds per step. Logits from
   inside the timed run are checked like any others. Also timed: a 64-token prompt followed by 1024 greedy tokens with
   the token read on the host every step (`w1_like_step_us`), and `generate()` when the engine has one.
4. No trace stage (no CUPTI here): `single_launch` and `gpu_never_waits` are not judged on this device.
5. **Serve and measure over HTTP: this is the score.** The engine behind the harness's server (`eval/serve_engine.py`),
   measured by the client that measured the general engines (`eval/client.py steady`): prompt 64, 1024 new tokens,
   greedy, batch 1, one warm-up and three repeats, on the synthetic prompts and on natural text. Track C counts tokens,
   not chunks. Gate `http_tokens_b1`: the tokens the server streams for the check's prompts, one request at a time, must
   be exactly the engine's own step-by-step greedy tokens.

Result: `status` (`valid` only if every gate passes), the headline `http_step_us_b1` (microseconds per token at batch 1
over HTTP, synthetic prompts) with `natural_b1` and `tokens_per_event` (tokens per verify step, your acceptance length)
next to it, `speedup` / `speedup_natural` against the baselines, `perf.b1` (`step_us`, `w1_like_step_us`, `mbu`),
`milestones_passed`. A `--quick` evaluation has no HTTP stage and is never ranked.

**MBU** = bytes a one-token step must read / step time / 565 GB/s. For this model at batch 1 that is about 5.6 GB (the
12 looped layers are read three times per token; 8 of 128 experts per layer application; the input embedding table not
counted), so 100% MBU is about 9.9 ms per token for one token per pass. Speculation is how to go below it: a verify
pass over k positions reads the dense weights once for all k (the experts: the union of what the k rows choose).

## The ladder

Climb in order. Each rung is a property the evaluator can see.

| Rung | Property | How it shows |
|---|---|---|
| `build` | loads, runs | evaluator runs |
| `correct` | your own MLX / Metal computes the target: operators, one layer, the loop, a decode step with caches; no mlx-lm in the step | all gates pass |
| `verify` | a target pass over k positions (k = 1..8) that gives every row the logits `step()` gives it, tested on its own | your tests; `generate` gate |
| `lossless_spec` | `submit` / `poll` draft with the DSpark head and verify; the streamed tokens are the step-by-step greedy tokens | gates `generate`, `http_tokens_b1`; `tokens_per_event` > 1 |
| `parity`, `h0_step` | faster than the fastest general engine at batch 1; by 10% | `speedup` (and `speedup_natural`) |
| `device_loop` | draft, verify, acceptance and the next draft without a host round trip per verify step | `generate` timed; host waits per token in your own counters |

What has worked on other GPUs (the CUDA runs in `runs/` and the playbook next to this file); hypotheses to measure here,
not a recipe: few large dispatches instead of one per operator; weights laid out so each byte is read once per step;
the experts of a verify pass gathered once for all rows; the argmax and the acceptance test on the GPU; the host kept
off the per-token path. Apple-specific things to measure rather than assume: dispatch and command-buffer overhead
(tens of microseconds per `mx.eval` round trip), threadgroup memory and SIMD-group reductions in your kernels, the
cost of NVFP4 / MXFP8 decoding in the inner loop against the 565 GB/s bound, and `mx.compile` of your own functions.

**The sync rule.** Nothing on the GPU waits for the host on the per-token path. Every `mx.eval`, `.item()`,
`np.array(...)` of a GPU result or `waitUntilCompleted` in a step is a host round trip; a speculative engine needs the
accepted count somewhere, so decide where it goes and count the round trips (`stats()`). Say in the hypothesis when a
change adds or removes one.

## Correctness is not negotiable

Performance numbers exist only for `valid` submissions. Typical ways to fail quietly: reading stale or unwritten
cache entries after a rejected draft (roll back the K/V, the short-convolution states and the looped layers' caches of
every pass to the accepted position), a verify row whose arithmetic order differs from the one-row step (a near-tie
flips and track C's exact gates fail), the sliding window (1024) at its boundary, the soft-capped logits
(30 tanh(z / 30)) and the unpadded vocabulary (200058 of 201024 rows of the unembedding). One measured trap on this Mac
(MLX 0.32, M5 Max): `mx.matmul` in float32 with more than one row takes a less exact path than with one row (row 0 of
an 8-row product differs from the 1-row product by about 1e-3 relative, in bfloat16 by up to 2e-3), so a verify pass
built from MLX's batched matmuls does not reproduce `step()`'s logits; write the verify rows and the one-row step with
the same kernel and the same accumulation order, and test bit equality row by row. Test each operator and layer
against the reference implementation before you submit; a submission is slow feedback, a unit test is fast.

## Logging: every step must be plottable

- **`--title`**: at most 80 characters, what changed, in plain words.
- **`--hypothesis`**: one or two sentences written **before** you see the result: what you changed, why it should
  help, the number you expect to move by how much. Never rewrite it afterwards.
- **`--category`**: the mechanism, not the file.
- One idea per submission. After each result, an entry in `notes/journal.md`: expected, measured, conclusion, next.
- `mk note` when something off the submission path matters (a dead end, a tool that misled you, a restructure).
- Commit your workspace to git with a message naming the submission id whenever a submission is valid.

## Working rhythm

Read the board, pick the largest remaining loss (measure to find it), state the hypothesis, make the smallest change
that tests it, check it with a test of your own, submit, record. Keep going until the time on the board runs out; do
not stop to ask for confirmation, nobody is watching the session live.
