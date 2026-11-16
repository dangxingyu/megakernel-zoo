---
name: megakernel
description: How to work in this run: build a decode megakernel for one model on one GPU, measure it through `mk`, and log every step so the progress can be plotted. Read before writing code, before every `mk submit`, and whenever a result surprises you.
---

# Building a model-specific decode megakernel

## What you are making

An engine that decodes the run's model on the run's card (`mk board` names both) faster than the general serving
engines. The thesis being tested: a general inference engine is the wrong shape for decoding; a kernel
written for one model, with nothing on the GPU ever waiting for the host, is faster. Your engine is the evidence.
The loop around you is not the product; the engine and its measured numbers are.

Target: at least **1.10x** the faster of vLLM and SGLang at batch 1 and at batch 8 (`mk board` shows the
reference lines). Matching them is not a result.

## The contract

`engine/api.py` in your workspace defines `Engine` (the starter there is the Hugging Face model behind it):

```python
Engine(model_dir: str, max_batch: int, max_context: int)
reset()                       # forget every sequence
step(tokens)                  # int64 cuda tensor [B]: next input token of each sequence (B fixed between resets);
                              # returns next-token logits, cuda tensor [B, vocab], bf16 or float32
close(), stats()              # optional; stats() is a dict of your own counters, stored with the result
generate(tokens, n_steps)     # optional, later rung: greedy decode inside the engine (argmax, feedback and the
                              # loop on the device). tokens [B] is the next input token; returns int64 cuda [B, n_steps]
```

`engine/build.sh` is run before every evaluation (cwd = workspace). Everything else under `engine/` is yours:
CUDA, C++, Python glue. The Hugging Face implementation of the model is the specification of what to compute.

Not allowed (a submission that does any of these is void whatever its numbers): calling another inference
engine or library to run the model in the timed path (vLLM, SGLang, MPK, TensorRT, llama.cpp, torch modules or
torch.compile of the HF model); caching or hard-coding outputs; a path that skips work when it recognizes the
evaluator's inputs; reading the evaluator's reference data. Using torch tensors to pass buffers, and CUTLASS /
ThunderKittens-style header code you compile into your own kernel, is fine.

## Your tools

You work on the machine that has the GPU. Build, test and profile there directly; the CUDA toolkit (`nvcc`), `nsys`
and a Python with torch are what the setup script installed (`$MK_PY`), and the model is at `$MK_MODEL_DIR`.

- `mk submit --title ... --hypothesis ... [--category ...]` sends `engine/` as it is now to the evaluator and returns
  the result. This is the only thing that counts. The evaluator runs on a copy of your source; `engine/build.sh` must
  be able to build everything from scratch. `--quick` runs a shorter evaluation that is logged but never ranked: use
  it rarely; a unit test of your own is the faster check.
- `mk board` shows every submission, the best so far, the reference lines and the time left.
- `mk note "..."` puts a line on the run's timeline.

Profile before you optimize, and after: kernels by GPU time, GPU idle share, idle gaps, host synchronizations.
One job at a time on the card: a test of yours running during an evaluation spoils its timing.

## What the evaluator does with a submission

1. **Build** (`engine/build.sh`).
2. **Check** against a float64 run of the Hugging Face model on fixed sequences: every position's logits, alone
   and in a batch of 8, under teacher forcing, to position 127 (gate `numeric`), and every 8th position to 1151
   in the batch of 8 (gate `numeric_long`). Gates: relative L2 error of the centred logits at most 2x the error
   the Hugging Face BF16 path itself has against float64, as the median and as the maximum over positions; top-1
   agreement with float64 within one point of Hugging Face BF16's; all logits finite; the same sequence alone
   and inside a batch within the same tolerance (bitwise equality is recorded: aim for it, it is what makes
   batching invisible to a user). The result's `gates` block has every number and its limit, and for the long
   positions the five worst ones. For this model the BF16 path is at 0.010 (median) and 0.08 (maximum) to
   position 127, and 0.016 and 0.53 on the long positions.
3. **Time**: per batch size (1 and 8), 64 warm-up steps, then 256 timed steps, three times; median microseconds
   per step. Logits from inside the timed run are checked like any others. Also timed: a 64-token prompt followed
   by 1024 greedy tokens with the chosen token read on the host at every step (`w1_like_step_us`), and
   `generate()` on both workloads when the engine has one.
4. **Trace** (batch 1, under CUPTI): per decode step, kernel launches, host synchronizations, runtime API calls,
   memcpys, the share of time the GPU ran nothing, and idle gaps of 20 us or more (the GPU waiting for the host).
5. **Serve and measure over HTTP: this is the score.** The evaluator starts the engine behind the harness's HTTP
   server (`eval/serve_engine.py`: an engine process and a front-end process that tokenizes, detokenizes and
   streams) and measures it with the client that measures the general engines (`eval/client.py steady`): prompt
   64, 1024 new tokens, greedy, one SSE chunk per token, batch 1 and batch 8, one warm-up and three repeats. The
   server drives the engine through `submit(rid, prompt_ids, max_new_tokens)` and `poll()` when the engine has
   both; otherwise through `reset()` and `step()`, one `step()` and an argmax per token. Gate `http_tokens`: the
   tokens the server streams for the check's prompts must be the engine's own step-by-step greedy tokens.

Result: `status` (`valid` only if every gate passes; performance numbers exist only for valid submissions), the
headline `http_step_us_b1` (microseconds per token at batch 1 over HTTP; `perf.b8.http_step_us` is the batch-8
figure), `perf.b1` / `perf.b8` with what was timed inside the engine's process (`step_us`, `w1_like_step_us`,
`tok_s`, `mbu`), the `trace` block, and `milestones_passed`. A `--quick` evaluation has no HTTP stage and is
never ranked.

**MBU** (memory bandwidth utilization) = bytes a step must read (all weights except the input embedding
table, plus the K and V of earlier positions) / step time / 3350 GB/s. Decoding at small batch is bounded by
reading the weights once per step; MBU says how close you are to that bound. For this model the weights are
about 15 GB, so 100% MBU would be about 4.5 ms per step; the general engines sit near 67%.

## The ladder

Climb in order. Each rung is a property the evaluator can see; `--milestone` names the rung a submission is
aiming at.

| Rung | Property | How it shows |
|---|---|---|
| `build` | compiles, loads | evaluator runs |
| `correct` | your own CUDA computes the model: operators, one layer, the stack, a decode step with KV cache; no torch module in the step | all gates pass |
| `single_launch` | a decode step is one kernel launch (or less): no library calls, no per-operator kernels | trace: launches per step <= 1.5 |
| `beats_mpk`, `parity`, `h0_step` | faster than MPK; than both general engines; than both by 10%, at B=1 and B=8 | speedup block |
| `device_loop` | `generate()` exists: argmax, token feedback and the loop run on the device and reproduce the step-by-step tokens | gate `generate`; `perf.b*.generate_step_us` is timed and can become the headline |
| `gpu_never_waits` | the GPU never waits for the host between steps (the host may still launch each step) | trace: host syncs and long idle gaps per step ~ 0 |

What has worked for others between `single_launch` and `h0_step` (ForgeMegakernel's milestones, MPK, Cohere's
megakernel, Hazy Research's "no bubbles"); treat these as hypotheses to test here, not as a recipe:

- one persistent kernel launched once, fed steps through memory, instead of a launch per step;
- the step expressed as a stream of typed instructions per SM (H100: 132 SMs), interpreted in the kernel, so
  that every SM always has work that reads memory;
- dependency counters between instructions instead of grid-wide barriers, so no SM waits at a barrier while
  memory sits unread;
- a shared pool of buffers / load balancing so that large matrix-vector products are split across SMs evenly;
- weights laid out and read so that each byte crosses HBM once per step (no re-reads, no padding reads).

**The sync rule.** The design principle of this project: nothing on the GPU blocks, and the only
synchronization is on the CPU side, off the per-token path. In practice kernels people write end up needing a
sync somewhere, and when the step is short that sync is what bounds the speed (on this GPU the general engines
show one host round trip per step once a step is under ~1.5 ms). Every sync point you leave in is a bet that
the step is long enough to hide it. Count yours (`stats()`), look at the trace block after every submission,
and say in the hypothesis when a change adds or removes one.

## Correctness is not negotiable

Performance numbers exist only for `valid` submissions. Engines that passed output-only checks have been
found to lose points on real tasks; that is why the check is against float64 at every position. Typical ways
to fail quietly: reading stale or out-of-bounds KV (poison freed or unwritten KV with NaN in your own tests),
a row's result depending on its neighbours in the batch (keep per-row arithmetic order independent of batch
composition), wrong position at a page or chunk boundary, reduced precision adopted for speed without saying
so. Test each operator and each layer against the Hugging Face implementation before you
submit; a submission is slow feedback, a unit test is fast.

## Logging: every step must be plottable

The run's value to people is the record of how the engine improved. The loop records numbers; only you can
record the reasons.

- **`--title`**: at most 80 characters, what changed, in plain words ("fuse RMSNorm into the attention entry").
- **`--hypothesis`**: one or two sentences written **before** you see the result: what you changed, why it
  should help, and the number you expect to move by how much ("removes 36 grid barriers; expect step_us_b1
  -3%"). A failed prediction is as valuable as a confirmed one; never rewrite it afterwards.
- **`--category`**: one of the listed categories; pick the mechanism, not the file.
- **`--parent`**: the submission this one was derived from, when it is not your previous one (after a revert,
  name the one you went back to).
- One idea per submission. Two changes at once cannot be told apart on the plot.
- After each result, add an entry to `notes/journal.md`: expected, measured, what you conclude, what next.
  When a result contradicts the hypothesis, find out why before moving on (profile; do not guess).
- `mk note` when something off the submission path matters (a dead end you abandoned, a tool that misled you,
  a decision to restructure).
- Commit your workspace to git with a message naming the submission id whenever a submission is valid.

## Working rhythm

Read the board, pick the largest remaining loss (profile to find it), state the hypothesis, make the smallest
change that tests it, check it with a test of your own, submit, record. Keep going until the time on the board runs out;
do not stop to ask for confirmation, nobody is watching the session live.
