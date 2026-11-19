---
name: megakernel
description: Reference for this run (Qwen3.8-27B on one NVIDIA B200 (a card of a GB200 node)): the engine contract, the `mk` commands, how the evaluator scores and what its gates check, how to log, and the facts of the model, the card and the baselines. Read it before writing code and before every `mk submit`, and again whenever a result surprises you.
---

# Reference: a decode engine for Qwen3.8-27B on one NVIDIA B200 (a card of a GB200 node)

This file holds what you cannot know without being told. What to study, what to build and in which order is yours
to decide (the run's first prompt states the goal, what you may do and what must stay the same).

## The contract

`engine/api.py` in your workspace defines `Engine` (the starter there is the Hugging Face model behind it):

```python
Engine(model_dir: str, max_batch: int, max_context: int)
reset()                       # forget every sequence
step(tokens)                  # int64 cuda tensor [B]: next input token of each sequence (B fixed between resets);
                              # returns next-token logits, cuda tensor [B, vocab], bf16 or float32
close(), stats()              # optional; stats() is a dict of your own counters, stored with the result
generate(tokens, n_steps)     # optional: greedy decode inside the engine (argmax, feedback and the loop on the
                              # device). tokens [B] is the next input token; returns int64 cuda [B, n_steps]
submit(rid, prompt, n)        # optional, used by the HTTP server of the evaluation when both exist: a request
poll()                        # arrives (int id, list of prompt token ids, number of new tokens; returns at once);
                              # poll() -> [(rid, new_tokens, finished), ...] since the last poll. Greedy, exactly
                              # n tokens per request; admission and batching are the engine's
```

`engine/build.sh` is run before every evaluation (cwd = workspace). Everything else under `engine/` is yours.
The Hugging Face implementation of the model (Qwen3_5ForCausalLM) is the specification of what to compute.

Not allowed (a submission that does any of these is void whatever its numbers): calling another inference
engine or library to run the model in the timed path (vLLM, SGLang, MPK, TensorRT, llama.cpp, another party's
megakernel, torch modules or torch.compile of the HF model); caching or hard-coding outputs; a path that skips
work when it recognizes the evaluator's inputs; reading the evaluator's reference data. Using torch tensors to
pass buffers, and CUTLASS / ThunderKittens-style header code you compile into your own kernel, is fine.

## Your tools

Everything that needs the GPU or real compute runs on the machine, never on the machine you are on (a login node
with no GPU for you; keep local commands light: editing, reading, git).

- `mk exec [--timeout S] -- CMD` runs CMD in the machine's copy of your workspace: builds, tests, measurements.
  The machine has the CUDA 13.2 toolkit (`nvcc`, headers), `nsys`, `ncu`, and `python` (also `$MK_PY`) with torch,
  transformers, safetensors, numpy and pytest. Build products stay in the machine between calls but are lost when
  the machine is replaced (`engine/build.sh` has to be able to rebuild everything). The command's output comes back
  to you (64 kB at most); files written to `$MK_EXEC_OUT` (2 MB at most in total) come back to
  `../execs/EID/out/` next to your workspace.
- **Several jobs at once.** The machine has 4 cards and runs up to one job per card. A job sees only its
  card (as device 0; `$MK_CARD` says which one it is) and is pinned to CPU cores of its own, so jobs do not share
  a card or cores. Each `mk exec` blocks until its command ends: start several from background shells, or use
  `mk exec --detach` and `mk wait EID`. Every card has its **own copy of the workspace** and its own build
  products (`mk exec --card N` asks for a card, to reuse what was built there; otherwise the lowest free card is
  taken). `mk gpu` and `mk status` show every card and what runs on it. A sealed evaluation (`mk submit`) always
  runs **alone**: it starts when the running jobs have ended, and nothing else starts until it is done, so that
  its timings are those of a quiet machine. A long detached job therefore delays a submission.
- **The model's files** are in the machine at `$MK_MODEL_DIR`: `config.json`, the tokenizer, and the safetensors
  shards listed in `model.safetensors.index.json` (the real weights; a shard's header lists every tensor's name,
  dtype and shape). The Hugging Face implementation is in the machine's Python:
  `python -c "import transformers, os; print(os.path.dirname(transformers.__file__))"`, then
  `models/qwen3_5/modeling_qwen3_5.py`; copy it to `$MK_EXEC_OUT` to read it here. Any measurement on the real
  weights or on real activations is an `mk exec` away. `$MK_HARNESS/eval/weights_census.py` is one example of
  an exact measurement over every weight of a checkpoint (integer counts on the GPU, no sampling); it is an
  example of how to measure exactly, not a list of what is worth measuring.
- `mk profile nsys -- CMD` (and `ncu` if `mk` lists it) returns a summary: kernels by GPU time, GPU idle share,
  idle gaps of 20 us or more, host synchronizations.
- `mk submit --title ... --hypothesis ... --category ... [--parent SID] [--milestone RUNG]` sends `engine/` for
  the sealed evaluation and returns its result. Only this counts. `mk submit` waits for the result; `mk wait SID`,
  `mk status SID`, `mk log SID` are there if it returns early. `--quick` runs a shorter evaluation that is logged
  but never ranked. `engine/` must be source only (no file over 2 MB, 10 MB in total).
- `mk board` shows every submission, the best so far, the reference lines and the time left.
- `mk note "..."` puts a line on the run's timeline. `mk gpu` shows every card's memory, utilization and job.

The machine can be replaced at any moment (a pause, then a fresh machine; a machine lives at most six hours). Nothing in it
is durable. Your workspace, your notes and git are.

## What the evaluator does with a submission

1. **Build** (`engine/build.sh`).
2. **Check** against a high-precision run of the Hugging Face model on sequences you cannot see: every
   position's logits, alone and in a batch of 8, under teacher forcing, to position 127, and every 8th position
   to 1151 in the batch of 8. The reference for this model is **float32**:
   the float64 copy of the text decoder would be 215 GB and the card has 199 GB; float32 (108 GB) still keeps 24
   significant bits where BF16 keeps 8.
   Gates: relative L2 error of the centred logits at most 2x the error the Hugging Face BF16 path itself has
   against the reference (median and maximum over positions); top-1 agreement with the reference within one
   point of Hugging Face BF16's; all logits finite; the same sequence alone and inside a batch within the same
   tolerance (bitwise equality is recorded). The result's `gates` block has every number and its limit.
   For this model the Hugging Face BF16 path is itself at a relative L2 distance of 0.010 (median over
   positions; maximum 0.19) from the reference and picks the reference's top-1 token at 97.1% of the positions
   (for Qwen3-8B the same figures are 0.010, 0.08 and 95.1%). The limits are therefore: median error at most
   0.020, maximum at most 0.38, top-1 agreement at least 96.1%. The starter (the same Hugging Face model stepped
   token by token) measured 0.010 / 0.07 and 97.0%.
3. **Time**: per batch size (1 and 8), 64 warm-up steps, then 256 timed steps, three times; median microseconds
   per step. Logits from inside the timed run are checked like any others. Also timed: a 64-token prompt followed
   by 1024 greedy tokens with the chosen token read on the host at every step (`w1_like_step_us`: the workload the
   baselines were measured on, without HTTP), and `generate()` on both workloads when the engine has one.
4. **Trace** (batch 1, under CUPTI): per decode step, kernel launches, host synchronizations, runtime API
   calls, memcpys, the share of time the GPU ran nothing, and idle gaps of 20 us or more.
5. **Serve and measure over HTTP: this is the score.** The evaluator starts the engine behind the harness's HTTP
   server (`$MK_HARNESS/eval/serve_engine.py`: an engine process, in the machine's Python with the workspace as
   working directory, and a front-end process that tokenizes, detokenizes and streams; a Unix socket between
   them) and measures it with the client that measured the baselines (`$MK_HARNESS/eval/client.py steady`):
   prompt 64, 1024 new tokens, greedy, one SSE chunk per token, batch 1 and batch 8, one warm-up and three
   repeats, server and client pinned to separate cores. Time per step = 1e6 * B / (decode tokens per second on
   the client's clock): the same quantity, client and workload as every baseline on the board.
   The server drives the engine through `submit(rid, prompt_ids, max_new_tokens)` (returns at once) and
   `poll() -> [(rid, new_tokens, finished), ...]` when the engine has both methods; otherwise through `reset()`
   and `step()`: requests that arrive together with the same prompt length form a batch, the prompt goes through
   `step()`, then one `step()` + argmax per token, with the chosen tokens copied to the host and sent at every
   step. `generate()` is not called by the server.
   Gate `http_tokens`: the tokens the server streams for the check's prompts (all eight at once) must be the
   engine's own step-by-step greedy tokens from stage 2 (mean matching prefix at least 0.75, at least half of the
   sequences identical).

Result: `status` (`valid` only if every gate passes; performance numbers exist only for valid submissions), the
headline `http_step_us_b1` (microseconds per token at batch 1 over HTTP; `perf.b8.http_step_us` is the batch-8
figure), the `http` block with the client's numbers, `speedup` against every baseline (HTTP figure against HTTP
figure), and `milestones_passed`. `perf.b1` / `perf.b8` also hold what was timed inside the engine's process
(`step_us`, `generate_step_us`, `w1_like_*`, `mbu`, `bytes_per_step`) and there is the `trace` block: these are
diagnostics, not the score. The in-process figure on the score's own workload is `w1_like_step_us` (prompt 64,
then 1024 greedy tokens, context up to 1088, the token read on the host at every step); `step_us` is timed at
contexts 64 to 320 without a host read. In the first evaluations of this kind (two engines on an H100, one on a
B200) `http_step_us` was within 6 to 11 us of `w1_like_step_us` at batch 1. A `--quick` evaluation has no HTTP
stage.

The starter in `engine/` (the Hugging Face model behind the contract) was evaluated before the run started (job
`<job>`): `valid`, 84536 us per step at batch 1 and 86372 us at batch 8, timed in process
(that evaluation had stages 1 to 4 only). It took 17 minutes, most of it loading the model three times and
stepping through PyTorch; stage 5 adds a fourth load and about 8700 steps. An engine that loads and steps faster
is evaluated faster.

**MBU** (memory bandwidth utilization) = bytes a step must read / step time / the card's peak bandwidth
(8000 GB/s). How the bytes are counted for this model: every text-decoder weight except the input embedding table: **51.25 GB per step**, plus the K and V of the
16 attention layers and the float32 state of the 48 recurrent layers (154 MB per sequence).
`bytes_per_step.how` in every result says how it was counted. Reading those bytes once at the card's peak rate
takes about 6406 us at batch 1: no engine that reads every byte it needs in the checkpoint's format can
be faster than that, and MBU says how close a step is to it.

`milestones_passed` lists properties the evaluator saw in a submission. They describe the result; they are not
a path to follow, and `--milestone` only records which one a submission was aiming at, if any.

| Name | Property | Seen in |
|---|---|---|
| `build` | compiles, loads | the evaluation runs |
| `correct` | every gate passes | `gates` |
| `single_launch` | a decode step is at most 1.5 kernel launches | `trace` |
| `parity`, `h0_step` | over HTTP, at least as fast as every general engine; faster than each by 10%; at batch 1 and batch 8 | `speedup` block |
| `device_loop` | `generate()` exists and reproduces the step-by-step greedy tokens (in-process diagnostic) | gate `generate` |
| `gpu_never_waits` | host synchronizations and idle gaps of 20 us or more per step are about zero | `trace` |

## The baselines

Microseconds per decode step, measured by this project on the same kind of card:

trials of 2026-11-10: <job> (105.0 tok/s at B=1, 781.9 at B=8),
<job> (98.9 / 747.3). SGLang with torch.compile (the strongest configuration on
Qwen3-8B/H100) was not measured for this model.

| System | Kind | Batch 1 | Batch 8 |
|---|---|---|---|
| SGLang 0.5.23, default flags (CUDA graphs) | general engine | 9523 | 10231 |
| vLLM 0.32.0, default flags | general engine | 10110 | 10705 |

The project's claim for a model is a speed of at least **1.10x** the fastest general engine at batch 1 and at
batch 8: 8657 us per token at batch 1 (9523 / 1.10) and 9301 us per step at batch 8 (10231 / 1.10).

These are trial numbers: a server end to end over HTTP on the client's clock (prompt 64, 1024 new tokens,
greedy). Your headline is measured the same way, by the same client, in the evaluation's own machine; the baselines
were measured in other machines on the same kind of card.

## Two results, kept apart

- **Track A** (aligned with the general engines): the same model, the same number formats they use (the BF16
  weights as the checkpoint stores them, BF16 activations and K/V where they use BF16, float32 where the
  Hugging Face path uses float32), the same data flow. Memory layout and how the kernels are organised are free.
  This is the apples-to-apples number.
- **Track B** (model-specific): anything lossless in addition: any change that leaves every result identical in exact arithmetic.

Every submission's hypothesis starts with "track A" or "track B" and one line saying which formats differ from
the checkpoint's. The best track-A engine stays buildable (a build switch is enough) so that it can be measured
again at any time. A storage change is claimed only after it has been verified bit for bit on **all** real
weights of the checkpoint (`$MK_HARNESS/eval/weights_roundtrip.py` is how this was done for an earlier engine's
packer; write the check your own format needs).

## Earlier runs

Other runs of this project, on this model and on others, are open to you in `runs/`
(start with its `README.md`): for every run the list of submissions (what was tried, why, what it measured, on
which track), the source of every submission and the run's own notes. Read them and take what helps. What worked
for another model or setting is a hypothesis for this checkpoint, not a fact: measure it here, and verify anything
you take over as strictly as your own work.

## Logging

The run's value to other people is the record of what was found and how the engine changed. The loop records
numbers; only you can record the reasons.

- **`--title`**: at most 80 characters, what changed, in plain words.
- **`--hypothesis`**: starts with the track (see above), then one or two sentences written **before** you see
  the result: what you changed, why it should help, and the number you expect to move by how much. A failed
  prediction is as valuable as a confirmed one; it is never rewritten afterwards.
- **`--category`**: one of the listed categories (`mk help`); the mechanism, not the file.
- **`--parent`**: the submission this one was derived from, when it is not your previous one.
- One idea per submission: two changes at once cannot be told apart afterwards.
- `notes/journal.md`: after each result, what was expected, what was measured, what you conclude.
- What you learn about the checkpoint itself goes into `notes/` as well, each finding with the measurement behind
  it (what was measured, on what, the number) and what it allows and what it does not.
- `mk note` when something off the submission path matters (a dead end, a tool that misled you, a decision).
- Commit your workspace to git, naming the submission id, whenever a submission is valid.

## The model: `Qwen/Qwen3.8-27B`

Read from the checkpoint's `config.json`, its safetensors headers and the Hugging Face source (transformers
5.22). The source in the machine is the specification; where this list and the source disagree, the source is right.

- The checkpoint is multimodal (`Qwen3_5ForConditionalGeneration`, 27.78 B parameters, 55.6 GB, 18 shards, all
  BF16). **This run decodes text only**: the text decoder is `model.language_model.*` plus `lm_head.weight`
  (26.90 B parameters; Hugging Face loads exactly these as `Qwen3_5ForCausalLM`). `model.visual.*` (0.92 GB) and
  the multi-token-prediction head `mtp.*` (0.85 GB) are not part of the function.
- Hidden size 5120, 64 layers, vocabulary 248320, output head not tied (2.54 GB), SwiGLU MLP with intermediate
  17408 in every layer. RMSNorm weights are stored centred (the norm multiplies by `1 + weight`).
- **Two kinds of token mixer** (`layer_types`): every fourth layer (3, 7, ..., 63: 16 layers) is full attention;
  the other 48 are "linear attention", a gated delta-rule recurrence with a fixed-size state and no K/V cache.
- Full attention: 24 heads, 4 K/V heads, head dimension 256, RMSNorm on Q and K per head, RoPE on the first
  quarter of the head dimension (theta 1e7), and an output gate: `q_proj` has twice the
  query width, the second half goes through a sigmoid and multiplies the attention output before `o_proj`.
- Linear attention (`linear_attn.*`): `in_proj_qkv` 5120 -> 10240 (Q 2048, K 2048, V 6144), a depthwise causal
  convolution over the last 4 inputs followed by SiLU, 16 key heads and 48 value heads of dimension 128
  (Q and K repeated 3x across value heads), Q and K L2-normalised, per-head decay `-exp(A_log) * softplus(a + dt_bias)`
  and write strength `sigmoid(b)`, a state of [48, 128, 128] per layer and sequence updated by the delta rule
  **in float32**, the output normalised by an RMSNorm gated with `silu(z)` and projected by `out_proj` 6144 -> 5120.
  A single-token step uses the recurrent form (`torch_recurrent_gated_delta_rule`); a multi-token pass uses a
  chunked form of the same recurrence.

## The card

- The engine's target is **one NVIDIA B200**; `nvidia-smi` names it "NVIDIA GB200", 189471 MiB, driver 595.71,
  CUDA 13.2. Ask the device for what is not listed here (compute capability, number of SMs, cache sizes).
- The machine is a whole GB200 node with **four** such cards. The other cards are for running several experiments at
  the same time (see "Your tools"); the engine, every evaluation and every reported number are for one card.
- The host is **aarch64**: two Grace CPUs (Neoverse-V2), 144 cores, no SMT. Host code and build scripts must build
  on Arm; nothing x86-only works. The machine is the whole node: nothing else runs on it.
- Peak memory bandwidth used for MBU: **8000 GB/s**. This figure is derived from the vendor's system numbers
  (two cards: 16 TB/s), not published per card and not measured by us. Treat MBU on this card as indicative.
- The card also has FP8 and MXFP8 paths. 8-bit weights or activations are a separate, later setting: this run is BF16.
