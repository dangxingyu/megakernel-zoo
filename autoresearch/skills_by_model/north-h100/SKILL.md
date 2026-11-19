---
name: megakernel
description: Reference for this run (North Mini Code 1.0 on one NVIDIA H100 80GB): the engine contract, the `mk` commands, how the evaluator scores and what its gates check, how to log, and the facts of the model, the card and the baselines. Read it before writing code and before every `mk submit`, and again whenever a result surprises you.
---

# Reference: a decode engine for North Mini Code 1.0 on one NVIDIA H100 80GB

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
The Hugging Face implementation of the model (Cohere2MoeForCausalLM) is the specification of what to compute.

Not allowed (a submission that does any of these is void whatever its numbers): calling another inference
engine or library to run the model in the timed path (vLLM, SGLang, MPK, TensorRT, llama.cpp, another party's
megakernel, torch modules or torch.compile of the HF model); caching or hard-coding outputs; a path that skips
work when it recognizes the evaluator's inputs; reading the evaluator's reference data. Using torch tensors to
pass buffers, and CUTLASS / ThunderKittens-style header code you compile into your own kernel, is fine.

## Your tools

Everything that needs the GPU or real compute runs on the pod, never on the machine you are on (a login node
with no GPU for you; keep local commands light: editing, reading, git).

- `mk exec [--timeout S] -- CMD` runs CMD in the pod's copy of your workspace: builds, tests, measurements.
  The pod has the CUDA 13.2 toolkit (`nvcc`, headers), `nsys`, `ncu`, and `python` (also `$MK_PY`) with torch,
  transformers, safetensors, numpy and pytest. Build products stay in the pod between calls but are lost when
  the pod is replaced (`engine/build.sh` has to be able to rebuild everything). The command's output comes back
  to you (64 kB at most); files written to `$MK_EXEC_OUT` (2 MB at most in total) come back to
  `../execs/EID/out/` next to your workspace.
- **Several jobs at once.** The pod has 8 cards and runs up to one job per card. A job sees only its
  card (as device 0; `$MK_CARD` says which one it is) and is pinned to CPU cores of its own, so jobs do not share
  a card or cores. Each `mk exec` blocks until its command ends: start several from background shells, or use
  `mk exec --detach` and `mk wait EID`. Every card has its **own copy of the workspace** and its own build
  products (`mk exec --card N` asks for a card, to reuse what was built there; otherwise the lowest free card is
  taken). `mk gpu` and `mk status` show every card and what runs on it. A sealed evaluation (`mk submit`) always
  runs **alone**: it starts when the running jobs have ended, and nothing else starts until it is done, so that
  its timings are those of a quiet pod. A long detached job therefore delays a submission.
- **The model's files** are in the pod at `$MK_MODEL_DIR`: `config.json`, the tokenizer, and the safetensors
  shards listed in `model.safetensors.index.json` (the real weights; a shard's header lists every tensor's name,
  dtype and shape). The Hugging Face implementation is in the pod's Python:
  `python -c "import transformers, os; print(os.path.dirname(transformers.__file__))"`, then
  `models/cohere2_moe/modeling_cohere2_moe.py`; copy it to `$MK_EXEC_OUT` to read it here. Any measurement on the real
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

The pod can be replaced at any moment (a pause, then a fresh pod; a pod lives at most six hours). Nothing in it
is durable. Your workspace, your notes and git are.

## What the evaluator does with a submission

1. **Build** (`engine/build.sh`).
2. **Check** against a high-precision run of the Hugging Face model on sequences you cannot see: every
   position's logits, alone and in a batch of 8, under teacher forcing, to position 127, and every 8th position
   to 1151 in the batch of 8. The reference for this model is **float32**:
   neither a float64 (244 GB) nor a float32 copy (122 GB) fits an H100, so the reference was made in float32 on
   a B200; a reference is only token ids and logits and does not depend on the card.
   Gates: relative L2 error of the centred logits at most 2x the error the Hugging Face BF16 path itself has
   against the reference (median and maximum over positions); top-1 agreement with the reference within one
   point of Hugging Face BF16's; all logits finite; the same sequence alone and inside a batch within the same
   tolerance (bitwise equality is recorded). The result's `gates` block has every number and its limit.
   For this model the Hugging Face BF16 path is itself at a relative L2 distance of 0.076 (median over
   positions; maximum 1.08) from the reference and picks the reference's top-1 token at 93.8% of the positions
   (for Qwen3-8B the same figures are 0.010, 0.08 and 95.1%). The limits are therefore: median error at most
   0.153, maximum at most 2.15, top-1 agreement at least 92.8%. The starter (the same Hugging Face model stepped
   token by token) measured 0.081 / 1.08 and 93.9%. Where that distance comes from was measured once (job
   <job>, the same 8 sequences of 1152 tokens, teacher forced): BF16 and the float32
   reference pick a different set of 8 experts in 37% of all (layer, token) pairs (5% in the first
   mixture-of-experts layer, about 50% in the middle of the stack; 97% of the tokens differ in at least one
   layer, the median token in 17 of 48; where the sets differ, 7 of the 8 experts are usually the same). The
   reference's own margin between its 8th and 9th router logit is 0.077 in the median and below 0.03 for a
   quarter of the pairs. With the expert choices forced to be the same, BF16 is at 0.012 to 0.026 (median) from
   the reference and agrees on the top-1 token at 99.6% of the positions; with its own choices, on the same
   positions, at 0.109 (maximum 1.0) and 98.6%. The gate does not separate the two effects: it is the same rule
   as for every other model.
3. **Time**: per batch size (1 and 8), 64 warm-up steps, then 256 timed steps, three times; median microseconds
   per step. Logits from inside the timed run are checked like any others. Also timed: a 64-token prompt followed
   by 1024 greedy tokens with the chosen token read on the host at every step (`w1_like_step_us`: the workload the
   baselines were measured on, without HTTP), and `generate()` on both workloads when the engine has one.
4. **Trace** (batch 1, under CUPTI): per decode step, kernel launches, host synchronizations, runtime API
   calls, memcpys, the share of time the GPU ran nothing, and idle gaps of 20 us or more.
5. **Serve and measure over HTTP: this is the score.** The evaluator starts the engine behind the harness's HTTP
   server (`$MK_HARNESS/eval/serve_engine.py`: an engine process, in the pod's Python with the workspace as
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
`<job>`): `valid`, 38159 us per step at batch 1 and 38314 us at batch 8, timed in
process (that evaluation had stages 1 to 4 only). It took 12 minutes, most of it loading the model three times
and stepping through PyTorch; stage 5 adds a fourth load and about 8700 steps. An engine that loads and steps
faster is evaluated faster.

**MBU** (memory bandwidth utilization) = bytes a step must read / step time / the card's peak bandwidth
(3350 GB/s). How the bytes are counted for this model: the experts are 58.0 GB of the 61.0 GB, and a token reads 8 of 128 per layer: about 3.62 GB of experts plus
2.99 GB of everything else (attention, the dense layer, routers, and the 1.07 GB table as the tied head) =
**6.61 GB per step at batch 1**. At batch 8 the sequences may share experts or not: between 6.61 GB (all route
alike) and 32.0 GB (all route apart); the result reports MBU for both (`mbu`, `mbu_high`).
`bytes_per_step.how` in every result says how it was counted. Reading those bytes once at the card's peak rate
takes about 1973 us at batch 1: no engine that reads every byte it needs in the checkpoint's format can
be faster than that, and MBU says how close a step is to it.

`milestones_passed` lists properties the evaluator saw in a submission. They describe the result; they are not
a path to follow, and `--milestone` only records which one a submission was aiming at, if any.

| Name | Property | Seen in |
|---|---|---|
| `build` | compiles, loads | the evaluation runs |
| `correct` | every gate passes | `gates` |
| `single_launch` | a decode step is at most 1.5 kernel launches | `trace` |
| `parity`, `h0_step` | over HTTP, at least as fast as every general engine; faster than each by 10%; at batch 1 and batch 8 | `speedup` block |
| `beats_cohere` | over HTTP, at least as fast as Cohere's megakernel at batch 1 and batch 8 | `speedup` block |
| `device_loop` | `generate()` exists and reproduces the step-by-step greedy tokens (in-process diagnostic) | gate `generate` |
| `gpu_never_waits` | host synchronizations and idle gaps of 20 us or more per step are about zero | `trace` |

## The baselines

Microseconds per decode step, measured by this project on the same kind of card:

<job> (Cohere's server: 276.6 tok/s at B=1, 1114.9 at B=8) and
<job> (vLLM: 198.1 / 862.3), 2026-11-10. Cohere's server streams only on its chat
endpoint: its requests carry the chat template (180 prompt tokens instead of 64) and at B=8 it sends several
tokens per chunk. vLLM needed --max-model-len 32768 (the model's own 500000 does not leave room for the KV cache
next to 61 GB of weights). SGLang does not list this model. Earlier bring-up trial <job>
(512 tokens): 286.2 / 1113.3 tok/s; Cohere's own decode-only runner: 287 / 977.

| System | Kind | Batch 1 | Batch 8 |
|---|---|---|---|
| vLLM 0.32.0, default flags plus --max-model-len 32768 | general engine | 5049 | 9277 |
| Cohere's megakernel @ 67d0b9ca, its own server at batch size 8 | megakernel written for this model | 3616 | 7176 |

The project's claim for a model is a speed of at least **1.10x** the fastest general engine at batch 1 and at
batch 8: 4590 us per token at batch 1 (5049 / 1.10) and 8434 us per step at batch 8 (9277 / 1.10).

These are trial numbers: a server end to end over HTTP on the client's clock (prompt 64, 1024 new tokens,
greedy). Your headline is measured the same way, by the same client, in the evaluation's own pod; the baselines
were measured in other pods on the same kind of card.

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

## The model: `CohereLabs/North-Mini-Code-1.0`

Read from the checkpoint's `config.json`, its safetensors headers and the Hugging Face source (transformers
5.22). The source in the pod is the specification; where this list and the source disagree, the source is right.

- Text-only decoder, 30.48 B parameters, all BF16, 61.0 GB in 49 safetensors shards: it takes 61 of the
  card's 85.5 GB. Hidden size 2048, 49 layers, vocabulary 262144. The output head is **tied** to the embedding:
  the checkpoint stores only `model.embed_tokens.weight` (1.07 GB), which is read in full as the head.
- **Parallel block**: a layer has one RMSNorm; attention and MLP both read the normed input and both are added
  to the residual (`x + attn(norm(x)) + mlp(norm(x))`).
- **Attention**: 32 heads, 4 K/V heads, head dimension 128, no Q/K norm. 36 layers are sliding-window attention
  (window 4096, which is longer than the 2048-token contexts of this evaluation) with RoPE (theta 50000); 13 are
  full attention (layers 0, 4, 8, ..., 48) without positional encoding, except layer 0, which has RoPE.
- **MLP**: layer 0 is dense SwiGLU (intermediate 3072). The other 48 layers are a mixture of experts: a router
  (linear 2048 -> 128 **in the model's dtype**, the top 8 logits kept, each weighted by its sigmoid, no
  renormalisation), 128 experts, each a SwiGLU MLP with intermediate size 768 (three matrices per expert in the
  checkpoint), no shared expert.
- Which experts a token uses is decided inside each layer from that layer's input. A discrete choice like this
  can differ between two correct implementations at a near-tie; the gates are relative to what the Hugging Face
  BF16 path itself does against the reference (see "What the evaluator does").
- **A human-written megakernel exists for exactly this model on this card**: Cohere's
  (`cohere-ai/cohere-megakernel`, commit 67d0b9ca; C++ host loop, one H100, batch 1 to 8). It is on the board as a
  baseline next to vLLM, and it is the most interesting comparison this project has: an engine written for this
  one checkpoint by the people who trained it. Its source is public and may be read; its code may not run in
  your timed path.

## The card

- The engine's target is **one NVIDIA H100 80GB HBM3** (81559 MiB, 132 SMs, driver 595.71, CUDA 13.2).
- The pod is a whole node with **eight** such cards. The other cards are for running several experiments at the
  same time (see "Your tools"); the engine, every evaluation and every reported number are for one card.
- The host is x86-64 (Xeon Platinum 8462Y+, 64 cores, 128 hardware threads). The pod is the whole node: nothing else runs on it.
- Peak memory bandwidth used for MBU: **3350 GB/s** (the vendor figure ForgeMegakernel uses; our own read-rate
  microbenchmark reaches 3085 GB/s on this card).
- 8-bit weights or activations (FP8) are a separate, later setting: this run is BF16.
