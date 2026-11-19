---
name: megakernel
description: Reference for this run (Inkling-2-Small on one GB200 node (four NVIDIA B200)): the engine contract, the `mk` commands, how the evaluator scores and what its gates check, how to log, and the facts of the model, the card and the baselines. Read it before writing code and before every `mk submit`, and again whenever a result surprises you.
---

# Reference: a decode engine for Inkling-2-Small on one GB200 node (four NVIDIA B200)

This file holds what you cannot know without being told. What to study, what to build and in which order is yours
to decide (the run's first prompt states the goal, what you may do and what must stay the same).

## The contract

`engine/api.py` in your workspace defines `Engine` (the starter there is the model in plain PyTorch behind it, using all four cards in turn):

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
The specification of what to compute is the SGLang pull request's model code (see "The model" below); `$MK_HARNESS/eval/inkling2_ref.py` is the same function in plain PyTorch. The tensors `step()` takes and returns may live on any of the node's cards.

Not allowed (a submission that does any of these is void whatever its numbers): calling another inference
engine or library to run the model in the timed path (vLLM, SGLang, MPK, TensorRT, llama.cpp, another party's
megakernel, the engines' Inkling-2 operators or model code, torch.compile of the reference file); caching or hard-coding outputs; a path that skips
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
- **The whole node.** The pod is one GB200 node and the engine's target is all of it: every job (`mk exec`, `mk profile`, the
  sealed evaluation) sees **all four cards** (devices 0-3, `$MK_CARDS` = 4, `$MK_TARGET` = node) and the jobs run **one
  at a time** in the order sent (`--card` is ignored). `$MK_SERVER_CPUS` holds most of the node's cores. `mk gpu` shows
  every card's memory, utilization and job. A long job delays everything behind it, a submission included.
- **The model's files** are in the pod at `$MK_MODEL_DIR`: `config.json`, `hf_quant_config.json`, the tokenizer, and the
  safetensors shards listed in `model.safetensors.index.json` (393.5 GB; a shard's header lists every tensor's name, dtype and
  shape). The specification's source files are not in the pod's Python: fetch them from GitHub at the pinned commit (the pod
  has network access), or read `$MK_HARNESS/eval/inkling2_ref.py`. Any measurement on the real weights or on real
  activations is an `mk exec` away. `$MK_HARNESS/eval/weights_census.py` is one example of an exact measurement over every
  weight of a checkpoint (integer counts on the GPU, no sampling); it is an example of how to measure exactly, not a list of
  what is worth measuring.
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
2. **Check** against the high-precision reference on sequences you cannot see: every
   position's logits, alone and in a batch of 8, under teacher forcing, to position 127, and every 8th position
   to 1151 in the batch of 8. The reference for this model is **float64**:
   the reference is `eval/inkling2_ref.py` in its high-precision mode: the discrete steps of the model exactly
   as defined (the stored weight codes and scales, the NVFP4 / MXFP8 quantization of every expert GEMM's input,
   the top-8 expert choice) and everything between them in float64; the weights are not copied to float64 as a
   whole (594 B parameters), each expert is expanded when it is used.
   Gates: relative L2 error of the centred logits at most 2x the error a general engine's own path has against the
   reference (median and maximum over positions; see below which path); top-1 agreement with the reference within
   one point of that path's; all logits finite; the same sequence alone and inside a batch within the same
   tolerance (bitwise equality is recorded). The result's `gates` block has every number and its limit.
   For this model there is no Hugging Face BF16 path. The path whose error sets the tolerance is a general
   engine's own: SGLang (the pull request's image, the baseline below) teacher-forced on the reference's
   sequences, its final hidden states taken from the server and put through the unembedding in float32 and the
   soft cap (job <job>). Against the float64 reference it is at a relative L2 distance of
   0.089 (median over the first 128 positions; maximum 0.35) and picks the reference's top-1 token at 91.5% of
   them; at the long positions 0.062 / 0.45 and 98.6%. The limits are therefore: median at most 0.179, maximum
   at most 0.707, top-1 at least 90.5% (first 128 positions); 0.124 / 0.91 / 97.6% at the long positions. The
   harness's own path in the engines' precision (`eval/inkling2_ref.py` mode fork, the starter's arithmetic)
   measured 0.081 / 0.30 / 94.4% against the reference and 0.084 / 0.38 / 92.5% against SGLang: the three paths
   are about equally far from each other. Where the distance comes from was not measured for this model; for the
   earlier mixture-of-experts models it was mostly expert choices that flip at near-ties (a third of the (layer,
   token) pairs), and this model adds a second discrete step, the E2M1 rounding of every expert input, where a
   BF16 difference can move a value to the next 4-bit code.
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

**MBU** (memory bandwidth utilization) = bytes a step must read / step time / the node's peak bandwidth (the sum over its four cards)
(32000 GB/s). How the bytes are counted for this model: the experts are 376.1 GB of the 387.6 GB of the text decoder, and a token reads 8 of 384 per layer
(17.92 MB per NVFP4 expert with its scales, 32.85 MB per MXFP8 expert): 7.84 GB of experts plus 11.46 GB read in full
(9.61 GB of BF16 attention, router and convolution weights, 1.85 GB unembedding) = **19.29 GB per step at batch 1**,
plus K and V (40 layers see at most 1024 positions, 8 see all) and the short-convolution state. At batch 8, between
19.29 GB (all sequences choose the same experts) and 74.1 GB (all choose different ones). The figure is the model's,
for the whole node; how an engine splits it over the cards (and whether it reads anything twice) is not counted.
`bytes_per_step.how` in every result says how it was counted. Reading those bytes once at the node's peak rate
takes about 603 us at batch 1: no engine that reads every byte it needs in the checkpoint's format can
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

Microseconds per decode step, measured by this project on the same kind of node:

<job> (2026-11-12, attempt 1 after a preemption): SGLang 212.27 tok/s at B=1 (tpot p50 4.71
ms), 1211.46 tok/s at B=8 (tpot p50 6.57 ms), 3/3 clean repeats each. vLLM (vllm/vllm-openai:inkling-2-small
arm64, its recipe's single_node_tp flags) did not start in that job: every TP worker failed in load_model /
create_model.

| System | Kind | Batch 1 | Batch 8 |
|---|---|---|---|
| SGLang PR #50186 image lmsysorg/sglang:dev-inkling-2-small (arm64 sha256:198af079c599), cookbook GB200 flags: --tp 4 --ep 4 --moe-a2a-backend megamoe --attention-backend fa4 --bf16-gemm-backend torch --dtype bfloat16 --page-size 128, SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096; CUDA graphs on (default) | general engine | 4711 | 6604 |

The project's claim for a model is a speed of at least **1.10x** the fastest general engine at batch 1 and at
batch 8: 4283 us per token at batch 1 (4711 / 1.10) and 6004 us per step at batch 8 (6604 / 1.10).

These are trial numbers: a server end to end over HTTP on the client's clock (prompt 64, 1024 new tokens,
greedy). Your headline is measured the same way, by the same client, in the evaluation's own pod; the baselines
were measured in other pods on the same kind of node.

## Two results, kept apart

- **Track A** (aligned with the general engines): the same model, the same number formats they use (the expert
  weights as the checkpoint stores them, NVFP4 and MXFP8 with their scales; every expert GEMM's input quantized as
  the model defines; BF16 weights and activations and BF16 K/V elsewhere; float32 where the specification uses
  float32: router logits, attention scores and softmax, attention-residual mixing, the unembedding's product), the
  same data flow. Memory layout, how the work is split over the four cards and how the kernels are organised are
  free. This is the apples-to-apples number.
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

## The model: `thinkingmachines/Inkling-2-Small`

Read from the checkpoint's `config.json`, `hf_quant_config.json`, its safetensors headers and the specification's source.
Where this list and the source disagree, the source is right.

- **The specification.** There is no Hugging Face implementation of this model (transformers 5.22 and its main
  branch know only Inkling v1). The public definitions of the function are the engines' own model code. The run's
  specification is the SGLang pull request's model code: sgl-project/sglang pull request #50186 ("[Feat] inkling2: Support Inkling-2-Small") at commit 237ba6baf2f2 (`python/sglang/srt/models/inkling2.py`,
  `inkling_common/inkling2_*.py`, its operators under `python/sglang/kernels/ops/inkling2/`), together with the
  checkpoint's `hf_quant_config.json`. The harness has the same function in plain PyTorch, `$MK_HARNESS/eval/inkling2_ref.py`
  (it began as the transcription of that model code in NovaSky-AI/SkyRL pull request #2730): the starter is that file in the
  engines' precision, and the evaluator's reference is that file in high precision. Where the two disagree, the SGLang code
  decides; tell the run's notes if you find a disagreement.
- Multimodal checkpoint (text and image in, text out); **this run decodes text only**: the vision tower (0.66 GB) and
  `mtp.safetensors` (a 3.4 GB DSpark draft model for speculative decoding) are not part of the function.
- 594 B parameters, about 17 B active per token; **393.5 GB on disk** in 24 shards (1770 tensors), in the format the model
  was trained in. It fits only spread over the node's cards.
- Hidden size 4608, 48 layers, vocabulary 201024 rows of which the first **200058** are real tokens: `step()` returns
  logits for those 200058. Output: unembedding (`model.llm.unembed.weight`, BF16, product in FP32), then
  30 * tanh(z / 30). The input embedding is RMS-normalised (`embed_norm`).
- **No residual stream: block attention residuals.** The 48 layers form 8 blocks of 6. Inside a block the attention
  and MoE outputs are summed into a partial block; at a block boundary the partial block is kept as a finished
  block and the sum restarts. The token embedding is the first finished block. Every sublayer's input is a softmax
  over depth of the finished blocks and the partial block at the same position: one learned FP32 query per sublayer
  (three for the attention of the 8 global layers: separate inputs for Q, K and V), RMS-normalised sources as keys and
  values. The final norm reads the same way (`output_query`).
- **Attention**, every layer: 48 query heads, 8 K/V heads, head dimension 128; Q, K, V, gate and output
  projections BF16. Layers 5, 11, 17, 23, 29, 35, 41, 47 are global; the other 40 see the last 1024 positions.
  No RoPE: a learned relative bias is added to the scores, r = x W_r (16 per head), L2-clipped, times a [16, 1024]
  matrix, for distances 0-1023 (0 beyond). Q and K are L2-clipped per head (threshold 22.63; Q with a learned
  per-head scale), V is RMS-normalised without gain; in the global layers Q and r are multiplied by
  1 + 0.1 log(max(1, (position + 1) / 128000)). Each token predicts a per-head sink logit; the output is scaled by
  sigmoid(logsumexp(scores) - sink) and gated by sigmoid(x W_gate) per element, then projected.
- **Short convolutions** (kernel 4, depthwise, causal, identity residual, scaled by 1/sqrt(5)) on K and on V, on the
  attention output and on the MoE output of every layer. In the global layers the K and V taps are dynamic: a
  fixed learned tap plus a per-token, per-K/V-head tap predicted from the layer input (scaled by 1/sqrt(9)). So a
  sequence carries, per layer, its K/V and the last three inputs of four convolutions.
- **MoE**, every layer: router 4608 -> 384 (BF16 weights, FP32 output), sigmoid scores plus a learned FP32 bias
  select the top 8; their weights are the sigmoids of the chosen logits, normalised, times 8. No shared expert.
  Expert: gate and up projections stored interleaved as one [4608, 4608] matrix (even rows gate, odd rows up,
  intermediate size 2304), down projection [4608, 2304]; activation 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25).
- **Expert precision (what the checkpoint stores and what the model computes).** Layers 0-39: **NVFP4** weights,
  two E2M1 codes per byte (low nibble first), an E4M3 scale per 16 inputs, and per expert a float32
  `amax` / `outer_scale_denominator`. Layers 40-47: **MXFP8** weights, E4M3 with a UE8M0 scale per 32 inputs. The
  model was trained W4A4: `hf_quant_config.json` requires every expert GEMM's **input** to be quantized to the same
  format (NVFP4: per-token amax, scale 1536 / amax, each block of 16 scaled to 6 and to 4 and the lower-error one
  kept ("four over six"), E2M1 round-to-nearest-even; MXFP8: blocks of 32, UE8M0 scale rounded up, E4M3). That
  quantization is part of the function: an engine that skips it computes another model. Everything else is BF16
  (attention, router and short-convolution weights 9.6 GB) or F32 (attention-residual queries, router bias).
- Both general engines run attention tensor-parallel and the experts expert-parallel over the four cards (vLLM's
  recipe: TP4 attention with EP4 experts); that is how they split it, not a rule.

## The node

- The engine's target is **one whole GB200 node: four NVIDIA B200** (`nvidia-smi` names each "NVIDIA GB200",
  189471 MiB each, 740 GiB = 795 GB together), driver 595.71.05, CUDA 13.2. Ask the devices for what is not listed
  here (compute capability, number of SMs, cache sizes). The engine owns all four cards; every evaluation and
  every reported number is for the node.
- **Between the cards** (measured on these nodes, 2026-11-12): `nvidia-smi topo -m` shows NV18 between every pair
  (18 NVLinks per card, each reported at 53.125 GB/s, the far end of every link is an NVLink switch); peer access
  (`cudaDeviceCanAccessPeer`) is available for all 12 ordered pairs, with NVLink atomics; `cuMemcpyPeer` of 2 GiB
  runs at 777-778 GB/s for every ordered pair (job `<job>`). From inside running kernels
  (job `<job>`, `scripts/ink/p2p_probe.cu` in the harness): one thread on card A writes a flag
  into card B's memory through a peer pointer and one thread on B answers the same way: **1.24-1.37 us one way**
  (card 0 with 1, 2, 3); the same flag exchange between two kernels on one card: 0.55 us. All four cards reading
  256 MiB from the other three at the same time: 1895 GB/s in total. A kernel reading its own card's memory:
  6445-6458 GB/s per card.
- The host is **aarch64**: two Grace CPUs (Neoverse-V2), 144 cores, no SMT; cards 0 and 1 are on NUMA node 0
  (cores 0-71), cards 2 and 3 on NUMA node 1 (cores 72-143). Host code and build scripts must build on Arm. The pod
  is the whole node: nothing else runs on it.
- Peak memory bandwidth used for MBU: **32000 GB/s** for the node (4 x 8000 GB/s; the per-card figure is derived
  from the vendor's system numbers, not published per card and not measured; our read kernel above reaches about
  6450 GB/s per card). Treat MBU as indicative.
