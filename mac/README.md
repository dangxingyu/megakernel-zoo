# megakernel-zoo on Apple silicon

One engine so far: [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26 B,
mixture of experts, NVFP4 experts, a DSpark draft head) on an **Apple M5 Max**, written by a coding agent (Claude) in
two six-hour runs against the repository's evaluator, in MLX and run-time Metal. It is a track-C engine: exact
speculative decoding. It drafts tokens with the checkpoint's own draft head and with n-gram copies, verifies them in
one pass, and returns exactly the tokens its own one-token step would.

Write-up: [`runs-mac/report-20261123/inkling-2-mini-m5max.html`](../runs-mac/report-20261123/inkling-2-mini-m5max.html)
(English and 中文). The blog post it follows: [A Megakernel per Model](https://dangxingyu.github.io/megakernel/).

## Speed

One sequence, greedy, over HTTP, a 64-token prompt and 1,024 new tokens (the repository's workload), on an M5 Max with
a 40-core GPU and 48 GB. Milliseconds per token; tokens per second in brackets.

| System | Weights | Synthetic prompts | Natural text |
|---|---|---|---|
| **megakernel, track C** | the checkpoint's NVFP4 / MXFP8 / BF16 | **8.10 (123)** | **5.14 (194)** |
| oMLX + DSpark | oQ4e requantization | 13.77 (73) | 8.31 (120) |
| oMLX | oQ4e requantization | 16.74 (60) | 16.78 (60) |
| llama.cpp + DSpark | Q4_K_M GGUF | 20.56 (49) | 16.32 (61) |
| llama.cpp | Q4_K_M GGUF | 25.13 (40) | 25.03 (40) |
| mlx-lm | the checkpoint's formats | 30.36 (33) | 30.49 (33) |
| Ollama 0.40.17 (MLX) | the checkpoint's formats | 31.29 (32) | 30.75 (33) |

That is 1.70× and 1.62× the fastest general engine we found for this model on a Mac, 3.7× mlx-lm on the same weights
and 3.1× llama.cpp.

- **Ours** is the median of six evaluations of the engine as the second run left it (8,040–8,333 and 5,124–5,300
  microseconds). The source in this repository, with everything below added, was evaluated once more in full: valid,
  the same gate values, 8,205 and 5,173 (`runs-mac/final-eval-20261123`).
- **The general engines** are two server starts × three prompts each; Ollama is one start. Ollama could not load the
  GGUF or the DSpark draft, and LM Studio was not measured. Scripts: [`baselines/mac`](../baselines/mac).
- **Laptop timing is noisy.** The same source spreads by up to 10% with the machine's state. Treat differences under
  5% as noise.

## Install

Needs an Apple-silicon Mac with about 36 GB of memory or more, Python 3.10 or newer and MLX 0.32 or newer. Tested on an
M5 Max with 48 GB, macOS 26, Python 3.12 and MLX 0.32.4. No Xcode and no CUDA: the Metal kernels are compiled at run
time through MLX.

```bash
git clone https://github.com/dangxingyu/megakernel-zoo.git
cd megakernel-zoo
uv venv .venv-mac --python 3.12 && source .venv-mac/bin/activate
pip install -e .            # on a Mac this installs MLX, not PyTorch
```

## Quick start

```bash
mkserve doctor thinkingmachines/Inkling-2-Mini     # chip, memory, MLX, the engine for this Mac
mkserve serve thinkingmachines/Inkling-2-Mini      # downloads the weights (about 19 GB); --model-dir serves a local copy
mkserve chat                                       # talk to it in the terminal
mkserve bench --batch 1                            # time it on the repository's workload
```

The server speaks the OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`), Anthropic (`/v1/messages`)
and Ollama (`/api/chat`, `/api/generate`) APIs on one port, with Inkling 2's reasoning, text and tool calls in the
fields each API uses. See [`mkserve/README.md`](../mkserve/README.md).

For a coding agent, give the engine the larger context and point the agent at it:

```bash
mkserve serve thinkingmachines/Inkling-2-Mini --max-context 32768
mkserve launch claude        # or: codex, opencode
```

## What the server does

| | |
|---|---|
| **Exact speculative decoding** | Greedy or sampled. A reply is exactly what one-token-per-step decoding gives, only faster. |
| **Sampling** | `temperature`, `top_p`, `top_k`, `seed`. The same seed gives the same reply. |
| **Prompt prefill** | About 1,300 prompt tokens per second: a 1,630-token prompt gets its first token after 1.2 s. |
| **Prefix cache** | A request that starts like the previous one keeps its K/V: the next turn of a chat with a 1,452-token system prompt starts after 0.2 s. Reported as cached tokens in `usage`. |
| **Context** | 16,384 tokens by default, 32,768 with `--max-context 32768`. |
| **One sequence** | Requests queue; a client that disconnects frees the engine at once. |

### Sampling

A sampled token is `argmax(logits / T + noise)`, where the noise is a hash of (seed, position, token). A verify row
therefore draws the token a one-token step draws, and draft-and-verify stays exact. The draft head draws with the same
noise. Over the chat API a story runs at 76 tokens/s greedy, 80 at temperature 0.7 and 68 at temperature 1.0 with
`top_p` 0.95; a technical answer at 102, 113 and 86. One token per pass, which was the only way to sample before
(`--mode step`), is about 68.

### Prefill: fast and exact

`mkserve` serves with `--prefill fast`: the prompt runs through MLX GEMMs on the checkpoint's stored codes and fused
attention, in passes of up to 256 rows. `--prefill exact` uses the step's own kernels instead, 3.4× slower on a
1,630-token prompt (4.2 s against 1.2 s). The difference:

- **Exact**: every reply equals the engine's step-by-step tokens, bit for bit. This is what the evaluator runs.
- **Fast**: the prompt's K/V differs in the last bits, so a reply can leave the step-by-step one after some tokens
  (on random prompts, 1 of 4 replies stayed identical for 512 tokens). It is no further from the float32 reference.
  Decoding after the prompt is exact either way.

### Context

| `--max-context` | Decode, short prompt | Memory in use |
|---|---|---|
| 4,096 | 12.3 ms per token | 22.1 GB |
| 16,384 (default) | 11.9 | 24.6 GB |
| 32,768 | 12.9 | 27.7 GB |

MLX keeps about 6 GB more in its buffer cache. A decode step takes 13.6 ms at position 0 and 19.0 ms at position
30,000. An agent-sized request (19,683 tokens of system prompt and tool definitions) took 21.7 s to its first token
the first time, then 1.07 s and 0.82 s on the next two turns from the cache.

### Other Apple chips

The engine is registered for the M5 Max. On another Apple chip `mkserve` serves the same engine with a note: the
kernels are not specific to one chip, but nothing else has been tested, and the speed will be that chip's own.
`mkserve doctor` checks the memory.

## How it was checked

- **The gates.** Before it is timed, an engine must match a float32 reference of the model (the checkpoint evaluated
  on the CPU) within twice the error of the repository's BF16 path, at every position up to 128 and every 8th up to
  1,152, with every top-1 token equal. The engine: median relative error of the logits 0.024 (limit 0.049), p99 0.101
  (limit 0.121), top-1 agreement 1.0.
- **Exactness.** The evaluator compares the tokens of the request path with the engine's own step-by-step tokens on
  every sequence. With sampling, 16 of 16 cases (four prompts, four settings, 256 tokens) equal step-by-step sampling
  with the same seed.
- **Long positions.** On 9,600 and 24,000-token texts the top-1 token equals float32's at all 58 checked positions,
  and at 24,000 tokens the error is within twice the BF16 path's.
- **Serving.** All twelve checks of `scripts/mkserve_check.py`, in native and in step mode.

Where the remaining error comes from (expert-routing flips, 83% of it) is in the write-up and in
[`runs-mac/deviation`](../runs-mac/deviation/README.md).

## Limits

- **One machine.** Verified on one M5 Max with 48 GB. Other chips and memory sizes are untested.
- **One sequence.** Requests wait for each other.
- **The reference** is float32 on one sequence, and the gates stop at position 1,152. The GPU engines of this
  repository are checked against FP64 on more sequences. Long positions were checked on two texts only.
- **No independent audit yet.** The GPU engines were audited after their runs; this one has not been.
- **Sampling** is exact against the engine's own sampler (Gumbel noise keyed by seed and position), not against any
  other implementation's random numbers.
- **Hard prose** decodes at about 12 ms per token, greedy or sampled: there the draft head's first token is right
  only 60–70% of the time.
- **The baseline table is provisional**: Ollama ran one server start and LM Studio was not measured.

## Where things are

| | |
|---|---|
| The engine | [`engines/inkling-2-mini_m5max/offline-c`](../engines/inkling-2-mini_m5max/offline-c) |
| Its registry entry | [`mkserve/registry.json`](../mkserve/registry.json), key `m5max` |
| The agent runs (every submission, hypothesis and score) | `runs-mac/fable-c-m5max-1`, `runs-mac/fable-c-m5max-2` |
| Prefill, prefix cache, long context, sampling: measurements and scripts | [`runs-mac/prefill-20261123`](../runs-mac/prefill-20261123/README.md) |
| Baselines | [`baselines/mac`](../baselines/mac), `eval/baselines_inkling-2-mini_m5max_c.json` |
| The reference and the evaluator on a Mac | `scripts/mac_make_reference.py`, `scripts/eval_setup_mac.sh` |

## An engine for another model or chip

The same agent run works on a Mac: `bash autoresearch/launch_mac.sh RUN_DIR 6` starts a six-hour track-C run from
[`starter_mac`](../starter_mac) against the Mac evaluator. See [Build your own engine](../README.md#build-your-own-engine)
and [CONTRIBUTING.md](../CONTRIBUTING.md).
