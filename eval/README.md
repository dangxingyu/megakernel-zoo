# Evaluator

Everything that decides whether an engine is correct and how fast it is. Agents cannot modify any of it during a run.

## Entry points

| File | What it does |
|---|---|
| [`kernel_eval.py`](kernel_eval.py) | Evaluates one engine: builds it, checks its logits against the FP64 reference, times it, and writes `result.json` (`"status": "valid"` or the first failed gate). |
| [`online_eval.py`](online_eval.py) | Evaluates a serving engine under a request trace: requests arrive and leave as in production. |
| [`release_check.py`](release_check.py) | The checks a finished engine passes before release: every batch size, slot order, neighbors, repetition, full context, restart. |
| [`make_reference.py`](make_reference.py) | Builds a model's reference data (FP64 logits on the public and hidden sequences) once per model. |
| [`client.py`](client.py) | The benchmark client for any OpenAI-compatible streaming server: steady decoding and trace replay, timed on the client's clock. |

## Supporting modules

| File | What it does |
|---|---|
| `_engine_runner.py`, `_online_runner.py` | Child processes that load the engine under test and drive it through the offline or online contract (the contracts are in their docstrings). |
| `serve_engine.py`, `http_stage.py` | Put an engine behind the same streaming HTTP endpoint as vLLM and SGLang, so that it is timed the same way. |
| `hf_text.py`, `inkling2_ref.py` | Reference implementations: any Hugging Face text decoder in a chosen precision; Inkling-2-Small and Inkling-2-Mini (its looped layers included) in plain PyTorch. |
| `spec_check.py` | Track C: a speculative engine's `submit`/`poll` tokens against its own `step()` greedy tokens, on the benchmark's prompts or natural text, at batch 1 or 8, requests together or staggered. |
| `modelinfo.py` | What the evaluator needs to know about a checkpoint (read from its `config.json`) and about a GPU (`CARDS`: memory, bandwidth). |
| `prompts.py`, `natural_prompts.py` | Prompts with an exact token count: random tokens for the benchmark, natural text and code for the extra workloads. |

## Data files, one per model and GPU

Named `<model>_<gpu>` like the engine directories.

| File | Contents |
|---|---|
| `baselines_<model>_<gpu>.json` | Time per token of the general engines (vLLM, SGLang, Cohere's megakernel for North Mini Code) on that GPU, with versions and the job they come from. |
| `refs_<model>_<gpu>.json` | The reference lines a run sees on its board: the HBM-bandwidth bound, the target, the general engines. |
| `refs/<hf-org>-<model>/ids.npy` | The token ids of the public reference sequences. The logits are rebuilt by `make_reference.py`. |
| `baselines_<model>_<gpu>_c.json`, `refs_<model>_<gpu>_c.json` | Track C: the general engines with their own speculative decoding (time per token counted in tokens), and the board's lines for a track-C run (`MK_TRACK=c`). |

Inkling-2-Mini is not in a `transformers` release yet: the evaluation environment installs the model's pull request
(huggingface/transformers#51050, pinned commit; `mk_tf_spec` in `scripts/env.sh`) as the standard BF16
implementation, and its FP64 reference comes from `inkling2_ref.py` (`MK_REF_IMPL=inkling2 bash scripts/eval_selftest.sh
thinkingmachines/Inkling-2-Mini`). The reference and the engines do not quantize the expert inputs; the checkpoint's
quantization config asks for it, and vLLM and SGLang do it, so their outputs are a slightly different function.

The hidden set (natural text, code and word problems from a secret seed, positions to 2,000) is not in this repository. Build your own with a seed only you know and pass it with `kernel_eval.py --ref-hidden`; [CONTRIBUTING.md](../CONTRIBUTING.md#your-own-hidden-set) has the commands.

## Tolerances

An engine passes when, on every checked position, its deviation from the FP64 reference stays within a multiple of the
standard BF16 implementation's (Hugging Face `transformers`): 2× on the median and 99th percentile of the logit error,
4× on the KL divergence of the next-token distribution, and top-1 agreement within one point. The multiples are per
model, so they follow its numerics. See the write-up's [validation section](https://dangxingyu.github.io/megakernel/#validation).

## On Apple silicon

The evaluator also runs on a Mac (tested on an M5 Max, 48 GB, macOS 26, MLX 0.32; no CUDA, no NVML, no Xcode). The GPU
path is unchanged; `eval/_device.py` picks the device (`MK_DEVICE=cuda | metal`, default `metal` on macOS) and holds the
only difference in the engine contract: on `metal`, `step()` takes a NumPy int64 array `[B]` and returns an `mx.array` or
NumPy array `[B, vocab]` (float32 / bf16 / fp16, possibly lazy: the harness `mx.async_eval`s every step and
`mx.eval` + `mx.synchronize()`s before reading a clock); `generate()` returns integers `[B, n]`; `submit` / `poll` are as
on the GPU. No torch is needed.

```bash
bash scripts/eval_setup_mac.sh                     # checks .venv-mac and the checkpoint; writes /tmp/mk/eval_env.json (track c, card m5max)
MK_REF_DIR=~/mk-ref/thinkingmachines-inkling-2-mini \
  .venv-mac/bin/python eval/kernel_eval.py --workspace starter_mac --out /tmp/result.json --track c
python3 autoresearch/start.py RUN --model thinkingmachines/Inkling-2-Mini --card m5max --track C --hours 12
```

What differs on a Mac:

- **Batch 1 only by default** (`MK_EVAL_BATCHES` / `--batches`, default `1`). The correctness check runs the eight
  reference sequences one at a time, so `batch` becomes a repeatability check and the slot-order `invariance` gate (bit
  for bit, batches > 1 only) is skipped; `MK_EVAL_BATCHES=1,8` restores the batch-8 check and measurement for an engine
  that serves eight. The HTTP stage measures batch 1 (synthetic and, in track C, natural text) and checks the server's
  tokens one request at a time (gate `http_tokens_b1`; in track C this check also runs on the GPU, so that the
  single-request speculative path is gated there too).
- **No trace stage** (no CUPTI / nsys): `single_launch` and `gpu_never_waits` are not judged. **No NVML** in the
  client (`http_stage.py` does not pass `--nvml` on macOS); the runner reports MLX's peak allocation instead.
- **Scratch** in `$TMPDIR` (no `/dev/shm`); **time limits** x4 (`eval_timeout_scale` of the card; `MK_EVAL_TIMEOUT_SCALE`
  overrides).
- **MBU** uses the bandwidth measured on this Mac by `scripts/mac_membw.py`: `CARDS["m5max"]` in `modelinfo.py`, 565 GB/s.
- **The reference** is not built here: copy the float64 reference made on a GPU machine (`make_reference.py`; `ids.npy`,
  `meta.json`, `logits64.npy`, `long_positions.npy`, `long_logits64.npy`) and point `MK_REF_DIR` or `--ref` at it. Without
  it the evaluation stops at once with status `error` naming the missing files.
- `release_check.py` checks batch sizes up to `--max-batch` (default 1 on metal); `spec_check.py gen` takes `--max-batch 1`
  for an engine that serves one sequence.
- The starting engine is `starter_mac/` (mlx-lm's `inkling2` model behind the contract, track C, one token per step);
  the run's prompt is `autoresearch/prompts/first_C_mac.md` and its skill `autoresearch/skills/megakernel/SKILL_mac.md`.
