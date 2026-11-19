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
| `hf_text.py`, `inkling2_ref.py` | Reference implementations: any Hugging Face text decoder in a chosen precision; Inkling-2-Small in plain PyTorch. |
| `modelinfo.py` | What the evaluator needs to know about a checkpoint (read from its `config.json`) and about a GPU (`CARDS`: memory, bandwidth). |
| `prompts.py`, `natural_prompts.py` | Prompts with an exact token count: random tokens for the benchmark, natural text and code for the extra workloads. |

## Data files, one per model and GPU

Named `<model>_<gpu>` like the engine directories.

| File | Contents |
|---|---|
| `baselines_<model>_<gpu>.json` | Time per token of the general engines (vLLM, SGLang, Cohere's megakernel for North Mini Code) on that GPU, with versions and the job they come from. |
| `refs_<model>_<gpu>.json` | The reference lines a run sees on its board: the HBM-bandwidth bound, the target, the general engines. |
| `refs/<hf-org>-<model>/ids.npy` | The token ids of the public reference sequences. The logits are rebuilt by `make_reference.py`; the hidden sequences are not in this repository. |

## Tolerances

An engine passes when, on every checked position, its deviation from the FP64 reference stays within a multiple of the
standard BF16 implementation's (Hugging Face `transformers`): 2× on the median and 99th percentile of the logit error,
4× on the KL divergence of the next-token distribution, and top-1 agreement within one point. The multiples are per
model, so they follow its numerics. See the write-up's [validation section](https://dangxingyu.github.io/megakernel/#validation).
