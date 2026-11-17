# mkserve: a megakernel engine behind the OpenAI API

    pip install -e .                                   # from the repository root; `python -m mkserve` works without installing
    mkserve doctor Qwen/Qwen3-8B                       # is this machine ready? card, CUDA toolkit, packages, weights
    mkserve list                                       # the models and cards that have engines
    mkserve serve Qwen/Qwen3-8B                        # the default engine of this model on this card, on port 8000
    mkserve chat                                       # talk to it in the terminal
    mkserve bench                                      # time it; --url points the same benchmark at vLLM or SGLang

| command | what it does |
|---|---|
| `doctor [MODEL]` | checks the card, `nvcc`, the Python packages, which engines exist for this card and whether the weights are downloaded; prints the fix for every line that fails |
| `list` | every (model, card, engine) with its track and kind; `*` marks the default |
| `pull MODEL` | builds the engine and downloads the weights now, so that `serve` starts at once |
| `serve MODEL` | `--engine NAME` or `--track A\|B`, `--port`, `--host`, `--model-dir` (a local snapshot), `--api-key KEY` (Bearer token on `/v1/*`), `--served-model-name`, `--mode auto\|native\|step` |
| `chat` | a streaming terminal chat with any running OpenAI-compatible server (`--url`, `--temperature`); prints tokens per second after each answer |
| `bench` | the project's workload against any OpenAI-compatible server: random 64-token prompts, 1,024 greedy tokens, 1 and 8 requests at once (`--batch`, `--prompt-tokens`, `--max-tokens`, `--rounds`); prints milliseconds per step, tokens per second and time to first token |

    curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
         -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'

Needs one supported card, `nvcc` on PATH, and Python with torch, numpy, aiohttp, tokenizers, jinja2 (huggingface_hub
to download weights; `--model-dir` serves a local snapshot). The engine is unpacked, checked against its registered
sha256 and built under `~/.cache/mkserve` (`--workdir`). There is no generic fallback: a model and card that are not
in `registry.json` are refused.

Endpoints: `/v1/models`, `/health`, `/metrics` (Prometheus text: requests, tokens, time to first token, time per output token), `/v1/completions`, `/v1/chat/completions`; streaming or not; `max_tokens`, `stop`,
`ignore_eos`, `seed`, `temperature`/`top_p`/`top_k` (step mode), `stream_options.include_usage`, `chat_template_kwargs`, usage. The chat template is the model's own.

## Two ways an engine is driven

| | native (engines with `submit`/`poll`) | step (every engine) |
|---|---|---|
| batching | the engine's own: requests of any length are taken while others decode | prompts of equal length that arrive together; nothing is admitted while a batch decodes |
| prompt | prefilled in chunks inside the engine | one token per step |
| sampling | greedy only (a sampling request is decoded greedily with a warning; `--strict-sampling` refuses it) | `temperature`, `top_p`, `seed` |
| a request that ends early | the answer ends, the engine's row decodes on to `max_tokens` (the contract has no cancel) | the row is dropped |

## Request validation

The engines do not check their inputs, so `mkserve` does before a request reaches one: an empty prompt, `max_tokens` below 1,
a token id outside the model's vocabulary, and a prompt plus `max_tokens` beyond the engine's context are answered with
HTTP 400. A request never reaches an engine it could crash or corrupt.

## Tests

`scripts/mkserve_check.py --port PORT --engine NAME --model MODEL` runs twelve checks against a served engine: the model
list, a completion, a streamed completion equal to the unstreamed text, a chat and a streamed chat, a stop string, four
chats of different lengths at once, sampling with a seed, decode speed, a long prompt, a request beyond the context, and
invalid inputs. `scripts/mkserve_test.sh` serves every engine of a model in turn and runs them. All 13 registered engines
pass all twelve on their card; `registry.json` records the date for each.

| Qwen3-8B on one H100 | `online-b` (native mode) | `offline-b` (step mode) |
|---|---|---|
| start (build and load) | 41 s | 20 s |
| first token, 64-token prompt | 0.007 s | 0.28 s |
| first token, 1,429-token prompt | 0.05 s | 5.9 s |
| decode, 1 sequence, 512 tokens | 5.54 ms per token | 4.13 ms per token |
| sampling with a seed | greedy (as documented) | reproducible per seed |

These figures are from one run of the test and illustrate the two modes; the benchmark numbers are in the top-level README.
