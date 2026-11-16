# megakernel-zoo

A megakernel is an inference engine compiled into one GPU kernel for one model on one type of card: the layers, the choice
of the next token and the loop over tokens run inside it, and nothing on the GPU waits for a host scheduler between tokens.

This repository has

- **engines** for five open models, each written by Claude in an autonomous run and scored by a fixed evaluator;
- **`mkserve`**, a small server that puts an engine behind the OpenAI API;
- **the suite** the engines were tested and timed with, including the comparison with vLLM and SGLang;
- **the autoresearch kit**: the skill, the prompts and the tools to run such a run on your own card with your own agent;
- **the record of every run**: each submission's source, the agent's hypothesis for it, and its result.

The write-up, with every measurement and how it was made: https://dangxingyu.github.io/megakernel-dashboard/

## Results

Time per token over HTTP with a 64-token prompt and 1,024 greedy new tokens, every system served in turn on the same card,
three rounds. The factor is the fastest general engine's time over ours, with one sequence and with eight.

| Model | Card | Fastest general engine, ms | Track A, ms | Track A | Track B, ms | Track B |
|---|---|---|---|---|---|---|
| Qwen3-8B | H100 | SGLang 6.13 / 6.84 | 5.38 / 5.64 | **1.14×** / **1.21×** | 4.16 / 4.47 | **1.47×** / **1.53×** |
| North Mini Code | H100 | Cohere's own megakernel 3.62 / 7.07 | 2.54 / 6.90 | **1.43×** / 1.02× | 2.15 / 6.05 | **1.68×** / **1.17×** |
| flint | B200 | vLLM 1.99 / 3.79 | 1.48 / 3.40 | **1.35×** / **1.12×** | 1.48 / 3.95 | **1.35×** / 0.96× |
| Qwen3.8-27B | B200 | SGLang 9.53 / 10.23 | 8.45 / 8.89 | **1.13×** / **1.15×** | 7.71 / 8.18 | **1.24×** / **1.25×** |
| Inkling-2-Small | 4 × B200 | SGLang 4.74 / 6.55 | 3.30 / 5.39 | **1.43×** / **1.22×** | 3.29 / 5.38 | **1.44×** / **1.22×** |

Track A keeps the checkpoint's number formats and data flow: the same computation, organised differently. Track B may also
change how the model is stored and what is computed once at load, without changing the function. The track-B engines of
Qwen3.8-27B and Inkling-2-Small give logits bit-identical to track A; the other three keep intermediate values in float32
where the standard implementation rounds to BF16, so they are at least as precise but not bit-identical.

## Serve an engine

```bash
uv venv && source .venv/bin/activate
uv pip install torch numpy safetensors tokenizers aiohttp jinja2 huggingface_hub
python -m mkserve list
python -m mkserve serve Qwen/Qwen3-8B            # needs the engine's card (here an H100) and nvcc on PATH
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64, "stream": true}'
```

The engine is built from its CUDA source on first use and the weights come from Hugging Face (`--model-dir` serves a local
snapshot). An engine exists for one model on one card type, and `mkserve` refuses any other pair.

| Model | Card | Engine | Track | Kind | Sequences | Context | From |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-8B | h100 | `online-b` (default) | B | online | 64 | 34,816 | run s3-q8b-online-a1, submission s0039 |
| Qwen/Qwen3-8B | h100 | `online-a` | A | online | 64 | 34,816 | run s3-q8b-online-a1, submission s0038 |
| Qwen/Qwen3-8B | h100 | `offline-b` | B | offline | 8 | 2,048 | run s10-q8b-h100-f1, submission s0002 |
| Qwen/Qwen3-8B | h100 | `offline-a` | A | offline | 8 | 2,048 | run s10-q8b-h100-f1, submission s0003 |
| RecursiveSI/flint | b200 | `offline-a` (default) | A | offline | 8 | 2,048 | run s4-flint-b200-a2, submission s0077 |
| RecursiveSI/flint | b200 | `offline-b` | B | offline | 8 | 2,048 | run s4-flint-b200-a2, submission s0055 |
| Qwen/Qwen3.8-27B | b200 | `offline-a` (default) | A | offline | 8 | 2,048 | run s9-qwen38-27b-b200-f1, submission s0022 |
| Qwen/Qwen3.8-27B | b200 | `offline-b` | B | offline | 8 | 2,048 | run s9-qwen38-27b-b200-f1, submission s0023 |
| CohereLabs/North-Mini-Code-1.0 | h100 | `offline-a` (default) | A | offline | 8 | 2,048 | run s4-north-h100-a1, submission s0022 |
| CohereLabs/North-Mini-Code-1.0 | h100 | `offline-b` | B | offline | 8 | 2,048 | run s4-north-h100-a1, submission s0021 |
| thinkingmachines/Inkling-2-Small | b200x4 | `offline-a` (default) | A | offline | 8 | 2,048 | run s7-inkling2s-b200x4-fb1, submission s0034 |
| thinkingmachines/Inkling-2-Small | b200x4 | `offline-b` | B | offline | 8 | 2,048 | run s7-inkling2s-b200x4-fb1, submission s0036 |
| thinkingmachines/Inkling-2-Small | b200x4 | `online-a` | A | online | 64 | 34,816 | run s8-inkling2s-b200x4-on1, submission s0043 |

Engines of kind `online` admit requests while others are decoding, process prompts in chunks and reuse shared prefixes.
Engines of kind `offline` are the fastest at steady decoding; they hold 8 sequences and a 2,048-token context, and
`mkserve` feeds them the prompt one token per pass. Sampling (`temperature`, `top_p`, `top_k`, `seed`) is done by
`mkserve` on the engine's logits. `mkserve` also checks every request before the engine sees it (context, token ids, an empty
prompt, `max_tokens`), because most engines do not check their inputs themselves.

## Test an engine

```bash
bash scripts/eval_selftest.sh Qwen/Qwen3-8B      # once per model: weights, a float64 reference made on this card, the evaluator checked on the starter
MK_MODEL=Qwen/Qwen3-8B bash scripts/eval_setup.sh
python eval/kernel_eval.py --workspace engines/qwen3-8b_h100/offline-a --out result.json
```

The evaluator builds the engine and gives it a time only if it is correct:

- logits against the float64 reference at every position, alone and in a batch of 8, within twice the error of the Hugging
  Face BF16 implementation; the same along 1,152 positions; top-1 agreement with float64;
- the engine's own decode loop and its HTTP path give the tokens of stepping it;
- the timed steps themselves are checked against the reference afterwards.

The reference is computed on your card, on the token sequences in `eval/refs/` (the ones our runs were scored on), so a
result is comparable with ours. `MK_REF_OWN_SEQUENCES=1` generates new sequences instead; expect the limit on the largest
error to move with them, since the largest of a thousand errors comes from a heavy tail (the write-up has the numbers).

`eval/release_check.py` adds every batch size from 1 to 8, slot order, other neighbours in the batch, repetition, the state
left by a full context, and a process restart. `scripts/compare.sh` serves vLLM, SGLang and any engines in turn on this card
and times them with one client (`eval/client.py`): steady decoding, sampled decoding, natural-text prompts, 1 to 64 requests,
long generations, long prompts, and the replay of a production request trace.

## Run your own

```bash
bash scripts/eval_selftest.sh Qwen/Qwen3-8B
python autoresearch/start.py my-run --model Qwen/Qwen3-8B --card h100 --hours 12
cd my-run/workspace && claude -p "$(cat ../PROMPT.md)" --dangerously-skip-permissions
```

`start.py` makes a workspace with an engine to start from (the Hugging Face model behind the engine contract, or any zoo
engine with `--engine`), installs the skill, writes the prompt, and gives the agent one tool: `mk submit` evaluates the
engine as it is and records the result, `mk board` shows every submission and the time left. Any coding agent that has a
shell works. The prompts state the goal, the rules and the two tracks; they name no optimization technique.

`autoresearch/skills/megakernel` is the skill every run starts from. `autoresearch/skills_by_model` has the skills of our
runs, with the facts of each model and card. For a model that is not in Hugging Face `transformers`, the reference has to
be written first; `eval/inkling2_ref.py` is the one we wrote for Inkling-2-Small.

Our own runs kept the agent on one machine and sent everything that needs the GPU to pods on a cluster. That loop depends
on our scheduler and is not included; the tools here are its single-machine equivalent.

## Limits

- An engine is tied to one model and one card type, and is rebuilt from source on the machine that serves it.
- The offline engines do steady decoding only: 8 sequences, 2,048 tokens of context.
- The serving engines remove the stalls a general engine shows under load. The one for Inkling-2-Small decodes about a
  third as fast as the offline engine; the write-up has its capacity against SGLang's at each arrival rate.
- Decoding inside the kernel is greedy; sampling runs outside it and costs 1 to 11%.
- Use the engines through `mkserve`. Called directly, most of them do not validate requests: an invalid token id or an
  oversized request can crash an engine or corrupt another request. The serving engines for Qwen3-8B match shared prompt
  prefixes by a 64-bit hash without comparing the tokens.
- Every engine was audited by an independent agent; the audit reports' findings are summarised in the write-up.
- Numbers are from single nodes and two to three rounds.

## Licence

Apache-2.0. `eval/inkling2_ref.py` began as a transcription of model code from NovaSky-AI/SkyRL (Apache-2.0), which is
noted in the file.
