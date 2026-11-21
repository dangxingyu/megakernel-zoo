# Contributing

Contributions are welcome in four forms, from most to least common:

| You want to | Start at |
|---|---|
| add a faster engine for a model and GPU that are already supported | [Add or improve an engine](#add-or-improve-an-engine) |
| support a new model or a new GPU | [A new model or GPU](#a-new-model-or-gpu) |
| improve the evaluator, `mkserve` or the autoresearch kit | [Evaluator, server and kit](#evaluator-server-and-kit) |
| report a wrong answer, a crash or a slow engine | [Reporting a problem](#reporting-a-problem) |

The bar for an engine is the one ours had to meet: its logits match the FP64 reference within the tolerances of the
evaluator in this repository, and its speed is measured by the same evaluator on the GPU it was written for.

## Add or improve an engine

**1. Set up the evaluator** for the model on your GPU. This downloads the weights, builds the FP64 reference and the
environment, and checks that the starter engine passes:

```bash
bash scripts/eval_selftest.sh <hf-org>/<model>
```

**2. Write the engine.** Either let an agent do it with the autoresearch kit, or write it yourself against the contract
in [`starter/engine/api.py`](starter/engine/api.py). An engine of this repository is a good starting point:

```bash
python autoresearch/start.py my-run --model <hf-org>/<model> --card <h100|b200> --hours 12
python autoresearch/start.py my-run --model <hf-org>/<model> --card <h100|b200> --engine engines/<model>_<gpu>/offline-a
```

[`engines/README.md`](engines/README.md) describes the layout of an engine directory and the two contracts (offline:
`reset`/`step`; online: `submit`/`poll`).

**3. Evaluate it.** The result must say `"status": "valid"`. The hidden set we used (natural text, code and word
problems, with a secret seed) is not in this repository: a public test set is one an agent can overfit. Without a
hidden set the evaluator skips that gate with a warning. We strongly encourage you to build your own, with a seed only
you know, and to evaluate against it before trusting a result ([Your own hidden set](#your-own-hidden-set)); we also
run ours on every pull request.

```bash
python eval/kernel_eval.py --workspace <engine directory> --out result.json

# the release checks and the same-GPU comparison take the engine as an archive in rescore_inbox/
mkdir -p rescore_inbox && tar czf rescore_inbox/my-engine.tgz -C <engine directory> .
export MK_MODEL=<hf-org>/<model> MK_HW=<h100|b200>
RELCHECK_SOURCES=my-engine bash scripts/release_check.sh             # every batch size, slot order, neighbors,
                                                                    # repetition, full context, restart
COMPARE_SYSTEMS="ours:my-engine sglang vllm" bash scripts/compare.sh  # vLLM and SGLang over HTTP, same client
```

**4. Add it to the repository:**

- the source under `engines/<model>_<gpu>/<name>/` (naming rules in [`engines/README.md`](engines/README.md));
- an entry in [`mkserve/registry.json`](mkserve/registry.json) (fields below);
- then run the static checks, which must all pass:

```bash
python scripts/check_engine.py engines/<model>_<gpu>/<name>
```

**5. Open a pull request.** The template asks for `result.json`, the GPU (`nvidia-smi -L`), the versions of the general
engines you compared with, and, for a track-B engine, what differs from track A and whether the logits are
bit-identical to the track-A build of the same source.

### Your own hidden set

An engine written by an agent should be judged on inputs the agent never saw. Build a private reference once per model,
from natural-text prompts and a seed you keep to yourself, and keep it outside the engine's workspace:

```bash
MK_NATURAL_SEED=<your secret number> python eval/natural_prompts.py <model dir> hidden_prompts.json 64 8
python eval/make_reference.py --model-dir <model dir> --out ~/hidden-ref/<model> --natural hidden_prompts.json \
    --seed <another secret number> --sequences 8 --prompt-tokens 64 --check-tokens 192 --long-tokens 2000 --long-every 16
python eval/kernel_eval.py --workspace <engine directory> --out result.json --ref-hidden ~/hidden-ref/<model>
```

`result.json` then reports a `hidden` gate (pass or fail only, so the numbers cannot leak back into a run). An engine
that passes the public set and fails your hidden one has learned the public inputs, not the model.

### Registry entry

```jsonc
"offline-b": {
  "track": "B",                                  // A, B or C, as in engine/TRACK
  "kind": "offline",                             // offline (reset/step) or online (submit/poll)
  "source": "../engines/qwen3-8b_h100/offline-b",
  "sha256": "b76d9f48be969976",                  // first 16 hex digits; scripts/check_engine.py prints the value
  "from": "your name or run, and what it is based on",
  "max_batch": 8,
  "max_context": 2048,
  "what": "one sentence: formats that differ from the checkpoint, bit-identity with track A, limits",
  "verified": "date: what was checked, on which GPU"
}
```

The model's `"default"` engine changes only when the new engine passes every check and is faster at batch size 1 and 8.

## Rules an engine has to follow

- **The model's function.** For every input the engine returns what the checkpoint defines: no quantization, pruning,
  distillation or approximation of weights or activations, and no part of the model skipped.
- **Track A** keeps the checkpoint's weights, number formats and data flow; every operator runs for every token and
  one token is produced per pass. **Track B** may change how the model is stored and what is computed at load time, if
  the function is unchanged in exact arithmetic. **Track C** drafts and verifies several tokens per pass (speculative
  decoding) and must be lossless: with greedy decoding, its tokens equal its own one-token-per-step greedy tokens for
  every sequence (`MK_TRACK=c` in `scripts/eval_setup.sh` turns on that gate and the token-based rate, and
  [`eval/spec_check.py`](eval/spec_check.py) checks it on natural text, at batch 1 and 8 and with requests that
  arrive one after another). The three are reported separately; track C is compared only with the general engines'
  own speculative decoding.
- **Self-contained.** No general inference engine or library model code (vLLM, SGLang, transformers, ...) in the timed
  path; everything is rebuilt from the submitted source by `engine/build.sh`, and nothing large is committed (weights
  and tables are made at load time).
- **No knowledge of the evaluator.** An engine must not recognize the evaluator's inputs, cache outputs, or read the
  reference.

## A new model or GPU

**A model that Hugging Face `transformers` can run** needs only step 1 above ([`eval/modelinfo.py`](eval/modelinfo.py)
reads the architecture from the checkpoint's `config.json`), plus one small file:
`eval/baselines_<model>_<gpu>.json`, the time per token of vLLM and SGLang on that GPU as measured by
`scripts/compare.sh`, so that runs and their board know what to beat. The existing files show the format.

**Any other model** needs a reference implementation that [`eval/make_reference.py`](eval/make_reference.py) can run
in FP64 (FP32 when FP64 does not fit); [`eval/inkling2_ref.py`](eval/inkling2_ref.py) is an example.

**A new GPU** needs an entry in the `CARDS` table of [`eval/modelinfo.py`](eval/modelinfo.py) (memory, bandwidth,
architecture) and, for `mkserve`, the name `nvidia-smi` reports, mapped in `detect_card` in
[`mkserve/__main__.py`](mkserve/__main__.py).

[`eval/README.md`](eval/README.md) maps the evaluator's files.

## Evaluator, server and kit

- **Evaluator** ([`eval/`](eval)): a change that makes a check stricter must say which engines of this repository
  still pass it (`python scripts/check_engine.py --all` for the static part, `scripts/release_check.sh` on a GPU for
  the rest). A check that can fail on a correct engine needs a test that shows it does not.
- **Server** ([`mkserve/`](mkserve)): `bash scripts/mkserve_test.sh` runs its tests; a change to the API must keep
  `scripts/mkserve_check.py` passing against every registered engine you can run.
- **Kit** ([`autoresearch/`](autoresearch)): prompts and skills name the rules and the tools, not optimization
  techniques; measured findings go into [`autoresearch/skills/megakernel/playbook.md`](autoresearch/skills/megakernel/playbook.md)
  with the run and submission they come from.

## Reporting a problem

Open an issue (there are templates for a wrong answer or crash, and for proposing an engine). Include the model, the GPU
(`nvidia-smi -L`), the CUDA toolkit and PyTorch versions, the command and its output. For a wrong answer from a served
engine, also include the request and the output of:

```bash
python scripts/mkserve_check.py --port <port> --engine <name> --model <model>
```
