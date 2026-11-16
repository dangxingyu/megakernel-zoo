# Contributing

Contributions of new engines, new models and cards, and improvements to the evaluator are welcome. The bar for an
engine is the same one ours had to meet: it is correct against the float64 reference, and its speed is measured by the
evaluator of this repository.

## Adding or improving an engine

1. **Set up the evaluator** for the model on your card (weights, a float64 reference, the environment):

   ```bash
   bash scripts/eval_selftest.sh <hf-org>/<model>
   ```

2. **Produce the engine.** Start a run with the autoresearch kit, or write the engine by hand against the contract in
   [`starter/engine/api.py`](starter/engine/api.py):

   ```bash
   python autoresearch/start.py my-run --model <hf-org>/<model> --card <h100|b200> --hours 12
   # or start from an engine of this repository: --engine engines/<model>_<card>/offline-a
   ```

3. **Evaluate it** and keep the result file:

   ```bash
   python eval/kernel_eval.py --workspace <engine directory> --out result.json     # status must be "valid"
   python eval/release_check.py --help                                             # the extended checks
   ```

4. **Open a pull request** with
   - the engine directory under `engines/<model>_<card>/<name>/` (source only: `engine/api.py`, `engine/build.sh`, the
     CUDA source, `engine/TRACK`);
   - an entry in [`mkserve/registry.json`](mkserve/registry.json) (track, kind, limits, where it came from);
   - `result.json` from step 3, the card it was measured on, and the versions of the general engines you compared with
     (`scripts/compare.sh` runs that comparison);
   - for a track-B engine, a statement of what differs from track A and whether the logits are bit-identical to the
     track-A build of the same source.

## Rules an engine has to follow

- **The model's function.** For every input the engine returns what the checkpoint defines: no quantization, pruning,
  distillation or approximation of weights or activations, and no part of the model skipped.
- **Track A** keeps the checkpoint's weights, number formats and data flow; every operator runs for every token and one
  token is produced per pass. **Track B** may change storage and load-time precomputation if the function is unchanged
  in exact arithmetic. The two are reported separately.
- **Self-contained.** No general inference engine or library model code in the timed path; everything is rebuilt from
  the submitted source by `engine/build.sh`.
- **No knowledge of the evaluator.** An engine must not recognise the evaluator's inputs, cache outputs, or read the
  reference.

## A new model

A model that Hugging Face `transformers` can run needs nothing beyond step 1. For any other model, add a reference
implementation that `eval/make_reference.py` can run in float64 (or float32 when float64 does not fit the card);
[`eval/inkling2_ref.py`](eval/inkling2_ref.py) is an example.

## Reporting a problem

Open an issue with the model, the card (`nvidia-smi`), the CUDA toolkit and PyTorch versions, the command, and the
output. For a wrong answer from a served engine, include the request and the output of
`python scripts/mkserve_check.py --port <port> --engine <name> --model <model>`.
