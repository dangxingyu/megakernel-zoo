## What this pull request adds

<!-- An engine, a model, a GPU, an evaluator or server change. One sentence. -->

## For a new or improved engine

- Engine directory: `engines/<model>_<gpu>/<name>/`
- GPU it was written and measured on (`nvidia-smi -L`):
- Track: A / B
- [ ] `python scripts/check_engine.py engines/<model>_<gpu>/<name>` passes
- [ ] `eval/kernel_eval.py` result attached, `"status": "valid"`
- [ ] `scripts/release_check.sh` passes (every batch size, slot order, neighbors, repetition, full context, restart)
- [ ] Same-GPU comparison from `scripts/compare.sh` attached, with the vLLM and SGLang versions
- [ ] Entry added to `mkserve/registry.json` (track, kind, source, sha256, from, limits, what)
- [ ] Track B only: what differs from track A, and whether the logits are bit-identical to the track-A build of the same source

Time per token (ms), batch size 1 / 8:

| | ours | fastest general engine |
|---|---|---|
| this engine | | |

## For an evaluator, server or kit change

- [ ] `python scripts/check_engine.py --all` still passes
- [ ] Which registered engines were re-run, and their results
