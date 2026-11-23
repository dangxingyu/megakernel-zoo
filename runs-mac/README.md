# Run records: Apple silicon

The records behind [`mac/README.md`](../mac/README.md), all from one Apple M5 Max (40-core GPU, 48 GB). The GPU runs are
in [`runs/`](../runs/README.md).

## The agent runs

One folder per run, in the form of `runs/`: `progress.md` lists every submission (what was tried, why, what it
measured), `src/` has the source of every submission and `notes/` the agent's own notes. `PROMPT.md` is the prompt the
agent was given, and `ledger.jsonl` the same submissions as data. Numbers in these files are the run's own
evaluator's.

| run | model and chip | |
|---|---|---|
| [`fable-c-m5max-1`](fable-c-m5max-1/progress.md) | Inkling-2-Mini on M5 Max | Score: `http_step_us_b1` (lower is better). Track C. Budget 6.0 agent-hours. 21 submissions, 21 valid. From 26,852 (no speculation) to 8,982. |
| [`fable-c-m5max-2`](fable-c-m5max-2/progress.md) | Inkling-2-Mini on M5 Max | Continues run 1 from its final source. Budget 6.0 agent-hours. 19 submissions, 19 valid. To 8,061 (median of the final source's six evaluations: 8,100). |

`scripts/export_run.py RUN_DIR` writes this form from a run directory. The agent transcripts and the workspaces are
not in the repository.

## After the runs

| folder | what |
|---|---|
| [`prefill-20261123`](prefill-20261123/README.md) | Prompt prefill in wide passes, the prefix cache, the 32,768-token context and sampling through draft-and-verify: timings, exactness and accuracy checks, and the scripts that made them |
| [`final-eval-20261123`](final-eval-20261123) | The full evaluator on the source in this repository, after all of the above: valid, 8,205 / 5,173 microseconds per token |
| [`deviation`](deviation/README.md) | Where the 0.025 error of every BF16 implementation against float32 comes from (expert-routing flips, 83%) |
| [`mkserve-check-20261123`](mkserve-check-20261123) | Logs of the twelve serving checks, in native and step mode |
| [`report-20261123`](report-20261123/inkling-2-mini-m5max.html) | The write-up as one HTML page, English and 中文 |

## Baselines

The general engines' measurements, summarized in `eval/baselines_inkling-2-mini_m5max_c.json` and described in
[`baselines/mac`](../baselines/mac/README.md): `baselines-mlx` (mlx-lm and oMLX), `baselines-llamacpp`,
`baselines-llamacpp-ds` (with the DSpark draft), `baselines-chat` (the chat workload) and
`baselines-20261123-ollama` (one server start; Ollama could not load the GGUF or the draft). `prompts/` has the
prompts they were given. The servers' own logs are not in the repository.
