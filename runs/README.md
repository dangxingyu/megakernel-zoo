# Run records

One folder per run: `progress.md` lists every submission (what was tried, why, what it measured), `src/` has the source of every
submission and `notes/` the agent's own notes. Numbers in these files are the run's own evaluator's; the audited, re-measured
results are in the top-level README.

| run | model and card | |
|---|---|---|
| [`s2-q8b-a1`](s2-q8b-a1/progress.md) | Qwen3-8B on H100 | Score: `step_us_b1` (lower is better). Budget 40.0 agent-hours. 66 submissions so far, 66 valid. |
| [`s3-q8b-online-a1`](s3-q8b-online-a1/progress.md) | Qwen3-8B on H100 | Score: `slo_share_r2`. Budget 40.0 agent-hours. 39 submissions so far, 36 valid. |
| [`s4-north-h100-a1`](s4-north-h100-a1/progress.md) | North-Mini-Code-1.0 on H100 | Score: `http_step_us_b1` (lower is better). Budget 24.0 agent-hours. 23 submissions so far, 23 valid. |
| [`s4-flint-b200-a2`](s4-flint-b200-a2/progress.md) | Flint on GB200 | Score: `http_step_us_b1` (lower is better). Budget 36.0 agent-hours. 77 submissions so far, 76 valid. |
| [`s4-qwen38-27b-b200-a1`](s4-qwen38-27b-b200-a1/progress.md) | Qwen3.8-27B on GB200 | Score: `http_step_us_b1` (lower is better). Budget 24.0 agent-hours. 50 submissions so far, 48 valid. |
| [`s5-qwen38-27b-b200-b1`](s5-qwen38-27b-b200-b1/progress.md) | Qwen3.8-27B on GB200 | Score: `http_step_us_b1` (lower is better). Budget 6.0 agent-hours. 17 submissions so far, 14 valid. |
| [`s6-inkling2s-b200x4-a1`](s6-inkling2s-b200x4-a1/progress.md) | Inkling-2-Small on GB200 | Score: `http_step_us_b1` (lower is better). Budget 24.0 agent-hours. 14 submissions so far, 11 valid. |
| [`s7-inkling2s-b200x4-fa1`](s7-inkling2s-b200x4-fa1/progress.md) | Inkling-2-Small on GB200 | Score: `http_step_us_b1` (lower is better). Budget 12.0 agent-hours. 20 submissions so far, 19 valid. |
| [`s7-inkling2s-b200x4-fb1`](s7-inkling2s-b200x4-fb1/progress.md) | Inkling-2-Small on GB200 | Score: `http_step_us_b1` (lower is better). Budget 15.0 agent-hours. 36 submissions so far, 35 valid. |
| [`s8-inkling2s-b200x4-on1`](s8-inkling2s-b200x4-on1/progress.md) | Inkling-2-Small on GB200 | Score: `slo_share_r8`. Budget 32.0 agent-hours. 43 submissions so far, 42 valid. |
| [`s9-qwen38-27b-b200-f1`](s9-qwen38-27b-b200-f1/progress.md) | Qwen3.8-27B on GB200 | Score: `http_step_us_b1` (lower is better). Budget 8.0 agent-hours. 23 submissions so far, 23 valid. |
| [`s10-q8b-h100-f1`](s10-q8b-h100-f1/progress.md) | Qwen3-8B on H100 | Score: `http_step_us_b1` (lower is better). Budget 3.0 agent-hours. 4 submissions so far, 4 valid. |
