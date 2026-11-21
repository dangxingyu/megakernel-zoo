# Run records

One folder per run: `progress.md` lists every submission (what was tried, why, what it measured), `src/` has the source of every
submission and `notes/` the agent's own notes. Numbers in these files are the run's own evaluator's; the audited, re-measured
results are in the top-level README.

| run | model and GPU | |
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
| [`s11-q8b-h100-b1`](s11-q8b-h100-b1/progress.md) | Qwen3-8B on H100 | Score: `http_step_us_b1` (lower is better). Budget 20.0 agent-hours. 25 submissions so far, 24 valid. |
| [`s11-north-h100-b1`](s11-north-h100-b1/progress.md) | North-Mini-Code-1.0 on H100 | Score: `http_step_us_b1` (lower is better). Budget 20.0 agent-hours. 26 submissions so far, 26 valid. |
| [`s11-flint-b200-b1`](s11-flint-b200-b1/progress.md) | flint on GB200 | Score: `http_step_us_b1` (lower is better). Budget 20.0 agent-hours. 27 submissions so far, 27 valid. |
| [`s11-q27-b200-b1`](s11-q27-b200-b1/progress.md) | Qwen3.8-27B on GB200 | Score: `http_step_us_b1` (lower is better). Budget 30.0 agent-hours. 29 submissions so far, 29 valid. |
| [`s12-oss20-h100-a1`](s12-oss20-h100-a1/progress.md) | gpt-oss-20b on H100 | Score: `http_step_us_b1` (lower is better). Budget 32.0 agent-hours. 50 submissions so far, 49 valid. |
| [`s12-q32-b200-a1`](s12-q32-b200-a1/progress.md) | Qwen3-32B on GB200 | Score: `http_step_us_b1` (lower is better). Budget 36.0 agent-hours. 47 submissions so far, 47 valid. |
| [`s12-oss120-b200-a1`](s12-oss120-b200-a1/progress.md) | gpt-oss-120b on GB200 | Score: `http_step_us_b1` (lower is better). Budget 36.0 agent-hours. 59 submissions so far, 57 valid. |
| [`s15-mini-b200-a1`](s15-mini-b200-a1/progress.md) | Inkling-2-Mini on GB200 | Score: `http_step_us_b1` (lower is better). Budget 23.0 agent-hours. 32 submissions so far, 32 valid. |
| [`s15-mini-b200-a2`](s15-mini-b200-a2/progress.md) | Inkling-2-Mini on GB200 | Score: `http_step_us_b1` (lower is better). Budget 14.0 agent-hours. 26 submissions so far, 25 valid. |
| [`s15-mini-b200-b1`](s15-mini-b200-b1/progress.md) | Inkling-2-Mini on GB200 | Score: `http_step_us_b1` (lower is better). Budget 14.0 agent-hours. 10 submissions so far, 8 valid. |
| [`s15-mini-b200-b2`](s15-mini-b200-b2/progress.md) | Inkling-2-Mini on GB200 | Score: `http_step_us_b1` (lower is better). Budget 14.0 agent-hours. 52 submissions so far, 50 valid. |
| [`s16-mini-b200-c1`](s16-mini-b200-c1/progress.md) | Inkling-2-Mini on GB200 | Score: `http_step_us_b1` (lower is better). Budget 10.0 agent-hours. 22 submissions so far, 22 valid. |
| [`s17-mini-b200-ma`](s17-mini-b200-ma/progress.md) | Inkling-2-Mini on GB200 | Score: `http_step_us_b1` (lower is better). Budget 5.0 agent-hours. 14 submissions so far, 14 valid. |
| [`s17-mini-b200-mb`](s17-mini-b200-mb/progress.md) | Inkling-2-Mini on GB200 | Score: `http_step_us_b1` (lower is better). Budget 5.0 agent-hours. 21 submissions so far, 21 valid. |
