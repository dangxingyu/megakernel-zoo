---
name: megakernel-online
description: How to work in this run: turn the decode megakernel into an engine that serves a production request trace (requests arriving and leaving, long prompts, shared prefixes) and is measured through `mk`. Read before writing code, before every `mk submit`, and whenever a result surprises you.
---

# From a decode kernel to an engine with no engine around it

## Where this starts

`engine/` holds a megakernel for **Qwen/Qwen3-8B** (BF16) on **one NVIDIA H100 80GB** that an earlier run
built: one kernel launch per decode step, own tensor-core matvec, greedy loop on the device. On steady decode
(64-token prompt, 1024 new tokens) it measured about 1.12x the fastest general engine at batch 1 and 8.
`notes/journal.md` is that run's journal: what worked, what did not, and why. Read it before changing the kernel.

That result is steady decode: the batch never changes. Serving is not like that. The thesis of this project is
that a model-specific kernel should need **no inference engine around it**: admission, batching, prefill, KV
memory, prefix reuse and eviction decided where the data is, nothing on the GPU waiting for a host scheduler.
This run tests that on a real workload.

## The workload

A replay of a production trace from Moonshot's Kimi (Mooncake, FAST'25; tool and agent traffic). Each request
has an arrival time, an input length, an output length and the hashes of its 512-token prefix blocks; prompts
are synthesized so that requests sharing leading hashes share exactly those tokens. In the replayed window
(`$MK_TRACE` in the pod, the first 120 s at a nominal 2 requests/s; about 190 requests):

- inputs average about 9,000 tokens (median about 6,000, a few above 30,000, cut at 32,768);
- outputs average about 200 tokens, the median is about 30;
- about 30% of prompt tokens repeat an earlier request's prefix in this window (55% over the whole trace).

So the load is prefill-heavy, sequences enter and leave constantly, and a long prompt can arrive while others
are generating. On exactly these requests the general engines measured (server end to end, one run each):

| | within service levels | TTFT p50 / p90 | time between tokens p99 | longest stall |
|---|---|---|---|---|
| vLLM 0.32.0 | 64% | 1.8 s / 4.9 s | 0.32 s | 0.5 s |
| SGLang 0.5.23 | 67% | 1.6 s / 5.2 s | 2.6 s | 7.1 s |

Service levels: first token within 5 s **and** mean time between a request's tokens at most 0.1 s. Their GPU
was only 87% busy at this rate and never waited for the host; what hurts them is that a long prefill stalls the
streams that are decoding (steady decode gives 7 ms between tokens; here the p99 is 0.3 to 2.6 s), and that
prefill itself is slow to start. `mk board` shows these lines.

## The online contract

Add to `Engine` in `engine/api.py` (the offline `reset` / `step` / `generate` stay and are still evaluated):

```python
submit(rid: int, prompt: list[int], max_new_tokens: int) -> None
    # a request arrives; return at once. Greedy; exactly max_new_tokens tokens, no early stop.
poll() -> list[tuple[int, list[int], bool]]
    # tokens produced since the last poll: (rid, tokens, finished). May do work, must return within a few steps.
idle() -> bool                    # nothing queued, nothing running
kv_pages() -> tuple[int, int]     # (free, total) pages of the KV store right now
drop_cache() -> None              # forget retained prefixes; an idle engine then has free == total
stats() -> dict                   # your counters: cache hits, evictions, recomputes, queue lengths, syncs ...
```

The driver calls `submit` at the arrival times and `poll` in a tight loop; a token's time is when `poll` first
returns it. Everything else is yours: when to admit, how to batch, how to prefill (one token per step is a
47-second first token for a 9,000-token prompt), where KV lives, what to share, what to evict and recompute.
`Engine(model_dir, max_batch, max_context)` is called with `max_batch=64`, `max_context=34816`.

## What the evaluator checks (`mk submit`)

1. The **offline** evaluation, short form: float64 check of every position under teacher forcing, batch gate,
   long-context gate to position 1151, steady-decode timing. The engine must still compute the model.
2. The **replay** above through `submit` / `poll`, then the invariants:
   - `liveness`: every request finishes with exactly the tokens it asked for; the engine is idle afterwards;
   - `schedule`: 12 probe requests give the same tokens in the replay as alone on an idle engine with an empty
     cache. Batching, prefix reuse, chunked prefill and eviction must not change a result;
   - `online_vs_step`: the offline reference prompts through `submit` / `poll`, alone and together, give the
     same tokens as step-by-step greedy;
   - `kv_conservation`: idle and after `drop_cache`, free pages == total pages, every time it is looked at.
   The token gates tolerate a rare divergence (at least 75% of sequences identical, mean matching prefix 0.9);
   the count of identical sequences is recorded. Aim for all: keep every row's arithmetic order independent of
   what else is in the batch and of whether its prefix came from a cache.
3. **Headline**: the share of the replay's requests within the service levels (`slo_share_r2`, higher is
   better), reported with TTFT, time between tokens (p50 / p99) and the longest stall, next to the engines.

`mk submit --quick` runs 40 s of the replay with 6 probes (logged, never ranked). The evaluator itself is in your
workspace as `tools/online_eval.py` (with `tools/_online_runner.py`, `tools/prompts.py`): in the pod,
`mk exec -- $MK_PY tools/online_eval.py --workspace . --out $MK_EXEC_OUT/r.json --quick` runs the same checks
without submitting, and `--rates "1 2 4"` sweeps the load. Use it to measure a hypothesis before you spend a
submission on it, and say in the hypothesis when you did.

## Ladder

| Rung | Property |
|---|---|
| `online` | the contract works and every gate passes, however slowly (for example requests one at a time) |
| `long_context` | correct to 34,816 positions; prefill of many tokens per pass instead of one token per step |
| `mixed` | several sequences at different positions decode together; a request joins and leaves without a reset |
| `prefill_with_decode` | a prefill does not stall the decoding streams: the decode rows ride the same pass over the weights |
| `prefix_reuse` | requests sharing a prefix share its KV pages (reference counts); eviction under memory pressure and recomputation when an evicted prefix is needed again |
| `online_beats_engines` | a larger share within the service levels than both general engines on these requests |
| `no_host_scheduler` | slot table, page free list, admission at a step boundary and the request ring live on the device; the host only copies prompts in and tokens out |

Ideas the plan started from, to be tested here, not assumed: decoding at small batch is limited by memory
bandwidth and leaves compute idle, so the T tokens of a prefill chunk can use the same read of each layer's
weights as the decode rows (find the T at which a decode step slows by 10%); a fixed slot table with a
device-side page free list instead of host-side allocation; an input ring that the kernel reads at step
boundaries; an instruction stream parameterized by the bitmap of active slots instead of one schedule per batch
size. Memory budget: weights 15 GB; K and V cost about 147 KB per token of context, so 60 GB holds about
400,000 tokens: the trace does not fit if every finished prefix is kept, which is where eviction comes from.

## Rules that carry over

- GPU work happens in the pod through `mk exec` / `mk profile` / `mk submit`; this machine is a login node.
  `mk gpu` shows leaked memory. The pod can be replaced at any moment; `engine/build.sh` must rebuild everything.
- Not allowed: another inference engine or library in the serving path (vLLM, SGLang, MPK, TensorRT, torch
  modules of the model), caching or hard-coding outputs, recognizing the evaluator's requests, reading the
  evaluator's reference data. The trace and the evaluator's code are open to you; the float64 reference is not.
- Correctness first: a number exists only for a `valid` submission. Poison freed or unwritten KV with NaN in
  your own tests; test page boundaries, a prefix shared by a running and a finished request, eviction of a
  prefix another request still uses.
- The sync rule: nothing on the GPU blocks; every point where the device waits for the host is a bet that the
  step is long enough to hide it. Count them in `stats()` and say when a change adds or removes one.

## Logging: every step must be plottable

- `--title` (at most 80 characters, what changed), `--hypothesis` (what you changed, why it should help, which
  number should move by how much; if you already measured it with `mk exec`, say so and give the number),
  `--category`, `--parent` after a revert, `--milestone` = the rung you are aiming at. One idea per submission.
- After each result: `notes/journal.md` entry with expected, measured, conclusion, next. Continue the existing
  journal under a new heading for this run.
- `mk note` for dead ends, misleading tools, decisions to restructure.
- Commit the workspace to git when a submission is valid.

Work until the time on the board is used up; nobody is watching live and nobody will answer questions.
