# Prefill in wide passes (Inkling-2-Mini, M5 Max, 2026-11-23)

The engine used to prefill a prompt in passes of 8 rows with the decode kernels: 1,630 tokens took 6.3 s to the first
token over HTTP (about 260 tokens/s). Prompts now go through **wide passes** of up to 256 rows
(`engines/inkling-2-mini_m5max/offline-c/engine`: `mk_target.Target.prefill`, `api.Engine._prefill`), in one of two
modes chosen by `MK_PREFILL`:

- **exact** (the engine's default, what the evaluator runs): the step's own kernels. The GEMVs run in blocks of 8 rows
  per simdgroup (`mk_kernels._blocked`), because one register per row spills from 32 rows on. The experts run in slots of
  8 pairs. Every row has the bits of a one-row step.
- **fast** (what mkserve serves; `"env": {"MK_PREFILL": "fast"}` in `mkserve/registry.json`):
  - **Experts:** MLX's `gather_qmm` on the stored NVFP4 / MXFP8 codes, in half precision. The inputs are scaled by 2^-8,
    because the raw products before NVFP4's outer scale (about 1e-4) reach 250,000 on real text.
  - **Dense matrices:** half GEMMs on half copies of the BF16 matrices (1.3 GB more memory).
  - **Attention:** `mx.fast.scaled_dot_product_attention`, with the relative bias and the window as an additive mask,
    and the per-row sink as one extra zero key.
  - Not the step's bits.
- **off:** the old 8-row passes.

In both wide modes a pass is padded to 32, 64, 128 or 256 rows. Padding rows come after the real ones, so they never
affect them. All four shapes are compiled at load.

## First token (engine submit/poll, real text; `scripts/t_api.py`)

| prompt tokens | 8-row passes (before) | exact wide | fast wide |
|---|---|---|---|
| 64 | 212 ms | 154 ms | 90 ms |
| 300 | 999 ms | 738 ms | 259 ms |
| 1,630 | 6,834 ms | 4,204 ms | 1,245 ms |
| 3,000 | — | — | 2,313 ms |

Over HTTP (mkserve, `../mkserve-check-20261123/check5.log`), with all 12 checks passing:

- The 1,630-token prompt's first token takes 1.20 s, against 6.3 s before.
- The 64-token prompt's first token takes 78 ms, against 206 ms.
- Decode runs at 325 tok/s, against 317 before.

## One 256-row pass at position 768 (`pass256_breakdown.txt`, `scripts/t_pass.py`)

Time per prompt token, split by replacing a part with a stand-in (`MK_WHATIF_NO`):

| part | exact | fast |
|---|---|---|
| experts | 1.84 ms | 0.38 ms |
| dense matrices | 0.40 ms | 0.06 ms |
| attention and the rest | 0.37 ms | 0.24 ms |
| **total** | **2.61 ms** | **0.68 ms** |

In the exact mode the experts are bound by fp32 FMA throughput: each pair's sums must keep the step's order, so no
matrix units. Compare the step's 8-row pass at 3.6 ms per token and the 16-row pass at 2.95. A 32-row pass of the step's
kernels took 274 ms (register spills), which is why the GEMVs run in blocks.

## Exactness and accuracy

- **Exact mode.** It is bit-identical to 8-row passes: the K/V of all 48 applications and the conv windows are equal,
  and so is the next token on 1,630 tokens (`scripts/t_exact.py`). `eval/spec_check.py` gives 4 of 4 prompts identical
  between online and step greedy over 512 tokens (`sc_step.json`, `sc_online.json`). `t_api.py` with `CHECK=1` also
  gives online = step greedy on 64, 300 and 1,630-token prompts.
- **Fast mode.** It is not step-exact. `spec_check` finds 1 of 4 identical, with first differences at tokens 1, 4 and
  287 (`sc_fast.json`). Against the float32 reference it is closer than the step, not further. The table below is on
  the reference's sequence, at its long positions 128–1144 (`scripts/t_dev.py`):

| configuration | positions | rel-L2 median | p99 | KL median | top-1 |
|---|---|---|---|---|---|
| step (exact passes) | all 128 | 0.0350 | 0.313 | 5.5e-5 | 0.992 |
| fast wide passes, their own rows | all 128 | 0.0174 | 0.332 | 1.0e-5 | 1.000 |
| step | ≥ 256 (112) | 0.0370 | 0.313 | 5.5e-5 | 0.991 |
| fast prefill of 256, then steps | ≥ 256 (112) | 0.0148 | 0.248 | 9.4e-6 | 1.000 |
| step | ≥ 512 (80) | 0.0428 | 0.380 | 4.8e-5 | 1.000 |
| fast prefill of 512, then steps | ≥ 512 (80) | 0.0292 | 0.375 | 1.3e-5 | 1.000 |
| step | ≥ 1024 (16) | 0.0679 | 0.153 | 7.6e-5 | 1.000 |
| fast prefill of 1024, then steps | ≥ 1024 (16) | 0.0476 | 0.200 | 1.3e-4 | 1.000 |

This is one sequence, so read it only as "no worse".

## Not done

- **Prefix cache.** Each request still prefills its whole prompt.
- **Attention.** In the exact mode, attention still uses the decode kernel, which reads every key once per row.
- **Fused kernels in the fast mode.** The fast mode spends 0.24 ms per token in the per-row kernels (depth mixes,
  gates, convs) and attention. Fusing them is the next step.

---

# Later the same day: prefix cache, long context, sampling

The same engine source. The sections above describe the prefill as first built; the "Not done" list there is out of
date (the prefix cache exists now).

## Prefix cache

The K/V arrays are by position, so a request that starts with the tokens of the cached sequence keeps them. The only
state that is not by position is the conv windows (the last 3 inputs of 4 convolutions in 48 applications, 3.8 MB), so
the target keeps a snapshot of them every 64 positions (256 for a context beyond 8,192), during prefill and decode.
A new request is cut back to the last snapshot inside the shared prefix (`Target.rollback`, `Engine._begin`) and
prefills only the rest. `reset()` is lazy: the old state is dropped when the next sequence shows its first tokens.

- **Exact mode** (`scripts/t_prefix.py`): 6 cases (next chat turn, same prompt, a prefix of it, a prompt that diverges
  after 1,000 tokens, nothing shared, a 3,000-token prompt plus reply): tokens identical to a full reset in all six.
- **Through mkserve** (`chat_turns_fast.txt`, fast mode, a 1,452-token system prompt): first token after 1.60 s on
  turn 1, then 0.20, 0.20 and 0.18 s with 1,408 to 1,536 prompt tokens kept.
- mkserve reports the kept tokens: OpenAI `usage.prompt_tokens_details.cached_tokens`, Responses
  `input_tokens_details.cached_tokens`, Anthropic `cache_read_input_tokens`, `/metrics`
  `mkserve_cached_prompt_tokens_total`.
- The engine has `cancel(rid)` now: a client that disconnects no longer holds the one sequence until `max_tokens`.

## Long context (4,096 to 32,768)

- **Attention scores.** The decode attention kernel kept one score per key in threadgroup memory (32 KB: at most 7,168
  keys). The 40 local applications (window 1,024) need 1,056 scores at any context. The 8 global ones keep them in a
  device buffer beyond 7,168 (`mk_attention_gate_big`: the same statements, so the same bits).
- **Log scaling.** Global layers scale q by `1 + 0.1 ln(max((position + 1) / 8192, 1))`. The engine never reached
  position 8,192 before; it applies the factor now when the context allows such positions.
- **Draft head.** Its passes rewrite its whole K/V, so its cost would grow with the context. Beyond 4,104 positions it
  is a ring over the last 4,104 (`mk_draft.Draft.ring`). This changes what is proposed, never what is returned.
- **K/V storage.** The big tier is one array per application now (it was one array for all 48): a flush copies 2 KB
  per position at a time instead of 96 KB. Peak memory at 32,768 went from 34 GB to 29 GB.

What it costs with a short prompt (`ctx_cost.txt`, evaluator prompts, 512 greedy tokens):

| max context | decode, ms per token | memory in use |
|---|---|---|
| 4,096 | 12.3 | 22.1 GB |
| 8,192 | 12.2 | 23.0 GB |
| 16,384 | 11.9 | 24.6 GB |
| 32,768 | 12.9 | 27.7 GB |

MLX keeps about 6 GB more in its buffer cache at every size. Registered: 16,384 by default, 32,768 with
`--max-context`.

Pass times by position at 32,768 (`scripts/t_ctx.py`): a one-row step takes 13.6 ms at position 0, 15.8 at 8,192 and
19.0 at 30,000; a 256-row fast prefill pass takes 0.76 ms per token at 4,096 and 1.10 at 30,000.

Checks:

- **Default context unchanged.** `eval/spec_check.py`: 4 of 4 identical, and the step tokens equal those of the source
  before any of these changes. An A/B of 512-token decodes against the original source: 20.9 and 22.6 s against 21.9
  and 21.5 s for the same four prompts (noise).
- **Online equals step at 32,768.** Prompts of 64, 3,000 and 9,000 tokens (`t_api.py`, `CHECK=1`).
- **Against float32** (`scripts/long_reference*.py`, `scripts/t_long_dev.py`; one text of repository files):

| sequence | positions | configuration | rel-L2 median | p99 | KL median | top-1 |
|---|---|---|---|---|---|---|
| 9,600 tokens | below 8,191 (7) | exact | 0.028 | 0.156 | 7.9e-4 | 1.000 |
| | 8,191 and above (24) | exact | 0.047 | 0.231 | 5.0e-4 | 1.000 |
| | 8,191 and above (24) | exact, no log scaling | 0.050 | 0.213 | 7.6e-4 | 0.958 |
| | below 8,191 (7) | fast wide rows | 0.031 | 0.219 | 3.3e-3 | 1.000 |
| | 8,191 and above (24) | fast wide rows | 0.051 | 0.238 | 5.2e-4 | 0.958 |
| 24,000 tokens | below 8,191 (3) | exact | 0.015 | 0.156 | 2.8e-4 | 1.000 |
| | 8,191 and above (24) | exact | 0.105 | 0.584 | 1.0e-5 | 1.000 |
| | 8,191 and above (24) | exact, no log scaling | 0.126 | 0.579 | 1.7e-4 | 1.000 |
| | 8,191 and above (24) | the reference's BF16 path | 0.089 | 0.483 | 3.7e-5 | 1.000 |

At 24,000 tokens the factor is 1.11 and leaving it out raises the median KL 17 times; the engine is within the
evaluator's 2x rule of the BF16 path there. This is one sequence per length.

- **An agent's request** (`agent_32k.txt`, Anthropic API, 19,683 tokens of system prompt and four tools, 32,768
  context): first token after 21.7 s on turn 1; 1.07 s and 0.82 s on turns 2 and 3 with about 19,500 tokens read from
  the cache; turn 3 is a correct tool call.

## Sampling with speculation

Native mode was greedy only. A sampled token is now a function of its row's logits and of (seed, position) alone:
`argmax(logits / T + Gumbel noise)`, the noise a hash of (seed, position, token id) (`mk_kernels.gumbel_add`,
`mk_target.sample_rows`); `top_p` / `top_k` restrict the argmax to the tokens at or above a threshold logit first. A
row of a verify pass then draws the token a one-row step draws, so draft-and-verify stays exact, and the draft head
draws with the same noise.

- **Exactness** (`scripts/t_sample.py`, `sample_exact_256.txt`): 4 prompts x 4 settings (T 0.7; T 1.0 with top_p 0.95;
  T 0.8 with top_k 40; T 0.3), 256 tokens each: `submit()` / `poll()` returns exactly the tokens a step-by-step loop
  draws with the same seed, in all 16; the same seed repeats, another seed differs.
- **The sampler** draws from softmax(logits / T): 4,000 draws match the exact probabilities, with and without top_p
  and top_k (the prototype's check).
- **Speed.** A sampling cycle costs what a greedy one does (about 20 ms); throughput follows the accepted drafts.
  In process, ms per token: notes 5.3 greedy, 6.2 at T 0.3, 8.4 at T 0.7; Python source 9.3, 9.2, 12.6; a story 11.7,
  13.8, 12.8. Over the chat API (`http_sampling.txt`): a story 76 tokens/s greedy, 80 at T 0.7, 68 at T 1.0 with
  top_p 0.95; a technical answer 102, 113, 86. Step-mode sampling, the only sampling before, is about 68.
- mkserve passes `temperature`, `top_p`, `top_k` and `seed` to an engine with `native_sampling`; the serving check's
  sampling test passes in native mode (12 of 12, `../mkserve-check-20261123/check7.log`).
