# fable-c-m5max-1: summary

Final source: **s0014** (the workspace's `engine/`): 8982 us/token synthetic (1.53x oMLX oQ4e + DSpark, 13767), 5333
us/token natural (1.56x, 8307), reproduced by s0016 (8975 / 5378); one-token step about 13970 us, engine greedy loop
about 13220 us (mlx-lm on the same formats: 30359). Every gate passes; the served tokens are the engine's own
step-by-step greedy tokens on every sequence. With DSpark drafts only (control s0013 on the s0012 engine, copy
proposals off): 9799 synthetic (1.41x), 7437 natural (1.12x). s0005, s0007, s0008, s0012 and s0014 / s0016 compute
bit-identical logits and decode the same continuations: those steps are clean A/Bs.

**How warm the machine is moves the result by about 10%.** The same s0014 source measured 8982 / 5333 and 8975 /
5378 when the evaluation started after 6-7 idle minutes (p99 gap between chunks 52.6 ms) and 9812 / 5856 when it
started right after another evaluation (s0021, gap 65 ms). The general engines' baselines were measured in their own
long runs; I do not know how warm.

**The board's best line (s0017, 8658) is not the final source.** s0015, s0017 and s0019 are one source (s0014 with a
draft threshold of 0.5 and no minimum row) and measured 10070 / 6297, 8658 / 5381 and 9070 / 5653. I first called
s0015 a disturbed run, then blamed the policy for being unstable, and s0021 shows that the slow runs follow the
machine's state instead: only s0017 started cool. So that policy has one cool measurement (-3.6% on synthetic text
against s0014, nothing on natural text), which is not enough to ship it; s0018 and s0020 ran warm and are not
evidence either way.

**Read the headline with its spread.** The three repeats of each workload are different prompts, and a 1024-token
continuation that falls into a loop decodes 2-3x faster (copy proposals, 16-row passes) than one that does not. Which
prompts loop is decided by last-bit numerics: over the numerically different builds of this run the same workload
gave 9.5 / 10.5 / 11.9 / 14.2 k us/token synthetic and 5.6 / 6.2 / 7.3 / 8.3 k natural. s0009 has faster passes than
s0008 (step 14719) and scores 11851 / 7267 because its continuations repeat less. Against oMLX + DSpark (13767 / 8307): s0003 1.31x / 1.00x, s0004 0.97x / 1.35x, s0005-s0008 1.41-1.45x / 1.44-1.47x,
s0009 1.16x / 1.14x, s0012 / s0014 (the s0008 draw, faster kernels) 1.52x / 1.54x and 1.53x / 1.56x. So the full engine (s0005 and later) has cleared the 1.10x line on both workloads in every
valid build, by a margin that ranges from 4% to 35% with the continuations. The control without copy proposals
(s0006, DSpark drafts only) is 1.235x / 0.981x.

## The engine (engine/)
- `mk_target.py`: Inkling-2-Mini's text model as one K-row pass (K consecutive positions of one sequence), K = 1 (the
  step), 2..8 (verify of DSpark drafts) or 16 (verify of copy proposals). Traced once per K with `mx.compile`.
- `mk_kernels.py`: the Metal kernels (through `mx.fast.metal_kernel`): BF16 GEMV (one simdgroup per output, K as a
  template, explicit fma), NVFP4 / MXFP8 expert kernels that read each distinct expert once per slot of 2 (row, expert)
  pairs, router top-8, attention with relative bias + sink + output gate over a two-tier K/V cache, depth mix with the
  norms folded in, message gates, k/v convs + clips, short conv + residual. Every per-row sum has an order that does
  not depend on K, so a verify row has the bits of the one-row step (tests/check.py: 0 rows differ for K = 2, 3, 5, 8
  and after rollbacks; tests/serve.py: 1024 served tokens equal the step-by-step tokens on 3 prompts).
- Weights stay in the stored formats (NVFP4 codes + E4M3 scales + amax / denominator, MXFP8 codes + UE8M0 scales,
  BF16). Activations enter the kernels as IEEE half (11 significant bits; the reference rounds them to BF16).
- `mk_draft.py`: the checkpoint's DSpark head (3 layers over anchor + 7 masks, context K/V from the target's features
  at visits 5, 23, 41 and the final norm, Markov head with the hidden_out x unembed fold), its matrices read from
  8-bit per-row-scaled copies (draft only). Plus copy proposals: when the last >= 5 committed tokens occurred earlier
  in the sequence, the tokens that followed are proposed (7, or 15 after a fully accepted copy cycle).
- `api.py`: step / generate (K = 1 passes) and submit / poll (draft, verify, accept; two host syncs per cycle).

## What moved the number (HTTP, us per token: synthetic / natural)
| sub | change | synthetic | natural | tokens/event |
|---|---|---|---|---|
| s0001 (quick) | own pass, no speculation | step 26852 | - | - |
| s0002 | DSpark draft + K-row verify, lazy MLX graphs | 15646 | 12704 | 2.90 / 4.21 |
| s0003 | mx.compile of the pass, shared expert reads, copy proposals | 10545 | 8335 | 3.04 / 5.08 |
| s0004 | depth-mix kernel, half activations, 16-row copy passes | 14178 | 6166 | 2.63 / 7.59 |
| s0005 | fused glue kernels, two-tier K/V cache, 8-bit draft copies | 9758 | 5731 | 2.90 / 7.21 |
| s0006 | control: s0005 with copy proposals off (DSpark drafts only) | 11145 | 8467 | 2.56 / 4.14 |
| s0007 | final source: s0005 + draft traced with mx.compile | 9745 | 5763 | 2.90 / 7.21 |
| s0008 | exact argmax through an 8-bit short list; draft Markov steps in two stages | 9496 | 5644 | 2.90 / 7.21 |
| s0009 | attention on 1024 virtual threads (faster small passes; other continuations) | 11851 | 7267 | 2.65 / 5.98 |
| s0010 | + qkv_post per simdgroup, 4-lane gate projections (step 13935) | incorrect: numeric p99 0.12095 > 0.12071 | | |
| s0011 (quick) | test: NVFP4 weights rounded to BF16 (hypothesis about the reference: rejected) | step 15470 | - | - |
| s0012 | bit-identical latency fixes in the small kernels (same logits as s0008) | 9029 | 5378 | 2.90 / 7.21 |
| s0013 | control: s0012 with copy proposals off (DSpark drafts only) | 9799 | 7437 | 2.56 / 4.14 |
| s0014 | down kernel with 4 weight rows per simdgroup in passes of 8+ rows (same logits) | 8982 | 5333 | 2.90 / 7.21 |
| s0015 | draft rows only while cumulative p >= 0.5, no minimum row (disturbed evaluation) | 10070 | 6297 | 2.77 / 6.85 |
| s0016 | the s0014 source again | 8975 | 5378 | 2.90 / 7.21 |
| s0017 | the s0015 source again | 8658 | 5381 | 2.77 / 6.85 |
| s0018 | control: s0017 with copy proposals off (DSpark drafts only) | 9910 | 7665 | 2.45 / 3.93 |
| s0019 | the s0015 / s0017 source a third time | 9070 | 5653 | 2.77 / 6.85 |
| s0020 | diagnosis: threshold 0.5 with at least one draft row | 9927 | 6203 | 2.82 / 6.99 |
| s0021 | the s0014 source a third time, started right after s0020 (warm machine) | 9812 | 5856 | 2.90 / 7.21 |

The synthetic headline is a median of three different prompts and depends on how many of them fall into a loop
(s0004's numerics changed the continuations: one of three looped instead of two); see notes/journal.md.

## Caveats
- Copy (n-gram) proposals are not the DSpark head. They are what the B200 track-C engine of this repository also
  uses. Controls with them off (every draft from the DSpark head): s0006 (s0005 engine) 11145 / 8467 = 1.235x / 0.981x
  oMLX+DSpark; s0013 (s0012 engine, the final draft policy) 9799 / 7437 = 1.41x / 1.12x; s0018 (the 0.5 / no-minimum
  policy, one run of a source that does not reproduce well) 9910 / 7665 = 1.39x / 1.08x. So with DSpark drafts alone
  the engine is about 1.4x on synthetic text and about 1.1x on natural text; the 1.5x figures need the copy proposals.
- generate() is a plain greedy loop of K = 1 passes (no speculation); speculation lives in submit / poll.
- The numeric gate's p99 is as sensitive to last-bit numerics as the continuations are: 0.064 to 0.121 over six
  builds against a limit of 0.1207 (s0008: 0.101; median 0.024 against 0.049). One build (s0010) landed above it.
  (s0012, s0014: 0.101, as s0008.) The error is not an offset: it grows with the position, because a flipped near-tie (router top-8) changes the
  history of every later position. Float32 activations cut it 3-4x and cost 10-23% of the speed (not kept).
- A build that changes any summation order is therefore two lottery draws (continuations, numeric p99); s0009 and
  s0010 were such builds. The speed they were after turned out not to need a new summation order: the small kernels
  were bound by loops with a load per iteration (~0.3 us each), and s0012 removes those with every sum's order kept
  (one-row pass 15.9 -> 14.3 ms, three rows 21.2 -> 19.6), so it is s0008's draw, faster.
